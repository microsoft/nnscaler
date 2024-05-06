# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict, Tuple
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from cube.graph.parser.fx.parser import FxModuleParser
from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.reducer import Reducer
from cube.runtime.gnorm import ParamsInfo

_logger = logging.getLogger(__name__)


class CubeModule(torch.nn.Module):
    """
    The module is responsible for parameter synchronization
    before training
    """

    def __init__(self):
        super().__init__()
        self._reducers: List[Reducer] = list()
        # Key: str, parameter name (from named_parameters)
        # Value: Tuple[int, Tuple[slice], int]: 
        # full tensor tid, 
        # position of sub tensor in full tensor, 
        # position of value in value partition.
        self._fullmap : Dict[str, Tuple[int, Tuple[slice], int]] = dict()

    @property
    def reducers(self):
        return self._reducers
    
    @property
    def fullmap(self):
        return self._fullmap

    def tid_of_param_name(self, name: str) -> int:
        # Return the tid of sub tensor with the parameter name
        # It is the last field of the parameter name, which is hacky
        if name not in self._fullmap:
            raise RuntimeError(f"Cannot find {name} in fullmap")
        return int(name.split('_')[-1])

    def add_reducer(self, reducer: Reducer):
        if not isinstance(reducer, Reducer):
            raise RuntimeError(f"Expected a Reducer but got {type(reducer)}")
        self._reducers.append(reducer)

    def zero_grad(self):
        """Make zero for gradients in weight reducer

        This only applies on the gradients of the parameters in each reducer.
        This function will be automatically inserted inside the generated code
        at the beginning of each iteration.

        If the function is under the context of `with cube.accum_mode()`, the zero of gradients
        will be skipped.
        """
        for reducer in self._reducers:
            reducer.zero_grad()

    def parameters_for_optimizer(self) -> List[torch.nn.Parameter]:
        """Get parameter list for optimizer"""
        params = []
        reducer_pids = set()
        for reducer in self._reducers:
            params += reducer.parameters_for_optimizer()
            reducer_pids.update(id(p) for p in reducer.params)
        for param in self.parameters():
            if id(param) not in reducer_pids:
                params.append(param)
        # print(f'> get out parameters: {sum(p.numel() for p in params)}')
        return params

    def parameters_for_broadcast(self) -> List[torch.nn.Parameter]:
        """
        This function is for broadcasting loaded weights from one scale unit to
        all other scale units to resume from sharded checkpoints globally.
        """
        params = []
        reducer_pids = set()
        for reducer in self._reducers:
            params.append(reducer._contiguous_params)
            reducer_pids.update(id(p) for p in reducer.params)
        for param in self.parameters():
            if id(param) not in reducer_pids:
                params.append(param)
        return params

    def parameters_for_calc_gnorm(self) -> List[ParamsInfo]:
        """Return the necessary information for calculating the gradient norm.
        
        Returns:
            List[Tuple[Tuple[int], List[torch.nn.Parameter], List[str], int]]: 
                A list of tuples, each tuple contains the following information:
                    Tuple[int]: the ranks spanned by the parameters in the tuple
                    List[torch.nn.Parameter]: the contiguous parameters in the tuple
                    List[str]: the names of the original parameters in the tuple
                    int: the number of the ZeRO groups for the parameters
        """
        paramid2name = {}
        for name, param in self.named_parameters():
            paramid2name[id(param)] = name

        params_info_for_gnorm = []
        reducer_pids = set()
        for reducer in self._reducers:
            param_names = [paramid2name[id(p)] for p in reducer.params]
            params_info = ParamsInfo(reducer.ranks, reducer.parameters_for_optimizer(),
                                     param_names, reducer.zero_ngroups)
            params_info_for_gnorm.append(params_info)
            reducer_pids.update(id(p) for p in reducer.params)
        for param in self.parameters():
            if id(param) not in reducer_pids:
                # zero_ngroups is 1, since there is no reducer for it and multiplying 1 does not change the result.
                params_info = ParamsInfo((dist.get_rank(),), [param], [paramid2name[id(param)]], 1)
                params_info_for_gnorm.append(params_info)
        return params_info_for_gnorm

    def gather_params(self):
        """
        Gather parameters

        This won't take effect when zero is not enabled.
        """
        for reducer in self._reducers:
            if reducer.zero:
                reducer.gather_params()

    def add_full_map(self, attr: str, tid: int, slicers: Tuple[slice], val_chunks: int):
        """
        Add an attribute map.
        The mapping includes current attribute name (str) to logical tensor id,
        and the mapping of logical tensor id including spatial (slice) and val chunks

        @param attr str: attribute name of this moudle
        @param tid int: full tensor id
        @param slicers Tuple[slice]: indexing from full tensor
        @param val_chunks int: the number of value chunks.
        """
        assert hasattr(self, attr), f"{attr} is not in the module"
        self._fullmap[attr] = (tid, slicers, val_chunks)

    # TODO: remove this function, use the property instead
    def get_full_map(self):
        return self._fullmap

    def load_attr_content(self, filename: str):
        with torch.no_grad():
            full = torch.load(filename)
            for attr in self._fullmap.keys():
                tensor: torch.Tensor = getattr(self, attr)
                tid, slicers, nchunks = self._fullmap[attr]
                content = full[tid][slicers] / nchunks
                tensor.copy_(content)
                # print(f'attr {attr}:\n{getattr(self, attr)}')

    def init_group(self, ranks: List[int]):
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected ranks to be List[int]")
        DeviceGroup().get_group(ranks)

    def get_checkpoint(self, optimizer: torch.optim.Optimizer = None):
        state_dict = super().state_dict()
        # backward compatibility
        # in old version, dist_param_map is not loaded in constructor
        # so we will try to load it from file on the fly.
        dist_param_map = getattr(self, '_dist_param_map', None)
        if not dist_param_map:
            assert os.path.isfile('dist_param_map.pt'), 'Cannot open distributed parameter mapping file: dist_param_map.pt'
            dist_param_map = torch.load('dist_param_map.pt')
        param_area_map = self._fullmap
        optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        return state_dict, dist_param_map, param_area_map, optimizer_state_dict

    def save_checkpoint(self, optimizer: torch.optim.Optimizer = None, filename_prefix: str = None):
        filename_prefix = 'dist_checkpoint' if filename_prefix is None else filename_prefix
        filename = f"{filename_prefix}-{DeviceGroup().rank}.ckpt"
        state_dict, dist_param_map, param_area_map, optimizer_state_dict = self.get_checkpoint(optimizer)

        _logger.info(f'saving distributed checkpoint to {filename}')
        torch.save({
            'state_dict': state_dict,
            'dist_param_map': dist_param_map,
            'param_area_map': param_area_map,
            'optim_state_dict': optimizer_state_dict,
        }, filename)

    @staticmethod
    def merge_partial_states(state_dicts, zero_idx_maps=None):
        """
        :param state_dicts: list of state_dict from different ranks
        state_dict(model_state_dict, optimizer_state_dict, dist_param_map, param_area_map)
        :return: merged state_dict(model_state_dict, optimizer_state_dict,)
        """
        assert len(state_dicts) > 0

        plan_ngpus = -1
        # TODO: remove this flag
        if 'PLAN_NGPUS' in os.environ:
            plan_ngpus = int(os.environ['PLAN_NGPUS'])
            assert plan_ngpus >= 1, plan_ngpus
            assert plan_ngpus <= len(state_dicts), f'plan_ngpus = {plan_ngpus}, len(state_dicts) = {len(state_dicts)}'
            assert len(state_dicts) % plan_ngpus == 0, f'plan_ngpus = {plan_ngpus}, len(state_dicts) = {len(state_dicts)}'
            _logger.info(f'plan_ngpus = {plan_ngpus}')

        # at first, merge the partitioned optimizer states due to zero to the zero-disabled format
        if zero_idx_maps is not None:
            if bool(int(os.environ.get('USE_ZERO', default=0))):
                def _check_state_size(opt_state_keys, bucket_state):
                    if len(opt_state_keys) <= 1:
                        return True
                    return all(bucket_state[key].shape == bucket_state[opt_state_keys[0]].shape
                               for key in opt_state_keys)

                def _retrieve_param_opt_state(bucket_states, pstart, pend, pshape, bucket_size):
                    assert bucket_size % len(bucket_states) == 0
                    opt_state_keys = list(bucket_states[0].keys())
                    if 'step' in bucket_states[0]:
                        opt_state_keys.remove('step')
                    assert _check_state_size(opt_state_keys, bucket_states[0]), f'the keys {opt_state_keys} have different shape'
                    # NOTE: only support adam for now
                    assert 'exp_avg' in opt_state_keys
                    assert 'exp_avg_sq' in opt_state_keys
                    chunk_size = bucket_size // len(bucket_states)
                    start_rank_id, start_offset = pstart // chunk_size, pstart % chunk_size
                    end_rank_id, end_offset = pend // chunk_size, pend % chunk_size
                    opt_states, opt_states_1d = {}, {}
                    for key in opt_state_keys:
                        opt_states[key] = torch.zeros(pshape, dtype=bucket_states[0][key].dtype,
                                                      device=bucket_states[0][key].device, requires_grad=False)
                        opt_states_1d[key] = opt_states[key].view(-1)

                    if start_rank_id == end_rank_id:
                        for key in opt_state_keys:
                            opt_states_1d[key][:] = bucket_states[start_rank_id][key][start_offset:end_offset]
                    else:
                        offset = chunk_size-start_offset
                        for key in opt_state_keys:
                            opt_states_1d[key][:offset] = bucket_states[start_rank_id][key][start_offset:]
                        for i in range(start_rank_id+1, end_rank_id):
                            for key in opt_state_keys:
                                opt_states_1d[key][offset:offset+chunk_size] = bucket_states[i][key][:]
                            offset += chunk_size
                        for key in opt_state_keys:
                            opt_states_1d[key][offset:] = bucket_states[end_rank_id][key][:end_offset]

                    if 'step' in bucket_states[0]:
                        opt_states['step'] = bucket_states[0]['step']
                    return opt_states

                opt_state_list = []
                worker_cnt = len(state_dicts)
                for work_idx in (range(worker_cnt) if plan_ngpus < 0 else range(plan_ngpus)):
                    model_idx2opt_idx, opt_idx2ranks = zero_idx_maps[work_idx]
                    opt_state = {}
                    for model_idx, opt_idx in model_idx2opt_idx.items():
                        if isinstance(opt_idx, int):
                            # the param without reducer
                            assert opt_idx2ranks[opt_idx] is None
                            # state_dicts [worker idx][opt state]['state'][param idx]
                            opt_state[model_idx] = state_dicts[work_idx][1]['state'][opt_idx]
                        else:
                            # the param in reducer bucket
                            opt_idx, pstart, pend, pshape = opt_idx
                            ranks, bucket_size = opt_idx2ranks[opt_idx]
                            bucket_states = [state_dicts[rank][1]['state'][opt_idx] for rank in ranks]
                            opt_state[model_idx] = _retrieve_param_opt_state(
                                bucket_states,
                                pstart,
                                pend,
                                pshape,
                                bucket_size)
                    opt_state_list.append(opt_state)
                    assert len(state_dicts[work_idx][1]['param_groups']) == 1, 'only support param_groups to be one group'
            else:
                if plan_ngpus > 0:
                    _logger.warning(f'plan_ngpus {plan_ngpus} not handled USE_ZERO == False')
                def _check_opt_state(opt_state):
                    cnt = 0
                    sorted_opt_state = {}
                    for idx in sorted(opt_state.keys()):
                        assert cnt == idx, f'opt state error: {idx} vs {cnt}, in {opt_state.keys()}'
                        sorted_opt_state[idx] = opt_state[idx]
                        cnt += 1
                    return sorted_opt_state
                optimizer_state_dict = {}
                worker_cnt = len(state_dicts)
                opt_state_list = []
                for work_idx in range(worker_cnt):
                    zero_idx2model_idx, model_idx2zero_idx, zero_rank_groups = zero_idx_maps[work_idx]
                    opt_state = {}
                    # first place local opt state to right index
                    if len(zero_idx2model_idx) == 0:
                        assert len(state_dicts[work_idx][1]['state']) == 0
                    for local_idx, val in state_dicts[work_idx][1]['state'].items(): # worker / last_optimizer_state / state
                        global_idx = zero_idx2model_idx[local_idx]
                        assert global_idx not in opt_state
                        opt_state[global_idx] = val
                    # for each rank group, copy opt state from other buckets
                    for rank_group, param_idx_buckets in zero_rank_groups.items():
                        for bucket_idx, rank in enumerate(rank_group):
                            if rank == work_idx: continue
                            for global_idx in param_idx_buckets[bucket_idx]:
                                other_local_idx = zero_idx_maps[rank][1][global_idx] # rank / model_idx2zero_idx / global_idx
                                assert global_idx not in opt_state
                                opt_state[global_idx] = state_dicts[rank][1]['state'][other_local_idx] # worker / last_optimizer_state / state / local idx
                    opt_state = _check_opt_state(opt_state)
                    opt_state_list.append(opt_state)
                    assert len(state_dicts[work_idx][1]['param_groups']) == 1, 'only support param_groups to be one group'
            # assign opt_state to state_dicts, cannot be assigned in the above loop
            opt_state_len = len(opt_state_list[0])
            for work_idx in (range(worker_cnt) if plan_ngpus < 0 else range(plan_ngpus)):
                state_dicts[work_idx][1]['state'] = opt_state_list[work_idx]
                state_dicts[work_idx][1]['param_groups'][0]['params'] = sorted(opt_state_list[work_idx].keys())
                assert len(opt_state_list[work_idx]) == opt_state_len

        # find tensor full shape
        param_max_dimsize = {}
        if plan_ngpus > 0:
            state_dicts = state_dicts[0:plan_ngpus]
        for model_state_dict, optimizer_state_dict, dist_param_map, param_area_map in state_dicts:
            for param_area in param_area_map.items():
                local_name = param_area[0][0:param_area[0].rfind('_')]
                assert len(local_name) > 0
                raw_name = dist_param_map[local_name]
                slices = param_area[1][1]
                if param_area[1][2] != 1:
                    _logger.error(f'value-split on {raw_name} is not supported')
                if raw_name in param_max_dimsize:
                    param_max_dimsize[raw_name] = max(param_max_dimsize[raw_name], slices)
                else:
                    param_max_dimsize[raw_name] = slices

        # create full tensors
        param_full_tensors = {}
        sample_step = -1
        optim_full_tensors: Dict[int, Dict[any, any]] = {}  # param_id, (state_name, state_val)
        for model_state_dict, optimizer_state_dict, dist_param_map, param_area_map in state_dicts:
            if len(optimizer_state_dict['state'].items()) > 0:
                optimizer_state_names = list(optimizer_state_dict['state'][0].keys())
                _logger.info(f'optimizer_state_names = {optimizer_state_names}')
                if 'step' in optimizer_state_names:
                    sample_step = optimizer_state_dict['state'][0]['step']
                    optimizer_state_names.remove('step')
                _logger.info(f'optimizer_state_names (without step) = {optimizer_state_names}')
            else:
                optimizer_state_names = []

            other_optim_keys = [key for key in optimizer_state_dict.keys() if key != 'state']
            optimizer_other_state_dict = {}
            for key in other_optim_keys:
                optimizer_other_state_dict[key] = optimizer_state_dict[key]

            # for raw_name in param_max_dimsize.keys():
            model_state_dict_keys = list(model_state_dict.keys())
            for param_area in param_area_map.items():
                local_name_with_id = param_area[0]
                local_name = local_name_with_id[0:local_name_with_id.rfind('_')]
                raw_name = dist_param_map[local_name]

                tensor_size_slice = param_max_dimsize[raw_name]
                tensor_size = []
                for dim_slice in tensor_size_slice:
                    tensor_size.append(dim_slice.stop)
                partial_tensor = model_state_dict[local_name_with_id]
                param_full_tensors[raw_name] = torch.zeros(tuple(tensor_size), dtype=partial_tensor.dtype)

                index = model_state_dict_keys.index(local_name_with_id)
                if index in optimizer_state_dict['state']:
                    for state_name in optimizer_state_names:  # 'step'
                        if index not in optim_full_tensors:
                            optim_full_tensors[index] = {}
                        optim_full_tensors[index][state_name] = torch.zeros(tuple(tensor_size))
                else:
                    _logger.info(f'merge_checkpoint skips {local_name_with_id}\'s optimizer state')
            break  # only create once

        # assign value
        for model_state_dict, optimizer_state_dict, dist_param_map, param_area_map in state_dicts:
            model_state_dict_keys = list(model_state_dict.keys())
            for param_area in param_area_map.items():
                local_name_with_id = param_area[0]
                local_name = local_name_with_id[0:local_name_with_id.rfind('_')]
                raw_name = dist_param_map[local_name]
                slices = param_area[1][1]
                partial_tensor = model_state_dict[local_name_with_id]
                param_full_tensors[raw_name][slices] = partial_tensor

                index = model_state_dict_keys.index(local_name_with_id)
                if index in optimizer_state_dict['state']:
                    states = optimizer_state_dict['state'][index]
                    for name in optimizer_state_names:
                        val = states[name]
                        optim_full_tensors[index][name][slices] = val
                        if sample_step > 0:
                            optim_full_tensors[index]['step'] = sample_step

        # print(f'param_full_tensors (assigned) = {param_full_tensors}')
        # print(f'optim_full_tensors (assigned) = {optim_full_tensors}')

        optimizer_other_state_dict.update({'state': optim_full_tensors})
        # dump to ckpt
        return param_full_tensors, optimizer_other_state_dict

    @staticmethod
    def merge_checkpoints(filename_prefix='dist_checkpoint'):
        ckpts = {}
        for rank in range(DeviceGroup().world_size):
            filename = f"{filename_prefix}-{rank}.ckpt"
            ckpts[rank] = torch.load(filename)
        _logger.info(f'checkpoints = {ckpts}')

        state_dicts = []
        for ckpt in ckpts.values():
            model_state_dict = ckpt['state_dict']
            dist_param_map = ckpt['dist_param_map']
            param_area_map = ckpt['param_area_map']
            optimizer_state_dict = ckpt['optim_state_dict']
            state_dicts.push(model_state_dict, optimizer_state_dict, dist_param_map, param_area_map, )

        merged_model_state_dict, merged_optimizer_state_dict = CubeModule.merge_partial_states(state_dicts)

        # dump to ckpt
        torch.save({'state_dict': merged_model_state_dict,
                    'optim_state_dict': merged_optimizer_state_dict
                    }, filename_prefix + '.full.ckpt')


class ParallelModule(CubeModule):
    COMPUTE_CONFIG_FILE = 'compute_config.pt'

    def __init__(self):
        if self.__class__  == ParallelModule:  # not init via super().__init__()
            raise RuntimeError(f"ParallelModule should not be initialized directly. Please derive it first")

        super().__init__()
        # this is used to allow multiple sync_grad() calls
        self._sync_grad_required = False

    def _post_init(self):
        module_file = Path(sys.modules[self.__module__].__file__)
        self.load_attr_content(module_file.with_name(f"{FxModuleParser.ATTR_CONTENT_FILE}"))
        self._dist_param_map = torch.load(module_file.with_name(f"{FxModuleParser.ATTR_MAP_FILE}"))
        self._compute_config = torch.load(module_file.with_name(f"{self.COMPUTE_CONFIG_FILE}"))

        for reducer in self.reducers:
            reducer.build_buckets()

    def forward(self, *args, **kwargs):
        if self.training:
            self._sync_grad_required = True  # mark sync_grad() can be called again
        return self._forward_impl(*args, **kwargs)

    def _forward_impl(self, *args, **kwargs):
        """
        forward implementation. Should be implemented by subclass
        """
        raise NotImplementedError

    def sync_grad(self):
        """
        synchronize gradients using allreduce (non-zero) or reduce-scatter (zero)
        """
        if self._sync_grad_required:
            self._sync_grad_required = False  # mark sync_grad() has been called
            for reducer in self._reducers:
                reducer.sync_grads()

    def get_dist_param_map(self):
        return self._dist_param_map

    def get_compute_config(self):
        return self._compute_config
