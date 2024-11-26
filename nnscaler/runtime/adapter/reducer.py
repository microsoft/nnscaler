#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Dict, Tuple, Any, Callable, Optional, Set, Sequence
from functools import partial
import math
import logging
import torch
from torch.utils.hooks import RemovableHandle

from nnscaler.runtime.device import DeviceGroup
from nnscaler.profiler.timer import CudaTimer
from nnscaler.flags import RuntimeFlag

_logger = logging.getLogger(__name__)


# According to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
# Any address of a variable residing in global memory or returned by one of the memory allocation 
# routines from the driver or runtime API is always aligned to at least 256 bytes.
# But in our practice, we found that 16 bytes alignment is enough, it can be modified if unaligned access is detected.
ALIGNED_BYTES = 16


def _aligned_nbyte(nelement: int, element_size: int, align_size: int = ALIGNED_BYTES) -> int:
    """
    Align the number of elements, so the total byte size of elements is multiple of `align_size`
    Returns:
        the aligned number of bytes
    """
    if align_size % element_size != 0:
        raise ValueError(f"align_size {align_size} must be divisible by element_size {element_size}")
    return (nelement * element_size + align_size - 1) // align_size * align_size


def _aligned_nelement(nelement: int, element_size: int, align_size: int = ALIGNED_BYTES) -> int:
    """
    Align the number of elements, so the total byte size of elements is multiple of `align_size`
    Returns:
        the aligned number of elements
    """
    return _aligned_nbyte(nelement, element_size, align_size) // element_size


def _get_reduce_op(reduce_op: str) -> torch.distributed.ReduceOp:
    """
    Get reduce op from string
    """
    reduce_op = reduce_op.lower()  # to lower case
    supported = ['sum', 'avg', 'mean', 'min', 'max']
    if reduce_op == 'sum':
        return torch.distributed.ReduceOp.SUM
    elif reduce_op == 'avg' or reduce_op == 'mean':
        return torch.distributed.ReduceOp.AVG
    elif reduce_op == 'min':
        return torch.distributed.ReduceOp.MIN
    elif reduce_op == 'max':
        return torch.distributed.ReduceOp.MAX
    raise KeyError(f"Unsupported reduce op {reduce_op}. Supported reduce op: {supported}")


class Bucket:
    def __init__(self, params: List[torch.nn.Parameter],
                 param_buffer: torch.Tensor, grad_buffer: torch.Tensor,
                 reduce_op: torch.distributed.ReduceOp,
                 group: torch.distributed.ProcessGroup, async_op: bool, zero: bool,
                 zero_subgroup: torch.distributed.ProcessGroup = None,
                 zero_crossgroup: torch.distributed.ProcessGroup = None,
                 zero_use_reduce_scatter: bool = False,
                 align_size: int = ALIGNED_BYTES,
    ):
        """
        Create a communication unit for parameter allreduce.

        One allreduce will be called for all gradients associated to the parameters.
        The parameters are assumed to participate in backward and generate gradient.

        Args:
            params (List[torch.nn.Parameter]): the parameters
            param_buffer (torch.Tensor): Paramter contiguous buffer
            grad_buffer (torch.Tensor): gradient contiguous buffer
            reduce_op (torch.distributed.ReduceOp): the reduce op used by collectives
            group (torch.distributed.ProcessGroup): communication group
            async_op (bool): whether to use asynchronous operation
            zero (bool): whether to use zero optimization on gradients
            zero_subgroup (torch.distributed.ProcessGroup): the subgroup for zero optimization the current rank belongs to
            zero_crossgroup (torch.distributed.ProcessGroup): the communication group for cross zero group allreduce when reduce scatter is enabled
            zero_use_reduce_scatter (bool): whether to use reduce scatter for zero optimization
            align_size (int): the alignment size in bytes for each parameter
        """

        self._params: List[torch.nn.Parameter] = params
        self._pofset: Dict[torch.nn.Parameter, int] = {}
        self._reduce_op = reduce_op
        self._group = group
        self._wsz: int = torch.distributed.get_world_size(group=self._group)
        self._async_param_cnt: int = 0  # flag for triggering async communication
        self._async_handle = None  # asynchrounous communication handler
        self._hooks: List[Tuple[Any, RemovableHandle]] = []

        self._async: bool = async_op
        self._zero: bool = zero
        self._zero_use_reduce_scatter = zero_use_reduce_scatter
        self._contiguous_params = param_buffer
        self._contiguous_grads = grad_buffer
        assert grad_buffer.size() == param_buffer.size()
        assert grad_buffer.size(0) % self._wsz == 0, "internal error: buffer size not chunkable"
        # the parameter exposed for optimizer
        self._param_for_optimizer: torch.nn.Parameter = None
        # total number of parameters
        self._align_size: int = align_size
        if self._align_size % ALIGNED_BYTES != 0:
            raise ValueError(f"align_size {self._align_size} must be divisible by {ALIGNED_BYTES}")

        self._numel: int = sum(p.numel() for p in self._params)
        self._aligned_numel: int = sum(_aligned_nelement(p.nelement(), p.element_size(), self._align_size) for p in self._params)

        self._zero_subgroup = self._group if zero_subgroup is None else zero_subgroup
        self._zgroup_sz: int = torch.distributed.get_world_size(group=self._zero_subgroup)
        self._zero_crossgroup = zero_crossgroup

        # pre and post hooks for gradient synchronization
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []

        # only async will enable contiguous gradient
        self.build()
        self.register_hooks()

    @property
    def numel(self) -> int:
        """total number of parameters in the bucket"""
        return self._numel

    @property
    def params(self) -> List[torch.nn.Parameter]:
        """Parameter list"""
        return self._params

    @property
    def zero(self) -> bool:
        """Whether enable zero for this bucket"""
        return self._zero

    def get_aligned_numel(self, param) -> int:
        """
        Get the aligned number of elements for a parameter
        """
        return _aligned_nelement(param.nelement(), param.element_size(), self._align_size)

    def _group_reduce_scatter(self):
        """currently this function is only used in synchronous mode"""
        rank = torch.distributed.get_rank(group=self._zero_subgroup)
        partial_tensor = self._contiguous_grads.chunk(self._zgroup_sz, dim=0)[rank]
        if self._zgroup_sz == self._wsz:
            # number of zero groups is 1, thus only reduce scatter is enough
            # in this case, self._group == self._zero_subgroup
            torch.distributed.reduce_scatter_tensor(
                partial_tensor, self._contiguous_grads,
                op=self._reduce_op, group=self._zero_subgroup)
        else:
            # two steps for group reduce scatter
            # step #1, allreduce across corresponding GPUs across groups
            torch.distributed.all_reduce(
                self._contiguous_grads, op=self._reduce_op, group=self._zero_crossgroup)
            # step #2, reduce scatter within each group
            torch.distributed.reduce_scatter_tensor(
                partial_tensor, self._contiguous_grads,
                op=self._reduce_op, group=self._zero_subgroup)

    def build(self):
        """
        Build offset for each parameter
        This should only be called once during the construction of bucket.
        """
        ofst = 0
        for param in self._params:
            self._pofset[param] = ofst
            ofst += _aligned_nelement(param.nelement(), param.element_size(), self._align_size)
        # build parameter for optimizer (shared storage).
        # Its gradient will be updated everytime calling `self.sync_grads()`
        if not self._zero:
            opt = self._contiguous_params
        else:
            rank = torch.distributed.get_rank(group=self._zero_subgroup)
            assert len(self._contiguous_params) % self._zgroup_sz == 0
            # Note:
            #  There may be paddings both in the middle and at the end of the contiguous buffer
            #  When there are paddings in the middle or end of the contiguous buffer,
            #  the calculation of gnorm is not affected as long as the paddings are all 0.
            #   So for now, it looks harmless.
            opt = self._contiguous_params.chunk(self._zgroup_sz)[rank]
        self._param_for_optimizer = torch.nn.Parameter(opt)

    def register_hooks(self):
        """
        Register post-backward hook to each paramter

        The post-backward will change the generated gradient from `.grad` to `self._contiguous_grads`.
        The `.grad` will always keep as None until the finish of allreduce sync.
        After allreduce sync, each parameter will be reset by its `.grad` attribute, which
        shares the same storage from `self._contiguous_grads`.

        This should only be called once during the construction of bucket.
        """

        @torch.no_grad()
        def post_grad_hook(param: torch.nn.Parameter, *unused):
            # stream = DeviceGroup().get_stream('reducer')
            ofst = self._pofset[param]
            # TODO: need to handle sparse gradients in torch.nn.Embedding
            self._contiguous_grads[ofst:ofst+param.numel()].add_(param.grad.data.view(-1))
            param.grad = None

            if RuntimeFlag.skip_reducer: return
            self._async_param_cnt += 1

            # perform all-reduce
            if self._async:
                if self._async_param_cnt > len(self._params):
                    raise RuntimeError(
                        "Detected gradient accumulation with asynchronous Reducer. "
                        "Users should run with `nnscaler.accum_mode` to manage gradient synchronization.")
                if self._async_param_cnt == len(self._params):
                    # apply pre hooks
                    self._apply_pre_hooks()
                    # communication
                    if self._zero and self._zero_use_reduce_scatter:
                        if self._zgroup_sz == self._wsz:
                            rank = torch.distributed.get_rank(group=self._group)
                            shards = list(self._contiguous_grads.chunk(self._wsz, dim=0))
                            # inplace reduce scatter is supported
                            # see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduceScatter
                            self._async_handle = torch.distributed.reduce_scatter(
                                shards[rank], shards, op=self._reduce_op,
                                group=self._group, async_op=True)
                        else:
                            assert False, "group zero + reducescatter is not supported in async mode, " \
                                          "because the two steps (allreduce, reducescatter) use " \
                                          "two communication groups, which may induce deadlock."
                            self._group_reduce_scatter()
                    else:
                        self._async_handle = torch.distributed.all_reduce(
                            self._contiguous_grads, op=self._reduce_op,
                            group=self._group, async_op=True)

        for param in self._params:
            # same trick with FSDP and Megatron
            # reference: https://github.com/pytorch/pytorch/blob/v1.13.1/torch/distributed/fsdp/fully_sharded_data_parallel.py#L3177-L3188
            param_tmp = param.expand_as(param)
            # gets its AccumulateGrad object
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            hook = grad_acc.register_hook(partial(post_grad_hook, param))
            # grad_acc must keep, otherwise the hook won't take effect
            self._hooks.append((grad_acc, hook))

    def sync_grads(self):
        """
        Wait until allreduce finished (async), or perform allreduce (sync).

        The `.grad` attribute for each parameter will also be set after
        the completion of allreduce.
        """
        rank = torch.distributed.get_rank(group=self._group)
        # async
        if self._async:
            if CudaTimer().enabled and CudaTimer().predefined:
                _logger.warning(
                    f'CudaTimer: the communication time of async reducer will not be recorded in `comm`')
            assert self._async_handle is not None
            self._async_handle.wait()
        else:
            CudaTimer().start('comm', predefined=True)
            # apply pre-hooks
            self._apply_pre_hooks()
            # synchrnoize gradients
            if self._zero and self._zero_use_reduce_scatter:
                self._group_reduce_scatter()
            else:
                torch.distributed.all_reduce(
                    self._contiguous_grads, op=self._reduce_op, group=self._group)
            CudaTimer().stop('comm', predefined=True)
        # grads = self._contiguous_grads.clone()
        for param in self._params:
            assert param.grad is None
            pofst = self._pofset[param]
            param.grad = self._contiguous_grads[pofst:pofst+param.numel()].view(param.size())

        # setup gradient for optimizer parameters
        if self._zero:
            rank = torch.distributed.get_rank(group=self._zero_subgroup)
            grad = self._contiguous_grads.chunk(self._zgroup_sz, dim=0)[rank]
            self._param_for_optimizer.grad = grad
        else:
            self._param_for_optimizer.grad = self._contiguous_grads

        # apply post-hooks
        self._apply_post_hooks()

    def gather_params(self):
        """
        All-gather parameters
        """
        assert self._zero, "gathering paramters is only for zero optimization."
        rank = torch.distributed.get_rank(group=self._zero_subgroup)
        CudaTimer().start(field_name='comm', predefined=True)
        src_tensor = self._contiguous_params.chunk(self._zgroup_sz, dim=0)[rank]
        torch.distributed.all_gather_into_tensor(self._contiguous_params, src_tensor, group=self._zero_subgroup)
        CudaTimer().stop(field_name='comm', predefined=True)

    def register_pre_hook(self, fn: Callable):
        """Register pre hooks to be applied before gradient synchronization.

        The pre-hooks will be applied one by one following the order of registration.

        Args:
            fn (Callable): a callable function that takes a gradient as input and optionally updates the gradient.
        """
        assert callable(fn), f"fn must be callable for pre hooks, but got {type(fn)}"
        self._pre_hooks.append(fn)

    def register_post_hook(self, fn: Callable):
        """Register post hooks to be applied after gradient synchronization.

        The post-hooks will be applied one by one following the order of registration.

        Args:
            fn (Callable): a callable function that takes a gradient as input and optionally updates the gradient.
        """
        assert callable(fn), f"fn must be callable for post hooks, but got {type(fn)}"
        self._post_hooks.append(fn)

    def _apply_pre_hooks(self):
        """Apply pre hooks before gradient synchronization.

        The pre-hooks will be applied one by one following the order of registration.
        """
        if len(self._pre_hooks) == 0: return
        grads = self._contiguous_grads
        for hook in self._pre_hooks:
            hook(grads)

    def _apply_post_hooks(self):
        """Apply post hooks after gradient synchronization.

        The post-hooks will be applied one by one following the order of registration.
        """
        if len(self._post_hooks) == 0: return
        grads = self._contiguous_grads
        for hook in self._post_hooks:
            hook(grads)

    def clear_pre_hooks(self):
        """Clear all pre hooks."""
        self._pre_hooks = []

    def clear_post_hooks(self):
        """Clear all post hooks."""
        self._post_hooks = []

    def reset(self):
        """Reset status."""
        self._async_param_cnt = 0
        self._async_handle = None


class Reducer:
    # the default bucket cap for async reducer in megabytes
    # with the same value as pytorch
    # https://github.com/pytorch/pytorch/blob/4fd16dd8aa259cd75c9a6d2ddcd8171cd1ee8e28/torch/nn/parallel/distributed.py#L548
    _DEFAULT_BUCKET_CAP_MB = 25  # 25MB, the same as pytorch

    def __init__(self, ranks: List[int], max_bucket_size_bytes: Optional[int] = None,
                 reduce_op: str = 'sum', async_op: bool = False,
                 zero: bool = False, zero_ngroups: int = 1,
                 zero_use_reduce_scatter: bool = False,
                 align_size: int = ALIGNED_BYTES
    ):
        """
        Create a reducer applied on a set of weights for weight reduction

        This assumes the communication group is already created by every rank.

        Args:
            ranks (List[int]): reducer communication group
            max_bucket_size_bytes (Optional[int]): largest bucket size for one-time communication,
                `0` or `None` will use default value,
                which is `_DEFAULT_BUCKET_CAP_MB` for async reducer, and no limit for sync reducer.
                Default is `None`
            reduce_op (str): reduce operation, can be 'sum', 'avg', 'max' or 'min' (default 'sum')
            async_op (bool): whether to overlap with backward computation (default False)
            zero (bool): whether to apply ZeRO optimization on gradients
            zero_ngroups (int): number of ZeRO subgroups in the original ZeRO group
            zero_use_reduce_scatter (bool): whether to use reduce scatter for zero optimization
            align_size (int): the alignment size in bytes for each parameter
        """
        self._params: List[torch.nn.Parameter] = list()
        self._param_ids: Set[int] = set()
        self._numel: int = 0
        self._ranks = ranks
        self._group = DeviceGroup().get_group(ranks)
        self._wsz: int = torch.distributed.get_world_size(group=self._group)

        self._bucket_size: Optional[int] = max_bucket_size_bytes
        if not self._bucket_size and async_op:
            self._bucket_size = self._DEFAULT_BUCKET_CAP_MB * 1024 * 1024

        self._reduce_op = _get_reduce_op(reduce_op)
        # buckets stands for a transission unit
        self._buckets: List[Bucket] = list()
        self._async: bool = async_op
        self._zero: bool = zero
        self._zero_use_reduce_scatter = zero_use_reduce_scatter
        self._align_size: int = align_size
        if self._align_size % ALIGNED_BYTES != 0:
            raise ValueError(f"align_size {self._align_size} must be divisible by {ALIGNED_BYTES}")

        # contiguous parameter buffer and gradient buffer
        self._contiguous_params: torch.Tensor = None
        self._contiguous_grads: torch.Tensor = None

        # build the subgroup of zero the current rank belongs to.
        # When zero_ngroups is larger than 1, the number of ranks
        # will be divided by zero_ngroups into sub rank groups,
        # allgather of weights will be done within each subgroup.
        # For example, if the ranks are [0, 1, 2, 3, 4, 5, 6, 7] and zero_ngroups=2,
        # the ranks will be divided into [0, 1, 2, 3] and [4, 5, 6, 7].
        # If the ranks are [0, 2, 4, 6], zero_ngroups=2, then the ranks
        # will be divided into [0, 2] and [4, 6].
        if self._zero and self._zero_use_reduce_scatter:
            _logger.info(f"Using reduce scatter for ZeRO optimization")
            # TODO: In current implementation of Bucket,
            # zero_use_reduce_scatter works when zero_ngroups > 1 in sync mode
            # We can enable it in sync mode when it is proved to be useful.
            if zero_ngroups > 1:
                raise ValueError("reduce scatter is not supported when zero_ngroups > 1")

        if zero_ngroups > 1:
            assert self._zero, f"USE_ZERO must be set when ZERO_NUM_GROUPS is larger than 1"
            assert len(ranks) % zero_ngroups == 0, f"length of ranks {ranks} must be divisible by zero factor {zero_ngroups}"
            curr_rank = torch.distributed.get_rank(group=self._group)
            zgroup_sz = len(ranks) // zero_ngroups
            group_idx = curr_rank // zgroup_sz
            sub_ranks = ranks[group_idx * zgroup_sz : (group_idx + 1) * zgroup_sz]
            if len(sub_ranks) > 1:
                assert DeviceGroup().group_exists(sub_ranks), f"zero subgroup {sub_ranks} does not exist in comm groups"
            self._zero_subgroup = DeviceGroup().get_group(sub_ranks)
            # crossgroup is for the allreduce across zero subgroups, it is only used when
            # reduce scatter is enabled and the number of zero subgroups is larger than 1.
            start_rank = curr_rank % zgroup_sz
            cross_ranks = ranks[start_rank::zgroup_sz]
            assert len(cross_ranks) == zero_ngroups
            self._zero_crossgroup = DeviceGroup().get_group(cross_ranks)
        else:
            assert zero_ngroups == 1, f"ZeRO number of groups must be 1, but got {zero_ngroups}"
            self._zero_subgroup = self._group
            self._zero_crossgroup = None
        self._zero_ngroups = zero_ngroups

    @property
    def zero_ngroups(self) -> int:
        return self._zero_ngroups

    @property
    def params(self) -> Tuple[torch.nn.Parameter, ...]:
        return tuple(self._params)

    @property
    def ranks(self) -> Tuple[int, ...]:
        return tuple(self._ranks)

    @property
    def numel(self) -> int:
        """Total number of parameters"""
        return self._numel

    @property
    def zero(self) -> bool:
        """Whether to apply zero optimization on gradients"""
        return self._zero

    @property
    def buckets(self) -> Tuple[Bucket, ...]:
        return tuple(self._buckets)

    @property
    def reduce_op(self) -> torch.distributed.ReduceOp:
        """Get reduce operation"""
        return self._reduce_op

    def add_param(self, param: torch.nn.Parameter):
        """
        Add a parameter to the reducer

        The reducer assumes the ordering of added parameter
        is consistent with forward order. Otherwise, the overlapping
        will show less benefits.

        @param param torch.nn.Parameter: the added parameter
        """
        if param.data.data_ptr() in self._param_ids:
            _logger.warning(
                f'rank [{torch.distributed.get_rank()}]: detected duplicated or shared parameters, ignored.')
            return
        self._params.append(param)
        self._param_ids.add(param.data.data_ptr())
        self._numel += param.numel()

    def build_buckets(self):
        """
        Build buckets the reducer.

        The parameters in each bucket have consistent data types,
        and each bucket contains at least one parameter.
        If the bucket contains more than 2 parameters, than the total size is samller
        than the max_bucket_size_bytes.
        """
        # step 1: build bucket for overlapping gradient synchronization
        # self._numel * 8 + 1 here is to make sure
        # the bucket size is larger than the total size of all parameters
        # 8 is the size of float64, which is the largest data type in PyTorch

        # TODO: we may use a small bucket size for the first bucket, which is used in pytorch
        # https://github.com/pytorch/pytorch/blob/4fd16dd8aa259cd75c9a6d2ddcd8171cd1ee8e28/torch/nn/parallel/distributed.py#L1172C17-L1172C36
        # TODO: use native version of reducer, which is more efficient
        #       (used in pytorch, with a couple percentage improvement)
        bucket_size = self._numel * 8 + 1 if not self._bucket_size else self._bucket_size

        # items in the bucket is params list
        seq_buckets: List[List[torch.nn.Parameter]] = []
        last_bucket_size = None

        assert len(set(p.dtype for p in self._params)) == 1, (
            "All parameters in the reducer should have the same data type"
        )
        for param in self._params:
            if param.requires_grad:
                cur_byte_size = _aligned_nelement(param.nelement(), param.element_size(), self._align_size) * param.element_size()
                # also work when cur_byte_size > bucket_size
                # It will go the `else` branch
                # and finish the current bucket and start a new bucket.
                # This new bucket will be sealed in the next iteration
                if len(seq_buckets) == 0:
                    seq_buckets.append([param])
                    last_bucket_size = cur_byte_size
                elif last_bucket_size + cur_byte_size <= bucket_size:
                    seq_buckets[-1].append(param)
                    last_bucket_size += cur_byte_size
                else:
                    seq_buckets.append([param])
                    last_bucket_size = cur_byte_size

        # step 2: build meta data for the offset of each bucket
        # the start of each bucket will be padded to the next multiple of `len(self.ranks)`
        buffer_length: int = 0
        starts, stops = [], []
        for params in seq_buckets:
            starts.append(buffer_length)
            numel = sum(_aligned_nelement(p.nelement(), p.element_size(), self._align_size) for p in params)
            # this pad is for zero, which needs numels in each Bucket can be divided by the number of ranks in this group * _align_size
            # so that each chunck during zero can be divided by _align_size
            align_nelements = self._align_size // params[0].element_size() * len(self._ranks)
            padding = (align_nelements - numel % align_nelements) % len(self._ranks)
            buffer_length += numel + padding
            stops.append(buffer_length)

        # step3: allocate memory
        # gradient buffer
        self._contiguous_grads: torch.Tensor = torch.zeros(
            (buffer_length,), dtype=self._params[0].dtype,
            device=torch.cuda.current_device(), requires_grad=False)
        # parameter buffer
        self._contiguous_params: torch.Tensor = torch.zeros(
            (buffer_length,), dtype=self._params[0].dtype,
            device=torch.cuda.current_device(), requires_grad=False)

        # step 4: build buckets
        buckets: List[Bucket] = []
        for params, start, stop in zip(seq_buckets, starts, stops):
            # replace underlying parameter content using shared storage from parameter
            ofst = start
            for param in params:
                with torch.no_grad():
                    self._contiguous_params[ofst:ofst+param.numel()].copy_(param.data.view(-1))
                    param.data = self._contiguous_params[ofst:ofst+param.numel()].view(param.size())
                aligned_nelements = _aligned_nelement(param.nelement(), param.element_size(), self._align_size)
                ofst += aligned_nelements
            # initialize buckets
            bucket = Bucket(
                params,
                self._contiguous_params[start:stop],
                self._contiguous_grads[start:stop],
                self._reduce_op,
                self._group,
                self._async,
                self._zero,
                self._zero_subgroup,
                self._zero_crossgroup,
                self._zero_use_reduce_scatter,
                self._align_size,
            )
            buckets.append(bucket)
        torch.cuda.empty_cache()

        # make it in reverse order as the backward happens from tail to head
        # it is not important but may be helpful for waiting cuda stream to finish
        self._buckets: List[Bucket] = list(reversed(buckets))
        assert len(self._buckets) > 0, (
            f"Find {len(self._params)} parameters in the reducer. "
            f"Make sure adding all parameters before building buckets")

    def sync_grads(self):
        """
        synchronize gradients using allreduce (non-zero) or reduce-scatter (zero)
        """
        if RuntimeFlag.skip_reducer: return
        for bucket in self._buckets:
            bucket.sync_grads()

    def gather_params(self):
        """Gather parameters with Zero optimizations after `optimizer.step()`.

        This is required when zero optimization is turned on.
        """
        if not self._zero: return
        for bucket in self._buckets:
            bucket.gather_params()

    def zero_grad(self):
        """Make gradient to be zero.

        This needs to be called at the beginning of every training iteration.
        """
        if RuntimeFlag.skip_zero_grad: return
        torch.cuda.synchronize()
        self._contiguous_grads.zero_()
        for bucket in self._buckets:
            bucket.reset()
            bucket._param_for_optimizer.grad = None
        for param in self.params:
            param.grad = None

    def parameters_for_optimizer(self) -> List[torch.nn.Parameter]:
        """
        Get parameters for optimizers
        Please note for ZeRO optimization,
        the returned parameters are not the same as the original parameters,
        and can have paddings (with value 0.0) both at the end and in the middle of paramters data.

        the calculation of gnorm is not affected as paddings are all 0.

        Returns:
            List[torch.nn.Parameter]: parameters for optimizer
        """
        params = []
        for bucket in self._buckets:
            params.append(bucket._param_for_optimizer)
        return params

    def broadcast_params(self):
        """
        broadcast parameters before training
        """
        for param in self._params:
            torch.distributed.broadcast(param, self.ranks[0], group=self._group)
        torch.cuda.synchronize()

    def register_pre_hook(self, fn: Callable):
        """Register a pre hook function before gradient update.

        A reducer can be registered by multiple hooks and the hooks will be
        applied in the order of registration.

        The hook function takes a contiguous buffer of local computed gradient
        and can optionally apply in-place operations on it.

        Example:

        ```
        hook = lambda grad: grad.div_(4)
        reducer.register_pre_hook(hook)
        ```

        Args:
            fn Callable:
                hook function that takes a gradient as input and optionally inplacemently updates it
        """
        assert callable(fn), f"pre hook function must be callable, but got {type(fn)}"
        for bucket in self._buckets:
            bucket.register_pre_hook(fn)

    def register_post_hook(self, fn: Callable):
        """
        Register a post hook function after gradient update.

        A reducer can be registered by multiple hooks and the hooks will be
        applied in the order of registration.

        The hook function takes a contiguous buffer of updated gradient
        and can only apply in-place operations on it.

        Example:

        ```
        hook = lambda grad: grad.clamp_(min=-1, max=1)
        reducer.register_post_hook(hook)
        ```

        Args:
            fn Callable:
                hook function that takes a gradient as input and optionally inplacemently updates it
        """
        assert callable(fn), f"post hook function must be callable, but got {type(fn)}"
        for bucket in self._buckets:
            bucket.register_post_hook(fn)

    def clear_pre_hooks(self):
        """Clear all pre hooks."""
        for bucket in self._buckets:
            bucket.clear_pre_hooks()

    def clear_post_hooks(self):
        """Clear all post hooks."""
        for bucket in self._buckets:
            bucket.clear_post_hooks()
