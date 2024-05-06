# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple, List, Dict, Callable, Optional, NewType, Set
import os
import torch
import time
import json

from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.segment import IRSegment
from cube.graph.function import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.graph.function.pyfunc import IRPyFunc

import cube
import operator
from cube.ir.cten import IRTensor, IRObject, IRCell
from cube.ir.operator import IRFwOperation
from cube.graph.parser.register import CustomizedOps


Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
Split = NewType('SplitConfig', Tuple[int, int, int])


_train_module_ref: torch.nn.Module = torch.nn.Module().train()
_eval_module_ref: torch.nn.Module = torch.nn.Module().eval()


class CompProfiler:

    @staticmethod
    def profile(node: IRCell, train: bool = True,
                warmup_sec: float = 2, prof_times: int = 10) -> Tuple[float, float, int, Tuple[int]]:
        """Profile a function node

        Note:
            The profiled memory contains:
                - intermediate memory (saved tensors for training and peak memory for inference)
                - output tensor memory

            The input tensors will not be considered (which
            should be considered by this operator's precessors)

        Args:
            func (Callable): the callable function, e.g., torch.nn.functional.linear
            shapes (Tuple[Tuple[int]]): the shapes of each input tensor
            dtypes (Tuple[torch.dtype], None): the dtype of each input tensor. Default will use torch.float32
            warmup_sec (float): warmup seconds
            prof_times (int): profile times

        Returns:
            float: inference time in milliseconds
            int: inference peak memory in bytes
            float: train time (forward + backward) in milliseconds
            int: train memory in bytes
        """
        torch.cuda.empty_cache()
        if isinstance(node, (IRGraphAnchor, IRPyFunc)) or node.name == 'multiref':
            return (0.0,) * 4
        func: Callable = CompProfiler.get_func(node)
        args, kwargs = CompProfiler.get_inputs(node, train=True)
    
        # prepare gradients
        with torch.no_grad():
            outputs = func(*args, **kwargs)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        assert all(torch.is_tensor(otensor) for otensor in outputs), \
            f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)
        del outputs

        def run_step(func, tensors, kwargs, backward: bool):
            if not backward:
                with torch.no_grad():
                    outputs = func(*tensors, **kwargs)
            else:
                outputs = func(*tensors, **kwargs)
                torch.autograd.backward(outputs, grads)
            return outputs

        # ================ measure training peak memory ====================
        
        # inference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mtic = torch.cuda.max_memory_allocated()  # in bytes
        run_step(func, args, kwargs, backward=False)
        torch.cuda.synchronize()
        mtoc = torch.cuda.max_memory_allocated()
        infer_memory = mtoc - mtic

        # training -- pack_hook only saves activation tensors.
        # weight tensors will not be saved and go through pack_hook

        # train_memory, used_tensor = 0, set()
        # def get_data_ptr(t: torch.Tensor):
        #     # torch 2.0: change x.torch.storage() -> x.torch.untyped_storage()
        #     return t.untyped_storage().data_ptr()
        # 
        # input_tensors = [t for t in args if isinstance(t, torch.Tensor)] + \
        #                 [t for t in kwargs.values() if isinstance(t, torch.Tensor)]
        # for t in input_tensors:
        #     used_tensor.add(get_data_ptr(t))
        # 
        # def pack_hook(x: torch.Tensor):
        #     nonlocal train_memory, used_tensor
        #     dptr = get_data_ptr(x)
        #     if dptr not in used_tensor:
        #         used_tensor.add(dptr)
        #         train_memory += x.numel() * x.element_size()
        #     return x
        # def unpack_hook(x): return x
        # 
        # if train:
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()
        #     with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        #         outputs = run_step(func, args, kwargs, backward=True)
        #     torch.cuda.synchronize()
        # 
        #     # include output tensors
        #     def get_tensor_from_complex(data_structure) -> List[torch.Tensor]:
        #         tensors = []
        #         if isinstance(data_structure, (tuple, list)):
        #             for t in data_structure:
        #                 tensors += get_tensor_from_complex(t)
        #         if isinstance(data_structure, dict):
        #             for t in data_structure.values():
        #                 tensors += get_tensor_from_complex(t)
        #         if isinstance(data_structure, torch.Tensor):
        #             tensors.append(data_structure)
        #         return tensors
        #     
        #     for t in get_tensor_from_complex(outputs):
        #         train_memory += t.numel() * t.element_size()
        # 
        # del used_tensor

        train_memory = 0
        if train:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            entry = torch.cuda.memory_allocated()  # bytes
            outputs = func(*args, **kwargs)
            torch.cuda.synchronize()
            train_memory = torch.cuda.memory_allocated() - entry
            assert train_memory >= 0, f"fn: {func.__name__}: Unexpected behaviour on decreased memory: {train_memory}"
            torch.autograd.backward(outputs, grads)
            torch.cuda.synchronize()
            del outputs

        # ===================================================================

        # warmup
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tic = time.time()
        while time.time() - tic < warmup_sec:
            run_step(func, args, kwargs, backward=train)
            torch.cuda.synchronize()

        def profile(backward: bool):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            for _ in range(prof_times):
                run_step(func, args, kwargs, backward=backward)
            torch.cuda.synchronize()
            toc = time.perf_counter()
            return (toc - tic) / prof_times * 1000  # in milliseconds

        infer_span = profile(backward=False)
        train_span = 0.0
        if train:
            train_span = profile(backward=True)
        
        return infer_span, infer_memory, train_span, train_memory

    @staticmethod
    def get_inputs(node: IRFwOperation, train: bool) -> Tuple[List, Dict]:
        # create data
        def dummy_torch_tensor(tensor: IRTensor):
            """Generate dummy input tenosrs"""
            dtype = tensor.dtype
            assert isinstance(dtype, torch.dtype), f"Found unkown dtype: {dtype}"
            constructor = torch.ones
            # constructor = torch.zeros if dtype in (torch.int64, torch.int32, torch.bool) else torch.rand
            return constructor(tuple(tensor.shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=tensor.requires_grad)

        args = [dummy_torch_tensor(t) if isinstance(t, IRTensor) else t for t in node.inputs()]
        # replace kwargs starting with 'self.xxx'
        kwargs = {}
        for name, value in node.kwargs.items():
            if isinstance(value, str) and value.startswith('self.'):
                value = getattr(_train_module_ref, value[5:]) if train else getattr(_eval_module_ref, value[5:])
            kwargs[name] = value
        
        return args, kwargs

    @staticmethod
    def get_func(node: IRFwOperation) -> Callable:
        """Get function call"""
        assert isinstance(node, IRFwOperation), f"Only support profiling forward operation but got {type(node)}"
        if node.signature in CustomizedOps.kOpCodeDef:
            fn = CustomizedOps.kOpRuntime[node.signature]
        else:
            fn = eval(node.signature)
        return fn


class ProfileDataBase:

    def __init__(self, filename: Optional[str] = None) -> None:
        """
        Create a database for profiling result
        """
        self._data: Dict[str, Dict[str, Tuple[float, float, int]]] = dict()
        if filename is not None:
            self.load(filename)

    def profile(self, node: IRFwOperation, train: bool = True, device: Optional[int] = None):
        """
        Profile a forward node in IRGraph on a specific device (default current device)
        
        Args:
            node (IRFwOperation): node of IRGraph
            device (int): the device that the node will execute on
        
        Returns:
            float: inference time in milliseconds
            int: inference peak memory in bytes
            float: train time (forward + backward) in milliseconds
            int: train memory in bytes
        """
        if isinstance(node, (IRGraphAnchor, IRPyFunc)) or node.name == 'multiref':
            return (0.0,) * 4

        if self.exist(node):
            return self.query(node)

        if isinstance(device, int):
            orig_device = torch.cuda.current_device()
            torch.cuda.set_device(device)

        color, default = '\033[31m', '\033[0m'

        infer_span, infer_memory, train_span, train_memory = CompProfiler.profile(node, train)
        # log to database
        self.insert(node, infer_span, infer_memory, train_span, train_memory)

        shapes = tuple(t.shape if isinstance(t, IRTensor) else None for t in node.inputs())
        dtypes = tuple(t.dtype if isinstance(t, IRTensor) else None for t in node.inputs())
        error = f'{color}None{default}'
        print(
            f"=> {node.signature} | shapes: {shapes} | dtypes: {dtypes} => "
            f"infer: {round(infer_span, 2) if isinstance(infer_span, float) else error} ms | "
            f"{infer_memory if isinstance(infer_memory, int) else None} bytes ; "
            f"train: {round(train_span, 2) if isinstance(train_span, float) else error} ms | "
            f"{train_memory if isinstance(train_memory, int) else error} bytes")

        if isinstance(device, int):
            torch.cuda.set_device(orig_device)
        return infer_span, infer_memory, train_span, train_memory

    def insert(self, node: IRCell, infer_span: float, infer_memory: int,
               train_span: float, train_memory: int):
        """Log (reset) the span of a node with key

        Args:
            node (IRCell): profiled node
            infer_span (float): inference time in milliseconds
            infer_memory (int): inference peak memory in bytes
            train_span (flaot): train time in milliseconds
            train_memory (int): train peak memory in bytes
        """
        name = node.signature
        key = self._serialize(node)
        assert isinstance(name, str) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        infer_span = infer_span if isinstance(infer_span, float) else None
        infer_memory = infer_memory if isinstance(infer_memory, int) else None
        train_span = train_span if isinstance(train_span, float) else None
        train_memory = train_memory if isinstance(train_memory, int) else None
        self._data[name][key] = (infer_span, infer_memory, train_span, train_memory)

    def exist(self, node: IRFwOperation) -> bool:
        """
        Check if the node has the performance recorded in the database

        Args:
            node (IRFwOperation): forward operation

        Returns:
            bool: True if the performance is recorded, else False
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return False
        if key not in self._data[node.signature]:
            return False
        return True

    def query(self, node: IRFwOperation) -> Tuple[float, int, float, int]:
        """Get the performance of a node

        Args:
            node (IRFwOperation): node in IRGraph

        Returns:
            float: inference time in milliseconds
            int: inference peak memory in bytes
            flaot: train time in milliseconds
            int: train peak memory in bytes
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return None
        if key not in self._data[node.signature]:
            return None
        return self._data[node.signature][key]

    def _serialize(self, node: IRFwOperation) -> str:
        """
        Serialize the shapes, dtypes and kwargs into a string

        e.g.,
            shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
        => ((1024,), (1024,1024)) : (torch.float32, torch.float32)

        Args:
            shapes (Tuple[Tuple[int]]): the shape of each tensor
            dtypes (Tuple[torch.dtype]): the dtype of each tensor

        Returns:
            key str: the serialized string
        """
        shapes, dtypes = [], []
        for t in node.inputs():
            if isinstance(t, IRTensor):
                shapes.append(t.shape)
                dtypes.append(t.dtype)
            elif isinstance(t, IRObject):
                raise RuntimeError('IRObject has not been supported in _serialize')
            else:
                shapes.append(None)
                dtypes.append(type(t))
        shapes = str(tuple(shapes))
        dtypes= str(tuple(dtypes))
        return shapes + ' : ' + dtypes

    def _deserialize(self, key: str) -> ShapesDTypes:
        """
        De-serialize the key string to shapes and dtypes

        e.g., (1024,)-(1024,1024) : torch.float32-torch.float32
        =>  shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)

        Args:
            key (str): the serialized string
        
        Returns:
            shapes_and_dtypes (ShapesDTypes): shapes and dtypes
        """
        shapes, dtypes = key.split(' : ')
        shapes = eval(shapes)
        dtypes = eval(dtypes)
        return shapes, dtypes

    def dump(self, file: str, override=False):
        """!
        dump the profiled data into json format

        Args:
            file (str): the json file name
            override (bool): True if the existed can be overrided else False
        """
        if os.path.exists(file):
            assert override, f"File {file} exists. Set override = True to force dump."
        with open(file, 'w') as f:
            json.dump(self._data, f)

    def load(self, file: str):
        """!
        load the profiled data into data base. The original existed one will be
        overrided by the loaded data.

        Args:
            file (str): the json file name
        """
        with open(file, 'r') as f:
            self._data = json.load(f)

    def __repr__(self) -> str:
        data = []
        for signature in self._data:
            for key in self._data[signature]:
                shapes, dtypes = self._deserialize(key)
                in_mem_info, param_mem_info, fw_span, bw_span, infer_mem, train_mem = self._data[signature][key]
                data.append(f'{signature}: shapes={shapes}, dtypes={dtypes}, in mem {in_mem_info} bytes, param mem {param_mem_info} bytes, fw span: {fw_span} ms, bw span: {bw_span} ms, infer mem {infer_mem} bytes, train mem {train_mem} bytes')
        data = '\n'.join(data)
        return data


class Estimator:
    """
    Estimator to measture the computation / memory cost of a subgraph
    """
    # skipped node name
    _skip_node_names: Set[str] = set()
    # node.name -> (idx, dim, num)
    _rules: Dict[str, Tuple[int, int, int]] = {}

    def __init__(self, 
                 optim_nstates: int = 2,
                 optim_keep_fp32_states: bool = True,
                 optim_keep_fp32_params: bool = False,
                 cache='./profile_database.json'):
        """
        Args:
            optim_nstates (int): number of optimizer states. By default to
                2 to align with Adam optimizer.
            optim_keep_fp32_states (bool): whether to keep fp32 states
                in optimizer. This is by default True in mixed-precision-training.
            optim_keep_fp32_params (bool): whether to keep a copy of
                fp32 params in optimizer. This is by default False in mixed-precision-training.
                The optimizer can be optimized to turn param into fp32 in ad-hoc during step.
            cache (str): cached file for profiled data.
        """
        self.optim_nstates: int = optim_nstates
        self.optim_keep_fp32_states: bool = optim_keep_fp32_states
        self.optim_keep_fp32_params: bool = optim_keep_fp32_params
        self.cache_file = cache
        reload = cache if os.path.exists(cache) else None
        self.database = ProfileDataBase(reload)

    def perf(self, node: IRFwOperation, split: Split = None, train: bool = True):
        """Query the node performance with partition configurations.

        If the performance data doesn't exsit in database, will profile in runtime.

        Args:
            node (IRFwOperation): the queried node
            split (Tuple[int, int, int] or None):
                Only for IRDimops, the config should be (idx, dim, num)

        Returns:
            infer_span (float): the time in milliseconds for forward time
            infer_mem (int): the peak memory in bytes after inference of the function
            train_span (float): the time in milliseconds for backward time
            train_mem (int): the peak memory in bytes during training
        """
        if isinstance(node, (IRGraphAnchor, IRPyFunc)) or node.name == 'multiref':
            return (0.0,) * 4
        if node.name in Estimator._skip_node_names:
            return (0.0,) * 4
        split = tuple(split) if split is not None else split
        if split is not None:
            assert isinstance(node, IRDimops)
            algo = node.algorithms('dim')
            idx, dim, num = split
            if num > 1:
                if not algo.satisfy(idx=idx, dim=dim, num=num):
                    return 1e9, 1e9, 1e9, 1e9
                node = algo.instantiate(idx=idx, dim=dim, num=num)[0]
        if self.database.exist(node):
            return self.database.query(node)
        
        multiple, origin_node = 1, node
        # split when a node is too large to be profiled inside one node
        if node.name in self._rules:
            idx, dim, num = self._rules[node.name]
            assert isinstance(node, IRDimops)
            algo = node.algorithms('dim')
            if num > 1 and algo.satisfy(idx=idx, dim=dim, num=num):
                node = algo.instantiate(idx=idx, dim=dim, num=num)[0]
                multiple = num

        outs = self.database.profile(node, train)
        if multiple > 1:
            # we assume linear scalability of memory and time span
            # as node is large enough
            outs = tuple(i * multiple for i in outs)
            self.database.insert(origin_node, *outs)
        return outs

    def profile_transform_space(self, node: IRFwOperation, max_num: int = 32, train: bool = True) -> None:
        """Profile the node at its full transformation space

        The profiled results will be logged into database.

        Note:
            Only IRDimops will be profiled with multiple transformation choices.
            Other nodes of IRFwOperation can only be replicated and therefore
            have no partitioning space. 

            The partitioned number will be limited to power of 2,
            e.g., 2, 4, 8, 16, ...

        Args:
            node (IRFwOperation): the profiled node
            max_num (int): the maximal number of partitioned node

        Returns:
            None
        """
        splits = [None]
        nodes = [node]
        if isinstance(node, IRDimops):
            algo = node.algorithms('dim')
            for (idx, dim) in node.transform_space():
                num = 2
                while num <= max_num:
                    if algo.satisfy(idx=idx, dim=dim, num=num):
                        sub_node = algo.instantiate(idx=idx, dim=dim, num=num)[0]
                        nodes.append(sub_node)
                        splits.append((idx, dim, num))
                        num *= 2
                    else:
                        break
        # print(f'profile {node.name} with configs: {splits}')
        for n in nodes:
            self.perf(n, train=train)

    def profile_model(self, graph: IRSegment):
        """Profile the operators in the graph
        
        Each operator will be profiled by all possible transformation space.
        The profiled results will be logged into database.

        Args:
            graph (IRSegment): the graph

        Returns:
            None
        """
        for node in graph.select(ntype=IRFwOperation):
            self.profile_transform_space(node, train=(node.mirror is not None))

    def peak_activation_mem(self, nodes: Tuple[IRFwOperation],
                            inflights: int = 1, train: bool = True):
        """Profile the activation memory cost of executing a sub-graph

        Args:
            nodes (Tuple[IRFwOperation]): the sub-graph
            inflights (int): the number of in-flight sub-graphs

        Returns:
            int: peak activation memory in byte size.
        """
        non_recompute_mem = 0
        recompute_mem, curr_recomp_id = [0], None
        bound_mem = 0  # recompute group boundary memory
        last_node = None
        for node in nodes:
            infer_span, infer_mem, train_span, train_mem = self.perf(node)
            node_mem = train_mem if train else infer_mem
            if node.recompute is None:
                non_recompute_mem += node_mem
                curr_recomp_id = None
            else:
                if node.recompute != curr_recomp_id:
                    recompute_mem.append(node_mem)
                    curr_recomp_id = node.recompute
                    if curr_recomp_id and last_node:
                        outputs = [t for t in last_node.outputs() if isinstance(t, IRTensor)]
                        bound_mem += sum(t.byte_size() for t in outputs)
                else:
                    recompute_mem[-1] += node_mem
                last_node = node
        return (non_recompute_mem + bound_mem) * inflights + max(recompute_mem)

    def peak_attr_mem(self, nodes: Tuple[IRFwOperation]) -> int:
        """Profile the weight memory cost of executing the sub-graph
        
        Returns:
            int: parameter memory in byte.
            int: buffer memory in byte.
        """
        param_mem, buffer_mem = 0, 0
        visited: Set[int] = set()
        for node in nodes:
            for t in node.inputs():
                if isinstance(t, IRTensor) and t.is_attr():
                    if t.tid in visited: continue
                    if t.is_param():
                        param_mem += t.byte_size()
                    else:
                        buffer_mem += t.byte_size()
                    visited.add(t)
        return param_mem, buffer_mem

    def attr_numel(self, nodes: Tuple[IRFwOperation]) -> int:
        """Get number of elements of attributes
        
        Returns:
            int: parameter count
            int: buffer count
        """
        param_cnt, buffer_cnt = 0, 0
        for node in nodes:
            for t in node.inputs():
                if isinstance(t, IRTensor) and t.is_attr():
                    if t.is_param():
                        param_cnt += t.nelement()
                    else:
                        buffer_cnt += t.nelement()
        return param_cnt, buffer_cnt

    def peak_mem(self, nodes: Tuple[IRFwOperation],
                 inflights: int = 1) -> int:
        """Profile the peak memory cost of executing the sub-graph
        
        Args:
            nodes (Tuple[IRFwOperation]): the sub-graph
            inflights (int): the number of in-flight sub-graphs

        Returns:
            int: peak memory in byte
        """
        act_mem = self.peak_activation_mem(nodes, inflights)
        param_mem, buffer_mem = self.peak_attr_mem(nodes)
        grad_mem = param_mem
        param_cnt, buffer_cnt = self.attr_numel(nodes)
        if self.optim_keep_fp32_states:
            opt_mem = self.optim_nstates * param_cnt * 4 
        else:
            opt_mem = self.optim_nstates * param_mem
        if self.optim_keep_fp32_params:
            opt_mem += param_cnt * 4
        # debug
        attr = round((param_mem + grad_mem + buffer_mem) / 1024 / 1024 / 1024, 2)
        opt = round(opt_mem / 1024 / 1024 / 1024, 2)
        act = round(act_mem / 1024 / 1024 / 1024, 2)
        print(f'memory: attr {attr} GB | opt {opt} GB | act {act} GB')
        return act_mem + param_mem + grad_mem + buffer_mem + opt_mem

    def __call__(self, nodes: Tuple[IRFwOperation],
                 splits: Optional[Tuple[Split]] = None,
                 inflights: int = 1):
        """Profile the cost of executing a sub-graph

        Note:
            optim_nstates can be set according to the following optimzier type:
                - Adam(W): 2
                - SGD: 1

        Args:
            nodes (Tuple[IRFwOperation]): the sub-graph
            split (Tupe[Tuple[int, int, int]] or None):
                the partition config for each node. Default None (no split).
            inflights (int): the number of in-flight sub-graphs

        Returns:
            latency float: latency in ms
            memory int: memory in bytes
        """
        splits = [None] * len(nodes) if splits is None else tuple(splits)

        # split
        snodes = []
        for node, split in zip(nodes, splits):
            if split is None:
                snodes.append(node)
            else:
                idx, dim, num = split
                algo = node.algorithms('dim')
                sub_node = algo.instantiate(idx=idx, dim=dim, num=num)[0]
                sub_node.recompute = node.recompute
                snodes.append(sub_node)

        # memory
        peak_mem = self.peak_mem(snodes, inflights)

        # latency
        latency = 0.
        trains = [n.mirror is not None for n in nodes]
        for snode, train in zip(snodes, trains):
            infer_span, infer_mem, train_span, train_mem = self.perf(snode)
            span = train_span if train else infer_span
            latency += span
        
        return latency, peak_mem

    def save(self):
        """Save the database to json file"""
        self.database.dump(self.cache_file, override=True)

    @staticmethod
    def register_rule(node_name: str, idx: int, dim: int, num: int):
        """Register a partition rule forced in profiling"""
        Estimator._rules[node_name] = (idx, dim, num)

    @staticmethod
    def register_skip_node(node_name: str):
        """Register node that will skip profiling"""
        Estimator._skip_node_names.add(node_name)
