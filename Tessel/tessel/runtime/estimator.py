# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple, List, Dict, Callable, Optional, NewType, Set, Union
import os
import torch
import time
import json

from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.segment import IRSegment
from cube.graph.function import IRGraphAnchor
from cube.graph.function.dimops import IRDimops, DimAnno
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


def get_partition_space(node: IRDimops) -> List[Tuple[int, int]]:
    """
    Get partition space of an IRDimops node

    @param node IRDimops
    @return space List[Tuple[int, int, int]]: tuple of configs: (idx, dim)
    """
    if not isinstance(node, IRDimops):
        return []
    visited : Set[str] = set()
    configs = []
    eshapes = node.anno.inputs() + node.anno.outputs()
    for idx, eshape in enumerate(eshapes):
        if idx < len(node.inputs()):
            if not isinstance(node.input(idx), IRTensor): continue
        for dim, edim in enumerate(eshape.dims):
            for identifier, reduce in zip(edim.identifiers, edim.reduces):
                if identifier in visited: continue
                visited.add(identifier)
                if identifier == '1' or node.anno.getlen(identifier) == 1: continue
                if reduce == DimAnno.ReduceType.Freeze: break
                dimlen = node.anno.getlen(identifier)
                algo = node.algorithms('dim')
                num = 2
                while num < min(16, dimlen) + 1:
                    if dimlen % num != 0:
                        num *= 2
                        continue
                    if not algo.satisfy(idx=idx, dim=dim, num=num): break
                    configs.append((idx, dim, num))
                    num *= 2
                break
    return configs



class Estimator:
    """
    Estimator to measture the computation / memory cost of a subgraph
    """

    def __init__(self, cache='./profile_database.json'):

        self.cache_file = cache
        reload = cache if os.path.exists(cache) else None
        self.database = ProfileDataBase(reload)

    def profile(self, node: IRFwOperation) -> Tuple[float, int]:
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            return 0.0, 0, 0.0, 0
        train = True if node.mirror is not None else False
        trials = [None,] + get_partition_space(node)
        trials = Estimator.special_rules(node, trials)
        for config in trials:
            if config is None:
                num = 1
                infer_span, infer_mem, train_span, train_mem = self.database.profile(node, train)
            else:
                idx, dim, num = config
                print(f'> ... try node {node.name} with idx={idx}, dim={dim}, num={num}')
                sub_node = node.algorithms('dim').instantiate(idx=idx, dim=dim, num=num)[0]
                infer_span, infer_mem, train_span, train_mem = self.database.profile(sub_node, train)
                if isinstance(train_span, float): break
            if isinstance(train_span, float): break
        assert isinstance(train_span, float), f"Failed to profile: {node}"
        infer_span, infer_mem = infer_span * num, infer_mem * num
        train_span, train_mem = train_span * num, train_mem * num
        self.database.insert(node, infer_span, infer_mem, train_span, train_mem)
        return infer_span, infer_mem, train_span, train_mem


    def __call__(self, nodes_or_segment: Union[Tuple[IRFwOperation], IRSegment]):
        """Profile the computation cost of a subgraph

        Args:
            nodes_or_segment (Tuple[IRFwOperation] | IRSegment):

        Returns:
            float: latency in ms
            int: memory in bytes
        """
        if isinstance(nodes_or_segment, IRSegment):
            train = nodes_or_segment.mirror is not None
        else:
            train = any(n.mirror is not None for n in nodes_or_segment)
        nodes = nodes_or_segment.nodes() if isinstance(nodes_or_segment, IRSegment) else nodes_or_segment
        memory, latency = 0.0, 0.0
        for node in nodes:
            if self.database.exist(node):
                infer_span, infer_mem, train_span, train_mem = self.database.query(node)
            else:
                infer_span, infer_mem, train_span, train_mem = self.profile(node)
            if train:
                memory += train_mem
                latency += train_span
            else:
                memory = max(memory, infer_mem)
                latency += infer_span
        return latency, memory

    def save(self):
        self.database.dump(self.cache_file, override=True)

    def special_rules(node, trials):
        # if node.name == 'embedding':  # for GPT
        #     trials = [(1, 0, 4),]
        # if node.name == 'window_attn':  # for Swin
        #     trials = [(1, 0, 4),]
        return trials
