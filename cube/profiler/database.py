# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Usage:
    python -m cube.profiler.database --export ./profile.dat.json
"""
from typing import Callable, Tuple, Union, Optional, Dict, NewType, List, Any
import torch
import time
import os
import json
import logging
import _operator

import cube
from cube.ir.cten import IRTensor, IRObject
from cube.ir.operator import IRFwOperation
from cube.graph.parser.register import CustomizedOps

_logger = logging.getLogger(__name__)

Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
NameOrFunc = Union[str, Callable]


_train_module_ref: torch.nn.Module = torch.nn.Module().train()
_eval_module_ref: torch.nn.Module = torch.nn.Module().eval()


class CompProfiler:

    @staticmethod
    def profile(func: Callable, shapes: Shapes, dtypes: DTypes,
                requires_grads: Tuple[bool], values: Tuple[Any],
                warmup_sec: float = 2, prof_times: int = 50,
                **kwargs) -> Tuple[float, float, int, Tuple[int]]:
        """
        Profile a function

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param shapes Tuple[Tuple[int]]: the shapes of each input tensor
        @param dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32
        @param warmup_sec float: warmup seconds
        @param prof_times int: profile times
        @param kwargs Dict: other keyword argument for func call.

        @return fw_span float: the time in milliseconds for forward time
        @return bw_span float: the time in milliseconds for backward time
        @return infer_mem int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        assert len(shapes) == len(dtypes), \
            f"func {func.__name__}: expected each shape has a corresponding dtype, but got {shapes} and {dtypes}"
        # create data
        assert dtypes is not None
        def gen_torch_tensors(shape, dtype, requires_grad):
            """Generate dummy input tenosrs"""
            constructor = torch.zeros if dtype in (torch.int64, torch.int32, torch.bool) else torch.rand
            return constructor(tuple(shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=requires_grad)

        tensors = tuple(
            gen_torch_tensors(shape, dtype, requires_grad) if isinstance(value, IRTensor) else value \
                for shape, dtype, requires_grad, value in zip(shapes, dtypes, requires_grads, values)
        )
        require_backward = any([t.requires_grad for t in tensors if hasattr(t, 'requires_grad')])
        # FIXME: reconsidering requires_grad
        if func.__name__ in ('type_as'):
            require_backward = False
        # repalce kwargs starting with 'self.xxx'
        train_kwargs, eval_kwargs = {}, {}
        for name, value in kwargs.items():
            if isinstance(value, str) and value.startswith('self.'):
                train_val = getattr(_train_module_ref, value[5:])
                eval_val = getattr(_eval_module_ref, value[5:])
            else:
                train_val = eval_val = value
            train_kwargs[name] = train_val
            eval_kwargs[name] = eval_val
        # run one sample
        outputs = func(*tensors, **train_kwargs)
        '''
        only profile IRDimops currently, which has at least one tensor output and
        may have non-tensor outputs (like list, tuple, dict, etc.). In additional,
        we assume that non-tensor outputs will not be used in backward.
        '''
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        outputs = tuple(filter(lambda x: torch.is_tensor(x) and x.requires_grad, outputs))
        assert all(torch.is_tensor(otensor) for otensor in outputs), \
            f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)

        def run_step(func, tensors, kwargs, backward: bool):
            outputs = func(*tensors, **kwargs)
            outputs = (outputs,) if torch.is_tensor(outputs) else outputs
            outputs = tuple(filter(lambda x: torch.is_tensor(x) and x.requires_grad, outputs))
            if backward:
                torch.autograd.backward(outputs, grads)
            return outputs

        # profile inference peak memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mtic = torch.cuda.max_memory_allocated()  # in bytes
        with torch.no_grad():
            run_step(func, tensors, eval_kwargs, backward=False)
        mtoc = torch.cuda.max_memory_allocated()  # in bytes
        infer_memory = mtoc - mtic

        train_mem_info = []
        train_mem2in_idx = []
        used_tensor = set()
        # ref torch/utils/checkpoint.py/_checkpoint_without_reentrant
        def pack_hook(x):
            nonlocal train_mem_info, used_tensor
            if x.storage().data_ptr() not in used_tensor:
                used_tensor.add(x.storage().data_ptr())
                byte_size = x.element_size()
                for dim in list(x.size()):
                    byte_size = byte_size * dim
                train_mem_info.append(byte_size)
                idx = -1
                for i, t in enumerate(tensors):
                    if not isinstance(t, torch.Tensor):
                        continue
                    if t.storage().data_ptr() == x.storage().data_ptr():
                        idx = i
                        break
                train_mem2in_idx.append(idx)
            return x
        
        def unpack_hook(x):
            return x

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            outs = run_step(func, tensors, train_kwargs, backward=require_backward)

        # warmup
        tic = time.time()
        while time.time() - tic < warmup_sec:
            run_step(func, tensors, train_kwargs, backward=require_backward)

        # profile forward only
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            with torch.no_grad():
                run_step(func, tensors, eval_kwargs, backward=False)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        fw_span = (toc - tic) / prof_times * 1000 # in milliseconds

        # profile forward + backward
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            run_step(func, tensors, train_kwargs, backward=require_backward)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        fwbw_span = (toc - tic) / prof_times * 1000 # in milliseconds
        bw_span = max(fwbw_span - fw_span, 0.0)

        return fw_span, bw_span, infer_memory, train_mem_info, train_mem2in_idx


class ProfileDataBase:

    def __init__(self, filename: Optional[str] = None) -> None:
        """!
        Create a database for profiling result
        """

        self._data: Dict[str, Dict[str, Tuple[float, float, int]]] = dict()
        if filename is not None:
            self.load(filename)

    @staticmethod
    def get_func(node: IRFwOperation) -> Tuple[Callable, Shapes, DTypes, Dict]:
        """
        Get function call and its arguments from a cude IRGraph node
        """
        assert isinstance(node, IRFwOperation), f"Only support profiling forward operation but got {type(node)}"

        if node.signature in CustomizedOps.kOpRuntime:
            fn = CustomizedOps.kOpRuntime[node.signature]
        else:
            fn = eval(node.signature)
        shapes, dtypes, requires_grads, values = [], [], [], []

        def extract_val(val: Union[IRObject, Any]) -> Any:
            if isinstance(val, IRObject):
                return extract_val(val.value)
            elif isinstance(val, tuple):
                return tuple([extract_val(v) for v in val])
            elif isinstance(val, dict):
                return {k: extract_val(v) for k, v in val.items()}
            elif isinstance(val, slice):
                return slice(extract_val(val.start), extract_val(val.stop), extract_val(val.step))
            else:
                return val

        for t in node.inputs():
            if isinstance(t, IRTensor):
                shapes.append(t.shape)
                dtypes.append(t.dtype)
                requires_grads.append(t.requires_grad)
                values.append(t)
            else:
                shapes.append(None)
                dtypes.append(None)
                requires_grads.append(None)
                values.append(extract_val(t))
        return fn, shapes, dtypes, requires_grads, values, extract_val(node.kwargs)

    def profile(self, node: IRFwOperation, device: Optional[int] = None, override: bool = False):
        """
        Profile a forward node in IRGraph on a specific device (default current device)
        
        @param node IRFwOperation: node of IRGraph
        @param device int: the device that the node will execute on

        @return in_mem_info Tuple[int]: byte sizes of input tensors
        @return param_mem_info Tuple[int]: byte sizes of param tensors
        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        @return residual_mem: ??
        """
        fn, shapes, dtypes, requires_grads, values, kwargs = ProfileDataBase.get_func(node)

        if not override and self.exist(node):
            return self.query(node)

        if isinstance(device, int):
            orig_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
        
        in_mem_info, param_mem_info = [], []
        residual_mem, input_count = 0, 0
        for t in node.inputs():
            if isinstance(t, IRTensor) and t.is_param():
                param_mem_info.append(t.byte_size())
            elif hasattr(t, 'byte_size'):
                input_count += 1
                if input_count == 1:
                    residual_mem += t.byte_size()
                in_mem_info.append(t.byte_size())
            else:
                _logger.warning(f'node {node}: skip input {t}')

        # run profiling
        try:
            fw_span, bw_span, infer_memory, train_mem_info, train_mem2in_idx = \
                CompProfiler.profile(fn, shapes, dtypes, requires_grads, values, **kwargs)
        except Exception:
            _logger.exception(f'fail to profile {node}, use default values')
            fw_span, bw_span, infer_memory, train_mem_info, train_mem2in_idx = 0, 0, 0, [], []
        # log to database
        key = self._serialize(node)
        self.insert(node.signature, key, in_mem_info, param_mem_info, fw_span, bw_span,\
            infer_memory, train_mem_info, residual_mem, train_mem2in_idx)
        _logger.info(
            f"profiled {node.signature} | shapes: {shapes} | dtypes: {dtypes} "
            f"=> in mem info: {in_mem_info} | param mem info: {param_mem_info} | fw: {round(fw_span, 2)} ms | "
            f"bw: {round(bw_span, 2)} ms | infer mem: {infer_memory} | train mem info: {train_mem_info} | idx: {train_mem2in_idx}")

        if isinstance(device, int):
            torch.cuda.set_device(orig_device)
        return tuple(in_mem_info), tuple(param_mem_info), fw_span, bw_span, infer_memory, \
            tuple(train_mem_info), residual_mem, tuple(train_mem2in_idx)

    def insert(self, name: str, key: str, in_mem_info: Tuple[int], param_mem_info: Tuple[int],
               fw_span: float, bw_span: float, infer_memory: int, train_mem_info: Tuple[int],
               residual_mem: int, train_mem2in_idx: Tuple[int]):
        """
        log the span of a function name with key

        @param name str: the function signature
        @param key str: the encoded shapes and dtypes of node inputs
        @param in_mem_info Tuple[int]: byte sizes of input tensors
        @param param_mem_info Tuple[int]: byte sizes of param tensors
        @param fw_span float: the forward span time in milliseconds
        @param bw_span float: the backward span time in milliseconds
        @param infer_memory int: the peak memory in bytes after inference of the function
        @param train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        assert isinstance(name, str) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        self._data[name][key] = (in_mem_info, param_mem_info, fw_span, bw_span, infer_memory, train_mem_info, residual_mem, train_mem2in_idx)

    def exist(self, node: IRFwOperation) -> bool:
        """
        Check if the node has the performance recorded in the database

        @param node IRFwOperation: forward operation

        @return exist bool: True if the performance is recorded, else False
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return False
        if key not in self._data[node.signature]:
            return False
        return True

    def query(self, node: IRFwOperation) -> Tuple[Tuple[int], Tuple[int], float, float, int, Tuple[int], int, Tuple[int]]:
        """!
        Get the performance number of a node in IRGraph

        @param node IRFwOperation: node in IRGraph

        @return in_mem_info Tuple[int]: byte sizes of input tensors
        @return param_mem_info Tuple[int]: byte sizes of param tensors
        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return None
        if key not in self._data[node.signature]:
            return None
        return self._data[node.signature][key]

    def query_func(self, signature, shapes, dtypes) -> Tuple[Tuple[int], Tuple[int], float, float, int, Tuple[int], int, Tuple[int]]:
        """
        Get performance number of given name (signature), shapes and dtypes
        
        @param signature str: function signature
        @param shapes Tuple[Tuple[int]]: the shape of each input tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return in_mem_info Tuple[int]: byte sizes of input tensors
        @return param_mem_info Tuple[int]: byte sizes of param tensors
        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_mem_info Tuple[int]: byte sizes of tensors saved for backward
        """
        key = self._serialize(shapes, dtypes)
        if signature not in self._data:
            return None
        if key not in self._data[signature]:
            return None
        return self._data[signature][key]

    def query_args(self, signature: str) -> Tuple[List[Shapes], List[DTypes]]:
        """
        Get the recorded shapes and dtypes of 
        """
        item_shapes, item_dtypes = [], []
        if signature not in self._data:
            return item_shapes, item_dtypes
        for shapes_dtypes_str in self._data[torch.signature].keys():
            shapes, dtypes = self._deserialize(shapes_dtypes_str)
            item_shapes.append(shapes)
            item_dtypes.append(dtypes)
        return item_shapes, item_dtypes

    def _serialize(self, node: IRFwOperation) -> str:
        """
        Serialize the shapes, dtypes and kwargs into a string

        e.g.,
            shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
        => (1024,)-(1024,1024) : torch.float32-torch.float32

        @param shapes Tuple[Tuple[int]]: the shape of each tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return key str: the serialized string
        """
        shapes, dtypes = [], []
        for t in node.inputs():
            if isinstance(t, IRTensor):
                shapes.append(t.shape)
                dtypes.append(t.dtype)
            # else:
            #     shapes.append(None)
            #     dtypes.append(type(t))
        shapes = '-'.join(str(tuple(shape)) if shape is not None else str(None) for shape in shapes)
        dtypes = '-'.join(str(dtype) for dtype in dtypes)
        return shapes + ' : ' + dtypes

    def _deserialize(self, key: str) -> ShapesDTypes:
        """
        De-serialize the key string to shapes and dtypes

        e.g., (1024,)-(1024,1024)=torch.float32-torch.float32
        =>  shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)

        @param key str: the serialized string
        @return shapes_and_dtypes ShapesDTypes: shapes and dtypes
        """
        shapes, dtypes = key.split(' : ')
        shapes = tuple(eval(shape) for shape in shapes.split('-'))
        dtypes = tuple(eval(dtype) for dtype in dtypes.split('-'))
        return shapes, dtypes

    def dump(self, file: str, override=False):
        """!
        dump the profiled data into json format

        @param file str: the file name
        @param override bool: True if the existed can be overrided else False
        """
        if os.path.exists(file):
            assert override, f"File {file} exists. Set override = True to force dump."
        with open(file, 'w') as f:
            json.dump(self._data, f)

        
    def dump_ops(self, file: str, override=False):
        if os.path.exists(file):
            assert override, f"File {file} exists. Set override = True to force dump."
        for signature in self._data.keys():
            file_n = os.path.join(file, signature +'.json')
            with open(file_n, 'w') as f:
                json.dump(self._data[signature], f, indent=2)   

    def dump_op(self, file: str, signature, override=False): 
        assert signature in self._data.keys(), f'this node not be profiled'
        file_n = os.path.join(file, signature +'.json')
        with open(file_n, 'w') as f:
            json.dump(self._data[signature], f, indent=2)

    def load(self, file: str):
        """!
        load the profiled data into data base. The original existed one will be
        overrided by the loaded data.

        @param file str: the file name
        """
        with open(file, 'r') as f:
            self._data = json.load(f)

    def load_ops(self, file: str):
        for filename in os.listdir(file):
            if filename.endswith('.json'):
                with open(os.path.join(file, filename)) as f:
                    signature = filename[:-len('.json')]
                    self._data[signature] = json.load(f)

    def __repr__(self) -> str:
        data = []
        for signature in self._data:
            for key in self._data[signature]:
                shapes, dtypes = self._deserialize(key)
                in_mem_info, param_mem_info, fw_span, bw_span, infer_mem, train_mem = self._data[signature][key]
                data.append(f'{signature}: shapes={shapes}, dtypes={dtypes}, in mem {in_mem_info} bytes, param mem {param_mem_info} bytes, fw span: {fw_span} ms, bw span: {bw_span} ms, infer mem {infer_mem} bytes, train mem {train_mem} bytes')
        data = '\n'.join(data)
        return data
