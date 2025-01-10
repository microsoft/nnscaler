#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Usage:
    python -m nnscaler.profiler.database --export ./profile.dat.json
"""
from typing import Callable, Tuple, Union, Optional, Dict, NewType, List, Any
import torch
import time
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import _operator # required by eval()
import nnscaler  # required by eval()
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.cten import IRTensor, IRObject
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.parser.register import CustomizedOps

_logger = logging.getLogger(__name__)

Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
RequiresGrad = NewType('RequiresGrad', Tuple[bool])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
NameOrFunc = Union[str, Callable]

_train_module_ref: torch.nn.Module = torch.nn.Module().train()
_eval_module_ref: torch.nn.Module = torch.nn.Module().eval()

# when profiling fails, we use the long default value as a penalty
_FAIL_FW_SPAN = 1000 * 1000  # 1000 seconds


@dataclass
class ProfiledMetrics:
    """!
    The profiling data of a function
    """
    # the bytes of each input tensors (i.e., activation tensors)
    # excluding parameter and buffer tensors for `node`, no matter the activation
    # tensor requires gradient or not
    in_mem_info: Tuple[int]
    # the bytes of every parameter and buffer tensor of `node`
    param_mem_info: Tuple[int]
    buffer_mem_info: Tuple[int]
    # the forward span time in milliseconds
    fw_span: float
    # the backward span time in milliseconds
    bw_span: float
    # the peak memory in bytes during inference of `node`
    infer_memory: int
    # the bytes of each activation tensor that is saved for backward
    train_mem_info: Tuple[int]
    # the index of the tensor saved for backward in `node.inputs()` list
    train_mem2in_idx: Tuple[int]

    def __repr__(self) -> str:
        contents = dict()
        for key, value in self.__dict__.items():
            if key in ('in_mem_info', 'param_mem_info', 'buffer_mem_info', 'train_mem_info'):
                contents[key] = [f'{v / 1024 / 1024:.2f} MB' for v in value]
            elif key == 'infer_memory':
                contents[key] = f'{value / 1024 / 1024:.2f} MB'
            elif key in ('fw_span', 'bw_span'):
                contents[key] = f'{value:.2f} ms'
            else:
                contents[key] = value
        return str(contents)


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

    # TODO: this function should rewrite with pytree
    def extract_val(val: Union[IRObject, Any]) -> Any:
        if isinstance(val, IRObject):
            return extract_val(val.value)
        elif isinstance(val, tuple):
            return tuple([extract_val(v) for v in val])
        elif isinstance(val, list):
            return list([extract_val(v) for v in val])
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


def profile(node: IRFwOperation, func: Callable, shapes: Shapes, dtypes: DTypes,
            requires_grads: Tuple[bool], values: Tuple[Any],
            warmup_sec: float = 2, prof_times: int = 20, max_prof_sec: float = 20,
            **kwargs) -> Tuple[float, float, int, Tuple[int]]:
    """
    Profile a function

    Args:
        node IRFwOperation: the node in IRGraph
        func Callable: the callable function, e.g., torch.nn.functional.linear
        shapes Tuple[Tuple[int]]: the shapes of each input tensor
        dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32
        requires_grads Tuple[bool]: whether the input tensor requires gradient
        values Tuple[Any]: the values of the inputs that are not IRTensor
        warmup_sec float: warmup seconds
        prof_times int: number of execution for profiling an operator
        max_prof_sec float: max seconds for profiling an operator's forward or backward
        kwargs Dict: other keyword argument for func call.

    Returns:
        fw_span float: the time in milliseconds for forward time
        bw_span float: the time in milliseconds for backward time
        infer_mem int: the peak memory in bytes after inference of the function
        train_mem_info Tuple[int]: byte sizes of activation tensors saved for backward
    """
    assert len(shapes) == len(dtypes), \
        f"func {func.__name__}: expected each shape has a corresponding dtype, but got {shapes} and {dtypes}"
    # create data
    assert dtypes is not None
    def gen_torch_tensors(shape, dtype, requires_grad):
        """Generate dummy input tenosrs"""
        constructor = torch.zeros if dtype in (torch.int64, torch.int32, torch.bool) else torch.rand
        return constructor(tuple(shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=requires_grad)

    if CustomizedOps.kOpInputGen.get(node.signature, None) is not None:
        in_tensors = CustomizedOps.kOpInputGen[node.signature](node)
    else:
        in_tensors = tuple(
            gen_torch_tensors(shape, dtype, requires_grad) if isinstance(value, IRTensor) else value \
                for shape, dtype, requires_grad, value in zip(shapes, dtypes, requires_grads, values)
        )
    # add clone() to avoid error "RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation."
    tensors = tuple([t.clone() if torch.is_tensor(t) else t for t in in_tensors])
    total_input_size = sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))
    require_backward = any([t.requires_grad for t in tensors if hasattr(t, 'requires_grad')])
    # FIXME: reconsidering requires_grad
    # the __name__ of function with type of torch.ScriptFunction is None
    if hasattr(func, '__name__') and func.__name__ in ('type_as'):
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

    # check whether func is a in-place operation
    for t1, t2 in zip(in_tensors, tensors):
        if torch.is_tensor(t1) and not torch.equal(t1, t2):
            _logger.warning(f"{node}: in-place operation detected, the input tensor is modified, will not profile backward")
            require_backward = False

    # only profile IRDimops currently, which has at least one tensor output and
    # may have non-tensor outputs (like list, tuple, dict, etc.). In addition,
    # we assume that non-tensor outputs will not be used in backward.
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
    infer_memory = mtoc - mtic + total_input_size

    train_mem_info = []
    train_mem2in_idx = []
    used_tensor = set()
    # ref torch/utils/checkpoint.py/_checkpoint_without_reentrant
    def pack_hook(x):
        nonlocal train_mem_info, used_tensor
        if x.untyped_storage().data_ptr() not in used_tensor:
            used_tensor.add(x.untyped_storage().data_ptr())
            byte_size = x.element_size()
            for dim in list(x.size()):
                byte_size = byte_size * dim
            idx = -1
            is_attr = False
            for i, t in enumerate(tensors):
                if not isinstance(t, torch.Tensor):
                    continue
                if t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
                    if node.inputs()[i].is_attr():
                        is_attr = True
                    idx = i
                    break
            if not is_attr:
                train_mem_info.append(byte_size)
                train_mem2in_idx.append(idx)
        return x

    def unpack_hook(x):
        return x

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        outs = run_step(func, tensors, train_kwargs, backward=require_backward)

    # warmup
    warmup_cnt = 0
    tic = time.perf_counter()
    while time.perf_counter() - tic < warmup_sec:
        run_step(func, tensors, train_kwargs, backward=require_backward)
        torch.cuda.synchronize()
        warmup_cnt += 1
    toc = time.perf_counter()
    func_duration = (toc - tic) / warmup_cnt
    real_prof_times = max(1, min(prof_times, math.ceil(max_prof_sec / func_duration)))

    # profile forward only
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for _ in range(real_prof_times):
        with torch.no_grad():
            run_step(func, tensors, eval_kwargs, backward=False)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    fw_span = (toc - tic) / real_prof_times * 1000 # in milliseconds

    # profile forward + backward
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for _ in range(real_prof_times):
        run_step(func, tensors, train_kwargs, backward=require_backward)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    fwbw_span = (toc - tic) / real_prof_times * 1000 # in milliseconds
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

    def profile(self, node: IRFwOperation, override: bool = False) -> ProfiledMetrics:
        """
        Profile a forward node in IRGraph

        Args:
            node IRFwOperation: node of IRGraph
            override bool: True if the existed can be overrided else False

        Returns:
            profiled_metrics ProfiledMetrics: the profiling data
        """
        if not override and self.exist(node):
            return self.query(node)

        # run profiling
        try:
            in_mem_info, param_mem_info, buffer_mem_info, in_mem_idx = [], [], [], []
            fn, shapes, dtypes, requires_grads, values, kwargs = get_func(node)

            for idx, t in enumerate(node.inputs()):
                if isinstance(t, IRTensor) and t.is_param():
                    param_mem_info.append(t.byte_size())
                elif isinstance(t, IRTensor) and t.is_buffer():
                    buffer_mem_info.append(t.byte_size())
                elif hasattr(t, 'byte_size'):
                    in_mem_info.append(t.byte_size())
                    in_mem_idx.append(idx)
                else:
                    _logger.debug(f'node {node}: skip input {t}')
            fw_span, bw_span, infer_memory, train_mem_info, train_mem2in_idx = \
                profile(node, fn, shapes, dtypes, requires_grads, values, **kwargs)
        except Exception:
            _logger.exception(f'fail to profile {node}, use default values')
            fw_span, bw_span = _FAIL_FW_SPAN, 2 * _FAIL_FW_SPAN
            infer_memory = 0
            for t in node.outputs():
                if isinstance(t, IRTensor):
                    infer_memory += t.byte_size()
            # by default, we assume that all the input tensors are saved for backward
            train_mem_info = copy.deepcopy(in_mem_info)
            train_mem2in_idx = in_mem_idx

        profiled_metrics = ProfiledMetrics(in_mem_info, param_mem_info, buffer_mem_info,
                                           fw_span, bw_span, infer_memory,
                                           train_mem_info, train_mem2in_idx)
        return profiled_metrics

    def insert(self, name: str, key: str, profiled_metrics: ProfiledMetrics):
        """
        Log the profiling numbers of a function name with key

        Args:
            name str: the function signature
            key str: the encoded shapes and dtypes of node inputs
            profiled_metrics ProfiledMetrics: the profiling data
        """
        assert isinstance(name, str) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        self._data[name][key] = profiled_metrics

    def exist(self, node: IRFwOperation) -> bool:
        """
        Check if the node has the performance recorded in the database

        @param node IRFwOperation: forward operation

        @return exist bool: True if the performance is recorded, else False
        """
        key = self._serialize(node)
        return self.exist_serialized(node.signature, key)

    def exist_serialized(self, signature: str, key: str) -> bool:
        """
        Check if the node has the performance recorded in the database

        Args:
            signature str: the signature of the function
            key str: the serialized key

        Returns:
            exist bool: True if the performance is recorded, else False
        """
        if signature not in self._data:
            return False
        if key not in self._data[signature]:
            return False
        return True

    def query(self, node: IRFwOperation) -> Optional[ProfiledMetrics]:
        """!
        Get the performance number of a node in IRGraph

        Args:
            node IRFwOperation: node in IRGraph

        Returns:
            profiled_metrics ProfiledMetrics: the profiling data
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
            requires_grads: (True, False)
        => (1024,)-(1024,1024) : torch.float32-torch.float32 : True-False

        Args:
            node IRFwOperation: node in IRGraph

        Returns:
            key str: the serialized string
        """
        shapes, dtypes, requires_grads = [], [], []
        for t in node.inputs() + node.outputs():
            if isinstance(t, IRTensor):
                shapes.append(t.shape)
                dtypes.append(t.dtype)
                requires_grads.append(t.requires_grad)
            # else:
            #     shapes.append(None)
            #     dtypes.append(type(t))
        shapes = '-'.join(str(tuple(shape)) if shape is not None else str(None) for shape in shapes)
        dtypes = '-'.join(str(dtype) for dtype in dtypes)
        requires_grads = '-'.join(str(require_grad) for require_grad in requires_grads)
        return shapes + ' : ' + dtypes + ' : ' + requires_grads

    def _deserialize(self, key: str) -> Tuple[Shapes, DTypes, RequiresGrad]:
        """
        De-serialize the key string to shapes and dtypes

        e.g., (1024,)-(1024,1024) : torch.float32-torch.float32 : True-False
        =>  shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
            requires_grads: (True, False)

        Args:
            key str: the serialized string

        Returns:
            shapes Shapes: the shapes of each input tensor
            dtypes DTypes: the dtypes of each input tensor
            requires_grads RequiresGrad: whether the input tensor requires gradient
        """
        shapes, dtypes, requires_grads = key.split(' : ')
        shapes = tuple(eval(shape) for shape in shapes.split('-'))
        dtypes = tuple(eval(dtype) for dtype in dtypes.split('-'))
        requires_grads = tuple(eval(require_grad) for require_grad in requires_grads.split('-'))
        return shapes, dtypes, requires_grads

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

    def dump_ops(self, folder: str, override=False):
        """
        dump the profiled data into json format, each operator is saved in a separate file with the signature as the file name

        Args:
            folder str: the folder name
            override bool: True if the existed can be overrided else False
        """
        folder = Path(folder)
        for signature in self._data:
            fname = folder / (signature + '.json')
            if fname.exists():
                assert override, f"File {fname} exists. Set override = True to force dump."
            with open(fname, 'w') as f:
                to_dump = {key: asdict(value) for key, value in self._data[signature].items()}
                json.dump(to_dump, f, indent=2)

    def load(self, file: str):
        """!
        load the profiled data into data base. The original existed one will be
        overrided by the loaded data.

        @param file str: the file name
        """
        with open(file, 'r') as f:
            self._data = json.load(f)

    def load_ops(self, folder: str):
        """
        load the profiled data from json files in a folder. Each operator is saved in a separate file with the signature as the file name

        Args:
            folder str: the folder name
        """
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                with open(os.path.join(folder, filename)) as f:
                    signature = filename[:-len('.json')]
                    loaded_json = json.load(f)
                    self._data[signature] = {key: ProfiledMetrics(**value) for key, value in loaded_json.items()}

    def __repr__(self) -> str:
        data = []
        for signature in self._data:
            for key in self._data[signature]:
                shapes, dtypes, requires_grads = self._deserialize(key)
                pmetrics = self._data[signature][key]
                data.append(f'{signature}: shapes={shapes}, dtypes={dtypes}, requires_grads={requires_grads}, profiled numbers: {pmetrics}.')
        data = '\n'.join(data)
        return data
