#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
IRCell:
    a graph node component serving for different purpose,
    e.g., operator, device graph, graph

IRTensor:
    Tensor representation serving for edges to connect IRCells

The input of IRCell are IRTensors or any deterministic values (e.g., int).
If an IRTensor is the input of Cell, then Cell.device \in IRTensor.deivce

The output of IRCell are IRTensors or any deterministic values (e.g., int)
If an IRTensor is the output of Cell, then Cell.device == IRTensor.device
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Union, Optional, Any, Dict, Callable
from collections import OrderedDict
import copy
import torch

from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.dtype import DTypeInfo
from nnscaler.utils import _DICT_ITEMS_TYPE, _DICT_VALUES_TYPE


NestedVarOrStatic = Any


class IRCell:
    r"""
    IRCell serves as a general node for different purpose
    """

    def __init__(self,
                 name: str,
                 signature: str,
                 input_length: int,
                 output_length: int):
        """
        Create a node with name (variable name) and module type (module_name)

        Notes: setting input / kwarg / output IRObject will cause a copy-on-write
        behavior, where the IRObject will be copied (see `set_input` and `set_output`)
        to prevent users accidently updating it outside.

        To update the IRObject in input / kwarg / output, please use `find`, `input(s)`,
        and `output(s)` to get the real instance tensor in the IRCell.

        Args:
            name (str): the cell name
            signature (str): the cell function signature,
                e.g., torch.functional.nn.linear
            input_length (int): the number of inputs for the op
            output_length (int): the number of outputs for the op
        """
        # node info
        self._id: int = IDGenerator().gen_cell_id()
        self.name: str = name
        self.signature = signature

        self._device: Tuple[int] = ()

        # input tensors
        self._inputs: List[NestedVarOrStatic] = [None,] * input_length
        self._kwargs: Dict[str, NestedVarOrStatic] = {}
        # output tensors
        self._outputs: List[NestedVarOrStatic] = [None,] * output_length

        self._mirror: Optional[IRCell] = None
        # the comment for code generation
        self._comment: Optional[str] = None
        # the module stack that preserves the hierarchy information
        self._module_stack: Optional[OrderedDict[str, Any]] = None
        # the operation context information
        self._op_context: Optional[Dict[str, Any]] = None

    @property
    def cid(self) -> int:
        """
        Get cell id

        @return cid int: the cell id.
        """
        return self._id

    @property
    def device(self) -> Tuple[int]:
        return self._device

    @device.setter
    def device(self, device_id: Union[int, List[int]]):
        """
        Set the operation device.
        """
        if isinstance(device_id, int):
            device_id = (device_id,)
        if not all([isinstance(devid, int) for devid in device_id]):
            raise KeyError("Require device Union[int, List[int]]")
        self._device = tuple(device_id)

    def dispatch(self, device: int):
        """
        Instantiate this node to a specified device. Its mirror node will also
        be dispatched and paired with this node.

        For single operators, the mirror node will be reserved.
        For nodes that cover multiple devices, e.g., IRSegment and IRAdapter,
        the mirror node will be removed and require additional `make_pair` elsewhere.

        @param device int: device id
        @return dispatched_node IRCell: the node that only has one device placement.
        """
        assert len(self.device) == 1, \
            f"Require dispatch implementation for node type: {type(self)}"
        if isinstance(self.mirror, IRCell):
            assert len(self.mirror.device) == 1, \
                f"IRCell got unexpected mirro node that has multiple device placement.\n{self.mirror}"
        assert device in self.device, f"Fail to dispatch to device {device}. node: {self}"
        return self

    @property
    def mirror(self):
        """
        The mirror cell. E.g., forward op / backward op.
        """
        return self._mirror

    @mirror.setter
    def mirror(self, other):
        raise RuntimeError("Use IRCell.make_pair instead")

    @staticmethod
    def make_pair(cell1, cell2):
        if isinstance(cell1, IRCell):
            cell1._mirror = cell2
        elif cell1 is not None:
            raise TypeError("Expected cell1 to be IRCell or None")
        if isinstance(cell2, IRCell):
            cell2._mirror = cell1
        elif cell2 is not None:
            raise TypeError("Expected cell2 to be IRCell or None")

    def isfw(self) -> bool:
        """
        Return if the IRCell is executed fully in forward phase.
        This needs to be overrided by derived classes
        """
        return True

    @property
    def kwargs(self) -> Dict[str, NestedVarOrStatic]:
        return self._kwargs

    def input(self, index: int) -> NestedVarOrStatic:
        """Get the index-th input

        Args:
            index (int): index of the inputs

        Returns:
            NestedVarOrStatic: (nested) IRObject or any static value (int, bool, str, etc)
        """
        return self._inputs[index]

    # 'maxsize=None' set no limit on cache growth, but it's ok since we have no args
    @lru_cache(maxsize=None)
    def inputs(self) -> Tuple[NestedVarOrStatic]:
        """Get all input values

        Returns:
            Tuple[NestedVarOrStatic]
        """
        return tuple(self._inputs)

    @lru_cache(maxsize=None)
    def iobjs(self) -> Tuple[IRObject]:
        """
        Get all IRObject in the inputs and kwargs.

        The order follows the inputs order then kwargs order.

        Returns:
            Tuple[IRObject]: all IRObject in the inputs
        """
        return tuple(IRCell.get_objects_from_complex([self._inputs, self._kwargs]))

    def output(self, index: int) -> NestedVarOrStatic:
        """Get the index-th output value

        Args:
            index (int): index of the outputs

        Returns:
            NestedVarOrStatic: (nested) IRObject or any static value (int, bool, str, etc)
        """
        return self._outputs[index]

    @lru_cache(maxsize=None)
    def oobjs(self) -> Tuple[IRObject]:
        """
        Get all IRObjects in the outputs.

        Returns:
            Tuple[IRObject]: all IRObject in the outputs
        """
        return tuple(IRCell.get_objects_from_complex(self._outputs))

    # 'maxsize=None' set no limit on cache growth, but it's ok since we have no args
    @lru_cache(maxsize=None)
    def outputs(self) -> Tuple[NestedVarOrStatic]:
        """Get all output values

        Returns:
            Tuple[NestedVarOrStatic]
        """
        return tuple(self._outputs)

    def _copy_and_set_cell(self, x: IRObject) -> IRObject:
        x = copy.copy(x)
        x.cell = self
        return x

    def reset_inputs(self, length:int) -> None:
        """
        Resize the inputs list to the new length and reset all input items to None.
        """
        self._inputs = [None] * length
        self.inputs.cache_clear()
        self.iobjs.cache_clear()

    def set_input(self, index: int, val: NestedVarOrStatic) -> NestedVarOrStatic:
        """Set the index-th input

        Args:
            val (NestedVarOrStatic): (nested) IRObject or any deterministic value (int, bool, str, etc)

        Returns:
            NestedVarOrStatic: copied value
        """
        # recursive set cell
        val = IRCell.modify_objects_of_complex(val, self._copy_and_set_cell)

        self._inputs[index] = val
        self.inputs.cache_clear()
        self.iobjs.cache_clear()
        return val

    def reset_kwargs(self) -> None:
        """
        Clear all kwargs
        """
        self._kwargs = {}
        self.iobjs.cache_clear()

    def set_kwarg(self, name: str, val: NestedVarOrStatic) -> NestedVarOrStatic:
        """Set the kwarg with name

        Args:
            val (NestedVarOrStatic): (nested) IRObject or any deterministic value (int, bool, str, etc)

        Returns:
            NestedVarOrStatic: copied value
        """
        # TODO: is it possible that kwargs can be IRTensor?
        # But it is used in unit tests.
        # if isinstance(val, IRTensor):
        #     raise ValueError("IRTensor is not allowed to be a kwarg")

        # recursive set cell
        val = IRCell.modify_objects_of_complex(val, self._copy_and_set_cell)

        self._kwargs[name] = val
        self.iobjs.cache_clear()
        return val

    def reset_outputs(self, length:int) -> None:
        """
        Resize the outputs list to the new length and reset all output items to None.
        """
        self._outputs = [None] * length
        self.outputs.cache_clear()
        self.oobjs.cache_clear()

    def set_output(self, index: int, val: NestedVarOrStatic):
        """
        Set the node outputs[output_index] with the tensor

        Args:
            val (NestedVarOrStatic): (nested) IRObject or any deterministic value (int, bool, str, etc)

        Returns:
            NestedVarOrStatic: copied value
        """
        # recursive set cell
        val = IRCell.modify_objects_of_complex(val, self._copy_and_set_cell)

        self._outputs[index] = val
        self.outputs.cache_clear()
        self.oobjs.cache_clear()
        return val

    def replace_input(self, old: IRObject, new: IRObject):
        """Replace the old input (including kwargs) with the new input

        Args:
            old (IRObject): the old input
            new (IRObject): the new input
        """
        def replace(obj):
            val = copy.copy(new) if obj == old else obj
            val.cell = self
            return val

        self._inputs = IRCell.modify_objects_of_complex(self._inputs, replace)
        self._kwargs = IRCell.modify_objects_of_complex(self._kwargs, replace)
        self.inputs.cache_clear()
        self.iobjs.cache_clear()

    def replace_output(self, old: IRObject, new: IRObject):
        """Replace the old output with the new output

        Args:
            old (IRObject): the old output
            new (IRObject): the new output
        """
        def replace(obj):
            val = copy.copy(new) if obj == old else obj
            val.cell = self
            return val

        self._outputs = IRCell.modify_objects_of_complex(self._outputs, replace)
        self.outputs.cache_clear()
        self.oobjs.cache_clear()

    def replace(self, old: IRObject, new: IRObject):
        """Replace the old object with the new object in inputs, kwargs, and outputs

        Args:
            old (IRObject): the old object
            new (IRObject): the new object
        """
        self.replace_input(old, new)
        self.replace_output(old, new)

    def find(self, obj: IRObject) -> Tuple[IRObject]:
        """Find all the objects equal to `obj` in inputs, kwargs, or outputs

        Args:
            obj (IRObject): the object to find

        Returns:
            Tuple[IRObject]: all the objects equal to `obj`
        """
        outs = []
        IRCell.get_objects_from_complex(self._inputs, outs)
        IRCell.get_objects_from_complex(self._kwargs, outs)
        IRCell.get_objects_from_complex(self._outputs, outs)
        return tuple(out for out in outs if out == obj)

    @property
    def comment(self) -> Optional[str]:
        return self._comment

    @comment.setter
    def comment(self, info: str):
        """
        Tag an info to the cell
        """
        assert isinstance(info, str), "comment only allowed to be string"
        self._comment = info

    @property
    def module_stack(self) -> Optional[OrderedDict[str, Any]]:
        return self._module_stack

    @module_stack.setter
    def module_stack(self, stack: OrderedDict[str, Any]):
        """
        Set the module stack
        """
        self._module_stack = stack

    @property
    def op_context(self) -> Optional[Dict[str, Any]]:
        return self._op_context

    @op_context.setter
    def op_context(self, context: Optional[Dict[str, Any]]):
        self._op_context = context

    def __repr__(self) -> str:
        """
        Cell string presentation
        """
        ins = [t for t in self.inputs() if isinstance(t, IRTensor)]
        dscp = (f"Cell{self._id}-{self.device}(sign={self.signature}, "
                f"inputs={ins}, "
                f"outputs={self.outputs()})")
        return dscp

    @staticmethod
    def get_objects_from_complex(val: Any, _objects: List[IRObject] = None) -> List[IRObject]:
        """Get all IRObjects from a complex data structure

        Supported complex of types: List, Tuple, Dict, IRTensor, IRObject

        Args:
            val (Any): the complex data structure to be modified
            _objects (List[IRObject] | None):
                if provided, the objects will be appened into this

        @return _objects List[IRObject]: all IRObject
        """
        _objects = [] if _objects is None else _objects
        if isinstance(val, (tuple, list)):
            for item in val:
                IRCell.get_objects_from_complex(item, _objects)
        elif isinstance(val, dict):
            for key, value in val.items():
                IRCell.get_objects_from_complex(key, _objects)
                IRCell.get_objects_from_complex(value, _objects)
        elif isinstance(val, slice):
            IRCell.get_objects_from_complex([val.start, val.stop, val.step], _objects)
        elif isinstance(val, IRObject):
            _objects.append(val)
        return _objects

    @staticmethod
    def modify_objects_of_complex(val: Any, modifier: Callable) -> Any:
        """Return a complex data structure with modified IRObjects

        Supported complex of types: List, Tuple, Dict, IRTensor, IRObject

        Args:
            val (Any): the complex data structure to be modified
            modifier (Callable): a modifier that takes an IRObject and return a new one.

        Return:
            new_val (Any): complex data structure with modified IRObjects
        """
        rcall = IRCell.modify_objects_of_complex
        if isinstance(val, tuple):
            return tuple(rcall(item, modifier) for item in val)
        if isinstance(val, list):
            return list(rcall(item, modifier) for item in val)
        if isinstance(val, dict):
            return {rcall(key, modifier):rcall(value, modifier) for key, value in val.items()}
        if isinstance(val, slice):
            return slice(rcall(val.start, modifier), rcall(val.stop, modifier), rcall(val.step, modifier))
        if isinstance(val, IRObject):
            return modifier(val)
        return val


class IRObject:
    """
    IRObject serves as general data of IRGraph edge
    """

    def __init__(self, name: Optional[str] = None, tid: Optional[int] = None, value: Optional[None] = None, is_constant: bool = True):
        """
        Args:
            name (str): object name
            tid (int): object unique id
            val (Any): the value of this object
            is_constant (bool): if the value is a constant during the whole training / inference
                This flag is only used in constant_folding mode, to prevent the object from being folded.
                An IROject is considered not constant when either of two satisifies:
                    1. val is a tensor
                    2. val is model input, or is the result of a non-torch operation on another not constant IRObject
                Please note is_constant flag is only used in parser,
                so after parser, you can totally ignore this flag.
        """
        self._id: int = tid if isinstance(tid, int) else IDGenerator().gen_tensor_id()
        self.name: str = name if name else 'obj'
        self._cell: Optional[IRCell] = None
        self._is_attr: bool = False
        self._value: Optional[Any] = value
        self._is_constant: bool = is_constant

    def __hash__(self) -> int:
        return self._id

    def getstate_for_dump(self):
        """
        __getstate__ method for pickle dump

        @warning: dump an IRObject will disconnect the tensor to its cell
        """
        state = self.__dict__.copy()
        # this will decouple the interconnected object and cell during dump.
        state['_cell'] = None
        return state

    @property
    def tid(self) -> int:
        """Get object id"""
        return self._id

    @property
    def cell(self) -> IRCell:
        return self._cell

    @cell.setter
    def cell(self, val: Optional[IRCell]):
        assert isinstance(val, IRCell) or val is None, "Expected cell to be Optional[IRCell]"
        self._cell = val

    @property
    def device(self) -> Tuple[int]:
        if self._cell:
            return tuple(self._cell.device)
        else:
            return ()

    @device.setter
    def device(self, val: Union[int, List[int]]):
        raise RuntimeError(
            "IRObject placement is not allowed to set manually"
        )

    @property
    def parent(self):
        """Get parent"""
        return self

    @property
    def value(self) -> Any:
        """Get example value"""
        return self._value

    @property
    def is_constant(self) -> bool:
        return self._is_constant

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, IRObject):
            return False
        return self._id == obj.tid

    def __copy__(self):
        """Copy this object but remove the cell information"""
        return IRObject(self.name, self._id, self._value, self._is_constant)

    def as_attr(self):
        """
        Set the obj as graph attributes
        """
        self._is_attr = True
        return self

    def is_attr(self) -> bool:
        """!
        Check if the object is graph attribute.

        @return is_attr boolean: True if is graph attribute (buffer or parameter or gradient of parameter)
        """
        return self._is_attr

    def overlap(self, other: Any) -> bool:
        """!
        Check whether two object can be overlapped
        """
        if isinstance(other, IRObject):
            return other.tid == self._id
        else:
            return False

    @classmethod
    def from_complex(cls,
        name: str,
        data: Any,
        *,
        collection_types: Tuple = (tuple, list, dict),
        tensor_types: Tuple = (torch.Tensor,),
        is_constant: bool = False,
        requires_grad: Optional[bool] = None,
        tosub: bool = False,
    ) -> Any:
        """
        Convert complex data type of
            collection_types (tuple, list, dict)
            tensor_types (has shape/dtype/requires_grad)
        into intermediate representation object.

        Rule:
            1. All tensor-like objects will be converted into IRFullTensor
            2. For any complex types,
                a. if there is no tensor-like object, the whole object will be converted into IRObject
                b. if there is tensor-like object, all items will be converted into IRObject.
        Examples:
            [1, 2, torch.tensor(3)]
                -> [IRObject(1), IRObject(2), IRFullTensor(3)]
            {'a': [1, 2, 3], 'b': torch.tensor(2)}
                -> {'a': IRObject([1, 2, 3]), 'b': IRFullTensor(2)}
            {'a': [1, 2, torch.tensor(3)], 'b': 2}
                -> {'a': [IRObject(1), IRObject(2), IRFullTensor(3)], 'b': IRObject(2)}
        Args:
            name (str): the object name
            data (Any): the complex data structure to be converted
            collection_types (Tuple): the complex data types to be converted
            tensor_types (Tuple): the tensor data types to be converted
            tosub(bool): whether convert full tensor to sub-tensor
            requires_grad (Optional[bool]): the requires_grad flag for the tensor-like object
                None: will respect the original requires_grad flag
                True: will set requires_grad to True
                False: will set requires_grad to False
        """
        from nnscaler.ir.tensor import IRFullTensor

        collection_types = tuple(collection_types)
        tensor_types = tuple(tensor_types)
        supported_collection_types = (tuple, list, dict, _DICT_VALUES_TYPE, _DICT_ITEMS_TYPE)
        if any(t not in supported_collection_types for t in collection_types):
            raise ValueError(f"Only support converting complex data type of {supported_collection_types}")

        def _inner(obj) -> Tuple[Any, bool]:
            # second return is to know if there is any tensor-like object

            if isinstance(obj, tensor_types):
                if requires_grad is None:
                    rg = obj.requires_grad
                else:
                    # PyTorch only supports floating point and complex tensors for autograd.
                    # To align with PyTorch, we set requires_grad to False for other types.
                    rg = requires_grad and (obj.dtype.is_floating_point or obj.dtype.is_complex)

                tensor = IRFullTensor(
                    list(obj.shape),
                    name,
                    dtype=obj.dtype,
                    requires_grad=rg,
                )
                if tosub:
                    tensor = tensor.tosub()
                tensor._value = obj  # is required in SemanticModel.forward
                return tensor, True

            if isinstance(obj, collection_types):
                if isinstance(obj, tuple):
                    result = [_inner(item) for item in obj]
                    if not any(r[1] for r in result):
                        return IRObject(name, value=obj, is_constant=is_constant), False
                    else:
                        return tuple(r[0] for r in result), True
                if isinstance(obj, list):
                    result = [_inner(item) for item in obj]
                    if not any(r[1] for r in result):
                        return IRObject(name, value=obj, is_constant=is_constant), False
                    else:
                        return [r[0] for r in result], True
                if isinstance(obj, dict):
                    if not all(isinstance(key, str) for key in obj.keys()):
                        raise TypeError(f"only support dict type with str key, but got {obj.keys()}.")
                    result = {k: _inner(v) for k, v in obj.items()}
                    if not any(r[1] for r in result.values()):
                        return IRObject(name, value=obj, is_constant=is_constant), False
                    else:
                        return {k: r[0] for k, r in result.items()}, True
                if isinstance(obj, _DICT_VALUES_TYPE):
                    result = [_inner(item) for item in obj]
                    if not any(r[1] for r in result):
                        return IRObject(name, value=obj, is_constant=is_constant), False
                    else:
                        return {k: r[0] for k, r in enumerate(result)}.values(), True
                if isinstance(obj, _DICT_ITEMS_TYPE):
                    result = {k: _inner(v) for k, v in obj}
                    if not any(r[1] for r in result.values()):
                        return IRObject(name, value=obj, is_constant=is_constant), False
                    else:
                        return {k: r[0] for k, r in result.items()}.items(), True
            # slice will go here, as its start/stop/step are never tensor-like objects
            return IRObject(name, value=obj, is_constant=is_constant), False

        return _inner(data)[0]

    @classmethod
    def tosub_complex(cls, obj: Any) -> Any:
        """
        Convert complex data type of tensor-like object into sub-tensor

        Args:
            obj (Any): the complex data structure to be converted
        """
        from nnscaler.ir.tensor import IRFullTensor
        modifier = lambda t: t.tosub() if isinstance(t, IRFullTensor) else t
        return IRCell.modify_objects_of_complex(obj, modifier)

    @classmethod
    def try_unwrap(cls, x: Union[Any, 'IRObject']) -> Any:
        """
        Unwrap the IRObject to its original value if it is an IRObject
        otherwise, go recursively.

        Args:
            x (Any): the object to unwrap

        Returns:
            Any: the original value
        """
        if isinstance(x, IRObject) and not isinstance(x, IRTensor):
            return x.value
        elif isinstance(x, (list, tuple)):
            return type(x)(cls.try_unwrap(v) for v in x)
        elif isinstance(x, dict):
            return {k: cls.try_unwrap(v) for k, v in x.items()}
        elif isinstance(x, slice):
            return slice(cls.try_unwrap(x.start), cls.try_unwrap(x.stop), cls.try_unwrap(x.step))
        else:
            return x

    def __repr__(self):
        return f'Object({self.name}{self.tid}, val={self.value}, is_constant={self.is_constant})'


class IRTensor(IRObject):
    """
    IRTensor serves as tensor data of IRGraph edge

    Note by setting IRTensor name to "None" indicates this tensor holds nothing
    and will be translated to None in code generation.

    Note scalar tensors will always be converted to 1-d tensors.
    So all further operations could ignore the scalar tensor case.
    You can get the original shape with `origin_shape` property.
    """

    def __init__(self, shape=None, name='tensor', dtype=None, tid=None, *,
        is_attr=False, is_grad=False, requires_grad=False, persistent=False
    ):
        super().__init__(name, tid, is_constant=False)
        self._is_scalar_tensor: bool = True
        self._shape: Tuple[int] = ()
        self._dtype: Optional[torch.dtype] = None
        # tensor gradient
        self._is_grad: bool = False
        self._requires_grad: bool = False
        # _persistent is a buffer only field, but in inference mode all params will be post-processed to buffers,
        # so set _persistent True in as_param() for register these params to persistent buffers.
        self._persistent: bool = False
        self._update(
            shape=shape if shape is not None else (),
            name=name,
            dtype=dtype,
            is_attr=is_attr,
            is_grad=is_grad,
            requires_grad=requires_grad,
            persistent=persistent,
        )
        self._cell: Optional[IRCell] = None

    def _update(
            self,
            shape=None,
            name=None,
            dtype=None,
            is_attr=None,
            is_grad=None,
            requires_grad=None,
            persistent=None,
    ):
        """
        Set tensor metadata
        """
        if shape is not None:
            self._is_scalar_tensor = not shape
            # will always convert scalar tensor to 1-d tensor
            self._shape: Tuple[int] = (1,) if not shape else tuple(shape)
        if name is not None or self.name is None:
            self.name = name
        if dtype is not None:
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    "Only support setting IRTensor with dtype of torch.dtype"
                )
            self._dtype = dtype
        if is_attr is not None:
            self._is_attr = is_attr
        if is_grad is not None:
            self._is_grad = is_grad
        if requires_grad is not None:
            self._requires_grad = requires_grad
        if persistent is not None:
            self._persistent = persistent

        return self

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Tensor data type"""
        return self._dtype

    def is_param(self) -> bool:
        """!
        Check if the tensor is parameter

        @return is_param boolean: True if is parameter.
        """
        return not self._is_grad and self._is_attr and self._requires_grad

    def is_buffer(self) -> bool:
        """!
        Check if the tensor is buffer.

        @return is_buffer boolean: True if is buffer.
        """
        return not self._is_grad and self._is_attr and not self._requires_grad

    def is_persistent(self) -> bool:
        """!
        Check if the tensor is persistent buffer.

        @return is_persistent boolean: True if is persistent.
        """
        return self.is_buffer() and self._persistent

    def is_grad(self) -> bool:
        """!
        Check if the tensor is gradient

        @return is_grad boolean: True if is gradient
        """
        return self._is_grad

    def is_scalar_tensor(self) -> bool:
        """!
        Check if the tensor is scalar tensor

        @return is_scalar_tensor boolean: True if is scalar tensor
        """
        return self._is_scalar_tensor

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def __copy__(self):
        """
        Copy the tensor that will have the exactly same id
        except the empty attached cell

        Returns:
            tensor
        """
        tensor = IRTensor(self._shape, self.name, tid=self._id)
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor.cell = None
        return tensor

    @property
    def origin_shape(self) -> Tuple[int]:
        """
        Get the original shape of the tensor
        (Because self.shape will convert scalar tensor to 1-dim tensor)
        """
        return self.shape if not self.is_scalar_tensor() else ()

    @property
    def shape(self) -> Tuple[int]:
        # NOTE: here return a tuple but not a real torch.Size obj may have risk, here is an example:
        # (torch.Size + tuple -> torch.Size) will change to (tuple + tuple -> tuple), is ok.
        # (torch.Size + list -> torch.Size) will change to (tuple + list -> error), is wrong.
        return self._shape

    def nelement(self) -> int:
        """
        Get total number of element in the tensor.
        """
        if self.shape is None:
            raise RuntimeError("Tensor shape is not set")
        cnt = 1
        for num in self.shape:
            cnt *= num
        return cnt

    def byte_size(self) -> int:
        return self.nelement() * DTypeInfo.get_byte_size(self.dtype)

    def backward(self) -> None:
        """
        Autograd backward on the tensor, which is used in @compile

        The backward will apply on the program graph

        @return None
        """
        from nnscaler.program import Program, is_global_graph_enabled
        assert is_global_graph_enabled(), "Require global graph enabled to call loss.backward()"
        graph = Program().get_graph()
        return graph.backward(self)

    def __repr__(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp
