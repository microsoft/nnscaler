from typing import Generator, Iterable, List, Any, Optional, Tuple
import logging

from cube.ir.cten import IRCell, IRTensor, IRObject
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.ir.adapter import IRWeightReducer, IRAdapter
from cube.ir.adapter.prim import CommPrim

from cube.graph.segment import IRSegment

from cube.codegen.frontend_mapping import Sign2EmitRule

from cube.flags import CompileFlag

_logger = logging.getLogger(__name__)


class IRValue:
    """
    A wrapper of the tensor name (as a variable name).
    This is used to avoid the tensor name to be quoted in repr.
    repr('name') => "'name'"
    repr(IRValue('name')) => "name"
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


def _safe_repr_value(val: Any, prefix_attr: Optional[str] = None) -> Any:
    """
    Return repr-able value of a tensor or value.
    For tensor, return IRValue({prefix}{tensor.name}_{tensor.tid})
    For non-tensor, return as it is

    Args:
        val (Any): tensor or non-tensor value
        prefix_attr (str): prefix to the tensor name if the tensor is an attribute
    Returns:
        the val that can be repr safely
    """
    if isinstance(val, IRObject):
        tensor_name = val.name
        if '.' in tensor_name:
            tensor_name = tensor_name.split('.')[0]
        name = '_'.join([tensor_name, str(val.tid)])
        if prefix_attr is not None and val.is_attr():
            name = prefix_attr + name
        return IRValue(name)
    elif isinstance(val, slice):
        return slice(_safe_repr_value(val.start, prefix_attr), _safe_repr_value(val.stop, prefix_attr), _safe_repr_value(val.step, prefix_attr))
    elif isinstance(val, dict):
        return {_safe_repr_value(k, prefix_attr): _safe_repr_value(v, prefix_attr) for k, v in val.items()}
    elif isinstance(val, list):
        return [_safe_repr_value(v, prefix_attr) for v in val]
    elif isinstance(val, tuple):
        # TODO: support subclasses of tuple, like torch.Size?
        return tuple(_safe_repr_value(v, prefix_attr) for v in val)
    elif isinstance(val, (int, str, bool, float, type(None), bytes)): # only primitive type supported
        return val
    raise ValueError(f'Unsupported data type: {type(val)}')


class CodeEmission:
    """
    Basic emission
    """
    def node_name(self, node: IRCell) -> str:
        return f"{node.name}{node.cid}"

    def tensor_name(self, val: Any, prefix_attr: Optional[str] = None) -> str:
        """
        Return representation of a value or a tensor.
        For tensor, return the {prefix}{tensor.name}_{tensor.tid}
        For non-tensor, return its repr

        Args:
            val (Any): tensor or non-tensor value
            prefix_attr (Optional[str]): prefix to the tensor name if the tensor is an attribute
        Returns:
            representation of the val in str
        """
        return repr(_safe_repr_value(val, prefix_attr))

    def complex_name(self, val: Any, prefix_attr: Optional[str]=None) -> str:
        """
        Return the val name with complex data type over IRObject
        Currently support complex data type of Dict, List, Tuple, IRObject
        """
        modifier = lambda t: IRValue(self.tensor_name(t, prefix_attr))
        val = IRSegment.modify_objects_of_complex(val, modifier)
        return str(val)

    def tuple_name(self, tensors: List[Any],
                   skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        """
        Return the tupled tensor name.

        @param tensors List[Any]: list of any value
        @param skip_attr bool: whether to skip graph attribute in the tensors
        @param prefix_attr bool: whether to add a prefix for graph attribute

        @return name str: the tupled tensor name
        """
        names = []
        for t in tensors:
            if isinstance(t, IRTensor) and skip_attr and t.is_attr():
                continue
            names.append(self.tensor_name(t, prefix_attr))
        name = '(' + ', '.join(names + ['']) + ')'
        return name

    def return_name(self, tensors: List[Any],
                    skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        names = []
        for t in tensors:
            if isinstance(t, IRTensor) and skip_attr and t.is_attr():
                continue
            names.append(self.tensor_name(t, prefix_attr))
        names = '_' if len(names) == 0 else ', '.join(names)
        return names

    def return_name_complex(self, vals: List[Any],
                            skip_attr: bool = False, prefix_attr: Optional[str] = None) -> str:
        names = []
        for t in vals:
            if isinstance(t, IRObject) and skip_attr and t.is_attr():
                continue
            names.append(self.complex_name(t, prefix_attr))
        names = '_' if len(names) == 0 else ', '.join(names)
        return names

    def kwargs_name(self, **kwargs) -> str:
        """Get kwarg name"""
        names = []
        # FIXME make the str include `""`
        # for name, val in kwargs.items():
        #     if isinstance(val, str) and not val.startswith('self.'):
        #         kwargs[name] = '"' + val + '"'
        # turn object into name
        modifier = lambda t: IRValue(self.tensor_name(t))
        kwargs = IRSegment.modify_objects_of_complex(kwargs, modifier)
        for name, val in kwargs.items():
            names.append(f'{name}={val}')
        name = ', '.join(names)
        return name


class FuncEmission(CodeEmission):
    def __init__(self):
        super().__init__()
        self._emit_rules = Sign2EmitRule()

    def emit_dataloader(self, node: IRDataOperation) -> List[str]:
        outputs = self.return_name(node.outputs())
        return [f'{outputs} = next({self.tensor_name(node.input(0))})']

    def emit_fnode(self, node: IRFwOperation, prefix_attr: str = None) -> List[str]:
        """Emit forward node code

        The result will look like (the lines are split into `List[str]`)
        ```
        # comment if have
        tensor_3333 = torch.view(tensor_2222, [1,2,3,4,5])
        ```

        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        assert isinstance(node, IRFwOperation)
        codes = []
        # insert comment
        if node.comment is not None:
            codes.append(f'# {node.comment}')

        signature = node.signature
        # setup arg string
        inputs = [self.tensor_name(t, prefix_attr=prefix_attr) for t in node.inputs()]
        # setup kwarg string
        kwargs = dict(**node.kwargs)
        for name, val in kwargs.items():
            if isinstance(val, str) and not val.startswith('self.'):
                kwargs[name] = '"' + val + '"'
        # turn IRObject into name
        modifier = lambda t: IRValue(self.tensor_name(t))
        kwargs = IRSegment.modify_objects_of_complex(kwargs, modifier)

        emit_rule = self._emit_rules.map(signature)
        body = emit_rule(node, inputs, kwargs)

        if len(node.outputs()) == 0:
            code = body
        else:
            outputs = [self.tensor_name(t) for t in node.outputs()]
            outputs = ', '.join(outputs)
            code = f'{outputs} = {body}'
        codes.append(code)
        return codes

    def emit_adapter(self, node: IRAdapter, prefix_attr: Optional[str] = None,
                     async_op: bool = False) -> List[str]:
        """
        Emit the statment of the adapter call

        The resultant `List[str]` will be lines of the statements of the final
        Python method for the targeted Segment,
        without the method signature and the return statement.

        Args:
            node (IRAdapter)
            prefix_attr (str | None): prefix to the tensor name
            async_op (bool): whether to enable async communication
        """
        codes = []
        assert len(node.device) == 1, f"Expected adapter to be dispatched:\n{node.extra_repr()}"
        prims = [node] if node.differentiable and node.custom else [prim for prim in node.prims]

        if async_op:
            # note async_op can only be applied when primitives satisfy:
            #   1) non-collective primitives perform before collective primitives.
            #   2) collectives running on same nccl stream (i.e., same device group)
            non_colls = [p for p in prims if not isinstance(p, CommPrim)]
            colls = [p for p in prims if isinstance(p, CommPrim)]
            # check condition 1)
            if len(non_colls) > 1:
                if max(prims.index(p) for p in non_colls) + 1 != len(non_colls):
                    async_op = False
            # check condition 2)
            devices = [set(p.device) for p in colls]
            if len(colls) > 1 and not all(devs == devices[0] for devs in devices[1:]):
                async_op = False

        for prim in prims:
            if len(prim.inputs()) == 1:
                itensors = self.tensor_name(prim.inputs()[0], prefix_attr=prefix_attr)
            else:
                itensors = self.tuple_name(prim.inputs(), prefix_attr=prefix_attr)
            prim_kwargs = dict(prim.kwargs)
            if async_op and isinstance(prim, CommPrim):
                prim_kwargs['async_op'] = True
            kwargs = self.kwargs_name(**prim_kwargs)
            outputs = self.return_name(prim.outputs())
            code = f'{outputs} = {prim.signature}({itensors}, {kwargs})'
            codes.append(code)
        return codes

    def emit_reducer(self, node: IRWeightReducer) -> List[str]:
        """
        Emit the statment to invoke a reducer object.

        The fields storing intermediate codes that are populated by this method:
        -   NONE
        """
        reducer_name = f'self.wreducer{node._id}'
        code = f'{reducer_name}.sync_grads()'
        return [code]

    def emit_release(self, tensors: Iterable[IRTensor]) -> str:
        tnames : Generator = (self.tensor_name(t) for t in tensors)
        return 'del ' + ', '.join(tnames)

    def get_backward_callsite_io_tensors(self, bwop: IRCell) -> Tuple:
        """
        Get backward inputs and outputs
        ```
        (input_tensors, output_tensors, output_grads, input_grads)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~
        #inputs to 'backward'                         outputs of 'backward'
        ```

        @return input_tensors List[IRSubTensor]: forward input tensors (backward input)
        @return output_tensors List[IRSubTensor]: forward output tensors (backward output)
        @return output_grads List[IRSubTensor]: gradient of forward output tensors
            (backward input)
        @return input_grads List[IRSubTensor]: gradient of forward input tensors
            (backward output)
        """
        assert not bwop.isfw()
        fwop: IRCell = bwop.mirror

        grad2tensor = {}
        for t in fwop.inputs() + fwop.outputs():
            if isinstance(t, IRSubTensor) and t.grad is not None:
                grad2tensor[t.grad] = t

        input_grads = [t for t in bwop.outputs() if isinstance(t, IRSubTensor)]
        output_grads = [t for t in bwop.inputs() if isinstance(t, IRSubTensor)]
        input_tensors = [grad2tensor[g] for g in input_grads if g in grad2tensor]
        output_tensors = [grad2tensor[g] for g in output_grads if g in grad2tensor]

        return input_tensors, output_tensors, output_grads, input_grads
