#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import logging
from pathlib import Path
from typing import Any, List, Tuple, Callable, Union, Dict, Type, Optional

import nnscaler
from nnscaler.utils import fields
from nnscaler.graph.tracer.metadata import OpContext
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.cten import IRObject, IRCell, IRTensor, IR
from nnscaler.graph.parser.frame import Frame
from nnscaler.graph.parser.mapping import SignFx2Op
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.function.function import any_ir_object_satisfy
from nnscaler.graph.tracer import TensorMetadata, DICT_KEYS_TYPE, DICT_VALUES_TYPE, DICT_ITEMS_TYPE

import torch.fx

_logger = logging.getLogger(__name__)


# virtual signature for `self.<attribute>`
SELF_GETATTR_SIG = 'self_getattr'


class FxModuleParser:
    """
    torch.fx module parser
    """

    ATTR_CONTENT_FILE_STEM = 'fullmodel.pt'
    ATTR_CONTENT_FILE_0 = 'fullmodel.pt.0'
    ATTR_CONTENT_FILE_FORMAT = '{stem}.{idx}'
    ATTR_MAP_FILE = 'dist_param_map.pt'

    @staticmethod
    def parse(module: torch.fx.GraphModule,
              dummy_inputs: Dict[str, Any],
              attr_savedir='./',
              *,
              save_content: bool = True,
              constant_folding: bool = False
        ) -> Tuple[List[IRObject], List[IRFwOperation], List[IRObject]]:
        """Parse torch.fx module into cube IR

        The overall entry to parse a torch.fx graph module

        Args:
            module (torch.fx.GraphModule): the torch.fx module
            dummy_inputs (Dict[str, Any]): the dummy inputs to run the module
            attr_savedir (str): the directory to save the attribute content
            save_content (bool): whether to save the content of the module
            constant_folding (bool): whether to parse the module with constant folding

        Returns:
            inputs (List[IRObject]): the input IRObjects
            all_ir_nodes (List[IRFwOperation]): the IRFwOperation nodes
            outputs (List[IRObject]): the output IRObjects
        """
        # frame will save the outputs of all ops (including `placeholder` and `output`)
        # because some ir ops creators just return ops with empty ouputs
        # (Those ops creators include user registered function, all functions returning tensors and more)
        # We will connect the real op outputs (saved in frame) to all ir op outputs and inputs later.

        frame = Frame()
        frame.push_var()

        # shape propagation
        assert isinstance(dummy_inputs, dict), "Expected dummy inputs to parse module"
        output_nodes = [node for node in module.graph.nodes if node.op == 'output']
        # currently fx graph always has only one output
        # even if a tuple/list is returned, it is still just one output
        assert len(output_nodes) == 1, f"Expect only one output, but got {len(output_nodes)}"
        output_node = output_nodes[0]
        # currently output of fx graph satisfies the following:
        assert len(output_node.args) == 1 and len(output_node.kwargs) == 0

        # create IRObjects and IRTensors
        for node in module.graph.nodes:
            if node.op == 'placeholder':
                FxModuleParser.init_objects(node, module, frame, is_constant=False)
            else:
                FxModuleParser.init_objects(node, module, frame, is_constant=True)

            # note the output node will be reset later by `parse_prim_output_node`
            # with the help of `parse_complex`

            # if fx graph output (output_node.args[0]) is a nested structure
            # the nested structure will be kept in `parse_complex`

            # but if the node is the only output of the graph
            # and it is a tuple, we need to keep it a tuple
            # to make sure the IRGraph has the correct output number
            # see `IRGrpah.from_logic_graph`

            val = frame.get_var(node.name)
            if node == output_node.args[0] \
                and IR.is_object(val) and isinstance(val.value, tuple):
                tuple_val = tuple(IRObject(name=node.name, value=v, is_constant=val.is_constant) for v in val.value)
                frame.set_var(node.name, tuple_val)

        # get graph inputs
        placeholders = [n for n in module.graph.nodes if n.op == 'placeholder']
        inputs = [frame.get_var(n.name) for n in placeholders]
        # - if the graph inputs contain nested strcuture,
        #   it should be wrapped into an IRObject
        for idx, placeholder in enumerate(placeholders):
            if not isinstance(inputs[idx], IRObject):
                obj = IRObject(name=placeholder.name, value=inputs[idx], is_constant=False)
                inputs[idx] = obj
                frame.set_var(placeholder.name, obj)

        # parse graph nodes
        all_ir_nodes = []
        for node in module.graph.nodes:
            ir_nodes = FxModuleParser.parse_node(node, module, constant_folding, frame)
            all_ir_nodes += ir_nodes

        # get graph outputs
        outputs = [frame.get_var(node.name) for node in module.graph.nodes if node.op == 'output']
        # currently fx graph always has only one output
        # even if a tuple/list is returned, it is still just one output
        assert len(outputs) == 1, f"Expect only one output, but got {len(outputs)}"

        if save_content:
            attr_savedir = Path(attr_savedir)
            frame.save_attr_content(attr_savedir / FxModuleParser.ATTR_CONTENT_FILE_STEM)
            frame.save_attr_map(attr_savedir / FxModuleParser.ATTR_MAP_FILE)

        frame.pop_var()
        return inputs, all_ir_nodes, outputs

    @staticmethod
    def parse_node(node: torch.fx.Node, module, constant_folding: bool, frame: Frame) -> List[IRFwOperation]:
        """
        Parse the node and return the IRFwOperation nodes
        """
        if node.op == 'placeholder':
            return []
        if node.op == 'output':
            return FxModuleParser.parse_prim_output_node(node, module, frame)
        if node.op in ('call_function', 'call_method'):
            return FxModuleParser.parse_prim_function_method(node, module, constant_folding, frame)
        if node.op == 'get_attr':
            return FxModuleParser.parse_prim_get_attr_node(node, module, frame)
        if node.op == 'call_module':
            return FxModuleParser.parse_prim_module(node, module, frame)
        else:
            raise TypeError(f"Unknown node kind {node.op}")

    @staticmethod
    def init_objects(node: torch.fx.Node, module: torch.fx.GraphModule,
                     frame: Frame, is_constant: bool = True):
        assert isinstance(node, torch.fx.Node)

        assert hasattr(node, 'meta') and 'tensor_meta' in node.meta, f"Node {node} should have tensor_meta"
        meta = node.meta['tensor_meta']
        val = IR.new(node.name, meta,
            tensor_types=(TensorMetadata,),
            is_constant=is_constant
        )
        frame.add_var(node.name, val)

    @staticmethod
    def parse_complex(val: Any, frame: Frame) -> Any:
        """parse complex fx.Node into IRObject

        The val is usually from a node's input or output, can be fx.Node nested
        by tuple/list/dict type, or a fx.Node itself.

        Args:
            val (Any): fx.Node nested by tuple/list/dict
            frame (Frame): the frame to get the fx.Node

        Returns:
            the copied structure where the fx.Node is replaced by IRObjects/IRTensors
        """
        # to support more nested types, we can refer to the implementation of
        # https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py
        if isinstance(val, tuple):
            return tuple(FxModuleParser.parse_complex(t, frame) for t in val)
        if isinstance(val, list):
            return list(FxModuleParser.parse_complex(t, frame) for t in val)
        if isinstance(val, dict):
            return {key: FxModuleParser.parse_complex(val, frame) for key, val in val.items()}
        # TODO: Currently slice/DICT_VALUES_TYPE/DICT_ITEMS_TYPE cases are never found.
        # We need to find some examples to test them.
        if isinstance(val, slice):
            return slice(FxModuleParser.parse_complex(val.start, frame),
                         FxModuleParser.parse_complex(val.stop, frame),
                         FxModuleParser.parse_complex(val.step, frame))
        # because fx node cannot be a dict key, so skip DICT_KEYS_TYPE here
        if isinstance(val, DICT_VALUES_TYPE):
            return tuple(FxModuleParser.parse_complex(x, frame) for x in val)
        if isinstance(val, DICT_ITEMS_TYPE):
            return tuple((i, FxModuleParser.parse_complex(x, frame)) for i, x in val)
        if isinstance(val, torch.fx.Node):
            return frame.get_var(val.name)
        return val

    @staticmethod
    def fetch_attr(mod: torch.fx.GraphModule, target: str):
        target_atoms = target.split('.')
        attr_itr = mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @staticmethod
    def parse_prim_module(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        prim_module = FxModuleParser.fetch_attr(module, node.target)
        if prim_module.__class__.__module__.startswith('torch.nn.modules'):
            raise RuntimeError(f'{prim_module.__class__.__module__} can not be parsed as leaf nodes')
        else:
            raise RuntimeError(f'unknown module: {prim_module.__class__.__module__}')

    @staticmethod
    def parse_prim_function_method(node: torch.fx.Node, module: torch.fx.GraphModule, constant_folding: bool, frame: Frame) -> List[IRFwOperation]:
        """
        Convert `call_function`/`call_method` op to IRFwOperation.

        Args:
            node (torch.fx.Node): the node to be parsed
            module (torch.fx.GraphModule): the module containing the node
            constant_folding (bool): global setting of whether to fold the constant

        Returns:
            List[IRFwOperation]: the IRFwOperation nodes.
                The returned list can be empty if the node is folded,
                or contains exactly one node if the node is not folded.

        """
        # get signature
        fsig = FxModuleParser._get_qualified_name(node.target, node)
        # get inputs
        input_vals = FxModuleParser.parse_complex(list(node.args), frame)
        kwargs = FxModuleParser.parse_complex(node.kwargs, frame)

        # use context constant_folding if set
        # Please note constant_folding only controls the output of the op
        # For op inputs, we will not fold/unfold them
        # when we enter the code block with different constant folding setting
        # as a workaround,
        # you can use `nnscaler.runtime.function.fold_constant` to fold inputs if needed
        op_context: Optional[Dict[str, Any]] = node.meta.get('op_context')
        if op_context is not None and op_context.get(fields(OpContext).constant_folding) is not None:
            constant_folding = op_context[fields(OpContext).constant_folding]

        if SignFx2Op.exist(fsig):
            ir_node = SignFx2Op.map(fsig)(*input_vals, **kwargs)
        else:
            # FIXME: handle cases for IRObject in kwargs
            # case1: unknown torch operator
            if FxModuleParser._is_torch_autograd_op(node, frame, fsig):
                _logger.warning(f'Find unknown pytorch operation: {fsig}')
                fname = fsig.split('.')[-1] if '.' in fsig else fsig
                ir_node = IRFwOperation(fname, fsig, input_vals, 1, **kwargs)
            # case2: python runtime function
            else:
                _logger.warning(f'Set python runtime function: {fsig}')
                is_constant = True
                if any_ir_object_satisfy(input_vals, lambda a: not a.is_constant):
                    warning_msg = f'non register python runtime function {fsig} has a non constant input: {input_vals}, ' + \
                            'You can register it as a customized function using nnscaler.register_op to remove this warning'
                    _logger.warning(warning_msg)
                    is_constant = False
                output = frame.get_var(node.name)
                if not isinstance(output, IRObject):
                    # avoid nested IRObject
                    output = IRObject(name=node.name, value=output, is_constant=is_constant)
                ir_node = IRPyFunc(fsig, input_vals, [output], **kwargs)

        if not isinstance(ir_node, IRCell):
            # SignFx2Op may return object that is not IRCell but a value (concrete  or IRObject),
            # for example Add or GetItem.
            # As node is deleted, we must set concrete value or IRTensor/IRObject into framework.

            # TODO: check the value saved in frame should equal to the value returned by the op
            frame.set_var(node.name, ir_node)
            return []

        FxModuleParser._set_node_meta(node, ir_node)

        # step 1: align the node output with the value in frame

        # handle the case when the function has no output
        # Currently this can only happened for user registered functions
        # We need to assume the function has side effect, and keep it in the graph
        if not ir_node.outputs():
            # To avoid the case that the function is annotated no output
            # but its output is used in other nodes.
            # By removing from frame,
            # we can catch the case earlier
            frame.del_val(node.name)
            # if the function has no output, just return
            return [ir_node]

        vals = frame.get_var(node.name)
        if len(ir_node.outputs()) == 1:
            vals = [vals]
        elif IR.is_object(vals):
            # fix the case that multiple outputs are returned as a single IRObject
            # Because IR.new doesn't know the number of outputs
            # it will wrap the output as a single IRObject if no tensor is in the outputs
            # so we need to unwrap it here, and align the outputs with annotations.
            is_constant = vals.is_constant
            vals = vals.value
            if not isinstance(vals, (list, tuple)):
                raise RuntimeError(f'Expect list or tuple for multiple outputs, but got {type(vals)}')
            vals = type(vals)(IRObject(name=node.name, value=v, is_constant=is_constant) for v in vals)
            frame.set_var(node.name, vals)

        # verify the inferred shape are consistent with actual output
        if isinstance(ir_node, IRFwOperation):
            ir_node.verify_shape(vals)

        assert len(vals) == len(ir_node.outputs()), f'{vals}, {ir_node.outputs()}'
        contains_undefined_output = any(v is IRObject.missing for v in ir_node.outputs())
        for i in range(len(vals)):
            if isinstance(ir_node.output(i), IRTensor) or ir_node.output(i) is IRObject.missing:
                # 1. output tensors are not set in function.py
                # 2. IRObject output from some functions (registered functions/getattr) are not set
                # For above two cases, we need to set them with values from frame.
                ir_node.set_output(i, vals[i])

        # update frame with ir output
        # Please note when there is only one output, we will unwrap it from `ir_node.outputs()` here
        frame.set_var(
            node.name,
            type(vals)(ir_node.outputs()) if len(ir_node.outputs()) > 1 else ir_node.output(0)
        )

        # update the name of output tensors
        # Note assignment is not allowed in lambda
        # so we use a helper function to update the name
        def _update_name(x: IRObject):
            x.name = node.name
        IR.modify_objects_inplace(ir_node.outputs(), _update_name)

        # step 2: constant folding

        # rules:
        # 1. never fold our own functions defined in `nnscaler.runtime.function` module.
        #    currently only `ifexpr` will go here, and it will never be folded.
        # 2. never fold tensors.
        # 3. never fold non-constant IRObject
        # 4. never fold functions that contains undefined output
        #    TODO: This is for backward compatibility, Not sure why.
        #    In current implementation,
        #    only user registered functions and `getattr` will have undefined output.
        #    So I think the original intention is to avoid folding user registered functions.
        # 5. Only fold primitive types (int, float, bool, None, str, Ellipsis) and its complex types
        def _is_primitive_type(val):
            # we don't fold a list/tuple/dict with length larger than this
            # Just a quick filter, and may not work when val has multiple nested levels
            FOLD_MAX_LEN = 10
            if isinstance(val, (list, tuple)):
                return len(val) < FOLD_MAX_LEN and all(_is_primitive_type(v) for v in val)
            elif isinstance(val, dict):
                return len(val) < FOLD_MAX_LEN and all(_is_primitive_type(k) and _is_primitive_type(v) for k, v in val.items())
            # use a white list instead of a black list
            return isinstance(val, (int, float, bool, type(None), str, type(Ellipsis)))

        # Note when it is not IRObject as a whole, we will not fold it
        if constant_folding and len(ir_node.outputs()) == 1 \
            and isinstance(ir_node.output(0), IRObject) \
            and not isinstance(ir_node.output(0), IRTensor) \
            and not contains_undefined_output \
            and not ir_node.signature.startswith(nnscaler.runtime.function.__name__ + '.')\
            and ir_node.output(0).is_constant \
            and _is_primitive_type(ir_node.output(0).value):
            frame.set_var(node.name, ir_node.output(0).value)
            return []
        else:
            return [ir_node]

    @staticmethod
    def parse_prim_get_attr_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRFwOperation]:
        """
        There are two types of get_attr, one is `FxNodeKind.PrimGetAttr` which is dealt with in this function.
        The other is `FxNodeKind.PrimCallFunction ` (i.e., <built-in function getattr>)
        which is dealt with by parse_prim_function_method.

        The object of get_attr node is always the traced module or its sub modules.
        node.target is the attribute name of the object.
        """
        ir_nodes = []
        concrete_value = FxModuleParser.fetch_attr(module, node.target)
        if isinstance(concrete_value, torch.Tensor):
            assert isinstance(concrete_value, torch.Tensor), \
                f"GetAttrPrim: expect tensor but got {type(concrete_value)}"
            exist_tensor = frame.get_attr_var(concrete_value)
            # the case that the parameter is the first time used by getattr
            if not exist_tensor:
                tensor = frame.get_var(node.name)
                # set tensor name same with the name in original model
                tensor.name = node.target
                if tensor.requires_grad:
                    tensor.as_param()
                else:
                    direct_module = module
                    full_qualified_name = node.target.split('.')
                    for name in full_qualified_name[:-1]:  # last one is the attribute name
                        direct_module = getattr(direct_module, name)
                    persistent = full_qualified_name[-1] not in direct_module._non_persistent_buffers_set
                    tensor.as_buffer(persistent=persistent)
                frame.add_attr(tensor, concrete_value, node.target)
            # the case that the parameter is consumed multiple times and registered previously
            else:
                frame.set_var(node.name, exist_tensor)
        else:
            assert isinstance(node.target, str), f"GetAttrPrim: expect `node.target` to be str but got {type(node.target)}"
            # in sub modules, the target is full qualified name (for example `embeddings.dropout.training`)
            if node.target.split('.')[-1] == 'training':
                # Let's just support `self.training` and ignore all other cases for now
                output = IRObject(name=node.name, value=frame.get_var(node.name), is_constant=False)
                ir_node = IRPyFunc(SELF_GETATTR_SIG, ['training'], [output])
                FxModuleParser._set_node_meta(node, ir_node)
                frame.set_var(node.name, output)
                # never fold the IRPyFunc node
                ir_nodes.append(ir_node)
            else:
                frame.set_var(node.name, concrete_value)

        return ir_nodes

    @staticmethod
    def parse_prim_output_node(node: torch.fx.Node, module: torch.fx.GraphModule, frame: Frame) -> List[IRCell]:
        assert len(node.args) == 1 and len(node.kwargs) == 0
        output = FxModuleParser.parse_complex(node.args[0], frame)
        frame.set_var(node.name, output)
        return []

    @staticmethod
    def _set_node_meta(node: torch.fx.Node, ir_node: Union[IRCell, Any]):
        if not isinstance(ir_node, IRCell):
            return

        ir_node.op_context = node.meta.get('op_context')
        module_stack = node.meta.get('nn_module_stack')
        ir_node.module_stack = module_stack
        comment = str(node.meta.get('frame_record', ''))
        if comment:
            ir_node.comment = comment

    @staticmethod
    def _get_qualified_name(node_target: Union[str, Callable[..., Any]], node: torch.fx.Node = None) -> str:
        if isinstance(node_target, str):
            assert node is not None
            return FxModuleParser._get_qualified_name_of_call_method(node_target, node)
        else:
            return FxModuleParser._get_qualified_name_of_call_function(node_target)

    @staticmethod
    def _get_qualified_name_of_call_function(node_target: Callable[..., Any]) -> str:
        """
        The target field of call_function node must be an callable object.
        """
        # # things like getattr just appear in builtins
        # if getattr(builtins, func.__name__, None) is func:
        #     return func.__name__
        # TODO(yizhu1): find a general solution
        assert callable(node_target)
        name = node_target.__name__
        module = FxModuleParser._find_module_of_method(node_target)
        module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
        return f'{module}.{name}'

    @staticmethod
    def _get_qualified_name_of_call_method(node_target: str, node: torch.fx.Node) -> str:
        """
        The target field of call_method node must be a string.
        """
        if not isinstance(node_target, str):
            raise ValueError(f'node_target must be a string, but got {type(node_target)} with value {node_target}')
        # NOTE(nishang): seems that we don't need to guess the method sig?
        # for module, module_name in [(torch, 'torch'), (torch.Tensor, 'torch.Tensor')]:
        #     lib_func = getattr(module, node_target, None)
        #     if lib_func is not None and callable(lib_func):
        #         return f'{module_name}.{node_target}'

        assert len(node.args) > 0, 'Expect an object as the first argument of call_method'
        # example node.args[0].meta is {'type': <class 'dict'>}
        in_type = node.args[0].meta['type']
        assert node_target in in_type().__dir__(), f'node_target = {node_target}, in_type().__dir__() = {in_type().__dir__()}'
        # TODO: for the history issue (please see the comment out lines after NOTE),
        # we should forward the torch.Tensor.xxx to torch.xxx if xxx existed under torch,
        # because many torch.Tensor functions are not included in the mapping.py,
        # we should add torch.Tensor.xxx in mapping.py
        if issubclass(in_type, torch.Tensor) and getattr(torch, node_target, None) and callable(getattr(torch, node_target)):
            return f'torch.{node.target}'
        # here forward torch.nn.Parameter.xxx to torch.Tensor.xxx
        elif issubclass(in_type, torch.Tensor) and getattr(torch.Tensor, node_target, None) and callable(getattr(torch.Tensor, node_target)):
            return f'torch.Tensor.{node.target}'
        else:
            return f'{in_type.__module__}.{in_type.__name__}.{node_target}'

    @staticmethod
    def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
        if getattr(orig_method, '__name__', None) == 'apply' and isinstance(getattr(orig_method, '__self__', None), Type) \
            and issubclass(orig_method.__self__, torch.autograd.Function):
            # for torch.autograd.Function
            return f'{orig_method.__self__.__module__}.{orig_method.__self__.__name__}'

        name = orig_method.__name__
        module = orig_method.__module__
        # if hasattr(orig_method, '__qualname__') and isinstance(orig_method.__qualname__, str):
        #     # if there has '.' in '__qualname__', it means this function is in a nested structure,
        #     #
        #     # for example, it is a method / function in a class:
        #     # torch.nn.Linear.forward.__module__ = torch.nn
        #     # torch.nn.Linear.forward.__name__ = forward
        #     # torch.nn.Linear.forward.__qualname__ = Linear.forward
        #     #
        #     # And in fx.node qualified name creating rule, the module also should include the class name,
        #     # in this example, the returned module should be `torch.nn.Linear`.
        #     splited_names = orig_method.__qualname__.split('.')
        #     class_name, name = splited_names[:-1], splited_names[-1]
        #     module = '.'.join([module] + class_name)
        if module == 'torch.autograd.grad_mode' and name in ['__enter__', '__exit__']:
            return 'torch.autograd.grad_mode.no_grad'
        if module is not None:
            return module
        if hasattr(orig_method, '__qualname__')\
            and isinstance(orig_method.__qualname__, str) and orig_method.__qualname__.startswith('_VariableFunctionsClass.'):
            return 'torch._C._VariableFunctions'
        for guess in [torch, getattr(torch.nn, 'functional')]:
            if getattr(guess, name, None) is orig_method:
                return guess.__name__
        raise RuntimeError(f'cannot find module for {orig_method}')

    @staticmethod
    def _is_torch_autograd_op(node: torch.fx.Node, frame: Frame, signature: str) -> bool:
        """Check whether the node is of a pytorch autograd operation."""
        # note: some python operations like torch.Tensor.size() doesn't return
        # an IRTensor, thus cannot be considered as a pytorch autograd operator.
        return signature.startswith('torch.') and \
               isinstance(frame.get_var(node.name), IRFullTensor)
