# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Known constraints:
    - not support for dynamic control flow
    - not support for dynamic shape
"""

import torch
import enum
import re
import logging
from typing import Any, List, Tuple, Callable, Dict, Set
import inspect
from pathlib import Path

from cube.ir.cten import IRObject, IRTensor
from cube.ir.dtype import DTypeInfo
from cube.ir.operator import IRFwOperation
from cube.graph.function.pyfunc import IRPyFunc
from cube.ir.tensor import IRFullTensor
from cube.graph.parser.frame import Frame
from .mapping import Sign2Op

_logger = logging.getLogger(__name__)

_refmodule = torch.nn.Module()


class ErasedDevice:
    pass


class ScriptNodeKind(enum.Enum):
    PrimGetAttr = 1
    PrimCallMethod = 2
    PrimCallFunction = 3  # -> the parser may end here
    PrimConstant = 4
    AtenOp = 5            # -> the parser may end here
    PrimIf = 6            # dynamic
    PrimListConstruct = 7
    PrimListUnpack = 8
    PrimTupleUnpack = 9
    PrimPythonOp = 10
    PrimDevice = 11       # erased
    PrimLoop = 12
    PrimSetAttr = 13


class ScriptModuleParser:

    ATTR_CONTENT_FILE = 'fullmodel.pt'
    ATTR_MAP_FILE = 'dist_param_map.pt'

    @staticmethod
    def parse_module(module: torch.jit.ScriptModule,
                     dummy_input: Dict[str, torch.Tensor],
                     attr_savedir='./',
                     *,
                     save_content: bool = True) \
        -> Tuple[List[IRFullTensor], List[IRFwOperation], List[IRFullTensor]]:
        """
        The overall entry to parse a torchscript graph module
        """
        nparams = sum(p.numel() for p in module.parameters())
        nbuffers = sum(b.numel() for b in module.buffers())
        _logger.info(f'parsing script model of {nparams}(+{nbuffers}) parameters (+buffers)')

        frame = Frame()
        frame.push_var()

        input_shapes, input_dtypes = [], []
        for t in dummy_input.values():
            if not isinstance(t, torch.Tensor):
                raise NotImplemented("Only support model with torch.Tensor as input")
            input_shapes.append(tuple(t.size()))
            input_dtypes.append(t.dtype)

        inputs = list(module.graph.inputs())[1:]
        if input_shapes is not None and len(input_shapes) != len(inputs):
            raise RuntimeError(f"Module {module.original_name} input shape mismatch (got {len(input_shapes)} != {len(inputs)})")

        # handle graph input -- Assuming all the inputs are tensors
        for idx, input in enumerate(inputs):
            if isinstance(input.type(), torch._C.TensorType):
                shape = input_shapes[idx]
                dtype = input_dtypes[idx]
                val = IRFullTensor(shape=shape, requires_grad=False, dtype=dtype, name=input.debugName())
            else:
                val = IRObject(name=input.debugName())
            frame.add_var(input.debugName(), val, graph_arg=idx)
        input_val = [frame.get_var(input.debugName()) for input in inputs]

        # handle graph parameters and buffers
        attr_data: Set[int] = set()
        for name, param in module.named_parameters():
            if id(param) in attr_data:
                continue
            attr_data.add(id(param))

            ir_tensor = IRFullTensor(
                shape=tuple(param.shape), name=name.replace('.', '_'),
                requires_grad=param.requires_grad, dtype=param.dtype
            )
            ir_tensor.as_param()
            frame.add_attr(ir_tensor, param, name)

        for name, buff in module.named_buffers():
            if id(buff) in attr_data:
                continue
            attr_data.add(id(buff))

            ir_tensor = IRFullTensor(
                shape=tuple(buff.shape), name=name.replace('.', '_'),
                requires_grad=buff.requires_grad, dtype=buff.dtype
            )
            ir_tensor.as_buffer()
            frame.add_attr(ir_tensor, buff, name)

        # handle nodes
        all_ir_nodes: List[IRFwOperation] = list()
        for node in module.graph.nodes():
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            for ir_node in ir_nodes:
                ScriptModuleParser.setup_node(ir_node)
            all_ir_nodes += ir_nodes

        # handle outputs
        output_var_name = [output.debugName() for output in module.graph.outputs()]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]
        
        # flatten output_val
        outputs = list()
        for val in output_val:
            if isinstance(val, list):
                outputs += val
            else:
                outputs.append(val)
        output_val = outputs

        frame.pop_var()
        if save_content:
            attr_savedir = Path(attr_savedir)
            frame.save_attr_content(attr_savedir / ScriptModuleParser.ATTR_CONTENT_FILE)
            # omit this
            # frame.save_attr_map(attr_savedir / ScriptModuleParser.ATTR_MAP_FILE)
        return input_val, all_ir_nodes, output_val

    @staticmethod
    def parse_module_method(module, method: torch._C.ScriptMethod, frame: Frame):
        """
        Parse module method
        """
        frame.push_var()

        input_var_name = [input.debugName() for input in method.graph.inputs()]
        kDefaultType = torch.get_default_dtype()

        for index, var_name in enumerate(input_var_name[1:]): # omit self
            frame.add_var(var_name, IRFullTensor(name=var_name, requires_grad=False, dtype=kDefaultType), graph_arg=index)

        input_val = [frame.get_var(var_name) for var_name in input_var_name[1:]]

        all_ir_nodes: List[IRFwOperation] = list()
        for node in method.graph.nodes():
            ir_nodes = ScriptModuleParser.parse_node(node, module, frame)
            for ir_node in ir_nodes:
                ScriptModuleParser.setup_node(ir_node)
            all_ir_nodes += ir_nodes

        # handle graph output
        output_var_name = [output.debugName() for output in method.graph.outputs()]
        output_val = [frame.get_var(var_name) for var_name in output_var_name]

        frame.pop_var()
        return input_val, all_ir_nodes, output_val

    @staticmethod
    def ntype(node: torch._C.Node):
        if node.kind() == 'prim::GetAttr':
            return ScriptNodeKind.PrimGetAttr
        if node.kind() == 'prim::SetAttr':
            return ScriptNodeKind.PrimSetAttr
        if node.kind() == 'prim::CallMethod':
            return ScriptNodeKind.PrimCallMethod
        if node.kind() == 'prim::CallFunction': # the op call
            return ScriptNodeKind.PrimCallFunction
        if node.kind() == 'prim::Constant':
            return ScriptNodeKind.PrimConstant
        if node.kind().startswith('aten::'):
            return ScriptNodeKind.AtenOp
        if node.kind() == 'prim::If':
            return ScriptNodeKind.PrimIf
        if node.kind() == 'prim::Loop':
            return ScriptNodeKind.PrimLoop
        if node.kind() == 'prim::ListConstruct':
            return ScriptNodeKind.PrimListConstruct
        if node.kind() == 'prim::TupleConstruct':
            return ScriptNodeKind.PrimListConstruct
        if node.kind() == 'prim::ListUnpack':
            return ScriptNodeKind.PrimListUnpack
        if node.kind() == 'prim::TupleUnpack':
            return ScriptNodeKind.PrimListUnpack
        if node.kind() == 'prim::PythonOp':
            return ScriptNodeKind.PrimPythonOp
        if node.kind() == 'prim::device':
            return ScriptNodeKind.PrimDevice
        raise RuntimeError(f"Unkown node kind {node.kind()} from torchscript module")

    @staticmethod
    def parse_node(node: torch._C.Node, module, frame: Frame) -> List[IRFwOperation]:
        # print("### parse_node {}".format(node))
        """
        Parse the node and return the IRFwOperation nodes
        """
        node_type = ScriptModuleParser.ntype(node)
        if node_type == ScriptNodeKind.PrimCallFunction:
            nodes = ScriptModuleParser.parse_prim_function_node(node, module, frame)
        elif node_type == ScriptNodeKind.AtenOp:
            nodes = ScriptModuleParser.parse_aten_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimCallMethod:
            nodes = ScriptModuleParser.parse_prim_method_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimGetAttr:
            nodes = ScriptModuleParser.parse_prim_attr_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimSetAttr:
            nodes = ScriptModuleParser.parse_prim_setattr_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimConstant:
            nodes = ScriptModuleParser.parse_prim_constant_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimListConstruct:
            nodes = ScriptModuleParser.parse_prim_list_construct_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimListUnpack:
            nodes = ScriptModuleParser.parse_prim_list_unpack_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimPythonOp:
            nodes = ScriptModuleParser.parse_prim_python_op_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimIf:
            nodes = ScriptModuleParser.parse_prim_if_node(node, module, frame)
        elif node_type == ScriptNodeKind.PrimLoop:
            nodes = ScriptModuleParser.parse_prim_loop_node(node, module, frame)
        # TODO bother assigning all ignored prim functions new NodeKinds?
        elif node_type == ScriptNodeKind.PrimDevice:
            nodes = ScriptModuleParser.parse_value_erased_node(node, module, frame, [ErasedDevice()])
        else:
            raise NotImplementedError(f"Un-supported node type {node_type}")
        return nodes

    @staticmethod
    def setup_node(node: IRFwOperation):
        """Setup node output shape, dtype and requires_grad"""
        if isinstance(node, IRFwOperation):
            # setup shape
            node.infer_shape()
            # setup requires grad
            dtypes = []
            requires_grad = False
            for t in node.inputs():
                if isinstance(t, IRTensor) and t.requires_grad:
                    dtypes.append(t.dtype)
                    requires_grad = True
            dtypes = [torch.get_default_dtype()] if len(dtypes) == 0 else dtypes
            dtype = DTypeInfo.promote(dtypes)
            for output in node.outputs():
                if isinstance(output, IRTensor):
                    output.parent.dtype = dtype
                    output.parent.requires_grad = requires_grad

    @staticmethod
    def parse_prim_function_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        parse node like:
            Tensor = prim::CallFunction(%5, %input.1, %3, %4)
            %5 : Function = prim::Constant[name="linear"]()
            %12 : (Tensor, Tensor) = prim::CallFunction(%5, %x1.1, %x2.1)
        """
        inputs = [input for input in node.inputs()]

        # get signature
        fnode = node.inputsAt(0).node()
        if not ScriptModuleParser.ntype(fnode) == ScriptNodeKind.PrimConstant:
            raise RuntimeError(f"Found unexpected function call node: {fnode}")
        fsig = frame.get_var(inputs[0].debugName())

        # get inputs
        input_vals = list()
        for index, input in enumerate(inputs[1:]):
            var_name = input.debugName()
            val = frame.get_var(var_name)
            input_vals.append(val)

        # map to IR operator
        ir_node = Sign2Op.map(fsig)(*input_vals)
        ScriptModuleParser.setup_node(ir_node)
        
        # push output in the frame
        # help: >>> a = torch._C.TupleType([torch._C.TensorType.getInferred()])
        #     : >>> dir(a)
        #     : >>> a.elements()  # [TensorType, TensorType]
        cnt = 0
        for output in node.outputs():
            if isinstance(output.type(), torch._C.TupleType):
                tuplen = len(output.type().elements())
                ir_output = [ir_node.output(idx) for idx in range(cnt, cnt+tuplen)]
                cnt += tuplen
            else:
                ir_output = ir_node.output(cnt)
                cnt += 1
            frame.add_var(output.debugName(), ir_output)

        if cnt != len(ir_node.outputs()):
            raise RuntimeError(
                f"Parse fail: {fsig} has {cnt} outputs != pre-defined {len(ir_node.outputs())}"
            )

        return [ir_node]

    @staticmethod
    def parse_aten_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse script module node like:
            %13 : Tensor = aten::gt(%output1.1, %output2.1)
        """
        fsig = node.kind()
        fsig = re.sub('aten::', 'torch.', fsig)
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        # handle inputs:
        input_val = [frame.get_var(input.debugName()) for input in inputs]

        # special handling on aten::size(tensor: tensor, dim: int)
        if fsig == 'torch.size':
            if len(inputs) == 2:
                tensor, dim = input_val
                output: int = tensor.shape[dim]
            else:
                tensor = input_val[0]
                output: List[int] = list(tensor.shape)
            frame.add_var(outputs[0].debugName(), output)
            return []

        # aten::__getitem__.t(t[](a) list, int idx) -> t(*)"
        # REMARK List-type only. '__getitem__' cannot serve as accessor to tensor element.
        elif fsig == 'torch.__getitem__':
            # NOTE there are other overloadings of '__getitem__' for 'str'(i.e. char list), 'Dict(t)' in TorchScript
            container, index = input_val
            frame.add_var(outputs[0].debugName(), container[index])
            return []

        elif fsig == 'torch.__range_length':
            lo, hi, step = input_val
            rng_len = ScriptModuleParser.aten___range_length(lo, hi, step)
            frame.add_var(outputs[0].debugName(), rng_len)
            return []

        elif fsig == 'torch.__derive_index':
            index, start, step = input_val
            derived = ScriptModuleParser.aten___derive_index(index, start, step)
            frame.add_var(outputs[0].debugName(), derived)
            return []

        # May be a symbolic object i.e. IRFwOperation,
        # or, occasionally this node can be statically evaluated, therefore a concrete value
        node = Sign2Op.map(fsig)(*input_val)

        if isinstance(node, IRFwOperation):
            # to create IR node

            ir_node = node
            ScriptModuleParser.setup_node(ir_node)

            if len(ir_node.outputs()) != len(outputs):
                assert len(outputs) == 1, (
                    f"Farse Fail: torchscript has different output number of IR node: {len(outputs)} != {len(ir_node.outputs())}\n"
                    f"This can only be happend to have pre-defined output number of 1"
                )
                node_outputs = (ir_node.outputs(),)
            else:
                node_outputs = ir_node.outputs()

            # handle outputs
            for output, node_output in zip(outputs, node_outputs):
                frame.add_var(output.debugName(), node_output)
            return [ir_node]

        else:
            # concrete value.
            assert len(outputs) == 1, "Cases with multiple outputs are only List/Tuple-Unpack and handled specially"
            frame.add_var(outputs[0].debugName(), node)
            return []

    @staticmethod
    def parse_prim_method_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse script module node like:
            %output.1 : Tensor = prim::CallMethod[name="forward"](%2, %x.1)

        prim::CallMethod has a underlying submodule
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]

        # forward
        label = node.s('name')
        # handle inputs -- in stack with reverse order
        for input in inputs[1:][::-1]:
            var_name = input.debugName()
            val = frame.get_var(var_name)
            frame.push_param(var_name)

        # recursively parse the module
        self_module = node.inputsAt(0).debugName() == 'self'
        if self_module:
            call_module = module
        else:
            call_module = frame.get_var(node.inputsAt(0).debugName())
            assert isinstance(call_module, torch.nn.Module), "the call module is not torch.nn.Module"
            # call_module = getattr(module, node.inputsAt(0).debugName())

        call_method = getattr(call_module, label)
        _, ir_nodes, outputs_val = ScriptModuleParser.parse_module_method(call_module, call_method, frame=frame)

        # pop out the frame
        frame.pop_param(times=len(inputs)-1)

        # handle outputs
        outputs = [output for output in node.outputs()]
        for output, val in zip(outputs, outputs_val):
            frame.add_var(output.debugName(), val)

        return ir_nodes

    @staticmethod
    def parse_prim_attr_node(node, module, frame: Frame) -> List[None]:
        """
        Parse script module node like:
            %2 :__torch__.torch.nn.modules.linear.___torch_mangle_0.Linear = prim::GetAttr[name="linear1"](%self)
            %3 : Tensor = prim::GetAttr[name="weight"](%self)
        Or:
            %embed.1 : __torch__.torch.nn.modules.sparse.Embedding = prim::GetAttr[name="embed"](%self)
            %embed.3 : Tensor = prim::CallMethod[name="forward"](%embed.1, %input_ids.1)

        This will add frame with the variable name and it's value

        The value can be:
            1). (IRFullTensor) the tensor edge in graph
            2). (str code) symbolic value based on runtime info (e.g., self.training)
            3). (str) Function or torch.nn.moudles

        Returns:
            Empty list
        """
        global _refmodule

        module_name = node.inputsAt(0).debugName()
        module = module if module_name == 'self' else frame.get_var(module_name)
        assert isinstance(module, torch.nn.Module)

        label = node.s('name')
        var_name = node.outputsAt(0).debugName()
        dtype = node.outputsAt(0).type().str()

        if dtype == 'Tensor?':
            tensor = getattr(module, label)
            if torch.is_tensor(tensor):
                dtype = 'Tensor'

        # this usually means weight (nn.Parameter in torch)
        if dtype == 'Tensor':
            tensor = getattr(module, label)
            ir_tensor = frame.get_attr_var(tensor)
            if ir_tensor is None:
                for name, param in module.named_parameters():
                    if param is tensor:
                        raise NotImplementedError(
                            f"param {name} is not registered in the frame."
                            f"parameters or buffers must be registered using "
                            f"`self.register_param()` or `self.register_buffer()`.")
                assert False
            frame.add_var(var_name, ir_tensor)
        # symbolic attributes
        elif dtype in ['bool', 'int', 'float']:
            if hasattr(_refmodule, label):
                val = 'self.' + label
            else:
                val = getattr(module, label)
            frame.add_var(var_name, val)
        # NoneType
        elif dtype == 'NoneType':
            frame.add_var(var_name, None)
        else:
            if isinstance(module, torch.nn.ModuleList):
                if str.isdecimal(label):
                    val = module[int(label)]
                else:
                    val = getattr(module, label)
            else:
                val = getattr(module, label)
            frame.add_var(var_name, val)
        return list()

    @staticmethod
    def parse_prim_setattr_node(node, module, frame) -> List[IRFwOperation]:
        """
         = prim::SetAttr[name="past_k"](%self, %k.1)
        """
        raise NotImplementedError("Not support for setattr")

    @staticmethod
    def parse_prim_constant_node(node, module, frame) -> List[None]:
        """
        Parse script module node like:
            %6 : Function = prim::Constant[name="dropout"]()
            %5 : bool = prim::Constant[value=0]()
        
        This will add frame with the variable name and it's value

        Returns:
            Empty list
        """
        if len(list(node.inputs())) != 0:
            raise RuntimeError(f"prim::Constant node: {node} has inputs")
        var_name = node.outputsAt(0).debugName()
        dtype = node.outputsAt(0).type().str()

        if dtype == 'Function':
            signature = repr(node.outputsAt(0).type())
            if '__torch__.' in signature:
                signature = re.sub('__torch__.', '', signature)
            frame.add_var(var_name, signature)
        else:
            val = node.outputsAt(0).toIValue()
            frame.add_var(var_name, val)
        return list()

    @staticmethod
    def parse_prim_if_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Parse script module node like 
            %output1 : Tensor, %output2 : Tensor = prim::If(%15) # /tmp/ipykernel_27188/2459450745.py:13:8
                block0():
                    -> (%1, %2)
                block1():
                    -> (%3, %4)
        
        and the only input (e.g. %15) must be of type bool.
        """

        inputs : List[torch._C.Value] = list(node.inputs())
        outputs : List[torch._C.Value] = list(node.outputs())

        assert len(inputs) == 1
        in_val = frame.get_var(inputs[0].debugName())
        if not isinstance(in_val, bool):
            raise RuntimeError("Dynamic Graph is not supported yet")
        
        # type: torch._C.Block
        true_block, false_block = node.blocks()
        chosen_block : torch._C.Block = true_block if in_val else false_block
        body_out_vars = list(chosen_block.outputs())

        all_ir_nodes : List[IRFwOperation] = []

        # Evaluate the 'eval_block' in a new frame, to isolate within-block variables from
        # polluting the current frame. And we'll manually bind all resultant variables later on.
        frame.push_var(inherit_from_top=True)

        # prim::If's blocks do not have any subgraph parameters, directly evaluate the body
        for subnode in chosen_block.nodes():
            sub_ir_nodes : List[IRFwOperation] = ScriptModuleParser.parse_node(subnode, module, frame)
            for sub_node in sub_ir_nodes:
                ScriptModuleParser.setup_node(sub_node)
            all_ir_nodes += sub_ir_nodes

        # retrieve the block's resultant values
        result_vals = [frame.get_var(body_out_var.debugName()) for body_out_var in body_out_vars]

        # clean up
        frame.pop_var()

        # bind the prim:If's resultant variables
        assert len(result_vals) == len(outputs)
        for output, out_val in zip(outputs, result_vals):
            frame.add_var(output.debugName(), out_val)
        
        return all_ir_nodes

    @staticmethod
    def parse_prim_loop_node(node, module, frame: Frame) -> List[IRFwOperation]:
        """
        Inputs:
            %max_iter_count : int
            %init_condition : bool
            %x_1 : T_1
            ...
            %x_N : T_N
            %dependencies : R

        Syntax:
            %y_1 : T_1, ..., %y_N : T_N = prim::Loop(%max_iter_count, %init_condition, %x_1, ..., %x_N)
              block0(%iter_step : int, %p_1 : T_1, ..., %p_N : T_N):
                ...
                %r_1 : T_1 = some_func(%x_1, %dependencies)
                ...
                %r_N : T_N = ...
                %next_condition : bool = ...
                -> (%next_condition, %r_1, ..., %r_N)

        REMARK:
            -   Outer variables (%dependencies) may be referenced in the Loop-body/subgraph, this is AKA _free variables_.
                In contrast, a standalone TorchScript function/graph will have all variables,
                including its parameters, defined within its scope.
                
                In other words, functions/graphs have no free variables.

        Semantics:
            -   The next step is evaluated if both (%iter_step < %max_iter_count) and (%next_condition == True).
            -   (%y_1, ..., %y_N) are bound to the last (%r_1, ..., %r_N) returned.
                If no step is ever evaluated, they are (%x_1, ..., %x_N).
        """
        inputs : List[torch._C.Value] = list(node.inputs())
        outputs : List[torch._C.Value] = list(node.outputs())

        in_vals = [frame.get_var(input.debugName()) for input in inputs]

        max_iter_count, init_condition = in_vals[0:2]
        if not isinstance(max_iter_count, int):
            raise RuntimeError("The upper bound of the loop must be able to be statically evaluated")
        if not isinstance(init_condition, bool):
            raise RuntimeError("The init condition of the loop must be able to be statically evaluated")

        # type: Subgraph
        loop_block : torch._C.Block = list(node.blocks())[0]

        body_in_vars : torch._C.Value = list(loop_block.inputs())
        iter_step_var = body_in_vars[0]
        p_vars = body_in_vars[1:]

        body_out_vars = list(loop_block.outputs())

        step = 0
        condition = init_condition
        loop_carried_vals = in_vals[2:]

        all_ir_nodes : List[IRFwOperation] = []

        while step < max_iter_count and condition:

            # create the context for evaluating the body, and bind loop variables %iter_step, %p_1, ...

            # Defensively we don't let variables defined in the Loop body subgraph pollute the outer graph.
            # So we'd better duplicate all existing variables into a new frame (namely 'inherit_from_top'), 
            # and clean up this new frame after the interpretation of the whole loop execution.
            frame.push_var(inherit_from_top=True)

            frame.add_var(iter_step_var.debugName(), step)

            # At the evaluation of each step, we cannot call Frame's 'push_param(var_name)' and 'add_var(var_name, val, graph_arg=N)' APIs,
            # because all intermediate loop-carried values do not have syntactically static names.
            #
            # For the sake of isolation, we don't bind carried values onto {y_i}s variables and overwrite the binding
            # during evaluation, either.
            assert len(p_vars) == len(loop_carried_vals)
            for p_var, carried_val in zip(p_vars, loop_carried_vals):
                frame.add_var(p_var.debugName(), carried_val)

            # evaluate the body block
            for subnode in loop_block.nodes():
                subnode : torch._C.Node
                sub_ir_nodes : List[IRFwOperation] = ScriptModuleParser.parse_node(subnode, module, frame)
                for sub_node in sub_ir_nodes:
                    ScriptModuleParser.setup_node(sub_node)
                all_ir_nodes += sub_ir_nodes

            # rebind for next step and clean-ups
            step_result_vals = [frame.get_var(body_out_var.debugName()) for body_out_var in body_out_vars]
            condition = step_result_vals[0]
            loop_carried_vals = step_result_vals[1:]
            step += 1

            frame.pop_var()

            if not isinstance(condition, bool):
                raise RuntimeError(f"At the {step}-th step the condition is not evaluated to a constant bool")

        assert len(outputs) == len(loop_carried_vals)
        for output, y_val in zip(outputs, loop_carried_vals):
            frame.add_var(output.debugName(), y_val)

        return all_ir_nodes

    @staticmethod
    def parse_prim_list_construct_node(node, module, frame: Frame) -> List[None]:
        """
        Parse script module node like
            %8 : int[] = prim::ListConstruct(%3)
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]
        assert len(outputs) == 1
        output = outputs[0]
        out_val = list()
        for input in inputs:
            out_val.append(frame.get_var(input.debugName()))
        frame.add_var(output.debugName(), out_val)
        return list()

    @staticmethod
    def parse_prim_list_unpack_node(node, module, frame: Frame) -> List[None]:
        """
        Parse script module node like:
            %q.1 : Tensor, %k.1 : Tensor, %v.1 : Tensor = prim::TupleUnpack(%11)
        """
        inputs = [input for input in node.inputs()]
        outputs = [output for output in node.outputs()]
        if len(inputs) != 1:
            raise RuntimeError("Find UnpackTuple has more than one input")
        if len(outputs) == 1:
            raise RuntimeError("Find UnpackTuple has only one output")
        tuple_inputs = frame.get_var(inputs[0].debugName())
        if len(tuple_inputs) != len(outputs):
            raise RuntimeError("Expected unpacked tuple number have same length of tupled input")
        for output, val in zip(outputs, tuple_inputs):
            frame.add_var(output.debugName(), val)
        return list()

    @staticmethod
    def parse_prim_python_op_node(node, module, frame):
        """
        parse node like:
            %64 : Tensor = ^OuterProductMean()(%opm_left.1, %opm_right.1, %outer_out_proj)
        """
        # get inputs
        input_vals = list()
        for input in node.inputs():
            var_name = input.debugName()
            val = frame.get_var(var_name)
            input_vals.append(val)
        
        func: Callable = node.pyobj()
        fsig = f'{inspect.getmodule(func).__name__}.{func.__name__}'

        # map to IR operator
        ir_node = Sign2Op.map(fsig)(*input_vals)
        ScriptModuleParser.setup_node(ir_node)

        cnt = 0
        for output in node.outputs():
            if isinstance(output.type(), torch._C.TupleType):
                tuplen = len(output.type().elements())
                ir_output = [ir_node.output(idx) for idx in range(cnt, cnt+tuplen)]
                cnt += tuplen
            else:
                ir_output = ir_node.output(cnt)
                cnt += 1
            frame.add_var(output.debugName(), ir_output)

        if cnt != len(ir_node.outputs()):
            raise RuntimeError(
                f"Parse fail: {fsig} has {cnt} outputs != pre-defined {len(ir_node.outputs())}"
            )
        return [ir_node]

    @staticmethod
    def parse_value_erased_node(node, module, frame, erased_vals: List[Any]):
        outputs = list(node.outputs())

        assert len(outputs) == len(erased_vals)
        for output, ev in zip(outputs, erased_vals):
            frame.add_var(output.debugName(), ev)
        return []


    @staticmethod
    def flatten(smodule, depth=0):
        """
        Flatten the recursive script module to function and aten primitives
        """
        # stashed_module = list()
        inputs = [input for input in smodule.graph.inputs()]
        print('    '*depth, f'graph inputs: {inputs}')
        if len(list(smodule.children())) == 0:
            for node in smodule.graph.nodes():
                print('    '*depth, node)
        else:
            for node in smodule.graph.nodes():
                print('    '*depth, node)
                if node.kind() == 'prim::CallMethod':
                    label = node.inputsAt(0).node().s('name')
                    submodule = getattr(smodule, label)
                    ScriptModuleParser.flatten(submodule, depth+1)

    @staticmethod
    def aten___range_length(lo, hi, step):
        """
        aten::__range_length(int lo, int hi, int step) -> int

        Python loops
            ```
            for i in range(L, H, S):
                use(i)
            ```
        will be translated to TorchScript
            ```
            _c = aten::__range_length(L, H, S)
            for _k < _c:
                i = aten::__derive_index(k, L, S)
                use(i)
            ```
        """
        if not (isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int)):
            raise RuntimeError("All inputs to __range_length must be statically evaluated")
        if step == 0:
            raise RuntimeError("Step cannot be zero")

        return len(range(lo, hi, step))

    @staticmethod
    def aten___derive_index(index, start, step):
        if not (isinstance(index, int) and isinstance(start, int) and isinstance(step, int)):
            raise RuntimeError("All inputs to __derive_index must be statically evaluated")

        return start + index * step



