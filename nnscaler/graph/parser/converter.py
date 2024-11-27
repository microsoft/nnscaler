#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Any, Dict, Union
import logging
from pathlib import Path
import operator
import warnings

from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.graph import IRGraph
from nnscaler.flags import CompileFlag

from nnscaler.graph.parser import FxModuleParser
from nnscaler.graph.tracer import concrete_trace
from nnscaler.graph.tracer.wrap_utils import Location, is_autograd_apply, LeafWrapInfo
from nnscaler.graph.tracer.torch_fx_patcher import side_effectful_inplace_ops

import nnscaler.runtime.function as cube_rt_function

import torch
import torch.fx
from torch.autograd.graph import saved_tensors_hooks

_logger = logging.getLogger(__name__)


class no_save_tensor_hook(saved_tensors_hooks):
    """skip saving tensors for backward since tracer only traces forward"""
    def __init__(self):
        def pack(x):
            return None
        def unpack(x):
            raise RuntimeError("not expecting backward to be called on this tensor")
        super().__init__(pack, unpack)


def _rewrite_inplace_ops(traced_model: torch.fx.GraphModule):
    """Rewrite inplace ops to use its outputs so we can track them in IRGraph

    x.add_(y)           =>  x = x.add_(y)
    operator.iadd(x, y) =>  x = operator.iadd(x, y)
    x += y              =>  x += y # no change

    Args:
        traced_model (torch.fx.GraphModule): fx graph to be modified
    """
    done_nodes = set()
    for n in traced_model.graph.nodes:
        done_nodes.add(n)
        # inplace operator on torch.Tensor has the pattern: first arg is tensor + "call_method" + method name end with single "_"
        if (
            (n.op == "call_method" and n.target.endswith("_") and not n.target.endswith("__"))
            and n.args[0].meta.get('type', None) == torch.Tensor
        ) or (n.op == "call_function" and n.target in side_effectful_inplace_ops):
            # setitem is a special inplace operator that returns None instead of the first modified argument,
            # to make it align with SSA format, we use cube runtime function to return the first argument
            if n.op == "call_function" and n.target == operator.setitem:
                n.target = cube_rt_function.setitem
            n.args[0].replace_all_uses_with(n, delete_user_cb=lambda node: not node in done_nodes)
    # we can't recompile
    # it will raise error if we have autograd, customized op, etc.
    # The good part is we don't need to generate python code
    # we will use the fx graph directly
    # traced_model.recompile()


def to_fx_graph(model: torch.nn.Module, dummy_input) -> torch.fx.GraphModule:
    """
    Convert torch.nn.Module based model into torch.fx.GraphModule
    Args:
        model (torch.nn.Module): single-device model description
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
    Returns:
        torch.fx.GraphModule representation of model
    """
    # get registered leaf function
    autowrap_funcs = [CustomizedOps.kOpRuntime[sign] for sign in CustomizedOps.kOpMap]
    # filter out torch.autograd.Function.apply as concrete trace already treats them as leaf function
    autowrap_funcs = [fn for fn in autowrap_funcs if not is_autograd_apply(fn)]
    leaf_functions = {func: LeafWrapInfo([], True, None) for func in autowrap_funcs if func is not None}

    # importlib functions
    # currently only import_module is handled in the code
    import importlib
    leaf_functions.update({
        func: LeafWrapInfo([Location(importlib, func.__name__)], False, None)
        for func in [importlib.import_module]
    })

    # get cube runtime functions
    cube_rt_funcs = [
        cube_rt_function.anchor,
        cube_rt_function.ifexpr,
        cube_rt_function.fold_constant
    ]
    leaf_functions.update({
        func: LeafWrapInfo([Location(cube_rt_function, func.__name__)], True, None)
        for func in cube_rt_funcs
    })
    dce_ignored_funcs = set(cube_rt_funcs)

    with no_save_tensor_hook(), warnings.catch_warnings():
        # ignore the warning from fx about get_attr
        warnings.filterwarnings("ignore", message=
            ".*does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target"
        )
        traced_model = concrete_trace(
            model,
            dummy_input,
            use_operator_patch=True,
            autowrap_leaf_function=leaf_functions,
            dce_ignored_function=dce_ignored_funcs,
            strategy=CompileFlag.trace_strategy,
            record_frames=not CompileFlag.disable_code_line_info,
        )
    _rewrite_inplace_ops(traced_model)
    return traced_model


def to_ir_graph(
    traced_model: torch.fx.GraphModule,
    dummy_input: Dict[str, Any],
    attr_savedir: Union[str, Path],
    constant_folding: bool = True,
) -> IRGraph:
    """Convert torch.fx.GraphModule based model into IRGraph

    Args:
        traced_model (torch.fx.GraphModule): single-device model description in fx format
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
        constant_folding (bool):
            whether to enable constant folding. Default True.
        attr_savedir (Union[str, Path]): directory to save content (attribtes)

    Returns:
        IRGraph: IRGraph of model
    """
    _logger.info(f"constant folding {'enabled' if constant_folding else 'disabled'} to parse graph")

    with no_save_tensor_hook():
        inputs, nodes, outputs = FxModuleParser.parse(
            traced_model, dummy_input,
            attr_savedir=attr_savedir,
            constant_folding=constant_folding,
            save_content=True,
        )
    module_name = traced_model.__class__.__name__

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph


def convert_model(
    model: torch.nn.Module,
    dummy_input: Dict[str, Any],
    attr_savedir: Union[str, Path],
    constant_folding: bool = True
) -> IRGraph:
    """Convert torch.nn.Module based model into IRGraph

    Args:
        model (torch.nn.Module): single-device model description
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
        constant_folding (bool):
            whether to use constant folding. Default True.
        attr_save_dir (Union[str, Path]): directory to save content (attribtes)

    Returns:
        IRGraph: IRGraph of model
    """
    traced_model = to_fx_graph(model, dummy_input)
    _logger.debug(f'the traced model is:\n{traced_model}')
    graph = to_ir_graph(traced_model, dummy_input, attr_savedir, constant_folding)
    return graph
