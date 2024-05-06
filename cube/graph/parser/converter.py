# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, Union
import logging
from pathlib import Path

from cube.ir.tensor import IRFullTensor
from cube.graph.parser.register import CustomizedOps
from cube.graph import IRGraph
from cube.flags import CompileFlag

from cube.graph.parser.fx.parser import FxModuleParser
from cube.graph.parser.fx.concrete_trace_utils import concrete_trace
from cube.graph.parser.fx.concrete_trace_utils.concrete_tracer import is_autograd_apply

import cube.runtime.function as cube_rt_function

import torch
import torch.fx
from torch.autograd.graph import saved_tensors_hooks

from cube.graph.parser.script.parser import ScriptModuleParser
from cube.flags import CompileFlag

_logger = logging.getLogger(__name__)


class no_save_tensor_hook(saved_tensors_hooks):
    """skip saving tensors for backward since tracer only traces forward"""
    def __init__(self):
        def pack(x):
            return None
        def unpack(x):
            raise RuntimeError("not expecting backward to be called on this tensor")
        super().__init__(pack, unpack)


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
    leaf_functions = {func: ([], True, None) for func in autowrap_funcs if func is not None}

    # get cube runtime functions
    cube_rt_funcs = [cube_rt_function.anchor]
    leaf_functions.update({func: ([(cube_rt_function, func.__name__)], True, None) for func in cube_rt_funcs})
    dce_ignored_funcs = set(cube_rt_funcs)

    with no_save_tensor_hook():
        traced_model = concrete_trace(
            model,
            dummy_input,
            use_operator_patch=True,
            autowrap_leaf_function=leaf_functions,
            dce_ignored_function=dce_ignored_funcs,
            cpu_offload=True,
            record_frames=not CompileFlag.disable_code_line_info,
        )
    return traced_model


def to_script_graph(model: torch.nn.Module):
    return torch.jit.script(model)


def to_ir_graph(
    traced_model: torch.fx.GraphModule,
    dummy_input: Dict[str, Any],
    attr_savedir: Union[str, Path],
    dynamic_shape: bool = False,
) -> IRGraph:
    """Convert torch.fx.GraphModule based model into IRGraph

    Args:
        traced_model (torch.fx.GraphModule): single-device model description in fx format
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
        dynamic_shape (bool):
            whether to use dynamic shape. Default False.
        attr_savedir (Union[str, Path]): directory to save content (attribtes)

    Returns:
        IRGraph: IRGraph of model
    """
    _logger.info(f"use {'dynamic' if dynamic_shape else 'static'} shape to parse graph")

    with no_save_tensor_hook():
        inputs, nodes, outputs = FxModuleParser.parse(
            traced_model, dummy_input,
            attr_savedir=attr_savedir,
            dynamic_shape=dynamic_shape,
            save_content=True,
        )
    module_name = traced_model.__class__.__name__

    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph


def script_to_ir_graph(
        script_module,
        dummy_input: Dict[str, torch.Tensor],
        attr_savedir: Union[str, Path]):
    """Convert torch.jit.script module into IRGraph"""
    save_content = False if attr_savedir is None else True
    inputs, nodes, outputs = ScriptModuleParser.parse_module(
        script_module, dummy_input, 
        attr_savedir=attr_savedir, save_content=save_content)
    module_name = script_module.original_name

    for input in inputs:
        if isinstance(input, IRFullTensor):
            input.requires_grad = False

    graph = IRGraph.from_logic_graph(nodes, inputs, outputs, module_name)
    return graph


def convert_model(
    model: torch.nn.Module,
    dummy_input: Dict[str, Any],
    attr_savedir: Union[str, Path],
    dynamic_shape: bool = False
) -> IRGraph:
    """Convert torch.nn.Module based model into IRGraph

    Args:
        model (torch.nn.Module): single-device model description
        dummy_input (Dict[str, Any]):
            dummy input of model, the keys are the names of forward arguments.
        dynamic_shape (bool):
            whether to use dynamic shape. Default False.
        attr_save_dir (Union[str, Path]): directory to save content (attribtes)

    Returns:
        IRGraph: IRGraph of model
    """
    if CompileFlag.use_jit_parser:
        traced_model = to_script_graph(model)
        graph = script_to_ir_graph(traced_model, dummy_input, attr_savedir)
    else:
        traced_model = to_fx_graph(model, dummy_input)
        graph = to_ir_graph(traced_model, dummy_input, attr_savedir, dynamic_shape)
    return graph
