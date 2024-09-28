#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple, Optional, Any, Dict, Union
import inspect

from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.operator import IRBpOperation, IRDataOperation

from nnscaler.graph import IRGraph
from nnscaler.graph import parser

from nnscaler.runtime.module import CubeModule
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.utils import MicroBatchDataLoader

from nnscaler.utils import load_model

import torch
import torch.utils.data as data


_program_graph: Optional[IRGraph] = None


def enable_global_graph():
    global _program_graph
    _program_graph = IRGraph([], [], [], 'program')


def disable_global_graph():
    global _program_graph
    _program_graph = None


def is_global_graph_enabled():
    return _program_graph is not None


class Program:
    """
    This is only used in @compile for backward compatibility.
    """
    def add_node(self, node: IRCell):
        _program_graph.insert(node, _program_graph.nnodes)

    def add_nodes(self, nodes: List[IRCell]):
        for node in nodes:
            self.add_node(node)

    def get_graph(self) -> IRGraph:
        return _program_graph

    def set_input(self, inputs: Tuple[Any]):
        _program_graph.reset_inputs(len(inputs))
        for idx, obj in enumerate(inputs):
            _program_graph.set_input(idx, obj)
        # update gradient
        for t in IRGraph.get_objects_from_complex(_program_graph.inputs()):
            if isinstance(t, IRSubTensor) and t.requires_grad:
                t.grad = t.parent.grad.tosub()

    def set_output(self, outputs: Tuple[Any]):
        _program_graph.reset_outputs(len(outputs))
        for idx, otensor in enumerate(outputs):
            _program_graph.set_output(idx, otensor)
        # update gradient
        for t in IRGraph.get_objects_from_complex(_program_graph.outputs()):
            if isinstance(t, IRSubTensor) and t.requires_grad:
                t.grad = t.parent.grad.tosub()

    def finalize(self):
        """
        Close the recording of program.
        If the program doesn't do backward, set all tensors with requires_grad=False.
        """
        # inference scenario, set all gradients to none.
        if not any(isinstance(node, IRBpOperation) for node in _program_graph.nodes()):
            _program_graph.no_backward()

    def clear(self):
        # will enable and create an empty global graph
        enable_global_graph()

    def __repr__(self):
        return repr(_program_graph)


class SemanticDataLoader:

    def __init__(self, dataloader: MicroBatchDataLoader):
        """Create semantic dataloader representing the dataloader in training iteration.

        Calling `next(SemanticDataLoader)` will generate an IRDataOperation in graph,
        which takes the `self.irobj` (i.e., reperesenting the non-tensor value of real
        dataloader instance) as input and produces outputs that are converted to
        IRObject or IRTensor. The IRDataOperation will be added to the final
        graph and generate code like `data = next(dataloader)`

        Args:
            dataloader (MicroBatchDataLoader): torch dataloader
        """
        if not isinstance(dataloader, MicroBatchDataLoader):
            raise TypeError("Expected data loader to be MicroBatchDataLoader")
        self.dataloader: data.DataLoader = dataloader
        # the IRObject representing the `dataloader` instance, which is only used by the
        # IRDataOperation. Since we already know the output of the dataloader,
        # we don't need to set the value for it.
        self.irobj = IRObject(name='dataloader', value=None, is_constant=False)

    def __iter__(self):
        return self

    def __next__(self):
        # get dataloader sample
        sample = next(iter(self.dataloader))
        if not isinstance(sample, tuple):
            sample = (sample,)
        # turn sample into IRObjects
        outputs = tuple(IRObject.from_complex('data', s, tosub=True, requires_grad=False, is_constant=False) for s in sample)
        outputs = tuple(IRObject('data', value=out) if not isinstance(out, IRObject) else out for out in outputs)
        # create dataloader operation
        # the `self.irobj` is the IRObject standing for the non-tensor value of real dataloader.
        # the `self.irobj` are also usually used as one input of the whole graph
        data_op = IRDataOperation(self.irobj, outputs)
        Program().add_node(data_op)
        # return the outputs in the same format with real dataloader
        outputs = outputs[0] if len(outputs) == 1 else outputs
        return outputs


class SemanticModel:

    def __init__(self, model: Optional[torch.nn.Module],
                 save_content: bool = True,
                 constant_folding: bool = True,
                 attr_savedir: str = './',
    ):
        """
        Create semantic model based on AI Scientist description.

        Args:
            model (Optional[torch.nn.Module]):
                single-device model description, only required for rank 0
            save_content (bool):
                whether to save the content of model and load it into generated model. Default True.
            constant_folding (bool):
                whether to enable constant folding. Default True.
            attr_savedir (str):
                directory to save content (attribtes)
        """
        if DeviceGroup().local_rank == 0 and model is not None:
            assert isinstance(model, torch.nn.Module), f"device of local_rank == 0 must provide model"
        self.model = model
        self._dummy_input: Dict[str, Any] = None
        self._ir_graph = None
        self._loaded_module: CubeModule = None
        # parser configuration
        self.save_content: bool = save_content
        self.constant_folding: bool = constant_folding
        self.attr_savedir: str = attr_savedir

    @property
    def dummy_input(self) -> Any:
        """Get dummy real-tensor input from on CPU"""
        return self._dummy_input

    @dummy_input.setter
    def dummy_input(self, val):

        def complex(val: Any):
            """Complex to CPU"""
            if isinstance(val, tuple):
                return tuple(complex(t) for t in val)
            if isinstance(val, list):
                return list(complex(t) for t in val)
            if isinstance(val, dict):
                return {complex(key):complex(val) for key, val in val.items()}
            if isinstance(val, torch.Tensor):
                return val.cpu()
            return val

        self._dummy_input = complex(val)

    def get_graph(self):
        return self._ir_graph

    def load_module(self, filename: str, load_fullmodelpt: Optional[bool] = None):
        """Load module from file.

        Args:
            filename (str): file path
            load_fullmodelpt (Optional[bool]): controls whether to load full model checkpoint.
                If None, use the default value of the semantic model.
        """
        if load_fullmodelpt is not None:
            load_content = load_fullmodelpt
        else:
            load_content = self.save_content
        self._loaded_module = load_model(filename, load_content)

    def get_gen_module(self) -> Optional[torch.nn.Module]:
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None

    def __call__(self, *args):
        """Forward the semantic model.

        This will parse the model into cube graph.

        Args:
            *args: input IRObjects

        Returns:
            graph outputs with IRObjects
        """
        assert self._ir_graph is None, \
            f"multiple forward on a semantic model is not allowed"
        if DeviceGroup().local_rank == 0:
            # collect dummy input
            if self.dummy_input is None:
                dummy_input = {}
                sig = inspect.signature(self.model.forward)
                for name, arg in zip(sig.parameters.keys(), args):
                    if isinstance(arg, IRObject):
                        value = arg.value
                        arg._value = None  # remove value to release memory
                    else:
                        value = arg
                    dummy_input[str(name)] = value
                self.dummy_input = dummy_input
            # parse graph
            self._ir_graph = parser.convert_model(
                self.model,
                dummy_input=self.dummy_input,
                attr_savedir=self.attr_savedir,
                constant_folding=self.constant_folding
            )
            return self._ir_graph(*args)
