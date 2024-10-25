#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Iterable, Dict, List, Any
import itertools

from nnscaler.ir.cten import IRCell, IRTensor, IRObject
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.graph.segment import IRSegment
from nnscaler.execplan.execplan import ExeReuseCell

from nnscaler.codegen.emit import FuncEmission


class LifeCycle:

    def __init__(self, nodes: List[IRCell], graph_inputs: List[Any], graph_outputs: List[Any]):

        graph_inputs = IRSegment.get_objects_from_complex(graph_inputs)
        graph_outputs = IRSegment.get_objects_from_complex(graph_outputs)
        func_emission = FuncEmission()

        self.nodes: Dict[IRCell, int] = {node: lid for lid, node in enumerate(nodes)}
        # the last line id of consuming or producing a tensor
        self.lifetime: Dict[IRObject, int] = {}
        # the tensors can be released given the finish of line id
        self.release: Dict[int, List[IRObject]] = {}

        # FIXME: consider the case of IRObject in the kwargs of IRFwOperation
        # is_activation = lambda t: isinstance(t, IRObject) and not t.is_attr()
        is_activation = lambda t: isinstance(t, IRSubTensor) and not t.is_attr()

        self.lifetime.update((tsin, 0) for tsin in graph_inputs if is_activation(tsin))

        for i, node in enumerate(nodes):

            outputs : Iterable[IRObject]
            inputs : Iterable[IRObject]

            if isinstance(node, (IRSegment, ExeReuseCell)):
                # this is only for scheduler to track the lifetime of tensors
                # for module code generation, no IRSegment/ExeReuseCell will be used.
                # so will not go into this branch.

                # forward segment
                if node.isfw():
                    outputs = node.outputs()
                    inputs = node.inputs()
                # backward segment
                else:
                    # in `_train_step`, we will explicitly call backward to generate gradients.
                    # When pipeline is enabled, there will be multiple backward calls in the same segment.
                    # So we also need to track the temporary gradient tensors
                    # and delete them after the backward call to save memory.
                    fw_inputs, fw_outputs, output_grads, input_grads = \
                        func_emission.get_backward_callsite_io_tensors(node)
                    # remove loss gradient
                    output_grads = [t for t in output_grads if not t.is_loss()]

                    outputs = input_grads
                    inputs = list(itertools.chain(fw_inputs, fw_outputs, output_grads))
            else:
                outputs = node.outputs()
                inputs = node.inputs()

            # aggressively mark all outputs for immediate deletion,
            # namely *after* 'i'-th statement, in case it's never used.
            self.lifetime.update((tout, i) for tout in IRSegment.get_objects_from_complex(outputs) if is_activation(tout))

            # "fast-forward" all inputs to the current statement, namely after 'i'-th node.
            self.lifetime.update((tin, i) for tin in IRSegment.get_objects_from_complex(inputs) if is_activation(tin))


        # Here (i+1) is always greater than 'len(nodes)'
        # Generally we don't manually release those tensors since the enclosing function is about to
        # return, all local variables are automatically released.
        # But we do need to update the lifetime of all outputs, to avoid early releasing.
        self.lifetime.update((tsout, len(nodes)) for tsout in graph_outputs if is_activation(tsout))

        for tensor, line_id in self.lifetime.items():
            self.release.setdefault(line_id, []).append(tensor)

    def release_tensors_after_line(self, line_id: int) -> List[IRSubTensor]:
        """
        Get the releasable IRSubTensors after finish of executing of `line_id`.

        @param line_id int

        @return tensors List[IRSubTensors]: tensors that can be released.
        """
        return self.release.get(line_id, [])

    def release_tensors_after_node(self, node: IRCell) -> List[IRSubTensor]:
        """
        Get the releasable IRSubTensors after finish of executing of the node.

        @param line_id int

        @return tensors List[IRSubTensors]: tensors that can be released.
        """
        assert node in self.nodes
        line_id = self.nodes[node]
        return self.release.get(line_id, [])

    def releasable_after_node(self, obj: IRObject, node: IRCell) -> bool:
        """
        Check if the tensor is releasable after executing the node.
        Please note that if it is not a IRSubTensor(is IRObject),
            we will never manually release it.

        Args:
            tensor (IRObject): the tensor to be checked
            node (IRCell): the node to be checked
        Returns:
            releasable (bool): whether the tensor is releasable after executing
        """
        if not isinstance(obj, IRSubTensor):
            return False

        assert node in self.nodes
        assert obj in self.lifetime
        line_id = self.nodes[node]
        return self.lifetime[obj] < line_id

    def releasable_after_line(self, obj: IRObject, line: int) -> bool:
        """
        Check if the tensor is releasable after executing the node.
        Please note that if it is not a IRSubTensor(is IRObject),
            we will never manually release it.

        Args:
            tensor (IRObject): the tensor to be checked
            line (int): the line to be checked
        Returns:
            releasable (bool): whether the tensor is releasable after specific line.
        """
        if not isinstance(obj, IRSubTensor):
            return False

        assert obj in self.lifetime
        return self.lifetime[obj] < line

    def get_line(self, node: IRCell) -> int:
        """
        Get line id of the node
        """
        return self.nodes[node]
