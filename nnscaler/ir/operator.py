#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Optional, Tuple, Any, Union, List, overload
import copy

from nnscaler.ir.cten import IRCell, IRTensor, IRObject
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.algorithm.factory import DistAlgorithmFactory
from nnscaler.algorithm.generics import GenericDistAlgo
from nnscaler.ir.dtype import DTypeInfo


class IRFwOperation(IRCell):
    """
    Forward operation
    """

    def __init__(self, name: str, signature: str,
                 inputs: List[IRObject], num_outputs: int, **kwargs):
        """!
        Create a forward operation.

        @param name str: the name of forward operation
        @param signature str: the signature of the forward operation
        @param input_length int: number of inputs
        @param output_length int: number of outputs
        """
        # recompute schedule
        self._recompute = None
        super().__init__(name, signature, len(inputs), num_outputs)

        # setup input
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

        # setup kwargs
        for name, value in kwargs.items():
            self.set_kwarg(name, value)

        # setup output
        outputs = [IRObject.missing for _ in range(num_outputs)]
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

    def infer_shape(self) -> dict[int, tuple[int, ...]]:
        """
        Infer output value shape for each output
        Will not update graph or shape of `self._outputs`
        """
        # by default, no shape inference
        return {}

    def verify_shape(self, outputs=None) -> None:
        """
        Verify the shape of the outputs with inferred shape of the operator.
        Raise error if shape mismatch.

        Args:
            outputs: the outputs to match. If None, use self.outputs()

        Raises:
            ValueError: if shape mismatch
        """
        infered_shapes = self.infer_shape()
        outputs = outputs if outputs is not None else self.outputs()
        for oidx in range(len(outputs)):
            if oidx not in infered_shapes:
                continue
            if not isinstance(outputs[oidx], IRTensor):
                raise ValueError(f'find type inference not match: {outputs[oidx]} expected to be a tensor')

            if tuple(outputs[oidx].shape) != tuple(infered_shapes[oidx]):
                raise ValueError(
                    f'find shape inference not match: {outputs[oidx].shape} vs {infered_shapes[oidx]}'
                    f'\nnode: {self}'
                )

    @property
    def recompute(self) -> Optional[int]:
        """!
        Get recompute group id.
        To enable recompute, a recompute group refers to a sequence of operators that
        will perform recompute optimization.

        @return group_id Optional[int]: None if no recompute, else a group id.
        """
        return self._recompute

    @recompute.setter
    def recompute(self, group_id: Optional[int]):
        """!
        Set recompute group

        @param group_id Optional[int]: recompute group id. None indicates no group is applied
        """
        assert group_id is None or isinstance(group_id, int), "Expect None or int"
        if isinstance(group_id, int) and self._recompute is not None:
            assert self._recompute == group_id, "The operator is set to recompute in another recompute group."
        self._recompute = group_id

    def algorithms(self) -> List[GenericDistAlgo]:
        """
        get all algorithms from algorithm factory

        Returns:
            List[GenericDistAlgo]: all possible algorithms
        """
        factory = DistAlgorithmFactory()
        return [template(self) for template in factory.algorithms(type(self))]

    def algorithm(self, tag: str) -> GenericDistAlgo:
        """
        get a specific algorithm from algorithm factory

        Args:
            tag (str): the tag of the algorithm

        Returns:
            GenericDistAlgo: the algorithm
        """
        factory = DistAlgorithmFactory()
        template = factory.algorithm(type(self), tag)
        return template(self)

    def replicate(self):
        """!
        Replicate the forward operation.
        The operator id, recompute and comment attribute will also be replicated.

        @return replica IRFwOperation: the replicated operator
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._id = self.cid
        # reset input and output
        cpy.reset_inputs(len(self.inputs()))
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy.reset_outputs(len(self.outputs()))
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy.reset_kwargs()
        for name, value in self.kwargs.items():
            cpy.set_kwarg(name, value)
        cpy._mirror = None
        cpy.recompute = self.recompute
        return cpy

    def __repr__(self) -> str:
        dscp = (f"FwOp{self._id}-{self.device}(name={self.name}, "
                f"inputs={self.inputs()}, "
                f"outputs={self.outputs()})")
        return dscp

    def extra_repr(self) -> str:
        dscp = (f"FwOp{self._id}-{self.device}(name={self.name}, "
                f"sign={self.signature}, "
                f"inputs={self.inputs()}, "
                f"outputs={self.outputs()})")
        return dscp


class IRBpOperation(IRCell):
    """
    Backward operation
    """

    def __init__(self, ograds: Tuple[Any], igrads: Tuple[Any]):
        """
        Create dummy backward node for forward inputs and forward outputs

        @param fwop IRFwOperation: forward operator
        """
        super().__init__(
            'backward', 'torch.autograd.grad',
            len(ograds), len(igrads)
        )
        for idx, ograd in enumerate(ograds):
            self.set_input(idx, ograd)
        for idx, igrad in enumerate(igrads):
            self.set_output(idx, igrad)

    def isfw(self) -> bool:
        return False

    def replicate(self):
        """
        Replicate the backward op
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._id = self.cid
        # reset input and output
        cpy.reset_inputs(len(self.inputs()))
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy.reset_outputs(len(self.outputs()))
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        assert not cpy.kwargs, "No kwargs for backward op"
        cpy._mirror = None
        return cpy

    def __repr__(self) -> str:
        dscp = (f"BwOp{self._id}-{self.device}(FwOp{self.mirror._id}, "
                f"inputs={self.inputs()}, "
                f"outputs={self.outputs()})")
        return dscp


class IRDataOperation(IRCell):
    """Dataloader operator

    The output of a dataloader operator is a tuple of (IRObject,).
    """

    def __init__(self, input: IRObject, outputs: Tuple[IRObject], name='dataloader'):
        signature = 'next'
        super().__init__(name, signature, 1, len(outputs))
        if not isinstance(input, IRObject):
            raise TypeError(f"input should be an IRObject, but got {type(output)}")
        self.set_input(0, input)
        for idx, output in enumerate(outputs):
            if not isinstance(output, IRObject):
                raise TypeError(f"output should be an IRObject, but got {type(output)}")
            self.set_output(idx, output)

    def replicate(self):
        """
        Replicate the Operation
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._id = self.cid
        # reset input and output
        cpy.reset_inputs(len(self.inputs()))
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy.reset_outputs(len(self.outputs()))
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy.reset_kwargs()
        for name, value in self.kwargs.items():
            cpy.set_kwarg(name, value)
        cpy._mirror = None
        return cpy

    def infer_shape(self):
        """
        Infer output value shape
        """
        return True

    def __repr__(self):
        dscp = (f"DataLoader{self._id}-{self.device}(outputs={self.outputs()})")
        return dscp

    def module_repr(self) -> str:
        return repr(self)
