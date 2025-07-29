from typing import Optional, Tuple, Any, Union, List
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
        # similar with set_input and set_output, the IRObject
        # in kwargs will be set with copy-on-write to avoid
        # potential modifications outside.
        def replace(t: IRObject):
            t = copy.copy(t)
            t.cell = self
            return t

        kwargs = IRCell.modify_objects_of_complex(kwargs, replace)
        self.kwargs.update(kwargs)

        # default infer rule
        requires_grad = any(
            t.requires_grad for t in inputs if isinstance(t, IRTensor))

        # setup output
        outputs = [IRFullTensor(requires_grad=requires_grad) for _ in range(num_outputs)]
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

    def infer_shape(self):
        """
        Infer output value shape
        """
        raise NotImplementedError

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

    def algorithms(self, tag: Optional[str] = None) -> Union[Tuple[GenericDistAlgo], GenericDistAlgo]:
        """
        get algorithm from algorithm factory

        @param tag Optional[str]: the queried tag (default None for all)

        @return algorithm(s) Union[Tuple[GenericDistAlgo], GenericDistAlgo]:
            If None (default), return all possible algorithms.
            Otherwise, return the specified one.
        """
        factory = DistAlgorithmFactory()
        if tag is None:
            templates = list()
            if factory.exist(type(self)):
                templates = factory.algorithms(type(self))
            algos = list()
            for template in templates:
                algos.append(template(self))
            return algos
        else:
            assert factory.exist(type(self), tag), f"Node {self} doesn't have transformation algorithm tag: {tag}"
            template = factory.algorithms(type(self), tag)
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
