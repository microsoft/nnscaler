#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional, Dict
import copy

from nnscaler.ir.adapter.prim import IRAdapterPrim, IdentityPrim
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.cten import IRCell


class IRAdapter(IRCell):

    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__(
            name='adapter', signature='adapter',
            input_length=len(inputs),
            output_length=len(outputs),
        )
        # we don't use input and output setter as this will
        # change tensor device info
        self._inputs = inputs
        self._outputs = outputs

        self._prims: List[IRAdapterPrim] = []
        self._differentiable = False
        self.custom = True

        device = set()
        for tensor in inputs + outputs:
            device.update(set(tensor.device))
        self.device = list(device)

        # recompute group id
        self._recompute = None

        # setup whether this adapter is for forward stage
        is_fw = any(not t.is_grad() for t in self.inputs() + self.outputs())
        is_bw = any(t.is_grad() for t in self.inputs() + self.outputs())
        assert not (is_fw and is_bw), "An IRAdapter cannot serve for both forward and backward stage"
        self._forward = is_fw

        self._cached_dispatch: Dict[int, IRAdapter] = {}

    @property
    def prims(self) -> List[IRAdapterPrim]:
        return copy.copy(self._prims)

    @prims.setter
    def prims(self, prims: List[IRAdapterPrim]):
        assert all(isinstance(prim, IRAdapterPrim) for prim in prims), "Expect List[IRAdapterPrim]"
        self._prims = prims

    @property
    def differentiable(self) -> bool:
        """
        return if the adapter is using differentiable primitives
        """
        return self._differentiable

    @differentiable.setter
    def differentiable(self, val: bool):
        self._differentiable = val

    def isfw(self) -> bool:
        return self._forward

    @property
    def forward(self) -> bool:
        """
        return True if this adapter serves in forward stage.
        """
        return self._forward

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

    def dispatch(self, devid: int, _mirror: bool = True):
        """
        Instantiate the adapter to a specific rank.

        @param devid int: device id

        @param adapter IRAdapter: the dispatched adapter
        """
        assert isinstance(devid, int), f"Expect devid to be int but got {devid}"
        if devid in self._cached_dispatch:
            return self._cached_dispatch[devid]
        prims = [prim.dispatch(devid) for prim in self.prims]
        prims = [prim for prim in prims if prim is not None]
        # get inputs
        inputs = []
        for itensor in self.inputs():
            if devid in itensor.device and itensor not in inputs:
                inputs.append(itensor) 
        outputs = []
        for otensor in self.outputs():
            if devid in otensor.device and otensor not in outputs:
                outputs.append(otensor)
        # insert identity prims
        if len(prims) == 0:
            assert all(otensor in inputs for otensor in outputs), \
                "output tensor not apear in input tensors for empty prims"
            for itensor in inputs:
                prims.append(IdentityPrim(itensor))
        # dispatch
        fadapter = IRAdapter(inputs, outputs)
        fadapter.prims = prims
        fadapter.name = self.name
        fadapter._id = self._id
        fadapter.differentiable = self.differentiable
        fadapter.custom = self.custom
        fadapter.recompute = self.recompute
        # dispatch for mirror
        if _mirror and isinstance(self.mirror, IRAdapter):
            badapter = self.mirror.dispatch(devid, _mirror=False)
            IRCell.make_pair(fadapter, badapter)
        self._cached_dispatch[devid] = fadapter
        return fadapter

    @staticmethod
    def merge(adapters: List):
        """!
        Merge adapters to one adapter
        """
        adapters : List[IRAdapter] = adapters
        assert all(isinstance(n, IRAdapter) for n in adapters)
        # TODO: check recompute consistency
        itensors = []
        otensors = []
        prims = []
        for adapter in adapters:
            itensors += adapter.inputs()
            otensors += adapter.outputs()
            prims += adapter.prims
        adapter = IRAdapter(itensors, otensors)
        adapter.prims = prims
        return adapter


    def __repr__(self):
        return f'Adapter-{self._id}{self.device}(inputs={self.inputs()}, outputs={self.outputs()})'

    def extra_repr(self) -> str:
        dscp = f'Adapter-{self._id}{self.device}(\n\tinputs={self.inputs()},\n\toutputs={self.outputs()}\n):'
        for prim in self.prims:
            dscp += '\n\t' + repr(prim)
        return dscp


class IRWeightReducer(IRCell):

    def __init__(self, weights: List[IRSubTensor], name='reducer'):
        if not all(isinstance(w, IRSubTensor) and w.is_param() for w in weights):
            raise RuntimeError("Expected a list of gradient IRSubTensor")
        signature = None
        super().__init__(name, signature, len(weights), 0)
        for idx, weight in enumerate(weights):
            self.set_input(idx, weight)

    def isfw(self) -> bool:
        return False
    
    def dispatch(self, device: int):
        return self

    def __repr__(self):
        dscp = f'WReducer{self._id}-{self.device}(inputs={self.inputs()})'
        return dscp

    def module_repr(self) -> str:
        return repr(self)
