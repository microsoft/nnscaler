#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
The primitive used for IRAdapter
"""

from typing import List, Optional, Union, Tuple
import copy

from nnscaler.ir.tensor import IRSubTensor, IndexMap, ValueMap


# the general adapter primitive class
class IRAdapterPrim:

    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor], **kwargs):
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        self._device = []
        self.kwargs = dict()
        for arg, val in kwargs.items():
            self.kwargs[arg] = val
        self.signature = None
        # whether the primitive is happened locally
        self.local: bool = False

    def input(self, idx:int):
        return self._inputs[idx]

    def inputs(self):
        return copy.copy(self._inputs)

    def output(self, idx:int):
        return self._outputs[idx]

    def outputs(self):
        return copy.copy(self._outputs)

    def dispatch(self, devid: int):
        if devid not in self.device:
            return None
        return self

    def volume(self) -> int:
        """
        Communication volume of the primitive. The total elements
        transferred in the network.

        @return nele int: the number of elements go through network
        """
        raise NotImplementedError("The communication cost is not implemented")

    @property
    def device(self) -> List[int]:
        return copy.copy(self._device)

    @device.setter
    def device(self, devs: Union[int, List[int]]):
        if isinstance(devs, int):
            devs = [devs]
        self._device = devs

# spatial abstract primitive
class SpatialPrim(IRAdapterPrim):
    """
    basic class for representing spatial primitives
    """
    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor], **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.device = list(set(t.device[0] for t in inputs))
        self.local = True

    def volume(self) -> int:
        return 0


# numerical abstract primitive
class ValuePrim(IRAdapterPrim):
    """
    basic class for representing numerical primitives
    """
    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__(inputs, outputs)
        self.device = list(set(t.device[0] for t in inputs))
        self.local = True

    def volume(self) -> int:
        return 0


# communication abstract primitive
class CommPrim(IRAdapterPrim):
    """
    communication primitive
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        devices = []
        for t in list(itensors) + list(otensors):
            devices += t.device
        self.device = list(set(devices))
        self.local = False

    def dispatch(self, devid: int) -> Optional[IRAdapterPrim]:
        """
        dispatch to a given device
        """
        if devid not in self.device:
            return None
        assert devid in self.device, f"device {devid} not applied for this comm primitive"
        itensors = [itensor for itensor in self.inputs() if devid in itensor.device]
        otensors = [otensor for otensor in self.outputs() if devid in otensor.device]
        prim = type(self)(itensors, otensors, **self.kwargs)
        prim.signature = self.signature
        return prim

    def __repr__(self) -> str:
        dscp = f'{self.outputs()} = {self.signature}({self.inputs()})'
        return dscp

# ======================================================

class IdentityPrim(SpatialPrim):

    def __init__(self, itensor: IRSubTensor):
        super().__init__([itensor], [itensor])
        self.signature = 'nnscaler.runtime.adapter.identity'

    def __repr__(self):
        dscp = f"{self.output(0)} = identity({self.input(0)})"
        return dscp


class SelectPrim(SpatialPrim):

    def __init__(self,
                 itensor: IRSubTensor,
                 indmap: IndexMap, valmap: ValueMap,
                 otensor: IRSubTensor):
        indmap = IndexMap(indmap).indices
        indmap = tuple(slice(s, e) for s, e in indmap)
        valmap = ValueMap(valmap).weight[1]
        super().__init__([itensor], [otensor], indmap=indmap, valmap=valmap)
        self.signature = f"nnscaler.runtime.adapter.select"

    def __repr__(self):
        dscp = f"{self.output(0)} = select({self.input(0)}, indmap={self.kwargs['indmap']}, valmap={self.kwargs['valmap']})"
        return dscp


class MergeDimPrim(SpatialPrim):
    """
    concatenate dimension
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor, dim: int) -> None:
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor], dim=dim)
        self.signature = 'nnscaler.runtime.adapter.smerge'

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.output(0)} = concat({self.inputs()}, dim={self.kwargs['dim']})"

# numerical primitive

class SumPrim(ValuePrim):

    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor):
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor])
        self.signature = 'nnscaler.runtime.adapter.vmerge'

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.output(0)} = add({self.inputs()})"


# communication primitive

class MovePrim(CommPrim):
    """
    P2P send/recv, non-differentiable
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        if len(kwargs) == 0:
            assert len(itensors) == 1 and len(otensors) == 1
            kwargs['shape'] = itensors[0].origin_shape
            kwargs['dtype'] = str(itensors[0].dtype)
            kwargs['src'] = itensors[0].device[0] if len(itensors[0].device) > 0 else None
            kwargs['dst'] = otensors[0].device[0] if len(otensors[0].device) > 0 else None
        shape, dtype, src, dst = kwargs['shape'], kwargs['dtype'], kwargs['src'], kwargs['dst']
        super().__init__(itensors, otensors, shape=shape, dtype=dtype, src=src, dst=dst)
        self.signature = 'nnscaler.runtime.adapter.move'

    def volume(self) -> int:
        if len(self._inputs) > 0:
            return self.input(0).nelement()
        return self.output(0).nelement()

    def __repr__(self):
        dscp = f"{self.outputs()} = move{self.device}({self.inputs()}, src={self.kwargs['src']}, dst={self.kwargs['dst']})"
        return dscp


class CollectivePrim(CommPrim):
    """
    Collective primitive, non-differentiable
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        if 'ranks' not in self.kwargs:
            self.kwargs['ranks'] = self.device


class RDScatterPrim(CommPrim):
    """
    P2P Cross-device dimension scatter, non-differentiable.

    Tensor[Tile0, Tile1]: device 0 -> Tensor[Tile0]: device0, Tensor[Tile1]: device1
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        """
        @param itensors List[IRSubTensor]: one tensor at device of `src`.
        @param otensors List[IRSubTensor]: each ran hosts one tenor partitioned by dim.
        @param dim int: the dimension that itensor will be partitioned
        """
        if len(kwargs) == 0:
            assert len(itensors) == 1
            kwargs['shape'] = itensors[0].origin_shape
            kwargs['dtype'] = str(itensors[0].dtype)
            kwargs['src'] = itensors[0].device[0] if len(itensors[0].device) > 0 else None
            kwargs['dsts'] = tuple(otensor.device[0] if len(otensor.device) > 0 else None for otensor in otensors)
        shape, dtype, src, dsts = kwargs['shape'], kwargs['dtype'], kwargs['src'], kwargs['dsts']
        super().__init__(itensors, otensors, shape=shape, dtype=dtype, dim=dim, src=src, dsts=dsts)
        self.signature = 'nnscaler.runtime.adapter.rdscatter'

    def volume(self) -> int:
        return self.input(0).nelement()

    def __repr__(self) -> str:
        inputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.inputs())
        outputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.outputs())
        return f"{outputs} = rdscatter{self.device}({inputs}, dim={self.kwargs['dim']})"


class RVScatterPrim(CollectivePrim):
    """
    P2P Cross-device dimension scatter, non-differentiable.

    Tensor[Tile0, Tile1]: device 0 -> Tensor[Tile0]: device0, Tensor[Tile1]: device1
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        """
        @param itensors List[IRSubTensor]: one tensor at device of `src`.
        @param otensors List[IRSubTensor]: each ran hosts one tenor partitioned by dim.
        @param dim int: the dimension that itensor will be partitioned
        """
        if len(kwargs) == 0:
            assert len(itensors) == 1
            kwargs['shape'] = itensors[0].origin_shape
            kwargs['dtype'] = str(itensors[0].dtype)
            kwargs['src'] = itensors[0].device[0] if len(itensors[0].device) > 0 else None
            kwargs['dsts'] = tuple(otensor.device[0] if len(otensor.device) > 0 else None for otensor in otensors)
        shape, dtype, src, dsts = kwargs['shape'], kwargs['dtype'], kwargs['src'], kwargs['dsts']
        super().__init__(itensors, otensors, shape=shape, dtype=dtype, src=src, dst=dsts)
        self.signature = 'nnscaler.runtime.adapter.rvscatter'

    def volume(self) -> int:
        return self.input(0).nelement() * len(self.outputs())

    def __repr__(self) -> str:
        inputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.inputs())
        outputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.outputs())
        return f"{outputs} = rvscatter{self.device}({inputs})"


class RDGatherPrim(CommPrim):
    """
    Gather tensors from remote devices to a local device.
    The local device doesn't have any tensor
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        if len(kwargs) == 0:
            assert len(otensors) == 1
            kwargs['shape'] = itensors[0].origin_shape
            kwargs['dtype'] = str(itensors[0].dtype)
            kwargs['srcs'] = tuple(itensor.device[0] if len(itensor.device) > 0 else None for itensor in itensors)
            kwargs['dst'] = otensors[0].device[0] if len(otensors[0].device) > 0 else None
        shape, dtype, srcs, dst = kwargs['shape'], kwargs['dtype'], kwargs['srcs'], kwargs['dst']
        super().__init__(itensors, otensors, shape=shape, dtype=dtype, srcs=srcs, dst=dst, dim=dim)
        self.signature = 'nnscaler.runtime.adapter.rdgather'

    def volume(self) -> int:
        return self.output(0).nelement()

    def __repr__(self) -> str:
        inputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.inputs())
        outputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.outputs())
        return f"{outputs} = rdgather{self.device}({inputs}, dim={self.kwargs['dim']})"


class RVGatherPrim(CollectivePrim):
    """
    Gather tensors from remote devices and sum in the local device.
    The local device doesn't have any tensor
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        if len(kwargs) == 0:
            assert len(otensors) == 1
            kwargs['shape'] = itensors[0].origin_shape
            kwargs['dtype'] = str(itensors[0].dtype)
            kwargs['srcs'] = tuple(otensor.device[0] if len(otensor.device) > 0 else None for otensor in otensors)
            kwargs['dst'] = otensors[0].device[0] if len(otensors[0].device) > 0 else None
        shape, dtype, srcs, dst = kwargs['shape'], kwargs['dtype'], kwargs['srcs'], kwargs['dst']
        super().__init__(itensors, otensors, shape=shape, dtype=dtype, srcs=srcs, dst=dst)
        self.signature = 'nnscaler.runtime.adapter.rvgather'

    def volume(self) -> int:
        return self.output(0).nelement() * len(self.inputs())

    def __repr__(self) -> str:
        inputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.inputs())
        outputs = ', '.join(f'{t.name}{t.tid}{t.shape}{t.valmap}' for t in self.outputs())
        return f"{outputs} = rvgather{self.device}({inputs})"


class BroadcastPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        if len(kwargs) == 0:
            assert len(itensors) == 1
            kwargs['shape'] = itensors[0].origin_shape
            kwargs['dtype'] = str(itensors[0].dtype)
            kwargs['src'] = itensors[0].device[0] if len(itensors[0].device) > 0 else None
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.broadcast'

    def volume(self) -> int:
        ndevs = len(self.outputs())
        return self.input(0).nelement() * (ndevs-1)

    def __repr__(self) -> str:
        return f"{self.outputs()} = broadcast{self.device}({self.inputs()}, src={self.kwargs['src']})"



class AllReducePrim(CollectivePrim):
    """
    non-differentiable allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.all_reduce'

    def volume(self) -> int:
        """
        Use ring-allreduce communication cost
        """
        ndevs = len(self.inputs())
        return 2 * (ndevs - 1) * self.input(0).nelement() // ndevs

    def __repr__(self) -> str:
        return f'{self.outputs()} = all_reduce[{self.device}]({self.inputs()})'


class AllGatherPrim(CollectivePrim):
    """
    non-differentiabl all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim=dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.all_gather'

    def volume(self) -> int:
        """
        Use ring-based communication cost
        """
        ndevs = len(self.inputs())
        return (ndevs - 1) * self.input(0).nelement()

    def __repr__(self) -> str:
        return f'{self.outputs()} = all_gather[{self.device}]({self.inputs()})'


class ReduceScatterPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim=dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.reduce_scatter'

    def volume(self) -> int:
        """
        Use ring-based communication cost
        """
        ndevs = len(self.inputs())
        # FIXME: temporally disable reduce scatter in code generation
        # which has parity issues for now.
        return 100 * (ndevs - 1) * self.input(0).nelement() // ndevs

    def __repr__(self) -> str:
        return f'{self.outputs()} = reduce_scatter[{self.device}]({self.inputs()})'


class ReducePrim(CollectivePrim):
    """
    non-differential reduce prim
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor, **kwargs):
        super().__init__(itensors, [otensor], dst=otensor.device[0], **kwargs)
        self.signature = 'nnscaler.runtime.adapter.reduce'

    def volume(self) -> int:
        ndevs = len(self.inputs())
        return self.input(0).nelement() * ndevs

    def __repr__(self) -> str:
        return f"{self.outputs()} = reduce[{self.device}]({self.inputs()}, dst={self.kwargs['dst']})"


class AllToAllPrim(CollectivePrim):
    """
    non-differentiable all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], idim: int, odim: int, **kwargs):
        """
        itensors: each rank hosts one tensor splitted by idim
        otensors: each rank hosts one tensor splitted by odim
        idim != odim
        """
        super().__init__(itensors, otensors, idim=idim, odim=odim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.all_to_all'

    def volume(self) -> int:
        ndevs = len(self.inputs())
        return self.input(0).nelement() * (ndevs - 1) // ndevs

    def __repr__(self) -> str:
        return f"{self.outputs()} = all_to_all[{self.device}]({self.inputs()}, idim={self.kwargs['idim']}, odim={self.kwargs['odim']})"


class ChunkPrim(CollectivePrim):
    """
    split dimension in n chunks and take idx-th chunk
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim=dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.chunk'

    def volume(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"{self.outputs()} = split[{self.device}]({self.inputs()}, dim={self.kwargs['dim']})"


class VChunkPrim(CollectivePrim):
    """
    split value in n chunks and take idx-th chunk
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.vchunk'

    def volume(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"{self.outputs()} = vsplit[{self.device}]({self.inputs()})"


class AllReduceIdentityPrim(AllReducePrim):
    """
    forward: allreduce.
    backward: identity
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.allreduce_identity'

    def __repr__(self) -> str:
        return f"{self.outputs()} = allreduce_identity[{self.device}]({self.inputs()})"


class IdentityAllreducePrim(AllReducePrim):
    """
    forward: identity
    backward: allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.identity_allreduce'

    def __repr__(self) -> str:
        return f"{self.outputs()} = identity_allreduce[{self.device}]({self.inputs()})"


class AllReduceAllReducePrim(AllReducePrim):
    """
    forward: allreduce
    backward: allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.allreduce_allreduce'

    def __repr__(self) -> str:
        return f"{self.outputs} = nn.allreduce_allreduce[{self.device}]({self.inputs()}"


class ReduceScatterAllGatherPrim(ReduceScatterPrim):
    """
    forward: reduce-scatter
    backward: all-gather
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.reducescatter_allgather'


class AllGatherReduceScatterPrim(AllGatherPrim):
    """
    forward: all-gather
    backward: reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.allgather_reducescatter'


class AllGatherSplitPrim(AllGatherPrim):
    """
    forward: all-gather
    backward: split
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.allgather_split'


class SplitAllGatherPrim(AllGatherPrim):
    """
    forward: split
    backward: all-gather
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.split_allgather'


class AllToAllAllToAllPrim(AllToAllPrim):
    """
    forward: all-to-all
    backward: all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], idim: int, odim: int, **kwargs):
        super().__init__(itensors, otensors, idim, odim, **kwargs)
        self.signature = 'nnscaler.runtime.adapter.nn.alltoall_alltoall'


class ReduceBroadcastPrim(CollectivePrim):
    """
    forward: broadcast
    backward: reduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dst: int, **kwargs):
        super().__init__(itensors, otensors, dst=dst, **kwargs)


class BroadcastRedducePrim(CollectivePrim):
    """
    forward: broadcast
    backward: reduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], src: int, **kwargs):
        super().__init__(itensors, otensors, src=src, **kwargs)
