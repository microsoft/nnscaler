#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, Iterator, List, Tuple, Optional
import copy
import numpy as np

from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.tensor import ValueMap


TShape = Tuple[int, ...]
TRVD = Tuple[int, ...]


class RVDLayout:
    """
    This class assumes a full-tensor can only be
    uniformly partitioned / replicated on dimensions and values.

    DNN clusters are usually equipped with homogeneous accelerator devices.
    Therefore, most parallelization plans partition operators evenly.
    Thus, a partition plan N-dim tensor layout can be simply represented as
    <R, V, dim1, ...,dimN>: R (replica), V (value), dim_i (dimension)

    which means:
     1) R(i), the tensor is replicated to i copies;
     2) V(j), value split, the tensor is decomposed to j copies with the same shape;
     3) D(k1,k2,...,kn), uniformly partition the tensor into k1 parts in
        the first dimension, k2 parts in the second dimension, so on
        so forth.

    We use RVD to denote the transformation of a tensor.
    For example, R(1)V(2)D(1,2) indicates a 2-D pTensor
    requires no replication, is decomposed into 2 vTensors with
    the same shape, and each is partitioned into 2 vTensors by
    partitioning the second axis.
    Thus, R(1)V(2)D(1,2) can represent 4 vTensors.

    RVD can represent both producer vTensors and consumer vTensors
    as they are both transformed from the pTensor.
    """

    def __init__(self, ftensor: IRFullTensor, subtensors: List[IRSubTensor], mats: np.ndarray):
        """
        ftensor: N-dim FullTensor
        subtensors: List[IRSubTensors]
        mats: Array[IRSubTensor]:
            (2+N)-dim matrix, with index respect to <R, V, dim1, ..., dimN>
        """
        self.ftensor = ftensor
        self.subtensors = subtensors
        self._mats = mats

    @property
    def R(self) -> int:
        return self._mats.shape[0]

    @property
    def V(self) -> int:
        return self._mats.shape[1]

    @property
    def D(self) -> Tuple[int]:
        return tuple(self._mats.shape[2:])

    @property
    def vec(self) -> Tuple[int]:
        return tuple(self._mats.shape)

    @property
    def ndims(self):
        return len(self._mats.shape)

    @property
    def ndevs(self):
        return len(self.subtensors)

    @property
    def mat(self):
        return self._mats

    def tensor(self, r: int, v: int, d: List[int]) -> IRSubTensor:
        """
        Get subtenor indexed by RVD position.
        """
        assert r <= self.R and v <= self.V and len(d) == len(self.D), "out of scope"
        indices = [r, v] + list(d)
        return self._mats[tuple(indices)]

    def __repr__(self):
        dscp = f'T{self.ftensor._id}<R({self.R}),V({self.V}),D({self.D})>'
        return dscp

    def __copy__(self):
        tensors = []
        for t in self.mat.flatten():
            tensor = copy.copy(t)
            tensor.cell = t.cell
            tensors.append(tensor)
        mat = np.array(tensors).reshape(self.mat.shape)
        return RVDLayout(self.ftensor, tensors, mat)

    def align(self, layout) -> bool:
        """
        Check whether the layout is same with self.

        The same means 1) sub-tenosrs are same 2) device are aligned

        @param layout RVDLayout

        @return same bool:
        """
        if not isinstance(layout, RVDLayout):
            return False
        tensors: List[IRSubTensor] = list(self.mat.flatten())
        for t in layout.mat.flatten():
            dev_match = False
            for idx in range(len(tensors)):
                t2 = tensors[idx]
                if t == t2 and set(t.device) == set(t2.device):
                    tensors.pop(idx)
                    dev_match = True
                    break
            if not dev_match: return False
        return True

    def inner_transpose(self, dim: int, chunks: int):
        """
        Transpose ordering of tensor within a dimension.
        The only goal is to shuffle the tensors (but RVD values are the same) in a dimension
        to try to find a better path.

        Currently only R abd V dim are using this function.
        If dim is 0 (R), then the tensor is shuffled in the first dimension.
            which means the dp units are shuffled.
            For example, we have 8 devices, and R=4, chunks=2, then
             before: devices of 0~3 replica: [0, 1], [2, 3], [4, 5], [6, 7]
             after: devices of 0~3 replica: [0, 1], [4, 5], [2, 3], [6, 7]
        If dim is 1 (V), we have similar behavior.
            For example, we have 8 devices, and R=1 V=4, chunks=2, then
             before: devices of 0~3 value partitions: [0, 1], [2, 3], [4, 5], [6, 7]
             after: devices of 0~3 value partitions: [0, 1], [4, 5], [2, 3], [6, 7]

        You can see after the shuffle, nothing is changed except the device assignment order.

        """
        assert 0 <= dim and dim < len(self._mats.shape)
        assert self.vec[dim] % chunks == 0
        ori_shape = list(self.vec)
        new_shape = list(self.vec)
        new_shape.insert(dim, self.vec[dim] // chunks)
        new_shape[dim+1] = chunks
        self._mats = self._mats.reshape(new_shape)
        axes = list(range(len(new_shape)))
        axes[dim], axes[dim+1] = axes[dim+1], axes[dim]
        self._mats = self._mats.transpose(axes)
        self._mats = self._mats.reshape(ori_shape)

    @staticmethod
    def dim2last(mat: np.ndarray, dim: int, chunk: int) -> np.ndarray:
        """
        Move the dimension that needs to be operated on to the last.
        So in the following operation we can operate on the last dimension, like
        ```
        for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
                prims.append(primitive(itensors, otensors))
        ```
        For example, if we want to transform R(1)V(2)D(1, 4) to R(1)V(1)D(1, 8).
        Essentially, we want to transform
            `imat[*, *, 0, *, *]` and `imat[*, *, 1, *, *]`
            to
            `omat[*, *, 0, *, *, 0] and `omat[*, *, 0, *, *, 1]`

            and reshape omat to R(1)V(1)D(1, 8)

        We don't bother to use a nested for loop, instead,
        we move the related dimension to the last, imat[*, *, V, *, *] -> imat[*, *, *, *, V]
        """
        shape = list(mat.shape)
        assert shape[dim] % chunk == 0
        shape[dim] = shape[dim] // chunk
        shape.insert(dim+1, chunk)
        mat = mat.reshape(shape)
        # move the axis to the last
        mat = np.moveaxis(mat, dim+1, -1)
        return mat

    @staticmethod
    def grid(ftensor: IRFullTensor, r: int, v: int, dims: Tuple[int], devices: Optional[Tuple[int, ...]] = None):
        """
        partition a ftensor using grid layout of <r, v, *dims>

        For device assignment, if devices is not None, assign devices in order.
        For example, you have 8 devices, and r=2, v=2, dims=(1, 2) Then
        1. Split devices into r groups, which mean the outmost is data parallelism.
           So (0, 1, 2, 3) is a sub group, and (4, 5, 6, 7) is another sub group
           These two sub groups are replicated.
        2. Split devices in each r-group into v groups.
           V is for value parallelism.
           When V > 1, the value is partitioned.
           That happens when previous forward op splits reducer dimention (the `+` in dimop annoation).
           For the example above, (0, 1, 2, 3) will be splitted into (0, 1) and (2, 3)
        3. Split devices in each v-group into dims groups. It is tensor parallelism,
           and is the innermost.
           So (0, 1) is splitted into (0,) and (1,)

        Please note that is not the only way to assign devices. But it is our best guess.
        `.inner_transpose()` can be used to shuffle the tensor within a dimension,
        and hope to find a match for devices

        TODO: We need to support more flexible device assignment.
        """
        dims = tuple(dims)
        def dummy_assign(tensor: IRSubTensor, devid: int):
            tensor.cell = IRCell('dummy', '', 0, 0)
            tensor.cell.device = devid

        mats = np.empty((r, v) + dims, dtype=IRSubTensor)
        all_subtensors = []

        def iter_idx(dims: List[int]) -> Iterator[Tuple[int, ...]]:
            if len(dims) == 0:
                yield ()
            else:
                for i in range(dims[0]):
                    for indices in iter_idx(dims[1:]):
                        yield (i,) + indices
        # generate tensor for each index
        for indices in iter_idx((v,)+dims):
            valmap = ValueMap((indices[0], v))
            indmap = []
            shape = []
            for dim, (nchunk, index) in enumerate(zip(dims, indices[1:])):
                assert ftensor.shape[dim] % nchunk == 0, f"not dividable for {nchunk} chunks over dim {dim}. ftensor shape: {ftensor.shape}"
                csize = ftensor.shape[dim] // nchunk
                start = csize * index
                indmap.append((start, start+csize))
                shape.append(csize)
            subtensor = ftensor.select(tuple(indmap), valmap)
            # replicate
            subtensors = [copy.copy(subtensor) for _ in range(r)]
            all_subtensors += subtensors
            mats[(slice(None),)+indices] = np.array(subtensors, dtype=IRSubTensor)

        # devices
        if devices is not None:
            assert len(devices) == len(all_subtensors), f"devices number {len(devices)} not match with RVD number {len(all_subtensors)}"
            for tensor, devid in zip(mats.flatten(), devices):
                dummy_assign(tensor, int(devid))

        return RVDLayout(ftensor, all_subtensors, mats)

    @staticmethod
    def togrid(ftensor: IRFullTensor, subtensors: List[IRSubTensor]):
        """
        Convert ftensor and subtensors into a RVDLayout.
        Here we requires all subtensors are well formed, and can be organized as R(...)V(...)D(...) format.

        Please note the devices are kept as it is, and may be different with how `.grid()` assigns the devices.

        Args:
            ftensor (IRFullTensor): full tensor
            subtensors (List[IRSubTensor]): subtensors of the full tensor.
        Returns:
            RVDLayout: rvd layout
        Raises:
            RuntimeError: if subtensors are not well formed.
        """
        _replica: int = None
        _value: int = None
        _dims: List[int] = [None] * len(ftensor.shape)
        # id(subtensor) -> [replica_index, value_index, dim1_index, dim2_index, ...]
        # Plese note key is not subtensor.id, but id(subtensor)
        _tindex: Dict[int, List[int]] = dict()

        ndims = len(ftensor.shape)

        # Key: subtensor id
        # Please note subtensors with same indmap and valmap have the same tid.
        # which indicates they are replicated.
        replicas: Dict[int, List[IRSubTensor]] = dict()
        vchunks: set = set()
        dchunks: List[set] = [set() for _ in range(ndims)]

        for subtensor in subtensors:
            oid = id(subtensor)
            # set up replica
            if subtensor.tid not in replicas:
                replicas[subtensor.tid] = []
            _tindex[oid] = [len(replicas[subtensor.tid])]
            replicas[subtensor.tid].append(subtensor)
            # setup value
            _tindex[oid].append(subtensor.valmap[0])
            vchunks.add(subtensor.valmap[1])
            # setup dimensions
            for dim in range(ndims):
                snele = subtensor.shape[dim]
                start = subtensor.indmap[dim][0]
                fnele = ftensor.shape[dim]
                if fnele % snele != 0 or start % snele != 0:
                    raise RuntimeError(
                        f"dimension split error:\n"
                        f"Full Tensor: {ftensor}\n"
                        f"full nele: {fnele}, sub nele: {snele}, start: {start}"
                    )
                dchunks[dim].add(fnele // snele)
                _tindex[oid].append(start // snele)
        # replica (R)
        nreplicas = set(len(ts) for ts in replicas.values())
        if len(nreplicas) != 1:
            raise RuntimeError(f"different replicas: {nreplicas}")
        _replica = list(nreplicas)[0]
        # value (V)
        nchunks = set(t.valmap[1] for t in subtensors)
        if len(nchunks) != 1:
            raise RuntimeError(f"different value split: {nchunks}")
        _value = list(nchunks)[0]
        # dimension (D)
        for dim in range(ndims):
            if len(dchunks[dim]) != 1:
                raise RuntimeError(f"different dimension split: {dchunks[dim]}")
            _dims[dim] = list(dchunks[dim])[0]

        # set matrix
        mats = np.empty([_replica, _value] + _dims, dtype=IRSubTensor)
        for subtensor in subtensors:
            idx = tuple(_tindex[id(subtensor)])
            assert mats[idx] is None, f"repeating entry. mutiple same {subtensor}"
            mats[tuple(idx)] = subtensor
        assert not (mats == None).any(), "at least one entry not set"
        return RVDLayout(ftensor, subtensors, mats)


class RVDInspector:

    @staticmethod
    def draw(prvd: RVDLayout, crvd: RVDLayout, outfile: str) -> None:
        """
        Draw producer RVDLayout and consumer RVDLayout
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib.axes

        max_dev = max(
            max(t.device[0] for t in prvd.subtensors), max(t.device[0] for t in crvd.subtensors)
        )
        min_dev = min(
            min(t.device[0] for t in prvd.subtensors), min(t.device[0] for t in crvd.subtensors)
        )
        devlen = max_dev - min_dev
        plt.close('all')
        plt.rcParams['figure.figsize'] = (4.0 * devlen, 7.0)
        fig, ax = plt.subplots()
        ax: matplotlib.axes.Axes

        fontsize = 30

        ax.set_xlim((-0.5, devlen+0.5))
        ax.set_ylim((0, 5))

        ptensors = prvd.mat.flatten().tolist()
        ctensors = crvd.mat.flatten().tolist()

        recflen = 0.8
        def draw_subtensor(t: IRSubTensor, xy: Tuple[int], color: str):
            assert len(t.shape) == 2, "Only able to draw 2-D tensor"
            x, y = xy
            # full tensor
            rec = Rectangle(xy, recflen, recflen, color='white', ec='black', lw=2.0)
            # sub tensor
            subx_nchunks = t.parent.shape[1] // t.shape[1]
            subw = recflen / subx_nchunks
            subx = x + subw * (t.indmap[1][0] // t.shape[1])

            suby_nchunks = t.parent.shape[0] // t.shape[0]
            subh = recflen / suby_nchunks
            suby = y + subh * (t.indmap[0][0] // t.shape[0])

            # if t.valmap != (0, 1):
            ax.text(x=x+recflen/2, y=y+recflen+recflen/2, s=f'val({t.valmap[0]}/{t.valmap[1]})',
                    fontsize=fontsize, ha='center', va='center', color='black')

            subrec = Rectangle((subx, suby), subw, subh, color=color, ec='black', lw=2.0)
            ax.add_artist(rec)
            ax.add_artist(subrec)

        for ptensor in ptensors:
            x, y = ptensor.device[0]-min_dev-0.4, 3
            draw_subtensor(ptensor, (x, y), 'blue')

        ax.text(x=-1, y=3+recflen/2, s='Producer',
                fontsize=fontsize, ha='center', va='center', color='black')

        for ctensor in ctensors:
            x, y = ctensor.device[0]-min_dev-0.4, 0.5
            draw_subtensor(ctensor, (x, y), 'orange')

        ax.text(x=-1, y=0.5+recflen/2, s='Consumer',
                fontsize=fontsize, ha='center', va='center', color='black')

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label2.set_fontsize(fontsize)

        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        ax.get_yaxis().set_visible(False)
        plt.savefig(outfile)

