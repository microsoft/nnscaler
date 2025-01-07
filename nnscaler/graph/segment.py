#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import contextmanager
from typing import Dict, Union, List, Optional, Set, Tuple, Any, Callable
import numpy as np
import logging

from nnscaler.ir.tensor import IRFullTensor, IRSubTensor, ValueMap
from nnscaler.ir.cten import IRTensor, IRCell, IRObject
from nnscaler.ir.operator import IRFwOperation, IRBpOperation
from nnscaler.ir.adapter import IRAdapter

from nnscaler.graph.function.function import MultiRef
from nnscaler.graph.function.pyfunc import IRPyFunc


_logger = logging.getLogger(__name__)


class CellPosition:

    def __init__(self, indices: Tuple[int]):
        assert all(isinstance(idx, int) for idx in indices) and len(indices) > 0
        self.indices = tuple(indices)

    def __hash__(self) -> int:
        return hash(self.indices)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, CellPosition), "Cannot compare with non-GraphIndex object"
        return self.indices == other.indices

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, CellPosition), "Cannot compare with non-GraphIndex object"
        if len(self.indices) < len(other.indices):
            return True
        if len(self.indices) > len(other.indices):
            return False
        for lidx, ridx in zip(self.indices, other.indices):
            if lidx >= ridx:
                return False
        return True

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        return not self <= other

    def __ge__(self, other: object) -> bool:
        return not self < other

    def __sub__(self, offset: int):
        assert isinstance(offset, int)
        indices = list(self.indices)
        indices[-1] -= offset
        return CellPosition(indices)

    def __add__(self, offset: int):
        assert isinstance(offset, int)
        indices = list(self.indices)
        indices[-1] += offset
        return CellPosition(indices)

    def __getitem__(self, idx: int) -> int:
        return self.indices[idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return repr(self.indices)


class IRSegment(IRCell):
    """
    A distributed sub-graph representing a piece of workload in parent IRGraph

    Input/output can be complex data type of Dict, List, Tuple on IRObjects

    Once the segment is generated, its input and output will be fixed.
    Inserting and removing nodes that could change input/output are not allowed.
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRObject], outputs: List[Any], name='segment'):
        super().__init__(name, '', len(inputs), len(outputs))

        self._nodes: List[IRCell] = []

        # full objects
        self._fobjects: Set[IRObject] = set()
        self._producers: Dict[IRFullTensor, List[IRCell]] = dict()
        self._consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        self._ptensors: Dict[IRFullTensor, List[IRSubTensor]] = dict()
        self._ctensors: Dict[IRFullTensor, List[IRSubTensor]] = dict()

        # attributes
        self._attributes: Set[IRFullTensor] = set()

        for idx, val in enumerate(inputs):
            self.set_input(idx, val)
        for idx, val in enumerate(outputs):
            self.set_output(idx, val)

        for t in IRSegment.get_objects_from_complex(list(inputs) + list(outputs)):
            self._add_ftensor(t.parent)

        for node in nodes:
            self.insert(node, self.nnodes)

        self._dispatch_cached: Dict[int, IRSegment] = {}

    def set_input(self, idx: int, val: Any):
        for t in IRSegment.get_objects_from_complex(val):
            self._add_ftensor(t.parent)
        return super().set_input(idx, val)

    def set_output(self, idx: int, val: Any):
        for t in IRSegment.get_objects_from_complex(val):
            self._add_ftensor(t.parent)
        return super().set_output(idx, val)

    def isfw(self) -> bool:
        return all(n.isfw() for n in self._nodes)
        # return self._have_forward

    def full_objects(self) -> Tuple[IRObject]:
        """Get all full objects of this graph.

        Note:
            The full tensor inside the node (e.g., IRSegment) will not be returned.

        Returns:
            fobjects List[IRObject]
        """
        return tuple(self._fobjects)

    def full_tensors(self) -> Tuple[IRFullTensor]:
        """
        Get all full tensors of this graph.
        Note the full tensor inside the node will not be returned.

        @return ftensors List[IRFullTensor]
        """
        return tuple(t for t in self._fobjects if isinstance(t, IRFullTensor))

    def attributes(self) -> Tuple[IRFullTensor]:
        """
        Get al full tensor attributes of this graph
        Note the full tensor inside the node will not be returned.

        @return ftensors List[IRFullTensor]
        """
        return tuple(self._attributes)

    # ========================= Basic Graph access =======================

    @property
    def device(self) -> List[int]:
        devices = set()
        for node in self._nodes:
            devices.update(node.device)
        devices = list(devices)
        devices.sort()
        return devices

    @property
    def nnodes(self) -> int:
        """
        Get total node number

        @return number int: the number of nodes
        """
        return len(self._nodes)

    def nodes(self, flatten = False) -> Tuple[IRCell]:
        """
        Get all the nodes.

        @param flatten bool: Flat the segment to get all the nested cells

        @return nodes List[IRCell]: all the nodes
        """
        if not flatten:
            return tuple(self._nodes)
        nodes = []
        for node in self._nodes:
            if not isinstance(node, IRSegment):
                nodes.append(node)
            else:
                nodes += list(node.nodes(flatten))
        return tuple(nodes)

    def node(self, index: Union[int, CellPosition]) -> IRCell:
        """
        Get node at position index

        @param index Union[int, CellPosition]: the node index

        @return node IRCell: the node.
        """
        pos = CellPosition((index,)) if isinstance(index, int) else index
        assert isinstance(pos, CellPosition), "Expect index to be int or CellPosition"
        node = self
        for idx in pos.indices:
            assert isinstance(node, IRSegment), "idx applies on a non-segment node"
            node = node._nodes[idx]
        return node

    def index(self, node: IRCell) -> CellPosition:
        """
        Get node index. The dispatched node (e.g., IRAdapter, IRSegment)
        will return the index to its un-dispatched node

        @param node IRCell: the queried node

        @return index int: the index
        """
        assert isinstance(node, IRCell)
        if node in self._nodes:
            return CellPosition((self._nodes.index(node),))
        for idx, segment in enumerate(self._nodes):
            if isinstance(segment, IRSegment):
                if segment.exist(node):
                    index = segment.index(node)
                    return CellPosition((idx,) + index.indices)
        raise KeyError(f"The queried node: {node} not in the graph")

    def multi_index(self, nodes: List[IRCell]) -> List[CellPosition]:
        """
        Get multiple node indices, traversing the graph only once
        to save time.

        Args:
            nodes (List[IRCell]): nodes to be indexed

        Returns:
            List[CellPosition]: indices of nodes
        """
        visited = 0
        indices = [None] * len(nodes)
        def dfs(seg: IRSegment, path: List[int]):
            nonlocal visited, indices
            for idx, node in enumerate(seg._nodes):
                if node in nodes:
                    indices[nodes.index(node)] = CellPosition(tuple(path + [idx]))
                    visited += 1
                if visited == len(nodes):
                    return
                if isinstance(node, IRSegment):
                    dfs(node, path + [idx])
        dfs(self, [])
        if visited != len(nodes):
            unvisited = []
            for idx, node in zip(indices, nodes):
                if idx is None:
                    unvisited.append(node)
            raise RuntimeError(f"Some of the queried nodes: {unvisited} not in the graph")
        return indices

    def segment(self, node: IRCell) -> IRCell:
        """
        Get the lowest segment that constains the node

        @param node IRCell: the queried node

        @return segment IRSegment
        """
        assert isinstance(node, IRCell), f"Expected IRCell, but got {node}"
        index = self.index(node)
        if len(index) == 1:
            return self
        else:
            return self.node(CellPosition(index.indices[:-1]))

    def producers(self, ftensor: IRFullTensor) -> Tuple[IRCell]:
        """
        Get producers of ftensor in execution order in this graph

        @param ftensor IRFullTensor: the queried full tensor.

        @return subtensors Tuple[IRSubTensor]: the producers.
        """
        return tuple(self._producers.get(ftensor, ()))

    def consumers(self, ftensor: IRFullTensor) -> Tuple[IRCell]:
        """
        Get consumers of ftensor in execution order in this graph

        @param ftensor IRFullTensor: the queried full tensor.

        @return subtensors Tuple[IRCell]: theconsumers
        """
        return tuple(self._consumers.get(ftensor, ()))

    def ptensors(self, ftensor: IRFullTensor) -> Tuple[IRSubTensor]:
        """Get produced sub-tensors of a full tensor (ftensor).

        A full tensor (ftensor) is originally produced by some operator(s).
        These operators can be further partitioned into multiple sub-operators.
        Each sub-operator potentially produces a smaller part of the ftensor (a.k.a. sub-tensor).
        This function returns all the sub-tensors that are produced by operators
        inside the segment.

        Args:
            ftensor (IRFullTensor): the queried full tensor.

        Returns:
            Tuple[IRSubTensor]: the produced sub-tensors.
        """
        return tuple(self._ptensors.get(ftensor, ()))

    def ctensors(self, ftensor: IRFullTensor) -> Tuple[IRSubTensor]:
        """Get consumed sub-tensors of a full tensor (ftensor)

        A full tensor (ftensor) is originally consumed by some operator(s).
        These operators can be further partitioned into multiple sub-operators.
        Each sub-operator potentially consumes a smaller part of the ftensor (a.k.a. sub-tensor).
        This function returns all the sub-tensors that are consumed by operators
        inside the segment.

        Args:
            ftensor (IRFullTensor): the queried full tensor.

        Returns:
            Tuple[IRSubTensor]: the consumed sub-tensors.
        """
        return tuple(self._ctensors.get(ftensor, ()))

    def infer_grad(self, ftensor: IRFullTensor) -> None:
        """
        Set gradient on sub-tensors of a fulltensor

        Note this can only be called when no operator transformation is
        applied for this graph.

        If a tensor is consumed by multiple consumers, the value map of its gradient
        will be in exponential format.

        E.g., t has consumed by node1, node2, node3 and node4.
        Then the gradient value_map of t (t.grad) of each consumer is (idx, nchunks):
            (0, 2), (2, 4), (6, 8), (7, 8),
        where:
              (0, 2) + (2, 4) + (6, 8) + (7, 8)
            = (0, 2) + (2, 4) + (3, 4)
            = (0, 2) + (1, 2)
            = FULL VALUE

        @param ftensor IRFullTensor: the full tensor.

        @return None: gradient are set to producer/consumer tensor's .grad
        """
        # check condition: no transformation
        assert len(self.producers(ftensor)) <= 1, (
            f"grad can only be set when no transformation is applied but got:\n"
            f"{self.debug_tensor_map_str(ftensor)}"
        )
        assert len(set(self.ctensors(ftensor))) <= 1, (
            f"grad can only be set when no transformation is applied but got:\n"
            f"{self.debug_tensor_map_str(ftensor)}"
        )

        fgrad = ftensor.grad
        # set for producer
        for ptensor, producer in zip(self.ptensors(ftensor), self.producers(ftensor)):
            # filter out non-autograd operators of IRPyFunc
            if isinstance(producer, IRPyFunc):
                if fgrad is not None:
                    msg = f'nnScaler does not support backward of IRPyFunc: {producer}, ' + \
                           'skip setting gradient, please register it as IRDimOps.'
                    _logger.warning(msg)
                continue
            grad = None if fgrad is None else fgrad.select(ptensor.indmap, (0, 1))
            for t in producer.find(ptensor):
                t.grad = grad

        # set for consumers
        # We strictly follow the `requires_grad` in the fx graph in most cases. However, we will
        # ignore the gradient when the corresponding subtensor is consumed by a IRPyFunc, since
        # nnScaler will not generate a backward node for IRPyFunc currently (check IRGraph.backward).
        consumers, ctensors = [], []
        for ctensor, consumer in zip(self.ctensors(ftensor), self.consumers(ftensor)):
            itensors = consumer.find(ctensor)
            # set by default None
            for itensor in itensors:
                itensor.grad = None
            if isinstance(consumer, IRPyFunc):
                continue
            # filter out non-autograd operators
            if fgrad is None: continue
            if isinstance(consumer, IRPyFunc): continue
            if any(isinstance(t, IRSubTensor) and t.requires_grad for t in consumer.outputs()):
                consumers.append(consumer)
                ctensors.append(ctensor)

        # set with value map
        curr_valmap = ValueMap((0, 1))
        nconsumers = len(consumers)
        for cidx, (ctensor, consumer) in enumerate(zip(ctensors, consumers)):
            valmap = curr_valmap.map((0, 2)) if cidx != nconsumers - 1 else curr_valmap
            grad = fgrad.select(ctensor.indmap, valmap)
            curr_valmap = curr_valmap.map((1, 2)) if cidx != nconsumers - 1 else curr_valmap
            for t in consumer.find(ctensor):
                t.grad = grad

    def debug_tensor_map_str(self, ftensor: Optional[IRFullTensor] = None) -> str:
        dscp : str = ''
        ftensors = [ftensor] if ftensor is not None else self._fobjects
        for ftensor in ftensors:
            dscp += f'====\nFull Tensor: {ftensor}\n'
            dscp += f'Producers:\n'
            for producer in self._producers[ftensor]:
                dscp += f'\t{producer}\n'
            dscp += f'Consumers:\n'
            for consumer in self._consumers[ftensor]:
                dscp += f'\t{consumer}\n'
        return dscp

    def create_bwop(self, fwop: IRFwOperation) -> IRBpOperation:
        """
        Create dummy backward operator for given forward operator.
        This assumes input/output tensors of fwop have been set by correct gradient tensors.

        This can only be called before any transformation / grouping

        @param fwop IRFwOperation: forward operation

        @return bwop IRBpOperation: the created backward operation
        """
        assert isinstance(fwop, IRFwOperation), "Expected IRFwOperation"
        fins = [t for t in fwop.iobjs() if isinstance(t, IRSubTensor)]
        fous = [t for t in fwop.oobjs() if isinstance(t, IRSubTensor)]
        igrads = [t.grad for t in fins if t.grad is not None]
        # note not all output tensors will be consumed by nodes, e.g., chunk.
        # for these cases, the backward op doesn't have exactly the same number of
        # backward inputs with the number of its forward outputs
        ograds = [t.grad for t in fous if t.grad is not None]
        bwop = IRBpOperation(ograds, igrads)
        IRCell.make_pair(fwop, bwop)
        return bwop

    # ====================== Basic Graph manipulations ======================

    def _add_ftensor(self, ftensor: IRObject):
        """
        Add a full tensor in segment if the segment doesn't have the tensor.
        """
        assert isinstance(ftensor, IRObject)
        if ftensor not in self._fobjects:
            self._fobjects.add(ftensor)
            self._producers[ftensor] = []
            self._consumers[ftensor] = []
            self._ptensors[ftensor] = []
            self._ctensors[ftensor] = []
        if ftensor.is_attr():
            self._attributes.add(ftensor)

    def _remove_ftensor(self, ftensor: IRObject):
        """
        Remove a full tensor in segment
        """
        assert isinstance(ftensor, IRObject)
        if ftensor in self._fobjects:
            self._fobjects.remove(ftensor)
            del self._producers[ftensor]
            del self._consumers[ftensor]
            del self._ptensors[ftensor]
            del self._ctensors[ftensor]
        if ftensor.is_attr() and ftensor in self._attributes:
            self._attributes.remove(ftensor)

    def _reorder_producer_consumer(self):
        """
        Re-order producers and consumers for each full tensor to match
        with the ordering of nodes.

        Note sub-segment will also be reordered.
        """
        # clear up
        self._fobjects, self._attributes = set(), set()
        self._producers, self._ptensors = dict(), dict()
        self._consumers, self._ctensors = dict(), dict()

        # set input and output
        for obj in self.iobjs():
            self._add_ftensor(obj.parent)
        for obj in self.oobjs():
            self._add_ftensor(obj.parent)

        # set producer and consumer
        # NOTE: we use `dict.fromkeys` to remove duplicate tensors
        # as well as keep the order of tensors
        for node in self._nodes:
            if isinstance(node, IRAdapter): continue
            for itensor in dict.fromkeys(node.iobjs()):
                ftensor = itensor.parent
                self._add_ftensor(ftensor)
                self._consumers[ftensor].append(node)
                self._ctensors[ftensor].append(itensor)
            for otensor in dict.fromkeys(node.oobjs()):
                ftensor = otensor.parent
                self._add_ftensor(ftensor)
                self._producers[ftensor].append(node)
                self._ptensors[ftensor].append(otensor)
            if isinstance(node, IRSegment):
                node._reorder_producer_consumer()

    def insert(self, node: IRCell, index: Union[int, CellPosition]):
        """
        Insert a node at index.

        Args:
            node (IRCell): the inserted node
            index (int or CellPosition): the index

        """
        pos = CellPosition((index,)) if isinstance(index, int) else index
        assert isinstance(pos, CellPosition), "Expect index to be int or CellPosition"

        if len(pos) == 1:
            index = pos[0]
            # insert node
            self._nodes.insert(index, node)
            if isinstance(node, IRAdapter): return
            # update producer and consumer
            # NOTE: we use `dict.fromkeys` to remove duplicate tensors
            # as well as keep the order of tensors
            # - consumer
            for itensor in dict.fromkeys(node.iobjs()):
                ftensor = itensor.parent
                self._add_ftensor(ftensor)
                self._consumers[ftensor].append(node)
                self._ctensors[ftensor].append(itensor)
            # - producer
            for otensor in dict.fromkeys(node.oobjs()):
                ftensor = otensor.parent
                self._add_ftensor(ftensor)
                self._producers[ftensor].append(node)
                self._ptensors[ftensor].append(otensor)
        else:
            segment = self._nodes[pos[0]]
            assert isinstance(segment, IRSegment), "Expected IRSegment"
            pos = CellPosition(pos.indices[1:])
            segment.insert(node, pos)

    def remove(self, node: IRCell, _pos: Union[int, CellPosition] = None) -> CellPosition:
        """
        Remove a node at index

        Args:
            node (IRCell): the removed node
            _pos (Optional[Union[int, CellPosition]): help to save cost if provide node position.

        Returns:
            CellPosition: the removed index
        """
        pos = self.index(node) if _pos is None else _pos
        assert self.node(pos) == node, \
            f"posititon doesn't not match with node:\n\t{node}\nGot:\n\t{self.node(pos)}"

        if len(pos.indices) == 1:
            index = pos[0]
            # remove
            self._nodes.pop(index)
            # update producer and consumer
            if isinstance(node, IRAdapter): return pos
            # consumer
            for itensor in dict.fromkeys(node.iobjs()):
                ftensor = itensor.parent
                idx = self._consumers[ftensor].index(node)
                self._consumers[ftensor].pop(idx)
                self._ctensors[ftensor].pop(idx)
                if len(self._consumers[ftensor]) == 0 and len(self._producers[ftensor]) == 0:
                    self._remove_ftensor(ftensor)
            # producer
            for otensor in dict.fromkeys(node.oobjs()):
                ftensor = otensor.parent
                idx = self._producers[ftensor].index(node)
                self._producers[ftensor].pop(idx)
                self._ptensors[ftensor].pop(idx)
                if len(self._consumers[ftensor]) == 0 and len(self._producers[ftensor]) == 0:
                    self._remove_ftensor(ftensor)
        else:
            segment = self._nodes[pos[0]]
            assert isinstance(segment, IRSegment)
            segment.remove(node, _pos=CellPosition(pos.indices[1:]))

        return pos

    def replace(self, node: IRCell, new_nodes: List[IRCell]) -> int:
        """
        Replace one node by multiple nodes

        # TODO: check input and output

        @param node IRCell: the replaced node
        @param new_nodes List[IRCell]: the nodes to be inserted.

        @return index int: the replaced node index
        """
        idx = self.remove(node)
        for new_node in new_nodes[::-1]:
            self.insert(new_node, idx)
        return idx

    def reorder(self, node: IRCell, index: int):
        """
        Reorder an existing node to the index.

        @param node IRCell: the node in this segment, not considering inner segments.
        @param index int: the index is under the view of nodes ordering before this call.

        @return None
        """
        prev_index = self._nodes.index(node)
        self.remove(node, prev_index)
        index = index if prev_index >= index else index - 1
        self.insert(index, node)

    @contextmanager
    def update(self, node):
        """
        Update a node. Note the related change in backward operator
        will not be automatically updated.

        TODO: update operator dependency

        e.g.,
            with graph.modify(node) as node:
                node.set_input(0, tensor)

        @param node IRCell: the node that must in the graph
        @return node IRCell: the modify node
        """
        index = self.remove(node)
        yield node
        self.insert(node, index)

    def exist(self, node: IRCell, flatten: bool = True) -> bool:
        """
        Check if the node is in this graph

        @param node IRCell: the queried node

        @return exsit bool: True if exist otherwise False
        """
        if node in self._nodes:
            return True
        if flatten:
            for segment in self._nodes:
                if not isinstance(segment, IRSegment): continue
                if segment.exist(node, flatten):
                    return True
        return False

    def select(self, name: Optional[str] = None, ntype: Optional[IRCell] = None, flatten: bool = True) -> List[IRCell]:
        """Select all the nodes that satisfy all the specified conditions.

        Note:
            Current IRGraph can have at most a 2-level hierarchy (IRGraph[IRSegment]).
            We don't allow IRSegment inside IRSegment. So when users try to index
            IRSegment, turn `flatten=False` will get the same result as `flatten=True`,
            and can save more time because `flatten=False` will not traverse the
            nodes in IRSegment.

        Args:
            name (Optional[str]): the node name
            ntype (Optional[Type]): the node type
            flatten (bool): whether to recursively search the nodes inside segments (Default True).

        Returns:
            List[IRCell]: the nodes that satisfied the name or ntype.
        """
        nodes = []
        for node in self._nodes:
            if (name is None or name == node.name) and \
               (ntype is None or isinstance(node, ntype)):
                nodes.append(node)
            # recursively search in sub-segment
            if flatten and isinstance(node, IRSegment):
                nodes += node.select(name, ntype, flatten)
        return nodes

    def finsert(self, fwop: IRFwOperation, index: Union[int, CellPosition]) -> IRFwOperation:
        """
        Insert a forward node and create its backward.
        The created backward operator will be happen right before
        the backward of fwop's previous forward node

        This requires the segment has its backward segment
        This assumes inputs/outputs tensors of fwop have been set with correct gradient

        @param fwop IRFwOperation: forward node
        @param index Union[int, CellPosition]: inserted position

        @return node IRFwOperation: the node itself
        """
        assert isinstance(fwop, IRFwOperation), "Only allow insert an IRFwOperation"
        pos = CellPosition((index,)) if isinstance(index, int) else index
        assert isinstance(pos, CellPosition), "Expect index to be int or CellPosition"

        index = pos.indices[-1]
        fsegment = self if len(pos) == 1 else self.node(CellPosition(pos.indices[1:]))
        fsegment.insert(fwop, index)
        # create backward
        bwop = fsegment.create_bwop(fwop)
        bwop.device = fwop.device
        # insert backward
        assert fsegment.mirror is not None, "Missing backward segment"
        bsegment: IRSegment = fsegment.mirror
        bidx = CellPosition((bsegment.nnodes,))
        for idx in range(index - 1, -1, -1):
            prev_fnode = fsegment.node(idx)
            if prev_fnode.mirror is not None:
                bidx = bsegment.index(prev_fnode.mirror)
                break
        bsegment.insert(bwop, bidx)
        return fwop

    # ===================== Advance Graph manipulations ==================

    def multiref(self, ftensor: IRFullTensor, comment: Optional[str] = None, *deprecated_args) -> IRFwOperation:
        """
        Multiref accepts a full tensor that used in multiple places (consumed by a node,
        or belongs to a graph's outputs). Its output tensors are full tensors with new
        ids and dispatched to the corresponding consumers.
        The input tensor can be parameter, buffer or activation tensors.

        Note that during the adapter generation (IRAdapterGener), the multiref inserted
        here will be partitioned automatically by `autoref`. Further more, multiref may
        be added to the graph at that time to reduce the communication time, check
        `gen_activation` and `local_consumer_multiref` for more details.

        Args:
            tensor (IRSubTensor): full tensor to be multiref.

        Returns:
            multiref (IRFwOperation): the inserted multiref operator.

        This function should be called before any graph transformation, like replicate,
        partition. The created multiref operator will be partitioned automatically when
        generating adapters.

        multiref can be regarded as an approach to create different aliases for the input
        full tensor, so we can overcome the limitation of communication generation logic
        and correctly generate communications.

        At runtime, multiref just creates multiple tensors with the same storage, as the following
        code snippet shows:
        ```python
        def multiref(tensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]:
            return tensor if times == 1 else tuple([tensor] * times)
        ```

        There are two kinds of communications in the system.
            - Adapter: Which is used to exchange tensors data during forward and backward across
              devices in the same scale unit. We use RVDLayout algorithm to generate adapters which
              is composed of collective primitives at runtime. The limitation here is the communication
              should be simple. If the communication is too complex, the generation will fail.
            - IRWeightReducer: Which is used to sync weight (parameter) gradients across devices after
              backward. IRWeightReducer will be mapped to nnscaler.runtime.adapter.Reducer in runtime.
              The limitation here is the weight should be ALL partitioned or ALL replicated (check
              gen_weight in IRAdapterGener). Put it in a simple word, reducer is added when the parameter
              can be simply aggregated (summed) across certain devices.

        multiref is here to rescue. Some typical usage of this function are listed below with explanations.

            - Adapter generation case 1: If the full tensor has multiple consumers, and consumers consume
              different portion of the full tensor (different tp partition). In this case, RVDLayout may
              fail to generate backward communication. We should use multiref to create an alias for each
              consumer. RVDLayout will generate communication between each alias and its consumer correctly.
              The inserted multiref will aggregate the gradients automatically in the backward pass according
              to the multiref's implementation and torch.autograd's mechanism.

              Example: If op1/op2 are consumers of fulltensor ft, and will be partitioned different:
                op1(ft)
                op2(ft)
              multref should be inserted
                ft1, ft2 = multiref(ft, 2)
                op1(ft1)
                op2(ft2)
              Note that when op1 is replicated over multiple devices, op2 partitions its another input (not ft2),
              although ft's indmap is same on op1 and op2, but we cannot add the gradients directly. As a result,
              the multiref is needed too.

            - Adapter generation case 2: If the full tensor has multiple consumers, but these consumers have
              different behavior in backward, ie, some of consumers generate gradient (normal torch ops), and
              some of consumers don't generate gradient (mostly IRPyFunc). In this case, we need to use multiref,
              so each alias can have different behaviors in backward.

              Example: If op1 (generate grad)/getitem (doesn't generate grad) are consumers of fulltensor ft, but with different backward behavior:
                torch_op(ft)
                getitem(ft)
              multref should be inserted
                ft1, ft2 = multiref(ft, 2)
                torch_op(ft1)
                getitem(ft2)

            - Adapter generation case 3: When the full tensor has consumers and also is graph's output (a specail
              consumer). If consumers and graph outputs satisify case 1 or case 2, we also need to insert multiref.
              It is a little difference with previous cases, because we don't update the tensor of graph outputs,
              but use the old name. This is correct since the IRPyFunc and the segment's outputs are forced to be
              replicated by the system.

              Example: If op is the only consumer of fulltensor ft:
                op(ft)
                return ft
              multref should be inserted
                ft1 = multiref(ft, 1)
                op(ft1)
                return ft # note old name is used.

            - IRWeightReducer generation case 1: when gradients over devices can not be accumulated directly to
              synchronize. This typically happens when a parameter is shared, especially in pipeline parallelism.
              Here we can use multiref to synchronize gradients, but the semantic is different. (TODO: add a new
              function to handle this case to make it more clear). With multiref, the weight becomes an activation,
              so no IRWeightReducer will be generated. Instead, Adapter will be used in runtime.

              Example: weight w is shared by two consumers, the distributed plan is two-stage pipeline,
                stage 0 uses gpu 0
                stage 1 uses gpu (1, 2)
              weights are all replicated in all devices.
              If we don't insert multiref, the weight will be held in both stages, but the communication will fail to generate.
              If we ignore the communication generation, the code will look like:
              gencode0:
              ```
              def __init__(....):
                self.w = torch.nn.Parameter(...)
                ...
              def forward(...):
                ...
                op1(self.w)
                ...
              ```
              gencode 1:
              ```
              def __init__(....):
                self.w = torch.nn.Parameter(...)
                ...
              def forward(...):
                ...
                op2(self.w)
                ...
              ```
              We cannot sum up the gradients on gpu 0/1/2 directly. In logic, the real gradient should be a sum of gpu0 and gpu1's gradients or
              gpu0 and gpu2's gradients. As the example shows, generating a reducer is hard in this case. Multiref is inserted to convert the
              param to activation to bypass the difficulty, the code will look like:
              gencode0:
              ```
              def __init__(....):
                self.w = torch.nn.Parameter(...)
                ...
              def forward(...):
                 ...
                 w1, w2 = multiref(self.w)
                 op1(w1)
                 ...
              ```
              gencode1:
              ```
              def __init__(....):
                ...
              def forward(w2, ...):
                ...
                op2(w2)
                ...
              ```
              You can see weight is gone in gencode1's constructor. Instead, it will be passed as forward argument. multiref here is just a way to
              convert weight to activation, change the way to generate communication and aggregate gradients by adapters and multiref correctly.
        """
        assert ftensor in self._fobjects, f"tensor: {ftensor} not in this graph."
        assert not ftensor.is_grad(), f"graph.multiref can only be applied on a non-gradient full tensor."
        # check no transformation
        assert len(self.ptensors(ftensor)) <= 1, f"no transformation should be called before multiref"
        assert len(set(self.ctensors(ftensor))) == 1, f"no transformation should be called before multiref"

        # create new full tensors
        consumers = self.consumers(ftensor)
        tensor = self.ctensors(ftensor)[0]
        ftensors: List[IRSubTensor] = [ftensor.like() for _ in consumers]
        otensors: List[IRSubTensor] = [ft.select(tensor.indmap, tensor.valmap) for ft in ftensors]
        # create multiref
        multiref = MultiRef(tensor, len(consumers))
        if comment:
            multiref.comment = comment
        for idx, otensor in enumerate(otensors):
            multiref.set_output(idx, otensor)
        # setup gradient
        req_grad = ftensor.requires_grad
        multiref.input(0).grad = ftensor.grad.select(tensor.indmap, (0, 1)) if req_grad else None
        for idx, output in enumerate(multiref.outputs()):
            if ftensor.grad is None or consumers[idx].mirror is None:
                output.grad = None
            else:
                output.grad = ftensors[idx].grad.select(tensor.indmap, (0,1))
        # insert multiref
        if len(self.producers(ftensor)) == 0:
            fidx = min(self.index(consumer) for consumer in self.consumers(ftensor))
        else:
            fidx = max(self.index(prod) for prod in self.producers(ftensor)) + 1
        # when the consumer is a IRPyFunc, the tensor at the consumer side will not have a grad
        # in this case, we only insert the multiref node in the forward graph
        req_backward = any(output.grad is not None for output in multiref.outputs())
        if req_backward:
            self.finsert(multiref, fidx)
        else:
            self.insert(multiref, fidx)
        # update forward / backward consumer
        for idx, consumer in enumerate(consumers):
            fidx = consumer.inputs().index(tensor)
            grad = consumer.input(fidx).grad
            # update forward
            with self.update(consumer):
                for fidx, t in enumerate(consumer.inputs()):
                    if tensor == t:
                        consumer.set_input(fidx, multiref.output(idx))
                        consumer.input(fidx).grad = multiref.output(idx).grad
            if consumer.mirror is None: continue
            # update backward
            with self.mirror.update(consumer.mirror) as bnode:
                for bidx, t in enumerate(bnode.outputs()):
                    if grad is not None and grad == t:
                        bnode.set_output(bidx, multiref.output(idx).grad)
        return multiref

    def single_consume(self, one_for_all: bool = True):
        """
        Transform graph to make each non-attribute tensor has up to
        one consumer. Multiref nodes will be inserted. The API is useful
        for cases like inference, where different consumers are partitioned
        with different tensor dimensions.

        This should be called before any graph transformation.

        e.g., original graph:

            t = producer(xx)
            ...
            xx = consumer1(t)
            ...
            xx = consumer2(t)
            ...
            xx = consumer3(t)
            ...

        If one_for_all is True, will be:

            t = producer(xx)
            t1, t2, t3 = multiref(t)
            ...
            xx = consumer1(t1)
            ...
            xx = consumer2(t2)
            ...
            xx = consumer3(t3)

        Otherwise:

            t = producer(xx)
            ...
            t1, t2 = multiref(t)
            xx = consumer1(t1)
            ...
            t3, t4 = multiref(t2)
            xx = consumer2(t3)
            ...
            xx = consumer3(t4)


        @param one_for_all bool: If True,
        one single multiref node will be created for each fulltensor. Otherwise,
        if a fulltensor has K consumers, then K-1 multiref nodes will be created.

        @return None
        """
        consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        producers: Dict[IRFullTensor, IRCell] = dict()
        if not one_for_all:
            for node in self.nodes():
                ftensors = set()
                for ftensor in node.inputs():
                    # remove redundant tensors within an operator
                    if isinstance(ftensor, IRFullTensor) and ftensor.tid not in ftensors:
                        ftensors.add(ftensor.tid)
                        if ftensor not in consumers:
                            consumers[ftensor] = []
                        consumers[ftensor].append(node)
                for ftensor in node.outputs():
                    if isinstance(ftensor, IRFullTensor):
                        producers[ftensor] = node
            for ftensor, cnodes in consumers.items():
                if len(cnodes) == 1 or ftensor.is_attr(): continue
                reftensor = ftensor
                ctensor = ftensor
                while len(cnodes) > 0:
                    consumer = cnodes.pop(0)
                    if len(cnodes) > 0:
                        itensors = [ftensor.like() for _ in range(2)]
                        multiref = MultiRef(reftensor, 2)
                        multiref.comment = 'create at IRSegment:single_consume'
                        for idx, itensor in enumerate(itensors):
                            multiref.set_output(idx, itensor)
                        multiref.verify_shape()
                        # insert multiref right before the consumor
                        idx = self.index(consumer)
                        # require backward
                        if any(itensor.requires_grad for itensor in node.inputs()):
                            self.finsert(multiref, idx)
                        else:
                            self.insert(multiref, idx)
                        ctensor, reftensor = itensors
                    else:
                        # the last consumer doesn't need multiref
                        ctensor = reftensor
                    # update consumer
                    while ftensor in consumer.inputs():
                        idx = consumer.inputs().index(ftensor)
                        consumer.set_input(idx, ctensor)
        else:
            for node in self.nodes():
                ftensors = set()
                for ftensor in node.inputs():
                    # remove redundant tensors within an operator
                    if isinstance(ftensor, IRFullTensor) and ftensor._id not in ftensors:
                        ftensors.add(ftensor._id)
                        if ftensor not in consumers:
                            consumers[ftensor] = []
                        consumers[ftensor].append(node)
                for ftensor in node.outputs():
                    if isinstance(ftensor, IRFullTensor):
                        producers[ftensor] = node
            for ftensor, cnodes in consumers.items():
                if len(cnodes) == 1 or ftensor.is_attr(): continue
                itensors = [ftensor.like() for _ in range(len(cnodes))]
                for itensor, consumer in zip(itensors, cnodes):
                    while ftensor in consumer.inputs():
                        idx = consumer.inputs().index(ftensor)
                        consumer.set_input(idx, itensor)
                # create and insert multiref operation
                multiref = MultiRef(ftensor, len(cnodes))
                multiref.comment = 'create at IRSegment:single_consume'
                for idx, itensor in enumerate(itensors):
                    multiref.set_output(idx, itensor)
                multiref.verify_shape()
                idx = self.index(producers[ftensor]) + 1 if ftensor in producers else 0
                # idx = nodes.index(cnodes[0])
                if any(itensor.requires_grad for itensor in node.inputs()):
                    self.finsert(multiref, idx)
                else:
                    self.insert(multiref, idx)

    # ====================== Graph Generations ============================

    @staticmethod
    def get_inputs(nodes: List[IRCell], exclude_attr: bool = True):
        """
        Get all the input tensors that are required by nodes.

        @param nodes List[IRCell]: the nodes

        @return inputs List[IRTensor]: the input tensors
        """
        all_outputs = list()
        for node in nodes:
            all_outputs.extend(node.outputs())
        inputs = list()
        for node in nodes:
            for input in node.inputs():
                if isinstance(input, IRTensor):
                    if exclude_attr and input.is_attr():
                        continue
                    if input not in all_outputs:
                        if input not in inputs:
                            inputs.append(input)
        return inputs

    @staticmethod
    def get_outputs(nodes: List[IRCell], exclude_attr: bool = True):
        """
        Get tensors that are produced but not consumed by nodes

        As long as the tensor is consumed in by the nodes, it will
        not be in the output. A tensor will not appear as output if it
        is double-consumed both outside and inside the nodes.

        @param nodes List[IRCell]: the nodes

        @return outputs List[IRTensor]: the output tensors
        """
        all_inputs = list()
        for node in nodes:
            all_inputs.extend(node.inputs())
        outputs = list()
        for node in nodes:
            for output in node.outputs():
                # not consumed tensor
                if isinstance(output, IRTensor):
                    if exclude_attr and output.is_attr():
                        continue
                    if output not in all_inputs:
                        if output not in outputs:
                            outputs.append(output)
                            continue
        return outputs

    def create_segment(self, nodes: List[IRCell], attr_as_inputs: bool = False) -> IRCell:
        """Create a segment (sub-graph) with part of the nodes.

        This only return the created segment without modifying the graph.

        Calling this requires that the dependencies are already materialized,
        i.e., every input IRSubTensor should have a corresponding producer. Two scenarios
        satisfy this condition:

        1) the node in the graph is not partitioned;

        2) the adapters (communication) are generated.

        Args:
            nodes (List[IRCell]): the subset nodes of this graph
            attr_as_inputs (bool): whether to treat attributes as segment inputs

        Returns:
            segment (IRSegment): the grouped segment.
        """
        segment = self
        segment_outputs = IRSegment.get_objects_from_complex(segment.outputs())

        # setup adapter dependency
        ad_consumers: Dict[Tuple[IRObject,int], Set[int]] = dict()
        ad_producers: Dict[Tuple[IRObject,int], Set[int]] = dict()
        for adapter in self.select(ntype=IRAdapter):
            for itensor in adapter.inputs():
                assert len(itensor.device) == 1
                ad_consumers.setdefault((itensor, itensor.device[0]), set()).add(adapter.cid)
            for otensor in adapter.outputs():
                assert len(otensor.device) == 1
                # for identity adapters, we remove it from producer side
                if (otensor, otensor.device[0]) not in ad_consumers:
                    ad_producers.setdefault((otensor, otensor.device[0]), set()).add(adapter.cid)

        # tensor and its device match
        dmatch = lambda t1, t2: t1 == t2 and t1.device == t2.device

        inputs, outputs = set(), set()
        sub_cids = set(node.cid for node in nodes)
        for node in nodes:
            for itensor in node.iobjs():
                if itensor.is_attr():
                    if attr_as_inputs:
                        inputs.add(itensor)
                    continue
                producers, ptensors = self.producers(itensor.parent), self.ptensors(itensor.parent)
                pids = set(p.cid for p, t in zip(producers, ptensors) if dmatch(t, itensor))
                if len(itensor.device) > 0:
                    assert len(itensor.device) == 1
                    pids.update(cid for cid in ad_producers.get((itensor, itensor.device[0]), []))
                # if no producers inside the nodes can produce data, set as input
                if all(pid not in sub_cids for pid in pids):
                    inputs.add(itensor)
            for otensor in node.oobjs():
                # if the tensor is required by segment outputs, set as output
                if otensor in segment_outputs:
                    outputs.add(otensor)
                    continue
                consumers, ctensors = self.consumers(otensor.parent), self.ctensors(otensor.parent)
                cids = set(c.cid for c, t in zip(consumers, ctensors) if dmatch(t, otensor))
                if len(otensor.device) > 0:
                    assert len(otensor.device) == 1
                    cids.update(cid for cid in ad_consumers.get((otensor, otensor.device[0]), []))
                # if the tensor is required by other nodes outside the nodes, set as output
                if any(cid not in sub_cids for cid in cids):
                    outputs.add(otensor)

        def order(tensors: Set[IRObject]) -> Tuple[IRObject]:
            """Reorder by logical tensor id. Temporally necessary for pipeline scheduling"""
            tensors = list(tensors)
            tids = np.array([t.parent.tid for t in tensors])
            indices = np.argsort(tids)
            return tuple(tensors[idx] for idx in indices)

        segment = IRSegment(nodes, order(inputs), order(outputs))
        return segment

    def dispatch(self, devid: int, _gen_mirror: bool = True) -> Optional[IRCell]:
        """
        Instantiate the segment to a specific device.

        @param devid int: the target device

        @return segment IRSegment: the instantiated segment
        """
        if devid not in self.device:
            return None
        if len(self.device) == 1 and self.device == [devid]:
            return self
        if devid in self._dispatch_cached:
            return self._dispatch_cached[devid]

        inputs, outputs, nodes = self.inputs(), self.outputs(), []
        for node in self._nodes:
            if devid in node.device:
                nodes.append(node.dispatch(devid))

        def order(tensors: Set[IRObject]) -> Tuple[IRObject]:
            """Reorder by logical tensor id. Temporally necessary for pipeline scheduling"""
            tensors = list(tensors)
            tids = np.array([t.parent.tid for t in tensors])
            indices = np.argsort(tids)
            return tuple(tensors[idx] for idx in indices)

        if self.isfw():
            inputs, outputs = order(inputs), order(outputs)

        segment = IRSegment(nodes, inputs, outputs, self.name)
        segment._id = self.cid
        if _gen_mirror and self.mirror is not None:
            msegment = self.mirror.dispatch(devid, _gen_mirror=False)
            IRCell.make_pair(segment, msegment)
        self._dispatch_cached[devid] = segment
        return segment


    # ========================== Graph Visualize ================================

    def __repr__(self):
        fw = 'f' if self.isfw() else 'b'
        inputs = tuple(t for t in self.inputs() if isinstance(t, IRObject) and not t.is_attr())
        if self.isfw():
            dscp = f"{fw}Graph{self.cid}-{self.device}(inputs={inputs}, outputs={self.outputs()})"
        else:
            dscp = f"{fw}Graph{self.cid}-{self.device}(fGraph{self.mirror.cid}, inputs={inputs}, outputs={self.outputs()})"
        return dscp

    def extra_repr(self) -> str:
        dscp = f"\n{self.name}:\n{'=' * len(self.name)}\n"
        # inputs
        dscp += f"Inputs: {self.inputs()}\n"
        for node in self._nodes:
            dscp += f"\n{node}"
            if isinstance(node, IRSegment):
                for subnode in node.nodes():
                    dscp += f"\n\t{subnode}"
        # outputs
        dscp += f"\nOutputs: {self.outputs()}\n{'=' * len(self.name)}\n"
        return dscp

