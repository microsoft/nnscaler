#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple, Callable, Set
import numpy as np
import itertools
import logging
import copy

from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.gener.concurrent import ConcurrentGener
import nnscaler.graph.gener.utils as utils
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment, CellPosition
from nnscaler.graph.function.pyfunc import IRPyFunc

from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.operator import IRFwOperation

from nnscaler.ir.adapter import IRAdapter, IRWeightReducer
from nnscaler.graph.function.function import Accum, Cat, MultiRef


DeviceID = int

_logger = logging.getLogger(__name__)


def create_dummy(segment: IRSegment, inputs: bool = True, outputs: bool = True) -> List[IRFwOperation]:
    """
    Create dummy operators segment inputs and outputs.

    @param segment IRSegment: the target segment
    @param inputs bool: True for creating dummy operators to produce segement's inputs
    @param outputs bool: True for creating dummpy operators to consume segment's outputs

    @return nodes List[IRCell]: the generated operation
    """
    # devices = segment.device
    input_producers: Dict[IRFullTensor, List[IRCell]] = {}
    output_consumers: Dict[IRFullTensor, List[IRCell]] = {}
    devices = segment.device
    # create inputs
    if inputs:
        input_objects = IRGraph.get_objects_from_complex(segment.inputs())
        for tensor in input_objects:
            if not isinstance(tensor, IRSubTensor): continue
            assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
            fwop = utils.DummyInputOuput(tensor, devices, is_output=True, name=f'segment{segment.cid}_input')
            if tensor.grad is not None:
                fwop.output(0).grad = tensor.parent.grad.select(tensor.indmap, (0, 1))
                fwop.output(0).grad.cell = fwop
            input_producers.setdefault(tensor.parent, []).append(fwop)
    # create outputs
    if outputs:
        output_objects = IRGraph.get_objects_from_complex(segment.outputs())
        for tensor in output_objects:
            if not isinstance(tensor, IRSubTensor): continue
            assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
            fwop = utils.DummyInputOuput(tensor, devices, is_input=True, name=f'segment{segment.cid}_output')
            if tensor.grad is not None:
                fwop.input(0).grad = tensor.parent.grad.select(tensor.indmap, (0, 1))
                fwop.input(0).grad.cell = fwop
            output_consumers.setdefault(tensor.parent, []).append(fwop)
    return input_producers, output_consumers


def expand_devices(tensors: List[Optional[IRSubTensor]],
                   producer: bool = False, consumer: bool = False) -> List[IRSubTensor]:
    """
    Scatter a tensor if it is on multiple devices. It produces a tensor list where
    each tensor is attached to one device, with tensor itself is replicated.

    @param tensors List[IRSubTensor]: each tensor can be on multiple devices.
    @param producer bool: if the tensor is producer role
    @param consumer bool: if the tensor is consumer role

    @return dtensors List[IRSubTensor]: each tensor is on one device
    """
    dtensors: Dict[int, List[IRSubTensor]] = {}
    for tensor in tensors:
        if tensor is None: continue
        assert len(tensor.device) > 0, f"find the tensor {tensor} is not assigned by devices"
        for devid in tensor.device:
            if tensor in dtensors.setdefault(devid, []):
                continue
            if producer:
                fwop = utils.DummyInputOuput(tensor, devid, is_output=True, name=tensor.cell.name)
                dtensors[devid].append(fwop.output(0))
            elif consumer:
                fwop = utils.DummyInputOuput(tensor, devid, is_input=True, name=tensor.cell.name)
                dtensors[devid].append(fwop.input(0))
            else:
                raise ValueError("At least one of producer or consumer")
    all_tensors = []
    for device_tensors in dtensors.values():
        all_tensors += device_tensors
    return all_tensors


class IRAdapterGener:

    @staticmethod
    def gen(graph: IRGraph, cost_fn: Optional[Callable] = None) -> IRGraph:
        """
        Generate tensor adapter for both activations and weights
        Note weight reducers are always append to the last.

        Args:
            graph (IRGraph): the graph without adapter
            cost_fn Optional[Callable]: takes an IRAdapterPrim and outputs a cost in float.
            default to be None, which means communication volume is used as cost.

        Returns:
            graph (IRGraph): the graph with adapter inserted
        """
        # reorder producer and consumer ordering
        graph._reorder_producer_consumer()
        _logger.info("finish reordering producer and consumer")
        # remove anchor node
        graph = IRAdapterGener.remove_anchor(graph)
        _logger.info("finish removing anchor nodes")
        # automatic replicate pyfunc
        graph = IRAdapterGener.auto_pyfunc(graph)
        _logger.info("finish replacing auto pyfunc")
        # automatic transform multiref
        graph = IRAdapterGener.autoref(graph)
        _logger.info("finish transforming multiref nodes")
        # generate adapters for activation
        graph = IRAdapterGener.gen_activation(graph, cost_fn=cost_fn)
        # generate weight reducer
        graph = IRAdapterGener.gen_weight(graph)
        # fuse consecutive non-differentiable adapters into one
        # graph = IRAdapterGener.fusion(graph)
        # print(graph.extra_repr())
        return graph


    @staticmethod
    def remove_anchor(graph: IRSegment):
        for anchor in graph.nodes():
            if isinstance(anchor, IRGraphAnchor):
                graph.remove(anchor)
                if anchor.mirror is not None:
                    graph.mirror.remove(anchor.mirror)
            elif isinstance(anchor, IRSegment):
                IRAdapterGener.remove_anchor(anchor)
        return graph

    @staticmethod
    def auto_pyfunc(graph: IRGraph):
        """Transform and assign IRPyFunc.

        Warning:
            Each IRPyFunc will be replicated to all devices of its segment.

            To restrict the replicated devices in pipeline-like scenarios, use `graph.staging`
            to group the operators into segments.

        Args:
            graph (IRGraph): the graph to be transformed

        Returns:
            graph (IRGraph): the transformed graph
        """
        for func in graph.select(ntype=IRPyFunc, flatten=True):
            # get devices it will lowered to
            segment: IRSegment = graph.segment(func)
            devices = set()

            # FIXME: this is temporally disabled as we don't track data dependencies inside
            # operator kwargs. This will be fixed in the future.
            # segment_outputs = IRSegment.get_objects_from_complex(segment.outputs())
            # for t in func.inputs():
            #     if not isinstance(t, IRObject): continue
            #     cells = segment.consumers(t.parent) if t.is_attr() else segment.producers(t.parent)
            #     for cell in cells:
            #         devices.update(cell.device)
            # for t in func.outputs():
            #     if not isinstance(t, IRObject): continue
            #     if t in segment_outputs:
            #         devices.update(segment.device)

            # if a pyfunc doesn't have input, it will be replicated
            # to all devices in its segment.
            if len(devices) == 0:
                devices = set(segment.device)
            # replicate
            pyfuncs = [func.replicate() for _ in devices]
            for devid, pyfunc in zip(sorted(devices), pyfuncs):
                pyfunc.device = devid
            # insert
            position = segment.remove(func)
            for pyfunc in pyfuncs[::-1]:
                segment.insert(pyfunc, position)
        return graph

    @staticmethod
    def gen_weight(graph: IRGraph) -> IRGraph:
        """Generate cross-device weight reducers for gradient accumulation.

        If a weight tensor is replicated across multiple devices by different / partitioned operators,
        the weight tensor is required to accumulate gradients according to chain rules.

        However, if the weight tensor is replicated across devices by replicated operators,
        the weight tensor doesn't need to accumulate gradients.

        Warning:
            1) Each weight tensor's consumers can only be ALL partitioned or ALL replicated.
            2) Weight partitions cannot be partially overlapped.
            3) Limited support for shared weight of multiple operators:
                - If operators are on different device group (e.g. pipeline),
                  operators can only be partitioned.
                - If operators are on same device group,
                  operators can either be all partitioned or all replicated.
        """
        sub_weights : Dict[IRFullTensor, List[IRSubTensor]] = dict()
        sub_weight_consumers: Dict[IRSubTensor, List[IRFwOperation]] = dict()

        def collect_sub_weight(graph: IRSegment):
            nonlocal sub_weights, sub_weight_consumers
            for ftensor in graph.attributes():
                if not ftensor.is_param(): continue
                for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
                    if ctensor.grad is None: continue
                    sub_weight_consumers.setdefault(ctensor, []).append(consumer)
                    sub_weights.setdefault(ftensor, []).append(ctensor)
            for segment in graph.select(ntype=IRSegment, flatten=False):
                if segment.isfw():
                    collect_sub_weight(segment)

        collect_sub_weight(graph)

        # check consistency in node replicate or node partition
        replicated = []
        for sub_weight, consumers in sub_weight_consumers.items():
            # suppose a weight is originally shared by 2 operators op1 and op2,
            # each operator is replicated on a same device group (e.g., rank 0 and rank 1).
            # then the device 0 has (op1, op2) and device 1 also has (op1, op2).
            # we don't need to accumulate gradients for the weight in this case.
            # this case can be checked by whether each device has same consumer set.
            dev_cids = dict()
            for consumer in consumers:
                dev_cids.setdefault(consumer.device[0], []).append(consumer.cid)
            dev_cids = [tuple(sorted(cids)) for cids in dev_cids.values()]
            cross_device_replicated = all(cids == dev_cids[0] for cids in dev_cids)

            # otherwise, we only support fully partitioned consumers,
            # the weight's gradient should be accumulated.
            fully_partitioned = len(set(c.cid for c in consumers)) == len(consumers)

            if not (cross_device_replicated or fully_partitioned):
                nl = '\n'
                raise RuntimeError(
                    f"The weight consumers can either be ALL replicated or ALL partitioned. "
                    f"Detected some consumers are replicated and some are partitioned.\n"
                    f"FullTensor weight: {sub_weight.parent}\n"
                    f"Consumers:\n{nl.join([repr(n) for n in consumers])}\n"
                )
            if cross_device_replicated == 1:  # replicated weights
                replicated.append(sub_weight)
        # check consistency in weight partition
        # note we don't support sub-weight tensors with partially shared part.
        # This is because the shared part may require reducer to accumulate gradients only for the
        # shared part, requiring a more fine-grained tensor granularity.
        # However, we don't support such fine-grained accumulation for now, and we only support
        # to either accumulate same sub-weight tensors or not accumulate non-overlapped sub-weight tensors.
        for ftensor, sub_ws in sub_weights.items():
            # all the sub weights can only be
            # 1) replicated (sw1 == sw2) or,
            # 2) partitioned without overlapping (not sw1.overlap(sw2))
            for sw1, sw2 in itertools.combinations(sub_ws, 2):
                if not (sw1 == sw2 or not sw1.overlap(sw2)):
                    nl = '\n'
                    raise RuntimeError(
                        f"Detected a weight is partitioned with partially shared part among its sub-tensors.\n"
                        f"To achieve this, users need to call `graph.multiref(weight)` inside the policy.\n"
                        f"FullTensor weight: {ftensor}\n"
                        f"Consumers:\n{nl.join([repr(w.cell) for w in sub_ws])}\n"
                    )

        # only record sub-weight that is consumed by multiple devices
        sub_weight_devices: Dict[IRSubTensor, Tuple[int,]] = dict()
        # - pop out replicated sub weights as they will have full gradients,
        # no need for reducer.
        for sub_weight in replicated:
            del sub_weight_consumers[sub_weight]
        # - gather sub weights that are consumed by same device groups
        for sub_weight, consumers in sub_weight_consumers.items():
            devices = set(consumer.device[0] for consumer in consumers)
            if len(devices) > 1:
                devices = tuple(sorted(devices))
                sub_weight_devices[sub_weight] = devices

        # create reducer
        reducers: Dict[Tuple[int,...], List[IRSubTensor]] = dict()
        for subw, devices in sub_weight_devices.items():
            reducers.setdefault(devices, []).append(subw)

        for devices, subws in reducers.items():
            for reducer in IRWeightReducer.from_weights(subws, devices):
                # insert reducer to as the last node.
                graph.insert(reducer, graph.nnodes)

        return graph

    @staticmethod
    def gen_activation(graph: IRSegment, allow_recompute: bool = True, cost_fn: Optional[Callable] = None) -> IRSegment:
        """!
        Generate adapter for activation tensors.
        The forward/backward adapter is inserted before the first consumers of its full tensor.

        Args:
            graph (IRGraph): the graph the requires for adapter.
            allow_recompute (bool): Allow adapter recomputes. If this enables, all adapters will be
                set to the same recompute group with its consumed node.
            cost_fn (Callable | None): takes an IRAdapterPrim and outputs a cost in float.
                default to be None, which will use communication volume.

        Returns:
            graph (IRGraph): the (inplace) modified graph with activation adapters.
        """
        def skip(ptensors: List[IRSubTensor], ctensors: List[IRSubTensor]) -> bool:
            # e.g., loss or parameter/buffer
            if len(ptensors) == 0 or len(ctensors) == 0:
                return True
            # direct connection
            for ctensor in ctensors:
                if not any(t == ctensor and set(ctensor.device).issubset(set(t.device)) for t in ptensors):
                    return False
            return True

        input_producer, output_consumer = create_dummy(graph, inputs=True, outputs=True)
        bgraph: Optional[IRSegment] = graph.mirror

        # Here are two optimization passes that are applied before generating communication adapters:
        # - local producer fusion: If an operator is partitioned and there are multiple
        #   different sub-tensors on the same device, insert appropriate concat or accumulate
        #   operators to merge the results on the current device before generating communication.
        #   This way, part of the communication between multiple devices can be converted into
        #   local data processing.
        #
        # - local consumer multiref: When a full tensor has multiple consumers and, after partitioning,
        #   there exists a device that contains multiple partitioned consumers (note that this pass assumes 
        #   these consumers share a same sub-tensor in the forward graph), a multiref node will be
        #   inserted before these consumers on that device. This way, during the backward pass through
        #   the multiref node, the gradients from the consumers are automatically accumulated together,
        #   avoiding the need for accumulation operations in the backward adapter. Note that to make
        #   this optimization work properly, `flatten_grad` should be called to adjust the valuemap of
        #   the gradient sub-tensors.
        #
        # Apart from the purpose of improving the efficiency of communication adapters, these two passes
        # also reduce the number of sub-tensors that need to be considered when generating adapters, which
        # can help to reduce the complexity of the adapter generation algorithm. More specifically, if the
        # plan of the input graph is SPMD, the local consumer multiref pass will ensure the number of
        # fptensors and bptensors is the same, which raise the possibility of generating high performance
        # collectives, like allgather, allreduce, etc.
        ftensors = []
        _cnt = 0
        for ftensor in graph.full_tensors():
            # backward adapter will be generated along with the forward adapter
            if ftensor.is_param() or ftensor.is_grad():
                continue
            # flatten gradient
            utils.flatten_grad(graph, ftensor)
            ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
            IRAdapterGener.local_consumer_multiref(graph, ftensor)
            ftensors.append(ftensor)
            _cnt = _cnt + 1
            if _cnt % 100 == 0:
                _logger.info(f'processed local fusion & multiref for {_cnt} tensors')
        _logger.info(f'finish local fusion & multiref for {_cnt} tensors')

        # reorder again since inserted multiref could be mis-ordered
        graph._reorder_producer_consumer()
        _logger.info("finish reordering producer and consumer")

        # generate adapter for intra-segments
        # FIXME: assume producers and consumers can run in parallel
        _cnt = 0
        for ftensor in ftensors:

            # debug
            # print(f'forward:\n{graph.debug_tensor_map_str(ftensor)}')
            # print(f'backward:\n{graph.mirror.debug_tensor_map_str(ftensor.grad)}')

            # producers can be operators and graph inputs
            fproducers, fptensors = graph.producers(ftensor), graph.ptensors(ftensor)
            if ftensor in input_producer:
                fptensors = fptensors + tuple(fop.output(0) for fop in input_producer[ftensor])
            fptensors = expand_devices(fptensors, producer=True)
            assert all(len(ptensor.device) == 1 for ptensor in fptensors), "Not support for multi-device"

            # consumers can be operators and graph outputs
            fconsumers, fctensors = graph.consumers(ftensor), graph.ctensors(ftensor)
            fctensors = expand_devices(fctensors, consumer=True)
            assert all(len(ctensor.device) == 1 for ctensor in fctensors), "Not support for multi-device"

            bproducers, bptensors = [], []
            bconsumers, bctensors = [], []
            if isinstance(ftensor.grad, IRFullTensor):
                bproducers, bptensors = bgraph.producers(ftensor.grad), bgraph.ptensors(ftensor.grad)
                bptensors = expand_devices(bptensors, producer=True)
                bconsumers, bctensors = bgraph.consumers(ftensor.grad), bgraph.ctensors(ftensor.grad)
                if ftensor in input_producer:
                    bctensors = bctensors + tuple(fwop.output(0).grad for fwop in input_producer[ftensor])
                bctensors = expand_devices(bctensors, consumer=True)
                assert all(len(ctensor.device) == 1 for ctensor in bctensors), "Not support for multi-device"

            fadapters = []

            # (activation -> activation) generation: generate communication adapters between producer operators
            # and consumer adapters.
            if (not skip(fptensors, fctensors)) or (not skip(bptensors, bctensors)):
                fadapter = ConcurrentGener.gen(fptensors, fctensors, bptensors, bctensors, cost_fn)
                if fadapter is not None:
                    fadapters.append(fadapter)

            # (activation -> graph/segment output) generation: generate communication adapters between
            # producer operators and graph/segment output tensors. Note graph/segment output tensors
            # always require for full-shape/value for output, while producers may partition them. Therefore,
            # we need to additionally generate adapters for this case.
            if ftensor in output_consumer:
                out_fctensors = tuple(fwop.input(0) for fwop in output_consumer[ftensor])
                out_fctensors = expand_devices(out_fctensors, consumer=True)
                out_bptensors = [t.grad for t in out_fctensors if isinstance(t, IRSubTensor)]
                # skip if the output is same with activation tensor
                if set(out_fctensors) == set(fctensors) and \
                   set(out_bptensors) == set(bptensors) and \
                   set(t.device[0] for t in out_fctensors) == set(t.device[0] for t in fctensors):
                    pass
                else:
                    fctensors = out_fctensors
                    bptensors = []
                    if isinstance(ftensor.grad, IRFullTensor):
                        bptensors = tuple(fwop.input(0).grad for fwop in output_consumer[ftensor])
                        bptensors = expand_devices(bptensors, producer=True)
                    if (not skip(fptensors, fctensors)) or (not skip(bptensors, bctensors)):
                        fadapter = ConcurrentGener.gen(fptensors, fctensors, bptensors, bctensors, cost_fn)
                        if fadapter is not None:
                            fadapters.append(fadapter)

            # insert adapters
            for fadapter in fadapters:
                if not isinstance(graph, IRGraph):
                    if not (fadapter.differentiable or fadapter.mirror is None):
                        raise NotImplementedError(
                            "Require adapter to be differentiable for nested IRAdapter.\n"
                            "Condition to be differentiable: prodcuers have same device set with consumers\n"
                            f"Failed FullTensor: {ftensor}"
                            f"{graph.debug_tensor_map_str(ftensor)}"
                            f"Failed FullTensor.grad: {ftensor.grad}"
                            f"{bgraph.debug_tensor_map_str(ftensor.grad) if ftensor.grad is not None else None}"
                        )

                badapter: Optional[IRAdapter] = fadapter.mirror

                # skip badapter if it doesn't contain any primitives
                if not fadapter.differentiable and (badapter is not None and len(badapter.prims) == 0):
                    badapter = None

                if (badapter is not None and len(fadapter.prims) == 0 and len(badapter.prims) == 0) or \
                   (badapter is None and len(fadapter.prims) == 0):
                    continue

                # insert forward adapter
                # graph.insert(fadapter, max(producers) + 1)
                if len(fconsumers) > 0:
                    fidx = min(graph.multi_index(fconsumers))
                else:
                    # no consumer: find the last forward node
                    for fidx, node in enumerate(graph.nodes()[::-1]):
                        if node.isfw():
                            fidx = CellPosition(tuple([graph.nnodes - fidx]))
                            break
                graph.insert(fadapter, fidx)
                # setup recompute
                if allow_recompute:
                    if fidx > CellPosition(tuple([0])):
                        prev_node = graph.node(fidx-1)
                        if isinstance(prev_node, (IRFwOperation, IRAdapter)):
                            fadapter.recompute = prev_node.recompute

                # insert backward adapter
                if badapter is not None:
                    assert isinstance(badapter, IRAdapter)
                    assert isinstance(bgraph, IRSegment)
                    if len(bproducers) > 0:
                        bidx = max(bgraph.multi_index(bproducers)) + 1
                    else:
                        # no producer: find the first backward node
                        for bidx, node in enumerate(bgraph.nodes()):
                            if not node.isfw(): break
                    bgraph.insert(badapter, bidx)
            _cnt = _cnt + 1
            if _cnt % 100 == 0:
                _logger.info(f'generated {_cnt} activation adapters')
        _logger.info(f'finish generating {_cnt} activation adapters')

        # generate adapter for each segment
        segments = [seg for seg in graph.nodes() if isinstance(seg, IRSegment) and seg.isfw()]
        for segment in segments:
            IRAdapterGener.gen_activation(segment, allow_recompute=allow_recompute, cost_fn=cost_fn)

        return graph

    @staticmethod
    def local_producer_fusion(graph: IRSegment, ftensor: IRFullTensor) -> IRFullTensor:
        """
        Fuse the producer tensors using concat and add.
        This will add a new full tensor by chaging from:
            producer --(ftensor)--> consumer
        to:
            producer --(ftensor)--> fused nodes --(new ftensor)--> consumer

        Recompute policy: if all the producers are recomputed in a same
        recompute group, then the additional generated cat/add are also
        apllied with same recompute region. Otherwise no recompute.

        Args:
            graph (IRSegment): the graph that contains the full tensor
            ftensor (IRFullTensor): the full tensor to be manipulated

        Returns:
            new_ftensor IRFullTensor: the new full tensor. If cannot fuse,
            return the original ftensor.
        """
        if not ftensor.requires_grad: return ftensor

        devtensors: Dict[DeviceID, List[IRSubTensor]] = dict()
        devops: Dict[DeviceID, List[IRCell]] = dict()

        # collect producers for each device
        for ptensor, producer in zip(graph.ptensors(ftensor), graph.producers(ftensor)):
            for devid in ptensor.device:
                if devid not in devtensors:
                    devtensors[devid], devops[devid] = [], []
                devtensors[devid].append(ptensor)
                devops[devid].append(producer)

        require_fusion = any(len(set(ts)) > 1 for ts in devtensors.values())
        if not require_fusion: return ftensor

        new_ftensor = ftensor.like()

        # update consumer
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            itensor = new_ftensor.select(ctensor.indmap, ctensor.valmap)
            igrad = new_ftensor.grad.select(ctensor.grad.indmap, ctensor.grad.valmap)
            with graph.update(consumer) as consumer:
                consumer.replace_input(ctensor, itensor)
            with graph.mirror.update(consumer.mirror) as bconsumer:
                bconsumer.replace_output(ctensor.grad, igrad)

        for devid in devtensors:
            indmaps = [t.indmap for t in devtensors[devid]]
            valmaps = [t.valmap for t in devtensors[devid]]
            split_dim = len(set(indmaps)) > 1
            split_val = len(set(valmaps)) > 1
            assert not (split_dim and split_val), (
                f"Not support for simutaneously partitioning tensor dimension and tensor value.\n"
                f"{graph.debug_tensor_map_str(ftensor)}"
            )

            node = None
            # get recomput group
            rcid = set(producer.recompute for producer in devops[devid])
            rcid = list(rcid)[0] if len(rcid) == 1 else None

            # split dimension case
            if split_dim:
                catdim: int = None
                for dim in range(len(ftensor.shape)):
                    dim_maps = [ind[dim] for ind in indmaps]
                    if set(len(dim_maps)) != 1:
                        assert catdim is None, (
                            f"Not support for multi-dim partitioning on local producers.\n"
                            f"{graph.debug_tensor_map_str(ftensor)}"
                        )
                        catdim = dim
                assert catdim is not None
                start_idx = np.array([ind[catdim][0] for ind in indmaps])
                indices = np.argsort(start_idx)
                ptensors = [devtensors[devid][idx] for idx in indices]
                try:
                    otensor = ptensors[0]
                    for t in ptensors[1:]:
                        otensor = otensor.concat(t, dim=catdim)
                except Exception as e:
                    raise RuntimeError(
                        f"Device {devid}: Fail to concat local produced tensors on dimension: {catdim}\n"
                        f"Users can try to adjust node ordering to meet with concat order.\n"
                        f"{graph.debug_tensor_map_str(ftensor)}"
                    )
                # set concat input / output
                node = Cat(ptensors, dim=catdim)
                node.set_output(0, new_ftensor.select(otensor.indmap, otensor.valmap))
                # set gradient
                for idx, ptensor in enumerate(ptensors):
                    node.input(idx).grad = ftensor.grad.select(ptensor.indmap, (0,1))
                node.output(0).grad = new_ftensor.grad.select(otensor.indmap, (0,1))

            # split value case
            if split_val:
                # reverse to meet with add order
                ptensors = devtensors[devid]
                try:
                    nchunks = [t.valmap[1] for t in ptensors]
                    if len(set(nchunks)) == 1:
                        otensor = ptensors[0].accum(ptensors[1:])
                    else:
                        # the add order is to adapt with ordering valmap ordering: (3/4) (2/4) (0/2)
                        ptensors = ptensors[::-1]
                        otensor = ptensors[0]
                        for t in ptensors[1:]:
                            otensor = otensor.accum(t)
                except Exception as e:
                    raise RuntimeError(
                        f"Device {devid}: Fail to accum local produced tensors\n"
                        f"Users can try to adjust node ordering to meet with accum order\n"
                        f"{graph.debug_tensor_map_str(ftensor)}"
                    )

                # === Optimization: quick accumulation to early release tensor
                lhs, rhs = ptensors[0], None
                for ptensor in ptensors[1:]:
                    rhs = ptensor
                    output = ftensor.like().select(ptensors[0].indmap, (0,1))
                    node = Accum(lhs, rhs)
                    node.set_output(0, output)
                    node.device = devid
                    node.recompute = rcid
                    graph.insert(node, graph.index(ptensor.cell) + 1)
                    lhs = output
                # remove last node for adaptation
                graph.remove(node)

                # === Orignal way to at alst release tensor
                # node = Accum(*ptensors)
                # # set gradient
                # for idx, ptensor in enumerate(ptensors):
                #     node.input(idx).grad = ftensor.grad.select(ptensor.indmap, (0,1))

                # set output
                node.set_output(0, new_ftensor.select(otensor.indmap, otensor.valmap))
                node.output(0).grad = new_ftensor.grad.select(otensor.indmap, (0,1))

            # no need for fusion, change the producer output to new tensor
            if node is None:
                for ptensor, producer in zip(devtensors[devid], devops[devid]):
                    otensor = new_ftensor.select(ptensor.indmap, ptensor.valmap)
                    ograd = new_ftensor.grad.select(ptensor.grad.indmap, ptensor.grad.valmap)
                    with graph.update(producer):
                        producer.replace_output(ptensor, otensor)
                        for t in producer.find(otensor):
                            t.grad = ograd
                    with graph.mirror.update(producer.mirror) as bproducer:
                        bproducer.replace_input(ptensor.grad, ograd)
            else:
                node.device = devid
                node.recompute = rcid
                # insert
                max_fid = max(graph.index(producer) for producer in devops[devid])
                graph.finsert(node, max_fid + 1)

        return new_ftensor

    @staticmethod
    def local_consumer_multiref(graph: IRSegment, ftensor: IRFullTensor):
        """!
        If a device have a same sub-tensor to be consumed multiple times,
        then create a multiref forward node for it to make
        each sub-tensor to be consumed only once in each device.

        producer -> consumers[0,1]

        producer -> multiref -> consumer[0]
                        |-----> consumer[1]

        Args:
            graph (IRSegment): the graph that contains the full tensor
            ftensor (IRFullTensor): the full tensor to be manipulated

        Returns:
            None: the graph is modified inplace.
        """
        if not ftensor.requires_grad: return

        devtensors : Dict[DeviceID, List[IRSubTensor]] = dict()
        devops : Dict[DeviceID, List[IRCell]] = dict()

        # collect consumer of each device
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            if consumer.mirror is None: continue
            for devid in ctensor.device:
                devtensors.setdefault(devid, []).append(ctensor)
                devops.setdefault(devid, []).append(consumer)
                assert devtensors[devid][0] == ctensor, (
                    f"Detect that a full tensor is partitioned differently on a device.\n"
                    f"To avoid this, need call graph.multiref before graph transformation.\n"
                    f"{graph.debug_tensor_map_str(ftensor)}"
                )

        require_multiref = any(len(ops) > 1 for ops in devops.values())
        if not require_multiref: return

        for devid in devtensors:
            grads: List[IRSubTensor] = [t.grad for t in devtensors[devid]][::-1]
            try:
                nchunks = [grad.valmap[1] for grad in grads]
                if len(set(nchunks)) == 1:
                    accum_grad = grads[0].accum(grads[1:])
                else:
                    # the add order is to adapt with ordering valmap ordering: (3/4) (2/4) (0/2)
                    accum_grad = grads[0]
                    for grad in grads[1:]:
                        accum_grad = accum_grad.accum(grad)
            except Exception as e:
                raise RuntimeError(
                    f"Device {devid}: Fail to accumulate local gradient: {ftensor.grad}\n"
                    f"Error information: {str(e)}\n"
                    f"Users can try:\n"
                    f"  1) Replicate all operators whose inputs have multi-consumed tensors\n"
                    f"  2) Partition all operators whose inputs have multi-consumed tensors\n"
                    f"  3) Call graph.multiref to divide tensors with different partition strategies\n"
                    f"{graph.debug_tensor_map_str(ftensor)}"
                    f"{graph.mirror.debug_tensor_map_str(ftensor.grad)}"
                )

            multiref = MultiRef(devtensors[devid][0], len(grads))
            multiref.comment = 'created at IRAdapterGener:local_consumer_multiref'
            # set input gradient
            multiref.input(0).grad = accum_grad
            # set output and its gradient
            for idx, ctensor in enumerate(devtensors[devid]):
                new_ftensor = ctensor.parent.like()
                otensor = new_ftensor.select(ctensor.indmap, (0,1))
                multiref.set_output(idx, otensor)
                multiref.output(idx).grad = new_ftensor.grad.select(ctensor.indmap, (0,1))
                # set corresponding consumer input and its backward
                consumer = devops[devid][idx]
                with graph.update(consumer):
                    consumer.replace_input(ctensor, otensor)
                    for t in consumer.find(otensor):
                        t.grad = new_ftensor.grad.select(ctensor.indmap, (0,1))
                with graph.mirror.update(consumer.mirror) as bconsumer:
                    bconsumer.replace_output(
                        ctensor.grad, new_ftensor.grad.select(ctensor.indmap, (0,1)))
            # insert multiref
            multiref.device = devid
            min_fidx = min(graph.index(consumer) for consumer in devops[devid])
            # set recompute id
            multiref.recompute = graph.node(min_fidx).recompute
            graph.finsert(multiref, min_fidx)

    @staticmethod
    def autoref(graph: IRSegment) -> IRGraph:
        """
        Automatically transform inserted multiref.
        Multiref is transformed to align with the output tensors on each device.

        Args:
            graph (IRGraph): the graph to be transformed

        Returns:
            graph (IRGraph): the graph with transformed multiref
        """
        for multiref in graph.select(name='multiref', flatten=False):
            # setup recompute
            idx = graph.index(multiref).indices[0]
            recompute = None
            neighbor = graph.node(idx-1) if idx > 0 else graph.node(idx+1)
            if isinstance(neighbor, IRFwOperation):
                recompute = neighbor.recompute

            ftensor: IRFullTensor = multiref.input(0).parent
            multirefs = []
            # by default follow producer transformation strategy
            ptensors = graph.ptensors(ftensor)
            if len(ptensors) > 0:
                # In order to generate correct adapters for multiref, we need to
                # ensure Multirefs below is ordered by devices, which is aligned
                # with consumer operators. As a result, we sort the ptensors here.
                ptensors = sorted(ptensors, key=lambda t: t.device[0])
                for tensor in ptensors:
                    mr = MultiRef(tensor, len(multiref.outputs()))
                    mr.comment = f'create at IRAdapterGener:autoref, comment before transformation: {multiref.comment}'
                    mr.input(0).grad = tensor.grad
                    for idx, out in enumerate(multiref.outputs()):
                        output = out.parent.select(tensor.indmap, tensor.valmap)
                        if out.grad is not None:
                            output.grad = out.grad.parent.select(tensor.indmap, (0,1))
                        mr.set_output(idx, output)
                    mr.device = tensor.device
                    mr.recompute = recompute
                    multirefs.append(mr)
            # otherwise replicate: usually for weight / graph inputs
            else:
                devices = set()
                for otensor in multiref.outputs():
                    ftensor = otensor.parent
                    for consumer in graph.consumers(ftensor):
                        devices.update(consumer.device)
                devices = sorted(devices)
                for devid in devices:
                    mr = multiref.replicate()
                    mr.device = devid
                    mr.recompute = recompute
                    multirefs.append(mr)
            assert len(multirefs) > 0
            # remove original multiref
            fidx = graph.remove(multiref)
            if multiref.mirror is not None:
                graph.mirror.remove(multiref.mirror)
            # insert multirefs
            req_bw = multiref.mirror is not None
            for ofst, multiref in enumerate(multirefs):
                if req_bw:
                    graph.finsert(multiref, fidx + ofst)
                else:
                    graph.insert(multiref, fidx + ofst)
        for segment in graph.select(ntype=IRSegment, flatten=False):
            if not segment.isfw(): continue
            IRAdapterGener.autoref(segment)
        return graph

    @staticmethod
    def fusion(graph: IRSegment) -> IRSegment:
        """
        Fuse consecutive adapters into one
        """
        fadapters, badapters = [], []
        for adapter in graph.nodes():
            if isinstance(adapter, IRAdapter) and adapter.isfw() and not adapter.differentiable:
                fadapters.append(adapter)
                if adapter.mirror is not None:
                    badapters.append(adapter.mirror)
                    # badapters.insert(0, adapter.mirror)
            else:
                if len(fadapters) > 1:
                    # reorder adapter to match output of segment. This is temporally
                    # necessary for pipeline scheduling with multiple output.
                    ftids = np.array([fadapter.input(0).parent.tid for fadapter in fadapters])
                    indices = np.argsort(ftids)
                    fadapters = [fadapters[idx] for idx in indices]
                    if len(badapters) > 0:
                        badapters = [badapters[idx] for idx in indices]
                    # insert fused fadapter
                    fused_fadapter = IRAdapter.merge(fadapters)
                    for adapter in fadapters:
                        idx = graph.remove(adapter)
                    graph.insert(fused_fadapter, idx)
                    # insert fused badapter
                    fused_badapter = IRAdapter.merge(badapters) if len(badapters) > 0 else None
                    for adapter in badapters:
                        idx = graph.remove(adapter)
                    if fused_badapter is not None:
                        graph.insert(fused_badapter, idx)
                    IRCell.make_pair(fused_fadapter, fused_badapter)
                fadapters, badapters = [], []

        for segment in graph.nodes():
            if isinstance(segment, IRSegment) and segment.isfw():
                IRAdapterGener.fusion(segment)

        return graph

    @staticmethod
    def tensor_merge(tensors: List[IRSubTensor], target: Optional[IRSubTensor] = None) -> List[Tuple[str, List, IRSubTensor]]:
        """
        Merge sub-tensors into one tensor or stop right after gets target tensor.

        Merge primtiives:
            "sum: output = sum(inputs)"
            "cat: output = cat(inputs, dim: int)

        @param tensors List[IRSubTensor]: list of tensors
        @param target Optional[IRSubTensor]: the target tensor (default None).

        @return primitives List[Tuple[str, List, IRSubTensor]]:
            List primitives of in forms of (op, inputs, outputs)
        """
        prims = []
        tensors = [t for t in tensors]
        while len(tensors) > 1:
            out = None
            for t1, t2 in itertools.combinations(tensors, 2):
                # try concat
                catdim = t1.catdim(t2)
                if catdim is not None:
                    tensors = [t1, t2] if t1.indmap[catdim][0] < t2.indmap[catdim][0] else [t2, t1]
                    out = tensors[0].concat(tensors[1], dim=catdim)
                    prims.append(('cat', tensors + [catdim], out))
                    break
                # try summation
                if t1.accumable(t2):
                    out = t1.accum(t2)
                    prims.append(('sum', [t1, t2], out))
                    break
            if out is not None:
                tensors.remove(t1)
                tensors.remove(t2)
                tensors.append(out)
                if target is not None and out == target: break
            else:
                remain = '\n\t'.join(t.extra_repr() for t in tensors)
                sprims = '\n\t'.join(repr(p) for p in prims)
                raise RuntimeError(
                    f"Fail to merge tensors into one tensor or cannot match with target.\n"
                    f"Remain Tensor:\n\t{remain}\n"
                    f"Existing primitives:\n\t{sprims}\n"
                )
        return prims
