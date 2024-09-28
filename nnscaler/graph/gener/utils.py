#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Utilities for gradient modification
"""
from typing import Dict, List, Union, Tuple
from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor, ValueMap


class DummyInputOuput(IRFwOperation):

    def __init__(self, tensor: IRSubTensor, device: Union[int, Tuple[int]], 
                 is_input=False, is_output=False,
                 name='dummy'):
        assert (is_input and not is_output) or (is_output and not is_input)
        inputs = [tensor] if is_input else []
        outputs = [tensor] if is_output else []
        super().__init__(name, name, inputs, len(outputs))
        if is_output:
            self.set_output(0, tensor)
        self.device = device
    
    def __repr__(self) -> str:
        return f'DummyInputOutput-{self.device}(inputs={self.inputs()}, outputs={self.outputs()})'


def tensor_vd_repr(t: IRSubTensor) -> str:
    """
    Tensor index-value partition representation
    """
    assert isinstance(t, IRSubTensor), f"expect IRSubTensor"
    identifier = 't' if not t.is_grad() else 'g'
    dchunks, dpos = [], []
    for dim in range(t.ndims):
        dchunks.append(t.parent.shape[dim] // t.shape[dim])
        dpos.append(t.indmap[dim][0] // t.shape[dim])
    indmap = ','.join(f'{idx}/{nchunks}' for idx, nchunks in zip(dpos, dchunks))
    dscp = f'{identifier}{t.tid}-{t.device}(p{t.parent.tid}, shape={t.shape}, D({indmap}), V({t.valmap[0]}/{t.valmap[1]})'
    return dscp


def convert_add_to_valmap(graph: IRGraph, add_node: IRFwOperation):
    """
    Remove add node by replacing with tensor valmap

    @param graph IRGraph: the program
    @param add_node IRFwOperation: the add forward operation
    """
    assert add_node.name == 'add'
    ptensors, producers = [], []
    for itensor in add_node.inputs():
        iptensors = graph.ptensors(itensor.parent)
        assert len(set(t.valmap for t in iptensors)) == len(iptensors)
        ptensors += iptensors
        producers += graph.producers(itensor.parent)
    ftensor = add_node.output(0).parent
    for idx, (ptensor, producer) in enumerate(zip(ptensors, producers)):
        fidx = producer.outputs().index(ptensor)
        bidx = producer.mirror.inputs().index(ptensor.grad)
        ptensor = ftensor.select(ptensor.indmap, (idx, len(producers)))
        ptensor.grad = ftensor.grad.select(ptensor.indmap, (0,1))
        with graph.update(producer):
            producer.set_output(fidx, ptensor)
        with graph.mirror.update(producer.mirror) as bnode:
            bnode.set_input(bidx, ptensor.grad)
    graph.remove(add_node)
    graph.mirror.remove(add_node.mirror)


def flatten_grad(graph: IRSegment, ftensor: IRFullTensor):
    """
    Reset gradient for consumers that are different (no replica)
    Gradient valuemap will be flatten inter-devices, e.g.,(0,3), (1,3), (2,3)
    Gradient valuemap will be exponent intra-devices, e.g., (0,2), (2,4), (3,4)

    @param graph IRGraph: the graph
    @param ftensor IRFullTensor: the fulltensor

    @return None: this is an inplacement update.
    """
    if not isinstance(ftensor.grad, IRFullTensor): return
    
    grads = [t.grad for t in graph.ctensors(ftensor)]
    # require each consumer is a different operator (no replica)
    if len(set(grads)) != len(grads): return
    
    # group consumers by same tensor and same device
    devtensors : Dict[IRSubTensor, Dict[int, List[IRFwOperation]]] = dict()
    for ctensor in graph.ctensors(ftensor):
        devtensors[ctensor] = dict()
    for ctensor in graph.ctensors(ftensor):
        if len(ctensor.device) > 1: return
        devtensors[ctensor][ctensor.device[0]] = []
    for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
        if consumer.mirror is None: continue
        devid = ctensor.device[0]
        devtensors[ctensor][devid].append(consumer)
    
    # setup gradient
    for ctensor in devtensors:
        nchunks = len(devtensors[ctensor])
        for vid, consumers in enumerate(devtensors[ctensor].values()):
            curr_valmap = ValueMap((vid, nchunks))
            for cidx, consumer in enumerate(consumers):
                valmap = curr_valmap.map((0, 2)) if cidx != len(consumers) - 1 else curr_valmap
                grad = ftensor.grad.select(ctensor.indmap, valmap)
                # update consumer and its mirror node
                assert consumer.mirror is not None, consumer
                with graph.update(consumer) as fnode:
                    for t in fnode.find(ctensor):
                        old_grad = t.grad
                        t.grad = grad
                with graph.mirror.update(consumer.mirror) as bnode:
                    bnode.replace_output(old_grad, grad)
                # update current valmap
                curr_valmap = curr_valmap.map((1, 2)) if cidx != len(consumers) - 1 else curr_valmap
