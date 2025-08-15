#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Union, Tuple, List, Optional, Dict, Any
import logging
import copy
import dill
import hashlib

from nnscaler.ir.cten import IRTensor, IRCell, IRObject
from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor, ValueMap

from nnscaler.graph.function.function import Identity
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.function.dimops import IRDimops, OpAnno
from nnscaler.graph.segment import IRSegment

from nnscaler.algorithm.generics import GenericDistAlgo


_logger = logging.getLogger(__name__)
FOp = Union[IRFwOperation, IRDataOperation]


class IRGraph(IRSegment):
    """
    IRGraph.

    IRGraph is used for reprensting a distributed training iteration.
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRTensor], outputs: List[IRTensor],
                 module_name: str):

        super().__init__(nodes, inputs, outputs, module_name)

        self._sched = None  # the schedule strategy

    @property
    def train(self) -> bool:
        """!
        Train flag.

        @return train bool: True if backward is required, otherwise False (inference only).
        """
        return any(not n.isfw() for n in reversed(self._nodes))

    # ================ Deep Learning Interfalce ======================

    def __call__(self, *args):
        """
        Register forward action
        """
        return self.forward(*args)

    def forward(self, *args: Tuple[IRObject]) -> Union[IRTensor, Tuple[IRTensor]]:
        """Forward the IRGraph to add model nodes into program.

        Args:
            args (Tuple[IRObject]): input IRObjects

        Returns:
            Any: output that can be nested structure of IRObjects
        """
        if not all(isinstance(arg, IRObject) for arg in args):
            raise TypeError("Expected input arguments to be IRObject")

        # align graph with input tensors
        iobjs: Tuple[IRObject, ...] = self.inputs()
        if len(args) != len(iobjs):
            _logger.error(
                f'cube graph forward: skipping arguments due to len(args) != len(itensors): '
                f'{len(args)} != {len(iobjs)}'
            )
            if len(args) > len(iobjs):
                args = args[:len(iobjs)]
                _logger.warning(f'cube graph forward: args shrinked into {args}')
            else:
                raise RuntimeError('len(args) < len(itensors)')

        # replace the graph input tensors with provided input tensors
        # this is to connect outside-model operators like dataloader.
        for idx, (iobj, arg) in enumerate(zip(iobjs, args)):
            # reset input
            self.set_input(idx, arg)
            # replace node inputs, kwargs and outputs
            for producer in self.producers(iobj.parent):
                with self.update(producer):
                    producer.replace_output(iobj, arg)
            for consumer in self.consumers(iobj.parent):
                with self.update(consumer):
                    consumer.replace_input(iobj, arg)
            # reset output
            self.replace_output(iobj, arg)

        # set global graph, so @compile can access it.
        # @compile needs a global graph to work
        from nnscaler.program import Program, is_global_graph_enabled
        if is_global_graph_enabled():
            Program().add_nodes(self.nodes())

        # return the output of the graph
        # the return value simulates the output of the model `forward`
        # e.g. If there is only one output, return the output tensor directly instead of a tuple
        if len(self.outputs()) == 1:
            return self.output(0)
        else:
            return self.outputs()

    def backward(self, loss: Optional[IRSubTensor] = None):
        """
        Backward the graph from the entry tensor of loss to complete the graph with backward operators.

        This will infer tensors' gradients by following rules:

        Conditions must satisfy for an forward op having its backward op:
            * one of its output tensors requires gradient
            * one of its output tensors is consumed by other forward ops

        For operators that doesn't need backward, all gradients of their
        input/output tensors will make to None (despite require_grad is True)

        Note grad of input tensors of a IRPyFunc will be None and we will not
        generate a backward node for IRPyFunc.

        Args:
            loss (IRSubTensor): the loss tensor, must be in the output
            of current graph. The loss shape should be (1,)

        Returns:
            self (IRGraph): updated graph with backward operators
        """
        # set mirror as self
        self._mirror = self

        if loss is not None:  # optimize graph with loss
            # set loss gradient
            loss.parent.to_loss()

        # update input gradient
        # Please note `infer_grad` will not set the grad of input tensors.
        for t in IRGraph.get_objects_from_complex(self.inputs()):
            if isinstance(t, IRSubTensor) and t.requires_grad:
                t.grad = t.parent.grad.tosub()

        # update output gradient
        # Please note `infer_grad` will not set the grad of output tensors.
        for t in IRGraph.get_objects_from_complex(self.outputs()):
            if isinstance(t, IRSubTensor) and t.requires_grad:
                t.grad = t.parent.grad.tosub()

        # infer gradient
        for ftensor in self.full_tensors():
            self.infer_grad(ftensor)

        # create backward node
        for fnode in self.nodes()[::-1]:
            assert not isinstance(fnode, IRSegment), "Internal Error: Segment should not appear for now"
            if not isinstance(fnode, IRFwOperation): continue
            outputs = [t for t in fnode.outputs() if isinstance(t, IRSubTensor)]
            # no backward op generated for fnode
            if all(t.grad is None for t in outputs):
                continue
            # create backward op and insert to graph
            bwop = self.create_bwop(fnode)
            self.insert(bwop, self.nnodes)

        return self

    # ========================== Graph Creation ========================

    @staticmethod
    def from_logic_graph(nodes: List[IRCell],
                         inputs: List[Any], outputs: List[Any],
                         module_name: str):
        """Generate IRGraph from logical graph (IRFullTensor)

        Args:
            nodes (List[IRCell]): nodes of the graph
            inputs (List[Any]): graph inputs
            outputs (List[Any]): graph outputs
            module_name (str): graph name

        Returns:
            IRGraph: the graph with each tensor is IRSubTensor.
        """
        # currently fx graph always has only one output
        assert len(outputs) == 1, "Single output graph is expected"
        if isinstance(outputs[0], tuple):
            # fx graph will always wrap the graph output with a tuple of outputs
            # case 1: the return value of graph looks like `return x, y, z`
            #    here `outputs` will be `[(x,y,z)]`
            #    we will remove the outer tuple to make graph outputs[0]/[1]/[2] as x/y/z respectively
            # case 2: the return value of graph is a single value `return [[x]]`
            #    here `outputs` will be `[[[x]]]`
            #    just meet our requirement, no need to change
            #    the graph outputs[0] is `[[x]]``
            # case 3: the return value of graph is a single value `return x`
            #    here `outputs` will be `[x]`
            #    just meet our requirement, no need to change
            #    the graph outputs[0] is `x`
            # Please note that
            # 1. we treat `return x, y, z` and `return tuple(x, y, z)` as the same
            # 2. we treat `return (x,)` and `return x` as the same
            # Case 2 can lead to problem because it changes the return of `module.forward`,
            #    so we will raise error for this case for now.
            outputs = outputs[0]
            if isinstance(outputs, tuple) and len(outputs) == 1:
                raise RuntimeError("Single tuple outputs (like `return (x,)`) is not supported")

        modifier = lambda t: t.tosub() if isinstance(t, IRFullTensor) else t
        # input / output
        inputs = [IRCell.modify_objects_of_complex(t, modifier) for t in inputs]
        outputs = [IRCell.modify_objects_of_complex(t, modifier) for t in outputs]
        # nodes
        for node in nodes:
            for idx, ftensor in enumerate(node.inputs()):
                subtensor = IRCell.modify_objects_of_complex(ftensor, modifier)
                node.set_input(idx, subtensor)
            for idx, ftensor in enumerate(node.outputs()):
                subtensor = IRCell.modify_objects_of_complex(ftensor, modifier)
                node.set_output(idx, subtensor)
            for key in node.kwargs.keys():
                subtensor = IRCell.modify_objects_of_complex(node.kwargs[key], modifier)
                node.set_kwarg(key, subtensor)

        graph = IRGraph(nodes, inputs, outputs, module_name)

        # check IRPyFunc
        requires_grad_pyfunc: List[IRPyFunc] = []
        for node in nodes:
            if not isinstance(node, IRPyFunc): continue
            if any(isinstance(t, IRSubTensor) and t.requires_grad for t in node.outputs()):
                requires_grad_pyfunc.append(node)
        if len(requires_grad_pyfunc) > 0:
            dscp = (f'nnScaler does not support to compute gradients for IRPyFunc.\n'
                    f'Following nodes require gradients, this may trigger error in backward:\n')
            for node in requires_grad_pyfunc:
                dscp += f'\t{node.signature}, cid: {node.cid}\n'
            _logger.warning(dscp)

        # check unused outputs
        unused_obj_nodes: Dict[IRObject, List[IRCell]] = {}
        graph_output_objects = [
            obj.parent for obj in IRSegment.get_objects_from_complex(graph.outputs())]
        for obj in graph.full_objects():
            # loss tensor will always not used
            if isinstance(obj, IRFullTensor) and obj.is_loss(): continue
            # we don't need to show unused backward ops
            if isinstance(obj, IRFullTensor) and obj.is_grad(): continue
            consumers = graph.consumers(obj)
            if len(consumers) == 0 and obj not in graph_output_objects:
                producers = [n for n in graph.producers(obj) if not isinstance(n, IRGraphAnchor)]
                if len(producers) > 0:
                    unused_obj_nodes.setdefault(obj, []).extend(graph.producers(obj))
        if len(unused_obj_nodes) > 0:
            dscp = (f'Following returns of nodes are not used by any other nodes.\n'
                    f'Please consider to remove them in the user defined model.\n')
            for obj, unused_nodes in unused_obj_nodes.items():
                dscp += f'{obj}:\n'
                for node in unused_nodes:
                    if node.comment is not None:
                        dscp += f'\t{node.comment}\n\t{node.name} (cid={node.cid})\n'
                    else:
                        dscp += f'\t{node.name} (cid={node.cid})\n'
            _logger.warning(dscp)

        return graph

    def use_dataloader_input(self):
        """
        connect the graph with dataloader input.
        """
        # replace graph inputs with dataloader
        # the IRObject representing the `dataloader` instance, which is only used by the
        # IRDataOperation. Since we already know the output of the dataloader,
        # we don't need to set the value for it.
        ir_root_obj = IRObject(name='dataloader', value=None, is_constant=False)
        data_op = IRDataOperation(ir_root_obj, self.inputs())
        # add the data operation to the graph, which will use `next` to get data.
        self.insert(data_op, 0)
        self.reset_inputs(1)
        self.set_input(0, ir_root_obj)

    def no_backward(self):
        """
        Set all tensors with requires_grad=False to simulate no backward scenario (inference only).
        """
        if any(isinstance(node, IRBpOperation) for node in self.nodes()):
            raise RuntimeError("Cannot set no_backward for a graph with backward operators")
        for ftensor in self.full_tensors():
            ftensor.requires_grad = False

    ##### Transformation Primitives #####

    def replicate(self, node: Union[IRFwOperation, IRDataOperation], times=1) -> List[IRCell]:
        """
        Partition Primitive:
            - replicate: replicate a forward or data operation multiple times.

        Each input and output will be replicated with no gradient accumulation.

        The backward of the forward operation will automatically be replicated.

        @param node Union[IRFwOperation, IRDataOperation]

        @return ops List[IRCell]: the replicated operators
        """
        if not isinstance(node, (IRFwOperation, IRDataOperation)):
            raise TypeError("Expected op to be forward op or data op")
        if not isinstance(times, int) or times < 1:
            raise TypeError("Expected times to be int and >= 1")
        if node.name == 'multiref':
            return [node]
        if isinstance(node, IRPyFunc):
            return [node]

        fsegment: IRSegment = self.segment(node)
        # replicate
        fnodes = [node.replicate() for _ in range(times)]
        # set gradient
        for fnode in fnodes:
            for rtensor, itensor in zip(fnode.inputs(), node.inputs()):
                if isinstance(rtensor, IRSubTensor):
                    rtensor.grad = copy.copy(itensor.grad)
            for rtensor, itensor in zip(fnode.outputs(), node.outputs()):
                if isinstance(rtensor, IRSubTensor):
                    rtensor.grad = copy.copy(itensor.grad)
        # insert forward
        for fnode in fnodes:
            self.copy_node_meta_info(node, fnode)
        fsegment.replace(node, fnodes)
        # insert backward
        bsegment: IRSegment = fsegment.mirror
        if isinstance(node.mirror, IRCell):
            bnodes = [node.mirror.replicate() for _ in range(times)]
            for bnode, fnode in zip(bnodes, fnodes[::-1]):
                IRCell.make_pair(fnode, bnode)
                bnode.device = fnode.device
            bsegment.replace(node.mirror, bnodes)
        return fnodes

    def partition(self, node: Union[IRFwOperation, IRDataOperation],
                  algo: GenericDistAlgo, **config) -> List[IRCell]:
        """
        Partition Primitive:
            - partition: partition a forward or data operation using algorithms.

        The comment in the node will be inherited to partitioned nodes.
        The backward of the forward operation will be automatically partitioned.

        Requirement to partition algorithm:
            if backward is required, the algorithm can only transform tensors in:
                replicate: results in gradient accumulation
                split dimensionL no gradient accumulation
                split value (outputs only): no gradient accumulation

        Difference of partition and replicate primitive:
          Both primitive may replicate the tensors, but `replicate` will not do gradient
          accumulation while `partition` will always require gradient accumulation on
          replicated tensors.

        @param node Union[IRFwOperation, IRDataOperation]: the node to partition
        @param algo GenericDistAlgo: the partition algorithm related to the node
        @param config Dict[str, Any]: the algorithm configuration, e.g., partition number

        @return ops List[IRCell]: partitioned sub-nodes
        """
        assert isinstance(algo, GenericDistAlgo) and node == algo.node, \
            f"The partition algorithm ({algo}) is not initialized for this node"
        assert isinstance(node, (IRFwOperation, IRDataOperation)), \
            f"Only allow op to be forward op or data op, but got: {node}"
        if node.name == 'multiref':
            _logger.warning(f'skip partitioning multiref ({node.cid}), which will be handled by system.')
            return [node]
        if isinstance(node, IRPyFunc):
            _logger.warning(f'skip partitioning pyfunc ({node.cid}), which will be handled by system.')
            return [node]

        # get partitioned sub-nodes
        fnodes = algo.instantiate(**config)
        if not fnodes:
            raise ValueError(f"Fail to partition node: {node}. Please check your config: {config}.")

        # insert forward node
        fsegment: IRSegment = self.segment(node)
        fsegment.replace(node, fnodes)

        if node.mirror is None: return fnodes

        valmaps: Dict[IRFullTensor, Optional[ValueMap]] = dict()
        for t in node.inputs() + node.outputs():
            if isinstance(t, IRSubTensor):
                valmaps[t.parent] = None if t.grad is None else ValueMap(t.grad.valmap)

        # gather consumers
        ctensors: Dict[IRFullTensor, List[IRSubTensor]] = dict()
        consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        for fnode in fnodes:
            for itensor in set(IRSegment.get_objects_from_complex(fnode.inputs())):
                if not isinstance(itensor, IRSubTensor): continue
                ctensors.setdefault(itensor.parent, []).append(itensor)
                consumers.setdefault(itensor.parent, []).append(fnode)
        # set up gradient
        for fnode in fnodes:
            for itensor in fnode.inputs():
                if not isinstance(itensor, IRSubTensor): continue
                ftensor = itensor.parent
                itensor.grad = None
                if valmaps[ftensor] is None: continue
                # collect consumers that consume the same sub_tensor
                consumers_of_same_tensor = []
                for idx, t in enumerate(ctensors[ftensor]):
                    if t == itensor:
                        consumers_of_same_tensor.append(consumers[ftensor][idx])
                consumers_of_same_tensor = consumers_of_same_tensor[::-1]  # make valmap grow with exec order
                # calculate value map
                valmap = valmaps[ftensor].map(
                    (consumers_of_same_tensor.index(fnode), len(consumers_of_same_tensor))
                )
                grad = ftensor.grad.select(itensor.indmap, valmap)
                itensor.grad = grad
            for otensor in fnode.outputs():
                if not isinstance(otensor, IRSubTensor): continue
                otensor.grad = None if valmaps[otensor.parent] is None else \
                    otensor.parent.grad.select(otensor.indmap, (0,1))

        # insert backward node
        bnodes = [fsegment.create_bwop(fnode) for fnode in fnodes[::-1]]
        for bnode in bnodes:
            bnode.device = node.device
        bsegment: IRSegment = fsegment.mirror
        bsegment.replace(node.mirror, bnodes)

        return fnodes

    def fuse(self, nodes: List[IRFwOperation],
             signature: Optional[str] = None,
             fuse_op_args: Optional[List[IRObject]] = None,
             fuse_op_kwargs: Optional[Dict[str, Any]] = None,
             fuse_op_outputs: Optional[List[IRObject]] = None,
             fuse_op_anno: str = None,
             fuse_op_name: str = None) -> IRDimops:
        """Fuse primitive.

        Fuse a list of forward operators into a single operator.
        The backward operators will be fused automatically.

        Note:
            1) fusion can by applied for consecutive operators on the same device (developer-level call).
            2) fusion can be applied before any node paritioning or after generation of adapters (system-level call).

        Args:
            nodes (List[IRFwOperation]): the operators to fuse.
            signature (Optional[str], optional):
                the signature of the fused operator. If not provided, the fusion will perform a simple grouping of operators,
                where the underlying runtime still call the unfused kernel one by one. If the signature is provided,
                the fusion will generate an IRDimops calling `signature`, which is expected to be a function signature
                of the fused operator. Defaults to None.
            fuse_op_args (Optional[List[IRObject]], optional):
                the arguments of the fused operator. Defaults to None.
            fuse_op_kwargs (Optional[Dict[str, Any]], optional):
                the keyword arguments of the fused operator. Defaults to None.
            fuse_op_outputs (Optional[List[IRObject]], optional):
                the outputs of the fused operator. Defaults to None.
            fuse_op_anno (str, optional):
                the annotation of the fused operator. Defaults to None.
            fuse_op_name (str, optional):
                the name of the fused operator. Defaults to None.

        Returns:
            IRDimops: the fused operator.
        """
        assert len(nodes) > 0, "Cannot fuse empty list of nodes"
        assert all([isinstance(node, IRFwOperation) for node in nodes]), \
            "Only forward operators are allowed to fuse"
        indices: List[int] = [self.index(node).indices[-1] for node in nodes]
        assert max(indices) - min(indices) + 1 == len(nodes), \
            "Only consecutive operators can be fused"

        segment: IRSegment = self.create_segment(nodes)
        # get inputs where tensors should appear in the front.
        inputs = list(segment.inputs())
        attributes = [segment.ctensors(attr)[0] for attr in segment.attributes()]
        inputs += attributes
        inputs = [t for t in inputs if isinstance(t, IRTensor)] + [t for t in inputs if not isinstance(t, IRTensor)]
        # get outputs
        outputs = list(segment.outputs())

        # reorder and check op inputs and outputs
        if fuse_op_args is not None:
            assert len(inputs) == len(fuse_op_args) and set(inputs) == set(fuse_op_args), \
                "inputs don't match"
            inputs = fuse_op_args
        kwargs = {} if fuse_op_kwargs is None else fuse_op_kwargs
        if fuse_op_kwargs is not None:
            assert len(outputs) == len(fuse_op_outputs) and set(outputs) == set(fuse_op_outputs), \
                "outputs don't match"
            outputs = fuse_op_outputs

        # create annotation. TODO: support partition
        if fuse_op_anno is None:
            in_shapes = [[str(dimlen) for dimlen in t.shape] for t in inputs if isinstance(t, IRTensor)]
            ou_shapes = [[str(dimlen) for dimlen in t.shape] for t in outputs if isinstance(t, IRTensor)]
            fuse_op_anno: str = OpAnno.create_op_str(in_shapes, ou_shapes)

        if fuse_op_name is None:
            if len(nodes) < 4:
                fuse_op_name = '_'.join(['fused'] + [node.name for node in nodes])
            else:
                fuse_op_name = '_'.join(['fused'] + [node.name for node in nodes[:3]] + ['etc'])

        # if signature is not provided, register the fused function by
        # grouping the node implementations together inside a function.
        # This doesn't make real fusion but can help reduce partition
        # search space for the policy.
        make_customized_op: bool = signature is None
        if signature is None:
            signature = f'{fuse_op_name}_{nodes[0].cid}_to_{nodes[-1].cid}'

        def fuse_op_fn(*args, **kwargs) -> IRDimops:
            return IRDimops(fuse_op_fn, fuse_op_name, signature, [fuse_op_anno], args, **kwargs)

        if make_customized_op:
            from nnscaler.graph.parser.register import CustomizedOps

            def to_name(t: Any) -> str:
                """Convert an object to its name."""
                if isinstance(t, IRObject):
                    return '_'.join([t.name, str(t.tid)])
                elif isinstance(t, str) and not t.startswith('self.'):
                    return f"'{t}'"
                return str(t)
            # function inputs / outputs
            func_inputs = ','.join(to_name(t) for t in inputs)
            func_kwargs = ','.join(f'{k}={to_name(v)}' for k, v in kwargs.items())
            func_outputs = ','.join([to_name(t) for t in outputs])
            # generate code
            code = [f'def {signature}({func_inputs}, {func_kwargs}):']
            for node in nodes:
                node_inputs = ','.join(to_name(t) for t in node.inputs())
                node_kwargs = ','.join(f'{k}={to_name(v)}' for k, v in node.kwargs.items())
                node_outputs = ','.join(to_name(t) for t in node.outputs()) if len(outputs) > 0 else '_'
                code += [f'\t{node_outputs} = {node.signature}({node_inputs}, {node_kwargs})']
            code.append(f'\treturn {func_outputs}')
            code = '\n'.join(code)
            CustomizedOps.register(
                signature, fuse_op_fn, code,
                lambda *args : NotImplementedError("a fused operator doesn't have runtime call")
            )

        fuse_op = fuse_op_fn(*inputs, **kwargs)
        for idx, output in enumerate(outputs):
            fuse_op.set_output(idx, output)

        # setup device
        if len(nodes[0].device) != 0:
            fuse_op.device = nodes[0].device

        # replace nodes with the fused operator
        # remove forward operators
        segment = self.segment(nodes[0])
        indices = [segment.remove(node).indices[-1] for node in nodes]
        idx = min(indices)
        # remove backward operators
        have_backward = any(node.mirror is not None for node in nodes)
        for node in nodes:
            if node.mirror is not None:
                segment.mirror.remove(node.mirror)
        # insert forward/backward operators
        if have_backward:
            segment.finsert(fuse_op, idx)
        else:
            segment.insert(fuse_op, idx)

        return fuse_op

    ## Spatial Primitives ##

    def assign(self, node: Union[IRFwOperation, IRDataOperation], device: int) -> bool:
        """
        Assign an operator (subgraph) to (multiple) rank(s).

        Corresponding backward operators (if have) will also be
        assigned to the same device.

        @param node Union[IRFwOperation, IRBpOperation, IRSegment]: operator
        @param device int: assigned device id

        @return sucess bool: always true
        """
        assert self.exist(node), f"{node} is not in the graph"
        if isinstance(node, IRSegment):
            assert node.isfw(), "Only forward segment is allowed to assign devices"
            for subnode in node.nodes():
                self.assign(subnode, device)
        else:
            assert isinstance(node, (IRFwOperation, IRDataOperation)), \
                "Only forward operators and dataloader operators are allowed to assign devices"
            node.device = device
            if node.mirror is not None:
                node.mirror.device = device
        return True

    def reside(self, tensor: IRSubTensor, devices: Union[int, List[int]]):
        """
        Allocate an attribute tensor to devices.
        """
        assert tensor.is_attr(), f"Only support to set devices for graph attribute tensors"
        raise NotImplementedError("Not supported yet")

    ## Schedule Policy Primitives ##

    def sequential(self, prev_nodes: Tuple[IRFwOperation], succ_nodes: Tuple[IRFwOperation]):
        """Schedule primitive: schedule prev_nodes right before the succ_nodes

        The position of `succ_nodes` will keep unchanged in the sequence
        while the `prev_nodes` will be scheduled right before the `succ_nodes`.
        Corresponding backward operators will also be re-ordered.

        The `prev_nodes` should be consecutive in the sequence.
        The `succ_nodes` should be consecutive in the sequence.

        Args:
            prev_nodes (Tuple[IRFwOperation]): the nodes to be scheduled right before `succ_nodes`
            succ_nodes (Tuple[IRFwOperation]): the nodes to be executed right after `prev_nodes`

        Returns:
            None
        """
        prev_indices = [self._nodes.index(n) for n in prev_nodes]
        succ_indices = [self._nodes.index(n) for n in succ_nodes]
        if len(prev_nodes) != max(prev_indices) - min(prev_indices) + 1:
            raise ValueError(
                f'prev_nodes are expected to be consecutive in node sequence: '
                f'{len(prev_nodes)} != {max(prev_indices) - min(prev_indices) + 1}'
            )
        if len(succ_nodes) != max(succ_indices) - min(succ_indices) + 1:
            raise ValueError(
                f'succ_nodes are expected to be consecutive in node sequence: '
                f'{len(succ_nodes)} != {max(succ_indices) - min(succ_indices) + 1}'
            )
        # check duplication
        if len(set(prev_indices)) != len(prev_indices):
            raise ValueError(f'find duplicated node in prev nodes')
        if len(set(succ_indices)) != len(succ_indices):
            raise ValueError(f'find duplicated node in succ nodes')
        if len(set(prev_indices).intersection(set(succ_indices))) != 0:
            raise ValueError(f'find duplicated node in both succ_nodes and prev_nodes')
        # TODO: check dependency

        seq = list(self._nodes)
        # cut out prev_nodes
        fstart, fend = min(prev_indices), max(prev_indices) + 1
        fnodes = seq[fstart:fend]
        seq = seq[:fstart] + seq[fend:]
        # insert prev_nodes
        ofst = min(succ_indices)
        if max(prev_indices) < min(succ_indices):
            ofst = ofst - len(fnodes)
        seq = seq[:ofst] + fnodes + seq[ofst:]

        # update order of backward node
        prev_bnodes = [n.mirror for n in prev_nodes[::-1] if n.mirror is not None]
        succ_bnodes = [n.mirror for n in succ_nodes[::-1] if n.mirror is not None]
        prev_bindx = [seq.index(n) for n in prev_bnodes]
        succ_bindx = [seq.index(n) for n in succ_bnodes]
        if len(prev_bnodes) > 0:
            # TODO: extend succ_nodes to find at least one forward op that has backward
            if len(succ_bnodes) == 0:
                raise NotImplementedError(f'backward of succ_nodes are expected')
            # cut out prev_backward_nodes
            bstart, bend = min(prev_bindx), max(prev_bindx) + 1
            bnodes = seq[bstart:bend]
            seq = seq[:bstart] + seq[bend:]
            # insert prev_backward_nodes
            ofst = max(succ_bindx) + 1
            if max(prev_bindx) < min(succ_bindx):
                ofst = ofst - len(bnodes)
            seq = seq[:ofst] + bnodes + seq[ofst:]
        # update sequence
        self._nodes = seq

    def depends(self, pre_node: IRCell, succ_node: IRCell) -> bool:
        """Check direct data dependency between two nodes.

        Check dependency of pre_node -> post_node.

        Note this function only checks direct data dependency that whether
        the outputs in `prev_node` and inputs in `post_node` have data dependency.

        The function cannot detect data dependency in graph like:
            pre_node -> (some nodes) ... -> post_node

        Args:
            pre_node (IRCell): the happen before node
            post_node (IRCell): the happen after node

        Returns:
            ret (bool): True if post_node depends on pre_node on dataflow, otherwise False.
        """
        input_objs = IRSegment.get_objects_from_complex(succ_node.inputs())
        for out_obj in IRSegment.get_objects_from_complex(pre_node.outputs()):
            for in_obj in input_objs:
                if out_obj.overlap(in_obj):
                    return True
        return False

    @property
    def sched(self):
        """ Get bound schedule plan

        Returns:
            sched (SchedulePlan | None): bound schedule plan
        """
        return self._sched

    def _bind_schedule(self, schedplan):
        """Set schedule plan for the execution

        This will be called when initiating a schedule plan for the graph.

        Args:
            schedplan (SchedulePlan)

        Returns:
            None
        """
        from nnscaler.graph.schedule import SchedulePlan
        if not isinstance(schedplan, SchedulePlan):
            raise TypeError(f"Expect a SchedulePlan but got: {type(schedplan)}")
        assert self._sched is None, "The graph is already bound with one schedule plan."
        self._sched = schedplan

    # ================= staging primitives ==================

    def group(self, nodes: List[IRCell]) -> IRSegment:
        """Group consecutive nodes into IRSegment.

        Note nodes should not have applied by any transformation.

        Args:
            nodes List[IRCell]: consecutive nodes in forward procedure

        Returns:
            segment IRSegment: the grouped segment
        """
        assert all(node.isfw() for node in nodes), f"Expected all nodes in forward procedure"
        fgraphs = [self.segment(fnode) for fnode in nodes]
        assert len(set(fgraphs)) == 1, "cross-segment grouping is not allowed yet."

        fgraph: IRSegment = fgraphs[0]
        findices: Tuple[int] = tuple(fgraph.index(node)[0] for node in nodes)
        min_fidx, max_fidx = min(findices), max(findices)
        assert max_fidx - min_fidx + 1 == len(nodes), "nodes should be in consecutive order"

        fsegment: IRSegment = fgraph.create_segment(nodes)
        for node in nodes:
            idx = fgraph.remove(node)
        fgraph.insert(fsegment, idx)

        # group for mirror nodes
        bnodes = [node.mirror for node in nodes if node.mirror is not None]
        if len(bnodes) == 0: return fsegment

        # check consecutive
        bgraph: IRSegment = fgraph.mirror
        bindices = [bgraph.index(bnode)[0] for bnode in bnodes]
        min_bidx, max_bidx = min(bindices), max(bindices)
        assert max_bidx - min_bidx + 1 == len(bnodes), \
            f"backward nodes are not consecutive. minbidx: {min_bidx}, maxbidx: {max_bidx}"

        # update gradient for fgraph
        for itensor in fsegment.inputs():
            if not isinstance(itensor, IRTensor): continue
            fgraph.infer_grad(itensor.parent)
        # update gradient inside segment
        for ftensor in fsegment.full_tensors():
            fsegment.infer_grad(ftensor)

        # create backward segment
        for bnode in bnodes:
            bidx = bgraph.remove(bnode)
        bnodes = [fsegment.create_bwop(fnode) for fnode in nodes[::-1] if fnode.mirror is not None]
        # get backward graph inputs
        output_grads = [t.grad for t in fsegment.outputs() if isinstance(t, IRSubTensor) and t.grad is not None]
        # get backward graph outputs
        input_grads = [t.grad for t in fsegment.inputs() if \
                       isinstance(t, IRSubTensor) and t.grad is not None]
        bsegment = IRSegment(bnodes, output_grads, input_grads)

        bgraph.insert(bsegment, bidx)
        IRCell.make_pair(fsegment, bsegment)
        return fsegment

    def blocking(self, nodes: Tuple[IRFwOperation]):
        """Group forward operators into blocks.

        The corresponding backward operators (if have) will also be grouped into stages
        Cross-stage dataflow will be limited to neighbor stages.
        This should be called before any operator partition.

        Args:
            nodes Tuple[IRFwOperations]: the start forward node of each stage.

        Returns:
            None
        """
        assert all(isinstance(node, IRFwOperation) for node in nodes), \
            f"Find node is not IRFwOperation or IRDataOperation: {node}"
        assert all(node in self._nodes for node in nodes), \
            f"Exist node is not in graph nodes"
        starts = list(self._nodes.index(node) for node in nodes)
        assert len(starts) > 0

        # multiref (created by graph.multiref) will be moved to the next stage (if possible) for optimization
        for sid in range(len(starts)):
            while starts[sid] > 0:
                node = self.node(starts[sid]-1)
                if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                    starts[sid] -= 1
                    continue
                break

        # adjust the start of the first stage to involve beginning operators
        for idx in range(starts[0]):
            node = self.node(idx)
            if isinstance(node, IRDataOperation):
                continue
            assert isinstance(node, IRFwOperation), \
                f"Expected nodes previous from the first stage are all IRFwOperation, but got {type(node)}"
            if node.name == 'multiref' or isinstance(node, IRPyFunc):
                pass
            else:
                _logger.warning(f'Detect a node: {node} that is previous from the first stage. Will be included inside the first stage')
            starts[0] = idx
            break

        last_fidx = 0
        for idx, node in enumerate(self._nodes):
            if not isinstance(node, IRBpOperation):
                last_fidx = idx

        fstages: List[List[IRCell]] = []
        for sid in range(len(starts)):
            begin = starts[sid]
            end = starts[sid+1] if sid != len(starts) - 1 else last_fidx + 1
            if begin >= end:
                _logger.warning(f"Detected stage {sid} doesn't have operators: [begin({begin}): end({end})). Skipped")
                continue
            fnodes = self._nodes[begin:end]
            assert all(isinstance(node, IRFwOperation) for node in fnodes), \
                f"find at least one nodes are not of IRFwOperation in the stage {sid}. They should be moved to the front"
            fstages.append(fnodes)

        # grouping into segment
        for sid in range(len(fstages)):
            self.group(fstages[sid])

    def staging(self, nodes: Tuple[IRFwOperation]):
        """Group forward operators into sequential stages.

        The corresponding backward operators (if have) will also be grouped into stages
        Cross-stage dataflow will be limited to neighbor stages.
        This should be called before any operator partition.

        The transformation and temporal scheduling can only be applied within each stage.
        For example, after staging, user cannot schedule a (transformed) node
        from one stage to another stage.

        Changes will be made:

        1). Identity creation:
            If a non-attribute tensor is produced / consumed not in
            neighbor stages,
                e.g.,
                    stage 1: t1 = producer()
                    stage 2: ...
                    stage 3: xx = consume(t1)
                             xx = consume(t1)
                    stage 4: ...
                    stage 5: xx = consume(t1)
            then Identity nodes will be created for every device in stage2:
                    stage 1: t1 = producer()
                    stage 2: t2 = identity(t1)
                    stage 3: t3 = identity(t2)
                             xx = consume(t3)
                             xx = consume(t3)
                    stage 4: t4 = identity(t3)
                    stage 5: t5 = identity(t4)
                             xx = consume(t5)

        Args:
            nodes Tuple[IRFwOperations]: the start forward node of each stage.

        Returns:
            None
        """
        for node in nodes:
            assert isinstance(node, IRFwOperation), f"Expected node to be IRFwOperation, but got {node}"
        assert all(node in self._nodes for node in nodes), \
            f"Exist node is not in graph nodes"
        starts = list(self._nodes.index(node) for node in nodes)
        assert len(starts) > 0

        # multiref (created by graph.auto_multiref) will be moved to the next stage (if possible) for optimization
        for sid in range(len(starts)):
            while starts[sid] > 0:
                node = self.node(starts[sid]-1)
                if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                    starts[sid] -= 1
                    continue
                break

        # adjust the start of the first stage to involve beginning operators
        for idx in range(starts[0]):
            node = self.node(idx)
            # IRDataOperation cannot be involved in the IRSegment in current
            # implementation.
            if isinstance(node, IRDataOperation): continue
            assert isinstance(node, IRFwOperation), \
                f"Expected nodes previous from the first stage are all IRFwOperation, but got {type(node)}"
            _logger.info(f'involve node {node.name}({node.cid}) into the first stage')
            starts[0] = idx
            break

        last_fidx = 0
        for idx, node in enumerate(self._nodes):
            if not isinstance(node, IRBpOperation):
                last_fidx = idx

        fstages: List[List[IRCell]] = []
        bstages: List[List[IRCell]] = []
        for sid in range(len(starts)):
            begin = starts[sid]
            end = starts[sid+1] if sid != len(starts) - 1 else last_fidx + 1
            if begin >= end:
                _logger.warning(
                    f"skip stage {sid} which doesn't have operators: [begin({begin}): end({end})).")
                continue
            fnodes = self._nodes[begin:end]
            assert all(isinstance(node, IRFwOperation) for node in fnodes), \
                f"find at least one nodes are not of IRFwOperation in the stage {sid}. They should be moved to the front"
            bnodes = [fnode.mirror for fnode in fnodes[::-1] if fnode.mirror is not None]
            fstages.append(fnodes)
            bstages = [bnodes] + bstages

        def get_sid(fnode: IRCell) -> Optional[int]:
            for idx, fnodes in enumerate(fstages):
                if fnode in fnodes:
                    return idx
            return None

        def insert_identity(tensor: IRSubTensor, sid: int) -> IRFwOperation:
            fwop = Identity(tensor)
            output = tensor.parent.like().tosub()
            fwop.set_output(0, output)
            if tensor.requires_grad:
                # set input grad
                igrad = tensor.parent.grad.select(tensor.indmap, tensor.valmap)
                fwop.input(0).grad = igrad
                # set output grad
                otensor = fwop.output(0).parent
                ograd = otensor.grad.select(tensor.indmap, (0,1))
                fwop.output(0).grad = ograd
                # insert identity
                fidx = self.index(fstages[sid][0])
                self.finsert(fwop, fidx)
            else:
                fidx = self.index(fstages[sid][0])
                self.insert(fwop, fidx)
            # update stage op group
            fstages[sid].insert(0, fwop)
            if isinstance(fwop.mirror, IRCell):
                bstages[sid].append(fwop.mirror)
            return fwop

        # create identity op for cross-stage dataflow
        for ftensor in self.full_tensors():
            if ftensor.is_grad() or ftensor.is_attr(): continue
            if len(self.consumers(ftensor)) == 0: continue

            assert len(self.producers(ftensor)) <= 1, \
                "The staging interface should be called before any operator partition."
            ctensors = self.ctensors(ftensor)
            if len(self.ctensors(ftensor)) > 0:
                assert all(ctensor == ctensors[0] for ctensor in ctensors), (
                    "The staging interface should be called before any operator partition."
                )

            producer, ptensor = self.producers(ftensor)[0], self.ptensors(ftensor)[0]
            psid = get_sid(producer)
            # outside of stages, not consider
            if psid is None: continue

            # group consumers into stages
            consumers = self.consumers(ftensor)
            csids = [get_sid(consumer) for consumer in consumers]
            buckets = [[] for _ in range(len(fstages))]
            for idx, csid in enumerate(csids):
                buckets[csid].append(consumers[idx])

            # go through each stage to generate identity operators
            out = ptensor
            end_sid = max(csids) + 1
            for sid in range(psid + 1, end_sid):
                # insert identity
                op = insert_identity(out, sid)
                out = op.output(0)
                # calculate gradient
                curr_valmap = ValueMap((0, 1))
                nconsumers = len(buckets[sid])
                fgrad = ftensor.grad
                for cidx, consumer in enumerate(buckets[sid]):
                    if fgrad is None:
                        grad = None
                    else:
                        valmap = curr_valmap.map((0, 2)) if cidx != nconsumers - 1 else curr_valmap
                        grad = fgrad.select(ptensor.indmap, valmap)
                        curr_valmap = curr_valmap.map((1, 2)) if cidx != nconsumers - 1 else curr_valmap
                    # update forward consumer
                    idx = consumer.inputs().index(ptensor)
                    tensor = consumer.input(idx)
                    with self.update(consumer) as consumer:
                        consumer.set_input(idx, out)
                        consumer.input(idx).grad = grad
                    # update backward
                    if tensor.grad is not None:
                        with self.update(consumer.mirror) as bconsumer:
                            idx = bconsumer.outputs().index(tensor.grad)
                            bconsumer.set_output(idx, grad)

        # grouping into segment
        for sid in range(len(fstages)):
            self.group(fstages[sid])


    # ================= Other optimizations ==================

    def recompute(self, nodes: Union[IRSegment, List[IRFwOperation]]) -> bool:
        """!
        Recompute a set of nodes. The forward nodes will be assigned with a unique
        recompute group id. A forward not can not be recomputed in different recompute groups.

        @param nodes List[IRFwOperation]: nodes for a recompute group

        @return success boolean: always success
        """
        assert all(isinstance(node, IRFwOperation) for node in nodes) or isinstance(nodes, IRSegment), \
            "Require forward nodes or a single segment"

        if isinstance(nodes, IRSegment):
            assert nodes.isfw() and (not nodes.isbw()), "Only forward IRSegment can recompute"
            return self.recompute(nodes.nodes())

        else:
            segments = [self.segment(node) for node in nodes]
            assert all(segment == segments[0] for segment in segments), \
                "Cross-segment recompute is not allowed yet"
            recompute_group_id: int = IDGenerator().gen_cell_id()
            start = 0
            for fnode in nodes:
                tensors = [t for t in fnode.inputs() if isinstance(t, IRSubTensor) and (not t.is_attr())]
                if all(t.grad is None for t in tensors):
                    start += 1
                    continue
                break
            skip = nodes[:start]
            nodes = nodes[start:]
            end = len(nodes)
            for fnode in nodes[::-1]:
                tensors = [t for t in fnode.inputs() if isinstance(t, IRSubTensor) and (not t.is_attr())]
                if all(t.grad is None for t in tensors):
                    end -= 1
                    continue
                break
            skip += nodes[end:]
            for node in skip:
                if isinstance(node, IRGraphAnchor): continue
                _logger.info(
                    f"skip recompute node: {node.name} ({node.cid}) as "
                    f"it doesn't require gradient and appears at head or tail."
                )
            nodes = nodes[:end]
            for fnode in nodes:
                fnode.recompute = recompute_group_id
        return True

    # =================== Helpers ====================

    def dumps(self) -> str:
        """
        Dump the graph into binary by dill
        """
        # FIXME: dump doesn't support customized op
        class PicklingContextSave:
            def __enter__(self):
                IRObject.__getstate__ = IRObject.getstate_for_dump
            def __exit__(self, exc_type, exc_value, traceback):
                IRObject.__getstate__ = lambda self: self.__dict__.copy()

        with PicklingContextSave():
            save = (IDGenerator().get_states(), self)
            return dill.dumps(save)

    def dump(self, filename: str) -> None:
        """
        Dump the graph into pickled format

        @param filename str
        """
        with open(filename, 'wb') as f:
            f.write(self.dumps())

    @staticmethod
    def from_dill(id_state, graph):
        """
        build instance from id_state and graph
        Note IDGenerator will also be reset to match with graph status

        Args:
            id_state : read from dill
            graph (IRGraph): read from dill

        Returns:
            IRGraph: the build graph
        """

        # recover IRGenerator
        IDGenerator().load_states(id_state)
        # recover cell
        def reset_node(segment: IRSegment):
            # input
            for t in segment.inputs():
                if isinstance(t, IRObject):
                    t.cell = segment
            # nodes
            for node in segment.nodes():
                for t in node.inputs() + node.outputs():
                    if isinstance(t, IRObject):
                        t.cell = node
                # recursively recover segments
                if isinstance(node, IRSegment):
                    reset_node(node)
            # output
            for t in IRSegment.get_objects_from_complex(segment.outputs()):
                t.cell = segment

        reset_node(graph)
        return graph

    @staticmethod
    def load(filename: str):
        """
        Load the graph from pickled file.
        Note IDGenerator will also be reset to match with graph status

        Args:
            filename (str): the file to load

        Returns:
            IRGraph: the built graph
        """
        with open(filename, 'rb') as f:
            id_state, graph = dill.load(f)

        return IRGraph.from_dill(id_state, graph)

    def checksum(self, strict: bool = True) -> str:
        """Get the MD5 checksum of the graph.

        This is used to guarantee the consistency of the graph between
        multiple nodes.

        Note:
            The checksum considers the IDGenerator status. If the user modifies
            the IDGenerator status (i.e., creating tensors or nodes), it will
            have a different checksum.

        Args:
            strict (bool): If True (by default), get the checksum of the whole graph status,
                including tensor shapes, tensor ids and node ids;
                Otherwise (i.e., False), only check the graph structure of node ids,
                node signatures without tensor ids.

        Returns:
            str: MD5 checksum (32-bit) of the graph status
        """
        max_tensor_id, max_cell_id = IDGenerator().get_states()
        if not strict:
            node_ids = tuple(n.cid for n in self.nodes())
            signatures = tuple(n.signature for n in self.nodes())
            checksum = hashlib.md5(str((max_tensor_id, max_cell_id, signatures, node_ids)).encode()).hexdigest()
        else:
            states = str((max_tensor_id, max_cell_id, self.extra_repr()))
            checksum = hashlib.md5(states.encode()).hexdigest()
        return checksum

    @staticmethod
    def copy_node_meta_info(src_node: Union[IRFwOperation, IRDataOperation], dest_node: Union[IRFwOperation, IRDataOperation]):
        """
        Copy meta information from src_node to dest_node.
        Current copy fields: ['recompute', 'comment', 'op_context', 'module_stack', 'device']
        """
        if isinstance(src_node, IRFwOperation):
            dest_node.recompute = src_node.recompute
        if isinstance(src_node.comment, str):
            dest_node.comment = src_node.comment
        if src_node.op_context is not None:
            dest_node.op_context = src_node.op_context
        dest_node.module_stack = src_node.module_stack
        dest_node.device = src_node.device
