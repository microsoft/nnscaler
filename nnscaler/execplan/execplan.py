from typing import Callable, Dict, List, Optional, Tuple, Any
import copy
import numpy as np
import sys

from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.tensor import IRSubTensor, IRFullTensor
from nnscaler.ir.adapter import IRAdapter, IRWeightReducer
from nnscaler.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from nnscaler.graph.graph import IRGraph, IRSegment
from nnscaler.graph.schedule.schedplan import SchedulePlan, Block


class ExeReuseCell(IRCell):
    """
    A cell that reuses a cell with new inputs and outputs
    This is designed for code shrinking of repeatedly executing
    same operator-sequences, e.g., different micro-batches
    execute on a same code piece.
    """

    def __init__(self, cell: IRCell,
                 inputs: List[IRSubTensor], outputs: List[IRCell]):
        assert len(inputs) == len(cell.inputs())
        assert len(outputs) == len(cell.outputs()), (
            f"output length mismatch: {cell}\n"
            f"cell outputs: {cell.outputs()}\noutputs: {outputs}")
        super().__init__(cell.name, cell.signature,
                         len(inputs), len(outputs))
        for idx, t in enumerate(inputs):
            self.set_input(idx, t)
        for idx, t in enumerate(outputs):
            self.set_output(idx, t)
        self._cell: IRCell = cell
        self._cached_dispatched: Dict[int, ExeReuseCell] = {}

    @property
    def device(self) -> int:
        return self._cell.device
    
    @property
    def cell(self) -> IRCell:
        return self._cell
    
    def isfw(self) -> bool:
        return self._cell.isfw()
    
    def dispatch(self, devid: int, _mirror = True):
        assert len(self.device) > 0 and devid in self.device, f"Cannot dispatch of ReuseCell {self} to device {devid}"
        if devid in self._cached_dispatched:
            return self._cached_dispatched[devid]
        
        inputs = []
        for t, cell_t in zip(self._inputs, self._cell.inputs()):
            if isinstance(cell_t, IRSubTensor) and devid not in cell_t.device:
                continue
            inputs.append(t)
        outputs = []
        for t, cell_t in zip(self._outputs, self._cell.outputs()):
            if isinstance(cell_t, IRSubTensor) and devid not in cell_t.device:
                continue
            outputs.append(t)
        reuse = ExeReuseCell(self._cell.dispatch(devid), inputs, outputs)
        reuse._id = self._id
        if _mirror and self.mirror is not None:
            mreuse = self.mirror.dispatch(devid, _mirror=False)
            IRCell.make_pair(reuse, mreuse)
        self._cached_dispatched[devid] = reuse
        return reuse
    
    def __repr__(self) -> str:
        return f'ReuseCell-{self.device}(name={self._cell.name}{self._cell.cid}, inputs={self.inputs()}, outputs={self.outputs()})'



class ExecutionPlan:
    """
    Execution plan for runtime execution.
    Each device will be assigned by its execution sequence
    """

    @staticmethod
    def from_graph(graph: IRGraph):
        """
        Create execution plan from IRGraph
        """
        return ExecutionPlan(graph, graph.nodes())

    @staticmethod
    def from_schedplan(schedplan: SchedulePlan):
        """Create execution plan from SchedulePlan

        A schedule plan has multiple micro-batches, where each micro-batch
        goes through the all operators in the model graph. So an operator
        will be executed multiple times with different data from different micro-batches.
        
        The IRGraph only contains operators / IRTensors / IRObjects of one micro-batch. 
        To represent data of a different micro-batch, we need to map the data in IRGraph to a 
        new one with different IDs.
        """
        graph_inputs = schedplan.graph.inputs()
        micro_objs: Dict[int, Dict[IRObject, IRObject]] = {}
        def get(tensor: IRObject, micro_idx: int) -> IRObject:
            """Get an IRObject same to tensor, but with different tid for each given micro-batch index"""
            if not isinstance(tensor, IRObject): return tensor
            # NOTE: the graph inputs (e.g., dataloader) serves as the global variables during the
            # execution of schedules, where every micro-batch shares the same one for execution.
            # Typically, the graph inputs can be dataloader object
            if tensor in graph_inputs: return tensor
            if micro_idx == 0: return tensor
            if not isinstance(tensor, IRSubTensor):
                # IRObject but not IRSubTensor
                micro_objs.setdefault(micro_idx, {}).setdefault(tensor, IRObject(tensor.name, value=tensor.value))
                t = micro_objs[micro_idx][tensor]
            else:
                # IRSubTensor
                ftensor = micro_objs.setdefault(micro_idx, {}).setdefault(tensor.parent, tensor.parent.like())
                t = ftensor.select(tensor.indmap, tensor.valmap)
                if tensor.grad is not None:
                    fgrad: IRFullTensor = ftensor.grad
                    micro_objs.setdefault(micro_idx, {}).setdefault(tensor.parent.grad, fgrad)
                    t.grad = fgrad.select(tensor.grad.indmap, tensor.grad.valmap)
            return t
        
        micro_fcells: Dict[(int, IRCell), ExeReuseCell] = {}
        def block2reuse(node: Block) -> ExeReuseCell:
            if node.content.isfw():
                key = (node.mid, node.content)
                if key in micro_fcells:
                    return micro_fcells[key]
                inputs = [get(t, node.mid) for t in node.content.inputs()]
                outputs = [get(t, node.mid) for t in node.content.outputs()]
                cell = ExeReuseCell(node.content, inputs, outputs)
                if isinstance(node.content.mirror, IRCell):
                    minputs = [get(t, node.mid) for t in node.content.mirror.inputs()]
                    moutputs = [get(t, node.mid) for t in node.content.mirror.outputs()]
                    mcell = ExeReuseCell(node.content.mirror, minputs, moutputs)
                    IRCell.make_pair(cell, mcell)
                micro_fcells[key] = cell
                return cell
            else:
                mcell = block2reuse(Block(node.content.mirror, node.mid, node.span))
                return mcell.mirror
            
        topo_seqs: List[IRCell] = []
        for block in schedplan.nodes():
            if isinstance(block, Block):
                block = block2reuse(block)
            assert isinstance(block, IRCell)
            topo_seqs.append(block)

        # set up returning outputs by packing output results from each micro-batch into a list
        outputs = []
        for mid in range(schedplan.nmicros):
            outs = []
            for output in schedplan.graph.outputs():
                outs.append(IRSegment.modify_objects_of_complex(output, lambda x: get(x, mid)))
            if len(outs) > 0:
                outputs.append(outs[0] if len(outs) == 1 else outs)

        execplan = ExecutionPlan(schedplan.graph, topo_seqs)
        execplan.set_outputs(outputs)

        return execplan

    def __init__(self, graph: IRGraph, topo_seqs: List[IRCell]):

        assert isinstance(graph, IRGraph), "Expected an IRGraph"
        self._graph = graph
        self._topo_seqs = topo_seqs
        self._seq: Dict[int, List[IRCell]] = {}
        self._outputs = list(graph.outputs())

        for node in self._topo_seqs:
            assert len(node.device) > 0, f"Node device not set: {node}"
            for device in node.device:
                self._seq.setdefault(device, []).append(node)

        def cached_dispatch(node: IRCell, devid: int,
                            dispatched: Dict[IRCell, IRCell]) -> IRCell:
            """Cached dispatch"""
            if node.isfw() or isinstance(node, IRWeightReducer):
                return dispatched.setdefault(node, node.dispatch(devid))
            fnode = node.mirror
            assert isinstance(fnode, IRCell), "Expected forward node as mirror"
            assert fnode.isfw()
            # return dispatched[fnode].mirror
            return dispatched.setdefault(fnode, fnode.dispatch(devid)).mirror

        # dispatch for a node that is executed on multiple devices
        for devid, nodes in self._seq.items():
            dispatched : Dict[IRCell, IRCell] = {}
            for idx in range(len(nodes)):
                node = nodes[idx]
                # print(f'handling {node}')
                if len(node.device) == 1: continue  # no need for dispatch
                dnode = cached_dispatch(node, devid, dispatched)
                nodes[idx] = dnode

    @property
    def graph(self) -> IRGraph:
        return self._graph

    @property
    def inference(self) -> bool:
        return not self._graph.train

    def outputs(self) -> List[Any]:
        """Get execution plan return outputs"""
        return self._outputs

    def devices(self) -> List[int]:
        """
        Get device set
        """
        devices = list(self._seq.keys())
        devices.sort()
        return devices

    def seq(self, devid: int) -> List[IRCell]:
        """
        Get a view of execution sequence for device id

        Note changing the list content will not change the execution plan.
        """
        if devid not in self._seq:
            return []
        return copy.copy(self._seq[devid])

    def at(self, devid: int) -> List[IRCell]:
        """
        Access the sequence for device id

        Note changing the list content will change the execution plan.
        """
        if devid not in self._seq:
            return []
        return self._seq[devid]

    def flatten(self, devid: int) -> List[IRCell]:
        """
        Flatten the sequence by expanding segments
        """
        assert devid in self._seq, f"device id {devid} not exists"
        nodes = []
        for node in self._seq[devid]:
            if isinstance(node, IRSegment):
                nodes += node.nodes()
            else:
                nodes.append(node)
        return nodes

    def set(self, devid: int, seq: List[IRCell]):
        """
        Set device sequence
        """
        if not all([isinstance(su, IRCell) for su in seq]):
            raise TypeError("Expected a list of Cell")
        self._seq[devid] = seq

    def set_outputs(self, outputs: List[Any]):
        if not isinstance(outputs, list):
            raise TypeError("Expected a list of outputs")
        self._outputs = outputs

    def visualize(self, outfile: str,
                  map2time: Optional[Callable] = None,
                  map2mem: Optional[Callable] = None,
                  map2name: Optional[Callable] = None):
        """
        Visualize the graph

        @param map2time Optional[Callable]: node to time (int) map.
        @param map2mem Optional[Callable]: node to memory consumption (int) map
        @param map2name Optional[Callable]: node to name (str) map
        @param outfile Optional[str]: the output file name.
            If given, will save the visualized execution plan in file.
        """
        ndevice = len(self.devices())
        # timeline [ [ (start_time, end_time), ... ], ... ]
        device_timeline = [list() for _ in range(ndevice)]
        device_nodes = [list() for _ in range(ndevice)]
        device_mem = [0] * ndevice
        device_peak_mem = [0] * ndevice

        if map2time is None:
            def map2time(node):
                if isinstance(node, IRDataOperation): return 0
                if isinstance(node, IRAdapter): return 0.25
                return 1 if node.isfw() else 2
        
        if map2mem is None:
            def map2mem(node):
                if isinstance(node, IRSegment):
                    peak_mem = 0
                    curr_mem = 0
                    for node in node.nodes():
                        curr_mem += map2mem(node)
                    peak_mem = max(curr_mem, peak_mem)
                if isinstance(node, IRFwOperation):
                    return 1
                if isinstance(node, IRBpOperation):
                    return -1
                return 0

        if map2name is None:
            def map2name(node):
                if isinstance(node, IRAdapter):
                    return ''
                else:
                    return f'f{node.cid}' if node.isfw() else f'b{node.cid}'

        def map2color(node):
            node = node.cell if isinstance(node, ExeReuseCell) else node
            if isinstance(node, IRAdapter):
                return '#70AD47'  # excel green
            if node.isfw():
                return '#4472C4'  # excel blue
            else:
                return '#ED7D31'  # excel orange


        # analyze device timeline

        def depends(prev: IRCell, next: IRCell) -> bool:
            for to in prev.outputs():
                if not isinstance(to, IRSubTensor): continue
                for ti in next.inputs():
                    if not isinstance(ti, IRSubTensor): continue
                    if to.overlap(ti): return True
            return False

        device_timeline: List[Tuple[int, int]] = [list() for _ in range(ndevice)]
        device_nodes: List[IRCell] = [list() for _ in range(ndevice)]
        device_mem = [0] * ndevice
        device_peak_mem = [0] * ndevice

        for node in self._topo_seqs:
            unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
            span, mem = map2time(unwrap_node), map2mem(unwrap_node)
            # calculate time
            start_times = []
            for device in node.device:
                # tight execution if no dependency
                if len(device_timeline[device]) == 0:
                    start_time = 1
                else:
                    start_time = device_timeline[device][-1][1]
                # check dependency
                for devid, timeline in enumerate(device_timeline):
                    dev_seq = device_nodes[devid]
                    if devid == device: continue
                    for nid, (_, end_time) in enumerate(timeline[::-1]):
                        other_node = dev_seq[::-1][nid]
                        if depends(other_node, node):
                            start_time = max(start_time, end_time)
                            break
                start_times.append(start_time)
            
            start_time = max(start_times)
            for device in node.device:
                # time
                device_timeline[device].append((start_time, start_time + span))
                device_nodes[device].append(node)
                # memory
                device_mem[device] += mem
                if device_peak_mem[device] < device_mem[device]:
                    device_peak_mem[device] = device_mem[device]

        max_time = max(
            [tline[-1][1] for tline in device_timeline if len(tline) != 0]
        )
        max_mem = max(device_peak_mem)
        # max_mem = sum(device_peak_mem)

        # draw the timeline

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import AutoMinorLocator
        plt.close('all')
        plt.rcParams['figure.figsize'] = (4.0 * max_time // ndevice, 4.0)
        fig, ax = plt.subplots()
        renderer = fig.canvas.get_renderer()

        # xaxis
        ax.set_xlim((1, max_time))
        plt.xticks(
            ticks=np.arange(1.5, max_time+0.5, 1.0, dtype=float),
            labels=np.arange(1, max_time, 1, dtype=int)
        )
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        ax.xaxis.grid(which='minor', linestyle='--')
        # yaxis
        ax.set_ylim((0.5, len(self.devices())+0.5))
        plt.yticks(list(range(1, len(self.devices())+1, 1)))
        ax.invert_yaxis()

        fontsize = [40]
        txts = list()
        for devid in range(ndevice):
            timeline = device_timeline[devid]
            nodes = device_nodes[devid]
            for node, (start, end) in zip(nodes, timeline):
                unwrap_node = node.cell if isinstance(node, ExeReuseCell) else node
                if end - start == 0:
                    continue
                # draw 
                color = map2color(unwrap_node)
                rec = Rectangle((start, devid + 0.5), end-start, 1,
                                color=color, ec='black', lw=1.5)
                ax.add_artist(rec)
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                anno = map2name(unwrap_node)
                if anno == '': continue
                txt = ax.text(x=cx, y=cy, s=anno, fontsize=40, ha='center', va='center', color='w')
                rbox = rec.get_window_extent(renderer)
                for fs in range(fontsize[0], 1, -2):
                    txt.set_fontsize(fs)
                    tbox = txt.get_window_extent(renderer)
                    if tbox.x0 > rbox.x0 and tbox.x1 < rbox.x1 and tbox.y0 > rbox.y0 and tbox.y1 < rbox.y1:
                        break
                fontsize[0] = min(fontsize[0], fs)
                txts.append(txt)
            
        # set font size to same
        for txt in txts:
            txt.set_fontsize(fontsize[0])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize[0])
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize[0])
        plt.xlabel('Time Step', fontsize=fontsize[0])
        plt.ylabel('Device ID', fontsize=fontsize[0])
        plt.tight_layout()
        plt.savefig(outfile)

        return max_time, max_mem

    def __repr__(self):
        dscp = f'Execution Plan ({self.graph.name}) (inference: {self.inference}):\n'
        for devid in self.devices():
            dscp += f'====> Device {devid}:\n'
            for node in self._seq[devid]:
                dscp += f'{node}\n'
        return dscp
