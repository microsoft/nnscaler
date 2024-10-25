#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional, Tuple, Dict, Any
import more_itertools
import logging
import copy
import torch
import numpy as np
import inspect

from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation
from nnscaler.ir.adapter import IRWeightReducer, IRAdapter
from nnscaler.ir.adapter.prim import CollectivePrim

from nnscaler.graph.graph import IRSegment
from nnscaler.graph.parser.register import CustomizedOps

from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.execplan import ExeReuseCell

from nnscaler.codegen.syntax.symtable import SymbolTable
from nnscaler.codegen.syntax.blocks import ClassBlock, FunctionBlock

from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.module.autograd import AutogradAdapterCodeGen
from nnscaler.codegen.lifecycle import LifeCycle

from nnscaler.flags import CompileFlag
from nnscaler import __version__ as runtime_version


_logger = logging.getLogger(__name__)


class ModuleCodeGen(FuncEmission):
    """
    Generate module code

    `ModuleCodeGen` traverses all IR nodes and categorizes their intermediately generated
    codes into different parts,
    then reorders and concatenates these parts into the final code for PyTorch to run.

    These parts are progressively stored into fields of `ModelCodeGen`

    - `init_code : List[str]`
        Statements like `import torch`

    - `model_init_statements : List[str]`
        Statements of the `__init__` constructor of the final `nn.Module` in codegen,

        E.g. (lines are split into `List[str]`)
        ```python
        self.init_group(ranks=[0, 1, 2, 3])
        self.weight_63 = torch.nn.Parameter(torch.empty((2048, 8192), dtype=torch.float32))
        self.add_full_map('weight_63', 3, (slice(0, 2048, None), slice(0, 8192, None)), 1)
        ```

        including:
        -- initialization of model weights, which are class fields;

    - `model_methods_bodies : List[List[str]]`
        Definitions of the Python code for forward computations like Segments or Adapters

        Note that codes within this field haven't been organized into valid Python methods,
        namely without signatures and return statements, both of which will be extracted
        from corresponding IRSegment/IRAdapter in later processes.
        E.g.
        ```
        [
            # intermediate codes for 'segment123(self, tensor_2222)'
            [
                'tensor_3333 = torch.view(tensor_2222, [1,2,3,4])'
            ]

            # intermediate codes for 'adapter456(self, tensor_4444)'
            [
                'tensor_5555 = nnscaler.runtime.adapter.all_reduce(tensor_4444, ranks=[0,1,2,3])'
            ]
        ]
        ```
    """

    def __init__(
        self,
        execplan: ExecutionPlan,
        runtime_ndevs: Optional[int] = None,
        *,
        scale_ndevs: Optional[int] = None
    ) -> None:
        """
        Create Module code generator

        Args:
            execplan (ExecutionPlan): execution plan
            runtime_ndevs (Optional[int]): the number of devices in runtime
            scale_ndevs (Optional[int]): Deprecated. Use `runtime_ndevs` instead
        """
        super().__init__()
        self.execplan: ExecutionPlan = execplan
        self.devices: Tuple[int] = tuple(sorted(execplan.graph.device))
        if self.devices != tuple(range(len(self.devices))):
            raise ValueError(f'device must be consecutive')

        if scale_ndevs is not None:
            _logger.warning("scale_ndevs is deprecated, please use runtime_ndevs instead")
            if runtime_ndevs is not None:
                raise ValueError("You cannot use runtime_ndevs and scale_ndevs at the same time")
        self.runtime_ndevs: int = runtime_ndevs or scale_ndevs or len(self.devices)
        # we will scale the graph as data parallelism
        # when we have more devices than the number of devices used in the graph
        # we need to do two things:
        # 1. update execplan with dp reducers (via add_scale_reducers)
        # 2. update node devices when emitting code (via scale)
        if self.runtime_ndevs % len(self.devices) != 0:
            raise ValueError(f'runtime_ndevs must be a multiple of {len(self.devices)}')
        self.enable_dp = self.runtime_ndevs > len(self.devices)

        self.init_code: List[str] = [
            '########## Generated Model Code ###########',
            'from typing import *',
            'from pathlib import Path',
            'import torch', 'import torch.utils.checkpoint as ckpt',
            'import nnscaler', 'import _operator', 'from numpy import inf', 'import builtins', '',
            f'runtime_version = {runtime_version!r}', '', ''
        ]

        if CompileFlag.use_nnfusion:
            self.init_code.extend(['import nnfusion', ''])

        # customized op code
        for op_impl in set(CustomizedOps.kOpCodeDef.values()):
            # self.init_code.append('@torch.jit.script')
            self.init_code.append(op_impl)
            self.init_code += ['']
        # module init code
        self.model_init_statements: List[str] = list()
        # module method bodies for forward computations, e.g. Segments, Adapters.
        self.model_methods_bodies: List[List[str]] = list()
        # module member name
        self.symbols = SymbolTable()
        # ref module to check shared variables
        self._ref_module = torch.nn.Module()
        # batch size
        self.batch_size = None
        # communication groups
        self.comm_groups: List[Tuple[int]] = self.get_comm_groups()
        self.add_scale_reducers()

    def add_scale_reducers(self):
        """
        Insert reducers to for scale scenario
        """
        if not self.enable_dp:
            return
        graph = self.execplan.graph
        # for each device, collect parameters in the all reducers and create a reducer for the rest
        for device in self.devices:
            # collect parameters in the all reducers belonging to this device
            all_params = set()
            for reducer in graph.select(ntype=IRWeightReducer):
                if device not in reducer.device: continue
                for param in reducer.inputs():
                    assert param not in all_params, \
                        f'detected a parameter {param} in multiple reducers on device {device}'
                all_params.update(reducer.inputs())
            # create reducers for the rest parameters used for this device
            # nnscaler's weights are either fully replicated or partitioned, which has been checked
            # at graph/gener/gen.py/gen_weights.
            # We decouple the replicated and partitioned weights to align with the calculation of
            # gradient norm which uses the replicated number of each weight to make the global value
            # correct.
            rest_params_replicated = []
            rest_params_partitioned = []

            def collect_rest_params(segment):
                """Resursively collect parameters. Note parameters can be in sub-segments,
                which is invisible to its top-level segment."""
                for param in segment.attributes():
                    if not param.is_param(): continue
                    for ctensor in segment.ctensors(param):
                        if device not in ctensor.device: continue
                        if ctensor not in all_params:
                            # a same parameter can be consumed multiple times by different operators
                            if ctensor.shape == ctensor.parent.shape:
                                if ctensor not in rest_params_replicated:
                                    rest_params_replicated.append(ctensor)
                            else:
                                if ctensor not in rest_params_partitioned:
                                    rest_params_partitioned.append(ctensor)
                for seg in segment.select(ntype=IRSegment, flatten=False):
                    collect_rest_params(seg)

            collect_rest_params(graph)
            # create reducer and append to the execution
            # device will be scaled in `self.scale`
            for reducer in IRWeightReducer.from_weights(rest_params_replicated, device):
                self.execplan.at(device).append(reducer)
            for reducer in IRWeightReducer.from_weights(rest_params_partitioned, device):
                self.execplan.at(device).append(reducer)

    def get_comm_groups(self):
        """
        Scale the communication groups to multiple devices
        using data parallelism.

        @warn this requires user side to setup dataloader
            for different GPUs
        """
        def _add_comm_for_group_zero(ranks):
            zero_comm_groups = []
            # Create communication group for each zero subgroup
            for i in range(CompileFlag.zero_ngroups):
                assert len(ranks) % CompileFlag.zero_ngroups == 0
                ranks_per_group = len(ranks) // CompileFlag.zero_ngroups
                zero_subgroup = tuple(ranks[i * ranks_per_group : (i + 1) * ranks_per_group])
                if len(zero_subgroup) > 1 and len(zero_subgroup) < len(ranks):
                    zero_comm_groups.append(zero_subgroup)
            # Create communication groups for cross group allreduce.
            # Note that this is only for the enabled reduce scatter of ZeRO.
            # For example, there are two ZeRO groups [0,1,2,3] and [4,5,6,7],
            # then we will create communication groups (0,4), (1,5), (2,6), (3,7).
            ranks_per_group = len(ranks) // CompileFlag.zero_ngroups
            for i in range(ranks_per_group):
                zero_crossgroup = tuple(ranks[i::ranks_per_group])
                if len(zero_crossgroup) > 1 and len(zero_crossgroup) < len(ranks):
                    zero_comm_groups.append(zero_crossgroup)
            return zero_comm_groups

        nreplica = self.runtime_ndevs // len(self.devices)
        # scale communication groups
        graph = self.execplan.graph
        comm_groups = []
        # communication groups for parameters that are in reducers
        reducers: List[IRWeightReducer] = graph.select(ntype=IRWeightReducer)
        for reducer in reducers:
            ranks = more_itertools.flatten(list(range(device, self.runtime_ndevs, len(self.devices))) \
                                           for device in reducer.device)
            ranks = tuple(sorted(ranks))
            comm_groups.append(ranks)
            # add comm groups for group ZeRO
            comm_groups.extend(_add_comm_for_group_zero(ranks))
        # communication groups for parameters that are outside reducers
        for device in self.devices:
            ranks = list(range(device, self.runtime_ndevs, len(self.devices)))
            if len(ranks) > 1:
                comm_groups.append(ranks)
                # add comm groups for group ZeRO
                comm_groups.extend(_add_comm_for_group_zero(ranks))
        # communication groups for activations
        adapters = graph.select(ntype=IRAdapter)
        for adapter in adapters:
            for prim in adapter.prims:
                if isinstance(prim, CollectivePrim):
                    ranks = np.array(tuple(sorted(prim.kwargs['ranks'])), dtype=int)
                    for i in range(nreplica):
                        shifted_ranks = tuple(ranks + i * len(self.devices))
                        shifted_ranks = tuple(int(rank) for rank in shifted_ranks)
                        if shifted_ranks not in comm_groups:
                            comm_groups.append(shifted_ranks)
        return comm_groups

    def scale(self, node: IRCell, device: int) -> IRCell:
        if not self.enable_dp:
            return node
        shift = (device // len(self.devices)) * len(self.devices)
        if isinstance(node, IRAdapter):
            adapter = copy.copy(node)
            adapter._id = node.cid
            adapter.kwargs.update(node.kwargs)
            prims = []
            for prim in adapter.prims:
                p = copy.copy(prim)
                p.kwargs = dict(**prim.kwargs)
                if 'ranks' in prim.kwargs:
                    p.kwargs['ranks'] = [rank + shift for rank in prim.kwargs['ranks']]
                if 'src' in prim.kwargs:
                    p.kwargs['src'] = prim.kwargs['src'] + shift
                if 'srcs' in prim.kwargs:
                    p.kwargs['srcs'] = [src + shift for src in prim.kwargs['srcs']]
                if 'dst' in prim.kwargs:
                    p.kwargs['dst'] = prim.kwargs['dst'] + shift
                if 'dsts' in prim.kwargs:
                    p.kwargs['dsts'] = [dst + shift for dst in prim.kwargs['dsts']]
                prims.append(p)
            adapter.prims = prims
            if node.isfw() and node.differentiable and node.custom:
                badapter = self.scale(node.mirror, device)
                IRCell.make_pair(adapter, badapter)
            return adapter
        if isinstance(node, IRWeightReducer):
            reducer = IRWeightReducer(node.inputs(), name=node.name)
            reducer._id = node.cid
            ranks = list(node.device)
            scale_ranks = []
            for rank in ranks:
                scale_ranks += list(range(rank, self.runtime_ndevs, len(self.devices)))
            reducer.device = sorted(scale_ranks)
            return reducer
        if isinstance(node, IRSegment) and node.isfw():
            nodes = [self.scale(n, device) for n in node.nodes()]
            segment = IRSegment(nodes, node.inputs(), node.outputs(), node.name)
            segment._id = node.cid
            return segment
        return node

    def gen(
        self,
        device: int,
        outfile: str = None,
        attach: bool = False,
        *,
        as_parallel_module: bool = False,
        end2end_mode: bool = False,
        forward_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate model implementation code based on the given graph.
        if as_parallel_module is True, we will create a forward method for the module.
        The arguments of the forward method will be same with original forward method with some exceptions:
        1. No positional only argument/keyword only argument support.
            For example
            ```python
            def forward(self, x, y, /, z=None, *, m=1, n=2):
                ...
            ```
            the bevaior of the forward method will be undefined, and should be avoided.
            Also the generated forward method will not have positional only argument/keyword only argument.
        2. *args is not supported, and will trigger runtime error.
           For example:
            ```python
            def forward(self, x, y, *args):
                ...
            ```
            will fail to generate forward method.
        3. **kwargs will be kept as it is.
            For example:
            ```python
            def forward(self, x, y, **kwargs):
                ...
            ```
            the generated forward method will be:
            ```python
            def forward(self, x, y, **kwargs):
                ...
            ```
            But you should not specify any argument in **kwargs when tracing the forward method.
            The behavior will be undefined if you rely on the argument in **kwargs.
        4. If an argument is found not in the traced graph,
            it will have default value None(no matter what the default value is in the original forward method),
            And when calling the generated forward method, you should not specify the argument.
            Otherwise, ValueError will be raised.
            For example:
            ```python
            def forward(self, x, y, z=1):
                ...
            ```
            If y and z are not in the traced graph, the generated forward method will be:
            ```python
            def forward(self, x, y=None, z=None):
                if y is not None: raise ValueError
                if z is not None: raise ValueError
                ...
            ```
        5. If an argument is used in the traced graph, the default value will be kept as it is.
            For example:
            ```python
            def forward(self, x, y, z=1):
                ...
            ```
            if z is used in the traced graph, the generated forward method will be:
            ```python
            def forward(self, x, y, z=1):
                ...
            ```
        6. A special case is, if an argument is after an unused argument, but doesn't have a default value.
           To make python happy, we have to give it a default value None.
            For example:
            ```python
            def forward(self, x, y, z):
                ...
            ```
            if y is not used in the traced graph, the generated forward method will be:
            ```python
            def forward(self, x, y=None, z=None):
                if y is not None: raise ValueError
                ...
            ```
            Please note z has to have default value None, otherwise, python will complain.
        Args:
            device (int): device id
            outfile (str): output file path
            attach (bool): whether to append to the file
            as_parallel_module (bool): whether to generate parallel module, which will
                1. Inherit from ParallelModule
                2. Has forward method
                3. Add more content to constructor
            end2end_mode (bool): whether to generate code for end2end mode.
                If True, a mocked `forward` will be generated which only raises NotImplementedError.
                If False, the real forward function will be generated.
                This is used only in parallel module.
            forward_args (Dict[str, Any]): argument names and their default values of forward function, if None, use node inputs.
                This is used only in parallel module.

        Returns:
            generated code
        """
        gencode = copy.copy(self.init_code)
        node_args: List[List[str]] = list()
        gen_nodes: List[IRCell] = list()

        device_map = device % len(self.devices)
        sequence = self.execplan.seq(device_map)
        unrolled_seqs = []
        for node in sequence:
            # unwrap from ExeReuseCell
            node = node.cell if isinstance(node, ExeReuseCell) else node
            unrolled_seqs.append(node)
        # we use ordered dict as ordered set
        sequence = tuple(dict.fromkeys(unrolled_seqs))

        # scale to multiple devices
        sequence = [self.scale(node, device) for node in sequence]

        # init customized adapter
        fsegments = [node for node in sequence if isinstance(node, IRSegment) and node.isfw()]
        autograd_adapter_gen = AutogradAdapterCodeGen()
        for seg in fsegments:
            for adapter in seg.select(ntype=IRAdapter):
                if adapter.differentiable and adapter.custom:
                    gencode += autograd_adapter_gen.gen(adapter) + ['', '']
                    adapter.signature = autograd_adapter_gen.name(adapter) + '.apply'

        # initialize communication groups
        self.emit_comm_groups()

        # we can have multiple segments in the graph when pipeline is enabled.
        # Here we don't use tid to sort parameters
        # because that assumption may be not true in the future,
        # and the current implementation is clearer and more robust.
        # key: parameter tensor, value: (segment index, node index)
        param_first_used_pos: Dict[IRFullTensor, Tuple[int, int]] = {}
        for i, n in enumerate(sequence):
            if isinstance(n, IRSegment) and n.isfw():
                for k, v in self._get_param_first_used_pos(n).items():
                    if k not in param_first_used_pos:
                        param_first_used_pos[k] = (i, v)

        # emit code
        for node in sequence:
            if isinstance(node, IRSegment):
                if not node.isfw(): continue  # skip backward segment
                codes = self.emit_segment(node, device)
            elif isinstance(node, IRFwOperation):
                raise RuntimeError(f"Unexcepted global-level op call: {node}")
            elif isinstance(node, IRAdapter):
                codes = self.emit_adapter(node, prefix_attr='self.', async_op=CompileFlag.async_comm)
            elif isinstance(node, IRWeightReducer):
                self.init_reducer(node, device, param_first_used_pos, as_parallel_module)
                codes = self.emit_reducer(node)
            elif isinstance(node, IRBpOperation):
                continue
            elif isinstance(node, IRDataOperation):
                continue
            else:
                raise RuntimeError(f"Un-recognized IRCell type: {type(node)}")

            # emit node tensor declaration into `__init__`
            # typically it's about the `nn.Parameter`
            self.init_attributes(node)

            # emit node code
            # codes : List[str]
            self.model_methods_bodies.append(codes)
            gen_nodes.append(node)

            args = list()
            for t in node.inputs():
                if isinstance(t, IRSubTensor):
                    if not t.is_attr():
                        args.append(self.tensor_name(t))
                else:
                    args.append(self.tensor_name(t))
            node_args.append(args)

        # generate full code
        with ClassBlock(
            class_name='GenModel',
            derived=[f'nnscaler.runtime.module.{"ParallelModule" if as_parallel_module else "CubeModule"}']
        ) as cb:
            graph_sched = self.execplan.graph.sched
            cb.insert_body(f'use_scheduler = {graph_sched is not None}')
            cb.insert_body(f'nmicros_per_scheduler_step = {graph_sched.nmicros if graph_sched is not None else 1}')

            if as_parallel_module:
                cb.insert_body(f'rank = {device}')  # save rank in class level
                # async_op, max_bucket_size_bytes and zero_use_reduce_scatter
                # parameters are for testing purpose
                # and will not expose to user
                with FunctionBlock(func_name='__init__',
                    args=[
                        'self',
                        'init_params=True',
                        '*',
                        f'async_op={CompileFlag.async_reducer}',
                        f'max_bucket_size_bytes={CompileFlag.max_reducer_bucket}',
                        f'zero_use_reduce_scatter={CompileFlag.zero_use_reduce_scatter}',
                    ]
                ) as ib:
                    ib.insert_body(self.model_init_statements)
                    ib.insert_body('')
                    ib.insert_body('self._post_init(init_params)')
            else:
                with FunctionBlock(func_name='__init__', args=['self']) as ib:
                    ib.insert_body(self.model_init_statements)
            cb.insert_body('')
            cb.insert_body(ib.code)
            segment_idxs =[]
            for idx, node in enumerate(gen_nodes):
                name = self.node_name(node)
                input_args = ['self'] + node_args[idx]
                forward_code = self.model_methods_bodies[idx]
                if isinstance(node, IRSegment):
                    segment_idxs.append(idx)

                with FunctionBlock(func_name=name, args=input_args) as fb:
                    fb.insert_body(forward_code)
                    # generate output
                    outputs = [self.tensor_name(t) for t in node.outputs()]
                    return_code = f"return {', '.join(outputs)}"
                    fb.insert_body(return_code)
                cb.insert_body('')
                if CompileFlag.use_nnfusion and name.startswith('segment'):
                    cb.insert_body('@nnfusion.jit')
                if CompileFlag.use_jit and name.startswith('segment'):
                    cb.insert_body('@torch.jit.script_method')
                cb.insert_body(fb.code)

            if as_parallel_module:
                if not segment_idxs:
                    raise RuntimeError("The graph has no segment, forward code cannot be generated.")
                cb.insert_body('')
                if not end2end_mode:
                    if len(segment_idxs) > 1:
                        raise RuntimeError("The graph has more than one segment, forward code cannot be generated.")
                    segment_idx = segment_idxs[0]
                    node = gen_nodes[segment_idx]
                    cb.insert_body(self._generate_forward(node, forward_args))
                else:
                    msg = "Code of forward is not generated. You should use module.train_step/module.infer_step instead."
                    with FunctionBlock(func_name='_forward_impl', args=['self', '*args', '**kwargs']) as fb:
                        fb.insert_body(f'raise NotImplementedError("{msg}")')
                    cb.insert_body(fb.code)

        gencode += cb.code
        gencode += ['']

        code = '\n'.join(gencode)
        # write to file
        if outfile:
            with open(outfile, 'a' if attach else 'w') as f:
                f.write(code)

        # clear used buffer
        self.clear()
        return code

    def _generate_forward(self, node, forward_args):
        # the orignal names of inputs
        inputs = [t.name for t in node.inputs() if not isinstance(t, IRSubTensor) or not t.is_attr()]

        unused_args = []
        forward_arg_resolved = []
        if forward_args:
            # check all inputs are in forward args
            for i in range(len(inputs)):
                if inputs[i] not in forward_args:
                    raise ValueError(f"Forward args mismatch: {inputs[i]} arg needed")

            forward_arg_names = list(forward_args.keys())
            def _get_resolved_arg(arg_name, default_value):
                if default_value is inspect.Parameter.empty:
                    return arg_name
                else:
                    return f'{arg_name}={repr(default_value)}'

            # find the first mismatch
            # here, we will keep the default values of the forward args
            for i in range(len(inputs)):
                if inputs[i] == forward_arg_names[i]:
                    default_value = forward_args[inputs[i]]
                    forward_arg_resolved.append(_get_resolved_arg(inputs[i], default_value))
                else:
                    break

            # check the rest of the forward args
            # if the arg is not in inputs, we will set the default value to None
            #     in runtime, we will make sure the user doesn't specify it.
            # if the arg is in inputs, we will keep the default value
            # Also *args and **kwargs are kept as it is.
            for i in range(len(forward_arg_resolved), len(forward_arg_names)):
                if not forward_arg_names[i].startswith('*'):
                    default_value = forward_args[forward_arg_names[i]]
                    if forward_arg_names[i] not in inputs:
                        unused_args.append(forward_arg_names[i])
                        forward_arg_resolved.append(f'{forward_arg_names[i]}=None')
                        _logger.warning(f'Unused forward argument `{forward_arg_names[i]}`.'
                                        f'The argument value will be ignored when you call module forward')
                    else:
                        forward_arg_resolved.append(
                            _get_resolved_arg(
                                forward_arg_names[i],
                                None if default_value is inspect.Parameter.empty else default_value
                            )
                        )
                else:
                    forward_arg_resolved.append(forward_arg_names[i])
        else:
            forward_arg_resolved = inputs

        with FunctionBlock(func_name='_forward_impl', args=['self'] + forward_arg_resolved) as fb:
            outputs = self.return_name(node.outputs(), skip_attr=True)
            call_code = f'{outputs} = self.{self.node_name(node)}({", ".join(inputs)})'
            # be sure the user doesn't specify unused args.
            for unused_arg in unused_args:
                fb.insert_body(f'if {unused_arg} is not None: raise ValueError("{unused_arg} is not used in graph tracing, so it must be None when running forward.")')
            fb.insert_body(call_code)
            return_code = f'return {self.return_name_complex(self.execplan.graph.outputs())}'
            fb.insert_body(return_code)

        return fb.code

    def emit_comm_groups(self):
        """
        Creating communication group requires all the devices
        enter the same call.

        The fields storing intermediate codes that are populated by this method:
        - `model_init_statements`
        """
        sign = 'self.init_group(ranks={ranks})'
        # create communication group
        self.model_init_statements.append('# communication groups')
        for ranks in self.comm_groups:
            code = sign.format(ranks=list(ranks))
            self.model_init_statements.append(code)
        self.model_init_statements.append(' ')

    def init_attributes(self, node: IRCell):
        """
        Emit tensor declaration code

        The fields storing intermediate codes that are populated by this method:
        - `model_init_statements`

        This method also populates `self.symbols : SymbolTable` to record
        the names of the variables for the tensors ever encountered.
        """
        psign = "self.register_parameter('{name}', torch.nn.Parameter(torch.empty({shape}, dtype={dtype})))"
        bsign = "self.register_buffer('{name}', torch.empty({shape}, dtype={dtype}), persistent={persistent})"
        map_sign = "self.add_full_map('{attr}', {tid}, {is_param}, '{orig_name}', {full_shape}, {slicers}, {val_chunks})"
        if not isinstance(node, IRSegment):
            for itensor in node.inputs():
                name = self.tensor_name(itensor, prefix_attr='self.')
                if isinstance(itensor, IRSubTensor):
                    if itensor.is_attr() and not self.symbols.exist(name):
                        self.symbols.create(name)
                        if itensor.is_param():
                            code = psign.format(
                                name=self.tensor_name(itensor),
                                shape=tuple(itensor.origin_shape),
                                dtype=itensor.dtype
                            )
                        elif itensor.is_buffer():
                            code = bsign.format(
                                name=self.tensor_name(itensor),
                                shape=tuple(itensor.origin_shape),
                                dtype=itensor.dtype,
                                persistent=itensor.is_persistent()
                            )
                        else:
                            raise RuntimeError(f"Unexpected tensor type: {itensor}")
                        self.model_init_statements.append(code)
                        slicers = tuple(slice(start, stop) for (start, stop) in itensor.indmap)
                        if itensor.is_scalar_tensor():
                            assert len(slicers) == 1 and slicers[0] == slice(0, 1), f"Unexpected slicers {slicers} for scalar tensor."
                            slicers = '...'  # Ellipsis slicer for scalar tensor, x[...] is equivalent to x
                        val_chunks = itensor.valmap[1]
                        code = map_sign.format(
                            attr=self.tensor_name(itensor),
                            tid=itensor.parent.tid,
                            is_param=itensor.is_param(),
                            orig_name=itensor.parent.name,
                            full_shape=tuple(itensor.parent.origin_shape),
                            slicers=str(slicers),
                            val_chunks=val_chunks
                        )
                        self.model_init_statements.append(code)
                        self.model_init_statements.append('')
                if isinstance(itensor, str):
                    if name.startswith('self.'):
                        if not hasattr(self._ref_module, name[5:]):
                            raise NotImplementedError("member attribute is not added")
            for output in node.outputs():
                self.symbols.create(self.tensor_name(output, prefix_attr='self.'))
        else:
            for sub_node in node.nodes():
                self.init_attributes(sub_node)
        return

    def init_reducer(self,
        node: IRWeightReducer,
        device: int,
        param_first_used_pos: Dict[IRFullTensor, int],
        as_parallel_module: bool = True,
    ) -> None:
        """
        Emit code to initialize involved reducer objects in `__init__`.

        The fields storing intermediate codes that are populated by this method:
        -   `model_init_statements`
        """
        # when parallel module is used,
        # `max_bucket_size_bytes` and `async_op` are passed as arguments
        max_nbytes = CompileFlag.max_reducer_bucket if not as_parallel_module else 'max_bucket_size_bytes'
        async_op = CompileFlag.async_reducer if not as_parallel_module else 'async_op'
        zero_use_reduce_scatter = CompileFlag.zero_use_reduce_scatter if not as_parallel_module else 'zero_use_reduce_scatter'

        zero = CompileFlag.use_zero
        zero_ngroups = CompileFlag.zero_ngroups
        reduce_op = f"'{CompileFlag.reducer_op}'"
        # reducer init interface
        reducer_init = (
            "{reducer} = nnscaler.runtime.adapter.Reducer("
            "ranks={ranks}, reduce_op={reduce_op}, "
            "async_op={async_op}, zero={zero}, max_bucket_size_bytes={max_nbytes}, "
            "zero_use_reduce_scatter={zero_use_reduce_scatter}, "
            "zero_ngroups={zero_ngroups})"
        )
        reducer_add = 'self.add_reducer({reducer})'
        add_param = '{reducer}.add_param({weight})'
        # create reducer in declare region
        weights = node.inputs()
        reducer_name = f'self.wreducer{node._id}'
        self.model_init_statements.append('')
        ranks = list(sorted(node.device))
        init_code = reducer_init.format(
            reducer=reducer_name, ranks=ranks, reduce_op=reduce_op,
            async_op=async_op, zero=zero, max_nbytes=max_nbytes,
            zero_ngroups=zero_ngroups, zero_use_reduce_scatter=zero_use_reduce_scatter
        )
        self.model_init_statements.append(init_code)
        # sort weights by first used time (which is gradient all-reduce time in reverse order)
        # so that weights with similar gradient all-reduce time are bucketed together
        weights = [
            self.tensor_name(t, prefix_attr='self.')
            for t in sorted(weights, key=lambda t: param_first_used_pos[t.parent])
        ]
        for weight in weights:
            add_param_code = add_param.format(reducer=reducer_name, weight=weight)
            self.model_init_statements.append(add_param_code)
        add_code = reducer_add.format(reducer=reducer_name)
        self.model_init_statements.append(add_code)

    def emit_segment(self, segment: IRSegment, runtime_devid: int) -> List[str]:
        """
        Emit IRSegment code.

        The returned `List[str]` will be lines of the statements of the final
        Python method for the targeted Segment.
        The returned lines will not include the signature and the return statement
        of the generated Python method. These lines will be put into `model_methods_bodies`
        and the missing Python-syntactic parts will be injected later on.

        e.g.
        ```
        [
            # no method signature
            'tensor_3333 = torch.view(tensor_2222, [1,2,3,4])',
            'tensor_2222 = None',   # if in dataflow there is no more reference
            'tensor_4444 = torch.sum(tensor_3333)',
            'def recompute(...):',
            '    return ...',
            'tensor_5555 = torch.utils.checkpoint(recompute, tensor_4444)',
            'tensor_4444 = None',   # if in dataflow there is no more reference
            # no return statement
        ]
        ```

        Nodes in the segment will group into recompute region

        The fields storing intermediate codes that are populated by this method:
        - NONE
        """
        nodes : List[IRCell] = segment.nodes()
        lifetime = LifeCycle(nodes, segment.inputs(), segment.outputs())
        rc_groups: List[List[IRCell]] = list(
            more_itertools.split_when(nodes, lambda prev, curr: prev.recompute != curr.recompute))

        codes: List[str] = []
        for rc_group in rc_groups:
            assert len(rc_group) > 0
            gid: Optional[int] = rc_group[0].recompute
            if gid is None:
                codes += self._emit_nodes(rc_group, lifetime, runtime_devid)
            else:
                # get recompute excution code
                rc_segment = segment.create_segment(rc_group)
                rc_codes = self._emit_recompute(rc_group,
                    rc_segment.inputs(), rc_segment.outputs(), lifetime, runtime_devid)
                codes += rc_codes
                # release input tensors after exiting a RC group:
                last_node = rc_group[-1]
                line = lifetime.get_line(last_node)
                if last_node != nodes[-1]: # skip if it is the last node
                    inputs_to_rel = [t for t in rc_segment.inputs() if lifetime.releasable_after_line(t, line)]
                    if len(inputs_to_rel) > 0:
                        del_stmt = self.emit_release(inputs_to_rel)
                        codes.append(del_stmt)

        return codes

    def _emit_nodes(self, nodes: List[IRCell], lifecycle: LifeCycle, runtime_devid: int) -> List[str]:
        """
        Emit code to invoke operations and adapter,
        e.g. (the lines are split into `List[str]`)

        ```
        tensor_2222 = torch.view(tensor_1111, size=[3,6,9])
        del tensor_1111    # if no more reference
        tensor_3333 = nnscaler.runtime.adapter.allgather_reducescatter(tensor_2222, dim=1, rank=[0,1])
        del tensor_2222    # if no more reference
        ```

        The fields storing intermediate codes that are populated by this method:
        - NONE
        """
        node_codes = []
        for node in nodes:
            # execute
            if isinstance(node, IRFwOperation):
                code = self.emit_fnode(node, runtime_devid=runtime_devid, plan_ndevs=len(self.devices), runtime_ndevs=self.runtime_ndevs, prefix_attr='self.')
                node_codes += code
            elif isinstance(node, IRAdapter):
                # for adapters inside an IRSegment, we don't apply async communication to it
                # as it is mostly in critical path.
                code = self.emit_adapter(node, async_op=False)
                node_codes += code
            else:
                raise RuntimeError(f"unexpected type {type(node)} in IRSegment")
            # release
            tensors_to_del = lifecycle.release_tensors_after_node(node)
            if len(tensors_to_del) > 0:
                node_codes.append(self.emit_release(tensors_to_del))

        return node_codes

    def _emit_recompute(self, nodes: Tuple[IRCell], inputs: List[IRSubTensor], outputs: List[IRSubTensor],
                        lifecycle: LifeCycle, runtime_devid: int) -> List[str]:
        """
        Emit code to define a Python function for Recomputing and invoke it
        e.g. (the lines are split into `List[str]`)

        ```
        def recompute(tensor_2222):
            tensor_3333 = torch.view(tensor_2222, size=[3,6,9])
            tensor_2222 = None      # no more reference
            return tensor_3333
        # in the beginning we have `import torch.utils.checkpoint as ckpt`
        tensor_4444 = ckpt.checkpoint(recompute, tensor_1111)
        ```

        REMARK:
        -   In the example above, 'tensor_2222' can be released within the RC subgraph, which also means that
            the variable for this tensor can also be released within the enclosing graph, after the 'checkpoint' call.
        -   The generated RC subgraph will have no "free variables".
            All involved tensors that are defined outside of the RC group are made explicit inputs;
            All tensors, that are defined within the RC group and are referenced after RC subgraph ends, are made explicit outputs;
            And if a within-RC-group tensors are not used anymore, it's not returned.

        The fields storing intermediate codes that are populated by this method:
        - NONE

        @return codes List[str]
        """
        assert len(nodes) > 0

        inputs = [t for t in inputs if not t.is_attr()]
        input_names = [self.tensor_name(t) for t in inputs]
        input_names_tuple = ', '.join(input_names)
        output_names = [self.tensor_name(t) for t in outputs]
        output_names_tuple = ', '.join(output_names)

        # 'graph.segment(nodes)' ensures that if a tensor is no longer used (in RC group or in later code),
        # it's not included in 'outputs'.
        # And we will not generate 'return' statement for it, since it will cause the error
        # that the variable is not defined (because it has been 'del'-ed).

        with FunctionBlock('recompute', input_names, False) as fb:
            # The nodes to recompute share the same space of line_ids (or "node ids") with non-recomputable nodes.
            # e.g. those ids in subgraphs are not 0-based, and incremented after the preceding non-rc nodes and so on.
            #
            # So within the recomputing subgraph, tensors can be released if they are no longer used
            # i.e. not returned by the 'def recompute(...)'
            # since 'execplan.graph.segment(nodes)' will make all "free variables" as explicit inputs/outputs
            # to that subgraph.

            # for ncode in ModuleCodeGen._emit_nodes(nodes, lifecycle):
            #     fb.insert_body(ncode)
            fb.insert_body(self._emit_nodes(nodes, lifecycle, runtime_devid))
            fb.insert_body(f'return {output_names_tuple}')
        codes = [''] + fb.code + ['']
        codes.append(
            f'{output_names_tuple} = ckpt.checkpoint(recompute, {input_names_tuple}, use_reentrant=False)'
        )

        return codes

    def _get_param_first_used_pos(self, segment: IRSegment) -> Dict[IRFullTensor, int]:
        """
        Get the first used node index of each parameter in the segment.
        """
        # get all the parameters in the segment
        first_used_pos: Dict[IRFullTensor, int] = {}

        for i, node in enumerate(segment.nodes()):
            # parameters are used as inputs of the node
            for tin in IRSegment.get_objects_from_complex(node.inputs()):
                if isinstance(tin, IRSubTensor) and tin.is_param() and tin.parent not in first_used_pos:
                    first_used_pos[tin.parent] = i

        return first_used_pos

    def clear(self):
        """
        Clear buffer that used for generating code
        """
        # module init code
        self.model_init_statements: List[str] = list()
        # module forward code
        self.model_methods_bodies: List[List[str]] = list()
        # module member name
        self.symbols = SymbolTable()
        # batch size
        self.batch_size = None
