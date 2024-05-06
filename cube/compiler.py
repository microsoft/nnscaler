# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Tuple, Union, Optional
import torch
import time
import os
import logging

import cube

from cube.ir.cten import IRObject
from cube.ir.tensor import IRFullTensor
from cube.ir.unique import IDGenerator
from cube.graph.gener.gen import IRAdapterGener
from cube.graph.graph import IRGraph
from cube.ir.cten import IRObject
from cube.ir.tensor import IRFullTensor
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.schedule.schedplan import SchedulePlan

from cube.execplan import ExecutionPlan
from cube.execplan.planpass.fusion import DiffFusion
from cube.execplan.planpass.grouping import Grouping

from cube.codegen import ModuleCodeGen, ScheduleCodeGen

from cube.runtime.device import DeviceGroup

from cube.program import Program, SemanticDataLoader, SemanticModel
from cube.flags import CompileFlag
from cube.utils import print_each_rank


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def compile(model: Optional[Union[torch.nn.Module, SemanticModel]], *args,
            PAS: Union[Callable, Tuple[Callable, Callable, Callable]] = None,
            model_dynamic_shape: bool = False,
            load_graph_file: Optional[str] = None,
            save_graph_file: Optional[str] = None,
            comm_cost_fn: Optional[Callable] = None,
            override = True,
            load_content = True,
            scale: Union[bool, int] = False) -> Callable:
    """Cube compile entry

    Examples:

    ```
    @cube.compile(model, data, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    ```

    Args:
        model (SemanticModel | torch.nn.Module | None): single-device model. For processes having `local_rank != 0`,
            the model can be passed with None to model initialization cost.
        args (Tuple[Any]): compile function example inputs
        PAS (Callable | Tuple[Callable, Callable, Callable]): policy to transform and schedule graph
        model_dynamic_shape (bool): whether to compile model with dynamic shape
        load_graph_file (str | None): 
            load cached graph. This will skip parsing the function and model.
            Note the user should keep correct `fullmodel.pt` if load_content is True.
        save_graph_file (str | None): save parsed graph before applying policy.
        comm_cost_fn (Optional[Callable]): communication cost function, which
            takes in an IRAdapterPrim, and outputs a cost in float. By default (None) use
            communication volume.
        override (bool): If true, the generated code will override exsisting
            files (if they are already existed.), otherwise, use the already existed
            generated code, i.e., the policy won't take effect. Default true.
        load_content (bool): If true, will load parameter from exsiting saved models.
            Otherwise, will initial model parameters with empty tensor.
        scale (Union[bool, int]): If true, will scale the generated code to the
            total launched number of GPUs. If int, will scale to the specified number.
            Default False, no scaling.

    Returns:
        Callable: compiled training iteration
    """
    # clean global status
    Program().clear()
    IDGenerator().clear()
    assert PAS is not None, f'PAS should be callable function'

    if DeviceGroup().local_rank == 0:
        assert isinstance(model, (torch.nn.Module, SemanticModel)), \
            f"Require torch.nn.Module or SemanticModel for processes with local rank 0."
    is_semantic_model = isinstance(model, SemanticModel)
    if not isinstance(model, SemanticModel):
        model = SemanticModel(model)
    model.save_content = load_content
    model.dynamic_shape = model_dynamic_shape

    inputs = [model]
    for arg in args:
        assert not isinstance(arg, (torch.nn.Module, SemanticModel)), f"Only one model can be input for compile"
        if isinstance(arg, torch.utils.data.DataLoader):
            arg = SemanticDataLoader(arg)
        elif isinstance(arg, torch.Tensor):
            tensor = arg
            arg = IRFullTensor(arg.shape, name='tensor', 
                               requires_grad=arg.requires_grad,
                               dtype=arg.dtype).tosub()
            arg._value = tensor
        else:
            arg = IRObject('obj', value=arg)
        inputs.append(arg)

    myrank = DeviceGroup().rank

    def decorator(fn: Callable) -> Callable:

        filename = 'gencode{}.py'

        if not override and os.path.exists(filename.format(myrank)):
            filename = filename.format(myrank)
            # load module code
            if is_semantic_model:
                _logger.info(f'loading existed module from {filename} ...')
                model.load_module(filename)
            # load schedule code
            _logger.info(f'loading existed schedule from {filename} ...')
            return cube.load_default_schedule(filename)

        ndevices = DeviceGroup().world_size
        local_ndevs = DeviceGroup().local_world_size
        nnodes = ndevices // local_ndevs
        if nnodes > 1:
            compile_ranks = list(range(0, ndevices, local_ndevs))
            compile_group = DeviceGroup().get_group(compile_ranks)

        if DeviceGroup().local_rank == 0:

            compile_start = time.time()
            resource = cube.runtime.resource.EnvResource()

            # run once to get model structure and tensor shape
            graph = None
            if load_graph_file is None:
                start = time.time()
                outputs = fn(*inputs)
                if outputs is None:
                    outputs = []
                elif not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]
                # setup program input
                pinputs = []
                for input in inputs[1:]: # we don't consider `model` as inputs
                    if isinstance(input, SemanticModel):
                        pinputs.append('model')
                    elif isinstance(input, SemanticDataLoader):
                        pinputs.append(input.object)
                    else:
                        pinputs.append(input)
                Program().set_input(pinputs)
                # setup program output
                Program().set_output(outputs)
                Program().finalize()
                span = time.time() - start
                graph = Program().get_graph()
                _logger.info('finish parsing iteration: {:.2f} s'.format(span))
            else:
                # get cube graph of a iteration from tracer or cached file
                start = time.time()
                graph = IRGraph.load(load_graph_file)
                span = time.time() - start
                _logger.info('finish loading graph from {}: {:.2f} s'.format(load_graph_file, span))

            if save_graph_file is not None and save_graph_file != load_graph_file:
                _logger.info(f'saving graph to {save_graph_file}')
                graph.dump(save_graph_file)

            # checking graph consistency between multiple nodes
            if nnodes > 1:
                checksum = graph.checksum(strict=True)
                _logger.debug(f'checking graph consistency (local md5: {checksum}) ...')
                state = torch.tensor([ord(c) for c in checksum], dtype=torch.int,
                                     device=torch.cuda.current_device())
                gather_list = None
                if DeviceGroup().node_rank == 0:
                    gather_list = [torch.empty_like(state) for _ in range(nnodes)]
                torch.distributed.gather(state, gather_list, dst=0, group=compile_group)
                if DeviceGroup().node_rank == 0:
                    inconsistent_nodes = []
                    for node_rank, checksum in enumerate(gather_list):
                        if state.ne(checksum).any():
                            inconsistent_nodes.append(node_rank)
                    if len(inconsistent_nodes) > 0:
                        raise RuntimeError(
                            f'graph status is inconsistent on node ranks: {inconsistent_nodes}. '
                            f'Please check pytorch version or re-run the compilation.'
                        )

            # run policy
            start = time.time()
            assert callable(PAS), f"Policy PAS is not callable"
            graph = PAS(graph, resource)
            span = time.time() - start
            _logger.info('finish policy expression: {:.2f} s'.format(span))

            if not isinstance(graph, IRGraph):
                raise RuntimeError("Expected policy return IRGraph")

            # check assignment and remove anchor node
            for node in graph.nodes(flatten=True):
                # skip graph anchor and multiref: they will be removed or replaced by system
                if isinstance(node, IRGraphAnchor) or node.name == 'multiref':
                    graph.assign(node, 0)
                if isinstance(node, IRPyFunc):
                    graph.assign(node, 0)
                if len(node.device) == 0:
                    raise RuntimeError(f"Node {node} device is not set")

            # generate adapter
            start = time.time()
            graph = IRAdapterGener.gen(graph, cost_fn=comm_cost_fn)
            span = time.time() - start
            _logger.info('finish generating adapters: {:.2f} s'.format(span))

            if graph.sched is not None:
                start = time.time()
                graph.sched.apply()
                span = time.time() - start
                _logger.info('finish planpass on applying schedule strategy: {:.2f} s'.format(span))

            # to execution plan
            start = time.time()
            if isinstance(graph.sched, SchedulePlan):
                execplan = ExecutionPlan.from_schedplan(graph.sched)
            else:
                execplan = ExecutionPlan.from_graph(graph)
            if CompileFlag.visualize_plan:
                execplan.visualize('plan.png')
            span = time.time() - start
            _logger.info('finish lowering to execution plan: {:.2f} s'.format(span))

            # plan pass for communication optimization
            start = time.time()
            execplan = DiffFusion.apply(execplan)
            span = time.time() - start
            _logger.info('finish planpass on diff-fusion operations: {:.2f} s'.format(span))

            # execplan.visualize(outfile='plan.png')

            # plan pass for computation grouping
            if not graph.sched:
                start = time.time()
                execplan = Grouping.apply(execplan)
                span = time.time() - start
                _logger.info('finish planpass on grouping operations: {:.2f} s'.format(span))

            # execplan.graph.reset_dependency()
            # execplan.analyze(outfile='execplan.png')

            start = time.time()
            local_world_size = DeviceGroup().local_world_size
            # code generation
            scale_ndevs = None
            if scale:
                scale_ndevs = resource.ngpus if isinstance(scale, bool) else scale
            mgener = ModuleCodeGen(execplan, scale_ndevs=scale_ndevs)
            sgener = ScheduleCodeGen(execplan, scale_ndevs=scale_ndevs)
            for local_rank in range(local_world_size):
                rank = DeviceGroup().node_rank * local_world_size + local_rank
                fname = filename.format(rank)
                # generate spatial module code
                mgener.gen(rank, outfile=fname, attach=False)
                # generate temporal schedule code
                sgener.gen(
                    device = rank,
                    outfile = fname,
                    attach=True
                )
            span = time.time() - start
            _logger.info('finish generating code: {:.2f} seconds'.format(span))

            compile_end = time.time()
            compile_time = compile_end - compile_start
            _logger.info('compile time: {:.2f} seconds'.format(compile_time))

        if torch.distributed.is_initialized():
            if DeviceGroup().local_rank != 0 and CompileFlag.worker_sleep > 0:
                _logger.info(f'rank [{DeviceGroup().rank}] starts sleeping {CompileFlag.worker_sleep} seconds...')
                time.sleep(CompileFlag.worker_sleep)
            torch.distributed.barrier()

        # load module
        filename = filename.format(myrank)
        print_each_rank(f'loading generated module from {filename} ...', logger=_logger)
        model.load_module(filename)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        model.dummy_input = None
        # load temporal schedule
        print_each_rank(f'loading generated schedule from {filename} ...', logger=_logger)
        return cube.load_default_schedule(filename)

    return decorator
