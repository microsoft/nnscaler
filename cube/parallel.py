# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
import types
from typing import Callable, Any, Dict, Optional, Type, Union, TypeVar
from pathlib import Path
import inspect
import sys
import importlib
from dataclasses import dataclass
from contextlib import contextmanager
import logging

import torch
from cube.graph.parser.fx.parser import FxModuleParser

from cube.ir.cten import IRObject
from cube.ir.tensor import IRFullTensor

from cube.flags import CompileFlag, RuntimeFlag

from cube.graph import IRGraph
from cube.graph import parser
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.schedule.schedplan import SchedulePlan
from cube.graph.gener.gen import IRAdapterGener

from cube.codegen import ModuleCodeGen
from cube.execplan import ExecutionPlan
from cube.execplan.planpass.grouping import Grouping
from cube.execplan.planpass.fusion import DiffFusion
from cube.ir.unique import IDGenerator
from cube.program import Program
from cube.runtime.adapter.reducer import Reducer
from cube.runtime.module import CubeModule, ParallelModule


logger = logging.getLogger(__name__)


@dataclass
class ComputeConfig:
    plan_ngpus: int
    runtime_ngpus: int


@contextmanager
def _flags(flags, warning_on_override=True, /, **kwargs):
    old_flags = {}
    for k, v in kwargs.items():
        old_flags[k] = getattr(flags, k)
        if old_flags[k] != v:
            if warning_on_override:
                logger.warning(f"{flags}.{k}={old_flags[k]} is not supported. Changed to {v}.")
        setattr(flags, k, v)
    try:
        yield
    finally:
        for k, v in old_flags.items():
            setattr(flags, k, v)


def _compile_flags():
    return _flags(CompileFlag, use_zero=False, async_reducer=False, reducer_op='sum', async_comm=False)


def _runtime_flags(**kwargs):
    return _flags(RuntimeFlag, False, **kwargs)


def _complex(val: Any):
    """Complex to CPU"""
    if isinstance(val, tuple):
        return tuple(_complex(t) for t in val)
    if isinstance(val, list):
        return list(_complex(t) for t in val)
    if isinstance(val, dict):
        return {_complex(key):_complex(val) for key, val in val.items()}
    if isinstance(val, set):
        return {_complex(t) for t in val}
    if isinstance(val, torch.Tensor):
        return val.cpu()
    return val


def _get_full_qualified_name(obj: Any) -> str:
    """Get full qualified name of an object"""
    if inspect.isclass(obj):
        return obj.__module__ + '.' + obj.__qualname__
    return obj.__module__ + '.' + obj.__class__.__qualname__


def _add_cube_savedir_to_syspath(cube_savedir: str) -> Path:
    cube_savedir = Path(cube_savedir).resolve()
    cube_savedir.mkdir(parents=True, exist_ok=True)
    if str(cube_savedir) not in sys.path:
        sys.path.append(str(cube_savedir))
    return cube_savedir


def _is_any_gencode_loaded(namespace: str) -> bool:
    """Check if a module is loaded"""
    for m in sys.modules.values():
        if m.__name__.startswith(namespace + '.' + _GENCODE_FILE_PREFIX):
            return True
    return False


def _get_arg_default_values(fn) -> Dict[str, Any]:
    args = inspect.signature(inspect.unwrap(fn))
    return {k: v.default for k, v in args.parameters.items()}


_GENCODE_FILE_PREFIX = 'gencode'
_GENCODE_FILE_TEMPLATE = _GENCODE_FILE_PREFIX + '{}.py'  # 'gencode{}.py'
_CUBE_MODULE_NAMESPACE = '_cube_modules'


def _gencode(
        module: torch.nn.Module,
        dummy_input: dict,
        pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
        compute_config: ComputeConfig,
        *,
        dynamic_shape: bool = True,
        cube_savedir: Union[str, Path] = './.cube',
        override: bool = False,
        instance_name: Optional[str] = None
    ) -> None:
    """
    Generate cube module source code from a torch module, and save it to file.
    Generated module will be save according to its full qualified name.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distingish them.

    For example, if the module is `torchscale.x.y`, then the generated module will be save to
    `cube_savedir/_cube_modules/torchscale/x/y/instance_name`.

    Args:
        module (torch.nn.Module): the module to be compiled
        dummy_input (dict): the dummy input for the module
        pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy
        compute_config (ComputeConfig): the environment resource
        dynamic_shape (bool): whether to use dynamic shape
        override (bool): If true, source code will be regenerated even if generated code exists.
        cube_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module.

    Returns:
        None
    """
    # put cube_savedir into sys.path
    # so we can import the generated module with its namespace later
    cube_savedir = _add_cube_savedir_to_syspath(cube_savedir)

    instance_name = instance_name.strip('.') if instance_name else ''
    instance_namespace = f'.{instance_name}' if instance_name else ''
    namespace = f'{_CUBE_MODULE_NAMESPACE}.{_get_full_qualified_name(module)}{instance_namespace}'
    outdir = cube_savedir / Path(namespace.replace('.', '/').strip('/'))
    outdir.mkdir(parents=True, exist_ok=True)

    # decision matrix for code generation
    # override flag | dir condition(imported, empty, match, unmatched) | action
    # ---------------------------------------------------------
    #   True   | empty | generate
    #   True   | imported | raise error
    #   True   | match | generate
    #   True   | unmatch | generate
    #   False  | empty | generate
    #   False  | match | do nothing
    #   False  | unmatch | raise error
    #   False  | imported | doesn't matter
    if not override:
        # check if the module is already generated
        expected_output_files = [outdir / _GENCODE_FILE_TEMPLATE.format(rank) for rank in range(compute_config.runtime_ngpus)]
        expected_output_files.append(outdir / FxModuleParser.ATTR_CONTENT_FILE)
        expected_output_files.append(outdir / FxModuleParser.ATTR_MAP_FILE)
        expected_output_files.append(outdir / ParallelModule.COMPUTE_CONFIG_FILE)
        existing_output_files = [f for f in outdir.glob('*') if f.is_file()]
        if existing_output_files:
            if all([output_file.exists() for output_file in expected_output_files]) \
                and len(existing_output_files) == len(expected_output_files) \
                and torch.load(outdir / ParallelModule.COMPUTE_CONFIG_FILE) == compute_config:
                return
            elif all(f.suffix != '.py'  for f in existing_output_files):
                # No python source code is generated.
                # which means its last generation failed.
                # in this case, we can reuse the same directory safely.
                pass
            else:
                raise RuntimeError(f'Output directory {outdir} is not empty. '
                                   f'And the existing files do not match with current config.')
    else:
        # check if the module is already loaded
        if _is_any_gencode_loaded(namespace):
            raise RuntimeError(f'Output directory {outdir} is already loaded. '
                               f'You can not override a loaded module.')
        # clear existing generated files
        for f in outdir.glob('*'):
            if f.is_file():
                f.unlink()

    # reset environment
    program = Program()
    program.clear()
    IDGenerator().clear()

    module.cpu()
    forward_args_default = _get_arg_default_values(module.forward)
    for v in forward_args_default.values():
        if v is not inspect.Parameter.empty and not isinstance(v, (int, str, float, bool, type(None))):
            raise ValueError(f"Default value type {type(v)} of forward args is not supported.")

    # generate fx graph
    dummy_input = _complex(dummy_input)
    fx_graph = parser.to_fx_graph(module, dummy_input)

    # generate ir logic graph
    ir_graph = parser.to_ir_graph(
        fx_graph, dummy_input, outdir, dynamic_shape
    )

    # generate dummy inputs for logic graph
    # that is, generate IRObject/IRFullTensor for fx graph dummpy input
    fx_input_nodes = [node for node in fx_graph.graph.nodes if node.op == 'placeholder']
    # the inputs of graph is different with original forward args
    # so we get the real forward args from fx inputs
    forward_args = {
        node.target: forward_args_default.get(node.target, inspect.Parameter.empty)
        for node in fx_input_nodes
    }
    ir_dummy_inputs = []
    for node in fx_input_nodes:
        if node.target.startswith('*'):  # *args or **kwargs
            if node.target.strip('*') in dummy_input:
                raise ValueError(f"Input {node.target}: *args or **kwargs is not suppported")
            ir_dummy_inputs.append(None)  # always set None to *args/**kwargs
        elif node.target in dummy_input:
            ir_dummy_inputs.append(dummy_input[node.target])
        else:
            raise ValueError(f"Input {node.target} not in dummy input. Default value is not supported.")
    for i in range(len(ir_dummy_inputs)):
        if isinstance(ir_dummy_inputs[i], torch.Tensor):
            # note: we will always set tensor to require gradient, which may
            # generate backward communications in adapter. However, as long as
            # the data doesn't require gradient in real runtime, the backward
            # communication will not be triggered.
            ir_dummy_inputs[i] = IRFullTensor(
                shape=ir_dummy_inputs[i].size(),
                name=fx_input_nodes[i].target,
                requires_grad=True,
                dtype=ir_dummy_inputs[i].dtype).tosub()
            ir_dummy_inputs[i].grad = ir_dummy_inputs[i].parent.grad.tosub()
        else:
            ir_dummy_inputs[i] = IRObject(
                name=fx_input_nodes[i].target,
                value=ir_dummy_inputs[i]
            )
    # generate complete ir graph
    ir_dummy_outputs = ir_graph(*ir_dummy_inputs)

    graph = program.get_graph()
    graph.backward()
    program.set_input(ir_dummy_inputs)
    if ir_dummy_outputs is None: ir_dummy_outputs = []
    elif not (isinstance(ir_dummy_outputs, tuple) or isinstance(ir_dummy_outputs, list)):
        ir_dummy_outputs = [ir_dummy_outputs]
    program.set_output(ir_dummy_outputs)
    program.finalize()

    graph = pas_policy(graph, compute_config)
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
    graph = IRAdapterGener.gen(graph, cost_fn=None)
    if graph.sched is not None:
        graph.sched.apply()

    if isinstance(graph.sched, SchedulePlan):
        execplan = ExecutionPlan.from_schedplan(graph.sched)
    else:
        execplan = ExecutionPlan.from_graph(graph)

    execplan = DiffFusion.apply(execplan)
    # plan pass for computation grouping
    if not graph.sched:
        execplan = Grouping.apply(execplan)

    # code generation
    torch.save(compute_config, outdir / ParallelModule.COMPUTE_CONFIG_FILE)
    runtime_ngpus = None if compute_config.plan_ngpus == compute_config.runtime_ngpus else compute_config.runtime_ngpus
    assert len(execplan.graph.device) == compute_config.plan_ngpus, f"{execplan.graph.device}"
    mgener = ModuleCodeGen(execplan, scale_ndevs=runtime_ngpus)
    for rank in range(compute_config.runtime_ngpus):
        filename = _GENCODE_FILE_TEMPLATE.format(rank)
        mgener.gen(rank, forward_args=forward_args, outfile=outdir / filename, attach=False, as_parallel_module=True)


def _load_cube_module_class(
    module_class: Type[torch.nn.Module],
    *,
    cube_savedir: Union[str, Path] = './.cube',
    instance_name: Optional[str] = None,
) -> Type[ParallelModule]:
    """
    Load the generated cube module class.

    Please note that the cube module class should be generated beforehand by _gencode().

    Args:
        module_class (Type[torch.nn.Module]): the original module class
        cube_savedir (Union[str, Path]): the directory to load generated code
        instance_name (Optional[str]): the instance name of the generated module.
    """
    _add_cube_savedir_to_syspath(cube_savedir)
    rank = torch.distributed.get_rank()
    instance_name = instance_name.strip('.') if instance_name else ''
    instance_namespace = f'.{instance_name}' if instance_name else ''
    gen_imported = importlib.import_module(
        f'{_CUBE_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}{instance_namespace}.{Path(_GENCODE_FILE_TEMPLATE.format(rank)).stem}'
    )
    cube_module_class = gen_imported.GenModel
    # rewrite class name and module name
    cube_module_class.__name__ = module_class.__name__
    cube_module_class.__qualname__ = module_class.__qualname__
    # cube_module_class.__module__ = module_class.__module__
    cube_module_class.__orig_module_class__ = module_class  # save the original module class
    return cube_module_class


def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_input: dict,
    pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
    compute_config: ComputeConfig,
    *,
    dynamic_shape: bool = True,
    cube_savedir: Union[str, Path] = './.cube',
    override: bool = False,
    instance_name: Optional[str] = None,
    load_module: bool = True,
) -> Union[None, ParallelModule, Type[ParallelModule]]:
    """
    Convert a torch.nn.Module object or class to CubeModule object or class.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distingish them.

    Currently you must use a shared file system to share the generated files (like mounted Azure Blob)
    Or you can unset load_module flag, and manually copy the generated files to other nodes.
    After all nodes have the generated files, you can call parallelize() again with load_module flag set.

    if the input is a module object.
        The module object will be copied to cpu to handle possible insufficient gpu memory.
        The training flag will be the same as the original module

    Args:
        module_or_module_class (Union[torch.nn.Module, Type[torch.nn.Module]]): the module or module class to be compiled
        dummy_input (dict): the dummy input for the module
        pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy
        compute_config (ComputeConfig): the environment resource
        dynamic_shape (bool): whether to use dynamic shape
        override (bool): If true, source code will be regenerated even if generated code exists.
        cube_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module.
        load_module (bool): whether to load the generated module after done.

    Returns:
        Union[CubeModule, Type[CubeModule], None]:
            if load_module flag is set, return the converted CubeModule object or class
            if load_module flag is not set, return None
    """
    if (
        isinstance(module_or_module_class, CubeModule) or
        (inspect.isclass(module_or_module_class) and issubclass(module_or_module_class, CubeModule))
    ):
        return module_or_module_class if load_module else None

    is_module_class = inspect.isclass(module_or_module_class)
    module_class = module_or_module_class if is_module_class else module_or_module_class.__class__

    # genereate code only in node0
    # if it is not in a torchrun environment, just generate.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if is_module_class:
            # it should only have 1 `self` parameter
            if len(inspect.signature(module_or_module_class.__init__).parameters) > 1:
                raise ValueError("Module class __init__ should be parameter-free.")
            try:
                module = module_or_module_class()
            except Exception as e:
                raise RuntimeError(f"Error when create module instance.") from e
        else:
            module = module_or_module_class

        if any(isinstance(m, CubeModule) for m in module.modules()):
            raise RuntimeError('CubeModule can not be nested.')
        with _compile_flags():
            _gencode(
                module,
                dummy_input,
                pas_policy,
                compute_config,
                dynamic_shape=dynamic_shape,
                override=override,
                cube_savedir=cube_savedir,
                instance_name=instance_name,
            )
        if is_module_class:
            del module

    if load_module:
        if not torch.distributed.is_initialized(): # we only support loading in torchrun environment
            raise RuntimeError("Load CubeModule failed: torch.distributed is not initialized.")
        torch.distributed.barrier()
        cube_module_class = _load_cube_module_class(
            module_class,
            cube_savedir=cube_savedir,
            instance_name=instance_name,
        )
        if is_module_class:
            return cube_module_class
        else:
            cube_module = cube_module_class()
            cube_module.train(module_or_module_class.training)  # set training state to the same as original module
            return cube_module


class ParallelOptimizer(torch.optim.Optimizer):
    """
    A optimizer stub to support parallelized module.
    The returned optimizer of build_optimizer() will have the same methods in this class.
    """
    def sync_shard_grad(self):
        """
        Sync the shard gradients of the module from nodes with same shard to the optimizer.
        Please note this is called automatically in optimizer.step().
        But If you want to access the gradients before optimizer.step(),
        you need to call this function manually.
        """
        ...

    def register_reducer_pre_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        """
        Register pre hooks to reducers which will be applied before gradient synchronization.

        The pre-hooks will be applied one by one following the order of registration.

        Args:
            fn (Callable[[Reducer, torch.Tensor], None]): a callable function that takes a reducer and a gradient as input and optionally updates the gradient.
        """
        ...

    def register_reducer_post_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        """
        Register post hooks to reducers which will be applied after gradient synchronization.

        The post-hooks will be applied one by one following the order of registration.

        Args:
            fn (Callable[[Reducer, torch.Tensor], None]): a callable function that takes a reducer and a gradient as input and optionally updates the gradient.
        """
        ...

OptimizerT = TypeVar('OptimizerT', bound=torch.optim.Optimizer)


def build_optimizer(
    module: torch.nn.Module,
    optimizer_fn: Union[Type[OptimizerT], Callable[..., OptimizerT]],
    *args,
    **kwargs,
) -> OptimizerT:
    """
    Build an optimizer for a module.

    To support parallelized module (CubeModule), we need to hook 4 places:
    1. optimizer constructor:
        the parameters of optimizer will not be the same with the parameters of the module if we use zero
        so we need to replace the parameters of optimizer with CubeModule.parameters_for_optimizer
        It is impossible to make this change transparent to end users.
    2. optimizer.step():
        we need to call optimier.sync_shard_grad() to sync the gradients of the module before optimizer.step().
        In zero mode, we have to call CubeModule.gather_params() after optimizer.step()
    3. optimizer.zero_grad():
        We need to call CubeModule.zero_grad() after optimizer.zero_grad()
    4. backward():
        you need to call optimizer.sync_shard_grad() manually if you want to read the gradients of the module before optimizer.step().

    Please note this DOES NOT work in end2end mode.

    Args:
        module (torch.nn.Module): the module to be optimized
        optimizer_fn (Union[Type[torch.optim.Optimizer], Callable[..., torch.optim.Optimizer]]):
            It can be the optimizer class or optimizer factory function.
            If it is a factory function, the signature should be the same with optimizer class constructor.
        *args: the args for optimizer constructor
        **kwargs: the kwargs for optimizer constructor

    Returns:
        torch.optim.Optimizer: the optimizer you should use to train the module
        The optimizer is created by optimizer_fn,
        and will be patched with the methods in ParallelModule class to support parallelized module.
    """

    if isinstance(module, CubeModule) and not isinstance(module, ParallelModule):
        raise RuntimeError("End2End mode is not supported")

    RuntimeFlag.skip_reducer = True
    RuntimeFlag.skip_zero_grad = False

    def _local_parameters(module: torch.nn.Module):
        gen = module._named_members(lambda m: m._parameters.items())
        for _, param in gen:
            yield param

    optimizer: torch.optim.Optimizer = optimizer_fn(_local_parameters(module), *args, **kwargs)

    def _step_pre_hook(opt, *args, **kwargs):
        opt.sync_shard_grad()
    def _step_post_hook(opt, *args, **kwargs):
        for m in module.modules():
            if isinstance(m, ParallelModule):
                m.gather_params()
            else:
                assert not isinstance(m, CubeModule), "Only ParallelModule is supported in this mode"
    optimizer.register_step_pre_hook(_step_pre_hook)
    optimizer.register_step_post_hook(_step_post_hook)

    orig_zero_grad = optimizer.zero_grad
    def _patched_zero_grad_hook(self, set_to_none: bool = True):
        orig_zero_grad(set_to_none)
        for m in module.modules():
            if isinstance(m, ParallelModule):
                m.zero_grad()
            else:
                assert not isinstance(m, CubeModule), "Only ParallelModule is supported in this mode"
    optimizer.zero_grad = types.MethodType(_patched_zero_grad_hook, optimizer)

    def _sync_shard_grad(self):
        with _runtime_flags(skip_reducer=False):
            for m in module.modules():
                if isinstance(m, ParallelModule):
                    m.sync_grad()
                else:
                    assert not isinstance(m, CubeModule), "Only ParallelModule is supported in this mode"
    optimizer.sync_shard_grad = types.MethodType(_sync_shard_grad, optimizer)

    def _register_reducer_pre_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        for m in module.modules():
            if isinstance(m, ParallelModule):
                for reducer in m.reducers:
                    reducer.register_pre_hook(partial(fn, reducer))
            else:
                assert not isinstance(m, CubeModule), "Only ParallelModule is supported in this mode"

    def _register_reducer_post_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        for m in module.modules():
            if isinstance(m, ParallelModule):
                for reducer in m.reducers:
                    reducer.register_post_hook(partial(fn, reducer))
            else:
                assert not isinstance(m, CubeModule), "Only ParallelModule is supported in this mode"

    optimizer.register_reducer_pre_hook = types.MethodType(_register_reducer_pre_hook, optimizer)
    optimizer.register_reducer_post_hook = types.MethodType(_register_reducer_post_hook, optimizer)

    return optimizer
