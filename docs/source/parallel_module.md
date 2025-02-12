# Paralleling a Module

nnScaler can transform a `torch.nn.Module` into a parallel module, which is a specialized version of `torch.nn.Module` capable of running across multiple GPUs or nodes. This process hides the complexity of distributed training and inference from the user.

Currently, we support three kinds of parallelism: data parallelism, tensor parallelism and pipeline parallelism. We can also combine them for better performance.

Data parallelism and tensor parallelism can be supported for any module, but pipeline parallelism is only supported for end2end modules for scheduling reason.

An end2end module is a module which satisfies:
- the first argument of `module.forward` is the data sample, and every other argument should have default value, and use its default value in `module.forward` function.
- the first return value of `module.forward` is the loss (scalar tensor)

The above restrictions are necessary for the pipeline parallelism to work. Of course, you can still use the parallel module without pipeline parallelism for end2end modules.

## Examples

- Example 1: Parallelize the whole module

```python
import torch
from nnscaler import parallelize, ComputeConfig, build_optimizer

class LLM(torch.nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

llm_sample_input = ...              # dummy input will be used to do tracing
pas_policy = ...                    # the PAS policy, you can use autodist pas
compute_config = ComputeConfig(
    plan_ngpus=...,
    runtime_ngpus=...,
    use_zero=...,
    ...,
)                                   # compute environment config
ParallelizedLLM = parallelize(
    LLM,
    {'x': llm_sample_input},
    pas_policy,
    compute_config,
)
```

- Example 2: Parallelize submodules.

In this case, for non-paralle modules, they are replicated inside unit, and run data parallelism across units. See more details about unit in [Compute Config](./trainer) section.

```python
import torch
from nnscaler import parallelize, ComputeConfig, build_optimizer

class HeavyModule(torch.nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

class ParallelizedLLM(torch.nn.Module):
    def __init__(self, ...):
        ...
        # use parallelize to convert submodules
        heavy_module_sample_input = ...     # dummpy input will be used to do tracing
        pas_policy = ...                    # the PAS policy, you can use autodist pas
        compute_config = ComputeConfig(
            plan_ngpus=...,
            runtime_ngpus=...,
            use_zero=...,
            ...,
        )                                  # compute environment config
        self.heavy_module = parallelize(
            HeavyModule(),
            {'x': heavy_module_sample_input},
            pas_policy,
            compute_config,
        )
        # you can add other submodules here
        ...

    def forward(self, x, ...):
        # call other submodules
        ...
        x = self.heavy_module(x)
        ...
        # call other submodules
        return x
```

For both example 1 & 2, you can train/infer that module in multiple GPUs/Nodes just like a normal `torch.nn.Module`:

```python
# do inference exactly the same way
def infer(model: ParallelizedLLM, x):
    model.eval()
    with torch.inference_mode():
        return model(x)


# do training exactly the same way
# except you need to patch your optimizer to support distributed training via build_optimizer
def train(model: ParallelizedLLM, data):
    loss_fn = ...
    # build_optimizer function will help to create a distributed optimizer
    optimizer = build_optimizer(model, ...)

    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

- Example 3: Parallelize end2end module.
```python
class End2EndMLP(nn.Module):
    def __init__(self):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(8):
            self.layers.append(nn.Linear(16, 16, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss

    llm_sample_input = {'data': ..., 'target': ...}  # dummy input will be used to do tracing
    pas_policy = ...                    # the PAS policy, you can use autodist pas
    compute_config = ComputeConfig(
        plan_ngpus=...,
        runtime_ngpus=...,
        use_zero=...,
        use_end2end=True,
        ...,
    )                                   # compute environment config
    ParallelizedPipelinedLLM = parallelize(
        LLM,
        {'data': llm_sample_input},
        pas_policy,
        compute_config,
    )
```

For end2end modules, you can't use `Module.forward`.
Instead, you must use `ParallelModule.train_step` and `ParallelModule.infer_step` to train/infer the module.

```python
def infer(model: ParallelizedPipelinedLLM, data):
    model.eval()
    with torch.inference_mode():
        return model.infer_step(data)


def train(model: ParallelizedPipelinedLLM, data):
    # build_optimizer function will help to create a distributed optimizer
    optimizer = build_optimizer(model, ...)

    for i, x in enumerate(data):
        model.train()
        losses = model.train_step(x)
        optimizer.step()
        optimizer.zero_grad()
```

### BroadcastGenFilesStrategy

The broadcast strategy for new generated files.
Please note we never broadcast reused files (i.e., specified by `ReuseType`.).

```python
class BroadcastGenFilesStrategy(Enum):
    NONE = 'none'
    ALL = 'all'
    NO_WEIGHTS = 'no_weights'
    CODE = 'code'
```

1. `None`:  nothing will be broadcasted.

    You need to do it by yourself or the generated files are save in a shared directory (like azure blob).

2. `ALL`: broadcast all the generated files to all nodes (Recommended).

    This is useful when you want to run the same code on all nodes.
    please note the init weight files can be huge.

3. `NO_WEIGHTS`: broadcast all except init weights (Only for experts).

    Without weights, you can only construct the parallel module with `init_params=False`.
    You can then
    - Safe way: you can use `broadcast_weights` to get the weights from the workers who have init weights. By default rank 0 will run the `parallelize` and store all the generated files. So if local world size is bigger than plan_ngpus, you can use `broadcast_weights` to get the weights from workers on node0.
    - Risk Way: Load the weights from a checkpoint file with `module.load_state_dict`, `load_merged_state_dict` or `load_deduped_state_dict`.

    Please note: the non-persistent buffers will remain uninitialized after loading the checkpoints,
    because they are not saved in the state dict.
    To make sure all the buffers are initialized,  you still need to set `init_params=True` to make sure non-persistent buffers are initialized if you want to initialize weights by loading a checkpoint.

4. `CODE`: broadcast the new generated code only (Not recommeneded)
    It's your responsibility to make sure other necessary files are available on all nodes.

Here are some guidelines to choose the strategy:

1. When restarting a training and there is a successful previous run: As we have a previous run, the compiling process has been done before. So there will be no new generated files and no broadcast will happen no matter what this option is. Please be sure the reuse flag of `parallelize` is `MATCH`, so we can make sure the generated code is the same with the previous run.

2. When training a model from scratch. If there is only one node, `none` is good enough.
If there are multiple nodes, here are some strategies:

a. If use `none`, the user should run `parallelize(..., load_module=False, ..)`, and then copy all files to all nodes manually, so all nodes have the same files. Then the user load the module by running `parallelize(..., load_module=True, ..)`.

b. if they are using a NAS-like device to save generated files, and the upload/download speed is fast in the cluster, they can also use `none`, and just run `parallelize(..., load_module=True, ..)` to do the training.

c. If use `all`, then user can just run `parallelize(..., load_module=True, ..)` safely. (remember to set `nccl` communication timeout to a very big value to tolerate the duration of this `nccl` broadcast). This is the most recommended way.

d. If use `no_weights`. then user can run `parallelize(..., load_module=True, init_module_params=rank<plan_ngpus, ..)`. After module is loaded, the user should call `broadcast_weights(plan_ngpus)` manually to synchronize the module weights before training (Note all submodules have the same `plan_ngpus`). Here is an example:
```python
class Module(torch.nn.Module):
    ...
plan_ngpus = ...
rank = torch.distributed.get_rank()
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', default=1))
assert local_world_size < plan_ngpus

parallel_module = parallelize(Module(), ..., load_module=True, init_module_params=rank<plan_ngpus, broadcast_strategy='no_weights', ...)

broadcast_weights(parallel_module, plan_ngpus)
# now the module is ready to train
```
`no_weights` option is only suggested for experts, because you must be very careful to make it right
when there are non-persistent buffers in the module.

e. Currently `code` option is provided just for completeness. Do not suggest users to use.

### Module Parallelization

We have `parallelize` function to Convert a torch.nn.Module to a ParallelModule.
```python
def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_forward_args: Dict[str, Any],
    pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
    compute_config: ComputeConfig,
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    reuse: Union[ReuseType, str] = ReuseType.MATCH,
    instance_name: Optional[str] = None,
    load_module: bool = True,
    module_dtype:  Optional[torch.dtype] = None,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    init_module_params: bool = True,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
) -> Union[None, ParallelModule, Type[ParallelModule]]:
```
It has the following parameters:

- `module_or_module_class` (`Union[torch.nn.Module, Type[torch.nn.Module]]`): the module or module class to be compiled. Please note if the input is a module object, we will return a `ParallelModule` object. If the input is a module class, we will return a `ParallelModule` class.

- `dummy_forward_args` (`Dict[str, Any]`): the dummy input for the module forward.
The keys are the argument names of `Module.forward` function,
and the values are the dummy input for the arguments.
The dummy forward args will be used to trace the module.
Please note the module can't be parallelize if `Module.forward` has positional-only arguments.

- `pas_policy` (`Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]]`): the pas (partition-assign-schedule) policy, which describes how to place all computations across devices.
You need either pass a builtin PAS policy name or a a custom policy function which should take an `IRGraph` and a `ComputeConfig` as input, and return a new `IRGraph` with the PAS policy applied.
 We have 6 builtin PAS policies: `dp`, `tp`, `pp`, `data`, `hybrid`, and `autodist`. Please note all builtin PAS policies except `autodist` are only for test purpose. The `autodist` policy is the recommended policy for most cases.
 For details, please refer to [PAS Policies](./trainer) section.

- `compute_config` (`ComputeConfig`): the environment resource

- `reuse` (`ReuseType`): specify which part can be reused.

- `gen_savedir` (`Union[str, Path]`): the directory to save generated code

- `instance_name` (`Optional[str]`): the instance name of the generated module. If it is `None`, will use the default name `_`.

- `load_module` (`bool`): whether to load the generated module or module class after parallelization is done.
Currently the module can only be loaded in `torchrun` environment. So you can do the parallelization in any environment (with `load_module` unset), and load the module in `torchrun` environment.

- `init_module_params` (`bool`): If true, when we construct the module, all its parameters are initialized with the same value with when we traced.
Otherwise, they will be empty tensor.
This parameter will be passed to the module constructor,
so it is only used when `module_or_module_class` is a module object, and `load_module` is true.
See more details in the `ParallelModule APIs` section.

- `module_dtype` (`Optional[torch.dtype]`): the dtype of the module. Keep the module as it is if it is None.

- `module_fn` (`Optional[Callable[[], torch.nn.Module]]`): the function to create the module. Will use `__init__` if it is None. This parameter is only used when `module_or_module_class` is a module class.

- `broadcast_strategy` (`Union[str, BroadcastGenFilesStrategy]`): the broadcast strategy for new generated files.

Note:

1. This function can be used to convert both module object and module class to parallel module or parallel module class.
Among key-value arguments,
`module_fn` and `module_dtype` control how to create the module object.
whereas `init_module_params` controls how to load parallel module object after parallelization is done.

2. If you want to save multiple instances of the same module (with different configurations),
you can specify the `instance_name` to distinguish them.

3. `load_module` flag should be used with `broadcast_strategy`. See more details in the `BroadcastGenFilesStrategy` section.

4. if `reuse` is not set to `ReuseType.MATCH`,
the generated code in outdir will be removed EVEN IF the code generation fails in this call.

5. For `broadcast_strategy`, Please note that the broadcast will only be done in `torchrun` environment, and will throw an error if `torch.distributed` is not initialized and `broadcast_strategy` is not `NONE`.


### Optimizer Creation

We have `build_optimizer` to build an optimizer for distributed training.
```python
def build_optimizer(
    module: torch.nn.Module,
    optimizer_fn: Union[Type[OptimizerT], Callable[..., OptimizerT]],
    compute_config: Optional[ComputeConfig] = None,
    **kwargs,
) -> OptimizerT:
```
It has the following parameters:
- `module` (`torch.nn.Module`): the module to be optimized
- `optimizer_fn` (`Union[Type[torch.optim.Optimizer], Callable[..., torch.optim.Optimizer]]`):
    It can be the optimizer class or optimizer factory function.
    The first parameter of the `optimizer_fn` should be the module parameters.
- compute_config (Optional[ComputeConfig]):
    The config will be used to generate communication reducer.
    If it is None, Default configuration will be used when creating reducer for non-parallel modules.
- **kwargs: the kwargs will pass to `optimizer_fn`.

To support distributed training, in the function we need to hook 4 places (which we have done for you in `build_optimizer`. That's why you should use `build_optimizer` to create optimizer):

1. optimizer constructor:
    the parameters of optimizer will not be the same with the parameters of the module if we use zero.
    So we need to replace the parameters of optimizer with `ParallelModule.parameters_for_optimizer`.

2. `optimizer.step()`:
    we need to call `optimizer.sync_shard_grad()` to sync the gradients of the module before `optimizer.step()`.
    In zero mode, we have to call `ParallelModule.gather_params()` after `optimizer.step()`

3. `optimizer.zero_grad()`:
    We need to call `ParallelModule.zero_grad()` after `optimizer.zero_grad()`

`build_optimizer` will patch optimizer for you. Besides the above patches, we also add several utility functions/variables to optimizer:

1. `sync_shard_grad`: Sync the shard gradients of the module from nodes with same shard to the optimizer.
Please note the gradients are `None` until `optimizer.sync_shard_grad()` is called.
This function is called in optimizer's pre-step hook.  You need to manually call it in two cases:
    - If you want to access the gradients before `optimizer.step()`.
    - When closure is used in optimizer.step(). In this case, optimizer's pre-step hook will be called before `train_step`, so no gradients are synced.

2. `scale_grads`: Scale the gradients of the module by multiplying a factor. This function is useful to avoid overflow when the gradients are large. Please note you can only call this function **after** `sync_shard_grad`, because the gradients are `None` until `sync_shard_grad` is called.

3. `clip_gnorm`: Clip the gradients with global norm, and return the global gnorm value, it will sync grads across devices if necessary. This function is useful to avoid gradient explosion.

4. `register_reducer_pre_hook`, `register_reducer_post_hook`: Register pre/post hooks to reducers which will be applied before/after gradient synchronization. These hooks will apply to all the reducers (including `_non_parallel_module_reducer`) in the optimizer.

You can use `register_reducer_pre_hook` and `register_reducer_post_hook` to do some operations before/after gradient synchronization. Not all parameters are managed by reducers, so it is tricky to use them. Actually we don't encourage you to use these functions.

Here is one example (Assume we calculate loss with sum) showing how to carefully scale down the gradient locally and scale up the gradient after reduce. This is useful to avoid overflow when the gradients are large:.

```python
num_scale_units = ...
optimizer.register_reducer_pre_hook(lambda reducer, grad: grad.div_(num_scale_units)) # scale down with factor num_scale_units before reduce
optimizer.register_reducer_post_hook(lambda reducer, grad: grad.mul_(num_scale_units) # scale up with factor num_scale_units after reduce
```

5. `_non_parallel_module_reducer`: The reducer for the modules which are not parallelized. It is used to sync the parameters in those modules across units.

### ParallelModule APIs

The `ParallelModule` is a subclass of `torch.nn.Module`. It has the following APIs:

1. constructor
```python
def __init__(self, init_params=True):
    ...
```
You can use `init_params` to control whether to initialize the module parameters with the module parameters' values when we trace it. You can set it to `False` if you don't want to.

As noted before, in most cases, you still need to set `init_params=True` to make sure non-persistent buffers are initialized if you want to initialize weights by loading a checkpoint.


2. `train_step`
```python
def train_step(self,
    samples: List[Any],
    is_dummy_batch: Optional[List[bool]],
    scale_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
) -> List[Any]:
    ...
```
The training step function. It should be called in the training loop.
Please note:
    1. This function is only supported in end2end mode.
    2. Gradient accumulation is done in this function.
        You shouldn't do it outside this function,
        because `zero_grad` will be called in the beginning of this function

It has the following arguments:
- `samples` (`List[Any]`): a list of samples.
        if pipeline is used, it must have the same length as configured to pas policy.
- `is_dummy_batch` (`Optional[List[bool]]`): indicates whether the each micro-batch is dummy
- `scale_fn` (`Optional[Callable[[torch.Tensor], torch.Tensor]]`): the function to scale the loss

And it will return a list of outputs for the samples.

3. `infer_step`
```python
def infer_step(self, samples: List[Any]) -> List[Any]:
    ...
```
The inference step function. It should be called in the inference loop.
The input is a list of samples, and returns a list of outputs for the samples. If pipeline is used, it must have the same length as configured to pas policy.

### Checkpoint support

You can save/load the checkpoints for parallel modules.
Each rank will save/load its own checkpoint just like the normal module.

Note: The only exception is the non-persistent buffers, which will remain uninitialized after loading the checkpoints, because they are not saved in the state dict. To make sure all the buffers are initialized, you must initialize the module with `init_params=True`.

You can also merge the checkpoints from different ranks to a single checkpoint.
We call it a merged checkpoint. The merged checkpoint can be loaded by original module directly.
So you can easily share the checkpoint with the original module.

On the other hand, a lot of weights/state in the module and the optimizer will be the same in the ranks in parallel training. So we can save a lot of space by deduping the state dicts before saving them to the disk.

We provide two functions to help you save/load the merged checkpoint for the parallel module,
and two other functions to help you save/load the deduped state dicts for the parallel module.

#### `merge_state_dicts`
```python
def merge_state_dicts(
    module_state_dicts: List[Dict[str, Any]],
    optimizer_state_dicts: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
```

Merge a list of shard state dicts (one for each rank) to a single full state dict
Note: Only Adam-like optimizers are supported for merging

Please Note:
    We don't guarantee the devices of tensors are the same in the merged state dict.
    You can assume the device of the tensors in the merged state dict can be one of the following:
        1. the current device when running this function
        2. the current cuda device when running this function
        3. the device of the tensor in the original state dict
    When you load the state dict from file, you can just use `torch.load(..., map_location='...')` to unify the device of the tensors.


#### `load_merged_state_dicts`
```python
def load_merged_state_dicts(
        module: torch.nn.Module,
        module_state_dict: Dict[str, Any],
        optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        *,
        device: Union[str, torch.device] = None
) -> None:
```
Load the merged state dicts to the module, and optionally the optimizer to a specified device.

Please note the `device` parameter. If it is None, we will use `torch.cuda.current_device()` to get the current device. If you want to load the state dict to a specific device, you can set it to the device you want.


#### `deduped_state_dicts`

In parallel training, a lot of weights/state in the module and the optimizer will be the same in the ranks. So we can save a lot of space by deduping the state dicts before saving them to the disk.

```python
def deduped_state_dict(
    module: torch.nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
```

#### `load_deduped_state_dict`

```python
def load_deduped_state_dict(
    module: torch.nn.Module,
    module_state_dict: Dict[str, Any],
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = None
) -> None:
```

### Dataset

We use the same dataset/dataloader as pytorch. For example, you can use `torch.utils.data.DistributedSampler` to create a distributed sampler.

`ParallelModule`s running in the same unit should use the same input, and will get the same output. `ParallelModule`s running in different units should use different input and will get different output (similar to data parallelism). The gradients of all parameters will be synced across all the devices automatically.

Take `torch.utils.data.DistributedSampler` for example, you can create the sampler like this:
```python
def create_distributed_sampler(dataset):
    return torch.utils.data.DistributedSampler(
        dataset=dataset,
        num_replicas=compute_config.runtime_ngpus // compute_config.plan_ngpus,
        rank=torch.distributed.get_rank() // compute_config.plan_ngpus,
        ...,
    )
```

### self.training support

To parallelize the training process, we firstly need to trace the module and get a static computational graph.

A common problem with static graph is that it is impossible to handle control flow.

But on the other hand, `self.training` is very common used in module forward method.
So we add a very limited support for `self.training` in tracing.

Please note that user code is flattened and transformed into a single `ParallelModule` at runtime, so `training` is a global module state, and we don't support the case that user want to set a sub-module's training to True but remaining modules to False.
