# PyTorch Lightning
[//]: # (# Pytorch Lightning support)

We support Pytorch Lightning by `NnScalerStrategy` and `NnScalerPrecision`. You can use `nnscaler` strategy in pytorch lightning like this:

```python
compute_config=ComputeConfig(...)
policy = ...
trainer = Trainer(
    ...,
    strategy=NnScalerStrategy(
        compute_config=compute_config, pas_policy=..., gen_savedir=...,
        ...
    ),
    plugins=[NnScalerPrecision(precision, ...)],
    ...
)
trainer.fit(...)
```

## Model

### Dummy input

We need a dummy input to trace the forward function. You can specify it in two ways:

1. Add `dummy_forward_args` property to your model class, which should be a dictionary of forward inputs.
2. You can also add `dummy_forward_args_fn`, which will be used to convert the sample (loaded from train dataloader) to forward inputs.

### Rewritten members

We will rewrite two functions:
1. `forward` function: As we explained before, the `forward` function will be replaced with a distributed version.
2. `log` function: We will rewrite the `log` function to force the `sync_dist_group` to be set properly when `sync_dist=True`.

We will also set all trainable modules to None to reduce memory usage.

To make sure the model can be used with nnscaler strategy, you should follow these rules:

1. All trainable parameters should only be used in forward function.
If it is used outside forward, it should be in torch.no_grad context.
Otherwise, as we don't create reduce-op outside forward, its gradient will be incorrect.
2. Train/Validate/Test should use exactly the same graph.
3. All functions replying on the trainable modules should be rewritten with forward function.
After our conversion, all those modules will be None.

## Strategy

The constructor argument of `NnScalerStrategy` is the combination of `Strategy`'s constructor and `nnscaler.parallize` function. You can refer to the documentation of `Strategy` and `nnscaler.parallize` for more details.

One special argument is `state_dict_type`, which specify the format in which the state of the model and optimizers gets saved into the checkpoint.

- `"sharded"`: Each rank saves its shard of weights and optimizer states to a file.
The checkpoint is a folder with as many files as the local world size.
- `"deduped"`: Each rank saves its deduped shard of weights and optimizer states to a file. The checkpoint is a folder with as many files as the local world size.

## Precision

It has exactly the same constructor arguments as `Precision`'s constructor.

Currently we support `32-true`, `16-true`, `bf16-true`, `16-mixed`, `bf16-mixed`.
You can specify a grad scaler when you use `16-true`.

## Checkpoint

If this is the first time you run the model, and you have a pretrained model, you must load the pretrained model before you pass it to the `Trainer` constructor. The tracing process will use the pretrained model weights to trace the forward function.

Just like other pytorch lightning strategy,
you can resume from a checkpoint by specifying the `ckpt_path` argument in the `Trainer.fit` function.
Please note when the parallel plan is changed (i.e you re-trace the model with different configurations),
the checkpoints become incompatible, and can't be loaded any more.
You must firstly merge the checkpoints to a merged checkpoint with `NnScalerStrategy.merge_checkpoint` and then load the merged checkpoint as a regular checkpoint.

```python
def merge_checkpoint(cls, checkpoint_files: List[str], output_file: str) -> None:
```
where `checkpoint_files` is a list of checkpoint files to merge, and `output_file` is the output file path.

## Limitations

Currently, nnScaler only supports: 
- single parameter group.
- single optimizer.
- single learning rate scheduler.