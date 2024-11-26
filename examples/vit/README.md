# ViT Example

## Introduction

This example demonstrates how to use nnscaler to fine-tuning a transformer model. Here we use ViT as an example.

## Requirements

To run this example, you need to install the packages listed in the `requirements.txt` file. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

*nnScaler* is a framework for distributed training by automatically partitioning the model. Apart from the core nnScaler library, it also includes a mini-trainer for modern model training. You can find related documents and examples at [nnScaler](https://github.com/microsoft/nnscaler)

*transformers* and *datasets* are required to prepare the data and loading the model.

The implementation is inspired by [here](https://medium.com/@supersjgk/fine-tuning-vision-transformer-with-hugging-face-and-pytorch-df19839d5396). Many thanks to the author.


### Run

First go to `examples/vit` directory, You can use the following command to run the example:

1. Use transformer.train() to train the model
    - `python examples/vit/vit_cli.py`: will use `DataParallel` to train the model.
        It will utilize all your GPUs in current node.
        You can specify the GPUs with `CUDA_VISIBLE_DEVICES` env variable.
    - `torchrun --nproc_per_node=<gpus> --nnodes=<nodes> examples/vit/vit_cli.py`: will use `DistributedDataParallel` to train the model.

2. Use nnscaler to train the model
    `torchrun --nproc_per_node=<gpus> --nnodes=<nodes> $(which nnscaler-train) -f train_cli_args.yaml`

In order to be consistent with `transformers.train()`,
we use dataloader/scheduler from `transformers`.
See `accelerator_dataloader_fn` and `scheduler_fn` functions in the code for details.
If you don't need to be consistent with `transformers`, you can just use your own dataloader/scheduler.

Please note `nnscaler` only supports 1 optimizer parameter group for now. So we also disable multiple parameter groups in the code when you use `transformers.train()`.
See `SingleParamGroupTrainer` in the code for details.

The loss of `nnscaler` will be exactly the same as `transformers` given gnorm clipping is disabled.
When gnorm clipping is enabled, the loss will be slightly different
due to the difference in gnorm calculation.
