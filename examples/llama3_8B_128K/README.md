# Introduction

This example demonstrates how to train llama3-8B-128k model with 8xH100s or 8xA100s.

# Requirements

To run this example, you need to install the following packages:

```text
nnscaler
transformers==4.40.0
datasets==2.20.0
apex
flash-attn
```

*nnScaler* is a framework for distributed training by automatically partitioning the model. Apart from the core nnScaler library, it also includes a mini-trainer for modern model training. You can find related documents and examples at [nnScaler](TODO).

*transformers* and *datasets* are required to prepare the data and loading the Llama model.

To speed up the training process, [*apex*](https://github.com/NVIDIA/apex) and [*flash-attn*](https://github.com/Dao-AILab/flash-attention) are required. You can install them by following instructions in their official repositories. We also recommend to launch the script under a Nvidia docker directly, like nvidia/pytorch:24.02-py3.

# Data Preparation

We use the [bookcorpus](https://huggingface.co/datasets/bookcorpus) dataset for training. The dataset is tokenized with the [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) tokenizer. Tokenized data is saved in the `bookcorpus_llama3_128K` directory.

```bash
python bookcorpus.py --data_path_or_name bookcorpus/bookcorpus --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct --save_path ./bookcorpus_llama3_128K --sequence_length 131072
```

# Training

nnScaler adopts a compiler approach to launch the distributed training. The processing pipeline is divided into two stages:

1. Compile stage: trace the original PyTorch model and get the dataflow graph. Analyze the graph and generate an efficient plan for distributed training. Generate python code for the runtime stage.
2. Runtime stage: run the generated python code to train the model.

For better user experience, we recommend to use separate commands for the compile and runtime stages. You can also use the `Run` command directly to combine the two stages.

**Note**: currently we only tested `"_attn_implementation": "flash_attention_2"` and `"use_cache": false` in the config file. Other configurations may not work properly.

## Register Customized Function

Llama3's vocabulary size is about 128K, which is much larger then the 32K in Llama2. At the same time the sequence length in this example is 128K, the output tensor size of the last projection layer is quite large: 128K x 128K x 2 bytes = 32GB.
Although this tensor can be partitioned evenly to 8 GPUs, 4GB memory is still quite large for limited GPU memory. What makes it worse is that we need to store additional 8GB for `log_softmax` and `cross_entropy_loss` computation.
In order to reduce the memory consumption:
- we split the input sequence on each device to chunks of 1K tokens
- for each chunk, we recompute a function which is composed of last projection layer, log_softmax and loss
- as a result, we only need to store the input tensor to the last projection layer, whose initial size is 128K x 4K x 2 bytes = 1GB, which is much smaller than 32GB

You can find the detailed implementation in `chunk_linear_cross_entropy.py`.
The interface of the `chunk_linear_cross_entropy` function is `(hidden_states: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, padding_idx: int, chunk_size: int) -> torch.Tensor`, where
- `hidden_states` is the output of the last transformer layer, with shape `[batch_size, sequence_length, hidden_size]`
- `weight` is the weight matrix of the last projection layer, with shape `[vocab_size, hidden_size]`
- `labels` is the target labels, with shape `[batch_size, sequence_length]`
- `padding_idx` is the padding index
- `chunk_size` is the size of the chunk, default is 1024

We want to register this function to nnScaler and tell it to partition this function along batch size or sequence dimension. A possible annotation is `b l d^, n^ d^, b l -> b l`. Here `b` stands for batch size, `l` stands for sequence length, `d` stands for hidden size, and `n` stands for vocab size. The `^` means the dimension cannot be partitioned. More details about the annotation can be found in related documents.

## Compile

```bash
python train.py --run_mode compile --model_id meta-llama/Meta-Llama-3-8B-Instruct --dataset_path ./bookcorpus_llama3_128K --plan_ngpus=8 --runtime_ngpus=8 2>&1 | tee compile.log
```

## Run

```bash
torchrun --nproc_per_node=8 train.py --model_id meta-llama/Meta-Llama-3-8B-Instruct --dataset_path ./bookcorpus_llama3_128K --plan_ngpus=8 --runtime_ngpus=8 2>&1 | tee run.log
```

## Checkpoint

This script will save the model checkpoint in the `./checkpoints` directory. You can change the checkpoint directory by updating the `CheckpointConfig` in the `train.py` script.

nnScalar saves checkpoints in shards: each rank may save parameters and optimizer states in a file. These checkpoints can be directly loaded by nnScaler if the partitioning strategy is the same. If you want to evaluate the checkpoints on downstream tasks, you need to merge the shards into a single file. You can use the following command to merge the shards:

```bash
python ckpt_merger.py --ckpt_dir ./checkpoints --output_fname ./merged.ckpt
```

The merged checkpoint can be loaded by nnScaler by setting the `--resume_path` option to the merged file.

If the script is modified for different hardware configurations.
- All sharded checkpoint files should be collected and placed in a same directory before `ckpt_merger.py` is called.
- If the config is changed (plan_ngus/runtime_ngus/etc), the sharded checkpoint can not be used anymore. You need to merge them so the trainer can load from merged checkpoint.

# Performance

The flops of the forward computation for Llama3 is

$2 \cdot ( param\_num \cdot seqlen + 2 \cdot layer\_num \cdot hidden\_dim \cdot seqlen ^ 2)$

For the 8B model, the forward flops is about 11104.35 TFLOPs. The detailed config is as following:
- $param\_num = 8 \times 10^9$
- $seqlen = 128 \times 1024$
- $layer\_num = 32$
- $hidden\_dim = 4096$

Generally, the computational cost of backpropagation is twice that of the forward pass. In addition, the gradient accumulation number is set to 4. As a result, the flops for a step of the training script is 133252.22 TFLOPs.

We execute the training script on a node with 8xH100 80GB HBM3. The time cost is about 41.12s for a step. The theoretical BF16 computational speed of the H100 is 989 TFLOPS. Combine them together, this script can achieve 40.96% MFU. You can optimize the performance furtherly by
- add more devices to avoid recomputation: in order to fit the model into the memory, we recompute by layer.
- do more kernel optimizations. For example, the swiglu activation can be fused into the matmul ahead of it.

# Debugging

Since the 128K config is challenging, it is recommended to use a smaller model for debugging. For example, you can use the following command to prepare data and train a smaller llama3 (same architecture, but with 4 decoder layers) model on two GPUs.

```bash
# prepare data
python bookcorpus.py --data_path_or_name bookcorpus/bookcorpus --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct --save_path ./bookcorpus_llama3_4K --sequence_length 4096

# build the mini model
python create_mini_model.py --model_id meta-llama/Meta-Llama-3-8B-Instruct --output_id ./llama3_mini

# compile and run using data parallelism + zero1
torchrun --nproc_per_node=2 train.py --plan_ngpus 1 --runtime_ngpus 2 --name llama3_debug --model_id ./llama3_mini --dataset_path ./bookcorpus_llama3_4K

```
