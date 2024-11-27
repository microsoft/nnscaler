# DeepSeek Example

## Introduction

This example demonstrates how to train deepseek-coder-v2-lite-2k on 8xH100s or 8xA100s.

## Requirements

To run this example, you need to install the following packages:

```text
nnscaler
transformers==4.40.0
datasets==2.20.0
apex
flash-attn
grouped_gemm==1.1.4
```

We recommend to launch the script under a Nvidia docker directly, like `nvidia/pytorch:24.02-py3`. You can find grouped_gemm at https://github.com/fanshiqing/grouped_gemm.

## Data Preparation

Like the *llama3_8B_128K* example, [bookcorpus](https://huggingface.co/datasets/bookcorpus) dataset is used for training. You can use the following command directly

```bash
python bookcorpus.py --data_path_or_name bookcorpus/bookcorpus --tokenizer_path_or_name deepseek-ai/DeepSeek-Coder-V2-Lite-Base --save_path ./bookcorpus_2k --sequence_length 2048
```

## Training

### Code Modification

Modeling is based on the open source version for [deepseek coder v2](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/tree/main). To boost the training performance and be compatible with nnScaler, the source code is modified. You can check modifications in details under `modeling` folder:

- `configuration_deepseek.py` and `modeling_deepseek.py` are identical with the public available ones.
- Token dispatching logics are in `moe_utils.py`, which is adapted from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py).
- Most of the modifications are in `modeling_deepseek_modifier.py`.

Similar to *llama3_8B_128K*, apex and flash-attn are introduced to reduce the execution time of RMSNorm and multi-head attention. In addition, there are several deepseek specific modifications:

- register the routing function with annotation to nnScaler, since it is composed of fine-grained irregular operators and generating the annoation automatically is non-trivial.
- the for loop based MoE implementation is replaced with an efficient implementation built on [cutlass](https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu). Along with kernel, separated expert weights are merged after loading the checkpoints.

### Distributed Config

The input data is organized into batches of 64 sequences whose length = 2048. The micro batch size is 4 and gradient accumulation step is 8. 8 GPUs are divided into 2 data parallel groups (4 GPUs maintain a full copy of weights).

You can use following commands to compile and run the model. Checkpoints can be merged by the script in *llama3_8B_128K*. If you want to load the weights to huggingface, the merged experts should be split to the original names.

**Compile**

```bash
python train.py --run_mode compile --model_id deepseek-ai/DeepSeek-Coder-V2-Lite-Base --dataset_path ./bookcorpus_2k --plan_ngpus=4 --runtime_ngpus=8 2>&1 | tee compile.log
```

**Run**

```bash
torchrun --nproc_per_node=8 train.py --model_id deepseek-ai/DeepSeek-Coder-V2-Lite-Base --dataset_path ./bookcorpus_2k --plan_ngpus=4 --runtime_ngpus=8 2>&1 | tee run.log
```

## Performance

We have tested the training script on 8xH100 and each step takes about 2s. A step is composed of 128K tokens and the number of activated params is about 2.65B. Combining them together, the MFU is about 13% (attention's FLOPs is omitted since the sequence is short in this ). The root cause is the low utilization rate of the MoE part. We collect statistics for the grouped gemm in the table below. Note that in deepseek coder v2 lite, there are 64 experts with hidden size = 2048, intermediate size = 1408, each token will be dispatched to 8 experts.

| # Dispatch Token | # Expert | forward / ms | backward / ms | MFU   |
| :----            | :----    | :----        | :----         | :---  |
| 4096             | 64       | 3.190        | 6.363         | 13.5% |
| 2048             | 32       | 1.851        | 3.367         | 12.3% |
| 8192             | 64       | 5.148        | 8.964         | 18.2% |
| 2048             | 16       | 1.613        | 2.459         | 15.8% |
| 16384            | 64       | 8.901        | 14.90         | 21.6% |
| 2048             | 8        | 1.663        | 2.329         | 16.1% |

To improve the performance, we recommend to

- Replace the cutlass kernel with better ones. Current script is based on grouped_gemm@v1.14.
- Fuse more kernels like rope and memory slicing in attention.
- There are about 16 * 8 = 128 GB space used to store the optimizer states. Adding more devices helps to save more memory and nnScaler can find a better plan then.
