######################
Advanced Llama Example
######################

************
Introduction
************

This example demonstrates how to train llama models in challenging distributed configurations by nnscaler.

************
Requirements
************

Assume following packages have been installed in the environment. ::

    nnscaler
    transformers==4.40.0
    datasets==2.20.0
    apex
    flash-attn

*nnScaler* is a framework for distributed training by automatically partitioning the model.
Apart from the core nnScaler library, it also includes a mini-trainer for modern model training.
You can find related documents and examples at `nnScaler <https://nnscaler.readthedocs.io/en/latest/>`_.

*transformers* and *datasets* are used to prepare the data and loading the Llama model.

To speed up the training,
`apex <https://github.com/NVIDIA/apex>`_ and `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ are required.
You can install them by following instructions in their official repositories.
We also recommend to launch training in a docker directly,
like ``nvidia/pytorch:24.02-py3`` and ``rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0``.

****************
Supported Models
****************

The following table lists the supported model architectures and their corresponding distributed environments.
A performance analysis for these will be provided later in the document.
We plan to support more model combinations in the future and encourage you to experiment and contribute.

+-------------------------------------+-----------------+-------------+---------------+
| Model ID                            | Sequence Length | Device Type | Device Number |
+=====================================+=================+=============+===============+
| meta-llama/Meta-Llama-3-8B-Instruct | 131072          | H100        | 8             |
+-------------------------------------+-----------------+-------------+---------------+
| meta-llama/Meta-Llama-3-70B         | 8192            | MI300       | 16            |
+-------------------------------------+-----------------+-------------+---------------+

****************
Data Preparation
****************

We use the `bookcorpus <https://huggingface.co/datasets/bookcorpus>`_ dataset for demonstrating in this doc.
You can change related code to support your own dataset.
Here we give an example that downloads and tokenizes ``bookcorpus`` for Llama.

In the example command below,
the dataset is tokenized by `Meta-Llama-3-8B-Instruct <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`_ tokenizer and grouped into 128K,
tokenized data is saved in ``bookcorpus_llama3_128K`` directory.

.. code-block:: bash

    python bookcorpus.py \
        --data_path_or_name bookcorpus/bookcorpus \
        --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
        --save_path ./bookcorpus_llama3_128K \
        --sequence_length 131072

********
Training
********

nnScaler adopts a compiler approach to train deep learning models on multiple deivices.
The processing pipeline is divided into two stages:

#. Compile stage: trace the original PyTorch model and get the dataflow graph.
   Analyze the graph and generate an efficient plan for distributed training.
   Generate python code for the runtime stage.
#. Runtime stage: run the generated python code to train the model.

For better user experience, we recommend to use separate commands for the compile and runtime stages at your first trial of nnScaler.
You can use the ``Run`` command directly to combine the two stages when you are familiar with the system.

**Note**: currently we only tested ``"_attn_implementation": "flash_attention_2"`` and ``"use_cache": false`` in the config file.
Other configurations may trigger errors.

Trace Strategy
==============

During compiling, the time cost of trace model graph can vary significantly depending on the tracing strategy employed.
Below are some reference time to trace ``meta-llama/Meta-Llama-3-8B-Instruct`` with different strategies and different context length,
the time tested on one single A100 80GB:

+------------------------+----------------+--------------+
| Strategy               | Context Length | Time/seconds |
+========================+================+==============+
| `reuse_cache`          | 8k             | 8.11         |
+------------------------+----------------+--------------+
| `reuse_cache`          | 32k            | 11.06        |
+------------------------+----------------+--------------+
| `reuse_cache`          | 64k            | 15.36        |
+------------------------+----------------+--------------+
| `reuse_cache`          | 128k           | 26.29        |
+------------------------+----------------+--------------+
| `cuda_run_cpu_offload` | 8k             | 55.28        |
+------------------------+----------------+--------------+
| `cuda_run_cpu_offload` | 32k            | 194.27       |
+------------------------+----------------+--------------+
| `cuda_run_cpu_offload` | 64k            | 342.15       |
+------------------------+----------------+--------------+
| `cuda_run_cpu_offload` | 128k           | 789.15       |
+------------------------+----------------+--------------+

The trace strategy can be changed by setting ``--trace_strategy`` option.
Please note that different strategies have different applicable scenarios.
For more information and explanation to the different strategies, please read :doc:`../parallel_module`.

Register Customized Function
============================

Llama3's vocabulary size is about 128K, which is much larger then the 32K in Llama2.
When the sequence length is very long like 128K,
the output tensor size of the last projection layer is quite large:
128K x 128K x 2 bytes = 32GB in fp16 or bf16.
Although this tensor can be partitioned evenly to 8 GPUs, 4GB memory is still large due to limited GPU memory.
What makes it worse is that we need to store additional 8GB for ``log_softmax`` and ``cross_entropy_loss`` computation.
In order to reduce the memory consumption:

* we split the input sequence on each device to chunks of 1K tokens
* for each chunk, we recompute a function which is composed of last projection layer, log_softmax and loss
* as a result, we only need to store the input tensor to the last projection layer,
  whose initial size is 128K x 4K x 2 bytes = 1GB, which is much smaller than 32GB

You can find the detailed implementation in ``chunk_linear_cross_entropy.py``.
The interface of the ``chunk_linear_cross_entropy`` function is
``(hidden_states: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, padding_idx: int, chunk_size: int) -> torch.Tensor``,
where

* ``hidden_states`` is the output of the last transformer layer, with shape ``[batch_size, sequence_length, hidden_size]``
* ``weight`` is the weight matrix of the last projection layer, with shape ``[vocab_size, hidden_size]``
* ``labels`` is the target labels, with shape ``[batch_size, sequence_length]``
* ``padding_idx`` is the padding index
* ``chunk_size`` is the size of the chunk, default is 1024

We want to register this function to nnScaler and tell it to partition this function along batch size or sequence dimension.
A possible annotation is ``b l d^, n^ d^, b l -> b l``.
Here ``b`` stands for batch size, ``l`` stands for sequence length, ``d`` stands for hidden size, and ``n`` stands for vocab size.
The ``^`` means the dimension cannot be partitioned.
More details about the annotation can be found in :doc:`../register_custom_op`.

You can enable this customized function by passing ``--enable-chunk-loss`` to ``train.py`` when compiling.
When the sequence length is small (like 8K), this option can be turned off.

Profile Communication
=====================

To generate an efficient distributed plan in your environment, we recommend to profile the intra-node communication before compiling.
The profiler records the time of different communication primitives (like allgather, allreduce, reducescatter and alltoall) for some message sizes.
If the profiling is skipped, the system will use MI250's data by default. You can use the command below to profile.

.. code-block:: bash

    torchrun --nnodes=<X> --nproc_per_node=<Y> -m nnscaler.profiler.benchmark_comm

Checkpoint
==========

``train.py`` will save the model checkpoint in the ``./checkpoints`` directory by default.
You can change the checkpoint directory by updating the ``CheckpointConfig`` in the source code.

nnScalar saves checkpoints in shards: each rank may save parameters and optimizer states in a file.
These checkpoints can be directly loaded by nnScaler if the partitioning strategy is the same.
If you want to evaluate the checkpoints on downstream tasks, you need to merge the shards into a single file.
You can use the following command to merge the shards:

.. code-block:: bash

    python ckpt_merger.py --ckpt_dir ./checkpoints --output_fname ./merged.ckpt

The merged checkpoint can be loaded by nnScaler by setting the ``--resume_path`` option to the merged file.

If the script is modified for different hardware configurations.

* All sharded checkpoint files should be collected and placed in a same directory before ``ckpt_merger.py`` is called.
* If the config is changed (plan_ngus/runtime_ngus/etc), the sharded checkpoint can not be used anymore.
  You need to merge them so the trainer can load from merged checkpoint.

********************
Performance Analysis
********************

The flops of the forward computation for llama is

.. math:: 2 \cdot ( param\_num \cdot seqlen + 2 \cdot layer\_num \cdot hidden\_dim \cdot seqlen ^ 2)

Llama3 8B 128K on 8xH100
========================

Commands below is used for this setting.

Compile
-------

.. code-block:: bash

    python train.py --run_mode compile --model_id meta-llama/Meta-Llama-3-8B-Instruct --dataset_path ./bookcorpus_llama3_128K --plan_ngpus=8 --runtime_ngpus=8 --recompute_modules LlamaDecoderLayer --enable-chunk-loss 2>&1 | tee compile.log

Run
---

.. code-block:: bash

    torchrun --nproc_per_node=8 train.py --model_id meta-llama/Meta-Llama-3-8B-Instruct --dataset_path ./bookcorpus_llama3_128K --plan_ngpus=8 --runtime_ngpus=8 --recompute_modules LlamaDecoderLayer --enable-chunk-loss 2>&1 | tee run.log

For the 8B model, the forward flops is about 11104.35 TFLOPs. The detailed config is as following:

* .. math:: param\_num = 8 \times 10^9
* .. math:: seqlen = 128 \times 1024
* .. math:: layer\_num = 32
* .. math:: hidden\_dim = 4096

Generally, the computational cost of backpropagation is twice that of the forward pass.
In addition, the gradient accumulation number is set to 4.
As a result, the flops for a step of the training script is 133252.22 TFLOPs.

We execute the training script on a node with 8xH100 80GB HBM3.
The time cost is about 41.12s for a step.
The theoretical BF16 computational speed of the H100 is 989 TFLOPS.
Combine them together, this script can achieve 40.96% MFU.
You can optimize the performance furtherly by

* add more devices to avoid recomputation: in order to fit the model into the memory, we recompute by layer.
* do more kernel optimizations. For example, the swiglu activation can be fused into the matmul ahead of it.

Llama3 70B 8K on 16xMI300
=========================

Different from the 8B example, a merged command is used for the multi-node setting.
Since 70b model is trained on 2 nodes, we use mpi to execute ``torchrun`` on them at the same time.
If you want to run the command on your own, you can replace ``MASTER_ADDR`` with the IP address of the first node,
``MASTER_PORT`` with the available port on the first node and fill ``OMPI_COMM_WORLD_RANK`` with 0 and 1 on two nodes respectively.

Combined Command
----------------

.. code-block:: bash

    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$$OMPI_COMM_WORLD_RANK --master_addr="$$MASTER_ADDR" --master_port=$$MASTER_PORT train.py --name llama3-70b --model_id meta-llama/Meta-Llama-3-70B --dataset_path ./bookcorpus_llama3_8K --gpu_mem_constraint 153 --plan_ngpus=8 --runtime_ngpus=16 --grad_accumulation_steps 64 --pipeline_pivots LlamaDecoderLayer --pipeline_nstages auto 2>&1 | tee run.log

Note that in the command above, we enable searching for pipeline parallelism and set the possible pipeline stage boundaries
by passing ``--pipeline_pivots LlamaDecoderLayer --pipeline_nstages auto``.

For the 70B model, the flops for forward and backward is about 3968.41 TFLOPs. The detailed config is as following:

* .. math:: param\_num = 70 \times 10^9
* .. math:: seqlen = 8192
* .. math:: layer\_num = 80
* .. math:: hidden\_dim = 8192

`MI300X <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf>`_'s
peak theoritical performance for BF16 is 1307.4 TFLOPS.
It takes about 100.3 s to finish 64 gradient accumulation steps in the experiment.
Combine them together, the MFU of this distributed plan is 24.2%.

Based on AutoDist's analysis, the low utilization results from following aspects

* We observe MFU for important operators are low.
  For example, ``linear``'s MFU is 40% ~ 50%, the real MFU of ``flash-attn`` is 14%.
* Like the 8B 128K example, we can fuse operators like RoPE and swiglu to reduce time.
* There are two pipeline stages each with 4 devices.
  In each stage, communication takes about 450ms and computation takes about 1000ms.
  According to our experiences, the communication time is higher than expected. Adding more devices may help to reduce it since the optimizer states still takes about 52GB in each device.
* Enlarge search space in the future.
  Currently we only consider plan_ngpus=8 and fix the pipeline schedule to be ``1f1b``.
  We can refine this assumption in the future.

*********
Debugging
*********

Since the large setting is challenging, it is recommended to use a smaller model for debugging.
For example, you can use the following command to prepare data and train a smaller llama3
(same architecture, but with 4 decoder layers) model on two GPUs.

.. code-block:: bash

    # prepare data
    python bookcorpus.py --data_path_or_name bookcorpus/bookcorpus --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct --save_path ./bookcorpus_llama3_4K --sequence_length 4096

    # build the mini model
    python create_mini_model.py --model_id meta-llama/Meta-Llama-3-8B-Instruct --output_id ./llama3_mini

    # compile and run using data parallelism + zero1
    torchrun --nproc_per_node=2 train.py --plan_ngpus 1 --runtime_ngpus 2 --name llama3_debug --model_id ./llama3_mini --dataset_path ./bookcorpus_llama3_4K
