##########################################
LongRope2 context length extension Example
##########################################

************
Introduction
************

`LongRoPE2 <https://arxiv.org/abs/2502.20082/>`_ is an advanced version of `LongRoPE <https://arxiv.org/abs/2402.13753>`_ that significantly improves long-context extension for RoPE-based LLMs. It has been adopted in Phi4-mini and Phi4-multimodal.

This example includes the training part for LongRope2. Before training, please using `LongRoPE repo <https://github.com/microsoft/LongRoPE>` for searching the rope extension scaling factor for your model.
This example provides the extension scaling factor of llama3-8b-base as a reference. If you want to have a try with llama3-8b-base, you can run this example directly.


***********
Preparation
***********

If this is the first time you use nnScalar, it would be better start with ``examples/llama`` for more using detail.
But it is OK to directly follow this example to run pass.

Assume following packages have been installed in the environment. ::

    nnscaler
    zstandard
    transformers>=4.48
    datasets
    tensorboard
    apex
    flash-attn

A new model config includes the longrope ``rope_scaling`` field and ``original_max_position_embeddings`` are needed, please reference ``examples/longrope2/llama3_8b_longrope2_config.json``


****************
Data Preparation
****************

We use ``HuggingFaceFW/fineweb-edu`` for short context window training and ``togethercomputer/RedPajama-Data-1T`` for long context window training.

.. code-block:: bash
    export PYTHONPATH=$PYTHONPATH:/home/USER_NAME/MagicCube:/home/USER_NAME/MagicCube/examples
    # download data to at MagicCube/examples/longrope2/data, will take around 100GB disk memory.
    python data/download.py
    # process the data to mix context window length format for long context training, will take around 900GB disk memory.
    python data/process.py --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B"

If you don't have large disk memory, i.e., 1 TB free memory, you could take a sub-dataset by modify the code.


********
Training
********

The main different compared with the common long context training example ``examples/llama`` is we need to pass ``--model_config`` to passin the rope extension scaling factor to the model.

.. code-block:: bash
    # compile the distributed code for llama3 model with dp2, tp4 on 8 gpus
    python train.py --run_mode compile --model_id "meta-llama/Meta-Llama-3-8B" --model_config llama3_8b_longrope2_config.json --dataset_path data/mix-context-win-short-8192-long-131072 --plan_ngpus=4 --runtime_ngpus=8 --recompute_modules LlamaDecoderLayer --gpu_mem_constraint 64 --enable-chunk-loss --grad_accumulation_steps 16 --max_train_steps 2250 2>&1 | tee compile.log
    # run the training job
    torchrun --nproc_per_node=8 train.py --model_id "meta-llama/Meta-Llama-3-8B" --model_config llama3_8b_longrope2_config.json --dataset_path data/mix-context-win-short-8192-long-131072 --plan_ngpus=4 --runtime_ngpus=8 --recompute_modules LlamaDecoderLayer --gpu_mem_constraint 64 --enable-chunk-loss --grad_accumulation_steps 16 --max_train_steps 2250 2>&1 | tee run.log


**********
Additional
**********

More details about how to change distributed plan or merge checkpoints, please reference ``examples/llama/README.rst``.
