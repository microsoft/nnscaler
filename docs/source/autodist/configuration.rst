AutoDist Configuration Reference
====================================

This document provides a comprehensive guide to all configuration options available in AutoDist's ``AutoDistConfig`` class.

Overview
--------

``AutoDistConfig`` is the central configuration class for AutoDist, allowing you to control various aspects of automatic parallelization including memory optimization, pipeline parallelism, tensor parallelism, and recomputation strategies.

Basic Usage
-----------

.. code-block:: python

    from nnscaler.autodist.autodist_config import AutoDistConfig
    
    # Basic configuration
    config = AutoDistConfig(
        task_name='my_experiment',
        memory_constraint=32,  # 32GB memory limit
        recompute_modules='transformer.layer'  # Recompute transformer layers
    )

Configuration Parameters
------------------------

Task Configuration
~~~~~~~~~~~~~~~~~~

**task_name** (*str*, optional, default: ``'default'``)
    The name of the current task to distinguish different runs. Used for naming saved plans and logs.

    .. code-block:: python
    
        config = AutoDistConfig(task_name='bert_large_training')

Memory Management
~~~~~~~~~~~~~~~~~

**consider_mem** (*bool*, optional, default: ``True``)
    Whether to consider memory constraints when searching for parallelization plans.

**memory_constraint** (*float*, optional, default: ``32``)
    The memory constraint for each device in GB. AutoDist will ensure that the parallelization plan fits within this memory limit.

    .. code-block:: python
    
        config = AutoDistConfig(memory_constraint=80)  # 80GB A100

**memory_granularity** (*int*, optional, default: ``1``)
    The memory granularity in bytes. Used for memory profiling and estimation.

**transient_mem_coef** (*float*, optional, default: ``2``)
    Coefficient for estimating transient memory size. Formula: ``transient_mem_size = transient_mem_coef * (1st_largest_infer_mem + 2nd_largest_infer_mem)``.
    
    Reduce this value if operators consume/generate very large tensors (≥4GB).

Optimizer Configuration
~~~~~~~~~~~~~~~~~~~~~~~

**opt_resident_coef** (*int*, optional, default: ``2``)
    Coefficient for optimizer resident state compared to model weight size.
    
    Common cases:
    
    - FP32 training with Adam: ``2`` (FP32 momentum1 + FP32 momentum2)
    - FP16/BF16 training with Adam: ``6`` (FP32 momentum1 + FP32 momentum2 + FP32 weight)
    - FP16/BF16 training with memory-efficient Adam: ``4`` (FP32 momentum1 + FP32 momentum2)

**opt_transient_coef** (*int*, optional, default: ``0``)
    Coefficient for optimizer transient state compared to model weight size.
    
    Common cases:
    
    - FP32 training with Adam: ``0``
    - FP16/BF16 training with Adam without internal cast: ``2`` (FP32 gradient)
    - FP16/BF16 training with memory-efficient Adam without internal cast: ``4`` (FP32 weight + FP32 gradient)

Recomputation
~~~~~~~~~~~~~

**recompute_modules** (*str*, optional, default: ``''``)
    Module names to recompute, separated by commas. Recomputation trades computation for memory by not storing intermediate activations during forward pass and recomputing them during backward pass. Note that recomputation still requires storing some tensors for gradient computation, so the memory savings depend on the specific model structure and recomputation granularity.
    
    Examples:
    
    .. code-block:: python
    
        # Recompute specific modules
        config = AutoDistConfig(recompute_modules='transformer.layer,attention')
        
        # Recompute entire model
        config = AutoDistConfig(recompute_modules='ROOT')
        
        # Recompute multiple specific modules
        config = AutoDistConfig(recompute_modules='encoder.layer,decoder.layer')
    
    **Note**: Module names can be any suffix of the full module name. For example, ``layer`` will match ``transformer.layer``, ``encoder.layer``, etc. ``ROOT`` recomputes the entire model but may not always provide maximum memory savings due to the need to store intermediate tensors for backward pass.

ZeRO Optimization
~~~~~~~~~~~~~~~~~

**zero_stage** (*int*, optional, default: ``0``)
    ZeRO optimization stage (see `ZeRO paper <https://arxiv.org/abs/1910.02054>`_).
    
    - ``0``: No ZeRO optimization
    - ``1``: Optimizer state partitioning

**zero_ngroups** (*int*, optional, default: ``1``)
    Number of ZeRO groups to balance memory usage and communication cost. Larger values use more memory but reduce communication overhead.

Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~~

**pipeline_pivots** (*str*, optional, default: ``''``)
    Module names that serve as pipeline stage boundaries, separated by commas.
    
    .. code-block:: python
    
        config = AutoDistConfig(pipeline_pivots='encoder,decoder')

**pipeline_nstages** (*int* or *'auto'*, optional, default: ``'auto'``)
    Number of pipeline stages. Set to ``1`` to disable pipeline parallelism.
    
    - ``'auto'``: Automatically determine optimal number of stages
    - ``int``: Fixed number of stages

**pipeline_scheduler** (*str*, optional, default: ``'1f1b'``)
    Pipeline scheduling strategy. Currently only supports ``'1f1b'`` (1-forward-1-backward).

**max_pipeline_bubble_ratio** (*float*, optional, default: ``0.2``)
    Maximum allowed bubble ratio in pipeline parallelism. Higher values allow more pipeline bubbles but explore larger search space.

**max_pipeline_unbalance_ratio** (*float*, optional, default: ``0.5``)
    Maximum unbalance ratio between pipeline stages (min_stage_time / max_stage_time). Higher values require better balance but reduce search space.

Mesh and Parallelism
~~~~~~~~~~~~~~~~~~~~

**mesh_row** (*int*, optional, default: ``1``)
    Number of available nodes in the device mesh.

**mesh_col** (*int*, optional, default: ``1``)
    Number of available devices per node in the device mesh.

**world_size** (*int*, optional, default: ``1``)
    Total number of devices (mesh_row × mesh_col × scale_factor).

**micro_batch_size** (*int*, optional, default: ``1``)
    Micro batch size for gradient accumulation.

**update_freq** (*int*, optional, default: ``1``)
    Update frequency. The effective batch size is micro_batch_size × update_freq.

Profiling and Search
~~~~~~~~~~~~~~~~~~~~

**profile_dir** (*str*, optional, default: ``~/.cache/nnscaler/autodist/1.0/get_node_arch()``)
    Directory to store profiling results for computation cost estimation.

**parallel_profile** (*bool*, optional, default: ``True``)
    Whether to profile on multiple devices in parallel. Set to ``False`` for sequential profiling on a single device.

**re_profile** (*bool*, optional, default: ``False``)
    Whether to override existing profiling results and re-profile operations.

**topk** (*int*, optional, default: ``20``)
    Number of parallelization plans to generate for robustness. Higher values provide more options but increase search time.

**solver** (*str*, optional, default: ``'dp'``)
    Solver algorithm for SPMD parallelism:
    
    - ``'dp'``: Dynamic programming
    - ``'ilp'``: Integer linear programming

**nproc** (*int*, optional, default: ``1``)
    Number of processes for pipeline parallelism search.

Plan Management
~~~~~~~~~~~~~~~

**load_plan_path** (*str*, optional, default: ``''``)
    Path to load an existing parallelization plan. When specified, skips plan searching and uses the loaded plan.

**save_plan_path** (*str*, optional, default: ``''``)
    Path to save the generated parallelization plan for reuse.

**partition_constraints_path** (*str*, optional, default: ``''``)
    Path to partition constraints file. See :doc:`solver_interface/partition_constraints` for details.

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

**is_train** (*bool*, optional, default: ``True``)
    Whether the model is for training or inference. Affects memory estimation and operator selection.

Debug and Optimization
~~~~~~~~~~~~~~~~~~~~~~

**verbose** (*bool*, optional, default: ``False``)
    Whether to print verbose information during plan generation.

**ignore_small_tensor_threshold** (*int*, optional, default: ``1``)
    Tensor size threshold (in elements) to ignore during analysis. Small tensors below this threshold are not considered for partitioning.

Example Configurations
----------------------

High Memory Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for large model training with high memory
    config = AutoDistConfig(
        task_name='large_model_training',
        memory_constraint=80,  # 80GB A100
        recompute_modules='transformer.layer',  # Selective recomputation
        zero_stage=1,  # Enable ZeRO stage 1
        zero_ngroups=4,  # Use 4 ZeRO groups
        opt_resident_coef=6,  # FP16 training with Adam
        opt_transient_coef=2,
        topk=50  # More plan options
    )

Pipeline Parallelism
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for pipeline parallelism
    config = AutoDistConfig(
        task_name='pipeline_training',
        pipeline_pivots='encoder,decoder',
        pipeline_nstages=4,
        pipeline_scheduler='1f1b',
        max_pipeline_bubble_ratio=0.1,  # Strict bubble control
        mesh_row=2,  # 2 nodes
        mesh_col=4,  # 4 GPUs per node
        micro_batch_size=2,
        update_freq=4  # Effective batch size = 2 * 4 = 8
    )

Memory-Efficient Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuration for memory-efficient training
    config = AutoDistConfig(
        task_name='efficient_training',
        is_train=True,
        consider_mem=True,
        memory_constraint=24,  # 24GB RTX 4090
        recompute_modules='attention,mlp',  # Selective recomputation
        solver='ilp',  # More precise optimization
        topk=10
    )

Best Practices
--------------

1. **Start Simple**: Begin with default settings and gradually tune parameters based on your needs.

2. **Memory Tuning**: 
   - Consider ``recompute_modules`` for memory savings, but note that more aggressive recomputation (like ``'ROOT'``) doesn't always provide maximum memory savings
   - Adjust ``memory_constraint`` based on your hardware
   - Fine-tune optimizer coefficients based on your training setup
   - Experiment with different recomputation granularities to find the optimal memory-computation trade-off

3. **Pipeline Parallelism**:
   - Choose ``pipeline_pivots`` at natural module boundaries
   - Start with ``pipeline_nstages='auto'`` to find optimal stages
   - Monitor bubble ratio and adjust ``max_pipeline_bubble_ratio``

4. **Profiling**:
   - Enable ``parallel_profile`` for faster profiling
   - Set ``re_profile=True`` when changing hardware or model architecture
   - Use appropriate ``profile_dir`` for different experiments

5. **Plan Management**:
   - Save successful plans with ``save_plan_path`` for reuse
   - Use descriptive ``task_name`` for better organization

Troubleshooting
---------------

**Out of Memory Errors**
    - Reduce ``memory_constraint``
    - Experiment with different ``recompute_modules`` strategies (selective recomputation may be more effective than ``'ROOT'``)
    - Increase ``zero_ngroups`` or enable higher ZeRO stages
    - Reduce ``transient_mem_coef``

**Slow Plan Generation**
    - Reduce ``topk`` for faster search
    - Use ``'dp'`` solver instead of ``'ilp'``
    - Set ``parallel_profile=True``
    - Increase ``ignore_small_tensor_threshold``

**Poor Performance**
    - Check ``max_pipeline_bubble_ratio`` if using pipeline parallelism
    - Verify ``mesh_row`` and ``mesh_col`` match your hardware
    - Tune ``micro_batch_size`` and ``update_freq``
    - Consider different ``recompute_modules`` strategies
