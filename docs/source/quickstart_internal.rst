###########
Get Started
###########

The nnScaler internal repo: https://msrasrg.visualstudio.com/SuperScaler/_git/MagicCube

If you do not have access, please contact nnscaler@service.microsoft.com

************
Installation
************

To get started, install the latest wheel from
`DevOps Artifacts <https://msrasrg.visualstudio.com/SuperScaler/_artifacts/feed/nightly/PyPI/nnscaler/overview/>`_.

If you are familiar with Azure stuffs, you can follow DevOps' guide to set up the repository.

Or if you prefer the simpler way, download the ``.whl`` file in the "Files" section of the website,
and install it locally:

::

    python -m pip install nnscaler-*.whl

**********
Quickstart
**********

The next step depends on your choice of the training framework.

- **No framework**: if you write your own training code and do not use a framework,
  see :ref:`Parallelize API` section.
- **Fairseq**: if you use fairseq, see :ref:`Fairseq` section.
- **Lightning**: TODO

.. _Parallelize API:

Parallelize API
===============

TODO: write a hello world example, assigned to Zhe Liu

If you write your own training code, you can use the *parallelize* API to make your model parallel:

.. code-block:: python

    import torch
    from nnscaler import parallelize, ComputeConfig, build_optimizer

    class LLM(torch.nn.Module):
        def __init__(self, ...):
            ...
        def forward(self, x):
            ...

    llm_sample_input = ...              # dummpy input will be used to do tracing
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

Example
-------

An example of the parallelize API is provided in the repo:
`train.py <https://msrasrg.visualstudio.com/SuperScaler/_git/MagicCube?path=/examples/mlp/train.py>`_

You can download and try it:  ::

    torchrun --nproc_per_node=4 --nnodes=1 train.py

Documentation
-------------

If the example works for you, you can now follow the documentation to parallelize your model:
:doc:`parallel_module`

.. _Fairseq:

Fairseq (To be retired)
=======

.. TODO:

    nnScaler provides `fairseq integration <https://msrasrg.visualstudio.com/SuperScaler/_git/Fairseq>`_.

    TODO: refine the example (and its doc), assigned to Youshan Miao

    TODO (long term): write an example using unmodified fairseq

    Installation
    ------------

    To use fairseq, clone the fork and install it:  ::

        python -m pip uninstall fairseq

        git clone https://msrasrg.visualstudio.com/SuperScaler/_git/Fairseq
        cd Fairseq
        python -m pip install -e .

    Example
    -------

    Follow the example
    `here <https://msrasrg.visualstudio.com/SuperScaler/_git/Fairseq?path=/nnscaler_examples/finetune_hf_model/Quickstart.md>`_.

