.. nnScaler documentation master file, created by
   sphinx-quickstart on Fri Apr 19 15:38:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

########
Overview
########

Welcome to nnScaler's documentation!
====================================

Project Website: https://github.com/microsoft/nnscaler

What is nnScaler?
-----------------

nnScaler is a parallelization engine that compiles a Deep neural network (DNN) model that designed for single-GPU execution into a program that capable of running in parallel across multiple GPUs.

.. image:: ./images/nnScaler_flow.png

System Highlights
-----------------

* Ease of Use: Enable parallelization with just a few lines of code, producing a Pythonic parallel program easy for further development.
* Extensibility: Seamlessly integrates new operators to support emerging models through nnScaler's exposed API.
* Reliability: Verified through extensive end-to-end training sessions, nnScaler is a dependable system.
* Performance: By exploring a larger parallelization space, nnScaler can significantly enhance parallel training performance.

``nnScaler`` allows **DNN scientist** to concentrate on model design with PyTorch on single GPU, while leaving parallelization complexities to the system. It introduces innovative parallelism techniques that surpass existing methods in performance. Additionally, nnScaler supports the extension of DNN modules with new structures or execution patterns, enabling users to parallelize custom DNN models.

``nnScaler`` helps **DNN system experts** to explore new DNN parallelization mechanisms and policies for emerging models. By providing user-defined functions for new operators not recognized by nnScaler, it ensures seamless parallelization of novel DNN models, such as facilitate long sequence support in LLMs.


Success Stories
---------------

nnScaler has been adopted by multiple projects, including both product and research explorations:
   * `(YOCO)You only cache once: Decoder-decoder architectures for language models <https://arxiv.org/abs/2405.05254>`_
   * `LongRoPE: Extending LLM context window beyond 2 million tokens <https://arxiv.org/abs/2402.13753>`_
   * Post training for the long context version of `Phi-3 series <https://arxiv.org/abs/2404.14219>`_ SLMs


Get Started
===========

* :doc:`quickstart`
* :doc:`examples/llama3_demo`
* :doc:`examples/llama`
* :doc:`examples/nanogpt`


Reference
---------

Please cite nnScaler in your publications if it helps your research::

    @inproceedings{lin2024nnscaler,
    title = {nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training},
    author={Lin, Zhiqi and Miao, Youshan and Zhang, Quanlu and Yang, Fan and Zhu, Yi and Li, Cheng and Maleki, Saeed and Cao, Xu and Shang, Ning and Yang, Yilei and Xu, Weijiang and Yang, Mao and Zhang, Lintao and Zhou, Lidong},
    booktitle={18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
    pages={347--363},
    year={2024}
    }

You may find the Artifact Evaluation for OSDI'24 with the guidance `here <https://github.com/microsoft/nnscaler/tree/osdi24ae>`_.

Contributing
------------

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_. For more information, see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or contact opencode@microsoft.com with any additional questions or comments.

Trademarks
----------

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow `Microsoft's Trademark & Brand Guidelines <https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general>`_. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-party's policies.

Contact
-------

You may find our public repo from https://github.com/microsoft/nnscaler or microsoft internal repo https://aka.ms/ms-nnscaler.
For any questions or inquiries, please contact us at nnscaler@service.microsoft.com.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   self
   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   examples/llama3_demo
   examples/llama
   examples/dagan
   examples/vit
   examples/deepseek
   examples/nanogpt

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Basic Usages

   trainer
   pytorch_lightning
   register_custom_op

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced Usages

   parallel_module
   dimops
   verify_op

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Miscellaneous

   control_flow
   faq
   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
