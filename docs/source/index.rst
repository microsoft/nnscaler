.. nnScaler documentation master file, created by
   sphinx-quickstart on Fri Apr 19 15:38:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

########
Overview
########

Welcome to nnScaler's documentation!
====================================

What is nnScaler?
-----------------

nnScaler is a parallelization engine that compiles a Deep neural network (DNN) model that designed for single-GPU execution into a program that capable of running in parallel across multiple GPUs.

.. image:: ./images/nnScaler_flow.png

System Highlights
-----------------

* Ease of Use: Only a few lines of code need to be changed to enable automated parallelization.
* Pythonic: The parallelization output is in PyTorch code, making it easy for users to understand and convenient for further development or customization.
* Extensibility: nnScaler exposes an API to support new operators for emerging models.
* Reliability: Verified through various end-to-end training sessions, nnScaler is a dependable system.
* Performance: By exploring a large parallelization space, nnScaler can significantly enhance parallel training performance.

For **DNN scientists**, they can concentrate on model design with PyTorch on single GPU, while leaving parallelization complexities to nnScaler. It introduces innovative parallelism techniques that surpass existing methods in performance. Additionally, nnScaler supports the extension of DNN modules with new structures or execution patterns, enabling users to parallelize their custom DNN models.

For **DNN system experts**, they can leverage nnScaler to explore new DNN parallelization mechanisms and policies for emerging models. By providing user-defined functions for new operators not recognized by nnScaler, it ensures seamless parallelization of novel DNN models. For example, to facilitate long sequence support in LLMs.

Get Started
===========

* :doc:`quickstart`
* :doc:`llama3_demo_example`
* :doc:`llama3_8b_128k_example`
* :doc:`nanogpt_example`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started

   self
   quickstart

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   llama3_demo_example
   llama3_8b_128k_example
   nanogpt_example

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: References

   trainer
   pytorch_lightning
   register_custom_op

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Miscellaneous

   install_from_source
   faq
   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
