Frequent Asked Questions
------------------------

* **What is nnScaler?**

nnScaler is a system that converts a Deep Neural Network (DNN) model designed for a single device (e.g., GPU) into a program capable of executing concurrently on multiple devices.

* **What can nnScaler do?**

Under the hood, nnScaler analyzes the given DNN models, plans appropriate parallelization strategies, and generates the corresponding execution code. This allows users to focus on single-device DNN model design while nnScaler handles the complex parallelization work, enabling high-performance parallel DNN execution with ease.

* **What is/are nnScaler's intended use(s)?**

Thanks to its high compatibility and extensibility, nnScaler can innovate a wide range of new DNN models and systems. This includes supporting new model structures, training patterns, and parallelization techniques that go beyond existing data-parallelism, tensor-parallelism, and pipeline parallelism.

* **How was nnScaler evaluated? What metrics are used to measure performance?**

For execution performance, nnScaler supports new parallelisms that outperform existing parallel execution approaches:

 1. Fitting larger DNN models on the same hardware.
 2. Providing faster execution for the same model on the same hardware (as detailed in our OSDI'24 paper).

For compatibility, nnScaler supports paralleling new DNN models by allowing user-defined functions (a few lines of code) for operators not recognized by nnScaler.

* **What are the limitations of nnScaler? How can users minimize the impact of nnScaler's limitations when using the system?**

- Certain DNN model architectures or execution patterns may violate nnScaler's assumptions and cannot be supported.
- nnScaler does not guarantee optimal parallelization. It may miss the best strategy, providing suboptimal solutions.
- Despite efforts to ensure correctness, nnScaler might generate parallelized programs that are inconsistent with the original single-device DNN model.

* **License**

- Please visit our `License Information <https://github.com/microsoft/nnscaler/blob/main/LICENSE>`_ for details.

* **Security**

- Please visit our `Security Information <https://github.com/microsoft/nnscaler/blob/main/SECURITY.md>`_ for details.