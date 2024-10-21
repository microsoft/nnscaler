Frequent asked questions
------------------------

**What is nnScaler?**

The nnScaler is a system that takes a DNN model that is designed for a single device, e.g., GPU, and automatically converts it into a program that can execute concurrently on multiple devices.

**What can nnScaler do?**

Under the hood, nnScaler analyzes the given DNN models, plans for appropriate parallelization strategies, and generates corresponding execution code. With nnScaler, users can focus on single-device DNN model design, offload the complex parallelization work to nnScaler, and easily achieve high-performance parallel DNN execution.

**What is/are nnScaler's intended use(s)?**

Due to high compatibility and extensibility, nnScaler can be used for the innovation of a wide range of new DNN models and DNN systems, including new model structures, training patterns, as well as new parallelization techniques that go beyond existing data-parallelism, tensor-parallelism, or pipeline parallelism.

**How was nnScaler evaluated? What metrics are used to measure performance?**

For execution performance, nnScaler can support new parallelisms that outperform existing parallel execution approaches:
1. Fitting larger DNN models given the same hardware.
2. Providing faster execution for the same model on the same hardware (included in our OSDI'24 paper).

For compatibility, nnScaler can support paralleling new DNN models by providing user-defined functions (a few lines of code) for the new operators unrecognized by the nnScaler.

**What are the limitations of nnScaler? How can users minimize the impact of nnScaler's limitations when using the system?**

- Certain DNN model architectures or execution patterns may violate the assumptions of nnScaler and, therefore, cannot be supported by nnScaler.
- The nnScaler does not guarantee the optimality of parallelization, so it is possible for nnScaler to miss the optimal parallelization strategy given DNN model and device settings, while only providing suboptimal solutions.
- Despite our best efforts to ensure the parallelization process is correct, it is possible for nnScaler to generate parallelized programs for concurrent execution that are inconsistent with the original DNN model for a single device.

**What operational factors and settings allow for effective and responsible use of nnScaler?**

- We provide documentation to guide users in the usage of the nnScaler.
- We provide parallelization examples that users can directly leverage for parallel execution if they intend to execute the same DNN models.
- We also provide certain cases of customization, including reconfiguring the device settings, adopting new DNN models in nnScaler, and supporting customized operators.

**What are extensions(plugins) in nnScaler and how does nnScaler use them?**

The nnScaler supports the extension with customized parallelization of DNN modules, allowing new DNN models to be parallelized. During this process, nnScaler will handle the new modules in the same way as those it already supports.

**What can nnScaler provide to extensions(plugins)?**

The nnScaler provides an easy-to-use interface so users can conveniently realize customized parallelization of certain DNN modules by only implementing a few user-defined functions.

**What kinds of issues may arise when using nnScaler enabled with extensions(plugins)?**

- When paralleling new DNN models, users may try some structures or execution patterns that violate the assumptions and fail to support.
- When adapting new DNN models for parallelization, users may incorrectly implement the user-defined function, causing nnScaler to produce incorrect parallelized programs.
- Certain unforeseen mistakes in nnScaler implementation may cause it to produce incorrect parallelized programs without warning, leading to incorrect execution.
- To mitigate unsupported issues, users may disable parallelization for the entire DNN model or certain parts of the model as a workaround.
- To mitigate incorrect execution, users may compare the parallelized programs and original DNN model execution on small datasets to confirm their consistency before deploying to large scale for long-term execution.