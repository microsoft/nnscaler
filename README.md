# nnScaler

### What is nnScaler?
The nnScaler is a system that takes a DNN model that designed for single device, e.g., GPU, and automatically convert it into the program that can execute concurrently on multiple devices. 

###	What can nnScaler do? 
Under the hood, nnScaler analyzes the given DNN models, plans for appropriate parallelization strategies, and generates corresponding execution code. With nnScaler, users can focus on single device DNN model design, and offload the complex parallelization work to nnScaler, and easily achieve high performance parallel DNN execution.
###	What is/are nnScaler’s intended use(s)?
Due to high compatibility and extensibility, nnScaler can be used for innovation of a wide range of new DNN models and DNN systems, including new model structure, training patterns, as well as new parallelization techniques that are beyond existing data-parallelism, tensor-parallelism or pipeline parallelism.
###	How was nnScaler evaluated? What metrics are used to measure performance?
For execution performance, nnScaler can support new parallelisms that outperform existing parallel execution approaches: 1) fitting larger DNN model given the same hardware; 2) providing faster execution for the same model on the same hardware. (included in our OSDI’24 paper)
For compatibility, nnScaler can support paralleling new DNN models by providing user-defined functions (a few lines of code) for the new operators unrecognized by the nnScaler.
###	What are the limitations of nnScaler? How can users minimize the impact of nnScaler’s limitations when using the system?
- It is possible certain DNN model architecture or execution patterns may violate the assumption of nnScaler, therefore cannot supported by nnScaler.
- The nnScaler does not guarantee the optimality of parallelization, so it is possible for nnScaler to miss the optimal parallelization strategy given DNN model and device settings, while only providing suboptimal solutions.
-	Even though we tried our best to make the parallelization process correct, it is possible for nnScaler that generates parallelized programs for concurrent execution inconsistent with the original DNN model for single device.
###	What operational factors and settings allow for effective and responsible use of nnScaler?
-	We provide documentation to guide users in the usage of the nnScaler.
-	We provide parallelization examples that users can directly leverage for parallel execution if they intend to execute the same DNN models.
-	We also provide certain cases of customization, including reconfiguring the device settings, adopting new DNN models in nnScaler, and supporting customized operators.

## Extending nnScaler

###	What are plugins and how does nnScaler use them?  
The nnScaler supports extending DNN modules with new structure or execution pattern, which enable users to parallelize their own new DNN models. Then during parallelization, nnScaler will process the new modules like the ones that already supported by it.
###	What can nnScaler provide to plug ins? 
The nnScaler provides an easy-to-use interface so users can conveniently realize the plug-in DNN module by implementing a few user-defined-functions. 
###	What kinds of issues may arise when using nnScaler enabled with plugins?  
-	When paralleling new DNN models, users may try on some structures or execution patterns that violate the assumption and fail to support.
-	When adapting new DNN models for parallelization, users may incorrectly implement the user-defined function, causing nnScaler producing incorrect parallelized programs.
-	Certain unforeseen mistakes in nnScaler implementation may cause producing incorrect parallelized programs without warning, leading to incorrect execution.
-	To mitigate unsupported issues, users may disable parallelization for the entire DNN model or certain parts of the model as an workaround.
-	To mitigate incorrect execution, users may compare the parallelized programs and original DNN model execution on small datasets to confirm their consistency before deploying to large scale for long-term execution.

## Reference

Please cite nnScaler in your publications if it helps your research:

```
@inproceedings {nnscaler-osdi24,
author = {Zhiqi Lin and Youshan Miao and Quanlu Zhang and Fan Yang and Yi Zhu and Cheng Li and Saeed Maleki and  Xu Cao and Ning Shang and Yilei Yang and Weijiang Xu and Mao Yang and Lintao Zhang and Lidong Zhou},
title = {nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training},
booktitle = {18th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 24)},
year = {2024},
publisher = {{USENIX} Association},
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
