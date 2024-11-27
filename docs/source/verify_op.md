# Verify-graph support
Used to verify operations in IRGraph to ensure their functionality and consistency across single and multiple GPUs.

Command-line interface for verifying operations in an IRGraph.

Usage:
```
python verify_graph_operations.py --graph <path_to_graph_ckp> --outdir <path_to_output_directory>
```

Parameters:

    --graph (str): Path to the graph checkpoint file (.ckp) to be loaded. This is the same graph used as the input for the pas policy.
    --outdir (str): Directory where verification results will be saved.


This script performs the following steps:
1.  Load the IRGraph: Reads the graph checkpoint file specified by the `--graph` argument.
2.  Verify Operations: Performs verification on the operations defined in the graph.  This includes:
    - Registering the operations for further testing.
    - Verifying single-GPU and multi-GPU functionality.
    - Checking the consistency of partitioned operations across different GPUs.
3.  Generate and Save Results: Outputs verification results, including loss values for single and multiple GPUs, and details of partition validations.

To test a module: you should first use parallelize to generate the required graph.ckp file, then test graph against the current script.

## Verify-dimops support
Define a configuration for verifying partition options of a tensor operation.
This configuration helps ensure that the operation's partitioning logic is valid
by specifying the function signature, arguments, expected outputs, and partitioning options.

## Example of Conv2D
This is used to verify that Conv2D's partition configuration is correct. This configuration defines a basic Conv2D operation with input Tensor, convolution kernel, and bias.
```python
@dataclass
class VerifyConfig:
    fsig: str
    args: List[TensorInfo]
    kwargs: Dict[str, Any]
    noutputs: int
    parti_options: List[Dict[str, int]]
    import_customized_func: str = ""
    non_grad_indices: List[int] = field(default_factory=list)

Parameters:
    fsig (str): Function signature of the operator to be tested.
    args (List[TensorInfo]): List of TensorInfo objects representing the input arguments for the operator.
    kwargs (Dict[str, Any]): Keyword arguments for the operator.
    noutputs (int): Number of outputs expected from the operator.
    parti_options (List[Dict[str, int]]): List of partition options specifying how to partition the operator.
    import_customized_func (str): A string containing import statements for any custom functions or modules required by the operator. This ensures that all necessary functions are available in the generated test code.
    non_grad_indices (List[int]): List of indices specifying which input tensors are buffer parameters 
                                    (e.g., running_mean, running_var) that should not participate in the 
                                    backward pass. These parameters will be detached and have their gradients 
                                    disabled during the test.
    
conv2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv2d',
    fsig = 'torch.conv2d',
    args = [
        TensorInfo('shape', (8, 32, 32)), 
        TensorInfo('shape', (16, 4, 3, 3)), 
        TensorInfo('shape', (16,))
    ],
    kwargs = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 2},
    parti_options = [{'idx': 0, 'dim': 0}],
    noutputs = 1,
)
verify_partition_options(conv2d_config)
```

## Examples for more operators 

```
dropout_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import dropout',
    fsig = 'torch.nn.functional.dropout',
    args = [
        TensorInfo('shape', (1, 512, 4, 4))
    ],
    kwargs = {'p': 0.5, 'training':False, 'inplace':False},
    parti_options = [{'idx': 0, 'dim': 1},
                     {'idx': 0, 'dim': 2},
                     {'idx': 0, 'dim': 3}],
    noutputs = 1,
)
verify_partition_options(dropout_config)


where_config = VerifyConfig(
    fsig='torch.where',
    args=[
        TensorInfo('shape', value=(1, 1, 9, 9)), 
        TensorInfo('shape', value=(1, 12, 9, 9)), 
        TensorInfo('shape', value=(1,)) 
    ],
    kwargs={},
    noutputs=1,
    parti_options=[{'idx': 1, 'dim': 1}], 
)
verify_partition_options(where_config)


view_config = VerifyConfig(
    fsig='torch.Tensor.view',
    args=[
        TensorInfo('shape', value=(1, 9, 768))
    ],
    kwargs={'size': (-1, 768)},
    noutputs=1,
    parti_options=[{'idx': 0, 'dim': 2}],
)
verify_partition_options(view_config)


embedding_config = VerifyConfig(
    import_customized_func='from torch.nn.functional import embedding',
    fsig='nnscaler.runtime.function.embedding',
    args=[
        TensorInfo('shape', value=(1, 9)),
        TensorInfo('shape', value=(50257, 768))
    ],
    kwargs={'padding_idx': None, 'start': 0, 'stop': 50257},
    parti_options=[{'idx': 1, 'dim': 1}], 
    noutputs=1
)
verify_partition_options(embedding_config)


dropout_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import dropout',
    fsig = 'torch.nn.functional.dropout',
    args = [
        TensorInfo('shape', (1, 9, 768))
    ],
    kwargs = {'p': 0.1, 'training':False, 'inplace':False},
    parti_options = [{'idx': 0, 'dim': 2}],
    noutputs = 1,
)
verify_partition_options(dropout_config)


fullslice_config = VerifyConfig(
    fsig='nnscaler.runtime.function.fullslice',
    args=[
        TensorInfo('shape', value=(1, 1, 1024, 1024)),
        TensorInfo('value', value=slice(None, None, None)),
        TensorInfo('value', value=slice(None, None, None)),
        TensorInfo('value', value=slice(IRObject('sub391', value=0, is_constant=True), IRObject('size_9388', value=9, is_constant=True), None)),
        TensorInfo('value', value=slice(None, IRObject('size_9388', value=9, is_constant=True), None))
    ],
    kwargs={},
    noutputs=1,
    parti_options=[],
)
verify_partition_options(fullslice_config)


conv2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv2d',
    fsig = 'torch.conv2d',
    args = [TensorInfo('shape',(8192, 4, 4)), TensorInfo('shape', (8192, 512, 3, 3)),TensorInfo('shape', (8192,))],
    kwargs = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 16},
    parti_options = [{'idx': 0, 'dim': 0}],
    noutputs = 1,
)
verify_partition_options(conv2d_config)
conv2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv2d',
    fsig = 'torch.conv2d',
    args = [
        TensorInfo('shape', (4, 8, 32, 32)), 
        TensorInfo('shape', (16, 4, 3, 3)), 
        TensorInfo('shape', (16,))
    ],
    kwargs = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 2},
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(conv2d_config)
conv2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv2d',
    fsig = 'torch.conv2d',
    args = [TensorInfo('shape',(1, 8192, 4, 4)), TensorInfo('shape', (8192, 512, 3, 3)),TensorInfo('shape', (8192,))],
    kwargs = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 16},
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(conv2d_config)


conv_transpose2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv_transpose2d',
    fsig = 'torch.conv_transpose2d',
    args = [TensorInfo('shape',(8192, 4, 4)), TensorInfo('shape', (8192, 512, 3, 3))],
    kwargs = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 16},
    parti_options = [{'idx': 0, 'dim': 0}],
    noutputs = 1,
)
verify_partition_options(conv_transpose2d_config)
conv_transpose2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv_transpose2d',
    fsig = 'torch.conv_transpose2d',
    args = [
        TensorInfo('shape', (512, 4, 4)), 
        TensorInfo('shape', (512, 512, 3, 3))  
    ],
    kwargs = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1}, 
    parti_options = [{'idx': 0, 'dim': 0}],
    noutputs = 1,
)

verify_partition_options(conv_transpose2d_config)
conv_transpose2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv_transpose2d',
    fsig = 'torch.conv_transpose2d',
    args = [TensorInfo('shape',(1, 8192, 4, 4)), TensorInfo('shape', (8192, 512, 3, 3))],
    kwargs = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 16},
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(conv_transpose2d_config)
conv_transpose2d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv_transpose2d',
    fsig = 'torch.conv_transpose2d',
    args = [
        TensorInfo('shape', (1, 512, 4, 4)), 
        TensorInfo('shape', (512, 512, 3, 3))  
    ],
    kwargs = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1}, 
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(conv_transpose2d_config)


conv1d_config = VerifyConfig(
    # import_customized_func = 'from torch.nn.functional import conv1d',
    fsig = 'torch.conv1d',
    args = [
        TensorInfo('shape',(1, 512, 400)), 
        TensorInfo('shape', (128, 512, 3))
    ],
    kwargs = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1},
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1, 
)
verify_partition_options(conv1d_config)
conv1d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv1d',
    fsig = 'torch.conv1d',
    args = [
        TensorInfo('shape', (1, 8192, 400)),  
        TensorInfo('shape', (128, 512, 3))    
    ],
    kwargs = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 16},
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(conv1d_config)


pose1d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv_transpose1d',
    fsig = 'torch.conv_transpose1d',
    args = [
        TensorInfo('shape', (1, 512, 100)),
        TensorInfo('shape', (512, 256, 3))
    ],
    kwargs = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 8}, 
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(pose1d_config)
pose1d_config = VerifyConfig(
    import_customized_func = 'from torch.nn.functional import conv_transpose1d',
    fsig = 'torch.conv_transpose1d',
    args = [
        TensorInfo('shape', (1, 512, 100)),
        TensorInfo('shape', (512, 256, 3)) 
    ],
    kwargs = {'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1, 'groups': 1}, 
    parti_options = [{'idx': 0, 'dim': 1}],
    noutputs = 1,
)
verify_partition_options(pose1d_config)


verify_config = VerifyConfig(
    fsig='nnscaler.graph.function.wrapnn.wrap_batchnorm2d_func',
    args=[
        TensorInfo('shape', value=(32, 64, 8, 8)),  
        TensorInfo('shape', value=(64,)),  
        TensorInfo('shape', value=(64,)), 
        TensorInfo('shape', value=(64,)), 
        TensorInfo('shape', value=(64,)), 
        TensorInfo('shape', value=(1,)),
    ],
    kwargs={
        'momentum': 0.1,
        'training': True,
        'track_running_stats': True,
        'eps': 1e-05
    },
    noutputs=1,
    parti_options=[{'idx': 0, 'dim': 0},
                   {'idx': 0, 'dim': 1}],
    non_grad_indices=[3, 4, 5]
)
verify_partition_options(verify_config)


addmm_config = VerifyConfig(
    import_customized_func='from torch import addmm',
    fsig='torch.addmm',
    args=[
        TensorInfo('shape', (2, 3)), 
        TensorInfo('shape', (2, 3)),  
        TensorInfo('shape', (3, 3))   
    ],
    kwargs={},
    parti_options=[{'idx': 0, 'dim': 0}, {'idx': 1, 'dim': 0}],
    noutputs=1
)

verify_partition_options(addmm_config)


verify_config = VerifyConfig(
    fsig='nnscaler.graph.function.wrapnn.wrap_instancenorm2d_func', 
    args=[
        TensorInfo('shape', value=(32, 64, 8, 8)),
        TensorInfo('shape', value=(64,)),
        TensorInfo('shape', value=(64,)),
        TensorInfo('shape', value=(64,)),
        TensorInfo('shape', value=(64,)),
    ],
    kwargs={
        'training': True, 
        'momentum':0.1,
        'eps': 1e-05
    },
    noutputs=1,
    parti_options=[{'idx': 0, 'dim': 0}],
    non_grad_indices=[3, 4]
)
verify_partition_options(verify_config)
```
