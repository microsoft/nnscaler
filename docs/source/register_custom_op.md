# Register a new operator/function

## Overview

During iterating the model, users may encounter the situation that some operator is failed to be traced by nnscaler concrete tracer. In this case, users can register this operator to nnscaler, then nnscaler will treat it as one simple operator instead of tracing into the sub-operators of this operator. The registration also tells nnscaler the feasible partition options of this operator.

Note, the registration only works for function while does not work for PyTorch Module, because nnscaler does not allow weight tensors to be managed by the registered operator. If you are dealing with a PyTorch Module, you can consider its underlying PyTorch function instead.

Taking `torch.nn.InstanceNorm2d` (or actually `torch.nn.functional.instance_norm`) as an example. Currently nnscaler does not support partitioning of this operator. If you use this operator in your model, you will see a warning message "Find unknown pytorch operation: torch.xxx.xxx" or "Set python runtime function: xxx". Then you can register this operator into nnscaler and specify its partition options as follows:

```python
import torch
import torch.nn as nn
import nnscaler
# you write a new function to wrap the operator.
# suggest to make all the argument of this function torch.Tensor,
# and *REMEMBER* to add type annotation for each input argument.

# the first argument of register is the annotation to indicate how this function can be partitioned,
# very similar to einsum expression. '^' means the corresponding dimension cannot be partitioned.
@nnscaler.register_op('n c h^ w^, c, c -> n c h^ w^', name='my_instance_norm')
def my_instance_norm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return nn.functional.instance_norm(input, weight=weight, bias=bias)
```

Here is another example to support a custom matmul operator:
```python
# file: custom_ops.py
def operator(x: torch.Tensor, w: torch.Tensor, h: float) -> torch.Tensor:
    out = torch.matmul(x, w)
    out = out.view(h, out.size(0) // h, out.size(1))
    return out

# file: main.py
import nnscaler
from custom_ops import operator
nnscaler.register_op('(h^ m^) kd+, kd+ n -> h^ m^ n', name='matmul_custom')(operator)
```


## Api Explains

```python
def register_op(
    annotation: Union[str, Callable],
    name: Optional[str] = None,
    code_impl_pattern: str = 'import',
    emit_fn: Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str] = None,
    transform_rules: Tuple[TransformRule] = None,
    input_gen_fn: Callable[IRFwOperation, List[torch.Tensor]] = None) -> Callable:
) -> Callable:
    ...
```

Register a function with dimension annotations.

This function is cooperated with nnscaler tracer. Users can only register global functions(which are defined in a module level, instead of ones defined inside a function / class or __main__ scope).

The annotation (`annotation`) specifies the number of inputs as `*args`,
and treat all the rest inputs as `**kwargs`.

For tensor-type inputs, the annotation should be a string of identifiers separated by space, e.g., `'a b'`;
For non-tensor-type inputs, the annotation should be specified '?'.

This function can be used as a decorator or a function.
Here are several Examples:

```python
import nnscaler
from third_party import func

nnscaler.register_op('a (b c) -> (a b) c')(func)
```

or,

```python
import nnscaler
from third_party import func

@nnscaler.register_op('a (b c) -> (a b) c')
def func(x, b = 4):
    ...
```

or,

```python
import nnscaler
from third_party import func

def anno_fn(*inputs, **kwargs):
    return 'a (b c) -> (a b) c'

nnscaler.register_op(anno_fn)(func)
```
This function has the following parameters:

- `annotation` (`str | Callable`): operator annotation, it can be:
    - op annotation: e.g., 'a (b c) -> (a b) c'
    - a callable function that generates op annotation (str). The function
    taks inputs and kwargs as arguments and returns the operator annotation.
- `name` (`str | None`): operator name. Only usable when `node_repr` is a string.
- `code_impl_pattern` (`str`): It can only be 'import' or 'source'. If 'import' (default), will generate code with import statement. If 'source', will take the source code directly.
- `emit_fn` (`Callable`): special emit function for codegen, it accepts the node, repred args, repred kwargs, runtime_devid,
    plan_ndevs, runtime_ndevs as input and returns the generated code. Check examples/customized_ops/ring_attention/zigzag_attn.py for more details.
    Default: None.
- `transform_rules` (`Tuple[TransformRule]`): a tuple of special TransformRules which will be used when partitioning the node.
    Default: None.
- `input_gen_fn` (`Callable`): input generator function for profiler, this function accepts the IRFwOperation as input and returns
    the list of input tensors, which is used during operator profiling. kwargs are same as that in the input node. By default, nnScaler's
    profiler will use `torch.rand` for floating point data types and `torch.zeros` for special types like `torch.int64` and `torch.bool`.
    However, input tensors' contents may influence operator's behavior and speed dramatically.
    Take function `nnscaler_moe_gmm` in `examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py` as an example. It dispatches
    tokens (`hidden_states`) to experts according to another input tensor `topk_idx`. In most of the training time, tokens are distributed
    evenly among experts with indices in `[local_expert_start, local_expert_end]`. Since `top_idx`'s type is `torch.int64`, if we generate
    it with `torch.zeros` then all of the tokens are dispatched to the 1st expert, which can be ilegal and far from the real profile statistics
    of the operator. By using `input_gen_fn`, we can provide compatible input tensors to the profiler so that the solver can generate a
    good distributed plan.
    Default: None.


## Dimension Annotion Operations

An operator has (multiple) input tensors and (multiple) output tensors.
Each tensor can be annotated with dimension annotations (DimAnno) using `identifiers`.
The same `identifier` indicates the they have the same real length.

### Dimension Annotation

  e.g., 'a+', 'ab^', 'cd', '(ab+ c^ d)', '64'

A dimension of a tensor can be annotated by {identifier}{reduction} template.

An `identifier` must be one of:
  1) symbolic annotation that must match with the criteria of python str.isidentifier.
  2) numeric string that must match with python str.isdecimal. This indicates the shape is the same value
     numeric string will always have '^' reduction type'

Special identifier:
  1) '*': this special identifier indicates the dimension is dynamic, which will automatically get expanded given the shape
  2) '?': this special identifier indicates the value is can only be replicated, no matter it is a tensor or a non-tensor.

A `reduction` can be a set of {'', '+', '^'}:
  '' indicates this dimension can be partitioned, and each output should have this dimension.
  '+' indicates this dimension can be partitioned, and each output doesn't have this and need to do sum-reduction.
  '^' means this dimension cannot be partitioned.

A dimension can also be annotated with inner-dimensions using brackets, i.e., '(' and ')'.
The value of inner dimension needs to be inferrable, or indicated by function args (of same name).

Please be very careful when you use '?'. If it depends on the tensor input,
then the tensor input should be marked as non-partitionable.

Example 1:
```python
@nnscaler.register_op('a^ b^ -> a^ b^, ?')
def op1(x: torch.Tensor):
    x = ...
    y = some_func(x)
    return x, y
```

Example 2:
```python
@nnscaler.register_op('a b -> a b, ?')
def op1(x: torch.Tensor):
    x = ...
    y = 10
    return x, y
```

In Example 1, as `y` has dependency on `x`, its value will be wrong if we partition `x`.
So `x` should be marked as non-partitionable.

In Example 2, `y` is a constant, and its value is independent of `x`.
So we can mark `x` partitioned.

### Shape Annotation

  e.g., 'a (c+ d^) e'

A shape annotation consists of dimension annotation separated by (multiple) spaces.


### Operator Annotation

  e.g., 'm k+, n k+ -> m n', '4 k+, k+ d -> 8 d', '* d^, s -> * s'

  An operator can be annotated with input shape annotations and output shape annotations.

  '->' seperates the inputs (left) and outputs (right) and ',' separates each input and output tensor.

  Identifiers in output tensor annotation needs to be
  1) apearred in input tensor annotations
  2) using numeric string

### Operator Partitioning Rule:

  1) Spatial Partition (dimension with '' reduce type):
      tensors can be uniformly partitioned on dimensions having spatial reduction type.
      other tensors in the operator that don't have this dimension will be replicated.

  2) Value Partition (dimension with '+' reduce type):
      * tensors can be uniformly partition on dimensions having numerical reduction type
      * other tensors in the the operator that don't have this dimension will be partitioned numerically.

  3) Illegal Splitting (dimension with '^' reduce type):
      * tensors can not be partitioned on dimensions having '^' reduction type.

### Hidden dimension

  Sometimes user need to reshape the tensor by splitting a dimension into multiple dimensions. For example, a tensor of (1024, 8) size needs to be reshaped into the shape of (8, 128, 8):

  ```python
  # annotation: (h t) k -> h t k
  def reshape(tensor: torch.Tensor, h : int = 8) -> torch.Tensor:
      out = tensor.reshape(h, tensor.size(0) // h, tensor.size(-1))
      return out
  ```

  This can be represented by annotating a dimension using brackets `()`. The bracket contains multple identifiers (and their reductions), like `'(h t)'` here for the first dimension of the input tensor. To help system infer the number of `h` and `t` in the annotation, the function requires to put in a same-named argument `h` or `t` (`h=8` here in example).

## Inplace Operators

We assume the module is SSA (static single-assignment), which means you should avoid change the input tensors inplace in your custom operators.

However, if you have to do this, it's your responsibility to make sure the inplace operation is correct. And to help us track the dependencies between tensors, you must return all the input tensors that are changed in the custom operators.

```python
# this is wrong
# because x is changed inplace, but it is not returned.
@nnscaler.register_op('*, * -> *', name='inplace_operator)
def inplace_operator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x.add_(y)
    z = x + y
    return z

# this is correct
@nnscaler.register_op('*, * -> *, *', name='inplace_operator)
def inplace_operator(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x.add_(y)
    z = x + y
    return x, z
```

## Optional Tensor Input

If you have an optional tensor input, you should tell `nnscaler` how this optional tensor will be used.

There are two cases:

1. The optional tensor can only be replicated. In this case, you should use '?' as the identifier.

```python
@nnscaler.register_op('a^ b^, ? -> a^ b^', name='optional_tensor')
def optional_op(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
    out = torch.triu(x)
    if y is not None:
        out += y
    return out
```

2. The optional tensor can be partitioned. In this case, you should use an annotation function to tell `nnscaler` how to partition the optional tensor when it is not None.

```python
def anno_fn(*inputs, **kwargs):
    if inputs[1] is None:
        return '*, ? -> *'
    else:
        return '*, * -> *'

@nnscaler.register_op(anno_fn, name='optional_tensor')
def optional_op(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
    if y is None:
        return x
    return x + y
```

Please note the value of the optional tensor should be consistent in runtime and tracing time. Which mean if the value of the optional tensor is `None` in tracing time, it should always be `None` in runtime, and if the value of optional tensor is not `None` in tracing time, it should always not be `None` in runtime. It may cause runtime error if the consistency is not guaranteed.
