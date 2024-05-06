# Dimop Tutorial

## Dimop: Dimension-annotated Operator 

### Annotation for Shape Inference and Transformation

SuperScaler uses annotation to represent an operator (Dimop).
The goal of annotation is for 1). shape inference and 2) transformation plan.

To annotate an operator, following example shows the annotation of matrix multiplication. An operator has inputs and outputs. The inputs can be tensors or non-tensors, while outputs are usually tensors.

```py
# annotation: m^ kd+, kd+ n -> m^ n
def operator(x: torch.Tensor, w: torch.Tensor, h: float) -> torch.Tensor:
    out = torch.matmul(x, w)
    return out
```

To separate inputs and outputs of an operator, `'->'` is a separation keyword where its left part are inputs and right part are outputs. Inside inputs and outputs region, annotation of each tensor is further separated by `','`. 

Every dimension of a tensor is annotated by a template of **{identifiers}{reduction}**, like `'m^ kd+'`, `'kd+ n'`, `'m^ n'`, where `m`, `kd` and `n` are identitifiers, `'^'` and `'+'` are reductions.

If a tensor is represented as `'m^ kd+'`, it indicates the tensor has two dimensions, the first dimension is `m` and the second dimension is `kd`. Dimensions need to be separated by space `' '`. 

* Identifiers

  Identifiers are served for shape inference. Same identifier of different tensors indicates they have same length of their dimension.

  An `identifier` must be one of:

    1) symbolic annotation that must match with the criterion of python `str.isidentifier`.

    2) numeric string that must match with python str.isdecimal. This indicates the shape is the same value. Numeric string will always have '^' reduction type

  Special identifier:

    1) `'*'`: this special identifier indicates the dimension is dynamic, which will automatically get expanded given the shape. If there are multiple `*` for different tensors, then they must have same shape for the expanded dimensions,
    
        e.g., `'* t -> a * t'` can be expanded into `'b c t -> a b c t'`

    2) `'?'`: this special identifier indicates the value is not a tensor, which will be ignored

  To infer the output shape, the identifiers in output tensors must be 1) appear in inputs or 2) numeric string.

* Reductions

  Reductions are served for transformation plans. The reduction can be one of {`''`, `'+'`, `'^'`}:
  
    * `''` (empty) indicates this dimension can be spatially partitioned, and each output that have this identifier will also be spatially partitioned.

    * `'+'` indicates this dimension can be spatially partitioned. And each output that doesn't have this identifier will be numerically partitioned (sum-reduction required).

    * `'^'` means this dimension cannot be partitioned.

### Advance

* Hidden dimension

  Sometimes user need to reshape the tensor by splitting a dimension into multiple dimensions. For example, a tensor of (1024, 8) size needs to be reshaped into the shape of (8, 128, 8):

  ```py
  # annotation: (h t) k -> h t k
  def reshape(tensor: torch.Tensor, h : int = 8) -> torch.Tensor:
      out = tensor.reshape(h, tensor.size(0) // h, tensor.size(-1))
      return out
  ```

  This can be represented by annotating a dimension using brackets `()`. The bracket contains multple identifiers (and their reductions), like `'(h t)'` here for the first dimension of the input tensor. To help system infer the number of `h` and `t` in the annotation, the function requires to put in a same-named argument `h` or `t` (`h=8` here in example).


## Register Python Functions as Operators

To register a customized "matmul" operator in the runtime, user can simply define a python function and add an decorator on the function with its annotations:

```py
@cube.graph.parser.register('(h^ m^) kd+, kd+ n -> h^ m^ n', name='matmul_custom')
def operator(x: torch.Tensor, w: torch.Tensor, h: float) -> torch.Tensor:
    out = torch.matmul(x, w)
    out = out.view(h, out.size(0) // h, out.size(1))
    return out


class Model(troch.nn.Module):

    ...

    def forward(x, w):
        ...
        out = operator(x, w)  # simply use it
        ...
```

During policy decsion, user can see the operator and its name is 'matmul_custom'. To partition the operator, user can get algorithm of tag `'dim'` and partition the annotated dimension, e.g., `kd+` and `n` of the above example.

```py
def PAS(graph: IRGraph, resource):
    for node in graph.nodes():
        if node.name == 'matmul_custom':
            algo = node.algorithms('dim')
            # partition kd+
            config = dict(idx=0, dim=1, num=resource.ngpus)
            subnodes = graph.partition(node, algo, **config)
            ...
        ...
    ...
    return graph
```

Note: we require user to add type annotation of output and input in the function, to help system understand each identifier number. The non-tensor inputs should be listed at the last and don't need to be represented into annotation.