# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Register cutomized function
"""

from typing import Dict, Callable, Optional, Union
from functools import partial
import inspect
import logging

from cube.graph.function.dimops import IRDimops, OpAnno
from cube.graph.parser.fx.concrete_trace_utils.concrete_tracer import is_autograd_apply
from cube.ir.operator import IRTensor

_logger = logging.getLogger(__name__)


class CustomizedOps:
    """Customized op registry."""

    # signature -> IRDimop creation function
    kOpMap: Dict[str, Callable] = {}
    # singature -> runtime function 
    kOpRuntime: Dict[str, Callable] = {}
    # signature -> runtime function implementation code
    kOpCodeDef: Dict[str, str] = {}

    @staticmethod
    def map(signature: str) -> Callable:
        """Get IRDimop creation function by signature
        
        Args:
            signature (str): operator signature

        Returns:
            Callable: IRDimop creation function
        """
        if signature not in CustomizedOps.kOpMap:
            raise KeyError(f"{signature} is not found in registered ops")
        return partial(CustomizedOps.kOpMap[signature], signature=signature)

    @staticmethod
    def exist(signature: str) -> bool:
        """Check if the signature is registered"""
        return signature in CustomizedOps.kOpMap

    @staticmethod
    def register(signature: str, op_create_fn: Callable, code: str, runtime_fn: Callable):
        """Register an operator

        Args:
            signature (str): operator signature
            op_create_fn (Callable): IRDimops creation function
            code (str): runtime function implementation code
            runtime_fn (Callable): runtime function

        Returns:
            None
        """
        builtins = ['_operator', 'torch', 'cube.runtime.function']
        if any(signature.startswith(builtin) for builtin in builtins):
            raise RuntimeError(f"Cannot register operators with signature starting from any of {builtins}")
        assert signature not in CustomizedOps.kOpMap, f"function {signature} is already registered"
        CustomizedOps.kOpMap[signature] = op_create_fn
        CustomizedOps.kOpRuntime[signature] = runtime_fn
        CustomizedOps.kOpCodeDef[signature] = code


def register(node_repr: Union[str, Callable], name: Optional[str] = None,
             code_impl_pattern: str = 'import') -> Callable:
    """
    Register a function with IRDimops annotations.

    This function is cooperated with IRDimops. Users can only register functions defined under a module, instead of
    ones defined inside a function / class or __main__ scope.

    The annotation (`node_repr`) specifies the number of inputs as *args,
    and treat all the rest inputs as **kwargs. 
    
    For tensor-type inputs, the annotation should be a string of identifiers separated by space, e.g., `'a b'`;
    For non-tensor-type inputs, the annotation should be specified '?'.

    Examples:

    ```python
    import cube
    from third_party import func

    cube.graph.parser.register('a (b c) -> (a b) c')(func)
    ```

    or,

    ```python
    import cube
    from third_party import func

    @cube.graph.parser.register('a (b c) -> (a b) c')
    def func(x, b = 4):
        xxx
    ```
    
    or,

    ```python
    import cube
    from third_party import func

    def anno_fn(*inputs, **kwargs):
        return 'a (b c) -> (a b) c'

    cube.graph.parser.register(anno_fn)(func)
    ```

    Args:
        node_repr (str | Callable): operator annotation of IRDimops or callable function that generates IRFwOperation.
            - op annotation: e.g., 'a (b c) -> (a b) c'
            - a callable function that generates op annotation (str). The function
            taks inputs and kwargs as arguments and returns the operator annotation.
        name (str | None): operator name. Only usable when node_repr is a string.
        code_impl_pattern (str):
            can only be 'import' or 'source'. If 'import', will generate code with
            import statement. If 'source', will take the source code directly.
            Default: 'import'.

    Returns:
        fn (Callable): the runtime function
    """

    def decorator(fn: Callable):
        nonlocal code_impl_pattern

        if not callable(fn):
            raise TypeError("Expected a runtime function")

        # step 1. get function signature and inputs
        def get_import_path(fn: Callable) -> str:
            if is_autograd_apply(fn):
                import_path = inspect.getmodule(fn.__self__).__name__
            else:
                import_path = inspect.getmodule(fn).__name__
            return import_path

        import_path = get_import_path(fn)
        if import_path == '__main__':
            raise NotImplementedError(
                f"Cannot register function {fsig} in __main__ module. "
                f"Try to define it in another module and import into main")

        if is_autograd_apply(fn):
            fsig = f'{import_path}.{fn.__self__.__name__}.apply'
            op_name = name if name is not None else fn.__self__.__name__
            args = inspect.signature(fn.__self__.forward)
            arg_names = list(args.parameters.keys())[1:]
        else:
            fsig = f'{import_path}.{fn.__name__}'
            op_name = name if name is not None else fn.__name__
            args = inspect.signature(fn)
            arg_names = list(args.parameters.keys())

        # step 2. get customized op code
        def get_source_code(fn: Callable) -> str:
            if is_autograd_apply(fn):
                code = inspect.getsource(fn.__self__)
                code = code[code.index(f'class {fn.__self__.__name__}'):]
            else:
                code = inspect.getsource(fn)
                code = code[code.index('def'):]
            return code
        
        def get_import_code(fn: Callable) -> str:
            import_path = get_import_path(fn)
            code = f'import {import_path}'
            return code

        if code_impl_pattern == 'import':
            code = get_import_code(fn)
        elif code_impl_pattern == 'source':
            code = get_source_code(fn)
        else:
            raise ValueError(f'code_impl_pattern should be either "import" or "source", got {code_impl_pattern}')

        # step 3. define customized IRDimops creation function
        if not (isinstance(node_repr, str) or callable(node_repr)):
            raise TypeError(f"node_repr should be either str or callable, got {type(node_repr)}")

        def udfop(*args, signature=None, **kwargs):
            anno = node_repr if isinstance(node_repr, str) else node_repr(*args, **kwargs)
            if not isinstance(anno, str):
                raise TypeError(f"node_repr should return a string, but got {type(anno)}: {anno}")
            anno = OpAnno(anno)
            ninputs = len(anno.inputs())
            if len(args) < ninputs:
                raise ValueError(f"calling function {signature} should include at least {ninputs} *args")
            tensors = args[:ninputs]
            for idx, t in enumerate(tensors):
                # argument check
                if str(anno.input(idx)) != '?':
                    if not isinstance(t, IRTensor):
                        raise ValueError(
                            f"{idx}-th input needs IRTensor, but got {type(t)}: {t}\n"
                            f"signature: {signature}\n"
                            f"annotation: {anno}")
            kwarg_names = [name for name in arg_names[ninputs:]]
            kwarg_vals = args[ninputs:]
            for name, val in zip(kwarg_names, kwarg_vals):
                kwargs[name] = val
            return IRDimops(udfop, op_name, signature, [repr(anno)], tensors, **kwargs)

        # step 4. register in CustomizedOps
        _logger.info(f'registering op {fsig}...')
        CustomizedOps.register(fsig, udfop, code, fn)
        return fn

    return decorator
