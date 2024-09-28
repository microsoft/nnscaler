#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Register cutomized function
"""

from typing import Dict, Callable, Optional, Union, List
from functools import partial
import inspect
import logging

from torch import ScriptFunction

from nnscaler.graph.function.dimops import IRDimops, OpAnno
from nnscaler.graph.parser.fx.concrete_trace_utils.wrap_utils import is_autograd_apply
from nnscaler.ir.operator import IRTensor, IRFwOperation

_logger = logging.getLogger(__name__)


class CustomizedOps:
    """Customized op registry."""

    # signature -> IRDimop creation function
    kOpMap: Dict[str, Callable] = {}
    # singature -> runtime function
    kOpRuntime: Dict[str, Callable] = {}
    # signature -> runtime function implementation code
    kOpCodeDef: Dict[str, str] = {}
    # signature -> special emit function, will not store if emit_fn is None
    # It accepts the node, repred args, repred kwargs, runtime_devid, plan_ndevs, runtime_ndevs
    # as input and returns the generated code.
    kOpEmit: Dict[str, Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str]] = {}

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
    def register(signature: str, op_create_fn: Callable, code: str, runtime_fn: Callable,
                 emit_fn: Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str] = None):
        """Register an operator

        Args:
            signature (str): operator signature
            op_create_fn (Callable): IRDimops creation function
            code (str): runtime function implementation code
            runtime_fn (Callable): runtime function
            emit_fn (Callable): special emit function for codegen, will use default emit function if emit_fn is None.
                                It accepts the node, repred args, repred kwargs, runtime_devid, plan_ndevs, runtime_ndevs
                                as input and returns the generated code.

        Returns:
            None
        """
        builtins = ['_operator.', 'torch.', 'nnscaler.runtime.function.']
        if any(signature.startswith(builtin) for builtin in builtins):
            raise RuntimeError(f"Cannot register operators with signature starting from any of {builtins}")
        assert signature not in CustomizedOps.kOpMap, f"function {signature} is already registered"
        CustomizedOps.kOpMap[signature] = op_create_fn
        CustomizedOps.kOpRuntime[signature] = runtime_fn
        CustomizedOps.kOpCodeDef[signature] = code
        if emit_fn is not None:
            CustomizedOps.kOpEmit[signature] = emit_fn


def register_op(annotation: Union[str, Callable], name: Optional[str] = None,
             code_impl_pattern: str = 'import', emit_fn: Callable[[IRFwOperation, List[str], Dict[str, str], int, int, int], str] = None) -> Callable:
    """
    Register a function with IRDimops annotations.

    This function is cooperated with IRDimops. Users can only register functions defined under a module, instead of
    ones defined inside a function / class or __main__ scope.

    The annotation (`annotation`) specifies the number of inputs as *args,
    and treat all the rest inputs as **kwargs.

    For tensor-type inputs, the annotation should be a string of identifiers separated by space, e.g., `'a b'`;
    For non-tensor-type inputs, the annotation should be specified '?'.

    Examples:

    ```python
    import nnscaler
    from third_party import func

    nnscaler.graph.parser.register('a (b c) -> (a b) c')(func)
    ```

    or,

    ```python
    import nnscaler
    from third_party import func

    @nnscaler.graph.parser.register('a (b c) -> (a b) c')
    def func(x, b = 4):
        xxx
    ```

    or,

    ```python
    import nnscaler
    from third_party import func

    def anno_fn(*inputs, **kwargs):
        return 'a (b c) -> (a b) c'

    nnscaler.graph.parser.register(anno_fn)(func)
    ```

    Args:
        annotation (str | Callable): operator annotation of IRDimops or callable function that generates IRFwOperation.
            - op annotation: e.g., 'a (b c) -> (a b) c'
            - a callable function that generates op annotation (str). The function
            taks inputs and kwargs as arguments and returns the operator annotation.
        name (str | None): operator name. Only usable when node_repr is a string.
        code_impl_pattern (str):
            can only be 'import' or 'source'. If 'import', will generate code with
            import statement. If 'source', will take the source code directly.
            Default: 'import'.
        emit_fn (Callable): special emit function for codegen, this emit accepts the node, repred args, repred kwargs, runtime_devid,
            plan_ndevs, runtime_ndevs as input and returns the generated code. Check examples/zigzag_ring_attention/zigzag_attn.py
            for more details.
            Default: None.

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
            elif isinstance(fn, ScriptFunction):
                # fn._torchdynamo_inline is the original function
                import_path = inspect.getmodule(fn._torchdynamo_inline).__name__
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
        elif isinstance(fn, ScriptFunction):
            # fn._torchdynamo_inline is the original function
            fsig = f'{import_path}.{fn._torchdynamo_inline.__name__}'
            op_name = name if name is not None else fn.name
            args = inspect.signature(fn._torchdynamo_inline)
            arg_names = list(args.parameters.keys())
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
            elif isinstance(fn, ScriptFunction):
                raise NotImplementedError('Do not support get source code for ScriptFunction.')
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
        if not (isinstance(annotation, str) or callable(annotation)):
            raise TypeError(f"annotation should be either str or callable, got {type(annotation)}")

        def udfop(*args, signature=None, **kwargs):
            anno = annotation if isinstance(annotation, str) else annotation(*args, **kwargs)
            if not isinstance(anno, str):
                raise TypeError(f"node_repr should return a string, but got {type(anno)}: {anno}")
            anno = OpAnno(anno)
            ninputs = len(anno.inputs())
            if len(args) < ninputs:
                raise ValueError(f"calling function {signature} should include at least {ninputs} *args")
            tensors = args[:ninputs]
            for idx, t in enumerate(tensors):
                # argument check
                if not anno.input(idx).ignore:
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
        CustomizedOps.register(fsig, udfop, code, fn, emit_fn)
        return fn

    return decorator


# [Deprecated] register_op alias
# Will remove in future.
register = register_op
