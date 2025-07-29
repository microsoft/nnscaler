from typing import Any, List

import z3
import numpy as np

from verdict.utils import unique

from .types import SymbExpr, SymbTensor, Shape


def create_name_tensor(shape, prefix):
    """Recursively create a tensor with the given dynamic shape."""
    if len(shape) == 1:
        return np.array([f"{prefix}[{i}]" for i in range(shape[0])])
    return np.array(
        [
            create_name_tensor(shape[1:], prefix=f"{prefix}[{i}]")
            for i in range(shape[0])
        ]
    )


def create_z3_tensor(shape: Shape, prefix: str, dtype: object, ctx=None) -> SymbTensor:
    names = create_name_tensor(shape, prefix)
    return create_z3_tensors_from_names(names, dtype, ctx)


def create_z3_tensors_from_names(names, dtype, ctx=None):
    return np.vectorize(lambda x: dtype(x, ctx=ctx))(names)


def equalize_z3tensors(ts: List[SymbTensor | z3.ArithRef]) -> List[SymbExpr]:
    unique([type(t) for t in ts])
    constraints = []
    t0 = ts[0]
    if type(t0) is not np.ndarray:
        constraints = [t0 == t for t in ts[1:]]
    else:
        shape = unique([t.shape for t in ts])
        for t in ts[1:]:
            constraints.extend(t0[i] == t[i] for i in np.ndindex(shape))
    return constraints


def concrete_z3(zt: SymbTensor | z3.ArithRef, model: z3.ModelRef) -> np.ndarray | float:
    if type(zt) is z3.ArithRef:
        return model.evaluate(zt, model_completion=True)
    flat_result = [z3_to_python_number(model.evaluate(x, model_completion=True)) for x in zt.ravel()]
    result = np.array(flat_result).reshape(zt.shape)
    return result


def z3_to_python_number(val: z3.ExprRef) -> int | float:
    if z3.is_int_value(val):
        return val.as_long()
    elif z3.is_rational_value(val):
        num = val.numerator_as_long()
        den = val.denominator_as_long()
        return num / den  # float
    else:
        raise TypeError(f"Unsupported Z3 value type: {val}")
