#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import z3
import numpy as np

def row_to_z3arr(symbols, default=0):
    """
    Converts a list or 1D NumPy array of Z3 expressions to a Z3 Array[int -> T].
    """
    # Handle NumPy array input
    if isinstance(symbols, np.ndarray):
        symbols = symbols.tolist()
    if not symbols:
        raise ValueError("Input symbol list/array must not be empty.")
    # hadle data types
    T = symbols[0].sort()
    default_expr = default if isinstance(default, z3.ExprRef) else T.cast(default)
    # convert
    arr = z3.K(z3.IntSort(symbols[0].ctx), default_expr)
    for i, sym in enumerate(symbols):
        arr = z3.Store(arr, i, sym)
    return arr


def z3arr_to_row(arr, length):
    return np.array([arr[i] for i in range(length)])


def getuop_topk_indices(ctx):
    """
    This defines uninterpreted topk. As topk involves max / argmax operations
    that is not easily expressible by symbolic variables. It also defines axioms
    for checking equivalence between topks.
    
    NOTE: current axioms is partial, assuming the input array is not sharded and 
    has the same order. If future parallelization requires topk over sharded arr 
    and aggregate later on, we need to add corresponding axioms to teach z3 for 
    checking equivalence.
    
    Axiom:
    (arr1 == arr2) && (k1 == k2)
        => topk(arr1,k1) == topk(arr2,k2)
    """
    OPNAME = "topk_indices"
    PREFIX = f"uninterpreted_{OPNAME}_input_"
    
    RealArrSort = z3.ArraySort(z3.IntSort(ctx=ctx), z3.RealSort(ctx=ctx))
    SelectedIndicesSort = z3.ArraySort(z3.IntSort(ctx=ctx), z3.IntSort(ctx=ctx))

    # Declare the uninterpreted function
    topk = z3.Function(OPNAME, RealArrSort, z3.IntSort(ctx=ctx), SelectedIndicesSort)

    # Symbolic arrays
    arr1 = z3.Const(PREFIX+"arr1", RealArrSort)
    arr2 = z3.Const(PREFIX+"arr2", RealArrSort)

    # Symbolic k
    k1 = z3.Int(PREFIX+"k1", ctx=ctx)
    k2 = z3.Int(PREFIX+"k2", ctx=ctx)
    
    axiom = z3.Implies(z3.And(arr1 == arr2, k1 == k2), topk(arr1, k1) == topk(arr2, k2), ctx=ctx)
    return topk, [axiom]


def getuop_topk_values(ctx):
    OPNAME = "topk_values"
    PREFIX = f"uninterpreted_{OPNAME}_input_"
    
    RealArrSort = z3.ArraySort(z3.IntSort(ctx=ctx), z3.RealSort(ctx=ctx))

    # Declare the uninterpreted function
    topk = z3.Function(OPNAME, RealArrSort, z3.IntSort(ctx=ctx), RealArrSort)

    # Symbolic arrays
    arr1 = z3.Const(PREFIX+"arr1", RealArrSort)
    arr2 = z3.Const(PREFIX+"arr2", RealArrSort)

    # Symbolic k
    k1 = z3.Int(PREFIX+"k1", ctx=ctx)
    k2 = z3.Int(PREFIX+"k2", ctx=ctx)
    
    axiom = z3.Implies(z3.And(arr1 == arr2, k1 == k2), topk(arr1, k1) == topk(arr2, k2), ctx=ctx)
    return topk, [axiom]


def getuop_idx2onehot(ctx):
    OPNAME = "idx2onehot"
    PREFIX = f"uninterpreted_{OPNAME}_input_"
    
    RealArrSort = z3.ArraySort(z3.IntSort(ctx=ctx), z3.RealSort(ctx=ctx))

    # Declare the uninterpreted function
    onehot = z3.Function(OPNAME, RealArrSort, RealArrSort)

    # Symbolic arrays
    idxs1 = z3.Const(PREFIX+"idxs1", RealArrSort)
    idxs2 = z3.Const(PREFIX+"idxs2", RealArrSort)
    
    axiom = z3.Implies(idxs1 == idxs2, onehot(idxs1) == onehot(idxs2), ctx=ctx)
    return onehot, [axiom]

def getuop_val2onehot(ctx):
    OPNAME = "val2onehot"
    PREFIX = f"uninterpreted_{OPNAME}_input_"
    
    RealArrSort = z3.ArraySort(z3.IntSort(ctx=ctx), z3.RealSort(ctx=ctx))

    # Declare the uninterpreted function
    extend_vals = z3.Function(OPNAME, RealArrSort, RealArrSort, RealArrSort)

    # Symbolic arrays
    idxs1 = z3.Const(PREFIX+"idxs1", RealArrSort)
    idxs2 = z3.Const(PREFIX+"idxs2", RealArrSort)
    vals1 = z3.Const(PREFIX+"vals1", RealArrSort)
    vals2 = z3.Const(PREFIX+"vals2", RealArrSort)
    
    axiom = z3.Implies(z3.And(idxs1 == idxs2, vals1 == vals2), extend_vals(idxs1, vals1) == extend_vals(idxs2, vals2), ctx=ctx)
    return extend_vals, [axiom]

