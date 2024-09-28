#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
This file is the pytree extension by nnscaler.
"""
from collections import namedtuple
from typing import Any, List, Tuple, Iterable

import torch
import inspect

from . import orig_func, _pytree
from ._pytree import *

import nnscaler.graph.parser.fx.concrete_trace_utils.concrete_proxy as cct


# if pytree is a ConcreteProxy, type(pytree) will return the type of ConcreteProxy.value
# so here we need add additional check for ConcreteProxy.
# this function will override the `_get_node_type` function defined in the original pytree code
def _get_node_type(pytree: _pytree.PyTree) -> Any:
    if orig_func.isinstance(pytree, cct.ConcreteProxy):
        return orig_func.type(pytree)
    if _pytree._is_namedtuple_instance(pytree):
        return namedtuple
    return type(pytree)

_pytree._get_node_type = _get_node_type


# by default, the registered types are:
#   builtins.tuple
#   builtins.list
#   builtins.dict
#   collections.namedtuple
#   collections.OrderedDict
#   collections.defaultdict
#   collections.deque

# register slice to pytree.
def _slice_flatten(d: slice) -> Tuple[List[Any], Context]:
    return [d.start, d.stop, d.step], None


def _slice_flatten_with_keys(
    d: Tuple[Any, ...]
) -> Tuple[List[Tuple[_pytree.KeyEntry, Any]], Context]:
    values, context = _slice_flatten(d)
    return [('start', values[0]), ('stop', values[1]), ('step', values[2])], context


def _slice_unflatten(values: Iterable[Any], context: Context) -> slice:
    return slice(*values)


_pytree._private_register_pytree_node(
    slice,
    _slice_flatten,
    _slice_unflatten,
    serialized_type_name="builtins.slice",
    flatten_with_keys_fn=_slice_flatten_with_keys,
)


# register return_types to pytree, copy from torch.return_types
return_types = torch._C._return_types

def pytree_register_structseq(cls):
    def structseq_flatten(structseq):
        return list(structseq), None

    def structseq_flatten_with_keys(structseq):
        values, context = structseq_flatten(structseq)
        return [(_pytree.SequenceKey(i), v) for i, v in enumerate(values)], context

    def structseq_unflatten(values, context):
        return cls(values)

    _pytree.register_pytree_node(
        cls,
        structseq_flatten,
        structseq_unflatten,
        flatten_with_keys_fn=structseq_flatten_with_keys,
    )

for name in dir(return_types):
    if name.startswith('__'):
        continue

    _attr = getattr(return_types, name)
    globals()[name] = _attr

    # torch.return_types is a structseq, aka a "namedtuple"-like thing defined by the Python C-API.
    if inspect.isclass(_attr) and issubclass(_attr, tuple):
        pytree_register_structseq(_attr)


def tree_leaves_with_spec(pytree: _pytree.PyTree, spec: TreeSpec) -> List:
    """
    Flat a pytree with a given spec.

    Example:

        pytree = [1, (2, {3: 4})]
        spec = TreeSpec([*, (*, *)])
    
        # the returned value is
        [1, 2, {3: 4}]
    """
    assert isinstance(spec, TreeSpec)

    if isinstance(spec, LeafSpec):
        return [pytree]

    flatten_fn = _pytree.SUPPORTED_NODES[spec.type].flatten_fn
    child_pytrees, _ = flatten_fn(pytree)

    if len(child_pytrees) != len(spec.children_specs):
        raise RuntimeError(f'The number of pytree children is not equal to the give specs.')

    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_leaves_with_spec(child, child_spec)
        result += flat

    return result


def get_common_spec(*specs: TreeSpec) -> TreeSpec:
    """
    Return the common part of treespecs.
    For example:
        specs[0] is {'a': [*,], 'b': [*, *]}
        specs[1] is {'a': [*,], 'b': [*, *, *]}
        common spec is {'a': [*,], 'b': *}
    """
    if tree_any(lambda spec: isinstance(spec, LeafSpec), specs):
        return LeafSpec()
    if all(spec.type == specs[0].type and spec.context == specs[0].context for spec in specs):
        if all(len(spec.children_specs) == len(specs[0].children_specs) for spec in specs):
            children_specs = [get_common_spec(*children_specs) for children_specs in zip(*(spec.children_specs for spec in specs))]
            return TreeSpec(type=specs[0].type, context=specs[0].context, children_specs=children_specs)
    return LeafSpec()
