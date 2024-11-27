#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.graph.tracer import pytree_utils
from nnscaler.graph.tracer.pytree_utils import (
    get_common_spec,
    tree_leaves_with_spec,
)
from nnscaler.graph.tracer.concrete_proxy import ConcreteProxy
from nnscaler.graph.tracer.concrete_tracer import (
    update_tree_proxy_value,
)


def test_pytree_related_utils():
    pytree_1 = {'a': [1, {'b': 2}]}
    pytree_2 = {'a': [3, 4]}
    pytree_3 = {'a': [5, [6, 7]]}

    pytree_1_spec = pytree_utils.tree_flatten(pytree_1)[1]
    pytree_2_spec = pytree_utils.tree_flatten(pytree_2)[1]
    pytree_3_spec = pytree_utils.tree_flatten(pytree_3)[1]

    # test get_common_spec
    common_spec = get_common_spec(pytree_1_spec, pytree_2_spec, pytree_3_spec)

    assert common_spec == \
        pytree_utils.TreeSpec(dict, ['a'], [pytree_utils.TreeSpec(list, None, [pytree_utils.LeafSpec(), pytree_utils.LeafSpec()])]),\
        f"expect TreeSpec(dict, ['a'], [TreeSpec(list, None, [*, *])]), but get {common_spec}"

    # test tree_leaves_with_spec
    assert tree_leaves_with_spec(pytree_1, common_spec) == [1, {'b': 2}]
    assert tree_leaves_with_spec(pytree_2, common_spec) == [3, 4]
    assert tree_leaves_with_spec(pytree_3, common_spec) == [5, [6, 7]]

    # test update_tree_proxy_value
    class DummyNode:
        pass
    class DummyTracer:
        pass

    pytree_src = {'key': 1, 'value': (2, 3)}
    pytree_dst = {
                    'key': ConcreteProxy(DummyNode(), 4, DummyTracer()),
                    'value': ConcreteProxy(
                                DummyNode(),
                                (5, ConcreteProxy(DummyNode(), 6, DummyTracer())),
                                DummyTracer()
                            )
                 }
    new_pytree = update_tree_proxy_value(pytree_dst, pytree_src)
    assert new_pytree['key'].value == 1
    assert new_pytree['value'].value[0] == 2
    assert new_pytree['value'].value[1].value == 3
