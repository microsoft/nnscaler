#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import operator
import _operator
import torch

from nnscaler.graph.parser.converter import to_fx_graph
from nnscaler.graph.tracer.torch_fx_patcher import side_effectful_inplace_ops
import nnscaler.runtime.function as cube_rt_function

from ...utils import replace_all_device_with


def test_side_effectful_inplace_ops():
    # this is for the validation of side_effectful_inplace_ops
    int_inplace_ops = {
        operator.iadd, operator.isub, operator.imul, operator.ifloordiv,
        operator.ilshift, operator.irshift,
        operator.imod, operator.ipow,
    }
    float_inplace_ops = {
        operator.itruediv,
    }
    bool_inplace_ops = {
        operator.iand, operator.ior, operator.ixor,
    }
    # add _operator versions to make it sure it still equals with operator version
    # in the future
    complex_inplace_ops = {
        operator.setitem, _operator.setitem
    }

    assert int_inplace_ops.union(float_inplace_ops, bool_inplace_ops, complex_inplace_ops) == side_effectful_inplace_ops

    not_implemented_mat_inplace_ops = {
        operator.imatmul
    }

    for intop in int_inplace_ops:
        x = torch.arange(1, 10, dtype=torch.int32)
        y = torch.arange(1, 10, dtype=torch.int32)
        orig_xid = id(x)
        orig_x = x.clone()
        intop(x, y)
        assert id(x) == orig_xid
        assert not torch.equal(x, orig_x)

    for floatop in float_inplace_ops:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        orig_xid = id(x)
        orig_x = x.clone()
        floatop(x, y)
        assert id(x) == orig_xid
        assert not torch.equal(x, orig_x)

    for boolop in bool_inplace_ops:
        x = torch.tensor([True, False, True])
        y = torch.tensor([True, True, False])
        orig_xid = id(x)
        orig_x = x.clone()
        boolop(x, y)
        assert id(x) == orig_xid
        assert not torch.equal(x, orig_x)

    for matop in not_implemented_mat_inplace_ops:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        orig_xid = id(x)
        orig_x = x.clone()
        z = matop(x, y)
        assert id(x) == orig_xid
        assert not torch.equal(z, orig_x)
        assert torch.equal(x, orig_x)  # not updated


class InplaceOpModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x += y
        x.add_(y)
        z = x + y
        x.add_(z)
        operator.iadd(x, y)
        w = torch.add(z, x)
        return w

@replace_all_device_with('cpu')
def test_inplace_op():
    model = InplaceOpModule()
    dummy_input = {'x': torch.tensor([1.0,2.0,3.0]), 'y': torch.tensor([4.0,5.0,6.0])}
    traced_graph = to_fx_graph(model, dummy_input)
    assert torch.equal(
        model(torch.tensor([1.0,2.0,3.0]), torch.tensor([4.0,5.0,6.0])),
        traced_graph(torch.tensor([1.0,2.0,3.0]), torch.tensor([4.0,5.0,6.0]))
    )
    nodes = list(traced_graph.graph.nodes)
    assert nodes[0].op == 'placeholder'
    assert nodes[0].name == 'x'
    assert nodes[1].op == 'placeholder'
    assert nodes[1].name == 'y'

    # x += y
    assert nodes[2].op == 'call_function'
    assert nodes[2].target == operator.iadd
    assert nodes[2].args[0] == nodes[0]
    assert nodes[2].args[1] == nodes[1]

    # x.add_(y)
    assert nodes[3].op == 'call_method'
    assert nodes[3].target == 'add_'
    assert nodes[3].args[0] == nodes[2]  # return value of `x += y`
    assert nodes[3].args[1] == nodes[1]

    #z = x + y
    assert nodes[4].op == 'call_function'
    assert nodes[4].target == operator.add
    assert nodes[4].args[0] == nodes[3]  # return value of `x.add_(y)`
    assert nodes[4].args[1] == nodes[1]

    # x.add_(z)
    assert nodes[5].op == 'call_method'
    assert nodes[5].target == 'add_'
    assert nodes[5].args[0] == nodes[3]  # return value of `x.add_(y)`
    assert nodes[5].args[1] == nodes[4]  # return value of `z = x + y`

    # operator.iadd(x, y)
    assert nodes[6].op == 'call_function'
    assert nodes[6].target == operator.iadd
    assert nodes[6].args[0] == nodes[5]  # return value of `x.add_(z)`
    assert nodes[6].args[1] == nodes[1]  # return value of `y`

    #w = torch.add(z, x)
    assert nodes[7].op == 'call_function'
    assert nodes[7].target == torch.add
    assert nodes[7].args[0] == nodes[4]  # return value of `z = x + y`
    assert nodes[7].args[1] == nodes[6]  # return value of `operator.iadd(x, y)`

    assert nodes[8].op == 'output'
    assert nodes[8].args[0] == nodes[7]  # return value of `return w`


class SetItemModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.graph_token_virtual_distance = torch.nn.Embedding(1, self.num_heads)
 
    def forward(self, x):
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        x[:, :, 1:, 0] = (
                x[:, :, 1:, 0] + t
        )
        x[:, :, 0, :]  = (
                x[:, :, 0, :] + t
        )
        return x
    

@replace_all_device_with('cpu')
def test_inplace_setitem_op():
    model = SetItemModule()
    dummy_input = {'x': torch.rand(4,4,4,4)}

    traced_graph = to_fx_graph(model, dummy_input)

    assert torch.equal(
        model(dummy_input['x']),
        traced_graph(dummy_input['x'])
    )

    nodes = list(traced_graph.graph.nodes)
    for idx, node in enumerate(nodes):
        target_name = node.target.__name__ if hasattr(node.target, '__name__') else node.target
        print(f'{idx}: ({node.op}) {node} = {target_name}{node.args}')

    assert nodes[0].op == 'placeholder'
    assert nodes[0].name == 'x'

    assert nodes[1].op == 'get_attr'
    assert nodes[1].name == 'graph_token_virtual_distance_weight'

    assert nodes[5].op == 'call_function'
    assert nodes[5].name == 'setitem'
    assert nodes[5].target == cube_rt_function.setitem
    
    assert nodes[6].op == 'call_function'
    assert nodes[6].name == 'getitem_1'
    assert nodes[6].args[0] == nodes[5]

    assert nodes[8].op == 'call_function'
    assert nodes[8].name == 'setitem_1'
    assert nodes[8].target == cube_rt_function.setitem
    assert nodes[8].args[0] == nodes[5]

    print(nodes[9].args[0])
    assert nodes[9].args[0] == nodes[8]


class InplaceOpBranchModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, ls: list):
        y = torch.ones_like(x)
        y.add_(1)
        # if y can be correct add 1 during trace, it will always go into the first branch
        if y.mean().item() > 1.5:
            x.sub_(y)
        else:
            x.mul_(y)
        ls[-1] = 1
        ls[-2] = 2
        return ls


@replace_all_device_with('cpu')
def test_inplace_op_value():
    model = InplaceOpBranchModule()
    dummy_input = {'x': torch.rand(10), 'ls': [3, 3, 3]}
    traced_graph = to_fx_graph(model, dummy_input)

    contains_sub_node, contains_mul_node = False, False
    setitem_count = 0
    for node in traced_graph.graph.nodes:
        if node.op == 'call_method':
            if node.target == 'sub_':
                contains_sub_node = True
            if node.target == 'mul_':
                contains_mul_node = True
        if node.op == 'call_function':
            if node.target is cube_rt_function.setitem:
                # this means during trace, the value of ls after setitem is correct
                if setitem_count == 0:
                    assert node.meta['tensor_meta'] == [3, 3, 1]
                if setitem_count == 1:
                    assert node.meta['tensor_meta'] == [3, 2, 1]
                setitem_count += 1

    # this means during trace, the value of y after inplace add is correct
    assert contains_sub_node and not contains_mul_node
    # there should have two setitem node in graph
    assert setitem_count == 2
