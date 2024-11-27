#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import torch
from nnscaler.program import SemanticModel, Program
from nnscaler.flags import CompileFlag
from nnscaler.ir.cten import IRObject


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_program_model_nested_input():

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.empty(4, 4))
            self.param2 = torch.nn.Parameter(torch.empty(4, 4))

        def forward(self, x: dict):
            shortcut = x['data']
            x = torch.matmul(x['data'], self.param1)
            x = x + self.param2
            x = x + shortcut
            x = x + self.param1
            return {'loss': torch.sum(x)}

    old_dev_mode = CompileFlag.dev_mode
    try:
        CompileFlag.dev_mode = True
        Program().clear()

        dummy_input = {'x': {'data': torch.randn(4, 4)}}
        module = MyModule()
        model = SemanticModel(module, save_content=False, constant_folding=True)

        obj = IRObject(value=dummy_input['x'])
        model(obj)
        graph = model.get_graph()
        print(graph.extra_repr())

        assert graph.input(0) == obj
        # getitem
        assert graph.node(0).input(0) == obj
        # getitem
        assert graph.node(1).input(0) == obj
    finally:
        CompileFlag.dev_mode = old_dev_mode
