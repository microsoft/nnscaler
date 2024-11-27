#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import tempfile
import pytest
from nnscaler.parallel import _gen_graph, ComputeConfig
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 20)

    def forward(self, x):
        return self.embed(x).sum()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_requires_grad():
    model = Model()
    model.train()

    dummy_input = {'x': torch.randint(0, 10, (10, 10))}

    with tempfile.TemporaryDirectory() as tempdir:

        graph, _ = _gen_graph(
            model,
            dummy_input,
            outdir=tempdir,
            constant_folding=True,
            end2end_mode=True,
        )
        embed_op = graph.nodes()[1]
        assert embed_op.inputs()[0].requires_grad == False
        assert embed_op.inputs()[1].requires_grad == True
