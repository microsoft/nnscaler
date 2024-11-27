#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
import os
from pathlib import Path
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.apis import parallelize_graph

import nnscaler
from nnscaler.ir.unique import IDGenerator
from nnscaler.graph.segment import IRSegment
from nnscaler.flags import CompileFlag
from nnscaler.runtime.utils import microbatches
from nnscaler.program import Program, SemanticDataLoader, SemanticModel
from nnscaler.graph.gener.gen import IRAdapterGener


class Model(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        x = torch.matmul(x, self.w)
        x = torch.nn.functional.relu(x)
        x = torch.matmul(x, self.w)
        return x.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_shared_param_pipeline():
    bsz, hidden_dim = 1024, 1024

    CompileFlag.dev_mode = True

    for idx, cfg_fname in enumerate(
        ['all_replicated_pp.json', 'replicated_and_partition.json']):
        with tempfile.TemporaryDirectory() as tempdir:
            model = Model(hidden_dim)
            model.train()

            program = Program()
            program.clear()
            IDGenerator().clear()

            dataloader = SemanticDataLoader(
                microbatches([{
                    'x': torch.randn(bsz, hidden_dim)
                }]))

            smodel = SemanticModel(model, attr_savedir=tempdir)
            smodel.dummy_input = {'x': torch.randn(bsz, hidden_dim)}
            smodel.constant_folding = True
            program.set_input([dataloader.irobj])
            ir_dummy_input = next(dataloader)
            outputs = smodel(ir_dummy_input)
            outputs.backward()
            program.set_output([outputs])
            program.finalize()
            ir_graph = program.get_graph()

            print(ir_graph.nodes())
            plan_path = Path(os.path.dirname(__file__)) / cfg_fname
            cfg = AutoDistConfig(load_plan_path=plan_path, mesh_col=4)
            graph = parallelize_graph(ir_graph, cfg)
            assert isinstance(graph.nodes()[4], IRSegment)
            # check multiref is correctly inserted at the 1st IRSegment (pipeline stage)
            has_multiref = False
            for node in graph.nodes()[4].nodes():
                if node.signature == 'nnscaler.runtime.function.multiref':
                    has_multiref = True
                    break
            assert has_multiref

            graph = IRAdapterGener.gen(graph, cost_fn=None)
            if graph.sched is not None:
                graph.sched.apply()
