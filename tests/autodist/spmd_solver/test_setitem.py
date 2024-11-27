#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import logging
import tempfile
import torch
import os
import nnscaler
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.profiler.database import ProfileDataBase
from ...utils import catch_log


class Module(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(Module, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        z = y1.new_empty(x.size(0), 2 * self.hidden_dim)
        z[:, :self.hidden_dim] = y1
        z[:, self.hidden_dim:] = y2
        return z.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_set_item():
    nnscaler.utils.set_default_logger_level(logging.INFO)
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()
    bsz, hidden_dim = 2, 10
    dummy_input = {
        'x': torch.rand(bsz, hidden_dim),
    }
    model = Module(hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)

        selected_nodes = [node for node in ir_graph.nodes() if 'setitem' in node.signature]
        db = ProfileDataBase()
        from nnscaler.profiler.database import _logger as _logger_profiler
        with catch_log(_logger_profiler) as log_stream_profiler:
            for node in selected_nodes:
                ret = db.profile(node)
            profiler_logs = log_stream_profiler.getvalue()
            profiler_logs = profiler_logs.split('\n')
            in_place_log = [log for log in profiler_logs if 'in-place operation detected, the input tensor is modified, will not profile backward' in log]
            assert len(in_place_log) == 2
            fail_log = [log for log in profiler_logs if 'fail to profile' in log]
            assert len(fail_log) == 0
