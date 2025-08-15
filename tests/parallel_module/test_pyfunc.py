#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
import math
import os
from pathlib import Path
from nnscaler.parallel import parallelize, ComputeConfig

from .common import init_distributed
from ..utils import replace_all_device_with, catch_log
from ..launch_torchrun import launch_torchrun


class MyMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x.mm(y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output.mm(y.t())
        grad_y = x.t().mm(grad_output)
        return grad_x, grad_y


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(10, 10))

    def forward(self, x):
        x = MyMatmul.apply(x[0], self.weight)
        return x


def _worker():
    init_distributed()

    dummy_input = {'x': (torch.rand(2, 10), torch.rand(10, 10))}
    from nnscaler.graph.parser.parser import _logger as _logger_parser
    from nnscaler.graph.graph import _logger as _logger_graph
    from nnscaler.graph.segment import _logger as _logger_seg
    with tempfile.TemporaryDirectory() as tempdir, \
        catch_log(_logger_parser) as log_stream_parser, \
        catch_log(_logger_seg) as log_stream_seg, \
        catch_log(_logger_graph) as log_stream_graph:

        m_new = parallelize(
            MyModule(),
            dummy_input,
            'dp',
            ComputeConfig(1, 1, use_end2end=False),
            gen_savedir=tempdir,
            load_module=True
        )
        parser_logs = log_stream_parser.getvalue()
        seg_logs = log_stream_seg.getvalue()
        graph_logs = log_stream_graph.getvalue()
        # parser.py: parse_prim_function_method
        assert 'Find unknown custom autograd operation' in parser_logs
        # segment.py: infer_grad
        assert 'nnScaler does not support backward of IRPyFunc' in seg_logs
        # graph.py: from_logic_graph
        assert 'nnScaler does not support to compute gradients for IRPyFunc.' in graph_logs

        # not registered, encounter NameError
        with pytest.raises(NameError):
            logit = m_new(dummy_input['x'])
            print(logit)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 1, reason='lack of gpu devices')
@replace_all_device_with('cpu')
def test_ir_pyfunc():
    launch_torchrun(1, _worker)
