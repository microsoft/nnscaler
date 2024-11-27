#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import logging
import tempfile
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.dimops import IRDimops

_logger = logging.getLogger(__name__)

def test_register_apex_fused_op():

    have_apex = True

    try:
        from apex.normalization.fused_layer_norm import FusedLayerNorm
        from apex.normalization.fused_layer_norm import FusedRMSNorm

    except Exception as e:
        _logger.warning(f'skip op registering test on external apex due to lack of apex installation.')
        have_apex = False
                
    if not have_apex:
        return
    
    class ApexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.empty(128, dtype=torch.float16))
            # fused layer norm
            self.fused_layer_norm = FusedLayerNorm((128,), eps=1e-5, elementwise_affine=False)
            self.fused_layer_norm_affine = FusedLayerNorm((128,), eps=1e-5, elementwise_affine=True)
            # fused rms norm
            self.fused_rms_norm = FusedRMSNorm((128,), eps=1e-5, elementwise_affine=False)
            self.fused_rms_norm_affine = FusedRMSNorm((128,), eps=1e-5, elementwise_affine=True)

        def forward(self, x):
            x = self.param + x
            x = self.fused_layer_norm(x)
            x = self.fused_layer_norm_affine(x)
            x = self.fused_rms_norm(x)
            x = self.fused_rms_norm_affine(x)
            return x

    sample = torch.randn((4, 128), dtype=torch.float16, device=torch.cuda.current_device())
    model = ApexModel().half()
    with tempfile.TemporaryDirectory() as tempdir:
        graph = convert_model(model, dummy_input={'x': sample}, attr_savedir=tempdir)
        print(graph.extra_repr())
        apex_nodes = [n for n in graph.select(ntype=IRFwOperation) if 'apex' in n.signature]
        assert len(apex_nodes) == 4, graph.extra_repr()
        assert all(isinstance(n, IRDimops) for n in apex_nodes), graph.extra_repr()
