#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig


class MLP(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class Layer(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.mlp = MLP(hidden_dim, ffn_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.mlp(x)
        x = x + residual
        return x

class Encoder(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Layer(hidden_dim, ffn_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Layer(hidden_dim, ffn_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, num_layers):
        super().__init__()
        self.encoder = Encoder(hidden_dim, ffn_dim, num_layers)
        self.decoder = Decoder(hidden_dim, ffn_dim, num_layers)

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        x = x.sum()
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_recompute():
    batch_size = 2
    hidden_dim, ffn_dim, num_layers = 1024, 4096, 1

    dummy_input = {'x': torch.randn(batch_size, hidden_dim)}
    model = Model(hidden_dim, ffn_dim, num_layers)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=False)

    config = AutoDistConfig(recompute_modules='Decoder.Layer')
    model_graph = ModelGraph(ir_graph, config)
    model_node = model_graph.scope_tree_root
    print(model_node)

    assert len(model_node.children) == 3
    encoder_node = model_node.children[0]
    decoder_node = model_node.children[1]
    print(decoder_node)

    assert len(decoder_node.children) == num_layers
    for layer_node in decoder_node.children:
        assert len(layer_node.children) == 3
        ln_node = layer_node.children[0]
        assert ln_node.leaf_size == 1
        assert ln_node.in_mem == batch_size * hidden_dim * 4
        assert ln_node.train_mem == batch_size * hidden_dim * 4 + batch_size * 8
        assert ln_node.param_mem == hidden_dim * 8
        assert ln_node.buffer_mem == 0
        mlp_node = layer_node.children[1]
        assert mlp_node.leaf_size == 3
        assert mlp_node.in_mem == batch_size * hidden_dim * 4
        assert mlp_node.train_mem == batch_size * hidden_dim * 4 + batch_size * ffn_dim * 8
        assert mlp_node.param_mem == hidden_dim * ffn_dim * 8
        assert mlp_node.buffer_mem == 0
        add_node = layer_node.children[2]
        assert add_node.leaf_size == 1
        assert add_node.in_mem == batch_size * hidden_dim * 8
        assert add_node.train_mem == 0
        assert add_node.param_mem == 0
        assert add_node.buffer_mem == 0

        assert layer_node.leaf_size == ln_node.leaf_size + mlp_node.leaf_size + add_node.leaf_size
        assert layer_node.in_mem == batch_size * hidden_dim * 4
        assert layer_node.train_mem == ln_node.train_mem + mlp_node.train_mem + add_node.train_mem
        assert layer_node.param_mem == ln_node.param_mem + mlp_node.param_mem + add_node.param_mem
        assert layer_node.buffer_mem == 0

    assert decoder_node.leaf_size == num_layers * layer_node.leaf_size
    assert decoder_node.in_mem == batch_size * hidden_dim * 4
    assert decoder_node.train_mem == num_layers * layer_node.train_mem
    assert decoder_node.param_mem == num_layers * layer_node.param_mem
    assert decoder_node.buffer_mem == 0

    assert model_node.leaf_size == encoder_node.leaf_size + decoder_node.leaf_size + 1
    assert model_node.in_mem == encoder_node.in_mem
    assert model_node.train_mem == encoder_node.train_mem + decoder_node.train_mem
    assert model_node.param_mem == encoder_node.param_mem + decoder_node.param_mem
    assert model_node.buffer_mem == encoder_node.buffer_mem + decoder_node.buffer_mem

    assert model_graph.min_recompute_mem == layer_node.train_mem
    fnodes = ir_graph.select(ntype=IRFwOperation)
    assert model_graph.recompute_groups == [fnodes[5 * (num_layers + i) : 5 * (num_layers + i) + 5] for i in range(num_layers)]

    # will label operator like GELU and add with `has_batch_dim=True`
    for op in model_graph.operator_list:
        assert op.has_batch_dim, f'{op} does not have batch dim'
