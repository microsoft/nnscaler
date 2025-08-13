#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.parallel import parallelize, ComputeConfig
from pathlib import Path
from tests.parallel_module.test_gencode import print_gencode, _gencode_contains


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_recompute_root_module():
    """
    Test that when recompute_modules='ROOT' is set, the entire module is marked for recompute
    """
    batch_size = 2
    hidden_dim, ffn_dim, num_layers = 64, 64, 1

    dummy_input = {'x': torch.randn(batch_size, hidden_dim)}
    model = Model(hidden_dim, ffn_dim, num_layers)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=False)

    # Test with ROOT recompute
    config = AutoDistConfig(recompute_modules='ROOT')
    model_graph = ModelGraph(ir_graph, config)

    print("=== ROOT recompute module configuration ===")
    print(f"min_recompute_mem: {model_graph.min_recompute_mem}")
    print(f"Number of recompute groups: {len(model_graph.recompute_groups)}")

    # With ROOT recompute, the entire model should be one recompute group
    # The recompute memory should be the total training memory of the entire model
    model_node = model_graph.scope_tree_root

    # Since we're recomputing the ROOT module, the min_recompute_mem should be
    # the training memory of the entire model
    expected_recompute_mem = model_node.train_mem
    assert model_graph.min_recompute_mem == expected_recompute_mem, \
        f"Expected recompute mem {expected_recompute_mem}, got {model_graph.min_recompute_mem}"

    # All forward operations should be in one big recompute group
    fnodes = ir_graph.select(ntype=IRFwOperation)
    print(f"Total forward nodes: {len(fnodes)}")
    print(f"Recompute groups: {len(model_graph.recompute_groups)}")

    # With ROOT recompute, there should be one recompute group containing all operations
    assert len(model_graph.recompute_groups) == 1, \
        f"Expected 1 recompute group for ROOT, got {len(model_graph.recompute_groups)}"

    # The single recompute group should contain all forward operations
    recompute_group = model_graph.recompute_groups[0]
    assert len(recompute_group) == len(fnodes), \
        f"Expected {len(fnodes)} nodes in recompute group, got {len(recompute_group)}"

    # Verify that all forward nodes are in the recompute group
    recompute_node_set = set(recompute_group)
    fnodes_set = set(fnodes)
    assert recompute_node_set == fnodes_set, \
        "Recompute group should contain exactly all forward nodes"

    print("ROOT recompute test passed: entire model is marked for recompute")


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10, bias=False)
        self.linear2 = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_parallelize_with_root_recompute():
    """
    Test parallelize with recompute_modules='ROOT' and examine generated code
    """
    m = SimpleModel()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        # Test with ROOT recompute
        pas_cfg = {
            'recompute_modules': 'ROOT',
            'parallel_profile': False
        }

        print("=== Testing parallelize with recompute_modules='ROOT' ===")
        parallelize(
            m,
            {'x': trace_data},
            'autodist',
            ComputeConfig(1, 1, use_end2end=True, pas_config=pas_cfg),
            reuse='override',
            gen_savedir=tempdir,
            load_module=False,
        )

        print("\n=== Generated code with ROOT recompute ===")
        print_gencode(tempdir, SimpleModel, 0)

        # Check that recompute is applied
        recompute_matches = _gencode_contains(tempdir, SimpleModel, 0, r'def recompute\(')
        checkpoint_matches = _gencode_contains(tempdir, SimpleModel, 0, r'ckpt\.checkpoint\(recompute')

        print(f"\nFound {len(recompute_matches)} recompute function definitions")
        print(f"Found {len(checkpoint_matches)} checkpoint calls")

        assert len(recompute_matches) >= 1, "Should generate at least one recompute function"
        assert len(checkpoint_matches) >= 1, "Should use checkpoint for recompute function"
