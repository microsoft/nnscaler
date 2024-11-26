#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import torch
import os
from nnscaler.parallel import _gen_graph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.algorithm.ops.dimops import gen_partitions

from ...utils import replace_all_device_with

class NaiveFFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 4096, bias=False)
        self.linear2 = torch.nn.Linear(4096, 1024, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@replace_all_device_with('cpu')
def test_gen_partitions():
    with tempfile.TemporaryDirectory() as tempdir:
        graph, _ = _gen_graph(NaiveFFN(), {'x': torch.randn(2, 128, 1024)}, tempdir, False)
        fc1, relu, fc2 = graph.select(ntype=IRFwOperation)
        assert len(gen_partitions(fc1, 1)) == 1
        # C(4, 1) + 1 = 5
        assert len(gen_partitions(fc1, 2)) == 5
        # C(4, 2) + 2 * C(4, 1) + 1 - 1 = 14
        assert len(gen_partitions(fc1, 4)) == 14
        # C(4, 1) + 1 - 1 = 4
        assert len(gen_partitions(fc1, 4, base=4, depth=1)) == 4


class DepthwiseConv2d(torch.nn.Module):
    def __init__(self, in_channels, multiplier_k, kernel_size, stride=1, padding=0):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels * multiplier_k, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
    
    def forward(self, x):
        return self.depthwise(x)


@replace_all_device_with('cpu')
def test_gen_partitions_depthwise_conv2d():
    in_channels = 8
    multiplier_k = 4
    kernel_size = 3
    stride = 1
    padding = 1
    batch_size = 16
    height = 256
    width = 256
    with tempfile.TemporaryDirectory() as tempdir:
        graph, _ = _gen_graph(DepthwiseConv2d(in_channels, multiplier_k, kernel_size, stride, padding), 
                              {'x': torch.randn(batch_size, in_channels, height, width)}, 
                              tempdir, False)
        depthwise = graph.select(ntype=IRFwOperation)[0]
        # anno: n (g 1^) 256^ 256^, (g 4^) 1^ 3^ 3^, (g 4^) -> n (g 4^) 256^ 256^
        assert len(gen_partitions(depthwise, 1)) == 1
        # n g, n/2 g, n g/2
        assert len(gen_partitions(depthwise, 2)) == 3
        # n g, n/2 g, n g/2, n/2 g/2, n g/2/2, n/2/2 g
        assert len(gen_partitions(depthwise, 4)) == 6
        # n g, n/4 g, n g/4
        assert len(gen_partitions(depthwise, 4, base=4, depth=1)) == 3

