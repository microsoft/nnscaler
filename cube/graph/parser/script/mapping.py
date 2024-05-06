# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Union
from functools import partial

import cube.graph.function as function
from cube.ir.operator import IRFwOperation
from cube.graph.parser.register import CustomizedOps


class Sign2Op:

    @staticmethod
    def map(signature: str) -> Callable[..., Union[IRFwOperation, int, float]]:
        """
        Map the signature to GenericLogicalOp
        """
        if signature in Sign2Op.kOpMap:
            return partial(Sign2Op.kOpMap[signature], signature=signature)
        if CustomizedOps.exist(signature):
            return CustomizedOps.map(signature)
        raise KeyError(f"{signature} is not supported yet")

    @staticmethod
    def exist(signature: str) -> bool:
        if signature in Sign2Op.kOpMap:
            return True
        if CustomizedOps.exist(signature):
            return True
        return False

    kOpMap = {
        'torch.nn.functional.linear' : function.Linear,
        'torch.linear': function.Linear,
        'torch.nn.functional.softmax' : function.Softmax,
        'torch.nn.functional.dropout' : function.Dropout,
        'torch.nn.functional.gelu' : function.GeLU,
        'torch.gelu' : function.GeLU,
        'torch.nn.functional.silu' : function.SiLU,
        'torch.silu' : function.SiLU,
        'torch.nn.functional.pad': function.Pad,
        'torch.nn.functional.layer_norm': function.LayerNorm,
        'torch.nn.functional.embedding': function.Embedding,
        'torch.nn.functional.cross_entropy': function.CrossEntropy,
        'torch.clone': function.Clone,

        # elementwise opeartors
        'torch.add' : function.Add,
        'torch.sub' : function.Sub,
        'torch.mul' : function.Mul,
        'torch.div' : function.Div,
        'torch.floordiv' : function.FloorDiv,
        'torch.neg': function.Neg,
        'torch.gt': function.CompareGT,
        'torch.lt': function.CompareLT,
        'torch.ge': function.CompareGE,
        'torch.le': function.CompareLE,
        'torch.pow': function.Pow,
        'torch.sin': function.Sin,
        'torch.cos': function.Cos,
        'torch.tanh': function.Tanh,

        'torch.bmm' : function.BatchLinear,
        'torch.matmul': function.Matmul,
        'torch.sum' : function.Sum,
        'torch.mean': function.Mean,

        'torch.transpose' : function.Transpose,
        'torch.view': function.View,
        'torch.reshape': function.Reshape,
        'torch.conv2d': function.Conv2D,
        'torch.conv3d': function.Conv3D,
        'torch.pad': function.Pad,
        'torch.select': function.Select,
        'torch.slice': function.Slice,

        'torch.repeat': function.Repeat,
        'torch.cat': function.Cat,
        'torch.stack': function.Stack,
        'torch.chunk': function.Chunk,
        'torch.flatten': function.Flatten,
        'torch.roll': function.Roll,

        'torch.adaptive_avg_pool1d': function.AdaptiveAvgPool1d,

        # runtime functions
        'cube.runtime.function.function.anchor': function.GraphAnchor,
        'cube.runtime.function.function.identity': function.Identity,
        'cube.runtime.function.function.multiref': function.MultiRef,
        'cube.runtime.function.function.accum': function.Accum,
    }
