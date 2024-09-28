#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List

from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.cten import IRTensor


class IRPad(IRFwOperation):
    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        # torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
        # pad: List[int]
        signature = 'torch.nn.functional.pad'
        assert len(inputs) == 1, "Expected only input, weight, bias as inputs"
        assert len(kwargs) == 3, "Expected 2 kwargs: mode, value"
        super().__init__(name, signature, inputs, 1, **kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.input(0).shape) == 0:
            return False

        pad  = self.kwargs['pad']
        assert len(pad) % 2 == 0, "IRPad::infer_shape len(pad) % 2 == 0"

        shape = self.input(0).shape
        for pad_idx, pad_size in enumerate(pad):
            shape[-1 - (pad_idx // 2)] += pad_size

        self.output(0).shape = shape
        return True

    def new(self, inputs: List, outputs: List, pad = None):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        if pad == None:
            pad = self.kwargs['pad']
        mode = self.kwargs['mode']
        value = self.kwargs['value']
        op = IRPad(self.signature, inputs, self.name,
                   pad=pad, mode=mode, value=value)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op


class IRConv2D(IRFwOperation):

    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        signature = 'nnscaler.runtime.function.conv2d'
        assert len(inputs) == 3, "Expected only input, weight, bias as inputs"
        assert len(kwargs) == 4, "Expected 4 kwargs: stride, padding, dialation, groups"
        super().__init__(name, signature, inputs, 1, **kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.input(0).shape) == 0 or len(self.input(1).shape) == 0:
            return False
        N = self.input(0).shape[0]
        iH, iW = self.input(0).shape[2:4]
        oC = self.input(1).shape[0]
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        dH = self.input(1).shape[2]
        dW = self.input(1).shape[3]
        oH = (iH + padding[0] + padding[1] - dilation[0] * (dH - 1) - 1) // stride[0] + 1
        oW = (iW + padding[2] + padding[3] - dilation[1] * (dW - 1) - 1) // stride[1] + 1
        shape = [N, oC, oH, oW]
        self.output(0).shape = shape
        return True

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        groups = self.kwargs['groups']
        op = IRConv2D(self.signature, inputs, self.name,
                      stride=stride, padding=padding, dilation=dilation, groups=groups)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op


class IRConv3D(IRFwOperation):

    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        signature = 'nnscaler.runtime.function.conv3d'
        assert len(inputs) == 3, "Expected only input, weight, bias as inputs"
        assert len(kwargs) == 4, "Expected 4 kwargs: stride, padding, dialation, groups"
        super().__init__(name, signature, inputs, 1, **kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.input(0).shape) == 0 or len(self.input(1).shape) == 0:
            return False
        N = self.input(0).shape[0]
        iC = self.input(0).shape[1]
        iT, iH, iW = self.input(0).shape[2:5]

        oC = self.input(1).shape[0]
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        dT = self.input(1).shape[2]
        dH = self.input(1).shape[3]
        dW = self.input(1).shape[4]

        oT = (iT + 2 * padding[0] - dilation[0] * (dT - 1) - 1) // stride[0] + 1
        oH = (iH + 2 * padding[1] - dilation[1] * (dH - 1) - 1) // stride[1] + 1
        oW = (iW + 2 * padding[2] - dilation[2] * (dW - 1) - 1) // stride[2] + 1
        shape = [N, oC, oT, oH, oW]

        self.output(0).shape = shape
        return True

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        groups = self.kwargs['groups']
        op = IRConv3D(self.signature, inputs, self.name,
                      stride=stride, padding=padding, dilation=dilation, groups=groups)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op
