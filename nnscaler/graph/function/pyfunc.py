#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Tuple

from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.cten import IRObject


class IRPyFunc(IRFwOperation):
    """
    Python runtime function
    """

    def __init__(self, signature: str,
                 inputs: Tuple[IRObject], outputs: Tuple[IRObject], **kwargs):
        name = signature.split('.')[-1]
        super().__init__(name, signature, inputs, len(outputs))
        for idx, t in enumerate(outputs):
            self.set_output(idx, t)
        self.kwargs.update(**kwargs)

    def infer_shape(self) -> dict[int, tuple[int, ...]]:
        """
        Shape will not be inferred for python runtime
        """
        return {}

    def __repr__(self) -> str:
        sign = self.signature.split('.')[-1]
        dscp = (f"PyOp{self._id}-{self.device}(sign={sign}, "
                f"inputs={self.inputs()}, "
                f"outputs={self.outputs()})")
        return dscp

    def extra_repr(self) -> str:
        sign = self.signature.split('.')[-1]
        dscp = (f"PyOp{self._id}-{self.device}(sign={sign}, "
                f"inputs={self.inputs()}, "
                f"outputs={self.outputs()})")
        return dscp
