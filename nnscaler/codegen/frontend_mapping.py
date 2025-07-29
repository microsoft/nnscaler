# Some operators should be specially handled during codegen to the frontend code,
# here we define the customized rule for code emisson.

from typing import Callable, Dict, List, Optional, Tuple

from nnscaler import ir
from nnscaler.ir.cten import IRTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.parser.register import CustomizedOps


class Sign2EmitRule:
    """Emit rule for frontend PyTorch codegen"""

    def __init__(self) -> None:
        # the registered emit rules
        self._sign2rule = {
            'torch.slice': self.emit_slice,
            'setattr': self.emit_setattr,
        }

    def map(self, signature: str) -> Callable:
        """Get the emit rule for the given signature

        Args:
            signature (str): signature of the operator

        Returns:
            Callable: emit rule that takes the node, args (List[str]) and kwargs (Dict[str, str]) as input
        """
        if signature in CustomizedOps.kOpEmit:
            return CustomizedOps.kOpEmit[signature]
        else:
            return self._sign2rule.get(signature, self.emit_common)

    def emit_common(self, node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
        """Default rule to join all args and kwargs"""

        signature = node.signature

        kw_pairs = list()
        for key, val in kwargs.items():
            code = f'{key}={val}'
            kw_pairs.append(code)

        args = ", ".join(list(args) + kw_pairs)
        return f"{signature}({args})"

    def emit_slice(self, node: IRFwOperation, arg_vars: List[str], kw_pairs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
        """Special rule for generating slice node

        The op is:
            aten::slice(input:Tensor, dim:int=0, start:Optional[int]=None, end:Optional[int]=None, step:int=1) -> Tensor

        but at the frontend such an invocation must be rewritten as 'x[:, l:h:s, :, :]'
        depending on the 'input's rank and the 'dim' value.
        """
        out_tensors : tuple = node.outputs()
        assert len(out_tensors) == 1
        out_tensor : IRTensor = out_tensors[0]

        assert len(arg_vars) == 1
        in_tensor_var : str = arg_vars[0]

        dim : int = kw_pairs["dim"]
        start : Optional[int] = kw_pairs["start"]
        end : Optional[int] = kw_pairs["end"]
        step : int = kw_pairs["step"]

        rank = len(out_tensor.shape)
        subscript_components = [":"] * rank

        slice_str = f"{start or ''}:{end or ''}:{step}"
        subscript_components[dim] = slice_str

        return f"{in_tensor_var}[{', '.join(subscript_components)}]"

    def emit_setattr(self, node, arg_vars: List[str], kw_pairs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
        """Special rule for generating setattr node
        """
        assert False, f"This emit rule is deprecated, please report if you reach here"
