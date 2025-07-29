from typing import Any, Dict

import numpy as np

from verdict.debug_print import dump_nodes, dump_stages
from verdict.symbolics import concrete_z3, SymbTensor
from verdict.stage import Stage
from verdict.graph import DFG, Tensor


def scale_slcs(slcs, org_shape, rx_shape):
    """
    Scale a list of (start, end) tuples from org_shape to rx_shape,
    and return as a tuple of slice objects.

    Parameters:
        slcs (list of (int, int)): List of (start, end) in org_shape.
        org_shape (tuple of int): Original shape of the tensor.
        rx_shape (tuple of int): Rescaled shape of the tensor.

    Returns:
        tuple of slice: Scaled slices for rx_shape.
    """
    assert len(slcs) == len(org_shape) == len(rx_shape), "Dimension mismatch"

    result = []
    for (start, end), org, rx in zip(slcs, org_shape, rx_shape):
        scale = rx / org
        new_start = int(round(start * scale))
        new_end = int(round(end * scale))
        result.append(slice(new_start, new_end))

    return tuple(result)



class Report:
    def __init__(self):
        # equivalence failure
        self.Gs: DFG = None
        self.Gp: DFG = None
        self.stage: Stage = None

        self.data: Dict[Tensor, SymbTensor] = None
        self.sat: Any = None
        self.model: Any = None

    def dump_z3(self, path=None) -> str:
        msgs = []
        pr = msgs.append

        def concrete(t: Tensor):
            return concrete_z3(self.data[t], self.model)

        pr(f"🚨 Stage {self.stage.id} solver result: {self.sat}\n")
        pr(dump_nodes(self.stage.snodes, self.Gs, None))
        pr(dump_nodes(self.stage.pnodes, self.Gp, None))
        # msgs = [
        #     print(dump_stages([self.stage], self.Gs, self.Gp, None)),
        # ]

        def pr_lng(l):
            pr("")
            pr(f"👉 {l.Ts}")
            concrete_Ts = concrete(l.Ts)
            pr(concrete_Ts)
            for slc in sorted(l.slice_map.keys()):
                equal_copies = l.slice_map[slc]
                rx_slc = scale_slcs(slc, l.full_shape, concrete_Ts.shape)
                sliced_concrete_Ts = concrete_Ts[rx_slc]
                pr(f"🍕 {tuple((int(s), int(e)) for (s, e) in slc)} {np.array2string(sliced_concrete_Ts, separator=' ', max_line_width=np.inf)}")
                for equal_copy in equal_copies:
                    reduced = np.zeros_like(sliced_concrete_Ts)
                    for i, t in enumerate(equal_copy):
                        concrete_Tp = concrete(t)
                        pr(
                            f"{'=' if i==0 else ' '}  ⨁ {t} {np.array2string(concrete_Tp, separator=' ', max_line_width=np.inf)}"
                        )
                        reduced += concrete_Tp
                    if not np.array_equal(reduced, sliced_concrete_Ts):
                        pr("🚨 ⬆️")

        pr("🔗 INPUT LINEAGES")
        for l in self.stage.input_lineages:
            pr_lng(l)
        pr("🔗 OUTPUT LINEAGES")
        for l in self.stage.output_lineages:
            pr_lng(l)
        return "\n".join([str(msg) for msg in msgs])


report = Report()
