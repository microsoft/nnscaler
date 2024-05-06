# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Tuple, Optional, Union
from cube.ir.operator import IRFwOperation


TransAlgo = Optional[Tuple[int, int]]


class Constraints:

    def __init__(self):

        self.op_trans: Dict[IRFwOperation, TransAlgo] = {}
        self.op_place: Dict[IRFwOperation, Tuple[int, int] | None] = {}

    def add_trans_constraints(self,
                              op: IRFwOperation,
                              algo: TransAlgo,
                              num: int | Tuple[int, int] | None):
        """Add constraints for operator transformation

        Args:
            op (IRFwOperation): operator to be constrained
            algo (TransAlgo): list of (idx, dim) tuples
            num (int, (int, int) or None): 
                The number (int) or min/max (int, int) number of transformed sub-operators.
        """
        if op in self.op_trans:
            raise KeyError(f"Operator {op.name}[{op.cid}] already has transformation constraints")
        if not (num is None or isinstance(num, int) or (len(num) == 2 and all(isinstance(n, int) for n in num))):
            raise TypeError(f"num must be one of None, int, or (min, max) ints, but got {num}")
        if isinstance(num, int):
            num = (num, num)
        self.op_trans[op] = (algo, num)
    
    def add_place_constraints(self,
                              op: IRFwOperation,
                              devices: Tuple[int]):
        """Add constraints for operator placement
        
        Notes:
            the op should be constrained in trans_constraints with
            the fixed num value to be same with len(devices).

        Args:
            op (IRFwOperation): operator to be constrained
            devices (Tuple[int]): list of devices for each sub-operator.
        """
        if op not in self.op_trans:
            raise KeyError(f"Operator {op.name}[{op.cid}] should have transformation constraints")
        _, nums = self.op_trans[op]
        if nums is None or nums[0] != nums[1]:
            raise ValueError(
                f"By setting placement constraints, the operator's transformation constraint "
                f"should have a fixed partitioned number, but got {nums} for node: {op}[{op.cid}]")
        num = nums[0]
        if len(devices) != num:
            raise ValueError(f"Expected devices to have length {num}, got {len(devices)}")
        self.op_place[op] = tuple(devices)

    def add_order_constraints(self,
                              op1: IRFwOperation, mbidx1: int,
                              op2: IRFwOperation, mbidx2: int):
        """Add constraints for operator orderings

        Args:
            op1 (IRFwOperation): operator to be constrained
            mbidx1 (int): micro-batch index of op1
            op2 (IRFwOperation): operator to be constrained
            mbidx2 (int): micro-batch index of op2
        """
        raise NotImplementedError

    def __repr__(self):
        dscp = 'Op Transformation Constraints:\n'
        for node, (algo, num) in self.op_trans.items():
            dscp += f"  {node.name}[{node.cid}]: algo={algo}, num={num}\n"
        dscp += '\n'
        dscp += 'Op Placement Constraints:\n'
        for node, devices in self.op_place.items():
            dscp += f"  {node.name}[{node.cid}]: devices={devices}\n"
        return dscp
