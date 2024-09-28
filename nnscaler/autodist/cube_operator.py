#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple, Dict, Set, Optional
from nnscaler.ir import IRTensor, IRFwOperation, IRSubTensor
from nnscaler.graph.function.dimops import DimAnno, IRDimops
from nnscaler.algorithm.ops.dimops import collect_split_info


class CubeOperator:
    """
    CubeOperator is a wrapper for IRFwOperation.
    Currently, it maintains the following information for an IRDimops:
    - in_tensors: input tensors, including parameters and buffers
    - out_tensors: output tensors
    - producers: operators that produce the input tensors
    - consumers: operators that consume the output tensors
    - dim_info: a mapping from dimension name to its position and reduce type
    - parallelable_dims: a set of dimension names that can be parallelized
    - recompute: a flag indicating whether the operator will be recomputed
    - recompute_start_op: a flag indicating whether the operator consumes tensors outside of a recompute region
    - has_batch_dim: a flag indicating whether the operator has a batch dimension
    - since there can be shared tensors in the model, we use the following vars to estimate the memory usage accurately:
    - omit_recompute_in_idx: a list of indices of input tensors that should be omitted
    - omit_train_idx: a list of indices of activation tensors that should be omitted
    - omit_param_idx: a list of indices of parameter tensors that should be omitted
    - omit_buffer_idx: a list of indices of buffer tensors that should be omitted
    """

    def __init__(self, ir_cell: IRFwOperation):
        self.ir_cell = ir_cell
        self.in_tensors, self.out_tensors = [], []
        self.op_name = self.ir_cell.signature

        self.producers: List[CubeOperator] = list()
        self.consumers: List[CubeOperator] = list()

        self.dim_info = {}
        self.parallelable_dims = set()
        self._has_sum_dim = False
        self._recompute = False
        self._recompute_start_op = False
        self._recompute_last_op = False
        self._has_attr = False

        self.omit_recompute_in_idx = []
        self.omit_train_idx = []
        self.omit_param_idx = []
        self.omit_buffer_idx = []
        self.has_batch_dim = False

        if not isinstance(ir_cell, IRDimops):
            return

        for item in ir_cell.inputs():
            if isinstance(item, IRTensor):
                self.in_tensors.append(item)
                if item.is_attr():
                    self._has_attr = True
        for item in ir_cell.outputs():
            if isinstance(item, IRTensor):
                self.out_tensors.append(item)

        self.collect_anno_info()

    @property
    def has_sum_dim(self):
        return self._has_sum_dim

    @property
    def has_attr(self):
        return self._has_attr

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value: bool):
        self._recompute = value

    @property
    def recompute_start_op(self):
        return self._recompute_start_op

    @recompute_start_op.setter
    def recompute_start_op(self, value: bool):
        self._recompute_start_op = value

    @property
    def recompute_last_op(self):
        return self._recompute_last_op

    @recompute_last_op.setter
    def recompute_last_op(self, value: bool):
        self._recompute_last_op = value

    def add_producer(self, producer: 'CubeOperator'):
        self.producers.append(producer)

    def add_consumer(self, consumer: 'CubeOperator'):
        self.consumers.append(consumer)

    def collect_anno_info(self):
        for idx_shape, shape_anno in enumerate(self.ir_cell.anno.inputs()):
            if not isinstance(self.ir_cell.inputs()[idx_shape], IRTensor):
                continue
            for idx_dim, dim_anno in enumerate(shape_anno.dims):
                for idx_id, identifier in enumerate(dim_anno.identifiers):
                    reduce_type = dim_anno.reduces[idx_id]
                    if reduce_type != DimAnno.ReduceType.Freeze:
                        self.parallelable_dims.add(identifier)
                    if reduce_type == DimAnno.ReduceType.Sum:
                        self._has_sum_dim = True
                    val = (idx_shape, idx_dim, idx_id, reduce_type)
                    if identifier not in self.dim_info:
                        self.dim_info[identifier] = val
                    else:
                        if reduce_type != self.dim_info[identifier][-1]:
                            raise ValueError(
                                f'inconsistent reduce type for {identifier} in {self.ir_cell} with {self.ir_cell.anno}'
                            )

    def dim_id2pos(self, dim_name: str) -> Tuple[int, int]:
        if dim_name == -1:
            return (-1, -1)
        else:
            assert dim_name in self.dim_info, f'{dim_name} not in {self.dim_info}'
            idx, dim, _, _ = self.dim_info[dim_name]
            return idx, dim

    def pos2dim_id(self, pos: Tuple[int, int]) -> str:
        if pos == (-1, -1):
            return -1
        else:
            if not isinstance(self.ir_cell, IRDimops):
                raise ValueError(f'{self.ir_cell} is not IRDimops')
            idx, dim = pos
            adim, reduce_type = self.ir_cell.algorithms(
                'dim').get_identifier_reduce(idx, dim, 2)
            assert adim is not None, f'cannot find dim at {pos} in {self.ir_cell}'
            return adim

    def get_reduce_type(self, dim_id: str):
        return self.dim_info[dim_id][-1]

    def __repr__(self):
        anno = self.ir_cell.anno if isinstance(self.ir_cell, IRDimops) else ''
        return f'Operator {self.ir_cell} {anno} at {self.ir_cell.comment}'
