# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional, Any, Dict, Union, Tuple
import numpy as np
import logging
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.dimops import IRDimops, DimAnno, DimopSplit, TransformRule
from cube.ir.tensor import IRSubTensor
from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from collections import deque

_logger = logging.getLogger(__name__)


class DimSplitEinops(GenericDistAlgo):
    """!
    Split Dimops at tensor dimension.

    Note: for dimensions of multiple identitifers, only the first identifier
    can be partitioned.

    Default rule for identifier split:
        * Sum-reduce identifier ('+'):
            * For inputs/outputs that have the identifier, will be partitioned on its diemension uniformly..
            * For inputs that don't have the identifier, will be replicated
            * For outputs that don't have the identifier, will be partitioned on its value uniformly.

        * Spatial identifier (''):
            * For inputs/outputs that have the identifier, will be partitioned on its diemnsion uniformly.
            * For inputs/outputs that don't have the identifier, will be replicated

        * Frozen identifier ('^'):
            * Cannot be partitioned.

        If the identifier appears as the same name in argument name, the
        argument will also be uniformly partitioned.

        Non-tensor will always be replicated.

    Note the default rule isn't always expressive for all possible partition algorithms.
    E.g., linear xw + b to partition on reduction dimension,
    whitch requires b to be value split but actually according to the default rule, will be replicated.
    Therefore we require special rules for such cases.
    """

    def __init__(self, node: IRDimops):
        if not isinstance(node, IRDimops):
            raise TypeError(f"Expect IRDimops")
        super().__init__(node)

    def get_identifier_reduce(self, idx: int, dim: int, num: int) -> Tuple[str, DimAnno.ReduceType]:
        """
        Get the partitioned identifier and reduction type.
        If the partitioned number is 1, return the first hidden identitifer
        Otherwise, return the first hidden identifier whose length > 1

        @param idx int: input/output index. Take the idx-th input tensor or (idx-ninputs)-th output
        @param dim int: input dimension

        @return identifier Optional[str]: annotated dimension identifier
        @return reduction Optional[DimAnno.ReduceType]
        """
        node: IRDimops = self.node
        eshapes = node.anno.inputs() + node.anno.outputs()
        hidx = None
        for hidx, adim in enumerate(eshapes[idx].dims[dim].identifiers):
            if num == 1: break
            dimlen = node.anno.getlen(adim)
            if adim == '1^' or dimlen == 1: continue
            break
        if hidx is None: return (None, None)
        reduce = eshapes[idx].dims[dim].reduces[hidx]
        return adim, reduce

    def satisfy(self, idx: int, dim: Union[int, str], num: int) -> bool:
        """
        Check whether the condition satisfies.

        @param idx int: input/output index. Take the idx-th input tensor or (idx-ninputs)-th output tensor
        @param dim Union[int, str]: tensor dimension or 'v', i.e., partition at value dimension.
        @param num int: chunks to partition the dimension

        @return satisfy bool: true if can be partitioned, elsewise false.
        """
        assert all(isinstance(cond, int) for cond in [idx, num]), "expect int condition"
        assert isinstance(dim, int) or dim == 'v', f"expect dim to be int or 'v'"
        node: IRDimops = self.node

        tensors = node.inputs() + node.outputs()
        assert isinstance(tensors[idx], IRSubTensor), f"partition on a non-tensor input/output"
        assert 0 <= idx and idx < len(tensors), f"index out of boundary: {idx} >= {len(tensors)}"

        tensors = node.inputs() + node.outputs()
        if isinstance(dim, int):
            dim = dim if dim >= 0 else dim + tensors[idx].ndims
            assert dim < tensors[idx].ndims, f"dimension output of boundary: {dim} >= {node.input(idx).ndims}"

        # try split at tensor spatial dimension
        if isinstance(dim, int):
            adim, reduce = self.get_identifier_reduce(idx, dim, num)
            if adim is None: return False
            dimlen = node.anno.getlen(adim)
            # first check node special rules first
            for rule in node.transform_rules:
                splits = rule.inputs() + rule.outputs()
                if splits[idx] == DimopSplit.D(dim):
                    return dimlen % num == 0
            # then check default rules
            if reduce == DimAnno.ReduceType.Freeze:
                return False
            return dimlen % num == 0
        else:
            for rule in node.transform_rules:
                splits = rule.inputs() + rule.outputs()
                if splits[idx].isV():
                    return True
            return False

    def instantiate(self, idx: int, dim: Union[int, str], num: int) -> Optional[List[IRDimops]]:

        node: IRDimops = self.node
        satisfy = self.satisfy(idx, dim, num)

        if isinstance(dim, int):
            adim, reduce = self.get_identifier_reduce(idx, dim, num)
        else:
            adim, reduce = 'Value', None

        if not satisfy:
            color, default = '\033[31m', '\033[0m'
            _logger.info(f"split {node.name}: {node.anno} | dim: {adim} num: {num} reduce: {reduce} ... {color}{'Failed!'}{default}")
            return None
        rule: TransformRule = self.infer(idx, dim, num)

        # transform
        def transform(tensor: Any, split: DimopSplit) -> List[Any]:
            if not isinstance(tensor, IRSubTensor):
                return [tensor] * num
            if split.isD():
                # get sub-tensors with nested partition on dims
                sub_tensors = tensor.split_dims(split.dims, (num,) * len(split.dims))
                # reshape to (num, num, ...) and select [i, i, ..., i] sub-tensor, i = 0 to num-1
                sub_tensors = np.array(sub_tensors, dtype=IRSubTensor).reshape((num,) * len(split.dims))
                sub_tensors = [sub_tensors[(i,) * len(split.dims)] for i in range(num)]
                return sub_tensors
            if split.isR():
                return tensor.replicate(num)
            if split.isV():
                return tensor.split_val(num)
            assert False, f"got unknown split: {split}"

        ins = list()
        for split, itensor in zip(rule.inputs(), node.inputs()):
            ins.append(transform(itensor, split))
        ous = list()
        for split, otensor in zip(rule.outputs(), node.outputs()):
            ous.append(transform(otensor, split))
        kwargs = rule.modifier()(node.kwargs, idx, dim, num)

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            sub_node: IRDimops = node.new(inputs, outputs, **kwargs)
            sub_node.infer_shape()
            sub_nodes.append(sub_node)

        return sub_nodes

    def infer(self, idx: int, dim: Union[int, str], num: int) -> Optional[TransformRule]:
        """
        Given the partition choice on `dim` dimension of idx-th input,
        return the partitioning of the output tensor.

        @param idx int: the input index
        @param dim int: the dimension to partition

        @return rule TransformRule: the transformation rule
        """
        node: IRDimops = self.node
        assert isinstance(dim, int) or dim == 'v', f"expect dim to be int or 'v'"
        # check node special rules first
        for r in node.transform_rules:
            splits = r.inputs() + r.outputs()
            if isinstance(dim, int):
                if splits[idx].isD():
                    # make negative offset to be possitive
                    ndims = len(node.input(idx).shape)
                    rdims = tuple((d + ndims) % ndims for d in splits[idx].dims)
                    if dim in rdims:
                        return r
            else:
                if splits[idx].isV():
                    return r
        # otherwise use default rule
        assert isinstance(dim, int), f"Error: expect dim to be int for default rules"
        adim, reduce = self.get_identifier_reduce(idx, dim, num)
        if reduce == DimAnno.ReduceType.Freeze:
            return None
        itransform, otransform = [], []
        # input
        for idx, idim in enumerate(node.anno.inputs()):
            dims = idim.getdims(adim)
            if len(dims) == 0:
                itransform.append(DimopSplit.R())
            else:
                if len(dims) > 1:
                    _logger.warning(
                        f'node ({self.node.name}-{self.node.cid}): detected an input tensor '
                        f'is split on {len(dims)} dimensions, this will cause data loss.',
                        stacklevel=0,
                    )
                itransform.append(DimopSplit.D(dims))
        # output
        for idx, odim in enumerate(node.anno.outputs()):
            dims = odim.getdims(adim)
            if len(dims) == 0:
                otransform.append(
                    DimopSplit.R() if reduce == DimAnno.ReduceType.Dim else DimopSplit.V()
                )
            else:
                if len(dims) > 1:
                    _logger.warning(
                        f'node ({self.node.name}-{self.node.cid}): detected an output tensor '
                        f'is split on {len(dims)} dimensions, this will cause data loss.',
                        stacklevel=0,
                    )
                otransform.append(DimopSplit.D(dims))
        # modifier
        def modify(kwargs: Dict, idx: int, dim: int, num: int):
            updated_kwargs = dict(**kwargs)
            if adim in updated_kwargs:
                assert updated_kwargs[adim] % num == 0, \
                    f"cannot set kwargs: {adim}: {updated_kwargs[adim]} % num ({num}) != 0"
                updated_kwargs[adim] = updated_kwargs[adim] // num
            return updated_kwargs

        return TransformRule(itransform, otransform, modify)


def collect_split_info(node: IRFwOperation):
    anno = node.anno

    split_info = {}

    for idx_shape, shape_anno in enumerate(anno.inputs()):
        if not isinstance(node.inputs()[idx_shape], IRSubTensor):
            continue
        for idx_dim, dim_anno in enumerate(shape_anno.dims):
            for idx_id, identifier in enumerate(dim_anno.identifiers):
                if dim_anno.reduces[idx_id] == DimAnno.ReduceType.Freeze:
                    continue
                if identifier not in split_info:
                    split_info[identifier] = (idx_shape, idx_dim, idx_id)

    return split_info

def gen_partitions(node: IRFwOperation, ngpus: int) -> List[IRFwOperation]:

    def gen_hash(node: IRFwOperation) -> str:
        ret = node.signature
        for it in node.inputs():
            if not isinstance(it, IRTensor): continue
            ret = ret + '-' + str(it.shape)
        return ret

    dq = deque()
    visited = set()
    dq.append((node, ngpus))
    visited.add(gen_hash(node))

    gen_nodes = []

    while dq:
        cur_node, cur_ngpus = dq.popleft()
        gen_nodes.append(cur_node)
        split_info = collect_split_info(cur_node)

        for key, val in split_info.items():
            idx_1st, dim_1st, _ = val
            dim_size = cur_node.anno.getlen(key)

            # TODO(yizhu1): only consider powers of 2 currently
            split_deg = 2
            while split_deg <= dim_size and split_deg <= cur_ngpus:
                if dim_size % split_deg != 0:
                    break

                new_nodes = cur_node.algorithms('dim').instantiate(idx=idx_1st, dim=dim_1st, num=split_deg)
                new_node = new_nodes[0]
                new_ngpus = cur_ngpus // split_deg

                cur_key = gen_hash(new_node)

                split_deg = split_deg * 2

                if cur_key in visited:
                    continue

                dq.append((new_node, new_ngpus))
                visited.add(cur_key)

    return gen_nodes