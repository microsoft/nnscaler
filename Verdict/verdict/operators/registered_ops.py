#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, final

import numpy as np
import z3
from uuid import uuid4

from verdict.log import logerr
from verdict.graph import DFG, Tensor, Node, DTag, World
from verdict.utils import unique
from verdict.symbolics import (
    SymbExpr,
    SymbShape,
    Shape,
    SymbolCtx,
    SymbTensor,
    equalize_z3tensors,
    create_z3_tensor,
)

from .kws import KW_CC_IDX, KW_CONSTS_KEY
from .names import OpName
from .dim_mapping import get_dim_mapping
from .uninterpret import row_to_z3arr, z3arr_to_row, getuop_topk_indices, getuop_topk_values, getuop_idx2onehot, getuop_val2onehot

"""
NOTE: For registration convenience, if and only if a class/operator 
is defined with an all-capital name will be treated as an 
symbolic operator, and get registered automatically. The all-capital
name should keep consistency with OpName.value[0].upper().
See get_op and OP_MAPPING.

E.g., in verdict.operators.names, the linear ops are defined as:
    FW_linear = ("linear", True)
    BW_linear = ("linear", False)
So they are registered as `class LINEAR`, with fw and bw pass and their
corresponding shape-reduction rules as separate methods.

We aggregate the specification of shape reduction rules, and the 
rewritten symbolic operations into one class for each op, due to
better clarity and maintainability.
"""


def _identity_shape_pass(
    inshapes: List[SymbShape], num_out: int = 1
) -> Tuple[List[SymbShape], List[SymbExpr]]:
    cons = [] if len(inshapes) == 1 else equalize_z3tensors(inshapes)
    return [inshapes[0].copy() for _ in range(num_out)], cons


def _ewdual_w_broadcast_shape_pass(
    inshapes: List[SymbShape],
    org_inshapes: List[Shape],
) -> Tuple[List[SymbShape], List[SymbExpr]]:
    input1_shape, input2_shape = inshapes
    input1_org_shape, input2_org_shape = org_inshapes
    assert len(input1_shape) >= len(input2_shape)
    shared_dims = len(input2_shape)

    # constraints
    constraints = []
    for d in range(-shared_dims, 0):
        if input1_org_shape[d] == input2_org_shape[d]:
            constraints.append(input1_shape[d] == input2_shape[d])
        else:
            # broadcast needed, so one of the inputs should have dim = 1
            if input1_org_shape[d] == 1:
                constraints.append(input1_shape[d] == 1)
            else:
                assert input2_org_shape[d] == 1
                constraints.append(input2_shape[d] == 1)
    # output
    output_shape = input1_shape.copy()
    return [output_shape], constraints


def _chunk(tensor: SymbTensor, nsplit: int, dim: int, i: int) -> SymbTensor:
    """Split tensor at dimension dim into nsplit and return i th part."""
    assert tensor.shape[dim] % nsplit == 0, f"{tensor.shape[dim]}//{nsplit}!=0"
    step = tensor.shape[dim] // nsplit
    slc = [slice(None)] * len(tensor.shape)
    slc[dim] = slice(i * step, (i + 1) * step)
    return tensor[tuple(slc)]


def _bw_matmul(g: SymbTensor, X: SymbTensor, W: SymbTensor) -> Tuple[List[SymbTensor], List[SymbExpr]]:
    """
    Compute the gradients of W given X, W and the gradient of Y.
    Forward computation is Y = X @ W
    Parameters:
    - X: numpy.ndarray with shape (b1, b2, ..., bn, m, k)
    - W: numpy.ndarray with shape (..., bn, k, n)
    - g: numpy.ndarray with shape (b1, b2, ..., bn, m, n)
    Returns:
    - g_X: numpy.ndarray with shape (b1, b2, ..., bn, m, k)
    - g_W: numpy.ndarray with shape (..., bn, k, n)
    """
    # Number of dimensions in X and g_Y
    num_dims_X = X.ndim
    num_dims_W = W.ndim
    assert num_dims_X - 2 <= 26, "einsum notation use single letter to denote dimension"
    assert num_dims_W - 2 <= 26, "einsum notation use single letter to denote dimension"
    batch_dims = "".join(chr(ord("A") + i) for i in range(num_dims_X - 2))
    subscript_X = batch_dims + "mk"
    subscript_W = subscript_X[-num_dims_W:-2] + "kn"
    subscript_g = batch_dims + "mn"

    g_X = np.einsum(f"{subscript_g},{subscript_W}->{subscript_X}", g, W)
    g_W = np.einsum(f"{subscript_X},{subscript_g}->{subscript_W}", X, g)
    return [g_X, g_W]


class SymbolicOperatorAbstract(ABC):
    @abstractmethod
    def infer_rxshape(
        self,
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbShape],
        rxshape_backend: str,
        ctx: SymbolCtx,
    ) -> Tuple[Dict[Tensor, SymbShape], List[SymbExpr]]: ...

    @abstractmethod
    def apply_op(
        self,
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbTensor],
        shapes: Dict[Tensor, Shape],
        apply_op_backend: str,
        ctx: SymbolCtx,
    ) -> Dict[Tensor, SymbShape]: ...


class SymbolicOperator(SymbolicOperatorAbstract):
    AUTO_INFER_BW_SHAPE = True

    @final
    def infer_rxshape(
        self,
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbShape],
        rxshape_backend: str,
        ctx: z3.Context,
    ) -> Tuple[Dict[Tensor, SymbShape], List[SymbExpr]]:
        if rxshape_backend == "z3":
            return self.z3_infer_rxshape(node, G, data, ctx)
        raise NotImplementedError(rxshape_backend)

    def z3_infer_rxshape(
        self,
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbShape],
        ctx: z3.Context,
    ) -> Tuple[Dict[Tensor, SymbShape], List[SymbExpr]]:
        isfw = G.node_opname(node).value[1]
        inputs = G.node_inputs(node)
        outputs = G.node_outputs(node)
        kwargs = G.node_kwargs(node)

        inshapes = [data[t] for t in inputs]
        org_inshapes = [G.tensor_shape(t) for t in inputs]
        if isfw is False and self.AUTO_INFER_BW_SHAPE:
            # for bw ops, if it enables auto inference of bw shapes,
            # we separate fw/mirror inputs from bw grad inputs using
            # `bw_parse_dataflow`, enforcing shape alignments, and
            # re-execute fw pass to get constraints
            new_shapes = {}
            cons = []
            fw_inputs, bw_inputs, bw_outputs = self.bw_parse_dataflow(inputs, outputs)
            fw_inshapes = [data[t] for t in fw_inputs]
            fw_org_inshapes = [G.tensor_shape(t) for t in fw_inputs]
            fw_outshapes, fw_cons = self.z3_fw_shape_pass(
                fw_inshapes, fw_org_inshapes, kwargs, G, ctx
            )
            cons.extend(fw_cons)
            assert len(fw_outshapes) == len(bw_inputs), f"{len(fw_outshapes)} != {len(bw_inputs)}"
            for fw_oshape, bw_in in zip(fw_outshapes, bw_inputs):
                if bw_in is not None:
                    cons.extend(equalize_z3tensors([fw_oshape, data[bw_in]]))
            assert len(fw_inshapes) == len(bw_outputs)
            for fw_ishape, bw_out in zip(fw_inshapes, bw_outputs):
                if bw_out is not None:
                    new_shapes[bw_out] = fw_ishape
            return new_shapes, cons
        else:
            shape_passer = {
                True: self.z3_fw_shape_pass,
                None: self.z3_default_shape_pass,
                False: self.z3_bw_shape_pass,
            }[isfw]
            outshapes, cons = shape_passer(inshapes, org_inshapes, kwargs, G, ctx)
            return {t: shape for t, shape in zip(outputs, outshapes)}, cons

    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Return: fw_inputs, bw_inputs, bw_outputs.
        Only called when isfw=False."""
        # NOTE: Default fw op has one tensor as output.
        # During bw, we normally append mirror inputs after
        # the original bw outputs, so simply separate at 1.
        # NOTE: For ops that have multiple fw output tensors,
        # or bear tensors the does not require gradients,
        # we will override this default pass in child classes.
        return inputs[1:], inputs[:1], outputs

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        raise NotImplementedError

    def z3_bw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        """Called when AUTO_INFER_BW_SHAPE == False."""
        raise NotImplementedError

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        raise NotImplementedError

    @final
    def apply_op(
        self,
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbTensor],
        shapes: Dict[Tensor, Shape],
        apply_op_backend: str,
        ctx: z3.Context,
    ) -> Tuple[Dict[Tensor, SymbTensor], List[SymbExpr]]:
        if apply_op_backend == "z3":
            return self.z3_apply_op(node, G, data, shapes, ctx)
        raise NotImplementedError(apply_op_backend)

    def z3_apply_op(
        self,
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbTensor],
        shapes: Dict[Tensor, Shape],
        ctx: z3.Context,
    ) -> Tuple[Dict[Tensor, SymbTensor], List[SymbExpr]]:
        isfw = G.node_opname(node).value[1]
        inputs = G.node_inputs(node)
        outputs = G.node_outputs(node)
        kwargs = G.node_kwargs(node)

        # pick the op_passer and prepare args accordingly
        if isfw is True or isfw is None:
            op_passer = {
                True: self.z3_fw_op_pass,
                None: self.z3_default_op_pass,
            }[isfw]
            insts = [data[t] for t in inputs]
            args = [node, insts, shapes, kwargs, G, ctx]
        elif isfw is False:
            op_passer = self.z3_bw_op_pass
            # NOTE: no need refresh new outputs here, which may contain None
            mirror_inputs, inputs, _ = self.bw_parse_dataflow(inputs, outputs)
            insts = [data[t] for t in inputs if t is not None]
            mirror_insts = [data[t] for t in mirror_inputs]
            args = [node, insts, mirror_insts, shapes, kwargs, G, ctx]

        output_symbols, cons = op_passer(*args)

        if type(output_symbols) is not list:
            logerr(
                "Op passers should return a list of SymbTensors.",
                OutType=type(output_symbols),
                Op=type(self),
                Isfw=isfw,
            )
            raise
        return {t: symb for t, symb in zip(outputs, output_symbols)}, cons

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        raise NotImplementedError(type(self))

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        raise NotImplementedError(type(self))

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        raise NotImplementedError(type(self))


BINOP_RX_MIN_UNIT = 2


class LINEAR(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp12-(1,)(name=linear, inputs=(t1033(p1031,(32, 128, 4096),d(),v(0/1)), w505(p30,(4096, 4096),d(),v(0/1))), outputs=(t506(p32,(32, 128, 4096),d(),v(0/1)),)) {'bias': None}
        constraints = []
        input_shape, weight_shape = inshapes[:2]
        weight_org_shape = org_inshapes[1]
        assert len(weight_shape) == 2, f"weight is not a matrix, shape={weight_shape}"

        # constraints
        if kwargs["bias"] is not None:
            bias_shape = inshapes[2]
            constraints.append(bias_shape[0] == 1)  # bias is a vector
            # bias shape matches thus broadcastable
            constraints.append(bias_shape[1] == weight_shape[0])
        # matmul shape matches thus multipliable
        constraints.append(input_shape[-1] == weight_shape[-1])
        constraints.append(
            weight_shape[-1] >= min(BINOP_RX_MIN_UNIT, weight_org_shape[-1])
        )

        # output
        output_shape = input_shape.copy()
        output_shape[-1] = weight_shape[0]
        return [output_shape], constraints

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        input, weight = insts
        result = input @ weight.T
        bias = kwargs["bias"]
        if bias is not None:
            result += bias
        return [result], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        input, weight = mirror_insts
        g_input, g_weight = _bw_matmul(g, input, weight.T)
        g_weight = g_weight.T
        # g_weight[0][0] += 1 # MUTATION
        grads = [g_input, g_weight]
        bias = kwargs["bias"]
        if bias is not None:
            g_bias = np.sum(g, axis=tuple(range(g.ndim - 1)))
            grads += [g_bias]
        return grads, []


class EMBEDDING(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp1-(1,)(name=embedding, inputs=(t621(p620,(32, 128),d(),v(0/1)), w493(p2,(51200, 4096),d(),v(0/1))), outputs=(t494(p4,(32, 128, 4096),d(),v(0/1)),)) {'padding_idx': None, 'start': 0, 'stop': 51200}
        input_shape, embedw_shape = inshapes
        num_embeddings, hidden_dim = embedw_shape
        num_dls = G.W.num_dp * G.W.num_mb
        num_indices_per_dl = np.prod(input_shape)
        output_shape = np.append(input_shape.copy(), embedw_shape[-1])
        return [output_shape], [num_embeddings >= num_dls * num_indices_per_dl]

    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # BwOp268-(1,)(FwOp1, inputs=(g624(p5,(32, 128, 4096),d(),v(0/1)),), outputs=(g623(p3,(51200, 4096),d(),v(0/1)),)) {'padding_idx': None, 'start': 0, 'stop': 51200}
        return inputs[1:], inputs[:1], [None, unique(outputs)]

    def _compute_input_token_ids(
        self, fw_insts: List[SymbTensor], dtag: DTag, W: World
    ):
        """Resolve symbolic X in to concrete X."""
        input, embedw = fw_insts
        input_shape, embedw_shape = input.shape, embedw.shape
        num_embeddings, hidden_dim = embedw_shape
        dl_shift = (dtag.dp * W.num_mb + dtag.mb) * np.prod(input_shape)
        flat_ids = np.arange(np.prod(input_shape), dtype=int) + dl_shift
        return flat_ids.reshape(input_shape)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        """torch.nn.functional.embedding; nnscaler.runtime.function.embedding"""
        input_ids = self._compute_input_token_ids(insts, G.node_dtag(node), G.W)
        input, embedw = insts
        out_zt = embedw[input_ids].astype(object)
        return [out_zt], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        input_ids = self._compute_input_token_ids(mirror_insts, G.node_dtag(node), G.W)
        input, embedw = mirror_insts
        g_embedw = np.zeros(embedw.shape, dtype=object)
        np.add.at(g_embedw, input_ids, g)
        return [g_embedw], []


class CREATE_MASK(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp2-(1,)(name=create_mask, inputs=(t621(p620,(32, 128),d(),v(0/1)), t494(p4,(32, 128, 4096),d(),v(0/1))), outputs=(t495(p6,(1, 1, 128, 128),d(),v(0/1)),)) {}
        token_org_shape = org_inshapes[0]
        tokens_shape, h_shape = inshapes
        bsz, seqlen = tokens_shape
        output_shape = np.array(
            [
                z3.IntVal(1, ctx=ctx),
                z3.IntVal(1, ctx=ctx),
                seqlen,
                seqlen,
            ]
        )
        return [output_shape], [seqlen >= min(BINOP_RX_MIN_UNIT, token_org_shape[1])]

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        tokens, h = insts
        seqlen = tokens.shape[1]
        NEG_INF = z3.IntVal(-1e6)
        mask = np.full((1, 1, seqlen, seqlen), NEG_INF)
        mask = np.triu(mask, k=1)
        mask = np.where(mask == 0, z3.IntVal(0), mask)
        return [mask], []


class MULTIREF(SymbolicOperator):
    AUTO_INFER_BW_SHAPE = False

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp549-(0,)(name=multiref, inputs=(t494(p4,(32, 128, 4096),d(),v(0/1)),), outputs=(t964(p962,(32, 128, 4096),d(),v(0/1)), t968(p966,(32, 128, 4096),d(),v(0/1)))) {'times': 2}
        return _identity_shape_pass(inshapes, kwargs["times"])

    def z3_bw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # BwOp568-(1,)(FwOp567, inputs=(g1086(p1084,(32, 128, 4096),d(),v(0/1)), g1090(p1088,(32, 128, 4096),d(),v(0/1))), outputs=(g671(p92,(32, 128, 4096),d(),v(0/1)),)) {}
        # NOTE: identity / multiref has no mirror inputs
        return _identity_shape_pass(inshapes)

    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        return [], inputs, outputs

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        """nnscaler.runtime.function.multiref"""
        input = unique(insts)
        times = kwargs["times"]
        return [input.copy() for _ in range(times)], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return [sum(insts)], []


class IDENTITY(SymbolicOperator):
    AUTO_INFER_BW_SHAPE = False

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_bw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        return [], inputs, outputs

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []


class FLOAT(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []


class POW(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x = unique(insts)
        i = unique(kwargs[KW_CONSTS_KEY])
        return [x**i], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g = unique(insts)
        x = unique(mirror_insts)
        i = unique(kwargs[KW_CONSTS_KEY])
        return [g * i * x ** (i - 1)], []


class MEAN(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp97-(3,)(name=mean, inputs=(t609(p249,(32, 128, 4096),d(),v(0/1)),), outputs=(t610(p251,(32, 128, 1),d(),v(0/1)),)) {'dim': (2,), 'keepdim': True}
        x_shape = unique(inshapes)
        x_org_shape = org_inshapes[0]
        dim = unique(kwargs["dim"])
        keepdim = kwargs["keepdim"]
        assert (
            dim == len(x_shape) - 1
        ), f"mean is expected to apply on the last dimension"

        cons = [x_shape[-1] >= min(BINOP_RX_MIN_UNIT, x_org_shape[-1])]

        # output
        output_shape = x_shape.copy()
        if keepdim:
            output_shape[-1] = z3.IntVal(1, ctx=ctx)
        else:
            output_shape = output_shape[:-1]
        return [output_shape], cons

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x = unique(insts)
        dim = kwargs["dim"]
        keepdim = kwargs["keepdim"]
        return [np.mean(x, axis=dim, keepdims=keepdim)], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        x: SymbTensor = unique(mirror_insts)
        dim = kwargs["dim"]
        keepdim = kwargs["keepdim"]
        # Compute the size of the reduced dimensions
        N = np.prod([x.shape[d] for d in dim])
        # Expand g to match the shape of x if keepdim=False
        if not keepdim:
            g = np.expand_dims(g, axis=dim)
        # Distribute g across the input, scaled by 1/N
        g_input = g / N
        # Broadcast g_input to match the shape of x
        return [np.broadcast_to(g_input, x.shape)], []


class ADD(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp6-(0,)(name=add, inputs=(t498(p11,(32, 128, 1),d(),v(0/1)), 1e-05), outputs=(t499(p13,(32, 128, 1),d(),v(0/1)),)) {'alpha': 1}
        if len(inshapes) == 2:
            return _ewdual_w_broadcast_shape_pass(inshapes, org_inshapes)
        else:
            assert len(inshapes) == 1
            return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        # FwOp6-(0,)(name=add, inputs=(t498(p11,(64, 128, 1),d(),v(0/1)), 1e-05), outputs=(t499(p13,(64, 128, 1),d(),v(0/1)),)) {'alpha': 1, '__consts': [1e-05]}
        # FwOp34-(0,)(name=add, inputs=(t786(p784,(64, 128, 4096),d(),v(0/1)), t532(p87,(64, 128, 4096),d(),v(0/1))), outputs=(t533(p89,(64, 128, 4096),d(),v(0/1)),)) {'alpha': 1, '__consts': []}
        input = insts[0]
        other = insts[1] if len(insts) > 1 else unique(kwargs[KW_CONSTS_KEY])
        alpha = kwargs["alpha"]
        return [input + alpha * other], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        # BwOp205-(0,)(FwOp6, inputs=(g632(p14,(64, 128, 1),d(),v(0/1)),), outputs=(g631(p12,(64, 128, 1),d(),v(0/1)),)) {'alpha': 1, '__consts': [1e-05]}
        # BwOp178-(0,)(FwOp34, inputs=(g668(p90,(64, 128, 4096),d(),v(0/1)),), outputs=(g787(p785,(64, 128, 4096),d(),v(0/1)), g667(p88,(64, 128, 4096),d(),v(0/1)))) {'alpha': 1, '__consts': []}
        g: SymbTensor = unique(insts)
        if len(mirror_insts) == 1:  # if `other` is constant
            return [g], []
        else:
            other = mirror_insts[1]
            alpha = kwargs["alpha"]
            # reshape-1 & sum axis0 for broadcasting
            g_other = np.sum(
                (alpha * g).reshape(-1, *other.shape), axis=0, keepdims=False
            )
            return [g, g_other], []


class RSQRT(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x = unique(insts)
        return [1 / x**0.5], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g = unique(insts)
        x = unique(mirror_insts)
        return [g * -0.5 * x**-1.5], []


class MUL(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp8-(0,)(name=mul, inputs=(t992(p990,(32, 128, 4096),d(),v(0/1)), t500(p15,(32, 128, 1),d(),v(0/1))), outputs=(t501(p17,(32, 128, 4096),d(),v(0/1)),)) {}
        if len(inshapes) == 2:
            return _ewdual_w_broadcast_shape_pass(inshapes, org_inshapes)
        else:
            assert len(inshapes) == 1
            return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        input = insts[0]
        other = insts[1] if len(insts) > 1 else unique(kwargs[KW_CONSTS_KEY])
        return [input * other], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        input, other = mirror_insts
        # reshape-1 & sum axis0 for broadcasting
        g_input = g * other
        g_other = g * input
        if other.shape != input.shape:
            other_shape = tuple(1 for _ in range(input.ndim - other.ndim)) + other.shape
            broadcast_axes = [
                i
                for i, (s1, s2) in enumerate(zip(input.shape, other_shape))
                if s2 == 1 and s1 != 1
            ]
            g_other = g_other.sum(axis=tuple(broadcast_axes), keepdims=False)
        g_other = g_other.reshape(other.shape)
        return [g_input, g_other], []


class TO(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []


class VIEW(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp15-(0,)(name=view, inputs=(t506(p32,(32, 128, 4096),d(),v(0/1)),), outputs=(t511(p42,(32, 128, 32, 128),d(),v(0/1)),)) {'size': (32, 128, 32, 128)}
        # FwOp32-(0,)(name=view, inputs=(t529(p81,(32, 128, 32, 128),d(),v(0/1)),), outputs=(t530(p83,(32, 128, 4096),d(),v(0/1)),)) {'size': (32, 128, -1)}
        inshape = unique(inshapes)
        org_inshape = unique(org_inshapes)
        org_outshape = kwargs["size"]
        dim_mapping = get_dim_mapping(org_inshape, org_outshape)

        op_id: str = str(uuid4())
        outshape = []
        cons = []
        for in_idxs, out_idxs in dim_mapping:
            assert len(in_idxs) == 1 or len(out_idxs) == 1
            # reshape validity
            if len(in_idxs) > 1:
                # reshape semantics
                outshape.append(np.prod([inshape[d] for d in in_idxs]))
                for d in in_idxs:
                    cons.append(inshape[d] >= min(BINOP_RX_MIN_UNIT, org_inshape[d]))
            elif len(out_idxs) > 1:
                new_full_outshape = create_z3_tensor(
                    (len(org_outshape),), f"view({op_id}).out.d", z3.Int, ctx
                )
                # reshape semantics
                outshape.extend([new_full_outshape[d] for d in out_idxs])
                for d in out_idxs:
                    cons.append(outshape[d] >= min(BINOP_RX_MIN_UNIT, org_outshape[d]))
                cons.append(
                    inshape[unique(in_idxs)] == np.prod([outshape[d] for d in out_idxs])
                )
            else:
                assert len(in_idxs) == len(out_idxs) == 1
                outshape.append(inshape[unique(in_idxs)])
        return [np.array(outshape)], cons

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x: SymbTensor = unique(insts)
        outshape = shapes[unique(G.node_outputs(node))]
        return [x.reshape(outshape)], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        x: SymbTensor = unique(mirror_insts)
        return [g.reshape(x.shape)], []


class APPLY_ROTARY_EMB(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp18-(0,)(name=apply_rotary_emb, inputs=(t511(p42,(32, 128, 32, 128),d(),v(0/1)), t512(p44,(32, 128, 32, 128),d(),v(0/1)), w514(p48,(128, 64),d(),v(0/1))), outputs=(t515(p49,(32, 128, 32, 128),d(),v(0/1)), t516(p51,(32, 128, 32, 128),d(),v(0/1)))) {}
        xq_shape, xk_shape, freqs_shape = inshapes
        assert len(xq_shape) == 4  # (bsz, seq_len, num_heads, head_dim)
        assert len(freqs_shape) == 2  # (seq_len, head_dim//2)

        cons = equalize_z3tensors([xq_shape, xk_shape]) + [
            freqs_shape[0] == xq_shape[1],
            freqs_shape[1] * 2 == xq_shape[3],
            freqs_shape[1] % 2 == 0,
        ]
        xq_out_shape = xq_shape.copy()
        xk_out_shape = xk_shape.copy()
        return [xq_out_shape, xk_out_shape], cons

    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # BwOp253-(0,)(FwOp18, inputs=(g650(p50,(32, 128, 32, 128),d(),v(0/1)), g651(p52,(32, 128, 32, 128),d(),v(0/1))), outputs=(g647(p43,(32, 128, 32, 128),d(),v(0/1)), g648(p45,(32, 128, 32, 128),d(),v(0/1)))) {}
        return inputs[2:], inputs[:2], [*outputs, None]

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        xq, xk, freqs_cis = insts
        # Reshape last dimension into pairs for rotary embedding
        xq_reshaped = xq.reshape(*xq.shape[:-1], -1, 2)
        xk_reshaped = xk.reshape(*xk.shape[:-1], -1, 2)
        # Broadcast freqs_cis to match xq and xk shape
        seqlen, hidden = freqs_cis.shape

        # NOTE: SUPER HACKY, treating cos and sin part the same
        freqs_cis = freqs_cis.reshape(seqlen, hidden)
        freqs_cis = freqs_cis[None, :, None, :, None]
        cos_part = freqs_cis[..., 0]  # Cosine component
        sin_part = freqs_cis[..., 0]  # Sine component

        # Apply rotary embedding
        xq_real = xq_reshaped[..., 0] * cos_part - xq_reshaped[..., 1] * sin_part
        xq_imag = xq_reshaped[..., 0] * sin_part + xq_reshaped[..., 1] * cos_part
        xk_real = xk_reshaped[..., 0] * cos_part - xk_reshaped[..., 1] * sin_part
        xk_imag = xk_reshaped[..., 0] * sin_part + xk_reshaped[..., 1] * cos_part
        # Combine real and imaginary parts into the original shape
        xq_transformed = np.stack([xq_real, xq_imag], axis=-1).reshape(*xq.shape)
        xk_transformed = np.stack([xk_real, xk_imag], axis=-1).reshape(*xk.shape)
        return [xq_transformed, xk_transformed], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        gxq, gxk = insts
        xq, xk, freqs_cis = mirror_insts
        # Reshape last dimension into pairs for rotary embedding
        gxq = gxq.reshape(*gxq.shape[:-1], -1, 2)
        gxk = gxk.reshape(*gxk.shape[:-1], -1, 2)
        # Broadcast freqs_cis to match xq and xk shape

        # NOTE: SUPER HACKY, treating cos and sin part the same; need revisit
        seqlen, hidden = freqs_cis.shape
        freqs_cis = freqs_cis.reshape(seqlen, hidden)
        freqs_cis = freqs_cis[None, :, None, :, None]
        cos_part = freqs_cis[..., 0]  # Cosine component
        sin_part = freqs_cis[..., 0]  # Sine component

        # Gradients w.r.t. xq
        gxq_real = gxq[..., 0]
        gxq_imag = gxq[..., 1]
        g_xq_reshaped_0 = gxq_real * cos_part + gxq_imag * sin_part
        g_xq_reshaped_1 = -gxq_real * sin_part + gxq_imag * cos_part
        # Gradients w.r.t. xk
        gxk_real = gxk[..., 0]
        gxk_imag = gxk[..., 1]
        g_xk_reshaped_0 = gxk_real * cos_part + gxk_imag * sin_part
        g_xk_reshaped_1 = -gxk_real * sin_part + gxk_imag * cos_part
        # Combine reshaped xq and xk gradients into the original shape
        g_xq = np.stack([g_xq_reshaped_0, g_xq_reshaped_1], axis=-1).reshape(*xq.shape)
        g_xk = np.stack([g_xk_reshaped_0, g_xk_reshaped_1], axis=-1).reshape(*xk.shape)
        return [g_xq, g_xk], []


class TRANSPOSE(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp19-(1,)(name=transpose, inputs=(t515(p49,(32, 128, 32, 128),d(),v(0/1)),), outputs=(t517(p57,(32, 32, 128, 128),d(),v(0/1)),)) {'dim0': 1, 'dim1': 2}
        input_shape = unique(inshapes)
        dim0 = kwargs["dim0"]
        dim1 = kwargs["dim1"]

        # output
        output_shape = input_shape.copy()
        output_shape[dim0], output_shape[dim1] = output_shape[dim1], output_shape[dim0]
        return [output_shape], []

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x: SymbTensor = unique(insts)
        axis = list(range(len(x.shape)))
        dim0, dim1 = kwargs["dim0"], kwargs["dim1"]
        axis[dim0], axis[dim1] = axis[dim1], axis[dim0]
        return [np.transpose(x, axis)], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        dim0, dim1 = kwargs["dim0"], kwargs["dim1"]
        axis = list(range(g.ndim))
        axis[dim0], axis[dim1] = axis[dim1], axis[dim0]
        return [np.transpose(g, axis)], []


class MATMUL(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp23-(1,)(name=matmul, inputs=(t517(p57,(32, 32, 128, 128),d(),v(0/1)), t520(p63,(32, 32, 128, 128),d(),v(0/1))), outputs=(t521(p65,(32, 32, 128, 128),d(),v(0/1)),)) {}
        mat1_shape, mat2_shape = inshapes
        mat1_org_shape = org_inshapes[0]
        assert len(mat1_shape) == len(mat2_shape) >= 2

        # constraints
        constraints = equalize_z3tensors([mat1_shape[:-2], mat2_shape[:-2]]) + [
            mat1_shape[-1] == mat2_shape[-2],
            mat1_shape[-1] >= min(BINOP_RX_MIN_UNIT, mat1_org_shape[-1]),
        ]

        # output
        output_shape = mat1_shape.copy()
        output_shape[-1] = mat2_shape[-1]
        return [output_shape], constraints

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        mat1, mat2 = insts
        return [mat1 @ mat2], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        x, w = mirror_insts
        return _bw_matmul(g, x, w), []


class DIV(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp24-(0,)(name=div, inputs=(t521(p65,(32, 32, 128, 128),d(),v(0/1)), 11.313708498984761), outputs=(t522(p67,(32, 32, 128, 128),d(),v(0/1)),)) {'rounding_mode': None}
        if len(inshapes) == 2:
            return _ewdual_w_broadcast_shape_pass(inshapes, org_inshapes)
        else:
            assert len(inshapes) == 1
            return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        input = insts[0]
        other = insts[1] if len(insts) > 1 else unique(kwargs[KW_CONSTS_KEY])
        return [input / other], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        assert len(mirror_insts) == 1, f"DIV.bw require denominator to be constant."
        other = unique(kwargs[KW_CONSTS_KEY])
        return [g / other], []


class APPLY_MASK(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp71-(3,)(name=apply_mask, inputs=(t578(p187,(32, 32, 128, 128),d(),v(0/1)), t778(p777,(1, 1, 128, 128),d(),v(0/1))), outputs=(t579(p189,(32, 32, 128, 128),d(),v(0/1)),)) {}
        x_shape, mask_shape = inshapes
        assert len(x_shape) == len(mask_shape) >= 3

        # constraints
        constraints = equalize_z3tensors([x_shape[2:], mask_shape[2:]]) + [
            mask_shape[0] == z3.IntVal(1, ctx=ctx),
            mask_shape[1] == z3.IntVal(1, ctx=ctx),
        ]

        # output
        output_shape = x_shape.copy()
        return [output_shape], constraints

    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # BwOp305-(3,)(FwOp71, inputs=(g726(p190,(32, 32, 128, 128),d(),v(0/1)),), outputs=(g725(p188,(32, 32, 128, 128),d(),v(0/1)),)) {}
        return inputs[1:], inputs[:1], [*outputs, None]

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x, mask = insts
        assert x.shape[-2:] == mask.shape[-2:]
        assert mask.shape[:-2] == (1, 1)
        return [x + mask], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []

def _softmax(x, dim=-1):
    e_x = np.e**x
    e_x_sum = np.sum(e_x, axis=dim, keepdims=True)
    return e_x / e_x_sum

class SOFTMAX(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp27-(0,)(name=softmax, inputs=(t524(p71,(32, 32, 128, 128),d(),v(0/1)),), outputs=(t525(p73,(32, 32, 128, 128),d(),v(0/1)),)) {'dim': -1, '_stacklevel': 3, 'dtype': None}
        input_shape = unique(inshapes)
        input_org_shape = org_inshapes[0]
        norm_axis = kwargs["dim"]

        # constraints
        cons = [
            input_shape[norm_axis] >= min(BINOP_RX_MIN_UNIT, input_org_shape[norm_axis])
        ]

        # output
        output_shape = input_shape.copy()
        return [output_shape], cons

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        input = unique(insts)
        dim = kwargs["dim"]
        return [_softmax(input, dim)], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        x: SymbTensor = unique(mirror_insts)
        dim = kwargs["dim"]
        softmax_output = _softmax(x, dim)
        g_input = softmax_output * (
            g - np.sum(g * softmax_output, axis=dim, keepdims=True)
        )
        return [g_input], []


class CONTIGUOUS(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp31-(0,)(name=contiguous, inputs=(t528(p79,(32, 128, 32, 128),d(),v(0/1)),), outputs=(t529(p81,(32, 128, 32, 128),d(),v(0/1)),)) {}
        return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []

def _sigmoid(x):
    return 1 / (1 + np.e**-x)

class SILU(SymbolicOperator):

    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        x = unique(insts)
        return [_sigmoid(x)], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        x: SymbTensor = unique(mirror_insts)
        sigmoid = _sigmoid(x)
        grad = sigmoid * (1 + x * (1 - sigmoid))
        return [g * grad], []


class SUM(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp105-(2,)(name=sum, inputs=(t619(p269,(32, 128, 51200),d(),v(0/1)),), outputs=(t492(p271,(1,),d(),v(0/1)),)) {}
        # constraints
        input_shape = unique(inshapes)
        input_org_shape = org_inshapes[0]

        cons = [
            d >= min(BINOP_RX_MIN_UNIT, org_d)
            for d, org_d in zip(input_shape, input_org_shape)
        ]

        output_shape = np.array([z3.IntVal(1, ctx=ctx)])
        return [output_shape], cons

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        input = unique(insts)
        return [np.sum(input, keepdims=True).reshape((1,))], []

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        g: SymbTensor = unique(insts)
        assert g.shape == (1,)
        input: SymbTensor = unique(mirror_insts)
        return [np.full(input.shape, g)], []


class IDENTITYPRIM(SymbolicOperator):

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)
    
    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []


class ALLREDUCEPRIM(SymbolicOperator):

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return [sum(insts)], []


class MOVEPRIM(SymbolicOperator):

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return insts, []


class CHUNKPRIM(SymbolicOperator):

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # [t907(p6,(1, 1, 128, np.int64(64)),d(3,),v(0/1))] = split[[0]]([t495(p6,(1, 1, 128, 128),d(),v(0/1))], dim=3) {'dim': 3, 'ranks': [0, 1]}
        dim = kwargs["dim"]
        ranks = kwargs["ranks"]
        npart = len(ranks)
        input_shape = unique(inshapes)

        cons = [input_shape[dim] % npart == 0]

        output_shape = input_shape.copy()
        output_shape[dim] /= npart
        return [output_shape], cons

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        dim = kwargs["dim"]
        ranks = kwargs["ranks"]
        i = kwargs[KW_CC_IDX]

        # i cannot be self's rank index within ranks, because order can be shuffled
        return [_chunk(tensor=unique(insts), nsplit=len(ranks), dim=dim, i=i)], []


class ALLGATHERPRIM(SymbolicOperator):

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # [g695(p128,(32, 128, np.int64(4096)),d(),v(0/1))] = all_gather[[0]]([g955(p128,(32, 128, np.int64(2048)),d(2,),v(0/1))]) {'dim': 2, 'ranks': [0, 1]}
        # [Tensor(wtype='P', rank=0, mb=0, tid=955, v=1), Tensor(wtype='P', rank=1, mb=0, tid=956, v=1)]
        # [Tensor(wtype='P', rank=0, mb=0, tid=695, v=1)]
        dim = kwargs["dim"]
        ranks = kwargs["ranks"]
        npart = len(ranks)

        cons = equalize_z3tensors(inshapes)

        output_shape = inshapes[0].copy()
        output_shape[dim] *= npart
        return [output_shape], cons

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        dim = kwargs["dim"]
        return [np.concatenate(insts, dim)], []


class LOCAL_GRAD_ACCUMULATION(SymbolicOperator):

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        return _identity_shape_pass(inshapes)

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        return [sum(insts)], []


class REDUCER(SymbolicOperator):
    cached_shape_result = {}
    cached_op_result = {}

    def z3_default_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        input_hash = tuple(sorted([hash(tuple(shape.tolist())) for shape in inshapes]))
        if input_hash not in self.cached_shape_result:
            self.cached_shape_result[input_hash] = _identity_shape_pass(inshapes)
        return self.cached_shape_result[input_hash]

    def z3_default_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        input_hash = hash(tuple(sorted(G.node_inputs(node))))
        if input_hash not in self.cached_op_result:
            self.cached_op_result[input_hash] = [sum(insts)]
        return self.cached_op_result[input_hash], []


class GATE_FW(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp91-(0,)(name=gate_fw, inputs=(t995(p993,(131072, 512),d(),v(0/1)), w640(p234,(16, 512),d(),v(0/1))), outputs=(t641(p236,(131072, 2),d(),v(0/1)), t642(p238,(131072, 2),d(),v(0/1)))) {'score_func': 'sigmoid', 'bias': None, 'n_groups': 8, 'topk_groups': 4, 'topk': 2, 'route_scale': 2.5, '__consts': []}
        x_shape, w_shape = inshapes
        topk = kwargs["topk"]
        
        constraints = [
            x_shape[-1] == w_shape[-1],
            w_shape[0] == G.W.n_routed_experts
        ]

        output_shape = x_shape.copy()
        output_shape[1] = z3.IntVal(topk, ctx=ctx)
        return [output_shape, output_shape], constraints
    
    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """ BwOp339-(2,)(FwOp91, inputs=(g806(p237,(32768, 2),d(),v(0/1)),), outputs=(g1018(p999,(32768, 512),d(),v(0/1)), g805(p235,(16, 512),d(),v(0/1)))) {'score_func': 'sigmoid', 'bias': None, 'n_groups': 8, 'topk_groups': 4, 'topk': 2, 'route_scale': 2.5, '__consts': []} 
        [Tensor(wtype='p', rank=2, mb=1, tid=806, v=1), Tensor(wtype='p', rank=2, mb=1, tid=1006, v=1), Tensor(wtype='p', rank=2, mb=-1, tid=640, v=0)] 
        [Tensor(wtype='p', rank=2, mb=1, tid=1018, v=1), Tensor(wtype='p', rank=2, mb=-1, tid=805, v=2)] """
        return inputs[1:], inputs[:1] + [None], outputs[:]
        

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        """
        NOTE: Due to difficulty of expressing topk (which performs max / argmax) using symbolics, we define topk as z3 uninterpreted function, and declare the properties for checking its equivalence. The re-written pass also simplifies the grouping logic, so the rewritten gate_fw is per-token exact topk. The following code is original torch implementation of the custom gate_fw.
        
        @nnscaler.register_op(f'a^ h^, e^ h^ -> a^ k^, a^ k^')
        def gate_fw(x: torch.Tensor, weight: torch.Tensor, score_func, bias, n_groups, topk_groups, topk, route_scale):
            scores = linear(x, weight)
            if score_func == "softmax":
                scores = scores.softmax(dim=-1, dtype=torch.float32)
            else:
                scores = scores.sigmoid()
            original_scores = scores
            if bias is not None:
                scores = scores + bias
            if n_groups > 1:
                scores = scores.view(x.size(0), n_groups, -1)
                if bias is None:
                    group_scores = scores.amax(dim=-1)
                else:
                    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
                indices = group_scores.topk(topk_groups, dim=-1)[1]
                mask = scores.new_ones(x.size(0), n_groups, dtype=bool).scatter_(1, indices, False)
                scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
            indices = torch.topk(scores, topk, dim=-1)[1]
            weights = original_scores.gather(1, indices)
            if score_func == "sigmoid":
                weights /= weights.sum(dim=-1, keepdim=True)
            weights *= route_scale
            return weights.type_as(x), indices
        """
        
        TopK_idx, cons1 = getuop_topk_indices(ctx)
        TopK_val, cons2 = getuop_topk_values(ctx)
        
        def topk_idx_of_elements(arr, k):
            infinite_arr = TopK_idx(arr, z3.IntVal(k, ctx=ctx))
            return z3arr_to_row(infinite_arr, k)
        
        def topk_val_of_elements(arr, k):
            infinite_arr = TopK_val(arr, z3.IntVal(k, ctx=ctx))
            return z3arr_to_row(infinite_arr, k)
            
        x, w = insts
        score_func = kwargs["score_func"]
        bias = kwargs["bias"]
        route_scale = kwargs["route_scale"]
        topk = kwargs["topk"]
        
        scores = x @ w.T
        if score_func == "softmax":
            scores = _softmax(scores, dim=-1)
        else:
            scores = _sigmoid(scores)
        if bias is not None:
            scores = scores + bias
            
        indices = []
        weights = []
        for row in scores:
            arr_z3 = row_to_z3arr(row)
            indices.append(topk_idx_of_elements(arr_z3, topk))
            weights.append(topk_val_of_elements(arr_z3, topk))
        indices = np.array(indices, dtype=object)
        weights = np.array(weights, dtype=object)
        if score_func == "sigmoid":
            weights /= weights.sum(axis=-1, keepdims=True)
        weights *= route_scale
        return [weights, indices], cons1 + cons2

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        # uops
        TopK_idx, cons1 = getuop_topk_indices(ctx)
        TopK_val, cons2 = getuop_topk_values(ctx)
        
        def topk_idx_of_elements(arr, k):
            infinite_arr = TopK_idx(arr, z3.IntVal(k, ctx=ctx))
            return z3arr_to_row(infinite_arr, k)
        
        def topk_val_of_elements(arr, k):
            infinite_arr = TopK_val(arr, z3.IntVal(k, ctx=ctx))
            return z3arr_to_row(infinite_arr, k)
        
        # variables
        grad_output = unique(insts)
        x, w = mirror_insts
        B = x.shape[0]
        score_func = kwargs["score_func"]
        bias = kwargs["bias"]
        route_scale = kwargs["route_scale"]
        topk = kwargs["topk"]
        n_global_experts = w.shape[0]
        
        # fw
        raw_scores = x @ w.T
        if score_func == "softmax":
            scores = _softmax(raw_scores, dim=-1)
        else:
            scores = _sigmoid(raw_scores)
        if bias is not None:
            scores = scores + bias
            
        indices = []
        weights = []
        indices_onehot = []
        weights_onehot = []
        for row in scores:
            arr_z3 = row_to_z3arr(row)
            indices.append(topk_idx_of_elements(arr_z3, topk))
            weights.append(topk_val_of_elements(arr_z3, topk))
            indices_onehot.append(topk_idx_of_elements(arr_z3, n_global_experts))
            weights_onehot.append(topk_val_of_elements(arr_z3, n_global_experts))
        indices = np.array(indices, dtype=object)
        weights = np.array(weights, dtype=object)
        indices_onehot = np.array(indices_onehot, dtype=object)
        weights_onehot = np.array(weights_onehot, dtype=object)
        if score_func == "sigmoid":
            weights /= weights.sum(axis=-1, keepdims=True)
        weights *= route_scale
        
        # bw
        grad_scores = np.zeros_like(raw_scores)  # (B, M)
        
        # Fill in only the top-k expert positions
        for b in range(B):
            grad_row = row_to_z3arr(grad_output[b])
            grad_row_onehot = topk_val_of_elements(grad_row, n_global_experts)
            grad_scores[b] = grad_row_onehot * indices_onehot[b] * route_scale  # dL/dscore_j

        # If softmax was used
        if score_func == "softmax":
            softmax_out = _softmax(raw_scores, dim=-1)  # (B, M)
            grad_softmax = grad_scores  # upstream gradient

            # Jacobian-vector product for softmax
            # dL/dz_i = sum_j dL/dy_j * dy_j/dz_i
            grad_scores = softmax_out * (
                grad_softmax - (grad_softmax * softmax_out).sum(dim=-1, keepdim=True)
            )

        elif score_func == "sigmoid":
            sigmoid_out = _sigmoid(raw_scores)  # (B, M)
            grad_scores = grad_scores * sigmoid_out * (1 - sigmoid_out)  # elementwise

        # ∂L/∂x = grad_scores @ w    -- shape (B, D)
        grad_x = grad_scores @ w

        # ∂L/∂w = grad_scoresᵀ @ x   -- shape (M, D)
        grad_w = grad_scores.T @ x

        ret = [grad_x, grad_w]
        # ∂L/∂bias = grad_scores.sum(dim=0)  -- optional
        if bias is not None:
            grad_bias = grad_scores.sum(dim=0)
            ret.append(grad_bias)
        return ret, []



class NNSCALER_MOE_GMM(SymbolicOperator):
    def z3_fw_shape_pass(
        self,
        inshapes: List[SymbShape],
        org_inshapes: List[Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbShape], List[SymbExpr]]:
        # FwOp403-(2,)(name=nnscaler_moe_gmm, inputs=(t1007(p1000,(32768, 512),d(),v(0/1)), t642(p238,(32768, 2),d(),v(0/1)), t641(p236,(32768, 2),d(),v(0/1)), w1110(p242,(8, 1024, 512),d(0,),v(0/1)), w1112(p244,(8, 1024, 512),d(0,),v(0/1)), w1114(p246,(8, 512, 1024),d(0,),v(0/1))), outputs=(t1116(p248,(32768, 512),d(),v(0/2)),)) {'n_routed_experts': 16, 'local_expert_start': 0, 'local_expert_end': 8, '__consts': []}
        # 'a h^, a k, a k, E+ d+ h^, E+ d+ h^, E+ h^ d+ -> a h^'
        x_shape, topkidx_shape, topkw_shape, gateproj_shape, upproj_shape, downproj_shape = inshapes
        x_a, x_h = x_shape
        i_a, i_k = topkidx_shape
        w_a, w_k = topkw_shape
        w11_e, w11_d, w11_h = gateproj_shape
        w33_e, w33_d, w33_h = upproj_shape
        w22_e, w22_h, w22_d = downproj_shape
        
        x_orgshape, topkidx_orgshape, _, w11_orgshape, _, _ = org_inshapes
        
        constraints = [
            x_a == i_a, x_a == w_a, # a
            x_h == w11_h, x_h == w22_h, x_h == w33_h, # h
            i_k == w_k, # k
            i_k == z3.IntVal(topkidx_orgshape[1], ctx=ctx),
            w11_e == w22_e, w11_e == w33_e, # e
            w11_d == w22_d, w11_d == w33_d, # d
            x_h >= min(BINOP_RX_MIN_UNIT, x_orgshape[1]), # h >= 2
            w11_d >= min(BINOP_RX_MIN_UNIT, w11_orgshape[1]), # d >= 2
            w11_e == z3.IntVal(w11_orgshape[0], ctx=ctx)
        ]

        output_shape = x_shape.copy()
        return [output_shape], constraints
    
    def bw_parse_dataflow(
        self, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        BwOp133-(0,)(FwOp92, inputs=(g810(p249,(131072, 512),d(),v(0/1)),), outputs=(g1000(p998,(131072, 512),d(),v(0/1)), g806(p237,(131072, 2),d(),v(0/1)), g807(p243,(16, 1024, 512),d(),v(0/1)), g808(p245,(16, 1024, 512),d(),v(0/1)), g809(p247,(16, 512, 1024),d(),v(0/1)))) {'n_routed_experts': 16, 'local_expert_start': 0, 'local_expert_end': 16, '__consts': []}
        [Tensor(wtype='s', rank=0, mb=0, tid=810, v=1), Tensor(wtype='s', rank=0, mb=0, tid=999, v=1), Tensor(wtype='s', rank=0, mb=0, tid=642, v=1), Tensor(wtype='s', rank=0, mb=0, tid=641, v=1), Tensor(wtype='s', rank=0, mb=-1, tid=643, v=0), Tensor(wtype='s', rank=0, mb=-1, tid=644, v=0), Tensor(wtype='s', rank=0, mb=-1, tid=645, v=0)]
        [Tensor(wtype='s', rank=0, mb=0, tid=1000, v=1), Tensor(wtype='s', rank=0, mb=0, tid=806, v=1), Tensor(wtype='s', rank=0, mb=-1, tid=807, v=1), Tensor(wtype='s', rank=0, mb=-1, tid=808, v=1), Tensor(wtype='s', rank=0, mb=-1, tid=809, v=1)]
        """
        return inputs[1:], inputs[:1], outputs[:1] + [None] + outputs[1:]

    def z3_fw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        """
        def nnscaler_moe_gmm(
            hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, 
            gate_projs: torch.Tensor, up_projs: torch.Tensor, down_projs: torch.Tensor,
            n_routed_experts: int, local_expert_start: int, local_expert_end: int):
            
            orig_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])

            with torch.no_grad():
                local_mask = (topk_idx >= local_expert_start) & (topk_idx < local_expert_end)
                local_idx = topk_idx.masked_select(local_mask)

            local_prob = topk_weight.masked_select(local_mask)
            local_prob = local_prob.view(-1, 1)
            local_map = local_mask.nonzero()[:, 0]
            local_map = local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = moe_gather.apply(hidden_states, local_map)

            with torch.no_grad():
                tokens_per_expert = torch.histc(local_idx, bins=local_expert_end - local_expert_start, min=local_expert_start, max=local_expert_end - 1)
                tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

            permuted_inputs, row_id_map = permute(local_hidden_states, local_idx)

            fc1_output = gmm(permuted_inputs, gate_projs, tokens_per_expert, trans_b=True)
            fc2_output = gmm(permuted_inputs, up_projs, tokens_per_expert, trans_b=True)
            intermediate_parallel = torch.nn.functional.silu(fc1_output) * fc2_output
            expert_outs = gmm(intermediate_parallel, down_projs, tokens_per_expert, trans_b=True)

            y = unpermute(expert_outs, row_id_map)
            y = y * local_prob
            y = moe_scatter.apply(y, local_map, hidden_states.shape)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            return y
        """
        IdxOneHot, cons1 = getuop_idx2onehot(ctx)
        ValOneHot, cons2 = getuop_val2onehot(ctx)
        
        hidden_states, topk_idx, topk_w, gate_projs, up_projs, down_projs = insts
        n_routed_experts = kwargs["n_routed_experts"]
        local_expert_start = kwargs["local_expert_start"]
        local_expert_end = kwargs["local_expert_end"]
        
        output = []
        for i, (embedding, row_topk_i, row_topk_w) in enumerate(zip(hidden_states, topk_idx, topk_w)):
            # Example: assume n_routed_expert = 4 and n_activate_expert (topk) = 2
            # then a selection e.g. (0,3) converts to one hot (1,0,0,1)
            # if the local expert has the shard exp0,exp1, then the local mask is (1,0)
            # The IdxOneHot is an uninterpret function which encodes the property that
            # given the same selection, the extended onehot mapping is the same
            z3arr_mask = IdxOneHot(row_to_z3arr(row_topk_i))
            local_mask = z3arr_to_row(z3arr_mask, n_routed_experts)[local_expert_start:local_expert_end]
            
            # Similarly for the uninterpreted ValOneHot, which encodes equivalence
            # propagation given the same local selection and topk_w
            z3arr_probs = ValOneHot(row_to_z3arr(row_topk_i),row_to_z3arr(row_topk_w))
            local_prob = z3arr_to_row(z3arr_probs, n_routed_experts)[local_expert_start:local_expert_end]
            
            embedding = embedding.reshape(1,-1)
            fc1_output = embedding @ gate_projs.transpose(0, 2, 1)
            fc2_output = embedding @ up_projs.transpose(0, 2, 1)
            silu = lambda x: x / (1 + np.e**-x)
            intermediate_parallel = silu(fc1_output) * fc2_output
            experts_out = intermediate_parallel @ down_projs.transpose(0, 2, 1)
            n_local_experts = len(local_mask)
            masked_experts_out = experts_out * (local_mask*local_prob).reshape(n_local_experts,1,1)
            reduced = np.sum(masked_experts_out, axis=0)
            output.append(reduced)
        output = np.concatenate(output, axis=0)
        assert output.shape == hidden_states.shape, f"{output.shape} != {hidden_states.shape}"
        return [output], cons1+cons2

    def z3_bw_op_pass(
        self,
        node: Node,
        insts: List[SymbTensor],
        mirror_insts: List[SymbTensor],
        shapes: Dict[Tensor, Shape],
        kwargs: Dict,
        G: DFG,
        ctx: z3.Context,
    ) -> Tuple[List[SymbTensor], List[SymbExpr]]:
        IdxOneHot, cons1 = getuop_idx2onehot(ctx)
        ValOneHot, cons2 = getuop_val2onehot(ctx)
        
        input_grads = unique(insts)
        hidden_states, topk_idxs, topk_weights, gate_projs, up_projs, down_projs = mirror_insts
        n_routed_experts = kwargs["n_routed_experts"]
        local_expert_start = kwargs["local_expert_start"]
        local_expert_end = kwargs["local_expert_end"]
        n_local_experts = local_expert_end - local_expert_start
        
        assert input_grads.shape == hidden_states.shape
        zero_val = z3.IntVal(0, ctx=ctx)
        g_hidden_states = np.full(hidden_states.shape, zero_val, dtype=object)
        g_weights = np.full(topk_weights.shape, zero_val, dtype=object)
        g_gate_projs = np.full(gate_projs.shape, zero_val, dtype=object)
        g_up_projs = np.full(up_projs.shape, zero_val, dtype=object)
        g_down_projs = np.full(down_projs.shape, zero_val, dtype=object)
        
        for i, (g, embedding, row_topk_i, row_topk_w) in enumerate(zip(input_grads, hidden_states, topk_idxs, topk_weights)):
            z3arr_mask = IdxOneHot(row_to_z3arr(row_topk_i))
            local_mask = z3arr_to_row(z3arr_mask, n_routed_experts)[local_expert_start:local_expert_end]
            
            z3arr_probs = ValOneHot(row_to_z3arr(row_topk_i),row_to_z3arr(row_topk_w))
            local_prob = z3arr_to_row(z3arr_probs, n_routed_experts)[local_expert_start:local_expert_end]
            
            g = g.reshape(1,-1)
            embedding = embedding.reshape(1,-1)
            fc1_output = embedding @ gate_projs.transpose(0, 2, 1)
            fc2_output = embedding @ up_projs.transpose(0, 2, 1)
            # silu = lambda x: x / (1 + np.e**-x)
            # intermediate_parallel = silu(fc1_output) * fc2_output
            # rid silu to reduce nonlinear complexity for solver
            intermediate_parallel = fc1_output * fc2_output
            experts_out = intermediate_parallel @ down_projs.transpose(0, 2, 1)
            masked_experts_out = experts_out * (local_mask*local_prob).reshape(n_local_experts,1,1)
            reduced = np.sum(masked_experts_out, axis=0)
            
            g_reduced = g; assert g_reduced.shape == reduced.shape
            g_meo = np.array([g_reduced for _ in range(n_local_experts)]); assert g_meo.shape == masked_experts_out.shape
            g_w = np.sum(g_meo*experts_out, axis=(1,2))*local_mask; assert g_w.shape == local_prob.shape
            g_eo = g_meo * (local_mask*local_prob).reshape(n_local_experts,1,1); assert g_eo.shape == experts_out.shape
            g_down = g_eo.transpose(0, 2, 1) @ intermediate_parallel; assert g_down.shape == down_projs.shape
            g_ip = g_eo @ down_projs; assert g_ip.shape == intermediate_parallel.shape
            # skip silu to reduce nonlinear solver complexity; sound, as self-element op does not effect data flow
            g_fc1 = g_ip * fc2_output; assert g_fc1.shape == fc1_output.shape
            g_fc2 = g_ip * fc1_output; assert g_fc2.shape == fc2_output.shape
            g_gate = g_fc1.transpose(0, 2, 1) @ embedding; assert g_gate.shape == gate_projs.shape
            g_up = g_fc2.transpose(0, 2, 1) @ embedding; assert g_up.shape == up_projs.shape
            g_emb = np.sum(g_fc1@gate_projs+g_fc2@up_projs, axis=0); assert g_emb.shape == embedding.shape
            
            g_hidden_states[i] = g_emb
            # g_weights[i] = g_w # TODO: current g_w is problematic; skip temporarily
            g_gate_projs += g_gate
            g_up_projs += g_up
            g_down_projs += g_down
        return [g_hidden_states, g_weights, g_gate_projs, g_up_projs, g_down_projs], cons1 + cons2



OP_MAPPING = {
    name: cls
    for name, cls in globals().items()
    if isinstance(cls, type) and name.isupper()
}


def get_op(opname: OpName) -> SymbolicOperatorAbstract:
    cls_name = opname.value[0].upper()
    if cls_name not in OP_MAPPING:
        raise NotImplementedError(cls_name)
    return OP_MAPPING[cls_name]()
