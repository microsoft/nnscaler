from enum import Enum

"""If an operator is injected (not an aligned operator), and is uni-directonal (can only appear in either forward or backward pass), then set `None` for the isfw field."""


class OpName(Enum):
    # UNI DIRECTIONAL OPERATOR

    DATALOADER = ("dataloader", True)

    # BI DIRECTIONAL OPERATOR

    FW_linear = ("linear", True)
    BW_linear = ("linear", False)

    FW_sum = ("sum", True)
    BW_sum = ("sum", False)

    FW_add = ("add", True)
    BW_add = ("add", False)

    FW_transpose = ("transpose", True)
    BW_transpose = ("transpose", False)

    FW_layernorm = ("layernorm", True)
    BW_layernorm = ("layernorm", False)

    FW_self_attention = ("self_attention", True)
    BW_self_attention = ("self_attention", False)

    FW_dropout = ("dropout", True)
    BW_dropout = ("dropout", False)

    FW_feedforward = ("feedforward", True)
    BW_feedforward = ("feedforward", False)

    FW_identity = ("identity", True)
    BW_identity = ("identity", False)

    FW_multiref = ("multiref", True)
    BW_multiref = ("multiref", False)

    FW_embedding = ("embedding", True)
    BW_embedding = ("embedding", False)

    FW_float = ("float", True)
    BW_float = ("float", False)

    FW_to = ("to", True)
    BW_to = ("to", False)

    FW_contiguous = ("contiguous", True)
    BW_contiguous = ("contiguous", False)

    FW_create_mask = ("create_mask", True)
    BW_create_mask = ("create_mask", False)

    FW_pow = ("pow", True)
    BW_pow = ("pow", False)

    FW_mean = ("mean", True)
    BW_mean = ("mean", False)

    FW_rsqrt = ("rsqrt", True)
    BW_rsqrt = ("rsqrt", False)

    FW_mul = ("mul", True)
    BW_mul = ("mul", False)

    FW_view = ("view", True)
    BW_view = ("view", False)

    FW_apply_rotary_emb = ("apply_rotary_emb", True)
    BW_apply_rotary_emb = ("apply_rotary_emb", False)

    FW_matmul = ("matmul", True)
    BW_matmul = ("matmul", False)

    FW_div = ("div", True)
    BW_div = ("div", False)

    FW_apply_mask = ("apply_mask", True)
    BW_apply_mask = ("apply_mask", False)

    FW_softmax = ("softmax", True)
    BW_softmax = ("softmax", False)

    FW_silu = ("silu", True)
    BW_silu = ("silu", False)
    
    FW_nns_moe_gate = ("gate_fw", True)
    BW_nns_moe_gate = ("gate_fw", False)
    
    FW_nns_moe_gmm = ("nnscaler_moe_gmm", True)
    BW_nns_moe_gmm = ("nnscaler_moe_gmm", False)
    

    # COMMUNICATION PRIM

    # comm primitives are shared for fw and bw, so set None for isfw
    ChunkPrim = ("ChunkPrim", None)
    MovePrim = ("MovePrim", None)
    AllGatherPrim = ("AllGatherPrim", None)
    AllReducePrim = ("AllReducePrim", None)
    BroadcastPrim = ("BroadcastPrim", None)
    AllToAllPrim = ("AllToAllPrim", None)
    IdentityPrim = ("IdentityPrim", None)

    # VERDICT OP

    LOCAL_GRAD_ACCUM = ("local_grad_accumulation", None)
    CROSS_DP_WRED = ("reducer", None)
