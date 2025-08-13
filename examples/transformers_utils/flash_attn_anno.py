#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Optional

from nnscaler.ir import IRTensor
from nnscaler.graph.parser.register import register_op
from transformers.modeling_flash_attention_utils import _flash_attention_forward

import torch


def flash_attention_anno(
        query_states: IRTensor,
        key_states: IRTensor,
        value_states: IRTensor,
        attention_mask: Optional[IRTensor],
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[IRTensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
        *args,
        **kwargs
        # added >= 4.47
        # cu_seq_lens_q: Optional[IRTensor] = None,
        # cu_seq_lens_k: Optional[IRTensor] = None,
        # max_length_q: Optional[int] = None,
        # max_length_k: Optional[int] = None,
        # target_dtype: Optional[torch.dtype] = None
    ) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'
    input_anno = f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^'

    if isinstance(attention_mask, IRTensor):
        # add attention mask annotation
        input_anno += ', b l^'
        if isinstance(position_ids, IRTensor):
            # add position_ids annotation
            input_anno += ', ?, ?, ?, b l^'
    elif isinstance(position_ids, IRTensor):
        # add position_ids annotation
        input_anno += ', ?, ?, ?, ?, b l^'

    if 'cu_seq_lens_q' in kwargs:
        cu_seq_lens_q = kwargs['cu_seq_lens_q']
        cu_seq_lens_k = kwargs['cu_seq_lens_k']
        assert not isinstance(cu_seq_lens_k, IRTensor) and not isinstance(cu_seq_lens_q, IRTensor), f'cu_seq_lens_k: {cu_seq_lens_k}, cu_seq_lens_q: {cu_seq_lens_q}, not supported'

    return f'{input_anno} -> b l^ {q_anno} vd^'


register_op(flash_attention_anno)(_flash_attention_forward)


# Copy from transformers/integrations/flash_attention.py
# To solve the issue of transformers/integrations/flash_attention.py using relative import _flash_attention_forward,
# and the anno issue mentioned in the following code snippet.
from typing import Optional, Tuple
import torch
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import modeling_flash_attention_utils

_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    # NNScaler: can not use for example, ```query_length=seq_len```, will case anno error,
    # all inputs that have annotation should not use xxx_name=xxx format
    attn_output = modeling_flash_attention_utils._flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        seq_len,
        module.is_causal,
        dropout,
        kwargs.pop("position_ids", None),
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None


ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
