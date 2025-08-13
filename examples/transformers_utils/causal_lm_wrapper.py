#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from nnscaler.cli.trainer_args import AggregatedOutputs

from .chunk_linear_cross_entropy import chunk_linear_cross_entropy


IGNORE_IDX = -100


class WrapperModel(torch.nn.Module):
    def __init__(self, model_id, config=None, enable_chunk_loss=False, attn_implementation='flash_attention_2'):
        super().__init__()
        if isinstance(config, str):
            config = AutoConfig.from_pretrained(config)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, config=config, attn_implementation=attn_implementation)
        self.enable_chunk_loss = enable_chunk_loss

    def forward(self, samples):
        if self.enable_chunk_loss:
            outputs = self.model.model(
                **samples['net_input'],
                use_cache=False,
                return_dict=False,
            )
            hidden_states = outputs[0]
            losses = chunk_linear_cross_entropy(hidden_states, self.model.lm_head.weight, samples['target'], IGNORE_IDX, 1024)
            loss = torch.sum(losses)
        else:
            outputs = self.model(
                **samples['net_input'],
                use_cache=False,
                return_dict=False,
            )
            logits = outputs[0].view(-1, outputs[0].size(-1))
            labels = samples['target'].view(-1)
            normalized_logits = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = torch.nn.functional.nll_loss(normalized_logits, labels, reduction='sum', ignore_index=IGNORE_IDX)
        return loss, loss.data, samples['ntokens'], samples['nsentences']


def aggregate_outputs_fn(loss_outputs, sync_group) -> AggregatedOutputs:
    losses, ntokens_info = [], []
    for _, loss, ntokens, _ in loss_outputs:
        losses.append(loss)
        ntokens_info.append(ntokens)

    loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
    torch.distributed.all_reduce(loss_sum, group=sync_group)
    ntokens_sum = torch.sum(torch.tensor(ntokens_info, dtype=torch.float64, device=torch.cuda.current_device()))
    torch.distributed.all_reduce(ntokens_sum, group=sync_group)
    num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
    torch.distributed.all_reduce(num_batches, group=sync_group)

    return AggregatedOutputs(
        loss_sum=loss_sum.item() / ntokens_sum.item(),
        num_batches=num_batches.item(),
        num_tokens=ntokens_sum.item(),
    )
