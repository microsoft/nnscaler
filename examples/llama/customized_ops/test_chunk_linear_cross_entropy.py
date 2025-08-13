#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from chunk_linear_cross_entropy import chunk_linear_cross_entropy, linear_cross_entropy


def test_chunk_linear_cross_entropy(
        bsz: int,
        seq_len: int,
        hidden_size: int,
        dict_size: int,
        dtype: torch.dtype,
        chunk_size: int = 1024,
        padding_idx: int = 1):
    print(f'test chunk linear cross entropy with {dtype}')
    device = torch.device('cuda')

    x = torch.randn(bsz, seq_len, hidden_size, dtype=dtype, device=device)
    x1 = x.clone().detach()
    w = torch.nn.Parameter(torch.randn(dict_size, hidden_size, dtype=dtype, device=device))
    w1 = w.clone().detach().requires_grad_(True)
    y = torch.randint(0, dict_size, (bsz, seq_len), dtype=torch.long, device=device)
    y1 = y.clone().detach()

    x1 = x1.reshape(bsz * seq_len, hidden_size)
    y1 = y1.reshape(bsz * seq_len)
    bsl_losses = linear_cross_entropy(x1, w1, y1, padding_idx).reshape(bsz, seq_len)
    bsl_loss = bsl_losses.sum()
    bsl_loss.backward()

    test_losses = chunk_linear_cross_entropy(x, w, y, padding_idx, chunk_size)
    test_loss = test_losses.sum()
    test_loss.backward()

    losses_diff = (bsl_losses - test_losses).abs()
    print(f'losses_diff max: {losses_diff.max().item()}, losses_diff mean: {losses_diff.mean().item()}')
    w_grad_diff = (w.grad - w1.grad).abs()
    print(f'w_grad_diff max: {w_grad_diff.max().item()}, w_grad_diff mean: {w_grad_diff.mean().item()}')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_chunk_linear_cross_entropy(2, 4096, 4096, 32000, torch.bfloat16)
    test_chunk_linear_cross_entropy(2, 4096, 4096, 32000, torch.float16)
