# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import cube


@cube.graph.parser.register('L^ N E^, H+ E^, H+, E^ H+ -> L^ N E^', name='feedforward')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor,
                dropout: float,
                is_training: bool = True) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, is_training, False)
    x = torch.nn.functional.linear(x, proj2, None)
    return x


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = feedforward(x, self.proj1, self.proj1_bias,
                        self.proj2, self.dropout, self.training)
        x = x + self.proj2_bias
        return x
