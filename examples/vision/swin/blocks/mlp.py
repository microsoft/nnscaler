#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.nn as nn
import nnscaler


@nnscaler.register_op('B HW^ E^, H+ E^, H+, E^ H+ -> B HW^ E^', name='feedforward')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor, dropout: float) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, True, False)
    x = torch.nn.functional.linear(x, proj2, None)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_w = torch.nn.Parameter(torch.empty(hidden_features, in_features))
        self.fc1_b = torch.nn.Parameter(torch.empty(hidden_features))
        self.fc2_w = torch.nn.Parameter(torch.empty(out_features, hidden_features))
        self.fc2_b = torch.nn.Parameter(torch.empty(out_features))
        self.drop = drop

    def forward(self, x):
        x = feedforward(x, self.fc1_w, self.fc1_b, self.fc2_w, self.drop)
        x = x + self.fc2_b
        x = torch.nn.functional.dropout(x, self.drop, True, False)
        return x
