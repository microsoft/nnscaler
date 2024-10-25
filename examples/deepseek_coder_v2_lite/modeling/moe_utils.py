#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file is adapted from the Megatron-LM project.

import torch


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each
       token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of
                                [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the
                                        capacity factor, should equal the number of tokens not
                                        dropped. By default, set to None, meaning no tokens are
                                        dropped.
        padded_mode (bool, optional): If True, indicating the indices are padded to
                                      [num_expert, capacity] to denote selected tokens per expert.
                                      Defaults to False.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        indices = indices.unsqueeze(1)

    topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    moe_gather_indices = (sorted_indices // topk).unsqueeze(1).expand(-1, tokens.size(-1))
    permuted_tokens = moe_gather.apply(tokens, moe_gather_indices)

    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the
    tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): 2D tensor [num_tokens*topk, hidden]. The tensor of permuted
                                        tokens to be unpermuted.
        sorted_indices (torch.Tensor): 1D tensor [num_tokens*topk]. The tensor of sorted indices
                                       used to unpermute the tokens.
        probs (torch.Tensor, optional): 2D tensor [num_tokens, topk]. The tensor of probabilities
                                        corresponding to the permuted tokens. If provided,
                                        the unpermuted tokens will be merged with their respective
                                        probabilities.
        padded_mode (bool, optional): If True, indicating the indices are padded to
                                      [num_expert, capacity] to denote selected tokens per expert.
                                      Defaults to False.
        restore_shape (torch.Size, optional): The input shape before permutation, only used in
                                              padding mode. Defaults to None.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(
        0
    ), f"Got {sorted_indices.numel()} != {permuted_tokens.size(0)}."
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        assert probs.dim() == 2, f"Expected 2D tensor for probs, got {probs.dim()} dims."
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    output_size = [num_unpermuted_tokens, permuted_tokens.shape[-1]]
    moe_scatter_indices = sorted_indices.unsqueeze(1).expand(-1, permuted_tokens.size(-1))
    unpermuted_tokens = moe_scatter.apply(permuted_tokens, moe_scatter_indices, output_size)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode.
       The input indices shape is [num_expert, capacity], it indicates which tokens were selected
       by each expert separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected
                                tokens for each expert.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

    return permuted_tokens, indices


def unpermute_with_padded_tokens(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their
    corresponding probabilities.

    This function takes a tensor of permuted tokens and reorders them according to the provided
    indices. It also combines the tokens with their associated probabilities.

    Parameters:
        permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected
                                tokens for each expert.
        probs (torch.Tensor): A tensor with the same shape as indices, containing probabilities
                              corresponding to each token.
        restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

    Returns:
        torch.Tensor: A tensor of unpermuted tokens, merged with their probabilities.

    """
    # Ensure permuted_tokens is 2D
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

    # Reshape and expand probabilities and indices to match permuted_tokens
    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    assert (
        permuted_tokens.shape == indices.shape
    ), "Shape mismatch between permuted_tokens and indices."

    # Combine tokens with their probabilities
    combined_output = probs * permuted_tokens

    # Prepare a tensor of zeros with the desired output shape
    empty_tokens = torch.zeros(
        restore_shape, dtype=combined_output.dtype, device=combined_output.device
    )

    # Scatter the combined tokens back to their original positions
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens


class moe_gather(torch.autograd.Function):
    """Gather the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_):
        """Gather the input tensor based on the map tensor."""
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter the grad_output tensor based on the map tensor."""
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(
            input_size, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class moe_scatter(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """Scatter the input tensor based on the map tensor."""
        ctx.map = map_

        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=input_.device)
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None
