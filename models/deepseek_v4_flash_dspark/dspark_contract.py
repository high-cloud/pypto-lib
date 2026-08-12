# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side DSpark drafter layout and golden contracts."""

from config import (
    BLOCK_SIZE,
    DSPARK_MARKOV_RANK,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_TOKENS_PADDED,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
)


def validate_dspark_batch(batch: int) -> None:
    """Validate a dynamic single-card drafter batch."""
    if batch not in (4, 8, 12, 16):
        raise ValueError("DSpark drafter batch must be one of {4, 8, 12, 16}")


def validate_markov_vocab(vocab_size: int) -> None:
    """Validate the tiled Markov sampler vocabulary bound."""
    if vocab_size <= 0 or vocab_size > 131072 or vocab_size % 256 != 0:
        raise ValueError("DSpark Markov vocab_size must be a multiple of 256 in (0, 131072]")


def build_query_layout(anchor_token_ids, context_lens):
    """Return padded anchor/noise IDs, positions, and the logical-row mask."""
    import torch

    if anchor_token_ids.ndim != 1 or context_lens.shape != anchor_token_ids.shape:
        raise ValueError("anchor_token_ids and context_lens must be rank-1 tensors with matching shapes")
    batch = int(anchor_token_ids.numel())
    validate_dspark_batch(batch)

    input_ids = torch.full(
        (batch, DSPARK_QUERY_TOKENS_PADDED),
        DSPARK_NOISE_TOKEN_ID,
        dtype=torch.int64,
        device=anchor_token_ids.device,
    )
    input_ids[:, 0] = anchor_token_ids.to(torch.int64)
    offsets = torch.arange(DSPARK_QUERY_TOKENS_PADDED, device=context_lens.device)
    positions = context_lens.to(torch.int32)[:, None] + 1 + offsets.to(torch.int32)[None, :]
    active = offsets[None, :] < DSPARK_QUERY_TOKENS
    active = active.expand(batch, -1).contiguous()
    return input_ids, positions.contiguous(), active


def build_noncausal_swa_metadata(block_table, context_lens):
    """Build padded DSpark slots where all seven queries share one visible row."""
    import torch

    batch = int(context_lens.numel())
    validate_dspark_batch(batch)
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must have shape [batch, max_blocks]")

    indices = torch.full(
        (batch, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH),
        -1,
        dtype=torch.int32,
        device=block_table.device,
    )
    lens = torch.zeros(
        (batch, DSPARK_QUERY_TOKENS_PADDED), dtype=torch.int32, device=block_table.device
    )
    slots = torch.full(
        (batch, DSPARK_QUERY_TOKENS_PADDED), -1, dtype=torch.int64, device=block_table.device
    )

    for request in range(batch):
        context_len = int(context_lens[request])
        start = max(0, context_len + 1 - M.sliding_window)
        end = context_len + 1 + DSPARK_QUERY_TOKENS
        visible = end - start
        if visible > DSPARK_SWA_INDEX_WIDTH:
            raise ValueError("DSpark SWA index width is too small for the visible set")
        for column, logical_position in enumerate(range(start, end)):
            logical_block = logical_position // BLOCK_SIZE
            if logical_block >= block_table.shape[1]:
                raise ValueError("block_table does not cover the DSpark visible positions")
            physical_block = int(block_table[request, logical_block])
            physical_slot = physical_block * BLOCK_SIZE + logical_position % BLOCK_SIZE
            indices[request, :DSPARK_QUERY_TOKENS, column] = physical_slot
        lens[request, :DSPARK_QUERY_TOKENS] = visible
        for query in range(DSPARK_QUERY_TOKENS):
            logical_position = context_len + 1 + query
            logical_block = logical_position // BLOCK_SIZE
            physical_block = int(block_table[request, logical_block])
            slots[request, query] = physical_block * BLOCK_SIZE + logical_position % BLOCK_SIZE

    return (
        slots.reshape(-1).contiguous(),
        indices.reshape(-1, DSPARK_SWA_INDEX_WIDTH).contiguous(),
        lens.reshape(-1).contiguous(),
    )


def golden_main_projection(target_hidden, main_proj_weight, main_norm_weight):
    """Reference ``main_norm(main_proj(cat(target hidden)))``."""
    import torch

    batch, target_layers, hidden = target_hidden.shape
    if target_layers != 3 or hidden != M.hidden_size:
        raise ValueError("target_hidden must have shape [batch, 3, hidden_size]")
    projected = target_hidden.float().reshape(batch, 3 * hidden).matmul(main_proj_weight.float().t())
    inv_rms = torch.rsqrt(projected.square().mean(-1, keepdim=True) + M.rms_norm_eps)
    return (projected * inv_rms * main_norm_weight.float()).to(torch.bfloat16)


def golden_markov_sample(
    head_hidden,
    norm_weight,
    lm_head_weight,
    markov_w1,
    markov_w2,
    anchor_token_ids,
):
    """Reference base LM logits plus sequential rank-256 Markov refinement."""
    import torch

    batch, steps, hidden = head_hidden.shape
    if steps != DSPARK_QUERY_TOKENS or hidden != M.hidden_size:
        raise ValueError("head_hidden must have shape [batch, 7, hidden_size]")
    if markov_w1.shape[1] != DSPARK_MARKOV_RANK or markov_w2.shape[1] != DSPARK_MARKOV_RANK:
        raise ValueError("Markov weights must use rank 256")

    x = head_hidden.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + M.rms_norm_eps)
    x = x * norm_weight.float()
    base_logits = torch.matmul(x, lm_head_weight.float().t())

    token_ids = torch.empty(batch, steps, dtype=torch.int32, device=head_hidden.device)
    markov_logits = torch.empty_like(base_logits)
    previous = anchor_token_ids.long()
    for step in range(steps):
        bias = markov_w1[previous].float().matmul(markov_w2.float().t())
        scores = base_logits[:, step, :] + bias
        markov_logits[:, step, :] = scores
        previous = torch.argmax(scores, dim=-1)
        token_ids[:, step] = previous.to(torch.int32)
    return token_ids, base_logits, markov_logits
