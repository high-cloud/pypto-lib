# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for the DeepSeek-V4-Flash DSpark drafter."""

import sys
from pathlib import Path

import pytest
import torch


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(MODEL_DIR))

from config import (  # noqa: E402
    BLOCK_SIZE,
    DSPARK_MARKOV_RANK,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_TOKENS_PADDED,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
)
from dspark_contract import (  # noqa: E402
    build_noncausal_swa_metadata,
    build_query_layout,
    golden_markov_sample,
    validate_dspark_batch,
    validate_markov_vocab,
)


@pytest.mark.parametrize("batch", [4, 8, 12, 16])
def test_query_layout_is_anchor_first_and_padded(batch):
    anchor = torch.arange(batch, dtype=torch.int64) + 100
    context_lens = torch.arange(batch, dtype=torch.int32) + 31

    input_ids, positions, active = build_query_layout(anchor, context_lens)

    assert input_ids.shape == (batch, DSPARK_QUERY_TOKENS_PADDED)
    assert torch.equal(input_ids[:, 0], anchor)
    assert torch.all(input_ids[:, 1:] == DSPARK_NOISE_TOKEN_ID)
    assert torch.all(active[:, :DSPARK_QUERY_TOKENS])
    assert not torch.any(active[:, DSPARK_QUERY_TOKENS:])
    assert torch.equal(positions[:, 0], context_lens + 1)
    assert torch.equal(positions[:, -1], context_lens + DSPARK_QUERY_TOKENS_PADDED)


@pytest.mark.parametrize("batch", [0, 1, 5, 20])
def test_batch_contract_rejects_unsupported_sizes(batch):
    with pytest.raises(ValueError, match="batch must be one of"):
        validate_dspark_batch(batch)


def test_markov_vocab_contract_covers_checkpoint_vocab():
    validate_markov_vocab(M.vocab_size)
    for vocab_size in (0, 255, 129281, 131328):
        with pytest.raises(ValueError, match="vocab_size"):
            validate_markov_vocab(vocab_size)


def _block_table(batch, physical_base=0, max_blocks=64):
    rows = torch.arange(max_blocks, dtype=torch.int32) + physical_base
    return rows.repeat(batch, 1)


def test_noncausal_swa_exposes_all_query_rows_and_masks_padding():
    batch = 4
    context_lens = torch.tensor([1, 31, 32, 129], dtype=torch.int32)
    slots, indices, lens = build_noncausal_swa_metadata(
        _block_table(batch), context_lens
    )
    indices = indices.reshape(batch, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH)
    lens = lens.reshape(batch, DSPARK_QUERY_TOKENS_PADDED)
    slots = slots.reshape(batch, DSPARK_QUERY_TOKENS_PADDED)

    for request, context_len in enumerate(context_lens.tolist()):
        visible_len = min(context_len + 1, M.sliding_window) + DSPARK_QUERY_TOKENS
        expected_query_slots = slots[request, :DSPARK_QUERY_TOKENS]
        for query in range(DSPARK_QUERY_TOKENS):
            row = indices[request, query, :visible_len]
            assert torch.equal(row[-DSPARK_QUERY_TOKENS:], expected_query_slots)
            assert int(lens[request, query]) == visible_len
        assert int(lens[request, -1]) == 0
        assert torch.all(indices[request, -1] == -1)
        assert int(slots[request, -1]) == -1


def test_swa_metadata_honors_page_mapping_and_layer_cache_isolation():
    batch = 4
    context_lens = torch.full((batch,), 65, dtype=torch.int32)
    table0 = _block_table(batch)
    table0[:, 1], table0[:, 2] = table0[:, 2].clone(), table0[:, 1].clone()
    table1 = table0 + 100

    slots0, indices0, _ = build_noncausal_swa_metadata(table0, context_lens)
    slots1, indices1, _ = build_noncausal_swa_metadata(table1, context_lens)

    # Position 32 is redirected through the swapped logical page.
    assert int(indices0[0, 32]) == 2 * BLOCK_SIZE
    # A different layer's page table produces disjoint physical cache slots.
    valid0 = indices0[0][indices0[0] >= 0]
    valid1 = indices1[0][indices1[0] >= 0]
    assert int(valid1.min() - valid0.min()) == 100 * BLOCK_SIZE
    assert int(slots1[0] - slots0[0]) == 100 * BLOCK_SIZE


def test_markov_sampling_is_sequential_but_base_logits_are_not():
    batch = 4
    vocab = 16
    head_hidden = torch.zeros(batch, DSPARK_QUERY_TOKENS, M.hidden_size, dtype=torch.bfloat16)
    norm_weight = torch.ones(M.hidden_size, dtype=torch.bfloat16)
    lm_head_weight = torch.zeros(vocab, M.hidden_size, dtype=torch.bfloat16)
    markov_w1 = torch.zeros(vocab, DSPARK_MARKOV_RANK, dtype=torch.bfloat16)
    markov_w2 = torch.zeros(vocab, DSPARK_MARKOV_RANK, dtype=torch.bfloat16)
    for token in range(vocab):
        markov_w1[token, token] = 1
        markov_w2[(token + 1) % vocab, token] = 10

    anchor0 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    anchor1 = anchor0 + 1
    ids0, base0, logits0 = golden_markov_sample(
        head_hidden, norm_weight, lm_head_weight, markov_w1, markov_w2, anchor0
    )
    ids1, base1, logits1 = golden_markov_sample(
        head_hidden, norm_weight, lm_head_weight, markov_w1, markov_w2, anchor1
    )

    assert torch.equal(ids0[:, 0], (anchor0 + 1).to(torch.int32))
    assert torch.equal(ids0[:, 1], (anchor0 + 2).to(torch.int32))
    assert torch.equal(base0, base1)
    assert not torch.equal(logits0, logits1)
    assert not torch.equal(ids0, ids1)
