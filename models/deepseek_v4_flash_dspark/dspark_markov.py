# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark final RMSNorm, shared LM head, and sequential Markov sampler."""

import pypto.language as pl

from config import DSPARK_MARKOV_RANK, DSPARK_QUERY_TOKENS, FLASH as M


B_DYN = pl.dynamic("DSPARK_MARKOV_B_DYN")
VOCAB_DYN = pl.dynamic("DSPARK_MARKOV_VOCAB_DYN")
D = M.hidden_size
EPS = M.rms_norm_eps
VOCAB_TILE = 256
VOCAB_CHUNKS_PAD = 512
K_TILE = 128
MARKOV_K_TILE = 128
MAX_BATCH = 16


@pl.jit.inline
def _normalize_head_hidden(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    normalized: pl.Tensor[[8, MAX_BATCH, D], pl.BF16],
):
    batch = pl.tensor.dim(head_hidden, 0)
    normalized_flat = pl.reshape(normalized, [8 * MAX_BATCH, D])
    with pl.spmd(batch, name_hint="dspark_head_norm") as norm_tid:
        request = pl.tile.get_block_idx()
        sq_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, D, K_TILE, stage=2):
            hidden_3d = pl.slice(
                head_hidden,
                [1, 8, K_TILE],
                [request, 0, k0],
                valid_shape=[1, DSPARK_QUERY_TOKENS, K_TILE],
            )
            hidden = pl.cast(pl.reshape(hidden_3d, [8, K_TILE]), pl.FP32)
            partial = pl.reshape(pl.row_sum(pl.mul(hidden, hidden)), [1, 8])
            sq_sum = pl.add(sq_sum, partial)
        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), high_precision=True)
        inv_rms_col = pl.reshape(inv_rms, [8, 1])
        for k0 in pl.pipeline(0, D, K_TILE, stage=2):
            hidden_3d = pl.slice(
                head_hidden,
                [1, 8, K_TILE],
                [request, 0, k0],
                valid_shape=[1, DSPARK_QUERY_TOKENS, K_TILE],
            )
            hidden = pl.cast(pl.reshape(hidden_3d, [8, K_TILE]), pl.FP32)
            weight = pl.cast(pl.reshape(norm_weight[k0 : k0 + K_TILE], [1, K_TILE]), pl.FP32)
            value = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(hidden, inv_rms_col), weight),
                pl.BF16,
                mode="rint",
            )
            for step in pl.range(8):
                row = step * MAX_BATCH + request
                normalized_flat[row : row + 1, k0 : k0 + K_TILE] = value[
                    step : step + 1, 0:K_TILE
                ]
    return norm_tid


@pl.jit.inline
def _markov_logits_step(
    normalized: pl.Tensor[[8, MAX_BATCH, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, DSPARK_MARKOV_RANK], pl.BF16],
    markov_embeds: pl.Tensor[[MAX_BATCH, DSPARK_MARKOV_RANK], pl.BF16],
    logits: pl.Tensor[[MAX_BATCH, VOCAB_DYN], pl.FP32],
    step: pl.Scalar[pl.INT32],
    dependency: pl.Scalar[pl.TASK_ID],
):
    vocab = pl.tensor.dim(lm_head_weight, 0)
    vocab_chunks = vocab // VOCAB_TILE
    with pl.spmd(
        vocab_chunks, name_hint="dspark_markov_logits", deps=[dependency]
    ) as logits_tid:
        chunk = pl.tile.get_block_idx()
        vocab0 = chunk * VOCAB_TILE
        step_index = pl.cast(step, pl.INDEX)
        hidden0 = pl.reshape(
            normalized[step_index : step_index + 1, 0:MAX_BATCH, 0:K_TILE], [MAX_BATCH, K_TILE]
        )
        weight0 = lm_head_weight[vocab0 : vocab0 + VOCAB_TILE, 0:K_TILE]
        base_acc = pl.matmul(hidden0, weight0, b_trans=True, out_dtype=pl.FP32)
        for k0 in pl.pipeline(K_TILE, D, K_TILE, stage=2):
            hidden_k = pl.reshape(
                normalized[step_index : step_index + 1, 0:MAX_BATCH, k0 : k0 + K_TILE],
                [MAX_BATCH, K_TILE],
            )
            weight_k = lm_head_weight[vocab0 : vocab0 + VOCAB_TILE, k0 : k0 + K_TILE]
            base_acc = pl.matmul_acc(base_acc, hidden_k, weight_k, b_trans=True)
        markov_acc = pl.matmul(
            markov_embeds[:, 0:MARKOV_K_TILE],
            markov_w2[vocab0 : vocab0 + VOCAB_TILE, 0:MARKOV_K_TILE],
            b_trans=True,
            out_dtype=pl.FP32,
        )
        markov_acc = pl.matmul_acc(
            markov_acc,
            markov_embeds[:, MARKOV_K_TILE:DSPARK_MARKOV_RANK],
            markov_w2[vocab0 : vocab0 + VOCAB_TILE, MARKOV_K_TILE:DSPARK_MARKOV_RANK],
            b_trans=True,
        )
        logits[0:MAX_BATCH, vocab0 : vocab0 + VOCAB_TILE] = pl.add(base_acc, markov_acc)
    return logits, logits_tid


@pl.jit.inline
def _markov_embed_step(
    markov_w1: pl.Tensor[[VOCAB_DYN, DSPARK_MARKOV_RANK], pl.BF16],
    previous_ids: pl.Tensor[[MAX_BATCH], pl.INT32],
    markov_embeds: pl.Tensor[[MAX_BATCH, DSPARK_MARKOV_RANK], pl.BF16],
    dependency: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(
        MAX_BATCH, name_hint="dspark_markov_embed", deps=[dependency]
    ) as embed_tid:
        request = pl.tile.get_block_idx()
        token = pl.cast(pl.read(previous_ids, [request]), pl.INDEX)
        markov_embeds[request : request + 1, :] = markov_w1[token : token + 1, :]
    return markov_embeds, embed_tid


@pl.jit.inline
def _greedy_step(
    logits: pl.Tensor[[MAX_BATCH, VOCAB_DYN], pl.FP32],
    sampled_ids: pl.Tensor[[MAX_BATCH], pl.INT32],
    batch: pl.Scalar[pl.INT32],
    dependency: pl.Scalar[pl.TASK_ID],
):
    vocab = pl.tensor.dim(logits, 1)
    vocab_chunks = vocab // VOCAB_TILE
    with pl.spmd(
        1, name_hint="dspark_markov_greedy", deps=[dependency]
    ) as greedy_tid:
        block = pl.tile.get_block_idx()
        for request in pl.range(block, batch):
            chunk_maxima = pl.full(
                [1, VOCAB_CHUNKS_PAD], dtype=pl.FP32, value=-3.402823e38
            )
            for chunk in pl.range(vocab_chunks):
                vocab0 = chunk * VOCAB_TILE
                scores = logits[request : request + 1, vocab0 : vocab0 + VOCAB_TILE]
                indices = pl.arange(0, [1, VOCAB_TILE], dtype=pl.UINT32)
                pairs = pl.sort32(scores, indices)
                pairs = pl.mrgsort(pairs, block_len=64)
                pairs = pl.mrgsort(pairs, block_len=256)
                top_pair = pairs[:, 0:32]
                top_value = pl.gather(top_pair, mask_pattern=pl.tile.MaskPattern.P0101)
                pl.write(chunk_maxima, [0, chunk], pl.read(top_value, [0, 0]))
            chunk_indices = pl.arange(0, [1, VOCAB_CHUNKS_PAD], dtype=pl.UINT32)
            chunk_pairs = pl.sort32(chunk_maxima, chunk_indices)
            chunk_pairs = pl.mrgsort(chunk_pairs, block_len=64)
            chunk_pairs = pl.mrgsort(chunk_pairs, block_len=256)
            best_value_tensor = pl.gather(
                chunk_pairs[:, 0:32], mask_pattern=pl.tile.MaskPattern.P0101
            )
            best_value = pl.read(best_value_tensor, [0, 0])
            best_chunk = pl.cast(0, pl.INT32)
            for chunk in pl.range(vocab_chunks):
                scan_chunk = vocab_chunks - 1 - chunk
                if pl.read(chunk_maxima, [0, scan_chunk]) == best_value:
                    best_chunk = pl.cast(scan_chunk, pl.INT32)
            chunk_base = best_chunk * pl.cast(VOCAB_TILE, pl.INT32)
            winning_scores = pl.slice(
                logits,
                [1, VOCAB_TILE],
                [pl.cast(request, pl.INDEX), pl.cast(chunk_base, pl.INDEX)],
            )
            best_offset = pl.cast(0, pl.INT32)
            for offset in pl.range(VOCAB_TILE):
                scan_offset = VOCAB_TILE - 1 - offset
                if pl.read(winning_scores, [0, scan_offset]) == best_value:
                    best_offset = pl.cast(scan_offset, pl.INT32)
            token_id = best_chunk * pl.cast(VOCAB_TILE, pl.INT32) + best_offset
            pl.write(sampled_ids, [request], token_id)
    return sampled_ids, greedy_tid


@pl.jit.inline(auto_scope=False)
def dspark_markov_sample(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    markov_w1: pl.Tensor[[VOCAB_DYN, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, DSPARK_MARKOV_RANK], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS], pl.INT32],
):
    batch = pl.tensor.dim(head_hidden, 0)
    vocab = pl.tensor.dim(lm_head_weight, 0)
    normalized = pl.create_tensor([8, MAX_BATCH, D], dtype=pl.BF16, init_value=0)
    embed0 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    embed1 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    embed2 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    embed3 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    embed4 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    embed5 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    embed6 = pl.create_tensor([MAX_BATCH, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    sampled0 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    sampled1 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    sampled2 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    sampled3 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    sampled4 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    sampled5 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    sampled6 = pl.create_tensor([MAX_BATCH], dtype=pl.INT32, init_value=0)
    logits0 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    logits1 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    logits2 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    logits3 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    logits4 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    logits5 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    logits6 = pl.create_tensor([MAX_BATCH, vocab], dtype=pl.FP32)
    with pl.manual_scope():
        norm_tid = _normalize_head_hidden(head_hidden, norm_weight, normalized)
        with pl.spmd(MAX_BATCH, name_hint="dspark_markov_seed") as seed_tid:
            request = pl.tile.get_block_idx()
            if request < batch:
                anchor = pl.cast(pl.read(anchor_token_ids, [request]), pl.INDEX)
                embed0[request : request + 1, :] = markov_w1[anchor : anchor + 1, :]
        start_tid = pl.system.task_dummy(deps=[norm_tid, seed_tid])
        logits0, l0 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed0, logits0, 0, start_tid
        )
        sampled0, g0 = _greedy_step(logits0, sampled0, batch, l0)
        embed1, e1 = _markov_embed_step(markov_w1, sampled0, embed1, g0)
        logits1, l1 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed1, logits1, 1, e1
        )
        sampled1, g1 = _greedy_step(logits1, sampled1, batch, l1)
        embed2, e2 = _markov_embed_step(markov_w1, sampled1, embed2, g1)
        logits2, l2 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed2, logits2, 2, e2
        )
        sampled2, g2 = _greedy_step(logits2, sampled2, batch, l2)
        embed3, e3 = _markov_embed_step(markov_w1, sampled2, embed3, g2)
        logits3, l3 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed3, logits3, 3, e3
        )
        sampled3, g3 = _greedy_step(logits3, sampled3, batch, l3)
        embed4, e4 = _markov_embed_step(markov_w1, sampled3, embed4, g3)
        logits4, l4 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed4, logits4, 4, e4
        )
        sampled4, g4 = _greedy_step(logits4, sampled4, batch, l4)
        embed5, e5 = _markov_embed_step(markov_w1, sampled4, embed5, g4)
        logits5, l5 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed5, logits5, 5, e5
        )
        sampled5, g5 = _greedy_step(logits5, sampled5, batch, l5)
        embed6, e6 = _markov_embed_step(markov_w1, sampled5, embed6, g5)
        logits6, l6 = _markov_logits_step(
            normalized, lm_head_weight, markov_w2, embed6, logits6, 6, e6
        )
        sampled6, g6 = _greedy_step(logits6, sampled6, batch, l6)
        with pl.spmd(1, name_hint="dspark_markov_pack", deps=[g6]) as pack_tid:
            block = pl.tile.get_block_idx()
            for request in pl.range(block, batch):
                pl.write(draft_token_ids, [request, 0], pl.read(sampled0, [request]))
                pl.write(draft_token_ids, [request, 1], pl.read(sampled1, [request]))
                pl.write(draft_token_ids, [request, 2], pl.read(sampled2, [request]))
                pl.write(draft_token_ids, [request, 3], pl.read(sampled3, [request]))
                pl.write(draft_token_ids, [request, 4], pl.read(sampled4, [request]))
                pl.write(draft_token_ids, [request, 5], pl.read(sampled5, [request]))
                pl.write(draft_token_ids, [request, 6], pl.read(sampled6, [request]))
    return draft_token_ids


@pl.jit
def dspark_markov_sample_test(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    markov_w1: pl.Tensor[[VOCAB_DYN, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, DSPARK_MARKOV_RANK], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS], pl.INT32]],
):
    head_hidden.bind_dynamic(0, B_DYN)
    lm_head_weight.bind_dynamic(0, VOCAB_DYN)
    markov_w1.bind_dynamic(0, VOCAB_DYN)
    markov_w2.bind_dynamic(0, VOCAB_DYN)
    anchor_token_ids.bind_dynamic(0, B_DYN)
    draft_token_ids.bind_dynamic(0, B_DYN)
    return dspark_markov_sample(
        head_hidden,
        norm_weight,
        lm_head_weight,
        markov_w1,
        markov_w2,
        anchor_token_ids,
        draft_token_ids,
    )


def golden_dspark_markov_sample(tensors):
    from dspark_contract import golden_markov_sample

    sampled, _, _ = golden_markov_sample(
        tensors["head_hidden"],
        tensors["norm_weight"],
        tensors["lm_head_weight"],
        tensors["markov_w1"],
        tensors["markov_w2"],
        tensors["anchor_token_ids"],
    )
    tensors["draft_token_ids"][:] = sampled


def build_tensor_specs(batch=4, vocab_size=1024):
    import torch
    from golden import TensorSpec

    from dspark_contract import validate_dspark_batch, validate_markov_vocab

    validate_dspark_batch(batch)
    validate_markov_vocab(vocab_size)

    def init_anchor_token_ids():
        return torch.arange(batch, dtype=torch.int64) % vocab_size

    def init_markov_w1():
        weight = torch.zeros(vocab_size, DSPARK_MARKOV_RANK, dtype=torch.bfloat16)
        chain_tokens = min(vocab_size - 1, batch + DSPARK_QUERY_TOKENS)
        for token in range(chain_tokens):
            weight[token, token] = 1
        return weight

    def init_markov_w2():
        weight = torch.zeros(vocab_size, DSPARK_MARKOV_RANK, dtype=torch.bfloat16)
        chain_tokens = min(vocab_size - 1, batch + DSPARK_QUERY_TOKENS)
        for token in range(chain_tokens):
            weight[token + 1, token] = 16
        return weight

    return [
        TensorSpec(
            "head_hidden",
            [batch, DSPARK_QUERY_TOKENS, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(batch, DSPARK_QUERY_TOKENS, D, dtype=torch.bfloat16),
        ),
        TensorSpec("norm_weight", [D], torch.bfloat16, init_value=lambda: torch.ones(D, dtype=torch.bfloat16)),
        TensorSpec(
            "lm_head_weight",
            [vocab_size, D],
            torch.bfloat16,
            init_value=lambda: torch.zeros(vocab_size, D, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "markov_w1",
            [vocab_size, DSPARK_MARKOV_RANK],
            torch.bfloat16,
            init_value=init_markov_w1,
        ),
        TensorSpec(
            "markov_w2",
            [vocab_size, DSPARK_MARKOV_RANK],
            torch.bfloat16,
            init_value=init_markov_w2,
        ),
        TensorSpec("anchor_token_ids", [batch], torch.int64, init_value=init_anchor_token_ids),
        TensorSpec("draft_token_ids", [batch, DSPARK_QUERY_TOKENS], torch.int32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash DSpark Markov sampler validation.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4, choices=[4, 8, 12, 16])
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run_jit(
        fn=dspark_markov_sample_test,
        specs=build_tensor_specs(args.batch, args.vocab_size),
        golden_fn=golden_dspark_markov_sample,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
