# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Compose the existing DSpark leaf operators into a three-layer drafter.

The caller owns request metadata. This file only connects the existing projection,
context-KV, embedding, HC, attention, MoE, LM-head, Markov-head, and sampling
implementations; it does not duplicate their math or implement serving-side
verification and rejection.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# Import moe before config-dependent leaves because --ep fixes the distributed
# shapes at module import time.
from moe import (
    AUX_PAD,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK,
    VOCAB,
    clear_moe_signals,
    moe,
)
from config import BLOCK_SIZE, FLASH as M, KV_ORI_BLOCK_NUM, MOE_TOKENS
from dspark_attention import B, INDEX_WIDTH, S, T as QUERY_TOKENS, dspark_attention
from dspark_context_kv import dspark_context_kv
from dspark_proj import MAIN_HIDDEN_DIM, dspark_proj
from hc_head import hc_head
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from lm_head import (
    DONE_VALUE as LM_HEAD_DONE_VALUE,
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    greedy_sample,
    lm_head,
)
from lookup_embedding import lookup_embedding
from markov_head import MARKOV_RANK, markov_head
from rmsnorm import rms_norm


N_DRAFT_LAYERS = 3
T = MOE_TOKENS
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
MAX_SEQ_LEN = M.max_position_embeddings

assert QUERY_TOKENS == B * S
assert QUERY_TOKENS < T
assert LM_HEAD_VOCAB == VOCAB
assert N_RANKS % LM_HEAD_TP_SIZE == 0


@pl.jit.inline
def add_markov_bias(
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_bias: pl.Tensor[[QUERY_TOKENS, VOCAB], pl.FP32],
):
    vocab_tile = 256
    work_items = QUERY_TOKENS * (VOCAB // vocab_tile)
    for block in pl.spmd(48, name_hint="add_markov_bias"):
        for work_idx in pl.range(block, work_items, 48):
            row = work_idx // (VOCAB // vocab_tile)
            vocab0 = (work_idx % (VOCAB // vocab_tile)) * vocab_tile
            logits[row : row + 1, vocab0 : vocab0 + vocab_tile] = pl.add(
                logits[row : row + 1, vocab0 : vocab0 + vocab_tile],
                logits_bias[row : row + 1, vocab0 : vocab0 + vocab_tile],
            )
    return logits


@pl.jit.inline(auto_scope=False)
def draft_layer(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[QUERY_TOKENS], pl.INT32],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    slot_mapping: pl.Tensor[[QUERY_TOKENS], pl.INT64],
    swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, mixed, post, comb)

    normalized = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(mixed, attn_norm_w, normalized)
    normalized_active: pl.Tensor[[QUERY_TOKENS, D], pl.BF16] = pl.slice(
        normalized, [QUERY_TOKENS, D], [0, 0]
    )

    attention_active = pl.create_tensor([QUERY_TOKENS, D], dtype=pl.BF16)
    dspark_attention(
        normalized_active,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, position_ids,
        kv_cache, slot_mapping, swa_indices, swa_lens,
        attn_sink, wo_a, wo_b, wo_b_scale,
        attention_active,
    )

    attention_padded = pl.create_tensor([T, D], dtype=pl.BF16, init_value=0)
    attention_padded[0:QUERY_TOKENS, 0:D] = attention_active
    attention_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hc_post_prefill(
        attention_padded,
        x_hc,
        post,
        comb,
        attention_hc,
        pl.const(QUERY_TOKENS, pl.INT32),
    )

    moe(
        attention_hc,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, pl.const(QUERY_TOKENS, pl.INT32), rank, moe_epoch,
    )
    return x_next


@pl.jit
def dspark_drafter(
    main_hidden: pl.Tensor[[B, MAIN_HIDDEN_DIM], pl.BF16],
    main_proj_w: pl.Tensor[[D, MAIN_HIDDEN_DIM], pl.BF16],
    main_norm_w: pl.Tensor[[D], pl.BF16],
    context_position_ids: pl.Tensor[[B], pl.INT32],
    context_slot_mapping: pl.Tensor[[N_DRAFT_LAYERS, B], pl.INT64],
    input_ids: pl.Tensor[[T], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[N_DRAFT_LAYERS, QUERY_TOKENS], pl.INT32],
    kv_cache: pl.Tensor[[N_DRAFT_LAYERS * KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    slot_mapping: pl.Tensor[[N_DRAFT_LAYERS, QUERY_TOKENS], pl.INT64],
    swa_indices: pl.Tensor[[N_DRAFT_LAYERS, B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[N_DRAFT_LAYERS, B], pl.INT32],
    attn_sink: pl.Tensor[[N_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_DRAFT_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_DRAFT_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_DRAFT_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    markov_w1: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, MARKOV_RANK], pl.BF16],
    head_hidden: pl.Out[pl.Tensor[[QUERY_TOKENS, D], pl.BF16]],
    draft_ids: pl.Out[pl.Tensor[[QUERY_TOKENS], pl.INT32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    main_x = pl.create_tensor([B, D], dtype=pl.BF16)
    dspark_proj(main_hidden, main_proj_w, main_norm_w, main_x)

    embedding_hidden = pl.create_tensor([T, D], dtype=pl.BF16)
    hidden = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    lookup_embedding(input_ids, embedding_weight, embedding_hidden, hidden)

    for layer in pl.range(N_DRAFT_LAYERS):
        layer_kv = pl.slice(
            kv_cache,
            [KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            [layer * KV_ORI_BLOCK_NUM, 0, 0, 0],
        )
        layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [layer * D, 0]
        )
        layer_gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [layer * HEAD_DIM]
        )
        layer_context_slots: pl.Tensor[[B], pl.INT64] = context_slot_mapping[layer]
        dspark_context_kv(
            main_x,
            layer_wkv,
            layer_gamma_ckv,
            freqs_cos,
            freqs_sin,
            context_position_ids,
            layer_context_slots,
            layer_kv,
        )

        layer_hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0]
        )
        layer_hc_attn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [layer * 3]
        )
        layer_hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [layer * MIX_HC]
        )
        layer_attn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_w, [D], [layer * D]
        )
        layer_wq_a: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [layer * D, 0]
        )
        layer_wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [layer * Q_LORA, 0]
        )
        layer_wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [layer * H * HEAD_DIM]
        )
        layer_gamma_cq: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [layer * Q_LORA]
        )
        layer_positions: pl.Tensor[[QUERY_TOKENS], pl.INT32] = position_ids[layer]
        layer_slots: pl.Tensor[[QUERY_TOKENS], pl.INT64] = slot_mapping[layer]
        layer_swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32] = swa_indices[layer]
        layer_swa_lens: pl.Tensor[[B], pl.INT32] = swa_lens[layer]
        layer_attn_sink: pl.Tensor[[H], pl.FP32] = pl.slice(
            attn_sink, [H], [layer * H]
        )
        layer_wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [layer * O_GROUPS, 0, 0]
        )
        layer_wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(
            wo_b, [D, O_GROUPS * O_LORA], [layer * D, 0]
        )
        layer_wo_b_scale: pl.Tensor[[D], pl.FP32] = pl.slice(
            wo_b_scale, [D], [layer * D]
        )
        layer_hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0]
        )
        layer_hc_ffn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [layer * 3]
        )
        layer_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [layer * MIX_HC]
        )
        layer_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [layer * D])
        layer_gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [layer * N_EXPERTS_GLOBAL, 0]
        )
        layer_gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [layer * N_EXPERTS_GLOBAL]
        )
        layer_tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [layer * VOCAB, 0]
        )
        layer_routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w1, [N_LOCAL, MOE_INTER, D], [layer * N_LOCAL, 0, 0]
        )
        layer_routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
            routed_w1_scale, [N_LOCAL, MOE_INTER], [layer * N_LOCAL, 0]
        )
        layer_routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w3, [N_LOCAL, MOE_INTER, D], [layer * N_LOCAL, 0, 0]
        )
        layer_routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
            routed_w3_scale, [N_LOCAL, MOE_INTER], [layer * N_LOCAL, 0]
        )
        layer_routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(
            routed_w2, [N_LOCAL, D, MOE_INTER], [layer * N_LOCAL, 0, 0]
        )
        layer_routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [layer * N_LOCAL, 0]
        )
        layer_shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [layer * MOE_INTER, 0]
        )
        layer_shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [layer * MOE_INTER]
        )
        layer_shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [layer * MOE_INTER, 0]
        )
        layer_shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [layer * MOE_INTER]
        )
        layer_shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(
            shared_w2, [D, MOE_INTER], [layer * D, 0]
        )
        layer_shared_w2_scale: pl.Tensor[[D], pl.FP32] = pl.slice(
            shared_w2_scale, [D], [layer * D]
        )
        hidden_next = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
        draft_layer(
            hidden,
            layer_hc_attn_fn, layer_hc_attn_scale, layer_hc_attn_base,
            layer_attn_norm_w, layer_wq_a, layer_wq_b, layer_wq_b_scale,
            layer_wkv,
            layer_gamma_cq,
            layer_gamma_ckv,
            freqs_cos,
            freqs_sin,
            layer_positions,
            layer_kv,
            layer_slots, layer_swa_indices, layer_swa_lens,
            layer_attn_sink, layer_wo_a, layer_wo_b, layer_wo_b_scale,
            layer_hc_ffn_fn, layer_hc_ffn_scale, layer_hc_ffn_base,
            layer_norm_w, layer_gate_w, layer_gate_bias, layer_tid2eid,
            input_ids,
            layer_routed_w1, layer_routed_w1_scale,
            layer_routed_w3, layer_routed_w3_scale,
            layer_routed_w2, layer_routed_w2_scale,
            layer_shared_w1, layer_shared_w1_scale,
            layer_shared_w3, layer_shared_w3_scale,
            layer_shared_w2, layer_shared_w2_scale,
            hidden_next,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.cast(layer, pl.INT32),
            rank,
            pl.cast(layer + 1, pl.INT32),
        )
        hidden = hidden_next

    clear_moe_signals(hidden, arrived, data_arrived, combine_arrived)
    head_padded = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(hidden, hc_head_fn, hc_head_scale, hc_head_base, head_padded)
    normalized_head = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(head_padded, final_norm_w, normalized_head)
    head_hidden[0:QUERY_TOKENS, 0:D] = normalized_head[0:QUERY_TOKENS, 0:D]

    logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    lm_head(
        normalized_head,
        lm_head_weight,
        logit_row_indices,
        logits,
        lm_head_hidden_window,
        lm_head_hidden_done,
        lm_head_logits_window,
        lm_head_logits_done,
        rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
        rank % LM_HEAD_TP_SIZE,
        pl.const(LM_HEAD_DONE_VALUE, pl.INT32),
    )

    logits_bias = pl.create_tensor([QUERY_TOKENS, VOCAB], dtype=pl.FP32)
    markov_embed = pl.create_tensor([QUERY_TOKENS, MARKOV_RANK], dtype=pl.BF16)
    active_input_ids: pl.Tensor[[QUERY_TOKENS], pl.INT64] = pl.slice(
        input_ids, [QUERY_TOKENS], [0]
    )
    markov_head(active_input_ids, markov_w1, markov_w2, logits_bias, markov_embed)
    add_markov_bias(logits, logits_bias)

    sampled_ids = pl.create_tensor([MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32)
    greedy_sample(logits, sampled_ids)
    for row in pl.spmd(QUERY_TOKENS, name_hint="store_draft_ids"):
        pl.write(draft_ids, [row], pl.read(sampled_ids, [row, 0]))
    return draft_ids


@pl.jit.host
def l3_dspark_drafter(
    main_hidden: pl.Tensor[[N_RANKS, B, MAIN_HIDDEN_DIM], pl.BF16],
    main_proj_w: pl.Tensor[[N_RANKS, D, MAIN_HIDDEN_DIM], pl.BF16],
    main_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    context_position_ids: pl.Tensor[[N_RANKS, B], pl.INT32],
    context_slot_mapping: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS, B], pl.INT64],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    embedding_weight: pl.Tensor[[N_RANKS, VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS, QUERY_TOKENS], pl.INT32],
    kv_cache: pl.InOut[
        pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    slot_mapping: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS, QUERY_TOKENS], pl.INT64],
    swa_indices: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS, B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS, B], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, N_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    markov_w1: pl.Tensor[[N_RANKS, VOCAB, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[N_RANKS, VOCAB, MARKOV_RANK], pl.BF16],
    head_hidden: pl.Out[pl.Tensor[[N_RANKS, QUERY_TOKENS, D], pl.BF16]],
    draft_ids: pl.Out[pl.Tensor[[N_RANKS, QUERY_TOKENS], pl.INT32]],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    lm_head_logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    lm_head_hidden_done_buf = pld.alloc_window_buffer(
        [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
    )
    lm_head_logits_done_buf = pld.alloc_window_buffer(
        [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
    )

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(
            lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16
        )
        lm_head_logits_window = pld.window(
            lm_head_logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32
        )
        lm_head_hidden_done = pld.window(
            lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        lm_head_logits_done = pld.window(
            lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        dspark_drafter(
            main_hidden[rank], main_proj_w[rank], main_norm_w[rank],
            context_position_ids[rank], context_slot_mapping[rank],
            input_ids[rank], embedding_weight[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank], position_ids[rank],
            kv_cache[rank], slot_mapping[rank], swa_indices[rank], swa_lens[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank], final_norm_w[rank],
            lm_head_weight[rank], logit_row_indices[rank], markov_w1[rank],
            markov_w2[rank], head_hidden[rank], draft_ids[rank],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            rank,
            device=rank,
        )


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def zeros(shape, dtype):
        return lambda: torch.zeros(shape, dtype=dtype)

    def ranked(name, shape, dtype, *, output=False, init=None):
        full_shape = [N_RANKS, *shape]
        return TensorSpec(
            name,
            full_shape,
            dtype,
            init_value=zeros(full_shape, dtype) if init is None else init,
            is_output=output,
        )

    def init_input_ids():
        ids = torch.arange(T, dtype=torch.int64)
        return ids.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_context_slots():
        slots = torch.empty(N_DRAFT_LAYERS, B, dtype=torch.int64)
        for layer in range(N_DRAFT_LAYERS):
            for request in range(B):
                slots[layer, request] = request * 2 * BLOCK_SIZE + BLOCK_SIZE - 1
        return slots.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_positions():
        rows = []
        for request in range(B):
            rows.extend(range(BLOCK_SIZE, BLOCK_SIZE + S))
        value = torch.tensor(rows, dtype=torch.int32)
        return value.reshape(1, 1, QUERY_TOKENS).expand(N_RANKS, N_DRAFT_LAYERS, -1).contiguous()

    def init_query_slots():
        rows = []
        for request in range(B):
            base = (request * 2 + 1) * BLOCK_SIZE
            rows.extend(base + offset for offset in range(S))
        value = torch.tensor(rows, dtype=torch.int64)
        return value.reshape(1, 1, QUERY_TOKENS).expand(N_RANKS, N_DRAFT_LAYERS, -1).contiguous()

    def init_swa_indices():
        value = torch.full((N_DRAFT_LAYERS, B, INDEX_WIDTH), -1, dtype=torch.int32)
        for layer in range(N_DRAFT_LAYERS):
            for request in range(B):
                context_slot = request * 2 * BLOCK_SIZE + BLOCK_SIZE - 1
                query_base = (request * 2 + 1) * BLOCK_SIZE
                value[layer, request, 0] = context_slot
                value[layer, request, 1 : 1 + S] = torch.arange(query_base, query_base + S)
        return value.unsqueeze(0).expand(N_RANKS, -1, -1, -1).contiguous()

    def init_tid2eid():
        token = torch.arange(VOCAB, dtype=torch.int32).reshape(VOCAB, 1)
        route = torch.arange(TOPK, dtype=torch.int32).reshape(1, TOPK)
        table = (token * TOPK + route).remainder(N_EXPERTS_GLOBAL)
        table = table.repeat(N_DRAFT_LAYERS, 1)
        return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_logit_row_indices():
        indices = torch.full((N_RANKS, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :QUERY_TOKENS] = torch.arange(QUERY_TOKENS, dtype=torch.int32)
        return indices

    specs = [
        ranked("main_hidden", [B, MAIN_HIDDEN_DIM], torch.bfloat16),
        ranked("main_proj_w", [D, MAIN_HIDDEN_DIM], torch.bfloat16),
        ranked("main_norm_w", [D], torch.bfloat16),
        ranked(
            "context_position_ids",
            [B],
            torch.int32,
            init=lambda: torch.full((N_RANKS, B), BLOCK_SIZE - 1, dtype=torch.int32),
        ),
        ranked("context_slot_mapping", [N_DRAFT_LAYERS, B], torch.int64, init=init_context_slots),
        ranked("input_ids", [T], torch.int64, init=init_input_ids),
        ranked("embedding_weight", [VOCAB, D], torch.bfloat16),
        ranked("hc_attn_fn", [N_DRAFT_LAYERS * MIX_HC, HC_DIM], torch.float32),
        ranked("hc_attn_scale", [N_DRAFT_LAYERS * 3], torch.float32),
        ranked("hc_attn_base", [N_DRAFT_LAYERS * MIX_HC], torch.float32),
        ranked("attn_norm_w", [N_DRAFT_LAYERS * D], torch.bfloat16),
        ranked("wq_a", [N_DRAFT_LAYERS * D, Q_LORA], torch.bfloat16),
        ranked("wq_b", [N_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], torch.int8),
        ranked("wq_b_scale", [N_DRAFT_LAYERS * H * HEAD_DIM], torch.float32),
        ranked("wkv", [N_DRAFT_LAYERS * D, HEAD_DIM], torch.bfloat16),
        ranked("gamma_cq", [N_DRAFT_LAYERS * Q_LORA], torch.bfloat16),
        ranked("gamma_ckv", [N_DRAFT_LAYERS * HEAD_DIM], torch.bfloat16),
        ranked("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16),
        ranked("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16),
        ranked("position_ids", [N_DRAFT_LAYERS, QUERY_TOKENS], torch.int32, init=init_positions),
        ranked(
            "kv_cache",
            [N_DRAFT_LAYERS * KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            output=True,
        ),
        ranked("slot_mapping", [N_DRAFT_LAYERS, QUERY_TOKENS], torch.int64, init=init_query_slots),
        ranked("swa_indices", [N_DRAFT_LAYERS, B, INDEX_WIDTH], torch.int32, init=init_swa_indices),
        ranked(
            "swa_lens",
            [N_DRAFT_LAYERS, B],
            torch.int32,
            init=lambda: torch.full((N_RANKS, N_DRAFT_LAYERS, B), S + 1, dtype=torch.int32),
        ),
        ranked("attn_sink", [N_DRAFT_LAYERS * H], torch.float32),
        ranked("wo_a", [N_DRAFT_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16),
        ranked("wo_b", [N_DRAFT_LAYERS * D, O_GROUPS * O_LORA], torch.int8),
        ranked("wo_b_scale", [N_DRAFT_LAYERS * D], torch.float32),
        ranked("hc_ffn_fn", [N_DRAFT_LAYERS * MIX_HC, HC_DIM], torch.float32),
        ranked("hc_ffn_scale", [N_DRAFT_LAYERS * 3], torch.float32),
        ranked("hc_ffn_base", [N_DRAFT_LAYERS * MIX_HC], torch.float32),
        ranked("norm_w", [N_DRAFT_LAYERS * D], torch.bfloat16),
        ranked("gate_w", [N_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], torch.float32),
        ranked("gate_bias", [N_DRAFT_LAYERS * N_EXPERTS_GLOBAL], torch.float32),
        ranked("tid2eid", [N_DRAFT_LAYERS * VOCAB, TOPK], torch.int32, init=init_tid2eid),
        ranked("routed_w1", [N_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], torch.int8),
        ranked("routed_w1_scale", [N_DRAFT_LAYERS * N_LOCAL, MOE_INTER], torch.float32),
        ranked("routed_w3", [N_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], torch.int8),
        ranked("routed_w3_scale", [N_DRAFT_LAYERS * N_LOCAL, MOE_INTER], torch.float32),
        ranked("routed_w2", [N_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], torch.int8),
        ranked("routed_w2_scale", [N_DRAFT_LAYERS * N_LOCAL, D], torch.float32),
        ranked("shared_w1", [N_DRAFT_LAYERS * MOE_INTER, D], torch.int8),
        ranked("shared_w1_scale", [N_DRAFT_LAYERS * MOE_INTER], torch.float32),
        ranked("shared_w3", [N_DRAFT_LAYERS * MOE_INTER, D], torch.int8),
        ranked("shared_w3_scale", [N_DRAFT_LAYERS * MOE_INTER], torch.float32),
        ranked("shared_w2", [N_DRAFT_LAYERS * D, MOE_INTER], torch.int8),
        ranked("shared_w2_scale", [N_DRAFT_LAYERS * D], torch.float32),
        ranked("hc_head_fn", [HC_MULT, HC_DIM], torch.float32),
        ranked("hc_head_scale", [1], torch.float32),
        ranked("hc_head_base", [HC_MULT], torch.float32),
        ranked("final_norm_w", [D], torch.bfloat16),
        ranked("lm_head_weight", [VOCAB_PER_TP, D], torch.bfloat16),
        ranked(
            "logit_row_indices",
            [MAX_LOGIT_ROWS],
            torch.int32,
            init=init_logit_row_indices,
        ),
        ranked("markov_w1", [VOCAB, MARKOV_RANK], torch.bfloat16),
        ranked("markov_w2", [VOCAB, MARKOV_RANK], torch.bfloat16),
        ranked("head_hidden", [QUERY_TOKENS, D], torch.bfloat16, output=True),
        ranked("draft_ids", [QUERY_TOKENS], torch.int32, output=True),
    ]
    return specs


def golden_zero_drafter(tensors):
    tensors["head_hidden"].zero_()
    tensors["draft_ids"].zero_()
    tensors["kv_cache"].zero_()


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="Validate the composed DSpark drafter.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8, 16])
    parser.add_argument("--tp", type=int, default=LM_HEAD_TP_SIZE, choices=[2, 4, 8, 16])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.tp != LM_HEAD_TP_SIZE:
        raise ValueError(f"expected --tp {LM_HEAD_TP_SIZE}, got {args.tp}")

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        raise ValueError(f"need {N_RANKS} devices, got {device_ids}")
    result = run_jit(
        fn=l3_dspark_drafter,
        specs=build_tensor_specs(),
        golden_fn=golden_zero_drafter,
        compile_only=args.compile_only,
        compile_cfg={
            "distributed_config": DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            )
        },
        runtime_cfg={"platform": args.platform},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
