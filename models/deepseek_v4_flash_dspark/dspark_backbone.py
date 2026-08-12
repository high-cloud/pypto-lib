# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three-layer DeepSeek-V4-Flash DSpark draft backbone."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    DSPARK_NUM_LAYERS,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_TOKENS_PADDED,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
    KV_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
)
from dspark_attention_swa import (
    HEAD_DIM,
    H,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    attention_swa,
)
from hc_head import hc_head
from dspark_metadata import dspark_metadata_core
from dspark_prepare import dspark_prepare
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


B_DYN = pl.dynamic("DSPARK_BACKBONE_B_DYN")
VOCAB_DYN = pl.dynamic("DSPARK_BACKBONE_VOCAB_DYN")
D = M.hidden_size
DRAFT_LAYER_BASE = M.num_hidden_layers
TILE_D = 512


@pl.jit(auto_scope=False)
def dspark_backbone(
    target_hidden: pl.Tensor[[B_DYN, DSPARK_NUM_LAYERS, D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    context_lens: pl.Tensor[[B_DYN], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    main_proj_weight: pl.Tensor[[D, DSPARK_NUM_LAYERS * D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    block_tables: pl.Tensor[
        [DSPARK_NUM_LAYERS, B_DYN, KV_ORI_MAX_BLOCKS], pl.INT32
    ],
    hc_attn_fn: pl.Tensor[[DSPARK_NUM_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_NUM_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_NUM_LAYERS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_NUM_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_NUM_LAYERS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_NUM_LAYERS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DSPARK_NUM_LAYERS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_NUM_LAYERS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_NUM_LAYERS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_NUM_LAYERS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[
        pl.Tensor[
            [DSPARK_NUM_LAYERS, KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    attn_sink: pl.Tensor[[DSPARK_NUM_LAYERS, H], pl.FP32],
    wo_a: pl.Tensor[
        [DSPARK_NUM_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
    ],
    wo_b: pl.Tensor[[DSPARK_NUM_LAYERS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[DSPARK_NUM_LAYERS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_NUM_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_NUM_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_NUM_LAYERS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[DSPARK_NUM_LAYERS, D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_NUM_LAYERS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_NUM_LAYERS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_NUM_LAYERS, VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[
        [DSPARK_NUM_LAYERS, N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [DSPARK_NUM_LAYERS, N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [DSPARK_NUM_LAYERS, N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [DSPARK_NUM_LAYERS, N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [DSPARK_NUM_LAYERS, N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[[DSPARK_NUM_LAYERS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_NUM_LAYERS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DSPARK_NUM_LAYERS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_NUM_LAYERS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DSPARK_NUM_LAYERS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_NUM_LAYERS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DSPARK_NUM_LAYERS, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(0, B_DYN)
    anchor_token_ids.bind_dynamic(0, B_DYN)
    context_lens.bind_dynamic(0, B_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(target_hidden, 0)
    tokens = batch * DSPARK_QUERY_TOKENS_PADDED
    main_x = pl.create_tensor([batch, D], dtype=pl.BF16)
    query_input_ids = pl.create_tensor(
        [batch, DSPARK_QUERY_TOKENS_PADDED], dtype=pl.INT64
    )
    query_positions = pl.create_tensor(
        [batch, DSPARK_QUERY_TOKENS_PADDED], dtype=pl.INT32
    )
    query_active = pl.create_tensor(
        [batch, DSPARK_QUERY_TOKENS_PADDED], dtype=pl.INT32
    )
    query_hidden = pl.create_tensor(
        [batch, DSPARK_QUERY_TOKENS_PADDED, HC_MULT, D], dtype=pl.FP32
    )
    slot_mapping = pl.create_tensor(
        [DSPARK_NUM_LAYERS, batch, DSPARK_QUERY_TOKENS_PADDED], dtype=pl.INT64
    )
    swa_indices = pl.create_tensor(
        [DSPARK_NUM_LAYERS, batch, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH],
        dtype=pl.INT32,
    )
    swa_lens = pl.create_tensor(
        [DSPARK_NUM_LAYERS, batch, DSPARK_QUERY_TOKENS_PADDED], dtype=pl.INT32
    )
    main_slot_mapping = pl.create_tensor(
        [DSPARK_NUM_LAYERS, batch], dtype=pl.INT64
    )
    with pl.scope():
        dspark_prepare(
            target_hidden,
            anchor_token_ids,
            context_lens,
            embed_weight,
            main_proj_weight,
            main_norm_weight,
            main_x,
            query_input_ids,
            query_positions,
            query_active,
            query_hidden,
        )
    with pl.scope():
        dspark_metadata_core(
            context_lens, block_tables, slot_mapping, swa_indices, swa_lens
        )
        for layer in pl.range(DSPARK_NUM_LAYERS):
            for request in pl.range(batch):
                position = pl.cast(pl.read(context_lens, [request]), pl.INDEX)
                logical_block = position // BLOCK_SIZE
                physical_block = pl.read(block_tables, [layer, request, logical_block])
                physical_slot = physical_block * BLOCK_SIZE + position % BLOCK_SIZE
                pl.write(
                    main_slot_mapping,
                    [layer, request],
                    pl.cast(physical_slot, pl.INT64),
                )
    x0 = pl.reshape(query_hidden, [tokens, HC_MULT, D])
    ids = pl.reshape(query_input_ids, [tokens])
    positions = pl.reshape(query_positions, [tokens])
    slots = pl.reshape(slot_mapping, [DSPARK_NUM_LAYERS, tokens])
    indices = pl.reshape(
        swa_indices, [DSPARK_NUM_LAYERS, tokens, DSPARK_SWA_INDEX_WIDTH]
    )
    lens = pl.reshape(swa_lens, [DSPARK_NUM_LAYERS, tokens])

    x1 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    x2 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    x3 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)

    with pl.scope():
        attention_swa(
            x0, main_x, main_slot_mapping[0], context_lens,
            hc_attn_fn[0], hc_attn_scale[0], hc_attn_base[0],
            attn_norm_w[0], wq_a[0], wq_b[0], wq_b_scale[0], wkv[0],
            gamma_cq[0], gamma_ckv[0], freqs_cos, freqs_sin, kv_cache[0],
            slots[0], indices[0], lens[0], positions, attn_sink[0], wo_a[0],
            wo_b[0], wo_b_scale[0], x1,
        )
    with pl.scope():
        moe(
            x1, hc_ffn_fn[0], hc_ffn_scale[0], hc_ffn_base[0], norm_w[0],
            gate_w[0], gate_bias[0], tid2eid[0], ids, routed_w1[0],
            routed_w1_scale[0], routed_w3[0], routed_w3_scale[0], routed_w2[0],
            routed_w2_scale[0], shared_w1[0], shared_w1_scale[0], shared_w3[0],
            shared_w3_scale[0], shared_w2[0], shared_w2_scale[0], x2,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived, DRAFT_LAYER_BASE, num_tokens, my_rank, 1,
        )
        clear_moe_signals(x2, arrived, data_arrived, combine_arrived)

    # Layers one and two reuse the same buffers only after the previous clear.
    with pl.scope():
        attention_swa(
            x2, main_x, main_slot_mapping[1], context_lens,
            hc_attn_fn[1], hc_attn_scale[1], hc_attn_base[1],
            attn_norm_w[1], wq_a[1], wq_b[1], wq_b_scale[1], wkv[1],
            gamma_cq[1], gamma_ckv[1], freqs_cos, freqs_sin, kv_cache[1],
            slots[1], indices[1], lens[1], positions, attn_sink[1], wo_a[1],
            wo_b[1], wo_b_scale[1], x1,
        )
    with pl.scope():
        moe(
            x1, hc_ffn_fn[1], hc_ffn_scale[1], hc_ffn_base[1], norm_w[1],
            gate_w[1], gate_bias[1], tid2eid[1], ids, routed_w1[1],
            routed_w1_scale[1], routed_w3[1], routed_w3_scale[1], routed_w2[1],
            routed_w2_scale[1], shared_w1[1], shared_w1_scale[1], shared_w3[1],
            shared_w3_scale[1], shared_w2[1], shared_w2_scale[1], x3,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived, DRAFT_LAYER_BASE + 1, num_tokens, my_rank, 2,
        )
        clear_moe_signals(x3, arrived, data_arrived, combine_arrived)
    with pl.scope():
        attention_swa(
            x3, main_x, main_slot_mapping[2], context_lens,
            hc_attn_fn[2], hc_attn_scale[2], hc_attn_base[2],
            attn_norm_w[2], wq_a[2], wq_b[2], wq_b_scale[2], wkv[2],
            gamma_cq[2], gamma_ckv[2], freqs_cos, freqs_sin, kv_cache[2],
            slots[2], indices[2], lens[2], positions, attn_sink[2], wo_a[2],
            wo_b[2], wo_b_scale[2], x1,
        )
    with pl.scope():
        moe(
            x1, hc_ffn_fn[2], hc_ffn_scale[2], hc_ffn_base[2], norm_w[2],
            gate_w[2], gate_bias[2], tid2eid[2], ids, routed_w1[2],
            routed_w1_scale[2], routed_w3[2], routed_w3_scale[2], routed_w2[2],
            routed_w2_scale[2], shared_w1[2], shared_w1_scale[2], shared_w3[2],
            shared_w3_scale[2], shared_w2[2], shared_w2_scale[2], x2,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived, DRAFT_LAYER_BASE + 2, num_tokens, my_rank, 3,
        )
        clear_moe_signals(x2, arrived, data_arrived, combine_arrived)

    padded_head = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.scope():
        hc_head(x2, hc_head_fn, hc_head_scale, hc_head_base, padded_head)
    output_flat = pl.reshape(head_hidden, [batch * DSPARK_QUERY_TOKENS, D])
    for task in pl.spmd(batch * DSPARK_QUERY_TOKENS * (D // TILE_D), name_hint="dspark_head_unpack"):
        output_row = task // (D // TILE_D)
        d0 = (task % (D // TILE_D)) * TILE_D
        request = output_row // DSPARK_QUERY_TOKENS
        query = output_row % DSPARK_QUERY_TOKENS
        padded_row = request * DSPARK_QUERY_TOKENS_PADDED + query
        output_flat[output_row : output_row + 1, d0 : d0 + TILE_D] = padded_head[
            padded_row : padded_row + 1, d0 : d0 + TILE_D
        ]
    return head_hidden
