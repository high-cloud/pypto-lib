# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Descriptor-driven core metadata lowering for DSpark packed prefill."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP, TP
from config import FLASH as MODEL_CONFIG
from lookup_embedding import lookup_embedding


REQUESTS_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_REQUESTS_DYN")
QUERY_ROWS_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_QUERY_ROWS_DYN")
SOURCE_TOKENS_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_SOURCE_TOKENS_DYN")
GROUP_TOKENS_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_GROUP_TOKENS_DYN")
LOCAL_TOKENS_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_LOCAL_TOKENS_DYN")
TABLE_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_DEPTH_DYN")
ROPE_POSITIONS_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_ROPE_POSITIONS_DYN")
VOCAB_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_VOCAB_DYN")

DESCRIPTOR_WIDTH = 11
CHUNK_START = 3
CHUNK_LEN = 4
PROMPT_LEN = 5
FLAGS = 10
PACKED_OFFSET = 6
TOKEN_SOURCE_OFFSET = 8
BLOCK_TABLE_ROW = 9
MAX_LOGIT_ROWS = 128
ROPE_DIM = 64
D = MODEL_CONFIG.hidden_size
HC_MULT = MODEL_CONFIG.hc_mult


@pl.jit.inline
def _prepare_prefill_embedding_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[SOURCE_TOKENS_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
):
    """Look up packed prompt embeddings using the existing resident lookup kernel."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    token_source.bind_dynamic(0, SOURCE_TOKENS_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    x_hc.bind_dynamic(0, GROUP_TOKENS_DYN)
    request_rows = pl.tensor.dim(descriptors, 0)
    group_tokens = pl.tensor.dim(x_hc, 0)
    ids = pl.create_tensor([group_tokens], dtype=pl.INT64)
    hidden = pl.create_tensor([group_tokens, D], dtype=pl.BF16)
    live_count = pl.create_tensor([1], dtype=pl.INT32)
    for _core in pl.spmd(1, name_hint="dspark_prefill_embedding_ids"):
        logical_length = pl.cast(0, pl.INT32)
        for row in pl.range(group_tokens):
            pl.write(ids, [row], pl.cast(0, pl.INT64))
        for request in pl.range(request_rows):
            chunk_len = pl.read(descriptors, [request, CHUNK_LEN])
            if chunk_len > 0:
                packed_start = pl.read(descriptors, [request, PACKED_OFFSET])
                source_start = pl.read(descriptors, [request, TOKEN_SOURCE_OFFSET])
                logical_length = pl.cast(pl.max(logical_length, packed_start + chunk_len), pl.INT32)
                for offset in pl.range(chunk_len):
                    pl.write(ids, [packed_start + offset], pl.read(token_source, [source_start + offset]))
        pl.write(live_count, [0], logical_length)
    hidden, embedded = lookup_embedding(ids, embed_weight, hidden, x_hc)
    flat = pl.reshape(embedded, [group_tokens * HC_MULT, D])
    for core in pl.spmd(48, name_hint="dspark_prefill_embedding_padding"):
        logical_length = pl.read(live_count, [0])
        for row in pl.range(core, group_tokens * HC_MULT, 48):
            for column in pl.range(0, D, 1024):
                values = flat[row:row + 1, column:column + 1024]
                if row >= logical_length * HC_MULT:
                    values = pl.full([1, 1024], dtype=pl.FP32, value=0.0)
                flat[row:row + 1, column:column + 1024] = values
    return x_hc


@pl.jit
def prepare_prefill_embedding_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[SOURCE_TOKENS_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
):
    return _prepare_prefill_embedding_rank(descriptors, token_source, embed_weight, x_hc)


@pl.jit.host
def l3_prepare_prefill_embedding(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[EP, SOURCE_TOKENS_DYN], pl.INT64],
    embed_weight: pl.Tensor[[EP, VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
):
    """Build each TP group's packed HC input from resident embedding weights."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    token_source.bind_dynamic(1, SOURCE_TOKENS_DYN)
    embed_weight.bind_dynamic(1, VOCAB_DYN)
    x_hc.bind_dynamic(1, GROUP_TOKENS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_embedding_rank(
            descriptors[rank], token_source[rank], embed_weight[rank], x_hc[rank], device=rank,
        )


@pl.jit.inline
def _prepare_prefill_rope_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    full_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    rows: pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16],
    rope_config: pl.Tensor[[1], pl.INT32],
):
    """Gather a full RoPE profile, with zero rows for padding and idle groups."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    full_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    rows.bind_dynamic(0, GROUP_TOKENS_DYN)
    request_rows = pl.tensor.dim(descriptors, 0)
    table_positions = pl.tensor.dim(full_table, 0)
    group_tokens = pl.tensor.dim(rows, 0)
    for _core in pl.spmd(1, name_hint="dspark_prefill_rope_prepare"):
        mode = pl.read(rope_config, [0])
        for row in pl.range(group_tokens):
            rows[row:row + 1, 0:ROPE_DIM] = pl.full([1, ROPE_DIM], dtype=pl.BF16, value=0.0)
        for request in pl.range(request_rows):
            chunk_len = pl.read(descriptors, [request, CHUNK_LEN])
            if chunk_len > 0:
                chunk_start = pl.read(descriptors, [request, CHUNK_START])
                packed_start = pl.read(descriptors, [request, PACKED_OFFSET])
                for offset in pl.range(chunk_len):
                    position = pl.cast(pl.min(chunk_start + offset, table_positions - 1), pl.INT32)
                    source_row = position
                    if mode == 4:
                        source_row = pl.cast(0, pl.INT32)
                        if (position + 1) % 4 == 0:
                            source_row = pl.cast(position - 3, pl.INT32)
                    elif mode == 128:
                        source_row = pl.cast(position - position % 128, pl.INT32)
                    target_row = packed_start + offset
                    rows[target_row:target_row + 1, 0:ROPE_DIM] = full_table[
                        source_row:source_row + 1, 0:ROPE_DIM
                    ]
    return rows


@pl.jit
def prepare_prefill_rope_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    full_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    rows: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    rope_config: pl.Tensor[[1], pl.INT32],
):
    return _prepare_prefill_rope_rank(descriptors, full_table, rows, rope_config)


@pl.jit.host
def l3_prepare_prefill_rope(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    full_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    rows: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    rope_config: pl.Tensor[[EP, 1], pl.INT32],
):
    """Gather prefill RoPE rows from rank-resident full tables."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    full_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    rows.bind_dynamic(1, GROUP_TOKENS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_rope_rank(
            descriptors[rank], full_table[rank], rows[rank], rope_config[rank], device=rank,
        )


@pl.jit.inline
def _prepare_prefill_slots_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    block_tables: pl.Tensor[[REQUESTS_DYN, TABLE_DEPTH_DYN], pl.INT32],
    slots: pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64],
    mapping_config: pl.Tensor[[2], pl.INT32],
):
    """Map live rows through raw, compressed, or compressor-state page tables."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    block_tables.bind_dynamic(0, REQUESTS_DYN)
    block_tables.bind_dynamic(1, TABLE_DEPTH_DYN)
    slots.bind_dynamic(0, GROUP_TOKENS_DYN)
    request_rows = pl.tensor.dim(descriptors, 0)
    table_depth = pl.tensor.dim(block_tables, 1)
    group_tokens = pl.tensor.dim(slots, 0)
    for _core in pl.spmd(1, name_hint="dspark_prefill_slot_prepare"):
        page_tokens = pl.read(mapping_config, [0])
        compress_ratio = pl.read(mapping_config, [1])
        for row in pl.range(group_tokens):
            pl.write(slots, [row], pl.cast(-1, pl.INT64))
        for request in pl.range(request_rows):
            chunk_len = pl.read(descriptors, [request, CHUNK_LEN])
            if chunk_len > 0:
                chunk_start = pl.read(descriptors, [request, CHUNK_START])
                packed_start = pl.read(descriptors, [request, PACKED_OFFSET])
                table_row = pl.read(descriptors, [request, BLOCK_TABLE_ROW])
                for offset in pl.range(chunk_len):
                    position = chunk_start + offset
                    cache_position = position // compress_ratio
                    logical_page = cache_position // page_tokens
                    if (position + 1) % compress_ratio == 0 and logical_page < table_depth:
                        page = pl.read(block_tables, [table_row, logical_page])
                        if page >= 0:
                            slot = pl.cast(page, pl.INT64) * page_tokens + cache_position % page_tokens
                            pl.write(slots, [packed_start + offset], slot)
    return slots


@pl.jit
def prepare_prefill_slots_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    block_tables: pl.Tensor[[REQUESTS_DYN, TABLE_DEPTH_DYN], pl.INT32],
    slots: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    mapping_config: pl.Tensor[[2], pl.INT32],
):
    return _prepare_prefill_slots_rank(descriptors, block_tables, slots, mapping_config)


@pl.jit.host
def l3_prepare_prefill_slots(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    block_tables: pl.Tensor[[EP, REQUESTS_DYN, TABLE_DEPTH_DYN], pl.INT32],
    slots: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    mapping_config: pl.Tensor[[EP, 2], pl.INT32],
):
    """Lower one cache mapping independently on all participating ranks."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    block_tables.bind_dynamic(1, REQUESTS_DYN)
    block_tables.bind_dynamic(2, TABLE_DEPTH_DYN)
    slots.bind_dynamic(1, GROUP_TOKENS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_slots_rank(
            descriptors[rank], block_tables[rank], slots[rank],
            mapping_config[rank], device=rank,
        )


@pl.jit.inline
def _prepare_prefill_core_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[SOURCE_TOKENS_DYN], pl.INT64],
    input_ids: pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[GROUP_TOKENS_DYN], pl.INT32],
    query_start_loc: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    terminal_only: pl.Scalar[pl.INT32],
):
    """Lower one rank's compact prefill descriptors into the core metadata."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    token_source.bind_dynamic(0, SOURCE_TOKENS_DYN)
    input_ids.bind_dynamic(0, LOCAL_TOKENS_DYN)
    position_ids_local.bind_dynamic(0, LOCAL_TOKENS_DYN)
    position_ids_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(0, QUERY_ROWS_DYN)
    request_rows = pl.tensor.dim(descriptors, 0)
    group_tokens = pl.tensor.dim(position_ids_full, 0)
    local_tokens = pl.tensor.dim(position_ids_local, 0)

    for core in pl.spmd(1, name_hint="dspark_prefill_descriptor_prepare"):
        for token in pl.range(local_tokens):
            pl.write(input_ids, [token], pl.cast(0, pl.INT64))
        for token in pl.range(group_tokens):
            pl.write(position_ids_full, [token], pl.cast(0, pl.INT32))
        for row in pl.range(request_rows + 1):
            pl.write(query_start_loc, [row], pl.cast(0, pl.INT32))
        for row in pl.range(MAX_LOGIT_ROWS):
            pl.write(logit_row_indices, [row], pl.cast(-1, pl.INT32))

        logical_length = pl.cast(0, pl.INT32)
        max_live_position = pl.cast(-1, pl.INT32)
        local_begin = tp_rank * local_tokens
        local_end = local_begin + local_tokens
        for request in pl.range(request_rows):
            chunk_len = pl.read(descriptors, [request, CHUNK_LEN])
            if chunk_len > 0:
                chunk_start = pl.read(descriptors, [request, CHUNK_START])
                packed_start = pl.read(descriptors, [request, PACKED_OFFSET])
                source_start = pl.read(descriptors, [request, TOKEN_SOURCE_OFFSET])
                ordinal = pl.read(descriptors, [request, BLOCK_TABLE_ROW])
                packed_end = pl.cast(packed_start + chunk_len, pl.INT32)
                logical_length = pl.cast(pl.max(logical_length, packed_end), pl.INT32)
                max_live_position = pl.cast(
                    pl.max(max_live_position, chunk_start + chunk_len - 1), pl.INT32
                )
                prompt_len = pl.read(descriptors, [request, PROMPT_LEN])
                flags = pl.read(descriptors, [request, FLAGS])
                if tp_rank == 0 and (
                    terminal_only == 0 or (chunk_start + chunk_len >= prompt_len and flags % 2 == 1)
                ):
                    pl.write(
                        logit_row_indices,
                        [ordinal],
                        pl.cast(packed_end - 1, pl.INT32),
                    )
                for token_offset in pl.range(chunk_len):
                    group_row = packed_start + token_offset
                    position = pl.cast(chunk_start + token_offset, pl.INT32)
                    pl.write(position_ids_full, [group_row], position)
                    if group_row >= local_begin and group_row < local_end:
                        local_row = group_row - local_begin
                        token_value = pl.read(
                            token_source, [source_start + token_offset]
                        )
                        pl.write(input_ids, [local_row], token_value)
            pl.write(query_start_loc, [request + 1], logical_length)

        if logical_length > 0:
            tail_start = max_live_position + 1
            for group_row in pl.range(logical_length, group_tokens):
                position = pl.cast(tail_start + group_row - logical_length, pl.INT32)
                pl.write(position_ids_full, [group_row], position)
        for local_row in pl.range(local_tokens):
            group_row = local_begin + local_row
            position = pl.read(position_ids_full, [group_row])
            pl.write(position_ids_local, [local_row], position)
    return (
        input_ids,
        position_ids_local,
        position_ids_full,
        query_start_loc,
        logit_row_indices,
    )


@pl.jit
def prepare_prefill_core_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[SOURCE_TOKENS_DYN], pl.INT64],
    input_ids: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT64]],
    position_ids_local: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32]],
    position_ids_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT32]],
    query_start_loc: pl.Out[pl.Tensor[[QUERY_ROWS_DYN], pl.INT32]],
    logit_row_indices: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    tp_rank: pl.Scalar[pl.INT32],
    terminal_only: pl.Scalar[pl.INT32],
):
    return _prepare_prefill_core_rank(descriptors, token_source, input_ids, position_ids_local, position_ids_full, query_start_loc, logit_row_indices, tp_rank, terminal_only)


@pl.jit.host
def l3_prepare_prefill_core(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[EP, SOURCE_TOKENS_DYN], pl.INT64],
    input_ids: pl.Out[pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT64]],
    position_ids_local: pl.Out[
        pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT32]
    ],
    position_ids_full: pl.Out[
        pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT32]
    ],
    query_start_loc: pl.Out[
        pl.Tensor[[EP, QUERY_ROWS_DYN], pl.INT32]
    ],
    logit_row_indices: pl.Out[
        pl.Tensor[[EP, MAX_LOGIT_ROWS], pl.INT32]
    ],
):
    """Launch descriptor lowering independently on every DSpark rank."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    token_source.bind_dynamic(1, SOURCE_TOKENS_DYN)
    input_ids.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(1, QUERY_ROWS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_core_rank(
            descriptors[rank],
            token_source[rank],
            input_ids[rank],
            position_ids_local[rank],
            position_ids_full[rank],
            query_start_loc[rank],
            logit_row_indices[rank],
            rank % TP,
            0,
            device=rank,
        )


@pl.jit.host
def l3_prepare_prefill_core_terminal(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[EP, SOURCE_TOKENS_DYN], pl.INT64],
    input_ids: pl.Out[pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT64]],
    position_ids_local: pl.Out[
        pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT32]
    ],
    position_ids_full: pl.Out[
        pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT32]
    ],
    query_start_loc: pl.Out[
        pl.Tensor[[EP, QUERY_ROWS_DYN], pl.INT32]
    ],
    logit_row_indices: pl.Out[
        pl.Tensor[[EP, MAX_LOGIT_ROWS], pl.INT32]
    ],
):
    """Launch descriptor lowering independently on every DSpark rank."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    token_source.bind_dynamic(1, SOURCE_TOKENS_DYN)
    input_ids.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(1, QUERY_ROWS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_core_rank(
            descriptors[rank],
            token_source[rank],
            input_ids[rank],
            position_ids_local[rank],
            position_ids_full[rank],
            query_start_loc[rank],
            logit_row_indices[rank],
            rank % TP,
            1,
            device=rank,
        )

TABLE_0_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_0_DEPTH_DYN")

TABLE_1_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_1_DEPTH_DYN")

TABLE_2_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_2_DEPTH_DYN")

TABLE_3_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_3_DEPTH_DYN")

TABLE_4_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_4_DEPTH_DYN")

TABLE_5_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_5_DEPTH_DYN")

TABLE_6_DEPTH_DYN = pl.dynamic("DSPARK_PREFILL_PREPARE_TABLE_6_DEPTH_DYN")


@pl.jit
def prepare_prefill_fused_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[SOURCE_TOKENS_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    mapping_configs: pl.Tensor[[7, 2], pl.INT32],
    rope_configs: pl.Tensor[[8, 1], pl.INT32],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_0_DEPTH_DYN], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_1_DEPTH_DYN], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_2_DEPTH_DYN], pl.INT32],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_3_DEPTH_DYN], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_4_DEPTH_DYN], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_5_DEPTH_DYN], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_6_DEPTH_DYN], pl.INT32],
    swa_rope_cos_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    swa_rope_sin_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio4_rope_cos_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio4_rope_sin_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio128_rope_cos_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio128_rope_sin_table: pl.Tensor[[ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    input_ids: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT64]],
    position_ids_local: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32]],
    position_ids_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT32]],
    query_start_loc: pl.Out[pl.Tensor[[QUERY_ROWS_DYN], pl.INT32]],
    logit_row_indices: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    ori_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    hca_cmp_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    csa_cmp_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    csa_idx_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    hca_state_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    csa_state_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    csa_inner_state_slot_mapping_full: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN], pl.INT64]],
    swa_freqs_cos: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    swa_freqs_sin: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    compressed_freqs_cos: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    compressed_freqs_sin: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    csa_cmp_freqs_cos: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    csa_cmp_freqs_sin: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    hca_cmp_freqs_cos: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    hca_cmp_freqs_sin: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    tp_rank: pl.Scalar[pl.INT32],
    terminal_only: pl.Scalar[pl.INT32],
):
    """Prepare core metadata, all slots/RoPE rows, and embedding in one submission."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    token_source.bind_dynamic(0, SOURCE_TOKENS_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    ori_block_table.bind_dynamic(1, TABLE_0_DEPTH_DYN)
    hca_cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(1, TABLE_1_DEPTH_DYN)
    csa_cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(1, TABLE_2_DEPTH_DYN)
    idx_block_table.bind_dynamic(0, REQUESTS_DYN)
    idx_block_table.bind_dynamic(1, TABLE_3_DEPTH_DYN)
    hca_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    hca_compress_state_block_table.bind_dynamic(1, TABLE_4_DEPTH_DYN)
    csa_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(1, TABLE_5_DEPTH_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, TABLE_6_DEPTH_DYN)
    swa_rope_cos_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    swa_rope_sin_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    ratio4_rope_cos_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    ratio4_rope_sin_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    ratio128_rope_cos_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    ratio128_rope_sin_table.bind_dynamic(0, ROPE_POSITIONS_DYN)
    input_ids.bind_dynamic(0, LOCAL_TOKENS_DYN)
    position_ids_local.bind_dynamic(0, LOCAL_TOKENS_DYN)
    position_ids_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(0, QUERY_ROWS_DYN)
    ori_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(0, GROUP_TOKENS_DYN)
    swa_freqs_cos.bind_dynamic(0, GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(0, GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(0, GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(0, GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(0, GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(0, GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(0, GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(0, GROUP_TOKENS_DYN)
    x_hc.bind_dynamic(0, GROUP_TOKENS_DYN)
    input_ids, position_ids_local, position_ids_full, query_start_loc, logit_row_indices = _prepare_prefill_core_rank(
        descriptors, token_source, input_ids, position_ids_local, position_ids_full,
        query_start_loc, logit_row_indices, tp_rank, terminal_only,
    )
    ori_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, ori_block_table, ori_slot_mapping_full, mapping_configs[0],
    )
    hca_cmp_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, hca_cmp_block_table, hca_cmp_slot_mapping_full, mapping_configs[1],
    )
    csa_cmp_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, csa_cmp_block_table, csa_cmp_slot_mapping_full, mapping_configs[2],
    )
    csa_idx_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, idx_block_table, csa_idx_slot_mapping_full, mapping_configs[3],
    )
    hca_state_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, hca_compress_state_block_table, hca_state_slot_mapping_full, mapping_configs[4],
    )
    csa_state_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, csa_compress_state_block_table, csa_state_slot_mapping_full, mapping_configs[5],
    )
    csa_inner_state_slot_mapping_full = _prepare_prefill_slots_rank(
        descriptors, csa_inner_compress_state_block_table, csa_inner_state_slot_mapping_full, mapping_configs[6],
    )
    swa_freqs_cos = _prepare_prefill_rope_rank(
        descriptors, swa_rope_cos_table, swa_freqs_cos, rope_configs[0],
    )
    swa_freqs_sin = _prepare_prefill_rope_rank(
        descriptors, swa_rope_sin_table, swa_freqs_sin, rope_configs[1],
    )
    compressed_freqs_cos = _prepare_prefill_rope_rank(
        descriptors, ratio128_rope_cos_table, compressed_freqs_cos, rope_configs[2],
    )
    compressed_freqs_sin = _prepare_prefill_rope_rank(
        descriptors, ratio128_rope_sin_table, compressed_freqs_sin, rope_configs[3],
    )
    csa_cmp_freqs_cos = _prepare_prefill_rope_rank(
        descriptors, ratio4_rope_cos_table, csa_cmp_freqs_cos, rope_configs[4],
    )
    csa_cmp_freqs_sin = _prepare_prefill_rope_rank(
        descriptors, ratio4_rope_sin_table, csa_cmp_freqs_sin, rope_configs[5],
    )
    hca_cmp_freqs_cos = _prepare_prefill_rope_rank(
        descriptors, ratio128_rope_cos_table, hca_cmp_freqs_cos, rope_configs[6],
    )
    hca_cmp_freqs_sin = _prepare_prefill_rope_rank(
        descriptors, ratio128_rope_sin_table, hca_cmp_freqs_sin, rope_configs[7],
    )
    x_hc = _prepare_prefill_embedding_rank(descriptors, token_source, embed_weight, x_hc)
    return (
        input_ids,
        position_ids_local,
        position_ids_full,
        query_start_loc,
        logit_row_indices,
        ori_slot_mapping_full,
        hca_cmp_slot_mapping_full,
        csa_cmp_slot_mapping_full,
        csa_idx_slot_mapping_full,
        hca_state_slot_mapping_full,
        csa_state_slot_mapping_full,
        csa_inner_state_slot_mapping_full,
        swa_freqs_cos,
        swa_freqs_sin,
        compressed_freqs_cos,
        compressed_freqs_sin,
        csa_cmp_freqs_cos,
        csa_cmp_freqs_sin,
        hca_cmp_freqs_cos,
        hca_cmp_freqs_sin,
        x_hc,
    )


@pl.jit.host
def l3_prepare_prefill_fused(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[EP, SOURCE_TOKENS_DYN], pl.INT64],
    embed_weight: pl.Tensor[[EP, VOCAB_DYN, D], pl.BF16],
    mapping_configs: pl.Tensor[[EP, 7, 2], pl.INT32],
    rope_configs: pl.Tensor[[EP, 8, 1], pl.INT32],
    ori_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_0_DEPTH_DYN], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_1_DEPTH_DYN], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_2_DEPTH_DYN], pl.INT32],
    idx_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_3_DEPTH_DYN], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_4_DEPTH_DYN], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_5_DEPTH_DYN], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_6_DEPTH_DYN], pl.INT32],
    swa_rope_cos_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    swa_rope_sin_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio4_rope_cos_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio4_rope_sin_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio128_rope_cos_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio128_rope_sin_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    input_ids: pl.Out[pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT64]],
    position_ids_local: pl.Out[pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT32]],
    position_ids_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT32]],
    query_start_loc: pl.Out[pl.Tensor[[EP, QUERY_ROWS_DYN], pl.INT32]],
    logit_row_indices: pl.Out[pl.Tensor[[EP, MAX_LOGIT_ROWS], pl.INT32]],
    ori_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    hca_cmp_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_cmp_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_idx_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    hca_state_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_state_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_inner_state_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    swa_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    swa_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    compressed_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    compressed_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    csa_cmp_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    csa_cmp_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    hca_cmp_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    hca_cmp_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
):
    """Prepare core metadata, all slots/RoPE rows, and embedding in one submission."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    token_source.bind_dynamic(1, SOURCE_TOKENS_DYN)
    embed_weight.bind_dynamic(1, VOCAB_DYN)
    ori_block_table.bind_dynamic(1, REQUESTS_DYN)
    ori_block_table.bind_dynamic(2, TABLE_0_DEPTH_DYN)
    hca_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(2, TABLE_1_DEPTH_DYN)
    csa_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(2, TABLE_2_DEPTH_DYN)
    idx_block_table.bind_dynamic(1, REQUESTS_DYN)
    idx_block_table.bind_dynamic(2, TABLE_3_DEPTH_DYN)
    hca_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    hca_compress_state_block_table.bind_dynamic(2, TABLE_4_DEPTH_DYN)
    csa_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(2, TABLE_5_DEPTH_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(2, TABLE_6_DEPTH_DYN)
    swa_rope_cos_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    swa_rope_sin_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio4_rope_cos_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio4_rope_sin_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio128_rope_cos_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio128_rope_sin_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    input_ids.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(1, QUERY_ROWS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    swa_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    x_hc.bind_dynamic(1, GROUP_TOKENS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_fused_rank(
            descriptors[rank],
            token_source[rank],
            embed_weight[rank],
            mapping_configs[rank],
            rope_configs[rank],
            ori_block_table[rank],
            hca_cmp_block_table[rank],
            csa_cmp_block_table[rank],
            idx_block_table[rank],
            hca_compress_state_block_table[rank],
            csa_compress_state_block_table[rank],
            csa_inner_compress_state_block_table[rank],
            swa_rope_cos_table[rank],
            swa_rope_sin_table[rank],
            ratio4_rope_cos_table[rank],
            ratio4_rope_sin_table[rank],
            ratio128_rope_cos_table[rank],
            ratio128_rope_sin_table[rank],
            input_ids[rank],
            position_ids_local[rank],
            position_ids_full[rank],
            query_start_loc[rank],
            logit_row_indices[rank],
            ori_slot_mapping_full[rank],
            hca_cmp_slot_mapping_full[rank],
            csa_cmp_slot_mapping_full[rank],
            csa_idx_slot_mapping_full[rank],
            hca_state_slot_mapping_full[rank],
            csa_state_slot_mapping_full[rank],
            csa_inner_state_slot_mapping_full[rank],
            swa_freqs_cos[rank],
            swa_freqs_sin[rank],
            compressed_freqs_cos[rank],
            compressed_freqs_sin[rank],
            csa_cmp_freqs_cos[rank],
            csa_cmp_freqs_sin[rank],
            hca_cmp_freqs_cos[rank],
            hca_cmp_freqs_sin[rank],
            x_hc[rank],
            rank % TP, 0, device=rank,
        )


@pl.jit.host
def l3_prepare_prefill_fused_terminal(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    token_source: pl.Tensor[[EP, SOURCE_TOKENS_DYN], pl.INT64],
    embed_weight: pl.Tensor[[EP, VOCAB_DYN, D], pl.BF16],
    mapping_configs: pl.Tensor[[EP, 7, 2], pl.INT32],
    rope_configs: pl.Tensor[[EP, 8, 1], pl.INT32],
    ori_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_0_DEPTH_DYN], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_1_DEPTH_DYN], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_2_DEPTH_DYN], pl.INT32],
    idx_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_3_DEPTH_DYN], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_4_DEPTH_DYN], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_5_DEPTH_DYN], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[EP, REQUESTS_DYN, TABLE_6_DEPTH_DYN], pl.INT32],
    swa_rope_cos_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    swa_rope_sin_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio4_rope_cos_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio4_rope_sin_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio128_rope_cos_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    ratio128_rope_sin_table: pl.Tensor[[EP, ROPE_POSITIONS_DYN, ROPE_DIM], pl.BF16],
    input_ids: pl.Out[pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT64]],
    position_ids_local: pl.Out[pl.Tensor[[EP, LOCAL_TOKENS_DYN], pl.INT32]],
    position_ids_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT32]],
    query_start_loc: pl.Out[pl.Tensor[[EP, QUERY_ROWS_DYN], pl.INT32]],
    logit_row_indices: pl.Out[pl.Tensor[[EP, MAX_LOGIT_ROWS], pl.INT32]],
    ori_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    hca_cmp_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_cmp_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_idx_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    hca_state_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_state_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    csa_inner_state_slot_mapping_full: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN], pl.INT64]],
    swa_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    swa_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    compressed_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    compressed_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    csa_cmp_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    csa_cmp_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    hca_cmp_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    hca_cmp_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, ROPE_DIM], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[EP, GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
):
    """Prepare core metadata, all slots/RoPE rows, and embedding in one submission."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    token_source.bind_dynamic(1, SOURCE_TOKENS_DYN)
    embed_weight.bind_dynamic(1, VOCAB_DYN)
    ori_block_table.bind_dynamic(1, REQUESTS_DYN)
    ori_block_table.bind_dynamic(2, TABLE_0_DEPTH_DYN)
    hca_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(2, TABLE_1_DEPTH_DYN)
    csa_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(2, TABLE_2_DEPTH_DYN)
    idx_block_table.bind_dynamic(1, REQUESTS_DYN)
    idx_block_table.bind_dynamic(2, TABLE_3_DEPTH_DYN)
    hca_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    hca_compress_state_block_table.bind_dynamic(2, TABLE_4_DEPTH_DYN)
    csa_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(2, TABLE_5_DEPTH_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(2, TABLE_6_DEPTH_DYN)
    swa_rope_cos_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    swa_rope_sin_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio4_rope_cos_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio4_rope_sin_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio128_rope_cos_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    ratio128_rope_sin_table.bind_dynamic(1, ROPE_POSITIONS_DYN)
    input_ids.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, LOCAL_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(1, QUERY_ROWS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, GROUP_TOKENS_DYN)
    swa_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(1, GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(1, GROUP_TOKENS_DYN)
    x_hc.bind_dynamic(1, GROUP_TOKENS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_fused_rank(
            descriptors[rank],
            token_source[rank],
            embed_weight[rank],
            mapping_configs[rank],
            rope_configs[rank],
            ori_block_table[rank],
            hca_cmp_block_table[rank],
            csa_cmp_block_table[rank],
            idx_block_table[rank],
            hca_compress_state_block_table[rank],
            csa_compress_state_block_table[rank],
            csa_inner_compress_state_block_table[rank],
            swa_rope_cos_table[rank],
            swa_rope_sin_table[rank],
            ratio4_rope_cos_table[rank],
            ratio4_rope_sin_table[rank],
            ratio128_rope_cos_table[rank],
            ratio128_rope_sin_table[rank],
            input_ids[rank],
            position_ids_local[rank],
            position_ids_full[rank],
            query_start_loc[rank],
            logit_row_indices[rank],
            ori_slot_mapping_full[rank],
            hca_cmp_slot_mapping_full[rank],
            csa_cmp_slot_mapping_full[rank],
            csa_idx_slot_mapping_full[rank],
            hca_state_slot_mapping_full[rank],
            csa_state_slot_mapping_full[rank],
            csa_inner_state_slot_mapping_full[rank],
            swa_freqs_cos[rank],
            swa_freqs_sin[rank],
            compressed_freqs_cos[rank],
            compressed_freqs_sin[rank],
            csa_cmp_freqs_cos[rank],
            csa_cmp_freqs_sin[rank],
            hca_cmp_freqs_cos[rank],
            hca_cmp_freqs_sin[rank],
            x_hc[rank],
            rank % TP, 1, device=rank,
        )
