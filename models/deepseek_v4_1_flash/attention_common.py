# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch reference composition shared by the six V4.1 attention modes."""

from dataclasses import dataclass

import torch

from models.deepseek_v4_1_flash.config import FLASH, AttentionMode
from models.deepseek_v4_1_flash.golden import (
    compressor_ratio1,
    compressor_ratio2_paged,
    merge_attention_stats,
    paged_indexer,
    paged_sparse_attention,
    paged_sparse_attention_stats,
    publish_cache,
    publish_index_key,
    qkv_proj_rope,
    rope_interleave,
    select_candidate_blocks,
)
from models.deepseek_v4_1_flash.quantization import (
    dequantize_mxfp4_cache,
    dequantize_mxfp8_cache,
    mxfp8_linear,
    quantize_mxfp4_cache,
    quantize_mxfp8_cache,
)


@dataclass(frozen=True)
class AttentionGoldenResult:
    """Attention output and every mutable or published state produced by a mode."""

    output: torch.Tensor
    window_cache: torch.Tensor
    window_cache_scale: torch.Tensor
    compressed_cache: torch.Tensor | None
    compressed_cache_scale: torch.Tensor | None
    index_cache: torch.Tensor | None
    index_cache_scale: torch.Tensor | None
    compressor_state: torch.Tensor | None
    topk_indices: torch.Tensor | None
    candidate_mask: torch.Tensor | None


def _project_output(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
) -> torch.Tensor:
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = torch.einsum("tgd,grd->tgr", grouped, wo_a)
    return mxfp8_linear(latent.flatten(-2), wo_b, wo_b_scale)


def _publish_window(
    window_cache: torch.Tensor,
    window_kv: torch.Tensor,
    window_slots: torch.Tensor,
) -> torch.Tensor:
    updated = window_cache.clone()
    publish_cache(updated, window_kv, window_slots)
    return updated


def golden_swa_attention(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate SWA and publish the current token KV rows."""
    query, window_kv, _ = qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
    )
    window_value = dequantize_mxfp8_cache(window_cache, window_cache_scale).to(window_kv.dtype)
    updated_window = _publish_window(window_value, window_kv, window_slots)
    window_payload, updated_window_scale = quantize_mxfp8_cache(updated_window)
    quantized_window = dequantize_mxfp8_cache(window_payload, updated_window_scale).to(query.dtype)
    window_stats = paged_sparse_attention_stats(query, quantized_window, window_indices)
    attended = merge_attention_stats((window_stats,), attn_sink).to(query.dtype)
    output = _project_output(attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale)
    return AttentionGoldenResult(
        output, window_payload, updated_window_scale, None, None, None, None, None, None, None
    )


def golden_compressed_attention(
    mode: AttentionMode,
    ratio: int,
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    compressed_indices: torch.Tensor | None,
    compressor_wkv: torch.Tensor | None,
    compressor_wgate: torch.Tensor | None,
    compressor_norm_weight: torch.Tensor | None,
    compressor_state_rows: torch.Tensor | None,
    compressor_state: torch.Tensor | None,
    compressed_slots: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    compressed_lens: torch.Tensor | None,
    compressed_rope_cos: torch.Tensor | None,
    compressed_rope_sin: torch.Tensor | None,
    index_wk: torch.Tensor | None,
    index_norm_weight: torch.Tensor | None,
    index_wq_b: torch.Tensor | None,
    index_wq_b_scale: torch.Tensor | None,
    index_weights_proj: torch.Tensor | None,
    index_cache: torch.Tensor | None,
    index_cache_scale: torch.Tensor | None,
    index_block_table: torch.Tensor | None,
    request_ids: torch.Tensor | None,
    candidate_mask: torch.Tensor | None,
) -> AttentionGoldenResult:
    """Evaluate C2A/C1A full, reindex, or reuse with paged cache state."""
    query, window_kv, query_latent = qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
    )
    window_value = dequantize_mxfp8_cache(window_cache, window_cache_scale).to(window_kv.dtype)
    updated_window = _publish_window(window_value, window_kv, window_slots)
    window_payload, updated_window_scale = quantize_mxfp8_cache(updated_window)
    quantized_window = dequantize_mxfp8_cache(window_payload, updated_window_scale).to(query.dtype)
    updated_compressed = dequantize_mxfp4_cache(
        compressed_cache, compressed_cache_scale, group_size=16, scale_format="e4m3"
    ).to(query.dtype)
    updated_index = None
    if index_cache is not None and index_cache_scale is not None:
        updated_index = dequantize_mxfp4_cache(
            index_cache, index_cache_scale, group_size=32, scale_format="e8m0"
        ).to(query.dtype)
    updated_state = None if compressor_state is None else compressor_state.clone()
    topk_indices = compressed_indices
    candidates = candidate_mask

    if mode is AttentionMode.FULL:
        if compressor_wkv is None or compressor_norm_weight is None or compressed_slots is None:
            raise ValueError("full mode requires compressor weights and compressed slots")
        if compressed_rope_cos is None or compressed_rope_sin is None or updated_index is None:
            raise ValueError("full mode requires compressed RoPE rows and an index cache")
        if ratio == 1:
            latent = compressor_ratio1(x, compressor_wkv, compressor_norm_weight)
            publish_mask = compressed_slots >= 0
        elif ratio == 2:
            if compressor_wgate is None or compressor_state_rows is None or updated_state is None:
                raise ValueError("ratio-2 full mode requires gate weights and recurrent state")
            if position_ids is None:
                raise ValueError("ratio-2 full mode requires absolute position ids")
            latent, publish_mask = compressor_ratio2_paged(
                x,
                position_ids,
                compressor_state_rows,
                updated_state,
                compressor_wkv,
                compressor_wgate,
                compressor_norm_weight,
            )
        else:
            raise ValueError(f"unsupported compression ratio {ratio}")
        latent_tail = rope_interleave(
            latent[..., -compressed_rope_cos.shape[-1] * 2 :], compressed_rope_cos, compressed_rope_sin
        )
        rotated_latent = torch.cat((latent[..., : -compressed_rope_cos.shape[-1] * 2], latent_tail), dim=-1)
        publish_cache(updated_compressed, rotated_latent, compressed_slots.masked_fill(~publish_mask, -1))
        if index_wk is None or index_norm_weight is None:
            raise ValueError("full mode requires index-key weights")
        publish_index_key(
            latent,
            publish_mask,
            index_wk,
            index_norm_weight,
            compressed_rope_cos,
            compressed_rope_sin,
            compressed_slots,
            updated_index,
        )

    if mode in (AttentionMode.FULL, AttentionMode.REINDEX):
        if index_wq_b is None or index_weights_proj is None:
            raise ValueError("indexing modes require query and score weights")
        if (
            updated_index is None
            or index_block_table is None
            or request_ids is None
            or compressed_lens is None
        ):
            raise ValueError("indexing modes require cache addressing and causal compressed lengths")
        if mode is AttentionMode.FULL:
            index_payload, updated_index_scale = quantize_mxfp4_cache(
                updated_index, group_size=32, scale_format="e8m0"
            )
            quantized_index = dequantize_mxfp4_cache(
                index_payload, updated_index_scale, group_size=32, scale_format="e8m0"
            ).to(query.dtype)
        else:
            index_payload = index_cache
            updated_index_scale = index_cache_scale
            quantized_index = updated_index
        scores, topk_indices = paged_indexer(
            x,
            query_latent,
            request_ids,
            quantized_index,
            index_block_table,
            compressed_lens,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            rope_cos,
            rope_sin,
            candidates=candidate_mask,
            topk=FLASH.index_topk,
        )
        if ratio == 1 and mode is AttentionMode.FULL:
            candidates = select_candidate_blocks(
                scores,
                compressed_lens,
                FLASH.candidate_topk_blocks,
                FLASH.candidate_block_size,
            )
    if topk_indices is None:
        raise ValueError("reuse mode requires published compressed Top-K indices")

    if mode is AttentionMode.FULL:
        compressed_payload, updated_compressed_scale = quantize_mxfp4_cache(
            updated_compressed, group_size=16, scale_format="e4m3"
        )
        quantized_compressed = dequantize_mxfp4_cache(
            compressed_payload, updated_compressed_scale, group_size=16, scale_format="e4m3"
        ).to(query.dtype)
    else:
        compressed_payload = compressed_cache
        updated_compressed_scale = compressed_cache_scale
        quantized_compressed = updated_compressed
    if updated_index is None:
        index_payload = None
        updated_index_scale = None
    attended = paged_sparse_attention(
        query, quantized_window, window_indices, quantized_compressed, topk_indices, attn_sink
    )
    output = _project_output(attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale)
    return AttentionGoldenResult(
        output,
        window_payload,
        updated_window_scale,
        compressed_payload,
        updated_compressed_scale,
        index_payload,
        updated_index_scale,
        updated_state,
        topk_indices,
        candidates,
    )
