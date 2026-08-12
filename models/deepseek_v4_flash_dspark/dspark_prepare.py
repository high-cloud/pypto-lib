# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark target-hidden projection and padded query-block construction."""

import pypto.language as pl

from config import (
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_TOKENS_PADDED,
    FLASH as M,
)


B_DYN = pl.dynamic("DSPARK_PREPARE_B_DYN")
VOCAB_DYN = pl.dynamic("DSPARK_PREPARE_VOCAB_DYN")

D = M.hidden_size
TARGET_LAYERS = 3
MAIN_K = TARGET_LAYERS * D
HC_MULT = M.hc_mult
EPS = M.rms_norm_eps
D_TILE = 512
MAIN_N_TILE = 256
MAIN_K_TILE = 128


@pl.jit.inline
def dspark_prepare(
    target_hidden: pl.Tensor[[B_DYN, TARGET_LAYERS, D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    context_lens: pl.Tensor[[B_DYN], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_K], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    main_x: pl.Tensor[[B_DYN, D], pl.BF16],
    query_input_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT64],
    query_positions: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT32],
    query_active: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT32],
    query_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED, HC_MULT, D], pl.FP32],
):
    batch = pl.tensor.dim(target_hidden, 0)
    target_flat = pl.reshape(target_hidden, [batch, MAIN_K])
    projected = pl.create_tensor([batch * 2, D], dtype=pl.FP32)

    for task in pl.spmd(batch * (D // MAIN_N_TILE), name_hint="dspark_main_proj"):
        request = task // (D // MAIN_N_TILE)
        n0 = (task % (D // MAIN_N_TILE)) * MAIN_N_TILE
        x0 = target_flat[request : request + 1, 0:MAIN_K_TILE]
        w0 = main_proj_weight[n0 : n0 + MAIN_N_TILE, 0:MAIN_K_TILE]
        acc = pl.matmul(x0, w0, b_trans=True, out_dtype=pl.FP32)
        for k0 in pl.pipeline(MAIN_K_TILE, MAIN_K, MAIN_K_TILE, stage=2):
            xk = target_flat[request : request + 1, k0 : k0 + MAIN_K_TILE]
            wk = main_proj_weight[n0 : n0 + MAIN_N_TILE, k0 : k0 + MAIN_K_TILE]
            acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
        projected[request * 2 : request * 2 + 1, n0 : n0 + MAIN_N_TILE] = acc
        projected[request * 2 + 1 : request * 2 + 2, n0 : n0 + MAIN_N_TILE] = acc

    normalized_projected = pl.create_tensor([batch * 2, D], dtype=pl.BF16)
    for block in pl.spmd(batch * 2 // 8, name_hint="dspark_main_norm"):
        row0 = block * 8
        sq_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x = projected[row0 : row0 + 8, d0 : d0 + D_TILE]
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, 8]))
        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), high_precision=True)
        inv_rms_col = pl.reshape(inv_rms, [8, 1])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x = projected[row0 : row0 + 8, d0 : d0 + D_TILE]
            weight = pl.reshape(main_norm_weight[d0 : d0 + D_TILE], [1, D_TILE])
            normalized = pl.col_expand_mul(
                pl.row_expand_mul(x, inv_rms_col), pl.cast(weight, pl.FP32)
            )
            normalized_projected[row0 : row0 + 8, d0 : d0 + D_TILE] = pl.cast(
                normalized, target_type=pl.BF16, mode="rint"
            )

    for request in pl.spmd(batch * (D // D_TILE), name_hint="dspark_main_norm_unpack"):
        batch_row = request // (D // D_TILE)
        d0 = (request % (D // D_TILE)) * D_TILE
        main_x[batch_row : batch_row + 1, d0 : d0 + D_TILE] = normalized_projected[
            batch_row * 2 : batch_row * 2 + 1, d0 : d0 + D_TILE
        ]

    token_count = batch * DSPARK_QUERY_TOKENS_PADDED
    ids_flat = pl.reshape(query_input_ids, [token_count])
    positions_flat = pl.reshape(query_positions, [token_count])
    active_flat = pl.reshape(query_active, [token_count])
    hidden_flat = pl.reshape(query_hidden, [token_count * HC_MULT, D])
    for token in pl.spmd(token_count, name_hint="dspark_query_layout"):
        request = token // DSPARK_QUERY_TOKENS_PADDED
        query = token % DSPARK_QUERY_TOKENS_PADDED
        token_id = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
        if query == 0:
            token_id = pl.read(anchor_token_ids, [request])
        pl.write(ids_flat, [token], token_id)
        context_len = pl.read(context_lens, [request])
        query_position = context_len + pl.cast(1, pl.INT32) + pl.cast(query, pl.INT32)
        pl.write(positions_flat, [token], query_position)
        is_active = pl.cast(query < DSPARK_QUERY_TOKENS, pl.INT32)
        pl.write(active_flat, [token], is_active)
        embedding_row = pl.cast(token_id, pl.INDEX)
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            embedding = embed_weight[embedding_row : embedding_row + 1, d0 : d0 + D_TILE]
            embedding_fp32 = pl.cast(embedding, pl.FP32)
            for hc in pl.range(HC_MULT):
                hidden_row = token * HC_MULT + hc
                hidden_flat[hidden_row : hidden_row + 1, d0 : d0 + D_TILE] = embedding_fp32
    return main_x, query_hidden


@pl.jit
def dspark_prepare_test(
    target_hidden: pl.Tensor[[B_DYN, TARGET_LAYERS, D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    context_lens: pl.Tensor[[B_DYN], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_K], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    main_x: pl.Out[pl.Tensor[[B_DYN, D], pl.BF16]],
    query_input_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT64]],
    query_positions: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT32]],
    query_active: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT32]],
    query_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_TOKENS_PADDED, HC_MULT, D], pl.FP32]],
):
    target_hidden.bind_dynamic(0, B_DYN)
    anchor_token_ids.bind_dynamic(0, B_DYN)
    context_lens.bind_dynamic(0, B_DYN)
    main_x.bind_dynamic(0, B_DYN)
    query_input_ids.bind_dynamic(0, B_DYN)
    query_positions.bind_dynamic(0, B_DYN)
    query_active.bind_dynamic(0, B_DYN)
    query_hidden.bind_dynamic(0, B_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    return dspark_prepare(
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


def golden_dspark_prepare(tensors):
    from dspark_contract import build_query_layout, golden_main_projection

    input_ids, positions, active = build_query_layout(
        tensors["anchor_token_ids"], tensors["context_lens"]
    )
    tensors["main_x"][:] = golden_main_projection(
        tensors["target_hidden"], tensors["main_proj_weight"], tensors["main_norm_weight"]
    )
    tensors["query_input_ids"][:] = input_ids
    tensors["query_positions"][:] = positions
    tensors["query_active"][:] = active.to(tensors["query_active"].dtype)
    embedded = tensors["embed_weight"].index_select(0, input_ids.reshape(-1))
    embedded = embedded.reshape(*input_ids.shape, 1, D).expand(-1, -1, HC_MULT, -1)
    tensors["query_hidden"][:] = embedded.float()


def build_tensor_specs(batch=4, vocab_size=M.vocab_size):
    import torch
    from golden import TensorSpec

    from dspark_contract import validate_dspark_batch

    validate_dspark_batch(batch)
    if vocab_size <= DSPARK_NOISE_TOKEN_ID:
        raise ValueError("embed vocabulary must contain the DSpark noise token")

    return [
        TensorSpec(
            "target_hidden",
            [batch, TARGET_LAYERS, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(batch, TARGET_LAYERS, D, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "anchor_token_ids",
            [batch],
            torch.int64,
            init_value=lambda: torch.arange(batch, dtype=torch.int64),
        ),
        TensorSpec(
            "context_lens",
            [batch],
            torch.int32,
            init_value=lambda: torch.arange(batch, dtype=torch.int32) + 31,
        ),
        TensorSpec(
            "embed_weight",
            [vocab_size, D],
            torch.bfloat16,
            init_value=lambda: torch.zeros(vocab_size, D, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "main_proj_weight",
            [D, MAIN_K],
            torch.bfloat16,
            init_value=lambda: torch.randn(D, MAIN_K, dtype=torch.bfloat16) / MAIN_K ** 0.5,
        ),
        TensorSpec(
            "main_norm_weight",
            [D],
            torch.bfloat16,
            init_value=lambda: torch.ones(D, dtype=torch.bfloat16),
        ),
        TensorSpec("main_x", [batch, D], torch.bfloat16, is_output=True),
        TensorSpec(
            "query_input_ids", [batch, DSPARK_QUERY_TOKENS_PADDED], torch.int64, is_output=True
        ),
        TensorSpec(
            "query_positions", [batch, DSPARK_QUERY_TOKENS_PADDED], torch.int32, is_output=True
        ),
        TensorSpec(
            "query_active", [batch, DSPARK_QUERY_TOKENS_PADDED], torch.int32, is_output=True
        ),
        TensorSpec(
            "query_hidden",
            [batch, DSPARK_QUERY_TOKENS_PADDED, HC_MULT, D],
            torch.float32,
            is_output=True,
        ),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash DSpark input preparation validation.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4, choices=[4, 8, 12, 16])
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run_jit(
        fn=dspark_prepare_test,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_dspark_prepare,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        compile_only=args.compile_only,
        compare_fn={"main_x": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=0.02)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
