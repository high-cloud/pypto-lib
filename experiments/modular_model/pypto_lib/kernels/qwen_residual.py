# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm, linear, and residual kernels for the modular Qwen demo."""

import pypto.language as pl

from pypto_lib.schedules import LinearSchedule, RmsNormSchedule


@pl.jit.inline
def rms_norm_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Tensor,
    *,
    HIDDEN_SIZE: pl.constexpr,
    EPS: pl.constexpr,
    HIDDEN_TILE: pl.constexpr,
    TOKEN_TILE: pl.constexpr,
    PIPELINE_STAGE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    for token_block in pl.spmd(tokens // TOKEN_TILE, name_hint="rms_norm"):
        token_offset = token_block * TOKEN_TILE
        square_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
        for hidden_offset in pl.pipeline(0, HIDDEN_SIZE, HIDDEN_TILE, stage=PIPELINE_STAGE):
            reduce_x_slice = x[
                token_offset : token_offset + TOKEN_TILE, hidden_offset : hidden_offset + HIDDEN_TILE
            ]
            reduce_x_chunk = pl.cast(reduce_x_slice, target_type=pl.FP32)
            x_square = pl.mul(reduce_x_chunk, reduce_x_chunk)
            chunk_sum = pl.reshape(pl.row_sum(x_square), [1, TOKEN_TILE])
            square_sum = pl.add(square_sum, chunk_sum)
        mean_square = pl.mul(square_sum, 1.0 / HIDDEN_SIZE)
        inv_rms = pl.rsqrt(pl.add(mean_square, EPS), high_precision=True)
        inv_rms_column = pl.reshape(inv_rms, [TOKEN_TILE, 1])
        for hidden_offset in pl.pipeline(0, HIDDEN_SIZE, HIDDEN_TILE, stage=PIPELINE_STAGE):
            apply_x_slice = x[
                token_offset : token_offset + TOKEN_TILE, hidden_offset : hidden_offset + HIDDEN_TILE
            ]
            apply_x_chunk = pl.cast(apply_x_slice, target_type=pl.FP32)
            weight_chunk = weight[:, hidden_offset : hidden_offset + HIDDEN_TILE]
            scaled = pl.row_expand_mul(apply_x_chunk, inv_rms_column)
            normed = pl.col_expand_mul(scaled, weight_chunk)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            out[token_offset : token_offset + TOKEN_TILE, hidden_offset : hidden_offset + HIDDEN_TILE] = (
                normed_bf16
            )
    return out


@pl.jit.inline
def linear_residual_kernel(
    x: pl.Tensor,
    residual: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Tensor,
    *,
    IN_FEATURES: pl.constexpr,
    OUT_FEATURES: pl.constexpr,
    K_TILE: pl.constexpr,
    N_TILE: pl.constexpr,
    PIPELINE_STAGE: pl.constexpr,
):
    for output_offset in pl.parallel(0, OUT_FEATURES, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear_residual"):
            x_init = x[:, 0:K_TILE]
            weight_init = weight[0:K_TILE, output_offset : output_offset + N_TILE]
            accumulator = pl.matmul(x_init, weight_init, out_dtype=pl.FP32)
            for input_offset in pl.pipeline(K_TILE, IN_FEATURES, K_TILE, stage=PIPELINE_STAGE):
                x_chunk = x[:, input_offset : input_offset + K_TILE]
                weight_chunk = weight[
                    input_offset : input_offset + K_TILE, output_offset : output_offset + N_TILE
                ]
                accumulator = pl.matmul_acc(accumulator, x_chunk, weight_chunk)
            residual_chunk = pl.cast(residual[:, output_offset : output_offset + N_TILE], target_type=pl.FP32)
            out[:, output_offset : output_offset + N_TILE] = pl.add(accumulator, residual_chunk)
    return out


def build_qwen_residual_jit(
    tokens: int,
    hidden_size: int,
    eps: float,
    rms_schedule: RmsNormSchedule,
    linear_schedule: LinearSchedule,
):
    """Build a plain JIT graph after Module elaboration and schedule selection."""

    @pl.jit
    def qwen_residual_demo(
        hidden_states: pl.Tensor[[tokens, hidden_size], pl.BF16],
        input_norm_weight: pl.Tensor[[1, hidden_size], pl.FP32],
        projection_weight: pl.Tensor[[hidden_size, hidden_size], pl.BF16],
        out: pl.Out[pl.Tensor[[tokens, hidden_size], pl.FP32]],
    ):
        token_dim = pl.tensor.dim(hidden_states, 0)
        hidden_dim = pl.tensor.dim(hidden_states, 1)
        normed = pl.create_tensor([token_dim, hidden_dim], dtype=pl.BF16)
        normed = rms_norm_kernel(
            hidden_states,
            input_norm_weight,
            normed,
            HIDDEN_SIZE=hidden_size,
            EPS=eps,
            HIDDEN_TILE=rms_schedule.hidden_tile,
            TOKEN_TILE=rms_schedule.token_tile,
            PIPELINE_STAGE=rms_schedule.pipeline_stage,
        )
        out = linear_residual_kernel(
            normed,
            hidden_states,
            projection_weight,
            out,
            IN_FEATURES=hidden_size,
            OUT_FEATURES=hidden_size,
            K_TILE=linear_schedule.k_tile,
            N_TILE=linear_schedule.n_tile,
            PIPELINE_STAGE=linear_schedule.pipeline_stage,
        )
        return out

    return qwen_residual_demo
