# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Config-specialized RMSNorm recipe shared by DeepSeek-V4 model variants."""

from dataclasses import dataclass

import pypto.language as pl


@dataclass(frozen=True)
class RmsNormSpec:
    """Model semantics consumed by the RMSNorm recipe."""

    hidden_size: int
    eps: float


@dataclass(frozen=True)
class RmsNormSchedule:
    """Compile-time schedule for one RMSNorm specialization."""

    hidden_tile: int = 128
    token_tile: int = 8
    pipeline_stage: int = 2


@dataclass(frozen=True)
class RmsNormRecipe:
    """RMSNorm model semantics and compile-time schedule."""

    spec: RmsNormSpec
    schedule: RmsNormSchedule
    token_dim: pl.DynVar

    def golden(self, x, norm_w):
        import torch

        x = x.float()
        norm_w = norm_w.float()
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.spec.eps)
        return (x * inv * norm_w).to(torch.bfloat16)


def _validate_rms_norm(spec: RmsNormSpec, schedule: RmsNormSchedule) -> None:
    if spec.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if spec.eps <= 0:
        raise ValueError("eps must be positive")
    if schedule.hidden_tile <= 0 or schedule.token_tile <= 0:
        raise ValueError("tile sizes must be positive")
    if schedule.pipeline_stage <= 0:
        raise ValueError("pipeline_stage must be positive")
    if spec.hidden_size % schedule.hidden_tile != 0:
        raise ValueError(
            f"hidden_size ({spec.hidden_size}) must be divisible by hidden_tile ({schedule.hidden_tile})"
        )


_TOKEN_DIM = pl.dynamic("DEEPSEEK_V4_FLASH_RMS_NORM_T_DYN")


@pl.jit.inline
def rms_norm(
    x: pl.Tensor,
    norm_w: pl.Tensor,
    x_normed: pl.Tensor,
    *,
    HIDDEN_SIZE: pl.constexpr,
    EPS: pl.constexpr,
    HIDDEN_TILE: pl.constexpr,
    TOKEN_TILE: pl.constexpr,
    PIPELINE_STAGE: pl.constexpr,
):
    t_dim = pl.tensor.dim(x, 0)
    with pl.spmd(t_dim // TOKEN_TILE, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
        tg_idx = pl.tile.get_block_idx()
        tg = tg_idx * TOKEN_TILE
        x_sq_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(HIDDEN_SIZE // HIDDEN_TILE, stage=PIPELINE_STAGE):
            rms_d0 = rms_db * HIDDEN_TILE
            rms_x_chunk = pl.cast(
                x[tg : tg + TOKEN_TILE, rms_d0 : rms_d0 + HIDDEN_TILE],
                target_type=pl.FP32,
            )
            rms_x_square = pl.mul(rms_x_chunk, rms_x_chunk)
            rms_x_sum = pl.row_sum(rms_x_square)
            rms_x_sum_row = pl.reshape(rms_x_sum, [1, TOKEN_TILE])
            x_sq_sum = pl.add(x_sq_sum, rms_x_sum_row)
        x_mean_square = pl.mul(x_sq_sum, 1.0 / HIDDEN_SIZE)
        x_inv_rms = pl.rsqrt(pl.add(x_mean_square, EPS), high_precision=True)
        x_inv_rms_t = pl.reshape(x_inv_rms, [TOKEN_TILE, 1])
        for apply_db in pl.pipeline(HIDDEN_SIZE // HIDDEN_TILE, stage=PIPELINE_STAGE):
            apply_d0 = apply_db * HIDDEN_TILE
            apply_x_chunk = pl.cast(
                x[tg : tg + TOKEN_TILE, apply_d0 : apply_d0 + HIDDEN_TILE],
                target_type=pl.FP32,
            )
            norm_w_slice = norm_w[apply_d0 : apply_d0 + HIDDEN_TILE]
            norm_w_row = pl.reshape(norm_w_slice, [1, HIDDEN_TILE])
            norm_w_chunk = pl.cast(norm_w_row, target_type=pl.FP32)
            x_scaled = pl.row_expand_mul(apply_x_chunk, x_inv_rms_t)
            x_normed_chunk = pl.col_expand_mul(x_scaled, norm_w_chunk)
            x_normed[tg : tg + TOKEN_TILE, apply_d0 : apply_d0 + HIDDEN_TILE] = pl.cast(
                x_normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

    return rms_tid


def build_rms_norm_recipe(
    spec: RmsNormSpec,
    schedule: RmsNormSchedule = RmsNormSchedule(),
) -> RmsNormRecipe:
    """Build the RMSNorm semantic and schedule configuration."""

    _validate_rms_norm(spec, schedule)
    return RmsNormRecipe(
        spec=spec,
        schedule=schedule,
        token_dim=_TOKEN_DIM,
    )


def build_rms_norm_tensor_specs(recipe: RmsNormRecipe, batch: int, seq: int):
    """Build the standalone golden-harness tensors for a workload."""

    import torch
    from golden import TensorSpec

    tokens = batch * seq
    hidden_size = recipe.spec.hidden_size

    def init_x():
        return torch.randn(tokens, hidden_size) - 0.5

    def init_norm_w():
        return torch.randn(hidden_size) * 0.1 + 1.0

    return [
        TensorSpec("x", [tokens, hidden_size], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [hidden_size], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("x_normed", [tokens, hidden_size], torch.bfloat16, is_output=True),
    ]
