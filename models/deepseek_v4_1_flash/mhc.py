# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC coefficient generation, stream collapse, residual expansion, and final collapse."""

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.config import D, HC_DIM, HC_MULT, MIX_HC, T_DYN
from models.deepseek_v4_1_flash.golden import hc_head, hc_mixes, hc_post, hc_pre


def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return hc_mixes(x_hc, function, scale, base)


def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_pre(x_hc, pre_mix).to(torch.bfloat16)


def golden_mhc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    return hc_post(sublayer, residual, post_mix, residual_mix).to(torch.float32)


def golden_mhc_head(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_head(x_hc, pre_mix)


@pl.jit.inline
def mhc_mixes(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
):
    raise NotImplementedError("mHC coefficient kernel body is assigned independently")


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("mHC pre kernel body is assigned independently")


@pl.jit.inline
def mhc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    raise NotImplementedError("mHC post kernel body is assigned independently")


@pl.jit.inline
def mhc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("mHC head kernel body is assigned independently")


__all__ = [
    "golden_mhc_head",
    "golden_mhc_mixes",
    "golden_mhc_post",
    "golden_mhc_pre",
    "mhc_head",
    "mhc_mixes",
    "mhc_post",
    "mhc_pre",
]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_mhc_goldens

    run_mhc_goldens(golden_mhc_mixes, golden_mhc_pre, golden_mhc_post, golden_mhc_head)
