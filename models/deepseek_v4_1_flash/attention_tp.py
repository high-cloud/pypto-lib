# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure tensor-parallel output reduction shared by every attention mode."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C


def golden_tp_output_all_reduce(output_partials: torch.Tensor) -> torch.Tensor:
    """Sum the row-parallel output projection from every TP rank."""
    return output_partials.float().sum(dim=0).to(output_partials.dtype)


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[C.T_DYN, C.D], pl.FP32],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    raise NotImplementedError("prefill TP output all-reduce body is assigned independently")


@pl.jit.inline(auto_scope=False)
def decode_tp_output_all_reduce(
    output_partial: pl.Tensor[[C.T_DYN, C.D], pl.FP32],
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    raise NotImplementedError("decode TP output all-reduce body is assigned independently")


__all__ = ["decode_tp_output_all_reduce", "golden_tp_output_all_reduce", "prefill_tp_output_all_reduce"]


if __name__ == "__main__":
    torch.manual_seed(17)
    partials = torch.randn(4, 7, 5)
    reduced = golden_tp_output_all_reduce(partials)
    torch.testing.assert_close(reduced, partials.float().sum(dim=0).to(partials.dtype))
    print(f"[GOLDEN] PASS attention pure-TP all-reduce output={tuple(reduced.shape)}")
