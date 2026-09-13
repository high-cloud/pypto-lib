# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expert-parallel MoE dispatch, local expert compute, and routed-output combine."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch
import torch.nn.functional as F

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.golden import gate, rms_norm
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp4
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp8
from models.deepseek_v4_1_flash.quantization import unpack_mx_b_scale


def _golden_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    gate_value = F.linear(x.float(), w1.float()).clamp(max=FLASH.swiglu_limit)
    up_value = F.linear(x.float(), w3.float()).clamp(-FLASH.swiglu_limit, FLASH.swiglu_limit)
    return F.linear(F.silu(gate_value) * up_value, w2.float())


def tp_token_owners(num_tokens: int, tp_size: int, device: torch.device | None = None) -> torch.Tensor:
    """Assign each replicated TP token row to one rank before EP dispatch."""
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    return torch.arange(num_tokens, device=device, dtype=torch.int32).remainder(tp_size)


def golden_moe(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    routed_w3: torch.Tensor,
    routed_w3_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w1_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    shared_w3: torch.Tensor,
    shared_w3_scale: torch.Tensor,
    token_owners: torch.Tensor | None = None,
    tp_size: int = 1,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Evaluate the gathered MoE result and validate unique TP row ownership."""
    shape = x.shape
    normalized = rms_norm(x.reshape(-1, shape[-1]), norm_weight)
    active_tokens = (
        normalized.shape[0] if num_tokens is None else min(max(num_tokens, 0), normalized.shape[0])
    )
    if token_owners is not None:
        if token_owners.shape != (normalized.shape[0],):
            raise ValueError("token_owners must contain one TP owner per input row")
        expected = tp_token_owners(normalized.shape[0], tp_size, token_owners.device)
        if not torch.equal(token_owners.to(torch.int32), expected):
            raise ValueError("token_owners must assign every replicated row to exactly one TP rank")
    normalized = normalized[:active_tokens]
    routed_w1 = dequantize_mxfp4(routed_w1, routed_w1_scale)
    routed_w2 = dequantize_mxfp4(routed_w2, routed_w2_scale)
    routed_w3 = dequantize_mxfp4(routed_w3, routed_w3_scale)
    shared_w1 = dequantize_mxfp8(shared_w1, unpack_mx_b_scale(shared_w1_scale)).transpose(-2, -1)
    shared_w2 = dequantize_mxfp8(shared_w2, unpack_mx_b_scale(shared_w2_scale)).transpose(-2, -1)
    shared_w3 = dequantize_mxfp8(shared_w3, unpack_mx_b_scale(shared_w3_scale)).transpose(-2, -1)
    route_weights, expert_indices = gate(normalized, gate_weight, correction_bias)
    output = _golden_expert(normalized, shared_w1, shared_w2, shared_w3)
    for expert_id in range(routed_w1.shape[0]):
        token_rows, route_columns = torch.where(expert_indices == expert_id)
        if token_rows.numel() == 0:
            continue
        routed = _golden_expert(
            normalized[token_rows], routed_w1[expert_id], routed_w2[expert_id], routed_w3[expert_id]
        )
        output[token_rows] += routed * route_weights[token_rows, route_columns].unsqueeze(-1)
    result = x.reshape(-1, shape[-1]).clone()
    result[:active_tokens] = output.to(x.dtype)
    return result.reshape(shape)


@pl.jit.inline(auto_scope=False)
def moe(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    norm_weight: pl.Tensor[[C.D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, C.D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
    routed_w1_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
    routed_w2: pl.Tensor[[C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER], pl.FP4],
    routed_w2_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER // C.MX_GROUP], pl.FP8E8M0],
    routed_w3: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
    routed_w3_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
    shared_w1: pl.Tensor[[C.D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[C.MOE_INTER, C.D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.MOE_INTER // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[C.D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    token_owners: pl.Tensor[[C.T_DYN], pl.INT32],
    recv_meta: pld.DistributedTensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D], pl.FP8E4M3FN],
    recv_scale: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D // C.MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.T_DYN * C.TOPK, C.D], pl.FP32],
    combine_arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    raise NotImplementedError("EP MoE kernel body is assigned independently")


__all__ = ["golden_moe", "moe", "tp_token_owners"]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_moe_golden

    run_moe_golden(golden_moe)
