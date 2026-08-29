# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end pypto-lib Module demo for a Qwen RMSNorm projection residual block."""

import argparse

import torch
from golden import TensorSpec as GoldenTensorSpec
from golden import ratio_allclose, run_jit

from pypto_lib import CompileProfile, TensorSpec, lower_model, nn


QWEN_HIDDEN_SIZE = 5120
QWEN_RMS_NORM_EPS = 1e-6
QWEN_DECODE_TOKENS = 16


class QwenResidualProjection(nn.Module):
    """Qwen input RMSNorm followed by a hidden projection and residual add."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size, eps)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return hidden_states + self.projection(self.input_norm(hidden_states))


def build_demo(target: str, tokens: int = QWEN_DECODE_TOKENS):
    """Build the semantic graph, select schedules, and emit a plain PyPTO JIT graph."""

    model = QwenResidualProjection(QWEN_HIDDEN_SIZE, QWEN_RMS_NORM_EPS)
    input_spec = TensorSpec((tokens, QWEN_HIDDEN_SIZE), "bf16")
    return lower_model(model, input_spec, profile=CompileProfile(tokens=tokens), target=target)


def build_tensor_specs(tokens: int = QWEN_DECODE_TOKENS):
    """Build the device ABI tensors for the demo."""

    def init_projection_weight():
        return torch.randn(QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE).mul_(QWEN_HIDDEN_SIZE**-0.5)

    return [
        GoldenTensorSpec("hidden_states", [tokens, QWEN_HIDDEN_SIZE], torch.bfloat16, init_value=torch.randn),
        GoldenTensorSpec("input_norm_weight", [1, QWEN_HIDDEN_SIZE], torch.float32, init_value=torch.ones),
        GoldenTensorSpec(
            "projection_weight",
            [QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE],
            torch.bfloat16,
            init_value=init_projection_weight,
        ),
        GoldenTensorSpec("out", [tokens, QWEN_HIDDEN_SIZE], torch.float32, is_output=True),
    ]


def golden_demo(tensors):
    """Compute the Qwen residual projection reference."""

    hidden_states = tensors["hidden_states"]
    hidden_fp32 = hidden_states.float()
    inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + QWEN_RMS_NORM_EPS)
    normed = (hidden_fp32 * inv_rms * tensors["input_norm_weight"]).to(torch.bfloat16)
    projection = torch.matmul(normed.float(), tensors["projection_weight"].float())
    tensors["out"][:] = projection + hidden_fp32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen pypto-lib Module lowering demo.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=QWEN_DECODE_TOKENS)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    bundle = build_demo(args.platform, args.tokens)
    print(bundle.summary())
    result = run_jit(
        fn=bundle.jit,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_demo,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        compare_fn={"out": ratio_allclose(atol=2e-2, rtol=2e-2)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
