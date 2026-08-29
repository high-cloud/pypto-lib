# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate two constexpr RMSNorm specializations in one JIT program."""

import pypto.language as pl

from deepseek_v4_common.rmsnorm import (
    RmsNormSchedule,
    RmsNormSpec,
    build_rms_norm_recipe,
    rms_norm,
)


# Model config.
SPEC_4096 = RmsNormSpec(hidden_size=4096, eps=1e-6)
SPEC_5120 = RmsNormSpec(hidden_size=5120, eps=1e-5)

# Tiling.
TOKENS = 16
SCHEDULE_4096 = RmsNormSchedule(hidden_tile=128, token_tile=8, pipeline_stage=2)
SCHEDULE_5120 = RmsNormSchedule(hidden_tile=256, token_tile=16, pipeline_stage=3)

RECIPE_4096 = build_rms_norm_recipe(SPEC_4096, SCHEDULE_4096)
RECIPE_5120 = build_rms_norm_recipe(SPEC_5120, SCHEDULE_5120)


@pl.jit
def rms_norm_dual_test(
    x_4096: pl.Tensor[[TOKENS, 4096], pl.BF16],
    norm_w_4096: pl.Tensor[[4096], pl.BF16],
    out_4096: pl.Out[pl.Tensor[[TOKENS, 4096], pl.BF16]],
    x_5120: pl.Tensor[[TOKENS, 5120], pl.BF16],
    norm_w_5120: pl.Tensor[[5120], pl.BF16],
    out_5120: pl.Out[pl.Tensor[[TOKENS, 5120], pl.BF16]],
):
    rms_norm(
        x_4096,
        norm_w_4096,
        out_4096,
        HIDDEN_SIZE=SPEC_4096.hidden_size,
        EPS=SPEC_4096.eps,
        HIDDEN_TILE=SCHEDULE_4096.hidden_tile,
        TOKEN_TILE=SCHEDULE_4096.token_tile,
        PIPELINE_STAGE=SCHEDULE_4096.pipeline_stage,
    )
    rms_norm(
        x_5120,
        norm_w_5120,
        out_5120,
        HIDDEN_SIZE=SPEC_5120.hidden_size,
        EPS=SPEC_5120.eps,
        HIDDEN_TILE=SCHEDULE_5120.hidden_tile,
        TOKEN_TILE=SCHEDULE_5120.token_tile,
        PIPELINE_STAGE=SCHEDULE_5120.pipeline_stage,
    )
    return out_4096, out_5120


def golden_rms_norm_dual(tensors):
    tensors["out_4096"][:] = RECIPE_4096.golden(tensors["x_4096"], tensors["norm_w_4096"])
    tensors["out_5120"][:] = RECIPE_5120.golden(tensors["x_5120"], tensors["norm_w_5120"])


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_4096():
        return torch.randn(TOKENS, 4096) - 0.5

    def init_norm_w_4096():
        return torch.randn(4096) * 0.1 + 1.0

    def init_x_5120():
        return torch.randn(TOKENS, 5120) - 0.5

    def init_norm_w_5120():
        return torch.randn(5120) * 0.1 + 1.0

    return [
        TensorSpec("x_4096", [TOKENS, 4096], torch.bfloat16, init_value=init_x_4096),
        TensorSpec("norm_w_4096", [4096], torch.bfloat16, init_value=init_norm_w_4096),
        TensorSpec("out_4096", [TOKENS, 4096], torch.bfloat16, is_output=True),
        TensorSpec("x_5120", [TOKENS, 5120], torch.bfloat16, init_value=init_x_5120),
        TensorSpec("norm_w_5120", [5120], torch.bfloat16, init_value=init_norm_w_5120),
        TensorSpec("out_5120", [TOKENS, 5120], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3", "a2a3sim", "a5", "a5sim"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run_jit(
        fn=rms_norm_dual_test,
        specs=build_tensor_specs(),
        golden_fn=golden_rms_norm_dual,
        runtime_cfg={"platform": args.platform, "device_id": args.device},
        compare_fn={
            "out_4096": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "out_5120": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
