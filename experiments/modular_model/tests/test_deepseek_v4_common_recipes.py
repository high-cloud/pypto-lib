# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
import sys

import pytest
import pypto.language as pl
import torch

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENT_ROOT))

from deepseek_v4_common.rmsnorm import (
    RmsNormSchedule,
    RmsNormSpec,
    build_rms_norm_recipe,
    rms_norm,
)


_FLASH_SPEC = RmsNormSpec(hidden_size=4096, eps=1e-6)


def test_flash_rms_norm_recipe_keeps_semantics_and_schedule() -> None:
    first = build_rms_norm_recipe(_FLASH_SPEC)
    second = build_rms_norm_recipe(_FLASH_SPEC, RmsNormSchedule())

    assert first == second
    assert first.spec == _FLASH_SPEC
    assert first.schedule == RmsNormSchedule()


def test_shared_rms_norm_recipe_is_isolated_from_production_models() -> None:
    source = (_EXPERIMENT_ROOT / "deepseek_v4_common" / "rmsnorm.py").read_text()

    assert "models.deepseek" not in source


@pytest.mark.parametrize(
    ("spec", "schedule", "message"),
    [
        (RmsNormSpec(hidden_size=0, eps=1e-6), RmsNormSchedule(), "hidden_size must be positive"),
        (RmsNormSpec(hidden_size=4096, eps=0.0), RmsNormSchedule(), "eps must be positive"),
        (
            _FLASH_SPEC,
            RmsNormSchedule(hidden_tile=192),
            "hidden_size (4096) must be divisible by hidden_tile (192)",
        ),
    ],
)
def test_rms_norm_recipe_rejects_invalid_config(spec, schedule, message) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        build_rms_norm_recipe(spec, schedule)


def test_rms_norm_call_builds_new_shape_specialization() -> None:
    recipe = build_rms_norm_recipe(
        RmsNormSpec(hidden_size=5120, eps=1e-5),
        RmsNormSchedule(hidden_tile=256, token_tile=16, pipeline_stage=3),
    )

    @pl.jit
    def rms_norm_5120(
        x: pl.Tensor[[16, 5120], pl.BF16],
        norm_w: pl.Tensor[[5120], pl.BF16],
        out: pl.Out[pl.Tensor[[16, 5120], pl.BF16]],
    ):
        rms_norm(
            x,
            norm_w,
            out,
            HIDDEN_SIZE=recipe.spec.hidden_size,
            EPS=recipe.spec.eps,
            HIDDEN_TILE=recipe.schedule.hidden_tile,
            TOKEN_TILE=recipe.schedule.token_tile,
            PIPELINE_STAGE=recipe.schedule.pipeline_stage,
        )
        return out

    deps = rms_norm_5120._get_deps()
    assert len(deps) == 1
    assert deps[0]._constexpr_values == {
        "HIDDEN_SIZE": 5120,
        "EPS": 1e-5,
        "HIDDEN_TILE": 256,
        "TOKEN_TILE": 16,
        "PIPELINE_STAGE": 3,
    }

    program = rms_norm_5120.lower()
    assert "rms_norm_5120" in [func.name for func in program.functions.values()]


def test_rms_norm_recipe_golden_uses_model_spec() -> None:
    recipe = build_rms_norm_recipe(_FLASH_SPEC)
    x = torch.arange(1, 4097, dtype=torch.float32).reshape(1, 4096).to(torch.bfloat16)
    norm_w = torch.ones(4096, dtype=torch.bfloat16)

    actual = recipe.golden(x, norm_w)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected)
