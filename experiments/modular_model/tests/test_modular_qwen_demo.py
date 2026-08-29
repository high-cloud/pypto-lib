# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path
import sys

import pytest


_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_DEMO_PATH = _EXPERIMENT_ROOT / "demos" / "qwen3_14b.py"
sys.path.insert(0, str(_EXPERIMENT_ROOT))


def _load_demo():
    spec = importlib.util.spec_from_file_location("qwen3_14b_modular_demo", _DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_module_elaborates_to_semantic_graph_and_plain_jit() -> None:
    demo = _load_demo()
    bundle = demo.build_demo("a2a3", tokens=16)

    assert [node.op for node in bundle.graph.nodes] == ["rms_norm", "linear", "add"]
    assert list(bundle.parameters) == ["input_norm.weight", "projection.weight"]
    assert bundle.parameter_bindings == {
        "input_norm.weight": "input_norm_weight",
        "projection.weight": "projection_weight",
    }
    assert bundle.jit.__name__ == "qwen_residual_demo"
    assert [dependency.__name__.split("__", 1)[0] for dependency in bundle.jit._get_deps()] == [
        "rms_norm_kernel",
        "linear_residual_kernel",
    ]
    program = bundle.jit.lower()
    assert "qwen_residual_demo" in [function.name for function in program.functions.values()]


def test_qwen_schedule_selection_is_outside_model_source() -> None:
    demo = _load_demo()
    bundle = demo.build_demo("a2a3", tokens=16)
    source = _DEMO_PATH.read_text()

    assert "HIDDEN_TILE" not in source
    assert "K_TILE" not in source
    assert bundle.schedules["input_norm"] == {
        "type": "RmsNormSchedule",
        "hidden_tile": 256,
        "token_tile": 16,
        "pipeline_stage": 2,
    }
    assert bundle.schedules["projection"] == {
        "type": "LinearSchedule",
        "k_tile": 128,
        "n_tile": 256,
        "pipeline_stage": 2,
    }


def test_qwen_schedule_rejects_an_uncovered_token_domain() -> None:
    demo = _load_demo()

    with pytest.raises(ValueError, match="no rms_norm schedule"):
        demo.build_demo("a2a3", tokens=7)
