# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Target-aware operator schedule registry."""

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RmsNormSchedule:
    """RMSNorm compile-time schedule."""

    hidden_tile: int
    token_tile: int
    pipeline_stage: int


@dataclass(frozen=True)
class LinearSchedule:
    """Linear compile-time schedule."""

    k_tile: int
    n_tile: int
    pipeline_stage: int


_REGISTRY: dict[tuple[str, str], list[Callable[[dict[str, Any]], Any | None]]] = {}


def register(op: str, target: str) -> Callable[[Callable[[dict[str, Any]], Any | None]], Callable]:
    """Register a schedule selector for one semantic operator and target."""

    def decorator(selector: Callable[[dict[str, Any]], Any | None]) -> Callable:
        _REGISTRY.setdefault((op, target), []).append(selector)
        return selector

    return decorator


def select(op: str, target: str, **meta: Any) -> Any:
    """Select the first schedule whose declared domain covers the metadata."""

    target_family = "a5" if target.startswith("a5") else "a2a3"
    selectors = _REGISTRY.get((op, target_family), ())
    for selector in selectors:
        schedule = selector(meta)
        if schedule is not None:
            return schedule
    raise ValueError(f"no {op} schedule for target={target!r}, metadata={meta}")


def manifest(schedule: Any) -> dict[str, Any]:
    """Return a stable serializable schedule description."""

    return {"type": type(schedule).__name__, **asdict(schedule)}


@register("rms_norm", target="a2a3")
def _a2a3_rms_norm(meta: dict[str, Any]) -> RmsNormSchedule | None:
    hidden_size = meta["hidden_size"]
    tokens = meta["tokens"]
    if hidden_size % 256 == 0 and tokens % 16 == 0:
        return RmsNormSchedule(hidden_tile=256, token_tile=16, pipeline_stage=2)
    if hidden_size % 128 == 0 and tokens % 8 == 0:
        return RmsNormSchedule(hidden_tile=128, token_tile=8, pipeline_stage=2)
    return None


@register("linear", target="a2a3")
def _a2a3_linear(meta: dict[str, Any]) -> LinearSchedule | None:
    in_features = meta["in_features"]
    out_features = meta["out_features"]
    if in_features % 128 == 0 and out_features % 256 == 0:
        return LinearSchedule(k_tile=128, n_tile=256, pipeline_stage=2)
    if in_features % 64 == 0 and out_features % 64 == 0:
        return LinearSchedule(k_tile=64, n_tile=64, pipeline_stage=2)
    return None
