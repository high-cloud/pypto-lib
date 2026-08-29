# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Module and parameter schema independent of PyPTO frontend semantics."""

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ParameterSpec:
    """Persistent parameter contract owned by pypto-lib."""

    shape: tuple[int, ...]
    dtype: Any


class Module:
    """Hierarchical semantic module with deterministic child and parameter names."""

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_qualified_name", "")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, ParameterSpec):
            self._parameters[name] = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _bind_names(self, prefix: str = "") -> None:
        object.__setattr__(self, "_qualified_name", prefix)
        for name, child in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            child._bind_names(child_prefix)

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield prefix, self
        for name, child in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, ParameterSpec]]:
        for name, parameter in self._parameters.items():
            parameter_name = f"{prefix}.{name}" if prefix else name
            yield parameter_name, parameter
        for name, child in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_parameters(child_prefix)

    def _parameter_name(self, local_name: str) -> str:
        return f"{self._qualified_name}.{local_name}" if self._qualified_name else local_name
