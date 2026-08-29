# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reusable semantic operators for model composition."""

from typing import Any

from pypto_lib.graph import GraphTensor, TensorSpec

from .module import Module, ParameterSpec


class RMSNorm(Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, hidden_size: int, eps: float, *, weight_dtype: Any = "fp32") -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = ParameterSpec((1, hidden_size), weight_dtype)

    def forward(self, x: GraphTensor) -> GraphTensor:
        if x.spec.shape[-1] != self.hidden_size:
            raise ValueError(f"RMSNorm expected hidden size {self.hidden_size}, got {x.spec.shape[-1]}")
        return x.builder.record(
            scope=self._qualified_name or "rms_norm",
            op="rms_norm",
            inputs=(x,),
            output_spec=TensorSpec(x.spec.shape, "bf16"),
            parameters=(self._parameter_name("weight"),),
            attrs={"hidden_size": self.hidden_size, "eps": self.eps},
        )


class Linear(Module):
    """Bias-free dense projection with a weight stored as input-by-output."""

    def __init__(self, in_features: int, out_features: int, *, weight_dtype: Any = "bf16") -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("linear features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ParameterSpec((in_features, out_features), weight_dtype)

    def forward(self, x: GraphTensor) -> GraphTensor:
        if len(x.spec.shape) != 2 or x.spec.shape[-1] != self.in_features:
            raise ValueError(f"Linear expected [..., {self.in_features}], got {x.spec.shape}")
        output_spec = TensorSpec((x.spec.shape[0], self.out_features), "fp32")
        return x.builder.record(
            scope=self._qualified_name or "linear",
            op="linear",
            inputs=(x,),
            output_spec=output_spec,
            parameters=(self._parameter_name("weight"),),
            attrs={"in_features": self.in_features, "out_features": self.out_features},
        )
