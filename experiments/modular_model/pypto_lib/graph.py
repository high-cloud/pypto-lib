# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Semantic graph objects produced by pypto-lib modules."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    """Shape and dtype contract for one semantic tensor."""

    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class Node:
    """One semantic operator invocation before schedule selection."""

    name: str
    op: str
    inputs: tuple[str, ...]
    output: str
    output_spec: TensorSpec
    parameters: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Graph:
    """Immutable semantic graph with named inputs and outputs."""

    inputs: dict[str, TensorSpec]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]


class GraphTensor:
    """Symbolic tensor used only while tracing a pypto-lib Module."""

    def __init__(self, builder: "GraphBuilder", name: str, spec: TensorSpec) -> None:
        self.builder = builder
        self.name = name
        self.spec = spec

    def __add__(self, other: "GraphTensor") -> "GraphTensor":
        return self.builder.add(self, other)


class GraphBuilder:
    """Records semantic operator calls without invoking PyPTO."""

    def __init__(self, input_name: str, input_spec: TensorSpec) -> None:
        self.inputs = {input_name: input_spec}
        self.nodes: list[Node] = []
        self._values = {input_name: GraphTensor(self, input_name, input_spec)}
        self._name_counts: dict[str, int] = {}

    @property
    def input(self) -> GraphTensor:
        return next(iter(self._values.values()))

    def record(
        self,
        *,
        scope: str,
        op: str,
        inputs: tuple[GraphTensor, ...],
        output_spec: TensorSpec,
        parameters: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> GraphTensor:
        count = self._name_counts.get(scope, 0)
        self._name_counts[scope] = count + 1
        node_name = scope if count == 0 else f"{scope}_{count}"
        output_name = f"{node_name}.output"
        node = Node(
            name=node_name,
            op=op,
            inputs=tuple(value.name for value in inputs),
            output=output_name,
            output_spec=output_spec,
            parameters=parameters,
            attrs=dict(attrs or {}),
        )
        self.nodes.append(node)
        output = GraphTensor(self, output_name, output_spec)
        self._values[output_name] = output
        return output

    def add(self, lhs: GraphTensor, rhs: GraphTensor) -> GraphTensor:
        if lhs.builder is not self or rhs.builder is not self:
            raise ValueError("add inputs must belong to the same graph")
        if lhs.spec.shape != rhs.spec.shape:
            raise ValueError(f"add shape mismatch: {lhs.spec.shape} != {rhs.spec.shape}")
        output_dtype = lhs.spec.dtype if lhs.spec.dtype == rhs.spec.dtype else "fp32"
        output_spec = TensorSpec(lhs.spec.shape, output_dtype)
        return self.record(scope="residual_add", op="add", inputs=(lhs, rhs), output_spec=output_spec)

    def finish(self, output: GraphTensor) -> Graph:
        if output.builder is not self:
            raise ValueError("graph output belongs to another graph")
        return Graph(inputs=dict(self.inputs), nodes=tuple(self.nodes), outputs=(output.name,))
