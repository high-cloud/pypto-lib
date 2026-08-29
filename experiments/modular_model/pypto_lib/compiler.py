# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Module tracing, schedule selection, and lowering to plain PyPTO JIT graphs."""

from dataclasses import dataclass
from typing import Any

from .graph import Graph, GraphBuilder, TensorSpec
from .kernels.qwen_residual import build_qwen_residual_jit
from .nn import Module, ParameterSpec
from .schedules import LinearSchedule, RmsNormSchedule, manifest, select


@dataclass(frozen=True)
class CompileProfile:
    """Finite workload profile selected before kernel compilation."""

    tokens: int

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("profile tokens must be positive")


@dataclass(frozen=True)
class ModelProgram:
    """Elaborated model bundle before device artifact compilation."""

    graph: Graph
    parameters: dict[str, ParameterSpec]
    parameter_bindings: dict[str, str]
    schedules: dict[str, dict[str, Any]]
    profile: CompileProfile
    target: str
    jit: Any

    def summary(self) -> str:
        """Render the semantic graph and selected schedules."""

        lines = [f"target={self.target} tokens={self.profile.tokens}"]
        lines.append("parameters:")
        for name, parameter in self.parameters.items():
            abi_name = self.parameter_bindings.get(name, "<unbound>")
            lines.append(f"  {name}: shape={parameter.shape} dtype={parameter.dtype} abi={abi_name}")
        lines.append("graph:")
        for node in self.graph.nodes:
            lines.append(f"  {node.name}: {node.op}({', '.join(node.inputs)}) -> {node.output}")
        lines.append("schedules:")
        for name, schedule in self.schedules.items():
            values = ", ".join(f"{key}={value}" for key, value in schedule.items())
            lines.append(f"  {name}: {values}")
        return "\n".join(lines)


def _trace(model: Module, input_spec: TensorSpec) -> Graph:
    model._bind_names()
    builder = GraphBuilder("hidden_states", input_spec)
    output = model(builder.input)
    return builder.finish(output)


def _match_qwen_residual(graph: Graph) -> tuple[Any, Any, Any]:
    if tuple(node.op for node in graph.nodes) != ("rms_norm", "linear", "add"):
        ops = tuple(node.op for node in graph.nodes)
        raise NotImplementedError(f"the first lowerer supports rms_norm -> linear -> add, got {ops}")
    rms_node, linear_node, add_node = graph.nodes
    input_name = next(iter(graph.inputs))
    if linear_node.inputs != (rms_node.output,):
        raise ValueError("linear must consume the RMSNorm output")
    if set(add_node.inputs) != {input_name, linear_node.output}:
        raise ValueError("residual add must combine the model input and linear output")
    return rms_node, linear_node, add_node


def lower_model(
    model: Module, input_spec: TensorSpec, *, profile: CompileProfile, target: str
) -> ModelProgram:
    """Lower a supported Module graph into a plain PyPTO JIT dependency graph."""

    if len(input_spec.shape) != 2:
        raise ValueError(f"model input must be rank two, got {input_spec.shape}")
    if input_spec.shape[0] != profile.tokens:
        raise ValueError(f"input tokens {input_spec.shape[0]} do not match profile {profile.tokens}")

    graph = _trace(model, input_spec)
    rms_node, linear_node, _ = _match_qwen_residual(graph)
    hidden_size = input_spec.shape[1]
    if linear_node.output_spec.shape != input_spec.shape:
        raise ValueError("the first residual lowerer requires a hidden-to-hidden linear")

    rms_schedule = select("rms_norm", target, hidden_size=hidden_size, tokens=profile.tokens)
    linear_schedule = select(
        "linear",
        target,
        in_features=linear_node.attrs["in_features"],
        out_features=linear_node.attrs["out_features"],
        tokens=profile.tokens,
    )
    if not isinstance(rms_schedule, RmsNormSchedule) or not isinstance(linear_schedule, LinearSchedule):
        raise TypeError("schedule registry returned an incompatible schedule")

    jit = build_qwen_residual_jit(
        tokens=profile.tokens,
        hidden_size=hidden_size,
        eps=rms_node.attrs["eps"],
        rms_schedule=rms_schedule,
        linear_schedule=linear_schedule,
    )
    schedules = {
        rms_node.name: manifest(rms_schedule),
        linear_node.name: manifest(linear_schedule),
    }
    parameter_bindings = {
        rms_node.parameters[0]: "input_norm_weight",
        linear_node.parameters[0]: "projection_weight",
    }
    return ModelProgram(
        graph=graph,
        parameters=dict(model.named_parameters()),
        parameter_bindings=parameter_bindings,
        schedules=schedules,
        profile=profile,
        target=target,
        jit=jit,
    )
