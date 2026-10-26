# Module worker prompt

Implement `<module_id>` against experiment `<experiment_dir>` and agreed spec
revision `<revision>`.

Main is the launched model agent that owns the implementation goal. Your scoped
module result returns to that agent; the outer launcher is not an acceptance
dependency. Do not complete the overall model goal from a module task.

- Read repository instructions, the model specification, module contract, and
  acceptance criteria before editing.
- Write scope: `<files/directories>`; preserve other agents' and user changes.
- Owning operator, golden, default validation/comparator, and composition consumers:
  `<MODULES.md ownership-map rows>`.
- Inputs and outputs: `<shapes, dtypes, layouts, directions, dynamic axes>`.
- Phase execution contract: `<token tiles, loop nesting, prefill/decode shapes>`.
- Weights and persistent state: `<mapping, partition, ownership, mutations>`.
- Dependencies and fixture provenance: `<paths and revisions>`.
- Platform/topology: `<agreed values>`.
- Acceptance command and per-output comparators: `<commands and gates>`.

Apply the [model rules](../../.claude/rules/models.md). Implement the module and
its deterministic golden entry point using the existing harness and comparators.
Validation and composition must exercise the same compute symbols and variants.
Keep permanent validation code in repository sources and frozen data in artifacts.
Use the setup and task-submit workflows for execution. Do not change interfaces, thresholds,
independent references, topology, or quantization policy to hide a failure.
Raise required contract changes with Main and continue independent work.
Preserve multi-token prefill shapes through projections and feed-forward work.
Report necessary recurrent scans separately; do not replace batched work with a
tokenwise complete-stack path to obtain a numerical pass. Deliver source and
generated loop/shape evidence for Main's structural acceptance.
For a first working path, return ownership, dependencies, call-depth responsibility,
and the actual bounded validation result before expanding the implementation.

Deliver changed paths, actual commands/exit codes, artifact locations, numerical
results for outputs and state mutations, remaining failures, and a minimal
reproducer when necessary. Main will independently verify your work.
