# Module worker prompt

Implement `<module_id>` against experiment `<experiment_dir>` and agreed spec
revision `<revision>`.

Main is the launched model agent that owns the implementation goal. Your scoped
module result returns to that agent; the outer launcher is not an acceptance
dependency. Do not complete the overall model goal from a module task.

- Read repository instructions, the model specification, module contract, and
  acceptance criteria before editing.
- Write scope: `<files/directories>`; preserve other agents' and user changes.
- Inputs and outputs: `<shapes, dtypes, layouts, directions, dynamic axes>`.
- Weights and persistent state: `<mapping, partition, ownership, mutations>`.
- Dependencies and fixture provenance: `<paths and revisions>`.
- Platform/topology: `<agreed values>`.
- Acceptance command and per-output comparators: `<commands and gates>`.

Implement the module and its deterministic golden entry point. Use the setup
and task-submit workflows for execution. Do not change interfaces, thresholds,
independent references, topology, or quantization policy to hide a failure.
Raise required contract changes with Main and continue independent work.

Deliver changed paths, actual commands/exit codes, artifact locations, numerical
results for outputs and state mutations, remaining failures, and a minimal
reproducer when necessary. Main will independently verify your work.
