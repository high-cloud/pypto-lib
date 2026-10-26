# Modules and composition

Spec revision: draft.

| Module | Producer / consumer | Interface / weights | Acceptance | Owner |
| --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | Unassigned |

For each module, expand all operands and results with shape, dtype, physical
layout, direction, dynamic dimensions, partition, ownership, and state updates.
Include rank communication dependencies when distributed.

Describe the dependency graph, model-specific half-layer groups, and assembly
recipe for each layer type. Different layer types may have different signatures.

## File ownership and validation map

Complete this map before implementation and keep it current. Apply the
[model rules](../../../.claude/rules/models.md) to the planned and delivered files.

| Operator / phase | Owning file / compute symbol | Golden source / symbol | Default validation command / comparator | Actual composition consumer / variant |
| --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD |

Account for each kernel, numerical primitive, shared child, and host utility.
Record justified kernel-only small operators explicitly. Name the actual called
symbols and variants on both validation and composition paths. For a replacement,
record affected consumers and whether the old path was removed or has an explicit
remaining use. Frozen fixture data may live in artifacts; the default validation
must not depend on validation code available only in an experiment directory.

## First working paths

| Path | Import / call graph and source locations | Validation and composition symbols / variants | Command / actual result | Rule findings / resolution |
| --- | --- | --- | --- | --- |
| First attention path | TBD | TBD | TBD | TBD |
| First MoE or dense feed-forward path | TBD | TBD | TBD | TBD |

Check these paths before scaling to the complete model, and each distinct mixer
as it becomes executable. Include bounded prefill/decode shapes and the default
validation entry. Explain any permitted extra call depth with the responsibility
of each retained level. Numerical passes alone do not close structural findings.

For distributed MoE, include the rank/group table, request/token ownership,
global-to-local expert IDs, shared-expert replication, routing and combine weights,
receive capacities, communication dependencies, and epoch/buffer lifetimes.
Add reference fixtures for local/remote/skewed traffic and idle-rank participation.

## Complete-model entries

| Entry | Source / symbol | Ordered weights and state | Supported shapes | Compile / run command |
| --- | --- | --- | --- | --- |
| Prefill | TBD | TBD | TBD | TBD |
| Decode | TBD | TBD | TBD | TBD |

Specify compilation coverage from embedding through all layers, final norm, and
output head. Record generated IR/task-graph artifacts and the generation caller.
Multiple internal tasks are allowed. A host loop submitting individual operators
does not implement a compiled complete-model entry. Freeze concrete source paths,
normally `prefill_fwd.py` and `decode_fwd.py`, during specification agreement.

## Execution structure

| Phase / module | Loop nesting and token tile | Effective operand shapes | State / communication behavior | Source and generated evidence |
| --- | --- | --- | --- | --- |
| Prefill projection / attention or recurrent mixer | TBD | TBD | TBD | TBD |
| Prefill feed-forward / routed and shared experts | TBD | TBD | TBD | TBD |
| Decode stack | TBD | TBD | TBD | TBD |

Record the actual helper call path and expanded compiled loops. Multi-token
prefill retains token tiles greater than one for projections and feed-forward
work; model-required recurrent scans may remain sequential. A prompt-token loop
enclosing all decoder layers is a diagnostic implementation and fails the default
prefill contract, even inside compiled orchestration. Shared implementation is
allowed when its actual phase-dependent shapes and execution satisfy the contract.

For MoE, derive worst-case receives from every source rank's dispatched token
tile and selected routes. Decode-sized communication buffers do not establish
prefill capacity. Identify the runnable structure check and its deliberate
tokenwise full-stack negative control; inspect both source and generated evidence.
