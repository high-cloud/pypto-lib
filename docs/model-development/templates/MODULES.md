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
