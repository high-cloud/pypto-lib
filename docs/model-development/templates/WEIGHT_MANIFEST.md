# Weight mapping

Spec revision: draft.

| Checkpoint name | Source shape / dtype | Transform / quantization | Rank-local shape / dtype | Kernel argument |
| --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD |

Record transpose, reshape, concat, padding, quantization axes/scales/zero-points,
ties/shared storage, rank slices, and hashes as appropriate. Account for every
required source and destination. List intentionally excluded tensors separately
with their agreed reason. Generate a machine-readable manifest if needed.

For MoE, distinguish total parameters, activated parameters and rank-resident
parameters. Record routed-expert ID ranges per rank and every replicated dense,
router, shared-expert and head tensor. Include padding/scales and peak staging or
conversion copies in memory estimates. Do not divide replicated bytes by EP size.
