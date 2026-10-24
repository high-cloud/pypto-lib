# Model specification

- Revision: draft; user agreement pending.
- Model/checkpoint identity and revision:
- Config, tokenizer, chat template, and original forward sources:
- Scope (text/multimodal, supported modes, exclusions):
- Target platform, device count, parallel degrees, and rank mapping:
- Quantization policy and checkpoint conversion:
- Weight, activation, accumulation, communication, and state dtypes:
- Supported batch, context, prefill, and decode shapes:
- Memory estimate (weights, state, scratch):
- Goal owner thread and agreed implementation objective:
- Prefill/decode source paths, callable signatures, and compilation boundary:
- Generation entry path and how it invokes compiled model forwards:
- Required deliverables and acceptance rows (including documentation):
- Original generation preflight, fixture-selection policy and frozen case IDs:
- Physical world/rank table; attention, expert, tensor and pipeline group membership:
- Per-rank request count; dense/shared replication and routed-expert ownership:
- Dispatch/combine protocol, maximum receives, EOS/idle-rank participation:
- Observed usable HBM per rank and peak memory breakdown with reserve:
- Checkpoint download/cache location, pinned revision, bytes and local availability:

## Architecture facts

| Fact | Value | Evidence |
| --- | --- | --- |
| Layer types and order | TBD | TBD |
| Hidden and projection dimensions | TBD | TBD |
| Attention, positions, and mask | TBD | TBD |
| Norm and feed-forward computation | TBD | TBD |
| Persistent state | TBD | TBD |
| Embedding, vocabulary, and LM head | TBD | TBD |

## Decisions for user agreement

List concrete alternatives, recommendations, and outstanding facts here.
