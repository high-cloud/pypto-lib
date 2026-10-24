# Future serving ABI

Spec revision: draft. This file defines interfaces only; serving is not executed.

The runnable model prefill/decode entries in `MODULES.md` are current required
deliverables. This file specifies how a future serving adapter will call them;
the serving exclusion does not defer implementation of those model entries.

- Inspected serving repository revision, or unavailable assumptions:
- Model matching, capabilities, and limits:
- Prefill entry point and ordered arguments:
- Decode entry point and ordered arguments:
- Weight preparation entry point:
- Dynamic axes and compile-time versus runtime values:
- Token, position, length, mask, block table, and slot mapping semantics:
- KV page size, layout, dtype, ownership, allocation, and update semantics:
- Other persistent state: shape, dtype, ownership, initialization, and updates:
- Logits / hidden / token output semantics:
- Embedding and sampling location:
- Tensor lifetime, aliasing, completion, and buffer reuse requirements:
- Proposed compile/runtime argument builders and startup compatibility checks:
- Unsupported capabilities and integration questions:
