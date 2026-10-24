# Acceptance criteria

Spec revision: draft. Numerical thresholds require agreement before acceptance.

The launched model Main Agent owns execution and closeout of every required gate.
Independent semantic references are required; external parent review is not.

## Required delivery gates

| Requirement | Source / artifact | Verification command | Required result |
| --- | --- | --- | --- |
| Complete-model compiled prefill | TBD | TBD | Graph covers embedding, every layer, final norm, and head |
| Complete-model compiled decode | TBD | TBD | Same coverage; consumes and updates agreed persistent state |
| Prefill-to-decode handoff | TBD | TBD | Actual compiled entries pass state and position checks |
| Agreed shape and capacity limits | TBD | TBD | Supported boundaries pass; unsupported inputs follow the contract |
| Real-weight mapping | TBD | TBD | All included weights mapped; exclusions and transforms justified |
| Generation execution path | TBD | TBD | Real device invokes the delivered compiled model entries |
| Documentation and interface agreement | TBD | TBD | Delivered paths, arguments, limits, and commands match sources |
| Memory feasibility | TBD | TBD | Per-rank resident and peak bytes fit observed HBM with reserve |
| Distributed MoE, when in scope | TBD | TBD | Exact card count; distinct DP inputs; local/remote/skewed routing and combine match reference |
| Distributed lifecycle, when in scope | TBD | TBD | Repeated epochs, state isolation and unequal EOS/idle ranks complete safely |

Record entry symbols, source hashes, compile commands, runtime invocation paths,
and generated graph coverage. Placeholder files, aliases, Python per-operator
submission loops, and compile-only success do not pass the model delivery gates.

## Numerical gates

| Module / composition | Fixture | Output or state | Comparator / threshold | Command |
| --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD |

Check shape/dtype, finite values, integer metadata, modified and untouched state,
prefill-to-decode handoff, and all agreed boundary cases. Use representative
original-framework references; measurement-only reports do not count as passes.

## Real-weight generation cases

Calibrate the candidate case pool on the original model before freezing required
cases. Record exact messages, IDs, original outputs and runtime/dtype provenance.
If the baseline fails, apply the authorized original-only selection policy before
candidate execution or resolve the specification; do not freeze an unattainable
answer/budget expectation. Uncalibrated cases cannot silently become hard gates.

| Case / messages / input tokens | Template and decoding settings | Budget / stop / truncation policy | Original result and evidence | Semantic expectation / review | Candidate command |
| --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | Coherent, prompt-relevant text | TBD |

Use deterministic greedy decoding and repeat a fixed case. Record token ids,
text, state progression, termination, revisions, topology, exit code, and logs.
Record the compiled prefill/decode entry and runtime artifact identities used.
Accept equivalent wording against the agreed meaning; keyword proxies do not
replace review. Keep factual capability, smoke, numerical gates and token parity
separate. On distributed runs record both ranks and cross-rank execution evidence.
Full-model golden is optional. Serving execution is outside this acceptance.

## Owner closeout

Map every agreed requirement to an actual result on the delivered revision in
`TASKS.md`. Rerun affected gates after changes. Retain failures and label optional
diagnostics separately. An unmet required row keeps the goal incomplete; continue
when implementable. After all required rows pass, save the native goal completion
result. Do not end with external parent acceptance pending.
