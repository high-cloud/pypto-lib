# Acceptance criteria

Spec revision: draft. Numerical thresholds require agreement before acceptance.

The launched model Main Agent owns execution and closeout of every required gate.
Independent semantic references are required; external parent review is not.

## Required delivery gates

| Requirement | Source / artifact | Verification command | Required result |
| --- | --- | --- | --- |
| Complete-model compiled prefill | TBD | TBD | Graph covers embedding, every layer, final norm, and head |
| Complete-model compiled decode | TBD | TBD | Same coverage; consumes and updates agreed persistent state |
| Prefill execution structure | TBD | TBD | Multi-token within-layer work; actual loop/shape evidence matches the frozen contract |
| Structural check negative control | TBD | TBD | Tokenwise complete-stack prefill is rejected, including renamed entry wrappers |
| First working paths | TBD | TBD | Attention and MoE/feed-forward structure checked before full-stack expansion |
| Engineering structure | TBD | TBD | Current ownership/call graph follows model rules; validation exercises delivered symbols and variants |
| Default validation and discovery | TBD | TBD | Bounded entries use existing harness/comparators and applicable CI discovery recognizes them |
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

## Engineering gates

Use the [model rules](../../../.claude/rules/models.md) as the canonical criteria.
The owner records evidence in the existing `MODULES.md` and `TASKS.md`:

- Before implementation: complete the file-ownership and validation map.
- On the first working attention and MoE/feed-forward paths: inspect actual
  dependencies, call depth, phase shapes, and default validation; resolve structural
  findings before expanding the pattern to the full model. Check each distinct
  mixer as it becomes executable.
- At closeout: update the map, identify the exact symbols and variants called by
  validation and composition, and account for obsolete paths and their consumers.
  Execute the bounded default validation entries and applicable existing CI
  discovery checks, recording commands, exit codes and discovered entries.

An entry that validates another variant does not accept the delivered variant.
Validate every delivered variant; retain other implementations only with an
explicit consumer and their own acceptance evidence. Repository sources must
contain the permanent validation code. Frozen fixture data and logs may remain
in experiment artifacts. Reuse
existing harness/comparator capabilities and add only missing model semantics.
Ruff success, source hashes, or generation cannot substitute for these checks.
Document permitted exceptions under the existing rules with actual consumers
and responsibilities; unresolved violations keep the corresponding row open.

## Execution-structure gates

These gates are required even without performance targets. Freeze representative
supported prefill lengths greater than one and a decode fixture. Inspect the
actual source call paths and generated IR/task graphs after helper expansion.
Record the following for every fixture and phase:

- Source/build identity, entry symbol, command, and actual checker result.
- Loop nesting and bounds; which loop encloses the complete decoder stack.
- Logical token axes, token tiles, and effective projection, attention/recurrent,
  feed-forward, and routed/shared-expert operand shapes.
- State updates and dispatch/combine granularity, including worst-case MoE receives.
- Evidence paths and locations supporting each conclusion; unknown evidence fails.

Require token tiles greater than one in prefill projections and feed-forward
work. Allow tiled scheduling and necessary causal/recurrent scans. Reject an
outer prompt-token loop that executes the complete decoder stack per token;
numerical equivalence, readable generation, and a compiled loop do not waive this
gate. Two identical entry wrappers are not proof of phase specialization, while
shared helpers with verified phase-dependent behavior are allowed.
Count logical prompt tokens rather than heads or padding. Sparse expert groups
may be empty or contain one row; require a repeated-expert fixture to demonstrate
that the implementation can group multiple tokens for expert computation.

Use the existing validation entry and inspection tools, adding only necessary
model-specific checks, against the real compiled path and
a deliberately tokenwise full-stack negative control. The latter must be
rejected even after renaming the entry. Check generated shapes and nesting,
not only source spelling, kernel names, or self-reported booleans. Missing or
unrecognized evidence must not produce a pass. Save both actual results before
complete-model generation. A tokenwise-only exception requires an explicit user
scope decision and must remain visible in the final delivery status.

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
result. Engineering acceptance, structural acceptance and its negative control
are required rows; module accuracy or generation cannot substitute for them.
Do not end with external parent acceptance pending.
