# Goal-driven model implementation

Use [PROMPT.md](PROMPT.md) as the Main Agent prompt and
[WORKER_PROMPT.md](WORKER_PROMPT.md) for module workers. Start with one goal,
one topology, and one quantization policy. Track progress with a short checklist
in `TASKS.md`; no workflow engine or state machine is required.

Copy the [templates listed below](#files) into an experiment directory. Set local paths
through environment variables or an ignored local input file. Keep checkpoints
read-only, outside the experiment. Store logs, fixtures, and generated builds
under the experiment's local artifact directory.

The user confirms the proposed model scope, topology, quantization policy,
interfaces, and acceptance criteria before kernel implementation. Routine fixes
within that agreement do not need another approval. Bring back only changes to
the agreed behavior or interfaces.

The current deliverable ends at real-weight generation. Define the future
serving ABI during specification, including recurrent state when applicable;
serving implementation and serving validation are future work.

## Launch and completion ownership

The launched model agent is Main and owns the complete implementation goal.
It implements the model, runs acceptance against independent references, fixes
failures, and closes the native goal. The launcher passes inputs and user
decisions and reports progress; external parent acceptance is not a completion
dependency. Module workers are optional and require delegation authorization.

Prepare and agree the specification before activating the implementation goal.
Then start or resume that same owner thread using the host's supported controls.
Verify its actual native goal status and a running turn, and retain the returned
handle. A prompt file or queued message alone is not proof of active continuation.
If the host cannot restore the existing goal, report that lifecycle limitation
explicitly instead of claiming the task is running or creating a duplicate.
See the [Codex Goals guide](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)
for the thread-scoped goal contract; available controls depend on the host.

Freeze concrete model entry paths and the compilation boundary during agreement.
Completion requires runnable complete-model PyPTO prefill/decode orchestration,
generated graph coverage, and real-weight device generation through those entries,
alongside module numerical gates and state/weight checks. The owner maps each
requirement to current evidence and records the actual native completion result.
Keep this process in the existing documents and checklist.

Apply the [model implementation rules](../../.claude/rules/models.md) through
three checkpoints: map file ownership, goldens, default validation and consumers
before implementation; check the first working attention and MoE/feed-forward
paths before full-stack expansion; verify final ownership, validation-to-composition
symbols, replacement cleanup and default validation/discovery before goal closeout.
The model agent owns these checks. Keep evidence in `MODULES.md` and `TASKS.md`;
reuse the existing harness, comparators and inspection tools. Model-specific
checks extend the owning validation entry instead of creating another framework.
Permanent validation code belongs in repository sources; artifacts may hold frozen
fixtures and execution records. Numerical or generation passes do not waive
engineering findings.

Execution structure is a separate required gate. Multi-token prefill must retain
prompt-token tiles within each layer's projections and feed-forward work;
model-required recurrence and tiled scheduling are allowed. An outer token loop
through the complete decoder stack is diagnostic-only by default. Excluding
performance tuning does not authorize that substitution. Shared helpers are
allowed only with verified phase-dependent behavior. The owner checks actual
source and generated loop/shape evidence and proves that its runnable structural
check, integrated with existing validation, rejects a tokenwise complete-stack
negative control. This gate does not
require a parent reviewer or a latency target. See the
[execution-structure gates](templates/ACCEPTANCE.md#execution-structure-gates).

Before the implementation goal, verify checkpoint/device memory feasibility and
calibrate generation fixtures against the original model with the intended chat
messages and token budget. Freeze numerical gates and generation semantics
separately. Equivalent wording requires semantic review; keyword presence alone
does not establish correctness. Report blockers promptly with a concrete remedy.

For distributed MoE work, record physical ranks and overlapping parallel groups,
replicated versus sharded weights, per-rank memory, request/state ownership, and
dispatch/combine validation. A two-card attention-DP2/expert-EP2 layout uses the
same two ranks for both roles; it does not imply four cards. Require real cross-rank
expert traffic and safe participation when one request finishes before its peer.

## Required inputs

- A local checkpoint, configuration, tokenizer, and original model source.
- The current PyPTO environment and its pinned toolchain.
- Target hardware availability and a proposed single topology.
- One proposed quantization policy, including any checkpoint conversion.
- Original-framework reference code and small representative input fixtures.

## Files

| File | Purpose |
| --- | --- |
| `MODEL_SPEC.md` | Model facts, agreed scope, topology, and quantization |
| `MODULES.md` | Module graph, tensor interfaces, owners, and composition |
| `ACCEPTANCE.md` | Delivery gates, per-output numerical gates, and generation cases |
| `WEIGHT_MANIFEST.md` | Checkpoint-to-kernel mappings and completeness |
| `SERVING_ABI.md` | Future serving adapter and persistent-state contract |
| `TASKS.md` | Checklist, commands, results, failures, and resumption notes |

Create fixture metadata and runnable reference/validation scripts as the model
requires them; do not add empty scripts merely to populate a directory tree.
Technical instructions remain canonical in `docs/pypto-coding/`,
`docs/run-and-validate/`, and `docs/debug-and-tune/`.
