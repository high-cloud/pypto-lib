# Experiment log

## Ownership and native goal

- Launched model Main Agent / owner thread:
- Agreed spec revision and full implementation objective:
- Actual native goal handle/status and observation time:
- Actual start/resume result and running turn evidence:

Activate the implementation goal after specification agreement. Reuse its owner
thread and preserve historical host results. The owner runs and closes acceptance;
the launcher is not a required reviewer. Record lifecycle limitations accurately.

## Checklist

- [ ] Model facts and environment inspected
- [ ] Concrete specification reviewed and agreed by user
- [ ] Pinned checkpoint availability and per-rank peak memory feasibility verified
- [ ] Generation cases calibrated on the original model and frozen before candidate runs
- [ ] Native implementation goal active in the owner thread
- [ ] Independent references and fixtures prepared
- [ ] Modules independently accepted
- [ ] Half-layers / representative layers accepted
- [ ] Prefill and consecutive decode composed
- [ ] Complete-model PyPTO prefill/decode source entries delivered and compiled
- [ ] Generated graph coverage and weight/state interfaces verified
- [ ] Persistent-state handoff checked
- [ ] Distributed routing/combine and unequal EOS/state isolation checked when in scope
- [ ] Real-weight mapping checked
- [ ] Real-device generation through compiled model entries accepted
- [ ] Future serving ABI documented
- [ ] Every agreed requirement mapped to passing evidence on delivered revision
- [ ] Owner recorded actual native goal completion

## Requirement closeout

| Requirement | Delivered source / artifact and revision | Command / actual result | Remaining work |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |

Include every agreed requirement. Keep optional diagnostics separate. Continue
work on unmet implementable rows instead of handing acceptance to the launcher.

## Evidence

Append commands, revisions, actual import paths, topology, exit codes, metrics,
artifact references, failures, and exact resumption instructions. Distinguish
inspection, compile, simulator, module golden, and real-weight generation.
Do not claim a passed check without its execution evidence.
Keep the current checklist, acceptance ledger and timestamped native observation
consistent. Report blockers promptly with exact evidence and a concrete remedy;
preserve historical observations separately and avoid unchanged audit loops.
