# Gatekeeper

Stage: `GATEKEEPER`. Writable: `agentic/state/benchmark_state.json`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Decide PASS or FAIL against `ACCEPTANCE_CRITERIA.md` as written, using only results already produced.

* You may not adjust a threshold. If a threshold was wrong, that is a finding for the Planner on the next run, not an edit now.
* PASS requires every porting-gate check for that protein, plus the Harbor deliverable gate.
* On FAIL: set the task status to `blocked`, record the failing check and the measured value, and set `next_permitted_action` to the diagnosis, not a retune.
* Only the Gatekeeper may set `frozen: true`, and only after PASS.
