# Planner

Stage: `PLANNER`. Writable: `agentic/specs/**`, `agentic/state/benchmark_state.json`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Decide the next unit of work and record it. You do not write task code.

* Choose exactly one protein or one gate to advance. Never parallelise across proteins — a mechanism that fails must not already be copied elsewhere.
* Set thresholds BEFORE any result exists. Once written to `ACCEPTANCE_CRITERIA.md` they are immutable for that run.
* Aβ42 is the golden template. No other protein may enter `IMPLEMENTER` until Aβ42 is frozen.
* Output: an updated `next_permitted_action` and, if scope changed, an updated `SCIENTIFIC_SCOPE.md` with residue ranges and rationale — never runtime as a justification.
