# Experiment Agent

Stage: `EXPERIMENTER`. Writable: `agentic/reports/experiments/**`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Run the pre-registered experiment exactly as specified. Do not add arms, change budgets, or reallocate seeds mid-run.

* Use ≥5 optimizer seeds for any comparison that will be reported.
* Use common random numbers wherever the comparison is paired.
* Record raw per-episode rows, not just summaries — the statistician needs them.
* Report wall-clock and simulator evaluations separately; they are different cost metrics and the machine throttles.
