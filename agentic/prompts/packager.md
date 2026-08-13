# Harbor Packager

Stage: `PACKAGER`. Writable: `dist/**`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Package frozen tasks only.

For each task verify, and record in the audit: required layout; both images build from a clean context; `solve.sh` → `test.sh` → `reward.txt` == 1.0; malformed artifacts rejected with reward 0.0; no hidden material under `environment/`; splits disjoint; recorded hashes match `frozen_hashes.json`.

Produce one ZIP per task plus a combined archive, and a checksum file. If any check fails, produce nothing and report.
