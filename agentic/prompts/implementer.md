# Implementer

Stage: `IMPLEMENTER`. Writable: `*/environment/**`, `*/instruction.md`, `*/task.toml`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Build the agent-visible side of one task.

* `environment/` contains: Dockerfile, simulator package, public profile, public training code, artifact contract, a zero `policy.json`.
* It must NOT contain the hidden profile, final-test seeds, anchors, or the challenge reference.
* `instruction.md` states goal, action space, what may be modified, the artifact path, and limits — with no hint of the hidden mechanism or the reference.
* `task.toml`: continuous reward, timeouts, CPU/RAM/storage, `no-network`, separate verifier environment.
* Every biological or physical term you add must be labelled [P] physical, [B] biological, or [S] synthetic, with a one-line rationale.
