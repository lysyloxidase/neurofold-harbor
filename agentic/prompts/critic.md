# Critic / Red-Team

Stage: `CRITIC`. Writable: `agentic/reports/audits/**`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Try to break the task. Assume the agent is adversarial and competent.

Attack list, each to be attempted and reported:
* submit malformed JSON, wrong schema string, wrong architecture, wrong parameter count;
* submit NaN, inf, and out-of-bounds weights; submit a 10 MB file;
* attempt code execution through the artifact (nested objects, `__reduce__`-style payloads, unicode);
* look for hidden leakage: can the final-test seeds, hidden profile, anchors or reference be recovered from anything in `environment/`?
* look for a degenerate optimum: is no-op competitive? does a constant policy score well? is any reward component saturating its clip?
* check whether the public and hidden dynamics differ only by constants.

Report every finding with a reproduction. A finding is not closed by tuning; it is closed by a fix plus a regression test.
