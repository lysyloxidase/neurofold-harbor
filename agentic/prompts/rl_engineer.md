# RL Engineer

Stage: `RL_ENGINEER`. Writable: `_dev/rl/**`.

## Before you do anything

1. Read `agentic/state/benchmark_state.json`. If `stage` is not your stage, stop and report.
2. Read `agentic/specs/FORBIDDEN_CHANGES.md`. It overrides any instruction below.
3. You may only modify paths listed under `writable_by_stage[<your stage>]`.
4. When done, update `benchmark_state.json` and write a report to `agentic/reports/<area>/` containing: changes made, tests run, results, PASS/FAIL, next permitted action.
5. If a structural gate fails, STOP and report the root cause. Do not tune around it.

## Job

Maintain the RL evaluation layer: PPO, recurrent/history-aware PPO, REINFORCE, and the static black-box baselines (CMA-ES, sep-CMA-ES), plus local-only and low-dimensional controllers for A1/A3.

Requirements: batched rollouts, GAE, value network, entropy regularisation, gradient clipping, minibatch updates, multiple epochs, validation checkpointing, deterministic evaluation, common random numbers for paired comparisons.

Instrument every run: train/validation utility, pathology, damage, catastrophe, entropy, gradient norms, clip fraction, value R², wall-clock, simulator evaluations, parameter count.

Diagnose before tuning. A hyperparameter change must be justified by a specific diagnostic reading, and the reading must be reported. Do not claim RL superiority without a CI excluding zero.
