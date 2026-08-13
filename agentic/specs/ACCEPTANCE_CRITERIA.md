# Acceptance criteria

Thresholds are **fixed here before results** and may not be changed after seeing
them (see `FORBIDDEN_CHANGES.md`). A gate that fails stops the pipeline; the
root cause is reported rather than tuned around.

## Porting gate — must pass before a reference policy is trained for a protein

| id | check | criterion |
|---|---|---|
| P1 | geometry responds | every controllable DOF moves the 3-D configuration; energy responds to every bead |
| P2 | reward calibration non-degenerate | no component MAD below `min_rel_scale`; `DegenerateCalibration` not raised; clipping fraction < 5% per component |
| P3 | no-op not optimum | at least one reference controller beats no-op, paired CRN, 95% CI excludes 0 |
| P4 | pathology can mature | under no-op, mean locked pairs > 1.0 |
| P5 | targeted action improves outcome | oracle vs no-op paired `d_z` ≥ 0.6, CI excludes 0 |
| P6 | blind/wrong targeting worse | oracle vs blind CI excludes 0 in oracle's favour; blind ≤ no-op + 0.05 |
| P7 | action order matters | identical action multiset, different order: CI on the difference excludes 0 |
| P8 | catastrophe controllable | oracle catastrophe rate < 10% |

## Structural gates — quality, evaluated on public splits before freeze

| id | claim under test | criterion |
|---|---|---|
| A1 | relational necessity | full edge-aware controller > local-only; hierarchical 95% CI excludes 0; effect ≥ 10% of gain over the zero anchor |
| A2 | sequential/history necessity | history-aware > matched static controller; CI excludes 0 |
| A3 | effective dimensionality | a ≤40-parameter controller must NOT reach ≥95% of full-policy gain |
| A4 | RL vs static black-box | reported, not required. RL superiority claimed only where CI excludes 0 |
| A5 | message-path ablation | deleting message weights measurably degrades utility and/or safety |
| A6 | action-order dependence | per-task unit test: `U(seq A) ≠ U(seq B)` for identical multisets |
| A7 | safety | strong policies: catastrophe < 10%, damage below the no-op anchor, no utility/safety exploit |
| A8 | statistical precision | ≥32 public validation episodes; ranking stable under q10/q20/q30 |

## Harbor deliverable gate — per task, before packaging

| id | check |
|---|---|
| H1 | required layout present: `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/test.sh`, `solution/solve.sh` |
| H2 | `reward = "continuous"`, `network_mode = "no-network"`, `environment_mode = "separate"` |
| H3 | both Docker images build from a clean context |
| H4 | `solve.sh` → `test.sh` → `reward.txt` == 1.0 (tolerance 1e-9) |
| H5 | malformed, NaN, wrong-dimension, out-of-bounds and oversize artifacts are rejected with reward 0.0 and an error, never a crash |
| H6 | verifier imports no agent-authored code and reads only `/logs/artifacts/policy.json` |
| H7 | no hidden leakage: final-test seeds, hidden profile and reference policy absent from `environment/` |
| H8 | all splits disjoint: train, public validation, calibration, author validation, final test |

## Verdicts are three-valued (binding)

A confidence interval spanning zero is **not** evidence of no effect. Collapsing
"no evidence of an effect" into FAIL is a methodological error: it lets absence
of power masquerade as a demonstrated negative, and it invites tuning against
noise. Every interval-based gate therefore returns one of:

| verdict | meaning |
|---|---|
| **PASS** | CI excludes zero in the hypothesised direction (either direction for two-sided gates such as A6/P7). |
| **FAIL** | Either the CI excludes zero in the OPPOSITE direction — evidence against the hypothesis — or the entire CI lies inside the negligible band, which is a genuine equivalence result. |
| **INCONCLUSIVE** | The interval spans zero and is wider than the negligible band. Limited power. **Does not block.** |

The **negligible band** is declared before any result is seen: 10% of the
optimisation gain over the zero anchor, the same fraction A1 uses as its
minimum interesting effect size.

Only **FAIL** blocks a port. INCONCLUSIVE is carried forward and must be stated
explicitly wherever the task is reported — it may never be silently rendered as
a pass.

## Statistical method (binding)

* Paired comparisons use **common random numbers**: identical initial state, OU
  path and acceptance draws at every step index.
* Uncertainty uses a **two-stage hierarchical bootstrap** (optimizer runs ×
  episodes) with the **full nonlinear utility recomputed inside every draw**.
  A difference-of-means bootstrap is invalid here and is not used.
* Winners are never selected from point estimates alone. Where intervals
  overlap, the result is reported as **statistically unresolved**, and that is
  not evidence of equivalence — it is reported as limited power.
