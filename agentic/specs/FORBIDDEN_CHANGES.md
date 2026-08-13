# Forbidden changes

Violating any of these invalidates the benchmark. They are checked mechanically
where possible by `_dev/final_audit.py` (layout, `task.toml` contract, Docker
builds, oracle reward, malformed-artifact rejection, leakage, split
disjointness, freeze-hash integrity) and `_dev/test_artifact_contract.py`
(artifact-contract red-team).

## Data governance

* **Never inspect or reuse v7 final-test seeds.** They were used during v7
  development and analysis and are contaminated for v8 purposes.
* **Never generate v8 final-test seeds before freeze.** Order is: freeze
  simulator, mechanism, observation schema, reward, anchors and docs → hash
  everything → only then generate seeds.
* **Never inspect final-test outcomes during development.** They are run once,
  at grading time.
* **Never tune reward weights, physics or calibration after seeing final-test
  results.**
* **Never change an acceptance threshold after seeing the result it judges.**
  Thresholds live in `ACCEPTANCE_CRITERIA.md` and are set before the run.
* All splits stay disjoint: train, public validation, calibration, author
  validation, final test. Enforced by `split_registry.json`.

## Process

* **Never port a mechanism that failed its porting gate.** A failing gate stops
  that protein and is reported with a root cause, not tuned around silently.
* **Never select a winner from point estimates** when intervals overlap.
* **Never claim RL superiority** without a CI excluding zero.
* **No performance refactor that alters dynamics** without a numerical-
  equivalence test demonstrating bit-identical or bounded-difference output.
  Performance work is allowed only after Aβ42 is stable.

## Artifact and verifier integrity

* The verifier must **never import or execute agent-authored code**. It reads
  one JSON file and validates it.
* `environment/` must never contain the hidden profile, the final-test seeds,
  the challenge reference, or the anchors.
* The artifact contract may not be loosened: schema string, exact architecture,
  exact parameter count, finite values only, `|w| ≤ 30`, file size ≤ 768 KiB.

## Framing

* Do not describe the environment as molecular dynamics, a folding free-energy
  predictor, an experimentally validated model, or a disease simulator.
* Do not describe the intervention as a mechanical force.
* Do not present the challenge reference as optimal. It is a reference, and the
  uncapped `extended_score` exists so stronger policies stay distinguishable.
* Do not remove or soften a negative result.
