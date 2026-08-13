# NeuroFold-Harbor v8

Five Harbor-compatible tasks that ask an agent to **control** a stochastic,
sequence-conditioned coarse-grained conformational surrogate of an aggregation-competent
protein region: suppress pathological β-register formation without damaging the chain.

Each task ships a frozen simulator, a frozen challenge reference policy that anchors reward
1.0, a hidden evaluation split, and the full validation record that decides whether the task
measures anything at all.

## Purpose

Most protein-ML benchmarks score a *prediction* against a known answer. This one scores a
**sequential decision policy** against a stochastic environment where the answer depends on
what the agent already did.

The design question behind it: what does an agent have to be able to do that a static
predictor cannot? Three properties are built in deliberately and then *tested for*, not
assumed:

* **Relational reasoning** — the class-interaction matrix is non-separable (σ₂/σ₁ = 0.87), so
  the energetic consequence of a contact cannot be recovered from the two beads' own
  features. Partner identity carries information.
* **Order dependence** — maturation and locking are hysteretic (τ_mat = 32). The same action
  multiset applied in a different order gives a different outcome.
* **Acting under partial observability** — latent oxidative stress and chaperone capacity are
  never observed, and crowding drifts upward, so the window for cheap intervention closes.

Every one of these claims is checked by a gate that can fail, and one of them did fail — see
[Validation status](#validation-status). A benchmark whose difficulty is asserted rather than
measured is not a benchmark; the validation harness in `_dev/` and the evidence in `agentic/`
are as much the deliverable as the tasks.

## Scientific scope

This is a **physics-informed coarse-grained stochastic conformational-control benchmark** at
5 residues per bead. It is explicitly **not**:

* atomistic or coarse-grained molecular dynamics — transitions use a Metropolis acceptance
  rule on a coarse energy model, not integrated equations of motion;
* a folding or free-energy predictor;
* an experimentally validated model of aggregation kinetics;
* a disease simulator, and not predictive of any therapeutic effect.

"Hydrogen bond", "chaperone" and "small molecule" appear as **coarse proxies** and are
labelled as such at every use site. Full rationale, including why fragments rather than full
chains, is in [`agentic/specs/SCIENTIFIC_SCOPE.md`](agentic/specs/SCIENTIFIC_SCOPE.md).

## The five tasks

| task | disease | fragment | source |
|---|---|---|---|
| [`alzheimer-abeta42-v8`](alzheimer-abeta42-v8/) | Alzheimer's | Aβ 1–42, full peptide | APP amyloid-β region |
| [`parkinson-alpha-synuclein-v8`](parkinson-alpha-synuclein-v8/) | Parkinson's | NAC domain 61–95 + flanks (55–105) | UniProt P37840 |
| [`alzheimer-tau-v8`](alzheimer-tau-v8/) | Alzheimer's / tauopathy | PHF6* (592–597) through PHF6 (623–628), 585–635 | UniProt P10636 |
| [`als-ftd-tdp43-v8`](als-ftd-tdp43-v8/) | ALS / FTD | LCD aggregation hotspot 311–360 | UniProt Q13148 |
| [`huntington-htt-polyq-v8`](huntington-htt-polyq-v8/) | Huntington's | exon-1 surrogate: N17 + polyQ36 + P11 | HTT exon-1 derived |

Fragments are the canonical aggregation-competent regions from the literature, not
runtime-driven truncations. Aβ42 is the **golden template**: built, validated and frozen
first; the other four are ports of the same frozen mechanism. Nothing is per-task tuned —
where a parameter had to change with protein size it changed by a **measured mass ratio**,
recorded in each `environment/profile.json` under `_extensive_parameter_normalisation`.

## The control problem

Two chains start with a partial antiparallel β-register already present. Left alone it
matures, **locks**, and drives pathology and irreversible damage upward.

Each step the agent picks a contact pair and a strength:

```
(i, j, strength)     strength ∈ [0, 1]
```

which applies a **contact-selective energy modulation** to that pair, after which the chain
relaxes under ordinary Metropolis dynamics. It is not a mechanical force — the agent does not
move coordinates, and whether anything moves is decided by the dynamics. A finite action
budget makes stronger actions cost more.

The agent submits only a **numeric artifact** (`/logs/artifacts/policy.json`, 2541
parameters, a 2-layer message-passing graph policy). No agent-written code runs during
scoring: the verifier reads one JSON file and validates it against a strict contract
(schema string, exact architecture, exact parameter count, finite values only, |w| ≤ 30,
≤ 768 KiB).

Reward is continuous in `[0, 1]`. The no-op policy anchors 0.0; the frozen challenge
reference anchors 1.0. Utility is deliberately not a mean:

```
U = 0.60·mean + 0.40·quantile(vals, 0.20) − 1.50·catastrophe_rate
```

so a policy cannot buy a good average with a heavy failure tail. `metrics.json` also reports
an uncapped `extended_score`, so policies stronger than the reference stay distinguishable.

## Harbor task structure

Each task directory is a self-contained Harbor task:

```
<task>/
  instruction.md          agent-facing task statement
  task.toml               Harbor contract: reward = "continuous",
                          environment_mode = "separate", network_mode = "no-network"
  environment/            agent image
    Dockerfile
    neurofold8/           frozen simulator (chem, energy, geom, env, policy, reward)
    neurofold_cli.py      inspect / init-policy / evaluate / publish
    train_cmaes.py        working sep-CMA-ES baseline over the policy vector
    policy_runtime.py     artifact schema + validator
    profile.json          public profile: 64 train seeds, 32 validation seeds
    reward_calibration.json
  tests/                  verifier image
    Dockerfile
    test.sh               entrypoint
    verifier.py           reads only /logs/artifacts/policy.json
    hidden_profile.json   hidden split: shifted regime, final-test seeds, frozen anchors
    neurofold8/           identical frozen simulator
  solution/
    solve.sh              publishes the reference policy — the oracle
    challenge_reference.json
```

`environment/` never contains final-test seeds, frozen anchors or the reference policy. That
is checked mechanically by the audit, per task, on every run.

## Running a task

Requires Docker. No network access is used at any stage.

```bash
git clone https://github.com/lysyloxidase/neurofold-harbor.git
cd neurofold-harbor/alzheimer-abeta42-v8

docker build -t nf8-env  environment/
docker build -t nf8-test tests/

docker volume create nf8-logs

# oracle run: publish the frozen reference policy
docker run --rm --network none -v nf8-logs:/logs \
           -v "$PWD/solution:/solution:ro" nf8-test sh /solution/solve.sh

# verify
docker run --rm --network none -v nf8-logs:/logs nf8-test bash /tests/test.sh
docker run --rm -v nf8-logs:/logs nf8-test cat /logs/verifier/reward.txt   # -> 1
docker run --rm -v nf8-logs:/logs nf8-test cat /logs/verifier/metrics.json
```

Working inside the agent environment:

```bash
docker run --rm -it --network none -v nf8-logs:/logs nf8-env bash

python /app/neurofold_cli.py inspect                              # task facts, tensor shapes
python /app/neurofold_cli.py init-policy --out /app/policy.json   # zero policy
python /app/train_cmaes.py --budget 6000 --out /app/policy.json   # baseline optimizer
python /app/neurofold_cli.py evaluate --policy /app/policy.json --split validation
python /app/neurofold_cli.py publish  --policy /app/policy.json   # -> /logs/artifacts/policy.json
```

Reproduce the full audit for all five tasks (builds ten images, runs ten evaluations):

```bash
python3 _dev/final_audit.py           # -> agentic/reports/audits/final_audit.json
python3 _dev/test_artifact_contract.py alzheimer-abeta42-v8
```

## Validation status

**Harbor deliverable gate — 5/5 tasks PASS, all eight checks.** Layout, `task.toml` contract,
both Docker builds, oracle reward = 1.0, malformed-artifact rejection, no hidden-state leakage
into `environment/`, split disjointness, freeze-hash integrity. Machine-readable:
[`agentic/reports/audits/final_audit.json`](agentic/reports/audits/final_audit.json).

| task | oracle reward | porting gate | A1 relational necessity |
|---|--:|---|---|
| `alzheimer-abeta42-v8` | 1.0 | 8/8 PASS | **PASS** (+0.557, 95% CI [+0.126, +0.698]) |
| `huntington-htt-polyq-v8` | 1.0 | 8/8 PASS | INCONCLUSIVE (+0.200, 95% CI [−0.120, +0.432]) |
| `alzheimer-tau-v8` | 1.0 | 8/8 PASS | not run |
| `parkinson-alpha-synuclein-v8` | 0.999999999953 | 7 PASS, 1 INCONCLUSIVE (P7) | not run |
| `als-ftd-tdp43-v8` | 1.0 | 6 PASS, 1 INCONCLUSIVE (P7), **1 FAIL (P8)** | not run |

α-synuclein's shortfall from 1.0 is 5×10⁻¹¹ — float rounding inside the container against the
host-measured anchor. The audit tolerance is 1e-9.

Before a reference policy is trained for a protein, the protein must pass an eight-check
**porting gate** (P1–P8) proving the mechanism actually transferred: geometry responds, reward
calibration is non-degenerate, no-op is not optimal, pathology matures, targeted action beats
blind action, order matters, catastrophe is controllable.

Verdicts are **three-valued** — PASS / INCONCLUSIVE / FAIL. A confidence interval spanning
zero means insufficient power, not evidence of absence; conflating the two would have silently
passed off two null results as findings.

## Limitations

Stated rather than hidden. None of these were tuned away.

* **TDP-43 P8 = FAIL.** The best of six hand-coded probe policies left 16.7% of episodes
  catastrophic against a 10% threshold. The verdict was not re-opened. A learned controller,
  evaluated under a rule fixed **before** the run on 256 fresh seeds, reaches **8.59%
  (22/256), 95% CI [5.5%, 12.1%]** — point estimate below threshold, interval crossing it.
  Both results are reported together; neither is quotable alone. Full record:
  [`agentic/reports/audits/tdp43_final_report.md`](agentic/reports/audits/tdp43_final_report.md).
* **α-synuclein P7 = INCONCLUSIVE.** Order effect d = −0.014, 95% CI [−0.175, +0.147] at 96
  development seeds: limited power, not absence of hysteresis.
* **A1 was run on two tasks only.** Relational necessity is *demonstrated* on Aβ42 and *not
  demonstrated either way* on HTT. It is unmeasured on tau, α-synuclein and TDP-43.
* **A2–A5 and A8 are unrun.** History necessity, effective dimensionality, RL-vs-black-box and
  precision-stability gates are specified in
  [`agentic/specs/ACCEPTANCE_CRITERIA.md`](agentic/specs/ACCEPTANCE_CRITERIA.md) but were not
  executed for v8.
* **The reference is an anchor, not an optimum, and its strength varies a lot by task.** On the
  hidden shifted regime the frozen reference leaves a catastrophe rate of 6.2% on Aβ42 but
  34.4% on HTT. Reward 1.0 means "matched this task's reference", so the tasks are not equally
  hard at equal reward. Use the uncapped `extended_score` when comparing strong policies.
* **Difficulty is `unbenchmarked`.** No external agent has attempted these tasks. There is no
  human baseline and no evidence about where current agents land.
* **The evaluation split is public in this repository.** See below.

### The hidden split is published

A Harbor task bundle includes its verifier and its solution, so `tests/hidden_profile.json`
(final-test seeds, shifted regime, frozen anchors) and `solution/challenge_reference.json`
(the oracle policy) are both in this repository. "Hidden" means hidden **from the agent at
runtime** — the agent image never contains them — not secret from a reader.

The consequence is that the published final-test split is no longer blind. Anyone can read the
128 seeds and the reference policy, and a policy tuned directly against them proves nothing.
For a genuinely blind evaluation, regenerate a private split:

```bash
python3 _dev/freeze_task.py --task <task> --episodes 128   # new per-task seed block + anchors
```

The seed block is drawn from a SHA-256 of the task slug, disjoint by construction from every
train, validation, calibration and development split.

## Governance

The rules the build ran under are in
[`agentic/specs/FORBIDDEN_CHANGES.md`](agentic/specs/FORBIDDEN_CHANGES.md). In short:

* Final-test seeds are generated **only at freeze**, per task, and never inspected during
  development.
* v7's final-test split is treated as **contaminated** and is never reused.
* No reward, threshold, chemistry, architecture or budget parameter was changed after any
  final-test or diagnostic result was seen.
* An acceptance threshold is never changed after seeing the result it judges.
* A failing mechanism is never ported. When HTT's polyQ register proved uncontrollable, the
  root cause was measured — the frozen chemistry lacked glutamine side-chain hydrogen bonding,
  giving only 1.8 simultaneously-held ladder pairs against Aβ42's 7.2 — the freeze was
  re-opened with authorisation, the polar-zipper mechanism was added to the **shared**
  chemistry, and all five tasks were re-validated and re-frozen.
* Performance refactors require numerical-equivalence tests.

State, hashes and split registry: [`agentic/state/`](agentic/state/). Role prompts for the
build workflow: [`agentic/prompts/`](agentic/prompts/).

## Repository layout

```
alzheimer-abeta42-v8/            golden template task
parkinson-alpha-synuclein-v8/
alzheimer-tau-v8/
als-ftd-tdp43-v8/
huntington-htt-polyq-v8/
agentic/
  specs/                         scope, acceptance criteria, forbidden changes
  state/                         benchmark state, split registry, frozen hashes
  reports/                       validation, experiments, audits
  prompts/                       role prompts for the build workflow
_dev/
  porting_gate.py                P1-P8
  verdict.py                     three-valued PASS / INCONCLUSIVE / FAIL
  new_task.py                    scaffold a protein, with extensive-parameter normalisation
  train_reference.py             sep-CMA-ES reference trainer
  freeze_task.py                 hidden profile, per-task final seeds, anchors
  final_audit.py                 the eight-check Harbor gate, all tasks
  test_artifact_contract.py      artifact-contract red-team
  package.py                     per-task and combined ZIPs
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
