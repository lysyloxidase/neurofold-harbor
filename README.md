# NeuroFold-Harbor v8

**A benchmark, and the measurement that showed it does not test what it was built to test.**

Five Harbor-compatible tasks ask an agent to control a stochastic, coarse-grained
conformational surrogate of an aggregation-competent protein region. They build, run, and
score deterministically; the audit passes 5/5. They were designed to require relational
reasoning over a contact graph.

They do not. A policy with three hand-set weights scores 1.0 on three of the five. That was
established after release, by gates the original release specified and never ran, and it
survived four attempts to design the shortcut away.

So this repository is two things, and the second is the more useful one:

1. **Five runnable Harbor tasks**, with their limits measured rather than asserted. Two of the
   five (Aβ42, TDP-43) still discriminate; three do not.
2. **A validation study with a negative result**, plus the instruments that produced it —
   an effective-dimensionality gate, an ablation verifier, an action-order gate, an
   information-level aliasing test, simulator invariants, and an external-validity test
   against the aggregation literature. None of these existed in the released version. Each
   found a real defect — including the external-validity test, which the model passes 4/4 on
   orderings its own scales force, and **fails 3 of 4** on familial Aβ mutations whose effects
   they do not, twice with the wrong sign.

Every number below is reproducible from a named command — see
[Verifying the claims](#verifying-the-claims).

---

> ## ⚠ Correction to the v8.0.0 release
>
> **This benchmark does not require relational reasoning, and the v8.0.0 release said it
> did.** Measurement after release contradicted the claim.
>
> A policy with **3 non-zero weights out of 2541**, hand-written and never trained, scores
> **1.0 through the released verifier on three of the five tasks** (HTT, α-synuclein, tau)
> and 0.795 / 0.581 on the other two.
>
> One earlier phrasing here has since been **corrected by measurement**, and the correction
> was then itself tested. All five anchors shipped with two sep-CMA-ES restarts, so all three
> "saturated" tasks were retrained with six:
>
> | task | frozen | retrained | 3 weights | verdict |
> |---|--:|--:|--:|---|
> | HTT | 1.000 | **1.608** | 1.562 | anchor was undertrained — **not** saturated |
> | α-synuclein | 1.000 | 0.946 | **1.045** | saturation confirmed |
> | tau | 1.000 | 0.974 | **1.086** | saturation confirmed |
>
> So **two of five** tasks are saturated, not three. On HTT, what remains is that 72 000
> training episodes over 2541 parameters buy 3% over three hand-set weights — inside this
> benchmark's own ±5% resolution.
>
> The claim rested on gate A1, which compares a full graph controller against the *same
> controller with its edges zeroed* — a crippled graph controller, never a cheap one. Both
> arms lose to the three weights. Gate A3, specified precisely to catch this, was never run
> for v8.0.
>
> Two further defects were measured. The reward's largest-weighted term,
> `pathology_reduction`, contributes **0.1%–24%** of the reference's advantage and
> **−101.6%** on α-synuclein, because pathology was not controllable at all under v8.0
> physics: `beta_ladder()` decides ladder membership from geometry, while the agent's only
> lever modulates energy, and the two were never connected. And the porting gate's
> development seeds (2000–2047) fully contain the public validation split (2000–2031).
>
> Four attempts to redesign the benchmark so relational structure *would* be necessary all
> failed, in a way that now looks structural: in a coarse-grained contact-selection task, a
> contact's pathological status is a local property, so the choice of target is locally
> decidable. An information-level test on 88 236 counterfactuals puts a local probe at
> **AUC 0.806** and two rounds of graph aggregation at **0.817** — a gain of **+0.011**.
>
> **The Monte Carlo sampler also has three defects.** Every proposal within a step was the
> same displacement vector; acceptance draws were reused across steps, 480 decisions from 98
> numbers; and the re-formation penalty was charged asymmetrically, so the chain had no
> stationary distribution. They are fixed on the branch `v8.1-physics-fixes` with regression
> tests — and the fix **breaks the task**: targeted and blind intervention become
> indistinguishable (catastrophe 0.78 vs 0.81) and the frozen reference scores 0.00076.
> v8.0's physical parameters were implicitly fitted to the broken sampler. Correcting it
> honestly requires recalibrating the whole damage and reward model, which is not done here.
>
> Full evidence, with every number and how it was produced:
> **[`agentic/reports/audits/v8.0_shortcut_finding.md`](agentic/reports/audits/v8.0_shortcut_finding.md)**
>
> The `v8.0.0` tag is left as released, as the historical record. What the tasks *do*
> measure is described in [What this measures](#what-this-measures) below.

---

## Purpose

Most protein-ML benchmarks score a *prediction* against a known answer. This one scores a
**sequential decision policy** against a stochastic environment where the answer depends on
what the agent already did.

The design question behind it: what does an agent have to be able to do that a static
predictor cannot? Three properties were built in deliberately and then *tested for*. One of
the three did not survive the test.

* **Relational reasoning** — ~~necessary~~ **not necessary; claim withdrawn.** The
  class-interaction matrix is non-separable (σ₂/σ₁ = 0.87), so the energetic consequence of a
  contact genuinely cannot be recovered from the two beads' own features. That is a property
  of the *energy model*, and it does not transfer to the *decision*: choosing which contact to
  act on is locally decidable, and three hand-set weights exploit exactly that. See the
  correction above.
* **Order dependence** — **holds.** Maturation and locking are hysteretic (τ_mat = 32).
  Permuting an identical action multiset changes the outcome in **100%** of episodes,
  mean |ΔU| = 0.0703, 95% CI [0.0368, 0.1262] (gate A6, `_dev/gate_a6.py`).
* **Acting under partial observability** — **holds by construction**, though not separately
  gated: latent oxidative stress and chaperone capacity are never observed, and crowding
  drifts upward, so the window for cheap intervention closes.

A benchmark whose difficulty is asserted rather than
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

| task | disease | fragment | source | usable? |
|---|---|---|---|---|
| [`alzheimer-abeta42-v8`](alzheimer-abeta42-v8/) | Alzheimer's | Aβ 1–42, full peptide | APP amyloid-β region | **yes** — 3-weight policy scores 0.795 |
| [`als-ftd-tdp43-v8`](als-ftd-tdp43-v8/) | ALS / FTD | LCD aggregation hotspot 311–360 | UniProt Q13148 | **yes** — scores 0.581, hardest task |
| [`parkinson-alpha-synuclein-v8`](parkinson-alpha-synuclein-v8/) | Parkinson's | NAC domain 61–95 + flanks (55–105) | UniProt P37840 | no — saturated at 1.0 |
| [`alzheimer-tau-v8`](alzheimer-tau-v8/) | Alzheimer's / tauopathy | PHF6* (592–597) through PHF6 (623–628), 585–635 | UniProt P10636 | no — saturated at 1.0 |
| [`huntington-htt-polyq-v8`](huntington-htt-polyq-v8/) | Huntington's | exon-1 surrogate: N17 + polyQ36 + P11 | HTT exon-1 derived | **only after retraining the anchor** — shipped reference is undertrained |

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

| task | oracle reward | porting gate | 3-weight policy scores | A1 (full GNN vs edge-ablated GNN) |
|---|--:|---|--:|---|
| `alzheimer-abeta42-v8` | 1.0 | 8/8 PASS | 0.795 | PASS (+0.557, 95% CI [+0.126, +0.698]) |
| `huntington-htt-polyq-v8` | 1.0 | 8/8 PASS | **1.000** (ext. 1.562) | INCONCLUSIVE (+0.200, 95% CI [−0.120, +0.432]) |
| `alzheimer-tau-v8` | 1.0 | 8/8 PASS | **1.000** | not run |
| `parkinson-alpha-synuclein-v8` | 0.999999999953 | 7 PASS, 1 INCONCLUSIVE (P7) | **1.000** | not run |
| `als-ftd-tdp43-v8` | 1.0 | 6 PASS, 1 INCONCLUSIVE (P7), **1 FAIL (P8)** | 0.581 | not run |

The A1 column is kept for the record but **does not support a necessity claim**: its ablation
arm is the same graph controller with its edge embedding zeroed, so it measures "full GNN
versus damaged GNN". Both arms lose to the 3-weight policy in the column beside it. The gate
that would have tested necessity, A3, was specified in
[`ACCEPTANCE_CRITERIA.md`](agentic/specs/ACCEPTANCE_CRITERIA.md) and never run for v8.0; when
finally run it **failed**, and kept failing across four redesigns.

α-synuclein's shortfall from 1.0 is 5×10⁻¹¹ — float rounding inside the container against the
host-measured anchor. The audit tolerance is 1e-9.

Before a reference policy is trained for a protein, the protein must pass an eight-check
**porting gate** (P1–P8) proving the mechanism actually transferred: geometry responds, reward
calibration is non-degenerate, no-op is not optimal, pathology matures, targeted action beats
blind action, order matters, catastrophe is controllable.

Verdicts are **three-valued** — PASS / INCONCLUSIVE / FAIL. A confidence interval spanning
zero means insufficient power, not evidence of absence; conflating the two would have silently
passed off two null results as findings.

## What this measures

Not relational reasoning — see the correction at the top. What the tasks do measure, and what
survives the audit:

control of a **stochastic, partially observed, hysteretic process under a finite action
budget**, where the order of interventions changes the outcome in 100% of episodes and a heavy
damage tail punishes optimising the average. The utility is deliberately not a mean:
`U = 0.60·mean + 0.40·q₂₀ − 1.50·catastrophe_rate`.

That is a real control problem. It is simply not the problem the v8.0.0 README claimed, and
three of the five tasks are additionally saturated by a trivial policy, so only Aβ42 and
TDP-43 discriminate at all.

## Limitations

Stated rather than hidden. None of these were tuned away.

* **Two of five tasks are saturated by a 3-weight policy** — α-synuclein (1.045) and tau
  (1.086), both confirmed by retraining their anchors with six restarts. `alzheimer-abeta42-v8`
  (0.795) and `als-ftd-tdp43-v8` (0.581) discriminate as shipped. HTT discriminates *only if
  its anchor is retrained*.
* **Model selection on the public validation split is close to unreliable.** 32 selection
  episodes against a ±5% measurement resolution. On both confirmed tasks the restart with the
  best validation utility scored *worse* on the hidden split than the shipped anchor.
  `instruction.md` nevertheless tells agents to select on that split.
* **HTT's challenge reference is undertrained** — uniquely among the five. Six restarts reach
  extended 1.608, a 61% larger gain over no-op, and restart-to-restart validation utility spans
  −0.298 to −0.553. Retrain the anchor before using the task.
* **The reward's largest-weighted term barely moves.** `pathology_reduction` (weight 0.34)
  contributes 0.1%–24% of the reference's advantage and −101.6% on α-synuclein, because
  pathology is not controllable under v8.0 physics. The energy terms that do the work are
  *anti-correlated* with it (r = −0.73).
* **Porting-gate development seeds overlap the public validation split** (2000–2047 contains
  2000–2031), so P1–P8 design decisions were measured on the episodes agents select models on.
* **The sampler has three defects and the calibration compensates for them.** Fixed on
  `v8.1-physics-fixes`; the fix breaks the task, which is the finding. See the correction above.
* **Coarse-graining is below the scale of the mechanism.** At 5 residues per bead the
  nucleating hexapeptides occupy ~1 bead (KLVFF 1.0, VQIINK 1.2, VQIVYK 1.2), so steric-zipper
  interdigitation is not representable. `align` uses `|cos|`, so parallel and antiparallel
  registers are indistinguishable to the energy model.
* **The model is internally consistent and externally unvalidated.** It passes four
  pre-registered ordering tests (`_dev/test_biological_ordering.py`, 4/4) — but each is close
  to arithmetically forced by scales the model already contains. On the harder test, familial
  Aβ mutations at E22/D23 that are known to increase aggregation
  (`_dev/test_familial_mutations.py`), it reproduces **1 of 4**, with Italian E22K and Iowa
  D23N pointing the wrong way. Nothing is calibrated against experiment.

* **TDP-43 P8 = FAIL.** The best of six hand-coded probe policies left 16.7% of episodes
  catastrophic against a 10% threshold. The verdict was not re-opened. A learned controller,
  evaluated under a rule fixed **before** the run on 256 fresh seeds, reaches **8.59%
  (22/256), 95% CI [5.5%, 12.1%]** — point estimate below threshold, interval crossing it.
  Both results are reported together; neither is quotable alone. Full record:
  [`agentic/reports/audits/tdp43_final_report.md`](agentic/reports/audits/tdp43_final_report.md).
* **α-synuclein P7 = INCONCLUSIVE.** Order effect d = −0.014, 95% CI [−0.175, +0.147] at 96
  development seeds: limited power, not absence of hysteresis.
* **A1 does not demonstrate what it was cited for.** It compares a full graph controller
  against an edge-ablated one — not against a cheap controller — and was run on two tasks
  only. Both of its arms lose to a 3-weight policy.
* **A3 failed, four times.** The effective-dimensionality gate was unrun for v8.0. Run
  afterwards it failed on v8.0 and on three successive redesigns; an information-level test
  on 88 236 counterfactuals then put two rounds of graph aggregation at **+0.011 AUC** over
  per-edge features.
* **A2, A4, A5 and A8 remain unrun.** History necessity, RL-vs-black-box and
  precision-stability gates are specified in
  [`agentic/specs/ACCEPTANCE_CRITERIA.md`](agentic/specs/ACCEPTANCE_CRITERIA.md) and were not
  executed. A3 and A6 have since been implemented (`_dev/gate_a3.py`, `_dev/gate_a6.py`).
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

## Verifying the claims

Nothing here asks to be taken on trust. Each claim maps to a command and to a stored result.

| claim | command | stored result |
|---|---|---|
| tasks build, run and score; oracle = 1.0 | `python3 _dev/final_audit.py` | `agentic/reports/audits/final_audit.json` |
| simulator invariants hold | `python3 _dev/test_physics.py <task>` | 14 checks, printed |
| the artifact contract rejects malformed input | `python3 _dev/test_artifact_contract.py <task>` | 19 cases, printed |
| a ≤40-parameter controller reaches ≥95% of full-policy gain (A3 FAIL) | `python3 _dev/gate_a3.py --task alzheimer-abeta42-v8 --budget 12000 --runs 3` | `agentic/reports/validation/a3_alzheimer-abeta42-v8.json` |
| the A3 ablation really removes message passing and history | `python3 _dev/verify_a3_masks.py` | printed; reports effective vs nominal dimension |
| order changes the outcome in 100% of episodes (A6 PASS) | `python3 _dev/gate_a6.py --task alzheimer-abeta42-v8` | `agentic/reports/validation/a6_alzheimer-abeta42-v8.json` |
| local edge features suffice; aggregation adds +0.011 AUC | `python3 _dev/gate_aliasing.py --task alzheimer-abeta42-v8` | `agentic/reports/validation/aliasing_alzheimer-abeta42-v8.json` |
| the sampler defects, and that fixing them breaks the task | `git checkout v8.1-physics-fixes && python3 _dev/test_physics.py alzheimer-abeta42-v8` | branch `v8.1-physics-fixes`, commit message carries the measured table |
| four redesigns failed to remove the shortcut | branch `v9.1-relational-composition` | commit messages carry each redesign and its result |
| the chemistry reproduces four known aggregation orderings (near-tautological) | `python3 _dev/test_biological_ordering.py` | `agentic/reports/validation/biological_ordering.json` |
| it reproduces only 1 of 4 familial Aβ mutation effects | `python3 _dev/test_familial_mutations.py` | `agentic/reports/validation/familial_mutations.json` |
| anchors retrained at 6 restarts: HTT undertrained, α-syn and tau genuinely saturated | `python3 _dev/train_reference.py --task <task> --budget 12000 --restarts 6` | `agentic/reports/validation/reference_retraining.json` |

The 3-weight policy that saturates three tasks is not a special harness: it is a normal
`policy.json` satisfying the published artifact contract, built by setting three weights of
2541 and scored through the released verifier. `_dev/gate_a3.py` constructs it
(`hand_set()`), and the aliasing sample cache is regenerated automatically if absent.

Branches referenced above are pushed and public. Where a claim rests on a run that has not
been repeated, it says so.

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
