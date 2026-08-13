# NeuroFold v8 — alpha-synuclein NAC domain (Parkinson's disease)

Control a stochastic, sequence-conditioned coarse-grained conformational
surrogate of a **α-synuclein dimer** and suppress pathological β-register formation
without damaging the chains.

This is an AI-for-Science / RL-compatible control benchmark. It is **not**
atomistic molecular dynamics, not a folding free-energy predictor, and not a
disease simulator.

## What you must produce

A single numeric artifact at:

```
/logs/artifacts/policy.json
```

Publish it with:

```bash
python /app/neurofold_cli.py publish --policy /app/policy.json
```

The schema is defined and validated in `/app/policy_runtime.py`. Only the
numeric policy is transferred to the evaluator — no code you write is executed
during scoring.

## The fragment

UniProt **P37840** residues 55–105. The fragment spans the NAC domain (61–95) — the region necessary and sufficient for α-synuclein fibril formation — plus flanks that retain the amphipathic-helix boundary and the start of the acidic C-terminal tail. The flanks are the protective structure the damage model distinguishes from the aggregation core.

The chain is coarse-grained at **5 residues per bead**, giving 11 beads per
chain and two chains per episode.

## The environment

Two chains start with a partial antiparallel β-register already present: an
early-stage nucleation event. Left alone, that register matures, **locks**, and
drives pathology and irreversible damage upward.

**Action.** Each step you choose a contact pair and a strength:

```
(i, j, strength)     strength in [0, 1]
```

This applies a *contact-selective destabilisation* to the interaction of that
specific pair — a coarse proxy for a small molecule, a chaperone-like
interaction, or a local solvation/screening change. It is **not** a mechanical
force: you do not move coordinates. After the modulation the chain relaxes under
ordinary Metropolis dynamics, and whether anything actually moves is decided by
those dynamics.

**What makes this hard.**

* Which pair you target matters. Destabilising a pathological register contact
  is the intended therapy; destabilising healthy structure costs damage.
* When you act matters. Once a register matures it locks, resists modulation,
  and becomes expensive to dissolve. The same actions applied later do less.
* Part of the environment state is not observable, including an oxidative-stress
  term and a chaperone-capacity term that gates repair. Crowding drifts upward
  during an episode, so the window for cheap intervention closes.
* You have a finite action budget; stronger actions cost more of it.

**What is specific to this protein.** The NAC domain is strongly hydrophobic and largely uncharged, so the register that forms is held together by hydrophobic and shape complementarity rather than by electrostatics. There is very little glutamine, so the polar-zipper term contributes almost nothing here — the same mechanism that dominates the polyQ task is nearly silent in this one.

## Observation

`neurofold_cli.py inspect` prints the exact shapes. You receive per-bead
features, per-contact **edge** features (distance, sequence separation, contact
strength, charge and hydrophobic and aromatic compatibility, orientation
alignment, same-chain flag, contact age, pair-class interaction term, ladder
membership), a recent action history, and noisy readings of some environmental
conditions.

You do **not** receive the objective decomposition, any "protective region"
label, the pathology order parameter, or the lock state.

## What you may change

Everything under `/app`. You may write your own training code, edit or replace
the provided baselines, and use any method — black-box optimization, RL,
analytic reasoning, or hybrids. Only `policy.json` is scored.

Provided starting points:

* `python /app/neurofold_cli.py inspect` — task facts and tensor shapes
* `python /app/neurofold_cli.py init-policy --out /app/policy.json` — zero policy
* `python /app/neurofold_cli.py evaluate --policy /app/policy.json --split validation`
* `python /app/train_cmaes.py --budget 6000 --out /app/policy.json` — a working
  separable CMA-ES baseline over the policy vector

## Splits

`profile.json` lists 64 public training seeds and 32 public validation seeds.
Use validation for model selection. Hidden evaluation uses **unseen** episodes
and a shifted mechanism and condition set, so a policy that only fits the public
episodes will not transfer.

## Scoring

Continuous reward in `[0, 1]`. The no-op policy anchors `0.0`; a frozen
challenge reference controller anchors `1.0`. The reference is **not** claimed
to be optimal — `metrics.json` also reports an uncapped `extended_score`, so
policies stronger than the reference remain distinguishable.

Optimize robust trajectory quality: reduce pathological register formation and
keep safe occupancy high, while limiting irreversible damage, action expenditure
and barrier costs.

## Limits

* `policy.json` must be at most 768 KiB, contain only finite numbers, match the
  declared architecture exactly, and keep every weight within ±30.
* No network access.


## Validation status

**Validation status.** Seven of the eight porting gates pass. P7 (action order matters) is **INCONCLUSIVE**: at 96 development seeds the measured order effect is d = −0.014, 95% CI [−0.175, +0.147], which spans zero *and* exceeds the negligible band, so the data show limited power rather than absence of an effect. Hysteresis is present in the mechanism; it is not resolved at this sample size for this fragment.
