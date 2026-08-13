"""Render instruction.md and task.toml for the ported v8 tasks.

The Aβ42 golden task keeps its hand-written pair; the four ported tasks share
its structure and differ only in the protein-specific sections. Nothing here
touches the environment, the reward, the seeds or the frozen anchors — it only
writes the two Harbor top-level files that `_dev/new_task.py` never emitted.

    python3 _dev/write_task_docs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "parkinson-alpha-synuclein-v8": dict(
        title="alpha-synuclein NAC domain (Parkinson's disease)",
        symbol="α-synuclein",
        disease="Parkinson's disease",
        protein="alpha-synuclein NAC fragment (UniProt P37840, residues 55-105)",
        beads=11,
        fragment=(
            "UniProt **P37840** residues 55–105. The fragment spans the NAC domain "
            "(61–95) — the region necessary and sufficient for α-synuclein fibril "
            "formation — plus flanks that retain the amphipathic-helix boundary and "
            "the start of the acidic C-terminal tail. The flanks are the protective "
            "structure the damage model distinguishes from the aggregation core."),
        biology=(
            "The NAC domain is strongly hydrophobic and largely uncharged, so the "
            "register that forms is held together by hydrophobic and shape "
            "complementarity rather than by electrostatics. There is very little "
            "glutamine, so the polar-zipper term contributes almost nothing here — "
            "the same mechanism that dominates the polyQ task is nearly silent in "
            "this one."),
        keywords=["parkinsons", "alpha-synuclein"],
        gate=("Porting gate: 7 PASS, 1 INCONCLUSIVE (P7 action-order, d=-0.014, "
              "95% CI [-0.175, +0.147] at 96 dev seeds — limited power, not evidence "
              "of absence). No A1 relational-necessity experiment was run for this task."),
        caveat=(
            "**Validation status.** Seven of the eight porting gates pass. P7 "
            "(action order matters) is **INCONCLUSIVE**: at 96 development seeds the "
            "measured order effect is d = −0.014, 95% CI [−0.175, +0.147], which "
            "spans zero *and* exceeds the negligible band, so the data show limited "
            "power rather than absence of an effect. Hysteresis is present in the "
            "mechanism; it is not resolved at this sample size for this fragment."),
    ),
    "alzheimer-tau-v8": dict(
        title="tau repeat domain, PHF6*/PHF6 (Alzheimer's disease and tauopathies)",
        symbol="tau",
        disease="Alzheimer's disease / tauopathy",
        protein="tau repeat-domain fragment PHF6*-PHF6 (UniProt P10636, residues 585-635)",
        beads=11,
        fragment=(
            "UniProt **P10636** residues 585–635. The fragment spans both "
            "hexapeptide motifs that nucleate paired helical filaments — PHF6* "
            "(VQIINK, 592–597) and PHF6 (VQIVYK, 623–628) — with the intervening "
            "repeat-domain segment kept so that both motifs *and their spacing* are "
            "present. The projection domain is omitted: it does not enter the "
            "cross-β core. Seven residues of flanking repeat-domain sequence on "
            "either side stay outside the aggregation core and act as the protective "
            "structure."),
        biology=(
            "Two separate nucleating motifs sit in one chain, so the pathological "
            "register can seed in more than one place and the useful intervention "
            "target is not fixed for the whole episode. Both motifs carry glutamine, "
            "so the polar-zipper term is active but far from saturated."),
        keywords=["alzheimers", "tau", "tauopathy"],
        gate="Porting gate: 8/8 PASS on first attempt, no per-task adjustment. "
             "No A1 relational-necessity experiment was run for this task.",
        caveat=None,
    ),
    "als-ftd-tdp43-v8": dict(
        title="TDP-43 low-complexity domain (ALS and frontotemporal dementia)",
        symbol="TDP-43",
        disease="ALS / FTD",
        protein="TDP-43 low-complexity domain fragment (UniProt Q13148, residues 311-360)",
        beads=10,
        fragment=(
            "UniProt **Q13148** residues 311–360, the conserved aggregation hotspot "
            "of the C-terminal low-complexity domain. It carries the aromatic "
            "'sticker' residues that drive LCD self-association. The folded RRM "
            "domains are omitted: they are not part of the aggregation core."),
        biology=(
            "A low-complexity domain behaves differently from a compact amyloid "
            "core. Contacts are numerous, weak and aromatic-driven rather than few "
            "and strong, so a single contact-selective action removes a smaller "
            "share of the stabilising interaction than it does in Aβ42. This is the "
            "hardest task in the set and the only one whose heuristic controllability "
            "gate fails — see below."),
        keywords=["als", "ftd", "tdp-43"],
        gate=("Porting gate: 6 PASS, 1 INCONCLUSIVE (P7), 1 FAIL (P8 catastrophe "
              "controllable under hand-coded probes). A learned reference controller "
              "trained with the standard pipeline reaches 8.59% catastrophe on 256 "
              "fresh seeds. See the validation note in instruction.md — both results "
              "are reported together."),
        caveat=(
            "**Validation status — read this before using the task.**\n\n"
            "This is the one task in the v8 set that does *not* pass the porting "
            "gate cleanly, and it ships with that stated rather than hidden.\n\n"
            "* **P8 (catastrophe controllable), heuristic probe: FAIL.** Six "
            "hand-coded policies were tested on 48 development seeds. The best of "
            "them — continuous oracle targeting at full strength — left 16.7% of "
            "episodes catastrophic, above the 10% threshold. That verdict stands and "
            "was never re-opened.\n"
            "* **Learned-control diagnostic: PASS.** Under a decision rule fixed and "
            "written down *before* the run, a policy trained with the identical "
            "pipeline and budget used for the other four tasks was evaluated on 256 "
            "fresh, disjoint seeds. Catastrophe rate **8.59% (22/256)**, 95% CI "
            "**[5.5%, 12.1%]**.\n\n"
            "The point estimate is under the 10% threshold; the confidence interval "
            "crosses it. So the honest statement is that the heavy damage tail is "
            "reducible by learned control but is **not demonstrated to be below "
            "threshold with confidence** at this sample size. The two results belong "
            "together: the heuristic probe used by P8 is not an adequate proxy for "
            "controllability on this fragment, and the task is harder than the other "
            "four. Nothing was retuned after either result was seen."),
    ),
    "huntington-htt-polyq-v8": dict(
        title="huntingtin exon-1 polyQ (Huntington's disease)",
        symbol="HTT polyQ",
        disease="Huntington's disease",
        protein="HTT exon-1 fragment: N17 + polyQ36 + polyP11",
        beads=13,
        fragment=(
            "An HTT exon-1 derived surrogate: the N17 amphipathic segment "
            "(residues 1–17), a disease-length **polyQ36** tract, and the "
            "polyproline P11 stretch that follows it. It is not the full exon 1 of "
            "any specific allele. N17 and P11 sit outside the aggregation core and "
            "act as the protective structure."),
        biology=(
            "A 36-glutamine tract is above the pathogenic repeat-length threshold. "
            "Its stability comes from **polar-zipper** hydrogen bonding between "
            "glutamine side chains — saturable and directional — which is the "
            "dominant term here and nearly absent in α-synuclein. Porting to this "
            "protein is what forced the mechanism into the shared chemistry: without "
            "it, the polyQ register was not controllable at all. The core is also "
            "large, which is why the extensive parameters (pathology scale, "
            "disruption penalty, elastic limit) are normalised by measured mass "
            "ratios rather than inherited from Aβ42."),
        keywords=["huntingtons", "polyglutamine", "htt"],
        gate=("Porting gate: 8/8 PASS, after adding the polar-zipper mechanism to "
              "the shared chemistry. A1 relational necessity: INCONCLUSIVE "
              "(+0.200, 95% CI [-0.120, +0.432])."),
        caveat=(
            "**Validation status.** All eight porting gates pass. The A1 "
            "relational-necessity experiment — does a full graph controller beat a "
            "local-only controller under common random numbers — is **INCONCLUSIVE** "
            "for this task: +0.200 utility, 95% CI [−0.120, +0.432]. On Aβ42 the "
            "same experiment is a clear PASS (+0.557, 95% CI [+0.126, +0.698]). "
            "Relational structure is therefore demonstrated to be necessary on Aβ42 "
            "and not demonstrated either way here."),
    ),
}

TOML = '''schema_version = "1.4"
reward = "continuous"

[task]
name = "neurofold/{slug}"
version = "8.0.0"
description = "{description}"
authors = [{{ name = "NeuroFold-Harbor" }}]
keywords = [
  "reinforcement-learning",
  "graph-neural-network",
  "protein",
  "neurodegeneration",
  "ai-for-science",
  "harbor",
  "sequential-decision-making",
{extra_keywords}]

[metadata]
difficulty = "unbenchmarked"
category = "ai-for-science"
reward_type = "continuous"
scientific_scope = "Physics-informed coarse-grained stochastic conformational-control benchmark at 5 residues per bead. Not atomistic MD, not a folding free-energy predictor, not a disease simulator."
disease = "{disease}"
protein = "{protein}"
action_space = "(i, j, strength): contact-selective energy modulation of pair (i,j) followed by ordinary Metropolis relaxation; not a mechanical force"
version_notes = "{version_notes}"

[agent]
timeout_sec = 3600.0

[verifier]
timeout_sec = 1800.0
environment_mode = "separate"

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 4096
network_mode = "no-network"

[verifier.environment]
build_timeout_sec = 900.0
cpus = 4
memory_mb = 4096
storage_mb = 4096
network_mode = "no-network"
'''

INSTRUCTION = '''# NeuroFold v8 — {title}

Control a stochastic, sequence-conditioned coarse-grained conformational
surrogate of a **{symbol} dimer** and suppress pathological β-register formation
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

{fragment}

The chain is coarse-grained at **5 residues per bead**, giving {beads} beads per
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

**What is specific to this protein.** {biology}

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
{caveat}'''


def main():
    for slug, m in TASKS.items():
        T = ROOT / slug
        prof = json.loads((T / "environment/profile.json").read_text())
        assert prof["slug"] == slug, slug

        desc = (f"Edge-aware, history-dependent conformational-control benchmark for a "
                f"{m['symbol']} dimer ({m['protein'].split('(')[0].strip()}): suppress "
                f"pathological beta-register maturation via contact-selective "
                f"destabilisation.")
        notes = ("v8: two-chain dimer; non-separable pair chemistry so partner identity "
                 "matters; directional saturable beta term and saturable polar-zipper "
                 "term; maturation/locking hysteresis so action ORDER matters; "
                 "irreversible damage from physical events only; partial observability; "
                 "action budget. Extensive parameters normalised by measured mass ratios, "
                 "not fitted. " + m["gate"])
        extra = "".join(f'  "{k}",\n' for k in m["keywords"])
        (T / "task.toml").write_text(TOML.format(
            slug=slug, description=desc, disease=prof["disease"], protein=m["protein"],
            version_notes=notes, extra_keywords=extra))

        caveat = ("\n\n## Validation status\n\n" + m["caveat"] + "\n") if m["caveat"] else "\n"
        (T / "instruction.md").write_text(INSTRUCTION.format(
            title=m["title"], symbol=m["symbol"], fragment=m["fragment"],
            beads=m["beads"], biology=m["biology"], caveat=caveat))
        print(f"wrote {slug}/task.toml and {slug}/instruction.md")


if __name__ == "__main__":
    main()
