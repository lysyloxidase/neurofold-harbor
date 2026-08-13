"""Scaffold a new v8 task from a fragment spec and normalise its extensive
parameters using the rules established on HTT and re-checked on alpha-synuclein.

Copies the frozen code unchanged; only the profile is task-specific.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = "alzheimer-abeta42-v8"
REF_CORE_MASS, REF_PROT_MASS = 4.0, 50.0

FRAGMENTS = {
    "als-ftd-tdp43-v8": {
        "disease": "ALS / FTD", "protein": "TDP-43 low-complexity domain fragment",
        "accession": "Q13148", "range": [311, 360],
        "full": ("MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDN"
                 "KRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVM"
                 "SQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSL"
                 "CGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMA"
                 "AAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSM"
                 "DSKSSGWGM"),
        "rationale": "The conserved aggregation hotspot of the C-terminal low-complexity domain. "
                     "Carries the aromatic/sticker residues that drive LCD self-association. The "
                     "folded RRM domains are omitted: they are not part of the aggregation core.",
        "core_frac": [0.20, 0.80],
    },
    "alzheimer-tau-v8": {
        "disease": "Alzheimer's disease / tauopathy", "protein": "tau repeat-domain fragment (PHF6*-PHF6)",
        "accession": "P10636", "range": [585, 635],
        "rationale": "Spans both hexapeptide motifs that nucleate paired helical filaments: "
                     "PHF6* (VQIINK, 592-597) and PHF6 (VQIVYK, 623-628), with the intervening "
                     "repeat-domain segment retained so both motifs and their spacing are present. "
                     "The projection domain is omitted: it does not enter the cross-beta core. "
                     "The fragment is extended seven residues either side of the 592-628 motif span "
                     "so that flanking repeat-domain sequence remains outside the aggregation core "
                     "and can act as the protective structure the damage model distinguishes.",
        "core_frac": [0.137, 0.863],
    },
}


def build(task, full_seq):
    spec = FRAGMENTS[task]
    lo, hi = spec["range"]
    frag = full_seq[lo - 1:hi]
    T = ROOT / task
    for sub in ("environment", "tests", "solution"):
        (T / sub).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(T / "environment/neurofold8", ignore_errors=True)
    shutil.copytree(ROOT / SRC / "environment/neurofold8", T / "environment/neurofold8")
    for f in ("policy_runtime.py", "neurofold_cli.py", "train_cmaes.py", "Dockerfile"):
        shutil.copy2(ROOT / SRC / "environment" / f, T / "environment" / f)
    for f in ("Dockerfile", "test.sh", "verifier.py"):
        shutil.copy2(ROOT / SRC / "tests" / f, T / "tests" / f)
    shutil.copy2(ROOT / SRC / "solution/solve.sh", T / "solution/solve.sh")

    d = json.loads((ROOT / SRC / "environment/profile.json").read_text())
    n = len(frag)
    c0, c1 = int(spec["core_frac"][0] * n), int(spec["core_frac"][1] * n)
    d.update({"slug": task, "disease": spec["disease"], "protein": spec["protein"],
              "accession": spec["accession"], "sequence": frag,
              "sequence_source": f"UniProt {spec['accession']} residues {lo}-{hi} (1-based). "
                                 + spec["rationale"],
              "fragment": {"uniprot_range": [lo, hi], "core": [c0, c1],
                           "note": "0-based half-open slices into `sequence`"},
              "regions": [{"name": "core", "start": c0, "end": c1, "weight": 1.0},
                          {"name": "turn", "start": c1, "end": n, "weight": 0.7},
                          {"name": "nterm", "start": 0, "end": max(1, c0), "weight": 0.4}],
              "version": "8.0-public"})
    for k in ("_pathology_normalisation", "_extensive_parameter_normalisation",
              "_polar_zipper", "_action_space", "_status", "_note"):
        d.pop(k, None)
    (T / "environment/profile.json").write_text(json.dumps(d, indent=2) + "\n")
    return frag


def normalise(task, workers=5):
    """Apply the established extensive-parameter rules from measured baselines."""
    env_dir = ROOT / task / "environment"
    for m in [m for m in sys.modules if m.startswith("neurofold8")]:
        del sys.modules[m]
    sys.path.insert(0, str(env_dir))
    from neurofold8.env import NeuroFoldV8Env
    prof = json.loads((env_dir / "profile.json").read_text())
    e = NeuroFoldV8Env(prof, seed=11)
    core_mass = float(np.outer(e.regions["core"], e.regions["core"]).sum())
    prot_mass = float(e._protective_pair.sum()) / 2
    eb = []
    for s in range(2000, 2016):
        x = NeuroFoldV8Env(prof, seed=s)
        while x.steps < x.max_steps and x.budget > 0:
            _, _, done, _ = x.step((0, 0, 0.0))
            a, b, _ = x.energy.bonded(x.x)
            eb.append(a + b)
            if done:
                break
    p90 = float(np.percentile(eb, 90))
    ref = json.loads((ROOT / SRC / "environment/profile.json").read_text())["physics"]
    ps = REF_CORE_MASS / max(core_mass, 1e-9)
    if prot_mass <= 0:
        raise SystemExit(
            f"{task}: protective-pair mass is zero - the fragment has no non-core region, so the "
            "oracle-vs-blind asymmetry (breaking healthy structure costs, dissolving the "
            "pathological register does not) cannot exist. Redefine the fragment with flanks.")
    pr_ = prot_mass / REF_PROT_MASS
    d = json.loads((env_dir / "profile.json").read_text())
    for k in ("path_inter", "path_intra", "path_nucleus", "path_locked"):
        d["physics"][k] = round(ref[k] * ps, 6)
    d["physics"]["safe_pathology"] = round(ref["safe_pathology"] * ps, 6)
    d["physics"]["eta_disrupt"] = round(ref["eta_disrupt"] / pr_, 6)
    d["physics"]["elastic_limit"] = round(p90, 2)
    d["_extensive_parameter_normalisation"] = {
        "rules": "Established on HTT, re-checked on alpha-synuclein: pathology by core-pair mass, "
                 "eta_disrupt by protective-pair mass, elastic_limit = p90 of the no-op bonded "
                 "baseline. Every factor is a measured ratio, not a fitted value.",
        "measured": {"beads": int(e.n), "core_pair_mass": core_mass,
                     "protective_pair_mass": prot_mass, "bonded_p90": p90},
        "applied": {"path_scale": ps, "eta_disrupt_divisor": pr_, "elastic_limit": round(p90, 2)}}
    (env_dir / "profile.json").write_text(json.dumps(d, indent=2) + "\n")
    sys.path.remove(str(env_dir))
    return {"beads": int(e.n), "core_mass": core_mass, "prot_mass": prot_mass, "p90": p90}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(FRAGMENTS))
    ap.add_argument("--tau-source", default=str(
        ROOT.parent / "harbor_neurofold_tasks_v7/alzheimer-tau/environment/profile.json"))
    a = ap.parse_args()
    spec = FRAGMENTS[a.task]
    full = spec.get("full")
    if full is None:                       # tau: read the canonical sequence from v7
        full = json.loads(Path(a.tau_source).read_text())["sequence"]
    frag = build(a.task, full)
    info = normalise(a.task)
    lo, hi = spec["range"]
    print(f"{a.task}: residues {lo}-{hi} = {len(frag)} aa -> {info['beads']} beads (2 chains)")
    print(f"  {frag}")
    print(f"  core_mass={info['core_mass']:.1f} prot_mass={info['prot_mass']:.0f} "
          f"bonded_p90={info['p90']:.2f}")


if __name__ == "__main__":
    main()
