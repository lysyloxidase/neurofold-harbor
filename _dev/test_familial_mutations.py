"""Harder external-validity test: familial Abeta mutations.

The ordering test in `_dev/test_biological_ordering.py` passes 4/4, but each of
its predictions is close to arithmetically forced by scales the model already
contains. This one is not. Four familial mutations clustered at the E22/D23 salt
bridge are known to INCREASE aggregation, and they do so for structural reasons
that simple hydropathy and beta-propensity scales do not encode -- in three of
the four cases the naive reading of those scales points the other way.

Prediction fixed before the run, from the literature, for all four:
pathology(mutant) > pathology(wild type).

    Arctic   E22G   enhanced protofibril formation
    Dutch    E22Q   enhanced aggregation
    Italian  E22K   enhanced aggregation
    Iowa     D23N   enhanced fibrillisation

Measured result: 1 of 4. Only Dutch E22Q passes, and it is the one case where the
model's own scales happen to point the right way -- E -> Q raises beta propensity
0.34 -> 0.63, removes a charge, and feeds the polar-zipper term, which keys on
glutamine. Arctic is null; Italian and Iowa have point estimates in the wrong
direction.

This is the honest state of the model's external validity.

    python3 _dev/test_familial_mutations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(8100, 8164))
TASK = "alzheimer-abeta42-v8"

MUTATIONS = {
    "Arctic  E22G": (22, "G"),
    "Dutch   E22Q": (22, "Q"),
    "Italian E22K": (22, "K"),
    "Iowa    D23N": (23, "N"),
}


def main():
    np.seterr(all="ignore")
    env_dir = ROOT / TASK / "environment"
    sys.path.insert(0, str(env_dir))
    from neurofold8.env import NeuroFoldV8Env

    base = json.loads((env_dir / "profile.json").read_text())
    wt = base["sequence"]
    assert wt[21] == "E" and wt[22] == "D", "Abeta numbering has shifted"

    def run(seq):
        prof = json.loads(json.dumps(base))
        prof["sequence"] = seq
        out = []
        for s in SEEDS:
            e = NeuroFoldV8Env(prof, seed=int(s))
            e.observe()
            while e.steps < e.max_steps and e.budget > 0:
                _, _, d, _ = e.step((0, 0.0, 0.0))
                if d:
                    break
            out.append(e.summary()["final_pathology"])
        return np.asarray(out)

    w = run(wt)
    rng = np.random.default_rng(5)
    print(f"familial Abeta mutations — prediction: every mutant aggregates MORE")
    print(f"{len(SEEDS)} seeds, no-op policy, paired bootstrap\n")
    print(f"{'variant':18s} {'pathology':>10s} {'vs WT':>9s} {'95% CI':>20s}  outcome")
    print(f"{'wild type':18s} {w.mean():10.3f}")
    right = 0
    out = {"wild_type_pathology": float(w.mean()), "n_seeds": len(SEEDS), "cases": {}}
    for name, (pos, aa) in MUTATIONS.items():
        seq = wt[:pos - 1] + aa + wt[pos:]
        v = run(seq)
        d = v - w
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        if lo > 0:
            verdict, right = "MATCHES", right + 1
        elif hi < 0:
            verdict = "CONTRADICTS"
        else:
            verdict = "null (point est. " + ("right" if d.mean() > 0 else "WRONG") + " way)"
        out["cases"][name.strip()] = {"pathology": float(v.mean()), "delta": float(d.mean()),
                                      "ci95": [float(lo), float(hi)], "outcome": verdict}
        print(f"{name:18s} {v.mean():10.3f} {d.mean():+9.3f} "
              f"[{lo:+7.3f},{hi:+7.3f}]  {verdict}")
    out["matched"] = right
    out["note"] = ("The model reproduces 1 of 4 familial mutation effects. The one it gets "
                   "is the one its own scales force. This is the limit of its external "
                   "validity and it is reported rather than omitted.")
    print(f"\nWYNIK: {right}/4 zgodnych z literatura")
    p = ROOT / "agentic/reports/validation/familial_mutations.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
