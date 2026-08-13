"""Gate A3 — effective dimensionality.

Specified in agentic/specs/ACCEPTANCE_CRITERIA.md and never run for v8.0:

    a <=40-parameter controller must NOT reach >=95% of full-policy gain

If a handful of parameters buys the whole gain, the task does not require the
architecture it ships with, and any claim that relational or history-dependent
structure is necessary is unsupported by the scoring.

Three arms, all evaluated on the same held-out seeds:

  hand-set        3 weights, written by hand, no optimisation at all. Reads the
                  observable `ladder` edge feature, selects on it, full strength.
                  This is the policy that saturated 3 of 5 v8.0 tasks.
  low-dim         <=40 free parameters (the readout head), optimised with the
                  same episode budget as the full arm.
  full            all 2541 parameters, same budget.

    python3 _dev/gate_a3.py --task alzheimer-abeta42-v8 --budget 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_dev"))
import verdict as V  # noqa: E402
from train_reference import Pool, sepcmaes  # noqa: E402

TRAIN = list(range(1000, 1016))
HOLD = list(range(7100, 7148))          # disjoint from train, validation and gate dev
A3_THRESHOLD = 0.95


def head_mask(spec, pr):
    """Indices of the readout head: <=40 free parameters.

    The head is the edge-selection readout w_edge_sel (36) plus a constant
    strength b_str and one W_edge entry that lets the head see an edge feature
    at all: 38 parameters. Everything else stays zero, so the controller has no
    message passing, no history and no state-dependent strength.
    """
    probe = np.arange(pr.PARAM_COUNT, dtype=float)
    p = spec.unpack(probe)
    idx = [int(x) for x in np.atleast_1d(p["w_edge_sel"]).ravel()]
    b = p["b_str"]
    idx.append(int(b) if np.ndim(b) == 0 else int(np.ravel(b)[0]))
    idx.append(int(np.atleast_2d(p["W_edge"])[0, pr.EDGE_DIM - 1]))
    return np.array(sorted(set(idx)), dtype=int)


def hand_set(spec, pr):
    """The 3-weight policy. No optimisation, written by hand in a minute."""
    vec = np.zeros(pr.PARAM_COUNT)
    p = spec.unpack(vec)
    p["W_edge"][0, pr.EDGE_DIM - 1] = 3.0        # read the observable 'ladder' feature
    p["w_edge_sel"][2 * spec.H + 0] = 1.0        # select the edge carrying it
    probe = np.arange(pr.PARAM_COUNT, dtype=float)
    b_str_at = spec.unpack(probe)["b_str"]
    vec[int(b_str_at)] = 3.0                     # constant full strength
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="alzheimer-abeta42-v8")
    ap.add_argument("--budget", type=int, default=4000, help="episodes per optimised arm")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    np.seterr(all="ignore")

    env_dir = ROOT / a.task / "environment"
    sys.path.insert(0, str(env_dir))
    import policy_runtime as pr
    from neurofold8.policy import GraphPolicySpec
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)

    mask = head_mask(spec, pr)
    assert len(mask) <= 40, f"low-dim arm has {len(mask)} parameters, spec says <=40"
    print(f"A3 — {a.task}")
    print(f"  low-dim arm: {len(mask)} free parameters of {pr.PARAM_COUNT}")
    print(f"  budget per optimised arm: {a.budget} episodes")
    print(f"  hold-out seeds: {HOLD[0]}-{HOLD[-1]}\n")

    pool = Pool(a.task, a.workers)
    rng = np.random.default_rng(903)
    try:
        gens = max(1, a.budget // len(TRAIN))

        def fb_full(X):
            return pool.batch(X, TRAIN)

        def fb_low(X):
            full = np.zeros((len(X), pr.PARAM_COUNT))
            full[:, mask] = X
            return pool.batch(full, TRAIN)

        print("  [low-dim]")
        x_low, _ = sepcmaes(fb_low, len(mask), np.zeros(len(mask)) + 0.02,
                            gens, rng, pr.MAX_ABS_WEIGHT)
        v_low = np.zeros(pr.PARAM_COUNT)
        v_low[mask] = x_low

        print("  [full]")
        x_full, _ = sepcmaes(fb_full, pr.PARAM_COUNT,
                             np.zeros(pr.PARAM_COUNT) + 0.02, gens, rng,
                             pr.MAX_ABS_WEIGHT)

        arms = {"zero (no-op)": np.zeros(pr.PARAM_COUNT),
                "hand-set (3 weights)": hand_set(spec, pr),
                f"low-dim ({len(mask)} params)": v_low,
                f"full ({pr.PARAM_COUNT} params)": x_full}
        res = {}
        print(f"\n{'arm':30s} {'utility':>9s} {'catastrophe':>12s} {'gain vs no-op':>14s}")
        for name, vec in arms.items():
            u, cat, _ = pool.full(vec, HOLD)
            res[name] = {"utility": float(u), "catastrophe": float(cat)}
            print(f"{name:30s} {u:9.4f} {cat:12.3f}", end="")
            print(f"{'':>14s}" if "zero" in name else "")
    finally:
        pool.close()

    base = res["zero (no-op)"]["utility"]
    full_gain = res[f"full ({pr.PARAM_COUNT} params)"]["utility"] - base
    out = {"task": a.task, "threshold": A3_THRESHOLD, "n_lowdim_params": int(len(mask)),
           "budget_episodes": a.budget, "holdout_seeds": [HOLD[0], HOLD[-1]], "arms": res}
    print()
    for name in arms:
        if "zero" in name:
            continue
        share = (res[name]["utility"] - base) / full_gain if full_gain else float("nan")
        res[name]["share_of_full_gain"] = float(share)
        print(f"  {name:30s} osiaga {100*share:6.1f}% zysku pelnej polityki")

    low_share = res[f"low-dim ({len(mask)} params)"]["share_of_full_gain"]
    hand_share = res["hand-set (3 weights)"]["share_of_full_gain"]
    worst = max(low_share, hand_share)
    verd = V.FAIL if worst >= A3_THRESHOLD else V.PASS
    out.update({"low_dim_share": low_share, "hand_set_share": hand_share,
                "verdict": verd,
                "note": ("A cheap controller reaches >=95% of the full policy's gain, so the "
                         "task does not require the shipped architecture."
                         if verd == V.FAIL else
                         "No cheap controller reaches 95% of the full policy's gain.")})
    print(f"\nA3: {verd}   (prog: tania polityka musi zostac PONIZEJ "
          f"{100*A3_THRESHOLD:.0f}%, najwyzsza tania = {100*worst:.1f}%)")
    p = ROOT / f"agentic/reports/validation/a3_{a.task}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
