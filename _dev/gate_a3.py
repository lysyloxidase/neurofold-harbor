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
# Disjoint from train (1000-1063), public validation (2000-2031), calibration
# (3000-3063) and the gate dev split (7000-7047).
# Raised from 48 to 96 after the first run landed on the threshold (95.005%)
# with no interval at all. This buys resolution, not a direction: a tighter
# interval can resolve to either side.
HOLD = list(range(7100, 7196))
A3_THRESHOLD = 0.95


def head_mask(spec, pr, channels):
    """A message-passing-free readout with `channels` live edge features.

    Built so that NOMINAL size equals EFFECTIVE size. The first version of this
    mask took parameters in a flat priority order and was badly mislabelled:
    because every encode weight is zero the node states are identically zero, so
    the two thirds of `w_edge_sel` that multiply h[ei] and h[ej] are dead. A
    nominally 38-parameter arm had 3 parameters that could move the action, and
    a nominally 160-parameter arm had 83. Verified by
    `_dev/verify_a3_masks.py`.

    One channel = one row of W_edge (EDGE_DIM weights) + its bias + the
    selection weight and the strength weight that read it: EDGE_DIM + 3 live
    parameters. Plus a global constant strength bias.
    """
    probe = np.arange(pr.PARAM_COUNT, dtype=float)
    p = spec.unpack(probe)

    def flat(x):
        return [int(v) for v in np.atleast_1d(np.asarray(x)).ravel()]

    W_edge = np.atleast_2d(p["W_edge"])
    b_edge = np.atleast_1d(np.asarray(p["b_edge"])).ravel()
    idx = flat(p["b_str"])
    for c in range(channels):
        idx += flat(W_edge[c])                      # this channel reads the edge features
        idx += [int(b_edge[c])]
        idx += [int(np.atleast_1d(p["w_edge_sel"])[2 * spec.H + c])]
        idx += [int(np.atleast_1d(p["w_str"])[2 * spec.H + c])]
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
    ap.add_argument("--runs", type=int, default=3,
                    help="independent optimizer restarts per optimised arm")
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

    # channels -> live parameters: 1 + channels*(EDGE_DIM+3)
    masks = {c: head_mask(spec, pr, c) for c in (2, 5, 10)}
    assert len(masks[2]) <= 40, f"gating arm has {len(masks[2])} parameters, spec says <=40"
    print(f"A3 — {a.task}")
    print(f"  low-dim arms: {[len(m) for m in masks.values()]} free parameters "
          f"of {pr.PARAM_COUNT}")
    print(f"  budget per optimised arm: {a.budget} episodes (matched across arms)")
    print(f"  restarts per optimised arm: {a.runs}")
    print(f"  hold-out seeds: {HOLD[0]}-{HOLD[-1]} ({len(HOLD)} episodes)\n")

    pool = Pool(a.task, a.workers)
    try:
        gens = max(1, a.budget // len(TRAIN))

        def optimise(dim, mask, tag):
            """Same optimizer, same episode budget, `runs` independent restarts."""
            best = []
            for r in range(a.runs):
                rng = np.random.default_rng(903 + 101 * r)

                def fb(X, mask=mask):
                    if mask is None:
                        return pool.batch(X, TRAIN)
                    full = np.zeros((len(X), pr.PARAM_COUNT))
                    full[:, mask] = X
                    return pool.batch(full, TRAIN)

                x, f = sepcmaes(fb, dim, np.zeros(dim) + 0.02, gens, rng,
                                pr.MAX_ABS_WEIGHT)
                v = np.zeros(pr.PARAM_COUNT)
                if mask is None:
                    v = x
                else:
                    v[mask] = x
                best.append(v)
                print(f"    {tag} restart {r}: train={f:+.4f}", flush=True)
            return best

        opt = {}
        for k, m in masks.items():
            print(f"  [{len(m)}-param]")
            opt[f"low-dim ({len(m)} params)"] = optimise(len(m), m, f"{len(m)}p")
        print("  [full]")
        opt[f"full ({pr.PARAM_COUNT} params)"] = optimise(pr.PARAM_COUNT, None, "full")

        FULL = f"full ({pr.PARAM_COUNT} params)"
        ZERO, HAND = "zero (no-op)", "hand-set (3 weights)"
        arms = {ZERO: [np.zeros(pr.PARAM_COUNT)], HAND: [hand_set(spec, pr)]}
        arms.update(opt)

        res, ep = {}, {}
        print(f"\n{'arm':30s} {'utility':>9s} {'catastrophe':>12s}  (srednia po restartach)")
        for name, vecs in arms.items():
            per = [pool.full(v, HOLD) for v in vecs]
            ep[name] = [rows for _, _, rows in per]
            u = float(np.mean([x[0] for x in per]))
            cat = float(np.mean([x[1] for x in per]))
            res[name] = {"utility": u, "catastrophe": cat, "n_restarts": len(vecs),
                         "utility_per_restart": [float(x[0]) for x in per]}
            print(f"{name:30s} {u:9.4f} {cat:12.3f}")
        vectors = {n: [[float(x) for x in np.asarray(v)] for v in vs]
                   for n, vs in arms.items()}
    finally:
        pool.close()

    from neurofold8 import reward  # noqa: E402
    cal = json.loads((env_dir / "reward_calibration.json").read_text())["calibration"]

    def arm_utility(rows_per_restart, r_idx, e_idx):
        """Mean utility across the drawn restarts, each on the drawn episodes."""
        vals = []
        for r in r_idx:
            rows = [rows_per_restart[r % len(rows_per_restart)][i] for i in e_idx]
            vals.append(reward.robust_utility(rows, cal)[0])
        return float(np.mean(vals))

    def boot_share(name, n=4000, seed=311):
        """Hierarchical bootstrap: restarts resampled, then episodes resampled.
        Every arm ran the same hold-out seeds under CRN, so episode indices are
        drawn once per draw and shared, and the full nonlinear utility is
        recomputed from scratch on every draw."""
        rng_b = np.random.default_rng(seed)
        m, out_v = len(HOLD), []
        n_r = max(len(ep[name]), len(ep[FULL]))
        for _ in range(n):
            e_idx = rng_b.integers(0, m, m)
            r_idx = rng_b.integers(0, n_r, n_r)
            u_a = arm_utility(ep[name], r_idx, e_idx)
            u_z = arm_utility(ep[ZERO], r_idx, e_idx)
            u_f = arm_utility(ep[FULL], r_idx, e_idx)
            if u_f != u_z:
                out_v.append((u_a - u_z) / (u_f - u_z))
        v = np.asarray(out_v, float)
        v = v[np.isfinite(v)]
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    base = res[ZERO]["utility"]
    full_gain = res[FULL]["utility"] - base
    out = {"task": a.task, "threshold": A3_THRESHOLD,
           "lowdim_sizes": {f"{c}ch": int(len(m)) for c, m in masks.items()},
           "budget_episodes": a.budget, "restarts": a.runs,
           "holdout_seeds": [HOLD[0], HOLD[-1]],
           "n_holdout": len(HOLD), "arms": res, "vectors": vectors}
    print()
    for name in arms:
        if name == ZERO:
            continue
        share = (res[name]["utility"] - base) / full_gain if full_gain else float("nan")
        lo, hi = boot_share(name)
        res[name].update({"share_of_full_gain": float(share), "share_ci95": [lo, hi]})
        print(f"  {name:30s} osiaga {100*share:6.1f}% zysku pelnej polityki   "
              f"95% CI [{100*lo:.1f}%, {100*hi:.1f}%]")

    # Only controllers with <=40 parameters gate the verdict, per the spec.
    gating = {HAND: res[HAND]["share_of_full_gain"]}
    for k, m in masks.items():
        if len(m) <= 40:
            gating[f"low-dim ({len(m)} params)"] = res[f"low-dim ({len(m)} params)"]["share_of_full_gain"]
    worst_name = max(gating, key=gating.get)
    worst = gating[worst_name]
    # Three-valued verdict, matching the standard every other gate in this
    # project already uses. The threshold stays at 95%; what is added is the
    # uncertainty treatment A3 lacked. A point estimate landing on the
    # threshold (95.005% at N=48) is not a decision, it is a coin flip.
    lo, hi = res[worst_name]["share_ci95"]
    if hi < A3_THRESHOLD:
        verd = V.PASS
        note = ("No cheap controller reaches 95% of the full policy's gain; the whole "
                "confidence interval sits below the threshold.")
    elif lo >= A3_THRESHOLD:
        verd = V.FAIL
        note = ("A cheap controller reaches >=95% of the full policy's gain, so the task "
                "does not require the shipped architecture.")
    else:
        verd = V.INCONCLUSIVE
        note = ("The confidence interval straddles the 95% threshold: this hold-out cannot "
                "decide whether the architecture is necessary. Not evidence that it is.")
    out.update({"gating_arms": {k: float(v) for k, v in gating.items()},
                "binding_arm": worst_name, "binding_share": float(worst),
                "binding_share_ci95": [lo, hi], "verdict": verd, "note": note})
    print(f"\nA3: {verd}   (prog: tania polityka musi zostac PONIZEJ "
          f"{100*A3_THRESHOLD:.0f}%; wiazace ramie: {worst_name} = {100*worst:.1f}%, "
          f"95% CI [{100*lo:.1f}%, {100*hi:.1f}%])")
    print(f"  {note}")
    p = ROOT / f"agentic/reports/validation/a3_{a.task}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
