"""TDP-43 final precision audit — single evaluation at N=256, no peeking.

Constraints honoured:
  * P8_original stays FAIL. This does not re-open it.
  * Nothing is changed: reward, thresholds, chemistry, policy, simulator.
  * One evaluation at 256 seeds. No intermediate inspection at 128.
  * No N increase and no retuning after this run.

Seeds 5000-5255: disjoint from train (1000-1063), validation (2000-2031),
gate dev (2000-2047), calibration (3000-3063), the earlier 64-seed diagnostic
(4000-4063) and the final-test range (900000+).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = "als-ftd-tdp43-v8"
N = 256
FRESH = list(range(5000, 5000 + N))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from porting_gate import Pool  # noqa: E402


def wilson(k, n, z=1.96):
    """Wilson score interval — correct for proportions near a boundary."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_rate(flags, n=20000, seed=606):
    rng = np.random.default_rng(seed)
    f = np.asarray(flags, float)
    bs = np.array([f[rng.integers(0, len(f), len(f))].mean() for _ in range(n)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    np.seterr(all="ignore")
    env_dir = ROOT / TASK / "environment"
    sys.path.insert(0, str(env_dir))
    import policy_runtime as pr
    from neurofold8 import reward
    from neurofold8.env import NeuroFoldV8Env
    from neurofold8.policy import GraphPolicy, GraphPolicySpec

    prof = json.loads((env_dir / "profile.json").read_text())
    cal = json.loads((env_dir / "reward_calibration.json").read_text())["calibration"]
    vec = pr.load_policy(ROOT / TASK / "solution/challenge_reference.json")
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)
    pol = GraphPolicy(spec)

    rows_learned = []
    for s in FRESH:
        env = NeuroFoldV8Env(prof, seed=int(s))
        obs = env.observe()
        h = np.zeros(spec.HH)
        while env.steps < env.max_steps and env.budget > 0:
            act, h = pol.act(vec, obs, h)
            obs, _, done, _ = env.step(act)
            if done:
                break
        rows_learned.append(env.summary())

    pool = Pool(TASK, prof, 5)
    try:
        heur = {k: pool.run(k, FRESH) for k in ("noop", "o10")}
    finally:
        pool.close()

    arms = {"no-op": heur["noop"], "heuristic oracle s=1.0": heur["o10"],
            "learned reference": rows_learned}
    out = {"task": TASK, "n_episodes": N, "seeds": [FRESH[0], FRESH[-1]],
           "P8_original": "FAIL", "threshold": 0.10,
           "note": "Final precision audit. Single evaluation at N=256, no intermediate "
                   "inspection. Does not re-open P8.", "arms": {}}

    print(f"TDP-43 precision audit — N={N} fresh seeds ({FRESH[0]}-{FRESH[-1]})\n")
    print(f"{'arm':26s} {'utility':>9s} {'catastr.':>9s} {'95% CI':>18s} "
          f"{'damage':>8s} {'pathology':>10s}")
    for k, rows in arms.items():
        u, c, _ = reward.robust_utility(rows, cal)
        flags = np.asarray([r["damage"] > reward.CATASTROPHE_DAMAGE for r in rows], float)
        blo, bhi = boot_rate(flags)
        wlo, whi = wilson(int(flags.sum()), len(flags))
        dm = np.asarray([r["damage"] for r in rows])
        e = {"utility": u, "catastrophe": c,
             "catastrophe_ci95_bootstrap": [blo, bhi],
             "catastrophe_ci95_wilson": [wlo, whi],
             "catastrophic_episodes": int(flags.sum()),
             "damage": float(dm.mean()), "damage_p90": float(np.percentile(dm, 90)),
             "pathology": float(np.mean([r["final_pathology"] for r in rows])),
             "safe_fraction": float(np.mean([r["safe_fraction"] for r in rows]))}
        out["arms"][k] = e
        print(f"{k:26s} {u:9.4f} {c:9.4f} [{blo:.4f},{bhi:.4f}] {dm.mean():8.3f} "
              f"{e['pathology']:10.4f}")

    L = out["arms"]["learned reference"]
    lc, lu = L["catastrophe"], L["utility"]
    nu = out["arms"]["no-op"]["utility"]
    if lc < 0.10:
        v, act = "PASS", "learned-control diagnostic PASS; TDP-43 may remain in the main set"
    elif lc <= 0.15:
        v, act = "INCONCLUSIVE", "keep TDP-43 experimental, not core"
    else:
        v, act = "FAIL", "exclude TDP-43 from the main set"
    out.update({"diagnostic_verdict": v, "action": act,
                "utility_improves_over_noop": bool(lu > nu)})
    print(f"\n  learned catastrophe {lc:.4f}  "
          f"({L['catastrophic_episodes']}/{N} episodes)")
    print(f"  bootstrap CI95 {L['catastrophe_ci95_bootstrap'][0]:.4f}-"
          f"{L['catastrophe_ci95_bootstrap'][1]:.4f}   "
          f"Wilson CI95 {L['catastrophe_ci95_wilson'][0]:.4f}-"
          f"{L['catastrophe_ci95_wilson'][1]:.4f}")
    print(f"\nVERDICT: {v}\n  -> {act}")
    print("  P8_original remains FAIL.")
    p = ROOT / "agentic/reports/audits/tdp43_precision_audit_N256.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
