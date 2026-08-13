"""Evaluate the TDP-43 learned reference against the pre-registered decision rule.

Fresh seeds 4000-4063: disjoint from train (1000-1063), validation (2000-2031),
the gate's dev range (2000-2047) and calibration (3000-3063).

Applies the rule fixed in
agentic/reports/audits/2026-08-13_tdp43_diagnostic_PREREGISTERED.md and does not
re-open P8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = "als-ftd-tdp43-v8"
FRESH = list(range(4000, 4064))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from porting_gate import Pool, POLICIES  # noqa: E402


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
    ref_path = ROOT / TASK / "solution/challenge_reference.json"
    if not ref_path.exists():
        raise SystemExit("no challenge_reference.json — training has not finished")
    vec = pr.load_policy(ref_path)

    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)
    pol = GraphPolicy(spec)

    def learned(seeds):
        rows = []
        for s in seeds:
            env = NeuroFoldV8Env(prof, seed=int(s))
            obs = env.observe()
            h = np.zeros(spec.HH)
            while env.steps < env.max_steps and env.budget > 0:
                act, h = pol.act(vec, obs, h)
                obs, _, done, _ = env.step(act)
                if done:
                    break
            rows.append(env.summary())
        return rows

    pool = Pool(TASK, prof, 5)
    try:
        heur = {k: pool.run(k, FRESH) for k in ("noop", "o10")}
    finally:
        pool.close()
    arms = {"no-op": heur["noop"], "heuristic oracle s=1.0": heur["o10"],
            "learned reference": learned(FRESH)}

    out = {"task": TASK, "seeds": FRESH, "n_episodes": len(FRESH),
           "P8_original": "FAIL", "threshold": 0.10,
           "note": "Diagnostic only. Does not re-open P8.", "arms": {}}
    print(f"TDP-43 diagnostic — {len(FRESH)} fresh seeds ({FRESH[0]}-{FRESH[-1]})\n")
    print(f"{'arm':26s} {'utility':>9s} {'catastr.':>9s} {'damage':>8s} {'pathology':>10s} {'p90 dmg':>8s}")
    for k, rows in arms.items():
        u, c, _ = reward.robust_utility(rows, cal)
        dm = np.asarray([r["damage"] for r in rows])
        e = {"utility": u, "catastrophe": c, "damage": float(dm.mean()),
             "damage_p90": float(np.percentile(dm, 90)),
             "pathology": float(np.mean([r["final_pathology"] for r in rows])),
             "safe_fraction": float(np.mean([r["safe_fraction"] for r in rows]))}
        out["arms"][k] = e
        print(f"{k:26s} {u:9.4f} {c:9.3f} {dm.mean():8.3f} {e['pathology']:10.4f} "
              f"{e['damage_p90']:8.3f}")

    lc = out["arms"]["learned reference"]["catastrophe"]
    lu = out["arms"]["learned reference"]["utility"]
    nu = out["arms"]["no-op"]["utility"]
    ho = out["arms"]["heuristic oracle s=1.0"]["catastrophe"]
    if lc < 0.10 and lu > nu:
        verdict, action = "PASS", "keep TDP-43; report heuristic P8 FAIL together with learned-control diagnostic PASS"
    elif lc <= 0.15:
        verdict, action = "INCONCLUSIVE", "do not freeze TDP-43; await a decision"
    else:
        verdict, action = "FAIL", "do not freeze TDP-43; ship four tasks"
    out.update({"learned_catastrophe": lc, "heuristic_catastrophe": ho,
                "diagnostic_verdict": verdict, "action": action})
    print(f"\n  heuristic oracle catastrophe: {ho:.3f}")
    print(f"  learned reference catastrophe: {lc:.3f}   (rule: <0.10 pass, 0.10-0.15 "
          f"inconclusive, >0.15 fail)")
    print(f"\nDIAGNOSTIC VERDICT: {verdict}\n  -> {action}")
    print("  P8_original remains FAIL regardless of this result.")
    p = ROOT / "agentic/reports/audits/tdp43_learned_control_diagnostic.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
