"""Higher-power re-test of P7 (action order) for one task.

Declared post hoc: the original P7 used 48 dev seeds and returned INCONCLUSIVE.
This re-runs the identical comparison with more seeds. The threshold and the
negligible band are unchanged; only n increases.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict as V
from porting_gate import Pool, POLICIES, boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", type=int, default=96)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    np.seterr(all="ignore")
    env_dir = ROOT / a.task / "environment"
    sys.path.insert(0, str(env_dir))
    from neurofold8 import reward

    prof = json.loads((env_dir / "profile.json").read_text())
    cal = json.loads((env_dir / "reward_calibration.json").read_text())["calibration"]
    DEV = list(range(2000, 2000 + a.seeds))
    pool = Pool(a.task, prof, a.workers)
    try:
        arms = {k: pool.run(k, DEV) for k in ("noop", "o03", "o06", "o10")}
        util = {k: reward.robust_utility(v, cal) for k, v in arms.items()}
        best = max(("o03", "o06", "o10"), key=lambda k: util[k][0])
        st = float(best[1:3]) / 10.0
        early = pool.run("_sched", DEV, (st, 0, 20))
        late = pool.run("_sched", DEV, (st, 60, 20))
    finally:
        pool.close()
    de = np.asarray([reward.episode_value(r, cal)[0] for r in early])
    dl = np.asarray([reward.episode_value(r, cal)[0] for r in late])
    lo, hi = boot(de - dl)
    neg = 0.10 * abs(util[best][0] - util["noop"][0])
    v = V.classify(lo, hi, neg, two_sided=True)
    out = V.summarise("P7_action_order_matters", lo, hi, (de - dl).mean(), neg, two_sided=True)
    out.update({"task": a.task, "seeds": a.seeds, "arm": best,
                "note_power": f"post-hoc power increase from 48 to {a.seeds} dev seeds; "
                              "threshold and negligible band unchanged"})
    print(f"P7 re-test — {a.task}, {a.seeds} seeds, arm {best}")
    print(f"  early-late = {(de-dl).mean():+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  band {neg:.3f}")
    print(f"  VERDICT: {v}")
    p = ROOT / "agentic/reports/validation" / f"p7_retest_{a.task}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
