"""Gate A6 — action-order dependence.

    U(sequence A) != U(sequence B) for the SAME action multiset

Specified in agentic/specs/ACCEPTANCE_CRITERIA.md as a per-task unit test and
never written for v8.0; P7 in the porting gate probed order with hand-coded
policies, but nothing asserted the invariant directly.

The test is a paired comparison under common random numbers: the same episode
seed, the same bag of (i, j, strength) actions, only the order permuted. If
maturation and locking are real, when a nucleus is attacked must matter, so the
two orders must separate.

Verdicts are three-valued, matching every other gate here: a CI spanning zero is
absence of power, not evidence that order is irrelevant.

    python3 _dev/gate_a6.py --task alzheimer-abeta42-v8
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

SEEDS = list(range(7200, 7264))     # disjoint from train, validation, calib, dev, A3 hold-out
N_PERM = 4


def rollout(Env, prof, seed, plan):
    env = Env(prof, seed=int(seed))
    env.observe()
    for act in plan:
        if env.steps >= env.max_steps or env.budget <= 0:
            break
        _, _, done, _ = env.step(act)
        if done:
            break
    return env.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="alzheimer-abeta42-v8")
    a = ap.parse_args()
    np.seterr(all="ignore")

    env_dir = ROOT / a.task / "environment"
    sys.path.insert(0, str(env_dir))
    from neurofold8 import reward
    from neurofold8.env import NeuroFoldV8Env

    prof = json.loads((env_dir / "profile.json").read_text())
    cal = json.loads((env_dir / "reward_calibration.json").read_text())["calibration"]
    n = len(prof["sequence"]) // prof["bead_size"] * prof["n_chains"]
    rng = np.random.default_rng(606)

    # One fixed multiset per seed: same actions, same count, only the order differs.
    diffs, rows_a, rows_b = [], [], []
    for s in SEEDS:
        r = np.random.default_rng(int(s))
        k = 24
        bag = [(int(r.integers(0, n)), int(r.integers(0, n)), float(r.uniform(0.4, 1.0)))
               for _ in range(k)]
        base = rollout(NeuroFoldV8Env, prof, s, bag)
        best = base
        for _ in range(N_PERM):
            perm = list(bag)
            rng.shuffle(perm)
            alt = rollout(NeuroFoldV8Env, prof, s, perm)
            if abs(reward.episode_value(alt, cal)[0] - reward.episode_value(base, cal)[0]) > \
               abs(reward.episode_value(best, cal)[0] - reward.episode_value(base, cal)[0]):
                best = alt
        rows_a.append(base)
        rows_b.append(best)
        diffs.append(reward.episode_value(best, cal)[0] - reward.episode_value(base, cal)[0])

    d = np.asarray(diffs, float)
    absd = np.abs(d)
    rng_b = np.random.default_rng(77)
    boots = np.array([absd[rng_b.integers(0, len(absd), len(absd))].mean()
                      for _ in range(20000)])
    lo, hi = np.quantile(boots, [0.025, 0.975])

    ua, _, _ = reward.robust_utility(rows_a, cal)
    ub, _, _ = reward.robust_utility(rows_b, cal)
    spread = abs(ua - ub)
    # Negligible band: 10% of the utility scale the task spans, matching the
    # convention used by the porting gate.
    negligible = 0.10 * abs(ua) if ua else 0.05
    if lo > negligible:
        verd = V.PASS
    elif hi < negligible:
        verd = V.FAIL
    else:
        verd = V.INCONCLUSIVE

    print(f"A6 — {a.task}   ({len(SEEDS)} seeds, {N_PERM} permutations each, "
          f"24 actions per multiset)")
    print(f"  mean |U(perm) - U(base)| = {absd.mean():.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  negligible band          = {negligible:.4f}")
    print(f"  aggregate utility base={ua:+.4f}  permuted={ub:+.4f}  spread={spread:.4f}")
    print(f"  episodes where order changed the outcome: "
          f"{100*np.mean(absd > 1e-9):.0f}%")
    print(f"\nA6: {verd}")
    out = {"task": a.task, "n_seeds": len(SEEDS), "n_permutations": N_PERM,
           "mean_abs_delta": float(absd.mean()), "ci95": [float(lo), float(hi)],
           "negligible_band": float(negligible), "utility_base": float(ua),
           "utility_permuted": float(ub),
           "fraction_order_sensitive": float(np.mean(absd > 1e-9)), "verdict": verd}
    p = ROOT / f"agentic/reports/validation/a6_{a.task}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
