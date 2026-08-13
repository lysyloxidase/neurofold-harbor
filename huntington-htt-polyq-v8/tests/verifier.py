"""Hidden verifier for NeuroFold-Harbor v8.

Reads ONLY the numeric artifact at /logs/artifacts/policy.json.  No
agent-authored code is imported or executed.  Evaluation runs on frozen
final-test seeds under a hidden profile whose conditions and mechanism weights
are shifted relative to the public environment.

Writes:
  /logs/verifier/reward.txt    continuous scalar in [0, 1]
  /logs/verifier/metrics.json  full detail, including an uncapped extended score
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tests")

from neurofold8 import reward as reward_mod
from neurofold8.env import NeuroFoldV8Env
from neurofold8.policy import GraphPolicy, GraphPolicySpec
import policy_runtime as pr

POLICY = Path("/logs/artifacts/policy.json")
LOG = Path("/logs/verifier")
PROFILE_PATH = Path("/tests/hidden_profile.json")
REFERENCE = Path("/tests/challenge_reference.json")
WORKERS = 4

_W = {}


def _init(vec, profile, cal):
    np.seterr(all="ignore")
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)
    _W.update(vec=np.asarray(vec, float), profile=profile, cal=cal,
              spec=spec, pol=GraphPolicy(spec))


def _run_seed(seed):
    env = NeuroFoldV8Env(_W["profile"], seed=int(seed))
    obs = env.observe()
    h = np.zeros(_W["spec"].HH)
    while env.steps < env.max_steps and env.budget > 0:
        act, h = _W["pol"].act(_W["vec"], obs, h)
        obs, _, done, _ = env.step(act)
        if done:
            break
    return env.summary()


def evaluate(vec, profile, cal, seeds):
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(WORKERS, len(seeds)), initializer=_init,
            initargs=(vec, profile, cal)) as pool:
        rows = list(pool.map(_run_seed, seeds))
    u, cat, vals = reward_mod.robust_utility(rows, cal)
    return rows, u, cat, vals


def bootstrap_ci(rows, cal, base, ref, n=500):
    rng = np.random.default_rng(8008)
    m = len(rows)
    gap = ref - base
    out = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        u, _, _ = reward_mod.robust_utility([rows[int(i)] for i in idx], cal)
        out.append((u - base) / gap)
    q = np.quantile(np.asarray(out, float), [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    try:
        if not POLICY.exists():
            raise FileNotFoundError("policy.json was not published to /logs/artifacts")
        vec = pr.load_policy(POLICY)          # strict schema / bounds / size validation

        profile = json.loads(PROFILE_PATH.read_text())
        cal = profile["reward_calibration"]
        seeds = [int(s) for s in profile["test_seeds"]]
        anchors = profile["frozen_anchors"]
        if len(seeds) != anchors["episodes"]:
            raise ValueError("final-test episode count mismatch")
        if hashlib.sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest() \
                != anchors["test_seed_sha256"]:
            raise ValueError("final-test seed hash mismatch")
        ref_sha = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        if ref_sha != anchors["policy_sha256"]["challenge_reference"]:
            raise ValueError("challenge reference hash mismatch")

        base = float(anchors["baseline_utility"])
        ref = float(anchors["reference_utility"])
        gap = ref - base
        if not np.isfinite(gap) or gap <= 1e-9:
            raise ValueError("invalid frozen anchor gap")

        rows, utility, catastrophe, vals = evaluate(vec, profile, cal, seeds)
        extended = float((utility - base) / gap)
        rew = float(np.clip(extended, 0.0, 1.0))
        ci = bootstrap_ci(rows, cal, base, ref)

        details = {
            "reward": rew,
            "extended_score": extended,
            "score_ci95_unclipped": ci,
            "raw_utility": utility,
            "utility_above_reference": float(utility - ref),
            "baseline_anchor": "zero_noop",
            "baseline_utility": base,
            "reference_anchor": "challenge_reference",
            "reference_utility": ref,
            "catastrophe_rate": catastrophe,
            "reference_catastrophe_rate": float(anchors["reference_catastrophe_rate"]),
            "baseline_catastrophe_rate": float(anchors["baseline_catastrophe_rate"]),
            "mean_final_pathology": float(np.mean([r["final_pathology"] for r in rows])),
            "mean_pathology_reduction": float(np.mean([r["pathology_reduction"] for r in rows])),
            "mean_damage": float(np.mean([r["damage"] for r in rows])),
            "mean_safe_fraction": float(np.mean([r["safe_fraction"] for r in rows])),
            "mean_action_energy": float(np.mean([r["action_energy"] for r in rows])),
            "episodes": len(rows),
            "normalization": anchors["normalization"],
            "policy_parameters": int(pr.PARAM_COUNT),
            "final_test_seed_sha256": anchors["test_seed_sha256"],
        }
    except Exception as exc:
        rew = 0.0
        details = {"reward": 0.0, "error": f"{type(exc).__name__}: {exc}"}

    (LOG / "reward.txt").write_text(f"{rew:.12g}\n")
    (LOG / "metrics.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
