"""Agent-facing CLI for NeuroFold-Harbor v8.

    python neurofold_cli.py inspect
    python neurofold_cli.py init-policy --out policy.json
    python neurofold_cli.py evaluate --policy policy.json --split validation
    python neurofold_cli.py publish  --policy policy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurofold8 import reward
from neurofold8.env import NeuroFoldV8Env
from neurofold8.policy import GraphPolicy, GraphPolicySpec
import policy_runtime as pr

HERE = Path(__file__).resolve().parent
PROFILE = json.loads((HERE / "profile.json").read_text())
CALIBRATION = json.loads((HERE / "reward_calibration.json").read_text())


def make_policy():
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM,
                           hidden=pr.HIDDEN, msg=pr.MSG, layers=pr.LAYERS,
                           hist_dim=pr.HIST_DIM, hist_hidden=pr.HIST_HIDDEN)
    return spec, GraphPolicy(spec)


def rollout(vec, seed, profile=PROFILE):
    spec, pol = make_policy()
    env = NeuroFoldV8Env(profile, seed=int(seed))
    obs = env.observe()
    h = np.zeros(spec.HH)
    while env.steps < env.max_steps and env.budget > 0:
        act, h = pol.act(vec, obs, h)
        obs, _, done, _ = env.step(act)
        if done:
            break
    return env.summary()


def evaluate(vec, seeds):
    rows = [rollout(vec, s) for s in seeds]
    u, cat, vals = reward.robust_utility(rows, CALIBRATION["calibration"])
    return {"utility": u, "catastrophe_rate": cat, "episodes": len(rows),
            "mean_pathology": float(np.mean([r["final_pathology"] for r in rows])),
            "mean_damage": float(np.mean([r["damage"] for r in rows])),
            "mean_safe_fraction": float(np.mean([r["safe_fraction"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    ip = sub.add_parser("init-policy"); ip.add_argument("--out", default="/app/policy.json")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--policy", default="/app/policy.json")
    ev.add_argument("--split", default="validation", choices=["train", "validation"])
    ev.add_argument("--limit", type=int, default=0)
    pb = sub.add_parser("publish"); pb.add_argument("--policy", default="/app/policy.json")
    a = ap.parse_args()

    if a.cmd == "inspect":
        env = NeuroFoldV8Env(PROFILE, seed=PROFILE["train_seeds"][0])
        obs = env.observe()
        print(json.dumps({
            "protein": PROFILE["protein"], "disease": PROFILE["disease"],
            "chains": PROFILE["n_chains"], "beads": env.n,
            "bead_size": PROFILE["bead_size"], "episode_steps": env.max_steps,
            "action": "(i, j, strength): contact-selective destabilisation of pair "
                      "(i,j), followed by ordinary Metropolis relaxation",
            "node_features": int(obs["node"].shape[1]),
            "edge_features": int(obs["edge"].shape[1]),
            "history_frames": int(obs["history"].shape[0]),
            "policy_parameters": pr.PARAM_COUNT,
            "train_seeds": PROFILE["train_seeds"],
            "validation_seeds": PROFILE["validation_seeds"],
            "note": "Hidden evaluation uses unseen episodes and a shifted "
                    "mechanism/condition set. No target basin is exposed.",
        }, indent=2))
        return

    if a.cmd == "init-policy":
        pr.save_policy(pr.zero_policy(), a.out)
        print(a.out)
        return

    if a.cmd == "publish":
        vec = pr.load_policy(a.policy)
        out = Path("/logs/artifacts/policy.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        pr.save_policy(vec, out)
        print(out)
        return

    vec = pr.load_policy(a.policy)
    seeds = PROFILE["train_seeds"] if a.split == "train" else PROFILE["validation_seeds"]
    if a.limit:
        seeds = seeds[:a.limit]
    print(json.dumps(evaluate(vec, seeds), indent=2))


if __name__ == "__main__":
    main()
