"""Freeze a v8 task: build the hidden profile, generate fresh final-test seeds,
measure the frozen anchors, and populate tests/.

Order matters and is enforced here: the hidden profile and the reference policy
are fixed FIRST, then the final-test seeds are generated, then the anchors are
measured once.  Nothing downstream re-tunes against those seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Hidden-side shift: a different dynamical regime, not a constant offset.
# Crowding runs higher (more aggressive association), the beta term and
# cooperativity are stronger, maturation is faster, and the modulation decays
# quicker — so a policy tuned only to the public regime does not transfer.
HIDDEN_SHIFT = {
    "conditions": {
        "crowding":    [1.06, 1.14],
        "screening":   [0.92, 1.00],
        "temperature": [1.00, 1.05],
        "oxidative":   [0.45, 0.65],
        "chaperone":   [0.25, 0.40],
    },
    "physics": {
        "eps_hb": 5.6,
        "eps_coop": 1.5,
        "tau_mat": 26,
        "mod_decay": 0.74,
        "block_decay": 0.70,
        "crowding_drift": 0.0022,
    },
}

_W = {}


def _winit(task, profile, vec):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    np.seterr(all="ignore")
    d = str(ROOT / task / "environment")
    if d not in sys.path:
        sys.path.insert(0, d)
    import policy_runtime as pr
    from neurofold8 import reward
    from neurofold8.env import NeuroFoldV8Env
    from neurofold8.policy import GraphPolicy, GraphPolicySpec
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)
    _W.update(pr=pr, reward=reward, Env=NeuroFoldV8Env, spec=spec,
              pol=GraphPolicy(spec), profile=profile, vec=np.asarray(vec, float))


def _run(seed):
    env = _W["Env"](_W["profile"], seed=int(seed))
    obs = env.observe()
    h = np.zeros(_W["spec"].HH)
    while env.steps < env.max_steps and env.budget > 0:
        act, h = _W["pol"].act(_W["vec"], obs, h)
        obs, _, done, _ = env.step(act)
        if done:
            break
    return env.summary()


def measure(task, profile, vec, seeds, cal, workers=5):
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_winit,
                             initargs=(task, profile, vec)) as pool:
        rows = list(pool.map(_run, seeds))
    sys.path.insert(0, str(ROOT / task / "environment"))
    from neurofold8 import reward
    u, cat, _ = reward.robust_utility(rows, cal)
    return u, cat, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="alzheimer-abeta42-v8")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    np.seterr(all="ignore")
    T = ROOT / a.task
    env_dir, tests_dir, sol_dir = T / "environment", T / "tests", T / "solution"
    sys.path.insert(0, str(env_dir))
    import policy_runtime as pr

    pub = json.loads((env_dir / "profile.json").read_text())
    cal = json.loads((env_dir / "reward_calibration.json").read_text())["calibration"]
    ref_path = sol_dir / "challenge_reference.json"
    if not ref_path.exists():
        raise SystemExit("challenge_reference.json missing — train it first")
    ref_vec = pr.load_policy(ref_path)

    # 1. hidden profile: shifted regime, no public seed lists carried over
    hid = json.loads(json.dumps(pub))
    hid["slug"] = pub["slug"] + "-hidden"
    hid["version"] = "8.0-hidden"
    hid["conditions"] = HIDDEN_SHIFT["conditions"]
    hid["physics"].update(HIDDEN_SHIFT["physics"])
    hid.pop("train_seeds", None)
    hid.pop("validation_seeds", None)
    hid["reward_calibration"] = cal

    # 2. fresh final-test seeds, disjoint from every public split.
    #    The draw is keyed on the task slug, so each task gets its own block and
    #    the block is still reproducible from the slug alone. A fixed constant
    #    here would hand every task the same 128 episodes.
    used = set(pub["train_seeds"]) | set(pub["validation_seeds"]) | set(range(3000, 3064))
    key = int.from_bytes(hashlib.sha256(a.task.encode()).digest()[:8], "big")
    rng = np.random.default_rng(key)
    seeds = []
    while len(seeds) < a.episodes:
        s = int(a.seed_base + rng.integers(0, 500000))
        if s not in used and s not in seeds:
            seeds.append(s)
    seeds.sort()
    hid["test_seeds"] = seeds

    # 3. anchors, measured once on the frozen seeds
    zero_u, zero_cat, _ = measure(a.task, hid, pr.zero_policy(), seeds, cal, a.workers)
    ref_u, ref_cat, _ = measure(a.task, hid, ref_vec, seeds, cal, a.workers)
    print(f"  anchors on {len(seeds)} final-test episodes:")
    print(f"    zero      utility={zero_u:+.4f} catastrophe={zero_cat:.3f}")
    print(f"    reference utility={ref_u:+.4f} catastrophe={ref_cat:.3f}")
    if ref_u - zero_u <= 1e-6:
        raise SystemExit("reference does not beat the zero anchor on the hidden split")

    hid["frozen_anchors"] = {
        "episodes": len(seeds),
        "lower_anchor_name": "zero_noop",
        "upper_anchor_name": "challenge_reference",
        "baseline_utility": zero_u,
        "reference_utility": ref_u,
        "baseline_catastrophe_rate": zero_cat,
        "reference_catastrophe_rate": ref_cat,
        "normalization": "reward = clip((utility - baseline_utility) / "
                         "(reference_utility - baseline_utility), 0, 1)",
        "test_seed_sha256": hashlib.sha256(
            json.dumps(seeds, separators=(",", ":")).encode()).hexdigest(),
        "policy_sha256": {"challenge_reference":
                          hashlib.sha256(ref_path.read_bytes()).hexdigest()},
    }
    (tests_dir / "hidden_profile.json").write_text(json.dumps(hid, indent=2) + "\n")

    # 4. populate tests/ with the code the verifier needs (no public seed lists)
    shutil.rmtree(tests_dir / "neurofold8", ignore_errors=True)
    shutil.copytree(env_dir / "neurofold8", tests_dir / "neurofold8")
    shutil.copy2(env_dir / "policy_runtime.py", tests_dir / "policy_runtime.py")
    shutil.copy2(ref_path, tests_dir / "challenge_reference.json")
    print(f"  wrote {tests_dir/'hidden_profile.json'} and populated tests/")


if __name__ == "__main__":
    main()
