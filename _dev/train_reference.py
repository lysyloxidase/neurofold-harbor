"""Dev-side trainer for the challenge reference policy (not shipped in the task).

Trains the shipped policy schema with sep-CMA-ES using a worker pool, selects on
the public validation split, and writes solution/challenge_reference.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _env_dir(task):
    return ROOT / task / "environment"


_W = {}


def _winit(task):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    np.seterr(all="ignore")
    d = str(_env_dir(task))
    if d not in sys.path:
        sys.path.insert(0, d)
    import policy_runtime as pr
    from neurofold8 import reward
    from neurofold8.env import NeuroFoldV8Env
    from neurofold8.policy import GraphPolicy, GraphPolicySpec
    prof = json.loads((_env_dir(task) / "profile.json").read_text())
    cal = json.loads((_env_dir(task) / "reward_calibration.json").read_text())["calibration"]
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)
    _W.update(pr=pr, reward=reward, Env=NeuroFoldV8Env, spec=spec,
              pol=GraphPolicy(spec), prof=prof, cal=cal)


def _rollout(vec, seed):
    env = _W["Env"](_W["prof"], seed=int(seed))
    obs = env.observe()
    h = np.zeros(_W["spec"].HH)
    while env.steps < env.max_steps and env.budget > 0:
        act, h = _W["pol"].act(vec, obs, h)
        obs, _, done, _ = env.step(act)
        if done:
            break
    return env.summary()


def _job(args):
    cid, vec, seeds = args
    rows = [_rollout(np.asarray(vec, float), s) for s in seeds]
    u, cat, _ = _W["reward"].robust_utility(rows, _W["cal"])
    return cid, u, cat, rows


class Pool:
    def __init__(self, task, workers):
        self.ex = ProcessPoolExecutor(max_workers=workers,
                                      mp_context=mp.get_context("spawn"),
                                      initializer=_winit, initargs=(task,))

    def batch(self, X, seeds):
        out = [0.0] * len(X)
        for cid, u, _, _ in self.ex.map(_job, [(i, x, seeds) for i, x in enumerate(X)],
                                        chunksize=1):
            out[cid] = u
        return np.asarray(out)

    def full(self, vec, seeds):
        _, u, cat, rows = list(self.ex.map(_job, [(0, vec, seeds)]))[0]
        return u, cat, rows

    def close(self):
        self.ex.shutdown(wait=True)


def sepcmaes(fb, dim, x0, budget_gen, rng, clip, sigma0=0.35):
    lam = 4 + int(3 * np.log(dim)); mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    corr = (dim + 2) / 3.0; c1 *= corr; cmu = min(1 - c1, cmu * corr)
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    xmean = np.asarray(x0, float).copy(); sigma = sigma0
    pc = np.zeros(dim); ps = np.zeros(dim); d = np.ones(dim)
    used = 0; best_x = xmean.copy(); best_f = -np.inf
    while used + lam <= budget_gen:
        z = rng.standard_normal((lam, dim))
        X = np.clip(xmean + sigma * z * d, -clip, clip)
        vals = fb(X); used += lam
        idx = np.argsort(-vals)
        if vals[idx[0]] > best_f:
            best_f, best_x = float(vals[idx[0]]), X[idx[0]].copy()
        xold = xmean; xmean = w @ X[idx[:mu]]
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ((xmean - xold) / (sigma * d))
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * used / lam)) / chiN < 1.4 + 2 / (dim + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        artmp = (X[idx[:mu]] - xold) / sigma
        cd = d * d
        cd = ((1 - c1 - cmu) * cd + c1 * (pc * pc + (not hsig) * cc * (2 - cc) * cd)
              + cmu * (w[:, None] * artmp * artmp).sum(0))
        d = np.sqrt(np.maximum(cd, 1e-20))
        sigma = float(np.clip(sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1)),
                              1e-8, 1e3))
        print(f"    gen evals={used} best_train={best_f:+.4f} sigma={sigma:.3f}", flush=True)
    return best_x, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="alzheimer-abeta42-v8")
    ap.add_argument("--budget", type=int, default=12000)
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    np.seterr(all="ignore")
    sys.path.insert(0, str(_env_dir(a.task)))
    import policy_runtime as pr
    prof = json.loads((_env_dir(a.task) / "profile.json").read_text())
    tr = prof["train_seeds"][:a.train_seeds]
    val = prof["validation_seeds"]

    pool = Pool(a.task, a.workers)
    best_overall, best_val = None, -np.inf
    try:
        for r in range(a.restarts):
            rng = np.random.default_rng(9000 + r)
            x0 = pr.random_policy(9000 + r, 0.02)
            t0 = time.perf_counter()
            print(f"  restart {r}:", flush=True)
            x, ftr = sepcmaes(lambda X: pool.batch(X, tr), pr.PARAM_COUNT, x0,
                              a.budget // len(tr), rng, pr.MAX_ABS_WEIGHT)
            u, cat, _ = pool.full(x, val)
            print(f"  restart {r}: train={ftr:+.4f} valid={u:+.4f} cat={cat:.3f} "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            if u > best_val:
                best_val, best_overall = u, x
    finally:
        pool.close()

    out = ROOT / a.task / "solution" / "challenge_reference.json"
    pr.save_policy(best_overall, out)
    print(json.dumps({"validation_utility": best_val, "artifact": str(out),
                      "parameters": int(pr.PARAM_COUNT)}, indent=2))


if __name__ == "__main__":
    main()
