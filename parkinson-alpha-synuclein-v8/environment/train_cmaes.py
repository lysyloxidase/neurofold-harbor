"""Public black-box training baseline: separable CMA-ES over the policy vector.

    python train_cmaes.py --budget 6000 --seeds 8 --out policy.json

Selection uses the public validation split only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import policy_runtime as pr
from neurofold_cli import PROFILE, evaluate, rollout
from neurofold8 import reward

CAL = json.loads((Path(__file__).resolve().parent / "reward_calibration.json").read_text())["calibration"]


def fitness(vec, seeds):
    return reward.robust_utility([rollout(vec, s) for s in seeds], CAL)[0]


def sepcmaes(f, dim, x0, budget_gen, rng, sigma0=0.35, log=None):
    lam = 4 + int(3 * np.log(dim))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    corr = (dim + 2) / 3.0
    c1 *= corr; cmu = min(1 - c1, cmu * corr)
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    xmean = np.asarray(x0, float).copy()
    sigma = sigma0
    pc = np.zeros(dim); ps = np.zeros(dim); d = np.ones(dim)
    used = 0; best_x = xmean.copy(); best_f = -np.inf
    while used + lam <= budget_gen:
        z = rng.standard_normal((lam, dim))
        X = np.clip(xmean + sigma * z * d, -pr.MAX_ABS_WEIGHT, pr.MAX_ABS_WEIGHT)
        vals = np.array([f(x) for x in X]); used += lam
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
        sigma = float(np.clip(sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1)), 1e-8, 1e3))
        if log is not None:
            log.append({"evals": used * len(TRAIN), "best_train": best_f})
    return best_x, best_f, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=6000, help="training episodes")
    ap.add_argument("--seeds", type=int, default=8, help="training seeds per evaluation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/app/policy.json")
    ap.add_argument("--init-scale", type=float, default=0.02)
    a = ap.parse_args()
    np.seterr(all="ignore")
    global TRAIN
    TRAIN = PROFILE["train_seeds"][:a.seeds]
    rng = np.random.default_rng(a.seed)
    x0 = pr.random_policy(a.seed, a.init_scale)
    t0 = time.perf_counter()
    best, ftrain, gens = sepcmaes(lambda x: fitness(x, TRAIN), pr.PARAM_COUNT, x0,
                                  a.budget // len(TRAIN), rng)
    val = evaluate(best, PROFILE["validation_seeds"])
    pr.save_policy(best, a.out)
    print(json.dumps({"train_fitness": ftrain, "validation": val,
                      "episodes": gens * len(TRAIN),
                      "wall_sec": time.perf_counter() - t0, "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
