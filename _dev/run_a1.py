"""A1 — relational necessity, per task.

Full edge-aware graph controller vs a local-only controller that sees every
per-node feature but cannot condition on WHICH partner a bead contacts.
Identical optimizer, budget, seeds and splits; selection on public validation.

Pass: full > local-only, hierarchical 95% CI excludes zero, effect >= 10% of the
gain over the zero anchor.

    python3 _dev/run_a1.py --task huntington-htt-polyq-v8 --budget 6000 --runs 5
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict as V

ROOT = Path(__file__).resolve().parents[1]
_W = {}


def _winit(task, local_only):
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
              pol=GraphPolicy(spec, local_only=local_only), local_only=local_only,
              profile=json.loads((ROOT / task / "environment/profile.json").read_text()),
              cal=json.loads((ROOT / task / "environment/reward_calibration.json").read_text())["calibration"])


def _job(args):
    cid, vec, seeds = args
    vec = np.asarray(vec, float)
    if _W["local_only"]:
        vec = vec.copy()
        vec[_W["spec"].message_param_index] = 0.0
    rows = []
    for s in seeds:
        env = _W["Env"](_W["profile"], seed=int(s))
        obs = env.observe()
        h = np.zeros(_W["spec"].HH)
        while env.steps < env.max_steps and env.budget > 0:
            act, h = _W["pol"].act(vec, obs, h)
            obs, _, done, _ = env.step(act)
            if done:
                break
        rows.append(env.summary())
    u, cat, vals = _W["reward"].robust_utility(rows, _W["cal"])
    return cid, u, cat, vals.tolist()


class Pool:
    def __init__(self, task, local_only, workers):
        self.ex = ProcessPoolExecutor(max_workers=workers,
                                      mp_context=mp.get_context("spawn"),
                                      initializer=_winit, initargs=(task, local_only))

    def batch(self, X, seeds):
        out = [0.0] * len(X)
        for cid, u, _, _ in self.ex.map(_job, [(i, x, seeds) for i, x in enumerate(X)],
                                        chunksize=1):
            out[cid] = u
        return np.asarray(out)

    def full(self, vec, seeds):
        _, u, cat, vals = list(self.ex.map(_job, [(0, vec, seeds)]))[0]
        return u, cat, vals

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
    return best_x, best_f, used


def hierarchical(VA, VB, n=20000, seed=909):
    """Runs resampled independently per arm; episodes paired (same eval set)."""
    VA, VB = np.asarray(VA, float), np.asarray(VB, float)
    SA, E = VA.shape
    SB = VB.shape[0]
    rng = np.random.default_rng(seed)
    f = lambda v: 0.60 * v.mean() + 0.40 * np.quantile(v, 0.20)
    diffs = np.empty(n)
    for b in range(n):
        ep = rng.integers(0, E, E)
        ia = rng.integers(0, SA, SA); ib = rng.integers(0, SB, SB)
        diffs[b] = (np.mean([f(VA[s, ep]) for s in ia])
                    - np.mean([f(VB[s, ep]) for s in ib]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi), float(np.mean(diffs > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--init-scale", type=float, default=0.02)
    a = ap.parse_args()
    np.seterr(all="ignore")
    env_dir = ROOT / a.task / "environment"
    sys.path.insert(0, str(env_dir))
    import policy_runtime as pr
    prof = json.loads((env_dir / "profile.json").read_text())
    tr = prof["train_seeds"][:a.train_seeds]
    val = prof["validation_seeds"]
    print(f"A1 — {a.task}: dim={pr.PARAM_COUNT}, budget={a.budget}, runs={a.runs}", flush=True)

    res = {"task": a.task, "budget": a.budget, "runs": a.runs,
           "controller_dim": int(pr.PARAM_COUNT), "arms": {}}

    zp = Pool(a.task, False, a.workers)
    try:
        zu, zc, _ = zp.full(pr.zero_policy(), val)
    finally:
        zp.close()
    res["anchor_zero"] = {"utility": zu, "catastrophe": zc}
    print(f"  anchor zero: {zu:+.4f} (cat {zc:.3f})", flush=True)

    for arm, lo_only in (("full_gnn", False), ("local_only", True)):
        pool = Pool(a.task, lo_only, a.workers)
        runs = []
        try:
            for r in range(a.runs):
                rng = np.random.default_rng(7000 + r)
                x0 = pr.random_policy(7000 + r, a.init_scale)
                if lo_only:
                    from neurofold8.policy import GraphPolicySpec
                    sp = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM,
                                         hidden=pr.HIDDEN, msg=pr.MSG, layers=pr.LAYERS,
                                         hist_dim=pr.HIST_DIM, hist_hidden=pr.HIST_HIDDEN)
                    x0 = x0.copy(); x0[sp.message_param_index] = 0.0
                t0 = time.perf_counter()
                best, ftr, gens = sepcmaes(lambda X: pool.batch(X, tr), pr.PARAM_COUNT,
                                           x0, a.budget // len(tr), rng, pr.MAX_ABS_WEIGHT)
                u, cat, vals = pool.full(best, val)
                runs.append({"run": r, "train": ftr, "valid": u, "catastrophe": cat,
                             "vals": vals, "episodes": gens * len(tr),
                             "wall_sec": time.perf_counter() - t0,
                             "vec": [float(x) for x in best]})
                print(f"  {arm:11s} run {r}: train={ftr:+.4f} valid={u:+.4f} "
                      f"cat={cat:.3f} {runs[-1]['wall_sec']:.0f}s", flush=True)
        finally:
            pool.close()
        res["arms"][arm] = runs

    VA = np.array([r["vals"] for r in res["arms"]["full_gnn"]])
    VB = np.array([r["vals"] for r in res["arms"]["local_only"]])
    lo, hi, p = hierarchical(VA, VB)
    mf = float(np.mean([r["valid"] for r in res["arms"]["full_gnn"]]))
    ml = float(np.mean([r["valid"] for r in res["arms"]["local_only"]]))
    gain = mf - zu
    frac = (mf - ml) / gain if abs(gain) > 1e-9 else None
    neg = 0.10 * abs(gain)                      # declared before the result: 10% of gain
    v = V.classify(lo, hi, neg)
    if v == V.PASS and (frac is None or frac < 0.10):
        v = V.INCONCLUSIVE                      # resolved but below the interesting-effect floor
    res["A1"] = {"mean_full": mf, "mean_local_only": ml, "difference": mf - ml,
                 "ci95": [lo, hi], "prob_full_gt_local": p, "gain_over_zero": gain,
                 "effect_fraction_of_gain": frac, "negligible_band": neg,
                 "verdict": v, "A1_pass": bool(v == V.PASS)}
    print(f"\n=== A1 verdict — {a.task} ===")
    print(f"  full GNN   {mf:+.4f}\n  local-only {ml:+.4f}")
    print(f"  difference {mf-ml:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  P={p:.3f}")
    print(f"  fraction of gain over zero: {frac}")
    print(f"  A1 VERDICT: {v}"
          + ("  (CI spans zero and exceeds the negligible band -> limited power, "
             "not evidence of absence)" if v == V.INCONCLUSIVE else ""))
    out = ROOT / "agentic/reports/experiments" / f"a1_{a.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
