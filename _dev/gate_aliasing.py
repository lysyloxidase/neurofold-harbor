"""Aliasing test: are per-edge local features sufficient to pick the target?

Pre-registered before the run (see the proposal in the session log). Nothing
below is chosen after seeing a result.

Ground truth is a counterfactual measured in the simulator, independent of any
policy and of the reward:

    V(s, i, j) = pathology( s -> block (i,j) -> H steps )
               - pathology( s -> no-op       -> H steps )

evaluated under common random numbers, so the difference is the effect of the
action and not of the noise. H = 12: a single block decays below the blocking
threshold in about 3 steps, and its effect on an assembly shows up within a
dozen.

Two questions, answered separately:

  A  Does aliasing exist? Reported as the fraction of contacts having an
     eps-twin whose value differs by more than delta, and -- threshold-free --
     as the ceiling on any function of local features, estimated from the
     residual variance of V inside eps-neighbourhoods. That ceiling is a Bayes
     limit for a local-only policy, however well trained.

  B  Does message passing recover it? Two probes on the same data predict V:
     one from the 11 local edge features, one from those plus two rounds of
     aggregation over the contact graph. Probes, not policies, so the question
     of INFORMATION is separated from the question of OPTIMIZATION.

Pre-registered verdict:

    local R2 < 0.50 and relational R2 > 0.75   -> PASS
    local R2 > 0.75                            -> FAIL (shortcut survives)
    otherwise                                  -> INCONCLUSIVE

    python3 _dev/gate_aliasing.py --task alzheimer-abeta42-v8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_dev"))
import verdict as V  # noqa: E402

HORIZON = 12
SNAP_EVERY = 8
EPS = 0.05
DELTA = 0.5
SEEDS = list(range(7300, 7396))      # disjoint from every other split
_W = {}


def _winit(task):
    env_dir = str(ROOT / task / "environment")
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)
    import numpy as _np
    _np.seterr(all="ignore")
    from neurofold8.env import NeuroFoldV8Env
    prof = json.loads((ROOT / task / "environment/profile.json").read_text())
    _W.update(Env=NeuroFoldV8Env, prof=prof)


def _clone_state(env):
    """Deep enough copy to replay a counterfactual from the same state."""
    import copy
    return copy.deepcopy(env)


def _episode(seed):
    """Snapshots of (edge features, edge index, counterfactual values).

    The graph is kept, not just the feature rows, so the relational probe can
    aggregate over it. Sampling is the expensive part; doing it once serves both
    halves of the test.
    """
    Env, prof = _W["Env"], _W["prof"]
    env = Env(prof, seed=int(seed))
    obs = env.observe()
    snaps = []
    while env.steps < env.max_steps and env.budget > 0:
        if env.steps % SNAP_EVERY == 0 and env.steps > 0:
            ei, ej = obs["edge_index"]
            base = _clone_state(env)
            for _ in range(HORIZON):
                if base.steps >= base.max_steps or base.budget <= 0:
                    break
                base.step((0, 0.0, 0.0))
            p_noop = float(base.pathology)
            vals = np.zeros(len(ei))
            for k in range(len(ei)):
                cf = _clone_state(env)
                cf.step((int(ei[k]), int(ej[k]), 1.0))
                for _ in range(HORIZON - 1):
                    if cf.steps >= cf.max_steps or cf.budget <= 0:
                        break
                    cf.step((0, 0.0, 0.0))
                vals[k] = float(cf.pathology) - p_noop
            snaps.append({"edge": obs["edge"].copy(),
                          "ei": np.asarray(ei).copy(), "ej": np.asarray(ej).copy(),
                          "n": int(env.n), "y": vals})
        obs, _, done, _ = env.step((0, 0.0, 0.0))
        if done:
            break
    return snaps


def relational_features(snap, rounds=2):
    """Two rounds of aggregation over the contact graph.

    Exactly what message passing has access to and a per-edge readout does not:
    for each edge, the mean and max of its neighbours' features at both
    endpoints, iterated. No hand-coded notion of nucleus, assembly or
    criticality appears here — only aggregation.
    """
    E, ei, ej, n = snap["edge"], snap["ei"], snap["ej"], snap["n"]
    extra = []
    cur = E
    for _ in range(rounds):
        d = cur.shape[1]                     # widens each round; size node to match
        node = np.zeros((n, d))
        cnt = np.zeros(n) + 1e-9
        np.add.at(node, ei, cur)
        np.add.at(cnt, ei, 1.0)
        np.add.at(node, ej, cur)
        np.add.at(cnt, ej, 1.0)
        mean_node = node / cnt[:, None]
        nmax = np.full((n, d), -np.inf)
        np.maximum.at(nmax, ei, cur)
        np.maximum.at(nmax, ej, cur)
        nmax[~np.isfinite(nmax)] = 0.0
        agg = np.concatenate([mean_node[ei], mean_node[ej],
                              nmax[ei], nmax[ej]], axis=1)
        extra.append(agg)
        cur = np.concatenate([E, mean_node[ei] + mean_node[ej]], axis=1)
    deg = np.zeros(n)
    np.add.at(deg, ei, 1.0)
    np.add.at(deg, ej, 1.0)
    extra.append(np.column_stack([deg[ei], deg[ej]]))
    return np.concatenate([E] + extra, axis=1)


def ridge_r2(X, y, folds=4, lam=1e-3):
    """Out-of-fold R^2 of a ridge fit; the probe is deliberately simple."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    parts = np.array_split(idx, folds)
    pred = np.zeros(n)
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        A = np.column_stack([X[tr], np.ones(len(tr))])
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([X[te], np.ones(len(te))]) @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def local_ceiling(Z, y, eps=EPS, k=24):
    """Bayes ceiling for any function of local features.

    Inside an eps-ball every local-only rule must return the same answer, so the
    best it can do is the neighbourhood mean. What is left is irreducible.
    """
    n = len(y)
    rng = np.random.default_rng(1)
    sub = rng.choice(n, min(n, 4000), replace=False)
    resid, tot = [], []
    for i in sub:
        d = np.linalg.norm(Z - Z[i], axis=1)
        nb = np.argsort(d)[:k]
        nb = nb[d[nb] <= eps]
        if len(nb) < 4:
            continue
        resid.append((y[i] - y[nb].mean()) ** 2)
        tot.append((y[i] - y.mean()) ** 2)
    if not resid:
        return float("nan"), 0
    return 1.0 - float(np.mean(resid)) / float(np.mean(tot)), len(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="alzheimer-abeta42-v8")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    np.seterr(all="ignore")

    print(f"test aliasingu — {a.task}")
    print(f"  horyzont H={HORIZON}, migawki co {SNAP_EVERY} krokow, "
          f"{len(SEEDS)} epizodow (seedy {SEEDS[0]}-{SEEDS[-1]})")
    print(f"  progi ustalone z gory: eps={EPS}, delta={DELTA}\n")

    # Sampling is the expensive half; cache it so a bug in the analysis cannot
    # throw the simulation away, as it did on the first attempt.
    cache = ROOT / "_dev" / f"aliasing_samples_{a.task}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        snaps = list(z["snaps"])
        print(f"  wczytano {len(snaps)} migawek z {cache.name}")
    else:
        ex = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_winit, initargs=(a.task,))
        snaps = []
        try:
            for i, s in enumerate(ex.map(_episode, SEEDS, chunksize=1)):
                snaps.extend(s)
                if (i + 1) % 16 == 0:
                    print(f"    {i+1}/{len(SEEDS)} epizodow, par: "
                          f"{sum(len(z['y']) for z in snaps)}", flush=True)
        finally:
            ex.shutdown(wait=True)
        np.savez_compressed(cache, snaps=np.array(snaps, dtype=object))
        print(f"  zapisano probki do {cache.name}")

    X = np.vstack([s["edge"] for s in snaps])
    y = np.concatenate([s["y"] for s in snaps])
    R = np.vstack([relational_features(s) for s in snaps])
    print(f"\n  zebrano {len(y)} par (stan, kontakt) z {len(snaps)} migawek")
    print(f"  cech lokalnych: {X.shape[1]}, relacyjnych: {R.shape[1]}")
    print(f"  V: srednia {y.mean():+.4f}, sd {y.std():.4f}, "
          f"|V|>delta w {100*np.mean(np.abs(y) > DELTA):.1f}%")

    def z(M):
        s = M.std(0)
        s[s == 0] = 1.0
        return (M - M.mean(0)) / s

    Z, ZR = z(X), z(R)
    ceil, used = local_ceiling(Z, y)
    r2_local = ridge_r2(Z, y)
    r2_rel = ridge_r2(ZR, y)

    print(f"\n  A. granica lokalna (Bayes, eps-sasiedztwa, n={used}): R2 = {ceil:.3f}")
    print(f"     sonda liniowa, cechy lokalne:                     R2 = {r2_local:.3f}")
    print(f"  B. sonda liniowa, cechy + 2 rundy agregacji:         R2 = {r2_rel:.3f}")
    print(f"     zysk z informacji relacyjnej:                     {r2_rel - r2_local:+.3f}")

    if ceil < 0.50 and r2_rel > 0.75:
        verd, note = V.PASS, ("Local features cannot determine the target while aggregation "
                              "over the contact graph can: the shortcut is gone.")
    elif ceil > 0.75:
        verd, note = V.FAIL, ("Local features still determine the target. The redesign did "
                              "not remove the shortcut.")
    else:
        verd, note = V.INCONCLUSIVE, ("Neither branch of the pre-registered rule is met.")

    print(f"\nALIASING: {verd}\n  {note}")
    out = {"task": a.task, "horizon": HORIZON, "eps": EPS, "delta": DELTA,
           "snapshots": len(snaps), "n_pairs": int(len(y)),
           "n_local_features": int(X.shape[1]), "n_relational_features": int(R.shape[1]),
           "V_mean": float(y.mean()), "V_sd": float(y.std()),
           "frac_abs_V_gt_delta": float(np.mean(np.abs(y) > DELTA)),
           "local_ceiling_r2": float(ceil), "local_probe_r2": float(r2_local),
           "relational_probe_r2": float(r2_rel),
           "relational_gain": float(r2_rel - r2_local),
           "verdict": verd, "note": note, "seeds": [SEEDS[0], SEEDS[-1]]}
    p = ROOT / f"agentic/reports/validation/aliasing_{a.task}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
