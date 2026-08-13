"""The 8 pre-training gates for a new protein.

Runs on public/dev splits only, before any reference policy is trained.
A failure stops the port and is reported with its measured value; it is not
tuned around.

    python3 _dev/porting_gate.py --task huntington-htt-polyq-v8
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict as V

ROOT = Path(__file__).resolve().parents[1]
CALIB = list(range(3000, 3032))
DEV = list(range(2000, 2048))

_W = {}


def _winit(task, profile):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    np.seterr(all="ignore")
    d = str(ROOT / task / "environment")
    if d not in sys.path:
        sys.path.insert(0, d)
    from neurofold8.env import NeuroFoldV8Env
    _W.update(Env=NeuroFoldV8Env, profile=profile)


# ---------------------------------------------------------------- policies
def noop(env):
    return (0, 0, 0.0)


def oracle(strength):
    def pol(env):
        ii, jj = np.nonzero(np.triu(env.aux["ladder"]))
        if len(ii) == 0:
            return (0, 0, 0.0)
        k = env.steps % len(ii)
        return (int(ii[k]), int(jj[k]), strength)
    return pol


def blind(strength):
    def pol(env):
        return (int(env.steps % env.n), int((env.steps * 7 + 3) % env.n), strength)
    return pol


def scheduled_action(env, strength, delay, n_actions):
    """Plain function, not a closure: closures cannot be pickled to spawn workers."""
    if delay <= env.steps < delay + n_actions:
        return oracle(strength)(env)
    return (0, 0, 0.0)


POLICIES = {"noop": noop, "o03": oracle(0.3), "o06": oracle(0.6), "o10": oracle(1.0),
            "b06": blind(0.6), "b10": blind(1.0)}


def _job(args):
    kind, seed, arg = args
    env = _W["Env"](_W["profile"], seed=int(seed))
    while env.steps < env.max_steps and env.budget > 0:
        if kind == "_sched":
            act = scheduled_action(env, *arg)
        else:
            act = POLICIES[kind](env)
        _, _, done, _ = env.step(act)
        if done:
            break
    return env.summary()


class Pool:
    def __init__(self, task, profile, workers=5):
        self.ex = ProcessPoolExecutor(max_workers=workers,
                                      mp_context=mp.get_context("spawn"),
                                      initializer=_winit, initargs=(task, profile))

    def run(self, kind, seeds, pol=None):
        return list(self.ex.map(_job, [(kind, s, pol) for s in seeds], chunksize=1))

    def close(self):
        self.ex.shutdown(wait=True)


def boot(d, n=20000, seed=11):
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.quantile(bs, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    np.seterr(all="ignore")
    env_dir = ROOT / a.task / "environment"
    sys.path.insert(0, str(env_dir))
    from neurofold8 import geom, reward
    from neurofold8.env import NeuroFoldV8Env

    prof = json.loads((env_dir / "profile.json").read_text())
    results, verdicts = {}, {}

    # ---- P1 geometry responds -------------------------------------------
    e = NeuroFoldV8Env(prof, seed=11)
    moved = sum(int(np.abs(geom.crankshaft(e.x, e.chain_id, i, 0.4) - e.x).max() > 1e-9)
                for i in range(e.n))
    moved_d = sum(int(np.abs(geom.displace(e.x, e.chain_id, i, 0.2) - e.x).max() > 1e-9)
                  for i in range(e.n))
    e0 = e.energy_total
    inert = [i for i in range(e.n)
             if abs(e.energy.total(geom.apply_action(e.x, e.chain_id, i, 0.6, 0.25),
                                   e.env_state["screening"],
                                   crowding=e.env_state["crowding"], mod=e.mod)[0] - e0) < 1e-9]
    verdicts["P1_geometry_responds"] = V.PASS if (moved == e.n and moved_d == e.n and not inert) else V.FAIL
    results["P1"] = {"beads": e.n, "crankshaft_moved": moved, "displace_moved": moved_d,
                     "energy_inert_beads": inert}

    pool = Pool(a.task, prof, a.workers)
    try:
        # ---- P2 reward calibration non-degenerate ------------------------
        rows = []
        for k in POLICIES:
            rows += pool.run(k, CALIB)
        try:
            cal = reward.calibrate(rows, strict=True)
            degenerate = False
        except reward.DegenerateCalibration as exc:
            cal, degenerate = reward.calibrate(rows, strict=False), str(exc)
        clip = {}
        for k in reward.COMPONENTS:
            v = np.asarray([r[k] for r in rows], float)
            z = (v - cal[k]["median"]) / cal[k]["scale"]
            clip[k] = float(np.mean(np.abs(z) >= 6.0))
        verdicts["P2_calibration_non_degenerate"] = V.PASS if (degenerate is False
                                                     and max(clip.values()) < 0.05) else V.FAIL
        results["P2"] = {"degenerate": degenerate,
                         "max_clip_fraction": max(clip.values()),
                         "scales": {k: cal[k]["scale"] for k in cal}}
        (ROOT / a.task / "environment" / "reward_calibration.json").write_text(json.dumps(
            {"weights": reward.WEIGHTS, "catastrophe_penalty": reward.CATASTROPHE_PENALTY,
             "catastrophe_damage_threshold": reward.CATASTROPHE_DAMAGE,
             "calibration_seeds": CALIB,
             "note": "Robust scales from a public policy mixture; calibration seeds only.",
             "calibration": cal}, indent=2) + "\n")

        # ---- evaluate the arms on the dev split (CRN) --------------------
        arms = {k: pool.run(k, DEV) for k in POLICIES}
        util = {k: reward.robust_utility(v, cal) for k, v in arms.items()}
        ep = {k: np.asarray([reward.episode_value(r, cal)[0] for r in arms[k]])
              for k in arms}
        results["arms"] = {k: {"utility": util[k][0], "catastrophe": util[k][1],
                               "pathology": float(np.mean([r["final_pathology"] for r in arms[k]])),
                               "damage": float(np.mean([r["damage"] for r in arms[k]])),
                               "locked": float(np.mean([r["locked_pairs"] for r in arms[k]]))}
                          for k in arms}

        # ---- P3 no-op not optimum ---------------------------------------
        best = max((k for k in arms if k != "noop"), key=lambda k: util[k][0])
        d = ep[best] - ep["noop"]
        lo, hi = boot(d)
        neg3 = 0.10 * abs(util[best][0] - util["noop"][0])
        verdicts["P3_noop_not_optimum"] = V.classify(lo, hi, neg3)
        results["P3"] = {"best_arm": best, "d": float(d.mean()), "ci": [float(lo), float(hi)]}

        # ---- P4 pathology matures ---------------------------------------
        locked = float(np.mean([r["locked_pairs"] for r in arms["noop"]]))
        verdicts["P4_pathology_matures"] = V.PASS if bool(locked > 1.0) else V.FAIL
        results["P4"] = {"noop_locked_pairs": locked,
                         "noop_n_run": float(np.mean([r["n_run"] for r in arms["noop"]]))}

        # ---- P5 targeted intervention works ------------------------------
        best_o = max(("o03", "o06", "o10"), key=lambda k: util[k][0])
        d = ep[best_o] - ep["noop"]
        sd = float(d.std(ddof=1))
        dz = float(d.mean() / sd) if sd > 1e-12 else float("nan")
        lo, hi = boot(d)
        neg5 = 0.10 * abs(util[best_o][0] - util["noop"][0])
        v5 = V.classify(lo, hi, neg5)
        # the d_z >= 0.6 requirement only downgrades a PASS to INCONCLUSIVE,
        # it never manufactures a FAIL out of an underpowered result
        verdicts["P5_targeted_improves"] = v5 if (v5 != V.PASS or dz >= 0.6) else V.INCONCLUSIVE
        results["P5"] = {"arm": best_o, "d_z": dz, "ci": [float(lo), float(hi)]}

        # ---- P6 blind targeting worse ------------------------------------
        bb = "b10" if util["b10"][0] > util["b06"][0] else "b06"
        d_ob = ep[best_o] - ep[bb]
        lo_ob, hi_ob = boot(d_ob)
        d_bn = ep[bb] - ep["noop"]
        neg6 = 0.10 * abs(util[best_o][0] - util["noop"][0])
        v6 = V.classify(lo_ob, hi_ob, neg6)
        verdicts["P6_blind_worse"] = v6 if (v6 != V.PASS or d_bn.mean() <= 0.05) else V.INCONCLUSIVE
        results["P6"] = {"oracle_vs_blind_d": float(d_ob.mean()),
                         "ci": [float(lo_ob), float(hi_ob)],
                         "blind_minus_noop": float(d_bn.mean())}

        # ---- P7 action order matters -------------------------------------
        st = float(best_o[1:3]) / 10.0
        early = pool.run("_sched", DEV, (st, 0, 20))
        late = pool.run("_sched", DEV, (st, 60, 20))
        de = np.asarray([reward.episode_value(r, cal)[0] for r in early])
        dl = np.asarray([reward.episode_value(r, cal)[0] for r in late])
        lo7, hi7 = boot(de - dl)
        neg7 = 0.10 * abs(util[best_o][0] - util["noop"][0])
        verdicts["P7_action_order_matters"] = V.classify(lo7, hi7, neg7, two_sided=True)
        results["P7"] = {"early_minus_late": float((de - dl).mean()),
                         "ci": [float(lo7), float(hi7)]}

        # ---- P8 catastrophe controllable ---------------------------------
        verdicts["P8_catastrophe_controllable"] = V.PASS if bool(util[best_o][1] < 0.10) else V.FAIL
        results["P8"] = {"arm": best_o, "catastrophe": util[best_o][1],
                         "noop_catastrophe": util["noop"][1]}
    finally:
        pool.close()

    print(f"\n=== porting gate — {a.task} ===")
    for k, v in results.get("arms", {}).items():
        print(f"  {k:6s} utility={v['utility']:+8.4f} cat={v['catastrophe']:.3f} "
              f"path={v['pathology']:7.3f} dmg={v['damage']:6.3f} locked={v['locked']:5.2f}")
    print()
    for k, ok in verdicts.items():
        print(f"  {ok:12s}  {k}")
    tal = V.tally(verdicts)
    blocked = V.blocking(verdicts)
    allok = not blocked
    print(f"\n  PASS {tal['PASS']}  INCONCLUSIVE {tal['INCONCLUSIVE']}  FAIL {tal['FAIL']}")
    print(f"PORTING GATE: {'PASS' if allok else 'BLOCKED by ' + ', '.join(blocked)}"
          + ("  (inconclusive gates carried forward, not blocking)" if tal['INCONCLUSIVE'] else ""))
    out = ROOT / "agentic/reports/validation" / f"porting_gate_{a.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": a.task, "verdicts": verdicts, "tally": tal, "blocking": blocked,
                               "results": results, "pass": allok}, indent=2,
                              default=float) + "\n")
    print(f"wrote {out}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
