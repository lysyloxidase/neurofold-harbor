"""Unit tests for the simulator itself.

v8.0 shipped with no tests below the statistical gates, so every physics defect
had to surface as a failed gate — late and expensive. These assert the
invariants directly.

    python3 _dev/test_physics.py [task]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def main(task="alzheimer-abeta42-v8"):
    np.seterr(all="ignore")
    env_dir = ROOT / task / "environment"
    sys.path.insert(0, str(env_dir))
    from neurofold8.env import NeuroFoldV8Env
    from neurofold8.policy import GraphPolicy, GraphPolicySpec
    import policy_runtime as pr

    prof = json.loads((env_dir / "profile.json").read_text())
    print(f"physics unit tests — {task}\n")

    # --- energy and observation are finite everywhere -----------------------
    bad_obs = bad_e = 0
    for s in range(20):
        env = NeuroFoldV8Env(prof, seed=s)
        obs = env.observe()
        for _ in range(96):
            for v in obs.values():
                arr = np.asarray(v, float)
                if arr.size and not np.all(np.isfinite(arr)):
                    bad_obs += 1
            if not np.isfinite(env.energy_total):
                bad_e += 1
            obs, _, done, _ = env.step((0, 0.0, 0.0))
            if done:
                break
    check("observations finite over 20 no-op episodes", bad_obs == 0, f"{bad_obs} non-finite")
    check("total energy finite over 20 no-op episodes", bad_e == 0, f"{bad_e} non-finite")

    # --- damage decreases only through the latent-gated repair term ---------
    # Damage is irreversible EXCEPT for slow repair gated by chaperone capacity
    # (env.py docstring, point 3), so the invariant is not monotonicity but a
    # bound: no step may shed more than eta_repair * chaperone_max * damage.
    eta = prof["physics"]["eta_repair"]
    chap_max = max(prof["conditions"]["chaperone"])
    env = NeuroFoldV8Env(prof, seed=11)
    env.observe()
    prev, worst = env.damage, 0.0
    while env.steps < env.max_steps and env.budget > 0:
        _, _, done, _ = env.step((3, 9, 0.7))
        drop = prev - env.damage
        if drop > 0:
            worst = max(worst, drop - eta * chap_max * prev)
        prev = env.damage
        if done:
            break
    check("damage sheds no more than the repair term allows", worst <= 1e-9,
          f"excess {worst:.3e}")
    check("damage stays within [0, damage_cap]",
          0.0 <= env.damage <= prof["physics"]["damage_cap"] + 1e-9)

    # --- pathology is non-negative -----------------------------------------
    env = NeuroFoldV8Env(prof, seed=12)
    env.observe()
    neg = False
    while env.steps < env.max_steps and env.budget > 0:
        _, _, done, _ = env.step((2, 8, 1.0))
        if env.pathology < -1e-12:
            neg = True
        if done:
            break
    check("pathology stays non-negative", not neg)

    # --- a self-pair action is inert ---------------------------------------
    a = NeuroFoldV8Env(prof, seed=13)
    a.observe()
    b = NeuroFoldV8Env(prof, seed=13)
    b.observe()
    for _ in range(20):
        a.step((5, 5, 1.0))
        b.step((0, 0, 0.0))
    check("action on (i,i) is inert", abs(a.energy_total - b.energy_total) < 1e-9,
          f"{a.energy_total} vs {b.energy_total}")

    # --- modulation decays at the documented rate --------------------------
    env = NeuroFoldV8Env(prof, seed=14)
    env.observe()
    i, j = 2, 10
    env.step((i, j, 1.0))
    m0 = float(env.mod[i, j])
    env.step((0, 0, 0.0))
    m1 = float(env.mod[i, j])
    check("modulation decays by mod_decay",
          abs(m1 - m0 * prof["physics"]["mod_decay"]) < 1e-9, f"{m1} vs {m0}*decay")
    check("modulation matrix stays symmetric",
          np.allclose(env.mod, env.mod.T))

    # --- v9 coupling: a blocked rung is not counted as a rung --------------
    blk = prof["physics"].get("ladder_mod_block", 0.0)
    if blk:
        env = NeuroFoldV8Env(prof, seed=15)
        env.observe()
        for _ in range(30):
            env.step((0, 0, 0.0))
        lad = env.aux["ladder"]
        ii, jj = np.nonzero(lad)
        if len(ii):
            i, j = int(ii[0]), int(jj[0])
            before = bool(env.aux["ladder"][i, j])
            env.mod[i, j] = env.mod[j, i] = min(0.98, blk + 0.4)
            env.step((0, 0, 0.0))
            after = bool(env.aux["ladder"][i, j])
            check("blocked rung leaves the ladder", before and not after,
                  f"before={before} after={after}")
        else:
            check("blocked rung leaves the ladder", False, "no ladder formed to test")
        # and the converse: an unblocked rung is unaffected by the new term
        env2 = NeuroFoldV8Env(prof, seed=15)
        env2.observe()
        for _ in range(30):
            env2.step((0, 0, 0.0))
        check("unblocked rungs unaffected", int(env2.aux["ladder"].sum()) > 0)

    # --- the policy survives an empty contact graph ------------------------
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)

    pol = GraphPolicy(spec)
    env = NeuroFoldV8Env(prof, seed=16)
    obs = env.observe()
    empty = copy.deepcopy(obs)
    empty["edge"] = np.zeros((0, pr.EDGE_DIM))
    empty["edge_index"] = np.zeros((2, 0), dtype=int)
    try:
        act, _ = pol.act(pr.zero_policy(), empty, np.zeros(spec.HH))
        check("policy handles an empty contact graph", act == (0, 0, 0.0), str(act))
    except Exception as exc:
        check("policy handles an empty contact graph", False, f"{type(exc).__name__}: {exc}")

    # --- determinism: same seed, same trajectory ---------------------------
    def roll(seed):
        e = NeuroFoldV8Env(prof, seed=seed)
        e.observe()
        for t in range(40):
            e.step((t % 5, 5 + t % 4, 0.5))
        return e.energy_total, e.pathology, e.damage
    check("same seed reproduces the trajectory", roll(17) == roll(17))

    print(f"\n{'PHYSICS TESTS: PASS' if not FAILS else 'PHYSICS TESTS: FAIL ' + str(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "alzheimer-abeta42-v8"))
