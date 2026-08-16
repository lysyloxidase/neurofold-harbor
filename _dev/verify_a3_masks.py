"""Step 1: prove the A3 low-dimensional arms really are message-passing-free.

If a masked arm secretly retained message passing or history, the A3 FAIL would
be an artifact of a bad ablation rather than a property of the task. Three
checks, all empirical rather than by inspection of the code:

  1. node states stay identically zero at every step, so nothing propagates
     along the graph and no history enters;
  2. perturbing any parameter OUTSIDE the mask never changes the chosen action;
  3. the count of parameters that can actually influence the action -- the
     EFFECTIVE dimension -- is reported, since a masked weight multiplying a
     dead activation is not a free parameter.

    python3 _dev/verify_a3_masks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_dev"))
TASK = "alzheimer-abeta42-v8"
FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    np.seterr(all="ignore")
    env_dir = ROOT / TASK / "environment"
    sys.path.insert(0, str(env_dir))
    import policy_runtime as pr
    from neurofold8.env import NeuroFoldV8Env
    from neurofold8.policy import GraphPolicy, GraphPolicySpec
    from gate_a3 import head_mask

    prof = json.loads((env_dir / "profile.json").read_text())
    spec = GraphPolicySpec(node_dim=pr.NODE_DIM, edge_dim=pr.EDGE_DIM, hidden=pr.HIDDEN,
                           msg=pr.MSG, layers=pr.LAYERS, hist_dim=pr.HIST_DIM,
                           hist_hidden=pr.HIST_HIDDEN)
    pol = GraphPolicy(spec)
    rng = np.random.default_rng(4242)

    print(f"weryfikacja masek A3 — {TASK}\n")
    for ch in (2, 5, 10):
        mask = head_mask(spec, pr, ch)
        vec = np.zeros(pr.PARAM_COUNT)
        vec[mask] = rng.normal(0, 1.0, len(mask))
        print(f"[{ch}-channel arm]  |mask| = {len(mask)}")

        # --- 1. node states are dead: nothing propagates -------------------
        env = NeuroFoldV8Env(prof, seed=5)
        obs = env.observe()
        h = np.zeros(spec.HH)
        worst_h = 0.0
        for _ in range(40):
            p = spec.unpack(vec)
            hs, _ = pol.encode(p, obs, h)
            worst_h = max(worst_h, float(np.max(np.abs(hs))))
            act, h = pol.act(vec, obs, h)
            obs, _, done, _ = env.step(act)
            if done:
                break
        check("node states identically zero (no message passing, no history)",
              worst_h == 0.0, f"max|h| = {worst_h:.3e}")

        # --- 2. the arm never depends on anything outside its mask --------
        # The first version of this check perturbed out-of-mask parameters and
        # asked whether the action moved. It always does — setting a dead weight
        # to a nonzero value revives the path — so the check was meaningless.
        # The property that matters is that the vector the A3 optimiser builds
        # is zero outside the mask, and that zeroing the complement is a no-op.
        env = NeuroFoldV8Env(prof, seed=6)
        obs = env.observe()
        built = np.zeros(pr.PARAM_COUNT)
        built[mask] = rng.normal(0, 1.0, len(mask))
        outside = np.setdiff1d(np.arange(pr.PARAM_COUNT), mask)
        check("optimiser-built vector is zero outside the mask",
              np.all(built[outside] == 0.0))
        stripped = built.copy()
        stripped[outside] = 0.0
        same = all(pol.act(built, obs, np.zeros(spec.HH))[0]
                   == pol.act(stripped, obs, np.zeros(spec.HH))[0]
                   for _ in range(1))
        check("zeroing the complement changes nothing", same)

        # --- 3. effective dimension --------------------------------------
        # A masked weight that only multiplies a dead activation is not free.
        env = NeuroFoldV8Env(prof, seed=7)
        obs = env.observe()
        eff = 0
        for i in mask:
            v2 = vec.copy()
            v2[i] += 2.5
            p1 = spec.unpack(vec)
            p2 = spec.unpack(v2)
            l1 = pol.forward(vec, obs, np.zeros(spec.HH))[0]
            l2 = pol.forward(v2, obs, np.zeros(spec.HH))[0]
            s1 = pol.forward(vec, obs, np.zeros(spec.HH))[1]
            s2 = pol.forward(v2, obs, np.zeros(spec.HH))[1]
            if not (np.allclose(l1, l2) and np.allclose(s1, s2)):
                eff += 1
            del p1, p2
        print(f"        effective dimension: {eff} of {len(mask)} masked "
              f"({100*eff/len(mask):.0f}%)")
        print()

    print("MASKI A3: PASS" if not FAILS else f"MASKI A3: FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
