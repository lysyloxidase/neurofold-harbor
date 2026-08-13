"""Normalized episode utility for v8.

v7 combined raw terms of wildly different magnitude, which is one reason a
handful of coefficients dominated everything.  Here every component is robust-
scaled by constants derived ONLY from a public calibration split (median and
MAD of the no-op and generic distributions), so each term enters the weighted
sum on a comparable scale, and the scale of every term is recorded.

Safety alignment from v7 is deliberately preserved: damage and catastrophe
enter with negative weight and the catastrophe penalty is sized so that an
unsafe trajectory cannot outrank a safe one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COMPONENTS = ["pathology_reduction", "improvement", "path_gain", "safe_fraction",
              "damage", "action_energy", "barrier_energy"]

WEIGHTS = {"pathology_reduction": 0.34, "improvement": 0.14, "path_gain": 0.12,
           "safe_fraction": 0.20, "damage": -0.12, "action_energy": -0.04,
           "barrier_energy": -0.04}
CATASTROPHE_PENALTY = 1.25
CATASTROPHE_DAMAGE = 0.75


class DegenerateCalibration(RuntimeError):
    pass


def calibrate(rows, min_rel_scale=0.02, strict=True):
    """Robust scale constants from a calibration split (median / MAD).

    The calibration distribution MUST contain interventions, not just the no-op
    policy.  Under no-op several components are exactly constant
    (safe_fraction == 1, damage == 0, action_energy == 0), so their MAD is zero
    and every real intervention then z-scores into the clipping limit — which
    made "do nothing" the optimum and silently dominated the whole reward.
    `strict` refuses to produce such a calibration rather than let it through.
    """
    cal = {}
    degenerate = []
    for k in COMPONENTS:
        v = np.asarray([r[k] for r in rows], float)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        spread = float(np.percentile(v, 90) - np.percentile(v, 10))
        scale = max(mad, 0.5 * spread)
        ref = max(abs(med), 1.0)
        if scale < min_rel_scale * ref:
            degenerate.append((k, scale))
            scale = min_rel_scale * ref
        cal[k] = {"median": med, "scale": float(scale),
                  "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90))}
    if degenerate and strict:
        raise DegenerateCalibration(
            "calibration split has no spread in: "
            + ", ".join(f"{k} (scale {s:.2e})" for k, s in degenerate)
            + " — include interventions, not only the no-op policy")
    return cal


def calibration_policies():
    """Policy mixture used to build a non-degenerate calibration distribution."""
    def noop(env):
        return (0, 0.0, 0.0)

    def ladder_target(env, force, until):
        if env.steps >= until:
            return (0, 0.0, 0.0)
        inv = env.aux["ladder"].sum(1)
        cand = np.flatnonzero(inv > 0)
        if len(cand) == 0:
            cand = np.array([int(np.argmax(env.density))])
        return (int(cand[env.steps % len(cand)]), force, force * 0.65)

    def rand_small(env, rng):
        return (int(rng.integers(env.n)), float(rng.uniform(-0.3, 0.3)),
                float(rng.uniform(-0.3, 0.3)))

    def rand_big(env, rng):
        return (int(rng.integers(env.n)), float(rng.uniform(-1, 1)),
                float(rng.uniform(-1, 1)))

    return {
        "noop": lambda e: noop(e),
        "gentle_early": lambda e: ladder_target(e, 0.5, 24),
        "strong_early": lambda e: ladder_target(e, 0.85, 24),
        "gentle_always": lambda e: ladder_target(e, 0.5, 10 ** 9),
        "random_small": rand_small,
        "random_big": rand_big,
    }


def episode_value(row, cal):
    total = 0.0
    parts = {}
    for k, w in WEIGHTS.items():
        z = (row[k] - cal[k]["median"]) / cal[k]["scale"]
        z = float(np.clip(z, -6.0, 6.0))
        parts[k] = w * z
        total += w * z
    cat = float(row["damage"] > CATASTROPHE_DAMAGE)
    total -= CATASTROPHE_PENALTY * cat
    parts["catastrophe"] = -CATASTROPHE_PENALTY * cat
    return total, parts


def robust_utility(rows, cal):
    """Same robust functional shape as v7 so the statistics carry over."""
    vals = np.asarray([episode_value(r, cal)[0] for r in rows], float)
    cats = np.asarray([float(r["damage"] > CATASTROPHE_DAMAGE) for r in rows], float)
    u = float(0.60 * vals.mean() + 0.40 * np.quantile(vals, 0.20) - 1.50 * cats.mean())
    return u, float(cats.mean()), vals


def save(cal, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"weights": WEIGHTS, "catastrophe_penalty": CATASTROPHE_PENALTY,
         "catastrophe_damage_threshold": CATASTROPHE_DAMAGE,
         "calibration": cal}, indent=2) + "\n")


def load(path):
    return json.loads(Path(path).read_text())["calibration"]
