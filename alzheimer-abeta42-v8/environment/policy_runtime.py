"""Policy artifact contract for NeuroFold-Harbor v8.

The submitted artifact is a NUMERIC policy only.  No agent-authored code is
executed by the verifier: this module parses a JSON object, validates it hard,
and returns plain NumPy arrays.

Schema
------
{
  "schema": "neurofold-graph-policy-v8",
  "hidden": 12, "msg": 12, "layers": 2,
  "params": [ <PARAM_COUNT floats> ]
}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "neurofold-graph-policy-v9.1"
MAX_ABS_WEIGHT = 30.0
MAX_POLICY_BYTES = 786432          # 768 KiB
HIDDEN, MSG, LAYERS = 12, 12, 2
# EDGE_DIM dropped 12 -> 11 in v9.1: the `ladder` membership flag was removed
# from the observation. It let a 3-parameter readout reach 95% of the full
# policy's gain, because it announced exactly what pathology counted.
NODE_DIM, EDGE_DIM, HIST_DIM, HIST_HIDDEN = 25, 11, 40, 8


def param_spec(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=HIDDEN, msg=MSG,
               layers=LAYERS, hist_dim=HIST_DIM, hist_hidden=HIST_HIDDEN):
    H, M, HH = hidden, msg, hist_hidden
    spec = [("W_node", (H, node_dim)), ("b_node", (H,)),
            ("W_hist", (HH, hist_dim)), ("b_hist", (HH,)),
            ("W_hin", (H, HH)),
            ("W_edge", (H, edge_dim)), ("b_edge", (H,))]
    for l in range(layers):
        spec += [(f"W_msg{l}", (M, 3 * H)), (f"b_msg{l}", (M,)),
                 (f"w_att{l}", (3 * H,)), (f"b_att{l}", ()),
                 (f"W_upd{l}", (H, H + M)), (f"b_upd{l}", (H,))]
    spec += [("w_edge_sel", (3 * H,)), ("b_edge_sel", ()),
             ("w_str", (3 * H,)), ("b_str", ()),
             ("w_val", (H,)), ("b_val", ())]
    return spec


def slices(spec=None):
    spec = spec or param_spec()
    out, i = {}, 0
    for k, s in spec:
        n = int(np.prod(s)) if s else 1
        out[k] = (i, i + n, s)
        i += n
    return out, i


SLICES, PARAM_COUNT = slices()


def validate_policy(obj):
    """Strict validation. Raises ValueError on anything unexpected."""
    if not isinstance(obj, dict):
        raise ValueError("policy must be a JSON object")
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"invalid schema (expected {SCHEMA!r})")
    for key, want in (("hidden", HIDDEN), ("msg", MSG), ("layers", LAYERS)):
        if int(obj.get(key, want)) != want:
            raise ValueError(f"architecture mismatch: {key}={obj.get(key)} expected {want}")
    params = obj.get("params")
    if not isinstance(params, list):
        raise ValueError("params must be a JSON array")
    if len(params) != PARAM_COUNT:
        raise ValueError(f"params has {len(params)} entries, expected {PARAM_COUNT}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in params):
        raise ValueError("params must contain only numbers")
    v = np.asarray(params, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("params must be finite (no NaN or inf)")
    mx = float(np.max(np.abs(v))) if v.size else 0.0
    if mx > MAX_ABS_WEIGHT:
        raise ValueError(f"max |weight| = {mx:.3f} exceeds bound {MAX_ABS_WEIGHT}")
    return v


def load_policy(path):
    p = Path(path)
    size = p.stat().st_size
    if size > MAX_POLICY_BYTES:
        raise ValueError(f"policy file is {size} bytes, limit {MAX_POLICY_BYTES}")
    return validate_policy(json.loads(p.read_text()))


def policy_to_json(vec):
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size != PARAM_COUNT:
        raise ValueError(f"expected {PARAM_COUNT} parameters, got {v.size}")
    return {"schema": SCHEMA, "hidden": HIDDEN, "msg": MSG, "layers": LAYERS,
            "params": [float(x) for x in v]}


def save_policy(vec, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(policy_to_json(vec)) + "\n")


def zero_policy():
    return np.zeros(PARAM_COUNT)


def random_policy(seed=0, scale=0.02):
    return np.random.default_rng(seed).normal(0.0, scale, PARAM_COUNT) / np.sqrt(HIDDEN)
