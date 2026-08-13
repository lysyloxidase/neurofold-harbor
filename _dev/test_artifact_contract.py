"""Red-team the artifact contract.

The verifier must reject every malformed submission with a clean error and
reward 0.0 — never a crash, never silent acceptance, and never execution of
anything the agent wrote.

Run:  python3 _dev/test_artifact_contract.py [task]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = sys.argv[1] if len(sys.argv) > 1 else "alzheimer-abeta42-v8"
sys.path.insert(0, str(ROOT / TASK / "environment"))
import policy_runtime as pr  # noqa: E402


def write(obj, raw=None):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.write(raw if raw is not None else json.dumps(obj))
    f.close()
    return f.name


def must_reject(name, obj=None, raw=None):
    path = write(obj, raw)
    try:
        pr.load_policy(path)
    except Exception as e:
        print(f"  REJECTED  {name:44s} ({type(e).__name__})")
        return True
    print(f"  ACCEPTED  {name:44s} <-- FAILURE, should have been rejected")
    return False


def must_accept(name, obj):
    path = write(obj)
    try:
        v = pr.load_policy(path)
        assert v.shape == (pr.PARAM_COUNT,)
        print(f"  accepted  {name:44s} ok")
        return True
    except Exception as e:
        print(f"  REJECTED  {name:44s} <-- FAILURE: {e}")
        return False


def main():
    good = pr.policy_to_json(pr.random_policy(0, 0.02))
    ok = []

    print(f"artifact contract red-team — {TASK}")
    print(f"  schema={pr.SCHEMA} params={pr.PARAM_COUNT} "
          f"max|w|={pr.MAX_ABS_WEIGHT} max_bytes={pr.MAX_POLICY_BYTES}\n")

    ok.append(must_accept("well-formed policy", good))

    ok.append(must_reject("not JSON at all", raw="this is not json"))
    ok.append(must_reject("JSON but not an object", raw="[1, 2, 3]"))
    ok.append(must_reject("missing schema", {k: v for k, v in good.items() if k != "schema"}))
    ok.append(must_reject("wrong schema string", dict(good, schema="neurofold-policy-v7")))
    ok.append(must_reject("wrong architecture (hidden)", dict(good, hidden=99)))
    ok.append(must_reject("wrong architecture (layers)", dict(good, layers=5)))
    ok.append(must_reject("params too few", dict(good, params=good["params"][:-1])))
    ok.append(must_reject("params too many", dict(good, params=good["params"] + [0.0])))
    ok.append(must_reject("params not a list", dict(good, params={"a": 1})))

    nan = dict(good, params=list(good["params"]))
    nan["params"][7] = float("nan")
    ok.append(must_reject("NaN weight", raw=json.dumps(nan).replace("NaN", "NaN")))
    inf = dict(good, params=list(good["params"]))
    inf["params"][7] = float("inf")
    ok.append(must_reject("inf weight", raw=json.dumps(inf).replace("Infinity", "Infinity")))

    big = dict(good, params=list(good["params"]))
    big["params"][3] = 1e6
    ok.append(must_reject("weight far above bound", big))
    edge = dict(good, params=list(good["params"]))
    edge["params"][3] = pr.MAX_ABS_WEIGHT + 1e-6
    ok.append(must_reject("weight just above bound", edge))

    ok.append(must_reject("booleans as params",
                          dict(good, params=[True] * pr.PARAM_COUNT)))
    ok.append(must_reject("strings as params",
                          dict(good, params=["0.1"] * pr.PARAM_COUNT)))
    ok.append(must_reject("nested objects as params",
                          dict(good, params=[{"__reduce__": ["os.system", ["id"]]}]
                               * pr.PARAM_COUNT)))
    ok.append(must_reject("oversize file",
                          dict(good, params=good["params"], padding="x" * (pr.MAX_POLICY_BYTES + 10))))

    at_bound = dict(good, params=list(good["params"]))
    at_bound["params"][3] = pr.MAX_ABS_WEIGHT
    ok.append(must_accept("weight exactly at bound", at_bound))

    print()
    print("ARTIFACT CONTRACT:", "PASS" if all(ok) else "FAIL")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
