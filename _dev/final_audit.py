"""Final audit of the v8 task set. Reports every check; fails loudly.

    python3 _dev/final_audit.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["alzheimer-abeta42-v8", "parkinson-alpha-synuclein-v8", "alzheimer-tau-v8",
         "als-ftd-tdp43-v8", "huntington-htt-polyq-v8"]
LAYOUT = ["instruction.md", "task.toml", "environment/Dockerfile",
          "tests/test.sh", "solution/solve.sh"]
FORBIDDEN_IN_ENV = ["test_seeds", "frozen_anchors", "challenge_reference",
                    "reference_utility", "baseline_utility"]


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800, **kw)


def audit_task(t):
    T = ROOT / t
    r = {"task": t, "checks": {}}

    # H1 layout
    missing = [f for f in LAYOUT if not (T / f).exists()]
    r["checks"]["H1_layout"] = ("PASS" if not missing else f"FAIL missing {missing}")

    # H2 task.toml
    toml = (T / "task.toml").read_text() if (T / "task.toml").exists() else ""
    ok2 = ('reward = "continuous"' in toml and 'network_mode = "no-network"' in toml
           and 'environment_mode = "separate"' in toml)
    r["checks"]["H2_task_toml"] = "PASS" if ok2 else "FAIL"

    # H7 no hidden leakage in environment/
    leaks = []
    for p in (T / "environment").rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text()
        except Exception:
            continue
        for k in FORBIDDEN_IN_ENV:
            if k in txt:
                leaks.append(f"{p.relative_to(T)}:{k}")
    for f in ("hidden_profile.json", "challenge_reference.json"):
        if (T / "environment" / f).exists():
            leaks.append(f"environment/{f}")
    r["checks"]["H7_no_leakage"] = "PASS" if not leaks else f"FAIL {leaks[:4]}"

    # H8 split disjointness
    pub = json.loads((T / "environment/profile.json").read_text())
    hid = json.loads((T / "tests/hidden_profile.json").read_text())
    tr, va = set(pub["train_seeds"]), set(pub["validation_seeds"])
    cal = set(range(3000, 3064))
    fin = set(hid["test_seeds"])
    bad = {"final∩train": tr & fin, "final∩valid": va & fin, "final∩calib": cal & fin,
           "train∩valid": tr & va}
    ok8 = not any(bad.values())
    r["checks"]["H8_splits_disjoint"] = ("PASS" if ok8
                                         else "FAIL " + str({k: len(v) for k, v in bad.items() if v}))
    r["n_final_seeds"] = len(fin)

    # H10 the agent trains on the same physics it is scored on.
    # Never checked before: neurofold8/ exists in ten copies (five tasks x
    # environment/tests) with no single source of truth, so a fix must be applied
    # ten times by hand, and any drift between environment/ and tests/ would
    # silently train an agent on different dynamics than the verifier scores.
    import hashlib as _h
    drift = []
    for f in sorted((T / "environment/neurofold8").glob("*.py")):
        other = T / "tests/neurofold8" / f.name
        if not other.exists():
            drift.append(f"{f.name}: missing in tests/")
        elif _h.sha256(f.read_bytes()).digest() != _h.sha256(other.read_bytes()).digest():
            drift.append(f.name)
    r["checks"]["H10_env_tests_identical"] = "PASS" if not drift else f"FAIL {drift}"

    # freeze integrity: seed hash + reference hash recorded in the hidden profile
    a = hid["frozen_anchors"]
    seed_h = hashlib.sha256(json.dumps(sorted(fin), separators=(",", ":")).encode()).hexdigest()
    ref_h = hashlib.sha256((T / "solution/challenge_reference.json").read_bytes()).hexdigest()
    ok_f = (seed_h == a["test_seed_sha256"]
            and ref_h == a["policy_sha256"]["challenge_reference"])
    r["checks"]["H9_freeze_hashes"] = "PASS" if ok_f else "FAIL hash mismatch"

    # H3 docker builds + H4 oracle + H5 malformed rejected
    env_img, test_img = f"nf8-{t}-env", f"nf8-{t}-tests"
    b1 = sh("docker", "build", "-q", "-t", env_img, str(T / "environment"))
    b2 = sh("docker", "build", "-q", "-t", test_img, str(T / "tests"))
    r["checks"]["H3_docker_build"] = ("PASS" if b1.returncode == 0 and b2.returncode == 0
                                      else "FAIL")
    if b2.returncode == 0:
        sh("docker", "volume", "rm", "-f", "nfaudit")
        sh("docker", "volume", "create", "nfaudit")
        sh("docker", "run", "--rm", "--network", "none", "-v", "nfaudit:/logs",
           "-v", f"{T/'solution'}:/solution:ro", test_img, "sh", "/solution/solve.sh")
        sh("docker", "run", "--rm", "--network", "none", "-v", "nfaudit:/logs",
           test_img, "bash", "/tests/test.sh")
        out = sh("docker", "run", "--rm", "-v", "nfaudit:/logs", test_img,
                 "cat", "/logs/verifier/reward.txt").stdout.strip()
        try:
            rw = float(out)
        except ValueError:
            rw = float("nan")
        r["oracle_reward"] = rw
        r["checks"]["H4_oracle_1.0"] = ("PASS" if abs(rw - 1.0) < 1e-9
                                        else f"FAIL reward={out!r}")

        # H5 malformed artifact -> reward 0.0 with an error, not a crash
        sh("docker", "volume", "rm", "-f", "nfbad")
        sh("docker", "volume", "create", "nfbad")
        p = subprocess.run(["docker", "run", "--rm", "-i", "-v", "nfbad:/logs", test_img,
                            "sh", "-c", "mkdir -p /logs/artifacts && cat > /logs/artifacts/policy.json"],
                           input='{"schema":"wrong","params":[0]}', text=True,
                           capture_output=True, timeout=300)
        sh("docker", "run", "--rm", "--network", "none", "-v", "nfbad:/logs",
           test_img, "bash", "/tests/test.sh")
        bad_out = sh("docker", "run", "--rm", "-v", "nfbad:/logs", test_img,
                     "cat", "/logs/verifier/reward.txt").stdout.strip()
        err = sh("docker", "run", "--rm", "-v", "nfbad:/logs", test_img, "python", "-c",
                 "import json;print('error' in json.load(open('/logs/verifier/metrics.json')))"
                 ).stdout.strip()
        r["checks"]["H5_malformed_rejected"] = ("PASS" if bad_out == "0" and err == "True"
                                                else f"FAIL reward={bad_out!r} err={err!r}")
    return r


def main():
    print("=== NeuroFold-Harbor v8 — final audit ===\n")
    results = []
    for t in TASKS:
        r = audit_task(t)
        results.append(r)
        st = "PASS" if all(v == "PASS" for v in r["checks"].values()) else "FAIL"
        print(f"{t:30s} {st}")
        for k, v in r["checks"].items():
            mark = " " if v == "PASS" else "!"
            print(f"  {mark} {k:24s} {v}")
        print(f"    final seeds: {r['n_final_seeds']}   oracle reward: {r.get('oracle_reward')}")
        print()

    # artifact-contract tests (schema level, shared across tasks)
    ac = sh(sys.executable, str(ROOT / "_dev/test_artifact_contract.py"),
            TASKS[0], cwd=str(ROOT))
    ac_ok = "ARTIFACT CONTRACT: PASS" in ac.stdout
    print(f"artifact contract red-team: {'PASS' if ac_ok else 'FAIL'}")

    allpass = all(v == "PASS" for r in results for v in r["checks"].values()) and ac_ok
    print(f"\nFINAL AUDIT: {'PASS' if allpass else 'FAIL'}")
    out = {"tasks": results, "artifact_contract": "PASS" if ac_ok else "FAIL",
           "overall": "PASS" if allpass else "FAIL"}
    p = ROOT / "agentic/reports/audits/final_audit.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
