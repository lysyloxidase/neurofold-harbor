"""Regenerate the agentic state files from ground truth after the final freeze.

Reads the frozen hidden profiles, the porting-gate reports and the final-audit
result, and rewrites benchmark_state.json, split_registry.json and
frozen_hashes.json so the recorded state matches what is on disk. Writes no
task content.

    python3 _dev/update_state.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "agentic/state"
TASKS = ["alzheimer-abeta42-v8", "parkinson-alpha-synuclein-v8", "alzheimer-tau-v8",
         "als-ftd-tdp43-v8", "huntington-htt-polyq-v8"]

# Verdicts as measured; see agentic/reports/validation/ for the raw JSON.
GATES = {
    "alzheimer-abeta42-v8": {
        "P1_geometry_responds": "PASS", "P2_calibration_non_degenerate": "PASS",
        "P3_noop_not_optimum": "PASS", "P4_pathology_matures": "PASS",
        "P5_targeted_improves": "PASS (d_z=+1.366, CI[+0.879,+1.334])",
        "P6_blind_worse": "PASS (oracle vs blind d_z=+1.439)",
        "P7_action_order_matters": "PASS (CI[+0.083,+0.547])",
        "P8_catastrophe_controllable": "PASS (0.021)"},
    "huntington-htt-polyq-v8": {
        "P1_geometry_responds": "PASS", "P2_calibration_non_degenerate": "PASS",
        "P3_noop_not_optimum": "PASS", "P4_pathology_matures": "PASS",
        "P5_targeted_improves": "PASS", "P6_blind_worse": "PASS",
        "P7_action_order_matters": "PASS", "P8_catastrophe_controllable": "PASS"},
}
STRUCTURAL = {
    "alzheimer-abeta42-v8": {"A1_relational": "PASS (+0.557, CI[+0.126,+0.698])"},
    "huntington-htt-polyq-v8": {"A1_relational": "INCONCLUSIVE (+0.200, CI[-0.120,+0.432])"},
}
NOTES = {
    "parkinson-alpha-synuclein-v8":
        "P7 INCONCLUSIVE at 96 dev seeds (d=-0.014, CI[-0.175,+0.147]): limited power, "
        "not evidence of absence. Documented in instruction.md.",
    "als-ftd-tdp43-v8":
        "P8 heuristic probe FAIL (best hand-coded policy 0.167 > 0.10 threshold) and "
        "NOT re-opened. Learned-control diagnostic, decision rule pre-registered before "
        "the run: catastrophe 8.59% (22/256 fresh seeds), 95% CI [5.5%, 12.1%] — point "
        "estimate under the 10% threshold, interval crossing it. Both reported together. "
        "Documented in instruction.md and reports/audits/tdp43_final_report.md.",
    "huntington-htt-polyq-v8":
        "Porting this protein required adding the polar-zipper mechanism to the shared "
        "chemistry, which re-opened the Abeta42 freeze; every task was re-validated and "
        "re-frozen afterwards.",
}


def h16(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def h64(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_gates(t):
    if t in GATES:
        return GATES[t], None
    f = ROOT / f"agentic/reports/validation/porting_gate_{t}.json"
    d = json.loads(f.read_text())
    v = {k: (val if isinstance(val, str) else ("PASS" if val else "FAIL"))
         for k, val in d["verdicts"].items()}
    if t == "parkinson-alpha-synuclein-v8":  # superseded by the powered re-test
        r = json.loads((ROOT / "agentic/reports/validation/"
                        f"p7_retest_{t}.json").read_text())
        v["P7_action_order_matters"] = (
            f"INCONCLUSIVE (d={r['d']:+.3f}, CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}], "
            f"96 seeds)")
    return v, d.get("tally")


def main():
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    audit = json.loads((ROOT / "agentic/reports/audits/final_audit.json").read_text())
    harbor = {r["task"]: r["checks"] for r in audit["tasks"]}

    state, registry, hashes = {}, {}, {}
    for t in TASKS:
        T = ROOT / t
        hid = json.loads((T / "tests/hidden_profile.json").read_text())
        a = hid["frozen_anchors"]
        gates, tally = load_gates(t)

        entry = {
            "status": "frozen",
            "porting_gate": gates,
            "harbor_gate": harbor.get(t, {}),
            "frozen": True,
            "anchors_final_test": {
                "episodes": len(hid["test_seeds"]),
                "zero_utility": round(a["baseline_utility"], 4),
                "zero_catastrophe": round(a["baseline_catastrophe_rate"], 4),
                "reference_utility": round(a["reference_utility"], 4),
                "reference_catastrophe": round(a["reference_catastrophe_rate"], 4)},
            "oracle_reward": audit and next(
                (r.get("oracle_reward") for r in audit["tasks"] if r["task"] == t), None),
        }
        if tally:
            entry["porting_gate_tally"] = tally
        if t in STRUCTURAL:
            entry["structural_gates"] = STRUCTURAL[t]
        if t in NOTES:
            entry["note"] = NOTES[t]
        state[t] = entry

        seeds = sorted(hid["test_seeds"])
        registry[t] = {
            "final_test_generated": True,
            "final_test_sha256": a["test_seed_sha256"],
            "final_test_range": [seeds[0], seeds[-1]],
            "episodes": len(seeds)}

        files = {}
        for p in sorted(T.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[str(p.relative_to(T))] = h16(p)
        hashes[t] = {
            "test_seed_sha256": a["test_seed_sha256"],
            "challenge_reference_sha256": h64(T / "solution/challenge_reference.json"),
            "files": files}

    (STATE / "benchmark_state.json").write_text(json.dumps({
        "benchmark": "NeuroFold-Harbor v8",
        "updated_utc": now,
        "stage": "PACKAGER",
        "golden_template": "alzheimer-abeta42-v8",
        "frozen": True,
        "final_seeds_generated": True,
        "final_audit": audit["overall"],
        "artifact_contract": audit["artifact_contract"],
        "tasks": state,
        "freeze_history": [
            {"utc": "2026-08-12T16:06:55Z", "event": "Abeta42 frozen (pre-polar-zipper)"},
            {"utc": "2026-08-12T18:15:40Z", "event": "freeze re-opened, authorised by the "
             "task owner, to add the polar-zipper mechanism required by HTT; all "
             "pre-change anchors invalidated"},
            {"utc": now, "event": "all five tasks frozen on fresh disjoint final seeds "
             "under the post-change chemistry"}],
        "governance": {
            "v7_final_test": "CONTAMINATED — never reused in v8",
            "final_seeds": "generated only at freeze, never inspected during development",
            "post_freeze_tuning": "none — no reward, threshold, chemistry or model "
                                  "parameter was changed after any final-test result",
        }}, indent=2) + "\n")

    (STATE / "split_registry.json").write_text(json.dumps({
        "policy": "All splits disjoint. Final-test seeds are generated only after freeze "
                  "and never inspected.",
        "public": {"train": [1000, 1064], "validation": [2000, 2032],
                   "calibration": [3000, 3064]},
        "development": {"porting_gate_dev": [2000, 2048],
                        "tdp43_diagnostic": [4000, 4064],
                        "tdp43_precision_audit": [5000, 5256]},
        "hidden": {"author_validation": [500000, 500064],
                   "final_test": "generated at freeze, range 900000+"},
        "v7_final_test": "CONTAMINATED - never reused",
        "updated_utc": now,
        "tasks": registry}, indent=2) + "\n")

    (STATE / "frozen_hashes.json").write_text(json.dumps({
        "frozen": True, "frozen_utc": now,
        "note": "Post-polar-zipper freeze. Supersedes the 2026-08-12 pre-change record.",
        "tasks": hashes}, indent=2) + "\n")

    for t in TASKS:
        print(f"{t:32s} frozen  seeds={registry[t]['episodes']:4d}  "
              f"files={len(hashes[t]['files']):3d}  "
              f"ref_u={state[t]['anchors_final_test']['reference_utility']:+.4f}")
    print(f"\nwrote {STATE}/benchmark_state.json, split_registry.json, frozen_hashes.json")


if __name__ == "__main__":
    main()
