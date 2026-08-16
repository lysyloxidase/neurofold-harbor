"""External-validity test: does the model reproduce known qualitative orderings?

Every previous audit of this project measured the benchmark against itself. This
one measures it against the aggregation literature. It needs no experimental
data, only facts that are not in dispute, encoded as ORDERING predictions that
the model can fail.

Predictions are fixed here, before the run, with their direction and their
rationale. A failure is reported as a failure.

  P-A  sequence order matters, not just composition
       A shuffled sequence with identical amino-acid composition must aggregate
       LESS than the ordered one. Amyloid nucleation depends on specific
       segments; if the model scores composition alone it is not modelling
       sequence-specific aggregation at all. This is the strongest test here
       because length and composition are held exactly constant.

  P-B  Abeta42 > C-terminal hydrophobicity knockout
       The two extra C-terminal residues (I41, A42) make Abeta42 markedly more
       aggregation-prone than Abeta40; one of the most robust facts in the
       amyloid literature. The naive comparison Abeta42 vs Abeta40 is CONFOUNDED
       here: 42 residues give 9 beads and 40 give 8, and pathology is extensive,
       so part of any difference is system size rather than sequence. The
       controlled form keeps all 42 residues and all 9 beads and replaces only
       I41/A42 with glycine-serine, isolating the C-terminal hydrophobic
       contribution. The confounded comparison is still reported, labelled.

  P-C  polyQ36 > polyQ20, at constant chain length
       The pathogenic repeat threshold is around 36-40. Q20 is normal-range.
       Chain length is held constant by padding with glycine so that only the
       tract length varies -- otherwise the comparison is confounded by system
       size, since pathology is extensive.

  P-D  PHF6 intact > PHF6 with a beta-breaking proline
       VQIVYK is necessary for tau filament formation. Substituting a proline
       into the hexapeptide (VQIVYK -> VQPVYK) is the textbook way to abolish
       beta-structure locally.

Verdicts are three-valued, matching every other gate here: a CI spanning zero is
absence of power, not evidence that the model is composition-blind.

    python3 _dev/test_biological_ordering.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_dev"))
import verdict as V  # noqa: E402

SEEDS = list(range(8000, 8064))     # disjoint from every split used anywhere else
BASE_TASK = "alzheimer-abeta42-v8"


def variant_profile(base, sequence, tag):
    """A profile identical to `base` except for the sequence under test.

    Nothing else is touched: same physics, same regions, same conditions. Any
    difference in outcome is attributable to the sequence.
    """
    d = json.loads(json.dumps(base))
    d["sequence"] = sequence
    d["slug"] = f"{base['slug']}-{tag}"
    return d


def run(profile, seeds, Env, reward, cal):
    rows = []
    for s in seeds:
        e = Env(profile, seed=int(s))
        e.observe()
        while e.steps < e.max_steps and e.budget > 0:
            _, _, done, _ = e.step((0, 0.0, 0.0))
            if done:
                break
        rows.append(e.summary())
    return rows


def paired_ci(a, b, n=20000, seed=17):
    """Bootstrap CI on the paired difference a - b (same seeds, CRN)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    np.seterr(all="ignore")
    env_dir = ROOT / BASE_TASK / "environment"
    sys.path.insert(0, str(env_dir))
    from neurofold8 import reward
    from neurofold8.env import NeuroFoldV8Env

    base = json.loads((env_dir / "profile.json").read_text())
    cal = json.loads((env_dir / "reward_calibration.json").read_text())["calibration"]
    ab42 = base["sequence"]
    ab40 = ab42[:-2]                                   # drop I41, A42 — CONFOUNDED, 8 beads
    ab42_ko = ab42[:-2] + "GS"                         # same 42 residues, same 9 beads

    rng = np.random.default_rng(99)
    scrambled = "".join(rng.permutation(list(ab42)))   # identical composition

    htt = json.loads((ROOT / "huntington-htt-polyq-v8/environment/profile.json").read_text())
    n17, tail = htt["sequence"][:17], htt["sequence"][53:]
    q36 = n17 + "Q" * 36 + tail
    q20 = n17 + "Q" * 20 + "G" * 16 + tail             # same length, shorter tract

    tau = json.loads((ROOT / "alzheimer-tau-v8/environment/profile.json").read_text())
    phf6_wt = tau["sequence"]
    phf6_mut = phf6_wt.replace("VQIVYK", "VQPVYK")     # beta-breaking proline

    cases = [
        ("P-A  order vs composition", base, ab42, scrambled, "ordered", "scrambled"),
        ("P-B  Abeta42 vs C-term knockout", base, ab42, ab42_ko, "Abeta42", "I41A42>GS"),
        ("P-B* Abeta42 vs Abeta40 (CONFOUNDED: 9 vs 8 beads)",
         base, ab42, ab40, "Abeta42", "Abeta40"),
        ("P-C  polyQ36 vs polyQ20", htt, q36, q20, "Q36", "Q20"),
        ("P-D  PHF6 vs PHF6-proline", tau, phf6_wt, phf6_mut, "PHF6", "VQPVYK"),
    ]

    print("test trafnosci zewnetrznej — czy model odtwarza znane uporzadkowania\n")
    print(f"  {len(SEEDS)} seedow, polityka no-op, ta sama fizyka, rozni sie tylko sekwencja")
    print(f"  predykcja w kazdym przypadku: pierwszy wariant agreguje WIECEJ\n")
    out = {"seeds": [SEEDS[0], SEEDS[-1]], "n_seeds": len(SEEDS), "cases": {}}
    tally = {V.PASS: 0, V.INCONCLUSIVE: 0, V.FAIL: 0}

    for name, host, seq_hi, seq_lo, lab_hi, lab_lo in cases:
        ph, pl = [], []
        for seq, acc in ((seq_hi, ph), (seq_lo, pl)):
            prof = variant_profile(host, seq, "var")
            for r in run(prof, SEEDS, NeuroFoldV8Env, reward, cal):
                acc.append(r["final_pathology"])
        d, lo, hi = paired_ci(ph, pl)
        # negligible band: 10% of the higher arm's mean, the convention used by
        # the porting gate throughout this project
        neg = 0.10 * abs(float(np.mean(ph))) if np.mean(ph) else 0.05
        verd = V.classify(lo, hi, neg)
        if not name.startswith("P-B*"):
            tally[verd] += 1
        from neurofold8 import chem as _chem
        out["cases"][name] = {"high": lab_hi, "low": lab_lo,
                              "beads_high": int(_chem.bead_properties(seq_hi)["n"]),
                              "beads_low": int(_chem.bead_properties(seq_lo)["n"]),
                              "pathology_high": float(np.mean(ph)),
                              "pathology_low": float(np.mean(pl)),
                              "delta": d, "ci95": [lo, hi],
                              "negligible_band": neg, "verdict": verd}
        print(f"{name}")
        print(f"    {lab_hi:>10s} {np.mean(ph):7.3f}   {lab_lo:>10s} {np.mean(pl):7.3f}")
        print(f"    roznica {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   -> {verd}")
        print()

    ok = tally[V.PASS]
    print(f"PODSUMOWANIE: {ok} PASS, {tally[V.INCONCLUSIVE]} INCONCLUSIVE, "
          f"{tally[V.FAIL]} FAIL  (z {len(cases)-1}; P-B* jest skonfundowany i nie liczy sie)")
    out["tally"] = {k: v for k, v in tally.items()}
    out["note"] = ("Predictions were fixed before the run. A FAIL means the model does not "
                   "reproduce an ordering that is not in dispute in the literature, and is "
                   "reported as such.")
    p = ROOT / "agentic/reports/validation/biological_ordering.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
