"""Three-valued gate verdicts: PASS / INCONCLUSIVE / FAIL.

A confidence interval spanning zero is NOT evidence of no effect — it is usually
absence of power. Collapsing those two into FAIL is a real methodological error:
it lets a null result masquerade as a demonstrated negative, and it invites
tuning against noise.

    PASS          CI excludes zero in the hypothesised direction.
    FAIL          Either the CI excludes zero in the OPPOSITE direction
                  (evidence against the hypothesis), or the whole CI lies
                  inside a pre-declared negligible band (a genuine equivalence
                  result: the effect, if any, is too small to matter).
    INCONCLUSIVE  Anything else: the interval spans zero and is wide enough
                  that a meaningful effect has not been excluded. This blocks
                  nothing on its own; it is reported as limited power.

The negligible band must be declared BEFORE the result is seen. Here it is 10%
of the optimisation gain over the zero anchor — the same fraction A1 already
uses as its minimum interesting effect size.
"""
from __future__ import annotations

PASS, INCONCLUSIVE, FAIL = "PASS", "INCONCLUSIVE", "FAIL"


def classify(lo, hi, negligible, two_sided=False):
    """Classify a 95% CI against a negligible band.

    two_sided=True when the hypothesis is 'there is an effect in either
    direction' (e.g. action order matters), rather than 'A beats B'.
    """
    negligible = abs(float(negligible))
    if two_sided:
        if lo > 0 or hi < 0:
            return PASS
    else:
        if lo > 0:
            return PASS
        if hi < 0:
            return FAIL                       # resolved the wrong way
    if -negligible <= lo and hi <= negligible:
        return FAIL                           # equivalence: bounded below what matters
    return INCONCLUSIVE


def summarise(name, lo, hi, d, negligible, two_sided=False):
    v = classify(lo, hi, negligible, two_sided)
    note = {
        PASS: "resolved",
        FAIL: ("effect resolved in the opposite direction" if hi < 0 and not two_sided
               else f"equivalence: |effect| bounded below the negligible band {negligible:.3f}"),
        INCONCLUSIVE: f"CI spans zero and exceeds the negligible band {negligible:.3f} — "
                      "limited power, not evidence of absence",
    }[v]
    return {"gate": name, "verdict": v, "d": float(d),
            "ci95": [float(lo), float(hi)], "negligible_band": float(negligible),
            "two_sided": bool(two_sided), "note": note}


def blocking(verdicts):
    """Only FAIL blocks. INCONCLUSIVE is reported and carried forward."""
    return [k for k, v in verdicts.items() if v == FAIL]


def tally(verdicts):
    c = {PASS: 0, INCONCLUSIVE: 0, FAIL: 0}
    for v in verdicts.values():
        c[v] = c.get(v, 0) + 1
    return c
