"""Sequence chemistry for NeuroFold v8 coarse-grained beads.

Every table here is a *coarse proxy*, not a force field parameter set.
Labelling used throughout:
  [P] physically motivated   [B] biologically motivated   [S] intentionally synthetic
"""
from __future__ import annotations

import numpy as np

BEAD_SIZE = 5

# [P] Kyte-Doolittle hydropathy
AA_HYDRO = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
            'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
            'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5}
# [P] formal charge at ~pH 7
AA_CHARGE = {**{a: 0.0 for a in AA_HYDRO}, 'K': 1.0, 'R': 1.0, 'H': 0.15,
             'D': -1.0, 'E': -1.0}
# [B] Chou-Fasman-like beta propensity
AA_BETA = {'V': 1.00, 'I': 0.95, 'Y': 0.90, 'F': 0.90, 'W': 0.85, 'T': 0.72,
           'C': 0.70, 'L': 0.68, 'Q': 0.63, 'M': 0.62, 'A': 0.55, 'R': 0.50,
           'N': 0.48, 'H': 0.47, 'K': 0.45, 'S': 0.40, 'D': 0.35, 'E': 0.34,
           'G': 0.25, 'P': 0.05}
# [B] aromatic / "sticker" character (pi-stacking proxy, relevant for LCD systems)
AA_AROM = {**{a: 0.0 for a in AA_HYDRO}, 'F': 1.0, 'W': 1.0, 'Y': 0.9, 'H': 0.35}
# [B] disorder propensity
AA_DISORDER = {'P': 1.00, 'G': 0.92, 'Q': 0.86, 'S': 0.84, 'E': 0.80, 'K': 0.76,
               'N': 0.74, 'D': 0.72, 'R': 0.66, 'T': 0.60, 'A': 0.42, 'H': 0.42,
               'M': 0.35, 'C': 0.32, 'F': 0.28, 'Y': 0.27, 'L': 0.25, 'V': 0.22,
               'I': 0.20, 'W': 0.18}

# ---------------------------------------------------------------- bead classes
# [B] six coarse chemical classes; the pair matrix below is defined over these.
CLASS_NAMES = ["aliphatic", "aromatic", "polar", "cationic", "anionic", "flexible"]
K_CLASS = len(CLASS_NAMES)


def _residue_class(aa):
    if aa in "AILMVC":
        return 0
    if aa in "FWY":
        return 1
    if aa in "STNQH":
        return 2
    if aa in "KR":
        return 3
    if aa in "DE":
        return 4
    return 5           # G, P


# [S] Non-separable class interaction matrix.  Negative = attractive.
#
# This matrix is the structural core of v8.  In v7 the pair energy factorized
# into per-bead scalars, so the harm of a contact was inferable from the two
# beads' own features and mean-field aggregation sufficed — which is exactly why
# deleting the whole message block cost nothing.  Here M is deliberately NOT of
# the form u @ v.T, so a controller must represent WHICH partner a bead touches.
#
# Encoded intent:
#   aromatic-aromatic  strongly attractive (sticker/pi-stacking proxy)
#   aromatic-aliphatic only mildly attractive  <- breaks separability
#   aliphatic-aliphatic moderately attractive
#   cationic-anionic   attractive; like-charge repulsive
#   flexible (G/P)     disrupts packing with everything
M_PAIR = np.array([
    # alip   arom   polar  cat    anion  flex
    [-0.90, -0.35, -0.05, +0.05, +0.05, +0.30],   # aliphatic
    [-0.35, -1.40, -0.10, -0.25, -0.20, +0.35],   # aromatic
    [-0.05, -0.10, -0.15, -0.10, -0.10, +0.10],   # polar
    [+0.05, -0.25, -0.10, +0.95, -1.10, +0.15],   # cationic
    [+0.05, -0.20, -0.10, -1.10, +0.85, +0.15],   # anionic
    [+0.30, +0.35, +0.10, +0.15, +0.15, +0.55],   # flexible
], dtype=float)


def separability_ratio(M=M_PAIR):
    """sigma_2 / sigma_1 of the pair matrix.

    A rank-1 (separable) matrix gives 0.  A value well above 0 certifies that
    the pair energy cannot be rewritten as a product of per-bead scalars, i.e.
    that partner identity carries information no per-node feature can supply.
    """
    s = np.linalg.svd(np.asarray(M, float), compute_uv=False)
    return float(s[1] / s[0])


def bead_properties(sequence, bead_size=BEAD_SIZE):
    """Aggregate residues into beads and return per-bead property arrays."""
    n = int(np.ceil(len(sequence) / bead_size))
    hyd = np.zeros(n); chg = np.zeros(n); bet = np.zeros(n)
    aro = np.zeros(n); dis = np.zeros(n); pro = np.zeros(n); gly = np.zeros(n)
    qfr = np.zeros(n)
    cls = np.zeros(n, dtype=int)
    bounds = []
    for i in range(n):
        a, b = i * bead_size, min(len(sequence), (i + 1) * bead_size)
        frag = sequence[a:b]
        bounds.append((a, b))
        hyd[i] = np.clip((np.mean([AA_HYDRO.get(c, 0.0) for c in frag]) + 4.5) / 9.0, 0, 1)
        chg[i] = np.clip(np.sum([AA_CHARGE.get(c, 0.0) for c in frag]) / max(1, len(frag)), -1, 1)
        bet[i] = np.mean([AA_BETA.get(c, 0.4) for c in frag])
        aro[i] = np.mean([AA_AROM.get(c, 0.0) for c in frag])
        dis[i] = np.mean([AA_DISORDER.get(c, 0.5) for c in frag])
        pro[i] = frag.count('P') / max(1, len(frag))
        gly[i] = frag.count('G') / max(1, len(frag))
        # [B] glutamine fraction: substrate for side-chain hydrogen bonding
        qfr[i] = frag.count('Q') / max(1, len(frag))
        counts = np.bincount([_residue_class(c) for c in frag], minlength=K_CLASS)
        cls[i] = int(np.argmax(counts))
    return {"n": n, "bounds": bounds, "hydro": hyd, "charge": chg, "beta": bet,
            "arom": aro, "disorder": dis, "pro": pro, "gly": gly, "qfrac": qfr,
            "cls": cls}


def region_weights(profile, n, bead_size=BEAD_SIZE):
    out = {}
    for r in profile.get("regions", []):
        w = np.zeros(n)
        for i in range(n):
            a, b = i * bead_size, (i + 1) * bead_size
            overlap = max(0, min(b, int(r["end"])) - max(a, int(r["start"])))
            w[i] = float(r["weight"]) * overlap / max(1, b - a)
        out[r["name"]] = w
    return out
