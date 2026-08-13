"""Cartesian chain geometry and local moves.

Beads live in R^3.  Control acts through two local moves with distinct geometric
meaning, both of which provably change the 3-D configuration:

  crankshaft  rotate bead i about the axis (x_{i-1}, x_{i+1});
              preserves both adjacent bond lengths exactly, changes angles and
              dihedrals.                                                  [P]
  displace    move bead i along the local normal; changes bond lengths and is
              therefore penalised by the bond term.                       [P]

Cartesian representation is deliberate: in v7 the chain was rebuilt from
internal coordinates with a fixed bond length, so bond stretching was not a
degree of freedom at all.
"""
from __future__ import annotations

import numpy as np


def _norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n if n > eps else eps)


def init_chain(rng, n, b0, jitter=0.35, start=None, direction=None):
    """Self-avoiding-ish random walk with roughly correct bond lengths."""
    x = np.zeros((n, 3))
    if start is not None:
        x[0] = start
    d = _norm(np.asarray(direction, float)) if direction is not None else _norm(rng.normal(size=3))
    for i in range(1, n):
        d = _norm(d + jitter * rng.normal(size=3))
        x[i] = x[i - 1] + b0 * d
    return x


def init_chain_ideal(rng, n, b0, theta0, start=None, axis=None, dihedral_spread=np.pi):
    """Chain with correct bond lengths and bond angles, random dihedrals.

    Starting from a random walk left the angle term dominating the energy, so the
    cheapest available "improvement" was relaxing an artefact of initialization
    rather than doing anything conformationally meaningful.  Building the chain
    at the angular minimum removes that shortcut: at reset the bonded terms are
    near zero and essentially all controllable energy is in the interaction
    terms the task is actually about.
    """
    x = np.zeros((n, 3))
    x[0] = np.zeros(3) if start is None else np.asarray(start, float)
    e1 = _norm(np.asarray(axis, float)) if axis is not None else _norm(rng.normal(size=3))
    if n == 1:
        return x
    x[1] = x[0] + b0 * e1
    helper = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = _norm(np.cross(e1, helper))
    prev_dir = e1
    ref = e2
    for i in range(2, n):
        # rotate the previous bond direction by (pi - theta0) about a random
        # dihedral axis, giving the requested bond angle exactly
        psi = rng.uniform(-dihedral_spread, dihedral_spread)
        axis_v = _norm(np.cos(psi) * ref + np.sin(psi) * np.cross(prev_dir, ref))
        ang = np.pi - theta0
        c, s = np.cos(ang), np.sin(ang)
        d = (prev_dir * c + np.cross(axis_v, prev_dir) * s
             + axis_v * np.dot(axis_v, prev_dir) * (1 - c))
        d = _norm(d)
        x[i] = x[i - 1] + b0 * d
        ref = _norm(np.cross(d, prev_dir)) if np.linalg.norm(np.cross(d, prev_dir)) > 1e-8 else ref
        prev_dir = d
    return x


def bond_lengths(x, chain_id):
    same = chain_id[:-1] == chain_id[1:]
    d = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return d, same


def local_directions(x, chain_id):
    """Unit chain-direction at each bead, used for the orientation descriptor."""
    n = len(x)
    u = np.zeros((n, 3))
    for i in range(n):
        lo = i - 1 if i > 0 and chain_id[i - 1] == chain_id[i] else i
        hi = i + 1 if i < n - 1 and chain_id[i + 1] == chain_id[i] else i
        v = x[hi] - x[lo]
        u[i] = _norm(v) if np.linalg.norm(v) > 1e-9 else np.array([1.0, 0.0, 0.0])
    return u


def seeded_antiparallel_pair(rng, n1, b0, gap, register_len, splay=1.15,
                             jitter=0.06, extend_noise=0.10, theta0=2.5):
    """Two chains with a PARTIAL antiparallel register already present.

    `jitter` and `extend_noise` are exogenous nuisance variance and are narrowed
    in the low-variance diagnostic tier.  The register OFFSET (which beads pair
    with which) is deliberately left varying, because that is the relational
    difficulty the benchmark is meant to test.

    Waiting for spontaneous beta nucleation inside a ~100-step episode is the
    wrong model: amyloid nucleation is genuinely rare and slow, and a direct
    probe showed it essentially never happens here (0/10 episodes reached a
    nucleus).  Instead the agent is handed an early-stage nucleation event and
    must decide whether to dissolve it *before* it matures and locks.

    Chain A runs along +x.  Chain B is placed antiparallel alongside it, but
    only `register_len` consecutive pairs are held at contact distance; the rest
    of chain B is splayed away, so the register is partial and extendable.

    Returns (xA, xB, registered_pairs) with pairs given as (i_in_A, j_in_B).
    """
    # Both strands are built AT the equilibrium bond angle.  Laying them out as
    # perfectly straight lines (bond angle pi) baked a permanent angular strain
    # into every seeded episode, which any random perturbation could relieve —
    # so blind flailing scored as "repair".  A beta strand is pleated, not
    # straight.
    ex = np.array([1.0, 0.0, 0.0])
    xA = init_chain_ideal(rng, n1, b0, theta0, start=np.zeros(3), axis=ex,
                          dihedral_spread=0.12)
    xA += extend_noise * rng.normal(size=xA.shape)

    base = init_chain_ideal(rng, n1, b0, theta0, start=np.zeros(3), axis=ex,
                            dihedral_spread=0.12)
    # Antiparallel in index space: B bead j pairs with A bead (n1-1-j).
    # Reversing the index order is a relabelling, so bond lengths and bond
    # angles are preserved exactly.
    xB = base[::-1].copy()

    start = int(rng.integers(0, max(1, n1 - register_len + 1)))
    keep = set(range(start, start + register_len))
    a_idx = sorted(keep)
    b_idx = [n1 - 1 - i for i in a_idx]
    xB += xA[a_idx].mean(0) - xB[b_idx].mean(0) + np.array([0.0, gap, 0.0])

    # Splay the unregistered tails by RIGID rotation about a hinge bead: the
    # tail keeps its internal geometry and only the hinge angle changes.
    lo_h, hi_h = min(b_idx), max(b_idx)
    for tail, sign in ((list(range(0, lo_h)), -1.0),
                       (list(range(hi_h + 1, n1)), +1.0)):
        if not tail:
            continue
        hinge = lo_h if sign < 0 else hi_h
        phi = sign * splay * (0.5 + 0.5 * rng.random())
        c, s = np.cos(phi), np.sin(phi)
        for j in tail:
            v = xB[j] - xB[hinge]
            xB[j] = xB[hinge] + (v * c + np.cross(ex, v) * s
                                 + ex * np.dot(ex, v) * (1 - c))
    xB += jitter * rng.normal(size=xB.shape)
    pairs = [(i, n1 - 1 - i + n1) for i in keep]     # B index offset by n1
    return xA, xB, pairs


def neighbors_in_chain(i, chain_id):
    prev = i - 1 if i > 0 and chain_id[i - 1] == chain_id[i] else None
    nxt = i + 1 if i < len(chain_id) - 1 and chain_id[i + 1] == chain_id[i] else None
    return prev, nxt


def crankshaft(x, chain_id, i, angle):
    """Rotate bead i about the (x_{i-1}, x_{i+1}) axis by `angle` radians."""
    prev, nxt = neighbors_in_chain(i, chain_id)
    y = x.copy()
    if prev is None or nxt is None:
        # terminal bead: rotate about the single bond axis through its neighbour
        anchor = nxt if prev is None else prev
        if anchor is None:
            return y
        axis_pt = x[anchor]
        ref = x[anchor] - x[i]
        helper = np.array([0.0, 0.0, 1.0]) if abs(ref[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _norm(np.cross(ref, helper))
        p = x[i] - axis_pt
    else:
        axis_pt = x[prev]
        axis = _norm(x[nxt] - x[prev])
        p = x[i] - axis_pt
    # Rodrigues rotation
    c, s = np.cos(angle), np.sin(angle)
    p_rot = p * c + np.cross(axis, p) * s + axis * np.dot(axis, p) * (1 - c)
    y[i] = axis_pt + p_rot
    return y


def displace(x, chain_id, i, amount):
    """Move bead i along its local normal (perpendicular to the chain tangent)."""
    prev, nxt = neighbors_in_chain(i, chain_id)
    y = x.copy()
    if prev is not None and nxt is not None:
        tangent = _norm(x[nxt] - x[prev])
        mid = 0.5 * (x[prev] + x[nxt])
        radial = x[i] - mid
        if np.linalg.norm(radial) < 1e-6:
            helper = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            radial = np.cross(tangent, helper)
        n_dir = _norm(radial - np.dot(radial, tangent) * tangent)
    else:
        anchor = nxt if prev is None else prev
        if anchor is None:
            return y
        tangent = _norm(x[i] - x[anchor])
        helper = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n_dir = _norm(np.cross(tangent, helper))
    y[i] = x[i] + amount * n_dir
    return y


def apply_action(x, chain_id, i, angle, amount):
    return displace(crankshaft(x, chain_id, i, angle), chain_id, i, amount)


def pair_distances(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)


def radius_of_gyration(x):
    c = x.mean(0)
    return float(np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1))))
