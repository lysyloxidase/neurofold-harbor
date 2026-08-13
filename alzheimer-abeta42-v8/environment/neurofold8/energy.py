"""Coarse-grained energy model.

E = E_bond + E_angle + E_tors + E_excl + E_pair + E_hb + E_coop

Labels: [P] physically motivated, [B] biologically motivated,
        [S] intentionally synthetic functional form.

The two terms that carry the structural intent of v8:

  E_pair  contains a NON-SEPARABLE class interaction matrix M, so the energetic
          consequence of a contact cannot be recovered from the two beads' own
          per-node features.  Partner identity matters.                    [S/B]

  E_hb    is directional: only chain segments that are roughly (anti)parallel
          gain from a beta-compatible contact, so orientation matters.       [P]
"""
from __future__ import annotations

import numpy as np

from . import chem, geom


class EnergyModel:
    def __init__(self, props, chain_id, params):
        self.p = params
        self.chain_id = np.asarray(chain_id, int)
        n = len(self.chain_id)
        self.n = n
        self.hydro = props["hydro"]
        self.charge = props["charge"]
        self.beta = props["beta"]
        self.arom = props["arom"]
        self.qfrac = props.get("qfrac", np.zeros(n))
        self.cls = props["cls"]

        idx = np.arange(n)
        same_chain = self.chain_id[:, None] == self.chain_id[None, :]
        sep = np.abs(idx[:, None] - idx[None, :])
        # nonbonded set: |i-j|>2 within a chain, or any cross-chain pair
        self.nb_mask = (same_chain & (sep > 2)) | (~same_chain)
        np.fill_diagonal(self.nb_mask, False)
        self.same_chain = same_chain
        self.sep = np.where(same_chain, sep, 999)

        # precomputed pair chemistry (invariant across the episode)
        self.hh = np.outer(self.hydro, self.hydro)
        self.qq = np.outer(self.charge, self.charge)
        self.aa = np.outer(self.arom, self.arom)
        self.bb = np.outer(self.beta, self.beta)
        self.qq_zip = np.outer(self.qfrac, self.qfrac)
        self.MM = chem.M_PAIR[np.ix_(self.cls, self.cls)]

        # bonded index sets (within chain only)
        self.bond_i = np.array([i for i in range(n - 1)
                                if self.chain_id[i] == self.chain_id[i + 1]], dtype=int)
        self.ang_i = np.array([i for i in range(1, n - 1)
                               if self.chain_id[i - 1] == self.chain_id[i] == self.chain_id[i + 1]],
                              dtype=int)
        self.tor_i = np.array([i for i in range(1, n - 2)
                               if len(set(self.chain_id[i - 1:i + 3])) == 1], dtype=int)

    # ---------------------------------------------------------------- pieces
    def contact_weight(self, dist):
        """Smooth switching function w(r) in [0,1].                       [P]"""
        p = self.p
        return 1.0 / (1.0 + np.exp(p["contact_sharpness"] * (dist - p["contact_cutoff"])))

    def bonded(self, x):
        p = self.p
        e_bond = e_ang = e_tor = 0.0
        if len(self.bond_i):
            d = np.linalg.norm(x[self.bond_i + 1] - x[self.bond_i], axis=1)
            e_bond = p["k_bond"] * float(np.sum((d - p["b0"]) ** 2))
        if len(self.ang_i):
            a = x[self.ang_i - 1] - x[self.ang_i]
            b = x[self.ang_i + 1] - x[self.ang_i]
            ca = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            e_ang = p["k_angle"] * float(np.sum((ca - np.cos(p["theta0"])) ** 2))
        if len(self.tor_i):
            i = self.tor_i
            b1 = x[i] - x[i - 1]
            b2 = x[i + 1] - x[i]
            b3 = x[i + 2] - x[i + 1]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            m = np.cross(n1, b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-12))
            phi = np.arctan2(np.sum(m * n2, axis=1), np.sum(n1 * n2, axis=1))
            e_tor = p["k_tors"] * float(np.sum(1.0 + np.cos(3 * phi - p["phi0"])))
        return e_bond, e_ang, e_tor

    def nonbonded(self, x, dist, w, u, screening, crowding=1.0, mod=None):
        """Returns (E_excl, E_pair, E_hb, align, contact) with contact = w*mask.

        `mod[i,j]` in [0,1) is a contact-selective suppression of the ATTRACTIVE
        interaction of that specific pair — a coarse proxy for a small molecule,
        chaperone-like interaction or local solvation/screening change that
        destabilises one particular contact.  It is not a mechanical force and is
        never described as one.  Repulsive terms are left untouched.
        """
        p = self.p
        mask = self.nb_mask
        contact = w * mask
        att = 1.0 if mod is None else (1.0 - mod)

        overlap = np.maximum(0.0, p["sigma_excl"] - dist) * mask
        e_excl = p["eps_excl"] * float(np.sum(overlap ** 2) / 2.0)

        pair_chem = (p["lam_hydro"] * self.hh
                     + p["lam_elec"] * self.qq / max(0.4, screening)
                     + p["lam_arom"] * self.aa
                     + p["lam_class"] * self.MM)
        # suppress only the attractive part of the modulated pairs
        attract = np.minimum(pair_chem, 0.0) * att
        repel = np.maximum(pair_chem, 0.0)
        e_pair = float(np.sum(contact * (attract + repel)) / 2.0)

        # [B] Macromolecular-crowding depletion attraction: a long-ranged,
        # crowding-scaled well that pulls chains together.  This is what makes
        # the UNCONTROLLED system aggregate, so the agent has a disease process
        # to counteract rather than an inert box.  Long-ranged (uses a wider
        # switching radius than the contact kernel) so association can start
        # before any contact exists.
        if p.get("lam_crowd", 0.0):
            wide = 1.0 / (1.0 + np.exp(p["crowd_sharp"] * (dist - p["crowd_range"])))
            drive = wide * mask * (self.hh + p.get("crowd_arom", 0.0) * self.aa)
            e_pair += -p["lam_crowd"] * crowding * float(np.sum(drive) / 2.0)

        # Directional AND SATURABLE beta-compatibility.                     [P]
        #
        # |cos| between local chain directions gives the directional part.  The
        # saturation is essential: backbone hydrogen bonding is limited to a
        # small number of partners per residue, so a dense globule gains nothing
        # from having eight neighbours.  Without the cap the lowest-energy state
        # is always a collapsed blob (every pair contributes), sheets never form,
        # and the nucleation mechanism is dead - which is exactly what the first
        # tuning sweep showed across the entire parameter grid.
        align = np.abs(u @ u.T)
        strength = contact * self.bb * align ** p["hb_sharp"] * att
        kmax = int(p.get("hb_max_partners", 2))
        if kmax < self.n - 1:
            part = np.partition(strength, -kmax, axis=1)[:, -kmax:]
            e_hb = -p["eps_hb"] * float(np.sum(part) / 2.0)
        else:
            e_hb = -p["eps_hb"] * float(np.sum(strength) / 2.0)
        # [B] Polar zipper: glutamine SIDE-CHAIN hydrogen bonding.
        #
        # polyQ aggregation is driven by inter-strand side-chain H-bond ladders
        # between glutamines (the polar-zipper picture), not by the hydrophobic
        # or aromatic terms.  Without this, Q maps to the generic `polar` class
        # (M[2,2] = -0.15 vs -0.90 for an aliphatic core, 6x weaker) and a polyQ
        # register does not persist: measured 1.8 ladder pairs held at once
        # versus 7.2 for Abeta42.  Directional and saturable for the same reason
        # the backbone term is.
        if p.get("eps_zip", 0.0):
            zs = contact * self.qq_zip * align ** p.get("zip_sharp", 2.0) * att
            kz = int(p.get("zip_max_partners", 2))
            if kz < self.n - 1:
                e_hb += -p["eps_zip"] * float(np.sum(
                    np.partition(zs, -kz, axis=1)[:, -kz:]) / 2.0)
            else:
                e_hb += -p["eps_zip"] * float(np.sum(zs) / 2.0)
        return e_excl, e_pair, e_hb, align, contact

    # ------------------------------------------------------------ nucleation
    def beta_ladder(self, contact, align, dist, mod=None):
        """Boolean matrix of beta-compatible, aligned, close pairs.        [B]

        A rung of a beta ladder *is* its inter-strand backbone hydrogen bond.
        When that interaction is blocked the rung does not hold, even while the
        two beads stay close and aligned: proximity alone is not a ladder.

        Without this coupling the agent's only lever (contact-selective energy
        modulation) acts on the energy while the order parameter reads pure
        geometry, and the two are causally disconnected.  Measured on v8.0,
        public validation split: suppressing the ENTIRE ladder at full strength
        every step with no decay moved pathology by at most -0.04 against no-op,
        and under the shipped physics it moved it the wrong way (+0.045).
        Widening the action to whole-ladder reach made it worse, not better
        (+0.5708 -> +0.5805).  The defect was structural, not parametric.

        `ladder_mod_block` is fixed by rule, not fitted: a rung counts as
        blocked once its attractive interaction is suppressed by more than half.
        """
        p = self.p
        lad = ((contact > p["ladder_contact"]) & (align > p["ladder_align"])
               & (self.bb > p["ladder_beta"]) & (dist < p["contact_cutoff"] * 1.05))
        blk = p.get("ladder_mod_block", 0.0)
        if mod is not None and blk > 0.0:
            lad = lad & (mod < blk)
        return lad

    @staticmethod
    def enumerate_runs(L):
        """Every maximal contiguous registry run, as a list of (i, j) pairs. [B]

        A beta nucleus is a *contiguous* stretch of rungs on one register, not a
        bag of pairs. Enumerating them makes it possible to ask which nucleus a
        rung belongs to and where inside it that rung sits -- the information a
        per-edge readout cannot recover, because a rung's own features are
        identical whether it sits at the end of a run or in its middle.

        Parallel registry follows (i+k, j+k); antiparallel follows (i+k, j-k).
        """
        n = L.shape[0]
        out = []
        for di, dj in ((1, 1), (1, -1)):
            seen = set()
            for i in range(n):
                for j in range(n):
                    if not L[i, j] or (i, j) in seen:
                        continue
                    pi, pj = i - di, j - dj
                    if 0 <= pi < n and 0 <= pj < n and L[pi, pj]:
                        continue          # not the start of its run
                    run, a, b = [], i, j
                    while 0 <= a < n and 0 <= b < n and L[a, b]:
                        run.append((a, b))
                        seen.add((a, b))
                        a += di
                        b += dj
                    out.append(run)
        return out

    @staticmethod
    def longest_run(L):
        """Longest ladder-like registry run in the boolean pair matrix.

        Parallel registry follows (i+k, j+k); antiparallel follows (i+k, j-k).
        The length of the longest such run is the nucleus size.            [B]
        """
        n = L.shape[0]
        best = 0
        for di, dj in ((1, 1), (1, -1)):
            run = np.zeros_like(L, dtype=int)
            order = range(n) if di > 0 else range(n - 1, -1, -1)
            for i in order:
                for j in range(n):
                    if not L[i, j]:
                        continue
                    pi, pj = i - di, j - dj
                    prev = run[pi, pj] if (0 <= pi < n and 0 <= pj < n) else 0
                    run[i, j] = prev + 1
                    if run[i, j] > best:
                        best = run[i, j]
        return int(best)

    def cooperative(self, n_run):
        p = self.p
        return -p["eps_coop"] * max(0.0, n_run - p["L_min"] + 1) ** p["coop_gamma"]

    def confinement(self, x, crowding=1.0):
        """[B] Crowded-compartment confinement.

        Without it the two chains simply diffuse apart and never associate, so
        no inter-chain mechanism can engage.  Physically this stands in for a
        crowded cellular interaction volume rather than an infinite dilute box.
        """
        p = self.p
        if not p.get("k_conf", 0.0):
            return 0.0
        c = x.mean(0)
        r = np.linalg.norm(x - c, axis=1)
        R = p["conf_radius"] / max(0.5, crowding)
        return p["k_conf"] * float(np.sum(np.maximum(0.0, r - R) ** 2))

    # ------------------------------------------------------------------ total
    def total(self, x, screening, n_run=None, crowding=1.0, mod=None):
        dist = geom.pair_distances(x)
        u = geom.local_directions(x, self.chain_id)
        w = self.contact_weight(dist)
        e_bond, e_ang, e_tor = self.bonded(x)
        e_excl, e_pair, e_hb, align, contact = self.nonbonded(
            x, dist, w, u, screening, crowding, mod)
        ladder = self.beta_ladder(contact, align, dist, mod)
        run = self.longest_run(ladder) if n_run is None else n_run
        e_coop = self.cooperative(run)
        e_conf = self.confinement(x, crowding)
        terms = {"bond": e_bond, "angle": e_ang, "torsion": e_tor, "excluded": e_excl,
                 "pair": e_pair, "hbond": e_hb, "cooperative": e_coop,
                 "confinement": e_conf}
        return (sum(terms.values()), terms,
                {"dist": dist, "contact": contact, "align": align, "u": u,
                 "ladder": ladder, "n_run": run})
