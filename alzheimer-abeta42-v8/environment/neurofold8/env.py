"""NeuroFold v8 environment.

Coarse stochastic conformational-control environment.  NOT molecular dynamics:
transitions use a Metropolis-like acceptance rule on a coarse energy model with
an explicit, state-dependent barrier.  This is called a coarse stochastic
transition model throughout, never "kinetics" in the molecular sense.

Sequential structure (the point of v8) comes from four coupled mechanisms:

  1. maturation   contacts that persist in a beta ladder for tau_mat steps LOCK
  2. hysteresis   breaking a locked contact costs an extra barrier, and the lock
                  only releases after the contact has been broken tau_reset steps
  3. damage       irreversible except for slow repair gated by a LATENT variable
  4. budget       actions cost budget; expensive sites cost more

Consequence: the same multiset of actions in a different order gives a different
outcome, because whether a nucleus had time to mature depends on when it was
attacked.
"""
from __future__ import annotations

import numpy as np

from . import chem, geom
from .energy import EnergyModel

HISTORY = 8


class NeuroFoldV8Env:
    def __init__(self, profile, seed, max_steps=None):
        self.profile = profile
        self.p = profile["physics"]
        self.seed = int(seed)
        self.max_steps = int(max_steps or profile.get("max_steps", 96))
        self.rng = np.random.default_rng(self.seed)
        self._build_static()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _build_static(self):
        seq = self.profile["sequence"]
        n_chains = int(self.profile.get("n_chains", 1))
        props1 = chem.bead_properties(seq)
        n1 = props1["n"]
        self.n_per_chain = n1
        self.n_chains = n_chains
        self.n = n1 * n_chains
        rep = lambda a: np.tile(np.asarray(a), n_chains)
        self.props = {k: rep(v) for k, v in props1.items()
                      if isinstance(v, np.ndarray)}
        self.props["cls"] = self.props["cls"].astype(int)
        self.chain_id = np.repeat(np.arange(n_chains), n1)
        self.regions = {k: rep(v) for k, v in
                        chem.region_weights(self.profile, n1).items()}
        self.energy = EnergyModel(self.props, self.chain_id, self.p)
        # Structurally important ("native") contacts: turn-region and N-terminal
        # contacts that are not part of the pathological ladder.  Only breaking
        # THESE costs permanent damage; dismantling the pathological register is
        # the intended therapy and is cheap.  Charging every non-ladder break
        # made the targeted and blind arms cost the same (0.275 vs 0.241).
        prot = np.maximum(self.regions.get("turn", np.zeros(self.n)),
                          self.regions.get("nterm", np.zeros(self.n)))
        self._protective_pair = np.outer(prot, prot) > 0.05

    # ------------------------------------------------------------------ reset
    def _prepare_common_random_numbers(self):
        """Precompute every stochastic draw, indexed by STEP rather than drawn
        from a running stream.

        With a single running generator the random stream diverges as soon as two
        policies take different actions, so a paired comparison silently mixes
        policy effect with noise realization.  Precomputing per-step noise means
        two policies on the same episode seed see identical initial conditions
        and identical disturbances at every step index, and differ only through
        their own actions — proper common random numbers.
        """
        rng = np.random.default_rng([self.seed, 12345])
        T = self.max_steps + 2
        self._crn_thermal = rng.normal(0.0, 1.0, (T, self.n, 3))
        self._crn_ou = rng.normal(0.0, 1.0, (T, len(self.profile["conditions"])))
        self._crn_accept = rng.random(T)
        self._crn_obs = rng.normal(0.0, 1.0, (T, 8))

    def reset(self):
        p = self.p
        # initial state uses its own stream so it is identical across policies
        rng = np.random.default_rng([self.seed, 1])
        self.rng = rng
        self._prepare_common_random_numbers()
        seed_prob = float(p.get("seed_register_prob", 0.0))
        self.seeded_pairs = []
        if self.n_chains == 2 and rng.random() < seed_prob:
            # hand the agent an early-stage nucleation event to act on
            lo, hi = p.get("seed_register_len", [2, 4])
            L = int(rng.integers(lo, hi + 1))
            xA, xB, pairs = geom.seeded_antiparallel_pair(
                rng, self.n_per_chain, p["b0"], p["seed_gap"], L,
                splay=p.get("seed_splay", 1.15),
                jitter=p.get("seed_jitter", 0.06),
                extend_noise=p.get("seed_extend_noise", 0.10),
                theta0=p["theta0"])
            self.x = np.vstack([xA, xB])
            self.seeded_pairs = pairs
            self.seed_register_len = L
        else:
            xs = []
            for c in range(self.n_chains):
                start = np.zeros(3) if c == 0 else rng.normal(0, 1.0, 3) + np.array(
                    [p["chain_separation"], 0.0, 0.0])
                axis = np.array([0.0, 1.0, 0.0]) + 0.25 * rng.normal(size=3)
                xs.append(geom.init_chain_ideal(rng, self.n_per_chain, p["b0"],
                                                p["theta0"], start=start, axis=axis,
                                                dihedral_spread=p.get("init_dihedral_spread", np.pi)))
            self.x = np.vstack(xs)
            self.seed_register_len = 0

        # latent environment (OU-like); only noisy proxies of some are observed
        cc = self.profile["conditions"]
        self.env_state = {k: float(rng.uniform(*v)) for k, v in cc.items()}
        self.env_mu = dict(self.env_state)

        n = self.n
        self.age = np.zeros((n, n), dtype=int)
        self.locked = np.zeros((n, n), dtype=bool)
        self.broken_for = np.zeros((n, n), dtype=int)
        self.mod = np.zeros((n, n))        # contact-selective energy suppression
        self.block = np.zeros((n, n))      # transient re-formation penalty
        self.strain = np.zeros(n)
        self.strain_streak = np.zeros(n, dtype=int)
        self.visits = np.zeros(n)
        self.damage = 0.0
        self.steps = 0
        self.accepted = 0
        self.action_energy = 0.0
        self.barrier_energy = 0.0
        self.budget = float(self.p["action_budget"])
        self.history = np.zeros((HISTORY, 5))
        self._recompute()
        self.initial_energy = self.energy_total
        self.initial_pathology = self.pathology
        self.energy_sum = 0.0
        self.safe_steps = 0
        return self.observe()

    # ------------------------------------------------------------- mechanics
    def _recompute(self, x=None):
        x = self.x if x is None else x
        e, terms, aux = self.energy.total(x, self.env_state["screening"],
                                          crowding=self.env_state["crowding"],
                                          mod=self.mod)
        self.energy_total = e
        self.energy_terms = terms
        self.aux = aux
        self.contact = aux["contact"]
        self.dist = aux["dist"]
        self.density = self.contact.sum(1)
        self.pathology = self._pathology(aux)
        return e

    def _pathology(self, aux):
        """Task-defined pathological order parameter.  [B]  post-A3 revision

        Kept OUT of the observation: the agent sees geometry and contacts, not
        this label.

        Why this is not a pair count. Until the A3 gate failed, pathology was
        driven by `n_run`, the length of the single longest run, so shortening
        that run *anywhere* paid the same. A hand-set 3-weight policy blocking
        an arbitrary ladder rung reached 96.2% of a trained 2541-parameter
        policy's gain (95% CI [92.1%, 99.8%]): the task did not require the
        architecture it ships with.

        Two changes, both aimed at making *which* rung is disrupted matter:

        1. Pathology sums a CONVEX function of length over every critical
           nucleus instead of reading the maximum. With exponent `path_exp`,
           splitting a run down the middle is worth far more than trimming its
           end: an 8-rung run scores (8-L_min+1)**2, halving it scores about a
           seventh of that, trimming one end about three quarters. A rung's own
           features are identical in both cases, so the choice can only be made
           by aggregating along the register.

        2. Only MATURE runs are critical. A young run is a decoy: ladder-
           positive, identical edge by edge, contributing nothing to pathology,
           yet still costing budget and possibly damage if attacked.
           Criticality is therefore dynamic — disrupting a nucleus resets its
           age and promotes whichever rival nucleus has been growing
           unattended, so the target moves during the episode.
        """
        p = self.p
        lad = aux["ladder"]
        L_min = p["L_min"]
        exp = p.get("path_exp", 2.0)
        crit_age = p.get("path_crit_age", 6.0)
        norm = p.get("path_norm", 1.0)

        total, locked_crit = 0.0, 0.0
        self._critical = np.zeros_like(lad, dtype=bool)
        for run in self.energy.enumerate_runs(lad):
            if len(run) < L_min:
                continue
            if float(np.mean([self.age[a, b] for a, b in run])) < crit_age:
                continue                    # decoy: ladder-positive, not pathological
            total += (len(run) - L_min + 1.0) ** exp
            for a, b in run:
                self._critical[a, b] = self._critical[b, a] = True
                if self.locked[a, b]:
                    locked_crit += 0.5
        return p["path_nucleus"] * total / norm + p["path_locked"] * locked_crit

    def _relax(self, x, i, n_steps=None):
        """Local elastic relaxation around bead i after an intervention.

        Separates REVERSIBLE elastic strain from IRREVERSIBLE damage.  A move
        transiently stretches bonds and can push neighbours together; a real
        chain relaxes that on a fast timescale.  Without this step every
        intervention, however well aimed, paid a permanent cost for elastic
        deformation it should simply have shed — which is what made effective
        intervention unaffordable.  Only violations that SURVIVE relaxation
        (true clashes, plastic geometry) go on to cause damage.        [P]
        """
        p = self.p
        n_steps = int(p.get("n_relax", 2)) if n_steps is None else n_steps
        if n_steps <= 0:
            return x
        rate = float(p.get("relax_rate", 0.45))
        b0 = p["b0"]
        sig = p["sigma_excl"]
        x = x.copy()
        touched = [i]
        prev, nxt = geom.neighbors_in_chain(i, self.chain_id)
        for j in (prev, nxt):
            if j is not None:
                touched.append(j)
        for _ in range(n_steps):
            # bond-length relaxation on the touched beads
            for a in touched:
                pa, na = geom.neighbors_in_chain(a, self.chain_id)
                for b in (pa, na):
                    if b is None:
                        continue
                    d = x[a] - x[b]
                    L = float(np.linalg.norm(d))
                    if L > 1e-9:
                        x[a] -= rate * (L - b0) * d / L
            # clash relief against every non-bonded partner
            for a in touched:
                d = x[a] - x
                L = np.linalg.norm(d, axis=1)
                bad = np.where(self.energy.nb_mask[a] & (L < sig) & (L > 1e-9))[0]
                for b in bad:
                    x[a] += rate * (sig - L[b]) * d[b] / L[b]
        return x

    def _barrier(self, i, mag):
        p = self.p
        lock_i = float(self.locked[i].any())
        return p["b_base"] * (1.0 + p["b_density"] * self.density[i]
                              + p["b_lock"] * lock_i
                              + p["b_move"] * mag * mag)

    def _update_topology(self):
        """Maturation and lock release — the hysteresis mechanism."""
        p = self.p
        lad = self.aux["ladder"]
        self.age = np.where(lad, self.age + 1, 0)
        self.broken_for = np.where(lad, 0, self.broken_for + 1)
        # a pair locks once it has been part of a ladder long enough
        mature = lad & (self.age >= p["tau_mat"])
        if self.aux["n_run"] >= p["L_min"]:
            self.locked |= mature
        # locks release only after a sustained break
        self.locked &= ~(self.broken_for >= p["tau_reset"])

    def _advance_environment(self):
        p = self.p
        eps = self._crn_ou[self.steps]
        for j, (k, mu) in enumerate(self.env_mu.items()):
            z = self.env_state[k]
            self.env_state[k] = float(z + p["ou_theta"] * (mu - z)
                                      + p["ou_sigma"] * eps[j])
        # crowding drifts upward over an episode: the window for cheap
        # intervention closes with time.                                   [B]
        self.env_state["crowding"] += p["crowding_drift"]

    # ------------------------------------------------------------------ step
    def step(self, action):
        """action = (i, j, strength): apply a contact-selective destabilisation
        to pair (i, j), then let the chain relax under ordinary Metropolis
        dynamics.

        The agent no longer displaces coordinates.  It modulates the local energy
        landscape — a coarse proxy for a small molecule, a chaperone-like
        interaction, or a local solvation/screening change that destabilises one
        specific contact.  Whether anything moves is then decided by the normal
        stochastic dynamics.  This decouples control from mechanical deformation:
        with a mechanical operator, efficacy and damage necessarily travelled the
        same channel, so any intervention strong enough to break a register also
        damaged the backbone.
        """
        if self.steps >= self.max_steps or self.budget <= 0:
            raise RuntimeError("episode terminated")
        i, j, strength = int(action[0]), int(action[1]), float(action[2])
        if not (0 <= i < self.n) or not (0 <= j < self.n) or not np.isfinite(strength):
            raise ValueError("invalid action")
        strength = float(np.clip(strength, 0.0, 1.0))
        p = self.p

        prev_energy = self.energy_total
        prev_pathology = self.pathology
        prev_damage = self.damage
        prev_contact = self.contact.copy()
        prev_ladder = self.aux["ladder"].copy()

        # ---- 1. contact-selective modulation ------------------------------
        if i != j and self.energy.nb_mask[i, j] and strength > 0:
            gain = p["mod_gain"] * strength
            self.mod[i, j] = min(p["mod_max"], self.mod[i, j] + gain)
            self.mod[j, i] = self.mod[i, j]
            self.block[i, j] = min(p["block_max"],
                                   self.block[i, j] + p["block_gain"] * strength)
            self.block[j, i] = self.block[i, j]
            applied = True
        else:
            applied = False

        # ---- 2. ordinary Metropolis relaxation ----------------------------
        # No agent forcing: single-bead thermal proposals accepted on the
        # modulated landscape.  A weakened pathological contact dissolves because
        # it is no longer favourable, not because it was pulled apart.
        sigma_th = p["thermal_noise"] * self.env_state["temperature"]
        kT = max(1e-6, p["kT"] * self.env_state["temperature"])
        n_metro = int(p.get("n_metropolis", 3))
        accepted_here = 0
        for m in range(n_metro):
            noise = sigma_th * self._crn_thermal[self.steps] * p.get("metro_scale", 1.0)
            cand = self.x + noise
            e_new, _, _ = self.energy.total(cand, self.env_state["screening"],
                                            crowding=self.env_state["crowding"],
                                            mod=self.mod)
            # re-formation penalty on blocked pairs discourages immediate re-pairing
            barrier = float(np.sum(self.block * (self.contact > p["contact_break_level"]))) / 2.0
            dE = (e_new - self.energy_total) + p["block_weight"] * barrier
            u = self._crn_accept[(self.steps + m) % len(self._crn_accept)]
            if dE <= 0 or u < np.exp(-min(60.0, dE / kT)):
                self.x = cand
                accepted_here += 1
            self._recompute()
        self.accepted += accepted_here

        # ---- 3. costs -----------------------------------------------------
        self.action_energy += strength
        self.barrier_energy += p["mod_cost"] * strength
        self.budget -= (1.0 + p["budget_move_cost"] * strength)
        if applied:
            self.visits[i] += 1
            self.visits[j] += 1

        self.mod *= p["mod_decay"]
        self.block *= p["block_decay"]
        self._update_topology()
        self._advance_environment()

        # ---- 4. reversible strain / irreversible damage --------------------
        self.strain *= p["strain_decay"]
        if applied:
            self.strain[i] += strength * 0.5
            self.strain[j] += strength * 0.5
        over = self.strain > p["strain_thresh"]
        self.strain_streak = np.where(over, self.strain_streak + 1, 0)

        clash = float(np.sum(np.maximum(0.0, p["sigma_excl"] - self.dist)
                             * self.energy.nb_mask) / 2.0)
        e_bond, e_ang, _ = self.energy.bonded(self.x)
        geom_violation = max(0.0, (e_bond + e_ang) - p["elastic_limit"])
        broke = ((prev_contact > p["contact_break_level"])
                 & (self.contact < p["contact_break_level"]))
        important = broke & (~prev_ladder) & self.energy.nb_mask & self._protective_pair
        disrupted = float(np.sum(important)) / 2.0

        inc = (p["eta_clash"] * clash
               + p["eta_geom"] * geom_violation
               + p["eta_disrupt"] * disrupted
               + p["eta_path"] * max(0.0, self.pathology - p["path_thresh"]))
        self.last_damage_terms = {
            "clash": p["eta_clash"] * clash,
            "geometry": p["eta_geom"] * geom_violation,
            "disrupted_structure": p["eta_disrupt"] * disrupted,
            "pathology": p["eta_path"] * max(0.0, self.pathology - p["path_thresh"])}
        repair = p["eta_repair"] * self.env_state["chaperone"] * self.damage
        self.damage = float(max(0.0, min(p["damage_cap"],
                                         self.damage + max(0.0, inc) - repair)))

        self.steps += 1
        self.energy_sum += self.energy_total
        safe = int(self.pathology < p["safe_pathology"] and clash < p["safe_clash"]
                   and self.damage < p["safe_damage"])
        self.safe_steps += safe

        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = [i / max(1, self.n - 1), j / max(1, self.n - 1), strength,
                           float(applied), self.steps / self.max_steps]

        dense = ((prev_energy - self.energy_total) * p["r_energy"]
                 + (prev_pathology - self.pathology) * p["r_path"]
                 + p["r_safe"] * safe
                 - p["r_damage"] * (self.damage - prev_damage)
                 - p["r_action"] * strength
                 - p["r_barrier"] * p["mod_cost"] * strength)
        done = self.steps >= self.max_steps or self.budget <= 0
        return self.observe(), float(dense), done, self.info()

    # ----------------------------------------------------------- observation
    def observe(self):
        """Node features, edge features and a history block.

        Deliberately NOT exposed: oxidative stress, chaperone capacity, lock
        flags, the pathology order parameter, reward decomposition, any
        region-level 'protective' label.
        """
        n = self.n
        p = self.p
        u = self.aux["u"]
        deg = self.contact.sum(1)
        node = np.column_stack([
            self.props["hydro"], self.props["charge"], self.props["beta"],
            self.props["arom"], self.props["disorder"], self.props["pro"],
            self.props["gly"],
            np.eye(chem.K_CLASS)[self.props["cls"]],       # one-hot bead class
            u,                                            # local chain direction
            np.clip(deg / max(1.0, 0.2 * n), 0, 1),
            np.clip(self.strain, 0, 3),
            np.clip(self.visits / max(1, self.max_steps / 8), 0, 1),
            self.chain_id.astype(float),
            np.full(n, self.steps / max(1, self.max_steps)),
            np.full(n, max(0.0, self.budget) / p["action_budget"]),
            np.full(n, self._noisy("crowding")),
            np.full(n, self._noisy("screening")),
            np.full(n, self._noisy("temperature")),
        ])
        ei, ej = np.nonzero(self.contact > p["edge_threshold"])
        keep = ei != ej
        ei, ej = ei[keep], ej[keep]
        if len(ei) == 0:
            ei = np.array([0]); ej = np.array([min(1, n - 1)])
        edge = np.column_stack([
            self.dist[ei, ej] / p["contact_cutoff"],
            self.contact[ei, ej],
            np.clip(self.energy.sep[ei, ej] / 10.0, 0, 1),
            self.props["charge"][ei] * self.props["charge"][ej],
            self.props["hydro"][ei] * self.props["hydro"][ej],
            self.props["arom"][ei] * self.props["arom"][ej],
            self.props["beta"][ei] * self.props["beta"][ej],
            self.aux["align"][ei, ej],
            (self.chain_id[ei] == self.chain_id[ej]).astype(float),
            np.clip(self.age[ei, ej] / max(1, p["tau_mat"]), 0, 2),
            self.energy.MM[ei, ej],
            self.aux["ladder"][ei, ej].astype(float),
        ])
        return {"node": node, "edge_index": np.vstack([ei, ej]), "edge": edge,
                "history": self.history.copy(),
                "global": np.array([self.steps / self.max_steps,
                                    max(0.0, self.budget) / p["action_budget"],
                                    self._noisy("crowding"), self._noisy("screening"),
                                    self._noisy("temperature"),
                                    float(self.accepted) / max(1, self.steps)])}

    _OBS_KEYS = {"crowding": 0, "screening": 1, "temperature": 2,
                 "oxidative": 3, "chaperone": 4}

    def _noisy(self, key):
        j = self._OBS_KEYS.get(key, 5)
        t = min(self.steps, self._crn_obs.shape[0] - 1)
        return float(self.env_state[key] + self.p["obs_noise"] * self._crn_obs[t, j])

    # ---------------------------------------------------------------- summary
    def info(self):
        return {"energy": self.energy_total, "pathology": self.pathology,
                "damage": self.damage, "n_run": self.aux["n_run"],
                "locked": int(self.locked.sum() // 2)}

    def summary(self):
        mean_e = self.energy_sum / max(1, self.steps)
        return {
            "initial_energy": float(self.initial_energy),
            "final_energy": float(self.energy_total),
            "improvement": float(self.initial_energy - self.energy_total),
            "path_gain": float(self.initial_energy - mean_e),
            "pathology_reduction": float(self.initial_pathology - self.pathology),
            "final_pathology": float(self.pathology),
            "safe_fraction": float(self.safe_steps / max(1, self.steps)),
            "damage": float(self.damage),
            "clash": float(np.sum(np.maximum(0.0, self.p["sigma_excl"] - self.dist)
                                  * self.energy.nb_mask) / 2.0),
            "action_energy": float(self.action_energy / max(1, self.steps)),
            "barrier_energy": float(self.barrier_energy / max(1, self.steps)),
            "accept_rate": float(self.accepted / max(1, self.steps)),
            "n_run": int(self.aux["n_run"]),
            "locked_pairs": int(self.locked.sum() // 2),
            "steps": int(self.steps),
            "budget_left": float(max(0.0, self.budget)),
            "rg": geom.radius_of_gyration(self.x),
        }
