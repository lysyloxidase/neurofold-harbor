# Scientific scope — NeuroFold-Harbor v8

## What this benchmark is

A **physics-informed coarse-grained stochastic conformational-control
benchmark** over disease-relevant, aggregation-competent protein regions at
5 residues per bead.

## What it is explicitly NOT

* not atomistic or coarse-grained molecular dynamics — transitions use a
  Metropolis-like acceptance rule on a coarse energy model, not integrated
  equations of motion;
* not a folding free-energy predictor;
* not an experimentally validated model of aggregation kinetics;
* not a disease simulator, and not predictive of any therapeutic effect;
* not a model of full-length proteins (see fragment rationale below).

Terms such as "hydrogen bond", "chaperone" and "small molecule" appear only as
**coarse proxies** and are labelled as such at every use site in the code.

## Why fragments, not full chains

Amyloid nucleation is driven by short aggregation-competent segments, while
large flanking regions of these proteins remain disordered and do not enter the
cross-β core. Modelling the full chain therefore spends the entire compute
budget on residues that do not participate in the mechanism under study, and —
more importantly — dilutes the signal the benchmark is trying to measure.

Each fragment below is the canonical aggregation-competent region reported for
that protein. Runtime was **not** the selection criterion; the ranges were
chosen from the aggregation literature and then checked for feasibility. Where a
biologically motivated fragment turned out to be expensive, the fragment was
kept and the episode budget adjusted instead.

## Fragments

| Task | Residue range (1-based, inclusive) | Length | Chains | Rationale |
|---|---|--:|--:|---|
| `alzheimer-abeta42-v8` | Aβ 1–42 (full peptide) | 42 | 2 | The full peptide *is* the aggregation unit; no truncation is appropriate. Contains the central hydrophobic cluster (KLVFF, 16–20), the turn region (22–29) including the D23–K28 pair, and the C-terminal hydrophobic core (30–42). |
| `parkinson-alpha-synuclein-v8` | 61–95 (NAC) plus flanks 55–60 and 96–105 | 51 | 2 | NAC is the necessary and sufficient region for α-syn fibril formation. Flanks retain the amphipathic-helix boundary on one side and the start of the acidic C-terminal tail on the other, so the protective long-range tail contact is representable. |
| `alzheimer-tau-v8` | 592–597 (PHF6*) through 623–628 (PHF6), i.e. 592–628 | 37 | 2 | PHF6* and PHF6 are the two hexapeptide motifs that nucleate paired helical filaments. The intervening repeat-domain segment is retained so both motifs and their spacing are present. The projection domain is omitted because it does not enter the cross-β core. |
| `als-ftd-tdp43-v8` | 311–360 (LCD aggregation hotspot) | 50 | 2 | The conserved region of the C-terminal low-complexity domain; contains the aromatic/sticker residues that drive LCD self-association. The RRM domains are omitted: they are folded and not part of the aggregation core. |
| `huntington-htt-polyq-v8` | N17 (1–17) + polyQ tract + polyP (proline-rich) | 62 | 2 | Nucleation depends on the polyQ tract length, with N17 modulating and the polyP region interrupting β-structure. A disease-length tract is used; the surrogate is explicitly not the full HTT exon 1 of any specific allele. |

Exact sequences and 0-based half-open slices are recorded in each task's
`environment/profile.json` under `sequence` and `fragment`.

## What the coarse-grained model represents

* **Beads**: 5 consecutive residues, carrying averaged hydropathy, net charge,
  β-propensity, aromatic/sticker score, disorder propensity, and a discrete
  chemical class.
* **Geometry**: Cartesian bead positions with bond-length, bond-angle and
  torsional terms, plus excluded volume. Both control degrees of freedom
  provably move the 3-D configuration (regression-tested).
* **Pair interactions**: hydrophobic, electrostatic (screened), aromatic, and a
  **non-separable** class-interaction matrix, so the energetic consequence of a
  contact cannot be recovered from the two beads' own features. Partner
  identity carries information.
* **β-register**: a directional and *saturable* term — alignment-dependent and
  capped at a small number of partners per bead, as backbone hydrogen bonding
  is — plus a cooperative nucleation term.
* **Maturation and hysteresis**: a register that persists locks, resists
  modulation, and only releases after a sustained break. This is what makes the
  *order* of interventions matter.
* **Damage**: irreversible, and charged only for physically meaningful events —
  elastic-limit violation, steric clash, and disruption of protective
  (non-pathological) structure.
* **Environment**: slowly varying crowding, screening and temperature, plus
  **latent** oxidative-stress and chaperone-capacity terms that are never
  observed and gate repair.

## What the intervention represents

The action is a **contact-selective destabilisation** of one specific pair,
followed by ordinary Metropolis relaxation. It is a coarse proxy for a small
molecule that destabilises a particular contact, a chaperone-like interaction,
or a local solvation/screening change. **It is not a mechanical force**: the
agent never displaces coordinates, and whether the structure moves is decided by
the stochastic dynamics.

This choice is deliberate. An earlier mechanical operator coupled efficacy and
backbone deformation through the same channel, so any intervention strong enough
to dissolve a register also damaged the chain, and no therapeutic window existed
(measured: every intervention arm scored below no-op).

## Known limitations

* Single-fragment, two-chain systems: higher-order oligomers and fibril
  elongation are not represented.
* The pair-class matrix and the cooperativity exponent are synthetic in
  functional form, chosen to create a non-separable relational structure rather
  than fitted to data.
* Bead-level averaging removes side-chain detail entirely.
* Absolute energies are in arbitrary units and are not comparable to kcal/mol.
