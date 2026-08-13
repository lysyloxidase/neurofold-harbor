# Stage report — HTT polyQ porting gate: **FAIL**

Task: `huntington-htt-polyq-v8`
Result: **BLOCKED at P7 and P8.** Port stopped. Aβ42 untouched and still frozen.
No further protein started.

## Fragment defined (step 1)

N17 (residues 1–17) + polyQ36 + polyP11 = **64 aa**, 13 beads per chain, two
chains, 26 beads total. Regions: `core` = polyQ (aggregation-competent),
`turn` = polyP (β-interrupting, protective), `nterm` = N17 (modulatory).
Recorded in `environment/profile.json` under `fragment`.

## What was fixed, and why it was not "tuning around a failure"

The first gate run failed P5, P7 and P8 because I had **under-executed step 2**:
I copied the frozen Aβ42 physics wholesale instead of deriving HTT-specific
pathology parameters. The gate caught that correctly.

Diagnosis produced a general finding:

> **Every damage channel that sums over beads or pairs is extensive.** Its
> natural scale grows with fragment size, so inheriting it from a smaller
> protein saturates the damage channel and masks every downstream gate.

Three channels were affected. Each was normalised by a **measured ratio**, not
fitted to the gate:

| channel | Aβ42 | HTT | normalisation |
|---|--:|--:|---|
| `path_*` (core-pair mass) | 4.0 | 207.4 | × 0.0193 |
| `elastic_limit` (bonded-term count) | baseline 14.10 | baseline 27.20 | same 1.418× multiple of baseline → 38.58 |
| `eta_disrupt` (protective-pair mass) | 50 | 92 | ÷ 1.840 → 0.0652 |

Only per-task **profile** values changed. Environment code, policy schema,
verifier and reward form remain frozen at Aβ42.

Oracle catastrophe fell monotonically across the three fixes:
**0.958 → 0.458 → 0.229 → 0.125** (threshold 0.10). I stopped there: the
principled normalisations are exhausted and any further adjustment would be
fitting to the gate.

## Final gate state

| gate | result |
|---|---|
| P1 geometry responds | PASS |
| P2 calibration non-degenerate | PASS |
| P3 no-op not optimum | PASS |
| P4 pathology matures | PASS |
| P5 targeted intervention works | PASS |
| P6 blind targeting worse | PASS |
| **P7 action order matters** | **FAIL** |
| **P8 catastrophe controllable** | **FAIL** (0.125 vs < 0.10) |

```
noop  utility=-1.8219 cat=0.562 dmg=1.018 locked=5.15
o10   utility=-0.5867 cat=0.125 dmg=0.377 locked=5.67
b10   utility=-1.8712 cat=0.521 dmg=1.103 locked=5.27
```

## Root cause

**The frozen chemistry does not represent the mechanism that drives polyQ
aggregation.**

Glutamine maps to the `polar` bead class. In the frozen pair matrix:

```
Abeta42 core (aliphatic):  M[0,0] = -0.90
HTT polyQ  (polar):        M[2,2] = -0.15     <- 6.0x weaker
```

Q also carries hydropathy −3.5 and zero aromatic character, so the hydrophobic
and sticker terms contribute nothing either. The consequence is measurable: the
β-register barely forms and does not persist.

| | ladder pairs held simultaneously | distinct pairs over an episode | register rotation |
|---|--:|--:|--:|
| Aβ42 | **7.2** | 9 | 1.3× (stable) |
| HTT | **1.8** | 5 | 2.7× (transient) |

With only ~1.8 pairs engaged at a time and the register shifting, there is no
persistent nucleus to attack early rather than late — so **P7 has nothing to
measure**, and the residual damage keeps catastrophe above threshold.

An earlier hypothesis of mine — that polyQ forms a *longer* register whose size
defeats single-pair intervention — was **wrong**, and the measurement above is
what refuted it. The register is smaller, not larger.

Physically this is expected and is a gap in the model, not in the protein:
polyQ aggregation is driven by **glutamine side-chain hydrogen bonding** (the
polar-zipper picture), a specific mechanism the frozen chemistry does not
represent. The Aβ42 chemistry was built around a hydrophobic/aromatic core and
generalises to proteins that aggregate the same way — not to a homopolymeric
polar tract.

## What a fix would require, and why I stopped

The defensible fix is a glutamine-specific side-chain hydrogen-bonding term:
a saturable, directional interaction between high-Q beads, analogous to the
existing backbone β term but keyed on Q content.

That touches `chem.py` / `energy.py` — **the frozen Aβ42 architecture**.
`FORBIDDEN_CHANGES.md` prohibits altering it during porting, and doing so would
invalidate the Aβ42 freeze including its anchors and final-test seeds.

This therefore needs a decision, and I am not taking it unilaterally. Options:

1. **Re-open the freeze** and add the polar-zipper term to the shared chemistry,
   then re-validate and re-freeze Aβ42 (its results should be unaffected, since
   Q content in Aβ42 is negligible — but that must be verified, not assumed).
2. **Ship HTT with a different fragment** whose aggregation is hydrophobic
   rather than polar — but no such fragment is faithful to Huntington's
   disease, so this would be choosing biology to fit the engine. Not recommended.
3. **Drop HTT from v8** and port the three remaining proteins, whose cores are
   hydrophobic (α-syn NAC) or aromatic (TDP-43 LCD, tau PHF6/PHF6*) and should
   fit the existing chemistry.

Option 3 unblocks progress immediately; option 1 is the scientifically complete
answer. Both are legitimate; option 2 is not.

## Next permitted action

Await a decision on the three options above. Do not port further proteins under
the assumption that HTT will be fixed, and do not modify Aβ42.
