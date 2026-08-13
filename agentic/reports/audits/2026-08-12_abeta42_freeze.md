# Stage report — Aβ42 golden template frozen

Stage: `IMPLEMENTER` → `TEST` → `CRITIC` → `GATEKEEPER` → `FREEZE`
Task: `alzheimer-abeta42-v8`
Result: **PASS — frozen**

## Changes made

* Built the full Harbor package: `instruction.md`, `task.toml`,
  `environment/` (Dockerfile, `neurofold8` package, public profile, CLI,
  CMA-ES baseline, reward calibration), `tests/` (Dockerfile, `test.sh`,
  verifier, hidden profile), `solution/` (`solve.sh`, challenge reference).
* Defined the artifact contract in `policy_runtime.py`: schema string, exact
  architecture, exactly 2541 parameters, finite values only, `|w| ≤ 30`,
  file ≤ 768 KiB.
* Built the reward calibration from a six-policy mixture on a dedicated
  calibration split, behind the `DegenerateCalibration` gate.
* Trained the challenge reference with sep-CMA-ES (2 restarts, 12k episodes
  each, selection on public validation only).
* Generated the hidden profile with a shifted regime, then 128 fresh
  final-test seeds, then measured the anchors — in that order.

## Tests run and results

### Artifact contract (19 cases, unit level)

All malformed inputs rejected, valid inputs accepted, including a weight
exactly at the bound. **PASS.**

### Harbor deliverable gate

| id | check | result |
|---|---|---|
| H1 | required layout | PASS |
| H2 | continuous reward, no-network, separate verifier | PASS |
| H3 | both images build from clean context | PASS |
| H4 | `solve.sh` → `test.sh` → reward | **PASS** (0.9999999999999998; `reward.txt` = `1`) |
| H5 | malformed artifacts rejected | **PASS 7/7**, each with a distinct, correct error |
| H6 | verifier imports no agent code | PASS — imports only stdlib, NumPy and `/tests` modules |
| H7 | no hidden leakage in `environment/` | PASS — no `test_seeds`, `frozen_anchors`, `challenge_reference` or `hidden_` |
| H8 | splits disjoint | PASS — final(128) ∩ train(64) = ∅, ∩ validation(32) = ∅, ∩ calibration(64) = ∅ |

H5 detail — each case produced its own error rather than one generic failure:

```
smieci (nie JSON)      JSONDecodeError
zly schemat            invalid schema (expected 'neurofold-graph-policy-v8')
zla liczba parametrow  params has 3 entries, expected 2541
zla architektura       architecture mismatch: hidden=99 expected 12
NaN w wagach           params must be finite (no NaN or inf)
waga poza limitem      max |weight| = 1000000000.000 exceeds bound
kod w params           params must contain only numbers
```

A first version of this test passed six identical cases because `docker run`
had no stdin attached; it was rerun with `-i` so each payload actually
reached the verifier. The initial result was not evidence of anything.

### Anchors

| policy | public validation | hidden final test (128 ep) |
|---|--:|--:|
| zero / no-op | −2.3058 (cat 0.688) | −2.3103 (cat 0.672) |
| challenge reference | **+0.0666** (cat 0.000) | −0.1460 (cat 0.070) |

The reference beats the hand-coded oracle (+0.0551) on public validation. It
loses 0.21 utility crossing to the hidden regime, which is the intended cost of
the condition/mechanism shift — meaningful, but it still transfers, clearing the
zero anchor by 2.16.

Zero policy scores exactly `reward = 0.0` through the real verifier, confirming
the lower anchor.

### Structural gates

| gate | status |
|---|---|
| A1 relational necessity | **PASS** +0.557, 95% CI [+0.126, +0.698], 29% of gain |
| A6 action-order dependence | **PASS** CI [+0.083, +0.547] |
| A7 safety | partial — reference catastrophe 0.070 on hidden split |
| A2, A3, A4, A5, A8 | not yet run |

## PASS / FAIL

**PASS.** Aβ42 is frozen: 30 files hashed into `frozen_hashes.json`,
final-test seed SHA-256 recorded in `split_registry.json`.

## Next permitted action

Port to **one** further protein through the 8-check porting gate. The Aβ42
architecture is now frozen: policy schema, reward definition, verifier and
artifact contract may not change during porting. If a port fails its gate, stop
that protein and report the root cause rather than tuning around it.
