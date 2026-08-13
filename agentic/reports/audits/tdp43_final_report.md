# TDP-43 — final validation record

`als-ftd-tdp43-v8`, UniProt Q13148 residues 311–360 (C-terminal low-complexity
domain aggregation hotspot).

This is the one task in the v8 set that does not pass the porting gate cleanly.
It ships with that stated. Both results below are reported together; neither is
quotable alone.

## Summary

| item | result |
|---|---|
| Heuristic porting gate P8 (catastrophe controllable) | **FAIL** — not re-opened |
| Learned-control diagnostic (pre-registered rule) | **PASS** |
| Learned reference catastrophe rate | **8.59% (22 / 256 fresh seeds)** |
| 95% CI (hierarchical bootstrap, 20 000 draws) | **[5.5%, 12.1%]** |
| 95% CI (Wilson score) | [5.7%, 12.7%] |
| Threshold | 10% |
| Interpretation | point estimate below threshold, **interval crosses it** |
| Porting-gate tally | 6 PASS, 1 INCONCLUSIVE (P7), 1 FAIL (P8) |

## 1. P8 failed on the heuristic probe, and stays failed

P8 asks whether the catastrophe rate can be brought under 10% by control. It is
probed with hand-coded policies. Six were tested on 48 development seeds:

| policy | catastrophe |
|---|--:|
| no-op | 0.646 |
| oracle continuous, s = 1.0 | **0.167** (best) |
| every 2 steps, s = 1.0 | 0.208 |
| every 3 steps, s = 1.0 | 0.250 |
| every 2 steps, s = 0.6 | 0.354 |
| steps 0–39 only | 0.417 |
| every 2 steps, until step 60 | 0.438 |

The best heuristic left 16.7% of episodes catastrophic, above threshold, so
**P8 = FAIL**. Damage under that policy is heavy-tailed: median 0.106, but 8 of
48 episodes exceed 0.75 and reach the 2.5 cap.

`P8_original` is recorded as FAIL in `agentic/state/benchmark_state.json` and in
`agentic/reports/validation/porting_gate_als-ftd-tdp43-v8.json`. Nothing below
changes it.

## 2. The diagnostic that followed, and its pre-registered rule

Aβ42's *trained* reference reaches catastrophe 0.000 where its heuristic oracle
reaches 0.021, so learned control can be materially better than the crude probe.
That raised a narrow question: is TDP-43's heavy tail irreducible, or is the
heuristic probe simply a poor proxy for controllability on a low-complexity
domain?

The decision rule was written down **before** the run, in
[`2026-08-13_tdp43_diagnostic_PREREGISTERED.md`](2026-08-13_tdp43_diagnostic_PREREGISTERED.md):

| learned catastrophe | outcome |
|---|---|
| < 10% and utility improves over no-op | keep the task; report heuristic P8 FAIL **together with** learned-control diagnostic PASS |
| 10–15% | INCONCLUSIVE — do not freeze |
| > 15% | FAIL — ship four tasks |

Method: the same trainer, architecture and budget as the other four tasks
(sep-CMA-ES, 12 000 episodes, 2 restarts, 8 training seeds, selection on the
public validation split). No reward, damage model, threshold or chemistry change.

A first diagnostic at 64 seeds returned 0.109 — inside the INCONCLUSIVE band. A
single confirmatory evaluation was then run at **N = 256 fresh seeds
(5000–5255)**, with no intermediate inspection at 128 and no option to extend N
afterwards.

## 3. Result at N = 256

Seeds 5000–5255, disjoint from train (1000–1063), validation (2000–2031), gate
development (2000–2047), calibration (3000–3063), the 64-seed diagnostic
(4000–4063) and the final-test range (900000+).

| arm | utility | catastrophe | 95% CI | mean damage | p90 damage | safe fraction |
|---|--:|--:|:--|--:|--:|--:|
| no-op | −2.0715 | 0.6328 (162/256) | [0.574, 0.691] | 1.250 | 2.500 | 0.371 |
| heuristic oracle s=1.0 | −0.3150 | 0.1289 (33/256) | [0.090, 0.172] | 0.317 | 0.825 | 0.457 |
| **learned reference** | **−0.1685** | **0.0859 (22/256)** | **[0.055, 0.121]** | 0.244 | 0.679 | 0.473 |

Utility improves over no-op, so the pre-registered PASS condition is met:
**learned-control diagnostic = PASS**.

## 4. What this does and does not establish

**Does.** A learned policy roughly halves the catastrophe rate of the best
hand-coded probe (12.9% → 8.6% on the same 256 seeds) and cuts p90 damage from
0.825 to 0.679. The heavy tail is *reducible*. P8's heuristic probe is therefore
not an adequate proxy for controllability on this fragment — which is a finding
about the gate, not an excuse for the task.

**Does not.** The 95% interval [5.5%, 12.1%] **crosses the 10% threshold**. The
point estimate is below it; the data do not establish that the true rate is
below it. Increasing N would narrow the interval, and N was deliberately not
increased after the result was seen — that would be selecting the sample size
against the outcome.

**Also does not.** This says nothing about whether the 10% threshold is the right
threshold. The diagnostic tests the probe, not the criterion.

## 5. Consequences for use

* TDP-43 stays in the main set, labelled the hardest task, with the failure and
  the diagnostic printed side by side in `instruction.md`.
* Anyone benchmarking on the v8 set should expect higher variance on this task
  than on the other four, and should not read a single-run difference here as a
  capability difference.
* If a stricter set is wanted, the four clean tasks (Aβ42, α-synuclein, tau, HTT)
  form it, and TDP-43 becomes an out-of-distribution held-out task.

## 6. Governance

* Nothing was tuned after any result was seen: not the reward, the damage model,
  the thresholds, the chemistry, the architecture or the training budget.
* N was fixed before the confirmatory run and not extended afterwards.
* The final-test seeds used for the frozen anchors (900000+) are disjoint from
  every seed used above and were never inspected.
* Raw data: [`tdp43_learned_control_diagnostic.json`](tdp43_learned_control_diagnostic.json)
  (N=64), [`tdp43_precision_audit_N256.json`](tdp43_precision_audit_N256.json)
  (N=256).
