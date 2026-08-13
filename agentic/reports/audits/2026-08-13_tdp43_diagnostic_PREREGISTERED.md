# TDP-43 — pre-registered post-hoc diagnostic

Written **before** the run. Thresholds below are fixed and will not be revisited
after the result is seen.

## Status being preserved

* `P8_original = FAIL` — the heuristic porting gate failed and **remains failed**.
* The catastrophe threshold remains **< 10%**.
* This run is **diagnostic only**. It is not a P8 rerun and cannot convert
  `P8_original` into a PASS.
* Nothing may be tuned after the result: not the reward, the damage model, the
  thresholds, the chemistry, or the gate status.

## What is already established (heuristic control)

Six hand-coded policies were tested on 48 dev seeds; none reached < 10%:

| policy | catastrophe |
|---|--:|
| no-op | 0.646 |
| oracle continuous s=1.0 | **0.167** (best) |
| every 2 steps, s=1.0 | 0.208 |
| every 3 steps, s=1.0 | 0.250 |
| every 2 steps, s=0.6 | 0.354 |
| steps 0–39 only | 0.417 |
| every 2 steps, until 60 | 0.438 |

Damage under the best heuristic is heavy-tailed: median 0.106, but 8 of 48
episodes exceed 0.75, reaching the 2.5 cap.

## Question this diagnostic asks

Aβ42's *trained* reference reached catastrophe 0.000 where its heuristic oracle
reached 0.021. So learned control can be materially better than the crude probe.
The question is narrow:

> Does a learned policy, trained with the identical pipeline and budget used for
> the other four tasks, bring TDP-43's catastrophe rate under control — or is the
> heavy tail irreducible?

This tests whether **P8's heuristic probe** is an adequate proxy for
controllability. It does not test whether P8's threshold is correct.

## Method

* Same trainer, same architecture, same budget as the other four tasks:
  sep-CMA-ES, 12000 episodes, 2 restarts, 8 training seeds, selection on the
  public validation split.
* Evaluation on **fresh seeds 4000–4063**, disjoint from train (1000–1063),
  validation (2000–2031), the gate's dev range (2000–2047) and calibration
  (3000–3063).
* Reported: utility, catastrophe rate, damage, pathology, against no-op and the
  best heuristic oracle on the same fresh seeds.

## Decision rule (fixed now)

| learned catastrophe | outcome |
|---|---|
| **< 10%** and utility improves over no-op | Keep TDP-43. Report as: heuristic P8 **FAIL**, learned-control diagnostic **PASS**. Both stated together, never the second alone. |
| **10–15%** | **INCONCLUSIVE.** TDP-43 not frozen pending a decision. |
| **> 15%** | **FAIL.** Do not freeze TDP-43. Ship four tasks. |
