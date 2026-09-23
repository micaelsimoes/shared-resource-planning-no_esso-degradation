# P5.15 Addendum 38: F2 finds a better plan than the corner; row 18 implemented and gated. Two rulings needed

**Planner report, 2026-09-23.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 38 (rulings A–D), with Addenda 35–37.
- **Frozen spec:** v21, `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json` (`a3ad7699`). All predictions were recorded in it before the runs.
- **Objective convention on every table:** Q = certified `gross_operational_cost`, settlement excluded; F = I + Q; value = Q(0) − Q(x).
- **F2 = the flexibility-price scenario (multiplier ×2).** Every F2 table is labelled a MODEL VARIANT and is never mixed with the baseline.
- **Status:** nothing is running. Two rulings are needed before the pilot (§5).

## 1. Verdict

1. **The method found a better plan than any hand-picked candidate.** F2's Phase B moved off the 2025 budget corner to a **2030 two-node plan**, improving F by **102,117** at 5.5× the resolution. This is the paper's non-trivial optimal plan, obtained by the full machinery.
2. **Two recorded predictions are refuted** — mine and the author's. The optimum is not the budget corner, and the ladder supplied **no** cached neighbours.
3. **Row 18 as signed is implemented and passes all three gates.** It is provably inert at one scenario, so every committed result stands unchanged.
4. **At α = 0.50 the DSOs already hold their schedules**: measured dispersion collapses from 3.22 MW (α = 0) to ~1e-6 MW. The pilot's dispersion result will be ≈ 0 — but now for a **priced** reason, which is interpretable, unlike the ≈ 0 the old quadratic produced mechanically.

## 2. F2 ladder (`a3a9fb47`): every marginal MWh pays

Node 7, 4 h, 2025, under the ×2 flexibility price. Three new points (E = 3, 4, 5); E = 0, 1, 2 are pinned from committed runs and reproduce bitwise.

| E (MWh) | cycles | value | value − I | resolution | the marginal MWh |
|---|---|---|---|---|---|
| 1 | 132 | 366,353 | +48,396 | 4,847 | pays |
| 2 | 144 | 727,076 | +91,162 | 1,742 | 360,723 vs 317,957 — pays |
| **3** (budget corner) | 160 | 1,080,058 | **+126,187** | 2,493 | 352,982 — pays |
| 4 | 136 | 1,444,761 | +172,933 | 3,828 | 364,703 — pays |
| **5** (capacity cap) | 170 | 1,813,641 | **+223,856** | 1,573 | 368,880 — pays |

- **Predictions held at 3 and 4 MWh** (+126,187 against a predicted 125–140k; +172,933 against 150–175k). At 5 MWh the measured +223,856 **exceeded** the predicted 175–200k.
- **The successive MWh values are not monotonically decreasing** (360,723 / 352,982 / 364,703 / 368,880): the fourth and fifth MWh are worth more than the third. The value is not simply concave in this régime.
- Certification took 160 and 170 cycles at the two largest cells, above the usual 110–140, with no barrier points.

## 3. F2 Phase B (`2e2ef65d`, `c39c4836`, `0012d85b`): the search leaves the corner

| poll | step | incumbent | outcome |
|---|---|---|---|
| 0 | 4 | 2025, node 7, 3 MWh | no feasible direction |
| 1 | 2 | same | no feasible direction |
| **2** | **1** | same | **success — 20 evaluated, 6 improvements** |
| 3 | 2 | new incumbent | failure |
| 4 | 1 | new incumbent | **stopped: completion 61 > cap 30** |

**The accepted plan:** 2030, **0.25 MVA / 0.5 MWh at node 5 plus 1.0 MVA / 3.5 MWh at node 7**, I = 992,268 (budget slack 7,732), F improved by **102,117** against a resolution of 18,450 (5.5×). value − I rises from +126,187 to **+228,304**.

**All six improving neighbours were 2030 designs.** The mechanism: discounting lowers I per MWh at 2030, so the same €1M buys materially more capacity, and at the ×2 flexibility price that extra capacity outweighs investing earlier. Under the *baseline* price the year ladder went the other way (delay lost value faster than cost) — the ordering reverses once storage is worth more than its alternative.

**Predictions refuted, both recorded before the run:**
- *"The optimum under the budget is the budget corner, 3 MWh."* It is not; the corner is beaten by 102,117.
- *"Phase B terminates with few new evaluations, since the ladder supplies most cached neighbours."* **0 of 20** were cache hits: the ladder's rungs are 2 lattice units apart, so no rung is a unit neighbour of another. All 20 feasible neighbours were off-duration (2.5/3.5 h) or 2030 points.

**The stop is by design, not a failure.** At the new incumbent the unit poll's completion has **61** feasible neighbours against the cap of 30, so the poll was refused rather than truncated (Planner ruling A2, frozen in the s47 launcher). 20 new evaluations, 0 barrier points, 5 polls. The certificate was **not** reached: mesh-local optimality of the new plan is **not** established.

## 4. Row 18 as signed (`700cf13c` merged; gates `4ce7447b`, `c9b39bf5`, `59476bff`)

Implemented exactly to Addendum 38's rulings, in an isolated worktree so that no production change reached the F2 campaign while it ran.

**Implementation.**
- **The charge**: linear, inside `objective_function_rule` (so it enters Q(x)), LP form with d⁺/d⁻ ≥ 0 against the DSO block's own coupled expectation, premium c_t = α·π̄_t hourly.
- **(A) TSO pinned**: its per-scenario interface deviations are fixed at zero — 144 first-pair entries free, all 432 other copies fixed.
- **(B) Storage**: one scenario-free variable per period referenced by every scenario's balance row, on all four block types; 72 per-scenario copies per family retained **unwired**, covering charge and discharge separately. Implemented as an alias of the first scenario pair rather than a new variable, so one-scenario column names are unchanged — the reason the bitwise gate can hold.
- **(C) Settlement split**: contracted part cancels as a transfer; the deviation part enters Q(x) with the premium. Measured on a built 2×2 block: settlement 66,032.760 = contracted 64,644.291 + deviation 1,388.469, against an independently recomputed covariance of 1,388.4689357747982 — difference **1.1e-12**. The TSO's deviation part is 3.8e-10, as the pinning requires.
- **(D) Voltage pin** kept as a solver-only term excluded from Q(x); the per-scenario mismatch is reported.
- The old quadratic is **unwired, not deleted**, with an armed tripwire showing 0 calls.
- **α defaults to inactive twice over**: 0.0 by default, and at one scenario the rows are not constructed at all.
- **Premium floor not needed**: the minimum mean hourly price is 3.26 (SRP1), 9.81 (2×2) and 6.81 (paper) €/MWh — checked first, no non-positive hour anywhere.
- **Polish asymmetry removed**: `model.objective` now equals `objective_function_rule` on all 32 blocks (max difference 0.0), restoring "polish Δ ≤ 0 by construction", which W37 had found fails above 1×1.
- 10/10 zero-solve checks; a finite-difference check of the premium at max relative error 0.0; all 93 tracked fixtures still unpickle.

**The three gates, all passed.**

| gate | result |
|---|---|
| SRP1 two-cycle bitwise identity | **PASS** — 0 diffs against the committed C\* reference, all trajectory fields identical, 153 solves declared and observed, guard exact, retired quadratic called 0 times |
| single-block A/B, α ∈ {0, 0.5, 1000} | **PASS** — dispersion 3.221 MW → 8.6e-7 → 2.5e-12; row 18 present iff α > 0; 12 solves verified exactly |
| 2×2 limit check under full ADMM | **PASS** — α → large collapses dispersion to 8.4e-10 MW, ratio 2.5e-10 against the α = 0.50 arm; cycle 46.7 s at 2×2 |

**The finding that matters for the pilot.** At α = 0.50 the premium is **below** each DSO's flexibility price in 9 of 24 hours, yet deviation still collapses to ~1e-6 MW. The reason is the unpriced **upward** flexibility found earlier: a DSO can shift load up for free, so its true cost of holding the schedule is well below `cost_flex`, and even a sub-`cost_flex` premium does not induce deviation. Consequences:
- the pilot's dispersion at α = 0.50 will be ≈ 0 — **priced, not mechanical**, and reportable as such;
- the ordered R3.6 sensitivity **{0.25, 0.5, 1.0} will likely read ≈ 0 at every point**; the threshold where deviation appears lies **between 0 and 0.5**, and locating it costs seconds at single-block level.

## 5. Two rulings needed

1. **Phase B stopped holding a better plan.** Either (a) continue from the new incumbent with a raised completion cap and evaluation budget — the completion is 61, so a cap of ~70 and a budget of ~80 would let the next poll run, at roughly 20 h for 61 evaluations at concurrency 5; or (b) report the 2030 two-node plan as the incumbent with *mesh-local optimality not established*, which is honest and costs nothing. My recommendation is (b) for the manuscript, with (a) only if a certificate is wanted for the headline plan.
2. **The α sensitivity.** Keep {0.25, 0.5, 1.0} as ordered, knowing it will likely read ≈ 0 throughout, or replace it with a range that brackets the measured threshold (e.g. {0.05, 0.1, 0.25, 0.5}) so the row shows where deviation actually appears? The single-block A/B can locate the threshold in seconds before the pilot commits 8–10 hours per evaluation.

## 6. What is ready

- **The pilot** (Addendum 36) at α = 0.50: 5 representative years, 4 days, 2×2 scenarios, x = 0 and the smallest node-7 4 h unit, concurrency 2, hull polish on both. Measured cycle time at 2×2 is 46.7 s under coordination, so ≈ 3–4 min per pilot cycle and ≈ 7–10 h per evaluation, at ≈ 3.9 GiB per process.
- **Everything row 18 needs is merged and gated**; the multi-scenario code paths were separately gated at 2×2 (8/8) in the previous round.

## 7. Evidence

| item | commit |
|---|---|
| spec v21 | `a3ad7699` |
| F2 ladder: launcher, spec, run | `6435adf7`, `a0714117`, `6a6da686`, `a3a9fb47` |
| F2 Phase B: launcher, spec, run | `2e2ef65d`, `c39c4836`, `0012d85b` |
| row 18: implementation, zero-solve checks, gate scripts, NL probe | `1d310565`, `d3c7442f`, `7fd8e5dd`, `63a2cadd` (merged at `700cf13c`) |
| gates: SRP1 bitwise, single-block A/B, 2×2 limit | `4ce7447b`, `c9b39bf5`, `59476bff` |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the manifests.
