# P5.15 Addendum 39: the multi-scenario pilot certifies; R = 0.937 confirmed at 0.942; α = 0.50 is a dispersing regime. Stop for review

**Planner report, 2026-09-24.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 36, 38, 39.
- **Frozen spec:** v22, `data/SRP1/Results/P515S52/frozen_s52_spec_v22_5d8df1e8.json` (`1c80ea55`). Every prediction below was recorded in it before the corresponding run.
- **Instance:** the paper's 5 representative years × 4 days × **2 market × 2 operation** scenarios, row 18 active at **α = 0.50**, baseline ageing, AA-on, campaign cost file, €1M budget. 80 network blocks + 3 ESSO, 83 solves per cycle.
- **Objective convention:** Q = certified `gross_operational_cost` with the **contracted** settlement and the solver-only voltage pin excluded; the row-18 charge and the settlement **deviation** part are inside Q (Addendum 38 C).
- **Status:** nothing is running. Stopped for review, as spec v22 ordered.

## 1. Verdict

1. **Both pilot evaluations certified** — x = 0 at 74 cycles, the smallest node-7 unit at 72 — in **4 hours** for the pair at concurrency 2, against my predicted 6–14 h.
2. **The R = 0.937 prediction is confirmed at 0.942.** Recorded before any multi-scenario run and derived purely from the mean-profile market spread, it lands within 0.5 points of the measured ratio.
3. **α = 0.50 is a dispersing regime, not a schedule-honouring one.** Certified interface dispersion reaches **16 % of mean flow** (peak 34.5 MW). Addendum 39's premise is refuted, and my pre-run prediction that it would be non-zero holds.
4. **The storage still does not pay** at 2×2 under the baseline flexibility price: value − I = **−73,636**, determinate.
5. **Two real bugs were caught by the pre-run audit**, both invisible at one scenario and both now fixed and gated.

## 2. The pilot (`298e58f0`, `ae951d31`, `e52c8135`, `bc937cde`, `f2174d7c`, `bb52aeaf`, `f239ee02`)

| | x = 0 | node 7, 0.25 MVA / 1.0 MWh |
|---|---|---|
| certified at cycle | 74 | 72 |
| Q (gross, settlement-excluded) | 841,964,496.83 | 841,720,175.54 |
| bar | 8,403.29 | 7,107.55 |
| rule-ten terminal ratio | 0.0056 | 0.0112 |
| σ ratio (band [1/3, 3]) | 0.4894 | 0.4789 |

**Value and the R prediction:**

| quantity | value |
|---|---|
| value = Q(0) − Q(unit) | **244,321** (resolution 15,511 — determinate) |
| value − I | **−73,636** (determinate) |
| SRP1 baseline value | 259,428 |
| **pilot / SRP1** | **0.9418** |
| **pre-registered R** | **0.937** (difference 0.0048) |

The R prediction came from the paper-scale mean price profile (91.7 against SRP1's 98.0 €/MWh 4 h spread) and was committed before any multi-scenario evaluation existed. It also carried a caveat — that the storage prices at the bus-7 marginal cost rather than the market spread — which could have broken it. It did not: **the mean-profile argument transfers**, which is what licenses using SRP1 results with a stated scenario caveat if paper scale proves unaffordable.

**Interface dispersion at α = 0.50, certified and settled** (the quantity the 2-cycle sweep could not produce):

| measure | value |
|---|---|
| E\|d\| summed over blocks | 1,786 MWh/day |
| Σωd² summed over blocks | 14,756 MW²h |
| peak block RMS | 9.69 MW |
| peak \|d\| | 34.51 MW |
| worst block RMS as a share of mean flow | **16.1 %** (node 5) |
| row-18 charge summed over blocks | 82,687 |

**Identities and checks:** all capture checks pass on both evaluations; the settlement split reconciles per block to **1e-15**; the covariance recomputed independently from the models matches the captured deviation to 1.8e-15. The aggregate residual (−566,865) sits close to the covariance (−543,142), the ~24k gap being the priced consensus residual at the certification tolerance — which is what Addendum 38 (C) says it should be.

**Scaling law, third and fourth points:**

| instance | blocks × scenarios | cycle time | per evaluation |
|---|---|---|---|
| SRP1 | 48 × 1 | 32 s | ~1 h |
| pilot 2×2 | 80 × 4 | **~3.3 min** (concurrency 2) | **~4 h for the pair** |
| paper 5×5 | 80 × 25 | 42.3 min | ~78 h (projected) |

## 3. What the pre-run audit caught

The Addendum 36 instruction to audit the manuscript outputs' scenario indexing before running paid for itself. Both defects are invisible at one scenario, and both would have corrupted headline outputs:

1. **The Excel "Scenario Dispersion" sheet read the per-scenario storage copies that row 18 left unwired**, reporting a spurious **98.75 MW** where the truth is 0.
2. **The S31C flexibility volumes counted the scenario-free TSO deviation once per scenario**, inflating them by the scenario count.

Both fixed, both verified bit-identical at SRP1, and the whole child-path change gated by a two-cycle SRP1 bitwise run (`e75a575e`): 0 diffs over 29 trajectory fields, 153 solves declared and observed, and a whole-arm comparison against the previous gate's arm showing the S31C output byte-identical.

## 4. α: what the standalone threshold was worth, and what replaced it

**Ruling 2 asked for α\* and a bracketing R3.6 row. The single-block answer is α\* = 0.1 — and it does not transfer.**

- **Standalone (`c0d941f2`, `d0d6d8d1`, `2b7d8647`, `8b5a1c38`):** dispersion 3.222 → 1.464 → 0.707 → 0.095 → 1.9e-4 MW at α = 0 / 0.01 / 0.02 / 0.05 / 0.1; α\* = 0.1, bracket (0.09, 0.1].
- **The mechanism is not the one the spec assumed.** Unpriced upward flexibility is ~0.001 MWh/day throughout. The DSO holds its schedule by **curtailing RES at a flat 1 €/MWh**, and α\* ≈ κ / min π̄ — set by one cheap hour and an arbitrary constant. **The "unpriced upward flexibility" convention in spec v22 is refuted and must not reach the manuscript.**
- **That economy is the *initialisation* solve's** (`f62043a3`, `bf28ee55`): there the interface import is **unpriced** (settlement weight 0, reference generator excluded from generation cost), so curtailment looks cheap. From cycle 1 the settlement prices the import and the explicit curtailment penalty is 0.
- **Under coordination the premium does something different and cleaner** (`4483fdab`, `6dc87aa3`, `93abd2eb`, `74923c25`):

| α | market-arbitrage part of Σωd² | operation part | covariance earned |
|---|---|---|---|
| 0 | 31,446 | 3,215 | −6,990 |
| 0.25 | 221 | 484 | −215 |
| 0.5 | 123 | 499 | −102 |
| 1 | 107 | 666 | −50 |

  At α = 0 the deviation is **91 % market-scenario arbitrage**: the DSO shifts import between price scenarios and earns the covariance. The premium suppresses exactly that — a 300-fold fall in the market part and a 140-fold fall in the earned covariance.
- **α\*(coordinated) could not be located, and 2-cycle arms cannot locate it.** The sweep stopped twice on non-monotonicity, both times in the *operation* part of three blocks. At cycle 2 the coordination gap dominates: the implied augmented-Lagrangian price reaches ~225 €/MWh at the misbehaving hours against a premium of ~5–10 €/MWh. **This is a methodological result, not a failure:** any α threshold read off unsettled arms is an artefact of the coordination state.

**Consequences for R3.6:** the ordered row {0, α\*/2, α\*, 2α\*, 0.5} built on α\* = 0.1 lies **entirely inside the dispersing regime** and would show no transition. A settled sweep is the only honest route, at roughly 36–72 min per α on the 2×2 instance (2–4 h for three or four points).

## 5. Manuscript wording (replacing the refuted sentence)

Proposed, to be confirmed:

> Each DSO commits to a day-ahead interface schedule and settles per-scenario deviations at the scenario price plus an imbalance premium α·π̄. Without the premium, the deviation is predominantly *market-price arbitrage*: the DSO shifts its import between price scenarios and earns the price–deviation covariance. The premium prices that arbitrage away — at α = 0.5 the earned covariance falls by two orders of magnitude relative to α = 0 — while the physical flexibility volumes are essentially unchanged. Residual dispersion at α = 0.5 is driven by the operation scenarios, which differ physically.

The earlier explanations — unpriced upward flexibility, and curtail-and-reimport — are both **withdrawn**: the first is contradicted by the measured flexibility legs, the second belongs to the uncoordinated initialisation economy.

## 6. Open items and rulings needed

1. **R3.6 α row.** Rebuild it on a settled coordinated sweep (2–4 h), or report α = 0.50's certified dispersion alone with the transition left unmeasured?
2. **Row 18 at initialisation.** The uncoordinated benchmark deliberately passes α = 0 ("no commitment to deviate from, no settlement"), but the ADMM initialisation passes the run's α, so the init DSO is charged for deviating in an economy with no settlement. The Advisor's minimal fix is to keep row 18 inactive at initialisation and activate it with the settlement weight — inert at one scenario, so the SRP1 bitwise gate would still hold, but it invalidates the α > 0 gate artifacts. Do it, or document and leave?
3. **Paper-scale route.** The pilot gives the cost anchor: ~3.3 min/cycle at 4 scenarios against 42.3 min at 25. Route (d) — the full instance serially here, two evaluations — remains ~78 h each. Route (c), a 3×3 reduction, now looks like ~9 min/cycle, i.e. ~16 h per evaluation. **And the confirmed R prediction strengthens the analytical fallback**: the mean-profile argument transfers, so SRP1-with-caveat is defensible if neither route is affordable.
4. **F2 certificate** (ruling 1) is specified and ready: the standard unit poll with snap-to-feasible, completion only if fewer than n + 1 feasible points remain, cap 60. It runs next unless you redirect.
5. **Models:** Worker and Planner are pinned to `claude-opus-5-5` (Planner effective at its next restart; this session ran `claude-opus-5`), Advisor on Fable 5.1. No file change was needed — the pins were already present. Recorded by date in spec v22.

## 7. Evidence

| item | commit |
|---|---|
| spec v22 | `1c80ea55` |
| α threshold: single block, fine grid, mechanism | `c0d941f2`, `d0d6d8d1`, `2b7d8647`, `8b5a1c38` |
| curtailment pricing by path; mechanism analysis script | `f62043a3`, `bf28ee55` |
| coordinated sweep and the committed decomposition | `4483fdab`, `6dc87aa3`, `93abd2eb`, `74923c25` |
| σ check for the pilot | `c8f4b4c7` |
| pilot build, checks, instance and specs; re-freeze without persistence | `298e58f0`, `ae951d31`, `e52c8135`; `bc937cde`, `f2174d7c` |
| SRP1 bitwise gate of the child-path change | `fe0319d6`, `e75a575e` |
| pilot repro; **the pilot** | `bb52aeaf`; **`f239ee02`** |

Every zero-solve claim is backed by an armed `SolveProfileGuard`. Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the manifests.
