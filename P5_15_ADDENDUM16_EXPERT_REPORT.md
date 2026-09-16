# P5.15 Addendum 16 — reference equilibrium and price-taker initialization: handoff to the Author and the External Expert

**Planner report, 2026-09-16.** Self-contained; continues `P5_15_STEP34_EXPERT_REPORT.md`. Authority:
`PLANNER_BRIEF_2026-09-13.md` Addendum 16. Instance throughout: candidate C\* (s = 0.96875 MVA, e = 3.875 MWh,
invested 2025, uniform across nodes 5/7/9). The configuration is the s34 re-scaling (σ fixed, D5 as scaled ESSO
coordination terms, S_ref = 2.5 MVA, γ = τρ) with ρ_ess starting at 0.1125. Every campaign ran attached and alone,
with stderr captured, a lock, a heartbeat and an exit-code file.

System cost = `gross_operational_cost`, settlements excluded; salvage is 0 in every run here, so gross = net.

## 1. Summary

| item | outcome |
|---|---|
| **Run 1 — reference equilibrium (cap 500)** | **Established.** First certified convergence of the programme: Boyd stop at cycle 477 |
| Z2 — is the SoH floor reachable at C\*? | **No.** Slack even under the price-taker upper bound |
| Price-taker initialization (implementation) | Done and verified on the real state |
| **Gate 3 — initialized run (cap 150)** | **FAIL** on its first criterion; the other two not valid as posed |
| Over-relaxation | **Not started** (Addendum 16: "not now") |

**Headline.** The price-taker start placed storage near its equilibrium from the first cycle and lowered cost at every
matched cycle, but it did **not** shorten the path to certification. It moved the slow approach from the storage
schedule into the storage **prices** (duals), which started at zero and build up by a fixed amount per cycle whatever
ρ is.

## 2. Run 1 — the reference equilibrium

Spec v5 (`f76c7574`); report `P5_15_S35REF_REPORT.md` (`ed1377c4`).

- **Boyd stop at cycle 477**, all three channels passing on cycles 475–477, inside the 500 cap, no ρ clamp.
- **System cost 651,039,166**; terminal objective step 256 against tolerance 65,104 (rule ten 0.0039): the cost is
  settled.
- **EFC/day 1.0590**; per cohort-year 1.058 / 0.894 / 0.682. 298 network failures (284 T1, 14 T2, none unrecovered),
  0 local-solve failures; wall 4 h 50 min.
- **Settling quality differs by channel.** V ended at 0.013 of threshold and PF at 0.090, but **storage stopped at
  0.988** (first passing at cycle 475) with EFC still rising (1.0542 at cycle 450, 1.0590 at 477). By the repository's
  own rule a channel ending at ~99 % of its threshold is *stopped*, not *settled*, so the storage schedule at the stop is
  a tolerance-level approximation of its limit.
- **Path dependence.** Starting ρ_ess at 0.1125 left ρ_pf unchanged at 0.198 until the cycle-60 backstop froze it,
  whereas s34 lowered it to 0.088. PF first passed at cycle 226 here against 133 in s34.

**A harness defect caught at evaluation.** The run's summary file records `stopped_by: "cap"`. The writer compared the
*first* cycle of the three-cycle convergence run (475) with the last (477). The first evaluation trusted that field and
reported "not established". It is retained unchanged; a v2 derives the stop from the trajectory and records v1's hash.
Earlier gates genuinely capped and are unaffected. The phase-2 gate arm derives the stop from the trajectory.

## 3. The degradation floor is slack at C\* (Z2)

Zero Pyomo/IPOPT solves; 756 LPs declared and matched exactly; the committed `EFC*` benchmark reproduced cell for cell.
Recorded in `P5_15_Z2_FLOOR_SLACK_NOTE.md` (`a993b088`) **before run 1 was launched**, including a revised prediction.

- On the harness's own EFC definition, with capacity wear iterated to convergence, the price-taker schedule gives
  EFC/day **1.192 / 1.189 / 0.965**. Terminal SoH is **0.589** against `soh_min` = 0.50, a margin of +0.089.
- The price-taker is an **upper bound** on storage use, so no coordinated equilibrium at C\* can reach the 1.4612
  threshold. The threshold itself was confirmed independently (recomputed 1.461187).
- **Run 1 confirmed it:** 0 of 18 floor rows active, minimum SoH 0.659, floor duals ~1e-10 (barrier level). **The
  degradation shadow price at C\* is zero.** Spec v5's original prediction ("EFC settles at the threshold, floor
  active") failed exactly as the pre-launch note said it would.

**Consequence:** Addendum 16's success definition — "equilibrium at the SoH threshold with the degradation constraint
active" — **cannot occur at C\***. It is returned to you, not redefined.

## 4. Price-taker initialization — implementation

Spec v6 (`a993b088`), reviewed before implementation.

- **LP.** A new Pyomo-free production module solves the price-taker schedule per node, coupled across years and days,
  with production efficiencies, capacity wear to a fixed point, and the SoH floor as **exact linear rows**. Its LP calls
  are counted separately from IPOPT solves.
- **What is initialized.** Initial values only, never fixes: the storage consensus z; the three agent copies; the
  **TSO proximal centres** (mandatory, otherwise the first move is halved); and the storage operator's SoH, energy and
  schedule.
  - **Storage duals start at zero.** A consistent non-zero value would need the TSO/DSO split of the gradient, which
    the LP does not provide.
  - **Network warm starts are left alone,** because the state of charge is reset every cycle.
- **Opt-in flag.** `admm.shared_ess_initialization`, defaulting to the previous behaviour, so other case studies are
  unchanged.
- **Verification.**
  - Phase 1 (`fecba762`): seven zero-solve checks, including exact reproduction of the committed `EFC*` and Z2 values.
  - Phase 2 (`4ea7da2e`): on the real state, captured immediately before cycle 1's first network solve, **864/864
    cells equal the LP**, storage duals are exactly zero, and all non-storage state is bit-identical to the standalone
    construction.
- **Preflight decision.** One signature ("no failures attributable to the initialization") was literally not met: a
  known-fragile block recovered once at cycle 2. The Planner recorded it (`P5_15_S35PT_PREFLIGHT_DECISION.md`) as not
  attributable — the same block failed 4× in run 1 and 3× in s34, and the rate was below background — and did **not**
  re-run for a pass.

## 5. Gate 3 — FAIL

Report `P5_15_S35PT_GATE3_REPORT.md` (`9cf5c1e0`); evidence `8c1cff39`. Evaluated independently from both runs'
per-cycle trajectories rather than any summary field.

| criterion (as authorized) | result | reading |
|---|---|---|
| (a) Boyd stop within 150, no clamp | **FAIL** | cap reached; terminal ratios V 0.156, **PF 1.415**, **storage 1.405** |
| (b) cost within the rule-nine bar | **INDETERMINATE** | 651,260,030 vs 651,039,166 (Δ 220,864; bar 5,992) — gate 3 never settled, so the bar does not apply |
| (c) EFC/day within 2 % | **not a fixed-point test here** | 1.1842 vs 1.0590 (11.8 %) — the runs approach from opposite directions and one never stopped |

The Planner's own evaluator initially marked (b) and (c) as well posed after checking only that run 1 had stopped. That
output is retained; v2 requires both runs to have stopped under Boyd.

**What the initialization achieved.**
- EFC/day **1.1918 at cycle 1** (run 1: 0.0003).
- Storage dual ratio 0.35 / 0.39 / 0.57 over cycles 1–3 (run 1: 19.4 / 17.8).
- **System cost below run 1 at every matched cycle** (904.9 vs 908.0 M at cycle 1; 651.26 vs 651.34 at 150).
- Network failures **0.45 per cycle** against 0.625.

**Why it still failed** — proposed by independent review, verified by the Planner against the trajectories:
- **The price-taker schedule lies above the coordinated equilibrium.** All of storage's economic value sits in the
  network objectives; the LP at market price ignores network limits and losses. Gate 3's EFC drifted steadily down:
  1.1918 → 1.1894 (c50) → 1.1868 (c100) → 1.1842 (c150).
- **The binding path is the build-up of the storage duals from zero.**
  - Their norm grows linearly at **5.63e-6 per cycle** (2.383e-3 at c113, 2.592e-3 at c150), toward run 1's steady
    **8.511e-3**: about **1,050 cycles** at that rate.
  - The storage primal residual holds at ≈6.8e-5 with constant sign, which is **transit, not a second fixed point**.
  - Because agents agree to within ~g/ρ, the duals grow by ~g per cycle **whatever ρ is** — the same ρ-invariant forcing
    found in s33e2.
  - The storage dual ratio therefore sat near 1.4 throughout (×0.97 per 30 cycles).
- **PF was on track and was cut off by the cap:** ×0.52 per 30 cycles, roughly 15–20 cycles from passing.

**Retracted:** "the storage walk is gone" (only the schedule was right; the prices were not). The Planner's pre-result
note predicted a PF-only failure; storage failed too.

## 6. What can and cannot be claimed

- **EFC is not pinned by the certified stop.** At run 1's certified point EFC/day is **≥ 1.059 and not identified from
  above**. If the limit is unique, it lies in **[1.059, 1.184]**; a fragile extrapolation of run 1 (slope halving from
  3.79e-4 to 1.77e-4 per cycle) suggests ≈1.07–1.08. The bracket is **one-sided**, not a symmetric ±6 %.
- **Cost is not certified to 0.03 % either.** The 0.034 % difference is path divergence between a settled run and one
  still descending (rule ten 0.088).
- **To avoid in any report or the paper:** EFC, storage utilisation, SoH or storage value "at the optimum"; a symmetric
  EFC bound; "cost certified to 0.03 %"; "price-taker initialization removes the walk"; "a different fixed point"
  (nothing supports one); "over-relaxation will fix storage"; gate 3's EFC per cohort-year as equilibrium values.

## 7. On over-relaxation

Addendum 16 names over-relaxation (Boyd §3.4.3, α ≈ 1.5) as the next lever if gate 3 fails. It scales the per-cycle
dual increment by at most about α — roughly **1.5×** on a path that needs **10× or more**. It would mainly shorten PF's
decay, which needs only ~20 more cycles, and its guarantees are for convex two-block ADMM, not this three-agent
weighted consensus with a proximal term. **The storage outcome would plausibly be unchanged. Not started.**

## 8. Decisions requested

**Next steps, ranked by information per cost:**
1. **Zero-solve dual-direction comparison** between gate 3 (cycle 150) and run 1 (cycle 477): per-agent storage dual
   norms and, where serialized, directions. Tests whether both runs head to one limit. *Planner can run on your
   approval.*
2. **Shadow-price LP:** the price-taker LP with run 1's terminal nodal prices instead of π, LP only. EFC ≈ 1.06–1.08
   would confirm the network mechanism and give an independent estimate of the limit. *Planner can run on your
   approval.*
3. **Initialize the storage duals as well as the schedule, from those shadow prices.** This targets the actual
   bottleneck. A midpoint schedule with zero duals would recreate the same slow build-up and is not history-free.
   *Author decision.*
4. **Alternatives:** a bounded extension of gate 3 from a checkpoint; reporting run 1 as the certified point with the
   one-sided EFC statement of §6; or ρ_pf balancing and freeze changes (PF timing only). *Author decisions.*
   Over-relaxation ranks below items 1–3.

**Questions still open:**
5. **Success at C\*.** The floor is slack under both the price-taker bound and the coordinated equilibrium. What is the
   success definition at this candidate?
6. **Insight (iii).** The degradation shadow price is zero at C\*. Should it be demonstrated at a candidate or price
   profile where the floor binds?
7. **What the paper reports for storage.** Given §6, should storage use be reported only as a one-sided bound at the
   certified point, with cost as the certified quantity?

## 9. Evidence index

| item | artifact | commit |
|---|---|---|
| prior handoff | `P5_15_STEP34_EXPERT_REPORT.md` | `5f1a16dc` |
| spec v5 (run 1) | `P515S35/frozen_s35_reference_spec_v5_995548ab.json` | `f76c7574` |
| Z2 floor slackness | `p515_s35_z2_floor_slackness.py`, `P515S35/Z2/`, `WORKER_REPORT_S35_Z2.md` | `49f24636` |
| floor-slack note and spec v6 | `P5_15_Z2_FLOOR_SLACK_NOTE.md`, `P515S35/frozen_s35pt_spec_v6_651a9d84.json` | `a993b088` |
| run 1 preparation | case file ρ_ess 0.1125; `s35ref` arm and preflight | `55d383ae`, `d0f1f218` |
| **run 1 report and evidence** | `P5_15_S35REF_REPORT.md`, `P515S35_REF_run/` (+ manifest) | `ed1377c4` |
| initialization phase 1 | `shared_ess_price_taker.py`, wrapper, checks | `fecba762` |
| initialization phase 2 | case-file flag; `s35pt` arm, Z3/Z4/Z7, preflight, decision note | `52d70dc9`, `4ea7da2e` |
| independent gate-3 evaluator; pre-result note | `p515_s35pt_evaluate.py`; `P5_15_S35PT_PRERESULT_NOTE.md` | `96eaeaae`, `17d75055` |
| **gate 3 evidence and report** | `P515S35_PT_run/` (+ manifest); `P5_15_S35PT_GATE3_REPORT.md` | `8c1cff39`, `9cf5c1e0` |

Every analysis script above runs with `SolveProfileGuard` armed and verified (zero permitted solves), except the Z2 and
price-taker LPs (scipy, declared counts, zero Pyomo/IPOPT entries) and the preflights, which declare exact solve counts.
Large capture directories (`esso_capture/`, `results/`, the 80 MB storage stride file) are hash-recorded in the
manifests rather than committed.
