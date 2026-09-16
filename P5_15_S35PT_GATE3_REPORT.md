# P5.15 Addendum 16 item 3 — gate s35pt (price-taker initialization): FAIL

**Planner report, 2026-09-16.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 16 item 3. Frozen spec v6
`data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json` (`a993b088`). Implementation `fecba762` (phase 1),
`52d70dc9` + `4ea7da2e` (phase 2). Launched at `4ea7da2e`. Evidence `8c1cff39`. **Stops for review; nothing further
started.**

System cost = `gross_operational_cost`, settlements excluded; salvage 0, so gross = net. Instance C\*. Reference is run 1
(`s35ref`, `ed1377c4`): Boyd stop at cycle 477, cost 651,039,166, EFC/day 1.0590.

## 1. Verdict

**Gate 3 FAILS.** Under an independent re-evaluation from both trajectories (`s35pt_evaluation_v2.json`, zero solves):

| criterion | result | reading |
|---|---|---|
| (a) Boyd stop within 150, no clamp | **FAIL** | cap reached; terminal ratios V 0.156, **PF 1.415**, **storage 1.405** — two channels above threshold |
| (b) cost within the rule-nine bar | **INDETERMINATE** | 651,260,030 vs 651,039,166 (Δ 220,864; bar 5,992) — but gate 3 **did not settle**, so the bar does not apply |
| (c) EFC/day within 2 % | **not a fixed-point test here** | 1.1842 vs 1.0590 (11.8 %) — the runs approach from opposite directions and one never stopped |

**Correction to my own evaluator.** Its first output marked (b) and (c) as well posed, because it checked only that the
*reference* had stopped. v1 is retained unchanged; v2 requires both runs to have stopped under Boyd, records v1's hash,
and reports (b) as indeterminate and (c) as not a fixed-point test.

Run facts: exit 0, wall 5,231 s, 7,772 solves, 68 network failures (65 T1, 3 T2, 0 unrecovered) = **0.45 per cycle**
against run 1's 0.625, 0 local-solve failures, objective rule ten 0.088 (still descending, −25,404 over cycles
120–150), SoH floor slack (minimum SoH 0.594, multiplier 0), settlement cancellation holds.

## 2. What the initialization did and did not do

**Verified and effective at the start.** z equalled the LP schedule on 864/864 cells before cycle 1. EFC/day was 1.1918
at cycle 1 (run 1: 0.0003). The storage dual ratio at cycles 1–3 was 0.35 / 0.39 / 0.57 (run 1: 19.4 / 17.8 / —).
System cost was **below run 1 at every matched cycle** — 904.9 vs 908.0 M at cycle 1, 651.53 vs 652.06 at 50,
651.26 vs 651.34 at 150 — and failures were fewer.

**It did not remove the storage walk; it moved it from the schedule into the storage prices.** The mechanism below was
proposed by independent review and verified by the Planner against the trajectories.

- **The price-taker schedule lies above the coordinated equilibrium.** The storage operator's objective holds only slack
  penalties and throughput regularization; all of storage's economic value sits in the network objectives, and the LP
  at market price π ignores network limits and losses. A coordinated equilibrium below the price-taker schedule is the
  expected direction, and gate 3's steady downward EFC drift confirms it: 1.1918 (c1) → 1.1894 (c50) → 1.1868 (c100)
  → 1.1842 (c150).
- **The binding path is the build-up of the storage duals, which started at zero.**
  - The storage dual norm ‖y‖ grows linearly at **5.63e-6 per cycle**: 2.383e-3 at cycle 113, 2.592e-3 at 150.
  - Run 1's is steady at **8.511e-3** over its final cycles. At gate 3's rate, closing that gap takes about **1,050
    cycles**.
  - The storage primal residual sits at ≈6.8e-5 with the same sign every cycle, which is **transit, not a fixed point**
    (a fixed point needs r → 0 and y constant).
  - Each cycle adds ρ·r to the duals; with agents agreeing to within ~g/ρ, the duals grow by ~g per cycle **whatever ρ
    is** — the same ρ-invariant forcing seen in `s33e2`.
  - The zero-dual start was the recommendation of the design review, which it now identifies as the bottleneck.
- **The storage dual ratio is therefore stuck near 1.4:** 1.455 (c10), 1.56–1.61 (c30–60), 1.405 (c150), decaying only
  ×0.972 per 30 cycles.
- **PF was on track and was cut off by the cap.** Its dual ratio fell ×0.524 per 30 cycles to 1.415, roughly 15–20
  cycles from passing. ρ_pf was lowered once at cycle 58 (0.198 → 0.132) and frozen at the cycle-60 backstop.

## 3. Retractions

- **"The storage walk is gone"** (`P5_15_S35PT_PRERESULT_NOTE.md`, and my status messages) — **withdrawn**. It was gone
  from the schedule for the first cycles; it persisted in the dual build-up.
- **The pre-result note's prediction was only half right.** PF did bind, but the note's PF-only framing does not cover
  the storage failure (1.405), which it did not predict.

## 4. What can and cannot be said about storage use at the certified point

- **Cost and EFC are not pinned equally.** The terminal costs differ by 0.034 % while EFC differs by 12 %, but gate 3
  is still descending (rule ten 0.088, PF ratio 1.4), so **0.034 % is not a certified cost precision**. It is path
  divergence between an unsettled and a settled run.
- **The EFC bracket is one-sided, not symmetric.** Run 1 is a certified point with EFC still rising, its slope halving
  from 3.79e-4 (cycles 400–450) to 1.77e-4 (450–477). Gate 3 is a cap-stopped point still falling. If the limit is
  unique, it lies in **[1.059, 1.184]**. A fragile extrapolation of run 1 suggests ≈1.07–1.08.
- **Defensible statement:** at run 1's certified point, EFC/day is **≥ 1.059 and not identified from above** by the
  residual test.

**Claims the report and the paper must avoid:** EFC, storage utilisation, SoH or storage value "at the optimum"; a
symmetric ±6 % bound; "cost certified to 0.03 %"; "price-taker initialization removes the walk"; "a different fixed
point" (nothing in the data supports one); "over-relaxation will fix storage"; and gate 3's EFC per cohort-year as
equilibrium values.

## 5. The lever Addendum 16 names, and why it is unlikely to be enough

Over-relaxation (Boyd §3.4.3, α ≈ 1.5) scales the per-cycle dual increment by at most about α: roughly 1.5× on a
path that needs 10× or more. It would mainly shorten PF's decay, which needs only ~20 more cycles anyway, and its
guarantees are for convex two-block ADMM, not this three-agent weighted consensus with a TSO proximal term. The storage
outcome would plausibly be unchanged. **It is not started** (Addendum 16: "not now").

## 6. Proposed next steps, ranked by information per cost (no action taken)

1. **Zero-solve dual-direction comparison** (Planner-authorizable). Compare per-agent storage dual norms and, where
   serialized, their directions between gate 3 at cycle 150 and run 1 at 477. If gate 3's duals point toward run 1's,
   a single limit is supported; if orthogonal or stalled, multiple equilibria gain weight.
2. **Shadow-price LP** (Planner-authorizable if the nodal duals exist). Re-solve the price-taker LP with run 1's
   terminal nodal prices instead of π, LP only. If it returns EFC ≈ 1.06–1.08, the network mechanism is confirmed and
   the limit gets an independent estimate.
3. **Initialize the storage duals as well as the schedule, from shadow prices** — **author decision.** This targets the
   actual bottleneck, the dual build-up. A midpoint schedule with zero duals would recreate the slow build-up and is
   not history-free.
4. **A bounded extension of gate 3 from a checkpoint** to test whether the decline accelerates or stays linear —
   author decision; expensive if it must run to the meeting point.
5. **Report run 1 as the certified point with the one-sided EFC statement of §4** — author decision.
6. **ρ_pf balancing or freeze changes** (PF timing only; the backstop froze ρ_pf at slow values in both runs) — author
   decision. Over-relaxation ranks below items 1–3.

## 7. Still open from earlier

- **Success at C\*.** The floor is slack under the price-taker bound and both runs, so "equilibrium at the threshold
  with the degradation constraint active" cannot occur here. What is the success definition?
- **Insight (iii).** The degradation shadow price is zero at C\*. Should it be demonstrated where the floor binds?

## Evidence

`data/SRP1/Results/P515S35_PT_run/` (+ manifest, 479 files): `g_baseline.json`, `boyd_terminal.json`,
`component_levels_terminal.json`, `interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json`,
`soh_floor_sidecar_baseline.jsonl`, `recourse_jump_sidecar_baseline.jsonl`, `network_failures_baseline.jsonl`,
`leak_classification_baseline.jsonl`, stdout, heartbeat, ESSO pickle; `s35pt_evaluation.json` (v1, retained) and
`s35pt_evaluation_v2.json`; `P515S35_PT_launch.log`, `P515S35_PT_exit_code.txt`. Pre-result records:
`P5_15_S35PT_PREFLIGHT_DECISION.md` (`4ea7da2e`), `P5_15_S35PT_PRERESULT_NOTE.md` (`17d75055`). Reference:
`P5_15_S35REF_REPORT.md`, `P515S35_REF_run/`.
