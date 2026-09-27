# TASKS — current order

**Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 48 (2026-09-25).
**Objective convention on every value:** Q = certified `gross_operational_cost`, settlement excluded.
**Updated at every transition; read first when resuming.**

## Addendum 48 order

- [x] **(b) zero-solve measurement** — `release_solution_bookkeeping` saving ≈ 3.98 GiB/child predicted (range 2.92–3.98); does not enter `evaluation_key` — commit `9db1c45b` (W90)
- [x] **Spec v35 freeze** — `8aa98dbf`, commit `60c9b7b1` — *superseded by v36* (margin rule mixed RSS with footprint; G6 V5 vacuity hole)
- [x] **Spec v36 freeze** — `14bbddc7`, predecessor v35 `8aa98dbf`; commits `eac9c6c7` (launcher), `d46f2d8c` (spec) — footprint-consistent persistence margin rule; G6 ≥ ⌈0.9·B⌉ floor-testable finals; 3.09e-6 provenanced to `P515S41/hull_polish` (sha `809e1c91`, Step 3.5 config caveat)
- [x] **Smoke campaign specs** — bon `6e8fe1f3` ((b) on), boff `b99acdc0` ((b) off); commit `28c56a6f`
- [x] **Reboot** (author) — booted 2026-09-25 22:33; OneDrive quit, project excluded from Spotlight; **A_post = 24.41 GiB**
- [x] **Two-arm smoke gate — PASS** — run ids `s53_w91_3x3_smoke_bon` / `s53_w91_3x3_smoke_boff`; evidence commit `56b98c13`
  - S16 per-cycle gross + residuals bitwise identical (b) on vs off: **True**; S17 footprint captured exactly both arms: **True**; per-arm S1–S15 14/14 both arms
  - footprint peaks: (b) on **18.53 GiB**, (b) off 23.13 GiB → saving **4.60 GiB** (both predictions missed high: W90 2.92–3.98, Add. 48 ~3); walls 2,454 s / 3,139 s — the difference is in the terminal workbook write (236 s vs 887 s), **not** the ADMM cycles (644/685/416 s vs 668/690/421 s)
  - (b)-on cycle walls 10.74 / 11.42 / 6.93 min (mean 9.70; prediction 10.1–12.5)
  - **persistence REFUSED**: P_persist 27.54 > 0.85·A_post 20.75 (also refused at g = 1) → hull polish omitted at 3×3
- [x] **Pair spec freeze** — `231558f0`, commit `b5fc49a5` — x0 `f6e9cd53` + n7_4h_e1 `c82522f4`; concurrency 1; (b) on; tail {True, 1e-6}; cap 500; persistence off; preflight 19.17 GiB (g × P_on, footprint)
- [x] **Pre-launch commit** — `c2362883` (TASKS.md, CLAUDE.md, STEP4_DFO_METHOD.md, PLANNER_BRIEF_2026-09-13.md)
- [x] **3×3 x = 0 CERTIFIED** — 2026-09-26 18:16:32 UTC, cycle 72, wall 28,536 s; **Q = 842,832,534.76** (gross, settlement excluded), bar 13,954.38; tail active cycles 64–72. **Prediction refuted:** 80/80 terminal solves at the μ floor predicted, 1/80 actual (all Optimal exits; μ fell 33×; floor fell 100–500×). **Objective still descending at certification:** −1,009 → −4,006 €/cycle over cycles 66–72.
- [x] **3×3 node-7 unit CERTIFIED** — 2026-09-27 02:38:33 UTC, cycle 69, wall 30,117 s; **Q = 842,595,839.51**, bar 4,010.57, rule ten 0.0200; tail active 61–69
- [x] **3×3 PAIR COMPLETE** — launcher exit 1 = **G6 failed on both cells** (recorded under Add. 50 Ruling 1, not a run failure)
  - **value = 236,695** (13.2× the bar-sum 17,965); **value − I = −81,262, determinate 4.5× — storage does NOT pay at 3×3**
  - **R = 0.9126** vs reference 259,375.33; ratio resolution 0.1405 → indistinguishable from the recorded **0.9331** (0.15×); below the [0.93, 1.09] band floor by 0.12× resolution
  - **Both cells still descending at certification, asymmetrically:** x = 0 −4,006 €/cycle, node-7 −1,688 €/cycle, both accelerating → if sustained, value shrinks and value − I grows more negative (the sign conclusion is robust to the drift)
  - **Add. 50 node-7 prediction partly refuted:** G6 fails and median μ/floor ≈ 3 ✓, but **39/80 at floor**, not ≤ 3/80; one `Acceptable Level` exit; one solve at μ/floor 27.6
  - **Persistence was OFF:** only ESSO models saved, no network `certified_models.pkl` → the continuation's bitwise gate is likely unmeetable
  - *Pair campaign (both cells):* campaign `s53_w91_3x3_pair`, spec `231558f0`; launched **2026-09-26 10:20:56 UTC**; preflight 20.36 GiB available vs 19.17 required (PASS; thin margin — compressor refilled to 5.07 GiB after the smoke; the (b)-off smoke arm ran at ~95 % of available and completed correctly). Sequential, concurrency 1: x0 then n7_4h_e1; nothing else runs on the Mac. Est. **~9.2 h/cell, ~18.4 h pair** (range 13.4–24.6 h; no-decay bound 19.8–31.0 h). Log `w90_3x3/pair_run_v36_launch.log`
- [x] **Advisor: uncoordinated-benchmark definition** — delivered 2026-09-26 during the pair (read-only, no compute). Recommends a **two-arm decomposition** (U-passive, U-price-taker), a hard-fixed TSO interface replacing the 9e10 tracking term, costing via `_get_operational_recourse_components` with the curtailment penalty reset to 0, and a new production function beside the existing one. **Ruled in Addendum 49** (three arms; claim = min(passive, price-taker) − coordinated) — see the Addendum 49 order below.
- [ ] **Stop for review** after the pair — R restated against **259,375.33**; R = 0.9331 recorded
- [ ] 42(1) linear-solver benchmark / 42(3) persistent-worker timing — idle time, after review
- [ ] Step 5 SRP1 rows

## Addendum 50 order — G6 failure benign; objective not settled at certification

- [x] **Stop and report** on the refuted floor prediction — correct per the stopping conditions; node-7 continues; **no rule changes mid-pair**
- [x] **G6: option (a)** — recorded as FAILED (1/80 at floor vs ≥ 72/80) with its mechanism, **not re-scoped**. The floor is IPOPT's `mu_min` clamp, not a target. Future G6 (frozen before the next run): depth = `Optimal Solution Found` under the tail tolerances; μ/floor reported, not gated. **Prediction for node-7:** G6 fails likewise (≤ 3/80 at floor, median μ/floor ≈ 3)
- [x] **Certified Q stands "certified on residuals"; its ± bar claim does NOT** (the bar bounds stopping slack, not a path still descending)
- [x] **W95: x = 0 diagnostics from records** — commit `d757394e` (stdlib only, 66.5 MiB peak, no model loading)
  - **Rule ten = 0.0475** (4,005.67 / 84,283.65) — "well inside", but its 1e-4-relative threshold is ~20× the drift step, so rule ten **cannot detect** a drift of this size
  - **G6 verification:** all 80 terminal solves `Optimal`; **all 80 pass all four terminal error metrics** (worst: NLP error 0.886·tol, complementarity 0.897·compl_inf_tol). All 79 above-floor solves end at the **same μ = 2.5059e-9**, one IPOPT update from the floor via the superlinear μ^1.5 term. 6 TSO solves have μ/floor 5.0–5.9 — **the (1, 5] band test is mis-derived** (it assumes only the linear 0.2 factor). Awaiting expert ruling on the band.
  - **H_row18 contradicted** (row-18 charge rises while Q falls; −7.5 % of ΔQ over 66–72). **H_B contradicted** (not uniform, not decreasing). **H_C mixed.**
  - **The drift PREDATES the tail:** same DSO7 blocks, same signs, over cycles 53–62; ‖Δz‖ smooth across cycle 64. The 63–65 swing coincides with **AA switching off** at 63. Reading: objective converges more slowly than the residuals; certification fired at 63 (PF dual ratio first < 1). Awaiting expert reading.
  - **Evidence base not yet committed:** the x = 0 eval records are hashed in W95's manifest but not committed — to be committed with the pair evidence after node-7 finishes.
- [x] **W96:** pair evidence (in `47a54c89`, record in `c2d4b4ce`'s message); node-7 diagnostics `c2d4b4ce`. **Drift predates the tail on node-7 too** (from cycle 50; tail at 61). Floor status set by objective scaling S, not depth. **Terminal state NOT restorable** — and persisted models would not have sufficed (production continuation resets AA, tail and iteration count)
- [ ] **▶ STOPPED FOR REVIEW** — report `P5_15_ADDENDUM48_50_3X3_REPORT.md`. **Decisions pending:** (1) continuation route — A replay-and-continue ≈ 22.8 h (recommended) / C none; (2) G6 floor criterion → Optimal + four terminal metrics
- [ ] **Continuation** of each cell, up to 30 cycles, one at a time, behind a **bitwise identity gate** on the terminal cycle (else a separately labelled run); stop at |ΔQ| < 500 €/cycle for 3 cycles. **Prediction:** a hump, peak within ~5 cycles; 20–60 k€ remaining descent per cell; inter-cell difference < 10 k€. **If steps still rise after 10 cycles or the total exceeds 100 k€: H_C likely, and no 3×3 result is used** before a settling criterion exists
- [ ] Report value and R **three ways**: at certification; after continuation; with the drift as explicit uncertainty
- [ ] Settling criterion for future specs — **Advisor review after the continuation data**, not before

## Addendum 49 order — uncoordinated benchmark (three arms, fixed interface, common Q)

- [x] **18.25 % coordination benefit WITHDRAWN** (measured on the transfer-payment recourse Addendum 11 retired)
- [x] **Code + tests WRITTEN, nothing executed** — commit `a8c58da0` (W93): new module `uncoordinated_benchmark.py` (kept out of `shared_resources_planning.py` so the pair's second cell imports unchanged code), harness `p515_s53_w93_uncoordinated_benchmark.py`, checks script. Certified x = 0 cell **did curtail** (589.18 € at weight 1), so the common-Q gate can discriminate tie-breaker 0 from 1. Arm Q = minimum over 3 starts — **conservative against the coordination claim**.
- [x] **W94: 24-solve check de-confounded** — commit `b9ba413d`, nothing executed. The check's fixed side now pins interface voltage to the penalty's target, from one shared helper (identical floats); the TSO **arm** keeps voltage within normal bounds (negative control fails if a pin is planted on it). Feasibility of the pin resolves by construction: its targets come from the certified coordinated point, where the TN itself held those voltages at buses 5/7/9.
- **Correction to Add. 49 wording:** on the interface-P channel only σ and the interface rating scale λ_t; S_ref and D5 act on the ESS channel only (W93, from source).
- [ ] *after the pair review:* λ_t vs π_t zero-solve look from the certified x = 0 cell → recorded as the prediction (Advisor's "inside the band" beside it as the competing prediction)
- [ ] Common-Q gate — new evaluation function reproduces the certified x = 0 `gross_operational_cost` from its persisted models (bitwise, or explained to the last digit)
- [ ] 24-solve penalty-vs-fixed TSO check — reported, not gating
- [ ] Three arms (passive, price-taker, coordinated) × three starts (cold, warm-from-certified, perturbed) — band measured on the arms
- [ ] Consistency re-evaluation — DN at the TN's actual interface voltage; one sequential pass if a DN limit is violated
- [ ] Report — claim = **min(passive, price-taker) − coordinated**; decomposition beside it
  - **Author requirement:** curtailed energy per arm beside the coordinated cell's **589.18** (`res_curtailment_definitional_at_weight_1`: € at 1 €/MWh, block-weighted — years × days × discount — so *weighted* MWh-equivalent). Report every arm on the **same weighting**, with raw MWh alongside.
- [ ] Second Advisor note — 3×3 convention (after the pair)
- **Tie-breaker — RULED (Add. 49 clarification, 2026-09-26):** *evaluation* Q uses **0 in every arm** (the certified value; the common-Q gate fixes it). *Decision* objectives: **0** for price-taker and coordinated; **1 €/MWh for the passive arm only** — the minimum-curtailment selection rule in an otherwise empty objective. Verify value-independence by re-solving passive at **0.1 and 10 €/MWh** and reporting the interface-schedule difference (expected within solver tolerance). Report curtailed energy per arm.
- **λ_t recovery — RULED:** read **both** the interface-P consensus-dual Param from the W86 x = 0 persisted models **and** the TN interface-bus power-balance dual from the suffixes; undo the ADMM scaling to €/MWh — on the interface-P channel only **σ and the interface rating** apply (S_ref, D5 are ESS-channel only; correction above). **Units check first:** reproduce Addendum 32's bus-7 marginal cost = DSO flexibility shadow price before the λ_t vs π_t table is used as a prediction.

## Carried from earlier addenda (closed)

- [x] Tight tail adopted (Add. 46 r7 / 48) — new SRP1 reference **R = 259,375.33** (ΔR = −52.44); re-cert evidence `0a4bf784`
- [x] G6 final scope written into the spec (v30, v32 recorded as POST-HOC; v35 tier-2 and v36 V5 fixes made before any run)
- [x] `identity_holds` on `s47_recert` recomputed True under the current formula (v35)
- [x] Bar caveat recorded: `bar_tail = bar_ref` bitwise — the two-run bar is not independent
