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
- [ ] **▶ ACTIVE — 3×3 pair, RUNNING** — campaign `s53_w91_3x3_pair`, spec `231558f0`; launched **2026-09-26 10:20:56 UTC**; preflight 20.36 GiB available vs 19.17 required (PASS; thin margin — compressor refilled to 5.07 GiB after the smoke; the (b)-off smoke arm ran at ~95 % of available and completed correctly). Sequential, concurrency 1: x0 then n7_4h_e1; nothing else runs on the Mac. Est. **~9.2 h/cell, ~18.4 h pair** (range 13.4–24.6 h; no-decay bound 19.8–31.0 h). Log `w90_3x3/pair_run_v36_launch.log`
- [x] **Advisor: uncoordinated-benchmark definition** — delivered 2026-09-26 during the pair (read-only, no compute). Recommends a **two-arm decomposition** (U-passive, U-price-taker), a hard-fixed TSO interface replacing the 9e10 tracking term, costing via `_get_operational_recourse_components` with the curtailment penalty reset to 0, and a new production function beside the existing one. **Awaiting author/expert ruling** (formulation-level choice). Runs only after the pair (one-run rule); ~120 SRP1 solves.
- [ ] **Stop for review** after the pair — R restated against **259,375.33**; R = 0.9331 recorded

## Addendum 49 order — uncoordinated benchmark (three arms, fixed interface, common Q)

- [x] **18.25 % coordination benefit WITHDRAWN** (measured on the transfer-payment recourse Addendum 11 retired)
- [ ] **▶ ACTIVE (during the pair) — code + tests WRITTEN, nothing executed** — not even tests that build a model (pair holds ≈ 22 of 24 GiB). Static checks only.
- [ ] *after the pair review:* λ_t vs π_t zero-solve look from the certified x = 0 cell → recorded as the prediction (Advisor's "inside the band" beside it as the competing prediction)
- [ ] Common-Q gate — new evaluation function reproduces the certified x = 0 `gross_operational_cost` from its persisted models (bitwise, or explained to the last digit)
- [ ] 24-solve penalty-vs-fixed TSO check — reported, not gating
- [ ] Three arms (passive, price-taker, coordinated) × three starts (cold, warm-from-certified, perturbed) — band measured on the arms
- [ ] Consistency re-evaluation — DN at the TN's actual interface voltage; one sequential pass if a DN limit is violated
- [ ] Report — claim = **min(passive, price-taker) − coordinated**; decomposition beside it
- [ ] Second Advisor note — 3×3 convention (after the pair)
- **Open for the author:** the tie-breaker's "production value" is 0 (coordinated ADMM subproblems) or 1 €/MWh (build default); carried as a declared parameter until ruled.
- [ ] 42(1) linear-solver benchmark / 42(3) persistent-worker timing — idle time, after review
- [ ] Step 5 SRP1 rows

## Carried from earlier addenda (closed)

- [x] Tight tail adopted (Add. 46 r7 / 48) — new SRP1 reference **R = 259,375.33** (ΔR = −52.44); re-cert evidence `0a4bf784`
- [x] G6 final scope written into the spec (v30, v32 recorded as POST-HOC; v35 tier-2 and v36 V5 fixes made before any run)
- [x] `identity_holds` on `s47_recert` recomputed True under the current formula (v35)
- [x] Bar caveat recorded: `bar_tail = bar_ref` bitwise — the two-run bar is not independent
