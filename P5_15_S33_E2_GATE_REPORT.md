# P5.15 Step 3.2 — E1, E4 and the γ = τρ gate (s33e2)

**Planner report, 2026-09-16.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 14. Frozen spec v3
`data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json` (`4e561d1a`), superseding s32 spec v2. **Stops for
review.**

Objective convention: system cost = `gross_operational_cost`, settlements excluded by construction; salvage 0, so
gross = net. Instance: C\* (s = 0.96875 MVA, e = 3.875 MWh, year 2025, uniform across nodes 5/7/9).

## 1. Order executed

E1 (zero solves) → E4 (noise floor, 8 solves) → E2 (one run, cap 150). E3 not run.

- **E1** (`254048f4`, attribution note `90c5da2c`): no voltage bound was close in s32 — V drifting toward V_max at
  6.5e-5 pu/cycle, closest sampled entry 0.061 pu away, 300–600 cycles from contact. D5 confirmed in code
  (σ = 9.363536e7 divides TSO/DSO objectives, not the ESSO's) but with no practical effect at the s32 terminal point
  (ESSO base objective ≈ −5.7e-3, a tolerance artifact). The two low-priority checks came back clean: the ESS `prev`
  copies and the TSO proximal centre stay consistent, and the PF ‖y‖ unit conversion is a self-cancelling round trip.
- **E4** (`24f808d7`, frozen spec `254048f4`): 8/8 replays optimal, warm repeat bitwise identical, fixture hashes
  verified. Per-channel solver noise **δ: V 1.04e-14, PF 1.25e-8, ESS 6.60e-8**. Frozen rule
  max(1e-5, 3·max δ) ⇒ **ε_abs = 1e-5, unchanged**. The noise floor is 4–10 orders below the steps s32 was taking, so
  **the s32 motion was never solver noise**.

## 2. Gate s33e2 — **FAILED as specified: 150 cycles, cap, no Boyd stop**

Launched at `0610741b`; exit 0; wall 4,841 s; evidence `534923e7`. The spec's falsifiable prediction held: cycles 1–2
reproduced s32 exactly (0 differing fields across 43 compared), first divergence at cycle 3.

| channel | primal r/ε_pri | dual s/ε_dual | ρ = γ | pass history |
|---|---|---|---|---|
| V | **0.064** | **0.105** | 0.0077 | both pass from cycle 48; 96 channel-pass cycles (s32 dual was 9.17) |
| PF | 0.400 | **3.083** | 0.198 | primal from 113; dual never; decaying ×0.552 per 50 cycles |
| ESS | 0.119 | **3.245** | 0.667 | primal in 140 cycles; dual never; decay ×0.997 per 50 cycles |

No cycle had all three channels pass; none had two. Binding test: ESS dual. Objective diagnostic: terminal step 5,179
against tolerance 65,383 (rule ten 0.079). 18 network failures (16 tier-1, **2 tier-2**), 0 unrecovered, 0 local-solve
failures; 7,721 solves = 51×151 + 20 retries. EFC/day max 0.067 (threshold 1.4612). Settlement cancellation holds.
System cost 653,826,842.

**Tying γ to ρ fixed the V channel, as intended.** V's dual ratio fell 9.17 → 0.105; its per-entry step is essentially
unchanged from s32 (9.6e-5 vs 9.2e-5), but ε_dual now tracks ρ = γ instead of being outrun by a fixed γ. The V drift
identified in s32 reached its destination: **52 of 864 terminal interface voltages sit within 1e-6 pu of a bound and
339 within 0.005 pu**, node maxima exactly 1.1 pu. V therefore passes **partly by bound saturation**, which is stated
here as a caveat, not hidden.

## 3. Why the ESS channel cannot be certified

All analysis below is zero-solve with guards armed: `p515_s33e2_channel_analysis.py`, `p515_s33e2_z1_esso_drift.py`,
`p515_s33e2_z2_price_alignment.py`.

**Observation.** The ESS consensus translates at a constant rate: per-entry step 4.26e-5 (4.26 × ε_abs, **645 × the E4
noise floor**), constant from cycle 31 to 150; ‖z_ESS‖ grows linearly 0.10 → 0.46 with coherence rising to 0.97;
cos(step_k, step_{k−1}) = **1.000** at every cycle from 31; ‖y_ESS‖ is flat at 0.0079–0.0083 from cycle 25 while ‖x‖
and ‖z‖ grow, and the three agent copies agree to ~1e-8.

**The residual is ρ-invariant (Planner-verified).** At the single ρ_ess change (cycle 11, ρ ×0.667) `s_rho_part` moved
only ×0.996, so the physical step grew ×1.495. The same signature appears on V late in its adaptation (×0.997 at
cycles 16–23) but not early (×0.573 at cycle 2). **In this regime the dual residual measures the forcing gradient,
not the motion**: Δz ≈ g/(ρa²) and s ≈ ‖g‖/a. Raising ρ_ess would shrink the physical step and leave the residual
unchanged.

**Z1 falsified the throughput-tilt hypothesis.** Storage throughput **rises** 0.0154 → 2.219 per node (+0.0150/cycle
over cycles 31–150), Σ|pnet| with it; the step moves *away* from zero (cos against −sign(pnet) = −0.461). The ESSO
throughput term (ε = 1e-5) penalises throughput, so it is not the driver; the mean dual on the storage operation
constraint is 2.4e-7, 0.024 × ε. No discontinuity at any of the 18 failure cycles (step 0.00191–0.00193 throughout),
so failure-induced multiplier bias is excluded. Barrier multipliers converge toward ~1e-5 from both sides.

**Z2 identifies the driver: temporal price arbitrage.** Against the within-day deviation of the hourly market price,
the drift direction has **cos = −0.516 and Pearson −0.516 at every node and in every window** (cycles 100–150, 31–150,
140–150): entries that charge sit ~33/MWh **below** the day mean, entries that discharge ~29 **above** it. The
direction is fixed, which is why the cosine is identical across windows. Shared-ESS *usage* is priced at **zero** in
both networks (Step 3.1 row 8) and the ESSO carries no degradation cost on this branch, so only round-trip efficiency
bounds the arbitrage. Storage sits at **0.79 % of rating**; at the observed rate the mean entry would reach rating in
~14,900 cycles.

**Conclusion.** The ESS channel is not a null space and not solver noise: it is a real, unrelieved economic gradient
being walked down at a rate set by ρa². **No residual-based rule can certify this channel at ε_abs 1e-5 while that
gradient persists**, because the residual does not shrink with ρ and the valley's end is ~10⁴ cycles away.

## 4. Corrections to earlier Planner statements

- **"System cost is still descending" is withdrawn.** Over cycles 101–150 the cost *oscillates*: range 185,571
  (2.8 × the objective tolerance), 20 of 50 steps positive, and a 16-cycle rise of +87,052 from c128 to c143. The
  low "net/abs −0.05" figure reflects oscillation, not slow descent.
- **The "−5.0e5 per pu of throughput" trade is withdrawn.** The ESS step is constant while the cost rate collapses
  from −22,547/cycle (31–150) to oscillation, so the ESS contribution is collinear with a constant and **not
  identifiable** from this trajectory.
- **No difference against s32 or s31c is claimed.** Neither reference settled and this run did not stop; the
  differences bound stopping slack only.
- **γ = τρ did not fix the ESS channel** and was not expected to; it fixed V.

## 5. Verdict per channel

| channel | verdict |
|---|---|
| V | certified from cycle 48 under the Boyd rule, **with bound saturation** (52 entries at a bound) |
| PF | not certified; on a credible decay path (×0.552 per 50 cycles ⇒ ~95 further cycles), **conditional** on the ESS rows settling, since PF and ESS are rows of the same TSO block |
| ESS | **not certifiable** at ε_abs 1e-5 by any residual rule while the arbitrage gradient persists |

Gate: **FAIL** (3-consecutive all-channel stop not reached). Not certified; no cap increase taken.

## 6. Decisions for review (no action taken)

Advisor-reviewed twice; the Planner verified every cited number against the artifacts.

1. **Z3 — suboptimality bound (zero solves; Planner-authorizable).** Replace the residual claim with Boyd §3.3's bound
   `p^k − p* ≤ ‖y‖‖r‖ + D‖s‖` per channel, in currency. This is the only certification route that survives §3.
   **It requires D5 first**, because `gap_proxy_G` is `None` precisely because σ is undefined for the ESSO block.
2. **D5 promoted (author decision on scope, 3.4).** Put the ESSO objective on the same σ convention as the networks.
3. **Z4 — the ~30-cycle cost oscillation (zero solves; Planner-authorizable).** Unexplained, larger than the objective
   tolerance, and outside the ESS channel. It may be the real obstacle to any stop. It should be attributed before
   further gates.
4. **3.3(b) fixed reference rating (already authorized) — record the prediction first.** With s ≈ ‖g‖/a, replacing
   2·max(S, 0.10) by S_ref = 2.5 MVA should **raise** the ESS dual ratio by roughly 2.6× (3.25 → ~8.4). If it does
   not, the mechanism above is damaged. It is a normalization-consistency fix, **not** a remedy for the drift.
5. **Not recommended:** raising ρ_ess (the residual is ρ-invariant); extending the cap (PF's projection is conditional
   and ESS is ~10⁴ cycles from its equilibrium); E3 fixed ρ (ρ is already constant after cycle 30).
6. **Author decisions, not the Planner's.** Shared-ESS usage is priced at zero (row 8) and the ESSO has no degradation
   cost on this branch, so **`EPS_ESSO_THROUGHPUT` = 1e-5 is currently the only price on storage throughput** — an
   economic choice made by default. Whether to restore a shared-ESS usage cost, and what the throughput constant
   should be, decide where the ESS equilibrium sits and therefore what any Step 3 gate can converge to.
7. **Instrumentation.** Any future ESS gate should assert a capture path for per-entry ESS z/x (or a stride) before the
   run; this stage had to reconstruct the drift from the ESSO captures.

## Evidence

- Specs: `frozen_s33_e2_spec_v3_825f1f02.json` (v3), `frozen_e4_noise_floor_spec_v1_35f54dfe.json`.
- E1: `p515_s33_e1_zero_solve.py`, `data/SRP1/Results/P515S33/E1/` (+ manifest), `WORKER_REPORT_S33_E1.md`.
- E4: `p515_s33_e4_noise_floor.py`, `data/SRP1/Results/P515S33/E4/` (+ manifest), `WORKER_REPORT_S33_E4.md`.
- Implementation: `a00cbb29` (production), `27de5670` (harness), `WORKER_REPORT_S33_E2_IMPL.md`; zero-solve checks
  `data/SRP1/Results/P515S33/zero_solve_checks_e2/`; preflight `data/SRP1/Results/P515S33/preflight_e2/`.
- Gate: `data/SRP1/Results/P515S33_E2_run/` — `g_baseline.json`, `boyd_terminal.json`, `component_levels_terminal.json`,
  `interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json`, `network_failures_baseline.jsonl`,
  stdout, heartbeat, ESSO pickle; `P515S33_E2_launch.log`, `P515S33_E2_exit_code.txt`.
- Analysis (all zero-solve, guards armed): `s33e2_evaluation.json`, `s33e2_channel_analysis.json`,
  `s33e2_z1_esso_drift.json`, `s33e2_z2_price_alignment.json`.
- Manifests: `evidence_manifest_sha256.json` (474 files; `esso_capture/` and `results/` hash-recorded, not committed)
  and `z1_z2_manifest_sha256.json`.
