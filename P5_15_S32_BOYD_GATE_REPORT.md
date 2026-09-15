# P5.15 Step 3.2 + 3.3(a) — s31c ρ/residual extraction and the Boyd gate (s32)

**Planner report, 2026-09-15.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addenda 9 (§3.2, §3.3(a)) and 13.
**Stops for review.**

Objective convention: system cost = `gross_operational_cost`, settlements excluded by construction; salvage is 0 in
both runs, so gross = net. Instance: C\* (s = 0.96875 MVA, e = 3.875 MWh, year 2025, uniform across nodes 5/7/9).

## 1. Lead A — s31c ρ/residual extraction

Zero solves, guard armed; commit `9cbfb4d1`; artifact `data/SRP1/Results/P515S32/s31c_rho_residual_extraction.json`.

- **Initial ρ.** s31c started from the harness ρ (v 1.5, pf 300, ess 1), not the case file (1.0).
- **ρ changed only in cycles 1–12.**
  - V was increased once: 1.5 → 2.25.
  - PF was decreased twelve times: 300 → 2.3122.
  - ESS stayed at 1.0.
- **Balancing never fired during the crawl (cycles 18–90).**
  - The freeze clause (both tolerance-normalized ratios ≤ 1) held V and ESS in all 73 cycles and PF in 72; the one exception was PF cycle 19, held by the dead band.
  - Without the freeze, the dead band alone would have raised ρ_pf in only 7 of the 73 cycles (61, 68, 71–73, 81, 82) and lowered ρ_v in 22.
  - **The freeze therefore explains the crawl only partially.**
- **Cross-checks.** Logged actions equal the rule recomputed at full precision. The reconstructed ρ matches all 13 in-cycle ρ prints. Log ratios agree with the trajectory within 3-decimal rounding.

## 2. Lead B — gate s32: **150 cycles, stopped by the cap. GATE FAILED (no Boyd stop).**

| | s31c (reference trajectory) | **s32** |
|---|---|---|
| stopping rule | composite (legacy residual AND objective) | **Boyd §3.3.1, all three channels** |
| initial ρ / balancing | harness 1.5/300/1, freeze clause | case file 1.0, spec v2 (ρ part of s), no freeze |
| cycles | 90 (cap) | **150 (cap)** |
| system cost at terminal | 661,400,927.74 | **654,413,409.74** |
| objective step / diagnostic tolerance (rule ten) | 1.96 | 0.46 (diagnostic only) |
| network failures | 9, all tier 1 | 17, all tier 1; 0 unrecovered |
| local-solve failures | 0 | 0 |
| solves | 4,650 | 7,718 = 51×151 + 17 retries |
| settlement cancellation closure | 2.1e-7 | 1.6e-7 (T_TSO + ΣT_DSO = +86,141; priced residual −86,141) |

Run facts: launched at `0bde2c25` under frozen spec v2 `516bd749`, exit 0, wall time 4,829 s. Evidence commit `9627525a`.

### Terminal Boyd residuals (cycle 150; ratio = value / threshold)

| channel | r / ε_pri | s / ε_dual | ρ-part of s / ε_dual | proximal share of s | ε_dual at √n·ε_abs floor | ρ terminal | n |
|---|---|---|---|---|---|---|---|
| V | **0.41** (pass) | **9.17** | 0.07 | 1.00 | ≈ 1.00 | 0.0077 (in force) → 0.0116 | 864 |
| PF | **3.54** | **15.78** | 3.06 | 0.98 | 0.91 | 0.198 | 1,728 |
| ESS | **1.77** | **5.56** | 0.83 | 0.99 | 0.999 | 0.0585 | 5,184 |

**Pass counts over 150 cycles:**
- V primal 139 (from cycle 10), dual 0.
- PF primal 0 (minimum ratio 2.19), dual 0.
- ESS primal 91, dual 0.
- No cycle had even two channels passing at once.

Binding test at terminal: PF dual.

**Counterfactual with the proximal part removed from s:**
- V would pass in 135 cycles (from 16).
- PF and ESS would never pass.
- All channels never pass together.

**The proximal term is not the only obstacle.**

### ρ trajectory (spec v2 balancing)

- **V.**
  - Decreased 12 times in cycles 2–47 (1.0 → 0.0077).
  - Then an exact period-3 limit cycle (decrease, increase, hold) between 0.0077 and 0.0116 from cycle 48 to 150.
  - The cycle follows arithmetically from the dead band. The ρ-part ratio equals ρ‖Δz‖/ε_dual, and ‖Δz‖ is fixed by γ, so it alternates across 5× the primal ratio.
- **PF.** Decreased at cycles 4, 5, 53 and 91 (1.0 → 0.198); held otherwise.
- **ESS.** Decreased every 21–22 cycles (12, 33, 54, 75, 97, 119, 141), 1.0 → 0.0585; never increased.
- **Cycle 1.** Held on all channels, as predicted before launch from the v1 preflight values.

### System cost against s31c

| cycle | 1 | 5 | 10 | 18 | 30 | 50 | 70 | 90 | 150 (s32) |
|---|---|---|---|---|---|---|---|---|---|
| s32 (M) | 968.49 | 799.10 | 651.28 | 668.78 | 663.77 | 659.20 | 656.31 | 655.29 | 654.41 |
| s31c (M) | 2,359.31 | 1,357.91 | 1,191.55 | 769.94 | 698.49 | 674.93 | 666.03 | 661.40 | — |

- **Terminal comparison.** s32@150 − s31c@90 = −6,987,518 (−1.06 %). The bar (sum of terminal steps, 160,011) is **not valid**, because neither run settled.
- **Matched cycle 90.** −6,113,118.
- **Attribution.** Initial ρ and the balancing rule differ, so none of these differences is attributable to the stopping rule, and **none is a limit statement**.

**Other terminal values.**
- Generation 450.75 M; DSO-internal flexibility 203.68 M.
- D rows −12,801 (floor); load and RES curtailment 0.
- EFC/day max 0.104, against a threshold of 1.46 (s31c: 0.0285).
- Per-DSO settlement +426.6 M / +318.6 M / +391.0 M (nodes 5/7/9); max |δ_P| 0.736 / 0.729 / 0.852 pu.

## 3. What the gate shows

Observation, evidence and verification come from `s32_supplementary.json` and `s32_e0_coherence.json`, both zero-solve with guards armed. The independent Advisor assessment was verified by the Planner where cited.

- **The old composite rule would have stopped s32 at cycle 48,** at 659.45 M; 71 cycles satisfy it. The run then descended a further 5.04 M (0.77 %) by cycle 150.
  - The net/abs recourse change is −0.93, −0.83, −0.67, −0.86 and −0.76 in the 25-cycle windows from cycle 26 on.
  - **The old rule certified slow change, not stationarity.** The Boyd rule's refusal to stop is correct behaviour on this trajectory.
- **The motion is deterministic drift, not noise.**
  - **V.** ‖z_V‖ rises in every one of cycles 26–150 at a constant 0.0019 per cycle. The per-cycle step ‖Δz_V‖ is constant at 0.0027 (RMS 9.2e-5 per entry) over cycles 25–150 and unaffected by ρ_v's five-fold change over that span (0.039 → 0.0077). Coherence Δ‖z‖/‖Δz‖ is 0.71 in every window after cycle 25.
  - **Interpretation of V.** A constant-speed step whose size does not depend on ρ is what a proximal-point step on a locally linear cost gives, since the TSO curvature in V is γ + ρ ≈ γ. It is inconsistent with zero-mean local-solve noise.
  - **PF.** ‖z‖ rises every cycle with coherence 0.16–0.25 while the step shrinks: 0.0366–0.0863 (cycles 26–50) → 0.0070–0.0102 (cycles 126–150).
- **Primal residuals are lag, not disagreement.** r / ‖Δz‖ averages 0.5–0.6 for PF and ≤ 0.3 for V and ESS after cycle 25. PF primal fails because the TSO is still moving, so relaxing ε_rel would certify motion.
- **ESS motion is released by each ρ_ess decrease.**
  - The ESS proximal step is flat over the three cycles up to each decrease, then rises over the next three, at all seven decreases.
  - ‖z_ESS‖ grows 0.10 → 0.80, accelerating, while cost gains shrink.
  - The objective-scaling concern (F5) is a competing explanation not yet separated: the ESSO objective is not divided by σ while TSO and DSO objectives are.

**Hypotheses.**

| | V | PF | ESS |
|---|---|---|---|
| (a) iterate-noise floor | contradicted at current magnitudes | contradicted | contradicted |
| (b) genuine slow motion | supported | supported | supported |
| (c) γ = 1 throttles TSO steps once balancing drives ρ ≪ γ | supported (γ/ρ ≈ 100) | supported (γ/ρ ≈ 5) | supported (γ/ρ ≈ 17) |
| (d) ρ changes sustain motion | rejected (step identical across ρ phases) | n/a | supported (step change after each decrease) |
| F5 ESSO objective scaling | — | — | open, not separated from (d) |

A noise floor below the current magnitudes is still unmeasured.

**Conclusion.** The failure is configurational, in how the fixed TSO proximal weight γ = 1 interacts with balancing that drives ρ one to two orders below it. The Boyd criterion itself is not at fault. **Relaxing ε is not justified:** at these magnitudes it would certify drift.

## 4. Changes made during this step (for the record)

- **Spec v1 → v2** (`166c7f68` → `9603eee2`).
  - The balancing dual ratio uses the ρ part of s; the stopping test is unchanged.
  - The v1 preflight showed the proximal part alone decided both cycle-1 decreases (PF 7,734 vs threshold 7,625, 5,469 without it; V 178 vs 157, 126 without it). The configured mechanism never included it.
  - Implementations: `2d765573`/`7305a476` (v1), `d06f3636`/`0bde2c25` (v2).
- **Mapping diagnostic** (`e788a771`). The V/PF Boyd mapping is consistent with the dual update to machine precision. ‖y‖ at cycle 1 includes one pre-loop λ update made by the initialization call — pre-existing behaviour, identical in s31c, recorded and not changed.
- **Not implemented.** The gap proxy G/Q is recorded as None: σ_b is undefined for the ESSO block (F5).

## 5. Decisions for review (no action taken)

The Advisor's recommended order, ranked by information per cost; the Planner concurs.

1. **E1 — zero solves.**
   - Terminal interface V against TSO voltage bounds, and the cycles to reach a bound at the observed rate. If V is heading to a bound, its drift is a true optimum being approached slowly (> 600 cycles at 6.6e-5 pu per cycle to e.g. +0.04 pu).
   - ESSO objective scale against TSO/DSO `effective_scale`. If the scales differ by orders of magnitude, F5 becomes the leading ESS explanation and may need to move ahead of 3.3(b).
   - Also check where `consensus_vars['ess']['tso']['prev']` is updated relative to the proximal centre, and the PF ‖y‖ unit (λ/s_base against the rating-normalized constraint). At most about 10 % on ε_dual, so this cannot change the verdict.
2. **E2 — one run, single factor: γ tied to ρ (γ = ρ, updated with it), all else as s32.**
   - The proximal term vanishes at a fixed point, so the problem solved is unchanged; only the method changes.
   - **Supports (c):** faster TSO steps, ‖z_V‖ plateauing, r and s falling together, cost ≤ 654.4 M reached earlier.
   - **Falsifies (c):** unchanged step speed.
   - **γ was stabilizing:** oscillation or failures appear; then try γ = 0.1 fixed.
3. **E3 — one run, fixed ρ (balancing off), γ = 1.** Only if the ESS ambiguity, (d) vs F5, remains after E1/E2.
4. **E4 — a few guarded solves: local noise floor.** Re-solve one TSO block at a frozen consensus point, warm and cold. Required before **any** ε change, which would be derived as ε_abs ≳ 3δ.
5. **Author questions carried forward.**
   - `minimum_consecutive_converged_cycles` 1 vs 3.
   - Whether the proximal regularization (γ) is part of the intended method or a stabilization aid that may be tied to ρ. It was enabled before Step 3.
   - Whether 3.3(b) and 3.4 proceed before 3.2 passes. The brief orders them after, and each gate assumes "converges under 3.2's rule".

## Evidence

- **s31c extraction:** `p515_s32_s31c_rho_extraction.py` → `data/SRP1/Results/P515S32/s31c_rho_residual_extraction.json` (+ manifest), commit `9cbfb4d1`.
- **Specs:** `data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json`, `frozen_s32_spec_v2_516bd749.json`.
- **Gate:** `data/SRP1/Results/P515S32_run/` — `g_baseline.json`, `boyd_terminal.json`, `component_levels_terminal.json`, `interface_settlement_detail_s31c.json`, `network_failures_baseline.jsonl`, stdout, heartbeat, ESSO pickle; `data/SRP1/Results/P515S32_launch.log`, `P515S32_exit_code.txt`.
- **Evaluation and analysis** (zero-solve, guards armed):
  - `p515_s32_evaluate.py` → `s32_evaluation.json`;
  - `p515_s32_supplementary.py` → `s32_supplementary.json`;
  - `p515_s32_e0_coherence.py` → `s32_e0_coherence.json`.
- **Manifests:** `evidence_manifest_sha256.json` covers 473 files (`esso_capture/` and `results/` hash-recorded, not committed); `s32_e0_manifest_sha256.json` covers E0.
- **Worker reports:** `WORKER_REPORT_S32_IMPL.md`, `WORKER_REPORT_S32_MAPPING_DIAG.md`, `WORKER_REPORT_S32V2_IMPL.md`.
