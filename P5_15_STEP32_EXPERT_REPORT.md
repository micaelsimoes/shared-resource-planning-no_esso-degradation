# P5.15 Step 3.2 — stopping rule and ρ policy: state of play for the external expert

**Planner report, 2026-09-16.** Self-contained. Authority: `PLANNER_BRIEF_2026-09-13.md`, Addenda 9 (§3.2, §3.3),
13 and 14. Instance throughout: candidate C\* (s = 0.96875 MVA, e = 3.875 MWh, invested 2025, uniform across active
nodes 5/7/9), production defaults (ε = 1e-5, ESSO `tol` 1e-10 / `acceptable_tol` 1e-9, recovery policy with tier 2),
cold start, one background run per gate.

Objective convention on every figure below: system cost = `gross_operational_cost`, settlements excluded by
construction; terminal salvage is 0 in all runs here, so gross = net.

## 1. Question this step had to answer

After Step 3.1 the reported recourse was made a system cost: the TSO interface flexibility charge (99.8 % of the prior
recourse, a transfer payment) was replaced by a signed linear interface settlement at the scenario price π_t, which
cancels at consensus (Addendum 12). That left the question the stopping rule had been hiding: **is the ADMM fixed point
the optimum the paper claims, and can the stopping rule certify it?** Step 3.2 replaces the composite
"consensus-AND-objective-motion" test with Boyd et al. (2011) §3.3.1 per channel; 3.3(a) puts residual balancing on
from the case-file ρ.

## 2. What has been established (with evidence)

### 2.1 The old composite rule certified slow change, not stationarity
On the s32 trajectory the old rule (legacy consensus tolerances AND |ΔQ| ≤ max(1e3, 1e-4·Q), one cycle) would have
stopped at **cycle 48 with 659.45 M**; the same run then fell a further **5.04 M (0.77 %)** by cycle 150. The Boyd rule
refused that point. This is the central methodological result of the step: the previous criterion cannot distinguish a
slow drift from a fixed point, because a drift of ~30 k per cycle satisfies a 1e-4 relative test indefinitely.

### 2.2 The stopping tolerance is derived, not chosen
`ε_abs` was fixed by measurement, not tuning (frozen spec `35f54dfe`, results commit `24f808d7`). Two exact pre-solve
blocks preserved during s32 (TSO case9 2025 Summer; DSO node 7 case33_2 2025 Autumn, both cycle 7) were re-solved
through production's own solver construction, four ways each — warm, warm repeat, cold, warm with IPOPT `tol` and
`acceptable_tol` divided by ten. 8/8 optimal; the warm repeat is bitwise identical; fixture hashes verified before any
solve.

| channel | solver noise δ (normalized RMS/entry) | s32 per-entry step at cycle 150 | ratio |
|---|---|---|---|
| V | 1.04e-14 | 9.2e-5 | 9e9 |
| PF | 1.25e-8 | 1.7e-4 | 1.4e4 |
| ESS | 6.60e-8 | 5.5e-5 | 8e2 |

Frozen rule `max(1e-5, 3·max δ)` ⇒ **ε_abs = 1e-5 unchanged**, ε_rel = 1e-4. The measurement matters beyond its value:
**the motion these gates refuse to certify is 3–9 orders of magnitude above solver noise.**

### 2.3 A fixed proximal weight was throttling the TSO update (s32 → s33e2)
s32 (fixed γ = 1, balancing free) hit the 150-cycle cap with **no channel passing its dual test in any cycle**. Per-cycle
diagnostics showed the TSO interface copies moving at a constant speed, independent of ρ, while balancing drove ρ one
to two orders below γ; below ρ ≈ γ the proximal term dominates the dual residual, which then cannot fall.

Addendum 14 accepted the diagnosis and tied the stabiliser to the penalty: **γ_c = τ·ρ_c per channel, τ = 1**
(Deng & Yin 2016), ρ adaptation frozen after cycle 30 (ADMM requires an eventually constant ρ), **3 consecutive
converged cycles**, γ read from the model in force. Implemented in `a00cbb29` / `27de5670` with 10 zero-solve checks
and a two-cycle preflight that reproduced s32's first two cycles exactly (0 differing fields across 43 compared),
diverging at cycle 3 as predicted.

**Result (gate s33e2, evidence `534923e7`, report `77b6ea05`): the voltage channel was fixed.** Its dual ratio fell
from 9.17 to 0.105 and it passes both tests from cycle 48 (96 channel-pass cycles). Its per-entry step is essentially
unchanged (9.6e-5 vs 9.2e-5); what changed is that ε_dual now tracks ρ = γ instead of being outrun by a fixed γ.

**Caveat stated plainly:** V passes partly by **bound saturation** — 52 of 864 terminal interface voltages sit within
1e-6 pu of a bound and 339 within 0.005 pu, node maxima exactly 1.1 pu. That is the bound the s32 drift was heading
for (E1 projected 300–600 cycles at the s32 rate).

### 2.4 The gate still fails, and the binding channel is the shared ESS
150 cycles, cap, exit 0; no cycle had all three channels pass, none had two.

| channel | primal r/ε_pri | dual s/ε_dual | proximal share of dual | ρ = γ terminal | pass history |
|---|---|---|---|---|---|
| V | 0.064 | 0.105 | 0.707 | 0.0077 | both from cycle 48 |
| PF | 0.400 | 3.083 | 0.707 | 0.198 | primal from 113; dual never; ×0.552 per 50 cycles |
| ESS | 0.119 | **3.245** | 0.487 | 0.667 | primal in 140 cycles; dual never; ×0.997 per 50 cycles |

ρ/γ: V 1.0 → 0.0077 (12 decreases, all before cycle 24); PF 1.0 → 0.198 (4, last at 29); ESS 1.0 → 0.667 (one, at 11);
**zero changes after the cycle-30 freeze**, verified. Objective diagnostic: terminal step 5,179 vs tolerance 65,383
(rule ten 0.079). 18 network failures (16 tier-1, 2 tier-2, 0 unrecovered), 0 local-solve failures, 7,721 solves =
51×151 + 20 retries. EFC/day max 0.067 (threshold 1.4612). Settlement cancellation holds (closure ~1e-7).

### 2.5 Why the ESS channel cannot be certified by a residual rule
All zero-solve, guards armed; every number below was verified by the Planner against the artifacts.

1. **The ESS dual residual is ρ-invariant.** At the single ρ_ess change (cycle 11, ρ ×0.667) the ρ-part of the residual
   moved ×0.996 while the physical step grew ×1.495. In this regime Δz ≈ g/(ρa²) and s ≈ ‖g‖/a: **the residual measures
   the forcing gradient, not the motion.** Raising ρ_ess would shrink the step and leave the test unchanged. The same
   signature appears on V late in its adaptation (×0.997, cycles 16–23) but not early (×0.573 at cycle 2), so it is
   regime-dependent rather than an artifact of the metric.
2. **The motion is a straight line at constant speed.** Per-entry ESS step 4.26e-5 (4.26 ε_abs, **645 × the measured
   noise floor**), constant from cycle 31 to 150; cos(step_k, step_{k−1}) = **1.000** throughout; ‖z_ESS‖ grows
   linearly 0.10 → 0.46 with coherence rising to 0.97; the multiplier norm is flat (0.0079–0.0083 from cycle 25) while
   the primal copies grow, and the three agent copies agree to ~1e-8.
3. **It is not the throughput regulariser.** Throughput *rises* (0.0154 → 2.219 per node, +0.0150/cycle over cycles
   31–150) and the step moves away from zero; the ESSO term (1e-5) penalises throughput. Mean dual on the storage
   operation constraint is 2.4e-7 = 0.024 × that constant. Failure-induced multiplier bias is excluded: no step
   discontinuity at any of the 18 failure cycles.
4. **It is temporal price arbitrage.** Against the within-day deviation of the hourly price π_t, the drift direction
   has **cos = −0.516 (Pearson −0.516) at every node and in every window** (cycles 100–150, 31–150, 140–150): entries
   that charge sit ~33 /MWh **below** the day mean, those that discharge ~29 **above**. The direction is fixed, which
   is why the cosine is identical across windows.
5. **Nothing prices it.** Shared-ESS *usage* is zero in both network objectives (Step 3.1, row 8) and the ESSO carries
   no degradation cost on this branch, so only round-trip efficiency bounds the arbitrage. Storage sits at **0.79 % of
   rating**; at the observed rate the mean entry reaches rating in ~14,900 cycles.

**Conclusion.** The ESS channel is neither a null space nor solver noise: it is a real, unrelieved economic gradient
walked down at a rate set by ρa². **No residual-based rule can certify it at ε_abs 1e-5 while that gradient persists.**

## 3. Claims withdrawn

Stated explicitly because earlier Planner notes asserted them:
- **"System cost is still descending" — withdrawn.** Over cycles 101–150 the cost *oscillates*: range 185,571 (2.8 ×
  the objective tolerance), 20 of 50 steps positive, a 16-cycle rise of +87,052 (c128 → c143).
- **The implied "−5.0e5 per pu of added throughput" trade — withdrawn.** The ESS step is constant while the cost rate
  collapses, so the ESS contribution is collinear with a constant and is **not identifiable** from this trajectory.
- **No difference against s32 or s31c is claimed.** s33e2 runs below s32 at matched cycles (−7.24 M at 30, −2.22 M at
  60, −1.40 M at 100, −0.59 M at 150; terminal 653,826,842 vs 654,413,410 at 150 and s31c 661,400,928 at 90), but
  neither reference settled and this run did not stop, so every bar is invalid.
- **γ = τρ did not fix the ESS channel** and was not expected to. It fixed V.

## 4. Questions for the expert

1. **Certification route.** If no residual rule can certify the ESS channel, the remaining option is a suboptimality
   bound — Boyd §3.3's `p^k − p* ≤ ‖y‖‖r‖ + D‖s‖` per channel, reported in currency with D a diameter bound on the
   normalized coordinates. Is that the right thing for the paper to report as the decomposition's optimality evidence
   (R2.5), in place of "converged at tolerance"? It requires **D5** first: the ESSO objective is not divided by σ while
   TSO and DSO objectives are, which is exactly why the gap proxy is currently recorded as `None` rather than
   approximated.
2. **The economics that set the equilibrium.** Shared-ESS usage is priced at zero and the ESSO has no degradation cost
   on this branch, so `EPS_ESSO_THROUGHPUT` = 1e-5 is *de facto* the only price on storage throughput — an economic
   choice made by default. The arbitrage equilibrium, and hence what any Step 3 gate can converge to, sits wherever
   that choice puts it. Should a shared-ESS usage cost be restored (Step 3.1 row 8 deferred cycling cost to the ESSO,
   which then has none), and what is the defensible value of the throughput constant?
3. **Bound saturation as a convergence mechanism.** V now passes with 52 entries pinned at 1.1 pu. Is convergence by
   activation of voltage bounds acceptable evidence for the paper, or should the reported gate require the channel to
   converge away from its bounds?
4. **An unexplained mode.** System cost oscillates with a ~30-cycle period and an amplitude larger than the objective
   tolerance, outside the ESS channel. We propose attributing it (zero solves) before any further gate; it may be the
   real obstacle to any stop.
5. **Order of the remaining sub-steps.** Addendum 14 has 3.3(b) and 3.4 wait for 3.2 to pass. Given §2.5, 3.2 may not
   pass until the ESS pricing question is decided. Options: (a) proceed to 3.3(b)/3.4 with the ESS channel documented
   as non-certifiable at ε_abs 1e-5; (b) settle pricing first; (c) adopt the suboptimality bound as the gate. We have
   taken no action pending this decision. For 3.3(b) the prediction is recorded in advance: with s ≈ ‖g‖/a, replacing
   2·max(S, 0.10) by a fixed S_ref = 2.5 MVA should **raise** the ESS dual ratio ~2.6× (3.25 → ~8.4); if it does not,
   the mechanism in §2.5 is damaged.

## 5. Evidence index

| item | artifact | commit |
|---|---|---|
| Step 3 economic baseline (row 3′) | `P5_15_S31C_ECONOMIC_BASELINE_REPORT.md`, `P515S31C_run/` | `8551412b` |
| s31c ρ/residual extraction | `p515_s32_s31c_rho_extraction.py`, `P515S32/s31c_rho_residual_extraction.json` | `9cbfb4d1` |
| Boyd stop + balancing, specs v1/v2 | `frozen_s32_spec_v1_14a18674.json`, `..._v2_516bd749.json` | `166c7f68`, `9603eee2` |
| s32 gate and report | `P515S32_run/`, `P5_15_S32_BOYD_GATE_REPORT.md` | `9627525a`, `13014984` |
| E1 (bounds, D5 evidence) | `p515_s33_e1_zero_solve.py`, `P515S33/E1/`, `WORKER_REPORT_S33_E1.md` | `254048f4` (note `90c5da2c`) |
| E4 noise floor | `frozen_e4_noise_floor_spec_v1_35f54dfe.json`, `P515S33/E4/`, `WORKER_REPORT_S33_E4.md` | `254048f4`, `24f808d7` |
| Spec v3 (γ = τρ, freeze 30, 3 cycles) | `frozen_s33_e2_spec_v3_825f1f02.json` | `4e561d1a` |
| E2 implementation | production, harness, checks, preflight, `WORKER_REPORT_S33_E2_IMPL.md` | `a00cbb29`, `27de5670` |
| E2 gate evidence | `P515S33_E2_run/` (+ manifest, 474 files) | `534923e7` |
| Z1/Z2 drift diagnostics and report | `s33e2_z1_esso_drift.json`, `s33e2_z2_price_alignment.json`, `P5_15_S33_E2_GATE_REPORT.md` | `77b6ea05` |

Every campaign ran attached and alone through the tool's background run with stderr captured, an exclusive lock, a
heartbeat and the Python exit code written to a file. Every analysis script above runs with `SolveProfileGuard` armed
(zero permitted solves, verified), except E4, which declares and verifies exactly 8. `esso_capture/` and `results/`
are hash-recorded in the manifests rather than committed.
