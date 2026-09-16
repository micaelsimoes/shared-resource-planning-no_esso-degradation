# P5.15 Steps 3.3(b)/3.4 — the re-scaling gate: handoff to the Author and the External Expert

**Planner report, 2026-09-16.** Self-contained; supersedes nothing in `P5_15_STEP32_EXPERT_REPORT.md` but continues
it. Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 15. Instance throughout: candidate C\* (s = 0.96875 MVA,
e = 3.875 MWh, invested 2025, uniform across nodes 5/7/9), production defaults, cold start, one attached background run
per gate with stderr captured, an exclusive lock, a heartbeat and the exit code written to a file.

System cost = `gross_operational_cost`, settlements excluded by construction; salvage is 0 in every run here, so
gross = net.

## 1. What Addendum 15 authorized, and what was executed

| item | status | evidence |
|---|---|---|
| Z4 — attribute the ~30-cycle cost mode (zero solves, not blocking) | done | `a3eca693` |
| `EFC*` — price-taker storage benchmark (zero production solves) | done | `d1b8cf9c` |
| 3.4 with 3.3(b) folded in — one production commit | done | `ef2040d7`, harness `db8eb90a` |
| zero-solve checks (10) and a two-cycle preflight with a **detector gate** | all pass | `ef2040d7`, `db8eb90a` |
| one gate run, cap 150, prediction recorded in advance | run; **fails its pass test** | `202fd246` |
| anything beyond that | **not started** — awaiting your decisions | — |

Frozen spec v4 (`9f182b79`) recorded the design, the predictions and the falsification criteria **before** the run.

## 2. Corrections to the record since the last handoff

Three statements were withdrawn or qualified. All are committed, and history was not rewritten.

1. **"Row 3′ moved where the optimum sits" — withdrawn** (`81e7f1c5`). It compared capped transients. Within `s33e2`
   storage throughput rose ≈144 × while still moving in a perfectly persistent direction, so 0.067 was a point on a
   trajectory, not an equilibrium; `s31`'s 1.2110 is likewise a transient of a diverging run.
2. **The storage collapse mechanism, restated and verified in code** (`81e7f1c5`). Before `s31c` the only price on
   interface energy movement was row 3: `cost_flex·(flex_p_down + flex_q_down)` on the TSO's ADN-interface loads
   (`model_construction_helpers.py:1692–1712`). A DSO's import is its reference generator, **excluded from
   `generation_cost`** for distribution networks (`:1663–1670`), and the shared ESS enters the **TSO node balance**
   (`:1329`) — so TSO-side storage displaced exactly the charged volume and was implicitly remunerated by it. Removing
   row 3 **deleted that remuneration**. Two qualifications the code imposes: the charge was **one-sided** (down legs
   only) and had **no DSO receipt term**, which is why it was a transfer and must not be restored as-is.
3. **My description of the late-run cost behaviour — corrected** (`f37a1c63`). Z4 shows a **decaying ~20-cycle**
   transient (envelope half-life ≈20 cycles, R² 0.84–0.87), not a persistent ~30-cycle mode: the "2.8 × tolerance"
   span I quoted was raw and trend-dominated, while the **detrended** amplitude over cycles 101–150 is 0.09–0.38 ×
   tolerance. It is **not** caused by the ρ/γ freeze — the same mode appears in `s32`, which has no freeze, at 2–3 ×
   the relative amplitude. Z4 also found a capture gap: the block-level `[RECOURSE JUMP]` diagnostic stopped at cycle
   62, so late-cycle localisation was unrecoverable. That capture is now unconditional.

## 3. `EFC*`: the benchmark that replaces "EFC/day was 1.1 before"

Zero production solves; 36 price-taker LPs in scipy with the guard armed at zero Pyomo/IPOPT entries; prices
cross-checked bitwise against the run's own serialized settlement detail (max abs diff 0.0 over 864 cells).

- **`EFC*` = 1.8520** for 2025 (max across nodes; 0.7964–1.8520 across the twelve year/day cells; 2.1500–2.4750 with
  round-trip efficiency forced to 1.0). Every cell binds **both** the converter rating and the SoC range.
- **Consequence:** "order 1" is a defensible *derived* target and is, if anything, conservative. But `EFC*` **exceeds
  the SoH-binding threshold 1.4612**, so the achievable equilibrium under the full formulation should be expected **at
  the threshold with the degradation constraint active**, not at 1.85.
- The `EFC*_row3` counterfactual was **skipped deliberately**: no committed algebraic mapping exists from the row-3
  charge onto the storage's own dispatch, and inventing one would be an approximation.

## 4. The gate: s34

**Verdict: FAIL on the pass test** — 150 cycles, cap, no cycle with all three channels passing. No channel was frozen
at a ρ clamp, so the failure is the stopping criterion itself. **It is nevertheless by a wide margin the closest the
programme has come, and it supports the Addendum 15 mechanism.**

| | s32 (fixed γ) | s33e2 (γ = τρ) | **s34 (re-scaled)** |
|---|---|---|---|
| channels passing at termination | 0 | 1 (V) | **2 (V and PF)** |
| cycles with ≥2 channels passing | 0 | 0 | **18** |
| storage dual ratio (binding) | 5.56 | 3.245 | **2.843** |
| EFC/day | 0.1038 | 0.0671 | **0.8382** |
| network failures | 17 | 18 | **138** (133 T1, 5 T2, 0 unrecovered) |
| system cost (M) | 654.41 | 653.83 | **651.18** |

Per channel at termination (ratio = value / threshold; ✓ = passes):

| channel | primal | dual | first pass | cycles passing | ρ (frozen at cycle) |
|---|---|---|---|---|---|
| V | **0.045** ✓ | **0.154** ✓ | 45 | 106 | 0.01155 (11) |
| PF | **0.476** ✓ | **0.726** ✓ | 133 | 18 | 0.088 (backstop, 60) |
| ESS | **0.347** ✓ | **2.843** ✗ | — | 0 | 0.1125 (12) |

Run facts: exit 0, wall 5,914 s, 7,844 solves = 51×151 + 143 retries, **0 local-solve failures**, objective step 4,682
against tolerance 65,119 (rule ten 0.072), settlement cancellation closure −2.8e-7, 52 of 864 terminal interface
voltages within 1e-6 pu of a bound (330 within 0.005 pu). Scaling in force: σ fixed 9.363536e7 with the computed value
matching to **3e-9**, `al_scale_esso` = 2.2721e5, `S_ref` = 2.5 MVA. Balancing **raised** ρ_v and ρ_ess from their low
starts rather than collapsing them.

### 4.1 Storage: released, but not yet arrived

- **EFC/day rose 0.00029 → 0.8382, increasing in 100 % of cycles** — 12.5 × the `s33e2` terminal, **45 %** of `EFC*`
  and **57 %** of the SoH threshold — and was **still rising at the cap** (0.00215/cycle over cycles 120–150).
- **Extrapolating those measured rates** (descriptive, not a prediction): ≈**289 cycles** beyond 150 to reach the SoH
  threshold; the storage dual ratio decays ×0.925 per 30 cycles, implying ≈**403 cycles** to reach 1.0.
- **Spec v4's falsifier did not fire.** The per-entry storage step is **5.14 ×** `s33e2`'s against a ×3 threshold, so
  the conjunction "rise < 3× **and** direction persisting" fails on its first clause. The step norm decays slowly
  (0.225 at cycle 10 → 0.0455 at 150, −7.5 % over the last 30 cycles) while the direction stays persistent (cosine
  0.9998 at 150). **Storage is still walking toward its equilibrium, not oscillating about it.**

**Reading.** Addendum 15 predicted that removing the σ/AL stiffness would let storage reach its arbitrage equilibrium
within ~50 cycles with EFC/day of order 1. **The direction is confirmed and most of the magnitude is realised; the
timescale was optimistic by roughly an order of magnitude.** The storage dual residual cannot pass the Boyd test until
the underlying gradient is largely spent.

### 4.2 What this gate cannot say

- **The `S_ref` ×2.58 multiplier is UNTESTED.** ρ, the storage-operator scaling, the normalization and the trajectory
  all changed together; the terminal 2.843 against `s33e2`'s 3.245 compares two different dynamics. Recorded as
  untested rather than asserted, as the review warned a combined gate would require.
- **No difference against `s31c`, `s32` or `s33e2` is claimed.** No run has settled, so every bar is invalid.

### 4.3 The cost of starting ρ low

**138 network failures against 18** in `s33e2` — 7.7 ×, concentrated in the distribution blocks, all `maxIterations`,
**all recovered** (133 first-tier, 5 second-tier), none unrecovered, and none reached the coordination loop. The ρ/2
term is the main convexifier of the local nonlinear problems, so a low ρ start costs one extra solve per failure (143
retries, 1.8 % of solves). It did not corrupt the run, but it bounds how much lower ρ can sensibly start.

## 5. Questions for the Author and the Expert

1. **Reachability, and what the paper claims.** Addendum 15 fixed R2.5's optimality evidence as the Boyd residuals with
   noise-floor-derived tolerances, the 3.5 polish gap and the active-bound count. On the measured rates the storage
   channel will not satisfy the residual test for several hundred more cycles. Three ways forward:
   **(a)** one run at a higher cap (~400–500, ≈4 hours at the observed 39 s/cycle) to let the gradient spend itself —
   **the Planner's recommendation**, because every rate is now measured and one run converts extrapolation into result;
   **(b)** accept a per-channel verdict, documenting storage as non-certifiable at ε_abs 1e-5 while its arbitrage
   gradient is live, and proceed to 3.5; **(c)** revisit the storage channel's ε now that its normalization is
   candidate-independent. Which?
2. **Confirm the D5 form.** Authorized as "ESSO objective on the networks' σ convention"; implemented as an
   **argmin-identical** scaling of the storage operator's *coordination* terms (`al_scale·AL` rather than
   `base/al_scale`), verified numerically to 1e-9. The reason is protective: dividing its base objective by ~2e5 would
   shrink the 1e3 feasibility-slack penalty that **enforces the storage dynamics** to ~5e-3 against its coordination
   terms, and would weaken the barrier-leak identity that EFC/day itself depends on. The preflight detector gate
   confirmed no leak worsening. Is the form accepted?
3. **The success target.** Since `EFC*` (1.852) exceeds the SoH threshold (1.4612), should an equilibrium **at the
   threshold with the degradation constraint active** be defined as the success case, and reported as such?
4. **ρ policy.** Given §4.3, should any re-run start ρ_ess at 0.05 again, or at the 0.1125 that balancing itself chose?
5. **Economics still open from the last handoff.** Shared-ESS usage remains priced at zero (row 8) and the storage
   operator has no degradation cost on this branch, so `EPS_ESSO_THROUGHPUT` = 1e-5 remains the only price on
   throughput. `EFC*` shows what a price-taker would do; the gate shows the coordination now lets storage move toward
   it. Whether that is the intended economics is still an author decision.

## 6. Evidence index

| item | artifact | commit |
|---|---|---|
| Step 3.2 handoff (prior) | `P5_15_STEP32_EXPERT_REPORT.md` | `8d99e200` |
| Addendum 15 recorded; then corrected | `REVISION_CONTEXT.md` | `b63cc3de`, `81e7f1c5` |
| EFC archaeology, correction, Z4 + `EFC*` record | `P5_15_EFC_ARCHAEOLOGY_NOTE.md` | `749feb13`, `81e7f1c5`, `f37a1c63` |
| Z4 oscillation attribution | `P515S33/Z4/`, `WORKER_REPORT_S33_Z4.md` | `a3eca693` |
| `EFC*` benchmark | `P515S34/EFC_benchmark/`, `WORKER_REPORT_S34_EFC_BENCHMARK.md` | `d1b8cf9c` |
| frozen spec v4 | `P515S34/frozen_s34_spec_v4_966940a7.json` | `9f182b79` |
| implementation + checks + preflight | production, harness, `WORKER_REPORT_S34_IMPL.md` | `ef2040d7`, `db8eb90a` |
| evaluation script (committed before the gate) | `p515_s34_evaluate.py` | `566c5b20` |
| **gate s34: report and evidence** | `P5_15_S34_GATE_REPORT.md`, `P515S34_run/` (+ manifest, 477 files) | `202fd246` |

Every analysis script above runs with `SolveProfileGuard` armed and verified (zero permitted solves), except `EFC*`
(36 scipy LPs, zero Pyomo/IPOPT entries) and the preflights, which declare and check exact solve counts.
`esso_capture/` and `results/` are hash-recorded in the manifests rather than committed.
