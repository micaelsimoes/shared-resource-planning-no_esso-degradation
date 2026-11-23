# S56 validation step 4: design before build (2026-11-23)

**Status: design.** Nothing is built. The Advisor reviewed it (2026-11-23), and the Planner decided the items below
inside the authorized step (S56_PLAN_v3.md §7, row 4; ≈ 10–20 h). Two items go to the Expert. The spec is frozen
only after step 3 has reported and diagnostics D-1 to D-3 have run.

## Decisions needed (Expert)

| # | decision | recommendation |
|---|---|---|
| X4-1 | **Rule for choosing between V-1 and V-2.** Keep the decision with the Expert; step 4 supplies measurements M1–M5 (below) | **Eliminate V-2** if its ΔV plateau exceeds the ΔV gate, or if the hard-pin cost Δ_pin exceeds the allowance (V-2 is then a modelling restriction, not an approximation that can be validated). **Eliminate V-1** if its V channel does not contract, or if it fails local solves repeatedly while V-2 settles. **If both survive,** choose V-1 unless its cycles × wall time exceed V-2's by more than 1.5×. V-1 has an exact centralized comparator and matches "P/Q/V consistent per scenario" |
| X4-2 | **Tail depth through the whole run** (a production solver setting). **Conditional:** asked only if D-1 shows that the local-solver floor exceeds the per-block allowance | If asked: run at tail depth from cycle 1 in every arm, or else calibrate only on tail-depth cycles and report the floor as a limitation |

## Planner decisions

1. **Calibration arm: V-1 only.** Of the three arms, only F1 + V-1 converges to step 3's C_central exactly:
   - at α = 0 row 18 is absent;
   - no pin is built;
   - settlements cancel per scenario at consensus.

   F1 + V-2 converges to a soft-pin problem with no exact comparator. The pinned arm converges to a different problem.
2. **Pinned arm: measures D_b = C_central,b − Q_pin,b** (contracted-excluded Q). This is the re-optimized analogue of D0.
   - It is resolved only against the pinned arm's own settling bars.
   - It is not D0 rescaled: D0 was at fixed DN dispatch and α = 0.5.
   - The pinned arm stays out of the calibration.
3. **Arms:**
   - F1 + V-1;
   - F1 + V-2;
   - **pinned with AA off.** This is the like-for-like control, because AA is forced off under the option;
   - pinned with AA on (the published configuration), only if memory allows. It sizes what forcing AA off costs.
4. **Two blocks as two single-block derived instances,** not a mask. This needs no production change, and it decouples
   each block's ρ adaptation and stopping. Zero-solve gates before launch:
   - per-block data bitwise equal to the 12-block instance;
   - the block-weight ratio recorded (dropping 2025 rescales the discount: an effective ρ shift of 10–22 %, with the
     fixed point unchanged);
   - σ inside its assert window: the largest weighted block-agent objective ≥ 3.12e7;
   - the harness and the ESSO accept an instance without 2025;
   - NL identity of the ADMM-prepared block models, up to the recorded scale.
5. **Allowance per block.** R is not rescaled. Each block's error is reported in € of total discounted planning cost
   and tested as e_b ≤ (R/4) · w_b / Σ₁₂ w, using the 12-block instance's own weights.
   - Step 4 has no pair, so the level error bounds the pair error conservatively.
   - The working level is **provisional until step 5.**
   - It is expressed as per-block observables (ΔP_s, ΔQ_s, ΔV_s max and ω-RMS; A_b) and applied as a per-block gate,
     because the Boyd norms are global and do not transfer from two blocks to twelve.
   - Extending step 3 to all 12 blocks (132 solves) is to be weighed at the pilot.
6. **S is the validated extrapolation form,** not the 10-cycle range. Under slow linear contraction the range
   under-covers. The extrapolation is checked on held-out cycles, then against C_central.
7. **Capture**, per cycle, behind a capture-path assertion:
   - per-block Q in both conventions;
   - tail state and ρ;
   - entry-level per-scenario primal copies on both sides (P, Q, V; expectation and deviation);
   - every dual on both sides, including the expectation V dual.

   **PR is recomputed after the run.** Production's `priced_residual_pr` holds only the P deviation price
   (Σ w ω (π + μ^TSO_dev) ΔP), and its price part equals −`interface_settlement_total`. It is not D3's PR. The
   interface-dual sidecar stays in full; at two blocks it is about 0.2 MB per cycle.
8. **Pre-run diagnostics,** minute-scale, after step 3:
   - **D-1, local-solver floor.** Fix one block's per-scenario interface P, Q, V in the ADMM-prepared local models at
     step 3's solution. Solve the four local problems at production depth and at tail depth, and sum. If the offset
     is ≤ ⅓ of the allowance, proceed. If it is far above, the floor bounds R_eff and X4-2 is raised.
   - **D-2, hard-pin merged solve** per block (M2: Δ_pin), warm-started from step 3.
   - **D-3, ADMM-only boxes, zero-solve.** Check step 3's interface values against the ADMM-side boxes: the TSO
     per-scenario exchange in pc_init ± rating, which is centred on the DSO's initial expectation. That box is what
     makes the cycle-1 move exactly 2 × rating, in both the coupled and the uncoupled runs.
9. **Sequence and budget:**
   1. zero-solve gates;
   2. a serial 3-cycle smoke per arm, which measures RSS, s/cycle and the identities;
   3. the main run, cap 300, arms concurrent. The first 3 cycles must match the serial smoke bitwise, with timing
      fields excluded in advance, because the concurrency gate did not cover the F1 path;
   4. a continuation of V-1 (and V-2 if it survives) to 600 cycles, only if still contracting;
   5. **a declared look at cycle 40:** fit the contraction rate. If the projected cycles to the allowance exceed 600,
      stop and report.

   If the arms must run serially, the order is V-1, then V-2, then pinned AA off.
10. **Predictions,** frozen with the spec once step 3's numbers are in:

    | # | prediction |
    |---|---|
    | P1 | F1 identities every cycle |
    | P2 | initialisation of pinned ≡ V-2 bitwise; V-1 differs |
    | P3 | V-1 e_b contracts to max(allowance, floor) |
    | P4 | D_b ≥ 0 on the most-surplus block, and much smaller on the other. **Sign prediction: failure stops** |
    | P5 | lim Q_V2 ≤ C_hard-pin + bars |
    | P6 | cycle and wall ratios |
    | P7 | the full-channel PR reduces \|Q_k − C\| at least fivefold after cycle 20 |
    | P8 | the tail-onset step is recorded |
    | P9 | RSS ≤ 6 GiB per child, 45–75 s per cycle |

    **P4 supersedes** the coupling proposal's V1 prediction "C_central − C_pinned ≤ 0 on the high-curtailment
    block". That prediction was recorded at α = 0.5 with row 18, before D0. D0 found the pinned accounting optimistic
    by about 0.93 M€, which points to the opposite sign.

## Hypotheses for a non-zero terminal offset in V-1

| # | hypothesis | separated by |
|---|---|---|
| H1 | ADMM consensus error | contraction with the residuals |
| H2 | local-solver barrier and scaling floor (inferred 10³–10⁵ € per block; unmeasured) | D-1 |
| H3 | a different local optimum | step 3 default vs warm start; D-1 |
| H4 | a convention or bound mismatch (ADMM-only boxes, parameter alignment) | D-3 and the step 3 audit |

## Risks recorded

- **The comparator's own spread may be about the size of the target.** The two-block allowance is about 3e-6 of the
  two-block cost, so step 3's spread (default vs warm vs Σω singles) is compared with it first.
- **G and the ESS channels cannot be calibrated here,** because they vanish at x0. That is step 5.
