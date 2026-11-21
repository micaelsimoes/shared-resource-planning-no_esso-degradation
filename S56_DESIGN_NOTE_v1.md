# S56 design note v1: Advisor review of v0 applied (2026-11-21; predecessor `S56_DESIGN_NOTE_v0.md`, commit `6d677758`)

**Status: draft.** Nothing has been run beyond the zero-solve audits.

**The stochastic (SP) arm is on hold.** It waits for audit D: whether the multi-scenario formulation makes the
TN–DN exchanges physically consistent in each scenario (the author's third request). If it does not, the SP model
used in every decision below may change, and D1–D8 are revisited.

## Verdicts (Advisor) and the Planner's decisions

| v0 decision | Advisor | Planner |
|---|---|---|
| D1 grid 2025/2030/2035 + 3 × 3 ensemble | accept, amended | adopted, with SP at concurrency 1 |
| D2 EV = mean of the realised draws, injection hook | accept | adopted; A5 keeps production's constant |
| D3 EV with structurally absent commitment terms | accept | adopted; α declared in both arms; EV records `row18_wired: False` |
| D4 search acceptance | amend | adopted as a proposal; the acceptance rule is the Expert's |
| D5 common evaluation | accept, amended | adopted: 2 × 2 cross table; one frozen settling rule |
| D6 parallelism | accept, amended | adopted: bitwise equivalence gate; footprint margin on both instances |
| D7 core rerun scope | accept | adopted; the primary configuration is the author's choice |
| D8 pilot | amend | adopted: a settled pair under both models replaces the single SP x = 0 run |

## Corrections to v0

- **σ_Q is a fit residual, not a replicate spread.** σ_Q = 18,449.66 € is the maximum absolute residual of the Phase A
  ladder fit, measured under the C3 calibration (`p515_s47_phase_b_record.py:785–790`). The oracle is bitwise
  deterministic, so unperturbed replicates would give zero.
- **The harness refuses a one-scenario derived instance.** It raises unless the instance has more than one scenario
  combination (`p515_s44_campaign_harness.py:3317–3328`), so the EV arm needs a harness change on top of the injection
  hook.
- **SP × 2 concurrent is refused by the note's own memory gate.** The projected 11–12 GiB per SP child is above
  0.85 × 24.4 / 2 = 10.4 GiB. SP is planned at concurrency 1.
- **Drift is not specific to SP.** Single-scenario cells were also unsettled at certification (Addendum 52: E10 of
  29–74 k€; x = 0 about 15 k€ below its settled value).

## A structural expectation

The likely outcome is a null that follows from the model's structure.

- **Price uncertainty cancels.** The storage schedule is common to all scenarios
  (`model_construction_helpers.py:841–852`), and the energy price enters linearly. So in both arms the storage earns
  the mean-profile spread: 91.30 €/MWh on the mean profile, against 94.00 €/MWh as the expectation over scenarios.
- **What is left is a quantity-uncertainty (Jensen) effect** at kinks: curtailment hours, flexibility bounds and
  network limits. Its sign is not known.
- **The ensemble is modest and unstructured.** Every quantity is drawn independently, so aggregates average out (loads
  1.6 %, wind about 10 %, price 6.5 %), and price and RES are uncorrelated.
- **The claim's scope** must be stated as: day-ahead-committed (scenario-common) storage, this ensemble, the investment
  decision only. The common evaluation re-optimises the schedule and commitments; only x is fixed.

| hypothesis | statement |
|---|---|
| H1 | V_SP(x) ≈ V_EV(x) within resolution, so the searches select the same plan |
| H2 | kink effects are material, so the searches can diverge for a model reason |
| H3 | any divergence is an artefact of drift or local-solution selection |

## Amended search (D4): a proposal for the Expert, as it changes the acceptance rule

1. **Classify at the production exit in both arms** with max(bar_x + bar_inc, σ_S56). This keeps the published searches
   as a third, matched, arm.
2. **σ_S56 = max(18,449.66, D_EV, D_SP)**, frozen before the pilot. D_arm = |V_exit − V_settled| plus the settled band
   sum, from that arm's pilot pair. **One common σ for both arms**, with the per-arm values reported.
3. **Confirmation before any incumbent move:** settle the incumbent and the best improver. Move only if the settled
   difference also clears max(band sum, 2τ).
4. **Per-poll diagnostics, not gates:** the additivity residual of composite cells, E10 at the exit, and the
   terminal-step ratio.
5. **Batch size.** Keep the logical batch at the published 5 in both arms, and set the physical concurrency per arm.
   This needs PB launcher plumbing, since PB pins `CONCURRENCY = 5`. Comparable effort comes from the same seed, Δ₀,
   caps, Halton sequence, lattice and σ.

## Common evaluation (D5)

- **A 2 × 2 cross table:** F_EV and F_SP, each at x_EV and at x_SP.
- **One settling rule**, frozen with the uncertified reporting form. Rule v6 has not been demonstrated on a
  multi-scenario storage cell.
- **Never mix exit and settled values** in one comparison.
- **Pre-declared reading:** F_SP(x_EV) determinately below F_SP(x_SP) means a limitation of the SP search. The SP search
  is then continued from x_EV.

## Parallelism (D6)

- **Candidate-level poll concurrency.** The order is safe: classification follows the full poll, and ties break on
  (F, I, label).
- **Equivalence gate, bitwise.** Over 3 cycles, compare one candidate run alone and the same candidate inside a full
  batch, in separate roots, on the per-cycle record.
- **Memory margin:** the footprint, on both instances.
- **Within-stage parallelism is not in this study.** A production change mid-campaign would break the launcher pins,
  and the last persistent-worker gate predates AA, the tail and ageing.

## Additional requirements found

1. **An armed in-child assertion** that the arrays used by the run equal the means. This guards against the EV case
   silently running as the published draw-0 instance.
2. **Check the alpha-row capture path at one scenario** (harness l. 5752).
3. **Re-check the premium floor** (a positive mean price in every hour) on the new 2030/2035 draws.
4. **A frozen resume protocol** for multi-day attached campaigns. Evaluation directories are write-once.
5. **Manuscript:** the published single scenario is draw 0. It must not be described as an expected profile.

## Pilot (D8), replacing v0's

1. **Build gates:**
   - prefix property bitwise at 2030 and 2035;
   - mean price positive in every hour;
   - the 12-block structure;
   - the EV injection assertion.
2. **Settled pair under both models on the new grid:** x = 0 and the reference unit, under SP and under EV. Each runs
   through the production exit to settling in one process.
   - **Cost:** about 13 h for SP and about 3 h for EV.
   - **What it delivers:** the H1 test, D_EV and D_SP, cycles and footprint to settling, and two D7 fixed-plan cells.
3. **3-cycle concurrency runs:** EV × 5, and the bitwise equivalence gate.
4. **Cheaper first look, optional (about 4 h):** EV x = 0 and the EV unit on the five-block EV instance, against the
   existing V_3×3 = 236,695. Its band is one-sided, about 12 k€.

**Reading:**
- |V_SP − V_EV| below max(σ_S56, 3 × the larger band) supports H1. The author then decides whether the full SP search
  is still wanted, or a reduced form.
- Above it supports H2. In that case the value decomposition is captured before the searches.

## Predictions to record before the pilot (Advisor; frozen in the pilot spec)

1. The prefix property holds bitwise at 2030 and 2035; the mean price is positive in every hour.
2. SP at 12 blocks:
   - 235–260 s per cycle;
   - footprint above 10.4 GiB, so SP × 2 is refused.
3. SP x = 0:
   - exits in 60–95 cycles, still moving 1–5 k€ per cycle;
   - its settled value lies within 15 k€ of the exit.
4. EV x = 0:
   - exits in 120–200 cycles, with a post-exit excursion of 15–40 k€;
   - EV is not better settled than SP.
5. Q_SP(0) − Q_EV(0) > 0, of order 3 % of Q.
6. V_EV(u) / V_SRP1(u) equals the ratio of the mean-profile to the draw-0 spread, computed at zero solves first.
7. V_SP(u) / V_EV(u) ∈ [0.93, 1.03], with |V_SP − V_EV| < σ_Q. The sign is weakly negative (low confidence).
8. The differential drift D_SP ≤ 10 k€.
9. Serial against concurrent: bitwise identical; contention 1.12–1.22×.
10. At 1×, both arms end at x = 0, with several neighbours within resolution.
11. At 2×, both arms reach the same plan, or plans within resolution of each other under both models.

## Foreseeable scale (projection)

**The full SP search is about 34 evaluations at about 5 h each,** if 2× repeats the published path: 20 evaluations,
then the completion cap.
- **Elapsed time:** about 7 days at the exit, and about 9–11 days with confirmations and common evaluations. This is
  serial, at concurrency 1, on this machine.
- **Every SP evaluation exceeds 4 h,** so each launch must be named by the author.

## Decisions for the author and the Expert (collected; asked once audit D is in)

1. **The grid** (D1). The 2030/2035 draws are a new ensemble.
2. **α = 0.5 on the new instance.**
3. **The primary configuration** (D7). The only option needing no ruling is single-scenario primary, with SP as the
   uncertainty study.
4. **The acceptance rule** (D4: σ_S56 and confirmation). Expert.
5. **The benchmark under uncertainty:** the definition of the static arms. Expert; only if SP is made primary.
6. **Machine time:** the pilot's SP runs (about 13 h), then each SP search launch.
7. **The formulation change,** if audit D finds scenario-wise inconsistency. Expert.
