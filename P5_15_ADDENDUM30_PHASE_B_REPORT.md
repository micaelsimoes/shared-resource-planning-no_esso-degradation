# P5.15 Addenda 30–31: baseline set, ladders re-run, Phase B record terminates at x = 0. Stop for review

**Planner report, 2026-09-22.** For the External Expert and the Author; self-contained.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addenda 29–31.
- **Frozen spec:** v17, `data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json` (`abf0cb30`). All predictions were recorded in it before any run.
- **Baseline** (Addendum 30): C2, i.e. 10,000 cycles at 0.80 DoD to 0.80 retention (k = 35,851), plus calendar fade φ_cal = 0.985, plus soh_min = 0.70. AA-on, cap 500, 10 cycles, concurrency 5.
- **Sensitivity set:** the C3 results are now the sensitivity set and appear in no table below.
- **Objective convention on every table:**
  - Q = certified `gross_operational_cost`, with the settlement excluded.
  - F = I + Q.
  - Value = Q(0) − Q(x).
  - Salvage and the settlement remainder are reported separately and excluded from F.
- **Status:** nothing is running. The memory task (Addendum 29) waits for this review.

## 1. Verdict

1. **Under the baseline, x = 0 still minimises F, and Phase B's formal record says so with a certificate that holds.**
   - Phase B polled every feasible lattice neighbour of x = 0: 14 points, all certified, all worse, with 0 indeterminate and 0 barrier.
   - The closest is 0.25 / 0.5 at node 9 invested in 2030, at +33,459.
2. **All 30 ladder points and both re-certification checks certified.**
   - The closest ladder point is node 5, 0.25 / 1.0, at +53,607, against a resolution of 14,750.
   - **The 0.70 floor binds in the 2035 block at every point and at C\***, so the degradation-aware constraint is live throughout.
3. **The new baseline is marginally *less* favourable to storage than C3.**
   - At the smallest node-7 unit the value is 259,428 against 261,808 under C3.
   - C2's gentler cycle ageing is outweighed by the calendar fade and by the floor, which caps cycling in the last block.
4. **The storage prices at the TSO's bus-7 marginal cost, not at the market price** (Addendum 31 (2)).
   - That marginal cost's daily 4 h spread is **80.6 €/MWh**, against the market's 98.0.
   - About 40 % of the value is DSO flexibility savings, including at nodes 5 and 9, where there is no storage.
   - The baseline captures **60.4 €/MWh per full cycle**.

## 2. Baseline and re-certification (S1, S2)

- **Case file** (`2466401d`):
  - The values are written as decided.
  - The ESSO read-back gives k = 35,851.36, φ = 0.985 and soh_min = 0.70.
  - At x = 0, the NL files IPOPT receives are **byte-identical** before and after, so Q(0) does not depend on ageing.
  - A literal structural-digest check at x = 0 **failed**: the differences sit in deactivated degradation rows and in the salvage expression, which is outside the objective. It is recorded as failed, with the analysis alongside.
- **Evaluation identity** (`65525006`):
  - The ESS ageing parameters now enter the evaluation key for specs that declare a baseline.
  - All 127 committed spec entries and 100 records recompute to identical keys.
  - A baseline evaluation cannot collide with its C3 counterpart.
- **Re-certification** (`79b99b59`):

| point | cycles | Q | value | value − I | floor |
|---|---|---|---|---|---|
| C\* | 87 (107 under C3) | 650,912,327 | 2,947,134 | −749,116 | binds in 2035 at nodes 5, 7 and 9 |
| n7 0.25 / 1.0 | 112 | 653,600,033 | **259,428** | **−58,529** | binds in 2035 (dual 1,723) |

- **Predictions** (spec v17): both points certify, the floor binds in 2035, and the unit's value falls in 255–272k. **All held.**
- **Wider bars.** Bars are ~25k at both points, against 9–11k under C3, so resolution is coarser under this baseline.

## 3. A1a re-run under the baseline (S3, 30 points, all certified; `7c563e9b`)

| node, duration \ E (MWh) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| node 5, 4 h: F − F(0) | **+53,607** | +129,035 | +204,356 | +253,117 | +329,249 |
| node 7, 4 h | +58,529 | +140,712 | +198,950 | +270,129 | +353,512 |
| node 9, 4 h | +55,800 | +148,518 | +177,693 | +255,250 | +335,441 |
| node 7, 2 h | +110,653 | +237,872 | +362,131 | +471,603 | +605,085 |

Every one of the 30 points is worse than x = 0. **The floor binds in the 2035 block at all 30.**

**Node-7 surface and break-even** (`p515_s47_baseline_tables.py`, `2846d325`; formulas in the output):

- **Fitted value:** 10,379 (± 4,686) + **233,136 (± 2,294)·E** + 52,699 (± 4,820)·P EUR, over n = 10 points, with residual rms 5,287 and max 8,223.
- **Unit costs (2025):** energy 253,878 €/MWh, power 256,317 €/MVA.

| quantity | baseline |
|---|---|
| Energy cost at which a marginal 4 h MWh pays | **182,231 ± 2,591 €/MWh** (0.72 of the current cost) |
| Energy cost at which the first 4 h unit pays | 195,348 at node 7; 200,271 at node 5 (0.77–0.79) |
| Marginal value/cost at 2 / 4 h | 0.68 / 0.77 |
| Marginal value/cost at 6 / 8 / 10 h (extrapolated) | 0.82 / 0.84 / 0.85 |

**Repeatability.** n7 0.25 / 1.0 ran in S2 and again in S3 under the same evaluation key. The two runs match **bitwise**: 112 cycles and every per-cycle field. That is the sixth determinism reproduction, and the first under the baseline.

## 4. Phase B formal record (S4; `353e094b` spec, `fd0c82bf` run)

- **Method.** MADS/OrthoMADS on the lattice (§5), with the Planner's rulings frozen in the spec.
- **The x = 0 difficulty and the ruling (A2).**
  - From x = 0, every rounded OrthoMADS direction is infeasible, because of P = 0 ⇔ E = 0 and the 2–4 h duration bound.
  - The ruling (A2) adds the **poll completion** at every unit poll: every feasible lattice point within one step of the incumbent.
  - Option (b), reparametrizing each node as (zP, zE − zP), is **referred to the author** as a method change.
- **Run.** Polls at Δ = 4 and Δ = 2 had no feasible points. The unit poll evaluated the 14 completion points in batches of 5, 5 and 4.

| neighbour (0.25 / 0.5 at…) | I(x) | F − F(0) | resolution |
|---|---|---|---|
| node 9, 2030 | 141,882 | **+33,459** | 18,450 |
| node 7, 2030 | 141,882 | +42,268 | 18,450 |
| node 5, 2025 | 191,018 | +43,020 | 40,189 |
| node 5, 2030 | 141,882 | +43,676 | 18,450 |
| node 9, 2025 | 191,018 | +54,489 | 47,205 |
| node 7, 2025 | 191,018 | +57,411 | 22,145 |
| two-node combinations (6) | 283,764–382,036 | +97,072 to +130,639 | — |
| three-node combinations (2) | 425,646 / 573,055 | +132,204 / +151,807 | — |

- **Certificate: holds.** No feasible lattice neighbour of x = 0 improves F: 14 of 14 show no improvement, with 0 indeterminate and 0 barrier.
- **New evaluations:** 14.
- **Prediction S4** was "terminates at x = 0 with at most a few new evaluations". The termination held. The count was 14 rather than "a few", because of the completion ruling.
- **Open (A9).** The §6 check against NOMAD has not been done: PyNomad is not installed, and installing it would change the canonical environment.

## 5. Addendum 31: market uncertainty and the price signal (zero solves)

**(1) Paper-scale market spreads** (`046d4d00`).

| profile | daily 4 h spread (€/MWh, horizon-weighted) |
|---|---|
| scenarios 1–5 | 95.3 / 94.8 / 91.9 / 94.2 / 97.0 |
| **probability-weighted mean profile** | **91.7** |
| SRP1 (single scenario) | 98.0 |

- **SRP1's one scenario is paper scenario 1 in 2025.** It is drawn from the same pool with the same seed, and is identical on all four days. After 2025 it is an independent single draw; it is not the mean profile.
- **Prediction, recorded for the paper-scale evaluations:**
  - R = mean-profile spread / SRP1's = **0.937** with 2 % weights (0.936 undiscounted; 0.915 with growth removed).
  - That is, paper-scale value per MWh ≈ 6 % below SRP1's.
- **What drives R.** Most of R comes from the paper instance's different set of years. Averaging over scenarios flattens the spread by only 3.1 %. The convexity inequality the expert stated always holds by construction; the informative number is the size of the gap.

**(2) What the storage actually sees** (`b7aca555`).

- **The storage faces the bus-7 marginal cost, not the market price.**
  - The ESSO objective carries no price.
  - Its injection enters only the TSO bus-7 balance row.
  - The market price enters the TSO through generator cost and the interface settlement.
- **Recovering the bus-7 marginal cost without a solve.** Terminal duals are not saved. An exact first-order identity on the recorded PF duals gives the bus-7 marginal cost. It was verified on the one saved solved TSO block per run, to 7e-7 €/MWh, and holds at termination because `interface_delta_p` stays interior (min margin 25.9 MW).
- **Its daily 4 h spread at x = 0 is 80.6 €/MWh**, 0.823 of the market's 98.0 (per year: 0.96 / 0.76 / 0.78).
- **Attribution of value** at n7 0.25 / 1.0 under the baseline (259,428; the parts sum exactly):

| source | EUR | share |
|---|---|---|
| TSO generation cost | 156,495 | 60 % |
| DSO node 7 flexibility | 38,432 | 15 % |
| DSO node 9 flexibility | 39,528 | 15 % |
| DSO node 5 flexibility | 24,953 | 10 % |

  The storage at node 7 shifts TSO prices, and that reduces flexibility use in all three DNs.

- **Captured against available spread** (€/MWh per full cycle, baseline; every term computed):

| term | €/MWh |
|---|---|
| market spread | 97.0 |
| seasonal weighting | −11.9 |
| flatness of the bus-7 marginal cost | −12.3 |
| dispatch timing and depth | −6.5 |
| round-trip efficiency (0.97 × 0.96) | **−7.5** (the expert's "≈ €7" confirmed) |
| system remainder (TSO −22.3, DSO +24.0) | +1.6 |
| **captured** | **60.4** |

  The C3 figure of 52.1 belongs to the sensitivity set.
- **The paper-scale R cannot be restated on the marginal cost without paper-scale solves.** No paper-scale run has reached equilibrium. R = 0.937 therefore stays on the market spread, with this caveat attached.

**Size of the planning signal.**
- The smallest unit's value is **3.97 × 10⁻⁴** of the system cost (10^−3.40, one part in 2,520).
- σ_Q is 1.6–2.8 × 10⁻⁵ of it.
- **"Four orders of magnitude" overstates the gap.** The accurate sentence: *"the planning signal is about 4 × 10⁻⁴ of the system cost, and is resolvable only because the certification bar and the polish gap hold the cost to about 2 × 10⁻⁵"*.

## 6. Manuscript items (Addenda 30–31), with the measured figures

1. **Ageing-sensitivity band** (C3-era batch, labelled as such): the smallest unit needs ×1.21 more value. The measured multipliers were 1.13 (C2), 1.06 (C4), 1.05 (C2 + fade), 1.01 (mid-block) and 1.22 (no ageing, which only breaks even).
2. **Elasticity mechanism:** value rises with available energy at elasticity ≈ 0.6 at fixed power (`p515_s46_ageing_mechanism.py`).
3. **Break-even energy cost (baseline):** 182.2k €/MWh at the margin; 195–200k for the first unit.
4. **Discount curve** (C3-era): value-to-cost 0.905 / 0.823 / 0.726 / 0.651 at 0 / 2 / 5 / 8 %.
5. **Captured spread (baseline):** 60.4 against 97.0 available, decomposed as in §5 (2).
6. **Structural finding** (Z1 wording): the storage sits at the upstream terminal (DN bus 1) of the single constrained branch 1–2. Its injection enters the interface balance, not the branch flow. It is **active** in the binding slots but cannot change the constrained flow. Future work: DN-internal siting.
7. **Non-anticipativity:** the schedule is scenario-independent, so the stochastic value is the mean-price value. Recourse dispatch is future work.
8. **Signal size:** use the wording in §5, not "four orders of magnitude".

## 7. Questions for the Expert and the Author

1. **Citations for the baseline.** Addendum 30 asks for them to be recorded in the spec, but none was supplied and none exists in the repository. Spec v17 records them as *to be supplied by the author*. Please supply the datasheet source for C2, φ_cal and the 0.70 floor.
2. **Phase B method, option (b).** Should a future non-trivial Phase B reparametrize each node as (zP, zE − zP), so that OrthoMADS directions are feasible at x = 0? The completion ruling makes the present record sound either way.
3. **The NOMAD comparison (§6, A9)** needs PyNomad installed in the canonical environment. Authorize it, or record the comparison as not done?
4. **One terminal TSO capture.** A single solve-bearing capture of a terminal TSO solve would identify which constraints push the bus-7 marginal cost away from the market price in 2030/2035. Is that wanted, or is the recovered spread sufficient?
5. **Memory task.** It is next, with the machine alone, under the decision rule in spec v16: resident memory ≤ 26 GiB after initialization, flat across cycles, and a cycle ≤ 25 min. Confirm it proceeds.
6. **Anomaly to note.** Some penalty components carry negative levels (e.g. voltage slack −11,797 at the baseline unit). They cancel in every value, but the sign is unexplained. Should a zero-solve check be run?

## 8. Evidence

| item | commit |
|---|---|
| spec v17 | `abf0cb30` |
| market spreads (A31 item 1); TSO marginal cost (A31 item 2) | `046d4d00`; `b7aca555` |
| key identity; case file; baseline launcher; S2/S3 specs | `65525006`; `2466401d`; `6642b33b`; `5000ed8f` |
| S2 run; S3 run | `79b99b59`; `7c563e9b` |
| Phase B launcher and rulings; Phase B spec; Phase B run | `4b8854b0`, `f6f86888`; `353e094b`; `fd0c82bf` |
| baseline tables | `2846d325` |

- Every zero-solve claim is backed by an armed `SolveProfileGuard`.
- Persisted models, per-entry strides, `esso_capture/` and `results/` are hash-recorded in the manifests.
