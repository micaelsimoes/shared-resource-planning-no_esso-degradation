# P5.15 S56: third handback to the Expert. Step 4 result and how calibration should stop (2026-11-23)

**What I am asking you to do.** Rule on decisions 1–3 below. Step 4 ran in about 1.7 h of the 10–20 h authorized.
Step 5 is held until you rule on decision 1, because its pair comparison depends on the answer.

The sources are listed here:

| content | file | commit |
|---|---|---|
| step 4 result | `P5_15_S56_STEP4_RESULT_REPORT.md` | `5f1db156` |
| zero-solve post-diagnostics | `P5_15_S56_STEP4_POSTDIAG_REPORT.md` | `598823f4` |
| post-diagnostics spec | `72f4d80e` | `9185df1b`, frozen before the run |

The Advisor's code reading is summarised under Found.

## Decisions needed

1. **How the calibration arm stops.** The per-block allowance, (R/4)·w_b/Σ₁₂w = 228.64 € (2025|Winter) or 187.56 €
   (2035|Summer), cannot be reached at production certification.
   - Certification admits about 38× the allowance in the unpriced settlement residual.
   - At the first certifying cycle on 2025|Winter that residual was 11.8× the allowance.
   - The allowance needs the interface ΔP to have ω-RMS ≈ 1.2–1.3e-4 MW. At exit it was 5.5e-4 MW on 2035|Summer
     and 9.4e-4 MW on 2025|Winter.

   Options:
   - **(a) Re-run V-1 (calibration arm) with `minimum_consecutive_converged_cycles` raised** (production default 10).
     - This is a configuration override for the calibration arm only. The trajectory up to the old exit is unchanged
       and must reproduce bitwise: 46 cycles on A, 107 on B.
     - Projected: A ≈ 106 cycles (≈ 37 min), B ≈ 210 cycles (≈ 2.6 h). Under 4 h, but not in the current order.
     - It is the only option that decides whether V-1's limit sits within the allowance on 2035|Summer, where it is
       not determinate now.
     - If the study's evaluations must stop the same way to support R, the per-evaluation cost rises by about 2×.
       The §8 projection would be redone with that.
   - **(b) Report the first-order-corrected estimate Q − price + dual as the calibration observable.**
     - This is an objective-convention change.
     - It is cheap, and it explains A (see Found). It is not determinate on B.
   - **(c) Report the limitation (R not demonstrable at production certification) and leave R.**

   **Recommendation: (a), with (b) reported beside it as a post hoc diagnostic.**
   - If (a) shows that R needs longer stopping, that is a cost fact the plan must carry. It is better measured now on
     two blocks than discovered in step 5.
   - The Tighter-Boyd-ε alternative is not recommended. It moves the tail-depth onset, so the path changes and the
     bitwise prefix is lost.
2. **M1 (F1 identities) failed as frozen, on rounding only. Recorded as failed, not re-scoped (G3 precedent).**
   - For later specs, I propose a tolerance of |a − b| ≤ c·u·Σ|summands| with c = 64, u = 2⁻⁵³, and an absolute floor
     of 1e-6 €.
   - Post-diagnostic D5 held under that rule on 277/277 cycles: worst 0.08–0.12 of the bound, signs mixed, and
     |Spearman ρ| ≤ 0.29 against |T| and against the residual. A wiring offset would show no such pattern.
   - **Recommendation: adopt it for later specs.**
3. **P7 had the wrong sign on its dominant term.**
   - Q is settlement-excluded-in-full, so the price term of the priced residual has to be subtracted. The frozen form
     added it.
   - For later specs, I propose the first-order estimate Q − price + dual.
   - Production's reporting field `priced_residual_pr`, (π + μ_TSO)·ΔP, mixes the two conventions. It is
     reporting-only: its dual part is 23–62 € here, and our evaluators compute their own value.
   - **Recommendation: re-define P7 for later specs, and leave the production field documented, not changed.**

## Blocked on the author

- **Push the new commits** when PyCharm is open: `e55eed90` through the commit of this report.
- **Merging the cohort follow-up branch `s56-cohort-followup`** needs your OK when its report arrives. The memory probes
  (e), the EV plumbing and the A/B run on the merged code.

## Changed

| commit | content |
|---|---|
| `4e6bb1d3`, `cb062834`, `765c2e5c`, `132524d8`, `5f1db156` | step 4 main run (spec v2 `7ca210fb`): preserved v1 attempt, spec, look-40 parameter script, evaluator, evidence and report |
| `9185df1b`, `598823f4` | post-diagnostics D1–D5 (spec `72f4d80e`, frozen before the run; guard counted exactly 0 solves) |
| `e55eed90`, `36dbcaeb` | TASKS |

No production file was changed.

## Found

The objective convention throughout is € of total discounted cost.
- Coupled Q: the settlement excluded in full.
- Pinned Q: the contracted settlement excluded.
- e_b = Q_b − w_b·C_central.

**Step 4 runs.**
- All 8 runs certified before the cap of 300.
  - Instance B (2025 + 2035 Summer): V-1 107 cycles, V-2 84, pinned AA-off 97, pinned AA-on 52.
  - Instance A (2025|Winter): 46, 40, 44, 36.
- 4,974 solves, equal to the declared base; no failures; peak RSS 3.53 GiB.

**Predictions against outcomes**

| # | prediction | outcome |
|---|---|---|
| M1 | identities to rel 1e-9 | **failed as frozen** (rounding; decision 2) |
| M2–M5 | | held |
| P3 | V-1 e_b(K) ≤ allowance | 2035\|Summer +110.13 ≤ 187.56, but a cancellation (below); **2025\|Winter failed**: −1,129.86 € |
| P4 | D_b ≥ 0 | **sign held** on both blocks: +61,054 € (2035\|Summer), +193,439 € (2025\|Winter); see below |
| P4 | "much smaller on the other" block | **failed** |
| P5 | | held |
| P6 | cycle and wall ratios | failed. The wall clock was contended by the cohort tests |
| P7 | Q + PR cuts \|Q − wC\| fivefold | **failed**: the sign was wrong (decision 3) |
| P9 | RSS | held |

**Mechanism on 2025|Winter: the certified point is not yet at consensus (H1).**
- The offset o1 = e_b − price_b, where price_b is the unpriced settlement, is positive and strictly decreasing over the
  tail. In V-1 it falls 349 → 197 € over cycles 38–46.
- The first-order estimate o2 = e_b − price + dual is +156.14 € at K. Recorded prediction: [100, 230]; held.
- The fitted limit of o2 is about 95–109 €, depending on the estimator. The frozen prediction placed it in [0, 100], so
  it **failed on its upper edge**.
- V-2, a restriction of the central problem, behaves the same way: o2 is +170.5 € and its limit about 155–162 €.
- So "below the optimum" is the sign an infeasible iterate gives, not a better solution. On A, Q − price + dual sits
  inside the allowance.
- What remains open: why o2 tends to about +100 €, not to the tail-depth solver floor of +45.67 €.

**2035|Summer: the "held" is a cancellation.**
- At K, e_b is −657.6 (price) + 767.75 (o1).
- o1 falls 997 → 768 over cycles 99–107, about 3.2 % per cycle. Its limit is not determinate: the estimators give
  −3,674, −432 and +540 €.
- **Ruled out by zero-solve checks:**
  - TSO/DSO dual inconsistency: λ_T = −λ_D bitwise at every coordinate in the last 10 cycles, and ρ_T = ρ_D.
  - An active ADMM-only box: the minimum margin is 27.3 MW / 97.9 MVAr.
- **Left:**
  - slow, second-order convergence;
  - a different local optimum.
- Only option (a) separates them.

**Interface voltage.** The interface V rows sit at their upper bound (1.1 pu) on every run, with V slacks fixed at 0
in the ADMM model. This is not a defect; it is recorded for step 5.

**P4, split by the Advisor.**

| block | wC − Q_pin (in full) | covariance settlement | D_b |
|---|---|---|---|
| 2025\|Winter | −153,820 € | +347,259 € | +193,439 € |
| 2035\|Summer | −768,511 € | +829,564 € | +61,054 € |

- The pinned-accounting term is larger in summer, as the premise had it.
- The two terms nearly cancel in summer, and not in winter.
- The term-by-term mapping onto your decomposition (optimism ≥ 0, DN re-optimisation gain ≤ 0) is the Advisor's.
  Please confirm it. D_b remains a two-block figure.

## Not confirmed

- **V-row bound multipliers** are not captured. Searched: the capture, the sidecar headers, `per_cycle_record`,
  `interface_voltage_terminal.json`, `boyd_terminal.json` and `component_levels_terminal.json`.
- **The projected cycle counts and hours under (a)** are a linear extrapolation of the tail decay, not a measurement.
- **The P6 wall ratios** were not re-measured on an idle machine.
