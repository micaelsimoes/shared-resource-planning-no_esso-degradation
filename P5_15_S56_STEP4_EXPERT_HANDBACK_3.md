# P5.15 S56: third handback to the Expert. Step 4 result, calibration stopping, cohort follow-up (2026-11-23)

**What I am asking you to do.** Rule on decisions 1–4.
- Decision 1 decides how step 5 is run, so **step 5 is held until you rule on it**.
- Decisions 2–4 apply to later specs and to the cohort formulation's record.
- Nothing else waits on you. The harness wiring of the cohort rows follows once the author merges the branch.

Step 4 used about 1.7 h of its authorized 10–20 h.

| content | file | commit |
|---|---|---|
| step 4 result | `P5_15_S56_STEP4_RESULT_REPORT.md` | `5f1db156` (spec `7ca210fb`) |
| step 4 zero-solve post-diagnostics | `P5_15_S56_STEP4_POSTDIAG_REPORT.md` | `598823f4` (spec `72f4d80e`, frozen at `9185df1b` before the run) |
| cohort follow-up (D1, D2, ruling 3, efficiency gate) | `P5_15_S56_COHORT_FOLLOWUP_REPORT.md`, on branch `s56-cohort-followup` | `a2077281` (spec `287c0da6`) |

## Decisions needed

### 1. How the calibration arm stops. Step 5 depends on this.

**The per-block allowance cannot be reached at production certification.** The allowance, (R/4)·w_b/Σ₁₂w, is
228.64 € (2025|Winter) and 187.56 € (2035|Summer).
- The binding term is the unpriced settlement residual Σwωπ(P_T − P_D).
- At the allowance's resolution, the interface ΔP needs an ω-RMS of about 1.2–1.3e-4 MW. At exit it was 9.4e-4 MW
  (2025|Winter) and 5.5e-4 MW (2035|Summer).
- At the first certifying cycle on 2025|Winter the residual was 11.8× the allowance. The Boyd stop (ratio 1) admits
  about 38×.

**Options**

| | option | change | cost | what it settles |
|---|---|---|---|---|
| (a) | Re-run the calibration arm V-1 with `minimum_consecutive_converged_cycles` raised (production default 10) | configuration override, calibration arm only | ≈ 37 min (A, ≈ 106 cycles); ≈ 2.6 h (B, ≈ 210 cycles); projected | whether V-1's limit lies within the allowance on 2035\|Summer (not determinate now); the trajectory up to the old exit (46 / 107 cycles) must reproduce bitwise |
| (b) | Report the first-order-corrected estimate Q − price + dual as the calibration observable | objective-convention change | none | explains 2025\|Winter; does not settle 2035\|Summer |
| (c) | Report the limitation ("R not demonstrated at production certification") and leave R | none | none | nothing further |

**Recommendation: (a), with (b) reported beside it as a post hoc diagnostic.**
- (a) is the only option that settles 2035|Summer.
- If R needs longer stopping, each study evaluation costs roughly 2× more. Better to measure that now on two blocks than
  to discover it in step 5. The plan's hour projection would be redone with it.
- Tightening the Boyd ε instead is not recommended. It moves the tail-depth onset, which changes the path and loses the
  bitwise prefix.

### 2. M1 (the F1 identities) failed as frozen, on rounding only

The failure is recorded and not re-scoped, as in the G3 precedent.

**Proposal for later specs:** |a − b| ≤ c·u·Σ|summands|, with c = 64, u = 2⁻⁵³, and an absolute floor of 1e-6 €.
- Under that rule the post-diagnostics held on 277/277 cycles: worst ratio 0.08–0.12 of the bound, signs mixed.
- |Spearman ρ| was ≤ 0.29 against both |T| and the residual. A wiring offset would be one-signed or would scale with a
  physical quantity.

**Recommendation: adopt.**

### 3. P7 (the priced-residual correction) had the wrong sign on its dominant term

- Q is "settlement excluded in full", so the price term has to be subtracted; the frozen form added it.
- **Proposal for later specs:** the first-order estimate Q − price + dual.
- Production's reporting field `priced_residual_pr`, (π + μ_TSO)·ΔP, mixes the two conventions.
  - It is reporting only. Its dual part here is 23–62 €.
  - Our evaluators compute their own value.

**Recommendation: re-define P7; document the production field and do not change it.**

### 4. The cohort complementarity row leaves a residual at its designed tolerance (T8)

**The prediction and the result.** T8 predicted within-cohort simultaneity ≤ 1e-6 MWh/day on the dump positive
control. It measured **2.475e-4 MWh/day**, against 0.8508 before the row. That is a 3,400× reduction, but it **failed
as frozen**.

**Why the prediction could not hold.** The row is the relaxed product (p_ch/S_c)(p_dch/S_c) ≤ 1e-4. That is the form
and tolerance of the network `sess_comp` row.
- With charging at 0.122 MW and S_c = 0.25 MVA, the row admits 5.1e-5 MW of discharge in the same hour.
- The leak sits at that bound: the row value is 1.00006–1.00008e-4 on all 8 multi-cohort block-days.
- It does not depend on the barrier parameter. A 3.1× change in final μ changes the leak by 2e-7 relative.
- It is distinct from the old barrier-level leak of 1.0e-4 MWh. That leak was symmetric, spread across the day and
  non-binding, and it is unchanged.

**Options**
- **(a)** Accept the residual as the row's designed slack. E2 is then met as "restricted", and the residual is reported.
- **(b)** Tighten the cohort row to, for example, 1e-6. This needs a new spec, the NL identity on multi-cohort builds,
  and two millisecond solves.
- **(c)** Exact complementarity.

**Recommendation: (a).**

## Blocked on the author

- **Merge** `s56-cohort-followup` into `feature/derivative-free-planning`. The merge tree is clean, and main has not
  touched the five production files since the branch point.
- **Push** the new commits, `e55eed90` through the commit of this report.

## Changed

| commit | content |
|---|---|
| `4e6bb1d3`, `cb062834`, `765c2e5c`, `132524d8`, `5f1db156` | step 4 main run: preserved v1 attempt, spec v2 `7ca210fb`, look-40 parameter script, evaluator, evidence (manifest `59bf89da`) and report |
| `9185df1b`, `598823f4` | step 4 post-diagnostics D1–D5 (spec `72f4d80e`; guard exactly 0 solves) |
| `437efb48`, `a2077281` (branch) | cohort follow-up: production change in 5 files (see below), spec `287c0da6` run evidence with manifest, T8 diagnosis (0 solves), report |

**The cohort change** is built on multi-cohort node-years only. It contains:
- network `sess_comp` off (D1);
- the per-cohort complementarity row, active (ruling 3);
- the per-cohort sum row p_ch + p_dch ≤ s_c;
- exact per-cohort closure, with no slack (D2);
- the efficiency-equality gate;
- the single-cohort activation option for the A/B.

**Default path:** the NL output is bitwise unchanged (T1), and all 118 fixtures load (T2).

**Not built yet.** These follow after the merge, under their own spec:
- the exit-certificate reading of the Q slack, the per-cohort closure and the detectors (E6);
- `cohort_formulation` in the harness keys and records;
- the new sync step in `uncoordinated_benchmark.py`, which builds network models without it.

## Found

**Objective convention (every value):** € of total discounted cost.
- Coupled Q: settlement excluded in full.
- Pinned Q: contracted settlement excluded.
- e_b = Q_b − w_b·C_central.

### Step 4 runs

- **All 8 runs certified before the cap of 300.**
  - B (2025 + 2035 Summer): V-1 107 cycles, V-2 84, pinned AA-off 97, AA-on 52.
  - A (2025|Winter): 46, 40, 44, 36.
- 4,974 solves, equal to the declared base; 0 failures; peak RSS 3.53 GiB.

| # | prediction | outcome |
|---|---|---|
| M1 | identities to rel 1e-9 | **failed as frozen** (rounding; decision 2) |
| M2–M5 | init identity, bitwise smoke prefix, capture, solve count | held |
| P3 | V-1 e_b(K) ≤ allowance | 2035\|Summer +110.13 ≤ 187.56, but a cancellation (below); **2025\|Winter failed**: −1,129.86 € |
| P4 | D_b ≥ 0 | sign **held**: +61,054 € (2035\|Summer) and +193,439 € (2025\|Winter) |
| P4 | "much smaller on the other block" | **failed** |
| P5 | lim Q_V2 ≤ C_hard-pin + bars | held |
| P6 | cycle and wall ratios | failed (the wall clock was contended by the cohort tests) |
| P7 | Q + PR cuts \|Q − wC\| fivefold | **failed**: wrong sign (decision 3) |
| P9 | RSS ≤ 6 GiB | held |

### 2025|Winter: the certified point is not yet at consensus (H1)

**Measured:**
- o1 = e_b − price (the unpriced settlement) is positive and strictly decreasing. In V-1 it falls from 349 to 197 €
  over cycles 38–46.
- o2 = e_b − price + dual is +156.14 € at K. The recorded range was [100, 230]; held.

**Fitted (not measured):**
- The limit of o2 is about 95–109 €, depending on the estimator. The recorded range was [0, 100], so it **failed on its
  upper edge**.

**V-2,** a restriction of the central problem, behaves the same way: o2 is +170.5 €, with a fitted limit of
155–162 €.

**Conclusion:** "below the optimum" is the sign an infeasible iterate gives, not a better solution.

**Open:** why o2 tends to about +100 €, rather than the tail-depth solver floor of +45.67 €.

### 2035|Summer: the "held" is a cancellation

- At K, e_b is −657.6 (price) + 767.75 (o1).
- o1 falls from 997 to 768 over cycles 99–107, about 3.2 % per cycle. Its limit is not determinate: the estimators
  give −3,674, −432 and +540 €.
- **Ruled out by zero-solve checks:**
  - dual inconsistency: λ_T = −λ_D bitwise at every coordinate in the last 10 cycles, and ρ_T = ρ_D;
  - an active ADMM-only box: the minimum margin is 27.3 MW / 97.9 MVAr.
- **Left:** slow second-order convergence, or a different local optimum. Only decision 1 (a) separates them.

### Interface voltage

The interface V rows sit at their upper bound of 1.1 pu on every run, with the V slacks fixed at 0 in the ADMM model.
This is not a defect. It is recorded for step 5.

### P4 split (the Advisor's mapping; please confirm)

| block | wC − Q_pin (in full) | covariance settlement | D_b |
|---|---|---|---|
| 2025\|Winter | −153,820 € | +347,259 € | +193,439 € |
| 2035\|Summer | −768,511 € | +829,564 € | +61,054 € |

- The pinned-accounting term is 2.4–3× larger in summer, as the premise had it.
- The two terms nearly cancel in summer, and not in winter.
- The signs match your decomposition (optimism ≥ 0, DN re-optimisation gain ≤ 0) term by term.
- D_b remains a two-block figure.

### Cohort follow-up

| test | outcome |
|---|---|
| T1–T7, T10 | held |
| T8 | **failed as frozen** (decision 4) |
| T9 ("unchanged elsewhere") | **failed**: 14 unpredicted changes, recorded and not re-scoped |

**T9 detail:**
- 10 of the changes are exactly the removed closure slacks, n × 1e3 × 1e-8.
- 4 are ≤ 1.5e-3 and unexplained.

**With TV on, inter-cohort transfer fell:**

| | transfer (MWh/day) | losses (MWh/day) |
|---|---|---|
| before the row | 0.30–0.85 | 0.021–0.061 |
| after the row | 0.12–0.15 | 0.008–0.011 |

E2 allows transfers with their losses accounted. Whether part of the old dump now moves through transfers is not
established.

**Production's own leak diagnostic would pass the T8 residual as barrier-level:** its bound is 9.4e-3 MWh against the
measured 3.9e-3 MWh. The E6 certificate wiring will read the row value directly.

## Not confirmed

- **Interface V-row bound multipliers.** Not captured. Searched: the capture, the sidecar headers, `per_cycle_record`,
  `interface_voltage_terminal.json`, `boyd_terminal.json` and `component_levels_terminal.json`.
- **The cycle counts and hours under decision 1 (a).** A linear extrapolation of the tail decay, not a measurement.
- **P6 wall ratios.** Not re-measured on an idle machine.
- **T8 binding.** It rests on the row value sitting at its bound and on its μ-independence. It does not rest on a
  positive dual: the hourly profile and the row multipliers were not saved. Two millisecond solves would settle it.
- **The 4 unexplained T9 changes.**
