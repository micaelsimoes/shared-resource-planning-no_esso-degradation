# P5.15 S56: D_b TN-only re-solve, T8 binding check, T9 three-change explanation (2026-11-23)

**Authority:** TASKS.md, "Expert rulings on handback 3 (2026-11-23)", items 4 and 5; the Planner's three tasks of
2026-11-23.

**Isolation:** every solve ran from a clean `git archive` export of **`4c586e5e`**. The working directory was the
export, so relative solver logs stayed there; no stray file appeared outside the output directories.
- The stage files were overlaid byte for byte from the commit that froze them (`d1acfb80`), as additions only.
- `EXPORT_PROVENANCE.json` records both commits and the sha256 of every root file. Each stage script checked itself,
  its spec and every imported root module against that record before running, and again at the end: 0 differences.
- Copies of the record sit in the roots as `EXPORT_PROVENANCE_run.json` and `EXPORT_PROVENANCE_zero_solve_diag.json`.
- Outputs were copied into the new roots in the main checkout.
- No production or harness file was edited. The cohort follow-up harness was imported unedited.
- No ADMM run. `TASKS.md`, `step4_main_v2/`, `step4_postdiag_v1/` and `cohort_followup_v1/` were only read.

## Blocked on Planner

1. **Task 1 cannot give the causal split as specified.** Arm B solved on none of the three blocks.
   - **What happened:** the TN-only re-solve at the realized DN exchanges ended "Converged to a point of local
     infeasibility" on 2025|Winter, 2035|Summer and 2025|Summer. The unscaled constraint violation at exit was 0.103,
     0.062 and 0.064 p.u.
   - **Consequence:** B1, B2 and B3 are **indeterminate** on both calibration blocks. The failure policy forbids a
     re-run or any tuning.
   - **Structural cause** (zero-solve check, written after the run):
     - The pinned arm's realized DN exchanges contain **net DN exports into the TN**. Summed over the three DNs, the
       minimum is −63.9 MW on 2025|Winter, −60.0 MW on 2035|Summer and −60.1 MW on 2025|Summer. This happens in 6,
       15 and 6 scenario-hours.
     - The TSO's scenario-common exchange is always a net import: at least +43.5, +32.7 and +37.9 MW.
     - The TN has no load outside the three ADN buses. Every generator has p_g ≥ 0 (lower bounds 0 to −0.001 MW).
       There is no node-balance slack and no storage at x0.
     - So an export can leave the TN only as network losses.
   - **This is not a proof of infeasibility.** The loss upper bound Σ r·(rate/v_min)² is 66.2 MW, which exceeds the
     largest export. IPOPT's "local infeasibility" is the only direct evidence.
   - **Options:**
     - **(a)** Re-define the intermediate point so the TN can refuse a DN export, for example a priced curtailment of
       the export at the TN side, or node-balance slacks at a declared price. This changes what the split measures,
       so it is an Expert ruling. Cost: 6 solves, new spec.
     - **(b)** Per-scenario TN re-solves (27 solves) on the scenarios without net export. Those are 6 of 9 scenarios on
       2025|Winter and 3 of 9 on each Summer block. This gives a partial split on that probability mass only, never a
       block total.
     - **(c)** Record that the D0-type composite does not exist for this instance and keep D_b provisional, as the step
       4 record already states.
   - **Recommendation:** (c), and (a) only if the Expert still wants a split. The finding itself is informative: the
     pinned arm settles financially some exchanges that the TN, alone, does not physically serve.
2. **The leak detector's verdict depends on the pricing form.** The ceiling is λ·L ≤ 1 % of R/4 = 25 €. On T8:
   - L priced in full at the TN expected day-ahead energy price gives **97.99 €** (discounted), 113.89 € undiscounted.
     **Above the ceiling.**
   - Only the conversion loss priced, (1/η_dch − η_ch)·L = 7.2 % of L, gives **7.02 €**. **Within the ceiling.**
   - The spec declared the full-L form as primary, as a conservative upper bound. **Which form does the ceiling mean?**
     This must be settled before the harness wiring spec freezes the detector.
3. **T9: one of the three changes is only partly explained** (t_ii_b, below).
   - 34 % of the change is caused by the new rows.
   - The other 66 % (−1.58e-4 MWh) appears when only the closure slacks are removed. The objective is unchanged to
     3.5e-11.
   - That is a measured attribution, not a demonstrated mechanism. Does it satisfy the Expert's "explain before the
     wiring spec"?

## Changed

| commit | content |
|---|---|
| `d1acfb80` | **before any solve:** spec v1 `04dd2bbe` (Task 1; no predecessor); spec v1 `c6c244c7` (Tasks 2 and 3; no predecessor); `p515_s56_db_tn_resolve.py`, `p515_s56_t8_binding_t9.py`, `p515_s56_db_t8_export.py` |
| `d7a9897b` | run evidence of both roots (except the reader-diagram PDFs) and `p515_s56_db_t9_zero_solve_diag.py` |
| (this commit) | this report; zero-solve diagnostics output `step4_db_resolve_v1/zero_solve_diag_v1/` and its logs; the two stage manifests `step4_db_resolve_v1_manifest_sha256.json` (69 files, 17 committed) and `cohort_t8_binding_v1_manifest_sha256.json` (232 files, 24 committed). The PDFs are hash-recorded only, as in the step 3 precedent |

**Runs** (attached, stdout and stderr captured; every stderr log is 0 bytes):
- **Task 1:** 32 s, 6 solves. The guard counted exactly 6 solves and 6 executions. The imported D0 guards counted 0;
  pickle 0.
- **Tasks 2 and 3:** 53 s, children 2 + 3 + 1 = 6 solves. The stage guard and the harness-module guard each counted
  exactly the declared number per child. The wrapper call count equals the declared count. The parent counted 0.
  Pickle 0. Every solve was "Optimal", one attempt, no recovery.
- **Diagnostics:** zero solves; guard verified at 0.

**Scripts:**
- `p515_s56_db_tn_resolve.py` re-uses D0 v2's `build_arm_v2` and v1's `block_data` / `extract`.
- `p515_s56_t8_binding_t9.py` calls the cohort follow-up harness's own `child_solves` with a spec tag. A class-level
  wrapper around production `SharedEnergyStorageData.optimize` reads duals and the hourly profile after each solve.
  For the T9 tags it also deactivates the declared row families before the solve.
- The zero-solve trials passed before the freeze: 131 checks in Task 1, and every T8/T9 test model built and captured
  with `--no-solve`.

## Found

### Task 1: D_b causal split

**Objective convention:** EUR of total discounted cost. w_b·C_central is gross: model.objective with settlement
weight 0, no pin, no row 18. Q_pin is contracted-settlement-excluded = w_b[TSO base + Σ_DN (DSO base + DSO covariance)].
C_TN-resolve = w_b[C_TN,B + Σ_DN DSO base].

| block (instance, K) | w_b | w_b·C_central | Q_pin | D_b (= step 4, bitwise) | Arm B (realized) | Arm A (control) |
|---|---|---|---|---|---|---|
| 2025\|Winter (A, 44) | 455 | 79,710,848.83 | 79,517,409.71 | +193,439.12 | local infeasibility, 146 it., viol. 0.103 p.u. | Optimal, 31 it. |
| 2035\|Summer (B, 97) | 373.258 | 57,114,972.23 | 57,053,918.54 | +61,053.69 | local infeasibility, 138 it., viol. 0.062 p.u. | Optimal, 37 it. |
| 2025\|Summer (B, 97; merged-only, not deciding) | 455 | 84,481,740.87 | 84,409,091.98 | +72,648.88 | local infeasibility, 138 it., viol. 0.064 p.u. | Optimal, 31 it. |

- **Instances:** pinned AA-off eval `f7d15674…` (A) and `28ae4f19…` (B), x0. Inputs were sha256-checked against the
  child manifests.
- **Q_pin recomputed** from the terminal multiscenario record equals the capture to rel ≤ 1e-12. D_b equals step 4's
  value bitwise.

| # | prediction | outcome |
|---|---|---|
| B1 | T1 = C_TN-resolve − Q_pin > 0 on both calibration blocks (numeric: T1 ≥ D_b − τ_b) | **indeterminate**: Arm B not loaded on either block |
| B2 | T2 = w_b·C_central − C_TN-resolve ≤ τ_b = w_b × 2 €/day (910.00 €; 746.52 €) | **indeterminate**: Arm B not loaded |
| B3 | T1 + T2 = D_b to \|a − b\| ≤ 64u·Σ\|summands\|, floor 1e-6 € | **indeterminate** (nothing to sum) |
| P_solves | 6/6 strict Optimal | **failed**: 3/6 (Arm A only) |
| CTRL (not deciding) | w_b(C_TN,A − TSO base) ≤ τ_b | **held** on all three: −50.75 € (2025\|Winter), −893.72 € (2035\|Summer), −820.92 € (2025\|Summer) |

- **CTRL formula and inputs.** w_b × (Arm A model.objective − TSO block_recourse_production_local), both in the run
  record. The script evaluates CTRL only when Arm B is valid too, so these values were computed from the record.
- **CTRL inputs per day:**

  | block | C_TN,A (€/day) | TSO base (€/day) |
  |---|---|---|
  | 2025\|Winter | 161,897.705 | 161,897.816 |
  | 2035\|Summer | 64,726.367 | 64,728.761 |
  | 2025\|Summer | 136,020.642 | 136,022.446 |

- **Resolution of the terms that could not be formed** (last-10-cycle max |step|):
  - Q_pin: 2,139 € (A), 1,976 € (B 2035), 2,096 € (B 2025);
  - w·Σ DN base: 657 €, 350 €, 217 €.

### Task 2: T8 binding check

**Second solve = T8b.** The report names T8b as the other case that failed T8 at the same L, and the
μ-independence argument rests on it. The 1e-6-tolerance alternative would be a formulation perturbation, which the
Expert did not ask for (decision 4 accepted (a)).

| # | prediction | outcome |
|---|---|---|
| T8-B1 | row dual nonzero wherever the row value ≥ 0.9999e-4 (operational: \|dual\| > 0 and min binding \|dual\| ≥ 100 × max \|dual\| over rows with value ≤ 0.5e-4) | **held** (T8 and T8b). Details below the table |
| T8-B2 | the first solve reproduces the frozen T8 bitwise | **held bitwise** (T8 and T8b). Details below the table |
| T8-B3 | the leak hours are the charge hours 1–4 | **failed as frozen.** Details below the table |

- **T8-B1 in numbers:**
  - 40 binding rows: the 2025 cohort, 8 multi-cohort block-days × 5 periods.
  - |dual| 0.0181–0.0580, all negative: Pyomo's sign for an upper-bounded row in a minimisation.
  - Non-binding rows (440): |dual| ≤ 9.09e-7, so the separation is ≥ 1.99e4.
  - The sum-row duals are ≤ 1.3e-9, i.e. inactive.
  - Bound multipliers at the binding hours are barrier-level, since both powers are interior: z_L(p_dch) 5.1e-10 to
    1.8e-6 and z_L(p_ch) 7.4e-10 to 2.6e-6 (the smaller power carries the larger multiplier: p_dch in charge hours,
    p_ch in period 0).
- **T8-B2 in numbers:** admm_objective −147.2438176611438, feasibility_penalty −0.009448425819217844, and
  L = 2.475074215145362e-4 MWh/day, all equal to the frozen record. μ_final 3.278e-11, equal to the frozen value.
- **T8-B3 in numbers:**
  - The binding periods are **{0, 1, 2, 3, 4}** on all 8 block-days, not {1, 2, 3, 4}.
  - **Period 0 is an inter-cohort transfer hour.** The request is 0 there. The 2025 cohort discharges 0.1438 MW with
    p_ch 4.35e-5 MW on the row's bound, while the 2030 cohort charges 0.1157 MW (2030|Spring).
  - Share of L in periods 1–4: 0.825–0.826 (2030 blocks) and 0.852–0.853 (2035 blocks). The rest is in period 0.
    Every other hour is 0.

**Hourly profile:** `cohort_t8_binding_v1/run/work_binding/stage_capture_binding.json` (`963d592b…`). It holds p_ch,
p_dch, SoC, the complementarity and sum row values and duals, and z_L/z_U of p_ch and p_dch, per cohort, block-day and
period, for T8 (capture `1`) and T8b (capture `2`).

**Leak detector** (T8; T8b equal within 2e-8 relative):
- L = Σ max(min(p_ch, p_dch), 0)·1 h = 1.941e-3 MWh over the representative days. All of it falls on the 2025 cohort
  in the 2030 and 2035 blocks, periods 0–4.
- Over the horizon, weighted by days × years: 0.886 MWh undiscounted, 0.765 MWh discounted (r = 0.02).
- **Price used:** the TN expected day-ahead energy price of that block-day and hour (SRP1 case, one market scenario;
  73–217 €/MWh over the leak hours, periods 0–4 of the 2030 and 2035 block-days). It is a € price directly comparable with the € ceiling. The row
  dual is in σ-scaled ESSO objective units, so it was not used.

| form | λ·L (€) | ceiling 25 € |
|---|---|---|
| L in full × energy price, discounted (declared primary) | 97.99 | **above** |
| same, undiscounted | 113.89 | above |
| conversion loss only, (1/η_dch − η_ch)·L × price, discounted | 7.02 | within |
| row-dual implied (objective units, beside) | 1.49e-4 | not comparable |

### Task 3: the three undiagnosed T9 changes

**Method.** Four ESSO solves under spec c6c244c7:
- t_ii_b, t_iii_indep and t_iv_peak in the after-build with **both new row families deactivated** (`t9_rows_off`);
- t_ii_b with **only the sum rows** deactivated (`t9_sum_off`).

Zero-solve: the committed T1 symbolic NL lists for the t_ii_b / t_iv_peak plan (srp1 p2025_2030, independent_units,
node 7). Between the v2 build and the after build exactly these items differ: +384 sum rows, +384 complementarity rows,
and −32 closure-slack columns (16 up, 16 down). The common rows and columns appear in the same order, in both the
init and the admm NL. So the rows-off solve differs from v2 **only** by the removed closure slacks. The NL diff does not
cover t_iii_indep's plan (2025 + 2035), because no T1 NL was written for it.

**Common fact.** In all three tests the objective (= the feasibility penalty) does not see the free dispatch. Rows-off
equals after to 3.5e-11, 1.4e-13 and 3.4e-11. After − v2 is exactly the closure-slack constant (+3.19997e-4, …).
Each moved quantity therefore lies on a degenerate optimal face, and the point IPOPT returns on it is set by the
barrier, not by the objective.

| # | prediction | outcome |
|---|---|---|
| T9-R | with rows 6b/6c off, each quantity returns to its v2 value within 5 % of its v2 → after change | held 3/4; **failed on t_ii_b** |
| T9-F | rows-off penalty = after to 1e-9 | **held** (≤ 3.5e-11 on all three) |
| T9-S | t_ii_b: sum rows carry ≥ 50 % of the row-family effect | **held**: 0.879 |

The T9-R outcomes per quantity:

| test, quantity | v2 | after | rows off | rows off − v2, as a fraction of the change | sum rows off only |
|---|---|---|---|---|---|
| t_ii_b, pre-charge rise (MWh) | 0.1283040 | 0.1280653 (−2.386e-4) | 0.1281456 | **−1.583e-4 (66 %)** | 0.1281360 |
| t_iii_indep, window margin min (MWh) | 0.00721786 | 0.00718373 (−3.41e-5) | 0.00721786 | −1.6e-11 (5e-7) | — |
| t_iv_peak, within-cohort L max (MWh) | 1.03351e-4 | 1.01418e-4 (−1.93e-6) | 1.03356e-4 | +4.2e-9 (0.2 %) | — |
| t_iv_peak, inter-cohort transfer (MWh) | 0.03995985 | 0.03996151 (+1.66e-6) | 0.03995987 | +1.9e-8 (1.2 %) | — |

**Mechanisms, with numbers:**
- **t_iii_indep window margin (−3.4e-5 MWh): explained.**
  - The request profile is fixed in every period, so only the split of a fixed aggregate between the two cohorts is
    free.
  - Both new row families are far from active: the complementarity body is ≤ 3.9e-16 against 1e-4, and the sum-row
    margin is ≥ 0.140 MW.
  - Their barrier terms alone move IPOPT's terminal split. With them off, the margin returns to v2 within 1.6e-11, and
    the SoC trajectories return within 3.0e-9 MWh (after: 4.5e-4).
  - The closure-slack removal has no measurable effect here.
- **t_iv_peak leak (−1.9e-6) and transfer (+1.7e-6 MWh): explained** by the rows. With them off, both return to v2
  within 0.2 % and 1.2 % of the change.
  - At the peak hour the 2025 cohort sits at its power box. Without the rows it delivers p_dch 0.249993 MW with p_ch
    2.68e-6, a sum-row margin of 4.3e-6 MW.
  - With the rows the margin is 1.05e-5 MW, and 4.5e-6 MW of peak delivery moves from the 2025 to the 2030 cohort
    (net −0.2499858 vs −0.2499903 MW).
  - The 2030 cohort's extra energy arrives through the inter-cohort transfer. The barrier-level leak shrinks.
  - The sum-versus-complementarity split was not measured for this test (no solve left).
  - The monitored scalars return, but the SoC trajectories do not fully: rows-off vs v2 is ≤ 4.5e-4 MWh, after vs v2
    is 1.9e-3.
- **t_ii_b pre-charge rise (−2.39e-4 MWh): partly explained.** The rise is free on a degenerate face: it must lie
  between the minimum needed, 0.0762 MWh, and the window headroom, 0.184 MWh. The measured attribution:

  | cause | effect on the rise (MWh) | share of the change |
  |---|---|---|
  | sum rows (after − sum-off) | −7.06e-5 | 30 % |
  | complementarity rows (sum-off − rows-off) | −9.7e-6 | 4 % |
  | closure-slack removal (rows-off − v2; the only remaining model difference, per the NL diff) | −1.58e-4 | **66 %** |

  - The sum row is the near-active one: 8.7e-9 MW from its bound at the discharge hour in the after solve.
  - The 66 % moves no objective value (3.5e-11). It is a quantified barrier-level drift on a flat face. The channel
    inside IPOPT is not identified: the objective scaling s_obj is 0.1 in both the v2 and the after record, and μ_final
    is 9.27e-12 (v2) vs 9.28e-12 (rows-off).

## Not confirmed

- **Whether Arm B is truly infeasible.**
  - Only IPOPT's local-infeasibility exits support it. No phase-1 or slack solve was permitted.
  - The loss bound (66.2 MW ≥ 63.9 MW) does not exclude feasibility.
  - Looked in: the Arm B IPOPT logs (`run/logs/*/B/`) and the zero-solve inventory (`zero_solve_diag_v1`).
- **Whether the D0 v2 3 × 3-pair instance avoided net exports for the same reason.** Not checked.
- **The IPOPT channel behind t_ii_b's 66 % closure-slack effect.** Candidates: the μ-update schedule, the
  slack-variable barrier terms, and the initial bound push. None was isolated, and no solve budget was left.
- **The t_iii_indep model difference.** It was not verified from an NL diff, because the T1 run wrote no NL for its
  plan. The rows-off result (5e-7 of the change remaining) makes any other difference immaterial for that quantity.
- **The t_iv_peak sum-versus-complementarity split.** Not measured.
- **The row-dual pricing of L in €.** Not done: the ESSO objective is σ-scaled (σ = 93,635,360), and a conversion
  rule was not part of the spec.
- **Contention.** The other Worker's calibration campaign (`p515_s56_step4_calib.py`, PID 14544, and its harness
  child) ran concurrently with both launches, as recorded in `other_p5_python_processes_at_launch`. It ran from the
  main checkout; these runs ran from the export. Wall times may include contention; no result depends on them.
