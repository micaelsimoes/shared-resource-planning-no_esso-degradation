# P5.16 D0 / T0.2-clarabel ablation — presolve of the empty-interior storage cones, max_iter 200 vs 5,000 (4 declared solves)

Worker, 2026-10-31. Planner ruling: option (b) + (a) as an ablation. Harness: `p516_d0_clarabel.py --ablation`
(sha256 `b0e991c7…57ca` at the run; the smoke mode without the flag is unchanged). Evidence:
`data/SRP1/Results/P516D0/t02_clarabel_ablation/` (new, untracked data root, 18 files, 7.1 MB). Hash inventory
`manifest_sha256.json` sha256 **`60bec4d95cd43d512dab4d8273420d9f3c5d8f9d434dcce7be0257601b989864`**.
Git HEAD at the run: `bad8cb84`. Child exit code **0**: 20 gate rows, all pass, 0 stops. Child wall 34.5 s.
The guard counted exactly **4** Clarabel solves (4 leaf entries + 4 solve calls, 0 blocked), 0 Gurobi optimize calls
and 0 Pyomo solves. The child launched 0 processes and made 0 write-opens in the repository outside the root.

## Blocked on Planner

1. **The presolve did not fix the stall: no solve converged (B1 and B2 FAILED). The Clarabel path is not ready for the
   108-solve run.** Neither returned value is a lower bound.
   - **Status.** At 200 iterations both groups ended `MaxIterations` again. At max_iter 5,000, x = 0 ended
     **`NumericalError` at iteration 1,436** and the unit ended **`InsufficientProgress` at iteration 1,235**.
     Neither run reached the cap.
   - **The presolve made no measurable difference at 200 iterations.** The primal residual at iteration 200 was:
     - x = 0: 9.69e-3 in the smoke, 9.73e-3 with the presolve;
     - unit: 1.37e-3 in both.
   - **Why it could not have mattered** (zero-solve, scratch). The cones with pinned heads constrained only isolated
     columns:
     - every `es_pnet` is already pinned by equality rows;
     - the 792 / 528 `es_qnet` columns that received a presolve row are free, appear in no linear row, carry zero cost
       and appear in no other quadratic row;
     - the multipliers on the presolve rows ended exactly 0.0 in all four solves.
     
     The empty-interior cones were therefore decoupled from the rest of the problem.
   - **More iterations reduce the primal residual slowly, then the solver gives up.**
     - x = 0: 9.73e-3 (it 200) → 1.91e-3 (it 1,000) → 1.74e-3 (it 1,436).
     - unit: 1.37e-3 (it 200) → 1.20e-4 (it 1,000) → 1.10e-4 (it 1,235).
     - Throughout, the dual residual is at 1e-12 to 1e-15 and Clarabel's gap at 1e-9 to 1e-11.
   - **The objective was still rising at termination**, by about 1 M€ per 100 iterations:
     - x = 0: 41.82 M€ (it 1,000) → 45.94 M€ (it 1,436);
     - unit: 53.62 M€ → 54.43 M€.
     - Q_cert⁺ is 64.00 / 63.96 M€.
   - **Where the returned points are infeasible** (it5000; zero-solve analysis of the saved vectors):
     - `d0_env_rewrite` equality rows of all three DSOs, up to 8.7e-3 pu (88 / 93 / 102 rows above 1e-6 at x = 0);
     - `node_balance_p` of all three DSOs, up to 4.9e-3 (about 730 rows each);
     - `d0_tr_cone` (transformer rotated cones), up to 2.7e-4;
     - the worst rows are the same family in every solve (`dso_9.d0_env_rewrite[47]`, `dso_9.d0_tr_cone[31,…]`).
   - **Options** (each needs new declared solves or a ruling):
     - **(d) Run T0.2 with Gurobi on the Studio**, as the R5 ruling already makes primary (TASKS.md: Gurobi works on
       the Studio, licence 2807206). Clarabel stays parked. No new decision is needed; this is the recommendation.
     - **(c) Objective scaling / settings ablation on Clarabel.** ‖q‖∞ = 4.6e7. The tiny generator-capability
       circles (radius 8e-4 to 9.6e-3 pu) end fully active with multipliers ‖z‖ up to 6.2e7. Changing scaling or
       settings is a solver-parameter change and needs authorisation.
     - **(e) One IPOPT smooth-form solve of one group** (already in `p516_d0_t02.py`). It tests whether the relaxation
       has a strictly feasible interior at all. The trace (objective rising monotonically while the primal residual
       stalls, κ/τ → 4e-4, no infeasibility certificate) is consistent with a feasible set that is thin or empty in
       the `d0_env_rewrite` / transformer-cone block. **Hypothesis, unverified.**
   - **Recommendation:** (d). Clarabel cannot serve as the fallback for T0.2 in its current form. If a fallback is
     still wanted, run (e) before (c), because (e) decides whether any conic solver can be expected to converge on
     these LPs.
2. **I omitted rows for cone components that the LP already pins (see Changed).** This departs from the literal
   instruction "p = 0, q = 0".
   - The task's form would have added rows for 936 (x = 0) / 624 (unit) components that the LP's own equality rows
     already pin. Every `es_pnet` is pinned, and 72 / 48 of the `es_qnet`.
   - Each such row is a linear combination of existing rows, so it adds a new rank deficiency to the zero cone. The
     set is identical either way.
   - Please confirm this reading of "same set, interior restored". The literal form would cost 2 more declared
     solves. I do not recommend it: the replaced cones are decoupled from the rest of the problem (item 1), so the
     literal form would differ only by the added dependent rows.
3. **Committing the evidence.** As instructed, I staged only the module and this report.
   - The small artefacts are: `manifest_sha256.json`, `predictions.json`, `ablation.json` (123 KB), `run.log`
     (305 KB), the 4 Clarabel logs (20–134 KB) and the provenance files.
   - The vectors (`vectors/*.npz`, 6.7 MB) are hash-recorded in the manifest.
   - Which of these to add is the Planner's call.

## Changed

- **`p516_d0_clarabel.py`**, extended (not committed):
  - **`--ablation` mode.** It runs the 4-solve plan, writes to `t02_clarabel_ablation/` (atomic, refuses if the root
    exists) and records `ablation.json`. Predictions B1–B5 and their operationalisation are written to
    `predictions.json` before any input is loaded.
  - **The smoke mode is bitwise reproducible.** Without the flag, a zero-solve rehearsal gives a `smoke.json`
    identical to the HEAD version's rehearsal, apart from the root path. The `.npz` vectors are equal too. In
    scratch, `build_conic(data)` gives the same A, b, row map, cone list and inventory as HEAD's on both LPs.
- **`build_conic(data, presolve=True)`:**
  - **Classification first.** Every quadratic row is classified before any row is built (the first misfit is still
    a STOP).
  - **`pinned_zero_columns`.** A column is pinned when it is fixed at LB = UB = 0, or when it is the only unpinned
    nonzero column of an `=` row whose rhs is exactly 0. The second rule is iterated to a fixed point.
  - **Replacement.** Every Lorentz cone whose head is pinned is replaced by zero-cone rows sqrt(d_i)·x_i = 0, one for
    each component not already pinned. Its SOC is removed. No column is substituted and no other cone is touched.
  - **`presolve_report`.** It records the chain provenance of every pinned head, plus the empty-interior candidates it
    did not alter.
- **Gate (ii) extension for a replaced cone.** The identity checked is ‖s̄‖² + Σ_pinned d·x_i² − e·t² = q(x) − rhs,
  to 8 eps. Linear and bound rows and the remaining cones are checked as before.
- **`gurobi_convention_duals`.** The presolve-row multipliers have no image in the Gurobi convention, because the
  replaced row's gradient vanishes at its apex. The strict residual (the definition used in the smoke) leaves them
  unmapped. Two further readings are reported: a *folded* one (RC[x_i] += sqrt(d_i)·z_i) and one restricted to the
  *untouched* columns.
- **New per-solve readings:** `returned_point_violation` (the max LP-row violation by class, with the worst row and
  the top families) and `small_circle_activity` (r < 1e-4).
- **Clarabel settings:** every value at its 0.11.1 default except `verbose = True` and `max_iter` (200 or 5,000; 200 is
  the default, set explicitly). qdldl, 1 thread, equilibration on.
- **Rehearsals:** zero solves, under `/tmp`. There were two in ablation mode (the second after the row-omission
  change) and one in smoke mode. All gates passed.
- **Leaf-guard negative test** (scratch): a `solve` call from outside the leaf raised `ClarabelSolveBlocked`, and
  `verify(0)` reported it.
- **Staged by filename:** `p516_d0_clarabel.py`, `P5_16_D0_T02_CLARABEL_REPORT.md`. Nothing committed.
- **Not touched:** production files, `p516_d0_t02.py`, `P516CC/`, `P516TPERF/` and TASKS.md.

## Found

**Instances.** Same as the smoke: W86 tight-tail cells, variant C (calendar-only), group 2025 Spring.
- x0: candidate_key `8435c718…`, cycle 132.
- unit_n7: candidate_key `db77e154…`, cycle 112.
- The LP hashes come from the T0.1c manifest (18 inputs hash-checked; all match).

**Objective convention:** gross operational cost per group, settlement excluded and salvage not subtracted. Q_cert⁺ is
the gross with the detector slacks at max(slack, 0) (§12.5), read from T0.1c `objective.json`.

**Presolve, zero-solve:**

| | x0 | unit_n7 |
|---|---|---|
| columns pinned | 1,434 | 1,104 |
| pinned heads (`es_s_rated`, via `rated_s_capacity` ← `rated_s_capacity_unit`) | 9 of 9 | 6 of 9 (`esso_7` kept) |
| Lorentz cones replaced / kept | 864 / 0 | 576 / 288 |
| zero rows added / components already pinned (no row) / replaced cones with no row | 792 / 936 / 72 | 528 / 624 / 48 |
| conic rows: zero / nonneg / SOC cones | 8,658 / 47,466 / 3,069 (smoke 7,866 / 47,466 / 3,933) | 8,690 / 49,109 / 3,405 (smoke 8,162 / 49,109 / 3,981) |
| not altered: rotated cones with a pinned factor; circles with r = 0 | 0; 0 | 0; 0 |
| not altered: small circles r < 1e-4 | 25 (`sg_capability`: 6 TSO + 19 DSO) | 73 (the same 25 + 48 `sess_converter_capability`, r = 6.25e-6) |
| not altered: rhs-0 `=` rows already implied by other rows in the LP | 216 | 144 |

- **Zero-solve gates on the presolved build** (x0 / unit; threshold in brackets):
  - (i) Clarabel vs LP objective at the certified point: 0.0 / 0.0 (1e-12);
  - (i) vs the T0.1c record: 1.52e-13 / 1.52e-13 (1e-12);
  - (ii) identity over-tolerance, linear and bound / cone / replaced cone: 0 / 0 / 0 on both groups (0);
  - (ii) class over-threshold: 0 / 0 (0);
  - inventory consistent: yes / yes.
- **At the certified point:** the pinned heads are ≤ 1.4e-40, the replaced rows' slack is 0.0, and the LP violation of
  the replaced rows is ≤ 6.2e-49.

**Per solve** (all with the presolve on):

| solve | status | it | r_prim | r_dual | gap abs (€) / rel | Q_LB (€) | Q_cert⁺ − Q_LB (k€, rel) | max LP-row violation (worst row) | ‖q + A'z‖∞ | stationarity, Gurobi convention (rel ‖c‖∞) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x0 it200 | MaxIterations | 200 | 9.73e-3 | 5.3e-12 | 0.147 / 8.8e-9 | 16,801,117.06 | 47,195.27 (0.737) | 2.85e-2 (`dso_9.d0_env_rewrite[46]`) | 1.0e-2 | 1.2e4 | 2.02 |
| x0 it5000 | NumericalError | 1,436 | 1.74e-3 | 1.9e-14 | 1.7e-3 / 3.7e-11 | 45,936,105.17 | 18,060.28 (0.282) | 8.68e-3 (`dso_9.d0_env_rewrite[47]`) | 2.6e-4 | 3.0e6 | 14.39 |
| unit it200 | MaxIterations | 200 | 1.37e-3 | 3.0e-13 | 0.684 / 1.5e-8 | 46,748,435.36 | 17,207.38 (0.269) | 5.05e-2 (`dso_9.d0_env_rewrite[37]`) | 3.5e-4 | 0.176 | 2.12 |
| unit it5000 | InsufficientProgress | 1,235 | 1.10e-4 | 4.4e-13 | 4.3e-4 / 7.9e-12 | 54,434,551.16 | 9,521.27 (0.149) | 7.91e-3 (`dso_9.d0_env_rewrite[47]`) | 6.0e-3 | 0.106 | 12.32 |

- **Reading the per-solve table:**
  - The duality gap recomputed as q·x − (−b·z) agrees with Clarabel's to within 3e-5 €.
  - Dual-cone violation is 0 on every solve.
  - Containment Q_LB − Q_cert⁺ − gap is negative on all four solves (−47.2 M€ … −9.5 M€). It is vacuous at
    unconverged points, so no stop rule applied.
  - The presolve-row multipliers ended at exactly 0 on all four solves, as do their slacks. Their columns are isolated,
    as shown above. The strict, folded and untouched readings of the stationarity residual are therefore identical,
    and the large values come from elsewhere: the argmax is
    `dso_5.pg[4,0,0,5]` / `dso_7.pg[3,0,0,19]`, the generator columns of the tiny `sg_capability` circles, where
    QCPi = −z₀/(2s₀) divides by a radius of about 1e-3.
  - The small circles end active (share of radius used ≥ 0.999 on 23 of 25 at x0 it5000) with ‖z‖ up to 6.2e7.
- **Determinism:** the first 200 iteration lines of the it200 and it5000 logs are identical per group (same md5).
- **T0.2 `lagrangian_bound`:** g_tight = −∞ on every solve (5,711–6,228 unbounded columns with a nonzero gradient).
  This is expected at unconverged points and is not a measurement.

**Predictions** (recorded first, in `predictions.json`):

| | prediction | outcome |
|---|---|---|
| B1 | with the presolve, both groups converge (Solved) within 200 iterations | **FAILED.** MaxIterations on both; residuals at it 200 unchanged from the smoke (9.73e-3 / 1.37e-3) |
| B2 | if B1 fails, 5,000 iterations suffice | **FAILED.** NumericalError (x0, it 1,436) and InsufficientProgress (unit, it 1,235), primal residual 1.74e-3 / 1.10e-4 |
| B3 | the converged x = 0 Q_LB is within 1e-2 relative of Q_cert⁺, containment holds | **INDETERMINATE:** no converged x = 0 solve |
| B4 | P7 Q_LB(unit) ≤ Q_LB(x0) between converged values | **INDETERMINATE:** no converged pair; no P7 gate was run. The readings between iterates (unit − x0: +29.95 M€ at it200, +8.50 M€ at it5000) are not bounds |
| B5 | the Gurobi-convention stationarity residual ≤ 1e-6 relative at a converged point | **INDETERMINATE:** no converged point (unconverged values 0.106 to 3.0e6 relative) |

**The Clarabel path's readiness for the 108-solve run: not ready.**
- **Measured cost:** 10.1 ms per iteration (qdldl, 1 thread), read + build 0.37 s per group, peak RSS 615 MB.
- **Run as configured here (max_iter 5,000):** each solve stops at 1,200–1,450 iterations without a bound, after about
  13.4 s. 108 × 13.36 s ≈ 24.0 min, plus 72 × 0.37 s ≈ 27 s of reads and builds.
- **That time buys no lower bound.** No projection for converged solves exists, since none converged.
- **Excluded:** T0.2 post-processing and the 24 IPOPT cross-checks.

**Unexpected findings.**
- **The LP itself has 216 (x = 0) / 144 (unit) rhs-0 equality rows that are implied by other rows:**
  `d0_cpl_ess_p_dso_esso`, `d0_cpl_ess_q_dso_esso` and `energy_storage_operation_agg`, 72 / 48 each. Every column of
  each such row is pinned by other rows, so the zero cone of the conic form is already rank-deficient. Not altered;
  it is a second candidate cause, untested.
- **Some coupling rows carry only the ESSO side.** At the non-invested nodes, `d0_cpl_ess_p_tso_esso` reads
  −0.01·es_pnet = 0 with no TSO column, and that row pins `es_pnet`.
  - x0: all 72 rows are of this kind.
  - unit: 48 rows are of this kind; the 24 rows at the invested node also carry `shared_es_pnet`.
  - The same holds for `_q_`.
  - Checked in scratch.
- **The certified point's `flex_p_down` sits on its upper bound** in the worst `d0_env_rewrite` rows (x_cert 0.00830226
  against ub 0.00830225). The Clarabel iterate exceeds it by 2e-5.

## Not confirmed

- **The cause of the stall.**
  - Ruled out: the empty-interior storage cones (this ablation, and their columns are isolated).
  - Not tested:
    - the implied equality rows (above);
    - the scaling of ‖q‖∞ = 4.6e7 against the tiny `sg_capability` circles;
    - a thin or empty interior in the `d0_env_rewrite` / transformer-cone block.
  - **Where I looked:** the 4 Clarabel logs, `ablation.json`, and a zero-solve per-family residual analysis of the it5000
    x = 0 vector. The analysis ran in scratch under `/tmp` and is not part of the evidence.
- **Whether the literal "p = 0, q = 0" form behaves differently.** Not run (Blocked item 2).
- **Whether the relaxation is feasible at all with exact bounds.** Not run. One IPOPT or Gurobi solve would decide it.
- **The Gurobi-convention dual mapping** is still unvalidated at a converged point. None exists yet.

---

# P5.16 D0 / T0.2-clarabel — the open-source conic path, smoke on two groups (2 declared solves)

Worker, 2026-10-31. Spec: `P5_16_D0_SPEC_v2.md` §4 G-D0-4/5/6, §5, §12.5, §13–§14 (ruling 5: Clarabel fallback).
Harness / module: `p516_d0_clarabel.py` (sha256 `d981bd36…c743` at the run). Evidence:
`data/SRP1/Results/P516D0/t02_clarabel_smoke/` (new, untracked data root, 11 files, 3.6 MB). Hash inventory
`manifest_sha256.json` sha256 **`13f146c73ac03959f2d07fdaf16cf2bfd9b64fbe888edeef99f45039bed66146`**.
Git HEAD at the run: `f6c23977`. Child exit code **2**: 17 gate rows, 1 failed (P7), 0 stops. Child wall 8.2 s.
The guard counted exactly 2 Clarabel solves, 0 Gurobi optimize calls and 0 Pyomo solves. The child launched 0
processes and made 0 write-opens in the repository outside the root.

## Blocked on Planner

1. **Clarabel did not converge on either group (prediction K1 FAILED).** Neither returned value is a lower bound.
   - **Status.** Both solves ended `MaxIterations` at Clarabel's default `max_iter = 200`.
   - **Dual side: almost feasible.** `res_dual` was 2.8e-12 (x = 0) and 1.3e-13 (unit). The unscaled residual
     ‖q + A'z‖∞ was 6.5e-3 and 1.6e-4 €/pu against ‖q‖∞ = 4.6e7.
   - **Primal side: still infeasible.** `res_primal` was 9.7e-3 (x = 0) and 1.4e-3 (unit). The returned x violates
     the LP rows by up to:
     - `d0_env_rewrite`: 2.8e-2 / 5.1e-2 pu;
     - DSO `node_balance_p`: 1.6e-2 / 3.0e-2 pu;
     - unit only, the shared-ESS SoC rows: up to 2.0e-3 pu.
   - **The primal residual is stalling, not slowly converging.** At x = 0 it went from 1.03e-2 at iteration 87 to
     9.69e-3 at iteration 200 (−6 %). At the unit it went from 2.26e-3 to 1.37e-3.
   - **The objective was still rising** when the run stopped: x = 0 went from 1.36e7 at iteration 147 to 1.71e7 at
     iteration 200.
   - **Clarabel's gap measure is not a convergence indicator here.** Its relative gap was 2.1e-8 and 1.7e-8 while the
     primal was 1e-2 infeasible.
   - **No infeasibility certificate.** κ/τ stayed at about 1 (1.15 / 0.85); `res_primal_inf` was 0.048 / 0.093;
     `res_dual_inf` was 0.70.
   - **Consequences:**
     - The containment gate (iii) "passes" vacuously.
     - The P7 gate (iv) "fails" as an artefact of the unconverged solves (Q_LB(unit) 48.53 M€ > Q_LB(x0) 17.11 M€).
     - K3 FAILED as worded (73 % gap at x = 0), but it measures nothing about the relaxation.
   - **No relaxation gap has been measured.** Options (each needs new declared solves):
     - **(a) Raise `max_iter`.** Re-smoke the same 2 solves with `max_iter` raised (e.g. 5,000) and every other
       setting at its default. This tests "slow" against "stalled". It costs about 1 min per solve at the measured
       11 ms per iteration. The stall rate above makes convergence unlikely.
     - **(b) Formulation-equivalent presolve before the conic build.** The suspect: an ESSO Lorentz cone whose head
       `es_s_rated` is pinned to 0 by an equality chain is {‖(p, q)‖ ≤ 0}, a cone with empty interior. The chain:
       - every zero-investment `es_s_rated_per_unit` is pinned to 0 by a single-column equality row (18 columns at
         x = 0, 15 at the unit);
       - `rated_s_capacity`: `es_s_rated = Σ es_s_rated_per_unit` then pins the head.
       
       There are 864 such cones at x = 0 and 576 at the unit. Presolve would propagate the pinning and replace each
       such cone by zero-cone rows p = q = 0. That is the same set, and it restores a strictly feasible interior.
       Fixed columns would be substituted. Gates (i)/(ii) apply unchanged. Then re-smoke with 2 solves.
       - Hypothesis, unverified: x = 0 has 864 such cones and stalls at 1e-2; the unit has 576 and stalls at 1.4e-3.
     - **(c) Objective scaling and/or settings.** Scale the objective (q / 1e6, with the value scaled back) and/or
       change Clarabel settings such as `equilibrate_max_iter` or `direct_solve_method = faer`. These are
       solver-parameter changes.
     - **(d) IPOPT as the primary path.** Make the IPOPT smooth form (already in `p516_d0_t02.py`) the primary path
       for the group solves. This changes the R5 ruling, which is for the expert and author.
   - **Recommendation:** (b) with (a) in one re-smoke of 2 solves. Both are formulation-preserving and cost seconds.
     If attribution matters, run a 4-solve ablation on x = 0 instead: (a) alone, (b) alone, (a) + (b), plus the
     unit with (a) + (b). If it still stalls, go to (d).
2. **Committing the evidence.** As instructed, I staged only the module and this report.
   - The small artefacts are: `manifest_sha256.json`, `predictions.json`, `smoke.json` (53 KB), `run.log` (46 KB),
     the 2 Clarabel logs (20 KB each) and the provenance files.
   - The vectors (`vectors/*.npz`, 3.4 MB) are hash-recorded in the manifest.
   - Which of these to add is the Planner's call.

## Changed

- **Packages installed** into `/opt/anaconda3/envs/opf_env_py311` with `python -m pip install clarabel`:
  - **clarabel 0.11.1** (Apache-2.0; wheel built 2025-06-11, rustc 1.87.0, aarch64-apple-darwin);
  - its dependencies **cffi 2.1.1** and **pycparser 3.12**.
  - numpy and scipy were already satisfied; nothing was upgraded.
  - **cvxpy was not installed.** The conic build is direct (scipy CSC).
- **`p516_d0_clarabel.py`** (new). It imports and reuses, from `p516_d0_t02.py` (not modified):
  - `read_lp_data`, `certified_point`, `parse_pyomo_name` and `quadratic_value_and_grad`;
  - `GurobiLeafGuard` (blocking: permitted count 0) and its audit hook;
  - `lagrangian_bound`, `rank_and_cycles` and `_cycle_basis`.
  
  From `p516_d0_t01b.py` it uses `CELLS`, `sha256_file` and `_git`. New code:
  - `build_conic`: LP → Clarabel (P = 0, q, A, b, cones) with a row map.
    - **Zero cone:** `=` rows, and fixed columns as e_k x = LB.
    - **Nonnegative cone:** `<` rows; `>` rows negated; finite UB and LB as rows.
    - **One SOC per quadratic row**, three patterns only, otherwise a STOP naming the row:
      - **rotated:** Σd x² − k·u·v ≤ 0 with LB(u), LB(v) ≥ 0, mapped to (u+v, u−v, 2√(d/k)·x). These are the
        branch and transformer cones.
      - **Lorentz:** Σd x² − e·t² ≤ 0 with LB(t) ≥ 0, mapped to (√e·t, √d·x). These are the ESSO converter rows.
      - **circle:** Σd x² ≤ r, mapped to (√r, √d·x). These are generator capability, branch thermal limits and the
        network-copy converter circles.
    - In each pattern q(x) − rhs = α(‖s̄‖² − s₀²) exactly (α = k/4 for rotated, 1 otherwise); this identity is gated.
  - `conic_dual_checks`: ‖q + A'z‖∞, primal residual, dual-cone membership, complementarity, and both gaps (Clarabel's
    and the recomputed q·x versus −b·z).
  - `gurobi_convention_duals`: Pi, QCPi and RC for the T0.2 post-processing.
  - `ClarabelLeafGuard`: patches `clarabel.DefaultSolver.solve` on the type. Before the run, a zero-solve negative
    test showed that a solve call from outside the leaf raises before reaching Clarabel.
  - The one leaf `_leaf_solve`.
  - The run structure follows T0.2:
    - a parent/child pair, with an atomic root that refuses if it exists;
    - stdout and stderr merged into `run.log`;
    - predictions written before any input is loaded;
    - every input hash-checked against the T0.1b/T0.1c manifests (18 files);
    - JSON written through `gate_result_io`;
    - the solution vectors saved before any post-processing;
    - both groups built and gated (i)/(ii) before the first solve.
- **The gurobipy reader under an empty HOME.**
  - The child runs with HOME set to a fresh, empty temporary directory, and gurobipy then reads the LPs under its
    bundled restricted licence. This is recorded in the provenance.
  - **Why the preflight warms matplotlib's font cache:** production's `network.py`, imported on unpickling, imports
    pyplot. Under an empty HOME, matplotlib launches `fc-list`, which the child's audit hook correctly refused in
    rehearsal 1.
  - **The fix:** the parent's preflight subprocess now builds the font cache inside that HOME.
- **Clarabel settings:** every value at its 0.11.1 default except `verbose = True`. The defaults are recorded in
  `predictions.json` and the settings in force per solve in `smoke.json`. In particular:
  - `tol_gap_abs`, `tol_gap_rel` and `tol_feas` are 1e-8;
  - `max_iter` is 200;
  - equilibration is on;
  - the linear solver resolved to qdldl, 1 thread.
- **Rehearsals:** zero solves, under `/tmp`, three of them; the third passed all 17 gate rows.
- **Staged by filename:** `p516_d0_clarabel.py`, `P5_16_D0_T02_CLARABEL_REPORT.md`. Nothing committed.
- **Not touched:** production files, `p516_d0_t02.py`, `P516CC/` and the Gurobi licence.

## Found

**Instances.** W86 tight-tail cells, variant C (calendar-only), group 2025 Spring. Objective convention: gross
operational cost per group, settlement excluded and salvage not subtracted. Q_cert⁺ = gross with the detector slacks
at max(slack, 0), per §12.5.

| group | candidate_key | LP sha256 | columns | conic rows (zero / nonneg / SOC) | SOC patterns (rotated / Lorentz / circle) |
|---|---|---|---|---|---|
| x0 2025 Spring C | `8435c718…` (cycle 132) | from the T0.1c manifest | 26,292 | 69,723 (7,866 / 47,466 / 3,933 cones) | 2,592 / 864 / 477 |
| unit_n7 2025 Spring C | `db77e154…` (cycle 112) | from the T0.1c manifest | 27,310 | 71,806 (8,162 / 49,109 / 3,981 cones) | 2,592 / 864 / 525 |

- Every quadratic row of both LPs fitted one of the three patterns. No mapping stop fired.

**Predictions** (recorded first, in `predictions.json`):

| | prediction | outcome |
|---|---|---|
| K1 | Clarabel converges on both groups (status Solved) to gap ≤ 1e-8 relative | **FAILED.** Both ended `MaxIterations` (200). Gap 2.08e-8 / 1.72e-8 relative, primal residual 9.7e-3 / 1.4e-3. |
| K2 | containment Q_LB ≤ Q_cert⁺ + solver gap | **HELD as computed, but vacuous.** Q_LB − Q_cert⁺ − gap = −46.89 M€ / −15.43 M€ at an unconverged iterate. |
| K3 | the x = 0 Q_LB within 1e-2 relative of Q_cert⁺ | **FAILED as worded, indeterminate in substance.** The gap read is 46,888.57 k€ (0.733 relative) at x = 0 and 15,428.83 k€ (0.241) at the unit. Neither is a measurement of the relaxation gap. |

**Gates.**

| gate | x0 | unit | threshold |
|---|---|---|---|
| conic inventory consistent with the LP | yes | yes | — |
| (i) objective, Clarabel model vs LP data at the certified point (rel) | 0.0 | 0.0 | 1e-12 |
| (i) objective at the certified point vs the T0.1c record (rel) | 1.52e-13 | 1.52e-13 | 1e-12 |
| (ii) rows whose conic violation ≠ the LP-form violation beyond fp: linear and bound / cone | 0 / 0 (worst cone ratio 0.093) | 0 / 0 (0.094) | 0 |
| (ii) violated rows over their class threshold | 0 | 0 | 0 |
| (iii) containment | pass (vacuous) | pass (vacuous) | ≤ 0 |
| (iv) P7: Q_LB(unit) − Q_LB(x0) − tolerance | **FAIL: +31.42 M€** (artefact of non-convergence) | | ≤ 0 |
| G-D0-6 guard exact (Clarabel 2 + 2 events; Pyomo 0; Gurobi 0) | pass | | |
| write-opens in the repository outside the root | 0 | | 0 |

- **(ii) classes at the certified point**, as the maximum violation and its ratio to the class threshold:

  | class | x0 | unit |
  |---|---|---|
  | coupling `d0_cpl_*` (reported) | 1.92e-5 | 1.91e-5 |
  | equality, mapping delta | 1.9e-11 (0.009) | 5.4e-12 (0.003) |
  | inequality | 1.16e-8 (0.82) | 1.16e-8 (0.82) |
  | bound | 1.0e-8 (0.83) | 1.0e-8 (0.83) |
  | quadratic | 1.0e-8 (0.83) | 1.0e-8 (0.83) |
  | variant-C capacity shift (§14.1) | — | 2.34e-4 (0.47; worst row `tso.sess_soc_final`) |

- **(v) Duals** (x0 / unit):
  - ‖q + A'z‖∞ = 6.5e-3 / 1.6e-4, which is 1.4e-10 / 3.5e-12 of ‖q‖∞.
  - Dual-cone violation: 0 (SOC); min z on the nonnegative cone 0.071 / 0.053.
  - Duality gap: 0.355 / 0.833 € (Clarabel), and the same recomputed as q·x − (−b·z).
  - The Gurobi-convention mapping's stationarity residual is 5.4e11 / 7.6e6. The mapping assumes cone complementarity
    z ∝ (s₀, −s̄), which an unconverged iterate does not satisfy, so this is expected.
  - T0.2's `lagrangian_bound` gives g_tight = −∞, from 5,712 / 6,217 unbounded columns with a nonzero gradient.
- **(vi) Time and memory per solve:**
  - setup 0.024 s;
  - solve 2.18 s (x0) and 2.31 s (unit), about 11 ms per iteration;
  - qdldl, nnz(L) 246,734 / 263,238;
  - peak RSS of the child 605 MB (`ru_maxrss`, which includes the unpickled blocks).
  - LP read + conic build: 0.57 s and 0.24 s.
- **Projected wall time for the 108 Gurobi-equivalent solves.** At 200 iterations per solve:
  108 × 2.27 s = 245 s, plus 72 reads and builds = 29 s, about 4.6 min solver-side. Because no solve converged, the
  projection for converged solves is open. At the measured 11.3 ms per iteration it scales as 108 × N × 11.3 ms:

  | N (iterations per solve) | projected time |
  |---|---|
  | 1,000 | about 20 min |
  | 5,000 | about 1.7 h |

  Excluded: T0.2 post-processing (Lagrangian margin, rank, capacity sensitivity, block unpickling) and the 24 IPOPT
  cross-checks.
- **(d)/(e) through T0.2's own functions** (`rank_and_cycles`) ran on the Clarabel iterates without error. Their
  values are meaningless at an unconverged point and are not reported as measurements.

**Unexpected findings.**
- **The task premise about T0.2's HOME handling does not match the file.**
  - The task said the restricted licence reads the LP files "with HOME pointed at an empty directory, as
    `p516_d0_t02.py` does".
  - `p516_d0_t02.py` contains no HOME handling (grep for `HOME` / `GRB_LICENSE` finds none).
  - Its parent preflight starts a Gurobi Env under the user's HOME, and would refuse with exit 6 on this machine.
  - In this module the HOME redirection is done in the parent, for the child.
- **An empty HOME breaks production unpickling under an audit hook.** It makes matplotlib rebuild its font cache via
  `fc-list` on the first import of pyplot (production `network.py` l.9). Any future harness that empties HOME must
  warm the cache outside the audited child, as done here.
- **numpy in opf_env_py311 carries two dist-info directories:** `numpy-2.4.1.dist-info` (2026-01-27) and
  `numpy-2.4.2.dist-info` (2026-03-06).
  - `numpy.__version__` reports 2.4.2, while pip's resolver reported 2.4.1 and `CLAUDE.local.md` records 2.4.1.
  - This predates this task; the install touched neither.
- **Clarabel's relative gap reached 2e-8 at 1e-2 primal infeasibility.** Any T0.2 use of Clarabel must gate on the
  status and on the primal and dual residuals, never on the gap alone.
- **No implied-equality row pairs and no near-degenerate boxes** (0 < ub − lb ≤ 1e-6) exist in either LP. This was a
  zero-solve scan of `<` / `>` rows with identical normalised coefficients. It removes one common cause of
  interior-point stalling.

## Not confirmed

- **Whether more iterations converge.** Only 200 iterations were run, and the declared count allowed no second
  attempt.
- **The degenerate-cone hypothesis** (Blocked item 1, option (b)).
  - What is verified: the 864 / 576 ESSO Lorentz heads are pinned to 0 through the equality chain. Each head
    `es_s_rated[k]` (lb 0) appears in one row `es_s_rated = Σ es_s_rated_per_unit`. Every zero-investment per-unit
    column is pinned to 0 by a single-column equality row: 18 at x = 0 and 15 at the unit, checked in scratch. The
    returned heads were 2–6e-8.
  - What is not verified: that they cause the stall. 25 circle cones also have r < 1e-4 (`sg_capability`), plus 48
    with r = 6.25e-6 at the unit (`sess_converter_capability`). They are another candidate, also untested.
- **The Gurobi-convention dual mapping** (QCPi = −z₀ / (2α s₀)) is derived, not validated. It can only be checked on
  a converged, complementary point. The conic dual residual ‖q + A'z‖∞ is computed directly and does not depend on it.
- **The RSS figure is a process maximum, not a per-solve increment.** The child's `ru_maxrss` was 605 MB both before
  and after each solve.
- **Clarabel 0.11.1 / qdldl behaviour on the T0.2 fixed-mode and freeze variants:** not exercised. They change bounds
  only.
- **Where I looked:** the two Clarabel iteration logs (`logs/*.clarabel.log`), `smoke.json`, the saved vectors (a
  zero-solve per-family residual analysis of the returned x, run in scratch under `/tmp` and not part of the
  evidence), and `p516_d0_t02.py` for the HOME handling.
