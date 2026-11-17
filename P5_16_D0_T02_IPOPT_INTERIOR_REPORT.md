# P5.16 D0 / T0.2-ipopt-interior — one IPOPT smooth-form solve of the x = 0, variant C, 2025 Spring group (1 declared solve)

Worker, 2026-10-31. Harness: `p516_d0_ipopt_interior.py` (new; sha256 `0438cfdc…be7e` at the run). Evidence:
`data/SRP1/Results/P516D0/t02_ipopt_interior/` (new, untracked data root, 9 files + manifest, 1.9 MB). Hash inventory
`manifest_sha256.json` sha256 **`ff2ea4889b634df8f269c7a843f28f8d2a35b7eb31c4ab9ae61349c2559e063c`**.
Git HEAD at the run: `120440ad`. Child exit code **0**: 8 gate rows, all pass, 0 stops. Child wall 51.6 s (of which the
rank inventory 47 s and the IPOPT solve 1.6 s). The guard counted exactly **1** IPOPT solve (Pyomo `OptSolver.solve` 1 +
`_execute_command` 1, 0 blocked); Gurobi optimize 0 and Clarabel solve 0, both verified. The child launched exactly one
process (the IPOPT executable, inside the leaf) and made 0 write-opens in the repository outside the root.

## Blocked on Planner

1. **I4 FAILED as worded, and the gap it measures is not resolved.** The IPOPT objective puts this group's gap at
   **5.844 k€ = 9.13e-5 relative**, below the predicted [1e-4, 1e-2]. That number is a single-group, single-solver,
   IPOPT-tolerance value, not a certified bound. Its resolution is about as large as the gap itself:
   - **IPOPT relaxes every bound and inequality by 1e-8.** This is its default `bound_relax_factor`, which the task's
     option set leaves unchanged. The returned point sits on that relaxation:
     - 333 / 333 generator circles active, violated by ≈1e-8 absolute (up to 1.5e-2 relative on r = 6.6e-7);
     - the d0 cones violated by ≈1e-8;
     - `voltage_magnitude_upper_cons` violated by up to 1.18e-8.
   - **The multipliers are huge**, so a 1e-8 relaxation is worth thousands of euros:
     - QCPi up to **3.3e9** on the small `sg_capability` circles;
     - QCPi up to **1.4e9** on the `dso.d0_cone` rows.
   - **First-order price of the relaxation:** Σ |multiplier| × violation at x* = **5,654 €**. The parts are:
     - quadratic rows 4,328 € (`d0_cone` 3,551, `sg_capability` 464, `d0_tr_cone` 313);
     - bounds 1,326 €;
     - linear rows 0.01 €.
   - **What follows.** The perturbation function is convex and nonincreasing, so v(0) ≥ v(u\*) + λ·u\*. The
     exact-feasibility optimum is therefore about **Q_LB + 5.65 k€**, and the group gap is about **0.19 k€ (3e-6)**.
   - **On the certified side**, the mapped point's own LP violations priced at the same multipliers come to **≤ 0.74 k€**
     (coupling 272 €, circles 464 €).
   - **Resolution.** The group gap lies in roughly **[−0.74, +0.19] k€** to first order: indeterminate below about 1 k€.
     What is determinate is that **gap ≤ 5.84 k€ (≤ 9.1e-5)**. Both readings sit below I4's lower edge.
   - These corrections are scratch readings; the formula is in Found §5.
   - **Decision needed for T0.2 (affects the 24 IPOPT cross-checks and the reading of every Q_LB).**
     - With multipliers of order 1e9, any solver's **primal feasibility tolerance**, not its primal-dual gap, sets
       Q_LB's resolution:
       - IPOPT: `bound_relax_factor` 1e-8;
       - Gurobi: `FeasibilityTol`, default 1e-6. Its effect on a barrier QCP solution is unmeasured.
     - G-D0-5's "primal-dual gap as Q_LB's resolution" does not capture this.
     - The G-D0-5 agreement threshold (1e-7 relative ≈ 6.4 €) is below this run's own relaxation effect (5.65 k€), so a
       Gurobi–IPOPT disagreement would be expected rather than diagnostic.
     - **Options:**
       - (a) Add Σ|λ|·violation per solve as a resolution column in T0.2. This is zero-solve post-processing.
       - (b) Run the IPOPT cross-checks with `bound_relax_factor 0`. This is a setting change, so it needs a ruling.
       - (c) Both.
     - **Recommendation:** (c).
2. **Clarabel attribution: the stall is not attributable to the LP's redundancy. Scaling or degeneracy is the
   remaining candidate; that is a hypothesis.**
   - **The relaxation is well-posed:** IPOPT reached `Optimal Solution Found` in 115 iterations with no restoration, in
     1.2 s. The equality rows hold to 1.3e-13 and the scaled NLP error is 8.4e-11. This holds with the caveat in Not
     confirmed: IPOPT's 1e-8 relaxation gives every row an interior.
   - **The redundancy is trivial.** The 216 dependent rows are all structural duplicates in components with one column:
     - 72 components of 3 rows × 1 column (`es_pnet` pinned three times);
     - 72 components of 2 rows × 1 column (`es_qnet` pinned twice).
   - **The rest of the equality block is well conditioned:** its smallest relative singular value is 1.2e-3.
   - **The candidate is the dual's scale:**
     - the optimal multipliers reach 3.3e9 (QCPi) and 4.9e7 (bound);
     - the objective entries span 1.5e5 – 4.6e7;
     - the circle radii go down to 6.6e-7 pu².
   - **Recommendation:** keep (d) Gurobi on the Studio as primary.
     - The equivalence-preserving export fix (drop the 216 duplicate rhs-0 rows, rescale the objective) is cheap.
     - Only its rescaling part is supported by this evidence.
     - If the Clarabel fallback is still wanted, test the fix in one 2-solve re-smoke on the same two groups (Clarabel
       path only; the Gurobi path reads the LPs as exported).
     - I do not recommend it ahead of the Studio run.
3. **Committing the evidence.** As instructed, I staged only the harness and this report; nothing is committed.
   - The small artefacts are: `manifest_sha256.json`, `predictions.json`, `interior.json` (688 KB), `run.log` (7 KB),
     the IPOPT log (16 KB) and the provenance files.
   - The vectors (`vectors/*.npz`, 1.2 MB) are hash-recorded in the manifest.
   - Which of these to add is the Planner's call.

## Changed

- **`p516_d0_ipopt_interior.py`** (new; staged, not committed).
  - **Structure.** It follows `p516_d0_clarabel.py`:
    - a parent/child pair with an atomic root that refuses if it exists;
    - stdout and stderr merged into `run.log`;
    - an empty HOME in the child for the gurobipy reader, with the matplotlib font cache warmed in the preflight;
    - predictions written before any input is loaded;
    - 12 inputs hash-checked against the T0.1b and T0.1c manifests;
    - JSON written through `gate_result_io`.
  - **The smooth form is `p516_d0_t02.build_ipopt_model` itself.** That function is a closure inside `child()`, so it
    cannot be imported. Its source text is extracted from `p516_d0_t02.py` with `ast`, checked against a pinned
    sha256 (`8a629019…`), and executed. `parse_ipopt_log` is reused the same way (`682f5b6d…`). `p516_d0_t02.py` is
    unchanged.
  - **Reused unchanged:**
    - from `p516_d0_t02`: `read_lp_data`, `certified_point`, `rank_and_cycles`, `_cycle_basis`, `lagrangian_bound`,
      `quadratic_value_and_grad`, `GurobiLeafGuard` and the audit hook;
    - from `p516_d0_clarabel`: `build_conic(presolve=True)` (for `presolve_report`; zero-solve), `pinned_zero_columns`,
      `classify_quadratic_row`, `lp_row_violations`, `returned_point_violation` and `ClarabelLeafGuard`.
  - **New code:**
    - `equality_rank_inventory`;
    - `column_inventory`;
    - `quadratic_inventory`;
    - `cone_slacks`: relative slack per row, absolute where the scale is ≤ 1e-12;
    - `parse_iterations`: the full iteration trace with restoration flags, final values at their last occurrence, and
      the timing block;
    - one leaf, `_leaf_solve`.
  - **One deviation from `p516_d0_t02.ipopt_solve`.** The leaf solves with `load_solutions=False`, then calls
    `m.solutions.load_from(results)`. A failed load is recorded instead of raising, and the start values are never read
    as x\*. At an Optimal exit the loaded values are the same.
  - **The raw vectors are saved before any post-processing**, and each post-processing block is guarded (a failure there
    is recorded as a STOP). No error occurred.
- **IPOPT options** (task): `tol 1e-9`, `max_iter 3000`, `mu_strategy adaptive`, `print_level 5`,
  `print_timing_statistics yes`, `linear_solver ma97`. Everything else is at the IPOPT 3.14.18 defaults, including
  `bound_relax_factor 1e-8`, `bound_push 1e-2` and `honor_original_bounds yes`. The command line is recorded in the log.
- **Rehearsals:** three, zero solves, under `/tmp`; all gates passed.
- **Not touched:** production files, `p516_d0_t02.py`, `p516_d0_clarabel.py`, the other D0 harnesses, `P516CC/`,
  `P516TPERF/` and TASKS.md.

## Found

**Instance.** Cell x0 (W86 tight-tail), candidate_key `8435c718…`, certification cycle 132, group 2025 Spring,
variant C (calendar-only). LP `t01c/lp/x0/2025_Spring_calendar.lp`, with its sha256 from the T0.1c manifest
`d2955128…`. 26,292 columns; 18,438 linear rows (7,476 `=`, 5,850 `<`, 5,112 `>`); 3,933 quadratic rows.

**Objective convention:** gross operational cost of the group, settlement excluded and salvage not subtracted.
Q_cert⁺ = gross with the detector slacks at max(slack, 0) (§12.5), from T0.1c `objective.json`:
- Q_cert⁺ = 63,996,389.50 €;
- gross = 63,995,186.36 €;
- Q_cert⁺ − gross = 1,203.15 €.

### 1. Redundancy inventory (zero solves)

**rhs-0 `=` rows whose columns are all pinned by other rows** (`presolve_report`): **216**.
- `d0_cpl_ess_p_dso_esso`: 72;
- `d0_cpl_ess_q_dso_esso`: 72;
- `esso.energy_storage_operation_agg`: 72.

1,434 columns are pinned to 0.

**Rank of the equality block.**
- **Method:** dense SVD (`scipy.linalg.svdvals`) per connected component of the row-column graph.
- **Numerical rank:** #{σ > σ_max,k · max(m_k, n_k) · eps} per component, which is numpy's `matrix_rank` default.
- **Dependent rows** are named by a pivoted QR of the component's transpose. This is one admissible choice, not a
  unique one.

| matrix | rows × cols | numerical rank | deficiency | of which structural (m_k > n_k) | smallest relative σ counted | relative σ below 1e-4 (any component) |
|---|---|---|---|---|---|---|
| A_eq | 7,476 × 26,292 | 7,260 | **216** | 216 | 1.25e-3 | 0 |
| A_eq without the 390 bound-fixed columns (= the conic zero cone less its 390 independent unit rows) | 7,476 × 25,902 | 7,260 | **216** | 216 | 1.24e-3 | 0 |

- **Where the deficiency sits.** It lies entirely in 144 components with one column:
  - 72 of 3 rows × 1 column: `d0_cpl_ess_p_tso_esso`, `d0_cpl_ess_p_dso_esso` and the linear
    `energy_storage_operation_agg`, all −0.01·`es_pnet` = 0 or `es_pnet` = 0;
  - 72 of 2 rows × 1 column: the two `_q_` coupling rows on `es_qnet`.
- **Dependent rows by family (QR choice):** `d0_cpl_ess_p_dso_esso` 72, `d0_cpl_ess_p_tso_esso` 72,
  `d0_cpl_ess_q_dso_esso` 72.
- **No numerical deficiency anywhere else.** The largest component (6,072 × 18,984) has full row rank, with smallest
  relative σ 1.25e-3.

**Columns.**
- 390 are bound-fixed (all at 0); 7,488 are free; 7,812 are unbounded.
- **0 columns appear in no row**, so 0 free columns appear in no row.
- **Zero-cost columns appearing only in quadratic rows: 864**, all free:
  - 792 `esso.es_qnet`, inside the pinned-head Lorentz cones;
  - 72 `dso.voltage_product_imag`, the transformer branch 31's WI, which appears in no linear row.
- **Objective entries:** 9,912 nonzero, |c| from 1.50e5 to 4.6e7.

### 2. Quadratic rows and the smooth form

**Inventory of the 3,933 quadratic rows:**

| pattern | family (count) |
|---|---|
| rotated | `dso.d0_cone` 2,304; `tso.d0_cone` 216; `dso.d0_tr_cone` 72 |
| Lorentz | `esso.energy_storage_operation_agg` 864, **all with a pinned head** (the set {x = 0}: no interior; the row's gradient vanishes at every feasible point) |
| circle | `dso.sg_capability` 216; `tso.sg_capability` 117; `dso.branch_flow_limit` 72; `dso.branch_flow_limit_ji` 72 |

- **Circle radii:** 0 circles have r = 0; **25 have r < 1e-4**:
  - TSO `sg_capability`: 6, r from 1.6e-6 to 4.7e-6;
  - DSO `sg_capability`: 19, r from 6.6e-7 to 9.3e-5.
- **Bilinear-factor lower bounds:**
  - `vmag_sqr` (Wii): min LB 0.7225 = 0.85² > 0;
  - **`dso.d0_tr_U`: min LB 0** (the transformer cone C² + D² ≤ U·Wjj).
- **Construction check.** The NL header (26,292 vars, 22,371 constraints, 7,476 equalities) equals the LP dimensions.
  IPOPT reports 25,902 variables (it removed the 390 bound-fixed columns) and 7,476 + 14,895 constraints. Both
  dimension gates pass.

### 3. Start point

- **The start** is the certified mapped point (T0.1c), clipped into the bounds: 13,147 columns moved, by at most
  1.0e-8.
- **LP objective at the start:** 63,996,433.97 €. The T0.1c identity at the unclipped point holds to 1.5e-13.
- **Its LP violation** is 1.92e-5 at most (`d0_cpl_v_W[5,7]`; 2,933 equality rows over 1e-8, all in the coupling and
  balance families). Inequalities and circles are violated by ≤ 1e-8.
- **IPOPT's iteration 0 is not the certified point.** It reports objective 1.78e8, inf_pr 6.94e-2 and inf_du 100.
  - IPOPT's default `bound_push` / `bound_frac` (1e-2) moved every variable that sat at a bound into the interior.
  - The detector slacks carry costs up to 4.6e7 per pu, which is why the objective jumped.
  - The certified start therefore did not survive IPOPT's initialisation.

### 4. The solve

| status | iterations | restoration | final inf_pr / inf_du (scaled) | complementarity | final lg(μ) | Q_LB (€) | wall |
|---|---|---|---|---|---|---|---|
| `Optimal Solution Found` (Pyomo: optimal / ok) | **115** | **0** | 1.34e-13 / 8.05e-11 | 8.41e-11 | −10.2 | **63,990,545.61** | 1.6 s (1.198 s in IPOPT + 0.091 s evaluations) |

- **Iteration path:**
  - inf_pr peaked at 3.63 (iteration 5) and fell to 4.6e-7 by iteration 24.
  - From iteration 86 to 110 it sat at **0.8–1.0e-8**, the relaxation width (1.3–1.4e-8 at iterations 80–85).
  - Hessian regularisation (inertia correction) occurred in two episodes, from the smooth form's indefinite rows:
    - iterations 1–7, lg(rg) −1.7 … −3.4;
    - iterations 63–71, lg(rg) from **+2.8** down to −1.0.
  - μ reached its floor (lg −11) at iterations 106–113, with inf_du oscillating up to 1.8e3 and five `H` line-search
    steps before the final iterations.
- **Unscaled final values:** dual infeasibility 3.7e-5, complementarity 3.9e-5, variable-bound violation 1.0e-8.
- **Multiplier sign convention:** the sign was read off stationarity (as in `p516_d0_t02`), +1. Stationarity residual
  3.7e-5 against 1.5e7 for the other sign. The recomputed reduced costs match IPOPT's to 3.7e-5.
- **Peak RSS:** the child's `ru_maxrss` is 2.44 GB, including the unpickled blocks; the IPOPT process 111 MB.

### 5. At the solution

- **LP violation at x\*:**
  - `=` rows: 1.3e-13;
  - inequality rows: 1.18e-8 (`dso_7.voltage_magnitude_upper_cons[1,0,0,12]`; 25 rows over 1e-8);
  - quadratic rows: 1.0e-8 (`dso_5.sg_capability[4,0,0,5]`; 11 rows over 1e-8);
  - bounds: 1.0e-8.

  All of these lie within IPOPT's 1e-8 relaxation.
- **Cone and circle slacks.** "Active" means relative slack ≤ 1e-6:

  | family | count | active | relative slack range at x\* |
  |---|---|---|---|
  | `tso.sg_capability` | 117 | **117** | −6.3e-3 … −3.4e-7 (on the relaxation) |
  | `dso.sg_capability` | 216 | **216** | −1.5e-2 … −2.6e-7 |
  | `dso.branch_flow_limit` / `_ji` | 72 / 72 | 4 / 0 | −1.0e-8 … 0.9994 / 9.7e-3 … 0.9994 |
  | `tso.d0_cone` | 216 | 144 | −1.3e-8 … **2.42e-2** |
  | `dso.d0_cone` | 2,304 | 2,232 | −8.6e-9 … 0.667 (the 72 inactive rows are the transformer branch 31, see Unexpected findings) |
  | `dso.d0_tr_cone` | 72 | 72 | −7.1e-9 … −6.8e-9 |
  | `esso` Lorentz (pinned heads) | 864 | — (scale 0) | absolute slack ≥ 3.2e-26; components ≤ 2e-25 |

  - **2,785 of the 3,069 rows with a scale are active.**
  - **Minimum relative slack:** −1.52e-2, at `dso_5.sg_capability[4,0,0,5]`, the r = 6.6e-7 circle violated by 1.0e-8.
  - **All 25 small circles (r < 1e-4) are active.** The largest |QCPi| is 2.1e9 (TSO) and 3.3e9 (DSO).
- **Interior (IPOPT barrier behaviour).** The run converged Optimal with μ → 6e-11 from IPOPT's strictly interior slack
  initialisation. That shows a strict interior of the **1e-8-relaxed** set, not of the exported set (Not confirmed).
- **Rank residuals** (`rank_and_cycles`):
  - **TSO:** max relative residual **2.42e-2** (branch 0; branch 3 has 2.41e-2, branch 6 1.82e-2); 1 cycle; cycle-angle
    residual 8.67e-3 rad.
  - **DSO 5 / 7 / 9:** the function reports max 0.667 on each. On the non-transformer branches every residual lies in
    [−8.6e-9, −6.8e-9]: rank-exact up to the 1e-8 relaxation, with 744 branch-periods per DSO.
  - **OLTC transformer-cone residual:** −6.8e-9 on all three DSOs.
- **Lagrangian margin:** g_tight = g_spec = −∞. 6,323 unbounded columns have a nonzero gradient (as ruled in §14.3).
  L_conv = 63,994,873.42.
- **Gap and containment** (THE FIRST GAP MEASUREMENT: single group, single solver, IPOPT tolerance; not a certified
  bound):
  - **Q_cert⁺ − Q_LB = 5,843.89 € = 5.844 k€, 9.13e-5 relative.** Against the gross it is 4,640.74 €.
  - **Containment Q_LB ≤ Q_cert⁺ + 64.0 € holds**, with margin −5,907.9 €.
  - For scale: the x0 cell's ADMM bar is 9,630 €, and that is a whole-cell, 12-group quantity.
- **Resolution of the gap** (scratch, zero-solve, from the saved vectors; not in the evidence root; Blocked item 1).
  - **Inputs:** `vectors/x0_2025_Spring_C.npz` (sha256 `83442210…`, in the manifest) and the hash-checked LP. The
    formula is exactly:
    ```
    lpv = p516_d0_clarabel.lp_row_violations(data, X)
    relax_price = sum(|Pi|*lpv['lin']) + sum(|QCPi|*max(lpv['quad_val'],0)) + sum(|RC|*max(lpv['lb'],lpv['ub']))
    cert_price  = same expression with lpv evaluated at x_start (the clipped certified point), multipliers at x*
    ```
  - **relax_price = 5,653.97 €.** Linear 0.014; quadratic 4,328.11 (`d0_cone` 3,551.20, `sg_capability` 463.70,
    `d0_tr_cone` 313.18, `branch_flow_limit` 0.03); bounds 1,325.84.
  - **cert_price = 735.66 €.** Linear 271.93 (`d0_cpl_pf_p` 142.03, `node_balance_p` 121.12); quadratic 463.73.
  - **Reading:** v(0) ≥ Q_LB + relax_price ⇒ gap ≲ 189.9 €. With the certified side, the gap lies in about
    [−0.74, +0.19] k€, which is first order only.
  - **TSO / DSO split of the objective difference** (start − x\*): TSO 5,085.05 €, DSO 803.31 €.

### Predictions (recorded first, in `predictions.json`)

| | prediction | outcome |
|---|---|---|
| I1 | the equality block has numerical rank deficiency ≥ 216 | **HELD.** Exactly 216, on A_eq and on the zero cone. All of it is structural (duplicated pins of 144 isolated columns); none is numerical (smallest counted relative σ 1.2e-3) |
| I2 | IPOPT converges (Optimal) within 500 iterations without restoration from the certified start | **HELD.** Optimal, 115 iterations, 0 restoration. IPOPT's bound push moved the start (iteration-0 inf_pr 6.9e-2) |
| I3 | containment holds | **HELD.** Q_LB − Q_cert⁺ = −5,843.89 € against a tolerance of +64.0 €. It still holds with the first-order relaxation correction (−189.9 €) |
| I4 | group gap in [1e-4, 1e-2] relative (H2 point estimate 1e-3) | **FAILED.** 9.13e-5 (5.844 k€). The first-order corrected value is ≈ 3e-6 (≈ 0.19 k€); unresolved below about 1 k€ (Blocked item 1) |
| I5 | TN rank residual max ≈ 2.45e-2 relative (within a factor 2) | **HELD.** 2.42e-2 (band [1.225e-2, 4.9e-2]). Cycle-angle residual 8.67e-3 rad (P9 point value 1.2e-2) |

**Gates:**
- LP objective at the certified point vs T0.1c: 1.5e-13 (threshold 1e-8);
- Wii LB > 0;
- rank bookkeeping;
- NL header = LP dimensions;
- IPOPT dimensions = LP dimensions less the fixed columns;
- IPOPT Optimal;
- G-D0-6 guard exact (1 + 1 / 0 / 0);
- repo writes outside the root = 0.

All 8 pass.

### Is the Clarabel stall a solver/conditioning matter or a well-posedness defect?

- **Within what one IPOPT solve can show, the stall is a solver/conditioning matter, not a well-posedness defect of the
  relaxation.**
  - The exported relaxation has a feasible point satisfying every equality row to 1e-13.
  - It has a finite optimum.
  - The KKT conditions hold at that optimum to 8e-11 (scaled).
  - A second-order interior-point method reached it in 115 iterations from a cold-pushed start.
- **The LP's redundancy does not explain the stall.** The 216 dependent rows are duplicated pins of isolated columns,
  and IPOPT/MA97 absorbed them without a restoration phase.
- **What distinguishes the problem numerically is its dual.**
  - Optimal multipliers reach 3.3e9 on circles of radius² 6.6e-7 … 9e-5, and 1.4e9 on the DSO cones.
  - The objective spans 1.5e5 – 4.6e7 per pu.
  - Clarabel's homogeneous embedding stalled with the objective rising and the primal residual stuck at about 1e-3.
    That is consistent with a solver that needs a very large dual norm to certify optimality. **This is a hypothesis.**
- **The export fix.**
  - The equivalence-preserving fix (remove the 216 dependent rhs-0 rows, rescale the objective) is cheap and safe for
    the Clarabel path.
  - Of its two parts, only the **rescaling** is supported by this evidence; the row removal addresses nothing measured
    here.
  - I recommend it only if the Clarabel fallback is still needed, as one 2-solve re-smoke. The Gurobi path on the Studio
    reads the LPs as exported.

## Unexpected findings

- **`p516_d0_t02.rank_and_cycles` reports the DSO rank residual as 0.667 because it includes the transformer branch's
  `d0_cone`.** Branch 31's WR and WI are inert:
  - WI appears in no linear row;
  - WR appears only in `voltage_product_real_nonnegative`;
  - neither carries cost.

  The transformer's physics is in `d0_tr_cone` (U, C, D), which is active to −6.8e-9. As written, T0.2's P1 (DSO
  rank-exact ≤ 1e-5) would read FAILED on every DSO for a definitional reason. Not fixed (outside scope). The likely
  remedy is to exclude transformer branches from the d0_cone residual, or report them separately.
- **IPOPT's default `bound_relax_factor 1e-8` matters at this problem's multiplier scale.** It is worth 5.65 k€ on one
  group, which bears directly on G-D0-5's IPOPT/Gurobi agreement threshold (Blocked item 1).
- **IPOPT's default bound push discards the certified start:** iteration-0 objective 1.78e8 against 6.40e7 at the
  certified point.
- **`dso.d0_tr_U` has LB 0.** The transformer cone's bilinear factor can reach 0, so the "Wii ≥ v_min² > 0" premise
  covers `vmag_sqr` only.
- **The audit hook's process-launch record** (`repr(args)[:300]`) begins to capture the environment dict of the IPOPT
  launch. It is truncated at 300 characters, and here it holds only the command line and the start of one path
  variable. Worth knowing for any harness that runs with credentials in the environment.
- **First-match log parsing** (`parse_ipopt_log`) is safe at print_level 5: each summary field occurs once. Checked
  with the last-occurrence parser.

## Not confirmed

- **A strict interior of the exported set.**
  - IPOPT works on the set with every bound and inequality relaxed by 1e-8·max(1, |bound|), which always has an interior.
    Its barrier convergence (μ → 6e-11) shows that interior, not the original one.
  - Analytically, the 864 pinned-head Lorentz rows have no interior: {p² + q² ≤ 0}. They are decoupled (T0.2-clarabel
    ablation), so Slater can hold at most for the remaining rows.
  - Whether it does was not tested. That would take a solve with `bound_relax_factor 0` or a strictly feasible witness.
- **The exact-feasibility gap.** The 0.19 k€ figure and the [−0.74, +0.19] k€ band are first-order scratch readings
  (formula above). They are not in the evidence root, and they assume exact multipliers and complementarity.
- **The cause of Clarabel's stall.** Only exclusions are established here: redundancy is excluded as sufficient, and
  well-posedness is not the cause. Scaling or dual-norm is the hypothesis, untested.
- **The unit cell and the other 11 groups:** not run (1 declared solve).
- **Gurobi's behaviour at this multiplier scale** (`FeasibilityTol` 1e-6 default): unmeasured.
- **Where I looked:**
  - the IPOPT log (`logs/x0_2025_Spring_C.ipopt.log`);
  - `interior.json` and the saved vectors;
  - three scratch zero-solve analyses under `/tmp/p516ipx/scripts` (`post.py`, `post2.py`, `post3.py`; not evidence):
    the bound-relaxation price, the certified-side price, the per-branch rank residuals and the transformer-branch
    column rows.

## T0.2 harness update per §15 (Worker, 2026-11-01; zero solves)

**Studio checkout target: `p516_d0_t02.py` sha256 `7fa39c4174be73ae32c72a1ccd3ed245d637d14f8aa44546a07cc0de577b4a1d`**
(predecessor `25fff859…` = the harness at `ff3153ff`). Staged by filename, not committed. Record schema v2 → **v3**
(`RECORD_SCHEMA`, `HARNESS_VERSION`, written to `predictions.json` and `t02.json`). Declared count unchanged
(`DECLARED` 72 + 12 + 24 + 24 = 132; 108 Gurobi + 24 IPOPT check unchanged); exit codes unchanged; no gate added,
removed or reworded. The terminated Worker's partial diff
(`P516D0/t02_harness_update/partial_s15_update_20261101_104527.diff`, sha256 `fd41eff4…`) passed `git apply --check`
and was applied as is; the rest was completed by hand.

**Changes.**
- **§15.1 resolution.** `resolution_price` (module level) prices Σ|λ|·max(0, violation) over linear rows, quadratic rows
  and bounds at the solver's own multipliers (Gurobi Pi / QCPi / RC; IPOPT sign·Pyomo dual and zL + zU). It runs after
  every solve (Gurobi in `gurobi_solve`, IPOPT in `ipopt_solve`; all variants, fixed-mode and freeze kinds).
  - `res_LB` is priced at the returned point. `res_cert` is priced at the certified mapped point clipped into that
    solve's bounds.
  - Each solve record carries `resolution` (per-part totals, max violation, number violated, top families) and
    `res_LB_eur` / `res_cert_eur`.
  - Every gap reading (R / A / C group LPs and the IPOPT cross-check's new `gap_abs`) carries
    `gap_interval_abs` / `_rel` = [gap − res_cert, gap + res_LB], as ruled.
  - Per cell: `sum_res_LB_eur`, `sum_res_cert_eur`, `sum_gap_interval_eur`, `class_point` (= `class`) and
    `class_upper_end`.
  - P6: `D_interval_eur` = [lo(unit) − hi(x0), hi(unit) − lo(x0)], with `D_conservative_end_eur` and
    `held_conservative_end`. Point verdicts are unchanged, and `verdict_{A,C,R}_conservative_end` is added.
  - The fixed-mode cost and the freeze bracket carry both solves' `res_LB` beside them. The G-D0-5 agreement record
    carries both resolutions beside it, without entering `held`.
- **§15.2 IPOPT.** The options are now: `tol 1e-9`, `max_iter 3000`, `print_level 5`, `bound_relax_factor 0`,
  `honor_original_bounds yes`, `bound_push 1e-10`, `bound_frac 1e-10`, `warm_start_init_point yes`.
  - The start is the certified mapped point clipped into bounds, unchanged from v2.
  - `iteration0` (objective, inf_pr, inf_du, lg μ) is parsed from the log. It is recorded in `start_kept_evidence`
    beside the start's own unscaled LP violation.
- **§15.3 Gurobi.** `FeasibilityTol 1e-9` and `OptimalityTol 1e-9` are added to the existing settings.
- **§15.4 `rank_and_cycles`.** The `d0_cone` of an OLTC branch is excluded from the rank residual. An OLTC branch here
  means `is_transformer`, in service, and with `d0_tr_U` columns in the block.
  - The excluded rows are reported under `transformer_d0_cone_excluded`. Transformers flagged but without OLTC columns
    are kept and listed.
  - P1 reads non-transformer branches and records the excluded branches. P8 still reads `d0_tr_cone` (`oltc`).
- **Resume.** A resumed group record whose `record_schema` ≠ v3 is refused (exit 5, as for any resume mismatch).

**Rehearsal** (`P516_D0_T02_REHEARSAL_DIR=/tmp/…`, all 24 groups, HOME = an empty scratch dir with the matplotlib font
cache pre-built; zero solves). The run used the final bytes `7fa39c41…` at HEAD `52f54a1b`. The /tmp roots were deleted
afterwards.
- **Child exit 0.** 362 gate rows, all pass, 0 stops. Guards counted 0 / 0 with no verify failures, and there were 0
  repository writes outside the root. Wall 65.9 s.
- **Bitwise baseline.** The HEAD harness (`25fff859…`) was rehearsed the same way and also gave 362 rows. All 362 rows
  are identical in gate name, scope, achieved value and pass. The 338 recorded in TASKS.md / `ff3153ff` is not
  reproduced by `25fff859` (Not confirmed below).
  - Every other difference between the two rehearsals is either an added v3 field or a field that changes from run to
    run (root path, timings, RSS).
  - The one exception is the intended change: the DSO rank records lose the 24 transformer branch-periods each.
- **New columns.** They are present and numeric on **132 / 132** planned solves (108 Gurobi + 24 IPOPT stand-ins). All
  96 gap readings carry an interval.
  - These are stand-in values: the certified point with zero row multipliers and RC = c. They exercise the path and
    are not measurements.
  - `iteration0` is `present: false` on the stand-ins, because no log exists.
- **Option dicts as printed.** These are the final settings for both solvers.
  - Gurobi:
    `{"QCPDual": 1, "NonConvex": 0, "BarQCPConvTol": 1e-09, "BarConvTol": 1e-10, "FeasibilityTol": 1e-09,
    "OptimalityTol": 1e-09, "Threads": 4, "LogToConsole": 0}`
  - IPOPT:
    `{"tol": 1e-09, "max_iter": 3000, "print_level": 5, "bound_relax_factor": 0.0, "honor_original_bounds": "yes",
    "bound_push": 1e-10, "bound_frac": 1e-10, "warm_start_init_point": "yes"}`
- **Transformer classification.** Branch 31 is classified as the OLTC transformer on DSO5 / DSO7 / DSO9 in every group,
  every variant and both cells (72 / 72 per DSO, 24 branch-periods each). The TSO has none excluded and none flagged.
- **Zero-solve cross-checks of the new code against this report's scratch figures.** These use the preserved vectors
  `vectors/x0_2025_Spring_C.npz` and the hash-checked LP.
  - `resolution_price` gives res_LB **5,653.97 €**: linear 0.014, quadratic 4,328.11, bounds 1,325.84.
  - It gives res_cert **735.66 €**: linear 271.93, quadratic 463.73.
  - Both equal Found §5 to the cent.
  - `parse_ipopt_iteration0` on this run's log returns objective 1.7820914e8, inf_pr 6.94e-2 and inf_du 100, as in
    Found §3.

**For the Planner (not acted on).**
- **The interval as ruled differs from Found §5's first-order bracket.** On this group the ruled
  [gap − res_cert, gap + res_LB] is **[5.11, 11.50] k€**. Found §5 gives v(0) ≥ Q_LB + res_LB, so the exact gap is
  ≲ gap − res_LB, and its lower end is −res_cert: **[−0.74, +0.19] k€**.
  - The ruled interval excludes Found §5's first-order estimate of the exact gap.
  - Implemented as ruled. Both prices are recorded per solve, so the other form is zero-solve post-processing.
- **`warm_start_init_point yes`** was not named verbatim in §15.2. It came from the partial patch, as the reading of
  "warm start from the certified mapped point", and I kept it.
  - The IPOPT 3.14 source falls back to the user-set `bound_push` / `bound_frac` when `warm_start_bound_push` /
    `warm_start_bound_frac` are unset (IpWarmStartIterateInitializer.cpp; slack pushes likewise). The certified start
    is therefore pushed by 1e-10.
  - The duals then start at z = 1, λ = 0 (AmplTNLP.cpp), with no least-squares λ estimate. Removing the option is a
    one-line change.
- **The IPOPT dimension gate will likely fail on the real run.** The gate is "IPOPT-reported dimensions == LP
  dimensions"; this run's IPOPT reported 25,902 variables against the LP's 26,292 (390 fixed columns removed).
  - It would fail on all 24 IPOPT solves, giving exit 2. It is not a stop.
  - Not changed, because changing it would change a gate's meaning.
- **The DSO freeze is not the §14.1 option (b).** `DSO_ESSO_frozen` freezes every `dso_*` / `esso_*` column, not the
  DSO W (+ U, C, D) and ESSO columns that §14.1 ruled. §14.1 states that this full freeze is infeasible on C by
  construction.
  - This is outside §15. Not changed.
- **The IPOPT options differ from the interior test's.** The interior test ran with `linear_solver ma97` and
  `mu_strategy adaptive`; the harness uses the IPOPT defaults for both. I2's convergence evidence therefore does not
  cover the harness's settings.

## §16 fixes (Worker, 2026-11-01; zero solves)

**Studio checkout target: `p516_d0_t02.py` sha256 `9fa3ef1e5a20c77284f7cec4d6511f2ad0277c2c07a5adc98273bcc403576b37`**
(predecessor `7fa39c41…`, the §15 form above). Staged by filename, not committed. Record schema stays **v3**;
`HARNESS_VERSION` and `RECORD_SCHEMA` cite §16. Declared count unchanged (132). No gate added or removed; one gate
redefined as §16.3 rules. The closures pinned by `p516_d0_ipopt_interior.py` (`build_ipopt_model`, `parse_ipopt_log`)
are unchanged; their hashes still match.

**Changes.**
- **§16.1, the firm interval.** `gap_interval` is replaced by `gap_firm_reading`. Every gap reading (R / A / C group LPs
  and the IPOPT cross-check) now carries these fields, each also as `_rel`:
  - `gap_firm_lower` = gap − res_LB − res_cert;
  - `gap_firm_upper` = gap + res_cert;
  - `gap_point_estimate` = gap − res_LB.
- **Per cell.**
  - The summed fields are `sum_gap_firm_lower_eur`, `sum_gap_firm_upper_eur` and `sum_gap_point_estimate_eur`, each also
    over Q_cert+.
  - The classes are `class` (raw gap, unchanged), `class_point_estimate` and `class_firm_upper` (conservative).
  - The readings carry `absolute_class_on_C_firm_upper`, and P3 carries `class_{R,C}_firm_upper`.
- **P6.** It is read conservatively at the firm upper ends. `D_firm_upper_eur` = Σ(gap + res_cert)(unit) −
  Σ(gap + res_cert)(x0), with `held_at_firm_upper` and `verdict_{A,C,R}_at_firm_upper`.
  - The interval-arithmetic range of the firm intervals is written beside it as `D_firm_interval_range_eur_informative`.
    It carries no verdict.
  - The raw-gap verdicts are unchanged.
  - The withdrawn §15 fields are no longer written: `gap_interval_*`, `class_point`, `class_upper_end`,
    `D_interval_eur` and `*_conservative_end`.
- **§16.2 / §16.5, IPOPT options.** The options are now `{tol 1e-9, max_iter 3000, print_level 5, bound_relax_factor 0,
  honor_original_bounds yes, bound_push 1e-10, bound_frac 1e-10, linear_solver ma97, mu_strategy adaptive}`.
  `warm_start_init_point` is removed.
- **§16.3, the dimension gate.** IPOPT-reported variables are compared with LP columns − columns with lb == ub, and eq +
  ineq with LP rows. `lp_dimensions` records `n_fixed_columns`, `vars_expected_by_ipopt` and, on a real run,
  `vars_reported_by_ipopt`.
  - On the interior group: 26,292 − 390 = 25,902, which is IPOPT's logged count; 7,476 + 14,895 = 22,371 rows.
  - Across the 24 groups the fixed count is 390 or 478.
- **§16.4, the freeze.** `DSO_ESSO_frozen` now freezes only the DSO `vmag_sqr` / `voltage_product_real` /
  `voltage_product_imag` columns, `d0_tr_U` / `d0_tr_C` / `d0_tr_D`, and every `esso_*` column. The columns are
  identified through the name map.
  - The key name is kept from v2. The pre-check record adds per-family frozen counts and a `[freeze]` console line.

**Rehearsal.** All 24 groups were rehearsed (`P516_D0_T02_REHEARSAL_DIR` under /tmp; HOME = scratch with a pre-built
matplotlib font cache; zero solves), on the final bytes `9fa3ef1e…`.
- **Run.** Child exit 0. 362 gate rows, all pass, 0 stops. Guards 0 / 0, and 0 repository writes outside the root. Wall
  63 s.
- **Baseline.** The `7fa39c41` bytes were rehearsed the same way, from a symlinked scratch checkout.
  - All 362 rows are identical in gate, scope, achieved value and pass.
  - The rehearsal cannot exercise the log-based dimension gate, because there is no IPOPT log; the NL-header gate fires
    instead and is unchanged.
- **Interval cross-check (zero-solve).** It was run on the preserved interior vectors (manifest-checked) and the
  hash-checked LP.
  - `resolution_price` gives res_LB 5,653.97 € and res_cert 735.66 €. The gap is 5,843.89 €.
  - `gap_firm_reading` gives the firm interval **[−545.74, +6,579.55] €** and the point estimate **189.92 €**.
  - Relative to Q_cert+: firm upper end 1.028e-4, point estimate 2.97e-6.

**Freeze pre-check, variant C, unit cell.** Max violation over rows whose columns are all fixed, at the certified point
clipped into bounds:

| freeze | columns frozen | rows checked | max violation | where |
|---|---|---|---|---|
| TN W (unchanged) | 648 | 840 | 8.7e-9 – 1.06e-8 | `tso.voltage_magnitude_upper_cons` |
| DSO (b) + ESSO (new) | 9,618 | 9,810 | **3.73e-7** (all 12 groups) | `esso_7.energy_storage_charging_discharging[3]` |
| full DSO + ESSO (`7fa39c41`) | 25,292 | — | 2.34e-4 – 4.86e-4 | `dso_7.sess_soc_final` |

**For the Planner (not acted on).**
- **The (b) pre-check is 3.73e-7, not ~1e-8.** A scratch zero-solve on 2025 Spring finds the excess only in the three
  inert `esso_7.energy_storage_charging_discharging` rows.
  - Cause: the certified ESSO `es_pch` / `es_pdch` sit at ≈ −1.0e-8. Clipping them to lb 0 (max clip 9.99e-9) shifts each
    row (es_avg plus 192 pch / pdch terms at coefficient 0.2445) by up to 3.73e-7. The row residual at the unclipped point is ~1e-15.
  - Leaving `es_avg_ch_dch_per_unit` / `es_D_per_unit` free makes the excess vanish: no row above 1e-8. These are the
    §14.2 inert chain. This variant was a diagnostic only; it is not in the harness.
- **The freezes may be infeasible for Gurobi.** Both frozen pre-checks exceed Gurobi `FeasibilityTol` 1e-9 (§15.3): TN W
  at ~1e-8 and (b) at 3.7e-7. The 24 freeze solves may therefore return INFEASIBLE from presolve.
  - The harness records that outcome; a freeze solve without an objective does not stop the run. The attribution bracket
    is then absent.
- **The interior group's conservative end classifies as H2.** It is 1.028e-4, and the harness rule is H1 ≤ 1e-4. §16.1's
  "borderline H1" reads as H2 here. It is a single group; the class is decided on Σ over 12 groups.
- **`linear_solver ma97` needs HSL on the Studio.** The preflight checks only that the IPOPT executable exists. Not
  verified on the Studio.

## §17 edits (Worker, 2026-11-01; zero solves)

**Studio checkout target: `p516_d0_t02.py` sha256 `5f5e500f03497537f9c22eb7fd235e8c4c3c249898aadd88924aa810b254992b`**
(predecessor `9fa3ef1e…`, the §16 form above). Staged by filename, not committed. Record schema stays **v3**;
`HARNESS_VERSION` and `RECORD_SCHEMA.spec_reference` cite §17 rulings 1, 2 and 5. Declared count unchanged (132). No
gate added, removed or redefined. Every §15 / §16 change is kept.

**Changes.**
- **§17.1, the DSO + ESSO freeze.** `DSO_ESSO_frozen` now leaves `es_avg_ch_dch_per_unit[*]` and `es_D_per_unit[*]`
  free (`ESSO_FREE_FAMILIES`, identified through the name map). Everything else in the ESSO is frozen as before.
  - The pre-check record adds `n_esso_columns_left_free_by_family`.
  - The harness raises if no such column is found, so the freeze cannot silently revert to the full-ESSO set.
- **§17.2, freeze feasibility tolerance.** `GUROBI_PARAMS_FREEZE` = `GUROBI_PARAMS` with `FeasibilityTol 1e-7`. It is
  used for the 24 freeze solves only (both freezes).
  - The 72 group LPs and the 12 fixed-mode solves keep `GUROBI_PARAMS` (1e-9).
  - Every Gurobi solve record carries the dict it ran with (`params`) and `params_set` (`standard` / `freeze`).
  - `predictions.json` records both dicts, and the console prints both.
- **§17.5, the ma97 preflight.** `_ipopt_ma97_check` runs `ipopt -v` and `ipopt --print-options`, which solve nothing.
  It parses the `linear_solver` entry's "Possible values", and the parent refuses with exit 6, writing nothing, unless
  ma97 is listed.
  - The banner, the `linear_solver` block, the listed solvers and the sha256 of the option listing are printed in the
    `[preflight]` line and written to `provenance_parent.json` (`preflight`) on a real run.
  - The rehearsal skips the preflight, as before.

**Rehearsal.** All 24 groups were rehearsed (`P516_D0_T02_REHEARSAL_DIR` under /tmp; HOME = scratch with a pre-built
matplotlib font cache; zero solves) on the final bytes `5f5e500f…`. The baseline was the `9fa3ef1e` bytes, rehearsed
the same way immediately before the edit.
- **Run.** Child exit 0. 362 gate rows, all pass, 0 stops. Guards counted 0 / 0 with no verify failures, there were 0
  process launches, and tracked status was unchanged.
- **Baseline.** All 362 rows are identical to the baseline in gate, scope, achieved value and pass.
- **Record diff against the baseline (recursive).** The only non-volatile differences are the intended ones:
  - the DSO + ESSO pre-check fields, on 12 / 12 unit groups;
  - `FeasibilityTol` 1e-9 → 1e-7 on the 24 freeze solves;
  - the new `params_set` field on all 108 Gurobi records;
  - `n_esso_columns_left_free_by_family`, the harness version / hash, the schema change list, and the `preflight` key
    (null in a rehearsal).
- **Per-solve settings in the rehearsal records.**

  | solves | `params_set` | `FeasibilityTol` | count |
  |---|---|---|---|
  | R / A / C group LPs | standard | 1e-9 | 72 |
  | fixed-mode | standard | 1e-9 | 12 |
  | TN W freeze | freeze | 1e-7 | 12 |
  | DSO + ESSO freeze | freeze | 1e-7 | 12 |

**Freeze pre-check, variant C, unit cell.** Max violation over rows whose columns are all fixed, at the certified point
clipped into bounds:

| freeze | columns frozen | rows checked | max violation | where |
|---|---|---|---|---|
| TN W (unchanged) | 648 | 840 | 8.68e-9 – 1.06e-8 | `tso.voltage_magnitude_upper_cons` |
| DSO (b) + ESSO, §16 (`9fa3ef1e`) | 9,618 | 9,810 | 3.73e-7 (all 12 groups) | `esso_7.energy_storage_charging_discharging[3]` |
| DSO (b) + ESSO, §17 (new) | 9,612 | 9,804 | **9.998e-9** (all 12 groups) | `esso_7.energy_storage_operation_agg[134]` |

- The columns left free are `esso_7.es_avg_ch_dch_per_unit` and `esso_7.es_D_per_unit`, 3 each, in every unit group.
  esso_5 and esso_9 have no columns of these families in the unit-cell LPs.
- Six rows leave the all-fixed set.
- The max clip into bounds is 9.994e-9; the worst row's violation is the size of that clip.
- **Prediction (§17.1, pre-check ≤ 2e-8): held** on 12 / 12 groups.
- **Prediction (worst row no longer a `energy_storage_charging_discharging` row): held.**

**Option dicts as printed.**
- Gurobi, for the group LPs and fixed-mode solves:
  `{"QCPDual": 1, "NonConvex": 0, "BarQCPConvTol": 1e-09, "BarConvTol": 1e-10, "FeasibilityTol": 1e-09,
  "OptimalityTol": 1e-09, "Threads": 4, "LogToConsole": 0}`
- Gurobi, for the freeze solves: the same dict with `"FeasibilityTol": 1e-07`.

**ma97 check, run on this machine (zero solves).**
- The banner is `Ipopt 3.14.18 (aarch64-apple-darwin24.5.0), ASL(20241111)`.
- The listed `linear_solver` values are ma27, ma57, ma77, ma86, ma97, pardiso, mumps and custom; ma97 is described as
  "use the Harwell routine HSL_MA97".
- The check gives ok = true. The option listing's sha256 is `f52b69af…`.
- **Negative control.** The same check pointed at `/usr/bin/true` gives ok = false. `parent()` then printed the refusal
  and returned **6** without creating its root (that root was under /tmp).
- The Studio's banner is not verified here.

The /tmp rehearsal roots and the earlier scratch (`/tmp/p516s16`, `/tmp/p516t02_*`) were deleted afterwards. None of
it is evidence.
