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
