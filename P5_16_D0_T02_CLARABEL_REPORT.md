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
