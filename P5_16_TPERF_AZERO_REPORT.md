# P5.16 T-perf A-zero: which objective term has gradient 1e5, and where an ADMM cycle's wall time goes

Worker report for the Planner, 2026-10-31. Two zero-solve follow-ups to `P5_16_TPERF_REPORT.md`. No solve was
run, no ADMM cycle was run, and no production file was changed.

**Objective convention:** no cost is computed or compared. Q1 reports gradients of the active solver objective,
`p58_rescaled_admm_objective`, at the certified cycle-112 point. Q2 reports times, counts and iteration numbers
only.

**Instance (both questions):**
- campaign `s53_w86_tail_recert` (spec `ddd6cd44`), cell `n7_4h_e1`;
- eval key `ca8927e75d628bd1…`, candidate_key `db77e1549af855bc…` (2025 cohort; node 7 0.25 MVA / 1.0 MWh; nodes 5
  and 9 empty);
- certified at cycle 112;
- blocks: TSO 2025 Spring (`case9`) and DSO node 7 2025 Spring (`case33_2`).

`certified_models.pkl` matched sha256 `fe06ca48…efda04` against the declared value and against both the campaign
and child manifests before unpickling. It was hashed again at the end of the run and was unchanged. Every block
was cloned before any evaluation.

## Blocked on Planner

1. **The Q2 prediction's verdict depends on what counts as "solve" time. You need to rule on the definition.**
   The two definitions differ by whether Pyomo's NL write and result load count as solve time or as non-solve time.
   - **Solve = the IPOPT process only** (NL write and result load count as non-solve): the prediction **holds**.
     The proxy gives a non-solve share of 62–63 %. Two Studio measurements from earlier, different runs give 46 %
     and 59.5 %.
   - **Solve = the whole `network.run_smopf` call** (NL write and result load count as solve): the prediction is
     **indeterminate**. The proxy median is 41.3 % / 42.9 %, only 1.3–2.9 points above the 40 % bar. A Studio that
     is just 2–5 % slower per solve than this machine would move it below 40 %, and the machine factor has not been
     measured. The only instrumented Studio split on record, P5.15 Step 3.6 on 10 cycles (a different run and
     code era), gives about 14 %, which contradicts the prediction.

   Settling the run_smopf definition for this cell needs per-solve wall times on the Studio. Those require a run,
   which this task does not name. That is your decision, or the author's.
2. **Commit of evidence.** As instructed, I staged only `p516_tperf_azero.py` and this report, and committed
   nothing. CLAUDE.md requires the report to be committed with its evidence and hash inventory:
   - this stage's evidence is `data/SRP1/Results/P516TPERF/azero/`, which is untracked;
   - the stage also read T-perf inputs that are untracked: the two `i_warm` NL files and four IPOPT logs under
     `data/SRP1/Results/P516TPERF/local/`. They are hash-pinned by the committed T-perf manifest `fbfd6310…` and
     re-listed in this stage's manifest.

## Changed

- **New harness `p516_tperf_azero.py`** (sha256 `170505dd…`, the bytes that ran).
  - It installs a blocking `SolveProfileGuard(())` before any production import and checks `verify(0)` exactly.
  - It takes an exclusive flock on `data/SRP1/Results/P516TPERF/.p516_tperf_azero.lock` and scans for other running
    copies.
  - It refuses to write into an output directory that exists, or under the D0 root.
  - It tees stdout and stderr into `run.log`.
  - `--scratch DIR` writes the same outputs to a new directory, for testing.
- **Output** `data/SRP1/Results/P516TPERF/azero/` (new, write-once):

  | File | sha256 |
  |---|---|
  | `q1_gradient.json` | `e2e78639…` |
  | `q2_cycle_time.json` | `86cebc51…` |
  | `run.log` | `a73346b6…` |
  | `manifest_sha256.json` | `2e06de77…` |

  The manifest lists 24 files with status `completed`: the outputs, the harness, the 11 cell inputs, the 7 T-perf
  inputs plus the T-perf manifest, and the prior-art table. All 24 re-hashed equal after the run.
- **Run conditions.** The run was attached, with both streams captured, on interpreter
  `/opt/anaconda3/envs/opf_env_py311/bin/python` and Pyomo 6.9.5.
  - Machine: Mac17,3. git HEAD at run time: `cdf8804a`.
  - Production files and `definitions.py`: `git status` clean. `p516_tperf_profile.py` was not touched.
  - Concurrent work: the D0 T0.1b Worker process (`p516_d0_t01b.py`) was running. Nothing in this stage is
    timing-sensitive. Its only timings are the harness's own clone and differentiate seconds, which are reported
    but not used.
- **Guard:** `verify(0) = []`. Counts: 0 permitted solves, 0 permitted execs, 0 blocked solves, 0 blocked execs.
  The guard was still installed at the end.
- **Scratch test before the run:** `/tmp/azero_scratch/test1` used the same harness bytes and gave the same results.
  An exploration script, `/tmp/azero_scratch/explore.py`, ran with the guard armed (`verify(0) = []`). Both are
  outside the repository.

## Found

### Q1: the 1e5 gradient comes from slack penalties at 1e3 × baseMVA. The economic gradients are 3.3e2–1.6e4.

**Objective structure** (asserted, not assumed). The active objective is `p58_rescaled_admm_objective`. It has the
form M · [ (Σ_j c_j) / eff + Σ_k AL_k ], with:
- M = eff = `admm_objective_scale` = 203,555.13 (σ = 93,635,360 divided by block weight 460);
- the c_j are the 8 top-level terms of `objective_function_rule`.

So the c_j enter at unit weight, in € of the representative day, and the augmented-Lagrangian (AL) and proximal
terms enter multiplied by eff.

Each term was differentiated separately with Pyomo `differentiate` (reverse_numeric). The per-term gradients sum
back to the total gradient to within 3.1e-16 (TSO) and 2.0e-15 (DSO).

| Block | Free variables in the objective | Max \|∂f/∂x\| | Variables with \|∂f/∂x\| ≥ 1e3 |
|---|---|---|---|
| TSO | 890 | 1.0e5 (2 variables) | 444 |
| DSO | 3,498 | 1.0e5 (66 variables) | 3,234 |

**NL cross-check.** I parsed the linear part (G0 segment) of the NL file IPOPT read in T-perf case `i_warm`. Its max
is 1e5. Its counts match the Pyomo gradient exactly on both blocks: 2 / 66 entries at 1e5, 288 / 1,584 at 5e4,
and 444 / 3,234 at or above 1e3. That agrees with T-perf's replica (`max_abs_obj_gradient` 1e5).

**Variables with |∂f/∂x| ≥ 1e3, grouped by the term they belong to.**
- All slack gradients equal prob × constant × (baseMVA if the production penalty function multiplies by it) ×
  M/eff exactly, with 0 mismatches in every family.
- prob = 1 and baseMVA = 100.
- A slack value of −1e-8 means the slack sits at its lower bound 0, inside IPOPT's bound relaxation.

| Block | Term | Variable family | n | \|∂f/∂x\| | Kind | Value at the certified point |
|---|---|---|---|---|---|---|
| TSO | `total_ess_complementarity_penalties` | `slack_shared_es_soc_final_up` / `_down` | 1 + 1 | **1.0e5** | slack penalty, `PENALTY_SHARED_ESS_BALANCE` 1e3 × baseMVA | −1.0e-8 (at lb 0) |
| TSO | `total_slack_penalties` | `slack_v_sqr_up` / `_down` | 144 + 144 | 5.0e4 | slack penalty, `PENALTY_VOLTAGE_SQUARED` 5e4 (no baseMVA) | −1.0e-8 .. 0; none > 1e-6 |
| TSO | `total_gen_cost` | `pg` | 66 | 1.27e3–1.58e4 | economic cost: c_p × baseMVA | — |
| TSO | `interface_settlement` (weight 1) | `interface_delta_p` | 66 | 1.27e3–1.58e4 | economic transfer (excluded from Q by convention) | — |
| TSO | AL linear dual, ess | `expected_shared_ess_p` | 10 | 1.07e3–2.79e3 | AL term (× eff) | — |
| TSO | AL linear dual, pf | `expected_interface_pf_p` | 12 | 1.20e3–1.69e3 | AL term (× eff) | — |
| DSO | `total_slack_penalties` | `slack_flex_p_balance_up` / `_down` | 32 + 32 | **1.0e5** | slack penalty, `PENALTY_FLEXIBILITY` 1e3 × baseMVA | −1.0e-8 (at lb 0); none > 1e-6 |
| DSO | `total_ess_complementarity_penalties` | `slack_shared_es_soc_final_up` / `_down` | 1 + 1 | **1.0e5** | slack penalty, `PENALTY_SHARED_ESS_BALANCE` 1e3 × baseMVA | −1.0e-8 (at lb 0) |
| DSO | `total_slack_penalties` | `slack_v_sqr_up` / `_down` | 792 + 792 | 5.0e4 | slack penalty, `PENALTY_VOLTAGE_SQUARED` | −1.0e-8 .. 0; none > 1e-6 |
| DSO | `total_flex_cost` | `flex_p_down`, `flex_q_down` | 768 + 768 | 2.13e3–5.53e3 | economic cost: cost_flex × baseMVA | — |
| DSO | `interface_settlement` | `pg` (reference generator), `shared_es_pnet` | 22 + 22 | 1.27e3–1.58e4 | economic transfer | — |
| DSO | AL linear dual, pf | `expected_interface_pf_p` | 4 | 1.26e3–1.69e3 | AL term (× eff) | — |

Notes on the table:
- **Penalty families not present.** `PENALTY_NODE_BALANCE` (1e6), `PENALTY_CURRENT` and `PENALTY_ESS_BALANCE`
  appear in no free variable of either active objective. That means no node-balance, branch-flow or local-ESS
  day-balance slack family is present, and the harness reports no unmapped slack family.
- **The 1e5 maximum.** It is 1e3 × baseMVA: `PENALTY_SHARED_ESS_BALANCE` on both blocks, and `PENALTY_FLEXIBILITY`
  on the DSO. Every penalized slack is at its lower bound, so none of these penalties is active at the certified
  point.
- **Economic gradients are already above IPOPT's scaling threshold.** IPOPT scales the objective when the largest
  gradient exceeds 100. Leaving out the slack-dominated variables, max |∂f/∂x| would be 15,809, the 158 €/MWh
  price × baseMVA. That alone gives df = 6.3e-3. So the slack penalties lower df by a factor of 6.3, from 6.3e-3 to
  1e-3, not by 1000. (This is arithmetic only, not a solve.)
- **Unit checks.**
  - For `pg`, the `total_gen_cost` contribution divided by baseMVA equals `network.cost_energy_p[0][p]` for all 72
    variables (max relative difference 1.3e-16).
  - For `flex_*_down`, the `total_flex_cost` contribution divided by baseMVA equals `cost_flex[0][p]` for all 1,536
    variables (1.6e-16).
  - There are 24 periods, so |∂f/∂x| / baseMVA is a price in €/MWh.

**Gradient distribution.** Counts are per decade of |∂f/∂x| over the free objective variables.

| Block | 0 | <1e-3 | [1e-3, 1) | [1, 1e1) | [1e1, 1e2) | [1e2, 1e3) | [1e3, 1e4) | [1e4, 1e5) | [1e5, 1e6) |
|---|---|---|---|---|---|---|---|---|---|
| TSO | 192 | 41 | 21 | 47 | 111 | 34 | 76 | 366 | 2 |
| DSO | 144 | 46 | 4 | 13 | 28 | 29 | 1,558 | 1,610 | 66 |

**Dual-infeasibility tolerance this implies.**
- **df.** df = 100 / max |∇f| = 1e-3. This matches IPOPT's printed "Scaling parameter for objective function =
  1.000000e-03" on both blocks.
- **Unscaled residual is 1000 × the scaled one.** There is no x scaling. Unscaled multipliers are λ = λ̃·dc/df and
  z = z̃/df, so the scaled stationarity residual equals df × ‖∇L‖∞ unscaled. The T-perf logs confirm it: unscaled
  over scaled dual infeasibility is exactly 1000.0 in all four solves.
- **s_d = 1.** At termination the printed overall NLP error equals the scaled dual infeasibility, and the dual
  infeasibility dominates both the constraint violation and the complementarity. So s_d = 1 (inferred, on all
  four solves).
- **Effective unscaled stationarity tolerance.** τ_u = tol · s_d / df = 1e-5 / 1e-3 = **1e-2** € per pu per
  representative day, for every variable. As a price that is **1e-4 €/MWh**.
  - `dual_inf_tol` is not set, so IPOPT's default of 1 applies, which is looser and does not bind.
  - On the acceptable path (`acceptable_tol` 1e-4), τ_u is 0.1, or 1e-3 €/MWh.
- **Against the economic gradients**:
  - TSO: 144 economic variables, |∇f| from 326 to 15,809 (3.26–158 €/MWh); τ_u / |∇f| from 6.3e-7 to 3.1e-5.
  - DSO: 1,584 economic variables, |∇f| from 326 to 15,809, median 4,528; τ_u / |∇f| from 6.3e-7 to 3.1e-5.

  So the df = 1e-3 scaling leaves economic stationarity enforced to at most 3.1e-5 relative, or 1e-4 €/MWh. The
  T-perf solves terminated at unscaled dual infeasibility 2.94e-3 (TSO; scaled 2.94e-6 against tol 1e-5, which is
  29 % of the tolerance) and 3.56e-5 (DSO).
- **Against the AL and proximal variables:**
  - TSO: 264 variables, median |∇f| 21, max 2,788;
  - DSO: 120 variables, median 8.0, max 1,690.

  Their τ_u / |∇f| is about 5e-4 (TSO) and 1.2e-3 (DSO) at the median, and over 1 for AL variables whose gradient
  at the certified point is below 1e-2.

### Q2: per-cycle wall time

**What exists for this cell.** The paths are under `…/evals/ca8927e75d628bd1_n7_4h_e1/`. All of these files are
committed, and each matched its campaign and child manifest pins.
- **Per-cycle wall: recorded.** Production prints `[INFO] \t - Iteration N: X.XX s` into `child_stdout.log` for all
  112 cycles. The interval runs from `iter_start` at the top of the cycle body to `iter_end` at its end, in
  `shared_resources_planning.py`.
  - Cross-check: the 113 harness `drained` UTC stamps in `convergence_depth_append_events.jsonl` give
    drain-to-drain intervals equal to the printed walls within 4.4 ms on cycles 2–112. Cycle 1 is the exception:
    36.42 s against 29.54 s, because that interval also contains the post-initialisation work.
  - The printed cycle walls sum to 3,247.28 s, against a production execution time of 3,279.41 s.
- **Per-solve wall: not recorded anywhere for this cell.**
  - `network_ipopt_solve_records.jsonl` (5,425 records: 5,424 primary, 1 recovery) has no time-like field. Its 23
    fields are listed in `q2_cycle_time.json`.
  - The cell's IPOPT logs are on the Studio (`…/P56A/evals/p515s44_s53_w86_tail_recert_ca8927e75d628bd1_run/logs`).
    They are not on this machine.
  - `evaluation_record.json`, `post_certification.json`, `g_s39_D.json`, `heartbeat_s39_D.json` and
    `wait4_rusage.json` carry only whole-run walls.
  - ESSO solves (3 per cycle, from `leak_classification_s39_D.jsonl`) have no time either.
- **Per cycle (112 cycles):**
  - 48 network solves; 49 in cycle 6, which had 1 recovery;
  - 3 ESSO solves;
  - IPOPT iterations per cycle: TSO median 497.5 (390–833), DSO median 1,756 (1,620–2,612);
  - **cycle wall median 28.70 s (27.55–33.93)**. The prediction's premise of 35 s/cycle does not describe this
    cell.

**Proxy for Σ solve (labelled as a proxy; it mixes machines).**
- **Model.** Per-solve time = F + r × iterations, where r = IPOPT OverallAlgorithm / iterations and F = (wall −
  OverallAlgorithm).
  - The wall is either the whole `run_smopf` call (which includes the NL write and result load) or the IPOPT
    process only.
  - Both were measured in T-perf on this machine (Mac17,3).
  - TSO records use the case9 parameters. All DSO records use the case33_2 parameters; case33_1 and case33_3 were
    never measured.
  - ESSO solves are left out of the proxy; their time falls in the residual.
- **Calibrations.** Calibration A uses `i_warm`; B uses `ii_cold`. Each was checked against the case it was not
  fitted to.

  | Calibration (checked on) | TSO run_smopf | TSO IPOPT process | DSO run_smopf | DSO IPOPT process |
  |---|---|---|---|---|
  | A (`ii_cold`) | +19.4 % | +24.5 % | −1.3 % | −5.8 % |
  | B (`i_warm`) | −15.6 % | −21.3 % | −0.8 % | +4.7 % |

  The DSO dominates Σ: 36 of the 48 solves, each taking about 3.7× as long as a TSO solve (T-perf run_smopf walls of
  0.43 s against 0.117 s).

**Per-cycle shares** (median, min and max over 112 cycles; the full per-cycle table is in `q2_cycle_time.json`):

| Proxy | Σ solve [s] | Non-solve share | Cycles below 40 % | Break-even factor k* |
|---|---|---|---|---|
| A, run_smopf | 16.86 (16.03–21.11) | **41.3 %** (35.2–45.2 %) | 17 / 112 | 1.02 (0.93–1.09) |
| B, run_smopf | 16.40 (15.56–20.99) | **42.9 %** (36.7–46.8 %) | 7 / 112 | 1.05 (0.95–1.13) |
| A, IPOPT process | 10.58 (9.76–14.68) | **63.2 %** (56.7–66.3 %) | 0 | 1.63 (1.39–1.78) |
| B, IPOPT process | 10.79 (9.94–15.24) | **62.5 %** (55.1–65.6 %) | 0 | 1.60 (1.34–1.75) |

k* = 0.6 × cycle wall / Σ proxy. It is how many times slower per solve the Studio would have to be than this
machine for the non-solve share to fall to 40 %.

**Prior art (context only: different runs and code era, on the Studio; hashes verified against their committed
manifests).**
- **P5.15 Step 3.6 parallel audit** (`data/SRP1/Results/P515S36/parallel_audit/per_cycle_table_compact.json`,
  `e32470fc…`, run P515S35_REF, 12 sampled cycles):
  - IPOPT "Total seconds in IPOPT" median 12.99 s of a 33.73 s wall;
  - non-IPOPT share median **59.5 %** (52.0–65.3 %);
  - ESSO IPOPT time 0.030–0.042 s per cycle, so ESSO is negligible in the residual.
- **P5.15 Step 3.6 instrumented 10-cycle timing**
  (`data/SRP1/Results/P515S36/step36_timing_10cyc_defectfix/phase_timing_analysis.json`, `6d7036fe…`, matching its
  manifest; read-only, not part of this stage's manifest), totals over 10 cycles:
  - wall 358.80 s, IPOPT 195.32 s;
  - NL write 69.64 s, sol parse 22.87 s, load_solution 18.58 s, solve-bundle glue 1.77 s;
  - node-7 failure-snapshot clone 31.49 s, param_update 4.40 s, admm_global 4.55 s.

  The non-IPOPT share is 45.6 %. Outside the solver calls (that is, outside IPOPT + NL write + sol parse + load +
  glue) the share is about **14 %**, of which the node-7 clone alone is 8.8 %.

**Prediction against outcome** (recorded in the harness and logged as the first line of `run.log` before any input
was read). Prediction: *"on SRP1 the non-solve share of a cycle is ≥ 40 % (35 s/cycle recorded vs ≈ 15 s of
solves measured in T-perf)"*.
- **Solve = the IPOPT process: HOLDS (proxy level).** Medians are 62.5–63.2 %, every cycle is ≥ 55 %, and k* is
  1.6. The Studio prior art is consistent (59.5 % and 45.6 %).
- **Solve = the whole `run_smopf` call: INDETERMINATE.**
  - The proxy medians, 41.3 % and 42.9 %, are above 40 %, but 17 and 7 cycles fall below it.
  - The margin corresponds to k* = 1.02–1.05, which is inside the unmeasured Studio-vs-Mac17,3 per-solve factor
    and inside the TSO proxy's held-out error.
  - The instrumented Studio prior art (about 14 %, a different run) contradicts the prediction.
- **The premise was off for this cell.** This cell's cycles take 28.7 s, not 35 s.

## Unexpected findings (not acted on)

- **Slack values are slightly negative at the certified point.** They sit at about −1.0e-8, inside IPOPT's bound
  relaxation. As a result the penalty totals there are slightly negative:
  - `total_slack_penalties`: −0.12 € (TSO) and −0.83 € (DSO);
  - `total_ess_complementarity_penalties`: −0.002 €.

  The values are in `q1_gradient.json` `base_terms`. They are negligible in size and are noted only because they
  sit inside Q's components.
- **Penalty constants that do not appear.** The T-perf report pointed to penalty constants of 1e3–1e6 in
  `definitions.py`. Only 5e4 and 1e3 × baseMVA occur as slack coefficients in these two active objectives, and no
  gradient exceeds 1e5. So none of the 1e6 constants (`PENALTY_NODE_BALANCE`, `PENALTY_GENERATION`, `PENALTY_ESS`,
  `PENALTY_SHARED_ESS`, `PENALTY_SETTLEMENT`) reaches a free variable of either active objective with its full
  weight. I checked this through the gradient; I did not trace each constant's use.
- **Earlier in-repository timing work bears on T-perf's lever table.**
  - The P5.15 Step 3.6 audits (`WORKER_REPORT_S36_PARALLEL_AUDIT.md`, `WORKER_REPORT_S36_TIMING_10CYC.md`) already
    measured a per-cycle phase split on the Studio.
  - P5.15 S40/S42 (`p515_s42_persistent_workers_timing_5cyc.py`) measured the persistent-worker parallel path.
  - `P5_16_TPERF_REPORT.md` does not cite them.

  Scope of this search: `git ls-files` and a grep of `p5*.py` for per-solve timing captures. Branches were not
  searched.

## Not confirmed

- **Per-solve wall times for this cell.** I looked in:
  - every field of `network_ipopt_solve_records.jsonl`;
  - the JSON keys of `evaluation_record.json`, `post_certification.json`, `g_s39_D.json`,
    `convergence_depth_tail_state.json` and `boyd_terminal.json`;
  - `child_stdout.log`;
  - the cell directory, to see whether the IPOPT logs were present. They are absent here; the records point to the
    Studio path.

  Q2's Σ solve is therefore a proxy, not a measurement.
- **The Studio/Mac17,3 per-solve speed ratio** was not measured. Every k* statement is conditional on it.
- **Per-solve times for case33_1 and case33_3** (DSO nodes 5 and 9) were never measured. The proxy assumes case33_2's
  values.
- **ESSO time in this cell.** It is assumed small only from the prior-art audit (0.03–0.04 s per cycle on a different
  run).
- **The Python-side breakdown of the residual** (param updates, captures, clones, ADMM global updates). This cell's
  records cannot give it; only the Step 3.6 prior art decomposes it, and for a different run.
- **s_d = 1 holds only at termination.** It was inferred from the printed final statistics. During the iterations
  s_d may exceed 1, which would loosen the test by the same factor. The per-row dc factors were not recomputed
  here; T-perf's replica did that.
- **Q1 covers the certified point of two blocks only**, TSO and DSO-7 for 2025 Spring. Penalty and price gradients
  are constant (they are linear coefficients). T-perf's replica found max |∇f| = 1e5 at the cold start point as
  well. The AL gradients are point-dependent.
