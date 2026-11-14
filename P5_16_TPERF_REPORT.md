# P5.16 T-perf — Part A report (this machine): where the solve time goes on the certified SRP1 unit-cell blocks

Worker report for the Planner, 2026-10-31. Governing spec: `P5_16_TPERF_SPEC_v1.md`. Part A only. Part B (Studio)
has not been run. No ADMM cycle was run. No production code was changed.

**Objective convention:** none. This report computes and compares no objective value. It reports times,
iterations, memory and scaling. The certified cost quoted under Instance only identifies the cell.

**Instance:** campaign `s53_w86_tail_recert` (spec `ddd6cd44`), cell `n7_4h_e1`, eval key `ca8927e75d628bd1…`,
candidate_key `db77e1549af855bc…` (2025 cohort; node 7 0.25 MVA / 1.0 MWh; nodes 5 and 9 empty). Certified at
cycle 112 (certified gross_operational_cost 653,599,356.24, settlement excluded). The blocks are TSO 2025 Spring
(`case9`) and DSO node 7 2025 Spring (`case33_2`). Before unpickling, `certified_models.pkl` (sha256 `fe06ca48…`)
and `esso_models_s39_D.pkl` (`95f7b38a…`) were checked against the task's declared hashes, the campaign manifest
and the child manifest. All three agreed.

## Blocked on Planner

1. **Predictions P2 and P3 failed in part on SRP1, and the spec names no fallback** (see Found). P2 failed on both
   counts: the DSO warm solve took 51 iterations against a bound of ≤ 30, and the cold/warm ratio was 2.55× (TSO)
   and 1.45× (DSO) against ≥ 3×. P3 failed on the DSO block: the scaling factors span 3.03 decades against ≥ 4.
   These are diagnostic predictions, not campaign gates. The Planner rules on what follows. Nothing was changed
   because of them.
2. **Part B candidate and configuration need confirming before the author runs `--studio`.** The spec does not name
   a candidate for the paper-scale blocks. The stub reuses the established paper-scale build path of
   `p515_s44_scale_measurement`:
   - the derived case, the production reader and `p515_g_g1_g4_admm_gates._construct_arm_planning` with
     `apply_rho=False`, which brings that path's D-arm budget and adaptive-penalty settings;
   - `create_*_model` with `.optimize` intercepted, then that stage's ADMM objective preparation.

   I chose the unit cell as the default candidate (node 7 0.25 / 1.0, 2025). The case order on the Studio is
   (ii) cold, then (i) and (iii), each warm from a clone of the post-(ii) block, because no certified point exists
   at paper scale. **The stub has only been compiled and imported. It has never run**, as the task instructed. It
   has no zero-solve dry mode of its own.
3. **A lever outside the spec's four turned up (production IPOPT configuration; expert ruling).** Re-solving the
   identical problem from its own optimum still took 22 (TSO) and 51 (DSO) iterations. The logs show the barrier
   restarting at lg(mu) = −1.0, because `mu_init` is not passed, so IPOPT uses its default 0.1. Details are under
   Found. This is not acted on.
4. **Commit.** As instructed, I staged only `p516_tperf_profile.py` and this report, and committed nothing.
   CLAUDE.md requires a report to be committed with its evidence base and hash inventory. That evidence is
   `data/SRP1/Results/P516TPERF/local/tperf_local.json` and `manifest_sha256.json`, both untracked. The Planner
   decides how to commit or hash-record them.

## Changed

- New harness `p516_tperf_profile.py` (sha256 `d013452f…`, the bytes that ran). It has four modes:
  - default: Part A;
  - `--dry-run <scratch>`: zero solves under a blocking guard;
  - `--manifest`;
  - `--studio`: the Part B stub, not run. It refuses on hw.model `Mac17,3` and below 18 GiB available.

  Concurrency is refused through an exclusive flock on `data/SRP1/Results/P516TPERF/.p516_tperf.lock` plus a
  process scan.
- Output `data/SRP1/Results/P516TPERF/local/` (new, write-once):
  - `tperf_local.json` (sha256 `2aa2c653…`);
  - `logs/`: six IPOPT `output_file` logs at file level 6, plus six process stdout+stderr logs;
  - `nl/`: the six NL files IPOPT actually read, hard-linked;
  - `manifest_sha256.json` (sha256 `fbfd6310…`, 26 files, including the launch logs and the inputs).
- Launch logs: `data/SRP1/Results/P516TPERF/local_launch_stdout.log` and `local_launch_stderr.log`. The stderr log
  is 0 bytes.
- Run: attached, alone, both streams captured, thread caps as the production child (`OMP_NUM_THREADS=1` and so on).
  git HEAD `42840e42`; production files clean (`git status` empty for network.py, network_data.py,
  shared_resources_planning.py, model_construction_helpers.py, helper_functions.py, network_parameters.py,
  p513_solve_profile_guard.py). IPOPT 3.14.18 with MA97, Pyomo 6.9.5.
- Earlier, two zero-solve dry runs went to `/tmp/tperf_dry1` and `/tmp/tperf_dry2`. They gave 0 permitted and 6
  blocked solves, the capture checklist held, and the replica self-test ran on all six NL files.
- **Solve guard.** `SolveProfileGuard` was bounded, with permitted site `network.py:_run_smopf_solver_attempt`,
  installed before any production import. `verify(6)` returned `[]`: counts 6 solves, 6 process launches,
  0 blocked. The per-case cap (no second solve inside one case, so no production recovery could add a 7th) blocked
  nothing. All six solves were primary attempts and all returned `optimal`.

How each case was produced:

- **(i) warm.** `network.run_smopf(model, params, from_warm_start=True)` on a clone of the certified block. The
  `network` and `params` are the block's own build-time objects, recovered from its pickled bound-rule partial (the
  p515_s48 precedent). This is the call `NetworkData.optimize` makes inside the two `update_*_and_solve` functions.
- **(ii) cold.** The production initialisation point:
  - every free Var reset to its `initialize=` rule (216 TSO and 768 DSO Vars have no initializer and were set to
    None);
  - `configure_shared_ess_operational_state` applied per shared ESS;
  - `_clear_multiplier_suffixes`;
  - bounds, fixed flags, active flags and Params asserted unchanged;
  - then `from_warm_start=False`.

  The problem solved is the certified cycle-112 problem, not the initialisation problem. That would need an ADMM
  initialisation.
- **(iii)** As (i), from a fresh clone, adding `print_info_string yes`, `print_level 5` and
  `nlp_scaling_method gradient-based`. The last two are already the defaults.

All cases add only `print_timing_statistics yes`, which affects output only, and a per-solve `output_file`.

## Found

### Measured table

One run per case. "Wall" is the `network.run_smopf` call. IPOPT internal times come from its timing statistics at
1 ms resolution.

| | TSO (i) warm | TSO (ii) cold | TSO (iii) warm+diag | DSO (i) warm | DSO (ii) cold | DSO (iii) warm+diag |
|---|---|---|---|---|---|---|
| IPOPT variables / eq / ineq | 2348 / 1513 / 1335 | same | same | 9246 / 4305 / 3498 | same | same |
| Jacobian nnz (eq + ineq) / Hessian nnz | 6116 + 2430 / 1938 | same | same | 24030 + 6876 / 5220 | same | same |
| iterations / restoration iterations | 22 / 0 | 56 / 0 | 22 / 0 | 51 / 0 | 74 / 0 | 51 / 0 |
| mu_final (floor status) | 4.545e-8 (at) | 4.545e-8 (at) | 4.545e-8 (at) | 9.091e-9 (at) | 9.091e-9 (at) | 9.091e-9 (at) |
| run_smopf wall [s] | 0.117 | 0.139 | 0.104 | 0.429 | 0.540 | 0.429 |
| Pyomo NL write [s] | 0.045 | 0.042 | 0.046 | 0.129 | 0.112 | 0.126 |
| result load: sol read + load_from [s] | 0.009 | 0.008 | 0.008 | 0.023 | 0.024 | 0.025 |
| IPOPT process [s] | 0.058 | 0.086 | 0.047 | 0.274 | 0.402 | 0.275 |
| IPOPT OverallAlgorithm [s] | 0.032 | 0.067 | 0.026 | 0.232 | 0.370 | 0.232 |
| linear solver (symbolic + numeric factorisation + backsolve) [s] | 0.018 | 0.041 | 0.014 | 0.163 | 0.268 | 0.162 |
| function evaluations [s] | 0.003 | 0.008 | 0.003 | 0.022 | 0.032 | 0.022 |
| (NL write + result load) / wall | **46.1 %** | 36.1 % | 51.9 % | 35.3 % | **25.2 %** | 35.2 % |
| linear solver / OverallAlgorithm | 56 % | 61 % | 54 % | 70 % | 72 % | 70 % |
| function evaluations / OverallAlgorithm | 9 % | 12 % | 12 % | 9.5 % | 8.6 % | 9.5 % |
| scaling: df; factor range; span | 1e-3; [1e-3, 1]; **3.000 dec** | same | same | 1e-3; [9.27e-4, 1]; **3.033 dec** | same | same |
| raw row max-gradient span (all rows) | 9.91 dec | 5.38 dec | 9.91 dec | 10.46 dec | 7.73 dec | 10.46 dec |
| free vars at a bound (rtol 1e-6 / 1e-8) | 406 / 344 of 2666 | same | same | 4293 / 3542 of 10074 | same | same |
| peak RSS: IPOPT child; Python increment during solve | 18.7 MB; +0.26 MB | 22.3 MB; +0.13 MB | 22.2 MB; +0.16 MB | 45.9 MB; +0.52 MB | 46.2 MB; +0.16 MB | 46.0 MB; +0.28 MB |

Notes on the table:

- **Size before presolve.** Pyomo free Vars are 2882 (TSO) and 10818 (DSO). The NL file has 2450 and 9330
  variables after Pyomo NL presolve. IPOPT sees 2348 and 9246 after it removes the lb = ub variables.
- **Repeatability.** Cases (i) and (iii) are the same warm problem. Their iterations are identical (22/22, 51/51).
  The wall spread is 11 % on the TSO and < 0.1 % on the DSO.
- **Overhead outside the algorithm.** IPOPT process time minus OverallAlgorithm is 19–26 ms on the TSO and 32–43 ms
  on the DSO. That covers process start, ASL reading the NL file and writing the sol file. The rest of run_smopf
  outside `solve()` (option assembly, suffix copy, the production solve-record parse, load_from) is 7 ms (TSO) and
  16–17 ms (DSO).
- **Memory.** The Python process holds about 1.8 GB, mostly left over from unpickling all 48 blocks (2.23 GB right
  after unpickling). `ru_maxrss` was 2.25 GB and is unchanged by the solves.
- **ESSO** (0 solves): node-7 model has 2949 Vars (1773 free) and 1200 active constraints; objective
  `admm_objective`.
- **Scaling-factor method.** The factors are a replica of IpGradientScaling, built from the NL file IPOPT read for
  that solve, evaluated at AmplTNLP's starting point. IPOPT 3.14.18 prints per-row factors only at level J_VECTOR
  (8), which also dumps every iterate, so it was not asked to print them. On all six solves the replica agrees with
  IPOPT's own log on nine checks: the printed objective factor (two lines), "c / d scaling provided", the variable,
  equality and inequality counts, and both Jacobian nnz counts.

### Predictions against outcomes (Part A)

- **P1 — NL write + result load ≥ 10 % of per-solve wall on SRP1 (≥ 20 % at paper scale): HOLDS on SRP1.** Measured
  25.2 % (DSO cold) to 51.9 % (TSO warm+diag), against the whole `run_smopf` wall. The NL write alone is 21–44 %.
  The paper-scale half is for the Studio.
- **P2 — warm iterations ≤ 30 on SRP1, and cold ≥ 3× warm: FAILS on SRP1, both parts.**
  - Warm: TSO 22 (holds); DSO 51 (fails).
  - Cold/warm: TSO 56/22 = 2.55×; DSO 74/51 = 1.45×. Both fail.
  - The cell's own committed production records show the same picture (zero-solve read of
    `network_ipopt_solve_records.jsonl`, hash pinned by both manifests). Over ADMM rounds 1–112, the TSO block took
    21–84 iterations (median 37; 7 of 112 rounds ≤ 30). The DSO block took 47–158 (median 53; 0 of 112 rounds ≤ 30).
    The initialisation solves (round 0, a different problem) took 32 and 46. In production the warm cycle solve is
    no cheaper than the cold initialisation.
  - The paper-scale half (≤ 60) is for the Studio.
- **P3 — scaling factors span ≥ 4 decades on the DSO and ≤ 3 on the TSO: DSO FAILS (3.033); TSO HOLDS at its bound
  (3.000).**
  - The TSO span is set entirely by the objective factor df = 1e-3. max|∇f| is exactly 1.0e5 in all six solves. In
    the TSO the constraint factors span only 0.78 decades: no equality-row scaling, and 168 inequality rows scaled,
    down to 0.166.
  - The DSO's extra 0.03 decades comes from one inequality row with row max gradient 1.08e5 (factor 9.27e-4). The
    DSO also scales 600 equality rows (down to 0.297) and 744 inequality rows.
  - Gradient-based scaling only scales down rows above 100. It cannot widen the span to reflect small rows. The raw
    row max-gradient spread, which is the conditioning quantity behind the "per-unit base" hypothesis, is 9.9 (TSO)
    and 10.5 (DSO) decades at the warm point, and 5.4 and 7.7 at the cold point. The smallest nonzero row maxima are
    7.5e-8 and 3.8e-6.
  - The factor span is reported because the spec names it. The raw spread is reported beside it.
  - Which term carries the 1e5 objective gradient (penalty constants 1e3–1e6 in `definitions.py:49-77`) was **not
    identified**.
- **P4 — linear solver ≥ 60 % of IPOPT time at paper scale: not measurable in Part A.** For reference only, on SRP1
  the linear solver is 54–61 % (TSO) and 70–72 % (DSO) of OverallAlgorithm. MA97 ran single-threaded under
  `OMP_NUM_THREADS=1`, as in production.
- **P5 — peak RSS of one paper-scale DSO solve ≥ 3 GiB above the build footprint: not measurable in Part A.** For
  reference only, on SRP1 the IPOPT child peaks at 46 MB and the Python process grows ≤ 0.52 MB during a solve.

### Observations (stated as such; not tested here)

- **The warm start restarts the barrier.** At iteration 0 every warm solve is at the previous optimum (inf_pr
  6.7e-6, TSO) but with lg(mu) = −1.0. mu then steps down: −3.8, −5.7, −7.3 on the TSO; −2.5, −3.8, −5.7, −8.0 on
  the DSO. inf_du jumps to 1e2–1e3 at each new barrier subproblem.
  - The passed options contain no `mu_init` (see `solver_options_passed` in the JSON), so IPOPT's default 0.1 is in
    force.
  - A scoped search found `mu_init` in no file of the HEAD tree (`git grep HEAD`) and in no .md or .py file of the
    working tree. On 8 of 25 local branches (including `master`) it appears only in `shared_energy_storage_data.py`,
    as the ESSO's `mu_init = 1e-9`.
  - Hypothesis (untested; it would need extra solves and a production-option change): the restart, not the start
    point, sets the warm iteration count.
- **Production resets SoC every cycle.** Before every cycle's solve,
  `update_transmission_coordination_model_and_solve` and the DSO counterpart call
  `configure_shared_ess_operational_state` (`shared_resources_planning.py:6116`, `:6450`). That function sets every
  `shared_es_soc` entry to 0.5 × E (`model_construction_helpers.py:1147-1150`). So production's warm start does not
  carry the previous SoC trajectory. Case (i) here did not apply that reset, being the certified point as stored.
  Its effect on iterations was not measured.

### Ranked lever table (as far as Part A supports it)

Shares are of the per-solve `run_smopf` wall on the SRP1 blocks on this machine (Mac17,3, not the Studio). They are
ranked by the measured share each lever could address.

| Rank | Lever | Measured share it could address (SRP1, Part A) | Gate a change would need | Risk |
|---|---|---|---|---|
| 1 | **Parallelism** (blocks within a round are independent) | Up to the whole per-block wall (100 % of `run_smopf`), per block. Not measured: the round-level speed-up, because blocks were solved one at a time. Memory per extra solve here is small (IPOPT ≤ 46 MB, Python +≤ 0.5 MB). The ~1.8 GB is held models, which parallel workers would duplicate. | Per-block results and the ADMM trajectory identical to sequential (the `p515_s36_parallel_audit` precedent). Each worker gets its own log path, never shared appended or relative files. | Memory at paper scale (P5, Studio); log-path collisions (the CLAUDE.md G1–G4 incident). No formulation change. |
| 2 | **NL-write avoidance** (persistent or in-process solver interface; also removes the sol read, load_from and process start) | NL write + result load **25–52 %**. Adding the IPOPT process overhead outside the algorithm (TSO 19–26 ms, DSO 32–43 ms) gives **31–72 %** per solve: TSO 50–72 %, DSO 31–45 %. | Same dispatch to tolerance and the same iteration counts on these blocks. Bitwise identity is unlikely, because Pyomo's NL presolve currently removes 432 TSO and 1488 DSO free Vars. | Changes the solve path: warm-start suffix transfer, presolve differences, the floor-status record parse (which reads the IPOPT `output_file`). No formulation change. |
| 3 | **Per-unit base / scaling** | Not measurable as time in Part A (a re-scaled re-solve would need extra solves). What it can act on is the iteration-proportional part. IPOPT OverallAlgorithm is 25–48 % (TSO; 48 % is the cold solve) and 54–69 % (DSO) of the wall, and the linear solver is 54–72 % of that. Measured conditioning: raw row max-gradient spread 9.9 / 10.5 decades; applied factors 3.0 / 3.03 decades; df = 1e-3 set by max|∇f| = 1e5. | C6 with the Step 1b candidate-3′ gate shape: same dispatch to tolerance, iterations and time before and after. | Formulation change. Enters the re-evaluation campaign. |
| 4 | **W-auxiliary structure** | Not measurable in Part A. No attribution of rows, nnz or time to the W-auxiliary components was made. | C6 / candidate-3′ shape, as above. | Formulation change. |
| — | *(outside the four)* **IPOPT warm-start configuration** (`mu_init`, warm-start pushes) | Iterations: warm at its own optimum is 22 / 51, against 56 / 74 cold. Production cycle medians are 37 / 53. Acts on the same iteration-proportional OverallAlgorithm share as lever 3. | Same dispatch to tolerance and the per-cycle trajectory gate. A production-configuration change, so expert ruling. | Changes the ADMM trajectory. Earlier warm-start work (H-WS, P5.13-D, P5.15-1a) shows this area is sensitive. |

## Not confirmed

- **All Studio quantities** (P1 and P2 paper-scale halves, P4, P5). The `--studio` stub is untested beyond compiling
  and importing. Checked: it reuses `p515_s44_scale_measurement` functions by import, and that stage's committed
  paper builds peaked at 16.1–16.7 GB RSS (`scale_measurement/paper_cycle_*/build_record.json`).
- **Timings are from this machine (Mac17,3), one run per case.** No Studio equivalence. IPOPT internal times have
  1 ms resolution (TSO function evaluations 3 ms ± 0.5 ms). Load average was 1.8 for the whole run. The process
  snapshot at each solve's start and end showed no other compute process: only shells, an http.server and the
  PyCharm console, and the D0 dry run was not running. Nothing was sampled between those snapshots.
- **Case (i) is not a production ADMM cycle.** It re-solves the identical cycle-112 problem from its own solution.
  The cycle's Param updates and SoC reset were not applied, because they need the ADMM state. The production
  per-round records above are the real cycle data.
- **Case (ii) is the initialisation point, not the initialisation problem.** It is the certified problem started from
  the initialisation point. The ADN-load pc/qc initial-consensus values and the initialisation objective were not
  reproduced.
- **Per-row scaling factors were not printed by IPOPT.** The replica is validated only on the printed objective
  factor, the c/d "provided" flags, the counts and the nnz.
- **The time saving from levers 2–4 and the warm-start lever.** Only the shares they could address were measured,
  not the speed-up any change would give.
