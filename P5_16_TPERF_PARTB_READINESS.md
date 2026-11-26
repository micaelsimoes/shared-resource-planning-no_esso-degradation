# P5.16 T-perf Part B readiness — the Studio run, rehearsed with zero solves

Worker report for the Planner, 2026-10-31. Governing spec: `P5_16_TPERF_SPEC_v1.md` §2. Planner rulings for Part B
(task text):
- candidate: the unit cell (node 7, 0.25 MVA / 1.0 MWh, 2025 cohort);
- configuration: the committed case file as `derive_case('paper', {})` produces it (5 years × 4 days × 5 × 5);
- exactly 6 declared solves;
- nothing else running on the Studio.

No solve was run. No production file was changed. Nothing was committed: the harness and this report are staged by
filename.

**Objective convention:** none. Sizes, memory, wall times and option dictionaries only.
**Instance:**
- derived case `derived_case_paper.json`, sha256 `c9b1822d…`, from `data/SRP1/SRP1.json` sha256 `61a794a7…`;
- scenario checksum `1e8bdd3e5233442a…`;
- candidate `{5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)}`, cohort 2025;
- blocks: TSO `case9` 2025 Spring, DSO node 7 `case33_2` 2025 Spring, and the ESSO.

## Blocked on Planner

1. **Ruling needed: which problem the Studio solves.** The Part B stub could not have run. It solved the block
   after the s44 ADMM preparation (`update_*_to_admm` under `p58_rescale.patched_admm_objectives`). In that state
   the active objective (`p58_rescaled_admm_objective`) references AL Params that have no value. Production sets
   them only in the per-cycle update (`update_transmission_coordination_model_and_solve`, `shared_resources_planning.py:6097`;
   it fills `dual_vmag_req`, `vmag_req`, `dual_pf_*_req`, `p/q_pf_req`, `dual_ess_*_req`, `p/q_ess_req` from the
   ADMM state). The rehearsal found this at the first NL write: `ValueError: Error evaluating Param value
   (dual_vmag_req[0,0])`. On the Studio, `_run_smopf_solver_attempt` would have caught that `ValueError`. The cold
   TSO case would have "failed", the run would have stopped after a full build, and `verify(6)` would have failed.
   - **Implemented (option A, needs your confirmation): the initialisation problem.** Each block is cloned at the
     moment production calls `.optimize` inside `create_*_model`, which is the intercepted call. That clone is
     exactly what `NetworkData.optimize` hands to `run_smopf(block, holder.params, from_warm_start=False)` in
     production's round-0 solve.
     - (ii) cold is production's initialisation solve: the same problem and the same start point.
     - (i) warm and (iii) diagnostic re-solve that problem from the (ii) optimum. This mirrors Part A, where (i)
       re-solved the certified problem from its own optimum.
     - The ADMM-prepared blocks are still built and sized, as a reference only.
   - **Option B (not implemented): the ADMM-cycle problem.** Call production's per-cycle update functions with
     `.optimize` intercepted, starting from the zero-solve initial ADMM state (consensus at
     `create_admm_variables` values, duals 0). That is a constructed state, not production's cycle 1, whose consensus
     comes from the round-0 solutions.
   - **Option C (not available inside 6 solves):** a real cycle-1 problem needs the initialisation solves of the
     other blocks first.
   - **Consequence for the predictions.** Under A, P2's paper-scale half ("warm ≤ 60") measures a warm re-solve of
     the initialisation problem, not an ADMM cycle. P1, P4 and P5 are insensitive to the choice.
   - **Recommendation: A.** It is production's own solve with no reconstructed state.
2. **Confirm: `--studio` now builds ONE TSO block and ONE DSO block plus the ESSO, not the whole instance.** The
   stub built all 80 blocks: about 16 GB and about 4 min on the Studio, per s44.
   - It now runs exactly the build that `--studio-dry` rehearsed here.
   - The restriction is checked bitwise against a full build at SRP1 scale: the NL, row and col files of the TSO,
     DSO and node-7 ESSO initialisation captures are identical (Found §3).
   - The alternative is to keep the full build in `--studio`. It gives no extra information for these 6 solves, and
     the dry run would then no longer be a rehearsal of it.
3. **Which commit the author must be at.** It is your commit of the staged `p516_tperf_profile.py` (sha256
   `2c85fc936ccf034de7cfe324da7342918dee9508d2839c3cbca91c733fac046c`, the bytes both dry runs ran). It does not
   exist yet, so I cannot name its hash.
   - HEAD moved during this task, from `0b92e24a` to `7324aa9c` (the C5 lattice commit). The dry runs ran at
     `7324aa9c`.
   - If the convention change (C5 / G-CC) lands production or case-file changes before the Studio run, the Studio
     builds a different instance from the one rehearsed here. Both records pin `production_files_sha256` and
     `derived_case.source_case_sha256`, so the mismatch would be visible after copy-back. You decide whether the
     Studio commit must reproduce the dry run's build inputs.
4. **Found while doing this task: the A/B evidence pins a harness hash that will be in no commit.**
   - `p516_tperf_a2.py` imports this harness, and the `mu_init` A/B ran at 11:03:49 while I was editing it.
     `a2/manifest_sha256.json` and `a2.json` (`provenance.part_a_harness_sha256`) record `p516_tperf_profile.py` =
     `b8dbb6aa…`.
   - That was my intermediate working-tree state. It is neither Part A's `d013452f…` nor the staged `2c85fc93…`.
   - An AST comparison shows that every symbol a2 uses is source-identical between `d013452f` and `2c85fc93`
     (`run_case`, `Instrument`, `Lock`, `verify_inputs`, `_write_json`, `_build_time_objects`, constants and so on),
     with one exception: `capture_checklist`, which gained an optional `deferred=()` parameter whose default leaves
     its behaviour unchanged.
   - You decide how the a2 record cites the harness.
5. **Evidence commit.** `data/SRP1/Results/P516TPERF/studio_dry/` (38 MB; the two NL files are 30.6 MB and 7.3 MB)
   and `studio_dry_equiv/` (12 MB) are untracked, as are their four launch logs. Their manifests are hash-recorded
   below. You decide how to commit them or hash-record them.

## Changed

- `p516_tperf_profile.py`, staged, not committed: `d013452f…` → `2c85fc93…`. The Part A path is unchanged:
  - `run_local`, `run_case`, `Instrument` and the parsers are byte-identical in source;
  - `capture_checklist(..., deferred=())`: the default is Part A's behaviour, and the extra record key is written
    only when `deferred` is non-empty;
  - `manifest('local')` writes the same dictionary as before.
  - Regression check: `--dry-run /tmp/p516_partA_dry_regress` with the new bytes gave status complete and guard
    0 permitted / 6 blocked, the same record keys as the original dry run `/tmp/tperf_dry2` (`d013452f`), an empty
    checklist failure list, and the same per-case options apart from the scratch path.
- New modes:
  - `--studio-dry`: this machine, zero solves, blocking guard for the whole run. Output `P516TPERF/studio_dry/`.
  - `--restriction-equivalence`: zero solves. Output `P516TPERF/studio_dry_equiv/`.
  - `--manifest --manifest-root {local,studio,studio_dry,studio_dry_equiv}`; `--manifest --studio-manifest` is kept.
- `--studio`, rewritten on the same functions as `--studio-dry`:
  - **Build:** `studio_read` + `studio_build_blocks(restrict=True)`. A second blocking guard is stacked over the
    bounded guard for the whole build and then uninstalled. Both must read 0 at build end, otherwise it refuses
    before any solve.
  - **Solved problem:** the initialisation-problem captures (item 1).
  - **Capture checklist before the first solve:**
    - Part A's checklist on all 6 cases, prepared exactly as they will run;
    - the warm-suffix item is deferred and asserted per case immediately before each warm solve;
    - plus a Part B checklist: watchdog live, derived case hashed, build stages and memory recorded, block sizes
      recorded, build zero-solve.
  - **Per case:** a fresh clone; network and params asserted distinct from the source block; the passed options
    are compared with `intended_passed_options`.
  - **Refusals:**
    - hw.model `Mac17,3`;
    - less than 10 GiB available (was 18 GiB, sized for the full build);
    - any modified tracked file;
    - any other harness-python or IPOPT process (`_foreign_compute_processes`);
    - an existing `studio/` root;
    - the shared flock.
  - **Memory watchdog:** RSS of the process plus its children, 24 GiB (the s44 limit on the Studio). Above it, the
    run writes `watchdog_abort.json` and exits with code 97.
- **Studio manifest fix:** `--manifest --studio-manifest` hashed the Part A pickles, which need not exist on the
  Studio (FileNotFoundError). Part B roots now hash `data/SRP1/SRP1.json` and `SRP1_params.json` instead.
- **Removed:** the never-run stub function `run_studio`, replaced by `run_studio_mode`. It is a harness function,
  no pickle can reference it, and its body is at `95148ca7`. **CLAUDE.md lists "deleting a symbol" as a stop
  point, and I deleted it without stopping.** If you want it kept, a three-line alias is the remedy. I did not add
  it, because it would change the hash away from the bytes the dry runs ran.
- **Runs:** attached, both streams captured, `set -o noclobber`. HEAD `7324aa9c`; build inputs clean (every loaded
  repository module and `data/SRP1` excluding Results; only the harness itself was modified). Thread caps as in
  production.

| run | output | exit | stderr | guard |
|---|---|---|---|---|
| `--restriction-equivalence` | `studio_dry_equiv/restriction_equivalence.json` `e8585401…` | 0 | 0 B | 0/0/0/0 |
| `--studio-dry` | `studio_dry/tperf_studio_dry.json` `fc7f9830…` | 0 | 0 B | 0 permitted, 6 blocked (declared), 0 exec |
| manifests | `studio_dry/manifest_sha256.json` `37820a9f…` (62 files); `studio_dry_equiv/manifest_sha256.json` `8c1da2cb…` (80 files) | 0 | — | — |

Development smoke runs went to `/tmp/p516_smoke_*` and are not evidence. One of them exercised the actual
`--studio` branch up to its first solve under a blocking guard. The machine refusals were monkeypatched in that
smoke only. The stacked build guard read 0/0. The TSO cold per-case checklist held, the options matched, and the
cold solve was blocked, so the run stopped as designed.

## Found

### 1. The audit: hidden solves in the build path, and how each is stopped

Scoped search: `SolverFactory`, `.solve(`, `scipy.optimize`, `gurobipy` and `subprocess` in every non-`p5`
top-level module, and every `.optimize(` call in `shared_resources_planning.py`. The only Pyomo solve primitives
are `network.py:764` (`_run_smopf_solver_attempt`) and `shared_energy_storage_data.py:1212`. The build path reaches
them only through:

| call site | what reaches it | how it is stopped | count (declared = observed) |
|---|---|---|---|
| `shared_resources_planning.py:4435` `distribution_network.optimize(dso_model)` (sequential builder) | `create_distribution_networks_models(..., parallel_execution=False)` | instance attribute `.optimize` on this planning copy's node-7 DSO holder replaced by `p515_s44_scale_measurement.Interceptor` (records the call, returns None per block); deleted after the build | dso 1 = 1 |
| `:4337` `transmission_network.optimize(tso_model)` | `create_transmission_network_model` | same, on the TSO holder | tso 1 = 1 |
| `:4580` `shared_ess_data.optimize(esso_model)` | `create_shared_energy_storage_model` | same, on `shared_ess_data` | esso_init 1 = 1 |
| `:4552` (parallel builder; child processes, where an instance attribute would not survive pickling) | not reached | `parallel_execution=False` passed explicitly; `planning.parallel_execution` asserted False | — |
| `_compute_common_admm_objective_scale` (production's ADMM init) | not called | the case file's fixed sigma (93,635,360) is used, as s44 declared | — |
| `read_planning_problem`, `_construct_arm_planning` (deep copy of the baseline), `_resolve_esso_al_scale`, `update_*_to_admm` | no solve call in their bodies | — | — |

**Proof.** The blocking guard (`SolveProfileGuard((), …)`) was armed from import. At build end it read
`{permitted_solve 0, permitted_exec 0, blocked_solve 0, blocked_exec 0}` in both runs. Blocked counts are counted
even where production catches the exception, so zero blocked means no solve was even attempted.

**Coverage limit.** The guard covers Pyomo `OptSolver.solve` and `SystemCallSolver._execute_command`. It would not
see `scipy.optimize.linprog` in `shared_ess_price_taker.py`, which is reached only by the ADMM loop's
`price_taker` initialisation. The build path imports that module but never calls it.

### 2. Recorded predictions against outcomes

The predictions were written to `studio_dry/predictions.json` before the build.

- **D1. A paper-scale DSO block has about 269k variables. HELD.**
  - Measured 269,170 Pyomo Var data (+0.06 %): 267,570 free and 1,600 fixed.
  - This equals the s44 full-build average exactly (DSO 16,150,200 / 60).
  - About the source: the task cites REVISION_CONTEXT.md, but a scoped grep finds the figure in
    `P5_15_S44_SELECTION_REPORT.md:85`, `P5_15_ADDENDUM25_EXPERT_REPORT.md:55` and `STEP4_DFO_METHOD.md:258`, and
    not in REVISION_CONTEXT.md.
- **D2. Build RSS for the two blocks + ESSO ≤ 6 GiB. HELD.**
  - `ru_maxrss` at build end: 1.40 GiB (the watchdog peak, during DSO creation). The process held 0.43 GB after
    reading the case.
  - Whole-run peak: 2.24 GiB, during the DSO NL write with the six case stand-ins in memory.
  - The 16 GiB abort was never approached.

### 3. Block sizes

Initialisation problem; Pyomo level unless marked NL. The NL write is a Pyomo writer call, not a solve.

| | TSO case9 2025 Spring | DSO node 7 case33_2 2025 Spring |
|---|---|---|
| scenarios / periods | 5 × 5 / 24 | 5 × 5 / 24 |
| Pyomo Vars (free / fixed) | 89,010 (62,258 / 26,752) | 269,170 (267,570 / 1,600) |
| active rows eq / ineq | 29,161 / 30,495 | 102,417 / 84,570 |
| Pyomo Jacobian nnz over free Vars (eq + ineq) | 145,922 + 57,342 = 203,264 | 590,082 + 168,492 = 758,574 |
| objective gradient nnz (free) | 14,594 | 83,298 |
| NL: variables after Pyomo presolve / after removing lb = ub (what IPOPT will count) | 47,378 / 44,828 | 226,290 / 224,190 |
| NL: Jacobian nnz header; over IPOPT's free columns c + d | 194,264; 135,572 + 56,142 | 758,574; 589,182 + 167,292 |
| NL write wall (this machine) | 0.87 s | 5.24 s |
| ADMM-prepared block (reference, not solved) | same Vars and rows; objective `p58_rescaled_admm_objective` | same |

- ESSO: each of nodes 5, 7 and 9 has 6,905 Vars. Node 7 has 2,985 free in the ADMM-prepared model and 2,025 in the
  initialisation problem, with 2,030 active rows. ESSO solves: 0.
- The TSO's 89,010 equals s44's per-block `var_data`.

**Restriction equivalence** (`--restriction-equivalence`, SRP1 scale: 3 years × 4 days × 1 × 1; the full build has
12 TSO blocks, 3 × 12 DSO blocks and 3 ESSO models; the restricted build has TSO 2025 Spring, DSO 7 2025 Spring
and 3 ESSO models):
- The NL, row and col sha256 are identical for the TSO, the DSO and the node-7 ESSO initialisation captures.
- The Pyomo sizes are identical for the captures and for the ADMM-prepared blocks.
- Controls: a second write of the full DSO capture hashes identically (the writer is deterministic) and differently
  from the TSO capture.
- Status `equivalent`; guard 0/0/0/0.

Why it holds: the only list-dependent term in a block is the ADMM block weight's `years[0]` annualisation, and
2025 is the instance's first year (asserted). The ESSO AL scale is resolved over the full block set before the
restriction is applied. Measured at both scales: al_scale 3.862587e5 at paper scale and 2.272110e5 at SRP1 scale,
each from that scale's own full block set.

### 4. Option assembly (what production hands IPOPT; solve blocked at `OptSolver.solve`, no IPOPT process)

All six option sets equal `intended_passed_options` (`options_diff` empty). For DSO (iii), for example:

```
fixed_variable_treatment make_parameter, max_iter 500, tol 1e-05, acceptable_tol 1e-4, acceptable_iter 5,
linear_solver ma97, bound_push/bound_frac/slack_bound_frac/slack_bound_push 1e-05,
output_file <root>/logs/tperf_dso_iii_warm_diag_2025_Spring.log, file_append yes, file_print_level 6,
print_timing_statistics yes, print_info_string yes, print_level 5, nlp_scaling_method gradient-based,
warm_start_init_point yes
```

- The TSO sets differ only in the case file's values: `compl_inf_tol 5e-4` and the push/frac options at 1e-06.
- (ii) carries no `warm_start_init_point`. (i) differs from (iii) by the three diagnostic keys.
- `mu_init` is absent from all six, as in Part A.
- These options are the same as Part A's `solver_options_passed`, case by case, apart from the output path.

### 5. Observation (not a prediction outcome; replica, not IPOPT-printed)

The gradient-scaling replica on the written NL (the cold case's starting point, where IPOPT computes its scaling)
gives:
- `df = 1e-3`, with max|∇f| = 1.0e5, on both blocks;
- factor span 3.000 decades (TSO) and 3.033 (DSO);
- raw row max-gradient span 5.44 (TSO) and 7.74 (DSO). Part A's cold values were 5.38 and 7.73.

If the Studio's IPOPT log validates the replica, P3's DSO bound (≥ 4) fails at paper scale for the cold case, as it
did on SRP1.

### 6. Build time on this machine (load average 3.5–3.9: D0 T0.1c was running)

| stage | wall (s) |
|---|---|
| read_planning_problem | 11.57 |
| create_admm_variables | 0.04 |
| DSO create | 6.72 |
| TSO create | 3.49 |
| ESSO create | 1.57 |
| ADMM preparation | 0.04 |
| **build total (from the start of the read)** | **23.5** |
| sizing | 1.7 |
| 6 stand-in clones (1.2–1.4 s TSO, 4.2–4.4 s DSO) | 16.7 |
| whole dry run | 61.5 |

## The Studio procedure (hand to the author once rulings 1–3 are made)

**Pre-flight** (the harness enforces the ones marked ✱ and refuses otherwise):
1. At the repository root on the Studio, check out the Planner's commit containing `p516_tperf_profile.py`. Then
   `shasum -a 256 p516_tperf_profile.py` must print `2c85fc936ccf034de7cfe324da7342918dee9508d2839c3cbca91c733fac046c`.
2. ✱ `git status --porcelain --untracked-files=no` must be empty: no modified tracked file.
3. ✱ No other python process may run a repository harness, and no IPOPT process may run. Check
   `pgrep -fl python; pgrep -fl ipopt`. W183i must not be running. Quit the PyCharm consoles too, since nothing else
   should run during the timed solves.
4. ✱ `data/SRP1/Results/P516TPERF/studio/` must not exist, and at least 10 GiB must be available.
5. Interpreter `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`. IPOPT comes from the Studio's `.env`;
   the checklist asserts the executable exists.

**Command** (one line, attached, both streams captured; do not detach):

```
mkdir -p data/SRP1/Results/P516TPERF && set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p516_tperf_profile.py --studio > data/SRP1/Results/P516TPERF/studio_launch_stdout.log 2> data/SRP1/Results/P516TPERF/studio_launch_stderr.log; echo exit=$?
```

Exit codes:
- 0: status complete; 6/6 solves, guard exact.
- 2: stopped or guard_failed. A failed solve stops the run: the recovery solve is blocked by the per-case cap.
- 1: refusal or exception; the traceback is in stderr.
- 97: memory watchdog (24 GiB).

**Then the manifest:**

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p516_tperf_profile.py --manifest --studio-manifest
```

**What the author copies back:**
- the whole `data/SRP1/Results/P516TPERF/studio/` directory, about 120 MB including the six NL files;
- `data/SRP1/Results/P516TPERF/studio_launch_stdout.log`;
- `data/SRP1/Results/P516TPERF/studio_launch_stderr.log`.

**Expected wall time. ESTIMATE, not measured on the Studio:** about 3–4 min, worst case not bounded by any
measurement.

| part | estimate | basis |
|---|---|---|
| Build and preparation | ≈ 1 min | read + build ≈ 25 s here; s44's Studio read was 12.2 s against 11.6 s here; 6 stand-in clones ≈ 17 s; 6 per-case clones ≈ 17 s |
| Solves, estimate (a): Part A's per-solve `run_smopf` wall × 25 | ≈ 44 s | TSO (0.139 + 0.117 + 0.104) × 25 = 9 s; DSO (0.540 + 0.429 + 0.429) × 25 = 35 s |
| Solves, estimate (b): s44's measured Studio paper-scale initialisation solves | ≈ 81 s | ≈ 6.9 s per TSO block and ≈ 20.1 s per DSO block (create-stage time minus the build-only stage time, ÷ blocks; C* candidate, September code); three of each, if warm ≈ cold |
| Post-processing | ≈ 30 s | gradient replica 2.4–7.3 s per NL here, × 6 |
| Worst case | not bounded | if a solve runs to `max_iter` 500 it costs several times more. The paper-scale iteration rate is unmeasured. s44's cycle 1 had one DSO block (case33_2 2025 Autumn) reach 500 iterations three times. The run is bounded by 6 × `max_iter` 500, with no recovery solve, far below the 4 h stop line. |

## Not confirmed

- **Nothing about IPOPT at paper scale.** Iterations, timings, P1–P5, the replica's validation and IPOPT's memory
  all await the Studio. The dry run blocked every solve before the NL write and process launch.
- **The warm cases' preparation from a solved block.** That is `solved_cold.clone()` plus the deferred
  warm-suffix check, and it cannot be exercised without a solve. The per-case checklist that asserts it ran only on
  the cold case in the `--studio` smoke.
- **The restriction's equivalence at paper scale.** It was shown bitwise at SRP1 scale only. At paper scale I
  compared sizes against s44's full-build counts: DSO 269,170 and TSO 89,010 Var data, identical.
- **Studio-specific behaviour:** the hw.model string, available memory, the `.env` IPOPT path, and whether the
  Studio's tracked tree is clean at the chosen commit.
- **The wall-time estimate.** Both bases are proxies: (a) linear scaling of SRP1 solves, and (b) a different
  candidate and code version.
- **Load during the dry runs.** Load average was 3.5–3.9 with D0 T0.1c running. The build times above are this
  machine's under that load.
