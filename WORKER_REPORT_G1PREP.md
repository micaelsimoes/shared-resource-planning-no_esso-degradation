# Worker Report -- P5.15-G1PREP (harness change + smoke test)

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 6 (harness-only change, "one gate per
Worker task with the exact command") and Addendum 5. Production baseline: commit
`7ca40b93` (P5.15-F, ESSO log fix) -- confirmed unchanged at the end of this task
(`git rev-parse HEAD` still `7ca40b93e3af416828aa8dcfd6ec1df01e006aa8`, no commit made).

**G1 was NOT launched by the Worker**, per the task's explicit instruction. The Planner
launches it after reading this report.

## Task received

Modify `p515_g_g1_g4_admm_gates.py` only (G1 path; keep g2/g4b/g3_init/g3_full working):
remove the harness-only ESSO-log-isolation workaround (production now handles it,
P5.15-F); give the `g1` CLI gate a fresh output root `data/SRP1/Results/P515G1/`
(refuse reuse); add a per-cycle heartbeat file; add per-period ESSO leak-mechanism
capture (pch/pdch/pnet/s_max, both IPOPT bound multipliers per leg, every constraint
dual referencing that period, identified generically via `identify_variables`) with a
first-solve pre-check that `ipopt_zL_out` is non-empty for `es_pch_per_unit`; add a
per-solve barrier-set/not-barrier-set/indeterminate leak classification with
predeclared thresholds; add network-failure capture and classification from the
(no-longer-discarded) stdout and the FrozenSMOPF directory; extend
`assert_g_capture_paths` (rule eleven) accordingly; run a <9-minute foreground smoke
test with `num_max_iters` overridden to 2 into a fresh `P515G1_smoke/` dir; report
evidence. No production file may change; no commit.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (all addenda, especially 3, 4, 5, 6)
- `p515_g_g1_g4_admm_gates.py` (the file modified)
- `shared_energy_storage_data.py` -- `SharedEnergyStorageData.__init__`/`optimize`,
  `_build_subproblem` (variable/constraint declarations, the barrier-identity comment),
  `_create_solver`, `_run_solver_attempt`, `_is_recoverable_shared_ess_failure`,
  `_optimize` (recovery path, `solver_recovery_diagnostics` / `complementarity_diagnostics_sink`
  population), `_parse_ipopt_barrier_terms`, `_get_esso_complementarity_diagnostics`,
  `_complementarity_ratio_for_model`, `_esso_cohort_pair_is_within_lifetime`,
  `_add_esso_cohort_constraint`
- `shared_resources_planning.py` -- `_run_operational_planning` (ADMM loop call sites for
  `create_shared_energy_storage_model` and `update_shared_energy_storages_coordination_model_and_solve`),
  those two functions themselves, `save_failed_tso_block` / `save_failed_dso_block` /
  `_save_frozen_network_block` / `_save_frozen_smopf_block`, `results_dir`/`logs_dir`
  construction
- `network.py` -- `_create_smopf_solver` (log naming/`file_append`), `_run_smopf`,
  `_print_network_failure_context`, `_is_recoverable_network_failure`
- `network_data.py` -- `NetworkData.optimize` (`failure_snapshot_callback` semantics)
- `p514_n_instrumented_cstar.py`, `p514_l_capacity_ladder.py`, `p56a_oracle.py`
  (`fresh_planning`, `WORK_DIR`), `p513_solve_profile_guard.py`
- `data/SRP1/case9/case9_params.json`, `data/SRP1/case33_2/case33_2_params.json`
  (network `output_file` stems, tol/acceptable settings)
- `.p515_g_gate.lock`, `P5_15_G1_G4_BLOCKED.md` (confirmed the hold's cause is already
  fixed at 7ca40b93; the lock itself left untouched, not deleted)

## Files modified

- `p515_g_g1_g4_admm_gates.py` (the only file changed; 648 insertions / 111 deletions,
  `git diff --stat` confirmed)

No other tracked file changed. `git status --short shared_energy_storage_data.py
shared_resources_planning.py network.py network_data.py definitions.py
model_construction_helpers.py` returns nothing.

## Changes made

1. **Removed the harness-only log workaround.** `unique_esso_logs()` and its use in
   `run_admm_arm` are deleted. Production's own per-solve, `logs_dir`-resolved ESSO log
   naming (`_create_solver`, P5.15-F) is relied on directly, confirmed working by the
   smoke test (see below).
2. **Stopped discarding stdout.** `redirect_stdout(io.StringIO())` is replaced by a
   `tee_stdout()` context manager that writes to both the real stdout (this is a
   foreground run) and a file, `stdout_<label>.log`, under the arm's output dir.
   Production's `[WARNING]`/`[INFO]` network- and ESSO-failure context lines land there
   and are the primary input to the network-failure scan.
3. **Output root for `g1`.** New constant `OUT_G1 = data/SRP1/Results/P515G1`.
   `_require_fresh_output_root()` is called before `run_admm_arm` for the `g1` CLI gate
   only (and, separately, by the smoke-test script for `P515G1_smoke`) -- "no overwrite,
   no reuse". `g2`/`g4b`/`g3_init`/`g3_full` continue to share the pre-existing `OUT`
   (`P515G`) exactly as before; NOT given the blanket freshness check, because they are
   separate process invocations that legitimately reuse that directory over time (see
   "Scope decision" below).
4. **Per-cycle heartbeat.** `<out>/heartbeat_<label>.json`, written atomically
   (temp file + `os.replace`) after every `update_shared_energy_storages_coordination_model_and_solve`
   call: `{cycle, utc_timestamp, wall_s, esso_solves_so_far, network_failures_so_far}`.
5. **Per-period ESSO leak-mechanism capture.** `<out>/esso_capture/<label>/node{id}_{init|cycleNNN}.jsonl`,
   one record per active (y_inv, y, d, p): `pch`, `pdch`, `pnet`, `s_max`, `zL_pch`,
   `zU_pch`, `zL_pdch`, `zU_pdch` (from `model.ipopt_zL_out`/`ipopt_zU_out`, `None` if a
   variable has no entry -- e.g. every `zU_*` here, since `pch`/`pdch` have no finite
   upper bound other than the converter circle), and every constraint dual whose row
   references that period's `pch`/`pdch` or the cohort's `es_D_per_unit`/`es_soh_per_unit_cumul`,
   identified GENERICALLY via `identify_variables` over
   `energy_storage_limits`/`energy_storage_operation_agg`/`energy_storage_cohort_pnet_share_h3`/
   `energy_storage_capacity_degradation`, cached once per node model (never hard-coded
   row indices). `r`/`class` per period use `mu_unscaled` parsed independently in the
   harness (see item 7). A **pre-check** (Addendum 6 item 4, first solve only) verifies
   `ipopt_zL_out` has at least one `es_pch_per_unit` entry and raises `RuntimeError`
   ("STOP") instead of substituting anything if it does not -- it did not fire (see
   Results).
6. **Per-solve leak classification.** `<out>/leak_classification_<label>.jsonl`, one
   line per (node, round): `mu_scaled`/`mu_unscaled`/`obj_scaled`/`obj_unscaled` (parsed
   independently in the harness -- production's own `_parse_ipopt_barrier_terms` returns
   only the scaled column and the ratio), the argmax period of `min(pch,pdch)/s_max`
   (reusing production's own `SED._complementarity_ratio_for_model`, not reimplemented),
   its `r = z_small*x_small/mu_unscaled` and class (predeclared thresholds: barrier-set
   `0.5<=r<=2`, not-barrier-set `r<0.1`, else indeterminate), its largest-magnitude
   captured dual, and class counts over all active periods of that solve.
7. **Network-failure capture and classification.** `<out>/network_failures_<label>.jsonl`,
   rebuilt (fresh, atomic) after every cycle and once more after the run: one line per
   failed (agent, network, year, day) block parsed from the tee'd stdout
   (`[WARNING] Network ... did not converge`, `[INFO] Retrying network solve...`,
   `[INFO] Network recovery solve succeeded...`, the final
   `[ERROR] Transmission network .../[WARNING] Distribution network node=...` lines),
   cross-tagged with the nearest preceding `ADMM Iteration N` line; plus one line per
   `sed.solver_recovery_diagnostics` entry (ESSO recovery events, tagged with cycle,
   `family: 'esso'` -- Addendum 6's "count ESSO ... the same way"); plus one line per
   FrozenSMOPF `.pkl` written during this run (`mtime >= run start`, metadata unpickled).
   Class in `{recovered, unrecovered, not_attempted}`. **Documented limitation** (in the
   code, `_last_exit_and_iterations`): network IPOPT logs stay cumulative across the
   WHOLE run (`file_append='yes'` in `network.py`, unaffected by the ESSO-only P5.15-F
   fix, and unchanged by this task per its no-production-edits constraint) and carry no
   per-cycle filename stamp, so the reported primary/recovery EXIT line and iteration
   count are a SNAPSHOT of that log's own last occurrence at scan time, not a guaranteed
   per-cycle isolate. The per-block `cycles_seen` list (from the stdout, which IS
   per-cycle) is the reliable per-cycle signal; the EXIT/iteration snapshot is
   corroborating detail.
8. **`assert_g_capture_paths` extended** (rule eleven): `planning.logs_dir`,
   `shared_ess_data.logs_dir`, `transmission_network.logs_dir` all set and absolute; the
   wrapper hooks installed; the output root verified fresh; `ipopt_zL_out`/`ipopt_zU_out`/
   `dual` present on a freshly-built, UNSOLVED probe subproblem (`SED._build_subproblem`
   called standalone, no `.solve()`, so the solve-profile guard's permitted-count
   identity is untouched by the probe).
9. **Labeling scope decision (not requested verbatim, judged necessary).** The task's
   examples name unlabeled files (`heartbeat.json`, `esso_capture/`). Because
   `run_admm_arm` is shared by `g2`/`g4b`/`g3_full`, which write into the SAME `OUT`
   directory across SEPARATE process invocations (confirmed already true of the
   pre-existing code: `g_{label}.json`, `esso_models_{label}.pkl`), unlabeled new
   filenames would collide across arms on the second sequential invocation and break
   "keep G2/G3-full/G4 arms working". All five new artifacts are therefore labeled
   per arm: `heartbeat_<label>.json`, `stdout_<label>.log`,
   `leak_classification_<label>.jsonl`, `network_failures_<label>.jsonl`,
   `esso_capture/<label>/...`. For the real `g1` run (label `control`, fresh root
   `P515G1`) this yields `heartbeat_control.json` etc. -- reported here so the Planner
   is not surprised by the exact filename.

## Commands / experiments run

1. `py_compile.compile('p515_g_g1_g4_admm_gates.py', doraise=True)` -- OK.
2. `import p515_g_g1_g4_admm_gates` (module import only, `__main__` never executed,
   lock never touched by this task) -- OK.
3. Preflight (no solve): built a `fresh_planning('preflight_g1prep')` planning object,
   confirmed `assert_g_capture_paths` raises with "hooks not installed" before entering
   `esso_capture_hooks`, and passes once inside it.
4. **Smoke test** (foreground, `python -u smoke_g1prep.py`, wall time 141.6 s, well
   under the 9-minute budget): imported `p515_g_g1_g4_admm_gates` and called
   `run_admm_arm('smoke', data/SRP1/Results/P515G1_smoke, k_override=None,
   num_max_iters_override=2)` directly -- the C\* control configuration
   (`N.S_INV=0.96875`, `N.E_INV=3.875`, `N.INVEST_YEAR=2025`, `N.BUDGET=5.0e6`,
   `N.REL=1e-4`, `N.RHO={'v':1.5,'pf':300.0,'ess':1.0}`) otherwise unchanged, only the
   iteration cap overridden via the new smoke-only kwarg (the `g1` CLI arm never passes
   it, so `N.CAP=90` stands for the real campaign). `p515_g_g1_g4_admm_gates.py` was
   never run via its own `__main__`; the lock file was never touched.
5. Post-run inspection: `du -sh`/per-file sizes of `esso_capture/smoke`; content dumps
   of `leak_classification_smoke.jsonl`, `heartbeat_smoke.json`,
   `network_failures_smoke.jsonl`, one `esso_capture` record; `grep` of
   `stdout_smoke.log` for the captured network-failure print trail; listing of
   `data/SRP1/Results/P56A/evals/p515g_smoke/logs/`; `git status`/`git diff --stat`/
   `git rev-parse HEAD` to confirm production and HEAD unchanged.

## Results

**Smoke run summary** (from `g_smoke.json` and stdout):
`recourse=1,832,134,032.52`, `cycles=2`, `local_solve_failures=0`,
`wall_clock_s=141`, EFC/day max `1.098` (threshold `1.4612`),
`esso_complementarity_diagnostics_by_round.grouping_clean = True` (observed 9 / expected
9 = 3 nodes x 3 rounds [init, cycle 1, cycle 2]). `solve_profile.identity_holds = False`
(154 permitted solves vs. the pre-existing `51*rows+51=153` formula) -- **expected, not
a defect**: cycle 1 had a real recoverable TSO failure (see below), adding one solve to
that cycle's 51.

**Required smoke evidence:**

- `heartbeat_smoke.json` at end: `{"cycle": 2, "utc_timestamp": "...T10:03:00...",
  "wall_s": 140.57, "esso_solves_so_far": 9, "network_failures_so_far": 1}` -- shows
  cycle 2, as required.
- `esso_capture/smoke/` has exactly the 9 required files: `node{5,7,9}_{init,cycle001,cycle002}.jsonl`,
  288 records each (3 years x 4 days x 24 periods, single active cohort (0,0) at this
  instance). Checked on `node5_cycle001.jsonl`: **288/288 records have non-null
  `zL_pch`**, **288/288 have non-null `zL_pdch`**, **288/288 have >=1 captured dual**
  (`zU_pch`/`zU_pdch` are null throughout -- `pch`/`pdch` have no explicit finite upper
  bound other than the converter circle, so IPOPT reports no upper-bound multiplier;
  this is the documented "record null" case, not a defect). The Addendum-6 first-solve
  pre-check ("STOP if `ipopt_zL_out` is empty for `es_pch_per_unit`") did **not** fire.
- `leak_classification_smoke.jsonl` has **exactly 9 lines** (one per node per round).
  See the argmax table below.
- `network_failures_smoke.jsonl` **exists and is non-empty** (1 line): a REAL, genuinely
  recovered TSO failure was captured end-to-end during the smoke run (case9, 2035,
  Winter, cycle 1) -- primary `maxIterations`, cold retry succeeded
  (`Optimal Solution Found.`, 37 iterations), classified `recovered`. `stdout_smoke.log`
  contains the exact production print trail:
  `[WARNING] Network primary solve did not converge for case9, year=2035, day=Winter: ...`,
  `[INFO] Retrying network solve once for case9, year=2035, day=Winter, cold start, with
  acceptable_iter=1, acceptable_tol=0.0001, warm_start_init_point=no.`,
  `[INFO] Network recovery solve succeeded for case9, year=2035, day=Winter.` No
  FrozenSMOPF snapshot was written for this block (correct: the callback only fires on
  a FINAL unsuccessful result, and this one recovered) -- `n_frozen_snapshots=0`,
  confirmed against `git status`/`ls -la data/SRP1/Results/FrozenSMOPF/` showing the
  directory's only two pre-existing files, both from 2026-09-13, untouched.
- ESSO per-solve logs came from production's `logs_dir`
  (`data/SRP1/Results/P56A/evals/p515g_smoke/logs/`), listed: `optim_log_esso_node{5,7,9}_{init,cycle001,cycle002}.txt`,
  9 distinct files, distinct sizes (4.2-8.1 KB), confirming one fresh log per solve (no
  accumulation) -- this is what makes the harness-only workaround obsolete.
- Guard: `permitted_solve=154`, `blocked_solve=0`. Wall time 141.6 s.

**Argmax classification table** (`leak_classification_smoke.jsonl`, verbatim):

| node | round | mu_unscaled | class_counts | argmax (y_inv,y,d,p) | pch | pdch | r | class | largest dual (component, value) |
|---|---|---|---|---|---|---|---|---|---|
| 5 | init | 3.1316e-08 | {barrier-set: 288} | (0,1,1,2) | 2.4854e-05 | 2.5241e-05 | 0.798 | barrier-set | energy_storage_operation_agg, 5.524e-06 |
| 7 | init | 3.1527e-08 | {barrier-set: 288} | (0,0,3,9) | 2.5031e-05 | 2.5065e-05 | 0.794 | barrier-set | energy_storage_capacity_degradation, -2.303e-06 |
| 9 | init | 3.1275e-08 | {barrier-set: 288} | (0,1,1,2) | 2.4855e-05 | 2.5243e-05 | 0.800 | barrier-set | energy_storage_operation_agg, 6.836e-06 |
| 5 | 001 | 2.9927e-08 | {barrier-set: 288} | (0,0,3,18) | 2.5046e-05 | 2.5051e-05 | 0.837 | barrier-set | energy_storage_capacity_degradation, -2.297e-07 |
| 7 | 001 | 3.0710e-08 | {barrier-set: 288} | (0,0,3,18) | 2.5045e-05 | 2.5052e-05 | 0.816 | barrier-set | energy_storage_capacity_degradation, -2.292e-07 |
| 9 | 001 | 2.9795e-08 | {barrier-set: 288} | (0,0,3,18) | 2.5052e-05 | 2.5045e-05 | 0.841 | barrier-set | energy_storage_capacity_degradation, -2.331e-07 |
| 5 | 002 | 4.2524e-08 | {barrier-set: 288} | (0,0,2,21) | 2.5042e-05 | 2.5055e-05 | 0.589 | barrier-set | energy_storage_capacity_degradation, -2.674e-07 |
| 7 | 002 | 4.3232e-08 | {barrier-set: 288} | (0,0,3,8) | 2.5031e-05 | 2.5066e-05 | 0.579 | barrier-set | energy_storage_operation_agg, 6.780e-07 |
| 9 | 002 | 4.2496e-08 | {barrier-set: 288} | (0,0,2,21) | 2.5048e-05 | 2.5049e-05 | 0.589 | barrier-set | energy_storage_capacity_degradation, -2.712e-07 |

All 9 solves classify **every** active period as `barrier-set` (`r` in [0.58, 0.84], all
inside the predeclared [0.5, 2] band). **This is a different finding from Addendum 6's
"not barrier-set" at C\*** -- expected, since the smoke instance is the CONTROL
configuration (S=0.96875 MVA, E=3.875 MWh uniform) at `tol=1e-8` remedy (h), not the
`k=11,541.56` C\* configuration Addendum 6 characterized. This smoke evidence is offered
only as proof the classification mechanism works, not as a finding about C\*.

**Disk footprint** (measured, for sizing a 90-cycle run):
`esso_capture/smoke/` = 3.0 MB for 9 files (3 nodes x 3 rounds), i.e. **~340 KB per
node per round**, **~1.02 MB per round** (3 nodes), **~93 MB for a full 91-round (init +
90 cycles) G1 campaign**. `leak_classification_smoke.jsonl` = 8.05 KB for 9 lines
(~895 B/line) -> ~73 KB for 91 rounds x 9 lines. `stdout_smoke.log` = 26.8 KB for 3
rounds (grows with recourse-jump diagnostic verbosity, which is state-dependent, not
purely linear) -> order 1-3 MB for 90 cycles. `g_smoke.json` = 26.5 KB for 2 cycles
(embeds the full cycle trajectory) -> order 1-1.5 MB for 90 cycles.
`esso_models_smoke.pkl` = 2.76 MB, roughly constant (final-state snapshot, not
cycle-dependent). **Total new-artifact footprint for a full G1 run: order 100 MB**,
dominated by `esso_capture/`.

## Validation

- Code executes correctly: smoke run completed with no exceptions, `local_solve_failures=0`.
- Requested diagnostics work: heartbeat, per-period ESSO capture (zL non-null, duals
  present, generic identification confirmed by a genuinely-inactive H3 row correctly
  showing `dual: null` for node 5 -- only one cohort active at this instance, so H3 rows
  are deactivated, and `model.dual.get(con)` correctly returns `None` for a deactivated
  row never solved this round), leak classification, and network-failure capture (a REAL
  recovered TSO failure, not a synthetic one) all produced the exact artifacts and
  content the task specified.
- Underlying numerical question (is C\*'s leak barrier-set): **not addressed by this
  task** -- this was a harness-preparation task; the smoke instance is the control
  configuration, not C\*, precisely to avoid pre-empting the real G1 run.
- No production file changed (`git status`/`git diff --stat` empty for all six named
  files). `git rev-parse HEAD` unchanged (`7ca40b93...`). No commit made. `.p515_g_gate.lock`
  untouched (`ls -la` timestamp still `Sep 13 22:58`, content unchanged).
- `data/SRP1/Results/P515G1/` does not exist -- confirmed the exact Planner command
  would pass `_require_fresh_output_root(OUT_G1)`.

## Unexpected findings

1. **The real `g1` run will NOT write into an entirely clean tree**, despite `P515G1/`
   itself being fresh. `data/SRP1/Results/P56A/evals/p515g_control/` already holds
   **~300 MB of residue** from the earlier killed G1 attempt (network IPOPT logs only,
   no ESSO logs -- consistent with a run that died before P5.15-F's ESSO log fix
   existed / before reaching a later cycle), dated 2026-09-13. `p56a_oracle.fresh_planning`
   assigns `planning.logs_dir` from the `eval_id` (`f'p515g_{label}'`, unchanged from
   before this task) and does not clear a pre-existing eval directory. Network IPOPT
   logs use `file_append='yes'` unconditionally (`network.py`, out of this task's scope
   to change), so a real `g1` run today would **append** its own iteration output onto
   this 300 MB of stale content in the SAME per-(case,year,day) log files, inflating
   them further and mixing two campaigns' output in one file. This does not corrupt
   THIS harness's own new artifacts (which live under `P515G1/`, fresh, and the
   per-solve `mu_final`/`s_obj` parse already takes the LAST occurrence in each file,
   so it is not fooled by the stale prefix) but it does affect: (a) log file sizes for
   the real campaign, (b) anyone reading those network logs directly expecting a clean
   run, (c) marginally the `_last_exit_and_iterations` snapshot in
   `network_failures_control.jsonl` if a genuinely-new failure's EXIT line happened to
   coincide... it will not, since `finditer`/`[-1]` always takes the newest block, but
   the file will be larger and slower to parse. **Not fixed** (destructive `data/`
   change, out of this task's authorization). Recommend the Planner either clear/rename
   `data/SRP1/Results/P56A/evals/p515g_control/` before launching `g1`, or explicitly
   accept the append. Same applies to `p515g_k10000`, `p515g_control_rep2`,
   `p515g_g3_full_node7` residue if those gates are re-run.
2. `solve_profile.identity_holds = False` on the smoke run is expected (one recoverable
   TSO failure added a solve beyond the pre-existing `51*rows+51` formula) but is worth
   flagging: the real G1 campaign will likely show the same non-identity on any cycle
   with a recovery, which is a FEATURE of this task's capture (the recovery is now
   visible in `network_failures_control.jsonl`), not a regression in the guard.
3. FrozenSMOPF (`data/SRP1/Results/FrozenSMOPF/`) is a single GLOBAL directory shared by
   every network agent/run (`transmission_network.results_dir` /
   `distribution_network.results_dir` both resolve to the one `planning_problem.results_dir`,
   unchanged production behaviour). The harness's frozen-snapshot scan filters by
   `mtime >= run start` to avoid attributing old residue to the current run; this was
   exercised (0 matches on the smoke run, correctly) but was NOT exercised against an
   actual NEW snapshot (the smoke run's one failure recovered, so no snapshot was
   written) -- flagged as not fully verified below.

## Remaining issues

- The FrozenSMOPF-scan path (`_scan_frozen_snapshots`, including the pickle-metadata
  read) was exercised for the "no snapshot present" case only; it has not been
  exercised against a genuine unrecovered failure that DOES write a snapshot. The
  metadata-extraction logic mirrors `_save_frozen_network_block`'s/`_save_frozen_smopf_block`'s
  own payload shape exactly (`payload['metadata']`), so I have high confidence it is
  correct, but this is not the same as having observed it fire.
- The network-failure EXIT/iteration snapshot is, by construction (documented in the
  code and above), a snapshot of the CUMULATIVE log's last occurrence, not a
  cycle-isolated read -- acceptable for this task ("the class is what matters"; item 6's
  own text acknowledges production only records recovery "in prints and per-attempt log
  files"), but the Planner should not read `primary_iterations`/`recovery_iterations`
  in `network_failures_*.jsonl` as guaranteed to belong to the LISTED cycle if the same
  (case,year,day) block failed on more than one cycle in the same run (the smoke run
  only exercised the single-failure case).
- Per-arm labeling (`heartbeat_<label>.json` etc., see "Changes made" item 9) is a
  necessary adaptation of the task's literal unlabeled examples; flagged for Planner
  awareness rather than treated as silently equivalent.
- Disk-footprint projection (~93 MB for `esso_capture/` over 91 rounds) is a linear
  extrapolation from one 2-cycle instance at the control candidate; the real campaign's
  active-cohort count could differ across cycles as the SoH floor deactivates cohorts
  over the horizon, which would only SHRINK the per-round size (fewer active
  cohort-periods), so this is a conservative (upper-bound-leaning) estimate, not a
  measured 90-cycle number.

## Questions for Planner

1. Clear/rename the `p515g_control` (and other `p515g_*`) residue under
   `data/SRP1/Results/P56A/evals/` before launching `g1`, or accept the log-append onto
   300 MB of stale content? I did not touch it (destructive `data/` change, out of this
   task's authorization).
2. Confirm the per-arm labeling scheme (`heartbeat_control.json` rather than
   `heartbeat.json`) is acceptable, since the brief's examples were unlabeled.

## Exact command confirmation

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_g_g1_g4_admm_gates.py g1 > data/SRP1/Results/P515G1_launch.log 2>&1
```

Once the Planner removes `.p515_g_gate.lock`, this command will: acquire the exclusive
run lock; call `_require_fresh_output_root(OUT_G1)` (currently passes --
`data/SRP1/Results/P515G1/` does not exist); then `run_admm_arm('control', OUT_G1)`
with `N.CAP=90` (the smoke override is never reached by this call path). Every NEW
harness artifact this task adds (`heartbeat_control.json`, `stdout_control.log`,
`leak_classification_control.jsonl`, `network_failures_control.jsonl`,
`esso_capture/control/...`, plus the pre-existing `g_control.json` and
`esso_models_control.pkl`) is written under `data/SRP1/Results/P515G1/`, which is fresh.
**Two production-controlled paths are NOT under `P515G1/`** and are outside this
harness's control (unchanged from the pre-existing code, which already called
`O.fresh_planning(f'p515g_{label}')`): the per-solve IPOPT logs themselves, under
`data/SRP1/Results/P56A/evals/p515g_control/logs/` (see Unexpected Finding 1 for why
that directory is not currently clean), and any FrozenSMOPF failure snapshot, under the
global, pre-existing `data/SRP1/Results/FrozenSMOPF/`.

## Key file paths

- Modified: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_g_g1_g4_admm_gates.py`
- Smoke output: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515G1_smoke/`
  (`g_smoke.json`, `heartbeat_smoke.json`, `leak_classification_smoke.jsonl`,
  `network_failures_smoke.jsonl`, `stdout_smoke.log`, `esso_capture/smoke/`,
  `esso_models_smoke.pkl`)
- Smoke ESSO/network per-solve logs: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P56A/evals/p515g_smoke/logs/`
- Residue flagged in Unexpected Finding 1: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P56A/evals/p515g_control/`
- Fresh, not-yet-created G1 root: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515G1/`
