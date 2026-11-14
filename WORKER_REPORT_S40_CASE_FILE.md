# Worker Report -- P5.15 Addendum 22 item (3): case file carries the oracle (arm s39_D)

## Task received

Write the oracle (arm `s39_D`) configuration into `data/SRP1/SRP1_params.json`, closing
Track B, and prove the case file alone reproduces the oracle. Authority:
`PLANNER_BRIEF_2026-09-13.md` Addendum 22 item (3); frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addendum 22, item `3_case_file`, `oracle.configuration`,
  `predecessor`, `planner_risk_notes_not_predictions`)
- `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`
- `p515_g_g1_g4_admm_gates.py` -- `S39_ARMS`, `_s39_configure_hook`, `run_s39_arm`,
  `_assert_s39_overrides_in_force`, `assert_s39_capture_paths`, `assert_s31c_capture_paths`,
  `write_boyd_terminal_s35ref`, `s38_pf_capture_hooks`, `s39_exempt_until_capture_hooks`,
  `run_admm_arm`, `_construct_arm_planning`, and (for item 3's superseded-stage search)
  `assert_s33e2_capture_paths`, `assert_s34_capture_paths`, `assert_s35ref_capture_paths`,
  `assert_s35pt_capture_paths`, `assert_s35ref_replay_capture_paths`,
  `assert_s37_capture_paths`, `assert_s38_capture_paths`, and their `__main__`/`run_*_arm`
  call sites
- `admm_parameters.py` (`ADMMParameters.__init__`, `_read_parameters_from_file`) -- confirmed
  every s39_D override key is expressible as a case-file key
- `data/SRP1/SRP1_params.json` (the file edited)
- `p59_rho.py` (`apply_rho_to_params` -- confirmed it applies to every network key already
  present in `admm.rho[group]`, including `esso` under `ess`)
- `p56a_oracle.py` (`fresh_planning`, `load_baseline` -- confirmed the production loader path,
  zero solves)
- `p513_solve_profile_guard.py` (`SolveProfileGuard`)
- `p515_s40_clone_capture_preflight.py` (comparator conventions reused by import: `_diff`,
  `_load_json`, `_load_jsonl`, `_sha256_file`, `EXCLUDE_KEY_NAMES`, `EXCLUDE_DOTTED_SUFFIXES`,
  `INTENTIONAL_DIFF_SUFFIXES`, `ARTIFACT_FILES`, `SIDECAR_JSONL_FILES`,
  `PRODUCTION_FILES_TO_CHECK_CLEAN`, `FORBIDDEN_LIVE_PROCESS_SUBSTRINGS`, `_ancestor_pids`)
- `data/SRP1/Results/P515S40/clone_preflight_v2/lightweight/` (the committed two-cycle oracle
  run compared against)
- For item 3's search: `p515_s33e2_zero_solve_checks.py`, `p515_s34_zero_solve_checks.py`,
  `p515_s35ref_preflight.py`, `p515_s35pt_preflight.py`, `p515_s37_zero_solve_checks.py`,
  `p515_s38_zero_solve_checks.py`, `p515_s38_balancing_replay.py`, `p515_s39_zero_solve_checks.py`,
  `p515_s35pt_phase1_checks.py`, `p515_s35pt_phase2_checks.py`

## Files modified

- `data/SRP1/SRP1_params.json` (production case file -- the single authorized production edit)

## Files created

- `p515_s40_case_file_oracle_checks.py` (zero-solve load check, item 1 of validation)
- `p515_s40_case_file_repro.py` (two-cycle case-file-alone reproduction, item 2 of validation)
- `data/SRP1/Results/P515S40/case_file_oracle_load_check.json` (result of item 1)
- `data/SRP1/Results/P515S40/case_file_repro/` (result of item 2: `g_s39_D.json`,
  `boyd_terminal.json`, `component_levels_terminal.json`,
  `interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json`, sidecars,
  `case_file_repro_results.json`, `manifest_sha256.json`)
- `data/SRP1/Results/P515S40/case_file_repro_launch.log` (attached run log, both streams)
- `WORKER_REPORT_S40_CASE_FILE.md` (this report)

## Changes made

### The case-file diff (`data/SRP1/SRP1_params.json`, `admm` block)

| key | old | new | verified against |
|---|---|---|---|
| `num_max_iters` | `25` | `300` | `S39_CAP` / spec v11 `oracle.configuration.cap` |
| `minimum_consecutive_converged_cycles` | `3` | `10` | `S39_REQUIRED_CONSECUTIVE_CYCLES` |
| `shared_ess_initialization` | `"price_taker"` | `"standalone"` | `S39_ARMS`/hook sets `'standalone'` |
| `penalty_update.freeze_backstop_cycle` | `60` | `200` | `S39_FREEZE_BACKSTOP_CYCLE` |
| `penalty_update.balancing_exempt_until` | absent (default `{}`) | `{"ess": {"dual_ratio_below": 1.0, "consecutive_cycles": 5}}` | `S39_ARMS['s39_D']['exempt_until']` |
| `rho.ess.{case9,case33_1,case33_2,case33_3,esso}` | `0.1125` | `0.01` | `S39_RHO_ESS`, applied via `RH.apply_rho_to_params` to every key already present in `rho['ess']` (confirmed `esso` is already a key) |
| `proximal_regularization.tso.tau` | `1.0` | `0.0` | `S39_TAU` |

Checked and confirmed **unchanged** (already matched D's configuration, so left untouched):
`tol.boyd.eps_abs` (1e-5), `tol.boyd.eps_rel` (1e-4), `rho.v.*` (0.0077 on every network),
`rho.pf.*` (0.198 on every network), `penalty_update.freeze_after_unchanged_cycles` (10),
`objective_scale` (93635360.0), `shared_ess_reference_rating_mva` (2.5), `esso_al_scale`
(`"sigma_over_median_block_weight"`), `adaptive_penalty` (`true`),
`proximal_regularization.tso.gamma_policy` (`"tied_to_rho"`), `penalty_update.
balancing_exempt_channels` (absent -> default `[]`, identical to D's own explicit `[]`),
`shared_ess_normalization_floor_mva` (0.10), the `dso` proximal block, `previous_iteration`,
`rho_previous_iter` (inert -- only loaded if `previous_iteration.ess.{tso,dso}` is `true`,
which it is not). `benders.num_max_iters` (line 6, a *different*, unrelated Benders-loop
setting) was **not** touched. `persistent_workers` (default off) and
`tso_snapshot_capture_mode` (default `'lightweight'`) are not case-file keys at all
(`admm_parameters.py` `__init__`) -- both stay at their code defaults, per the task.

Diff is minimal (13 insertions / 10 deletions), preserving the file's tab indentation; no
other key was reformatted.

Committed **alone**: `git commit --only -- data/SRP1/SRP1_params.json` (commit `fb3de341`),
per the frozen spec's own risk note that this is the first edit to this case file in the
campaign.

## Commands / experiments run

1. `git diff -- data/SRP1/SRP1_params.json` -- inspected the diff before committing.
2. `python3 -c "import json; json.load(...)"` -- JSON well-formedness check.
3. `p515_s40_case_file_oracle_checks.py` (item 1, zero-solve):
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s40_case_file_oracle_checks.py
   ```
4. `git commit --only -- data/SRP1/SRP1_params.json ...` (commit `fb3de341`).
5. `git add -- p515_s40_case_file_oracle_checks.py p515_s40_case_file_repro.py && git commit ...`
   (commit `ee976eff`).
6. Preconditions verified by hand before the repro run: `.p515_g_gate.lock` absent; `ps aux`
   scanned for `p515_g_g1_g4_admm_gates.py` / `p515_s39_` / `p515_s40_` / timing processes
   (none found); `git status --porcelain -- shared_resources_planning.py network.py
   network_data.py shared_energy_storage_data.py admm_parameters.py
   p515_g_g1_g4_admm_gates.py data/SRP1/SRP1_params.json` clean (after the case-file commit).
7. `p515_s40_case_file_repro.py` (item 2, two-cycle case-file-alone reproduction), run
   attached, alone, both streams captured:
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s40_case_file_repro.py \
       > data/SRP1/Results/P515S40/case_file_repro_launch.log 2>&1
   ```
   Exit code 1 (the script's own `sys.exit(1)` on `bitwise_identical=False` -- see Results).
8. Item 3 (zero-solve): repo-wide `grep` for every rule-eleven checklist key and inline
   assertion that reads `minimum_consecutive_converged_cycles`, `freeze_backstop_cycle`,
   `rho.ess`/`0.1125`, `tau`/`gamma_tau`, or `shared_ess_initialization`/`price_taker`
   directly from a freshly-loaded (non-overridden, or partially-overridden) case-file object,
   across `p515_g_g1_g4_admm_gates.py` and every `p515_*_zero_solve_checks.py` /
   `p515_*_preflight.py` / `p515_*_balancing_replay.py` script in the repo root, read to
   confirm whether the field checked is overridden before the assertion or read straight from
   the case file.

## Results

### Item 1 -- zero-solve load check (`case_file_oracle_load_check.json`)

`oracle_relevant_fields_match = True`. `scalar_diffs = {}`, `dict_diffs = {}` -- every
scalar and dict-valued field on `ADMMParameters` (`num_max_iters`,
`minimum_consecutive_converged_cycles`, `shared_ess_normalization_floor_mva`,
`adaptive_penalty`, `objective_scale`, `objective_scale_assert_factor`,
`shared_ess_reference_rating_mva`, `shared_ess_initialization`, `boyd_eps_source`, `tol`,
`rho`, `penalty_update`, `proximal_regularization`, `esso_al_scale`) loaded from the edited
case file matches, exactly, the same fields on a fresh planning object with
`_s39_configure_hook('s39_D')` applied. Guard: `SolveProfileGuard(permitted=())`,
observed `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`
-- zero solves on either side, confirmed by the armed guard, not asserted.

The only differences are the `_source` provenance bookkeeping fields (expected, reported
separately, not gated): `shared_ess_initialization_source` (`'case_file'` vs
`'p515s39_override_not_case_file'`), `balancing_exempt_channels_source` (`'default'` vs the
override string -- both reduce to the *same value* `[]`), `balancing_exempt_until_source`
(`'case_file'` vs the override string). `objective_scale_source` and
`shared_ess_reference_rating_source` are identical (`'case_file'` both sides, unaffected by
this task).

**No field the override path sets is inexpressible in the case file** -- every key
`_s39_configure_hook` touches has a corresponding case-file key per `admm_parameters.py`'s
`_read_parameters_from_file`.

### Item 2 -- two-cycle case-file-alone reproduction (`case_file_repro/case_file_repro_results.json`)

`bitwise_identical = False`, `total_n_diffs = 6`, `all_artifacts_present = True`.
Recourse and gross operational cost at both cycles are **bitwise identical**
(`740742829.5748339` both sides at cycle 2); `boyd_terminal.json`,
`component_levels_terminal.json`, `interface_settlement_detail_s31c.json`,
`interface_voltage_terminal.json`, `ess_entry_stride_baseline.jsonl`,
`soh_floor_sidecar_baseline.jsonl`, `pf_entry_stride_s39_D.jsonl`,
`ess_exempt_until_state_s39_D.jsonl` are **all 0-diff**. `solve_profile.identity_holds =
True` (153/153 solves), `local_solve_failures = 0`, zero network failures.

All 6 diffs are fully diagnosed, none rationalized away, comparator not adjusted:

1-2. `report.rule_eleven_checklist.s39_pre_solve_override_verification` (a dict of 14
   `True` values) and `report.rule_eleven_checklist.s40_tso_snapshot_capture_mode_override`
   (`True`) exist in the oracle report but are `<MISSING>` in the case-file-alone report.
   **Expected by construction**: both keys are written *by the override hooks themselves*
   (`_s39_configure_hook`'s own `_assert_s39_overrides_in_force` call, and
   `p515_s40_clone_capture_preflight.py`'s `_capture_mode_hook_override` wrapper around it)
   -- since this run deliberately calls neither hook (`pre_solve_hook=None`), there is
   nothing to write these keys. They are not numeric artifacts; they are provenance markers
   of *which code path ran*, and their absence is exactly what "case file alone, no
   configure hook" means.

3-4. `g_s39_D.json.rule_eleven_checklist.s39_pre_solve_override_verification` /
   `.s40_tso_snapshot_capture_mode_override` -- the same two keys, doubled by the report/
   g-json double compare (as `p515_s40_clone_capture_preflight.py`'s own docstring already
   notes for a different field).

5-6. `recourse_jump_sidecar_baseline.jsonl[1].objective_component_block_deltas[8].component`
   / `[9].component`: oracle has `generation_cost` at index 8 and `economic_market_cost` at
   index 9; the case-file-alone run has them **swapped**. Verified directly (see Validation
   below): the **values** at both indices are bitwise identical
   (`-85382653.71541198` both sides, under both labels) -- only the *order* of two tied
   entries differs. Root cause, located in source: `p515_g_g1_g4_admm_gates.py` line ~2830,
   `for key in set(flat_current) | set(flat_previous):` -- Python `set` iteration order for
   string-keyed sets depends on the process's hash seed
   (`PYTHONHASHSEED`, randomized per-process by default), and the subsequent
   `obj_deltas.sort(key=lambda e: e['abs_delta'], reverse=True)` is a **stable** sort, so two
   entries with *exactly equal* `abs_delta` keep whatever relative order the set handed them
   -- which can differ between two separate Python process invocations. This is a
   **pre-existing, general non-determinism in production capture code**, orthogonal to the
   case-file change: it would occur between any two separate process runs of the identical
   override-hook oracle arm too (e.g. between `clone_preflight` and `clone_preflight_v2`,
   both `run_s39_arm` invocations). It is reported, not fixed (out of this task's scope: "no
   production code edits").

Per instruction, I am **stopping here and reporting**, not declaring the gate passed or
adjusting the comparator. The strict verdict is `bitwise_identical = False`. Every
*numeric* artifact and every numeric value in every artifact matches exactly; the 6 diffs
are (a) two provenance-checklist keys, doubled, that exist only because of which hook ran,
and (b) one non-deterministic list-order tie between two numerically-identical entries in a
top-10 diagnostic list. The Planner should judge whether this satisfies "bitwise identity on
every numeric artifact."

### Item 3 -- superseded rule-eleven checklists (zero-solve, grep + read, not executed)

Search scope, stated explicitly (rule seven): grepped every `.py` file in the repository
root for direct references to `minimum_consecutive_converged_cycles`, `freeze_backstop_cycle`,
`0.1125` / `rho_ess`, `tau` / `gamma_tau`, and `price_taker`, then read each hit's enclosing
function to determine whether the checked field is (a) explicitly overridden by that
stage's own code before the assertion (unaffected by the case-file edit) or (b) read
straight from a freshly-loaded case-file object (now broken). This is a source-grep search,
not an execution trace or AST analysis; a check phrased without one of these literal tokens
could exist and would not have been found.

**Newly superseded by this edit** (would now raise/report `False` if re-launched; were
`True`/passing against the case file before this commit):

| file:line | function / stage | checklist key | old (asserted) | new (case file) |
|---|---|---|---|---|
| `p515_g_g1_g4_admm_gates.py:2300` | `assert_s33e2_capture_paths` (gate `s33e2`, `__main__` ~6474) | `gamma_tau_is_1` | `tau == 1.0` | `0.0` |
| `p515_g_g1_g4_admm_gates.py:2302` | same | `minimum_consecutive_converged_cycles_is_3` | `== 3` | `10` |
| `p515_g_g1_g4_admm_gates.py:2706` | `assert_s34_capture_paths` (gate `s34`, `__main__` ~6525) | `freeze_backstop_cycle_is_60` | `== 60` | `200` |
| `p515_g_g1_g4_admm_gates.py:2707` | same | `minimum_consecutive_converged_cycles_is_3` | `== 3` | `10` |
| `p515_g_g1_g4_admm_gates.py:~3319` | `assert_s35ref_capture_paths` (gate `s35ref` ~6598; reused by `assert_s35pt_capture_paths` at 3803 and `assert_s35ref_replay_capture_paths` at 4367) | `initial_rho_v_pf_ess_matches_spec_v5` | `rho.ess == 0.1125` | `0.01` |
| `p515_g_g1_g4_admm_gates.py:3330` | same (+ s35pt, s35ref_replay) | `freeze_backstop_cycle_is_60` | `== 60` | `200` |
| `p515_g_g1_g4_admm_gates.py:3331` | same (+ s35pt, s35ref_replay) | `minimum_consecutive_converged_cycles_is_3` | `== 3` | `10` |
| `p515_g_g1_g4_admm_gates.py:3816` | `assert_s35pt_capture_paths` (gate `s35pt`, `__main__` ~6678) | `shared_ess_initialization_is_price_taker` | `== 'price_taker'` | `'standalone'` (also a **behavioral** supersession: the s35pt arm's whole point -- exercising `_initialize_shared_ess_from_price_taker` -- would silently not fire) |
| `p515_g_g1_g4_admm_gates.py:4687` | `assert_s37_capture_paths` (`run_s37_arm`, both arms, ~4895) | `freeze_backstop_cycle_is_60` | `== 60` | `200` |
| `p515_g_g1_g4_admm_gates.py:4688` | same | `minimum_consecutive_converged_cycles_is_3` | `== 3` | `10` |
| `p515_g_g1_g4_admm_gates.py:5301` | `assert_s38_capture_paths` (`run_s38_arm`, both arms, ~5574) | `minimum_consecutive_converged_cycles_is_3` | `== 3` | `10` |
| `p515_s34_zero_solve_checks.py:795` | `srp1_ok` (reads case file directly via `ADMMParameters().read_parameters_from_file`) | `srp1_v4['freeze_backstop_cycle'] == 60` | `60` | `200` |
| `p515_s33e2_zero_solve_checks.py:769` | `srp1_ok` (same pattern) | `srp1_e2['tau'] == 1.0 and srp1_e2['minimum_consecutive_converged_cycles'] == 3` | `1.0` / `3` | `0.0` / `10` |
| `p515_s37_zero_solve_checks.py:279-280` | inline `assert` in the synthetic-replay harness (`_build_admm_ready_state`, documented at lines 273-278 as "SRP1's own case file ... sets `freeze_backstop_cycle=60` -- deliberately LEFT ACTIVE here") | hard `assert ... == 60` (both `admm_params_exempt` and `admm_params_plain`) | `60` | `200` -- **raises `AssertionError`**, not a soft checklist |
| `p515_s35ref_preflight.py:157-158,290,320` | `rho_ess_ok`, feeds `preflight_passed` | `rho.ess == 0.1125` (`G.S35REF_INITIAL_RHO_ESS`) | `0.1125` | `0.01` |
| `p515_s35pt_preflight.py:127-128,321,339` | `rho_ess_ok`, feeds `preflight_passed` | `rho.ess == 0.1125` (`G.S35PT_INITIAL_RHO_ESS`) | `0.1125` | `0.01` |

**Already broken before this edit** (pre-existing, unrelated to this task, noted for
completeness so it is not misattributed): `assert_s34_capture_paths`'s
`initial_rho_v_pf_ess_matches_spec_v4` (`p515_g_g1_g4_admm_gates.py:2699`, expects
`rho.ess == 0.05`; the case file already carried `0.1125` before this task started);
`assert_s33e2_capture_paths`'s rho-all-`1.0` check (~2305-2310, case file was already
0.0077/0.198/0.1125, never `1.0`); `p515_s34_zero_solve_checks.py`'s
`all(v == 0.05 for v in srp1_v4['rho_ess'])` (same pre-existing mismatch, read from disk).

**Confirmed unaffected** (explicitly override the field in their own code before checking,
so the case-file default no longer reaches the assertion): `assert_s37_capture_paths` /
`assert_s38_capture_paths` / `assert_s39_capture_paths` rho checks (each arm fixes its own
rho via `RH.apply_rho_to_params` in its precheck); `assert_s38_capture_paths`'s
`shared_ess_initialization` check (each s38 arm forces `'standalone'`);
`p515_s38_zero_solve_checks.py`, `p515_s38_balancing_replay.py`, `p515_s39_zero_solve_checks.py`
(all three explicitly set `freeze_backstop_cycle = 200` / `minimum_consecutive_converged_cycles
= 10` in their own setup, already anticipating the new default); `assert_s39_capture_paths`
itself (its precheck applies the full `_s39_configure_hook`-equivalent override before
checking -- confirmed passing by item 1's own run of the real hook). Generic, config-
independent checks (`N.assert_capture_paths_exist`, `assert_g_capture_paths`,
`assert_s31_capture_paths`, `assert_s31c_capture_paths`) do not reference any of these five
case-file keys and are unaffected.

**Not fully resolved** (out of the grep's positive-evidence reach; flagged, not asserted
clean): `p515_s35pt_phase1_checks.py` / `p515_s35pt_phase2_checks.py` reference
`price_taker` extensively but appear (from the lines read) to drive the flag through
explicit monkeypatches/overrides rather than reading the case-file default -- not verified
by execution.

## Validation

- Item 1: executed; `oracle_relevant_fields_match = True`; guard confirms 0 solves both
  sides (see Results).
- Item 2: executed once, write-once root, never re-run; confirmed the recourse_jump swap is
  a pure order tie (identical value `-85382653.71541198` under both labels at both
  positions) by direct inspection of both sidecar files, not inferred.
- Item 3: source-grep + manual read of each hit's enclosing function; not executed (the task
  scoped this item as zero-solve / list-only, and re-running any of these superseded stages
  is explicitly out of scope -- "no long runs").
- `git diff --stat -- data/SRP1/SRP1_params.json` inspected before commit: 13
  insertions, 10 deletions, no unintended key touched.
- `git status --porcelain` reviewed before every `git add`; only the intended paths staged.

## Unexpected findings

- The `objective_component_block_deltas` list in the recourse-jump sidecar is subject to
  Python per-process hash-seed-dependent tie-break ordering whenever two components have
  exactly equal `abs_delta` (`p515_g_g1_g4_admm_gates.py` ~line 2830,
  `for key in set(flat_current) | set(flat_previous):` followed by a stable sort). This is a
  pre-existing property of production capture code, unrelated to this task, and would affect
  *any* two separate-process reruns of the same arm, not just a case-file-vs-override
  comparison. Not fixed (no production-code edits authorized here); reported per CLAUDE.md's
  rule against rationalizing away a diff.
- `assert_s34_capture_paths` and `assert_s33e2_capture_paths` were **already** asserting
  stale rho values (`0.05` and `1.0` respectively) against the case file **before** this
  task's edit -- i.e. two of the "superseded" stages were already non-reproducible for an
  unrelated, pre-existing reason. Distinguished from the newly-superseded checks above so
  the two are not conflated.

## Remaining issues

- Item 2's strict gate (`bitwise_identical`) reads `False`. Every numeric value matches;
  the 6 diffs are two non-numeric provenance-checklist keys (doubled) and one
  order-only tie in a diagnostic top-10 list. Planner judgment needed on whether this
  satisfies "bitwise identity on every numeric artifact" or whether a follow-up (e.g. fixing
  the `set`-iteration non-determinism, out of this task's scope) is required first.
- Item 3's list is a grep-based search, explicitly scoped as such; `p515_s35pt_phase1_checks.py`
  / `p515_s35pt_phase2_checks.py` were not conclusively resolved either way.
- Per the "NOT permitted" list, items 4/5/6 of Addendum 22 (the polish gap, `REVISION_CONTEXT.md`
  head update, Step 3.7 Anderson acceleration) are untouched -- not part of this task.

## Questions for Planner

1. Does item 2's result (0 diffs on every numeric artifact and value; 6 diffs confined to
   two provenance-only checklist keys and one order-tie in a diagnostic list) count as
   satisfying "case file alone reproduces the oracle," or is a stricter re-run (e.g. with
   `PYTHONHASHSEED` fixed) required to also close the `set`-iteration tie?
2. Should the `p515_s35pt_phase1_checks.py` / `p515_s35pt_phase2_checks.py` price-taker flag
   dependence be checked more rigorously before the case-file supersession list is treated as
   complete?

## Commit hashes

1. `fb3de341` -- case file alone (`data/SRP1/SRP1_params.json`).
2. `ee976eff` -- `p515_s40_case_file_oracle_checks.py` + `p515_s40_case_file_repro.py`.
3. (this commit) -- evidence (`case_file_oracle_load_check.json`, `case_file_repro/`,
   `case_file_repro_launch.log`) + this report.
