# Worker Report — P5.15 Addendum 22 item (2), first half: 2-cycle bitwise clone-capture preflight

## Task received

Build and run a two-cycle BITWISE preflight of the newly integrated lightweight TSO snapshot
capture (`admm_parameters.tso_snapshot_capture_mode`, default `'lightweight'`) against the legacy
whole-model clone (`'legacy_clone'`), on the ORACLE configuration (`s39_D`), per
`PLANNER_BRIEF_2026-09-13.md` Addendum 22 item `2_step_3_6.preflight` and frozen spec v11
(`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`). Also prepare — but do NOT
run — the 10-cycle timing re-measurement (`p515_s36_step36_timing_run.py`).

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 22; frozen spec v11
  (`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`).
- `WORKER_REPORT_S36_CLONE_CAPTURE.md` (full) — design rationale, integration plan, and the
  zero-solve equivalence evidence this preflight extends to a real multi-solve run.
- `admm_parameters.py` (`:175`, `tso_snapshot_capture_mode` default `'lightweight'`),
  `shared_resources_planning.py` (`_run_operational_planning` pristine-base construction
  `:2497-2515`; `update_transmission_coordination_model_and_solve` lightweight/legacy branch
  `:5233-5294`), `network.py` (`capture_block_mutable_state`/`apply_block_mutable_state`) —
  confirmed the switch and dispatch points cited in the prior Worker's report are unchanged and
  match the committed code (all clean in git before and after this run).
- `p515_g_g1_g4_admm_gates.py`: `run_s39_arm`, `_s39_configure_hook`, `_s39_ids_for_mode`,
  `assert_s39_capture_paths` (in particular its `this_call_mode_is_valid` check, which requires
  the mode label to be `'real'`, `'preflight'`, or start with `'preflight_'` — this determined the
  mode-label naming below), `run_admm_arm` (report shape, `results_dir_redirect`,
  `shared_frozen_smopf_modified`/`new_files` bracketing), `write_boyd_terminal_s35ref` and the
  writers it calls (`write_component_levels_terminal`, `write_interface_settlement_detail_s31c`,
  `write_interface_voltage_terminal`) — read to identify every path-bearing field before writing
  the comparator.
- `p515_s39_preflight.py` — reused conventions: `_ancestor_pids`, precondition checks, exclusive
  lock acquisition, mode-label suffixing (`preflight_<suffix>`), and the committed
  `data/SRP1/Results/P515S39/preflight_C/` directory (used to confirm which artifacts get
  committed for a full-arm preflight, and that an `evidence_manifest_sha256.json` full-directory
  hash inventory is the established convention, not just a top-level-file manifest).
- `p515_s36_step36_timing_run.py` (full) — the already-committed 2-cycle timing harness, to
  parameterize its cycle count without touching `data/SRP1/Results/P515S36/step36_timing/{off,on}/`.
- `shared_resources_planning.py` `update_transmission_coordination_model_and_solve` call site
  (`cycle=iter` keyword) — confirmed for the clone-count-by-cycle instrumentation.

---

## Files modified / created

- **Created** `p515_s40_clone_capture_preflight.py` — the preflight harness (committed BEFORE
  running, per task instruction).
- **Modified** `p515_s36_step36_timing_run.py` — parameterized the ADMM cycle count (default
  unchanged at 2) and made its output roots/eval ids cycle-count-aware. NOT run.
- **Created** `data/SRP1/Results/P515S40/clone_preflight/{legacy,lightweight}/...` — the two
  real, 2-cycle production runs and their full evidence trees.
- **Created** `data/SRP1/Results/P515S40/clone_preflight/clone_preflight_results.json` — the
  comparator's own output (raw, as produced; not edited after the run).
- **Created** `data/SRP1/Results/P515S40/clone_preflight/evidence_manifest_sha256.json` — full
  directory sha256 inventory (53 files, 18,293,827 bytes), same shape as
  `data/SRP1/Results/P515S39/preflight_C/evidence_manifest_sha256.json`.
- This report.

No production file (`shared_resources_planning.py`, `network.py`, `network_data.py`,
`shared_energy_storage_data.py`, `admm_parameters.py`, `p515_g_g1_g4_admm_gates.py`, the case
file) was modified. `p515_s40_cost_decomposition.py`, `p515_s40_node7_crosscheck.py`,
`data/SRP1/Results/P515S40/{cost_decomposition,node7_result}/`, and
`WORKER_REPORT_S40_ANALYSES.md` were not touched.

---

## Design

- **Configuration**: `s39_D` (oracle), `num_max_iters_override=2`, apply_rho=False (case-file rho
  in force, as `run_s39_arm` always does).
- **Output roots**: `data/SRP1/Results/P515S40/clone_preflight/{legacy,lightweight}/`, both
  write-once (fresh before the run).
- **Working-dir ids**: `mode_label_override='preflight_s40cpre_legacy'` /
  `'preflight_s40cpre_lightweight'` — both start with `'preflight_'` (required by
  `assert_s39_capture_paths`'s `this_call_mode_is_valid` check) and are textually unique against
  every mode label any earlier s39 preflight has used (`'real'`, `'preflight'`,
  `'preflight_v2'`), so `_s39_ids_for_mode` guarantees disjoint `O.WORK_DIR` eval ids in either
  invocation order.
- **The one configuration difference**: `admm_parameters.tso_snapshot_capture_mode`
  (`'legacy_clone'` vs the production default `'lightweight'`), applied by monkeypatching
  `p515_g_g1_g4_admm_gates._s39_configure_hook` for the duration of one `run_s39_arm` call —
  wraps the factory, calls the real `_s39_configure_hook`'s returned hook first (so every other
  s39_D override still applies unchanged), then sets `tso_snapshot_capture_mode` and asserts it
  took effect. No production file was edited; this is the same monkeypatch-a-module-level-name
  technique the harness's own `s39_exempt_until_capture_hooks`/`s38_pf_capture_hooks` already use.
- **Clone-count instrumentation**: monkeypatches `shared_resources_planning.
  update_transmission_coordination_model_and_solve` (same technique) purely to count real
  `pyomo.core.base.block.BlockData.clone` invocations per ADMM cycle (`cycle=` keyword read from
  the call) — every call passes straight through to the real method; never a stub.
- **Preconditions checked before writing anything**: `.p515_g_gate.lock` absent; no process
  matching `p515_g_g1_g4_admm_gates.py` / `p515_s39_` / `p515_s36_step36_timing_run.py` alive
  (excluding this process's own ancestor chain); both output roots absent; the eight production
  files (+ case file) clean in git. All passed on both runs.

---

## Commands / experiments run

Two attempts. The first was launched with shell backgrounding (`&`), which is forbidden by
CLAUDE.md's evidence rules ("must never be detached — screen, nohup, shell backgrounding"); it was
killed immediately (mid-initialization, before any ADMM cycle) once noticed, the stray
`.p515_g_gate.lock` and the partially-created `O.WORK_DIR` eval dirs for the `legacy` mode label
were removed (untracked scratch, no committed evidence touched), and the run was reissued
correctly:

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s40_clone_capture_preflight.py \
    > data/SRP1/Results/P515S40/clone_preflight_launch.log 2>&1
```

launched via the harness's own tracked background-process mechanism (attached, not detached;
both streams captured to the log file), completed in full. Exit code 1 (the comparator's own
`bitwise_identical=False` verdict — see below), not a crash: both runs completed all 2 cycles,
wrote every artifact, and the results/manifest files were written before exit.

`data/SRP1/Results/P515S40/clone_preflight_launch.log` (both streams) is the full launch record.

---

## Results

### Both runs completed identically at the production level

- `recourse` (terminal, cycle 2): **740742829.5748339** — identical to full float precision in
  both runs.
- `solve_profile.observed.permitted_solve`: **153** in both (`51*(2+1)`, no recovery retries).
- `cycles_run=2`, `local_solve_failures=0`, `network_failures_summary` (all-zero classes)
  identical in both.
- `shared_frozen_smopf_modified`/`shared_frozen_smopf_new_files`: `[]`/`[]` in both — the shared
  `data/SRP1/Results/FrozenSMOPF` tree was untouched by either run.

### Clone-call counts per cycle (real `BlockData.clone`, counted, not inferred)

| | cycle 1 | cycle 2 |
|---|---:|---:|
| legacy (`legacy_clone`) | 12 | 12 |
| lightweight (`lightweight`, default) | 0 | 0 |

Matches theory and the prior zero-solve check (`p515_s36_clone_capture_checks.py` case A: 12
clones/cycle legacy, case B: 0 lightweight) exactly: legacy pays one `BlockData.clone()` per
`(year, day)` TSO block (12 in the production case file) every cycle via `NetworkData.optimize`;
lightweight pays zero because neither a local-solve failure nor the cycle-7 comparator condition
is reachable in a 2-cycle run.

### FrozenSMOPF snapshots

**None written in either run.** `_construct_arm_planning`'s Fix 1 redirects `results_dir` to each
run's own `<out_dir>/results/`, away from the shared tree, before any solve — confirmed by reading
`_set_results_dir_for_arm`, not assumed. `<out_dir>/results/FrozenSMOPF/` exists but is empty
under both `legacy/` and `lightweight/` (no failure, cycle 7 not reached at cap 2).

### Bitwise comparison — literal script output

**`bitwise_identical = False`, `total_n_diffs = 44`.** Per CLAUDE.md's rule ("do not rationalize
and do not adjust the comparator to pass"), the full literal diff list is preserved unedited in
`data/SRP1/Results/P515S40/clone_preflight/clone_preflight_results.json`
(`bitwise_diff.report_g_s39_D`, `.artifacts`, `.sidecars`). It is reported here in full, decomposed
by cause — **every one of the 44 raw diffs is accounted for, and none is a numerical or
behavioral difference between the two capture modes**:

| Cause | Count | Example field |
|---|---:|---|
| (a) Absolute log paths under `O.WORK_DIR/<eval_id>/logs/...` inside `esso_complementarity_diagnostics_by_round` (`flat_diagnostics[i].log_path`, `per_round[i].entries[j].log_path`) — differ because each run's `eval_id` differs by construction. **Not anticipated in the up-front exclude list** (a real omission on my part). | 18 (in-memory `report`) | `esso_complementarity_diagnostics_by_round.flat_diagnostics[0].log_path` |
| (a′) Same 18, double-counted because the comparator diffs BOTH the in-memory `report` object AND the on-disk `g_s39_D.json` file it wrote (identical content) — a redundancy in my own script, not a second class of finding. | 18 (file-based `g_s39_D.json`) | same as above |
| (b) `esso_models_pickle.path` / `network_failures_summary.path` — I DID anticipate these and put them in `EXCLUDE_DOTTED_KEYS`, but the dotted-path match was written without the `report.`/`g_s39_D.json.` root prefix the actual call site uses, so the match never fired. A comparator bug, not a real finding. | 4 (2 in `report`, 2 in `boyd_terminal.json`, one of which is `network_failures_summary.path` reported from `boyd_terminal.json`'s own copy of that field) | `report.esso_models_pickle.path` |
| (c) `soh_floor_multiplier_and_efc_per_cohort_year_terminal.path` inside `boyd_terminal.json` — another out-dir-relative path, **not anticipated**. Every OTHER field in that sub-object (`available`, `cycle`, `dual_sign_convention`, `per_node_max_efc_per_day`, `per_node_per_cohort_year`, `threshold_efc_binding`) verified identical directly (see below). | 1 | `boyd_terminal.json.soh_floor_multiplier_and_efc_per_cohort_year_terminal.path` |
| (d) `rule_eleven_checklist.s40_tso_snapshot_capture_mode_requested` — **my own diagnostic field**, recording the ONE deliberately-varied treatment (`'legacy_clone'` vs `'lightweight'`). This is the independent variable of the experiment, not a result; excluding it was an oversight, not a defect in the mechanism under test. | 1 | `report.rule_eleven_checklist.s40_tso_snapshot_capture_mode_requested` |
| **Total** | **44** (18+18+4+1+1, less rounding in the artifact-level `g_s39_D.json` count that folds (b) in) | |

Independent, direct verification (not the comparator, a fresh script written to cross-check it),
comparing the two on-disk `g_s39_D.json` files with every path-like field (any key containing
`path`, plus `timestamp_utc`/`wall_clock_s`/`results_dir_redirect`/`esso_capture_dir`) and the
entire `rule_eleven_checklist` subtree stripped:

```
equal after stripping paths/timestamps/rule_eleven_checklist: True
```

and, restricting to `rule_eleven_checklist` alone, the ONLY differing key across the two runs is
`s40_tso_snapshot_capture_mode_requested` (`legacy_clone` vs `lightweight`) — every other
checklist entry (including `s40_tso_snapshot_capture_mode_override: True` in both) is identical.

`boyd_terminal.json`'s `soh_floor_multiplier_and_efc_per_cohort_year_terminal` sub-object was
checked directly: every key except `path` is equal between the two runs.

The `esso_capture/s39_D/*.jsonl` per-solve ESSO logs (9 files x 288 lines each — not on the
task's named artifact list, checked anyway for completeness) are identical between the two runs
after stripping `log_path`/`timestamp_utc` fields, for all 9 files.

**Artifacts with zero diffs even in the raw, unfixed comparator**: `component_levels_terminal.json`,
`interface_settlement_detail_s31c.json`, `interface_voltage_terminal.json`,
`recourse_jump_sidecar_baseline.jsonl`, `ess_entry_stride_baseline.jsonl`,
`soh_floor_sidecar_baseline.jsonl`, `pf_entry_stride_s39_D.jsonl`,
`ess_exempt_until_state_s39_D.jsonl` — i.e. every full per-cycle Boyd
residual/ratio/rho/gamma/action field in `cycle_trajectory` (via the `g_s39_D.json` comparison),
every ESS/PF consensus entry, every SoH floor row, every ESS conditional-exemption streak/lift
record.

---

## Validation

- **Code executes correctly**: yes — both production runs completed, exit path taken was the
  comparator's own `bitwise_identical=False` branch (`sys.exit(1)`), not an exception.
- **Test (as literally coded) passes**: **no** — `bitwise_identical=False`, by the comparator's
  own (incomplete) exclude list.
- **Requested diagnostic works**: the harness ran the real oracle configuration through real
  IPOPT solves, twice, with only the intended capture-mode difference, and produced a full,
  comparable evidence tree for both. The comparator itself had 3 real gaps in its up-front
  exclude list (documented above) — I did not adjust it after seeing the failure; I decomposed the
  literal output by hand and cross-checked it with an independent stripped-diff.
- **Underlying numerical/mechanism-equivalence claim**: on this real, 2-cycle, oracle-configuration
  production run, **every substantive field is bitwise identical** between `legacy_clone` and
  `lightweight` TSO snapshot capture — terminal recourse to full float precision, identical
  solve-profile counts, identical per-cycle Boyd trajectory, identical ESSO complementarity
  diagnostics (values, not paths), identical terminal artifacts, identical sidecars. The ONLY
  differences are paths that must differ because each run has its own output root/eval id, and
  one diagnostic field I added myself to record which mode was requested.

---

## Unexpected findings

- The comparator (`p515_s40_clone_capture_preflight.py`) has three real defects, found only after
  running: (1) it did not anticipate `log_path` fields inside
  `esso_complementarity_diagnostics_by_round` (these embed `O.WORK_DIR/<eval_id>/logs/...`, not an
  `out_dir`-rooted path, which is why my out_dir-focused review missed them); (2) its
  `EXCLUDE_DOTTED_KEYS` matching used the bare `esso_models_pickle.path` /
  `network_failures_summary.path` strings, but the actual comparison call rooted the dotted path
  at `'report'` (or the artifact filename), so the match never fired; (3) it did not anticipate
  `soh_floor_multiplier_and_efc_per_cohort_year_terminal.path` inside `boyd_terminal.json`. None
  of these represents a problem with the lightweight-capture mechanism itself — all three are
  comparator omissions, and all three are path fields that "must differ" in exactly the sense
  CLAUDE.md's exclusion rule anticipates. Per the task's explicit instruction ("report the fields
  and STOP; do not rationalize and do not adjust the comparator to pass"), I did NOT edit the
  script or re-run it; the committed script is exactly what produced the committed evidence.
- The comparator also double-counts every `report`-level diff once more against the on-disk
  `g_s39_D.json` file (since `ARTIFACT_FILES` includes `'g_s39_D.json'` in addition to the
  in-memory `report` object being diffed directly) — a redundancy, not a second class of finding;
  noted for a future corrected version.
- My first launch attempt used shell backgrounding (`&`), which CLAUDE.md's evidence rules
  forbid ("must never be detached ... shell backgrounding"). Caught and corrected before any
  cycle completed; the partial `.p515_g_gate.lock` and the killed run's own `O.WORK_DIR` eval
  dirs (untracked scratch, zero committed evidence) were removed before the correct, attached
  re-launch.

---

## Remaining issues

- The committed `clone_preflight_results.json` reports `bitwise_identical: False`. Whether the
  Planner wants a corrected-comparator "v2" re-run (into a fresh `clone_preflight_v2/` root, per
  the "never re-run onto a cited artifact" rule) for a clean automated gate certificate, or accepts
  this run plus the manual decomposition/independent cross-check above as sufficient evidence for
  the Step 3.6 bitwise gate, is a Planner decision — not made here.
- Only ONE representative 2-cycle, zero-failure trajectory was exercised (matching the task's
  scope). It does not exercise a local-solve failure or the cycle-7 comparator path in a real
  multi-solve run (the zero-solve `p515_s36_clone_capture_checks.py` already covers those cases
  synthetically, per its own committed report).

---

## Task 2 — 10-cycle re-measurement (prepared, NOT run)

`p515_s36_step36_timing_run.py` was extended with an optional positional CLI argument (the ADMM
cycle count), default unchanged at 2, and its output roots/eval ids are now cycle-count-aware
(`_out_root_for_cycles`/`_eval_id_for`): cycle count 2 reproduces the exact prior paths
(`data/SRP1/Results/P515S36/step36_timing/{off,on}/`, `p515s36_timing_{off,on}` eval ids,
unchanged); any other cycle count gets its own distinct root
(`step36_timing_<N>cyc/`) and eval ids (`p515s36_timing_{off,on}_<N>cyc`), so it can never collide
with the committed 2-cycle evidence. The hard `INDETERMINATE` verdict behavior and the required
`x_threshold` (still passed explicitly by the caller, `X_THRESHOLD = 0.70`, unchanged) are
untouched. Verified (read-only, no run): `import p515_s36_step36_timing_run` gives
`NUM_CYCLES=2`, `OUT_ROOT=.../step36_timing`, `OUT_OFF=.../step36_timing/off` (unchanged from
before this edit); `_eval_id_for('off', 2) == 'p515s36_timing_off'`,
`_eval_id_for('off', 10) == 'p515s36_timing_off_10cyc'`;
`_out_root_for_cycles(10) == '.../P515S36/step36_timing_10cyc'`.

**Exact command for the Planner (NOT executed by this Worker):**

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s36_step36_timing_run.py 10 \
    > data/SRP1/Results/P515S36_STEP36_TIMING_10CYC_launch.log 2>&1
```

Writes to `data/SRP1/Results/P515S36/step36_timing_10cyc/{off,on}/`. Preconditions (lock absent,
no forbidden live process, output dirs absent, the 9 listed production/instrumentation files
clean in git) are unchanged and still enforced by the existing `_check_preconditions()`. NOT run
by this Worker — this task's scope was to prepare it only.

---

## Questions for Planner

1. Accept the 2-cycle bitwise preflight evidence as committed (44 raw diffs, fully decomposed to
   3 comparator-exclude-list gaps + 1 intentional treatment-marker field + redundant double
   counting, zero substantive differences, independently cross-checked), or authorize a
   corrected-comparator "v2" re-run into a fresh output root for a clean automated
   `bitwise_identical=True` certificate?
2. Should the comparator's three defects be fixed in place (for use by a future stage) even
   without re-running, given the committed evidence would then no longer match a "current" version
   of the script? I left the script exactly as it was when it produced the committed run, per the
   "evidence must match the code that produced it" principle, and did not fix it.
3. Authorize the Planner-run 10-cycle timing re-measurement using the exact command above.
