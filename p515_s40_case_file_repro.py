"""
P5.15 Addendum 22 item (3) -- two-cycle CASE-FILE-ALONE reproduction of the
oracle (arm `s39_D`).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 22, item `3_case_file`;
frozen spec v11 `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`.

`data/SRP1/SRP1_params.json` now carries D's configuration directly (committed
alone, per the spec's own risk note -- the first change to the case file in
this campaign; see `p515_s40_case_file_oracle_checks.py` for the zero-solve
field-by-field proof that the case file, loaded through the real production
loader, is numerically identical -- aside from `_source` provenance
bookkeeping -- to a fresh planning object with the oracle's own override hook
(`p515_g_g1_g4_admm_gates._s39_configure_hook('s39_D')`) applied).

This script runs `run_admm_arm` DIRECTLY (never `run_s39_arm`, whose own
pre-check structurally re-applies the override AND whose
`assert_s39_capture_paths` hard-asserts the OVERRIDE source string -- see the
Worker report for why that check is now superseded), with:
  - `label='s39_D'`             (matches the oracle's own artifact filenames,
                                  e.g. `g_s39_D.json`, `pf_entry_stride_s39_D.jsonl`)
  - `apply_rho=False`           (the harness's own `N.RHO` override is NEVER
                                  applied; the case-file rho -- now D's rho --
                                  is what flows through)
  - `pre_solve_hook=None`       (NO s39-specific Python override of any kind
                                  -- configuration comes from the case file
                                  alone)
  - `full_diagnostics_in_rows=True`  (matches the oracle run's own row shape)
  - the SAME generic capture-hook wiring every s34/s35ref/s38/s39 arm uses
    (`s38_pf_capture_hooks`, `s39_exempt_until_capture_hooks`,
    `write_boyd_terminal_s35ref`) -- these are capture-only wrappers (they
    read the REAL production function's own return value and write sidecars;
    they never mutate `planning.params`), so using them here is "capture
    every arm uses", not a configure hook.
  - `num_max_iters_override=2`  (this script's own 2-cycle smoke-test cap;
                                  `num_max_iters` itself is NOT exercised by a
                                  2-cycle run -- covered instead, zero-solve,
                                  by `p515_s40_case_file_oracle_checks.py`)

Diffs the result, BITWISE, against the committed two-cycle oracle run
`data/SRP1/Results/P515S40/clone_preflight_v2/lightweight/` (produced by
`run_s39_arm('s39_D', num_max_iters_override=2, ...)`, i.e. the override
path), reusing `p515_s40_clone_capture_preflight.py`'s own comparator
(`_diff`), exclusion sets (`EXCLUDE_KEY_NAMES`, `EXCLUDE_DOTTED_SUFFIXES`,
`INTENTIONAL_DIFF_SUFFIXES`) and artifact/sidecar file lists BY IMPORT --
never reimplemented. Nothing is added to those exclusion sets here; any
field outside them that differs is reported as-is, not rationalized away
(CLAUDE.md: "do not adjust the comparator").

One EXPECTED, already-anticipated structural difference: the oracle report's
`rule_eleven_checklist.s39_pre_solve_override_verification` block exists only
because `_s39_configure_hook` ran on that arm (it writes the block itself,
after checking its own overrides took effect); this run never calls that
hook, so the key is simply absent here. This is reported, not excluded.

======================================================================
EXACT LAUNCH COMMAND
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s40_case_file_repro.py \\
        > data/SRP1/Results/P515S40/case_file_repro_launch.log 2>&1

Run attached, alone (no `screen`/`nohup`/backgrounding), both streams
captured via the shell redirection above.

======================================================================
PRECONDITIONS (checked before anything is written; refuses loudly otherwise)
======================================================================
  1. `.p515_g_gate.lock` does not already exist.
  2. No OTHER process (excluding this process's own ancestor chain) matches
     `p515_g_g1_g4_admm_gates.py`, `p515_s39_`, `p515_s40_`, or
     `p515_s36_step36_timing_run.py` in `ps aux`.
  3. The output root does not exist yet (write-once).
  4. The production files this task touches (including the case file) are
     clean in git -- so this run executes against exactly the committed
     case-file change, not a mid-edit tree. The case-file change is committed
     FIRST (separately), then this script is run.
  5. The committed oracle comparator directory exists.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator conventions, BY IMPORT
import shared_energy_storage_data as SED  # noqa: E402

ARM_LABEL = 's39_D'
NUM_CYCLES = 2
RUN_EVAL_ID = 'p515s40_case_file_repro_run'
PRECHECK_EVAL_ID = 'p515s40_case_file_repro_precheck'

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'case_file_repro')
ORACLE_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'clone_preflight_v2', 'lightweight')

# Extends `CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS` (which does not itself list
# 'p515_s40_') per this task's own precondition list.
FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + ('p515_s40_',)


def _check_preconditions():
    failures = []
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded_pids = {str(p) for p in CP._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue
        if any(substring in line for substring in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    if os.path.exists(OUT_DIR):
        failures.append(f'output directory already exists (write-once): {OUT_DIR}')

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(CP.PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    if not os.path.isdir(ORACLE_DIR):
        failures.append(f'committed oracle two-cycle run directory missing: {ORACLE_DIR}')

    return failures


def _build_floor_rows():
    """Zero-solve SoH floor-row identification, structural -- independent of
    ADMM overrides (mirrors `assert_s39_capture_paths`'s own pre-solve block,
    on a PLAIN, non-overridden planning object)."""
    precheck_planning = G.O.fresh_planning(PRECHECK_EVAL_ID)
    capture_checklist = G.assert_s31c_capture_paths(precheck_planning)
    active_nodes = list(precheck_planning.shared_ess_data.active_distribution_network_nodes)
    probe_esso_models = {node_id: SED._build_subproblem(precheck_planning.shared_ess_data, node_id)
                          for node_id in active_nodes}
    floor_rows_by_node, floor_counts_by_node = G._identify_soh_floor_rows(probe_esso_models)
    del probe_esso_models, precheck_planning
    return capture_checklist, floor_rows_by_node, floor_counts_by_node


def main():
    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[S40-CASE-FILE-REPRO PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S40-CASE-FILE-REPRO] preconditions passed (no lock, no forbidden process, '
          'fresh output dir, production files clean, oracle dir present).')

    G._acquire_exclusive_run_lock()
    os.makedirs(OUT_DIR, exist_ok=True)

    started = time.time()
    _capture_checklist, floor_rows_by_node, floor_counts_by_node = _build_floor_rows()
    print(f'[S40-CASE-FILE-REPRO] soh_floor_row_counts_by_node='
          f"{ {n: len(r) for n, r in floor_rows_by_node.items()} }")

    recourse_jump_path = os.path.join(OUT_DIR, 'recourse_jump_sidecar_baseline.jsonl')
    ess_stride_path = os.path.join(OUT_DIR, 'ess_entry_stride_baseline.jsonl')
    floor_sidecar_path = os.path.join(OUT_DIR, 'soh_floor_sidecar_baseline.jsonl')
    pf_stride_path = os.path.join(OUT_DIR, f'pf_entry_stride_{ARM_LABEL}.jsonl')
    exempt_until_state_path = os.path.join(OUT_DIR, f'ess_exempt_until_state_{ARM_LABEL}.jsonl')
    for path in (recourse_jump_path, ess_stride_path, floor_sidecar_path, pf_stride_path,
                exempt_until_state_path):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')

    def _hook(planning, sed, models, rows, report, out_dir, label):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, REPO)
        report['s38_pf_entry_stride_sidecar_path'] = os.path.relpath(pf_stride_path, REPO)
        report['s39_ess_exempt_until_state_sidecar_path'] = os.path.relpath(exempt_until_state_path, REPO)
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=floor_sidecar_path)

    print(f'[S40-CASE-FILE-REPRO] run CASE-FILE-ALONE (label={ARM_LABEL!r}, apply_rho=False, '
          f'pre_solve_hook=None), {NUM_CYCLES} cycles.')
    with G.s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                pf_stride_path, floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(exempt_until_state_path):
        report, report_path = G.run_admm_arm(
            ARM_LABEL, OUT_DIR, k_override=None, eval_id=RUN_EVAL_ID,
            num_max_iters_override=NUM_CYCLES, apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=_hook, pre_solve_hook=None)

    print(f"[S40-CASE-FILE-REPRO] cycles_run={report['cycles_run']} recourse={report['recourse']} "
          f"wall={report['wall_clock_s']:.1f}s")

    # ---- bitwise diff against the committed oracle two-cycle run ----------
    oracle_g_path = os.path.join(ORACLE_DIR, 'g_s39_D.json')
    oracle_report = CP._load_json(oracle_g_path)
    diff_report = CP._diff(oracle_report, report, 'report') if oracle_report is not None else None

    artifact_diffs = {}
    for fname in CP.ARTIFACT_FILES:
        a_path = os.path.join(ORACLE_DIR, fname)
        b_path = os.path.join(OUT_DIR, fname)
        a_json = CP._load_json(a_path)
        b_json = CP._load_json(b_path)
        if a_json is None or b_json is None:
            artifact_diffs[fname] = {
                'error': f'missing artifact: oracle_exists={a_json is not None} '
                         f'case_file_alone_exists={b_json is not None}',
                'diffs': [], 'n_diffs': None,
            }
            continue
        d = CP._diff(a_json, b_json, fname)
        artifact_diffs[fname] = {'diffs': d, 'n_diffs': len(d)}

    sidecar_diffs = {}
    for fname in CP.SIDECAR_JSONL_FILES:
        a_path = os.path.join(ORACLE_DIR, fname)
        b_path = os.path.join(OUT_DIR, fname)
        a_rows = CP._load_jsonl(a_path)
        b_rows = CP._load_jsonl(b_path)
        if a_rows is None or b_rows is None:
            sidecar_diffs[fname] = {
                'error': f'missing sidecar: oracle_exists={a_rows is not None} '
                         f'case_file_alone_exists={b_rows is not None}',
                'diffs': [], 'n_diffs': None,
            }
            continue
        d = CP._diff(a_rows, b_rows, fname)
        sidecar_diffs[fname] = {'diffs': d, 'n_diffs': len(d), 'n_rows': len(a_rows)}

    report_diffs_n = len(diff_report) if diff_report is not None else None
    all_diffs_n = (
        (report_diffs_n or 0)
        + sum((v['n_diffs'] or 0) for v in artifact_diffs.values())
        + sum((v['n_diffs'] or 0) for v in sidecar_diffs.values())
    )
    all_artifacts_present = (
        oracle_report is not None
        and all('error' not in v for v in artifact_diffs.values())
        and all('error' not in v for v in sidecar_diffs.values())
    )
    bitwise_identical = (all_diffs_n == 0) and all_artifacts_present

    payload = {
        'stage': 'P5.15 Addendum 22 item (3) -- case-file-alone 2-cycle reproduction of the oracle',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 22 item 3_case_file',
            'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm_label': ARM_LABEL,
        'num_cycles': NUM_CYCLES,
        'case_file_alone': {
            'out_dir': os.path.relpath(OUT_DIR, REPO),
            'report_path': os.path.relpath(report_path, REPO),
            'apply_rho': False,
            'pre_solve_hook': None,
        },
        'oracle': {
            'out_dir': os.path.relpath(ORACLE_DIR, REPO),
            'source': "run_s39_arm('s39_D', num_max_iters_override=2, ...) -- the override path",
        },
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'intentional_difference_suffixes': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
        'bitwise_diff': {
            'report_g_s39_D': {'diffs': diff_report, 'n_diffs': report_diffs_n},
            'artifacts': artifact_diffs,
            'sidecars': sidecar_diffs,
            'total_n_diffs': all_diffs_n,
            'all_artifacts_present': all_artifacts_present,
        },
        'bitwise_identical': bitwise_identical,
        'wall_clock_s': time.time() - started,
    }

    results_path = os.path.join(OUT_DIR, 'case_file_repro_results.json')
    if os.path.exists(results_path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {results_path}')
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S40-CASE-FILE-REPRO] wrote {results_path}')
    print(f'[S40-CASE-FILE-REPRO] bitwise_identical={bitwise_identical} total_n_diffs={all_diffs_n} '
          f'all_artifacts_present={all_artifacts_present}')
    if not bitwise_identical:
        print('[S40-CASE-FILE-REPRO] *** BITWISE COMPARISON FAILED / DIFFERS *** -- see diffs above/in '
              f'{results_path}. Reported as-is; NOT rationalized; comparator NOT adjusted.')
        first_diffs = (
            (diff_report or [])
            + [d for v in artifact_diffs.values() for d in (v.get('diffs') or [])]
            + [d for v in sidecar_diffs.values() for d in (v.get('diffs') or [])]
        )
        for d in first_diffs[:20]:
            print(f'  [FIRST DIFFS] {d}')

    manifest = {os.path.relpath(results_path, REPO): CP._sha256_file(results_path)}
    manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(manifest_path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {manifest_path}')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S40-CASE-FILE-REPRO] wrote {manifest_path}')

    if not bitwise_identical:
        sys.exit(1)


if __name__ == '__main__':
    main()
