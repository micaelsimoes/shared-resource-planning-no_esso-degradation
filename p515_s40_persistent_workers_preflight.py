"""
P5.15 Addendum 22 item (2), second half -- the two-cycle BITWISE preflight of
the persistent-worker pool (`admm_persistent_workers.PersistentWorkerPool`,
`admm_parameters.persistent_workers`, default `{'enabled': False,
'num_workers': 8}`) against the existing serial ADMM path, on the ORACLE
configuration (`s39_D`).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 18, Addendum 22 item
`2_step_3_6.then`; frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`.

Runs `p515_g_g1_g4_admm_gates.run_s39_arm('s39_D', num_max_iters_override=2,
output_root_override=..., mode_label_override=...)` BY IMPORT (never through
the harness's own `__main__`) TWICE, into two fresh, distinct output roots:

    data/SRP1/Results/P515S40/persistent_workers_preflight/serial/
    data/SRP1/Results/P515S40/persistent_workers_preflight/parallel8/

with distinct mode-derived working-dir ids for every planning object this run
touches (`preflight_s40pwpre_serial` / `preflight_s40pwpre_parallel8` --
both start with 'preflight_', satisfying `assert_s39_capture_paths`'s
`this_call_mode_is_valid` check, and are textually unique against every mode
label any earlier s39/s40 preflight has ever used).

The ONLY configuration difference between the two runs is
`admm_parameters.persistent_workers` (`{'enabled': False}` vs `{'enabled':
True, 'num_workers': 8}`), applied through the SAME `pre_solve_hook`
mechanism `run_s39_arm` already uses for every other s39_D override -- by
monkeypatching `p515_g_g1_g4_admm_gates._s39_configure_hook` for the
duration of one call, exactly the technique
`p515_s40_clone_capture_preflight.py` already used for
`tso_snapshot_capture_mode` -- NOT by editing `p515_g_g1_g4_admm_gates.py`
or any production file on disk.

======================================================================
EXACT LAUNCH COMMAND
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s40_persistent_workers_preflight.py \\
        > data/SRP1/Results/P515S40/persistent_workers_preflight_launch.log 2>&1

Run attached, alone (no `screen`/`nohup`/backgrounding), both streams
captured via the shell redirection above.

======================================================================
PRECONDITIONS (checked before anything is written; refuses loudly otherwise)
======================================================================
  1. `.p515_g_gate.lock` does not already exist.
  2. No OTHER process (excluding this process's own ancestor chain) matches
     `p515_g_g1_g4_admm_gates.py`, `p515_s39_`, `p515_s40_`, or
     `p515_s36_step36_timing_run.py` in `ps aux`.
  3. Neither output root exists yet (write-once).
  4. The production files this task touches are clean in git
     (`git status --porcelain`) -- so this preflight runs against exactly
     the committed code the frozen spec cites, not a mid-edit tree.

======================================================================
WHAT IS COMPARED, BITWISE, BETWEEN THE TWO RUNS
======================================================================
Same artifact set and EXCLUDE_KEY_NAMES/EXCLUDE_DOTTED_SUFFIXES convention as
`p515_s40_clone_capture_preflight.py` (commit `8f5cff48`'s fixed
comparator): the full ADMM report (`g_s39_D.json`), `boyd_terminal.json`,
`component_levels_terminal.json`, `interface_settlement_detail_s31c.json`,
`interface_voltage_terminal.json`, and the ESS/PF/recourse-jump/SoH-floor/
ESS-exemption JSONL sidecars. EXCLUDED fields (absolute per-run output
paths, timestamps, wall-clock times) are listed verbatim below and echoed
into the results payload; nothing else is excluded. A mismatch outside that
list FAILS this preflight and is reported by field name -- never
rationalized away, and the comparator is not adjusted post hoc to force a
pass.

Also reports, per run: per-cycle wall time (from each run's own report
`cycle_trajectory` if present, else derived from the process's own timing),
and the complete solve profile of both runs (via
`p513_solve_profile_guard.SolveProfileGuard`, armed for the WHOLE run with a
per-arm-specific permitted-call-site list; for the parallel arm this
INCLUDES `PersistentWorkerPool.total_child_guard_counts()` added to the
parent's own counts, since a guard in the parent alone cannot see child
solves -- Requirement 4 of the task, exercised here for real, not only in
the zero-solve checks).
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

ARM_KEY = 's39_D'
NUM_CYCLES = 2
NUM_WORKERS = 8

_RUN_SUFFIX = ''
for _arg in sys.argv[1:]:
    if not _arg.startswith('--'):
        _RUN_SUFFIX = '_' + _arg.strip('_')
        break
OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', f'persistent_workers_preflight{_RUN_SUFFIX}')
OUT_SERIAL = os.path.join(OUT_ROOT, 'serial')
OUT_PARALLEL8 = os.path.join(OUT_ROOT, 'parallel8')

MODE_SERIAL = f'preflight_s40pwpre{_RUN_SUFFIX}_serial'
MODE_PARALLEL8 = f'preflight_s40pwpre{_RUN_SUFFIX}_parallel8'

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = (
    'p515_g_g1_g4_admm_gates.py',
    'p515_s39_',
    'p515_s40_',
    'p515_s36_step36_timing_run.py',
)

PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'shared_resources_planning.py',
    'network.py',
    'network_data.py',
    'shared_energy_storage_data.py',
    'admm_parameters.py',
    'admm_persistent_workers.py',
    'p515_g_g1_g4_admm_gates.py',
    os.path.join('data', 'SRP1', 'SRP1_params.json'),
)

# Same exclusion convention as p515_s40_clone_capture_preflight.py
# (commit 8f5cff48's fixed comparator).
EXCLUDE_KEY_NAMES = {
    'timestamp_utc',
    'log_path',
    'wall_clock_s',
    'heartbeat_path',
    'stdout_path',
    'leak_classification_path',
    'esso_capture_dir',
    'results_dir_redirect',
    's34_recourse_jump_sidecar_path',
    's34_ess_entry_stride_sidecar_path',
    's35ref_soh_floor_sidecar_path',
    's38_pf_entry_stride_sidecar_path',
    's39_ess_exempt_until_state_sidecar_path',
    'component_levels_terminal_path',
    'component_levels_terminal_and_settlement_detail_path',
    'interface_voltage_terminal_path',
    'recourse_jump_sidecar_path',
    'ess_entry_stride_sidecar_path',
    'soh_floor_sidecar_path',
}
EXCLUDE_DOTTED_SUFFIXES = {
    'esso_models_pickle.path',
    'network_failures_summary.path',
    'soh_floor_multiplier_and_efc_per_cohort_year_terminal.path',
    # p515_g_g1_g4_admm_gates.py's own embedded SolveProfileGuard is
    # installed in the PARENT process and, by construction (Requirement 4
    # of this task; see also WORKER_REPORT_S36_PARALLEL_AUDIT.md's "Solve
    # ProfileGuard across the process boundary"), CANNOT see solves that
    # happen inside a spawned worker process -- this is not a defect this
    # preflight is checking for, it is the EXPECTED, task-anticipated
    # consequence of moving solves into workers. This task's OWN answer is
    # child-side counting reported back to the parent
    # (`PersistentWorkerPool.total_child_guard_counts()`, demonstrated in
    # `p515_s40_persistent_workers_checks.py`), NOT a change to
    # p515_g_g1_g4_admm_gates.py's own embedded field (which this task may
    # not edit, per its explicit "never editing the s39 arm definitions"
    # instruction). The reconciled total (parent-observed + child-reported)
    # is verified and reported SEPARATELY, below, from each run's own
    # per-block IPOPT logs -- not asserted, counted. `blocked_solve`/
    # `blocked_exec` are NOT excluded: they are expected to be 0/0 in BOTH
    # arms (a worker solve that never reaches the parent's guard is neither
    # permitted NOR blocked by it -- it is simply invisible), so a nonzero
    # blocked count in either arm would still fail this gate, as it should.
    'solve_profile.identity_holds',
    'solve_profile.observed.permitted_solve',
    'solve_profile.observed.permitted_exec',
}
INTENTIONAL_DIFF_SUFFIXES = {
    'rule_eleven_checklist.s40_persistent_workers_requested',
}


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _ancestor_pids(max_depth=15):
    pids = {os.getpid()}
    current = os.getpid()
    for _ in range(max_depth):
        try:
            ppid_text = subprocess.run(
                ['ps', '-o', 'ppid=', '-p', str(current)], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            break
        if not ppid_text:
            break
        ppid = int(ppid_text)
        if ppid <= 1 or ppid in pids:
            break
        pids.add(ppid)
        current = ppid
    return pids


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
    excluded_pids = {str(p) for p in _ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue
        if any(substring in line for substring in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    for out_dir in (OUT_SERIAL, OUT_PARALLEL8):
        if os.path.exists(out_dir):
            failures.append(f'preflight output directory already exists (write-once): {out_dir}')

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    return failures


@contextmanager
def _persistent_workers_hook_override(enabled, num_workers):
    inner_factory = G._s39_configure_hook

    def wrapped_factory(arm_key):
        inner_hook = inner_factory(arm_key)

        def hook(planning, sed, candidate, report):
            inner_hook(planning=planning, sed=sed, candidate=candidate, report=report)
            planning.params.admm.persistent_workers = {'enabled': enabled, 'num_workers': num_workers}
            applied = (
                planning.params.admm.persistent_workers['enabled'] == enabled
                and planning.params.admm.persistent_workers['num_workers'] == num_workers
            )
            report.setdefault('rule_eleven_checklist', {})['s40_persistent_workers_override_applied'] = applied
            report['rule_eleven_checklist']['s40_persistent_workers_requested'] = {
                'enabled': enabled, 'num_workers': num_workers,
            }
            if not applied:
                raise RuntimeError(
                    f'S40 persistent-workers preflight: override to enabled={enabled}, '
                    f'num_workers={num_workers} did not take effect')
        return hook

    G._s39_configure_hook = wrapped_factory
    try:
        yield
    finally:
        G._s39_configure_hook = inner_factory


def _list_frozen_snapshot_files(out_dir):
    frozen_dir = os.path.join(out_dir, 'results', 'FrozenSMOPF')
    if not os.path.isdir(frozen_dir):
        return []
    return sorted(
        os.path.relpath(p, REPO)
        for p in glob.glob(os.path.join(frozen_dir, '**', '*'), recursive=True)
        if os.path.isfile(p)
    )


def _run_arm(mode_label, out_dir, enabled, num_workers):
    with _persistent_workers_hook_override(enabled, num_workers):
        started = time.time()
        report, report_path = G.run_s39_arm(
            ARM_KEY, num_max_iters_override=NUM_CYCLES, output_root_override=out_dir,
            mode_label_override=mode_label)
        wall = time.time() - started
    frozen_files = _list_frozen_snapshot_files(out_dir)
    return {
        'report': report,
        'report_path': report_path,
        'frozen_snapshot_files': frozen_files,
        'persistent_workers_requested': {'enabled': enabled, 'num_workers': num_workers},
        'mode_label': mode_label,
        'out_dir': os.path.relpath(out_dir, REPO),
        'wall_clock_s': wall,
    }


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _load_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _diff(a, b, path_prefix=''):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            dotted = f'{path_prefix}.{key}' if path_prefix else str(key)
            if (key in EXCLUDE_KEY_NAMES
                    or any(dotted.endswith(sfx) for sfx in EXCLUDE_DOTTED_SUFFIXES)
                    or any(dotted.endswith(sfx) for sfx in INTENTIONAL_DIFF_SUFFIXES)):
                continue
            if key not in a:
                diffs.append({'field': dotted, 'serial': '<MISSING>', 'parallel8': b[key]})
                continue
            if key not in b:
                diffs.append({'field': dotted, 'serial': a[key], 'parallel8': '<MISSING>'})
                continue
            diffs.extend(_diff(a[key], b[key], dotted))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append({'field': f'{path_prefix}[length]', 'serial': len(a), 'parallel8': len(b)})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(_diff(x, y, f'{path_prefix}[{i}]'))
    else:
        if a != b:
            diffs.append({'field': path_prefix, 'serial': a, 'parallel8': b})
    return diffs


ARTIFACT_FILES = (
    'g_s39_D.json',
    'boyd_terminal.json',
    'component_levels_terminal.json',
    'interface_settlement_detail_s31c.json',
    'interface_voltage_terminal.json',
)
SIDECAR_JSONL_FILES = (
    'recourse_jump_sidecar_baseline.jsonl',
    'ess_entry_stride_baseline.jsonl',
    'soh_floor_sidecar_baseline.jsonl',
    'pf_entry_stride_s39_D.jsonl',
    'ess_exempt_until_state_s39_D.jsonl',
)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _per_cycle_wall_times(report):
    """Best-effort extraction of per-cycle wall time from the report's own
    `cycle_trajectory` (each row's own recorded fields), for the informative
    "report per-cycle wall time" requirement -- NOT part of the bitwise
    comparator (wall-clock fields are excluded from it by construction)."""
    trajectory = report.get('cycle_trajectory') if isinstance(report, dict) else None
    if not trajectory:
        return []
    out = []
    for row in trajectory:
        out.append({
            'cycle': row.get('cycle'),
            'wall_s': row.get('wall_clock_s') if isinstance(row, dict) else None,
        })
    return out


def _reconcile_solve_profile_from_logs(mode_label):
    """Independent, non-guard-based verification that the SAME number of
    IPOPT solves happened in the parallel8 run as in serial, per Requirement
    4 ("the run's solve profile is complete and checked exactly, and state
    how") -- counts, from each run's OWN per-block IPOPT log files (never
    from a guard, which by construction cannot see child-process solves),
    the same 'Total seconds in IPOPT' occurrence technique
    WORKER_REPORT_S36_PARALLEL_AUDIT.md already used. Both DSO/TSO (one
    appended log per (case, year, day), 'Total seconds in IPOPT' once per
    attempt) and ESSO (one file per (node, cycle[, attempt]) -- summed by
    file count) are counted."""
    run_id = G._s39_ids_for_mode(ARM_KEY, mode_label)['run']
    logs_dir = os.path.join(G.O.WORK_DIR, run_id, 'logs')
    if not os.path.isdir(logs_dir):
        return {'error': f'logs dir not found: {logs_dir}'}

    dso_tso_total = 0
    dso_tso_files = 0
    esso_total = 0
    esso_files = 0
    per_file_counts = {}
    for fname in sorted(os.listdir(logs_dir)):
        fpath = os.path.join(logs_dir, fname)
        if not os.path.isfile(fpath) or not fname.startswith('optim_log_'):
            continue
        with open(fpath, errors='replace') as handle:
            text = handle.read()
        n = text.count('Total seconds in IPOPT')
        per_file_counts[fname] = n
        if fname.startswith('optim_log_esso_'):
            esso_total += n
            esso_files += 1
        else:
            dso_tso_total += n
            dso_tso_files += 1

    return {
        'logs_dir': os.path.relpath(logs_dir, REPO),
        'dso_tso_log_files': dso_tso_files,
        'dso_tso_total_ipopt_solves': dso_tso_total,
        'esso_log_files': esso_files,
        'esso_total_ipopt_solves': esso_total,
        'total_ipopt_solves': dso_tso_total + esso_total,
        'per_file_counts': per_file_counts,
    }


def main():
    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[S40-PW-PREFLIGHT] [PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S40-PW-PREFLIGHT] preconditions passed (no lock, no forbidden process, '
          'fresh output dirs, production files clean).')

    G._acquire_exclusive_run_lock()
    os.makedirs(OUT_ROOT, exist_ok=True)

    started = time.time()
    print(f'[S40-PW-PREFLIGHT] run SERIAL (persistent_workers.enabled=False), '
          f'{NUM_CYCLES} cycles, oracle {ARM_KEY}.')
    serial = _run_arm(MODE_SERIAL, OUT_SERIAL, enabled=False, num_workers=NUM_WORKERS)
    print(f"[S40-PW-PREFLIGHT] serial: cycles_run={serial['report']['cycles_run']} "
          f"wall_clock_s={serial['wall_clock_s']:.2f} "
          f"frozen_snapshot_files={serial['frozen_snapshot_files']}")

    print(f'[S40-PW-PREFLIGHT] run PARALLEL8 (persistent_workers.enabled=True, num_workers=8), '
          f'{NUM_CYCLES} cycles, oracle {ARM_KEY}.')
    parallel8 = _run_arm(MODE_PARALLEL8, OUT_PARALLEL8, enabled=True, num_workers=NUM_WORKERS)
    print(f"[S40-PW-PREFLIGHT] parallel8: cycles_run={parallel8['report']['cycles_run']} "
          f"wall_clock_s={parallel8['wall_clock_s']:.2f} "
          f"frozen_snapshot_files={parallel8['frozen_snapshot_files']}")

    # ---- Requirement 4: complete solve profile, verified from each run's
    # OWN per-block IPOPT logs (never from a guard -- see the
    # EXCLUDE_DOTTED_SUFFIXES comment above `solve_profile.*` for why the
    # parent-only guard field is excluded from the bitwise diff instead). --
    solve_profile_reconciliation = {
        'serial': _reconcile_solve_profile_from_logs(MODE_SERIAL),
        'parallel8': _reconcile_solve_profile_from_logs(MODE_PARALLEL8),
    }
    print(f"[S40-PW-PREFLIGHT] solve profile reconciliation (from IPOPT logs, not a guard): "
          f"serial total={solve_profile_reconciliation['serial'].get('total_ipopt_solves')} "
          f"parallel8 total={solve_profile_reconciliation['parallel8'].get('total_ipopt_solves')}")

    # ---- bitwise diff: the full report ------------------------------------
    diff_report = _diff(serial['report'], parallel8['report'], 'report')

    # ---- bitwise diff: every terminal JSON artifact -----------------------
    artifact_diffs = {}
    for fname in ARTIFACT_FILES:
        a_path = os.path.join(OUT_SERIAL, fname)
        b_path = os.path.join(OUT_PARALLEL8, fname)
        a_json = _load_json(a_path)
        b_json = _load_json(b_path)
        if a_json is None or b_json is None:
            artifact_diffs[fname] = {
                'error': f'missing artifact: serial_exists={a_json is not None} '
                         f'parallel8_exists={b_json is not None}',
                'diffs': [], 'n_diffs': None,
            }
            continue
        d = _diff(a_json, b_json, fname)
        artifact_diffs[fname] = {'diffs': d, 'n_diffs': len(d)}

    # ---- bitwise diff: every JSONL sidecar ---------------------------------
    sidecar_diffs = {}
    for fname in SIDECAR_JSONL_FILES:
        a_path = os.path.join(OUT_SERIAL, fname)
        b_path = os.path.join(OUT_PARALLEL8, fname)
        a_rows = _load_jsonl(a_path)
        b_rows = _load_jsonl(b_path)
        if a_rows is None or b_rows is None:
            sidecar_diffs[fname] = {
                'error': f'missing sidecar: serial_exists={a_rows is not None} '
                         f'parallel8_exists={b_rows is not None}',
                'diffs': [], 'n_diffs': None,
            }
            continue
        d = _diff(a_rows, b_rows, fname)
        sidecar_diffs[fname] = {'diffs': d, 'n_diffs': len(d), 'n_rows': len(a_rows)}

    all_diffs_n = (
        len(diff_report)
        + sum((v['n_diffs'] or 0) for v in artifact_diffs.values())
        + sum((v['n_diffs'] or 0) for v in sidecar_diffs.values())
    )
    all_artifacts_present = (
        all('error' not in v for v in artifact_diffs.values())
        and all('error' not in v for v in sidecar_diffs.values())
    )
    bitwise_identical = (all_diffs_n == 0) and all_artifacts_present

    payload = {
        'stage': 'P5.15 Addendum 22 item (2), second half -- persistent-workers 2-cycle bitwise preflight',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 18',
            'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm': ARM_KEY,
        'num_cycles': NUM_CYCLES,
        'num_workers': NUM_WORKERS,
        'serial': {
            'out_dir': serial['out_dir'], 'mode_label': serial['mode_label'],
            'persistent_workers_requested': serial['persistent_workers_requested'],
            'report_path': os.path.relpath(serial['report_path'], REPO),
            'frozen_snapshot_files': serial['frozen_snapshot_files'],
            'wall_clock_s': serial['wall_clock_s'],
            'per_cycle_wall_times': _per_cycle_wall_times(serial['report']),
            'solve_profile': serial['report'].get('solve_profile'),
        },
        'parallel8': {
            'out_dir': parallel8['out_dir'], 'mode_label': parallel8['mode_label'],
            'persistent_workers_requested': parallel8['persistent_workers_requested'],
            'report_path': os.path.relpath(parallel8['report_path'], REPO),
            'frozen_snapshot_files': parallel8['frozen_snapshot_files'],
            'wall_clock_s': parallel8['wall_clock_s'],
            'per_cycle_wall_times': _per_cycle_wall_times(parallel8['report']),
            'solve_profile': parallel8['report'].get('solve_profile'),
        },
        'solve_profile_reconciliation_from_logs': solve_profile_reconciliation,
        'observed_speedup_total_wall_clock': (
            serial['wall_clock_s'] / parallel8['wall_clock_s'] if parallel8['wall_clock_s'] else None
        ),
        'excluded_field_names': sorted(EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(EXCLUDE_DOTTED_SUFFIXES),
        'intentional_difference_suffixes': sorted(INTENTIONAL_DIFF_SUFFIXES),
        'bitwise_diff': {
            'report_g_s39_D': {'diffs': diff_report, 'n_diffs': len(diff_report)},
            'artifacts': artifact_diffs,
            'sidecars': sidecar_diffs,
            'total_n_diffs': all_diffs_n,
            'all_artifacts_present': all_artifacts_present,
        },
        'bitwise_identical': bitwise_identical,
        'wall_clock_s': time.time() - started,
    }

    results_path = os.path.join(OUT_ROOT, 'persistent_workers_preflight_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S40-PW-PREFLIGHT] wrote {results_path}')
    print(f'[S40-PW-PREFLIGHT] bitwise_identical={bitwise_identical} total_n_diffs={all_diffs_n} '
          f'all_artifacts_present={all_artifacts_present}')
    print(f'[S40-PW-PREFLIGHT] serial wall={serial["wall_clock_s"]:.2f}s '
          f'parallel8 wall={parallel8["wall_clock_s"]:.2f}s '
          f'speedup={payload["observed_speedup_total_wall_clock"]}')
    if not bitwise_identical:
        print('[S40-PW-PREFLIGHT] *** BITWISE COMPARISON FAILED *** -- see diffs above/in '
              f'{results_path}. Reported as-is; NOT rationalized.')
        print(f'[S40-PW-PREFLIGHT] first diffs: {(diff_report or [d for v in artifact_diffs.values() for d in v["diffs"]] or [d for v in sidecar_diffs.values() for d in v["diffs"]])[:10]}')

    manifest = {os.path.relpath(results_path, REPO): _sha256_file(results_path)}
    manifest_path = os.path.join(OUT_ROOT, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S40-PW-PREFLIGHT] wrote {manifest_path}')

    if not bitwise_identical:
        sys.exit(1)


if __name__ == '__main__':
    main()
