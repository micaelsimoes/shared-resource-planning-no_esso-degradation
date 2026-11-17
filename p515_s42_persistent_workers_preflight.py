"""
P5.15 Addendum 23/24, Step 3.6 persistent-worker BOUNDED task, item 3, gate
half 2 -- the two-cycle BITWISE preflight of the persistent-worker pool
(`admm_persistent_workers.PersistentWorkerPool`, `admm_parameters.
persistent_workers`) against the existing serial ADMM path, on the ORACLE
configuration (`s39_D`), WITH the item 2/3 fixes now in production: the
bound-restore fix (`network.capture_block_mutable_state`/
`apply_block_mutable_state`, raw `._lb`/`._ub` capture + `skip_validation=
True`) and the DSO node-7 lightweight capture (`admm_parameters.
dso_snapshot_capture_mode`, default 'lightweight').

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 23 "Step 3.6 persistent
workers"; frozen spec v12 item3_persistent_workers_bounded_task; frozen spec
v13 data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json.

Modeled directly on `p515_s40_persistent_workers_preflight.py` (the WORKER_
REPORT_S40_PERSISTENT_WORKERS.md gate that first found the
`esso_models_pickle.bytes` divergence and the W1001 warnings) -- the
differences are: (a) fresh output root under `P515S42/persistent_workers_
task/`, never reusing a P515S40 path; (b) `esso_models_pickle.bytes` is
(still) NOT excluded from the bitwise diff -- this gate's entire purpose is
to confirm the fix makes it identical, so excluding it would silently
remove the one field this task exists to fix; (c) per-arm W1001 warning
counting via a temporary OS-level fd 1/2 redirect around each arm (workers
are spawned, via `multiprocessing`'s `spawn` context, INSIDE that redirect
window, so they inherit the redirected fds and their own Pyomo `W1001`
log lines -- emitted through the stdlib `logging` module, `pyomo.core`
logger, `logger.warning(..., extra={'id': 'W1001'})` -- land in the per-arm
capture file, never only in the master launch log).

======================================================================
EXACT LAUNCH COMMAND
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s42_persistent_workers_preflight.py \\
        > data/SRP1/Results/P515S42/persistent_workers_task/pw_preflight_launch.log 2>&1

Run attached, alone (no `screen`/`nohup`/backgrounding), both streams
captured via the shell redirection above.

======================================================================
PRECONDITIONS (checked before anything is written; refuses loudly otherwise)
======================================================================
  1. `.p515_g_gate.lock` does not already exist.
  2. No OTHER process (excluding this process's own ancestor chain) matches
     `p515_g_g1_g4_admm_gates.py`, `p515_s39_`, `p515_s40_`, `p515_s42_`, or
     `p515_s36_step36_timing_run.py` in `ps aux`.
  3. Neither output root exists yet (write-once).
  4. The production files this task touches are clean in git.

======================================================================
WHAT IS COMPARED, BITWISE, BETWEEN THE TWO RUNS
======================================================================
Same artifact set and EXCLUDE_KEY_NAMES/EXCLUDE_DOTTED_SUFFIXES convention as
`p515_s40_persistent_workers_preflight.py` (commit `8f5cff48`'s fixed
comparator) -- `esso_models_pickle.bytes` INCLUDED.
"""

import glob
import hashlib
import json
import os
import re
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
OUT_ROOT = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S42', 'persistent_workers_task', f'pw_preflight{_RUN_SUFFIX}')
OUT_SERIAL = os.path.join(OUT_ROOT, 'serial')
OUT_PARALLEL8 = os.path.join(OUT_ROOT, 'parallel8')

MODE_SERIAL = f'preflight_s42pwpre{_RUN_SUFFIX}_serial'
MODE_PARALLEL8 = f'preflight_s42pwpre{_RUN_SUFFIX}_parallel8'

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = (
    'p515_g_g1_g4_admm_gates.py',
    'p515_s39_',
    'p515_s40_',
    'p515_s42_',
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

# `esso_models_pickle.bytes` is DELIBERATELY not in this set -- see module
# docstring. Only `.path` (an absolute, per-run path) is excluded.
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
    # Requirement 4 (WORKER_REPORT_S40_PERSISTENT_WORKERS.md): the parent's
    # OWN embedded SolveProfileGuard cannot see solves inside a spawned
    # worker process, by construction -- this task's own answer
    # (`PersistentWorkerPool.total_child_guard_counts()`) is verified
    # SEPARATELY, below, from each run's own per-block IPOPT logs.
    'solve_profile.identity_holds',
    'solve_profile.observed.permitted_solve',
    'solve_profile.observed.permitted_exec',
}
INTENTIONAL_DIFF_SUFFIXES = {
    'rule_eleven_checklist.s42_persistent_workers_requested',
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
            report.setdefault('rule_eleven_checklist', {})['s42_persistent_workers_override_applied'] = applied
            report['rule_eleven_checklist']['s42_persistent_workers_requested'] = {
                'enabled': enabled, 'num_workers': num_workers,
            }
            if not applied:
                raise RuntimeError(
                    f'S42 persistent-workers preflight: override to enabled={enabled}, '
                    f'num_workers={num_workers} did not take effect')
        return hook

    G._s39_configure_hook = wrapped_factory
    try:
        yield
    finally:
        G._s39_configure_hook = inner_factory


@contextmanager
def _capture_fd_output(path):
    """Temporarily redirects OS-level fd 1 and 2 (stdout/stderr) to `path`.
    A spawned `multiprocessing` child inherits whatever fd 1/2 point to AT
    PROCESS-CREATION TIME -- since worker processes are spawned INSIDE this
    context (deep inside `G.run_s39_arm`), their own Pyomo `logging.
    lastResort` warning output (which writes to the process's own
    `sys.stderr`, itself backed by fd 2) lands in `path` too, not only the
    parent's own prints. Restored exactly afterward; the outer shell
    redirection (`> launch.log 2>&1`) resumes receiving output once this
    context exits."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    out_file = open(path, 'w')
    try:
        os.dup2(out_file.fileno(), 1)
        os.dup2(out_file.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        out_file.close()


def _count_w1001(path):
    if not os.path.exists(path):
        return None
    count = 0
    with open(path, errors='replace') as handle:
        for line in handle:
            if 'W1001' in line:
                count += 1
    return count


def _list_frozen_snapshot_files(out_dir):
    frozen_dir = os.path.join(out_dir, 'results', 'FrozenSMOPF')
    if not os.path.isdir(frozen_dir):
        return []
    return sorted(
        os.path.relpath(p, REPO)
        for p in glob.glob(os.path.join(frozen_dir, '**', '*'), recursive=True)
        if os.path.isfile(p)
    )


def _run_arm(mode_label, out_dir, enabled, num_workers, stdouterr_path):
    with _persistent_workers_hook_override(enabled, num_workers):
        started = time.time()
        with _capture_fd_output(stdouterr_path):
            report, report_path = G.run_s39_arm(
                ARM_KEY, num_max_iters_override=NUM_CYCLES, output_root_override=out_dir,
                mode_label_override=mode_label)
        wall = time.time() - started
    frozen_files = _list_frozen_snapshot_files(out_dir)
    w1001_count = _count_w1001(stdouterr_path)
    return {
        'report': report,
        'report_path': report_path,
        'frozen_snapshot_files': frozen_files,
        'persistent_workers_requested': {'enabled': enabled, 'num_workers': num_workers},
        'mode_label': mode_label,
        'out_dir': os.path.relpath(out_dir, REPO),
        'wall_clock_s': wall,
        'stdouterr_path': os.path.relpath(stdouterr_path, REPO),
        'w1001_warning_count': w1001_count,
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


def _per_cycle_wall_from_log(stdouterr_path):
    """Best-effort per-cycle wall time, parsed from the run's own
    `[INFO] \\t - Iteration N: X.XX s` print (shared_resources_planning.py's
    ADMM loop) -- NOT part of the bitwise comparator (wall-clock fields are
    excluded from it by construction), purely informative."""
    if not os.path.exists(stdouterr_path):
        return []
    pattern = re.compile(r'Iteration (\d+): ([\d.]+) s')
    out = []
    with open(stdouterr_path, errors='replace') as handle:
        for line in handle:
            m = pattern.search(line)
            if m:
                out.append({'cycle': int(m.group(1)), 'wall_s': float(m.group(2))})
    return out


def main():
    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[S42-PW-PREFLIGHT] [PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S42-PW-PREFLIGHT] preconditions passed (no lock, no forbidden process, '
          'fresh output dirs, production files clean).')

    G._acquire_exclusive_run_lock()
    os.makedirs(OUT_ROOT, exist_ok=True)

    started = time.time()
    print(f'[S42-PW-PREFLIGHT] run SERIAL (persistent_workers.enabled=False), '
          f'{NUM_CYCLES} cycles, oracle {ARM_KEY}.')
    serial = _run_arm(MODE_SERIAL, OUT_SERIAL, enabled=False, num_workers=NUM_WORKERS,
                       stdouterr_path=os.path.join(OUT_ROOT, f'{MODE_SERIAL}_stdouterr.log'))
    print(f"[S42-PW-PREFLIGHT] serial: cycles_run={serial['report']['cycles_run']} "
          f"wall_clock_s={serial['wall_clock_s']:.2f} "
          f"w1001_warning_count={serial['w1001_warning_count']} "
          f"frozen_snapshot_files={serial['frozen_snapshot_files']}")

    print(f'[S42-PW-PREFLIGHT] run PARALLEL8 (persistent_workers.enabled=True, num_workers=8), '
          f'{NUM_CYCLES} cycles, oracle {ARM_KEY}.')
    parallel8 = _run_arm(MODE_PARALLEL8, OUT_PARALLEL8, enabled=True, num_workers=NUM_WORKERS,
                          stdouterr_path=os.path.join(OUT_ROOT, f'{MODE_PARALLEL8}_stdouterr.log'))
    print(f"[S42-PW-PREFLIGHT] parallel8: cycles_run={parallel8['report']['cycles_run']} "
          f"wall_clock_s={parallel8['wall_clock_s']:.2f} "
          f"w1001_warning_count={parallel8['w1001_warning_count']} "
          f"frozen_snapshot_files={parallel8['frozen_snapshot_files']}")

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

    esso_bytes_field_present = any(
        d['field'].endswith('esso_models_pickle.bytes') for d in diff_report
    )

    payload = {
        'stage': 'P5.15 Addendum 23/24, Step 3.6 persistent-worker bounded task item 3 -- '
                 'persistent-workers 2-cycle bitwise preflight, WITH the item 2/3 fixes',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 23',
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json item3_persistent_workers_bounded_task',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
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
            'per_cycle_wall_times_from_log': _per_cycle_wall_from_log(
                os.path.join(REPO, serial['stdouterr_path'])),
            'stdouterr_path': serial['stdouterr_path'],
            'w1001_warning_count': serial['w1001_warning_count'],
        },
        'parallel8': {
            'out_dir': parallel8['out_dir'], 'mode_label': parallel8['mode_label'],
            'persistent_workers_requested': parallel8['persistent_workers_requested'],
            'report_path': os.path.relpath(parallel8['report_path'], REPO),
            'frozen_snapshot_files': parallel8['frozen_snapshot_files'],
            'wall_clock_s': parallel8['wall_clock_s'],
            'per_cycle_wall_times_from_log': _per_cycle_wall_from_log(
                os.path.join(REPO, parallel8['stdouterr_path'])),
            'stdouterr_path': parallel8['stdouterr_path'],
            'w1001_warning_count': parallel8['w1001_warning_count'],
        },
        'observed_speedup_total_wall_clock': (
            serial['wall_clock_s'] / parallel8['wall_clock_s'] if parallel8['wall_clock_s'] else None
        ),
        'esso_models_pickle_bytes_field_included_in_diff': True,
        'esso_models_pickle_bytes_diverges': esso_bytes_field_present,
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
    print(f'[S42-PW-PREFLIGHT] wrote {results_path}')
    print(f'[S42-PW-PREFLIGHT] bitwise_identical={bitwise_identical} total_n_diffs={all_diffs_n} '
          f'all_artifacts_present={all_artifacts_present} '
          f'esso_models_pickle_bytes_diverges={esso_bytes_field_present}')
    print(f'[S42-PW-PREFLIGHT] serial wall={serial["wall_clock_s"]:.2f}s w1001={serial["w1001_warning_count"]} '
          f'parallel8 wall={parallel8["wall_clock_s"]:.2f}s w1001={parallel8["w1001_warning_count"]} '
          f'speedup={payload["observed_speedup_total_wall_clock"]}')
    if not bitwise_identical:
        print('[S42-PW-PREFLIGHT] *** BITWISE COMPARISON FAILED *** -- see diffs above/in '
              f'{results_path}. Reported as-is; NOT rationalized.')
        first_diffs = (diff_report or [d for v in artifact_diffs.values() for d in v['diffs']]
                       or [d for v in sidecar_diffs.values() for d in v['diffs']])
        print(f'[S42-PW-PREFLIGHT] first diffs: {first_diffs[:10]}')

    manifest = {os.path.relpath(results_path, REPO): _sha256_file(results_path)}
    manifest_path = os.path.join(OUT_ROOT, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-PW-PREFLIGHT] wrote {manifest_path}')

    if not bitwise_identical:
        sys.exit(1)


if __name__ == '__main__':
    main()
