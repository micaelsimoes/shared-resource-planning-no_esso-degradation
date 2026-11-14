"""
P5.15 Addendum 23/24, Step 3.6 persistent-worker BOUNDED task, item 1 --
5-cycle per-phase timing on the PARALLEL path (8 workers), oracle
configuration (`s39_D`), with items 2/3's fixes already in production.

NO algorithmic change: every timing hook below is a harness-side monkeypatch
of an already-existing function/bound-method reference, installed in the
PARENT process only, restored in a `finally`. It never edits a call site's
arguments or return value -- every wrapped call still calls straight through
to the real, unmodified production code.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 23 "Step 3.6 persistent
workers"; frozen spec v12 item3_persistent_workers_bounded_task ("5-cycle
per-phase timing on the parallel path").

======================================================================
WHAT IS MEASURED, PER CYCLE
======================================================================
  - `wall_s`: this cycle's total wall time, parsed from the run's own
    `[INFO] \\t - Iteration N: X.XX s` print (shared_resources_planning.py's
    ADMM loop) -- the SAME technique the timing gate's own launch-log
    parsing already uses elsewhere in this programme.
  - `param_update_s` / `param_update_calls`: cumulative time inside the
    PARENT-side per-block mutation functions (`_apply_dso_block_params`,
    `_apply_tso_block_params`, `_apply_esso_block_params`) -- narrow
    re-statements of the existing serial per-cycle Param-update loops,
    applied to the parent's OWN resident copy before it is shipped.
  - `capture_s` / `capture_calls`: cumulative time inside
    `capture_block_state_for_ipc` (the generic, clone-free snapshot of
    "everything that changed", taken right after the mutation above, before
    the state is pickled and sent to the owning worker).
  - `dispatch_<kind>_s` (`dso`/`tso`/`esso`): cumulative time inside
    `PersistentWorkerPool._dispatch_stage` for that stage -- covers
    EVERYTHING from `task_queue.put(...)` (serialize + enqueue) through
    `result_q.get(...)` (blocking wait, dominated by the worker's own solve
    + its own state-apply + its own result serialize) to result assembly.
    This bucket is NOT further split into "pure IPC" vs "worker execution"
    without editing worker-side code (out of this task's "no algorithmic
    change" scope) -- reported as one bucket, honestly labeled.
  - `ipopt_<agent>_s` (`dso`/`tso`/`esso`), POST-HOC: summed "Total seconds
    in IPOPT" values parsed directly from each worker-written per-block
    IPOPT log file under `<out_dir>/results/Logs/`, attributed to a cycle by
    POSITION within that file (see `_attribute_ipopt_entries_to_cycles`) --
    this is IPOPT time "as seen inside workers", the one quantity no
    parent-side wrap can see at all (the solve itself runs in a spawned
    process).
  - `serial_other_s` = `wall_s` - (`param_update_s` + `capture_s` +
    sum(`dispatch_<kind>_s`)) -- "whatever else the parent does serially"
    (proximal-centre update, convergence/residual bookkeeping, penalty
    updates, Python/GC overhead between stages) -- the residual the
    task exists to locate.

======================================================================
EXACT LAUNCH COMMAND
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s42_persistent_workers_timing_5cyc.py \\
        > data/SRP1/Results/P515S42/persistent_workers_task/timing_5cyc_launch.log 2>&1

Run attached, alone, both streams captured.

PRECONDITIONS: same as `p515_s42_persistent_workers_preflight.py` (lock
absent, no forbidden process, fresh output root, production files clean).
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
import admm_persistent_workers as apw  # noqa: E402
from p515_s40_persistent_workers_preflight import _persistent_workers_hook_override  # noqa: E402

ARM_KEY = 's39_D'
NUM_CYCLES = 5
NUM_WORKERS = 8

_RUN_SUFFIX = ''
for _arg in sys.argv[1:]:
    if not _arg.startswith('--'):
        _RUN_SUFFIX = '_' + _arg.strip('_')
        break
OUT_ROOT = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S42', 'persistent_workers_task', f'timing_5cyc{_RUN_SUFFIX}')
OUT_DIR = os.path.join(OUT_ROOT, 'parallel8')
MODE_LABEL = f'preflight_s42timing5{_RUN_SUFFIX}_parallel8'

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

    if os.path.exists(OUT_DIR):
        failures.append(f'output directory already exists (write-once): {OUT_DIR}')

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


# ---------------------------------------------------------------------------
# Harness-side timing hooks -- parent process only, monkeypatched module-
# level function objects / one class-level bound method, restored in
# `finally`. Every wrapper calls straight through; none changes behaviour.
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.current_cycle = None
        self.per_cycle = {}

    def bucket(self, cycle):
        return self.per_cycle.setdefault(cycle, {
            'param_update_s': 0.0, 'param_update_calls': 0,
            'capture_s': 0.0, 'capture_calls': 0,
            'dispatch_dso_s': 0.0, 'dispatch_tso_s': 0.0, 'dispatch_esso_s': 0.0,
            'dispatch_dso_calls': 0, 'dispatch_tso_calls': 0, 'dispatch_esso_calls': 0,
        })

    def add(self, cycle, key, seconds):
        b = self.bucket(cycle)
        b[f'{key}_s'] += seconds
        b[f'{key}_calls'] += 1


@contextmanager
def _install_timing_hooks(recorder):
    orig_run_dso_stage = apw.PersistentWorkerPool.run_dso_stage
    orig_run_tso_stage = apw.PersistentWorkerPool.run_tso_stage
    orig_run_esso_stage = apw.PersistentWorkerPool.run_esso_stage
    orig_dispatch_stage = apw.PersistentWorkerPool._dispatch_stage
    orig_capture = apw.capture_block_state_for_ipc
    orig_apply_dso = apw._apply_dso_block_params
    orig_apply_tso = apw._apply_tso_block_params
    orig_apply_esso = apw._apply_esso_block_params

    def make_stage_wrapper(orig_fn):
        def wrapper(self, *args, **kwargs):
            cycle = kwargs.get('cycle')
            recorder.current_cycle = cycle
            recorder.bucket(cycle)
            return orig_fn(self, *args, **kwargs)
        return wrapper

    def wrapped_dispatch_stage(self, kind, canonical_keys, cycle, from_warm_start, captured_states):
        t0 = time.perf_counter()
        try:
            return orig_dispatch_stage(self, kind, canonical_keys, cycle, from_warm_start, captured_states)
        finally:
            recorder.add(cycle, f'dispatch_{kind}', time.perf_counter() - t0)

    def wrapped_capture(model):
        t0 = time.perf_counter()
        try:
            return orig_capture(model)
        finally:
            recorder.add(recorder.current_cycle, 'capture', time.perf_counter() - t0)

    def make_apply_wrapper(orig_fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_fn(*args, **kwargs)
            finally:
                recorder.add(recorder.current_cycle, 'param_update', time.perf_counter() - t0)
        return wrapper

    apw.PersistentWorkerPool.run_dso_stage = make_stage_wrapper(orig_run_dso_stage)
    apw.PersistentWorkerPool.run_tso_stage = make_stage_wrapper(orig_run_tso_stage)
    apw.PersistentWorkerPool.run_esso_stage = make_stage_wrapper(orig_run_esso_stage)
    apw.PersistentWorkerPool._dispatch_stage = wrapped_dispatch_stage
    apw.capture_block_state_for_ipc = wrapped_capture
    apw._apply_dso_block_params = make_apply_wrapper(orig_apply_dso)
    apw._apply_tso_block_params = make_apply_wrapper(orig_apply_tso)
    apw._apply_esso_block_params = make_apply_wrapper(orig_apply_esso)
    try:
        yield recorder
    finally:
        apw.PersistentWorkerPool.run_dso_stage = orig_run_dso_stage
        apw.PersistentWorkerPool.run_tso_stage = orig_run_tso_stage
        apw.PersistentWorkerPool.run_esso_stage = orig_run_esso_stage
        apw.PersistentWorkerPool._dispatch_stage = orig_dispatch_stage
        apw.capture_block_state_for_ipc = orig_capture
        apw._apply_dso_block_params = orig_apply_dso
        apw._apply_tso_block_params = orig_apply_tso
        apw._apply_esso_block_params = orig_apply_esso


def _per_cycle_wall_from_log(path):
    pattern = re.compile(r'Iteration (\d+): ([\d.]+) s')
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors='replace') as handle:
        for line in handle:
            m = pattern.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


IPOPT_TIME_RE = re.compile(r'Total seconds in IPOPT\s*=\s*([\d.eE+-]+)')


def _agent_for_logfile(fname):
    if fname.startswith('optim_log_esso_'):
        return 'esso'
    # TSO uses case9*; every other DSO case name in this case file is
    # case33_*. Read, not assumed: the actual case-name prefixes present in
    # the log directory are reported alongside this attribution.
    if fname.startswith('optim_log_case9'):
        return 'tso'
    return 'dso'


ESSO_CYCLE_FNAME_RE = re.compile(r'optim_log_esso_node\d+_cycle(\d+)')
ESSO_INIT_FNAME_RE = re.compile(r'optim_log_esso_node\d+_init')


def _parse_ipopt_logs(logs_dir, num_cycles):
    """DSO/TSO: every per-block log ACCUMULATES one 'Total seconds in IPOPT'
    entry per solve ATTEMPT, in chronological (append) order -- the very
    first entry per file is the PRE-LOOP initialization solve (runs in the
    PARENT, before the persistent-worker pool is even constructed -- see
    `_run_operational_planning`'s own `tso_pristine_base`/pool-construction
    sequencing, both AFTER the initial model-construction solve); every
    entry after that is one of the `num_cycles` ADMM cycles, each run
    INSIDE a worker. This attribution is verified per file (entry count
    reported), not assumed silently: a file with more than `num_cycles + 1`
    entries (a retry) is flagged, not forced into the num_cycles buckets.

    ESSO: read directly, NOT assumed to follow the same convention --
    `shared_energy_storage_data.py` writes one SEPARATE log file per
    (node, cycle) (`optim_log_esso_node{N}_cycle{NNN}.txt` /
    `..._init.txt`), each holding exactly one 'Total seconds in IPOPT'
    entry; the cycle number is read from the filename itself, never
    inferred positionally."""
    per_agent_per_cycle = {'dso': {}, 'tso': {}, 'esso': {}}
    per_file_entry_counts = {}
    anomalous_files = []
    if not os.path.isdir(logs_dir):
        return per_agent_per_cycle, per_file_entry_counts, anomalous_files
    for fname in sorted(os.listdir(logs_dir)):
        fpath = os.path.join(logs_dir, fname)
        if not os.path.isfile(fpath) or not fname.startswith('optim_log_'):
            continue
        with open(fpath, errors='replace') as handle:
            text = handle.read()
        values = [float(m.group(1)) for m in IPOPT_TIME_RE.finditer(text)]
        per_file_entry_counts[fname] = len(values)
        agent = _agent_for_logfile(fname)

        if agent == 'esso':
            m_cycle = ESSO_CYCLE_FNAME_RE.search(fname)
            m_init = ESSO_INIT_FNAME_RE.search(fname)
            if len(values) != 1 or (m_cycle is None and m_init is None):
                anomalous_files.append({'file': fname, 'agent': agent, 'n_entries': len(values),
                                         'expected': 1, 'note': 'esso per-cycle-file convention'})
                continue
            if m_init is not None:
                continue  # pre-loop init solve, not one of the num_cycles ADMM cycles.
            cycle = int(m_cycle.group(1))
            per_agent_per_cycle['esso'][cycle] = per_agent_per_cycle['esso'].get(cycle, 0.0) + values[0]
            continue

        if len(values) != num_cycles + 1:
            anomalous_files.append({'file': fname, 'agent': agent, 'n_entries': len(values),
                                     'expected': num_cycles + 1})
            # Still attribute the LAST num_cycles entries (best effort,
            # reported as anomalous above, not silently trusted).
            cycle_values = values[-num_cycles:] if len(values) >= num_cycles else values
            offset = num_cycles - len(cycle_values)
        else:
            cycle_values = values[1:]
            offset = 0
        for i, v in enumerate(cycle_values):
            cycle = i + 1 + offset
            per_agent_per_cycle[agent][cycle] = per_agent_per_cycle[agent].get(cycle, 0.0) + v
    return per_agent_per_cycle, per_file_entry_counts, anomalous_files


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[S42-TIMING-5CYC] [PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S42-TIMING-5CYC] preconditions passed (no lock, no forbidden process, '
          'fresh output dir, production files clean).')

    G._acquire_exclusive_run_lock()
    os.makedirs(OUT_ROOT, exist_ok=True)

    stdouterr_path = os.path.join(OUT_ROOT, f'{MODE_LABEL}_stdouterr.log')
    recorder = _Recorder()
    started = time.time()
    with _persistent_workers_hook_override(True, NUM_WORKERS), _install_timing_hooks(recorder):
        # Tee stdout/stderr to a file too (for the Iteration-wall-time parse)
        # while still letting this process's own prints reach the launch log.
        import contextlib
        tee_started = time.time()
        with open(stdouterr_path, 'w') as tee_file:
            class _Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data):
                    for s in self.streams:
                        s.write(data)

                def flush(self):
                    for s in self.streams:
                        s.flush()

            real_stdout, real_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(real_stdout, tee_file)
            sys.stderr = _Tee(real_stderr, tee_file)
            try:
                report, report_path = G.run_s39_arm(
                    ARM_KEY, num_max_iters_override=NUM_CYCLES, output_root_override=OUT_DIR,
                    mode_label_override=MODE_LABEL)
            finally:
                sys.stdout, sys.stderr = real_stdout, real_stderr
    wall_total = time.time() - started

    per_cycle_wall = _per_cycle_wall_from_log(stdouterr_path)
    run_id = G._s39_ids_for_mode(ARM_KEY, MODE_LABEL)['run']
    logs_dir = os.path.join(G.O.WORK_DIR, run_id, 'logs')
    ipopt_per_agent_per_cycle, per_file_entry_counts, anomalous_files = _parse_ipopt_logs(logs_dir, NUM_CYCLES)

    per_cycle_report = []
    for cycle in range(1, NUM_CYCLES + 1):
        b = recorder.per_cycle.get(cycle, {})
        wall_s = per_cycle_wall.get(cycle)
        dispatch_total = b.get('dispatch_dso_s', 0.0) + b.get('dispatch_tso_s', 0.0) + b.get('dispatch_esso_s', 0.0)
        parent_state_sync = b.get('param_update_s', 0.0) + b.get('capture_s', 0.0)
        serial_other = (wall_s - parent_state_sync - dispatch_total) if wall_s is not None else None
        row = {
            'cycle': cycle,
            'wall_s': wall_s,
            'param_update_s': b.get('param_update_s', 0.0),
            'param_update_calls': b.get('param_update_calls', 0),
            'capture_s': b.get('capture_s', 0.0),
            'capture_calls': b.get('capture_calls', 0),
            'parent_state_sync_s': parent_state_sync,
            'dispatch_dso_s': b.get('dispatch_dso_s', 0.0),
            'dispatch_tso_s': b.get('dispatch_tso_s', 0.0),
            'dispatch_esso_s': b.get('dispatch_esso_s', 0.0),
            'dispatch_total_s': dispatch_total,
            'ipopt_dso_s': ipopt_per_agent_per_cycle['dso'].get(cycle),
            'ipopt_tso_s': ipopt_per_agent_per_cycle['tso'].get(cycle),
            'ipopt_esso_s': ipopt_per_agent_per_cycle['esso'].get(cycle),
            'serial_other_s': serial_other,
        }
        row['ipopt_total_s'] = sum(
            v for v in (row['ipopt_dso_s'], row['ipopt_tso_s'], row['ipopt_esso_s']) if v is not None
        ) or None
        row['dispatch_minus_ipopt_s'] = (
            (dispatch_total - row['ipopt_total_s']) if row['ipopt_total_s'] is not None else None
        )
        per_cycle_report.append(row)

    def _sum(key):
        vals = [r[key] for r in per_cycle_report if r.get(key) is not None]
        return sum(vals) if vals else None

    aggregate = {
        'wall_total_s': _sum('wall_s'),
        'param_update_total_s': _sum('param_update_s'),
        'capture_total_s': _sum('capture_s'),
        'parent_state_sync_total_s': _sum('parent_state_sync_s'),
        'dispatch_total_s': _sum('dispatch_total_s'),
        'ipopt_total_s': _sum('ipopt_total_s'),
        'dispatch_minus_ipopt_total_s': _sum('dispatch_minus_ipopt_s'),
        'serial_other_total_s': _sum('serial_other_s'),
    }

    payload = {
        'stage': 'P5.15 Addendum 23/24, Step 3.6 persistent-worker bounded task item 1 -- '
                 '5-cycle per-phase timing, parallel path (8 workers)',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 23',
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json item3_persistent_workers_bounded_task',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm': ARM_KEY,
        'num_cycles': NUM_CYCLES,
        'num_workers': NUM_WORKERS,
        'out_dir': os.path.relpath(OUT_DIR, REPO),
        'report_path': os.path.relpath(report_path, REPO),
        'stdouterr_path': os.path.relpath(stdouterr_path, REPO),
        'logs_dir': os.path.relpath(logs_dir, REPO),
        'per_cycle': per_cycle_report,
        'aggregate': aggregate,
        'ipopt_log_entry_counts_per_file': per_file_entry_counts,
        'ipopt_log_anomalous_files': anomalous_files,
        'cycles_run': report.get('cycles_run'),
        'local_solve_failures': report.get('local_solve_failures'),
        'wall_clock_s_total_script': wall_total,
    }

    results_path = os.path.join(OUT_ROOT, 'timing_5cyc_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S42-TIMING-5CYC] wrote {results_path}')
    print(f'[S42-TIMING-5CYC] aggregate={aggregate}')
    print(f'[S42-TIMING-5CYC] ipopt_log_anomalous_files={anomalous_files}')

    manifest = {os.path.relpath(results_path, REPO): _sha256_file(results_path)}
    manifest_path = os.path.join(OUT_ROOT, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-TIMING-5CYC] wrote {manifest_path}')


if __name__ == '__main__':
    main()
