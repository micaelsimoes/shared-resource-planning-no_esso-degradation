"""
P5.15 Step 3.6, Worker task W3 -- the two-cycle recorder-off / recorder-on
measurement entry point (design `P5_15_STEP36_TIMING_DESIGN.md` §2.4/§5).

*** DO NOT RUN. The Planner launches this script; this task explicitly
    forbids executing it (the machine is reserved for a numerical campaign). ***

Reuses the campaign harness's OWN run machinery, read-only
(`p515_g_g1_g4_admm_gates.py`, imported as `G`, never edited -- another
Worker is editing that file concurrently): `G.run_admm_arm`,
`G._acquire_exclusive_run_lock`, `G.assert_s35ref_capture_paths`,
`G.s35ref_capture_hooks`, `G.write_boyd_terminal_s35ref`, `G.O`
(`p56a_oracle`) -- exactly the same functions the committed `s35ref` (run 1)
CLI branch (`p515_g_g1_g4_admm_gates.py`, `elif gate == 's35ref':`) calls, in
the SAME order, with the SAME configuration class (k_override=None,
apply_rho=False -- case-file rho in force, N.RHO NOT applied,
full_diagnostics_in_rows=True, investment_map=None -- uniform N.S_INV/N.E_INV
across active nodes), the ONLY deliberate difference being `num_max_iters_override
=2` (a two-cycle preflight, design §5, not the certified 500-cycle run) and
fresh eval ids / output roots so nothing committed is ever touched.

======================================================================
WHAT THIS SCRIPT DOES (when the Planner runs it)
======================================================================
  1. Precondition checks (all must pass BEFORE anything is written):
       a. `.p515_g_gate.lock` does not already exist.
       b. no OTHER `p515_g_g1_g4_admm_gates.py` process is alive (`ps aux`
          scan, excluding this script's own PID).
       c. neither output directory
          (`data/SRP1/Results/P515S36/step36_timing/{off,on}/`) exists yet.
       d. every production file this instrumentation reads (NOT edits) is
          clean in git (`git status --porcelain` on the exact file list in
          `_PRODUCTION_FILES_TO_CHECK_CLEAN`) -- a defensive check that the
          measurement is against the SAME code the wrap-point citations in
          `WORKER_REPORT_S36_TIMING.md` describe, not a mid-edit tree.
  2. Acquires the campaign harness's OWN exclusive run lock
     (`G._acquire_exclusive_run_lock()`, `.p515_g_gate.lock`, `O_CREAT|O_EXCL`
     -- reused verbatim, not re-implemented, so its semantics are identical
     by construction) -- refuses to run if another copy of
     `p515_g_g1_g4_admm_gates.py` (or this script) already holds it.
  3. Run OFF: two ADMM cycles, cold, recorder NOT installed -- byte-for-byte
     today's production path (design §2.2: an unmodified call, since this
     harness-side deviation never threads a `timing_recorder` kwarg into
     production AT ALL; "OFF" here means this script's OWN
     `p515_s36_step36_timing.recorder_installed(...)` context manager is
     simply not entered for this run).
  4. Run ON: the SAME two cycles, cold, with
     `p515_s36_step36_timing.recorder_installed(recorder, inject_report_timing=True)`
     wrapped AROUND the `G.run_admm_arm(...)` call (i.e. OUTSIDE
     `run_admm_arm`'s own `SolveProfileGuard.install()/uninstall()` pair --
     LIFO nesting, see the module docstring of `p515_s36_step36_timing.py`,
     "Composability with SolveProfileGuard"). `inject_report_timing=True`
     is the design §2.5 one-off cross-check (Pyomo's own `report_timing=True`
     print output), used for EXACTLY this one designated run, never the
     permanent mechanism.
  5. Diffs the two runs BITWISE (design §2.4) on: per-cycle recourse, every
     Boyd residual/ratio field `A.cycle_row` + the raw `admm_diagnostics`
     entries carry (`full_diagnostics_in_rows=True` puts every field in
     `report['cycle_trajectory']`), rho/gamma trajectories (`rho_*_before`,
     `rho_*_after`, `rho_*_action`, `gamma_*_before`, `gamma_*_after`, all
     already in the raw diagnostics rows), the SoH floor-multiplier sidecar
     (`s35ref_capture_hooks`' own artifact, one file per run), and the ESSO
     complementarity detector (`report['esso_complementarity_diagnostics_by_round']`).
     A non-bitwise diff means the instrumentation was wired incorrectly
     (design §2.4) -- NOT that timing measurement is inherently risky --
     and the run's timing analysis is NOT to be reported as valid until
     the diff is clean.
  6. Writes the design §5 timing analysis
     (`p515_s36_step36_timing.analyze_phase_timing`) from the ON run's
     recorder JSONL sidecar plus its captured stdout's
     `"[INFO] \\t - Iteration {iter}: {X:.2f} s"` lines (production's own
     per-cycle wall-time print, `shared_resources_planning.py:3007`, parsed
     the SAME way `p515_s36_parallel_audit.py`'s `parse_cycle_wall_times`
     already does -- reused pattern, reproduced here rather than imported
     since that script has no public API for it) and, if `inject_report_timing`
     produced any, the NL-write sub-times Pyomo's own `report_timing=True`
     print emits (`"N.NN seconds required to write file"`).

======================================================================
EXACT LAUNCH COMMAND (for the Planner; NOT executed by this Worker)
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s36_step36_timing_run.py \\
        > data/SRP1/Results/P515S36_STEP36_TIMING_launch.log 2>&1

Attached, alone, both streams captured -- no `screen`/`nohup`/backgrounding
(CLAUDE.md's campaign-running evidence rule). Refuses to run concurrently
with `p515_g_g1_g4_admm_gates.py` or with a second copy of itself (the shared
`.p515_g_gate.lock`).

Expected wall time: two ADMM cycles at the s35ref reference configuration's
per-cycle wall time (`WORKER_REPORT_S36_PARALLEL_AUDIT.md` median 33.7 s,
cold cycle 1 alone measured 30.3 s there) TWICE (OFF then ON), i.e.
approximately 1-2 minutes total, not counting model construction/
initialization (~tens of seconds, `P515S35_REF_run` evidence) paid once per
run -- so a few minutes end to end, not the ~4-5 hour scale of a capped-500
certification run.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_ROOT = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36', 'step36_timing')
OUT_OFF = os.path.join(OUT_ROOT, 'off')
OUT_ON = os.path.join(OUT_ROOT, 'on')

# Production files this instrumentation reads (via the wrap points cited in
# WORKER_REPORT_S36_TIMING.md) but never edits -- checked clean in git before
# the measurement runs, so the run is provably against the code those
# citations describe.
_PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'shared_resources_planning.py',
    'network.py',
    'network_data.py',
    'shared_energy_storage_data.py',
)

ITERATION_LINE_RE = re.compile(r'Iteration (\d+):\s*([0-9.]+)\s*s')
# Pyomo `report_timing=True`'s own NL-write print (pyomo/opt/base/solvers.py
# OptSolver._presolve): "   N.NN seconds required to write file"
REPORT_TIMING_NL_WRITE_RE = re.compile(r'([0-9.]+)\s+seconds required to write file')


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stderr(path):
    """This script's OWN stderr capture -- `p515_g_g1_g4_admm_gates.tee_stdout`
    only redirects stdout (cited, not edited); the CLAUDE.md evidence rule
    requires BOTH streams captured, so this adds the stderr half locally."""
    with open(path, 'w') as handle:
        tee = _Tee(sys.stderr, handle)
        with redirect_stderr(tee):
            yield path


def _check_preconditions():
    failures = []

    lock_path = os.path.join(REPO_ROOT, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    this_pid = str(os.getpid())
    for line in ps_output.splitlines():
        if 'p515_g_g1_g4_admm_gates.py' in line:
            fields = line.split()
            pid = fields[1] if len(fields) > 1 else None
            if pid != this_pid:
                failures.append(f'a p515_g_g1_g4_admm_gates.py process appears to be alive: {line}')

    for path in (OUT_OFF, OUT_ON):
        if os.path.exists(path):
            failures.append(f'output directory already exists (write-once): {path}')

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(_PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    except Exception as error:
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    return failures


def parse_cycle_wall_times(stdout_path):
    """Same pattern as p515_s36_parallel_audit.py's parse_cycle_wall_times
    (reproduced, not imported -- that script exposes no public API and is a
    committed, frozen audit artifact)."""
    wall = {}
    with open(stdout_path, 'r', errors='replace') as handle:
        for line in handle:
            m = ITERATION_LINE_RE.search(line)
            if m:
                wall[int(m.group(1))] = float(m.group(2))
    return wall


def parse_report_timing_nl_write_seconds(stdout_path):
    """Sum of every Pyomo `report_timing=True` "seconds required to write
    file" line in the captured stdout -- design §2.5's one-off cross-check.
    Returns a single aggregate float (NOT per-block -- report_timing's print
    output is not tagged with the (agent, block) key this recorder uses, so
    disaggregating it further would require re-deriving order from the
    surrounding `[INFO]` prints; out of scope for this one-off cross-check,
    which only needs the AGGREGATE NL-write share for the design §5
    overhead_local formula)."""
    total = 0.0
    found = False
    with open(stdout_path, 'r', errors='replace') as handle:
        for line in handle:
            m = REPORT_TIMING_NL_WRITE_RE.search(line)
            if m:
                total += float(m.group(1))
                found = True
    return total if found else None


def _run_one(label, out_dir, eval_id, recorder=None, inject_report_timing=False):
    """Mirrors the committed `elif gate == 's35ref':` branch of
    `p515_g_g1_g4_admm_gates.py` (cited, not copied -- every called function
    below is `G.<name>`, the SAME object that branch calls), with
    `num_max_iters_override=2` (design §5's two-cycle preflight) instead of
    `G.S35REF_CAP` (500, the certified run-1 cap) and a fresh `out_dir`/
    `eval_id` pair so nothing committed under `P515S35_REF_run` is ever
    touched.
    """
    import p515_g_g1_g4_admm_gates as G

    os.makedirs(out_dir, exist_ok=True)

    preflight_eval_id = f'{eval_id}_preflight_capture_check'
    preflight_eval_dir = os.path.join(G.O.WORK_DIR, preflight_eval_id)
    if os.path.exists(preflight_eval_dir):
        raise RuntimeError(f'refusing to start: preflight eval dir already exists: {preflight_eval_dir}')
    preflight_planning = G.O.fresh_planning(preflight_eval_id)
    checklist, floor_rows_by_node = G.assert_s35ref_capture_paths(preflight_planning)
    del preflight_planning
    print(f'[P5.15-S36-TIMING {label}] capture-path pre-flight passed: {checklist}')

    recourse_jump_path = os.path.join(out_dir, f'recourse_jump_sidecar_{label}.jsonl')
    ess_stride_path = os.path.join(out_dir, f'ess_entry_stride_{label}.jsonl')
    floor_sidecar_path = os.path.join(out_dir, f'soh_floor_sidecar_{label}.jsonl')
    for path in (recourse_jump_path, ess_stride_path, floor_sidecar_path):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')

    def _hook(planning, sed, models, rows, report, out_dir, label):
        # NOTE: the parameter name `label` here is NOT this function's own choice --
        # `run_admm_arm` calls `post_run_hook(**hook_kwargs)` with a literal `label=label`
        # key (shared_resources_planning... see p515_g_g1_g4_admm_gates.py `run_admm_arm`,
        # `hook_kwargs = dict(..., label=label)`), so the parameter name must match exactly.
        report['soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, REPO_ROOT)
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=floor_sidecar_path)

    def _do_run():
        with G.s35ref_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                    floor_rows_by_node, stride=1):
            return G.run_admm_arm(
                label, out_dir, k_override=None, investment_map=None,
                num_max_iters_override=2, eval_id=eval_id, post_run_hook=_hook,
                apply_rho=False, full_diagnostics_in_rows=True)

    if recorder is not None:
        import p515_s36_step36_timing as T
        with T.recorder_installed(recorder, inject_report_timing=inject_report_timing):
            report, path = _do_run()
    else:
        report, path = _do_run()

    return report, path, floor_sidecar_path


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bitwise_diff(report_off, report_on, floor_sidecar_off_path, floor_sidecar_on_path):
    """Design §2.4: recorder OFF vs ON must be bitwise identical on every
    numeric artifact already used as the determinism reference elsewhere in
    this programme. Compares:
      - `cycle_trajectory` (every field `A.cycle_row` derives PLUS, since
        `full_diagnostics_in_rows=True`, every raw `admm_diagnostics` field --
        Boyd residuals/ratios, rho_*_before/after/action, gamma_*_before/after,
        objective_change_ratio, gap_proxy_*, EFC/day, recourse);
      - `esso_complementarity_diagnostics_by_round` (the detector);
      - the SoH floor-multiplier sidecar (`s35ref_capture_hooks`' own
        per-run JSONL artifact -- the SoH trajectory -- read and compared
        line-by-line, since it is a sidecar file, not part of `report`).
    Returns a dict with 'identical': bool and, if not, the first field/row
    where the two runs diverge.
    """
    diffs = []
    rows_off = report_off.get('cycle_trajectory', [])
    rows_on = report_on.get('cycle_trajectory', [])
    if len(rows_off) != len(rows_on):
        diffs.append({'field': 'cycle_trajectory_length', 'off': len(rows_off), 'on': len(rows_on)})
    else:
        for i, (row_off, row_on) in enumerate(zip(rows_off, rows_on)):
            keys = set(row_off) | set(row_on)
            for key in sorted(keys):
                if row_off.get(key) != row_on.get(key):
                    diffs.append({'field': f'cycle_trajectory[{i}].{key}',
                                  'off': row_off.get(key), 'on': row_on.get(key)})

    detector_off = report_off.get('esso_complementarity_diagnostics_by_round')
    detector_on = report_on.get('esso_complementarity_diagnostics_by_round')
    if detector_off != detector_on:
        diffs.append({'field': 'esso_complementarity_diagnostics_by_round',
                      'off': detector_off, 'on': detector_on})

    soh_off = _read_jsonl(floor_sidecar_off_path)
    soh_on = _read_jsonl(floor_sidecar_on_path)
    if len(soh_off) != len(soh_on):
        diffs.append({'field': 'soh_floor_sidecar_length', 'off': len(soh_off), 'on': len(soh_on)})
    else:
        for i, (row_off, row_on) in enumerate(zip(soh_off, soh_on)):
            if row_off != row_on:
                diffs.append({'field': f'soh_floor_sidecar[{i}]', 'off': row_off, 'on': row_on})

    return {'identical': len(diffs) == 0, 'diffs': diffs[:50], 'n_diffs': len(diffs)}


def main():
    raise SystemExit(
        'p515_s36_step36_timing_run.py is NOT to be executed by the Worker that wrote it '
        '(P5.15 Step 3.6, Worker task W3: "DO NOT RUN IT"). This guard is the FAIL-SAFE, '
        'not the primary control -- the primary control is that no agent invokes this file. '
        'The Planner removes this guard (or invokes main_() directly) when authorizing the '
        'measurement run, per the exact launch command in this file\'s module docstring.')


def main_():
    """The real entry point, split from `main()` so the fail-safe above cannot
    be bypassed by merely calling `python p515_s36_step36_timing_run.py` --
    the Planner must edit this file (or invoke `main_()` from a fresh
    process) to actually run it, an explicit, auditable action."""
    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)

    import p515_g_g1_g4_admm_gates as G
    import p515_s36_step36_timing as T

    G._acquire_exclusive_run_lock()

    os.makedirs(OUT_ROOT, exist_ok=True)
    stderr_path = os.path.join(OUT_ROOT, 'stderr_combined.log')
    started = time.time()
    with tee_stderr(stderr_path):
        print('[P5.15-S36-TIMING] run OFF (recorder not installed) -- 2 cycles, cold, s35ref config class.')
        report_off, path_off, floor_off = _run_one('off', OUT_OFF, 'p515s36_timing_off', recorder=None)

        print('[P5.15-S36-TIMING] run ON (recorder installed, report_timing cross-check) -- 2 cycles, cold.')
        recorder = T.PhaseTimingRecorder()
        report_on, path_on, floor_on = _run_one('on', OUT_ON, 'p515s36_timing_on', recorder=recorder,
                                                 inject_report_timing=True)

        n_records = recorder.to_jsonl(os.path.join(OUT_ON, 'phase_timing_records.jsonl'))
        print(f'[P5.15-S36-TIMING] wrote {n_records} raw phase-timing records.')

        diff = _bitwise_diff(report_off, report_on, floor_off, floor_on)
        diff_path = os.path.join(OUT_ROOT, 'bitwise_identity_check.json')
        with open(diff_path, 'w') as handle:
            json.dump(diff, handle, indent=1, default=str)
        print(f'[P5.15-S36-TIMING] bitwise identity (off vs on): {diff["identical"]} '
              f'({diff["n_diffs"]} diffs) -- see {diff_path}')

        derived = T.derive_param_update_and_bookkeeping(recorder.records)
        all_records = recorder.records + derived
        production_iter_wall = parse_cycle_wall_times(os.path.join(OUT_ON, 'stdout_on.log'))
        nl_write_total = parse_report_timing_nl_write_seconds(os.path.join(OUT_ON, 'stdout_on.log'))
        nl_write_seconds = {'aggregate': nl_write_total} if nl_write_total is not None else None
        analysis = T.analyze_phase_timing(
            all_records, production_iter_wall_by_cycle=production_iter_wall,
            x_threshold=0.70, projection_workers=8, nl_write_seconds=nl_write_seconds)
        analysis_path = os.path.join(OUT_ROOT, 'phase_timing_analysis.json')
        with open(analysis_path, 'w') as handle:
            json.dump(analysis, handle, indent=1, default=str)
        print(f'[P5.15-S36-TIMING] wrote {analysis_path}')
        print(f"[P5.15-S36-TIMING] verdict (X=70%): {analysis['verdict_pass']} "
              f"(ratio={analysis['verdict_ratio']})")
        print(f"[P5.15-S36-TIMING] projected 8-worker speedup: "
              f"{analysis.get('speedup_at_8_workers')}")

    manifest = {}
    for root, _dirs, files in os.walk(OUT_ROOT):
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, 'rb') as handle:
                manifest[os.path.relpath(fpath, OUT_ROOT)] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_ROOT, 'manifest_sha256.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f'[P5.15-S36-TIMING] total wall time: {time.time() - started:.1f} s')
    print(f'[P5.15-S36-TIMING] wrote sha256 manifest: {manifest_path}')


if __name__ == '__main__':
    main()
