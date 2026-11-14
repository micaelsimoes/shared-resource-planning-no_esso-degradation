"""
P5.15 Step 3.6 Worker task -- ZERO-SOLVE re-analysis of the already-captured
Step 3.6 timing measurement (commit `1e7b0a69` at the time this task started,
`data/SRP1/Results/P515S36/step36_timing/{off,on}/`). Corrects four defects
the Planner found in the v1 analysis (`P5_15_STEP36_TIMING_DESIGN.md` §1/§2/
§5; `WORKER_REPORT_S36_TIMING_MEASUREMENT.md` states the findings in full):

  D1. `derive_param_update_and_bookkeeping` (in `p515_s36_step36_timing.py`)
      incorrectly subtracted `clone` from `block_total` when deriving
      `bookkeeping`, even though `clone()` (network_data.py:61) is a SIBLING
      call to `run_smopf()` (:62-63), not nested inside it -- FIXED in
      `p515_s36_step36_timing.py` (see its module docstring / this task's
      Worker Report).
  D2. v1's totals silently included the 156 pre-ADMM-loop INITIALIZATION
      solve records (`cycle is None`) alongside the two sampled ADMM cycles
      -- FIXED: `analyze_phase_timing` now restricts every total to sampled
      cycles and reports initialization/recovery separately.
  D3. v1 classified the ENTIRE `solve_bundle - NL-write` remainder (which
      includes the real, parallelizable IPOPT subprocess time) as SERIAL
      overhead -- FIXED: this script parses Pyomo's own `report_timing=True`
      stdout (captured on the ON run only) into a per-solve
      {nl_write, ipopt, sol_parse} breakdown, matched 1:1 by call order to
      the recorder's own `solve_bundle` records, and passes it to
      `analyze_phase_timing` as `solve_bundle_subtimes` so only the
      `.sol`-parse remainder (local) -- not the IPOPT compute time
      (parallelizable solve work, handled by the LPT/Amdahl projection) --
      is folded into `overhead_local`.
  D4. v1's `bitwise_identity_check.json` reported 1 diff, entirely because
      `esso_complementarity_diagnostics_by_round` entries carry a `log_path`
      field that necessarily differs between the OFF and ON run's own output
      directories -- FIXED: this script re-derives the comparison from the
      SAME saved `g_off.json`/`g_on.json` artifacts, excluding `log_path`
      (and only that field -- explicitly listed, not a wildcard) and reports
      identical/not.

Zero solves: `SolveProfileGuard(permitted=())` is armed for the WHOLE
script; every input is read from artifacts ALREADY on disk (the v1
measurement run, its captured stdout, its saved per-run report JSON, and
the per-block IPOPT logs the production run itself wrote under
`data/SRP1/Results/P56A/evals/p515s36_timing_{off,on}/logs/`). No harness,
no `p515_g_g1_g4_admm_gates.py`, no production entry point is invoked.

Write-once: every output is written to a NEW path (`*_v2.json`); this
script refuses to run if any v2 output path already exists, and never
opens a v1 artifact for writing.

Usage:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python \\
        p515_s36_step36_timing_reanalyze.py
"""

import glob
import hashlib
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s36_step36_timing as T  # noqa: E402

OUT_ROOT = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36', 'step36_timing')
OUT_OFF = os.path.join(OUT_ROOT, 'off')
OUT_ON = os.path.join(OUT_ROOT, 'on')

ANALYSIS_V1_PATH = os.path.join(OUT_ROOT, 'phase_timing_analysis.json')
ANALYSIS_V2_PATH = os.path.join(OUT_ROOT, 'phase_timing_analysis_v2.json')
IDENTITY_V1_PATH = os.path.join(OUT_ROOT, 'bitwise_identity_check.json')
IDENTITY_V2_PATH = os.path.join(OUT_ROOT, 'bitwise_identity_check_v2.json')
MANIFEST_V2_PATH = os.path.join(OUT_ROOT, 'manifest_sha256_v2.json')

X_THRESHOLD = 0.70  # same threshold the v1 measurement screened against (Addendum 20 interim)

ITERATION_LINE_RE = re.compile(r'Iteration (\d+):\s*([0-9.]+)\s*s')
NL_WRITE_RE = re.compile(r'([0-9.]+)\s+seconds required to write file')
SOLVER_RE = re.compile(r'([0-9.]+)\s+seconds required for solver')
LOGREAD_RE = re.compile(r'([0-9.]+)\s+seconds required to read logfile')
SOLREAD_RE = re.compile(r'([0-9.]+)\s+seconds required to read solution file')
IPOPT_LOG_TOTAL_RE = re.compile(r'Total seconds in IPOPT\s*=\s*([0-9.]+)')

# D4: the ONLY field excluded from the identity comparison -- a path field
# that necessarily differs between the OFF and ON run's own output
# directories, listed explicitly (never a wildcard/substring match).
IDENTITY_EXCLUDED_FIELDS = ('log_path',)


# ======================================================================
# 0. Capture-path assertion -- fail fast, before computing anything, if a
#    quantity this script's frozen deliverable list requires has no file to
#    read it from (CLAUDE.md evidence rule: "assert the capture path before
#    executing").
# ======================================================================
def assert_capture_paths():
    required = {
        'on/phase_timing_records.jsonl': os.path.join(OUT_ON, 'phase_timing_records.jsonl'),
        'on/stdout_on.log (report_timing + iteration wall times)':
            os.path.join(OUT_ON, 'stdout_on.log'),
        'off/g_off.json (cycle_trajectory / esso diagnostics / soh path)':
            os.path.join(OUT_OFF, 'g_off.json'),
        'on/g_on.json': os.path.join(OUT_ON, 'g_on.json'),
        'off/soh_floor_sidecar_off.jsonl': os.path.join(OUT_OFF, 'soh_floor_sidecar_off.jsonl'),
        'on/soh_floor_sidecar_on.jsonl': os.path.join(OUT_ON, 'soh_floor_sidecar_on.jsonl'),
        'off/network_failures_off.jsonl': os.path.join(OUT_OFF, 'network_failures_off.jsonl'),
        'on/network_failures_on.jsonl': os.path.join(OUT_ON, 'network_failures_on.jsonl'),
        'v1 analysis (phase_timing_analysis.json, for the discrepancy table)': ANALYSIS_V1_PATH,
        'v1 identity check (bitwise_identity_check.json)': IDENTITY_V1_PATH,
    }
    missing = [label for label, path in required.items() if not os.path.exists(path)]
    if missing:
        raise RuntimeError(
            f'Capture-path assertion failed -- the following required inputs are '
            f'missing, cannot proceed: {missing}')
    return required


def assert_write_once():
    existing = [p for p in (ANALYSIS_V2_PATH, IDENTITY_V2_PATH, MANIFEST_V2_PATH) if os.path.exists(p)]
    if existing:
        raise RuntimeError(f'refusing to overwrite existing v2 artifact(s): {existing}')


# ======================================================================
# 1. D3 -- parse Pyomo's report_timing stdout into per-solve {nl_write,
#    ipopt, sol_parse}, matched 1:1 by call order to the recorder's own
#    solve_bundle records (both are produced by the SAME single-threaded,
#    synchronous run -- solve() prints its report_timing lines synchronously
#    before returning, so stdout order == solve_bundle call order; validated
#    below by an exact count match, which is itself part of this function's
#    contract, not assumed silently).
# ======================================================================
def parse_report_timing_groups(stdout_path):
    """Returns a list of dicts, one per solve, in stdout order:
    {'nl_write': s, 'ipopt': s, 'log_read': s, 'sol_read': s}. A solve's
    group is considered complete (used downstream) only if all four keys
    are present -- an incomplete trailing group (e.g. output truncated
    mid-solve) is dropped, not padded with a guess."""
    groups = []
    cur = {}
    with open(stdout_path, 'r', errors='replace') as handle:
        for line in handle:
            m = NL_WRITE_RE.search(line)
            if m:
                if cur:
                    groups.append(cur)
                cur = {'nl_write': float(m.group(1))}
                continue
            m = SOLVER_RE.search(line)
            if m:
                cur['ipopt'] = float(m.group(1))
                continue
            m = LOGREAD_RE.search(line)
            if m:
                cur['log_read'] = float(m.group(1))
                continue
            m = SOLREAD_RE.search(line)
            if m:
                cur['sol_read'] = float(m.group(1))
                continue
    if cur:
        groups.append(cur)
    return [g for g in groups if {'nl_write', 'ipopt', 'log_read', 'sol_read'} <= g.keys()]


def build_solve_bundle_subtimes(records, stdout_path):
    """Zips the report_timing groups (stdout order) against the recorder's
    own `solve_bundle` records (sorted by `seq`, i.e. also call order),
    keyed by `seq`. Raises if the counts do not match exactly -- a count
    mismatch means the 1:1 ordering assumption this function depends on
    does not hold for this run, and silently zipping a shorter/longer list
    would misattribute times to the wrong block."""
    sb_records = sorted((r for r in records if r['phase'] == 'solve_bundle'), key=lambda r: r['seq'])
    groups = parse_report_timing_groups(stdout_path)
    if len(sb_records) != len(groups):
        raise RuntimeError(
            f'report_timing group count ({len(groups)}) does not match solve_bundle '
            f'record count ({len(sb_records)}) -- refusing to zip them 1:1 (would '
            f'misattribute NL-write/IPOPT/sol-parse times to the wrong solve). '
            f'This can happen if report_timing was not injected for every solve, or '
            f'if stdout was truncated.')
    subtimes = {}
    for record, group in zip(sb_records, groups):
        subtimes[record['seq']] = {
            'nl_write': group['nl_write'],
            'ipopt': group['ipopt'],
            'sol_parse': group['log_read'] + group['sol_read'],
        }
    return subtimes, sb_records, groups


def parse_cycle_wall_times(stdout_path):
    """Same pattern as `p515_s36_step36_timing_run.py::parse_cycle_wall_times`
    (reproduced, not imported -- avoids importing that module's own
    `main()`-adjacent state; this is a pure, zero-solve text parse)."""
    wall = {}
    with open(stdout_path, 'r', errors='replace') as handle:
        for line in handle:
            m = ITERATION_LINE_RE.search(line)
            if m:
                wall[int(m.group(1))] = float(m.group(2))
    return wall


def cross_check_ipopt_log_totals(on_logs_glob):
    """Informational cross-check ONLY (design §2.5): sums every
    'Total seconds in IPOPT' occurrence across every per-block IPOPT log the
    ON run itself wrote (`data/SRP1/Results/P56A/evals/p515s36_timing_on/
    logs/*`). This figure is IPOPT's own internal compute time; Pyomo's
    'seconds required for solver' (used as the 'ipopt' component of
    `solve_bundle_subtimes` above) wraps the whole `_apply_solver` call,
    including subprocess launch/fork/exec -- design §2.5 explicitly predicts
    a gap between the two ("subprocess launch overhead, not IPOPT compute").
    Returns (n_occurrences, total_seconds) or (0, None) if the logs
    directory does not exist (NOT fatal -- this is a cross-check, not a
    required input; `assert_capture_paths` does not require it)."""
    paths = sorted(glob.glob(on_logs_glob))
    if not paths:
        return 0, None
    total = 0.0
    n = 0
    for path in paths:
        with open(path, 'r', errors='replace') as handle:
            for line in handle:
                m = IPOPT_LOG_TOTAL_RE.search(line)
                if m:
                    total += float(m.group(1))
                    n += 1
    return n, (total if n else None)


# ======================================================================
# 2. D4 -- identity check v2, excluding IDENTITY_EXCLUDED_FIELDS, re-derived
#    from the SAME saved g_off.json/g_on.json + soh sidecars this task must
#    not re-run to obtain (re-running would overwrite the v1 evidence this
#    report cites -- CLAUDE.md evidence rule).
# ======================================================================
def _strip_excluded_fields(obj, excluded):
    """Recursively rebuild `obj`, dropping any dict key in `excluded` at any
    depth. Only used for the identity comparison -- never mutates a source
    file on disk."""
    if isinstance(obj, dict):
        return {k: _strip_excluded_fields(v, excluded) for k, v in obj.items() if k not in excluded}
    if isinstance(obj, list):
        return [_strip_excluded_fields(v, excluded) for v in obj]
    return obj


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


def bitwise_diff_v2(report_off, report_on, floor_off_path, floor_on_path, excluded_fields):
    """Same three comparisons as `p515_s36_step36_timing_run.py::_bitwise_diff`
    (cycle_trajectory, esso_complementarity_diagnostics_by_round, SoH floor
    sidecar), but with `excluded_fields` stripped (recursively) from BOTH
    sides before comparing -- so a legitimate, expected difference (e.g. a
    path field that necessarily differs between two output directories) is
    not reported as a numeric/behavioural divergence."""
    diffs = []

    rows_off = _strip_excluded_fields(report_off.get('cycle_trajectory', []), excluded_fields)
    rows_on = _strip_excluded_fields(report_on.get('cycle_trajectory', []), excluded_fields)
    if len(rows_off) != len(rows_on):
        diffs.append({'field': 'cycle_trajectory_length', 'off': len(rows_off), 'on': len(rows_on)})
    else:
        for i, (row_off, row_on) in enumerate(zip(rows_off, rows_on)):
            keys = set(row_off) | set(row_on)
            for key in sorted(keys):
                if row_off.get(key) != row_on.get(key):
                    diffs.append({'field': f'cycle_trajectory[{i}].{key}',
                                  'off': row_off.get(key), 'on': row_on.get(key)})

    detector_off = _strip_excluded_fields(
        report_off.get('esso_complementarity_diagnostics_by_round'), excluded_fields)
    detector_on = _strip_excluded_fields(
        report_on.get('esso_complementarity_diagnostics_by_round'), excluded_fields)
    if detector_off != detector_on:
        diffs.append({'field': 'esso_complementarity_diagnostics_by_round',
                      'off': detector_off, 'on': detector_on})

    soh_off = _strip_excluded_fields(_read_jsonl(floor_off_path), excluded_fields)
    soh_on = _strip_excluded_fields(_read_jsonl(floor_on_path), excluded_fields)
    if len(soh_off) != len(soh_on):
        diffs.append({'field': 'soh_floor_sidecar_length', 'off': len(soh_off), 'on': len(soh_on)})
    else:
        for i, (row_off, row_on) in enumerate(zip(soh_off, soh_on)):
            if row_off != row_on:
                diffs.append({'field': f'soh_floor_sidecar[{i}]', 'off': row_off, 'on': row_on})

    return {'identical': len(diffs) == 0, 'diffs': diffs[:50], 'n_diffs': len(diffs),
            'excluded_fields': list(excluded_fields)}


# ======================================================================
# 3. Manifest -- ALL measurement artifacts under off/ and on/, EXCLUDING
#    esso_capture/ and results/ (which get ONE directory-level hash each,
#    per the task's own "hash-recorded only" instruction, not a full
#    per-file expansion), plus v1+v2 analyses, identity checks, launch
#    logs (including the refused attempt + its exit code), stderr, and
#    this task's scripts.
# ======================================================================
def _sha256_file(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _sha256_directory(dir_path):
    """One aggregate hash for an entire directory: sha256 of the sorted
    '<relpath>:<sha256>\\n' listing of every file inside it. Deterministic,
    reproducible, and changes if ANY file inside changes -- 'hash-recorded'
    without expanding every file into its own manifest entry."""
    entries = []
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, dir_path)
            entries.append(f'{rel}:{_sha256_file(fpath)}')
    entries.sort()
    listing = '\n'.join(entries).encode('utf-8')
    return hashlib.sha256(listing).hexdigest(), len(entries)


def build_manifest_v2():
    manifest = {}

    for label, root in (('off', OUT_OFF), ('on', OUT_ON)):
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if entry in ('esso_capture', 'results') and os.path.isdir(full):
                digest, n_files = _sha256_directory(full)
                manifest[f'{label}/{entry} (directory hash, {n_files} files, hash-recorded only)'] = digest
                continue
            if os.path.isdir(full):
                for droot, _dirs, files in os.walk(full):
                    for fname in sorted(files):
                        fpath = os.path.join(droot, fname)
                        manifest[os.path.relpath(fpath, OUT_ROOT)] = _sha256_file(fpath)
            else:
                manifest[os.path.relpath(full, OUT_ROOT)] = _sha256_file(full)

    # v1 + v2 analyses, identity checks (both versions -- v1 is evidence too)
    for path in (ANALYSIS_V1_PATH, IDENTITY_V1_PATH):
        manifest[os.path.relpath(path, OUT_ROOT)] = _sha256_file(path)

    # Launch logs, incl. the refused self-match attempt + its exit code, and
    # the real run's exit code -- all under data/SRP1/Results/ directly.
    for pattern in ('P515S36_STEP36_TIMING_launch*.log', 'P515S36_STEP36_TIMING_exit_code*.txt'):
        for path in sorted(glob.glob(os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', pattern))):
            manifest[os.path.relpath(path, OUT_ROOT)] = _sha256_file(path)

    # This task's own scripts.
    for fname in ('p515_s36_step36_timing.py', 'p515_s36_step36_timing_run.py',
                  'p515_s36_step36_timing_checks.py', 'p515_s36_step36_timing_reanalyze.py'):
        path = os.path.join(REPO_ROOT, fname)
        manifest[os.path.relpath(path, OUT_ROOT)] = _sha256_file(path)

    return manifest


def main():
    guard = SolveProfileGuard(permitted=(), label='p515_s36_step36_timing_reanalyze (zero solves)').install()
    try:
        assert_capture_paths()
        assert_write_once()

        # ---- load raw records + derive ----
        raw_records = []
        with open(os.path.join(OUT_ON, 'phase_timing_records.jsonl'), 'r') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))
        derived_records = T.derive_param_update_and_bookkeeping(raw_records)  # raises on defect (D1 guard)
        all_records = raw_records + derived_records

        stdout_on_path = os.path.join(OUT_ON, 'stdout_on.log')
        production_iter_wall = parse_cycle_wall_times(stdout_on_path)
        solve_bundle_subtimes, sb_records, rt_groups = build_solve_bundle_subtimes(
            raw_records, stdout_on_path)

        analysis = T.analyze_phase_timing(
            all_records, x_threshold=X_THRESHOLD,
            production_iter_wall_by_cycle=production_iter_wall,
            projection_workers=8, solve_bundle_subtimes=solve_bundle_subtimes)

        # ---- per-cycle analysis (deliverable: "Include per cycle (1, 2)") ----
        # Re-run analyze_phase_timing on records restricted to EACH single
        # sampled cycle, so overhead_local/serial, the X ratio, the verdict,
        # f_serial and speedup(8) are each reported PER CYCLE, not only
        # aggregated across both -- `solve_bundle_subtimes` is passed
        # unfiltered (keyed by seq, a global identifier; only the seqs
        # actually present in that cycle's records will be looked up).
        per_cycle_analysis = {}
        for cycle in sorted(production_iter_wall.keys()):
            cycle_raw = [r for r in raw_records if r['cycle'] == cycle]
            cycle_derived = T.derive_param_update_and_bookkeeping(cycle_raw)
            cycle_all = cycle_raw + cycle_derived
            per_cycle_analysis[cycle] = T.analyze_phase_timing(
                cycle_all, x_threshold=X_THRESHOLD,
                production_iter_wall_by_cycle={cycle: production_iter_wall[cycle]},
                projection_workers=8, solve_bundle_subtimes=solve_bundle_subtimes)
        analysis['per_cycle_analysis'] = per_cycle_analysis

        # ---- D3 cross-check: report_timing 'seconds required for solver' vs
        #      the per-block IPOPT logs' own 'Total seconds in IPOPT' ----
        n_ipopt_log_occurrences, ipopt_log_total = cross_check_ipopt_log_totals(
            os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56A', 'evals',
                         'p515s36_timing_on', 'logs', '*'))
        report_timing_ipopt_total_all_solves = sum(g['ipopt'] for g in rt_groups)
        analysis['ipopt_crosscheck'] = {
            'report_timing_solver_seconds_all_solves_incl_init': report_timing_ipopt_total_all_solves,
            'ipopt_log_total_seconds_in_ipopt_all_occurrences': ipopt_log_total,
            'ipopt_log_n_occurrences': n_ipopt_log_occurrences,
            'note': ('IPOPT log Total seconds in IPOPT is pure IPOPT internal compute; '
                     'report_timing "seconds required for solver" wraps the whole '
                     '_apply_solver call (subprocess launch/fork/exec included) -- design '
                     'Sec 2.5 predicts these differ by subprocess launch overhead. Both '
                     'sums are over ALL 154 solves (init + cycle 1 + cycle 2), NOT cycle-'
                     'scoped, since the IPOPT logs commingle cycles for DSO/TSO '
                     '(file_append) and cannot be reliably split by cycle without the '
                     'report_timing-vs-solve_bundle seq alignment this script already '
                     'uses for the primary (cycle-scoped) figures above.'),
        }

        # ---- D2: recovery event cross-reference (network_failures_on.jsonl) ----
        recovery_events = _read_jsonl(os.path.join(OUT_ON, 'network_failures_on.jsonl'))
        analysis['recovery_events_on_run'] = recovery_events

        # ---- discrepancy vs WORKER_REPORT_S36_PARALLEL_AUDIT.md's earlier figures ----
        analysis['comparison_to_parallel_audit'] = {
            'parallel_audit_median_ipopt_total_s': 12.990,
            'parallel_audit_median_overhead_s': 21.488,
            'parallel_audit_median_wall_s': 33.725,
            'this_run_cycle_wall_s': production_iter_wall,
            'note': ('The parallel audit figures are medians over 477 cycles of a DIFFERENT '
                     '(500-cycle capped) reference run; this measurement is 2 cold cycles of '
                     'a fresh run at the same s35ref configuration class -- NOT the same '
                     'instance, so a point-vs-median comparison is directional only, stated '
                     'here rather than treated as a discrepancy needing reconciliation.'),
        }

        with open(ANALYSIS_V2_PATH, 'w') as handle:
            json.dump(analysis, handle, indent=1, default=str)
        print(f'[P5.15-S36-TIMING-REANALYZE] wrote {ANALYSIS_V2_PATH}')
        print(f"[P5.15-S36-TIMING-REANALYZE] verdict (X={X_THRESHOLD:.0%}): "
              f"{analysis['verdict_pass']} (ratio={analysis['verdict_ratio']})")
        print(f"[P5.15-S36-TIMING-REANALYZE] projected 8-worker speedup: "
              f"{analysis.get('speedup_at_8_workers')}")

        # ---- D4: identity check v2 ----
        g_off = json.load(open(os.path.join(OUT_OFF, 'g_off.json')))
        g_on = json.load(open(os.path.join(OUT_ON, 'g_on.json')))
        floor_off_path = os.path.join(OUT_OFF, 'soh_floor_sidecar_off.jsonl')
        floor_on_path = os.path.join(OUT_ON, 'soh_floor_sidecar_on.jsonl')
        identity_v2 = bitwise_diff_v2(g_off, g_on, floor_off_path, floor_on_path,
                                      IDENTITY_EXCLUDED_FIELDS)
        with open(IDENTITY_V2_PATH, 'w') as handle:
            json.dump(identity_v2, handle, indent=1, default=str)
        print(f'[P5.15-S36-TIMING-REANALYZE] identity v2 (log_path excluded): '
              f'{identity_v2["identical"]} ({identity_v2["n_diffs"]} diffs)')

        # ---- manifest v2 ----
        manifest = build_manifest_v2()
        with open(MANIFEST_V2_PATH, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(f'[P5.15-S36-TIMING-REANALYZE] wrote {MANIFEST_V2_PATH} ({len(manifest)} entries)')

    finally:
        guard_failures = guard.verify(expected_solves=0, expected_execs=0)
        guard.uninstall()
        if guard_failures:
            raise RuntimeError(f'SolveProfileGuard failed at expected_solves=0: {guard_failures}')
        print('[OK] SolveProfileGuard verified: 0 permitted solves, 0 blocked solves (zero-solve reanalysis).')


if __name__ == '__main__':
    main()
