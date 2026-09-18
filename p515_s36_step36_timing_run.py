"""
P5.15 Step 3.6, Worker task W3 -- the recorder-off / recorder-on phase-timing
measurement entry point (design `P5_15_STEP36_TIMING_DESIGN.md` §2.4/§5).
Two cycles by default (the already-committed, already-run evidence under
`data/SRP1/Results/P515S36/step36_timing/{off,on}/`); an optional positional
CLI argument (P5.15 Addendum 22 item (2) follow-up, Worker-prepared, NOT run
by that Worker) parameterizes the cycle count for a LONGER re-measurement
(e.g. 10 cycles) into its OWN, cycle-count-aware output root and eval ids
(`_out_root_for_cycles`/`_eval_id_for` below), so a longer run can never
collide with, or overwrite, the committed 2-cycle evidence.

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
across active nodes), the ONLY deliberate differences being `num_max_iters_
override=<cycle count>` (2 by default, design §5's two-cycle preflight; NOT
the certified 500-cycle run) and fresh eval ids / output roots so nothing
committed is ever touched.

======================================================================
WHAT THIS SCRIPT DOES (when the Planner runs it)
======================================================================
  1. Precondition checks (all must pass BEFORE anything is written) -- THIS
     is the safety mechanism guarding accidental execution (Planner decision,
     P5.15 Step 3.6 follow-up, item 4: no more `main()`/`main_()` fail-safe
     split -- `python p515_s36_step36_timing_run.py` runs the real entry
     point directly, and these checks are what must refuse when it is not
     safe to proceed):
       a. `.p515_g_gate.lock` does not already exist.
       b. no OTHER `p515_g_g1_g4_admm_gates.py` process, and no `p515_s38_*`
          numerical-campaign-arm process, is alive (`ps aux` scan, excluding
          this script's own PID -- `_FORBIDDEN_LIVE_PROCESS_SUBSTRINGS`).
       c. neither output directory
          (`data/SRP1/Results/P515S36/step36_timing/{off,on}/`) exists yet.
       d. every file this instrumentation reads (NOT edits) is clean in git
          (`git status --porcelain` on the exact file list in
          `_PRODUCTION_FILES_TO_CHECK_CLEAN` -- widened, item 1, beyond the
          original four production files to also cover
          `admm_parameters.py`, `p515_g_g1_g4_admm_gates.py`,
          `data/SRP1/SRP1_params.json`, and this instrumentation's own three
          source files) -- a defensive check that the measurement is against
          the SAME code the wrap-point citations in
          `WORKER_REPORT_S36_TIMING.md` describe, not a mid-edit tree.
  2. Acquires the campaign harness's OWN exclusive run lock
     (`G._acquire_exclusive_run_lock()`, `.p515_g_gate.lock`, `O_CREAT|O_EXCL`
     -- reused verbatim, not re-implemented, so its semantics are identical
     by construction) -- refuses to run if another copy of
     `p515_g_g1_g4_admm_gates.py` (or this script) already holds it.
  3. Run OFF: `NUM_CYCLES` ADMM cycles (2 by default; parameterized by an
     optional CLI argument, see "EXACT LAUNCH COMMAND" below), cold, recorder
     NOT installed -- byte-for-byte today's production path (design §2.2: an
     unmodified call, since this harness-side deviation never threads a
     `timing_recorder` kwarg into production AT ALL; "OFF" here means this
     script's OWN `p515_s36_step36_timing.recorder_installed(...)` context
     manager is simply not entered for this run).
  4. Run ON: the SAME `NUM_CYCLES` cycles, cold, with
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
Original 2-cycle measurement (already run; re-running this exact command
would collide with the committed evidence -- do not re-issue it):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s36_step36_timing_run.py \\
        > data/SRP1/Results/P515S36_STEP36_TIMING_launch.log 2>&1

10-cycle re-measurement (P5.15 Addendum 22 item (2) follow-up; NOT run by the
Worker who added this parameter -- writes to
`data/SRP1/Results/P515S36/step36_timing_10cyc/{off,on}/`, distinct from the
committed 2-cycle evidence):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s36_step36_timing_run.py 10 \\
        > data/SRP1/Results/P515S36_STEP36_TIMING_10CYC_launch.log 2>&1

Attached, alone, both streams captured -- no `screen`/`nohup`/backgrounding
(CLAUDE.md's campaign-running evidence rule). Refuses to run concurrently
with `p515_g_g1_g4_admm_gates.py` or with a second copy of itself (the shared
`.p515_g_gate.lock`).

Expected wall time (2-cycle default): two ADMM cycles at the s35ref reference
configuration's
per-cycle wall time (`WORKER_REPORT_S36_PARALLEL_AUDIT.md` median 33.7 s,
cold cycle 1 alone measured 30.3 s there) TWICE (OFF then ON), i.e.
approximately 1-2 minutes total, not counting model construction/
initialization (~tens of seconds, `P515S35_REF_run` evidence) paid once per
run -- so a few minutes end to end, not the ~4-5 hour scale of a capped-500
certification run.

Expected wall time (10-cycle re-measurement): ten cycles at the same
per-cycle wall time, TWICE (OFF then ON), i.e. roughly 5x the 2-cycle
figure above (order 5-10 minutes of cycle time per run, so order 10-20
minutes total across both runs, plus the two initializations) -- still far
below the capped-500 certification-run scale. Not independently measured by
this Worker (this script was prepared, not run, per this task's scope).
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


def _parse_num_cycles(argv):
    """P5.15 Addendum 22 item (2) follow-up (Worker task, S40 clone-capture
    preflight): optional positional CLI argument, the ADMM cycle count for
    this measurement. Default UNCHANGED at 2 (the already-committed,
    already-run evidence under `data/SRP1/Results/P515S36/step36_timing/`) --
    `python p515_s36_step36_timing_run.py` with no argument reproduces
    exactly the prior 2-cycle invocation, same output roots, same eval ids.
    Any OTHER value (e.g. 10, for the Planner's 10-cycle re-measurement) is
    validated as a positive integer and routed to cycle-count-aware output
    roots/eval ids (`_out_root_for_cycles`/`_eval_id_for` below) so it can
    NEVER collide with the committed 2-cycle evidence, regardless of
    argument order or repeated invocation."""
    if len(argv) < 2:
        return 2
    try:
        value = int(argv[1])
    except ValueError:
        raise SystemExit(f'invalid cycle count {argv[1]!r}: must be a positive integer')
    if value < 1:
        raise SystemExit(f'invalid cycle count {value}: must be a positive integer')
    return value


def _parse_run_suffix(argv):
    """P5.15 Addendum 22 item (2), timing-defects Worker task: optional
    SECOND positional CLI argument, a free-text label further distinguishing
    this run's output root / eval ids from an EARLIER run at the SAME cycle
    count. Needed because `step36_timing_10cyc/{off,on}` already exists (the
    pre-fix, committed 10-cycle evidence that showed the two defects this
    task fixes) and must never be overwritten or re-run onto (CLAUDE.md
    evidence rule: never re-run a harness onto an artifact a committed report
    cites). Default '' reproduces the EXACT prior naming for every cycle
    count that has not yet been re-run under a suffix -- this parameter does
    not change behavior for any invocation that omits it."""
    if len(argv) < 3:
        return ''
    label = argv[2].strip('_')
    if not label:
        raise SystemExit(f'invalid run suffix {argv[2]!r}: must be non-empty once stripped of underscores')
    return '_' + label


NUM_CYCLES = _parse_num_cycles(sys.argv)
RUN_SUFFIX = _parse_run_suffix(sys.argv)


def _out_root_for_cycles(num_cycles, run_suffix=''):
    """`step36_timing/` (unchanged path) for the default 2-cycle, no-suffix
    case -- ANY other cycle count, or ANY non-empty suffix, gets its OWN,
    distinct root (`step36_timing_<N>cyc[<suffix>]/`), so a re-measurement
    can never write into, or collide with, an earlier committed evidence
    directory."""
    if num_cycles == 2 and not run_suffix:
        return os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36', 'step36_timing')
    return os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S36',
                         f'step36_timing_{num_cycles}cyc{run_suffix}')


def _eval_id_for(label, num_cycles, run_suffix=''):
    """`p515s36_timing_<off|on>` (unchanged) for the default 2-cycle,
    no-suffix case; `p515s36_timing_<off|on>_<N>cyc[<suffix>]` otherwise --
    same collision-avoidance reasoning as `_out_root_for_cycles`, applied to
    the `O.WORK_DIR` eval ids (which persist independently of `OUT_ROOT` and
    would otherwise collide across runs even if the output directories did
    not)."""
    if num_cycles == 2 and not run_suffix:
        return f'p515s36_timing_{label}'
    return f'p515s36_timing_{label}_{num_cycles}cyc{run_suffix}'


OUT_ROOT = _out_root_for_cycles(NUM_CYCLES, RUN_SUFFIX)
OUT_OFF = os.path.join(OUT_ROOT, 'off')
OUT_ON = os.path.join(OUT_ROOT, 'on')

# Files this instrumentation reads (production files via the wrap points cited
# in WORKER_REPORT_S36_TIMING.md; the campaign harness and its params file
# read-only; this script's own three sibling files) but never edits -- checked
# clean in git before the measurement runs, so the run is provably against the
# code those citations describe. Widened per Planner decision (P5.15 Step 3.6
# follow-up, item 1) beyond the original four production files to also cover
# the harness this script reuses read-only, its params file, and this
# instrumentation's own three source files.
_PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'shared_resources_planning.py',
    'network.py',
    'network_data.py',
    'shared_energy_storage_data.py',
    'admm_parameters.py',
    'p515_g_g1_g4_admm_gates.py',
    os.path.join('data', 'SRP1', 'SRP1_params.json'),
    'p515_s36_step36_timing.py',
    'p515_s36_step36_timing_run.py',
    'p515_s36_step36_timing_checks.py',
)

# Process-table substrings that must not match any OTHER live process (this
# script's own PID is always excluded) before this measurement is allowed to
# start -- the campaign harness itself, and any s38 numerical-campaign arm
# (Planner decision, item 1: "refuse if any process matching
# p515_g_g1_g4_admm_gates.py or p515_s38_ is alive").
_FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = ('p515_g_g1_g4_admm_gates.py', 'p515_s38_')

ITERATION_LINE_RE = re.compile(r'Iteration (\d+):\s*([0-9.]+)\s*s')
# Pyomo `report_timing=True`'s own NL-write print (pyomo/opt/base/solvers.py
# OptSolver._presolve): "   N.NN seconds required to write file"
REPORT_TIMING_NL_WRITE_RE = re.compile(r'([0-9.]+)\s+seconds required to write file')

# P5.15 Addendum 22 item (2), timing-defects Worker task (Defect 2): the full
# four-line report_timing group this entry point's `main()` did NOT parse
# before this fix -- it only ever built the AGGREGATE NL-write figure above
# and passed it through `analyze_phase_timing`'s DEGRADED (v1-equivalent)
# fallback path, which is why `solve_bundle_subtimes_fully_covered` was
# always `False` here regardless of cycle count (see `build_solve_bundle_
# subtimes_with_coverage` below and the Worker Report for the full
# root-cause finding). These four patterns are the same ones
# `p515_s36_step36_timing_reanalyze.py::parse_report_timing_groups` already
# uses (reproduced, not imported -- that script is a separate, frozen,
# 2-cycle-only artifact; this module owns its own copy).
REPORT_TIMING_SOLVER_RE = re.compile(r'([0-9.]+)\s+seconds required for solver')
REPORT_TIMING_LOGREAD_RE = re.compile(r'([0-9.]+)\s+seconds required to read logfile')
REPORT_TIMING_SOLREAD_RE = re.compile(r'([0-9.]+)\s+seconds required to read solution file')


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
        if any(substring in line for substring in _FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            fields = line.split()
            pid = fields[1] if len(fields) > 1 else None
            if pid != this_pid:
                failures.append(f'a forbidden process appears to be alive '
                                 f'(matches {_FORBIDDEN_LIVE_PROCESS_SUBSTRINGS}): {line}')

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


def parse_report_timing_groups(stdout_path):
    """P5.15 Addendum 22 item (2), timing-defects Worker task (Defect 2).
    Same grouping logic as `p515_s36_step36_timing_reanalyze.py`'s function of
    the same name (reproduced, not imported): one dict per real `solver.solve(
    ..., report_timing=True)` call, in stdout order, with keys `nl_write`,
    `ipopt`, `log_read`, `sol_read` when all four of that solve's report_timing
    lines were found. Returns `(complete_groups, n_incomplete)` -- unlike the
    2-cycle-only reanalyze script's version, this ALSO reports how many
    started-but-incomplete groups were dropped (e.g. truncated stdout), so
    that count is visible to the caller rather than silently discarded."""
    groups = []
    cur = {}
    with open(stdout_path, 'r', errors='replace') as handle:
        for line in handle:
            m = REPORT_TIMING_NL_WRITE_RE.search(line)
            if m:
                if cur:
                    groups.append(cur)
                cur = {'nl_write': float(m.group(1))}
                continue
            m = REPORT_TIMING_SOLVER_RE.search(line)
            if m:
                cur['ipopt'] = float(m.group(1))
                continue
            m = REPORT_TIMING_LOGREAD_RE.search(line)
            if m:
                cur['log_read'] = float(m.group(1))
                continue
            m = REPORT_TIMING_SOLREAD_RE.search(line)
            if m:
                cur['sol_read'] = float(m.group(1))
                continue
    if cur:
        groups.append(cur)
    required = {'nl_write', 'ipopt', 'log_read', 'sol_read'}
    complete = [g for g in groups if required <= g.keys()]
    return complete, len(groups) - len(complete)


def build_solve_bundle_subtimes_with_coverage(records, stdout_path):
    """P5.15 Addendum 22 item (2), timing-defects Worker task (Defect 2 fix).

    ROOT CAUSE this function fixes: this script's `main()` previously called
    `analyze_phase_timing(...)` WITHOUT ever passing `solve_bundle_subtimes`
    at all -- only the aggregate `nl_write_seconds` figure. Since
    `analyze_phase_timing` defaults `solve_bundle_subtimes` to `{}` when not
    given, `fully_covered` (`sb_seqs.issubset(solve_bundle_subtimes.keys())`)
    was `False` by construction, for EVERY invocation of this script
    (2-cycle default included), regardless of whether Pyomo's report_timing
    stdout actually covered every solve. The only code path that ever built
    the real per-solve {nl_write, ipopt, sol_parse} split was
    `p515_s36_step36_timing_reanalyze.py`, a SEPARATE script hardcoded to the
    committed 2-cycle `step36_timing/` output root -- it was never run, and
    cannot be pointed, at `step36_timing_10cyc/`. Verified directly against
    the already-captured 10-cycle evidence
    (`data/SRP1/Results/P515S36/step36_timing_10cyc/on/{phase_timing_records.
    jsonl,stdout_on.log}`): 564 raw `solve_bundle` records (51/cycle x 10
    cycles + 3 tier-1 recovery retries + 51 pre-loop init) against 564
    COMPLETE report_timing groups parsed from that run's own stdout -- an
    EXACT match, zero incomplete groups. The 10-cycle INDETERMINATE verdict
    was therefore never caused by a genuine report_timing coverage failure
    at that scale (retries, interleaving, or buffering do not break the
    parse here); it was caused by this wiring gap. See the Worker Report for
    the full evidence.

    This function closes the wiring gap AND, per the task's own instruction
    ("fix the parser so coverage is exact or the shortfall is reported per
    record"), makes the matching itself defensive: it only trusts the
    positional (stdout-order == solve_bundle-call-order) correspondence when
    the two sequences have EXACTLY the same length. On any count mismatch it
    does NOT guess which prefix/subset is still trustworthy (a mismatch can
    occur anywhere in the sequence, so no positional subset is safe to
    assume) -- it reports EVERY raw `solve_bundle` record's coverage status
    individually (`covered: False` for all of them in that case), never
    silently drops a record from the report.

    Returns `(subtimes, coverage)`:
      subtimes: {seq: {'nl_write':.., 'ipopt':.., 'sol_parse':..}}, populated
                only when `coverage['status'] == 'exact'`.
      coverage: {'status': 'exact' | 'mismatch',
                 'n_solve_bundle_records': int,
                 'n_report_timing_groups_complete': int,
                 'n_report_timing_groups_incomplete': int,
                 'per_record': [{'seq', 'cycle', 'agent', 'block', 'attempt',
                                  'covered'}, ...]}  -- one entry per raw
                 solve_bundle record, in seq order, ALWAYS present.
    """
    sb_records = sorted((r for r in records if r['phase'] == 'solve_bundle'), key=lambda r: r['seq'])
    groups, n_incomplete = parse_report_timing_groups(stdout_path)
    exact = (len(sb_records) == len(groups))
    subtimes = {}
    per_record = []
    for i, record in enumerate(sb_records):
        covered = exact
        per_record.append({
            'seq': record['seq'], 'cycle': record['cycle'], 'agent': record['agent'],
            'block': record['block'], 'attempt': record['attempt'], 'covered': covered,
        })
        if covered:
            group = groups[i]
            subtimes[record['seq']] = {
                'nl_write': group['nl_write'],
                'ipopt': group['ipopt'],
                'sol_parse': group['log_read'] + group['sol_read'],
            }
    coverage = {
        'status': 'exact' if exact else 'mismatch',
        'n_solve_bundle_records': len(sb_records),
        'n_report_timing_groups_complete': len(groups),
        'n_report_timing_groups_incomplete': n_incomplete,
        'per_record': per_record,
    }
    return subtimes, coverage


def _run_one(label, out_dir, eval_id, recorder=None, inject_report_timing=False, num_cycles=2):
    """Mirrors the committed `elif gate == 's35ref':` branch of
    `p515_g_g1_g4_admm_gates.py` (cited, not copied -- every called function
    below is `G.<name>`, the SAME object that branch calls), with
    `num_max_iters_override=num_cycles` (design §5's two-cycle preflight by
    default; `num_cycles` parameterizes this for the Planner's 10-cycle
    re-measurement, P5.15 Addendum 22 item (2) follow-up) instead of
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
                num_max_iters_override=num_cycles, eval_id=eval_id, post_run_hook=_hook,
                apply_rho=False, full_diagnostics_in_rows=True)

    if recorder is not None:
        import p515_s36_step36_timing as T
        with T.recorder_installed(recorder, inject_report_timing=inject_report_timing):
            report, path = _do_run()
    else:
        report, path = _do_run()

    return report, path, floor_sidecar_path


# Item 3 (Planner decision): `analyze_phase_timing`'s `x_threshold` has no
# default -- this script states the threshold it screens against explicitly,
# here, once, as the single source of truth for this entry point.
X_THRESHOLD = 0.70


def verdict_is_indeterminate(analysis):
    """Item 2 (Planner decision): the degraded-mode hard non-verdict is the
    literal string `p515_s36_step36_timing._DEGRADED_VERDICT`, never a bool --
    so any non-bool `verdict_pass` means the run must be treated as
    INDETERMINATE, not PASS/FAIL. A module-level function (not inlined in
    `main()`) so it can be unit-tested against a synthetic `analysis` dict
    without running the measurement."""
    return not isinstance(analysis.get('verdict_pass'), bool)


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


# P5.15 Addendum 22 item (2), timing-defects Worker task ("Also" item): the
# ONLY field excluded from the identity comparison below -- a path field that
# necessarily differs between the OFF and ON run's own output/eval
# directories (`.../evals/p515s36_timing_off*/logs/...` vs
# `.../evals/p515s36_timing_on*/logs/...`). Same technique the Planner
# already applied to `p515_s40_clone_capture_preflight.py` (commit
# `8f5cff48`): key-name exclusion, recursive at any depth, listed explicitly
# (never a wildcard/substring match). Confirmed against the already-captured
# 10-cycle evidence (`bitwise_identity_check.json`): all 66 leaf diffs are
# `esso_complementarity_diagnostics_by_round[...].log_path`, nothing else.
IDENTITY_EXCLUDED_FIELDS = ('log_path',)


def _strip_excluded_fields(obj, excluded):
    """Recursively rebuild `obj`, dropping any dict key in `excluded` at any
    depth. Only used for the identity comparison below -- never mutates a
    source file on disk. Same technique as
    `p515_s36_step36_timing_reanalyze.py::_strip_excluded_fields` (D4 fix)."""
    if isinstance(obj, dict):
        return {k: _strip_excluded_fields(v, excluded) for k, v in obj.items() if k not in excluded}
    if isinstance(obj, list):
        return [_strip_excluded_fields(v, excluded) for v in obj]
    return obj


def _bitwise_diff(report_off, report_on, floor_sidecar_off_path, floor_sidecar_on_path,
                   excluded_fields=IDENTITY_EXCLUDED_FIELDS):
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
    `excluded_fields` (default `IDENTITY_EXCLUDED_FIELDS`) is stripped,
    recursively, from BOTH sides before comparing -- see that constant's
    comment for why `log_path` is the one legitimate exclusion.
    Returns a dict with 'identical': bool and, if not, every field/row
    where the two runs diverge (capped at 50, as before).
    """
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

    soh_off = _strip_excluded_fields(_read_jsonl(floor_sidecar_off_path), excluded_fields)
    soh_on = _strip_excluded_fields(_read_jsonl(floor_sidecar_on_path), excluded_fields)
    if len(soh_off) != len(soh_on):
        diffs.append({'field': 'soh_floor_sidecar_length', 'off': len(soh_off), 'on': len(soh_on)})
    else:
        for i, (row_off, row_on) in enumerate(zip(soh_off, soh_on)):
            if row_off != row_on:
                diffs.append({'field': f'soh_floor_sidecar[{i}]', 'off': row_off, 'on': row_on})

    return {'identical': len(diffs) == 0, 'diffs': diffs[:50], 'n_diffs': len(diffs),
            'excluded_fields': list(excluded_fields)}


def main():
    """The measurement entry point (Planner decision, P5.15 Step 3.6 follow-up,
    item 4: the earlier `main()`/`main_()` fail-safe split is retired --
    `_check_preconditions()` IS the safety mechanism now, not a raise-unconditionally
    stub. `python p515_s36_step36_timing_run.py` runs this directly; it refuses to
    proceed unless every precondition passes (lock absent, no forbidden process
    alive, output dirs absent, the full file list clean in git)."""
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
        print(f'[P5.15-S36-TIMING] run OFF (recorder not installed) -- {NUM_CYCLES} cycles, '
              f'cold, s35ref config class. run_suffix={RUN_SUFFIX!r}')
        report_off, path_off, floor_off = _run_one(
            'off', OUT_OFF, _eval_id_for('off', NUM_CYCLES, RUN_SUFFIX), recorder=None,
            num_cycles=NUM_CYCLES)

        print(f'[P5.15-S36-TIMING] run ON (recorder installed, report_timing cross-check) -- '
              f'{NUM_CYCLES} cycles, cold.')
        recorder = T.PhaseTimingRecorder()
        report_on, path_on, floor_on = _run_one(
            'on', OUT_ON, _eval_id_for('on', NUM_CYCLES, RUN_SUFFIX), recorder=recorder,
            inject_report_timing=True, num_cycles=NUM_CYCLES)

        n_records = recorder.to_jsonl(os.path.join(OUT_ON, 'phase_timing_records.jsonl'))
        print(f'[P5.15-S36-TIMING] wrote {n_records} raw phase-timing records.')

        diff = _bitwise_diff(report_off, report_on, floor_off, floor_on)
        diff_path = os.path.join(OUT_ROOT, 'bitwise_identity_check.json')
        with open(diff_path, 'w') as handle:
            json.dump(diff, handle, indent=1, default=str)
        print(f'[P5.15-S36-TIMING] bitwise identity (off vs on, log_path excluded): '
              f'{diff["identical"]} ({diff["n_diffs"]} diffs) -- see {diff_path}')

        derived = T.derive_param_update_and_bookkeeping(recorder.records)
        all_records = recorder.records + derived
        stdout_on_path = os.path.join(OUT_ON, 'stdout_on.log')
        production_iter_wall = parse_cycle_wall_times(stdout_on_path)
        nl_write_total = parse_report_timing_nl_write_seconds(stdout_on_path)
        nl_write_seconds = {'aggregate': nl_write_total} if nl_write_total is not None else None

        # P5.15 Addendum 22 item (2), timing-defects Worker task (Defect 2
        # fix): actually build the D3 report_timing b/c/d1 split and pass it
        # to `analyze_phase_timing` -- previously this call never did, so
        # `solve_bundle_subtimes_fully_covered` was always False here (see
        # `build_solve_bundle_subtimes_with_coverage`'s docstring for the
        # full root-cause finding). Only pass it through when coverage is
        # exact; otherwise `analyze_phase_timing` falls back to its own
        # documented degraded mode, and the coverage shortfall is still
        # recorded in full (per record) under
        # `solve_bundle_subtime_coverage` below, never silently dropped.
        solve_bundle_subtimes, subtime_coverage = build_solve_bundle_subtimes_with_coverage(
            recorder.records, stdout_on_path)
        print(f"[P5.15-S36-TIMING] solve_bundle report_timing coverage: "
              f"status={subtime_coverage['status']} "
              f"n_solve_bundle_records={subtime_coverage['n_solve_bundle_records']} "
              f"n_report_timing_groups_complete={subtime_coverage['n_report_timing_groups_complete']} "
              f"n_report_timing_groups_incomplete={subtime_coverage['n_report_timing_groups_incomplete']}")

        analysis = T.analyze_phase_timing(
            all_records, x_threshold=X_THRESHOLD, production_iter_wall_by_cycle=production_iter_wall,
            projection_workers=8, nl_write_seconds=nl_write_seconds,
            solve_bundle_subtimes=(solve_bundle_subtimes if subtime_coverage['status'] == 'exact' else None))
        analysis['solve_bundle_subtime_coverage'] = subtime_coverage

        # P5.15 Addendum 22 item (2), timing-defects Worker task (Defect 2
        # fix): per-cycle analysis ("Include per cycle" deliverable) -- this
        # entry point never computed it before (only the 2-cycle-only
        # `p515_s36_step36_timing_reanalyze.py` did); re-run
        # `analyze_phase_timing` restricted to each SAMPLED cycle
        # individually, same pattern as that script's own per-cycle loop.
        per_cycle_analysis = {}
        for cycle in sorted(production_iter_wall.keys()):
            cycle_raw = [r for r in recorder.records if r['cycle'] == cycle]
            cycle_derived = T.derive_param_update_and_bookkeeping(cycle_raw)
            cycle_all = cycle_raw + cycle_derived
            per_cycle_analysis[cycle] = T.analyze_phase_timing(
                cycle_all, x_threshold=X_THRESHOLD,
                production_iter_wall_by_cycle={cycle: production_iter_wall[cycle]},
                projection_workers=8,
                solve_bundle_subtimes=(solve_bundle_subtimes if subtime_coverage['status'] == 'exact' else None))
        analysis['per_cycle_analysis'] = per_cycle_analysis

        analysis_path = os.path.join(OUT_ROOT, 'phase_timing_analysis.json')
        with open(analysis_path, 'w') as handle:
            json.dump(analysis, handle, indent=1, default=str)
        print(f'[P5.15-S36-TIMING] wrote {analysis_path}')
        print(f"[P5.15-S36-TIMING] verdict (X={X_THRESHOLD:.0%}): {analysis['verdict_pass']} "
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

    # Item 2 (Planner decision): a degraded (indeterminate) verdict is a hard
    # non-verdict -- everything above is still written in full, but the
    # process exits non-zero AFTER writing, so a degraded run can never be
    # mistaken for a decisive PASS/FAIL by an automated caller checking the
    # exit code alone.
    if verdict_is_indeterminate(analysis):
        print(f"[P5.15-S36-TIMING] verdict is INDETERMINATE ({analysis['verdict_pass']!r}) -- "
              f"exiting non-zero. All measured tables were still written above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
