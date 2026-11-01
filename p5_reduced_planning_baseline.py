"""
Stage P5 -- reduced planning baseline with the accepted P4 production formulation.

Runs the REAL production planning workflow

    SharedResourcesPlanning('data/SRP1', 'SRP1.json').run_planning_problem()

on the established reduced research baseline (seed 2026; years 2025/2030/2035;
Spring/Summer/Autumn/Winter; TSO case9; DSOs case33_1@5, case33_2@7,
case33_3@9), and collects the evidence the P5 specification asks for.

This is an integration/stability baseline: NOTHING is tuned. No formulation,
solver, ADMM, Benders or ESS setting is touched. The script only

  - tees the production console output to a log file,
  - post-hoc parses the exact INFO/WARNING lines production already prints,
  - serializes a JSON report.

Failure-event parsing reuses `p45_seed2026_smoke_test.parse_local_solve_events`
(the accepted P4.5 parser) rather than duplicating it.

    python p5_reduced_planning_baseline.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p45_seed2026_smoke_test import parse_local_solve_events  # noqa: E402  (reuse)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P5')
LOG_PATH = os.path.join(OUT_DIR, 'p5_console.log')
REPORT_PATH = os.path.join(OUT_DIR, 'p5_report.json')


class Tee:
    """Write production stdout to both the console (progress visibility) and a
    log file (post-hoc parsing). Purely an observer."""

    def __init__(self, stream, path):
        self.stream = stream
        self.handle = open(path, 'w', buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.handle.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def close(self):
        self.handle.close()


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO_ROOT).decode()
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Parsers over the exact lines production already prints
# ---------------------------------------------------------------------------
ITER_RE = re.compile(
    r"\[INFO\] Iteration #(\d+) \| Source = ([^|]*)\| Master = ([^|]*)\| Alpha = ([^|]*)\| "
    r"Investment = ([^|]*)\| Gross recourse = ([^|]*)\| Salvage = ([^|]*)\| Net recourse = ([^|]*)\| "
    r"Candidate = ([^|]*)\| UB = ([^|]*)\| Gap = ([^|]*)\| ESSO violation = (.*)")
TERMINATION_RE = re.compile(r"\[INFO\] Planning termination reason: (\S+?)\.")
EXEC_TIME_RE = re.compile(r"\[INFO\] Execution time: ([0-9.]+) s")
ADMM_CONVERGED_RE = re.compile(r"\[INFO\] \s*- ADMM converged in (\d+) iteration\(s\)\.")
ADMM_ITER_RE = re.compile(r"\[INFO\] \s*- ADMM Iteration (\d+)")
CONVERGED_AT_RE = re.compile(r"\[INFO\] Benders-type procedure converged at iteration (\d+)\.")
STATIONARITY_OK_RE = re.compile(r"Recourse stationarity ok!")
STATIONARITY_FAIL_RE = re.compile(r"Recourse stationarity failed\. ([0-9.eE+-]+) > ([0-9.eE+-]+)")
STATIONARITY_NA_RE = re.compile(r"Recourse stationarity (unavailable|requires)")
PRIMAL_OK_RE = re.compile(r"Primal residuals ok!")
DUAL_OK_RE = re.compile(r"Dual residuals ok!")
PRIMAL_FAIL_RE = re.compile(r"(\S[^\n]*?) (max|mean) primal residual failed\. ([0-9.eE+-]+) > ([0-9.eE+-]+)")
DUAL_FAIL_RE = re.compile(r"(\S[^\n]*?) mean dual residual failed\. ([0-9.eE+-]+) > ([0-9.eE+-]+)")
WARN_RE = re.compile(r"\[WARNING\] (.+)")


def parse_planning_log(text):
    iterations = []
    for m in ITER_RE.finditer(text):
        iterations.append({
            'iteration': int(m.group(1)),
            'candidate_source': m.group(2).strip(),
            'master_estimate': m.group(3).strip(),
            'alpha': m.group(4).strip(),
            'investment_cost': m.group(5).strip(),
            'gross_recourse': m.group(6).strip(),
            'salvage': m.group(7).strip(),
            'net_recourse': m.group(8).strip(),
            'candidate_total': m.group(9).strip(),
            'upper_bound': m.group(10).strip(),
            'gap': m.group(11).strip(),
            'esso_violation': m.group(12).strip(),
        })

    admm_converged = [int(m.group(1)) for m in ADMM_CONVERGED_RE.finditer(text)]
    admm_iter_marks = [int(m.group(1)) for m in ADMM_ITER_RE.finditer(text)]

    # count ADMM "runs": a run ends whenever a converged line appears; a run that
    # never converges shows as a cycle sequence with no converged line after it
    n_admm_runs_converged = len(admm_converged)

    warnings = {}
    for m in WARN_RE.finditer(text):
        key = re.sub(r"[0-9][0-9.eE+-]*", "<num>", m.group(1).strip())[:160]
        warnings[key] = warnings.get(key, 0) + 1

    stationarity = {
        'ok_count': len(STATIONARITY_OK_RE.findall(text)),
        'failed': [{'change': a, 'tolerance': b} for a, b in STATIONARITY_FAIL_RE.findall(text)],
        'unavailable_count': len(STATIONARITY_NA_RE.findall(text)),
    }
    residuals = {
        'primal_ok_count': len(PRIMAL_OK_RE.findall(text)),
        'dual_ok_count': len(DUAL_OK_RE.findall(text)),
        'primal_failures': [
            {'group': g.strip()[-40:], 'kind': k, 'value': v, 'tolerance': t}
            for g, k, v, t in PRIMAL_FAIL_RE.findall(text)][:50],
        'dual_failures': [
            {'group': g.strip()[-40:], 'value': v, 'tolerance': t}
            for g, v, t in DUAL_FAIL_RE.findall(text)][:50],
    }

    termination = TERMINATION_RE.findall(text)
    converged_at = CONVERGED_AT_RE.findall(text)
    exec_time = EXEC_TIME_RE.findall(text)

    return {
        'outer_iterations': iterations,
        'n_outer_iterations': len(iterations),
        'termination_reason': termination[-1] if termination else None,
        'benders_converged_at_iteration': int(converged_at[-1]) if converged_at else None,
        'production_execution_time_s': float(exec_time[-1]) if exec_time else None,
        'admm_runs_converged': n_admm_runs_converged,
        'admm_cycles_per_converged_run': admm_converged,
        'admm_max_cycle_marker_seen': max(admm_iter_marks) if admm_iter_marks else None,
        'recourse_stationarity': stationarity,
        'admm_residual_checks': residuals,
        'warning_histogram': dict(sorted(warnings.items(), key=lambda kv: -kv[1])),
    }


def failure_pattern_summary(events):
    """Group failure contexts by network/year/day to expose repeated patterns."""
    pattern = {}
    for event in events:
        ctx = event['context']
        pattern.setdefault(ctx, {'count': 0, 'stages': {}})
        pattern[ctx]['count'] += 1
        stage = event['stage']
        pattern[ctx]['stages'][stage] = pattern[ctx]['stages'].get(stage, 0) + 1
    ranked = sorted(pattern.items(), key=lambda kv: -kv[1]['count'])
    flagged = {
        'case33_2_node7_family': sum(
            v['count'] for k, v in pattern.items() if 'case33_2' in k),
        'tso_case9_family': sum(
            v['count'] for k, v in pattern.items() if 'case9' in k),
    }
    return {'by_context': dict(ranked), 'flagged_families': flagged}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    report = {
        'stage': 'P5',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': git_head(),
        'git_tracked_dirty': git_dirty(),
        'command': f"SharedResourcesPlanning('{SPEC_DIR}', '{SPEC_FILE}').run_planning_problem()",
        'log_path': LOG_PATH,
    }

    tee = Tee(sys.__stdout__, LOG_PATH)
    original_stdout = sys.stdout
    sys.stdout = tee
    started = time.time()
    outcome = {'completed': False, 'exception': None, 'system_exit_code': None}
    try:
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        report['effective_config'] = {
            'years': [str(y) for y in planning.years],
            'days': [str(d) for d in planning.days],
            'num_instants': planning.num_instants,
            'random_seed': getattr(planning, 'random_seed', None),
            'transmission_network': planning.transmission_network.name,
            'distribution_networks': sorted(str(k) for k in planning.distribution_networks),
        }
        planning.run_planning_problem()
        outcome['completed'] = True
    except SystemExit as exc:
        outcome['system_exit_code'] = exc.code
    except Exception as exc:  # noqa: BLE001 -- record, never swallow silently
        outcome['exception'] = f'{type(exc).__name__}: {exc}'
    finally:
        wall = time.time() - started
        sys.stdout = original_stdout
        tee.flush()
        tee.close()

    with open(LOG_PATH, 'r', errors='replace') as handle:
        log_text = handle.read()

    report['outcome'] = outcome
    report['wall_clock_s'] = wall
    report['planning'] = parse_planning_log(log_text)
    events = parse_local_solve_events(log_text)
    report['local_solve_events'] = events
    report['failure_patterns'] = failure_pattern_summary(events['events'])

    checksum = re.findall(r"\[INFO\] Scenario checksum: (\S+)", log_text)
    seed_line = re.findall(r"\[INFO\] Scenario random seed: (\S+)", log_text)
    report['scenario_checksum'] = checksum[-1] if checksum else None
    report['scenario_seed'] = seed_line[-1] if seed_line else None

    with open(REPORT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    p = report['planning']
    print(f"\n[P5] report -> {REPORT_PATH}")
    print(f"[P5] log    -> {LOG_PATH}")
    print(f"[P5] completed={outcome['completed']} exit={outcome['system_exit_code']} "
          f"exception={outcome['exception']}")
    print(f"[P5] wall clock {wall:.1f} s | production exec {p['production_execution_time_s']} s")
    print(f"[P5] outer iterations={p['n_outer_iterations']} termination={p['termination_reason']} "
          f"converged_at={p['benders_converged_at_iteration']}")
    print(f"[P5] ADMM runs converged={p['admm_runs_converged']} cycles={p['admm_cycles_per_converged_run']}")
    print(f"[P5] local primary failures={events['n_primary_local_failures']} "
          f"recoveries_succeeded={events['n_recovery_succeeded']} "
          f"persistent={events['n_persistent_for_cycle_total']}")
    print(f"[P5] failure families: {report['failure_patterns']['flagged_families']}")
    for it in p['outer_iterations']:
        print(f"   iter {it['iteration']}: src={it['candidate_source']} master={it['master_estimate']} "
              f"inv={it['investment_cost']} net_rec={it['net_recourse']} cand={it['candidate_total']} "
              f"UB={it['upper_bound']} gap={it['gap']} esso_viol={it['esso_violation']}")


if __name__ == '__main__':
    main()
