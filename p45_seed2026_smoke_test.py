"""
P4.5 -- Seed-2026 distributed operational smoke test.

Runs the SAME reduced distributed-operational-only configuration used for
the post-vmag_nodes P2.10 smoke, through the unmodified production entry
point: SharedResourcesPlanning.run_operational_planning(type='distributed',
candidate_solution=...) with the exact same construction main.py uses
(same spec file, same candidate: s_inv=1.00 MVA, e_inv=3.00 MVAh, node 7,
investment year 2025). This is NOT the full Benders planning loop --
run_planning_problem() is never called here, exactly as in main.py (where
that call is commented out).

This script changes nothing in the production code and touches no solver,
ADMM, sess_comp, SOC, degradation, or ordinary/standard ESS settings. It
only:
  (a) calls the real, unmodified production functions with the same
      arguments main.py uses;
  (b) tees stdout to a log file while the run is in progress, purely to
      post-hoc count local primary failures / recoveries / persistent
      failures from the exact log lines _run_smopf() (network.py) already
      prints on every local SMOPF solve -- no parsing logic is inserted
      into production code, this only reads what it already prints;
  (c) serializes the structured per-cycle ADMM diagnostics
      (state['admm_diagnostics']) and the ESSO solver-recovery
      diagnostics (state['solver_recovery_diagnostics']) that the
      production code already computes and returns;
  (d) calls the existing, unmodified _get_tso_voltage_slack_state()
      production function once on the final TSO model, to summarize
      voltage-slack usage at the end of the run.

Writes: data/SRP1/Results/FrozenSMOPF/P45/p45_report.json
        data/SRP1/Results/FrozenSMOPF/P45/p45_console.log
"""
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / 'data' / 'SRP1' / 'Results' / 'FrozenSMOPF' / 'P45'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / 'p45_report.json'
LOG_PATH = OUTPUT_DIR / 'p45_console.log'

import shared_resources_planning as srp_module
from shared_resources_planning import SharedResourcesPlanning


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


class _Tee:
    """Write to both the real stdout and an in-memory buffer, so console
    output during the run is preserved for the user to watch live AND
    captured for post-hoc parsing. No production code is touched."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def parse_local_solve_events(log_text):
    """Post-hoc parse of the exact WARNING/INFO lines _run_smopf() in
    network.py already prints for every local TSO/DSO SMOPF solve."""

    primary_fail = re.findall(
        r"\[WARNING\] Network primary solve did not converge for ([^:]+): ([^|]*)\| warm_start=(\S+)",
        log_text,
    )
    solver_fail_no_recovery = re.findall(
        r"\[WARNING\] Network solver did not converge for ([^:]+): ([^|]*)\| warm_start=(\S+)",
        log_text,
    )
    recovery_success = re.findall(
        r"\[INFO\] Network recovery solve succeeded for ([^.]+)\.",
        log_text,
    )
    recovery_fail = re.findall(
        r"\[WARNING\] Network recovery solve did not converge for ([^:]+): ([^|]*)\| warm_start=(\S+)",
        log_text,
    )

    events = []
    for context, summary, warm in primary_fail:
        events.append({'context': context.strip(), 'stage': 'primary_failed_recovery_attempted', 'summary': summary.strip(), 'warm_start': warm})
    for context, summary, warm in solver_fail_no_recovery:
        events.append({'context': context.strip(), 'stage': 'primary_failed_no_recovery_attempted', 'summary': summary.strip(), 'warm_start': warm})
    for context, summary, warm in recovery_fail:
        events.append({'context': context.strip(), 'stage': 'recovery_failed_persistent', 'summary': summary.strip(), 'warm_start': warm})

    recovered_contexts = [c.strip() for c in recovery_success]

    n_primary_failures = len(primary_fail) + len(solver_fail_no_recovery)
    n_recovery_attempted = len(primary_fail)
    n_recovery_succeeded = len(recovery_success)
    n_recovery_failed = len(recovery_fail)
    n_no_recovery_failures = len(solver_fail_no_recovery)
    n_persistent_for_cycle = n_recovery_failed + n_no_recovery_failures

    return {
        'n_primary_local_failures': n_primary_failures,
        'n_recovery_attempted': n_recovery_attempted,
        'n_recovery_succeeded': n_recovery_succeeded,
        'n_recovery_failed_persistent': n_recovery_failed,
        'n_failed_no_recovery_attempted_persistent': n_no_recovery_failures,
        'n_persistent_for_cycle_total': n_persistent_for_cycle,
        'recovered_contexts': recovered_contexts,
        'events': events,
    }


def summarize_voltage_slack_state(vs_state):
    if not vs_state:
        return {'available': False, 'reason': 'voltage slacks disabled or empty state'}
    active = []
    max_fraction = 0.0
    for key, entry in vs_state.items():
        frac = entry.get('ub_fraction', 0.0) or 0.0
        if entry.get('slack_sqr', 0.0) and entry.get('slack_sqr', 0.0) > 1e-9:
            active.append(entry)
        if frac > max_fraction:
            max_fraction = frac
    active_sorted = sorted(active, key=lambda e: e.get('ub_fraction', 0.0), reverse=True)
    worst = active_sorted[:10]
    return {
        'available': True,
        'total_slack_rows': len(vs_state),
        'n_active_nonzero_slack_rows': len(active),
        'max_ub_fraction': max_fraction,
        'worst_10_rows': [
            {
                'node_id': e.get('node_id'),
                'market_scenario': e.get('market_scenario'),
                'operation_scenario': e.get('operation_scenario'),
                'period': e.get('period'),
                'direction': e.get('direction'),
                'slack_sqr': e.get('slack_sqr'),
                'ub_fraction': e.get('ub_fraction'),
                'vmag_kv': e.get('vmag_kv'),
                'v_min': e.get('v_min'),
                'v_max': e.get('v_max'),
            }
            for e in worst
        ],
    }


def main():
    data_dir = os.path.join(os.getcwd(), 'data', 'SRP1')
    spec_file = 'SRP1.json'
    spec_path = os.path.join(data_dir, spec_file)

    planning_problem = SharedResourcesPlanning(data_dir, spec_file)
    planning_problem.read_planning_problem()

    candidate_params = dict(s_inv=1.00, e_inv=3.00, node_id=7, investment_year=2025)
    candidate_solution = planning_problem.get_test_candidate_solution(**candidate_params)

    capture = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, capture)
    start = time.time()
    error_text = None
    try:
        convergence, results, models, sensitivities, primal_evolution, state = \
            planning_problem.run_operational_planning(
                type='distributed',
                candidate_solution=candidate_solution,
                print_results=True,
                debug_flag=False,
                filename='SRP1_operational_planning_results_distributed_P45_seed2026_smoke',
                return_state=True,
            )
    except Exception as exc:
        error_text = repr(exc)
        convergence, results, models, sensitivities, primal_evolution, state = (
            False, None, None, None, None, {'admm_diagnostics': [], 'solver_recovery_diagnostics': []}
        )
    finally:
        sys.stdout = real_stdout
    total_time = time.time() - start

    log_text = capture.getvalue()
    LOG_PATH.write_text(log_text)

    admm_diagnostics = state.get('admm_diagnostics', []) if isinstance(state, dict) else []
    solver_recovery_diagnostics = state.get('solver_recovery_diagnostics', []) if isinstance(state, dict) else []
    local_solve_summary = parse_local_solve_events(log_text)

    final_cycle = admm_diagnostics[-1] if admm_diagnostics else None

    vs_summary = {'available': False, 'reason': 'models unavailable (run failed before completion)'}
    if models is not None and models.get('tso') is not None:
        try:
            vs_state = srp_module._get_tso_voltage_slack_state(planning_problem, models['tso'])
            vs_summary = summarize_voltage_slack_state(vs_state)
        except Exception as exc:
            vs_summary = {'available': False, 'error': repr(exc)}

    rho_evolution = [
        {
            'cycle': d['cycle'],
            'rho_v_before': d['rho_v_before'], 'rho_v_after': d['rho_v_after'], 'rho_v_action': d['rho_v_action'],
            'rho_pf_before': d['rho_pf_before'], 'rho_pf_after': d['rho_pf_after'], 'rho_pf_action': d['rho_pf_action'],
            'rho_ess_before': d['rho_ess_before'], 'rho_ess_after': d['rho_ess_after'], 'rho_ess_action': d['rho_ess_action'],
        }
        for d in admm_diagnostics
    ]

    report = {
        'error': error_text,
        'initialization_failed': state.get('initialization_failed', False) if isinstance(state, dict) else None,
        'convergence': convergence,
        'n_admm_cycles': len(admm_diagnostics),
        'total_execution_time_s': total_time,
        'candidate_solution_params': candidate_params,
        'random_seed': getattr(planning_problem, 'random_seed', None),
        'spec_file_sha256': sha256_of(spec_path),
        'local_solve_event_summary': local_solve_summary,
        'esso_solver_recovery_diagnostics': solver_recovery_diagnostics,
        'admm_diagnostics_all_cycles': admm_diagnostics,
        'rho_evolution': rho_evolution,
        'final_cycle_summary': final_cycle,
        'final_voltage_slack_state_summary': vs_summary,
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print('\n[P45] Report written to', REPORT_PATH)
    print('[P45] Console log written to', LOG_PATH)
    print('[P45] convergence =', convergence, '| cycles =', len(admm_diagnostics), '| time =', total_time)
    print('[P45] primary local failures =', local_solve_summary['n_primary_local_failures'],
          '| persistent-for-cycle =', local_solve_summary['n_persistent_for_cycle_total'])


if __name__ == '__main__':
    main()
