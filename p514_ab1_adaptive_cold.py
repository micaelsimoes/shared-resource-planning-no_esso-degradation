"""
AB1 — adaptive rho on the COLD path. Two arms, one varied setting.

Frozen design: data/SRP1/Results/P514S2/frozen_ab1_adaptive_cold_v2_6e23cac0.json

  control    adaptive_penalty = False, cold, rho_pf = 300, cap 50
  treatment  adaptive_penalty = True,  cold, rho_pf = 300, cap 50

The control is a DELIVERABLE, not merely a control: it is the R trajectory finally
allowed to finish, and it answers what the cold path actually costs.

The cap comes from the hypothesis, not from the R run's inherited num_max_iters = 21.
EITHER arm reaching the cap is predeclared INCONCLUSIVE for that arm: no cold-path cost
number may be extrapolated from a capped run.

Production machinery is reused unchanged (p512_a's cold-path recipe); the only varied
setting is the adaptive flag.

    python p514_ab1_adaptive_cold.py <arm>        # arm in {control, treatment}
"""

import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p512_a_cold_rescaled_convergence as A  # noqa: E402  (cycle_row, RHO)
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P514AB1')
SPEC = 'data/SRP1/Results/P514S2/frozen_ab1_adaptive_cold_v2_6e23cac0.json'
CAP = 50
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
SOLVES_PER_CYCLE = 51
ARMS = {'control': False, 'treatment': True}
# Verified against the source before arming: network.py:543 calls solver.solve inside
# _run_smopf_solver_attempt, and shared_energy_storage_data.py:928 inside
# _run_solver_attempt. A wrong name here would block every solve.
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]


def rho_row(entry):
    """Production already records the penalty trace; take it verbatim."""
    return {k: entry.get(k) for k in (
        'cycle', 'rho_v_before', 'rho_pf_before', 'rho_ess_before',
        'rho_v_after', 'rho_pf_after', 'rho_ess_after',
        'rho_v_action', 'rho_pf_action', 'rho_ess_action',
        'primal_pf_ratio', 'dual_pf_mean_ratio', 'primal_v_ratio', 'dual_v_mean_ratio')}


def main(arm):
    if arm not in ARMS:
        raise SystemExit(f'arm must be one of {sorted(ARMS)}')
    adaptive = ARMS[arm]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'ab1_{arm}.json')

    report = {'stage': 'AB1', 'arm': arm, 'spec': SPEC,
              'adaptive_penalty': adaptive, 'rho_requested': RHO, 'cap': CAP,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'single_varied_setting': 'adaptive_penalty',
              'cycles': [], 'rho_trace': []}

    guard = SolveProfileGuard(PERMITTED, label=f'AB1 {arm}').install()
    started = time.time()
    try:
        console = io.StringIO()
        with redirect_stdout(console):
            planning = O.fresh_planning(f'ab1_{arm}')
            report['rho_params_replaced'] = RH.apply_rho_to_params(planning, RHO)
            report['adaptive_before'] = RH.set_adaptive_penalty(planning, adaptive)
            report['num_max_iters_before'] = planning.params.admm.num_max_iters
            planning.params.admm.num_max_iters = CAP
            report['penalty_update_in_force'] = dict(planning.params.admm.penalty_update)

            candidate = srp._build_positive_bootstrap_candidate(
                planning, planning.params.benders.positive_bootstrap)
            with R.patched_admm_objectives() as applied:
                convergence, _, _models, _, _, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
        report['blocks_rescaled_at_build'] = len([v for v in applied.values() if v])
    finally:
        guard.uninstall()
        report['wall_clock_s'] = time.time() - started

    report['production_reported_convergence'] = bool(convergence)
    previous = None
    for entry in (state.get('admm_diagnostics') or []):
        report['cycles'].append(A.cycle_row(entry, previous))
        report['rho_trace'].append(rho_row(entry))
        if entry.get('recourse') is not None:
            previous = entry['recourse']

    cycles_run = len(report['cycles'])
    converged_at = next((c['cycle'] for c in report['cycles'] if c['cycle_convergence']), None)
    report['cycles_run'] = cycles_run
    report['converged_at_cycle'] = converged_at
    report['hit_cap'] = converged_at is None and cycles_run >= CAP
    report['primary_outcome'] = {
        'cycles_to_convergence': converged_at,
        'implied_local_solves': (converged_at * SOLVES_PER_CYCLE) if converged_at else None,
        'status': ('INCONCLUSIVE — hit the cap; no cold-path cost may be extrapolated'
                   if report['hit_cap'] else 'converged' if converged_at else 'stopped without convergence'),
    }
    final_rho = report['rho_trace'][-1].get('rho_pf_after') if report['rho_trace'] else None
    report['final_rho_pf'] = final_rho
    report['attractor_secondary'] = {
        'predicted_band': [80, 200], 'predicted_most_likely': 133.33,
        'observed_final_rho_pf': final_rho,
        'in_band': (80 <= final_rho <= 200) if isinstance(final_rho, (int, float)) else None,
        'decrease_actions': sum(1 for r in report['rho_trace'] if r.get('rho_pf_action') == 'decreased'),
        'increase_actions': sum(1 for r in report['rho_trace'] if r.get('rho_pf_action') == 'increased'),
    }
    observed = guard.counts['permitted_solve']
    expected = SOLVES_PER_CYCLE * cycles_run
    report['solve_profile'] = {
        'observed': dict(guard.counts), 'upper_bound': CAP * SOLVES_PER_CYCLE,
        'identity_expected_51x_cycles': expected,
        'identity_holds': observed == expected,
        'note': 'initialization solves may sit outside the identity; a mismatch is reported, not absorbed',
    }
    report['any_local_solve_failure'] = any(c['local_solves_ok'] is False for c in report['cycles'])
    report['final_recourse'] = report['cycles'][-1]['recourse'] if report['cycles'] else None

    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[AB1 {arm}] cycles_run={cycles_run} converged_at={converged_at} '
          f'hit_cap={report["hit_cap"]} final_rho_pf={final_rho} '
          f'solves={observed} wall={report["wall_clock_s"]:.0f}s -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'control'))
