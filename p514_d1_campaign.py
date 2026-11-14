"""
Track D1 — three-cell campaign. Frozen spec:
    data/SRP1/Results/P514D/frozen_d1_campaign_v1_2c8c7bef.json

    python p514_d1_campaign.py <cell>    # cell1 | cell2 | cell3

Overrides (predeclared varied settings, harness-only, no file edited):
    budget = 5.0e6            derived so max_capacity binds before the budget at every ratio
    admm.tol.objective.rel    = 1e-4
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p512_a_cold_rescaled_convergence as A  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514D')
SPEC = 'data/SRP1/Results/P514D/frozen_d1_campaign_v1_2c8c7bef.json'
BUDGET, REL, CAP = 5.0e6, 1e-4, 90
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
S_INV, E_INV, INVEST_YEAR = 1.00, 4.00, 2025
# P5.14-M: capacity overridable on the command line for the C* evaluation
VANISH = 1e-6
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]
SLACK_PREFIXES = ('slack_', 'penalty_')


def apply_overrides(planning):
    planning.params.admm.num_max_iters = CAP
    planning.params.admm.tol['objective']['rel'] = REL
    planning.shared_ess_data.params.budget = BUDGET
    return {'budget': planning.shared_ess_data.params.budget,
            'tol_objective': dict(planning.params.admm.tol['objective']),
            'tol_stationarity': dict(planning.params.admm.tol['stationarity']),
            'tol_consensus': dict(planning.params.admm.tol['consensus']),
            'num_max_iters': planning.params.admm.num_max_iters}


def harvest_slacks(models):
    """Every slack/penalty Var in every model, named individually with its max |value|."""
    found = {}
    def walk(model):
        for comp in model.component_objects(pe.Var, active=None):
            name = comp.local_name
            if not name.startswith(SLACK_PREFIXES):
                return_value = None
            if name.startswith(SLACK_PREFIXES):
                worst = 0.0
                for key in comp:
                    try:
                        value = pe.value(comp[key], exception=False)
                    except Exception:
                        value = None
                    if value is not None:
                        worst = max(worst, abs(value))
                prev = found.get(name, {'max_abs': 0.0, 'n_components': 0})
                found[name] = {'max_abs': max(prev['max_abs'], worst),
                               'n_components': prev['n_components'] + 1}
    def descend(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                descend(value)
        elif hasattr(obj, 'component_objects'):
            walk(obj)
    descend(models)
    for name, entry in found.items():
        entry['non_vanishing'] = entry['max_abs'] > VANISH
        entry['threshold'] = VANISH
    return found


def main(cell):
    os.makedirs(OUT, exist_ok=True)
    guard = SolveProfileGuard(PERMITTED, label=f'D1 {cell}').install()
    started = time.time()
    report = {'stage': 'Track D1', 'cell': cell, 'spec': SPEC,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'predeclared_varied_settings': {'budget': BUDGET, 'tol_objective_rel': REL}}
    try:
        with redirect_stdout(io.StringIO()) as console:
            planning = O.fresh_planning(f'd1_{cell}')
            report['effective_configuration'] = apply_overrides(planning)
            if cell == 'cell1':
                report['mode'] = "uncoordinated (no consensus loop, no ESSO)"
                results, models = planning.run_operational_planning(
                    type='uncoordinated', print_results=False)
                state = None
            else:
                RH.apply_rho_to_params(planning, RHO)
                RH.set_adaptive_penalty(planning, True)
                report['effective_configuration']['rho'] = RHO
                report['effective_configuration']['adaptive_penalty'] = True
                report['mode'] = 'cold ADMM (distributed)'
                if cell == 'cell2':
                    candidate = planning.get_initial_candidate_solution()
                else:
                    candidate = planning.get_initial_candidate_solution()
                    for node_id in planning.shared_ess_data.active_distribution_network_nodes:
                        candidate['investment'][node_id][INVEST_YEAR]['s'] = S_INV
                        candidate['investment'][node_id][INVEST_YEAR]['e'] = E_INV
                    srp._rebuild_candidate_total_capacities(planning, candidate)
                report['instance'] = {
                    'investment': {str(n): {str(y): dict(v) for y, v in d.items()}
                                   for n, d in candidate['investment'].items()},
                    'total_capacity': {str(n): {str(y): dict(v) for y, v in d.items()}
                                       for n, d in candidate.get('total_capacity', {}).items()}}
                feasible, reason = srp._check_candidate_first_stage_feasibility(planning, candidate)
                report['first_stage_feasible'] = {'feasible': feasible, 'reason': reason}
                with R.patched_admm_objectives():
                    _c, results, models, _s, _p, state = planning.run_operational_planning(
                        type='distributed', candidate_solution=deepcopy(candidate),
                        print_results=False, debug_flag=False, return_state=True)
        report['console_tail'] = console.getvalue()[-1500:]
    finally:
        guard.uninstall()
        report['wall_clock_s'] = time.time() - started

    if state is not None:
        rows = [A.cycle_row(e, None) for e in (state.get('admm_diagnostics') or [])]
        report['cycles'] = rows
        report['cycles_run'] = len(rows)
        report['converged_at_cycle'] = next((r['cycle'] for r in rows if r['cycle_convergence']), None)
        report['hit_cap'] = report['converged_at_cycle'] is None and len(rows) >= CAP
        last = rows[-1] if rows else {}
        report['recourse'] = last.get('recourse')
        report['gross_operational_cost'] = last.get('gross_operational_cost')
        report['terminal_objective_change_abs'] = last.get('objective_change_abs')
        report['terminal_objective_tolerance'] = last.get('objective_tolerance')
        report['rule_ten_terminal_step_over_threshold'] = (
            last.get('objective_change_abs') / last.get('objective_tolerance')
            if last.get('objective_change_abs') and last.get('objective_tolerance') else None)
        report['local_solve_failures'] = sum(1 for r in rows if r.get('local_solves_ok') is False)
        report['final_rho_pf'] = (state.get('admm_diagnostics') or [{}])[-1].get('rho_pf_after')
    report['slacks_and_penalties'] = harvest_slacks(models)
    report['non_vanishing_terms'] = sorted(
        n for n, e in report['slacks_and_penalties'].items() if e['non_vanishing'])
    report['solve_profile'] = {'observed': dict(guard.counts),
                               'blocked': guard.counts['blocked_solve'] + guard.counts['blocked_exec']}
    path = os.path.join(OUT, f'd1_{cell}.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f"[D1 {cell}] solves={guard.counts['permitted_solve']} "
          f"cycles={report.get('cycles_run')} converged={report.get('converged_at_cycle')} "
          f"recourse={report.get('recourse')} gross={report.get('gross_operational_cost')} "
          f"non_vanishing={report['non_vanishing_terms']} wall={report['wall_clock_s']:.0f}s")
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2:                      # optional capacity override (P5.14-M)
        S_INV = float(sys.argv[2])
        E_INV = S_INV * 4.0
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'cell1'))
