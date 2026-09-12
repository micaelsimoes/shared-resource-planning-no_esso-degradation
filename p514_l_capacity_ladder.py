"""
P5.14-L — capacity ladder. Frozen spec:
    data/SRP1/Results/P514L/frozen_ladder_v1_c1ea6393.json

Replicates the initialization block of _run_operational_planning verbatim and STOPS there;
ADMM is never started. For any failing ESSO subproblem the model is re-solved once with the
solution loaded manually, so per-constraint residuals can be read at the terminal point.

    python p514_l_capacity_ladder.py <s_mva>
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe
import pyomo.opt as po

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514L')
SPEC = 'data/SRP1/Results/P514L/frozen_ladder_v1_c1ea6393.json'
RATIO, INVEST_YEAR = 4.0, 2025
BUDGET, REL, CAP = 5.0e6, 1e-4, 90
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt'),
             ('p514_l_capacity_ladder.py', 'diagnostic_resolve')]
AGG_ROWS = ('energy_storage_operation_agg',)
COMP_ROWS = ('energy_storage_complementarity',)


def diagnostic_resolve(model, params):
    """Re-solve one ESSO model and load the terminal point, feasible or not."""
    solver = po.SolverFactory(params.solver, executable=params.solver_path)
    for key, value in (params.options or {}).items():
        solver.options[key] = value
    result = solver.solve(model, tee=False, load_solutions=False)
    loaded = True
    try:
        model.solutions.load_from(result)
    except Exception:
        loaded = False
    return result, loaded


def residuals(model, top=12):
    """Per-row violation of every constraint, at the model's current point."""
    rows = []
    for comp in model.component_objects(pe.Constraint, active=True):
        for key in comp:
            con = comp[key]
            try:
                body = pe.value(con.body, exception=False)
            except Exception:
                continue
            if body is None:
                continue
            violation = 0.0
            if con.has_ub():
                violation = max(violation, body - pe.value(con.upper))
            if con.has_lb():
                violation = max(violation, pe.value(con.lower) - body)
            if violation > 0:
                rows.append({'component': comp.local_name, 'key': str(key),
                             'violation': violation, 'body': body})
    rows.sort(key=lambda r: -r['violation'])
    by_component = {}
    for row in rows:
        entry = by_component.setdefault(row['component'], {'n_rows': 0, 'max_violation': 0.0})
        entry['n_rows'] += 1
        entry['max_violation'] = max(entry['max_violation'], row['violation'])
    return {'total_violated_rows': len(rows), 'worst': rows[:top], 'by_component': by_component}


def main(s_mva):
    s_mva = float(s_mva)
    e_mwh = s_mva * RATIO
    os.makedirs(OUT, exist_ok=True)
    guard = SolveProfileGuard(PERMITTED, label=f'ladder s={s_mva}').install()
    started = time.time()
    report = {'stage': 'P5.14-L', 'spec': SPEC, 's_mva': s_mva, 'e_mwh': e_mwh,
              'ratio': RATIO, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'nodes': {}}
    try:
        with redirect_stdout(io.StringIO()):
            planning = O.fresh_planning(f'ladder_s{s_mva:g}')
            planning.params.admm.num_max_iters = CAP
            planning.params.admm.tol['objective']['rel'] = REL
            planning.shared_ess_data.params.budget = BUDGET
            RH.apply_rho_to_params(planning, RHO)
            RH.set_adaptive_penalty(planning, True)
            sed = planning.shared_ess_data

            candidate = planning.get_initial_candidate_solution()
            for node_id in sed.active_distribution_network_nodes:
                candidate['investment'][node_id][INVEST_YEAR]['s'] = s_mva
                candidate['investment'][node_id][INVEST_YEAR]['e'] = e_mwh
            srp._rebuild_candidate_total_capacities(planning, candidate)
            feasible, reason = srp._check_candidate_first_stage_feasibility(planning, candidate)

            results = {}
            with R.patched_admm_objectives():
                consensus_vars, _dual = srp.create_admm_variables(planning)
                _dso, results['dso'] = srp.create_distribution_networks_models(
                    planning.distribution_networks, consensus_vars,
                    candidate['total_capacity'],
                    parallel_execution=planning.parallel_execution)
                _tso, results['tso'] = srp.create_transmission_network_model(
                    planning, consensus_vars, candidate['total_capacity'])
                esso_models, results['esso'] = srp.create_shared_energy_storage_model(
                    sed, consensus_vars, candidate['investment'])

            solves_after_init = guard.counts['permitted_solve']
            for node_id in sorted(sed.active_distribution_network_nodes):
                res = results['esso'][node_id]
                ok = srp._solver_result_succeeded(res)
                entry = {'succeeded': bool(ok),
                         'termination': str(res.solver.termination_condition),
                         'status': str(res.solver.status)}
                if not ok:
                    _r, loaded = diagnostic_resolve(esso_models[node_id],
                                                    sed.params.solver_params)
                    entry['diagnostic_resolve_loaded'] = loaded
                    entry['residuals'] = residuals(esso_models[node_id])
                    by = entry['residuals']['by_component']
                    entry['aggregate_complementarity_violated'] = any(
                        name in by for name in AGG_ROWS)
                    entry['cohort_complementarity_violated'] = any(
                        name in by for name in COMP_ROWS)
                report['nodes'][str(node_id)] = entry
    finally:
        guard.uninstall()

    report['first_stage_feasible'] = {'feasible': feasible, 'reason': reason}
    report['all_esso_succeeded'] = all(v['succeeded'] for v in report['nodes'].values())
    report['failed_nodes'] = [n for n, v in report['nodes'].items() if not v['succeeded']]
    report['solve_profile'] = {'observed': dict(guard.counts),
                               'initialization_solves': solves_after_init,
                               'diagnostic_solves': guard.counts['permitted_solve'] - solves_after_init,
                               'blocked': guard.counts['blocked_solve']}
    report['wall_clock_s'] = time.time() - started
    path = os.path.join(OUT, f'ladder_s{s_mva:g}.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f"[ladder s={s_mva:g} e={e_mwh:g}] all_ok={report['all_esso_succeeded']} "
          f"failed={report['failed_nodes']} solves={guard.counts['permitted_solve']} "
          f"wall={report['wall_clock_s']:.0f}s")
    for node, v in report['nodes'].items():
        if not v['succeeded']:
            by = v.get('residuals', {}).get('by_component', {})
            print(f"   node {node}: {v['termination']}, violated components: "
                  f"{ {k: round(x['max_violation'], 8) for k, x in by.items()} }")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 1.00))
