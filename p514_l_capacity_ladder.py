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
from types import SimpleNamespace

import pyomo.environ as pe
import pyomo.opt as po

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_energy_storage_data as sesd  # noqa: E402
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
# P5.15-1 repair (Addendum-2 task a): `energy_storage_complementarity` was DELETED
# by the ESSO reformulation (commit b03c9b14) -- complementarity is no longer an
# enforced constraint row, so a row-name lookup can never fire again (silently
# vacuous). It is replaced below by the post-solve DETECTOR
# `SharedEnergyStorageData.get_complementarity_violation` (implementation
# `_get_complementarity_violation`, shared_energy_storage_data.py:1433), called
# with the exact signature the Planner specified: `shared_ess_data.get_
# complementarity_violation(models)`, over the full esso_models dict -- this
# reports ONE violation across all active nodes/cohorts/periods, not a
# per-node value, because that is the function's own signature.


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


def node_complementarity_absolute(shared_ess_data, esso_models, node_id):
    """Per-node absolute complementarity violation, scoped to a single node.

    `get_complementarity_violation` takes the whole model dict and loops over
    `shared_ess_data.active_distribution_network_nodes` internally, so it always
    returns one violation across ALL active nodes. To read a single node's value
    without mutating the real `shared_ess_data` instance, this calls the SAME
    underlying detector function (`_get_complementarity_violation`) with a
    throwaway proxy object exposing only the one attribute it reads
    (`active_distribution_network_nodes`). No arithmetic is duplicated; the
    detector's own filtering (`_esso_cohort_pair_is_within_lifetime`,
    `_esso_cohort_inactive`) and threshold (1e-6 * s_max) are reused verbatim.
    """
    proxy = SimpleNamespace(active_distribution_network_nodes=[node_id])
    return sesd._get_complementarity_violation(proxy, esso_models)


def node_complementarity_ratio(esso_models, node_id):
    """Harness-only ratio form: max over the SAME active cohort-periods the
    detector scans of min(pch, pdch) / s_max, skipping any pair with s_max == 0.
    Not produced by production code (per task instructions); reuses the
    detector's own active-pair filter (`_esso_cohort_pair_is_within_lifetime`,
    `_esso_cohort_inactive`) so the set of scanned (y_inv, y, d, p) is identical
    to the absolute-violation scan above, only the reported quantity differs.
    """
    model = esso_models[node_id]
    max_ratio = 0.0
    for y_inv in model.years:
        for y in model.years:
            if not sesd._esso_cohort_pair_is_within_lifetime(model, y_inv, y):
                continue
            if model._esso_cohort_inactive.get(y_inv, False):
                continue
            s_max = pe.value(model.es_s_rated_per_unit[y_inv, y])
            if s_max == 0:
                continue
            for d in model.days:
                for p in model.periods:
                    pch = pe.value(model.es_pch_per_unit[y_inv, y, d, p])
                    pdch = pe.value(model.es_pdch_per_unit[y_inv, y, d, p])
                    ratio = min(pch, pdch) / s_max
                    if ratio > max_ratio:
                        max_ratio = ratio
    return max_ratio


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
                    # P5.15-1 repair: `energy_storage_complementarity` no longer
                    # exists as a constraint row (deleted by the reformulation),
                    # so this is now the post-solve DETECTOR read AT THE SAME
                    # diagnostic-resolved terminal point the residuals above use,
                    # scoped to this one node (see node_complementarity_absolute).
                    # `cohort_complementarity_violated` changes meaning from "row
                    # X appears among violated constraints" (row-existence) to
                    # "the detector's own tolerance (1e-6 * s_max) is exceeded
                    # somewhere in this node's active cohort-periods" -- reported
                    # here as a SEMANTIC CHOICE for the Planner, not a mechanical
                    # rename.
                    entry['cohort_complementarity_violation_abs'] = (
                        node_complementarity_absolute(sed, esso_models, node_id))
                    entry['cohort_complementarity_violation_ratio'] = (
                        node_complementarity_ratio(esso_models, node_id))
                    entry['cohort_complementarity_violated'] = (
                        entry['cohort_complementarity_violation_abs'] > 0.0)
                report['nodes'][str(node_id)] = entry

            # Global form of the detector, called with the EXACT signature
            # specified by the Planner (`shared_ess_data.get_complementarity_
            # violation(models)`, over the full esso_models dict) -- this is
            # the function's native scope (all active nodes/cohorts/periods at
            # once), independent of which individual node's ESSO solve failed.
            report['complementarity_detector_global'] = {
                'absolute_violation': sed.get_complementarity_violation(esso_models),
                'note': ('max(0, min(pch, pdch) - 1e-6*s_max) over every active '
                         'cohort-period across ALL active nodes; 0.0 means the '
                         'detector found no counterexample anywhere, not a proof '
                         'complementarity holds everywhere (e.g. at s_max == 0).'),
            }
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
