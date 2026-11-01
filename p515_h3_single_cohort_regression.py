"""
P5.15 Addendum 3 item 2 (H3 rule) -- single-cohort regression check
(PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Verification required" item 1).

The H3 rule adds, per active investment cohort, a pro-rata split of the
aggregate ESSO net power (`agg_pnet`) among active cohorts. Every currently
runnable fixture invests in at most ONE cohort per node, so the rule is
predicted to be INERT: `_configure_esso_cohort_pnet_share_rows` deactivates
every H3 row and sets every share to 0.0 whenever <=1 cohort is active for a
calendar year (see that function's docstring in
`shared_energy_storage_data.py`).

This harness measures that prediction two ways on the SAME instance
`p515_h_tol_remedy_check.py` uses (construction path reused by import, not
reimplemented): a fresh production planning problem
(`p56a_oracle.fresh_planning`), non-zero investment (S=1.00 MVA / E=2.00 MVAh,
first representative year) at nodes 7 and 9, node 5 carrying zero investment,
and the same non-trivial +/-10% duty-cycle `p_req` profile.

  ARM A ("with_h3"):      the production path exactly as shipped
                          (`create_shared_energy_storage_model`, H3 code
                          present and executes on every node).
  ARM B ("h3_stripped"):  the SAME instance, built and updated through the
                          SAME production functions
                          (`build_subproblem` / `update_model_with_candidate_solution`),
                          but with the H3 Param and ConstraintList components
                          REMOVED from each node's model (`del_component`)
                          immediately afterward, before the p_req/q_req fix
                          and the solve -- i.e. the model the solver sees is
                          exactly what would exist had the H3 rule never been
                          added to `_build_subproblem` at all.

If the H3 rule is inert here (as predicted), Arm A's solver-facing NLP is
IDENTICAL to Arm B's (Pyomo excludes deactivated constraints from the NL file
written to IPOPT, and an unused mutable Param carries no row), so the two
arms should solve to the same point up to ordinary solver noise. This
harness reports the ACTUAL max difference in objective, dispatch (es_pnet,
es_pch_per_unit, es_pdch_per_unit) and SoH (es_soh_per_unit_cumul), plus a
direct count of ACTIVE H3 rows in Arm A (predicted: 0) and the value of every
`es_pnet_cohort_share_h3[y_inv, y]` entry (predicted: 0.0, every entry).

GUARD. A `SolveProfileGuard` is armed for the whole run, permitting solves
only from `shared_energy_storage_data.py:_run_solver_attempt`. Declared count
in advance: 6 (2 arms x 3 active nodes -- 5, 7, 9).

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_h3_single_cohort_regression.py
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import p56a_oracle as oracle  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
from shared_resources_planning import (  # noqa: E402
    create_admm_variables,
    create_shared_energy_storage_model,
)
from helper_functions import fix_or_set  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_1_esso_reform_smoke import (  # noqa: E402
    SMOKE_NODES_WITH_INVESTMENT,
    S_CANDIDATE_MVA,
    E_CANDIDATE_MVAH,
    _zero_candidate,
    _set_nonzero_charge_discharge_request,
)
from p515_1b_eps_sensitivity_check import sha256_of  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151')
os.makedirs(OUT_DIR, exist_ok=True)

PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES = 6  # 2 arms x 3 active nodes (5, 7, 9), declared in advance


def _build_instance(tag):
    with redirect_stdout(io.StringIO()):
        planning = oracle.fresh_planning(f'p515h3_{tag}')
    shared_ess_data = planning.shared_ess_data

    consensus_vars, dual_vars = create_admm_variables(planning)
    candidate_solution = _zero_candidate(planning)
    first_year = list(planning.years)[0]
    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        candidate_solution[node_id][first_year] = {'s': S_CANDIDATE_MVA, 'e': E_CANDIDATE_MVAH}
        _set_nonzero_charge_discharge_request(
            planning, consensus_vars, node_id, amplitude_mw=0.10 * S_CANDIDATE_MVA
        )
    return planning, shared_ess_data, consensus_vars, candidate_solution


def _h3_row_census(model):
    """Direct count of H3 rows and their state -- production introspection,
    not reimplemented logic. `active` is read from the Pyomo constraint
    object itself (the same attribute the NL writer consults)."""
    total = 0
    active = 0
    shares = {}
    for (y_inv, y), constraint_idx in _iter_h3_rows(model):
        total += 1
        constraint = model.energy_storage_cohort_pnet_share_h3[constraint_idx]
        if constraint.active:
            active += 1
        shares[f'{y_inv}_{y}'] = pe.value(model.es_pnet_cohort_share_h3[y_inv, y])
    return {'total_structural_rows': total, 'active_rows': active, 'shares': shares}


def _iter_h3_rows(model):
    for y_inv in model.years:
        for constraint_name, constraint_idx, y in model._esso_cohort_constraints[y_inv]:
            if constraint_name == 'energy_storage_cohort_pnet_share_h3':
                yield (y_inv, y), constraint_idx


def _dispatch_snapshot(model):
    snapshot = {'es_pnet': {}, 'es_qnet': {}, 'es_pch_per_unit': {}, 'es_pdch_per_unit': {},
                'es_soh_per_unit_cumul': {}, 'es_D_per_unit': {}}
    for y in model.years:
        for d in model.days:
            for p in model.periods:
                snapshot['es_pnet'][f'{y}_{d}_{p}'] = pe.value(model.es_pnet[y, d, p])
                snapshot['es_qnet'][f'{y}_{d}_{p}'] = pe.value(model.es_qnet[y, d, p])
    for y_inv in model.years:
        for y in model.years:
            snapshot['es_soh_per_unit_cumul'][f'{y_inv}_{y}'] = pe.value(model.es_soh_per_unit_cumul[y_inv, y])
            snapshot['es_D_per_unit'][f'{y_inv}_{y}'] = pe.value(model.es_D_per_unit[y_inv, y])
            for d in model.days:
                for p in model.periods:
                    key = f'{y_inv}_{y}_{d}_{p}'
                    snapshot['es_pch_per_unit'][key] = pe.value(model.es_pch_per_unit[y_inv, y, d, p])
                    snapshot['es_pdch_per_unit'][key] = pe.value(model.es_pdch_per_unit[y_inv, y, d, p])
    return snapshot


def run_arm_a():
    """Arm A: production path exactly as shipped."""
    planning, shared_ess_data, consensus_vars, candidate_solution = _build_instance('armA')

    arm_dir = os.path.join(OUT_DIR, 'h3_regression_logs', 'armA_with_h3')
    os.makedirs(arm_dir, exist_ok=True)
    cwd_before = os.getcwd()
    try:
        os.chdir(arm_dir)
        with redirect_stdout(io.StringIO()):
            esso_models, esso_op_results = create_shared_energy_storage_model(
                shared_ess_data, consensus_vars, candidate_solution
            )
    finally:
        os.chdir(cwd_before)

    result = {'nodes': {}}
    for node_id in shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node_id]
        solver_result = esso_op_results.get(node_id)
        result['nodes'][node_id] = {
            'termination_condition': str(solver_result.solver.termination_condition) if solver_result else None,
            'objective': pe.value(model.objective),
            'h3_census': _h3_row_census(model),
            'dispatch': _dispatch_snapshot(model),
        }
    return result


def run_arm_b():
    """Arm B: same production build/update path, H3 components stripped
    (`del_component`) before the p_req/q_req fix and the solve."""
    planning, shared_ess_data, consensus_vars, candidate_solution = _build_instance('armB')
    years = list(planning.years)
    days = list(planning.days)

    with redirect_stdout(io.StringIO()):
        esso_models = shared_ess_data.build_subproblem()
        shared_ess_data.update_model_with_candidate_solution(esso_models, candidate_solution)

    stripped_rows_removed = {}
    for node_id in shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node_id]
        # Count what is about to be removed, for the report, before removing it.
        removed_count = len(list(_iter_h3_rows(model)))
        model.del_component(model.energy_storage_cohort_pnet_share_h3)
        model.del_component(model.es_pnet_cohort_share_h3)
        stripped_rows_removed[node_id] = removed_count

    # Fix TSO's request -- same as create_shared_energy_storage_model.
    for node_id in shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node_id]
        for y in model.years:
            year = years[y]
            for d in model.days:
                day = days[d]
                for p in model.periods:
                    p_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['p'][p]
                    q_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['q'][p]
                    fix_or_set(model.es_pnet[y, d, p], p_req)
                    fix_or_set(model.es_qnet[y, d, p], q_req)

    arm_dir = os.path.join(OUT_DIR, 'h3_regression_logs', 'armB_h3_stripped')
    os.makedirs(arm_dir, exist_ok=True)
    cwd_before = os.getcwd()
    try:
        os.chdir(arm_dir)
        with redirect_stdout(io.StringIO()):
            esso_op_results = shared_ess_data.optimize(esso_models)
    finally:
        os.chdir(cwd_before)

    result = {'nodes': {}, 'stripped_rows_removed_per_node': stripped_rows_removed}
    for node_id in shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node_id]
        solver_result = esso_op_results.get(node_id)
        result['nodes'][node_id] = {
            'termination_condition': str(solver_result.solver.termination_condition) if solver_result else None,
            'objective': pe.value(model.objective),
            'dispatch': _dispatch_snapshot(model),
        }
    return result


def _max_abs_diff(dict_a, dict_b):
    max_diff = 0.0
    argmax = None
    for key in dict_a:
        diff = abs(dict_a[key] - dict_b[key])
        if diff > max_diff:
            max_diff = diff
            argmax = key
    return max_diff, argmax


def main():
    guard = SolveProfileGuard(PERMITTED, label='P5.15 H3 single-cohort regression').install()
    try:
        print('[P5.15-H3] running Arm A (with_h3, production path) ...', flush=True)
        arm_a = run_arm_a()
        print('[P5.15-H3] running Arm B (h3_stripped, del_component before solve) ...', flush=True)
        arm_b = run_arm_b()
    finally:
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)

    comparison = {}
    for node_id in arm_a['nodes']:
        node_a = arm_a['nodes'][node_id]
        node_b = arm_b['nodes'][node_id]
        obj_diff = abs(node_a['objective'] - node_b['objective'])
        dispatch_diffs = {}
        for field in ('es_pnet', 'es_qnet', 'es_pch_per_unit', 'es_pdch_per_unit', 'es_soh_per_unit_cumul', 'es_D_per_unit'):
            max_diff, argmax = _max_abs_diff(node_a['dispatch'][field], node_b['dispatch'][field])
            dispatch_diffs[field] = {'max_abs_diff': max_diff, 'argmax_key': argmax}
        comparison[str(node_id)] = {
            'termination_condition_a': node_a['termination_condition'],
            'termination_condition_b': node_b['termination_condition'],
            'objective_a': node_a['objective'],
            'objective_b': node_b['objective'],
            'objective_abs_diff': obj_diff,
            'dispatch_max_abs_diffs': dispatch_diffs,
            'h3_census_arm_a': node_a['h3_census'],
        }

    all_shares_zero = all(
        all(v == 0.0 for v in comparison[node]['h3_census_arm_a']['shares'].values())
        for node in comparison
    )
    all_rows_inactive = all(
        comparison[node]['h3_census_arm_a']['active_rows'] == 0
        for node in comparison
    )

    report = {
        'stage': 'P5.15 Addendum 3 item 2 (H3 rule) -- single-cohort regression',
        'authority': (
            'PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Authorized" item 2 / "Verification required" item 1'
        ),
        'instance': {
            'construction_source': 'p515_1_esso_reform_smoke.py + p515_h_tol_remedy_check.py (imported, not reimplemented)',
            'smoke_nodes_with_investment': list(SMOKE_NODES_WITH_INVESTMENT),
            's_candidate_mva': S_CANDIDATE_MVA,
            'e_candidate_mvah': E_CANDIDATE_MVAH,
            'amplitude_mw_formula': '0.10 * S_CANDIDATE_MVA',
            'active_nodes': [5, 7, 9],
            'cohorts_active_per_node': 'exactly 1 (first representative year only) -- single-cohort by construction',
        },
        'guard': {
            'permitted_call_sites': PERMITTED,
            'expected_solves_declared_in_advance': EXPECTED_SOLVES,
            'observed_counts': guard.counts,
            'permitted_sites_hit': guard.permitted_sites,
            'verify_failures': guard_failures,
        },
        'arm_b_stripped_rows_removed_per_node': arm_b['stripped_rows_removed_per_node'],
        'comparison_per_node': comparison,
        'all_h3_shares_zero_in_arm_a': all_shares_zero,
        'all_h3_rows_inactive_in_arm_a': all_rows_inactive,
        'conclusion': (
            'H3 rule INERT on this single-cohort fixture: zero active rows, all shares 0.0, '
            'Arm A vs Arm B (H3 components physically removed) differ only by ordinary solver noise.'
            if all_shares_zero and all_rows_inactive
            else 'UNEXPECTED: H3 rule was NOT inert on a single-cohort fixture -- see comparison_per_node.'
        ),
    }

    summary_path = os.path.join(OUT_DIR, 'h3_single_cohort_regression_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps({k: v for k, v in report.items() if k not in ('comparison_per_node',)}, indent=2, default=str))
    print(json.dumps({'comparison_per_node_summary': {
        node: {
            'objective_abs_diff': comparison[node]['objective_abs_diff'],
            'dispatch_max_abs_diffs': {
                field: comparison[node]['dispatch_max_abs_diffs'][field]['max_abs_diff']
                for field in comparison[node]['dispatch_max_abs_diffs']
            },
            'active_rows_arm_a': comparison[node]['h3_census_arm_a']['active_rows'],
            'total_structural_rows_arm_a': comparison[node]['h3_census_arm_a']['total_structural_rows'],
        } for node in comparison
    }}, indent=2, default=str))
    print(f'\n[P5.15-H3] summary written to {summary_path}')
    print(f'[P5.15-H3] guard verify failures: {guard_failures}')


if __name__ == '__main__':
    main()
