"""
Stage P5.1-B -- shared-ESS `sess_snet_def` scaling-window diagnostic.

Question: is there a common intermediate upper bound Kmax on
`sess_snet_def_kappa` that solves BOTH

  * the three P5 bootstrap failures that need substantially less scaling than
    production (kappa_A ~ 3.1e3 - 9.4e3), and
  * the `case33_3 / node 9 / 2025 / Autumn` case that regresses when scaling is
    reduced all the way to 100?

Ladder, per case, on fresh identical clones of the same production pre-solve
state:

    kappa = min(1/S_rated, Kmax)   for Kmax in {100, 300, 1000, 3000}
    kappa = 1/S_rated              (untouched production baseline)

The ONLY quantity that differs across branches is
`model.sess_snet_def_kappa[idx]`. Everything else -- primal starting point,
bounds, objective, candidate capacity, constraint components, indices,
activation state, suffixes, IPOPT options, MA97, exact-Hessian primary path,
`from_warm_start=False` -- is identical and verified so before each solve.

Diagnostic only. No production code is modified; the bootstrap candidate and
the pre-solve initialization state come from the real production path, reused
from `p51_small_capacity_scaling_diagnostic`.

    python p51b_scaling_window_diagnostic.py
"""

import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

# reuse the accepted P5.1 machinery rather than duplicating it
from p51_small_capacity_scaling_diagnostic import (  # noqa: E402
    build_dso_initialization_models,
    primal_start_signature,
    shared_ess_solution_summary,
    shared_ess_state,
    solve_branch,
    structural_signature,
)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P51B')

# (label, network name, DSO connection node, year, day, group)
CASES = [
    ('n5_2030_Winter', 'case33_1', 5, 2030, 'Winter', 'production_failing'),
    ('n5_2035_Winter', 'case33_1', 5, 2035, 'Winter', 'production_failing'),
    ('n9_2025_Summer', 'case33_3', 9, 2025, 'Summer', 'production_failing'),
    ('n9_2025_Autumn', 'case33_3', 9, 2025, 'Autumn', 'b100_regression'),
]

# None == untouched production 1/S_rated
LADDER = [100.0, 300.0, 1000.0, 3000.0, None]


def ladder_label(kmax):
    return 'production' if kmax is None else str(int(kmax))


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def run_case_ladder(distribution_network, dso_models, label, name, node_id, year, day, group):
    network = distribution_network.network[year][day]
    ref_node_id = network.get_reference_node_id()
    idx = network.get_shared_energy_storage_idx(ref_node_id)
    base_model = dso_models[year][day]

    s_rated_pu = float(pe.value(base_model.shared_es_s_rated_fixed[idx]))
    kappa_production = float(pe.value(base_model.sess_snet_def_kappa[idx]))

    reference = {
        'structure': structural_signature(base_model, idx),
        'primal': primal_start_signature(base_model),
        'shared_ess': shared_ess_state(base_model, idx),
        'objective_at_start': float(pe.value(base_model.objective)),
    }

    branches = {}
    for kmax in LADDER:
        key = ladder_label(kmax)
        model = base_model.clone()
        if kmax is None:
            kappa = kappa_production          # untouched production baseline
        else:
            kappa = min(kappa_production, kmax)
            model.sess_snet_def_kappa[idx].set_value(kappa)

        pre = {
            'structure': structural_signature(model, idx),
            'primal': primal_start_signature(model),
            'shared_ess': shared_ess_state(model, idx),
            'objective_at_start': float(pe.value(model.objective)),
        }
        identity = {
            'primal_start_identical': pre['primal'] == reference['primal'],
            'bounds_identical': abs(pre['primal']['bounds_fingerprint']
                                    - reference['primal']['bounds_fingerprint']) < 1e-9,
            'objective_at_start_identical': abs(pre['objective_at_start']
                                                - reference['objective_at_start']) < 1e-9,
            'component_names_identical': (pre['structure']['constraint_component_names']
                                          == reference['structure']['constraint_component_names']),
            'index_tuples_identical': (pre['structure']['sess_snet_def_index_tuples_for_idx']
                                       == reference['structure']['sess_snet_def_index_tuples_for_idx']),
            'total_constraint_data_identical': (pre['structure']['total_constraint_data']
                                                == reference['structure']['total_constraint_data']),
            'row_activity_identical': (pre['structure']['sess_snet_def_rows_active']
                                       == reference['structure']['sess_snet_def_rows_active']),
            'sess_comp_rows_identical': (pre['structure']['sess_comp_rows']
                                         == reference['structure']['sess_comp_rows']),
            'capacity_identical': (pre['shared_ess']['s_rated_fixed_pu']
                                   == reference['shared_ess']['s_rated_fixed_pu']
                                   and pre['shared_ess']['e_rated_fixed_pu']
                                   == reference['shared_ess']['e_rated_fixed_pu']),
            'no_replacement_component': not pre['structure']['has_replacement_component'],
        }

        solve, _result = solve_branch(distribution_network, model, year, day,
                                      key, f'optim_log_{name}')
        solve['shared_ess_solution'] = shared_ess_solution_summary(model, idx, s_rated_pu)
        solve['kappa'] = kappa
        solve['row_second_derivative_coefficient'] = 2.0 * kappa
        solve['kappa_is_capped'] = (kmax is not None and kappa_production > kmax)
        solve['pre_solve_identity_checks'] = identity
        solve['structure_unchanged_after_solve'] = (
            structural_signature(model, idx)['sess_snet_def_row_object_ids']
            == pre['structure']['sess_snet_def_row_object_ids'])
        branches[key] = solve

    return {
        'label': label, 'group': group, 'network': name, 'node_id': node_id,
        'year': year, 'day': day, 'shared_ess_index': idx,
        's_rated_pu': s_rated_pu,
        's_rated_MVA': s_rated_pu * network.baseMVA,
        'kappa_production': kappa_production,
        'branches': branches,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.1-B', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'ladder': [ladder_label(k) for k in LADDER]}

    quiet = io.StringIO()
    with redirect_stdout(quiet):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]
    report['candidate_total_capacity'] = {
        str(n): {str(y): dict(v) for y, v in years.items()}
        for n, years in candidate['total_capacity'].items()}

    needed_nodes = sorted({c[2] for c in CASES})
    dso_models_by_node = {}
    with redirect_stdout(quiet):
        for node_id in needed_nodes:
            dso_models_by_node[node_id] = build_dso_initialization_models(
                planning.distribution_networks[node_id], candidate['total_capacity'])

    results = []
    for label, name, node_id, year, day, group in CASES:
        print(f'[P5.1-B] running ladder for {label} ...', flush=True)
        results.append(run_case_ladder(planning.distribution_networks[node_id],
                                       dso_models_by_node[node_id],
                                       label, name, node_id, year, day, group))
    report['cases'] = results

    # ---- success matrix + common-window decision ----
    matrix = {}
    for case in results:
        matrix[case['label']] = {
            key: {'succeeded': branch['succeeded'],
                  'termination': branch['termination_condition'],
                  'iterations': branch['ipopt'].get('iterations'),
                  'kappa': branch['kappa']}
            for key, branch in case['branches'].items()}
    report['success_matrix'] = matrix

    common = [ladder_label(k) for k in LADDER
              if all(matrix[c['label']][ladder_label(k)]['succeeded'] for c in results)]
    report['common_successful_kmax'] = common
    # "largest common-successful Kmax": production is the largest possible scale
    order = {ladder_label(k): (float('inf') if k is None else k) for k in LADDER}
    selected = max(common, key=lambda k: order[k]) if common else None
    report['selected_kmax'] = selected

    # ---- full initialization gate ----
    if selected is None or selected == 'production':
        report['full_initialization_gate'] = {
            'run': False,
            'reason': ('no tested Kmax succeeds for all four decisive cases'
                       if selected is None else
                       'the only common-successful branch is untouched production, '
                       'which already fails the three P5 cases in the full run')}
    else:
        cap_value = float(selected)
        original_scale = mch.shared_ess_snet_def_scale

        def capped_scale(s_capacity):
            return min(original_scale(s_capacity), cap_value)

        gate = {'run': True, 'cap': cap_value, 'cap_applied_in_memory_only': True}
        try:
            mch.shared_ess_snet_def_scale = capped_scale
            with redirect_stdout(quiet):
                planning2 = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
                planning2.read_planning_problem()
                candidate2 = srp._build_positive_bootstrap_candidate(
                    planning2, planning2.params.benders.positive_bootstrap)
                consensus_vars, _dual = srp.create_admm_variables(planning2)
                res = {'tso': dict(), 'dso': dict(), 'esso': dict()}
                _dso_models, res['dso'] = srp.create_distribution_networks_models(
                    planning2.distribution_networks, consensus_vars,
                    candidate2['total_capacity'],
                    parallel_execution=planning2.parallel_execution)
                _tso_model, res['tso'] = srp.create_transmission_network_model(
                    planning2, consensus_vars, candidate2['total_capacity'])
                _esso_model, res['esso'] = srp.create_shared_energy_storage_model(
                    planning2.shared_ess_data, consensus_vars, candidate2['investment'])
                all_ok = srp._admm_local_solves_succeeded(planning2, res)

            def walk(container, prefix):
                out = {}
                if isinstance(container, dict):
                    for key, value in container.items():
                        out.update(walk(value, f'{prefix}_{key}'))
                else:
                    out[prefix] = {
                        'termination': str(getattr(container.solver, 'termination_condition', None)),
                        'status': str(getattr(container.solver, 'status', None)),
                        'succeeded': bool(srp._solver_result_succeeded(container)),
                    }
                return out

            solves = {}
            for agent in ('dso', 'tso', 'esso'):
                try:
                    solves.update(walk(res[agent], agent))
                except Exception as error:
                    solves[f'{agent}_unparsed'] = str(error)
            failures = {k: v for k, v in solves.items()
                        if isinstance(v, dict) and not v.get('succeeded')}
            gate.update({
                'total_local_solves': len(solves),
                'n_failures': len(failures),
                'failures': failures,
                'all_solves': solves,
                'admm_local_solves_succeeded': bool(all_ok),
                'would_enter_admm': bool(all_ok),
            })
        finally:
            mch.shared_ess_snet_def_scale = original_scale
        report['full_initialization_gate'] = gate

    out_path = os.path.join(OUT_DIR, 'p51b_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[P5.1-B] report -> {out_path}')
    print(f"[P5.1-B] checksum {report['scenario_checksum']}")
    header = f"{'case':18s} " + " ".join(f"{ladder_label(k):>12s}" for k in LADDER)
    print(header)
    for case in results:
        row = f"{case['label']:18s} "
        for k in LADDER:
            cell = matrix[case['label']][ladder_label(k)]
            mark = 'OK' if cell['succeeded'] else 'FAIL'
            row += f"{mark + '/' + str(cell['iterations']):>12s} "
        print(row)
    print(f"[P5.1-B] common successful Kmax: {common or 'NONE'} | selected: {selected}")
    print(f"[P5.1-B] gate: {json.dumps(report['full_initialization_gate'])[:400]}")


if __name__ == '__main__':
    main()
