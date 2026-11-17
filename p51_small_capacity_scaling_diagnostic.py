"""
Stage P5.1 -- small-capacity shared-ESS `sess_snet_def` scaling diagnostic.

Question: are the three P5 iteration-2 `positive_bootstrap` initialization
failures caused by extrapolating the accepted production row scaling
`kappa = 1/S_rated` far beyond its P4.3-P4.5 validated range?

Method (diagnostic only -- NO production change):

  * the bootstrap candidate is produced by the REAL production generator
    `_build_positive_bootstrap_candidate`, never reconstructed by hand;
  * each DSO initialization model is built by replaying the exact production
    pre-solve sequence used by
    `create_distribution_networks_models_sequential` -- same calls, same
    order -- stopping immediately before IPOPT;
  * that pre-solve state is preserved and cloned twice per case;
  * branch A leaves the production scale untouched (kappa_A = 1/S_rated);
  * branch B100 sets ONLY the existing mutable shared-ESS scale parameter
    `model.sess_snet_def_kappa[idx]` to min(1/S_rated, 100);
  * both branches solve through the production path
    `Network.run_smopf(..., from_warm_start=False)` (cold, as in the real
    initialization), with identical IPOPT/MA97/Hessian options.

No constraint component is created, deactivated, reindexed or re-expressed;
no variable, bound, objective, SOC equation, `sess_comp`, ADMM quantity or
solver option is touched.

    python p51_small_capacity_scaling_diagnostic.py
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import partial

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from model_construction_helpers import configure_shared_ess_operational_state  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P51')
KAPPA_CAP = 100.0

# (network name, DSO connection node, year, day)
FAILED_CASES = [
    ('case33_1', 5, 2030, 'Winter'),
    ('case33_1', 5, 2035, 'Winter'),
    ('case33_3', 9, 2025, 'Summer'),
]
CONTROL_CASES = [
    ('case33_1', 5, 2025, 'Winter'),
    ('case33_3', 9, 2030, 'Summer'),
    ('case33_2', 7, 2025, 'Winter'),
]

IPOPT_RE = {
    'iterations': re.compile(r'Number of Iterations\.*:\s*(\d+)'),
    'objective': re.compile(r'Objective\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'primal_infeasibility': re.compile(r'Constraint violation\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'dual_infeasibility': re.compile(r'Dual infeasibility\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'complementarity': re.compile(r'Complementarity\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'overall_nlp_error': re.compile(r'Overall NLP error\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'exit_status': re.compile(r'EXIT: (.+?)\.'),
}


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def parse_ipopt(text):
    out = {}
    for key, pattern in IPOPT_RE.items():
        match = None
        for match in pattern.finditer(text):
            pass
        if match:
            out[key] = match.groups() if len(match.groups()) > 1 else match.group(1)
    return out


def clear_logs(network, prefix):
    logs_dir = getattr(network, 'logs_dir', None)
    if logs_dir and os.path.isdir(logs_dir):
        for name in os.listdir(logs_dir):
            if name.startswith(prefix):
                try:
                    os.remove(os.path.join(logs_dir, name))
                except OSError:
                    pass


def read_logs(network, prefix):
    text = ''
    logs_dir = getattr(network, 'logs_dir', None)
    if logs_dir and os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir)):
            if name.startswith(prefix):
                with open(os.path.join(logs_dir, name), 'rb') as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 400_000))
                    text += handle.read().decode('utf-8', errors='replace')
    return text


# ---------------------------------------------------------------------------
#  Reproduce the production DSO initialization pre-solve state
# ---------------------------------------------------------------------------
def build_dso_initialization_models(distribution_network, total_capacity):
    """Replay `create_distribution_networks_models_sequential` exactly, up to
    (but not including) `distribution_network.optimize(...)`."""
    distribution_network.update_data_with_candidate_solution(total_capacity)
    dso_model = distribution_network.build_model()
    distribution_network.update_model_with_candidate_solution(dso_model, total_capacity)

    for year in distribution_network.years:
        for day in distribution_network.days:
            model = dso_model[year][day]
            network = distribution_network.network[year][day]
            ref_node_id = network.get_reference_node_id()
            shared_ess_idx = network.get_shared_energy_storage_idx(ref_node_id)

            model.expected_interface_vmag = pe.Var(model.periods, domain=pe.NonNegativeReals, initialize=1.00)
            model.expected_interface_pf_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
            model.expected_interface_pf_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
            model.expected_shared_ess_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
            model.expected_shared_ess_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
            model.expected_interface_vmag_def = pe.Constraint(
                model.periods, rule=partial(srp.dn_interface_expected_vmag_rule, network=network))
            model.expected_interface_pf_p_def = pe.Constraint(
                model.periods, rule=partial(srp.dn_interface_expected_pf_p_rule, network=network))
            model.expected_interface_pf_q_def = pe.Constraint(
                model.periods, rule=partial(srp.dn_interface_expected_pf_q_rule, network=network))
            model.expected_shared_ess_p_def = pe.Constraint(
                model.periods, rule=partial(srp.dn_interface_expected_sess_p_rule,
                                            network=network, shared_ess_idx=shared_ess_idx))
            model.expected_shared_ess_q_def = pe.Constraint(
                model.periods, rule=partial(srp.dn_interface_expected_sess_q_rule,
                                            network=network, shared_ess_idx=shared_ess_idx))
            configure_shared_ess_operational_state(
                model, shared_ess_idx,
                pe.value(model.shared_es_s_rated_fixed[shared_ess_idx]),
                pe.value(model.shared_es_e_rated_fixed[shared_ess_idx]))
            srp._add_dso_scenario_deviation_penalty(model, network)

    return dso_model


# ---------------------------------------------------------------------------
#  Structural / identity snapshots
# ---------------------------------------------------------------------------
def structural_signature(model, shared_ess_idx):
    rows = [tuple(i) for i in model.sess_snet_def if i[0] == shared_ess_idx]
    return {
        'sess_snet_def_component_id': id(model.sess_snet_def),
        'sess_snet_def_local_name': model.sess_snet_def.local_name,
        'sess_snet_def_rows_for_idx': len(rows),
        'sess_snet_def_index_tuples_for_idx': [list(r) for r in rows],
        'sess_snet_def_row_object_ids': {str(r): id(model.sess_snet_def[r]) for r in rows},
        'sess_snet_def_rows_active': [bool(model.sess_snet_def[r].active) for r in rows],
        'total_constraint_data': sum(
            len(list(c.values())) for c in model.component_objects(pe.Constraint, active=None)),
        'constraint_component_names': sorted(
            c.local_name for c in model.component_objects(pe.Constraint, active=None)),
        'has_replacement_component': any(
            c.local_name.startswith('sess_snet_def') and c.local_name != 'sess_snet_def'
            for c in model.component_objects(pe.Constraint, active=None)),
        'sess_comp_rows': len(list(model.sess_comp)) if hasattr(model, 'sess_comp') else 0,
        'ess_snet_def_rows': len(list(model.ess_snet_def)) if hasattr(model, 'ess_snet_def') else 0,
        'ordinary_scale_present': hasattr(model, 'ess_snet_def_scale'),
    }


def primal_start_signature(model):
    """Cheap fingerprint of every variable's starting value and bounds."""
    total = 0.0
    count = 0
    bounds_hash = 0.0
    for var in model.component_objects(pe.Var, active=True):
        for data in var.values():
            value = data.value
            if value is not None:
                total += float(value)
            lower = data.lb if data.lb is not None else 0.0
            upper = data.ub if data.ub is not None else 0.0
            bounds_hash += float(lower) * 1.000001 + float(upper) * 0.999999
            count += 1
    return {'n_var_data': count, 'sum_start_values': total, 'bounds_fingerprint': bounds_hash}


def shared_ess_state(model, idx):
    return {
        's_rated_fixed_pu': float(pe.value(model.shared_es_s_rated_fixed[idx])),
        'e_rated_fixed_pu': float(pe.value(model.shared_es_e_rated_fixed[idx])),
        'kappa': float(pe.value(model.sess_snet_def_kappa[idx])),
    }


def solve_branch(distribution_network, model, year, day, label, log_prefix):
    network = distribution_network.network[year][day]
    params = distribution_network.params
    clear_logs(network, log_prefix)
    buffer = io.StringIO()
    started = time.time()
    with redirect_stdout(buffer):
        result = network.run_smopf(model, params, from_warm_start=False, print_header=False)
    runtime = time.time() - started
    console = buffer.getvalue()
    log_text = read_logs(network, log_prefix)
    return {
        'branch': label,
        'status': str(getattr(result.solver, 'status', None)) if result is not None else None,
        'termination_condition': str(getattr(result.solver, 'termination_condition', None)) if result is not None else None,
        'message': str(getattr(result.solver, 'message', None)) if result is not None else None,
        'succeeded': bool(srp._solver_result_succeeded(result)) if result is not None else False,
        'runtime_s': runtime,
        'ipopt': parse_ipopt(log_text or console),
        'recovery_attempted': '[INFO] Retrying network solve once' in console,
    }, result


def shared_ess_solution_summary(model, idx, s_rated_pu):
    sch_v, sdch_v, pnet_v, qnet_v, g_v = [], [], [], [], []
    for index in model.sess_snet_def:
        if index[0] != idx:
            continue
        e, s_m, s_o, p = index
        sch = pe.value(model.shared_es_sch[e, s_m, s_o, p])
        sdch = pe.value(model.shared_es_sdch[e, s_m, s_o, p])
        pnet = pe.value(model.shared_es_pnet[e, s_m, s_o, p])
        qnet = pe.value(model.shared_es_qnet[e, s_m, s_o, p])
        if None in (sch, sdch, pnet, qnet):
            continue
        sch_v.append(float(sch)); sdch_v.append(float(sdch))
        pnet_v.append(float(pnet)); qnet_v.append(float(qnet))
        g_v.append((sch - sdch) ** 2 - pnet ** 2 - qnet ** 2)
    if not g_v:
        return {'available': False}
    worst = max(abs(v) for v in g_v)
    return {
        'available': True,
        'sch_range': [min(sch_v), max(sch_v)],
        'sdch_range': [min(sdch_v), max(sdch_v)],
        'pnet_range': [min(pnet_v), max(pnet_v)],
        'qnet_range': [min(qnet_v), max(qnet_v)],
        'max_abs_original_g': worst,
        'max_abs_original_g_normalized_by_s_rated_sq': worst / (s_rated_pu ** 2) if s_rated_pu else None,
    }


def run_case(distribution_network, dso_models, node_id, name, year, day, report_key):
    network = distribution_network.network[year][day]
    ref_node_id = network.get_reference_node_id()
    idx = network.get_shared_energy_storage_idx(ref_node_id)
    base_model = dso_models[year][day]

    s_rated_pu = float(pe.value(base_model.shared_es_s_rated_fixed[idx]))
    kappa_a = float(pe.value(base_model.sess_snet_def_kappa[idx]))
    kappa_b = min(1.0 / max(s_rated_pu, 0.01), KAPPA_CAP) if s_rated_pu else KAPPA_CAP
    kappa_b = min(kappa_a, KAPPA_CAP)

    model_a = base_model.clone()
    model_b = base_model.clone()

    pre_a = {'structure': structural_signature(model_a, idx),
             'primal': primal_start_signature(model_a),
             'shared_ess': shared_ess_state(model_a, idx),
             'objective_at_start': float(pe.value(model_a.objective))}
    # the ONLY change in branch B: the existing mutable shared-ESS scale param
    model_b.sess_snet_def_kappa[idx].set_value(kappa_b)
    pre_b = {'structure': structural_signature(model_b, idx),
             'primal': primal_start_signature(model_b),
             'shared_ess': shared_ess_state(model_b, idx),
             'objective_at_start': float(pe.value(model_b.objective))}

    identical = {
        'primal_start_identical': pre_a['primal'] == pre_b['primal'],
        'bounds_identical': abs(pre_a['primal']['bounds_fingerprint'] - pre_b['primal']['bounds_fingerprint']) < 1e-9,
        'objective_at_start_identical': abs(pre_a['objective_at_start'] - pre_b['objective_at_start']) < 1e-9,
        'constraint_component_names_identical':
            pre_a['structure']['constraint_component_names'] == pre_b['structure']['constraint_component_names'],
        'index_tuples_identical':
            pre_a['structure']['sess_snet_def_index_tuples_for_idx'] == pre_b['structure']['sess_snet_def_index_tuples_for_idx'],
        'total_constraint_data_identical':
            pre_a['structure']['total_constraint_data'] == pre_b['structure']['total_constraint_data'],
        'row_activity_identical':
            pre_a['structure']['sess_snet_def_rows_active'] == pre_b['structure']['sess_snet_def_rows_active'],
        'sess_comp_rows_identical': pre_a['structure']['sess_comp_rows'] == pre_b['structure']['sess_comp_rows'],
        'ordinary_scale_present_both': pre_a['structure']['ordinary_scale_present'] and pre_b['structure']['ordinary_scale_present'],
        'shared_ess_capacity_identical':
            pre_a['shared_ess']['s_rated_fixed_pu'] == pre_b['shared_ess']['s_rated_fixed_pu']
            and pre_a['shared_ess']['e_rated_fixed_pu'] == pre_b['shared_ess']['e_rated_fixed_pu'],
        'only_kappa_differs': pre_a['shared_ess']['kappa'] != pre_b['shared_ess']['kappa'],
        'no_replacement_component_either': not pre_a['structure']['has_replacement_component']
                                           and not pre_b['structure']['has_replacement_component'],
    }

    log_prefix = f'optim_log_{name}'
    solve_a, _ = solve_branch(distribution_network, model_a, year, day, 'A_production', log_prefix)
    solve_b, _ = solve_branch(distribution_network, model_b, year, day, 'B100_capped', log_prefix)

    solve_a['shared_ess_solution'] = shared_ess_solution_summary(model_a, idx, s_rated_pu)
    solve_b['shared_ess_solution'] = shared_ess_solution_summary(model_b, idx, s_rated_pu)

    post = {
        'structure_A_unchanged_after_solve': structural_signature(model_a, idx)['sess_snet_def_component_id']
                                             == pre_a['structure']['sess_snet_def_component_id'],
        'structure_B_unchanged_after_solve': structural_signature(model_b, idx)['sess_snet_def_component_id']
                                             == pre_b['structure']['sess_snet_def_component_id'],
        'B_row_object_ids_unchanged': structural_signature(model_b, idx)['sess_snet_def_row_object_ids']
                                      == pre_b['structure']['sess_snet_def_row_object_ids'],
    }

    return {
        'case': report_key,
        'network': name, 'node_id': node_id, 'year': year, 'day': day,
        'shared_ess_index': idx,
        's_rated_pu': s_rated_pu,
        's_rated_MVA': s_rated_pu * network.baseMVA,
        'kappa_A': kappa_a,
        'kappa_B100': kappa_b,
        'row_second_derivative_coefficient_A': 2.0 * kappa_a,
        'row_second_derivative_coefficient_B': 2.0 * kappa_b,
        'pre_solve_identity_checks': identical,
        'A': solve_a,
        'B100': solve_b,
        'structural_post_checks': post,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.1', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'kappa_cap': KAPPA_CAP}

    quiet = io.StringIO()
    with redirect_stdout(quiet):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        # REAL production bootstrap candidate -- not reconstructed by hand
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report['scenario_checksum'] = (re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]
    report['candidate'] = {
        'source': 'production _build_positive_bootstrap_candidate',
        'investment': {str(n): {str(y): dict(v) for y, v in years.items()}
                       for n, years in candidate['investment'].items()},
        'total_capacity': {str(n): {str(y): dict(v) for y, v in years.items()}
                           for n, years in candidate['total_capacity'].items()},
    }

    # Build every DSO initialization model once (production pre-solve sequence)
    dso_models_by_node = {}
    with redirect_stdout(quiet):
        for node_id in planning.distribution_networks:
            dso_models_by_node[node_id] = build_dso_initialization_models(
                planning.distribution_networks[node_id], candidate['total_capacity'])
    report['pre_solve_state_captured'] = True

    results = []
    for name, node_id, year, day in FAILED_CASES:
        results.append(run_case(planning.distribution_networks[node_id],
                                dso_models_by_node[node_id], node_id, name, year, day, 'failed'))
    for name, node_id, year, day in CONTROL_CASES:
        results.append(run_case(planning.distribution_networks[node_id],
                                dso_models_by_node[node_id], node_id, name, year, day, 'control'))
    report['cases'] = results

    failed = [r for r in results if r['case'] == 'failed']
    controls = [r for r in results if r['case'] == 'control']
    gate = {
        'all_failed_cases_fail_under_A': all(not r['A']['succeeded'] for r in failed),
        'all_failed_cases_succeed_under_B100': all(r['B100']['succeeded'] for r in failed),
        'all_controls_succeed_under_A': all(r['A']['succeeded'] for r in controls),
        'all_controls_still_succeed_under_B100': all(r['B100']['succeeded'] for r in controls),
    }
    gate['proceed_to_integration_check'] = (
        gate['all_failed_cases_succeed_under_B100'] and gate['all_controls_still_succeed_under_B100'])
    report['decision_gate'] = gate

    # ---- integration check (only if the gate opens) ----
    if gate['proceed_to_integration_check']:
        original_scale = mch.shared_ess_snet_def_scale

        def capped_scale(s_capacity):
            return min(original_scale(s_capacity), KAPPA_CAP)

        integration = {'cap_applied_in_memory_only': True}
        try:
            mch.shared_ess_snet_def_scale = capped_scale
            with redirect_stdout(quiet):
                planning2 = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
                planning2.read_planning_problem()
                candidate2 = srp._build_positive_bootstrap_candidate(
                    planning2, planning2.params.benders.positive_bootstrap)
                consensus_vars, _dual_vars = srp.create_admm_variables(planning2)
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
            integration['all_local_initialization_solves_succeeded'] = bool(all_ok)
            integration['would_enter_admm'] = bool(all_ok)
            per_case = {}
            for node_id, per_year in res['dso'].items():
                for year, per_day in per_year.items():
                    for day, result in per_day.items():
                        per_case[f'dso_{node_id}_{year}_{day}'] = str(
                            getattr(result.solver, 'termination_condition', None))
            integration['dso_terminations'] = per_case
        finally:
            mch.shared_ess_snet_def_scale = original_scale
        report['integration_check'] = integration
    else:
        report['integration_check'] = {'skipped': True,
                                       'reason': 'decision gate did not open'}

    out_path = os.path.join(OUT_DIR, 'p51_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'[P5.1] report -> {out_path}')
    print(f"[P5.1] checksum {report['scenario_checksum']}")
    for r in results:
        print(f"  [{r['case']:7s}] {r['network']} n{r['node_id']} {r['year']} {r['day']:6s} "
              f"S={r['s_rated_MVA']:.4f} MVA ({r['s_rated_pu']:.2e} pu) "
              f"kA={r['kappa_A']:.1f} kB={r['kappa_B100']:.1f} | "
              f"A: {r['A']['termination_condition']} it={r['A']['ipopt'].get('iterations')} | "
              f"B: {r['B100']['termination_condition']} it={r['B100']['ipopt'].get('iterations')}")
    print(f"[P5.1] gate: {json.dumps(report['decision_gate'])}")
    print(f"[P5.1] integration: {json.dumps(report['integration_check'])[:300]}")


if __name__ == '__main__':
    main()
