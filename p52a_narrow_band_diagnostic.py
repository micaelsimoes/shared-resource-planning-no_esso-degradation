"""
Stage P5.2-A -- narrow two-sided band relaxation of shared-ESS `sess_snet_def`.

Hypothesis: with

    g = (sch - sdch)^2 - pnet^2 - qnet^2

grad(g) = 0 exactly at sch = sdch = pnet = qnet = 0, and several failed
cold-start states sit exactly there after thousands of IPOPT iterations. Row
scaling cannot remove that degeneracy, because kappa * grad(g) = 0 for every
finite kappa. This stage tests whether replacing the hard equality with a very
narrow TWO-SIDED band makes the zero-dispatch start strictly interior rather
than equality-active.

    A (production)  :            kappa * g == 0
    B (narrow band) : -kappa*eps*S^2 <= kappa * g <= +kappa*eps*S^2

with eps_rel = 1e-5 and the UNTOUCHED production kappa = 1/S_rated in both
branches (no cap). Branch B is applied in place on the existing
`sess_snet_def` ConstraintData objects via `set_value(pe.inequality(...))`, so
the component, row objects and indices are preserved.

Diagnostic only. No production file is modified.

    python p52a_narrow_band_diagnostic.py
"""

import io
import json
import math
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

from p51_small_capacity_scaling_diagnostic import (  # noqa: E402  (reuse)
    build_dso_initialization_models,
    primal_start_signature,
    shared_ess_state,
    solve_branch,
    structural_signature,
)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P52A')
EPSILON_REL = 1e-5

# (label, network, node, year, day, group)
SENSITIVE_CASES = [
    ('n5_2030_Winter', 'case33_1', 5, 2030, 'Winter', 'P5_original_failure'),
    ('n5_2035_Winter', 'case33_1', 5, 2035, 'Winter', 'P5_original_failure'),
    ('n9_2025_Summer', 'case33_3', 9, 2025, 'Summer', 'P5_original_failure'),
    ('n9_2025_Autumn', 'case33_3', 9, 2025, 'Autumn', 'P5.1_cap100_regression'),
    ('n5_2030_Spring', 'case33_1', 5, 2030, 'Spring', 'P5.1B_cap1000_regression'),
    ('n5_2035_Autumn', 'case33_1', 5, 2035, 'Autumn', 'P5.1B_cap1000_regression'),
    ('n7_2030_Summer', 'case33_2', 7, 2030, 'Summer', 'P5.1B_cap1000_regression'),
    ('n7_2035_Spring', 'case33_2', 7, 2035, 'Spring', 'P5.1B_cap1000_regression'),
]
CONTROL_CASES = [
    ('n5_2025_Winter', 'case33_1', 5, 2025, 'Winter', 'control'),
    ('n7_2025_Winter', 'case33_2', 7, 2025, 'Winter', 'control'),
    ('n9_2030_Summer', 'case33_3', 9, 2030, 'Summer', 'control'),
]


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def apply_narrow_band(model, idx, epsilon_rel, s_rated_pu):
    """Convert the EXISTING sess_snet_def rows for `idx` from `kappa*g == 0` to
    a two-sided band, in place on the same ConstraintData objects."""
    kappa = float(pe.value(model.sess_snet_def_kappa[idx]))
    epsilon_abs = epsilon_rel * (s_rated_pu ** 2)          # physical band on g
    bound = kappa * epsilon_abs                            # band on the scaled row
    converted = 0
    for index in model.sess_snet_def:
        if index[0] != idx:
            continue
        con = model.sess_snet_def[index]
        con.set_value(pe.inequality(-bound, con.body, bound))
        converted += 1
    return {'kappa': kappa, 'epsilon_rel': epsilon_rel, 'epsilon_abs_pu2': epsilon_abs,
            'scaled_lower_bound': -bound, 'scaled_upper_bound': bound,
            'rows_converted': converted}


def constraint_form_summary(model, idx):
    forms = {'equality': 0, 'ranged': 0, 'other': 0}
    sample = None
    for index in model.sess_snet_def:
        if index[0] != idx:
            continue
        con = model.sess_snet_def[index]
        if con.equality:
            forms['equality'] += 1
        elif con.lower is not None and con.upper is not None:
            forms['ranged'] += 1
        else:
            forms['other'] += 1
        if sample is None:
            sample = {'lower': str(con.lower), 'upper': str(con.upper),
                      'body': str(con.body)[:160]}
    return {'row_forms': forms, 'sample_row': sample}


def physical_metrics(model, idx, s_rated_pu, base_mva, epsilon_abs):
    """Original UNSCALED g, apparent-power mismatch, and band activity."""
    rows = []
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
        g = (sch - sdch) ** 2 - pnet ** 2 - qnet ** 2
        delta_s = abs(abs(sch - sdch) - math.sqrt(pnet ** 2 + qnet ** 2))
        ratio = abs(g) / epsilon_abs if epsilon_abs else float('inf')
        rows.append({'period': p, 'g': g, 'delta_s': delta_s, 'band_ratio': ratio,
                     'sch': sch, 'sdch': sdch, 'pnet': pnet, 'qnet': qnet})
    if not rows:
        return {'available': False}

    max_abs_g = max(abs(r['g']) for r in rows)
    max_delta = max(r['delta_s'] for r in rows)
    near_boundary = [r for r in rows if r['band_ratio'] >= 0.9]
    active = [r for r in rows if r['band_ratio'] >= 1.0 - 1e-9]
    well_inside = [r for r in rows if r['band_ratio'] < 0.9]
    return {
        'available': True,
        'n_periods': len(rows),
        'sch_range': [min(r['sch'] for r in rows), max(r['sch'] for r in rows)],
        'sdch_range': [min(r['sdch'] for r in rows), max(r['sdch'] for r in rows)],
        'pnet_range': [min(r['pnet'] for r in rows), max(r['pnet'] for r in rows)],
        'qnet_range': [min(r['qnet'] for r in rows), max(r['qnet'] for r in rows)],
        'max_abs_g': max_abs_g,
        'max_abs_g_normalized_by_s_rated_sq': max_abs_g / (s_rated_pu ** 2),
        'max_delta_S_pu': max_delta,
        'max_delta_S_MVA': max_delta * base_mva,
        'max_delta_S_over_s_rated': max_delta / s_rated_pu if s_rated_pu else None,
        'max_band_ratio': max(r['band_ratio'] for r in rows),
        'n_periods_well_inside_band': len(well_inside),
        'n_periods_within_10pct_of_boundary': len(near_boundary),
        'n_periods_active_at_boundary': len(active),
        'band_classification': ('exactly active' if active else
                                'near a bound' if near_boundary else
                                'well inside the band'),
    }


def run_case(distribution_network, dso_models, label, name, node_id, year, day, group):
    network = distribution_network.network[year][day]
    ref_node_id = network.get_reference_node_id()
    idx = network.get_shared_energy_storage_idx(ref_node_id)
    base_model = dso_models[year][day]
    base_mva = network.baseMVA

    s_rated_pu = float(pe.value(base_model.shared_es_s_rated_fixed[idx]))
    kappa = float(pe.value(base_model.sess_snet_def_kappa[idx]))
    epsilon_abs = EPSILON_REL * (s_rated_pu ** 2)

    model_a = base_model.clone()
    model_b = base_model.clone()

    pre_a = {'structure': structural_signature(model_a, idx),
             'primal': primal_start_signature(model_a),
             'shared_ess': shared_ess_state(model_a, idx),
             'objective_at_start': float(pe.value(model_a.objective)),
             'forms': constraint_form_summary(model_a, idx)}
    row_ids_b_before = structural_signature(model_b, idx)['sess_snet_def_row_object_ids']
    band = apply_narrow_band(model_b, idx, EPSILON_REL, s_rated_pu)
    pre_b = {'structure': structural_signature(model_b, idx),
             'primal': primal_start_signature(model_b),
             'shared_ess': shared_ess_state(model_b, idx),
             'objective_at_start': float(pe.value(model_b.objective)),
             'forms': constraint_form_summary(model_b, idx)}

    discipline = {
        'component_object_unchanged': (pre_b['structure']['sess_snet_def_component_id']
                                       == structural_signature(model_b, idx)['sess_snet_def_component_id']),
        'row_object_ids_unchanged': pre_b['structure']['sess_snet_def_row_object_ids'] == row_ids_b_before,
        'index_tuples_unchanged': (pre_a['structure']['sess_snet_def_index_tuples_for_idx']
                                   == pre_b['structure']['sess_snet_def_index_tuples_for_idx']),
        'total_constraint_data_unchanged': (pre_a['structure']['total_constraint_data']
                                            == pre_b['structure']['total_constraint_data']),
        'no_replacement_component': not pre_b['structure']['has_replacement_component'],
        'component_names_identical': (pre_a['structure']['constraint_component_names']
                                      == pre_b['structure']['constraint_component_names']),
        'primal_start_identical': pre_a['primal'] == pre_b['primal'],
        'bounds_fingerprint_identical': abs(pre_a['primal']['bounds_fingerprint']
                                            - pre_b['primal']['bounds_fingerprint']) < 1e-9,
        'objective_at_start_identical': abs(pre_a['objective_at_start']
                                            - pre_b['objective_at_start']) < 1e-9,
        'sess_comp_rows_identical': (pre_a['structure']['sess_comp_rows']
                                     == pre_b['structure']['sess_comp_rows']),
        'ordinary_ess_rows_identical': (pre_a['structure']['ess_snet_def_rows']
                                        == pre_b['structure']['ess_snet_def_rows']),
        'kappa_identical': pre_a['shared_ess']['kappa'] == pre_b['shared_ess']['kappa'],
        'capacity_identical': (pre_a['shared_ess']['s_rated_fixed_pu']
                               == pre_b['shared_ess']['s_rated_fixed_pu']),
        'A_rows_all_equality': pre_a['forms']['row_forms']['equality'] > 0
                               and pre_a['forms']['row_forms']['ranged'] == 0,
        'B_rows_all_ranged': pre_b['forms']['row_forms']['ranged'] > 0
                             and pre_b['forms']['row_forms']['equality'] == 0,
        'row_activity_identical': (pre_a['structure']['sess_snet_def_rows_active']
                                   == pre_b['structure']['sess_snet_def_rows_active']),
    }

    log_prefix = f'optim_log_{name}'
    solve_a, _ = solve_branch(distribution_network, model_a, year, day, 'A_equality', log_prefix)
    solve_b, _ = solve_branch(distribution_network, model_b, year, day, 'B_narrow_band', log_prefix)
    solve_a['physical'] = physical_metrics(model_a, idx, s_rated_pu, base_mva, epsilon_abs)
    solve_b['physical'] = physical_metrics(model_b, idx, s_rated_pu, base_mva, epsilon_abs)

    return {
        'label': label, 'group': group, 'network': name, 'node_id': node_id,
        'year': year, 'day': day, 'shared_ess_index': idx,
        's_rated_pu': s_rated_pu, 's_rated_MVA': s_rated_pu * base_mva,
        'kappa': kappa, 'band': band,
        'discipline_checks': discipline,
        'A': solve_a, 'B': solve_b,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.2-A', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'epsilon_rel': EPSILON_REL}

    quiet = io.StringIO()
    with redirect_stdout(quiet):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]

    dso_models_by_node = {}
    with redirect_stdout(quiet):
        for node_id in sorted(planning.distribution_networks):
            dso_models_by_node[node_id] = build_dso_initialization_models(
                planning.distribution_networks[node_id], candidate['total_capacity'])

    results = []
    for label, name, node_id, year, day, group in SENSITIVE_CASES + CONTROL_CASES:
        print(f'[P5.2-A] {label} ...', flush=True)
        results.append(run_case(planning.distribution_networks[node_id],
                                dso_models_by_node[node_id],
                                label, name, node_id, year, day, group))
    report['cases'] = results

    sensitive = [r for r in results if r['group'] != 'control']
    controls = [r for r in results if r['group'] == 'control']
    gate = {
        'all_sensitive_B_succeed': all(r['B']['succeeded'] for r in sensitive),
        'all_sensitive_B_primary_no_recovery': all(
            r['B']['succeeded'] and not r['B']['recovery_attempted'] for r in sensitive),
        'all_controls_A_succeed': all(r['A']['succeeded'] for r in controls),
        'all_controls_B_succeed': all(r['B']['succeeded'] for r in controls),
        'max_band_ratio_over_all_successful_B': max(
            [r['B']['physical'].get('max_band_ratio', 0.0)
             for r in results if r['B']['succeeded'] and r['B']['physical'].get('available')] or [0.0]),
        'any_B_period_active_at_boundary': any(
            r['B']['physical'].get('n_periods_active_at_boundary', 0) > 0
            for r in results if r['B']['succeeded'] and r['B']['physical'].get('available')),
    }
    gate['proceed_to_full_initialization'] = (
        gate['all_sensitive_B_succeed'] and gate['all_controls_B_succeed'])
    report['gate'] = gate

    # ---- full 51-solve initialization with the band, production kappa ----
    if gate['proceed_to_full_initialization']:
        original_configure = srp.configure_shared_ess_operational_state

        def configure_then_band(model, shared_ess_idx, s_capacity, e_capacity, *args, **kwargs):
            out = original_configure(model, shared_ess_idx, s_capacity, e_capacity, *args, **kwargs)
            rows_active = any(model.sess_snet_def[i].active
                              for i in model.sess_snet_def if i[0] == shared_ess_idx)
            if rows_active and s_capacity and s_capacity > 0:
                apply_narrow_band(model, shared_ess_idx, EPSILON_REL, float(s_capacity))
            return out

        gate_report = {'run': True, 'epsilon_rel': EPSILON_REL,
                       'kappa_policy': 'untouched production 1/S_rated',
                       'applied_in_memory_only': True}
        try:
            srp.configure_shared_ess_operational_state = configure_then_band
            with redirect_stdout(quiet):
                planning2 = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
                planning2.read_planning_problem()
                candidate2 = srp._build_positive_bootstrap_candidate(
                    planning2, planning2.params.benders.positive_bootstrap)
                consensus_vars, _dual = srp.create_admm_variables(planning2)
                res = {'tso': dict(), 'dso': dict(), 'esso': dict()}
                _dm, res['dso'] = srp.create_distribution_networks_models(
                    planning2.distribution_networks, consensus_vars,
                    candidate2['total_capacity'],
                    parallel_execution=planning2.parallel_execution)
                _tm, res['tso'] = srp.create_transmission_network_model(
                    planning2, consensus_vars, candidate2['total_capacity'])
                _em, res['esso'] = srp.create_shared_energy_storage_model(
                    planning2.shared_ess_data, consensus_vars, candidate2['investment'])
                all_ok = srp._admm_local_solves_succeeded(planning2, res)
        finally:
            srp.configure_shared_ess_operational_state = original_configure

        def enumerate_results(container, prefix, out):
            if hasattr(container, 'solver'):
                out[prefix] = {
                    'status': str(getattr(container.solver, 'status', None)),
                    'termination': str(getattr(container.solver, 'termination_condition', None)),
                    'succeeded': bool(srp._solver_result_succeeded(container))}
                return out
            try:
                items = container.items()
            except AttributeError:
                out[prefix] = {'unparsed': type(container).__name__}
                return out
            for key, value in items:
                enumerate_results(value, f'{prefix}/{key}', out)
            return out

        solves = {}
        for agent in ('dso', 'tso', 'esso'):
            enumerate_results(res[agent], agent, solves)
        failures = {k: v for k, v in solves.items() if not v.get('succeeded', False)}
        per_agent = {}
        for key, value in solves.items():
            agent = key.split('/')[0]
            entry = per_agent.setdefault(agent, {'total': 0, 'failures': 0})
            entry['total'] += 1
            if not value.get('succeeded', False):
                entry['failures'] += 1
        gate_report.update({
            'total_local_solves': len(solves), 'per_agent': per_agent,
            'n_failures': len(failures), 'failures': failures,
            'all_solves': solves,
            'admm_local_solves_succeeded': bool(all_ok),
            'would_enter_admm': bool(all_ok)})
        report['full_initialization'] = gate_report
    else:
        report['full_initialization'] = {'run': False, 'reason': 'gate did not open'}

    out_path = os.path.join(OUT_DIR, 'p52a_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[P5.2-A] report -> {out_path}')
    print(f"[P5.2-A] checksum {report['scenario_checksum']}  eps_rel={EPSILON_REL}")
    for r in results:
        a, b = r['A'], r['B']
        ph = b['physical']
        print(f"  {r['label']:16s} [{r['group']:24s}] S={r['s_rated_MVA']:.5f}MVA k={r['kappa']:8.1f} "
              f"| A: {a['termination_condition']:18s} it={a['ipopt'].get('iterations')} "
              f"| B: {b['termination_condition']:18s} it={b['ipopt'].get('iterations')} "
              f"| B band_ratio={ph.get('max_band_ratio', float('nan')):.3e} "
              f"dS/S={ph.get('max_delta_S_over_s_rated', float('nan')):.3e}")
    print(f"[P5.2-A] gate: {json.dumps(report['gate'])}")
    fi = report['full_initialization']
    if fi.get('run'):
        print(f"[P5.2-A] full init: solves={fi['total_local_solves']} per_agent={fi['per_agent']} "
              f"failures={fi['n_failures']} enter_admm={fi['would_enter_admm']}")
        for k, v in fi['failures'].items():
            print(f"    FAIL {k}: {v['status']}/{v['termination']}")


if __name__ == '__main__':
    main()
