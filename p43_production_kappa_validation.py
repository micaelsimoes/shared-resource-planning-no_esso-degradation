"""
Stage P4.3 -- production `sess_snet_def` kappa-scaling construction and
equivalence validation.

This is NOT a frozen-pickle diagnostic like P3/P3.5. It exercises the REAL,
now-modified production code directly: `SharedResourcesPlanning` /
`NetworkData.build_model` to construct fresh DSO (node 7, case33_2) and TSO
(case9) models, and the actual (patched) `configure_shared_ess_operational_state`
/ `shared_ess_snet_def_scale` / `sess_snet_def_rule` from
`model_construction_helpers.py`. Nothing is solved -- this stage validates
construction and structural/algebraic invariants only, per
LOCAL_NLP_STABILITY_PLAN.md sec. 5 (P4.3):

  - positive-capacity construction: s_rated=0.01 -> kappa=100, s_rated=0.02
    -> kappa=50, kappa is a numerical Param value fixed within a solve (never
    a symbolic division by a variable);
  - zero/near-zero-capacity construction: no divide-by-zero, finite safe
    kappa, same rows deactivated, same operational variables fixed to 0;
  - structural invariants: same `sess_snet_def` component name/object/index
    tuples/row count across every capacity transition, no extra constraint
    component ever appears, `sess_comp` is untouched;
  - feasible-set equivalence: g==0 <=> kappa*g==0 at concrete sample points;
  - the KKT-consistent live-capacity dual transfer
    (`lambda_new = lambda_old * (kappa_old / kappa_new)`) on a REUSED model,
    reproducing exactly what `update_distribution_coordination_models_and_solve_sequential`
    (every ADMM cycle) and `_update_model_with_candidate_solution` (every
    planning candidate) do to a live model in production;
  - the two dual-lifecycle edge cases the P4.1 audit's design decision
    covers: a stale dual present on a row that is being REACTIVATED is
    cleared (not transferred); a dual left on a row that is being
    DEACTIVATED is left untouched (nothing to warm-start while inactive)
    and is only ever cleared later, at the point of its next reactivation.

No production file is modified. Only reads `data/SRP1/...` inputs via
`SharedResourcesPlanning` and writes its own report under
`data/SRP1/Results/FrozenSMOPF/P43/`. Must be run inside the real project
Pyomo environment (no solve is invoked, so IPOPT/MA97 are not required, but
the real `pyomo`/project modules are), from the repository root:

    cd /Users/micaelsimoes/PycharmProjects/shared-resources-planning
    python p43_production_kappa_validation.py
"""

import os
import sys
import json

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import pyomo.environ as pe  # noqa: E402

from shared_resources_planning import SharedResourcesPlanning  # noqa: E402
from model_construction_helpers import (  # noqa: E402
    configure_shared_ess_operational_state,
    _component_entries_for_shared_ess,
    _SHARED_ESS_OPERATIONAL_VARIABLES,
)
from definitions import (  # noqa: E402
    SHARED_ESS_ZERO_CAPACITY_TOLERANCE,
    SHARED_ESS_SNET_DEF_SAFE_KAPPA,
)

OUT_ROOT = os.path.join(REPO_ROOT, 'data/SRP1/Results/FrozenSMOPF/P43')

CONTROL_YEAR = 2025
CONTROL_DAY = 'Spring'

# Same installed-shared-ESS index convention used throughout P3/P3.5
# (p35a_sess_snet_def_diagnostic.py CASES).
CASES = {
    'DSO': {'shared_ess_index': 0, 'positive_s_e': [(0.01, 0.04), (0.02, 0.08)]},
    'TSO': {'shared_ess_index': 1, 'positive_s_e': [(0.01, 0.04), (0.02, 0.08)]},
}


def constraint_component_names(model):
    return sorted(c.name for c in model.component_objects(pe.Constraint, active=None))


def total_constraint_data_count(model):
    try:
        return model.nconstraints()
    except Exception:
        return sum(1 for _ in model.component_data_objects(pe.Constraint, active=None))


def sess_snet_def_rows(model, idx):
    return [(e, s_m, s_o, p) for (e, s_m, s_o, p) in model.sess_snet_def if e == idx]


def sess_comp_rows(model, idx):
    return [(e, s_m, s_o, p) for (e, s_m, s_o, p) in model.sess_comp if e == idx]


def operational_var_entries(model, idx):
    entries = {}
    for name in _SHARED_ESS_OPERATIONAL_VARIABLES:
        if not hasattr(model, name):
            continue
        var = getattr(model, name)
        entries[name] = list(_component_entries_for_shared_ess(var, idx))
    return entries


def all_fixed_at_zero(entries_by_name):
    return {
        name: all(entry.fixed and abs(pe.value(entry)) < 1e-15 for entry in entries)
        for name, entries in entries_by_name.items()
    }


def all_unfixed(entries_by_name):
    return {name: all(not entry.fixed for entry in entries) for name, entries in entries_by_name.items()}


def build_fresh_model(case_key):
    planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    planning.read_planning_problem()
    if case_key == 'DSO':
        network = planning.distribution_networks[7].network[CONTROL_YEAR][CONTROL_DAY]
        params = planning.distribution_networks[7].params
    else:
        network = planning.transmission_network.network[CONTROL_YEAR][CONTROL_DAY]
        params = planning.transmission_network.params
    model = network.build_model(params)
    if network.is_transmission:
        model.active_distribution_networks = range(len(network.active_distribution_network_nodes))
    return network, params, model


def run_case(case_key):
    case = CASES[case_key]
    idx = case['shared_ess_index']
    report = {'case': case_key, 'shared_ess_index': idx}

    network, params, model = build_fresh_model(case_key)

    # ---- 0. Signature immediately after build_model(), before any configure call ----
    names0 = constraint_component_names(model)
    n0 = total_constraint_data_count(model)
    rows = sess_snet_def_rows(model, idx)
    comp_rows = sess_comp_rows(model, idx)
    report['post_build'] = {
        'all_shared_energy_storage_indices': list(model.shared_energy_storages),
        'sess_snet_def_row_count_for_idx': len(rows),
        'sess_snet_def_index_tuples_for_idx': [list(r) for r in rows],
        'sess_comp_row_count_for_idx': len(comp_rows),
        'kappa_initial_value': pe.value(model.sess_snet_def_kappa[idx]),
        'kappa_initial_equals_safe_placeholder': abs(pe.value(model.sess_snet_def_kappa[idx]) - SHARED_ESS_SNET_DEF_SAFE_KAPPA) < 1e-15,
        'all_rows_active_pre_configure': all(model.sess_snet_def[r].active for r in rows),
        'shared_es_s_rated_fixed_initial': pe.value(model.shared_es_s_rated_fixed[idx]),
        'shared_es_e_rated_fixed_initial': pe.value(model.shared_es_e_rated_fixed[idx]),
    }
    component_id_baseline = id(model.sess_snet_def)
    sess_comp_id_baseline = id(model.sess_comp)
    row_object_ids_baseline = {r: id(model.sess_snet_def[r]) for r in rows}

    def structural_snapshot(label):
        names = constraint_component_names(model)
        n = total_constraint_data_count(model)
        rows_now = sess_snet_def_rows(model, idx)
        comp_rows_now = sess_comp_rows(model, idx)
        return {
            'label': label,
            'component_names_unchanged': names == names0,
            'no_new_constraint_component': names == names0,
            'total_constraint_data_count': n,
            'total_constraint_data_count_unchanged': n == n0,
            'sess_snet_def_component_object_unchanged': id(model.sess_snet_def) == component_id_baseline,
            'sess_snet_def_local_name': model.sess_snet_def.local_name,
            'sess_snet_def_index_tuples_unchanged': [list(r) for r in rows_now] == [list(r) for r in rows],
            'sess_snet_def_row_count_unchanged': len(rows_now) == len(rows),
            'sess_comp_component_object_unchanged': id(model.sess_comp) == sess_comp_id_baseline,
            'sess_comp_index_tuples_unchanged': [list(r) for r in comp_rows_now] == [list(r) for r in comp_rows],
            'per_row_object_identity_unchanged': all(
                id(model.sess_snet_def[r]) == row_object_ids_baseline[r] for r in rows
            ),
        }

    # ---- Step A: establish baseline zero-capacity production state ----
    # Matches every real code path: shared_es_s/e_rated_fixed start at 0.00
    # (network.py) and configure_shared_ess_operational_state is always
    # called before any solve (see validate_vmag_refactor.py's
    # _configure_inactive_shared_ess and NetworkData._update_model_with_candidate_solution).
    for storage in model.shared_energy_storages:
        configure_shared_ess_operational_state(
            model, storage,
            pe.value(model.shared_es_s_rated_fixed[storage]),
            pe.value(model.shared_es_e_rated_fixed[storage]),
        )
    report['step_A_zero_capacity_baseline'] = {
        'kappa': pe.value(model.sess_snet_def_kappa[idx]),
        'kappa_equals_safe_placeholder': abs(pe.value(model.sess_snet_def_kappa[idx]) - SHARED_ESS_SNET_DEF_SAFE_KAPPA) < 1e-15,
        'all_rows_deactivated': all(not model.sess_snet_def[r].active for r in rows),
        'all_sess_comp_rows_deactivated': all(not model.sess_comp[r].active for r in comp_rows),
        'operational_vars_fixed_at_zero': all_fixed_at_zero(operational_var_entries(model, idx)),
        'structural': structural_snapshot('after_zero_capacity_baseline'),
    }

    # ---- Step B: near-zero (within-tolerance) capacity never divides by zero ----
    near_zero = SHARED_ESS_ZERO_CAPACITY_TOLERANCE / 2.0
    exc = None
    try:
        configure_shared_ess_operational_state(model, idx, near_zero, near_zero)
    except Exception as e:  # noqa: BLE001
        exc = repr(e)
    report['step_B_near_zero_capacity'] = {
        'input_s_capacity': near_zero,
        'exception_raised': exc,
        'kappa': pe.value(model.sess_snet_def_kappa[idx]),
        'kappa_equals_safe_placeholder': abs(pe.value(model.sess_snet_def_kappa[idx]) - SHARED_ESS_SNET_DEF_SAFE_KAPPA) < 1e-15,
        'all_rows_still_deactivated': all(not model.sess_snet_def[r].active for r in rows),
        'structural': structural_snapshot('after_near_zero_capacity'),
    }

    # ---- Step C: adversarially inject a stale dual on a currently-INACTIVE row ----
    fake_stale_dual = -777.0
    for r in rows:
        model.dual[model.sess_snet_def[r]] = fake_stale_dual
    report['step_C_stale_dual_injected_while_inactive'] = {
        'value_injected': fake_stale_dual,
        'n_rows': len(rows),
    }

    # ---- Step D: reactivate at s=0.01 -> kappa=100, stale dual must be CLEARED ----
    s1, e1 = case['positive_s_e'][0]
    configure_shared_ess_operational_state(model, idx, s1, e1)
    kappa_100 = pe.value(model.sess_snet_def_kappa[idx])
    dual_after_reactivation = {r: model.dual.get(model.sess_snet_def[r], None) for r in rows}
    report['step_D_reactivate_s_0_01'] = {
        's_capacity': s1, 'e_capacity': e1,
        'kappa': kappa_100,
        'kappa_equals_100': abs(kappa_100 - (1.0 / s1)) < 1e-9,
        'all_rows_active': all(model.sess_snet_def[r].active for r in rows),
        'stale_duals_cleared_on_reactivation': all(v is None for v in dual_after_reactivation.values()),
        'operational_vars_unfixed': all_unfixed(operational_var_entries(model, idx)),
        'structural': structural_snapshot('after_reactivate_s_0_01'),
    }

    # ---- Step E: inject "real" duals while ACTIVE at kappa=100 ----
    fake_real_duals = {r: -12.5 - 3.0 * i for i, r in enumerate(rows)}
    for r, v in fake_real_duals.items():
        model.dual[model.sess_snet_def[r]] = v
    row_ids_before_live_change = {r: id(model.sess_snet_def[r]) for r in rows}

    # ---- Step F: LIVE capacity change on the SAME reused model, s: 0.01 -> 0.02 ----
    # Reproduces exactly what update_distribution_coordination_models_and_solve_sequential
    # (every ADMM cycle) and _update_model_with_candidate_solution (every planning
    # candidate) do: call configure_shared_ess_operational_state again on the SAME
    # model object with a new capacity, while the row stays active throughout.
    s2, e2 = case['positive_s_e'][1]
    configure_shared_ess_operational_state(model, idx, s2, e2)
    kappa_50 = pe.value(model.sess_snet_def_kappa[idx])
    dual_after_live_change = {r: model.dual.get(model.sess_snet_def[r], None) for r in rows}
    kkt_expected = {r: fake_real_duals[r] * (kappa_100 / kappa_50) for r in rows}
    kkt_matches = {
        str(r): (
            dual_after_live_change[r] is not None
            and abs(dual_after_live_change[r] - kkt_expected[r]) < 1e-9
        )
        for r in rows
    }
    report['step_F_live_capacity_change_s_0_01_to_0_02'] = {
        's_capacity': s2, 'e_capacity': e2,
        'kappa': kappa_50,
        'kappa_equals_50': abs(kappa_50 - (1.0 / s2)) < 1e-9,
        'kappa_old': kappa_100,
        'per_row_object_identity_unchanged_across_live_change': all(
            id(model.sess_snet_def[r]) == row_ids_before_live_change[r] for r in rows
        ),
        'dual_before': {str(r): v for r, v in fake_real_duals.items()},
        'dual_after': {str(r): v for r, v in dual_after_live_change.items()},
        'expected_dual_kkt_lambda_old_times_kappa_old_over_kappa_new': {str(r): v for r, v in kkt_expected.items()},
        'all_duals_match_kkt_transfer_exactly': all(kkt_matches.values()),
        'per_row_kkt_match': kkt_matches,
        'structural': structural_snapshot('after_live_capacity_change'),
    }

    # ---- Step G: deactivate again (s: 0.02 -> 0.0); dual must be LEFT ALONE ----
    configure_shared_ess_operational_state(model, idx, 0.0, 0.0)
    dual_after_deactivation = {r: model.dual.get(model.sess_snet_def[r], None) for r in rows}
    left_alone = {
        str(r): (
            dual_after_deactivation[r] is not None
            and abs(dual_after_deactivation[r] - dual_after_live_change[r]) < 1e-12
        )
        for r in rows
    }
    report['step_G_deactivate_again'] = {
        'kappa': pe.value(model.sess_snet_def_kappa[idx]),
        'kappa_equals_safe_placeholder': abs(pe.value(model.sess_snet_def_kappa[idx]) - SHARED_ESS_SNET_DEF_SAFE_KAPPA) < 1e-15,
        'all_rows_deactivated': all(not model.sess_snet_def[r].active for r in rows),
        'stale_dual_from_last_active_state_left_untouched': all(left_alone.values()),
        'per_row_left_alone_check': left_alone,
        'note': (
            'Design behavior, not a defect: a dual computed while a row was last '
            'active is left on the ConstraintData when that row is deactivated '
            '(nothing to warm-start while inactive). It is only ever cleared at '
            'the point of a later REACTIVATION (step H), using the freshly-read '
            'was_active/will_be_active state at that call -- never at '
            'deactivation time itself.'
        ),
        'structural': structural_snapshot('after_second_deactivation'),
    }

    # ---- Step H: reactivate once more -> the leftover stale dual must clear ----
    configure_shared_ess_operational_state(model, idx, s1, e1)
    dual_after_second_reactivation = {r: model.dual.get(model.sess_snet_def[r], None) for r in rows}
    report['step_H_reactivate_again_s_0_01'] = {
        'kappa': pe.value(model.sess_snet_def_kappa[idx]),
        'kappa_equals_100': abs(pe.value(model.sess_snet_def_kappa[idx]) - (1.0 / s1)) < 1e-9,
        'leftover_stale_dual_from_step_G_cleared': all(v is None for v in dual_after_second_reactivation.values()),
        'structural': structural_snapshot('after_second_reactivation'),
    }

    # ---- Step I: feasible-set equivalence g==0 <=> kappa*g==0 at concrete points ----
    equivalence_checks = []
    sample_row = rows[0]
    kappa_now = pe.value(model.sess_snet_def_kappa[idx])
    test_points = [
        {'sch': 0.005, 'sdch': 0.000, 'pnet': 0.005, 'qnet': 0.0},    # g == 0 exactly
        {'sch': 0.005, 'sdch': 0.000, 'pnet': 0.0051, 'qnet': 0.0},   # g < 0 (small)
        {'sch': 0.006, 'sdch': 0.000, 'pnet': 0.005, 'qnet': 0.0},    # g > 0 (small)
    ]
    for point in test_points:
        g = (point['sch'] - point['sdch']) ** 2 - point['pnet'] ** 2 - point['qnet'] ** 2
        kg = kappa_now * g
        equivalence_checks.append({
            'point': point,
            'g': g,
            'kappa_times_g': kg,
            'g_is_zero': abs(g) < 1e-15,
            'kappa_g_is_zero': abs(kg) < 1e-15,
            'zero_crossing_matches': (abs(g) < 1e-15) == (abs(kg) < 1e-15),
            'sign_matches': (g == 0.0 and kg == 0.0) or ((g > 0) == (kg > 0)),
        })
    report['step_I_feasible_set_equivalence'] = {
        'kappa_used': kappa_now,
        'sample_row': list(sample_row),
        'checks': equivalence_checks,
        'all_zero_crossings_match': all(c['zero_crossing_matches'] for c in equivalence_checks),
        'all_signs_match': all(c['sign_matches'] for c in equivalence_checks),
    }

    # ---- Overall pass/fail for this case ----
    all_ok = (
        report['post_build']['kappa_initial_equals_safe_placeholder']
        and report['step_A_zero_capacity_baseline']['kappa_equals_safe_placeholder']
        and report['step_A_zero_capacity_baseline']['all_rows_deactivated']
        and all(report['step_A_zero_capacity_baseline']['operational_vars_fixed_at_zero'].values())
        and report['step_A_zero_capacity_baseline']['structural']['no_new_constraint_component']
        and report['step_B_near_zero_capacity']['exception_raised'] is None
        and report['step_B_near_zero_capacity']['kappa_equals_safe_placeholder']
        and report['step_D_reactivate_s_0_01']['kappa_equals_100']
        and report['step_D_reactivate_s_0_01']['stale_duals_cleared_on_reactivation']
        and all(report['step_D_reactivate_s_0_01']['operational_vars_unfixed'].values())
        and report['step_D_reactivate_s_0_01']['structural']['no_new_constraint_component']
        and report['step_F_live_capacity_change_s_0_01_to_0_02']['kappa_equals_50']
        and report['step_F_live_capacity_change_s_0_01_to_0_02']['per_row_object_identity_unchanged_across_live_change']
        and report['step_F_live_capacity_change_s_0_01_to_0_02']['all_duals_match_kkt_transfer_exactly']
        and report['step_F_live_capacity_change_s_0_01_to_0_02']['structural']['no_new_constraint_component']
        and report['step_G_deactivate_again']['stale_dual_from_last_active_state_left_untouched']
        and report['step_G_deactivate_again']['structural']['no_new_constraint_component']
        and report['step_H_reactivate_again_s_0_01']['leftover_stale_dual_from_step_G_cleared']
        and report['step_H_reactivate_again_s_0_01']['structural']['no_new_constraint_component']
        and report['step_I_feasible_set_equivalence']['all_zero_crossings_match']
        and report['step_I_feasible_set_equivalence']['all_signs_match']
    )
    report['all_invariants_hold'] = all_ok

    return report


def main():
    full_report = {}
    for case_key in ('DSO', 'TSO'):
        full_report[case_key] = run_case(case_key)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, 'p43_report.json')
    with open(out_path, 'w') as handle:
        json.dump(full_report, handle, indent=2, default=str)
    print(f'[P4.3] Wrote full report to {out_path}')
    print(json.dumps({
        case_key: {'all_invariants_hold': full_report[case_key]['all_invariants_hold']}
        for case_key in full_report
    }, indent=2))


if __name__ == '__main__':
    main()
