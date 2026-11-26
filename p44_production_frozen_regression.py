"""
Stage P4.4 -- production `sess_snet_def` frozen-regression replay.

Unlike P3/P3.5 (frozen diagnostics with hand-written scaled-replacement or
in-place-mutation logic implemented INSIDE the diagnostic script itself),
this stage replays every preserved P3 failure through the REAL, now-patched
production functions:

  - `model_construction_helpers.sess_snet_def_rule` -- called directly, not
    reimplemented, so each row's rewritten expression is byte-for-byte what
    a model BUILT today under current production code would contain;
  - `model_construction_helpers.shared_ess_snet_def_scale` -- the real
    kappa_e = 1/S_scale_e helper;
  - `model_construction_helpers._sync_sess_snet_def_scale` -- the real
    KKT-consistent dual-transfer function used by
    `configure_shared_ess_operational_state` in production.

A frozen pickle predates the P4 code change: its `sess_snet_def` rows were
built with the OLD rule (`g == 0`, no `sess_snet_def_kappa` reference at
all), and a Pyomo `ConstraintData`'s algebraic body is fixed at construction
time -- reloading the pickle does not retroactively re-run today's rule. So
"replaying under production code" necessarily means re-deriving each frozen
row's body via a call to the CURRENT `sess_snet_def_rule` and writing it
back onto the SAME `ConstraintData` object with `.set_value(...)` (the
identical in-place-mutation mechanism P3.5-D already proved is safe and
confound-free) -- then applying the real `_sync_sess_snet_def_scale` for the
dual transfer, exactly as `configure_shared_ess_operational_state` would
during a live capacity update. The frozen state is treated as an implicit
kappa_old=1.0 (the pre-P4 `g==0` relation), transferring to
kappa_new = shared_ess_snet_def_scale(s_rated) = 100.0 for both decisive
cases (s_rated=0.01).

Every required structural/identity/dual-transform invariant from P3.5-D is
re-proven here, through the real production functions, before any solve.

Required first gate (LOCAL_NLP_STABILITY_PLAN.md sec. 6): both decisive
cases (DSO Autumn/cycle8, TSO Summer/cycle6) must reach primary success, no
recovery, with the original unscaled relation within normal feasibility
tolerance. If either fails, this script STOPS -- it does not replay the
other five preserved P3 failures or the two bonus matched-success controls.

No production file is modified. Must be run inside the real project
Pyomo/IPOPT/MA97 environment, from the repository root:

    cd /Users/micaelsimoes/PycharmProjects/shared-resources-planning
    python p44_production_frozen_regression.py

Writes its own report under data/SRP1/Results/FrozenSMOPF/P44/ only.
"""

import os
import json

import pyomo.environ as pe  # noqa: E402

from p35a_sess_snet_def_diagnostic import (  # noqa: E402  (reuse, do not duplicate)
    REPO_ROOT,
    sha256_of,
    load_frozen_model,
    shared_ess_capacity,
    sess_snet_def_rows,
    row_state,
    solve_case,
)
from p35c_sess_snet_def_dual_warmstart_diagnostic import residual_summary  # noqa: E402  (reuse)
from model_construction_helpers import (  # noqa: E402  (the REAL, patched production functions)
    sess_snet_def_rule,
    shared_ess_snet_def_scale,
    _sync_sess_snet_def_scale,
)
from definitions import SHARED_ESS_SNET_DEF_SAFE_KAPPA  # noqa: E402

OUT_ROOT = os.path.join(REPO_ROOT, 'data/SRP1/Results/FrozenSMOPF/P44')

EXPECTED_S_RATED = 0.01
EXPECTED_KAPPA = 100.0

_DSO_COMMON = {
    'network_name': 'case33_2',
    'params_file': 'data/SRP1/case33_2/case33_2_params.json',
    'is_transmission': False,
    'shared_ess_index': 0,
    'from_warm_start': True,
}
_TSO_COMMON = {
    'network_name': 'case9',
    'params_file': 'data/SRP1/case9/case9_params.json',
    'is_transmission': True,
    'shared_ess_index': 1,
    'from_warm_start': True,
}

DECISIVE = {
    'DSO_decisive_Autumn_cycle8': {
        **_DSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl',
        'sha256': '066117b88085e5d8b20ec4da684555902d57565044d4cf293516796637074711',
        'year': 2025, 'day': 'Autumn', 'cycle': 8,
    },
    'TSO_decisive_Summer_cycle6': {
        **_TSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl',
        'sha256': '51d9097418561612d61367d12600ea3929b622c367a5feeef1ab6efd4b08355e',
        'year': 2025, 'day': 'Summer', 'cycle': 6,
    },
}

OTHER_PRESERVED = {
    'DSO_Summer_cycle1': {
        **_DSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Summer_cycle1.pkl',
        'sha256': '0da3dbe21029fc49621fde46ad0717be6ff4fd064712e3a4d8893478d1c34794',
        'year': 2025, 'day': 'Summer', 'cycle': 1,
    },
    'DSO_2030_Winter_cycle1': {
        **_DSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2030_Winter_cycle1.pkl',
        'sha256': 'ec058125d7eb088b092312863a759d9c3ced6421675d3b8fb1ea2e6ceb4708fb',
        'year': 2030, 'day': 'Winter', 'cycle': 1,
    },
    'DSO_Autumn_cycle12': {
        **_DSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle12.pkl',
        'sha256': '113909c05b89fbf71ca16e5470c3a48a4f0bdc258057e227f33b50444c8c3e23',
        'year': 2025, 'day': 'Autumn', 'cycle': 12,
    },
    'DSO_Autumn_cycle13': {
        **_DSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle13.pkl',
        'sha256': '92e0a1d5ada0f8565e9dce6431775f2330f961fa151b938fbf15264f372471cd',
        'year': 2025, 'day': 'Autumn', 'cycle': 13,
    },
    'TSO_Winter_cycle5': {
        **_TSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Winter_cycle5.pkl',
        'sha256': '228fedcda91c8db5e4f3d72a886c1f27614a35362da96e39ab36c291295df728',
        'year': 2025, 'day': 'Winter', 'cycle': 5,
    },
}

# Bonus, not required by LOCAL_NLP_STABILITY_PLAN.md sec. 6: two matched
# P3 successes (no numerical failure at all), replayed only as an
# additional no-regression check if the decisive gate passes.
BONUS_CONTROLS = {
    'DSO_matched_success_Autumn_cycle7': {
        **_DSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
        'sha256': '8eabd9ee566182a887e2e790e8ab7993ab922e22d79e3712fdf9288fb726acbe',
        'year': 2025, 'day': 'Autumn', 'cycle': 7,
    },
    'TSO_matched_success_Summer_cycle7': {
        **_TSO_COMMON,
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl',
        'sha256': '15ce6ebef2511655b7202652cbce9327ef4295773458ddc2c9b7f94fc5daa8e9',
        'year': 2025, 'day': 'Summer', 'cycle': 7,
    },
}


def constraint_component_names(model):
    return sorted(c.name for c in model.component_objects(pe.Constraint, active=None))


def total_constraint_data_count(model):
    try:
        return model.nconstraints()
    except Exception:
        return sum(1 for _ in model.component_data_objects(pe.Constraint, active=None))


def apply_production_normalization(model, idx, rows):
    """Reproduce, on a frozen pre-P4 model, exactly what a model built under
    CURRENT production code would contain for `sess_snet_def`, using the
    real production functions (not a reimplementation):

      1. add `sess_snet_def_kappa` if this frozen model predates it, with
         the same declaration network.py uses;
      2. rewrite each row's body by calling the current `sess_snet_def_rule`
         directly, via `.set_value()` on the SAME ConstraintData object
         (P3.5-D's proven-safe in-place-mutation mechanism);
      3. transfer the frozen dual via the real `_sync_sess_snet_def_scale`,
         treating the frozen (pre-P4, literal g==0) state as kappa_old=1.0.

    Returns (kappa_new, con_data_objects) for the caller's invariant proof.
    """
    if not hasattr(model, 'sess_snet_def_kappa'):
        model.sess_snet_def_kappa = pe.Param(
            model.shared_energy_storages, mutable=True,
            initialize=SHARED_ESS_SNET_DEF_SAFE_KAPPA,
        )
    s_rated = pe.value(model.shared_es_s_rated_fixed[idx])
    kappa_new = shared_ess_snet_def_scale(s_rated)

    for row in rows:
        e, s_m, s_o, p = row
        new_expr = sess_snet_def_rule(model, e, s_m, s_o, p)  # the REAL production rule, called directly
        model.sess_snet_def[row].set_value(new_expr)

    con_data_objects = [model.sess_snet_def[r] for r in rows]
    _sync_sess_snet_def_scale(  # the REAL production dual-transfer function
        model, con_data_objects, idx,
        kappa_old=1.0, kappa_new=kappa_new,
        was_active=True, will_be_active=True,
    )
    return kappa_new, con_data_objects


def run_case(case_key, case):
    pkl_path = os.path.join(REPO_ROOT, case['pkl_path'])
    idx = case['shared_ess_index']
    report = {'case': case_key, 'config': case}

    # --- Frozen-file verification --------------------------------------
    actual_hash = sha256_of(pkl_path)
    report['hash_check'] = {
        'path': case['pkl_path'],
        'expected_sha256': case['sha256'],
        'actual_sha256': actual_hash,
        'match': actual_hash == case['sha256'],
    }
    if actual_hash != case['sha256']:
        report['stop_reason'] = 'HASH MISMATCH -- STOPPING per plan sec. 4'
        return report

    # --- A: untouched baseline ------------------------------------------
    meta_a, model_a = load_frozen_model(pkl_path)
    model_a = model_a.clone()
    all_indices = list(model_a.shared_energy_storages)
    s_rated, e_rated = shared_ess_capacity(model_a, idx)
    report['shared_ess_capacity_check'] = {
        'expected_index': idx,
        'all_indices': all_indices,
        's_rated': s_rated, 'e_rated': e_rated,
        's_rated_matches_expected_0_01': (s_rated is not None and abs(s_rated - EXPECTED_S_RATED) < 1e-9),
    }
    if not s_rated or s_rated <= 0:
        report['stop_reason'] = f'Installed index {idx} has non-positive s_rated={s_rated} -- STOPPING'
        return report

    solve_a = solve_case(case_key, case, model_a, subdir='A_baseline')
    report['A_baseline'] = {k: v for k, v in solve_a.items() if k != 'result'}

    # --- B: production-normalization replay ------------------------------
    meta_b, model_b = load_frozen_model(pkl_path)
    model_b = model_b.clone()
    rows = sess_snet_def_rows(model_b, idx)

    component_before = model_b.sess_snet_def
    component_id_before = id(component_before)
    names_before = constraint_component_names(model_b)
    n_constraints_before = total_constraint_data_count(model_b)
    index_tuples_before = sorted(component_before.keys())
    data_identity_before = {row: id(component_before[row]) for row in rows}
    active_before = {row: component_before[row].active for row in rows}
    dual_before = {row: model_b.dual.get(component_before[row], None) for row in rows}

    kappa_new, con_data_objects = apply_production_normalization(model_b, idx, rows)

    component_after = model_b.sess_snet_def
    component_id_after = id(component_after)
    names_after = constraint_component_names(model_b)
    n_constraints_after = total_constraint_data_count(model_b)
    index_tuples_after = sorted(component_after.keys())
    data_identity_after = {row: id(component_after[row]) for row in rows}
    active_after = {row: component_after[row].active for row in rows}
    dual_after_readback = {row: model_b.dual.get(component_after[row], None) for row in rows}

    per_row_identity_unchanged = {str(row): (data_identity_before[row] == data_identity_after[row]) for row in rows}

    identity_checks = {
        'component_python_object_unchanged': component_id_before == component_id_after,
        'component_local_name_is_sess_snet_def': component_after.local_name == 'sess_snet_def',
        'no_new_constraint_component_created': names_before == names_after,
        'total_constraint_data_count_unchanged': n_constraints_before == n_constraints_after,
        'index_tuples_unchanged': index_tuples_before == index_tuples_after,
        'all_rows_still_active_no_deactivation': all(active_after[row] for row in rows),
        'per_row_object_identity_unchanged': per_row_identity_unchanged,
        'all_row_identities_unchanged': all(per_row_identity_unchanged.values()),
        'kappa_new': kappa_new,
        'kappa_new_equals_expected_100': abs(kappa_new - EXPECTED_KAPPA) < 1e-6,
    }

    dual_checks = []
    all_duals_correct = True
    n_with_original_dual = 0
    for row in rows:
        before = dual_before[row]
        after_readback = dual_after_readback[row]
        if before is not None:
            n_with_original_dual += 1
        expected_after = (before * (1.0 / kappa_new)) if before is not None else None
        matches = (
            before is not None
            and after_readback is not None
            and abs(after_readback - expected_after) < 1e-9
        )
        if not matches:
            all_duals_correct = False
        dual_checks.append({
            'row': list(row),
            'dual_before_frozen': before,
            'dual_after_production_transfer': after_readback,
            'expected_dual_before_times_kappa_old_over_kappa_new': expected_after,
            'matches': matches,
        })

    report['production_normalization_invariant_checks'] = identity_checks
    report['dual_transform_checks'] = {
        'kappa_old_implicit': 1.0,
        'kappa_new': kappa_new,
        'n_rows': len(rows),
        'n_with_original_dual': n_with_original_dual,
        'all_transformed_duals_match_kkt_rule': all_duals_correct,
        'per_row': dual_checks,
    }

    all_invariants_hold = (
        identity_checks['component_python_object_unchanged']
        and identity_checks['component_local_name_is_sess_snet_def']
        and identity_checks['no_new_constraint_component_created']
        and identity_checks['total_constraint_data_count_unchanged']
        and identity_checks['index_tuples_unchanged']
        and identity_checks['all_rows_still_active_no_deactivation']
        and identity_checks['all_row_identities_unchanged']
        and identity_checks['kappa_new_equals_expected_100']
        and all_duals_correct
    )

    if not all_invariants_hold:
        report['stop_reason'] = (
            'PRODUCTION NORMALIZATION INVARIANT VIOLATION -- refusing to solve B '
            'or report it as valid evidence. See production_normalization_invariant_checks '
            '/ dual_transform_checks for details.'
        )
        return report

    solve_b = solve_case(case_key, case, model_b, subdir='B_production_normalized')
    report['B_production_normalized'] = {k: v for k, v in solve_b.items() if k != 'result'}

    states_post = [row_state(model_b, *row) for row in rows]
    report['B_final_original_unscaled_residual'] = residual_summary(states_post, s_rated, rows)

    max_normalized_residual = (
        report['B_final_original_unscaled_residual']['max_normalized_residual']
        if report['B_final_original_unscaled_residual'] else None
    )

    report['pattern_check'] = {
        'A_failed': not solve_a['succeeded'],
        'B_succeeded': solve_b['succeeded'],
        'B_succeeded_without_recovery': solve_b['succeeded'] and len(solve_b['log_files']) <= 1,
        'B_residual_within_normal_tolerance': (
            max_normalized_residual is not None and max_normalized_residual < 1e-6
        ),
    }

    return report


def gate_passed(report):
    if 'stop_reason' in report:
        return False
    pattern = report.get('pattern_check', {})
    return bool(
        pattern.get('B_succeeded')
        and pattern.get('B_succeeded_without_recovery')
        and pattern.get('B_residual_within_normal_tolerance')
    )


def main():
    full_report = {'decisive': {}, 'other_preserved': {}, 'bonus_controls': {}}

    for case_key, case in DECISIVE.items():
        full_report['decisive'][case_key] = run_case(case_key, case)

    decisive_gate = all(gate_passed(r) for r in full_report['decisive'].values())
    full_report['required_first_gate'] = {
        'both_decisive_cases_primary_success_no_recovery_within_tolerance': decisive_gate,
        'per_case': {
            k: {
                'gate_passed': gate_passed(v),
                'stop_reason': v.get('stop_reason'),
                'pattern_check': v.get('pattern_check'),
            }
            for k, v in full_report['decisive'].items()
        },
    }

    if not decisive_gate:
        full_report['stop_reason'] = (
            'REQUIRED FIRST GATE FAILED -- per LOCAL_NLP_STABILITY_PLAN.md sec. 6, '
            'not replaying other_preserved or bonus_controls. See required_first_gate.'
        )
    else:
        for case_key, case in OTHER_PRESERVED.items():
            full_report['other_preserved'][case_key] = run_case(case_key, case)
        for case_key, case in BONUS_CONTROLS.items():
            full_report['bonus_controls'][case_key] = run_case(case_key, case)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, 'p44_report.json')
    with open(out_path, 'w') as handle:
        json.dump(full_report, handle, indent=2, default=str)
    print(f'[P4.4] Wrote full report to {out_path}')

    summary = {'required_first_gate': full_report['required_first_gate']['both_decisive_cases_primary_success_no_recovery_within_tolerance']}
    for section in ('decisive', 'other_preserved', 'bonus_controls'):
        summary[section] = {
            k: v.get('pattern_check', {'stop_reason': v.get('stop_reason')})
            for k, v in full_report[section].items()
        }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
