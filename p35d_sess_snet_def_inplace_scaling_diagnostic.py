"""
Stage P3.5-D -- sess_snet_def IN-PLACE equivalent scaling (no new component).

P3.5-C's R1 control showed that, for the DSO case, simply reformulating
`sess_snet_def` as a *freshly created* replacement Constraint component --
even when algebraically identical (kappa=1) and even with the original dual
copied over exactly -- was enough on its own to flip failure to success.
That confounds every P3.5-A/B/C result that relied on creating a new
Constraint component: none of them can distinguish "the scaling helped"
from "creating a new component object changed IPOPT/MA97's row/column
ordering and that alone helped".

This stage removes that confound entirely: it mutates the ORIGINAL
`sess_snet_def` ConstraintData objects in place (same Python object, same
container component, same index, same position in the model), changing only
their algebraic body from `g == 0` to `100.0 * g == 0` via `.set_value(...)`.
No Constraint component is deactivated or created. The corresponding
`model.dual` entry for each mutated row is updated on that SAME object from
`dual_before` to `dual_before / 100.0`. Nothing else is touched.

Must be run inside the real project Pyomo/IPOPT/MA97 environment, from the
repository root:

    cd /Users/micaelsimoes/PycharmProjects/shared-resources-planning
    python p35d_sess_snet_def_inplace_scaling_diagnostic.py

Writes its own log/result/report files under
data/SRP1/Results/FrozenSMOPF/P35D/ only. No production file is modified.
"""

import os
import json

import pyomo.environ as pe  # noqa: E402

from p35a_sess_snet_def_diagnostic import (  # noqa: E402  (reuse, do not duplicate)
    REPO_ROOT,
    CASES,
    sha256_of,
    load_frozen_model,
    shared_ess_index_set,
    shared_ess_capacity,
    sess_snet_def_rows,
    row_state,
    solve_case,
)
from p35c_sess_snet_def_dual_warmstart_diagnostic import residual_summary  # noqa: E402  (reuse)

OUT_ROOT = os.path.join(REPO_ROOT, 'data/SRP1/Results/FrozenSMOPF/P35D')

EXPECTED_S_RATED = 0.01
KAPPA = 100.0


def total_constraint_data_count(model):
    try:
        return model.nconstraints()
    except Exception:
        return sum(1 for _ in model.component_data_objects(pe.Constraint, active=None))


def constraint_component_names(model):
    return sorted(c.name for c in model.component_objects(pe.Constraint, active=None))


def scale_sess_snet_def_in_place(model, rows, kappa):
    """Mutate each ConstraintData's expression in place: g==0 -> kappa*g==0.
    Same Python object, same container (`model.sess_snet_def`), same index.
    No deactivate(), no new Constraint component."""
    for row in rows:
        e, s_m, s_o, p = row
        con_data = model.sess_snet_def[row]
        sch = model.shared_es_sch[e, s_m, s_o, p]
        sdch = model.shared_es_sdch[e, s_m, s_o, p]
        pnet = model.shared_es_pnet[e, s_m, s_o, p]
        qnet = model.shared_es_qnet[e, s_m, s_o, p]
        con_data.set_value(kappa * ((sch - sdch) ** 2 - pnet ** 2 - qnet ** 2) == 0)


def run_case(case_key):
    case = CASES[case_key]
    pkl_path = os.path.join(REPO_ROOT, case['pkl_path'])
    idx = case['shared_ess_index']

    report = {'case': case_key, 'config': case, 'kappa': KAPPA}

    # --- Frozen-file verification ------------------------------------------
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

    # --- A: untouched baseline ----------------------------------------------
    meta_a, model_a = load_frozen_model(pkl_path)
    model_a = model_a.clone()

    all_indices = shared_ess_index_set(model_a)
    capacities = {i: shared_ess_capacity(model_a, i) for i in all_indices}
    s_rated, e_rated = capacities.get(idx, (None, None))
    report['shared_ess_capacity_check'] = {
        'expected_index': idx,
        'all_indices_and_capacity_s_e': capacities,
        's_rated_matches_expected_0_01': (s_rated is not None and abs(s_rated - EXPECTED_S_RATED) < 1e-9),
    }
    if not s_rated or s_rated <= 0:
        report['stop_reason'] = f'Installed index {idx} has non-positive s_rated={s_rated} -- STOPPING'
        return report

    solve_a = solve_case(case_key, case, model_a, subdir='A_baseline')
    report['A_baseline'] = {k: v for k, v in solve_a.items() if k != 'result'}

    # --- B100-inplace: independent fresh clone ------------------------------
    meta_b, model_b = load_frozen_model(pkl_path)
    model_b = model_b.clone()
    rows = sess_snet_def_rows(model_b, idx)

    # --- BEFORE snapshot (for the required proof) ---------------------------
    component_before = model_b.sess_snet_def
    component_id_before = id(component_before)
    names_before = constraint_component_names(model_b)
    n_constraints_before = total_constraint_data_count(model_b)
    index_tuples_before = sorted(component_before.keys())
    data_identity_before = {row: id(component_before[row]) for row in rows}
    active_before = {row: component_before[row].active for row in rows}
    dual_before = {row: model_b.dual.get(component_before[row], None) for row in rows}

    # --- Apply the ONLY authorized change: in-place expression scaling ------
    scale_sess_snet_def_in_place(model_b, rows, KAPPA)

    # --- Apply the ONLY authorized dual change: dual_after = dual_before/kappa, same object
    dual_after = {}
    for row in rows:
        con_data = component_before[row]  # same object reference throughout
        before = dual_before[row]
        if before is None:
            dual_after[row] = None
            continue
        after = before / KAPPA
        model_b.dual[con_data] = after
        dual_after[row] = after

    # --- AFTER snapshot -------------------------------------------------------
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
        'constraint_component_names_before': names_before,
        'constraint_component_names_after': names_after,
        'no_new_constraint_component_created': names_before == names_after,
        'no_sess_snet_def_scaled_attribute_exists': not hasattr(model_b, 'sess_snet_def_scaled'),
        'total_constraint_data_count_before': n_constraints_before,
        'total_constraint_data_count_after': n_constraints_after,
        'total_constraint_data_count_unchanged': n_constraints_before == n_constraints_after,
        'index_tuples_before_count': len(index_tuples_before),
        'index_tuples_after_count': len(index_tuples_after),
        'index_tuples_unchanged': index_tuples_before == index_tuples_after,
        'all_24_rows_still_active_no_deactivation': all(active_after[row] for row in rows),
        'per_row_object_identity_unchanged': per_row_identity_unchanged,
        'all_row_identities_unchanged': all(per_row_identity_unchanged.values()),
    }

    dual_checks = []
    all_duals_correct = True
    n_with_original_dual = 0
    for row in rows:
        before = dual_before[row]
        after_set = dual_after[row]
        after_readback = dual_after_readback[row]
        if before is not None:
            n_with_original_dual += 1
        expected_after = (before / KAPPA) if before is not None else None
        matches = (
            before is not None
            and after_readback is not None
            and abs(after_readback - expected_after) < 1e-12
            and after_set == after_readback
        )
        if not matches:
            all_duals_correct = False
        dual_checks.append({
            'row': list(row),
            'dual_before': before,
            'dual_after_set': after_set,
            'dual_after_readback_same_object': after_readback,
            'expected_dual_before_over_kappa': expected_after,
            'matches': matches,
        })

    report['in_place_invariant_checks'] = identity_checks
    report['dual_transform_checks'] = {
        'kappa': KAPPA,
        'n_rows': len(rows),
        'n_with_original_dual': n_with_original_dual,
        'all_transformed_duals_equal_before_over_kappa_on_same_object': all_duals_correct,
        'per_row': dual_checks,
    }

    all_invariants_hold = (
        identity_checks['component_python_object_unchanged']
        and identity_checks['component_local_name_is_sess_snet_def']
        and identity_checks['no_new_constraint_component_created']
        and identity_checks['no_sess_snet_def_scaled_attribute_exists']
        and identity_checks['total_constraint_data_count_unchanged']
        and identity_checks['index_tuples_unchanged']
        and identity_checks['all_24_rows_still_active_no_deactivation']
        and identity_checks['all_row_identities_unchanged']
        and all_duals_correct
    )

    if not all_invariants_hold:
        report['stop_reason'] = (
            'IN-PLACE INVARIANT VIOLATION -- the required proof failed, refusing to '
            'solve B100-inplace or report it as valid evidence. See '
            'in_place_invariant_checks / dual_transform_checks for details.'
        )
        return report

    # --- Solve B100-inplace ---------------------------------------------------
    solve_b = solve_case(case_key, case, model_b, subdir='B100_inplace')
    report['B100_inplace'] = {k: v for k, v in solve_b.items() if k != 'result'}

    # --- Final ORIGINAL unscaled relation at the B solution --------------------
    states_post = [row_state(model_b, *row) for row in rows]
    report['B_final_original_unscaled_residual'] = residual_summary(states_post, s_rated, rows)

    report['pattern_check'] = {
        'A_failed': not solve_a['succeeded'],
        'B_succeeded': solve_b['succeeded'],
        'B_succeeded_without_recovery': solve_b['succeeded'] and len(solve_b['log_files']) <= 1,
    }

    return report


def main():
    full_report = {}
    for case_key in ('DSO', 'TSO'):
        full_report[case_key] = run_case(case_key)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, 'p35d_report.json')
    with open(out_path, 'w') as handle:
        json.dump(full_report, handle, indent=2, default=str)
    print(f'[P3.5-D] Wrote full report to {out_path}')
    print(json.dumps(full_report, indent=2, default=str))


if __name__ == '__main__':
    main()
