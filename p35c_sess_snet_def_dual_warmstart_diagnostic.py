"""
Stage P3.5-C -- sess_snet_def dual-multiplier warm-start confound check.

Planner concern: the production model declares
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
(network.py, alongside model.ipopt_zL_in/zU_in) explicitly to "Obtain dual
solutions from previous solve and send to warm start". Both P3.5-A and
P3.5-B ran with from_warm_start=True, deactivated the original
`sess_snet_def` rows, and created BRAND NEW replacement Constraint
components (`sess_snet_def_scaled`) over the same index tuples -- but never
copied any `model.dual` entry from the old row objects to the new ones.
Since a Pyomo Suffix maps by object identity, the new rows started with NO
dual entry at all, so IPOPT's warm start used its own default multiplier
initialization for those 24 rows in every B/variant solve, not the
frozen ADMM-iterate value the old rows carried. This stage isolates that
confound from the scaling effect itself.

Part 1 (this script, per case, on a pristine untouched clone, no solve):
  - how many of the 24 original sess_snet_def rows (installed index only)
    have an entry in model.dual;
  - min/median/max absolute value of those entries.
(Static review of p35b_sess_snet_def_scaling_diagnostic.py already shows it
never references `model.dual`/`Suffix` at all -- so no such copy/remap was
ever performed there. This script's Part 1 output is the quantitative half
of that finding: what those un-transferred multipliers actually were.)

Part 2 -- P3.5-C, three independent fresh clones per case:
  A     -- untouched baseline (original sess_snet_def + original dual state).
  R1    -- identity replacement control: deactivate the 24 original rows,
           add one-for-one replacement rows
               1.0 * ((sch - sdch)^2 - pnet^2 - qnet^2) == 0
           and copy each original constraint dual exactly:
               dual_new = dual_old
  R100  -- scaled replacement: same construction with kappa=100.0,
               100.0 * ((sch - sdch)^2 - pnet^2 - qnet^2) == 0
           and map each original constraint dual as:
               dual_new = dual_old / 100.0

sess_comp, all bound multipliers (ipopt_zL_*/zU_*), primal values, solver
options, ADMM parameters, objective, and every other model component are
left untouched in every clone. Must be run inside the real project
Pyomo/IPOPT/MA97 environment, from the repository root:

    cd /Users/micaelsimoes/PycharmProjects/shared-resources-planning
    python p35c_sess_snet_def_dual_warmstart_diagnostic.py

Writes its own log/result/report files under
data/SRP1/Results/FrozenSMOPF/P35C/ only. No production file is modified.
"""

import os
import json
import statistics

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
from p35b_sess_snet_def_scaling_diagnostic import add_scaled_sess_snet_def  # noqa: E402  (reuse, kappa-parametrized)

OUT_ROOT = os.path.join(REPO_ROOT, 'data/SRP1/Results/FrozenSMOPF/P35C')

EXPECTED_S_RATED = 0.01
R1_KAPPA = 1.0
R100_KAPPA = 100.0


def dual_snapshot_for_rows(model, rows):
    """Read-only: model.dual entry (if any) for each original sess_snet_def row."""
    entries = []
    has_dual = hasattr(model, 'dual')
    for row in rows:
        val = None
        if has_dual:
            old_data = model.sess_snet_def[row]
            val = model.dual.get(old_data, None)
        entries.append({'row': list(row), 'dual': val})
    return has_dual, entries


def summarize_dual_entries(entries):
    present = [e['dual'] for e in entries if e['dual'] is not None]
    abs_vals = sorted(abs(v) for v in present)
    return {
        'n_rows': len(entries),
        'n_with_dual_entry': len(present),
        'n_missing_dual_entry': len(entries) - len(present),
        'abs_dual': {
            'min': abs_vals[0] if abs_vals else None,
            'median': statistics.median(abs_vals) if abs_vals else None,
            'max': abs_vals[-1] if abs_vals else None,
        },
    }


def transfer_duals(model, rows, old_component, new_component, scale):
    """Copy model.dual[old_component[row]] -> model.dual[new_component[row]] * scale,
    for every row that actually has an original entry. Rows without one are
    left untouched (nothing to copy) and reported separately."""
    transferred = []
    skipped_rows = []
    for row in rows:
        old_data = old_component[row]
        old_val = model.dual.get(old_data, None)
        if old_val is None:
            skipped_rows.append(list(row))
            continue
        new_data = new_component[row]
        new_val = old_val * scale
        model.dual[new_data] = new_val
        transferred.append({'row': list(row), 'dual_old': old_val, 'scale': scale, 'dual_new': new_val})
    return {
        'n_rows': len(rows),
        'n_transferred': len(transferred),
        'n_skipped_missing_original': len(skipped_rows),
        'skipped_rows': skipped_rows,
        'transferred': transferred,
    }


def residual_summary(states_post, s_rated, rows):
    abs_g_post = [abs(s['g']) for s in states_post]
    if not abs_g_post:
        return None
    worst_i = max(range(len(states_post)), key=lambda i: abs_g_post[i])
    return {
        'max_abs_g_final': max(abs_g_post),
        'median_abs_g_final': statistics.median(abs_g_post),
        'max_normalized_residual': max(abs_g_post) / max(s_rated ** 2, 1e-12),
        'worst_row_index_e_sm_so_p': list(rows[worst_i]),
        'worst_row_state': states_post[worst_i],
    }


def run_case(case_key):
    case = CASES[case_key]
    pkl_path = os.path.join(REPO_ROOT, case['pkl_path'])
    idx = case['shared_ess_index']

    report = {'case': case_key, 'config': case}

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

    # --- PART 1: original frozen model.dual entries for the 24 rows --------
    # Independent, untouched, throwaway clone -- read-only, no solve, no change.
    meta_probe, model_probe = load_frozen_model(pkl_path)
    model_probe = model_probe.clone()
    report['metadata'] = meta_probe

    all_indices = shared_ess_index_set(model_probe)
    capacities = {i: shared_ess_capacity(model_probe, i) for i in all_indices}
    s_rated, e_rated = capacities.get(idx, (None, None))
    report['shared_ess_capacity_check'] = {
        'expected_index': idx,
        'all_indices_and_capacity_s_e': capacities,
        's_rated_matches_expected_0_01': (s_rated is not None and abs(s_rated - EXPECTED_S_RATED) < 1e-9),
    }
    if not s_rated or s_rated <= 0:
        report['stop_reason'] = f'Installed index {idx} has non-positive s_rated={s_rated} -- STOPPING'
        return report

    rows = sess_snet_def_rows(model_probe, idx)
    has_dual, dual_entries = dual_snapshot_for_rows(model_probe, rows)
    report['part1_original_dual_state'] = {
        'model_has_dual_suffix': has_dual,
        'summary': summarize_dual_entries(dual_entries),
        'per_row_detail': dual_entries,
    }

    # --- A: untouched baseline ----------------------------------------------
    meta_a, model_a = load_frozen_model(pkl_path)
    model_a = model_a.clone()
    solve_a = solve_case(case_key, case, model_a, subdir='A_baseline')
    report['A_baseline'] = {k: v for k, v in solve_a.items() if k != 'result'}

    # --- R1: identity replacement (kappa=1.0) + exact dual copy -------------
    meta_r1, model_r1 = load_frozen_model(pkl_path)
    model_r1 = model_r1.clone()
    rows_r1 = sess_snet_def_rows(model_r1, idx)
    original_component_r1 = model_r1.sess_snet_def
    for row in rows_r1:
        model_r1.sess_snet_def[row].deactivate()
    scaled_r1 = add_scaled_sess_snet_def(model_r1, idx, rows_r1, kappa=R1_KAPPA)
    replacement_tuples_r1 = list(scaled_r1.keys())
    report['R1_construction'] = {
        'kappa': R1_KAPPA,
        'original_rows_deactivated': len(rows_r1),
        'replacement_rows_added': len(replacement_tuples_r1),
        'index_tuples_match_one_to_one': sorted(replacement_tuples_r1) == sorted(rows_r1),
    }
    # R1 is the identity-replacement control: dual_new = dual_old (scale=1.0).
    report['R1_dual_transfer'] = transfer_duals(model_r1, rows_r1, original_component_r1, scaled_r1, scale=1.0)
    solve_r1 = solve_case(case_key, case, model_r1, subdir='R1_identity_dualcopy')
    report['R1'] = {k: v for k, v in solve_r1.items() if k != 'result'}
    states_post_r1 = [row_state(model_r1, *row) for row in rows_r1]
    report['R1_final_original_unscaled_residual'] = residual_summary(states_post_r1, s_rated, rows_r1)

    # --- R100: scaled replacement (kappa=100.0) + scaled dual copy ----------
    meta_r100, model_r100 = load_frozen_model(pkl_path)
    model_r100 = model_r100.clone()
    rows_r100 = sess_snet_def_rows(model_r100, idx)
    original_component_r100 = model_r100.sess_snet_def
    for row in rows_r100:
        model_r100.sess_snet_def[row].deactivate()
    scaled_r100 = add_scaled_sess_snet_def(model_r100, idx, rows_r100, kappa=R100_KAPPA)
    replacement_tuples_r100 = list(scaled_r100.keys())
    report['R100_construction'] = {
        'kappa': R100_KAPPA,
        'kappa_matches_1_over_s_rated': abs(R100_KAPPA - (1.0 / s_rated)) < 1e-6,
        'original_rows_deactivated': len(rows_r100),
        'replacement_rows_added': len(replacement_tuples_r100),
        'index_tuples_match_one_to_one': sorted(replacement_tuples_r100) == sorted(rows_r100),
    }
    report['R100_dual_transfer'] = transfer_duals(model_r100, rows_r100, original_component_r100, scaled_r100, scale=1.0 / R100_KAPPA)
    solve_r100 = solve_case(case_key, case, model_r100, subdir='R100_scaled_dualcopy')
    report['R100'] = {k: v for k, v in solve_r100.items() if k != 'result'}
    states_post_r100 = [row_state(model_r100, *row) for row in rows_r100]
    report['R100_final_original_unscaled_residual'] = residual_summary(states_post_r100, s_rated, rows_r100)

    # --- Critical expected pattern -------------------------------------------
    report['pattern_check'] = {
        'A_failed': not solve_a['succeeded'],
        'R1_failed': not solve_r1['succeeded'],
        'R100_succeeded': solve_r100['succeeded'],
        'R100_succeeded_without_recovery': solve_r100['succeeded'] and len(solve_r100['log_files']) <= 1,
        'matches_expected_A_FAIL_R1_FAIL_R100_PRIMARY_SUCCESS': (
            (not solve_a['succeeded'])
            and (not solve_r1['succeeded'])
            and solve_r100['succeeded']
            and len(solve_r100['log_files']) <= 1
        ),
    }

    return report


def main():
    full_report = {}
    for case_key in ('DSO', 'TSO'):
        full_report[case_key] = run_case(case_key)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, 'p35c_report.json')
    with open(out_path, 'w') as handle:
        json.dump(full_report, handle, indent=2, default=str)
    print(f'[P3.5-C] Wrote full report to {out_path}')
    print(json.dumps(full_report, indent=2, default=str))


if __name__ == '__main__':
    main()
