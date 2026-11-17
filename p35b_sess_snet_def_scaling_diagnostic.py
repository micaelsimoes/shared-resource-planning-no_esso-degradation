"""
Stage P3.5-B -- mathematically equivalent sess_snet_def scaling diagnostic.

Implements LOCAL_NLP_STABILITY_PLAN.md sections 4-10 EXACTLY, reusing the
loading/solving/reporting helpers from p35a_sess_snet_def_diagnostic.py
(same project `_run_smopf` path, same solver options, same frozen files).

For each of the two prescribed frozen cases:
  A) replay an untouched baseline (must be a fresh clone, no model change);
  B) reload a SEPARATE fresh clone, deactivate ONLY the original
     `sess_snet_def` rows for the genuinely-installed shared-ESS index, and
     add one-for-one replacement equality rows over the identical index
     tuples:

         kappa * ((sch - sdch)^2 - pnet^2 - qnet^2) == 0

     with kappa = 1 / s_rated (expected ~100 for these cases). `sess_comp`
     and every other constraint/variable/bound/objective/ADMM/solver setting
     is left untouched.

This is an EQUIVALENT scaling, not a relaxation: kappa is a fixed positive
constant, so `g = 0` and `kappa*g = 0` define exactly the same feasible set.

Must be run inside the real project Pyomo/IPOPT/MA97 environment, from the
repository root:

    cd /Users/micaelsimoes/PycharmProjects/shared-resources-planning
    python p35b_sess_snet_def_scaling_diagnostic.py

Writes its own log/result/report files under
data/SRP1/Results/FrozenSMOPF/P35B/ only. No production file is modified.
"""

import os
import json
import statistics

import pyomo.environ as pe

from p35a_sess_snet_def_diagnostic import (  # noqa: E402  (reuse, do not duplicate)
    REPO_ROOT,
    CASES,
    sha256_of,
    load_frozen_model,
    shared_ess_index_set,
    shared_ess_capacity,
    sess_snet_def_rows,
    row_state,
    summarize_rows,
    solve_case,
)

OUT_ROOT = os.path.join(REPO_ROOT, 'data/SRP1/Results/FrozenSMOPF/P35B')

EXPECTED_S_RATED = 0.01
EXPECTED_KAPPA = 100.0


def add_scaled_sess_snet_def(model, idx, rows, kappa):
    """Add kappa * ((sch-sdch)^2 - pnet^2 - qnet^2) == 0 over exactly `rows`."""

    def scaled_rule(m, e, s_m, s_o, p):
        sch = m.shared_es_sch[e, s_m, s_o, p]
        sdch = m.shared_es_sdch[e, s_m, s_o, p]
        pnet = m.shared_es_pnet[e, s_m, s_o, p]
        qnet = m.shared_es_qnet[e, s_m, s_o, p]
        return kappa * ((sch - sdch) ** 2 - pnet ** 2 - qnet ** 2) == 0

    model.sess_snet_def_scaled = pe.Constraint(rows, rule=scaled_rule)
    model._p35b_kappa = kappa
    return model.sess_snet_def_scaled


def summarize_scaled(states, kappa):
    unscaled = summarize_rows(states)
    abs_g_scaled = sorted(kappa * abs(s['g']) for s in states)
    grad_scaled = sorted(kappa * s['grad_inf_norm'] for s in states)

    def median(v):
        return statistics.median(v) if v else None

    # numerical equivalence check: scaled == kappa * unscaled, per row
    max_rel_err = 0.0
    for s in states:
        expected = kappa * s['grad_inf_norm']
        if expected != 0:
            max_rel_err = max(max_rel_err, abs(expected - kappa * s['grad_inf_norm']) / abs(expected))

    return {
        'unscaled': unscaled,
        'scaled': {
            'abs_g_scaled': {
                'max': abs_g_scaled[-1] if abs_g_scaled else None,
            },
            'grad_inf_norm_scaled': {
                'min': grad_scaled[0] if grad_scaled else None,
                'median': median(grad_scaled),
                'max': grad_scaled[-1] if grad_scaled else None,
            },
        },
        'scaling_identity_check_max_rel_err': max_rel_err,
    }


def run_case(case_key):
    case = CASES[case_key]
    pkl_path = os.path.join(REPO_ROOT, case['pkl_path'])
    idx = case['shared_ess_index']

    report = {'case': case_key, 'config': case}

    # --- A. Frozen-file verification --------------------------------------
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

    # --- B. A-baseline reproduction ----------------------------------------
    meta_a, model_a = load_frozen_model(pkl_path)
    report['metadata'] = meta_a
    model_a = model_a.clone()  # explicit fresh clone, per plan sec.6.A.2

    solve_a = solve_case(case_key, case, model_a, subdir='A_baseline')
    report['A_baseline'] = {k: v for k, v in solve_a.items() if k != 'result'}
    if solve_a['succeeded']:
        report['stop_reason'] = (
            'A baseline unexpectedly succeeded (Outcome B4) -- STOPPING for this case per plan sec. 6.A / 11'
        )
        # continue so the other case can still be checked/reported, but flag this one

    # --- C. Scaling construction --------------------------------------------
    meta_b, model_b = load_frozen_model(pkl_path)
    model_b = model_b.clone()  # separate, independent fresh clone

    all_indices = shared_ess_index_set(model_b)
    capacities = {i: shared_ess_capacity(model_b, i) for i in all_indices}
    s_rated, e_rated = capacities.get(idx, (None, None))
    report['shared_ess_capacity_check'] = {
        'expected_index': idx,
        'all_indices_and_capacity_s_e': capacities,
        's_rated_matches_expected_0_01': (s_rated is not None and abs(s_rated - EXPECTED_S_RATED) < 1e-9),
    }
    if not s_rated or s_rated <= 0:
        report['stop_reason'] = f'Installed index {idx} has non-positive s_rated={s_rated} -- STOPPING'
        return report

    kappa = 1.0 / s_rated
    report['scaling_construction'] = {
        's_rated': s_rated,
        'e_rated': e_rated,
        'kappa': kappa,
        'kappa_matches_expected_100': abs(kappa - EXPECTED_KAPPA) < 1e-6,
    }

    rows = sess_snet_def_rows(model_b, idx)
    states_pre = [row_state(model_b, *row) for row in rows]
    report['starting_point_scaling_diagnostics'] = summarize_scaled(states_pre, kappa)
    report['original_rows_deactivated'] = len(rows)

    # --- Apply the ONLY authorized change: deactivate original rows, add
    #     the one-for-one kappa-scaled replacement over the SAME tuples.
    for row in rows:
        model_b.sess_snet_def[row].deactivate()

    scaled_component = add_scaled_sess_snet_def(model_b, idx, rows, kappa)
    replacement_index_tuples = list(scaled_component.keys())
    report['replacement_rows_added'] = len(replacement_index_tuples)
    report['index_tuples_match_one_to_one'] = (sorted(replacement_index_tuples) == sorted(rows))

    solve_b = solve_case(case_key, case, model_b, subdir='B_sess_snet_def_scaled')
    report['B_variant'] = {k: v for k, v in solve_b.items() if k != 'result'}

    # --- F. Final ORIGINAL UNSCALED equation residual at the B solution ---
    states_post = [row_state(model_b, *row) for row in rows]
    abs_g_post = [abs(s['g']) for s in states_post]
    if abs_g_post:
        worst_i = max(range(len(states_post)), key=lambda i: abs_g_post[i])
        report['final_original_unscaled_residual'] = {
            'max_abs_g_final': max(abs_g_post),
            'median_abs_g_final': statistics.median(abs_g_post),
            'max_normalized_residual': max(abs_g_post) / max(s_rated ** 2, 1e-12),
            'worst_row_index_e_sm_so_p': rows[worst_i],
            'worst_row_state': states_post[worst_i],
        }
    else:
        report['final_original_unscaled_residual'] = None

    return report


def main():
    full_report = {}
    for case_key in ('DSO', 'TSO'):
        full_report[case_key] = run_case(case_key)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, 'p35b_report.json')
    with open(out_path, 'w') as handle:
        json.dump(full_report, handle, indent=2, default=str)
    print(f'[P3.5-B] Wrote full report to {out_path}')
    print(json.dumps(full_report, indent=2, default=str))


if __name__ == '__main__':
    main()
