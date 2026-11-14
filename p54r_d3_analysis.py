"""
Stage P5.4-R2 analysis -- canonical tol_cut, cut safety, and the like-for-like
comparison against the noncanonical srp_env run.

tol_cut is derived, not chosen:
  tol_repeat  identical-candidate repeatability of the canonical recourse
  tol_admm    |objective_change_rel| at the cycle where ADMM declared convergence,
              times |Q0| -- the precision to which the converged recourse is
              actually determined
  tol_cut     = max(tol_repeat, tol_admm)

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p54r_d3_analysis.py
"""

import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54R_D3')
ADMM_REPORT = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54F', 'p54f_report.json')

# noncanonical srp_env figures, retained for the side-by-side (do not recompute)
NONCANONICAL = {
    'checksum': '4d948b9b8d8d05f06270dfc8357cc9c9706e7bddb48a1fc74c6842fda07cb0a1',
    'Q0': 848258809.8141167,
    'cycles': 9,
    'tol_cut': 3.046e5,
    'worst_cut_gap': -11476300.297777627,
    'n_decisive_violations': 6,
    'n_evaluated': 8,
}


def main():
    groups = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'p54r_d3_*.json'))):
        with open(path) as handle:
            data = json.load(handle)
        groups[data['group']] = data
    if not groups:
        print('[R2] no canonical group reports found')
        return

    base = next(iter(groups.values()))
    q0 = base['D3_1_base']['Q0']
    checksum = base['provenance']['scenario_checksum']

    # every group re-ran the base independently -> determinism evidence
    base_values = {g: d['D3_1_base']['Q0'] for g, d in groups.items()}
    base_cycles = {g: d['D3_1_base']['n_cycles'] for g, d in groups.items()}
    repeats = {g: d['D3_6_repeatability'] for g, d in groups.items()}
    tol_repeat = max(r['absolute_spread'] for r in repeats.values())
    cross_group_spread = max(base_values.values()) - min(base_values.values())

    # ADMM stopping drift from the canonical distributed run
    admm_drift, admm_q0 = None, None
    if os.path.exists(ADMM_REPORT):
        with open(ADMM_REPORT) as handle:
            f = json.load(handle)
        diags = f.get('state_metrics', {}).get('admm_diagnostics', [])
        if diags:
            admm_drift = abs(diags[-1].get('objective_change_rel') or 0.0)
            admm_q0 = diags[-1].get('recourse')
    consistent = (admm_q0 is not None and abs(admm_q0 - q0) < 1e-6)
    if admm_drift is None:
        admm_drift = 1e-3       # configured objective_relative_tolerance
    tol_admm = admm_drift * abs(q0)
    tol_cut = max(tol_repeat, cross_group_spread, tol_admm)

    rows = []
    for g, d in groups.items():
        for c in d.get('D3_candidates', []):
            rows.append({**c, 'group': g})

    evaluated = [r for r in rows if r.get('cut_gap') is not None]
    violations = [r for r in evaluated if r['cut_gap'] < -tol_cut]
    marginal = [r for r in evaluated if -tol_cut <= r['cut_gap'] < 0]
    predicted = [abs(r['predicted_delta_g0T_dx']) for r in rows]

    same = [r for r in evaluated if r.get('same_branch_as_base')]
    diff = [r for r in evaluated if r.get('same_branch_as_base') is False]

    def lin(group):
        if not group:
            return {'n': 0}
        errs = [r['linearity_abs_error'] for r in group if r.get('linearity_abs_error') is not None]
        rels = [r['linearity_rel_error'] for r in group if r.get('linearity_rel_error') is not None]
        return {'n': len(group),
                'max_abs_error': max(errs) if errs else None,
                'median_abs_error': statistics.median(errs) if errs else None,
                'max_rel_error': max(rels) if rels else None,
                'median_rel_error': statistics.median(rels) if rels else None}

    g0 = base['D3_2_cut']['g0']
    report = {
        'stage': 'P5.4-R2 analysis',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'environment': 'canonical opf_env_py311',
        'scenario_checksum': checksum,
        'Q0': q0,
        'base_run': base['D3_1_base'],
        'determinism': {
            'base_Q0_per_group': base_values,
            'base_cycles_per_group': base_cycles,
            'cross_group_spread': cross_group_spread,
            'repeat_spread_per_group': {g: r['absolute_spread'] for g, r in repeats.items()},
        },
        'tol_derivation': {
            'tol_repeat_identical_candidate': tol_repeat,
            'cross_group_base_spread': cross_group_spread,
            'admm_final_objective_change_rel': admm_drift,
            'admm_report_Q0_matches': consistent,
            'tol_admm_absolute': tol_admm,
            'tol_cut': tol_cut,
            'tol_cut_relative': tol_cut / abs(q0),
        },
        'g0': g0,
        'g0_sign_summary': base['D3_2_cut']['g0_sign_summary'],
        'resolvability': {
            'n_candidates': len(rows),
            'max_abs_predicted_delta': max(predicted) if predicted else None,
            'max_predicted_over_tol_cut': (max(predicted) / tol_cut) if predicted else None,
            'n_predictions_below_tol_cut': sum(1 for p in predicted if p < tol_cut),
        },
        'D3_6_cut_safety': {
            'n_evaluated': len(evaluated),
            'n_decisive_violations': len(violations),
            'n_negative_within_tolerance': len(marginal),
            'n_negative_total': sum(1 for r in evaluated if r['cut_gap'] < 0),
            'min_cut_gap': min((r['cut_gap'] for r in evaluated), default=None),
            'min_cut_gap_relative': (min((r['cut_gap'] for r in evaluated), default=0.0)
                                     / abs(q0)) if evaluated else None,
            'all_gaps': [{'label': r['label'], 'Q_best_observed': r['Q_best_observed'],
                          'L_x': r['L_x'], 'predicted': r['predicted_delta_g0T_dx'],
                          'observed_delta_Q': r['observed_delta_Q'],
                          'cut_gap': r['cut_gap'],
                          'decisive': r['cut_gap'] < -tol_cut}
                         for r in sorted(evaluated, key=lambda r: r['cut_gap'])],
        },
        'D3_7_linearity': {
            'same_branch_as_base': lin(same),
            'different_branch': lin(diff),
            'n_distinct_branches': len({r['branch_A'] for r in rows if r.get('branch_A')}),
        },
        'side_by_side': {
            'canonical_opf_env_py311': {
                'checksum': checksum, 'Q0': q0,
                'cycles': base['D3_1_base']['n_cycles'],
                'tol_cut': tol_cut,
                'worst_cut_gap': min((r['cut_gap'] for r in evaluated), default=None),
                'n_decisive_violations': len(violations),
                'n_evaluated': len(evaluated),
            },
            'noncanonical_srp_env': NONCANONICAL,
        },
    }

    out = os.path.join(OUT_DIR, 'p54r_d3_analysis.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    t = report['tol_derivation']
    print(f'[R2] canonical Q0 = {q0:.6f}   checksum {checksum[:16]}...')
    print(f"[R2] determinism: base Q0 per group {base_values}")
    print(f"     cross-group spread {cross_group_spread:.3e}; "
          f"identical-candidate repeat spread {tol_repeat:.3e}")
    print(f"[R2] tol_cut = {t['tol_cut']:.6e} ({t['tol_cut_relative']:.3e} relative)")
    print(f"     admm_final_rel_change={t['admm_final_objective_change_rel']:.3e} "
          f"(ADMM report Q0 matches: {t['admm_report_Q0_matches']})")
    r = report['resolvability']
    print(f"[R2] resolvability: max|predicted| = {r['max_abs_predicted_delta']:.4e} "
          f"= {r['max_predicted_over_tol_cut']:.4f} x tol_cut; "
          f"{r['n_predictions_below_tol_cut']}/{r['n_candidates']} below tol_cut")
    s = report['D3_6_cut_safety']
    print(f"\n[R2/D3.6] evaluated={s['n_evaluated']} negative={s['n_negative_total']} "
          f"decisive={s['n_decisive_violations']} within_tol={s['n_negative_within_tolerance']}")
    print(f"          min cut_gap = {s['min_cut_gap']:+.4e} ({s['min_cut_gap_relative']:+.3e} rel)")
    print(f"\n{'candidate':26s} {'Q_best':>18} {'L(x)':>18} {'predicted':>12} {'cut_gap':>14}  decisive")
    for gp in s['all_gaps']:
        print(f"{gp['label']:26s} {gp['Q_best_observed']:>18.2f} {gp['L_x']:>18.2f} "
              f"{gp['predicted']:>12.2f} {gp['cut_gap']:>+14.2f}  {gp['decisive']}")
    l = report['D3_7_linearity']
    print(f"\n[R2/D3.7] same-branch {l['same_branch_as_base']}")
    print(f"          diff-branch {l['different_branch']}")
    print(f"          distinct branches: {l['n_distinct_branches']}")
    sb = report['side_by_side']
    print('\n[R2] SIDE BY SIDE')
    print(f"{'':26s} {'canonical opf_env_py311':>28} {'noncanonical srp_env':>26}")
    for key in ('checksum', 'Q0', 'cycles', 'tol_cut', 'worst_cut_gap',
                'n_decisive_violations'):
        a, b = sb['canonical_opf_env_py311'][key], sb['noncanonical_srp_env'][key]
        fa = a[:16] + '...' if isinstance(a, str) else (f'{a:.6f}' if isinstance(a, float) else a)
        fb = b[:16] + '...' if isinstance(b, str) else (f'{b:.6f}' if isinstance(b, float) else b)
        print(f'{key:26s} {str(fa):>28} {str(fb):>26}')
    print(f'\n[R2] analysis -> {out}')


if __name__ == '__main__':
    main()
