"""
Stage P5.4-D2.10 -- locate operating regimes where the shared-ESS power capacity
S and the energy capacity E are actually BINDING, and repeat the sensitivity
decomposition and the sensitivity-clean A/B there.

The positive-bootstrap population is not sufficient on its own: across the eight
cases audited in `p54d2_sensitivity_clean_ab.py` only one has S binding and none
has E binding, so the plan's D2.10 requirement cannot be met from it. Rather
than report that requirement as satisfied, this script scans capacity scales --
still the production formulation and the production solve path, only a different
candidate capacity -- until a binding case is found, and reports honestly if
none is.

    python p54d2_binding_regime_scan.py
"""

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p54d2_sensitivity_clean_ab import (binding_report, max_abs_diff,  # noqa: E402
                                        physical_profile, solve_variant)
from p54d2_sensitivity_root_cause import (active_set_signature, build_case,  # noqa: E402
                                          objective_of, sensitivity_decomposition)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2')

S_SCALES = (1.0, 5.0, 20.0, 100.0)
E_SCALES = (1.0, 0.5, 0.2, 0.05, 0.02)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def probe(case, s_scale, e_scale):
    s = case['s_pu'] * s_scale
    e = case['e_pu'] * e_scale
    mA, _rA, okA, _i, _l = solve_variant(case['network'], case['params'], case['e'],
                                         s, e, clean=False)
    if not okA:
        return {'s_scale': s_scale, 'e_scale': e_scale, 'solved': False}
    b = binding_report(mA, case['e'])
    decS = sensitivity_decomposition(mA, case['e'], 'S')
    decE = sensitivity_decomposition(mA, case['e'], 'E')
    return {'s_scale': s_scale, 'e_scale': e_scale, 'solved': True,
            'objective': objective_of(mA), 'binding': b,
            'S_fixing_dual': decS['fixing_row_dual'],
            'S_bound_contribution': decS['bound_contribution'],
            'S_corrected': decS['corrected_total_derivative'],
            'S_fixing_share': decS['fixing_row_share_of_total'],
            'E_fixing_dual': decE['fixing_row_dual'],
            'E_bound_contribution': decE['bound_contribution']}


def ab_at(case, s, e):
    mA, _rA, okA, _i, _l = solve_variant(case['network'], case['params'], case['e'],
                                         s, e, clean=False)
    mB, _rB, okB, iB, _l = solve_variant(case['network'], case['params'], case['e'],
                                         s, e, clean=True)
    if not (okA and okB):
        return {'A_solved': okA, 'B_solved': okB}
    qa, qb = objective_of(mA), objective_of(mB)
    pa, pb = physical_profile(mA, case['e']), physical_profile(mB, case['e'])
    decA = sensitivity_decomposition(mA, case['e'], 'S')
    decB = sensitivity_decomposition(mB, case['e'], 'S')
    return {
        'A_solved': True, 'B_solved': True, 's_pu': s, 'e_pu': e,
        'objective_A': qa, 'objective_B': qb,
        'objective_rel_diff': abs(qa - qb) / max(abs(qa), 1e-30),
        'max_abs_diff': {k: max_abs_diff(pa.get(k), pb.get(k)) for k in pa},
        'same_ess_active_set': (
            json.dumps(active_set_signature(mA, case['e']), sort_keys=True)
            == json.dumps(active_set_signature(mB, case['e']), sort_keys=True)),
        'S_decomposition_A': {k: decA[k] for k in (
            'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative',
            'fixing_row_share_of_total')},
        'S_decomposition_B': {k: decB[k] for k in (
            'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative')},
        'B_bound_contribution_is_zero': abs(decB['bound_contribution']) < 1e-12,
        'binding_A': binding_report(mA, case['e']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='dso:9:2030:Winter,tso:0:2025:Winter')
    parser.add_argument('--out', default='p54d2_binding_scan.json')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D2.10 binding-regime scan', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              's_scales': list(S_SCALES), 'e_scales': list(E_SCALES),
              'note': ('production formulation and production solve path; only the '
                       'candidate capacity differs from the bootstrap value')}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    results = {}
    for spec in args.cases.split(','):
        agent, ident, year, day = spec.split(':')
        case = build_case(planning, candidate, agent, ident, year, day)
        print(f"\n[D2.10] {case['tag']}  S0={case['s_pu']:.4e} E0={case['e_pu']:.4e}",
              flush=True)
        grid = []
        for ss in S_SCALES:
            for es in E_SCALES:
                rec = probe(case, ss, es)
                grid.append(rec)
                if not rec['solved']:
                    print(f"    S x{ss:<6g} E x{es:<6g}  solve failed")
                    continue
                b = rec['binding']
                print(f"    S x{ss:<6g} E x{es:<6g}  S_binding={str(b['S_binding']):5s} "
                      f"E_binding={str(b['E_binding']):5s}  "
                      f"sum_slack={b['min_active_sum_slack']:+.3e} "
                      f"soc_up_slack={b['soc_upper_slack']:+.3e}  "
                      f"fix={rec['S_fixing_dual']:+.4e} bound={rec['S_bound_contribution']:+.4e}")
        entry = {'grid': grid,
                 's_binding_points': [g for g in grid if g.get('solved') and g['binding']['S_binding']],
                 'e_binding_points': [g for g in grid if g.get('solved') and g['binding']['E_binding']]}

        # repeat the A/B at one S-binding and one E-binding point, if found
        for label, points in (('S_binding', entry['s_binding_points']),
                              ('E_binding', entry['e_binding_points'])):
            if not points:
                entry[f'ab_at_{label}'] = {'found': False}
                print(f"    no {label} point found in the scanned grid")
                continue
            g = points[0]
            ab = ab_at(case, case['s_pu'] * g['s_scale'], case['e_pu'] * g['e_scale'])
            ab['found'] = True
            ab['s_scale'] = g['s_scale']
            ab['e_scale'] = g['e_scale']
            entry[f'ab_at_{label}'] = ab
            if ab.get('A_solved') and ab.get('B_solved'):
                print(f"    A/B at {label} (S x{g['s_scale']:g}, E x{g['e_scale']:g}): "
                      f"obj_rel_diff={ab['objective_rel_diff']:.3e} "
                      f"same_set={ab['same_ess_active_set']} "
                      f"A_bound={ab['S_decomposition_A']['bound_contribution']:+.4e} "
                      f"B_bound={ab['S_decomposition_B']['bound_contribution']:+.4e}")
        results[case['tag']] = entry

    report['cases'] = results
    report['summary'] = {
        tag: {'n_S_binding': len(v['s_binding_points']),
              'n_E_binding': len(v['e_binding_points'])}
        for tag, v in results.items()}
    out = os.path.join(OUT_DIR, args.out)
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f"\n[D2.10] summary: {json.dumps(report['summary'], indent=1)}")
    print(f'[D2.10] report -> {out}')


if __name__ == '__main__':
    main()
