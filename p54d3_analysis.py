"""
Stage P5.4-D3.6-D3.9 -- combine the D3 candidate groups and classify.

Derives `tol_cut` from measured quantities rather than picking a number:

  tol_repeat  identical-candidate repeatability of the distributed recourse
  tol_admm    |objective_change_rel| at the cycle where ADMM declared
              convergence, times |Q0| -- i.e. how far the recourse was still
              moving when the run stopped, which is the precision to which the
              converged recourse is actually determined
  tol_cut     = max(tol_repeat, tol_admm)

Then evaluates:
  D3.6  cut_gap = Q_best_observed - L(x);  cut_gap < -tol_cut is decisive
  D3.7  local linearity, split by same-branch / different-branch
  D3.8  lower envelope of observed recourse over each S sweep
  D3.9  classification

    python p54d3_analysis.py
"""

import glob
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D3')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def load_groups():
    base, groups = None, {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'p54d3_*.json'))):
        with open(path) as h:
            data = json.load(h)
        name = data.get('group')
        if name == 'base' and base is None:
            base = data
        if data.get('D3_candidates'):
            groups[name] = data
        if base is None and 'D3_1_base' in data:
            base = data
    return base, groups


def main():
    base, groups = load_groups()
    if base is None:
        print('[D3] no base report found')
        return

    q0 = base['D3_1_base']['Q0']
    rep = base.get('D3_6_repeatability', {})
    tol_repeat = rep.get('absolute_spread', 0.0) or 0.0

    # the ADMM's own stopping drift, read from the base run's final cycle
    final_rel_change = None
    for path in glob.glob(os.path.join(OUT_DIR, 'p54f_report.json')):
        pass
    admm_path = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54F', 'p54f_report.json')
    if os.path.exists(admm_path):
        with open(admm_path) as h:
            f = json.load(h)
        diags = f.get('state_metrics', {}).get('admm_diagnostics', [])
        if diags:
            final_rel_change = abs(diags[-1].get('objective_change_rel') or 0.0)
    if final_rel_change is None:
        final_rel_change = 1e-3        # the configured objective_relative_tolerance
    tol_admm = final_rel_change * abs(q0)
    tol_cut = max(tol_repeat, tol_admm)

    report = {
        'stage': 'P5.4-D3.6-D3.9', 'git_head': git_head(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'Q0': q0,
        'tol_derivation': {
            'tol_repeat_identical_candidate': tol_repeat,
            'admm_final_objective_change_rel': final_rel_change,
            'tol_admm_absolute': tol_admm,
            'tol_cut': tol_cut,
            'tol_cut_relative': tol_cut / abs(q0),
            'basis': ('tol_repeat measured by re-solving the identical candidate; '
                      'tol_admm is the residual relative objective drift at the cycle '
                      'where ADMM declared convergence, times |Q0| -- the precision to '
                      'which the converged recourse is actually determined'),
        },
    }

    rows = []
    for name, data in groups.items():
        for c in data['D3_candidates']:
            rows.append({**c, 'group': name})

    # scale comparison: is the cut's predicted effect even resolvable?
    predicted = [abs(r['predicted_delta_g0T_dx']) for r in rows
                 if r.get('predicted_delta_g0T_dx') is not None]
    report['resolvability'] = {
        'n_candidates': len(rows),
        'max_abs_predicted_delta': max(predicted) if predicted else None,
        'median_abs_predicted_delta': statistics.median(predicted) if predicted else None,
        'tol_cut': tol_cut,
        'max_predicted_over_tol_cut': (max(predicted) / tol_cut) if predicted and tol_cut else None,
        'n_predictions_below_tol_cut': sum(1 for p in predicted if p < tol_cut),
        'note': ('a predicted effect smaller than tol_cut cannot be confirmed against '
                 'the observed recourse, whatever the sign of cut_gap'),
    }

    # D3.6 cut safety
    evaluated = [r for r in rows if r.get('cut_gap') is not None]
    violations = [r for r in evaluated if r['cut_gap'] < -tol_cut]
    marginal = [r for r in evaluated if -tol_cut <= r['cut_gap'] < 0]
    report['D3_6_cut_safety'] = {
        'n_evaluated': len(evaluated),
        'n_decisive_violations': len(violations),
        'n_negative_within_tolerance': len(marginal),
        'min_cut_gap': min((r['cut_gap'] for r in evaluated), default=None),
        'min_cut_gap_relative': (min((r['cut_gap'] for r in evaluated), default=0.0)
                                 / abs(q0)) if evaluated else None,
        'violations': [{k: r[k] for k in ('label', 'cut_gap', 'Q_best_observed', 'L_x',
                                          'observed_delta_Q', 'predicted_delta_g0T_dx')}
                       for r in sorted(violations, key=lambda r: r['cut_gap'])[:20]],
        'all_gaps': [{'label': r['label'], 'cut_gap': r['cut_gap'],
                      'cut_gap_relative': r['cut_gap'] / abs(q0)} for r in evaluated],
    }

    # D3.7 local linearity, split by branch
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

    report['D3_7_linearity'] = {
        'same_branch_as_base': lin(same),
        'different_branch': lin(diff),
        'n_distinct_branches_observed': len({r['branch_A'] for r in rows
                                             if r.get('branch_A') is not None}),
        'detail': [{k: r.get(k) for k in ('label', 'observed_delta_Q',
                                          'predicted_delta_g0T_dx', 'linearity_abs_error',
                                          'linearity_rel_error', 'same_branch_as_base',
                                          'branch_A', 'branch_B')} for r in evaluated],
    }

    # D3.8 S monotonicity of the observed lower envelope
    s_rows = sorted([r for r in rows if r.get('kind') == 's' and r.get('Q_best_observed')],
                    key=lambda r: (r['node'], r['rel']))
    envelopes = {}
    for r in s_rows:
        envelopes.setdefault(r['node'], []).append((r['rel'], r['Q_best_observed']))
    mono = {}
    for node, pts in envelopes.items():
        pts = sorted(pts)
        seq = [q for _, q in pts]
        non_increasing = all(seq[i + 1] <= seq[i] + tol_cut for i in range(len(seq) - 1))
        mono[str(node)] = {
            'points': [{'rel': a, 'Q_best': b} for a, b in pts],
            'lower_envelope_non_increasing_within_tol': non_increasing,
            'range': (max(seq) - min(seq)) if seq else None,
        }
    report['D3_8_S_monotonicity'] = {
        'per_node': mono,
        'all_non_increasing': all(v['lower_envelope_non_increasing_within_tol']
                                  for v in mono.values()) if mono else None,
        'note': ('increasing S enlarges the feasible set, so the best observed lower '
                 'envelope should not systematically increase; if it does, the lower '
                 'branch has not been reliably recovered'),
    }

    out = os.path.join(OUT_DIR, 'p54d3_analysis.json')
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)

    print(f"[D3] Q0 = {q0:.6f}")
    t = report['tol_derivation']
    print(f"[D3] tol_cut = {t['tol_cut']:.6e} ({t['tol_cut_relative']:.3e} relative)")
    print(f"     tol_repeat={t['tol_repeat_identical_candidate']:.3e}  "
          f"admm_final_rel_change={t['admm_final_objective_change_rel']:.3e}")
    r = report['resolvability']
    print(f"[D3] resolvability: max|predicted| = {r['max_abs_predicted_delta']:.4e}, "
          f"{r['max_predicted_over_tol_cut']:.4f} x tol_cut; "
          f"{r['n_predictions_below_tol_cut']}/{r['n_candidates']} predictions below tol_cut")
    s = report['D3_6_cut_safety']
    print(f"[D3.6] evaluated={s['n_evaluated']} decisive_violations={s['n_decisive_violations']} "
          f"negative_within_tol={s['n_negative_within_tolerance']} "
          f"min_gap={s['min_cut_gap']:.4e} ({s['min_cut_gap_relative']:.3e} rel)")
    for v in s['violations'][:10]:
        print(f"    VIOLATION {v['label']:26s} gap={v['cut_gap']:+.4e} "
              f"Q_best={v['Q_best_observed']:.4f} L={v['L_x']:.4f}")
    l = report['D3_7_linearity']
    print(f"[D3.7] same-branch: {l['same_branch_as_base']}")
    print(f"       different-branch: {l['different_branch']}")
    print(f"       distinct branches observed: {l['n_distinct_branches_observed']}")
    print(f"[D3.8] S lower envelope non-increasing: {report['D3_8_S_monotonicity']['all_non_increasing']}")
    for node, v in report['D3_8_S_monotonicity']['per_node'].items():
        print(f"    node {node}: {v['lower_envelope_non_increasing_within_tol']} "
              f"range={v['range']:.4e}")
    print(f'\n[D3] analysis -> {out}')


if __name__ == '__main__':
    main()
