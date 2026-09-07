"""
Stage P5.4-E -- production validation of the active-energy shared-ESS formulation.

Runs the complete positive-bootstrap initialization using the ACTUAL production
implementation (no diagnostic wrapper), repeats the targeted derivative/rank
audit, and audits normalized physical residuals of the two new nonlinear
inequalities.

    python p54e_production_validation.py
"""

import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from definitions import ESS_COMPLEMENTARITY_TOLERANCE, SHARED_ESS_ZERO_CAPACITY_TOLERANCE  # noqa: E402
from p53a2_jacobian_correction import analyse_model, singular_spectrum  # noqa: E402
from p53b3_active_power_ess import jacobian_for, run_branch  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54E')
JAC_TAGS = {'case33_1/2030/Winter', 'case9/2025/Winter'}


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-E', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'note': 'production implementation, no diagnostic wrapper'}

    print('[P5.4-E] running production positive-bootstrap initialization ...', flush=True)
    res = run_branch(prototype=False, capture_tags=JAC_TAGS)
    agg = res['aggregates']
    report['aggregates'] = agg
    report['per_solve'] = res['per_solve']
    print(f"    {agg['per_agent']} persistent={agg['persistent_failures']} "
          f"{agg['persistent_failure_ids']}", flush=True)

    jac = {}
    for tag in sorted(JAC_TAGS):
        if tag in res['captured']:
            print(f'[P5.4-E] jacobian {tag} ...', flush=True)
            spec = jacobian_for(res['captured'][tag], tag)
            comps = {c for c in spec['zero_row_components']}
            spec['sess_snet_def_component_exists'] = hasattr(
                res['captured'][tag], 'sess_snet_def')
            spec['new_ess_rows_introduce_zero_gradient_equalities'] = any(
                c.startswith('sess_') for c in comps)
            jac[tag] = spec
    report['jacobian'] = jac

    # ---- normalized physical residual audit over every active shared-ESS row ----
    cap_res, comp_res, worst = [], [], {'capability': None, 'complementarity': None}
    for tag, rec in res['physics'].items():
        s = rec['s_rated']
        if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
            continue
        for i in range(len(rec['pnet'])):
            pnet, qnet = rec['pnet'][i], rec['qnet'][i]
            pch, pdch = rec['pch'][i], rec['pdch'][i]
            c = max((pnet ** 2 + qnet ** 2 - s ** 2) / s ** 2, 0.0)
            k = max((pch * pdch - ESS_COMPLEMENTARITY_TOLERANCE * s ** 2) / s ** 2, 0.0)
            cap_res.append(c); comp_res.append(k)
            if worst['capability'] is None or c > worst['capability'][0]:
                worst['capability'] = (c, tag, i)
            if worst['complementarity'] is None or k > worst['complementarity'][0]:
                worst['complementarity'] = (k, tag, i)

    def summarize(values, label):
        return {
            'n_rows': len(values),
            'max': max(values) if values else None,
            'mean': statistics.fmean(values) if values else None,
            'p95': percentile(values, 0.95),
            'n_above_1e-6': sum(1 for v in values if v > 1e-6),
            'n_above_1e-4': sum(1 for v in values if v > 1e-4),
            'frac_above_1e-6': (sum(1 for v in values if v > 1e-6) / len(values)) if values else None,
            'worst_row': {'value': worst[label][0], 'model': worst[label][1],
                          'period': worst[label][2]} if worst[label] else None,
        }

    report['physical_residuals'] = {
        'converter_capability_normalized': summarize(cap_res, 'capability'),
        'complementarity_normalized': summarize(comp_res, 'complementarity'),
        'definitions': {
            'capability': 'max((pnet^2+qnet^2 - S^2)/S^2, 0)',
            'complementarity': 'max((pch*pdch - eps*S^2)/S^2, 0)',
            'eps': ESS_COMPLEMENTARITY_TOLERANCE},
    }

    res.pop('captured', None)
    report['physics_sample'] = dict(list(res['physics'].items())[:2])

    out = os.path.join(OUT_DIR, 'p54e_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[P5.4-E] report -> {out}')
    print(f"  solves={agg['total_local_solves']} {agg['per_agent']}")
    print(f"  primary_failures={agg['primary_failures']} recovery={agg['recovery_attempts']} "
          f"persistent={agg['persistent_failures']} {agg['persistent_failure_ids']}")
    print(f"  iterations={agg['iterations']}  runtime={agg['runtime_s']:.0f}s")
    for tag, spec in jac.items():
        f, r = spec['full'], spec['reduced']
        print(f"  {tag:24s} sess_snet_def_exists={spec['sess_snet_def_component_exists']} "
              f"zeroRows={f['n_exactly_zero_rows']} owners={spec['zero_row_components']} "
              f"smin_full={f['sigma_min']:.4e} reduced_cond={r['condition_number']:.4e}")
    for key, block in report['physical_residuals'].items():
        if key == 'definitions':
            continue
        print(f"  {key}: max={block['max']:.3e} mean={block['mean']:.3e} p95={block['p95']:.3e} "
              f">1e-6: {block['n_above_1e-6']} >1e-4: {block['n_above_1e-4']} of {block['n_rows']}")
        print(f"      worst: {block['worst_row']}")


if __name__ == '__main__':
    main()
