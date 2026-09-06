"""
Stage P5.4-E2 -- physical significance of the active-power complementarity residual.

Uses the already-validated production positive-bootstrap solutions. For every
active shared-ESS row it separates

  * the PRODUCT residual, which is what the inequality actually constrains, from
  * the CIRCULATING power `c = min(pch, pdch)`, which is the physically
    meaningful simultaneous charge/discharge quantity.

The product ratio alone overstates simultaneity: `sqrt(pch*pdch)/S` is a
geometric mean, and a large product can come from one large and one small
directional power. `min(pch, pdch)` is the component that genuinely flows both
ways at once.

The circulating component produces no net active injection but does destroy
stored energy, because `eta_ch*c - c/eta_dch < 0`. That artificial loss is
quantified here as `E_circ_loss = c * dt * (1/eta_dch - eta_ch)`.

No formulation change. Diagnostic only.

    python p54e2_complementarity_significance.py
"""

import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from definitions import (ESS_COMPLEMENTARITY_TOLERANCE, PENALTY_ESS_COMPLEMENTARITY,  # noqa: E402
                         SHARED_ESS_ZERO_CAPACITY_TOLERANCE)

from p53b3_active_power_ess import run_branch  # noqa: E402  (production, prototype=False)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54E2')


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


def stats(values, label):
    if not values:
        return {'label': label, 'n': 0}
    return {
        'label': label, 'n': len(values),
        'max': max(values), 'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'p95': percentile(values, 0.95), 'p99': percentile(values, 0.99),
        'n_nonzero': sum(1 for v in values if v > 0.0),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-E2', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'eps': ESS_COMPLEMENTARITY_TOLERANCE,
              'method': ('production positive-bootstrap solutions; product residual '
                         'separated from circulating power min(pch, pdch)')}

    print('[E2] running production positive-bootstrap initialization ...', flush=True)
    res = run_branch(prototype=False, capture_tags=set())
    report['aggregates'] = res['aggregates']
    print(f"    {res['aggregates']['per_agent']} "
          f"persistent={res['aggregates']['persistent_failures']}", flush=True)

    rows = []
    per_model_day = {}
    for tag, rec in res['physics'].items():
        s = rec['s_rated']
        if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
            continue
        # strict access: every physical parameter comes from the live production
        # model. A missing field must fail loudly rather than be defaulted.
        eff_ch = rec['eff_ch']
        eff_dch = rec['eff_dch']
        dt = rec['dt']
        base = rec['base_mva']
        e_rated = rec['e_rated']
        day_loss = 0.0
        day_legit = 0.0
        for i in range(len(rec['pch'])):
            pch, pdch = rec['pch'][i], rec['pdch'][i]
            c = min(pch, pdch)
            loss = c * dt * (1.0 / eff_dch - eff_ch)
            legit = eff_ch * pch * dt + pdch * dt / eff_dch
            day_loss += loss
            day_legit += legit
            rows.append({
                'model': tag, 'period': i, 's_rated': s,
                'pch': pch, 'pdch': pdch,
                'pnet': rec['pnet'][i], 'qnet': rec['qnet'][i], 'soc': rec['soc'][i],
                'pch_norm': pch / s, 'pdch_norm': pdch / s,
                'p_circ_norm': c / s, 'p_circ_MW': c * base,
                'p_net_norm': abs(pch - pdch) / s,
                'r_prod': pch * pdch / s ** 2,
                'r_violation': max(pch * pdch / s ** 2 - ESS_COMPLEMENTARITY_TOLERANCE, 0.0),
                'E_circ_loss_pu': loss,
                'E_circ_loss_MWh': loss * base,
                'complementarity_limit': ESS_COMPLEMENTARITY_TOLERANCE,
            })
        per_model_day[tag] = {
            's_rated': s, 'e_rated': e_rated,
            'E_circ_loss_day_pu': day_loss,
            'E_circ_loss_day_MWh': day_loss * base,
            'E_circ_loss_day_over_E_rated': (day_loss / e_rated) if e_rated else None,
            'legit_throughput_day_pu': day_legit,
            'circ_loss_share_of_throughput': (day_loss / day_legit) if day_legit > 0 else 0.0,
        }

    circ_norm = [r['p_circ_norm'] for r in rows]
    circ_mw = [r['p_circ_MW'] for r in rows]
    report['n_rows'] = len(rows)
    report['p_circ_norm_stats'] = stats(circ_norm, 'min(pch,pdch)/S')
    report['p_circ_MW_stats'] = stats(circ_mw, 'min(pch,pdch) [MW]')
    report['p_circ_norm_thresholds'] = {
        'above_1e-4_S': sum(1 for v in circ_norm if v > 1e-4),
        'above_1e-3_S': sum(1 for v in circ_norm if v > 1e-3),
        'above_1e-2_S': sum(1 for v in circ_norm if v > 1e-2),
    }
    report['r_prod_stats'] = stats([r['r_prod'] for r in rows], 'pch*pdch/S^2')
    report['r_violation_stats'] = stats([r['r_violation'] for r in rows], 'product violation')

    # per-agent split: P5.4-E audited DSO rows only, so the TSO rows must be
    # reported separately rather than folded into a single population.
    report['per_agent'] = {}
    for agent in ('dso', 'tso'):
        sub = [r for r in rows if r['model'].startswith(agent + '/')]
        if not sub:
            continue
        report['per_agent'][agent] = {
            'n_rows': len(sub),
            'p_circ_norm': stats([r['p_circ_norm'] for r in sub], 'min(pch,pdch)/S'),
            'p_circ_MW': stats([r['p_circ_MW'] for r in sub], 'min(pch,pdch) [MW]'),
            'r_prod': stats([r['r_prod'] for r in sub], 'pch*pdch/S^2'),
            'r_violation': stats([r['r_violation'] for r in sub], 'product violation'),
            'n_violating_product': sum(1 for r in sub if r['r_violation'] > 0.0),
            'above_1e-2_S': sum(1 for r in sub if r['p_circ_norm'] > 1e-2),
        }

    worst = sorted(rows, key=lambda r: -r['p_circ_norm'])[:20]
    report['worst_20_by_circulating_power'] = worst
    report['all_rows'] = rows

    losses = [r['E_circ_loss_pu'] for r in rows]
    report['artificial_cycling_loss'] = {
        'formula': 'E_circ_loss = min(pch,pdch) * dt * (1/eta_dch - eta_ch)',
        'max_per_period_pu': max(losses) if losses else 0.0,
        'max_per_period_MWh': max(r['E_circ_loss_MWh'] for r in rows) if rows else 0.0,
        'per_model_day': per_model_day,
        'worst_day_over_E_rated': max(
            (v['E_circ_loss_day_over_E_rated'] or 0.0) for v in per_model_day.values())
        if per_model_day else 0.0,
        'worst_day_share_of_throughput': max(
            v['circ_loss_share_of_throughput'] for v in per_model_day.values())
        if per_model_day else 0.0,
    }

    # objective penalty associated with the product term
    worst_pen = max((PENALTY_ESS_COMPLEMENTARITY * r['pch'] * r['pdch'] for r in rows),
                    default=0.0)
    report['objective_penalty'] = {
        'PENALTY_ESS_COMPLEMENTARITY': PENALTY_ESS_COMPLEMENTARITY,
        'note': 'penalty term is base * PENALTY * pch * pdch (per row, per scenario)',
        'worst_row_penalty_per_unit_base': worst_pen,
    }

    with open(os.path.join(OUT_DIR, 'p54e2_report.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f"\n[E2] rows audited: {report['n_rows']}")
    for key in ('p_circ_norm_stats', 'p_circ_MW_stats', 'r_prod_stats', 'r_violation_stats'):
        b = report[key]
        print(f"  {b['label']:24s} max={b['max']:.4e} mean={b['mean']:.4e} "
              f"median={b['median']:.4e} p95={b['p95']:.4e} p99={b['p99']:.4e} nonzero={b['n_nonzero']}")
    print(f"  p_circ thresholds: {report['p_circ_norm_thresholds']}")
    a = report['artificial_cycling_loss']
    print(f"  circulating loss: max/period {a['max_per_period_pu']:.4e} pu "
          f"({a['max_per_period_MWh']:.4e} MWh)")
    print(f"     worst day / E_rated = {a['worst_day_over_E_rated']:.4e}; "
          f"worst day share of throughput = {a['worst_day_share_of_throughput']:.4e}")
    print("\n  worst 5 rows by circulating power:")
    for r in worst[:5]:
        print(f"   {r['model']:22s} p{r['period']:2d} S={r['s_rated']:.3e} "
              f"pch/S={r['pch_norm']:.4e} pdch/S={r['pdch_norm']:.4e} "
              f"min/S={r['p_circ_norm']:.4e} ({r['p_circ_MW']:.3e} MW) "
              f"r_prod={r['r_prod']:.4e}")


if __name__ == '__main__':
    main()
