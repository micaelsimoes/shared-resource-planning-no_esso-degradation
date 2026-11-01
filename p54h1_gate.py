"""
Stage P5.4-H1.8/H1.9 -- production positive-bootstrap gate and complementarity
acceptance metrics for the dimensionless charge/discharge formulation.

Runs the exact production positive-bootstrap population with the real
implementation, then reports, for every shared-network row and every ESSO row
(per cohort and aggregated):

    pch_hat, pdch_hat, pch_hat*pdch_hat, max(pch_hat*pdch_hat - eps, 0)
    physical min(pch, pdch)/S

With eps = 1e-4, exact enforcement bounds equal-direction simultaneous
charge/discharge at sqrt(eps) = 1e-2 of rating. That is checked empirically
rather than assumed.

    python p54h1_gate.py
"""

import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from definitions import ESS_COMPLEMENTARITY_TOLERANCE, SHARED_ESS_ZERO_CAPACITY_TOLERANCE  # noqa: E402
from model_construction_helpers import period_duration_hours  # noqa: E402
from p53b3_active_power_ess import run_branch  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54H1')
EPS = ESS_COMPLEMENTARITY_TOLERANCE
SQRT_EPS = EPS ** 0.5

# accepted pre-H1 active-energy baseline (P5.4-D revalidation, commit c3526ec8)
BASELINE = {'iterations_total': 1556, 'runtime_s': 32.0,
            'network_max_p_circ_norm': 0.11562, 'esso_aggregate_max_p_circ_norm': 0.1912}


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


def block(values, label):
    if not values:
        return {'label': label, 'n': 0}
    return {'label': label, 'n': len(values),
            'max': max(values), 'mean': statistics.fmean(values),
            'median': statistics.median(values),
            'p95': percentile(values, 0.95), 'p99': percentile(values, 0.99),
            'count_gt_1e-3': sum(1 for v in values if v > 1e-3),
            'count_gt_1e-2': sum(1 for v in values if v > 1e-2)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-H1.8/H1.9', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'eps': EPS, 'sqrt_eps': SQRT_EPS, 'baseline': BASELINE}

    print('[H1] running production positive-bootstrap initialization ...', flush=True)
    res = run_branch(prototype=False, capture_tags=set())
    agg = res['aggregates']
    report['H1_8_gate'] = {
        'per_agent': agg['per_agent'],
        'primary_failures': agg['primary_failures'],
        'recovery_attempts': agg['recovery_attempts'],
        'recovery_successes': agg['recovery_successes'],
        'persistent_failures': agg['persistent_failures'],
        'persistent_failure_ids': agg['persistent_failure_ids'],
        'iterations': agg['iterations'],
        'runtime_s': agg['runtime_s'],
        'gate_36_12_3': (agg['per_agent'].get('dso', {}).get('succeeded') == 36
                         and agg['per_agent'].get('tso', {}).get('succeeded') == 12
                         and agg['per_agent'].get('esso', {}).get('succeeded') == 3),
        'zero_persistent_failures': agg['persistent_failures'] == 0,
        'iteration_delta_vs_baseline': (agg['iterations']['total'] or 0) - BASELINE['iterations_total'],
    }
    print(f"    {agg['per_agent']} persistent={agg['persistent_failures']} "
          f"iters={agg['iterations']['total']} runtime={agg['runtime_s']:.0f}s", flush=True)

    # ---------------- network rows ----------------
    net_rows = []
    for tag, rec in res['physics'].items():
        s = rec['s_rated']
        if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
            continue
        eff_ch, eff_dch, dt, base = rec['eff_ch'], rec['eff_dch'], rec['dt'], rec['base_mva']
        e_rated = rec['e_rated']
        day_loss = day_legit = 0.0
        for i in range(len(rec['pch'])):
            pch, pdch = rec['pch'][i], rec['pdch'][i]
            hc, hd = rec['pch_hat'][i], rec['pdch_hat'][i]
            c = min(pch, pdch)
            loss = c * dt * (1.0 / eff_dch - eff_ch)
            day_loss += loss
            day_legit += eff_ch * pch * dt + pdch * dt / eff_dch
            net_rows.append({
                'model': tag, 'period': i, 's_rated': s,
                'pch': pch, 'pdch': pdch,
                'pch_hat': hc, 'pdch_hat': hd,
                'hat_product': hc * hd,
                'hat_violation': max(hc * hd - EPS, 0.0),
                'p_circ_norm': c / s, 'p_circ_MW': c * base,
                'link_ch_residual': pch - s * hc,
                'link_dch_residual': pdch - s * hd,
                'E_circ_loss_MWh': loss * base,
            })
        report.setdefault('network_per_model_day', {})[tag] = {
            's_rated': s, 'e_rated': e_rated,
            'E_circ_loss_day_MWh': day_loss * base,
            'E_circ_loss_day_over_E_rated': (day_loss / e_rated) if e_rated else None,
            'circ_loss_share_of_throughput': (day_loss / day_legit) if day_legit > 0 else 0.0,
        }

    # ---------------- ESSO rows ----------------
    esso_cohort_rows, esso_agg_rows = [], []
    for nid, m in res['esso_models'].items():
        if not srp._solver_result_succeeded(res['esso_results'][nid]):
            continue
        dt = float(period_duration_hours(m))
        for y in m.years:
            s_total = float(pe.value(m.es_s_rated[y]))
            for d in m.days:
                for p in m.periods:
                    pch_a = sum(float(pe.value(m.es_pch_per_unit[yi, y, d, p])) for yi in m.years)
                    pdch_a = sum(float(pe.value(m.es_pdch_per_unit[yi, y, d, p])) for yi in m.years)
                    hca = float(pe.value(m.es_pch_hat_agg[y, d, p]))
                    hda = float(pe.value(m.es_pdch_hat_agg[y, d, p]))
                    if s_total > SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                        esso_agg_rows.append({
                            'node': nid, 'year': y, 'day': d, 'period': p,
                            's_total': s_total, 'pch': pch_a, 'pdch': pdch_a,
                            'pch_hat': hca, 'pdch_hat': hda,
                            'hat_product': hca * hda,
                            'hat_violation': max(hca * hda - EPS, 0.0),
                            'p_circ_norm': min(pch_a, pdch_a) / s_total,
                            'link_ch_residual': pch_a - s_total * hca,
                            'link_dch_residual': pdch_a - s_total * hda,
                        })
                    for yi in m.years:
                        s_c = float(pe.value(m.es_s_rated_per_unit[yi, y]))
                        if s_c <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                            continue
                        pch = float(pe.value(m.es_pch_per_unit[yi, y, d, p]))
                        pdch = float(pe.value(m.es_pdch_per_unit[yi, y, d, p]))
                        hc = float(pe.value(m.es_pch_hat_per_unit[yi, y, d, p]))
                        hd = float(pe.value(m.es_pdch_hat_per_unit[yi, y, d, p]))
                        esso_cohort_rows.append({
                            'node': nid, 'cohort': yi, 'year': y, 'day': d, 'period': p,
                            's_cohort': s_c, 'pch': pch, 'pdch': pdch,
                            'pch_hat': hc, 'pdch_hat': hd,
                            'hat_product': hc * hd,
                            'hat_violation': max(hc * hd - EPS, 0.0),
                            'p_circ_norm': min(pch, pdch) / s_c,
                            'link_ch_residual': pch - s_c * hc,
                            'link_dch_residual': pdch - s_c * hd,
                        })

    def summarize(rows, name):
        return {
            'population': name, 'n_rows': len(rows),
            'pch_hat': block([r['pch_hat'] for r in rows], 'pch_hat'),
            'pdch_hat': block([r['pdch_hat'] for r in rows], 'pdch_hat'),
            'hat_product': block([r['hat_product'] for r in rows], 'pch_hat*pdch_hat'),
            'hat_violation': block([r['hat_violation'] for r in rows],
                                   'max(pch_hat*pdch_hat - eps, 0)'),
            'p_circ_norm': block([r['p_circ_norm'] for r in rows], 'min(pch,pdch)/S'),
            'n_violating': sum(1 for r in rows if r['hat_violation'] > 0.0),
            'max_link_residual': max(
                (max(abs(r['link_ch_residual']), abs(r['link_dch_residual'])) for r in rows),
                default=0.0),
            'max_p_circ_norm_vs_sqrt_eps': (
                max((r['p_circ_norm'] for r in rows), default=0.0) / SQRT_EPS),
            'worst_5': sorted(rows, key=lambda r: -r['p_circ_norm'])[:5],
        }

    dso_rows = [r for r in net_rows if r['model'].startswith('dso/')]
    tso_rows = [r for r in net_rows if r['model'].startswith('tso/')]
    report['H1_9_metrics'] = {
        'network_all': summarize(net_rows, 'shared network ESS (DSO+TSO)'),
        'network_dso': summarize(dso_rows, 'shared network ESS (DSO)'),
        'network_tso': summarize(tso_rows, 'shared network ESS (TSO)'),
        'esso_cohort': summarize(esso_cohort_rows, 'ESSO per cohort'),
        'esso_aggregate': summarize(esso_agg_rows, 'ESSO aggregate'),
    }

    pm = report.get('network_per_model_day', {})
    report['circulating_loss'] = {
        'formula': 'E_circ_loss = min(pch,pdch) * dt * (1/eta_dch - eta_ch)',
        'max_per_period_MWh': max((r['E_circ_loss_MWh'] for r in net_rows), default=0.0),
        'worst_day_MWh': max((v['E_circ_loss_day_MWh'] for v in pm.values()), default=0.0),
        'worst_day_over_E_rated': max(
            ((v['E_circ_loss_day_over_E_rated'] or 0.0) for v in pm.values()), default=0.0),
        'worst_day_share_of_throughput': max(
            (v['circ_loss_share_of_throughput'] for v in pm.values()), default=0.0),
    }

    g = report['H1_8_gate']
    m = report['H1_9_metrics']
    report['gate_evaluation'] = {
        '1_all_local_solves_succeed': g['gate_36_12_3'] and g['zero_persistent_failures'],
        '3_network_complementarity_resolved': m['network_all']['hat_violation']['max'] <= 1e-8,
        '4_esso_cohort_complementarity_resolved': m['esso_cohort']['hat_violation']['max'] <= 1e-8,
        '5_esso_aggregate_complementarity_resolved': m['esso_aggregate']['hat_violation']['max'] <= 1e-8,
        '6_p_circ_within_sqrt_eps_allowance': (
            m['network_all']['p_circ_norm']['max'] <= SQRT_EPS * 1.05
            and m['esso_aggregate']['p_circ_norm']['max'] <= SQRT_EPS * 1.05),
    }

    out = os.path.join(OUT_DIR, 'p54h1_gate_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[H1] report -> {out}')
    print(f"  H1.8 gate: {g['per_agent']} primary_fail={g['primary_failures']} "
          f"recovery={g['recovery_attempts']} persistent={g['persistent_failures']}")
    print(f"    iterations {g['iterations']} (baseline {BASELINE['iterations_total']}, "
          f"delta {g['iteration_delta_vs_baseline']:+d})  runtime {g['runtime_s']:.0f}s")
    print(f"\n  H1.9 metrics (eps={EPS:g}, sqrt(eps)={SQRT_EPS:g}):")
    for key in ('network_all', 'network_dso', 'network_tso', 'esso_cohort', 'esso_aggregate'):
        b = m[key]
        hv, pc = b['hat_violation'], b['p_circ_norm']
        print(f"   {b['population']:32s} n={b['n_rows']:5d} violating={b['n_violating']:5d} "
              f"max_viol={hv['max']:.4e} link_res={b['max_link_residual']:.3e}")
        print(f"      hat_product max={b['hat_product']['max']:.4e} p95={b['hat_product']['p95']:.4e}")
        print(f"      p_circ/S max={pc['max']:.4e} mean={pc['mean']:.4e} med={pc['median']:.4e} "
              f"p95={pc['p95']:.4e} p99={pc['p99']:.4e} >1e-3:{pc['count_gt_1e-3']} "
              f">1e-2:{pc['count_gt_1e-2']}  (max/sqrt_eps = {b['max_p_circ_norm_vs_sqrt_eps']:.4f})")
    c = report['circulating_loss']
    print(f"\n  circulating loss: max/period {c['max_per_period_MWh']:.4e} MWh; "
          f"worst day {c['worst_day_MWh']:.4e} MWh = {c['worst_day_over_E_rated']:.4e} E_rated, "
          f"{c['worst_day_share_of_throughput']:.4e} of throughput")
    print(f"\n  gate evaluation: {json.dumps(report['gate_evaluation'], indent=1)}")


if __name__ == '__main__':
    main()
