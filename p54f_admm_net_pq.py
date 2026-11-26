"""
Stage P5.4-F -- live distributed ADMM with net-P/Q coordination only.

Runs the EXACT positive-bootstrap candidate through the real production
distributed operational planning loop:

    SharedResourcesPlanning.run_operational_planning(type='distributed', ...)

The candidate is built by the production `_build_positive_bootstrap_candidate`
and passed straight through -- it has the same shape as the candidate
`get_test_candidate_solution` produces, which is what `main.py` passes, so no
reconstruction is involved.

Nothing about ADMM is changed: rho, tolerances, proximal regularization,
objective scaling, IPOPT options, MA97/exact-Hessian policy and the recovery
policy are all untouched. The outer planning loop is NOT run.

Complementarity and energy-consistency figures collected here are SANITY
DIAGNOSTICS. They are read off the converged models after the fact and are
never fed into a consensus update.

    python p54f_admm_net_pq.py
"""

import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from definitions import ESS_COMPLEMENTARITY_TOLERANCE, SHARED_ESS_ZERO_CAPACITY_TOLERANCE  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54F')
EPS = ESS_COMPLEMENTARITY_TOLERANCE
SQRT_EPS = EPS ** 0.5


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
    return {'label': label, 'n': len(values), 'max': max(values),
            'mean': statistics.fmean(values), 'median': statistics.median(values),
            'p95': percentile(values, 0.95), 'p99': percentile(values, 0.99),
            'count_gt_1e-3': sum(1 for v in values if v > 1e-3),
            'count_gt_1e-2': sum(1 for v in values if v > 1e-2)}


def parse_console(text):
    """Pull the per-cycle ADMM trace out of the production console output."""
    cycles = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r'ADMM\s+cycle|Iteration\s+\d+|cycle\s+\d+', stripped, re.I):
            cycles.append(stripped)
    return cycles


def shared_ess_rows(model, e_idx, s_rated, tag, eff_ch, eff_dch, base):
    rows = []
    for p in model.periods:
        pch = float(pe.value(model.shared_es_pch[e_idx, 0, 0, p]))
        pdch = float(pe.value(model.shared_es_pdch[e_idx, 0, 0, p]))
        hc = float(pe.value(model.shared_es_pch_hat[e_idx, 0, 0, p]))
        hd = float(pe.value(model.shared_es_pdch_hat[e_idx, 0, 0, p]))
        rows.append({
            'model': tag, 'period': p, 's_rated': s_rated,
            'pch': pch, 'pdch': pdch,
            'pnet': float(pe.value(model.shared_es_pnet[e_idx, 0, 0, p])),
            'qnet': float(pe.value(model.shared_es_qnet[e_idx, 0, 0, p])),
            'pch_hat': hc, 'pdch_hat': hd,
            'hat_product': hc * hd,
            'hat_violation': max(hc * hd - EPS, 0.0),
            'p_circ_norm': min(pch, pdch) / s_rated,
            'p_cell': eff_ch * pch - pdch / eff_dch,
            'p_cell_MW': (eff_ch * pch - pdch / eff_dch) * base,
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-F', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'eps': EPS, 'sqrt_eps': SQRT_EPS,
              'coordination': 'net P/Q + interface voltage only; no pch/pdch/SOC consensus'}

    console = io.StringIO()
    started = time.time()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        convergence, results, models, sensitivities, primal_evolution, state = \
            planning.run_operational_planning(
                type='distributed',
                candidate_solution=candidate,
                print_results=False,
                debug_flag=False,
                return_state=True)
    runtime = time.time() - started
    text = console.getvalue()

    with open(os.path.join(OUT_DIR, 'p54f_console.log'), 'w') as handle:
        handle.write(text)

    report['run'] = {
        'converged': bool(convergence),
        'runtime_s': runtime,
        'initialization_failed': bool(state.get('initialization_failed', False)),
        'state_keys': sorted(state.keys()),
        'n_primal_evolution_entries': len(primal_evolution) if primal_evolution else 0,
        'primal_evolution': primal_evolution,
        'console_tail': text[-6000:],
        'cycle_lines': parse_console(text)[-80:],
    }
    print(f"[F] converged={convergence} runtime={runtime:.0f}s "
          f"init_failed={report['run']['initialization_failed']}", flush=True)

    # ---- residual metrics from the production diagnostic, if exposed on state ----
    for key in ('admm_diagnostics', 'solver_recovery_diagnostics', 'admm_residuals',
                'residuals', 'convergence_history', 'recourse_stationarity',
                'rho_evolution', 'cycles'):
        if key in state:
            report.setdefault('state_metrics', {})[key] = state[key]

    # ---- complementarity + energy-consistency sanity diagnostics ----
    net_rows = []
    tso_models = models.get('tso')
    dso_models = models.get('dso')
    esso_models = models.get('esso')

    for node_id, per_year in (dso_models or {}).items():
        dso = planning.distribution_networks[node_id]
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                net = dso.network[year][day]
                e_idx = net.get_shared_energy_storage_idx(net.get_reference_node_id())
                s = float(pe.value(model.shared_es_s_rated_fixed[e_idx]))
                if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                    continue
                sess = net.shared_energy_storages[e_idx]
                net_rows.extend(shared_ess_rows(
                    model, e_idx, s, f'dso/{node_id}/{year}/{day}',
                    sess.eff_ch, sess.eff_dch, net.baseMVA))

    for year, per_day in (tso_models or {}).items():
        for day, model in per_day.items():
            net = planning.transmission_network.network[year][day]
            for e_idx, sess in enumerate(net.shared_energy_storages):
                s = float(pe.value(model.shared_es_s_rated_fixed[e_idx]))
                if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                    continue
                net_rows.extend(shared_ess_rows(
                    model, e_idx, s, f'tso/{e_idx}/{year}/{day}',
                    sess.eff_ch, sess.eff_dch, net.baseMVA))

    esso_rows = []
    for nid, m in (esso_models or {}).items():
        for y in m.years:
            s_total = float(pe.value(m.es_s_rated[y]))
            if s_total <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                continue
            for d in m.days:
                for p in m.periods:
                    pch = sum(float(pe.value(m.es_pch_per_unit[yi, y, d, p])) for yi in m.years)
                    pdch = sum(float(pe.value(m.es_pdch_per_unit[yi, y, d, p])) for yi in m.years)
                    hc = float(pe.value(m.es_pch_hat_agg[y, d, p]))
                    hd = float(pe.value(m.es_pdch_hat_agg[y, d, p]))
                    esso_rows.append({
                        'node': nid, 'year': y, 'day': d, 'period': p,
                        's_total': s_total, 'pch': pch, 'pdch': pdch,
                        'pnet': float(pe.value(m.es_pnet[y, d, p])),
                        'qnet': float(pe.value(m.es_qnet[y, d, p])),
                        'pch_hat': hc, 'pdch_hat': hd,
                        'hat_product': hc * hd,
                        'hat_violation': max(hc * hd - EPS, 0.0),
                        'p_circ_norm': min(pch, pdch) / s_total,
                    })

    def summarize(rows, name):
        if not rows:
            return {'population': name, 'n_rows': 0}
        return {'population': name, 'n_rows': len(rows),
                'hat_product': block([r['hat_product'] for r in rows], 'pch_hat*pdch_hat'),
                'hat_violation': block([r['hat_violation'] for r in rows], 'violation'),
                'p_circ_norm': block([r['p_circ_norm'] for r in rows], 'min(pch,pdch)/S'),
                'n_violating': sum(1 for r in rows if r['hat_violation'] > 0.0),
                'max_p_circ_over_sqrt_eps': max(r['p_circ_norm'] for r in rows) / SQRT_EPS,
                'worst_5': sorted(rows, key=lambda r: -r['p_circ_norm'])[:5]}

    report['F_complementarity_sanity'] = {
        'network_all': summarize(net_rows, 'shared network ESS (DSO+TSO)'),
        'network_dso': summarize([r for r in net_rows if r['model'].startswith('dso/')], 'DSO'),
        'network_tso': summarize([r for r in net_rows if r['model'].startswith('tso/')], 'TSO'),
        'esso_aggregate': summarize(esso_rows, 'ESSO aggregate'),
        'note': 'sanity diagnostics only; never fed into a consensus update',
    }

    # ---- energy-consistency: compare p_cell for the same shared ESS ----
    by_key = {}
    for r in net_rows:
        agent, ident, year, day = r['model'].split('/')
        by_key.setdefault((year, day, r['period']), {}).setdefault(agent, []).append(r)
    consistency = []
    for (year, day, period), agents in sorted(by_key.items()):
        if 'dso' not in agents or 'tso' not in agents:
            continue
        dso_cell = [a['p_cell'] for a in agents['dso']]
        tso_cell = [a['p_cell'] for a in agents['tso']]
        dso_pnet = [a['pnet'] for a in agents['dso']]
        tso_pnet = [a['pnet'] for a in agents['tso']]
        consistency.append({
            'year': year, 'day': day, 'period': period,
            'dso_p_cell_sum': sum(dso_cell), 'tso_p_cell_sum': sum(tso_cell),
            'abs_p_cell_difference': abs(sum(dso_cell) - sum(tso_cell)),
            'dso_pnet_sum': sum(dso_pnet), 'tso_pnet_sum': sum(tso_pnet),
            'abs_pnet_difference': abs(sum(dso_pnet) - sum(tso_pnet)),
        })
    if consistency:
        report['F_energy_consistency'] = {
            'n_compared': len(consistency),
            'max_abs_p_cell_difference': max(c['abs_p_cell_difference'] for c in consistency),
            'mean_abs_p_cell_difference': statistics.fmean(
                c['abs_p_cell_difference'] for c in consistency),
            'max_abs_pnet_difference': max(c['abs_pnet_difference'] for c in consistency),
            'worst_5': sorted(consistency, key=lambda c: -c['abs_p_cell_difference'])[:5],
            'note': ('diagnostic only -- p_cell = eta_ch*pch - pdch/eta_dch; not an '
                     'ADMM residual. Compares the SUM over the shared ESS devices each '
                     'agent represents for the same year/day/period.'),
        }

    out = os.path.join(OUT_DIR, 'p54f_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[F] report -> {out}')
    print(f"  converged={report['run']['converged']} runtime={runtime:.0f}s")
    for key, b in report['F_complementarity_sanity'].items():
        if key == 'note' or not b.get('n_rows'):
            continue
        pc = b['p_circ_norm']
        print(f"   {b['population']:30s} n={b['n_rows']:5d} violating={b['n_violating']} "
              f"max_viol={b['hat_violation']['max']:.4e}")
        print(f"      p_circ/S max={pc['max']:.4e} mean={pc['mean']:.4e} med={pc['median']:.4e} "
              f"p95={pc['p95']:.4e} p99={pc['p99']:.4e} >1e-3:{pc['count_gt_1e-3']} "
              f">1e-2:{pc['count_gt_1e-2']} (max/sqrt_eps={b['max_p_circ_over_sqrt_eps']:.4f})")
    if 'F_energy_consistency' in report:
        c = report['F_energy_consistency']
        print(f"   energy consistency: n={c['n_compared']} "
              f"max|dP_cell|={c['max_abs_p_cell_difference']:.4e} "
              f"mean={c['mean_abs_p_cell_difference']:.4e} "
              f"max|dPnet|={c['max_abs_pnet_difference']:.4e}")


if __name__ == '__main__':
    main()
