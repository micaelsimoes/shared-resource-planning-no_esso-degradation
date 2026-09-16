"""P5.15 Step 3.2 E2 - Z1: what drives the shared-ESS consensus drift (zero solves, SolveProfileGuard armed).

Source: data/SRP1/Results/P515S33_E2_run/esso_capture/baseline/node{5,7,9}_cycle{001..150}.jsonl, one row per
(year y, day d, period p) with pch, pdch, pnet (pu of the ESSO's own base), slacks, barrier multipliers zL_pch/zL_pdch
and the constraint duals. These were written by the harness during the gate; nothing is recomputed by solving.

Per node and cycle k:
  throughput(k)   = sum over entries of (pch + pdch)          (the quantity EPS_ESSO_THROUGHPUT prices)
  abs_pnet(k)     = sum |pnet|
  d_pnet(k)       = pnet(k) - pnet(k-1)                        (vector over entries, fixed (y,d,p) order)
  step_norm(k)    = ||d_pnet(k)||
  cos_consecutive(k) = cos(d_pnet(k), d_pnet(k-1))             (1 = the drift keeps the same direction)
  cos_toward_zero(k) = cos(d_pnet(k), -sign(pnet(k-1)))        (1 = the step reduces |pnet|, i.e. reduces throughput,
                                                                which is the direction -grad of the throughput term)
  dual_agg(k)     = mean and spread of the 'energy_storage_operation_agg' dual   (compare EPS_ESSO_THROUGHPUT)
  zL_pch(k), zL_pdch(k) = mean barrier multipliers                                (H2: barrier-bias drift)
Discontinuity test (H3): the network-failure cycles from network_failures_baseline.jsonl are reported with the
step_norm just before and after each.
Write-once output: data/SRP1/Results/P515S33_E2_run/s33e2_z1_esso_drift.json
"""
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33_E2_run')
CAP = os.path.join(RUN, 'esso_capture', 'baseline')
OUT = os.path.join(RUN, 's33e2_z1_esso_drift.json')
NODES = (5, 7, 9)
SAMPLE = (2, 5, 10, 25, 31, 50, 75, 100, 125, 150)


def _vec(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (r['y'], r['d'], r['p']))
    keys = [(r['y'], r['d'], r['p']) for r in rows]
    pnet = [float(r['pnet']) for r in rows]
    thr = sum(float(r['pch']) + float(r['pdch']) for r in rows)
    abs_pnet = sum(abs(v) for v in pnet)
    duals = []
    for r in rows:
        for d in r.get('duals') or []:
            if d.get('component') == 'energy_storage_operation_agg' and d.get('dual') is not None:
                duals.append(float(d['dual']))
    zl_pch = [float(r['zL_pch']) for r in rows if r.get('zL_pch') is not None]
    zl_pdch = [float(r['zL_pdch']) for r in rows if r.get('zL_pdch') is not None]
    return {'keys': keys, 'pnet': pnet, 'throughput': thr, 'abs_pnet': abs_pnet,
            'dual_agg_mean': (sum(duals) / len(duals)) if duals else None,
            'dual_agg_min': min(duals) if duals else None, 'dual_agg_max': max(duals) if duals else None,
            'zL_pch_mean': (sum(zl_pch) / len(zl_pch)) if zl_pch else None,
            'zL_pdch_mean': (sum(zl_pdch) / len(zl_pdch)) if zl_pdch else None}


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s33e2 Z1').install()
    try:
        eps = float(re.search(r'^EPS_ESSO_THROUGHPUT\s*=\s*([0-9eE.+-]+)', open(os.path.join(REPO, 'definitions.py')).read(), re.M).group(1))
        fail_cycles = sorted({json.loads(l)['cycle'] for l in open(os.path.join(RUN, 'network_failures_baseline.jsonl')) if l.strip()})
        out = {'stage': 'P5.15 s33e2 Z1 - shared-ESS consensus drift mechanism', 'eps_esso_throughput': eps,
               'network_failure_cycles': fail_cycles, 'nodes': {}}
        for node in NODES:
            per_cycle, prev, prev_step = [], None, None
            for k in range(1, 151):
                path = os.path.join(CAP, f'node{node}_cycle{k:03d}.jsonl')
                if not os.path.exists(path):
                    continue
                cur = _vec(path)
                entry = {'cycle': k, 'throughput': cur['throughput'], 'abs_pnet': cur['abs_pnet'],
                         'dual_agg_mean': cur['dual_agg_mean'], 'dual_agg_min': cur['dual_agg_min'],
                         'dual_agg_max': cur['dual_agg_max'], 'zL_pch_mean': cur['zL_pch_mean'],
                         'zL_pdch_mean': cur['zL_pdch_mean'], 'step_norm': None, 'cos_consecutive': None,
                         'cos_toward_zero': None}
                if prev is not None and prev['keys'] == cur['keys']:
                    step = [c - p for c, p in zip(cur['pnet'], prev['pnet'])]
                    entry['step_norm'] = math.sqrt(sum(x * x for x in step))
                    entry['cos_toward_zero'] = _cos(step, [-(1.0 if v > 0 else (-1.0 if v < 0 else 0.0)) for v in prev['pnet']])
                    if prev_step is not None:
                        entry['cos_consecutive'] = _cos(step, prev_step)
                    prev_step = step
                per_cycle.append(entry)
                prev = cur
            tail = [e for e in per_cycle if e['cycle'] >= 31 and e['step_norm'] is not None]
            out['nodes'][str(node)] = {
                'sampled': [e for e in per_cycle if e['cycle'] in SAMPLE],
                'throughput_first_last': [per_cycle[0]['throughput'], per_cycle[-1]['throughput']],
                'throughput_change_per_cycle_31_150': (per_cycle[-1]['throughput'] - per_cycle[30]['throughput']) / (len(per_cycle) - 31),
                'abs_pnet_first_last': [per_cycle[0]['abs_pnet'], per_cycle[-1]['abs_pnet']],
                'step_norm_mean_31_150': sum(e['step_norm'] for e in tail) / len(tail),
                'step_norm_min_max_31_150': [min(e['step_norm'] for e in tail), max(e['step_norm'] for e in tail)],
                'cos_consecutive_mean_31_150': sum(e['cos_consecutive'] for e in tail if e['cos_consecutive'] is not None) / len([e for e in tail if e['cos_consecutive'] is not None]),
                'cos_toward_zero_mean_31_150': sum(e['cos_toward_zero'] for e in tail if e['cos_toward_zero'] is not None) / len([e for e in tail if e['cos_toward_zero'] is not None]),
                'dual_agg_mean_terminal': per_cycle[-1]['dual_agg_mean'],
                'dual_agg_over_eps_terminal': (per_cycle[-1]['dual_agg_mean'] / eps) if per_cycle[-1]['dual_agg_mean'] else None,
                'zL_pch_mean_first_last': [per_cycle[0]['zL_pch_mean'], per_cycle[-1]['zL_pch_mean']],
                'zL_pdch_mean_first_last': [per_cycle[0]['zL_pdch_mean'], per_cycle[-1]['zL_pdch_mean']],
                'step_norm_around_failure_cycles': [
                    {'failure_cycle': fc,
                     'step_norm': [next((e['step_norm'] for e in per_cycle if e['cycle'] == c), None) for c in range(fc - 1, fc + 3)]}
                    for fc in fail_cycles],
            }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('eps_esso_throughput', eps, '| failure cycles', fail_cycles)
    for node, n in out['nodes'].items():
        print(f"node {node}: throughput {n['throughput_first_last'][0]:.6g} -> {n['throughput_first_last'][1]:.6g} "
              f"({n['throughput_change_per_cycle_31_150']:+.3g}/cycle over 31-150) | sum|pnet| "
              f"{n['abs_pnet_first_last'][0]:.6g} -> {n['abs_pnet_first_last'][1]:.6g}")
        print(f"   step_norm mean {n['step_norm_mean_31_150']:.4g} range {n['step_norm_min_max_31_150'][0]:.3g}-{n['step_norm_min_max_31_150'][1]:.3g} | "
              f"cos(consecutive) {n['cos_consecutive_mean_31_150']:.3f} | cos(toward zero) {n['cos_toward_zero_mean_31_150']:.3f}")
        print(f"   dual_agg terminal {n['dual_agg_mean_terminal']:.6g} = {n['dual_agg_over_eps_terminal']:.4f} x eps | "
              f"zL_pch {n['zL_pch_mean_first_last'][0]:.3g} -> {n['zL_pch_mean_first_last'][1]:.3g} | "
              f"zL_pdch {n['zL_pdch_mean_first_last'][0]:.3g} -> {n['zL_pdch_mean_first_last'][1]:.3g}")
    print('guard', out['solve_profile_guard'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
