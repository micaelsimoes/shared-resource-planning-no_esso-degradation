"""P5.15 S56 step 4 CALIBRATION RE-RUN -- zero-solve evaluation of the two V-1 runs under spec v3
(`step4_calib_v1/frozen_step4_calib_spec_v3_<sha8>.json`): the frozen decision rule per calibration block, C1-C10,
M1 at the adopted tolerance (Expert ruling 2), the per-cycle captures (e_b, price_b, o1, o2 with its dual split,
Delta P / Q / V omega-RMS and max, the interface V upper-limit multipliers), the run records, and the tables.
Every definition is the spec's `decision_rule.operational`; the computations IMPORT (never copy) the step 4 evaluator
(`Run`, `capture_scenario_mismatch`, `settling_bar`, `v_base`, `m1`) and the post-diagnostics (`channel_parts`).
This process never solves: SolveProfileGuard(permitted=()) installed before any project import and verified at 0.

Usage (canonical interpreter):
  p515_s56_step4_calib_evaluate.py --spec <v3 spec>                  the evaluation (JSON into the spec's main dir)
  p515_s56_step4_calib_evaluate.py --dry-run-step4 --out <scratch>   code-path dry run on step 4 main V-1 (spec v2);
                                                                     output to a scratch path only
Objective convention (every value): EUR of total discounted cost; Q_b = settlement excluded in full (V-1 primary; pin
excluded, salvage separate); e_b = Q_b - w_b C_central; price_b = -sum_agents(b) w settlement_total; o1 = e_b - price_b;
o2 = e_b - price_b + dual_b (TSO-side expectation P/Q/V + omega-weighted deviation P/Q/V dual terms).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 4 calibration evaluation (zero-solve)').install()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s56_step4_main as L  # noqa: E402
import p515_s56_step4_main_evaluate as E  # noqa: E402
import p515_s56_step4_postdiag as PD  # noqa: E402

PHASE = 'calib'
GiB = 1024.0 ** 3
C_ROUND = 64
U_ROUND = Fraction(1, 2 ** 53)
FLOOR_EUR = Fraction(1, 10 ** 6)
C6_REL = 1e-3
C6_FRACTION = 0.90
ABS_THRESHOLDS = (1e-8, 1e-6, 1e-4, 1e-2)
AT_LIMIT_REL = 1e-6
TABLE_EVERY = 20
INST_BLOCKS = {'A': ['2025|Winter'], 'B': ['2025|Summer', '2035|Summer']}
CALIBRATED = {'A': '2025|Winter', 'B': '2035|Summer'}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 step4 calib eval] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _fin(x):
    return x if (x is None or not isinstance(x, float) or math.isfinite(x)) else repr(x)


def write_new(path, obj):
    with open(path, 'x', encoding='utf-8') as handle:
        GRIO.dump(obj, handle, indent=1, default=GRIO.json_default_item)


# ======================================================================================================================
#  per-block series
# ======================================================================================================================
def block_series(run, bk, wc):
    """{cycle: row} with e_b, price, the dual split, o1, o2, and Delta P/Q/V omega-RMS and max (post hoc)."""
    vbT = E.v_base(run)
    parts = PD.channel_parts(run, bk, vbT)
    q = run.q_series(bk)
    out = {}
    for ln in run.cap:
        k = ln['cycle']
        if k not in q:
            continue
        p = parts[k]
        e = q[k] - wc
        mm = E.capture_scenario_mismatch(run, ln, bk)
        out[k] = {'cycle': k, 'Q_b': q[k], 'e_b': e, 'price_b': p['price'], 'dual_b': p['dual'],
                  'exp_p': p['exp_p'], 'exp_q': p['exp_q'], 'exp_v': p['exp_v'], 'dev_p': p['dev_p'],
                  'dev_q': p['dev_q'], 'dev_v': p['dev_v'], 'o1': e - p['price'], 'o2': e - p['price'] + p['dual'],
                  'dP_omega_rms_MW': mm['p']['omega_rms'], 'dP_max_MW': mm['p']['max'],
                  'dQ_omega_rms_MVAr': mm['q']['omega_rms'], 'dQ_max_MVAr': mm['q']['max'],
                  'dV_omega_rms_kV': mm['v']['omega_rms'], 'dV_max_kV': mm['v']['max'],
                  'tail_depth': k in run.tail_cycles}
    return out, vbT


def decision(series, tail_cycles, cap, cycles_run, allowance, k_old, w_b, sum12):
    ks = sorted(series)
    out = {'allowance_eur': allowance, 'cap': cap, 'cycles_run': cycles_run, 'K_old': k_old}
    if cycles_run != cap:
        out.update({'evaluable': False, 'why': f'the run stopped at cycle {cycles_run} < cap {cap}'})
    else:
        out['evaluable'] = True
        ok_from = None   # smallest k0 such that every cycle k in [k0, cap] has |e| <= allowance
        for k in reversed(ks):
            if abs(series[k]['e_b']) <= allowance:
                ok_from = k
            else:
                break
        cand = [k for k in tail_cycles if ok_from is not None and k >= ok_from]
        k_in = min(cand) if cand else None
        out['first_cycle_of_the_final_within_allowance_stretch'] = ok_from
        out['k_in'] = k_in
        out['cap_minus_k_in'] = (cap - k_in) if k_in is not None else None
        out['enters_and_stays'] = k_in is not None and cap - k_in >= 20
        if k_in is not None:
            rng = [k for k in ks if k_in <= k <= cap]
            out['measured_level_dP_omega_rms_MW'] = max(series[k]['dP_omega_rms_MW'] for k in rng)
            out['measured_level_cycle'] = max(rng, key=lambda k: series[k]['dP_omega_rms_MW'])
            out['beside_max_dQ_omega_rms_MVAr'] = max(series[k]['dQ_omega_rms_MVAr'] for k in rng)
            out['beside_max_dV_omega_rms_kV'] = max(series[k]['dV_omega_rms_kV'] for k in rng)
            out['beside_max_abs_dP_MW'] = max(series[k]['dP_max_MW'] for k in rng)
            out['beside_max_abs_dQ_MVAr'] = max(series[k]['dQ_max_MVAr'] for k in rng)
            out['beside_max_abs_dV_kV'] = max(series[k]['dV_max_kV'] for k in rng)
        # beside: the same rule on tail-depth cycles only (not deciding)
        tail_ok = None
        for k in reversed([k for k in ks if k in set(tail_cycles)]):
            if abs(series[k]['e_b']) <= allowance:
                tail_ok = k
            else:
                break
        out['beside_tail_depth_only_k_in'] = tail_ok
    after = [k for k in ks if k >= k_old]
    if after:
        kmin = min(after, key=lambda k: abs(series[k]['e_b']))
        out['min_abs_e_b_k_ge_K_old'] = abs(series[kmin]['e_b'])
        out['min_abs_e_b_cycle'] = kmin
        out['R_eff_eur'] = 4.0 * abs(series[kmin]['e_b']) * sum12 / w_b
    K = ks[-1]
    out['terminal'] = {'cycle': K, 'e_b': series[K]['e_b'], 'o1': series[K]['o1'], 'o2': series[K]['o2'],
                       'price_b': series[K]['price_b'], 'dual_b': series[K]['dual_b'],
                       'dP_omega_rms_MW': series[K]['dP_omega_rms_MW']}
    out['terminal_settling_bar_e_b'] = E.settling_bar({k: series[k]['e_b'] for k in ks}, tail_cycles, W=20, h=3)
    out['terminal_settling_bar_o1'] = E.settling_bar({k: series[k]['o1'] for k in ks}, tail_cycles, W=20, h=3)
    out['terminal_settling_bar_o2'] = E.settling_bar({k: series[k]['o2'] for k in ks}, tail_cycles, W=20, h=3)
    return out


def table(series, k_in, cap):
    ks = sorted(series)
    pick = sorted({k for k in ks if k % TABLE_EVERY == 0} | ({k_in} if k_in else set()) | {ks[-1]})
    cols = ('e_b', 'o1', 'o2', 'price_b', 'dual_b', 'exp_p', 'exp_q', 'exp_v', 'dev_p', 'dev_q', 'dev_v',
            'dP_omega_rms_MW', 'dQ_omega_rms_MVAr', 'dV_omega_rms_kV', 'dP_max_MW', 'tail_depth')
    return [{'cycle': k, 'row_kind': ('k_in' if k == k_in else ('cap' if k == cap else 'every 20')),
             **{c: series[k][c] for c in cols}} for k in pick]


# ======================================================================================================================
#  M1 at the adopted tolerance
# ======================================================================================================================
def m1_adopted(run):
    per = []
    for ln, r, rr in zip(run.cap, run.rows, run.resp):
        pa = ln['q_blocks']['per_agent']
        F = Fraction
        T = sum((F(v['weight']) * F(v['settlement_total_unweighted']) for v in pa.values()), F(0))
        S_T = sum((abs(F(v['weight']) * F(v['settlement_total_unweighted'])) for v in pa.values()), F(0))
        pr = F(r['priced_residual_price_part'])
        b_i = max(C_ROUND * U_ROUND * (S_T + abs(pr)), FLOOR_EUR)
        D = sum((F(v['weight']) * (F(v['settlement_total_unweighted']) - F(v['settlement_contracted_unweighted']))
                 for v in pa.values()), F(0))
        gc, g = F(r['gross_operational_cost_contracted_excluded']), F(r['gross_operational_cost'])
        S_ii = sum(((abs(F(v['settlement_excluded_in_full'])) + abs(F(v['contracted_excluded']))
                     + abs(F(v['weight']) * F(v['settlement_total_unweighted']))
                     + abs(F(v['weight']) * F(v['settlement_contracted_unweighted']))) for v in pa.values()), F(0))
        b_ii = max(C_ROUND * U_ROUND * (S_ii + abs(gc) + abs(g)), FLOOR_EUR)
        Dd = sum((F(v['weight']) * (F(v['settlement_total_unweighted']) - F(v['settlement_contracted_unweighted']))
                  for lab, v in pa.items() if lab.startswith('DSO')), F(0))
        S_d = sum((abs(F(v['weight']) * (F(v['settlement_total_unweighted']) - F(v['settlement_contracted_unweighted'])))
                   for lab, v in pa.items() if lab.startswith('DSO')), F(0))
        cov = rr.get('covariance_dso_weighted')
        d_dso = abs(Dd - F(cov)) if cov is not None else None
        b_dso = max(C_ROUND * U_ROUND * (S_d + abs(F(cov))), FLOOR_EUR) if cov is not None else None
        vp = ln.get('voltage_pin') or {}
        pin_ok = (vp.get('total') == 0.0 and bool(vp.get('per_agent')) and all(
            a.get('weighted') == 0.0 and a.get('pin_constructed') is False for a in vp['per_agent'].values()))
        e = {'cycle': ln['cycle'], 'i_abs': float(abs(pr + T)), 'i_bound': float(b_i),
             'i_ratio': float(abs(pr + T) / b_i), 'ii_abs': float(abs((gc - g) - D)), 'ii_bound': float(b_ii),
             'ii_ratio': float(abs((gc - g) - D) / b_ii),
             'dso_abs': float(d_dso) if d_dso is not None else None,
             'dso_ratio': float(d_dso / b_dso) if d_dso is not None else None, 'pin_ok': pin_ok}
        e['held'] = (e['i_ratio'] <= 1.0 and e['ii_ratio'] <= 1.0 and e['dso_ratio'] is not None
                     and e['dso_ratio'] <= 1.0 and pin_ok)
        per.append(e)
    ev = E.m1(run)   # gauge (iii) and the terminal pin from the step 4 evaluator, unchanged
    gauge_ok = not ev['gauge_failures_first_10']
    pin_terminal = ev['v1_terminal_voltage_pin_total']
    return {'n_cycles': len(per), 'n_failed_cycles': sum(not e['held'] for e in per),
            'failed_cycles_first_10': [e for e in per if not e['held']][:10],
            'worst_ratio': {x: max((e[x] for e in per if e[x] is not None), default=None)
                            for x in ('i_ratio', 'ii_ratio', 'dso_ratio')},
            'worst_abs': {x: max((e[x] for e in per if e[x] is not None), default=None)
                          for x in ('i_abs', 'ii_abs', 'dso_abs')},
            'gauge_worst_ratio': ev['gauge_worst_ratio'], 'gauge_held': gauge_ok,
            'terminal_record_voltage_pin_total': pin_terminal,
            'held': (all(e['held'] for e in per) and len(per) == len(run.rows) and gauge_ok and pin_terminal == 0.0),
            'tolerance': ('|a - b| <= max(64 u sum|summands| over both sides, 1e-6 EUR), u = 2^-53 (Expert ruling 2); '
                          'gauge 1e-9 x max|mu|; pin == 0.0')}


# ======================================================================================================================
#  interface V upper-limit multipliers
# ======================================================================================================================
def vmult_side_stats(line, header, side):
    blocks = [tuple(b) for b in header['blocks']]
    arr = line[side]
    ms, margins, comps, raw_neg = [], [], [], {a: 0 for a in ('row_dual', 'zU_vmag_sqr', 'zU_vmag', 'zU_e')}
    raw_n = {a: 0 for a in raw_neg}
    for bi, (node, year, day) in enumerate(blocks):
        vmax = header['per_block'][f'{side}|{node}|{year}|{day}']['v_max_pu']
        for si in range(len(header['scenarios'])):
            for t in range(header['n_periods']):
                row = arr['row_dual'][bi][si][t]
                zs = arr['zU_vmag_sqr'][bi][si][t]
                zv = arr['zU_vmag'][bi][si][t]
                ze = arr['zU_e'][bi][si][t]
                vsq = arr['vmag_sqr'][bi][si][t]
                ee = arr['e'][bi][si][t]
                v = math.sqrt(max(vsq, 0.0))
                c = {'row': abs(row or 0.0), 'vmag_sqr': abs(zs or 0.0),
                     'vmag': abs(zv or 0.0) / (2.0 * v) if v > 0 else 0.0,
                     'e': abs(ze or 0.0) / (2.0 * abs(ee)) if ee else 0.0}
                for a, x in (('row_dual', row), ('zU_vmag_sqr', zs), ('zU_vmag', zv), ('zU_e', ze)):
                    if x is not None:
                        raw_n[a] += 1
                        raw_neg[a] += x < 0
                ms.append(sum(c.values()))
                comps.append(max(c, key=c.get))
                margins.append((vmax - v, AT_LIMIT_REL * max(1.0, vmax)))
    n = len(ms)
    mx = max(ms) if ms else 0.0
    nz = [m >= C6_REL * mx and mx > 0 for m in ms]
    at = [mg <= tol for mg, tol in margins]
    dom = {}
    for c, z in zip(comps, nz):
        if z:
            dom[c] = dom.get(c, 0) + 1
    srt = sorted(ms)
    return {'n_entries': n, 'max_m': mx, 'median_m': srt[n // 2] if n else None, 'min_m': srt[0] if n else None,
            'nonzero_rel_fraction': sum(nz) / n if n else None,
            'abs_threshold_fractions': {str(t): sum(m >= t for m in ms) / n for t in ABS_THRESHOLDS} if n else None,
            'at_limit_by_margin_fraction': sum(at) / n if n else None,
            'joint': {'nonzero_and_at_limit': sum(a and b for a, b in zip(nz, at)),
                      'nonzero_not_at_limit': sum(a and not b for a, b in zip(nz, at)),
                      'zero_and_at_limit': sum((not a) and b for a, b in zip(nz, at)),
                      'zero_not_at_limit': sum((not a) and (not b) for a, b in zip(nz, at))},
            'dominant_component_among_nonzero': dom,
            'raw_negative_count': raw_neg, 'raw_present_count': raw_n,
            'min_margin_pu': min(m for m, _t in margins) if margins else None,
            'max_margin_pu': max(m for m, _t in margins) if margins else None}


def vmult(run):
    p = os.path.join(run.ev, H.S56_INTERFACE_V_MULTIPLIER_FILE)
    if not os.path.exists(p):
        return {'present': False}
    lines = E._jsonl(p)
    header, body = lines[0], [x for x in lines[1:] if not x.get('header')]
    per_cycle = []
    for ln in body:
        if not ln.get('captured'):
            per_cycle.append({'cycle': ln['cycle'], 'captured': False, 'error': ln.get('error')})
            continue
        s = {side: vmult_side_stats(ln, header, side) for side in ('tso', 'dso')}
        per_cycle.append({'cycle': ln['cycle'], 'captured': True,
                          **{f'{side}_{k}': s[side][k] for side in s for k in (
                              'nonzero_rel_fraction', 'at_limit_by_margin_fraction', 'max_m', 'median_m')}})
    term = body[-1]
    terminal = {side: vmult_side_stats(term, header, side) for side in ('tso', 'dso')} if term.get('captured') else None
    return {'present': True, 'file_sha256': _sha(p), 'n_lines': len(body), 'header_per_block': header['per_block'],
            'all_captured': all(x.get('captured') for x in body), 'terminal_cycle': term['cycle'],
            'terminal': terminal, 'per_cycle': per_cycle}


# ======================================================================================================================
#  the evaluation
# ======================================================================================================================
def evaluate_run(spec, inst, run, cap, launcher_rec):
    blocks = spec['ii_central_and_allowance']['blocks']
    sum12 = spec['ii_central_and_allowance']['sum_12_w']
    k_old = spec['decision_rule']['K_old'][inst]
    rows = run.rows
    out = {'eval_dir': os.path.relpath(run.ev, REPO), 'exit_code_txt': run.exit_txt, 'cycles_run': len(rows),
           'cap': cap, 'reached_cap': len(rows) == cap, 'tail_depth_cycles': run.tail_cycles,
           'n_tail_depth_cycles': len(run.tail_cycles), 'blocks': {}}
    rec = run.rec
    out['record'] = {k: rec.get(k) for k in ('status', 'certified_cost', 'certification', 'cycles_run',
                                             'stopped_by_trajectory', 'terminal_ratios_per_channel', 'rule_ten',
                                             'solve_profile', 'peak_rss', 'local_solve_failures', 'eval_key')}
    out['record']['s56_certification_stop_summary'] = rec.get('s56_certification_stop_summary')
    out['record']['s56_interface_v_multiplier_summary'] = rec.get('s56_interface_v_multiplier_summary')
    walls = [r.get('cycle_wall_s') for r in rows if r.get('cycle_wall_s') is not None]
    out['mean_cycle_wall_s'] = sum(walls) / len(walls) if walls else None
    out['sum_cycle_wall_s'] = sum(walls) if walls else None
    out['non_optimal_blocks_total'] = sum(len(r.get('non_optimal_blocks') or []) for r in rows)
    out['local_solves_not_ok_cycles'] = [r['cycle'] for r in rows if r.get('local_solves_ok') is False]
    out['rho_freeze_active_first_cycle'] = next((r['cycle'] for r in rows if r.get('rho_freeze_active')), None)
    out['boyd_all_pass_cycles'] = sum(1 for r in rows if r.get('boyd_all_pass'))
    out['launcher'] = {k: (launcher_rec or {}).get(k) for k in ('prefix_gate_post_run', 'monitor', 'result_summary',
                                                                'eval_key_equals_step4_main_v1', 'wall_s', 'error')}
    for bk in INST_BLOCKS[inst]:
        b = blocks[bk]
        wc = b['w_b_C_central_eur']
        series, vbT = block_series(run, bk, wc)
        brec = {'calibrated': b['calibrated'], 'w_b': b['w_b'], 'w_b_C_central_eur': wc,
                'C_central_status': 'calibrated' if b['calibrated'] else 'MERGED-ONLY (reported, not deciding)',
                'v_base_kv': vbT, 'series': [series[k] for k in sorted(series)]}
        if b['calibrated']:
            d = decision(series, run.tail_cycles, cap, len(rows), b['allowance_eur'], k_old, b['w_b'], sum12)
            brec['decision'] = d
            brec['table'] = table(series, d.get('k_in'), cap)
        else:
            brec['table'] = table(series, None, cap)
            ks = sorted(series)
            brec['terminal_settling_bar_e_b'] = E.settling_bar({k: series[k]['e_b'] for k in ks}, run.tail_cycles,
                                                               W=20, h=3)
        out['blocks'][bk] = brec
    out['M1_adopted'] = m1_adopted(run)
    out['v_multipliers'] = vmult(run)
    return out


def predictions(spec, res):
    A, B = res['instances'].get('A'), res['instances'].get('B')
    out = {}

    def dec(inst):
        r = res['instances'].get(inst)
        return (r or {}).get('blocks', {}).get(CALIBRATED[inst], {}).get('decision') if r else None

    def series_at(inst, bk, k):
        r = res['instances'].get(inst)
        if not r:
            return None
        s = {x['cycle']: x for x in r['blocks'][bk]['series']}
        return s.get(k)

    lg = {i: ((res['instances'].get(i) or {}).get('launcher') or {}) for i in ('A', 'B')}
    c1 = {i: ((lg[i].get('prefix_gate_post_run') or {}).get('held') is True
              and (lg[i].get('monitor') or {}).get('n_mismatches') == 0) for i in ('A', 'B') if res['instances'].get(i)}
    out['C1'] = {'by_instance': c1, 'held': len(c1) == 2 and all(c1.values())}
    dA = dec('A')
    if dA and dA.get('evaluable'):
        eA = series_at('A', '2025|Winter', spec['cap'][PHASE]['A'])
        out['C2'] = {'enters_and_stays': dA['enters_and_stays'], 'k_in': dA.get('k_in'),
                     'abs_e_b_cap': abs(eA['e_b']) if eA else None,
                     'held': bool(dA['enters_and_stays'] and dA['k_in'] <= 100 and eA and abs(eA['e_b']) <= 150.0)}
    else:
        out['C2'] = {'held': None, 'why': 'A not evaluable'}
    dB = dec('B')
    out['C3'] = {'recorded_outcome': (None if not (dB and dB.get('evaluable')) else
                                      ('enters and stays' if dB['enters_and_stays'] else 'does not enter and stay')),
                 'k_in': (dB or {}).get('k_in')}
    c4 = {}
    for inst in ('A', 'B'):
        s = series_at(inst, CALIBRATED[inst], spec['cap'][PHASE][inst])
        if s is not None:
            c4[CALIBRATED[inst]] = {'price_b_cap': s['price_b'], 'held': abs(s['price_b']) <= 50.0}
    out['C4'] = {'blocks': c4, 'held': len(c4) == 2 and all(v['held'] for v in c4.values())}
    c5 = {}
    for inst, (lo, hi) in (('A', (18.0, 26.0)), ('B', (38.0, 52.0))):
        r = res['instances'].get(inst)
        if not r:
            continue
        rss = ((r['record'].get('peak_rss') or {}).get('child_python_process_ru_maxrss'))
        ipopt = ((r['record'].get('peak_rss') or {}).get('solver_subprocesses_max_ru_maxrss'))
        c5[inst] = {'peak_child_rss_gib': rss / GiB if rss else None, 'ipopt_max_rss_gib': ipopt / GiB if ipopt else None,
                    'mean_cycle_wall_s': r['mean_cycle_wall_s'], 'range_s': [lo, hi],
                    'held': bool(rss and rss / GiB <= 4.0 and r['mean_cycle_wall_s'] is not None
                                 and lo <= r['mean_cycle_wall_s'] <= hi)}
    out['C5'] = {'by_instance': c5, 'held': len(c5) == 2 and all(v['held'] for v in c5.values())}
    c6, c9 = {}, {}
    for inst in ('A', 'B'):
        r = res['instances'].get(inst)
        vm = (r or {}).get('v_multipliers') or {}
        if not vm.get('terminal'):
            continue
        for side in ('tso', 'dso'):
            t = vm['terminal'][side]
            c6[f'{inst}|{side}'] = {'nonzero_rel_fraction': t['nonzero_rel_fraction'],
                                    'held': t['nonzero_rel_fraction'] >= C6_FRACTION}
            c9[f'{inst}|{side}'] = {'at_limit_fraction': t['at_limit_by_margin_fraction'],
                                    'held': 0.10 <= t['at_limit_by_margin_fraction'] <= 0.40}
    out['C6'] = {'by_instance_side': c6, 'held': len(c6) == 4 and all(v['held'] for v in c6.values())}
    sA = series_at('A', '2025|Winter', spec['cap'][PHASE]['A'])
    out['C7'] = {'o1_cap': sA['o1'] if sA else None, 'range': [40.0, 110.0],
                 'held': bool(sA and 40.0 <= sA['o1'] <= 110.0)}
    out['C8'] = {'o2_cap': sA['o2'] if sA else None, 'range': [80.0, 130.0],
                 'held': bool(sA and 80.0 <= sA['o2'] <= 130.0)}
    out['C9'] = {'by_instance_side': c9, 'held': len(c9) == 4 and all(v['held'] for v in c9.values())}
    if dA and dA.get('enters_and_stays'):
        lv = dA['measured_level_dP_omega_rms_MW']
        out['C10'] = {'level_MW': lv, 'range': [3e-5, 2.5e-4], 'held': 3e-5 <= lv <= 2.5e-4}
    else:
        out['C10'] = {'held': None, 'why': 'not evaluated: A does not enter and stay (or not evaluable)'}
    return out


def verdict(spec, res):
    ds = {}
    for inst in ('A', 'B'):
        r = res['instances'].get(inst)
        if r:
            ds[CALIBRATED[inst]] = r['blocks'][CALIBRATED[inst]]['decision']
    if len(ds) == 2 and all(d.get('evaluable') for d in ds.values()) and all(d['enters_and_stays']
                                                                             for d in ds.values()):
        level = max(d['measured_level_dP_omega_rms_MW'] for d in ds.values())
        return {'outcome': 'both calibration blocks enter and stay',
                'proposal': (f'added stopping criterion (proposal only, no production change): interface Delta P '
                             f'omega-RMS <= {level!r} MW (the larger of the two measured levels)'),
                'proposed_level_MW': level, 'per_block': ds}
    return {'outcome': 'R not demonstrated at this stopping depth',
            'per_block': {bk: {k: d.get(k) for k in ('evaluable', 'why', 'k_in', 'enters_and_stays', 'R_eff_eur',
                                                     'min_abs_e_b_k_ge_K_old', 'min_abs_e_b_cycle', 'terminal')}
                          | {'terminal_settling_bar_e_b': {k: d['terminal_settling_bar_e_b'].get(k)
                                                          for k in ('S_b', 'label', 'valid', 'Q_hat_inf', 'H')}}
                          for bk, d in ds.items()},
            'note': 'Nothing touches R.'}


def run_main(spec_path):
    spec, spec_sha, spec_rel = None, None, None
    import p515_s56_step4_calib as C
    spec, spec_sha, spec_rel = C.load_spec(spec_path)
    L.SPEC_ROOTS = spec['roots']['campaign_roots']
    res = {'schema': 'p515_s56_step4_calib_evaluation_v1', 'utc': _utc(), 'spec': {'path': spec_rel,
                                                                                   'sha256': spec_sha},
           'script': SCRIPT, 'script_sha256': _sha(SCRIPT), 'objective_convention': spec['objective_convention'],
           'definitions': spec['decision_rule']['operational'], 'instances': {}, 'errors': {}}
    for inst in ('A', 'B'):
        lpath = os.path.join(C.CALIB_DIR, f"{spec['roots']['root_tag']}_{PHASE}_{inst}_launcher.json")
        if not os.path.exists(lpath):
            res['errors'][inst] = 'no launcher record (not run)'
            continue
        try:
            lrec = E._load(lpath)
            run = E.Run(spec, PHASE, inst, 'V1')
            res['instances'][inst] = evaluate_run(spec, inst, run, int(spec['cap'][PHASE][inst]), lrec)
            res['instances'][inst]['launcher_record'] = {'path': os.path.relpath(lpath, REPO), 'sha256': _sha(lpath)}
        except Exception:  # noqa: BLE001
            res['errors'][inst] = traceback.format_exc()
    res['predictions'] = predictions(spec, res)
    res['decision_rule_verdict'] = verdict(spec, res)
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    path = os.path.join(C.CALIB_DIR, f'calib_evaluation_{_stamp()}.json')
    write_new(path, res)
    _log(f"-> {path}; errors {list(res['errors'])}; verdict {res['decision_rule_verdict']['outcome']}; guard "
         f"{GUARD.counts}")
    return 0 if not res['errors'] and not res['guard']['verify_failures'] else 2


def dry_run_step4(out):
    """Code paths on step 4 main V-1 (spec v2): no launcher record, no multiplier file, caps 300 (not reached)."""
    spec2, _s, _r = L._spec(os.path.join('data', 'SRP1', 'Results', 'P515S56', 'step4_main_v2',
                                         'frozen_step4_main_spec_v2_7ca210fb.json'))
    import p515_s56_step4_calib as C
    spec3, _s3, _r3 = C.load_spec(os.path.join(C.CALIB_DIR, sorted(
        f for f in os.listdir(C.CALIB_DIR) if f.startswith(C.SPEC_PREFIX))[0]))
    res = {'instances': {}}
    for inst in ('A', 'B'):
        run = E.Run(spec2, 'main', inst, 'V1')
        res['instances'][inst] = evaluate_run(spec3, inst, run, 300, None)
    res['predictions'] = predictions(spec3, res)
    res['decision_rule_verdict'] = verdict(spec3, res)
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    write_new(out, res)
    _log(f'dry run -> {out}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec')
    ap.add_argument('--dry-run-step4', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.dry_run_step4 and a.out:
        return dry_run_step4(a.out)
    if a.spec:
        return run_main(a.spec)
    ap.error('--spec S | --dry-run-step4 --out <scratch path>')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
