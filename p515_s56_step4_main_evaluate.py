"""P5.15 S56 validation step 4 MAIN RUN -- zero-solve evaluation of the frozen prediction set (spec v2 7ca210fb = spec
v1 3dfb2889 verbatim except roots): must-hold M1-M5, recorded P3-P9 and the 'also record' items, each by the spec's
`vi_predictions.operational` definition. Reads the run artifacts only; this process never solves
(SolveProfileGuard(permitted=()) for the whole process, verified at 0).

Interpretations the operational text leaves open are named in INTERPRETATIONS and written into the output.

Usage: <canonical python> p515_s56_step4_main_evaluate.py --spec S [--out PATH]
Objective convention: EUR of total discounted cost. Coupled arms: Q = settlement_excluded_in_full (primary); pinned
arms: Q = contracted_excluded. C_central: step 3 gross EUR/day; e_b = Q_b - w_b C_central; D_b = w_b C_central - Q_pin,b.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 4 main evaluation (this process never solves)').install()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402

import p515_s44_campaign_harness as H  # noqa: E402
import p515_s56_step4_main as L  # noqa: E402  (launcher: roots, labels, helpers; never solves)
import p515_s56_step4_smoke as SM  # noqa: E402

GiB = 1024.0 ** 3
REL_TOL = 1e-9
M4_REL = 1e-12
INTERPRETATIONS = {
    'tail_depth_cycle': 'spec iii: round k of network_ipopt_solve_records.jsonl (final file; the append file when the '
                        'final one is absent) has >= 1 record and every record compl_inf_tol_in_force == 1e-6',
    'S_b_one_sign': ('"valid only with >= 6 steps of one sign": every step of the window (excluding none) has the same '
                     'sign and there are >= 6 steps; steps are taken only between consecutive cycle numbers both in the '
                     'window set'),
    'S_b_heldout_prediction': ('Q_hat(k0 + m) = Q(k0) + delta_k0 x sum_{j=1..m} r^j, k0 the last fitted cycle, delta_k0 '
                               'its observed step, r from the fit on the window minus its last h = 3 cycles (the same '
                               'form as Q_hat_inf = Q(K) + delta_K r / (1 - r))'),
    'omega_rms': ('omega-RMS of dX_s over a block (year, day): sqrt(sum_s omega_s x mean over the block\'s nodes and '
                  'periods of dX_s^2); max = max over nodes, scenarios and periods of |dX_s|'),
    'v_base_kv': ('v_b per node = capture expectation TSO v (terminal line) / interface_voltage_terminal tso_pu, per '
                  'period; checked constant to rel 1e-9 over periods; if the ratio is ~1 the capture v is in pu and '
                  'v_b = 1 is used (recorded)'),
    'P6_wall': 'mean of per_cycle_record cycle_wall_s over cycles 1..cycles(arm)',
    'M3_float_repr': 'json.dumps(sort_keys=True) after removing the excluded keys recursively; exact string equality',
}


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _strip(obj, keys):
    if isinstance(obj, dict):
        return {k: _strip(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip(v, keys) for v in obj]
    return obj


def _canon(obj, keys):
    return json.dumps(_strip(obj, keys), sort_keys=True)


def eval_dir(root, label):
    ev = os.path.join(root, 'evals')
    hits = [d for d in sorted(os.listdir(ev)) if d.endswith('_' + H._sanitize_id(label))]
    if len(hits) != 1:
        raise RuntimeError(f'expected one eval dir for {label} under {ev}: {hits}')
    return os.path.join(ev, hits[0])


def solve_records(ev):
    p = os.path.join(ev, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE)
    if os.path.exists(p):
        return _jsonl(p), H.NETWORK_IPOPT_SOLVE_RECORDS_FILE
    return _jsonl(os.path.join(ev, H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE)), H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE


def probabilities(ev):
    ms = _load(os.path.join(ev, H.MULTISCENARIO_TERMINAL_FILE))
    out = {}
    for b in ms['blocks'].values():
        if b.get('kind') == 'TSO':
            out[f"{b['year']}|{b['day']}"] = {s: v['probability'] for s, v in b['per_scenario_costs'].items()}
    return out


# ======================================================================================================================
#  settling bar S_b (spec operational 'S_b')
# ======================================================================================================================
def _ols(xs, ys):
    return L._ols(xs, ys)


def settling_bar(series, cycles=None, W=20, h=3):
    """series: {cycle: Q}. cycles: the window set (None = every cycle of the series). Returns the S_b record."""
    cs = sorted(cycles if cycles is not None else series)
    cs = [c for c in cs if c in series and series[c] is not None]
    n = len(cs)
    win = cs[-min(W, n):] if n else []
    out = {'window_cycles': [win[0], win[-1]] if win else None, 'n_window': len(win)}
    if not win:
        out.update({'valid': False, 'S_b': None, 'label': 'no cycles'})
        return out
    K = win[-1]
    steps = [(k, series[k] - series[k - 1]) for k in win[1:] if (k - 1) in win]
    out['n_steps'] = len(steps)
    signs = {math.copysign(1.0, d) for _, d in steps if d != 0.0}
    one_sign = len(steps) >= 6 and len(signs) == 1 and all(d != 0.0 for _, d in steps)
    rng10 = [series[k] for k in win[-10:]]
    fallback = {'valid': False, 'S_b': max(rng10) - min(rng10), 'label': 'range (no validated extrapolation)',
                'range_last_10_cycles': [win[-10:][0], win[-1]]}
    if not one_sign:
        out.update(fallback)
        out['reason'] = 'fewer than 6 steps or steps of both signs'
        return out
    fit_steps = [(k, d) for k, d in steps if k <= win[-1 - h]]
    held_cycles = win[-h:]
    if len(fit_steps) < 2:
        out.update(fallback)
        out['reason'] = 'fewer than 2 fitted steps'
        return out
    a, b = _ols([k for k, _ in fit_steps], [math.log(abs(d)) for _, d in fit_steps])
    r_fit = math.exp(b)
    k0 = fit_steps[-1][0]
    d0 = fit_steps[-1][1]
    held = []
    for k in held_cycles:
        m = k - k0
        pred = series[k0] + d0 * sum(r_fit ** j for j in range(1, m + 1))
        held.append({'cycle': k, 'predicted': pred, 'observed': series[k], 'abs_error': abs(pred - series[k])})
    H_err = max(x['abs_error'] for x in held) if held else 0.0
    a2, b2 = _ols([k for k, _ in steps], [math.log(abs(d)) for _, d in steps])
    r = math.exp(b2)
    out.update({'fit_r_heldout': r_fit, 'heldout': held, 'H': H_err, 'r': r})
    if r >= 1.0 or r_fit >= 1.0:
        out.update(fallback)
        out['reason'] = f'r >= 1 (r {r}, r_fit {r_fit})'
        return out
    dK = series[K] - series[K - 1] if (K - 1) in series else steps[-1][1]
    q_inf = series[K] + dK * r / (1.0 - r)
    out.update({'valid': True, 'label': 'validated extrapolation', 'Q_K': series[K], 'K': K, 'delta_K': dK,
                'Q_hat_inf': q_inf, 'S_b': abs(q_inf - series[K]) + H_err})
    return out


# ======================================================================================================================
#  per-run reading
# ======================================================================================================================
class Run:
    def __init__(self, spec, phase, inst, arm):
        self.phase, self.inst, self.arm = phase, inst, arm
        self.label = L.label_of(phase, arm)
        self.root = L.root_path(phase, inst)
        self.ev = eval_dir(self.root, self.label)
        self.rec = _load(os.path.join(self.ev, 'evaluation_record.json'))
        self.rows = _jsonl(os.path.join(self.ev, 'per_cycle_record.jsonl'))
        self.resp = _jsonl(os.path.join(self.ev, H.PER_CYCLE_RESPONSE_FILE))
        with open(os.path.join(self.ev, 'exit_code.txt')) as f:
            self.exit_txt = int(f.read().strip())
        cap = _jsonl(os.path.join(self.ev, H.S56_CYCLE_CAPTURE_FILE))
        self.cap_header, self.cap = cap[0], [x for x in cap[1:] if not x.get('header')]
        du = _jsonl(os.path.join(self.ev, 'interface_duals_per_cycle.jsonl'))
        self.dual_header, self.duals = du[0], [x for x in du[1:] if not x.get('header')]
        self.solve_recs, self.solve_recs_file = solve_records(self.ev)
        self.tail = L.tail_depth_rounds(self.solve_recs)
        self.tail_cycles = sorted(k for k, v in self.tail.items() if v and k >= 1)
        self.coupled = arm in L.COUPLED
        self.primary = 'settlement_excluded_in_full' if self.coupled else 'contracted_excluded'
        self.probs = probabilities(self.ev)
        self.blocks = [tuple(b) for b in self.cap_header['blocks']]

    def q_series(self, bk, conv=None):
        conv = conv or self.primary
        return {x['cycle']: x['q_blocks']['per_year_day'][bk][conv] for x in self.cap if x.get('captured')}

    def row(self, k):
        return self.rows[k - 1]


def launcher_record(phase, inst):
    p = os.path.join(L.MAIN_DIR, f'{L.ROOT_TAG}_{phase}_{inst}_launcher.json')
    return _load(p) if os.path.exists(p) else None


# ---------------------------------------------------------------------------------------------------------------------
def m1(run):
    """Coupled arms: F1 identities every cycle (operational M1)."""
    per = []
    rc = run.rec.get('recourse_components') or {}
    K = len(run.rows)
    for ln, r, rr in zip(run.cap, run.rows, run.resp):
        pa = ln['q_blocks']['per_agent']
        T = sum(v['weight'] * v['settlement_total_unweighted'] for v in pa.values())
        D = sum(v['weight'] * (v['settlement_total_unweighted'] - v['settlement_contracted_unweighted'])
                for v in pa.values())
        Ddso = sum(v['weight'] * (v['settlement_total_unweighted'] - v['settlement_contracted_unweighted'])
                   for lab, v in pa.items() if lab.startswith('DSO'))
        lhs2 = r['gross_operational_cost_contracted_excluded'] - r['gross_operational_cost']
        # recorded beside, NOT deciding (the frozen M1 test is the rel above): absolute differences and the
        # differences relative to the sum of absolute summands (the floating-point scale of each side)
        absT = sum(abs(v['weight'] * v['settlement_total_unweighted']) for v in pa.values())
        absD = sum(abs(v['weight'] * (v['settlement_total_unweighted'] - v['settlement_contracted_unweighted']))
                   for v in pa.values())
        g_scale = max(abs(r['gross_operational_cost_contracted_excluded']), abs(r['gross_operational_cost']))
        d_i = abs(r['priced_residual_price_part'] + T)
        d_ii = abs(lhs2 - D)
        e_beside = {'i_abs_diff': d_i, 'i_rel_to_sum_abs': d_i / max(absT, 1.0), 'i_T': T, 'i_sum_abs_T': absT,
                    'ii_abs_diff': d_ii, 'ii_rel_to_sum_abs': d_ii / max(absD, 1.0), 'ii_D': D, 'ii_sum_abs_D': absD,
                    'ii_abs_diff_over_gross_scale': d_ii / max(g_scale, 1.0)}
        e = {'cycle': ln['cycle'], 'i_rel': _rel(r['priced_residual_price_part'], -T), 'ii_rel': _rel(lhs2, D),
             'beside': e_beside,
             'ii_dso_rel': (_rel(Ddso, rr['covariance_dso_weighted'])
                            if rr.get('covariance_dso_weighted') is not None else None)}
        if ln['cycle'] == K:
            e['ii_terminal_record_rel'] = _rel(D, rc['interface_settlement_deviation_total'])
            e['i_terminal_record_rel'] = _rel(T, rc['interface_settlement_total'])
        if run.arm == 'V1':
            vp = ln.get('voltage_pin') or {}
            e['v1_pin_ok'] = (vp.get('total') == 0.0 and all(
                a.get('weighted') == 0.0 and a.get('pin_constructed') is False for a in (vp.get('per_agent') or {}).values())
                and bool(vp.get('per_agent')))
        e['held'] = (e['i_rel'] <= REL_TOL and e['ii_rel'] <= REL_TOL and e['ii_dso_rel'] is not None
                     and e['ii_dso_rel'] <= REL_TOL
                     and all(e[x] <= REL_TOL for x in ('ii_terminal_record_rel', 'i_terminal_record_rel') if x in e)
                     and e.get('v1_pin_ok', True))
        per.append(e)
    dh = run.dual_header
    scen = [tuple(x) for x in dh['deviation_channel']['scenarios']]
    gauge_fail, gauge_worst = [], 0.0
    for line in run.duals:
        for field in dh['deviation_channel']['fields']:
            arr = line[field]
            mx = max((abs(x) for blk in arr for sc in blk for x in sc), default=0.0)
            worst = 0.0
            for bi, blk in enumerate(arr):
                node, year, day = run.blocks[bi]
                om = run.probs[f'{year}|{day}']
                for t in range(len(blk[0])):
                    g = sum(om[f'{s_m}_{s_o}'] * blk[si][t] for si, (s_m, s_o) in enumerate(scen))
                    worst = max(worst, abs(g))
            ratio = worst / mx if mx > 0 else 0.0
            gauge_worst = max(gauge_worst, ratio)
            if worst > REL_TOL * mx:
                gauge_fail.append({'cycle': line['cycle'], 'field': field, 'worst': worst, 'max_abs_mu': mx})
    pin_terminal = rc.get('voltage_pin_total') if run.arm == 'V1' else None
    held = (len(per) == K and all(e['held'] for e in per) and not gauge_fail and len(run.duals) == K
            and (run.arm != 'V1' or pin_terminal == 0.0))
    return {'held': held, 'n_cycles': K, 'n_failed_cycles': sum(not e['held'] for e in per),
            'worst': {x: max((e[x] for e in per if e.get(x) is not None), default=None)
                      for x in ('i_rel', 'ii_rel', 'ii_dso_rel')},
            'failed_cycles_first_10': [e for e in per if not e['held']][:10],
            'failed_cycles': [e['cycle'] for e in per if not e['held']],
            'beside_not_deciding_worst': {x: max(e['beside'][x] for e in per) for x in (
                'i_abs_diff', 'i_rel_to_sum_abs', 'ii_abs_diff', 'ii_rel_to_sum_abs', 'ii_abs_diff_over_gross_scale')},
            'gauge_worst_ratio': gauge_worst, 'gauge_failures_first_10': gauge_fail[:10],
            'v1_terminal_voltage_pin_total': pin_terminal}


def m3(run, inst):
    """First 3 cycles bitwise vs the serial smoke counterpart (timing fields excluded in advance)."""
    ex = set(L.M3_EXCLUDED_KEYS)
    sev = L.smoke_eval_dir(inst, run.arm)
    out = {'smoke_eval_dir': os.path.relpath(sev, REPO), 'files': {}}
    # per_cycle_record
    a = run.rows[:L.M3_CYCLES]
    b = _jsonl(os.path.join(sev, 'per_cycle_record.jsonl'))[:L.M3_CYCLES]
    only = sorted(set().union(*[set(x) for x in a]) ^ set().union(*[set(x) for x in b]))
    diffs = []
    for x, y in zip(a, b):
        common = set(x) & set(y)
        for k in sorted(common - ex):
            if _canon(x[k], ex) != _canon(y[k], ex):
                diffs.append({'cycle': x.get('cycle'), 'key': k, 'main': x[k], 'smoke': y[k]})
    out['files']['per_cycle_record.jsonl'] = {'n_main': len(a), 'n_smoke': len(b), 'keys_in_one_only': only,
                                              'n_diffs': len(diffs), 'diffs_first_10': diffs[:10],
                                              'equal': len(a) == len(b) == L.M3_CYCLES and not diffs}
    # interface duals
    a = [x for x in run.duals if x['cycle'] <= L.M3_CYCLES]
    b = [x for x in _jsonl(os.path.join(sev, 'interface_duals_per_cycle.jsonl')) if not x.get('header')][:L.M3_CYCLES]
    d = [x['cycle'] for x, y in zip(a, b) if _canon(x, ex) != _canon(y, ex)]
    out['files']['interface_duals_per_cycle.jsonl'] = {'n_main': len(a), 'n_smoke': len(b), 'differing_cycles': d,
                                                       'equal': len(a) == len(b) == L.M3_CYCLES and not d}
    # capture (coupled arms only)
    if run.coupled:
        ex2 = ex | {'voltage_pin'}
        a = [x for x in run.cap if x['cycle'] <= L.M3_CYCLES]
        b = [x for x in _jsonl(os.path.join(sev, H.S56_CYCLE_CAPTURE_FILE)) if not x.get('header')][:L.M3_CYCLES]
        d = [x['cycle'] for x, y in zip(a, b) if _canon(x, ex2) != _canon(y, ex2)]
        out['files']['s56_cycle_capture.jsonl'] = {'n_main': len(a), 'n_smoke': len(b), 'differing_cycles': d,
                                                   'voltage_pin_listed_not_compared': True,
                                                   'equal': len(a) == len(b) == L.M3_CYCLES and not d}
    # network solve records rounds 0..3 in file order
    a = [x for x in run.solve_recs if x.get('round') is not None and x['round'] <= L.M3_CYCLES]
    b = [x for x in _jsonl(os.path.join(sev, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
         if x.get('round') is not None and x['round'] <= L.M3_CYCLES]
    d = [i for i, (x, y) in enumerate(zip(a, b)) if _canon(x, ex) != _canon(y, ex)]
    out['files']['network_ipopt_solve_records.jsonl'] = {'source_main': run.solve_recs_file, 'n_main': len(a),
                                                         'n_smoke': len(b), 'differing_indices': d[:20],
                                                         'equal': len(a) == len(b) and len(a) > 0 and not d}
    out['held'] = all(v['equal'] for v in out['files'].values())
    return out


def m4(run):
    ck = run.rec.get('s56_cycle_capture_checklist_asserted_before_run') or {}
    sm = run.rec.get('s56_cycle_capture_summary') or {}
    K = run.rec.get('cycles_run')
    worst, bad = 0.0, []
    for ln, r in zip(run.cap, run.rows):
        s = sum(v[run.primary] for v in ln['q_blocks']['per_year_day'].values())
        rel = _rel(s, r['gross_operational_cost'])
        worst = max(worst, rel)
        if rel > M4_REL:
            bad.append({'cycle': ln['cycle'], 'rel': rel})
    uncaptured = [x['cycle'] for x in run.cap if not x.get('captured')]
    held = (bool(ck) and all(v is True for v in ck.values()) and sm.get('ok') is True
            and sm.get('cycle_lines_written') == K and len(run.cap) == K and not uncaptured and not bad)
    return {'held': held, 'checklist_all_true': bool(ck) and all(v is True for v in ck.values()),
            'summary_ok': sm.get('ok'), 'cycle_lines_written': sm.get('cycle_lines_written'), 'cycles_run': K,
            'uncaptured_cycles': uncaptured, 'worst_rel_capture_sum_vs_gross': worst, 'failed_first_10': bad[:10],
            'primary': run.primary}


def m5(run, spec, launcher_rec):
    sp = run.rec.get('solve_profile') or {}
    obs = sp.get('observed') or {}
    per_round = spec['solves']['per_round'][run.inst]
    K = run.rec.get('cycles_run')
    res = next((x for x in (launcher_rec or {}).get('results') or [] if x.get('arm') == run.arm), {})
    lg = (launcher_rec or {}).get('guard') or {}
    out = {'identity_holds': sp.get('identity_holds'), 'base_solves': sp.get('base_solves'),
           'declared_base': per_round * (K + 1), 'observed': obs, 'retry_solves_credited': sp.get('retry_solves_credited'),
           'expected_solves': sp.get('expected_solves'), 'exit_code_parent': res.get('exit_code'),
           'exit_code_txt': run.exit_txt, 'launcher_guard_verify_failures': lg.get('verify_failures'),
           'launcher_guard_counts': lg.get('counts')}
    out['held'] = (sp.get('identity_holds') is True and sp.get('base_solves') == per_round * (K + 1)
                   and obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0 and res.get('exit_code') == 0
                   and run.exit_txt == 0 and launcher_rec is not None and not lg.get('verify_failures'))
    return out


def m2(runs_main, inst):
    """Initialisation identity (operational M2), per instance."""
    g = {a: runs_main[a].rec.get('initialisation_identity') or {} for a in L.ARMS}
    smoke_hex = {}
    for a in L.ARMS:
        p, _ = L.smoke_init_record(inst, a)
        r = _load(p)['record']
        smoke_hex[a] = (r['gross_operational_cost_hex'], r['components_hex'])
    main_hex = {}
    for a in L.ARMS:
        p = os.path.join(REPO, g[a]['path']) if not os.path.isabs(g[a]['path']) else g[a]['path']
        r = _load(p)['record']
        main_hex[a] = (r['gross_operational_cost_hex'], r['components_hex'])
    pinned_group = ('V2', 'pinned_aa_off', 'pinned_aa_on')
    eq3 = len({json.dumps(main_hex[a], sort_keys=True) for a in pinned_group}) == 1
    eq_smoke = all(json.dumps(main_hex[a], sort_keys=True) == json.dumps(smoke_hex[a], sort_keys=True) for a in L.ARMS)
    comps = {a: {'n_compared': g[a].get('n_compared'), 'all_equal': g[a].get('all_equal'),
                 'group': g[a].get('initialisation_identity_group'),
                 'compared_labels': sorted(c.get('label') for c in g[a].get('compared') or [])} for a in L.ARMS}
    v1_labels_ok = all(lab.startswith('x0_V1') for lab in comps['V1']['compared_labels'])
    held = (eq3 and eq_smoke and all(comps[a]['all_equal'] is True and (comps[a]['n_compared'] or 0) >= 1
                                     for a in pinned_group)
            and comps['V1']['group'] == 'voltage_pin_not_constructed' and comps['V1']['all_equal'] is True
            and main_hex['V1'][0] != main_hex['V2'][0] and v1_labels_ok)
    return {'held': held, 'pinned_group_equal_bitwise': eq3, 'equal_to_smoke_hex': eq_smoke,
            'hex': {a: main_hex[a][0] for a in L.ARMS}, 'comparisons': comps, 'v1_compared_only_within_group': v1_labels_ok}


# ---------------------------------------------------------------------------------------------------------------------
def capture_scenario_mismatch(run, ln, bk):
    """Per block (year|day): max and omega-RMS of dP_s, dQ_s, dV_s (coupled arms: expectation + deviation copies)."""
    if not run.coupled:
        return None
    pe, pd = ln['primal_expectation'], ln['primal_deviation']
    scen = [f'{a}_{b}' for a, b in run.cap_header['deviation_scenarios']]
    om = run.probs[bk]
    out = {}
    for kind in ('p', 'q', 'v'):
        mx, acc, n = 0.0, 0.0, 0
        dt, dd = pd['tso'].get(kind), pd['dso'].get(kind)   # V-2: no 'v' deviation kind (scenario-common V)
        for bi, (node, year, day) in enumerate(run.blocks):
            if f'{year}|{day}' != bk:
                continue
            for si, s in enumerate(scen):
                for t in range(len(pe['tso'][kind][bi])):
                    d = ((pe['tso'][kind][bi][t] + (dt[bi][si][t] if dt is not None else 0.0))
                         - (pe['dso'][kind][bi][t] + (dd[bi][si][t] if dd is not None else 0.0)))
                    mx = max(mx, abs(d))
                    acc += om[s] * d * d
            n += len(pe['tso'][kind][bi])
        out[kind] = {'max': mx, 'omega_rms': math.sqrt(acc / n) if n else None,
                     'deviation_kind_present': dt is not None}
    return out


def v_base(run):
    """Interface base kV per node (INTERPRETATIONS v_base_kv)."""
    vt = _load(os.path.join(run.ev, 'interface_voltage_terminal.json'))['entries']
    last = run.cap[-1]['primal_expectation']['tso']['v']
    ratios = {}
    for e in vt:
        bi = run.blocks.index((e['node_id'], int(e['year']), e['day']))
        ratios.setdefault(e['node_id'], []).append(last[bi][e['period']] / e['tso_pu'])
    out = {}
    for node, rs in ratios.items():
        const = max(rs) - min(rs) <= 1e-9 * max(abs(max(rs)), 1.0)
        out[node] = {'v_b_kv': rs[0], 'constant': const, 'min': min(rs), 'max': max(rs)}
    return out


def priced_residual(run, bk, vb):
    """P7: full-channel PR_b(k) and its absolute form A_b(k), per cycle, plus the P-deviation part for validation."""
    meta = {(m['node_id'], m['year'], m['day']): m for m in run.dual_header['metadata']['per_block']}
    scen = [f'{a}_{b}' for a, b in run.dual_header['deviation_channel']['scenarios']]
    om = run.probs[bk]
    duals_by_cycle = {x['cycle']: x for x in run.duals}
    out = {}
    for ln in run.cap:
        k = ln['cycle']
        du = duals_by_cycle.get(k)
        if du is None:
            continue
        pa = ln['q_blocks']['per_agent']
        price = -sum(v['weight'] * v['settlement_total_unweighted'] for lab, v in pa.items()
                     if lab.split('|', 1)[1] == bk)
        exp_part = absf = dev_p = dev_all = 0.0
        pe, pd = ln['primal_expectation'], ln['primal_deviation']
        dv_tso = ln['dual_expectation_v']['tso']
        for bi, (node, year, day) in enumerate(run.blocks):
            if f'{year}|{day}' != bk:
                continue
            m = meta[(node, year, day)]
            w = pa[f'TSO|{year}|{day}']['weight']
            fac_pq = w * m['admm_objective_scale_tso'] / (m['tso_base_mva'] * m['interface_rating_mva'])
            fac_v = w * m['admm_objective_scale_tso'] / (vb[node]['v_b_kv'] ** 2)
            for t in range(len(pe['tso']['p'][bi])):
                dP = pe['tso']['p'][bi][t] - pe['dso']['p'][bi][t]
                dQ = pe['tso']['q'][bi][t] - pe['dso']['q'][bi][t]
                dV = pe['tso']['v'][bi][t] - pe['dso']['v'][bi][t]
                terms = [du['lambda_pf_p_tso'][bi][t] * dP * fac_pq, du['lambda_pf_q_tso'][bi][t] * dQ * fac_pq,
                         dv_tso[bi][t] * dV * fac_v]
                exp_part += sum(terms)
                absf += sum(abs(x) for x in terms)
                for si, s in enumerate(scen):
                    dPs = (pe['tso']['p'][bi][t] + pd['tso']['p'][bi][si][t]) - (pe['dso']['p'][bi][t]
                                                                                 + pd['dso']['p'][bi][si][t])
                    dQs = (pe['tso']['q'][bi][t] + pd['tso']['q'][bi][si][t]) - (pe['dso']['q'][bi][t]
                                                                                 + pd['dso']['q'][bi][si][t])
                    dVs = (pe['tso']['v'][bi][t] + pd['tso']['v'][bi][si][t]) - (pe['dso']['v'][bi][t]
                                                                                 + pd['dso']['v'][bi][si][t])
                    tp = om[s] * du['lambda_dev_p_tso'][bi][si][t] * dPs * fac_pq
                    tq = om[s] * du['lambda_dev_q_tso'][bi][si][t] * dQs * fac_pq
                    tv = om[s] * du['lambda_dev_v_tso'][bi][si][t] * dVs * fac_v
                    dev_p += tp
                    dev_all += tp + tq + tv
                    absf += abs(tp) + abs(tq) + abs(tv)
        out[k] = {'price': price, 'expectation_part': exp_part, 'deviation_part': dev_all, 'deviation_p_part': dev_p,
                  'PR': price + exp_part + dev_all, 'A': abs(price) + absf}
    return out


def p7_validation(run, vb):
    """The P-deviation part summed over every block vs production priced_residual_dual_part, rel 1e-9 every cycle."""
    bks = sorted({f'{y}|{d}' for _, y, d in run.blocks})
    per = {bk: priced_residual(run, bk, vb) for bk in bks}
    worst, bad = 0.0, []
    for r in run.rows:
        k = r['cycle']
        s = sum(per[bk][k]['deviation_p_part'] for bk in bks if k in per[bk])
        rel = _rel(s, r['priced_residual_dual_part'])
        worst = max(worst, rel)
        if rel > REL_TOL:
            bad.append({'cycle': k, 'ours': s, 'production': r['priced_residual_dual_part'], 'rel': rel})
    return {'worst_rel': worst, 'n_failed': len(bad), 'failed_first_5': bad[:5], 'held': not bad}, per


def p6_series(run, ratings):
    """P6 residual from the capture's primal_expectation: sqrt(sum ((x_dso - z_tso) / rating)^2) over p, q."""
    out = {}
    for ln in run.cap:
        pe = ln['primal_expectation']
        acc = 0.0
        for bi, blk in enumerate(run.blocks):
            R = ratings[blk]
            for kind in ('p', 'q'):
                for x, z in zip(pe['dso'][kind][bi], pe['tso'][kind][bi]):
                    acc += ((x - z) / R) ** 2
        out[ln['cycle']] = math.sqrt(acc)
    return out


# ======================================================================================================================
def evaluate_instance(spec, inst, res):
    blocks = spec['ii_central_and_allowance']['blocks']
    sum12 = spec['ii_central_and_allowance']['sum_12_w']
    lrec = launcher_record('main', inst)
    out = {'launcher_record': lrec and {k: lrec.get(k) for k in ('utc_start', 'utc_end', 'wall_s', 'error',
                                                                 'result_summary', 'guard', 'refusals')}}
    runs = {}
    for a in L.ARMS:
        try:
            runs[a] = Run(spec, 'main', inst, a)
        except Exception:  # noqa: BLE001
            res['errors'][f'{inst}|{a}|load'] = traceback.format_exc()
    out['runs'] = {}
    ratings = None
    for a, run in runs.items():
        o = {'eval_dir': os.path.relpath(run.ev, REPO), 'status': run.rec.get('status'),
             'cycles_run': run.rec.get('cycles_run'), 'certification': run.rec.get('certification'),
             'wall_time_s': run.rec.get('wall_time_s'),
             'cycle_wall_s': {'mean': sum(r['cycle_wall_s'] for r in run.rows) / len(run.rows),
                              'min': min(r['cycle_wall_s'] for r in run.rows),
                              'max': max(r['cycle_wall_s'] for r in run.rows),
                              'sum': sum(r['cycle_wall_s'] for r in run.rows)},
             'solve_profile': run.rec.get('solve_profile'), 'local_solve_failures': run.rec.get('local_solve_failures'),
             'non_optimal_blocks_cycles': [r['cycle'] for r in run.rows if r.get('non_optimal_blocks')],
             'rule_ten': run.rec.get('rule_ten'), 'terminal_ratios_per_channel': run.rec.get('terminal_ratios_per_channel'),
             'tail_depth_cycles': {'n': len(run.tail_cycles), 'first': run.tail_cycles[0] if run.tail_cycles else None,
                                   'last': run.tail_cycles[-1] if run.tail_cycles else None,
                                   'list': run.tail_cycles},
             'solve_records_file': run.solve_recs_file}
        try:
            if run.coupled:
                o['M1'] = m1(run)
        except Exception:  # noqa: BLE001
            res['errors'][f'{inst}|{a}|M1'] = traceback.format_exc()
        for name, fn in (('M3', lambda r=run: m3(r, inst)), ('M4', lambda r=run: m4(r)),
                         ('M5', lambda r=run: m5(r, spec, lrec))):
            try:
                o[name] = fn()
            except Exception:  # noqa: BLE001
                res['errors'][f'{inst}|{a}|{name}'] = traceback.format_exc()
        rss = (run.rec.get('peak_rss') or {})
        o['P9'] = {'child_python_peak_rss_gib': (rss.get('child_python_process_ru_maxrss') or 0) / GiB,
                   'ipopt_max_rss_gib': (rss.get('solver_subprocesses_max_ru_maxrss') or 0) / GiB,
                   'held': bool(rss.get('child_python_process_ru_maxrss'))
                   and rss['child_python_process_ru_maxrss'] <= 6 * GiB}
        # P8: tail onset step
        if run.tail_cycles:
            K1 = run.tail_cycles[0]
            if K1 >= 2:
                o['P8'] = {'K1': K1, 'step_total': run.row(K1)['gross_operational_cost']
                           - run.row(K1 - 1)['gross_operational_cost'],
                           'step_per_block': {bk: run.q_series(bk)[K1] - run.q_series(bk)[K1 - 1]
                                              for bk in spec['i_arms_instances_cap']['instances'][inst]['blocks']}}
            else:
                o['P8'] = {'K1': K1, 'note': 'tail onset at cycle 1 (no previous cycle)'}
        else:
            o['P8'] = {'K1': None, 'note': 'no tail onset'}
        # per-block Q trajectories, e_b and settling bars (run's last cycles)
        o['blocks'] = {}
        for bk in spec['i_arms_instances_cap']['instances'][inst]['blocks']:
            b = blocks[bk]
            q = run.q_series(bk)
            K = max(q)
            e = {k: v - b['w_b_C_central_eur'] for k, v in q.items()}
            o['blocks'][bk] = {'calibrated': b['calibrated'], 'w_b_C_central_eur': b['w_b_C_central_eur'],
                               'Q_K': q[K], 'e_b_K': e[K], 'K': K, 'allowance_eur': b['allowance_eur'],
                               'S_b_last_cycles': settling_bar(q),
                               'S_b_tail_depth_cycles': settling_bar(q, run.tail_cycles) if run.tail_cycles else None,
                               'e_b_last_10': {str(k): e[k] for k in sorted(e)[-10:]}}
            if not run.coupled:
                o['blocks'][bk]['settlement_excluded_in_full_K'] = run.q_series(bk, 'settlement_excluded_in_full')[K]
        # dV trajectory
        if run.coupled:
            o['dV_trajectory'] = {'production_v_max_pu': [r.get('scenario_mismatch_v_max_pu') for r in run.rows],
                                  'production_v_rms_pu': [r.get('scenario_mismatch_v_rms_pu') for r in run.rows]}
            o['dV_terminal_per_block'] = {bk: capture_scenario_mismatch(run, run.cap[-1], bk)
                                          for bk in spec['i_arms_instances_cap']['instances'][inst]['blocks']}
        if ratings is None:
            ratings = {(m['node_id'], m['year'], m['day']): m['interface_rating_mva']
                       for m in run.dual_header['metadata']['per_block']}
        out['runs'][a] = o
    # ---- M2
    try:
        out['M2'] = m2(runs, inst)
    except Exception:  # noqa: BLE001
        res['errors'][f'{inst}|M2'] = traceback.format_exc()
    # ---- P3, working level, P7 (V1)
    if 'V1' in runs:
        run = runs['V1']
        try:
            vb = v_base(run)
            v_ok = all(v['constant'] for v in vb.values())
            if all(abs(v['v_b_kv'] - 1.0) < 1e-6 for v in vb.values()):
                vb_note = 'capture v in pu (ratio 1): v_b = 1'
            else:
                vb_note = 'capture v in kV'
            val, per = p7_validation(run, vb)
            out['v_base'] = {'per_node': vb, 'constant': v_ok, 'note': vb_note}
            out['P7_validation_P_deviation_part'] = val
        except Exception:  # noqa: BLE001
            res['errors'][f'{inst}|V1|P7prep'] = traceback.format_exc()
            per = None
        out['P3'], out['P7'], out['working_level'] = {}, {}, {}
        rows_by_k = {r['cycle']: r for r in run.rows}
        for bk in L.CALIBRATED[inst]:
            b = blocks[bk]
            wc, allow = b['w_b_C_central_eur'], b['allowance_eur']
            q = run.q_series(bk)
            e = {k: v - wc for k, v in q.items()}
            T = run.tail_cycles
            p3 = {'allowance_eur': allow, 'tail_depth_cycles_n': len(T)}
            if not T:
                p3.update({'held': None, 'status': 'not evaluable (no tail-depth cycle)',
                           'production_depth_terminal_e_b': e[max(e)]})
            else:
                K = T[-1]
                kmin = min(T, key=lambda k: abs(e[k]))
                sb = settling_bar(q, T)
                pr_k = per[bk][K]['PR'] if per and K in per[bk] else None
                terms = {'S_b': sb.get('S_b'), '|PR_b(K)|': abs(pr_k) if pr_k is not None else None,
                         'D1_tail_depth_floor': b['D1_floor_eur']['tail_depth']}
                dom = max((kk for kk in terms if terms[kk] is not None), key=lambda kk: terms[kk])
                p3.update({'K_last_tail_depth_cycle': K, 'e_b_K': e[K], 'held': abs(e[K]) <= allow,
                           'min_abs_e_b_tail': abs(e[kmin]), 'min_at_cycle': kmin,
                           'R_eff_eur': 4.0 * abs(e[kmin]) * sum12 / b['w_b'], 'S_b_tail': sb,
                           'terms': terms, 'dominant_term': dom,
                           'ratio_e_b_K_over_allowance': abs(e[K]) / allow})
            out['P3'][bk] = p3
            # P7
            if per:
                ks = [k for k in sorted(per[bk]) if k >= 21 and k in q]
                lhs = sum(abs(q[k] + per[bk][k]['PR'] - wc) for k in ks)
                rhs = sum(abs(q[k] - wc) for k in ks)
                out['P7'][bk] = {'n_cycles': len(ks), 'sum_abs_Q_plus_PR_minus_wC': lhs, 'sum_abs_Q_minus_wC': rhs,
                                 'reduction_factor': (rhs / lhs) if lhs > 0 else math.inf,
                                 'held': bool(ks) and lhs <= rhs / 5.0,
                                 'per_cycle_last_10': {str(k): {'e_b': q[k] - wc, 'PR': per[bk][k]['PR'],
                                                                'A': per[bk][k]['A'],
                                                                'e_b_plus_PR': q[k] + per[bk][k]['PR'] - wc}
                                                       for k in ks[-10:]},
                                 'terminal': {'cycle': max(per[bk]), **per[bk][max(per[bk])]}}
            # working level
            if T:
                def m_of(k):
                    r = rows_by_k[k]
                    return max(max(r.get(f'boyd_{c}_primal_ratio') or 0.0, r.get(f'boyd_{c}_dual_ratio') or 0.0)
                               for c in ('v', 'pf', 'ess'))
                levels = sorted({m_of(k) for k in T}, reverse=True)
                best = None
                for s in levels:
                    ks = next((k for k in T if m_of(k) <= s), None)
                    if ks is None:
                        continue
                    if all(abs(e[k]) <= allow for k in T if k >= ks):
                        best = (s, ks)
                        break
                if best is None:
                    out['working_level'][bk] = {'status': 'not attained'}
                else:
                    s, ks = best
                    ln = next(x for x in run.cap if x['cycle'] == ks)
                    out['working_level'][bk] = {'s_star': s, 'k_s_star': ks, 'e_b_at_k': e[ks],
                                                'observables': capture_scenario_mismatch(run, ln, bk),
                                                'A_b': per[bk][ks]['A'] if per and ks in per[bk] else None}
            else:
                out['working_level'][bk] = {'status': 'not evaluable (no tail-depth cycle)'}
    # ---- P5 (V2), M4 effort (V1, V2)
    out['P5'], out['M4_effort'] = {}, {}
    for bk in L.CALIBRATED[inst]:
        b = blocks[bk]
        if 'V2' in runs:
            run = runs['V2']
            q = run.q_series(bk)
            T = run.tail_cycles
            sb = settling_bar(q, T) if T else settling_bar(q)
            lim = sb['Q_hat_inf'] if sb.get('valid') else q[max(q)]
            target = b['w_b'] * (b['C_central_eur_per_day'] + b['delta_pin_eur_per_day'])
            out['P5'][bk] = {'lim_Q_V2': lim, 'lim_kind': 'validated extrapolation' if sb.get('valid')
                             else 'terminal value (no validated extrapolation)',
                             'window': 'tail-depth cycles' if T else 'last cycles (no tail-depth cycle)',
                             'w_b_C_plus_delta_pin': target, 'excess': lim - target, 'S_b_V2': sb.get('S_b'),
                             'held': (lim - target) <= (sb.get('S_b') or 0.0),
                             'caveat': 'local-solution caveat: lim Q_V2 is a local ADMM limit'}
        for a in ('V1', 'V2'):
            if a not in runs:
                continue
            run = runs[a]
            q = run.q_series(bk)
            sb = settling_bar(q, run.tail_cycles) if run.tail_cycles else settling_bar(q)
            lim = sb['Q_hat_inf'] if sb.get('valid') else q[max(q)]
            ks = sorted(q)
            first = None
            for i, k in enumerate(ks):
                if all(abs(q[j] - lim) <= b['allowance_eur'] for j in ks[i:]):
                    first = k
                    break
            wall = (sum(r['cycle_wall_s'] for r in run.rows if r['cycle'] <= first) if first else None)
            out['M4_effort'].setdefault(bk, {})[a] = {'own_limit': lim, 'limit_kind': sb.get('label'),
                                                      'first_cycle_within_allowance_thereafter': first,
                                                      'cumulative_wall_s': wall}
        ef = out['M4_effort'].get(bk, {})
        if all(ef.get(a, {}).get('first_cycle_within_allowance_thereafter') for a in ('V1', 'V2')):
            ef['ratio_cycles_V1_over_V2'] = (ef['V1']['first_cycle_within_allowance_thereafter']
                                             / ef['V2']['first_cycle_within_allowance_thereafter'])
            ef['ratio_wall_V1_over_V2'] = ef['V1']['cumulative_wall_s'] / ef['V2']['cumulative_wall_s']
    # ---- P4 (pinned AA-off; per block of the instance)
    out['P4_D_b'] = {}
    if 'pinned_aa_off' in runs:
        run = runs['pinned_aa_off']
        for bk in spec['i_arms_instances_cap']['instances'][inst]['blocks']:
            q = run.q_series(bk, 'contracted_excluded')
            K = max(q)
            sb = settling_bar(q)
            out['P4_D_b'][bk] = {'w_b_C_central': blocks[bk]['w_b_C_central_eur'], 'Q_pin_K': q[K], 'K': K,
                                 'D_b': blocks[bk]['w_b_C_central_eur'] - q[K], 'bar_S_b': sb.get('S_b'),
                                 'bar_kind': sb.get('label'),
                                 'C_central_status': ('MERGED-ONLY (reported, not calibrated)'
                                                      if bk == '2025|Summer' else 'calibrated block')}
        if 'pinned_aa_on' in runs:
            run2 = runs['pinned_aa_on']
            out['pinned_aa_off_vs_on'] = {}
            for bk in spec['i_arms_instances_cap']['instances'][inst]['blocks']:
                q1, q2 = run.q_series(bk, 'contracted_excluded'), run2.q_series(bk, 'contracted_excluded')
                s1, s2 = settling_bar(q1), settling_bar(q2)
                out['pinned_aa_off_vs_on'][bk] = {'Q_off_K': q1[max(q1)], 'Q_on_K': q2[max(q2)],
                                                  'diff_off_minus_on': q1[max(q1)] - q2[max(q2)],
                                                  'S_off': s1.get('S_b'), 'S_on': s2.get('S_b'),
                                                  'K_off': max(q1), 'K_on': max(q2)}
    # ---- P6
    try:
        if ratings is not None and len(runs) == 4:
            ser = {a: p6_series(runs[a], ratings) for a in L.ARMS}
            Lv = max(min(s.values()) for s in ser.values())
            cyc = {a: next(k for k in sorted(s) if s[k] <= Lv) for a, s in ser.items()}
            wall = {a: sum(r['cycle_wall_s'] for r in runs[a].rows if r['cycle'] <= cyc[a]) / cyc[a] for a in L.ARMS}
            chk = {}
            for a in L.COUPLED:
                worst = max(_rel(ser[a][r['cycle']], r['boyd_pf_r_expectation_part']) for r in runs[a].rows)
                chk[a] = worst
            r1 = cyc['V1'] / cyc['pinned_aa_off']
            r2 = wall['V1'] / wall['pinned_aa_off']
            r3 = cyc['pinned_aa_off'] / cyc['pinned_aa_on']
            out['P6'] = {'L_matched_residual': Lv, 'cycles': cyc, 'mean_wall_s': wall,
                         'cycles_V1_over_off': r1, 'wall_V1_over_off': r2, 'cycles_off_over_on': r3,
                         'cycles_V2_over_off': cyc['V2'] / cyc['pinned_aa_off'],
                         'wall_V2_over_off': wall['V2'] / wall['pinned_aa_off'],
                         'formula_check_vs_production_boyd_pf_r_expectation_part_worst_rel': chk,
                         'min_residual_per_arm': {a: min(s.values()) for a, s in ser.items()},
                         'held': 1.0 <= r1 <= 2.0 and 0.9 <= r2 <= 1.2 and r3 >= 2.0}
    except Exception:  # noqa: BLE001
        res['errors'][f'{inst}|P6'] = traceback.format_exc()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--instances', default='B,A')
    a = ap.parse_args()
    spec, spec_sha, spec_rel = L._spec(a.spec)
    res = {'schema': 'p515_s56_step4_main_evaluation_v1', 'utc': L._utc(), 'spec': {'path': spec_rel,
                                                                                   'sha256': spec_sha},
           'script': SCRIPT, 'script_sha256': L._sha(SCRIPT), 'git_head': L._git('rev-parse', 'HEAD').strip(),
           'objective_convention': spec['objective_convention'], 'interpretations': INTERPRETATIONS,
           'instances': {}, 'errors': {}}
    for inst in [x for x in a.instances.split(',') if x]:
        try:
            res['instances'][inst] = evaluate_instance(spec, inst, res)
        except Exception:  # noqa: BLE001
            res['errors'][f'{inst}'] = traceback.format_exc()
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    path = os.path.join(L.MAIN_DIR, f'step4_main_evaluation_{L._stamp()}.json')
    L.write_json(path, res)
    print(f'evaluation -> {path}; errors {list(res["errors"])}; guard {res["guard"]}')
    return 0 if not res['errors'] else 2


if __name__ == '__main__':
    sys.exit(main())
