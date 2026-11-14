"""P5.15 W126 -- records-only diagnosis of the FROZEN pf PRIMAL RESIDUAL on the F2 challenger re-settling cell.
ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Nothing is built, unpickled or solved; only JSON / JSONL records are read (the large JSONL sidecars streamed line by
line). Every input's sha256 is checked against the campaign manifest that hash-records it; inputs that are git-tracked
must also be clean against HEAD (the per-entry strides and the ESS schedule / ESSO captures are hash-recorded only).

INSTANCE: campaign s53_w118_resettle_r2_f2_challenger (cell spec c7c9aea8), eval key 1fe91e86f11e76af..., candidate key
4032a138... (label f2_challenger, investment year 2030, n5 (0.25, 1.0), n7 (1.0, 3.0), n9 (0, 0), flexibility price x2),
replaying original eval e28de4ac... bitwise through k0 = 152; holds (AA off, tail on, rho frozen) from 153; cap 261.

PRODUCTION FACTS USED (read from shared_resources_planning.py, not re-implemented as solver code):
  * pf is TWO-BLOCK ADMM with NO separate consensus variable: x = DSO interface copy, z = TSO interface copy
    (get_admm_boyd_residual_metrics docstring). Cycle order: DSO solve (against the TSO copy of the previous cycle,
    stride field z_tso_prev) -> TSO solve (against the DSO copy just produced) -> dual update
    (_update_interface_power_flow_variables):  lambda_dso += rho_pf * (x - z) / rating * s_base_dso,
    lambda_tso += rho_pf * (z - x) / rating * s_base_tso.   Boyd: r_entry = (x - z) / rating; y = lambda_dso / s_base_dso.
  * ESS is weighted consensus ADMM over TSO / DSO / ESSO with z: r_entry = a * (x_agent - z), a = 1 / (2 * S_ref),
    S_ref = admm.shared_ess_reference_rating_mva of the case file (every agent, every node); the reconstruction is
    validated against the production Boyd ess r (creep sidecar) before it is used.
  * The TSO interface is anchor + interface_delta (delta bounded by +/- interface rating, no cost on delta).

FORMULAS (window W1 = 152..261 for task 1; W2 = 200..261 for the task-2 histories; W3 = 188..261 for trends):
  D(k) = ||y(k) - y(k-1)||_2 over the 1728 pf entries (interface_duals sidecar, lambda_pf_{p,q}_dso / dso_base_mva);
  E(k) = rho_pf(k) * boyd.pf.r(k) (creep sidecar);  ratio(k) = D / E;  per-entry dev(k) = max |dy - rho * r_entry|
  (stride).  t_sum(k) = sum_{p entries} w * pi * (x - z)  (W112 identity; pi, w from interface_settlement_detail_s31c).
  EUR/MWh-equivalent of a pf dual (DSO side): y * admm_objective_scale_dso / rating_MVA (conversion note of the
  interface_duals header: lambda_dso_linear = pi + eff_dso * dual_pf_p_req / (r_pu * B)).
  Dominant set: entries ranked by their mean share of ||r||^2 over W3; the smallest prefix reaching 99 % cumulative.

VERDICT RULE (fixed in this file before the script was run; NOTE: thresholds were set after an exploratory look at
the stride -- see the report):
  (b) dual not applied: on W1 cycles whose previous cycle had no accepted AA step, max |ratio - 1| > 1e-6, OR the
      dominant entries' y changed by less than 1e-12 over W3.
  (a) no response to a growing dual: not (b) AND on every dominant entry |gap(261) - gap(188)| / |gap(188)| < 1e-4
      while |y(261) - y(188)| / |y(188)| >= 0.10.
  (c) slow mode: not (b) AND on some dominant entry the gap moved by >= 1e-4 relative over W3 toward zero.
  undetermined otherwise.  (a) is BEHAVIOURAL; which constraint holds each copy is reported separately (task 3) and
  only where the records carry the bound.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w126_pf_stall/):
  w126_pf_stall.json, launch.log, manifest_sha256.json
Smoke (nothing written): add --dry. Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w126_pf_stall && set -o noclobber && \\
    nice -n 10 /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w126_pf_stall.py \\
        > data/SRP1/Results/P515S53/w126_pf_stall/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w126_pf_stall.py --manifest
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W126 pf stall zero-solve').install()

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

K0, CAP = 152, 261
W1 = (152, 261)
W2 = (200, 261)
W3 = (188, 261)
TOP_N = 20
LAPSE_CYCLES = list(range(198, 207))
LAPSE_BASELINE = 200
DOMINANT_CUM_SHARE = 0.99
VERDICT = {'ratio_tol': 1e-6, 'dy_min': 1e-12, 'gap_rel_frozen': 1e-4, 'dy_rel_growing': 0.10}
GAP_MW_FLAG = 1e-3          # cross-tab: an entry "carries a gap" when |x_p - z_p| > 1e-3 MW
AT_BOUND_MW = 1e-6          # ESS power "at s_max" when | |p| - s_max | <= 1e-6 MW
AT_BOUND_PU = 1e-6          # voltage "at bound" when distance <= 1e-6 pu (the terminal file's own summary threshold)
NOT_IN_RECORDS = ['DSO flexibility per hour (up / down) and its bounds', 'ESS state of charge / energy bounds',
                  'TSO generator outputs and limits', 'TSO / DSO branch flows and limits (other than the interface '
                  'rating)', 'DSO internal node voltages', 'TSO RES available / curtailed per hour',
                  'model Param values dual_pf_p_req actually passed to each solve (needs a model load)',
                  'interface voltage / delta per cycle (terminal cycle 261 only)']

THIS = os.path.abspath(__file__)
CAMPAIGN = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle', 'campaign_s53_w118_resettle_r2_f2_challenger')
EVAL = os.path.join(CAMPAIGN, 'evals', '1fe91e86f11e76af_f2_challenger')
MANIFEST = os.path.join(CAMPAIGN, 'campaign_manifest_sha256.json')
SPEC = os.path.join(CAMPAIGN, 'campaign_spec_s53_w118_resettle_r2_f2_challenger_c7c9aea8.json')
CASE_FILE = os.path.join('data', 'SRP1', 'SRP1_params.json')
F = {k: os.path.join(EVAL, v) for k, v in {
    'stride': 'pf_entry_stride_s39_D.jsonl', 'duals': 'interface_duals_per_cycle.jsonl',
    'creep': 'creep_diagnostic_per_cycle.jsonl', 'resettle': 'resettle_cycle_record.jsonl',
    'pcr': 'per_cycle_record.jsonl', 'aa': 'aa_per_cycle.jsonl', 'ess': 'ess_schedule_per_cycle.jsonl',
    'vterm': 'interface_voltage_terminal.json', 'detail': 'interface_settlement_detail_s31c.json',
    'fail': 'network_failures_s39_D.jsonl', 'ipopt': 'network_ipopt_solve_records.jsonl',
    'comp': 'component_levels_terminal.json', 'evalrec': 'evaluation_record.json',
    'esso5': os.path.join('esso_capture', 's39_D', 'node5_cycle261.jsonl'),
    'esso7': os.path.join('esso_capture', 's39_D', 'node7_cycle261.jsonl'),
}.items()}
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'w126_pf_stall')
OUT_JSON = os.path.join(OUT_DIR, 'w126_pf_stall.json')
YEARS = ['2025', '2030', '2035']
DAYS_ORDER = ['Spring', 'Summer', 'Autumn', 'Winter']   # esso_capture d index; asserted against the ESS schedule


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=REPO)


def _verify_inputs():
    man = json.load(open(os.path.join(REPO, MANIFEST)))
    out = {}
    for rel in list(F.values()) + [SPEC]:
        now = _sha(os.path.join(REPO, rel))
        pinned = man.get(rel)
        if pinned is None:
            raise RuntimeError(f'{rel}: not hash-recorded in {MANIFEST}')
        if pinned != now:
            raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned}')
        tracked = _git('ls-files', '--error-unmatch', rel).returncode == 0
        clean = (_git('diff', '--quiet', 'HEAD', '--', rel).returncode == 0) if tracked else None
        if tracked and not clean:
            raise RuntimeError(f'{rel}: git-tracked but not clean against HEAD')
        out[rel] = {'sha256': now, 'manifest': MANIFEST, 'git_tracked': tracked, 'clean_vs_head': clean}
    # case file: pinned by the campaign spec
    spec = json.load(open(os.path.join(REPO, SPEC)))
    case_sha = _sha(os.path.join(REPO, CASE_FILE))
    if case_sha != spec['configuration']['case_file_sha256']:
        raise RuntimeError(f'{CASE_FILE}: sha256 {case_sha} != spec case_file_sha256')
    out[CASE_FILE] = {'sha256': case_sha, 'pinned_by': SPEC + ' configuration.case_file_sha256'}
    return out, spec


def _jsonl(rel, skip_header=True):
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if skip_header and row.get('header'):
                continue
            yield row


def _by_cycle(rel):
    out = {}
    for row in _jsonl(rel):
        if row['cycle'] in out:
            raise RuntimeError(f'{rel}: cycle {row["cycle"]} twice')
        out[row['cycle']] = row
    return out


def _ols(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    mx, my = xs.mean(), ys.mean()
    sxx = ((xs - mx) ** 2).sum()
    b = ((xs - mx) * (ys - my)).sum() / sxx
    a = my - b * mx
    res = ys - (a + b * xs)
    return {'slope_per_cycle': float(b), 'intercept': float(a), 'max_abs_residual': float(np.abs(res).max()),
            'rms_residual': float(np.sqrt((res ** 2).mean()))}


def _price_weight(detail):
    pi, w = {}, {}
    for n, by_year in detail['interface_reporting_detail'].items():
        for y, by_day in by_year.items():
            for d, dd in by_day.items():
                if dd.get('n_market_scenarios', 1) != 1 or dd.get('n_operation_scenarios', 1) != 1:
                    raise RuntimeError(f'{n}|{y}|{d}: more than one scenario')
                for t, pdet in dd['periods'].items():
                    pi[(int(n), str(y), str(d), int(t))] = pdet['price_per_mwh']
    for n, v in detail['interface_consensus_residual_per_dso'].items():
        for key, pdet in v['periods'].items():
            y, d, t = key.split('|')
            w[(int(n), y, d, int(t))] = pdet['admm_block_weight']
    if set(pi) != set(w):
        raise RuntimeError('price and weight key sets differ')
    return pi, w


def _load_stride():
    keys, cycles = None, []
    arr = {f: [] for f in ('x', 'z', 'zp', 'lam', 'r', 'rho', 'rating', 'sb')}
    prod_r, identity = [], []
    for row in _jsonl(F['stride']):
        es = row['entries']
        kk = [(int(e['node_id']), str(e['year']), str(e['day']), e['power_type'], int(e['period'])) for e in es]
        if keys is None:
            keys = kk
        elif kk != keys:
            raise RuntimeError(f'stride entry order changed at cycle {row["cycle"]}')
        cycles.append(row['cycle'])
        arr['x'].append([e['x_dso'] for e in es])
        arr['z'].append([e['z_tso_current'] for e in es])
        arr['zp'].append([e['z_tso_prev'] for e in es])
        arr['lam'].append([e['lambda_dso'] for e in es])
        arr['r'].append([e['r'] for e in es])
        arr['rho'].append([e['rho_pf'] for e in es])
        arr['rating'].append([e['interface_rating'] for e in es])
        arr['sb'].append([e['s_base_dso'] for e in es])
        prod_r.append(row['production_boyd_pf_r'])
        identity.append(bool(row['identity_holds']))
    if cycles != list(range(1, CAP + 1)):
        raise RuntimeError(f'stride cycles {cycles[0]}..{cycles[-1]} n={len(cycles)}')
    out = {k: np.array(v, float) for k, v in arr.items()}
    out['keys'] = keys
    out['prod_r'] = np.array(prod_r)
    out['identity_all'] = all(identity)
    return out


def _load_duals():
    header, rows = None, {}
    for row in _jsonl(F['duals'], skip_header=False):
        if row.get('header'):
            header = row
            continue
        if not row.get('captured'):
            raise RuntimeError(f'duals: cycle {row.get("cycle")} not captured')
        rows[row['cycle']] = row
    return header, rows


def run():
    t0 = time.time()
    if os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')
    inputs, spec = _verify_inputs()
    _log(f'{len(inputs)} inputs verified (campaign manifest / spec pin)')
    cand = spec['candidates'][0]
    evalrec = json.load(open(os.path.join(REPO, F['evalrec'])))
    case = json.load(open(os.path.join(REPO, CASE_FILE)))
    s_ref = case['admm']['shared_ess_reference_rating_mva']
    eps_abs, eps_rel = case['admm']['tol']['boyd']['eps_abs'], case['admm']['tol']['boyd']['eps_rel']

    # ---------------- load ----------------
    S = _load_stride()
    keys = S['keys']
    kidx = {k: i for i, k in enumerate(keys)}
    ci = {k: k - 1 for k in range(1, CAP + 1)}          # cycle -> row
    Y = S['lam'] / S['sb']
    creep = {}
    for row in _jsonl(F['creep']):
        creep[row['cycle']] = {'pf': row['boyd']['pf'], 'ess': row['boyd']['ess'], 'v': row['boyd']['v'],
                               't_sum': row.get('t_sum'), 'gross': row['gross']}
    pcr = _by_cycle(F['pcr'])
    aa = _by_cycle(F['aa'])
    res = _by_cycle(F['resettle'])
    dheader, drows = _load_duals()
    detail = json.load(open(os.path.join(REPO, F['detail'])))
    pi, w = _price_weight(detail)
    _log(f'loaded: stride {S["x"].shape}, creep {len(creep)}, pcr {len(pcr)}, duals {len(drows)}')

    # ---------------- task 1: dual path ----------------
    blocks = dheader['metadata']['per_block']
    scale_dso = {(int(b['node_id']), str(b['year']), str(b['day'])): b['admm_objective_scale_dso'] for b in blocks}

    def _dual_vec(row):
        v = []
        for b, meta in enumerate(blocks):
            for kind in ('p', 'q'):
                for t in range(dheader['n_periods']):
                    v.append(row[f'lambda_pf_{kind}_dso'][b][t] / meta['dso_base_mva'])
        return np.array(v)

    dual_vecs = {k: _dual_vec(drows[k]) for k in range(W1[0] - 1, CAP + 1)}
    # cross-check stride lambda vs sidecar lambda (same quantity, same read point)
    order_map = []
    for b, meta in enumerate(blocks):
        for kind in ('p', 'q'):
            for t in range(dheader['n_periods']):
                order_map.append(kidx[(int(meta['node_id']), str(meta['year']), str(meta['day']), kind, t)])
    order_map = np.array(order_map)
    max_stride_vs_sidecar = max(float(np.abs(dual_vecs[k] - Y[ci[k]][order_map]).max()) for k in dual_vecs)
    t1 = []
    for k in range(W1[0], CAP + 1):
        rho_set = sorted(set(S['rho'][ci[k]].tolist()))
        rho = rho_set[0] if len(rho_set) == 1 else None
        r = creep[k]['pf']['r']
        D = float(np.linalg.norm(dual_vecs[k] - dual_vecs[k - 1]))
        E = rho * r if rho is not None else None
        dy = Y[ci[k]] - Y[ci[k - 1]]
        t1.append({'cycle': k, 'rho_pf': rho, 'rho_pf_set': rho_set, 'boyd_pf_r': r,
                   'stride_r_equals_creep_r': S['prod_r'][ci[k]] == r,
                   'norm_dy_dso': D, 'rho_times_r': E, 'ratio_D_over_E': D / E if E else None,
                   'stride_max_entry_abs_dy_minus_rho_r': float(np.abs(dy - S['rho'][ci[k]] * S['r'][ci[k]]).max()),
                   'norm_y_dso_sidecar': float(np.linalg.norm(dual_vecs[k])), 'norm_y_creep': creep[k]['pf']['norm_y'],
                   'aa_accepted_prev_cycle': bool(aa[k - 1]['aa_accepted']), 'aa_accepted': bool(aa[k]['aa_accepted']),
                   'boyd_pf_primal_ratio': pcr[k]['boyd_pf_primal_ratio'], 'eps_pri_pf': creep[k]['pf']['eps_pri'],
                   'rho_pf_after': pcr[k]['rho_pf_after'], 'local_solves_ok': pcr[k]['local_solves_ok']})
    no_aa = [row for row in t1 if not row['aa_accepted_prev_cycle']]
    ks1 = [row['cycle'] for row in t1]
    t1_summary = {
        'window': list(W1),
        'n_cycles': len(t1), 'n_cycles_prev_aa_accepted': len(t1) - len(no_aa),
        'cycles_prev_aa_accepted': [row['cycle'] for row in t1 if row['aa_accepted_prev_cycle']],
        'ratio_no_prev_aa': {'min': min(r['ratio_D_over_E'] for r in no_aa), 'max': max(r['ratio_D_over_E'] for r in no_aa),
                             'max_abs_minus_1': max(abs(r['ratio_D_over_E'] - 1) for r in no_aa)},
        'max_entry_abs_dy_minus_rho_r_no_prev_aa': max(r['stride_max_entry_abs_dy_minus_rho_r'] for r in no_aa),
        'rho_pf_values': sorted({r['rho_pf'] for r in t1}),
        'all_stride_r_equal_creep_r': all(r['stride_r_equals_creep_r'] for r in t1),
        'max_abs_stride_lambda_vs_sidecar_lambda_over_dso_base': max_stride_vs_sidecar,
        'norm_y_first_last': [t1[0]['norm_y_dso_sidecar'], t1[-1]['norm_y_dso_sidecar']],
        'norm_y_ols_153_261': _ols(ks1[1:], [r['norm_y_dso_sidecar'] for r in t1[1:]]),
        'norm_dy_ols_153_261': _ols(ks1[1:], [r['norm_dy_dso'] for r in t1[1:]]),
        'boyd_pf_r_188_261': {'min': min(creep[k]['pf']['r'] for k in range(W3[0], W3[1] + 1)),
                              'max': max(creep[k]['pf']['r'] for k in range(W3[0], W3[1] + 1)),
                              'ols': _ols(list(range(W3[0], W3[1] + 1)), [creep[k]['pf']['r'] for k in range(W3[0], W3[1] + 1)])},
        'boyd_pf_primal_ratio_188_261': {'min': min(pcr[k]['boyd_pf_primal_ratio'] for k in range(W3[0], W3[1] + 1)),
                                         'max': max(pcr[k]['boyd_pf_primal_ratio'] for k in range(W3[0], W3[1] + 1))},
    }
    _log(f'task 1: ratio (no prev AA) {t1_summary["ratio_no_prev_aa"]}; rho {t1_summary["rho_pf_values"]}')

    # ---------------- task 2: localisation ----------------
    R2 = S['r'] ** 2
    tot = R2.sum(axis=1)
    share = R2 / tot[:, None]
    w3 = slice(ci[W3[0]], ci[W3[1]] + 1)
    w2 = slice(ci[W2[0]], ci[W2[1]] + 1)
    mean_share_w3 = share[w3].mean(axis=0)
    order_w3 = np.argsort(-mean_share_w3)
    cum = np.cumsum(mean_share_w3[order_w3])
    n_dom = int(np.searchsorted(cum, DOMINANT_CUM_SHARE) + 1)
    dominant = [int(j) for j in order_w3[:n_dom]]
    mean_r2_w2 = R2[w2].mean(axis=0)
    top = [int(j) for j in np.argsort(-mean_r2_w2)[:TOP_N]]
    top_set_per_cycle_stable = all(
        set(np.argsort(-R2[ci[k]])[:len(dominant)].tolist()) == set(dominant) for k in range(W3[0], W3[1] + 1))

    def _key(j):
        n, y, d, pt, t = keys[j]
        return {'node_id': n, 'year': y, 'day': d, 'power_type': pt, 'period': t, 'hour': t + 1}

    def _eur_mwh(j, k):
        n, y, d, pt, t = keys[j]
        return float(Y[ci[k], j] * scale_dso[(n, y, d)] / S['rating'][ci[k], j])

    def _entry_block(j, hist_window):
        ks = list(range(hist_window[0], hist_window[1] + 1))
        rows = [ci[k] for k in ks]
        gap = S['x'][rows, j] - S['z'][rows, j]
        dx = np.diff(S['x'][rows, j])
        dz = np.diff(S['z'][rows, j])
        g188, g261 = S['x'][ci[W3[0]], j] - S['z'][ci[W3[0]], j], S['x'][ci[CAP], j] - S['z'][ci[CAP], j]
        y188, y261 = Y[ci[W3[0]], j], Y[ci[CAP], j]
        gap_ols = _ols(list(range(W3[0], W3[1] + 1)), (S['x'][w3, j] - S['z'][w3, j]).tolist())
        y_ols = _ols(list(range(W3[0], W3[1] + 1)), Y[w3, j].tolist())
        rho_r_mean = float((S['rho'][w3, j] * S['r'][w3, j]).mean())
        closing = gap_ols['slope_per_cycle'] * math.copysign(1.0, -g261) if g261 else 0.0
        return {
            **_key(j), 'rating_mva': float(S['rating'][ci[CAP], j]),
            'mean_share_r2_188_261': float(mean_share_w3[j]), 'mean_r2_200_261': float(mean_r2_w2[j]),
            'share_r2_at_261': float(share[ci[CAP], j]),
            'x_dso_261': float(S['x'][ci[CAP], j]), 'z_tso_261': float(S['z'][ci[CAP], j]),
            'z_tso_prev_261': float(S['zp'][ci[CAP], j]), 'gap_x_minus_z_261_mw': float(g261),
            'y_dso_261': float(Y[ci[CAP], j]), 'y_eur_per_mwh_equiv_261': _eur_mwh(j, CAP),
            'y_eur_per_mwh_equiv_188': _eur_mwh(j, W3[0]),
            'z_between_copies_note': 'no separate consensus variable in pf two-block ADMM: z IS the TSO copy',
            'max_abs_step_x_mw_in_window': float(np.abs(dx).max()), 'max_abs_step_z_mw_in_window': float(np.abs(dz).max()),
            'x_change_188_261_mw': float(S['x'][ci[CAP], j] - S['x'][ci[W3[0]], j]),
            'z_change_188_261_mw': float(S['z'][ci[CAP], j] - S['z'][ci[W3[0]], j]),
            'gap_change_188_261_mw': float(g261 - g188),
            'gap_rel_change_188_261': float(abs(g261 - g188) / abs(g188)) if g188 else None,
            'gap_ols_188_261': gap_ols, 'gap_closing_rate_mw_per_cycle': float(closing),
            'cycles_to_close_at_ols_rate': float(abs(g261) / closing) if closing > 0 else None,
            'y_change_188_261': float(y261 - y188), 'y_rel_change_188_261': float(abs(y261 - y188) / abs(y188)) if y188 else None,
            'y_ols_188_261': y_ols, 'rho_r_mean_188_261': rho_r_mean,
            'y_slope_over_rho_r': y_ols['slope_per_cycle'] / rho_r_mean if rho_r_mean else None,
            'response_d_gap_over_d_y': float((g261 - g188) / (y261 - y188)) if y261 != y188 else None,
            'history': {'cycles': ks, 'x_dso': S['x'][rows, j].tolist(), 'z_tso': S['z'][rows, j].tolist(),
                        'z_tso_prev': S['zp'][rows, j].tolist(), 'r': S['r'][rows, j].tolist(),
                        'y_dso': Y[rows, j].tolist(), 'gap_mw': gap.tolist()},
        }

    dominant_rows = [_entry_block(j, (1, CAP)) for j in dominant]
    for row, j in zip(dominant_rows, dominant):
        g = np.abs(S['x'][:, j] - S['z'][:, j])
        onset = next((k for k in range(1, CAP + 1) if all(g[ci[m]] > 0.5 * g[ci[CAP]] for m in range(k, CAP + 1))), None)
        row['onset_cycle_gap_above_half_terminal_through_261'] = onset
        row['y_sign_change_cycles'] = [k for k in range(2, CAP + 1) if Y[ci[k], j] * Y[ci[k - 1], j] < 0]
    top_rows = [_entry_block(j, W2) for j in top]
    # aggregates node x year x day x type
    agg = {}
    for j, (n, y, d, pt, t) in enumerate(keys):
        key = f'{n}|{y}|{d}|{pt}'
        a = agg.setdefault(key, {'share_r2_at_261': 0.0, 'mean_share_r2_188_261': 0.0})
        a['share_r2_at_261'] += float(share[ci[CAP], j])
        a['mean_share_r2_188_261'] += float(mean_share_w3[j])
    agg_sorted = dict(sorted(agg.items(), key=lambda kv: -kv[1]['mean_share_r2_188_261']))
    # residual without the dominant set, vs eps_pri
    mask = np.ones(len(keys), bool)
    mask[dominant] = False
    r_rest = {k: float(np.sqrt(R2[ci[k], mask].sum())) for k in range(W1[0], CAP + 1)}
    rest = {'r_without_dominant_261': r_rest[CAP], 'eps_pri_261': creep[CAP]['pf']['eps_pri'],
            'primal_ratio_without_dominant_261': r_rest[CAP] / creep[CAP]['pf']['eps_pri'],
            'primal_ratio_without_dominant_188_261_max': max(r_rest[k] / creep[k]['pf']['eps_pri']
                                                              for k in range(W3[0], W3[1] + 1))}
    # t_sum decomposition
    t_rows = []
    pmask = np.array([k[3] == 'p' for k in keys])
    wpi = np.array([w[(k[0], k[1], k[2], k[4])] * pi[(k[0], k[1], k[2], k[4])] if k[3] == 'p' else 0.0 for k in keys])
    dom_p = [j for j in dominant if keys[j][3] == 'p']
    for k in range(W1[0], CAP + 1):
        term = wpi * (S['x'][ci[k]] - S['z'][ci[k]])
        t_sum = float(term[pmask].sum())
        t_dom = float(term[dom_p].sum())
        rr = res[k]
        t_rows.append({'cycle': k, 't_sum_from_stride': t_sum, 't_sum_resettle_record': rr['t_sum'],
                       'diff': t_sum - rr['t_sum'], 't_dominant_entries': t_dom,
                       'share_t_dominant': t_dom / t_sum if t_sum else None,
                       't_by_node_resettle': rr['t_by_node'], 'sum_abs_gap_mw_p_resettle': rr['sum_abs_gap_mw_p'],
                       'sum_abs_gap_mw_p_dominant': float(np.abs(S['x'][ci[k], dom_p] - S['z'][ci[k], dom_p]).sum())})
    t_summary = {'max_abs_diff_stride_vs_record': max(abs(r['diff']) for r in t_rows),
                 't_sum_261': t_rows[-1]['t_sum_from_stride'], 't_dominant_261': t_rows[-1]['t_dominant_entries'],
                 'share_dominant_min_188_261': min(r['share_t_dominant'] for r in t_rows if r['cycle'] >= W3[0]),
                 'share_dominant_max_188_261': max(r['share_t_dominant'] for r in t_rows if r['cycle'] >= W3[0]),
                 'dominant_price_weight': {f'{keys[j][0]}|{keys[j][1]}|{keys[j][2]}|{keys[j][4]}':
                                           {'pi_eur_per_mwh': pi[(keys[j][0], keys[j][1], keys[j][2], keys[j][4])],
                                            'admm_block_weight': w[(keys[j][0], keys[j][1], keys[j][2], keys[j][4])]}
                                           for j in dom_p}}
    _log(f'task 2: dominant {[keys[j] for j in dominant]}; t_sum check {t_summary["max_abs_diff_stride_vs_record"]!r}')

    # ---------------- task 3: bounds ----------------
    vterm = json.load(open(os.path.join(REPO, F['vterm'])))
    vmap = {(int(e['node_id']), str(e['year']), str(e['day']), int(e['period'])): e for e in vterm['entries']}
    esso_smax = {}
    for n, rel in ((5, F['esso5']), (7, F['esso7'])):
        for row in _jsonl(rel):
            yr = YEARS[int(row['y'])]
            esso_smax.setdefault((n, yr), set()).add(row['s_max'])
    ess_header = next(_jsonl(F['ess'], skip_header=False))
    ess_blocks = [(int(b[0]), str(b[1]), str(b[2])) for b in ess_header['blocks']]
    bidx = {b: i for i, b in enumerate(ess_blocks)}
    # ESS schedule: keep cycles W2 (task 3) and LAPSE (task 4) and all cycles for the ess r reconstruction
    ess_keep = {}
    ess_recon = {}
    a_ess = 1.0 / (2.0 * s_ref)
    for row in _jsonl(F['ess']):
        k = row['cycle']
        sq_r = sq_x = sq_z = 0.0
        for fam in ('p', 'q'):
            z = np.array(row[fam]['z'])
            for ag in ('tso', 'dso', 'esso'):
                x = np.array(row[fam][ag])
                sq_r += float(((a_ess * (x - z)) ** 2).sum())
                sq_x += float(((a_ess * x) ** 2).sum())
                sq_z += float(((a_ess * z) ** 2).sum())
        n_ess = 3 * 2 * len(ess_blocks) * 24
        eps = math.sqrt(n_ess) * eps_abs + eps_rel * max(math.sqrt(sq_x), math.sqrt(sq_z))
        ess_recon[k] = {'r': math.sqrt(sq_r), 'eps_pri': eps, 'primal_ratio': math.sqrt(sq_r) / eps,
                        'creep_r': creep[k]['ess']['r'], 'pcr_ratio': pcr[k]['boyd_ess_primal_ratio']}
        if W2[0] <= k <= CAP or k in LAPSE_CYCLES:
            ess_keep[k] = row
    recon_err = max(abs(v['r'] - v['creep_r']) / v['creep_r'] for v in ess_recon.values() if v['creep_r'])
    ratio_err = max(abs(v['primal_ratio'] - v['pcr_ratio']) for v in ess_recon.values())
    # day-index mapping check: esso_capture d index vs the ESS schedule esso copy at cycle 261
    map_check = []
    for n, rel in ((5, F['esso5']), (7, F['esso7'])):
        for row in _jsonl(rel):
            if int(row['y_inv']) != int(row['y']):
                continue
            b = bidx[(n, YEARS[int(row['y'])], DAYS_ORDER[int(row['d'])])]
            map_check.append(abs(ess_keep[CAP]['p']['esso'][b][int(row['p'])] - row['pnet']))
    day_map_max_abs = max(map_check)
    if day_map_max_abs > 1e-9:
        raise RuntimeError(f'esso_capture day-index mapping check failed: {day_map_max_abs}')

    def _ess_at(n, y, d, t, k):
        b = bidx[(n, y, d)]
        row = ess_keep[k]
        return {fam: {ag: row[fam][ag][b][t] for ag in row[fam]} for fam in ('p', 'q', 'charge', 'discharge')}

    def _bounds(j):
        n, y, d, pt, t = keys[j]
        jp, jq = kidx[(n, y, d, 'p', t)], kidx[(n, y, d, 'q', t)]
        rating = float(S['rating'][ci[CAP], j])
        s_x = math.hypot(S['x'][ci[CAP], jp], S['x'][ci[CAP], jq])
        s_z = math.hypot(S['z'][ci[CAP], jp], S['z'][ci[CAP], jq])
        v = vmap[(n, y, d, t)]
        det = detail['interface_reporting_detail'][str(n)][y][d]['periods'][str(t)]
        smax = sorted(esso_smax.get((n, y), set()))
        ess261 = _ess_at(n, y, d, t, CAP)
        ess_p_hist = {ag: [_ess_at(n, y, d, t, k)['p'][ag] for k in range(W2[0], W2[1] + 1)]
                      for ag in ('tso', 'dso', 'esso', 'z')}
        if smax:
            s = smax[-1]
            at = {ag: abs(abs(ess261['p'][ag]) - s) <= AT_BOUND_MW for ag in ('tso', 'dso', 'esso', 'z')}
            ess_block = {'s_max_mw_esso_capture_261': smax, 'p_261': ess261['p'], 'q_261': ess261['q'],
                         'charge_261': ess261['charge'], 'discharge_261': ess261['discharge'],
                         'abs_p_minus_s_max_261': {ag: abs(ess261['p'][ag]) - s for ag in ess261['p']},
                         'at_s_max_within_1e-6_mw_261': at,
                         'p_range_200_261': {ag: [min(v_), max(v_)] for ag, v_ in ess_p_hist.items()},
                         'sign_note': 'p = charge - discharge (MW); negative = discharging'}
        else:
            ess_block = {'s_max_mw_esso_capture_261': None, 'p_261': ess261['p'],
                         'note': 'no shared ESS invested at this node for this year (candidate); every copy 0'}
        return {**_key(j),
                'interface_rating_mva': rating, 'apparent_power_dso_mva_261': s_x, 'apparent_power_tso_mva_261': s_z,
                'rating_margin_dso_mva': rating - s_x, 'rating_margin_tso_mva': rating - s_z,
                'interface_voltage_terminal_261': {kk: v[kk] for kk in ('tso_pu', 'dso_pu', 'v_min_pu', 'v_max_pu',
                                                                        'distance_to_nearest_bound_pu', 'nearest_bound')},
                'voltage_at_bound_within_1e-6_pu': v['distance_to_nearest_bound_pu'] <= AT_BOUND_PU,
                'tso_interface_delta_terminal_261': {'delta_p_mw': det['delta_p_mw'], 'delta_q_mvar': det['delta_q_mvar'],
                                                     'anchor_p_mw': det['anchor_p_mw'], 'anchor_q_mvar': det['anchor_q_mvar'],
                                                     'delta_bounds_mw': [-rating, rating],
                                                     'p_int_tso_expected_mw': det['p_int_tso_expected_mw'],
                                                     'p_int_dso_expected_mw': det['p_int_dso_expected_mw']},
                'shared_ess': ess_block}

    bounds_top = [_bounds(j) for j in top]
    # cross-tab over every (node, year, day, period) at 261
    xtab = {'gap_and_ess_at_smax': 0, 'gap_no_ess_bound': 0, 'nogap_and_ess_at_smax': 0, 'nogap_no_ess_bound': 0,
            'gap_and_v_at_bound': 0, 'gap_v_free': 0, 'nogap_and_v_at_bound': 0, 'nogap_v_free': 0,
            'gap_entries': []}
    for (n, y, d, t), v in sorted(vmap.items()):
        jp = kidx[(n, y, d, 'p', t)]
        gap = abs(S['x'][ci[CAP], jp] - S['z'][ci[CAP], jp]) > GAP_MW_FLAG
        smax = sorted(esso_smax.get((n, y), set()))
        ess_bd = bool(smax) and abs(abs(_ess_at(n, y, d, t, CAP)['p']['dso']) - smax[-1]) <= AT_BOUND_MW
        v_bd = v['distance_to_nearest_bound_pu'] <= AT_BOUND_PU
        xtab[('gap' if gap else 'nogap') + ('_and_ess_at_smax' if ess_bd else '_no_ess_bound')] += 1
        xtab[('gap' if gap else 'nogap') + ('_and_v_at_bound' if v_bd else '_v_free')] += 1
        if gap:
            xtab['gap_entries'].append({'node_id': n, 'year': y, 'day': d, 'period': t,
                                        'gap_mw': float(S['x'][ci[CAP], jp] - S['z'][ci[CAP], jp]),
                                        'dso_ess_at_smax': ess_bd, 'voltage_at_bound': v_bd})
    # the dominant block's 24 hours (context: is the ESS / voltage pattern specific to the gap hour?)
    dom_blocks = sorted({(keys[j][0], keys[j][1], keys[j][2]) for j in dominant})
    block_hours = {}
    for (n, y, d) in dom_blocks:
        hrs = []
        smax = sorted(esso_smax.get((n, y), set()))
        for t in range(24):
            jp = kidx[(n, y, d, 'p', t)]
            e = _ess_at(n, y, d, t, CAP)
            hrs.append({'period': t, 'gap_mw': float(S['x'][ci[CAP], jp] - S['z'][ci[CAP], jp]),
                        'x_dso_mw': float(S['x'][ci[CAP], jp]), 'z_tso_mw': float(S['z'][ci[CAP], jp]),
                        'ess_p_dso': e['p']['dso'], 'ess_p_tso': e['p']['tso'], 'ess_p_esso': e['p']['esso'],
                        'ess_dso_at_smax': bool(smax) and abs(abs(e['p']['dso']) - smax[-1]) <= AT_BOUND_MW,
                        'v_dist_to_bound_pu': vmap[(n, y, d, t)]['distance_to_nearest_bound_pu'],
                        'price_eur_per_mwh': pi[(n, y, d, t)]})
        block_hours[f'{n}|{y}|{d}'] = hrs
    comp = json.load(open(os.path.join(REPO, F['comp'])))['blocks']
    comp_ctx = {}
    for (n, y, d) in dom_blocks:
        for bk in (f'TSO|{y}|{d}', f'DSO|{n}|{y}|{d}'):
            if bk in comp:
                comp_ctx[bk] = comp[bk]['unweighted']
    # IPOPT exits for the dominant blocks over W1
    netname = {5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
    ipopt_rows = [json.loads(line) for line in open(os.path.join(REPO, F['ipopt'])) if line.strip()]
    ip_dom = {}
    for (n, y, d) in dom_blocks:
        for net in ('case9', netname[n]):
            sel = [r for r in ipopt_rows if r['network'] == net and str(r['year']) == y and r['day'] == d
                   and W1[0] <= r['round'] <= CAP]
            ip_dom[f'{net}|{y}|{d}'] = {
                'n_records': len(sel), 'rounds_covered': sorted({r['round'] for r in sel}) == list(range(W1[0], CAP + 1)),
                'exits': {e: sum(1 for r in sel if r['exit'] == e) for e in sorted({r['exit'] for r in sel})},
                'attempts': {a: sum(1 for r in sel if r['attempt'] == a) for a in sorted({r['attempt'] for r in sel})},
                'iterations_min_max': [min(r['iterations'] for r in sel), max(r['iterations'] for r in sel)] if sel else None}
    _log(f'task 3: ESS r reconstruction rel err {recon_err!r}; xtab {({k: v for k, v in xtab.items() if k != "gap_entries"})}')

    # ---------------- task 4: the lapse at 202-203 ----------------
    fails = [json.loads(line) for line in open(os.path.join(REPO, F['fail'])) if line.strip()]
    events = [{k: f.get(k) for k in ('cycle', 'agent', 'node_id', 'network_name', 'year', 'day', 'primary_termination',
                                      'termination', 'class')}
              for f in fails if f.get('record_type') == 'network_block' and W1[0] <= (f.get('cycle') or 0) <= CAP]

    def _ess_entries(k):
        row = ess_keep[k]
        out = []
        for fam in ('p', 'q'):
            z = row[fam]['z']
            for b, (n, y, d) in enumerate(ess_blocks):
                for ag in ('tso', 'dso', 'esso'):
                    for t in range(24):
                        out.append(((n, y, d, fam, t, ag), a_ess * (row[fam][ag][b][t] - z[b][t])))
        return out

    base = dict(_ess_entries(LAPSE_BASELINE))
    lapse = []
    for k in LAPSE_CYCLES:
        ents = _ess_entries(k)
        r2tot = sum(v * v for _, v in ents)
        top_e = sorted(ents, key=lambda kv: -kv[1] ** 2)[:10]
        by_block = {}
        by_agent = {}
        for (n, y, d, fam, t, ag), v in ents:
            by_block[f'{n}|{y}|{d}'] = by_block.get(f'{n}|{y}|{d}', 0.0) + v * v
            by_agent[f'{n}|{y}|{d}|{ag}'] = by_agent.get(f'{n}|{y}|{d}|{ag}', 0.0) + v * v
        delta = sorted(((kk, v * v - base[kk] ** 2) for kk, v in ents), key=lambda kv: -kv[1])[:10]
        lapse.append({
            'cycle': k, 'ess_r_reconstructed': ess_recon[k]['r'], 'ess_r_creep': ess_recon[k]['creep_r'],
            'ess_primal_ratio_reconstructed': ess_recon[k]['primal_ratio'], 'ess_primal_ratio_pcr': ess_recon[k]['pcr_ratio'],
            'pf_primal_ratio_pcr': pcr[k]['boyd_pf_primal_ratio'], 'v_primal_ratio_pcr': pcr[k]['boyd_v_primal_ratio'],
            'local_solves_ok': pcr[k]['local_solves_ok'],
            'top10_entries_by_r2': [{'node_id': e[0][0], 'year': e[0][1], 'day': e[0][2], 'power_type': e[0][3],
                                     'period': e[0][4], 'agent': e[0][5], 'r_entry': e[1],
                                     'share_r2': e[1] ** 2 / r2tot} for e in top_e],
            'top10_entries_by_r2_increase_vs_200': [{'node_id': e[0][0], 'year': e[0][1], 'day': e[0][2],
                                                     'power_type': e[0][3], 'period': e[0][4], 'agent': e[0][5],
                                                     'r2_increase': e[1]} for e in delta],
            'share_r2_by_block_top5': dict(sorted(((b, v / r2tot) for b, v in by_block.items()), key=lambda kv: -kv[1])[:5]),
            'share_r2_by_block_agent_top5': dict(sorted(((b, v / r2tot) for b, v in by_agent.items()),
                                                        key=lambda kv: -kv[1])[:5]),
            'failure_events_this_cycle': [e for e in events if e['cycle'] == k]})
    ip_lapse = [{kk: r[kk] for kk in ('round', 'network', 'year', 'day', 'agent', 'attempt', 'warm_start', 'iterations', 'exit')}
                for r in ipopt_rows if r['round'] in LAPSE_CYCLES and (r['attempt'] != 'primary' or
                                                                        r['exit'] != 'Optimal Solution Found.')]
    coincidence = []
    for ev in [e for e in events if e['cycle'] in LAPSE_CYCLES]:
        blk = f'{ev["node_id"]}|{ev["year"]}|{ev["day"]}' if ev['node_id'] is not None else None
        rows_k = {r['cycle']: r for r in lapse}
        entry = {'event': ev}
        for k in (ev['cycle'], ev['cycle'] + 1):
            if k in rows_k:
                top_blocks = list(rows_k[k]['share_r2_by_block_top5'].items())
                if blk is not None:
                    entry[f'share_r2_of_event_block_at_{k}'] = dict(top_blocks).get(blk)
                else:
                    entry[f'share_r2_of_blocks_with_event_year_day_at_{k}'] = {
                        b: s for b, s in top_blocks if b.split('|')[1] == str(ev['year']) and b.split('|')[2] == ev['day']}
                entry[f'top_block_at_{k}'] = top_blocks[0]
        coincidence.append(entry)
    _log(f'task 4: ess ratio {[(r["cycle"], round(r["ess_primal_ratio_pcr"], 4)) for r in lapse]}')

    # ---------------- task 5: verdict ----------------
    dom_rel_gap = [r['gap_rel_change_188_261'] for r in dominant_rows]
    dom_rel_y = [r['y_rel_change_188_261'] for r in dominant_rows]
    dom_abs_dy = [abs(r['y_change_188_261']) for r in dominant_rows]
    cond_b = (t1_summary['ratio_no_prev_aa']['max_abs_minus_1'] > VERDICT['ratio_tol']
              or min(dom_abs_dy) < VERDICT['dy_min'])
    cond_a = (not cond_b and all(g < VERDICT['gap_rel_frozen'] for g in dom_rel_gap)
              and all(v >= VERDICT['dy_rel_growing'] for v in dom_rel_y))
    cond_c = (not cond_b and any(r['gap_rel_change_188_261'] >= VERDICT['gap_rel_frozen']
                                 and r['gap_closing_rate_mw_per_cycle'] > 0 for r in dominant_rows))
    verdict = '(b)' if cond_b else '(a)' if cond_a else '(c)' if cond_c else 'undetermined'

    guard_failures = _GUARD.verify(0)
    result = {
        'stage': 'P5.15 W126 frozen pf residual on the F2 challenger (records only, zero solves, no model loads)',
        'utc': _utc(), 'git_head': _git('rev-parse', 'HEAD').stdout.strip(), 'script_sha256': _sha(THIS),
        'solve_profile_guard': {'permitted': [], 'verify_0_failures': guard_failures, 'counts': dict(_GUARD.counts)},
        'instance': {'campaign': CAMPAIGN, 'eval_dir': EVAL, 'eval_key': evalrec['eval_key'],
                     'candidate_key': evalrec['candidate_key'], 'candidate_label': evalrec['candidate_label'],
                     'candidate_canonical': cand['canonical'], 'flex_price_multiplier': cand['flex_price_multiplier'],
                     'replayed_original_eval_key': cand['settling_resettle']['replay_reference']['original_eval_key'],
                     'k0': K0, 'cap': CAP},
        'objective_convention': 't_sum = sum w * pi * (x_dso - z_tso) over pf p entries (EUR, block-weighted; the '
                                'priced interface gap of the settling rule); no gross / net cost is reported here',
        'inputs_sha256': inputs,
        'constants': {'W1': W1, 'W2': W2, 'W3': W3, 'TOP_N': TOP_N, 'DOMINANT_CUM_SHARE': DOMINANT_CUM_SHARE,
                      'VERDICT': VERDICT, 'GAP_MW_FLAG': GAP_MW_FLAG, 'AT_BOUND_MW': AT_BOUND_MW,
                      'AT_BOUND_PU': AT_BOUND_PU, 'shared_ess_reference_rating_mva': s_ref, 'ess_a': a_ess,
                      'eps_abs': eps_abs, 'eps_rel': eps_rel, 'LAPSE_CYCLES': LAPSE_CYCLES,
                      'LAPSE_BASELINE': LAPSE_BASELINE},
        'stride_identity_holds_all_cycles': S['identity_all'],
        'task1_dual_path': {'summary': t1_summary, 'per_cycle': t1},
        'task2_localisation': {
            'dominant_set_rule': f'entries ranked by mean share of ||r_pf||^2 over {W3}; smallest prefix >= '
                                 f'{DOMINANT_CUM_SHARE}',
            'dominant_cumulative_mean_share': float(cum[n_dom - 1]),
            'dominant_same_top_set_every_cycle_188_261': top_set_per_cycle_stable,
            'dominant_entries_full_history_1_261': dominant_rows,
            'top20_by_mean_r2_200_261': top_rows,
            'aggregate_node_year_day_type': agg_sorted,
            'residual_without_dominant': rest,
            't_sum': {'summary': t_summary, 'per_cycle': t_rows},
            'consensus_variable_note': 'pf uses two-block ADMM (get_admm_boyd_residual_metrics docstring): x = DSO copy, '
                                       'z = TSO copy, no third consensus value. The DSO solves against z_tso_prev, the '
                                       'TSO against the DSO copy just produced.'},
        'task3_bounds': {'top20': bounds_top, 'crosstab_all_864_hours_at_261': xtab,
                         'dominant_block_24_hours_at_261': block_hours,
                         'component_levels_terminal_block_context': comp_ctx,
                         'ipopt_exits_dominant_blocks_152_261': ip_dom,
                         'ess_r_reconstruction': {'max_rel_err_vs_creep_r_all_cycles': recon_err,
                                                  'max_abs_err_primal_ratio_vs_pcr': ratio_err,
                                                  'esso_capture_day_index_check_max_abs': day_map_max_abs},
                         'not_in_records': NOT_IN_RECORDS},
        'task4_ess_lapse': {'per_cycle': lapse, 'failure_events_152_261': events,
                            'ipopt_non_primary_or_non_optimal_in_lapse_cycles': ip_lapse,
                            'coincidence': coincidence,
                            'ess_primal_ratio_series_152_261': {str(k): ess_recon[k]['pcr_ratio'] for k in range(W1[0], CAP + 1)}},
        'task5_verdict': {'rule': VERDICT, 'cond_b': cond_b, 'cond_a': cond_a, 'cond_c': cond_c, 'verdict': verdict,
                          'dominant_gap_rel_change_188_261': dom_rel_gap, 'dominant_y_rel_change_188_261': dom_rel_y,
                          'scope': '(a) here is behavioural (copies do not respond to a linearly growing dual); the '
                                   'constraint holding each copy is identified only as far as task 3 records allow'},
        'wall_s': time.time() - t0,
    }
    if '--dry' in sys.argv:
        GRIO.check(result, default=GRIO.json_default_item)
        _log(f'DRY: verdict {verdict} (a {cond_a}, b {cond_b}, c {cond_c}); guard verify(0) {guard_failures}; '
             f'nothing written; wall {time.time() - t0:.1f} s')
        return 0 if not guard_failures else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'x') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    _log(f'verdict {verdict} (a {cond_a}, b {cond_b}, c {cond_c}); guard verify(0) {guard_failures}; '
         f'counts {dict(_GUARD.counts)}; wrote {OUT_JSON}; wall {time.time() - t0:.1f} s')
    return 0 if not guard_failures else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    res = json.load(open(OUT_JSON))
    for rel, v in res['inputs_sha256'].items():
        entries[rel + ' (input)'] = v['sha256']
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else run())
