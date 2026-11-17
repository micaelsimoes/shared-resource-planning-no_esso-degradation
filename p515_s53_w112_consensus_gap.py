"""P5.15 W112 (Addendum 54 Ruling 1 follow-up) -- records-only diagnostics of the PRICED INTERFACE-CONSENSUS GAP.
ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Nothing is built, nothing is unpickled, nothing is solved; only committed / hash-recorded JSON(L) records are read,
streamed line by line.

PRE-REGISTRATION: data/SRP1/Results/P515S53/w112_consensus_gap/w112_preregistration_v1_ba384b28.json, committed in
cbdd78ec BEFORE this script computed anything (verdict rule, quarters, fit definitions, dual-path ratio, tolerances).
Its sha256 is verified at start.

INSTANCE (task 1-3): SRP1 C* (candidate_key 578636da...), campaign s53_w105_c_star_ext_r2, eval key 8864266d...;
cycles 1..187 a bitwise replay of W104 (campaign s53_w101_srp1_cont_c_star, eval 4bf36c151fd10613_c_star).

FORMULAS.
  t_sum(k) = sum_{pf entries, power_type p} w(y,d) * pi(n,y,d,t) * (x_dso(k) - z_tso(k))
      x_dso, z_tso : pf_entry_stride_s39_D.jsonl entries 'x_dso', 'z_tso_current' (= consensus_vars['pf'][dso|tso]
                     ['current'], MW = expected_interface_pf_p * s_base, read at get_admm_boyd_residual_metrics)
      pi           : interface_settlement_detail_s31c.json interface_reporting_detail[n][y][d].periods[t].price_per_mwh
      w            : interface_consensus_residual_per_dso[n].periods['y|d|t'].admm_block_weight
    = the identity of p515_g_g1_g4_admm_gates._s31c_interface_detail: t_tso_plus_t_dso_terminal
      = -sum_n sum_pi_baseMVA_residual_weighted = sum w * pi * (p_DSO - p_TSO) (one market / operation scenario, asserted).
  dQ(k) = Q(k) - Q(k-1), Q = per_cycle_record gross_operational_cost; dt(k) = t_sum(k) - t_sum(k-1);
  dQ_cc(k) = dQ(k) + dt(k)  (Q_cc = Q + t_sum).
  Dual path: D(k) = ||y(k) - y(k-1)||_2 over all 1728 pf entries, y = lambda_pf_dso / dso_base_mva (λ sidecar);
             E(k) = rho_pf(k) * ||r_pf(k)|| (creep boyd.pf.r; rho_pf = stride entry rho_pf in force);
             ratio = D / E; per-entry check |dy - rho * r_entry| from the stride.
  Verdict, fits, quarters: exactly as the preregistration file.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w112_consensus_gap/):
  w112_consensus_gap.json (tasks 1, 2, 3, 5), w112_settlement_scan.json (task 4), launch.log, manifest_sha256.json

Launch (attached, alone, both streams captured), then the manifest:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w112_consensus_gap.py \\
        > data/SRP1/Results/P515S53/w112_consensus_gap/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w112_consensus_gap.py --manifest
"""
import glob
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

_GUARD = SolveProfileGuard((), label='P5.15 W112 consensus gap zero-solve').install()

import gate_result_io as GRIO  # noqa: E402

THIS = os.path.abspath(__file__)
RES = os.path.join('data', 'SRP1', 'Results')
OUT_DIR = os.path.join(REPO, RES, 'P515S53', 'w112_consensus_gap')
OUT_JSON = os.path.join(OUT_DIR, 'w112_consensus_gap.json')
OUT_SCAN = os.path.join(OUT_DIR, 'w112_settlement_scan.json')
PREREG_REL = os.path.join(RES, 'P515S53', 'w112_consensus_gap', 'w112_preregistration_v1_ba384b28.json')
PREREG_SHA_PREFIX = 'ba384b28'

CELLS = {
    'W110_c_star_ext': os.path.join(RES, 'P515S53', 'w105_c_star_extension', 'campaign_s53_w105_c_star_ext_r2',
                                    'evals', '8864266d3c064ed0_c_star_ext'),
    'W104_c_star': os.path.join(RES, 'P515S53', 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_c_star',
                                'evals', '4bf36c151fd10613_c_star'),
    'W86_c_star_87': os.path.join(RES, 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals',
                                  '96c5aa50cc229cc1_c_star'),
    'S47_recert_c_star_87': os.path.join(RES, 'P515S47', 'campaign_s47_recert', 'evals', '070f833e1e318f85_c_star'),
    'W86_x0_old_132': os.path.join(RES, 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals',
                                   '5cfe69a615ae3708_x0'),
    'W101_x0_settled_181': os.path.join(RES, 'P515S53', 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0',
                                        'evals', 'd110bd1a5977df1e_x0'),
    'W86_unit_old_112': os.path.join(RES, 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals',
                                     'ca8927e75d628bd1_n7_4h_e1'),
    'W101_unit_settled_172': os.path.join(RES, 'P515S53', 'w101_srp1_continuation',
                                          'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals', '3f084f2ffaeef2b7_n7_4h_e1'),
}
STRIDE = 'pf_entry_stride_s39_D.jsonl'
DUALS = 'interface_duals_per_cycle.jsonl'
CREEP = 'creep_diagnostic_per_cycle.jsonl'
PCR = 'per_cycle_record.jsonl'
DETAIL = 'interface_settlement_detail_s31c.json'
EVALREC = 'evaluation_record.json'
PLANNER_ENDPOINTS = {'c_star_87_S47': 26306.59, 'c_star_187_W104': 6908.97, 'c_star_287_W110': 15615.70,
                     'x0_132': 23802.19, 'x0_181': 142.53, 'unit_112': 34057.43, 'unit_172': 662.99,
                     'gross_dV': -5835.70}
QUARTERS = [(188, 212), (213, 237), (238, 262), (263, 287)]
QUARTERS_SUPP = [(88, 112), (113, 137), (138, 162), (163, 187)]
VALIDATION_ABS_TOL = 0.01
TRIAGE_NOTE = ('No eval-dir prefix list for the 33 triage cells was found. Searched: '
               'P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md (Triage section; every 8-16 hex token), TASKS.md '
               '(triage lines; 12-16 hex tokens), REVISION_CONTEXT.md and PLANNER_BRIEF_2026-09-13.md Addenda 53-54 '
               '("triage"; hex tokens), repository file names matching *triage*. The report lists triage ITEMS '
               '(B, C, E, G, H, I, J, L) with cell counts, not eval dirs. No cell is flagged.')


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


def _campaign_manifest(eval_rel):
    root = eval_rel.split(os.sep + 'evals' + os.sep)[0]
    path = os.path.join(REPO, root, 'campaign_manifest_sha256.json')
    return (os.path.relpath(path, REPO), json.load(open(path))) if os.path.exists(path) else (None, {})


def _verify_inputs(cell_names, file_names):
    """sha256 of every input against its campaign manifest; raise on a mismatch or a missing entry."""
    out = {}
    for cell in cell_names:
        rel_dir = CELLS[cell]
        man_rel, man = _campaign_manifest(rel_dir)
        for name in file_names:
            rel = os.path.join(rel_dir, name)
            if not os.path.exists(os.path.join(REPO, rel)):
                continue
            now = _sha(os.path.join(REPO, rel))
            pinned = man.get(rel)
            if pinned is None:
                raise RuntimeError(f'{rel}: not hash-recorded in {man_rel}')
            if now != pinned:
                raise RuntimeError(f'{rel}: sha256 {now} != manifest {pinned} ({man_rel})')
            out[rel] = {'sha256': now, 'manifest': man_rel}
    return out


def _read_by_cycle(rel, skip_header=False):
    out = {}
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if skip_header and row.get('header'):
                continue
            if row['cycle'] in out:
                raise RuntimeError(f'{rel}: cycle {row["cycle"]} appears twice')
            out[row['cycle']] = row
    return out


def _price_weight(detail):
    """pi and w keyed (node str, year str, day str, period int), exactly as the identity uses them."""
    pi, w = {}, {}
    for n, by_year in detail['interface_reporting_detail'].items():
        for y, by_day in by_year.items():
            for d, dd in by_day.items():
                # Pre-W35 detail files carry no scenario-count fields (SRP1 has one scenario throughout);
                # the check is applied wherever the field exists.
                if dd.get('n_market_scenarios', 1) != 1 or dd.get('n_operation_scenarios', 1) != 1:
                    raise RuntimeError(f'{n}|{y}|{d}: more than one scenario; the identity form differs')
                for t, pdet in dd['periods'].items():
                    pi[(str(n), str(y), str(d), int(t))] = pdet['price_per_mwh']
    for n, v in detail['interface_consensus_residual_per_dso'].items():
        for key, pdet in v['periods'].items():
            y, d, t = key.split('|')
            w[(str(n), y, d, int(t))] = pdet['admm_block_weight']
    if set(pi) != set(w):
        raise RuntimeError('price and weight key sets differ')
    return pi, w


def _detail_identity(detail):
    residual_side = -sum(v['sum_pi_baseMVA_residual_weighted']
                         for v in detail['interface_consensus_residual_per_dso'].values())
    return {'t_tso_plus_t_dso_terminal': detail['t_tso_plus_t_dso_terminal'],
            'minus_sum_weighted_priced_residual': residual_side,
            'difference': detail['t_tso_plus_t_dso_terminal'] - residual_side,
            'cycles_run': detail['cycles_run'], 'label': detail.get('label'),
            'converged_at_cycle': detail.get('converged_at_cycle')}


def _stream_stride(rel, pi, w, keep_cycles=None):
    """Streams the stride once. Per cycle: t_sum, per node, Boyd r (production and p-only), and the entry-level dual
    check dy - rho * r against the previous cycle's lambda_dso / s_base_dso."""
    per = {}
    prev_y = None
    prev_cycle = None
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            k = row['cycle']
            if not row['identity_holds']:
                raise RuntimeError(f'{rel}: identity_holds false at cycle {k}')
            t_sum, t_node = 0.0, {}
            sumsq_r_p, sumsq_dy, max_dev, rhos = 0.0, 0.0, 0.0, set()
            abs_gap_mw_p = 0.0
            y_now = {}
            for e in row['entries']:
                key = (str(e['node_id']), str(e['year']), str(e['day']), e['power_type'], int(e['period']))
                y_now[key] = e['lambda_dso'] / e['s_base_dso']
                rhos.add(e['rho_pf'])
                if e['power_type'] == 'p':
                    pk = (key[0], key[1], key[2], key[4])
                    term = w[pk] * pi[pk] * (e['x_dso'] - e['z_tso_current'])
                    t_sum += term
                    t_node[key[0]] = t_node.get(key[0], 0.0) + term
                    sumsq_r_p += e['r'] ** 2
                    abs_gap_mw_p += abs(e['x_dso'] - e['z_tso_current'])
                if prev_y is not None:
                    dy = y_now[key] - prev_y[key]
                    sumsq_dy += dy ** 2
                    max_dev = max(max_dev, abs(dy - e['rho_pf'] * e['r']))
            rec = {'t_sum': t_sum, 't_by_node': t_node, 'boyd_pf_r': row['production_boyd_pf_r'],
                   'r_p_only': math.sqrt(sumsq_r_p), 'sum_abs_gap_mw_p': abs_gap_mw_p,
                   'rho_pf_set': sorted(rhos), 'n_entries': len(row['entries'])}
            if prev_y is not None and prev_cycle == k - 1:
                rec['stride_norm_dy'] = math.sqrt(sumsq_dy)
                rec['stride_max_entry_abs_dy_minus_rho_r'] = max_dev
            if keep_cycles is None or k in keep_cycles:
                per[k] = rec
            prev_y, prev_cycle = y_now, k
    return per


def _stream_duals(rel):
    per = {}
    header = None
    prev = None
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('header'):
                header = row
                continue
            if not row.get('captured'):
                raise RuntimeError(f'{rel}: cycle {row.get("cycle")} not captured')
            blocks = header['metadata']['per_block']
            y_dso, y_tso = [], []
            anti = 0.0
            for b, meta in enumerate(blocks):
                for kind in ('p', 'q'):
                    for t in range(header['n_periods']):
                        ld = row[f'lambda_pf_{kind}_dso'][b][t]
                        lt = row[f'lambda_pf_{kind}_tso'][b][t]
                        y_dso.append(ld / meta['dso_base_mva'])
                        y_tso.append(lt / meta['tso_base_mva'])
                        anti = max(anti, abs(ld + lt))
            rec = {'norm_y_dso': math.sqrt(sum(v * v for v in y_dso)), 'max_abs_lambda_tso_plus_dso': anti}
            if prev is not None and prev[0] == row['cycle'] - 1:
                rec['norm_dy_dso'] = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_dso, prev[1])))
                rec['norm_dy_tso'] = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_tso, prev[2])))
            per[row['cycle']] = rec
            prev = (row['cycle'], y_dso, y_tso)
    return header, per


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _ols_slope(xs, ys):
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None


def _fit_primary(means):
    m = means[:3]
    if not (all(v > 0 for v in m) or all(v < 0 for v in m)):
        return {'fittable': False, 'reason': 'quarter means Q1-Q3 do not share a sign (log form undefined)',
                'means_q1_q3': m}
    sgn = 1.0 if m[0] > 0 else -1.0
    xs = [0.0, 1.0, 2.0]
    ls = [math.log(v / sgn) for v in m]
    b = _ols_slope(xs, ls)
    la = _mean(ls) - b * _mean(xs)
    pred = sgn * math.exp(la + b * 3.0)
    obs = means[3]
    return {'fittable': True, 'model': 'm_j = a * exp(b * (j - 1)), LS on log(m_j / sgn)', 'a': sgn * math.exp(la),
            'b': b, 'fitted_q1_q3': [sgn * math.exp(la + b * x) for x in xs],
            'predicted_q4': pred, 'observed_q4': obs, 'error_obs_minus_pred': obs - pred,
            'relative_error': (obs - pred) / abs(obs) if obs else None}


def _fit_secondary(series, primary):
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except Exception as exc:  # noqa: BLE001
        return {'fitted': False, 'reason': f'import failed: {exc!r}'}
    ks = list(range(188, 263))
    ys = [series[k] for k in ks]
    if primary.get('fittable'):
        p0 = [primary['a'], primary['b'] / 25.0]
    else:
        p0 = [_mean(ys), 0.0]

    def f(k, a, b):
        return a * np.exp(b * (k - 188.0))

    try:
        popt, _ = curve_fit(f, np.array(ks, dtype=float), np.array(ys, dtype=float), p0=p0, maxfev=20000)
    except Exception as exc:  # noqa: BLE001
        return {'fitted': False, 'reason': f'curve_fit failed: {exc!r}', 'p0': p0}
    a, b = float(popt[0]), float(popt[1])
    pred = _mean([a * math.exp(b * (k - 188.0)) for k in range(263, 288)])
    obs = _mean([series[k] for k in range(263, 288)])
    return {'fitted': True, 'model': 'y(k) = a * exp(b * (k - 188)), per-cycle 188-262, scipy curve_fit',
            'p0': p0, 'a': a, 'b_per_cycle': b, 'predicted_q4_mean': pred, 'observed_q4_mean': obs,
            'error_obs_minus_pred': obs - pred, 'relative_error': (obs - pred) / abs(obs) if obs else None}


def _quarter_table(dq, dt, quarters):
    rows = []
    for s, e in quarters:
        ks = range(s, e + 1)
        sq, st = sum(dq[k] for k in ks), sum(dt[k] for k in ks)
        rows.append({'cycles': [s, e], 'mean_dQ': sq / len(ks), 'mean_dt_sum': st / len(ks),
                     'mean_dQ_cc': (sq + st) / len(ks), 'sum_dQ': sq, 'sum_dt_sum': st, 'sum_dQ_cc': sq + st,
                     'share_dt_over_dQ': st / sq if sq else None,
                     'share_dt_over_abs_dQ': st / abs(sq) if sq else None})
    return rows


def _verdict(share):
    if share < 0.20:
        return 'H_pf-gap minor'
    if share >= 0.40:
        return 'gross creep is the wrong quantity'
    return 'mixed'


def _eval_record(rel_dir):
    path = os.path.join(REPO, rel_dir, EVALREC)
    return json.load(open(path)) if os.path.exists(path) else None


def _scan(manifest_cache):
    rows = []
    paths = sorted(glob.glob(os.path.join(REPO, RES, '**', DETAIL), recursive=True))
    for path in paths:
        rel = os.path.relpath(path, REPO)
        rel_dir = os.path.dirname(rel)
        detail = json.load(open(path))
        ident = _detail_identity(detail)
        rec = _eval_record(rel_dir)
        sha = _sha(path)
        hash_records = []
        d = os.path.dirname(path)
        stop = os.path.join(REPO, RES)
        while d.startswith(stop) and d != stop:
            for man_path in glob.glob(os.path.join(d, '*manifest*.json')):
                if man_path not in manifest_cache:
                    try:
                        loaded = json.load(open(man_path))
                        manifest_cache[man_path] = loaded if isinstance(loaded, dict) else {}
                    except Exception:  # noqa: BLE001
                        manifest_cache[man_path] = {}
                man = manifest_cache[man_path]
                for key in (rel, rel + ' (input)'):
                    if key in man and isinstance(man[key], str):
                        hash_records.append({'manifest': os.path.relpath(man_path, REPO), 'key': key,
                                             'matches': man[key] == sha})
            d = os.path.dirname(d)
        campaign = rel_dir.split(os.sep + 'evals' + os.sep)[0] if (os.sep + 'evals' + os.sep) in rel_dir else None
        row = {'path': rel, 'stage_dir': rel_dir.split(os.sep)[3] if len(rel_dir.split(os.sep)) > 3 else None,
               'campaign_dir': campaign, 'eval_dir': os.path.basename(rel_dir), 'sha256': sha,
               'label': ident['label'], 'cycles_run': ident['cycles_run'],
               'converged_at_cycle_in_detail': ident['converged_at_cycle'],
               't_sum_terminal': ident['t_tso_plus_t_dso_terminal'],
               'identity_difference_vs_minus_residual': ident['difference'],
               'hash_records': hash_records, 'triage_flag': None}
        if rec is not None:
            cert = rec.get('certification') or {}
            row.update({'eval_key': rec.get('eval_key'), 'campaign_id': rec.get('campaign_id'),
                        'candidate_label': rec.get('candidate_label'), 'candidate_key': rec.get('candidate_key'),
                        'status': rec.get('status'), 'certification_cycle': rec.get('certification_cycle'),
                        'certified': cert.get('certified'), 'certified_cost_gross': rec.get('certified_cost'),
                        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
                        'eval_record_cycles_run': rec.get('cycles_run')})
            gross = rec.get('terminal_gross_operational_cost')
            row['Q_cc_terminal'] = (gross + ident['t_tso_plus_t_dso_terminal']) if gross is not None else None
        else:
            row.update({'eval_key': None, 'status': 'not given in results (no evaluation_record.json in the dir)'})
        rows.append(row)
    return rows


def run():
    t0 = time.time()
    for p in (OUT_JSON, OUT_SCAN):
        if os.path.exists(p):
            raise RuntimeError(f'{p} exists; write-once')
    prereg_sha = _sha(os.path.join(REPO, PREREG_REL))
    if not prereg_sha.startswith(PREREG_SHA_PREFIX):
        raise RuntimeError(f'preregistration sha {prereg_sha} does not start with {PREREG_SHA_PREFIX}')
    prereg = json.load(open(os.path.join(REPO, PREREG_REL)))
    _log(f'preregistration verified: {PREREG_REL} sha256 {prereg_sha}')

    inputs = _verify_inputs(['W110_c_star_ext'], [STRIDE, DUALS, CREEP, PCR, DETAIL])
    inputs.update(_verify_inputs(['W104_c_star', 'W101_x0_settled_181', 'W101_unit_settled_172'],
                                 [STRIDE, PCR, DETAIL]))
    inputs.update(_verify_inputs(['W86_c_star_87', 'S47_recert_c_star_87', 'W86_x0_old_132', 'W86_unit_old_112'],
                                 [DETAIL, PCR]))
    _log(f'{len(inputs)} inputs verified against their campaign manifests')
    evalrecs = {c: _eval_record(CELLS[c]) for c in CELLS}

    details = {c: json.load(open(os.path.join(REPO, CELLS[c], DETAIL))) for c in CELLS}
    identities = {c: _detail_identity(details[c]) for c in CELLS}
    pi, w = _price_weight(details['W110_c_star_ext'])
    pw_equal = {}
    for c in CELLS:
        pi_c, w_c = _price_weight(details[c])
        pw_equal[c] = {'price_equal_bitwise': pi_c == pi, 'weight_equal_bitwise': w_c == w}
    _log(f'price / weight maps: {len(pi)} keys; equal across cells: {pw_equal}')
    del details

    # ---------------- task 1: per-cycle t_sum (C*) ----------------
    w110 = _stream_stride(os.path.join(CELLS['W110_c_star_ext'], STRIDE), pi, w)
    if sorted(w110) != list(range(1, 288)):
        raise RuntimeError(f'W110 stride cycles {min(w110)}..{max(w110)} n={len(w110)}')
    _log(f'W110 stride streamed: t(87) {w110[87]["t_sum"]!r} t(187) {w110[187]["t_sum"]!r} t(287) {w110[287]["t_sum"]!r}')
    w104 = _stream_stride(os.path.join(CELLS['W104_c_star'], STRIDE), pi, w)
    replay = {'cycles_compared': [min(w104), max(w104)],
              'n_cycles_t_sum_bitwise_equal': sum(1 for k in w104 if w104[k]['t_sum'] == w110[k]['t_sum']),
              'n_cycles': len(w104),
              'max_abs_t_sum_difference': max(abs(w104[k]['t_sum'] - w110[k]['t_sum']) for k in w104)}
    _log(f'W104 vs W110 stride replay: {replay}')

    def _val(computed, ident, tol=VALIDATION_ABS_TOL):
        d1 = computed - ident['t_tso_plus_t_dso_terminal']
        d2 = computed - ident['minus_sum_weighted_priced_residual']
        return {'computed_from_stride': computed, 'terminal_t_tso_plus_t_dso': ident['t_tso_plus_t_dso_terminal'],
                'terminal_minus_sum_weighted_residual': ident['minus_sum_weighted_priced_residual'],
                'diff_vs_settlement_side': d1, 'diff_vs_residual_side': d2, 'pass_abs_le_tol': abs(d1) <= tol,
                'tol_eur': tol, 'terminal_cycles_run': ident['cycles_run']}

    validation = {
        't287_vs_W110_terminal': _val(w110[287]['t_sum'], identities['W110_c_star_ext']),
        't187_vs_W104_terminal': _val(w110[187]['t_sum'], identities['W104_c_star']),
        'supplementary_t87_vs_W86_c_star_terminal': _val(w110[87]['t_sum'], identities['W86_c_star_87']),
        'supplementary_t87_vs_S47_recert_c_star_terminal': _val(w110[87]['t_sum'], identities['S47_recert_c_star_87']),
        'planner_endpoints': PLANNER_ENDPOINTS,
    }
    gated_pass = (validation['t287_vs_W110_terminal']['pass_abs_le_tol']
                  and validation['t187_vs_W104_terminal']['pass_abs_le_tol'])
    _log(f'validation gated pass {gated_pass}: '
         f'{ {k: (v["diff_vs_settlement_side"], v["pass_abs_le_tol"]) for k, v in validation.items() if isinstance(v, dict) and "pass_abs_le_tol" in v} }')

    # ---------------- task 2 ----------------
    pcr = _read_by_cycle(os.path.join(CELLS['W110_c_star_ext'], PCR))
    pcr104 = _read_by_cycle(os.path.join(CELLS['W104_c_star'], PCR))
    q = {k: pcr[k]['gross_operational_cost'] for k in pcr}
    q_replay_bitwise = all(pcr104[k]['gross_operational_cost'] == q[k] for k in pcr104)
    t = {k: w110[k]['t_sum'] for k in w110}
    dq = {k: q[k] - q[k - 1] for k in range(2, 288)}
    dt = {k: t[k] - t[k - 1] for k in range(2, 288)}
    dqcc = {k: dq[k] + dt[k] for k in dq}
    quarters = _quarter_table(dq, dt, QUARTERS)
    quarters_supp = _quarter_table(dq, dt, QUARTERS_SUPP)
    dQ_total = q[287] - q[187]
    dt_total = t[287] - t[187]
    share_overall = dt_total / abs(dQ_total)
    verdict = _verdict(share_overall)
    fits = {}
    for name, series in (('dQ', dq), ('dQ_cc', dqcc)):
        means = [_mean([series[k] for k in range(s, e + 1)]) for s, e in QUARTERS]
        prim = _fit_primary(means)
        fits[name] = {'quarter_means': means, 'primary': prim, 'secondary': _fit_secondary(series, prim)}
    _log(f'task 2: dQ_total {dQ_total!r} dt_total {dt_total!r} share {share_overall!r} -> {verdict}')

    # ---------------- task 3: dual path ----------------
    duals_header, duals = _stream_duals(os.path.join(CELLS['W110_c_star_ext'], DUALS))
    creep = {}
    with open(os.path.join(REPO, CELLS['W110_c_star_ext'], CREEP)) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                creep[row['cycle']] = {'r': row['boyd']['pf']['r'], 'norm_y': row['boyd']['pf']['norm_y'],
                                       's': row['boyd']['pf']['s'], 'gross': row['gross']}
    aa = _read_by_cycle(os.path.join(CELLS['W110_c_star_ext'], 'aa_per_cycle.jsonl'))
    dual_rows = []
    for k in range(2, 288):
        rhos = w110[k]['rho_pf_set']
        rho = rhos[0] if len(rhos) == 1 else None
        E = rho * creep[k]['r'] if rho is not None else None
        D = duals[k].get('norm_dy_dso')
        dual_rows.append({
            'cycle': k, 'rho_pf': rho, 'rho_pf_set': rhos, 'boyd_pf_r': creep[k]['r'],
            'norm_dy_dso': D, 'norm_dy_tso': duals[k].get('norm_dy_tso'), 'rho_times_r': E,
            'ratio_D_over_E': (D / E) if (D is not None and E) else None,
            'stride_norm_dy': w110[k].get('stride_norm_dy'),
            'stride_max_entry_abs_dy_minus_rho_r': w110[k].get('stride_max_entry_abs_dy_minus_rho_r'),
            'creep_r_equals_stride_production_r': creep[k]['r'] == w110[k]['boyd_pf_r'],
            'creep_norm_y_minus_sidecar_norm_y': creep[k]['norm_y'] - duals[k]['norm_y_dso'],
            'max_abs_lambda_tso_plus_dso': duals[k]['max_abs_lambda_tso_plus_dso'],
            'aa_accepted_prev_cycle': aa[k - 1]['aa_accepted'], 'pcr_rho_pf_after_prev': pcr[k - 1]['rho_pf_after'],
            'local_solves_ok': pcr[k]['local_solves_ok']})
    by_k = {r['cycle']: r for r in dual_rows}

    def _ratio_stats(s, e):
        rs = [by_k[k]['ratio_D_over_E'] for k in range(s, e + 1)]
        ks = list(range(s, e + 1))
        return {'cycles': [s, e], 'mean': _mean(rs), 'min': min(rs), 'max': max(rs), 'ols_slope_per_cycle': _ols_slope(ks, rs),
                'max_entry_abs_dy_minus_rho_r': max(by_k[k]['stride_max_entry_abs_dy_minus_rho_r'] for k in ks),
                'max_rel_err_stride_vs_sidecar_norm_dy': max(abs(by_k[k]['stride_norm_dy'] - by_k[k]['norm_dy_dso'])
                                                             / by_k[k]['norm_dy_dso'] for k in ks),
                'all_creep_r_equal_stride_r': all(by_k[k]['creep_r_equals_stride_production_r'] for k in ks),
                'max_abs_creep_norm_y_minus_sidecar': max(abs(by_k[k]['creep_norm_y_minus_sidecar_norm_y']) for k in ks),
                'max_abs_lambda_tso_plus_dso': max(by_k[k]['max_abs_lambda_tso_plus_dso'] for k in ks),
                'any_aa_accepted_prev': any(by_k[k]['aa_accepted_prev_cycle'] for k in ks),
                'rho_pf_values': sorted({by_k[k]['rho_pf'] for k in ks}),
                'all_local_solves_ok': all(by_k[k]['local_solves_ok'] for k in ks)}
    dual_summary = {'88_287': _ratio_stats(88, 287), '188_287': _ratio_stats(188, 287),
                    'quarters': [_ratio_stats(s, e) for s, e in QUARTERS_SUPP + QUARTERS]}
    _log(f'task 3: ratio 88-287 {dual_summary["88_287"]["mean"]!r} slope {dual_summary["88_287"]["ols_slope_per_cycle"]!r}')

    # ---------------- task 5: settled references ----------------
    refs = {}
    for name, old_cell, set_cell, k_old, k_set in (('x0', 'W86_x0_old_132', 'W101_x0_settled_181', 132, 181),
                                                   ('unit', 'W86_unit_old_112', 'W101_unit_settled_172', 112, 172)):
        stride = _stream_stride(os.path.join(CELLS[set_cell], STRIDE), pi, w)
        pcr_set = _read_by_cycle(os.path.join(CELLS[set_cell], PCR))
        q_old = evalrecs[old_cell]['certified_cost']
        q_set = evalrecs[set_cell]['certified_cost']
        t_old = identities[old_cell]['t_tso_plus_t_dso_terminal']
        t_set = identities[set_cell]['t_tso_plus_t_dso_terminal']
        refs[name] = {
            'old_cell': CELLS[old_cell], 'settled_cell': CELLS[set_cell],
            'candidate_key_old': evalrecs[old_cell]['candidate_key'],
            'candidate_key_settled': evalrecs[set_cell]['candidate_key'],
            'k_old': k_old, 'k_settled': k_set,
            'cert_cycle_old': evalrecs[old_cell]['certification_cycle'],
            'cert_cycle_settled': evalrecs[set_cell]['certification_cycle'],
            'Q_old': q_old, 'Q_settled': q_set, 't_old': t_old, 't_settled': t_set,
            'Qcc_old': q_old + t_old, 'Qcc_settled': q_set + t_set,
            's_gross': q_set - q_old, 's_cc': (q_set + t_set) - (q_old + t_old),
            'cross_check_continuation_stride': {
                't_at_k_old_from_settled_run_stride': stride[k_old]['t_sum'],
                'diff_vs_old_terminal': stride[k_old]['t_sum'] - t_old,
                't_at_k_settled_from_stride': stride[k_set]['t_sum'],
                'diff_vs_settled_terminal': stride[k_set]['t_sum'] - t_set,
                'Q_at_k_old_in_settled_run_pcr': pcr_set[k_old]['gross_operational_cost'],
                'Q_at_k_old_bitwise_equal_old_cert': pcr_set[k_old]['gross_operational_cost'] == q_old},
            't_series_k_old_to_k_settled': {str(k): stride[k]['t_sum'] for k in range(k_old, k_set + 1)},
        }
        del stride
    V = {'V_old_gross': refs['x0']['Q_old'] - refs['unit']['Q_old'],
         'V_settled_gross': refs['x0']['Q_settled'] - refs['unit']['Q_settled'],
         'V_old_cc': refs['x0']['Qcc_old'] - refs['unit']['Qcc_old'],
         'V_settled_cc': refs['x0']['Qcc_settled'] - refs['unit']['Qcc_settled']}
    V['dV_gross'] = V['V_settled_gross'] - V['V_old_gross']
    V['dV_cc'] = V['V_settled_cc'] - V['V_old_cc']
    V['dV_gross_planner'] = PLANNER_ENDPOINTS['gross_dV']
    c_star_cc = {str(k): {'Q': q[k], 't_sum': t[k], 'Q_cc': q[k] + t[k]} for k in (87, 187, 287)}
    _log(f'task 5: {V}')

    # ---------------- task 4: scan ----------------
    scan_rows = _scan({})
    _log(f'task 4: {len(scan_rows)} settlement detail files scanned')

    guard_failures = _GUARD.verify(0)
    per_cycle = [{'cycle': k, 'Q': q[k], 't_sum': t[k], 'Q_cc': q[k] + t[k],
                  'dQ': dq.get(k), 'dt_sum': dt.get(k), 'dQ_cc': dqcc.get(k),
                  't_by_node': w110[k]['t_by_node'], 'boyd_pf_r': w110[k]['boyd_pf_r'],
                  'r_p_only': w110[k]['r_p_only'], 'sum_abs_gap_mw_p': w110[k]['sum_abs_gap_mw_p']}
                 for k in range(1, 288)]
    common = {'utc': _utc(),
              'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                         cwd=REPO).stdout.strip(),
              'script_sha256': _sha(THIS), 'preregistration': {'path': PREREG_REL, 'sha256': prereg_sha},
              'solve_profile_guard': {'permitted': [], 'verify_0_failures': guard_failures,
                                      'counts': dict(_GUARD.counts)}}
    result = {
        'stage': 'P5.15 W112 priced interface-consensus gap (records only, zero solves)', **common,
        'objective_convention': 'Q = gross_operational_cost (settlement excluded); Q_cc = Q + t_sum (t_sum = '
                                't_tso_plus_t_dso identity, block-weighted)',
        'instance': {'cell': CELLS['W110_c_star_ext'],
                     'candidate_key': evalrecs['W110_c_star_ext']['candidate_key'],
                     'eval_key': evalrecs['W110_c_star_ext']['eval_key']},
        'inputs_sha256': inputs, 'price_weight_equal_across_cells': pw_equal, 'terminal_identities': identities,
        'task1': {'source': 'per-entry pf copies (pf_entry_stride_s39_D.jsonl), no interpolation',
                  'validation': validation, 'validation_gated_pass': gated_pass, 'w104_replay_t_sum': replay,
                  'Q_replay_bitwise_1_187': q_replay_bitwise},
        'task2': {'quarters_188_287': quarters, 'supplementary_quarters_88_187': quarters_supp,
                  'dQ_total_188_287': dQ_total, 'dt_total_188_287': dt_total, 'dQcc_total_188_287': dQ_total + dt_total,
                  'share_overall_signed': share_overall, 'share_overall_abs': abs(dt_total) / abs(dQ_total),
                  'verdict_rule': prereg['verdict_rule'], 'verdict': verdict, 'fits': fits,
                  'c_star_Q_cc_at': c_star_cc},
        'task3': {'summary': dual_summary, 'per_cycle': dual_rows,
                  'duals_header_source': duals_header.get('source')},
        'task5': {'references': refs, 'V': V},
        'per_cycle': per_cycle,
        'wall_s': time.time() - t0,
    }
    scan = {'stage': 'P5.15 W112 task 4 -- terminal t_sum of every interface_settlement_detail_s31c.json under '
                     'data/SRP1/Results', **common,
            'searched': f'glob {RES}/**/{DETAIL} (recursive), all subdirectories', 'n_files': len(scan_rows),
            'triage_note': TRIAGE_NOTE, 'rows': scan_rows}
    with open(OUT_JSON, 'x') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    with open(OUT_SCAN, 'x') as handle:
        GRIO.dump(scan, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    _log(f'wrote {OUT_JSON} and {OUT_SCAN}; guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; '
         f'wall {time.time() - t0:.1f} s')
    return 0 if (not guard_failures and gated_pass) else 1


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
