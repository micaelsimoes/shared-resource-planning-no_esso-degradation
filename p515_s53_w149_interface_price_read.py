"""P5.15 W149 -- interface prices of h_f9eae48f for the dead-zone table (Addendum 63). ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end. Importing `p515_s53_w127_stall_constraints` (whose read-only KKT functions are reused for part B) installs that
module's own zero-solve guard as well; BOTH guards are verified at exactly 0.

PART A -- records only (committed JSON / JSONL, every input git-tracked, clean against HEAD and sha256-verified against
its campaign manifest). Cells:
  h_f9eae48f  campaign s53_w142_resettle_v6_h_f9eae48f, eval aed2d618..., candidate key db77e154... (2025; n7 0.25 MVA
              1.0 MWh; m 1.5); uncertified at cap 194 (gap clause). Window W = 138..194 (the Planner's stationary window).
  h_aa8a76d7  control x = 0 at m 1.5, eval 22dd5ef8..., candidate key 8435c718...; certified 153 (records end at 153).
              Window = 138..153 (the records stop at the certification cycle).
  f2_challenger / f2_incumbent  the F2 precedent (campaigns s53_w118_resettle_r2_f2_*; eval 1fe91e86... / 24c5ccb6...),
              windows 188..261 (W126's W3) and 208..281 (same length, ending at the incumbent's cap).
PART B -- terminal persisted models of the two h cells (certified_models.pkl, NOT committed; sha256 verified against the
campaign manifest AND post_certification.json BEFORE unpickling), read with W127's `Block` / `tso_block` / `dso_block`
(values, bounds, Params and the IPOPT suffixes only; W127's finite-difference curvature probes restore every value
bit-exactly). Precedent: W127 loaded certified_models.pkl under the same armed zero-solve guard. The F2 row's model
evidence is read from W127's committed JSON (challenger only; W127 never loaded the incumbent).

FORMULAS (production facts, W126 docstring; identities CHECKED here, not assumed):
  pf is two-block ADMM, NO separate consensus variable: x = DSO interface copy, z = TSO interface copy.
  Dual update (_update_interface_power_flow_variables):  lambda_dso += rho_pf (x - z) / rating * B_dso,
  lambda_tso += rho_pf (z - x) / rating * B_tso.  The capture (interface_duals_per_cycle) is read after the cycle's
  dual update and before AA.  Hence, with rho constant and no AA step accepted on k-1 or k:
      gap_k = x_k - z_k [MW] = (lambda_dso(k) - lambda_dso(k-1)) * rating / (rho_pf * B_dso)        (DERIVED)
  CHECKED per cycle against the recorded t_sum, t_by_node and sum_abs_gap_mw_p (resettle_cycle_record), and at the
  terminal cycle entry by entry against interface_settlement_detail residual_mw (= z - x).
  t_sum = sum_{p entries} w * pi * (x - z)  (W112 identity; w = admm_block_weight, pi = price_per_mwh).
  Interface prices, EUR/MWh, LINEAR form of uncoordinated_benchmark.interface_price_terms (the dual_vars value is the
  Param dual_pf_p_req * B of the next solve; eff = admm_objective_scale of that side):
      lambda_DSO = pi + eff_dso * (lambda_pf_p_dso / B_dso) / rating        lambda_TSO = pi - eff_tso * (lambda_pf_p_tso / B_tso) / rating
  The FULL (AL-inclusive) form needs each side's E and z per cycle (pf_entry_stride, untracked) -- not read; at the
  terminal cycle W127's dso_block gives the DSO pf AL gradient on x from the model.

RULES (fixed in this file BEFORE the run; thresholds set after an exploratory records-only look at h_f9eae48f, before
any model was loaded):
  price stationary      |lambda(W_end) - lambda(W_start)| <= PRICE_STAT_EUR (0.01 EUR/MWh) on that entry.
  gap stationary        relative change of the entry gap over the window < GAP_STAT_REL (0.10).
  contributing entries  the smallest prefix of p entries, ranked by terminal contribution of t_sum's sign, whose sum
                        reaches CONTRIB_CUM (0.80) of the terminal t_sum.
  model blocks          (year, day) blocks carrying >= BLOCK_SHARE_MIN (0.05) of the terminal t_sum (summed over
                        nodes) in h_f9eae48f; all 24 periods, TSO + every DSO; the same blocks on the control.
  TN at a bound (entry) every TSO generator classed 'costed (conventional)' by W127 (positive objective gradient) is
                        at its lb or ub (W127 `_at`, 1e-6 relative) in that period.
  DN flex at the kink   every flex_p_up and flex_p_down of that DSO at lb in that period AND W127's cheapest feasible
                        import-raising lever is a flexibility variable (the F2 terms of W127).
  verdict  reproduced            both kinks on every contributing entry, and every contributing node's block gap stationary;
           partially reproduced  either kink on >= half of the contributing entries;
           not reproduced        otherwise.
  REPORT-ONLY fields (prefix `report_only_`; NOT in the verdict): added after the --dry smoke and one scratch run
  (output in the session scratchpad, outside the repository; not evidence) had shown the model reads; the verdict rule above was not changed.
    DN flex interior      some flex_p_up / flex_p_down of that DSO strictly between its bounds in that period.
    TSO lever = shared ESS  W127's cheapest feasible injection-lowering TSO lever is a shared_es_* variable (F2: yes).

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w149_h_f9eae48f_prices/):
  w149_interface_price_read.json, launch.log (both streams, captured by the launcher), manifest_sha256.json (--manifest)
Smoke (nothing written): add --dry. Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w149_h_f9eae48f_prices && set -o noclobber && \\
    nice -n 10 /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w149_interface_price_read.py \\
        > data/SRP1/Results/P515S53/w149_h_f9eae48f_prices/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w149_interface_price_read.py --manifest
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W149 interface price read zero-solve').install()

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w127_stall_constraints as W127  # noqa: E402 -- installs W127's own zero-solve guard (verified too)

THIS = os.path.abspath(__file__)
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'w149_h_f9eae48f_prices')
OUT_JSON = os.path.join(OUT_DIR, 'w149_interface_price_read.json')
V6 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6')
W118 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle')
CELLS = {
    'h_f9eae48f': {'campaign': os.path.join(V6, 'campaign_s53_w142_resettle_v6_h_f9eae48f'),
                   'eval': 'aed2d618a19ca483_h_f9eae48f', 'window': (138, 194), 'model': True},
    'h_aa8a76d7': {'campaign': os.path.join(V6, 'campaign_s53_w142_resettle_v6_h_aa8a76d7'),
                   'eval': '22dd5ef8a9223f56_h_aa8a76d7', 'window': (138, 153), 'model': True},
    'f2_challenger': {'campaign': os.path.join(W118, 'campaign_s53_w118_resettle_r2_f2_challenger'),
                      'eval': '1fe91e86f11e76af_f2_challenger', 'window': (188, 261), 'model': False},
    'f2_incumbent': {'campaign': os.path.join(W118, 'campaign_s53_w118_resettle_r2_f2_incumbent'),
                     'eval': '24c5ccb6f285219f_f2_incumbent', 'window': (208, 281), 'model': False},
}
SUBJECT, CONTROL = 'h_f9eae48f', 'h_aa8a76d7'
RECORD_FILES = ('interface_duals_per_cycle.jsonl', 'resettle_cycle_record.jsonl', 'per_cycle_record.jsonl',
                'aa_per_cycle.jsonl', 'interface_settlement_detail_s31c.json', 'evaluation_record.json',
                'resettle_decision.json', 'network_ipopt_solve_records.jsonl', 'component_levels_terminal.json',
                'interface_voltage_terminal.json')
F2_ARTEFACTS = {  # committed W126 / W127 outputs, each verified against its own committed manifest
    'w126': (os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w126_pf_stall', 'w126_pf_stall.json'),
             os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w126_pf_stall', 'manifest_sha256.json')),
    'w127': (os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w127_stall_constraints', 'w127_stall_constraints.json'),
             os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w127_stall_constraints', 'manifest_sha256.json')),
}
NODES = (5, 7, 9)
PRICE_STAT_EUR = 0.01
GAP_STAT_REL = 0.10
CONTRIB_CUM = 0.80
BLOCK_SHARE_MIN = 0.05
TOP_N = 15
NETNAME = {5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
NOT_IN_COMMITTED_RECORDS = [
    'TSO generator outputs and limits per hour (component_levels_terminal.json carries block-level cost totals only)',
    'DSO flexibility per hour and its bounds (interface_settlement_detail carries per-DSO |delta| volume totals only)',
    'per-cycle interface copies x (DSO) and z (TSO) per entry (pf_entry_stride_s39_D.jsonl, hash-recorded, untracked); '
    'derived here from the dual increments and checked',
    'a separate pf consensus variable / consensus dual (pf is two-block ADMM; none exists)',
    'the FULL (AL-inclusive) interface price per cycle (needs E and z per side per cycle: the untracked stride)',
]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=REPO)


def _f(x):
    return None if x is None else float(x)


def _verify_records(cell):
    c = CELLS[cell]
    man_rel = os.path.join(c['campaign'], 'campaign_manifest_sha256.json')
    man = json.load(open(os.path.join(REPO, man_rel)))
    out = {}
    for name in RECORD_FILES:
        rel = os.path.join(c['campaign'], 'evals', c['eval'], name)
        if not os.path.exists(os.path.join(REPO, rel)):
            out[rel] = {'present': False}
            continue
        now = W127._sha(os.path.join(REPO, rel))
        if man.get(rel) != now:
            raise RuntimeError(f'{rel}: sha256 {now} != campaign manifest {man.get(rel)}')
        if _git('ls-files', '--error-unmatch', rel).returncode != 0:
            raise RuntimeError(f'{rel}: not git-tracked (part A reads committed records only)')
        if _git('diff', '--quiet', 'HEAD', '--', rel).returncode != 0:
            raise RuntimeError(f'{rel}: not clean against HEAD')
        out[rel] = {'present': True, 'sha256': now, 'manifest': man_rel, 'git_tracked': True, 'clean_vs_head': True}
    return out


def _jsonl(rel):
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _ols(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    b = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
    res = ys - (ys.mean() + b * (xs - xs.mean()))
    return {'slope_per_cycle': float(b), 'max_abs_residual': float(np.abs(res).max())}


# ------------------------------------------------------------------------------------------------ part A (records)
def records_cell(cell):
    c = CELLS[cell]
    ev = os.path.join(c['campaign'], 'evals', c['eval'])
    w0, w1 = c['window']
    inputs = _verify_records(cell)
    evalrec = json.load(open(os.path.join(REPO, ev, 'evaluation_record.json')))
    rows = {}
    header = None
    for r in _jsonl(os.path.join(ev, 'interface_duals_per_cycle.jsonl')):
        if r.get('header'):
            header = r
            continue
        if not r.get('captured'):
            raise RuntimeError(f'{cell}: duals cycle {r.get("cycle")} not captured')
        rows[r['cycle']] = r
    K = max(rows)
    if w1 != K or sorted(rows) != list(range(1, K + 1)):
        raise RuntimeError(f'{cell}: dual capture cycles 1..{K}, window end {w1}')
    blocks = [(int(b[0]), str(b[1]), str(b[2])) for b in header['blocks']]
    meta = header['metadata']['per_block']
    for b, m in zip(blocks, meta):
        if (int(m['node_id']), str(m['year']), str(m['day'])) != b:
            raise RuntimeError(f'{cell}: header block order mismatch')
    T_ = header['n_periods']
    B_d = np.array([m['dso_base_mva'] for m in meta])
    B_t = np.array([m['tso_base_mva'] for m in meta])
    rat = np.array([m['interface_rating_mva'] for m in meta])
    eff_d = np.array([m['admm_objective_scale_dso'] for m in meta])
    eff_t = np.array([m['admm_objective_scale_tso'] for m in meta])
    det = json.load(open(os.path.join(REPO, ev, 'interface_settlement_detail_s31c.json')))
    pi = np.zeros((len(blocks), T_))
    w = np.zeros((len(blocks), T_))
    resid = np.zeros((len(blocks), T_))
    for i, (n, y, d) in enumerate(blocks):
        dd = det['interface_reporting_detail'][str(n)][y][d]
        if dd.get('n_market_scenarios', 1) != 1 or dd.get('n_operation_scenarios', 1) != 1:
            raise RuntimeError(f'{cell}: more than one scenario in {n}|{y}|{d}')
        for t in range(T_):
            pi[i, t] = dd['periods'][str(t)]['price_per_mwh']
            pr = det['interface_consensus_residual_per_dso'][str(n)]['periods'][f'{y}|{d}|{t}']
            w[i, t] = pr['admm_block_weight']
            resid[i, t] = pr['residual_mw']
    pcr = {r['cycle']: r for r in _jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))}
    aa = {r['cycle']: r for r in _jsonl(os.path.join(ev, 'aa_per_cycle.jsonl'))}
    rs = {int(r['cycle']): r for r in _jsonl(os.path.join(ev, 'resettle_cycle_record.jsonl'))}
    lam_d = np.array([rows[k]['lambda_pf_p_dso'] for k in range(1, K + 1)])
    lam_t = np.array([rows[k]['lambda_pf_p_tso'] for k in range(1, K + 1)])
    ks = list(range(w0, w1 + 1))
    # preconditions of the derivation: rho constant, no AA accepted, on w0-1..K
    rho_set = sorted({float(pcr[k]['rho_pf_after']) for k in range(w0 - 2, K + 1)})
    aa_acc = [k for k in range(w0 - 1, K + 1) if aa[k]['aa_accepted']]
    if len(rho_set) != 1 or aa_acc:
        raise RuntimeError(f'{cell}: derivation preconditions fail: rho {rho_set}, AA accepted {aa_acc}')
    rho = rho_set[0]
    gap = {k: (lam_d[k - 1] - lam_d[k - 2]) * rat[:, None] / (rho * B_d[:, None]) for k in ks}   # x - z, MW
    wpi = w * pi
    node_of = np.array([b[0] for b in blocks])
    chk = []
    for k in ks:
        r = rs[k]
        tb = {int(n): float(v) for n, v in r['t_by_node'].items()}
        term = wpi * gap[k]
        chk.append({'cycle': k, 't_sum_derived': float(term.sum()), 't_sum_recorded': float(r['t_sum']),
                    'abs_diff': abs(float(term.sum()) - float(r['t_sum'])),
                    'max_abs_diff_by_node': max(abs(float(term[node_of == n].sum()) - tb[n]) for n in tb),
                    'sum_abs_gap_derived': float(np.abs(gap[k]).sum()),
                    'sum_abs_gap_recorded': float(r['sum_abs_gap_mw_p'])})
    identity = {
        'rho_pf': rho, 'aa_accepted_in_window': aa_acc,
        'max_abs_diff_t_sum_eur': max(c_['abs_diff'] for c_ in chk),
        'max_abs_diff_t_by_node_eur': max(c_['max_abs_diff_by_node'] for c_ in chk),
        'max_abs_diff_sum_abs_gap_mw': max(abs(c_['sum_abs_gap_derived'] - c_['sum_abs_gap_recorded']) for c_ in chk),
        'terminal_max_abs_diff_vs_settlement_residual_mw': float(np.abs(gap[K] + resid).max()),
        'terminal_max_abs_residual_mw': float(np.abs(resid).max()),
        'max_abs_lambda_tso_plus_lambda_dso_window': float(np.abs(lam_t[w0 - 1:] + lam_d[w0 - 1:]).max()),
    }
    if identity['max_abs_diff_t_sum_eur'] > 1e-4 or identity['terminal_max_abs_diff_vs_settlement_residual_mw'] > 1e-9:
        raise RuntimeError(f'{cell}: derived-gap identity fails: {identity}')
    price_d = {k: pi + eff_d[:, None] * (lam_d[k - 1] / B_d[:, None]) / rat[:, None] for k in ks}
    price_t = {k: pi - eff_t[:, None] * (lam_t[k - 1] / B_t[:, None]) / rat[:, None] for k in ks}
    t_term = wpi * gap[K]
    t_sum_K = float(t_term.sum())

    def ekey(i, t):
        n, y, d = blocks[i]
        return {'node_id': n, 'year': y, 'day': d, 'period': t, 'hour': t + 1}

    def entry(i, t, history=False):
        g = np.array([gap[k][i, t] for k in ks])
        pd_ = np.array([price_d[k][i, t] for k in ks])
        pt_ = np.array([price_t[k][i, t] for k in ks])
        out = {**ekey(i, t), 'pi_eur_mwh': float(pi[i, t]), 'admm_block_weight': float(w[i, t]),
               'contribution_t_terminal_eur': float(t_term[i, t]),
               'share_of_t_sum_terminal': float(t_term[i, t] / t_sum_K) if t_sum_K else None,
               'contribution_t_window_mean_eur': float(np.mean([wpi[i, t] * gap[k][i, t] for k in ks])),
               'gap_x_minus_z_mw': {'start': float(g[0]), 'end': float(g[-1]), 'min': float(g.min()), 'max': float(g.max()),
                                    'rel_change': float(abs(g[-1] - g[0]) / abs(g[0])) if g[0] else None,
                                    'ols': _ols(ks, g)},
               'lambda_dso_eur_mwh': {'start': float(pd_[0]), 'end': float(pd_[-1]), 'change': float(pd_[-1] - pd_[0]),
                                      'ols': _ols(ks, pd_)},
               'lambda_tso_eur_mwh': {'start': float(pt_[0]), 'end': float(pt_[-1]), 'change': float(pt_[-1] - pt_[0])},
               'spread_dso_minus_tso_eur_mwh_max_abs_window': float(np.abs(pd_ - pt_).max()),
               'dual_term_eur_mwh_end': float(pd_[-1] - pi[i, t]),
               'price_stationary': bool(abs(pd_[-1] - pd_[0]) <= PRICE_STAT_EUR and abs(pt_[-1] - pt_[0]) <= PRICE_STAT_EUR)}
        if history:
            out['history'] = {'cycles': ks, 'gap_mw': g.tolist(), 'lambda_dso_eur_mwh': pd_.tolist(),
                              'lambda_tso_eur_mwh': pt_.tolist(), 'raw_lambda_pf_p_dso': [float(lam_d[k - 1][i, t]) for k in ks],
                              'raw_lambda_pf_p_tso': [float(lam_t[k - 1][i, t]) for k in ks]}
        return out

    flat = [(i, t) for i in range(len(blocks)) for t in range(T_)]
    sgn = 1.0 if t_sum_K >= 0 else -1.0
    ranked = sorted(flat, key=lambda it: -sgn * t_term[it])
    cum, contrib = 0.0, []
    for it in ranked:
        cum += t_term[it]
        contrib.append(it)
        if abs(cum) >= CONTRIB_CUM * abs(t_sum_K):
            break
    top = sorted(flat, key=lambda it: -abs(t_term[it]))[:TOP_N]
    by_block = {}
    for i, (n, y, d) in enumerate(blocks):
        g0, g1 = gap[w0][i].sum(), gap[K][i].sum()
        by_block[f'{n}|{y}|{d}'] = {'t_terminal_eur': float(t_term[i].sum()), 'share_terminal': float(t_term[i].sum() / t_sum_K),
                                    't_window_mean_eur': float(np.mean([(wpi[i] * gap[k][i]).sum() for k in ks])),
                                    'sum_gap_mw_start': float(g0), 'sum_gap_mw_end': float(g1),
                                    'sum_abs_gap_mw_end': float(np.abs(gap[K][i]).sum()),
                                    'block_gap_rel_change': float(abs(g1 - g0) / abs(g0)) if g0 else None,
                                    'block_gap_stationary': bool(g0 != 0 and abs(g1 - g0) / abs(g0) < GAP_STAT_REL)}
    by_block = dict(sorted(by_block.items(), key=lambda kv: kv[1]['t_terminal_eur'] * -sgn))
    by_yd = {}
    for i, (n, y, d) in enumerate(blocks):
        by_yd[f'{y}|{d}'] = by_yd.get(f'{y}|{d}', 0.0) + float(t_term[i].sum())
    by_hour = {}
    for i, t in flat:
        by_hour[t + 1] = by_hour.get(t + 1, 0.0) + float(t_term[i, t])
    all_entries = [entry(i, t) for i, t in flat]
    n_price_moving = sum(1 for e in all_entries if not e['price_stationary'])
    n_gap1mw = sum(1 for e in all_entries if abs(e['gap_x_minus_z_mw']['end']) > 1e-3)
    dec = json.load(open(os.path.join(REPO, ev, 'resettle_decision.json')))
    ip = {'_round_range_in_file': None}
    rounds = []
    for r in _jsonl(os.path.join(ev, 'network_ipopt_solve_records.jsonl')):
        rounds.append(r['round'])
        if w0 <= r['round'] <= K:
            key = f"{r['network']}|{r['year']}|{r['day']}"
            s = ip.setdefault(key, {'n': 0, 'non_primary_or_non_optimal': []})
            s['n'] += 1
            if r['attempt'] != 'primary' or r['exit'] != 'Optimal Solution Found.':
                s['non_primary_or_non_optimal'].append({kk: r[kk] for kk in ('round', 'attempt', 'exit')})
    ip['_round_range_in_file'] = [min(rounds), max(rounds)]
    ip['_window_filter'] = f"{w0} <= round <= {K}, 'round' as recorded (W126 convention: round read as the cycle)"
    base_yd = {}
    for i, (n, y, d) in enumerate(blocks):
        if B_d[i] != B_t[i] or base_yd.get(f'{y}|{d}', B_d[i]) != B_d[i]:
            raise RuntimeError(f'{cell}: base MVA differs across sides / nodes in {y}|{d}')
        base_yd[f'{y}|{d}'] = float(B_d[i])
    res = {
        'base_mva_by_year_day': base_yd,
        'instance': {'campaign': c['campaign'], 'eval_dir': ev, 'eval_key': evalrec['eval_key'],
                     'candidate_key': evalrec['candidate_key'], 'candidate_label': evalrec['candidate_label'],
                     'cycles_recorded': K, 'window': [w0, w1], 'gap_clause_refused_at_cap': dec.get('gap_clause_refused_at_cap'),
                     'status_fields': {k: dec.get(k) for k in ('cap', 'k_star', 'branch', 'certified', 'status') if k in dec}},
        'inputs_sha256': inputs,
        'identity_checks': identity,
        't_sum_terminal_eur': t_sum_K, 't_sum_recorded_terminal_eur': float(rs[K]['t_sum']),
        't_sum_window': {'min': min(float(rs[k]['t_sum']) for k in ks), 'max': max(float(rs[k]['t_sum']) for k in ks),
                         'start': float(rs[w0]['t_sum']), 'end': float(rs[K]['t_sum'])},
        'sum_abs_gap_mw_p_window': {'start': float(rs[w0]['sum_abs_gap_mw_p']), 'end': float(rs[K]['sum_abs_gap_mw_p'])},
        'pf_primal_ratio_window': {'start': float(pcr[w0]['boyd_pf_primal_ratio']), 'end': float(pcr[K]['boyd_pf_primal_ratio'])},
        'by_year_day_terminal_eur': dict(sorted(by_yd.items(), key=lambda kv: kv[1] * -sgn)),
        'by_node_year_day': by_block,
        'by_hour_terminal_eur': by_hour,
        'contributing_entries': {'rule': f'smallest prefix (ranked by terminal contribution of t_sum sign) reaching '
                                         f'{CONTRIB_CUM} of t_sum', 'n': len(contrib),
                                 'share_reached': float(sum(t_term[it] for it in contrib) / t_sum_K),
                                 'entries': [ekey(i, t) for i, t in contrib]},
        'top_entries': [entry(i, t, history=True) for i, t in top],
        'all_p_entries_summary': all_entries,
        'stationarity_summary': {'n_entries': len(all_entries), 'n_price_moving_gt_0p01_eur': n_price_moving,
                                 'max_abs_price_change_window_eur_mwh': max(abs(e['lambda_dso_eur_mwh']['change']) for e in all_entries),
                                 'n_entries_abs_gap_gt_1e-3_mw_terminal': n_gap1mw,
                                 'max_abs_spread_dso_tso_eur_mwh': max(e['spread_dso_minus_tso_eur_mwh_max_abs_window'] for e in all_entries)},
        'ipopt_records_window': ip,
    }
    return res, contrib, blocks


# ------------------------------------------------------------------------------------------------ part B (models)
def _verify_pkl(cell):
    c = CELLS[cell]
    rel = os.path.join(c['campaign'], 'evals', c['eval'], 'certified_models.pkl')
    man = json.load(open(os.path.join(REPO, c['campaign'], 'campaign_manifest_sha256.json')))
    post = json.load(open(os.path.join(REPO, c['campaign'], 'evals', c['eval'], 'post_certification.json')))
    now = W127._sha(os.path.join(REPO, rel))
    if man.get(rel) != now or post['persisted_models']['sha256'] != now:
        raise RuntimeError(f'{rel}: sha256 {now} vs manifest {man.get(rel)} vs post_certification {post["persisted_models"]["sha256"]}')
    return rel, {'sha256': now, 'manifest': os.path.join(c['campaign'], 'campaign_manifest_sha256.json'),
                 'post_certification_cycle': post['certification']['certification_cycle'], 'committed': False}


def _tso_compact(t):
    return {'generators': [{k: g[k] for k in ('var', 'value', 'lb', 'ub', 'at_lb', 'at_ub', 'zL', 'zU',
                                              'class_by_objective_gradient', 'cost_gradient_eur_per_pu', 'pg_avail_param')}
                           for g in t['generators_hour4']],
            'per_dn': [{'dn': p['dn'], 'z_tso_copy_pu': p['z_tso_copy_pu'], 'x_dso_req_pu': p['x_dso_req_pu'],
                        'pf_AL_gradient_on_z_eur_per_pu': p['pf_AL_gradient_on_z_eur_per_pu'],
                        'interface_delta_at_bound': bool(p['interface_delta_p']['at_lb'] or p['interface_delta_p']['at_ub'])}
                       for p in t['per_dn']],
            'node_balance_p_duals': t['node_balance_p_duals'],
            'cheapest_feasible_lever': ({k: t['cheapest_feasible_lever'][k] for k in ('var', 'direction', 'first_order_dJ_per_pu')}
                                        if t['cheapest_feasible_lever'] else None),
            'kkt_relative': t['kkt_check']['relative']}


def _dso_compact(d):
    fam = d['var_family_summary_hour4_active']
    lev = d['cheapest_feasible_lever']
    return {'gap_x_minus_z_mw': d['gap_x_minus_z_mw'], 'pf_AL_gradient_on_x_eur_per_pu': d['pf_AL_gradient_on_x_eur_per_pu'],
            'node_balance_p_dual_ref_bus': d['node_balance_p_dual_ref_bus'],
            'flex_p_up_active': fam.get('flex_p_up'), 'flex_p_down_active': fam.get('flex_p_down'),
            'n_flex_p_up': sum(1 for v in d['tables']['vars_hour4'] if v['var'].startswith('flex_p_up[')),
            'n_flex_p_down': sum(1 for v in d['tables']['vars_hour4'] if v['var'].startswith('flex_p_down[')),
            'flex_p_up_headroom_pu': d['flex_p_up_headroom_hour4_pu'],
            'cheapest_feasible_lever': ({k: lev[k] for k in ('var', 'direction', 'first_order_dJ_per_pu', 'value', 'lb', 'ub')}
                                        if lev else None),
            'kkt_relative': d['kkt_check']['relative']}


def models_cell(cell, year_days, base_mva):
    rel, info = _verify_pkl(cell)
    import pyomo.environ as pe
    from pyomo.core.expr.calculus.derivatives import differentiate, Modes
    from pyomo.core.expr.visitor import identify_variables
    tools = (pe, differentiate, Modes, identify_variables)
    with open(os.path.join(REPO, rel), 'rb') as handle:
        payload = pickle.load(handle)
    _log(f'{cell}: certified models unpickled ({info["sha256"][:8]})')
    out = {'pkl': {rel: info}, 'blocks': {}}
    for (y, d) in year_days:
        yk = int(y)
        tso = W127.Block(payload['tso'][yk][d], *tools)
        dsos = {n: W127.Block(payload['dso'][n][yk][d], *tools) for n in NODES if n in payload['dso']}
        base = base_mva[f'{y}|{d}']   # recorded dso_base_mva = tso_base_mva of the dual-capture header (asserted)
        per_p = []
        day_rows = {}
        ess_day = {}
        for p in range(24):
            W127.P = p          # W127's functions read the period from this module constant
            tb = W127.tso_block(tso, base)
            row = {'period': p, 'hour': p + 1, 'tso': _tso_compact(tb), 'dso': {}}
            for n, blk in dsos.items():
                db = W127.dso_block(blk, tso, n, base)
                row['dso'][str(n)] = _dso_compact(db)
                if p == 0:
                    day_rows[str(n)] = [r for r in db['tables'].get('rows_day_level', [])]
                    ess_day[str(n)] = db['shared_ess_day']
            per_p.append(row)
        W127.P = 3
        out['blocks'][f'{y}|{d}'] = {'per_period': per_p, 'dso_day_level_rows': day_rows, 'dso_shared_ess_day': ess_day,
                                     'kkt_relative_tso': tso.kkt_check()['relative'],
                                     'kkt_relative_dso': {str(n): b.kkt_check()['relative'] for n, b in dsos.items()},
                                     'base_mva_used': base}
        _log(f'{cell} {y}|{d}: 24 periods read')
    del payload
    return out


def kinks(models, entries):
    """Per contributing entry: TN conventional all at a bound; DN flex all at lb and cheapest lever = flexibility."""
    rows = []
    for e in entries:
        blk = models['blocks'].get(f"{e['year']}|{e['day']}")
        if blk is None:
            rows.append({**e, 'read': False})
            continue
        r = blk['per_period'][e['period']]
        conv = [g for g in r['tso']['generators'] if g['class_by_objective_gradient'] == 'costed (conventional)']
        tn = bool(conv) and all(g['at_lb'] or g['at_ub'] for g in conv)
        d = r['dso'][str(e['node_id'])]
        up, dn = d['flex_p_up_active'] or {}, d['flex_p_down_active'] or {}
        zero = (up.get('n_at_lb', 0) == d['n_flex_p_up'] and dn.get('n_at_lb', 0) == d['n_flex_p_down'])
        lev = d['cheapest_feasible_lever']
        lever_flex = bool(lev) and lev['var'].startswith(('flex_p_up[', 'flex_p_down['))
        n_int = ((d['n_flex_p_up'] - up.get('n_at_lb', 0) - up.get('n_at_ub', 0))
                 + (d['n_flex_p_down'] - dn.get('n_at_lb', 0) - dn.get('n_at_ub', 0)))
        tlev = r['tso']['cheapest_feasible_lever']
        rows.append({**e, 'read': True, 'tn_conventional_all_at_bound': tn,
                     'report_only_dn_n_flex_interior': int(n_int), 'report_only_dn_flex_interior': bool(n_int > 0),
                     'report_only_tso_cheapest_lever': tlev,
                     'report_only_tso_cheapest_lever_is_shared_ess': bool(tlev) and tlev['var'].startswith('shared_es_'),
                     'tn_conventional': [{k: g[k] for k in ('var', 'value', 'lb', 'ub', 'at_lb', 'at_ub', 'zL', 'zU')} for g in conv],
                     'dn_flex_all_at_lb': bool(zero), 'dn_flex_p_up_n_at_lb': up.get('n_at_lb', 0), 'dn_n_flex_p_up': d['n_flex_p_up'],
                     'dn_flex_p_down_n_at_lb': dn.get('n_at_lb', 0), 'dn_n_flex_p_down': d['n_flex_p_down'],
                     'dn_cheapest_lever': lev, 'dn_cheapest_lever_is_flex': lever_flex,
                     'dn_pf_AL_gradient_on_x_eur_per_pu': d['pf_AL_gradient_on_x_eur_per_pu'],
                     'dn_flex_at_kink': bool(zero and lever_flex), 'gap_x_minus_z_mw_model': d['gap_x_minus_z_mw']})
    return rows


def run():
    t0 = time.time()
    dry = '--dry' in sys.argv
    if os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')
    if not dry and not os.path.isdir(OUT_DIR):
        raise RuntimeError(f'{OUT_DIR} missing; the launcher creates it')
    result = {'stage': 'P5.15 W149 interface price read for the dead-zone table (zero solves)', 'utc': _utc(),
              'git_head': _git('rev-parse', 'HEAD').stdout.strip(), 'script_sha256': W127._sha(THIS),
              'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 63',
              'objective_convention': 't_sum = sum w * pi * (x_dso - z_tso) over pf p entries (EUR, block-weighted; the '
                                      'priced interface-consensus gap of the settling rule). No gross / net cost is '
                                      'reported here.',
              'constants': {'PRICE_STAT_EUR': PRICE_STAT_EUR, 'GAP_STAT_REL': GAP_STAT_REL, 'CONTRIB_CUM': CONTRIB_CUM,
                            'BLOCK_SHARE_MIN': BLOCK_SHARE_MIN, 'TOP_N': TOP_N, 'W127_active_tol': W127.TOL},
              'not_in_committed_records': NOT_IN_COMMITTED_RECORDS, 'records': {}}
    contrib = {}
    for cell in CELLS:
        res, cb, blocks = records_cell(cell)
        result['records'][cell] = res
        contrib[cell] = res['contributing_entries']['entries']
        _log(f"{cell}: t_sum {res['t_sum_terminal_eur']:.1f} (identity t_sum diff {res['identity_checks']['max_abs_diff_t_sum_eur']:.2e}, "
             f"terminal residual diff {res['identity_checks']['terminal_max_abs_diff_vs_settlement_residual_mw']:.2e}); "
             f"{len(cb)} contributing entries; top block {next(iter(res['by_node_year_day']))}")
    # F2 committed artefacts (W126, W127)
    f2 = {}
    for key, (rel, man_rel) in F2_ARTEFACTS.items():
        now = W127._sha(os.path.join(REPO, rel))
        if json.load(open(os.path.join(REPO, man_rel))).get(rel) != now:
            raise RuntimeError(f'{rel}: sha256 {now} != {man_rel}')
        f2[key] = {'path': rel, 'sha256': now, 'manifest': man_rel}
    w127j = json.load(open(os.path.join(REPO, F2_ARTEFACTS['w127'][0])))
    ch = w127j['challenger']
    conv = [g for g in ch['tso']['generators_hour4'] if g['class_by_objective_gradient'] == 'costed (conventional)']
    f2_dn = {}
    for n, d in ch['dso'].items():
        fam = d['var_family_summary_hour4_active']
        n_up = sum(1 for v in d['tables']['vars_hour4'] if v['var'].startswith('flex_p_up['))
        n_dn = sum(1 for v in d['tables']['vars_hour4'] if v['var'].startswith('flex_p_down['))
        lev = d['cheapest_feasible_lever']
        f2_dn[n] = {'flex_all_at_lb': bool(fam.get('flex_p_up', {}).get('n_at_lb') == n_up and fam.get('flex_p_down', {}).get('n_at_lb') == n_dn),
                    'cheapest_lever': lev['var'], 'cheapest_lever_eur_per_pu': lev['first_order_dJ_per_pu'],
                    'pf_AL_gradient_on_x_eur_per_pu': d['pf_AL_gradient_on_x_eur_per_pu']}
    f2['w127_challenger_block'] = w127j['block']
    f2['w127_tn_conventional_hour4'] = [{k: g[k] for k in ('var', 'value', 'lb', 'at_lb', 'zL')} for g in conv]
    f2['w127_tn_all_at_bound'] = bool(conv) and all(g['at_lb'] or g['at_ub'] for g in conv)
    f2['w127_dn'] = f2_dn
    f2['w127_dn_flex_at_kink_all_nodes'] = all(v['flex_all_at_lb'] and v['cheapest_lever'].startswith(('flex_p_up[', 'flex_p_down['))
                                               for v in f2_dn.values())
    f2['w127_tso_cheapest_lever'] = ch['tso']['cheapest_feasible_lever']['var']
    result['f2_committed'] = f2
    # part B: model reads
    sub = result['records'][SUBJECT]
    yd_share = {}
    for k, v in sub['by_year_day_terminal_eur'].items():
        yd_share[k] = v / sub['t_sum_terminal_eur']
    year_days = [tuple(k.split('|')) for k, s in yd_share.items() if s >= BLOCK_SHARE_MIN]
    result['model_blocks'] = {'rule': f'(year, day) with >= {BLOCK_SHARE_MIN} of the subject terminal t_sum',
                              'shares': yd_share, 'selected': ['|'.join(b) for b in year_days]}
    models = {}
    for cell in (SUBJECT, CONTROL):
        models[cell] = models_cell(cell, year_days, sub['base_mva_by_year_day'])
    result['models'] = models
    # kinks on the subject's contributing entries; the same entries on the control
    kk = {SUBJECT: kinks(models[SUBJECT], contrib[SUBJECT]), CONTROL: kinks(models[CONTROL], contrib[SUBJECT])}
    result['kinks_on_subject_contributing_entries'] = kk
    verdict = {}
    for cell, rows in kk.items():
        read = [r for r in rows if r['read']]
        n_tn = sum(1 for r in read if r['tn_conventional_all_at_bound'])
        n_dn = sum(1 for r in read if r['dn_flex_at_kink'])
        n_zero = sum(1 for r in read if r['dn_flex_all_at_lb'])
        nodes = sorted({r['node_id'] for r in rows})
        recs = result['records'][cell]['by_node_year_day']
        stat = {}
        for r in rows:
            key = f"{r['node_id']}|{r['year']}|{r['day']}"
            stat[key] = recs[key]['block_gap_stationary']
        n = len(rows)
        if len(read) == n and n_tn == n and n_dn == n and all(stat.values()):
            v = 'reproduced'
        elif max(n_tn, n_dn) >= 0.5 * n:
            v = 'partially reproduced'
        else:
            v = 'not reproduced'
        verdict[cell] = {'n_entries': n, 'n_read': len(read), 'n_tn_conventional_all_at_bound': n_tn,
                         'n_dn_flex_at_kink': n_dn, 'n_dn_flex_all_at_lb': n_zero,
                         'n_dn_cheapest_lever_is_flex': sum(1 for r in read if r['dn_cheapest_lever_is_flex']),
                         'report_only_n_dn_flex_interior': sum(1 for r in read if r['report_only_dn_flex_interior']),
                         'report_only_n_tso_cheapest_lever_is_shared_ess': sum(1 for r in read if r['report_only_tso_cheapest_lever_is_shared_ess']),
                         'block_gap_stationary': stat, 'nodes': nodes, 'verdict': v}
    result['verdict'] = verdict
    # table in the shape of the manuscript dead-zone table
    def _top(cell):
        e = result['records'][cell]['top_entries'][0]
        return f"n{e['node_id']} {e['year']} {e['day']} h{e['hour']}"

    def _yn(n, tot):
        return 'y' if n == tot else ('n' if n == 0 else f'partial ({n}/{tot})')
    w126j = json.load(open(os.path.join(REPO, F2_ARTEFACTS['w126'][0])))
    table = []
    for cell, m, storage in ((SUBJECT, 1.5, 'n7 0.25 MVA / 1.0 MWh (2025)'), (CONTROL, 1.5, 'none (x = 0)')):
        vd = verdict[cell]
        table.append({'cell': cell, 'm': m, 'storage': storage,
                      't_sum_at_cap_eur': result['records'][cell]['t_sum_terminal_eur'],
                      'terminal_cycle': result['records'][cell]['instance']['cycles_recorded'],
                      'top_interface_period': _top(cell),
                      'tn_at_bound': _yn(vd['n_tn_conventional_all_at_bound'], vd['n_entries']),
                      'dn_flex_marginal_at_kink': _yn(vd['n_dn_flex_at_kink'], vd['n_entries']),
                      'report_only_dn_flex_interior': _yn(vd['report_only_n_dn_flex_interior'], vd['n_entries']),
                      'report_only_tso_cheapest_lever_is_shared_ess': _yn(vd['report_only_n_tso_cheapest_lever_is_shared_ess'], vd['n_entries']),
                      'verdict': vd['verdict'] if cell == SUBJECT else f"control: {vd['verdict']} on the subject's entries",
                      'evidence': 'W149 parts A + B'})
    table.append({'cell': 'f2_challenger', 'm': 2.0, 'storage': 'n5 0.25/1.0, n7 1.0/3.0 (2030)',
                  't_sum_at_cap_eur': result['records']['f2_challenger']['t_sum_terminal_eur'], 'terminal_cycle': 261,
                  'top_interface_period': _top('f2_challenger'),
                  'tn_at_bound': 'y' if f2['w127_tn_all_at_bound'] else 'n',
                  'dn_flex_marginal_at_kink': 'y' if f2['w127_dn_flex_at_kink_all_nodes'] else 'n',
                  'report_only_dn_flex_interior': 'n' if all(v['flex_all_at_lb'] for v in f2_dn.values()) else 'y (some node)',
                  'report_only_tso_cheapest_lever_is_shared_ess': 'y' if f2['w127_tso_cheapest_lever'].startswith('shared_es_') else 'n',
                  'verdict': 'dual dead zone (W127 verdict, Addendum 57 report)',
                  'evidence': f"W126 {f2['w126']['sha256'][:8]} (t_sum at 261 {w126j['task2_localisation']['t_sum']['summary']['t_sum_261']:.1f}); "
                              f"W127 {f2['w127']['sha256'][:8]} (2030 Autumn h4 only)"})
    table.append({'cell': 'f2_incumbent', 'm': 2.0, 'storage': 'n5 0.25/0.5, n7 1.0/3.5 (2030)',
                  't_sum_at_cap_eur': result['records']['f2_incumbent']['t_sum_terminal_eur'], 'terminal_cycle': 281,
                  'top_interface_period': _top('f2_incumbent'),
                  'tn_at_bound': 'not recorded (W127 loaded the challenger only)',
                  'dn_flex_marginal_at_kink': 'not recorded (W127 loaded the challenger only)',
                  'report_only_dn_flex_interior': 'not recorded (W127 loaded the challenger only)',
                  'report_only_tso_cheapest_lever_is_shared_ess': 'not recorded (W127 loaded the challenger only)',
                  'verdict': 'reported with the challenger as the F2 dead zone (Addendum 57 report); no model read',
                  'evidence': 'W149 part A only'})
    result['dead_zone_table'] = table
    g_ours = _GUARD.verify(0)
    g_w127 = W127._GUARD.verify(0)
    result['solve_profile_guard'] = {'permitted': [], 'w149_verify_0': g_ours, 'w149_counts': dict(_GUARD.counts),
                                     'w127_module_verify_0': g_w127, 'w127_module_counts': dict(W127._GUARD.counts)}
    result['wall_s'] = time.time() - t0
    for row in table:
        _log(f"TABLE {row['cell']}: m {row['m']}, t_sum {row['t_sum_at_cap_eur']:.1f}, top {row['top_interface_period']}, "
             f"TN {row['tn_at_bound']}, DN {row['dn_flex_marginal_at_kink']}, {row['verdict']}; report-only: DN flex "
             f"interior {row['report_only_dn_flex_interior']}, TSO lever = shared ESS {row['report_only_tso_cheapest_lever_is_shared_ess']}")
    ok = not g_ours and not g_w127
    if dry:
        GRIO.check(result, default=GRIO.json_default_item)
        _log(f'DRY: nothing written; guards {g_ours} / {g_w127}; wall {result["wall_s"]:.1f} s')
        return 0 if ok else 1
    with open(OUT_JSON, 'x') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    _log(f'wrote {OUT_JSON}; guards verify(0) {g_ours} / {g_w127}; counts {dict(_GUARD.counts)} / '
         f'{dict(W127._GUARD.counts)}; wall {result["wall_s"]:.1f} s')
    return 0 if ok else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = W127._sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = W127._sha(THIS)
    res = json.load(open(OUT_JSON))
    for cell in res['records'].values():
        for rel, v in cell['inputs_sha256'].items():
            if v.get('present'):
                entries[rel + ' (input)'] = v['sha256']
    for cell in res['models'].values():
        for rel, v in cell['pkl'].items():
            entries[rel + ' (hash-recorded input, not committed)'] = v['sha256']
    for key in ('w126', 'w127'):
        v = res['f2_committed'][key]
        entries[v['path'] + ' (input)'] = v['sha256']
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else run())
