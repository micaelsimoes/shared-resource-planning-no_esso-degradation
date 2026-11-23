"""P5.15 S56 -- diagnostic 3: the economic scale of TN-DN interface residuals, from the saved 3 x 3 runs (ZERO SOLVES,
no unpickling, no model built).

Authority: S56_PLAN_v2.md Expert item X5 (signed priced residual, a FIRST-ORDER DIAGNOSTIC, NEVER A BOUND), confirmed in
S56_PLAN_v3.md section 3 ("diagnostic only": not used to replace an objective, rank plans, or appear beside the main
results) and the PR / A terms of S56_DECOMPOSITION_NOTE.md; TASKS.md S56 "Zero-solve diagnostics before the pilot spec"
item (3). Pairing and sign convention: audit D (`p515_s56_interface_consistency.py`,
data/SRP1/Results/P515S56/interface_consistency/).
RESUMED 2026-11-23: never launched into diag_neighbourhood_residual_v1/ (preserved untouched; this script refuses to write
into it). Output now goes to diag_neighbourhood_residual_v2/.
DUAL SOURCES: these three archived runs carry NO `interface_duals_per_cycle.jsonl` (that file exists only in later
single-scenario SRP1 runs, P515S53/w101 onwards); the consensus dual is recovered from `pf_entry_stride_*.jsonl`
(`lambda_dso`, every cycle) and the terminal `coordination_by_dso_block` of response_terminal.json (`dual_pf_p_req`,
the terminal solve's), cross-checked against each other below. The absence is recorded as a check.

RUNS (archived five-block 3 x 3; every input file sha256-checked against the run's own child manifest):
  3x3_pair_x0            W91 pair, x = 0                      (P515S53/w90_3x3/.../f6e9cd53fdbb8ee8_x0)
  3x3_pair_n7_4h_e1      W91 pair, node-7 unit                (P515S53/w90_3x3/.../c82522f470b35b58_n7_4h_e1)
  3x3_x0_continuation_r2 W98 x = 0 continuation r2            (P515S53/w98_continuation/.../25b92ae0f1f2c02e_x0_cont)

SOURCES per run:
  multiscenario_terminal.json  per (DN, block): admm_block_weight w_b; per joint scenario s = '<s_m>_<s_o>' the
                               probability omega_s and the TSO-side and DSO-side interface P per hour (MW, DN-import
                               positive); the committed (expected) profiles Pbar^T, Pbar^D; the DSO block's price array
                               pi[t][s_m] and pibar[t] (`dispersion`).
  response_terminal.json       `coordination_by_dso_block` (p515_s51_2x2_limit_gate.coordination_record, decoded with
                               p515_s44_campaign_harness.decode_response_payload): per DSO block the model's
                               dual_pf_p_req[t] AS SET FOR ITS LAST SOLVE, rho_pf, interface_rating_pu, the block's
                               effective objective scale sigma_eff = admm_objective_scale, the TSO request and pbar, and
                               lambda_al_per_mwh.
  pf_entry_stride_*.jsonl      per ADMM cycle k, per (DN, block, hour, p/q): x_dso = Pbar^D(k) (MW), z_tso_current =
                               Pbar^T(k) (MW), lambda_dso = production dual_vars after the cycle-k update, rating (MVA),
                               s_base, rho_pf.

THE ADMM SCALING (production `update_distribution_models_to_admm`, `_update_interface_power_flow_variables` and the
DSO model update): the DSO block's AL objective is
    f_b / sigma_eff + lambda_m[t] * (Pbar^D_pu[t] - req_pu[t]) / rating_pu + rho/2 * (...)^2,   sigma_eff = sigma / w_b,
with f_b the block's per-representative-day, probability-weighted cost (EUR) and lambda_m = dual_vars / s_base (the
model Param). Undoing it, the consensus dual in EUR per MWh of the block objective is
    lambda_EUR[t] = sigma_eff * lambda_m[t] / (rating_pu * s_base) = sigma_eff * lambda_dualvars[t] / (rating_MVA * s_base)
and w_b * lambda_EUR is its value in the Q convention (Q = sum_b w_b f_b). Checked: sigma_eff * w_b == sigma for every
block; lambda_dualvars(stride, cycle N-1) / s_base == the coordination dual (the terminal solve's); the captured
lambda_al_per_mwh == sigma_eff (lambda_m + rho (pbar_pu - req_pu)/rating_pu)/(rating_pu s_base).

QUANTITIES (per DN n and total over the three DNs; b over the 20 blocks, s over the 9 joint scenarios, t over 24 h;
w_b = admm_block_weight = years x days x 1.02^-(y - 2025); omega_s from the capture):
  H        = sum_b w_b * 24                                      weighted (discounted) horizon hours
  K_pi     = sum_b w_b sum_s omega_s sum_t |pi_s,t|              EUR per MW of a uniform same-sign residual, scenario price
  K_pibar  = sum_b w_b sum_t |pibar_t|                           (informational)
  K_lambda = sum_b w_b sum_t |lambda_EUR,t|                      the consensus dual, terminal solve's (coordination)
  K_lambda_after = the same with the dual after the final update (stride cycle N)          (informational)
  K_lambda_AL    = the same with lambda_al_per_mwh (dual + rho term, the AL marginal price)  (informational)
  dP_s,t   = P_TSO[s,t] - P_DSO[s,t]                             (TN - DN, audit D's pairing; MW)
  PR       = sum_b w_b sum_s omega_s sum_t pi_s,t dP_s,t         signed priced residual (EUR)   [= -(T_TSO + T_DSO)]
  APR      = sum_b w_b sum_s omega_s sum_t pi_s,t |dP_s,t|       absolute form (prices are checked >= 0; if any price
                                                                  were negative APR_absprod = ... |pi dP| is reported)
  ratio    = |PR| / APR
  decomposition (TN side scenario-common, checked): dP = c_t - d_s,t, c_t = Pbar^T - Pbar^D, d = P_DSO - Pbar^D:
           PR = PR_c - PR_d, PR_c = sum_b w_b sum_t pibar_t c_t, PR_d = sum_b w_b sum_s omega_s sum_t pi_s,t d_s,t
  EC_lambda = sum_b w_b sum_t lambda_EUR,t (Pbar^T_t - Pbar^D_t) the expectation-channel-only part, consensus dual;
           its absolute form sum w |lambda_EUR| |c|; and EC_pibar = PR_c (the same channel at the expected price)
EARLIER CYCLES: the per-scenario paired profiles exist only at the terminal point (audit D; confirmed here: no
per-cycle per-scenario interface P file in any of the three run directories, listed). The expectation channel exists per
cycle in pf_entry_stride; at cycles round(N/4), round(N/2), round(3N/4) and N the script reports K_lambda(k),
EC_lambda(k) (dual faced at cycle k = stride(k-1), residual c(k)) and EC_pibar(k), labelled PARTIAL (no PR / APR).

THESE ARE FIRST-ORDER DIAGNOSTICS (a price times a residual), NOT ERROR BOUNDS on any objective difference.

GUARDS: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0; pickle.load /
pickle.loads blocked and counted (0).
Output: into the diagnostic directory (default data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2/), which must
already exist (diagnostic 1 creates it) and may hold only diagnostic 1's outputs and the two scripts' launch logs; refuses
if its own outputs exist or another copy is running: residual_scale.json, manifest_residual_sha256.json.
Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_residual_scale.py \\
        > data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2/launch_residual_stdout.log \\
        2> data/SRP1/Results/P515S56/diag_neighbourhood_residual_v2/launch_residual_stderr.log
Exit: 0 written and every check holds; 3 written with failed checks; 2 refused before computing.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 diagnostic 3 residual scale (never solves)').install()

import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('diagnostic 3: pickle.load is blocked (no unpickling)')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('diagnostic 3: pickle.loads is blocked (no unpickling)')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402  (stdlib-only at import)

SCRIPT = os.path.basename(os.path.abspath(__file__))
R = os.path.join('data', 'SRP1', 'Results')
RUNS = [
    ('3x3_pair_x0', os.path.join(R, 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals', 'f6e9cd53fdbb8ee8_x0')),
    ('3x3_pair_n7_4h_e1', os.path.join(R, 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals',
                                       'c82522f470b35b58_n7_4h_e1')),
    ('3x3_x0_continuation_r2', os.path.join(R, 'P515S53', 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2',
                                            'evals', '25b92ae0f1f2c02e_x0_cont')),
]
AUDIT_D = os.path.join(R, 'P515S56', 'interface_consistency', 'interface_consistency.json')
ADVISOR_K_RANGE_EUR_PER_MW = (6.0e6, 18.0e6)
ADVISOR_HOURS = 1.19e5
DN1_OUTPUTS = {'neighbourhood_sizes.json', 'neighbourhood_sizes_table.csv', 'manifest_neighbourhood_sha256.json',
               'launch_neighbourhood_stdout.log', 'launch_neighbourhood_stderr.log'}
OWN_LOGS = {'launch_residual_stdout.log', 'launch_residual_stderr.log'}
OUT_JSON = 'residual_scale.json'
OUT_MANIFEST = 'manifest_residual_sha256.json'
REL_TOL = 1e-9
CHECKS = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:400]}' if detail is not None and not ok else ''))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(argv, cwd):
    r = subprocess.run(['git', *argv], cwd=cwd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _input_record(path, data_root, pin=None):
    tracked = _git(['ls-files', '--', path], data_root)
    tracked = bool(tracked and tracked.strip())
    clean = _git(['status', '--porcelain', '--', path], data_root) if tracked else None
    rec = {'path': path, 'sha256': _sha(path), 'tracked': tracked,
           'clean': (not clean.strip()) if clean is not None else None}
    if pin is not None:
        rec['pinned_sha256_child_manifest'] = pin
        rec['pinned_ok'] = pin == rec['sha256']
    return rec


def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, found = os.getpid(), []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


# ======================================================================================================================
def analyse_terminal(label, m, coord, s_base):
    """Terminal-point K / PR / APR / EC per DN (formulas in the module docstring)."""
    blocks = m['blocks']
    tso_keys = sorted(k for k in blocks if k.startswith('TSO|'))
    nodes = sorted({k.split('|')[1] for k in blocks if k.startswith('DSO|')}, key=int)
    worst = {'w_tso_vs_dso': 0.0, 'omega_tso_vs_dso': 0.0, 'tn_scenario_vs_committed_mw': 0.0,
             'pibar_vs_sum_omega_pi': 0.0, 'sigma_eff_times_w_vs_sigma_rel': 0.0, 'lambda_al_recomputed_rel': 0.0,
             'coord_pbar_vs_committed_dso_mw': 0.0, 'coord_req_vs_committed_tso_mw': 0.0,
             'price_across_dns_same_block': 0.0, 'decomposition_rel': 0.0}
    min_price = math.inf
    sigmas = set()
    per_dn, hours_by_dn = {}, {}
    prices_by_block = {}
    for node in nodes:
        acc = {k: 0.0 for k in ('H_weighted_hours', 'K_pi', 'K_pibar', 'K_lambda', 'K_lambda_AL', 'PR', 'APR',
                                'APR_absprod', 'PR_c_pibar', 'PR_d', 'EC_lambda', 'EC_lambda_abs', 'sum_w_abs_c_mwh',
                                'sum_w_omega_abs_dP_mwh', 'sum_w_lambda_signed')}
        acc['n_blocks'] = 0
        lam_terminal = {}
        for tkey in tso_keys:
            _, year, day = tkey.split('|')
            dkey = f'DSO|{node}|{year}|{day}'
            tb, db = blocks[tkey], blocks[dkey]
            w = float(db['admm_block_weight'])
            worst['w_tso_vs_dso'] = max(worst['w_tso_vs_dso'], abs(w - float(tb['admm_block_weight'])))
            tprof = tb['interface_profiles']['per_scenario'][node]
            tcom = tb['interface_profiles']['committed'][node]
            dprof = db['interface_profiles']['per_scenario']
            dcom = db['interface_profiles']['committed']
            pi = db['dispersion']['pi_by_market_by_hour']
            pibar = db['dispersion']['pibar_by_hour']
            prev = prices_by_block.setdefault((year, day), pi)
            worst['price_across_dns_same_block'] = max(worst['price_across_dns_same_block'],
                                                       max(abs(float(a) - float(b)) for ra, rb in zip(prev, pi)
                                                           for a, b in zip(ra, rb)))
            c = coord[dkey]
            hours = c['hours']
            rating_pu = float(c['interface_rating_pu'])
            sig_eff = float(c['admm_objective_scale'])
            rho = float(c['rho_pf'])
            sigma = sig_eff * w
            sigmas.add(sigma)
            scen = sorted(dprof.keys())
            if sorted(tprof.keys()) != scen:
                raise RuntimeError(f'{label} {dkey}: scenario key sets differ')
            omegas = {s: float(db['per_scenario_costs'][s]['probability']) for s in scen}
            for s in scen:
                worst['omega_tso_vs_dso'] = max(worst['omega_tso_vs_dso'],
                                                abs(omegas[s] - float(tb['per_scenario_costs'][s]['probability'])))
            acc['n_blocks'] += 1
            lam_terminal[(year, day)] = []
            for t in range(24):
                h = hours[t]
                if int(h['period']) != t:
                    raise RuntimeError(f'{label} {dkey}: coordination hour order')
                lam_m = float(h['dual_pf_p_req'])
                lam_eur = sig_eff * lam_m / (rating_pu * s_base)
                lam_terminal[(year, day)].append(lam_eur)
                pbar_pu = float(h['pbar_mw']) / s_base
                req_pu = float(h['p_pf_req_mw']) / s_base
                lam_al = sig_eff * (lam_m + rho * (pbar_pu - req_pu) / rating_pu) / (rating_pu * s_base)
                worst['lambda_al_recomputed_rel'] = max(worst['lambda_al_recomputed_rel'],
                                                        abs(lam_al - float(h['lambda_al_per_mwh']))
                                                        / max(abs(float(h['lambda_al_per_mwh'])), 1e-6))
                worst['coord_pbar_vs_committed_dso_mw'] = max(worst['coord_pbar_vs_committed_dso_mw'],
                                                              abs(float(h['pbar_mw']) - float(dcom['p_mw'][t])))
                c_t = float(tcom['p_mw'][t]) - float(dcom['p_mw'][t])
                worst['coord_req_vs_committed_tso_mw'] = max(worst['coord_req_vs_committed_tso_mw'],
                                                             abs(float(h['p_pf_req_mw']) - float(tcom['p_mw'][t])))
                acc['H_weighted_hours'] += w
                acc['K_pibar'] += w * abs(float(pibar[t]))
                acc['K_lambda'] += w * abs(lam_eur)
                acc['K_lambda_AL'] += w * abs(float(h['lambda_al_per_mwh']))
                acc['EC_lambda'] += w * lam_eur * c_t
                acc['EC_lambda_abs'] += w * abs(lam_eur) * abs(c_t)
                acc['PR_c_pibar'] += w * float(pibar[t]) * c_t
                acc['sum_w_abs_c_mwh'] += w * abs(c_t)
                acc['sum_w_lambda_signed'] += w * lam_eur
                sum_omega_pi = 0.0
                for s in scen:
                    s_m = int(s.split('_')[0])
                    price = float(pi[t][s_m])
                    min_price = min(min_price, price)
                    om = omegas[s]
                    sum_omega_pi += om * price
                    dP = float(tprof[s]['p'][t]) - float(dprof[s]['p'][t])
                    d_st = float(dprof[s]['p'][t]) - float(dcom['p_mw'][t])
                    worst['tn_scenario_vs_committed_mw'] = max(worst['tn_scenario_vs_committed_mw'],
                                                               abs(float(tprof[s]['p'][t]) - float(tcom['p_mw'][t])))
                    acc['K_pi'] += w * om * abs(price)
                    acc['PR'] += w * om * price * dP
                    acc['APR'] += w * om * price * abs(dP)
                    acc['APR_absprod'] += w * om * abs(price * dP)
                    acc['PR_d'] += w * om * price * d_st
                    acc['sum_w_omega_abs_dP_mwh'] += w * om * abs(dP)
                worst['pibar_vs_sum_omega_pi'] = max(worst['pibar_vs_sum_omega_pi'],
                                                     abs(sum_omega_pi - float(pibar[t])))
        acc['H_weighted_hours'] = acc['H_weighted_hours']  # sum_b w_b * 24 (accumulated per hour)
        acc['ratio_abs_PR_over_APR'] = abs(acc['PR']) / acc['APR'] if acc['APR'] else None
        acc['mean_abs_pi_eur_per_mwh'] = acc['K_pi'] / acc['H_weighted_hours']
        acc['mean_abs_lambda_eur_per_mwh'] = acc['K_lambda'] / acc['H_weighted_hours']
        acc['decomposition_PR_minus_(PR_c - PR_d)'] = acc['PR'] - (acc['PR_c_pibar'] - acc['PR_d'])
        worst['decomposition_rel'] = max(worst['decomposition_rel'],
                                         abs(acc['decomposition_PR_minus_(PR_c - PR_d)']) / max(abs(acc['PR']), 1.0))
        per_dn[node] = acc
        hours_by_dn[node] = lam_terminal
    total = {k: sum(per_dn[n][k] for n in nodes) for k in ('K_pi', 'K_pibar', 'K_lambda', 'K_lambda_AL', 'PR', 'APR',
                                                           'APR_absprod', 'PR_c_pibar', 'PR_d', 'EC_lambda',
                                                           'EC_lambda_abs', 'sum_w_abs_c_mwh',
                                                           'sum_w_omega_abs_dP_mwh')}
    total['ratio_abs_PR_over_APR'] = abs(total['PR']) / total['APR'] if total['APR'] else None
    total['H_weighted_hours_per_interface'] = per_dn[nodes[0]]['H_weighted_hours']
    sig_list = sorted(sigmas)
    worst['sigma_eff_times_w_vs_sigma_rel'] = (sig_list[-1] - sig_list[0]) / sig_list[0]
    return {'per_dn': per_dn, 'total': total, 'worst': worst, 'min_price_eur_per_mwh': min_price,
            'sigma_common': sig_list[0], 'nodes': nodes}, hours_by_dn


def stride_cycles(path, want_cycles):
    """{cycle: entries} for the wanted cycles (and their predecessors), read line by line."""
    need = set(want_cycles) | {k - 1 for k in want_cycles if k > 1}
    out, all_cycles = {}, []
    with open(path) as handle:
        for ln in handle:
            if not ln.strip():
                continue
            d = json.loads(ln)
            all_cycles.append(d['cycle'])
            if d['cycle'] in need:
                if d.get('stride') != 1:
                    raise RuntimeError(f'{path}: cycle {d["cycle"]} stride {d.get("stride")} != 1')
                out[d['cycle']] = d['entries']
    return out, all_cycles


def analyse_cycle(entries_k, entries_prev, m, sigma, s_base):
    """Expectation channel at cycle k: lambda faced = stride(k-1) (production dual_vars after cycle k-1's update),
    c(k) = z_tso(k) - x_dso(k). P entries only."""
    blocks = m['blocks']
    prev = {(e['node_id'], e['year'], e['day'], e['period']): e for e in entries_prev if e['power_type'] == 'p'} \
        if entries_prev is not None else None
    per_dn = {}
    worst_sbase = 0.0
    for e in entries_k:
        if e['power_type'] != 'p':
            continue
        node = str(e['node_id'])
        db = blocks[f'DSO|{node}|{e["year"]}|{e["day"]}']
        w = float(db['admm_block_weight'])
        t = int(e['period'])
        pibar = float(db['dispersion']['pibar_by_hour'][t])
        worst_sbase = max(worst_sbase, abs(float(e['s_base_dso']) - s_base))
        rating = float(e['interface_rating'])
        c_t = float(e['z_tso_current']) - float(e['x_dso'])
        acc = per_dn.setdefault(node, {'K_lambda_faced': 0.0, 'K_lambda_after': 0.0, 'EC_lambda_faced': 0.0,
                                       'EC_lambda_faced_abs': 0.0, 'EC_pibar': 0.0, 'sum_w_abs_c_mwh': 0.0,
                                       'max_abs_c_mw': 0.0, 'n_entries': 0})
        lam_after = (sigma / w) * float(e['lambda_dso']) / (rating * s_base)
        acc['K_lambda_after'] += w * abs(lam_after)
        if prev is not None:
            lam_f = (sigma / w) * float(prev[(e['node_id'], e['year'], e['day'], e['period'])]['lambda_dso']) \
                / (rating * s_base)
            acc['K_lambda_faced'] += w * abs(lam_f)
            acc['EC_lambda_faced'] += w * lam_f * c_t
            acc['EC_lambda_faced_abs'] += w * abs(lam_f) * abs(c_t)
        acc['EC_pibar'] += w * pibar * c_t
        acc['sum_w_abs_c_mwh'] += w * abs(c_t)
        acc['max_abs_c_mw'] = max(acc['max_abs_c_mw'], abs(c_t))
        acc['n_entries'] += 1
    total = {k: sum(v[k] for v in per_dn.values()) for k in ('K_lambda_faced', 'K_lambda_after', 'EC_lambda_faced',
                                                             'EC_lambda_faced_abs', 'EC_pibar', 'sum_w_abs_c_mwh')}
    total['max_abs_c_mw'] = max(v['max_abs_c_mw'] for v in per_dn.values())
    return {'per_dn': per_dn, 'total': total, 'worst_s_base_mismatch': worst_sbase}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=CODE_ROOT)
    ap.add_argument('--out', default=os.path.join('data', 'SRP1', 'Results', 'P515S56', 'diag_neighbourhood_residual_v2'))
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    os.chdir(data_root)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    others = _other_copy_running()
    if others:
        _log(f'REFUSE: another copy of {SCRIPT} is running: {others}')
        return 2
    if os.path.basename(os.path.normpath(out_dir)) == 'diag_neighbourhood_residual_v1':
        _log(f'REFUSE: {out_dir} is the preserved partial directory of the interrupted 2026-11-21 attempt')
        return 2
    if not os.path.isdir(out_dir):
        _log(f'REFUSE: {out_dir} does not exist (diagnostic 1 creates it)')
        return 2
    present = set(os.listdir(out_dir))
    if not present <= (DN1_OUTPUTS | OWN_LOGS):
        _log(f'REFUSE: {out_dir} holds unexpected files {sorted(present - DN1_OUTPUTS - OWN_LOGS)} '
             f'(own outputs {OUT_JSON}/{OUT_MANIFEST} must not exist)')
        return 2

    code_files = {name: {'file': os.path.abspath(mod.__file__), 'sha256': _sha(mod.__file__)}
                  for name, mod in (('gate_result_io', GRIO), ('p515_s44_campaign_harness', H))}
    code_files['this_script'] = {'file': os.path.abspath(__file__), 'sha256': _sha(__file__)}
    code_files['p513_solve_profile_guard'] = {'file': os.path.join(CODE_ROOT, 'p513_solve_profile_guard.py'),
                                              'sha256': _sha(os.path.join(CODE_ROOT, 'p513_solve_profile_guard.py'))}
    _ck('every imported repository module resolved from the code root (no mixed trees)',
        all(os.path.dirname(v['file']) == CODE_ROOT for v in code_files.values()), code_files)
    export_record = None
    if os.path.isfile(os.path.join(CODE_ROOT, 'EXPORT_RECORD.json')):
        with open(os.path.join(CODE_ROOT, 'EXPORT_RECORD.json')) as handle:
            export_record = json.load(handle)

    with open(AUDIT_D) as handle:
        audit_d = json.load(handle)
    s_base_set = {float(f['tn_baseMVA']) for f in audit_d['network_facts'].values()} | {
        float(v['baseMVA']) for f in audit_d['network_facts'].values() for v in f['dn'].values()}
    _ck('baseMVA = 100 on the TN and every DN, every 3 x 3 year (audit D network facts)', s_base_set == {100.0},
        sorted(s_base_set))
    s_base = 100.0
    inputs = {'audit_D': _input_record(AUDIT_D, data_root)}
    results = {}
    for label, edir in RUNS:
        _log(f'== {label} ({edir})')
        man = json.load(open(os.path.join(edir, 'child_manifest_sha256.json')))
        stride_files = sorted(glob.glob(os.path.join(edir, 'pf_entry_stride*.jsonl')))
        if len(stride_files) != 1:
            raise RuntimeError(f'{label}: expected one pf_entry_stride file, found {stride_files}')
        files = {'multiscenario_terminal.json': os.path.join(edir, 'multiscenario_terminal.json'),
                 'response_terminal.json': os.path.join(edir, 'response_terminal.json'),
                 'evaluation_record.json': os.path.join(edir, 'evaluation_record.json'),
                 'pf_entry_stride': stride_files[0]}
        rec_in = {}
        for k, p in files.items():
            rec_in[k] = _input_record(p, data_root, pin=man.get(p))
            _ck(f'{label}: {os.path.basename(p)} sha256 == child manifest pin', rec_in[k]['pinned_ok'], rec_in[k])
        rec_in['child_manifest_sha256.json'] = _input_record(os.path.join(edir, 'child_manifest_sha256.json'),
                                                             data_root)
        inputs[label] = rec_in
        m = json.load(open(files['multiscenario_terminal.json']))
        resp = H.decode_response_payload(json.load(open(files['response_terminal.json'])))
        coord = resp['coordination_by_dso_block']
        ev = json.load(open(files['evaluation_record.json']))
        n_cycles = int(ev['cycles_run'])
        _ck(f'{label}: 60 coordination blocks, none with a capture error',
            len(coord) == 60 and not any('capture_error' in c for c in coord.values()), len(coord))
        term, lam_terminal = analyse_terminal(label, m, coord, s_base)
        w = term['worst']
        _ck(f'{label}: TSO and DSO block weights and scenario probabilities equal',
            w['w_tso_vs_dso'] <= 1e-12 and w['omega_tso_vs_dso'] <= 1e-12, w)
        _ck(f'{label}: TN interface P scenario-common (max |P_TSO[s] - Pbar_TSO| <= 1e-9 MW)',
            w['tn_scenario_vs_committed_mw'] <= 1e-9, w['tn_scenario_vs_committed_mw'])
        _ck(f'{label}: pibar == sum_s omega_s pi_s (<= 1e-9 EUR/MWh)', w['pibar_vs_sum_omega_pi'] <= 1e-9,
            w['pibar_vs_sum_omega_pi'])
        _ck(f'{label}: sigma_eff * w_b == sigma, one value for all 60 DSO blocks (rel <= 1e-12)',
            w['sigma_eff_times_w_vs_sigma_rel'] <= 1e-12, [w['sigma_eff_times_w_vs_sigma_rel'], term['sigma_common']])
        _ck(f'{label}: captured lambda_al_per_mwh == sigma_eff (lambda_m + rho (pbar-req)/rating)/(rating s_base) '
            f'(rel <= 1e-9, floor 1e-6 EUR/MWh)', w['lambda_al_recomputed_rel'] <= 1e-9, w['lambda_al_recomputed_rel'])
        _ck(f'{label}: coordination pbar == committed DSO profile (<= 1e-9 MW)',
            w['coord_pbar_vs_committed_dso_mw'] <= 1e-9, w['coord_pbar_vs_committed_dso_mw'])
        _ck(f'{label}: decomposition PR == PR_c - PR_d (rel <= 1e-9)', w['decomposition_rel'] <= 1e-9,
            w['decomposition_rel'])
        _ck(f'{label}: prices identical across the three DN blocks of each (year, day)',
            w['price_across_dns_same_block'] == 0.0, w['price_across_dns_same_block'])
        # reconciliation with production totals and audit D
        si = m['summary']['settlement_identity']
        pr_vs_prod = _rel(term['total']['PR'], -float(si['residual_interface_settlement_total']))
        _ck(f'{label}: PR == -(T_TSO + T_DSO) of production (rel <= {REL_TOL:g})', pr_vs_prod <= REL_TOL,
            [term['total']['PR'], si['residual_interface_settlement_total']])
        prd_vs_prod = _rel(term['total']['PR_d'], sum(float(v) for v in si['dso_deviation_part'].values()))
        _ck(f'{label}: PR_d == production DSO covariance (rel <= {REL_TOL:g})', prd_vs_prod <= REL_TOL,
            [term['total']['PR_d'], si['dso_deviation_part']])
        ad = audit_d['results'][label]
        ad_ok = all(_rel(term['per_dn'][n]['PR'], -float(ad['energy'][n]['C_net_eur_weighted'])) <= REL_TOL
                    for n in term['nodes'])
        _ck(f'{label}: PR per DN == -audit D C_net_eur_weighted per DN (rel <= {REL_TOL:g})', ad_ok,
            {n: [term['per_dn'][n]['PR'], ad['energy'][n]['C_net_eur_weighted']] for n in term['nodes']})
        _ck(f'{label}: audit D run identity (candidate key, cycles) == this run', ad['run']['candidate_key']
            == ev.get('candidate_key') and ad['run']['cycles_run'] == n_cycles,
            [ad['run']['candidate_key'], ev.get('candidate_key'), ad['run']['cycles_run'], n_cycles])
        if term['min_price_eur_per_mwh'] < 0:
            _log(f'NOTE {label}: negative scenario price present ({term["min_price_eur_per_mwh"]}); APR_absprod differs')

        # per-cycle (expectation channel only)
        want = sorted({max(2, round(n_cycles / 4)), max(2, round(n_cycles / 2)), max(2, round(3 * n_cycles / 4)),
                       n_cycles})
        by_cycle, all_cycles = stride_cycles(files['pf_entry_stride'], want)
        _ck(f'{label}: stride file holds every cycle 1..N exactly once', all_cycles == list(range(1, n_cycles + 1)),
            [all_cycles[:3], all_cycles[-3:], len(all_cycles)])
        sigma = term['sigma_common']
        cyc_out = {}
        for k in want:
            a = analyse_cycle(by_cycle[k], by_cycle.get(k - 1), m, sigma, s_base)
            cyc_out[str(k)] = a
        _ck(f'{label}: stride s_base_dso == 100 at the reported cycles',
            all(v['worst_s_base_mismatch'] == 0.0 for v in cyc_out.values()))
        # terminal cross-checks: stride(N) primal == multiscenario committed; stride(N-1) dual / s_base == coordination
        ent_n = {(str(e['node_id']), e['year'], e['day'], int(e['period'])): e for e in by_cycle[n_cycles]
                 if e['power_type'] == 'p'}
        ent_p = {(str(e['node_id']), e['year'], e['day'], int(e['period'])): e for e in by_cycle[n_cycles - 1]
                 if e['power_type'] == 'p'}
        worst_x = worst_z = worst_l = 0.0
        for dkey, c in coord.items():
            _, node, year, day = dkey.split('|')
            db = m['blocks'][dkey]
            tb = m['blocks'][f'TSO|{year}|{day}']
            for t in range(24):
                e = ent_n[(node, year, day, t)]
                worst_x = max(worst_x, abs(float(e['x_dso']) - float(db['interface_profiles']['committed']['p_mw'][t])))
                worst_z = max(worst_z, abs(float(e['z_tso_current'])
                                           - float(tb['interface_profiles']['committed'][node]['p_mw'][t])))
                lm = float(c['hours'][t]['dual_pf_p_req'])
                worst_l = max(worst_l, abs(float(ent_p[(node, year, day, t)]['lambda_dso']) / s_base - lm)
                              / max(abs(lm), 1e-12))
        _ck(f'{label}: stride cycle N x_dso / z_tso == committed DSO / TSO profiles (<= 1e-9 MW)',
            worst_x <= 1e-9 and worst_z <= 1e-9, [worst_x, worst_z])
        _ck(f'{label}: stride cycle N-1 lambda_dso / s_base == the terminal solve\'s dual_pf_p_req (rel <= 1e-12)',
            worst_l <= 1e-12, worst_l)
        tn = cyc_out[str(n_cycles)]['total']
        _ck(f'{label}: cycle-N stride K_lambda_faced == terminal K_lambda (rel <= 1e-9)',
            _rel(tn['K_lambda_faced'], term['total']['K_lambda']) <= 1e-9, [tn['K_lambda_faced'],
                                                                              term['total']['K_lambda']])
        _ck(f'{label}: cycle-N stride EC_lambda_faced == terminal EC_lambda (abs <= 1e-6 EUR)',
            abs(tn['EC_lambda_faced'] - term['total']['EC_lambda']) <= 1e-6, [tn['EC_lambda_faced'],
                                                                               term['total']['EC_lambda']])
        _ck(f'{label}: cycle-N stride EC_pibar == terminal PR_c (abs <= 1e-6 EUR)',
            abs(tn['EC_pibar'] - term['total']['PR_c_pibar']) <= 1e-6, [tn['EC_pibar'], term['total']['PR_c_pibar']])
        term['total']['K_lambda_after_final_update'] = tn['K_lambda_after']
        phases = None
        ccr = os.path.join(edir, 'continuation_cycle_record.jsonl')
        if os.path.isfile(ccr):   # W98 continuation: cycles 1..72 replay the original run, then continuation
            rec_in['continuation_cycle_record.jsonl'] = _input_record(ccr, data_root, pin=man.get(ccr))
            _ck(f'{label}: continuation_cycle_record.jsonl sha256 == child manifest pin',
                rec_in['continuation_cycle_record.jsonl']['pinned_ok'])
            ph = {json.loads(ln)['cycle']: json.loads(ln)['phase'] for ln in open(ccr) if ln.strip()}
            phases = {str(k): ph.get(k) for k in want}
            for k in want:
                cyc_out[str(k)]['phase'] = ph.get(k)
        # per-cycle paired data: confirm absent
        listing = sorted(os.path.relpath(os.path.join(root, f), edir) for root, _dirs, fs in os.walk(edir) for f in fs)
        paired_candidates = [f for f in listing if any(tok in os.path.basename(f) for tok in ('per_scenario', 'paired'))]
        duals_per_cycle_files = [f for f in listing if 'interface_duals_per_cycle' in os.path.basename(f)]
        per_cycle_record = [json.loads(ln) for ln in open(os.path.join(edir, 'per_cycle_record.jsonl')) if ln.strip()]
        pcr_fields = sorted(per_cycle_record[0].keys())
        per_scen_fields = [f for f in pcr_fields if 'scenario' in f or f.startswith('dP') or f.startswith('delta')]
        _ck(f'{label}: no per-cycle per-scenario paired interface file in the run directory tree (audit D confirmed)',
            not paired_candidates and not per_scen_fields, [paired_candidates, per_scen_fields])
        _ck(f'{label}: no interface_duals_per_cycle file in the run directory tree (duals taken from pf_entry_stride '
            f'+ the terminal coordination record instead)', not duals_per_cycle_files, duals_per_cycle_files)
        results[label] = {
            'run': {'eval_dir': edir, 'candidate_label': ev.get('candidate_label'),
                    'candidate_key': ev.get('candidate_key'), 'eval_key': ev.get('eval_key'),
                    'status': ev.get('status'), 'cycles_run': n_cycles,
                    'certification_cycle': (ev.get('certification') or {}).get('certification_cycle'),
                    'alpha_in_force': m['summary'].get('alpha_in_force'),
                    'gross_operational_cost_terminal': (m['summary'].get('recourse_components') or {}).get(
                        'gross_operational_cost'),
                    'objective_convention': 'gross_operational_cost (the capture summary convention)'},
            'terminal': term,
            'terminal_lambda_eur_per_mwh_by_dn_block': {n: {f'{y}|{d}': v for (y, d), v in blk.items()}
                                                        for n, blk in lam_terminal.items()},
            'earlier_cycles_expectation_channel_only_PARTIAL': {'cycles': want, 'by_cycle': cyc_out,
                                                                'phase_by_cycle': phases,
                                                                'cycle_rule': ('Python round (half to even) of N/4, N/2, 3N/4, '
                                                                               'N (floor 2); lambda faced at cycle k '
                                                                               '= stride(k-1)'),
                                                                'note': ('no per-scenario paired data per cycle: '
                                                                         'PR / APR are terminal-only')},
            'per_cycle_paired_data_search': {'scope': 'recursive listing of the eval directory (os.walk)',
                                             'directory_listing': listing,
                                             'files_matching_paired_tokens': paired_candidates,
                                             'interface_duals_per_cycle_files': duals_per_cycle_files,
                                             'per_cycle_record_fields': pcr_fields,
                                             'tokens': ['per_scenario', 'paired', 'interface_duals_per_cycle '
                                                        '(separate check)',
                                                        'per_cycle_record fields with scenario / dP / delta']},
        }
        t = term['total']
        _log(f'{label}: K_pi = {t["K_pi"]:,.0f} EUR/MW (3 DNs); K_lambda = {t["K_lambda"]:,.1f}; PR = {t["PR"]:,.0f}; '
             f'APR = {t["APR"]:,.0f}; |PR|/APR = {t["ratio_abs_PR_over_APR"]:.4f}; EC_lambda = {t["EC_lambda"]:,.2f}')

    k_pi_per_if = [results[lab]['terminal']['per_dn'][n]['K_pi'] for lab, _e in RUNS
                   for n in results[lab]['terminal']['nodes']]
    hours = [results[lab]['terminal']['total']['H_weighted_hours_per_interface'] for lab, _e in RUNS]
    _ck('weighted horizon hours identical across the three runs', max(hours) - min(hours) <= 1e-9, hours)
    comparison = {
        'advisor_K_range_eur_per_mw': list(ADVISOR_K_RANGE_EUR_PER_MW), 'advisor_hours': ADVISOR_HOURS,
        'H_weighted_hours_exact': hours[0], 'H_over_advisor_hours': hours[0] / ADVISOR_HOURS,
        'K_pi_per_interface_min_max_over_runs_and_DNs': [min(k_pi_per_if), max(k_pi_per_if)],
        'K_pi_per_interface_inside_advisor_range': all(ADVISOR_K_RANGE_EUR_PER_MW[0] <= k <= ADVISOR_K_RANGE_EUR_PER_MW[1]
                                                       for k in k_pi_per_if),
        'K_pi_total_3_interfaces_by_run': {lab: results[lab]['terminal']['total']['K_pi'] for lab, _e in RUNS},
        'K_lambda_total_3_interfaces_by_run': {lab: results[lab]['terminal']['total']['K_lambda'] for lab, _e in RUNS},
        'K_lambda_over_K_pi_by_run': {lab: results[lab]['terminal']['total']['K_lambda']
                                      / results[lab]['terminal']['total']['K_pi'] for lab, _e in RUNS},
    }
    guard_fail = GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', not guard_fail, guard_fail)
    _ck('pickle.load / pickle.loads never reached', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    payload = {
        'schema': 'p515_s56_diag_residual_scale_v1', 'script': SCRIPT, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0,
        'interpreter': sys.executable, 'python': platform.python_version(), 'code_root': CODE_ROOT,
        'data_root': data_root, 'export_record': export_record,
        'git_head_data_root': (_git(['rev-parse', 'HEAD'], data_root) or '').strip(),
        'solve_claim': ('ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any repository '
                        'import, verify(0)); NO UNPICKLING (pickle.load/loads blocked, counted 0)'),
        'reading': 'FIRST-ORDER DIAGNOSTICS (a price times a residual), NOT ERROR BOUNDS on any objective difference',
        'formula': __doc__.split('QUANTITIES')[1].split('THESE ARE FIRST-ORDER')[0].strip(),
        'scaling': __doc__.split('THE ADMM SCALING')[1].split('QUANTITIES')[0].strip(),
        's_base_mva': s_base, 'results': results, 'comparison_with_advisor': comparison,
        'dual_side': ('K_lambda, EC_lambda use the DSO-side expectation-channel dual (production '
                      "dual_vars['pf']['dso'], the multiplier in the DSO block's AL). Production also keeps a TSO-side "
                      "dual (dual_vars['pf']['tso']); it is not captured in these three run directories "
                      '(pf_entry_stride holds lambda_dso only; coordination_by_dso_block holds the DSO model Param; '
                      'no lambda_tso / dual_tso / tso_dual / lambda_pf_tso token in any top-level *.json / *.jsonl '
                      'of the x0 pair directory, searched 2026-11-23)'),
        'interface_duals_per_cycle': ('absent from all three run directory trees (checked per run); the per-cycle '
                                      'duals are those of pf_entry_stride'),
        'inputs': inputs, 'code_files': code_files,
        'checks': CHECKS, 'failed_checks': [c['check'] for c in CHECKS if not c['ok']],
        'all_checks_ok': all(c['ok'] for c in CHECKS),
        'guards': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                   'verify_0': guard_fail, 'pickle_counts': dict(PICKLE_COUNTS)},
    }
    out_path = os.path.join(out_dir, OUT_JSON)
    with open(out_path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1)
    manifest = {'outputs': {OUT_JSON: _sha(out_path)}, 'inputs': inputs, 'code_files': code_files,
                'export_record': export_record,
                'note': 'the launch logs are written while the script runs; their sha256 are recorded in the report'}
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1)
    _log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
