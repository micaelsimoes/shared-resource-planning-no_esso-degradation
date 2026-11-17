"""P5.15 Addendum 27 W18 -- Phase A evidence tables for the review report (ZERO SOLVES).

Reads ONLY committed artifacts; runs no harness, no campaign, no model build. A
`SolveProfileGuard(permitted=())` is installed for the whole computation and verified at exactly
0 solves / 0 process launches afterwards (CLAUDE.md, armed-guard rule).

Every statistic is computed here and its formula is written into the output next to it
(CLAUDE.md: "Preserve the formula, not only the inputs").

INPUTS (sha256 pinned in the output and in the manifest):
  * data/SRP1/Results/P515S45/campaign_s45_{a0_c7,a1a,a1b,a2,a3}/campaign_results.json and, per
    evaluation, evaluation_record.json and per_cycle_record.jsonl (68 evaluations, 64 distinct
    candidate keys; 4 keys evaluated twice, once in A0 and once in A1a);
  * I(x): data/SRP1/Results/P515S45/investment_cost/investment_cost_results.json (W2) and
    data/SRP1/Results/P515S45/investment_cost_a2a3/investment_cost_a2a3_results.json (W16);
  * paper scale: data/SRP1/Results/P515S44/scale_measurement/{paper_cycle_snapoff_r1,
    paper_cycle_snapoff_r2,srp1_cycle_snapoff_r1}/rss_samples_cycle.jsonl and summary.json.

OUTPUTS (new directory data/SRP1/Results/P515S45/phase_a_tables/):
  phase_a_tables.json (machine), phase_a_tables.md (human), launch.log, manifest_sha256.json.

Usage (attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S45/phase_a_tables
    python p515_s45_phase_a_tables.py > data/SRP1/Results/P515S45/phase_a_tables/launch.log 2>&1
    python p515_s45_phase_a_tables.py --manifest
"""
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

STAGE = 'P5.15 Addendum 27 W18 -- Phase A evidence tables for the review report (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 27; Planner task W18',
             'CLAUDE.md evidence rules (formula preservation, difference with its resolution, rule ten, '
             'objective convention on every table)']

_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
_P44S = os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement')
OUT_REL = os.path.join(_P45, 'phase_a_tables')
OUT_DIR = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'phase_a_tables.json'
MD_NAME = 'phase_a_tables.md'
MANIFEST_NAME = 'manifest_sha256.json'

STAGES = ['a0_c7', 'a1a', 'a1b', 'a2', 'a3']
EXPECTED_EVALS = 68
EXPECTED_DISTINCT = 64
IX_W2_REL = os.path.join(_P45, 'investment_cost', 'investment_cost_results.json')
IX_W16_REL = os.path.join(_P45, 'investment_cost_a2a3', 'investment_cost_a2a3_results.json')
SCALE_RUNS = ['paper_cycle_snapoff_r1', 'paper_cycle_snapoff_r2', 'srp1_cycle_snapoff_r1']

T2_THRESHOLD_EUR = 150000.0
T2_TERMINAL_STEP_TEST_EUR = 2000.0
UNFINISHED_STEP_EUR = 3000.0
WINDOW = 10
GIB = 2 ** 30
DURATIONS_H = [2, 4, 6, 8, 10]
EVALUATED_DURATION_RANGE_H = (2.0, 4.0)

OBJECTIVE_CONVENTION = ('Q(x) = certified_cost_gross_settlement_excluded = GROSS operational cost, settlement-'
                        'EXCLUDED (the oracle cost convention); F(x) = I(x) + Q(x) with I(x) from the corrected '
                        'cost file; terminal_salvage_value and net_operational_recourse = gross - salvage are '
                        'REPORTED, EXCLUDED from F; the settlement remainder (T_TSO + sum T_DSO) is REPORTED, '
                        'EXCLUDED from Q and F. Currency EUR.')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha256_file(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _git(args):
    res = subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def _ancestor_pids():
    chain, pid = [], os.getpid()
    while pid and pid not in chain:
        chain.append(pid)
        out = subprocess.run(['ps', '-o', 'ppid=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        pid = int(out) if out.isdigit() else 0
    return chain


def _preflight_no_concurrent_harness():
    """Same rule and exclusion as p515_s45_investment_cost_a2a3._preflight_no_concurrent_harness."""
    res = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    mine = set(_ancestor_pids())
    others = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) in mine:
            continue
        if 'python' in parts[1] and re.search(r'\bp5\d', parts[1]):
            others.append(line.strip()[:300])
    if others:
        raise RuntimeError('refusing to run concurrently with another harness process: ' + ' | '.join(others))
    return {'checked_with': 'ps -axo pid=,command=', 'rule': 'refuse while any other p5* python process is alive',
            'self_and_ancestor_pids_excluded': sorted(mine), 'other_p5_harness_processes': []}


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _Inputs:
    """Every file read is registered here with its sha256 and git state."""

    def __init__(self):
        self.files = {}

    def load(self, rel, jsonl=False):
        path = os.path.join(REPO, rel)
        sha = _sha256_file(path)
        tracked = bool(_git(['ls-files', '--', rel])[1].strip())
        self.files[rel] = {'sha256': sha, 'git_tracked': tracked}
        if jsonl:
            return _read_jsonl(path)
        with open(path) as handle:
            return json.load(handle)

    def check_clean(self):
        rc, out, err = _git(['status', '--porcelain', '--'] + sorted(self.files))
        dirty = [line for line in out.splitlines() if line.strip()]
        untracked = [rel for rel, v in self.files.items() if not v['git_tracked']]
        if rc or dirty or untracked:
            raise RuntimeError(f'inputs not committed/clean: dirty={dirty} untracked={untracked} err={err}')
        for v in self.files.values():
            v['git_clean'] = True


# ======================================================================================
#  T1 / T2
# ======================================================================================
def _active_nodes(canon):
    return sorted((n for n, (p, e) in canon['nodes'].items() if p or e), key=int)


def _node7_alone_2025(canon):
    return canon['investment_year'] == 2025 and _active_nodes(canon) == ['7']


def _gross_series(rows):
    cycles = [r['cycle'] for r in rows]
    if cycles != list(range(1, len(rows) + 1)):
        raise RuntimeError('per-cycle record not contiguous from 1')
    return [r['gross_operational_cost'] for r in rows]


def _window_stats(gross):
    k = len(gross)
    if k < WINDOW + 1:
        raise RuntimeError(f'trajectory of {k} cycles is shorter than the window + 1')
    steps = [gross[i] - gross[i - 1] for i in range(k - WINDOW, k)]
    nonincr = all(s <= 0 for s in steps)
    nondecr = all(s >= 0 for s in steps)
    return {
        'terminal_gross_step_abs': abs(gross[-1] - gross[-2]),
        'terminal_gross_step_signed': gross[-1] - gross[-2],
        'window_descent_signed': gross[-1 - WINDOW] - gross[-1],
        'window_steps_signed': steps,
        'window_monotone': nonincr or nondecr,
        'window_direction': 'non-increasing' if nonincr else ('non-decreasing' if nondecr else 'mixed'),
        'window_bar_max_abs_step': max(abs(s) for s in steps),
    }


def build_evaluations(inp):
    evals = []
    for stage in STAGES:
        rel = os.path.join(_P45, f'campaign_s45_{stage}', 'campaign_results.json')
        camp = inp.load(rel)
        if camp.get('STOP_FOR_REVIEW'):
            raise RuntimeError(f'{rel}: STOP_FOR_REVIEW')
        for label, p in camp['points'].items():
            rec = inp.load(os.path.join(p['eval_dir'], 'evaluation_record.json'))
            rows = inp.load(p['per_cycle_trajectory']['path'], jsonl=True)
            if _sha256_file(os.path.join(REPO, p['per_cycle_trajectory']['path'])) != p['per_cycle_trajectory']['sha256']:
                raise RuntimeError(f'{label}: per-cycle sha256 differs from the campaign record')
            if p['status'] != 'certified' or rec['status'] != 'certified':
                raise RuntimeError(f'{stage}:{label} not certified')
            if rec['candidate_key'] != p['candidate_key']:
                raise RuntimeError(f'{label}: candidate key mismatch record vs campaign')
            gross = _gross_series(rows)
            q = p['certified_cost_gross_settlement_excluded']
            if gross[-1] != q or rec['certified_cost'] != q or len(rows) != p['cycles_run']:
                raise RuntimeError(f'{label}: terminal gross / cycles disagree with the record')
            w = _window_stats(gross)
            if abs(w['window_bar_max_abs_step'] - p['bar']['value']) > 1e-6:
                raise RuntimeError(f'{label}: recomputed bar {w["window_bar_max_abs_step"]} != {p["bar"]["value"]}')
            if abs(w['terminal_gross_step_abs'] - p['rule_ten']['terminal_gross_step_abs']) > 1e-6:
                raise RuntimeError(f'{label}: terminal gross step disagrees with rule_ten')
            last = rows[-1]
            sr = rec['settlement_remainder']
            if sr['value'] != rec['recourse_components']['interface_settlement_total']:
                raise RuntimeError(f'{label}: settlement remainder field mismatch')
            storage = {}
            floor_active = []
            for n in _active_nodes(p['candidate_canonical']):
                s = rec['storage_per_node'][n]
                if not s['has_storage']:
                    raise RuntimeError(f'{label}: node {n} active in candidate but has_storage False')
                storage[n] = {'s_mva': s['s_mva'], 'e_mwh': s['e_mwh'],
                              'efc_per_day_max': s['efc_per_day_max'],
                              'terminal_soh_min_over_active_cohort_years': s['terminal_soh_min_over_active_cohort_years'],
                              'soh_floor_rows_active_at_terminal': s['soh_floor_rows_active_at_terminal']}
                if s['soh_floor_rows_active_at_terminal']:
                    floor_active.append(n)
            for n, s in rec['storage_per_node'].items():   # zero nodes must carry no floor rows either
                if n not in storage and s['soh_floor_rows_active_at_terminal']:
                    floor_active.append(n)
            evals.append({
                'stage': stage, 'label': label, 'identity': f'{stage}:{label}', 'eval_dir': p['eval_dir'],
                'candidate_key': p['candidate_key'], 'candidate_canonical': p['candidate_canonical'],
                'investment_year': p['candidate_canonical']['investment_year'], 'active_nodes': _active_nodes(p['candidate_canonical']),
                'cycles_run': p['cycles_run'], 'certification_cycle': p['certification_cycle'],
                'Q_gross_eur': q, 'I_x_eur_campaign': p['I_x_eur'], 'F_eur_campaign': p['F_eur'],
                'bar_eur': p['bar']['value'], 'bar_definition': p['bar']['definition'],
                **w,
                'terminal_step_over_threshold_production_net': p['rule_ten']['terminal_step_over_threshold_production'],
                'terminal_gross_step_over_threshold': p['rule_ten']['terminal_gross_step_over_threshold'],
                'terminal_objective_tolerance': p['rule_ten']['terminal_objective_tolerance'],
                'settlement_remainder_eur': sr['value'],
                'settlement_remainder_field': ('evaluation_record.json settlement_remainder.value '
                                               '(= recourse_components.interface_settlement_total)'),
                'terminal_salvage_value_eur': rec['recourse_components']['terminal_salvage_value'],
                'net_operational_recourse_eur': rec['recourse_components']['net_operational_recourse'],
                'per_cycle_terminal_salvage_value_eur': last['terminal_salvage_value'],
                'storage_per_active_node': storage,
                'soh_floor_row_active_at_terminal': bool(floor_active),
                'soh_floor_active_nodes': floor_active,
                'wall_run_admm_arm_s': p['wall_time']['record']['run_admm_arm_s'],
            })
    return evals


def attach_ix(evals, ix_w2, ix_w16):
    by_key = {}
    for src, table in (('W2 investment_cost_results.json', ix_w2), ('W16 investment_cost_a2a3_results.json', ix_w16)):
        for name, c in table['candidates'].items():
            by_key.setdefault(c['candidate_key'], []).append((src, name, c['I_new_eur'], c.get('I_old_eur')))
    for e in evals:
        hits = by_key.get(e['candidate_key'])
        if not hits:
            raise RuntimeError(f'{e["identity"]}: candidate key not in either I(x) table')
        vals = {h[2] for h in hits}
        if len(vals) != 1:
            raise RuntimeError(f'{e["identity"]}: I(x) tables disagree {hits}')
        e['I_x_eur'] = hits[0][2]
        e['I_x_source'] = [f'{h[0]} candidates[{h[1]!r}]' for h in hits]
        olds = [h[3] for h in hits if h[3] is not None]
        e['I_x_old_file_eur'] = olds[0] if olds else None
        if abs(e['I_x_eur'] - e['I_x_eur_campaign']) > 1e-6:
            raise RuntimeError(f'{e["identity"]}: I(x) table {e["I_x_eur"]} != campaign {e["I_x_eur_campaign"]}')
        e['F_eur'] = e['I_x_eur'] + e['Q_gross_eur']
        if abs(e['F_eur'] - e['F_eur_campaign']) > 1e-6:
            raise RuntimeError(f'{e["identity"]}: F recomputed != campaign F')


def finish_t1(evals):
    keys = {}
    for e in evals:
        keys.setdefault(e['candidate_key'], []).append(e['identity'])
    x0 = [e for e in evals if not e['active_nodes']]
    if len(x0) != 1:
        raise RuntimeError(f'expected exactly one x = 0 evaluation, found {len(x0)}')
    x0 = x0[0]
    for e in evals:
        others = [i for i in keys[e['candidate_key']] if i != e['identity']]
        e['duplicate_of'] = others
        e['is_duplicate_key'] = bool(others)
        e['F_minus_F0_eur'] = e['F_eur'] - x0['F_eur']
        e['F_minus_F0_resolution_bar_eur'] = e['bar_eur'] + x0['bar_eur'] if e is not x0 else 0.0
        e['F_minus_F0_resolution_terminal_eur'] = (e['terminal_gross_step_abs'] + x0['terminal_gross_step_abs']
                                                   if e is not x0 else 0.0)
        e['value_eur'] = x0['Q_gross_eur'] - e['Q_gross_eur']
        e['unfinished_settling_flag'] = (e['terminal_gross_step_abs'] >= UNFINISHED_STEP_EUR and e['window_monotone'])
        e['T2_terminal_step_lt_2000'] = e['terminal_gross_step_abs'] < T2_TERMINAL_STEP_TEST_EUR
    return x0, keys


# ======================================================================================
#  T3 break-even
# ======================================================================================
def _ols(rows):
    import numpy as np
    X = np.array([[1.0, r['E'], r['P']] for r in rows])
    y = np.array([r['value'] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    rss = float(resid @ resid)
    sigma2 = rss / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return beta, cov, resid, rss, sigma2, n, k


def build_t3(evals, x0, ix_w2):
    unit = ix_w2['expected_unit_costs_per_case_year']
    e_new = unit['new']['2025']['energy_eur_per_mwh_discounted']
    p_new = unit['new']['2025']['power_eur_per_mva_discounted']
    e_old = unit['old']['2025']['energy_eur_per_mwh_discounted']
    p_old = unit['old']['2025']['power_eur_per_mva_discounted']
    pts, seen = [], {}
    for e in evals:
        if not _node7_alone_2025(e['candidate_canonical']):
            continue
        if e['candidate_key'] in seen:
            seen[e['candidate_key']]['duplicate_evaluations'].append(e['identity'])
            if e['Q_gross_eur'] != seen[e['candidate_key']]['Q_gross_eur']:
                raise RuntimeError('duplicate evaluations of one key differ in Q; the fit rule assumes they do not')
            continue
        P, E = e['candidate_canonical']['nodes']['7']
        row = {'identity': e['identity'], 'candidate_key': e['candidate_key'], 'P': P, 'E': E,
               'duration_h': E / P, 'Q_gross_eur': e['Q_gross_eur'], 'value': e['value_eur'],
               'bar_eur': e['bar_eur'], 'terminal_gross_step_abs': e['terminal_gross_step_abs'],
               'I_x_eur': e['I_x_eur'], 'I_x_old_file_eur': e['I_x_old_file_eur'],
               'duplicate_evaluations': [e['identity']]}
        seen[e['candidate_key']] = row
        pts.append(row)
    beta, cov, resid, rss, sigma2, n, k = _ols(pts)
    a, b, c = (float(v) for v in beta)
    se = [math.sqrt(float(cov[i, i])) for i in range(3)]
    for r, res in zip(pts, resid):
        r['fitted_value'] = r['value'] - float(res)
        r['residual'] = float(res)
    rms = math.sqrt(rss / n)
    formulas = {
        'value': 'value(x) = Q(0) - Q(x), Q = gross settlement-excluded cost, Q(0) = Q of the x = 0 evaluation '
                 f'({x0["identity"]}); positive = storage lowers operating cost',
        'fit': 'OLS (numpy.linalg.lstsq, rcond=None) of value on X = [1, E, P], one row per DISTINCT candidate key '
               '(the A0/A1a duplicate keys have bit-identical Q and enter once); x = 0 is NOT a row',
        'standard_errors': 'sqrt(diag(s^2 (X^T X)^-1)), s^2 = RSS / (n - 3)',
        'residual_rms': 'sqrt(RSS / n)',
        'residual_max': 'max |value_i - fitted_i|',
        'unit_costs': 'read from W2 investment_cost_results.json expected_unit_costs_per_case_year.{new,old}.2025.'
                      '{energy_eur_per_mwh_discounted, power_eur_per_mva_discounted} (discount multiplier 1.0 at 2025)',
        'i_breakeven_energy_cost_smallest_4h': 'e* = (V_meas(0.25/1.0) - p_cost * 0.25) / 1.0',
        'ii_breakeven_energy_cost_marginal_4h_MWh': 'e* = b + c/4 - p_cost/4  (one extra MWh at 4 h brings 0.25 MVA; '
                                                   'value b + c/4, cost e + p/4); se = sqrt(var b + var c/16 + cov(b,c)/2)',
        'iii_value_multiplier': 'm = I(x) / V_meas(x): the factor by which the measured value would have to grow for '
                                'storage to pay (F(x) = F(0) when m V = I)',
        'iv_old_file': 'm_old = I_old(x) / V_meas(x), I_old from the W2 table I_old_eur field (checked against '
                       'e_old E + p_old P); the break-even energy costs (i), (ii) do not depend on the energy unit '
                       'cost and are compared with e_old',
        'v_ratio_duration': 'r(h) = (b + c/h) / (e_cost + p_cost/h) per MWh at duration h = E/P',
    }
    # (i)
    smallest = [r for r in pts if r['P'] == 0.25 and r['E'] == 1.0]
    largest = [r for r in pts if r['P'] == 1.25 and r['E'] == 5.0]
    if len(smallest) != 1 or len(largest) != 1:
        raise RuntimeError('0.25/1.0 or 1.25/5.0 node-7 2025 point missing')
    s0, l0 = smallest[0], largest[0]
    i_e = (s0['value'] - p_new * 0.25) / 1.0
    ii_e = b + c / 4 - p_new / 4
    ii_se = math.sqrt(float(cov[1, 1]) + float(cov[2, 2]) / 16 + 2 * float(cov[1, 2]) / 4)

    def _mult(r, I_new, I_old):
        return {'identity': r['identity'], 'P': r['P'], 'E': r['E'], 'V_measured_eur': r['value'],
                'V_resolution_bar_eur': r['bar_eur'] + x0['bar_eur'],
                'V_fitted_eur': r['fitted_value'],
                'I_new_eur': I_new, 'm_new_measured': I_new / r['value'], 'm_new_fitted': I_new / r['fitted_value'],
                'I_old_eur': I_old, 'm_old_measured': I_old / r['value'], 'm_old_fitted': I_old / r['fitted_value']}
    for r in (s0, l0):
        if abs(r['I_x_eur'] - (e_new * r['E'] + p_new * r['P'])) > 1e-6:
            raise RuntimeError('I(x) not linear in the table unit costs')
        if r['I_x_old_file_eur'] is None or abs(r['I_x_old_file_eur'] - (e_old * r['E'] + p_old * r['P'])) > 1e-6:
            raise RuntimeError('I_old(x) not consistent with the old unit costs')
    ratio = []
    for h in DURATIONS_H:
        val = b + c / h
        ratio.append({'h': h, 'value_per_MWh_eur': val,
                      'cost_per_MWh_new_eur': e_new + p_new / h, 'ratio_new': val / (e_new + p_new / h),
                      'cost_per_MWh_old_eur': e_old + p_old / h, 'ratio_old': val / (e_old + p_old / h),
                      'extrapolation': not (EVALUATED_DURATION_RANGE_H[0] <= h <= EVALUATED_DURATION_RANGE_H[1])})
    return {
        'objective_convention': OBJECTIVE_CONVENTION,
        'formulas': formulas,
        'n': n, 'points': pts,
        'coefficients': {'a_eur': a, 'b_eur_per_MWh': b, 'c_eur_per_MVA': c},
        'standard_errors': {'a_eur': se[0], 'b_eur_per_MWh': se[1], 'c_eur_per_MVA': se[2]},
        'cov_b_c': float(cov[1, 2]),
        'residual_rms_eur': rms, 'residual_max_abs_eur': max(abs(r['residual']) for r in pts),
        'dof': n - k, 's_eur': math.sqrt(sigma2),
        'unit_costs_2025': {'energy_new_eur_per_MWh': e_new, 'power_new_eur_per_MVA': p_new,
                            'energy_old_eur_per_MWh': e_old, 'power_old_eur_per_MVA': p_old},
        'i_breakeven_energy_cost_smallest_4h_eur_per_MWh': {
            'identity': s0['identity'], 'duplicate_evaluations': s0['duplicate_evaluations'],
            'V_measured_eur': s0['value'], 'V_resolution_bar_eur': s0['bar_eur'] + x0['bar_eur'],
            'e_star': i_e, 'e_new': e_new, 'e_old': e_old,
            'e_star_over_e_new': i_e / e_new, 'e_star_over_e_old': i_e / e_old},
        'ii_breakeven_energy_cost_marginal_4h_MWh_eur_per_MWh': {
            'e_star': ii_e, 'se': ii_se, 'marginal_value_per_4h_MWh_eur': b + c / 4,
            'e_star_over_e_new': ii_e / e_new, 'e_star_over_e_old': ii_e / e_old},
        'iii_iv_value_multiplier': [_mult(s0, s0['I_x_eur'], s0['I_x_old_file_eur']),
                                    _mult(l0, l0['I_x_eur'], l0['I_x_old_file_eur'])],
        'v_ratio_by_duration': ratio,
        'v_label': ('EXTRAPOLATION beyond the evaluated 2-4 h range for h > 4 (the fit has no point with E/P > 4); '
                    'linear-in-(E, P) marginal value assumed'),
    }


# ======================================================================================
#  T4 additivity
# ======================================================================================
def build_t4(evals, x0, t3):
    by_key_first = {}
    for e in evals:
        by_key_first.setdefault(e['candidate_key'], e)

    def single(node, P, E):
        hits = [e for e in by_key_first.values() if e['investment_year'] == 2025 and e['active_nodes'] == [node]
                and e['candidate_canonical']['nodes'][node] == [P, E]]
        if len(hits) != 1:
            raise RuntimeError(f'single-node point n{node} {P}/{E}: {len(hits)} distinct matches')
        return hits[0]
    rows = []
    for e in by_key_first.values():
        if len(e['active_nodes']) < 2:
            continue
        parts = [single(n, *e['candidate_canonical']['nodes'][n]) for n in e['active_nodes']]
        s = sum(p['value_eur'] for p in parts)
        kk = len(parts)
        res_bar = e['bar_eur'] + sum(p['bar_eur'] for p in parts) + abs(kk - 1) * x0['bar_eur']
        res_term = (e['terminal_gross_step_abs'] + sum(p['terminal_gross_step_abs'] for p in parts)
                    + abs(kk - 1) * x0['terminal_gross_step_abs'])
        diff = e['value_eur'] - s
        rows.append({'identity': e['identity'], 'candidate_canonical': e['candidate_canonical'],
                     'V_combined_eur': e['value_eur'], 'singles': [{'identity': p['identity'],
                                                                     'V_eur': p['value_eur']} for p in parts],
                     'sum_single_V_eur': s, 'ratio': e['value_eur'] / s, 'difference_eur': diff,
                     'resolution_bar_eur': res_bar, 'resolution_terminal_eur': res_term,
                     'abs_diff_over_resolution_bar': abs(diff) / res_bar,
                     'abs_diff_over_fit_residual_rms': abs(diff) / t3['residual_rms_eur']})
    return {'objective_convention': OBJECTIVE_CONVENTION,
            'formulas': {'V': 'V(x) = Q(0) - Q(x) (gross, settlement-excluded)',
                         'ratio': 'V(combined) / sum_n V(single node n at the same per-node setting, 2025)',
                         'difference': 'V(combined) - sum_n V(single_n)',
                         'resolution_bar': ('bar(combined) + sum_n bar(single_n) + |k - 1| bar(x0), k = number of '
                                            'nodes (Q(0) enters the difference k - 1 times); bar = max |gross step| '
                                            'over the last 10 cycles'),
                         'resolution_terminal': 'the same with the terminal gross step in place of the bar',
                         'fit_residual_rms': 'T3 node-7 2025 fit residual rms (a scale of lattice-surface noise)'},
            'fit_residual_rms_eur': t3['residual_rms_eur'], 'rows': rows}


# ======================================================================================
#  T5 paper-scale memory and timing
# ======================================================================================
def _label_runs(samples):
    runs = []
    for r in samples:
        if not runs or runs[-1]['stage'] != r['stage']:
            runs.append({'stage': r['stage'], 'first': r, 'last': r, 'n_samples': 1})
        else:
            runs[-1]['last'] = r
            runs[-1]['n_samples'] += 1
    return runs


def _solve_runs(runs, prefix_filter=None):
    out = []
    for i, r in enumerate(runs):
        if ':: solve ' not in r['stage']:
            continue
        if prefix_filter and not r['stage'].startswith(prefix_filter):
            continue
        name = r['stage'].split(':: solve ')[1]
        out.append({'index': i, 'stage': r['stage'], 'network': name.split()[0],
                    'agent': 'TSO' if 'TSO' in r['stage'].split('::')[0] else 'DSO',
                    'interval_s': r['last']['t'] - r['first']['t'], 'n_samples': r['n_samples'],
                    't_first': r['first']['t'], 't_last': r['last']['t'],
                    'footprint_self_after_bytes': r['last']['footprint_self'],
                    'completed': i < len(runs) - 1})
    return out


def _incr_stats(vals):
    return {'n': len(vals), 'mean_gib': statistics.mean(vals), 'median_gib': statistics.median(vals),
            'first10_mean_gib': statistics.mean(vals[:10]), 'last10_mean_gib': statistics.mean(vals[-10:]),
            'min_gib': min(vals), 'max_gib': max(vals)}


def build_t5(inp, evals):
    out = {'definitions': {
        'block_solve_wall_time': ('stage-label interval: t(last sample carrying the label) - t(first sample carrying '
                                  'it), from rss_samples_cycle.jsonl (sampler period 0.5 s, so each interval '
                                  'UNDER-states the true solve time by up to one period); labels '
                                  '"init: DSO|TSO build + solve :: solve <network> <year> <season>"; the label running '
                                  'at a watchdog abort is not a completed solve and is excluded from times'),
        'footprint_after_solve': 'footprint_self of the LAST sample carrying the solve label',
        'per_solve_increment_all_consecutive': ('footprint_after(solve j) - footprint_after(solve j-1) over consecutive '
                                                'solve labels in sample order, INCLUDING the label running at the '
                                                'abort and the steps that span an unlabelled network-build block'),
        'per_solve_increment_within_network': 'the same, restricted to consecutive solves of the SAME network '
                                              '(no build in between) and to completed labels',
        'units': 'GiB = 2^30 bytes'}}
    samples = {}
    for run in SCALE_RUNS:
        samples[run] = inp.load(os.path.join(_P44S, run, 'rss_samples_cycle.jsonl'), jsonl=True)
        summ = inp.load(os.path.join(_P44S, run, 'summary.json'))
        out.setdefault('runs', {})[run] = {'cycle_status': summ.get('cycle_status'),
                                           'cycle_exit_code': summ.get('cycle_child', {}).get('exit_code'),
                                           'stage_at_abort': (summ.get('cycle_abort') or {}).get('stage_at_abort'),
                                           'abort_cause': (summ.get('cycle_abort') or {}).get('cause'),
                                           'n_samples': len(samples[run])}
    # paper wall times and footprints
    dso_t, tso_t = [], []
    for run in ('paper_cycle_snapoff_r1', 'paper_cycle_snapoff_r2'):
        runs = _label_runs(samples[run])
        sol = _solve_runs(runs)
        R = out['runs'][run]
        R['n_solve_labels'] = len(sol)
        R['n_completed_solve_labels'] = sum(s['completed'] for s in sol)
        R['solves'] = [{k: s[k] for k in ('stage', 'agent', 'interval_s', 'n_samples', 'completed')}
                       | {'footprint_self_after_gib': s['footprint_self_after_bytes'] / GIB} for s in sol]
        for agent, bucket in (('DSO', dso_t), ('TSO', tso_t)):
            ts = [s['interval_s'] for s in sol if s['agent'] == agent and s['completed']]
            bucket.extend(ts)
            R[f'{agent}_interval_s'] = (None if not ts else {'n': len(ts), 'mean': statistics.mean(ts),
                                                             'median': statistics.median(ts), 'min': min(ts),
                                                             'max': max(ts)})
        fp = [s['footprint_self_after_bytes'] / GIB for s in sol]
        allc = [b - a for a, b in zip(fp, fp[1:])]
        within = [(sol[j]['footprint_self_after_bytes'] - sol[j - 1]['footprint_self_after_bytes']) / GIB
                  for j in range(1, len(sol)) if sol[j]['network'] == sol[j - 1]['network'] and sol[j]['completed']]
        R['increment_all_consecutive'] = _incr_stats(allc)
        R['increment_within_network_completed'] = _incr_stats(within)
        builds = []
        for i, r in enumerate(runs):
            if r['stage'].endswith('build + solve') and i > 0:
                builds.append({'stage': r['stage'], 'interval_s': r['last']['t'] - r['first']['t'],
                               'footprint_increment_gib': (r['last']['footprint_self'] - runs[i - 1]['last']['footprint_self']) / GIB,
                               'completed': i < len(runs) - 1})
        R['unlabelled_build_blocks'] = builds
        # variants for reproducing the Planner's scratch figure
        if run == 'paper_cycle_snapoff_r2':
            first_build_end = next(r for r in runs if r['stage'] == 'init: DSO build + solve')['last']['footprint_self']
            R['planner_reproduction_variants'] = {
                'A_mean_all_consecutive_74_steps_gib': statistics.mean(allc),
                'A_last10_mean_gib': statistics.mean(allc[-10:]),
                'B_(fp_after_last_label - fp_end_first_DSO_build_block)/75_gib': (fp[-1] - first_build_end / GIB) / 75,
                'C_(fp_after_last_label - fp_after_first_solve)/74_gib': (fp[-1] - fp[0]) / 74,
                'D_mean_completed_only_73_steps_gib': statistics.mean(allc[:-1]),
                'D_last10_mean_completed_only_gib': statistics.mean(allc[-11:-1]),
            }
    out['paper_block_interval_s'] = {
        'DSO': {'n': len(dso_t), 'mean': statistics.mean(dso_t), 'median': statistics.median(dso_t),
                'source': 'r1 + r2 completed init DSO solve labels'},
        'TSO': {'n': len(tso_t), 'mean': statistics.mean(tso_t), 'median': statistics.median(tso_t),
                'source': 'r2 completed init TSO solve labels (r1 aborted before the TSO)'}}
    # SRP1
    runs = _label_runs(samples['srp1_cycle_snapoff_r1'])
    init_sol = _solve_runs(runs, 'init:')
    cyc_sol = _solve_runs(runs, 'cycle:')

    def _pass(sol, name):
        fp = [s['footprint_self_after_bytes'] / GIB for s in sol]
        inc = [b - a for a, b in zip(fp, fp[1:])]
        within = [(sol[j]['footprint_self_after_bytes'] - sol[j - 1]['footprint_self_after_bytes']) / GIB
                  for j in range(1, len(sol)) if sol[j]['network'] == sol[j - 1]['network']]
        return {'pass': name, 'n_solve_labels_observed': len(sol),
                'footprint_after_first_label_gib': fp[0], 'footprint_after_last_label_gib': fp[-1],
                'total_increment_first_to_last_gib': fp[-1] - fp[0],
                'increment_per_consecutive_step_gib': _incr_stats(inc),
                'increment_within_network_gib': _incr_stats(within) if within else None,
                'within_network_sum_gib': sum(within),
                'n_within_network_steps': len(within),
                'note': ('SRP1 solves are shorter than the 0.5 s sampler period, so labels are aliased: only '
                         f'{len(sol)} of the 48 network solves of the pass carry a sample; the first-to-last total '
                         'is exact for the span it covers, the per-label counts are not per-solve counts')}
    esso = [r for r in runs if r['stage'].startswith('init: ESSO')]
    out['srp1'] = {'init_pass': _pass(init_sol, 'initialization'), 'cycle1_pass': _pass(cyc_sol, 'cycle 1'),
                   'init_includes_network_build_blocks': [r['stage'] for r in runs[1:]
                                                          if r['stage'].endswith('build + solve')
                                                          and r['stage'].startswith('init:')],
                   'init_esso_label_footprint_gib': esso[0]['last']['footprint_self'] / GIB if esso else None,
                   'declared_solves_per_round': 51}
    # projection
    distinct = {}
    for e in evals:
        distinct.setdefault(e['candidate_key'], e)
    cyc = [e['cycles_run'] for e in distinct.values()]
    k_med = statistics.median(cyc)
    per_cycle_srp1 = [e['wall_run_admm_arm_s'] / e['cycles_run'] for e in distinct.values()]
    r2_builds = out['runs']['paper_cycle_snapoff_r2']['unlabelled_build_blocks']
    if not all(b['completed'] for b in r2_builds):
        raise RuntimeError('an r2 build block did not complete')
    build_s = sum(b['interval_s'] for b in r2_builds)
    t_cycle = 60 * out['paper_block_interval_s']['DSO']['mean'] + 20 * out['paper_block_interval_s']['TSO']['mean']
    out['projection'] = {
        'formula_per_cycle': ('T_cycle = n_DSO_blocks * mean(DSO interval) + n_TSO_blocks * mean(TSO interval), '
                              'n_DSO_blocks = 60, n_TSO_blocks = 20 (paper build block_counts); ESSO solves and ADMM '
                              'bookkeeping NOT included (not measured at paper scale) -> a LOWER bound under the '
                              'assumption that cycle solves cost what cold init solves cost'),
        'formula_per_evaluation': ('T_eval = T_build + (K + 1) * T_cycle; +1 = the initialization round; T_build = sum '
                                   'of the unlabelled "init: <agent> build + solve" block intervals of r2 (3 DSO + 1 '
                                   'TSO network builds)'),
        'K_assumed': k_med,
        'K_source': (f'median cycles_run over the {len(cyc)} distinct Phase A candidates at SRP1 (min {min(cyc)}, '
                     f'max {max(cyc)}); ASSUMES paper scale needs the same cycle count -- unverified'),
        'T_cycle_s': t_cycle, 'T_build_s': build_s,
        'T_eval_s': build_s + (k_med + 1) * t_cycle, 'T_eval_h': (build_s + (k_med + 1) * t_cycle) / 3600,
        'T_eval_at_min_max_K_h': [(build_s + (min(cyc) + 1) * t_cycle) / 3600, (build_s + (max(cyc) + 1) * t_cycle) / 3600],
        'srp1_reference_per_cycle_s': {'definition': 'run_admm_arm_s / cycles_run per distinct Phase A candidate',
                                       'mean': statistics.mean(per_cycle_srp1),
                                       'median': statistics.median(per_cycle_srp1)},
        'memory_caveat': ('neither paper run completed initialization (r1 aborted at the 24 GiB process-tree '
                          'watchdog in the DSO init, r2 at the swap-growth guard in the TSO init); the '
                          'projection is a wall-time figure only and says nothing about feasibility in memory'),
    }
    return out


# ======================================================================================
#  Markdown
# ======================================================================================
def _f(v, nd=0):
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, (int, float)):
        return f'{v:,.{nd}f}'
    return str(v)


def _cand(e):
    parts = [f'n{n} {e["candidate_canonical"]["nodes"][n][0]}/{e["candidate_canonical"]["nodes"][n][1]}'
             for n in e['active_nodes']]
    return (' + '.join(parts) or 'x = 0') + f' @{e["investment_year"]}'


def _t1_rows(evs, extra_t2=False):
    h = ('| # | stage:label | dup | key | x (P MVA/E MWh) | cyc | Q gross | I(x) | F | F - F0 | res. bar | bar | '
         'term. step | win. descent | mono | r10 net | r10 gross | settl. rem. | salvage | net recourse | '
         'EFC/day max (node) | min SoH (node) | floor row |' + (' step<2000 |' if extra_t2 else ''))
    lines = [h, '|' + '---|' * (h.count('|') - 1)]
    for i, e in enumerate(evs, 1):
        efc = '; '.join(f'n{n} {_f(s["efc_per_day_max"], 3)}' for n, s in e['storage_per_active_node'].items()) or '-'
        soh = '; '.join(f'n{n} {_f(s["terminal_soh_min_over_active_cohort_years"], 3)}'
                        for n, s in e['storage_per_active_node'].items()) or '-'
        lines.append('| ' + ' | '.join([
            str(i), e['identity'], ('dup of ' + ','.join(e['duplicate_of'])) if e['duplicate_of'] else '',
            e['candidate_key'][:12], _cand(e), str(e['cycles_run']), _f(e['Q_gross_eur']), _f(e['I_x_eur']),
            _f(e['F_eur']), _f(e['F_minus_F0_eur']), _f(e['F_minus_F0_resolution_bar_eur']), _f(e['bar_eur']),
            _f(e['terminal_gross_step_abs']), _f(e['window_descent_signed']),
            (e['window_direction'] if e['window_monotone'] else 'no'),
            _f(e['terminal_step_over_threshold_production_net'], 4), _f(e['terminal_gross_step_over_threshold'], 4),
            _f(e['settlement_remainder_eur']), _f(e['terminal_salvage_value_eur'], 2),
            _f(e['net_operational_recourse_eur']), efc, soh, _f(e['soh_floor_row_active_at_terminal'])]
            + ([_f(e['T2_terminal_step_lt_2000'])] if extra_t2 else [])) + ' |')
    return lines


def render_md(res):
    L = [f'# {STAGE}', '', f'Generated {res["generated_utc"]} at git HEAD `{res["git_HEAD"]}` by '
         f'`p515_s45_phase_a_tables.py` (sha256 `{res["script_sha256"]}`). Zero solves: guard counts '
         f'`{res["solve_profile_guard"]["counts"]}`, verify(0) failures `{res["solve_profile_guard"]["verify_0_failures"]}`.',
         '', f'**Objective convention (all tables):** {OBJECTIVE_CONVENTION}', '',
         '## Column formulas (T1, T2)', '']
    for k, v in res['T1']['formulas'].items():
        L.append(f'- **{k}**: {v}')
    L += ['', f'## T1 -- all {len(res["T1"]["rows"])} evaluations ({res["T1"]["n_distinct_keys"]} distinct keys)', '',
          f'Objective convention: {OBJECTIVE_CONVENTION}', '']
    L += _t1_rows(res['T1']['rows'])
    L += ['', f'## T2 -- decision-relevant subset: F - F(x=0) < {T2_THRESHOLD_EUR:,.0f} EUR '
              f'({len(res["T2"]["rows"])} rows)', '', f'Objective convention: {OBJECTIVE_CONVENTION}', '',
          f'Last column: terminal gross step < {T2_TERMINAL_STEP_TEST_EUR:,.0f} EUR (Advisor test for excluding a '
          'size-dependent unfinished-descent bias).', '']
    L += _t1_rows(res['T2']['rows'], extra_t2=True)
    L += ['', f'Unfinished-settling flag (terminal step >= {UNFINISHED_STEP_EUR:,.0f} and monotone 10-step window): '
              + (', '.join(res['unfinished_settling_points']) or 'none'), '']
    t3 = res['T3']
    L += ['## T3 -- break-even from the node-7 2025 surface', '', f'Objective convention: {OBJECTIVE_CONVENTION}', '']
    for k, v in t3['formulas'].items():
        L.append(f'- **{k}**: {v}')
    L += ['', f'n = {t3["n"]} distinct candidates:', '', '| identity | P | E | h | value | fitted | residual | bar | term. step |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in t3['points']:
        L.append(f'| {r["identity"]} (evals: {", ".join(r["duplicate_evaluations"])}) | {r["P"]} | {r["E"]} | '
                 f'{r["duration_h"]:.2f} | {_f(r["value"])} | {_f(r["fitted_value"])} | {_f(r["residual"])} | '
                 f'{_f(r["bar_eur"])} | {_f(r["terminal_gross_step_abs"])} |')
    c, s = t3['coefficients'], t3['standard_errors']
    u = t3['unit_costs_2025']
    L += ['', f'- a = {_f(c["a_eur"])} +/- {_f(s["a_eur"])} EUR; b = {_f(c["b_eur_per_MWh"])} +/- '
              f'{_f(s["b_eur_per_MWh"])} EUR/MWh; c = {_f(c["c_eur_per_MVA"])} +/- {_f(s["c_eur_per_MVA"])} EUR/MVA; '
              f'cov(b,c) = {t3["cov_b_c"]:.6g}; dof = {t3["dof"]}',
          f'- residual rms = {_f(t3["residual_rms_eur"])} EUR; residual max |.| = {_f(t3["residual_max_abs_eur"])} EUR',
          f'- unit costs 2025 (from W2 table): energy new {_f(u["energy_new_eur_per_MWh"], 2)}, power '
          f'{_f(u["power_new_eur_per_MVA"], 2)}; energy old {_f(u["energy_old_eur_per_MWh"], 2)}, power old '
          f'{_f(u["power_old_eur_per_MVA"], 2)}']
    i = t3['i_breakeven_energy_cost_smallest_4h_eur_per_MWh']
    ii = t3['ii_breakeven_energy_cost_marginal_4h_MWh_eur_per_MWh']
    L += [f'- (i) smallest 4 h unit {i["identity"]}: V = {_f(i["V_measured_eur"])} (resolution {_f(i["V_resolution_bar_eur"])}); '
          f'break-even energy cost e* = {_f(i["e_star"], 2)} EUR/MWh = {i["e_star_over_e_new"]:.4f} x new, '
          f'{i["e_star_over_e_old"]:.4f} x old',
          f'- (ii) marginal 4 h MWh: value b + c/4 = {_f(ii["marginal_value_per_4h_MWh_eur"], 2)}; e* = '
          f'{_f(ii["e_star"], 2)} +/- {_f(ii["se"], 2)} EUR/MWh = {ii["e_star_over_e_new"]:.4f} x new, '
          f'{ii["e_star_over_e_old"]:.4f} x old']
    for m in t3['iii_iv_value_multiplier']:
        L.append(f'- (iii)/(iv) {m["identity"]} ({m["P"]}/{m["E"]}): V meas {_f(m["V_measured_eur"])} (fit '
                 f'{_f(m["V_fitted_eur"])}); I new {_f(m["I_new_eur"])} -> m = {m["m_new_measured"]:.3f} (fit '
                 f'{m["m_new_fitted"]:.3f}); I old {_f(m["I_old_eur"])} -> m = {m["m_old_measured"]:.3f} (fit '
                 f'{m["m_old_fitted"]:.3f})')
    L += ['', f'(v) {t3["v_label"]}', '', '| h | value/MWh | cost/MWh new | ratio new | cost/MWh old | ratio old | extrapolation |',
          '|---|---|---|---|---|---|---|']
    for r in t3['v_ratio_by_duration']:
        L.append(f'| {r["h"]} | {_f(r["value_per_MWh_eur"], 2)} | {_f(r["cost_per_MWh_new_eur"], 2)} | '
                 f'{r["ratio_new"]:.4f} | {_f(r["cost_per_MWh_old_eur"], 2)} | {r["ratio_old"]:.4f} | '
                 f'{"EXTRAPOLATION" if r["extrapolation"] else "in range"} |')
    t4 = res['T4']
    L += ['', '## T4 -- additivity', '', f'Objective convention: {OBJECTIVE_CONVENTION}', '']
    for k, v in t4['formulas'].items():
        L.append(f'- **{k}**: {v}')
    L += ['', '| combined | V combined | singles | sum singles | ratio | difference | res. bar | res. terminal | '
              '|diff|/res.bar | |diff|/fit rms |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in t4['rows']:
        L.append(f'| {r["identity"]} | {_f(r["V_combined_eur"])} | '
                 + '; '.join(f'{s["identity"]} {_f(s["V_eur"])}' for s in r['singles'])
                 + f' | {_f(r["sum_single_V_eur"])} | {r["ratio"]:.4f} | {_f(r["difference_eur"])} | '
                   f'{_f(r["resolution_bar_eur"])} | {_f(r["resolution_terminal_eur"])} | '
                   f'{r["abs_diff_over_resolution_bar"]:.3f} | {r["abs_diff_over_fit_residual_rms"]:.3f} |')
    L.append(f'\nfit residual rms (T3) = {_f(t4["fit_residual_rms_eur"])} EUR')
    t5 = res['T5']
    L += ['', '## T5 -- paper-scale memory and timing', '']
    for k, v in t5['definitions'].items():
        L.append(f'- **{k}**: {v}')
    L.append('')
    for run, R in t5['runs'].items():
        L.append(f'### {run}: status {R["cycle_status"]} (exit {R["cycle_exit_code"]}), abort at '
                 f'"{R["stage_at_abort"]}" ({R["abort_cause"]})')
        if 'solves' not in R:
            continue
        L.append(f'solve labels {R["n_solve_labels"]} (completed {R["n_completed_solve_labels"]}); '
                 f'DSO interval {R["DSO_interval_s"]}; TSO interval {R["TSO_interval_s"]}')
        for key in ('increment_all_consecutive', 'increment_within_network_completed'):
            st = R[key]
            L.append(f'- {key}: n {st["n"]}, mean {st["mean_gib"]:.4f}, median {st["median_gib"]:.4f}, first-10 '
                     f'{st["first10_mean_gib"]:.4f}, last-10 {st["last10_mean_gib"]:.4f} GiB')
        for b in R['unlabelled_build_blocks']:
            L.append(f'- build block "{b["stage"]}": {b["interval_s"]:.1f} s, footprint +{b["footprint_increment_gib"]:.3f} GiB'
                     + ('' if b['completed'] else ' (ABORTED inside this block; incomplete)'))
        if 'planner_reproduction_variants' in R:
            L.append('- Planner-figure reproduction variants: ' + json.dumps(
                {k: round(v, 4) for k, v in R['planner_reproduction_variants'].items()}))
        L += ['', '| solve label | interval s | samples | footprint after GiB | completed |', '|---|---|---|---|---|']
        for s in R['solves']:
            L.append(f'| {s["stage"]} | {s["interval_s"]:.3f} | {s["n_samples"]} | {s["footprint_self_after_gib"]:.3f} | '
                     f'{_f(s["completed"])} |')
        L.append('')
    L.append(f'paper block intervals (s): {json.dumps(t5["paper_block_interval_s"])}')
    for p in ('init_pass', 'cycle1_pass'):
        S = t5['srp1'][p]
        L.append(f'- SRP1 {S["pass"]}: {S["n_solve_labels_observed"]} solve labels observed; footprint '
                 f'{S["footprint_after_first_label_gib"]:.4f} -> {S["footprint_after_last_label_gib"]:.4f} GiB, total '
                 f'+{S["total_increment_first_to_last_gib"]:.4f} GiB; per consecutive step mean '
                 f'{S["increment_per_consecutive_step_gib"]["mean_gib"]:.4f}; within-network steps only (no build '
                 f'block in between): {S["n_within_network_steps"]} steps, sum +{S["within_network_sum_gib"]:.4f} GiB. '
                 f'{S["note"]}')
    L.append(f'- SRP1 init pass spans the build blocks {t5["srp1"]["init_includes_network_build_blocks"]}')
    P = t5['projection']
    L += ['', f'Projection: {P["formula_per_cycle"]}. {P["formula_per_evaluation"]}. K = {P["K_assumed"]} '
              f'({P["K_source"]}).', f'- T_cycle = {P["T_cycle_s"]:.1f} s; T_build = {P["T_build_s"]:.1f} s; '
              f'T_eval = {P["T_eval_s"]:.0f} s = {P["T_eval_h"]:.2f} h (K min/max: {P["T_eval_at_min_max_K_h"][0]:.2f} / '
              f'{P["T_eval_at_min_max_K_h"][1]:.2f} h)',
          f'- SRP1 reference per-cycle wall: {json.dumps(P["srp1_reference_per_cycle_s"])}',
          f'- {P["memory_caveat"]}', '']
    return '\n'.join(L) + '\n'


# ======================================================================================
#  manifest
# ======================================================================================
def _write_manifest():
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    files = {}
    for name in sorted(os.listdir(OUT_DIR)):
        cand = os.path.join(OUT_DIR, name)
        if cand == path or not os.path.isfile(cand):
            continue
        files[os.path.relpath(cand, REPO)] = {'sha256': _sha256_file(cand), 'bytes': os.path.getsize(cand)}
    with open(os.path.join(OUT_DIR, RESULTS_NAME)) as handle:
        inputs = json.load(handle)['inputs']
    with open(path, 'w') as handle:
        json.dump({'stage': STAGE, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'git_HEAD': _git(['rev-parse', 'HEAD'])[1].strip(),
                   'script': {'path': 'p515_s45_phase_a_tables.py',
                              'sha256': _sha256_file(os.path.abspath(__file__))},
                   'files': files, 'hash_inventory_of_inputs': inputs}, handle, indent=1)
    print(f'wrote {path} ({len(files)} files, {len(inputs)} inputs)')


# ======================================================================================
#  main
# ======================================================================================
def main():
    started = datetime.now(timezone.utc).isoformat()
    results_path = os.path.join(OUT_DIR, RESULTS_NAME)
    if os.path.exists(results_path):
        raise RuntimeError(f'refusing to overwrite {results_path}')
    os.makedirs(OUT_DIR, exist_ok=True)
    guard = SolveProfileGuard(permitted=(), label='P5.15 A27 W18 phase A tables').install()
    try:
        concurrency = _preflight_no_concurrent_harness()
        _log('[W18] concurrency preflight passed')
        inp = _Inputs()
        evals = build_evaluations(inp)
        if len(evals) != EXPECTED_EVALS:
            raise RuntimeError(f'{len(evals)} evaluations != {EXPECTED_EVALS}')
        ix_w2 = inp.load(IX_W2_REL)
        ix_w16 = inp.load(IX_W16_REL)
        attach_ix(evals, ix_w2, ix_w16)
        x0, keys = finish_t1(evals)
        n_dup = sum(1 for v in keys.values() if len(v) > 1)
        if len(keys) != EXPECTED_DISTINCT or n_dup != 4:
            raise RuntimeError(f'{len(keys)} distinct keys / {n_dup} duplicated != {EXPECTED_DISTINCT} / 4')
        _log(f'[W18] {len(evals)} evaluations, {len(keys)} distinct keys, x0 = {x0["identity"]}')
        t3 = build_t3(evals, x0, ix_w2)
        t4 = build_t4(evals, x0, t3)
        t5 = build_t5(inp, evals)
        inp.check_clean()
        _log('[W18] all inputs git-tracked and clean')
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(f'solve-profile guard: {failures}')
    t1_formulas = {
        'Q': 'certified_cost_gross_settlement_excluded (campaign_results.json); equals the terminal per-cycle '
             'gross_operational_cost (checked)',
        'I(x)': 'I_new_eur of the candidate key in the W2 or W16 I(x) table (checked equal to the campaign I_x_eur)',
        'F': 'I(x) + Q',
        'F - F0': f'F(x) - F(x = 0), x = 0 = {x0["identity"]}',
        'res. bar': 'resolution of F - F0: bar(x) + bar(x0)',
        'bar': 'max over the last 10 cycles of |Q[k] - Q[k-1]| (gross; campaign bar.value, recomputed and checked)',
        'term. step': '|Q[K] - Q[K-1]|, K = last cycle (per_cycle_record.jsonl)',
        'win. descent': 'Q[K-10] - Q[K] (signed; > 0 = still descending)',
        'mono': 'last 10 gross steps Q[k] - Q[k-1], k = K-9..K, all <= 0 (non-increasing) or all >= 0 '
                '(non-decreasing); "no" = mixed signs',
        'r10 net': 'rule ten, production: terminal |net recourse step| / terminal objective tolerance',
        'r10 gross': 'rule ten, gross: terminal |gross step| / terminal objective tolerance',
        'settl. rem.': 'evaluation_record.json settlement_remainder.value = recourse_components.'
                       'interface_settlement_total = T_TSO + sum T_DSO at the certified point',
        'salvage': 'evaluation_record.json recourse_components.terminal_salvage_value',
        'net recourse': 'recourse_components.net_operational_recourse = gross - salvage',
        'EFC/day max': 'storage_per_node[n].efc_per_day_max (max over the active cohort-years)',
        'min SoH': 'storage_per_node[n].terminal_soh_min_over_active_cohort_years',
        'floor row': 'any storage_per_node[*].soh_floor_rows_active_at_terminal non-empty',
        'unfinished flag': f'terminal step >= {UNFINISHED_STEP_EUR:,.0f} EUR AND mono != "no"',
        'T2 filter': f'F - F0 < {T2_THRESHOLD_EUR:,.0f} EUR (x = 0 itself included as the reference row)',
        'step<2000': f'terminal step < {T2_TERMINAL_STEP_TEST_EUR:,.0f} EUR',
    }
    evals_sorted = sorted(evals, key=lambda e: (STAGES.index(e['stage']), e['label']))
    t2 = sorted([e for e in evals if e['F_minus_F0_eur'] < T2_THRESHOLD_EUR], key=lambda e: e['F_minus_F0_eur'])
    res = {
        'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'git_HEAD': _git(['rev-parse', 'HEAD'])[1].strip(),
        'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_0_failures': failures},
        'concurrency': concurrency,
        'objective_convention': OBJECTIVE_CONVENTION,
        'inputs': inp.files,
        'T1': {'objective_convention': OBJECTIVE_CONVENTION, 'formulas': t1_formulas,
               'n_evaluations': len(evals), 'n_distinct_keys': len(keys),
               'duplicate_keys': {k: v for k, v in keys.items() if len(v) > 1}, 'x0': x0['identity'],
               'rows': evals_sorted},
        'T2': {'objective_convention': OBJECTIVE_CONVENTION, 'filter': t1_formulas['T2 filter'],
               'rows': t2},
        'unfinished_settling_points': [e['identity'] for e in evals_sorted if e['unfinished_settling_flag']],
        'T3': t3, 'T4': t4, 'T5': t5,
    }
    with open(results_path, 'w') as handle:
        json.dump(res, handle, indent=1)
    with open(os.path.join(OUT_DIR, MD_NAME), 'w') as handle:
        handle.write(render_md(res))
    _log(f'[W18] guard counts {guard.counts}; verify(0) failures {failures}')
    _log(f'[W18] T2 rows: {len(t2)}; unfinished-settling points: {res["unfinished_settling_points"]}')
    _log(f'[W18] T3 n={t3["n"]} a,b,c={t3["coefficients"]} se={t3["standard_errors"]} rms={t3["residual_rms_eur"]:.1f}')
    _log(f'[W18] wrote {results_path} and {MD_NAME}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        _write_manifest()
    else:
        main()
