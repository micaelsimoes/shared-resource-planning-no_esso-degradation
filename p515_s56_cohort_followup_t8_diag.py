"""P5.15 S56 cohort follow-up -- T8 diagnosis, ZERO-SOLVE, from the saved outputs of the spec v1 (287c0da6) run.

T8 (frozen): within-cohort simultaneity L = sum_t min(pch_c, pdch_c) dt <= 1e-6 MWh/day per cohort after the change
(H_B dump case, complementarity row active). Measured 2.475e-4 MWh/day: FAILED as frozen. This script only reads the
committed run record and computes the quantities that discriminate two readings:

  (B) barrier residual: the leak is the interior-point distance from the pch = 0 / pdch = 0 vertex, set by IPOPT's
      terminal barrier parameter (production's barrier identity, shared_energy_storage_data.py: min(pch, pdch) =
      mu_final / (s_obj * (a_ch + a_dch))); it scales with mu_final and leaves an inactive row's body below its bound;
  (G) genuine simultaneous charge and discharge permitted by the row: the per-cohort complementarity row
      (pch / S_c) * (pdch / S_c) <= ESS_COMPLEMENTARITY_TOLERANCE is a RELAXATION; at a binding row
      min(pch, pdch) = TOL * S_c^2 / max(pch, pdch), independent of mu_final.

Formulas (every statistic written below is computed here, from the files named in `INPUTS`):
  hour_equivalents          = L_mwh_per_day / max_hourly_min_mw   (dt = 1 h; ~24 for a leak spread over the day)
  argmax_product_over_S2    = pch * pdch / S_c^2 at the production diagnostic's argmax cohort-period
                              (complementarity_diagnostics_last.complementarity_ratio_argmax), S_c = the cohort's
                              rated (investment) power from the spec plan
  implied_pch_at_charge_hour = (dE_max + m / eta_dch) / eta_ch, dE_max = the largest hourly rise of the cohort's
                              stored energy (soc_mwh, 'variable' source; period 0 from INIT * E_av = 0.5 * E_av),
                              m = the block-day's max hourly min(pch, pdch); ASSUMES the max-min hour is a max-charge
                              hour (checked directly only at the argmax record)
  implied_product_over_S2   = implied_pch_at_charge_hour * m / S_c^2
  mu_ratio / leak_ratio     = T8 vs T8b (same profile and kappa, two vs three cohorts): (B) predicts leak_ratio ~
                              mu_ratio, (G) predicts leak_ratio ~ 1
  closure_slack_penalty_delta (T9): v2 -> after change of `feasibility_penalty` divided by
                              (n_closure_slack_vars * PENALTY_SHARED_ESS_BALANCE), n = 2 * the closure rows of the
                              test plan (the 8c9fd6ec closure-slack pair per cohort-block-day, unwired by D2)

Guard: SolveProfileGuard(permitted=()) installed before any import and verified at exactly 0 solves / 0 launches.
Output (write-once): data/SRP1/Results/P515S56/cohort_followup_v1/t8_diagnosis/t8_diagnosis.json (+ manifest).
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_cohort_followup_t8_diag.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 cohort follow-up T8 diagnosis (zero-solve)').install()

import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
from definitions import PENALTY_SHARED_ESS_BALANCE, ESS_COMPLEMENTARITY_TOLERANCE  # noqa: E402

RES = 'data/SRP1/Results/P515S56'
RUN = f'{RES}/cohort_followup_v1/run'
OUT = f'{RES}/cohort_followup_v1/t8_diagnosis'
INPUTS = {
    'spec': f'{RES}/frozen_cohort_followup_spec_v1_287c0da6.json',
    'tests': f'{RUN}/cohort_followup_tests.json',
    'solves_after': f'{RUN}/child_solves_after.json',
    'solves_before': f'{RUN}/child_solves_before.json',
    'v2_solves': f'{RES}/cohort_esso_v2/child_solves.json',
    'log_t8_after': f'{RUN}/ipopt_logs/after/optim_log_esso_node7_cycle008.txt.gz',
    'log_t8b_after': f'{RUN}/ipopt_logs/after/optim_log_esso_node7_cycle011.txt.gz',
}
SIGNIFICANT_MWH = 1e-6  # list a cohort block-day only when L exceeds this (the T8 threshold)


def _sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load(key):
    with open(os.path.join(HERE, INPUTS[key])) as handle:
        return json.load(handle)


def _ipopt_terminal(key):
    with gzip.open(os.path.join(HERE, INPUTS[key]), 'rt') as handle:
        text = handle.read()
    iters = re.findall(r'^\s*\d+r?\s+\S+\s+\S+\s+\S+\s+(-?\d+\.\d+)\s', text, flags=re.M)
    def last(label):
        found = re.findall(rf'^{label}\.*:\s+(\S+)\s+(\S+)', text, flags=re.M)
        return {'scaled': float(found[-1][0]), 'unscaled': float(found[-1][1])} if found else None
    return {'lg_mu_last_iteration': float(iters[-1]) if iters else None,
            'iterations': int(re.findall(r'^Number of Iterations\.*:\s+(\d+)', text, flags=re.M)[-1]),
            'complementarity': last('Complementarity'),
            'constraint_violation': last('Constraint violation'),
            'variable_bound_violation': last('Variable bound violation'),
            'exit': re.findall(r'^EXIT: (.*)$', text, flags=re.M)[-1]}


def _leak_table(test):
    rows = []
    for block_day, rec in sorted(test['detectors']['block_day'].items()):
        for cohort, pc in sorted(rec['per_cohort'].items()):
            total = pc.get('within_cohort_simultaneity_mwh')
            peak = pc.get('within_cohort_simultaneity_max_mw')
            if total is None or total <= SIGNIFICANT_MWH:
                continue
            rows.append({'block_day': block_day, 'cohort_index': cohort, 'n_active': rec['n_active'],
                         'L_mwh_per_day': total, 'max_hourly_min_mw': peak,
                         'hour_equivalents': total / peak if peak else None})
    return rows


def _floor(test):
    """The barrier-level floor in the same solve: L of cohorts with no simultaneity incentive (min over block-days)."""
    values = [pc['within_cohort_simultaneity_mwh'] for rec in test['detectors']['block_day'].values()
              for pc in rec['per_cohort'].values() if pc.get('within_cohort_simultaneity_mwh') is not None]
    return {'min_L_mwh_per_day': min(values), 'per_hour_mw_if_uniform_24h': min(values) / 24.0}


def _charge_hour_check(test, s_c):
    out = []
    for block_day, rec in sorted(test['detectors']['block_day'].items()):
        if rec['n_active'] < 2:
            continue
        pc = rec['per_cohort']['0']
        m = pc['within_cohort_simultaneity_max_mw']
        if not m or m <= 0:
            continue
        soc = pc['soc_mwh']
        e0 = 0.5 * pc['e_available_mwh']
        d_e = [soc[0] - e0] + [soc[i] - soc[i - 1] for i in range(1, len(soc))]
        i_max = max(range(len(d_e)), key=lambda k: d_e[k])
        pch = (d_e[i_max] + m / rec['eta_dch']) / rec['eta_ch']
        out.append({'block_day': block_day, 'soc_source': pc['soc_source'], 'max_hourly_min_mw': m,
                    'dE_max_mwh': d_e[i_max], 'dE_max_period': i_max,
                    'charge_periods_dE_gt_1e-3': [k for k in range(len(d_e)) if d_e[k] > 1e-3],
                    'implied_pch_at_charge_hour_mw': pch, 'implied_product_over_S2': pch * m / s_c ** 2,
                    'implied_product_over_tol': pch * m / s_c ** 2 / ESS_COMPLEMENTARITY_TOLERANCE})
    return out


def _argmax(test, s_c):
    a = test['solve']['complementarity_diagnostics_last']['complementarity_ratio_argmax']
    product = a['pch'] * a['pdch'] / s_c ** 2
    return {'record': a, 'S_c_mva': s_c, 'product_over_S2': product,
            'product_over_tol': product / ESS_COMPLEMENTARITY_TOLERANCE,
            'pch_over_pdch': a['pch'] / a['pdch'] if a['pdch'] else None}


def main():
    os.makedirs(os.path.join(HERE, OUT), exist_ok=False)
    spec = _load('spec')
    tests = _load('tests')
    after = _load('solves_after')['tests']
    before = _load('solves_before')['tests']
    v2 = _load('v2_solves')['tests']
    s_c = float(spec['solves']['t_v']['plan'][0][2])  # 2025 cohort rated power (MVA) of the H_B plan

    t8, t8b = after['t_v_tv_on'], after['t8b_two_cohort']
    mu8 = t8['solve']['complementarity_diagnostics_last']['mu_final']
    mu8b = t8b['solve']['complementarity_diagnostics_last']['mu_final']
    l8 = t8['within']['2025']['max_mwh_per_day']
    l8b = t8b['within']['2025']['max_mwh_per_day']
    m8 = t8['within']['2025']['max_mw']
    m8b = t8b['within']['2025']['max_mw']

    result = {
        'schema': 'p515_s56_cohort_followup_t8_diag_v1',
        'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'instance': {'node': 7, 'case': 'T8 = after-tree t_v_tv_on (H_B: plan ' + json.dumps(spec['solves']['t_v']['plan'])
                     + ', TV on, kappa ' + repr(t8['kappa']) + '); T8b = t8b_two_cohort (plan '
                     + json.dumps(spec['solves']['t_v']['plan_two_cohort']) + ')',
                     'spec_sha256': _sha(os.path.join(HERE, INPUTS['spec']))},
        'inputs_sha256': {k: _sha(os.path.join(HERE, v)) for k, v in INPUTS.items()},
        'constants': {'ESS_COMPLEMENTARITY_TOLERANCE': ESS_COMPLEMENTARITY_TOLERANCE,
                      'PENALTY_SHARED_ESS_BALANCE': PENALTY_SHARED_ESS_BALANCE, 'S_c_mva_2025_cohort': s_c},
        't8_frozen': tests['gates']['T8_T9']['T8_predictions'],
        'row': {'t8': t8['rows'], 't8b': t8b['rows'],
                't8_body_max_minus_tol': t8['rows']['complementarity_body_max'] - t8['rows']['complementarity_tolerance'],
                't8b_body_max_minus_tol': t8b['rows']['complementarity_body_max'] - t8b['rows']['complementarity_tolerance']},
        'argmax': {'t8': _argmax(t8, s_c), 't8b': _argmax(t8b, s_c)},
        'charge_hour_check': {'t8': _charge_hour_check(t8, s_c), 't8b': _charge_hour_check(t8b, s_c)},
        'mu_independence': {'mu_final_t8': mu8, 'mu_final_t8b': mu8b, 'mu_ratio': mu8 / mu8b,
                            'L_t8': l8, 'L_t8b': l8b, 'L_ratio': l8 / l8b,
                            'max_hourly_min_t8': m8, 'max_hourly_min_t8b': m8b, 'max_hourly_min_ratio': m8 / m8b},
        'ipopt_terminal': {'t8': _ipopt_terminal('log_t8_after'), 't8b': _ipopt_terminal('log_t8b_after'),
                           'solver_options': _load('solves_after')['solver_options']},
        'leak_tables': {
            'after_t_v_tv_on': _leak_table(t8), 'after_t8b_two_cohort': _leak_table(t8b),
            'before_t_v_tv_on': _leak_table(before['t_v_tv_on']),
            'v2_t_v_tv_on': _leak_table(v2['t_v_tv_on']),
            'v2_t_iv_peak_barrier_identity_reference': _leak_table(v2['t_iv_peak']),
            'after_t_iv_peak': _leak_table(after['t_iv_peak'])},
        'reference_rows': {'after_t_iv_peak': after['t_iv_peak'].get('rows'),
                           'after_t_iv_peak_argmax': _argmax(after['t_iv_peak'], s_c),
                           'v2_t_iv_peak_argmax': v2['t_iv_peak']['solve']['complementarity_diagnostics_last'][
                               'complementarity_ratio_argmax']},
        'barrier_floor_same_solve': {'t8': _floor(t8), 't8b': _floor(t8b)},
    }

    # T9: the unpredicted feasibility_penalty changes against the unwired closure-slack term.
    # The closure-slack pair existed per (cohort, block, day) on EVERY multi-cohort block of the solved ESSO model
    # (all blocks and days are in the model even where a test reads one block-day), so the count comes from the
    # test's plan: active cohorts per block = plan cohorts installed in or before the block (no cohort reaches its
    # calendar life inside 2025-2035; cross-checked below against T3's measured closure-row count for p2025_2030).
    years = sorted(int(y) for y in spec['grids']['srp1']['years'])
    n_days = 4

    def _closure_rows(plan):
        rows = 0
        for y in years:
            n = sum(1 for entry in plan if int(entry[1]) <= y)
            if n >= 2:
                rows += n * n_days
        return rows

    plan_of = {'t_ii_a_indep': spec['solves']['t_ii']['plan'], 't_ii_b': spec['solves']['t_ii']['plan'],
               't_iii_indep': spec['solves']['t_iii']['plan'], 't_iv_peak': spec['solves']['t_iv']['plan'],
               't_iv_control': spec['solves']['t_iv']['plan'], 't_v_tv_on': spec['solves']['t_v']['plan'],
               't_v_tv_off_oldest': spec['solves']['t_v']['plan'], 't_v_tv_off_newest': spec['solves']['t_v']['plan']}
    t3_measured = tests['gates']['T3']['srp1']['measured']['esso_closure_rows']
    t9 = {'_closure_rows_cross_check': {'plan': spec['t3']['plan'], 'formula': _closure_rows(spec['t3']['plan']),
                                        't3_measured': t3_measured}}
    comparison = tests['gates']['T8_T9']['T9_comparison']
    for name, rec in comparison.items():
        test, _, quantity = name.partition('.')
        if quantity != 'feasibility_penalty' or not rec['changed'] or test not in after:
            continue
        n_vars = 2 * _closure_rows(plan_of[test])
        delta = rec['new'] - rec['v2']
        t9[test] = {'v2': rec['v2'], 'new': rec['new'], 'delta': delta, 'n_closure_slack_vars_v2': n_vars,
                    'implied_value_per_var': -delta / (n_vars * PENALTY_SHARED_ESS_BALANCE),
                    'p_slack_max_abs_after': after[test]['detectors']['slacks']['p']['max_abs']}
    result['t9_closure_slack_penalty_delta'] = t9

    failures = GUARD.verify(0)
    result['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': failures}
    target = os.path.join(HERE, OUT, 't8_diagnosis.json')
    with open(target, 'x') as handle:
        handle.write(GRIO.dumps(result, indent=1, sort_keys=True))
    manifest = {'schema': 'p515_s56_cohort_followup_t8_diag_v1_manifest',
                'files': {os.path.relpath(target, HERE): _sha(target),
                          os.path.basename(__file__): _sha(os.path.abspath(__file__)), **{
                              v: _sha(os.path.join(HERE, v)) for v in INPUTS.values()}}}
    with open(os.path.join(HERE, OUT, 'manifest_sha256.json'), 'x') as handle:
        handle.write(GRIO.dumps(manifest, indent=1, sort_keys=True))
    print(f'wrote {target}; guard {result["guard"]}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
