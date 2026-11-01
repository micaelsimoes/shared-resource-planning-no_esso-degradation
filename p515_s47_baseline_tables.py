"""
P5.15 Addendum 30 -- baseline tables for the Phase B review report (zero solves; reads committed JSON).

BASELINE = C2 (cycles 10000, DoD 0.80, EOL retention 0.80) + phi_cal 0.985 + soh_min 0.70.
C3 results are the sensitivity set and are NOT used here.

Formulas (all stated in the output JSON):
  value(x)        = Q(0) - Q(x)   (Q = certified gross_operational_cost, settlement excluded)
  F(x)            = I(x) + Q(x)
  node-7 fit      : value = a + b*E + c*P, OLS over the 10 node-7-alone 2025 S3 points, SE from
                    sigma^2 (X'X)^-1 with sigma^2 = RSS / (n - 3)
  breakeven_first : e* = (V_meas - 0.25 * p_cost) / 1.0     smallest 4 h unit (0.25 MVA / 1.0 MWh),
                    V_meas = its measured value; p_cost = 2025 power unit cost
  breakeven_marg  : e* = b + c/4 - p_cost/4                  marginal 4 h MWh
  ratio(h)        = (b + c/h) / (e_cost + p_cost/h)          marginal value/cost at duration h
Unit costs are read from the pinned W2 I(x) table (9e623dd3).

Output: data/SRP1/Results/P515S47/baseline_tables/baseline_tables.json (+ launch.log)
Command:
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_baseline_tables.py \
      > data/SRP1/Results/P515S47/baseline_tables/launch.log 2>&1
"""
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S47 baseline tables (no solves)').install()

R = 'data/SRP1/Results'
S2 = f'{R}/P515S47/campaign_s47_recert/campaign_results.json'
S3 = f'{R}/P515S47/campaign_s47_a1a_baseline/campaign_results.json'
PB = f'{R}/P515S47/campaign_s47_phase_b/campaign_results.json'
W2 = f'{R}/P515S45/investment_cost/investment_cost_results.json'
OUT = f'{R}/P515S47/baseline_tables'


def sha(p):
    with open(os.path.join(REPO, p), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(p):
    return json.load(open(os.path.join(REPO, p)))


def points(res):
    pts = res['points']
    return list(pts.values()) if isinstance(pts, dict) else pts


def main():
    out_path = os.path.join(REPO, OUT, 'baseline_tables.json')
    if os.path.exists(out_path):
        raise SystemExit(f'refusing to overwrite {out_path}')
    s3 = points(load(S3))
    w2 = load(W2)
    # unit costs: read the 2025 per-unit power/energy cost from the W2 table's candidate split
    unit = w2['candidates']['n7_4h_e1']
    assert unit['candidate_canonical'] == {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.25, 1.0], '9': [0.0, 0.0]}}
    p_cost = unit['I_new_power_eur'] / 0.25
    e_cost = unit['I_new_energy_eur'] / 1.0
    q0 = s3[0]['Q0_eur'] if 'Q0_eur' in s3[0] else None
    n7 = [p for p in s3 if p['label'].startswith('n7_')]
    X = np.array([[1.0, p['candidate_canonical']['nodes']['7'][1], p['candidate_canonical']['nodes']['7'][0]] for p in n7])
    y = np.array([p['value_eur'] for p in n7])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = len(y) - 3
    s2 = float(res @ res) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    a, b, c = (float(v) for v in coef)
    first = next(p for p in s3 if p['label'] == 'n7_4h_e1')
    best = min(s3, key=lambda p: p['F_eur'])
    be_first = (first['value_eur'] - 0.25 * p_cost) / 1.0
    be_first_best = (best['value_eur'] - best['candidate_canonical']['nodes'][[k for k, v in best['candidate_canonical']['nodes'].items() if v[0] > 0][0]][0] * p_cost) / 1.0
    be_marg = b + c / 4 - p_cost / 4
    ratios = {h: (b + c / h) / (e_cost + p_cost / h) for h in (2, 4, 6, 8, 10)}
    pb = load(PB)
    unit_poll = pb['poll_history'][-1]
    poll_rows = sorted(
        [{'label': cnd['label'], 'I_x_eur': cnd['I_x_eur'], 'F_minus_F0_eur': -cnd['F_inc_minus_F_eur'],
          'resolution_eur': cnd['resolution_eur'], 'outcome': cnd['outcome']}
         for cnd in unit_poll['candidates'] if cnd.get('F_eur') is not None],
        key=lambda r: r['F_minus_F0_eur'])
    doc = {
        'stage': 'P5.15 Addendum 30 baseline tables (zero solves)',
        'baseline': 'C2 (10000, 0.80, 0.80) + phi_cal 0.985 + soh_min 0.70 -- not mixed with C3',
        'objective_convention': 'Q = certified gross_operational_cost (settlement excluded); F = I + Q; value = Q(0) - Q(x)',
        'unit_costs_2025': {'power_eur_per_mva': p_cost, 'energy_eur_per_mwh': e_cost, 'source': W2},
        'node7_fit': {'n': len(y), 'labels': [p['label'] for p in n7], 'a': a, 'b': b, 'c': c,
                      'se': {'a': float(se[0]), 'b': float(se[1]), 'c': float(se[2])},
                      'residual_rms': float(np.sqrt((res ** 2).mean())), 'residual_max_abs': float(abs(res).max()),
                      'formula': 'value = a + b*E + c*P, OLS; SE = sqrt(diag(sigma^2 (X^T X)^-1)), sigma^2 = RSS/(n-3)'},
        'breakeven': {
            'first_unit_n7_eur_per_mwh': be_first, 'first_unit_n7_ratio_to_current': be_first / e_cost,
            'first_unit_best_node_label': best['label'], 'first_unit_best_node_eur_per_mwh': be_first_best,
            'first_unit_best_node_ratio_to_current': be_first_best / e_cost,
            'marginal_4h_eur_per_mwh': be_marg, 'marginal_4h_ratio_to_current': be_marg / e_cost,
            'marginal_4h_se_eur_per_mwh': float(np.sqrt(se[1] ** 2 + (se[2] / 4) ** 2)),
            'formulas': {'first': '(V_meas - P*p_cost)/E', 'marginal': 'b + c/4 - p_cost/4 (SE ignores the b-c covariance)'}},
        'value_to_cost_ratio_by_duration': {str(h): r for h, r in ratios.items()},
        'value_to_cost_note': 'h > 4 extrapolates beyond the evaluated 2-4 h range',
        'closest_ladder_point': {'label': best['label'], 'F_minus_F0_eur': best['F_eur'] - q0,
                                 'value_eur': best['value_eur'], 'I_x_eur': best['I_x_eur']},
        'phase_b_unit_poll': poll_rows,
        'phase_b_certificate': pb['termination_certificate'],
        'inputs_sha256': {p: sha(p) for p in (S2, S3, PB, W2)},
        'script_sha256': sha(os.path.relpath(os.path.abspath(__file__), REPO)),
    }
    failures = GUARD.verify(0)
    GUARD.uninstall()
    doc['solve_profile_guard'] = {'counts': GUARD.counts, 'verify0_failures': failures}
    os.makedirs(os.path.join(REPO, OUT), exist_ok=True)
    with open(out_path, 'x') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ('unit_costs_2025', 'node7_fit', 'breakeven', 'value_to_cost_ratio_by_duration', 'closest_ladder_point')}, indent=1))
    print('guard', GUARD.counts, failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
