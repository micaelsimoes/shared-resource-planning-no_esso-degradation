"""P5.15 Addendum 68 decision 2, task W170 -- provenance of the 3 x 3 mean-profile prediction R = 0.9331.
ZERO SOLVES, no model construction, no pickle, no production read. Reads committed JSON records only.

WHAT IT DOES
  The prediction R = 0.9331 (Addendum 44, "R = 0.9331 recorded"; T11 row "R predicted from the mean-profile spread")
  is W52's R_r2 of the market subset [1, 2, 3] (p515_s53_3x3_selection.py @ 7550f441, record
  data/SRP1/Results/P515S53/selection_3x3/selection_3x3.json). W52 gives the formula and stores its inputs
  explicitly, so this script re-applies the formula to the stored inputs (no recomputation of any price):
    numerator   spread_r2([1,2,3]) = sum_(y,d) w_r2(y,d) * s(y,d) / sum_(y,d) w_r2(y,d) over the 3 x 3 / paper
                horizon (instances.paper.years, .days of selection_3x3.json), s(y,d) = market.ranking[subset
                [1,2,3]].per_block_spread[y/d] (4 h spread of the 1/3-renormalized mean market-price profile);
    denominator SRP1 Z4 spread_r2 = the same weighted average over Z4's per_representative_day rows of
                data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json (SRP1 horizon, single scenario);
    w_r2(y,d) = num_years[y] * num_days[d] / 1.02^(y - 2025)  (production's block weight, network_data.py
                get_primal_value).
  It checks the reproduction against the recorded values to 1e-9, and records (as readings, not predictions):
    * the total block weights of the two horizons (the ratio of weighted AVERAGES equals the ratio of weighted
      TOTALS only when the totals agree);
    * the same-year (2025) ratio from the per-year spreads already committed in both records;
    * the identity of the 3 x 3 instance as run with the inputs (years, scenario prefix, probabilities) from
      the W89 instance record.

OUTPUT (write-once, new directory) data/SRP1/Results/P515S53/w170_r_prediction/:
    w170_r_prediction.json, launch.log, manifest_sha256.json

LAUNCH (repo root, canonical interpreter, attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w170_r_prediction
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w170_r_prediction_check.py \\
        > data/SRP1/Results/P515S53/w170_r_prediction/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w170_r_prediction_check.py --manifest
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W170 R prediction provenance (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W170: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W170: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402

OUT_REL = 'data/SRP1/Results/P515S53/w170_r_prediction'
OUT_DIR = os.path.join(REPO, OUT_REL)
OUT_JSON = 'w170_r_prediction.json'
TOL = 1e-9
RATE = 0.02            # production DiscountFactor of both case files (recorded in selection_3x3.json instances)
Y0 = 2025
F_TH = 'top4_mean_minus_bottom4_mean'

INPUTS = {  # path -> sha256 required (refuse on any difference)
    'selection_3x3': ('data/SRP1/Results/P515S53/selection_3x3/selection_3x3.json',
                      '291be9c798d7914a8d5962871aa266e3e88d9a6f01ef8cbaa9a69101605cdd12'),
    'z4': ('data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json',
           '16579f3202781f74b1d2d83ae2809340d1334718a2dee4e73f4895f05e570ef6'),
    'instance_record_3x3': ('data/SRP1/Results/P515S53/w89_3x3/instance/instance_record.json',
                            'a47dc358edc1a69279ddd1965a587efc2906eb6c770d336cb9c8f34227f910ea'),
    'case_3x3': ('data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json',
                 '2a64e3c5bd06c30c8a015207f41b69257b001094cbd4b185a84394a6765c5871'),
    'campaign_3x3': ('data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json',
                     '587d3f1afb658e8a501d34b51bb8b269dd0c229391c1a26469f01001a8aad566'),
    'frozen_tables': ('data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json',
                      '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'),
    'case_srp1': ('data/SRP1/SRP1.json', '61a794a7ce7a3fb983f2e92128eec446dd7dbb17ad75a3b3b1c5735f8bd4e4ef'),
    'w52_script': ('p515_s53_3x3_selection.py', 'f129a01e9e6492c3ab668cc8b075e4048c5fd7f7a0ff9269ec73899babb607d8'),
    'w22_script': ('p515_s47_market_spreads.py', '7524f32e4a9eea642833baef410b942bb949ff6db7a557d5dd47fbbe89863cc1'),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, check=True, text=True).stdout


def _w(year, num_years, num_days, rate):
    return num_years * num_days / (1.0 + rate) ** (int(year) - Y0)


def write_manifest():
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        if name == 'manifest_sha256.json':
            continue
        entries[name] = _sha(os.path.join(OUT_DIR, name))
    with open(os.path.join(OUT_DIR, 'manifest_sha256.json'), 'w') as f:
        json.dump({'created_utc': _utc(), 'directory': OUT_REL, 'files': entries,
                   'script': {os.path.basename(__file__): _sha(os.path.abspath(__file__))},
                   'inputs': {k: {'path': p, 'sha256': s} for k, (p, s) in INPUTS.items()}}, f, indent=1)
    print('manifest written:', entries)


def main():
    if '--manifest' in sys.argv:
        write_manifest()
        return 0
    started = _utc()
    if os.path.exists(os.path.join(OUT_DIR, OUT_JSON)):
        raise RuntimeError(f'{OUT_REL}/{OUT_JSON} exists -- write-once, refusing')
    os.makedirs(OUT_DIR, exist_ok=True)
    checks = {}
    hashes = {}
    for k, (p, s) in INPUTS.items():
        got = _sha(os.path.join(REPO, p))
        hashes[k] = {'path': p, 'sha256': got, 'declared': s}
        checks[f'input_sha256_{k}'] = got == s
    if not all(checks.values()):
        raise RuntimeError(f'input hash mismatch: {[k for k, v in checks.items() if not v]}')

    sel = json.load(open(INPUTS['selection_3x3'][0]))
    z4 = json.load(open(INPUTS['z4'][0]))['Z4']
    inst = json.load(open(INPUTS['instance_record_3x3'][0]))
    case3 = json.load(open(INPUTS['case_3x3'][0]))
    srp1 = json.load(open(INPUTS['case_srp1'][0]))
    camp = json.load(open(INPUTS['campaign_3x3'][0]))
    tabs = json.load(open(INPUTS['frozen_tables'][0]))

    # ---------------- numerator: 3 x 3 / paper horizon, subset [1, 2, 3] ----------------
    py, pd = sel['instances']['paper']['years'], sel['instances']['paper']['days']
    row = [r for r in sel['market']['ranking'] if r['subset'] == [1, 2, 3]]
    checks['subset_123_found_once'] = len(row) == 1
    row = row[0]
    num_rows = []
    for y in sorted(py):
        for d in pd:
            wu = py[y] * pd[d]
            num_rows.append({'year': int(y), 'day': d, 'num_years': py[y], 'num_days': pd[d], 'w_u': wu,
                             'w_r2': _w(y, py[y], pd[d], RATE), 's': row['per_block_spread'][f'{y}/{d}']})
    checks['per_block_spread_covers_all_20_blocks'] = len(num_rows) == len(row['per_block_spread']) == 20
    sw_u_n = sum(r['w_u'] for r in num_rows)
    sw_r_n = sum(r['w_r2'] for r in num_rows)
    num_u = sum(r['w_u'] * r['s'] for r in num_rows) / sw_u_n
    num_r = sum(r['w_r2'] * r['s'] for r in num_rows) / sw_r_n

    # ---------------- denominator: SRP1 horizon, single scenario (Z4) ----------------
    den_rows = []
    for r in z4['per_representative_day']:
        den_rows.append({'year': int(r['year']), 'day': r['day'], 'num_years': r['num_years'],
                         'num_days': r['num_days'], 'w_u': r['weight_undiscounted'], 'w_r2': r['weight_model_r2'],
                         'w_r2_recomputed': _w(r['year'], r['num_years'], r['num_days'], RATE), 's': r[F_TH]})
    checks['z4_weights_equal_formula'] = all(abs(r['w_r2'] - r['w_r2_recomputed']) <= TOL for r in den_rows)
    checks['z4_years_equal_srp1_case'] = (sorted({str(r['year']): r['num_years'] for r in den_rows}.items())
                                          == sorted(srp1['Years'].items()))
    sw_u_d = sum(r['w_u'] for r in den_rows)
    sw_r_d = sum(r['w_r2'] for r in den_rows)
    den_u = sum(r['w_u'] * r['s'] for r in den_rows) / sw_u_d
    den_r = sum(r['w_r2'] * r['s'] for r in den_rows) / sw_r_d

    R_r2 = num_r / den_r
    R_u = num_u / den_u
    checks['num_r2_reproduces_recorded'] = abs(num_r - row['spread_r2']) <= TOL
    checks['num_u_reproduces_recorded'] = abs(num_u - row['spread_u']) <= TOL
    checks['den_r2_reproduces_Z4'] = abs(den_r - z4['spread_summary']['all_years_model_block_weight_r2'][F_TH]) <= TOL
    checks['den_u_reproduces_Z4'] = abs(den_u - z4['spread_summary']['all_years_day_weighted_undiscounted'][F_TH]) <= TOL
    checks['R_r2_reproduces_recorded'] = abs(R_r2 - row['R_r2']) <= TOL
    checks['R_u_reproduces_recorded'] = abs(R_u - row['R_u']) <= TOL
    checks['R_r2_rounds_to_0.9331'] = round(R_r2, 4) == 0.9331
    checks['campaign_R_prefix_recorded_is_0.9331'] = camp['value_and_R']['R_prefix_recorded'] == 0.9331
    t11 = [r for r in tabs['tables']['three_by_three']['rows'] if r['quantity'] == 'R predicted from the mean-profile spread']
    checks['T11_row_is_0.9331'] = len(t11) == 1 and t11[0]['value'] == 0.9331

    # ---------------- 3 x 3 instance as run vs the numerator's inputs ----------------
    checks['case_3x3_years_equal_numerator_years'] = case3['Years'] == py
    checks['case_3x3_days_equal_numerator_days'] = case3['Days'] == pd
    checks['case_3x3_discount_equals_rate'] = case3['DiscountFactor'] == RATE and srp1['DiscountFactor'] == RATE
    pv = inst['prefix_verification']
    checks['3x3_realized_market_subset_is_123'] = pv['realized_subset_market'] == [1, 2, 3]
    checks['3x3_prefix_holds_bitwise_0_mismatches'] = pv['prefix_holds'] is True and pv['n_mismatches'] == 0
    checks['3x3_probabilities_one_third'] = pv['probabilities_3x3_all_one_third'] is True
    checks['numerator_pm_renormalized_one_third'] = sel['instances']['paper']['pm']['2025'] == [0.2] * 5

    # ---------------- readings (not predictions) ----------------
    per_year_n = row['per_year_spread_u']
    per_year_d = {k: v[F_TH] for k, v in z4['spread_summary']['per_year_day_weighted'].items()}
    readings = {
        'total_block_weight_undiscounted': {'3x3': sw_u_n, 'SRP1': sw_u_d, 'ratio_3x3_over_SRP1': sw_u_n / sw_u_d},
        'total_block_weight_model_r2': {'3x3': sw_r_n, 'SRP1': sw_r_d, 'ratio_3x3_over_SRP1': sw_r_n / sw_r_d},
        'ratio_of_weighted_totals_r2': (num_r * sw_r_n) / (den_r * sw_r_d),
        'ratio_of_weighted_totals_u': (num_u * sw_u_n) / (den_u * sw_u_d),
        'note_totals': ('R_r2 is a ratio of weighted AVERAGES (each instance normalised by its own total weight). '
                        'Under the same first-order model the ratio of weighted TOTALS (which is what a value '
                        'ratio V_3x3 / V_SRP1 compares) is R_r2 x (sum w_r2 3x3 / sum w_r2 SRP1). Reading only.'),
        'per_year_spread_u_3x3_subset_123': per_year_n,
        'per_year_spread_u_SRP1': per_year_d,
        'same_year_2025_ratio_u': per_year_n['2025'] / per_year_d['2025'],
        'note_2025': ('2025 is the only year both horizons share; SRP1 2025 prices = paper scenario 1 in 2025 '
                      '(Addendum 32). The ratio at 2025 holds the year fixed; R itself mixes the scenario set with '
                      'the two horizons. Reading only, not a prediction.'),
        'R_gn_recorded': row['R_gn'],
    }
    readings['decomposition_r2'] = {
        'R_r2': R_r2,
        'mean_year_weighted_r2_3x3': sum(r['w_r2'] * r['year'] for r in num_rows) / sw_r_n,
        'mean_year_weighted_r2_SRP1': sum(r['w_r2'] * r['year'] for r in den_rows) / sw_r_d,
    }

    GUARD_FAIL = GUARD.verify(0)
    checks['solve_guard_verify_0'] = GUARD_FAIL == []
    checks['pickle_blocked_and_unused'] = (pickle.load is _blocked_load and pickle.loads is _blocked_loads
                                           and PICKLE_COUNTS == {'load': 0, 'loads': 0})
    out = {
        'stage': 'P5.15 Addendum 68 decision 2, W170 -- provenance of the 3 x 3 mean-profile prediction R = 0.9331',
        'started_utc': started, 'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
        'script_sha256': _sha(os.path.abspath(__file__)), 'inputs': hashes,
        'formula': ('R_r2 = [sum_(y,d) w_r2 * s_mp(y,d) / sum w_r2]_(3x3 horizon, mean profile of market scen. 1-3, '
                    'pm 1/3) / [sum_(y,d) w_r2 * s(y,d) / sum w_r2]_(SRP1 horizon, its single scenario); s = mean of '
                    'the 4 highest minus mean of the 4 lowest of the 24 hourly market prices; w_r2 = num_years * '
                    'num_days / 1.02^(y - 2025)'),
        'numerator': {'horizon_years': py, 'days': pd, 'spread_r2': num_r, 'spread_u': num_u,
                      'recorded_spread_r2': row['spread_r2'], 'per_block': num_rows},
        'denominator': {'horizon_years': srp1['Years'], 'spread_r2': den_r, 'spread_u': den_u, 'per_block': den_rows},
        'R_r2_reproduced': R_r2, 'R_u_reproduced': R_u, 'R_r2_recorded': row['R_r2'], 'R_recorded_4dp': 0.9331,
        'readings': readings,
        'value_ratio_definition_T11': tabs['tables']['three_by_three']['formula_R'],
        'guard': {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD_FAIL},
        'pickle_counts': dict(PICKLE_COUNTS),
        'checks': checks, 'ok': all(checks.values()), 'ended_utc': _utc(),
    }
    with open(os.path.join(OUT_DIR, OUT_JSON), 'w') as f:
        GRIO.dump(out, f, indent=1)
    print(json.dumps({'R_r2_reproduced': R_r2, 'R_u_reproduced': R_u, 'readings': {
        k: v for k, v in readings.items() if not k.startswith('note')}, 'checks': checks}, indent=1))
    print('ok =', out['ok'])
    pickle.load, pickle.loads = _PICKLE_ORIG
    return 0 if out['ok'] else 2


if __name__ == '__main__':
    sys.exit(main())
