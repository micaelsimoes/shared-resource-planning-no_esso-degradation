"""
P5.15 Addendum 28 -- ageing-batch mechanism table (zero solves; reads committed JSON only).

Preserves the formulas behind the Planner's reading of the ageing batch
(campaign s46_ageing, commit 1545571e) so that every figure cited in
P5_15_ADDENDUM28_AGEING_REPORT.md is recomputable:

  w_y            = 1 / 1.02**(y - 2025), y in {2025, 2030, 2035}
                   (production's per-representative-year discount factor,
                   shared_resources_planning.py:3421-3424; W19 Z3)
  AE             = sum_y w_y * SoH_used[y] / sum_y w_y      (MWh per rated MWh;
                   SoH_used = the SoH the model applies to available energy)
  EFC            = sum_y w_y * EFC_per_day[y] / sum_y w_y
  value          = Q(0) - Q_variant(x)          (gross, settlement excluded)
  value_per_AE   = value / AE
  elasticity     = ln(value / value_C3) / ln(AE / AE_C3)

C3 (baseline) per-block SoH_used and EFC/day are read from the committed A1a
record of n7_4h_e1 (the baseline's ageing capture: storage_per_node['7']);
the five variants' from their committed evaluation records
(ageing_trajectory_terminal.nodes['7'].cells).

Output: data/SRP1/Results/P515S46/ageing_mechanism/ageing_mechanism.json
Command:
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_ageing_mechanism.py \
      > data/SRP1/Results/P515S46/ageing_mechanism/launch.log 2>&1
"""
import glob
import hashlib
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S46 ageing mechanism table (no solves)').install()

OUT_DIR = os.path.join(REPO, 'data/SRP1/Results/P515S46/ageing_mechanism')
RESULTS = os.path.join(REPO, 'data/SRP1/Results/P515S46/campaign_s46_ageing/campaign_results.json')
A1A_GLOB = os.path.join(REPO, 'data/SRP1/Results/P515S45/campaign_s45_a1a/evals/*_n7_4h_e1/evaluation_record.json')
YEARS = (2025, 2030, 2035)
W = [1.0 / 1.02 ** (y - 2025) for y in YEARS]


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def pv(values):
    return sum(w * v for w, v in zip(W, values)) / sum(W)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'ageing_mechanism.json')
    if os.path.exists(out_path):
        raise SystemExit(f'refusing to overwrite {out_path}')
    inputs = {}
    res = json.load(open(RESULTS))
    inputs[os.path.relpath(RESULTS, REPO)] = sha(RESULTS)
    base_path = glob.glob(A1A_GLOB)
    assert len(base_path) == 1, base_path
    base_path = base_path[0]
    inputs[os.path.relpath(base_path, REPO)] = sha(base_path)
    base = json.load(open(base_path))
    sp = base['storage_per_node']['7']
    efc_c3 = [sp['efc_per_day_per_cohort_year'][f'(0, {i})'] for i in range(3)]
    soh_c3 = [sp['terminal_soh_per_active_cohort_year'][f'(0, {i})'] for i in range(3)]
    q0 = None
    value_c3 = None
    rows = []
    pts = res['points']
    for p in (pts.values() if isinstance(pts, dict) else pts):
        q0 = p['Q0_eur'] if q0 is None else q0
        assert p['Q0_eur'] == q0
        value_c3 = p['value_C3_eur'] if value_c3 is None else value_c3
        rec_path = os.path.join(REPO, p['eval_dir'], 'evaluation_record.json')
        inputs[os.path.relpath(rec_path, REPO)] = sha(rec_path)
        rec = json.load(open(rec_path))
        cells = sorted(rec['ageing_trajectory_terminal']['nodes']['7']['cells'], key=lambda c: c['y'])
        assert [c['y'] for c in cells] == [0, 1, 2]
        rows.append({
            'variant': p['variant'], 'label': p['label'], 'candidate_key': p['candidate_key'],
            'eval_key': p['eval_key'], 'status': p['status'],
            'soh_used': [c['soh_used_for_available_energy'] for c in cells],
            'efc_per_day': [c['efc_per_day'] for c in cells],
            'value_eur': p['value_eur'], 'value_minus_I_eur': p['value_minus_I_eur'],
        })
    ae_c3 = pv(soh_c3)
    efc_c3_pv = pv(efc_c3)
    table = [{'variant': 'C3 (baseline)', 'soh_used': soh_c3, 'efc_per_day': efc_c3, 'AE': ae_c3,
              'AE_over_C3': 1.0, 'value_eur': value_c3, 'value_over_C3': 1.0, 'EFC': efc_c3_pv,
              'EFC_over_C3': 1.0, 'value_per_AE': value_c3 / ae_c3, 'elasticity': None}]
    for r in rows:
        ae = pv(r['soh_used'])
        efc = pv(r['efc_per_day'])
        vr = r['value_eur'] / value_c3
        table.append(dict(r, AE=ae, AE_over_C3=ae / ae_c3, value_over_C3=vr, EFC=efc,
                          EFC_over_C3=efc / efc_c3_pv, value_per_AE=r['value_eur'] / ae,
                          elasticity=(math.log(vr) / math.log(ae / ae_c3)) if abs(ae / ae_c3 - 1) > 1e-9 else None))
    doc = {
        'stage': 'P5.15 Addendum 28 ageing-batch mechanism table (zero solves)',
        'instance': {'candidate': 'node 7, 0.25 MVA / 1.0 MWh, 2025, other nodes zero',
                     'candidate_key': rows[0]['candidate_key']},
        'formulas': {
            'w_y': '1 / 1.02**(y - 2025), y in {2025, 2030, 2035}',
            'AE': 'sum_y w_y * SoH_used[y] / sum_y w_y',
            'EFC': 'sum_y w_y * EFC_per_day[y] / sum_y w_y',
            'value': 'Q(0) - Q_variant(x); Q = certified gross_operational_cost, settlement excluded',
            'value_per_AE': 'value / AE',
            'elasticity': 'ln(value / value_C3) / ln(AE / AE_C3)'},
        'Q0_eur': q0, 'value_C3_eur': value_c3,
        'labelling': 'MODEL VARIANTS - not the baseline; C3 row is the baseline, shown for reference only',
        'table': table,
        'inputs_sha256': inputs,
        'script_sha256': sha(os.path.abspath(__file__)),
    }
    GUARD.uninstall() if hasattr(GUARD, 'uninstall') else None
    failures = GUARD.verify(0) if hasattr(GUARD, 'verify') else []
    doc['solve_profile_guard'] = {'counts': getattr(GUARD, 'counts', lambda: None)() if callable(getattr(GUARD, 'counts', None)) else getattr(GUARD, 'counts', None),
                                  'verify0_failures': failures}
    with open(out_path, 'x') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    for t in table:
        print('%-14s AE %.4f (x%.3f)  value %9.0f (x%.3f)  EFC %.3f (x%.3f)  value/AE %8.0f  elasticity %s'
              % (t['variant'], t['AE'], t['AE_over_C3'], t['value_eur'], t['value_over_C3'], t['EFC'],
                 t['EFC_over_C3'], t['value_per_AE'],
                 'n/a' if t['elasticity'] is None else '%.3f' % t['elasticity']))
    print('guard verify(0) failures:', failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
