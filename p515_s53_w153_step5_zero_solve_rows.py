"""P5.15 Planner task W153 -- THE STEP 5 ZERO-SOLVE ROWS on the settled v6 / extension results.
ZERO SOLVES, NO MODEL LOADS.

Authority: PLANNER_BRIEF_2026-09-13.md -- the Step 5 definition (Roadmap), the Addendum 31 "Step 5 list" (R3.6 rows:
discount -- zero-solve; salvage -- zero-solve add-back; certification statistics for the method section) and the
discount ruling of Addendum 28 ("re-weight the per-representative-year components of Q(x) - Q(0) at the smallest node-7
4 h unit for 0/2/5/8 % and report value-to-cost at each ... plus the average daily price spread against the captured
spread ... The rate stays at 2 % and is not tuned toward a result").

GUARDS. An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import; every imported
module's own zero-permit guard is collected as well; all are verified at exactly 0 at the end. `pickle.load` and
`pickle.loads` are replaced by blocking counters BEFORE any project import, for the whole run, and verified at 0.
Only JSON / JSONL records are read; every one is sha256-verified against the committed manifest that lists it.

WHAT IS COMPUTED (the formulas are also written into the JSON outputs)
  1. DISCOUNT ROW (w153_discount_row.json). The earlier zero-solve computation is W19's Z3/Z4
     (p515_s46_zero_solve_reports.py, commit dd3afa6e; output data/SRP1/Results/P515S46/zero_solve_reports/). Its
     formula is REUSED, not reinvented: `S46._per_year` (called) splits each record's committed per-block component
     levels (component_levels_terminal.json) into per-representative-year gross cost, weighted at 2 % and undiscounted
     (unweighted x num_years x num_days); V_y = Q_y(0) - Q_y(x); V(r) = sum_y V_y_undiscounted / (1 + r)^(y - 2025);
     value_to_cost(r) = V(r) / I. `S46._per_year` reads the year / day counts and the rate from a planning object; a
     plain namespace built from data/SRP1/SRP1.json ("Years", "Days", "DiscountFactor") stands in for it (no model is
     built or loaded) and the production block weight recorded in every block (admm_block_weight) is checked against
     it. Applied to the SETTLED references: x = 0 eval d110bd1a (certified Q181) and the unit n7_4h_e1 eval 3f084f2f
     (certified Q172); both runs ended at their certifying cycle, so their terminal component levels ARE the
     certified-cycle levels (checked). The earlier (pre-settling, Phase A a1a / x0) row is read from the committed Z3
     output and put beside. Price spread: the per-(year, day) prices recorded by the settled runs
     (interface_settlement_detail_s31c.json, price_per_mwh) are checked equal to the prices W19 read from the market
     data, and W19's spread statistics and captured-spread formula are applied with the settled V and the settled
     unit's EFC/day (evaluation_record.json storage_per_node).
  2. SALVAGE ADD-BACK ROW (w153_salvage_addback.json). Every claim in the two final summaries
     (w142_summary_after_34_l_195156fa.json: 48; w142_ext_summary_after_07_pb_y2025_n5_v6.json: item_E 11 + Phase B 1)
     -- READ, not re-scored: gross d_Q and the gross verdict as recorded. Net of salvage: where the record carries a
     net figure (the G rows) it is read; otherwise d_net is the claim's own form on the recorded views' Q_net
     (F: [Q_net(o) + I(o)] - [Q_net(r) + I(r)]; value: [Q_net(r) - Q_net(o)] - [I(o) - I(r)]), and its verdict is
     given by the committed v6 scorer function (`p515_s53_w142_determinacy.resolve_v6`, called) on the recorded
     views -- labelled "computed here (no committed net figure)". The salvage credit of every cell is listed.
  3. CERTIFICATION STATISTICS (w153_certification_statistics.json) over the 34 v6 stage cells, the 7 extension cells
     and d_c52e1670 ("v6 from records": its certificate record + its v5 run's cycle records): per cell and in total.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w153_step5_rows/):
  w153_discount_row.json, w153_salvage_addback.json, w153_certification_statistics.json; launch.log beside;
  manifest_sha256.json written by the separate --manifest invocation (after the run, so it can hash the launch log).
Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w153_step5_rows && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w153_step5_zero_solve_rows.py \\
        > data/SRP1/Results/P515S53/w153_step5_rows/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w153_step5_zero_solve_rows.py --manifest \\
        > data/SRP1/Results/P515S53/w153_step5_rows/manifest_launch.log 2>&1
"""
import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

# ---- the no-model-load guard: pickle.load / pickle.loads raise for the whole run (installed before any project import)
PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W153: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W153: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W153 Step 5 zero-solve rows (never solves)').install()

import gate_result_io as GRIO  # noqa: E402 -- the one JSON writer (W100)
import settling_criterion_v6 as SC6  # noqa: E402 -- TAU, EPS0
import p515_s46_zero_solve_reports as S46  # noqa: E402 -- W19's Z3/Z4 formula (arms its own zero-permit guard)
import p515_s53_w142_determinacy as DET  # noqa: E402 -- the v6 scorer (imports W132's launcher: arms its guards)


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w153_step5_rows', GUARD), ('s46_zero_solve_reports', S46.GUARD)) + tuple(DET.L132.GUARDS))

TAU = SC6.TAU
EPS0 = SC6.EPS0
STAGE = 'P5.15 Planner task W153 -- Step 5 zero-solve rows (discount, salvage add-back, certification statistics)'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w153_step5_rows')
OUT_DISCOUNT = 'w153_discount_row.json'
OUT_SALVAGE = 'w153_salvage_addback.json'
OUT_CERT = 'w153_certification_statistics.json'
OUT_MAN = 'manifest_sha256.json'
LAUNCH_LOGS = ('launch.log', 'manifest_launch.log')

CASE_REL = os.path.join('data', 'SRP1', 'SRP1.json')
W101 = os.path.join(S53, 'w101_srp1_continuation')
X0 = {'name': 'x0 settled (W101 d110bd1a, certified at 181)',
      'campaign_root': os.path.join(W101, 'campaign_s53_w101_srp1_cont_x0'),
      'eval_dir': os.path.join(W101, 'campaign_s53_w101_srp1_cont_x0', 'evals', 'd110bd1a5977df1e_x0'),
      'eval_key_prefix': 'd110bd1a', 'k_star': 181}
UNIT = {'name': 'unit n7_4h_e1 settled (W101 3f084f2f, certified at 172)',
        'campaign_root': os.path.join(W101, 'campaign_s53_w101_srp1_cont_n7_4h_e1'),
        'eval_dir': os.path.join(W101, 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals', '3f084f2ffaeef2b7_n7_4h_e1'),
        'eval_key_prefix': '3f084f2f', 'k_star': 172}
NODE, S_MVA, E_MWH = S46.NODE, S46.S_MVA, S46.E_MWH          # node 7, 0.25 MVA, 1.0 MWh (the smallest node-7 4 h unit)
RATES = S46.RATES                                             # (0.0, 0.02, 0.05, 0.08)
PRODUCTION_RATE = S46.PRODUCTION_RATE                         # 0.02 -- not tuned
S46_OUT = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S46', 'zero_solve_reports', 'zero_solve_reports.json'),
           'manifest': os.path.join('data', 'SRP1', 'Results', 'P515S46', 'zero_solve_reports', 'manifest_sha256.json'),
           'commit': 'dd3afa6e', 'script': 'p515_s46_zero_solve_reports.py'}
V6_SUMMARY = {'path': os.path.join(S53, 'w142_resettle_v6', 'w142_summary_after_34_l_195156fa.json'),
              'manifest': os.path.join(S53, 'w142_resettle_v6', 'w142_summary_after_34_l_195156fa_manifest_sha256.json')}
EXT_SUMMARY = {'path': os.path.join(S53, 'w142_resettle_ext_v6', 'w142_ext_summary_after_07_pb_y2025_n5_v6.json'),
               'manifest': os.path.join(S53, 'w142_resettle_ext_v6',
                                        'w142_ext_summary_after_07_pb_y2025_n5_v6_manifest_sha256.json')}
V6_ROOT = os.path.join(S53, 'w142_resettle_v6')
EXT_ROOT = os.path.join(S53, 'w142_resettle_ext_v6')
V6_STAGE_SPEC = os.path.join(V6_ROOT, 'frozen_s53_resettle_spec_v6_96c23404.json')
EXT_STAGE_SPEC = os.path.join(EXT_ROOT, 'frozen_s53_resettle_ext_spec_v3_84775dc4.json')
NOT_V6_STAGE_CELLS = ('b_2a0ba8b2', 'b_0dd237f0', 'b_4649234b', 'd_c52e1670')   # v4 / v5 / v6-from-records
D_FROM_RECORDS = {'cell': 'd_c52e1670',
                  'certificate': os.path.join(V6_ROOT, 'v6_from_records', 'd_c52e1670_v6_from_records_certificate.json'),
                  'certificate_manifest': os.path.join(V6_ROOT, 'v6_from_records', 'manifest_sha256.json'),
                  'campaign_root': os.path.join(S53, 'w139_resettle_v5', 'campaign_s53_w139_resettle_v5_d_c52e1670')}
RANGE_FLAG = 0.95
ABS_SALVAGE_NEGLIGIBLE_EUR = 1.0   # stated before the run: |salvage| below 1 EUR is reported as nil (records hold ~1e-34)

INPUTS = {}   # rel path -> sha256 (every input read, verified against its manifest)


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _manifest_entry(man, rel):
    """sha256 a manifest records for rel (flat {path: sha} or W19's {'files': {path: {'sha256'}}})."""
    if rel in man and isinstance(man[rel], str):
        return man[rel]
    files = man.get('files') if isinstance(man.get('files'), dict) else {}
    v = files.get(rel)
    if isinstance(v, dict):
        return v.get('sha256')
    return v if isinstance(v, str) else None


_MANIFESTS = {}


def _verified(rel, manifest_rel):
    """Read-side pin: rel's sha256 must equal its entry in the committed manifest manifest_rel."""
    if manifest_rel not in _MANIFESTS:
        with open(os.path.join(REPO, manifest_rel)) as handle:
            _MANIFESTS[manifest_rel] = json.load(handle)
        INPUTS[manifest_rel] = _sha(manifest_rel)
    want = _manifest_entry(_MANIFESTS[manifest_rel], rel)
    got = _sha(rel)
    if want != got:
        raise RuntimeError(f'{rel}: sha256 {got} != manifest {manifest_rel} entry {want}')
    INPUTS[rel] = got
    return rel


def _load(rel, manifest_rel=None):
    if manifest_rel is not None:
        _verified(rel, manifest_rel)
    else:
        INPUTS[rel] = _sha(rel)
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _jsonl(rel, manifest_rel):
    _verified(rel, manifest_rel)
    with open(os.path.join(REPO, rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _dist(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    q = statistics.quantiles(v, n=4, method='inclusive') if len(v) >= 2 else [v[0]] * 3
    return {'n': len(v), 'min': v[0], 'q1': q[0], 'median': statistics.median(v), 'q3': q[2], 'max': v[-1],
            'mean': statistics.fmean(v)}


# ======================================================================================================================
#  1. discount row
# ======================================================================================================================
def _case_namespace():
    case = _load(CASE_REL)
    years = {int(y): int(n) for y, n in case['Years'].items()}
    days = {d: int(n) for d, n in case['Days'].items()}
    tn = SimpleNamespace(years=years, days=days, discount_factor=float(case['DiscountFactor']))
    return case, SimpleNamespace(transmission_network=tn)


def discount_row():
    checks, out = {}, {}
    case, planning = _case_namespace()
    tn = planning.transmission_network
    years = list(tn.years)
    y0 = int(years[0])
    checks['d_case_discount_factor_is_the_production_rate_0_02'] = tn.discount_factor == PRODUCTION_RATE
    checks['d_case_num_years_equal_across_blocks'] = len(set(tn.years.values())) == 1

    recs, lv, dec = {}, {}, {}
    for tag, ref in (('x0', X0), ('unit', UNIT)):
        man = os.path.join(ref['campaign_root'], 'campaign_manifest_sha256.json')
        recs[tag] = _load(os.path.join(ref['eval_dir'], 'evaluation_record.json'), man)
        lv[tag] = _load(os.path.join(ref['eval_dir'], 'component_levels_terminal.json'), man)
        dec[tag] = _load(os.path.join(ref['eval_dir'], 'settling_decision.json'), man)
        checks[f'd_{tag}_eval_key_prefix'] = recs[tag]['eval_key'].startswith(ref['eval_key_prefix'])
        checks[f'd_{tag}_settling_certified_at_k_star'] = (dec[tag]['status'] == 'certified'
                                                          and dec[tag]['k_star'] == ref['k_star'])
        checks[f'd_{tag}_run_ended_at_k_star'] = recs[tag]['cycles_run'] == ref['k_star']
        checks[f'd_{tag}_component_levels_at_k_star'] = lv[tag]['cycles_run'] == ref['k_star']
        checks[f'd_{tag}_certified_cost_is_Q_k_star'] = recs[tag]['certified_cost'] == dec[tag]['Q_k_star']
        checks[f'd_{tag}_component_levels_gross_is_certified_cost'] = (
            lv[tag]['recourse_components']['gross_operational_cost'] == recs[tag]['certified_cost'])

    # instance identity -- recomputed with the production canonicaliser / key (the W19 precedent)
    H = S46.H
    unit_canon = H.canonical_candidate({n: ((S_MVA, E_MWH) if n == NODE else (0.0, 0.0)) for n in H.ACTIVE_NODES})
    x0_canon = H.canonical_candidate({n: (0.0, 0.0) for n in H.ACTIVE_NODES})
    unit_key, x0_key = H.candidate_key(unit_canon), H.candidate_key(x0_canon)
    checks['d_unit_candidate_key_recomputed'] = recs['unit']['candidate_key'] == unit_key == S46.EXPECTED_KEY
    checks['d_x0_candidate_key_recomputed'] = recs['x0']['candidate_key'] == x0_key
    out['instance'] = {
        'unit': {'candidate_canonical': unit_canon, 'candidate_key': unit_key, 'eval_key': recs['unit']['eval_key'],
                 'campaign_id': recs['unit']['campaign_id'], 'eval_dir': UNIT['eval_dir'], 'k_star': UNIT['k_star'],
                 'certified_cost_gross_eur': recs['unit']['certified_cost'], 'name': UNIT['name']},
        'x0': {'candidate_canonical': x0_canon, 'candidate_key': x0_key, 'eval_key': recs['x0']['eval_key'],
               'campaign_id': recs['x0']['campaign_id'], 'eval_dir': X0['eval_dir'], 'k_star': X0['k_star'],
               'certified_cost_gross_eur': recs['x0']['certified_cost'], 'name': X0['name']}}

    # W19's per-year split (CALLED)
    q = {}
    for tag in ('x0', 'unit'):
        per_year, total_w, w_dev = S46._per_year(lv[tag], planning)
        gross = lv[tag]['recourse_components']['gross_operational_cost']
        q[tag] = {'per_year': per_year, 'sum_blocks_weighted_r2': total_w, 'recorded_gross_operational_cost': gross,
                  'abs_sum_minus_recorded': abs(total_w - gross), 'max_block_weight_deviation': w_dev,
                  'n_blocks': len(lv[tag]['blocks'])}
        checks[f'd_{tag}_blocks_reconcile_to_gross_within_0.01_eur'] = abs(total_w - gross) < 0.01
        checks[f'd_{tag}_recorded_block_weights_match_case_formula'] = w_dev < 1e-9
        checks[f'd_{tag}_48_blocks'] = len(lv[tag]['blocks']) == 48
        checks[f'd_{tag}_undiscounted_times_factor_equals_weighted'] = all(
            abs(v['undiscounted'] / ((1 + PRODUCTION_RATE) ** (int(y) - y0)) - v['weighted_r2'])
            < 1e-6 * abs(v['weighted_r2']) for y, v in per_year.items())
    out['Q_per_year'] = q
    v_year = {str(y): {'V_weighted_r2': q['x0']['per_year'][str(y)]['weighted_r2'] - q['unit']['per_year'][str(y)]['weighted_r2'],
                       'V_undiscounted': q['x0']['per_year'][str(y)]['undiscounted'] - q['unit']['per_year'][str(y)]['undiscounted']}
              for y in years}
    out['V_per_year'] = v_year
    v_cert = recs['x0']['certified_cost'] - recs['unit']['certified_cost']
    v_r2 = sum(v['V_weighted_r2'] for v in v_year.values())
    v_r2_re = sum(v['V_undiscounted'] / (1 + PRODUCTION_RATE) ** (int(y) - y0) for y, v in v_year.items())
    out['reconstruction'] = {'certified_value_eur': v_cert,
                             'certified_value_definition': 'Q181(x0 d110bd1a) - Q172(unit 3f084f2f), gross, settlement excluded',
                             'V_r2_from_weighted_blocks': v_r2, 'V_r2_from_undiscounted_reweighted': v_r2_re,
                             'abs_diff_weighted': abs(v_r2 - v_cert), 'abs_diff_reweighted': abs(v_r2_re - v_cert),
                             'tolerance_eur': S46.Z3_RECONSTRUCTION_TOL_EUR}
    checks['d_r2_reconstruction_within_tolerance'] = (abs(v_r2 - v_cert) <= S46.Z3_RECONSTRUCTION_TOL_EUR
                                                      and abs(v_r2_re - v_cert) <= S46.Z3_RECONSTRUCTION_TOL_EUR)

    # I(x) -- the committed unit cost (ext summary item_E.I_unit), cross-checked
    ext = _load(EXT_SUMMARY['path'], EXT_SUMMARY['manifest'])
    v6 = _load(V6_SUMMARY['path'], V6_SUMMARY['manifest'])
    s46 = _load(S46_OUT['path'], S46_OUT['manifest'])
    i_x = ext['item_E']['I_unit']
    head = [c for c in v6['claims_scored'] if c['claim_id'] == 'CHECK:headline_V_minus_I_settled']
    checks['d_I_equals_W19_I_x'] = i_x == s46['Z3']['I_x_eur']
    checks['d_I_equals_headline_claim_I_other'] = len(head) == 1 and head[0]['I_other'] == i_x
    checks['d_ext_Q0_is_x0_Q181'] = ext['item_E']['Q0'] == recs['x0']['certified_cost']
    out['I_x_eur'] = i_x
    out['I_x_source'] = (f"{EXT_SUMMARY['path']} item_E.I_unit (paid in 2025, factor 1.0, held fixed); equal to W19's "
                         f"Z3 I_x_eur and to the v6 summary's CHECK:headline_V_minus_I_settled I_other")

    # resolution: (i) W19's formula on the settled records; (ii) the Addendum 61 determinacy threshold
    bar_sum = recs['unit']['bar']['value'] + recs['x0']['bar']['value']
    bands = {'x0': dec['x0']['band_width'], 'unit': dec['unit']['band_width']}
    a61_thr = max(SC6.DETERMINACY_BAR_FACTOR * max(bands.values()), SC6.DETERMINACY_TAU_MULTIPLE * TAU)
    rows = []
    for r in RATES:
        per = {y: v['V_undiscounted'] / (1 + r) ** (int(y) - y0) for y, v in v_year.items()}
        val = sum(per.values())
        scale = max(((1 + PRODUCTION_RATE) / (1 + r)) ** (int(y) - y0) for y in v_year)
        thr = a61_thr * scale
        rows.append({'rate': r, 'V_per_year': per, 'V': val, 'I': i_x, 'value_to_cost': val / i_x,
                     'value_minus_I': val - i_x,
                     'resolution_w19_bar_sum_conservative': bar_sum * scale,
                     'a61_threshold_conservative': thr,
                     'value_minus_I_over_a61_threshold': (val - i_x) / thr,
                     'value_minus_I_verdict_a61': 'determinate' if abs(val - i_x) >= thr else 'within resolution',
                     'reweighting_scale_factor_max_over_years': scale})
    v2 = [x for x in rows if x['rate'] == PRODUCTION_RATE][0]['V']
    for x in rows:
        x['V_over_V_2pct'] = x['V'] / v2
        x['change_vs_2pct_pct'] = 100.0 * (x['V'] / v2 - 1.0)
    # the headline claim is value-form: d = [Q(x0) - Q(unit)] - [I(unit) - 0] = value - I at 2 %
    checks['d_headline_claim_d_Q_equals_value_minus_I_at_2pct'] = (
        len(head) == 1 and abs(head[0]['d_Q'] - (v2 - i_x)) <= S46.Z3_RECONSTRUCTION_TOL_EUR)
    out['table'] = rows
    out['resolution_note'] = (
        '(i) W19 formula: bar(x) + bar(x0), each = max |gross step| over the last 10 cycles of the record '
        f"({recs['unit']['bar']['value']:.2f} + {recs['x0']['bar']['value']:.2f}), scaled by max_y ((1.02/(1+r))^(y-2025)); "
        '(ii) the Addendum 61 rule for a difference of two certified cells, max(3 x the larger band width, 2 TAU) '
        f"(bands x0 {bands['x0']:.2f}, unit {bands['unit']:.2f}; TAU {TAU:.2f}), with the same conservative scaling; "
        'the per-year split of either bar is not recorded, so the scaling bounds it.')
    out['bands'] = bands
    out['bar_sum_w19_formula'] = bar_sum
    out['a61_threshold_at_2pct'] = a61_thr

    # the earlier (pre-settling) row beside
    z3 = s46['Z3']
    earlier = []
    for x in z3['table']:
        mine = [m for m in rows if m['rate'] == x['rate']][0]
        earlier.append({'rate': x['rate'], 'V_earlier': x['V'], 'value_to_cost_earlier': x['value_to_cost'],
                        'value_minus_I_earlier': x['V'] - z3['I_x_eur'], 'change_vs_2pct_pct_earlier': x['change_vs_2pct_pct'],
                        'V_per_year_earlier': x['V_per_year'],
                        'V_settled': mine['V'], 'value_to_cost_settled': mine['value_to_cost'],
                        'value_minus_I_settled': mine['value_minus_I'],
                        'V_settled_minus_earlier': mine['V'] - x['V']})
    out['earlier_row'] = {
        'source': f"{S46_OUT['path']} Z3 (W19, commit {S46_OUT['commit']}, script {S46_OUT['script']})",
        'instance_earlier': ('Phase A, certified under the frozen oracle: a1a n7_4h_e1 eval 7eb1ce62 (Q '
                             f"{s46['instance']['records']['a1a']['certified_cost_gross_eur']!r}) and x0 eval 7aa017f0 "
                             f"(Q {s46['instance']['records']['x0']['certified_cost_gross_eur']!r}); minimum SoH 0.50 era"),
        'candidate_key_same_as_settled_unit': s46['instance']['candidate_key'] == unit_key,
        'configuration_differences': {
            'note': ('the two rows differ in the ageing configuration as well as in settling, so their difference is NOT a '
                     'settling effect alone'),
            'earlier_ageing_as_consumed_w19_Z2': {'k_cl_eff': s46['Z2']['k_cl_eff_as_consumed'],
                                                  'phi_cal': s46['Z2']['phi_cal_as_consumed'],
                                                  'soh_min': s46['Z2']['soh_min']},
            'settled_ageing_baseline_label': recs['unit'].get('ess_ageing_baseline_label'),
            'settled_ageing_baseline': recs['unit'].get('ess_ageing_baseline'),
            'earlier_trajectory_w19_Z2': [{'year_block': t['year_block'], 'efc_per_day': t['efc_per_day'],
                                           'soh_end_of_block': t['soh_used_for_available_energy_end_of_block'],
                                           'floor_row_active': t['floor_row_active']} for t in s46['Z2']['trajectory']]},
        'V_per_year_earlier': z3['V_per_year'], 'rows': earlier}

    # ageing state of the settled unit (the context of the per-year split)
    st = recs['unit']['storage_per_node'][str(NODE)]
    traj = []
    efc_total = 0.0
    efc_disc = 0.0
    for yi, y in enumerate(years):
        key = f'(0, {yi})'
        efc = st['efc_per_day_per_cohort_year'][key]
        blk_cycles = 365.0 * tn.years[y] * efc
        efc_total += blk_cycles
        efc_disc += blk_cycles / (1 + PRODUCTION_RATE) ** (int(y) - y0)
        traj.append({'year_block': y, 'efc_per_day': efc, 'EFC_block_cycles': blk_cycles,
                     'soh_end_of_block': st['terminal_soh_per_active_cohort_year'][key],
                     'floor_row_active': key in st['soh_floor_rows_active_at_terminal'],
                     'e_available_mwh': st['published_available_capacity_terminal'][str(y)]['e_available']})
    checks['d_days_sum_is_365'] = sum(tn.days.values()) == 365
    out['settled_unit_ageing'] = {'per_year': traj, 'EFC_horizon_cycles': efc_total,
                                  'EFC_horizon_discounted_2pct_cycles': efc_disc,
                                  'source': f"{UNIT['eval_dir']}/evaluation_record.json storage_per_node['7']"}

    # price spread: prices recorded by the settled runs, checked against W19's market prices
    z4 = s46['Z4']
    w19_prices = {(int(r['year']), r['day']): r['prices'] for r in z4['per_representative_day']}
    max_dev = 0.0
    rec_prices = {}
    for tag, ref in (('unit', UNIT), ('x0', X0)):
        man = os.path.join(ref['campaign_root'], 'campaign_manifest_sha256.json')
        det = _load(os.path.join(ref['eval_dir'], 'interface_settlement_detail_s31c.json'), man)['interface_reporting_detail']
        for node, dd in det.items():
            for y, days in dd.items():
                for day, blk in days.items():
                    arr = [blk['periods'][str(p)]['price_per_mwh'] for p in range(len(blk['periods']))]
                    rec_prices.setdefault((int(y), day), arr)
                    max_dev = max(max_dev, max(abs(a - b) for a, b in zip(arr, w19_prices[(int(y), day)])))
    checks['d_recorded_prices_equal_w19_market_prices_within_1e-9'] = max_dev < 1e-9 and set(rec_prices) == set(w19_prices)
    h = S46.DURATION_H
    prow = []
    for (y, day), arr in sorted(rec_prices.items()):
        srt = sorted(arr)
        prow.append({'year': y, 'day': day, 'weight_undiscounted': tn.years[y] * tn.days[day],
                     'weight_model_r2': tn.years[y] * tn.days[day] / (1 + PRODUCTION_RATE) ** (y - y0),
                     'max_minus_min': srt[-1] - srt[0],
                     f'top{h}_mean_minus_bottom{h}_mean': sum(srt[-h:]) / h - sum(srt[:h]) / h,
                     'mean': sum(arr) / len(arr)})

    def wavg(field, wkey, subset=None):
        rs = [r for r in prow if subset is None or subset(r)]
        return sum(r[field] * r[wkey] for r in rs) / sum(r[wkey] for r in rs)
    f_mm, f_th = 'max_minus_min', f'top{h}_mean_minus_bottom{h}_mean'
    spread = {'all_years_day_weighted_undiscounted': {f_mm: wavg(f_mm, 'weight_undiscounted'),
                                                      f_th: wavg(f_th, 'weight_undiscounted'),
                                                      'mean_price': wavg('mean', 'weight_undiscounted')},
              'all_years_model_block_weight_r2': {f_mm: wavg(f_mm, 'weight_model_r2'), f_th: wavg(f_th, 'weight_model_r2'),
                                                  'mean_price': wavg('mean', 'weight_model_r2')},
              'per_year_day_weighted': {str(y): {f_mm: wavg(f_mm, 'weight_undiscounted', lambda r, y=y: r['year'] == y),
                                                 f_th: wavg(f_th, 'weight_undiscounted', lambda r, y=y: r['year'] == y)}
                                        for y in years}}
    checks['d_spread_summary_reproduces_w19'] = all(
        abs(spread['all_years_day_weighted_undiscounted'][k] - z4['spread_summary']['all_years_day_weighted_undiscounted'][k]) < 1e-9
        for k in (f_mm, f_th, 'mean_price'))
    v0 = [x for x in rows if x['rate'] == 0.0][0]['V']
    captured = {'a_literal_V2pct_over_undiscounted_EFC': (v2 / E_MWH) / efc_total,
                'b_consistent_undiscounted_V0_over_undiscounted_EFC': (v0 / E_MWH) / efc_total,
                'c_consistent_discounted_V2pct_over_discounted_EFC': (v2 / E_MWH) / efc_disc,
                'per_year_undiscounted': {str(t['year_block']): (v_year[str(t['year_block'])]['V_undiscounted'] / E_MWH)
                                          / t['EFC_block_cycles'] for t in traj}}
    cs_old = z4['captured_spread']
    out['price_spread'] = {
        'formula_spread': z4['spread_formula'], 'formula_captured': cs_old['formula'],
        'prices_source': ('per-(year, day) price_per_mwh recorded by the settled runs (interface_settlement_detail_s31c.json, '
                          'every node), checked equal to the market prices W19 read through production'),
        'max_abs_dev_recorded_vs_w19': max_dev,
        'market_spread': spread,
        'captured_settled': captured,
        'captured_earlier_w19': {k: cs_old[k] for k in ('a_literal_V2pct_over_undiscounted_EFC',
                                                        'b_consistent_undiscounted_V0_over_undiscounted_EFC',
                                                        'c_consistent_discounted_V2pct_over_discounted_EFC')},
        'captured_earlier_w19_per_year': {y: v['captured_eur_per_mwh_cycle'] for y, v in cs_old['per_year_undiscounted'].items()},
        'expert_expectation': S46.EXPERT_PREDICTION_Z4 + ' (Addendum 28)',
        'captured_over_top4_spread_settled_c': captured['c_consistent_discounted_V2pct_over_discounted_EFC']
        / spread['all_years_day_weighted_undiscounted'][f_th]}
    out['formula'] = (
        "W19's Z3 formula, reused: Q_y = sum over the TSO block and the three DSO blocks of every day of year y of the gross "
        'settlement-excluded per-block cost (sum of ' + ' + '.join(S46.GROSS_COMPONENTS) + ', as recorded in '
        'component_levels_terminal.json); undiscounted Q_y = sum of unweighted * num_years * num_days; V_y = Q_y(0) - Q_y(x); '
        'V(r) = sum_y V_y_undiscounted / (1 + r)^(y - 2025); value_to_cost(r) = V(r) / I(x); value - I = V(r) - I(x). '
        'Production applies ONE discount factor per representative year to all 5 years of the block (W19 Z3 '
        'production_convention); I(x) is paid in 2025 and not discounted.')
    out['rate_statement'] = 'The production rate stays at 2 % and is not tuned (Addendum 28).'
    out['objective_convention'] = 'Q = gross_operational_cost, settlement EXCLUDED; V = Q(0) - Q(x); EUR'
    return checks, out


# ======================================================================================================================
#  2. salvage add-back row
# ======================================================================================================================
VIEW_MATCH_KEYS = ('status', 'Q', 't', 'salvage', 'band')


def _form_d(form, ir, io, qr, qo):
    """The claim's own form (W132 `score_claim`'s d): F -> (qo + Io) - (qr + Ir); value -> (qr - qo) - (Io - Ir)."""
    if form == 'F':
        return (qo + io) - (qr + ir)
    return (qr - qo) - (io - ir)


def salvage_row():
    checks, out = {}, {}
    v6 = _load(V6_SUMMARY['path'], V6_SUMMARY['manifest'])
    ext = _load(EXT_SUMMARY['path'], EXT_SUMMARY['manifest'])
    views = {}
    for src, doc in (('v6', v6), ('ext', ext)):
        for name, r in doc['reports'].items():
            views[f'{src}:{name}'] = r['view']
        for name, r in doc['references'].items():
            views[f'{src}:ref:{name}'] = dict(r)
    claims = ([('v6', c) for c in v6['claims_scored']] + [('ext', c) for c in ext['item_E']['claims'].values()]
              + [('ext', c) for c in ext['phase_b_claim']])
    checks['s_claim_count_48_plus_12'] = (len(v6['claims_scored']) == 48
                                          and len(ext['item_E']['claims']) + len(ext['phase_b_claim']) == 12)
    checks['s_every_completed_claim_present'] = (
        sorted(c['claim_id'] for s, c in claims if s == 'v6') == sorted(v6['claims_complete'])
        and sorted(c['claim_id'] for s, c in claims if s == 'ext') == sorted(ext['claims_complete']))

    def match(v):
        return sorted({k.split(':', 1)[1] for k, w in views.items() if all(w.get(f) == v.get(f) for f in VIEW_MATCH_KEYS)})
    rows, d_recompute_dev, net_x = [], 0.0, []
    for src, c in claims:
        ref, oth = c['ref'], c['other']
        d_q_re = _form_d(c['form'], c['I_ref'], c['I_other'], ref['Q'], oth['Q'])
        d_recompute_dev = max(d_recompute_dev, abs(d_q_re - c['d_Q']))
        d_n = _form_d(c['form'], c['I_ref'], c['I_other'], ref['Q_net'], oth['Q_net'])
        d_ncc = _form_d(c['form'], c['I_ref'], c['I_other'], ref['Q_net'] + ref['t'], oth['Q_net'] + oth['t'])
        res = DET.resolve_v6(d_n, d_ncc, (ref, oth))
        rec_net = c['net_of_salvage'] if isinstance(c['net_of_salvage'], dict) else None
        if rec_net is not None:
            net_x.append({'claim_id': c['claim_id'], 'd_net_equal': d_n == rec_net['d_net'],
                          'verdict_equal': res['verdict'] == rec_net['resolution']['verdict']})
            net_d, net_v, net_src = rec_net['d_net'], rec_net['resolution']['verdict'], 'recorded (claim record net_of_salvage)'
            net_thr = rec_net['resolution'].get('threshold', rec_net['resolution'].get('bar'))
        else:
            net_d, net_v, net_src = d_n, res['verdict'], 'computed here (no committed net figure): DET.resolve_v6 on the recorded views'
            net_thr = res.get('threshold', res.get('bar'))
        g = c['gross']
        rows.append({
            'summary': src, 'claim_id': c['claim_id'], 'statement': c['statement'], 'form': c['form'],
            'claim_type': c['claim_type'], 'primary_convention': c['objective_convention_primary'],
            'ref_cells': match(ref), 'other_cells': match(oth),
            'ref_status': ref['status'], 'other_status': oth['status'],
            'Q_ref': ref['Q'], 'Q_other': oth['Q'], 'Q_net_ref': ref['Q_net'], 'Q_net_other': oth['Q_net'],
            'salvage_ref': ref['salvage'], 'salvage_other': oth['salvage'], 'I_ref': c['I_ref'], 'I_other': c['I_other'],
            'd_gross': c['d_Q'], 'gross_rule': g['rule'], 'gross_threshold': g.get('threshold', g.get('bar')),
            'gross_verdict': g['verdict'],
            'd_net': net_d, 'net_threshold': net_thr, 'net_verdict': net_v, 'net_source': net_src,
            'salvage_effect_d_net_minus_d_gross': net_d - c['d_Q'],
            'primary_verdict_recorded': c['verdict'],
            'sign_gross_positive': c['d_Q'] > 0, 'sign_net_positive': net_d > 0,
            'verdict_differs_between_conventions': g['verdict'] != net_v,
            'sign_differs_between_conventions': (c['d_Q'] > 0) != (net_d > 0),
            'salvage_nil_both_cells': (abs(ref['salvage']) < ABS_SALVAGE_NEGLIGIBLE_EUR
                                       and abs(oth['salvage']) < ABS_SALVAGE_NEGLIGIBLE_EUR)})
    checks['s_d_gross_recomputed_from_form_equals_record_within_1e-6'] = d_recompute_dev < 1e-6
    checks['s_recorded_net_figures_reproduced_by_the_form_and_scorer'] = (
        bool(net_x) and all(x['d_net_equal'] and x['verdict_equal'] for x in net_x))
    checks['s_every_claim_view_matched_to_a_cell'] = all(r['ref_cells'] and r['other_cells'] for r in rows)
    out['claims'] = rows
    out['recorded_net_crosscheck'] = net_x
    cells = []
    for k, v in views.items():
        cells.append({'cell': k.split(':', 1)[1], 'summary': k.split(':', 1)[0], 'status': v.get('status'),
                      'Q_gross': v.get('Q'), 'Q_net': v.get('Q_net'), 'salvage_credit': v.get('salvage'),
                      'salvage_nil': v.get('salvage') is not None and abs(v['salvage']) < ABS_SALVAGE_NEGLIGIBLE_EUR,
                      'gross_minus_net_minus_salvage': (v['Q'] - v['Q_net'] - v['salvage'])
                      if None not in (v.get('Q'), v.get('Q_net'), v.get('salvage')) else None})
    checks['s_every_cell_gross_minus_net_equals_salvage_within_1e-6'] = all(
        c['gross_minus_net_minus_salvage'] is None or abs(c['gross_minus_net_minus_salvage']) < 1e-6 for c in cells)
    out['cells'] = cells
    out['totals'] = {
        'n_claims': len(rows),
        'n_verdict_differs': sum(r['verdict_differs_between_conventions'] for r in rows),
        'claims_verdict_differs': [r['claim_id'] for r in rows if r['verdict_differs_between_conventions']],
        'n_sign_differs': sum(r['sign_differs_between_conventions'] for r in rows),
        'n_claims_with_salvage_on_either_cell': sum(not r['salvage_nil_both_cells'] for r in rows),
        'n_cells_listed': len(cells),
        'cells_with_salvage': sorted({c['cell'] for c in cells if not c['salvage_nil']}),
        'net_source_counts': {s: sum(r['net_source'] == s for r in rows) for s in sorted({r['net_source'] for r in rows})}}
    out['definitions'] = {
        'gross': 'Q = gross_operational_cost, settlement EXCLUDED (the primary convention, Addendum 58 Ruling 3)',
        'net': 'Q_net = net_operational_recourse = Q - terminal salvage credit (the record\'s own field)',
        'd_forms': ("F: d = [Q(o) + I(o)] - [Q(r) + I(r)]; value: d = [Q(r) - Q(o)] - [I(o) - I(r)]; net: the same with "
                    "Q_net (W132 score_claim)"),
        'verdict_rule': DET.RULE_TEXT,
        'salvage_nil_threshold_eur': ABS_SALVAGE_NEGLIGIBLE_EUR,
        'not_rescored': ('gross d and gross verdict are READ from the claim records; a net verdict is read where the record '
                         'has one (G rows) and computed with the committed scorer function otherwise')}
    return checks, out


# ======================================================================================================================
#  3. certification statistics
# ======================================================================================================================
def _family(block):
    return block.split('|', 1)[0]


def _cell_stats(cell, stage, dec, rows, cres, report, mode):
    checks = {}
    by_c = {r['cycle']: r for r in rows}
    status = dec['status']
    k_star = dec.get('k_star')
    N, k0 = dec['N'], dec['k0']
    end = k_star if status == 'certified' else dec.get('k_cap', dec.get('cap'))
    last = rows[-1]['cycle']
    checks['cycles_contiguous_from_1'] = [r['cycle'] for r in rows] == list(range(1, len(rows) + 1))
    if mode == 'run':
        checks['run_ended_at_end_cycle'] = last == end
    nc_all = sorted(r['cycle'] for r in rows if r['cycle'] <= end and r['all_clean_k'] is False)
    checks['decision_non_clean_cycles_equal_cycle_records_through_end'] = sorted(dec['non_clean_cycles']) == nc_all
    after = [r for r in rows if N < r['cycle'] <= end]
    nc_after = [{'cycle': r['cycle'], 'blocks': r['non_clean_blocks']} for r in after if r['all_clean_k'] is False]
    fam = {'TSO': 0, 'DSO': 0, 'ESSO': 0}
    for e in nc_after:
        for b in e['blocks']:
            fam[_family(b)] = fam.get(_family(b), 0) + 1
    no_after = [r['cycle'] for r in after if r['all_optimal_k'] is False]
    acc_all = [{'cycle': r['cycle'], 'blocks': r['acceptable_clean_blocks']} for r in rows
               if r['cycle'] <= end and r.get('acceptable_clean_blocks')]
    acc_after = [e for e in acc_all if e['cycle'] > N]
    acc_fam = {}
    for e in acc_after:
        for b in e['blocks']:
            acc_fam[_family(b)] = acc_fam.get(_family(b), 0) + 1
    # terminal step
    q_end, q_prev = by_c[end]['gross'], by_c[end - 1]['gross']
    step = abs(q_end - q_prev)
    rep_ratio = report.get('terminal_step_over_EPS0', report.get('terminal_step_over_EPS0_at_k_star'))
    checks['terminal_step_over_EPS0_equals_summary'] = rep_ratio is not None and abs(step / EPS0 - rep_ratio) < 1e-9
    # turning points at the decision; the certificate's reads
    T = dec.get('T') or []
    tp = [t[0] for t in T]
    nc_set = set(nc_all)
    tp_nc = [t for t in tp if t in nc_set]
    tp_adj = [t for t in tp if (t in nc_set or t - 1 in nc_set or t + 1 in nc_set)]
    last3 = tp[-3:]
    branch = dec.get('branch')
    cert_tp_applies = status == 'certified' and branch == 'oscillatory'
    certA = dec.get('certA_parts') or ((by_c[end].get('settling') or {}).get('certA_parts') or {})
    oow = dec.get('out_of_window_reads') or {}
    lapses = dec.get('lapse_events') or []
    gaps = dec.get('gap_refusals') or []
    reasons = dec.get('reasons') or []
    if status == 'certified':
        cause, contributing = None, []
    else:
        contributing = ([f'gap clause ({len(gaps)} refusals; refused at cap {dec.get("gap_clause_refused_at_cap")})'] if gaps else []) \
            + ([f'lapse reset ({len(lapses)}: cycles {[e["cycle"] for e in lapses]})'] if lapses else []) \
            + (['growth test (swings_growing)'] if 'swings_growing' in reasons else []) \
            + ([f'vetoes ({dec.get("n_vetoes")})'] if dec.get('n_vetoes') else []) \
            + [f'reasons at cap: {reasons}']
        cause = ('gap clause' if gaps else 'lapse reset' if lapses else 'growth test' if 'swings_growing' in reasons
                 else 'other')
    durs = [r['t_end_s'] - r['t_start_s'] for r in rows if r.get('t_end_s') is not None and r.get('t_start_s') is not None]
    wall_s = cres['parent_view']['wall_s']
    # summary agreement
    checks['status_equals_summary'] = report.get('status') == status
    checks['k_star_equals_summary'] = report.get('k_star') == k_star
    if status == 'certified':
        checks['range_over_tau_equals_summary'] = report.get('range_over_tau') == dec.get('range_over_tau')
    checks['n_vetoes_equals_summary'] = (report.get('n_vetoes', 0) or 0) == (dec.get('n_vetoes', 0) or 0)
    row = {
        'cell': cell, 'stage': stage, 'mode': mode, 'item': report.get('item'), 'gated': report.get('gated'),
        'eval_key': report.get('eval_key'), 'candidate_key': report.get('candidate_key'),
        'status': status, 'branch': branch, 'cause': cause, 'contributing': contributing,
        'k0_first_residual_pass_N': N, 'k0_at_decision': k0, 'k_star': k_star, 'cap': dec.get('cap'),
        'end_cycle': end, 'cycles_recorded': last,
        'k_star_minus_k0': (k_star - k0) if k_star is not None else None,
        'k_star_minus_N': (k_star - N) if k_star is not None else None,
        'end_minus_N': end - N,
        'range_over_tau': dec.get('range_over_tau') if status == 'certified' else None,
        'range_over_tau_ge_0_95': (dec.get('range_over_tau') >= RANGE_FLAG) if status == 'certified' else None,
        'band_width': dec.get('band_width'), 'W': dec.get('W'), 'P_hat': dec.get('P_hat'),
        'terminal_step_abs': step, 'terminal_step_over_EPS0': step / EPS0,
        'n_vetoes': dec.get('n_vetoes') or 0,
        'veto_cycles': [v['cycle'] for v in (dec.get('vetoes') or [])],
        'turning_point_floor_rejections': len(dec.get('turning_point_floor_rejections') or []),
        'turning_point_floor_rejection_cycles': [x['cycle'] for x in (dec.get('turning_point_floor_rejections') or [])],
        'growth_floor_swings_excluded_at_end': certA.get('swings_excluded_below_floor'),
        'lapse_events': lapses, 'n_gap_refusals': len(gaps), 'gap_clause_refused_at_cap': dec.get('gap_clause_refused_at_cap'),
        'reasons_at_cap': reasons,
        'non_clean_cycles_after_N': [e['cycle'] for e in nc_after],
        'non_clean_after_N_detail': nc_after,
        'non_clean_block_events_after_N_by_family': fam,
        'non_optimal_cycles_after_N': no_after,
        'acceptable_clean_block_events_all': sum(len(e['blocks']) for e in acc_all),
        'acceptable_clean_block_events_after_N': sum(len(e['blocks']) for e in acc_after),
        'acceptable_clean_after_N_by_family': acc_fam,
        'acceptable_clean_after_N_detail': acc_after,
        'turning_points': tp, 'turning_points_last_three': last3,
        'turning_points_at_non_clean_cycle': tp_nc,
        'turning_points_within_1_of_non_clean_cycle': tp_adj,
        'certificate_reads_turning_points': cert_tp_applies,
        'certificate_turning_point_at_non_clean': bool(tp_nc) if cert_tp_applies else None,
        'certificate_last_three_turning_points_at_non_clean': bool([t for t in last3 if t in nc_set]) if cert_tp_applies else None,
        'certificate_turning_point_within_1_of_non_clean': bool(tp_adj) if cert_tp_applies else None,
        'out_of_window_non_clean_read': oow.get('any_non_clean_read_outside_window'),
        'out_of_window_non_clean_cycles': oow.get('non_clean_cycles_read_outside_window'),
        'wall_s_parent': wall_s, 'wall_s_per_cycle': wall_s / last,
        'child_cycle_duration_s': _dist(durs)}
    return checks, row


def certification_statistics():
    checks, out = {}, {}
    v6 = _load(V6_SUMMARY['path'], V6_SUMMARY['manifest'])
    ext = _load(EXT_SUMMARY['path'], EXT_SUMMARY['manifest'])
    v6_cells = [c for c in v6['reports'] if c not in NOT_V6_STAGE_CELLS]
    ext_cells = list(ext['reports'])
    checks['c_34_v6_stage_cells'] = len(v6_cells) == 34
    checks['c_7_extension_cells'] = len(ext_cells) == 7
    rows = []
    cell_checks = {}
    plan = [(c, 'v6 stage spec 96c23404', os.path.join(V6_ROOT, f'campaign_s53_w142_resettle_v6_{c}'), v6['reports'][c])
            for c in v6_cells] + \
           [(c, 'ext spec v3 84775dc4', os.path.join(EXT_ROOT, f'campaign_s53_w142_resettle_ext_v6_{c}'), ext['reports'][c])
            for c in ext_cells]
    for cell, stage, root, report in plan:
        man = os.path.join(root, 'campaign_manifest_sha256.json')
        evs = [d for d in os.listdir(os.path.join(REPO, root, 'evals')) if d.endswith('_' + cell)]
        if len(evs) != 1:
            raise RuntimeError(f'{cell}: expected one eval dir, found {evs}')
        ev = os.path.join(root, 'evals', evs[0])
        dec = _load(os.path.join(ev, 'resettle_decision.json'), man)
        crs = _jsonl(os.path.join(ev, 'resettle_cycle_record.jsonl'), man)
        cres = _load(os.path.join(root, 'campaign_results.json'), man)
        ck, row = _cell_stats(cell, stage, dec, crs, cres, report, 'run')
        row['eval_dir'] = ev
        cell_checks[cell] = ck
        rows.append(row)
    # d_c52e1670 -- v6 from records
    d = D_FROM_RECORDS
    cert = _load(d['certificate'], d['certificate_manifest'])
    src = cert['source_run']
    man = os.path.join(d['campaign_root'], 'campaign_manifest_sha256.json')
    crs = _jsonl(os.path.join(src['eval_dir'], 'resettle_cycle_record.jsonl'), man)
    cres = _load(os.path.join(d['campaign_root'], 'campaign_results.json'), man)
    dec = dict(cert['decision'])
    ck, row = _cell_stats(d['cell'], 'v6 from records (certificate; source v5 run 51280961, stage spec ab32ffc9)',
                          dec, crs, cres, v6['reports'][d['cell']], 'from_records')
    row['eval_dir'] = src['eval_dir']
    row['note'] = (f"the v5 run continued to {src['cycles_recorded']} cycles; every per-cell statistic is read through the "
                   'certifying cycle k* = 150 except the wall time, which is the v5 run\'s own (all cycles)')
    cell_checks[d['cell']] = ck
    rows.append(row)
    checks['c_42_cells'] = len(rows) == 42
    checks['c_every_cell_check_holds'] = all(all(v for v in ck.values()) for ck in cell_checks.values())
    out['cell_checks_failed'] = {c: [k for k, v in ck.items() if not v] for c, ck in cell_checks.items()
                                 if not all(ck.values())}
    out['cells'] = rows

    cert_rows = [r for r in rows if r['status'] == 'certified']
    unc_rows = [r for r in rows if r['status'] != 'certified']
    causes = {}
    for r in unc_rows:
        causes.setdefault(r['cause'], []).append(r['cell'])
    tot = {
        'n_cells': len(rows),
        'n_certified': len(cert_rows), 'n_uncertified': len(unc_rows),
        'uncertified_by_cause': {k: {'n': len(causes.get(k, [])), 'cells': causes.get(k, [])}
                                 for k in ('gap clause', 'lapse reset', 'growth test', 'other')},
        'branch_counts_certified': {b: sum(r['branch'] == b for r in cert_rows) for b in ('oscillatory', 'monotone')},
        'k0_first_N_certified': _dist([r['k0_first_residual_pass_N'] for r in cert_rows]),
        'k0_first_N_all': _dist([r['k0_first_residual_pass_N'] for r in rows]),
        'k_star': _dist([r['k_star'] for r in cert_rows]),
        'cycles_after_k0_k_star_minus_k0_at_decision': _dist([r['k_star_minus_k0'] for r in cert_rows]),
        'cycles_after_k0_k_star_minus_N': _dist([r['k_star_minus_N'] for r in cert_rows]),
        'uncertified_cap_minus_N': _dist([r['end_minus_N'] for r in unc_rows]),
        'range_over_tau_at_k_star': _dist([r['range_over_tau'] for r in cert_rows]),
        'range_over_tau_sorted': sorted((r['range_over_tau'], r['cell']) for r in cert_rows),
        'n_range_over_tau_ge_0_95': sum(bool(r['range_over_tau_ge_0_95']) for r in cert_rows),
        'cells_range_over_tau_ge_0_95': [r['cell'] for r in cert_rows if r['range_over_tau_ge_0_95']],
        'terminal_step_over_EPS0_certified': _dist([r['terminal_step_over_EPS0'] for r in cert_rows]),
        'terminal_step_over_EPS0_uncertified': _dist([r['terminal_step_over_EPS0'] for r in unc_rows]),
        'terminal_step_over_EPS0_all': _dist([r['terminal_step_over_EPS0'] for r in rows]),
        'n_cells_terminal_step_gt_EPS0': sum(r['terminal_step_over_EPS0'] > 1.0 for r in rows),
        'vetoes_total': sum(r['n_vetoes'] for r in rows),
        'cells_with_vetoes': {r['cell']: r['n_vetoes'] for r in rows if r['n_vetoes']},
        'turning_point_floor_rejections_total': sum(r['turning_point_floor_rejections'] for r in rows),
        'cells_with_turning_point_floor_rejections': {r['cell']: r['turning_point_floor_rejections'] for r in rows
                                                      if r['turning_point_floor_rejections']},
        'cells_with_growth_floor_exclusion_at_end': {r['cell']: r['growth_floor_swings_excluded_at_end'] for r in rows
                                                     if r['growth_floor_swings_excluded_at_end']},
        'lapse_events_total': sum(len(r['lapse_events']) for r in rows),
        'cells_with_lapse_events': {r['cell']: [e['cycle'] for e in r['lapse_events']] for r in rows if r['lapse_events']},
        'non_clean_cycles_after_N_total': sum(len(r['non_clean_cycles_after_N']) for r in rows),
        'non_clean_block_events_after_N_by_family_total': {
            f: sum(r['non_clean_block_events_after_N_by_family'].get(f, 0) for r in rows) for f in ('TSO', 'DSO', 'ESSO')},
        'cells_with_non_clean_after_N': {r['cell']: r['non_clean_cycles_after_N'] for r in rows if r['non_clean_cycles_after_N']},
        'non_clean_after_N_per_cell': _dist([len(r['non_clean_cycles_after_N']) for r in rows]),
        'acceptable_clean_block_events_after_N_total': sum(r['acceptable_clean_block_events_after_N'] for r in rows),
        'acceptable_clean_block_events_all_total': sum(r['acceptable_clean_block_events_all'] for r in rows),
        'acceptable_clean_after_N_by_family_total': {
            f: sum(r['acceptable_clean_after_N_by_family'].get(f, 0) for r in rows) for f in ('TSO', 'DSO', 'ESSO')},
        'cells_with_acceptable_clean_after_N': {r['cell']: r['acceptable_clean_block_events_after_N'] for r in rows
                                                if r['acceptable_clean_block_events_after_N']},
        'certificates_with_a_turning_point_at_a_non_clean_cycle': [r['cell'] for r in cert_rows
                                                                   if r['certificate_turning_point_at_non_clean']],
        'certificates_with_last_three_turning_points_at_non_clean': [
            r['cell'] for r in cert_rows if r['certificate_last_three_turning_points_at_non_clean']],
        'certificates_with_a_turning_point_within_1_of_non_clean': [
            r['cell'] for r in cert_rows if r['certificate_turning_point_within_1_of_non_clean']],
        'certificates_with_a_non_clean_cycle_read_out_of_window': [r['cell'] for r in cert_rows
                                                                    if r['out_of_window_non_clean_read']],
        'wall_s_per_cycle': _dist([r['wall_s_per_cycle'] for r in rows]),
        'child_cycle_duration_median_s': _dist([r['child_cycle_duration_s']['median'] for r in rows]),
        'wall_s_total': sum(r['wall_s_parent'] for r in rows),
        'cycles_total': sum(r['cycles_recorded'] for r in rows)}
    out['totals'] = tot
    out['definitions'] = {
        'cells': ('the 34 cells of the v6 stage (stage spec 96c23404; the v6 summary reports minus the three B cells kept '
                  'under v4/v5 and d_c52e1670), the 7 extension cells (ext spec v3 84775dc4) and d_c52e1670 (v6 from its '
                  'records: the committed certificate + its v5 run\'s cycle records). The B cells are not included.'),
        'end_cycle': 'k* (certified) or the cap (uncertified); per-cell statistics are read through it',
        'N': 'the first residual pass (decision N, version-2 definition); "after k0" counts use N < cycle <= end cycle',
        'k0_at_decision': 'decision k0 (moved by a lapse reset; equal to N where no lapse occurred)',
        'cause': ('uncertified cells by first matching cause: gap clause (decision gap_refusals non-empty -- the committed '
                  "summary's gap_refused_cell definition), else lapse reset (decision lapse_events non-empty), else growth "
                  'test (reasons at cap include swings_growing), else other; every contributing factor is listed per cell'),
        'range_over_tau': 'decision range_over_tau at k* (certified cells only); flagged at >= 0.95',
        'terminal_step_over_EPS0': '|Q(end) - Q(end - 1)| / EPS0 from the cycle records (gross); EPS0 = TAU / 100 = '
                                   f'{EPS0!r}; cross-checked against the summary per cell',
        'vetoes': 'decision n_vetoes (certification refused at a cycle because of a non-clean cycle in the window)',
        'floor_rejections': ('turning-point floor (B) rejections recorded by the decision; growth-test floor (A) '
                             'exclusions at the end cycle from certA_parts swings_excluded_below_floor'),
        'non_clean': 'cycle records all_clean_k False (some block\'s final exit not clean); block events by the block name prefix',
        'acceptable_clean': 'cycle records acceptable_clean_blocks (a primary Acceptable within 10x: clean under v5/v6)',
        'turning_point_at_non_clean': ('certified OSCILLATORY cells only (a monotone certificate reads no turning point): '
                                       'a turning point of the decision T whose cycle is non-clean; also reported: the last '
                                       'three turning points, and the +/-1-cycle neighbourhood'),
        'wall': 'campaign_results parent_view wall_s / cycles recorded; child per-cycle duration t_end_s - t_start_s',
        'TAU': TAU, 'EPS0': EPS0}
    return checks, out


# ======================================================================================================================
#  main / manifest
# ======================================================================================================================
def _write(out_dir, name, doc):
    with open(os.path.join(out_dir, name), 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=False)


def _guards_report():
    return {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}


def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    if not os.path.isdir(out_dir):
        raise RuntimeError(f'{OUT_DIR_REL} must exist (created by the launch command, which writes launch.log there)')
    for n in (OUT_DISCOUNT, OUT_SALVAGE, OUT_CERT, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, n)):
            raise RuntimeError(f'{n} exists in {OUT_DIR_REL}: write-once, refusing')
    head = _git(['rev-parse', 'HEAD']).strip()
    porcelain = _git(['status', '--porcelain', '--untracked-files=no'])
    _log(f'{STAGE}; git HEAD {head}; tracked changes: {porcelain.strip() or "none"}')
    common = {'stage': STAGE, 'git_HEAD': head, 'git_status_porcelain_tracked': porcelain,
              'script': {'path': os.path.basename(__file__), 'sha256': _sha(os.path.basename(__file__))},
              'authority': ['PLANNER_BRIEF_2026-09-13.md Roadmap Step 5; Addendum 28 (discount ruling); Addendum 31 '
                            '(Step 5 list)', 'Planner task W153']}
    results, all_checks, errors = {}, {}, {}
    for key, fn in (('discount', discount_row), ('salvage', salvage_row), ('certification', certification_statistics)):
        try:
            ck, doc = fn()
        except Exception:
            errors[key] = traceback.format_exc()
            _log(f'ERROR in {key}:\n{errors[key]}')
            ck, doc = {f'{key}_ran': False}, None
        results[key] = doc
        all_checks[key] = ck
    guards = _guards_report()
    failed = {k: [n for n, v in ck.items() if not v] for k, ck in all_checks.items()}
    pickle_ok = PICKLE_COUNTS == {'load': 0, 'loads': 0}
    ok = (not errors and not any(failed.values()) and pickle_ok
          and all(not v['verify_0_failures'] for v in guards.values()))
    tail = {'checks': all_checks, 'failed_checks': failed, 'errors': errors, 'solve_profile_guards': guards,
            'pickle_guard': dict(PICKLE_COUNTS), 'all_pass': ok, 'ended_utc': _utc()}
    for name, key in ((OUT_DISCOUNT, 'discount'), (OUT_SALVAGE, 'salvage'), (OUT_CERT, 'certification')):
        doc = dict(common)
        doc['section'] = key
        doc['result'] = results[key]
        doc['section_checks'] = all_checks[key]
        doc.update(tail)
        doc['inputs_sha256'] = dict(sorted(INPUTS.items()))
        _write(out_dir, name, doc)
    _print_tables(results)
    _log(f"checks failed: {failed}; errors: {list(errors)}; pickle {PICKLE_COUNTS}; guards "
         f"{({k: v['verify_0_failures'] for k, v in guards.items()})}; inputs read {len(INPUTS)}; wall {time.time() - t0:.1f} s")
    _log(f'all_pass {ok}')
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(0 if ok else 1)


def _print_tables(results):
    d = results.get('discount')
    if d:
        _log(f"DISCOUNT instance: unit {d['instance']['unit']['candidate_key'][:8]} eval {d['instance']['unit']['eval_key'][:8]} "
             f"Q172 {d['instance']['unit']['certified_cost_gross_eur']!r}; x0 eval {d['instance']['x0']['eval_key'][:8]} "
             f"Q181 {d['instance']['x0']['certified_cost_gross_eur']!r}; I {d['I_x_eur']!r}")
        _log(f"DISCOUNT reconstruction: certified V {d['reconstruction']['certified_value_eur']:.4f}; from blocks "
             f"{d['reconstruction']['V_r2_from_weighted_blocks']:.4f}; re-weighted {d['reconstruction']['V_r2_from_undiscounted_reweighted']:.4f}")
        for y, v in d['V_per_year'].items():
            _log(f"DISCOUNT V_{y}: weighted 2 % {v['V_weighted_r2']:.2f}; undiscounted {v['V_undiscounted']:.2f}")
        for r, e in zip(d['table'], d['earlier_row']['rows']):
            _log(f"DISCOUNT r {r['rate']:.2f}: V {r['V']:.2f} ({r['change_vs_2pct_pct']:+.2f} % vs 2 %) value/I "
                 f"{r['value_to_cost']:.4f} value-I {r['value_minus_I']:.2f} ({r['value_minus_I_over_a61_threshold']:.2f}x "
                 f"thr {r['a61_threshold_conservative']:.2f}: {r['value_minus_I_verdict_a61']}) | earlier V {e['V_earlier']:.2f} "
                 f"value/I {e['value_to_cost_earlier']:.4f} value-I {e['value_minus_I_earlier']:.2f}")
        ps = d['price_spread']
        _log(f"SPREAD market (day-weighted, undiscounted): {ps['market_spread']['all_years_day_weighted_undiscounted']}; "
             f"captured settled {ps['captured_settled']}; earlier {ps['captured_earlier_w19']} {ps['captured_earlier_w19_per_year']}")
        _log(f"AGEING settled unit: {[(t['year_block'], round(t['efc_per_day'], 4), round(t['soh_end_of_block'], 4), t['floor_row_active']) for t in d['settled_unit_ageing']['per_year']]}")
    s = results.get('salvage')
    if s:
        for r in s['claims']:
            _log(f"SALVAGE {r['claim_id']}: gross {r['d_gross']:.2f} [{r['gross_verdict']}] net {r['d_net']:.2f} "
                 f"[{r['net_verdict']}] salvage ref {r['salvage_ref']:.2f} other {r['salvage_other']:.2f} cells "
                 f"{r['ref_cells']} / {r['other_cells']} differs {r['verdict_differs_between_conventions']}")
        _log(f"SALVAGE totals {s['totals']}")
    c = results.get('certification')
    if c:
        for r in c['cells']:
            _log(f"CERT {r['cell']}: {r['status']} {r['branch']} cause {r['cause']} N {r['k0_first_residual_pass_N']} k0 "
                 f"{r['k0_at_decision']} k* {r['k_star']} end {r['end_cycle']} r/tau {r['range_over_tau']} step/EPS0 "
                 f"{r['terminal_step_over_EPS0']:.3f} vetoes {r['n_vetoes']} tpfr {r['turning_point_floor_rejections']} "
                 f"nc>N {r['non_clean_cycles_after_N']} fam {r['non_clean_block_events_after_N_by_family']} acc>N "
                 f"{r['acceptable_clean_block_events_after_N']} tp@nc {r['turning_points_at_non_clean_cycle']} wall/cyc "
                 f"{r['wall_s_per_cycle']:.1f}")
        _log(f"CERT totals {json.dumps(c['totals'], default=str)}")
        if c['cell_checks_failed']:
            _log(f"CERT cell checks failed: {c['cell_checks_failed']}")


def write_manifest():
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    if os.path.exists(os.path.join(out_dir, OUT_MAN)):
        raise RuntimeError(f'{OUT_MAN} exists: write-once, refusing')
    with open(os.path.join(out_dir, OUT_DISCOUNT)) as handle:
        inputs = json.load(handle)['inputs_sha256']
    files = [os.path.basename(__file__)] + [os.path.join(OUT_DIR_REL, n) for n in (OUT_DISCOUNT, OUT_SALVAGE, OUT_CERT,
                                                                                    LAUNCH_LOGS[0])]
    man = {rel: _sha(rel) for rel in files}
    man.update(inputs)
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(dict(sorted(man.items())), handle, indent=1)
    _log(f'manifest: {len(man)} entries ({len(files)} produced, {len(inputs)} inputs); guards '
         f"{({k: v['verify_0_failures'] for k, v in _guards_report().items()})}; pickle {PICKLE_COUNTS}")
    ok = PICKLE_COUNTS == {'load': 0, 'loads': 0} and all(not v['verify_0_failures'] for v in _guards_report().values())
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if '--manifest' in sys.argv[1:]:
        write_manifest()
    else:
        main()
