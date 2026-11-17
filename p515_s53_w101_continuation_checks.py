"""
P5.15 Addendum 53, Planner task W101 -- ZERO-SOLVE checks for the SRP1 settling continuation (spec v39).

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (W98's checks
module, imported for its stand-in helpers, arms its own; both are verified at 0). No model is built and nothing is
solved: the lambda_t write-only proof reads the committed certified x = 0 models (unpickled, sha256-verified against
the recert's committed manifest) and a fresh SRP1 planning object (data read only).

WHAT IS CHECKED
  T   the pure stop rule (`settling_criterion`) on the 12 trajectories of the task, each with its expected verdict.
  R   the pure rule replayed on the five committed records (3 x 3 x = 0 continuation cycles 1-88, 3 x 3 node-7 1-69, the
      three SRP1 tight-tail references 1..N) against the task's expected values; no 3 x 3 certification before 88, no
      SRP1 certification at k <= N (with decisions allowed from cycle 1). P_MAX computed by the module's own function.
  L   lambda_t (`interface_dual_capture`): the capture path; the wrapper passes production's arguments and return value
      through unchanged (object identity); WRITE-ONLY on real SRP1 objects (every Var value / fixed flag / bound, every
      Param value, every Constraint / Objective active flag of the 36 + 12 certified blocks, the solver options, the
      ADMM parameters, the network data read, dual_vars and consensus_vars: fingerprints before == after); the shape;
      the sidecar size per cycle at SRP1 and at 3 x 3.
  H   the hooks (`p515_s53_w101_settling_continuation_hooks`): H1 AA hold (stand-in identity through N, forced copy
      after); H2 AA hold with REAL production AA; H3 tail hold with REAL production functions; H4 rho hold with REAL
      production `_update_admm_penalties`; H5 the in-cycle replay gate and the settling decision driven by the
      RECORDED production values of each SRP1 reference (g_s39_D.json rows) through the real wrappers -- bitwise through
      N on all three cells, the rule's in-cycle state equal to the pure replay, then a synthetic continuation that the
      rule certifies (certificate length 0 at k*, the decision file, summary ok), a one-ulp divergence at cycle 40
      (ABORT at 40 with its magnitude, nothing after), a failed cycle (lapse), the cap (uncertified record); H6 the real
      install layering (settling hooks first, the harness appender on top; every production function restored;
      `_child_real` enters them in that order); H7 the certificate length is written only by the disable / restore /
      settling decision (source), with a negative control.
  K   keys: for EVERY entry of every committed campaign spec the W101 harness's key equals the pre-W101 harness's
      (4a80c3e2, sha256 63e06859, loaded from git); the three continuation keys follow the declared formula, their base
      keys equal the recert's eval keys, and they appear in no committed spec outside the W101 root.
  P   `assert_settling_preconditions` (checklist a-f, h) holds for each cell's declaration and refuses negative
      controls; the validator refuses an `early_stop` key and malformed declarations.

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w101_continuation_checks.py \\
      > data/SRP1/Results/P515S53/w101_srp1_continuation/zero_solve_checks_launch.log 2>&1
"""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W101 settling-continuation zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w101_settling_continuation_hooks as C  # noqa: E402
import p515_s53_w98_continuation_checks as K98  # noqa: E402 -- its stand-in helpers (arms its own permitted=() guard)
import settling_criterion as SC  # noqa: E402

GUARDS = (('w101_checks', GUARD), ('w98_checks_imported', K98.GUARD))
W101_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation')
OUT_DIR_REL = os.path.join(W101_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w101_zero_solve_checks.json'
OUT_MANIFEST = 'w101_zero_solve_checks_manifest_sha256.json'
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
RECORDS = {
    '3x3_x0_continuation_1_88': {
        'path': os.path.join(_P53, 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2', 'evals',
                             '25b92ae0f1f2c02e_x0_cont', 'per_cycle_record.jsonl'), 'N': 88},
    '3x3_node7_1_69': {
        'path': os.path.join(_P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals', 'c82522f470b35b58_n7_4h_e1',
                             'per_cycle_record.jsonl'), 'N': 69},
    'srp1_x0': {'path': C.reference_path('x0'), 'N': C.CELLS['x0']['N']},
    'srp1_n7_4h_e1': {'path': C.reference_path('n7_4h_e1'), 'N': C.CELLS['n7_4h_e1']['N']},
    'srp1_c_star': {'path': C.reference_path('c_star'), 'N': C.CELLS['c_star']['N']},
}
SRP1_RECORD_NAMES = ('srp1_x0', 'srp1_n7_4h_e1', 'srp1_c_star')
PRE_W101_HARNESS = {'commit': '4a80c3e2', 'sha256': '63e06859c7f7786faae8109a52041176b402905085b9864d3c651e6ee67f9fb3'}
RECERT_MANIFEST_REL = os.path.join(C.RECERT_ROOT, 'campaign_manifest_sha256.json')
CERTIFIED_MODELS_X0_REL = os.path.join(C.RECERT_ROOT, 'evals', C.CELLS['x0']['eval_dir'], 'certified_models.pkl')
ADVISOR_P_MAX = 22
SHAPE_3X3 = {'nodes': 3, 'years': 5, 'days': 4, 'periods': 24,
             'basis': ('3 x 3 derived instance: 80 network blocks per round = (1 TSO + 3 DSO) x 5 years x 4 days '
                       '(p515_s53_w89_3x3_campaign.BLOCKS_PER_ROUND), 24 periods, 3 active DN nodes')}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_record(name):
    rows = _read_jsonl(_abs(RECORDS[name]['path']))
    q = {r['cycle']: r['gross_operational_cost'] for r in rows}
    b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
    return q, b, rows


def compute_p_max():
    recs = {}
    for name in SRP1_RECORD_NAMES:
        q, _b, _rows = load_record(name)
        recs[name] = (q, RECORDS[name]['N'])
    return SC.p_max_from_records(recs)


# ======================================================================================================================
#  T -- the 12 trajectories
# ======================================================================================================================
def _run(q, b, n, cap, p_max):
    out, dec, rule = SC.replay(q, b, n, cap, p_max)
    return out, dec, rule


def _tp(rule):
    return [[t, kind] for t, kind, _v in rule.sign.T]


def _certified_ks(out):
    return [r['k'] for r in out if str(r.get('decision') or '').startswith('certified')]


def tests_T(p_max):
    res = {}
    x0q, x0b, _ = load_record('3x3_x0_continuation_1_88')
    n7q, n7b, _ = load_record('3x3_node7_1_69')

    def trunc(q, b, k):
        return {c: q[c] for c in range(1, k + 1)}, {c: b[c] for c in range(1, k + 1)}

    # 1-4: the 3 x 3 x = 0 record truncated; decisions allowed from cycle 1 (N = 0), cap = the truncation
    for tid, k, exp in (('T1_x0_trunc_72', 72, {'T': [[66, 'max']], 'reason': 'insufficient_turning_points'}),
                        ('T2_x0_trunc_77', 77, {'T': [[66, 'max']], 'reason': 'insufficient_turning_points'}),
                        ('T3_x0_trunc_86', 86, {'T': [[66, 'max'], [77, 'min']], 'reason': 'insufficient_turning_points',
                                                'A_rounded': [29140]}),
                        ('T4_x0_trunc_88', 88, {'T': [[66, 'max'], [77, 'min'], [86, 'max']],
                                                'reason': 'range_above_tau', 'A_rounded': [29140, 7824]})):
        q, b = trunc(x0q, x0b, k)
        out, dec, rule = _run(q, b, 0, k, p_max)
        got = {'k0': rule.k0, 'T': _tp(rule), 'A': list(rule.sign.A), 'certified_at': _certified_ks(out),
               'status': dec['status'], 'reasons': dec.get('reasons'),
               'sign_changes': [r['k'] for r in out if r['sign_change']]}
        ok = (got['status'] == 'uncertified' and not got['certified_at'] and got['T'] == exp['T']
              and exp['reason'] in (got['reasons'] or []) and got['k0'] == 63)
        if 'A_rounded' in exp:
            ok = ok and [round(a) for a in got['A']] == exp['A_rounded']
        if tid == 'T4_x0_trunc_88':
            last = out[-1]
            got.update({'sign_change_at_88': last['sign_change'] is True and last['k'] == 88,
                        'turning_point_at_88': last['turning_point'], 'P_hat': last['P_hat'], 'W': last['W'],
                        'window': last['window'], 'range': last['range'], 'range_over_tau': last['range_over_tau']})
            ok = ok and got['sign_change_at_88'] and last['turning_point'][:2] == [86, 'max'] \
                and last['P_hat'] == 20 and last['W'] == 22 and last['window'] == [67, 88] \
                and round(last['range'], 2) == 28130.18
        res[tid] = {'ok': bool(ok), 'expected': exp, 'got': got}
    out, dec, rule = _run(n7q, n7b, 0, 69, p_max)
    got = {'k0': rule.k0, 'T': _tp(rule), 'status': dec['status'], 'reasons': dec.get('reasons'),
           'certified_at': _certified_ks(out)}
    res['T5_node7_1_69'] = {'ok': got['k0'] == 60 and got['T'] == [[65, 'max']] and got['status'] == 'uncertified'
                            and not got['certified_at'], 'expected': {'k0': 60, 'T': [[65, 'max']]}, 'got': got}

    # 6 / 7: damped cosine, Boyd false 1-10
    def damped(boyd_false=()):
        q = {k: 1e9 + 30000.0 * 0.88 ** (k - 11) * math.cos(2 * math.pi * (k - 11) / 20.0) for k in range(1, 201)}
        b = {k: (k >= 11 and k not in boyd_false) for k in range(1, 201)}
        return q, b
    q6, b6 = damped()
    out6, dec6, rule6 = _run(q6, b6, 0, 200, p_max)
    k6 = dec6.get('k_star')
    ok6 = (dec6['status'] == 'certified' and dec6['branch'] == 'oscillatory' and len(dec6['T']) >= 3
           and all(dec6['A'][i + 1] <= dec6['A'][i] for i in range(len(dec6['A']) - 1))
           and 18 <= dec6['P_hat'] <= 22 and dec6['range'] <= SC.TAU and abs(q6[k6] - 1e9) < SC.TAU)
    res['T6_damped_cosine'] = {'ok': bool(ok6), 'k_star': k6, 'k0': dec6.get('k0'), 'branch': dec6.get('branch'),
                               'T': dec6.get('T'), 'A': dec6.get('A'), 'P_hat': dec6.get('P_hat'), 'W': dec6.get('W'),
                               'range': dec6.get('range'), 'band': dec6.get('band'),
                               'Q_k_star_minus_1e9': (q6[k6] - 1e9) if k6 else None}
    lapse_at = 11 + 25
    q7, b7 = damped(boyd_false=(lapse_at,))
    out7, dec7, rule7 = _run(q7, b7, 0, 200, p_max)
    ok7 = (dec7['status'] == 'certified' and dec7['k_star'] > k6 and len(rule7.lapses) == 1
           and rule7.lapses[0]['cycle'] == lapse_at and dec7['k0'] == lapse_at + 1)
    res['T7_damped_cosine_lapse_at_k0_plus_25'] = {'ok': bool(ok7), 'lapse_cycle': lapse_at, 'k_star': dec7.get('k_star'),
                                                   'k_star_T6': k6, 'k0_after_reset': dec7.get('k0'),
                                                   'lapse_events': rule7.lapses, 'branch': dec7.get('branch')}
    # 8: constant creep -80 for 120 cycles
    q8 = {k: 1e9 - 80.0 * (k - 1) for k in range(1, 121)}
    b8 = {k: True for k in q8}
    out8, dec8, _r8 = _run(q8, b8, 0, 120, p_max)
    res['T8_constant_creep_minus_80'] = {'ok': dec8['status'] == 'uncertified' and not _certified_ks(out8)
                                         and 'monotone_not_decreasing' in dec8['reasons'],
                                         'status': dec8['status'], 'reasons': dec8['reasons'],
                                         'band': dec8.get('band'), 'band_width': dec8.get('band_width')}
    # 9: decaying creep dQ_k = -150 * 0.97^(k - k0)
    q9 = {1: 1e9}
    for k in range(2, 201):
        q9[k] = q9[k - 1] - 150.0 * 0.97 ** (k - 1)
    b9 = {k: True for k in q9}
    out9, dec9, _r9 = _run(q9, b9, 0, 200, p_max)
    expected9 = 1 + SC.K_EXCL + 2 * p_max - 1
    res['T9_decaying_creep'] = {'ok': dec9['status'] == 'certified' and dec9['branch'] == 'monotone'
                                and dec9['k_star'] == expected9, 'k_star': dec9.get('k_star'),
                                'expected_k_star_k0_plus_K_EXCL_plus_L_MONO_minus_1': expected9,
                                'branch': dec9.get('branch'), 'range': dec9.get('range'), 'band': dec9.get('band'),
                                'certB_parts': dec9.get('certB_parts')}
    # 10: growing oscillation, turning points every 10 cycles, half-swings 1000 * 1.15^j
    q10 = {}
    tps = [(5 + 10 * j) for j in range(16)]
    vals = [1e9]
    for j in range(15):
        vals.append(vals[-1] + (-1) ** (j + 1) * 1000.0 * 1.15 ** j)
    for k in range(1, tps[0] + 1):
        q10[k] = vals[0] - 500.0 + 500.0 * (k - 1) / (tps[0] - 1)
    for j in range(15):
        a, bb = tps[j], tps[j + 1]
        for k in range(a + 1, bb + 1):
            q10[k] = vals[j] + (vals[j + 1] - vals[j]) * (k - a) / (bb - a)
    cap10 = 150
    b10 = {k: True for k in range(1, cap10 + 1)}
    out10, dec10, r10 = _run({k: q10[k] for k in range(1, cap10 + 1)}, b10, 0, cap10, p_max)
    res['T10_growing_oscillation'] = {'ok': dec10['status'] == 'uncertified' and not _certified_ks(out10)
                                      and 'swings_growing' in dec10['reasons'], 'status': dec10['status'],
                                      'reasons': dec10['reasons'], 'T': [[t, kd] for t, kd, _ in r10.sign.T],
                                      'A': list(r10.sign.A)}
    # 11: jitter
    q11a = {k: 1e9 + 20.0 * (-1) ** k for k in range(1, 121)}
    out11a, dec11a, _ = _run(q11a, {k: True for k in q11a}, 0, 120, p_max)
    ok11a = (dec11a['status'] == 'certified' and dec11a['branch'] == 'monotone'
             and dec11a['certB_parts']['stationary'] is True and dec11a['band_width'] <= 40.0)
    q11b = {k: 1e9 + 60.0 * (-1) ** k for k in range(1, 121)}
    out11b, dec11b, _ = _run(q11b, {k: True for k in q11b}, 0, 120, p_max)
    ok11b = dec11b['status'] == 'certified' and dec11b['branch'] == 'oscillatory'
    rnd = random.Random(20260927)
    q11c = {k: 1e9 + (-1) ** k * rnd.uniform(40.0, 80.0) for k in range(1, 121)}
    out11c, dec11c, r11c = _run(q11c, {k: True for k in q11c}, 0, 120, p_max)
    rnd2 = random.Random(20260927)
    q11d = {k: 1e9 + (-1) ** k * rnd2.uniform(20.0, 40.0) for k in range(1, 121)}
    out11d, dec11d, _ = _run(q11d, {k: True for k in q11d}, 0, 120, p_max)
    res['T11_jitter'] = {
        'ok': bool(ok11a and ok11b),
        'pm20_stationary': {'ok': bool(ok11a), 'k_star': dec11a.get('k_star'), 'branch': dec11a.get('branch'),
                            'band_width': dec11a.get('band_width')},
        'pm60_equal_amplitudes': {'ok': bool(ok11b), 'k_star': dec11b.get('k_star'), 'branch': dec11b.get('branch'),
                                  'P_hat': dec11b.get('P_hat'), 'W': dec11b.get('W'),
                                  'band_width': dec11b.get('band_width')},
        'random_amplitudes_pm_40_80_seed_20260927_recorded_not_gated': {
            'reading': 'Q_k = 1e9 + (-1)^k a_k, a_k ~ U(40, 80) (amplitude = the half-swing about 1e9, as for +-20 / +-60)',
            'status': dec11c['status'], 'k_star': dec11c.get('k_star'), 'branch': dec11c.get('branch'),
            'reasons': dec11c.get('reasons'), 'band': dec11c.get('band'), 'band_width': dec11c.get('band_width'),
            'n_turning_points': len(r11c.sign.T)},
        'random_peak_to_peak_40_80_variant_recorded_not_gated': {
            'reading': 'Q_k = 1e9 + (-1)^k a_k, a_k ~ U(20, 40) (peak-to-peak 40-80), same seed',
            'status': dec11d['status'], 'k_star': dec11d.get('k_star'), 'branch': dec11d.get('branch'),
            'reasons': dec11d.get('reasons'), 'band_width': dec11d.get('band_width')},
    }
    # 12: steps +500, 0.0, -500
    q12 = {1: 1e9, 2: 1e9, 3: 1e9, 4: 1e9, 5: 1e9, 6: 1e9 + 500.0, 7: 1e9 + 500.0, 8: 1e9, 9: 1e9, 10: 1e9}
    out12, dec12, r12 = _run(q12, {k: True for k in q12}, 0, 10, p_max)
    by = {r['k']: r for r in out12}
    ok12 = (by[8]['sign_change'] is True and not by[6]['sign_change'] and not by[7]['sign_change']
            and by[8]['turning_point'][:2] == [6, 'max'] and by[7]['s_k'] == 0)
    res['T12_plus500_zero_minus500'] = {'ok': bool(ok12), 'sign_change_cycles': [r['k'] for r in out12 if r['sign_change']],
                                        'turning_point': by[8]['turning_point'],
                                        'note': 'the +500 step is cycle 6, the 0.0 step cycle 7, the -500 step cycle 8'}
    return res


# ======================================================================================================================
#  R -- replay on the five committed records
# ======================================================================================================================
def tests_R(p_max):
    res = {}
    for name, rec in RECORDS.items():
        q, b, rows = load_record(name)
        n_rec = rec['N']
        sha = H.sha256_file(_abs(rec['path']))
        # decisions allowed from cycle 1 (N = 0): the rule must not certify anywhere in the record
        out0, dec0, rule0 = _run(q, b, 0, n_rec, p_max)
        eligible_changes = [r['k'] for r in out0 if r['sign_change']]
        entry = {'path': rec['path'], 'sha256': sha, 'cycles': len(rows), 'k0': rule0.k0,
                 'sign_changes': eligible_changes, 'T': [[t, kd, v] for t, kd, v in rule0.sign.T],
                 'A': list(rule0.sign.A), 'status_at_end_decisions_from_cycle_1': dec0['status'],
                 'reasons_at_end': dec0.get('reasons'), 'certified_at_any_k_decisions_from_cycle_1': _certified_ks(out0),
                 'literal_two_sign_change_report_at_end': SC.literal_two_sign_change_report(rule0, n_rec)}
        if name.startswith('srp1'):
            outn, decn, _rn = _run(q, b, n_rec, n_rec, p_max)
            entry['with_N_equal_cert_cycle'] = {'N': n_rec, 'certified_at': _certified_ks(outn),
                                                'status': decn['status']}
        last = out0[-1]
        entry.update({'P_hat': last.get('P_hat'), 'W': last.get('W'), 'window': last.get('window'),
                      'range': last.get('range'), 'range_over_tau': last.get('range_over_tau')})
        res[name] = entry
    x0, n7 = res['3x3_x0_continuation_1_88'], res['3x3_node7_1_69']
    checks = {
        'x0_k0_63': x0['k0'] == 63,
        'x0_sign_changes_67_78_88': x0['sign_changes'] == [67, 78, 88],
        'x0_T_66max_77min_86max': [t[:2] for t in x0['T']] == [[66, 'max'], [77, 'min'], [86, 'max']],
        'x0_A_29140_7824': [round(a) for a in x0['A']] == [29140, 7824],
        'x0_P_hat_20_W_22': x0['P_hat'] == 20 and x0['W'] == 22,
        'x0_range_67_88_28130.18_gt_tau': x0['window'] == [67, 88] and round(x0['range'], 2) == 28130.18
        and x0['range'] > SC.TAU,
        'x0_not_certified_through_88': not x0['certified_at_any_k_decisions_from_cycle_1'],
        'n7_k0_60_T_65max_not_certified': (n7['k0'] == 60 and [t[:2] for t in n7['T']] == [[65, 'max']]
                                           and not n7['certified_at_any_k_decisions_from_cycle_1']),
        'srp1_x0_at_most_1_sign_change': len(res['srp1_x0']['sign_changes']) <= 1,
        'srp1_unit_at_most_1_sign_change': len(res['srp1_n7_4h_e1']['sign_changes']) <= 1,
        'srp1_c_star_0_sign_changes': len(res['srp1_c_star']['sign_changes']) == 0,
        'srp1_none_certified_at_k_le_N': all(not res[n]['certified_at_any_k_decisions_from_cycle_1']
                                             and not res[n]['with_N_equal_cert_cycle']['certified_at']
                                             for n in SRP1_RECORD_NAMES),
    }
    dQ87 = load_record('3x3_x0_continuation_1_88')[0]
    checks_detail = {'x0_dQ87': dQ87[87] - dQ87[86], 'x0_dQ88': dQ87[88] - dQ87[87], 'EPS0': SC.EPS0}
    checks['x0_dQ87_below_EPS0_so_the_change_registers_at_88'] = abs(checks_detail['x0_dQ87']) < SC.EPS0
    defect = (not checks['x0_not_certified_through_88'] or not checks['srp1_none_certified_at_k_le_N']
              or not checks['n7_k0_60_T_65max_not_certified'])
    return {'records': res, 'checks': checks, 'detail': checks_detail, 'defect_stop_condition': defect,
            'holds': all(checks.values())}


# ======================================================================================================================
#  L -- lambda_t capture
# ======================================================================================================================
def _fingerprint_models(models):
    import pyomo.environ as pe
    h = hashlib.sha256()
    counts = {'vars': 0, 'params': 0, 'constraints': 0, 'objectives': 0, 'blocks': 0}

    def blk(b):
        counts['blocks'] += 1
        for v in b.component_data_objects(pe.Var, descend_into=True, sort=True):
            val = v.value
            h.update(f'{v.name}|{float.hex(float(val)) if val is not None else None}|{v.fixed}|{v.lb}|{v.ub}\n'.encode())
            counts['vars'] += 1
        for pc in b.component_objects(pe.Param, descend_into=True, sort=True):
            for idx in sorted(pc.keys(), key=repr):
                try:
                    val = pe.value(pc[idx])
                    txt = float.hex(float(val)) if isinstance(val, (int, float)) else repr(val)
                except Exception as error:  # noqa: BLE001
                    txt = f'<{type(error).__name__}>'
                h.update(f'{pc.name}[{idx!r}]|{pc.mutable}|{txt}\n'.encode())
                counts['params'] += 1
        for c in b.component_data_objects(pe.Constraint, descend_into=True, sort=True):
            h.update(f'{c.name}|{c.active}\n'.encode())
            counts['constraints'] += 1
        for o in b.component_data_objects(pe.Objective, descend_into=True, sort=True):
            h.update(f'{o.name}|{o.active}\n'.encode())
            counts['objectives'] += 1

    for y in sorted(models['tso'], key=str):
        for d in sorted(models['tso'][y], key=str):
            blk(models['tso'][y][d])
    for nd in sorted(models['dso'], key=str):
        for y in sorted(models['dso'][nd], key=str):
            for d in sorted(models['dso'][nd][y], key=str):
                blk(models['dso'][nd][y][d])
    return h.hexdigest(), counts


def _deep_hex(obj, h):
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(f'k:{k!r}\n'.encode())
            _deep_hex(obj[k], h)
    elif isinstance(obj, (list, tuple)):
        h.update(f'l:{len(obj)}\n'.encode())
        for v in obj:
            _deep_hex(v, h)
    elif isinstance(obj, float):
        h.update(f'f:{float.hex(obj)}\n'.encode())
    else:
        h.update(f'o:{obj!r}\n'.encode())


def _fingerprint_planning(planning, srp):
    h = hashlib.sha256()
    for label, nd in srp._convergence_depth_tail_holders(planning):
        _deep_hex({'label': label, 'options': dict(nd.params.solver_params.options or {}),
                   'recovery': dict(nd.params.solver_params.recovery_options or {})}, h)
        for y in nd.years:
            for d in nd.days:
                net = nd.network[y][d]
                _deep_hex({'baseMVA': float(net.baseMVA),
                           'branches': [(br.fbus, br.tbus, float(br.rate), bool(br.status)) for br in net.branches]}, h)
    _deep_hex(json.loads(json.dumps(vars(planning.params.admm), default=str, sort_keys=True)), h)
    return h.hexdigest()


def tests_L():
    import numpy as np  # noqa: F401 -- numpy floats in the capture path are handled by float()
    import p56a_oracle as O
    import shared_resources_planning as srp
    res = {}
    # L1 capture path
    try:
        res['L1_capture_path'] = {'holds': True, 'checklist': IDC.assert_capture_path()}
    except Exception as error:  # noqa: BLE001
        res['L1_capture_path'] = {'holds': False, 'error': f'{type(error).__name__}: {error}'}
    # L2 pass-through identity (stand-in original) + cycle numbering + close()
    seen = []
    sentinel_out = object()

    def original(pp, tso, dso, esso, cv, dv, ap):
        seen.append((pp, tso, dso, esso, cv, dv, ap))
        return sentinel_out
    sink = []
    cap = IDC.InterfaceDualCapture(eval_dir=None, sink=sink)
    w = IDC.make_wrapper(cap, original)
    fake_pp = SimpleNamespace(active_distribution_network_nodes=[5], years={2025: 1}, days={'Spring': 1},
                              transmission_network=SimpleNamespace(network={2025: {'Spring': SimpleNamespace(baseMVA=100.0)}}),
                              distribution_networks={5: SimpleNamespace(network={2025: {'Spring': SimpleNamespace(
                                  baseMVA=100.0, get_interface_branch_rating=lambda: 200.0)}})})
    dv = {'pf': {'tso': {'current': {5: {2025: {'Spring': {'p': [1.5, -2.25], 'q': [0.0, 3.0]}}}}},
                 'dso': {'current': {5: {2025: {'Spring': {'p': [-1.5, 2.25], 'q': [0.5, -3.0]}}}}}}}
    args = [(fake_pp, {2025: {'Spring': SimpleNamespace()}}, {5: {2025: {'Spring': SimpleNamespace()}}}, object(),
             {}, dv, SimpleNamespace(objective_scale=7.0)) for _ in range(3)]
    outs = [w(*a) for a in args]
    cap.close()
    w(*args[0])
    identity = all(o is sentinel_out for o in outs) and all(
        all(x is y for x, y in zip(seen[i], args[i])) for i in range(3))
    lines = [o for f, o in sink if f == IDC.INTERFACE_DUAL_FILE]
    res['L2_passthrough_identity_and_numbering'] = {
        'holds': (identity and len(lines) == 4 and lines[0].get('header') is True
                  and [x['cycle'] for x in lines[1:]] == [1, 2, 3] and lines[1]['lambda_pf_p_tso'] == [[1.5, -2.25]]
                  and lines[1]['lambda_pf_q_dso'] == [[0.5, -3.0]] and cap.summary()['ok'] and len(seen) == 4
                  and cap.calls == 3),
        'arguments_and_return_identical': identity, 'lines': len(lines), 'calls_captured_before_close': cap.calls,
        'original_called_after_close_too': len(seen) == 4, 'summary': cap.summary()}
    # L3 write-only on real SRP1 objects
    manifest = json.load(open(_abs(RECERT_MANIFEST_REL)))
    want = manifest[CERTIFIED_MODELS_X0_REL]
    got = H.sha256_file(_abs(CERTIFIED_MODELS_X0_REL))
    eval_id = f'p515s53_w101_lambda_writeonly_{int(time.time())}'
    work = os.path.join(O.WORK_DIR, eval_id)
    try:
        if got != want:
            raise RuntimeError(f'certified_models.pkl sha256 {got} != committed manifest {want}')
        with open(_abs(CERTIFIED_MODELS_X0_REL), 'rb') as handle:
            models = pickle.load(handle)
        planning = O.fresh_planning(eval_id)
        cv, dv = srp.create_admm_variables(planning)
        rnd = random.Random(53)
        for group in ('vmag', 'pf', 'ess'):
            for agent, tree in dv[group].items():
                def fill(node):
                    if isinstance(node, dict):
                        for k in node:
                            if isinstance(node[k], list):
                                node[k] = [rnd.uniform(-5e3, 5e3) for _ in node[k]]
                            elif isinstance(node[k], float):
                                node[k] = rnd.uniform(-5e3, 5e3)
                            else:
                                fill(node[k])
                fill(tree)
        before_m, counts = _fingerprint_models(models)
        before_p = _fingerprint_planning(planning, srp)
        hdv, hcv = hashlib.sha256(), hashlib.sha256()
        _deep_hex(dv, hdv)
        _deep_hex(cv, hcv)
        before_dv, before_cv = hdv.hexdigest(), hcv.hexdigest()
        sink3 = []
        cap3 = IDC.InterfaceDualCapture(eval_dir=None, sink=sink3)
        w3 = IDC.make_wrapper(cap3, lambda *a: 'production-return')
        rets = [w3(planning, models['tso'], models['dso'], None, cv, dv, planning.params.admm) for _ in range(3)]
        after_m, counts2 = _fingerprint_models(models)
        after_p = _fingerprint_planning(planning, srp)
        hdv, hcv = hashlib.sha256(), hashlib.sha256()
        _deep_hex(dv, hdv)
        _deep_hex(cv, hcv)
        lines3 = [o for f, o in sink3 if f == IDC.INTERFACE_DUAL_FILE]
        header = lines3[0]
        first = lines3[1]
        blocks = IDC.block_order(planning)
        spot = blocks[7]
        spot_ok = (first['lambda_pf_p_tso'][7] == dv['pf']['tso']['current'][spot[0]][spot[1]][spot[2]]['p']
                   and first['lambda_pf_q_dso'][7] == dv['pf']['dso']['current'][spot[0]][spot[1]][spot[2]]['q'])
        mutable = sorted({(b['admm_objective_scale_tso_mutable'], b['admm_objective_scale_dso_mutable'])
                          for b in header['metadata']['per_block']})
        line_bytes = [len(GRIO.dumps(x, default=GRIO.json_default).encode()) + 1 for x in lines3]
        res['L3_write_only_real_srp1'] = {
            'holds': (before_m == after_m and before_p == after_p and before_dv == hdv.hexdigest()
                      and before_cv == hcv.hexdigest() and counts == counts2 and all(r == 'production-return'
                                                                                     for r in rets)
                      and cap3.summary()['ok'] and spot_ok and mutable == [(False, False)]),
            'certified_models': {'path': CERTIFIED_MODELS_X0_REL, 'sha256': got, 'manifest_sha256': want},
            'model_fingerprint_before': before_m, 'model_fingerprint_after': after_m, 'components': counts,
            'planning_fingerprint_equal': before_p == after_p, 'dual_vars_equal': before_dv == hdv.hexdigest(),
            'consensus_vars_equal': before_cv == hcv.hexdigest(), 'spot_check_block_7_equal': spot_ok,
            'admm_objective_scale_mutable_tso_dso': mutable,
            'shape': {'n_blocks': header['n_blocks'], 'n_periods': header['n_periods'], 'blocks_first3': header['blocks'][:3],
                      'nodes': sorted({b[0] for b in header['blocks']}), 'years': sorted({b[1] for b in header['blocks']}),
                      'days': sorted({b[2] for b in header['blocks']}, key=str),
                      'dual_vars_leaf': "dual_vars['pf'][agent]['current'][node][year][day]['p'|'q'] = list of 24 floats",
                      'fields': header['fields'], 'floats_per_cycle_line': 4 * header['n_blocks'] * header['n_periods']},
            'metadata_first_block': header['metadata']['per_block'][0],
            'sigma_fixed': header['metadata']['sigma_fixed_admm_parameters_objective_scale'],
            'bytes': {'header': line_bytes[0], 'cycle_line_full_precision_uniform_pm5e3': line_bytes[1]},
            'work_dir_created_for_fresh_planning': os.path.relpath(work, REPO)}
    except Exception as error:  # noqa: BLE001
        res['L3_write_only_real_srp1'] = {'holds': False, 'error': f'{type(error).__name__}: {error}',
                                          'traceback': traceback.format_exc()}
    finally:
        if os.path.isdir(work) and not any(files for _r, _d, files in os.walk(work)):
            shutil.rmtree(work)   # fresh_planning's empty log dir, created by this check only
    # L4 size per cycle at SRP1 and at 3 x 3 (full-precision values, the same writer)
    def line_bytes(n_blocks, periods, seed):
        r = random.Random(seed)
        line = {'cycle': 150, 'captured': True, 'error': None, 'capture_s': 0.0123456789}
        for _g, _a, _k, f in IDC.CHANNEL_FIELDS:
            line[f] = [[r.uniform(-5e3, 5e3) for _ in range(periods)] for _ in range(n_blocks)]
        return len(GRIO.dumps(line, default=GRIO.json_default).encode()) + 1
    srp1_blocks = (res.get('L3_write_only_real_srp1', {}).get('shape') or {}).get('n_blocks') or 36
    b_srp1 = line_bytes(srp1_blocks, 24, 1)
    b_3x3 = line_bytes(SHAPE_3X3['nodes'] * SHAPE_3X3['years'] * SHAPE_3X3['days'], SHAPE_3X3['periods'], 2)
    res['L4_size_per_cycle'] = {
        'holds': True, 'basis': ('JSON bytes of one cycle line with every dual a full-precision float (uniform '
                                 '+-5e3; json repr ~17-19 significant characters); exact duals may be shorter '
                                 '(e.g. 0.0 before the first update); header written once'),
        'srp1': {'n_blocks': srp1_blocks, 'floats': 4 * srp1_blocks * 24, 'bytes_per_cycle': b_srp1,
                 'x0_worst_case_232_cycles_MiB': 232 * b_srp1 / 2 ** 20},
        '3x3': {'shape': SHAPE_3X3, 'n_blocks': 60, 'floats': 4 * 60 * 24, 'bytes_per_cycle': b_3x3,
                'per_100_cycles_MiB': 100 * b_3x3 / 2 ** 20}}
    return res


# ======================================================================================================================
#  H -- the hooks
# ======================================================================================================================
def _standin_originals(scripts):
    """Stand-in originals returning scripted values; each records its calls."""
    calls = {}

    def log(name, *a, **k):
        calls.setdefault(name, []).append((a, k))

    def baseline(pp, admm):
        log('baseline', pp, admm)
        return 'baseline-return'

    def apply(pp, admm, active, base, cycle):
        log('apply', pp, admm, active, base, cycle)
        return {'active': bool(active), 'cycle': cycle}

    def aa(aa_state, layout, cv, dv, wb, rho, boyd, it):
        log('aa', aa_state, layout, cv, dv, wb, rho, boyd, it)
        return {'cycle': it, 'action': C.AA_OFF_ACTION if boyd['all_boyd_pass'] else 'accepted'}

    def nxt(conv, aa_enabled, aa_record):
        log('next', conv, aa_enabled, aa_record)
        return bool(conv)

    def pen(tso, dso, esso, rm, bm, params, iter=None, allow_update=True, freeze_state=None):
        log('pen', iter, allow_update)
        return scripts['pen'].pop(0)

    def recourse(pp, models):
        log('recourse', pp, models)
        return scripts['rc'].pop(0)

    def efc(esso):
        log('efc', esso)
        return scripts['efc'].pop(0)

    return {'_capture_convergence_depth_tail_baseline': baseline, '_apply_convergence_depth_tail': apply,
            '_anderson_acceleration_cycle_step': aa, '_convergence_depth_tail_next_state': nxt,
            '_update_admm_penalties': pen, '_get_operational_recourse_components': recourse,
            '_get_admm_efc_per_day_max': efc}, calls


def _drive(w, c, *, boyd_metrics, local_ok, next_conv, active, admm):
    """One emulated production cycle through the wrappers, in production's call order: apply -> (AA -> recourse if
    every local solve succeeded) -> next-state -> penalties -> EFC."""
    w['_apply_convergence_depth_tail'](object(), admm, active, object(), c)
    if local_ok:
        aa_rec = w['_anderson_acceleration_cycle_step'](object(), None, {}, {}, None, {}, boyd_metrics, c)
        w['_get_operational_recourse_components'](object(), {'cycle': c})
    else:
        aa_rec = {'cycle': c, 'action': 'skipped (local solve failure this cycle)'}
    w['_convergence_depth_tail_next_state'](bool(boyd_metrics['all_boyd_pass'] and local_ok), True, aa_rec)
    w['_update_admm_penalties']({}, {}, {}, {}, boyd_metrics, object(), iter=c, allow_update=local_ok,
                                freeze_state={})
    w['_get_admm_efc_per_day_max'](object())


def _row_objects(row):
    """Production's objects for one cycle, rebuilt from a RECORDED raw row (g_s39_D.json): the boyd_metrics dict,
    the recourse components, the penalty-update return value and the EFC value."""
    bm = {'all_boyd_pass': row['boyd_all_pass']}
    for g in C.CHANNELS:
        bm[g] = {'primal_ratio': row[f'boyd_{g}_primal_ratio'], 'dual_ratio': row[f'boyd_{g}_dual_ratio'],
                 'channel_pass': row[f'boyd_{g}_channel_pass'], 'r': row.get(f'boyd_{g}_r'), 's': row.get(f'boyd_{g}_s')}
    rc = {'gross_operational_cost': row['gross_operational_cost'], 'net_operational_recourse': row['recourse'],
          'terminal_salvage_value': row['terminal_salvage_value']}
    pen = ({g: row[f'rho_{g}_action'] for g in C.CHANNELS}, {g: row[f'rho_{g}_before'] for g in C.CHANNELS},
           {g: row[f'rho_{g}_after'] for g in C.CHANNELS}, {g: row.get(f'gamma_{g}_before') for g in C.CHANNELS},
           {g: row.get(f'gamma_{g}_after') for g in C.CHANNELS}, row['rho_freeze_active'],
           {g: {'frozen': row.get(f'rho_frozen_{g}')} for g in C.CHANNELS})
    return bm, rc, pen, row['efc_per_day_max']


def _state_for(cell, p_max, reference, cap=None):
    decl = C.declaration_for(cell, p_max)
    sink = []
    st = C.ContinuationState(decl, None, cap or (decl['hold_after_cycle'] + 100), reference=reference, sink=sink)
    return st, sink, decl


def tests_H(p_max):
    import shared_resources_planning as srp
    res = {}
    # ---- H5: recorded replay through the real wrappers, then continuation ------------------------------------------
    h5 = {}
    for cell in C.CELL_ORDER:
        decl = C.declaration_for(cell, p_max)
        reference = C.load_replay_reference(decl)
        raw = {r['cycle']: r for r in json.load(open(_abs(os.path.join(
            C.RECERT_ROOT, 'evals', C.CELLS[cell]['eval_dir'], 'g_s39_D.json'))))['cycle_trajectory']}
        n = decl['hold_after_cycle']
        cap = n + 100
        out = {}
        for variant in ('bitwise_then_certify', 'one_ulp_at_40', 'lapse_then_cap'):
            scripts = {'pen': [], 'rc': [], 'efc': []}
            orig, calls = _standin_originals(scripts)
            st, sink, _ = _state_for(cell, p_max, reference, cap=cap)
            w = C.make_wrappers(st, orig)
            admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
            w['_capture_convergence_depth_tail_baseline'](object(), admm)
            raised = None
            q_n = raw[n]['gross_operational_cost']
            direction = 1.0 if raw[n]['gross_operational_cost'] >= raw[n - 1]['gross_operational_cost'] else -1.0
            try:
                for c in range(1, cap + 1):
                    if c <= n:
                        bm, rc, pen, efc = _row_objects(raw[c])
                        if variant == 'one_ulp_at_40' and c == 40:
                            rc = dict(rc)
                            rc['gross_operational_cost'] = math.nextafter(rc['gross_operational_cost'], math.inf)
                        local_ok = raw[c]['local_solves_ok']
                    else:
                        j = c - n
                        if variant == 'lapse_then_cap':
                            q = q_n - 80.0 * j            # constant creep: never certifies
                            boyd = not (j == 5)
                        else:
                            # continue in the direction of the last recorded step with geometrically decaying steps:
                            # no new sign change, so the MONOTONE branch must certify once its window clears the last
                            # recorded sign change and its range is <= tau
                            q = q_n + direction * 2000.0 * (1.0 - 0.9 ** j)
                            boyd = True
                        bm = {'all_boyd_pass': boyd, **{g: {'primal_ratio': 0.5, 'dual_ratio': 0.5, 'channel_pass': True,
                                                            'r': 1.0, 's': 1.0} for g in C.CHANNELS}}
                        rc = {'gross_operational_cost': q, 'net_operational_recourse': q, 'terminal_salvage_value': 0.0}
                        _a, _b, after, _c, _d, _e, _f = _row_objects(raw[n])[2]
                        pen = ({g: 'held (frozen after 10 unchanged cycles)' for g in C.CHANNELS}, dict(after),
                               dict(after), {g: 0.0 for g in C.CHANNELS}, {g: 0.0 for g in C.CHANNELS}, True,
                               {g: {'frozen': True} for g in C.CHANNELS})
                        efc = None
                        local_ok = True
                    scripts['pen'].append(pen)
                    if local_ok:
                        scripts['rc'].append(rc)
                    scripts['efc'].append(efc)
                    with contextlib.redirect_stdout(io.StringIO()):
                        _drive(w, c, boyd_metrics=bm, local_ok=local_ok, next_conv=None, active=c > n, admm=admm)
                    if admm.minimum_consecutive_converged_cycles == C.SETTLING_CERTIFIED_THRESHOLD:
                        break
                w['_apply_convergence_depth_tail'](object(), admm, False, object(), None)
            except RuntimeError as error:
                raised = str(error)
            lines = [o for f, o in sink if f == C.CYCLE_FILE]
            decisions = [o for f, o in sink if f == C.DECISION_FILE]
            summ = st.summary()
            # the in-cycle rule state equals the pure replay over the same (Q, boyd)
            qd = {x['cycle']: x.get('gross') for x in lines}
            bd = {x['cycle']: bool(x.get('boyd_k')) for x in lines}
            pure_rule = SC.SettlingRule(n, cap, p_max)      # the same n and cap as the in-cycle rule
            pure_out = [pure_rule.observe(k, qd[k], bd[k]) for k in sorted(qd)]
            pure_dec = pure_rule.decision
            in_cycle = [x['settling'] for x in lines]
            pure_equal = [json.dumps(a, sort_keys=True, default=str) for a in in_cycle] == \
                [json.dumps(b_, sort_keys=True, default=str) for b_ in pure_out[:len(in_cycle)]]
            entry = {'raised': raised, 'lines': len(lines), 'replay_bitwise_through': summ['replay_bitwise_through_cycle'],
                     'first_divergence': summ['replay_first_divergence'], 'stopped_by': summ['stopped_by'],
                     'settling_status': summ['settling_status'], 'k_star': summ['k_star'], 'branch': summ['branch'],
                     'decision_file_lines': len(decisions), 'summary_ok': summ['ok'],
                     'certificate_length_after': admm.minimum_consecutive_converged_cycles,
                     'in_cycle_rule_equals_pure_replay': pure_equal,
                     'pure_replay_status': (pure_dec or {}).get('status'),
                     'lapse_events': summ['lapse_events'],
                     'holds_after_N': sorted({json.dumps(x['holds'], sort_keys=True) for x in lines if x['cycle'] > n}),
                     'holds_through_N': sorted({json.dumps(x['holds'], sort_keys=True) for x in lines if x['cycle'] <= n}),
                     'pf_primal_recorded_every_line': all('boyd_pf_primal_ratio' in x for x in lines),
                     'aa_calls_forced_after_N': sum(1 for a, _k in calls.get('aa', []) if a[7] > n
                                                    and a[6]['all_boyd_pass'] is True),
                     'pen_allow_update_after_N': sorted({k_ for i_, k_ in [(a[0], a[1]) for a, _k in calls.get('pen', [])]
                                                         if i_ > n})}
            if variant == 'bitwise_then_certify':
                entry['ok'] = (raised is None and summ['replay_bitwise_through_cycle'] == n
                               and summ['settling_status'] == 'certified' and summ['stopped_by'] == 'settling_rule'
                               and summ['k_star'] is not None and summ['k_star'] > n and len(decisions) == 1
                               and summ['ok'] and pure_equal and (pure_dec or {}).get('k_star') == summ['k_star']
                               and admm.minimum_consecutive_converged_cycles == 10 and st.phase == 'ended'
                               and entry['holds_through_N'] == [json.dumps({'aa': False, 'rho': False, 'tail_apply': False,
                                                                            'tail_next': False}, sort_keys=True)]
                               and entry['holds_after_N'] == [json.dumps({'aa': True, 'rho': True, 'tail_apply': True,
                                                                          'tail_next': True}, sort_keys=True)]
                               and entry['pen_allow_update_after_N'] == [False])
            elif variant == 'one_ulp_at_40':
                entry['ok'] = (raised is not None and 'REPLAY DIVERGED at cycle 40' in raised and len(lines) == 40
                               and (summ['replay_first_divergence'] or {}).get('fields_differing')
                               == ['gross_operational_cost'] and summ['replay_bitwise_through_cycle'] == 39
                               and not summ['ok'] and summ['stopped_by'] == 'replay_divergence_abort'
                               and lines[-1].get('replay_equal') is False)
            else:
                entry['ok'] = (raised is None and summ['settling_status'] == 'uncertified' and len(lines) == cap
                               and len(summ['lapse_events']) == 1 and summ['lapse_events'][0]['cycle'] == n + 5
                               and summ['stopped_by'] == 'cap' and len(decisions) == 1 and summ['ok'] and pure_equal
                               and 'monotone_not_decreasing' in (st.decision or {}).get('reasons', []))
                entry['uncertified_reasons'] = (st.decision or {}).get('reasons')
            out[variant] = entry
        h5[cell] = {'N': n, 'cap': cap, 'variants': out, 'ok': all(v['ok'] for v in out.values())}
    res['H5_recorded_replay_gate_and_settling_decision'] = {'holds': all(v['ok'] for v in h5.values()), 'cells': h5}

    # ---- H1: AA hold, stand-in identity ------------------------------------------------------------------------------
    st, _sink, decl = _state_for('c_star', p_max, {})
    n = decl['hold_after_cycle']
    seen = []

    def aa_orig(aa_state, layout, cv, dv, wb, rho, boyd, it):
        seen.append((aa_state, layout, cv, dv, wb, rho, boyd, it))
        return {'cycle': it, 'action': C.AA_OFF_ACTION if boyd['all_boyd_pass'] else 'accepted'}
    w = C.make_wrappers(st, {'_anderson_acceleration_cycle_step': aa_orig, **{k: None for k in C.WRAPPED
                                                                             if k != '_anderson_acceleration_cycle_step'}})
    inert, acts = True, True
    for c in (n - 1, n, n + 1, n + 2):
        st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
        boyd = {'all_boyd_pass': False, **{g: {'primal_ratio': 2.0, 'dual_ratio': 3.0} for g in C.CHANNELS}}
        snap = copy.deepcopy(boyd)
        a = (object(), 'layout', {}, {}, object(), {}, boyd, c)
        rec = w['_anderson_acceleration_cycle_step'](*a)
        got = seen[-1]
        if c <= n:
            inert = inert and all(x is y for x, y in zip(got, a)) and rec['action'] == 'accepted' \
                and st.cur['boyd_k'] is False
        else:
            acts = acts and got[6] is not boyd and got[6]['all_boyd_pass'] is True and boyd == snap \
                and rec['action'] == C.AA_OFF_ACTION and st.cur['boyd_k'] is False \
                and st.cur['aa']['natural_all_boyd_pass'] is False
    res['H1_aa_hold_standin'] = {'holds': inert and acts, 'inert_through_N': inert, 'acts_after_N': acts,
                                 'boyd_k_is_the_natural_value': True}

    # ---- H2: AA hold with REAL production ----------------------------------------------------------------------------
    res['H2_aa_hold_real_production'] = _h2_aa_real(p_max)
    # ---- H3: tail hold with REAL production ---------------------------------------------------------------------------
    res['H3_tail_hold_real_production'] = _h3_tail_real(p_max, srp)
    # ---- H4: rho hold with REAL production ----------------------------------------------------------------------------
    res['H4_rho_hold_real_production'] = _h4_rho_real(p_max, srp)
    # ---- H6: layering, real install -----------------------------------------------------------------------------------
    res['H6_layering_real_install'] = _h6_layering(p_max, srp)
    # ---- H7: certificate length written only by disable / restore / settling decision ---------------------------------
    got = C._certificate_length_writes_in_source()
    tampered = sorted(got + ['st.params.minimum_consecutive_converged_cycles = 0'])
    res['H7_certificate_length_writes'] = {
        'holds': got == sorted(C.CERTIFICATE_LENGTH_WRITES) and tampered != sorted(C.CERTIFICATE_LENGTH_WRITES),
        'writes_in_source': got, 'expected': sorted(C.CERTIFICATE_LENGTH_WRITES),
        'negative_control_extra_write_detected': tampered != sorted(C.CERTIFICATE_LENGTH_WRITES),
        'no_early_stop_symbol_in_hooks_source': 'early_stop' not in inspect.getsource(C.make_wrappers)}
    res['H7_certificate_length_writes']['holds'] = (res['H7_certificate_length_writes']['holds']
                                                    and res['H7_certificate_length_writes'][
                                                        'no_early_stop_symbol_in_hooks_source'])
    return res


def _h2_aa_real(p_max):
    import numpy as np
    import admm_anderson_acceleration as aam
    import shared_resources_planning as srp
    dim = 6
    rng = np.random.default_rng(20260927)
    A = 0.9 * np.eye(dim) + 0.02 * rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)
    real_collect, real_write = aam.collect_w, aam.write_back_w
    st0, _s, decl = _state_for('c_star', p_max, {})
    n = decl['hold_after_cycle']

    def collect(layout, cv, dv, rho, check_antisymmetry=True):
        return np.array(cv['w'], dtype=float)

    def write_back(layout, w, cv, dv, rho):
        cv['w'] = np.array(w, dtype=float)

    def run(wrapped):
        st, _sink, _d = _state_for('c_star', p_max, {})
        w = C.make_wrappers(st, {'_anderson_acceleration_cycle_step': srp._anderson_acceleration_cycle_step,
                                 **{k: None for k in C.WRAPPED if k != '_anderson_acceleration_cycle_step'}})
        step = w['_anderson_acceleration_cycle_step'] if wrapped else srp._anderson_acceleration_cycle_step
        state = aam.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')
        cv = {'w': np.zeros(dim)}
        trace = []
        for c in range(1, n + 5):
            st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
            w_before = np.array(cv['w'])
            cv['w'] = A @ cv['w'] + b * (1.0 + 0.1 * np.sin(c))
            r = 1.0 / c
            boyd = {'all_boyd_pass': False, **{g: {'r': r, 's': r, 'primal_ratio': r, 'dual_ratio': r}
                                               for g in C.CHANNELS}}
            rec = step(state, None, cv, {}, w_before, {'v': 1.0, 'pf': 1.0, 'ess': 1.0}, boyd, c)
            trace.append({'cycle': c, 'action': rec['action'], 'w_hex': [float.hex(float(x)) for x in cv['w']],
                          'boyd_seen_by_caller': boyd['all_boyd_pass']})
        return trace

    aam.collect_w, aam.write_back_w = collect, write_back
    try:
        ref = run(False)
        wrp = run(True)
    finally:
        aam.collect_w, aam.write_back_w = real_collect, real_write
    restored = aam.collect_w is real_collect and aam.write_back_w is real_write
    inert = all(ref[i] == wrp[i] for i in range(n))
    after = [(ref[i]['action'], wrp[i]['action'], ref[i]['w_hex'] != wrp[i]['w_hex']) for i in range(n, n + 4)]
    acts = (all(a == 'accepted' and bw == C.AA_OFF_ACTION for a, bw, _d in after) and all(d for _a, _b, d in after)
            and all(not t['boyd_seen_by_caller'] for t in wrp))
    n_acc = sum(1 for t in ref[:n] if t['action'] == 'accepted')
    return {'holds': inert and acts and restored and n_acc > 0, 'inert_through_N_bitwise': inert, 'acts_after_N': acts,
            'n_accepted_extrapolations_through_N': n_acc, 'N': n, 'aam_restored': restored,
            'after_N': [{'cycle': n + 1 + i, 'unwrapped': a, 'wrapped': bw, 'iterates_differ': d}
                        for i, (a, bw, d) in enumerate(after)]}


def _h3_tail_real(p_max, srp):
    names = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
             '_convergence_depth_tail_next_state')
    real = {k: getattr(srp, k) for k in names}
    _st, _s, decl = _state_for('c_star', p_max, {})
    n = decl['hold_after_cycle']

    def predicate(c):
        return 78 <= c <= n            # the recorded c_star pattern: Boyd pass from 78; False after N (the held case)

    def run(wrapped):
        st, _sink, _d = _state_for('c_star', p_max, {})
        w = C.make_wrappers(st, dict(real, **{k: None for k in C.WRAPPED if k not in real}))
        fn = w if wrapped else real
        pp = K98._fake_holders()
        admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                               minimum_consecutive_converged_cycles=10)
        base = fn['_capture_convergence_depth_tail_baseline'](pp, admm)
        active_next = False
        trace = []
        for c in range(1, n + 5):
            if wrapped and st.cur is not None:
                st.cur['finalized'] = True     # the tail-only drive has no penalty / EFC calls (tested in H5)
            rec = fn['_apply_convergence_depth_tail'](pp, admm, active_next, base, c)
            opts = K98._holder_options(pp)
            conv = predicate(c)
            aa_rec = {'cycle': c, 'action': C.AA_OFF_ACTION if conv else 'accepted'}
            active_next = fn['_convergence_depth_tail_next_state'](conv, True, aa_rec)
            trace.append({'cycle': c, 'apply_record': rec, 'options_after_apply': opts, 'next': bool(active_next)})
        return trace, admm

    ref, _a = run(False)
    wrp, admm_w = run(True)
    inert = all(ref[i] == wrp[i] for i in range(n))
    after = wrp[n:]
    acts = (all(t['apply_record']['active'] is True for t in after)
            and all(set(v['compl_inf_tol'] for v in t['options_after_apply'].values()) == {1e-6} for t in after)
            and all(t['next'] is True for t in after) and any(r['apply_record']['active'] is False for r in ref[n + 1:]))
    return {'holds': inert and acts and admm_w.minimum_consecutive_converged_cycles == C.CERTIFICATION_DISABLED_THRESHOLD,
            'inert_through_N': inert, 'acts_after_N': acts, 'N': n,
            'unwrapped_tail_after_N': [r['apply_record']['active'] for r in ref[n:]],
            'wrapped_tail_after_N': [t['apply_record']['active'] for t in after],
            'certificate_length_after_baseline': admm_w.minimum_consecutive_converged_cycles}


def _h4_rho_real(p_max, srp):
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    admm = params.admm
    real = srp._update_admm_penalties
    _st, _s, decl = _state_for('c_star', p_max, {})
    n = decl['hold_after_cycle']

    def metrics(c):
        hi = (c % 2 == 1)
        rm = {'primal': {g: 1.0 for g in C.CHANNELS} | {f'{g}_mean': 1.0 for g in C.CHANNELS},
              'dual': {f'{g}_mean': 1.0 for g in C.CHANNELS}}
        bm = {g: {'primal_ratio': 50.0 if hi else 0.5, 'dual_ratio_balance': 0.5 if hi else 50.0, 'dual_ratio': 0.5,
                  'r': 1.0, 's': 1.0, 'eps_pri': 1.0, 'eps_dual': 1.0} for g in C.CHANNELS}
        return rm, bm

    def run(wrapped):
        st, _sink, _d = _state_for('c_star', p_max, {})
        w = C.make_wrappers(st, dict({'_update_admm_penalties': real},
                                     **{k: None for k in C.WRAPPED if k != '_update_admm_penalties'}))
        tso, dso, esso = K98._rho_models()
        fs = srp._init_admm_freeze_state()
        trace = []
        with contextlib.redirect_stdout(io.StringIO()):
            for c in range(1, n + 5):
                st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
                rm, bm = metrics(c)
                fn = w['_update_admm_penalties'] if wrapped else real
                actions, before, after, bg, ag, rfa, fs = fn(tso, dso, esso, rm, bm, admm, iter=c, allow_update=True,
                                                             freeze_state=fs)
                trace.append({'cycle': c, 'actions': dict(actions), 'rho_hex': K98._rho_read(tso, dso, esso),
                              'freeze_state': copy.deepcopy(fs), 'changed': before != after})
        return trace

    ref = run(False)
    wrp = run(True)
    inert = all(ref[i] == wrp[i] for i in range(n))
    moved = sum(1 for t in ref[:n] if t['changed'])
    after_ref = [t['changed'] for t in ref[n:]]
    after_wrp = [t['changed'] for t in wrp[n:]]
    frozen = all(wrp[i]['rho_hex'] == wrp[n - 1]['rho_hex'] for i in range(n, n + 4))
    acts = all(after_ref) and not any(after_wrp) and frozen
    return {'holds': inert and acts and moved > 0, 'inert_through_N_bitwise': inert, 'acts_after_N': acts, 'N': n,
            'cycles_with_rho_change_through_N': moved, 'unwrapped_changed_after_N': after_ref,
            'wrapped_changed_after_N': after_wrp, 'wrapped_rho_after_N_equals_rho_at_N_bitwise': frozen}


def _h6_layering(p_max, srp):
    before = {name: getattr(srp, name) for name in C.WRAPPED + ('_drain_network_ipopt_solve_records',
                                                                 'get_admm_boyd_residual_metrics')}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w101_layering_')
    decl = C.declaration_for('c_star', p_max)
    n = decl['hold_after_cycle']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with C.settling_continuation_hooks(scratch, decl, holder, cap=n + 100) as st:
            with IDC.interface_dual_capture_hooks(scratch, holder):
                with H.convergence_depth_append_hooks(stub):
                    base = srp._capture_convergence_depth_tail_baseline(pp, admm)
                    active = False
                    for c in range(1, n + 3):
                        if st.cur is not None:
                            st.cur['finalized'] = True
                        srp._apply_convergence_depth_tail(pp, admm, active, base, c)
                        conv = c <= n
                        active = srp._convergence_depth_tail_next_state(
                            conv if c <= n else False, True,
                            {'cycle': c, 'action': C.AA_OFF_ACTION if (conv or c > n) else 'accepted'})
                    st.cur['finalized'] = True
                    srp._apply_convergence_depth_tail(pp, admm, active, base, None)
        files_written = sorted(os.listdir(scratch))
    finally:
        shutil.rmtree(scratch)
    after = {name: getattr(srp, name) for name in before}
    restored = all(after[k] is before[k] for k in before)
    next_events = [e[1]['value'] for e in stub.events if e[0] == 'next_state']
    apply_events = [e[1] for e in stub.events if e[0] == 'apply']
    appender_saw_held = (next_events[n] is True and apply_events[n + 1]['record']['active'] is True)
    child_src = inspect.getsource(H._child_real)
    i_cont = child_src.find('with continuation_cm, \\')
    i_settle = child_src.find('settling_cm, \\', i_cont)
    i_idc = child_src.find('IDC.interface_dual_capture_hooks(eval_dir, holder) as dual_capture', i_cont)
    i_s38 = child_src.find('G.s38_pf_capture_hooks(', i_cont)
    i_app = child_src.find('convergence_depth_append_hooks(appender)', i_cont)
    order_ok = 0 < i_cont < i_settle < i_idc < i_s38 < i_app
    pre_ok = (child_src.find('W101C.assert_settling_preconditions(settling, spec, tail_checklist, aa_on)')
              < child_src.find('G.run_admm_arm(') and child_src.find('IDC.assert_capture_path()')
              < child_src.find('G.run_admm_arm('))
    summ = holder.get(C.SUMMARY_KEY) or {}
    return {'holds': (appender_saw_held and restored and order_ok and pre_ok and summ.get('phase') == 'ended'
                      and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                      and admm.minimum_consecutive_converged_cycles == 10),
            'appender_recorded_held_tail_value_after_N': appender_saw_held,
            'production_functions_restored_on_exit': restored,
            'child_real_order_continuation_settling_idc_s38_appender': order_ok,
            'child_real_preconditions_before_run_admm_arm': pre_ok,
            'files_written_by_install_without_cycles': files_written,
            'summary_phase': summ.get('phase'), 'summary_errors': summ.get('errors')}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w101():
    import importlib.util
    import subprocess
    src = subprocess.run(['git', 'show', f"{PRE_W101_HARNESS['commit']}:p515_s44_campaign_harness.py"], cwd=REPO,
                         capture_output=True, check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != PRE_W101_HARNESS['sha256']:
        raise RuntimeError(f'pre-W101 harness sha256 {sha} != pinned {PRE_W101_HARNESS["sha256"]}')
    tmp = tempfile.mkdtemp(prefix='w101_pre_harness_')
    path = os.path.join(tmp, '_w101_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w101_pre_harness', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(tmp)
    return mod, sha


def _key_args(spec, e):
    cfg = spec.get('configuration') or {}
    overrides = e.get('overrides') if 'overrides' in e else (cfg.get('overrides') or {})
    return (e['key'], overrides), dict(
        case_file_aa=cfg.get('case_file_anderson_acceleration'), model_variant=e.get('model_variant'),
        ess_ageing_baseline=cfg.get('ess_ageing_baseline'), flex_price_multiplier=e.get('flex_price_multiplier'),
        derived_instance=cfg.get('derived_instance'), interface_deviation_premium=e.get('interface_deviation_premium'),
        convergence_depth_tail=cfg.get('convergence_depth_tail'),
        certification_continuation=e.get('certification_continuation'))


def continuation_keys(p_max):
    recert = json.load(open(_abs(os.path.join(C.RECERT_ROOT, 'campaign_spec_s53_w86_tail_recert_ddd6cd44.json'))))
    cfg = recert['configuration']
    out = {}
    for e in recert['candidates']:
        kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
                  convergence_depth_tail=cfg['convergence_depth_tail'])
        base = H.evaluation_key(e['key'], e['overrides'], **kw)
        decl = C.declaration_for(e['label'], p_max)
        cont = H.evaluation_key(e['key'], e['overrides'], settling_continuation=decl, **kw)
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': e['eval_key'], 'settling_continuation': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[e['label']] = {'recert_eval_key': e['eval_key'], 'base_key_now': base, 'continuation_key': cont,
                           'base_equals_recert': base == e['eval_key'], 'formula_holds': cont == formula,
                           'differs_from_recert': cont != e['eval_key']}
    return out


def tests_K(p_max):
    pre, pre_sha = _harness_pre_w101()
    n_specs = n_entries = n_equal = n_settling = n_settling_ok = 0
    mismatch, errors, settling_bad = [], [], []
    committed = {}
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        for e in spec.get('candidates') or []:
            n_entries += 1
            committed.setdefault(H._entry_eval_key(e), []).append(rel)
            try:
                args, kw = _key_args(spec, e)
                if 'settling_continuation' in e:
                    n_settling += 1
                    base_new = H.evaluation_key(*args, **kw)
                    base_old = pre.evaluation_key(*args, **kw)
                    new = H.evaluation_key(*args, settling_continuation=e['settling_continuation'], **kw)
                    formula = hashlib.sha256(json.dumps(
                        {'base_evaluation_key': base_old,
                         'settling_continuation': C.validate_settling_continuation(e['settling_continuation'])},
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    ok = (rel.startswith(W101_ROOT_REL + os.sep) and base_new == base_old and new == formula
                          and e.get('eval_key') == new)
                    n_settling_ok += int(ok)
                    if not ok:
                        settling_bad.append({'spec': rel, 'label': e.get('label')})
                    continue
                new, old = H.evaluation_key(*args, **kw), pre.evaluation_key(*args, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    cont = continuation_keys(p_max)
    outside = {lab: [r for r in committed.get(v['continuation_key'], []) if not r.startswith(W101_ROOT_REL + os.sep)]
               for lab, v in cont.items()}
    holds = (not mismatch and not errors and not settling_bad and n_equal == n_entries - n_settling
             and n_settling_ok == n_settling and all(v['base_equals_recert'] and v['formula_holds']
                                                     and v['differs_from_recert'] for v in cont.values())
             and not any(outside.values()))
    return {'holds': holds, 'pre_w101_harness': {**PRE_W101_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries,
            'entries_new_equals_pre_w101': n_equal, 'mismatches': mismatch, 'errors': errors,
            'settling_entries': {'n': n_settling, 'ok': n_settling_ok, 'bad': settling_bad},
            'continuation_keys': cont, 'continuation_keys_in_committed_specs_outside_w101_root': outside}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P(p_max):
    out = {}
    ok = True
    for cell in C.CELL_ORDER:
        decl = C.declaration_for(cell, p_max)
        n = decl['hold_after_cycle']
        try:
            good = C.assert_settling_preconditions(decl, {'cap': n + 100}, {'tail_enabled_for_this_run': True}, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        negatives = {}
        for name, (spec, tail, aa) in {'cap_N_plus_30': ({'cap': n + 30}, {'tail_enabled_for_this_run': True}, True),
                                       'tail_off': ({'cap': n + 100}, {'tail_enabled_for_this_run': False}, True),
                                       'aa_off': ({'cap': n + 100}, {'tail_enabled_for_this_run': True}, False)}.items():
            try:
                C.assert_settling_preconditions(decl, spec, tail, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {error}'
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0, 'consecutive_cycles': 3}),
               'wrong_N': dict(decl, hold_after_cycle=n - 1),
               'wrong_sha': dict(decl, replay_reference=dict(decl['replay_reference'], sha256='0' * 64)),
               'extra_key': dict(decl, extra=1),
               'l_mono_not_2_p_max': dict(decl, settling_rule=dict(decl['settling_rule'], l_mono=43)),
               'abort_false': dict(decl, abort_on_replay_divergence=False),
               'cap_after_hold_30': dict(decl, cap_after_hold=30),
               'tau_changed': dict(decl, settling_rule=dict(decl['settling_rule'], tau=9078.14))}
        refused = {}
        for name, d in bad.items():
            try:
                C.validate_settling_continuation(d)
                refused[name] = False
            except ValueError as error:
                refused[name] = str(error)[:160]
        cell_ok = good_ok and all(v.startswith('refused') for v in negatives.values()) and all(refused.values())
        ok = ok and cell_ok
        out[cell] = {'ok': cell_ok, 'checklist': good, 'negative_controls': negatives, 'validator_refuses': refused}
    return {'holds': ok, 'cells': out}


# ======================================================================================================================
def run_all_checks():
    out = {}
    try:
        p_max, p_detail = compute_p_max()
    except Exception as error:  # noqa: BLE001
        return {'all_hold': False, 'error': f'P_MAX: {type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    out['P_MAX'] = {'value': p_max, 'function': 'settling_criterion.p_max_from_records', 'advisor_hand_derived': ADVISOR_P_MAX,
                    'advisor_without_floor': 25, 'equals_advisor': p_max == ADVISOR_P_MAX, 'detail': p_detail,
                    'without_floor_this_function': {
                        n: (lambda T: max((T[i + 2][0] - T[i][0]) for i in range(len(T) - 2)) if len(T) >= 3 else None)(
                            SC.turning_points(load_record(n)[0], 1, RECORDS[n]['N'], eps0=0.0)['T'])
                        for n in SRP1_RECORD_NAMES}}
    sections = (('T', tests_T), ('R', tests_R), ('L', lambda _p: tests_L()), ('H', tests_H), ('K', tests_K),
                ('P', tests_P))
    ok = True
    for sid, fn in sections:
        t0 = time.time()
        try:
            r = fn(p_max)
        except Exception as error:  # noqa: BLE001 -- recorded as a failing section
            r = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        r_ok = _section_ok(sid, r)
        out[sid] = {'holds': r_ok, 'wall_s': time.time() - t0, 'result': r}
        ok = ok and r_ok
    return {'all_hold': ok, 'p_max': p_max, 'constants': SC.constants(p_max), 'readings': SC.READINGS,
            'sections': out, 'P_MAX': out['P_MAX']}


def _section_ok(sid, r):
    if 'error' in r and len(r) <= 2:
        return False
    if sid == 'T':
        return all(v.get('ok') is True for v in r.values())
    if sid in ('R', 'K', 'P'):
        return r.get('holds') is True
    return all(v.get('holds') is True for v in r.values())


def main():
    started = _utc()
    out_dir = _abs(OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_FILE, OUT_MANIFEST):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    res = run_all_checks()
    code_pins = {rel: H.sha256_file(_abs(rel)) for rel in (
        os.path.basename(__file__), 'settling_criterion.py', 'interface_dual_capture.py',
        'p515_s53_w101_settling_continuation_hooks.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py',
        'shared_resources_planning.py', 'admm_anderson_acceleration.py', 'p515_s53_w98_continuation_checks.py')}
    guards = {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}
    doc = {'schema': 'p515_s53_w101_zero_solve_checks_v1', 'task': 'W101 (PLANNER_BRIEF_2026-09-13.md Addendum 53)',
           'started_utc': started, 'finished_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
           'code_sha256': code_pins, 'guards': guards, **res}
    path = os.path.join(out_dir, OUT_FILE)
    with open(path, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    manifest = {os.path.relpath(path, REPO): H.sha256_file(path)}
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    print(f"[W101-CHECKS] P_MAX = {res.get('p_max')} (advisor 22)")
    for sid, r in (res.get('sections') or {}).items():
        if sid == 'P_MAX':
            continue
        print(f"[W101-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W101-CHECKS] all_hold={res.get('all_hold')} guards={guards}")
    print(f"[W101-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (res.get('all_hold') and guards_ok) else 1)


if __name__ == '__main__':
    main()
