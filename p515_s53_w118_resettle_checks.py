"""
P5.15 Addendum 57 (Decisions 2 and 3) / Addendum 54 Ruling 2, Planner task W118 -- ZERO-SOLVE checks for the SRP1
re-settling campaign (frozen_s53_resettle_spec_v1).

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W105 / W101 /
W98 checks modules and the W112 script, imported for their helpers, arm their own; every guard is verified at 0).
No model is built and nothing is solved: the t_sum proof reads a fresh SRP1 planning object (data only) and committed,
manifest-verified records.

WHAT IS CHECKED
  V   the stop rule VERSION 2 (`settling_criterion_v2`): W101's twelve trajectories run against it (t_sum = 0 so the gap
      clause is vacuous; the expected verdicts restated for L = 60), plus V13 a gap-clause refusal (then a later
      certification) and V13b a gap that never closes (uncertified, reason gap_clause); V14 the monotone branch refusing
      C*'s recorded 188-287 trajectory on |last step| x 60 > tau; V15 the monotone branch certifying a fast geometric
      decay; V16 the uncertified reporting form; V17 the dynamic cap; V0 `settling_criterion.py` (version 1)
      byte-identical to the file v39 / v41 pin.
  R   the rule v2 replayed on committed records with the per-cycle t_sum from `pf_entry_stride` (W112's functions,
      inputs verified against their campaign manifests): the C* extension (W110, 1..287) NEVER certifies; x = 0
      (W102, 1..181) and the unit (W103, 1..172): where v2 would certify them (report-only; their certificates are not
      changed); each record's terminal t_sum against its interface_settlement_detail_s31c.json identity (<= 0.01 EUR).
  O   the ten original cells and the inputs in force now: each original record committed, in its campaign manifest,
      its first residual pass == N_old - 9 with Boyd passing every cycle to N_old; the base key (original
      configuration) == the original eval key; lattice legality (current lattice: E/P in [2, 4], 0.25 MVA / 0.5 MWh
      steps, P <= 2.5, E <= 5, budget; and membership of the current Phase B admissible domain); the case, cost and ESS
      parameter files in force now; the I_j sources; the settled x0 comparator (Q181, band, t).
  H   the hooks (`p515_s53_w118_resettle_hooks`) driven through the REAL wrappers: H1 every gated cell with its
      ORIGINAL recorded values bitwise through k0, the overlap k0+1..N_old recorded (rho / gamma frozen at k0's values after k0), then a synthetic continuation (a recorded Boyd lapse at N_old + 1 resets the rule's k0) that
      the rule certifies (certificate length 0 at k*, decision file, summary ok, in-cycle rule == pure replay, holds
      inert through k0 and held after); H2 a one-ulp divergence at 40 ABORTS at 40; H3 a gap refusal then a
      certification; H4 a gated cell creeping to its cap (uncertified, stopped by the cap); H5 an ungated cell (first
      pass at 50, creep): stopped by the dynamic rule cap 159 (certificate length 0), holds from 51; H6 the in-cycle
      t_sum function on REAL SRP1 data (fresh planning, production's price and weight functions) and the recorded
      consensus copies of W102's cycle 181: equals W112's formula and the terminal identity; write-only (planning
      fingerprint and the consensus dict unchanged); H7-H9 the AA / tail / rho holds with REAL production functions;
      H10 the real install layering; H11 the certificate length written only by disable / restore / rule end.
  K   keys: for EVERY entry of every committed campaign spec the W118 harness's key equals the pre-W118 harness's
      (49342e8c, sha256 pinned, loaded from git); the ten re-settling keys follow the declared formula and appear in no
      committed campaign spec OUTSIDE THE W118 STAGE ROOT (the rule "a pre-run check that scans committed artefacts
      excludes the run's own"); negative controls (a planted spec outside the root; one in a sibling directory sharing
      the root's name prefix) refused; positive control (a planted spec in a W118 campaign root) accepted.
  P   `assert_resettle_preconditions` holds for every cell's declaration with its cap and refuses negative controls; the
      validator refuses an `early_stop` key and malformed declarations.
  W   (main only; not in the launcher's inline re-run) W100's repository-wide boolean-typing test, output to a new
      write-once file in this stage's checks directory.

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w118_resettle_checks.py \\
      > data/SRP1/Results/P515S53/w118_resettle/zero_solve_checks_launch.log 2>&1
"""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import shutil
import subprocess
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W118 re-settling zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w105_settling_extension_hooks as E105  # noqa: E402
import p515_s53_w105_extension_checks as K105  # noqa: E402 -- its helpers (arms its own permitted=() guard)
import p515_s53_w101_continuation_checks as K101  # noqa: E402 -- its twelve-trajectory fixtures (arms its own guard)
import p515_s53_w98_continuation_checks as K98  # noqa: E402 -- imported by the two above (its own guard)
import p515_s53_w112_consensus_gap as W112  # noqa: E402 -- the validated t_sum-from-stride functions (its own guard)
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w118_checks', GUARD), ('w105_checks_imported', K105.GUARD), ('w101_checks_imported', K101.GUARD),
                  ('w98_checks_imported', K98.GUARD), ('w112_imported', W112._GUARD)))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W118_ROOT_REL = os.path.join(_P53, 'w118_resettle')
OUT_DIR_REL = os.path.join(W118_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w118_zero_solve_checks.json'
OUT_MANIFEST = 'w118_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w118_bool_typing_test.json'
KEY_EXCLUDED_ROOTS = (W118_ROOT_REL,)
PRE_W118_HARNESS = {'commit': '49342e8c', 'sha256': '68a9265afe71d374a3058cee24ad1d614d71c6d10d5c4cb69bfc32cd78637b82'}
# settling_criterion.py (version 1) as pinned by the frozen stage specs v39 (8a612429) and v41 (fcea4b38)
V1_PINS = {'v39': os.path.join(_P53, 'frozen_s53_spec_v39_8a612429.json'),
           'v41': os.path.join(_P53, 'frozen_s53_spec_v41_fcea4b38.json')}
# committed records (each file verified against its campaign manifest by W112's _verify_inputs)
X0_CELL, UNIT_CELL, CSTAR_CELL = 'W101_x0_settled_181', 'W101_unit_settled_172', 'W110_c_star_ext'
DECISIONS = {'x0': os.path.join(W112.CELLS[X0_CELL], 'settling_decision.json'),
             'unit': os.path.join(W112.CELLS[UNIT_CELL], 'settling_decision.json')}
# the settled x = 0 comparator (Addendum 54: V_SRP1 settled; W106 Q181)
Q181 = 653873702.1876609
X0_BAND_WIDTH = None   # read from the committed decision at check time (4,209.17 rounded)
T_X0 = 142.52829384803772
# the current inputs
CASE_FILE_SHA256 = 'dbfdb2a07d12bfedab5e66bf94b3df98e5ab0305616652b4083266476972006b'
ESS_PARAMS_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
ESS_PARAMS_SHA256_C2 = '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706'
COST_FILE_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')
COST_FILE_SHA256 = 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'
PHASE_B_SPEC_REL = os.path.join(R._S47, 'campaign_spec_s47_phase_b_8cfa264e.json')
F2_SPEC_REL = os.path.join(R._S53F2, 'campaign_spec_s53_f2_certificate_r1_803571c0.json')
# the I_j sources (cited): Phase B state (I_x_eur per eval key), F2 states, W2 investment-cost table (year ladder)
I_SOURCES = {
    'phase_b_state': os.path.join(R._S47, 'phase_b_state.json'),
    'f2_phase_b_state': os.path.join(R._S51, 'phase_b_state.json'),
    'f2_certificate_state': os.path.join(R._S53F2, 'continuation_state.json'),
    'w2_investment_cost': os.path.join('data', 'SRP1', 'Results', 'P515S45', 'investment_cost',
                                       'investment_cost_results.json'),
}
LATTICE = {'p_step_mva': 0.25, 'e_step_mwh': 0.5, 'min_e_over_p': 2.0, 'max_e_over_p': 4.0, 'p_max_mva': 2.5,
           'e_max_mwh': 5.0, 'budget_eur': 1.0e6, 'years': (2025, 2030, 2035),
           'source': ('STEP4_DFO_METHOD.md lattice (0.25 MVA / 0.5 MWh; 2 h <= E/P <= 4 h; P <= 2.5 MVA; E <= 5.0 MWh); '
                      'data/SRP1/SharedESS/SRP1_ESS_Params.json min/max_energy_to_power_factor 2 / 4, max_capacity 5, '
                      'budget 1e6')}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jt(x):
    return GRIO.dumps(x, default=GRIO.json_default, sort_keys=True)


def _git_tracked_clean(rel):
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=REPO, capture_output=True).returncode == 0
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    return tracked and not dirty


# ======================================================================================================================
#  V -- the stop rule v2
# ======================================================================================================================
def _v2(q, b, p_max=R.P_MAX, t=None, cap=None, **kw):
    t = {k: 0.0 for k in q} if t is None else t
    if cap is None and 'cap_after_first_k0' not in kw:
        cap = max(q)
    return SC2.replay(q, b, t, p_max, cap=cap, **kw)


def _tp2(rule):
    return [[x, kind] for x, kind, _v in rule.sign.T]


def _cert_ks(out):
    return [r['k'] for r in out if str(r.get('decision') or '').startswith('certified')]


def cstar_188_287():
    """C*'s recorded Q over 188..287 (W110 per_cycle_record, manifest-verified)."""
    W112._verify_inputs([CSTAR_CELL], [W112.PCR])
    rows = {r['cycle']: r for r in _read_jsonl(_abs(os.path.join(W112.CELLS[CSTAR_CELL], W112.PCR)))}
    return {k: rows[k]['gross_operational_cost'] for k in range(188, 288)}


def tests_V():
    res = {}
    p_max = R.P_MAX
    x0q, x0b, _ = K101.load_record('3x3_x0_continuation_1_88')
    n7q, n7b, _ = K101.load_record('3x3_node7_1_69')

    def trunc(q, b, k):
        return {c: q[c] for c in range(1, k + 1)}, {c: b[c] for c in range(1, k + 1)}

    for tid, k, exp in (('T1_x0_trunc_72', 72, {'T': [[66, 'max']], 'reason': 'insufficient_turning_points'}),
                        ('T2_x0_trunc_77', 77, {'T': [[66, 'max']], 'reason': 'insufficient_turning_points'}),
                        ('T3_x0_trunc_86', 86, {'T': [[66, 'max'], [77, 'min']], 'reason': 'insufficient_turning_points',
                                                'A_rounded': [29140]}),
                        ('T4_x0_trunc_88', 88, {'T': [[66, 'max'], [77, 'min'], [86, 'max']],
                                                'reason': 'range_above_tau', 'A_rounded': [29140, 7824]})):
        q, b = trunc(x0q, x0b, k)
        out, dec, rule = _v2(q, b)
        got = {'k0': rule.k0, 'N': rule.n, 'T': _tp2(rule), 'A': list(rule.sign.A), 'certified_at': _cert_ks(out),
               'status': dec['status'], 'reasons': dec.get('reasons')}
        ok = (got['status'] == 'uncertified' and not got['certified_at'] and got['T'] == exp['T']
              and exp['reason'] in (got['reasons'] or []) and got['k0'] == 63 and got['N'] == 63)
        if 'A_rounded' in exp:
            ok = ok and [round(a) for a in got['A']] == exp['A_rounded']
        if tid == 'T4_x0_trunc_88':
            last = out[-1]
            ok = ok and last['sign_change'] is True and last['P_hat'] == 20 and last['W'] == 22 \
                and last['window'] == [67, 88] and round(last['range'], 2) == 28130.18
            got.update({'P_hat': last['P_hat'], 'W': last['W'], 'window': last['window'], 'range': last['range']})
        res[tid] = {'ok': bool(ok), 'expected': exp, 'got': got}
    out, dec, rule = _v2(n7q, n7b)
    res['T5_node7_1_69'] = {'ok': rule.k0 == 60 and _tp2(rule) == [[65, 'max']] and dec['status'] == 'uncertified'
                            and not _cert_ks(out), 'k0': rule.k0, 'T': _tp2(rule), 'status': dec['status']}

    def damped(boyd_false=()):
        q = {k: 1e9 + 30000.0 * 0.88 ** (k - 11) * math.cos(2 * math.pi * (k - 11) / 20.0) for k in range(1, 201)}
        b = {k: (k >= 11 and k not in boyd_false) for k in range(1, 201)}
        return q, b
    q6, b6 = damped()
    out6, dec6, rule6 = _v2(q6, b6)
    k6 = dec6.get('k_star')
    k6_v1 = SC1.replay(q6, b6, 0, 200, 22)[1].get('k_star')
    ok6 = (dec6['status'] == 'certified' and dec6['branch'] == 'oscillatory' and 18 <= dec6['P_hat'] <= 22
           and dec6['range'] <= SC2.TAU and abs(q6[k6] - 1e9) < SC2.TAU and k6 == k6_v1)
    res['T6_damped_cosine'] = {'ok': bool(ok6), 'k_star': k6, 'k_star_version_1': k6_v1, 'branch': dec6.get('branch'),
                               'P_hat': dec6.get('P_hat'), 'range': dec6.get('range')}
    lapse_at = 11 + 25
    q7, b7 = damped(boyd_false=(lapse_at,))
    out7, dec7, rule7 = _v2(q7, b7)
    res['T7_damped_cosine_lapse_at_k0_plus_25'] = {
        'ok': bool(dec7['status'] == 'certified' and dec7['k_star'] > k6 and len(rule7.lapses) == 1
                   and rule7.lapses[0]['cycle'] == lapse_at and dec7['k0'] == lapse_at + 1 and rule7.n == 11),
        'k_star': dec7.get('k_star'), 'k0_after_reset': dec7.get('k0'), 'N_first_k0': rule7.n}
    q8 = {k: 1e9 - 80.0 * (k - 1) for k in range(1, 121)}
    out8, dec8, _r8 = _v2(q8, {k: True for k in q8})
    res['T8_constant_creep_minus_80'] = {
        'ok': dec8['status'] == 'uncertified' and not _cert_ks(out8) and 'monotone_not_decreasing' in dec8['reasons']
        and 'monotone_last_step_times_L_above_tau' in dec8['reasons'], 'reasons': dec8['reasons'],
        'band_width': dec8.get('band_width'), 'drift': dec8.get('drift_rate_mean_dQ_last_25')}
    q9 = {1: 1e9}
    for k in range(2, 201):
        q9[k] = q9[k - 1] - 150.0 * 0.97 ** (k - 1)
    out9, dec9, _r9 = _v2(q9, {k: True for k in q9})
    expected9 = 1 + SC2.K_EXCL + 2 * p_max - 1
    res['T9_decaying_creep'] = {'ok': dec9['status'] == 'certified' and dec9['branch'] == 'monotone'
                                and dec9['k_star'] == expected9, 'k_star': dec9.get('k_star'),
                                'expected_k_star_k0_plus_K_EXCL_plus_L_minus_1': expected9,
                                'last_step_times_L': (dec9.get('certB_parts') or {}).get('last_step_times_L'),
                                'range': dec9.get('range')}
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
    q10 = {k: q10[k] for k in range(1, 151)}
    out10, dec10, _r10 = _v2(q10, {k: True for k in q10})
    res['T10_growing_oscillation'] = {'ok': dec10['status'] == 'uncertified' and not _cert_ks(out10)
                                      and 'swings_growing' in dec10['reasons'], 'reasons': dec10['reasons']}
    q11a = {k: 1e9 + 20.0 * (-1) ** k for k in range(1, 121)}
    _o, d11a, _ = _v2(q11a, {k: True for k in q11a})
    q11b = {k: 1e9 + 60.0 * (-1) ** k for k in range(1, 121)}
    _o, d11b, _ = _v2(q11b, {k: True for k in q11b})
    ok11a = (d11a['status'] == 'certified' and d11a['branch'] == 'monotone' and d11a['certB_parts']['stationary'] is True
             and d11a['band_width'] <= 40.0 and d11a['k_star'] == 1 + SC2.K_EXCL + 2 * p_max - 1)
    ok11b = d11b['status'] == 'certified' and d11b['branch'] == 'oscillatory'
    res['T11_jitter'] = {'ok': bool(ok11a and ok11b),
                         'pm20_stationary': {'k_star': d11a.get('k_star'), 'branch': d11a.get('branch')},
                         'pm60_equal_amplitudes': {'k_star': d11b.get('k_star'), 'branch': d11b.get('branch')}}
    q12 = {1: 1e9, 2: 1e9, 3: 1e9, 4: 1e9, 5: 1e9, 6: 1e9 + 500.0, 7: 1e9 + 500.0, 8: 1e9, 9: 1e9, 10: 1e9}
    out12, _d12, _r12 = _v2(q12, {k: True for k in q12})
    by = {r['k']: r for r in out12}
    res['T12_plus500_zero_minus500'] = {'ok': bool(by[8]['sign_change'] is True and not by[6]['sign_change']
                                                   and not by[7]['sign_change'] and by[8]['turning_point'][:2] == [6, 'max']
                                                   and by[7]['s_k'] == 0),
                                        'sign_change_cycles': [r['k'] for r in out12 if r['sign_change']]}
    # ---- new: the gap clause ----------------------------------------------------------------------------------------
    t13 = {k: (3000.0 if k <= k6 + 5 else -100.0) for k in q6}
    out13, dec13, r13 = _v2(q6, b6, t=t13)
    res['V13_gap_clause_refusal_then_certification'] = {
        'ok': bool(dec13['status'] == 'certified' and dec13['k_star'] > k6 and r13.gap_refusals
                   and r13.gap_refusals[0]['cycle'] == k6 and all(g['abs_t_sum'] > SC2.GAP_BOUND for g in r13.gap_refusals)
                   and abs(dec13['t_sum_k_star']) <= SC2.GAP_BOUND
                   and any(r.get('decision') == 'gap_clause_refused' for r in out13)),
        'k_star_without_gap': k6, 'k_star_with_gap': dec13.get('k_star'), 'n_refusals': len(r13.gap_refusals),
        'first_refusal': r13.gap_refusals[0] if r13.gap_refusals else None, 't_sum_k_star': dec13.get('t_sum_k_star')}
    t13b = {k: 3000.0 for k in q6}
    out13b, dec13b, r13b = _v2(q6, b6, t=t13b)
    res['V13b_gap_never_closes_uncertified'] = {
        'ok': bool(dec13b['status'] == 'uncertified' and dec13b['reasons'] == ['gap_clause'] and not _cert_ks(out13b)
                   and dec13b['t_sum_at_cap'] == 3000.0 and dec13b['gap_clause_refused_at_cap'] is True),
        'reasons': dec13b['reasons'], 'n_refusals': len(r13b.gap_refusals), 't_sum_at_cap': dec13b.get('t_sum_at_cap')}
    t_none = {k: (None if k == k6 else 0.0) for k in q6}
    out_n, dec_n, r_n = _v2(q6, b6, t=t_none)
    res['V13c_t_sum_missing_fails_the_clause'] = {
        'ok': bool(r_n.gap_refusals and r_n.gap_refusals[0]['cycle'] == k6 and r_n.gap_refusals[0]['t_sum'] is None
                   and dec_n['status'] == 'certified' and dec_n['k_star'] == k6 + 1),
        'k_star': dec_n.get('k_star'), 'refusals': r_n.gap_refusals[:2]}
    # ---- new: the amended monotone branch ------------------------------------------------------------------------------
    qc = cstar_188_287()
    q14 = {k - 187: v for k, v in qc.items()}
    out14, dec14, r14 = _v2(q14, {k: True for k in q14})
    reached = [r for r in out14 if r.get('certB_parts') and r['certB_parts']['lo_ge_k0_plus_K_EXCL']
               and r['certB_parts']['no_sign_change_in_window']]
    last14 = out14[-1]
    b14 = last14.get('certB_parts') or {}
    res['V14_monotone_refuses_c_star_188_287'] = {
        'ok': bool(dec14['status'] == 'uncertified' and not _cert_ks(out14) and not any(r['certB'] for r in out14)
                   and reached and all(r['certB_parts']['last_step_times_L_le_tau'] is False for r in reached)
                   and 'monotone_last_step_times_L_above_tau' in dec14['reasons']),
        'cycles_mapped': '188..287 -> 1..100 (boyd True, t_sum 0: the monotone clause isolated)',
        'n_cycles_monotone_window_reached_without_sign_change': len(reached),
        'first_reached_cycle_original_numbering': (reached[0]['k'] + 187) if reached else None,
        'at_287': {'last_step_abs': b14.get('last_step_abs'), 'last_step_times_L': b14.get('last_step_times_L'),
                   'tau': SC2.TAU, 'steps_decreasing': b14.get('steps_decreasing'),
                   'strictly_decreasing': b14.get('strictly_decreasing'), 'range': b14.get('range'),
                   'range_le_tau': b14.get('range_le_tau')},
        'literal_reading_refuses_on_the_new_clause_alone': (b14.get('steps_decreasing') is True
                                                            and b14.get('last_step_times_L_le_tau') is False),
        'reasons_at_287': dec14['reasons'], 'sign_changes_in_window': [r['k'] + 187 for r in out14 if r['sign_change']]}
    q15 = {1: 1e9}
    for k in range(2, 201):
        q15[k] = q15[k - 1] - 3000.0 * 0.7 ** (k - 1)
    out15, dec15, _r15 = _v2(q15, {k: True for k in q15})
    res['V15_monotone_certifies_fast_geometric_decay'] = {
        'ok': bool(dec15['status'] == 'certified' and dec15['branch'] == 'monotone'
                   and dec15['k_star'] == 1 + SC2.K_EXCL + 2 * p_max - 1
                   and dec15['certB_parts']['last_step_times_L_le_tau'] is True),
        'k_star': dec15.get('k_star'), 'last_step_times_L': dec15['certB_parts'].get('last_step_times_L'),
        'range': dec15.get('range')}
    # ---- the uncertified reporting form (frozen) ------------------------------------------------------------------------
    q16 = {k: 1e9 - 80.0 * (k - 1) for k in range(1, 121)}
    t16 = {k: 10.0 * k for k in q16}
    _o, d16, _r = _v2(q16, {k: True for k in q16}, t=t16)
    form = ('band', 'band_width', 'band_window', 'drift_rate_mean_dQ_last_25', 'dQ_cc_rate_mean_last_25',
            't_sum_at_cap', 'Q_cc_at_cap', 'reasons', 'k_cap')
    res['V16_uncertified_form'] = {
        'ok': bool(d16['status'] == 'uncertified' and all(f in d16 for f in form) and d16['drift_rate_mean_dQ_last_25']
                   == -80.0 and abs(d16['dQ_cc_rate_mean_last_25'] - (-70.0)) < 1e-9 and d16['band_window'] == [61, 120]
                   and d16['band_width'] == 80.0 * 59 and d16['t_sum_at_cap'] == 1200.0),
        'form_fields': list(form), 'decision': {f: d16.get(f) for f in form}}
    # ---- the dynamic cap ----------------------------------------------------------------------------------------------
    q17 = {k: 1e9 - 80.0 * k for k in range(1, 400)}
    b17 = {k: k >= 50 for k in q17}
    o17, d17, r17 = SC2.replay(q17, b17, {k: 0.0 for k in q17}, p_max, cap_after_first_k0=R.CAP_AFTER_K0)
    o17n, d17n, _ = SC2.replay(q17, {k: False for k in q17}, {}, p_max, cap_after_first_k0=R.CAP_AFTER_K0)
    o17c, d17c, _ = SC2.replay(q17, {k: k >= 250 for k in q17}, {k: 0.0 for k in q17}, p_max,
                               cap_after_first_k0=R.CAP_AFTER_K0)
    res['V17_dynamic_cap'] = {
        'ok': bool(d17['status'] == 'uncertified' and d17['k_cap'] == 50 + 109 and r17.n == 50 and len(o17) == 159
                   and d17n['status'] == 'uncertified' and d17n['k_cap'] == 300 and d17n['reasons'] == ['no_residual_pass']
                   and d17c['k_cap'] == 300),
        'first_pass_50_cap': d17.get('k_cap'), 'no_pass_cap': d17n.get('k_cap'), 'no_pass_reasons': d17n.get('reasons'),
        'first_pass_250_cap_ceiling': d17c.get('k_cap')}
    # ---- V0: version 1 unchanged ---------------------------------------------------------------------------------------
    now = H.sha256_file(_abs('settling_criterion.py'))
    pins = {k: (json.load(open(_abs(p)))['code_sha256'] or {}).get('settling_criterion.py') for k, p in V1_PINS.items()}
    res['V0_settling_criterion_v1_byte_identical'] = {
        'ok': bool(all(v == now for v in pins.values()) and _git_tracked_clean('settling_criterion.py')),
        'sha256_now': now, 'pinned': pins}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  R -- replays on committed records (t_sum from the pf stride)
# ======================================================================================================================
def record_series(cell):
    """(Q, boyd, t_sum, terminal validation) for a W112 cell: per_cycle_record + pf stride (W112 functions)."""
    W112._verify_inputs([cell], [W112.PCR, W112.STRIDE, W112.DETAIL])
    rel = W112.CELLS[cell]
    rows = {r['cycle']: r for r in _read_jsonl(_abs(os.path.join(rel, W112.PCR)))}
    detail = json.load(open(_abs(os.path.join(rel, W112.DETAIL))))
    pi, w = W112._price_weight(detail)
    per = W112._stream_stride(os.path.join(rel, W112.STRIDE), pi, w)
    q = {k: rows[k]['gross_operational_cost'] for k in rows}
    b = {k: bool(rows[k]['boyd_all_pass'] and rows[k]['local_solves_ok']) for k in rows}
    t = {k: per[k]['t_sum'] for k in per}
    last = max(rows)
    ident = W112._detail_identity(detail)
    val = {'last_cycle': last, 't_sum_from_stride': t.get(last), 'terminal_t_tso_plus_t_dso': ident['t_tso_plus_t_dso_terminal'],
           'abs_diff': abs(t.get(last) - ident['t_tso_plus_t_dso_terminal']), 'tol_eur': 0.01,
           'cycles_run_in_detail': ident['cycles_run']}
    val['pass'] = val['abs_diff'] <= 0.01 and ident['cycles_run'] == last
    return q, b, t, val, rows


def tests_R():
    res = {}
    p_max = R.P_MAX
    # the C* extension: NEVER certifies
    q, b, t, val, _rows = record_series(CSTAR_CELL)
    out, dec, rule = SC2.replay(q, b, t, p_max, cap=max(q))
    would_branch = [r['k'] for r in out if r.get('branch_would_certify')]
    res['R1_c_star_extension_1_287_never_certifies'] = {
        'ok': bool(dec['status'] == 'uncertified' and not _cert_ks(out) and val['pass'] and max(q) == 287),
        'status': dec['status'], 'reasons': dec.get('reasons'), 'k0': rule.k0, 'N_first_k0': rule.n,
        'n_turning_points': len(rule.sign.T), 'T': dec.get('T'), 'band': dec.get('band'),
        'band_width': dec.get('band_width'), 'drift_rate_mean_dQ_last_25': dec.get('drift_rate_mean_dQ_last_25'),
        'dQ_cc_rate_mean_last_25': dec.get('dQ_cc_rate_mean_last_25'), 't_sum_at_287': dec.get('t_sum_at_cap'),
        'cycles_where_a_branch_held_but_the_gap_refused': would_branch, 'gap_refusals': rule.gap_refusals[:10],
        'terminal_validation': val}
    # x = 0 and the unit: where v2 would certify (report-only)
    for name, cell in (('R2_x0_W102_1_181', X0_CELL), ('R3_unit_W103_1_172', UNIT_CELL)):
        q, b, t, val, rows = record_series(cell)
        out, dec, rule = SC2.replay(q, b, t, p_max, cap=max(q))
        k1 = json.load(open(_abs(DECISIONS['x0' if cell == X0_CELL else 'unit'])))
        refusals = [r['k'] for r in out if r.get('decision') == 'gap_clause_refused']
        res[name] = {
            'ok': bool(val['pass']), 'report_only': True,
            'status_within_record': dec['status'],
            'k_star_v2': dec.get('k_star'), 'branch_v2': dec.get('branch'), 't_sum_at_k_star': dec.get('t_sum_k_star'),
            'Q_at_k_star': dec.get('Q_k_star'), 'band_width_v2': dec.get('band_width'),
            'range_over_tau_v2': dec.get('range_over_tau'), 'k0': dec.get('k0') or rule.k0, 'N_first_k0': rule.n,
            'gap_refusal_cycles': refusals, 'uncertified_reasons_at_record_end': dec.get('reasons'),
            'k_star_version_1_certificate': k1.get('k_star'), 'branch_version_1': k1.get('branch'),
            'same_cycle_as_version_1': dec.get('k_star') == k1.get('k_star'),
            'terminal_validation': val}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  O -- the original cells and the inputs in force now
# ======================================================================================================================
def _spec_entry(cell):
    c = R.CELLS[cell]
    spec = json.load(open(_abs(os.path.join(c['orig_root'], c['orig_spec']))))
    hits = [e for e in spec['candidates'] if e.get('eval_key') == c['orig_eval_key']]
    return spec, (hits[0] if len(hits) == 1 else None)


def lattice_check(canonical, i_eur):
    """The current lattice rules on one single-cohort plan; returns {rule: bool} and the per-node E/P."""
    L = LATTICE
    out, ep = {}, {}
    yr = canonical['investment_year']
    out['year_on_the_lattice'] = yr in L['years']
    ok_steps = ok_ratio = ok_bounds = ok_zero = True
    for n, (p, e) in canonical['nodes'].items():
        p, e = float(p), float(e)
        ok_steps &= abs(p / L['p_step_mva'] - round(p / L['p_step_mva'])) < 1e-12 and \
            abs(e / L['e_step_mwh'] - round(e / L['e_step_mwh'])) < 1e-12
        ok_zero &= (p == 0.0) == (e == 0.0)
        if p > 0:
            ep[n] = e / p
            ok_ratio &= L['min_e_over_p'] <= e / p <= L['max_e_over_p']
        ok_bounds &= 0.0 <= p <= L['p_max_mva'] and 0.0 <= e <= L['e_max_mwh']
    out.update({'steps_0_25_mva_0_5_mwh': bool(ok_steps), 'p_zero_iff_e_zero': bool(ok_zero),
                'e_over_p_in_2_4': bool(ok_ratio), 'p_le_2_5_e_le_5': bool(ok_bounds),
                'budget_i_le_1e6': (i_eur is not None and i_eur <= L['budget_eur'])})
    return out, ep


def i_j(cell):
    """I_j from the cited source (Phase B / F2 states: the I_x_eur of the eval key's poll candidate; the year ladder:
    the W2 investment-cost table's I_new_eur of the label)."""
    c = R.CELLS[cell]
    if c['group'] == 'year_ladder':
        doc = json.load(open(_abs(I_SOURCES['w2_investment_cost'])))
        lab = c['orig_label']
        v = doc['candidates'][lab]['I_new_eur']
        return v, {'source': I_SOURCES['w2_investment_cost'], 'field': f"candidates['{lab}'].I_new_eur",
                   'sha256': H.sha256_file(_abs(I_SOURCES['w2_investment_cost'])),
                   'candidate_key_in_source': doc['candidates'][lab]['candidate_key']}
    src = {'phase_b': 'phase_b_state', 'f2': ('f2_certificate_state' if cell == 'f2_challenger' else 'f2_phase_b_state')}
    key = src[c['group']]
    doc = json.load(open(_abs(I_SOURCES[key])))
    vals = set()
    for h in doc['history']:
        inc = h['incumbent']
        if inc.get('eval_key') == c['orig_eval_key']:
            vals.add(('incumbent', inc['I']))
        for cand in h['candidates']:
            if cand.get('eval_key') == c['orig_eval_key']:
                vals.add(('poll', cand['I_x_eur']))
    distinct = sorted({v for _k, v in vals})
    return (distinct[0] if len(distinct) == 1 else None), {
        'source': I_SOURCES[key], 'field': 'history[*].candidates[eval_key].I_x_eur / history[*].incumbent.I',
        'sha256': H.sha256_file(_abs(I_SOURCES[key])), 'values_found': sorted(vals), 'distinct': distinct}


def tests_O():
    res = {}
    cells = {}
    for cell in R.CELL_ORDER:
        c = R.CELLS[cell]
        spec, entry = _spec_entry(cell)
        man = json.load(open(_abs(os.path.join(c['orig_root'], 'campaign_manifest_sha256.json'))))
        rel = R.reference_path(cell)
        sha = H.sha256_file(_abs(rel))
        rows = _read_jsonl(_abs(rel))
        passes = [r['cycle'] for r in rows if r['boyd_all_pass'] and r['local_solves_ok']]
        n_old = rows[-1]['cycle']
        cfg = spec['configuration']
        base_orig = H.evaluation_key(entry['key'], entry['overrides'],
                                     case_file_aa=cfg.get('case_file_anderson_acceleration'),
                                     ess_ageing_baseline=cfg.get('ess_ageing_baseline'),
                                     flex_price_multiplier=entry.get('flex_price_multiplier'),
                                     convergence_depth_tail=cfg.get('convergence_depth_tail')) if entry else None
        i_val, i_src = i_j(cell)
        lat, ep = lattice_check(entry['canonical'], i_val) if entry else ({}, {})
        pb = json.load(open(_abs(PHASE_B_SPEC_REL))) if c['group'] != 'f2' else json.load(open(_abs(F2_SPEC_REL)))
        in_domain = any(x['key'] == entry['key'] for x in pb['candidates']) if entry else False
        rec = json.load(open(_abs(os.path.join(R.original_eval_dir(cell), 'evaluation_record.json'))))
        parts = {
            'original_spec_entry_found': entry is not None,
            'entry_eval_dir_and_label_as_pinned': bool(entry) and entry['eval_dir'] == c['orig_eval_dir']
            and entry['label'] == c['orig_label'],
            'per_cycle_record_sha256_as_pinned': sha == c['per_cycle_record_sha256'],
            'per_cycle_record_in_campaign_manifest': man.get(rel) == sha,
            'per_cycle_record_committed_clean': _git_tracked_clean(rel),
            'N_old_as_pinned': n_old == c['N_old'],
            'first_residual_pass_is_N_old_minus_9': bool(passes) and passes[0] == c['k0'] == n_old - 9,
            'boyd_passes_every_cycle_k0_to_N_old': passes == list(range(c['k0'], n_old + 1)),
            'no_failed_local_solve': all(r['local_solves_ok'] for r in rows),
            'base_key_original_configuration_equals_original_eval_key': base_orig == c['orig_eval_key'],
            'original_record_certified_at_N_old': rec.get('status') == 'certified' and rec.get('cycles_run') == n_old,
            'flex_multiplier_as_pinned': (entry or {}).get('flex_price_multiplier') == c['flex_price_multiplier'],
            'I_j_found_unique': i_val is not None,
            'lattice_legal': bool(lat) and all(lat.values()),
            'in_the_current_admissible_domain': in_domain,
        }
        cells[cell] = {'ok': all(parts.values()), 'parts': parts, 'k0': c['k0'], 'N_old': n_old,
                       'Q_N_old': rows[-1]['gross_operational_cost'], 'Q_k0': rows[c['k0'] - 1]['gross_operational_cost'],
                       'canonical': (entry or {}).get('canonical'), 'candidate_key': (entry or {}).get('key'),
                       'e_over_p': ep, 'lattice': lat, 'I_j': i_val, 'I_j_source': i_src,
                       'original': {'campaign_id': c['orig_campaign_id'], 'spec': os.path.join(c['orig_root'], c['orig_spec']),
                                    'spec_sha256': H.sha256_file(_abs(os.path.join(c['orig_root'], c['orig_spec']))),
                                    'eval_dir': R.original_eval_dir(cell), 'eval_key': c['orig_eval_key'],
                                    'per_cycle_record': rel, 'per_cycle_record_sha256': sha,
                                    'configuration': {k: cfg.get(k) for k in ('case_file_sha256',
                                                                              'case_file_anderson_acceleration',
                                                                              'ess_ageing_baseline_label',
                                                                              'convergence_depth_tail')},
                                    'ess_params_file_at_run': cfg.get('ess_params_file'),
                                    'harness_sha256': spec['harness']['sha256'], 'git_head': spec.get('git_head')},
                       't_sum_terminal_original': json.load(open(_abs(os.path.join(R.original_eval_dir(cell),
                                                                                   'interface_settlement_detail_s31c.json'))
                                                                 ))['t_tso_plus_t_dso_terminal']}
    res['cells'] = cells
    ess = json.load(open(_abs(ESS_PARAMS_REL)))
    x0dec = json.load(open(_abs(DECISIONS['x0'])))
    x0rows = {r['cycle']: r for r in _read_jsonl(_abs(os.path.join(W112.CELLS[X0_CELL], W112.PCR)))}
    x0det = json.load(open(_abs(os.path.join(W112.CELLS[X0_CELL], W112.DETAIL))))
    W112._verify_inputs([X0_CELL], [W112.PCR, W112.DETAIL])
    inputs = {
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': H.sha256_file(H.CASE_FILE), 'expected': CASE_FILE_SHA256},
        'ess_params_file': {'path': ESS_PARAMS_REL, 'sha256': H.sha256_file(_abs(ESS_PARAMS_REL)),
                            'expected_C2': ESS_PARAMS_SHA256_C2,
                            'max_energy_to_power_factor': ess['max_energy_to_power_factor'],
                            'min_energy_to_power_factor': ess['min_energy_to_power_factor'],
                            'max_capacity': ess['max_capacity'], 'budget': ess['budget'],
                            'calibration': ess['ageing']['calibration'].get('eol_retention_r'),
                            'minimum_soh': ess['ageing']['minimum_soh'],
                            'last_commit': H._git(['log', '-1', '--format=%H', '--', ESS_PARAMS_REL])},
        'cost_file': {'path': COST_FILE_REL, 'sha256': H.sha256_file(_abs(COST_FILE_REL)), 'expected': COST_FILE_SHA256,
                      'last_commit': H._git(['log', '-1', '--format=%H', '--', COST_FILE_REL])},
    }
    res['inputs_now'] = inputs
    res['x0_comparator'] = {'Q181': x0rows[181]['gross_operational_cost'], 'Q181_pinned': Q181,
                            'band_width': x0dec['band_width'], 'k_star': x0dec['k_star'],
                            't_x0_terminal': x0det['t_tso_plus_t_dso_terminal'], 't_x0_pinned': T_X0,
                            'source': {'per_cycle_record': os.path.join(W112.CELLS[X0_CELL], W112.PCR),
                                       'decision': DECISIONS['x0'], 'detail': os.path.join(W112.CELLS[X0_CELL],
                                                                                            W112.DETAIL)}}
    parts = {
        'every_cell_ok': all(v['ok'] for v in cells.values()),
        'case_file_as_every_original_and_now': inputs['case_file']['sha256'] == CASE_FILE_SHA256 and all(
            v['original']['configuration']['case_file_sha256'] == CASE_FILE_SHA256 for v in cells.values()),
        'ess_params_now_C2_E_over_P_2_4': (inputs['ess_params_file']['sha256'] == ESS_PARAMS_SHA256_C2
                                           and ess['max_energy_to_power_factor'] == 4.0
                                           and ess['min_energy_to_power_factor'] == 2.0
                                           and _git_tracked_clean(ESS_PARAMS_REL)),
        'gated_cells_ran_with_the_current_ess_params': all(
            (cells[c]['original']['ess_params_file_at_run'] or {}).get('sha256') == ESS_PARAMS_SHA256_C2
            for c in R.GATED_CELLS),
        'year_ladder_ran_without_a_c2_declaration': all(
            cells[c]['original']['configuration']['ess_ageing_baseline_label'] is None for c in R.UNGATED_CELLS),
        'no_original_had_the_tail': all(cells[c]['original']['configuration']['convergence_depth_tail'] is None
                                        for c in R.CELL_ORDER),
        'cost_file_as_pinned': inputs['cost_file']['sha256'] == COST_FILE_SHA256 and _git_tracked_clean(COST_FILE_REL),
        'x0_comparator_Q181_bitwise': x0rows[181]['gross_operational_cost'] == Q181 and x0dec['k_star'] == 181,
        'x0_comparator_t_bitwise': x0det['t_tso_plus_t_dso_terminal'] == T_X0,
    }
    res['parts'] = parts
    res['holds'] = all(parts.values())
    return res


# ======================================================================================================================
#  H -- the hooks through the real wrappers
# ======================================================================================================================
NODES, YEARS, DAYS, PERIODS = (5, 7, 9), (2025, 2030, 2035), ('Spring', 'Summer', 'Autumn', 'Winter'), 24


def _fake_world(t_target):
    """K105's fake SRP1-shaped ESS world plus a pf consensus block, num_instants and fake pricing (w = 1, pi = 1): each
    cycle's t_sum is set to `t_target(c)` by spreading it evenly over the 864 active-power entries."""
    pp, tso, dso, esso, cv, set_ess = K105.fake_ess_world()
    pp.num_instants = PERIODS
    cv['pf'] = {side: {'current': {n: {y: {d: {'p': [0.0] * PERIODS, 'q': [0.0] * PERIODS} for d in DAYS}
                                       for y in YEARS} for n in NODES}} for side in ('tso', 'dso')}
    n_entries = len(NODES) * len(YEARS) * len(DAYS) * PERIODS

    def set_cycle(c):
        set_ess(c)
        v = t_target(c)
        gap = 0.0 if v is None else v / n_entries
        for n in NODES:
            for y in YEARS:
                for d in DAYS:
                    cv['pf']['tso']['current'][n][y][d]['p'] = [10.0] * PERIODS
                    cv['pf']['dso']['current'][n][y][d]['p'] = [10.0 + gap] * PERIODS
    return pp, tso, dso, esso, cv, set_cycle


def _fake_srp(blocks_by_cycle, st):
    def blocks_fn(pp, models):
        return blocks_by_cycle[st.cycle][0]

    def obj_fn(pp, models):
        return blocks_by_cycle[st.cycle][1]
    return SimpleNamespace(_get_operational_recourse_block_components=blocks_fn,
                           _get_operational_objective_component_blocks=obj_fn,
                           _get_admm_block_weight=lambda tn, y, d: 1.0,
                           _expected_market_price=lambda model, network, p: 1.0)


def _blocks_for(gross, salvage):
    """48 network blocks sharing the gross equally (generation only) + SALVAGE, and their component dicts."""
    each = gross / 48.0
    blocks, obj = {}, {}
    keys = [('TSO', None, str(y), d) for y in YEARS for d in DAYS] + \
           [('DSO', n, str(y), d) for n in NODES for y in YEARS for d in DAYS]
    for k in keys:
        blocks[k] = each
        obj[k] = {'generation_cost': each, 'flexibility_cost': 0.0, 'load_curtailment_cost': 0.0,
                  'res_curtailment_penalty': 0.0, 'ess_usage_penalty': 0.0, 'ess_complementarity_penalties': 0.0,
                  'slack_penalties': 0.0, 'classified_total': each}
    blocks[('SALVAGE', None, None, None)] = -float(salvage or 0.0)
    return blocks, obj


def _pen_row(row):
    return K101._row_objects(row)[2]


def _frozen_pen(row):
    """The penalty-update return value of a held cycle: rho / gamma as after `row`, unchanged (before == after)."""
    _a, _b, after, _bg, ag, _rfa, _fs = _pen_row(row)
    return ({g: 'held (frozen after 10 unchanged cycles)' for g in R.CHANNELS}, dict(after), dict(after), dict(ag),
            dict(ag), True, {g: {'frozen': True} for g in R.CHANNELS})


def drive(cell, variant='certify', synth_len=160, first_pass_at=50):
    """The REAL wrappers (R.make_wrappers) driven in production's call order. A gated cell: its ORIGINAL recorded values
    (per_cycle_record + g_s39_D rows) through N_old, then a synthetic continuation; an ungated cell: a synthetic run
    (Boyd from `first_pass_at`). Variants: certify (damped oscillation, t_sum small; a gated cell has a recorded Boyd
    lapse at N_old + 1), ulp_at_40, gap (as certify, t_sum above the bound for the first 80 continuation cycles), creep
    (constant creep to the cap)."""
    decl = R.declaration_for(cell)
    ref = R.load_replay_reference(decl)
    cap = R.spec_cap(cell)
    sink = []
    st = R.ResettleState(decl, None, cap, reference=ref, sink=sink)
    scripts = {'boyd': [], 'aa': [], 'next': [], 'pen': [], 'rc': [], 'efc': []}
    orig, calls = K105._standins(None, scripts)
    blocks_by_cycle = {}
    gated = decl['replay_reference'] is not None
    c_info = R.CELLS[cell]
    g_rows = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(R.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if gated else {})
    n_old = c_info['N_old'] if gated else None
    q_anchor = ref[n_old]['gross_operational_cost'] if gated else 6.5e8

    def t_target(c):
        if gated and c <= n_old:
            return 12345.0
        j = c - (n_old if gated else first_pass_at)
        if variant == 'gap' and j <= 80:
            return 5000.0
        return 150.0
    pp, tso, dso, esso, cv, set_cycle = _fake_world(t_target)
    w = R.make_wrappers(st, orig, srp_module=_fake_srp(blocks_by_cycle, st))
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    models = {'tso': tso, 'dso': dso, 'esso': esso}
    esso_terms_orig = E105.esso_side_terms
    E105.esso_side_terms = lambda pp_, em: {str(n): {'salvage_value': 0.0, 'feasibility_penalty': 0.0} for n in NODES}
    raised = None
    template = g_rows.get(n_old) if gated else json.load(open(_abs(os.path.join(
        R.original_eval_dir('pb_y2025_n5'), 'g_s39_D.json'))))['cycle_trajectory'][-1]
    pen_hold = _frozen_pen(g_rows[c_info['k0']] if gated else template)
    try:
        w['_capture_convergence_depth_tail_baseline'](pp, admm)
        for c in range(1, cap + 1):
            if gated and c <= n_old:
                row, g = ref[c], g_rows[c]
                bm = K105.boyd_from_g(g)
                rc = {'gross_operational_cost': row['gross_operational_cost'], 'net_operational_recourse': row['recourse'],
                      'terminal_salvage_value': row['terminal_salvage_value']}
                if variant == 'ulp_at_40' and c == 40:
                    rc['gross_operational_cost'] = math.nextafter(rc['gross_operational_cost'], math.inf)
                pen = _pen_row(g) if c <= c_info['k0'] else pen_hold
                efc = row['efc_per_day_max']
                local_ok = row['local_solves_ok']
                active = c > c_info['k0']
            else:
                base = n_old if gated else first_pass_at
                j = c - base
                if variant in ('certify', 'gap'):
                    qs = 6000.0 * (0.6 ** (j / 15.0)) * math.cos(2 * math.pi * j / 30.0)
                else:
                    qs = -300.0 * j
                bm = copy.deepcopy(K105.boyd_from_g(template))
                passing = gated or c >= first_pass_at
                # a gated synthetic oscillation: a Boyd lapse at N_old + 1 (recorded; the holds continue) resets the
                # rule's k0 so the synthetic swings are judged on their own (the recorded descent to N_old would
                # otherwise be the first half-swing -- W105 R5's device)
                if gated and variant in ('certify', 'gap') and c == n_old + 1:
                    passing = False
                bm['all_boyd_pass'] = bool(passing)
                q = q_anchor + qs
                rc = {'gross_operational_cost': q, 'net_operational_recourse': q, 'terminal_salvage_value': 0.0}
                pen = pen_hold
                efc = 1.0
                local_ok = True
                active = st.held(c)
            blocks_by_cycle[c] = _blocks_for(rc['gross_operational_cost'], rc['terminal_salvage_value'])
            set_cycle(c)
            scripts['boyd'].append(bm)
            scripts['aa'].append(None)
            scripts['next'].append(None)
            scripts['pen'].append(pen)
            if local_ok:
                scripts['rc'].append(rc)
            scripts['efc'].append(efc)
            with contextlib.redirect_stdout(io.StringIO()):
                w['_apply_convergence_depth_tail'](pp, admm, active, object(), c)
                w['get_admm_boyd_residual_metrics'](pp, tso, dso, esso, cv, {}, admm)
                if local_ok:
                    aa_rec = w['_anderson_acceleration_cycle_step'](object(), None, cv, {}, None, {}, bm, c)
                    w['_get_operational_recourse_components'](pp, models)
                else:
                    aa_rec = {'cycle': c, 'action': 'skipped (local solve failure this cycle)'}
                w['_convergence_depth_tail_next_state'](bool(bm['all_boyd_pass'] and local_ok), True, aa_rec)
                w['_update_admm_penalties']({}, {}, {}, {}, bm, object(), iter=c, allow_update=local_ok,
                                            freeze_state={})
                w['_get_admm_efc_per_day_max'](esso)
            if admm.minimum_consecutive_converged_cycles == R.SETTLING_END_THRESHOLD:
                break
        w['_apply_convergence_depth_tail'](pp, admm, False, object(), None)
    except RuntimeError as error:
        raised = str(error)
    finally:
        E105.esso_side_terms = esso_terms_orig
    files = {}
    for fname, obj in sink:
        files.setdefault(fname, []).append(obj)
    return {'state': st, 'files': files, 'raised': raised, 'calls': calls, 'admm': admm}


def _pure_equal(st, lines):
    cr = st.decl['cap_rule']
    if cr['kind'] == 'fixed':
        rule = SC2.SettlingRuleV2(R.P_MAX, cap=cr['cap'])
    else:
        rule = SC2.SettlingRuleV2(R.P_MAX, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])
    pure = [rule.observe(x['cycle'], x.get('gross'), bool(x.get('boyd_k')), x.get('t_sum')) for x in lines]
    return [_jt(x.get('settling')) for x in lines] == [_jt(p) for p in pure], rule.decision


def _drive_summary(d):
    st, files = d['state'], d['files']
    lines = files.get(R.CYCLE_FILE, [])
    summ = st.summary()
    pure_ok, pure_dec = _pure_equal(st, lines) if lines else (False, None)
    fp = st.first_pass
    holds_pre = sorted({json.dumps(x['holds'], sort_keys=True) for x in lines if fp is None or x['cycle'] <= fp})
    holds_post = sorted({json.dumps(x['holds'], sort_keys=True) for x in lines if fp is not None and x['cycle'] > fp})
    return {'raised': d['raised'], 'lines': len(lines), 'creep_lines': len(files.get(R.CREEP_FILE, [])),
            'decision_files': len(files.get(R.DECISION_FILE, [])), 'summary_ok': summ['ok'],
            'stopped_by': summ['stopped_by'], 'status': summ['settling_status'], 'k_star': summ['k_star'],
            'branch': summ['branch'], 'first_pass': fp, 'replay_bitwise_through': summ['replay_bitwise_through_cycle'],
            'first_divergence': summ['replay_first_divergence'], 'n_overlap': len(summ['overlap_k0_plus_1_to_N_old']),
            'overlap_all_zero': all(o['Q_new_minus_Q_old'] == 0.0 for o in summ['overlap_k0_plus_1_to_N_old']),
            'gap_refusals': len(summ['gap_refusals']), 'n_lapses': len(summ['lapse_events']), 'certificate_length_after': d['admm'].minimum_consecutive_converged_cycles,
            'ended_by': st.ended_by, 'in_cycle_rule_equals_pure_replay': pure_ok,
            'pure_decision_status': (pure_dec or {}).get('status'),
            't_sum_every_line': all(isinstance(x.get('t_sum'), float) for x in lines),
            'q_cc_every_line': all(isinstance(x.get('Q_cc'), float) for x in lines if x.get('gross') is not None),
            'holds_through_first_pass': holds_pre, 'holds_after_first_pass': holds_post,
            'capture_errors': summ['capture_errors'][:3], 'errors': summ['errors'][:3],
            'last_cycle': st.cycle, 'rule_cap': st.rule.cap}


HOLDS_OFF = json.dumps({'aa': False, 'rho': False, 'tail_apply': False, 'tail_next': False}, sort_keys=True)
HOLDS_ON = json.dumps({'aa': True, 'rho': True, 'tail_apply': True, 'tail_next': True}, sort_keys=True)


def _h_real(srp, which):
    """K105's real-production hold tests re-run on the W118 state, the first residual pass preset at the W105 hold
    cycle (87) so the holds engage from 88 exactly as there; returns K105's result dict."""
    real_state = K105._state
    real_make = K105.E.make_wrappers
    real_wrapped = K105.E.WRAPPED
    real_n = K105.E.N_HOLD

    def state():
        st = R.ResettleState(R.declaration_for('pb_y2025_n5'), None, R.spec_cap('pb_y2025_n5'), reference={}, sink=[])
        st.first_pass = real_n
        return st
    K105._state = state
    K105.E.make_wrappers = lambda st, originals, srp_module=None: R.make_wrappers(st, originals, srp_module=srp_module)
    K105.E.WRAPPED = R.WRAPPED
    try:
        if which == 'aa':
            return K105._h_aa_real()
        if which == 'tail':
            return K105._h_tail_real(srp)
        return K105._h_rho_real(srp)
    finally:
        K105._state = real_state
        K105.E.make_wrappers = real_make
        K105.E.WRAPPED = real_wrapped


def _h_layering(srp):
    before = {name: getattr(srp, name) for name in R.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w118_layering_')
    cell = 'pb_y2025_n5'
    n = R.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with R.settling_resettle_hooks(scratch, R.declaration_for(cell), holder, cap=R.spec_cap(cell)) as st:
            st.first_pass = n
            installed_inner = srp.get_admm_boyd_residual_metrics
            with IDC.interface_dual_capture_hooks(scratch, holder):
                idc_outer = srp.get_admm_boyd_residual_metrics is not installed_inner
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
                            {'cycle': c, 'action': R.AA_OFF_ACTION if (conv or c > n) else 'accepted'})
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
    i_ext = child_src.find('extension_cm, \\', i_cont)
    i_res = child_src.find('resettle_cm, \\', i_cont)
    i_idc = child_src.find('IDC.interface_dual_capture_hooks(eval_dir, holder) as dual_capture', i_cont)
    i_s38 = child_src.find('G.s38_pf_capture_hooks(', i_cont)
    i_app = child_src.find('convergence_depth_append_hooks(appender)', i_cont)
    order_ok = 0 < i_cont < i_settle < i_ext < i_res < i_idc < i_s38 < i_app
    pre_ok = (0 < child_src.find('W118C.assert_resettle_preconditions(resettle, spec, tail_checklist, aa_on)')
              < child_src.find('G.run_admm_arm('))
    summ = holder.get(R.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and order_ok and pre_ok and idc_outer
                          and summ.get('phase') == 'ended' and summ.get('certificate_length_restored_at_exit') == 10
                          and not summ.get('errors') and admm.minimum_consecutive_converged_cycles == 10),
            'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'child_real_order_continuation_settling_extension_resettle_idc_s38_appender': order_ok,
            'child_real_checklist_before_run_admm_arm': pre_ok, 'files_written_by_install_without_cycles': files_written,
            'summary_phase': summ.get('phase'), 'summary_errors': summ.get('errors')}


def _h_t_sum_real():
    """The in-cycle t_sum function on REAL SRP1 data: a fresh planning object (data only), production's price and
    weight functions, and the consensus copies of W102's cycle 181 read from its pf stride; against W112's formula on
    the same stride and the terminal identity; write-only (planning fingerprint, consensus dict)."""
    import p56a_oracle as O
    import shared_resources_planning as srp
    W112._verify_inputs([X0_CELL], [W112.STRIDE, W112.DETAIL, W112.PCR])
    rel = W112.CELLS[X0_CELL]
    detail = json.load(open(_abs(os.path.join(rel, W112.DETAIL))))
    pi_d, w_d = W112._price_weight(detail)
    per = W112._stream_stride(os.path.join(rel, W112.STRIDE), pi_d, w_d, keep_cycles={181})
    entries = None
    with open(_abs(os.path.join(rel, W112.STRIDE))) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                if r['cycle'] == 181:
                    entries = r['entries']
                    break
    eval_id = f'p515s53_w118_t_sum_check_{int(time.time())}'
    work = os.path.join(O.WORK_DIR, eval_id)
    try:
        planning = O.fresh_planning(eval_id)
        years = list(planning.years)
        days = list(planning.days)
        ymap = {str(y): y for y in years}
        cv = {'pf': {side: {'current': {n: {y: {d: {'p': [None] * planning.num_instants,
                                                    'q': [None] * planning.num_instants} for d in days}
                                            for y in years} for n in planning.active_distribution_network_nodes}}
                     for side in ('tso', 'dso')}}
        for e in entries:
            y = ymap[str(e['year'])]
            cv['pf']['dso']['current'][e['node_id']][y][e['day']][e['power_type']][e['period']] = e['x_dso']
            cv['pf']['tso']['current'][e['node_id']][y][e['day']][e['power_type']][e['period']] = e['z_tso_current']
        tn = planning.transmission_network
        one_market = all(len(tn.network[y][d].prob_market_scenarios) == 1 for y in years for d in days)
        tso_stub = {y: {d: SimpleNamespace(scenarios_market=[0]) for d in days} for y in years}
        fp_before = K105.fingerprint_planning(planning, srp)
        cv_before = _jt(cv)
        pw = R.interface_price_weight(planning, tso_stub, srp)
        ts, by_node, abs_gap = R.t_sum_from_consensus(planning, cv, pw)
        fp_after = K105.fingerprint_planning(planning, srp)
        cv_after = _jt(cv)
        mism_pi = [k for k, (pi, w) in pw.items() if pi != pi_d[(str(k[0]), str(k[1]), str(k[2]), k[3])]]
        mism_w = [k for k, (pi, w) in pw.items() if w != w_d[(str(k[0]), str(k[1]), str(k[2]), k[3])]]
    finally:
        if os.path.isdir(work) and not any(files for _r, _d, files in os.walk(work)):
            shutil.rmtree(work)
    ident = W112._detail_identity(detail)
    out = {'t_sum_in_cycle_function': ts, 't_sum_w112_formula_from_stride': per[181]['t_sum'],
           'terminal_t_tso_plus_t_dso': ident['t_tso_plus_t_dso_terminal'],
           'diff_vs_w112': ts - per[181]['t_sum'], 'diff_vs_terminal': ts - ident['t_tso_plus_t_dso_terminal'],
           'price_equal_bitwise_to_detail': not mism_pi, 'weight_equal_bitwise_to_detail': not mism_w,
           'n_price_weight_entries': len(pw), 'one_market_scenario_every_block': one_market,
           'by_node': by_node, 'sum_abs_gap_mw_p': abs_gap, 'price_weight_sha256': R.price_weight_digest(pw),
           'write_only_planning_fingerprint_unchanged': fp_before == fp_after,
           'write_only_consensus_unchanged': cv_before == cv_after}
    out['holds'] = bool(ts == per[181]['t_sum'] and abs(out['diff_vs_terminal']) <= 0.01 and not mism_pi
                        and not mism_w and one_market and len(pw) == 864 and fp_before == fp_after
                        and cv_before == cv_after)
    return out


def tests_H():
    import shared_resources_planning as srp
    res = {}
    h1 = {}
    for cell in R.GATED_CELLS:
        d = drive(cell, 'certify')
        s = _drive_summary(d)
        c = R.CELLS[cell]
        s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                       and s['n_lapses'] == 1
                       and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                       and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                       and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                       and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay']
                       and s['certificate_length_after'] == 10 and s['lines'] == s['k_star'] == s['creep_lines']
                       and s['holds_through_first_pass'] == [HOLDS_OFF] and s['holds_after_first_pass'] == [HOLDS_ON]
                       and s['t_sum_every_line'] and s['q_cc_every_line'] and not s['capture_errors'])
        h1[cell] = s
    res['H1_gated_original_values_bitwise_through_k0_then_certify'] = {'holds': all(v['ok'] for v in h1.values()),
                                                                        'cells': h1}
    d = drive('f2_challenger', 'ulp_at_40')
    s = _drive_summary(d)
    s['ok'] = bool(s['raised'] and 'REPLAY DIVERGED at cycle 40' in s['raised'] and s['lines'] == 40
                   and (s['first_divergence'] or {}).get('fields_differing') == ['gross_operational_cost']
                   and s['replay_bitwise_through'] == 39 and not s['summary_ok']
                   and s['stopped_by'] == 'replay_divergence_abort')
    res['H2_one_ulp_at_40_aborts'] = {'holds': s['ok'], 'detail': s}
    d = drive('pb_y2030_n9', 'gap')
    s = _drive_summary(d)
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['gap_refusals'] >= 1
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay']
                   and s['k_star'] > R.CELLS['pb_y2030_n9']['N_old'] + 80)
    res['H3_gap_refusal_then_certification'] = {'holds': s['ok'], 'detail': s}
    d = drive('pb_y2025_n7', 'creep')
    s = _drive_summary(d)
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'cap'
                   and s['last_cycle'] == R.spec_cap('pb_y2025_n7') and s['certificate_length_after'] == 10
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['ended_by'] == 'cap')
    res['H4_gated_creep_uncertified_at_the_cap'] = {'holds': s['ok'], 'detail': s}
    d = drive('yl_y2030', 'creep', first_pass_at=50)
    s = _drive_summary(d)
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'rule_cap'
                   and s['last_cycle'] == 50 + R.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 50
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['n_overlap'] == 0
                   and s['holds_through_first_pass'] == [HOLDS_OFF] and s['holds_after_first_pass'] == [HOLDS_ON])
    res['H5_ungated_dynamic_rule_cap'] = {'holds': s['ok'], 'detail': s}
    d = drive('yl_y2035', 'certify', first_pass_at=100)
    s = _drive_summary(d)
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['first_pass'] == 100 and s['summary_ok']
                   and s['in_cycle_rule_equals_pure_replay'] and s['stopped_by'] == 'settling_rule')
    res['H5b_ungated_certifies'] = {'holds': s['ok'], 'detail': s}
    try:
        res['H6_t_sum_real_srp1_data'] = _h_t_sum_real()
    except Exception as error:  # noqa: BLE001
        res['H6_t_sum_real_srp1_data'] = {'holds': False, 'error': f'{type(error).__name__}: {error}',
                                          'traceback': traceback.format_exc()}
    res['H7_aa_hold_real_production'] = _h_real(srp, 'aa')
    res['H8_tail_hold_real_production'] = _h_real(srp, 'tail')
    res['H9_rho_hold_real_production'] = _h_real(srp, 'rho')
    res['H10_layering_real_install'] = _h_layering(srp)
    got = R._certificate_length_writes_in_source()
    tampered = sorted(got + ['st.params.minimum_consecutive_converged_cycles = 0'])
    res['H11_certificate_length_writes'] = {
        'holds': (got == sorted(R.CERTIFICATE_LENGTH_WRITES) and tampered != sorted(R.CERTIFICATE_LENGTH_WRITES)
                  and 'early_stop' not in inspect.getsource(R.make_wrappers)),
        'writes_in_source': got, 'negative_control_extra_write_detected': tampered != sorted(R.CERTIFICATE_LENGTH_WRITES)}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w118():
    import importlib.util
    src = subprocess.run(['git', 'show', f"{PRE_W118_HARNESS['commit']}:p515_s44_campaign_harness.py"], cwd=REPO,
                         capture_output=True, check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != PRE_W118_HARNESS['sha256']:
        raise RuntimeError(f'pre-W118 harness sha256 {sha} != pinned {PRE_W118_HARNESS["sha256"]}')
    tmp = tempfile.mkdtemp(prefix='w118_pre_harness_')
    path = os.path.join(tmp, '_w118_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w118_pre_harness', path)
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
        certification_continuation=e.get('certification_continuation'),
        settling_continuation=e.get('settling_continuation'), settling_extension=e.get('settling_extension'))


def configuration_for(cell):
    """The re-settling campaign configuration (the current production configuration): the Phase B spec's case-file AA
    and C2 declarations, the tight tail declared on."""
    cfg = json.load(open(_abs(PHASE_B_SPEC_REL)))['configuration']
    return {'case_file_anderson_acceleration': dict(cfg['case_file_anderson_acceleration']),
            'ess_ageing_baseline': json.loads(json.dumps(cfg['ess_ageing_baseline'])),
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': {'enabled': True, 'compl_inf_tol': 1e-06}}


def resettle_keys(pre=None):
    out = {}
    for cell in R.CELL_ORDER:
        spec, e = _spec_entry(cell)
        cfg = configuration_for(cell)
        kw = dict(case_file_aa=cfg['case_file_anderson_acceleration'], ess_ageing_baseline=cfg['ess_ageing_baseline'],
                  flex_price_multiplier=e.get('flex_price_multiplier'),
                  convergence_depth_tail=cfg['convergence_depth_tail'])
        base = H.evaluation_key(e['key'], e['overrides'], **kw)
        decl = R.declaration_for(cell)
        key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=decl, **kw)
        base_pre = pre.evaluation_key(e['key'], e['overrides'], **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[cell] = {'candidate_key': e['key'], 'original_eval_key': e['eval_key'], 'base_key_now': base,
                     'resettle_key': key, 'formula_holds': key == formula, 'base_equals_pre_w118': base == base_pre,
                     'differs_from_original': key != e['eval_key'] and base != e['eval_key']}
    return out


def _key_holders(scanned, key):
    return sorted({rel for rel, spec in scanned for e in spec.get('candidates') or [] if H._entry_eval_key(e) == key})


def _outside_roots(rels, exclude_roots=KEY_EXCLUDED_ROOTS):
    return [r for r in rels if not any(r.startswith(root + os.sep) for root in exclude_roots)]


def tests_K():
    pre, pre_sha = _harness_pre_w118()
    n_specs = n_entries = n_equal = 0
    mismatch, errors = [], []
    scanned = []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        scanned.append((rel, spec))
        for e in spec.get('candidates') or []:
            n_entries += 1
            try:
                if 'settling_resettle' in e:
                    raise RuntimeError('a committed entry declares settling_resettle (not expected before the freeze)')
                args, kw = _key_args(spec, e)
                new, old = H.evaluation_key(*args, **kw), pre.evaluation_key(*args, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    keys = resettle_keys(pre)
    holders = {cell: _key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: _outside_roots(h) for cell, h in holders.items()}
    probe = keys['f2_challenger']['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w118_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    planted_outside_rel = os.path.join(_P53, 'w118_planted_negative_control', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_sibling_rel = os.path.join(W118_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_own_rel = os.path.join(W118_ROOT_REL, 'campaign_s53_w118_resettle_f2_challenger',
                                   'campaign_spec_s53_w118_resettle_f2_challenger_00000000.json')
    ctl_outside = _outside_roots(_key_holders(scanned + [planted(planted_outside_rel)], probe))
    ctl_sibling = _outside_roots(_key_holders(scanned + [planted(planted_sibling_rel)], probe))
    own_holders = _key_holders(scanned + [planted(planted_own_rel)], probe)
    ctl_own = _outside_roots(own_holders)
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_entry_equal': n_equal == n_entries and n_entries > 0,
        'resettle_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'resettle_base_equals_pre_w118_every_cell': all(v['base_equals_pre_w118'] for v in keys.values()),
        'resettle_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'resettle_keys_distinct': len({v['resettle_key'] for v in keys.values()}) == len(keys),
        'resettle_keys_absent_from_committed_specs_outside_own_root': not any(outside.values()),
        'control_planted_outside_root_refused': planted_outside_rel in ctl_outside,
        'control_planted_sibling_prefix_refused': planted_sibling_rel in ctl_sibling,
        'control_own_campaign_spec_accepted': planted_own_rel in own_holders and planted_own_rel not in ctl_own,
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w118_harness': {**PRE_W118_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'resettle_keys_in_committed_specs_all_REPORTED': holders, 'key_excluded_roots': list(KEY_EXCLUDED_ROOTS),
            'controls': {'planted_outside': {'rel': planted_outside_rel, 'outside_found': ctl_outside},
                         'planted_sibling': {'rel': planted_sibling_rel, 'outside_found': ctl_sibling},
                         'planted_own': {'rel': planted_own_rel, 'holders': own_holders, 'outside_found': ctl_own}}}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    out = {}
    ok = True
    for cell in R.CELL_ORDER:
        decl = R.declaration_for(cell)
        cap = R.spec_cap(cell)
        try:
            good = R.assert_resettle_preconditions(decl, {'cap': cap}, {'tail_enabled_for_this_run': True}, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        negatives = {}
        for name, (spec, tail, aa) in {'cap_wrong': ({'cap': cap - 1}, {'tail_enabled_for_this_run': True}, True),
                                       'tail_off': ({'cap': cap}, {'tail_enabled_for_this_run': False}, True),
                                       'aa_off': ({'cap': cap}, {'tail_enabled_for_this_run': True}, False)}.items():
            try:
                R.assert_resettle_preconditions(decl, spec, tail, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {str(error)[:200]}'
        rule = decl['settling_rule']
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'gap_bound_tau': dict(decl, settling_rule=dict(rule, gap_bound=SC2.TAU)),
               'cap_rule_changed': dict(decl, cap_rule={'kind': 'fixed', 'cap': 300, 'formula': 'x'}),
               'rule_module_v1': dict(decl, settling_rule=dict(rule, module='settling_criterion'))}
        if decl['replay_reference'] is not None:
            bad.update({'wrong_k0': dict(decl, first_residual_pass_expected=decl['first_residual_pass_expected'] - 1),
                        'wrong_sha': dict(decl, replay_reference=dict(decl['replay_reference'], sha256='0' * 64)),
                        'abort_false': dict(decl, abort_on_replay_divergence=False),
                        'no_reference': dict(decl, replay_reference=None)})
        else:
            bad.update({'invented_reference': dict(decl, replay_reference={'per_cycle_record': 'x', 'sha256': '0' * 64}),
                        'abort_true': dict(decl, abort_on_replay_divergence=True)})
        refused = {}
        for name, d in bad.items():
            try:
                R.validate_settling_resettle(d)
                refused[name] = False
            except ValueError as error:
                refused[name] = str(error)[:160]
        cell_ok = good_ok and all(v.startswith('refused') for v in negatives.values()) and all(refused.values())
        ok = ok and cell_ok
        out[cell] = {'ok': cell_ok, 'checklist': good, 'negative_controls': negatives, 'validator_refuses': refused}
    return {'holds': ok, 'cells': out}


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('R', tests_R), ('O', tests_O), ('H', tests_H), ('K', tests_K), ('P', tests_P))


def run_all_checks():
    out = {}
    ok = True
    for sid, fn in SECTIONS:
        t0 = time.time()
        try:
            r = fn()
        except Exception as error:  # noqa: BLE001 -- recorded as a failing section
            r = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        r_ok = r.get('holds') is True
        out[sid] = {'holds': r_ok, 'wall_s': time.time() - t0, 'result': r}
        ok = ok and r_ok
    return {'all_hold': ok, 'p_max': R.P_MAX, 'constants': SC2.constants(R.P_MAX), 'readings': SC2.READINGS,
            'sections': out}


CODE_PINNED_BY_CHECKS = (os.path.basename(__file__), 'settling_criterion_v2.py', 'settling_criterion.py',
                         'p515_s53_w118_resettle_hooks.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py',
                         'interface_dual_capture.py', 'shared_resources_planning.py', 'admm_anderson_acceleration.py',
                         'p515_s53_w105_settling_extension_hooks.py', 'p515_s53_w101_settling_continuation_hooks.py',
                         'p515_s53_w105_extension_checks.py', 'p515_s53_w101_continuation_checks.py',
                         'p515_s53_w98_continuation_checks.py', 'p515_s53_w112_consensus_gap.py',
                         'p515_g_g1_g4_admm_gates.py', 'p515_gate_result_bool_typing_test.py')


def run_typing_test(out_dir):
    """W100's repository-wide boolean-typing test, output to a new write-once file (subprocess; stdlib only)."""
    out_path = os.path.join(out_dir, TYPING_OUT)
    if os.path.exists(out_path):
        raise SystemExit(f'refusing to overwrite existing artifact: {out_path}')
    t0 = time.time()
    proc = subprocess.run([sys.executable, '-u', 'p515_gate_result_bool_typing_test.py', '--out', out_path], cwd=REPO,
                          capture_output=True, text=True)
    doc = json.load(open(out_path)) if os.path.isfile(out_path) else None
    return {'exit_code': proc.returncode, 'pass': proc.returncode == 0, 'wall_s': time.time() - t0,
            'out': os.path.relpath(out_path, REPO), 'stdout_tail': proc.stdout[-2000:], 'stderr_tail': proc.stderr[-2000:],
            'verdict': (doc or {}).get('verdict'), 'n_files_scanned': (doc or {}).get('n_files_scanned')}


def main():
    started = _utc()
    out_dir = _abs(OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_FILE, OUT_MANIFEST, TYPING_OUT):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    res = run_all_checks()
    typing = run_typing_test(out_dir)
    code_pins = {rel: H.sha256_file(_abs(rel)) for rel in CODE_PINNED_BY_CHECKS}
    guards = {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}
    doc = {'schema': 'p515_s53_w118_zero_solve_checks_v1',
           'task': 'W118 (PLANNER_BRIEF_2026-09-13.md Addendum 57 Decisions 2 and 3; Addendum 54 Ruling 2)',
           'started_utc': started, 'finished_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
           'code_sha256': code_pins, 'guards': guards, 'W_bool_typing_test': typing, **res,
           'all_hold_including_typing_test': bool(res['all_hold'] and typing['pass'])}
    path = os.path.join(out_dir, OUT_FILE)
    with open(path, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    manifest = {os.path.relpath(path, REPO): H.sha256_file(path)}
    tpath = os.path.join(out_dir, TYPING_OUT)
    if os.path.isfile(tpath):
        manifest[os.path.relpath(tpath, REPO)] = H.sha256_file(tpath)
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    for sid, r in res['sections'].items():
        print(f"[W118-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W118-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W118-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) guards={guards}")
    print(f"[W118-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
