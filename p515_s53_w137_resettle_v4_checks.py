"""
P5.15 Addendum 59 and its Supplement, Planner task W137 -- ZERO-SOLVE checks for the v4 re-settling campaign (the same 38
cells as W132; frozen_s53_resettle_spec_v4, predecessor v3 139d1e62). W132's section set, re-targeted to v4.

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W132 checks
module, imported for its helpers, arms its own and the modules it imports arm theirs; every guard is verified at 0).
No model is built and nothing is solved: H12 and H6-H9 read a fresh SRP1 planning object (data only); everything else
reads committed, manifest- or pin-verified records.

WHAT IS CHECKED
  V   the stop rule VERSION 4 (`settling_criterion_v4`): V_v2 the W118 version-2 suite (unchanged, run as committed);
      V_eq version 4 with every cycle Optimal reproduces version 2 record by record and decision by decision on W132's
      trajectories; V_gamma version 4 equals version 3's gamma shadow (W131's construction) on every trajectory with
      planted non-Optimal cycles; the UNIT TESTS: V4a an Acceptable cycle OUTSIDE the window does not block (same k*,
      same window, recorded among the out-of-window reads); V4b one INSIDE the window blocks (vetoed at k*, reason
      certification_vetoed_non_optimal, certified later once it leaves the window); V4c there is NO reset (k0, N, the
      sign state and the turning points identical to the all-Optimal run through the veto; no lapse); V4d a Boyd lapse
      still resets; V4e the monotone veto; V4f a veto at the cap is a failing reason; V4g the sub-test read enumeration
      and the out-of-window reads; V21 the per-cell ceiling; V22 all_optimal must be a bool; V0 versions 1, 2 and 3
      byte-identical to their pins.
  R   VALIDATION BY REPLAY on the 13 cells (W136's inputs): the 10 W118 cells and the 2 settled references fed W131's
      per-cycle all_optimal (w131_prefreeze_diagnostics.json afd5a0d7, pinned), and W133's cell 1 (b_2a0ba8b2, ed71177e)
      fed its committed in-run exit capture (all_optimal_k, G24-cross-checked): cell 1 certifies at 173 (= its v3
      gamma shadow), pb_y2025_n9 keeps 174, pb_y2025_n5 is NOT certified within its records, the other 10 keep their
      committed decision; each certification with its out-of-window reads.
  O   the 38 original cells (W132's section O, reused unchanged: the cells are W132's).
  S   production_since_originals: no uncommitted change to a file this run uses (gate); production changes since each
      original's git head (report).
  H   the hooks through the REAL nine wrappers (W132's, reused) on the v4 state with stand-in production: H1 every gated
      cell with its ORIGINAL values bitwise through k0, then a synthetic continuation that certifies; H2 a one-ulp
      divergence aborts; H3 a non-Optimal ESSO exit OUTSIDE the window does not block (no lapse, same k*); H3b one
      INSIDE the window vetoes (no lapse, later k*); H4 a gated creep uncertified at the cap and the record label fix;
      H5 / H5b ungated dynamic rule cap / certification; H6 2ab0ce2d to its cap 437; H7-H9 the AA / tail / rho holds
      with REAL production; H10 the real install and the harness dispatch (v4 -> this, v3 -> W132, W135 -> W135, W118 ->
      W118); H11 certificate-length writes; H12 W132's exit wrapper on REAL production (reused).
  K   keys: every entry of every committed campaign spec keys identically under this harness and the pre-W137 harness
      (9fd91ded, bf0a757c, sha256 pinned, from git) -- the 38 W132 v3 entries, W118's and every other committed key
      included -- except the OWN entries of the two campaigns this harness adds a route for: v4 entries accepted only
      inside the v4 stage root, W135 entries only inside the W135 stage root, each only if its frozen key follows the
      formula over the pre-W137 base key (the rule: a pre-run check that scans committed artefacts excludes the run's
      own, applied per root); the 38 v4 keys follow the formula, are distinct, differ from the originals and from the
      38 v3 keys, and appear in no committed spec outside the v4 root; planted controls; the pre-W137 harness refuses a
      v4 declaration.
  P   `assert_resettle_preconditions` holds for every cell with its cap; negative controls refused; the validator
      refuses early_stop and malformed declarations (a v3 rule declaration included).
  X   W132's status-label writer checks (reused) and G8 ON PRODUCTION'S CERTIFICATE: W133's committed cell-1 record
      (settling label not_certified, production certified, pkl written) passes; synthetic controls fail.
  D   the harness is EXACTLY the pre-W137 harness plus W135's prepared patch plus the v4 branch (6 added lines in
      `resettle_hooks_module`, none removed; every existing branch identical); its sha256 is the pinned post-v4 sha.
  W   (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, alone, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w137_resettle_v4_checks.py \\
      > data/SRP1/Results/P515S53/w137_resettle_v4/zero_solve_checks_launch.log 2>&1
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W137 v4 re-settling zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w137_resettle_v4_hooks as V4  # noqa: E402
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w135_resettle_ext_hooks as W135  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w132_resettle_v3_checks as K132  # noqa: E402 -- the v3 suite and helpers (arms its own guard)
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v3 as SC3  # noqa: E402
import settling_criterion_v4 as SC4  # noqa: E402

K118, K105, K98 = K132.K118, K132.K105, K132.K98


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w137_checks', GUARD),) + tuple(K132.GUARDS))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W137_ROOT_REL = os.path.join(_P53, 'w137_resettle_v4')
W135_ROOT_REL = os.path.join(_P53, 'w135_resettle_ext')
W132_ROOT_REL = K132.W132_ROOT_REL
W118_ROOT_REL = K132.W118_ROOT_REL
OUT_DIR_REL = os.path.join(W137_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w137_zero_solve_checks.json'
OUT_MANIFEST = 'w137_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w137_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = 's53_w137_resettle_v4_'
W135_CAMPAIGN_ID_PREFIX = 's53_w135_resettle_ext_'
KEY_OWN_ROOTS = {'v4': W137_ROOT_REL, 'w135': W135_ROOT_REL}
# the harness before W137 (pinned by the W132 stage spec 139d1e62; last changed by 9fd91ded)
PRE_W137_HARNESS = {'commit': '9fd91ded4e68ec848e2a3684621c9c7016267f3d',
                    'sha256': 'bf0a757c0d36da406c5388f2af4f36aed84e6baa4f569f3658e3218b449a5c6a'}
W135_PATCH_REL = 'p515_s53_w135_harness_router.patch'
HARNESS_POST_W135_PATCH_SHA256 = '861d6070c107d9f4930ca3b6961945d99440d77a7d43b75631adee14ecc0442d'
V4_BRANCH_LINES = ('    import p515_s53_w137_resettle_v4_hooks as W137C',
                   '    if W137C.is_v4_declaration(value):',
                   '        return W137C')
HARNESS_POST_V4_SHA256 = '25cc4559db5ebe9904d9ed8f97f9a6182263d3e5126b935e47dc21727040ce10'
W132_STAGE_SPEC = {'path': os.path.join(W132_ROOT_REL, 'frozen_s53_resettle_spec_v3_139d1e62.json'),
                   'sha256': '139d1e62339248df16b6fbf8019de9fb8569285f658b7fa7b1cd1ed71d060136'}
W131 = K132.W131
W118_CELL_DIRS = K132.W118_CELL_DIRS
REF_CELLS = K132.REF_CELLS
# W133's cell 1 (b_2a0ba8b2) under v3, ed71177e: the committed run whose trajectory v4 is predicted to reproduce
CELL1 = 'b_2a0ba8b2'
CELL1_V3 = {'campaign_root': os.path.join(W132_ROOT_REL, 'campaign_s53_w132_resettle_v3_b_2a0ba8b2'),
            'eval_dir': os.path.join(W132_ROOT_REL, 'campaign_s53_w132_resettle_v3_b_2a0ba8b2', 'evals',
                                     'cf592cc94ce0d1ef_b_2a0ba8b2'),
            'commit': 'ed71177e', 'cap': 213, 'gamma_shadow_k_star': 173}
EXPECTED_V4_REPLAY = {
    'rule': ('Addendum 59 Supplement (reading (a)): cell 1 certifies at 173 (the endorsed prediction), pb_y2025_n9 '
             'keeps 174, pb_y2025_n5 is not certified within its records, the other ten keep their committed k* (the '
             'two F2 cells stay uncertified at their caps)'),
    'k_star': {'b_2a0ba8b2': 173, 'pb_y2025_n9': 174},
    'not_certified_within_records': ['pb_y2025_n5'],
}
PRODUCTION_FILES = K132.PRODUCTION_FILES


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jt(x):
    return K132._jt(x)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _clean(rel):
    return K118._git_tracked_clean(rel)


# ======================================================================================================================
#  V -- the stop rule v4
# ======================================================================================================================
_V4_ONLY_KEYS = ('all_optimal_k', 'vetoed', 'out_of_window_reads')
_V4_DEC_KEYS = ('version', 'reading', 'N_definition', 'window_rule', 'non_optimal_cycles', 'vetoes', 'n_vetoes',
                'cap_ceiling', 'window_all_optimal', 'out_of_window_reads', 'certification_vetoed_at_cap')


def _v2_view(rec):
    return {k: v for k, v in rec.items() if k not in _V4_ONLY_KEYS}


def _dec_view(dec):
    return None if dec is None else {k: v for k, v in dec.items() if k not in _V4_DEC_KEYS}


def _replay4(q, b, t, ao, kw):
    kw4 = dict(kw)
    kw4.setdefault('cap_ceiling', SC2.CAP_CEILING)
    return SC4.replay(q, b, t, ao, R.P_MAX, **kw4)


def _gamma_alone(q, b, t, ao, n, kw):
    """Version 3's gamma shadow run alone over cycles 1..n (W131's construction; K132's helper)."""
    kw3 = dict(kw)
    kw3.setdefault('cap_ceiling', SC2.CAP_CEILING)
    return K132._gamma_alone(q, b, t, ao, n, kw3)


def _summ(d):
    d = d or {}
    return {k: d.get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'window', 'branch')}


def tests_V():
    res = {}
    v2 = K118.tests_V()
    res['V_v2_suite_as_committed'] = {'ok': v2['holds'] is True,
                                      'tests': {k: v.get('ok') for k, v in v2['tests'].items()}}
    trajs, q6, b6, k6 = K132._trajectories()
    eq = {}
    for name, q, b, t, kw in trajs:
        o2, d2, _r2 = SC2.replay(q, b, t, R.P_MAX, **kw)
        o4, d4, r4 = _replay4(q, b, t, {k: True for k in q}, kw)
        recs_equal = [_jt(_v2_view(a)) for a in o4] == [_jt(x) for x in o2]
        dec_equal = _jt(_dec_view(d4)) == _jt({k: v for k, v in (d2 or {}).items() if k != 'version'} if d2 else None)
        eq[name] = {'ok': bool(recs_equal and dec_equal and (d4 or {}).get('version') == 4 and not r4.vetoes
                               and not r4.non_optimal_cycles),
                    'records_equal': recs_equal, 'decision_equal': dec_equal,
                    'status': (d4 or {}).get('status'), 'k_star': (d4 or {}).get('k_star')}
    res['V_eq_all_optimal_reproduces_version_2'] = {'ok': all(v['ok'] for v in eq.values()), 'trajectories': eq}
    # V_gamma: version 4 == version 3's gamma shadow run alone (W131's construction) with planted non-Optimal cycles
    gam = {}
    for name, q, b, t, kw in trajs:
        _o, d0, _r = _replay4(q, b, t, {k: True for k in q}, kw)
        n = max(q)
        ks = sorted(q)
        plants = {'none': set(), 'every_7th': {k for k in ks if k % 7 == 0}}
        if (d0 or {}).get('status') == 'certified':
            ks_ = d0['k_star']
            plants['at_k_star_minus_1'] = {ks_ - 1}
            plants['just_before_window'] = {d0['window'][0] - 1}
        for pname, non in plants.items():
            ao = {k: k not in non for k in q}
            _o4, d4, _r4 = _replay4(q, b, t, ao, kw)
            g = _gamma_alone(q, b, t, ao, n, kw)
            same = _summ(d4) == _summ(g)
            gam[f'{name}|{pname}'] = {'ok': bool(same), 'v4': _summ(d4), 'gamma_v3': _summ(g)}
    res['V_gamma_v4_equals_v3_gamma_shadow'] = {'ok': all(v['ok'] for v in gam.values()), 'n': len(gam),
                                                'failing': {k: v for k, v in gam.items() if not v['ok']}}
    # the damped cosine (q6, b6): all-Optimal k* = k6
    t0 = {k: 0.0 for k in q6}
    o_all, d_all, r_all = _replay4(q6, b6, t0, {k: True for k in q6}, {'cap': 200})
    win = d_all['window']
    # V4a: an Acceptable cycle OUTSIDE the window (just before it, after k0) does not block
    out_c = win[0] - 1
    o_a, d_a, r_a = _replay4(q6, b6, t0, {k: k != out_c for k in q6}, {'cap': 200})
    oow = d_a.get('out_of_window_reads') or {}
    res['V4a_acceptable_outside_the_window_does_not_block'] = {
        'ok': bool(d_a['status'] == 'certified' and d_a['k_star'] == d_all['k_star'] == k6 and d_a['window'] == win
                   and not r_a.vetoes and r_a.lapses == [] and d_a['window_all_optimal'] is True
                   and out_c in oow.get('non_optimal_cycles_read_outside_window', [])
                   and oow.get('any_non_optimal_read_outside_window') is True),
        'k_star': d_a['k_star'], 'k_star_all_optimal': d_all['k_star'], 'window': d_a['window'],
        'non_optimal_cycle': out_c, 'out_of_window_reads_non_optimal': oow.get('non_optimal_cycles_read_outside_window')}
    # V4b: one INSIDE the window blocks at k6 (reason recorded), certified later once it has left the window
    in_c = k6 - 2
    o_b, d_b, r_b = _replay4(q6, b6, t0, {k: k != in_c for k in q6}, {'cap': 200})
    rec_k6 = o_b[k6 - 1]
    res['V4b_acceptable_inside_the_window_blocks'] = {
        'ok': bool(rec_k6['vetoed'] is not None and SC4.VETO_REASON in (rec_k6['reasons'] or [])
                   and rec_k6['decision'] == SC4.VETO_REASON and r_b.vetoes and r_b.vetoes[0]['cycle'] == k6
                   and all(in_c in br['non_optimal_in_window'] for v in r_b.vetoes for br in v['branches'])
                   and d_b['status'] == 'certified' and d_b['k_star'] > k6
                   and not (d_b['window'][0] <= in_c <= d_b['window'][1]) and d_b['window_all_optimal'] is True),
        'veto_at_k6': rec_k6['vetoed'], 'k_star': d_b['k_star'], 'window': d_b['window'], 'n_vetoes': len(r_b.vetoes),
        'non_optimal_cycle': in_c}
    # V4c: NO reset -- through k6 the rule records equal the all-Optimal run's except the veto fields
    same_through = all(_jt({k: v for k, v in a.items() if k not in ('all_optimal_k', 'vetoed', 'decision', 'reasons')})
                       == _jt({k: v for k, v in b.items() if k not in ('all_optimal_k', 'vetoed', 'decision', 'reasons')})
                       for a, b in zip(o_b[:k6 - 1], o_all[:k6 - 1]))
    res['V4c_no_reset'] = {
        'ok': bool(same_through and r_b.lapses == [] and rec_k6['k0'] == o_all[k6 - 1]['k0'] == d_all['k0']
                   and rec_k6['T'] == o_all[k6 - 1]['T'] and d_b['k0'] == d_all['k0'] and d_b['N'] == d_all['N']
                   and d_b['lapse_events'] == []),
        'k0_after_the_non_optimal_cycle': d_b['k0'], 'k0_all_optimal': d_all['k0'], 'lapses': r_b.lapses,
        'records_through_k6_equal_but_the_veto': same_through}
    # V4d: a Boyd lapse still resets (as version 2)
    q7, b7 = q6, {k: (b6[k] and k != 36) for k in q6}
    _o2, d2l, _ = SC2.replay(q7, b7, t0, R.P_MAX, cap=200)
    _o4, d4l, r4l = _replay4(q7, b7, t0, {k: True for k in q7}, {'cap': 200})
    res['V4d_boyd_lapse_still_resets'] = {'ok': bool(d4l['k0'] == d2l['k0'] == 37 and d4l['k_star'] == d2l['k_star']
                                                     and len(r4l.lapses) == 1 and r4l.lapses[0]['cycle'] == 36),
                                          'k0': d4l['k0'], 'k_star': d4l['k_star']}
    # V4e: the monotone veto on the decaying creep (certifies monotone when all-Optimal)
    q9 = {1: 1e9}
    for k in range(2, 201):
        q9[k] = q9[k - 1] - 150.0 * 0.97 ** (k - 1)
    b9, t9 = {k: True for k in q9}, {k: 0.0 for k in q9}
    _o, d9, _r = _replay4(q9, b9, t9, {k: True for k in q9}, {'cap': 200})
    km = d9['k_star']
    nm = km - 30
    o9v, d9v, r9v = _replay4(q9, b9, t9, {k: k != nm for k in q9}, {'cap': 200})
    res['V4e_monotone_veto'] = {
        'ok': bool(d9['branch'] == 'monotone' and o9v[km - 1]['vetoed'] is not None
                   and o9v[km - 1]['vetoed']['branches'][-1]['branch'] == 'monotone'
                   and (d9v['status'] != 'certified' or d9v['k_star'] > km and d9v['window'][0] > nm)
                   and r9v.lapses == []),
        'k_star_all_optimal': km, 'branch_all_optimal': d9['branch'], 'non_optimal_cycle': nm,
        'decision_with_it': _summ(d9v), 'n_vetoes': len(r9v.vetoes)}
    # V4f: a veto AT the cap is a failing reason of the uncertified decision
    cap_f = k6
    _o, d_f, _r = _replay4(q6, b6, t0, {k: k != in_c for k in q6}, {'cap': cap_f})
    res['V4f_veto_at_the_cap_is_a_failing_reason'] = {
        'ok': bool(d_f['status'] == 'uncertified' and d_f['k_cap'] == cap_f and SC4.VETO_REASON in d_f['reasons']
                   and d_f['certification_vetoed_at_cap'] is True),
        'reasons': d_f['reasons']}
    # V4g: the sub-test read enumeration and the out-of-window reads of an oscillatory certification
    enum = SC4.SUB_TEST_READS
    osc = {e['sub_test']: e['inside_the_window_as_implemented'] for e in enum['oscillatory']}
    mono = {e['sub_test']: e['inside_the_window_as_implemented'] for e in enum['monotone']}
    oa = d_all['out_of_window_reads']
    T = d_all['T']
    tp_out = [x[0] for x in T if x[0] < win[0]]
    res['V4g_sub_test_reads_enumerated_and_recorded'] = {
        'ok': bool(osc == {'at_least_3_turning_points': False, 'swings_non_increasing_all_pairs': False,
                           'P_hat_and_W': False, 'window_inside_run': False, 'range_le_tau': True,
                           'gap_clause': True, 'veto': True}
                   and mono == {'lo_ge_k0_plus_K_EXCL': False, 'no_sign_change_in_window': False,
                                'range_le_tau': True, 'steps_decreasing': False, 'last_step_times_L_le_tau': True,
                                'gap_clause': True, 'veto': True}
                   and oa['turning_points_outside_window'] == tp_out and oa['any_out_of_window_read'] is True
                   and oa['reads']['P_hat']['P_hat'] == T[-1][0] - T[-3][0]
                   and oa['reads']['sign_state_Q_span']['outside_window'] == [d_all['k0'] + SC4.K_EXCL - 1, win[0] - 1]
                   and d9['out_of_window_reads']['reads']['steps_decreasing_first_step']['cycles_outside_window']
                   == [d9['window'][0] - 1]),
        'oscillatory_inside_flags': osc, 'monotone_inside_flags': mono, 'damped_cosine_out_of_window_reads': oa}
    # V21: the per-cell ceiling
    try:
        r437 = SC4.SettlingRuleV4(R.P_MAX, cap=437, cap_ceiling=437)
        ok437 = r437.cap == 437
    except Exception:  # noqa: BLE001
        ok437 = False
    refused = {}
    for name, kw in (('cap_437_ceiling_300', {'cap': 437, 'cap_ceiling': 300}), ('no_ceiling', {'cap': 200}),
                     ('no_ceiling_dynamic', {'cap_after_first_k0': 109})):
        try:
            SC4.SettlingRuleV4(R.P_MAX, **kw)
            refused[name] = False
        except ValueError:
            refused[name] = True
    res['V21_per_cell_ceiling'] = {'ok': bool(ok437 and all(refused.values())), 'cap_437_ceiling_437_accepted': ok437,
                                   'refused': refused}
    bad = {}
    for v in (None, 1, 'True'):
        try:
            SC4.SettlingRuleV4(R.P_MAX, cap=10, cap_ceiling=300).observe(1, 1.0, True, 0.0, v)
            bad[repr(v)] = False
        except TypeError:
            bad[repr(v)] = True
    res['V22_all_optimal_is_a_bool'] = {'ok': all(bad.values()), 'refused': bad}
    # V0: versions 1, 2 and 3 byte-identical to their pins
    now1, now2, now3 = _sha('settling_criterion.py'), _sha('settling_criterion_v2.py'), _sha('settling_criterion_v3.py')
    pins1 = {k: (json.load(open(_abs(p)))['code_sha256'] or {}).get('settling_criterion.py')
             for k, p in K118.V1_PINS.items()}
    pin2 = json.load(open(_abs(K132.W118_SPEC['path'])))['code_sha256'].get('settling_criterion_v2.py')
    ss = json.load(open(_abs(W132_STAGE_SPEC['path'])))
    pin3 = ss['pins']['code_sha256'].get('settling_criterion_v3.py')
    res['V0_versions_1_2_3_byte_identical'] = {
        'ok': bool(all(v == now1 for v in pins1.values()) and now2 == pin2 and now3 == pin3
                   and _sha(W132_STAGE_SPEC['path']) == W132_STAGE_SPEC['sha256']
                   and all(_clean(f) for f in ('settling_criterion.py', 'settling_criterion_v2.py',
                                               'settling_criterion_v3.py'))),
        'settling_criterion_py': {'now': now1, 'pins_v39_v41': pins1},
        'settling_criterion_v2_py': {'now': now2, 'pin_fc791891': pin2},
        'settling_criterion_v3_py': {'now': now3, 'pin_139d1e62': pin3}}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  R -- validation by replay on the 13 cells
# ======================================================================================================================
def cell1_v3_series():
    """W133's committed cell-1 run (ed71177e): (q, boyd, t_sum, all_optimal, n, inputs), every file verified against the
    campaign manifest committed with it."""
    ev = CELL1_V3['eval_dir']
    man = json.load(open(_abs(os.path.join(CELL1_V3['campaign_root'], 'campaign_manifest_sha256.json'))))
    inputs = {}
    for f in ('per_cycle_record.jsonl', R.CYCLE_FILE, R.DECISION_FILE, 'evaluation_record.json'):
        rel = os.path.join(ev, f)
        now = _sha(rel)
        if man.get(rel) != now or not _clean(rel):
            raise RuntimeError(f'{rel}: sha256 {now} != manifest {man.get(rel)} or not committed clean')
        inputs[rel] = now
    rows = _read_jsonl(_abs(os.path.join(ev, 'per_cycle_record.jsonl')))
    lines = {x['cycle']: x for x in _read_jsonl(_abs(os.path.join(ev, R.CYCLE_FILE)))}
    q = {r['cycle']: r['gross_operational_cost'] for r in rows}
    b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
    t = {k: lines[k].get('t_sum') for k in q}
    ao = {k: lines[k]['all_optimal_k'] is True for k in q}
    return q, b, t, ao, len(rows), inputs, json.load(open(_abs(os.path.join(ev, R.DECISION_FILE))))


def replay_inputs():
    """The 13 cells' series (W136's inputs): {cell: (q, b, t, all_optimal, n, kw, committed, inputs)}."""
    doc, sha = K132._pinned_json(W131)
    t1 = doc['task1']
    out = {}
    for cell, sub in W118_CELL_DIRS.items():
        eval_rel = os.path.join(W118_ROOT_REL, sub)
        inputs = K132._verify_w118(eval_rel, ['per_cycle_record.jsonl', R.CYCLE_FILE, R.DECISION_FILE])
        rows = _read_jsonl(_abs(os.path.join(eval_rel, 'per_cycle_record.jsonl')))
        lines = {x['cycle']: x for x in _read_jsonl(_abs(os.path.join(eval_rel, R.CYCLE_FILE)))}
        dec = json.load(open(_abs(os.path.join(eval_rel, R.DECISION_FILE))))
        w = t1[cell]
        n = len(rows)
        if not w['sources']['coverage_every_cycle_48_network_3_esso'] or w['cycles_recorded'] != n:
            raise RuntimeError(f'{cell}: W131 coverage / cycles {w["cycles_recorded"]} vs {n}')
        nonopt = set(w['counts']['non_optimal_cycles'])
        q = {r['cycle']: r['gross_operational_cost'] for r in rows}
        b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
        t = {k: (lines.get(k) or {}).get('t_sum') for k in q}
        cr = w['rule']['cap_rule']
        kw = ({'cap': cr['cap'], 'cap_ceiling': SC2.CAP_CEILING} if cr['kind'] == 'fixed' else
              {'cap_after_first_k0': cr['after_first_k0'], 'cap_ceiling': cr['ceiling']})
        out[cell] = (q, b, t, {k: k not in nonopt for k in q}, n, kw,
                     {k: dec.get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'window', 'branch')},
                     {**inputs, W131['path']: sha}, 'W131 task1 non_optimal_cycles (network records + ESSO logs)')
    for name, (w112_cell, _spec_rel) in REF_CELLS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            q, b, t, val, rows = K118.record_series(w112_cell)
        if not val['pass']:
            raise RuntimeError(f'{name}: terminal t_sum validation fails {val}')
        w = t1[name]
        nonopt = set(w['counts']['non_optimal_cycles'])
        cap = w['rule']['cap']
        out[name] = (q, b, t, {k: k not in nonopt for k in q}, len(rows), {'cap': cap, 'cap_ceiling': max(cap, 300)},
                     {k: w['committed_decision'].get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'window',
                                                                 'branch')},
                     {W131['path']: sha, 'terminal_t_sum_validation': val},
                     'W131 task1 non_optimal_cycles (network records + ESSO logs)')
    q, b, t, ao, n, inputs, dec3 = cell1_v3_series()
    out[CELL1] = (q, b, t, ao, n, {'cap': CELL1_V3['cap'], 'cap_ceiling': 300},
                  {'status': dec3.get('status'), 'k_star': dec3.get('k_star'), 'k_cap': dec3.get('k_cap'),
                   'v3_gamma_report_only_k_star': (dec3.get('gamma_report_only') or {}).get('k_star')},
                  inputs, 'the in-run exit capture all_optimal_k (resettle_cycle_record.jsonl; G24: 0 disagreements)')
    return out


def replay_cell(q, b, t, ao, n, kw):
    _o, d, r = _replay4(q, b, t, ao, dict(kw, last=n))
    if d is None:
        s = {'status': f'not certified within the recorded cycles 1..{n}', 'k_star': None, 'k_cap': None,
             'k0_at_end': r.k0, 'vetoes': list(r.vetoes)}
    else:
        s = {k: d.get(k) for k in ('status', 'k_star', 'k_cap', 'branch', 'k0', 'N', 'W', 'window', 'P_hat', 'T', 'A',
                                   'band_width', 'range_over_tau', 't_sum_k_star', 'reasons', 'window_all_optimal',
                                   'out_of_window_reads')}
        s['vetoes'] = list(r.vetoes)
        s['lapse_events'] = list(r.lapses)
        s['n_gap_refusals'] = len(r.gap_refusals)
    s['non_optimal_cycles'] = sorted(k for k, v in ao.items() if not v)
    return s


def tests_R():
    ins = replay_inputs()
    cells = {}
    for cell, (q, b, t, ao, n, kw, committed, inputs, source) in ins.items():
        s = replay_cell(q, b, t, ao, n, kw)
        if cell in EXPECTED_V4_REPLAY['k_star']:
            want = {'status': 'certified', 'k_star': EXPECTED_V4_REPLAY['k_star'][cell]}
        elif cell in EXPECTED_V4_REPLAY['not_certified_within_records']:
            want = {'status': f'not certified within the recorded cycles 1..{n}', 'k_star': None}
        else:
            want = {'status': committed['status'], 'k_star': committed.get('k_star')}
        ok = s['status'] == want['status'] and s['k_star'] == want['k_star']
        if committed.get('status') == 'uncertified' and cell not in EXPECTED_V4_REPLAY['k_star']:
            ok = ok and s.get('k_cap') == committed.get('k_cap')
        if cell == CELL1:
            ok = ok and s['k_star'] == committed.get('v3_gamma_report_only_k_star') == CELL1_V3['gamma_shadow_k_star']
        cells[cell] = {'ok': bool(ok), 'expected': want, 'v4': s, 'committed': committed, 'all_optimal_source': source,
                       'inputs_sha256': inputs}
    certified = {c: v['v4'] for c, v in cells.items() if v['v4']['status'] == 'certified'}
    return {'holds': all(v['ok'] for v in cells.values()) and len(cells) == 13, 'expected': EXPECTED_V4_REPLAY,
            'cells': cells,
            'summary': {c: {'status': v['v4']['status'], 'k_star': v['v4']['k_star'], 'k_cap': v['v4'].get('k_cap'),
                            'window': v['v4'].get('window'),
                            'turning_points_outside_window': (v['v4'].get('out_of_window_reads') or {}).get(
                                'turning_points_outside_window'),
                            'non_optimal_cycles_read_outside_window': (v['v4'].get('out_of_window_reads') or {}).get(
                                'non_optimal_cycles_read_outside_window'),
                            'n_vetoes': len(v['v4'].get('vetoes') or [])} for c, v in cells.items()},
            'n_certified': len(certified)}


# ======================================================================================================================
#  O, S -- W132's, reused (the cells are W132's)
# ======================================================================================================================
def tests_O():
    return K132.tests_O()


def tests_S():
    r = K132.production_since_originals(CODE_PINNED_BY_CHECKS)
    return {'holds': r['ok'], **r}


# ======================================================================================================================
#  H -- the hooks through the real wrappers, on the v4 state
# ======================================================================================================================
fake_results = K132.fake_results
HOLDS_OFF = K132.HOLDS_OFF
HOLDS_ON = K132.HOLDS_ON


def drive(cell, variant='certify', first_pass_at=50, nonopt=None, module=V4, state_cls=None):
    """W132's `drive` (p515_s53_w132_resettle_v3_checks.drive) restated for the v4 declaration and state: the REAL nine
    wrappers (`V.make_wrappers`) driven in production's call order with stand-in production. A gated cell: its
    ORIGINAL values through N_old, then a synthetic continuation; an ungated cell: a synthetic run (Boyd from
    `first_pass_at`). Variants: certify, ulp_at_40, gap, creep. `nonopt` = {cycle: [block keys]} injects Acceptable
    exits. `module` / `state_cls`: the declaration module and state (W135 reuses this with its own)."""
    decl = module.declaration_for(cell)
    ref = module.load_replay_reference(decl)
    cap = module.spec_cap(cell)
    sink = []
    st = (state_cls or V4.ResettleStateV4)(decl, None, cap, reference=ref, sink=sink)
    scripts = {'boyd': [], 'aa': [], 'next': [], 'pen': [], 'rc': [], 'efc': []}
    orig, calls = K105._standins(None, scripts)
    local_script = []
    orig['_admm_local_solves_succeeded'] = lambda pp_, results: local_script.pop(0)
    blocks_by_cycle = {}
    gated = decl['replay_reference'] is not None
    c_info = module.CELLS[cell]
    g_rows = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(module.original_eval_dir(cell),
                                                                          'g_s39_D.json'))))['cycle_trajectory']}
              if gated else {})
    n_old = c_info['N_old'] if gated else None
    q_anchor = ref[n_old]['gross_operational_cost'] if gated else 6.5e8
    nonopt = nonopt or {}

    def t_target(c):
        if gated and c <= n_old:
            return 12345.0
        j = c - (n_old if gated else first_pass_at)
        if variant == 'gap' and j <= 80:
            return 5000.0
        return 150.0
    pp, tso, dso, esso, cv, set_cycle = K118._fake_world(t_target)
    w = V.make_wrappers(st, orig, H.ipopt_exit_class, srp_module=K118._fake_srp(blocks_by_cycle, st),
                        classifier_label='p515_s44_campaign_harness.ipopt_exit_class (checks)')
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    models = {'tso': tso, 'dso': dso, 'esso': esso}
    esso_terms_orig = K105.E.esso_side_terms
    K105.E.esso_side_terms = lambda pp_, em: {str(n): {'salvage_value': 0.0, 'feasibility_penalty': 0.0}
                                              for n in K118.NODES}
    raised = None
    template = g_rows.get(n_old) if gated else json.load(open(_abs(os.path.join(
        V.original_eval_dir('b_2a0ba8b2'), 'g_s39_D.json'))))['cycle_trajectory'][-1]
    pen_hold = K118._frozen_pen(g_rows[c_info['k0']] if gated else template)
    try:
        local_script.append(True)
        w['_admm_local_solves_succeeded'](pp, fake_results(pp))       # the initialisation check (round 0)
        w['_capture_convergence_depth_tail_baseline'](pp, admm)
        for c in range(1, cap + 1):
            if gated and c <= n_old:
                row, g = ref[c], g_rows[c]
                bm = K105.boyd_from_g(g)
                rc = {'gross_operational_cost': row['gross_operational_cost'], 'net_operational_recourse': row['recourse'],
                      'terminal_salvage_value': row['terminal_salvage_value']}
                if variant == 'ulp_at_40' and c == 40:
                    rc['gross_operational_cost'] = math.nextafter(rc['gross_operational_cost'], math.inf)
                pen = K118._pen_row(g) if c <= c_info['k0'] else pen_hold
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
                if gated and variant in ('certify', 'gap') and c == n_old + 1:
                    passing = False
                bm['all_boyd_pass'] = bool(passing)
                q = q_anchor + qs
                rc = {'gross_operational_cost': q, 'net_operational_recourse': q, 'terminal_salvage_value': 0.0}
                pen = pen_hold
                efc = 1.0
                local_ok = True
                active = st.held(c)
            blocks_by_cycle[c] = K118._blocks_for(rc['gross_operational_cost'], rc['terminal_salvage_value'])
            set_cycle(c)
            scripts['boyd'].append(bm)
            scripts['aa'].append(None)
            scripts['next'].append(None)
            scripts['pen'].append(pen)
            if local_ok:
                scripts['rc'].append(rc)
            scripts['efc'].append(efc)
            local_script.append(bool(local_ok))
            with contextlib.redirect_stdout(io.StringIO()):
                w['_apply_convergence_depth_tail'](pp, admm, active, object(), c)
                w['get_admm_boyd_residual_metrics'](pp, tso, dso, esso, cv, {}, admm)
                w['_admm_local_solves_succeeded'](pp, fake_results(pp, nonopt.get(c, ())))
                if local_ok:
                    aa_rec = w['_anderson_acceleration_cycle_step'](object(), None, cv, {}, None, {}, bm, c)
                    w['_get_operational_recourse_components'](pp, models)
                else:
                    aa_rec = {'cycle': c, 'action': 'skipped (local solve failure this cycle)'}
                w['_convergence_depth_tail_next_state'](bool(bm['all_boyd_pass'] and local_ok), True, aa_rec)
                w['_update_admm_penalties']({}, {}, {}, {}, bm, object(), iter=c, allow_update=local_ok,
                                            freeze_state={})
                w['_get_admm_efc_per_day_max'](esso)
            if admm.minimum_consecutive_converged_cycles == V.SETTLING_END_THRESHOLD:
                break
        w['_apply_convergence_depth_tail'](pp, admm, False, object(), None)
    except RuntimeError as error:
        raised = str(error)
    finally:
        K105.E.esso_side_terms = esso_terms_orig
    files = {}
    for fname, obj in sink:
        files.setdefault(fname, []).append(obj)
    return {'state': st, 'files': files, 'raised': raised, 'calls': calls, 'admm': admm}


def pure_rule(decl):
    cr = decl['cap_rule']
    p_max = decl['settling_rule']['p_max']
    if cr['kind'] == 'fixed':
        return SC4.SettlingRuleV4(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return SC4.SettlingRuleV4(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


def pure_equal(decl, lines):
    """The pure v4 rule over the cycle lines (Q, the v2 boyd_k, t_sum, all_optimal_k) reproduces every in-cycle rule
    record; returns (equal, decision)."""
    rule = pure_rule(decl)
    pure = [rule.observe(x['cycle'], x.get('gross'), bool(x.get('boyd_k')), x.get('t_sum'), bool(x.get('all_optimal_k')))
            for x in lines]
    return [_jt(x.get('settling')) for x in lines] == [_jt(p) for p in pure], rule.decision


def _drive_summary(d):
    st, files = d['state'], d['files']
    lines = files.get(V.CYCLE_FILE, [])
    summ = st.summary()
    pure_ok, pure_dec = pure_equal(st.decl, lines) if lines else (False, None)
    fp = st.first_pass
    holds_pre = sorted({json.dumps(x['holds'], sort_keys=True) for x in lines if fp is None or x['cycle'] <= fp})
    holds_post = sorted({json.dumps(x['holds'], sort_keys=True) for x in lines if fp is not None and x['cycle'] > fp})
    dec = st.decision or {}
    return {'raised': d['raised'], 'lines': len(lines), 'creep_lines': len(files.get(V.CREEP_FILE, [])),
            'decision_files': len(files.get(V.DECISION_FILE, [])), 'summary_ok': summ['ok'],
            'stopped_by': summ['stopped_by'], 'status': summ['settling_status'], 'k_star': summ['k_star'],
            'branch': summ['branch'], 'window': dec.get('window'), 'first_pass': fp,
            'first_k0_v2': summ.get('first_k0_v2'), 'non_optimal_cycles': summ.get('non_optimal_cycles'),
            'n_vetoes': summ.get('n_vetoes'), 'vetoes': summ.get('vetoes'),
            'replay_bitwise_through': summ['replay_bitwise_through_cycle'],
            'first_divergence': summ['replay_first_divergence'], 'n_overlap': len(summ['overlap_k0_plus_1_to_N_old']),
            'overlap_all_zero': all(o['Q_new_minus_Q_old'] == 0.0 for o in summ['overlap_k0_plus_1_to_N_old']),
            'gap_refusals': len(summ['gap_refusals']), 'n_lapses': len(summ['lapse_events']),
            'certificate_length_after': d['admm'].minimum_consecutive_converged_cycles, 'ended_by': st.ended_by,
            'in_cycle_rule_equals_pure_replay': pure_ok, 'pure_decision_status': (pure_dec or {}).get('status'),
            'exit_capture_complete': summ.get('exit_capture_complete'),
            'exits_51_every_line': all((x.get('ipopt_exit_counts') or {}).get('total') == 51
                                       and len(x.get('ipopt_exit_by_block') or {}) == 51 for x in lines),
            'all_optimal_every_line_is_bool': all(isinstance(x.get('all_optimal_k'), bool) for x in lines),
            't_sum_every_line': all(isinstance(x.get('t_sum'), float) for x in lines),
            'holds_through_first_pass': holds_pre, 'holds_after_first_pass': holds_post,
            'decision_version': dec.get('version'), 'out_of_window_reads_recorded': dec.get('out_of_window_reads')
            is not None if dec.get('status') == 'certified' else None,
            'capture_errors': summ['capture_errors'][:3], 'errors': summ['errors'][:3],
            'last_cycle': st.cycle, 'rule_cap': st.rule.cap, 'summary': summ}


def _h_real_v4(srp, which):
    """K105's real-production hold tests on the v4 state and the nine wrappers (first residual pass preset at W105's
    hold cycle, so the holds engage from there exactly as there) -- W132's `_h_real_v3` on the v4 state."""
    real_state, real_make, real_wrapped, real_n = K105._state, K105.E.make_wrappers, K105.E.WRAPPED, K105.E.N_HOLD
    cell = V4.CELL_ORDER[0]

    def state():
        st = V4.ResettleStateV4(V4.declaration_for(cell), None, V4.spec_cap(cell), reference={}, sink=[])
        st.first_pass = real_n
        return st
    K105._state = state
    K105.E.make_wrappers = lambda st, originals, srp_module=None: V.make_wrappers(
        st, {**originals, '_admm_local_solves_succeeded': originals.get('_admm_local_solves_succeeded')},
        H.ipopt_exit_class, srp_module=srp_module)
    K105.E.WRAPPED = V.WRAPPED
    try:
        if which == 'aa':
            return K105._h_aa_real()
        if which == 'tail':
            return K105._h_tail_real(srp)
        return K105._h_rho_real(srp)
    finally:
        K105._state, K105.E.make_wrappers, K105.E.WRAPPED = real_state, real_make, real_wrapped


def dispatch_checks():
    """The harness routes each declaration family to its module and validates it unchanged."""
    v4d, v3d = V4.declaration_for(CELL1), V.declaration_for(CELL1)
    w135d, w118d = W135.declaration_for(W135.CELL_ORDER[0]), R.declaration_for('f2_challenger')
    return {'v4_to_w137': H.resettle_hooks_module(v4d) is V4,
            'v3_to_w132': H.resettle_hooks_module(v3d) is V,
            'w135_to_w135': H.resettle_hooks_module(w135d) is W135,
            'w118_to_w118': H.resettle_hooks_module(w118d) is R,
            'harness_validates_v4': H.validate_settling_resettle(v4d) == v4d,
            'harness_validates_v3_unchanged': H.validate_settling_resettle(v3d) == v3d,
            'harness_validates_w135': H.validate_settling_resettle(w135d) == w135d,
            'harness_validates_w118_unchanged': H.validate_settling_resettle(w118d) == w118d}


def _h_layering(srp):
    """W132's real-install test (`K132._h_layering`) restated for the v4 state and the four-way dispatch."""
    before = {name: getattr(srp, name) for name in V.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w137_layering_')
    cell = CELL1
    n = V4.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with V4.settling_resettle_hooks(scratch, V4.declaration_for(cell), holder, cap=V4.spec_cap(cell)) as st:
            installed = {name: getattr(srp, name) for name in V.WRAPPED}
            all_installed = all(installed[k] is not before[k] for k in V.WRAPPED)
            v4_state = isinstance(st, V4.ResettleStateV4) and isinstance(st.rule, V4.HookedRuleV4)
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
    i_disp = child_src.find('W118C = resettle_hooks_module(resettle)')
    i_pre = child_src.find('W118C.assert_resettle_preconditions(resettle, spec, tail_checklist, aa_on)')
    i_cm = child_src.find('W118C.settling_resettle_hooks(eval_dir, resettle, holder, int(spec[\'cap\']))')
    pre_ok = 0 < i_disp < i_pre < child_src.find('G.run_admm_arm(') and i_cm > i_pre
    disp = dispatch_checks()
    summ = holder.get(V.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and order_ok and pre_ok and idc_outer and all_installed
                          and v4_state and all(disp.values()) and summ.get('phase') == 'ended'
                          and summ.get('schema') == V4.SCHEMA
                          and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                          and admm.minimum_consecutive_converged_cycles == 10),
            'nine_wrappers_installed': all_installed, 'v4_state_and_rule': v4_state,
            'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'child_real_order_continuation_settling_extension_resettle_idc_s38_appender': order_ok,
            'child_real_dispatch_then_checklist_before_run_admm_arm': pre_ok, 'harness_dispatch': disp,
            'files_written_by_install_without_cycles': files_written, 'summary_phase': summ.get('phase'),
            'summary_schema': summ.get('schema'), 'summary_errors': summ.get('errors')}


def tests_H():
    import shared_resources_planning as srp
    res = {}
    h1 = {}
    for cell in V4.GATED_CELLS:
        s = _drive_summary(drive(cell, 'certify'))
        c = V4.CELLS[cell]
        s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                       and s['first_k0_v2'] == c['k0'] and s['n_lapses'] == 1 + len(c['original_lapses_after_k0'])
                       and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                       and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                       and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                       and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['exit_capture_complete']
                       and s['exits_51_every_line'] and s['all_optimal_every_line_is_bool']
                       and s['non_optimal_cycles'] == [] and s['n_vetoes'] == 0 and s['certificate_length_after'] == 10
                       and s['lines'] == s['k_star'] == s['creep_lines'] and s['decision_version'] == 4
                       and s['out_of_window_reads_recorded'] is True
                       and s['holds_through_first_pass'] == [HOLDS_OFF] and s['holds_after_first_pass'] == [HOLDS_ON]
                       and s['t_sum_every_line'] and not s['capture_errors'])
        s.pop('summary')
        h1[cell] = s
    res['H1_gated_original_values_bitwise_through_k0_then_certify'] = {'holds': all(v['ok'] for v in h1.values()),
                                                                        'cells': h1}
    s = _drive_summary(drive('l_7b199ef9', 'ulp_at_40'))
    s.pop('summary')
    s['ok'] = bool(s['raised'] and 'REPLAY DIVERGED at cycle 40' in s['raised'] and s['lines'] == 40
                   and (s['first_divergence'] or {}).get('fields_differing') == ['gross_operational_cost']
                   and s['replay_bitwise_through'] == 39 and not s['summary_ok']
                   and s['stopped_by'] == 'replay_divergence_abort')
    res['H2_one_ulp_at_40_aborts'] = {'holds': s['ok'], 'detail': s}
    cell = 'h_74eda68d'
    base = _drive_summary(drive(cell, 'certify'))
    lo = base['window'][0]
    s_out = _drive_summary(drive(cell, 'certify', nonopt={lo - 1: ('ESSO|7',)}))
    s3 = {k: v for k, v in s_out.items() if k != 'summary'}
    s3['ok'] = bool(s_out['raised'] is None and s_out['status'] == 'certified' and s_out['k_star'] == base['k_star']
                    and s_out['n_lapses'] == base['n_lapses'] and s_out['non_optimal_cycles'] == [lo - 1]
                    and s_out['n_vetoes'] == 0 and s_out['in_cycle_rule_equals_pure_replay'] and s_out['summary_ok'])
    s3['k_star_all_optimal'] = base['k_star']
    res['H3_non_optimal_esso_exit_outside_the_window_does_not_block'] = {'holds': s3['ok'], 'detail': s3}
    s_in = _drive_summary(drive(cell, 'certify', nonopt={base['k_star'] - 1: ('ESSO|7',)}))
    s3b = {k: v for k, v in s_in.items() if k != 'summary'}
    s3b['ok'] = bool(s_in['raised'] is None and s_in['n_lapses'] == base['n_lapses']
                     and s_in['non_optimal_cycles'] == [base['k_star'] - 1] and s_in['n_vetoes'] >= 1
                     and s_in['vetoes'][0]['cycle'] == base['k_star'] and s_in['in_cycle_rule_equals_pure_replay']
                     and s_in['summary_ok'] and (s_in['status'] != 'certified' or s_in['k_star'] > base['k_star']))
    s3b['k_star_all_optimal'] = base['k_star']
    res['H3b_non_optimal_esso_exit_inside_the_window_vetoes_no_lapse'] = {'holds': s3b['ok'], 'detail': s3b}
    d4 = drive('c_156ce2d1', 'creep')
    s = _drive_summary(d4)
    rec = {'status': 'certified', 'barrier': False, 'barrier_cause': None, 'certified_cost': 1.0,
           'certification_cycle': s['last_cycle'], 'terminal_gross_operational_cost': 1.0,
           'settling_resettle_summary': s['summary']}
    relabelled = H._apply_settling_resettle_status(dict(rec))
    summ4 = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'cap'
                   and s['last_cycle'] == V4.spec_cap('c_156ce2d1') and s['certificate_length_after'] == 10
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['ended_by'] == 'cap'
                   and relabelled['status'] == 'not_certified' and relabelled['certified_cost'] is None
                   and relabelled['status_production_trajectory']['status'] == 'certified'
                   and summ4['settling_status'] == 'uncertified')
    s['record_label_after_writer_fix'] = {k: relabelled[k] for k in ('status', 'barrier', 'barrier_cause',
                                                                     'certified_cost')}
    res['H4_gated_creep_uncertified_at_the_cap_and_label'] = {'holds': s['ok'], 'detail': s}
    s = _drive_summary(drive('g_37b5c499', 'creep', first_pass_at=50))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'rule_cap'
                   and s['last_cycle'] == 50 + V4.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 50
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['n_overlap'] == 0
                   and s['holds_through_first_pass'] == [HOLDS_OFF] and s['holds_after_first_pass'] == [HOLDS_ON])
    res['H5_ungated_dynamic_rule_cap'] = {'holds': s['ok'], 'detail': s}
    s = _drive_summary(drive('g_9abf31d4', 'certify', first_pass_at=100))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['first_pass'] == 100 and s['summary_ok']
                   and s['in_cycle_rule_equals_pure_replay'] and s['stopped_by'] == 'settling_rule')
    res['H5b_ungated_certifies'] = {'holds': s['ok'], 'detail': s}
    s = _drive_summary(drive('l_2ab0ce2d', 'creep'))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['last_cycle'] == 437
                   and s['rule_cap'] == 437 and s['stopped_by'] == 'cap' and s['summary_ok']
                   and s['replay_bitwise_through'] == 328 and s['in_cycle_rule_equals_pure_replay'])
    res['H6_cap_437_above_300'] = {'holds': s['ok'], 'detail': s}
    res['H7_aa_hold_real_production'] = _h_real_v4(srp, 'aa')
    res['H8_tail_hold_real_production'] = _h_real_v4(srp, 'tail')
    res['H9_rho_hold_real_production'] = _h_real_v4(srp, 'rho')
    res['H10_layering_real_install_and_dispatch'] = _h_layering(srp)
    got = V._certificate_length_writes_in_source()
    in_cycle_v4_src = ''.join(inspect.getsource(o) for o in (V4.HookedRuleV4, V4.ResettleStateV4, V4._v4_rule,
                                                              V4.settling_resettle_hooks, SC4))
    res['H11_certificate_length_writes'] = {
        'holds': (got == sorted(R.CERTIFICATE_LENGTH_WRITES) and 'early_stop' not in in_cycle_v4_src
                  and 'minimum_consecutive_converged_cycles' not in in_cycle_v4_src),
        'writes_in_source': got,
        'v4_in_cycle_code_scanned': ['HookedRuleV4', 'ResettleStateV4', '_v4_rule', 'settling_resettle_hooks',
                                     'settling_criterion_v4 (module)']}
    try:
        res['H12_w132_exit_wrapper_on_real_production_reused'] = K132._h_exit_real()
    except Exception as error:  # noqa: BLE001
        res['H12_w132_exit_wrapper_on_real_production_reused'] = {
            'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_at(commit, sha256, modname):
    import importlib.util
    src = subprocess.run(['git', 'show', f'{commit}:p515_s44_campaign_harness.py'], cwd=REPO, capture_output=True,
                         check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != sha256:
        raise RuntimeError(f'harness at {commit}: sha256 {sha} != pinned {sha256}')
    tmp = tempfile.mkdtemp(prefix='w137_pre_harness_')
    path = os.path.join(tmp, f'{modname}.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(tmp)
    return mod, sha, src.decode()


def _harness_pre_w137():
    return _harness_at(PRE_W137_HARNESS['commit'], PRE_W137_HARNESS['sha256'], '_w137_pre_harness')


def resettle_keys(pre=None):
    """The 38 v4 keys: formula over the (pre-W137) base key; the v3 key of the same cell beside."""
    out = {}
    v3 = K132.resettle_keys(pre)
    for cell in V4.CELL_ORDER:
        _spec, e, kw = K132.resettle_kwargs(cell)
        base = H.evaluation_key(e['key'], e['overrides'], **kw)
        decl = V4.declaration_for(cell)
        key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=decl, **kw)
        base_pre = pre.evaluation_key(e['key'], e['overrides'], **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[cell] = {'candidate_key': e['key'], 'original_eval_key': e['eval_key'], 'base_key_now': base,
                     'resettle_key': key, 'v3_resettle_key': v3[cell]['resettle_key'], 'formula_holds': key == formula,
                     'base_equals_pre_w137': base == base_pre,
                     'differs_from_original': key != e['eval_key'] and base != e['eval_key'],
                     'differs_from_v3': key != v3[cell]['resettle_key']}
    return out


def _key_holders(scanned, key):
    return sorted({rel for rel, spec in scanned for e in spec.get('candidates') or [] if H._entry_eval_key(e) == key})


def _outside_root(rels, root):
    return [r for r in rels if not r.startswith(root + os.sep)]


def w132_frozen_keys():
    """{cell: frozen eval_key} of the 38 committed W132 v3 campaign specs (from the W132 stage spec's pins)."""
    ss = json.load(open(_abs(W132_STAGE_SPEC['path'])))
    out = {}
    for cell, pin in ss['pins']['campaign_specs'].items():
        spec = json.load(open(_abs(pin['path'])))
        out[cell] = spec['candidates'][0]['eval_key']
    return out


def tests_K():
    pre, pre_sha, _src = _harness_pre_w137()
    n_specs = n_entries = n_equal = n_frozen_equal = 0
    own = {'v4': {'n': 0, 'ok': 0, 'bad': []}, 'w135': {'n': 0, 'ok': 0, 'bad': []}}
    n_w118 = n_w132 = 0
    w132_equal = {}
    mismatch, errors = [], []
    scanned = []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        scanned.append((rel, spec))
        for e in spec.get('candidates') or []:
            n_entries += 1
            try:
                args, kw = K132._key_args(spec, e)
                rs = e.get('settling_resettle')
                fam = 'v4' if V4.is_v4_declaration(rs) else ('w135' if W135.is_w135_declaration(rs) else None)
                if fam is not None:
                    # the OWN entries of a campaign this harness adds a route for (the rule: a pre-run scan excludes
                    # the run's own, per root): accepted only inside that campaign's stage root and only if the frozen
                    # key follows the declared formula over the PRE-W137 base key (the pre-W137 harness cannot key them)
                    mod = V4 if fam == 'v4' else W135
                    own[fam]['n'] += 1
                    decl = mod.validate_settling_resettle(rs)
                    kw_mv = dict(kw)
                    formula = hashlib.sha256(json.dumps({'base_evaluation_key': pre.evaluation_key(*args, **kw_mv),
                                                         'settling_resettle': decl}, sort_keys=True,
                                                        separators=(',', ':')).encode()).hexdigest()
                    now = H.evaluation_key(*args, settling_resettle=decl, **kw)
                    inside = rel.startswith(KEY_OWN_ROOTS[fam] + os.sep)
                    if inside and now == formula == H._entry_eval_key(e):
                        own[fam]['ok'] += 1
                    else:
                        own[fam]['bad'].append({'spec': rel, 'label': e.get('label'), 'inside_own_root': inside,
                                                'formula_equals_frozen': formula == H._entry_eval_key(e)})
                    continue
                if V.is_v3_declaration(rs):
                    n_w132 += 1
                elif rs is not None:
                    n_w118 += 1
                new = H.evaluation_key(*args, settling_resettle=rs, **kw)
                old = pre.evaluation_key(*args, settling_resettle=rs, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
                if H._entry_eval_key(e) == new:
                    n_frozen_equal += 1
                if V.is_v3_declaration(rs):
                    w132_equal[rs['cell']] = H._entry_eval_key(e) == new
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    keys = resettle_keys(pre)
    v3_frozen = w132_frozen_keys()
    holders = {cell: _key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: _outside_root(h, W137_ROOT_REL) for cell, h in holders.items()}
    probe = keys[V4.CELL_ORDER[0]]['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w137_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    planted_outside_rel = os.path.join(_P53, 'w137_planted_negative_control', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_sibling_rel = os.path.join(W137_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_w135_rel = os.path.join(W135_ROOT_REL, 'campaign_planted', 'campaign_spec_planted_00000000.json')
    planted_own_rel = os.path.join(W137_ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{V4.CELL_ORDER[0]}',
                                   f'campaign_spec_{CAMPAIGN_ID_PREFIX}{V4.CELL_ORDER[0]}_00000000.json')
    ctl_outside = _outside_root(_key_holders(scanned + [planted(planted_outside_rel)], probe), W137_ROOT_REL)
    ctl_sibling = _outside_root(_key_holders(scanned + [planted(planted_sibling_rel)], probe), W137_ROOT_REL)
    ctl_w135 = _outside_root(_key_holders(scanned + [planted(planted_w135_rel)], probe), W137_ROOT_REL)
    own_holders = _key_holders(scanned + [planted(planted_own_rel)], probe)
    ctl_own = _outside_root(own_holders, W137_ROOT_REL)
    try:
        pre.validate_settling_resettle(V4.declaration_for(CELL1))
        pre_refuses = False
    except ValueError:
        pre_refuses = True
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_other_entry_equal': (n_equal == n_entries - own['v4']['n'] - own['w135']['n']
                                                   and n_entries > 0),
        'w118_resettle_entries_scanned_and_equal': n_w118 >= 10,
        'w132_v3_38_keys_unchanged': (n_w132 == 38 and len(w132_equal) == 38 and all(w132_equal.values())
                                      and all(v3_frozen[c] == keys[c]['v3_resettle_key'] for c in V4.CELL_ORDER)),
        'own_v4_entries_inside_the_v4_root_follow_the_formula': own['v4']['ok'] == own['v4']['n'] and not own['v4']['bad'],
        'own_w135_entries_inside_the_w135_root_follow_the_formula': (own['w135']['ok'] == own['w135']['n']
                                                                     and not own['w135']['bad']),
        'v4_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'v4_base_equals_pre_w137_every_cell': all(v['base_equals_pre_w137'] for v in keys.values()),
        'v4_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'v4_keys_differ_from_the_v3_keys': all(v['differs_from_v3'] for v in keys.values()),
        'v4_keys_distinct': len({v['resettle_key'] for v in keys.values()}) == len(keys) == 38,
        'v4_keys_absent_from_committed_specs_outside_the_v4_root': not any(outside.values()),
        'control_planted_outside_root_refused': planted_outside_rel in ctl_outside,
        'control_planted_sibling_prefix_refused': planted_sibling_rel in ctl_sibling,
        'control_planted_in_the_w135_root_refused': planted_w135_rel in ctl_w135,
        'control_own_campaign_spec_accepted': planted_own_rel in own_holders and planted_own_rel not in ctl_own,
        'control_v4_declaration_refused_by_the_pre_w137_harness': pre_refuses,
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w137_harness': {**PRE_W137_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'entries_whose_frozen_eval_key_equals_the_recomputed_REPORTED': n_frozen_equal,
            'w118_resettle_entries': n_w118, 'w132_v3_resettle_entries': n_w132,
            'own_entries': {k: {'n': v['n'], 'ok': v['ok'], 'bad': v['bad'][:20]} for k, v in own.items()},
            'own_roots': KEY_OWN_ROOTS, 'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'v3_to_v4_key_diff': {c: {'v3': keys[c]['v3_resettle_key'], 'v4': keys[c]['resettle_key']}
                                  for c in V4.CELL_ORDER},
            'v4_keys_in_committed_specs_all_REPORTED': holders,
            'controls': {'planted_outside': {'rel': planted_outside_rel, 'outside_found': ctl_outside},
                         'planted_sibling': {'rel': planted_sibling_rel, 'outside_found': ctl_sibling},
                         'planted_w135_root': {'rel': planted_w135_rel, 'outside_found': ctl_w135},
                         'planted_own': {'rel': planted_own_rel, 'holders': own_holders, 'outside_found': ctl_own}}}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    out = {}
    ok = True
    for cell in V4.CELL_ORDER:
        decl = V4.declaration_for(cell)
        cap = V4.spec_cap(cell)
        try:
            good = V4.assert_resettle_preconditions(decl, {'cap': cap}, {'tail_enabled_for_this_run': True}, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        negatives = {}
        for name, (spec, tail, aa) in {'cap_wrong': ({'cap': cap - 1}, {'tail_enabled_for_this_run': True}, True),
                                       'tail_off': ({'cap': cap}, {'tail_enabled_for_this_run': False}, True),
                                       'aa_off': ({'cap': cap}, {'tail_enabled_for_this_run': True}, False)}.items():
            try:
                V4.assert_resettle_preconditions(decl, spec, tail, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {str(error)[:200]}'
        rule = decl['settling_rule']
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'v3_schema_with_v4_rule': dict(decl, schema=V.DECLARATION_SCHEMA),
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'gap_bound_tau': dict(decl, settling_rule=dict(rule, gap_bound=SC4.TAU)),
               'rule_v3': dict(decl, settling_rule=V.settling_rule_declaration()),
               'reset_on_non_optimal': dict(decl, settling_rule=dict(rule, reset_on_non_optimal=True)),
               'retry_tier': dict(decl, settling_rule=dict(rule, retry_tier='tier1')),
               'cap_rule_changed': dict(decl, cap_rule=dict(decl['cap_rule'], ceiling=999)),
               'exit_capture_off': dict(decl, captures=dict(decl['captures'], ipopt_exit_by_block=False))}
        if decl['replay_reference'] is not None:
            bad.update({'wrong_k0': dict(decl, first_residual_pass_expected=decl['first_residual_pass_expected'] - 1),
                        'wrong_sha': dict(decl, replay_reference=dict(decl['replay_reference'], sha256='0' * 64)),
                        'abort_false': dict(decl, abort_on_replay_divergence=False),
                        'no_reference': dict(decl, replay_reference=None),
                        'lapses_changed': dict(decl, original_lapses_after_k0=[999])})
        else:
            bad.update({'invented_reference': dict(decl, replay_reference={'per_cycle_record': 'x', 'sha256': '0' * 64}),
                        'abort_true': dict(decl, abort_on_replay_divergence=True)})
        refused = {}
        for name, d in bad.items():
            try:
                V4.validate_settling_resettle(d)
                refused[name] = False
            except ValueError as error:
                refused[name] = str(error)[:160]
        cell_ok = good_ok and all(v.startswith('refused') for v in negatives.values()) and all(refused.values())
        ok = ok and cell_ok
        out[cell] = {'ok': cell_ok, 'n_checklist_items': len(good) if isinstance(good, dict) else None,
                     'checklist_failing': (sorted(k for k, v in good.items() if v is not True) if good_ok is False
                                           and 'error' not in good else good.get('error')),
                     'negative_controls': negatives, 'validator_refuses': refused}
    return {'holds': ok, 'cells': out}


# ======================================================================================================================
#  X -- W132's status-label writer checks (reused) and G8 on production's certificate
# ======================================================================================================================
def tests_X():
    base = K132.tests_X()
    ev = CELL1_V3['eval_dir']
    rel = os.path.join(ev, 'evaluation_record.json')
    rec = json.load(open(_abs(rel)))
    g8_ok, g8 = V4.persistence_check_production_certificate(rec, _abs(ev))
    old_formula_would_fail = not ((rec.get('status') == 'certified') == os.path.exists(
        _abs(os.path.join(ev, 'certified_models.pkl'))))
    tmp = tempfile.mkdtemp(prefix='w137_g8_')
    try:
        with open(os.path.join(tmp, 'certified_models.pkl'), 'wb') as handle:
            handle.write(b'x')
        ctl = {}
        ctl['production_uncertified_but_pkl_written_fails'] = not V4.persistence_check_production_certificate(
            {'status_production_trajectory': {'status': 'not_certified'}, 'post_certification': {'status': 'evaluated'}},
            tmp, verify_pkl_sha256=False)[0]
        ctl['no_production_view_fails'] = not V4.persistence_check_production_certificate(
            {'status': 'certified', 'post_certification': {'status': 'evaluated'}}, tmp, verify_pkl_sha256=False)[0]
        os.remove(os.path.join(tmp, 'certified_models.pkl'))
        ctl['production_certified_without_pkl_fails'] = not V4.persistence_check_production_certificate(
            {'status_production_trajectory': {'status': 'certified'}, 'post_certification': {'status': 'evaluated'}},
            tmp)[0]
        ctl['production_uncertified_skipped_no_pkl_passes'] = V4.persistence_check_production_certificate(
            {'status_production_trajectory': {'status': 'not_certified'}, 'post_certification': {'status': 'skipped'}},
            tmp)[0]
    finally:
        shutil.rmtree(tmp)
    parts = {'w132_status_label_checks_hold': base['holds'] is True,
             'g8_cell1_v3_record_passes_on_production_certificate': g8_ok,
             'g8_cell1_pkl_sha256_matches_the_recorded': g8.get('pkl_sha256_matches_recorded_report_only') is True,
             'g8_cell1_settling_label_not_certified_production_certified': (
                 rec.get('status') == 'not_certified' and g8['production_trajectory_status'] == 'certified'),
             'the_w101_formula_status_iff_pkl_fails_on_it_as_w133_recorded': old_formula_would_fail,
             'cell1_record_committed_clean': _clean(rel),
             **{f'g8_control_{k}': v for k, v in ctl.items()}}
    return {'holds': all(parts.values()), 'parts': parts, 'w132_status_label': base,
            'g8_cell1_rescore': {'record': rel, 'record_sha256': _sha(rel), 'ok': g8_ok, 'detail': g8}}


# ======================================================================================================================
#  D -- the router: pre-W137 harness + W135's prepared patch + the v4 branch, nothing else
# ======================================================================================================================
def harness_router_check():
    _pre, _sha_pre, pre_src = _harness_pre_w137()
    now_src = open(_abs('p515_s44_campaign_harness.py')).read()
    import difflib
    diff = [ln for ln in difflib.unified_diff(pre_src.splitlines(), now_src.splitlines(), lineterm='', n=0)
            if not ln.startswith(('---', '+++', '@@'))]
    added = [ln[1:] for ln in diff if ln.startswith('+')]
    removed = [ln[1:] for ln in diff if ln.startswith('-')]
    patch = open(_abs(W135_PATCH_REL)).read().splitlines()
    patch_added = [ln[1:] for ln in patch if ln.startswith('+') and not ln.startswith('+++')]
    fn_src = inspect.getsource(H.resettle_hooks_module)
    i132 = fn_src.find('W132C.is_v3_declaration(value)')
    i135 = fn_src.find('W135C.is_w135_declaration(value)')
    i137 = fn_src.find('W137C.is_v4_declaration(value)')
    i118 = fn_src.find('import p515_s53_w118_resettle_hooks as W118C')
    sha_now = _sha('p515_s44_campaign_harness.py')
    patch_log = H._git(['log', '--format=%H', '-1', '--', 'p515_s44_campaign_harness.py'])
    parts = {
        'no_line_removed': removed == [],
        'exactly_six_lines_added': len(added) == 6,
        'first_three_added_are_the_w135_patch': added[:3] == patch_added and len(patch_added) == 3,
        'last_three_added_are_the_v4_branch': tuple(added[3:]) == V4_BRANCH_LINES,
        'order_w132_then_w135_then_v4_then_w118': 0 <= i132 < i135 < i137 < i118,
        'harness_committed_clean': _clean('p515_s44_campaign_harness.py'),
        'harness_sha256_is_the_pinned_post_v4_sha256': sha_now == HARNESS_POST_V4_SHA256,
        'dispatch_four_families': all(dispatch_checks().values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'added': added, 'removed': removed,
            'harness_sha256_now': sha_now, 'pre_w137_harness_sha256': PRE_W137_HARNESS['sha256'],
            'post_w135_patch_sha256': HARNESS_POST_W135_PATCH_SHA256,
            'post_v4_sha256_pinned': HARNESS_POST_V4_SHA256, 'last_harness_commit': patch_log}


def tests_D():
    return harness_router_check()


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('R', tests_R), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('K', tests_K),
            ('P', tests_P), ('X', tests_X), ('D', tests_D))

CODE_PINNED_BY_CHECKS = (os.path.basename(__file__), 'settling_criterion_v4.py', 'settling_criterion_v3.py',
                         'settling_criterion_v2.py', 'settling_criterion.py', 'p515_s53_w137_resettle_v4_hooks.py',
                         'p515_s53_w132_resettle_v3_hooks.py', 'p515_s53_w132_resettle_v3_checks.py',
                         'p515_s53_w135_resettle_ext_hooks.py', W135_PATCH_REL,
                         'p515_s53_w118_resettle_hooks.py', 'p515_s53_w118_resettle_checks.py',
                         'p515_s44_campaign_harness.py', 'gate_result_io.py', 'interface_dual_capture.py',
                         'shared_resources_planning.py', 'admm_anderson_acceleration.py',
                         'shared_energy_storage_data.py', 'network.py', 'helper_functions.py',
                         'p515_s53_w105_settling_extension_hooks.py', 'p515_s53_w101_settling_continuation_hooks.py',
                         'p515_s53_w105_extension_checks.py', 'p515_s53_w101_continuation_checks.py',
                         'p515_s53_w98_continuation_checks.py', 'p515_s53_w112_consensus_gap.py',
                         'p515_g_g1_g4_admm_gates.py', 'p515_gate_result_bool_typing_test.py')


def run_all_checks(sections=SECTIONS):
    out = {}
    ok = True
    for sid, fn in sections:
        t0 = time.time()
        try:
            r = fn()
        except Exception as error:  # noqa: BLE001 -- recorded as a failing section
            r = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        r_ok = r.get('holds') is True
        out[sid] = {'holds': r_ok, 'wall_s': time.time() - t0, 'result': r}
        ok = ok and r_ok
    return {'all_hold': ok, 'p_max': V4.P_MAX, 'constants': SC4.constants(V4.P_MAX), 'readings': SC4.READINGS,
            'sub_test_reads': SC4.SUB_TEST_READS, 'sections': out}


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
    doc = {'schema': 'p515_s53_w137_zero_solve_checks_v1',
           'task': 'W137 (PLANNER_BRIEF_2026-09-13.md Addendum 59 and its Supplement)',
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
        print(f"[W137-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W137-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W137-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) guards={guards}")
    print(f"[W137-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
