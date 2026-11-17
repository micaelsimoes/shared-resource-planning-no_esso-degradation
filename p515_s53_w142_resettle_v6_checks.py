"""
P5.15 Addendum 61, Planner task W142 -- ZERO-SOLVE checks for the v6 re-settling campaign (the 34 cells not yet decided;
frozen_s53_resettle_spec_v6, predecessor v5 ab32ffc9). W139's section set, re-targeted to v6, plus the v6 unit tests for
the growth-test floor (A) and the turning-point floor (B) and the determinacy floor; the W139 checks module is imported
for every rule-independent section and helper and NOT edited.

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W139 checks
module, imported, arms its own and the modules it imports arm theirs; every guard is verified at 0). No model is built
and nothing is solved: H13 and C read a fresh SRP1 planning object (data only); everything else reads committed, manifest-
or pin-verified records or synthetic files in a temporary directory.

WHAT IS CHECKED
  V   the stop rule VERSION 6 (`settling_criterion_v6`): V_v5 the W139 version-5 suite (unchanged, run as committed);
      V_eq_v5 version 6 with both floors at 0 (a test subclass) reproduces version 5 record by record (the v6-only record
      keys stripped) and decision by decision on every W132 trajectory with planted non-clean cycles; V_v6_vs_v5 the
      real v6 on the same trajectories, every difference REPORTED; THE UNIT TESTS: V6a the growth test (A) -- a swing
      below F excluded, an earlier growth trend compared ACROSS an excluded swing, a swing equal to F kept, F = 0 the
      all-pairs test; V6b the turning-point floor (B) -- a sub-F reversal rejected, the popped point carried, j_change
      reverted, a swing equal to F registers, F = 0 version 1's sign state on every trajectory; V6c through the rule: the
      d_c52e1670 record (51280961) -- v5 uncertified, v6 certifies at 150, (A) alone (TP floor 0) at 139 (the W142
      replay's arms), a growing oscillation with swings above F still refused (swings_growing); V6d the determinacy floor
      (the 2 TAU term, the 3 x larger-bar term, the >= boundary); V6e the floor constants (TAU / 10) and the declaration;
      V21 the per-cell ceiling; V22 all_clean must be a bool; V0 versions 1-5 byte-identical to their pins.
  R   v6 FROM RECORDS (the committed W142 outputs, manifest-verified): the floor replay's decision ('A_and_B') is what
      settling_criterion_v6 carries; v6 recomputed on every record equals the replay's (A)+(B) arm and the committed
      table; d_c52e1670's certificate record recomputed from its v5 run's committed inputs (k*, window, band, t_sum, the
      out-of-window reads); the determinacy re-scoring recomputed (every row, no verdict change as committed); the
      movement record recomputed.
  O   the original cells (W132's section O, reused unchanged: the 34 v6 cells are among them).
  S   production_since_originals: no uncommitted change to a file this run uses (gate); report as W139.
  H   the hooks through the REAL nine wrappers (W118's eight + W139's v5 exit wrapper, unchanged) on the v6 state (W139's
      `drive`, with this module's declarations and state): H1 every gated cell bitwise through k0 then certifies; H2 a
      one-ulp divergence aborts; H3 the clean rule inside the window unchanged (2.51x primary does not veto, 11x and a
      recovery veto); H4 a gated creep uncertified at the cap and the record label; H5 / H5b ungated dynamic cap /
      certification; H6 2ab0ce2d to its cap 437; H7-H9 the AA / tail / rho holds with REAL production on the v6 state;
      H10 the real install and the harness dispatch (v6 -> this, v6 extension -> its module, v5 -> W139, v5 extension,
      v4, v3, W135, W118); H11 certificate-length writes; H12 / H13 the exit wrappers on real production (W139's,
      reused: the exit wrapper is unchanged in v6).
  C   the clean capture checklist (W139's section C, reused: unchanged in v6).
  C2  the capture replayed on the committed v4 cell-3 run (W139's section C2, reused).
  K   keys: every entry of every committed campaign spec keys identically under this harness and the pre-W142 harness
      (b37ea166, 961dbb3e, sha256 pinned, from git) -- the v5 and v5-extension entries included -- except the OWN entries
      of the two families the v6 route adds (v6 entries only inside the v6 stage root, v6-extension entries only inside
      the v6-extension root, each only if its frozen key follows the formula over the pre-W142 base key); the 34 v6 keys
      follow the formula, are distinct, differ from the originals and from the v3, v4 and v5 keys of the same cell, and
      appear in no committed spec outside the v6 root (a pre-run scan excludes the run's own root); planted controls;
      the pre-W142 harness refuses a v6 declaration.
  P   `assert_resettle_preconditions` holds for every cell with its cap; negative controls refused; the validator
      refuses early_stop and malformed declarations (a v5 rule declaration and a changed floor included).
  X   W137's section X reused (W132's status-label writer checks; G8 on production's certificate).
  D   the harness is EXACTLY the pre-W142 harness (961dbb3e) plus the v6 branch (3 added lines in
      `resettle_hooks_module`, after the v5 branch and before the W118 fallback, none removed; every existing branch
      identical); its sha256 is the pinned post-v6 sha.
  W   (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, alone, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w142_resettle_v6_checks.py \\
      > data/SRP1/Results/P515S53/w142_resettle_v6/zero_solve_checks_launch.log 2>&1
"""
import contextlib
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W142 v6 re-settling zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402
import p515_s53_w139_resettle_v5_hooks as V5  # noqa: E402
import p515_s53_w137_resettle_v4_hooks as V4  # noqa: E402
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w135_resettle_ext_hooks as W135  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w139_resettle_v5_checks as K139  # noqa: E402 -- the v5 suite and helpers (arms its own guard)
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v5 as SC5  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import pickle  # noqa: E402
import p515_s53_w141_swing_variants as W141  # noqa: E402 -- record inputs (arms its guards; installs a pickle block)

# W141's module blocks pickle.load / pickle.loads at import (its own no-model-load claim). These checks make no such
# claim (H13 and C read a fresh SRP1 planning object), so the originals are restored at once; W141's counters stay 0.
pickle.load, pickle.loads = W141._PICKLE_ORIG

K137 = K139.K137
K132 = K139.K132
K118, K105, K98 = K132.K118, K132.K105, K132.K98


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w142_checks', GUARD),) + tuple(K139.GUARDS) + tuple(W141.GUARDS))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W142_ROOT_REL = os.path.join(_P53, 'w142_resettle_v6')
W142_EXT_ROOT_REL = os.path.join(_P53, 'w142_resettle_ext_v6')
W139_ROOT_REL = K139.W139_ROOT_REL
W139_EXT_ROOT_REL = K139.W139_EXT_ROOT_REL
W137_ROOT_REL = K137.W137_ROOT_REL
W135_ROOT_REL = K137.W135_ROOT_REL
OUT_DIR_REL = os.path.join(W142_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w142_zero_solve_checks.json'
OUT_MANIFEST = 'w142_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w142_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = 's53_w142_resettle_v6_'
EXT_CAMPAIGN_ID_PREFIX = 's53_w142_resettle_ext_v6_'
KEY_OWN_ROOTS = {'v6': W142_ROOT_REL, 'ext_v6': W142_EXT_ROOT_REL}
# the harness before W142 (pinned by the v5 stage spec ab32ffc9; last changed by b37ea166, W139's router commit)
PRE_W142_HARNESS = {'commit': 'b37ea1661eb0f85804b063a0bf380f2e90b6e1e0',
                    'sha256': '961dbb3e6a6808bcc3665a081024af12a196e8f4ddf2b80e2f70dbf7edb25638'}
V6_BRANCH_LINES = ('    import p515_s53_w142_resettle_v6_hooks as W142C',
                   '    if W142C.is_v6_declaration(value):',
                   '        return W142C.hooks_module(value)')
HARNESS_POST_V6_SHA256 = 'ca46892ba1fb75913962c508f4a230ce5356ebcbbd1ef4b43031c8242ee98e8d'   # pre-W142 + the v6 branch
V5_STAGE_SPEC = {'path': os.path.join(W139_ROOT_REL, 'frozen_s53_resettle_spec_v5_ab32ffc9.json'), 'sha256': 'ab32ffc9'}
FROM_RECORDS = {'path': os.path.join(W142_ROOT_REL, 'v6_from_records', 'w142_v6_from_records.json'),
                'certificate': os.path.join(W142_ROOT_REL, 'v6_from_records',
                                            'd_c52e1670_v6_from_records_certificate.json'),
                'manifest': os.path.join(W142_ROOT_REL, 'v6_from_records', 'manifest_sha256.json')}
FLOOR_REPLAY = {'path': os.path.join(W142_ROOT_REL, 'floor_replay', 'w142_floor_replay.json'),
                'manifest': os.path.join(W142_ROOT_REL, 'floor_replay', 'manifest_sha256.json')}
D_CELL = 'd_c52e1670'
D_RECORD = 'd_c52e1670@v5'
V6_ONLY_RECORD_KEYS = ('turning_point_floor_rejection',)
V6_ONLY_APART_KEYS = ('swings_non_increasing_floored', 'growth_floor', 'turning_point_floor',
                      'swings_excluded_below_floor', 'swing_pairs_compared', 'swings_non_increasing_all_pairs_note')
V6_ONLY_DECISION_KEYS = ('swing_floor', 'turning_point_floor_rejections')
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


def harness_post_v6_sha256():
    """The pinned post-v6 harness sha256 (computed before the router commit from the pre-W142 harness plus the three
    branch lines; the router commit's message carries it)."""
    return HARNESS_POST_V6_SHA256


class _ZeroFloorV6(SC6.SettlingRuleV6):
    """Checks only: version 6 with both floors at 0 (must reproduce version 5)."""
    GROWTH_FLOOR = 0.0
    TP_FLOOR = 0.0


class _GrowthOnlyV6(SC6.SettlingRuleV6):
    """Checks only: version 6 with the turning-point floor (B) off -- (A) alone."""
    TP_FLOOR = 0.0


def _replay6(q, b, t, ac, kw, rule_class=None):
    kw6 = dict(kw)
    kw6.setdefault('cap_ceiling', SC2.CAP_CEILING)
    return SC6.replay(q, b, t, ac, R.P_MAX, rule_class=rule_class, **kw6)


def _replay5(q, b, t, ac, kw):
    return K139._replay5(q, b, t, ac, kw)


def _strip_v6(x, depth=0):
    """A v6 record / decision with the v6-only keys removed (for the F = 0 equality with version 5)."""
    if isinstance(x, dict):
        drop = set(V6_ONLY_RECORD_KEYS) | set(V6_ONLY_APART_KEYS) | set(V6_ONLY_DECISION_KEYS)
        return {k: _strip_v6(v, depth + 1) for k, v in x.items() if k not in drop}
    if isinstance(x, list):
        return [_strip_v6(v, depth + 1) for v in x]
    return x


_DEC_VERSION_ONLY = ('version', 'reading', 'N_definition', 'window_rule', 'clean_rule')


def _dec_cmp(d):
    """A decision (v5 or v6) for the F = 0 equality: the v6-only keys, the version-stamp keys and the enumeration's
    name (settling_criterion_v5 / _v6.SUB_TEST_READS) removed on both sides."""
    if d is None:
        return None
    out = {k: v for k, v in _strip_v6(d).items() if k not in _DEC_VERSION_ONLY}
    oow = out.get('out_of_window_reads')
    if isinstance(oow, dict):
        oow = dict(oow)
        oow.pop('enumeration', None)
        reads = dict(oow.get('reads') or {})
        reads.pop('turning_point_floor', None)
        reads.pop('swings_excluded_below_growth_floor', None)
        oow['reads'] = reads
        out['out_of_window_reads'] = oow
    return out


def _rec_cmp(r):
    out = _strip_v6(r)
    if isinstance(out.get('out_of_window_reads'), dict):
        out['out_of_window_reads'] = _dec_cmp({'out_of_window_reads': out['out_of_window_reads']})['out_of_window_reads']
    return out


def _summ(d):
    d = d or {}
    return {k: d.get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'window', 'branch')}


# ======================================================================================================================
#  V -- the stop rule v6
# ======================================================================================================================
def unit_tests_A():
    F = SC6.SWING_FLOOR
    g = SC6.growth_test
    res = {}
    ok1, d1 = g([5000.0, 200.0, 6000.0], F)
    ok2, d2 = g([5000.0, 200.0, 3000.0], F)
    ok3, d3 = g([5000.0, F, 6000.0], F)
    ok4, d4 = g([5000.0, F - 1e-9, 4000.0], F)
    ok5, _d5 = g([226.0, 125.0], F)
    res['V6a1_growth_across_an_excluded_swing_is_still_compared'] = {
        'ok': ok1 is False and d1['excluded_swing_indices'] == [1] and d1['pairs_compared'] == [[0, 2]],
        'A': [5000.0, 200.0, 6000.0], 'detail': d1}
    res['V6a2_decreasing_swings_around_a_blip_pass'] = {'ok': ok2 is True and d2['pairs_compared'] == [[0, 2]],
                                                        'detail': d2}
    res['V6a3_a_swing_equal_to_F_is_kept'] = {'ok': ok3 is False and d3['excluded_swing_indices'] == []
                                              and d3['pairs_compared'] == [[0, 1], [1, 2]], 'detail': d3}
    res['V6a4_a_swing_just_below_F_is_excluded'] = {'ok': ok4 is True and d4['excluded_swing_indices'] == [1],
                                                    'detail': d4}
    res['V6a5_only_sub_F_swings_compare_nothing'] = {'ok': ok5 is True}
    allp = {}
    for A in ([3.0, 2.0, 1.0], [1.0, 2.0], [5.0, 5.0, 4.0], [], [7.0]):
        allp[str(A)] = g(A, 0.0)[0] == all(A[i + 1] <= A[i] for i in range(len(A) - 1))
    res['V6a6_floor_0_is_the_all_pairs_test'] = {'ok': all(allp.values()), 'cases': allp}
    return res


def _sign_run(q, floor):
    st = SC6.FloorSignState(floor)
    ref = SC1._SignState()
    trace = []
    for k in range(2, max(q) + 1):
        s = SC1.sign_of_step(q[k] - q[k - 1])
        a = st.update(q, k, s)
        b = ref.update(q, k, s)
        trace.append((k, a, b, st.j_change, ref.j_change))
    return st, ref, trace


def unit_tests_B():
    F = SC6.SWING_FLOOR
    res = {}
    # up 5 steps, a sub-F blip (down 200, up 150), up again, then a large fall: the blip's reversal is rejected
    q = {1: 0.0}
    steps = [1000.0] * 5 + [-200.0, 150.0] + [1000.0] * 3 + [-2000.0] * 4 + [1500.0] * 3
    for i, s in enumerate(steps):
        q[i + 2] = q[i + 1] + s
    st, ref, trace = _sign_run(q, F)
    rej = st.rejections
    res['V6b1_sub_F_reversal_rejected_carried_and_j_change_reverted'] = {
        'ok': bool(len(rej) == 1 and rej[0]['closed_swing'] == 200.0 and [t[0] for t in st.T][:1] == [11]
                   and len(ref.T) > len(st.T) and st.carry is None),
        'T_floor': st.T, 'T_version1': ref.T, 'A_floor': st.A, 'A_version1': ref.A, 'rejections': rej}
    # j_change: after the rejection, j_change equals its value before the popped turning point registered
    k_rej = rej[0]['cycle'] if rej else None
    jc_after = next((t[3] for t in trace if t[0] == k_rej), 'missing')
    res['V6b2_j_change_reverted_at_the_rejection'] = {'ok': jc_after is None, 'j_change_after_rejection': jc_after}
    # a swing exactly F registers (values chosen so that |Q_t - Q_T[-1]| == F exactly in floating point)
    q2 = {1: -2000.0, 2: -1000.0, 3: F, 4: 0.0, 5: 1000.0}
    st2, _r2, _t2 = _sign_run(q2, F)
    res['V6b3_a_swing_equal_to_F_registers'] = {
        'ok': [t[0] for t in st2.T] == [3, 4] and st2.A == [F] and not st2.rejections, 'T': st2.T, 'A': st2.A}
    # floor 0 == version 1 on every W132 trajectory
    trajs, _q6, _b6, _k6 = K132._trajectories()
    eq0 = {}
    for name, qq, _b, _t, _kw in trajs:
        qq = {k: v for k, v in qq.items() if v is not None}
        if len(qq) < 3 or sorted(qq) != list(range(min(qq), max(qq) + 1)):
            continue
        q1 = {k - min(qq) + 1: v for k, v in qq.items()}
        s0, r0, _ = _sign_run(q1, 0.0)
        eq0[name] = s0.T == r0.T and s0.A == r0.A and s0.j_change == r0.j_change
    res['V6b4_floor_0_is_version_1_sign_state'] = {'ok': bool(eq0) and all(eq0.values()), 'per_trajectory': eq0}
    return res


def determinacy_unit_tests():
    T2 = 2.0 * SC6.TAU
    f = SC6.determinate_certified
    cases = {
        'two_tau_binds_small_bars': f(T2, 1000.0, 2000.0) == (True, T2, '2 TAU'),
        'just_below_two_tau': f(T2 - 1e-6, 1000.0, 2000.0)[0] is False,
        'three_x_larger_bar_binds': f(3 * 4404.7, 4209.2, 4404.7)[0:1] == (True,)
        and f(3 * 4404.7, 4209.2, 4404.7)[2] == '3 x larger bar',
        'three_x_boundary_is_ge': f(3.0 * 5000.0, 5000.0, 100.0)[0] is True,
        'sign_irrelevant': f(-60000.0, 4000.0, 4500.0)[0] is True,
        'within_resolution': f(12000.0, 4200.0, 4400.0)[0] is False,
        'threshold_formula': SC6.determinacy_threshold(3000.0, 4000.0) == max(12000.0, T2),
    }
    return {'V6d_determinacy_floor': {'ok': all(cases.values()), 'cases': cases, 'two_tau': T2}}


def _d_record():
    """The d_c52e1670 v5 run inputs (W141's reader: every input sha256-verified against the campaign manifest)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return W141.v5_run_inputs()[D_RECORD]


def tests_V():
    res = {}
    v5 = K139.tests_V()
    res['V_v5_suite_as_committed'] = {'ok': v5['holds'] is True,
                                      'tests': {k: v.get('ok') for k, v in v5['tests'].items()}}
    trajs, q6, b6, k6 = K132._trajectories()
    eq, diff6 = {}, {}
    for name, q, b, t, kw in trajs:
        _o, d0, _r = _replay5(q, b, t, {k: True for k in q}, kw)
        ks = sorted(q)
        plants = {'none': set(), 'every_7th': {k for k in ks if k % 7 == 0}}
        if (d0 or {}).get('status') == 'certified':
            plants['at_k_star_minus_1'] = {d0['k_star'] - 1}
            plants['just_before_window'] = {d0['window'][0] - 1}
        for pname, non in plants.items():
            flags = {k: k not in non for k in q}
            o5, d5, r5 = _replay5(q, b, t, flags, kw)
            oz, dz, rz = _replay6(q, b, t, flags, kw, rule_class=_ZeroFloorV6)
            recs_equal = [_jt(_rec_cmp(x)) for x in o5] == [_jt(_rec_cmp(x)) for x in oz]
            dec_equal = _jt(_dec_cmp(d5)) == _jt(_dec_cmp(dz))
            eq[f'{name}|{pname}'] = {'ok': bool(recs_equal and dec_equal and (dz or {}).get('version') == 6
                                                and len(r5.vetoes) == len(rz.vetoes)),
                                     'records_equal': recs_equal, 'decision_equal': dec_equal, 'v6_f0': _summ(dz)}
            o6, d6, r6 = _replay6(q, b, t, flags, kw)
            if _summ(d6) != _summ(d5):
                diff6[f'{name}|{pname}'] = {'v5': _summ(d5), 'v6': _summ(d6),
                                            'v6_growth_only_A': _summ(_replay6(q, b, t, flags, kw,
                                                                               rule_class=_GrowthOnlyV6)[1]),
                                            'rejections': r6._all_rejections()[:5]}
    res['V_eq_v5_at_floor_0'] = {'ok': all(v['ok'] for v in eq.values()), 'n': len(eq),
                                 'failing': {k: v for k, v in eq.items() if not v['ok']}}
    res['V_v6_vs_v5_on_the_trajectories_REPORTED'] = {'ok': True, 'n_differ': len(diff6), 'differ': diff6,
                                                      'note': 'report-only: where the floors change a synthetic decision'}
    res.update(unit_tests_A())
    res.update(unit_tests_B())
    res.update(determinacy_unit_tests())
    # V6c: through the rule on the d_c52e1670 record
    rec = _d_record()
    q, b, t, n, kw = rec['q'], rec['b'], rec['t'], rec['n'], rec['kw']
    ac = rec['in_run_all_clean']
    _o5, d5, _r5 = SC5.replay(q, b, t, ac, R.P_MAX, last=n, **kw)
    _o6, d6, r6 = SC6.replay(q, b, t, ac, R.P_MAX, last=n, **kw)
    _oa, da, _ra = SC6.replay(q, b, t, ac, R.P_MAX, last=n, rule_class=_GrowthOnlyV6, **kw)
    res['V6c_d_c52e1670_through_the_rule'] = {
        'ok': bool((d5 or {}).get('status') == 'uncertified' and 'swings_growing' in (d5 or {}).get('reasons', [])
                   and (d6 or {}).get('status') == 'certified' and d6['k_star'] == 150 and d6['window'] == [120, 150]
                   and (da or {}).get('status') == 'certified' and da['k_star'] == 139
                   and len(r6._all_rejections()) == 1 and r6._all_rejections()[0]['cycle'] == 113),
        'v5': _summ(d5), 'v6': _summ(d6), 'v6_growth_only_A': _summ(da), 'rejections': r6._all_rejections()}
    # V6c2: a growing oscillation with every swing above F is still refused
    qg = {k: 1e9 + (1500.0 + 60.0 * k) * math.cos(2 * math.pi * k / 16.0) for k in range(1, 201)}
    _og, dg, _rg = _replay6(qg, {k: True for k in qg}, {k: 0.0 for k in qg}, {k: True for k in qg}, {'cap': 200})
    res['V6c2_growing_oscillation_above_F_refused'] = {
        'ok': (dg or {}).get('status') == 'uncertified' and 'swings_growing' in (dg or {}).get('reasons', []),
        'v6': _summ(dg), 'reasons': (dg or {}).get('reasons')}
    # V6e: the constants and the declaration
    decl = V6.settling_rule_declaration()
    res['V6e_floor_constants_and_declaration'] = {
        'ok': bool(SC6.SWING_FLOOR == SC6.TAU / 10.0 == SC6.GROWTH_TEST_FLOOR == SC6.TURNING_POINT_FLOOR
                   and SC6.SettlingRuleV6.GROWTH_FLOOR == SC6.SettlingRuleV6.TP_FLOOR == SC6.SWING_FLOOR
                   and list(SC6.CARRIES) == ['A_growth_test_floor', 'B_turning_point_floor']
                   and decl['version'] == 6 and decl['swing_floor']['F'] == SC6.SWING_FLOOR
                   and decl['module'] == 'settling_criterion_v6' and SC6.TAU == SC5.TAU and SC6.EPS0 == SC5.EPS0
                   and SC6.GAP_BOUND == SC5.GAP_BOUND and SC6.CLEAN_FACTOR == SC5.CLEAN_FACTOR),
        'F': SC6.SWING_FLOOR, 'declaration_swing_floor': decl['swing_floor']}
    # V21 / V22
    try:
        ok437 = SC6.SettlingRuleV6(R.P_MAX, cap=437, cap_ceiling=437).cap == 437
    except Exception:  # noqa: BLE001
        ok437 = False
    refused = {}
    for name, kw_ in (('cap_437_ceiling_300', {'cap': 437, 'cap_ceiling': 300}), ('no_ceiling', {'cap': 200}),
                      ('no_ceiling_dynamic', {'cap_after_first_k0': 109})):
        try:
            SC6.SettlingRuleV6(R.P_MAX, **kw_)
            refused[name] = False
        except ValueError:
            refused[name] = True
    res['V21_per_cell_ceiling'] = {'ok': bool(ok437 and all(refused.values())), 'refused': refused}
    bad = {}
    for v in (None, 1, 'True'):
        try:
            SC6.SettlingRuleV6(R.P_MAX, cap=10, cap_ceiling=300).observe(1, 1.0, True, 0.0, v)
            bad[repr(v)] = False
        except TypeError:
            bad[repr(v)] = True
    res['V22_all_clean_is_a_bool'] = {'ok': all(bad.values()), 'refused': bad}
    ss5 = json.load(open(_abs(V5_STAGE_SPEC['path'])))
    pins5 = ss5['pins']['code_sha256']
    names = ('settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v3.py', 'settling_criterion_v4.py',
             'settling_criterion_v5.py')
    now = {nm: _sha(nm) for nm in names}
    res['V0_versions_1_to_5_byte_identical'] = {
        'ok': bool(all(now[nm] == pins5.get(nm) for nm in names) and _sha(V5_STAGE_SPEC['path']).startswith(
            V5_STAGE_SPEC['sha256']) and all(_clean(nm) for nm in names)),
        'now': now, 'pins_ab32ffc9': {nm: pins5.get(nm) for nm in names}}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  R -- v6 from records (the committed W142 outputs)
# ======================================================================================================================
def _pinned_from_records():
    man = json.load(open(_abs(FROM_RECORDS['manifest'])))
    out = {}
    for key in ('path', 'certificate'):
        rel = FROM_RECORDS[key]
        sha = _sha(rel)
        if man.get(rel) != sha or not _clean(rel):
            raise RuntimeError(f'{rel}: sha256 {sha} != manifest {man.get(rel)} or not committed clean')
        out[key] = (json.load(open(_abs(rel))), sha)
    rman = json.load(open(_abs(FLOOR_REPLAY['manifest'])))
    rsha = _sha(FLOOR_REPLAY['path'])
    if rman.get(FLOOR_REPLAY['path']) != rsha or not _clean(FLOOR_REPLAY['path']):
        raise RuntimeError('the floor replay output is not as its manifest / not committed clean')
    out['replay'] = (json.load(open(_abs(FLOOR_REPLAY['path']))), rsha)
    return out


def tests_R():
    import p515_s53_w142_v6_from_records as F
    pins = _pinned_from_records()
    doc, doc_sha = pins['path']
    cert, cert_sha = pins['certificate']
    replay, replay_sha = pins['replay']
    with contextlib.redirect_stdout(io.StringIO()):
        recs = W141.W139.record_inputs()
    recs.update(W141.v5_run_inputs())
    v6_all, mismatch = F.v6_on_records(replay, recs)
    table_equal = _jt({r: v['v6'] for r, v in v6_all.items()}) == _jt({r: v['v6'] for r, v in
                                                                         doc['item1_v6_on_every_record'].items()})
    cert2, rep2 = F.d_cell_certificate(recs, replay)
    cert_fields = ('k_star', 'window', 'W', 'P_hat', 'band', 'band_width', 'range_over_tau', 't_sum_k_star', 'Q_k_star',
                   'T', 'A', 'turning_point_floor_rejections', 'out_of_window_reads', 'all_clean_crosscheck')
    cert_equal = {f: _jt(cert.get(f)) == _jt(cert2.get(f)) for f in cert_fields}
    with contextlib.redirect_stdout(io.StringIO()):
        w138, _ = F._pinned(F.W138_SUMMARY)
        w118, _ = F._pinned(F.W118_SUMMARY)
        w130, _ = F._pinned(F.W130)
    c5 = json.load(open(_abs(os.path.join(F.CELL3_V5['campaign_root'], 'campaign_results.json'))))
    rows = (F.rescore_w138(w138) + [F.rescore_b_n9_e3_current(w138, c5['cell_report']['view'])] + F.rescore_w118(w118)
            + F.rescore_w130(w130))
    rescore_equal = _jt(rows) == _jt(doc['item3_determinacy_rescore']['rows'])
    mov = F.movement_record(recs, cert2)
    parts = {
        'replay_decision_is_what_v6_carries': (replay['decision']['v6_carries'] == 'A_and_B'
                                               and list(SC6.CARRIES) == ['A_growth_test_floor', 'B_turning_point_floor']
                                               and replay['decision']['AB_changes_a_committed_certificate'] == []),
        'replay_self_tests_and_clean_crosscheck': (replay['self_tests']['A_f0_equals_V0_and_AB_equals_w141_V3_10_every_record']
                                                   is True and replay['all_clean_crosscheck_all_equal'] is True),
        'v6_equals_the_replay_AB_arm_every_record': not mismatch and len(v6_all) == 18,
        'v6_table_equals_the_committed_table': table_equal,
        'only_d_c52e1670_changes_vs_v5': [r for r, v in v6_all.items() if v['v6_changes_vs_v5']] == [D_RECORD],
        'certificate_recomputed_equal': all(cert_equal.values()),
        'certificate_certified_at_150_labelled': (cert['label'] == 'v6 from records' and cert['k_star'] == 150
                                                  and cert['decision']['status'] == 'certified'
                                                  and cert['decision']['version'] == 6),
        'certificate_t_sum_within_the_gap_bound': abs(cert['t_sum_k_star']) <= SC6.GAP_BOUND,
        'certificate_all_clean_in_run_equals_the_logs': cert['all_clean_crosscheck']['equal'] is True,
        'determinacy_rescore_recomputed_equal': rescore_equal,
        'determinacy_no_verdict_change': doc['item3_determinacy_rescore']['verdict_changes'] == []
        and not any(r.get('verdict_changed') for r in rows),
        'determinacy_every_committed_verdict_reproduced': not doc['item3_determinacy_rescore']['not_reproduced'],
        'movement_recomputed_equal': _jt(mov) == _jt(doc['item3_movement_record']),
        'guards_zero_in_the_committed_runs': all(not v['verify_0_failures'] for v in doc['guards'].values())
        and all(not v['verify_0_failures'] for v in replay['guards'].values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'certificate_fields_equal': cert_equal,
            'from_records': {'path': FROM_RECORDS['path'], 'sha256': doc_sha},
            'certificate': {'path': FROM_RECORDS['certificate'], 'sha256': cert_sha},
            'floor_replay': {'path': FLOOR_REPLAY['path'], 'sha256': replay_sha},
            'd_c52e1670_v6_from_records': {k: cert.get(k) for k in ('k_star', 'window', 'W', 'band_width',
                                                                     'range_over_tau', 't_sum_k_star')},
            'movement': {k: mov[k] for k in ('largest_endpoint_movement_over_tau', 'largest_excursion_over_tau',
                                             'statement_holds_on_the_endpoint_movement')}}


# ======================================================================================================================
#  O, S -- W132's, reused
# ======================================================================================================================
def tests_O():
    r = K132.tests_O()
    missing = [c for c in V6.CELL_ORDER if c not in (r.get('cells') or {})]
    return dict(r, v6_cells_all_in_section_o=not missing, holds=bool(r.get('holds') and not missing))


def tests_S():
    r = K132.production_since_originals(CODE_PINNED_BY_CHECKS)
    return {'holds': r['ok'], **r}


# ======================================================================================================================
#  H -- the hooks through the real wrappers, on the v6 state
# ======================================================================================================================
HOLDS_OFF = K132.HOLDS_OFF
HOLDS_ON = K132.HOLDS_ON


def drive(cell, variant='certify', first_pass_at=50, plan=None, module=V6, state_cls=None):
    return K139.drive(cell, variant, first_pass_at=first_pass_at, plan=plan, module=module,
                      state_cls=state_cls or V6.ResettleStateV6)


def pure_rule(decl):
    cr = decl['cap_rule']
    p_max = decl['settling_rule']['p_max']
    if cr['kind'] == 'fixed':
        return SC6.SettlingRuleV6(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return SC6.SettlingRuleV6(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


def pure_equal(decl, lines):
    rule = pure_rule(decl)
    pure = [rule.observe(x['cycle'], x.get('gross'), bool(x.get('boyd_k')), x.get('t_sum'), bool(x.get('all_clean_k')))
            for x in lines]
    return [_jt(x.get('settling')) for x in lines] == [_jt(p) for p in pure], rule.decision


def drive_summary(d):
    """W139's drive summary with the in-cycle rule compared against the PURE v6 rule (W139's compares against v5)."""
    s = K139._drive_summary(d)
    lines = d['files'].get(V.CYCLE_FILE, [])
    ok, dec = pure_equal(d['state'].decl, lines) if lines else (False, None)
    s['in_cycle_rule_equals_pure_replay'] = ok
    s['pure_decision_status'] = (dec or {}).get('status')
    return s


def _h_real_v6(srp, which):
    """K105's real-production hold tests on the v6 state and the nine wrappers (W139's `_h_real_v5` on the v6 state)."""
    real_state, real_make, real_wrapped = K105._state, K105.E.make_wrappers, K105.E.WRAPPED
    cell = V6.CELL_ORDER[0]

    def state():
        st = V6.ResettleStateV6(V6.declaration_for(cell), None, V6.spec_cap(cell), reference={}, sink=[])
        st.first_pass = K105.E.N_HOLD
        return st
    K105._state = state
    K105.E.make_wrappers = lambda st, originals, srp_module=None: V6.make_wrappers(
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
    import importlib
    ext6 = importlib.import_module(V6.EXT_HOOKS_MODULE)
    ext5 = importlib.import_module(V5.EXT_HOOKS_MODULE)
    v6d = V6.declaration_for(V6.CELL_ORDER[0])
    ext6d = ext6.declaration_for(ext6.CELL_ORDER[0])
    v5d = V5.declaration_for(K139.CELL3)
    ext5d = ext5.declaration_for(ext5.CELL_ORDER[0])
    v4d, v3d = V4.declaration_for(K137.CELL1), V.declaration_for(K137.CELL1)
    w135d, w118d = W135.declaration_for(W135.CELL_ORDER[0]), R.declaration_for('f2_challenger')
    return {'v6_to_w142': H.resettle_hooks_module(v6d) is V6,
            'ext_v6_to_the_w142_extension': H.resettle_hooks_module(ext6d) is ext6,
            'v5_to_w139': H.resettle_hooks_module(v5d) is V5,
            'ext_v5_to_the_w139_extension': H.resettle_hooks_module(ext5d) is ext5,
            'v4_to_w137': H.resettle_hooks_module(v4d) is V4,
            'v3_to_w132': H.resettle_hooks_module(v3d) is V,
            'w135_to_w135': H.resettle_hooks_module(w135d) is W135,
            'w118_to_w118': H.resettle_hooks_module(w118d) is R,
            'harness_validates_v6': H.validate_settling_resettle(v6d) == v6d,
            'harness_validates_ext_v6': H.validate_settling_resettle(ext6d) == ext6d,
            'harness_validates_v5_unchanged': H.validate_settling_resettle(v5d) == v5d,
            'harness_validates_ext_v5_unchanged': H.validate_settling_resettle(ext5d) == ext5d,
            'harness_validates_v4_unchanged': H.validate_settling_resettle(v4d) == v4d,
            'harness_validates_v3_unchanged': H.validate_settling_resettle(v3d) == v3d,
            'harness_validates_w135_unchanged': H.validate_settling_resettle(w135d) == w135d,
            'harness_validates_w118_unchanged': H.validate_settling_resettle(w118d) == w118d}


def _h_layering(srp):
    """W139's real-install test restated for the v6 state and the eight-way dispatch."""
    before = {name: getattr(srp, name) for name in V.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w142_layering_')
    cell = V6.CELL_ORDER[0]
    n = V6.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with V6.settling_resettle_hooks(scratch, V6.declaration_for(cell), holder, cap=V6.spec_cap(cell)) as st:
            installed = {name: getattr(srp, name) for name in V.WRAPPED}
            all_installed = all(installed[k] is not before[k] for k in V.WRAPPED)
            v6_state = isinstance(st, V6.ResettleStateV6) and isinstance(st.rule, V6.HookedRuleV6)
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
    i_disp = child_src.find('W118C = resettle_hooks_module(resettle)')
    i_pre = child_src.find('W118C.assert_resettle_preconditions(resettle, spec, tail_checklist, aa_on)')
    i_cm = child_src.find('W118C.settling_resettle_hooks(eval_dir, resettle, holder, int(spec[\'cap\']))')
    pre_ok = 0 < i_disp < i_pre < child_src.find('G.run_admm_arm(') and i_cm > i_pre
    disp = dispatch_checks()
    summ = holder.get(V.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and pre_ok and idc_outer and all_installed
                          and v6_state and all(disp.values()) and summ.get('phase') == 'ended'
                          and summ.get('schema') == V6.SCHEMA and summ.get('criterion_version') == 6
                          and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                          and admm.minimum_consecutive_converged_cycles == 10),
            'nine_wrappers_installed': all_installed, 'v6_state_and_rule': v6_state,
            'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'child_real_dispatch_then_checklist_before_run_admm_arm': pre_ok, 'harness_dispatch': disp,
            'files_written_by_install_without_cycles': files_written, 'summary_phase': summ.get('phase'),
            'summary_schema': summ.get('schema'), 'summary_errors': summ.get('errors')}


def tests_H():
    import shared_resources_planning as srp
    res = {}
    h1 = {}
    for cell in V6.GATED_CELLS:
        s = drive_summary(drive(cell, 'certify'))
        c = V6.CELLS[cell]
        s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                       and s['first_k0_v2'] == c['k0'] and s['n_lapses'] == 1 + len(c['original_lapses_after_k0'])
                       and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                       and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                       and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                       and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['exit_capture_complete']
                       and s['clean_capture_complete'] and s['exits_51_every_line'] and s['clean_51_every_line']
                       and s['all_clean_every_line_is_bool'] and s['all_clean_is_the_conjunction']
                       and s['non_clean_cycles'] == [] and s['n_vetoes'] == 0 and s['certificate_length_after'] == 10
                       and s['lines'] == s['k_star'] == s['creep_lines'] and s['decision_version'] == 6
                       and s['out_of_window_reads_recorded'] is True
                       and s['holds_through_first_pass'] == [HOLDS_OFF] and s['holds_after_first_pass'] == [HOLDS_ON]
                       and s['t_sum_every_line'] and not s['capture_errors'])
        s.pop('summary')
        h1[cell] = s
    res['H1_gated_original_values_bitwise_through_k0_then_certify'] = {'holds': all(v['ok'] for v in h1.values()),
                                                                        'cells': h1}
    s = drive_summary(drive('l_7b199ef9', 'ulp_at_40'))
    s.pop('summary')
    s['ok'] = bool(s['raised'] and 'REPLAY DIVERGED at cycle 40' in s['raised'] and s['lines'] == 40
                   and (s['first_divergence'] or {}).get('fields_differing') == ['gross_operational_cost']
                   and s['replay_bitwise_through'] == 39 and not s['summary_ok']
                   and s['stopped_by'] == 'replay_divergence_abort')
    res['H2_one_ulp_at_40_aborts'] = {'holds': s['ok'], 'detail': s}
    cell = 'h_74eda68d'
    base = drive_summary(drive(cell, 'certify'))
    ks = base['k_star']
    probe = drive(cell, 'certify', plan={})
    keys = list(probe['files'][V.CYCLE_FILE][0]['ipopt_exit_by_block'])
    dso7 = next(k for k in keys if k.startswith('DSO|7|') and k.endswith('|Winter'))
    tso = next(k for k in keys if k.startswith('TSO|'))

    def one(name, plan, want_veto):
        s_ = drive_summary(drive(cell, 'certify', plan=plan))
        d = {k: v for k, v in s_.items() if k != 'summary'}
        if want_veto:
            ok = (s_['raised'] is None and s_['n_lapses'] == base['n_lapses'] and s_['n_vetoes'] >= 1
                  and s_['vetoes'][0]['cycle'] == ks and s_['in_cycle_rule_equals_pure_replay'] and s_['summary_ok']
                  and (s_['status'] != 'certified' or s_['k_star'] > ks) and s_['all_clean_is_the_conjunction'])
        else:
            ok = (s_['raised'] is None and s_['status'] == 'certified' and s_['k_star'] == ks
                  and s_['n_lapses'] == base['n_lapses'] and s_['n_vetoes'] == 0 and s_['non_clean_cycles'] == []
                  and s_['in_cycle_rule_equals_pure_replay'] and s_['summary_ok']
                  and s_['all_clean_is_the_conjunction'])
        d['ok'] = bool(ok)
        d['k_star_all_optimal'] = ks
        res[name] = {'holds': d['ok'], 'detail': d}
    one('H3_primary_acceptable_2p51_inside_the_window_does_not_veto', {ks - 1: {dso7: 'acc_2p51'}}, False)
    one('H3b_primary_acceptable_11x_inside_the_window_vetoes', {ks - 1: {dso7: 'acc_11x'}}, True)
    one('H3c_recovery_acceptable_inside_the_window_vetoes', {ks - 1: {tso: 'acc_2p51@recovery'}}, True)
    one('H3d_esso_acceptable_within_10x_does_not_veto', {ks - 1: {'ESSO|7': 'esso_acc_within'}}, False)
    one('H3d2_esso_acceptable_11x_vetoes', {ks - 1: {'ESSO|7': 'esso_acc_11x'}}, True)
    one('H3e_optimal_recovery_does_not_veto', {ks - 1: {tso: 'opt@recovery'}}, False)
    one('H3f_primary_acceptable_exactly_10x_does_not_veto', {ks - 1: {dso7: 'acc_10x'}}, False)
    d4 = drive('c_156ce2d1', 'creep')
    s = drive_summary(d4)
    rec = {'status': 'certified', 'barrier': False, 'barrier_cause': None, 'certified_cost': 1.0,
           'certification_cycle': s['last_cycle'], 'terminal_gross_operational_cost': 1.0,
           'settling_resettle_summary': s['summary']}
    relabelled = H._apply_settling_resettle_status(dict(rec))
    summ4 = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'cap'
                   and s['last_cycle'] == V6.spec_cap('c_156ce2d1') and s['certificate_length_after'] == 10
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['ended_by'] == 'cap'
                   and relabelled['status'] == 'not_certified' and relabelled['certified_cost'] is None
                   and relabelled['status_production_trajectory']['status'] == 'certified'
                   and summ4['settling_status'] == 'uncertified' and summ4['schema'] == V6.SCHEMA)
    res['H4_gated_creep_uncertified_at_the_cap_and_label'] = {'holds': s['ok'], 'detail': s}
    s = drive_summary(drive('g_37b5c499', 'creep', first_pass_at=50))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'rule_cap'
                   and s['last_cycle'] == 50 + V6.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 50
                   and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['n_overlap'] == 0
                   and s['holds_through_first_pass'] == [HOLDS_OFF] and s['holds_after_first_pass'] == [HOLDS_ON])
    res['H5_ungated_dynamic_rule_cap'] = {'holds': s['ok'], 'detail': s}
    s = drive_summary(drive('g_9abf31d4', 'certify', first_pass_at=100))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['first_pass'] == 100 and s['summary_ok']
                   and s['in_cycle_rule_equals_pure_replay'] and s['stopped_by'] == 'settling_rule')
    res['H5b_ungated_certifies'] = {'holds': s['ok'], 'detail': s}
    s = drive_summary(drive('l_2ab0ce2d', 'creep'))
    s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['last_cycle'] == 437
                   and s['rule_cap'] == 437 and s['stopped_by'] == 'cap' and s['summary_ok']
                   and s['replay_bitwise_through'] == 328 and s['in_cycle_rule_equals_pure_replay'])
    res['H6_cap_437_above_300'] = {'holds': s['ok'], 'detail': s}
    res['H7_aa_hold_real_production'] = _h_real_v6(srp, 'aa')
    res['H8_tail_hold_real_production'] = _h_real_v6(srp, 'tail')
    res['H9_rho_hold_real_production'] = _h_real_v6(srp, 'rho')
    res['H10_layering_real_install_and_dispatch'] = _h_layering(srp)
    got = V._certificate_length_writes_in_source()
    in_cycle_v6_src = ''.join(inspect.getsource(o) for o in (V6.HookedRuleV6, V6.ResettleStateV6, V6._v6_rule,
                                                              V6.settling_resettle_hooks, SC6))
    res['H11_certificate_length_writes'] = {
        'holds': (got == sorted(R.CERTIFICATE_LENGTH_WRITES) and 'early_stop' not in in_cycle_v6_src
                  and 'minimum_consecutive_converged_cycles' not in in_cycle_v6_src),
        'writes_in_source': got}
    for name, fn in (('H12_w132_exit_wrapper_on_real_production_reused', K132._h_exit_real),
                     ('H13_w139_v5_exit_wrapper_on_real_production_structures_reused', K139._h_exit_real_v5)):
        try:
            res[name] = fn()
        except Exception as error:  # noqa: BLE001
            res[name] = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    res['H14_v6_make_wrappers_is_the_w139_function'] = {
        'holds': V6.make_wrappers is V5.make_wrappers and V6.make_exit_wrapper is V5.make_exit_wrapper}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  C, C2 -- W139's, reused (the capture is unchanged in v6)
# ======================================================================================================================
def tests_C():
    return K139.tests_C()


def tests_C2():
    return K139.tests_C2()


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w142():
    return K137._harness_at(PRE_W142_HARNESS['commit'], PRE_W142_HARNESS['sha256'], '_w142_pre_harness')


def resettle_keys(pre=None):
    """The 34 v6 keys: formula over the (pre-W142) base key; the v5, v4 and v3 keys of the same cell beside."""
    out = {}
    for cell in V6.CELL_ORDER:
        _spec, e, kw = K132.resettle_kwargs(cell)
        base = H.evaluation_key(e['key'], e['overrides'], **kw)
        decl = V6.declaration_for(cell)
        key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=decl, **kw)
        key_v5 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V5.declaration_for(cell), **kw)
        key_v4 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V4.declaration_for(cell), **kw)
        key_v3 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V.declaration_for(cell), **kw)
        base_pre = pre.evaluation_key(e['key'], e['overrides'], **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[cell] = {'candidate_key': e['key'], 'original_eval_key': e['eval_key'], 'base_key_now': base,
                     'resettle_key': key, 'v5_resettle_key': key_v5, 'v4_resettle_key': key_v4,
                     'v3_resettle_key': key_v3, 'formula_holds': key == formula, 'base_equals_pre_w142': base == base_pre,
                     'differs_from_original': key != e['eval_key'] and base != e['eval_key'],
                     'differs_from_v5_v4_v3': key not in (key_v5, key_v4, key_v3)}
    return out


def tests_K():
    import importlib
    ext6 = importlib.import_module(V6.EXT_HOOKS_MODULE)
    ext5 = importlib.import_module(V5.EXT_HOOKS_MODULE)
    pre, pre_sha, _src = _harness_pre_w142()
    n_specs = n_entries = n_equal = n_frozen_equal = 0
    own = {'v6': {'n': 0, 'ok': 0, 'bad': []}, 'ext_v6': {'n': 0, 'ok': 0, 'bad': []}}
    fam_count = {'w118': 0, 'v3': 0, 'v4': 0, 'w135': 0, 'v5': 0, 'ext_v5': 0}
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
                fam = None
                if isinstance(rs, dict) and rs.get('schema') == V6.DECLARATION_SCHEMA:
                    fam, mod = 'v6', V6
                elif isinstance(rs, dict) and rs.get('schema') == V6.EXT_DECLARATION_SCHEMA:
                    fam, mod = 'ext_v6', ext6
                if fam is not None:
                    own[fam]['n'] += 1
                    decl = mod.validate_settling_resettle(rs)
                    formula = hashlib.sha256(json.dumps({'base_evaluation_key': pre.evaluation_key(*args, **kw),
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
                if isinstance(rs, dict) and rs.get('schema') == V5.DECLARATION_SCHEMA:
                    fam_count['v5'] += 1
                elif isinstance(rs, dict) and rs.get('schema') == V5.EXT_DECLARATION_SCHEMA:
                    fam_count['ext_v5'] += 1
                elif V4.is_v4_declaration(rs):
                    fam_count['v4'] += 1
                elif V.is_v3_declaration(rs):
                    fam_count['v3'] += 1
                elif W135.is_w135_declaration(rs):
                    fam_count['w135'] += 1
                elif rs is not None:
                    fam_count['w118'] += 1
                new = H.evaluation_key(*args, settling_resettle=rs, **kw)
                old = pre.evaluation_key(*args, settling_resettle=rs, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_equal += 1
                if H._entry_eval_key(e) == new:
                    n_frozen_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    keys = resettle_keys(pre)
    holders = {cell: K137._key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: K137._outside_root(h, W142_ROOT_REL) for cell, h in holders.items()}
    probe = keys[V6.CELL_ORDER[0]]['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w142_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    ctl = {}
    for name, rel in (('outside', os.path.join(_P53, 'w142_planted_negative_control', 'campaign_planted',
                                               'campaign_spec_planted_00000000.json')),
                      ('sibling_prefix', os.path.join(W142_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                                      'campaign_spec_planted_00000000.json')),
                      ('ext_v6_root', os.path.join(W142_EXT_ROOT_REL, 'campaign_planted',
                                                   'campaign_spec_planted_00000000.json')),
                      ('v5_root', os.path.join(W139_ROOT_REL, 'campaign_planted', 'campaign_spec_planted_00000000.json')),
                      ('v4_root', os.path.join(W137_ROOT_REL, 'campaign_planted', 'campaign_spec_planted_00000000.json')),
                      ('w135_root', os.path.join(W135_ROOT_REL, 'campaign_planted',
                                                 'campaign_spec_planted_00000000.json'))):
        ctl[name] = {'rel': rel, 'refused': rel in K137._outside_root(K137._key_holders(scanned + [planted(rel)], probe),
                                                                      W142_ROOT_REL)}
    own_rel = os.path.join(W142_ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{V6.CELL_ORDER[0]}',
                           f'campaign_spec_{CAMPAIGN_ID_PREFIX}{V6.CELL_ORDER[0]}_00000000.json')
    own_holders = K137._key_holders(scanned + [planted(own_rel)], probe)
    try:
        pre.validate_settling_resettle(V6.declaration_for(V6.CELL_ORDER[0]))
        pre_refuses = False
    except ValueError:
        pre_refuses = True
    try:
        pre.validate_settling_resettle(ext6.declaration_for(ext6.CELL_ORDER[0]))
        pre_refuses_ext = False
    except ValueError:
        pre_refuses_ext = True
    ss5 = json.load(open(_abs(V5_STAGE_SPEC['path'])))
    v5_frozen = {c: json.load(open(_abs(ss5['pins']['campaign_specs'][c]['path'])))['candidates'][0]['eval_key']
                 for c in V6.CELL_ORDER}
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_other_entry_equal': (n_equal == n_entries - own['v6']['n'] - own['ext_v6']['n']
                                                   and n_entries > 0),
        'w118_entries_scanned_and_equal': fam_count['w118'] >= 10,
        'v3_38_entries_scanned_and_equal': fam_count['v3'] == 38,
        'v4_38_entries_scanned_and_equal': fam_count['v4'] == 38,
        'w135_7_entries_scanned_and_equal': fam_count['w135'] == 7,
        'v5_36_entries_scanned_and_equal': fam_count['v5'] == 36,
        'ext_v5_7_entries_scanned_and_equal': fam_count['ext_v5'] == 7,
        'own_v6_entries_inside_the_v6_root_follow_the_formula': own['v6']['ok'] == own['v6']['n'] and not own['v6']['bad'],
        'own_ext_v6_entries_inside_the_ext_root_follow_the_formula': (own['ext_v6']['ok'] == own['ext_v6']['n']
                                                                      and not own['ext_v6']['bad']),
        'v6_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'v6_base_equals_pre_w142_every_cell': all(v['base_equals_pre_w142'] for v in keys.values()),
        'v6_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'v6_keys_differ_from_the_v5_v4_v3_keys': all(v['differs_from_v5_v4_v3'] for v in keys.values()),
        'v5_keys_recomputed_equal_the_frozen_v5_specs': all(v5_frozen[c] == keys[c]['v5_resettle_key']
                                                            for c in V6.CELL_ORDER),
        'v6_keys_distinct': len({v['resettle_key'] for v in keys.values()}) == len(keys) == 34,
        'v6_keys_absent_from_committed_specs_outside_the_v6_root': not any(outside.values()),
        **{f'control_planted_{k}_refused': v['refused'] for k, v in ctl.items()},
        'control_own_campaign_spec_accepted': own_rel in own_holders and own_rel not in K137._outside_root(
            own_holders, W142_ROOT_REL),
        'control_v6_declaration_refused_by_the_pre_w142_harness': pre_refuses,
        'control_ext_v6_declaration_refused_by_the_pre_w142_harness': pre_refuses_ext,
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w142_harness': {**PRE_W142_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'entries_whose_frozen_eval_key_equals_the_recomputed_REPORTED': n_frozen_equal,
            'families_scanned': fam_count,
            'own_entries': {k: {'n': v['n'], 'ok': v['ok'], 'bad': v['bad'][:20]} for k, v in own.items()},
            'own_roots': KEY_OWN_ROOTS, 'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'v6_keys_in_committed_specs_all_REPORTED': holders, 'controls': ctl,
            'control_own': {'rel': own_rel, 'holders': own_holders}}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    out = {}
    ok = True
    tail = {'convergence_depth_tail': {'enabled': True, 'compl_inf_tol': 1e-6}}
    for cell in V6.CELL_ORDER:
        decl = V6.declaration_for(cell)
        cap = V6.spec_cap(cell)
        try:
            good = V6.assert_resettle_preconditions(decl, {'cap': cap, 'configuration': tail},
                                                    {'tail_enabled_for_this_run': True}, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        negatives = {}
        for name, (spec, tl, aa) in {
                'cap_wrong': ({'cap': cap - 1, 'configuration': tail}, {'tail_enabled_for_this_run': True}, True),
                'tail_off': ({'cap': cap, 'configuration': tail}, {'tail_enabled_for_this_run': False}, True),
                'aa_off': ({'cap': cap, 'configuration': tail}, {'tail_enabled_for_this_run': True}, False),
                'tail_value_not_the_table': ({'cap': cap, 'configuration': {'convergence_depth_tail': {
                    'enabled': True, 'compl_inf_tol': 1e-5}}}, {'tail_enabled_for_this_run': True}, True)}.items():
            try:
                V6.assert_resettle_preconditions(decl, spec, tl, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {str(error)[:200]}'
        rule = decl['settling_rule']
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'cell_kept_under_v4': dict(decl, cell='b_2a0ba8b2'),
               'cell_certified_under_v5': dict(decl, cell='b_4649234b'),
               'cell_certified_v6_from_records': dict(decl, cell=D_CELL),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'v5_schema_with_v6_rule': dict(decl, schema=V5.DECLARATION_SCHEMA),
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'rule_v5': dict(decl, settling_rule=V5.settling_rule_declaration()),
               'floor_tau_over_20': dict(decl, settling_rule=dict(rule, swing_floor=dict(rule['swing_floor'],
                                                                                         F=SC6.TAU / 20.0))),
               'floor_B_dropped': dict(decl, settling_rule=dict(rule, swing_floor=dict(
                   rule['swing_floor'], carries=['A_growth_test_floor']))),
               'clean_factor_5': dict(decl, settling_rule=dict(rule, clean=dict(rule['clean'], factor=5.0))),
               'reset_on_non_clean': dict(decl, settling_rule=dict(rule, reset_on_non_clean=True)),
               'cap_rule_changed': dict(decl, cap_rule=dict(decl['cap_rule'], ceiling=999)),
               'clean_capture_off': dict(decl, captures=dict(decl['captures'], exit_clean_by_block=False))}
        if decl['replay_reference'] is not None:
            bad.update({'wrong_k0': dict(decl, first_residual_pass_expected=decl['first_residual_pass_expected'] - 1),
                        'wrong_sha': dict(decl, replay_reference=dict(decl['replay_reference'], sha256='0' * 64)),
                        'no_reference': dict(decl, replay_reference=None)})
        refused = {}
        for name, d in bad.items():
            try:
                V6.validate_settling_resettle(d)
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
#  X -- W137's section X reused;  D -- the router
# ======================================================================================================================
def tests_X():
    return K137.tests_X()


def harness_router_check():
    _pre, _sha_pre, pre_src = _harness_pre_w142()
    now_src = open(_abs('p515_s44_campaign_harness.py')).read()
    import difflib
    diff = [ln for ln in difflib.unified_diff(pre_src.splitlines(), now_src.splitlines(), lineterm='', n=0)
            if not ln.startswith(('---', '+++', '@@'))]
    added = [ln[1:] for ln in diff if ln.startswith('+')]
    removed = [ln[1:] for ln in diff if ln.startswith('-')]
    fn_src = inspect.getsource(H.resettle_hooks_module)
    i132 = fn_src.find('W132C.is_v3_declaration(value)')
    i135 = fn_src.find('W135C.is_w135_declaration(value)')
    i137 = fn_src.find('W137C.is_v4_declaration(value)')
    i139 = fn_src.find('W139C.is_v5_declaration(value)')
    i142 = fn_src.find('W142C.is_v6_declaration(value)')
    i118 = fn_src.find('import p515_s53_w118_resettle_hooks as W118C')
    sha_now = _sha('p515_s44_campaign_harness.py')
    last = H._git(['log', '--format=%H %s', '-1', '--', 'p515_s44_campaign_harness.py'])
    pinned = harness_post_v6_sha256()
    parts = {
        'no_line_removed': removed == [],
        'exactly_three_lines_added': len(added) == 3,
        'the_three_added_are_the_v6_branch': tuple(added) == V6_BRANCH_LINES,
        'order_v3_w135_v4_v5_v6_then_w118': 0 <= i132 < i135 < i137 < i139 < i142 < i118,
        'harness_committed_clean': _clean('p515_s44_campaign_harness.py'),
        'harness_sha256_is_the_pinned_post_v6_sha256': pinned is not None and sha_now == pinned,
        'last_harness_commit_message_carries_the_sha': pinned is not None and pinned in last,
        'dispatch_eight_families': all(dispatch_checks().values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'added': added, 'removed': removed,
            'harness_sha256_now': sha_now, 'pre_w142_harness': PRE_W142_HARNESS, 'post_v6_sha256_pinned': pinned,
            'last_harness_commit': last}


def tests_D():
    return harness_router_check()


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('R', tests_R), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('C', tests_C),
            ('C2', tests_C2), ('K', tests_K), ('P', tests_P), ('X', tests_X), ('D', tests_D))

CODE_PINNED_BY_CHECKS = tuple(dict.fromkeys(
    (os.path.basename(__file__), 'settling_criterion_v6.py', 'p515_s53_w142_resettle_v6_hooks.py',
     'p515_s53_w142_resettle_ext_v6_hooks.py', 'p515_s53_w142_v6_from_records.py', 'p515_s53_w142_floor_replay.py',
     'p515_s53_w142_determinacy.py', 'p515_s53_w141_swing_variants.py', 'p515_s53_w139_resettle_v5_checks.py',
     'p515_s53_w139_resettle_ext_v5_hooks.py', 'p515_s53_w132_resettle_v3_campaign.py')
    + K139.CODE_PINNED_BY_CHECKS))


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
    return {'all_hold': ok, 'p_max': V6.P_MAX, 'constants': SC6.constants(V6.P_MAX), 'readings': SC6.READINGS,
            'sub_test_reads': SC6.SUB_TEST_READS,
            'clean_rule': {'factor': SC6.CLEAN_FACTOR, 'metric_table': SC6.METRIC_TABLE,
                           'tolerances': SC6.TOLERANCES, 'tolerance_sources': SC6.TOLERANCE_SOURCES},
            'sections': out}


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
    doc = {'schema': 'p515_s53_w142_zero_solve_checks_v1',
           'task': 'W142 (PLANNER_BRIEF_2026-09-13.md Addendum 61)',
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
        print(f"[W142-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W142-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W142-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) guards={guards}")
    print(f"[W142-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
