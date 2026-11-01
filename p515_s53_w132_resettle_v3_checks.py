"""
P5.15 Addendum 58, Planner task W132 -- ZERO-SOLVE checks for the v3 re-settling campaign (claim groups 1, 2 and 4;
38 cells; frozen_s53_resettle_spec_v3, predecessor v2 fc791891).

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W118 / W105 /
W101 / W98 checks modules and the W112 script, imported for their helpers, arm their own; every guard is verified at 0).
No model is built and nothing is solved: H12 and H6 read a fresh SRP1 planning object (data only); everything else reads
committed, manifest- or pin-verified records.

WHAT IS CHECKED
  V   the stop rule VERSION 3 (`settling_criterion_v3`): V_v2 the W118 version-2 suite (unchanged, run as committed);
      V_eq version 3 with every cycle Optimal reproduces version 2 record by record and decision by decision on the
      suite's trajectories; V18 a non-Optimal LAPSE (k0 reset, N and the dynamic cap unchanged, both k0's recorded);
      V19 a GAMMA VETO (no reset; certification blocked while the non-Optimal cycle lies in the window; alpha
      unaffected); V20 a non-Optimal cycle AT the v2 first pass (N and the cap keyed on v2, first_k0_alpha later); V21
      the per-cell ceiling (437 accepted with ceiling 437, refused with 300; no ceiling refused); V22 all_optimal must be a
      bool; V0 settling_criterion.py (v1) and settling_criterion_v2.py byte-identical to their pins.
  R   VALIDATION BY REPLAY: version 3 on the committed records of the 10 W118 cells and the 2 settled references, fed
      the per-cycle all_optimal W131 reconstructed (w131_prefreeze_diagnostics.json afd5a0d7, manifest-verified): alpha
      and gamma reproduce W131's (status, k*, k0; F2 cells uncertified) -- 11 hold, pb_y2025_n5 not certified within
      its records; the all-Optimal base replay reproduces every committed W118 decision; x0 / unit certify at 181 / 172.
  O   the 38 original cells: record committed, in its campaign manifest, sha as pinned, first residual pass and lapses
      as declared, base key (original configuration) == original eval key; CONFIGURATION IDENTITY against the original
      spec (case file, AA, arm, rho, diagnostics, overrides, flexibility multiplier, investment year; gated cells: the
      ESS ageing declaration and the C2 ESS parameter file; ungated G cells: the declared first-C2 difference); the tail
      the one declared change; the lattice check SCOPED TO E/P for every point (2 <= E/P <= 4; no substitution; the
      other lattice rules reported); I_j per cell from W117's claims (3b2e76de; w117_triage_recompute.json).
  S   production_since_originals: production .py changes between each original run's git head and HEAD (report); no
      uncommitted change to a file this run uses (gate).
  H   the hooks through the REAL wrappers (W118's eight + the exit wrapper) with stand-in production: H1 every gated
      cell with its ORIGINAL values bitwise through k0, then a synthetic continuation that certifies (51 exits every
      line; in-cycle rule == pure v3 replay); H2 a one-ulp divergence aborts; H3 a non-Optimal ESSO exit is a lapse
      (cause non_optimal), certification later; H4 a gated creep uncertified at the cap and the record label fix on its
      summary; H5 / H5b ungated: dynamic rule cap / certification; H6 2ab0ce2d to its cap 437 (above 300); H7-H9 the
      AA / tail / rho holds with REAL production; H10 the real install layering (nine wrappers; the harness dispatch and
      order); H11 certificate-length writes; H12 the exit wrapper on REAL production `_admm_local_solves_succeeded`
      with real pyomo SolverResults on a fresh SRP1 planning: 51 entries in production's order, ESSO INCLUDED, value
      unchanged, an Acceptable ESSO exit -> all_optimal False, a missing key raises.
  K   keys: every entry of every committed campaign spec keys identically under this harness and the pre-W132 harness
      (e2c1ac09, sha256 pinned, from git) -- W118's settling_resettle entries included -- except the campaign's OWN v3
      entries, accepted only inside the W132 stage root and only if their frozen key follows the formula; the 38 keys
      follow the formula, are distinct, differ from the originals and appear in no committed spec outside the own root;
      planted controls (outside refused; sibling-prefix refused; own accepted).
  P   `assert_resettle_preconditions` holds for every cell with its cap; negative controls refused; the validator
      refuses early_stop and malformed declarations.
  X   the status-label writer fix (`p515_s44_campaign_harness._apply_settling_resettle_status`): W123's committed F2
      challenger record ('certified' at an uncertified cap) relabels to not_certified; a certified summary keeps
      certified; records without a re-settling summary are unchanged; build_evaluation_record applies it before the
      child writes the record and prints the finish line (source order).
  W   (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w132_resettle_v3_checks.py \\
      > data/SRP1/Results/P515S53/w132_resettle_v3/zero_solve_checks_launch.log 2>&1
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W132 v3 re-settling zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w118_resettle_checks as K118  # noqa: E402 -- the v2 suite and helpers (arms its own guard)
import p515_s53_w105_extension_checks as K105  # noqa: E402 -- imported by K118 (its own guard)
import p515_s53_w101_continuation_checks as K101  # noqa: E402 -- imported by K118 (its own guard)
import p515_s53_w98_continuation_checks as K98  # noqa: E402 -- imported by K118 (its own guard)
import p515_s53_w112_consensus_gap as W112  # noqa: E402 -- imported by K118 (its own guard)
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v3 as SC3  # noqa: E402


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w132_checks', GUARD),) + tuple(K118.GUARDS))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W132_ROOT_REL = os.path.join(_P53, 'w132_resettle_v3')
W118_ROOT_REL = K118.W118_ROOT_REL
OUT_DIR_REL = os.path.join(W132_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w132_zero_solve_checks.json'
OUT_MANIFEST = 'w132_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w132_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = 's53_w132_resettle_v3_'
KEY_EXCLUDED_ROOTS = (W132_ROOT_REL,)
PRE_W132_HARNESS = {'commit': 'e2c1ac0939685360fb3106578d9a4b695c07576d',
                    'sha256': '4115fc8376ab7038b4a9adbca34c4d432a07cb6e91fa21f103949b5c885a36ae'}
W118_SPEC = {'path': os.path.join(W118_ROOT_REL, 'frozen_s53_resettle_spec_v2_fc791891.json'),
             'sha256': 'fc791891'}
W131 = {'path': os.path.join(_P53, 'w131_prefreeze', 'w131_prefreeze_diagnostics.json'),
        'manifest': os.path.join(_P53, 'w131_prefreeze', 'manifest_sha256.json'),
        'sha256': 'afd5a0d7cde901d4a8813cb5d4fc56729da475e2fd14b5dac061c12eecd9d144', 'commit': '7a5c5717'}
W117 = {'path': os.path.join(_P53, 'w117_triage_recompute', 'w117_triage_recompute.json'),
        'manifest': os.path.join(_P53, 'w117_triage_recompute', 'manifest_sha256.json'),
        'sha256': 'e91fd86cf3615eed238dcb35cce6700edf396e9a8f1ad89f710415b313cf0333', 'commit': '3b2e76de'}
W118_CELL_DIRS = {
    'pb_y2030_n9': 'campaign_s53_w118_resettle_r2_pb_y2030_n9/evals/344c936fc23919be_pb_y2030_n9',
    'pb_y2030_n7': 'campaign_s53_w118_resettle_r2_pb_y2030_n7/evals/828f03fa9ac2ca6e_pb_y2030_n7',
    'pb_y2025_n5': 'campaign_s53_w118_resettle_r2_pb_y2025_n5/evals/ca29c5e818366a6e_pb_y2025_n5',
    'pb_y2030_n5': 'campaign_s53_w118_resettle_r2_pb_y2030_n5/evals/3846675acea98ea7_pb_y2030_n5',
    'pb_y2025_n9': 'campaign_s53_w118_resettle_r2_pb_y2025_n9/evals/b7bce5a8bf2365b9_pb_y2025_n9',
    'pb_y2025_n7': 'campaign_s53_w118_resettle_r2_pb_y2025_n7/evals/b9b8e4be4a6c82b6_pb_y2025_n7',
    'yl_y2030': 'campaign_s53_w118_resettle_r2_yl_y2030/evals/6e4f11a95dc3d585_yl_y2030',
    'yl_y2035': 'campaign_s53_w118_resettle_r2_yl_y2035/evals/3a2387f46448f9db_yl_y2035',
    'f2_challenger': 'campaign_s53_w118_resettle_r2_f2_challenger/evals/1fe91e86f11e76af_f2_challenger',
    'f2_incumbent': 'campaign_s53_w118_resettle_r2_f2_incumbent/evals/24c5ccb6f285219f_f2_incumbent',
}
REF_CELLS = {'x0': (K118.X0_CELL, os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0',
                                               'campaign_spec_s53_w101_srp1_cont_x0_d1d5c3bc.json')),
             'unit_n7_4h_e1': (K118.UNIT_CELL, None)}
EXPECTED_V3_REPLAY = {
    'rule': ('W131 (7a5c5717): 11 of 12 hold (the 10 W118 cells and the 2 references reproduce their committed '
             'decision under alpha and gamma; the F2 pair stays uncertified), pb_y2025_n5 is NOT certified within its '
             'recorded cycles 1..167 (non-Optimal cycle 167 = k*); pb_y2025_n9 alpha k0 moves 113 -> 135, still 174'),
    'not_certified_within_records': ['pb_y2025_n5'],
}
CASE_FILE_SHA256 = K118.CASE_FILE_SHA256
ESS_PARAMS_SHA256_C2 = K118.ESS_PARAMS_SHA256_C2
LATTICE_EP = {'min_e_over_p': 2.0, 'max_e_over_p': 4.0,
              'source': 'data/SRP1/SharedESS/SRP1_ESS_Params.json min/max_energy_to_power_factor 2 / 4 (current)'}
IDENTITY_KEYS = ('arm_label', 'apply_rho', 'case_file', 'case_file_sha256', 'case_file_anderson_acceleration',
                 'full_diagnostics_in_rows', 'overrides')
GATED_IDENTITY_KEYS = ('ess_ageing_baseline', 'ess_ageing_baseline_label')
PRODUCTION_FILES = ('shared_resources_planning.py', 'network.py', 'admm_parameters.py', 'admm_anderson_acceleration.py',
                    'model_construction_helpers.py', 'shared_energy_storage_data.py', 'helper_functions.py')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jt(x):
    return GRIO.dumps(x, default=GRIO.json_default, sort_keys=True)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git_tracked_clean(rel):
    return K118._git_tracked_clean(rel)


def _pinned_json(pin):
    """A committed JSON output, its sha256 checked against its pin and its manifest; git-clean."""
    now = _sha(pin['path'])
    man = json.load(open(_abs(pin['manifest'])))
    ok = (now == pin.get('sha256', now) and man.get(pin['path']) == now and _git_tracked_clean(pin['path']))
    if not ok:
        raise RuntimeError(f"{pin['path']}: sha256 {now} vs pin {pin.get('sha256')} / manifest {man.get(pin['path'])}, "
                           f"clean {_git_tracked_clean(pin['path'])}")
    return json.load(open(_abs(pin['path']))), now


# ======================================================================================================================
#  V -- the stop rule v3
# ======================================================================================================================
_V3_ONLY_KEYS = ('boyd_k_v2', 'all_optimal_k', 'boyd_k_v3', 'first_residual_pass_v2', 'first_k0_alpha', 'lapse_cause',
                 'gamma')


def _v2_view(rec):
    return {k: v for k, v in rec.items() if k not in _V3_ONLY_KEYS and k != 'first_residual_pass'}


def _dec_view(dec):
    if dec is None:
        return None
    out = {k: v for k, v in dec.items() if k not in ('version', 'reading', 'N_definition', 'first_k0_alpha',
                                                   'non_optimal_cycles', 'cap_ceiling', 'gamma_report_only',
                                                   'lapse_events')}
    out['lapse_events'] = [{k: e[k] for k in ('cycle', 'k0_before', 'q_is_none')} for e in dec.get('lapse_events') or []]
    return out


def _both(q, b, t, kw, all_opt=None):
    """(v2 run, v3 run) on the same trajectory; v3 fed all_optimal True everywhere unless given."""
    all_opt = {k: True for k in q} if all_opt is None else all_opt
    o2, d2, r2 = SC2.replay(q, b, t, R.P_MAX, **kw)
    kw3 = dict(kw)
    kw3.setdefault('cap_ceiling', SC2.CAP_CEILING)
    o3, d3, r3 = SC3.replay(q, b, t, all_opt, R.P_MAX, **kw3)
    return (o2, d2, r2), (o3, d3, r3)


def _trajectories():
    """The W118 version-2 suite's trajectories (restated from p515_s53_w118_resettle_checks.tests_V) for the version-3
    equivalence test: (name, q, boyd, t_sum, replay kwargs)."""
    out = []
    x0q, x0b, _ = K101.load_record('3x3_x0_continuation_1_88')
    n7q, n7b, _ = K101.load_record('3x3_node7_1_69')
    for k in (72, 77, 86, 88):
        q = {c: x0q[c] for c in range(1, k + 1)}
        out.append((f'x0_trunc_{k}', q, {c: x0b[c] for c in q}, {c: 0.0 for c in q}, {'cap': k}))
    out.append(('node7_1_69', n7q, n7b, {c: 0.0 for c in n7q}, {'cap': max(n7q)}))

    def damped(boyd_false=()):
        q = {k: 1e9 + 30000.0 * 0.88 ** (k - 11) * math.cos(2 * math.pi * (k - 11) / 20.0) for k in range(1, 201)}
        return q, {k: (k >= 11 and k not in boyd_false) for k in range(1, 201)}
    q6, b6 = damped()
    out.append(('damped_cosine', q6, b6, {k: 0.0 for k in q6}, {'cap': 200}))
    q7, b7 = damped(boyd_false=(36,))
    out.append(('damped_cosine_boyd_lapse_36', q7, b7, {k: 0.0 for k in q7}, {'cap': 200}))
    q8 = {k: 1e9 - 80.0 * (k - 1) for k in range(1, 121)}
    out.append(('constant_creep', q8, {k: True for k in q8}, {k: 0.0 for k in q8}, {'cap': 120}))
    q9 = {1: 1e9}
    for k in range(2, 201):
        q9[k] = q9[k - 1] - 150.0 * 0.97 ** (k - 1)
    out.append(('decaying_creep', q9, {k: True for k in q9}, {k: 0.0 for k in q9}, {'cap': 200}))
    q11 = {k: 1e9 + 60.0 * (-1) ** k for k in range(1, 121)}
    out.append(('jitter_pm60', q11, {k: True for k in q11}, {k: 0.0 for k in q11}, {'cap': 120}))
    q11a = {k: 1e9 + 20.0 * (-1) ** k for k in range(1, 121)}
    out.append(('jitter_pm20', q11a, {k: True for k in q11a}, {k: 0.0 for k in q11a}, {'cap': 120}))
    k6 = SC2.replay(q6, b6, {k: 0.0 for k in q6}, R.P_MAX, cap=200)[1]['k_star']
    out.append(('gap_refusal_then_certify', q6, b6, {k: (3000.0 if k <= k6 + 5 else -100.0) for k in q6}, {'cap': 200}))
    out.append(('gap_never_closes', q6, b6, {k: 3000.0 for k in q6}, {'cap': 200}))
    qc = K118.cstar_188_287()
    q14 = {k - 187: v for k, v in qc.items()}
    out.append(('c_star_188_287', q14, {k: True for k in q14}, {k: 0.0 for k in q14}, {'cap': 100}))
    q15 = {1: 1e9}
    for k in range(2, 201):
        q15[k] = q15[k - 1] - 3000.0 * 0.7 ** (k - 1)
    out.append(('fast_geometric', q15, {k: True for k in q15}, {k: 0.0 for k in q15}, {'cap': 200}))
    q16 = {k: 1e9 - 80.0 * (k - 1) for k in range(1, 121)}
    out.append(('uncertified_form', q16, {k: True for k in q16}, {k: 10.0 * k for k in q16}, {'cap': 120}))
    q17 = {k: 1e9 - 80.0 * k for k in range(1, 400)}
    out.append(('dynamic_cap_first_pass_50', q17, {k: k >= 50 for k in q17}, {k: 0.0 for k in q17},
                {'cap_after_first_k0': R.CAP_AFTER_K0}))
    out.append(('dynamic_cap_no_pass', q17, {k: False for k in q17}, {}, {'cap_after_first_k0': R.CAP_AFTER_K0}))
    out.append(('dynamic_cap_first_pass_250', q17, {k: k >= 250 for k in q17}, {k: 0.0 for k in q17},
                {'cap_after_first_k0': R.CAP_AFTER_K0}))
    return out, q6, b6, k6


def tests_V():
    res = {}
    v2 = K118.tests_V()
    res['V_v2_suite_as_committed'] = {'ok': v2['holds'] is True,
                                      'tests': {k: v.get('ok') for k, v in v2['tests'].items()}}
    trajs, q6, b6, k6 = _trajectories()
    eq = {}
    for name, q, b, t, kw in trajs:
        (o2, d2, _r2), (o3, d3, r3) = _both(q, b, t, kw)
        recs_equal = [_jt(_v2_view(a)) for a in o3] == [_jt(_v2_view(x)) for x in o2]
        dec_equal = _jt(_dec_view(d3)) == _jt(_dec_view(d2))
        eq[name] = {'ok': bool(recs_equal and dec_equal and (d3 or {}).get('version') == 3
                               and not r3.non_optimal_cycles),
                    'records_equal': recs_equal, 'decision_equal': dec_equal,
                    'status': (d3 or {}).get('status'), 'k_star': (d3 or {}).get('k_star')}
    res['V_eq_all_optimal_reproduces_version_2'] = {'ok': all(v['ok'] for v in eq.values()), 'trajectories': eq}
    # V18: a non-Optimal lapse (alpha) at k0 + 25 of the damped cosine
    t0 = {k: 0.0 for k in q6}
    ao = {k: k != 36 for k in q6}
    o18, d18, r18 = SC3.replay(q6, b6, t0, ao, R.P_MAX, cap=200, cap_ceiling=300)
    _o7, d7, _r7 = SC2.replay(q6, {k: (b6[k] and k != 36) for k in q6}, t0, R.P_MAX, cap=200)
    rec36 = o18[35]
    res['V18_non_optimal_lapse'] = {
        'ok': bool(d18['status'] == 'certified' and d18['k_star'] > k6 and d18['k0'] == 37 and r18.n == 11
                   and d18['N'] == 11 and d18['first_k0_alpha'] == 11 and len(r18.lapses) == 1
                   and r18.lapses[0]['cycle'] == 36 and r18.lapses[0]['cause'] == ['non_optimal']
                   and rec36['lapse'] is True and rec36['boyd_k_v2'] is True and rec36['all_optimal_k'] is False
                   and d18['non_optimal_cycles'] == [36] and d18['k_star'] == d7['k_star']
                   and d18['version'] == 3 and d18['reading'] == 'alpha'),
        'k_star': d18['k_star'], 'k_star_all_optimal': k6, 'k_star_v2_with_boyd_lapse_at_36': d7['k_star'],
        'k0': d18['k0'], 'N': d18['N'], 'first_k0_alpha': d18['first_k0_alpha'], 'lapses': r18.lapses}
    # V19: the gamma veto on the same trajectory (report-only; alpha unaffected)
    g = d18['gamma_report_only']
    gd = r18.gamma_decision or {}
    veto_windows_contain_36 = all((v['window_a'][0] <= 36 <= v['window_a'][1]) or
                                  (v['window_b'][0] <= 36 <= v['window_b'][1]) for v in r18.gamma.vetoes)
    # gamma running alone to its own decision (alpha stops the shadow at alpha's k*)
    gs = SC3.GammaShadow(R.P_MAX, cap=200, cap_ceiling=300)
    for k in range(1, 201):
        gs.observe(k, q6[k], b6[k], 0.0, ao[k])
        if gs.decision is not None:
            break
    gdec = gs.decision or {}
    res['V19_gamma_veto'] = {
        'ok': bool(r18.gamma.vetoes and veto_windows_contain_36 and gs.vetoes and gdec.get('status') == 'certified'
                   and gdec.get('k_star') > k6 and not (gdec['window'][0] <= 36 <= gdec['window'][1])
                   and gs.lapses == [] and gdec.get('k0') == 11
                   and all(v['cycle'] >= k6 for v in gs.vetoes) and gs.vetoes[0]['cycle'] == k6
                   and d18['status'] == 'certified'),
        'gamma_alone': {'status': gdec.get('status'), 'k_star': gdec.get('k_star'), 'window': gdec.get('window'),
                        'k0_no_reset': gdec.get('k0'), 'n_vetoes': len(gs.vetoes),
                        'first_veto': gs.vetoes[0] if gs.vetoes else None},
        'gamma_as_recorded_in_the_alpha_decision': {k: g.get(k) for k in ('status', 'k_star', 'n_vetoes')},
        'gamma_decision_in_rule': {k: gd.get(k) for k in ('status', 'k_star')}}
    # V20: non-Optimal AT the v2 first pass (dynamic cap): N and the cap keyed on v2; first_k0_alpha later
    q20 = {k: 1e9 - 80.0 * k for k in range(1, 400)}
    b20 = {k: k >= 50 for k in q20}
    ao20 = {k: k != 50 for k in q20}
    _o20, d20, r20 = SC3.replay(q20, b20, {k: 0.0 for k in q20}, ao20, R.P_MAX, cap_after_first_k0=R.CAP_AFTER_K0,
                                cap_ceiling=300)
    _o20w, d20w, _ = SC3.replay(q20, {k: k >= 51 for k in q20}, {k: 0.0 for k in q20}, {k: True for k in q20},
                                R.P_MAX, cap_after_first_k0=R.CAP_AFTER_K0, cap_ceiling=300)
    res['V20_non_optimal_at_the_v2_first_pass'] = {
        'ok': bool(r20.n == 50 and d20['k_cap'] == 159 and r20.first_k0_alpha == 51 and d20['N'] == 50
                   and d20w['k_cap'] == 160 and not r20.lapses),
        'N': r20.n, 'first_k0_alpha': r20.first_k0_alpha, 'cap': d20['k_cap'],
        'cap_if_keyed_on_the_alpha_pass_would_be': d20w['k_cap']}
    # V21: the per-cell ceiling
    try:
        r437 = SC3.SettlingRuleV3(R.P_MAX, cap=437, cap_ceiling=437)
        ok437 = r437.cap == 437
    except Exception:  # noqa: BLE001
        ok437 = False
    refused = {}
    for name, kw in (('cap_437_ceiling_300', {'cap': 437, 'cap_ceiling': 300}), ('no_ceiling', {'cap': 200}),
                     ('no_ceiling_dynamic', {'cap_after_first_k0': 109})):
        try:
            SC3.SettlingRuleV3(R.P_MAX, **kw)
            refused[name] = False
        except ValueError:
            refused[name] = True
    q21 = {k: 1e9 + 30000.0 * 0.88 ** (k - 330) * math.cos(2 * math.pi * (k - 330) / 20.0) if k >= 330 else 1e9 + 5e4
           for k in range(1, 438)}
    _o21, d21, _ = SC3.replay(q21, {k: k >= 330 for k in q21}, {k: 0.0 for k in q21}, {k: True for k in q21},
                              R.P_MAX, cap=437, cap_ceiling=437)
    res['V21_per_cell_ceiling'] = {'ok': bool(ok437 and all(refused.values()) and d21['status'] == 'certified'
                                              and d21['k_star'] > 330 and d21['cap_ceiling'] == 437),
                                   'cap_437_ceiling_437_accepted': ok437, 'refused': refused,
                                   'certifies_beyond_300': d21.get('k_star')}
    # V22: all_optimal must be a bool
    bad = {}
    for v in (None, 1, 'True'):
        try:
            SC3.SettlingRuleV3(R.P_MAX, cap=10, cap_ceiling=300).observe(1, 1.0, True, 0.0, v)
            bad[repr(v)] = False
        except TypeError:
            bad[repr(v)] = True
    res['V22_all_optimal_is_a_bool'] = {'ok': all(bad.values()), 'refused': bad}
    # V0: versions 1 and 2 byte-identical
    now1, now2 = _sha('settling_criterion.py'), _sha('settling_criterion_v2.py')
    pins1 = {k: (json.load(open(_abs(p)))['code_sha256'] or {}).get('settling_criterion.py')
             for k, p in K118.V1_PINS.items()}
    w118 = json.load(open(_abs(W118_SPEC['path'])))
    pin2 = w118['code_sha256'].get('settling_criterion_v2.py')
    res['V0_versions_1_and_2_byte_identical'] = {
        'ok': bool(all(v == now1 for v in pins1.values()) and now2 == pin2 and _sha(W118_SPEC['path']).startswith(
            W118_SPEC['sha256']) and _git_tracked_clean('settling_criterion.py')
            and _git_tracked_clean('settling_criterion_v2.py')),
        'settling_criterion_py': {'now': now1, 'pins_v39_v41': pins1},
        'settling_criterion_v2_py': {'now': now2, 'pin_fc791891': pin2}}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  R -- validation by replay against W131
# ======================================================================================================================
def _verify_w118(eval_rel, names):
    root = eval_rel.split(os.sep + 'evals' + os.sep)[0]
    man = json.load(open(_abs(os.path.join(root, 'campaign_manifest_sha256.json'))))
    out = {}
    for n in names:
        rel = os.path.join(eval_rel, n)
        now = _sha(rel)
        if man.get(rel) != now:
            raise RuntimeError(f'{rel}: sha256 {now} != manifest {man.get(rel)}')
        out[rel] = now
    return out


def _w131_summary(s):
    return {k: s.get(k) for k in ('status', 'k_star', 'k0', 'k_cap')}


def _gamma_alone(q, b, t, ao, n, kw):
    """Reading gamma run on its own over the recorded cycles 1..n (as W131 ran it; inside the rule the shadow stops
    when alpha decides)."""
    gs = SC3.GammaShadow(R.P_MAX, **kw)
    for k in range(1, n + 1):
        if k > gs.effective_cap():
            break
        gs.observe(k, q.get(k), bool(b.get(k, False)), t.get(k), ao[k])
        if gs.decision is not None:
            break
    return gs.decision


def tests_R():
    doc, sha = _pinned_json(W131)
    t1 = doc['task1']
    res = {'w131': {'path': W131['path'], 'sha256': sha, 'commit': W131['commit']}}
    cells = {}
    for cell, sub in W118_CELL_DIRS.items():
        eval_rel = os.path.join(W118_ROOT_REL, sub)
        inputs = _verify_w118(eval_rel, ['per_cycle_record.jsonl', R.CYCLE_FILE, R.DECISION_FILE])
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
        ao = {k: k not in nonopt for k in q}
        cr = w['rule']['cap_rule']
        kw = ({'cap': cr['cap'], 'cap_ceiling': SC2.CAP_CEILING} if cr['kind'] == 'fixed' else
              {'cap_after_first_k0': cr['after_first_k0'], 'cap_ceiling': cr['ceiling']})
        _ob, base, _rb = SC3.replay(q, b, t, {k: True for k in q}, R.P_MAX, last=n, **kw)
        _oa, a, ra = SC3.replay(q, b, t, ao, R.P_MAX, last=n, **kw)
        g = _gamma_alone(q, b, t, ao, n, kw)
        base_repro = all(_jt((base or {}).get(k)) == _jt(dec.get(k)) for k in ('status', 'k_star', 'branch', 'k0',
                                                                              'window', 'k_cap'))
        wa, wg = w['v3_alpha'], w['gamma']
        a_s = {'status': (a or {}).get('status') or f'not certified within the recorded cycles 1..{n}',
               'k_star': (a or {}).get('k_star'), 'k0': (a or {}).get('k0') if a else None,
               'k_cap': (a or {}).get('k_cap')}
        g_s = {'status': (g or {}).get('status') or f'not certified within the recorded cycles 1..{n}',
               'k_star': (g or {}).get('k_star'), 'k_cap': (g or {}).get('k_cap')}
        alpha_equal = (a_s['status'] == wa['status'] and a_s['k_star'] == wa['k_star'] and a_s['k0'] == wa.get('k0')
                       and a_s['k_cap'] == wa.get('k_cap'))
        gamma_equal = g_s['status'] == wg['status'] and g_s['k_star'] == wg['k_star'] and g_s['k_cap'] == wg.get('k_cap')
        holds = (a_s['status'] == dec['status'] and a_s['k_star'] == dec.get('k_star')
                 and g_s['status'] == dec['status'] and g_s['k_star'] == dec.get('k_star'))
        cells[cell] = {'ok': bool(base_repro and alpha_equal and gamma_equal), 'holds_under_v3': holds,
                       'committed': {k: dec.get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'N')},
                       'non_optimal_cycles_w131': sorted(nonopt), 'base_all_optimal_reproduces_committed': base_repro,
                       'v3_alpha': {**a_s, 'N_v2_first_pass': ra.n, 'first_k0_alpha': ra.first_k0_alpha},
                       'w131_alpha': _w131_summary(wa), 'alpha_equal_w131': alpha_equal,
                       'v3_gamma': g_s, 'w131_gamma': _w131_summary(wg), 'gamma_equal_w131': gamma_equal,
                       'inputs_sha256': inputs}
    for name, (w112_cell, spec_rel) in REF_CELLS.items():
        q, b, t, val, rows = K118.record_series(w112_cell)
        w = t1[name]
        n = len(rows)
        nonopt = set(w['counts']['non_optimal_cycles'])
        ao = {k: k not in nonopt for k in q}
        cap = w['rule']['cap']
        kw = {'cap': cap, 'cap_ceiling': max(cap, SC2.CAP_CEILING)}
        _oa, a, ra = SC3.replay(q, b, t, ao, R.P_MAX, last=n, **kw)
        g = _gamma_alone(q, b, t, ao, n, kw)
        wa, wg = w['v3_alpha'], w['gamma']
        a_s = {'status': (a or {}).get('status') or f'not certified within the recorded cycles 1..{n}',
               'k_star': (a or {}).get('k_star')}
        g_s = {'status': (g or {}).get('status') or f'not certified within the recorded cycles 1..{n}',
               'k_star': (g or {}).get('k_star')}
        same = (a_s['status'] == wa['status'] and a_s['k_star'] == wa['k_star'] and g_s['status'] == wg['status']
                and g_s['k_star'] == wg['k_star'])
        cells[name] = {'ok': bool(same and val['pass']), 'holds_under_v3': a_s['status'] == 'certified' and
                       a_s['k_star'] == w['committed_decision']['k_star'] and g_s['k_star'] == a_s['k_star'],
                       'note': ('W131 replayed the references under their own rule (settling_criterion v1, N 132 / '
                                '112, P_MAX 22); version 3 here (P_MAX 30, cap the W101 spec cap): status and k* '
                                'compared; k0 recorded beside'),
                       'committed': w['committed_decision'], 'non_optimal_cycles_w131': sorted(nonopt),
                       'v3_alpha': {**a_s, 'k0': (a or {}).get('k0'), 'N_v2_first_pass': ra.n},
                       'w131_alpha': _w131_summary(wa), 'v3_gamma': g_s, 'w131_gamma': _w131_summary(wg),
                       'terminal_t_sum_validation': val}
    n_hold = sum(1 for v in cells.values() if v['holds_under_v3'])
    not_cert = sorted(c for c, v in cells.items() if v['v3_alpha']['status'].startswith('not certified'))
    res['cells'] = cells
    res['summary'] = {'n_cells': len(cells), 'n_hold_under_alpha_and_gamma': n_hold,
                      'not_certified_within_records': not_cert, 'expected': EXPECTED_V3_REPLAY}
    res['holds'] = bool(all(v['ok'] for v in cells.values()) and n_hold == 11
                        and not_cert == EXPECTED_V3_REPLAY['not_certified_within_records'])
    return res


# ======================================================================================================================
#  O -- the 38 original cells, their configuration identity, the inputs now
# ======================================================================================================================
def spec_entry(cell):
    c = V.CELLS[cell]
    spec = json.load(open(_abs(os.path.join(c['orig_root'], c['orig_spec']))))
    hits = [e for e in spec['candidates'] if e.get('eval_key') == c['orig_eval_key']]
    return spec, (hits[0] if len(hits) == 1 else None)


def configuration_now():
    """The current production configuration (as W118): the Phase B spec's case-file AA and C2 declarations, the tight
    tail declared on."""
    return K118.configuration_for(None)


def i_values(cell, claims):
    """I of the cell from every W117 claim that names it (as 'ref' or 'other'); unique or None."""
    key = V.CELLS[cell]['orig_eval_key']
    vals = set()
    ids = []
    for c in claims:
        if (c.get('other') or {}).get('eval_key') == key:
            vals.add(c['I_other'])
            ids.append(c['claim_id'])
        if (c.get('ref') or {}).get('eval_key') == key:
            vals.add(c['I_ref'])
            ids.append(c['claim_id'])
    distinct = sorted(vals)
    return (distinct[0] if len(distinct) == 1 else None), {'claims': ids, 'distinct': distinct,
                                                           'note': ('each claim is scored with its own I (W117); a '
                                                                    'cell cited from two sources may carry values '
                                                                    'that differ in the last bits')}


def tests_O():
    res = {}
    cells = {}
    w117, w117_sha = _pinned_json(W117)
    claims = w117['claims']
    now = configuration_now()
    case_now = H.sha256_file(H.CASE_FILE)
    ess_now = _sha(K118.ESS_PARAMS_REL)
    ess = json.load(open(_abs(K118.ESS_PARAMS_REL)))
    for cell in V.CELL_ORDER:
        c = V.CELLS[cell]
        spec, entry = spec_entry(cell)
        man = json.load(open(_abs(os.path.join(c['orig_root'], 'campaign_manifest_sha256.json'))))
        rel = V.reference_path(cell)
        sha = _sha(rel)
        rows = _read_jsonl(_abs(rel))
        passes = [r['cycle'] for r in rows if r['boyd_all_pass'] and r['local_solves_ok']]
        n_old = rows[-1]['cycle']
        lapses = [k for k in range(passes[0], n_old + 1) if k not in passes] if passes else None
        cfg = spec['configuration']
        base_orig = H.evaluation_key(entry['key'], entry['overrides'],
                                     case_file_aa=cfg.get('case_file_anderson_acceleration'),
                                     ess_ageing_baseline=cfg.get('ess_ageing_baseline'),
                                     flex_price_multiplier=entry.get('flex_price_multiplier'),
                                     convergence_depth_tail=cfg.get('convergence_depth_tail')) if entry else None
        rec = json.load(open(_abs(os.path.join(V.original_eval_dir(cell), 'evaluation_record.json'))))
        i_val, i_src = i_values(cell, claims)
        ep = {n: float(e) / float(p) for n, (p, e) in entry['canonical']['nodes'].items() if float(p) > 0} if entry else {}
        lat_full, _ = K118.lattice_check(entry['canonical'], i_val) if entry else ({}, {})
        ident = {k: cfg.get(k) == (now[k] if k in now else _w101_ref_cfg().get(k)) for k in IDENTITY_KEYS}
        ident['case_file_sha256'] = cfg.get('case_file_sha256') == case_now == CASE_FILE_SHA256
        if c['gated']:
            ident.update({k: cfg.get(k) == now[k] for k in GATED_IDENTITY_KEYS})
            ident['ess_params_file_at_run_is_C2_now'] = ((cfg.get('ess_params_file') or {}).get('sha256')
                                                         == ESS_PARAMS_SHA256_C2 == ess_now)
        else:
            ident['ungated_declared_difference_no_c2_declaration_at_run'] = (
                cfg.get('ess_ageing_baseline_label') is None and cfg.get('ess_params_file') is None)
        ident['tail_absent_at_run_the_one_declared_change'] = cfg.get('convergence_depth_tail') is None
        ident['entry_overrides_empty'] = (entry or {}).get('overrides') == {}
        ident['entry_no_model_variant_derived_instance_or_premium'] = not any(
            k in (entry or {}) for k in ('model_variant', 'derived_instance', 'interface_deviation_premium'))
        ident['no_derived_instance_in_configuration'] = cfg.get('derived_instance') is None
        parts = {
            'original_spec_entry_found': entry is not None,
            'entry_eval_dir_and_label_as_pinned': bool(entry) and entry['eval_dir'] == c['orig_eval_dir']
            and entry['label'] == c['orig_label'],
            'per_cycle_record_sha256_as_pinned': sha == c['per_cycle_record_sha256'],
            'per_cycle_record_in_campaign_manifest': man.get(rel) == sha,
            'per_cycle_record_committed_clean': _git_tracked_clean(rel),
            'N_old_as_pinned': n_old == c['N_old'],
            'first_residual_pass_as_pinned': bool(passes) and passes[0] == c['k0'],
            'lapses_after_k0_as_pinned': lapses == c['original_lapses_after_k0'],
            'N_old_passes': bool(passes) and passes[-1] == n_old,
            'base_key_original_configuration_equals_original_eval_key': base_orig == c['orig_eval_key'],
            'original_record_certified_at_N_old': rec.get('status') == 'certified' and rec.get('cycles_run') == n_old,
            'flex_multiplier_as_pinned': (entry or {}).get('flex_price_multiplier') == c['flex_price_multiplier'],
            'I_in_w117_claims_consistent_to_1e-6': bool(i_src['distinct'])
            and max(i_src['distinct']) - min(i_src['distinct']) <= 1e-6,
            'lattice_e_over_p_in_2_4_every_storage_node': bool(entry) and all(
                LATTICE_EP['min_e_over_p'] <= v <= LATTICE_EP['max_e_over_p'] for v in ep.values()),
            'cap_within_ceiling': V.spec_cap(cell) <= c['cap_ceiling'],
            **{f'identity:{k}': bool(v) for k, v in ident.items()},
        }
        cells[cell] = {'ok': all(parts.values()), 'parts': parts, 'item': c['item'], 'gated': c['gated'],
                       'k0': c['k0'], 'N_old': n_old, 'original_lapses_after_k0': lapses,
                       'Q_N_old': rows[-1]['gross_operational_cost'],
                       'canonical': (entry or {}).get('canonical'), 'candidate_key': (entry or {}).get('key'),
                       'investment_year': ((entry or {}).get('canonical') or {}).get('investment_year'),
                       'e_over_p': ep, 'lattice_other_rules_reported': lat_full, 'I': i_val, 'I_source': {
                           'w117': W117['path'], 'sha256': w117_sha, **i_src},
                       'cap_rule': V.cap_rule(cell), 'spec_cap': V.spec_cap(cell),
                       'dead_zone_candidate': cell in V.DEAD_ZONE_CANDIDATES,
                       'dead_zone_borderline': cell in V.DEAD_ZONE_BORDERLINE,
                       'original': {'campaign_id': c['orig_campaign_id'],
                                    'spec': os.path.join(c['orig_root'], c['orig_spec']),
                                    'spec_sha256': _sha(os.path.join(c['orig_root'], c['orig_spec'])),
                                    'eval_dir': V.original_eval_dir(cell), 'eval_key': c['orig_eval_key'],
                                    'per_cycle_record': rel, 'per_cycle_record_sha256': sha,
                                    'git_head': spec.get('git_head'), 'harness_sha256': spec['harness']['sha256'],
                                    'configuration': {k: cfg.get(k) for k in IDENTITY_KEYS + GATED_IDENTITY_KEYS
                                                      + ('convergence_depth_tail',)},
                                    'ess_params_file_at_run': cfg.get('ess_params_file'),
                                    'flex_price_multiplier': c['flex_price_multiplier']},
                       'identity_vs_original': ident,
                       't_sum_terminal_original': json.load(open(_abs(os.path.join(
                           V.original_eval_dir(cell), 'interface_settlement_detail_s31c.json'))))[
                           't_tso_plus_t_dso_terminal']}
    res['cells'] = cells
    res['inputs_now'] = {'case_file': {'path': H.CASE_FILE_REL, 'sha256': case_now},
                         'ess_params_file': {'path': K118.ESS_PARAMS_REL, 'sha256': ess_now,
                                             'max_energy_to_power_factor': ess['max_energy_to_power_factor'],
                                             'min_energy_to_power_factor': ess['min_energy_to_power_factor'],
                                             'minimum_soh': ess['ageing']['minimum_soh']},
                         'cost_file': {'path': K118.COST_FILE_REL, 'sha256': _sha(K118.COST_FILE_REL)},
                         'configuration_now': now}
    res['lattice_rule'] = {**LATTICE_EP, 'scope': ('Planner ruling (TASKS.md Addendum 58): the lattice check is SCOPED '
                                                   'TO E/P for ladder points; no substitution. Applied to every cell; '
                                                   'the other W118 lattice rules reported, not gating')}
    res['parts'] = {'every_cell_ok': all(v['ok'] for v in cells.values()), 'n_cells_38': len(cells) == 38,
                    'case_file_now_as_pinned': case_now == CASE_FILE_SHA256,
                    'ess_params_now_C2': ess_now == ESS_PARAMS_SHA256_C2 and _git_tracked_clean(K118.ESS_PARAMS_REL),
                    'cost_file_as_pinned': _sha(K118.COST_FILE_REL) == K118.COST_FILE_SHA256}
    res['holds'] = all(res['parts'].values())
    return res


def _w101_ref_cfg():
    return json.load(open(_abs(REF_CELLS['x0'][1])))['configuration']


# ======================================================================================================================
#  S -- production since the originals
# ======================================================================================================================
def production_since_originals(used_files=()):
    """REPORT: production .py changes between each original run's recorded git head and HEAD. GATE: no uncommitted
    change to any file the run uses (`used_files` plus the production files and the harness's clean list)."""
    heads = {}
    for cell in V.CELL_ORDER:
        spec = spec_entry(cell)[0]
        heads.setdefault(spec.get('git_head'), []).append(cell)
    per = {}
    for head, cells in heads.items():
        diff = H._git(['diff', '--name-status', head, 'HEAD', '--', *PRODUCTION_FILES]).splitlines() if head else []
        log = H._git(['log', '--format=%h %s', f'{head}..HEAD', '--', *PRODUCTION_FILES]).splitlines() if head else []
        per[head] = {'cells': cells, 'production_changed_since': diff, 'n_commits_touching_production': len(log),
                     'commits_touching_production_first20': log[:20]}
    dirty = H._git(['status', '--porcelain', '--untracked-files=no', '--', '*.py']).splitlines()
    used = set(used_files) | set(PRODUCTION_FILES) | set(H.PRODUCTION_FILES_TO_CHECK_CLEAN)
    dirty_used = [d for d in dirty if d.split()[-1] in used]
    return {'per_original_git_head': per, 'uncommitted_tracked_py': dirty, 'uncommitted_files_this_run_uses': dirty_used,
            'head': H._git(['rev-parse', 'HEAD']),
            'note': ('production changed since the pre-tail originals (the tail W84-W86, later writer-only / keyed '
                     'changes); the in-cycle bitwise gate through k0 is the evidence for each gated cell (W86 / W118 '
                     'showed pre-tail runs replay bitwise to the first tail cycle; W123-W129 held it on 10 cells)'),
            'ok': not dirty_used}


def tests_S():
    r = production_since_originals(CODE_PINNED_BY_CHECKS)
    return {'holds': r['ok'], **r}


# ======================================================================================================================
#  H -- the hooks through the real wrappers
# ======================================================================================================================
_OPT = 'Ipopt 3.14.18\\x3a Optimal Solution Found'
_ACC = 'Ipopt 3.14.18\\x3a Solved To Acceptable Level.'


def fake_results(pp, non_optimal=()):
    """Stand-in `results` in production's shape; every block Optimal except the named keys (Acceptable)."""
    def one(key):
        return SimpleNamespace(solver=SimpleNamespace(message=_ACC if key in non_optimal else _OPT))
    out = {'tso': {}, 'dso': {}, 'esso': {}}
    for y in pp.years:
        for d in pp.days:
            out['tso'].setdefault(y, {})[d] = one(f'TSO|{y}|{d}')
            for n in pp.active_distribution_network_nodes:
                out['dso'].setdefault(n, {}).setdefault(y, {})[d] = one(f'DSO|{n}|{y}|{d}')
    for n in pp.active_distribution_network_nodes:
        out['esso'][n] = one(f'ESSO|{n}')
    return out


def drive(cell, variant='certify', first_pass_at=50, nonopt=None):
    """The REAL nine wrappers (V.make_wrappers) driven in production's call order (apply, Boyd, local-solve check, AA,
    recourse, tail next-state, penalties, EFC). A gated cell: its ORIGINAL values through N_old, then a synthetic
    continuation; an ungated cell: a synthetic run (Boyd from `first_pass_at`). Variants: certify, ulp_at_40, gap, creep.
    `nonopt` = {cycle: [block keys]} injects Acceptable exits."""
    decl = V.declaration_for(cell)
    ref = V.load_replay_reference(decl)
    cap = V.spec_cap(cell)
    sink = []
    st = V.ResettleStateV3(decl, None, cap, reference=ref, sink=sink)
    scripts = {'boyd': [], 'aa': [], 'next': [], 'pen': [], 'rc': [], 'efc': []}
    orig, calls = K105._standins(None, scripts)
    local_script = []
    orig['_admm_local_solves_succeeded'] = lambda pp_, results: local_script.pop(0)
    blocks_by_cycle = {}
    gated = decl['replay_reference'] is not None
    c_info = V.CELLS[cell]
    g_rows = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(V.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if gated else {})
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
    if cr['kind'] == 'fixed':
        return SC3.SettlingRuleV3(V.P_MAX, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return SC3.SettlingRuleV3(V.P_MAX, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


def pure_equal(decl, lines):
    """The pure v3 rule over the cycle lines (Q, the v2 boyd_k, t_sum, all_optimal_k) reproduces every in-cycle rule
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
    return {'raised': d['raised'], 'lines': len(lines), 'creep_lines': len(files.get(V.CREEP_FILE, [])),
            'decision_files': len(files.get(V.DECISION_FILE, [])), 'summary_ok': summ['ok'],
            'stopped_by': summ['stopped_by'], 'status': summ['settling_status'], 'k_star': summ['k_star'],
            'branch': summ['branch'], 'first_pass': fp, 'first_k0_v2': summ.get('first_k0_v2'),
            'first_k0_alpha': summ.get('first_k0_alpha'), 'non_optimal_cycles': summ.get('non_optimal_cycles'),
            'replay_bitwise_through': summ['replay_bitwise_through_cycle'],
            'first_divergence': summ['replay_first_divergence'], 'n_overlap': len(summ['overlap_k0_plus_1_to_N_old']),
            'overlap_all_zero': all(o['Q_new_minus_Q_old'] == 0.0 for o in summ['overlap_k0_plus_1_to_N_old']),
            'gap_refusals': len(summ['gap_refusals']), 'n_lapses': len(summ['lapse_events']),
            'lapse_causes': [e.get('cause') for e in summ['lapse_events']],
            'certificate_length_after': d['admm'].minimum_consecutive_converged_cycles, 'ended_by': st.ended_by,
            'in_cycle_rule_equals_pure_replay': pure_ok, 'pure_decision_status': (pure_dec or {}).get('status'),
            'exit_capture_complete': summ.get('exit_capture_complete'),
            'exits_51_every_line': all((x.get('ipopt_exit_counts') or {}).get('total') == 51
                                       and len(x.get('ipopt_exit_by_block') or {}) == 51 for x in lines),
            'all_optimal_every_line_is_bool': all(isinstance(x.get('all_optimal_k'), bool) for x in lines),
            't_sum_every_line': all(isinstance(x.get('t_sum'), float) for x in lines),
            'holds_through_first_pass': holds_pre, 'holds_after_first_pass': holds_post,
            'capture_errors': summ['capture_errors'][:3], 'errors': summ['errors'][:3],
            'last_cycle': st.cycle, 'rule_cap': st.rule.cap, 'summary': summ}


HOLDS_OFF = K118.HOLDS_OFF
HOLDS_ON = K118.HOLDS_ON


def _h_real_v3(srp, which):
    """K105's real-production hold tests on the v3 state and the nine v3 wrappers (first residual pass preset at W105's
    hold cycle, so the holds engage from there exactly as there)."""
    real_state, real_make, real_wrapped, real_n = K105._state, K105.E.make_wrappers, K105.E.WRAPPED, K105.E.N_HOLD
    cell = V.CELL_ORDER[0]

    def state():
        st = V.ResettleStateV3(V.declaration_for(cell), None, V.spec_cap(cell), reference={}, sink=[])
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


def _h_layering(srp):
    before = {name: getattr(srp, name) for name in V.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w132_layering_')
    cell = 'b_2a0ba8b2'
    n = V.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with V.settling_resettle_hooks(scratch, V.declaration_for(cell), holder, cap=V.spec_cap(cell)) as st:
            installed = {name: getattr(srp, name) for name in V.WRAPPED}
            all_installed = all(installed[k] is not before[k] for k in V.WRAPPED)
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
                            {'cycle': c, 'action': V.R.AA_OFF_ACTION if (conv or c > n) else 'accepted'})
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
    dispatch_ok = (H.resettle_hooks_module(V.declaration_for(cell)) is V
                   and H.resettle_hooks_module(R.declaration_for('f2_challenger')) is R
                   and H.validate_settling_resettle(V.declaration_for(cell)) == V.declaration_for(cell)
                   and H.validate_settling_resettle(R.declaration_for('f2_challenger')) == R.declaration_for('f2_challenger'))
    summ = holder.get(V.SUMMARY_KEY) or {}
    return {'holds': bool(appender_saw_held and restored and order_ok and pre_ok and idc_outer and all_installed
                          and dispatch_ok and summ.get('phase') == 'ended'
                          and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                          and admm.minimum_consecutive_converged_cycles == 10),
            'nine_wrappers_installed': all_installed, 'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'child_real_order_continuation_settling_extension_resettle_idc_s38_appender': order_ok,
            'child_real_dispatch_then_checklist_before_run_admm_arm': pre_ok,
            'harness_dispatch_v3_to_w132_w118_to_w118': dispatch_ok, 'files_written_by_install_without_cycles':
            files_written, 'summary_phase': summ.get('phase'), 'summary_errors': summ.get('errors')}


def _h_exit_real():
    """The exit wrapper over REAL production `_admm_local_solves_succeeded` with real pyomo SolverResults on a fresh
    SRP1 planning (data only; nothing built or solved)."""
    import p56a_oracle as O
    import shared_resources_planning as srp
    from pyomo.opt import SolverResults, SolverStatus, TerminationCondition
    eval_id = f'p515s53_w132_exit_check_{int(time.time())}'
    work = os.path.join(O.WORK_DIR, eval_id)

    def result(msg, status=SolverStatus.ok, term=TerminationCondition.optimal):
        r = SolverResults()
        r.solver.status = status
        r.solver.termination_condition = term
        r.solver.message = msg
        return r
    try:
        planning = O.fresh_planning(eval_id)
        keys = V.block_keys(planning)

        def results_with(overrides=None):
            out = {'tso': {}, 'dso': {}, 'esso': {}}
            for key in keys:
                kind, n, y, d = key
                r = (overrides or {}).get(V._key_text(key), result(_OPT))
                if kind == 'TSO':
                    out['tso'].setdefault(y, {})[d] = r
                elif kind == 'DSO':
                    out['dso'].setdefault(n, {}).setdefault(y, {})[d] = r
                else:
                    out['esso'][n] = r
            return out
        cell = V.CELL_ORDER[0]
        cases = {}
        for name, overrides, want_ok, want_all_opt in (
                ('all_optimal', None, True, True),
                ('esso_7_acceptable', {'ESSO|7': result(_ACC)}, True, False),
                ('tso_acceptable', {f'TSO|{list(planning.years)[1]}|{list(planning.days)[2]}': result(_ACC)}, True,
                 False),
                ('esso_9_failed', {'ESSO|9': result('Ipopt 3.14.18\\x3a Maximum Number of Iterations Exceeded.',
                                                    SolverStatus.warning, TerminationCondition.maxIterations)},
                 False, False)):
            st = V.ResettleStateV3(V.declaration_for(cell), None, V.spec_cap(cell), reference={}, sink=[])
            w = V.make_exit_wrapper(st, srp._admm_local_solves_succeeded, H.ipopt_exit_class, 'real')
            res_ = results_with(overrides)
            ok_prod = srp._admm_local_solves_succeeded(planning, res_)
            w(planning, res_)                                    # initialisation call
            st.phase, st.cycle, st.cur = 'in_cycle', 1, {'cycle': 1}
            ok_wrapped = w(planning, res_)
            by = st.cur['ipopt_exit_by_block']
            cases[name] = {'value_unchanged': ok_wrapped == ok_prod == want_ok,
                           'all_optimal_k': st.cur['all_optimal_k'], 'all_optimal_as_expected':
                           st.cur['all_optimal_k'] is want_all_opt, 'counts': st.cur['ipopt_exit_counts'],
                           'order_is_production': list(by) == [V._key_text(k) for k in keys],
                           'esso_entries': {k: v for k, v in by.items() if k.startswith('ESSO')},
                           'non_optimal_blocks': st.cur['non_optimal_blocks'], 'init_recorded': st.init_exit is not None}
        # a missing ESSO key RAISES
        st = V.ResettleStateV3(V.declaration_for(cell), None, V.spec_cap(cell), reference={}, sink=[])
        w = V.make_exit_wrapper(st, lambda pp, r: True, H.ipopt_exit_class, 'real')
        w(planning, results_with())
        st.phase, st.cycle, st.cur = 'in_cycle', 1, {'cycle': 1}
        broken = results_with()
        del broken['esso'][7]
        try:
            w(planning, broken)
            missing_raises = False
        except KeyError:
            missing_raises = True
    finally:
        if os.path.isdir(work) and not any(files for _r, _d, files in os.walk(work)):
            shutil.rmtree(work)
    ok = (all(c['value_unchanged'] and c['all_optimal_as_expected'] and c['order_is_production']
              and c['counts'] == {'tso': 12, 'dso': 36, 'esso': 3, 'total': 51} and c['init_recorded']
              for c in cases.values())
          and cases['esso_7_acceptable']['non_optimal_blocks'] == ['ESSO|7']
          and cases['esso_7_acceptable']['esso_entries']['ESSO|7']['class'] == 'acceptable'
          and cases['esso_9_failed']['esso_entries']['ESSO|9']['class'] == 'other' and missing_raises)
    return {'holds': bool(ok), 'cases': cases, 'missing_esso_key_raises': missing_raises,
            'n_block_keys': len(keys), 'the_wrapper_sees_the_esso_results': all(
                len(c['esso_entries']) == 3 for c in cases.values())}


def tests_H():
    import shared_resources_planning as srp
    res = {}
    h1 = {}
    for cell in V.GATED_CELLS:
        s = _drive_summary(drive(cell, 'certify'))
        c = V.CELLS[cell]
        s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                       and s['first_k0_v2'] == c['k0'] and s['n_lapses'] == 1 + len(c['original_lapses_after_k0'])
                       and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                       and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                       and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                       and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['exit_capture_complete']
                       and s['exits_51_every_line'] and s['all_optimal_every_line_is_bool']
                       and s['non_optimal_cycles'] == [] and s['certificate_length_after'] == 10
                       and s['lines'] == s['k_star'] == s['creep_lines']
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
    n_old = V.CELLS[cell]['N_old']
    s = _drive_summary(drive(cell, 'certify', nonopt={n_old + 20: ('ESSO|7',)}))
    s3 = {k: v for k, v in s.items() if k != 'summary'}
    s3['ok'] = bool(s['raised'] is None and s['status'] == 'certified' and s['n_lapses'] == 2
                    and s['lapse_causes'][-1] == ['non_optimal'] and s['non_optimal_cycles'] == [n_old + 20]
                    and s['k_star'] > base['k_star'] and s['in_cycle_rule_equals_pure_replay']
                    and s['summary_ok'] and s['first_k0_v2'] == V.CELLS[cell]['k0'])
    s3['k_star_all_optimal'] = base['k_star']
    s3['gamma_report_only'] = s['summary'].get('gamma_report_only', {}).get('status')
    res['H3_non_optimal_esso_exit_is_a_lapse'] = {'holds': s3['ok'], 'detail': s3}
    d4 = drive('c_156ce2d1', 'creep')
    s = _drive_summary(d4)
    rec = {'status': 'certified', 'barrier': False, 'barrier_cause': None, 'certified_cost': 1.0,
           'certification_cycle': s['last_cycle'], 'terminal_gross_operational_cost': 1.0,
           'settling_resettle_summary': s['summary']}
    relabelled = H._apply_settling_resettle_status(dict(rec))
    summ4 = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'cap'
                   and s['last_cycle'] == V.spec_cap('c_156ce2d1') and s['certificate_length_after'] == 10
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
                   and s['last_cycle'] == 50 + V.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 50
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
    res['H7_aa_hold_real_production'] = _h_real_v3(srp, 'aa')
    res['H8_tail_hold_real_production'] = _h_real_v3(srp, 'tail')
    res['H9_rho_hold_real_production'] = _h_real_v3(srp, 'rho')
    res['H10_layering_real_install'] = _h_layering(srp)
    got = V._certificate_length_writes_in_source()
    res['H11_certificate_length_writes'] = {
        'holds': got == sorted(R.CERTIFICATE_LENGTH_WRITES) and 'early_stop' not in inspect.getsource(V.make_exit_wrapper),
        'writes_in_source': got}
    try:
        res['H12_exit_wrapper_on_real_production'] = _h_exit_real()
    except Exception as error:  # noqa: BLE001
        res['H12_exit_wrapper_on_real_production'] = {'holds': False, 'error': f'{type(error).__name__}: {error}',
                                                      'traceback': traceback.format_exc()}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w132():
    import importlib.util
    src = subprocess.run(['git', 'show', f"{PRE_W132_HARNESS['commit']}:p515_s44_campaign_harness.py"], cwd=REPO,
                         capture_output=True, check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != PRE_W132_HARNESS['sha256']:
        raise RuntimeError(f'pre-W132 harness sha256 {sha} != pinned {PRE_W132_HARNESS["sha256"]}')
    tmp = tempfile.mkdtemp(prefix='w132_pre_harness_')
    path = os.path.join(tmp, '_w132_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w132_pre_harness', path)
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


def resettle_kwargs(cell):
    spec, e = spec_entry(cell)
    cfg = configuration_now()
    return spec, e, dict(case_file_aa=cfg['case_file_anderson_acceleration'],
                         ess_ageing_baseline=cfg['ess_ageing_baseline'],
                         flex_price_multiplier=e.get('flex_price_multiplier'),
                         convergence_depth_tail=cfg['convergence_depth_tail'])


def resettle_keys(pre=None):
    out = {}
    for cell in V.CELL_ORDER:
        _spec, e, kw = resettle_kwargs(cell)
        base = H.evaluation_key(e['key'], e['overrides'], **kw)
        decl = V.declaration_for(cell)
        key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=decl, **kw)
        base_pre = pre.evaluation_key(e['key'], e['overrides'], **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[cell] = {'candidate_key': e['key'], 'original_eval_key': e['eval_key'], 'base_key_now': base,
                     'resettle_key': key, 'formula_holds': key == formula, 'base_equals_pre_w132': base == base_pre,
                     'differs_from_original': key != e['eval_key'] and base != e['eval_key']}
    return out


def _key_holders(scanned, key):
    return sorted({rel for rel, spec in scanned for e in spec.get('candidates') or [] if H._entry_eval_key(e) == key})


def _outside_roots(rels, exclude_roots=KEY_EXCLUDED_ROOTS):
    return [r for r in rels if not any(r.startswith(root + os.sep) for root in exclude_roots)]


def tests_K():
    pre, pre_sha = _harness_pre_w132()
    n_specs = n_entries = n_equal = n_own = n_own_ok = n_w118 = n_frozen_equal = 0
    mismatch, errors, own_bad = [], [], []
    scanned = []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(_abs(rel)))
        n_specs += 1
        scanned.append((rel, spec))
        for e in spec.get('candidates') or []:
            n_entries += 1
            try:
                args, kw = _key_args(spec, e)
                rs = e.get('settling_resettle')
                if V.is_v3_declaration(rs):
                    # the campaign's OWN committed specs (the rule: a pre-run check that scans committed artefacts
                    # excludes the run's own) -- accepted only inside the W132 stage root, and only if the frozen eval
                    # key follows the declared formula over the PRE-W132 base key
                    n_own += 1
                    decl = V.validate_settling_resettle(rs)
                    formula = hashlib.sha256(json.dumps({'base_evaluation_key': pre.evaluation_key(*args, **kw),
                                                         'settling_resettle': decl}, sort_keys=True,
                                                        separators=(',', ':')).encode()).hexdigest()
                    now = H.evaluation_key(*args, settling_resettle=decl, **kw)
                    inside = rel.startswith(W132_ROOT_REL + os.sep)
                    if inside and now == formula == H._entry_eval_key(e):
                        n_own_ok += 1
                    else:
                        own_bad.append({'spec': rel, 'label': e.get('label'), 'inside_own_root': inside,
                                        'formula_equals_frozen': formula == H._entry_eval_key(e)})
                    continue
                if rs is not None:
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
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
    keys = resettle_keys(pre)
    holders = {cell: _key_holders(scanned, v['resettle_key']) for cell, v in keys.items()}
    outside = {cell: _outside_roots(h) for cell, h in holders.items()}
    probe = keys[V.CELL_ORDER[0]]['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w132_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    planted_outside_rel = os.path.join(_P53, 'w132_planted_negative_control', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_sibling_rel = os.path.join(W132_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                       'campaign_spec_planted_00000000.json')
    planted_own_rel = os.path.join(W132_ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{V.CELL_ORDER[0]}',
                                   f'campaign_spec_{CAMPAIGN_ID_PREFIX}{V.CELL_ORDER[0]}_00000000.json')
    ctl_outside = _outside_roots(_key_holders(scanned + [planted(planted_outside_rel)], probe))
    ctl_sibling = _outside_roots(_key_holders(scanned + [planted(planted_sibling_rel)], probe))
    own_holders = _key_holders(scanned + [planted(planted_own_rel)], probe)
    ctl_own = _outside_roots(own_holders)
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_entry_equal': n_equal == n_entries - n_own and n_entries > 0,
        'w118_resettle_entries_scanned_and_equal': n_w118 >= 10,
        'own_v3_entries_inside_root_follow_the_formula': n_own_ok == n_own and not own_bad,
        'resettle_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'resettle_base_equals_pre_w132_every_cell': all(v['base_equals_pre_w132'] for v in keys.values()),
        'resettle_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'resettle_keys_distinct': len({v['resettle_key'] for v in keys.values()}) == len(keys) == 38,
        'resettle_keys_absent_from_committed_specs_outside_own_root': not any(outside.values()),
        'control_planted_outside_root_refused': planted_outside_rel in ctl_outside,
        'control_planted_sibling_prefix_refused': planted_sibling_rel in ctl_sibling,
        'control_own_campaign_spec_accepted': planted_own_rel in own_holders and planted_own_rel not in ctl_own,
        'control_v3_declaration_refused_by_the_pre_w132_harness': _pre_refuses_v3(pre),
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w132_harness': {**PRE_W132_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'entries_whose_frozen_eval_key_equals_the_recomputed_REPORTED': n_frozen_equal,
            'w118_resettle_entries': n_w118, 'own_v3_entries': {'n': n_own, 'ok': n_own_ok, 'bad': own_bad[:20]},
            'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'resettle_keys_in_committed_specs_all_REPORTED': holders, 'key_excluded_roots': list(KEY_EXCLUDED_ROOTS),
            'controls': {'planted_outside': {'rel': planted_outside_rel, 'outside_found': ctl_outside},
                         'planted_sibling': {'rel': planted_sibling_rel, 'outside_found': ctl_sibling},
                         'planted_own': {'rel': planted_own_rel, 'holders': own_holders, 'outside_found': ctl_own}}}


def _pre_refuses_v3(pre):
    try:
        pre.validate_settling_resettle(V.declaration_for(V.CELL_ORDER[0]))
        return False
    except ValueError:
        return True


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    out = {}
    ok = True
    for cell in V.CELL_ORDER:
        decl = V.declaration_for(cell)
        cap = V.spec_cap(cell)
        try:
            good = V.assert_resettle_preconditions(decl, {'cap': cap}, {'tail_enabled_for_this_run': True}, True)
            good_ok = all(good.values())
        except Exception as error:  # noqa: BLE001
            good, good_ok = {'error': f'{type(error).__name__}: {error}'}, False
        negatives = {}
        for name, (spec, tail, aa) in {'cap_wrong': ({'cap': cap - 1}, {'tail_enabled_for_this_run': True}, True),
                                       'tail_off': ({'cap': cap}, {'tail_enabled_for_this_run': False}, True),
                                       'aa_off': ({'cap': cap}, {'tail_enabled_for_this_run': True}, False)}.items():
            try:
                V.assert_resettle_preconditions(decl, spec, tail, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {str(error)[:200]}'
        rule = decl['settling_rule']
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'gap_bound_tau': dict(decl, settling_rule=dict(rule, gap_bound=SC3.TAU)),
               'rule_v2': dict(decl, settling_rule=dict(rule, module='settling_criterion_v2', version=2)),
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
                V.validate_settling_resettle(d)
                refused[name] = False
            except ValueError as error:
                refused[name] = str(error)[:160]
        cell_ok = good_ok and all(v.startswith('refused') for v in negatives.values()) and all(refused.values())
        ok = ok and cell_ok
        out[cell] = {'ok': cell_ok, 'n_checklist_items': len(good) if isinstance(good, dict) else None,
                     'checklist_failing': sorted(k for k, v in good.items() if v is not True) if good_ok is False
                     and 'error' not in good else good.get('error') if isinstance(good, dict) else None,
                     'negative_controls': negatives, 'validator_refuses': refused}
    return {'holds': ok, 'cells': out}


# ======================================================================================================================
#  X -- the status-label writer fix
# ======================================================================================================================
W123_RECORD = os.path.join(W118_ROOT_REL, W118_CELL_DIRS['f2_challenger'], 'evaluation_record.json')
W128_RECORD = os.path.join(W118_ROOT_REL, W118_CELL_DIRS['pb_y2030_n9'], 'evaluation_record.json')
PLAIN_RECORD = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0', 'evals',
                            'd110bd1a5977df1e_x0', 'evaluation_record.json')


def tests_X():
    res = {}
    for name, rel in (('w123_f2_challenger', W123_RECORD), ('w128_pb_y2030_n9', W128_RECORD), ('plain_w101_x0',
                                                                                               PLAIN_RECORD)):
        rec = json.load(open(_abs(rel)))
        before = copy.deepcopy(rec)
        after = H._apply_settling_resettle_status(copy.deepcopy(rec))
        summ = rec.get('settling_resettle_summary')
        res[name] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': _git_tracked_clean(rel),
                     'status_before': before.get('status'), 'status_after': after.get('status'),
                     'settling_status': (summ or {}).get('settling_status'),
                     'certified_cost_after': after.get('certified_cost')}
    x = res
    src_b = inspect.getsource(H.build_evaluation_record)
    src_c = inspect.getsource(H._child_real)
    source_ok = (src_b.find('record.update(extra)') < src_b.find('_apply_settling_resettle_status(record)')
                 < src_b.find('return record')
                 and src_c.find('record = build_evaluation_record(') < src_c.find(
                     "_write_once_json(os.path.join(eval_dir, 'evaluation_record.json'), record)")
                 < src_c.find("status={record['status']}"))
    parts = {
        'w123_uncertified_relabelled_not_certified': (x['w123_f2_challenger']['status_before'] == 'certified'
                                                      and x['w123_f2_challenger']['settling_status'] == 'uncertified'
                                                      and x['w123_f2_challenger']['status_after'] == 'not_certified'
                                                      and x['w123_f2_challenger']['certified_cost_after'] is None),
        'w128_certified_stays_certified': (x['w128_pb_y2030_n9']['settling_status'] == 'certified'
                                           and x['w128_pb_y2030_n9']['status_after'] == 'certified'
                                           and x['w128_pb_y2030_n9']['certified_cost_after'] is not None),
        'record_without_summary_unchanged': (json.load(open(_abs(PLAIN_RECORD)))
                                             == H._apply_settling_resettle_status(json.load(open(_abs(PLAIN_RECORD))))),
        'writer_applies_before_the_record_write_and_the_finish_line': source_ok,
        'inputs_committed_clean': all(v['committed_clean'] for v in x.values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'records': x}


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('R', tests_R), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('K', tests_K),
            ('P', tests_P), ('X', tests_X))

CODE_PINNED_BY_CHECKS = (os.path.basename(__file__), 'settling_criterion_v3.py', 'settling_criterion_v2.py',
                         'settling_criterion.py', 'p515_s53_w132_resettle_v3_hooks.py', 'p515_s53_w118_resettle_hooks.py',
                         'p515_s53_w118_resettle_checks.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py',
                         'interface_dual_capture.py', 'shared_resources_planning.py', 'admm_anderson_acceleration.py',
                         'shared_energy_storage_data.py', 'network.py', 'helper_functions.py',
                         'p515_s53_w105_settling_extension_hooks.py', 'p515_s53_w101_settling_continuation_hooks.py',
                         'p515_s53_w105_extension_checks.py', 'p515_s53_w101_continuation_checks.py',
                         'p515_s53_w98_continuation_checks.py', 'p515_s53_w112_consensus_gap.py',
                         'p515_g_g1_g4_admm_gates.py', 'p515_gate_result_bool_typing_test.py')


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
    return {'all_hold': ok, 'p_max': V.P_MAX, 'constants': SC3.constants(V.P_MAX), 'readings': SC3.READINGS,
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
    doc = {'schema': 'p515_s53_w132_zero_solve_checks_v1',
           'task': 'W132 (PLANNER_BRIEF_2026-09-13.md Addendum 58; TASKS.md Addendum 58 Planner rulings)',
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
        print(f"[W132-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W132-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W132-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) guards={guards}")
    print(f"[W132-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
