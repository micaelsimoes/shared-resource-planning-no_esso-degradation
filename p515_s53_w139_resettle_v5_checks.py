"""
P5.15 Addendum 60, Planner task W139 -- ZERO-SOLVE checks for the v5 re-settling campaign (the 36 cells of the v4
campaign not yet certified under v4; frozen_s53_resettle_spec_v5, predecessor v4 e11fbc89). W137's section set,
re-targeted to v5, plus the v5 unit tests and the clean capture on real production structures.

A `SolveProfileGuard(permitted=())` is armed at import for the whole run and verified at exactly 0 (the W137 checks
module, imported for its helpers, arms its own and the modules it imports arm theirs; every guard is verified at 0).
No model is built and nothing is solved: H13 and C read a fresh SRP1 planning object (data only); everything else reads
committed, manifest- or pin-verified records or synthetic files in a temporary directory.

WHAT IS CHECKED
  V   the stop rule VERSION 5 (`settling_criterion_v5`): V_v4 the W137 version-4 suite (unchanged, run as committed);
      V_eq_v4 version 5 fed a flag series reproduces version 4 fed the same series, record by record and decision by
      decision, under `V4_FIELD_RENAMES`, on every W132 trajectory with planted non-clean cycles (identical otherwise);
      THE UNIT TESTS of the clean rule (Addendum 60): V5a a primary Acceptable within 10x is clean (the recorded DSO7
      2025 Winter metrics, 2.51x) and exactly 10x is clean; V5b one at 11x -- on each of the four metrics -- is not;
      V5c a recovery (and a tier-2 recovery) Acceptable within 10x is not; V5d ESSO covered (its own tolerances: within
      10x clean, 11x not, recovery not); V5e an Optimal exit is clean on any tier, any other exit / no exit / missing
      metrics is not; V5f through the rule: a primary Acceptable within 10x inside the window does NOT veto (v5 certifies
      at the all-Optimal k*, v4 vetoes), at 11x it vetoes exactly as v4; V5g the metric table (scaled NLP error;
      unscaled dual infeasibility, constraint violation, complementarity) and the tolerance table against source; V21
      the per-cell ceiling; V22 all_clean must be a bool; V0 versions 1-4 byte-identical to their pins.
  R   v5 FROM RECORDS (committed item-5 table w139_v5_from_records.json, pinned): every record's v5 decision recomputed
      from its replay inputs and the committed non-clean cycles; v5 never certifies later than v4 where v4 certified
      (strictly more permissive); cell 3 (b_4649234b, v4 run 0734f103) certifies from its records; pb_y2025_n5's cycle
      167 (a recovery Acceptable) stays non-clean and the record stays uncertified; the per-cycle classification and the
      rule inputs reproduce the committed table.
  O   the 38 original cells (W132's section O, reused unchanged: the 36 v5 cells are among them).
  S   production_since_originals: no uncommitted change to a file this run uses (gate); production changes since each
      original's git head (report).
  H   the hooks through the REAL nine wrappers (W118's eight + the v5 exit wrapper) on the v5 state with stand-in
      production and synthetic IPOPT logs: H1 every gated cell with its ORIGINAL values bitwise through k0, then a
      synthetic continuation that certifies; H2 a one-ulp divergence aborts; H3 a primary Acceptable 2.51x INSIDE the
      window does not veto (same k* as all-Optimal); H3b one at 11x vetoes (no lapse, later k*); H3c a recovery
      Acceptable inside the window vetoes; H3d an ESSO Acceptable within 10x does not veto, at 11x it does; H3e an
      Optimal recovery does not veto; H4 a gated creep uncertified at the cap and the record label; H5 / H5b ungated
      dynamic rule cap / certification; H6 2ab0ce2d to its cap 437; H7-H9 the AA / tail / rho holds with REAL
      production; H10 the real install and the harness dispatch (v5 -> this, v5 extension -> its module, v4 -> W137,
      v3 -> W132, W135 -> W135, W118 -> W118); H11 certificate-length writes; H12 W132's exit wrapper on REAL
      production (reused); H13 THE v5 EXIT WRAPPER ON REAL PRODUCTION STRUCTURES: a fresh SRP1 planning object, attempt
      records appended to its REAL Network deques by production's own `network._append_ipopt_solve_record` over
      synthetic IPOPT logs, ESSO entries in its REAL lists, real pyomo SolverResults, real
      `_admm_local_solves_succeeded`: the capture reads what production wrote, classifies, and CONSUMES NOTHING
      (production's drain afterwards still returns every record).
  C   the clean capture checklist (the source facts; the tolerance table against the case files and
      ESSO_TOL_OVERRIDES; the classifier unit cases); the parser on the recorded DSO7 2025 Winter cycle-120 segment of
      W133's cell 1; a stale / missing capture, a tol in force other than the table, a log EXIT class other than the
      result's, an ESSO log of another cycle and an accepted ESSO exit without its entry all raise.
  C2  THE CAPTURE REPLAYED ON A COMMITTED RUN: the in-cycle capture functions fed, cycle by cycle, what production held
      at the exit wrapper in the committed v4 cell-3 run (0734f103: its network attempt records by round, its ESSO
      complementarity entries) reproduce the committed item-5 table (the non-clean cycles, every non-Optimal exit's
      tier, reason and metrics) and the v5 decision from records.
  K   keys: every entry of every committed campaign spec keys identically under this harness and the pre-W139 harness
      (4ef636b1, 25cc4559, sha256 pinned, from git) -- the 38 v4 and the 38 v3 entries, W135's and W118's included --
      except the OWN entries of the two families the v5 route adds: v5 entries accepted only inside the v5 stage root,
      v5-extension entries only inside the v5-extension stage root, each only if its frozen key follows the formula over
      the pre-W139 base key (a pre-run scan excludes the run's own, per root, over ALL roots); the 36 v5 keys follow the
      formula, are distinct, differ from the originals and from the v3 and v4 keys, and appear in no committed spec
      outside the v5 root; planted controls; the pre-W139 harness refuses a v5 declaration.
  P   `assert_resettle_preconditions` holds for every cell with its cap; negative controls refused; the validator
      refuses early_stop and malformed declarations (a v4 rule declaration included).
  X   W137's section X reused (W132's status-label writer checks; G8 on production's certificate).
  D   the harness is EXACTLY the pre-W139 harness (25cc4559) plus the v5 branch (3 added lines in
      `resettle_hooks_module`, after the v4 branch and before the W118 fallback, none removed; every existing branch
      identical); its sha256 is the pinned post-v5 sha.
  W   (main only) W100's repository-wide boolean-typing test, output to a new write-once file.

Run (repo root, canonical interpreter, attached, alone, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w139_resettle_v5_checks.py \\
      > data/SRP1/Results/P515S53/w139_resettle_v5/zero_solve_checks_launch.log 2>&1
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W139 v5 re-settling zero-solve checks (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import interface_dual_capture as IDC  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w139_resettle_v5_hooks as V5  # noqa: E402
import p515_s53_w137_resettle_v4_hooks as V4  # noqa: E402
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w135_resettle_ext_hooks as W135  # noqa: E402
import p515_s53_w118_resettle_hooks as R  # noqa: E402
import p515_s53_w137_resettle_v4_checks as K137  # noqa: E402 -- the v4 suite and helpers (arms its own guard)
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v4 as SC4  # noqa: E402
import settling_criterion_v5 as SC5  # noqa: E402

K132 = K137.K132
K118, K105, K98 = K132.K118, K132.K105, K132.K98


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w139_checks', GUARD),) + tuple(K137.GUARDS))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W139_ROOT_REL = os.path.join(_P53, 'w139_resettle_v5')
W139_EXT_ROOT_REL = os.path.join(_P53, 'w139_resettle_ext_v5')
W137_ROOT_REL = K137.W137_ROOT_REL
W135_ROOT_REL = K137.W135_ROOT_REL
OUT_DIR_REL = os.path.join(W139_ROOT_REL, 'zero_solve_checks')
OUT_FILE = 'w139_zero_solve_checks.json'
OUT_MANIFEST = 'w139_zero_solve_checks_manifest_sha256.json'
TYPING_OUT = 'w139_bool_typing_test.json'
CAMPAIGN_ID_PREFIX = 's53_w139_resettle_v5_'
EXT_CAMPAIGN_ID_PREFIX = 's53_w139_resettle_ext_v5_'
KEY_OWN_ROOTS = {'v5': W139_ROOT_REL, 'ext_v5': W139_EXT_ROOT_REL}
# the harness before W139 (pinned by the v4 stage spec e11fbc89; last changed by 4ef636b1, router commit 2 of W137)
PRE_W139_HARNESS = {'commit': '4ef636b125165c4f0cc5b10255831f0c23a14f35',
                    'sha256': '25cc4559db5ebe9904d9ed8f97f9a6182263d3e5126b935e47dc21727040ce10'}
V5_BRANCH_LINES = ('    import p515_s53_w139_resettle_v5_hooks as W139C',
                   '    if W139C.is_v5_declaration(value):',
                   '        return W139C.hooks_module(value)')
HARNESS_POST_V5_SHA256 = '961dbb3e6a6808bcc3665a081024af12a196e8f4ddf2b80e2f70dbf7edb25638'   # pre-W139 + the v5 branch
V4_STAGE_SPEC = {'path': os.path.join(W137_ROOT_REL, 'frozen_s53_resettle_spec_v4_e11fbc89.json'),
                 'sha256': 'e11fbc89'}
FROM_RECORDS = {'path': os.path.join(W139_ROOT_REL, 'v5_from_records', 'w139_v5_from_records.json'),
                'manifest': os.path.join(W139_ROOT_REL, 'v5_from_records', 'manifest_sha256.json')}
CELL3 = 'b_4649234b'
CELL3_V4_RECORD = 'b_4649234b@v4'
PB_N5 = {'record': 'pb_y2025_n5', 'cycle': 167, 'block': 'TSO|2035|Spring'}
PRODUCTION_FILES = K132.PRODUCTION_FILES
_OPT, _ACC = K132._OPT, K132._ACC
_MAXIT = 'Ipopt 3.14.18\\x3a Maximum Number of Iterations Exceeded.'


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


def harness_post_v5_sha256():
    """The pinned post-v5 harness sha256 (computed before the router commit from the pre-W139 harness plus the three
    branch lines; the router commit's message carries it)."""
    return HARNESS_POST_V5_SHA256


# ======================================================================================================================
#  synthetic IPOPT logs (the final summary the capture parses)
# ======================================================================================================================
_T = SC5.TOLERANCES
# profile -> (exit text, {line: (scaled, unscaled)})
_DSO7 = {'Dual infeasibility': (1.141920220071313e-05, 0.01141920220071313),
         'Constraint violation': (4.473030956199322e-14, 7.670947210769441e-14),
         'Variable bound violation': (9.977039085510026e-09, 9.977039085510026e-09),
         'Complementarity': (2.5123656330789977e-09, 2.5123656330789977e-06),
         'Overall NLP error': (1.141920220071313e-05, 0.01141920220071313)}
PROFILES = {
    'opt': ('Optimal Solution Found.', {'Dual infeasibility': (1e-9, 1e-6), 'Constraint violation': (1e-14, 1e-14),
                                        'Variable bound violation': (1e-9, 1e-9), 'Complementarity': (1e-9, 1e-6),
                                        'Overall NLP error': (1e-6, 1e-3)}),
    'acc_2p51': ('Solved To Acceptable Level.', _DSO7),
    'acc_11x': ('Solved To Acceptable Level.', dict(_DSO7, **{'Complementarity': (1.1e-8, 1.1e-5)})),
    'acc_10x': ('Solved To Acceptable Level.', dict(_DSO7, **{'Complementarity': (1e-8, 10.0 * 1e-6)})),
    'maxit': ('Maximum Number of Iterations Exceeded.', dict(_DSO7, **{'Complementarity': (1e-3, 1.0)})),
    'esso_opt': ('Optimal Solution Found.', {'Dual infeasibility': (1e-12, 1e-11), 'Constraint violation': (1e-15, 1e-14),
                                             'Variable bound violation': (1e-8, 1e-8),
                                             'Complementarity': (1e-12, 1e-11), 'Overall NLP error': (1e-12, 1e-11)}),
    'esso_acc_within': ('Solved To Acceptable Level.', {'Dual infeasibility': (1e-12, 1e-11),
                                                        'Constraint violation': (1e-15, 1e-14),
                                                        'Variable bound violation': (1e-8, 1e-8),
                                                        'Complementarity': (2.5e-5, 2.5e-4),
                                                        'Overall NLP error': (5e-10, 5e-9)}),
    'esso_acc_11x': ('Solved To Acceptable Level.', {'Dual infeasibility': (1e-12, 1e-11),
                                                     'Constraint violation': (1e-15, 1e-14),
                                                     'Variable bound violation': (1e-8, 1e-8),
                                                     'Complementarity': (1.1e-4, 1.1e-3),
                                                     'Overall NLP error': (5e-10, 5e-9)}),
}


def synthetic_log_text(profile, with_options=True):
    exit_text, lines = PROFILES[profile]
    parts = []
    if with_options:
        parts.append('List of options:\n\n                                    Name   Value                # times used\n'
                     '                                     tol = 1e-05                     2\n'
                     '                           compl_inf_tol = 1e-06                     2\n\n')
    parts.append('******************************************************************************\n'
                 'This is Ipopt version 3.14.18, running with linear solver ma97.\n\n'
                 'Current barrier parameter mu = 2.5e-09\nobjective scaling factor = 0.001\n\n'
                 'Number of Iterations....: 43\n\n'
                 '                                   (scaled)                 (unscaled)\n'
                 'Objective...............:   7.3088814497116420e+01    7.3088814497116410e+04\n')
    for name in ('Dual infeasibility', 'Constraint violation', 'Variable bound violation', 'Complementarity',
                 'Overall NLP error'):
        label = {'Dual infeasibility': 'Dual infeasibility......', 'Constraint violation': 'Constraint violation....',
                 'Variable bound violation': 'Variable bound violation', 'Complementarity': 'Complementarity.........',
                 'Overall NLP error': 'Overall NLP error.......'}[name]
        s, u = lines[name]
        parts.append(f'{label}:   {s:.16e}    {u:.16e}\n')
    parts.append(f'\n\nNumber of objective function evaluations             = 44\nTotal seconds in IPOPT                   '
                 f'            = 0.139\n\nEXIT: {exit_text}\n')
    return ''.join(parts)


def profile_metrics(profile):
    _e, lines = PROFILES[profile]
    return {m: lines[row['log_line']][0 if row['column'] == 'scaled' else 1] for m, row in SC5.METRIC_TABLE.items()}


class SyntheticCapture:
    """Stand-in capture sources for the drive: per cycle and block, a profile ('opt' default, 'acc_2p51', 'acc_11x',
    'acc_2p51@recovery', 'opt@recovery', 'esso_acc_within', 'esso_acc_11x', ...): the network chain reader returns
    production-shaped attempt records over synthetic log files; before each local-solve check `stage(pp, cycle)` appends
    the ESSO entries production would append."""

    def __init__(self, tmp, plan):
        self.tmp = tmp
        self.plan = plan or {}
        self.cycle = None
        self.files = {}

    def _file(self, name, profile):
        path = os.path.join(self.tmp, name)
        if not os.path.exists(path):
            with open(path, 'w') as handle:
                handle.write(synthetic_log_text(profile))
        return path

    def profile(self, cycle, key):
        return (self.plan.get(cycle) or {}).get(key, 'esso_opt' if key.startswith('ESSO') else 'opt')

    def message(self, cycle, key):
        p = self.profile(cycle, key).split('@')[0]
        return _OPT if PROFILES[p][0].startswith('Optimal') else (_ACC if 'Acceptable' in PROFILES[p][0] else _MAXIT)

    def net_chain_reader(self, pp):
        out = {}
        for key in V.block_keys(pp):
            kt = V._key_text(key)
            if kt.startswith('ESSO'):
                continue
            prof = self.profile(self.cycle, kt)
            base, _, tier = prof.partition('@')
            chain = []
            if tier:
                p0 = self._file('net_primary_maxit.log', 'maxit')
                chain.append({'attempt': 'primary', 'log_path': p0, 'log_bytes': [0, os.path.getsize(p0)],
                              'exit': PROFILES['maxit'][0], 'tol_in_force': _T['network']['tol'],
                              'compl_inf_tol_in_force': 1e-6})
            p = self._file(f'net_{base}.log', base)
            chain.append({'attempt': tier or 'primary', 'log_path': p, 'log_bytes': [0, os.path.getsize(p)],
                          'exit': PROFILES[base][0], 'tol_in_force': _T['network']['tol'],
                          'compl_inf_tol_in_force': 1e-6})
            out[kt] = chain
        return out

    def stage(self, pp, cycle):
        """Append this call's ESSO entries (cycle None = the initialisation) to pp.shared_ess_data's lists."""
        self.cycle = cycle
        sed = pp.shared_ess_data
        stamp = 'init' if cycle is None else f'cycle{cycle:03d}'
        for n in pp.active_distribution_network_nodes:
            prof = self.profile(cycle, f'ESSO|{n}')
            base, _, tier = prof.partition('@')
            suffix = {'': '', 'recovery': '_recovery', 'recovery_tier2': '_recovery_tier2'}[tier]
            path = self._file(f'optim_log_esso_node{n}_{stamp}{suffix}.txt', base)
            if PROFILES[base][0].startswith(('Optimal', 'Solved')):
                sed.esso_complementarity_diagnostics.append({'node_id': n, 'log_path': path})
            if tier:
                sed.solver_recovery_diagnostics.append({'subsystem': 'esso', 'node_id': n, 'tier': 'tier1',
                                                        'tier2_attempted': tier == 'recovery_tier2',
                                                        'recovery_log': path})

    def results(self, pp, cycle):
        out = {'tso': {}, 'dso': {}, 'esso': {}}
        for key in V.block_keys(pp):
            kind, n, y, d = key
            r = SimpleNamespace(solver=SimpleNamespace(message=self.message(cycle, V._key_text(key))))
            if kind == 'TSO':
                out['tso'].setdefault(y, {})[d] = r
            elif kind == 'DSO':
                out['dso'].setdefault(n, {}).setdefault(y, {})[d] = r
            else:
                out['esso'][n] = r
        return out


def _tol_ok(_pp):
    return True, {'stand_in': True}


# ======================================================================================================================
#  V -- the stop rule v5
# ======================================================================================================================
def _rename(x):
    """A version-4 record / decision under V4_FIELD_RENAMES (keys and the veto-reason string values)."""
    if isinstance(x, dict):
        return {SC5.V4_FIELD_RENAMES.get(k, k): _rename(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rename(v) for v in x]
    if isinstance(x, tuple):
        return [_rename(v) for v in x]
    if isinstance(x, str):
        return SC5.V4_FIELD_RENAMES.get(x, x)
    return x


_DEC_VERSION_ONLY = ('version', 'reading', 'N_definition', 'window_rule', 'clean_rule')


def _dec_cmp(d):
    return None if d is None else {k: v for k, v in d.items() if k not in _DEC_VERSION_ONLY}


def _replay5(q, b, t, ac, kw):
    kw5 = dict(kw)
    kw5.setdefault('cap_ceiling', SC2.CAP_CEILING)
    return SC5.replay(q, b, t, ac, R.P_MAX, **kw5)


def _replay4(q, b, t, ao, kw):
    kw4 = dict(kw)
    kw4.setdefault('cap_ceiling', SC2.CAP_CEILING)
    return SC4.replay(q, b, t, ao, R.P_MAX, **kw4)


def _summ(d):
    d = d or {}
    return {k: d.get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'window', 'branch')}


def classifier_unit_tests():
    net, esso = _T['network'], _T['esso']
    dso7 = {'overall_nlp_error': 1.141920220071313e-05, 'dual_infeasibility': 0.01141920220071313,
            'constraint_violation': 7.670947210769441e-14, 'complementarity': 2.5123656330789977e-06}
    c = SC5.classify_block_exit
    res = {}
    a = c('acceptable', 'primary', dso7, 'network')
    exact = dict(dso7, complementarity=10.0 * net['compl_inf_tol'])
    res['V5a_primary_acceptable_within_10x_is_clean'] = {
        'ok': bool(a['clean'] and a['reason'] == 'acceptable_primary_within_factor'
                   and abs(a['max_ratio'] - 2.5123656330789977) < 1e-12 and a['max_ratio_metric'] == 'complementarity'
                   and c('acceptable', 'primary', exact, 'network')['clean']),
        'dso7_2025_winter_recorded_metrics': dso7, 'ratios': a['ratios'], 'max_ratio': a['max_ratio'],
        'exactly_10x_complementarity_clean': c('acceptable', 'primary', exact, 'network')['clean']}
    over = {}
    for m, row in SC5.METRIC_TABLE.items():
        tol = net[row['option']]
        mm = dict(dso7, **{m: 11.0 * tol})
        r = c('acceptable', 'primary', mm, 'network')
        over[m] = {'clean': r['clean'], 'reason': r['reason'], 'ratio': r['ratios'][m]}
    res['V5b_primary_acceptable_at_11x_vetoes_each_metric'] = {
        'ok': all(v['clean'] is False and v['reason'] == 'acceptable_beyond_factor' and abs(v['ratio'] - 11.0) < 1e-9
                  for v in over.values()), 'per_metric': over}
    rec = {t: c('acceptable', t, dso7, 'network') for t in ('recovery', 'recovery_tier2')}
    res['V5c_recovery_acceptable_within_10x_vetoes'] = {
        'ok': all(v['clean'] is False and v['reason'] == 'acceptable_not_primary' for v in rec.values()),
        'per_tier': {t: v['reason'] for t, v in rec.items()},
        'pb_y2025_n5_like_2234x_recovery': c('acceptable', 'recovery',
                                             dict(dso7, complementarity=2234.36 * net['compl_inf_tol']),
                                             'network')['reason']}
    e_in = {'overall_nlp_error': 5e-10, 'dual_infeasibility': 1e-11, 'constraint_violation': 1e-14,
            'complementarity': 2.5e-4}
    e_over = dict(e_in, complementarity=11.0 * esso['compl_inf_tol'])
    e_nlp_over = dict(e_in, overall_nlp_error=11.0 * esso['tol'])
    ee = {'within': c('acceptable', 'primary', e_in, 'esso'), 'compl_11x': c('acceptable', 'primary', e_over, 'esso'),
          'nlp_11x': c('acceptable', 'primary', e_nlp_over, 'esso'),
          'recovery_within': c('acceptable', 'recovery', e_in, 'esso'), 'optimal': c('optimal', 'primary', None, 'esso'),
          'network_metrics_on_esso_tolerances': c('acceptable', 'primary', dso7, 'esso')}
    res['V5d_esso_covered_its_own_tolerances'] = {
        'ok': bool(ee['within']['clean'] and not ee['compl_11x']['clean'] and not ee['nlp_11x']['clean']
                   and not ee['recovery_within']['clean'] and ee['optimal']['clean']
                   and ee['network_metrics_on_esso_tolerances']['reason'] == 'acceptable_beyond_factor'
                   and ee['within']['tolerances'] == esso),
        'cases': {k: {'clean': v['clean'], 'reason': v['reason'], 'max_ratio': v['max_ratio']} for k, v in ee.items()}}
    other = {'optimal_primary': c('optimal', 'primary', None, 'network'),
             'optimal_recovery': c('optimal', 'recovery', None, 'network'),
             'optimal_tier2': c('optimal', 'recovery_tier2', None, 'network'),
             'other_exit': c('other', 'primary', dso7, 'network'), 'no_exit': c(None, None, None, 'network'),
             'acceptable_no_metrics': c('acceptable', 'primary', None, 'network'),
             'acceptable_partial_metrics': c('acceptable', 'primary', dict(dso7, complementarity=None), 'network'),
             'acceptable_nan_metric': c('acceptable', 'primary', dict(dso7, complementarity=float('nan')), 'network'),
             'acceptable_unknown_tier': c('acceptable', None, dso7, 'network')}
    res['V5e_optimal_any_tier_clean_everything_else_not'] = {
        'ok': bool(other['optimal_primary']['clean'] and other['optimal_recovery']['clean']
                   and other['optimal_tier2']['clean'] and not other['other_exit']['clean']
                   and not other['no_exit']['clean'] and not other['acceptable_no_metrics']['clean']
                   and not other['acceptable_partial_metrics']['clean'] and not other['acceptable_nan_metric']['clean']
                   and not other['acceptable_unknown_tier']['clean']),
        'cases': {k: v['reason'] for k, v in other.items()}}
    try:
        c('acceptable', 'primary', dso7, 'tso')
        fam_refused = False
    except ValueError:
        fam_refused = True
    res['V5e_unknown_family_refused'] = {'ok': fam_refused}
    return res


def tests_V():
    res = {}
    v4 = K137.tests_V()
    res['V_v4_suite_as_committed'] = {'ok': v4['holds'] is True,
                                      'tests': {k: v.get('ok') for k, v in v4['tests'].items()}}
    trajs, q6, b6, k6 = K132._trajectories()
    eq = {}
    for name, q, b, t, kw in trajs:
        _o, d0, _r = _replay4(q, b, t, {k: True for k in q}, kw)
        ks = sorted(q)
        plants = {'none': set(), 'every_7th': {k for k in ks if k % 7 == 0}}
        if (d0 or {}).get('status') == 'certified':
            plants['at_k_star_minus_1'] = {d0['k_star'] - 1}
            plants['just_before_window'] = {d0['window'][0] - 1}
        for pname, non in plants.items():
            flags = {k: k not in non for k in q}
            o4, d4, r4 = _replay4(q, b, t, flags, kw)
            o5, d5, r5 = _replay5(q, b, t, flags, kw)
            recs_equal = [_jt(_rename(a)) for a in o4] == [_jt(x) for x in o5]
            dec_equal = _jt(_dec_cmp(_rename(d4))) == _jt(_dec_cmp(d5))
            eq[f'{name}|{pname}'] = {'ok': bool(recs_equal and dec_equal and (d5 or {}).get('version') == 5
                                                and len(r4.vetoes) == len(r5.vetoes)),
                                     'records_equal': recs_equal, 'decision_equal': dec_equal, 'v5': _summ(d5)}
    res['V_eq_v4_under_the_renames'] = {'ok': all(v['ok'] for v in eq.values()), 'n': len(eq),
                                        'failing': {k: v for k, v in eq.items() if not v['ok']}}
    res.update(classifier_unit_tests())
    # V5f: through the rule -- the damped cosine: a primary Acceptable within 10x inside the window does NOT veto
    t0 = {k: 0.0 for k in q6}
    _o, d_all, _r = _replay5(q6, b6, t0, {k: True for k in q6}, {'cap': 200})
    in_c = k6 - 2
    dso7 = profile_metrics('acc_2p51')
    within = SC5.classify_block_exit('acceptable', 'primary', dso7, 'network')['clean']
    over = SC5.classify_block_exit('acceptable', 'primary', profile_metrics('acc_11x'), 'network')['clean']
    _o5, d5w, r5w = _replay5(q6, b6, t0, {k: (within if k == in_c else True) for k in q6}, {'cap': 200})
    _o4, d4w, r4w = _replay4(q6, b6, t0, {k: k != in_c for k in q6}, {'cap': 200})
    _o5o, d5o, r5o = _replay5(q6, b6, t0, {k: (over if k == in_c else True) for k in q6}, {'cap': 200})
    res['V5f_through_the_rule'] = {
        'ok': bool(within is True and over is False and d5w['status'] == 'certified' and d5w['k_star'] == d_all['k_star']
                   == k6 and not r5w.vetoes and r4w.vetoes and r4w.vetoes[0]['cycle'] == k6
                   and (d4w.get('k_star') or 10 ** 9) > k6 and r5o.vetoes and r5o.vetoes[0]['cycle'] == k6
                   and _summ(d5o) == _summ(d4w)),
        'k_star_all_optimal': d_all['k_star'], 'acceptable_cycle': in_c,
        'v5_within_10x': _summ(d5w), 'v4_same_series': _summ(d4w), 'v5_at_11x': _summ(d5o)}
    # V5g: the metric table and the tolerance table against source
    ok_tol, tol = V5.case_file_tolerances()
    res['V5g_metric_table_and_tolerances_against_source'] = {
        'ok': bool(ok_tol and SC5.METRIC_TABLE['overall_nlp_error']['column'] == 'scaled'
                   and all(SC5.METRIC_TABLE[m]['column'] == 'unscaled' for m in ('dual_infeasibility',
                                                                               'constraint_violation', 'complementarity'))
                   and [SC5.METRIC_TABLE[m]['option'] for m in SC5.METRICS] == ['tol', 'dual_inf_tol', 'constr_viol_tol',
                                                                               'compl_inf_tol']
                   and SC5.CLEAN_FACTOR == 10.0),
        'metric_table': SC5.METRIC_TABLE, 'tolerances': SC5.TOLERANCES, 'source_check': tol}
    # V21 / V22
    try:
        r437 = SC5.SettlingRuleV5(R.P_MAX, cap=437, cap_ceiling=437)
        ok437 = r437.cap == 437
    except Exception:  # noqa: BLE001
        ok437 = False
    refused = {}
    for name, kw in (('cap_437_ceiling_300', {'cap': 437, 'cap_ceiling': 300}), ('no_ceiling', {'cap': 200}),
                     ('no_ceiling_dynamic', {'cap_after_first_k0': 109})):
        try:
            SC5.SettlingRuleV5(R.P_MAX, **kw)
            refused[name] = False
        except ValueError:
            refused[name] = True
    res['V21_per_cell_ceiling'] = {'ok': bool(ok437 and all(refused.values())), 'cap_437_ceiling_437_accepted': ok437,
                                   'refused': refused}
    bad = {}
    for v in (None, 1, 'True'):
        try:
            SC5.SettlingRuleV5(R.P_MAX, cap=10, cap_ceiling=300).observe(1, 1.0, True, 0.0, v)
            bad[repr(v)] = False
        except TypeError:
            bad[repr(v)] = True
    res['V22_all_clean_is_a_bool'] = {'ok': all(bad.values()), 'refused': bad}
    # V0: versions 1-4 byte-identical to their pins (v4 pinned by the v4 stage spec e11fbc89)
    ss4 = json.load(open(_abs(V4_STAGE_SPEC['path'])))
    pins4 = ss4['pins']['code_sha256']
    names = ('settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v3.py', 'settling_criterion_v4.py')
    now = {n: _sha(n) for n in names}
    res['V0_versions_1_to_4_byte_identical'] = {
        'ok': bool(all(now[n] == pins4.get(n) for n in names) and _sha(V4_STAGE_SPEC['path']).startswith(
            V4_STAGE_SPEC['sha256']) and all(_clean(n) for n in names)),
        'now': now, 'pins_e11fbc89': {n: pins4.get(n) for n in names}}
    return {'holds': all(v['ok'] for v in res.values()), 'tests': res}


# ======================================================================================================================
#  R -- v5 from records (the committed item-5 table)
# ======================================================================================================================
def _from_records_doc():
    rel, man_rel = FROM_RECORDS['path'], FROM_RECORDS['manifest']
    man = json.load(open(_abs(man_rel)))
    sha = _sha(rel)
    if man.get(rel) != sha or not _clean(rel) or not _clean(man_rel):
        raise RuntimeError(f'{rel}: sha256 {sha} != manifest {man.get(rel)} or not committed clean')
    return json.load(open(_abs(rel))), sha


def tests_R():
    import p515_s53_w139_v5_from_records as F
    doc, sha = _from_records_doc()
    with contextlib.redirect_stdout(io.StringIO()):
        recs = F.record_inputs()
    cells = {}
    for rid, rep in doc['reports'].items():
        rec = recs[rid]
        n = rec['n']
        nonclean = set(rep['non_clean_cycles'])
        nonopt = set(rep['non_optimal_cycles'])
        ac = {k: k not in nonclean for k in rec['q']}
        ao = {k: k not in nonopt for k in rec['q']}
        _o5, d5, r5 = _replay5(rec['q'], rec['b'], rec['t'], ac, dict(rec['kw'], last=n))
        _o4, d4, r4 = _replay4(rec['q'], rec['b'], rec['t'], ao, dict(rec['kw'], last=n))
        s5 = F._summ(d5, r5, n)
        s4 = F._summ(d4, r4, n)
        committed5 = rep['v5_from_records']
        same = (s5['status'], s5.get('k_star'), s5.get('window')) == (committed5['status'], committed5.get('k_star'),
                                                                       committed5.get('window'))
        monotone = (s4['status'] != 'certified' or (s5['status'] == 'certified' and s5['k_star'] <= s4['k_star']))
        nonclean_subset = nonclean <= nonopt
        cells[rid] = {'ok': bool(same and monotone and nonclean_subset
                                 and (rep.get('in_run_all_optimal_equals_records') in (None, True))),
                      'v5_recomputed': {k: s5.get(k) for k in ('status', 'k_star', 'window')},
                      'v5_committed': {k: committed5.get(k) for k in ('status', 'k_star', 'window')},
                      'v4_recomputed': {k: s4.get(k) for k in ('status', 'k_star')},
                      'v5_not_later_than_v4': monotone, 'non_clean_subset_of_non_optimal': nonclean_subset}
    c3 = doc['reports'][CELL3_V4_RECORD]['v5_from_records']
    pb = doc['pb_y2025_n5_cycle_167_check']
    parts = {
        'every_record_recomputed_and_consistent': all(v['ok'] for v in cells.values()) and len(cells) == 16,
        'cell3_certifies_from_its_v4_records': c3['status'] == 'certified' and c3['k_star'] is not None,
        'cell3_prediction_input_equals_the_table': doc['planner_cell3_prediction_input']['prediction_cycle']
        == c3['k_star'],
        'pb_y2025_n5_cycle_167_non_clean_recovery': (pb['non_clean'] is True and pb['reason'] == 'acceptable_not_primary'
                                                     and pb['still_vetoes'] is True and pb['max_ratio'] > 2000.0),
        'reader_crosscheck_all_equal': doc['reader_crosscheck_all_equal'] is True,
        'guards_zero': all(not v['verify_0_failures'] for v in doc['guards'].values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'from_records': {'path': FROM_RECORDS['path'], 'sha256': sha},
            'cells': cells, 'cell3_v5_from_records': c3, 'pb_y2025_n5': pb}


# ======================================================================================================================
#  O, S -- W132's / W137's, reused
# ======================================================================================================================
def tests_O():
    r = K132.tests_O()
    missing = [c for c in V5.CELL_ORDER if c not in (r.get('cells') or {})]
    return dict(r, v5_cells_all_in_section_o=not missing, holds=bool(r.get('holds') and not missing))


def tests_S():
    r = K132.production_since_originals(CODE_PINNED_BY_CHECKS)
    return {'holds': r['ok'], **r}


# ======================================================================================================================
#  H -- the hooks through the real wrappers, on the v5 state
# ======================================================================================================================
fake_results = K132.fake_results
HOLDS_OFF = K132.HOLDS_OFF
HOLDS_ON = K132.HOLDS_ON


def drive(cell, variant='certify', first_pass_at=50, plan=None, module=V5, state_cls=None, tmp=None):
    """K137's `drive` restated for the v5 declaration, state and exit wrapper: the REAL nine wrappers
    (`V5.make_wrappers`) driven in production's call order with stand-in production; the clean capture reads a
    SyntheticCapture (production-shaped attempt records over synthetic IPOPT logs; ESSO entries appended to a stand-in
    shared_ess_data before each local-solve check). `plan` = {cycle: {block key: profile}}."""
    own_tmp = tmp is None
    tmp = tmp or tempfile.mkdtemp(prefix='w139_drive_')
    try:
        decl = module.declaration_for(cell)
        ref = module.load_replay_reference(decl)
        cap = module.spec_cap(cell)
        sink = []
        st = (state_cls or V5.ResettleStateV5)(decl, None, cap, reference=ref, sink=sink)
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
        cap_src = SyntheticCapture(tmp, plan)

        def t_target(c):
            if gated and c <= n_old:
                return 12345.0
            j = c - (n_old if gated else first_pass_at)
            if variant == 'gap' and j <= 80:
                return 5000.0
            return 150.0
        pp, tso, dso, esso, cv, set_cycle = K118._fake_world(t_target)
        pp.shared_ess_data = SimpleNamespace(esso_complementarity_diagnostics=[], solver_recovery_diagnostics=[])
        w = V5.make_wrappers(st, orig, H.ipopt_exit_class, srp_module=K118._fake_srp(blocks_by_cycle, st),
                             classifier_label='p515_s44_campaign_harness.ipopt_exit_class (checks)',
                             net_chain_reader=cap_src.net_chain_reader, tolerance_reader=_tol_ok)
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
            cap_src.stage(pp, None)
            w['_admm_local_solves_succeeded'](pp, cap_src.results(pp, None))       # the initialisation check
            w['_capture_convergence_depth_tail_baseline'](pp, admm)
            for c in range(1, cap + 1):
                if gated and c <= n_old:
                    row, g = ref[c], g_rows[c]
                    bm = K105.boyd_from_g(g)
                    rc = {'gross_operational_cost': row['gross_operational_cost'],
                          'net_operational_recourse': row['recourse'],
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
                    cap_src.stage(pp, c)
                    w['_admm_local_solves_succeeded'](pp, cap_src.results(pp, c))
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
    finally:
        if own_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def pure_rule(decl):
    cr = decl['cap_rule']
    p_max = decl['settling_rule']['p_max']
    if cr['kind'] == 'fixed':
        return SC5.SettlingRuleV5(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return SC5.SettlingRuleV5(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


def pure_equal(decl, lines):
    """The pure v5 rule over the cycle lines (Q, the v2 boyd_k, t_sum, all_clean_k) reproduces every in-cycle rule
    record; returns (equal, decision)."""
    rule = pure_rule(decl)
    pure = [rule.observe(x['cycle'], x.get('gross'), bool(x.get('boyd_k')), x.get('t_sum'), bool(x.get('all_clean_k')))
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
            'first_k0_v2': summ.get('first_k0_v2'), 'non_clean_cycles': summ.get('non_clean_cycles'),
            'non_optimal_cycles': summ.get('non_optimal_cycles'),
            'n_vetoes': summ.get('n_vetoes'), 'vetoes': summ.get('vetoes'),
            'replay_bitwise_through': summ['replay_bitwise_through_cycle'],
            'first_divergence': summ['replay_first_divergence'], 'n_overlap': len(summ['overlap_k0_plus_1_to_N_old']),
            'overlap_all_zero': all(o['Q_new_minus_Q_old'] == 0.0 for o in summ['overlap_k0_plus_1_to_N_old']),
            'gap_refusals': len(summ['gap_refusals']), 'n_lapses': len(summ['lapse_events']),
            'certificate_length_after': d['admm'].minimum_consecutive_converged_cycles, 'ended_by': st.ended_by,
            'in_cycle_rule_equals_pure_replay': pure_ok, 'pure_decision_status': (pure_dec or {}).get('status'),
            'exit_capture_complete': summ.get('exit_capture_complete'),
            'clean_capture_complete': summ.get('clean_capture_complete'),
            'exits_51_every_line': all((x.get('ipopt_exit_counts') or {}).get('total') == 51
                                       and len(x.get('ipopt_exit_by_block') or {}) == 51 for x in lines),
            'clean_51_every_line': all(len(x.get('exit_clean_by_block') or {}) == 51 for x in lines),
            'all_clean_every_line_is_bool': all(isinstance(x.get('all_clean_k'), bool) for x in lines),
            'all_clean_is_the_conjunction': all(x.get('all_clean_k') == all(v['clean'] for v in (
                x.get('exit_clean_by_block') or {}).values()) for x in lines),
            't_sum_every_line': all(isinstance(x.get('t_sum'), float) for x in lines),
            'holds_through_first_pass': holds_pre, 'holds_after_first_pass': holds_post,
            'decision_version': dec.get('version'), 'out_of_window_reads_recorded': dec.get('out_of_window_reads')
            is not None if dec.get('status') == 'certified' else None,
            'acceptable_clean_cycles': sorted(x['cycle'] for x in lines if x.get('acceptable_clean_blocks')),
            'capture_errors': summ['capture_errors'][:3], 'errors': summ['errors'][:3],
            'last_cycle': st.cycle, 'rule_cap': st.rule.cap, 'summary': summ,
            'sample_clean_entry': next((x['exit_clean_by_block'][b] for x in lines for b in (x.get('non_clean_blocks')
                                                                                             or x.get('acceptable_clean_blocks') or [])), None)}


def _h_real_v5(srp, which):
    """K105's real-production hold tests on the v5 state and the nine v5 wrappers (K137's `_h_real_v4` on the v5 state)."""
    real_state, real_make, real_wrapped = K105._state, K105.E.make_wrappers, K105.E.WRAPPED
    cell = V5.CELL_ORDER[0]

    def state():
        st = V5.ResettleStateV5(V5.declaration_for(cell), None, V5.spec_cap(cell), reference={}, sink=[])
        st.first_pass = K105.E.N_HOLD
        return st
    K105._state = state
    K105.E.make_wrappers = lambda st, originals, srp_module=None: V5.make_wrappers(
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
    ext = importlib.import_module(V5.EXT_HOOKS_MODULE)
    v5d = V5.declaration_for(CELL3)
    extd = ext.declaration_for(ext.CELL_ORDER[0])
    v4d, v3d = V4.declaration_for(K137.CELL1), V.declaration_for(K137.CELL1)
    w135d, w118d = W135.declaration_for(W135.CELL_ORDER[0]), R.declaration_for('f2_challenger')
    return {'v5_to_w139': H.resettle_hooks_module(v5d) is V5,
            'ext_v5_to_the_w139_extension': H.resettle_hooks_module(extd) is ext,
            'v4_to_w137': H.resettle_hooks_module(v4d) is V4,
            'v3_to_w132': H.resettle_hooks_module(v3d) is V,
            'w135_to_w135': H.resettle_hooks_module(w135d) is W135,
            'w118_to_w118': H.resettle_hooks_module(w118d) is R,
            'harness_validates_v5': H.validate_settling_resettle(v5d) == v5d,
            'harness_validates_ext_v5': H.validate_settling_resettle(extd) == extd,
            'harness_validates_v4_unchanged': H.validate_settling_resettle(v4d) == v4d,
            'harness_validates_v3_unchanged': H.validate_settling_resettle(v3d) == v3d,
            'harness_validates_w135_unchanged': H.validate_settling_resettle(w135d) == w135d,
            'harness_validates_w118_unchanged': H.validate_settling_resettle(w118d) == w118d}


def _h_layering(srp):
    """K137's real-install test restated for the v5 state and the six-way dispatch. The capture readers are the REAL
    ones; the fake holders carry no network deques, so no in-cycle local-solve check is made (H13 covers it)."""
    before = {name: getattr(srp, name) for name in V.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = K98._AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w139_layering_')
    cell = CELL3
    n = V5.CELLS[cell]['k0']
    pp = K98._fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    try:
        with V5.settling_resettle_hooks(scratch, V5.declaration_for(cell), holder, cap=V5.spec_cap(cell)) as st:
            installed = {name: getattr(srp, name) for name in V.WRAPPED}
            all_installed = all(installed[k] is not before[k] for k in V.WRAPPED)
            v5_state = isinstance(st, V5.ResettleStateV5) and isinstance(st.rule, V5.HookedRuleV5)
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
                          and v5_state and all(disp.values()) and summ.get('phase') == 'ended'
                          and summ.get('schema') == V5.SCHEMA
                          and summ.get('certificate_length_restored_at_exit') == 10 and not summ.get('errors')
                          and admm.minimum_consecutive_converged_cycles == 10),
            'nine_wrappers_installed': all_installed, 'v5_state_and_rule': v5_state,
            'appender_recorded_held_tail_value_after_k0': appender_saw_held,
            'production_functions_restored_on_exit': restored, 'idc_wraps_the_resettle_boyd_wrapper': idc_outer,
            'child_real_order_continuation_settling_extension_resettle_idc_s38_appender': order_ok,
            'child_real_dispatch_then_checklist_before_run_admm_arm': pre_ok, 'harness_dispatch': disp,
            'files_written_by_install_without_cycles': files_written, 'summary_phase': summ.get('phase'),
            'summary_schema': summ.get('schema'), 'summary_errors': summ.get('errors')}


def _h_exit_real_v5():
    """H13: the v5 exit wrapper over REAL production structures -- a fresh SRP1 planning object (data only; nothing built
    or solved): attempt records appended to its REAL Network deques by production's own
    `network._append_ipopt_solve_record` over synthetic IPOPT logs (a primary Acceptable at 2.51x on DSO7 2025 Winter; a
    TSO block with a maxIter primary + a recovery Acceptable; Optimal elsewhere), ESSO entries appended to its REAL
    lists (ESSO 7 Acceptable within 10x), real pyomo SolverResults, REAL `_admm_local_solves_succeeded`, the REAL
    in-memory tolerance reader. Then production's own drain returns every record (nothing consumed)."""
    import p56a_oracle as O
    import network as NET
    import shared_resources_planning as srp
    from pyomo.opt import SolverResults, SolverStatus, TerminationCondition
    eval_id = f'p515s53_w139_exit_check_{int(time.time())}'
    work = os.path.join(O.WORK_DIR, eval_id)
    tmp = tempfile.mkdtemp(prefix='w139_h13_')

    def result(msg, status=SolverStatus.ok, term=TerminationCondition.optimal):
        r = SolverResults()
        r.solver.status = status
        r.solver.termination_condition = term
        r.solver.message = msg
        return r
    try:
        planning = O.fresh_planning(eval_id)
        keys = V.block_keys(planning)
        years, days = list(planning.years), list(planning.days)
        dso7_key = f'DSO|7|{years[0]}|{days[3]}'
        tso_key = f'TSO|{years[2]}|{days[0]}'
        plan = {dso7_key: 'acc_2p51', tso_key: 'acc_2p51@recovery', 'ESSO|7': 'esso_acc_within'}

        def append_records(cycle):
            n_rec = 0
            for label, nd in srp._convergence_depth_tail_holders(planning):
                for y in nd.years:
                    for d in nd.days:
                        net = nd.network[y][d]
                        kt = f'TSO|{y}|{d}' if label == 'TSO' else f'DSO|{int(label[3:])}|{y}|{d}'
                        prof = plan.get(kt, 'opt')
                        base, _, tier = prof.partition('@')
                        path = os.path.join(tmp, f'{kt.replace("|", "_")}_c{cycle}.log')
                        chain = ([('maxit', None)] if tier else []) + [(base, tier or None)]
                        for p, suffix in chain:
                            off = os.path.getsize(path) if os.path.exists(path) else 0
                            with open(path, 'a') as handle:
                                handle.write(synthetic_log_text(p))
                            solver = SimpleNamespace(options={'tol': 1e-5, 'compl_inf_tol': 1e-6})
                            NET._append_ipopt_solve_record(net, nd.params, solver, path, off, suffix, True)
                            n_rec += 1
            return n_rec

        def stage_esso(cycle):
            sed = planning.shared_ess_data
            for n in planning.active_distribution_network_nodes:
                prof = plan.get(f'ESSO|{n}', 'esso_opt')
                path = os.path.join(tmp, f'optim_log_esso_node{n}_{"init" if cycle is None else f"cycle{cycle:03d}"}.txt')
                with open(path, 'w') as handle:
                    handle.write(synthetic_log_text(prof, with_options=False))
                sed.esso_complementarity_diagnostics.append({'node_id': n, 'log_path': path})

        def results_for():
            out = {'tso': {}, 'dso': {}, 'esso': {}}
            for key in keys:
                kind, n, y, d = key
                prof = plan.get(V._key_text(key), 'opt').split('@')[0]
                msg = _ACC if prof.startswith(('acc', 'esso_acc')) else _OPT
                r = result(msg)
                if kind == 'TSO':
                    out['tso'].setdefault(y, {})[d] = r
                elif kind == 'DSO':
                    out['dso'].setdefault(n, {}).setdefault(y, {})[d] = r
                else:
                    out['esso'][n] = r
            return out
        stale = len(srp._drain_network_ipopt_solve_records(planning, None))
        cell = CELL3
        st = V5.ResettleStateV5(V5.declaration_for(cell), None, V5.spec_cap(cell), reference={}, sink=[])
        w = V5.make_exit_wrapper(st, srp._admm_local_solves_succeeded, H.ipopt_exit_class, 'real')
        stage_esso(None)
        ok_init = w(planning, results_for())                       # the initialisation call (records already drained)
        n_rec = append_records(1)
        stage_esso(1)
        st.phase, st.cycle, st.cur = 'in_cycle', 1, {'cycle': 1}
        res_ = results_for()
        ok_prod = srp._admm_local_solves_succeeded(planning, res_)
        ok_wrapped = w(planning, res_)
        entries = st.cur['exit_clean_by_block']
        cur1 = st.cur
        drained = srp._drain_network_ipopt_solve_records(planning, 1)
        # a stale cycle (no records appended): the capture RAISES
        st.phase, st.cycle, st.cur = 'in_cycle', 2, {'cycle': 2}
        stage_esso(2)
        try:
            w(planning, results_for())
            stale_raises = False
        except RuntimeError as error:
            stale_raises = 'no attempt record' in str(error)
        tol_ok, tol_parts = V5.tolerance_state_in_memory(planning)
        e_dso7, e_tso, e_esso = entries[dso7_key], entries[tso_key], entries['ESSO|7']
        parts = {
            'value_unchanged': ok_wrapped == ok_prod is True and ok_init is True,
            'tolerance_state_in_memory_holds_on_fresh_srp1': tol_ok and bool((st.tolerance_in_memory or {}).get('ok')),
            'fifty_one_entries_in_production_order': list(entries) == [V._key_text(k) for k in keys],
            'dso7_primary_acceptable_2p51_clean': (e_dso7['class'] == 'acceptable' and e_dso7['attempt'] == 'primary'
                                                   and e_dso7['clean'] is True
                                                   and abs(e_dso7['max_ratio'] - 2.5123656330789977) < 1e-9),
            'tso_recovery_acceptable_not_clean': (e_tso['attempt'] == 'recovery' and e_tso['attempts'] == [
                'primary', 'recovery'] and e_tso['clean'] is False and e_tso['reason'] == 'acceptable_not_primary'),
            'esso7_acceptable_within_clean': (e_esso['class'] == 'acceptable' and e_esso['attempt'] == 'primary'
                                              and e_esso['clean'] is True),
            'all_clean_false_by_the_tso_recovery': cur1['all_clean_k'] is False and st.all_clean[1] is False,
            'non_clean_blocks_exactly_the_tso_recovery': cur1['non_clean_blocks'] == [tso_key],
            'metrics_equal_the_synthetic_values': e_dso7['metrics'] == profile_metrics('acc_2p51'),
            'production_record_tol_in_force_read': e_dso7['tol_in_force'] == 1e-5,
            'nothing_consumed_drain_returns_every_record': len(drained) == n_rec == 48 + 1,
            'stale_cycle_raises': stale_raises,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isdir(work) and not any(files for _r, _d, files in os.walk(work)):
            shutil.rmtree(work)
    return {'holds': all(parts.values()), 'parts': parts, 'stale_records_discarded_before': stale,
            'tolerance_parts': tol_parts, 'dso7_entry': e_dso7, 'tso_entry': e_tso, 'esso7_entry': e_esso}


def tests_H():
    import shared_resources_planning as srp
    res = {}
    h1 = {}
    for cell in V5.GATED_CELLS:
        s = _drive_summary(drive(cell, 'certify'))
        c = V5.CELLS[cell]
        s['ok'] = bool(s['raised'] is None and s['replay_bitwise_through'] == c['k0'] and s['first_pass'] == c['k0']
                       and s['first_k0_v2'] == c['k0'] and s['n_lapses'] == 1 + len(c['original_lapses_after_k0'])
                       and s['n_overlap'] == c['N_old'] - c['k0'] and s['overlap_all_zero']
                       and s['status'] == 'certified' and s['stopped_by'] == 'settling_rule'
                       and s['k_star'] is not None and s['k_star'] > c['N_old'] and s['decision_files'] == 1
                       and s['summary_ok'] and s['in_cycle_rule_equals_pure_replay'] and s['exit_capture_complete']
                       and s['clean_capture_complete'] and s['exits_51_every_line'] and s['clean_51_every_line']
                       and s['all_clean_every_line_is_bool'] and s['all_clean_is_the_conjunction']
                       and s['non_clean_cycles'] == [] and s['n_vetoes'] == 0 and s['certificate_length_after'] == 10
                       and s['lines'] == s['k_star'] == s['creep_lines'] and s['decision_version'] == 5
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
    ks = base['k_star']
    dso7 = f'DSO|7|{sorted(K118.YEARS)[0]}|Winter'
    probe = drive(cell, 'certify', plan={})
    keys = list(probe['files'][V.CYCLE_FILE][0]['ipopt_exit_by_block'])
    dso7 = next(k for k in keys if k.startswith('DSO|7|') and k.endswith('|Winter'))
    tso = next(k for k in keys if k.startswith('TSO|'))

    def one(name, plan, want_veto):
        s_ = _drive_summary(drive(cell, 'certify', plan=plan))
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
    s = _drive_summary(d4)
    rec = {'status': 'certified', 'barrier': False, 'barrier_cause': None, 'certified_cost': 1.0,
           'certification_cycle': s['last_cycle'], 'terminal_gross_operational_cost': 1.0,
           'settling_resettle_summary': s['summary']}
    relabelled = H._apply_settling_resettle_status(dict(rec))
    summ4 = s.pop('summary')
    s['ok'] = bool(s['raised'] is None and s['status'] == 'uncertified' and s['stopped_by'] == 'cap'
                   and s['last_cycle'] == V5.spec_cap('c_156ce2d1') and s['certificate_length_after'] == 10
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
                   and s['last_cycle'] == 50 + V5.CAP_AFTER_K0 == s['rule_cap'] and s['first_pass'] == 50
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
    res['H7_aa_hold_real_production'] = _h_real_v5(srp, 'aa')
    res['H8_tail_hold_real_production'] = _h_real_v5(srp, 'tail')
    res['H9_rho_hold_real_production'] = _h_real_v5(srp, 'rho')
    res['H10_layering_real_install_and_dispatch'] = _h_layering(srp)
    got = V._certificate_length_writes_in_source()
    in_cycle_v5_src = ''.join(inspect.getsource(o) for o in (V5.HookedRuleV5, V5.ResettleStateV5, V5._v5_rule,
                                                              V5.settling_resettle_hooks, V5.make_exit_wrapper, SC5))
    res['H11_certificate_length_writes'] = {
        'holds': (got == sorted(R.CERTIFICATE_LENGTH_WRITES) and 'early_stop' not in in_cycle_v5_src
                  and 'minimum_consecutive_converged_cycles' not in in_cycle_v5_src),
        'writes_in_source': got}
    for name, fn in (('H12_w132_exit_wrapper_on_real_production_reused', K132._h_exit_real),
                     ('H13_v5_exit_wrapper_on_real_production_structures', _h_exit_real_v5)):
        try:
            res[name] = fn()
        except Exception as error:  # noqa: BLE001
            res[name] = {'holds': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return {'holds': all(v.get('holds') is True for v in res.values()), 'tests': res}


# ======================================================================================================================
#  C -- the clean capture checklist
# ======================================================================================================================
def tests_C():
    cl = V5.clean_capture_checklist(1e-6)
    parts = {k: bool(v) for k, v in cl.items()}
    parts['tail_value_other_than_the_table_refused'] = V5.clean_capture_checklist(1e-4)[
        'tolerance_table:network_compl_inf_tol_is_the_spec_tail_value'] is False
    # parse_final_summary on the recorded DSO7 log segment of W133's cell 1 (the committed network record, cycle 120)
    ev = K137.CELL1_V3['eval_dir']
    recs = _read_jsonl(_abs(os.path.join(ev, 'network_ipopt_solve_records.jsonl')))
    r120 = [r for r in recs if r['round'] == 120 and r['agent'] == 'DSO7' and str(r['year']) == '2025'
            and r['day'] == 'Winter']
    parsed = V5.parse_final_summary(V5.read_segment_tail(r120[-1]['log_path'], *r120[-1]['log_bytes'])) if r120 else {}
    want = {'overall_nlp_error': 1.141920220071313e-05, 'dual_infeasibility': 0.01141920220071313,
            'constraint_violation': 7.670947210769441e-14, 'complementarity': 2.5123656330789977e-06}
    parts['parse_final_summary_on_the_recorded_dso7_cycle_120_segment'] = (parsed.get('metrics') == want
                                                                           and parsed.get('exit')
                                                                           == 'Solved To Acceptable Level.')
    for name, text in (('no_summary', 'EXIT: Optimal Solution Found.\n'),
                       ('partial_summary', 'Number of Iterations....: 3\nDual infeasibility......:   1e-3    1e-2\n')):
        parts[f'parse_{name}_gives_no_metrics'] = V5.parse_final_summary(text)['metrics'] is None
    try:
        V5.network_block_capture([], 'optimal')
        parts['empty_chain_raises'] = False
    except RuntimeError:
        parts['empty_chain_raises'] = True
    try:
        V5.network_block_capture([{'attempt': 'recovery'}], 'optimal')
        parts['chain_not_starting_primary_raises'] = False
    except RuntimeError:
        parts['chain_not_starting_primary_raises'] = True
    tmp = tempfile.mkdtemp(prefix='w139_c_')
    try:
        p = os.path.join(tmp, 'x.log')
        with open(p, 'w') as handle:
            handle.write(synthetic_log_text('acc_2p51'))
        chain = [{'attempt': 'primary', 'log_path': p, 'log_bytes': [0, os.path.getsize(p)],
                  'exit': 'Solved To Acceptable Level.', 'tol_in_force': 1e-4, 'compl_inf_tol_in_force': 1e-6}]
        try:
            V5.network_block_capture(chain, 'acceptable')
            parts['tol_in_force_other_than_the_table_raises'] = False
        except RuntimeError:
            parts['tol_in_force_other_than_the_table_raises'] = True
        chain[0]['tol_in_force'] = 1e-5
        try:
            V5.network_block_capture(chain, 'optimal')
            parts['log_exit_class_other_than_the_result_raises'] = False
        except RuntimeError:
            parts['log_exit_class_other_than_the_result_raises'] = True
        e = V5.esso_block_capture(7, 3, 'acceptable', [{'node_id': 7, 'log_path': os.path.join(
            tmp, 'optim_log_esso_node7_cycle004.txt')}], [])
        parts['esso_log_of_another_cycle_raises'] = False
    except RuntimeError:
        parts['esso_log_of_another_cycle_raises'] = True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        V5.esso_block_capture(7, 3, 'acceptable', [], [])
        parts['esso_accepted_without_its_entry_raises'] = False
    except RuntimeError:
        parts['esso_accepted_without_its_entry_raises'] = True
    return {'holds': all(parts.values()), 'parts': parts, 'checklist': cl,
            'dso7_cycle_120_parsed': parsed.get('metrics'), 'dso7_record': {k: (r120[-1] if r120 else {}).get(k) for k in (
                'attempt', 'log_bytes', 'exit', 'tol_in_force', 'compl_inf_tol_in_force')}}


def capture_replay_from_records(eval_rel, lines=None):
    """THE CAPTURE FUNCTIONS REPLAYED ON A COMMITTED RUN (zero solves): for every cycle of a committed v4 run, the
    in-cycle v5 capture (`V5.network_block_capture` / `V5.esso_block_capture`) fed what production held at the exit
    wrapper -- the network attempt chain of the round (the committed network_ipopt_solve_records.jsonl, grouped by round
    in order) and the ESSO complementarity entries of the cycle (the committed leak_classification_s39_D.jsonl: node,
    cycle, log path of the loaded attempt) -- with the exit classes of the run's own in-cycle capture. Returns
    ({cycle: {block: entry}}, {cycle: all_clean}); raises on any capture error (as the wrapper would)."""
    lines = lines if lines is not None else {x['cycle']: x for x in _read_jsonl(_abs(os.path.join(eval_rel,
                                                                                                    V.CYCLE_FILE)))}
    recs = _read_jsonl(_abs(os.path.join(eval_rel, 'network_ipopt_solve_records.jsonl')))
    chains = {}
    for r in recs:
        key = (f"TSO|{r['year']}|{r['day']}" if r['agent'] == 'TSO'
               else f"DSO|{int(r['agent'][3:])}|{r['year']}|{r['day']}")
        chains.setdefault(r['round'], {}).setdefault(key, []).append(r)
    leak = _read_jsonl(_abs(os.path.join(eval_rel, 'leak_classification_s39_D.jsonl')))
    events = _read_jsonl(_abs(os.path.join(eval_rel, 'esso_recovery_events_s39_D.jsonl')))
    if events:
        raise RuntimeError(f'{eval_rel}: ESSO recovery events present; this replay covers runs without ESSO recoveries')
    comp = {}
    for e in leak:
        comp.setdefault(e['cycle'], []).append({'node_id': e['node_id'], 'log_path': e['log_path']})
    out, all_clean = {}, {}
    for k in sorted(lines):
        by_block = lines[k]['ipopt_exit_by_block']
        ent = {}
        for key, v in by_block.items():
            if key.startswith('ESSO'):
                ent[key] = V5.esso_block_capture(int(key.split('|')[1]), k, v['class'], comp.get(f'{k:03d}', []), [])
            else:
                ent[key] = V5.network_block_capture(chains.get(k, {}).get(key) or [], v['class'])
        out[k] = ent
        all_clean[k] = all(e['clean'] for e in ent.values())
    return out, all_clean


def tests_C2():
    """The capture replayed on the committed v4 cell-3 run (0734f103): it runs without a capture error on every block of
    every cycle and reproduces the committed item-5 table (the non-clean cycles; every non-Optimal final exit's tier,
    reason and metrics); the v5 rule over its flags certifies where the table says."""
    import p515_s53_w139_v5_from_records as F
    doc, sha = _from_records_doc()
    rep = doc['reports'][CELL3_V4_RECORD]
    eval_rel = rep['eval_dir']
    t0 = time.time()
    entries, all_clean = capture_replay_from_records(eval_rel)
    wall = time.time() - t0
    nonclean = sorted(k for k, v in all_clean.items() if not v)
    table = {(e['cycle'], e['block']): e for e in rep['non_optimal_final_exits']}
    mism = []
    for (k, blk), e in table.items():
        got = entries[k][blk]
        if (got['attempt'], got['clean'], got['reason'], got['metrics']) != (e['attempt'], e['clean'], e['reason'],
                                                                                 e['metrics']):
            mism.append({'cycle': k, 'block': blk, 'capture': {x: got[x] for x in ('attempt', 'clean', 'reason')},
                         'table': {x: e[x] for x in ('attempt', 'clean', 'reason')}})
    extra_nonopt = [(k, b) for k, ent in entries.items() for b, e in ent.items()
                    if e['class'] != SC5.OPTIMAL_CLASS and (k, b) not in table]
    with contextlib.redirect_stdout(io.StringIO()):
        recs = F.record_inputs()
    r = recs[CELL3_V4_RECORD]
    _o, d5, _r5 = _replay5(r['q'], r['b'], r['t'], all_clean, dict(r['kw'], last=r['n']))
    parts = {'every_cycle_captured': sorted(entries) == list(range(1, r['n'] + 1))
             and all(len(v) == 51 for v in entries.values()),
             'non_clean_cycles_equal_the_table': nonclean == rep['non_clean_cycles'],
             'every_non_optimal_exit_equals_the_table': not mism and not extra_nonopt,
             'v5_from_the_capture_equals_the_table': (d5 or {}).get('k_star') == rep['v5_from_records']['k_star']
             and (d5 or {}).get('status') == rep['v5_from_records']['status']}
    return {'holds': all(parts.values()), 'parts': parts, 'record': CELL3_V4_RECORD, 'eval_dir': eval_rel,
            'from_records_sha256': sha, 'wall_s': wall, 'mismatches': mism[:10], 'extra_non_optimal': extra_nonopt[:10],
            'v5_from_capture': _summ(d5), 'n_acceptable_clean_blocks': sum(
                1 for ent in entries.values() for e in ent.values() if e['reason'] == 'acceptable_primary_within_factor')}


# ======================================================================================================================
#  K -- keys
# ======================================================================================================================
def _harness_pre_w139():
    return K137._harness_at(PRE_W139_HARNESS['commit'], PRE_W139_HARNESS['sha256'], '_w139_pre_harness')


def resettle_keys(pre=None):
    """The 36 v5 keys: formula over the (pre-W139) base key; the v3 and v4 keys of the same cell beside."""
    out = {}
    for cell in V5.CELL_ORDER:
        _spec, e, kw = K132.resettle_kwargs(cell)
        base = H.evaluation_key(e['key'], e['overrides'], **kw)
        decl = V5.declaration_for(cell)
        key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=decl, **kw)
        key_v4 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V4.declaration_for(cell), **kw)
        key_v3 = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V.declaration_for(cell), **kw)
        base_pre = pre.evaluation_key(e['key'], e['overrides'], **kw) if pre is not None else base
        formula = hashlib.sha256(json.dumps({'base_evaluation_key': base_pre, 'settling_resettle': decl},
                                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        out[cell] = {'candidate_key': e['key'], 'original_eval_key': e['eval_key'], 'base_key_now': base,
                     'resettle_key': key, 'v4_resettle_key': key_v4, 'v3_resettle_key': key_v3,
                     'formula_holds': key == formula, 'base_equals_pre_w139': base == base_pre,
                     'differs_from_original': key != e['eval_key'] and base != e['eval_key'],
                     'differs_from_v4_and_v3': key not in (key_v4, key_v3)}
    return out


def tests_K():
    import importlib
    ext = importlib.import_module(V5.EXT_HOOKS_MODULE)
    pre, pre_sha, _src = _harness_pre_w139()
    n_specs = n_entries = n_equal = n_frozen_equal = 0
    own = {'v5': {'n': 0, 'ok': 0, 'bad': []}, 'ext_v5': {'n': 0, 'ok': 0, 'bad': []}}
    fam_count = {'w118': 0, 'v3': 0, 'v4': 0, 'w135': 0}
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
                if isinstance(rs, dict) and rs.get('schema') == V5.DECLARATION_SCHEMA:
                    fam, mod = 'v5', V5
                elif isinstance(rs, dict) and rs.get('schema') == V5.EXT_DECLARATION_SCHEMA:
                    fam, mod = 'ext_v5', ext
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
                if V4.is_v4_declaration(rs):
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
    outside = {cell: K137._outside_root(h, W139_ROOT_REL) for cell, h in holders.items()}
    probe = keys[V5.CELL_ORDER[0]]['resettle_key']

    def planted(rel):
        return (rel, {'campaign_id': 'w139_planted_control', 'candidates': [{'label': 'planted', 'key': '0' * 64,
                                                                             'eval_key': probe}]})
    ctl = {}
    for name, rel in (('outside', os.path.join(_P53, 'w139_planted_negative_control', 'campaign_planted',
                                               'campaign_spec_planted_00000000.json')),
                      ('sibling_prefix', os.path.join(W139_ROOT_REL + '_planted_sibling', 'campaign_planted',
                                                      'campaign_spec_planted_00000000.json')),
                      ('ext_v5_root', os.path.join(W139_EXT_ROOT_REL, 'campaign_planted',
                                                   'campaign_spec_planted_00000000.json')),
                      ('v4_root', os.path.join(W137_ROOT_REL, 'campaign_planted', 'campaign_spec_planted_00000000.json')),
                      ('w135_root', os.path.join(W135_ROOT_REL, 'campaign_planted',
                                                 'campaign_spec_planted_00000000.json'))):
        ctl[name] = {'rel': rel, 'refused': rel in K137._outside_root(K137._key_holders(scanned + [planted(rel)], probe),
                                                                      W139_ROOT_REL)}
    own_rel = os.path.join(W139_ROOT_REL, f'campaign_{CAMPAIGN_ID_PREFIX}{V5.CELL_ORDER[0]}',
                           f'campaign_spec_{CAMPAIGN_ID_PREFIX}{V5.CELL_ORDER[0]}_00000000.json')
    own_holders = K137._key_holders(scanned + [planted(own_rel)], probe)
    try:
        pre.validate_settling_resettle(V5.declaration_for(CELL3))
        pre_refuses = False
    except ValueError:
        pre_refuses = True
    ss4 = json.load(open(_abs(V4_STAGE_SPEC['path'])))
    v4_frozen = {c: json.load(open(_abs(ss4['pins']['campaign_specs'][c]['path'])))['candidates'][0]['eval_key']
                 for c in V5.CELL_ORDER}
    parts = {
        'key_regression_no_mismatch': not mismatch, 'key_regression_no_errors': not errors,
        'key_regression_every_other_entry_equal': (n_equal == n_entries - own['v5']['n'] - own['ext_v5']['n']
                                                   and n_entries > 0),
        'w118_entries_scanned_and_equal': fam_count['w118'] >= 10,
        'v3_38_entries_scanned_and_equal': fam_count['v3'] == 38,
        'v4_38_entries_scanned_and_equal': fam_count['v4'] == 38,
        'w135_7_entries_scanned_and_equal': fam_count['w135'] == 7,
        'own_v5_entries_inside_the_v5_root_follow_the_formula': own['v5']['ok'] == own['v5']['n'] and not own['v5']['bad'],
        'own_ext_v5_entries_inside_the_ext_root_follow_the_formula': (own['ext_v5']['ok'] == own['ext_v5']['n']
                                                                      and not own['ext_v5']['bad']),
        'v5_formula_holds_every_cell': all(v['formula_holds'] for v in keys.values()),
        'v5_base_equals_pre_w139_every_cell': all(v['base_equals_pre_w139'] for v in keys.values()),
        'v5_keys_differ_from_originals': all(v['differs_from_original'] for v in keys.values()),
        'v5_keys_differ_from_the_v4_and_v3_keys': all(v['differs_from_v4_and_v3'] for v in keys.values()),
        'v4_keys_recomputed_equal_the_frozen_v4_specs': all(v4_frozen[c] == keys[c]['v4_resettle_key']
                                                            for c in V5.CELL_ORDER),
        'v5_keys_distinct': len({v['resettle_key'] for v in keys.values()}) == len(keys) == 36,
        'v5_keys_absent_from_committed_specs_outside_the_v5_root': not any(outside.values()),
        **{f'control_planted_{k}_refused': v['refused'] for k, v in ctl.items()},
        'control_own_campaign_spec_accepted': own_rel in own_holders and own_rel not in K137._outside_root(
            own_holders, W139_ROOT_REL),
        'control_v5_declaration_refused_by_the_pre_w139_harness': pre_refuses,
    }
    return {'holds': all(v is True for v in parts.values()), 'parts': parts,
            'pre_w139_harness': {**PRE_W139_HARNESS, 'sha256_loaded': pre_sha},
            'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries, 'entries_equal': n_equal,
            'entries_whose_frozen_eval_key_equals_the_recomputed_REPORTED': n_frozen_equal,
            'families_scanned': fam_count,
            'own_entries': {k: {'n': v['n'], 'ok': v['ok'], 'bad': v['bad'][:20]} for k, v in own.items()},
            'own_roots': KEY_OWN_ROOTS, 'mismatches': mismatch[:20], 'errors': errors[:20], 'resettle_keys': keys,
            'v5_keys_in_committed_specs_all_REPORTED': holders, 'controls': ctl,
            'control_own': {'rel': own_rel, 'holders': own_holders}}


# ======================================================================================================================
#  P -- preconditions and validator
# ======================================================================================================================
def tests_P():
    out = {}
    ok = True
    tail = {'convergence_depth_tail': {'enabled': True, 'compl_inf_tol': 1e-6}}
    for cell in V5.CELL_ORDER:
        decl = V5.declaration_for(cell)
        cap = V5.spec_cap(cell)
        try:
            good = V5.assert_resettle_preconditions(decl, {'cap': cap, 'configuration': tail},
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
                V5.assert_resettle_preconditions(decl, spec, tl, aa)
                negatives[name] = 'NOT refused'
            except RuntimeError as error:
                negatives[name] = f'refused: {str(error)[:200]}'
        rule = decl['settling_rule']
        bad = {'early_stop_key': dict(decl, early_stop={'abs_gross_step_below_eur': 500.0}),
               'extra_key': dict(decl, extra=1),
               'unknown_cell': dict(decl, cell='x0'),
               'cell_kept_under_v4': dict(decl, cell='b_2a0ba8b2'),
               'no_schema': {k: v for k, v in decl.items() if k != 'schema'},
               'v4_schema_with_v5_rule': dict(decl, schema=V4.DECLARATION_SCHEMA),
               'p_max_22': dict(decl, settling_rule=dict(rule, p_max=22, l_mono=44)),
               'rule_v4': dict(decl, settling_rule=V4.settling_rule_declaration()),
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
                V5.validate_settling_resettle(d)
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
    _pre, _sha_pre, pre_src = _harness_pre_w139()
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
    i118 = fn_src.find('import p515_s53_w118_resettle_hooks as W118C')
    sha_now = _sha('p515_s44_campaign_harness.py')
    last = H._git(['log', '--format=%H %s', '-1', '--', 'p515_s44_campaign_harness.py'])
    pinned = harness_post_v5_sha256()
    parts = {
        'no_line_removed': removed == [],
        'exactly_three_lines_added': len(added) == 3,
        'the_three_added_are_the_v5_branch': tuple(added) == V5_BRANCH_LINES,
        'order_v3_w135_v4_v5_then_w118': 0 <= i132 < i135 < i137 < i139 < i118,
        'harness_committed_clean': _clean('p515_s44_campaign_harness.py'),
        'harness_sha256_is_the_pinned_post_v5_sha256': pinned is not None and sha_now == pinned,
        'last_harness_commit_message_carries_the_sha': pinned is not None and pinned in last,
        'dispatch_six_families': all(dispatch_checks().values()),
    }
    return {'holds': all(parts.values()), 'parts': parts, 'added': added, 'removed': removed,
            'harness_sha256_now': sha_now, 'pre_w139_harness': PRE_W139_HARNESS, 'post_v5_sha256_pinned': pinned,
            'last_harness_commit': last}


def tests_D():
    return harness_router_check()


# ======================================================================================================================
SECTIONS = (('V', tests_V), ('R', tests_R), ('O', tests_O), ('S', tests_S), ('H', tests_H), ('C', tests_C),
            ('C2', tests_C2), ('K', tests_K), ('P', tests_P), ('X', tests_X), ('D', tests_D))

CODE_PINNED_BY_CHECKS = (os.path.basename(__file__), 'settling_criterion_v5.py', 'settling_criterion_v4.py',
                         'settling_criterion_v3.py', 'settling_criterion_v2.py', 'settling_criterion.py',
                         'p515_s53_w139_resettle_v5_hooks.py', 'p515_s53_w139_resettle_ext_v5_hooks.py',
                         'p515_s53_w139_v5_from_records.py',
                         'p515_s53_w137_resettle_v4_hooks.py', 'p515_s53_w137_resettle_v4_checks.py',
                         'p515_s53_w137_recurring_acceptable.py', 'p515_s53_w131_prefreeze_diagnostics.py',
                         'p515_s53_w132_resettle_v3_hooks.py', 'p515_s53_w132_resettle_v3_checks.py',
                         'p515_s53_w135_resettle_ext_hooks.py',
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
    return {'all_hold': ok, 'p_max': V5.P_MAX, 'constants': SC5.constants(V5.P_MAX), 'readings': SC5.READINGS,
            'sub_test_reads': SC5.SUB_TEST_READS, 'clean_rule': {'factor': SC5.CLEAN_FACTOR,
                                                                  'metric_table': SC5.METRIC_TABLE,
                                                                  'tolerances': SC5.TOLERANCES,
                                                                  'tolerance_sources': SC5.TOLERANCE_SOURCES},
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
    doc = {'schema': 'p515_s53_w139_zero_solve_checks_v1',
           'task': 'W139 (PLANNER_BRIEF_2026-09-13.md Addendum 60)',
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
        print(f"[W139-CHECKS] {sid}: holds={r['holds']} wall={r['wall_s']:.1f}s"
              + (f" error={r['result'].get('error')}" if isinstance(r['result'], dict) and r['result'].get('error') else ''))
    print(f"[W139-CHECKS] W100 typing test: pass={typing['pass']} exit={typing['exit_code']} wall={typing['wall_s']:.0f}s")
    print(f"[W139-CHECKS] all_hold={res['all_hold']} (with typing {doc['all_hold_including_typing_test']}) guards={guards}")
    print(f"[W139-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values())
    sys.exit(0 if (doc['all_hold_including_typing_test'] and guards_ok) else 1)


if __name__ == '__main__':
    main()
