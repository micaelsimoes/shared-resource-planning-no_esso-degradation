"""P5.15 S56 2x PHASE -- ZERO-SOLVE TESTS of the phase build (TASKS.md "Author decisions on report (i)" 2026-11-24,
step 5a; engineering build: a script with pass/fail checks). Every value is SYNTHETIC (landscape or stub children).

Guards (installed before any repository import): SolveProfileGuard(permitted=()), verify(0) at the end; pickle.load /
loads blocked and counted (0). Every driver subprocess arms its own guard (`p515_s56_search_driver.main`).

TESTS
  T1  pure units: classify_settled boundaries; efficiency-rule boundary (11 / 22 vs 12 / 22, uncertified rows in the
      denominator); candidate selection (distinct node AND year, confirm-candidate only, fewer logged); the ladder rule
      (best duration, two best node-years, the inadmissible 1.0 MVA x 4 h 2025 point skipped); the DEFAULT search path
      byte-identical to the policy at PRE_PHASE_REV 6b1d98b2 (the commit before the phase build; v1 compared against
      HEAD, vacuous after the build commit) on synthetic searches from three starts, both class ranks; the plain
      synthetic landscape byte-identical to 6b1d98b2's; both comparisons asserted non-vacuous; the capture checklist.
  T2  the whole phase through the launch CLI (synthetic oracle), exit-then-confirm branch, a search that moves:
      screen -> ladder -> table -> candidates -> polls -> pool -> search -> results; codes 4...4,0; the final state
      byte-equal to the in-process phase; stage order; every launch <= 5 evaluations at one depth.
  T3  the settled-direct branch through the CLI: exit 0; every poll evaluation after the efficiency decision settled;
      byte-equal to in process; count identities hold.
  T4  the 120 h ceiling (a test ceiling, simulated walls): exit 6 before the first block that would cross it; the
      record's arithmetic; nothing launched; a re-issue stops again (6) with no new launch.
  T5  review stops (exit 5, reason in the state): harness_error in the screen; offset above e at a candidate
      confirmation; offset above e at a search confirmation; the not_settled review trigger (settled-direct candidate
      poll); start not certified (x0 exit uncertified).
  T6  kill-and-resume at several transitions (one per stage): final state and results state byte-equal to T2's
      uninterrupted run; killed evaluations uncounted; stale lock taken over.
  T7  the author's command (`phase_author_command`, the exact text with a scratch --lock-path) on a synthetic root:
      the loop prints each exit code, stops at 0, one log pair per invocation; re-issued on the finished root: exit 0.
  T8  stub children (the harness `evaluate` path, harness stub mode, 1x test configuration): the whole phase, exit 0,
      byte-equal to the in-process synthetic phase.
  T9  refusals: real phase freeze without predictions; synthetic phase not test-only; --preflight-only on a synthetic
      phase; a tampered launch record (replay refuses, nothing written); an existing root (write-once).
SPEC v2 TESTS (Advisor pre-launch review; Planner decisions R1, R4, P1, P2; TASKS.md step 5a, 2026-11-24):
  T10 change 1 (P1): the efficiency rule decides on the share of rows NOT classed worse (confirm_candidate +
      unresolved); 11/22 vs 12/22; uncertified rows in the denominator, not the numerator; per-class counts; the v1
      band fraction recorded beside it and read by no decision (rows where the two measures disagree, both ways).
  T11 change 3 (R4): the exit-view check -- bitwise (one ulp in F or bar, read_cycle + 1, an absent view: mismatch);
      in the phase: a mismatch stops with exit_view_mismatch at candidate_confirmation (CLI: exit 5); a first-in-order
      harness_error precedes it; it precedes the offset stop; mode 'record' records and continues; equal on T2.
  T12 change 4 (P2): the un-launched predecessor: verification (spec + freeze logs only; .DS_Store ignored; refused
      with launches/, an eval dir, a state view, a wrong sha); the collision scan excludes a declared verified root,
      refuses an undeclared one, refuses when the declared one has launches; on the REAL artefacts: the real v2 spec's
      keys collide with the v1 root when it is not declared, and the scan passes when it is.
  T13 change 5 (P2): continuation on stubs -- after exit 6 (raised ceiling, carried hours) and after exit 5
      (exit_view_mismatch, --no-pilot-exit-views: x0 re-run at exit depth, the check recorded); decisions equal to the
      uninterrupted stub phase; no record re-requested; refusals (incomplete predecessor launch, a completed predecessor,
      a wrong sha, a synthetic oracle, a raised ceiling without a continuation, an un-launched root as continuation).
  T14 change 6: code drift -- a spec whose frozen code hash differs: --phase-run and --preflight-only refuse (exit 2)
      naming the file, nothing written.
  T15 change 7: files named .DS_Store under a phase root (top level, launches/, a launch dir, evals/, an interrupted
      launch) are neither manifested nor foreign nor killed evaluations; the phase proceeds; a root holding only one
      can be frozen.
  T16 change 8: `s56_phase_check.py` on crafted launch records (PASS, each FAIL criterion, missing record); the
      single-invocation command (launch 001 logs, refuses a re-run) followed by the loop continuing at 002; no command
      line contains the harness hazard string.
1x PHASE TESTS (TASKS.md step 5d, 2026-11-26; the declared price level --phase-flex 1.0):
  T17 2x option-off identity: the driver at PRE_1X_REV (8fe07ca5, the commit before the 1x build) loaded beside the
      working tree; the REAL 2x spec (as v2: non-continuation; as v3: continuation of v2 with v1 un-launched) and a
      test-only stub spec built by both are equal in every field except frozen_utc, code_sha256 and git_head; the
      collision-scan outcomes equal; non-vacuous (the two driver files differ).
  T18 the REAL 1x spec built in process: arm ev_1x, ceiling 60, exit-view check stop, no pilot exit view, no cache
      source, the 1x authority and price-level record, Q1-Q9 required, the 1x capture items; it differs from the 2x spec
      only in the declared fields (flat path diff); every one of the 74,084 eval keys is the 1.0-multiplier key (1.0 is
      the absent key by the harness rule) and none is a 2x key; the harness freezes the real 1x configuration (scratch
      dir outside the repository; keys recompute); the freeze collision scan over the tracked artefacts and over every
      phase / pilot launch artefact on disk: 0 collisions; refusals (a 1x ceiling other than 60, a price level outside
      {1.0, 2.0}, a 1x spec serving pilot exit views, a 1x continuation of a 2x root, the CLI --phase-flex 1.5).
  T19 the 1x flow (synthetic and stub children, test-only, --phase-flex 1.0 --no-pilot-exit-views): nothing
      precached; launch 1 = x0's exit run alone; x0 run at exit and settled, the unit and the staged plan settled (the
      unit's screen row an exit run), no pool member served from cache; exit 0; state byte-equal to in process; the test
      ceiling defaults to 60; the author commands for the real 1x root carry no hazard string.

Launch (attached, both streams captured):
  python -u p515_s56_phase_tests.py --out <new dir> > <dir>_stdout.log 2> <dir>_stderr.log
Exit: 0 every check holds; 3 a check failed; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 phase tests (never solve)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('phase tests: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('phase tests: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
CHECKS = []
# The pre-phase revision (the last commit before the 2x phase build 637090ba): the default-path identity checks of T1
# compare against THIS revision, never HEAD (v1 of these tests compared against HEAD, which after the build commit is
# the new code itself -- a vacuous comparison; v2 fixes it and asserts the two files differ).
PRE_PHASE_REV = '6b1d98b2'
# The pre-1x revision (TASKS 5d): the 2x option-off identity of T17 compares the working tree against THIS revision.
PRE_1X_REV = '8fe07ca5'
PHASE1X_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'phase1x_ev_v1')


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(test, name, ok, detail=None):
    CHECKS.append({'test': test, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{test}] {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


# ======================================================================================================================
#  fixtures
# ======================================================================================================================
def staged_z(lat):
    x0 = lat.x0()
    return lat.with_pair(lat.with_pair(lat.with_pair(lat.with_pair(x0, 5, 2030, 2, 2), 7, 2025, 1, 2), 7, 2030, 1, 1),
                         9, 2035, 1, 1)


def configs(lat):
    st = staged_z(lat)
    t_move = lat.with_pair(st, 9, 2035, 1, 2)
    t_direct = lat.with_pair(lat.with_pair(st, 9, 2035, 2, 4), 5, 2035, 1, 2)
    exit_move = {'landscape': {'seed': 'm1', 'alpha': 30000.0, 'target': list(t_move)}}
    direct = {'landscape': {'seed': 'm2', 'alpha': 3000.0, 'target': list(t_direct)}}
    return {'exit_move': exit_move, 'direct': direct, 't_move': t_move, 't_direct': t_direct}


def with_knobs(tc, **knobs):
    tc = copy.deepcopy(tc)
    tc['landscape'].setdefault('knobs', {}).update(knobs)
    return tc


def drv(argv, data_root, log_prefix, env_extra=None):
    env = dict(os.environ)
    env.pop(D.KILL_ENV, None)
    env.update(env_extra or {})
    with open(log_prefix + '_stdout.log', 'w') as so, open(log_prefix + '_stderr.log', 'w') as se:
        p = subprocess.run([sys.executable, '-u', os.path.join(CODE_ROOT, 'p515_s56_search_driver.py'), *argv,
                            '--data-root', data_root], stdout=so, stderr=se, env=env, cwd=CODE_ROOT,
                           start_new_session=True)
    return p.returncode


def freeze(base, tag, data_root, tc, oracle='synthetic', phase_id=None, extra=()):
    root = os.path.join(base, tag)
    logs = os.path.join(base, f'{tag}_logs')
    os.makedirs(logs)
    tcp = os.path.join(base, f'{tag}_test_config.json')
    D.write_once_json(tcp, tc)
    rc = drv(['--phase-freeze', '--root', root, '--phase-id', phase_id or f'ph_{tag}'[:40], '--oracle', oracle,
              '--test-only',
              '--phase-test-config', tcp, *extra], data_root, os.path.join(logs, 'freeze'))
    return root, logs, rc


def run_phase(root, data_root, lock, logs, kill=None, max_launches=400, env_extra=None, stop_codes=None):
    """Re-issue --phase-run until the code is not 4 (the author's loop). `kill` = (transition, launch[, delay])."""
    codes, killed = [], False
    for i in range(max_launches):
        extra = dict(env_extra or {})
        if kill and not killed:
            extra[D.KILL_ENV] = f'{kill[0]}:{kill[1]}:{kill[2] if len(kill) > 2 else 0.3}'
        argv = ['--phase-run', '--root', root, '--lock-path', lock]
        if killed:
            argv.append('--take-over-stale-lock')
        rc = drv(argv, data_root, os.path.join(logs, f'run_{i:03d}'), extra)
        codes.append(rc)
        if rc == -9:
            killed = True
            continue
        if rc != D.EXIT_SUSPENDED:
            return rc, codes
    return None, codes


def final(root):
    vp = os.path.join(root, D.PHASE_STATE_VIEW)
    view = D.load_json(vp) if os.path.exists(vp) else None
    rp = os.path.join(root, D.PHASE_RESULTS_FILE)
    res = D.load_json(rp) if os.path.exists(rp) else None
    return {'state': P.canonical_json(view['state']) if view else None,
            'n_launches': view['n_launches'] if view else None,
            'results_state': P.canonical_json(res['state']) if res else None, 'results': res}


def launch_records(root):
    n = D._complete_indices(root)[1]
    return [D.load_json(D.launch_paths(root, k)[1]) for k in n]


def in_process(root, lat):
    spec = D.load_phase_spec(root)[2]
    st, _rec = D.phase_in_process(spec, lat)
    return P.canonical_json(st), st


# ======================================================================================================================
#  T1 pure units
# ======================================================================================================================
def _se(F, S=100.0, PR=0.0, G=0.0, A=0.0, gates=True, status='settled'):
    return {'depth': 'settled', 'status': status, 'F': F, 'V': {'S': S, 'PR': PR, 'G': G, 'A': A},
            'gates_pass': gates}


def _load_head_module(name, rel, out_dir, rev=PRE_PHASE_REV):
    src = subprocess.run(['git', 'show', f'{rev}:{rel}'], cwd=CODE_ROOT, capture_output=True, text=True, check=True)
    path = os.path.join(out_dir, f'pre_{rev}_{rel}')
    with open(path, 'w') as h:
        h.write(src.stdout)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, D.sha256_file(path)


def t1(lat, data_root, out_dir):
    R = P.R_EUR
    inc = _se(800e6)
    cases = {
        'dF = -R exactly -> improver': (_se(800e6 - R), 'improver'),
        'dF = -R + 0.01 -> unresolved': (_se(800e6 - R + 0.01), 'unresolved'),
        'dF = +R exactly -> worse': (_se(800e6 + R), 'worse'),
        'V_pair = R/4 exactly -> improver': (_se(800e6 - 2 * R, S=R / 4 - 100.0), 'improver'),
        'V_pair > R/4 -> unresolved': (_se(800e6 - 2 * R, S=R / 4 - 100.0 + 1e-6), 'unresolved'),
        'A sum = R/2 exactly -> improver': (_se(800e6 - 2 * R, A=R / 2), 'improver'),
        'A sum > R/2 -> unresolved': (_se(800e6 - 2 * R, A=R / 2 + 1e-6), 'unresolved'),
        'a gate fails -> unresolved': (_se(800e6 - 2 * R, gates=False), 'unresolved'),
        'not_settled -> not_settled': (_se(800e6 - 2 * R, status='not_settled'), 'not_settled'),
        'harness_error -> harness_error': (_se(None, status='harness_error'), 'harness_error'),
        'unknown status -> harness_error': (_se(800e6, status='certified'), 'harness_error'),
        'missing V -> harness_error': ({'depth': 'settled', 'status': 'settled', 'F': 1.0, 'gates_pass': True},
                                       'harness_error'),
    }
    got = {k: P.classify_settled(x, inc)['class'] for k, (x, _w) in cases.items()}
    ck('T1', 'classify_settled boundaries (12 cases)', all(got[k] == w for k, (_x, w) in cases.items()), got)
    refused = []
    for name, (x, i) in {'mixed depth': ({'depth': 'exit', 'status': 'certified', 'F': 1.0, 'bar': 1.0}, inc),
                         'incumbent not settled': (_se(1.0), _se(1.0, status='not_settled')),
                         'incumbent gate fails': (_se(1.0), _se(1.0, gates=False))}.items():
        try:
            P.classify_settled(x, i)
            refused.append((name, False))
        except P.PolicyError:
            refused.append((name, True))
    ck('T1', 'classify_settled refuses mixed depth and a non-usable incumbent', all(v for _n, v in refused), refused)
    # efficiency rule (spec v2 measure: rows not classed worse; T10 holds the full set)
    row = (lambda cls, band=False: {'class': cls, 'in_efficiency_band': band})
    e11 = P.efficiency_rule([row('unresolved')] * 11 + [row('worse')] * 11, 17494.0)
    e12 = P.efficiency_rule([row('confirm_candidate')] * 12 + [row('worse')] * 10, 17494.0)
    e_unc = P.efficiency_rule([row('unresolved')] * 11 + [row('uncertified')] * 1 + [row('worse')] * 10, 17494.0)
    ck('T1', 'efficiency rule: 11/22 (= 1/2) -> exit_then_confirm; 12/22 -> settled_direct; uncertified rows stay in '
             'the denominator', e11['branch'] == 'exit_then_confirm' and e12['branch'] == 'settled_direct'
       and e_unc['n_rows'] == 22 and e_unc['branch'] == 'exit_then_confirm', [e11, e12, e_unc])
    # band membership in the table row: |dF| <= e_EV exactly
    x0e = {'depth': 'exit', 'status': 'certified', 'F': 800e6, 'bar': 1000.0}
    z1 = lat.with_pair(lat.x0(), 5, 2025, 1, 2)
    r_in = P._table_row(lat, z1, 'screen', dict(x0e, F=800e6 + 17494.0), x0e, R, 17494.0)
    r_out = P._table_row(lat, z1, 'screen', dict(x0e, F=800e6 + 17494.01), x0e, R, 17494.0)
    ck('T1', 'table row: |dF_exit| = e_EV in band, e_EV + 0.01 out; class by classify_exit with e = max(e_EV, bar '
             'sum)', r_in['in_efficiency_band'] and not r_out['in_efficiency_band'] and r_in['e'] == 17494.0
       and r_in['class'] == 'unresolved', [r_in, r_out])
    # candidate selection
    rows = [{'label': f'r{i}', 'node': n, 'year': y, 'F_exit': f, 'class': c} for i, (n, y, f, c) in enumerate([
        (7, 2025, 1.0, 'confirm_candidate'), (7, 2030, 2.0, 'confirm_candidate'), (5, 2025, 3.0, 'confirm_candidate'),
        (9, 2030, 4.0, 'unresolved'), (5, 2035, 5.0, 'confirm_candidate')])]
    sel, picked = P.select_candidates(rows, 2)
    sel1, picked1 = P.select_candidates(rows[:2], 2)
    ck('T1', 'candidates: best, then the next distinct in node AND year (same node and same year passed over); only '
             'confirm-candidate rows; fewer logged', [r['label'] for r in picked] == ['r0', 'r4']
       and len(sel['passed_over']) == 2 and [r['label'] for r in picked1] == ['r0'] and sel1['log'], [sel, sel1])
    # ladder rule
    def srow(n, y, zp, ze, f):
        z = lat.with_pair(lat.x0(), n, y, zp, ze)
        return {'label': lat.label(z), 'z': list(z), 'node': n, 'year': y, 'zp': zp, 'ze': ze, 'F_exit': f,
                'status': 'certified', 'duration': '4h' if ze == 2 * zp else '2h'}
    srows = [srow(7, 2025, 1, 2, 1.0), srow(9, 2030, 1, 2, 2.0), srow(5, 2035, 1, 1, 1.5), srow(5, 2025, 1, 2, 3.0)]
    cfg = {'ladder_zp': [2, 4], 'ladder_n_node_years': 2}
    lp = P.ladder_plan(lat, cfg, srows)
    ck('T1', 'ladder: best duration 4 h; node-years (7, 2025), (9, 2030); 1.0 MVA x 4 MWh in 2025 skipped (budget); '
             '3 plans', lp['best_duration'] == '4h' and lp['node_years'] == [[7, 2025], [9, 2030]]
       and [p['label'] for p in lp['plans']] == ['n7_y2025_p0.5_e2', 'n9_y2030_p0.5_e2', 'n9_y2030_p1_e4']
       and [s['label'] for s in lp['skipped']] == ['n7_y2025_p1_e4'] and 'budget' in lp['skipped'][0]['reasons'][0],
       lp)
    # default search path byte-identical to the pre-phase policy (PRE_PHASE_REV)
    head_p, head_sha = _load_head_module('pre_policy', 'p515_s56_search_policy.py', out_dir)
    ck('T1', f'the comparison is not vacuous: the policy at {PRE_PHASE_REV} differs from the working-tree policy',
       head_sha != D.sha256_file(P.__file__), [head_sha, D.sha256_file(P.__file__)])
    env = D.LatticeEnv(lat)
    same = []
    for start in ('x0', 'reference_unit', 'plan_2030'):
        for rank in (P.CLASS_RANK, P.CLASS_RANK_REVERSED):
            z0 = D.named_start(lat, start)
            land = D.SyntheticLandscape(f't1-{start}', D.named_start(lat, 'plan_2030'), z0)
            ev = (lambda a, land=land: {(tuple(z), a['depth']): land.summary(z, a['depth']) for z in a['zs']})
            cfg0 = {'start': list(z0), 'n_max': 80, 'arm': 't1', 'domain_size': D.DOMAIN_SIZE,
                    'class_rank': list(rank)}
            s_new, r_new = P.run_to_end(P.initial_state(cfg0), {}, env, ev)
            s_old, r_old = head_p.run_to_end(head_p.initial_state(cfg0), {}, env, ev)
            rep_new = P.terminal_report(s_new, r_new)
            rep_old = head_p.terminal_report(s_old, r_old)
            same.append({'start': start, 'rank': rank[0], 'state_equal': P.canonical_json(s_new) ==
                         P.canonical_json(s_old), 'report_equal': P.canonical_json(rep_new) == P.canonical_json(rep_old),
                         'budget': s_new['budget_used'], 'moves': len(s_new['incumbent_history']) - 1})
    ck('T1', 'default search path (no poll_depth / poll_only): state and terminal report byte-identical to the policy '
             f'at {PRE_PHASE_REV} ({head_sha[:12]}) on 6 synthetic searches', all(r['state_equal'] and r['report_equal']
                                                                         for r in same), same)
    head_d_src = subprocess.run(['git', 'show', f'{PRE_PHASE_REV}:p515_s56_search_driver.py'], cwd=CODE_ROOT,
                                capture_output=True, text=True, check=True).stdout
    ck('T1', f'the landscape comparison is not vacuous: the driver at {PRE_PHASE_REV} differs from the working tree',
       D.sha256_text(head_d_src) != D.sha256_file(D.__file__))
    ns = {}
    start = head_d_src.index('class SyntheticLandscape:')
    end = head_d_src.index('def synthetic_landscape_of(')
    exec(compile('import math\nimport hashlib\nimport p515_s56_search_policy as P\n'
                 'def sha256_text(t):\n    return hashlib.sha256(t.encode()).hexdigest()\n' +
                 head_d_src[start:end], 'head_landscape', 'exec'), ns)
    old_land = ns['SyntheticLandscape']('lseed', D.named_start(lat, 'plan_2030'), lat.x0())
    new_land = D.SyntheticLandscape('lseed', D.named_start(lat, 'plan_2030'), lat.x0())
    dom = D.enumerate_domain(lat)[::97]
    eq = all(P.canonical_json(old_land.summary(z, d)) == P.canonical_json(new_land.summary(z, d))
             for z in dom for d in P.DEPTHS) and old_land.describe() == new_land.describe()
    ck('T1', f'the plain synthetic landscape (no knobs) byte-identical to {PRE_PHASE_REV}\'s on {len(dom)} plans x 2 '
             'depths',
       eq)
    import p515_s44_campaign_harness as H
    try:
        cl = D.phase_capture_checklist(H)
        ok = all(cl.values())
    except D.Refused as error:
        cl, ok = str(error), False
    ck('T1', 'phase capture-path checklist (rule eleven) holds on the committed harness', ok, cl)
    return {'default_path_identity': same, 'head_policy_sha256': head_sha}


# ======================================================================================================================
#  T2 / T3 whole phase through the CLI
# ======================================================================================================================
def check_launch_shapes(test, root):
    lrs = launch_records(root)
    ok = all(1 <= len(lr['action']['zs']) <= P.BLOCK and lr['action']['depth'] in P.DEPTHS and
             len(lr['records']) == len(lr['action']['zs']) for lr in lrs)
    stages = []
    for lr in lrs:
        if not stages or stages[-1] != lr['action']['stage']:
            stages.append(lr['action']['stage'])
    order = [s for s in P.PHASE_STAGES if s in stages]
    ck(test, 'every launch is one block of <= 5 evaluations at one depth; stages run in order', ok and stages == order,
       stages)
    return lrs, stages


def t2(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't2')
    os.makedirs(base)
    root, logs, rc0 = freeze(base, 'exit_move', data_root, cf['exit_move'])
    t0 = time.time()
    rc, codes = run_phase(root, data_root, os.path.join(base, 'lock'), logs)
    wall = time.time() - t0
    f = final(root)
    ref, st = in_process(root, lat)
    ck('T2', 'whole phase (exit-then-confirm) through the launch CLI: freeze 0, codes 4...4, 0', rc0 == 0 and rc == 0
       and set(codes[:-1]) == {4}, {'codes_tail': codes[-3:], 'n': len(codes)})
    ck('T2', 'final state == results state == the in-process phase (byte-equal)', f['state'] == ref ==
       f['results_state'])
    lrs, stages = check_launch_shapes('T2', root)
    rep = f['results']['report']
    sr = rep['search_report']
    ck('T2', 'the phase ran every stage: 18 screen rows + ladder rows in the table, efficiency branch exit_then_confirm, '
             '2 candidates distinct in node and year, both polled completely, best start from the pool, a search '
             'with >= 1 move ending complete_poll_no_accepted_candidate',
       len([r for r in rep['marginal_value_table']['rows'] if r['source'] == 'screen']) == 18
       and rep['efficiency']['branch'] == 'exit_then_confirm' and len(rep['candidates']) == 2
       and rep['candidates'][0]['node'] != rep['candidates'][1]['node']
       and rep['candidates'][0]['year'] != rep['candidates'][1]['year']
       and all(c['poll_report']['termination']['reason'] == 'complete_poll_only'
               and c['poll_report']['class_partition_final_poll']['not_reached'] == 0 for c in rep['candidates'])
       and rep['pool']['chosen'] is not None and sr['n_moves'] >= 1
       and sr['termination']['reason'] == 'complete_poll_no_accepted_candidate'
       and stages == list(P.PHASE_STAGES[:3]) + ['candidate_polls', 'search'],
       {'stages': stages, 'efficiency': rep['efficiency'], 'cands': [c['label'] for c in rep['candidates']],
        'pool': rep['pool']['chosen'], 'search': sr['termination']})
    ck('T2', 'spec v2: the efficiency record carries the not-worse measure, the per-class counts and the v1 band measure '
             '(information only); each candidate\'s exit-view check is recorded, equal, mode stop',
       rep['efficiency']['measure'] == 'not_worse_against_x0'
       and sum(rep['efficiency']['class_counts'].values()) == rep['efficiency']['n_rows']
       and rep['efficiency']['n_not_worse'] == rep['efficiency']['class_counts']['confirm_candidate']
       + rep['efficiency']['class_counts']['unresolved']
       and 'band_measure_information_only' in rep['efficiency']
       and all(c['exit_view_check']['equal'] is True and c['exit_view_check']['mode'] == 'stop'
               for c in rep['candidates']) and f['results']['prediction_inputs']['Q7']['n_equal'] == 2,
       {'efficiency': rep['efficiency'], 'q7': f['results']['prediction_inputs']['Q7']})
    ck('T2', 'count identities hold in the candidate polls and the search', all(
        c['poll_report']['count_identity']['holds'] for c in rep['candidates']) and sr['count_identity']['holds'])
    unit = D.named_start(lat, 'reference_unit')
    reqd = {(tuple(e['z']), e['depth']) for e in st['evaluated']}
    pre = {(tuple(z), d) for z in (lat.x0(), unit, staged_z(lat)) for d in P.DEPTHS}
    ck('T2', 'the cached pool members (x0, unit, staged: the pilot records in the real phase) are never requested; the '
             'unit\'s screen row is served from the cache', not (reqd & pre) and any(
                 r['label'] == lat.label(unit) and r['served_as'] == 'synthetic_precached'
                 for r in rep['marginal_value_table']['rows']))
    acc = sum(lr['accounted_wall_s'] for lr in lrs) / 3600.0
    ck('T2', 'hours: cumulative = sum of the launch records\' accounted walls; every launch passed the ceiling check '
             'before it ran', abs(f['results']['hours']['cumulative_h'] - acc) < 1e-9 and all(
                 not lr['ceiling_check_before_launch']['stop'] for lr in lrs), [f['results']['hours']['cumulative_h'],
                                                                                 acc])
    return {'root': root, 'codes': len(codes), 'wall_s': wall, 'n_launches': len(lrs), 'stages': stages,
            'evaluations': rep['evaluations_by_stage_depth'], 'reference_state_sha256': D.sha256_text(ref)}


def t3(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't3')
    os.makedirs(base)
    root, logs, rc0 = freeze(base, 'direct', data_root, cf['direct'])
    rc, codes = run_phase(root, data_root, os.path.join(base, 'lock'), logs)
    f = final(root)
    ref, st = in_process(root, lat)
    rep = f['results']['report']
    ck('T3', 'settled-direct branch: exit 0; final state byte-equal to in process', rc0 == 0 and rc == 0 and
       f['state'] == ref == f['results_state'], codes[-3:])
    lrs, stages = check_launch_shapes('T3', root)
    post = [lr for lr in lrs if lr['action']['stage'] in ('candidate_polls', 'search')]
    sr = rep['search_report']
    ck('T3', 'efficiency fraction > 1/2 -> settled_direct; every candidate-poll and search evaluation at settled depth; '
             'polls classified on settled values; the search moved; count identities hold',
       rep['efficiency']['branch'] == 'settled_direct' and post and all(lr['action']['depth'] == 'settled'
                                                                        for lr in post)
       and all('improver_unaccepted' in c['poll_report']['class_partition_final_poll'] for c in rep['candidates'])
       and sr['n_moves'] >= 1 and 'accepted' in sr['class_partition_final_poll']
       and sr['count_identity']['holds'] and all(c['poll_report']['count_identity']['holds']
                                                 for c in rep['candidates']),
       {'efficiency': rep['efficiency'], 'search': sr['termination'], 'moves': sr['n_moves']})
    return {'root': root, 'n_launches': len(lrs), 'efficiency': rep['efficiency'],
            'evaluations': rep['evaluations_by_stage_depth']}


# ======================================================================================================================
#  T4 ceiling, T5 review stops
# ======================================================================================================================
def t4(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't4')
    os.makedirs(base)
    tc = copy.deepcopy(cf['exit_move'])
    tc['ceiling_h'] = 12.0
    root, logs, rc0 = freeze(base, 'ceiling', data_root, tc)
    rc, codes = run_phase(root, data_root, os.path.join(base, 'lock'), logs)
    lrs = launch_records(root)
    cdir = os.path.join(root, 'ceiling')
    recs = [D.load_json(os.path.join(cdir, n)) for n in sorted(os.listdir(cdir))] if os.path.isdir(cdir) else []
    c = recs[0]['ceiling_check'] if recs else {}
    cum = sum(lr['accounted_wall_s'] for lr in lrs) / 3600.0
    spec = D.load_phase_spec(root)[2]
    d = (recs[0]['next_action']['depth'] if recs else None)
    meas = [lr['accounted_wall_s'] / 3600.0 for lr in lrs if lr['action']['depth'] == d]
    proj = max([spec['phase']['ceiling']['projection_h'][d]] + meas) if d else None
    ok_arith = bool(recs) and abs(c['cumulative_h'] - cum) < 1e-9 and abs(c['projected_next_block_h'] - proj) < 1e-12 \
        and c['cumulative_plus_projected_h'] > 12.0 and all(
            lr['ceiling_check_before_launch']['cumulative_plus_projected_h'] <= 12.0 for lr in lrs)
    n_before = len(lrs)
    rc2 = drv(['--phase-run', '--root', root, '--lock-path', os.path.join(base, 'lock')], data_root,
              os.path.join(logs, 'reissue'))
    recs2 = sorted(os.listdir(cdir))
    ck('T4', 'the ceiling stops with exit 6 before the first block that would cross it; nothing launched; the record\'s '
             'arithmetic (cumulative = sum of launch walls; projected = max(frozen, measured same depth)); a re-issue '
             'stops again (6), no new launch', rc0 == 0 and rc == D.EXIT_CEILING and ok_arith and rc2 == D.EXIT_CEILING
       and len(launch_records(root)) == n_before and len(recs2) == 2 and not os.path.exists(
           os.path.join(root, D.PHASE_RESULTS_FILE)), {'rc': rc, 'rc2': rc2, 'check': c, 'cum': cum, 'proj': proj})
    return {'root': root, 'n_launches': n_before, 'ceiling_check': c}


def t5(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't5')
    os.makedirs(base)
    x0 = lat.x0()
    out = {}
    # candidate 1 of the exit_move configuration (in process) for the offset knob; the search's accepted plan
    spec0 = D.phase_spec(phase_id='probe', oracle='synthetic', test_only=True, data_root=data_root,
                         test_config=cf['exit_move'])
    st0, _r = D.phase_in_process(spec0, lat)
    cand1 = st0['candidates'][0]['z']
    acc = st0['search']['state']['incumbent_history'][1]['z']
    spec1 = D.phase_spec(phase_id='probe', oracle='synthetic', test_only=True, data_root=data_root,
                         test_config=cf['direct'])
    st1, _r = D.phase_in_process(spec1, lat)
    c1 = st1['candidates'][0]
    pre = {tuple(z) for z in spec1['oracle']['precached']}   # the pool members stand for the (settled) pilot records
    n_c1 = [p['z'] for p in D.LatticeEnv(lat).neighbourhood_points(tuple(c1['z'])) if tuple(p['z']) not in pre]
    cases = {
        'harness_error_screen': (with_knobs(cf['exit_move'], exit_harness_error=[list(lat.with_pair(x0, 5, 2025, 1, 2))]),
                                 'harness_error', 'screen'),
        'offset_candidate_confirmation': (with_knobs(cf['exit_move'], settled_offset=[[cand1, 60000.0]]),
                                          'stop_for_review_offset', 'candidate_confirmation'),
        'offset_search_confirmation': (with_knobs(cf['exit_move'], settled_offset=[[acc, 60000.0]]),
                                       'stop_for_review_offset', 'search'),
        'not_settled_trigger': (with_knobs(cf['direct'], not_settled=[list(z) for z in n_c1]),
                                'stop_for_review_not_settled', 'candidate_polls'),
        'start_not_certified': (with_knobs(cf['exit_move'], exit_uncertified=[list(x0)]),
                                'start_incumbent_not_certified', 'screen'),
        'exit_view_mismatch_candidate': (with_knobs(cf['exit_move'], exit_view_mismatch=[[cand1, 'F']]),
                                         'exit_view_mismatch', 'candidate_confirmation'),
    }
    for tag, (tc, reason, stage) in cases.items():
        root, logs, rc0 = freeze(base, tag, data_root, tc)
        rc, codes = run_phase(root, data_root, os.path.join(base, f'{tag}_lock'), logs)
        f = final(root)
        term = (f['results'] or {}).get('report', {}).get('termination') or {}
        ref, _st = in_process(root, lat)
        detail = {'rc': rc, 'termination': term, 'codes_tail': codes[-3:]}
        if tag == 'not_settled_trigger':
            dt = term.get('detail') or {}
            rv = (dt.get('sub') or {}).get('detail') or dt   # the sub-search's own trigger, or the phase-wide one
            detail['review'] = rv
            extra = rv.get('consecutive_not_settled', 0) >= 3 or rv.get('rule_share') is True
        else:
            extra = True
        ck('T5', f'{tag}: exit 5, termination {reason} at stage {stage}; results written; byte-equal to in process',
           rc0 == 0 and rc == D.EXIT_REVIEW and term.get('reason') == reason and term.get('stage') == stage and extra
           and f['state'] == ref, detail)
        out[tag] = detail
    return out


# ======================================================================================================================
#  T6 kill and resume, T7 the author's loop, T8 stub children, T9 refusals
# ======================================================================================================================
def t6(lat, data_root, out_dir, cf, ref):
    base = os.path.join(out_dir, 't6')
    os.makedirs(base)
    lrs = launch_records(ref['root'])
    first = {}
    for i, lr in enumerate(lrs, 1):
        first.setdefault((lr['action']['stage'], lr['action']['purpose']), i)
    srch = [i for i, lr in enumerate(lrs, 1) if lr['action']['stage'] == 'search']
    conf = next(i for i, lr in enumerate(lrs, 1) if lr['action']['stage'] == 'search'
                and lr['action']['depth'] == 'settled')
    last = len(lrs)
    targets = [('after_lock', 1), ('after_launch_dir', 2), ('during_evaluate', first[('ladder', 'ladder')], 0.0),
               ('after_evaluate', first[('candidate_confirmation', 'candidate_confirmation')]),
               ('after_launch_record', first[('candidate_polls', 'candidate_poll')]),
               ('after_manifest', srch[0]), ('after_state_view', conf), ('after_results', last)]
    ref_f = final(ref['root'])
    rows = []
    for k, tgt in enumerate(targets):
        root, logs, rc0 = freeze(base, f'kill_{k:02d}_{tgt[0]}_l{tgt[1]}', data_root, cf['exit_move'],
                                 phase_id='ph_exit_move')   # the reference's id (it enters the state)
        rc, codes = run_phase(root, data_root, os.path.join(base, f'lock_{k:02d}'), logs, kill=tgt)
        f = final(root)
        idir = os.path.join(root, 'interrupted')
        irecs = [D.load_json(os.path.join(idir, n)) for n in sorted(os.listdir(idir))
                 if n.endswith('.json')] if os.path.isdir(idir) else []
        stale = os.path.isdir(os.path.join(root, 'stale_locks'))
        st = json.loads(f['state']) if f['state'] else {}
        same = f['state'] == ref_f['state'] and f['results_state'] == ref_f['results_state'] and \
            f['n_launches'] == ref_f['n_launches']
        rows.append({'target': list(tgt), 'exit': rc, 'killed': -9 in codes, 'same_history': same,
                     'interrupted_records': len(irecs), 'killed_evaluations': sum(len(r['killed_evaluations'])
                                                                                  for r in irecs),
                     'stale_lock_taken_over': stale, 'evaluated': len(st.get('evaluated', []))})
        ck('T6', f'kill at {tgt[0]} (launch {tgt[1]}, stage {lrs[tgt[1] - 1]["action"]["stage"]}) + resume: phase state '
                 f'and results byte-equal to the uninterrupted run; killed evaluations uncounted; stale lock taken over',
           rc0 == 0 and rc == 0 and -9 in codes and same and stale
           and all(not e['counted'] for r in irecs for e in r['killed_evaluations']), rows[-1])
    return {'targets': rows}


def t7(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't7')
    os.makedirs(base)
    root, logs, rc0 = freeze(base, 'author_loop', data_root, cf['exit_move'])
    lock = os.path.join(base, 'lock')
    rel = os.path.relpath(root, data_root)
    cmd = D.phase_author_command(data_root, rel, python=sys.executable, extra_args=f' --lock-path {lock}')
    env = dict(os.environ)
    env.pop(D.KILL_ENV, None)
    p = subprocess.run(['zsh', '-c', cmd], capture_output=True, text=True, env=env)
    lines = [ln for ln in p.stdout.splitlines() if ' launch ' in ln and ' exit ' in ln]
    ex = [int(ln.rsplit('exit ', 1)[1]) for ln in lines]
    logs_n = sorted(n for n in os.listdir(root) if n.startswith('launch_') and n.endswith('_stdout.log'))
    errs = sorted(n for n in os.listdir(root) if n.startswith('launch_') and n.endswith('_stderr.log'))
    f = final(root)
    ref, _st = in_process(root, lat)
    p2 = subprocess.run(['zsh', '-c', cmd], capture_output=True, text=True, env=env)
    lines2 = [ln for ln in p2.stdout.splitlines() if ' launch ' in ln and ' exit ' in ln]
    with open(os.path.join(base, 'author_loop_stdout.log'), 'w') as h:
        h.write(p.stdout)
    with open(os.path.join(base, 'author_loop_stderr.log'), 'w') as h:
        h.write(p.stderr)
    with open(os.path.join(base, 'author_loop_reissue_stdout.log'), 'w') as h:
        h.write(p2.stdout)
    ck('T7', 'the author\'s command (exact text, scratch --lock-path): prints every exit code, 4...4 then 0, stops; '
             'one stdout/stderr log pair per invocation (empty stderr); final state byte-equal to in process',
       rc0 == 0 and p.returncode == 0 and ex and ex[-1] == 0 and set(ex[:-1]) <= {4} and len(logs_n) == len(ex)
       == len(errs) and all(os.path.getsize(os.path.join(root, n)) == 0 for n in errs) and f['state'] == ref
       and 'loop stopped: exit 0' in p.stdout, {'exits': ex[-3:], 'n': len(ex), 'stderr': p.stderr[-300:]})
    ck('T7', 'the command re-issued on the finished phase: one invocation, exit 0, the next log number',
       len(lines2) == 1 and lines2[0].endswith('exit 0') and f'launch {len(ex) + 1:03d}' in lines2[0], lines2)
    return {'command_template': cmd, 'n_invocations': len(ex)}


def stub_harness_config():
    import p515_s44_campaign_harness as H
    import p515_s56_ev_harness_tests as EVT
    derived, evp = EVT.srp1grid_ev_declaration()
    cfg, required = EVT._spec_like(derived, cap=500)
    opts = {H.S56_COHORT_FORMULATION_KEY: 'independent_units_all_node_years', H.S56_TV_IN_OBJECTIVE_KEY: True,
            H.S56_ESSO_SLACK_PENALTY_KEY: 1e4, 'flex_price_multiplier': 1.0, H.S56_EV_PROFILES_KEY: dict(evp)}
    settled = dict(opts, **{H.S56_SETTLED_STOP_DISABLED_KEY: True, H.S56_PRICING_CAPTURE_KEY: True})
    return {'by_depth': {'exit': {'configuration': cfg, 'cap': 500, 'entry_options': opts},
                         'settled': {'configuration': cfg, 'cap': 400, 'entry_options': settled}},
            'concurrency': 5, 'required_consecutive_cycles': required, 'memory_per_slot_gib': 3.29,
            'note': 'TEST-ONLY 1x configuration at caps 500 / 400 (no committed evaluation carries these keys)'}


def t8(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't8')
    os.makedirs(base)
    tc = copy.deepcopy(cf['exit_move'])
    tc['k_ev'] = 400
    tc['harness'] = stub_harness_config()
    root, logs, rc0 = freeze(base, 'stub', data_root, tc, oracle='stub_children')
    t0 = time.time()
    rc, codes = run_phase(root, data_root, os.path.join(base, 'lock'), logs,
                          env_extra={D.STUB_DELAY_ENV: '0.2:0.6:t8'})
    f = final(root)
    ref, _st = in_process(root, lat)
    spec = D.load_phase_spec(root)[2]
    ck('T8', 'stub children (harness evaluate, stub mode): freeze (collision scan clean) and the whole phase, exit 0; '
             'final state byte-equal to the in-process synthetic phase', rc0 == 0 and rc == 0 and f['state'] == ref
       and not spec['collision_scan']['collisions'], {'rc0': rc0, 'rc': rc, 'codes_tail': codes[-3:]})
    lrs = launch_records(root)
    ck('T8', 'each stub launch froze one harness spec whose keys recompute (replay verified every launch)',
       all(len([n for n in os.listdir(D.launch_paths(root, k)[0]) if n.startswith('campaign_spec_')]) == 1
           for k in range(1, len(lrs) + 1)))
    return {'root': root, 'n_launches': len(lrs), 'wall_s': time.time() - t0}


def t9(lat, data_root, out_dir, cf, ref_root):
    base = os.path.join(out_dir, 't9')
    os.makedirs(base)
    logs = os.path.join(base, 'logs')
    os.makedirs(logs)
    r1 = drv(['--phase-freeze', '--root', os.path.join(base, 'real_no_pred'), '--phase-id', 'x', '--oracle',
              'harness'], data_root, os.path.join(logs, 'real_no_pred'))
    r2 = drv(['--phase-freeze', '--root', os.path.join(base, 'syn_not_test'), '--phase-id', 'x', '--oracle',
              'synthetic'], data_root, os.path.join(logs, 'syn_not_test'))
    r3 = drv(['--phase-run', '--root', ref_root, '--lock-path', os.path.join(base, 'lock'), '--preflight-only',
              '--out', os.path.join(base, 'pf')], data_root, os.path.join(logs, 'pf_synthetic'))
    ck('T9', 'refused (exit 2): real freeze without predictions; synthetic phase not test-only; --preflight-only on a '
             'synthetic phase', (r1, r2, r3) == (2, 2, 2) and not os.path.exists(os.path.join(base, 'real_no_pred'))
       and not os.path.exists(os.path.join(base, 'pf')), [r1, r2, r3])
    # tamper: a copy of a 3-launch phase; alter a record summary
    root, flogs, _rc = freeze(base, 'tamper', data_root, cf['exit_move'])
    lock = os.path.join(base, 'tlock')
    for i in range(3):
        drv(['--phase-run', '--root', root, '--lock-path', lock], data_root, os.path.join(flogs, f'run_{i}'))
    lr_p = D.launch_paths(root, 2)[1]
    rec = D.load_json(os.path.join(root, D.load_json(lr_p)['records'][0]['record_file']))
    rf = os.path.join(root, D.load_json(lr_p)['records'][0]['record_file'])
    orig = open(rf).read()
    os.chmod(rf, 0o644)
    with open(rf, 'w') as h:
        h.write(orig.replace('"F": ', '"F": 1', 1))
    before = sorted(D.files_under(root))
    r4 = drv(['--phase-run', '--root', root, '--lock-path', lock], data_root, os.path.join(flogs, 'tampered'))
    after = sorted(D.files_under(root))
    with open(rf, 'w') as h:
        h.write(orig)
    r5 = drv(['--phase-run', '--root', root, '--lock-path', lock], data_root, os.path.join(flogs, 'restored'))
    r6 = drv(['--phase-freeze', '--root', root, '--phase-id', 'x', '--oracle', 'synthetic', '--test-only'],
             data_root, os.path.join(flogs, 'refreeze'))
    ck('T9', 'a tampered evaluation record: the replay refuses (exit 2), nothing written; restored -> proceeds (4); '
             're-freezing an existing root refused (write-once)', r4 == 2 and before == after and r5 == 4 and r6 == 2,
       [r4, r5, r6, rec.get('schema')])
    return {'codes': [r1, r2, r3, r4, r5, r6]}


# ======================================================================================================================
#  SPEC v2 tests: T10 efficiency measure (P1), T11 exit-view check (R4), T12 un-launched predecessor + collision scan,
#  T13 continuation (stubs), T14 code drift, T15 .DS_Store, T16 the first-block hold commands and the check script
# ======================================================================================================================
HAZARD = 'p515_s' + '4'   # the harness pre-launch scan's process-name hazard (never written literally here)


def dummy_predictions(ids=None):
    return {'predictions': {q: {'statement': 'test', 'evaluated_as': 'test', 'on_failure': 'test'}
                            for q in (ids or D.PHASE_PREDICTION_IDS)}}


def t10(lat, data_root, out_dir):
    row = (lambda cls, band=False: {'class': cls, 'in_efficiency_band': band})
    e = P.efficiency_rule([row('confirm_candidate')] * 5 + [row('unresolved')] * 6 + [row('worse')] * 10
                          + [row('uncertified')], 17494.0)
    ck('T10', 'efficiency (P1): numerator = confirm_candidate + unresolved; per-class counts recorded; uncertified in '
              'the denominator only; 11/22 -> exit_then_confirm',
       e['n_not_worse'] == 11 and e['n_rows'] == 22 and e['fraction'] == [1, 2] and e['branch'] == 'exit_then_confirm'
       and e['class_counts'] == {'confirm_candidate': 5, 'unresolved': 6, 'worse': 10, 'uncertified': 1,
                                 'harness_error': 0} and e['measure'] == 'not_worse_against_x0', e)
    e2 = P.efficiency_rule([row('confirm_candidate')] * 6 + [row('unresolved')] * 6 + [row('worse')] * 10, 17494.0)
    ck('T10', 'efficiency (P1): 12/22 not worse -> settled_direct (strict > 1/2, exact rational)',
       e2['fraction'] == [6, 11] and e2['branch'] == 'settled_direct', e2)
    # the measures disagree: rows near -20 kEUR are confirm-candidates OUTSIDE the +- e_EV band (P1's case)
    a = P.efficiency_rule([row('confirm_candidate', False)] * 14 + [row('worse')] * 8, 17494.0)
    b = P.efficiency_rule([row('unresolved', True)] * 3 + [row('worse', True)] * 9 + [row('worse')] * 10, 17494.0)
    ck('T10', 'the v1 band measure is information only: 14/22 not worse with 0 in band -> settled_direct (v1 would '
              'give exit_then_confirm); 3/22 not worse with 12 in band -> exit_then_confirm (v1 would give '
              'settled_direct)',
       a['branch'] == 'settled_direct' and a['band_measure_information_only']['branch_the_v1_measure_would_give']
       == 'exit_then_confirm' and a['band_measure_information_only']['n_in_band'] == 0
       and b['branch'] == 'exit_then_confirm' and b['band_measure_information_only']['branch_the_v1_measure_would_give']
       == 'settled_direct', [a, b])
    try:
        P.efficiency_rule([row('mystery')], 1.0)
        bad = False
    except P.PolicyError:
        bad = True
    ck('T10', 'an unknown row class is refused (no silent bucket)', bad)
    # in the phase: the efficiency decision reads the class, never the band (table rows from the real classifier)
    x0e = {'depth': 'exit', 'status': 'certified', 'F': 800e6, 'bar': 1000.0}
    z1 = lat.with_pair(lat.x0(), 5, 2025, 1, 2)
    r = P._table_row(lat, z1, 'screen', dict(x0e, F=800e6 - 20000.0, bar=1000.0), x0e, P.R_EUR, 17494.0)
    ck('T10', 'a row at dF_exit = -20,000 EUR is confirm_candidate (counts as not worse) and outside the band',
       r['class'] == 'confirm_candidate' and r['in_efficiency_band'] is False, r)
    return {'examples': {'at_half': e, 'disagree_a': a['branch'], 'disagree_b': b['branch']}}


def t11(lat, data_root, out_dir, cf):
    ex = {'depth': 'exit', 'status': 'certified', 'F': 811976794.7754139, 'bar': 2490.025410294533, 'read_cycle': 127}
    se = {'depth': 'settled', 'status': 'settled', 'exit_view': {'status': 'certified', 'F': 811976794.7754139,
                                                                  'bar': 2490.025410294533, 'read_cycle': 127}}
    eq = P.exit_view_check(ex, se)
    import math as _m
    ulp_f = P.exit_view_check(dict(ex, F=_m.nextafter(ex['F'], _m.inf)), se)
    ulp_b = P.exit_view_check(ex, dict(se, exit_view=dict(se['exit_view'], bar=_m.nextafter(ex['bar'], 0.0))))
    rc = P.exit_view_check(dict(ex, read_cycle=128), se)
    rcf = P.exit_view_check(dict(ex, read_cycle=127.0), se)
    ab = P.exit_view_check(ex, {'depth': 'settled', 'status': 'settled'})
    ck('T11', 'exit-view check (R4) is bitwise: equal -> equal; one ulp in F or bar, read_cycle + 1, read_cycle as a '
              'float, an absent exit_view -> mismatch naming the field',
       eq['equal'] and not ulp_f['equal'] and ulp_f['differing'] == ['F'] and not ulp_b['equal']
       and ulp_b['differing'] == ['bar'] and rc['differing'] == ['read_cycle'] and rcf['differing'] == ['read_cycle']
       and not ab['equal'] and ab['exit_view'] is None, [eq, ulp_f, ulp_b, rc, rcf, ab])
    # in the phase (in process)
    spec0 = D.phase_spec(phase_id='probe', oracle='synthetic', test_only=True, data_root=data_root,
                         test_config=cf['exit_move'])
    st0, _r = D.phase_in_process(spec0, lat)
    c1, c2 = st0['candidates'][0]['z'], st0['candidates'][1]['z']
    out = {}

    def run(tag, knobs, mode=None):
        sp = D.phase_spec(phase_id='probe', oracle='synthetic', test_only=True, data_root=data_root,
                          test_config=with_knobs(cf['exit_move'], **knobs))
        if mode:
            sp['policy']['exit_view_check'] = mode
        st, _rr = D.phase_in_process(sp, lat)
        out[tag] = {'termination': st['termination'], 'checks': [c.get('exit_view_check') for c in
                                                                 st['candidates'] or []], 'log': st['log']}
        return st
    a = run('mismatch_F', {'exit_view_mismatch': [[c1, 'F']]})
    b = run('mismatch_read_cycle_c2', {'exit_view_mismatch': [[c2, 'read_cycle']]})
    c = run('absent', {'exit_view_mismatch': [[c1, 'absent']]})
    d = run('harness_error_precedes', {'settled_harness_error': [c1], 'exit_view_mismatch': [[c2, 'F']]})
    e = run('mismatch_precedes_offset', {'settled_offset': [[c1, 60000.0]], 'exit_view_mismatch': [[c2, 'bar']]})
    f = run('mode_record', {'exit_view_mismatch': [[c1, 'F']]}, mode='record')
    g = run('offset_alone', {'settled_offset': [[c1, 60000.0]]})
    ok = (a['termination']['reason'] == 'exit_view_mismatch' and a['termination']['stage'] == 'candidate_confirmation'
          and a['termination']['detail']['mismatches'][0]['differing'] == ['F']
          and b['termination']['reason'] == 'exit_view_mismatch'
          and b['termination']['detail']['mismatches'][0]['differing'] == ['read_cycle']
          and c['termination']['reason'] == 'exit_view_mismatch'
          and d['termination']['reason'] == 'harness_error'
          and e['termination']['reason'] == 'exit_view_mismatch'
          and g['termination']['reason'] == 'stop_for_review_offset'
          and f['termination']['reason'] == 'phase_complete'
          and f['candidates'][0]['exit_view_check']['equal'] is False
          and f['candidates'][0]['exit_view_check']['mode'] == 'record'
          and any('mode record' in m for m in f['log']))
    ck('T11', 'in the phase: a mismatch stops exit_view_mismatch at candidate_confirmation (either candidate, any '
              'field, an absent view); a first-in-order harness_error precedes it; it precedes the offset stop (the '
              'offset alone still stops); mode record records the mismatch and the phase completes', ok,
       {k: v['termination'] for k, v in out.items()})
    try:
        P.phase_initial_state(dict(D.phase_config(spec0), exit_view_check='sometimes'))
        refused = False
    except P.PolicyError:
        refused = True
    ck('T11', 'an unknown exit_view_check mode is refused', refused)
    return {k: v['termination'] for k, v in out.items()}


def t12(lat, data_root, out_dir, cf):
    import shutil
    base = os.path.join(out_dir, 't12')
    os.makedirs(base)
    tc = copy.deepcopy(cf['exit_move'])
    tc['k_ev'] = 400
    tc['harness'] = stub_harness_config()
    spec = D.phase_spec(phase_id='t12', oracle='stub_children', test_only=True, data_root=data_root, test_config=tc)
    orc = D.HarnessOracle(spec, lat, lock_path=None)
    k = orc.eval_key(lat.x0(), P.DEPTH_EXIT)

    def pred_root(tag):
        r = os.path.join(base, tag)
        D.phase_freeze(r, {'schema': D.PHASE_SPEC_SCHEMA, 'tag': tag,
                           'cache_sources': [{'entries': [{'eval_key': k}]}]})
        for n in ('freeze_stdout.log', 'freeze_stderr.log'):
            open(os.path.join(r, n), 'w').close()
        return r
    ok_root = pred_root('pred_ok')
    open(os.path.join(ok_root, '.DS_Store'), 'w').close()
    sha_ok = D.load_phase_spec(ok_root)[1]
    own = os.path.join(base, 'own_root')

    def files(roots):
        return sorted(f for r in roots for f in D.files_under(r) if os.path.basename(f).startswith('phase_spec_'))

    def scan(preds, roots):
        sp = copy.deepcopy(spec)
        sp['predecessors'] = {'unlaunched': preds}
        try:
            return D.freeze_collision_scan(sp, lat, own, data_root, files=files(roots)), None
        except D.Refused as error:
            return None, error.reasons
    rec_ok = D.verify_unlaunched_predecessor(data_root, ok_root, sha_ok[:8])
    a, ra = scan([rec_ok], [ok_root])
    b, rb = scan([], [ok_root])
    ck('T12', 'a declared, verified un-launched predecessor (spec + freeze logs; .DS_Store ignored) is excluded: the '
              'scan passes, 1 file excluded; the same root undeclared: the scan refuses, naming its spec',
       a is not None and a['n_excluded_predecessor_roots'] == 1 and not a['collisions'] and rb
       and any('pred_ok' in x for x in rb), {'declared': a, 'undeclared': rb})
    # launched / altered predecessors
    cases = {}
    for tag, mutate in (('launches_dir', lambda r: os.makedirs(os.path.join(r, 'launches', 'launch_001'))),
                        ('eval_dir', lambda r: os.makedirs(os.path.join(r, 'evals', 'k_label'))),
                        ('state_view', lambda r: D.write_once_json(os.path.join(r, D.PHASE_STATE_VIEW), {})),
                        ('launch_log', lambda r: open(os.path.join(r, 'launch_001_stdout.log'), 'w').close())):
        r = os.path.join(base, f'pred_{tag}')
        shutil.copytree(ok_root, r)
        rec = D.verify_unlaunched_predecessor(data_root, r, sha_ok[:8])   # verified while clean ...
        mutate(r)                                                         # ... then launched / altered
        res, reasons = scan([rec], [r])
        cases[tag] = reasons
    try:
        D.verify_unlaunched_predecessor(data_root, ok_root, 'f' * 8)
        wrong = None
    except D.Refused as error:
        wrong = error.reasons
    ck('T12', 'the scan refuses when the declared predecessor has launches (launches/), an eval dir, a state view or '
              'a launch log (re-verified at the scan); a wrong declared sha is refused',
       all(cases[t] for t in cases) and any('directories' in x for x in cases['launches_dir'])
       and any('directories' in x for x in cases['eval_dir']) and any('other than' in x for x in cases['state_view'])
       and wrong, {'cases': cases, 'wrong_sha': wrong})
    # the REAL artefacts: the real v2 configuration's keys against the tracked specs, v1 declared / undeclared
    v1 = os.path.join(D.RESULTS_PARENT_REL, 'phase2x_ev_v1')
    t0 = time.time()
    real = D.phase_spec(phase_id='t12_real_probe', oracle='harness', test_only=False, data_root=data_root,
                        predictions=dummy_predictions(), unlaunched_predecessors=[(v1, 'de4e947e')])
    # the scan's own root is the REAL v2 root (read only, never written here): once spec v2 is frozen and committed it
    # is a tracked artefact carrying the same pilot keys, and the scan must exclude the run's own root (CLAUDE.md,
    # W105). v1 of this check used a scratch own root, valid only before the v2 freeze (tests v3, failed as run).
    own_real = os.path.join(data_root, D.RESULTS_PARENT_REL, 'phase2x_ev_v2')
    try:
        decl = D.freeze_collision_scan(real, lat, own_real, data_root)
        rd = None
    except D.Refused as error:
        decl, rd = None, error.reasons
    real_u = copy.deepcopy(real)
    real_u['predecessors']['unlaunched'] = []
    try:
        D.freeze_collision_scan(real_u, lat, own_real, data_root)
        ru = None
    except D.Refused as error:
        ru = error.reasons
    v2_tracked = any(os.path.dirname(f) == own_real for f in D.committed_spec_files(data_root))
    ck('T12', 'REAL artefacts (tracked specs; own root = the real v2 root): the real v2 configuration with v1 '
              '(de4e947e) declared un-launched -> scan passes, v1\'s spec excluded (and v2\'s own spec, if tracked, as '
              'the own root); v1 undeclared -> the scan refuses, naming v1\'s spec and nothing else',
       decl is not None and not decl['collisions'] and decl['n_excluded_predecessor_roots'] == 1
       and decl['excluded_predecessor_files'] == [os.path.join(v1, 'phase_spec_de4e947e.json')]
       and decl['n_excluded_own_root'] == (1 if v2_tracked else 0)
       and ru is not None and len(ru) == 1 and 'phase2x_ev_v1/phase_spec_de4e947e.json' in ru[0]
       and ru[0].startswith('collision scan: 1 artefact(s)'),
       {'declared': decl, 'refused': rd, 'undeclared_refusal': ru, 'v2_spec_tracked': v2_tracked,
        'wall_s': time.time() - t0})
    ck('T12', 'the real v2 spec built in process: v1 recorded as un-launched predecessor (files = spec + 2 freeze '
              'logs), cache-root key audit ok, pilot exit views on, exit-view check mode stop',
       real['predecessors']['unlaunched'][0]['files'] == ['freeze_stderr.log', 'freeze_stdout.log',
                                                         'phase_spec_de4e947e.json']
       and real['cache_root_key_audit']['ok'] and real['phase']['pilot_exit_views'] is True
       and real['policy']['exit_view_check'] == 'stop' and real['cache_sources'][0]['exit_views'] is True,
       real['predecessors'])
    return {'declared_real': decl, 'undeclared_real_refusal': ru}


def _decisions(rep):
    t = rep['marginal_value_table'] or {}
    rows = [{k: v for k, v in r.items() if k != 'served_as'} for r in t.get('rows') or []]
    return P.canonical_json({
        'rows': rows, 'efficiency': rep['efficiency'],
        'candidates': [[c['label'], (c.get('confirmation') or {}).get('result'), c.get('confirmed')]
                       for c in rep['candidates'] or []],
        'pool': (rep['pool'] or {}).get('chosen_label'),
        'incumbents': [h['z'] for h in (rep['search_incumbent_history'] or [])],
        'search_termination': ((rep['search_report'] or {}).get('termination') or {}).get('reason')})


def t13(lat, data_root, out_dir, cf, t8_root):
    import shutil
    base = os.path.join(out_dir, 't13')
    os.makedirs(base)
    tc = copy.deepcopy(cf['exit_move'])
    tc['k_ev'] = 400
    tc['harness'] = stub_harness_config()
    stub_env = {D.STUB_DELAY_ENV: '0.05:0.2:t13'}
    ref = final(t8_root)['results']
    ref_req = {(P.canonical_json(e['z']), e['depth']) for e in ref['state']['evaluated']}
    # (a) exit 6 -> continuation with a raised ceiling
    tca = dict(copy.deepcopy(tc), ceiling_h=12.0)
    ra, la, rc0 = freeze(base, 'pred_ceiling', data_root, tca, oracle='stub_children')
    rca, codes_a = run_phase(ra, data_root, os.path.join(base, 'lock_a'), la, env_extra=stub_env)
    sha_a = D.load_phase_spec(ra)[1]
    hours_a = D.phase_hours(ra, len(D._complete_indices(ra)[1]))['cumulative_h']
    rb = os.path.join(base, 'cont_ceiling')
    lb = os.path.join(base, 'cont_ceiling_logs')
    os.makedirs(lb)
    tcp = os.path.join(base, 'cont_ceiling_test_config.json')
    D.write_once_json(tcp, tc)
    rcf = drv(['--phase-freeze', '--root', rb, '--phase-id', 'ph_cont_ceiling', '--oracle', 'stub_children',
               '--test-only', '--phase-test-config', tcp, '--continuation-of', f'{ra}@{sha_a[:8]}', '--ceiling-h',
               '1000', '--ceiling-note', 'TEST: raised ceiling'], data_root, os.path.join(lb, 'freeze'))
    rcb, codes_b = run_phase(rb, data_root, os.path.join(base, 'lock_b'), lb, env_extra=stub_env)
    fb = final(rb)
    spb = D.load_phase_spec(rb)[2]
    req_a = {(P.canonical_json(e['z']), e['depth']) for e in json.loads(final(ra)['state'])['evaluated']}
    req_b = {(P.canonical_json(e['z']), e['depth']) for e in fb['results']['state']['evaluated']}
    lrs_b = launch_records(rb)
    ck('T13', 'continuation after exit 6 (stubs): predecessor stops 6; the continuation (raised ceiling 1000, note) '
              'freezes with it as cache root and runs to exit 0; its decisions equal the uninterrupted stub phase\'s '
              '(T8); predecessor + continuation requests = T8\'s requests, disjoint',
       rc0 == 0 and rca == D.EXIT_CEILING and rcf == 0 and rcb == 0
       and _decisions(fb['results']['report']) == _decisions(ref['report'])
       and not (req_a & req_b) and (req_a | req_b) == ref_req,
       {'codes_a': codes_a[-2:], 'rcf': rcf, 'rcb': rcb, 'n_a': len(req_a), 'n_b': len(req_b), 'n_ref': len(ref_req)})
    ck('T13', 'the continuation carries the predecessor\'s measured hours into its ceiling (carried_h = predecessor '
              'cumulative; every launch\'s ceiling check includes it); its spec records the predecessor (sha, launches, '
              'stop exit 6) and serves no exit view from it',
       spb['phase']['ceiling']['carried_h'] == hours_a and all(
           lr['ceiling_check_before_launch']['carried_h'] == hours_a for lr in lrs_b)
       and spb['predecessors']['continuation'][0]['spec_sha256'] == sha_a
       and spb['predecessors']['continuation'][0]['stop']['exit'] == D.EXIT_CEILING
       and spb['cache_sources'][-1]['exit_views'] is False and spb['phase']['ceiling']['ceiling_h'] == 1000.0
       and fb['results']['prediction_inputs']['carried_h'] == hours_a,
       {'carried': spb['phase']['ceiling']['carried_h'], 'hours_a': hours_a})
    # (b) exit 5 (exit_view_mismatch) -> continuation without the pilot exit views (the R4 fallback)
    spec0 = D.phase_spec(phase_id='probe', oracle='synthetic', test_only=True, data_root=data_root,
                         test_config=cf['exit_move'])
    st0, _r = D.phase_in_process(spec0, lat)
    cand1 = st0['candidates'][0]['z']
    tcm = copy.deepcopy(tc)
    tcm['landscape'].setdefault('knobs', {})['exit_view_mismatch'] = [[cand1, 'F']]
    rm_, lm, rc0m = freeze(base, 'pred_mismatch', data_root, tcm, oracle='stub_children')
    rcm, codes_m = run_phase(rm_, data_root, os.path.join(base, 'lock_m'), lm, env_extra=stub_env)
    term_m = final(rm_)['results']['report']['termination']
    sha_m = D.load_phase_spec(rm_)[1]
    rn = os.path.join(base, 'cont_mismatch')
    ln = os.path.join(base, 'cont_mismatch_logs')
    os.makedirs(ln)
    tcpm = os.path.join(base, 'cont_mismatch_test_config.json')
    D.write_once_json(tcpm, tcm)
    rcfn = drv(['--phase-freeze', '--root', rn, '--phase-id', 'ph_cont_mismatch', '--oracle', 'stub_children',
                '--test-only', '--phase-test-config', tcpm, '--continuation-of', f'{rm_}@{sha_m[:8]}',
                '--no-pilot-exit-views'], data_root, os.path.join(ln, 'freeze'))
    rcn, codes_n = run_phase(rn, data_root, os.path.join(base, 'lock_n'), ln, env_extra=stub_env)
    fn = final(rn)
    lrs_n = launch_records(rn)
    c0 = (fn['results']['report']['candidates'] or [{}])[0]
    ck('T13', 'continuation after exit 5 (exit_view_mismatch, stubs): predecessor stops 5 at candidate_confirmation; '
              'the continuation with --no-pilot-exit-views first runs x0 at exit depth, records the mismatch (mode '
              'record) and completes (exit 0)',
       rc0m == 0 and rcm == D.EXIT_REVIEW and term_m['reason'] == 'exit_view_mismatch' and rcfn == 0 and rcn == 0
       and lrs_n and lrs_n[0]['action']['zs'] == [list(lat.x0())] and lrs_n[0]['action']['depth'] == 'exit'
       and c0.get('exit_view_check', {}).get('equal') is False and c0['exit_view_check']['mode'] == 'record'
       and fn['results']['report']['termination']['reason'] == 'phase_complete',
       {'rcm': rcm, 'term_m': term_m, 'rcfn': rcfn, 'rcn': rcn,
        'first_action': lrs_n[0]['action'] if lrs_n else None})
    # (c) refusals
    ref_cases = {}

    def refuse(tag, **kw):
        try:
            D.phase_spec(phase_id='r', test_only=True, data_root=data_root, **kw)
            ref_cases[tag] = None
        except D.Refused as error:
            ref_cases[tag] = error.reasons
    inc = os.path.join(base, 'pred_incomplete')
    shutil.copytree(ra, inc)
    os.makedirs(D.launch_paths(inc, len(D._complete_indices(inc)[1]) + 1)[0])
    refuse('incomplete_launch', oracle='stub_children', test_config=tc, continuation_of=[(inc, sha_a[:8])])
    refuse('completed_predecessor', oracle='stub_children', test_config=tc,
           continuation_of=[(t8_root, D.load_phase_spec(t8_root)[1][:8])])
    refuse('wrong_sha', oracle='stub_children', test_config=tc, continuation_of=[(ra, 'f' * 8)])
    refuse('synthetic_oracle', oracle='synthetic', test_config=cf['exit_move'], continuation_of=[(ra, sha_a[:8])])
    refuse('ceiling_without_continuation', oracle='stub_children', test_config=tc, ceiling_h=500.0,
           ceiling_note='x')
    refuse('no_views_without_continuation', oracle='stub_children', test_config=tc, pilot_exit_views=False)
    unl, _l, _rc = freeze(base, 'unlaunched_as_continuation', data_root, tc, oracle='stub_children')
    refuse('unlaunched_as_continuation', oracle='stub_children', test_config=tc,
           continuation_of=[(unl, D.load_phase_spec(unl)[1][:8])])
    ck('T13', 'continuation refusals: an incomplete predecessor launch, a completed (phase_complete) predecessor, a wrong '
              'sha, a synthetic oracle, a raised ceiling or --no-pilot-exit-views without a continuation, an '
              'un-launched root declared as continuation', all(ref_cases.values()), ref_cases)
    # (d) a predecessor launched further after the continuation's freeze: every continuation launch refuses
    a2 = os.path.join(base, 'pred_grows')
    shutil.copytree(ra, a2)
    rg = os.path.join(base, 'cont_grows')
    lg = os.path.join(base, 'cont_grows_logs')
    os.makedirs(lg)
    rcg0 = drv(['--phase-freeze', '--root', rg, '--phase-id', 'ph_cont_grows', '--oracle', 'stub_children',
                '--test-only', '--phase-test-config', tcp, '--continuation-of', f'{a2}@{sha_a[:8]}', '--ceiling-h',
                '1000', '--ceiling-note', 'TEST'], data_root, os.path.join(lg, 'freeze'))
    os.makedirs(D.launch_paths(a2, len(D._complete_indices(a2)[1]) + 1)[0])
    before = sorted(D.files_under(rg))
    rcg = drv(['--phase-run', '--root', rg, '--lock-path', os.path.join(base, 'lock_g')], data_root,
              os.path.join(lg, 'run'), stub_env)
    msg = open(os.path.join(lg, 'run_stdout.log')).read()
    ck('T13', 'a continuation whose predecessor gained a launch after the freeze refuses every launch (exit 2), nothing '
              'written', rcg0 == 0 and rcg == D.EXIT_REFUSED and sorted(D.files_under(rg)) == before
       and 'continuation predecessor' in msg, {'rcg0': rcg0, 'rcg': rcg, 'msg': msg[-400:]})
    return {'ceiling_case': {'codes_a': len(codes_a), 'codes_b': len(codes_b), 'carried_h': hours_a},
            'mismatch_case': {'term_pred': term_m, 'codes_n': len(codes_n)}, 'refusals': ref_cases}


def t14(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't14')
    os.makedirs(base)
    spec = D.phase_spec(phase_id='t14', oracle='synthetic', test_only=True, data_root=data_root,
                        test_config=cf['exit_move'])
    ok_root = os.path.join(base, 'no_drift')
    D.phase_freeze(ok_root, spec)
    bad = copy.deepcopy(spec)
    bad['code_sha256']['p515_s56_search_policy.py'] = '0' * 64
    root = os.path.join(base, 'drift')
    D.phase_freeze(root, bad)
    logs = os.path.join(base, 'logs')
    os.makedirs(logs)
    before = sorted(D.files_under(base))
    r1 = drv(['--phase-run', '--root', root, '--lock-path', os.path.join(base, 'lock')], data_root,
             os.path.join(logs, 'run'))
    r2 = drv(['--phase-run', '--root', root, '--lock-path', os.path.join(base, 'lock'), '--preflight-only', '--out',
              os.path.join(base, 'pf')], data_root, os.path.join(logs, 'pf'))
    after = sorted(f for f in D.files_under(base) if not f.startswith(logs))
    m1 = open(os.path.join(logs, 'run_stdout.log')).read()
    m2 = open(os.path.join(logs, 'pf_stdout.log')).read()
    lock_left = os.path.exists(os.path.join(base, 'lock'))
    r3 = drv(['--phase-run', '--root', ok_root, '--lock-path', os.path.join(base, 'lock')], data_root,
             os.path.join(logs, 'ok'))
    ck('T14', 'code drift: --phase-run and --preflight-only refuse (exit 2) naming the file, nothing written, no lock '
              'left; the same spec without drift launches (exit 4)',
       r1 == D.EXIT_REFUSED and r2 == D.EXIT_REFUSED and 'code drift: p515_s56_search_policy.py' in m1
       and 'code drift: p515_s56_search_policy.py' in m2 and before == [f for f in after]
       and not os.path.exists(os.path.join(base, 'pf')) and not lock_left
       and r3 == D.EXIT_SUSPENDED, {'r': [r1, r2, r3], 'm1': m1[-300:]})
    drift = D.code_drift(bad)
    ck('T14', 'code_drift lists exactly the altered file', [d['file'] for d in drift] == ['p515_s56_search_policy.py']
       and not D.code_drift(spec), drift)
    return {'codes': [r1, r2, r3]}


def t15(lat, data_root, out_dir, cf, t2_root):
    base = os.path.join(out_dir, 't15')
    os.makedirs(base)
    root, logs, rc0 = freeze(base, 'ds_store', data_root, cf['exit_move'], phase_id='ph_exit_move')
    lock = os.path.join(base, 'lock')
    for i in range(3):
        drv(['--phase-run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, f'pre_{i}'))
    l1 = D.launch_paths(root, 1)[0]
    planted = [os.path.join(root, '.DS_Store'), os.path.join(root, 'launches', '.DS_Store'),
               os.path.join(l1, '.DS_Store'), os.path.join(l1, 'evals', '.DS_Store')]
    for f in planted:
        open(f, 'w').close()
    r4 = drv(['--phase-run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, 'after_plant'))
    man4 = D.load_json(D.launch_paths(root, 4)[2])
    # a killed launch whose launch dir holds only .DS_Store files
    r5 = drv(['--phase-run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, 'killed'),
             {D.KILL_ENV: 'after_launch_dir:5:0'})
    l5 = D.launch_paths(root, 5)[0]
    os.makedirs(os.path.join(l5, 'evals'), exist_ok=True)
    open(os.path.join(l5, 'evals', '.DS_Store'), 'w').close()
    open(os.path.join(l5, '.DS_Store'), 'w').close()
    r6 = drv(['--phase-run', '--root', root, '--lock-path', lock, '--take-over-stale-lock'], data_root,
             os.path.join(logs, 'resume'))
    rc, codes = run_phase(root, data_root, lock, logs)
    irecs = [D.load_json(os.path.join(root, 'interrupted', n)) for n in sorted(os.listdir(os.path.join(root,
                                                                                                    'interrupted')))
             if n.endswith('.json')] if os.path.isdir(os.path.join(root, 'interrupted')) else []
    f = final(root)
    ref = final(t2_root)
    ck('T15', '.DS_Store planted at the top level, in launches/, in a launch dir and in its evals/: the next launch runs '
              '(4) and its manifest lists none; a killed launch holding only .DS_Store files is moved aside with 0 '
              'killed evaluations; the phase completes with the state of T2 (byte-equal)',
       rc0 == 0 and r4 == D.EXIT_SUSPENDED and not any('.DS_Store' in k for k in man4) and r5 == -9
       and r6 == D.EXIT_SUSPENDED and len(irecs) == 1 and irecs[0]['killed_evaluations'] == [] and rc == 0 and f['state'] == ref['state'],
       {'r4': r4, 'r5': r5, 'r6': r6, 'rc': rc, 'irecs': irecs})
    ck('T15', 'files_under ignores .DS_Store; a root holding only .DS_Store can be frozen',
       not any(os.path.basename(x) == '.DS_Store' for x in D.files_under(root)) and _freeze_into_ds_only(base))
    return {'interrupted': irecs}


def _freeze_into_ds_only(base):
    r = os.path.join(base, 'ds_only')
    os.makedirs(r)
    open(os.path.join(r, '.DS_Store'), 'w').close()
    try:
        D.phase_freeze(r, {'schema': D.PHASE_SPEC_SCHEMA, 'x': 1})
        return True
    except D.Refused:
        return False


def t16(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't16')
    os.makedirs(base)
    chk = os.path.join(CODE_ROOT, D.PHASE_CHECK_SCRIPT)

    def craft(tag, summaries, manifest=True):
        r = os.path.join(base, tag)
        os.makedirs(os.path.join(r, 'launches'))
        lr = {'schema': D.PHASE_LAUNCH_SCHEMA, 'launch_index': 1, 'phase_id': 't16', 'wall_s': 4000.0,
              'action': {'stage': 'screen', 'purpose': 'screen', 'depth': 'exit', 'zs': [[0]] * len(summaries)},
              'records': [{'z': [i], 'depth': 'exit', 'eval_dir': f'launches/launch_001/evals/{i:016d}_p{i}',
                           'summary': sm} for i, sm in enumerate(summaries)]}
        rp = os.path.join(r, 'launches', 'launch_001.json')
        sha = D.write_once_json(rp, lr)
        if manifest:
            D.write_once_json(os.path.join(r, 'launches', 'launch_001_manifest_sha256.json'),
                              {'launches/launch_001.json': sha})
        return r
    g = {'child_exit_0': True, 'solve_profile_identity_holds': True, 'nothing_blocked': True}
    good = {'status': 'certified', 'read_cycle': 121, 'guards': g, 'wall_s': 3600.0, 'F': 1.0, 'bar': 1.0}
    roots = {'pass': craft('pass', [good, dict(good, wall_s=0.7 * 3600), dict(good, wall_s=1.6 * 3600)]),
             'uncertified': craft('uncertified', [good, dict(good, status='not_certified')]),
             'read_cycle_300': craft('read_cycle_300', [dict(good, read_cycle=300)]),
             'guard_false': craft('guard_false', [dict(good, guards=dict(g, nothing_blocked=False))]),
             'no_guards': craft('no_guards', [dict(good, guards={})]),
             'wall_short': craft('wall_short', [dict(good, wall_s=0.69 * 3600)]),
             'wall_long': craft('wall_long', [dict(good, wall_s=1.61 * 3600)]),
             'no_manifest': craft('no_manifest', [good], manifest=False)}
    res = {}
    for tag, r in roots.items():
        p = subprocess.run([sys.executable, chk, r], capture_output=True, text=True)
        res[tag] = {'rc': p.returncode, 'n_pass': p.stdout.count('\nPASS'), 'n_fail': p.stdout.count('\nFAIL'),
                    'tail': p.stdout[-200:]}
    p = subprocess.run([sys.executable, chk, os.path.join(base, 'absent')], capture_output=True, text=True)
    res['missing'] = {'rc': p.returncode}
    ck('T16', 's56_phase_check.py: all-pass record -> 0 (walls 0.7 and 1.6 h inclusive); uncertified, read_cycle 300, '
              'a guard false, no guards, wall 0.69 / 1.61 h -> 1 with FAIL; no manifest or no record -> 2',
       res['pass']['rc'] == 0 and res['pass']['n_pass'] == 3 and all(res[t]['rc'] == 1 and res[t]['n_fail'] >= 1 for t in
                                                                     ('uncertified', 'read_cycle_300', 'guard_false',
                                                                      'no_guards', 'wall_short', 'wall_long'))
       and res['no_manifest']['rc'] == 2 and res['missing']['rc'] == 2, res)
    # the single invocation, then the loop
    root, logs, rc0 = freeze(base, 'first_block', data_root, cf['exit_move'])
    lock = os.path.join(base, 'lock')
    rel = os.path.relpath(root, data_root)
    one = D.phase_single_launch_command(data_root, rel, python=sys.executable, extra_args=f' --lock-path {lock}')
    loop = D.phase_author_command(data_root, rel, python=sys.executable, extra_args=f' --lock-path {lock}')
    real_rel = os.path.join(D.RESULTS_PARENT_REL, 'phase2x_ev_v2')
    real_cmds = [D.phase_single_launch_command('/repo', real_rel), D.phase_author_command('/repo', real_rel),
                 D.phase_check_command('/repo', real_rel)]
    env = dict(os.environ)
    env.pop(D.KILL_ENV, None)
    p1 = subprocess.run(['zsh', '-c', one], capture_output=True, text=True, env=env)
    p1b = subprocess.run(['zsh', '-c', one], capture_output=True, text=True, env=env)
    n_launch_after_one = len(D._complete_indices(root)[1])
    p2 = subprocess.run(['zsh', '-c', loop], capture_output=True, text=True, env=env)
    lines = [ln for ln in p2.stdout.splitlines() if ' launch ' in ln and ' exit ' in ln]
    f = final(root)
    ref, _st = in_process(root, lat)
    for name, txt in (('one_stdout.log', p1.stdout), ('one_again_stdout.log', p1b.stdout),
                      ('loop_stdout.log', p2.stdout), ('loop_stderr.log', p2.stderr)):
        with open(os.path.join(base, name), 'w') as h:
            h.write(txt)
    ck('T16', 'first-block hold: the single invocation runs launch 001 (exit 4, logs launch_001_*), a second issue '
              'refuses to overwrite them; the loop then starts at 002 and runs to exit 0, state byte-equal to in '
              'process', rc0 == 0 and p1.stdout.strip().endswith('launch 001 exit 4') and n_launch_after_one == 1
       and 'logs exist: not re-run' in p1b.stdout and lines and ' launch 002 exit ' in lines[0]
       and lines[-1].endswith('exit 0') and f['state'] == ref,
       {'one': p1.stdout[-200:], 'again': p1b.stdout[-200:], 'loop_first': lines[:1], 'loop_last': lines[-1:]})
    ck('T16', 'no author command line (single, loop, check) for the real v2 root contains the harness hazard string',
       all(HAZARD not in c for c in real_cmds) and HAZARD not in os.path.basename(chk), real_cmds)
    return {'check_script': res, 'real_commands': real_cmds}


# ======================================================================================================================
#  1x PHASE tests (TASKS 5d): T17 2x option-off identity, T18 the real 1x spec, T19 the 1x flow
# ======================================================================================================================
SPEC_VOLATILE = ('frozen_utc', 'code_sha256', 'git_head')
FLAG_1X = ('--phase-flex', '1.0', '--no-pilot-exit-views')


def _flat(obj, path=''):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flat(v, f'{path}/{k}'))
        return out if obj else {path: {}}
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flat(v, f'{path}[{i}]'))
        return out if obj else {path: []}
    return {path: P.canonical_json(obj)}


def _diff_paths(a, b):
    fa, fb = _flat(a), _flat(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def _stable(spec):
    return {k: v for k, v in spec.items() if k not in SPEC_VOLATILE}


def _build(mod, **kw):
    try:
        return mod.phase_spec(**kw), None
    except Exception as error:   # either module's Refused (two classes)
        return None, list(getattr(error, 'reasons', [repr(error)]))


def _scan(mod, spec, lat, own, data_root):
    try:
        return mod.freeze_collision_scan(spec, lat, own, data_root), None
    except Exception as error:
        return None, list(getattr(error, 'reasons', [repr(error)]))


def t17(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't17')
    os.makedirs(base)
    old, old_sha = _load_head_module('pre1x_driver', 'p515_s56_search_driver.py', base, rev=PRE_1X_REV)
    old.CODE_ROOT = CODE_ROOT   # its source-text capture checks read the repository files, as the module at its rev
    ck('T17', f'the comparison is not vacuous: the driver at {PRE_1X_REV} differs from the working-tree driver',
       old_sha != D.sha256_file(D.__file__), [old_sha, D.sha256_file(D.__file__)])
    v1 = os.path.join(D.RESULTS_PARENT_REL, 'phase2x_ev_v1')
    v2 = os.path.join(D.RESULTS_PARENT_REL, 'phase2x_ev_v2')
    tc_stub = copy.deepcopy(cf['exit_move'])
    tc_stub['k_ev'] = 400
    tc_stub['harness'] = stub_harness_config()
    pp = os.path.join(data_root, D.RESULTS_PARENT_REL, 'phase2x_ev_v2_predictions.json')   # v2's and v3's file
    pred_v3 = dict(D.load_json(pp), source_file_sha256=D.sha256_file(pp))
    cases = {
        'real_2x_as_v2': dict(phase_id='t17_2x', oracle='harness', test_only=False, data_root=data_root,
                              predictions=dummy_predictions()),
        'real_2x_as_v3': dict(phase_id='s56phase2x_ev_v3', oracle='harness', test_only=False, data_root=data_root,
                              predictions=pred_v3, unlaunched_predecessors=[(v1, 'de4e947e')],
                              continuation_of=[(v2, '18a224d1')]),
        'stub_2x': dict(phase_id='t17_stub', oracle='stub_children', test_only=True, data_root=data_root,
                        test_config=tc_stub),
        'synthetic_2x': dict(phase_id='t17_syn', oracle='synthetic', test_only=True, data_root=data_root,
                             test_config=cf['exit_move']),
    }
    res, built = {}, {}
    for name, kw in cases.items():
        t0 = time.time()
        s_new, e_new = _build(D, **kw)
        s_old, e_old = _build(old, **kw)
        both = s_new is not None and s_old is not None
        diff = _diff_paths(_stable(s_new), _stable(s_old)) if both else None
        res[name] = {'built_new': s_new is not None, 'built_old': s_old is not None, 'refused_new': e_new,
                     'refused_old': e_old, 'differing_paths': diff, 'wall_s': time.time() - t0,
                     'volatile_differing': [k for k in SPEC_VOLATILE if both and s_new.get(k) != s_old.get(k)]}
        built[name] = (s_new, s_old)
        ck('T17', f'{name}: the spec built by the working tree equals the spec built by {PRE_1X_REV} in every field '
                  f'except {list(SPEC_VOLATILE)}', both and diff == [] and P.canonical_json(_stable(s_new))
           == P.canonical_json(_stable(s_old)), res[name])
    sn, so = built['real_2x_as_v2']
    own = os.path.join(base, 'own_root_2x')
    c_new, r_new = _scan(D, sn, lat, own, data_root)
    c_old, r_old = _scan(old, so, lat, own, data_root)
    ck('T17', 'the 2x freeze collision scan: the same outcome under both drivers (result or refusal, equal)',
       P.canonical_json([c_new, r_new]) == P.canonical_json([c_old, r_old]),
       {'new': [c_new, r_new], 'old': [c_old, r_old]})
    # T12's REAL-artefact call exactly (the v2 declaration with v1 un-launched; own root = the real v2 root): its
    # outcome is a property of the tracked artefacts, not of the driver version (phase1x_tests_v1: T12 failed on the v3
    # artefacts tracked since tests v4 -- 690d4737, 50d7dc67)
    kw12 = dict(phase_id='t12_real_probe', oracle='harness', test_only=False, data_root=data_root,
                predictions=dummy_predictions(), unlaunched_predecessors=[(v1, 'de4e947e')])
    s12n, e12n = _build(D, **kw12)
    s12o, e12o = _build(old, **kw12)
    own_v2 = os.path.join(data_root, v2)
    a12 = _scan(D, s12n, lat, own_v2, data_root) if s12n is not None else (None, e12n)
    b12 = _scan(old, s12o, lat, own_v2, data_root) if s12o is not None else (None, e12o)
    ck('T17', 'T12\'s real-artefact scan (v2 declaration, v1 un-launched, own root = the real v2 root): the same '
              'outcome under both drivers', s12n is not None and s12o is not None
       and P.canonical_json(a12) == P.canonical_json(b12), {'new': a12, 'old': b12})
    res['t12_real_scan_both_drivers'] = {'new': a12, 'old_equal': P.canonical_json(a12) == P.canonical_json(b12)}
    v3p = os.path.join(data_root, D.RESULTS_PARENT_REL, 'phase2x_ev_v3', 'phase_spec_600bbceb.json')
    v3 = D.load_json(v3p)
    info_v3 = _diff_paths({k: v for k, v in _stable(v3).items() if k != 'collision_scan'},
                          _stable(built['real_2x_as_v3'][0]))
    res['information_only_v3_committed_spec_vs_rebuilt'] = {
        'differing_paths': info_v3[:60], 'n_differing_paths': len(info_v3),
        'note': 'the committed v3 spec (frozen 2026-11-25) against the same declaration (phase id, predictions file, '
                'v1 un-launched, continuation of v2) rebuilt now by the working tree, collision_scan and the volatile '
                'fields excluded; a difference here would be tracked files added since its freeze (cache-root audit) '
                'or v2-root state -- information, read by no check'}
    res['collision_scan_2x'] = {'new': [c_new, r_new], 'old_equal': P.canonical_json([c_new, r_new])
                                == P.canonical_json([c_old, r_old])}
    return res


def t18(lat, data_root, out_dir):
    import shutil
    import tempfile
    import p515_s44_campaign_harness as H
    base = os.path.join(out_dir, 't18')
    os.makedirs(base)
    pred1 = dummy_predictions(D.PHASE_PREDICTION_IDS_1X)
    kw1 = dict(phase_id='t18_probe', oracle='harness', test_only=False, data_root=data_root, predictions=pred1,
               pilot_exit_views=False, flex_price=1.0)
    t0 = time.time()
    s1 = D.phase_spec(**kw1)
    s2 = D.phase_spec(phase_id='t18_probe', oracle='harness', test_only=False, data_root=data_root,
                      predictions=dummy_predictions())
    build_s = time.time() - t0
    hz = s1['harness']['by_depth']
    cl1, cl2 = s1['capture_path_checklist'], s2['capture_path_checklist']
    added = sorted(set(cl1) - set(cl2))
    ck('T18', 'the real 1x spec: arm ev_1x, ceiling 60 (carried 0, not raised), exit-view check stop, no pilot exit '
              'view, no cache source, no pool cache member, the 1x authority, price_level 1.0 with D1 / D2, the '
              'restated projection; the 2x spec unchanged on these fields',
       s1['arm'] == 'ev_1x' and s1['phase']['ceiling']['ceiling_h'] == 60.0 and s1['phase']['ceiling']['carried_h']
       == 0.0 and s1['phase']['ceiling']['raised_by_continuation'] is None and s1['policy']['exit_view_check'] == 'stop'
       and s1['phase']['pilot_exit_views'] is False and s1['cache_sources'] == []
       and s1['phase']['pool']['cache_members'] == [] and s1['authority'] == D.PHASE_AUTHORITY_1X
       and s1['phase']['price_level']['flex_price_multiplier'] == 1.0 and 'D1' in s1['phase']['price_level']
       and 'D2' in s1['phase']['price_level'] and s1['phase']['ceiling']['projection_restated_1x']
       == D.PHASE_PROJECTION_1X and s2['arm'] == 'ev_2x' and s2['phase']['ceiling']['ceiling_h'] == 120.0
       and 'price_level' not in s2['phase'] and s2['phase']['pilot_exit_views'] is True,
       {'arm': s1['arm'], 'ceiling': s1['phase']['ceiling']['ceiling_h'], 'mode': s1['policy']['exit_view_check'],
        'cache_sources': s1['cache_sources']})
    ck('T18', 'harness configuration: flex_price_multiplier 1.0 at both depths; everything else equal to the 2x '
              'configuration (entry options, configuration, caps, concurrency, cycles, memory)',
       all(hz[d]['entry_options'].get('flex_price_multiplier') == 1.0 for d in P.DEPTHS)
       and D.phase_flex_of(s1) == 1.0 and D.phase_flex_of(s2) == 2.0
       and _diff_paths({k: v for k, v in s1['harness'].items() if k != 'source'},
                       {k: v for k, v in s2['harness'].items() if k != 'source'})
       == sorted(f'/by_depth/{d}/entry_options/flex_price_multiplier' for d in P.DEPTHS),
       {d: hz[d]['entry_options'] for d in P.DEPTHS})
    ck('T18', 'capture checklist (rule eleven): every 2x item held with the same value, plus the 1x items (Q8 price '
              'level, Q9 leak, the ESSO recovery reconciliation), all true',
       all(cl1.get(k) == v for k, v in cl2.items()) and len(added) == 5 and all(cl1[k] is True for k in added)
       and all(v for v in cl1.values()), {'added': added})
    allowed = ('/arm', '/authority', '/policy/n_max_note', '/phase/ceiling/ceiling_h',
               '/phase/ceiling/projection_restated_1x', '/phase/price_level', '/phase/pilot_exit_views',
               '/harness/by_depth/exit/entry_options/flex_price_multiplier',
               '/harness/by_depth/settled/entry_options/flex_price_multiplier', '/harness/source/price_level',
               '/capture_path_checklist', '/predictions_recorded_before_run', '/cache_sources', '/cache_exit_views',
               '/cache_root_key_audit')
    diff = _diff_paths(_stable(s1), _stable(s2))
    outside = [k for k in diff if not any(k == a or k.startswith(a + '/') or k.startswith(a + '[') for a in allowed)]
    ck('T18', 'the 1x spec differs from the 2x spec only in the declared fields (flat path diff; no other field, in '
              'particular policy R / e_EV / K_EV / caps / efficiency / screen / ladder / mapping / projected_hours, '
              'differs)', not outside and diff, {'outside': outside, 'n_diff': len(diff), 'diff': diff[:40]})
    aud = s1['cache_root_key_audit']
    ck('T18', 'cache-root key audit with no cache root: ok, 0 files, 0 keys found; 2^1 x 37,042 domain keys distinct',
       aud['ok'] and aud['n_files_in_cache_roots'] == 0 and not aud['keys_found'] and s1['domain_keys_distinct'],
       aud)
    # every eval key the 1x phase can request
    t1 = time.time()
    dom = D.enumerate_domain(lat)
    o1, o2 = D.HarnessOracle(s1, lat, lock_path=None), D.HarnessOracle(s2, lat, lock_path=None)
    k1, k2, bad_absent, bad_flex = set(), set(), [], []
    for z in dom:
        ckey = H.candidate_key(lat.canonical_of_z(tuple(z)))
        for d in P.DEPTHS:
            k = o1.eval_key(z, d)
            opts = o1.entry_options(z, d)
            if opts.get('flex_price_multiplier') != 1.0:
                bad_flex.append(lat.label(tuple(z)))
            absent = {kk: vv for kk, vv in opts.items() if kk != 'flex_price_multiplier'}
            if D.compose_eval_key(H, hz[d]['configuration'], absent, int(hz[d]['cap']), ckey) != k:
                bad_absent.append((lat.label(tuple(z)), d))
            k1.add(k)
            k2.add(o2.eval_key(z, d))
    keys_s = time.time() - t1
    ck('T18', 'every eval key of the 1x phase (37,042 plans x 2 depths) is composed with flex_price_multiplier 1.0 and '
              'equals the multiplier-absent key (the harness rule: 1.0 = today\'s price, not keyed); 74,084 distinct; '
              '0 shared with the 74,084 2x keys',
       not bad_flex and not bad_absent and len(k1) == 2 * D.DOMAIN_SIZE == len(k2) and not (k1 & k2),
       {'bad_flex': bad_flex[:5], 'bad_absent': bad_absent[:5], 'n1': len(k1), 'n2': len(k2),
        'shared': len(k1 & k2), 'wall_s': keys_s})
    # the harness freezes the REAL 1x configuration (scratch dir outside the repository; nothing else written)
    tmp = tempfile.mkdtemp(prefix='s56_t18_harness_freeze_')
    probe = {}
    try:
        x0, unit = lat.x0(), D.named_start(lat, 'reference_unit')
        for d in P.DEPTHS:
            bd = hz[d]
            cands = [(D.eval_label(lat, z, d, s1['arm']), None, o1.entry_options(z, d)) for z in (x0, unit)]
            cfg = dict(copy.deepcopy(bd['configuration']), name=f't18 harness freeze probe {d}')
            try:
                _p, _sha, hspec = H.freeze_campaign_spec(
                    os.path.join(tmp, d), f't18_probe_{d}', cands, cfg, bd['cap'], s1['harness']['concurrency'],
                    D.AUTHORITY, required_consecutive_cycles=s1['harness']['required_consecutive_cycles'],
                    extra={'test_only_probe': 'T18, scratch dir, deleted after the check'})
                ents = hspec['candidates']
                probe[d] = {'frozen': True, 'keys_recompute': [e['eval_key'] for e in ents]
                            == [o1.eval_key(z, d) for z in (x0, unit)],
                            'flex_in_entries': [e.get('flex_price_multiplier') for e in ents],
                            'flex_label_in_entries': [e.get('flex_price_label') for e in ents]}
            except Exception as error:
                probe[d] = {'frozen': False, 'error': repr(error)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ck('T18', 'the harness freezes the real 1x configuration for x0 and the unit at both depths (no refusal of the 1.0 '
              'multiplier on S56 EV entries): keys recompute, entries carry flex_price_multiplier 1.0 and no flex '
              'label (1.0 runs unlabelled); scratch dir outside the repository, removed',
       all(probe.get(d, {}).get('frozen') and probe[d]['keys_recompute'] and probe[d]['flex_in_entries'] == [1.0, 1.0]
           and probe[d]['flex_label_in_entries'] == [None, None] for d in P.DEPTHS) and not os.path.exists(tmp),
       probe)
    # collisions: the tracked artefacts (the freeze scan), then every phase / pilot artefact on disk
    own = os.path.join(data_root, PHASE1X_ROOT_REL)
    try:
        cs, cs_err = D.freeze_collision_scan(s1, lat, own, data_root), None
    except D.Refused as error:
        cs, cs_err = None, error.reasons
    disk_roots = ['phase2x_ev_v1', 'phase2x_ev_v2', 'phase2x_ev_v3', 'pilot_ev_v1']
    disk = []
    for r in disk_roots:
        for f in D.files_under(os.path.join(data_root, D.RESULTS_PARENT_REL, r)):
            b = os.path.basename(f)
            if (b.startswith(('campaign_spec_', 'phase_spec_', 'pilot_spec_')) and b.endswith('.json')) \
                    or b == 'evaluation_record.json' or (b.startswith('launch_') and b.endswith('.json')):
                disk.append(f)
    dscan = D.collision_scan(disk, k1, own)
    ck('T18', 'collision scan of the 1x keys: 0 collisions over the tracked artefacts (the freeze scan; own root = the '
              'real 1x root, absent) and 0 over every spec, launch record and evaluation record on disk under the 2x '
              'phase roots v1-v3 and the pilot root',
       cs is not None and not cs['collisions'] and cs['n_excluded_cache_roots'] == 0 and not dscan['collisions']
       and dscan['n_files_scanned'] == len(disk) > 0,
       {'freeze_scan': cs, 'refused': cs_err, 'disk_files': len(disk), 'disk_collisions': dscan['collisions']})
    # refusals
    v2 = os.path.join(D.RESULTS_PARENT_REL, 'phase2x_ev_v2')
    ref = {}
    for name, kw in (('ceiling_80', dict(kw1, ceiling_h=80.0, ceiling_note='test')),
                     ('ceiling_60_declared_without_continuation', dict(kw1, ceiling_h=60.0, ceiling_note='test')),
                     ('price_1.5', dict(kw1, flex_price=1.5)), ('price_0', dict(kw1, flex_price=0.0)),
                     ('price_bool', dict(kw1, flex_price=True)), ('price_str', dict(kw1, flex_price='1.0')),
                     ('1x_serving_pilot_exit_views', dict(kw1, pilot_exit_views=True)),
                     ('1x_continuation_of_2x_v2', dict(kw1, continuation_of=[(v2, '18a224d1')])),
                     ('2x_no_pilot_exit_views_unchanged', dict(phase_id='x', oracle='harness', test_only=False,
                                                                data_root=data_root, predictions=dummy_predictions(),
                                                                pilot_exit_views=False)),
                     ('1x_predictions_q1_q7_only', dict(kw1, predictions=dummy_predictions()))):
        _s, e = _build(D, **kw)
        ref[name] = e
    ck('T18', 'refused: a 1x ceiling other than 60 (and any --ceiling-h without a continuation), a price level outside '
              '{1.0, 2.0} (1.5, 0, a bool, a string), a 1x spec serving pilot exit views, a 1x continuation of the 2x '
              'root v2 (named as a price-level refusal), predictions without Q8 / Q9; the 2x non-continuation '
              '--no-pilot-exit-views refusal unchanged',
       all(ref[k] for k in ref) and any('price level' in x for x in ref['1x_continuation_of_2x_v2'])
       and any('Q8' in x and 'Q9' in x for x in ref['1x_predictions_q1_q7_only'])
       and any('not in' in x for x in ref['price_1.5']), ref)
    pp = os.path.join(base, 'pred_1x_dummy.json')
    D.write_once_json(pp, pred1)
    logs = os.path.join(base, 'logs')
    os.makedirs(logs)
    rc = drv(['--phase-freeze', '--root', os.path.join(base, 'cli_15'), '--phase-id', 'x', '--oracle', 'harness',
              '--predictions', pp, '--phase-flex', '1.5', '--no-pilot-exit-views'], data_root,
             os.path.join(logs, 'cli_15'))
    ck('T18', 'CLI --phase-flex 1.5: exit 2, nothing written', rc == 2 and not os.path.exists(os.path.join(base,
                                                                                                         'cli_15')), rc)
    return {'build_s': build_s, 'keys_s': keys_s, 'n_keys_1x': len(k1), 'freeze_collision_scan': cs,
            'disk_scan_files': len(disk), 'refusals': ref, 'diff_vs_2x': diff, 'checklist_added': added,
            'harness_freeze_probe': probe}


def t19(lat, data_root, out_dir, cf):
    base = os.path.join(out_dir, 't19')
    os.makedirs(base)
    tc = copy.deepcopy(cf['exit_move'])
    tc['ceiling_h'] = 1000.0   # synthetic walls: the flow, not the ceiling, is under test here
    root, logs, rc0 = freeze(base, 'syn1x', data_root, tc, extra=FLAG_1X)
    spec = D.load_phase_spec(root)[2] if rc0 == 0 else {}
    rc, codes = run_phase(root, data_root, os.path.join(base, 'lock'), logs)
    f = final(root)
    ref, st = in_process(root, lat)
    lrs = launch_records(root)
    x0, unit, staged = lat.x0(), D.named_start(lat, 'reference_unit'), staged_z(lat)
    rep = f['results']['report'] if f['results'] else {}
    reqd = {(tuple(e['z']), e['depth']) for e in st['evaluated']}
    rows = (rep.get('marginal_value_table') or {}).get('rows') or []
    ck('T19', '1x synthetic phase: freeze 0 (arm ev_1x, nothing precached, no exit view, mode stop), codes 4...4, 0; '
              'final state == results state == in process',
       rc0 == 0 and spec.get('arm') == 'ev_1x' and spec['oracle']['precached'] == []
       and spec['phase']['pilot_exit_views'] is False and spec['policy']['exit_view_check'] == 'stop'
       and rc == 0 and set(codes[:-1]) == {4} and f['state'] == ref == f['results_state'],
       {'rc0': rc0, 'rc': rc, 'codes_tail': codes[-3:]})
    a1, a2 = lrs[0]['action'], lrs[1]['action']
    ck('T19', 'launch 1 = x0\'s exit run alone (purpose x0_exit); launch 2 = the first screen block of 5',
       a1['purpose'] == 'x0_exit' and a1['depth'] == 'exit' and a1['zs'] == [list(x0)] and a2['purpose'] == 'screen'
       and len(a2['zs']) == 5 and a2['depth'] == 'exit', [a1, a2])
    ck('T19', 'x0 run at exit and settled, the unit and the staged plan settled (the unit\'s screen row an exit run); '
              'no record served from a cache or precached; x0\'s exit check from its own run',
       {(tuple(x0), 'exit'), (tuple(x0), 'settled'), (tuple(unit), 'exit'), (tuple(unit), 'settled'),
        (tuple(staged), 'settled')} <= reqd and all(r.get('served_as') is None for r in rows)
       and (rep.get('table') or rep.get('marginal_value_table') or {}).get('x0', {}).get('served_as') is None
       and all(m.get('served_as') is None for m in (rep.get('pool') or {}).get('members') or []),
       {'pool': (rep.get('pool') or {}).get('members'), 'n_rows': len(rows)})
    # stub children (the harness evaluate path, harness stub mode; the test harness config is at 1.0)
    tcs = copy.deepcopy(cf['exit_move'])
    tcs['k_ev'] = 400
    tcs['harness'] = stub_harness_config()
    tcs['ceiling_h'] = 1000.0
    sroot, slogs, src0 = freeze(base, 'stub1x', data_root, tcs, oracle='stub_children', extra=FLAG_1X)
    src, scodes = run_phase(sroot, data_root, os.path.join(base, 'slock'), slogs,
                            env_extra={D.STUB_DELAY_ENV: '0.1:0.3:t19'})
    sf = final(sroot)
    sref, _sst = in_process(sroot, lat)
    sspec = D.load_phase_spec(sroot)[2] if src0 == 0 else {}
    ck('T19', '1x stub-children phase: freeze 0 (collision scan clean, no cache source), exit 0, state byte-equal to '
              'the in-process phase', src0 == 0 and src == 0 and sf['state'] == sref and sspec.get('cache_sources') == []
       and not sspec['collision_scan']['collisions'], {'rc0': src0, 'rc': src, 'codes_tail': scodes[-3:]})
    # the test-only 1x ceiling default
    r60, _l60, rc60 = freeze(base, 'ceiling_default', data_root, cf['exit_move'], extra=FLAG_1X)
    c60 = D.load_phase_spec(r60)[2]['phase']['ceiling']['ceiling_h'] if rc60 == 0 else None
    ck('T19', 'a 1x spec without a declared ceiling gets 60 h (test-only default = the real 1x ceiling)', c60 == 60.0,
       c60)
    real_cmds = [D.phase_single_launch_command('/repo', PHASE1X_ROOT_REL),
                 D.phase_author_command('/repo', PHASE1X_ROOT_REL), D.phase_check_command('/repo', PHASE1X_ROOT_REL)]
    ck('T19', 'no author command line (single, loop, check) for the real 1x root contains the harness hazard string',
       all(HAZARD not in c for c in real_cmds), real_cmds)
    return {'root': root, 'n_launches': len(lrs), 'stub_launches': len(launch_records(sroot)),
            'evaluations': rep.get('evaluations_by_stage_depth'), 'real_commands': real_cmds}


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=CODE_ROOT)
    ap.add_argument('--tests', default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19')
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
    if os.path.exists(out_dir) and [n for n in os.listdir(out_dir) if not n.endswith('.log')]:
        log(f'REFUSE: {out_dir} exists and holds non-log files')
        return 2
    os.makedirs(out_dir, exist_ok=True)
    tests = [t.strip() for t in args.tests.split(',') if t.strip()]
    started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
    lat, inst = D.build_lattice(data_root)
    cf = configs(lat)
    res = {}
    if '1' in tests:
        res['T1'] = t1(lat, data_root, out_dir)
    if any(t in tests for t in ('2', '6', '15')):
        res['T2'] = t2(lat, data_root, out_dir, cf)
    if '3' in tests:
        res['T3'] = t3(lat, data_root, out_dir, cf)
    if '4' in tests:
        res['T4'] = t4(lat, data_root, out_dir, cf)
    if '5' in tests:
        res['T5'] = t5(lat, data_root, out_dir, cf)
    if '6' in tests:
        res['T6'] = t6(lat, data_root, out_dir, cf, res['T2'])
    if '7' in tests:
        res['T7'] = t7(lat, data_root, out_dir, cf)
    if any(t in tests for t in ('8', '13')):
        res['T8'] = t8(lat, data_root, out_dir, cf)
    if '9' in tests:
        res['T9'] = t9(lat, data_root, out_dir, cf, res['T2']['root'] if 'T2' in res else None)
    if '10' in tests:
        res['T10'] = t10(lat, data_root, out_dir)
    if '11' in tests:
        res['T11'] = t11(lat, data_root, out_dir, cf)
    if '12' in tests:
        res['T12'] = t12(lat, data_root, out_dir, cf)
    if '13' in tests:
        res['T13'] = t13(lat, data_root, out_dir, cf, res['T8']['root'])
    if '14' in tests:
        res['T14'] = t14(lat, data_root, out_dir, cf)
    if '15' in tests:
        res['T15'] = t15(lat, data_root, out_dir, cf, res['T2']['root'])
    if '16' in tests:
        res['T16'] = t16(lat, data_root, out_dir, cf)
    if '17' in tests:
        res['T17'] = t17(lat, data_root, out_dir, cf)
    if '18' in tests:
        res['T18'] = t18(lat, data_root, out_dir)
    if '19' in tests:
        res['T19'] = t19(lat, data_root, out_dir, cf)
    gf = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == []', not gf, gf)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code = {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s56_neighbourhood.py',
        'p516_lattice.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py', 'p513_solve_profile_guard.py',
        's56_phase_check.py')}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *code], cwd=CODE_ROOT, capture_output=True,
                           text=True).stdout
    payload = {'schema': 'p515_s56_phase_tests_v2', 'script': SCRIPT, 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0,
               'interpreter': sys.executable, 'python': platform.python_version(), 'git_head': head,
               'code_files_dirty_vs_head': dirty, 'instance': inst, 'tests': tests,
               'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any '
                              'repository import, verify(0)); pickle blocked (0); every driver subprocess arms its own '
                              'guard and exits non-zero on a failed verify',
               'values_note': 'every objective value here is SYNTHETIC (landscape or stub children)',
               'results': res, 'guards': {'counts': dict(GUARD.counts), 'verify_0': gf, 'pickle': dict(PICKLE_COUNTS)},
               'code_files': code, 'checks': CHECKS,
               'failed_checks': [f"[{c['test']}] {c['check']}" for c in CHECKS if not c['ok']],
               'all_checks_ok': all(c['ok'] for c in CHECKS)}
    sha = D.write_once_json(os.path.join(out_dir, 'phase_tests.json'), payload)
    files = [f for f in D.files_under(out_dir) if os.path.basename(f) != 'phase_tests.json']
    D.write_once_json(os.path.join(out_dir, 'manifest_sha256.json'),
                      {'phase_tests.json': sha, 'files': {os.path.relpath(f, out_dir): D.sha256_file(f) for f in files},
                       'code_files': code,
                       'note': 'test trees (t2..t9) are hash-recorded only: scratch phase roots'})
    log(f'checks {sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} ok; wall {time.time() - t0:.0f} s; failed '
        f'{payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
