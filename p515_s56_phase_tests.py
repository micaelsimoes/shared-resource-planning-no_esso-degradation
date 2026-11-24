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


def freeze(base, tag, data_root, tc, oracle='synthetic', phase_id=None):
    root = os.path.join(base, tag)
    logs = os.path.join(base, f'{tag}_logs')
    os.makedirs(logs)
    tcp = os.path.join(base, f'{tag}_test_config.json')
    D.write_once_json(tcp, tc)
    rc = drv(['--phase-freeze', '--root', root, '--phase-id', phase_id or f'ph_{tag}'[:40], '--oracle', oracle,
              '--test-only',
              '--phase-test-config', tcp], data_root, os.path.join(logs, 'freeze'))
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
    # efficiency rule
    row = (lambda band, st='certified': {'in_efficiency_band': band, 'status': st})
    e11 = P.efficiency_rule([row(True)] * 11 + [row(False)] * 11, 17494.0)
    e12 = P.efficiency_rule([row(True)] * 12 + [row(False)] * 10, 17494.0)
    e_unc = P.efficiency_rule([row(True)] * 11 + [row(False, 'not_certified')] * 1 + [row(False)] * 10, 17494.0)
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
#  main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=CODE_ROOT)
    ap.add_argument('--tests', default='1,2,3,4,5,6,7,8,9')
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
    if any(t in tests for t in ('2', '6')):
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
    if '8' in tests:
        res['T8'] = t8(lat, data_root, out_dir, cf)
    if '9' in tests:
        res['T9'] = t9(lat, data_root, out_dir, cf, res['T2']['root'] if 'T2' in res else None)
    gf = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == []', not gf, gf)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code = {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s56_neighbourhood.py',
        'p516_lattice.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *code], cwd=CODE_ROOT, capture_output=True,
                           text=True).stdout
    payload = {'schema': 'p515_s56_phase_tests_v1', 'script': SCRIPT, 'started_utc': started,
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
