"""P5.15 S56 1x PHASE -- ZERO-SOLVE RESUME CHECK of the frozen 1x root (TASKS.md step 5d, 2026-11-26).

Reads the frozen 1x phase root (read only) and checks, with the driver's own code, what the author's first
`--phase-run` will do -- before any launch:
  A  the spec as frozen: its name carries its hash; schema; arm ev_1x; price level 1.0 at both harness depths with the
     `phase.price_level` record (D1, D2); ceiling 60 h, carried 0, not raised; no cache source and no pool cache
     member; pilot_exit_views False; exit-view check 'stop'; Q1-Q9 recorded; capture checklist all true; collision scan
     0 collisions; cache-root key audit ok; no code drift against the spec; no predecessor.
  B  the root holds only the spec and its freeze logs (no launch, no state view, no results).
  C  the resume point: the driver's cache (`phase_cache`) is EMPTY (D2: nothing carries over); the replay of the
     (empty) chain gives the initial state; one `P.phase_step` -> the first action = x0's exit run alone (stage screen,
     purpose x0_exit, depth exit, zs [x0]); the ceiling check before that block passes (0 h + projected exit block);
     MEMORY for a full block (`memory_preflight` at the spec's concurrency 5 x 3.29 GiB: the --preflight-only record
     checks the NEXT block only, 1 slot for x0's exit run) and DISK: free space against the most the 60 h ceiling
     admits at the frozen block projections -- floor(ceiling / projection_h[d]) blocks x 5 evaluations x the largest
     eval dir measured at depth d in the 2x roots (v2, v3; read now) -- for the worse depth; the 2x R5 reading (~300
     evaluations x ~391 MB = 117 GB; TASKS 5a R5) is recorded beside it, information only.
  D  the design order after it, in process on the frozen configuration with a SYNTHETIC landscape (values are not
     evaluations of the system; zero solves): x0 exit -> the screen in blocks of <= 5 (18 plans, the reference unit's
     row an exit run) -> the ladder -> candidate confirmation (x0 settled + the candidates) -> candidate polls -> the
     pool (the unit and the staged plan settled) -> the search; every action one depth, <= 5 plans; nothing served
     from a cache.

Guards: SolveProfileGuard(permitted=()) installed before any repository import, verify(0); pickle load/loads blocked
(0). Writes only --out (a NEW directory).

Launch (attached, both streams captured):
  python -u p515_s56_phase1x_resume_check.py --root data/SRP1/Results/P515S56/phase1x_ev_v1 --out <new dir>
      > <new dir>_stdout.log 2> <new dir>_stderr.log
Exit: 0 every check holds; 3 a check failed; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 1x resume check (never solve)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('1x resume check: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('1x resume check: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
CHECKS = []
DISK_R5_BYTES = 117e9   # TASKS 5a R5 (2x): eval dir ~391 MB, ~300 evaluations ~ 117 GB


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(test, name, ok, detail=None):
    CHECKS.append({'test': test, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{test}] {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


def eval_dir_sizes_2x(data_root):
    """Measured eval-dir sizes of the 2x phase (roots v2, v3; every launch's evals/<dir>, by depth from the label)."""
    out = {d: [] for d in P.DEPTHS}
    for r in ('phase2x_ev_v2', 'phase2x_ev_v3'):
        ld = os.path.join(data_root, D.RESULTS_PARENT_REL, r, 'launches')
        for n in sorted(os.listdir(ld)):
            ed = os.path.join(ld, n, 'evals')
            if not os.path.isdir(ed):
                continue
            for e in sorted(os.listdir(ed)):
                d = next((x for x in P.DEPTHS if f'__{x}__' in e), None)
                if d is not None:
                    out[d].append(sum(os.path.getsize(f) for f in D.files_under(os.path.join(ed, e))))
    return {d: {'n': len(v), 'max_bytes': max(v), 'mean_bytes': sum(v) / len(v)} for d, v in out.items()}


def part_a(spec, digest, path):
    ph, pol, hz = spec['phase'], spec['policy'], spec.get('harness') or {}
    ck('A', 'the spec file name carries its sha256 prefix; schema p515_s56_phase_spec_v2; oracle harness; not test-only',
       os.path.basename(path) == f'phase_spec_{digest[:8]}.json' and spec['schema'] == D.PHASE_SPEC_SCHEMA
       and spec['oracle']['kind'] == 'harness' and spec['test_only'] is False, os.path.basename(path))
    ck('A', 'arm ev_1x; price level 1.0 at both harness depths; phase.price_level 1.0 with D1 and D2; the 1x authority',
       spec['arm'] == 'ev_1x' and D.phase_flex_of(spec) == 1.0
       and (ph.get('price_level') or {}).get('flex_price_multiplier') == 1.0 and 'D1' in ph['price_level']
       and 'D2' in ph['price_level'] and spec['authority'] == D.PHASE_AUTHORITY_1X,
       {'arm': spec['arm'], 'flex': D.phase_flex_of(spec)})
    ck('A', 'ceiling 60 h, carried 0, not raised; R 10,000; e_EV 17,494 (rounded); K_EV 300; caps 300 / 300; block 5; '
            'cap 80; concurrency 5; memory 3.29 GiB per slot',
       ph['ceiling']['ceiling_h'] == 60.0 and ph['ceiling']['carried_h'] == 0.0
       and ph['ceiling']['raised_by_continuation'] is None and pol['R'] == 10000.0
       and round(pol['e_ev']) == D.PHASE_E_EV_ROUNDED and pol['k_ev'] == 300 and pol['block'] == 5
       and pol['cap'] == 80 and hz['by_depth']['exit']['cap'] == 300 and hz['by_depth']['settled']['cap'] == 300
       and hz['concurrency'] == 5 and hz['memory_per_slot_gib'] == 3.29,
       {'ceiling': ph['ceiling']['ceiling_h'], 'e_ev': pol['e_ev']})
    ck('A', 'no cache source, no pool cache member, no pilot exit view, exit-view check stop (Q7 automatic)',
       spec['cache_sources'] == [] and ph['pool']['cache_members'] == [] and ph['pilot_exit_views'] is False
       and pol['exit_view_check'] == 'stop', {'mode': pol['exit_view_check']})
    pr = (spec.get('predictions_recorded_before_run') or {}).get('predictions') or {}
    ck('A', 'predictions Q1-Q9 recorded (statement, evaluated_as, on_failure), with the source file sha256',
       all(q in pr and all(isinstance(pr[q].get(f), str) and pr[q][f] for f in ('statement', 'evaluated_as',
                                                                                  'on_failure'))
           for q in D.PHASE_PREDICTION_IDS_1X)
       and isinstance(spec['predictions_recorded_before_run'].get('source_file_sha256'), str), sorted(pr))
    ck('A', 'capture checklist all true, the 1x items present; collision scan 0 collisions; cache-root key audit ok; '
            'domain keys distinct',
       all(spec['capture_path_checklist'].values()) and len(spec['capture_path_checklist']) == 17
       and spec['collision_scan']['collisions'] == [] and spec['cache_root_key_audit']['ok']
       and spec['domain_keys_distinct'] is True,
       {'n_checklist': len(spec['capture_path_checklist']), 'scan': spec['collision_scan']})
    drift = D.code_drift(spec)
    ck('A', 'no code drift against the spec (the files a --phase-run checks)', drift == [], drift)
    ck('A', 'no predecessor (un-launched or continuation)',
       spec['predecessors']['unlaunched'] == [] and spec['predecessors']['continuation'] == [], spec['predecessors'])


def part_b(root):
    names = sorted(n for n in os.listdir(root) if n not in D.IGNORED_FILE_NAMES)
    spec_n = [n for n in names if n.startswith('phase_spec_')]
    ck('B', 'the root holds only the spec and its freeze logs (no launches/, state view, results, launch logs)',
       len(spec_n) == 1 and set(names) <= set(spec_n) | {'freeze_stdout.log', 'freeze_stderr.log'}, names)
    idx, complete = D._complete_indices(root)
    ck('B', 'no launch (complete or incomplete)', idx == [] and complete == [], [idx, complete])
    return names


def part_c(root, spec, digest, data_root, lattice, env):
    oracle = D.make_oracle(spec, lattice, None)
    cache = D.phase_cache(spec, oracle, data_root, lattice)
    ck('C', 'the driver\'s cache is empty (D2: nothing carries over; no exit view served)', cache == {},
       sorted(cache)[:5])
    state, records, chain, _states = D.phase_replay(root, spec, digest, lattice, env, oracle, cache)
    init = P.phase_initial_state(D.phase_config(spec))
    ck('C', 'the replay of the empty chain gives the initial phase state', chain == [] and records == {}
       and P.canonical_json(state) == P.canonical_json(init))
    st2, action = P.phase_step(state, records, env)
    x0 = lattice.x0()
    ok = (action['kind'] == 'evaluate' and action['stage'] == 'screen' and action['purpose'] == 'x0_exit'
          and action['depth'] == P.DEPTH_EXIT and action['zs'] == [list(x0)] and action['action_index'] == 0)
    ck('C', 'the first action = x0\'s exit run alone (stage screen, purpose x0_exit, depth exit)', ok, action)
    hours = D.phase_hours(root, 0)
    cc = D.ceiling_check(spec, hours, action['depth'])
    ck('C', 'the ceiling check before the first block passes (0 h + projected exit block <= 60 h)',
       not cc['stop'] and hours['cumulative_h'] == 0.0, cc)
    mem5 = D.memory_preflight(spec['harness']['concurrency'], spec['harness']['memory_per_slot_gib'])
    ck('C', f"memory for a full block: available >= {spec['harness']['concurrency']} x "
            f"{spec['harness']['memory_per_slot_gib']} GiB (read now; the author runs with the Claude app closed)",
       mem5.get('pass') is True, mem5)
    du = shutil.disk_usage(data_root)
    sizes = eval_dir_sizes_2x(data_root)
    ce = spec['phase']['ceiling']
    worst = {d: int(ce['ceiling_h'] // ce['projection_h'][d]) * spec['policy']['block'] * sizes[d]['max_bytes']
             for d in P.DEPTHS}
    need = max(worst.values())
    ck('C', 'disk: free space >= the most the 60 h ceiling admits (blocks at the frozen projections x 5 x the largest '
            '2x eval dir at that depth, worse depth)', du.free >= need,
       {'free_gib': du.free / D.GIB, 'required_gib': need / D.GIB, 'by_depth_gib': {d: v / D.GIB for d, v in
                                                                                      worst.items()}})
    key = oracle.eval_key(x0, P.DEPTH_EXIT)
    lab = D.eval_label(lattice, x0, P.DEPTH_EXIT, spec['arm'])
    return {'first_action': action, 'ceiling_check': cc, 'x0_exit_eval_key': key, 'x0_exit_label': lab,
            'x0_exit_entry_options': oracle.entry_options(x0, P.DEPTH_EXIT), 'state_after_first_step': st2,
            'memory_full_block': mem5,
            'disk': {'free_bytes': du.free, 'free_gib': du.free / D.GIB, 'total_bytes': du.total,
                     'eval_dir_sizes_2x': sizes, 'ceiling_bounded_worst_bytes': worst, 'required_bytes': need,
                     'information_only_R5_2x_bytes': DISK_R5_BYTES,
                     'information_only_R5_covered': du.free >= DISK_R5_BYTES}}


def part_d(spec, lattice, env):
    """The design order on the frozen configuration with a synthetic landscape (in process, zero solves)."""
    x0 = lattice.x0()
    unit = D.named_start(lattice, 'reference_unit')
    staged = next(tuple(m['z']) for m in spec['phase']['pool']['fixed'] if m['name'] == 'pilot_staged')
    land = D.SyntheticLandscape('resume-1x', staged, x0, knobs={'always_ok': [list(x0), list(unit), list(staged)],
                                                                'exit_views': True})
    state = P.phase_initial_state(D.phase_config(spec))
    records, actions = {}, []
    for _ in range(100000):
        state, action = P.phase_step(state, records, env)
        if action['kind'] == 'terminated':
            break
        actions.append({'stage': action['stage'], 'purpose': action['purpose'], 'depth': action['depth'],
                        'labels': [lattice.label(tuple(z)) for z in action['zs']]})
        for z in action['zs']:
            records[(tuple(z), action['depth'])] = land.summary(z, action['depth'])
    first = actions[0]
    screen = [lattice.label(tuple(r['z'])) for r in spec['phase']['screen']]
    scr_actions = [a for a in actions if a['purpose'] == 'screen']
    ck('D', 'synthetic order: x0 exit first, alone; then the 18 screen plans in blocks of <= 5 at exit (spec order), '
            'the unit\'s screen row among them', first['purpose'] == 'x0_exit' and first['labels'] == [lattice.label(x0)]
       and [lab for a in scr_actions for lab in a['labels']] == screen and all(len(a['labels']) <= 5 and a['depth']
                                                                               == 'exit' for a in scr_actions)
       and lattice.label(unit) in screen and actions[1]['purpose'] == 'screen', actions[:6])
    stages = []
    for a in actions:
        if not stages or stages[-1] != a['stage']:
            stages.append(a['stage'])
    pool_req = [lab for a in actions if a['purpose'] == 'pool_settled' for lab in a['labels']]
    conf = [a for a in actions if a['purpose'] == 'candidate_confirmation']
    settled_req = {lab for a in actions if a['depth'] == P.DEPTH_SETTLED for lab in a['labels']}
    ck('D', 'synthetic order: x0 settled in the candidate confirmation; x0, the unit and the staged plan each run at '
            'settled depth by the phase (the unit in the confirmation when it is a candidate, else in the pool); '
            'every action <= 5 plans at one depth; no (plan, depth) requested twice',
       (not conf or lattice.label(x0) in conf[0]['labels'])
       and {lattice.label(x0), lattice.label(unit), lattice.label(tuple(staged))} <= settled_req
       and all(len(a['labels']) <= 5 for a in actions)
       and len({(lab, a['depth']) for a in actions for lab in a['labels']})
       == sum(len(a['labels']) for a in actions),
       {'stages': stages, 'pool_settled': pool_req, 'confirmation': conf[:1]})
    tr = state.get('termination') or {}
    return {'note': 'SYNTHETIC values (SyntheticLandscape seed resume-1x, target the staged plan): the ORDER of the '
                    'requests on the frozen configuration, not evaluations of the system',
            'stages_in_order': stages, 'n_actions': len(actions), 'actions_head': actions[:8],
            'pool_settled_requests': pool_req, 'termination': tr,
            'efficiency_branch': (state.get('efficiency') or {}).get('branch')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=CODE_ROOT)
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    root = args.root if os.path.isabs(args.root) else os.path.join(data_root, args.root)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
    if os.path.exists(out_dir):
        log(f'REFUSE: --out exists (write-once): {out_dir}')
        return 2
    started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
    try:
        path, digest, spec = D.load_phase_spec(root)
    except D.Refused as error:
        log(f'REFUSE: {error.reasons}')
        return 2
    lattice, inst = D.build_lattice(data_root)
    env = D.LatticeEnv(lattice)
    part_a(spec, digest, path)
    names = part_b(root)
    res_c = part_c(root, spec, digest, data_root, lattice, env)
    res_d = part_d(spec, lattice, env)
    ck('B', 'the root is unchanged by this check (same file list before and after)',
       sorted(n for n in os.listdir(root) if n not in D.IGNORED_FILE_NAMES) == names)
    gf = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == []', not gf, gf)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code = {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s56_neighbourhood.py',
        'p516_lattice.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *code], cwd=CODE_ROOT, capture_output=True,
                           text=True).stdout
    payload = {'schema': 'p515_s56_phase1x_resume_check_v1', 'script': SCRIPT, 'part': 'resume',
               'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'wall_s': time.time() - t0, 'interpreter': sys.executable, 'python': platform.python_version(),
               'git_head': head, 'code_files_dirty_vs_head': dirty, 'code_files': code, 'instance': inst,
               'phase_root': os.path.relpath(root, data_root), 'phase_spec': os.path.basename(path),
               'phase_spec_sha256': digest,
               'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any '
                              'repository import, verify(0)); pickle blocked (0); the phase root read only',
               'objective_convention': 'no objective value of the system is read or written here; part D values '
                                       'are SYNTHETIC',
               'results': {'resume_point': res_c, 'synthetic_order': res_d},
               'guards': {'counts': dict(GUARD.counts), 'verify_0': gf, 'pickle': dict(PICKLE_COUNTS)},
               'checks': CHECKS, 'failed_checks': [f"[{c['test']}] {c['check']}" for c in CHECKS if not c['ok']],
               'all_checks_ok': all(c['ok'] for c in CHECKS)}
    os.makedirs(out_dir)
    sha = D.write_once_json(os.path.join(out_dir, 'resume_check.json'), payload)
    D.write_once_json(os.path.join(out_dir, 'resume_check_manifest_sha256.json'), {'resume_check.json': sha})
    log(f'checks {sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} ok; wall {time.time() - t0:.1f} s; failed '
        f'{payload["failed_checks"]}; record sha256 {sha}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
