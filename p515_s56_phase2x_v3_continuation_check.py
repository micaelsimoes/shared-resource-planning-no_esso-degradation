"""P5.15 S56 2x PHASE -- ZERO-SOLVE CONTINUATION CHECK for spec v3 (TASKS.md step 5b, Planner decision 2026-11-25:
the extended solve-profile guard, oracle mapping v2, then a continuation spec v3 of phase2x_ev_v2 resuming at candidate
2's poll with every v2 record served from cache).

PART remap  (before the v3 freeze; reads the v2 root only)
  A  every record of every complete v2 launch: the summary the launch record stored (mapping v1, as run) against the
     summary the working-tree mapping (v2) derives from the same eval dir now: status and guard changes.
  B  the v2 launch chain replayed in process with the STORED (v1) summaries: each launch's action re-derived and
     compared with the recorded action; after launch 10 the phase terminates stop_for_review_not_settled with the
     recorded terminal state (a check that this re-derivation reproduces v2 as run).
  C  the same chain with the RE-MAPPED (v2 mapping) summaries: each launch's action compared with the recorded one;
     after launch 10, the next action (what v2 would have done under the corrected guard). Reported: the marginal-
     value table row of each changed record, the efficiency rule, the candidate selection, both candidate polls'
     classifications and their not_settled reviews, and the phase-wide review.
PART resume (after the v3 freeze; reads the v2 and v3 roots only)
  D  the v3 spec as frozen: its cache (`phase_cache`: the pilot source + the v2 continuation source), the v3 replay (no
     launch), one `phase_step` -> the first v3 action. Checked: it equals C's next action; it is the remainder of
     candidate 2's poll; none of its plans is cached; every v2 record and the pilot cache records are served from
     cache, and the stage-by-stage decisions equal C's. Compared on decision-bearing fields: BOOKKEEPING (sub-search ids,
     a sub-search's action_index and budget_used: requests issued in this phase) is reported, not compared.
PART freeze-sim (zero solves; nothing frozen in the repository)
  E  the v3 freeze in process with the v2 launch artefacts added to the tracked-artefact list (the real freeze refused
     at the cache-root key audit: only v2's spec is tracked), frozen into a scratch root under --out, then D on it.

Guards: SolveProfileGuard(permitted=()) installed before any repository import, verify(0); pickle load/loads blocked
(0). Nothing is written outside --out (a NEW directory, outside both phase roots).

Launch (attached, both streams captured):
  python -u p515_s56_phase2x_v3_continuation_check.py --part remap --out <new dir> > <dir>_stdout.log 2> ..._stderr.log
  python -u p515_s56_phase2x_v3_continuation_check.py --part resume --v3-root <root> --remap <remap dir> --out <new dir>
  python -u p515_s56_phase2x_v3_continuation_check.py --part freeze-sim --remap <remap dir> --out <new dir>
Exit: 0 every check holds; 3 a check failed; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 v3 continuation check (never solve)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('continuation check: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('continuation check: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
CHECKS = []
V2_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'phase2x_ev_v2')
CJ = P.canonical_json


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(test, name, ok, detail=None):
    CHECKS.append({'test': test, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{test}] {name}' + (f' -- {str(detail)[:800]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


def v2_launches(v2_root):
    _idx, complete = D._complete_indices(v2_root)
    return [D.load_json(D.launch_paths(v2_root, n)[1]) for n in complete]


def chain_replay(cache, launches, summary_of, env):
    """Deliver each v2 launch's records (summary_of(record) -> summary) after re-deriving its action. -> (state after
    the last launch's delivery and the next step, next action, per-launch [(action equal, state equal)], records)."""
    records = dict(cache)
    state = None
    per = []
    for lr in launches:
        if state is None:
            state = P.phase_initial_state(lr['state_with_action']['config'])
        state, action = P.phase_step(state, records, env)
        per.append({'launch': lr['launch_index'], 'action_equal': CJ(action) == CJ(lr['action']),
                    'state_equal': CJ(state) == CJ(lr['state_with_action']),
                    'action': {k: action.get(k) for k in ('kind', 'stage', 'purpose', 'depth')}})
        if action.get('kind') != 'evaluate':
            return state, action, per, records
        for r in lr['records']:
            records[(tuple(r['z']), r['depth'])] = summary_of(r)
    state, action = P.phase_step(state, records, env)
    return state, action, per, records


def decisions(state, lat):
    """The decision-bearing parts of a phase state (independent of cache-hit bookkeeping)."""
    tab = state.get('table') or {}
    out = {'stage': state['stage'], 'terminated': state['terminated'], 'termination': state.get('termination'),
           'x0_exit_check': state.get('x0_exit_check'),
           'ladder_plans': [p.get('label') for p in ((state.get('ladder') or {}).get('plans') or [])],
           'table': [{k: r.get(k) for k in ('label', 'source', 'status', 'F_exit', 'dF_exit', 'bar', 'e', 'class')}
                     for r in tab.get('rows') or []],
           'efficiency': {k: (state.get('efficiency') or {}).get(k) for k in ('n_not_worse', 'n_rows', 'class_counts',
                                                                               'fraction', 'branch')},
           'candidates_picked': (state.get('candidate_selection') or {}).get('picked'),
           'candidates': []}
    for i, c in enumerate(state.get('candidates') or []):
        ps = c.get('poll_state')
        open_poll = ps is None and state.get('sub') is not None and state.get('cand_pos') == i
        if open_poll:
            ps = state['sub']   # the candidate's poll while it is open (P.phase_step: state['sub'])
        polls = []
        if ps:
            for p in (ps.get('polls') or []) + ([ps['poll']] if ps.get('poll') else []):
                polls.append({'index': p['index'], 'n': p['n'], 'pos': p['pos'], 'closed': p.get('closed'),
                              'classified': [{k: e.get(k) for k in ('pos', 'label', 'settled_class', 'exit_class',
                                                                    'dF_set', 'V_pair', 'gates_pass', 'failed')}
                                             for e in p['classified']]})
        out['candidates'].append({'label': c['label'], 'confirmed': c.get('confirmed'),
                                  'confirmation_result': (c.get('confirmation') or {}).get('result'),
                                  'dF_set': (c.get('confirmation') or {}).get('dF_set'),
                                  'exit_view_check_equal': (c.get('exit_view_check') or {}).get('equal'),
                                  'poll_skipped': c.get('poll_skipped'), 'poll_open': open_poll,
                                  'poll_search_id': ((ps or {}).get('config') or {}).get('search_id'),
                                  'poll_not_settled_review': P.not_settled_review(ps) if ps else None,
                                  'poll_termination': (ps or {}).get('termination'), 'polls': polls})
    return out


# Fields that differ between a continuation and the run it continues by construction, and decide nothing: the
# sub-searches' ids carry the phase id; a sub-search's action_index and budget_used count the requests issued IN THIS
# phase (a cached plan is classified without a request; the sub-search N_max is the domain size, non-binding).
BOOKKEEPING = ('sub_action.action_index', 'candidates[].poll_search_id', 'candidates[].poll_termination.budget_used')


def action_decision(a):
    sub = {k: v for k, v in (a.get('sub_action') or {}).items() if k != 'action_index'}
    return dict({k: a.get(k) for k in ('kind', 'stage', 'purpose', 'depth', 'zs', 'labels')}, sub_action=sub)


def strip_bookkeeping(d):
    import copy as _copy
    d = _copy.deepcopy(d)
    book = []
    for c in d.get('candidates') or []:
        pt = c.get('poll_termination') or {}
        book.append({'label': c['label'], 'poll_search_id': c.pop('poll_search_id', None),
                     'budget_used': pt.pop('budget_used', None) if isinstance(pt, dict) else None})
    return d, book


def phase_wide_review(state):
    return P.phase_not_settled_review(state)


def part_remap(data_root, out_dir):
    v2_root = os.path.join(data_root, V2_ROOT_REL)
    _sp, v2_sha, v2_spec = D.load_phase_spec(v2_root)
    lattice, _inst = D.build_lattice(data_root)
    env = D.LatticeEnv(lattice)
    oracle = D.make_oracle(v2_spec, lattice, lock_path=None)
    cache = D.phase_cache(v2_spec, oracle, data_root, lattice)
    launches = v2_launches(v2_root)
    res_v2 = D.load_json(os.path.join(v2_root, D.PHASE_RESULTS_FILE))
    ck('pre', f'v2: {len(launches)} complete launches, results after launch {res_v2["n_launches"]}, terminated '
              f'{res_v2["report"]["termination"]["reason"]}',
       len(launches) == res_v2['n_launches'] == 10
       and res_v2['report']['termination']['reason'] == 'stop_for_review_not_settled')
    # A
    remapped, rows_a = {}, []
    for lr in launches:
        for r in lr['records']:
            z, depth = tuple(r['z']), r['depth']
            ed = os.path.join(v2_root, r['eval_dir'])
            summ, rfile = oracle.summary_from_dir(ed, z, depth)
            same_file = D.sha256_file(rfile) == r['record_sha256']
            old = r['summary']
            remapped[(z, depth)] = summ
            changed = {k: [old.get(k), summ.get(k)] for k in sorted(set(old) | set(summ))
                       if k not in ('mapping_schema', 'solve_profile_reconciliation')
                       and CJ(old.get(k)) != CJ(summ.get(k))}
            rows_a.append({'launch': lr['launch_index'], 'label': lattice.label(z), 'depth': depth,
                           'eval_dir': r['eval_dir'], 'record_unchanged_since_launch': same_file,
                           'old_status': old.get('status'), 'new_status': summ.get('status'),
                           'old_guard': (old.get('guards') or {}).get('solve_profile_identity_holds'),
                           'new_guard': (summ.get('guards') or {}).get('solve_profile_identity_holds'),
                           'reconciliation_via': (summ.get('solve_profile_reconciliation') or {}).get('via'),
                           'changed_fields': sorted(changed), 'changed': changed if changed else None})
    ck('A', f'{len(rows_a)} v2 records re-read; every record file unchanged since its launch',
       all(r['record_unchanged_since_launch'] for r in rows_a))
    ch = [r for r in rows_a if r['changed_fields']]
    ck('A', 'the records whose summary changes are exactly the 4 with an ESSO recovery event, each from guard False to '
            'True via the extended identity',
       len(ch) == 4 and all(r['old_guard'] is False and r['new_guard'] is True
                            and r['reconciliation_via'] == 'extended_esso_recovery' for r in ch),
       [(r['label'], r['changed_fields']) for r in ch])
    # B
    sB, aB, perB, _rB = chain_replay(cache, launches, lambda r: r['summary'], env)
    ck('B', 'stored (v1) summaries: every launch action and state_with_action re-derived equal to the recorded ones',
       all(p['action_equal'] and p['state_equal'] for p in perB), [p for p in perB if not (p['action_equal']
                                                                                          and p['state_equal'])])
    ck('B', 'stored summaries: after launch 10 the phase terminates stop_for_review_not_settled with the recorded '
            'terminal state', aB.get('kind') == 'terminated' and CJ(sB) == CJ(res_v2['state']),
       [aB.get('kind'), (aB.get('termination') or {}).get('reason')])
    # C
    sC, aC, perC, _rC = chain_replay(cache, launches, lambda r: remapped[(tuple(r['z']), r['depth'])], env)
    ck('C', 're-mapped (v2) summaries: every launch action 1..10 re-derived equal to the recorded one (the corrected '
            'guard changes no request of v2)', all(p['action_equal'] for p in perC),
       [p for p in perC if not p['action_equal']])
    dB, dC = decisions(sB, lattice), decisions(sC, lattice)
    ck('C', 're-mapped summaries: after launch 10 the phase continues (an evaluate action), stage candidate_polls',
       aC.get('kind') == 'evaluate' and sC['stage'] == 'candidate_polls',
       [aC.get('kind'), sC['stage'], (aC.get('termination') or {}).get('reason')])
    ck('C', 'the ladder plans, the candidates picked and the efficiency branch are unchanged',
       dB['ladder_plans'] == dC['ladder_plans'] and dB['candidates_picked'] == dC['candidates_picked']
       and dB['efficiency']['branch'] == dC['efficiency']['branch'],
       [dB['ladder_plans'], dC['ladder_plans'], dB['candidates_picked'], dC['candidates_picked']])
    next_action = {k: aC.get(k) for k in ('kind', 'stage', 'purpose', 'depth', 'zs', 'sub_action')}
    next_action['labels'] = [lattice.label(tuple(z)) for z in aC.get('zs') or []]
    payload = {'v2_phase_spec_sha256': v2_sha, 'A_records': rows_a,
               'A_changed': [{k: r[k] for k in ('launch', 'label', 'depth', 'old_status', 'new_status',
                                                'changed_fields')} for r in ch],
               'B_per_launch': perB, 'B_decisions': dB, 'B_phase_not_settled_review': phase_wide_review(sB),
               'C_per_launch': perC, 'C_decisions': dC, 'C_phase_not_settled_review': phase_wide_review(sC),
               'C_next_action': next_action, 'C_state_sha256': D.sha256_text(CJ(sC)),
               'cache_records': sorted([lattice.label(z), d, s.get('served_as'), s.get('status')]
                                       for (z, d), s in cache.items())}
    return payload


def part_resume(data_root, out_dir, v3_root, remap_dir):
    rm = D.load_json(os.path.join(remap_dir, 'continuation_check.json'))
    ck('pre', 'the remap record passed', rm['all_checks_ok'] is True)
    C = rm['results']
    _sp, v3_sha, spec = D.load_phase_spec(v3_root)
    drift = D.code_drift(spec)
    ck('pre', 'no code drift against the v3 spec', not drift, drift)
    D.verify_predecessors_at_launch(spec, data_root)
    lattice, _inst = D.build_lattice(data_root)
    env = D.LatticeEnv(lattice)
    oracle = D.make_oracle(spec, lattice, lock_path=None)
    cache = D.phase_cache(spec, oracle, data_root, lattice)
    state, records, chain, _states = D.phase_replay(v3_root, spec, v3_sha, lattice, env, oracle, cache)
    ck('D', 'v3 root: no launch yet (replay chain empty)', chain == [], chain)
    state, action = P.phase_step(state, records, env)
    labels = [lattice.label(tuple(z)) for z in action.get('zs') or []]
    got = {k: action.get(k) for k in ('kind', 'stage', 'purpose', 'depth', 'zs', 'sub_action')}
    got['labels'] = labels
    ga, ca = action_decision(got), action_decision(C['C_next_action'])
    ck('D', 'the first v3 action equals the remap part C next action (v2 under the corrected guard): kind, stage, '
            'purpose, depth, plans, sub-action (bookkeeping fields apart: BOOKKEEPING)', CJ(ga) == CJ(ca), [ga, ca])
    action_bookkeeping = {'v3': (got.get('sub_action') or {}).get('action_index'),
                          'remap_C': (C['C_next_action'].get('sub_action') or {}).get('action_index')}
    cands = state.get('candidates') or []
    sub = state.get('sub') or {}
    ck('D', 'the first v3 action is candidate 2\'s poll (n7_y2025_p0.25_e1), settled, stage candidate_polls',
       action.get('kind') == 'evaluate' and state['stage'] == 'candidate_polls' and action.get('depth') == 'settled'
       and action.get('purpose') == 'candidate_poll' and len(cands) == 2 and state.get('cand_pos') == 1
       and cands[1]['label'] == 'n7_y2025_p0.25_e1' and str((sub.get('config') or {}).get('search_id')).endswith('_cand2_poll')
       and list(sub.get('incumbent') or []) == list(cands[1]['z']),
       [state['stage'], action.get('depth'), action.get('purpose'), state.get('cand_pos'),
        (sub.get('config') or {}).get('search_id')])
    ck('D', 'none of the requested plans is cached', not [z for z in action['zs']
                                                          if (tuple(z), action['depth']) in records], labels)
    src = {}
    for (z, d), s in cache.items():
        src.setdefault(s.get('served_as'), 0)
        src[s.get('served_as')] += 1
    v2_entries = [e for s in spec['cache_sources'] if s.get('kind') == 'continuation_predecessor' for e in s['entries']]
    v2_root = os.path.join(data_root, V2_ROOT_REL)
    n_v2 = sum(len(lr['records']) for lr in v2_launches(v2_root))
    ck('D', f'every v2 record ({n_v2}) is indexed in the v3 cache, none ignored',
       len(v2_entries) == n_v2 and not [i for s in spec['cache_sources'] for i in s.get('ignored') or []
                                        if s.get('kind') == 'continuation_predecessor'],
       [len(v2_entries), n_v2])
    ev = state['evaluated']
    ck('D', 'the v3 state has evaluated nothing (every decision so far read from cache)', ev == [], len(ev))
    dD = decisions(state, lattice)
    dC = C['C_decisions']
    sD, bD = strip_bookkeeping(dD)
    sC, bC = strip_bookkeeping(dC)
    same = {k: CJ(sD[k]) == CJ(sC[k]) for k in sC}
    ck('D', 'the v3 decisions (table, efficiency, candidates, confirmations, both polls so far) equal remap part C',
       all(same.values()), [k for k, v in same.items() if not v])
    hours = spec['phase']['ceiling']
    return {'v3_phase_spec_sha256': v3_sha, 'first_action': got, 'stage': state['stage'],
            'cache_served_as_counts': src, 'n_cache_records': len(cache), 'n_v2_entries': len(v2_entries),
            'n_v2_records': n_v2, 'decisions': dD, 'decisions_equal_remap_C': same,
            'bookkeeping': {'rule': BOOKKEEPING, 'sub_action_index': action_bookkeeping, 'v3': bD, 'remap_C': bC},
            'phase_not_settled_review': phase_wide_review(state),
            'ceiling': {k: hours.get(k) for k in ('ceiling_h', 'carried_h', 'raised_by_continuation')},
            'cache_records': sorted([lattice.label(z), d, s.get('served_as'), s.get('status')]
                                    for (z, d), s in cache.items())}


def part_freeze_sim(data_root, out_dir, remap_dir):
    """E (zero solves; NOTHING frozen in the repository): the v3 freeze as `--phase-freeze` would run it, in process,
    with the tracked-artefact list of the collision scan and the cache-root key audit (`committed_spec_files`)
    AUGMENTED by the v2 root's launch artefacts of the same name patterns (launches/launch_*.json and the campaign_spec_*
    in each launch dir: what tracking them would add), frozen into a SCRATCH root under --out; then part D on it. Why:
    the real freeze refused at the cache-root key audit because only v2's spec is tracked in the v2 root."""
    v2 = os.path.join(data_root, V2_ROOT_REL)
    extra = sorted(f for f in D.files_under(os.path.join(v2, 'launches'))
                   if os.path.basename(f).startswith('campaign_spec_')
                   or (os.path.basename(f).startswith('launch_') and f.endswith('.json') and '/launches/' in f))
    tracked = D.committed_spec_files(data_root)
    added = [f for f in extra if f not in tracked]
    orig = D.committed_spec_files
    D.committed_spec_files = lambda root: orig(root) + added
    try:
        pp = os.path.join(data_root, 'data', 'SRP1', 'Results', 'P515S56', 'phase2x_ev_v2_predictions.json')
        pred = dict(D.load_json(pp), source_file_sha256=D.sha256_file(pp))
        spec = D.phase_spec(phase_id='s56phase2x_ev_v3', oracle='harness', test_only=False, data_root=data_root,
                            predictions=pred,
                            unlaunched_predecessors=[(os.path.join('data', 'SRP1', 'Results', 'P515S56',
                                                                   'phase2x_ev_v1'), 'de4e947e')],
                            continuation_of=[(V2_ROOT_REL, '18a224d1')])
        root = os.path.join(out_dir, 'sim_root', 'phase2x_ev_v3')
        spec['collision_scan'] = D.freeze_collision_scan(spec, D.build_lattice(data_root)[0], root, data_root)
        path, digest = D.phase_freeze(root, spec)
    finally:
        D.committed_spec_files = orig
    ck('E', f'simulated freeze: the cache-root key audit holds with the {len(added)} v2 launch artefacts scanned',
       spec['cache_root_key_audit']['ok'] is True, spec['cache_root_key_audit'])
    ck('E', 'simulated freeze: collision scan clean', spec['collision_scan']['collisions'] == [], spec['collision_scan'])
    ce = spec['phase']['ceiling']
    ck('E', 'simulated freeze: ceiling 120 h, not raised, the v2 measured hours carried',
       ce['ceiling_h'] == 120.0 and ce['raised_by_continuation'] is None and 17.9 < ce['carried_h'] < 18.1, ce)
    pol = spec['policy']
    ck('E', 'simulated freeze: R, e_EV, K_EV, exit-view check, efficiency threshold, pilot exit views as in v2',
       pol['R'] == 10000.0 and round(pol['e_ev']) == 17494 and pol['k_ev'] == 300 and pol['exit_view_check'] == 'stop'
       and spec['phase']['efficiency']['threshold'] == [1, 2] and spec['phase']['pilot_exit_views'] is True,
       {k: pol[k] for k in ('R', 'e_ev', 'k_ev', 'exit_view_check')})
    res = part_resume(data_root, out_dir, root, remap_dir)
    return {'simulated': True, 'sim_spec_rel': os.path.relpath(path, out_dir), 'sim_spec_sha256': digest,
            'added_to_tracked_list': [os.path.relpath(f, data_root) for f in added],
            'cache_root_key_audit': spec['cache_root_key_audit'], 'collision_scan': spec['collision_scan'],
            'ceiling': {k: ce[k] for k in ('ceiling_h', 'carried_h', 'raised_by_continuation')},
            'predecessors': spec['predecessors'], 'code_sha256': spec['code_sha256'], 'git_head': spec['git_head'],
            'resume': res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('remap', 'resume', 'freeze-sim'), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--v3-root')
    ap.add_argument('--remap')
    ap.add_argument('--data-root', default=CODE_ROOT)
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
    if os.path.exists(out_dir) and [n for n in os.listdir(out_dir) if not n.endswith('.log')]:
        log(f'REFUSE: {out_dir} exists and holds non-log files')
        return 2
    for forbidden in (V2_ROOT_REL, args.v3_root):
        if forbidden and os.path.realpath(out_dir).startswith(os.path.realpath(os.path.join(data_root, forbidden))):
            log(f'REFUSE: --out inside a phase root ({forbidden})')
            return 2
    os.makedirs(out_dir, exist_ok=True)
    started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
    if args.part == 'remap':
        res = part_remap(data_root, out_dir)
    elif args.part == 'freeze-sim':
        rmd = args.remap if os.path.isabs(args.remap) else os.path.join(data_root, args.remap)
        res = part_freeze_sim(data_root, out_dir, rmd)
    else:
        v3 = args.v3_root if os.path.isabs(args.v3_root) else os.path.join(data_root, args.v3_root)
        rmd = args.remap if os.path.isabs(args.remap) else os.path.join(data_root, args.remap)
        res = part_resume(data_root, out_dir, v3, rmd)
    gf = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == []', not gf, gf)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code = {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s44_campaign_harness.py',
        'p515_s56_neighbourhood.py', 'p516_lattice.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *code], cwd=CODE_ROOT, capture_output=True,
                           text=True).stdout
    payload = {'schema': 'p515_s56_phase2x_v3_continuation_check_v1', 'part': args.part, 'script': SCRIPT,
               'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'wall_s': time.time() - t0, 'interpreter': sys.executable, 'python': platform.python_version(),
               'git_head': head, 'code_files_dirty_vs_head': dirty,
               'objective_convention': ('F = I(x) + gross_operational_cost - TV credit (oracle mapping '
                                        f'{D.MAPPING_SCHEMA}), EUR of total discounted planning cost; dF_exit = '
                                        'F_exit(x) - F_exit(x0); dF_set = F_set(x) - F_set(incumbent)'),
               'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any '
                              'repository import, verify(0)); pickle blocked (0); phase roots read only',
               'results': res, 'guards': {'counts': dict(GUARD.counts), 'verify_0': gf, 'pickle': dict(PICKLE_COUNTS)},
               'code_files': code, 'checks': CHECKS,
               'failed_checks': [f"[{c['test']}] {c['check']}" for c in CHECKS if not c['ok']],
               'all_checks_ok': all(c['ok'] for c in CHECKS)}
    sha = D.write_once_json(os.path.join(out_dir, 'continuation_check.json'), payload)
    log(f'checks {sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} ok; wall {time.time() - t0:.0f} s; record sha256 {sha}; '
        f'failed {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
