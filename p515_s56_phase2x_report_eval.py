"""P5.15 S56 EV 2x PHASE -- END-OF-PHASE REPORT EVALUATION (ZERO SOLVES; read only on both phase roots).

Authority: TASKS.md step 5b ("COMPLETE 2026-11-26 ... End-of-phase report Worker active"); the Planner's end-of-phase
task of 2026-11-26. Every number in P5_15_S56_PHASE2X_REPORT.md that is not a field copied from a phase record is
defined here (CLAUDE.md: preserve the formula, not only the inputs).

INPUTS (read only): the v2 root (spec 18a224d1, 10 launches, stopped exit 5) and the v3 continuation root (spec
600bbceb, 3 launches, phase_complete); the pilot cache root through the v3 spec's cache index.

PART A  the v3 phase as the driver sees it: `phase_cache` (pilot + v2 continuation source, mapped with the working-tree
        mapping v2 = the corrected solve-profile guard) and `phase_replay` of the v3 chain (every launch record, manifest,
        hash chain, action and state re-derived), then one `phase_step` -> the terminal state, compared with
        phase_results.json (canonical JSON equality).
PART B  per-evaluation facts of every evaluation either phase launched (v2 1-10, v3 1-3): the mapped summary (v2 records
        re-mapped under the corrected guard, as the v3 cache serves them), and from evaluation_record.json: peak RSS
        (child_python_process_ru_maxrss / 2^30), local_solve_failures, n_esso_recovery_events, the child wall; from
        esso_recovery_events_s39_D.jsonl: each event's node and the event node's cohorts (E in MWh).
        LEAK (loss-priced) = summary.leak.priced_loss_eur (settled summaries; the mapping's leak detector);
        ESSO SLACK = summary.V_parts.G.esso_slack (EUR, settled summaries) and summary.esso_slack_max_abs_net.
PART C  hours: per launch (phase_results.hours.per_launch + the launch records' UTC stamps), per stage (sum of launch
        walls by `purpose`), against the 120 h ceiling and the v2 spec's frozen projection (settled_direct|*|search_low:
        blocks x per-block projection).
PART D  the marginal-value table (v3 state, 22 rows) with, for every row plan that has a settled record (any source),
        F_set and `P.classify_settled` against x0's settled record.
PART E  candidates (confirmation, exit-view check, complete poll tables), pool, search (incumbent history, terminal
        poll), and the INDISTINGUISHABLE set (plan v4.1 section 4): every evaluated plan within R of the final
        incumbent -- settled-evaluated plans by |F_set(x) - F_set(inc)| < R; exit-only plans by
        |F_exit(x) - F_exit(inc)| < R (exit values only against exit values, P.classify_exit with e_floor = e_EV).
PART F  the zero-solve neighbourhood check of the final incumbent: `p515_s56_neighbourhood.s56_neighbourhood`,
        `p516_lattice.GeneralLattice.neighbourhood` (R3 raw, dropped with reasons) and `raw_added_moves` (d, e, f and the
        informational e_opp), each refused raw move with `reasons` and the independent `family_violations`; I(x) vs B;
        |N| vs the policy CAP; the R3 raw size bound.
PART G  the zero-solve collision check for tracking v3's campaign specs: eval keys carried by v3's campaign specs,
        launch records and manifests (`_collect_eval_keys`) against (i) the 2x domain keys of the v3 configuration and
        (ii) the same domain under flex_price_multiplier 1.0 (all else equal: the 1x configuration as far as the key
        is concerned); `collision_scan` simulated for a later 1x freeze and a 2x control freeze (with and without v2/v3
        declared as cache roots), over the tracked artefacts plus v3's launch-dir artefacts.
PART H  recorded predictions Q1-Q7 against outcomes (phase-wide: v2 + v3 records).
PART I  projections (PROJECTED, from measured launch walls and the cache state): the 2x controls and the 1x phase.

MANIFEST MODE (--manifest): the hash inventory of both phase roots (every file), this script's output directory and its
logs; per file sha256, bytes, and `committed` by the S56 selection rule `p515_s56_ev_harness_tests.committed_by_rule`
(no size exemption applies here), with the PART G outcome applied to v3's campaign specs.

GUARDS: SolveProfileGuard(permitted=()) installed before any repository import, verify(0); pickle load / loads blocked
(0). Writes only --out (a NEW directory, outside both phase roots) or the manifest file (write-once, outside both roots).
Launch (attached, both streams captured):
  python -u p515_s56_phase2x_report_eval.py --out data/SRP1/Results/P515S56/phase2x_ev_end_report_eval \\
      > data/SRP1/Results/P515S56/phase2x_ev_end_report_eval_stdout.log \\
      2> data/SRP1/Results/P515S56/phase2x_ev_end_report_eval_stderr.log
  python -u p515_s56_phase2x_report_eval.py --manifest --eval-dir <the --out above> \\
      --target data/SRP1/Results/P515S56/phase2x_ev_end_evidence_manifest_sha256.json
Exit: 0 written and every check holds; 3 written with a failed check; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 2x phase report evaluation (never solves)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('report evaluation: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('report evaluation: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
SCHEMA = 'p515_s56_phase2x_report_eval_v1'
REL = os.path.join('data', 'SRP1', 'Results', 'P515S56')
V2_ROOT_REL = os.path.join(REL, 'phase2x_ev_v2')
V3_ROOT_REL = os.path.join(REL, 'phase2x_ev_v3')
V2_SPEC_SHA = '18a224d13f9bc04e3c3177fb678ced009f5f845b77ab895817228b2ad28c1ce8'
V3_SPEC_SHA = '600bbceb98b8d88a3a487f95ba8201bd037759b7696e1c182bc4c21a423facd7'
GIB = float(1 << 30)
ESSO_EVENTS_FILE = 'esso_recovery_events_s39_D.jsonl'
CHECKS = []
CJ = P.canonical_json


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(part, name, ok, detail=None):
    CHECKS.append({'part': part, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{part}] {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


def _h(seconds):
    return float(seconds) / 3600.0


def _stats(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return None
    return {'n': len(vals), 'min': min(vals), 'max': max(vals), 'mean': sum(vals) / len(vals), 'sum': sum(vals)}


def _launches(root):
    _idx, complete = D._complete_indices(root)
    return [D.load_json(D.launch_paths(root, n)[1]) for n in complete]


# ======================================================================================================================
#  PART A -- the v3 phase as the driver sees it
# ======================================================================================================================
def part_a(data_root):
    v3_root = os.path.join(data_root, V3_ROOT_REL)
    _sp, v3_sha, spec = D.load_phase_spec(v3_root)
    ck('A', 'v3 spec sha256 = 600bbceb...', v3_sha == V3_SPEC_SHA, v3_sha)
    drift = D.code_drift(spec)
    ck('A', 'no code drift against the v3 spec (driver, policy, neighbourhood, lattice, harness, gate_result_io)',
       not drift, drift)
    D.verify_predecessors_at_launch(spec, data_root)
    ck('A', 'predecessors re-verified (v1 un-launched, v2 continuation)', True)
    lattice, _inst = D.build_lattice(data_root)
    env = D.LatticeEnv(lattice)
    oracle = D.make_oracle(spec, lattice, lock_path=None)
    cache = D.phase_cache(spec, oracle, data_root, lattice)
    state, records, chain, _states = D.phase_replay(v3_root, spec, v3_sha, lattice, env, oracle, cache)
    ck('A', 'v3 chain replays: 3 launches, every record / manifest / action / state re-derived',
       len(chain) == 3, len(chain))
    state, action = P.phase_step(state, records, env)
    res = D.load_json(os.path.join(v3_root, D.PHASE_RESULTS_FILE))
    ck('A', 'one more phase_step terminates phase_complete', action.get('kind') == 'terminated'
       and (action.get('termination') or {}).get('reason') == 'phase_complete', action)
    ck('A', 'the replayed terminal state equals phase_results.json state (canonical JSON)',
       CJ(state) == CJ(res['state']))
    man = D.load_json(os.path.join(v3_root, D.PHASE_RESULTS_MANIFEST))
    ck('A', 'phase_results manifest holds the results hash',
       man.get(D.PHASE_RESULTS_FILE) == D.sha256_file(os.path.join(v3_root, D.PHASE_RESULTS_FILE)), man)
    return spec, lattice, env, oracle, cache, records, state, res


# ======================================================================================================================
#  PART B -- per-evaluation facts
# ======================================================================================================================
def _cohorts(lattice, z):
    import p516_lattice as GL
    return [{'node': n, 'year': y, 'P_mva': GL.GRANULE_P_MVA * lattice.pair(tuple(z), n, y)[0],
             'E_mwh': GL.GRANULE_E_MWH * lattice.pair(tuple(z), n, y)[1]} for n, y in lattice.active_cohorts(tuple(z))]


def part_b(data_root, lattice, oracle, records):
    out = []
    for phase, rel in (('v2', V2_ROOT_REL), ('v3', V3_ROOT_REL)):
        root = os.path.join(data_root, rel)
        for lr in _launches(root):
            for r in lr['records']:
                z, depth = tuple(r['z']), r['depth']
                ed = os.path.join(root, r['eval_dir'])
                summ = records.get((z, depth))
                stored = r['summary']
                if summ is None:   # v3 records are in the replay's records; v2 records in the v3 cache
                    raise D.Refused(f'{phase} record {lattice.label(z)} {depth} not served by the v3 phase')
                rec = D.load_json(os.path.join(ed, 'evaluation_record.json'))
                ev_path = os.path.join(ed, ESSO_EVENTS_FILE)
                events = []
                if os.path.exists(ev_path):
                    with open(ev_path) as handle:
                        events = [json.loads(line) for line in handle if line.strip()]
                coh = _cohorts(lattice, z)
                ev_rows = []
                for e in events:
                    node_coh = [c for c in coh if c['node'] == int(e['node_id'])]
                    ev_rows.append({'node_id': e['node_id'], 'cycle': e.get('cycle'), 'tier': e.get('tier'),
                                    'tier2_attempted': e.get('tier2_attempted'),
                                    'recovery_succeeded': e.get('recovery_succeeded'),
                                    'primary_result': e.get('primary_result'),
                                    'node_cohorts': node_coh,
                                    'node_has_0p5_mwh_cohort': any(c['E_mwh'] == 0.5 for c in node_coh)})
                nfs = rec.get('network_failures_summary') or {}
                rss = (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss')
                leak = summ.get('leak') or {}
                g = (summ.get('V_parts') or {}).get('G') or {}
                out.append({
                    'phase': phase, 'launch': lr['launch_index'], 'purpose': lr['action']['purpose'],
                    'stage': lr['action']['stage'], 'depth': depth, 'z': list(z), 'label': lattice.label(z),
                    'candidate_key': lattice.key(z), 'eval_key': r['eval_key'], 'eval_dir': os.path.relpath(ed, data_root),
                    'I_x_eur': lattice.investment_cost(z), 'cohorts': coh,
                    'status_as_run': stored.get('status'), 'status': summ.get('status'),
                    'guard_as_run': (stored.get('guards') or {}).get('solve_profile_identity_holds'),
                    'guard': (summ.get('guards') or {}).get('solve_profile_identity_holds'),
                    'reconciliation_via': (summ.get('solve_profile_reconciliation') or {}).get('via'),
                    'F': summ.get('F'), 'bar': summ.get('bar'), 'read_cycle': summ.get('read_cycle'),
                    'V': summ.get('V'), 'gates_pass': summ.get('gates_pass'), 'gates': summ.get('gates'),
                    'exit_view': summ.get('exit_view'),
                    'leak_priced_loss_eur': leak.get('priced_loss_eur'), 'leak_priced_full_eur': leak.get('priced_full_eur'),
                    'leak_L_weighted_mwh': leak.get('L_weighted_mwh'), 'leak_within_ceiling': leak.get('within_ceiling'),
                    'esso_slack_eur': g.get('esso_slack'), 'esso_slack_max_abs_net': summ.get('esso_slack_max_abs_net'),
                    'closure_max_abs_net_mwh': summ.get('closure_max_abs_net_mwh'),
                    'child_wall_s': summ.get('wall_s'),
                    'peak_rss_gib': (float(rss) / GIB) if rss is not None else None,
                    'local_solve_failures': rec.get('local_solve_failures'),
                    'n_esso_recovery_events_record': nfs.get('n_esso_recovery_events'),
                    'network_failure_classes': nfs.get('classes'),
                    'esso_events': ev_rows, 'n_esso_events_file': len(events)})
    ck('B', 'every record of both phases is served by the v3 phase (v2 via its cache, v3 via its replay)', True,
       len(out))
    ck('B', 'ESSO events file count == record n_esso_recovery_events on every evaluation',
       all(e['n_esso_events_file'] == e['n_esso_recovery_events_record'] for e in out),
       [(e['label'], e['n_esso_events_file'], e['n_esso_recovery_events_record']) for e in out
        if e['n_esso_events_file'] != e['n_esso_recovery_events_record']])
    return out


# ======================================================================================================================
#  PART C -- hours
# ======================================================================================================================
STAGE_OF_PURPOSE = {'x0_exit': 'screen', 'screen': 'screen', 'ladder': 'ladder',
                    'candidate_confirmation': 'candidate_confirmations', 'candidate_poll': 'candidate_polls',
                    'pool_settled': 'pool', 'search': 'search'}


def part_c(data_root, evals):
    rows = []
    for phase, rel in (('v2', V2_ROOT_REL), ('v3', V3_ROOT_REL)):
        root = os.path.join(data_root, rel)
        res = D.load_json(os.path.join(root, D.PHASE_RESULTS_FILE))
        lrs = {lr['launch_index']: lr for lr in _launches(root)}
        for pl in res['hours']['per_launch']:
            lr = lrs[pl['launch']]
            ck('C', f'{phase} launch {pl["launch"]}: wall in phase_results == launch record wall',
               pl['wall_s'] == lr['wall_s'] and pl['purpose'] == lr['action']['purpose'])
            child = [e['child_wall_s'] for e in evals if e['phase'] == phase and e['launch'] == pl['launch']]
            rows.append({'phase': phase, 'launch': pl['launch'], 'purpose': pl['purpose'],
                         'stage': STAGE_OF_PURPOSE[pl['purpose']], 'depth': pl['depth'],
                         'n_evaluations': pl['n_evaluations'], 'wall_s': pl['wall_s'], 'wall_h': _h(pl['wall_s']),
                         'started_utc': lr['started_utc'], 'finished_utc': lr['finished_utc'],
                         'labels': [e['label'] for e in evals
                                    if e['phase'] == phase and e['launch'] == pl['launch']],
                         'child_wall_h': _stats([_h(c) for c in child])})
        ck('C', f'{phase}: interrupted launches 0 (interrupted hours 0)', res['hours']['interrupted'] == []
           and res['hours']['interrupted_launches_h'] == 0)
    by_stage = {}
    for r in rows:
        s = by_stage.setdefault(r['stage'], {'v2_h': 0.0, 'v3_h': 0.0, 'launches': 0, 'evaluations': 0})
        s[f"{r['phase']}_h"] += r['wall_h']
        s['launches'] += 1
        s['evaluations'] += r['n_evaluations']
    for s in ('screen', 'ladder', 'candidate_confirmations', 'candidate_polls', 'pool', 'search'):
        by_stage.setdefault(s, {'v2_h': 0.0, 'v3_h': 0.0, 'launches': 0, 'evaluations': 0})
    for s in by_stage.values():
        s['total_h'] = s['v2_h'] + s['v3_h']
    v2_h = sum(r['wall_h'] for r in rows if r['phase'] == 'v2')
    v3_h = sum(r['wall_h'] for r in rows if r['phase'] == 'v3')
    res2 = D.load_json(os.path.join(data_root, V2_ROOT_REL, D.PHASE_RESULTS_FILE))
    res3 = D.load_json(os.path.join(data_root, V3_ROOT_REL, D.PHASE_RESULTS_FILE))
    ck('C', 'sum of v2 launch walls == v2 cumulative_h', abs(v2_h - res2['hours']['cumulative_h']) < 1e-9)
    ck('C', 'sum of v3 launch walls == v3 cumulative_h', abs(v3_h - res3['hours']['cumulative_h']) < 1e-9)
    ck('C', 'v3 carried_h == v2 cumulative_h', res3['prediction_inputs']['carried_h'] == res2['hours']['cumulative_h'])
    # the frozen projection (v2 spec): the settled-direct, no-move case (the branch and shape that ran)
    spec2 = D.load_json(os.path.join(data_root, V2_ROOT_REL, 'phase_spec_18a224d1.json'))
    ph = spec2['projected_hours']
    pe, ps = ph['per_block_h']['exit'], ph['per_block_h']['settled']
    case = ph['cases']['settled_direct|cand_small|search_low']
    proj = {'screen': case['screen_blocks'] * pe, 'ladder': case['ladder_blocks'] * pe,
            'candidate_confirmations': case['confirmation_blocks'] * ps,
            'candidate_polls': case['poll_blocks_candidates'] * ps, 'search': case['search_poll_blocks'] * ps}
    proj_total = sum(proj.values())
    ck('C', 'the projection re-composes to the spec case total (65.24 h)', abs(proj_total - case['hours']) < 1e-6,
       [proj_total, case['hours']])
    per_block = {'exit': [r['wall_h'] for r in rows if r['depth'] == 'exit'],
                 'settled': [r['wall_h'] for r in rows if r['depth'] == 'settled'],
                 'settled_block_of_5': [r['wall_h'] for r in rows if r['depth'] == 'settled' and r['n_evaluations'] == 5],
                 'exit_block_of_5': [r['wall_h'] for r in rows if r['depth'] == 'exit' and r['n_evaluations'] == 5]}
    return {'per_launch': rows, 'by_stage': by_stage, 'v2_h': v2_h, 'v3_h': v3_h, 'total_h': v2_h + v3_h,
            'ceiling_h': 120.0, 'share_of_ceiling': (v2_h + v3_h) / 120.0,
            'projection_v2_spec': {'case': 'settled_direct|cand_small|search_low (= cand_large: same blocks)',
                                   'blocks': {k: case[k] for k in ('screen_blocks', 'ladder_blocks',
                                                                    'confirmation_blocks', 'poll_blocks_candidates',
                                                                    'search_poll_blocks')},
                                   'per_block_h': {'exit': pe, 'settled': ps}, 'by_stage_h': proj,
                                   'total_h': proj_total},
            'measured_block_walls_h': {k: _stats(v) for k, v in per_block.items()},
            'child_walls_h': {d: _stats([_h(e['child_wall_s']) for e in evals if e['depth'] == d])
                              for d in P.DEPTHS}}


# ======================================================================================================================
#  PART D -- the marginal-value table with settled values
# ======================================================================================================================
def _settled(records, z):
    return records.get((tuple(z), P.DEPTH_SETTLED))


def part_d(lattice, records, state):
    x0 = tuple(state['config']['x0'])
    x0s = _settled(records, x0)
    rows = []
    for r in state['table']['rows']:
        s = _settled(records, r['z'])
        row = {k: r.get(k) for k in ('label', 'source', 'node', 'year', 'zp', 'ze', 'duration', 'status', 'F_exit',
                                     'bar', 'dF_exit', 'e', 'class', 'served_as')}
        row['I_x_eur'] = lattice.investment_cost(tuple(r['z']))
        if s is not None:
            c = P.classify_settled(s, x0s)
            row['settled'] = {'status': s.get('status'), 'F_set': s.get('F'), 'dF_set_vs_x0': c['dF'],
                              'V_pair_vs_x0': c['V_pair'], 'A_sum_vs_x0': c['A_sum'], 'class_vs_x0': c['class'],
                              'failed': c['failed'], 'served_as': s.get('served_as'),
                              'exit_view': s.get('exit_view')}
        else:
            row['settled'] = None
        rows.append(row)
    ck('D', '22 rows (18 screen + 4 ladder)', len(rows) == 22 and sum(r['source'] == 'screen' for r in rows) == 18)
    return {'x0': {'F_exit': state['table']['x0']['F_exit'], 'bar_exit': state['table']['x0']['bar'],
                   'served_as': state['table']['x0']['served_as'], 'F_set': x0s.get('F'), 'status_set': x0s.get('status')},
            'rows': rows, 'n_with_settled': sum(r['settled'] is not None for r in rows)}


# ======================================================================================================================
#  PART E -- candidates, polls, pool, search, the indistinguishable set
# ======================================================================================================================
def _poll_rows(lattice, poll, order_by_pos):
    out = []
    for e in poll['classified']:
        o = order_by_pos.get(e['pos'], {})
        out.append({'pos': e['pos'], 'label': e['label'], 'move': o.get('move'), 'class_of_move': o.get('class'),
                    'settled_class': e.get('settled_class'), 'dF_set': e.get('dF_set'), 'V_pair': e.get('V_pair'),
                    'A_sum': e.get('A_sum'), 'gates_pass': e.get('gates_pass'), 'failed': e.get('failed'),
                    'cache_hit': e.get('cache_hit'), 'I_x_eur': lattice.investment_cost(tuple(e['z']))})
    return out


def part_e(lattice, records, state, evals, R, e_ev):
    cands = []
    for c in state['candidates']:
        ps = c['poll_state']
        polls = []
        for p in ps['polls'] + ([ps['poll']] if ps.get('poll') else []):
            order_by_pos = {o['pos']: o for o in p['order']}
            polls.append({'index': p['index'], 'N': p['n'], 'k_classified': len(p['classified']),
                          'closed': p.get('closed'), 'cache_hits_at_start': p.get('cache_hits_at_start'),
                          'rows': _poll_rows(lattice, p, order_by_pos)})
        exit_rec = records.get((tuple(c['z']), P.DEPTH_EXIT))
        cands.append({'label': c['label'], 'z': c['z'], 'I_x_eur': lattice.investment_cost(tuple(c['z'])),
                      'F_exit': c['F_exit'], 'confirmation': c['confirmation'], 'confirmed': c['confirmed'],
                      'exit_view_check': c['exit_view_check'],
                      'exit_record_served_as': (exit_rec or {}).get('served_as'),
                      'polls': polls, 'poll_report': c['poll_report']})
    srch = state['search']['state']
    spolls = []
    for p in srch['polls'] + ([srch['poll']] if srch.get('poll') else []):
        order_by_pos = {o['pos']: o for o in p['order']}
        spolls.append({'index': p['index'], 'N': p['n'], 'k_classified': len(p['classified']),
                       'closed': p.get('closed'), 'cache_hits_at_start': p.get('cache_hits_at_start'),
                       'requested_in_poll': p.get('requested_in_poll'),
                       'partition': {k: sum(1 for e in p['classified'] if e.get('settled_class') == k)
                                     for k in ('improver', 'worse', 'unresolved', 'not_settled', 'harness_error')},
                       'rows': _poll_rows(lattice, p, order_by_pos)})
    inc = tuple(srch['incumbent'])
    inc_set = _settled(records, inc)
    inc_exit = records.get((inc, P.DEPTH_EXIT))
    # indistinguishable set: every evaluated plan (any source the phase read) within R of the final incumbent
    plans = {}
    for (z, d), s in records.items():
        plans.setdefault(z, {})[d] = s
    near = []
    for z, by in sorted(plans.items(), key=lambda kv: lattice.label(kv[0])):
        if z == inc:
            continue
        if P.DEPTH_SETTLED in by and by[P.DEPTH_SETTLED].get('status') == 'settled':
            c = P.classify_settled(by[P.DEPTH_SETTLED], inc_set)
            if abs(c['dF']) < R:
                near.append({'label': lattice.label(z), 'depth': 'settled', 'dF': c['dF'], 'V_pair': c['V_pair'],
                             'A_sum': c['A_sum'], 'class': c['class'], 'failed': c['failed'],
                             'I_x_eur': lattice.investment_cost(z)})
        elif P.DEPTH_EXIT in by and by[P.DEPTH_EXIT].get('status') == 'certified':
            c = P.classify_exit(by[P.DEPTH_EXIT], inc_exit, R, e_ev)
            if abs(c['dF']) < R:
                near.append({'label': lattice.label(z), 'depth': 'exit (no settled record)', 'dF': c['dF'],
                             'e': c['e'], 'class': c['class'], 'I_x_eur': lattice.investment_cost(z),
                             'served_as': by[P.DEPTH_EXIT].get('served_as')})
    n_plans = len(plans)
    return {'candidates': cands, 'pool': state['pool'], 'search': {
        'start': srch['config']['start'], 'start_label': lattice.label(tuple(srch['config']['start'])),
        'incumbent_history': srch['incumbent_history'], 'final_incumbent': list(inc),
        'final_incumbent_label': lattice.label(inc), 'termination': srch['termination'],
        'budget_used': srch['budget_used'], 'polls': spolls,
        'F_set_inc': inc_set.get('F'), 'F_exit_inc': (inc_exit or {}).get('F')},
        'indistinguishable': {'rule': 'plan v4.1 section 4: every evaluated plan within R of the final incumbent; '
                                      'settled plans |F_set(x) - F_set(inc)| < R; plans with an exit record only: '
                                      '|F_exit(x) - F_exit(inc)| < R (exit values against exit values)',
                              'R': R, 'n_plans_with_any_record': n_plans, 'plans': near}}


# ======================================================================================================================
#  PART F -- the neighbourhood of the final incumbent (zero solves)
# ======================================================================================================================
def part_f(lattice, z_inc, polled_labels):
    import p515_s56_neighbourhood as NB
    z_inc = tuple(z_inc)
    nb = NB.s56_neighbourhood(lattice, z_inc)
    r3 = lattice.neighbourhood(z_inc)
    raw, triggers = NB.raw_added_moves(lattice, z_inc)
    B = lattice.inst.budget_eur
    I_inc = lattice.investment_cost(z_inc)
    pts_n = [p for p in nb['points'] if p['class'] in NB.CLASS_ORDER]   # e_opp is informational, not in N(x)
    pts_info = [p for p in nb['points'] if p['class'] not in NB.CLASS_ORDER]
    in_n = {lattice.label(tuple(p['z'])): p for p in pts_n}
    generated = []
    for p in r3['dropped']:
        generated.append(('r3', p['part'], p['move'], tuple(p['z'])))
    for p in r3['points']:
        generated.append(('r3', p['part'], p['move'], tuple(p['z'])))
    for cls, move, z in raw:
        generated.append((cls, None, move, tuple(z)))
    rows = []
    for cls, part, move, z in generated:
        why = lattice.reasons(z)
        fam = {k: v for k, v in lattice.family_violations(z).items() if v}
        lab = lattice.label(z)
        rows.append({'class': cls, 'part': part, 'move': move, 'label': lab, 'admissible': not why,
                     'in_N': lab in in_n and cls != 'e_opp', 'reasons': why,
                     'family_violations': {k: [list(x) if isinstance(x, tuple) else x for x in v]
                                           for k, v in fam.items()},
                     'I_x_eur': lattice.investment_cost(z), 'I_over_B_eur': lattice.investment_cost(z) - B,
                     'budget_is_sole_family': set(fam) == {'energy_storage_investment'}})
    refused = [r for r in rows if not r['admissible']]
    ck('F', '|N(final incumbent)| = 10 and equals the polled set',
       nb['size'] == 10 and set(in_n) == set(polled_labels), [nb['size'], sorted(set(in_n) ^ set(polled_labels))])
    ck('F', '|N| <= CAP (80)', nb['size'] <= P.CAP)
    cum = lattice.cumulative_energy(z_inc)
    return {'incumbent': list(z_inc), 'label': lattice.label(z_inc), 'I_x_eur': I_inc, 'budget_eur': B,
            'budget_slack_eur': B - I_inc, 'cumulative_energy_mwh': {f'{n}|{y}': v for (n, y), v in cum.items()},
            'max_capacity_mwh': lattice.inst.max_capacity_mwh,
            'duration_window_h': [lattice.inst.min_ratio, lattice.inst.max_ratio],
            'size': nb['size'], 'size_r3': nb['size_r3'], 'counts': nb['counts'], 'cap': P.CAP,
            'r3_raw': r3['n_raw'], 'r3_raw_by_part': r3['n_raw_by_part'], 'r3_size_bound': r3['size_bound'],
            'f_triggers': nb['f_triggers_detail'], 'n_f_triggers': len(triggers),
            'N_points': [{'class': p['class'], 'move': p['move'], 'label': lattice.label(tuple(p['z'])),
                          'I_x_eur': lattice.investment_cost(tuple(p['z']))} for p in pts_n],
            'e_opp_informational_points': [{'move': p['move'], 'label': lattice.label(tuple(p['z']))} for p in pts_info],
            'generated': rows, 'n_generated': len(rows), 'n_refused': len(refused),
            'refused_by_family': _count_fams(refused),
            'refused_budget_only': [r['label'] + ' (' + r['move'] + ')' for r in refused if r['budget_is_sole_family']],
            'unit_increment_costs_eur': _unit_costs(lattice)}


def _count_fams(rows):
    out = {}
    for r in rows:
        key = '+'.join(sorted(r['family_violations'])) or 'none'
        out[key] = out.get(key, 0) + 1
    return out


def _unit_costs(lattice):
    """I of one 0.25 MVA / 0.5 MWh granule pair (power, energy) per year: the cost of the smallest increase."""
    import p516_lattice as GL
    out = {}
    for y in lattice.years:
        p = sum(a for a, _b in lattice._coef[y]) * GL.GRANULE_P_MVA
        e = sum(b for _a, b in lattice._coef[y]) * GL.GRANULE_E_MWH
        out[str(y)] = {'zP+1_eur': p, 'zE+1_eur': e, 'activate_0.25_0.5_eur': p + e, 'activate_0.25_1_eur': p + 2 * e}
    return out


# ======================================================================================================================
#  PART G -- collision check for tracking v3's campaign specs (zero solves)
# ======================================================================================================================
def _keys_of(path):
    out = set()
    D._collect_eval_keys(D.load_json(path), out)
    return out


def part_g(data_root, spec, lattice, oracle, evals):
    v3_root = os.path.join(data_root, V3_ROOT_REL)
    v2_root = os.path.join(data_root, V2_ROOT_REL)
    tracked = D.committed_spec_files(data_root)
    v3_specs, v3_lrecs = [], []
    for dp, _dn, fn in os.walk(os.path.join(v3_root, 'launches')):
        for f in fn:
            p = os.path.join(dp, f)
            if f.startswith('campaign_spec_') and f.endswith('.json'):
                v3_specs.append(p)
            elif f.startswith('launch_') and f.endswith('.json') and os.path.basename(dp) == 'launches':
                v3_lrecs.append(p)
    # every file under the v3 root the scan scope would read (committed_spec_files' filter), once tracked
    scope = []
    for dp, _dn, fn in os.walk(v3_root):
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), data_root)
            if f.startswith(('campaign_spec_', 'search_spec_', 'phase_spec_')) or \
                    (f.startswith('launch_') and f.endswith('.json') and '/launches/' in rel):
                scope.append(os.path.join(data_root, rel))
    ck('G', 'v3: 3 campaign specs, 6 launch records/manifests', len(v3_specs) == 3 and len(v3_lrecs) == 6,
       [len(v3_specs), len(v3_lrecs)])
    ck('G', 'the scan scope under the v3 root = its phase spec + the 3 campaign specs + the 6 launch records/manifests',
       len(scope) == 10, sorted(os.path.relpath(s, data_root) for s in scope))
    k_specs = set().union(*[_keys_of(p) for p in v3_specs])
    k_lrecs = set().union(*[_keys_of(p) for p in v3_lrecs])
    k_v3_records = {e['eval_key'] for e in evals if e['phase'] == 'v3'}
    domain = D.enumerate_domain(lattice)
    k2 = {oracle.eval_key(z, d) for z in domain for d in P.DEPTHS}
    spec1 = copy.deepcopy(spec)
    for d in P.DEPTHS:
        spec1['harness']['by_depth'][d]['entry_options']['flex_price_multiplier'] = 1.0
    o1 = D.make_oracle(spec1, lattice, lock_path=None)
    k1 = {o1.eval_key(z, d) for z in domain for d in P.DEPTHS}
    # the cap enters the key: one plan, settled depth, cap 300 vs 400 (same options)
    z_inc = tuple(spec['phase']['x0'])
    H = oracle.H
    bd = spec['harness']['by_depth']['settled']
    ck_x0 = H.candidate_key(lattice.canonical_of_z(z_inc))
    key300 = D.compose_eval_key(H, bd['configuration'], oracle.entry_options(z_inc, 'settled'), 300, ck_x0)
    key400 = D.compose_eval_key(H, bd['configuration'], oracle.entry_options(z_inc, 'settled'), 400, ck_x0)
    ck('G', 'compose_eval_key reproduces the oracle key at cap 300', key300 == oracle.eval_key(z_inc, 'settled'))
    ck('G', 'the cap enters the key (cap 400 != cap 300)', key400 != key300)
    ck('G', 'the price level enters the key (1x domain keys disjoint from 2x domain keys)', not (k1 & k2),
       len(k1 & k2))
    ck('G', f'domain keys distinct: 2x {len(k2)}, 1x {len(k1)} (= 2 x {D.DOMAIN_SIZE})',
       len(k2) == len(k1) == 2 * D.DOMAIN_SIZE)
    ck('G', 'keys carried by the v3 campaign specs are a subset of the keys carried by its launch records',
       k_specs <= k_lrecs, len(k_specs - k_lrecs))
    ck('G', 'the 2x domain keys the v3 campaign specs carry == the keys of the 11 v3 evaluations',
       (k_specs & k2) == k_v3_records and len(k_v3_records) == 11, [len(k_specs & k2), len(k_v3_records)])
    ck('G', 'no 1x domain key in any v3 campaign spec, launch record or manifest', not (k1 & (k_specs | k_lrecs)))
    # simulated scans (the freeze's collision_scan) over the tracked artefacts + v3's launch-dir artefacts
    files = sorted(set(tracked) | set(scope))
    scratch_own = os.path.join(data_root, REL, '__hypothetical_new_root__')
    pilot = os.path.join(data_root, REL, 'pilot_ev_v1')
    v1 = os.path.join(data_root, REL, 'phase2x_ev_v1')
    scans = {
        '1x_phase_freeze (no cache roots; v1 declared un-launched)': D.collision_scan(
            files, k1, scratch_own, allowed_roots=[], predecessor_roots=[v1]),
        '2x_control_freeze_with_pilot_v2_v3_as_cache_roots': D.collision_scan(
            files, k2, scratch_own, allowed_roots=[pilot, v2_root, v3_root], predecessor_roots=[v1]),
        '2x_control_freeze_with_pilot_v2_as_cache_roots_v3_not_declared': D.collision_scan(
            files, k2, scratch_own, allowed_roots=[pilot, v2_root], predecessor_roots=[v1]),
        '2x_control_freeze_v3_not_declared_launch_records_only_tracked': D.collision_scan(
            sorted(set(tracked) | {s for s in scope if s not in v3_specs}), k2, scratch_own,
            allowed_roots=[pilot, v2_root], predecessor_roots=[v1]),
    }
    summ = {k: {'n_files_scanned': v['n_files_scanned'], 'n_collisions': len(v['collisions']),
                'collision_files': [os.path.relpath(c['file'], data_root) for c in v['collisions']],
                'n_excluded_cache_roots': len(v['excluded_cache_roots'])} for k, v in scans.items()}
    k_1x = summ['1x_phase_freeze (no cache roots; v1 declared un-launched)']
    ck('G', 'simulated 1x freeze scan: 0 collisions with v3 campaign specs tracked', k_1x['n_collisions'] == 0, k_1x)
    k_c = summ['2x_control_freeze_with_pilot_v2_v3_as_cache_roots']
    ck('G', 'simulated 2x control freeze declaring pilot, v2, v3 as cache roots: 0 collisions', k_c['n_collisions'] == 0,
       k_c)
    a = set(summ['2x_control_freeze_with_pilot_v2_as_cache_roots_v3_not_declared']['collision_files'])
    b = set(summ['2x_control_freeze_v3_not_declared_launch_records_only_tracked']['collision_files'])
    ck('G', 'without v3 declared, the launch records alone already collide (tracking the specs adds no new blocking '
            'condition: same refusal either way)', bool(b) and b <= a and (a - b) <= {os.path.relpath(s, data_root)
                                                                                     for s in v3_specs},
       [sorted(a), sorted(b)])
    return {'v3_campaign_specs': [os.path.relpath(p, data_root) for p in v3_specs],
            'v3_launch_records_and_manifests': [os.path.relpath(p, data_root) for p in v3_lrecs],
            'n_keys_in_v3_campaign_specs': len(k_specs), 'n_keys_in_v3_launch_records': len(k_lrecs),
            'n_2x_domain_keys': len(k2), 'n_1x_domain_keys': len(k1),
            'v3_campaign_spec_keys_in_2x_domain': len(k_specs & k2), 'v3_campaign_spec_keys_in_1x_domain': len(k_specs & k1),
            'cap_key_demo': {'x0_settled_cap300': key300[:16], 'x0_settled_cap400': key400[:16]},
            'one_x_assumption': 'the 1x configuration = the v3 configuration with flex_price_multiplier 1.0 at both '
                                'depths; any further difference (cap, options) only changes keys further',
            'simulated_scans': summ, 'decision': 'track v3 campaign specs' if all(
                c['ok'] for c in CHECKS if c['part'] == 'G') else 'track launch records and manifests only'}


# ======================================================================================================================
#  PART H -- predictions
# ======================================================================================================================
def part_h(data_root, state, res3, evals, cache, table, cands):
    pr = D.load_json(os.path.join(data_root, REL, 'phase2x_ev_v2_predictions.json'))['predictions']
    screen = [r for r in table['rows'] if r['source'] == 'screen']
    q1_fail = [(r['label'], r['dF_exit']) for r in screen if not (r['status'] == 'certified' and r['dF_exit'] < 0)]
    he_all = [k for k, s in cache.items() if s.get('status') == 'harness_error'] + \
        [e['label'] for e in evals if e['status'] == 'harness_error']
    lsf = sum(int(e['local_solve_failures'] or 0) for e in evals)
    rss = max(evals, key=lambda e: e['peak_rss_gib'])
    out = {
        'Q1': {'statement': pr['Q1']['statement'], 'n_rows': len(screen),
               'n_certified': sum(r['status'] == 'certified' for r in screen),
               'n_dF_exit_lt_0': sum(r['dF_exit'] < 0 for r in screen),
               'n_dF_exit_lt_minus_R': sum(r['dF_exit'] < -10000.0 for r in screen),
               'failing_rows': q1_fail, 'held': not q1_fail, 'role': 'recorded, non-stopping (R1)'},
        'Q2': {'statement': pr['Q2']['statement'], 'harness_error_summaries': len(he_all),
               'local_solve_failures_total': lsf, 'n_evaluations': len(evals), 'held': not he_all and lsf == 0},
        'Q3': {'statement': pr['Q3']['statement'], 'best_start': state['pool']['chosen_label'],
               'held': state['pool']['chosen_label'] != 'x0'},
        'Q4': {'statement': pr['Q4']['statement'], 'v2_termination': 'stop_for_review_not_settled (exit 5)',
               'v3_termination': res3['report']['termination']['reason'],
               'total_h': None, 'held': None},
        'Q5': {'statement': pr['Q5']['statement'], 'peak_child_rss_gib': rss['peak_rss_gib'], 'run': rss['label'],
               'run_depth': rss['depth'], 'run_phase_launch': f"{rss['phase']} {rss['launch']}",
               'held': rss['peak_rss_gib'] <= 3.29},
        'Q6': {'statement': pr['Q6']['statement'], 'efficiency': state['efficiency'], 'held': 'recorded'},
        'Q7': {'statement': pr['Q7']['statement'],
               'per_candidate': [{'label': c['label'], 'equal': c['exit_view_check']['equal'],
                                  'exit_run': c['exit_view_check']['exit_run'],
                                  'exit_record_served_as': c['exit_record_served_as'],
                                  'independent': c['exit_record_served_as'] != 'exit_view_of_settled_cache_record'}
                                 for c in cands],
               'held': all(c['exit_view_check']['equal'] for c in cands)}}
    return out


# ======================================================================================================================
#  PART I -- projections (PROJECTED)
# ======================================================================================================================
def part_i(hours, nbhd, state, lattice):
    ex = hours['measured_block_walls_h']['exit']
    st = hours['measured_block_walls_h']['settled']
    exit_rng = [ex['min'], ex['max']]
    set_rng = [st['min'], st['max']]

    def rng(n_exit, n_set, add=(0.0, 0.0)):
        return [n_exit * exit_rng[0] + n_set * set_rng[0] + add[0], n_exit * exit_rng[1] + n_set * set_rng[1] + add[1]]

    import p515_s56_neighbourhood as NB
    n_x0 = NB.s56_neighbourhood(lattice, lattice.x0())
    screen = {tuple(z) for z in state['config']['screen']}
    n_x0_eq_screen = {tuple(p['z']) for p in n_x0['points']} == screen
    cand_polls = [(c['poll_state']['polls'] + ([c['poll_state']['poll']] if c['poll_state'].get('poll') else []))[0]['n']
                  for c in state['candidates']]
    out = {
        'measured_inputs': {'exit_launch_wall_h_range': exit_rng, 'settled_launch_wall_h_range': set_rng,
                            'n_exit_launches': ex['n'], 'n_settled_launches': st['n']},
        'controls_2x': {
            'start_rule': 'the phase pool rule (argmin settled F) gives the same start: ' + state['pool']['chosen_label'],
            'N_start': nbhd['size'], 'N_start_cached_settled': nbhd['size'],
            'a_settled_direct_with_v2_v3_cache_h': [0.0, 0.0],
            'a_note': 'every neighbour of the start has a settled record (candidate 1 poll): 0 evaluations; the control '
                      'ends at its first (complete) poll with 0 moves -- identical outcome for both controls',
            'b_existing_plain_search_exit_polls_h_per_first_control': rng(3, 0),
            'b_note': 'the existing plain-search path (--freeze, exit-depth polls, e_floor 15,000 EUR) has no exit '
                      'record of the 10 neighbours (2 exit blocks) nor of the exit-less pool members x0, unit and '
                      'staged (1 exit block); settled confirmations all cached; the second control then reuses the '
                      'first control\'s exit records (0 h)'},
        'phase_1x': {
            'note': 'no record carries over (price level in the key, PART G); x0 exit + settled, the unit and the '
                    'staged plan settled are re-run at 1x by the phase itself',
            'N_x0_equals_screen': n_x0_eq_screen, 'N_x0_size': n_x0['size'],
            'components_h': {
                'x0_exit': rng(1, 0), 'screen_18': rng(4, 0), 'ladder_4': rng(1, 0),
                'confirmations_x0_plus_2_candidates': rng(0, 1), 'pool_unit_staged_settled': rng(0, 1)},
            'candidate_poll_blocks_2x_shape': {'N_cand': cand_polls,
                                               'blocks': sum(math.ceil(n / 5) for n in cand_polls)},
        },
    }
    base = [sum(v[0] for v in out['phase_1x']['components_h'].values()),
            sum(v[1] for v in out['phase_1x']['components_h'].values())]
    nb_blocks = out['phase_1x']['candidate_poll_blocks_2x_shape']['blocks']
    out['phase_1x']['scenarios_h'] = {
        'settled_direct_2x_shape_no_move': [base[0] + nb_blocks * set_rng[0], base[1] + nb_blocks * set_rng[1]],
        'exit_then_confirm_2x_shape_no_move_0_confirmations': [base[0] + nb_blocks * exit_rng[0],
                                                               base[1] + nb_blocks * exit_rng[1]],
        'exit_then_confirm_per_confirmation': set_rng,
        'no_confirm_candidate_start_x0': [base[0] - set_rng[0], base[1] - set_rng[1]],
        'basis': 'base = x0 exit + screen (4 blocks) + ladder (1) + one settled confirmation launch + one pool '
                 'launch; candidate polls at the 2x shape (|N| 10 and 22 -> 7 blocks); no search move; '
                 'no_confirm_candidate_start_x0: no candidate, no confirmation launch, N(x0) = the screen (cached)'}
    return out


# ======================================================================================================================
#  manifest mode
# ======================================================================================================================
def manifest(data_root, eval_dir, target):
    import p515_s56_ev_harness_tests as T   # installs its own permitted=() guard (verified at 0 below)
    if os.path.exists(target):
        raise D.Refused(f'{target} exists (write-once)')
    rep = D.load_json(os.path.join(eval_dir, 'report_eval.json'))
    track_specs = rep['results']['G']['decision'] == 'track v3 campaign specs'
    roots = [os.path.join(data_root, V2_ROOT_REL), os.path.join(data_root, V3_ROOT_REL), eval_dir]
    extra = [os.path.join(data_root, REL, f'{os.path.basename(eval_dir)}_{s}.log') for s in ('stdout', 'stderr')]
    files = {}
    for root in roots:
        for dp, _dn, fn in os.walk(root):
            for f in sorted(fn):
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, data_root)
                files[rel] = p
    for p in extra:
        files[os.path.relpath(p, data_root)] = p
    out = {}
    for rel, p in sorted(files.items()):
        size = os.path.getsize(p)
        c = T.committed_by_rule(rel, size)
        why = 'selection rule'
        if c and os.path.basename(rel).startswith('campaign_spec_') and V3_ROOT_REL in rel and not track_specs:
            c, why = False, 'PART G: v3 campaign specs not tracked'
        out[rel] = {'sha256': D.sha256_file(p), 'bytes': size, 'committed': bool(c), 'committed_reason': why}
    obj = {'schema': SCHEMA + '_manifest', 'utc': datetime.now(timezone.utc).isoformat(),
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True,
                                      text=True).stdout.strip(),
           'selection_rule': {'source': 'p515_s56_ev_harness_tests.committed_by_rule',
                              'excluded_parts': list(T.COMMIT_EXCLUDED_PARTS),
                              'excluded_suffixes': list(T.COMMIT_EXCLUDED_SUFFIXES), 'max_bytes': T.COMMIT_MAX_BYTES,
                              'size_exempt': list(T.COMMIT_SIZE_EXEMPT),
                              'v3_campaign_specs_tracked': track_specs},
           'roots': [os.path.relpath(r, data_root) for r in roots], 'extra_files': [os.path.relpath(p, data_root)
                                                                                     for p in extra],
           'n_files': len(out), 'n_committed': sum(1 for v in out.values() if v['committed']),
           'bytes_total': sum(v['bytes'] for v in out.values()),
           'bytes_committed': sum(v['bytes'] for v in out.values() if v['committed']),
           'files': out}
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'x') as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
    log(f'manifest: {obj["n_files"]} files ({obj["n_committed"]} committed, {obj["bytes_committed"] / 1e6:.1f} MB) -> '
        f'{os.path.relpath(target, data_root)} ({D.sha256_file(target)})')
    g = [('own', GUARD), ('p515_s56_ev_harness_tests', T.GUARD)]
    bad = [(n, gg.verify(0)) for n, gg in g if gg.verify(0)]
    if bad or any(PICKLE_COUNTS.values()):
        log(f'GUARD FAILURE {bad} pickle {PICKLE_COUNTS}')
        return 3
    log(f'guards verify(0): [] (own and test module); pickle {PICKLE_COUNTS}')
    return 0


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--manifest', action='store_true')
    ap.add_argument('--eval-dir')
    ap.add_argument('--target')
    a = ap.parse_args()
    data_root = CODE_ROOT
    if a.manifest:
        if not (a.eval_dir and a.target):
            ap.error('--manifest needs --eval-dir and --target')
        return manifest(data_root, os.path.abspath(a.eval_dir), os.path.abspath(a.target))
    if not a.out:
        ap.error('--out')
    out_dir = os.path.abspath(a.out)
    for r in (V2_ROOT_REL, V3_ROOT_REL):
        if out_dir.startswith(os.path.join(data_root, r)):
            raise D.Refused(f'--out inside a phase root: {out_dir}')
    if os.path.exists(out_dir):
        raise D.Refused(f'{out_dir} exists (write-once)')
    log(f'{SCRIPT}: zero solves; guard armed; pickle blocked')
    spec, lattice, env, oracle, cache, records, state, res3 = part_a(data_root)
    R, e_ev = spec['policy']['R'], spec['policy']['e_ev']
    evals = part_b(data_root, lattice, oracle, records)
    hours = part_c(data_root, evals)
    table = part_d(lattice, records, state)
    e = part_e(lattice, records, state, evals, R, e_ev)
    polled = [r['label'] for r in e['search']['polls'][-1]['rows']]
    nbhd = part_f(lattice, e['search']['final_incumbent'], polled)
    g = part_g(data_root, spec, lattice, oracle, evals)
    h = part_h(data_root, state, res3, evals, cache, table, e['candidates'])
    h['Q4']['total_h'] = hours['total_h']
    h['Q4']['held'] = h['Q4']['v3_termination'] == 'phase_complete' and hours['total_h'] <= 120.0
    proj = part_i(hours, nbhd, state, lattice)
    settled = [x for x in evals if x['depth'] == 'settled']
    mem = {'peak_rss_gib_by_depth': {d: _stats([x['peak_rss_gib'] for x in evals if x['depth'] == d])
                                     for d in P.DEPTHS},
           'leak_priced_loss_eur_settled': _stats([x['leak_priced_loss_eur'] for x in settled]),
           'leak_priced_full_eur_settled': _stats([x['leak_priced_full_eur'] for x in settled]),
           'esso_slack_eur_settled': _stats([x['esso_slack_eur'] for x in settled]),
           'esso_slack_max_abs_net_settled': _stats([x['esso_slack_max_abs_net'] for x in settled]),
           'esso_events_by_plan': [{'label': x['label'], 'depth': x['depth'], 'phase_launch': f"{x['phase']} {x['launch']}",
                                    'n_events': x['n_esso_events_file'], 'events': x['esso_events']}
                                   for x in evals if x['n_esso_events_file']],
           'plans_with_a_0p5_mwh_cohort': sum(any(c['E_mwh'] == 0.5 for c in x['cohorts']) for x in evals),
           'plans_with_a_0p5_mwh_cohort_and_an_event': sum(any(c['E_mwh'] == 0.5 for c in x['cohorts'])
                                                           and x['n_esso_events_file'] > 0 for x in evals)}
    ev_all = [ev for x in evals for ev in x['esso_events']]
    ck('B', 'every ESSO recovery event is on a node holding a 0.5 MWh cohort',
       all(ev['node_has_0p5_mwh_cohort'] for ev in ev_all), len(ev_all))
    ck('B', 'every ESSO recovery event recovered at tier 1', all(ev['recovery_succeeded'] is True and ev['tier'] == 'tier1'
                                                                 and not ev['tier2_attempted'] for ev in ev_all))
    payload = {'schema': SCHEMA, 'utc': datetime.now(timezone.utc).isoformat(), 'script': SCRIPT,
               'script_sha256': D.sha256_file(os.path.abspath(__file__)),
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True,
                                          text=True).stdout.strip(),
               'objective_convention': 'F = I(x) + gross_operational_cost - TV credit, EUR of total discounted '
                                       'planning cost (oracle mapping v2; gross settlement-excluded)',
               'results': {'hours': hours, 'table': table, 'E': e, 'neighbourhood_final_incumbent': nbhd, 'G': g,
                           'predictions': h, 'projections': proj, 'leak_slack_memory': mem, 'evaluations': evals},
               'checks': CHECKS}
    os.makedirs(out_dir)
    gv = GUARD.verify(0)
    payload['guard'] = {'verify_0': gv, 'counts': dict(GUARD.counts), 'pickle': dict(PICKLE_COUNTS)}
    payload['all_checks_ok'] = all(c['ok'] for c in CHECKS) and not gv and not any(PICKLE_COUNTS.values())
    path = os.path.join(out_dir, 'report_eval.json')
    with open(path, 'x') as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    log(f'wrote {os.path.relpath(path, data_root)} ({D.sha256_file(path)}); checks {sum(c["ok"] for c in CHECKS)}/'
        f'{len(CHECKS)}; guard verify(0) {gv}; pickle {PICKLE_COUNTS}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except D.Refused as error:
        print(f'REFUSED: {error}', flush=True)
        sys.exit(2)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
