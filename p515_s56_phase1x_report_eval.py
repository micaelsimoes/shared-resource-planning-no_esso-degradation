"""P5.15 S56 EV 1x PHASE -- END-OF-PHASE EVALUATION FOR REPORT (ii) (ZERO SOLVES; read only on every phase root).

Authority: TASKS.md step 5e' ("1x end-of-phase evaluation + report (ii) (Worker, zero solves)"); the Planner's task of
2026-11-26. Every number in P5_15_S56_REPORT_ii.md that is not a field copied from a phase record is defined here
(CLAUDE.md: preserve the formula, not only the inputs). Pattern and helpers: p515_s56_phase2x_report_eval.py (imported).

INPUTS (read only): the 1x root phase1x_ev_v1 (spec feb487b7, 7 launches, phase_complete, no cache source); the 1x
predictions file (de8313a4); the 2x reference phase2x_ev_v3/phase_results.json (sha256 named by the predictions file)
and the committed 2x evaluation phase2x_ev_end_report_eval/report_eval.json (4427e142); the memory probe record
memory_probe_v1/srp1grid/memory_probe_srp1grid.json (SP s/cycle on the SRP1 grid).

OBJECTIVE CONVENTION (every EUR figure): F = I(x) + gross_operational_cost - TV credit, EUR of total discounted planning
cost (oracle mapping v2; gross settlement-excluded). dF_exit = F_exit(x) - F_exit(x0); dF_set = F_set(x) - F_set(ref).

PART A  the 1x phase as the driver sees it: spec sha, NO CODE DRIFT against the spec's code_sha256 (the six driver /
        policy / neighbourhood / lattice / harness / gate_result_io files), `phase_cache` (empty: no cache source),
        `phase_replay` of the 7-launch chain (every launch record, manifest, hash chain, action, state and record
        re-derived from its eval dir), one more `phase_step` -> phase_complete, state == phase_results.json state.
PART B  per-evaluation facts (26 evaluations): mapped summary, cycles_run, child wall, peak RSS
        (child_python_process_ru_maxrss / 2^30), local_solve_failures, ESSO recovery events (file rows == record count),
        leak (settled: leak.priced_loss_eur / priced_full_eur / L_weighted_mwh), ESSO slack V_parts.G.esso_slack, gates.
PART C  hours: per launch (wall_s, UTC stamps, child walls), per stage (x0 exit, screen, ladder, pool settled, search;
        sum of launch walls by purpose), sum == cumulative_h; against the restated 1x projection (spec
        phase.ceiling.projection_restated_1x) and the spec's own projected_hours (report only).
PART D  the marginal-value table (22 rows) with, per row: e recomputed = max(e_EV, bar_x + bar_x0) (== recorded e),
        class recomputed with P.classify_exit(x, x0, R, e_floor = e_EV) (== recorded class), and the MARGINS
          to_worse   = dF_exit - (R + e)     (>= 0: worse; the amount by which the row clears the worse boundary)
          to_confirm = dF_exit - (-R + e)    (<= 0: confirm-candidate)
        for unresolved rows both distances are reported: (R + e) - dF_exit and dF_exit - (-R + e).
        The two closest worse rows = the two smallest to_worse over rows classed worse (screen and ladder).
PART E  exit-to-settled OFFSETS at 1x for the plans with an exit-depth run AND a settled run (x0, the reference unit):
          exit run (read_cycle, F, bar) vs the settled summary's exit_view (P.exit_view_check, float.hex identity), and
          vs an independent re-read D.map_record(D.read_eval_files(settled dir), 'exit', spec mapping) (the Q7 reading);
          offset(x) = F_set(x) - F_exit(x); pair offset = offset(unit) - offset(x0)
                    = [F_set(unit) - F_set(x0)] - [F_exit(unit) - F_exit(x0)];
          e used for the unit row = max(e_EV, bar_unit + bar_x0).
        Information beside it (NOT an exit run): the staged plan's offset against its settled run's exit view.
        Per-cycle F of the settled runs (D.f_series) at fixed cycles, the pair difference unit - x0 along the
        trajectory, and the prefix identity F_settled_run(k) == F_exit_run(k) for k <= the exit run's cycles_run.
        The 2x offsets for comparison, from the committed 2x evaluation (x0: exit view of the pilot x0_i1).
PART F  pool, search (incumbent history, terminal poll), N(x0) == the 18 screen plans, the INDISTINGUISHABLE set (plan
        v4.1 section 4: settled |F_set(x) - F_set(x0)| < R; exit-only |F_exit(x) - F_exit(x0)| < R).
PART G  zero-solve check for tracking the 1x campaign specs: keys carried by the 1x campaign specs subset of those of the
        1x launch records; their 1x domain keys == the 26 evaluation keys; simulated freeze scans of a later 1x
        confirmation-only spec with / without phase1x_ev_v1 declared as a cache root.
PART H  recorded predictions Q1-Q9 against outcomes, each by its frozen `evaluated_as`.
PART I  2x vs 1x: per screen label dF_exit at both levels; x0 F at both levels (exit and settled); the selected plans.
PART J  projections (PROJECTED): the settled confirmation launch of 5 at 1x (measured settled launch walls); t_SP
        extrapolated = memory-probe SP s/cycle (cycle 1, cycle 2) x cycle counts measured in the EV phases (exit read
        cycles at 1x and 2x; K_EV = 300); the next ordered items with plan v4.1 section 8's ranges beside.

MANIFEST MODE (--manifest): hash inventory of the 1x root (every file), this script's output dir and its logs; per file
sha256, bytes, `committed` by `p515_s56_ev_harness_tests.committed_by_rule` (PART G decides the campaign specs).

GUARDS: SolveProfileGuard(permitted=()) installed before any repository import, verify(0) (own, the imported 2x
evaluation module's, and in manifest mode the test module's); pickle load / loads blocked (0). Writes only --out (a NEW
directory outside every phase root) or the manifest file (write-once).
Launch (attached, both streams captured):
  python -u p515_s56_phase1x_report_eval.py --out data/SRP1/Results/P515S56/phase1x_ev_end_report_eval \\
      > data/SRP1/Results/P515S56/phase1x_ev_end_report_eval_stdout.log \\
      2> data/SRP1/Results/P515S56/phase1x_ev_end_report_eval_stderr.log
  python -u p515_s56_phase1x_report_eval.py --manifest --eval-dir <the --out above> \\
      --target data/SRP1/Results/P515S56/phase1x_ev_end_evidence_manifest_sha256.json
Exit: 0 written and every check holds; 3 written with a failed check; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 1x phase report evaluation (never solves)').install()
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

import p515_s56_phase2x_report_eval as R2  # noqa: E402  (installs its own permitted=() guard; verified at 0 below)

pickle.load, pickle.loads = _blocked_load, _blocked_loads   # this module's counters (R2 re-patched them on import)

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
SCHEMA = 'p515_s56_phase1x_report_eval_v1'
REL = os.path.join('data', 'SRP1', 'Results', 'P515S56')
ROOT_REL = os.path.join(REL, 'phase1x_ev_v1')
SPEC_SHA = 'feb487b788c9a769113a1e24e32c4db8e8c296375553168aa12bcee118d8c0ac'
PRED_REL = os.path.join(REL, 'phase1x_ev_v1_predictions.json')
PRED_SHA = 'de8313a44aae2806db3d06d24cc4b15be8376e195a08207616e7dc958bc7ce65'
R2_EVAL_REL = os.path.join(REL, 'phase2x_ev_end_report_eval', 'report_eval.json')
R2_EVAL_SHA = '4427e1421e0cf88c8026a37dfa87b1d2afc817f724c8d7b562efb67ba0dc4e09'   # P5_15_S56_PHASE2X_REPORT.md
R2_MANIFEST_REL = os.path.join(REL, 'phase2x_ev_end_evidence_manifest_sha256.json')
PROBE_REL = os.path.join(REL, 'memory_probe_v1', 'srp1grid', 'memory_probe_srp1grid.json')
PROTECTED_ROOTS = ('phase1x_ev_v1', 'phase2x_ev_v1', 'phase2x_ev_v2', 'phase2x_ev_v3', 'pilot_ev_v1')
UNIT_LABEL = 'n7_y2025_p0.25_e1'
STAGED_NAME = 'pilot_staged'
TRAJ_CYCLES = (80, 85, 90, 100, 125, 150, 175, 200, 250, 300)
CHECKS = []
CJ = P.canonical_json
_stats, _h, _cohorts, _launches = R2._stats, R2._h, R2._cohorts, R2._launches


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(part, name, ok, detail=None):
    CHECKS.append({'part': part, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{part}] {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


# ======================================================================================================================
#  PART A -- the 1x phase as the driver sees it
# ======================================================================================================================
def part_a(data_root):
    root = os.path.join(data_root, ROOT_REL)
    _sp, sha, spec = D.load_phase_spec(root)
    ck('A', 'spec sha256 = feb487b7...', sha == SPEC_SHA, sha)
    drift = D.code_drift(spec)
    ck('A', 'no code drift against the spec code_sha256 (driver, policy, neighbourhood, lattice, harness, '
            'gate_result_io): the checkout is byte-identical to the frozen code', not drift, drift)
    ck('A', 'no cache source declared (D2: no record carries over)', not spec.get('cache_sources'),
       spec.get('cache_sources'))
    D.verify_predecessors_at_launch(spec, data_root)
    pre = spec.get('predecessors') or {}
    ck('A', 'predecessors verified (none: no continuation, no un-launched predecessor)',
       not pre.get('continuation') and not pre.get('unlaunched'), pre)
    lattice, _inst = D.build_lattice(data_root)
    env = D.LatticeEnv(lattice)
    oracle = D.make_oracle(spec, lattice, lock_path=None)
    cache = D.phase_cache(spec, oracle, data_root, lattice)
    ck('A', 'phase cache empty', cache == {}, len(cache))
    state, records, chain, _states = D.phase_replay(root, spec, sha, lattice, env, oracle, cache)
    ck('A', 'chain replays: 7 launches, every record / manifest / action / state re-derived', len(chain) == 7,
       len(chain))
    state, action = P.phase_step(state, records, env)
    res = D.load_json(os.path.join(root, D.PHASE_RESULTS_FILE))
    ck('A', 'one more phase_step terminates phase_complete', action.get('kind') == 'terminated'
       and (action.get('termination') or {}).get('reason') == 'phase_complete', action)
    ck('A', 'the replayed terminal state equals phase_results.json state (canonical JSON)', CJ(state) == CJ(res['state']))
    man = D.load_json(os.path.join(root, D.PHASE_RESULTS_MANIFEST))
    res_sha = D.sha256_file(os.path.join(root, D.PHASE_RESULTS_FILE))
    ck('A', 'phase_results manifest holds the results hash', man.get(D.PHASE_RESULTS_FILE) == res_sha, man)
    ck('A', 'phase_results.json names this spec', res.get('phase_spec_sha256') == SPEC_SHA)
    ck('A', '26 records (x0 exit, 18 screen, 4 ladder at exit; 3 settled)', len(records) == 26
       and sum(d == P.DEPTH_SETTLED for (_z, d) in records) == 3, len(records))
    return spec, lattice, env, oracle, records, state, res, res_sha


# ======================================================================================================================
#  PART B -- per-evaluation facts
# ======================================================================================================================
def part_b(data_root, lattice, records):
    root = os.path.join(data_root, ROOT_REL)
    out = []
    for lr in _launches(root):
        for r in lr['records']:
            z, depth = tuple(r['z']), r['depth']
            ed = os.path.join(root, r['eval_dir'])
            summ = records[(z, depth)]
            rec = D.load_json(os.path.join(ed, 'evaluation_record.json'))
            ev_path = os.path.join(ed, R2.ESSO_EVENTS_FILE)
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
                                'primary_result': e.get('primary_result'), 'node_cohorts': node_coh,
                                'node_has_0p5_mwh_cohort': any(c['E_mwh'] == 0.5 for c in node_coh)})
            nfs = rec.get('network_failures_summary') or {}
            rss = (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss')
            leak = summ.get('leak') or {}
            g = (summ.get('V_parts') or {}).get('G') or {}
            out.append({
                'launch': lr['launch_index'], 'purpose': lr['action']['purpose'], 'stage': lr['action']['stage'],
                'depth': depth, 'z': list(z), 'label': lattice.label(z), 'candidate_key': lattice.key(z),
                'eval_key': r['eval_key'], 'eval_dir': os.path.relpath(ed, data_root),
                'I_x_eur': lattice.investment_cost(z), 'cohorts': coh,
                'status': summ.get('status'), 'guard': (summ.get('guards') or {}).get('solve_profile_identity_holds'),
                'guards': summ.get('guards'),
                'reconciliation_via': (summ.get('solve_profile_reconciliation') or {}).get('via'),
                'F': summ.get('F'), 'bar': summ.get('bar'), 'read_cycle': summ.get('read_cycle'),
                'cycles_run': summ.get('cycles_run'), 'V': summ.get('V'), 'gates_pass': summ.get('gates_pass'),
                'gates': summ.get('gates'), 'settled_checks': summ.get('settled_checks'),
                'exit_view': summ.get('exit_view'),
                'leak_priced_loss_eur': leak.get('priced_loss_eur'), 'leak_priced_full_eur': leak.get('priced_full_eur'),
                'leak_L_weighted_mwh': leak.get('L_weighted_mwh'), 'leak_within_ceiling': leak.get('within_ceiling'),
                'esso_slack_eur': g.get('esso_slack'), 'esso_slack_max_abs_net': summ.get('esso_slack_max_abs_net'),
                'closure_max_abs_net_mwh': summ.get('closure_max_abs_net_mwh'),
                'child_wall_s': summ.get('wall_s'),
                'peak_rss_gib': (float(rss) / R2.GIB) if rss is not None else None,
                'local_solve_failures': rec.get('local_solve_failures'),
                'n_esso_recovery_events_record': nfs.get('n_esso_recovery_events'),
                'network_failure_classes': nfs.get('classes'),
                'esso_events': ev_rows, 'n_esso_events_file': len(events)})
    ck('B', '26 evaluations launched by the phase (every one served by the replay)', len(out) == 26, len(out))
    ck('B', 'ESSO events file rows == record n_esso_recovery_events on every evaluation',
       all(e['n_esso_events_file'] == e['n_esso_recovery_events_record'] for e in out),
       [(e['label'], e['n_esso_events_file'], e['n_esso_recovery_events_record']) for e in out
        if e['n_esso_events_file'] != e['n_esso_recovery_events_record']])
    ck('B', 'every exit run stopped at its read cycle (cycles_run == read_cycle)',
       all(e['cycles_run'] == e['read_cycle'] for e in out if e['depth'] == P.DEPTH_EXIT))
    return out


# ======================================================================================================================
#  PART C -- hours
# ======================================================================================================================
STAGE_OF_PURPOSE = {'x0_exit': 'x0_exit', 'screen': 'screen', 'ladder': 'ladder',
                    'candidate_confirmation': 'candidate_confirmations', 'candidate_poll': 'candidate_polls',
                    'pool_settled': 'pool_settled', 'search': 'search'}
STAGES = ('x0_exit', 'screen', 'ladder', 'candidate_confirmations', 'candidate_polls', 'pool_settled', 'search')


def part_c(data_root, spec, res, evals):
    root = os.path.join(data_root, ROOT_REL)
    lrs = {lr['launch_index']: lr for lr in _launches(root)}
    rows = []
    for pl in res['hours']['per_launch']:
        lr = lrs[pl['launch']]
        ck('C', f'launch {pl["launch"]}: wall in phase_results == launch record wall',
           pl['wall_s'] == lr['wall_s'] and pl['purpose'] == lr['action']['purpose'])
        mine = [e for e in evals if e['launch'] == pl['launch']]
        rows.append({'launch': pl['launch'], 'purpose': pl['purpose'], 'stage': STAGE_OF_PURPOSE[pl['purpose']],
                     'depth': pl['depth'], 'n_evaluations': pl['n_evaluations'], 'wall_s': pl['wall_s'],
                     'wall_h': _h(pl['wall_s']), 'started_utc': lr['started_utc'], 'finished_utc': lr['finished_utc'],
                     'labels': [e['label'] for e in mine],
                     'child_wall_h': _stats([_h(e['child_wall_s']) for e in mine]),
                     'cycles_run': _stats([e['cycles_run'] for e in mine])})
    ck('C', 'interrupted launches 0 (interrupted hours 0)', res['hours']['interrupted'] == []
       and res['hours']['interrupted_launches_h'] == 0)
    by_stage = {s: {'h': 0.0, 'launches': 0, 'evaluations': 0} for s in STAGES}
    for r in rows:
        by_stage[r['stage']]['h'] += r['wall_h']
        by_stage[r['stage']]['launches'] += 1
        by_stage[r['stage']]['evaluations'] += r['n_evaluations']
    total = sum(r['wall_h'] for r in rows)
    ck('C', 'sum of launch walls == cumulative_h', abs(total - res['hours']['cumulative_h']) < 1e-9,
       [total, res['hours']['cumulative_h']])
    ce = spec['phase']['ceiling']
    restated = ce['projection_restated_1x']
    ph = spec['projected_hours']
    case_ran = 'exit_then_confirm|cand_small|search_low|q0/1'
    per_block = {'exit': [r['wall_h'] for r in rows if r['depth'] == 'exit'],
                 'exit_block_of_5': [r['wall_h'] for r in rows if r['depth'] == 'exit' and r['n_evaluations'] == 5],
                 'settled': [r['wall_h'] for r in rows if r['depth'] == 'settled']}
    return {'per_launch': rows, 'by_stage': by_stage, 'total_h': total, 'ceiling_h': ce['ceiling_h'],
            'share_of_ceiling': total / ce['ceiling_h'],
            'projection_restated_1x': restated,
            'spec_projected_hours_report_only': {'case_nearest_to_what_ran': case_ran,
                                                 'case': ph['cases'].get(case_ran),
                                                 'note': 'assumes two candidates polled (8 candidate poll blocks); no '
                                                         'candidate existed at 1x, so no spec case matches exactly',
                                                 'per_block_h': ph['per_block_h']},
            'measured_block_walls_h': {k: _stats(v) for k, v in per_block.items()},
            'child_walls_h': {d: _stats([_h(e['child_wall_s']) for e in evals if e['depth'] == d]) for d in P.DEPTHS},
            'exit_read_cycles': _stats([e['read_cycle'] for e in evals if e['depth'] == P.DEPTH_EXIT])}


# ======================================================================================================================
#  PART D -- the marginal-value table with margins
# ======================================================================================================================
def part_d(lattice, records, state, R, e_ev):
    x0 = tuple(state['config']['x0'])
    x0e = records[(x0, P.DEPTH_EXIT)]
    x0s = records[(x0, P.DEPTH_SETTLED)]
    tx0 = state['table']['x0']
    ck('D', 'table x0 = x0 exit run (F, bar bitwise)', P._bitwise(tx0['F_exit']) == P._bitwise(x0e['F'])
       and P._bitwise(tx0['bar']) == P._bitwise(x0e['bar']))
    rows, bad_e, bad_c = [], [], []
    for r in state['table']['rows']:
        z = tuple(r['z'])
        xe = records[(z, P.DEPTH_EXIT)]
        c = P.classify_exit(xe, x0e, R, e_ev)
        e_re = max(e_ev, float(xe['bar']) + float(x0e['bar']))
        if e_re != r['e'] or c['e'] != r['e']:
            bad_e.append((r['label'], r['e'], e_re, c['e']))
        if c['class'] != r['class'] or c['dF'] != r['dF_exit']:
            bad_c.append((r['label'], r['class'], c['class'], r['dF_exit'], c['dF']))
        dF, e = r['dF_exit'], r['e']
        row = {k: r.get(k) for k in ('label', 'source', 'node', 'year', 'zp', 'ze', 'duration', 'status', 'F_exit',
                                     'bar', 'dF_exit', 'e', 'class', 'served_as')}
        row.update({'I_x_eur': lattice.investment_cost(z), 'read_cycle': xe.get('read_cycle'),
                    'e_is_e_ev': e == e_ev, 'bar_sum': float(xe['bar']) + float(x0e['bar']),
                    'worse_boundary_R_plus_e': R + e, 'confirm_boundary_minus_R_plus_e': -R + e,
                    'margin_to_worse': dF - (R + e), 'margin_to_confirm': dF - (-R + e)})
        if r['class'] == 'unresolved':
            row['unresolved_distance_below_worse_boundary'] = (R + e) - dF
            row['unresolved_distance_above_confirm_boundary'] = dF - (-R + e)
        s = records.get((z, P.DEPTH_SETTLED))
        if s is not None:
            cs = P.classify_settled(s, x0s)
            row['settled'] = {'status': s.get('status'), 'F_set': s.get('F'), 'dF_set_vs_x0': cs['dF'],
                              'V_pair_vs_x0': cs['V_pair'], 'A_sum_vs_x0': cs['A_sum'], 'class_vs_x0': cs['class'],
                              'failed': cs['failed']}
        else:
            row['settled'] = None
        rows.append(row)
    ck('D', '22 rows (18 screen + 4 ladder)', len(rows) == 22 and sum(r['source'] == 'screen' for r in rows) == 18)
    ck('D', 'e recomputed = max(e_EV, bar_x + bar_x0) equals the recorded e on every row', not bad_e, bad_e)
    ck('D', 'P.classify_exit recomputed equals the recorded class and dF_exit on every row', not bad_c, bad_c)
    worse = sorted((r for r in rows if r['class'] == 'worse'), key=lambda r: r['margin_to_worse'])
    ck('D', 'every worse row has margin_to_worse >= 0; every unresolved row lies strictly between the boundaries',
       all(r['margin_to_worse'] >= 0 for r in worse) and all(
           r['margin_to_worse'] < 0 < r['margin_to_confirm'] for r in rows if r['class'] == 'unresolved'))
    return {'x0': {'F_exit': tx0['F_exit'], 'bar_exit': tx0['bar'], 'read_cycle': x0e.get('read_cycle'),
                   'F_set': x0s.get('F'), 'status_set': x0s.get('status')},
            'R': R, 'e_ev': e_ev,
            'margin_definitions': {'margin_to_worse': 'dF_exit - (R + e); >= 0 is worse',
                                   'margin_to_confirm': 'dF_exit - (-R + e); <= 0 is confirm-candidate',
                                   'e': 'max(e_EV, bar_x + bar_x0)'},
            'rows': rows, 'class_counts': {k: sum(r['class'] == k for r in rows) for k in P.EXIT_CLASSES},
            'worse_by_margin': [{'label': r['label'], 'source': r['source'], 'dF_exit': r['dF_exit'], 'e': r['e'],
                                 'margin_to_worse': r['margin_to_worse']} for r in worse],
            'two_closest_worse': [r['label'] for r in worse[:2]],
            'unresolved': [{'label': r['label'], 'dF_exit': r['dF_exit'], 'e': r['e'],
                            'below_worse_boundary': r['unresolved_distance_below_worse_boundary'],
                            'above_confirm_boundary': r['unresolved_distance_above_confirm_boundary']}
                           for r in rows if r['class'] == 'unresolved']}


# ======================================================================================================================
#  PART E -- exit-to-settled offsets
# ======================================================================================================================
def _settled_dir(data_root, evals, label):
    ds = [e['eval_dir'] for e in evals if e['label'] == label and e['depth'] == P.DEPTH_SETTLED]
    return os.path.join(data_root, ds[0])


def _exit_dir(data_root, evals, label):
    ds = [e['eval_dir'] for e in evals if e['label'] == label and e['depth'] == P.DEPTH_EXIT]
    return os.path.join(data_root, ds[0]) if ds else None


def _traj(data_root, evals, label):
    files = D.read_eval_files(_settled_dir(data_root, evals, label))
    _inv, F, bad = D.f_series(files['record'], files['rows'])
    return F, bad


def part_e(data_root, spec, lattice, records, evals, state, R, e_ev):
    mapping = spec['mapping']
    x0 = tuple(state['config']['x0'])
    labels = {'x0': x0}
    for m in state['pool']['members']:
        labels[m['label'] if m['name'] != STAGED_NAME else STAGED_NAME] = tuple(m['z'])
    unit = labels[UNIT_LABEL]
    staged = labels[STAGED_NAME]
    per_plan = {}
    for name, z in (('x0', x0), (UNIT_LABEL, unit), (STAGED_NAME, staged)):
        lab = lattice.label(z)
        s = records[(z, P.DEPTH_SETTLED)]
        xe = records.get((z, P.DEPTH_EXIT))
        reread = D.map_record(D.read_eval_files(_settled_dir(data_root, evals, lab)), P.DEPTH_EXIT, mapping)
        ev = s.get('exit_view') or {}
        same_reread = all(P._bitwise(reread.get(f)) == P._bitwise(ev.get(f)) for f in P.EXIT_VIEW_FIELDS)
        ck('E', f'{name}: settled summary exit_view == independent re-read map_record(files, exit) (bitwise)',
           same_reread, [reread.get(f) for f in P.EXIT_VIEW_FIELDS])
        row = {'label': lab, 'F_set': s['F'], 'settled_status': s['status'], 'exit_view': ev,
               'exit_view_reread': {f: reread.get(f) for f in P.EXIT_VIEW_FIELDS + ('status', 'read_at')},
               'exit_view_reread_equal': same_reread}
        if xe is not None:
            chk = P.exit_view_check(xe, s)
            ck('E', f'{name}: exit run (read_cycle, F, bar) == settled exit_view, bitwise (the Q7 reading)',
               chk['equal'], chk)
            row.update({'has_exit_run': True, 'F_exit': xe['F'], 'bar_exit': xe['bar'],
                        'read_cycle_exit': xe['read_cycle'], 'exit_view_check': chk,
                        'offset_F_set_minus_F_exit': s['F'] - xe['F']})
        else:
            row.update({'has_exit_run': False, 'F_exit_view': ev.get('F'), 'bar_exit_view': ev.get('bar'),
                        'read_cycle_exit_view': ev.get('read_cycle'),
                        'offset_F_set_minus_F_exit_view': s['F'] - ev['F'],
                        'note': 'information only: no exit run; the offset is against the exit view of the settled run'})
        per_plan[name] = row
    px, pu, ps = per_plan['x0'], per_plan[UNIT_LABEL], per_plan[STAGED_NAME]
    pair = {'dF_exit_unit_minus_x0': pu['F_exit'] - px['F_exit'], 'dF_set_unit_minus_x0': pu['F_set'] - px['F_set'],
            'pair_offset': (pu['F_set'] - px['F_set']) - (pu['F_exit'] - px['F_exit']),
            'pair_offset_as_offset_difference': pu['offset_F_set_minus_F_exit'] - px['offset_F_set_minus_F_exit'],
            'e_EV': e_ev, 'bar_unit_exit': pu['bar_exit'], 'bar_x0_exit': px['bar_exit'],
            'e_used_for_unit_row': max(e_ev, pu['bar_exit'] + px['bar_exit'])}
    pair['pair_offset_exceeds_e_EV'] = abs(pair['pair_offset']) > e_ev
    pair['pair_offset_within_unit_row_e'] = abs(pair['pair_offset']) <= pair['e_used_for_unit_row']
    staged_pair = {'dF_exit_view_staged_minus_x0': ps['F_exit_view'] - px['F_exit'],
                   'dF_set_staged_minus_x0': ps['F_set'] - px['F_set']}
    staged_pair['pair_offset_information_only'] = staged_pair['dF_set_staged_minus_x0'] - \
        staged_pair['dF_exit_view_staged_minus_x0']
    # trajectories (settled runs) and the prefix identity with the exit runs
    traj, prefix = {}, {}
    for name in ('x0', UNIT_LABEL, STAGED_NAME):
        lab = per_plan[name]['label']
        F, bad = _traj(data_root, evals, lab)
        ck('E', f'{name}: recourse identity gross - TV == net on every settled-run cycle', not bad, bad[:5])
        traj[name] = F
        ed = _exit_dir(data_root, evals, lab)
        if ed:
            fx = D.read_eval_files(ed)
            _i, Fx, _b = D.f_series(fx['record'], fx['rows'])
            ks = sorted(Fx)
            first_diff = next((k for k in ks if P._bitwise(Fx[k]) != P._bitwise(F.get(k))), None)
            prefix[name] = {'exit_run_cycles': len(ks), 'first_differing_cycle': first_diff,
                            'identical_prefix': first_diff is None}
            ck('E', f'{name}: settled run F(k) == exit run F(k) bitwise for every exit-run cycle', first_diff is None,
               prefix[name])
    at = {}
    for k in TRAJ_CYCLES + (px['read_cycle_exit'], pu['read_cycle_exit']):
        a, b, c = traj['x0'].get(k), traj[UNIT_LABEL].get(k), traj[STAGED_NAME].get(k)
        at[str(k)] = {'F_x0': a, 'F_unit': b, 'F_staged': c,
                      'unit_minus_x0': (b - a) if a is not None and b is not None else None,
                      'staged_minus_x0': (c - a) if a is not None and c is not None else None}
    kmin = max(px['read_cycle_exit'], pu['read_cycle_exit'])
    diffs = {k: traj[UNIT_LABEL][k] - traj['x0'][k] for k in range(kmin, 301)
             if traj[UNIT_LABEL].get(k) is not None and traj['x0'].get(k) is not None}
    kmax_d = max(diffs, key=lambda k: diffs[k])
    kmin_d = min(diffs, key=lambda k: diffs[k])
    last_step = {name: abs(traj[name][300] - traj[name][299]) for name in traj}
    # the 2x comparison (committed 2x evaluation)
    r2 = D.load_json(os.path.join(data_root, R2_EVAL_REL))
    t2 = r2['results']['table']
    x2 = t2['x0']
    off2 = {'x0': {'F_exit': x2['F_exit'], 'F_set': x2['F_set'], 'offset': x2['F_set'] - x2['F_exit'],
                   'exit_served_as': x2['served_as']}}
    for row in t2['rows']:
        if row.get('settled'):
            off2[row['label']] = {'F_exit': row['F_exit'], 'F_set': row['settled']['F_set'],
                                  'offset': row['settled']['F_set'] - row['F_exit'],
                                  'dF_exit': row['dF_exit'], 'dF_set': row['settled']['dF_set_vs_x0'],
                                  'pair_offset_vs_x0': row['settled']['dF_set_vs_x0'] - row['dF_exit']}
    return {'per_plan': per_plan, 'pair_unit_x0': pair, 'staged_information_only': staged_pair,
            'trajectory_at_cycles': at, 'prefix_identity': prefix,
            'pair_unit_minus_x0_over_cycles_from_read_to_300': {
                'from_cycle': kmin, 'min': diffs[kmin_d], 'min_cycle': kmin_d, 'max': diffs[kmax_d],
                'max_cycle': kmax_d, 'at_300': diffs.get(300)},
            'settled_last_step_abs_F300_minus_F299': last_step,
            'two_x_offsets_from_committed_evaluation': off2,
            'definitions': {'offset': 'F_set(x) - F_exit(x)',
                            'pair_offset': '[F_set(unit) - F_set(x0)] - [F_exit(unit) - F_exit(x0)]',
                            'e_unit_row': 'max(e_EV, bar_unit + bar_x0)'}}


# ======================================================================================================================
#  PART F -- pool, search, N(x0), indistinguishable
# ======================================================================================================================
def part_f(lattice, records, state, R, e_ev):
    import p515_s56_neighbourhood as NB
    srch = state['search']['state']
    polls = []
    for p in srch['polls'] + ([srch['poll']] if srch.get('poll') else []):
        polls.append({'index': p['index'], 'N': p['n'], 'k_classified': len(p['classified']),
                      'closed': p.get('closed'), 'cache_hits_at_start': p.get('cache_hits_at_start'),
                      'requested_in_poll': p.get('requested_in_poll'),
                      'partition': {k: sum(1 for e in p['classified'] if e.get('exit_class') == k)
                                    for k in P.EXIT_CLASSES},
                      'rows': [{'pos': e['pos'], 'label': e['label'], 'exit_class': e.get('exit_class'),
                                'dF_exit': e.get('dF_exit'), 'e': e.get('e'), 'criterion': e.get('criterion'),
                                'confirmation': e.get('confirmation'), 'cache_hit': e.get('cache_hit')}
                               for e in p['classified']]})
    inc = tuple(srch['incumbent'])
    x0 = tuple(state['config']['x0'])
    ck('F', 'final incumbent = x0; 0 moves', inc == x0 and len(srch['incumbent_history']) == 1)
    nb = NB.s56_neighbourhood(lattice, x0)
    screen = {tuple(z) for z in state['config']['screen']}
    ck('F', 'N(x0) = the 18 screen plans (zero-solve neighbourhood module)',
       {tuple(p['z']) for p in nb['points']} == screen and nb['size'] == 18, nb['size'])
    last = polls[-1]
    ck('F', 'terminal poll 18 of 18 classified, all from cache',
       last['N'] == 18 and last['k_classified'] == 18 and all(r['cache_hit'] for r in last['rows']))
    inc_set, inc_exit = records[(inc, P.DEPTH_SETTLED)], records[(inc, P.DEPTH_EXIT)]
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
                near.append({'label': lattice.label(z), 'depth': 'settled', 'dF': c['dF']})
        elif P.DEPTH_EXIT in by and by[P.DEPTH_EXIT].get('status') == 'certified':
            c = P.classify_exit(by[P.DEPTH_EXIT], inc_exit, R, e_ev)
            if abs(c['dF']) < R:
                near.append({'label': lattice.label(z), 'depth': 'exit', 'dF': c['dF']})
    return {'pool': state['pool'], 'search': {'start_label': lattice.label(tuple(srch['config']['start'])),
                                              'incumbent_history': srch['incumbent_history'],
                                              'final_incumbent_label': lattice.label(inc),
                                              'termination': srch['termination'], 'budget_used': srch['budget_used'],
                                              'polls': polls},
            'N_x0': {'size': nb['size'], 'counts': nb['counts'], 'equals_screen': True},
            'indistinguishable': {'rule': 'plan v4.1 section 4: every evaluated plan within R of the final incumbent '
                                          '(settled vs settled; exit-only vs the incumbent exit run)',
                                  'R': R, 'n_plans_with_any_record': len(plans), 'plans': near}}


# ======================================================================================================================
#  PART G -- tracking the 1x campaign specs (zero solves)
# ======================================================================================================================
def part_g(data_root, spec, lattice, oracle, evals):
    root = os.path.join(data_root, ROOT_REL)
    specs, lrecs, scope = [], [], []
    for dp, _dn, fn in os.walk(os.path.join(root, 'launches')):
        for f in fn:
            p = os.path.join(dp, f)
            if f.startswith('campaign_spec_') and f.endswith('.json'):
                specs.append(p)
            elif f.startswith('launch_') and f.endswith('.json') and os.path.basename(dp) == 'launches':
                lrecs.append(p)
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), data_root)
            if f.startswith(('campaign_spec_', 'search_spec_', 'phase_spec_')) or \
                    (f.startswith('launch_') and f.endswith('.json') and '/launches/' in rel):
                scope.append(os.path.join(data_root, rel))
    ck('G', '7 campaign specs, 14 launch records/manifests', len(specs) == 7 and len(lrecs) == 14,
       [len(specs), len(lrecs)])
    k_specs = set().union(*[R2._keys_of(p) for p in specs])
    k_lrecs = set().union(*[R2._keys_of(p) for p in lrecs])
    k_evals = {e['eval_key'] for e in evals}
    domain = D.enumerate_domain(lattice)
    k1 = {oracle.eval_key(z, d) for z in domain for d in P.DEPTHS}
    ck('G', 'keys carried by the 1x campaign specs are a subset of those of its launch records', k_specs <= k_lrecs,
       len(k_specs - k_lrecs))
    ck('G', 'the 1x domain keys the campaign specs carry == the 26 evaluation keys', (k_specs & k1) == k_evals
       and len(k_evals) == 26, [len(k_specs & k1), len(k_evals)])
    tracked = D.committed_spec_files(data_root)
    files = sorted(set(tracked) | set(scope))
    own = os.path.join(data_root, REL, '__hypothetical_1x_confirmation_root__')
    scans = {
        '1x_confirmation_freeze_declaring_phase1x_ev_v1_as_cache_root': D.collision_scan(
            files, k1, own, allowed_roots=[root], predecessor_roots=[]),
        '1x_confirmation_freeze_phase1x_ev_v1_not_declared': D.collision_scan(
            files, k1, own, allowed_roots=[], predecessor_roots=[]),
        '1x_confirmation_freeze_not_declared_launch_records_only': D.collision_scan(
            sorted(set(tracked) | {s for s in scope if s not in specs}), k1, own, allowed_roots=[],
            predecessor_roots=[]),
    }
    summ = {k: {'n_files_scanned': v['n_files_scanned'], 'n_collisions': len(v['collisions']),
                'collision_files': [os.path.relpath(c['file'], data_root) for c in v['collisions']]}
            for k, v in scans.items()}
    a = summ['1x_confirmation_freeze_declaring_phase1x_ev_v1_as_cache_root']
    ck('G', 'simulated 1x confirmation freeze declaring phase1x_ev_v1 as a cache root: 0 collisions',
       a['n_collisions'] == 0, a)
    s_all = set(summ['1x_confirmation_freeze_phase1x_ev_v1_not_declared']['collision_files'])
    s_lr = set(summ['1x_confirmation_freeze_not_declared_launch_records_only']['collision_files'])
    ck('G', 'without the root declared, the launch records alone already collide (tracking the campaign specs adds no '
            'new blocking condition)', bool(s_lr) and s_lr <= s_all and (s_all - s_lr) <= {
                os.path.relpath(s, data_root) for s in specs}, [sorted(s_all), sorted(s_lr)])
    return {'campaign_specs': [os.path.relpath(p, data_root) for p in sorted(specs)],
            'n_keys_campaign_specs': len(k_specs), 'n_keys_launch_records': len(k_lrecs), 'n_1x_domain_keys': len(k1),
            'simulated_scans': summ,
            'decision': 'track 1x campaign specs' if all(c['ok'] for c in CHECKS if c['part'] == 'G')
            else 'track launch records and manifests only'}


# ======================================================================================================================
#  PART H -- predictions Q1-Q9 (each by its frozen evaluated_as)
# ======================================================================================================================
def part_h(data_root, res, records, evals, table, lattice):
    pred_path = os.path.join(data_root, PRED_REL)
    ck('H', 'predictions file sha256 = de8313a4...', D.sha256_file(pred_path) == PRED_SHA)
    pred = D.load_json(pred_path)
    pr = pred['predictions']
    pi = res['prediction_inputs']
    rep = res['report']
    scr = [r for r in rep['marginal_value_table']['rows'] if r['source'] == 'screen']
    out = {}
    # Q1
    n_lt0 = sum(1 for r in scr if isinstance(r['dF_exit'], (int, float)) and r['dF_exit'] < 0)
    ck('H', 'Q1 input recomputes from the table rows', n_lt0 == pi['Q1']['screen_dF_exit_below_0'])
    out['Q1'] = {'statement': pr['Q1']['statement'], 'evaluated_as': pr['Q1']['evaluated_as'],
                 'screen_dF_exit_below_0': pi['Q1']['screen_dF_exit_below_0'], 'threshold_max': 9,
                 'screen_rows': pi['Q1']['screen_rows'], 'screen_certified': pi['Q1']['screen_certified'],
                 'rows_without_numeric_dF_exit': pi['Q1']['screen_rows'] - pi['Q1']['screen_certified'],
                 'min_screen_dF_exit': min(r['dF_exit'] for r in scr),
                 'max_screen_dF_exit': pi['Q1']['max_screen_dF_exit'],
                 'held': pi['Q1']['screen_dF_exit_below_0'] <= 9}
    # Q2
    he = sum(1 for s in records.values() if s.get('status') == 'harness_error')
    lsf = sum(int(e['local_solve_failures'] or 0) for e in evals)
    ck('H', 'Q2 inputs recompute (harness_error summaries, local-solve failures)',
       he == pi['Q2']['harness_error_summaries'] and lsf == pi['Q2']['local_solve_failures_total'])
    out['Q2'] = {'statement': pr['Q2']['statement'], 'evaluated_as': pr['Q2']['evaluated_as'],
                 'harness_error_summaries': pi['Q2']['harness_error_summaries'],
                 'local_solve_failures_total': pi['Q2']['local_solve_failures_total'],
                 'records_read': pi['Q2']['records_read'],
                 'held': pi['Q2']['harness_error_summaries'] == 0 and pi['Q2']['local_solve_failures_total'] == 0}
    # Q3
    out['Q3'] = {'statement': pr['Q3']['statement'], 'evaluated_as': pr['Q3']['evaluated_as'],
                 'chosen_label': rep['pool']['chosen_label'], 'best_start_is_x0': pi['Q3']['best_start_is_x0'],
                 'runner_up': rep['pool'].get('runner_up'), 'held': rep['pool']['chosen_label'] == 'x0'}
    # Q4
    out['Q4'] = {'statement': pr['Q4']['statement'], 'evaluated_as': pr['Q4']['evaluated_as'],
                 'termination_reason': rep['termination']['reason'], 'cumulative_h': res['hours']['cumulative_h'],
                 'held': rep['termination']['reason'] == 'phase_complete' and res['hours']['cumulative_h'] <= 60.0}
    # Q5
    rss = max(evals, key=lambda e: e['peak_rss_gib'])
    ck('H', 'Q5 input recomputes (max child RSS over the 26 evaluations)',
       rss['peak_rss_gib'] == pi['Q5']['peak_child_rss_gib'])
    out['Q5'] = {'statement': pr['Q5']['statement'], 'evaluated_as': pr['Q5']['evaluated_as'],
                 'peak_child_rss_gib': pi['Q5']['peak_child_rss_gib'], 'run': rss['label'], 'run_depth': rss['depth'],
                 'run_launch': rss['launch'], 'held': pi['Q5']['peak_child_rss_gib'] <= 3.29}
    # Q6
    eff = rep['efficiency']
    need = ('n_not_worse', 'n_rows', 'class_counts', 'fraction', 'branch', 'rule', 'band_measure_information_only')
    out['Q6'] = {'statement': pr['Q6']['statement'], 'evaluated_as': pr['Q6']['evaluated_as'],
                 'efficiency': eff, 'fields_present': all(k in eff for k in need),
                 'held': 'recorded' if all(k in eff for k in need) else False}
    # Q7
    out['Q7'] = {'statement': pr['Q7']['statement'], 'evaluated_as': pr['Q7']['evaluated_as'],
                 'n_candidates': len(rep['candidates']), 'n_checked': pi['Q7']['n_checked'],
                 'n_equal': pi['Q7']['n_equal'], 'termination_reason': pi['Q7']['termination_reason'],
                 'held': 'vacuous (no candidate: 0 rows classed confirm_candidate)' if not rep['candidates']
                 else pi['Q7']['n_equal'] == pi['Q7']['n_checked']}
    # Q8
    ref = pred['reference_2x']
    v3 = os.path.join(data_root, ref['phase_results'])
    v3_sha = D.sha256_file(v3)
    ck('H', 'Q8: sha256 of phase2x_ev_v3/phase_results.json == the sha256 the predictions file names',
       v3_sha == ref['phase_results_sha256'], v3_sha)
    v3r = D.load_json(v3)
    v3_scr = {r['label']: r for r in v3r['report']['marginal_value_table']['rows'] if r['source'] == 'screen'}
    ck('H', 'Q8: the 2x values copied into the predictions file == the v3 results rows (bitwise)',
       set(v3_scr) == set(ref['screen_dF_exit_2x']) and all(
           P._bitwise(v3_scr[k]['dF_exit']) == P._bitwise(v) for k, v in ref['screen_dF_exit_2x'].items()))
    one = {r['label']: r['dF_exit'] for r in scr}
    ck('H', 'Q8: 1x dF_exit from the table rows == prediction_inputs.Q8.screen_dF_exit',
       one == pi['Q8']['screen_dF_exit'])
    d = {k: one[k] - v3_scr[k]['dF_exit'] for k in sorted(one)
         if isinstance(one.get(k), (int, float)) and isinstance(v3_scr.get(k, {}).get('dF_exit'), (int, float))}
    mean_d = sum(d.values()) / len(d)
    out['Q8'] = {'statement': pr['Q8']['statement'], 'evaluated_as': pr['Q8']['evaluated_as'],
                 'reference_2x_sha256': v3_sha, 'n_labels': len(d), 'mean_difference': mean_d,
                 'n_positive': sum(v > 0 for v in d.values()), 'min_difference': min(d.values()),
                 'max_difference': max(d.values()), 'per_label': d,
                 'held': mean_d > 0 and len(d) == 18}
    # Q9
    st = [(lattice.label(z), s) for (z, dd), s in records.items() if dd == P.DEPTH_SETTLED]
    per = [{'label': lab, 'priced_loss_eur': (s.get('leak') or {}).get('priced_loss_eur'),
            'priced_full_eur': (s.get('leak') or {}).get('priced_full_eur'),
            'L_weighted_mwh': (s.get('leak') or {}).get('L_weighted_mwh'),
            'within_ceiling': (s.get('leak') or {}).get('within_ceiling')} for lab, s in sorted(st)]
    numeric = [p for p in per if isinstance(p['priced_loss_eur'], (int, float))]
    without = [p['label'] for p in per if not isinstance(p['priced_loss_eur'], (int, float))]
    above = [p['label'] for p in numeric if p['priced_loss_eur'] > 25.0]
    ck('H', 'Q9 inputs recompute (n settled summaries, n above ceiling, without leak)',
       len(st) == pi['Q9']['n_settled_summaries'] and len(above) == pi['Q9']['n_above_ceiling']
       and len(without) == len(pi['Q9']['without_leak_z']))
    out['Q9'] = {'statement': pr['Q9']['statement'], 'evaluated_as': pr['Q9']['evaluated_as'],
                 'per_settled_evaluation': per, 'n_settled_summaries': len(st), 'n_above_ceiling': len(above),
                 'above': above, 'without_numeric_leak_listed_not_counted': without,
                 'held': not above and not without}
    return out


# ======================================================================================================================
#  PART I -- 2x vs 1x
# ======================================================================================================================
def part_i(data_root, table, q8, res):
    r2 = D.load_json(os.path.join(data_root, R2_EVAL_REL))
    t2 = r2['results']['table']
    e2 = r2['results']['E']
    one = {r['label']: r for r in table['rows']}
    two = {r['label']: r for r in t2['rows']}
    rows = []
    for lab in sorted(q8['per_label']):
        rows.append({'label': lab, 'dF_exit_2x': two[lab]['dF_exit'], 'class_2x': two[lab]['class'],
                     'dF_exit_1x': one[lab]['dF_exit'], 'class_1x': one[lab]['class'],
                     'difference_1x_minus_2x': q8['per_label'][lab]})
    return {'screen': rows,
            'x0': {'2x': {'F_exit': t2['x0']['F_exit'], 'bar_exit': t2['x0']['bar_exit'], 'F_set': t2['x0']['F_set'],
                          'exit_served_as': t2['x0']['served_as']},
                   '1x': {'F_exit': table['x0']['F_exit'], 'bar_exit': table['x0']['bar_exit'],
                          'F_set': table['x0']['F_set'], 'exit_served_as': 'exit run of the phase'}},
            'selected': {'2x': {'label': e2['search']['final_incumbent_label'], 'F_set': e2['search']['F_set_inc'],
                                'dF_set_vs_x0': e2['search']['F_set_inc'] - t2['x0']['F_set'],
                                'note': 'budget-limited (author 2026-11-26: B = 1,000,000 EUR kept)'},
                         '1x': {'label': res['report']['pool']['chosen_label'], 'F_set': table['x0']['F_set'],
                                'note': 'search from x0, 0 moves'}},
            'source_2x': {'path': R2_EVAL_REL, 'sha256': R2_EVAL_SHA}}


# ======================================================================================================================
#  PART J -- projections (PROJECTED)
# ======================================================================================================================
def part_j(data_root, hours, table):
    r2 = D.load_json(os.path.join(data_root, R2_EVAL_REL))
    h2 = r2['results']['hours']
    set2 = [r['wall_h'] for r in h2['per_launch'] if r['depth'] == 'settled']
    set2_5 = [r['wall_h'] for r in h2['per_launch'] if r['depth'] == 'settled' and r['n_evaluations'] == 5]
    set1 = [r['wall_h'] for r in hours['per_launch'] if r['depth'] == 'settled']
    exit2 = [e['read_cycle'] for e in r2['results']['evaluations'] if e['depth'] == 'exit']
    exit5 = [r['wall_h'] for r in hours['per_launch'] if r['depth'] == 'exit' and r['n_evaluations'] == 5] + \
        [r['wall_h'] for r in h2['per_launch'] if r['depth'] == 'exit' and r['n_evaluations'] == 5]
    probe = D.load_json(os.path.join(data_root, PROBE_REL))
    s_cyc = probe['readings']['cycle_wall_s']
    ck('J', 'memory probe SRP1-grid SP: 2 cycle walls read (cycle 1, cycle 2)', len(s_cyc) == 2, s_cyc)
    c1, c2 = float(s_cyc[0]), float(s_cyc[1])
    exit1 = hours['exit_read_cycles']

    def t_sp(n):   # [all cycles at cycle-2 wall, all cycles at cycle-1 wall], hours
        return [n * c2 / 3600.0, n * c1 / 3600.0]
    cyc = {'exit_1x_min': exit1['min'], 'exit_1x_max': exit1['max'], 'exit_2x_min': min(exit2),
           'exit_2x_max': max(exit2), 'settled_K_EV': 300}
    tsp = {k: t_sp(v) for k, v in cyc.items()}
    n_sp = 6   # {x0, the 2x-selected plan, the reference unit} x {1x, 2x}
    two = [r for r in table['two_closest_worse']]
    unres = [r['label'] for r in table['unresolved']]
    return {
        'confirmation_launch_1x': {
            'plans': unres + two, 'n': len(unres) + len(two),
            'measured_1x_pool_settled_launch_wall_h': set1,
            'measured_2x_settled_launch_walls_h': _stats(set2),
            'measured_2x_settled_launch_walls_block_of_5_h': _stats(set2_5),
            'projected_h': [min(set1 + set2_5), max(set1 + set2_5)],
            'basis': 'one settled launch of 5 at K_EV = 300; x0 settled at 1x exists (reused); range = min/max of the '
                     '1x pool settled launch wall (3 evaluations) and the 2x settled launch walls of 5'},
        't_sp_extrapolation': {
            'probe_source': PROBE_REL, 'probe_cycle_wall_s': [c1, c2],
            'reading': 'cycle 1 = 452.0 s, cycle 2 = 307.6 s (memory_probe_srp1grid.json readings.cycle_wall_s); '
                       'SRP1-grid 3 x 3 SP instance, staged plan, two cycles only',
            'cycle_counts_measured_in_ev_phases': cyc,
            'formula': 't_SP(n cycles) = n x s/cycle, bracketed by the cycle-2 wall (low) and the cycle-1 wall (high); '
                       'EXTRAPOLATION: EV cycle counts applied to SP, s/cycle measured on 2 cycles only',
            't_sp_h': tsp,
            'sp_evaluations_6_at_K300_serial_h': [n_sp * tsp['settled_K_EV'][0], n_sp * tsp['settled_K_EV'][1]],
            'sp_evaluations_6_at_exit_depth_serial_h': [n_sp * tsp['exit_1x_min'][0], n_sp * tsp['exit_2x_max'][1]],
            'sp_evaluation_set': '{x0, the 2x-selected plan n5_y2030_p1_e4, the reference unit} x {1x, 2x} = 6',
            'dependence_stress_test_2_at_K300_h': [2 * tsp['settled_K_EV'][0], 2 * tsp['settled_K_EV'][1]],
            'plan_v4_1_section_8': {'t_SP_h': [6.5, 15.0], 'sp_evaluations_h': [40, 120],
                                    'dependence_stress_test_h': [13, 30], 'wait_and_see_h': [7, 11],
                                    'step5_pair_h': [5, 12]}},
        'wait_and_see': {
            'n_single_scenario_evaluations': 2 * 9 * 2, 'blocks_of_5': math.ceil(2 * 9 * 2 / 5),
            'projected_h_settled_depth': [math.ceil(36 / 5) * min(set1 + set2_5), math.ceil(36 / 5) * max(set1 + set2_5)],
            'projected_h_exit_depth': [math.ceil(36 / 5) * min(exit5), math.ceil(36 / 5) * max(exit5)],
            'basis': '2 plans x 9 scenarios x {1x, 2x} single-scenario evaluations, 5-way, at the measured EV block '
                     'walls (exit: 1x and 2x exit launches of 5; settled: as the confirmation launch); a single-scenario '
                     'evaluation assumed to cost an EV evaluation (EXTRAPOLATION)'},
        'calibration_a': {'A_h': 37.0 / 60.0, 'B_h': 2.6, 'source': 'TASKS.md (Deferred to idle time; handback 3)'}}


# ======================================================================================================================
#  manifest mode
# ======================================================================================================================
def manifest(data_root, eval_dir, target):
    import p515_s56_ev_harness_tests as T   # installs its own permitted=() guard (verified at 0 below)
    if os.path.exists(target):
        raise D.Refused(f'{target} exists (write-once)')
    rep = D.load_json(os.path.join(eval_dir, 'report_eval.json'))
    track_specs = rep['results']['G']['decision'] == 'track 1x campaign specs'
    roots = [os.path.join(data_root, ROOT_REL), eval_dir]
    extra = [os.path.join(data_root, REL, f'{os.path.basename(eval_dir)}_{s}.log') for s in ('stdout', 'stderr')]
    files = {}
    for root in roots:
        for dp, _dn, fn in os.walk(root):
            for f in sorted(fn):
                p = os.path.join(dp, f)
                files[os.path.relpath(p, data_root)] = p
    for p in extra:
        files[os.path.relpath(p, data_root)] = p
    out = {}
    for rel, p in sorted(files.items()):
        size = os.path.getsize(p)
        c = T.committed_by_rule(rel, size)
        why = 'selection rule'
        if c and os.path.basename(rel).startswith('campaign_spec_') and not track_specs:
            c, why = False, 'PART G: 1x campaign specs not tracked'
        out[rel] = {'sha256': D.sha256_file(p), 'bytes': size, 'committed': bool(c), 'committed_reason': why}
    obj = {'schema': SCHEMA + '_manifest', 'utc': datetime.now(timezone.utc).isoformat(),
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True,
                                      text=True).stdout.strip(),
           'selection_rule': {'source': 'p515_s56_ev_harness_tests.committed_by_rule',
                              'excluded_parts': list(T.COMMIT_EXCLUDED_PARTS),
                              'excluded_suffixes': list(T.COMMIT_EXCLUDED_SUFFIXES), 'max_bytes': T.COMMIT_MAX_BYTES,
                              'size_exempt': list(T.COMMIT_SIZE_EXEMPT), '1x_campaign_specs_tracked': track_specs},
           'roots': [os.path.relpath(r, data_root) for r in roots],
           'extra_files': [os.path.relpath(p, data_root) for p in extra],
           'n_files': len(out), 'n_committed': sum(1 for v in out.values() if v['committed']),
           'bytes_total': sum(v['bytes'] for v in out.values()),
           'bytes_committed': sum(v['bytes'] for v in out.values() if v['committed']), 'files': out}
    if obj['bytes_committed'] > 1_000_000_000:
        raise D.Refused(f'committed evidence {obj["bytes_committed"]} bytes > 1 GB (stop and report)')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'x') as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
    log(f'manifest: {obj["n_files"]} files ({obj["n_committed"]} committed, {obj["bytes_committed"] / 1e6:.1f} MB of '
        f'{obj["bytes_total"] / 1e9:.2f} GB) -> {os.path.relpath(target, data_root)} ({D.sha256_file(target)})')
    g = [('own', GUARD), ('p515_s56_phase2x_report_eval', R2.GUARD), ('p515_s56_ev_harness_tests', T.GUARD)]
    bad = [(n, gg.verify(0)) for n, gg in g if gg.verify(0)]
    if bad or any(PICKLE_COUNTS.values()) or any(R2.PICKLE_COUNTS.values()):
        log(f'GUARD FAILURE {bad} pickle {PICKLE_COUNTS} {R2.PICKLE_COUNTS}')
        return 3
    log(f'guards verify(0): [] (own, 2x evaluation module, test module); pickle {PICKLE_COUNTS}')
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
    for r in PROTECTED_ROOTS:
        if out_dir.startswith(os.path.join(data_root, REL, r) + os.sep) or out_dir == os.path.join(data_root, REL, r):
            raise D.Refused(f'--out inside a phase root: {out_dir}')
    if os.path.exists(out_dir):
        raise D.Refused(f'{out_dir} exists (write-once)')
    log(f'{SCRIPT}: zero solves; guard armed; pickle blocked')
    spec, lattice, env, oracle, records, state, res, res_sha = part_a(data_root)
    R, e_ev = spec['policy']['R'], spec['policy']['e_ev']
    evals = part_b(data_root, lattice, records)
    hours = part_c(data_root, spec, res, evals)
    table = part_d(lattice, records, state, R, e_ev)
    offsets = part_e(data_root, spec, lattice, records, evals, state, R, e_ev)
    fsec = part_f(lattice, records, state, R, e_ev)
    g = part_g(data_root, spec, lattice, oracle, evals)
    h = part_h(data_root, res, records, evals, table, lattice)
    two = part_i(data_root, table, h['Q8'], res)
    ck('I', 'committed 2x evaluation sha256 = 4427e142...',
       D.sha256_file(os.path.join(data_root, R2_EVAL_REL)) == R2_EVAL_SHA)
    proj = part_j(data_root, hours, table)
    settled = [x for x in evals if x['depth'] == 'settled']
    mem = {'peak_rss_gib_by_depth': {d: _stats([x['peak_rss_gib'] for x in evals if x['depth'] == d])
                                     for d in P.DEPTHS},
           'leak_priced_loss_eur_settled': _stats([x['leak_priced_loss_eur'] for x in settled]),
           'leak_priced_full_eur_settled': _stats([x['leak_priced_full_eur'] for x in settled]),
           'esso_slack_eur_settled': _stats([x['esso_slack_eur'] for x in settled]),
           'esso_slack_max_abs_net_settled': _stats([x['esso_slack_max_abs_net'] for x in settled]),
           'settled_runs': [{'label': x['label'], 'status': x['status'], 'gates_pass': x['gates_pass'],
                             'gates': x['gates'], 'settled_checks': x['settled_checks'], 'V': x['V'],
                             'leak_priced_loss_eur': x['leak_priced_loss_eur'],
                             'esso_slack_eur': x['esso_slack_eur'], 'cycles_run': x['cycles_run']} for x in settled],
           'esso_events_by_plan': [{'label': x['label'], 'depth': x['depth'], 'launch': x['launch'],
                                    'n_events': x['n_esso_events_file'], 'events': x['esso_events'],
                                    'reconciliation_via': x['reconciliation_via']}
                                   for x in evals if x['n_esso_events_file']],
           'n_esso_events_total': sum(x['n_esso_events_file'] for x in evals),
           'plans_with_a_0p5_mwh_cohort': sum(any(c['E_mwh'] == 0.5 for c in x['cohorts']) for x in evals),
           'terminal_step_to_threshold_ratio': 'not computed: the 2x evaluation did not report it (no 2x definition '
                                               'to reuse); settled last step |F(300) - F(299)| is in PART E'}
    ev_all = [ev for x in evals for ev in x['esso_events']]
    ck('B', 'every ESSO recovery event recovered at tier 1 (vacuous if none)',
       all(ev['recovery_succeeded'] is True and ev['tier'] == 'tier1' and not ev['tier2_attempted'] for ev in ev_all),
       len(ev_all))
    ck('B', 'all 3 settled runs: settled, gates pass, every settled check true',
       all(x['status'] == 'settled' and x['gates_pass'] is True and all((x['settled_checks'] or {}).values())
           for x in settled) and len(settled) == 3)
    ck('B', 'every guard exact on all 26 evaluations', all(all((x['guards'] or {}).values()) for x in evals))
    ck('A', 'no code drift at the end of the evaluation', not D.code_drift(spec), D.code_drift(spec))
    payload = {'schema': SCHEMA, 'utc': datetime.now(timezone.utc).isoformat(), 'script': SCRIPT,
               'script_sha256': D.sha256_file(os.path.abspath(__file__)),
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True,
                                          text=True).stdout.strip(),
               'inputs': {'phase_root': ROOT_REL, 'phase_spec_sha256': SPEC_SHA, 'phase_results_sha256': res_sha,
                          'predictions': PRED_REL, 'predictions_sha256': PRED_SHA, 'two_x_evaluation': R2_EVAL_REL,
                          'two_x_evaluation_sha256': R2_EVAL_SHA, 'memory_probe': PROBE_REL,
                          'memory_probe_sha256': D.sha256_file(os.path.join(data_root, PROBE_REL)),
                          'code_sha256_now': D.code_hashes()},
               'objective_convention': 'F = I(x) + gross_operational_cost - TV credit, EUR of total discounted '
                                       'planning cost (oracle mapping v2; gross settlement-excluded)',
               'results': {'hours': hours, 'table': table, 'offsets': offsets, 'pool_search': fsec, 'G': g,
                           'predictions': h, 'two_x_vs_one_x': two, 'projections': proj, 'leak_slack_memory': mem,
                           'evaluations': evals},
               'checks': CHECKS}
    os.makedirs(out_dir)
    gv, gv2 = GUARD.verify(0), R2.GUARD.verify(0)
    payload['guard'] = {'verify_0': gv, 'verify_0_2x_module': gv2, 'counts': dict(GUARD.counts),
                        'pickle': dict(PICKLE_COUNTS), 'pickle_2x_module': dict(R2.PICKLE_COUNTS)}
    payload['all_checks_ok'] = all(c['ok'] for c in CHECKS) and not gv and not gv2 and \
        not any(PICKLE_COUNTS.values()) and not any(R2.PICKLE_COUNTS.values())
    path = os.path.join(out_dir, 'report_eval.json')
    with open(path, 'x') as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    log(f'wrote {os.path.relpath(path, data_root)} ({D.sha256_file(path)}); checks {sum(c["ok"] for c in CHECKS)}/'
        f'{len(CHECKS)}; guard verify(0) {gv} {gv2}; pickle {PICKLE_COUNTS} {R2.PICKLE_COUNTS}')
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
