"""
P5.15 W183g (b) -- the ZERO-SOLVE dry-run gate of the E1 launcher v2 (PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 2,
E1 poll rule B+; ruling 3, A7 confirmed; ruling 4 (iv), the batch seeds and the roots).

WHAT IS TESTED. `p515_s55_e1_launcher.run_continuation_e1` (the launcher's own loop: a copy of
`p515_s53_f2_certificate.run_continuation` with the B+ advance) AGAINST THE IMPORTED, UNTOUCHED
`p515_s53_f2_certificate.run_continuation` -- a real equivalence test, never a copy compared with itself: both loops
receive the same lattice, the same key_of, a `copy.deepcopy` of the same cache (entries carrying status 'certified',
Q, bar, canonical, as the loop reads them), the same incumbent, the SAME pure label-keyed stub evaluator object and the
same callbacks; both full return dicts are serialised with the same writer (`gate_result_io.dumps(sort_keys=True)`)
and compared AS BYTES after removing exactly the two new keys (`halton_indices_skipped`, `advance_rule`) from the
copy's records, whose values are asserted to be the trivial ones; the stub's call logs, the loop log lines and the
callback sequences are compared too.

  1. ADVANCE OFF == IMPORTED LOOP, 8 cases (4 seeds x 2 arms, the incumbents of `first_poll_by_seed`): expected
     STOP_FOR_REVIEW_completion_cap at poll 0 for the six storage cases, the 14-point completion for x = 0.
  2. ADVANCE ON, from W183b's committed table (`w183b_e1_dryrun.json`, sha256 pinned): unit polled at k = 2 (8
     points), skipped [0, 1]; 7: 0.5/2 at k = 2 (9), skipped [0, 1]; the historical incumbent at k = 1 (8), skipped
     [0]; x = 0 at k = 0 (completion 14), nothing skipped; every skipped / polled size checked against the table; the
     k = 0...9 table reproduced by the unchanged `first_poll_by_seed`.
  3. PATHS THE POLL-0 GATE DOES NOT REACH, each a paired run (advance off: bitwise equal to the imported loop; advance
     on: equal up to the first advance, then only the declared differences): A7 per poll; A7 overall; an accepted
     point moving the incumbent to a storage point (next poll examined at k + 1, advance thereafter; the index
     continuing from the last polled index + 1); budget exhaustion (max_new_evaluations = 0; the unit with
     max_new = 5); max_polls = 1 with an improving stub (x = 0 and the unit); window exhaustion (iii) on an incumbent
     found by a domain scan with pure `C.build_poll` (completion > 30, < 8 distinct points at 10 consecutive k).
  4. The unified diff of inspect.getsource(C.run_continuation) against run_continuation_e1, recorded; asserted to
     touch only the signature, the explicit k_next state and the inserted B+ block (which holds the two record fields).
  5. The v1 dry-run assertions on the v2 launcher: `p515_s55_e1_launcher.dry_run` run in-process into
     <out>/launcher_dry_run/ with the final-head G0 pointer (7 keys + 1 alias, base keys rows 35 / 40, root-scoped
     cache, pins, preflight); plus g0_check with require_current_production=True (the --freeze / --run condition) and
     the declared roots / batch seeds of ruling 4 (iv).

GUARDS. SolveProfileGuard(permitted=()) installed before any other project import and verified at exactly 0 with
every guard the launcher's imports install; pickle.load / loads blocked (the launcher's counters); the launcher's
pyomo ConcreteModel tripwire (no model build) and its 52 tripwires on the committed launchers' functions at 0.

EXACT COMMAND (repo root, canonical interpreter, attached, alone, both streams captured):
  mkdir data/SRP1/Results/P515S55/w183g_b_e1_v2_dryrun && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183g_b_e1_v2_dryrun.py \\
      --out data/SRP1/Results/P515S55/w183g_b_e1_v2_dryrun \\
      --g0-artifact data/SRP1/Results/P515S55/w183_nl_post_w183c_off/w183_nl_digest.json.gz \\
      --g0-sha256 ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a \\
      --scratch <a new directory outside the repository> \\
      > data/SRP1/Results/P515S55/w183g_b_e1_v2_dryrun/launch.log 2>&1
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GATE_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183g (b) E1 v2 dry-run gate (never solves)').install()

import argparse  # noqa: E402
import copy  # noqa: E402
import difflib  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import p515_s55_e1_launcher as L  # noqa: E402  (installs its guard, blocks pickle, the model-build and tripwires)

C, PB, H, GRIO = L.C, L.PB, L.H, L.GRIO

SCRIPT_NAME = os.path.basename(__file__)
OWN_SUBSTRINGS = ('p515_s55_w183g_b_e1_v2_dryrun', 'p515_s55_e1_launcher')
OUT_FILE = 'w183g_b_e1_v2_dryrun.json'
OUT_RUNS_FILE = 'w183g_b_loop_outputs.json.gz'
OUT_MANIFEST = 'w183g_b_e1_v2_dryrun_manifest_sha256.json'
LAUNCHER_DRY_RUN_SUBDIR = 'launcher_dry_run'
W183B_DRYRUN = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183b_e1_dryrun', 'w183b_e1_dryrun.json'),
                'sha256': 'e56a534ca7a6f29cddd9a8ee4ebc752088413ad65c65c186f418193f1032afb3'}
LAUNCHER_V1_SHA256 = '320d503cca1bb68736ca0ddac78a6927f4701a44cec6ee8b035cb143ee1d6be7'
G0_FINAL = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183_nl_post_w183c_off',
                                 'w183_nl_digest.json.gz'),
            'sha256': 'ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a'}
UNCHANGED_MODULES = ('p515_s53_f2_certificate.py', 'p515_s47_phase_b_record.py', 'p515_s44_campaign_harness.py')
NEW_FIELDS = ('halton_indices_skipped', 'advance_rule')
# Addendum 85 ruling 2: the expected B+ first poll per seed (checks, not hypotheses: they follow from W183b's table)
EXPECTED_B_PLUS = {'unit': {'k': 2, 'size': 8, 'skipped': [0, 1], 'completion_used': False},
                   'n7_05_2': {'k': 2, 'size': 9, 'skipped': [0, 1], 'completion_used': False},
                   'hist_inc': {'k': 1, 'size': 8, 'skipped': [0], 'completion_used': False},
                   'x0': {'k': 0, 'size': 14, 'skipped': [], 'completion_used': True}}
EXPECTED_STORAGE_REFUSAL_COMPLETION = {'unit': 31, 'n7_05_2': 34, 'hist_inc': 61}
SCAN_K_MAX = 99            # every window a <= 10-launched-poll arm can examine lies in k = 0 ... 99
STUB_BAR = 1000.0
Q0 = 8.0e8


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W183g-b] {msg}', flush=True)


def _sha(rel):
    return H.sha256_file(rel if os.path.isabs(rel) else os.path.join(REPO, rel))


def _ser(obj):
    return GRIO.dumps(obj, sort_keys=True)


def _git(args):
    return H._git(args)


def _own_copies_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and 'python' in parts[1] \
                and any(s in parts[1] for s in OWN_SUBSTRINGS):
            hits.append(line.strip())
    return hits


# ======================================================================================================================
#  the stub evaluator (pure, label-keyed) and the incumbents
# ======================================================================================================================
def q_hash(label, spread=2_000_000):
    """A deterministic Q from the label only: Q0 +- spread (sha256 of the label)."""
    h = int(hashlib.sha256(label.encode()).hexdigest()[:12], 16)
    return Q0 + float(h % (2 * spread + 1) - spread)


class LabelStub:
    """evaluate_fn for both loops: the record of z depends on lattice.label(z) only (the eval key is the arm's key_of
    of z). `q_of` : label -> Q; `barrier` : the labels returned not certified. Logs every call (the batch labels)."""

    def __init__(self, lattice, key_of, q_of, barrier=(), name='stub'):
        self.lattice, self.key_of, self.q_of, self.barrier, self.name = lattice, key_of, q_of, set(barrier), name
        self.calls = []

    def record(self, z):
        lab = self.lattice.label(z)
        bad = lab in self.barrier
        return {'label': lab, 'status': 'not_certified' if bad else 'certified', 'eval_key': self.key_of(z),
                'Q': None if bad else self.q_of(lab), 'bar': None if bad else STUB_BAR,
                'canonical': L.canonical_of(self.lattice, z), 'barrier_cause': 'w183g_b_stub_barrier' if bad else None,
                'source': {'kind': 'w183g_b_stub', 'stub': self.name}}

    def __call__(self, batch):
        self.calls.append([self.lattice.label(z) for z in batch])
        return [self.record(z) for z in batch]


def incumbent_and_cache(lattice, key_of, stub, z_inc, cache_points):
    """The incumbent (from the stub's own record of z_inc) and a cache of the stub's records of `cache_points`."""
    cache = {}
    for z in cache_points:
        rec = stub.record(z)
        cache[rec['eval_key']] = {k: rec[k] for k in ('label', 'status', 'Q', 'bar', 'canonical', 'barrier_cause',
                                                      'source')}
    rec = stub.record(z_inc)
    if rec['status'] != 'certified':
        raise AssertionError('the incumbent must be certified in the stub')
    i_x = lattice.investment_cost(z_inc)
    inc = {'eval_key': rec['eval_key'], 'z': tuple(z_inc), 'label': rec['label'], 'I': i_x, 'Q': rec['Q'],
           'F': i_x + rec['Q'], 'bar': rec['bar'], 'source': rec['source']}
    if rec['eval_key'] not in cache:
        raise AssertionError('the incumbent must be in the cache')
    return inc, cache


# ======================================================================================================================
#  one paired scenario: imported loop, the copy with advance OFF, the copy with advance ON
# ======================================================================================================================
def _run(fn, stub, lattice, cache, key_of, inc, sigma_q, kw, extra=None):
    logs, starts, finishes = [], [], []
    n_calls0 = len(stub.calls)
    out = fn(lattice, copy.deepcopy(cache), key_of, dict(inc), stub, sigma_q, log=logs.append,
             on_poll=lambda r: finishes.append(r['poll_index']),
             on_poll_start=lambda r: starts.append(r['poll_index']), **kw, **(extra or {}))
    return {'out': out, 'calls': copy.deepcopy(stub.calls[n_calls0:]), 'logs': logs, 'on_poll_start': starts,
            'on_poll': finishes}


def strip_new_fields(out):
    """A deepcopy of a run_continuation_e1 return with exactly the two new keys removed from every history record;
    returns (stripped, [(skipped, advance_rule) per record])."""
    s = copy.deepcopy(out)
    popped = []
    for rec in s['history']:
        popped.append((rec.pop('halton_indices_skipped'), rec.pop('advance_rule')))
        if any(f in rec for f in NEW_FIELDS):
            raise AssertionError('new field still present')
    return s, popped


def projection_of_refused_record(rec):
    """W183g (b)'s OWN projection (independent of L._skipped_index_entry) of an imported-loop record refused by the
    completion cap onto the declared skipped-index fields."""
    return {'k': rec['poll_index'], 'halton_t': rec['halton_t'], 'halton_u': rec['halton_u'],
            'directions': rec['directions'], 'snap_table': rec['snap_table'],
            'n_distinct_feasible_poll_points': rec['n_distinct_feasible_poll_points'],
            'completion': {'n_feasible': rec['completion']['n_feasible'], 'over_cap': rec['completion']['over_cap'],
                           'cap': rec['completion']['cap']},
            'candidates': [{'label': c['label'], 'poll_part': c['poll_part'], 'direction_index': c['direction_index'],
                            'disposition': c['disposition']} for c in rec['candidates']]}


def trivial_values_ok(rec_stripped, popped, advance, window):
    """The two new fields at their trivial values (no advance happened at this record)."""
    skipped, rule = popped
    refused = rec_stripped.get('decision') == 'stopped_for_review_completion_cap'
    return (skipped == [] and rule.get('enabled') is advance and rule.get('window') == window
            and rule.get('first_index') == rec_stripped['poll_index'] and rule.get('n_examined') == 1
            and rule.get('polled_index') == (None if refused else rec_stripped['poll_index'])
            and rule.get('text') == L.B_PLUS_RULE and sorted(rule) == sorted(L.ADVANCE_RULE_FIELDS))


def compare_off(imp, off, window):
    """advance OFF against the imported loop: bytes after removing the two fields; calls, logs, callbacks equal."""
    stripped, popped = strip_new_fields(off['out'])
    a, b = _ser(imp['out']), _ser(stripped)
    triv = [trivial_values_ok(r, p, False, window) for r, p in zip(stripped['history'], popped)]
    return {'return_dict_bytes_equal_after_removing_the_two_fields': a == b,
            'imported_sha256': hashlib.sha256(a.encode()).hexdigest(),
            'off_stripped_sha256': hashlib.sha256(b.encode()).hexdigest(), 'n_bytes': len(a),
            'two_fields_trivial_on_every_record': all(triv) and len(triv) == len(stripped['history']),
            'stub_calls_equal': imp['calls'] == off['calls'], 'log_lines_equal': imp['logs'] == off['logs'],
            'on_poll_start_equal': imp['on_poll_start'] == off['on_poll_start'],
            'on_poll_equal': imp['on_poll'] == off['on_poll'], 'n_polls': len(imp['out']['history'])}


def compare_on(imp, on, window):
    """advance ON against the imported loop: equal up to the first advance, then only the declared differences.
    First advance = the first record whose advance_rule.n_examined > 1."""
    stripped, popped = strip_new_fields(on['out'])
    hist_on, hist_imp = on['out']['history'], imp['out']['history']
    j = next((i for i, r in enumerate(hist_on) if r['advance_rule']['n_examined'] > 1), None)
    res = {'first_advance_record_index': j}
    if j is None:
        a, b = _ser(imp['out']), _ser(stripped)
        triv = [trivial_values_ok(r, p, True, window) for r, p in zip(stripped['history'], popped)]
        res.update({'advance_happened': False,
                    'return_dict_bytes_equal_after_removing_the_two_fields': a == b,
                    'two_fields_trivial_on_every_record': all(triv) and len(triv) == len(stripped['history']),
                    'stub_calls_equal': imp['calls'] == on['calls'], 'log_lines_equal': imp['logs'] == on['logs'],
                    'on_poll_start_equal': imp['on_poll_start'] == on['on_poll_start']})
        res['ok'] = all(v for k, v in res.items() if k not in ('first_advance_record_index', 'advance_happened'))
        return res
    rec_on = hist_on[j]
    rule, skipped = rec_on['advance_rule'], rec_on['halton_indices_skipped']
    prior_equal = all(_ser(stripped['history'][i]) == _ser(hist_imp[i]) for i in range(j)) and len(hist_imp) > j
    prior_trivial = all(trivial_values_ok(stripped['history'][i], popped[i], True, window) for i in range(j))
    rec_imp = hist_imp[j] if len(hist_imp) > j else None
    imp_refused_here = (rec_imp is not None and rec_imp['decision'] == 'stopped_for_review_completion_cap'
                        and len(hist_imp) == j + 1
                        and imp['out']['termination']['reason'] == 'STOP_FOR_REVIEW_completion_cap')
    case_iii = rule['polled_index'] is None
    res.update({'advance_happened': True, 'case': 'iii (window exhausted)' if case_iii else 'ii (advanced and polled)',
                'records_before_first_advance_bytes_equal': prior_equal,
                'records_before_first_advance_two_fields_trivial': prior_trivial,
                'imported_refused_at_the_same_record_and_stopped': imp_refused_here,
                'first_index_equals_imported_refused_index': rec_imp is not None
                and rule['first_index'] == rec_imp['poll_index'],
                'n_examined': rule['n_examined'], 'skipped_ks': [s['k'] for s in skipped],
                'polled_index': rule['polled_index'], 'window': rule['window']})
    if case_iii:
        res['iii_record_bytes_equal_to_imported_record'] = _ser(stripped['history'][j]) == _ser(rec_imp)
        res['iii_skipped_are_the_further_indices'] = ([s['k'] for s in skipped]
                                                      == list(range(rule['first_index'] + 1,
                                                                    rule['first_index'] + window)))
        res['iii_n_examined_eq_window'] = rule['n_examined'] == window
        res['iii_every_skipped_below_8'] = all(s['n_distinct_feasible_poll_points'] < C.MIN_FEASIBLE_POLL_POINTS
                                               for s in skipped)
        res['iii_termination_equal'] = _ser(imp['out']['termination']) == _ser(on['out']['termination'])
        res['iii_return_dict_bytes_equal_after_removing_the_two_fields'] = _ser(imp['out']) == _ser(stripped)
    else:
        res['skipped_0_equals_projection_of_the_imported_refused_record'] = (
            rec_imp is not None and _ser(skipped[0]) == _ser(projection_of_refused_record(rec_imp)))
        res['skipped_contiguous_from_first_index'] = ([s['k'] for s in skipped]
                                                      == list(range(rule['first_index'], rec_on['poll_index'])))
        res['polled_index_is_poll_index'] = rule['polled_index'] == rec_on['poll_index']
        res['n_examined_eq_len_skipped_plus_1'] = rule['n_examined'] == len(skipped) + 1 <= window
        res['every_skipped_below_8_over_cap'] = all(
            s['n_distinct_feasible_poll_points'] < C.MIN_FEASIBLE_POLL_POINTS and s['completion']['over_cap']
            for s in skipped)
        res['polled_record_has_8_or_more_no_completion'] = (
            rec_on['n_distinct_feasible_poll_points'] >= C.MIN_FEASIBLE_POLL_POINTS
            and rec_on['completion_triggered'] is False)
        res['stub_calls_equal_up_to_the_first_advance'] = on['calls'][:len(imp['calls'])] == imp['calls']
    res['on_poll_start_once_per_launched_poll_on_the_chosen_record'] = (
        on['on_poll_start'] == [r['poll_index'] for r in hist_on] == on['on_poll'])
    res['index_continues_from_last_polled_plus_1'] = all(
        hist_on[i + 1]['advance_rule']['first_index'] == hist_on[i]['poll_index'] + 1 for i in range(len(hist_on) - 1))
    res['ok'] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


def paired(name, lattice, key_of, stub, inc, cache, sigma_q, kw, window=L.ADVANCE_WINDOW):
    imp = _run(C.run_continuation, stub, lattice, cache, key_of, inc, sigma_q, kw)
    off = _run(L.run_continuation_e1, stub, lattice, cache, key_of, inc, sigma_q, kw,
               {'advance': False, 'advance_window': window})
    on = _run(L.run_continuation_e1, stub, lattice, cache, key_of, inc, sigma_q, kw,
              {'advance': True, 'advance_window': window})
    c_off, c_on = compare_off(imp, off, window), compare_on(imp, on, window)
    summary = {}
    for tag, r in (('imported', imp), ('off', off), ('on', on)):
        o = r['out']
        summary[tag] = {'termination': o['termination'], 'n_polls': o['n_polls'],
                        'n_new_evaluations': o['n_new_evaluations'],
                        'n_barrier_new_evaluations': o['n_barrier_new_evaluations'],
                        'final_incumbent': o['incumbent']['label'],
                        'polls': [{'poll_index': h['poll_index'], 'halton_t': h['halton_t'],
                                   'incumbent': h['incumbent']['label'],
                                   'n_distinct': h['n_distinct_feasible_poll_points'],
                                   'completion_triggered': h['completion_triggered'],
                                   'completion_n_feasible': (h['completion'] or {}).get('n_feasible'),
                                   'n_polled': sum(c['disposition'] in ('cache_hit', 'new_evaluation')
                                                   for c in h['candidates']),
                                   'n_new': h['n_new_evaluations'], 'decision': h['decision'],
                                   'skipped_ks': [s['k'] for s in h.get('halton_indices_skipped', [])],
                                   'advance_rule': {k: v for k, v in (h.get('advance_rule') or {}).items()
                                                    if k != 'text'} or None}
                                  for h in o['history']],
                        'n_stub_calls': len(r['calls'])}
    _log(f"{name}: imported {summary['imported']['termination']['reason']} ({summary['imported']['n_polls']} polls); "
         f"off == imported {c_off['return_dict_bytes_equal_after_removing_the_two_fields']}; on "
         f"{summary['on']['termination']['reason']} ({summary['on']['n_polls']} polls, ks "
         f"{[p['poll_index'] for p in summary['on']['polls']]}), on-vs-imported ok {c_on['ok']}")
    runs = {'imported': imp, 'off': off, 'on': on}
    return {'name': name, 'kwargs': kw, 'window': window, 'incumbent': inc['label'], 'stub': stub.name,
            'off_vs_imported': c_off, 'on_vs_imported': c_on, 'summary': summary}, runs


def first_record(r):
    return r['out']['history'][0]


# ======================================================================================================================
#  helpers to build the path scenarios (pure listings with C.build_poll; nothing evaluated)
# ======================================================================================================================
def poll_points(lattice, key_of, z, k, sigma_q):
    inc = {'eval_key': key_of(z), 'z': tuple(z), 'label': lattice.label(z), 'I': 0.0, 'Q': None, 'F': None,
           'bar': None}
    rec, comp = C.build_poll(lattice, {}, key_of, inc, k, sigma_q)
    if comp is not None and comp['over_cap']:
        return rec, None
    return rec, [tuple(c['z']) for c in rec['candidates'] if c['disposition'] in ('cache_hit', 'new_evaluation')]


def scan_window_exhaustion(lattice, key_of, sigma_q, window=L.ADVANCE_WINDOW, k_max=SCAN_K_MAX):
    """Every domain point whose completion exceeds the cap and which has < 8 distinct feasible poll points at
    `window` consecutive Halton indices within k = 0 ... k_max (pure C.build_poll, empty cache; the count does not
    depend on the arm: the key is injective on canonical points in both arms). A run of `window` consecutive failures
    contains exactly one index k = window - 1 (mod window): only those are probed first, then extended."""
    started = time.time()
    over = [z for z in lattice.domain() if lattice.completion(z)['n_feasible'] > L.COMPLETION_CAP]
    hits, n_builds = [], 0
    for idx, z in enumerate(over):
        memo = {}
        inc = {'eval_key': key_of(z), 'z': z, 'label': lattice.label(z), 'I': 0.0, 'Q': None, 'F': None,
               'bar': None}

        def nd(k):
            nonlocal n_builds
            if k not in memo:
                rec, _comp = C.build_poll(lattice, {}, key_of, inc, k, sigma_q)
                n_builds += 1
                memo[k] = rec['n_distinct_feasible_poll_points']
            return memo[k]
        for kc in range(window - 1, k_max + 1, window):
            if nd(kc) >= C.MIN_FEASIBLE_POLL_POINTS:
                continue
            lo = kc
            while lo - 1 >= 0 and kc - (lo - 1) < window and nd(lo - 1) < C.MIN_FEASIBLE_POLL_POINTS:
                lo -= 1
            hi = kc
            while hi + 1 <= k_max and hi - lo + 1 < window and nd(hi + 1) < C.MIN_FEASIBLE_POLL_POINTS:
                hi += 1
            if hi - lo + 1 >= window:
                hits.append({'label': lattice.label(z), 'z': list(z),
                             'completion_n_feasible': lattice.completion(z)['n_feasible'],
                             'window_first_k': lo, 'window_last_k': hi,
                             'n_distinct_in_window': [memo[k] for k in range(lo, hi + 1)]})
                break
        if idx % 200 == 0:
            _log(f'  scan {idx}/{len(over)}: {n_builds} build_poll calls, {len(hits)} hits, '
                 f'{time.time() - started:.0f} s')
    return {'rule': ('domain points with completion > 30 having < 8 distinct feasible poll points at 10 consecutive '
                     f'Halton indices within k = 0 ... {k_max} (every window an arm of <= 10 launched polls can '
                     'examine); probe k = 9, 19, ..., then extend; first hit per point recorded; pure C.build_poll, '
                     'empty cache, arm A keys'),
            'n_domain': len(lattice.domain()), 'n_completion_over_cap': len(over), 'n_build_poll_calls': n_builds,
            'n_hits': len(hits), 'hits': hits, 'wall_s': time.time() - started}


# ======================================================================================================================
#  item 4: the source diff
# ======================================================================================================================
def source_diff():
    old = inspect.getsource(C.run_continuation).splitlines()
    new = inspect.getsource(L.run_continuation_e1).splitlines()
    unified = '\n'.join(difflib.unified_diff(old, new, 'p515_s53_f2_certificate.run_continuation',
                                             'p515_s55_e1_launcher.run_continuation_e1', lineterm='', n=2))
    header_old = next(i for i, ln in enumerate(old) if ln.rstrip().endswith('):')) + 1
    header_new = next(i for i, ln in enumerate(new) if ln.rstrip().endswith('):')) + 1
    b0 = next(i for i, ln in enumerate(new) if ln.strip().startswith('# >>> B+ advance'))
    b1 = next(i for i, ln in enumerate(new) if ln.strip().startswith('# <<< B+ advance'))
    k_state_added = {'    k_next = k0', '        k = k_next'}
    k_state_removed = {'        k = k0 + i'}
    rows, bad = [], []
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        for i in range(i1, i2):
            cat = ('signature' if i < header_old else 'k_next_state' if old[i] in k_state_removed else None)
            rows.append({'side': '-', 'line_no': i + 1, 'text': old[i], 'category': cat})
            if cat is None:
                bad.append(rows[-1])
        for j in range(j1, j2):
            if j < header_new:
                cat = 'signature'
            elif new[j] in k_state_added:
                cat = 'k_next_state'
            elif b0 <= j <= b1:
                cat = ('inserted_block:record_field' if ("record['halton_indices_skipped']" in new[j]
                                                         or "record['advance_rule']" in new[j]
                                                         or (j > 0 and "record['advance_rule']" in new[j - 1])
                                                         or "'n_examined': n_examined, 'text'" in new[j]
                                                         or "'polled_index': None if" in new[j])
                       else 'inserted_block:k_next_state' if new[j].strip() == 'k_next = k + 1'
                       else 'inserted_block')
            else:
                cat = None
            rows.append({'side': '+', 'line_no': j + 1, 'text': new[j], 'category': cat})
            if cat is None:
                bad.append(rows[-1])
    docstring_old = old[header_old:header_old + 3]
    docstring_new = new[header_new:header_new + 3]
    counts = {}
    for r in rows:
        counts[f"{r['side']}{r['category']}"] = counts.get(f"{r['side']}{r['category']}", 0) + 1
    return {'unified_diff': unified, 'classified_lines': rows, 'counts_by_side_and_category': counts,
            'uncategorised_lines': bad, 'only_declared_categories': not bad,
            'body_after_block_identical': old[old.index('        cands = record[\'candidates\']'):]
            == new[new.index('        cands = record[\'candidates\']'):],
            'docstring_identical': docstring_old == docstring_new,
            'old_sha256': hashlib.sha256('\n'.join(old).encode()).hexdigest(),
            'new_sha256': hashlib.sha256('\n'.join(new).encode()).hexdigest(),
            'note': ('the signature differs necessarily (the name, the E1 defaults and the advance / advance_window '
                     'parameters); every other changed line is the k_next state or inside the marked B+ block, which '
                     'holds the two record fields')}


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description='W183g (b) zero-solve dry-run gate of the E1 launcher v2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--g0-artifact', required=True)
    ap.add_argument('--g0-sha256', required=True)
    ap.add_argument('--scratch', required=True)
    args = ap.parse_args()
    started = time.time()
    alive = _own_copies_alive()
    if alive:
        raise SystemExit(f'REFUSED: another copy of this gate or of the launcher is alive: {alive}')
    out_rel, out_abs = L.resolve_root(args.out)
    names = sorted(os.listdir(out_abs)) if os.path.isdir(out_abs) else None
    if names is None or [n for n in names if n != 'launch.log']:
        raise SystemExit(f'REFUSED: --out must be a fresh directory holding only launch.log: {names}')
    if (args.g0_artifact, args.g0_sha256) != (G0_FINAL['path'], G0_FINAL['sha256']):
        raise SystemExit(f'REFUSED: the G0 pointer must be the final-head one (Addendum 85 ruling 4 v): {G0_FINAL}')
    res = {'schema': 'p515_s55_w183g_b_e1_v2_dryrun_v1', 'started_utc': L._utc(), 'argv': sys.argv,
           'interpreter': sys.executable, 'git_head': _git(['rev-parse', 'HEAD']),
           'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 85 rulings 2, 3, 4 (iv); W183g order item (b)'],
           'objective_convention': ('no objective value of the problem is computed: every Q is a synthetic stub value '
                                    '(F = I + Q as the loop defines it, Q = gross_operational_cost convention)'),
           'zero_solves': 'SolveProfileGuard(permitted=()) before any other project import; verified at 0'}
    checks = {}
    # ---- 0. pins and provenance ----
    prov = {'launcher_sha256': _sha(L.SCRIPT_NAME), 'launcher_v1_sha256': LAUNCHER_V1_SHA256,
            'gate_sha256': _sha(SCRIPT_NAME),
            'unchanged_modules': {m: {'sha256': _sha(m), 'git_clean': not _git(['status', '--porcelain', '--', m]),
                                      'tracked': bool(_git(['ls-files', '--', m]))} for m in UNCHANGED_MODULES},
            'w183b_dryrun': dict(W183B_DRYRUN, sha256_on_disk=_sha(W183B_DRYRUN['path'])),
            'launcher_and_gate_committed_clean': not _git(['status', '--porcelain', '--', L.SCRIPT_NAME,
                                                           SCRIPT_NAME])}
    res['provenance'] = prov
    checks['provenance'] = {
        'w183b_dryrun_sha256_pinned': prov['w183b_dryrun']['sha256_on_disk'] == W183B_DRYRUN['sha256'],
        'certificate_pb_harness_unchanged_clean': all(v['git_clean'] and v['tracked']
                                                      for v in prov['unchanged_modules'].values()),
        'launcher_is_v2_not_v1': prov['launcher_sha256'] != LAUNCHER_V1_SHA256,
        'launcher_and_gate_committed_clean': prov['launcher_and_gate_committed_clean']}
    w183b = json.load(open(W183B_DRYRUN['path']))
    table = w183b['first_poll_if_each_seed_were_the_incumbent']
    lattice, lev, lfail = L.build_lattice()
    if lfail:
        raise SystemExit(f'REFUSED (search tables): {lfail}')
    sigma_q = lev['sigma_Q']['sigma_Q_eur']
    seeds = {s: z for s, z, _n, _y, _t in L.SEEDS}
    runs_out = {}

    # ---- 1 + 2. the 8 seed-as-incumbent cases: advance off == imported; advance on (B+) against the table ----
    item1, item2 = {}, {}
    for arm in L.MODELS:
        key_of = L.make_arm_key_of(lattice, arm)
        for seed, z in seeds.items():
            case = f'{seed}_{arm}'
            stub = LabelStub(lattice, key_of, q_hash, barrier=(), name=f'q_hash_{case}')
            inc, cache = incumbent_and_cache(lattice, key_of, stub, z, seeds.values())
            kw = {'k0': L.HALTON_K0, 'max_new_evaluations': L.N_MAX_NEW_EVALUATIONS, 'max_polls': L.MAX_POLLS,
                  'batch_size': 1, 'completion_cap': L.COMPLETION_CAP}
            rec, runs = paired(f'seed case {case}', lattice, key_of, stub, inc, cache, sigma_q, kw)
            runs_out[f'item12_{case}'] = runs
            imp0, off0, on0 = (first_record(runs[t]) for t in ('imported', 'off', 'on'))
            polled = [c for c in on0['candidates'] if c['disposition'] in ('cache_hit', 'new_evaluation')]
            exp = EXPECTED_B_PLUS[seed]
            if seed == 'x0':
                expected_poll0 = (imp0['completion_triggered'] is True and imp0['completion']['n_feasible'] == 14
                                  and not imp0['completion']['over_cap']
                                  and sum(c['poll_part'] == 'completion' for c in imp0['candidates']) == 14
                                  and sum(c['disposition'] in ('cache_hit', 'new_evaluation')
                                          for c in imp0['candidates']) == 14)
            else:
                expected_poll0 = (imp0['decision'] == 'stopped_for_review_completion_cap'
                                  and runs['imported']['out']['termination']['reason']
                                  == 'STOP_FOR_REVIEW_completion_cap' and runs['imported']['out']['n_polls'] == 1
                                  and imp0['completion']['n_feasible'] == EXPECTED_STORAGE_REFUSAL_COMPLETION[seed]
                                  and runs['imported']['calls'] == [])
            item1[case] = dict(rec['off_vs_imported'], expected_poll0_decision=expected_poll0,
                               imported_poll0={'decision': imp0['decision'], 'n_distinct':
                                               imp0['n_distinct_feasible_poll_points'],
                                               'completion_n_feasible': (imp0['completion'] or {}).get('n_feasible'),
                                               'termination': runs['imported']['out']['termination']['reason']})
            trow = table[case]
            by_k = {r['k']: r for r in trow['same_incumbent_at_k_0_to_9']}
            skipped = on0['halton_indices_skipped']
            item2[case] = {
                'polled_k': on0['poll_index'], 'expected_k': exp['k'], 'poll_size': len(polled),
                'expected_size': exp['size'], 'skipped_ks': [s['k'] for s in skipped],
                'expected_skipped': exp['skipped'], 'completion_used': on0['completion_triggered'],
                'k_ok': on0['poll_index'] == exp['k'], 'size_ok': len(polled) == exp['size'],
                'skipped_ok': [s['k'] for s in skipped] == exp['skipped'],
                'completion_use_ok': on0['completion_triggered'] is exp['completion_used'],
                'polled_n_distinct_equals_w183b_table': (on0['n_distinct_feasible_poll_points']
                                                         == by_k[on0['poll_index']]['n_distinct']),
                'polled_trigger_equals_w183b_table': (on0['completion_triggered']
                                                      == by_k[on0['poll_index']]['completion_triggered']),
                'skipped_sizes_equal_w183b_table': all(
                    s['n_distinct_feasible_poll_points'] == by_k[s['k']]['n_distinct'] and by_k[s['k']]['over_cap']
                    and s['completion']['n_feasible'] == trow['completion_n_feasible'] for s in skipped),
                'halton_t_equals_17_plus_k': on0['halton_t'] == PB.HALTON_T0 + on0['poll_index'] == 17 + exp['k'],
                'on_vs_imported': rec['on_vs_imported'], 'on_summary': rec['summary']['on']}
            item2[case]['ok'] = all(item2[case][k] for k in (
                'k_ok', 'size_ok', 'skipped_ok', 'completion_use_ok', 'polled_n_distinct_equals_w183b_table',
                'polled_trigger_equals_w183b_table', 'skipped_sizes_equal_w183b_table',
                'halton_t_equals_17_plus_k')) and rec['on_vs_imported']['ok']
            _log(f'  {case}: B+ first poll k = {on0["poll_index"]} size {len(polled)} skipped '
                 f'{[s["k"] for s in skipped]} (expected k {exp["k"]} size {exp["size"]} skipped {exp["skipped"]})')
    res['item1_advance_off_equals_imported_8_cases'] = item1
    res['item2_advance_on_8_cases'] = item2
    checks['item1'] = {f'{c}:{k}': v for c, r in item1.items() for k, v in r.items() if isinstance(v, bool)}
    checks['item2'] = {f'{c}:ok': r['ok'] for c, r in item2.items()}
    # the k = 0 ... 9 table, reproduced by the unchanged first_poll_by_seed
    reproduced = L.first_poll_by_seed(lattice, sigma_q)
    res['item2_table_reproduced'] = {'equal_to_w183b_committed_table': json.loads(_ser(reproduced)) == table,
                                     'bytes_equal_sorted': _ser(reproduced) == _ser(table)}
    checks['item2']['w183b_k_0_9_table_reproduced'] = res['item2_table_reproduced']['equal_to_w183b_committed_table']

    # ---- 3. the paths the poll-0 gate does not reach ----
    key_a = L.make_arm_key_of(lattice, 'A')
    z0, z_unit = seeds['x0'], seeds['unit']
    std = {'k0': L.HALTON_K0, 'max_new_evaluations': L.N_MAX_NEW_EVALUATIONS, 'max_polls': L.MAX_POLLS,
           'completion_cap': L.COMPLETION_CAP}
    item3 = {}

    def _scenario(name, stub, z_inc, kw, window=L.ADVANCE_WINDOW, cache_points=None):
        inc, cache = incumbent_and_cache(lattice, key_a, stub, z_inc, cache_points or [z_inc])
        rec, runs = paired(name, lattice, key_a, stub, inc, cache, sigma_q, kw, window)
        runs_out[f'item3_{name}'] = runs
        item3[name] = rec
        return rec, runs

    # (a) A7 per poll: x = 0, every new point a barrier, batch size 1 -> stop after the 2nd barrier
    all_labels = {lattice.label(z) for z in lattice.domain()}
    rec, runs = _scenario('A7_per_poll', LabelStub(lattice, key_a, lambda lab: Q0, barrier=all_labels - {'x0'},
                                                   name='all_barrier'), z0, dict(std, batch_size=1))
    t = runs['on']['out']['termination']
    rec['target_path'] = {'imported_and_on_stop_barrier_rule': all(
        runs[x]['out']['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule' for x in ('imported', 'on')),
        'barrier_new_this_poll_2': t.get('barrier_new_this_poll') == 2, 'n_new_2': runs['on']['out'][
            'n_new_evaluations'] == 2}
    # (b) A7 overall: x0 -> p1 -> p2, one barrier per poll (batch size 1) -> stop at the 3rd barrier overall
    p1 = next(z for z in poll_points(lattice, key_a, z0, 0, sigma_q)[1]
              if lattice.label(z) == 'y2030__n7_p0.25_e0.5__n9_p0.25_e0.5')
    pts0 = poll_points(lattice, key_a, z0, 0, sigma_q)[1]
    pts1 = poll_points(lattice, key_a, p1, 1, sigma_q)[1]
    p2 = next(z for z in pts1 if z not in pts0 and z != z0
              and (poll_points(lattice, key_a, z, 2, sigma_q)[1] or []) and
              poll_points(lattice, key_a, z, 2, sigma_q)[0]['completion_triggered'] is False)
    pts2 = poll_points(lattice, key_a, p2, 2, sigma_q)[1]
    b0 = next(z for z in pts0 if z not in (p1,))
    b1 = next(z for z in pts1 if z not in pts0 and z not in (p2, z0))
    b2 = next(z for z in pts2 if z not in pts0 and z not in pts1 and z not in (z0, p1))
    q_b = {lattice.label(p1): Q0 - 3.0e6, lattice.label(p2): Q0 - 6.0e6}
    rec, runs = _scenario('A7_overall', LabelStub(lattice, key_a, lambda lab: q_b.get(lab, Q0 + 1.0e6),
                                                  barrier={lattice.label(b) for b in (b0, b1, b2)},
                                                  name='one_barrier_per_poll'), z0, dict(std, batch_size=1))
    t = runs['on']['out']['termination']
    rec['design'] = {'p1': lattice.label(p1), 'p2': lattice.label(p2), 'barriers': [lattice.label(b) for b in
                                                                                     (b0, b1, b2)]}
    rec['target_path'] = {'imported_and_on_stop_barrier_rule': all(
        runs[x]['out']['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule' for x in ('imported', 'on')),
        'barrier_new_overall_3': t.get('barrier_new_overall') == 3,
        'barrier_new_this_poll_below_2': (t.get('barrier_new_this_poll') or 0) < 2,
        'three_polls': runs['on']['out']['n_polls'] == 3}
    # (c) an accepted point moving the incumbent to a storage point whose frozen next poll the cap refuses
    p_st = next(z for z in pts0 if lattice.label(z) == 'y2030__n9_p0.25_e0.5')
    rec_k1, pts_k1 = poll_points(lattice, key_a, p_st, 1, sigma_q)
    rec_k2, pts_k2 = poll_points(lattice, key_a, p_st, 2, sigma_q)
    p_next = next(z for z in pts_k2 if z != z0 and z not in pts0 and lattice.has_storage(z))
    q_c = {lattice.label(p_st): Q0 - 3.0e6, lattice.label(p_next): Q0 - 6.0e6}
    rec, runs = _scenario('accepted_point_moves_incumbent_to_storage',
                          LabelStub(lattice, key_a, lambda lab: q_c.get(lab, Q0 + 1.0e6), name='move_to_storage'),
                          z0, dict(std, batch_size=4))
    hon = runs['on']['out']['history']
    rec['design'] = {'p1_storage': lattice.label(p_st), 'p1_at_k1': {'n_distinct': rec_k1[
        'n_distinct_feasible_poll_points'], 'over_cap': pts_k1 is None}, 'p2': lattice.label(p_next)}
    rec['target_path'] = {
        'imported_stops_on_the_cap_at_poll_1_k_1': (runs['imported']['out']['termination']['reason']
                                                    == 'STOP_FOR_REVIEW_completion_cap'
                                                    and runs['imported']['out']['history'][-1]['poll_index'] == 1),
        'on_poll_1_examines_k_1_first': len(hon) > 1 and hon[1]['advance_rule']['first_index'] == 1,
        'on_poll_1_advances': len(hon) > 1 and hon[1]['advance_rule']['n_examined'] > 1,
        'on_moves_on_and_next_poll_starts_at_last_polled_plus_1': len(hon) > 2 and hon[2]['advance_rule'][
            'first_index'] == hon[1]['poll_index'] + 1}
    # (d) budget exhaustion
    rec, runs = _scenario('budget_zero_x0', LabelStub(lattice, key_a, q_hash, name='q_hash'), z0,
                          dict(std, batch_size=1, max_new_evaluations=0))
    rec['target_path'] = {'all_three_budget_refusal_at_poll_0': all(
        runs[x]['out']['termination']['reason'] == 'evaluation_budget_exhausted'
        and runs[x]['out']['n_polls'] == 1 for x in ('imported', 'off', 'on'))}
    rec, runs = _scenario('budget_5_unit', LabelStub(lattice, key_a, q_hash, name='q_hash'), z_unit,
                          dict(std, batch_size=1, max_new_evaluations=5))
    h0 = runs['on']['out']['history'][0]
    rec['target_path'] = {
        'imported_refuses_on_the_cap_at_k_0': runs['imported']['out']['termination']['reason']
        == 'STOP_FOR_REVIEW_completion_cap',
        'on_polled_at_k_2_then_budget_refusal': (h0['poll_index'] == 2
                                                 and h0['decision'] == 'not_launched_evaluation_budget'
                                                 and runs['on']['out']['termination']['reason']
                                                 == 'evaluation_budget_exhausted'),
        'on_skipped_0_1_recorded': [s['k'] for s in h0['halton_indices_skipped']] == [0, 1],
        'on_nothing_evaluated': runs['on']['calls'] == [] and runs['on']['out']['n_new_evaluations'] == 0}
    # (e) max_polls = 1 with an improving stub
    q_e = {lattice.label(z): Q0 - 5.0e6 for z in pts0[:1]}
    rec, runs = _scenario('max_polls_1_x0', LabelStub(lattice, key_a, lambda lab: q_e.get(lab, Q0 + 1.0e6),
                                                      name='improving_x0'), z0, dict(std, batch_size=4, max_polls=1))
    rec['target_path'] = {'all_three_success_then_max_polls': all(
        runs[x]['out']['termination']['reason'] == 'max_polls_reached'
        and runs[x]['out']['history'][0]['decision'] == 'success' for x in ('imported', 'off', 'on'))}
    unit_k2 = poll_points(lattice, key_a, z_unit, 2, sigma_q)[1]
    q_e2 = {lattice.label(z): Q0 - 5.0e6 for z in unit_k2[:1]}
    rec, runs = _scenario('max_polls_1_unit', LabelStub(lattice, key_a, lambda lab: q_e2.get(lab, Q0 + 1.0e6),
                                                        name='improving_unit'), z_unit,
                          dict(std, batch_size=4, max_polls=1))
    rec['target_path'] = {
        'imported_refuses_on_the_cap': runs['imported']['out']['termination']['reason']
        == 'STOP_FOR_REVIEW_completion_cap',
        'on_success_at_k_2_then_max_polls': (runs['on']['out']['termination']['reason'] == 'max_polls_reached'
                                             and runs['on']['out']['history'][0]['poll_index'] == 2
                                             and runs['on']['out']['history'][0]['decision'] == 'success')}
    # (f) window exhaustion (iii): the domain scan, then the production window on the found incumbent
    _log('domain scan for window exhaustion (pure C.build_poll) ...')
    scan = scan_window_exhaustion(lattice, key_a, sigma_q)
    res['item3_window_exhaustion_scan'] = scan
    _log(f"scan: {scan['n_hits']} incumbents with an exhausted 10-index window in k = 0 ... {SCAN_K_MAX} "
         f"({scan['n_build_poll_calls']} build_poll calls, {scan['wall_s']:.0f} s)")
    if scan['hits']:
        hit = min(scan['hits'], key=lambda h: (h['window_first_k'], h['label']))
        z_hit = tuple(hit['z'])
        rec, runs = _scenario('window_exhaustion_iii_production_window_10',
                              LabelStub(lattice, key_a, q_hash, name='q_hash'), z_hit,
                              dict(std, batch_size=1, k0=hit['window_first_k']))
        rec['design'] = {'incumbent': hit, 'production_window_reached': True}
        h0 = runs['on']['out']['history'][0]
        rec['target_path'] = {
            'all_three_stop_on_the_cap': all(runs[x]['out']['termination']['reason']
                                             == 'STOP_FOR_REVIEW_completion_cap' for x in ('imported', 'off', 'on')),
            'on_n_examined_10_polled_none': h0['advance_rule']['n_examined'] == 10
            and h0['advance_rule']['polled_index'] is None,
            'on_skipped_are_the_9_further_indices': [s['k'] for s in h0['halton_indices_skipped']]
            == list(range(hit['window_first_k'] + 1, hit['window_first_k'] + 10))}
        res['item3_window_exhaustion_production_window_reachable'] = True
    else:
        res['item3_window_exhaustion_production_window_reachable'] = False
    checks['item3'] = {}
    for name, r in item3.items():
        checks['item3'][f'{name}:off_bytes_equal'] = r['off_vs_imported'][
            'return_dict_bytes_equal_after_removing_the_two_fields']
        for k in ('two_fields_trivial_on_every_record', 'stub_calls_equal', 'log_lines_equal', 'on_poll_start_equal',
                  'on_poll_equal'):
            checks['item3'][f'{name}:off_{k}'] = r['off_vs_imported'][k]
        checks['item3'][f'{name}:on_vs_imported_ok'] = r['on_vs_imported']['ok']
        for k, v in (r.get('target_path') or {}).items():
            checks['item3'][f'{name}:target:{k}'] = v
    checks['item3']['window_exhaustion_found_at_production_window'] = res[
        'item3_window_exhaustion_production_window_reachable']
    res['item3_paths'] = item3

    # ---- 4. the source diff ----
    diff = source_diff()
    res['item4_source_diff'] = diff
    checks['item4'] = {'only_signature_k_next_state_and_inserted_block': diff['only_declared_categories'],
                       'loop_body_after_the_block_identical': diff['body_after_block_identical'],
                       'docstring_identical': diff['docstring_identical']}
    _log(f"source diff: {diff['counts_by_side_and_category']}; uncategorised {len(diff['uncategorised_lines'])}")

    # ---- 5. the v1 dry-run assertions on the v2 launcher, the final-head G0, the declared roots ----
    sub_abs = os.path.join(out_abs, LAUNCHER_DRY_RUN_SUBDIR)
    os.makedirs(sub_abs)
    sub_rel, sub_abs = L.resolve_root(sub_abs)
    dargs = SimpleNamespace(concurrency=1, g0_artifact=args.g0_artifact, g0_sha256=args.g0_sha256,
                            scratch=args.scratch, allow_uncommitted=False)
    code = L.dry_run(dargs, sub_rel, sub_abs)
    dr = json.load(open(os.path.join(sub_abs, L.DRY_RUN_FILE)))
    cells = {(c['row'], c['cell'][:12]): c for c in dr['committed_cell_comparison']['cells']}
    plan = L.seed_plan(lattice)
    x0_keys = {m: next(r['eval_key'] for r in plan if r['seed'] == 'x0' and r['model'] == m) for m in L.MODELS}
    g0_ok, g0_info = L.g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=True)
    labels7 = sorted(r['label'] for r in plan if r['disposition'] == 'evaluate')
    roots = {
        'batch_seeds_are_seed_labels_of_the_plan': set(L.BATCH_SEEDS) <= set(labels7),
        'batch_root_accepts_the_four_at_4': not L.declared_root_refusals(L.E1_BATCH_ROOT_REL, 'seeds',
                                                                         list(L.BATCH_SEEDS), 4, labels7),
        'batch_root_refuses_concurrency_1': bool(L.declared_root_refusals(L.E1_BATCH_ROOT_REL, 'seeds',
                                                                          list(L.BATCH_SEEDS), 1, labels7)),
        'batch_root_refuses_all_seven': bool(L.declared_root_refusals(L.E1_BATCH_ROOT_REL, 'seeds', labels7, 4,
                                                                      labels7)),
        'batch_root_refuses_search_phases': all(L.declared_root_refusals(L.E1_BATCH_ROOT_REL, p, None, 4, labels7)
                                                for p in ('search_N', 'search_A')),
        'e1_root_accepts_all_seven': not L.declared_root_refusals(L.E1_ROOT_REL, 'seeds', labels7, 1, labels7),
        'e1_root_refuses_a_subset': bool(L.declared_root_refusals(L.E1_ROOT_REL, 'seeds', list(L.BATCH_SEEDS), 4,
                                                                  labels7)),
        'e1_root_accepts_search_phases': all(not L.declared_root_refusals(L.E1_ROOT_REL, p, None, 1, labels7)
                                             for p in ('search_N', 'search_A')),
        'undeclared_root_refused': bool(L.declared_root_refusals(sub_rel, 'seeds', labels7, 1, labels7)),
        'e1_root_declares_cross_phase': 'cross' in L.DECLARED_ROOTS[L.E1_ROOT_REL]['phases'],
        'would_be_spec_records_declared_roots_and_b_plus_rule': (
            dr['would_be_seeds_spec']['content']['extra'].get('declared_roots') == json.loads(
                _ser(L.DECLARED_ROOTS))
            and dr['would_be_seeds_spec']['content']['extra']['poll_design'].get('b_plus_rule') == L.B_PLUS_RULE)}
    res['item5'] = {'launcher_dry_run_exit_code': code, 'launcher_dry_run_root': sub_rel,
                    'launcher_dry_run_file': os.path.join(sub_rel, L.DRY_RUN_FILE),
                    'launcher_dry_run_sha256': _sha(os.path.join(sub_rel, L.DRY_RUN_FILE)),
                    'launcher_dry_run_failing_assertions': dr['failing_assertions'],
                    'seed_key_summary': dr['seed_key_summary'], 'prediction': dr['prediction'],
                    'alias': dr['alias'], 'g0_require_current_production': {'accepted': g0_ok, **g0_info},
                    'committed_cells': {f'{k[0]}:{k[1]}': {kk: v.get(kk) for kk in (
                        'committed_base_key', 'launcher_key_2x', 'base_key_equals_launcher_key',
                        'differing_key_inputs')} for k, v in cells.items()},
                    'declared_roots': roots}
    cell35 = next(c for c in dr['committed_cell_comparison']['cells'] if c['row'] == 35)
    cell40 = next(c for c in dr['committed_cell_comparison']['cells'] if c['row'] == 40
                  and c['cell'].startswith('ref:5ca4f86c (issuing run'))
    checks['item5'] = {
        'launcher_dry_run_exit_0_all_assertions_hold': code == 0 and dr['all_assertions_hold'] is True,
        'seven_keys_held': dr['prediction']['held'] is True and dr['seed_key_summary'][
            'n_distinct_keys_to_evaluate'] == 7,
        'one_alias_g0_final_head': (dr['seed_key_summary']['n_aliases'] == 1 and dr['alias']['g0_accepted'] is True
                                    and dr['alias']['g0_artifact_sha256'] == G0_FINAL['sha256']
                                    and dr['g0']['artifact_production']['equals_current_production'] is True),
        'g0_accepted_with_require_current_production': g0_ok,
        'base_key_equal_row_35': cell35.get('base_key_equals_launcher_key') is True,
        'base_key_equal_row_40': cell40.get('base_key_equals_launcher_key') is True,
        'root_scoped_cache': all(dr['assertions']['cache_scope'].values()),
        'pins': all(dr['assertions']['pins_recorded'].values()),
        'preflight_recorded': all(dr['assertions']['concurrency_and_preflight'].values()),
        'rule_eleven_e1_loop_checks': all(v for k, v in dr['rule_eleven']['checks'].items()
                                          if k.startswith('e1_loop')) and sum(
            k.startswith('e1_loop') for k in dr['rule_eleven']['checks']) == 8,
        'first_poll_table_in_dry_run_equals_w183b': dr['first_poll_if_each_seed_were_the_incumbent'] == table,
        **{f'roots:{k}': v for k, v in roots.items()}}

    # ---- guards, outputs ----
    g = L.guards_report()
    gate_failures = GATE_GUARD.verify(0)
    res['guards'] = dict(g, gate_guard={'counts': dict(GATE_GUARD.counts), 'verify_0_failures': gate_failures},
                         n_guards_verified=len(L.all_guards()))
    checks['guards'] = {'all_zero': g['all_zero'] and not gate_failures,
                        'gate_guard_among_verified': any(gd is GATE_GUARD for _n, gd in L.all_guards())}
    res['checks'] = checks
    failing = sorted(f'{grp}.{k}' for grp, cs in checks.items() for k, v in cs.items() if v is not True)
    res['failing_checks'] = failing
    res['GATE_PASS'] = not failing
    res['finished_utc'] = L._utc()
    res['wall_clock_s'] = time.time() - started
    runs_payload = {name: {tag: {'out': r['out'], 'calls': r['calls'], 'logs': r['logs'],
                                 'on_poll_start': r['on_poll_start'], 'on_poll': r['on_poll']}
                           for tag, r in runs.items()} for name, runs in runs_out.items()}
    runs_path = os.path.join(out_abs, OUT_RUNS_FILE)
    with open(runs_path, 'xb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as gz:
            gz.write(GRIO.dumps(runs_payload, sort_keys=True).encode())
    res['loop_outputs_file'] = {'path': os.path.join(out_rel, OUT_RUNS_FILE), 'sha256': _sha(runs_path),
                                'content': 'every paired run: full return dict, stub calls, log lines, callbacks'}
    out_path = os.path.join(out_abs, OUT_FILE)
    sha = L._write_once_json(out_path, res)
    manifest = {os.path.join(out_rel, OUT_FILE): sha, os.path.join(out_rel, OUT_RUNS_FILE): _sha(runs_path),
                os.path.join(sub_rel, L.DRY_RUN_FILE): _sha(os.path.join(sub_rel, L.DRY_RUN_FILE)),
                os.path.join(sub_rel, L.DRY_RUN_MANIFEST): _sha(os.path.join(sub_rel, L.DRY_RUN_MANIFEST)),
                L.SCRIPT_NAME: _sha(L.SCRIPT_NAME), SCRIPT_NAME: _sha(SCRIPT_NAME),
                W183B_DRYRUN['path']: _sha(W183B_DRYRUN['path']), G0_FINAL['path']: _sha(G0_FINAL['path']),
                **{m: _sha(m) for m in UNCHANGED_MODULES}}
    L._write_once_json(os.path.join(out_abs, OUT_MANIFEST), manifest)
    _log(f'failing checks: {failing}')
    _log(f"GATE_PASS={res['GATE_PASS']}; wrote {os.path.join(out_rel, OUT_FILE)} sha256 {sha}")
    _log(f"guards: all_zero={g['all_zero']} gate_guard={GATE_GUARD.counts} pickle={g['pickle']['counts']} "
         f"build={g['model_build_tripwire']}")
    for _n, guard in reversed(L.all_guards()):
        guard.uninstall()
    return 0 if res['GATE_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
