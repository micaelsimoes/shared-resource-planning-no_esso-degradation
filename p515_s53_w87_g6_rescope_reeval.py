"""
P5.15 Addendum 46 ruling 7, Planner task W87 -- RE-SCOPE gate G6 of the tight-tail SRP1 re-certification and
RE-EVALUATE the three cells from their PERSISTED records. ZERO SOLVES. Frozen stage spec v30 (predecessor v29
9161ff00, NOT edited).

WHY. The W86 re-certification (campaign s53_w86_tail_recert, spec ddd6cd44) certified all three cells at their
reference cycles, but its per-entry gate G6_floor_records judged EVERY network IPOPT record of the run -- including
the pre-tail rounds solved at PRODUCTION compl_inf_tol, which the tail neither touches nor is responsible for. On C*
it failed on 3 pre-tail records (tier-2 recoveries passing mu_strategy adaptive, so the monotone mu-floor formula is
declared not applicable: parse_reason set), and because v29's fallback test counts a failed per-entry gate as "fails
to certify", C* tripped the fallback. That is the repository's "scope a gate per arm" failure (CLAUDE.md stage
templates; P5.14-N precedent): a gate judging a population it was not designed for.

WHAT CHANGES (v30): ONLY the POPULATION G6 judges. The per-record predicate is v29's, unchanged. The pre-tail records
are reported as EVIDENCE, never gated. G1-G5, G7-G9 and the fallback test (including the "moves materially" test) are
v29's verbatim.

HOW. Everything is re-evaluated from files: the launcher's own gate / comparison functions are used BY IMPORT
(`p515_s53_w86_tail_recert_campaign.evaluation_checks` and `.compare`), so every unchanged gate is computed by the same
code that produced the W86 verdict, and the W86 values are REPRODUCED as a check before anything new is read.
`SolveProfileGuard(permitted=())` is armed at import, BEFORE any project import, and verified at exactly 0 on every
exit path (together with the imported launcher's own permitted=() guard).

MODES (repo root, canonical interpreter; attached, alone, both streams captured):
  --freeze-spec                   writes data/SRP1/Results/P515S53/frozen_s53_spec_v30_<sha8>.json (write-once, named
                                  by its sha256): the re-scoped G6, the reason, the evidence base pinned, the pre-tail
                                  evidence definition, the unchanged gates and fallback test, predictions.
  --run --spec-sha256 S           the re-evaluation -> data/SRP1/Results/P515S53/g6_rescope_w87/reeval_w87.json +
                                  manifest (a NEW root; nothing committed is re-run onto or modified).
Exit codes: 0 done (whatever the verdict), 1 a precondition / integrity / guard failure.

EXACT COMMANDS:
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w87_g6_rescope_reeval.py \\
      --freeze-spec > data/SRP1/Results/P515S53/g6_rescope_w87_freeze_spec_v30_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w87_g6_rescope_reeval.py \\
      --run --spec-sha256 <sha> > data/SRP1/Results/P515S53/g6_rescope_w87_launch.log 2>&1
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W87 G6 re-scope re-evaluation (never solves)').install()

import p515_s53_w86_tail_recert_campaign as L  # noqa: E402 -- arms its own PARENT_GUARD permitted=() at import
import p515_s44_campaign_harness as H  # noqa: E402

SCRIPT_NAME = os.path.basename(__file__)
STAGE_TEXT = ('P5.15 Addendum 46 ruling 7, W87 -- G6 re-scoped to the tail window and terminal round; the W86 '
              'tight-tail re-certification re-evaluated from its persisted records (zero solves)')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
SPEC_V29 = {'path': os.path.join(_P53, 'frozen_s53_spec_v29_9161ff00.json'),
            'sha256': '9161ff00273e2e1907aa75121ac7454d9f33ec39f91c82f93513efae78dfcb98'}
SPEC_V30_PREFIX = 'frozen_s53_spec_v30_'
RECERT_ROOT = os.path.join(_P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert')
RECERT_SPEC = {'path': os.path.join(RECERT_ROOT, 'campaign_spec_s53_w86_tail_recert_ddd6cd44.json'),
               'sha256': 'ddd6cd4422b56f946eeeb427dff44ef572490bd34c606fad96eaf00fd4a84ec1'}
RECERT_RESULTS = os.path.join(RECERT_ROOT, 'campaign_results.json')
RECERT_MANIFEST = os.path.join(RECERT_ROOT, 'campaign_manifest_sha256.json')
OUT_ROOT = os.path.join(_P53, 'g6_rescope_w87')
OUT_FILE = 'reeval_w87.json'
OUT_MANIFEST = 'reeval_w87_manifest_sha256.json'
LABELS = ('c_star', 'n7_4h_e1', 'x0')
BLOCKS_PER_ROUND = 48          # (1 TSO + 3 DSO) x 3 years x 4 days: one primary attempt per network block per round
TAIL_TOL = 1e-6
# The files the re-evaluation reads per cell (all committed in 0a4bf784 and listed in the campaign manifest).
CELL_FILES = ('evaluation_record.json', 'exit_code.txt', 'per_cycle_record.jsonl', 'network_ipopt_solve_records.jsonl',
              'network_ipopt_solve_records_append.jsonl', 'convergence_depth_append_events.jsonl',
              'convergence_depth_tail_state.json', 'network_failures_s39_D.jsonl', 'post_certification.json')
CERTIFIED_MODELS = 'certified_models.pkl'   # NOT committed; hash-recorded in the campaign manifest (G8 reads it)
REF_FILES = ('evaluation_record.json', 'per_cycle_record.jsonl', 'network_failures_s39_D.jsonl')

# ----------------------------------------------------------------------------------------------------------------------
#  the re-scoped gate (v30) -- stated operationally
# ----------------------------------------------------------------------------------------------------------------------
G6_RESCOPED = {
    'name': 'G6_floor_records_tail_window',
    'replaces': 'v29 per_entry_gates.G6_floor_records (same predicate, population re-scoped)',
    'population': ('P(cell) = { r in <eval_dir>/network_ipopt_solve_records.jsonl : r.round in W u {T} }, where '
                   'W = { p.cycle : p in convergence_depth_tail_state.json per_cycle, p.active is True } (the cycles '
                   'the tail was actually active, per the persisted tail state) and T = evaluation_record.cycles_run '
                   '(the terminal round). Round index == ADMM cycle (round 0 = the initialisation round). Every '
                   'attempt in those rounds is included: primary, tier-1 recovery, tier-2 recovery.'),
    'predicate_per_record_unchanged_from_v29': ('parse_reason is None AND options_list_agrees is True AND '
                                                'compl_inf_tol_in_force == 1e-6 if r.round in W, else == the '
                                                'network\'s production value (TSO case9 5e-4 passed; DSO key absent '
                                                '-> IPOPT default 1e-4), from the persisted tail baseline'),
    'non_vacuity': ('P non-empty; T in W u {T} trivially, and every round of W u {T} holds exactly 48 primary-attempt '
                    'records (one per network block) and >= 48 records in all -- too few fails as loudly as a bad '
                    'record'),
    'passes_iff': 'non_vacuity holds AND every record of P satisfies the predicate',
    'excluded_population': ('records in rounds NOT in W u {T}: the pre-tail rounds, solved at PRODUCTION '
                            'compl_inf_tol, which the tail does not touch. They are REPORTED as evidence (pre_tail_'
                            'evidence below), never gated.'),
    'not_judged': ('floor_status (at / above / below) was NOT part of v29 G6 and is not part of the re-scoped G6: the '
                   'predicate is unchanged. floor_status is REPORTED for the window, the terminal round and the '
                   'pre-tail rounds.'),
}
REASON = (
    'v29 G6 judged every record of the run. On C* it failed on exactly 3 records, all pre-tail (rounds 11, 13, 23, at '
    'production compl_inf_tol 1e-4): tier-2 recovery attempts, which pass mu_strategy adaptive, for which '
    'network.parse_ipopt_attempt_segment DECLARES the monotone mu-floor formula not applicable (parse_reason set, '
    'mu_floor / floor_status None) -- by design, not a log-read failure. Those solves precede the tail\'s first '
    'active cycle (79) and are a property of the instance (the reference had the same 35 pre-tail failure events, '
    'identical block for block, and the same 40 retries in all), not of the tail; judging them made the fallback fire on a gate designed for something else ("scope a gate per arm", '
    'CLAUDE.md; P5.14-N precedent). The re-scope changes the POPULATION only; no threshold is raised and no predicate '
    'is relaxed.')
PRE_TAIL_EVIDENCE = {
    'status': 'REPORTED, NOT GATED -- a property of the instance, not of the tail',
    'population': 'records with round NOT in W u {T} (so rounds 0 .. min(W) - 1 for a run whose tail stayed active '
                  'through T)',
    'reported': ['n records', 'attempt tally', 'exit tally', 'floor_status tally by agent (at / above / below / None)',
                 'n above', 'n unparsable (parse_reason not None) with each record in full',
                 'n records failing the unchanged G6 predicate'],
}
REPORTED_NOT_GATED = {
    'window_and_terminal_floor': 'floor_status / exit / attempt tallies in W and in {T}; every non-primary or '
                                 'non-at record of W listed',
    'g6_v29_recomputed': ('v29 G6 recomputed over ALL records (the original population): must REPRODUCE the W86 '
                          'verdict (integrity check), then reported, not gated'),
    'retry_comparison': ('per cell, network_failures_s39_D.jsonl of the run vs the reference: event counts, retries '
                         'credited (recovery_attempted + tier2_attempted), events before min(W) identical as keyed '
                         'tuples, the window events of each run; per_cycle gross identical up to min(W) - 1'),
    'bar_window': 'the cycle holding the max step of each run\'s bar window (explains bar_tail vs bar_ref)',
}
SCORING = {
    'author': ('holds on a cell iff |dQ| <= 1e-6 |Q_ref| (author_band_ratio <= 1): "unchanged or within 1e-6 '
               'relative"'),
    'worker_W1_w83': ('dQ ~ -3.4e3 EUR (~ -5e-6 relative): reported as dQ / (-3.4e3) and dQ_relative / (-5e-6); '
                      'sign holds iff dQ < 0; magnitude CONSISTENT iff 1.7e3 <= |dQ| <= 6.8e3 (a factor of 2 either '
                      'way -- a tolerance declared HERE, v29 declared none)'),
    'worker_W2_to_W7': 'scored as worded in v29 predictions_recorded_before_the_run.worker',
    'v28_P7_P8_P10': ('P7: >= 1 tail-window solve exits "Solved To Acceptable Level." in a cell; P8: median '
                      'iterations of window PRIMARY solves vs v28\'s reference medians (TSO 30, DSO 44 -- stated for '
                      'C* cycles 79-87 only; not scorable on the other cells); P10: window retries <= 2 x the '
                      'reference\'s window retries'),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _jsonl(rel):
    with open(_abs(rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_state(rel):
    return {'git_tracked': bool(H._git(['ls-files', '--', rel]).strip()),
            'git_clean': not bool(H._git(['status', '--porcelain', '--', rel]).strip())}


def guards_verify():
    return {'w87_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)},
            'imported_launcher_guard': {'counts': dict(L.PARENT_GUARD.counts),
                                        'verify_0_failures': L.PARENT_GUARD.verify(0)}}


def _guards_ok(g):
    return not g['w87_guard']['verify_0_failures'] and not g['imported_launcher_guard']['verify_0_failures']


def _finish(code, extra_msg=''):
    """EVERY exit path: verify both guards at exactly 0, uninstall, exit (1 if a guard fails)."""
    g = guards_verify()
    _log(f'[W87] guards {g} {extra_msg}')
    L.PARENT_GUARD.uninstall()
    GUARD.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _cells_from_recert_spec():
    spec = _load(RECERT_SPEC['path'])
    return {e['label']: e for e in spec['candidates']}


def evidence_base():
    """The committed files the re-evaluation reads, pinned by sha256 and cross-checked against the campaign manifest."""
    manifest = _load(RECERT_MANIFEST)
    entries = _cells_from_recert_spec()
    v29 = _load(SPEC_V29['path'])
    out = {'campaign_spec': {**RECERT_SPEC, **_git_state(RECERT_SPEC['path'])},
           'campaign_results': {'path': RECERT_RESULTS, 'sha256': _sha(RECERT_RESULTS), **_git_state(RECERT_RESULTS)},
           'campaign_manifest': {'path': RECERT_MANIFEST, 'sha256': _sha(RECERT_MANIFEST),
                                 'n_entries': len(manifest), **_git_state(RECERT_MANIFEST)},
           'spec_v29': {**SPEC_V29, **_git_state(SPEC_V29['path'])}, 'cells': {}, 'references': {}}
    failures = []
    if _sha(RECERT_SPEC['path']) != RECERT_SPEC['sha256']:
        failures.append('campaign spec sha256 differs from ddd6cd44')
    if _sha(SPEC_V29['path']) != SPEC_V29['sha256']:
        failures.append('v29 sha256 differs from 9161ff00')
    for label in LABELS:
        d = os.path.join(RECERT_ROOT, 'evals', entries[label]['eval_dir'])
        files = {}
        for f in CELL_FILES:
            rel = os.path.join(d, f)
            files[f] = {'sha256': _sha(rel), 'manifest_sha256': manifest.get(rel), **_git_state(rel)}
            if files[f]['sha256'] != files[f]['manifest_sha256']:
                failures.append(f'{rel}: sha256 != campaign manifest')
        pkl = os.path.join(d, CERTIFIED_MODELS)
        files[CERTIFIED_MODELS] = {'sha256': _sha(pkl) if os.path.isfile(_abs(pkl)) else None,
                                   'manifest_sha256': manifest.get(pkl), 'git_tracked': False,
                                   'note': 'not committed (165 MB); hash-recorded in the campaign manifest'}
        if files[CERTIFIED_MODELS]['sha256'] != files[CERTIFIED_MODELS]['manifest_sha256']:
            failures.append(f'{pkl}: sha256 != campaign manifest')
        out['cells'][label] = {'eval_dir': d, 'eval_key': entries[label]['eval_key'],
                               'candidate_key': entries[label]['key'], 'files': files}
        ref_dir = v29['references'][label]['eval_dir']
        out['references'][label] = {'eval_dir': ref_dir, 'eval_key': v29['references'][label]['eval_key'],
                                    'files': {f: {'sha256': _sha(os.path.join(ref_dir, f)),
                                                  **_git_state(os.path.join(ref_dir, f))} for f in REF_FILES}}
        for f in ('evaluation_record.json', 'per_cycle_record.jsonl'):
            if out['references'][label]['files'][f]['sha256'] != v29['references'][label]['files_sha256'][f]:
                failures.append(f'reference {label} {f}: sha256 != v29 pin')
    for name, item in [('campaign_spec', out['campaign_spec']), ('campaign_results', out['campaign_results']),
                       ('campaign_manifest', out['campaign_manifest']), ('spec_v29', out['spec_v29'])] + [
            (f'{lab}:{f}', v) for lab, c in out['cells'].items() for f, v in c['files'].items() if f != CERTIFIED_MODELS
    ] + [(f'ref {lab}:{f}', v) for lab, c in out['references'].items() for f, v in c['files'].items()]:
        if not (item.get('git_tracked') and item.get('git_clean')):
            failures.append(f'{name} not committed / not clean: {item}')
    return out, failures


# ----------------------------------------------------------------------------------------------------------------------
#  the re-evaluation (files only)
# ----------------------------------------------------------------------------------------------------------------------
def _predicate_failure(r, window, prod):
    want = TAIL_TOL if r.get('round') in window else prod.get(r.get('network'))
    ok = (r.get('compl_inf_tol_in_force') == want and r.get('parse_reason') is None
          and r.get('options_list_agrees') is True)
    return None if ok else {k: r.get(k) for k in ('network', 'year', 'day', 'round', 'attempt',
                                                  'compl_inf_tol_in_force', 'parse_reason', 'options_list_agrees')}


def _tallies(rs):
    return {'n': len(rs),
            'attempt': dict(sorted(Counter(r.get('attempt') for r in rs).items())),
            'exit': dict(sorted(Counter(str(r.get('exit')) for r in rs).items())),
            'floor_status_by_agent': dict(sorted(Counter(
                f"{'TSO' if r.get('agent') == 'TSO' else 'DSO'}|{r.get('floor_status')}" for r in rs).items())),
            'n_above': sum(1 for r in rs if r.get('floor_status') == 'above'),
            'n_unparsable': sum(1 for r in rs if r.get('parse_reason') is not None),
            'n_max_iter': sum(1 for r in rs if r.get('exit') == 'Maximum Number of Iterations Exceeded.'),
            'n_acceptable': sum(1 for r in rs if r.get('exit') == 'Solved To Acceptable Level.')}


def _brief(r):
    return {k: r.get(k) for k in ('round', 'network', 'agent', 'year', 'day', 'attempt', 'warm_start',
                                  'compl_inf_tol_in_force', 'mu_strategy_passed', 'mu_final', 'mu_floor',
                                  'mu_over_floor', 'floor_status', 'iterations', 'exit', 'parse_reason',
                                  'log_path', 'log_bytes')}


def g6_evaluate(eval_dir, rec):
    records = _jsonl(os.path.join(eval_dir, 'network_ipopt_solve_records.jsonl'))
    ts = _load(os.path.join(eval_dir, 'convergence_depth_tail_state.json'))
    window = {p['cycle'] for p in (ts.get('per_cycle') or []) if p.get('active')}
    terminal = rec.get('cycles_run')
    judged_rounds = window | {terminal}
    prod = L._production_compl_inf_tol(ts.get('baseline'))
    pop = [r for r in records if r.get('round') in judged_rounds]
    pre = [r for r in records if r.get('round') not in judged_rounds]
    per_round = {k: {'n': sum(1 for r in pop if r.get('round') == k),
                     'n_primary': sum(1 for r in pop if r.get('round') == k and r.get('attempt') == 'primary')}
                 for k in sorted(judged_rounds)}
    non_vacuous = bool(pop) and all(v['n_primary'] == BLOCKS_PER_ROUND and v['n'] >= BLOCKS_PER_ROUND
                                    for v in per_round.values())
    bad = [b for b in (_predicate_failure(r, window, prod) for r in pop) if b]
    bad_all = [b for b in (_predicate_failure(r, window, prod) for r in records) if b]
    pre_bad = [b for b in (_predicate_failure(r, window, prod) for r in pre) if b]
    first_window = min(window) if window else None
    window_contiguous_to_terminal = bool(window) and sorted(window) == list(range(first_window, terminal + 1))
    return {
        'gate_pass': non_vacuous and not bad,
        'window_W': sorted(window), 'terminal_round_T': terminal, 'judged_rounds': sorted(judged_rounds),
        'window_contiguous_through_T': window_contiguous_to_terminal,
        'production_compl_inf_tol': prod, 'n_records_total': len(records),
        'population_n': len(pop), 'per_round_counts': per_round, 'non_vacuous': non_vacuous,
        'n_bad': len(bad), 'bad': bad,
        'reported_window': {**_tallies([r for r in pop if r.get('round') in window]),
                            'irregular_records': [_brief(r) for r in pop if r.get('round') in window and (
                                r.get('attempt') != 'primary' or r.get('floor_status') != 'at'
                                or r.get('exit') != 'Optimal Solution Found.')],
                            'median_iterations_primary': {
                                agent: (statistics.median([r['iterations'] for r in pop if r.get('round') in window
                                                           and r.get('attempt') == 'primary'
                                                           and (r.get('agent') == 'TSO') == (agent == 'TSO')])
                                        if window else None) for agent in ('TSO', 'DSO')}},
        'reported_terminal_round': {**_tallies([r for r in records if r.get('round') == terminal]),
                                    'all_at_floor_at_1e-6': all(
                                        r.get('floor_status') == 'at' and r.get('compl_inf_tol_in_force') == TAIL_TOL
                                        for r in records if r.get('round') == terminal)},
        'pre_tail_evidence': {**_tallies(pre), 'rounds': [min((r['round'] for r in pre), default=None),
                                                          max((r['round'] for r in pre), default=None)],
                              'n_failing_unchanged_predicate': len(pre_bad),
                              'unparsable_records': [
                                  {**_brief(r), 'in_window': r.get('round') in window,
                                   'block_chain': [_brief(x) for x in records if x.get('round') == r.get('round')
                                                   and (x.get('network'), x.get('year'), x.get('day'))
                                                   == (r.get('network'), r.get('year'), r.get('day'))]}
                                  for r in pre if r.get('parse_reason') is not None],
                              'max_iter_rounds': sorted({r['round'] for r in pre if r.get('exit')
                                                         == 'Maximum Number of Iterations Exceeded.'})},
        'g6_v29_recomputed_all_records': {'pass': bool(records) and not bad_all, 'n_bad': len(bad_all),
                                          'bad': bad_all},
    }


def _events(rel_dir):
    out = []
    for e in _jsonl(os.path.join(rel_dir, 'network_failures_s39_D.jsonl')):
        out.append({'cycle': e.get('cycle'), 'network': e.get('network_name'), 'year': e.get('year'),
                    'day': e.get('day'), 'primary_termination': e.get('primary_termination'),
                    'recovery_attempted': e.get('recovery_attempted'), 'tier2_attempted': e.get('tier2_attempted'),
                    'termination': e.get('termination'), 'class': e.get('class')})
    return out


def retry_comparison(eval_dir, ref_dir, rec, first_window):
    ev_t, ev_r = _events(eval_dir), _events(ref_dir)
    key = lambda e: tuple(e[k] for k in ('cycle', 'network', 'year', 'day', 'primary_termination',  # noqa: E731
                                         'recovery_attempted', 'tier2_attempted', 'termination', 'class'))
    credit = lambda evs: sum(int(bool(e['recovery_attempted'])) + int(bool(e['tier2_attempted']))  # noqa: E731
                             for e in evs)
    pre_t = [key(e) for e in ev_t if first_window is None or e['cycle'] < first_window]
    pre_r = [key(e) for e in ev_r if first_window is None or e['cycle'] < first_window]
    win_t = [e for e in ev_t if first_window is not None and e['cycle'] >= first_window]
    win_r = [e for e in ev_r if first_window is not None and e['cycle'] >= first_window]
    rows_t = _jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    rows_r = _jsonl(os.path.join(ref_dir, 'per_cycle_record.jsonl'))
    first_diff = next((a['cycle'] for a, b in zip(rows_t, rows_r)
                       if a['gross_operational_cost'] != b['gross_operational_cost']), None)
    sp_r = _load(os.path.join(ref_dir, 'evaluation_record.json')).get('solve_profile') or {}
    return {
        'tail': {'n_events': len(ev_t), 'retries_from_events': credit(ev_t),
                 'retry_solves_credited_in_record': (rec.get('solve_profile') or {}).get('retry_solves_credited'),
                 'observed_solves': ((rec.get('solve_profile') or {}).get('observed') or {}).get('permitted_solve'),
                 'n_pre_window_events': len(pre_t), 'window_events': win_t,
                 'retries_in_window': credit(win_t)},
        'reference': {'n_events': len(ev_r), 'retries_from_events': credit(ev_r),
                      'observed_solves': (sp_r.get('observed') or {}).get('permitted_solve'),
                      'n_pre_window_events': len(pre_r), 'window_events': win_r, 'retries_in_window': credit(win_r)},
        'pre_window_events_identical': pre_t == pre_r,
        'window_events_identical': [key(e) for e in win_t] == [key(e) for e in win_r],
        'per_cycle_gross_first_differing_cycle': first_diff, 'first_window_cycle': first_window,
        'per_cycle_gross_identical_before_window': first_diff is None or (first_window is not None
                                                                          and first_diff >= first_window),
    }


def bar_window(rel_dir):
    bar = _load(os.path.join(rel_dir, 'evaluation_record.json')).get('bar') or {}
    win = bar.get('window') or []
    top = max(win, key=lambda w: w.get('gross_step_abs') or -1) if win else {}
    return {'value': bar.get('value'), 'window_cycles': [w.get('cycle') for w in win],
            'max_step_cycle': top.get('cycle'), 'max_step': top.get('gross_step_abs')}


def score(cmp, v29_pred):
    dq, rel = cmp['dQ'], cmp['dQ_relative']
    return {
        'author_within_1e-6_relative': cmp['author_band_ratio'] <= 1.0,
        'author_band_ratio': cmp['author_band_ratio'],
        'W1_sign_negative': dq < 0, 'W1_dQ_over_predicted': dq / -3.4e3, 'W1_dQrel_over_predicted': rel / -5e-6,
        'W1_magnitude_consistent_factor2': 1.7e3 <= abs(dq) <= 6.8e3,
        'W2_inside_own_bar': abs(dq) < cmp['bar_ref'],
        'W3_author_band_refuted': abs(dq) > 1e-6 * abs(cmp['Q_ref']),
        'W4_determinate_beyond_stopping_slack': cmp['determinate_beyond_stopping_slack'],
        'W6_cert_cycle_within_plus3': (cmp['cert_cycle_tail'] is not None
                                       and 0 <= cmp['cert_cycle_tail'] - cmp['cert_cycle_ref'] <= 3),
        'W6_cert_cycle_delta': (cmp['cert_cycle_tail'] - cmp['cert_cycle_ref'])
        if cmp['cert_cycle_tail'] is not None else None,
    }


# ----------------------------------------------------------------------------------------------------------------------
#  freeze v30
# ----------------------------------------------------------------------------------------------------------------------
def _find_v30():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V30_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    return rel, _sha(rel)


PREDICTIONS = {
    'recorded': ('BEFORE the recorded re-evaluation runs, but NOT BLIND. Seen first: the W86 launch log and '
                 'campaign_results.json, and the W87 Worker\'s pre-freeze read of the persisted non-at / unparsable '
                 'records. P1-P5 were written into this script BEFORE the Worker then DRY-RAN reevaluate() (a '
                 'scratchpad import writing nothing to the repository, both guards verify(0) == []) to test the code '
                 'before freezing; that dry run agreed with P1-P5, and this spec is frozen AFTER it. They are a '
                 'record of expectation, not a test of an unknown outcome.'),
    'P1_rescoped_G6': ('PASS on all three cells: C* window 79-87 (434 records incl. 2 tier-1 recoveries in rounds 82 '
                       '/ 84), unit 104-112 (432), x0 124-132 (432); 0 records failing the unchanged predicate in '
                       'any window; the 3 C* unparsable records lie at rounds 11, 13, 23, outside W'),
    'P2_other_gates': 'G1-G5, G7-G9 True on all three cells, identical to W86',
    'P3_fallback': ('NOT triggered on any cell: all three certified, every gate True under v30, |dQ| < bar_ref '
                    '(own-bar ratios ~0.031 / ~0.027 / ~0.076)'),
    'P4_reproduction': ('v29 G6 recomputed over all records reproduces W86 exactly (C* False, 3 bad records; unit, '
                        'x0 True); every comparison quantity equals W86\'s campaign_results bit for bit'),
    'P5_pre_tail_evidence_c_star': ('C* pre-tail: 45 above + 3 unparsable (rounds 0-78), 38 max_iter exits; unit 1 '
                                    'above; x0 0'),
}


def v30_content(evidence):
    v29 = _load(SPEC_V29['path'])
    return {
        'schema': 'p515_frozen_spec_v30', 'version': 30, 'stage': STAGE_TEXT,
        'authority': ['Planner task W87 (re-scope G6; re-evaluate from persisted records; zero solves)',
                      'CLAUDE.md stage templates: "Scope a gate per arm" (P5.14-N precedent)',
                      'PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7'],
        'predecessor': {'path': SPEC_V29['path'], 'sha256': _sha(SPEC_V29['path'])},
        'predecessor_not_edited': 'v29 stays as frozen; v30 re-scopes one of its per-entry gates',
        'reason': REASON,
        'per_entry_gates': {
            **{k: v for k, v in v29['per_entry_gates'].items() if k != 'G6_floor_records'},
            'G6_floor_records_tail_window': G6_RESCOPED,
            'unchanged_from_v29': 'G1-G5, G7-G9, applies_to, comparison: verbatim; computed by the W86 launcher\'s '
                                  'own evaluation_checks (by import)',
        },
        'v29_G6_verbatim_superseded': v29['per_entry_gates']['G6_floor_records'],
        'fallback_test_operational': v29['fallback_test_operational'],
        'fallback_test_note': ('verbatim from v29; "fails to certify" counts a failed per-entry gate, now with G6 '
                               're-scoped; "moves materially" (|Q_tail - Q_ref| > bar_ref) unchanged'),
        'pre_tail_evidence': PRE_TAIL_EVIDENCE,
        'reported_not_gated': REPORTED_NOT_GATED,
        'integrity_checks': ('before any new quantity is read: the evidence base hashes equal this spec\'s pins and '
                             'the campaign manifest; the re-evaluated G1-G5, G7-G9, the recomputed v29 G6 and every '
                             'comparison quantity equal W86\'s campaign_results; any mismatch -> exit 1'),
        'references': v29['references'], 'reference_R': v29['reference_R'],
        'predictions_scored': {'v29_predictions_recorded_before_the_run': v29['predictions_recorded_before_the_run'],
                               'scoring_rules': SCORING},
        'predictions_recorded_before_the_reevaluation': PREDICTIONS,
        'evidence_base': evidence,
        'output': {'root': OUT_ROOT, 'file': OUT_FILE, 'manifest': OUT_MANIFEST,
                   'note': 'a NEW root; nothing committed is re-run onto or modified'},
        'solve_claim': ('ZERO SOLVES, guard-verified: SolveProfileGuard(permitted=()) armed at import before any '
                        'project import, and the imported launcher\'s own permitted=() guard, both verify(0) == [] on '
                        'every exit path'),
        'script': SCRIPT_NAME, 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'launcher_sha256': H.sha256_file(os.path.abspath(L.__file__)),
        'git_head_at_freeze': H._git(['rev-parse', 'HEAD']), 'frozen_utc': _utc(),
    }


def freeze_spec():
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V30_PREFIX))
    if existing:
        _log(f'[W87-V30 PRECONDITION FAILED] v30 already exists (write-once): {existing}')
        _finish(1)
    evidence, failures = evidence_base()
    launcher_now = H.sha256_file(os.path.abspath(L.__file__))
    if _load(SPEC_V29['path'])['launcher_sha256'] != launcher_now:
        failures.append('the W86 launcher changed since v29 froze')
    if _load(RECERT_SPEC['path'])['extra']['campaign_script_sha256'] != launcher_now:
        failures.append('the W86 launcher differs from the one the re-certification ran (campaign spec pin)')
    if _load(RECERT_SPEC['path'])['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('the campaign harness differs from the one the re-certification ran (campaign spec pin)')
    if failures:
        for f in failures:
            _log(f'[W87-V30 PRECONDITION FAILED] {f}')
        _finish(1)
    content = v30_content(evidence)
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V30_PREFIX}{sha[:8]}.json')
    with open(_abs(rel), 'x') as handle:
        handle.write(text)
    if _sha(rel) != sha:
        raise RuntimeError('v30 written bytes do not hash to the name')
    _log(f'[W87-V30] {STAGE_TEXT}')
    _log(f"[W87-V30] wrote {rel} sha256={sha} (predecessor {content['predecessor']})")
    _log(f"[W87-V30] G6 re-scoped: {G6_RESCOPED['population']}")
    _log(f'[W87-V30] evidence base: {len(evidence["cells"])} cells, campaign results {evidence["campaign_results"]["sha256"]}')
    _finish(0, f'-- run with --run --spec-sha256 {sha}')


# ----------------------------------------------------------------------------------------------------------------------
#  run
# ----------------------------------------------------------------------------------------------------------------------
def reevaluate():
    """Files only: every gate, the comparison, the integrity reproduction of W86, R, the scores."""
    w86 = _load(RECERT_RESULTS)
    entries = _cells_from_recert_spec()
    refs = _load(SPEC_V29['path'])['references']
    per_cell, integrity = {}, {}
    for label in LABELS:
        e = entries[label]
        eval_dir = os.path.join(RECERT_ROOT, 'evals', e['eval_dir'])
        rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
        exit_code = int(open(_abs(os.path.join(eval_dir, 'exit_code.txt'))).read().strip())
        g = {'G1_harness_clean': (exit_code == 0 and rec.get('status') in ('certified', 'not_certified')
                                  and not os.path.exists(_abs(os.path.join(eval_dir, 'parent_barrier_record.json')))),
             'G2_eval_key': rec.get('eval_key') == L.EXPECTED_EVAL_KEYS[label]}
        c, detail = L.evaluation_checks(_abs(eval_dir), rec, e['working_dir_ids']['run'])
        g.update({'G3_append_reconcile': c.get('append_reconciles_byte_identical', False),
                  'G4_tail_state_check': c.get('tail_state_check_matches', False),
                  'G5_solve_profile_reconciled_per_event': c.get('solve_profile_reconciled_per_event', False)})
        g6 = g6_evaluate(eval_dir, rec)
        g['G6_floor_records_tail_window'] = g6['gate_pass']
        g.update({'G7_append_sealed': c.get('append_sealed_after_reconcile', False),
                  'G8_post_certification': c.get('post_certification_persisted',
                                                 c.get('post_certification_skipped_uncertified', False)),
                  'G9_ess_ageing_readback': c.get('ess_ageing_readback_all_match', False)})
        cmp = L.compare(label, rec, refs[label], _abs(eval_dir))
        fails_to_certify = rec.get('status') != 'certified' or not all(g.values())
        moves = cmp['dQ'] is not None and abs(cmp['dQ']) > refs[label]['bar']
        # integrity: reproduce W86
        w = w86['per_cell'][label]
        w_g = w['gates']
        repro = {
            'exit_code_equals_w86_parent_view': exit_code == w['exit_code'],
            'unchanged_gates_equal_w86': all(g[k] == w_g[k] for k in w_g if k != 'G6_floor_records'),
            'v29_G6_recomputed_equals_w86': (g6['g6_v29_recomputed_all_records']['pass'] == w_g['G6_floor_records']
                                             and g6['g6_v29_recomputed_all_records']['n_bad']
                                             == w['detail']['floor']['n_bad']
                                             and g6['g6_v29_recomputed_all_records']['bad'][:10]
                                             == w['detail']['floor']['bad_first']),
            'v29_G6_via_launcher_equals_w86': c.get('floor_records_at_compl_inf_tol_in_force')
            == w_g['G6_floor_records'],
            'comparison_equals_w86': cmp == w['comparison'],
        }
        integrity[label] = repro
        rc = retry_comparison(eval_dir, refs[label]['eval_dir'], rec, min(g6['window_W']) if g6['window_W'] else None)
        per_cell[label] = {
            'gates_v30': g, 'gates_v30_pass': all(g.values()),
            'fallback_test_v30': {'fails_to_certify': fails_to_certify, 'moves_materially': moves,
                                  'triggered': fails_to_certify or moves},
            'fallback_test_w86_as_run': w['fallback_test'],
            'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
            'certification_cycle': rec.get('certification_cycle'), 'eval_dir': eval_dir, 'eval_key': rec.get('eval_key'),
            'candidate_key': rec.get('candidate_key'),
            'g6': g6, 'comparison': cmp, 'retry_comparison': rc,
            'bar_window': {'tail': bar_window(eval_dir), 'reference': bar_window(refs[label]['eval_dir'])},
            'scores': score(cmp, None),
        }
    integrity_ok = all(all(v.values()) for v in integrity.values())
    q = {k: per_cell[k]['comparison']['Q_tail'] for k in per_cell}
    r_ref = refs['x0']['Q_gross'] - refs['n7_4h_e1']['Q_gross']
    r_tail = q['x0'] - q['n7_4h_e1']
    bar_sum = refs['x0']['bar'] + refs['n7_4h_e1']['bar']
    R = {'objective_convention': 'gross_operational_cost (settlement excluded)', 'R_tail': r_tail, 'R_ref': r_ref,
         'dR': r_tail - r_ref, 'bar_sum_ref': bar_sum, 'abs_dR_over_bar_sum': abs(r_tail - r_ref) / bar_sum,
         'W7_R_change_below_bar_sum': abs(r_tail - r_ref) < bar_sum, 'equals_w86': {
             'R_tail': r_tail == w86['R']['R_tail'], 'R_ref': r_ref == w86['R']['R_ref'],
             'dR': (r_tail - r_ref) == w86['R']['dR']}}
    supplementary = {
        'W5_all_certify': all(per_cell[k]['status'] == 'certified' for k in LABELS),
        'P7_acceptable_in_window': {k: per_cell[k]['g6']['reported_window']['n_acceptable'] for k in LABELS},
        'P8_c_star_median_iterations_window_primary_vs_v28_reference_TSO30_DSO44':
            per_cell['c_star']['g6']['reported_window']['median_iterations_primary'],
        'P10_window_retries_tail_vs_ref': {k: [per_cell[k]['retry_comparison']['tail']['retries_in_window'],
                                               per_cell[k]['retry_comparison']['reference']['retries_in_window']]
                                           for k in LABELS},
    }
    fallback_cells = [k for k in LABELS if per_cell[k]['fallback_test_v30']['triggered']]
    return {'per_cell': per_cell, 'integrity': integrity, 'integrity_ok': integrity_ok, 'R': R,
            'supplementary': supplementary, 'fallback_cells': fallback_cells, 'w86': w86}


def run(spec_sha256, started):
    tag = 'W87-REEVAL'
    failures = []
    v30_rel, v30_sha = _find_v30()
    if v30_rel is None or v30_sha != spec_sha256 or not os.path.basename(v30_rel).startswith(
            f'{SPEC_V30_PREFIX}{spec_sha256[:8]}'):
        failures.append(f'v30 not found / sha mismatch: {v30_rel} {v30_sha} vs {spec_sha256}')
    else:
        failures += [f'v30 {k} False' for k, v in _git_state(v30_rel).items() if not v]
    if os.path.exists(_abs(OUT_ROOT)):
        failures.append(f'output root exists (write-once): {OUT_ROOT}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    v30 = _load(v30_rel)
    if v30['script_sha256'] != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script changed since v30 froze')
    if v30['launcher_sha256'] != H.sha256_file(os.path.abspath(L.__file__)):
        failures.append('the W86 launcher changed since v30 froze')
    evidence, more = evidence_base()
    failures += more
    for label in LABELS:
        for f, v in evidence['cells'][label]['files'].items():
            if v['sha256'] != v30['evidence_base']['cells'][label]['files'][f]['sha256']:
                failures.append(f'{label} {f}: sha256 != v30 pin')
    if evidence['campaign_results']['sha256'] != v30['evidence_base']['campaign_results']['sha256']:
        failures.append('campaign_results sha256 != v30 pin')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)

    body = reevaluate()
    per_cell, integrity, integrity_ok, R = body['per_cell'], body['integrity'], body['integrity_ok'], body['R']
    supplementary, fallback_cells, w86 = body['supplementary'], body['fallback_cells'], body['w86']
    g = guards_verify()
    results = {
        'stage': STAGE_TEXT, 'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'spec_v30': {'path': v30_rel, 'sha256': v30_sha}, 'spec_v29': SPEC_V29, 'evidence_base': evidence,
        'objective_convention': 'Q gross_operational_cost (settlement excluded) on every table',
        'g6_rescoped_definition': G6_RESCOPED,
        'integrity_reproduces_w86': integrity, 'integrity_ok': integrity_ok,
        'per_cell': per_cell, 'all_gates_pass_v30': all(per_cell[k]['gates_v30_pass'] for k in LABELS),
        'fallback_triggered_v30': bool(fallback_cells), 'fallback_cells_v30': fallback_cells,
        'fallback_w86_as_run': {'triggered': w86['fallback_triggered'], 'cells': w86['fallback_cells']},
        'R': R, 'supplementary_scores': supplementary,
        'solve_claim': 'ZERO SOLVES, guard-verified (both guards permitted=(), verify(0))',
        'guards': g, 'wall_s': time.time() - started,
    }
    os.makedirs(_abs(OUT_ROOT))
    H._write_once_json(_abs(os.path.join(OUT_ROOT, OUT_FILE)), results)
    man = {os.path.join(OUT_ROOT, OUT_FILE): _sha(os.path.join(OUT_ROOT, OUT_FILE))}
    H._write_once_json(_abs(os.path.join(OUT_ROOT, OUT_MANIFEST)), man)
    _log(f'[{tag}] {STAGE_TEXT}; v30 {v30_sha}')
    _log(f'[{tag}] integrity (W86 reproduced): {integrity_ok} {integrity}')
    for k in LABELS:
        p = per_cell[k]
        _log(f"[{tag}] {k}: status {p['status']} cert {p['certification_cycle']} gates_v30 {p['gates_v30']} "
             f"fallback_v30 {p['fallback_test_v30']} (W86 as run {p['fallback_test_w86_as_run']})")
        _log(f"[{tag}]   G6 window {p['g6']['window_W']} T {p['g6']['terminal_round_T']} population "
             f"{p['g6']['population_n']} bad {p['g6']['n_bad']} non_vacuous {p['g6']['non_vacuous']}; "
             f"v29 G6 recomputed {p['g6']['g6_v29_recomputed_all_records']['pass']} "
             f"({p['g6']['g6_v29_recomputed_all_records']['n_bad']} bad)")
        _log(f"[{tag}]   comparison {p['comparison']}")
        _log(f"[{tag}]   scores {p['scores']}")
    _log(f"[{tag}] fallback v30 {fallback_cells}; R {R}")
    _log(f'[{tag}] wrote {os.path.join(OUT_ROOT, OUT_FILE)} sha256 {man[os.path.join(OUT_ROOT, OUT_FILE)]}')
    _finish(0 if integrity_ok else 1, f'integrity_ok={integrity_ok} wall={time.time() - started:.1f}s')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    try:
        if args.freeze_spec:
            freeze_spec()
        else:
            if not args.spec_sha256:
                parser.error('--run requires --spec-sha256')
            run(args.spec_sha256, started)
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
