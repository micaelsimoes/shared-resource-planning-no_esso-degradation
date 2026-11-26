"""
P5.15 Addendum 46 ruling 7, Planner task W88 -- two ZERO-SOLVE items.

ITEM 1. STRENGTHEN gate G6 of the tight-tail SRP1 re-certification into frozen spec v31 (predecessor v30 bb6703da,
NOT edited) and RE-EVALUATE the three cells from their PERSISTED records.
  Ruling 1 -- floor status is GATED in the terminal round: every terminal-round record whose monotone mu-floor formula
             IS applicable must have floor_status 'at'; records where it is NOT applicable are counted and reported,
             never judged on floor status; non-vacuity: >= 48 applicable records in the terminal round.
  Ruling 2 -- a tier-2 retry (mu_strategy adaptive) INSIDE the judged population no longer fails by construction: it
             is judged on the conditions that DO apply (options_list_agrees, compl_inf_tol_in_force == 1e-6 in the
             tail window) and its mu-floor test is recorded NOT APPLICABLE, counted. It is not excused from anything
             else: any OTHER parse problem, or a not-applicable declaration outside the tier-2 / adaptive class, fails.
  Everything else is v30's: the population (W u {T}), the compl_inf_tol / options_list_agrees conditions, v30's
  non-vacuity, G1-G5, G7-G9 and the fallback test including "moves materially" (|dQ| > bar_ref), all verbatim.

ITEM 2. The campaign_s47_recert identity look (Addendum 46 order, before that reference is reused): why its recorded
`solve_profile.identity_holds` is False, whether the count reconciles, what committed artefact depends on the flag,
whether the reference value is affected, and the SCOPE of the search -- all computed from committed files and git.

HOW. Files only. W87's harness is imported so every v30 quantity is computed by the committed code that produced the
committed W87 re-evaluation, and that re-evaluation is REPRODUCED (compared against the committed reeval_w87.json)
before any new quantity is read. SolveProfileGuard(permitted=()) is armed at import BEFORE any project import; the
imported W87 guard and the imported W86 launcher guard (both permitted=()) are verified too: all three verify(0) == []
on every exit path.

MODES (repo root, canonical interpreter; attached, alone, both streams captured):
  --freeze-spec                   writes data/SRP1/Results/P515S53/frozen_s53_spec_v31_<sha8>.json (write-once, named
                                  by its sha256).
  --run --spec-sha256 S           item 1 -> data/SRP1/Results/P515S53/g6_floor_w88/reeval_w88.json + manifest (NEW root)
  --s47-identity-look             item 2 -> data/SRP1/Results/P515S53/s47_identity_look_w88/s47_identity_look.json +
                                  manifest (NEW root; the s47 campaign and its artefacts are only read)
Exit codes: 0 done (whatever the verdict), 1 a precondition / integrity / self-test / guard failure.

EXACT COMMANDS:
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w88_g6_floor_reeval.py \\
      --freeze-spec > data/SRP1/Results/P515S53/g6_floor_w88_freeze_spec_v31_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w88_g6_floor_reeval.py \\
      --run --spec-sha256 <sha> > data/SRP1/Results/P515S53/g6_floor_w88_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w88_g6_floor_reeval.py \\
      --s47-identity-look > data/SRP1/Results/P515S53/s47_identity_look_w88_launch.log 2>&1
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W88 G6 floor gate + s47 identity look (never solves)').install()

# Arms W87's own GUARD (permitted=()) and, through it, the W86 launcher's PARENT_GUARD (permitted=()), at import.
import p515_s53_w87_g6_rescope_reeval as W  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402 -- event_level_solve_reconciliation only (installs no guard at import)

L = W.L
H = W.H

SCRIPT_NAME = os.path.basename(__file__)
STAGE_TEXT = ('P5.15 Addendum 46 ruling 7, W88 -- G6 strengthened (terminal-round floor status gated; tier-2 retries '
              'judged on the conditions that apply); the W86 tight-tail re-certification re-evaluated from its '
              'persisted records under v31 (zero solves); the campaign_s47_recert identity look (zero solves)')
_P53 = W._P53
SPEC_V30 = {'path': os.path.join(_P53, 'frozen_s53_spec_v30_bb6703da.json'),
            'sha256': 'bb6703da668e9c5fc951ddcae24937cd6b85f16499d3d042f5e094fd59ae9023'}
SPEC_V31_PREFIX = 'frozen_s53_spec_v31_'
W87_OUT = {'path': os.path.join(W.OUT_ROOT, W.OUT_FILE),
           'sha256': '1ccdef2a1c7925b605874762d9a36ef0d36e5fba5459df9c32078c7f63b1ccbd'}
W87_MANIFEST = os.path.join(W.OUT_ROOT, W.OUT_MANIFEST)
OUT_ROOT = os.path.join(_P53, 'g6_floor_w88')
OUT_FILE = 'reeval_w88.json'
OUT_MANIFEST = 'reeval_w88_manifest_sha256.json'
S47_OUT_ROOT = os.path.join(_P53, 's47_identity_look_w88')
S47_OUT_FILE = 's47_identity_look.json'
S47_OUT_MANIFEST = 's47_identity_look_manifest_sha256.json'
LABELS = W.LABELS
BLOCKS_PER_ROUND = W.BLOCKS_PER_ROUND   # 48 = (1 TSO + 3 DSO) x 3 years x 4 days
TAIL_TOL = W.TAIL_TOL                   # 1e-6
NETWORK_PY = 'network.py'
# The parser's own declaration (network.parse_ipopt_attempt_segment): the text it appends to parse_reason when an
# attempt passes a blocker of the monotone mu-floor formula. Asserted present in network.py before any use.
NA_CLAUSE = 'monotone mu-floor formula not applicable'
NA_SOURCE_LINE = "reasons.append(f'monotone mu-floor formula not applicable: {blockers} passed')"
NA_DECLARED_CLASS = {'attempt': 'recovery_tier2', 'mu_strategy_passed': 'adaptive'}

# ----------------------------------------------------------------------------------------------------------------------
#  G6 under v31 -- stated operationally
# ----------------------------------------------------------------------------------------------------------------------
G6_V31 = {
    'name': 'G6_floor_records_v31',
    'replaces': 'v30 per_entry_gates.G6_floor_records_tail_window',
    'population_unchanged_from_v30': ('P(cell) = { r in <eval_dir>/network_ipopt_solve_records.jsonl : r.round in W '
                                      'u {T} }, W = { p.cycle : p in convergence_depth_tail_state.json per_cycle, '
                                      'p.active is True }, T = evaluation_record.cycles_run; every attempt (primary, '
                                      'recovery, recovery_tier2)'),
    'applicability': ('a record is NOT APPLICABLE iff its parse_reason, split on "; " (the parser\'s join), holds a '
                      f'component starting "{NA_CLAUSE}:" -- the declaration network.parse_ipopt_attempt_segment '
                      'makes when the attempt passes mu_strategy / barrier_tol_factor / mu_target / mu_min. Every '
                      'other record is APPLICABLE. residual_reasons = the parse_reason components other than that '
                      'declaration.'),
    'predicate_applicable_record': ('compl_inf_tol_in_force == 1e-6 if r.round in W else the network\'s production '
                                    'value; AND options_list_agrees is True; AND parse_reason is None [v30 verbatim]; '
                                    'AND, if r.round == T: floor_status == "at" [RULING 1, new]'),
    'predicate_not_applicable_record': ('compl_inf_tol_in_force == 1e-6 if r.round in W else the production value; AND '
                                        'options_list_agrees is True; AND residual_reasons is empty (no parse problem '
                                        'other than the declaration); AND attempt == "recovery_tier2" AND '
                                        'mu_strategy_passed == "adaptive" (the declared class -- a not-applicable '
                                        'record of any other kind FAILS) [RULING 2, new]. Its floor test is recorded '
                                        'NOT APPLICABLE and counted, in every judged round including T.'),
    'non_vacuity': ('v30 verbatim (P non-empty; every round of W u {T} holds exactly 48 primary-attempt records and >= '
                    '48 records) AND [RULING 1, new] the terminal round T holds >= 48 APPLICABLE records'),
    'passes_iff': 'non_vacuity holds AND every record of P satisfies its predicate',
    'floor_status_definition': ('the persisted field, computed by the production parser: "at" iff |mu_final / '
                                'mu_floor - 1| <= 1e-3, mu_floor = min(tol, compl_inf_tol * obj_scaling) / 11. '
                                'Recomputed from the persisted mu_over_floor and REPORTED (agreement), not gated.'),
    'not_judged': ('floor status outside T (window rounds before T: reported, as in v30); exit status (reported); '
                   'pre-tail records (reported evidence, as in v30)'),
    'design_consequence_recorded': ('ruling 1 judges EVERY applicable terminal-round record, including a primary '
                                    'that failed and was retried: a terminal-round primary that exits at max_iter '
                                    'above the floor fails G6 even if its tier-1 retry reaches the floor. None of '
                                    'the three cells has a non-primary record in T (W87 reported tallies); stated '
                                    'here so the rule is not discovered later'),
}
REASON = (
    'v30 G6 re-scoped the population but still did not test what the tail exists to achieve -- that the terminal '
    'round is solved at the mu floor; floor status was only reported (Planner W88 ruling 1). And under v30 a tier-2 '
    'retry inside the window would fail G6 BY CONSTRUCTION, because the parser always sets parse_reason on tier-2 '
    '(mu_strategy adaptive -> the monotone floor formula is declared not applicable) -- the same spurious-failure '
    'mechanism v30 removed for the pre-tail rounds (Planner W88 ruling 2). v31 adds the terminal floor test and '
    'judges not-applicable records on the conditions that apply. No threshold is raised or relaxed elsewhere; the '
    'fallback test is unchanged.')

# Self-tests: the v31 predicate on REAL persisted C* records, deep-copied and altered, with the verdict declared here
# before the run. They exercise the ruling-2 path, which the three cells' own populations may not reach.
SELF_TESTS = [
    {'id': 'T1', 'base': 'c_star tier-2 record (round 11)', 'transform': 'round -> min(W); compl_inf_tol_in_force -> 1e-6',
     'expect_v31': [], 'expect_v30_fails': True,
     'why': 'a legitimate adaptive-mu retry in the window: v31 passes it; v30 failed it by construction'},
    {'id': 'T2', 'base': 'c_star tier-2 record (round 11)', 'transform': 'round -> min(W); compl_inf_tol_in_force kept 1e-4',
     'expect_v31': ['compl_inf_tol_in_force'], 'expect_v30_fails': True,
     'why': 'the tolerance condition still applies to a tier-2 record'},
    {'id': 'T3', 'base': 'c_star tier-2 record (round 11)',
     'transform': 'round -> min(W); compl_inf_tol_in_force -> 1e-6; options_list_agrees -> False',
     'expect_v31': ['options_list_agrees'], 'expect_v30_fails': True,
     'why': 'the options-list condition still applies to a tier-2 record'},
    {'id': 'T4', 'base': 'c_star tier-2 record (round 11)',
     'transform': ('round -> min(W); compl_inf_tol_in_force -> 1e-6; parse_reason += "; no IPOPT options list '
                   'precedes the banner in the attempt segment"'),
     'expect_v31': ['parse_reason_beyond_not_applicable'], 'expect_v30_fails': True,
     'why': 'a parse problem other than the declaration is not excused'},
    {'id': 'T5', 'base': 'c_star tier-2 record (round 11)',
     'transform': 'round -> min(W); compl_inf_tol_in_force -> 1e-6; attempt -> "primary"',
     'expect_v31': ['not_applicable_outside_declared_class'], 'expect_v30_fails': True,
     'why': 'a not-applicable declaration outside the tier-2 / adaptive class fails'},
    {'id': 'T6', 'base': 'c_star tier-2 record (round 11)', 'transform': 'round -> T; compl_inf_tol_in_force -> 1e-6',
     'expect_v31': [], 'expect_v30_fails': True,
     'why': 'a tier-2 retry in the terminal round: floor test NOT APPLICABLE, counted, not judged'},
    {'id': 'T7', 'base': 'c_star terminal-round primary record (round T, floor at)', 'transform': 'none',
     'expect_v31': [], 'expect_v30_fails': False, 'why': 'a real terminal-round record at the floor passes'},
    {'id': 'T8', 'base': 'c_star terminal-round primary record (round T, floor at)', 'transform': 'floor_status -> "above"',
     'expect_v31': ['terminal_floor_status_not_at'], 'expect_v30_fails': False,
     'why': 'ruling 1: an applicable terminal-round record not at the floor fails (v30 did not judge it)'},
    {'id': 'T9', 'base': 'c_star window primary record above the floor (round 82, max_iter)', 'transform': 'none',
     'expect_v31': [], 'expect_v30_fails': False,
     'why': 'floor status is judged in T only; a window record above the floor before T is reported, not judged'},
    {'id': 'T10', 'base': 'non-vacuity on a synthetic count', 'transform': ('terminal round with 48 primary records '
                                                                             'of which 1 not applicable'),
     'expect_non_vacuous': False, 'why': 'fewer than 48 applicable records in T fails as loudly as a bad record'},
]

CAVEATS = {
    'bar_tail_not_independent': (
        'bar_tail equals bar_ref EXACTLY on all three cells: each run\'s 10-cycle bar window (cycles T-9 .. T) starts '
        'on the cycle just BEFORE the first tail cycle, that pre-tail cycle holds the window\'s largest gross step, '
        'and it is bitwise identical in both runs (the runs agree bitwise before the tail). The "two-run bar" '
        'bar_ref + bar_tail therefore counts ONE pre-tail step twice and is NOT an independent measure; every ratio '
        'computed against it (two_run_bar_resolution) carries this caveat. MEASURED per cell in the run output.'),
    'bitwise_before_tail': (
        'The per-cycle gross_operational_cost trajectory is bitwise identical (float.hex) on every cycle before the '
        'first tail cycle, on every cell, first differing at the first tail cycle (W87 reported 79 / 104 / 124). This '
        'is the MEASURED form of Addendum 46\'s "everything before the tail bitwise unchanged". MEASURED per cell in '
        'the run output (number of cycles compared, first differing cycle).'),
}

PREDICTIONS = {
    'recorded': (
        'BEFORE the recorded re-evaluation runs, and NOT BLIND. Seen first: the committed W87 re-evaluation '
        '(reeval_w87.json) including its REPORTED terminal-round tallies (48 primary records per cell, all "at", 0 '
        'unparsable), window tallies (0 tier-2 records in any W) and caveat inputs (bar windows, first differing '
        'cycles). Before the freeze the Worker ran ONLY: py_compile; the self-tests below (which read the C* records '
        'file and use three real C* records: the round-11 tier-2 record, a round-T primary, the round-82 primary above '
        'the floor); g6_v31_evaluate and caveats_measured on SYNTHETIC made-up files in the scratchpad; the '
        'read-only preconditions; and the item-2 look into a scratchpad path -- all through a scratchpad import '
        'writing nothing to the repository. The per-cell v31 G6 evaluation and the caveat measurement were NOT '
        'executed on the persisted cell records before the freeze. These predictions are a record of expectation from reported tallies, not a test of an '
        'unknown outcome.'),
    'P1_G6_v31': ('PASS on all three cells: population 434 / 432 / 432 (unchanged from v30); terminal round T = 87 / 112 '
                  '/ 132 holds 48 applicable records, all "at", 0 not applicable; 0 not-applicable records anywhere in '
                  'W u {T} on any cell (so the ruling-2 path is exercised only by the self-tests); 0 bad records'),
    'P2_other_gates': 'G1-G5, G7-G9 True on all three cells, identical to W87 / W86',
    'P3_fallback': 'NOT triggered on any cell (all certified, every v31 gate True, |dQ| < bar_ref)',
    'P4_reproduction': 'W87\'s committed per-cell results, R and fallback reproduced exactly',
    'P5_self_tests': 'every self-test verdict as declared',
    'P6_caveats': ('bar_tail == bar_ref exactly, max step at cycles 78 / 103 / 123 (first tail cycle - 1) in both runs, '
                   'bitwise equal; per-cycle gross bitwise identical on cycles 1..78 / 1..103 / 1..123, first differing '
                   'at 79 / 104 / 124'),
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
    return W._git_state(rel)


def _git_rc(args):
    """git with the return code kept (git grep exits 1 on no match)."""
    p = subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True)
    return p.returncode, p.stdout


def _roundtrip(obj):
    return json.loads(json.dumps(obj, default=H._json_default))


def _hex(x):
    return float(x).hex() if isinstance(x, (int, float)) and not isinstance(x, bool) else repr(x)


def guards_verify():
    return {'w88_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)},
            'imported_w87_guard': {'counts': dict(W.GUARD.counts), 'verify_0_failures': W.GUARD.verify(0)},
            'imported_launcher_guard': {'counts': dict(L.PARENT_GUARD.counts),
                                        'verify_0_failures': L.PARENT_GUARD.verify(0)}}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    """EVERY exit path: verify all three guards at exactly 0, uninstall (LIFO), exit (1 if a guard fails)."""
    g = guards_verify()
    _log(f'[W88] guards {g} {extra_msg}')
    L.PARENT_GUARD.uninstall()
    W.GUARD.uninstall()
    GUARD.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


# ----------------------------------------------------------------------------------------------------------------------
#  item 1 -- the v31 predicate
# ----------------------------------------------------------------------------------------------------------------------
def na_source_check():
    with open(_abs(NETWORK_PY)) as handle:
        src = handle.read()
    return {'path': NETWORK_PY, 'sha256': _sha(NETWORK_PY), **_git_state(NETWORK_PY),
            'declaration_line_present': NA_SOURCE_LINE in src, 'declaration_line': NA_SOURCE_LINE,
            'join': "record['parse_reason'] = '; '.join(reasons) if reasons else None",
            'join_present': "record['parse_reason'] = '; '.join(reasons) if reasons else None" in src}


def classify(r):
    parts = [p for p in (r.get('parse_reason') or '').split('; ') if p] if r.get('parse_reason') is not None else []
    na = [p for p in parts if p.startswith(NA_CLAUSE + ':')]
    residual = [p for p in parts if not p.startswith(NA_CLAUSE + ':')]
    if r.get('parse_reason') is not None and not parts:
        residual = [r.get('parse_reason')]   # an empty-string parse_reason is a parse problem, not a declaration
    return (not na), residual


def v31_failures(r, window, terminal, prod):
    """The reasons record r fails the v31 predicate ([] = passes)."""
    out = []
    want = TAIL_TOL if r.get('round') in window else prod.get(r.get('network'))
    if r.get('compl_inf_tol_in_force') != want:
        out.append('compl_inf_tol_in_force')
    if r.get('options_list_agrees') is not True:
        out.append('options_list_agrees')
    applicable, residual = classify(r)
    if applicable:
        if r.get('parse_reason') is not None:
            out.append('parse_reason')
        if r.get('round') == terminal and r.get('floor_status') != 'at':
            out.append('terminal_floor_status_not_at')
    else:
        if residual:
            out.append('parse_reason_beyond_not_applicable')
        if any(r.get(k) != v for k, v in NA_DECLARED_CLASS.items()):
            out.append('not_applicable_outside_declared_class')
    return out


def non_vacuous(per_round, n_applicable_terminal, population_n):
    v30_part = bool(population_n) and all(v['n_primary'] == BLOCKS_PER_ROUND and v['n'] >= BLOCKS_PER_ROUND
                                          for v in per_round.values())
    return {'v30_part': v30_part, 'terminal_applicable_ge_48': n_applicable_terminal >= BLOCKS_PER_ROUND,
            'holds': v30_part and n_applicable_terminal >= BLOCKS_PER_ROUND}


def _brief(r):
    return W._brief(r)


def g6_v31_evaluate(eval_dir, rec):
    records = _jsonl(os.path.join(eval_dir, 'network_ipopt_solve_records.jsonl'))
    ts = _load(os.path.join(eval_dir, 'convergence_depth_tail_state.json'))
    window = {p['cycle'] for p in (ts.get('per_cycle') or []) if p.get('active')}
    terminal = rec.get('cycles_run')
    judged_rounds = window | {terminal}
    prod = L._production_compl_inf_tol(ts.get('baseline'))
    pop = [r for r in records if r.get('round') in judged_rounds]
    per_round = {k: {'n': sum(1 for r in pop if r.get('round') == k),
                     'n_primary': sum(1 for r in pop if r.get('round') == k and r.get('attempt') == 'primary')}
                 for k in sorted(judged_rounds)}
    term = [r for r in pop if r.get('round') == terminal]
    term_app = [r for r in term if classify(r)[0]]
    term_na = [r for r in term if not classify(r)[0]]
    win_not_t = [r for r in pop if r.get('round') != terminal]
    pre = [r for r in records if r.get('round') not in judged_rounds]
    nv = non_vacuous(per_round, len(term_app), len(pop))
    bad = []
    for r in pop:
        f = v31_failures(r, window, terminal, prod)
        if f:
            bad.append({**_brief(r), 'failures': f})
    floor_recomputed = []
    for r in term_app:
        m = r.get('mu_over_floor')
        rec_at = (m is not None and abs(m - 1.0) <= 1e-3)
        floor_recomputed.append(rec_at == (r.get('floor_status') == 'at'))
    return {
        'gate_pass': nv['holds'] and not bad,
        'window_W': sorted(window), 'terminal_round_T': terminal, 'judged_rounds': sorted(judged_rounds),
        'production_compl_inf_tol': prod, 'n_records_total': len(records), 'population_n': len(pop),
        'per_round_counts': per_round, 'non_vacuity': nv, 'n_bad': len(bad), 'bad': bad,
        'terminal_round': {
            'n': len(term), 'n_applicable': len(term_app), 'n_not_applicable': len(term_na),
            'applicable_floor_status': dict(sorted(Counter(str(r.get('floor_status')) for r in term_app).items())),
            'applicable_attempts': dict(sorted(Counter(r.get('attempt') for r in term_app).items())),
            'exit': dict(sorted(Counter(str(r.get('exit')) for r in term).items())),
            'not_applicable_records': [_brief(r) for r in term_na],
            'floor_status_recomputed_from_mu_over_floor_agrees': all(floor_recomputed),
            'mu_over_floor_range_applicable': ([min(r['mu_over_floor'] for r in term_app),
                                                max(r['mu_over_floor'] for r in term_app)]
                                               if term_app and all(r.get('mu_over_floor') is not None
                                                                   for r in term_app) else None)},
        'window_rounds_before_T': {
            'n': len(win_not_t), 'n_applicable': sum(1 for r in win_not_t if classify(r)[0]),
            'n_not_applicable': sum(1 for r in win_not_t if not classify(r)[0]),
            'not_applicable_records': [_brief(r) for r in win_not_t if not classify(r)[0]],
            'floor_status_applicable_reported_not_judged': dict(sorted(Counter(
                str(r.get('floor_status')) for r in win_not_t if classify(r)[0]).items()))},
        'pre_tail_reported_not_gated': {
            'n': len(pre), 'n_applicable': sum(1 for r in pre if classify(r)[0]),
            'n_not_applicable': sum(1 for r in pre if not classify(r)[0]),
            'not_applicable_rounds': sorted({r['round'] for r in pre if not classify(r)[0]}),
            'n_not_applicable_in_declared_class': sum(
                1 for r in pre if not classify(r)[0] and all(r.get(k) == v for k, v in NA_DECLARED_CLASS.items())),
            'n_not_applicable_with_residual_reasons': sum(1 for r in pre if not classify(r)[0] and classify(r)[1])},
    }


def self_tests():
    """The declared self-tests, on real C* records (deep copies). Returns (results, all_ok)."""
    entries = W._cells_from_recert_spec()
    d = os.path.join(W.RECERT_ROOT, 'evals', entries['c_star']['eval_dir'])
    rec = _load(os.path.join(d, 'evaluation_record.json'))
    records = _jsonl(os.path.join(d, 'network_ipopt_solve_records.jsonl'))
    ts = _load(os.path.join(d, 'convergence_depth_tail_state.json'))
    window = {p['cycle'] for p in (ts.get('per_cycle') or []) if p.get('active')}
    terminal = rec.get('cycles_run')
    prod = L._production_compl_inf_tol(ts.get('baseline'))
    t2 = [r for r in records if r.get('round') == 11 and r.get('attempt') == 'recovery_tier2']
    tp = [r for r in records if r.get('round') == terminal and r.get('attempt') == 'primary'
          and r.get('floor_status') == 'at']
    w82 = [r for r in records if r.get('round') == 82 and r.get('attempt') == 'primary'
           and r.get('floor_status') == 'above']
    if len(t2) != 1 or not tp or len(w82) != 1:
        return {'base_records_found': {'tier2_round11': len(t2), 'terminal_primary_at': len(tp),
                                       'round82_primary_above': len(w82)}}, False
    base = {'T1': t2[0], 'T2': t2[0], 'T3': t2[0], 'T4': t2[0], 'T5': t2[0], 'T6': t2[0], 'T7': tp[0], 'T8': tp[0],
            'T9': w82[0]}
    first = min(window)

    def alter(tid, r):
        r = copy.deepcopy(r)
        if tid in ('T1', 'T2', 'T3', 'T4', 'T5'):
            r['round'] = first
        if tid == 'T6':
            r['round'] = terminal
        if tid in ('T1', 'T3', 'T4', 'T5', 'T6'):
            r['compl_inf_tol_in_force'] = TAIL_TOL
        if tid == 'T3':
            r['options_list_agrees'] = False
        if tid == 'T4':
            r['parse_reason'] = r['parse_reason'] + '; no IPOPT options list precedes the banner in the attempt segment'
        if tid == 'T5':
            r['attempt'] = 'primary'
        if tid == 'T8':
            r['floor_status'] = 'above'
        return r

    out, ok = [], True
    for t in SELF_TESTS:
        if t['id'] == 'T10':
            per_round = {terminal: {'n': 48, 'n_primary': 48}}
            nv = non_vacuous(per_round, 47, 48)
            res = {'id': 'T10', 'observed_non_vacuous': nv['holds'], 'expect_non_vacuous': t['expect_non_vacuous'],
                   'ok': nv['holds'] == t['expect_non_vacuous']}
        else:
            r = alter(t['id'], base[t['id']])
            f31 = v31_failures(r, window, terminal, prod)
            f30 = W._predicate_failure(r, window, prod)
            res = {'id': t['id'], 'base_record': _brief(base[t['id']]), 'tested_record': _brief(r),
                   'observed_v31_failures': f31, 'expect_v31': t['expect_v31'],
                   'observed_v30_fails': f30 is not None, 'expect_v30_fails': t['expect_v30_fails'],
                   'applicable': classify(r)[0],
                   'ok': f31 == t['expect_v31'] and (f30 is not None) == t['expect_v30_fails']}
        ok = ok and res['ok']
        out.append(res)
    return out, ok


# ----------------------------------------------------------------------------------------------------------------------
#  item 1 -- caveats, measured
# ----------------------------------------------------------------------------------------------------------------------
def caveats_measured(eval_dir, ref_dir, first_window):
    rt = _load(os.path.join(eval_dir, 'evaluation_record.json'))
    rr = _load(os.path.join(ref_dir, 'evaluation_record.json'))
    bt, br = rt.get('bar') or {}, rr.get('bar') or {}
    wt, wr = bt.get('window') or [], br.get('window') or []
    top_t = max(wt, key=lambda w: w.get('gross_step_abs') or -1)
    top_r = max(wr, key=lambda w: w.get('gross_step_abs') or -1)
    tail_only = lambda win: max((w['gross_step_abs'] for w in win if w['cycle'] >= first_window),  # noqa: E731
                                default=None)
    bar = {
        'bar_tail': bt.get('value'), 'bar_ref': br.get('value'),
        'bar_tail_equals_bar_ref_bitwise': _hex(bt.get('value')) == _hex(br.get('value')),
        'window_cycles_tail': [w['cycle'] for w in wt], 'window_cycles_ref': [w['cycle'] for w in wr],
        'max_step_cycle_tail': top_t.get('cycle'), 'max_step_cycle_ref': top_r.get('cycle'),
        'max_step_cycle_is_first_tail_cycle_minus_1': (top_t.get('cycle') == first_window - 1
                                                       and top_r.get('cycle') == first_window - 1),
        'max_step_bitwise_equal_across_runs': _hex(top_t.get('gross_step_abs')) == _hex(top_r.get('gross_step_abs')),
        'max_step_equals_bar_value_both': (_hex(top_t.get('gross_step_abs')) == _hex(bt.get('value'))
                                           and _hex(top_r.get('gross_step_abs')) == _hex(br.get('value'))),
        'max_step_within_tail_cycles_REPORTED_not_a_bar': {'tail': tail_only(wt), 'ref': tail_only(wr),
                                                           'definition': 'max gross_step_abs over bar-window entries '
                                                                         'with cycle >= first tail cycle'},
    }
    bar['two_run_bar_not_independent'] = bool(bar['bar_tail_equals_bar_ref_bitwise']
                                              and bar['max_step_cycle_is_first_tail_cycle_minus_1']
                                              and bar['max_step_bitwise_equal_across_runs'])
    rows_t = _jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    rows_r = _jsonl(os.path.join(ref_dir, 'per_cycle_record.jsonl'))
    by_t = {x['cycle']: x['gross_operational_cost'] for x in rows_t}
    by_r = {x['cycle']: x['gross_operational_cost'] for x in rows_r}
    pre_cycles = sorted(c for c in by_t if c < first_window)
    common = sorted(set(by_t) & set(by_r))
    first_diff = next((c for c in common if _hex(by_t[c]) != _hex(by_r[c])), None)
    bitwise = {
        'definition': 'per_cycle_record.jsonl gross_operational_cost compared by float.hex, tail run vs reference',
        'cycles_compared_before_first_tail_cycle': [pre_cycles[0], pre_cycles[-1]] if pre_cycles else None,
        'n_cycles_compared_before_first_tail_cycle': len(pre_cycles),
        'same_cycle_set_before_first_tail_cycle': pre_cycles == sorted(c for c in by_r if c < first_window),
        'all_bitwise_identical_before_first_tail_cycle': (bool(pre_cycles) and all(
            c in by_r and _hex(by_t[c]) == _hex(by_r[c]) for c in pre_cycles)),
        'first_differing_cycle': first_diff, 'first_tail_cycle': first_window,
        'first_difference_at_first_tail_cycle': first_diff == first_window,
        'difference_at_first_differing_cycle': (by_t[first_diff] - by_r[first_diff]) if first_diff else None,
    }
    return {'bar': bar, 'bitwise_before_tail': bitwise}


# ----------------------------------------------------------------------------------------------------------------------
#  freeze v31
# ----------------------------------------------------------------------------------------------------------------------
def _find_v31():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V31_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    return rel, _sha(rel)


def preconditions():
    failures = []
    v30 = _load(SPEC_V30['path'])
    if _sha(SPEC_V30['path']) != SPEC_V30['sha256']:
        failures.append('v30 sha256 differs from bb6703da')
    if H.sha256_file(os.path.abspath(W.__file__)) != v30['script_sha256']:
        failures.append('the imported W87 harness differs from the one v30 pinned')
    if H.sha256_file(os.path.abspath(L.__file__)) != v30['launcher_sha256']:
        failures.append('the imported W86 launcher differs from the one v30 pinned')
    if _sha(W87_OUT['path']) != W87_OUT['sha256']:
        failures.append('reeval_w87.json sha256 differs from its pin')
    if _load(W87_MANIFEST).get(W87_OUT['path']) != W87_OUT['sha256']:
        failures.append('reeval_w87.json pin differs from the W87 manifest')
    for rel in (SPEC_V30['path'], W87_OUT['path'], W87_MANIFEST, os.path.abspath(W.__file__), os.path.abspath(L.__file__)):
        st = _git_state(os.path.relpath(rel, REPO) if os.path.isabs(rel) else rel)
        if not (st['git_tracked'] and st['git_clean']):
            failures.append(f'{rel} not committed / not clean')
    na = na_source_check()
    if not (na['declaration_line_present'] and na['join_present'] and na['git_tracked'] and na['git_clean']):
        failures.append(f'network.py not-applicable declaration / join not found or file not committed-clean: {na}')
    evidence, more = W.evidence_base()
    failures += more
    return failures, evidence, na


def v31_content(evidence, na):
    v30 = _load(SPEC_V30['path'])
    return {
        'schema': 'p515_frozen_spec_v31', 'version': 31, 'stage': STAGE_TEXT,
        'authority': ['Planner task W88 item 1 (rulings 1 and 2; re-evaluate from persisted records; zero solves)',
                      'PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7'],
        'predecessor': {'path': SPEC_V30['path'], 'sha256': _sha(SPEC_V30['path'])},
        'predecessor_not_edited': 'v30 stays as frozen; v31 strengthens one of its per-entry gates',
        'reason': REASON,
        'per_entry_gates': {
            **{k: v for k, v in v30['per_entry_gates'].items()
               if k not in ('G6_floor_records_tail_window', 'unchanged_from_v29')},
            'G6_floor_records_v31': G6_V31,
            'unchanged_from_v30': ('G1-G5, G7-G9, applies_to, comparison: verbatim from v30 (= v29); computed by the '
                                   'W86 launcher\'s own evaluation_checks through the committed W87 harness (by import)'),
        },
        'v30_G6_verbatim_superseded': v30['per_entry_gates']['G6_floor_records_tail_window'],
        'not_applicable_declaration_source': na,
        'fallback_test_operational': v30['fallback_test_operational'],
        'fallback_test_note': ('verbatim from v30 / v29; "fails to certify" counts a failed per-entry gate, now with G6 '
                               'under v31; "moves materially" (|Q_tail - Q_ref| > bar_ref) unchanged'),
        'self_tests_declared': SELF_TESTS,
        'caveats_travelling_with_the_numbers': CAVEATS,
        'integrity_checks': ('before any new quantity is read: the evidence base hashes equal v30\'s pins and the '
                             'campaign manifest; the committed W87 harness re-evaluates and its per-cell results, R, '
                             'fallback and integrity equal the committed reeval_w87.json (1ccdef2a) exactly; the '
                             'self-tests give their declared verdicts; any mismatch -> exit 1'),
        'references': v30['references'], 'reference_R': v30['reference_R'],
        'predictions_recorded_before_the_reevaluation': PREDICTIONS,
        'evidence_base': {**evidence, 'w87_reevaluation': {**W87_OUT, **_git_state(W87_OUT['path'])},
                          'spec_v30': {**SPEC_V30, **_git_state(SPEC_V30['path'])},
                          'w87_harness_sha256': H.sha256_file(os.path.abspath(W.__file__))},
        'output': {'root': OUT_ROOT, 'file': OUT_FILE, 'manifest': OUT_MANIFEST,
                   'note': 'a NEW root; nothing committed is re-run onto or modified'},
        'solve_claim': ('ZERO SOLVES, guard-verified: SolveProfileGuard(permitted=()) armed at import before any '
                        'project import, plus the imported W87 and W86-launcher permitted=() guards; all three '
                        'verify(0) == [] on every exit path'),
        'script': SCRIPT_NAME, 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'launcher_sha256': H.sha256_file(os.path.abspath(L.__file__)),
        'git_head_at_freeze': H._git(['rev-parse', 'HEAD']), 'frozen_utc': _utc(),
    }


def freeze_spec():
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V31_PREFIX))
    if existing:
        _log(f'[W88-V31 PRECONDITION FAILED] v31 already exists (write-once): {existing}')
        _finish(1)
    failures, evidence, na = preconditions()
    if failures:
        for f in failures:
            _log(f'[W88-V31 PRECONDITION FAILED] {f}')
        _finish(1)
    content = v31_content(evidence, na)
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V31_PREFIX}{sha[:8]}.json')
    with open(_abs(rel), 'x') as handle:
        handle.write(text)
    if _sha(rel) != sha:
        raise RuntimeError('v31 written bytes do not hash to the name')
    _log(f'[W88-V31] {STAGE_TEXT}')
    _log(f"[W88-V31] wrote {rel} sha256={sha} (predecessor {content['predecessor']})")
    _log(f"[W88-V31] G6 v31: applicable {G6_V31['predicate_applicable_record']}")
    _log(f"[W88-V31] G6 v31: not applicable {G6_V31['predicate_not_applicable_record']}")
    _log(f"[W88-V31] G6 v31: non-vacuity {G6_V31['non_vacuity']}")
    _finish(0, f'-- run with --run --spec-sha256 {sha}')


# ----------------------------------------------------------------------------------------------------------------------
#  item 1 -- run
# ----------------------------------------------------------------------------------------------------------------------
W87_COMPARED_PER_CELL_KEYS = ('gates_v30', 'gates_v30_pass', 'fallback_test_v30', 'fallback_test_w86_as_run', 'status',
                              'cycles_run', 'certification_cycle', 'eval_dir', 'eval_key', 'candidate_key', 'g6',
                              'comparison', 'retry_comparison', 'bar_window', 'scores')


def run(spec_sha256, started):
    tag = 'W88-REEVAL'
    failures = []
    v31_rel, v31_sha = _find_v31()
    if v31_rel is None or v31_sha != spec_sha256 or not os.path.basename(v31_rel).startswith(
            f'{SPEC_V31_PREFIX}{spec_sha256[:8]}'):
        failures.append(f'v31 not found / sha mismatch: {v31_rel} {v31_sha} vs {spec_sha256}')
    else:
        failures += [f'v31 {k} False' for k, v in _git_state(v31_rel).items() if not v]
    if os.path.exists(_abs(OUT_ROOT)):
        failures.append(f'output root exists (write-once): {OUT_ROOT}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    v31 = _load(v31_rel)
    if v31['script_sha256'] != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script changed since v31 froze')
    if v31['launcher_sha256'] != H.sha256_file(os.path.abspath(L.__file__)):
        failures.append('the W86 launcher changed since v31 froze')
    more, evidence, na = preconditions()
    failures += more
    for label in LABELS:
        for f, v in evidence['cells'][label]['files'].items():
            if v['sha256'] != v31['evidence_base']['cells'][label]['files'][f]['sha256']:
                failures.append(f'{label} {f}: sha256 != v31 pin')
    if evidence['campaign_results']['sha256'] != v31['evidence_base']['campaign_results']['sha256']:
        failures.append('campaign_results sha256 != v31 pin')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)

    # --- integrity 1: W87 reproduced exactly by its own committed code ---
    body = W.reevaluate()
    w87 = _load(W87_OUT['path'])
    mine = _roundtrip({k: {kk: body['per_cell'][k][kk] for kk in W87_COMPARED_PER_CELL_KEYS} for k in LABELS})
    theirs = {k: {kk: w87['per_cell'][k][kk] for kk in W87_COMPARED_PER_CELL_KEYS} for k in LABELS}
    integrity = {
        'w87_per_cell_equal': {k: mine[k] == theirs[k] for k in LABELS},
        'w87_R_equal': _roundtrip(body['R']) == w87['R'],
        'w87_fallback_cells_equal': body['fallback_cells'] == w87['fallback_cells_v30'],
        'w87_integrity_ok_reproduced': body['integrity_ok'] is True and w87['integrity_ok'] is True,
        'w87_integrity_detail_equal': _roundtrip(body['integrity']) == w87['integrity_reproduces_w86'],
    }
    # --- integrity 2: self-tests ---
    st, st_ok = self_tests()
    integrity['self_tests_all_as_declared'] = st_ok
    integrity_ok = (all(integrity['w87_per_cell_equal'].values()) and integrity['w87_R_equal']
                    and integrity['w87_fallback_cells_equal'] and integrity['w87_integrity_ok_reproduced']
                    and integrity['w87_integrity_detail_equal'] and st_ok)
    _log(f'[{tag}] integrity (W87 reproduced, self-tests): {integrity_ok} {integrity}')

    entries = W._cells_from_recert_spec()
    refs = v31['references']
    per_cell = {}
    for label in LABELS:
        p87 = body['per_cell'][label]
        eval_dir = os.path.join(W.RECERT_ROOT, 'evals', entries[label]['eval_dir'])
        rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
        g6 = g6_v31_evaluate(eval_dir, rec)
        gates = {k: v for k, v in p87['gates_v30'].items() if k != 'G6_floor_records_tail_window'}
        gates['G6_floor_records_v31'] = g6['gate_pass']
        cmp = p87['comparison']
        fails = rec.get('status') != 'certified' or not all(gates.values())
        moves = cmp['dQ'] is not None and abs(cmp['dQ']) > refs[label]['bar']
        first_window = min(g6['window_W']) if g6['window_W'] else None
        per_cell[label] = {
            'gates_v31': gates, 'gates_v31_pass': all(gates.values()),
            'G6_v30_as_reevaluated_by_w87': p87['gates_v30']['G6_floor_records_tail_window'],
            'fallback_test_v31': {'fails_to_certify': fails, 'moves_materially': moves, 'triggered': fails or moves},
            'fallback_test_v30_w87': p87['fallback_test_v30'],
            'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
            'certification_cycle': rec.get('certification_cycle'), 'eval_dir': eval_dir,
            'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
            'g6_v31': g6,
            'comparison_objective_convention': cmp.get('objective_convention'),
            'comparison': cmp,
            'caveats_measured': caveats_measured(eval_dir, refs[label]['eval_dir'], first_window),
        }
    fallback_cells = [k for k in LABELS if per_cell[k]['fallback_test_v31']['triggered']]
    g = guards_verify()
    results = {
        'stage': STAGE_TEXT, 'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'spec_v31': {'path': v31_rel, 'sha256': v31_sha}, 'spec_v30': SPEC_V30, 'evidence_base': evidence,
        'not_applicable_declaration_source': na,
        'objective_convention': 'Q gross_operational_cost (settlement excluded) on every table',
        'g6_v31_definition': G6_V31, 'caveats': CAVEATS,
        'integrity': integrity, 'integrity_ok': integrity_ok, 'self_tests': st,
        'per_cell': per_cell, 'all_gates_pass_v31': all(per_cell[k]['gates_v31_pass'] for k in LABELS),
        'fallback_triggered_v31': bool(fallback_cells), 'fallback_cells_v31': fallback_cells,
        'R_from_w87_reproduced': body['R'],
        'solve_claim': 'ZERO SOLVES, guard-verified (three guards permitted=(), verify(0))',
        'guards': g, 'wall_s': time.time() - started,
    }
    os.makedirs(_abs(OUT_ROOT))
    H._write_once_json(_abs(os.path.join(OUT_ROOT, OUT_FILE)), results)
    man = {os.path.join(OUT_ROOT, OUT_FILE): _sha(os.path.join(OUT_ROOT, OUT_FILE))}
    H._write_once_json(_abs(os.path.join(OUT_ROOT, OUT_MANIFEST)), man)
    _log(f'[{tag}] {STAGE_TEXT}; v31 {v31_sha}')
    for t in st:
        _log(f"[{tag}] self-test {t['id']} ok={t['ok']} v31={t.get('observed_v31_failures')} "
             f"v30_fails={t.get('observed_v30_fails')} non_vacuous={t.get('observed_non_vacuous')}")
    for k in LABELS:
        p = per_cell[k]
        gg = p['g6_v31']
        _log(f"[{tag}] {k}: status {p['status']} cert {p['certification_cycle']} gates_v31 {p['gates_v31']} "
             f"fallback_v31 {p['fallback_test_v31']}")
        _log(f"[{tag}]   G6 v31: W {gg['window_W'][0]}-{gg['window_W'][-1]} T {gg['terminal_round_T']} population "
             f"{gg['population_n']} bad {gg['n_bad']} non_vacuity {gg['non_vacuity']}; terminal {gg['terminal_round']['n']} "
             f"records, applicable {gg['terminal_round']['n_applicable']} {gg['terminal_round']['applicable_floor_status']}, "
             f"not applicable {gg['terminal_round']['n_not_applicable']}; window-before-T not applicable "
             f"{gg['window_rounds_before_T']['n_not_applicable']}; pre-tail not applicable "
             f"{gg['pre_tail_reported_not_gated']['n_not_applicable']} (rounds "
             f"{gg['pre_tail_reported_not_gated']['not_applicable_rounds']})")
        cv = p['caveats_measured']
        _log(f"[{tag}]   caveat bar: {cv['bar']}")
        _log(f"[{tag}]   caveat bitwise: {cv['bitwise_before_tail']}")
    _log(f'[{tag}] fallback v31 {fallback_cells}; all gates pass {results["all_gates_pass_v31"]}')
    _log(f'[{tag}] wrote {os.path.join(OUT_ROOT, OUT_FILE)} sha256 {man[os.path.join(OUT_ROOT, OUT_FILE)]}')
    _finish(0 if integrity_ok else 1, f'integrity_ok={integrity_ok} wall={time.time() - started:.1f}s')


# ----------------------------------------------------------------------------------------------------------------------
#  item 2 -- the s47_recert identity look
# ----------------------------------------------------------------------------------------------------------------------
S47_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert')
S47_SPEC = {'path': os.path.join(S47_ROOT, 'campaign_spec_s47_recert_902f93aa.json'),
            'sha256': '902f93aa22093455daa50927bdfa28bc885b2c5491151d211dc943bb3de86ee4'}
S47_MANIFEST = os.path.join(S47_ROOT, 'campaign_manifest_sha256.json')
S47_CELLS = {'c_star': os.path.join(S47_ROOT, 'evals', '070f833e1e318f85_c_star'),
             'n7_4h_e1': os.path.join(S47_ROOT, 'evals', 'bd504ecf5a288d44_n7_4h_e1')}
S47_CELL_FILES = ('evaluation_record.json', 'network_failures_s39_D.jsonl', 'esso_recovery_events_s39_D.jsonl',
                  'per_cycle_record.jsonl', 'launch.json')
GATES_PY = 'p515_g_g1_g4_admm_gates.py'
FORMULA_FIX_COMMIT = 'cb545653'          # W35 (ii): "instance-derived solve count" (git log -S on GATES_PY)
S47_RESULTS_COMMIT = '79b99b59'          # the commit that added the s47_recert results
PRIOR_ART = {
    'harness_identity_checks': os.path.join('data', 'SRP1', 'Results', 'P515S50', 'harness_identity_checks',
                                            'harness_identity_checks.json'),
    'harness_identity_checks_post_item3': os.path.join('data', 'SRP1', 'Results', 'P515S50',
                                                       'harness_identity_checks_post_item3',
                                                       'harness_identity_checks.json'),
    'generalization_checks': os.path.join('data', 'SRP1', 'Results', 'P515S50', 'generalization_checks',
                                          'generalization_checks.json'),
    'unrecovered_failure_policy': os.path.join('data', 'SRP1', 'Results', 'P515S50', 'unrecovered_failure_policy.md'),
    'gap_closeout_w82': os.path.join(_P53, 'gap_closeout_w82', 'gap_closeout_w82.json'),
}
SEARCH_PATTERNS = ('s47_recert', '070f833e1e318f85', 'bd504ecf5a288d44', '902f93aa')
# Hand classification (the Worker's, W88) of every tracked file that names an s47_recert identifier AND contains
# 'identity_holds'. The run ASSERTS the observed intersection equals this key set exactly, so a new file is flagged.
CONSUMER_CLASSIFICATION = {
    'PLANNER_BRIEF_2026-09-13.md': 'instruction text ordering this look; no dependency',
    'data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/evaluation_record.json':
        'a DIFFERENT campaign (s47_a1a_baseline) sharing the eval key; its own flag (also False, same pre-W35 formula)',
    'data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/g_s39_D.json':
        'as above (the arm report of that campaign)',
    'data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/evaluation_record.json':
        'ORIGIN of the flag (False)',
    'data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json':
        'ORIGIN (arm report holding the same solve_profile)',
    'data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/evaluation_record.json':
        'ORIGIN of the flag (False)',
    'data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/g_s39_D.json':
        'ORIGIN (arm report holding the same solve_profile)',
    'data/SRP1/Results/P515S49/flex_price_gate/gate.json': 'its own arm\'s identity; names the s47 C* arm report as a path only',
    'data/SRP1/Results/P515S49/flex_price_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S49/memory_fix_gate/gate.json': 'its own arm\'s identity; names the s47 C* arm report as a path only',
    'data/SRP1/Results/P515S49/memory_fix_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S50/generalization_checks/generalization_checks.json':
        'READS the s47 C* flag and REQUIRES it to be FALSE (check B: recorded False AND per-event identity holds) -- '
        'depends on the flag being False, not True',
    'data/SRP1/Results/P515S50/generalization_gate/gate.json': 'its own arm\'s identity; s47 C* named as a path only',
    'data/SRP1/Results/P515S50/generalization_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S50/harness_identity_checks/harness_identity_checks.json':
        'RE-DERIVES the per-event identity for both s47 cells (holds: 4528 = 4488 + 40; 5764 = 5763 + 1); does not '
        'read the recorded flag',
    'data/SRP1/Results/P515S50/harness_identity_checks_launch.log': 'launch log of the above',
    'data/SRP1/Results/P515S50/harness_identity_checks_post_item3/harness_identity_checks.json': 'as above (re-run)',
    'data/SRP1/Results/P515S50/harness_identity_checks_post_item3_launch.log': 'launch log of the above',
    'data/SRP1/Results/P515S51/srp1_bitwise_gate/gate.json': 'its own arm\'s identity; s47 named as a path only',
    'data/SRP1/Results/P515S51/srp1_bitwise_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S52/campaign_s52_pilot_nopersist/campaign_results.json':
        'its own records\' solve reconciliation; s47 campaign_results named as an input path',
    'data/SRP1/Results/P515S52/campaign_s52_pilot_repro_nopersist/campaign_results.json': 'as above',
    'data/SRP1/Results/P515S52/srp1_bitwise_gate/gate.json': 'its own arm\'s identity; s47 named as a path only',
    'data/SRP1/Results/P515S52/srp1_bitwise_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S53/frozen_s53_spec_v29_9161ff00.json':
        'pins the s47 references (Q, bar, solve_profile_observed); its G5 text judges the W86 cells\' own flag, not '
        'the references\'',
    'data/SRP1/Results/P515S53/frozen_s53_spec_v30_bb6703da.json': 'as v29',
    'data/SRP1/Results/P515S53/gap_closeout_w82/gap_closeout_w82.json':
        'REPORTS the s47 unit flag (False) beside an INDEPENDENT log-banner count that holds (5764); not gated on it',
    'data/SRP1/Results/P515S53/srp1_bitwise_gate/gate.json': 'its own arm\'s identity; s47 named as a path only',
    'data/SRP1/Results/P515S53/srp1_bitwise_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S53/tight_tail_w83/srp1_bitwise_gate/gate.json': 'its own arm\'s identity; s47 named as a path only',
    'data/SRP1/Results/P515S53/tight_tail_w83/srp1_bitwise_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S53/tight_tail_w84/srp1_bitwise_gate/gate.json': 'its own arm\'s identity; s47 named as a path only',
    'data/SRP1/Results/P515S53/tight_tail_w84/srp1_bitwise_gate/gate.md': 'as gate.json',
    'data/SRP1/Results/P515S53/tight_tail_w86/campaign_s53_w86_tail_recert/campaign_results.json':
        'G5 on the W86 cells\' OWN flag (instance-derived, per-event); the s47 references enter only as Q / bar / '
        'observed counts',
    'data/SRP1/Results/P515S53/tight_tail_w86/campaign_s53_w86_tail_smoke/campaign_spec_s53_w86_tail_smoke_37266e01.json':
        'smoke spec; its own identity text',
    'data/SRP1/Results/P515S53/tight_tail_w86/campaign_s53_w86_tail_smoke/smoke_gate.json': 'smoke run\'s own flag',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_n7u_grad/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_n7u_us0p001/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_n7u_us0p002/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_n7u_us0p003/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_n7u_us0p005/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_x0_grad/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_x0_us0p001/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_x0_us0p002/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_x0_us0p003/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'data/SRP1/Results/P515S53/scaling_pin_w78/arms/s53w78_x0_us0p005/arm_result.json':
        'its own arm\'s identity; the s47 C* arm report / spec named as a comparator path only',
    'p515_s49_flex_price_gate.py': 'computes its own arm\'s identity; s47 C* named in the docstring',
    'p515_s49_memory_fix_gate.py': 'computes its own arm\'s identity; s47 C* arm report used as a comparator path',
    'p515_s50_generalization_checks.py': 'reads the s47 C* flag; requires it FALSE (see generalization_checks.json)',
    'p515_s50_generalization_gate.py': 'computes its own arm\'s identity; s47 C* named as a path',
    'p515_s50_harness_identity_checks.py': 're-derives the per-event identity for both s47 cells',
    'p515_s52_pilot_campaign.py': 'its own records\' reconciliation; s47 campaign_results named as an input',
    'p515_s53_w82_gap_closeout.py': 'reports the s47 unit flag beside an independent log-banner count (gates on the latter)',
    'p515_s53_w86_tail_recert_campaign.py': 'G5 on the W86 cells\' own flag; the s47 references read for Q / bar only',
}


# W88's own files (tracked once committed): classified by prefix, since the v31 file name holds its hash.
OWN_FILE_PREFIXES = {
    'p515_s53_w88_g6_floor_reeval.py': 'this W88 harness itself (names s47 identifiers and the flag to look at them)',
    os.path.join(_P53, SPEC_V31_PREFIX): 'spec v31: as v29 / v30 (reference values only; G5 text on the W86 cells)',
    os.path.join(_P53, 'g6_floor_w88'): 'W88 item 1 output / logs (reference values only)',
    os.path.join(_P53, 's47_identity_look_w88'): 'W88 item 2 output / logs (this look)',
}


def classify_path(f):
    if f in CONSUMER_CLASSIFICATION:
        return CONSUMER_CLASSIFICATION[f]
    for prefix, note in OWN_FILE_PREFIXES.items():
        if f.startswith(prefix):
            return note
    return None

# The dependency reading of the classification above, as explicit sets (every member is a classified file).
_S47E = 'data/SRP1/Results/P515S47/campaign_s47_recert/evals/'
ORIGIN_FILES = {_S47E + '070f833e1e318f85_c_star/evaluation_record.json', _S47E + '070f833e1e318f85_c_star/g_s39_D.json',
                _S47E + 'bd504ecf5a288d44_n7_4h_e1/evaluation_record.json', _S47E + 'bd504ecf5a288d44_n7_4h_e1/g_s39_D.json'}
REQUIRES_FLAG_TRUE = set()   # the Worker found none
REQUIRES_FLAG_FALSE = {'data/SRP1/Results/P515S50/generalization_checks/generalization_checks.json',
                       'p515_s50_generalization_checks.py'}
REPORTS_FLAG_NOT_GATED = {'data/SRP1/Results/P515S53/gap_closeout_w82/gap_closeout_w82.json',
                          'p515_s53_w82_gap_closeout.py'}
REDERIVES_INDEPENDENTLY = {'data/SRP1/Results/P515S50/harness_identity_checks/harness_identity_checks.json',
                           'data/SRP1/Results/P515S50/harness_identity_checks_launch.log',
                           'data/SRP1/Results/P515S50/harness_identity_checks_post_item3/harness_identity_checks.json',
                           'data/SRP1/Results/P515S50/harness_identity_checks_post_item3_launch.log',
                           'p515_s50_harness_identity_checks.py'}



def _formula_at(commit):
    """The `identity_holds` assignment of run_admm_arm in GATES_PY at `commit`, and whether it comes after the run."""
    rc, src = _git_rc(['show', f'{commit}:{GATES_PY}'])
    if rc != 0:
        return {'commit': commit, 'error': 'git show failed'}
    lines = src.splitlines()
    start = next((i for i, t in enumerate(lines) if t.startswith('def run_admm_arm(')), None)
    if start is None:
        return {'commit': commit, 'error': 'run_admm_arm not found'}
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('def ')), len(lines))
    flag = next((i for i in range(start, end) if "'identity_holds':" in lines[i]), None)
    run_call = [i for i in range(start, end) if 'planning.run_operational_planning(' in lines[i]]
    uninstall = [i for i in range(start, end) if lines[i].strip() == 'guard.uninstall()']
    return {'commit': commit, 'function': 'run_admm_arm', 'function_def_line': start + 1,
            'flag_line': (flag + 1) if flag is not None else None,
            'flag_text': [lines[j].strip() for j in range(flag, min(flag + 2, end))] if flag is not None else None,
            'run_operational_planning_call_lines': [i + 1 for i in run_call],
            'guard_uninstall_lines': [i + 1 for i in uninstall],
            'flag_computed_after_the_run_and_guard_uninstall': bool(
                flag is not None and run_call and uninstall and max(run_call) < flag and max(uninstall) < flag)}


def s47_identity_look(started):
    tag = 'W88-S47'
    failures = []
    if os.path.exists(_abs(S47_OUT_ROOT)):
        failures.append(f'output root exists (write-once): {S47_OUT_ROOT}')
    if _sha(S47_SPEC['path']) != S47_SPEC['sha256']:
        failures.append('s47_recert campaign spec sha256 != 902f93aa')
    manifest = _load(S47_MANIFEST)
    pins = {'campaign_spec': {**S47_SPEC, **_git_state(S47_SPEC['path'])},
            'campaign_manifest': {'path': S47_MANIFEST, 'sha256': _sha(S47_MANIFEST), **_git_state(S47_MANIFEST)},
            'cells': {}}
    for label, d in S47_CELLS.items():
        pins['cells'][label] = {}
        for f in S47_CELL_FILES:
            rel = os.path.join(d, f)
            pins['cells'][label][f] = {'sha256': _sha(rel), 'manifest_sha256': manifest.get(rel), **_git_state(rel)}
            if pins['cells'][label][f]['manifest_sha256'] is not None and \
                    pins['cells'][label][f]['manifest_sha256'] != pins['cells'][label][f]['sha256']:
                failures.append(f'{rel}: sha256 != campaign manifest')
            if not (pins['cells'][label][f]['git_tracked'] and pins['cells'][label][f]['git_clean']):
                failures.append(f'{rel}: not committed / not clean')
    for k, rel in PRIOR_ART.items():
        st = _git_state(rel)
        if not (st['git_tracked'] and st['git_clean']):
            failures.append(f'prior art {rel} not committed / clean')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    spec = _load(S47_SPEC['path'])
    git_head_at_run = spec.get('git_head')

    # --- 1. why the flag is False ---
    case = _load(os.path.join('data', 'SRP1', 'SRP1.json'))
    n_yd = len(case['Years']) * len(case['Days'])
    per_cycle = (1 + len(case['DistributionNetworks'])) * n_yd + len({dn['connection_node_id']
                                                                      for dn in case['DistributionNetworks']})
    cells = {}
    v29 = _load(W.SPEC_V29['path'])
    for label, d in S47_CELLS.items():
        rec = _load(os.path.join(d, 'evaluation_record.json'))
        sp = rec.get('solve_profile') or {}
        observed = (sp.get('observed') or {}).get('permitted_solve')
        k = rec.get('cycles_run')
        old_expected = 51 * k + 51
        ev = S.event_level_solve_reconciliation(rec, per_cycle * (k + 1))
        esso_bytes = os.path.getsize(_abs(os.path.join(d, 'esso_recovery_events_s39_D.jsonl')))
        events = _jsonl(os.path.join(d, 'network_failures_s39_D.jsonl'))
        cells[label] = {
            'eval_dir': d, 'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
            'status': rec.get('status'), 'cycles_run': k, 'certification_cycle': rec.get('certification_cycle'),
            'recorded_solve_profile_verbatim': sp,
            'recorded_solve_profile_keys': sorted(sp.keys()),
            'recorded_keys_are_the_pre_W35_schema': sorted(sp.keys()) == ['identity_holds', 'observed'],
            'pre_W35_formula': 'identity_holds = (permitted_solve == 51 * len(rows) + 51), len(rows) = cycles_run',
            'pre_W35_expected': old_expected, 'observed': observed,
            'pre_W35_formula_reproduces_recorded_flag': (observed == old_expected) == sp.get('identity_holds'),
            'shortfall_explained': observed - old_expected,
            'per_event_rule': {
                'definition': ('observed == solves_per_cycle x (cycles_run + 1) + sum over network-failure events of '
                               '[recovery_attempted] + [tier2_attempted]; unsupported if any ESSO recovery event or '
                               'indeterminate event (p515_s44_scale_measurement.event_level_solve_reconciliation, '
                               'called here)'),
                'solves_per_cycle_from_case_file': per_cycle,
                'solves_per_cycle_derivation': '(1 + n_dso) x n_years x n_days + n_esso_nodes from data/SRP1/SRP1.json',
                'case_file_sha256_now': _sha(os.path.join('data', 'SRP1', 'SRP1.json')),
                'case_file_sha256_pinned_by_s47_spec': (spec.get('configuration') or {}).get('case_file_sha256'),
                **ev,
                'holds': ev['expected'] is not None and observed == ev['expected'],
                'n_events_file_lines': len(events),
                'events': [{k2: e.get(k2) for k2 in ('cycle', 'network_name', 'year', 'day', 'class',
                                                     'recovery_attempted', 'tier2_attempted', 'primary_termination',
                                                     'termination')} for e in events],
                'esso_recovery_events_file_bytes': esso_bytes},
            'record_blocked_counts_zero': ((sp.get('observed') or {}).get('blocked_solve') == 0
                                           and (sp.get('observed') or {}).get('blocked_exec') == 0),
            'Q_gross_certified_cost': rec.get('certified_cost'),
            'Q_gross_equals_v29_reference_pin': rec.get('certified_cost') == v29['references'][label]['Q_gross'],
            'record_sha_equals_v29_pin': (_sha(os.path.join(d, 'evaluation_record.json'))
                                          == v29['references'][label]['files_sha256']['evaluation_record.json']),
        }
    formula_run = _formula_at(git_head_at_run)
    formula_fix_parent = _formula_at(f'{FORMULA_FIX_COMMIT}^')
    formula_fix = _formula_at(FORMULA_FIX_COMMIT)
    rc_anc1, _ = _git_rc(['merge-base', '--is-ancestor', S47_RESULTS_COMMIT, FORMULA_FIX_COMMIT])
    rc_anc2, _ = _git_rc(['merge-base', '--is-ancestor', git_head_at_run, S47_RESULTS_COMMIT])
    _, gates_log = _git_rc(['log', '--format=%h %aI %s', f'{git_head_at_run}..{S47_RESULTS_COMMIT}', '--', GATES_PY])
    _, harness_log = _git_rc(['log', '--format=%h %aI %s', f'{git_head_at_run}..{S47_RESULTS_COMMIT}', '--',
                              'p515_s44_campaign_harness.py'])
    _, harness_src = _git_rc(['show', f'{git_head_at_run}:p515_s44_campaign_harness.py'])
    launch = {label: _load(os.path.join(d, 'launch.json')) for label, d in S47_CELLS.items()}
    why = {
        'campaign_git_head_at_freeze': git_head_at_run,
        'formula_at_campaign_git_head': formula_run,
        'formula_at_parent_of_fix_commit': formula_fix_parent,
        'formula_at_fix_commit': formula_fix,
        'fix_commit': FORMULA_FIX_COMMIT, 's47_results_commit': S47_RESULTS_COMMIT,
        's47_results_commit_is_ancestor_of_fix_commit': rc_anc1 == 0,
        'campaign_git_head_is_ancestor_of_results_commit': rc_anc2 == 0,
        'gates_file_commits_between_campaign_head_and_results_commit': [x for x in gates_log.splitlines() if x],
        'harness_file_commits_between_campaign_head_and_results_commit': [x for x in harness_log.splitlines() if x],
        'harness_sha256_at_campaign_head': hashlib.sha256(harness_src.encode()).hexdigest(),
        'harness_sha256_pinned_by_s47_spec': (spec.get('harness') or {}).get('sha256'),
        'harness_at_campaign_head_mentions_identity_holds': 'identity_holds' in harness_src,
        'child_started_utc': {k: v.get('started_utc') for k, v in launch.items()},
        'scope_note': ('git fixes the COMMITTED gates file at the campaign head and shows no commit to it between that '
                       'head and the results commit; an uncommitted working-tree edit during the run cannot be excluded '
                       'by git alone -- the recorded solve_profile having EXACTLY the two keys of the committed '
                       'pre-W35 code (a post-W35 record, e.g. the W86 C* cell, carries the keys listed in '
                       'post_W35_record_solve_profile_keys) is the corroborating evidence'),
        'post_W35_record_solve_profile_keys': sorted((_load(os.path.join(
            W.RECERT_ROOT, 'evals', W._cells_from_recert_spec()['c_star']['eval_dir'],
            'evaluation_record.json')).get('solve_profile') or {}).keys()),
    }

    # --- 2. prior art (committed re-derivations) ---
    hic = _load(PRIOR_ART['harness_identity_checks'])
    hic2 = _load(PRIOR_ART['harness_identity_checks_post_item3'])
    gen = _load(PRIOR_ART['generalization_checks'])
    w82 = _load(PRIOR_ART['gap_closeout_w82'])
    prior = {
        'files': {k: {'path': rel, 'sha256': _sha(rel), **_git_state(rel)} for k, rel in PRIOR_ART.items()},
        'harness_identity_checks_s47_rows': [r for r in hic.get('records', []) if 's47_recert' in str(r.get('label'))],
        'harness_identity_checks_post_item3_s47_rows': [r for r in hic2.get('records', [])
                                                        if 's47_recert' in str(r.get('label'))],
        'generalization_checks_c_star': (gen.get('sections', {}).get('B_run_admm_arm_solve_count', {})
                                         .get('c_star_identity_now_holds_where_it_was_recorded_false')),
        'w82_unit_log_banner_count': (w82.get('item3_srp1_c2_baseline', {}).get('unit', {})
                                      .get('solve_count_identity')),
        'gates_file_comment_names_this_case': ('4528 observed against 4488 base, 34 tier-1 and 3 tier-2 recoveries'
                                               in open(_abs(GATES_PY)).read()),
    }

    # --- 3. what depends on the flag: the search, recorded ---
    a_args = ['grep', '-l'] + sum([['-e', p] for p in SEARCH_PATTERNS], [])
    _, a = _git_rc(a_args)
    _, b = _git_rc(['grep', '-l', 'identity_holds'])
    inter = sorted(set(a.split()) & set(b.split()))
    _, refs_out = _git_rc(['for-each-ref', '--format=%(refname:short)', 'refs/heads', 'refs/remotes'])
    head = H._git(['rev-parse', 'HEAD'])
    branch_search = []
    for ref in [x for x in refs_out.splitlines() if x]:
        _, ra = _git_rc(['grep', '-l'] + sum([['-e', p] for p in SEARCH_PATTERNS], []) + [ref])
        files_a = {x.split(':', 1)[1] for x in ra.split() if ':' in x}
        if not files_a:
            continue
        _, rb = _git_rc(['grep', '-l', 'identity_holds', ref])
        files_b = {x.split(':', 1)[1] for x in rb.split() if ':' in x}
        rc_a, _ = _git_rc(['merge-base', '--is-ancestor', ref, 'HEAD'])
        both = sorted(files_a & files_b)
        not_at_head = []
        for f in both:
            rc1, blob_ref = _git_rc(['rev-parse', f'{ref}:{f}'])
            rc2, blob_head = _git_rc(['rev-parse', f'HEAD:{f}'])
            if rc2 != 0 or blob_ref.strip() != blob_head.strip():
                not_at_head.append({'path': f, 'at_head': rc2 == 0})
        branch_search.append({'ref': ref, 'sha': H._git(['rev-parse', ref]), 'is_ancestor_of_HEAD': rc_a == 0,
                              'is_HEAD': H._git(['rev-parse', ref]) == head,
                              'n_files_naming_s47': len(files_a), 'n_files_also_identity_holds': len(both),
                              'intersection_files_differing_from_or_absent_at_HEAD': not_at_head})
    classified = {f: classify_path(f) for f in inter}
    search = {
        'patterns_naming_s47': list(SEARCH_PATTERNS), 'pattern_flag': 'identity_holds',
        'method': ('git grep -l over tracked files of the working tree (includes the two uncommitted modifications of '
                   'tracked files) for any s47 pattern, intersected with git grep -l identity_holds; every file of the '
                   'intersection hand-classified (CONSUMER_CLASSIFICATION, in this script); the same intersection '
                   'computed on every local and remote branch ref, recording which files differ from HEAD'),
        'covers': 'stage scripts (*.py incl. docstrings), stage artefacts (data/ tracked JSON / logs / md), reports '
                  '(*.md), specs; NOT untracked files (the run logs under data/SRP1/Results/P56A, pickles)',
        'intersection_at_worktree': inter, 'n_intersection': len(inter),
        'classification': classified,
        'intersection_equals_classified_set': (all(c is not None for c in classified.values())
                                               and set(CONSUMER_CLASSIFICATION) <= set(inter)),
        'unclassified': sorted(f for f, c in classified.items() if c is None),
        'classified_but_absent': sorted(set(CONSUMER_CLASSIFICATION) - set(inter)),
        'branches': branch_search,
        'dependency_sets_are_classified': all(f in CONSUMER_CLASSIFICATION for f in (
            ORIGIN_FILES | REQUIRES_FLAG_TRUE | REQUIRES_FLAG_FALSE | REPORTS_FLAG_NOT_GATED | REDERIVES_INDEPENDENTLY)),
        'origin_files': sorted(ORIGIN_FILES),
        'files_depending_on_flag_true': sorted(REQUIRES_FLAG_TRUE),
        'files_depending_on_flag_false': sorted(REQUIRES_FLAG_FALSE),
        'files_reporting_flag_not_gated': sorted(REPORTS_FLAG_NOT_GATED),
        'files_rederiving_the_count_independently': sorted(REDERIVES_INDEPENDENTLY),
        'all_other_classified_files': 'own-arm flags, reference values only, instruction text or a different campaign',
    }

    # --- 4. the reference value ---
    x0_rec = _load(os.path.join(v29['references']['x0']['eval_dir'], 'evaluation_record.json'))
    r_ref = x0_rec.get('certified_cost') - cells['n7_4h_e1']['Q_gross_certified_cost']
    reference = {
        'objective_convention': 'gross_operational_cost (settlement excluded)',
        'R_ref_definition': 'Q(x0, s48_x0_capture C2 record) - Q(n7_4h_e1, s47_recert) , Q = evaluation_record.certified_cost',
        'Q_x0': x0_rec.get('certified_cost'), 'Q_unit': cells['n7_4h_e1']['Q_gross_certified_cost'],
        'R_ref_recomputed': r_ref, 'R_ref_v29_pin': v29['reference_R']['R_ref'],
        'R_ref_equals_v29_pin_bitwise': _hex(r_ref) == _hex(v29['reference_R']['R_ref']),
        'flag_is_post_run_reporting': formula_run.get('flag_computed_after_the_run_and_guard_uninstall'),
        'count_reconciles_per_event_both_cells': all(c['per_event_rule']['holds'] for c in cells.values()),
        'no_blocked_solve_in_either_cell': all(c['record_blocked_counts_zero'] for c in cells.values()),
    }
    reference['affected'] = not (reference['R_ref_equals_v29_pin_bitwise'] and reference['flag_is_post_run_reporting']
                                 and reference['count_reconciles_per_event_both_cells']
                                 and reference['no_blocked_solve_in_either_cell'])

    g = guards_verify()
    results = {'stage': STAGE_TEXT + ' -- item 2', 'utc': _utc(), 'git_head': head, 'pins': pins,
               'cells': cells, 'why_false': why, 'prior_art': prior, 'dependency_search': search,
               'reference_value': reference, 'script': SCRIPT_NAME,
               'script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'solve_claim': 'ZERO SOLVES, guard-verified (three guards permitted=(), verify(0))', 'guards': g,
               'wall_s': time.time() - started}
    os.makedirs(_abs(S47_OUT_ROOT))
    H._write_once_json(_abs(os.path.join(S47_OUT_ROOT, S47_OUT_FILE)), results)
    man = {os.path.join(S47_OUT_ROOT, S47_OUT_FILE): _sha(os.path.join(S47_OUT_ROOT, S47_OUT_FILE))}
    H._write_once_json(_abs(os.path.join(S47_OUT_ROOT, S47_OUT_MANIFEST)), man)
    for label, c in cells.items():
        _log(f"[{tag}] {label}: recorded {c['recorded_solve_profile_verbatim']} keys pre-W35 "
             f"{c['recorded_keys_are_the_pre_W35_schema']}; pre-W35 expected {c['pre_W35_expected']} vs observed "
             f"{c['observed']} (reproduces flag {c['pre_W35_formula_reproduces_recorded_flag']}); per-event base "
             f"{c['per_event_rule']['base']} + retries {c['per_event_rule']['retry_solves_credited']} = "
             f"{c['per_event_rule']['expected']} holds {c['per_event_rule']['holds']} (supported "
             f"{c['per_event_rule']['supported']}, ESSO events file {c['per_event_rule']['esso_recovery_events_file_bytes']} B)")
    _log(f"[{tag}] formula at campaign head {why['campaign_git_head_at_freeze'][:8]}: {formula_run}")
    _log(f"[{tag}] results commit ancestor of fix {why['s47_results_commit_is_ancestor_of_fix_commit']}; gates commits "
         f"in run window {why['gates_file_commits_between_campaign_head_and_results_commit']}")
    _log(f"[{tag}] search: {search['n_intersection']} files; equals classified set "
         f"{search['intersection_equals_classified_set']}; unclassified {search['unclassified']}; depends on True "
         f"{search['files_depending_on_flag_true']}; depends on False {search['files_depending_on_flag_false']}")
    _log(f"[{tag}] branches with s47 files: {[(b['ref'], b['is_ancestor_of_HEAD'], len(b['intersection_files_differing_from_or_absent_at_HEAD'])) for b in branch_search]}")
    _log(f"[{tag}] reference: {reference}")
    _log(f'[{tag}] wrote {os.path.join(S47_OUT_ROOT, S47_OUT_FILE)} sha256 {man[os.path.join(S47_OUT_ROOT, S47_OUT_FILE)]}')
    ok = search['intersection_equals_classified_set'] and search['dependency_sets_are_classified']
    _finish(0 if ok else 1, f'classification_complete={ok} wall={time.time() - started:.1f}s')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--s47-identity-look', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    try:
        if args.freeze_spec:
            freeze_spec()
        elif args.s47_identity_look:
            s47_identity_look(started)
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
