"""
P5.15 Addendum 46 ruling 7, Planner task W89 step 1 -- G6 v32: the terminal round is judged on the FINAL ACCEPTED
ATTEMPT PER BLOCK. Frozen spec v32 (predecessor v31 cded3496, NOT edited); the W86 tight-tail re-certification
RE-EVALUATED from its PERSISTED records under v32. ZERO SOLVES.

WHY (Planner W89 step 1, from the W88 question 1). v31 ruling 1 judged EVERY applicable terminal-round record on floor
status, so a primary that hit max_iter and was then successfully retried would fail G6 although the block converged
properly. The gate's question is "did each block's terminal solve reach the floor", not "did every attempt".

RULING (Planner): judge the FINAL ACCEPTED attempt per block in T. Superseded attempts (a primary or tier-1 later
retried) are counted and reported, never judged on floor status. Non-vacuity preserved: exactly one judged attempt per
block in T and at least B judged applicable records (B = network blocks per round). Negative controls: a terminal round
whose final accepted attempt is `above` FAILS; one where only a superseded attempt is `above` PASSES; a block with no
accepted attempt in T FAILS.

THE DEFINITION (stated operationally; the frozen spec carries it verbatim, G6_V32 below):
  block      (round, network, year, day) of a persisted network_ipopt_solve_records.jsonl record.
  ladder     production's retry ladder (network.py `_run_smopf`): primary -> recovery (tier 1, only after a primary
             whose termination is recoverable) -> recovery_tier2 (only after a tier-1 attempt that did not succeed).
             A block's attempts, ordered by that ladder, must be EXACTLY a prefix of (primary, recovery, recovery_tier2)
             -- one attempt per label, no gap, no unknown label; anything else is `ladder_malformed`.
  final      the LAST attempt of the block's ladder -- the attempt whose result production keeps.
  superseded every other attempt of the block (a primary or tier-1 later retried).
  accepted   the attempt's persisted `exit` is one of ACCEPTED_EXITS = ('Optimal Solution Found.', 'Solved To Acceptable
             Level.'): the two IPOPT exits whose AMPL solve_result_num (0, 1) Pyomo's .sol reader maps to SolverStatus.ok
             + TerminationCondition.optimal -- the condition production's `helper_functions.solver_result_succeeded`
             accepts. Every other exit string, and a missing exit, is NOT accepted. Validated on the real cells against
             production's own log-based classification (network_failures_s39_D.jsonl) -- see ACCEPTANCE_CROSS_CHECK.
  final accepted attempt  the final attempt, when it is accepted. A block whose final attempt is not accepted has NO
             accepted attempt (production's ladder stops at the final attempt).

Everything else is v31's: the population (W u {T}), the per-record conditions on EVERY record of the population
(compl_inf_tol_in_force, options_list_agrees, parse_reason; ruling 2 for not-applicable tier-2 records), v30's
non-vacuity, G1-G5, G7-G9 and the fallback test including "moves materially" (|dQ| > bar_ref), all verbatim.

HOW. Files only. W88's harness is imported (and, through it, W87's and the W86 launcher) so every v31 quantity is
computed by the committed code that produced the committed W88 re-evaluation, and that re-evaluation is REPRODUCED
(compared against the committed reeval_w88.json) before any new quantity is read. SolveProfileGuard(permitted=()) is
armed at import BEFORE any project import; the imported W88, W87 and W86-launcher guards (all permitted=()) are
verified too: all four verify(0) == [] on every exit path.

MODES (repo root, canonical interpreter; attached, alone, both streams captured):
  --freeze-spec                   writes data/SRP1/Results/P515S53/frozen_s53_spec_v32_<sha8>.json (write-once, named
                                  by its sha256).
  --run --spec-sha256 S           -> data/SRP1/Results/P515S53/g6_final_attempt_w89/reeval_w89.json + manifest (NEW root)
Exit codes: 0 done (whatever the verdict), 1 a precondition / integrity / self-test / guard failure.

EXACT COMMANDS:
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w89_g6_final_attempt_reeval.py --freeze-spec \\
      > data/SRP1/Results/P515S53/g6_final_attempt_w89_freeze_spec_v32_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w89_g6_final_attempt_reeval.py --run --spec-sha256 <sha> \\
      > data/SRP1/Results/P515S53/g6_final_attempt_w89_launch.log 2>&1

The v32 evaluator (`g6_v32_evaluate_records`, `g6_v32_evaluate`) takes B (blocks per round) as an argument, so the
3 x 3 launcher (W89 step 2) applies the same gate with B = 80.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W89 G6 final-accepted-attempt gate (never solves)').install()

# Arms W88's GUARD, W87's GUARD and the W86 launcher's PARENT_GUARD (all permitted=()), at import.
import p515_s53_w88_g6_floor_reeval as M88  # noqa: E402

W = M88.W
L = M88.L
H = M88.H

SCRIPT_NAME = os.path.basename(__file__)
STAGE_TEXT = ('P5.15 Addendum 46 ruling 7, W89 step 1 -- G6 v32: the terminal round judged on the final accepted '
              'attempt per block (superseded attempts counted and reported, never judged on floor status); the W86 '
              'tight-tail re-certification re-evaluated from its persisted records under v32 (zero solves)')
_P53 = W._P53
SPEC_V31 = {'path': os.path.join(_P53, 'frozen_s53_spec_v31_cded3496.json'),
            'sha256': 'cded34965024b04fbd1d59275924f43788e2bbb0420e914d0d426f02c6d1c54b'}
SPEC_V32_PREFIX = 'frozen_s53_spec_v32_'
W88_OUT = {'path': os.path.join(M88.OUT_ROOT, M88.OUT_FILE),
           'sha256': 'cf0309aff5c9fbd96f2b73e9e83175241782cc30786d960e432089b663cc2e78'}
W88_MANIFEST = os.path.join(M88.OUT_ROOT, M88.OUT_MANIFEST)
W88_SCRIPT_SHA256 = 'df6488ebaef4d5808ce1814b23198997fed7f3230e280381a65201561f75bec8'   # = v31 script_sha256
OUT_ROOT = os.path.join(_P53, 'g6_final_attempt_w89')
OUT_FILE = 'reeval_w89.json'
OUT_MANIFEST = 'reeval_w89_manifest_sha256.json'
LABELS = W.LABELS
SRP1_BLOCKS_PER_ROUND = W.BLOCKS_PER_ROUND   # 48 = (1 TSO + 3 DSO) x 3 years x 4 days (SRP1)
TAIL_TOL = W.TAIL_TOL                        # 1e-6
LADDER = ('primary', 'recovery', 'recovery_tier2')
ACCEPTED_EXITS = ('Optimal Solution Found.', 'Solved To Acceptable Level.')
FAILURE_EVENTS_FILE = 'network_failures_s39_D.jsonl'
HELPER_PY = 'helper_functions.py'
ACCEPT_SOURCE_LINES = ('po.TerminationCondition.optimal,', "result.solver.status == po.SolverStatus.ok",
                       'and result.solver.termination_condition in accepted_termination_conditions')
LADDER_SOURCE_LINES = ("log_suffix='recovery')", "log_suffix='recovery_tier2')",
                       'if not solver_result_succeeded(recovery_result):',
                       'recovery_attempted = _is_recoverable_network_failure(primary_result, params)')
_NEVER = object()   # a round no record carries: v31's per-record predicate without its terminal floor clause

# ----------------------------------------------------------------------------------------------------------------------
#  G6 under v32 -- stated operationally
# ----------------------------------------------------------------------------------------------------------------------
G6_V32 = {
    'name': 'G6_floor_records_v32',
    'replaces': 'v31 per_entry_gates.G6_floor_records_v31',
    'B_blocks_per_round': ('B = (1 + n_dso) x n_years x n_days of the instance = the number of network blocks solved '
                           'per round (SRP1: 48; the 3 x 3 instance: 80); stated operationally, not as a literal'),
    'population_unchanged_from_v30': M88.G6_V31['population_unchanged_from_v30'],
    'applicability_unchanged_from_v31': M88.G6_V31['applicability'],
    'block': 'b(r) = (r.round, r.network, r.year, r.day)',
    'ladder': ('production\'s retry ladder (network.py _run_smopf): primary -> recovery (tier 1) -> recovery_tier2. A '
               'block\'s attempts, ordered by that ladder, must be EXACTLY a prefix of (primary, recovery, '
               'recovery_tier2): one attempt per label, no gap, no unknown label; otherwise the block fails '
               '"ladder_malformed"'),
    'final_attempt': 'the LAST attempt of the block\'s ladder (the attempt whose result production keeps)',
    'superseded_attempts': 'every other attempt of the block (a primary or a tier-1 attempt later retried)',
    'accepted': (f'r.exit in {list(ACCEPTED_EXITS)} -- the two IPOPT exits whose AMPL solve_result_num (0, 1) Pyomo\'s '
                 '.sol reader maps to SolverStatus.ok + TerminationCondition.optimal, i.e. what '
                 'helper_functions.solver_result_succeeded accepts; any other exit string, or none, is NOT accepted'),
    'final_accepted_attempt': ('the final attempt of the block, when it is accepted; a block whose final attempt is not '
                               'accepted has NO accepted attempt'),
    'predicate_every_record_of_P': ('v31\'s per-record predicate WITHOUT its terminal floor clause, on EVERY record of P '
                                    '(superseded attempts included): applicable -> compl_inf_tol_in_force == 1e-6 in W '
                                    'else production, options_list_agrees, parse_reason None; not applicable (ruling 2) '
                                    '-> the same tolerance / options conditions, no residual parse reason, declared '
                                    'class recovery_tier2 / adaptive [v31 verbatim]'),
    'predicate_terminal_block': ('for EVERY block of T: ladder well formed; no superseded attempt accepted (production '
                                 'retries only an attempt that did not succeed); the final attempt accepted '
                                 '("no_accepted_attempt" otherwise); and, if the final attempt is APPLICABLE, its '
                                 'floor_status == "at" ("final_accepted_attempt_floor_not_at" otherwise). A NOT '
                                 'applicable final attempt (tier-2, adaptive mu) has its floor test recorded NOT '
                                 'APPLICABLE and counted -- see non-vacuity [RULING, new]'),
    'superseded_in_T': 'counted and reported (attempt, exit, floor_status, mu_over_floor), never judged on floor status',
    'non_vacuity': ('v30 verbatim (P non-empty; every round of W u {T} holds exactly B primary-attempt records and >= B '
                    'records) AND T holds exactly B blocks AND every block of T has exactly one judged attempt AND the '
                    'judged applicable attempts of T number >= B [v32; replaces v31\'s ">= 48 applicable records in T", '
                    'which counted superseded attempts too]'),
    'passes_iff': 'non_vacuity holds AND every record of P satisfies its predicate AND every block of T satisfies its',
    'floor_status_definition': M88.G6_V31['floor_status_definition'],
    'not_judged': ('floor status of superseded attempts in T and of every record outside T (reported); exit status of '
                   'records outside T (reported); pre-tail records (reported evidence, as in v30)'),
    'design_consequence_recorded': (
        'non-vacuity requires >= B judged APPLICABLE attempts in T, so with exactly B blocks EVERY block\'s final '
        'attempt must be applicable: a block of T whose final accepted attempt is a tier-2 retry (mu_strategy '
        'adaptive; floor formula not applicable) makes G6 FAIL through non-vacuity -- its terminal solve cannot be '
        'shown at the floor. Ruling 2 still passes the record itself (it is judged on the conditions that apply). '
        'Self-test T6 declares exactly this outcome. None of the three SRP1 cells has a non-primary record in T (W88 '
        'reported tallies); stated here so the rule is not discovered later'),
    'scope_limit': ('the ruling-2 path (a not-applicable tier-2 record inside W u {T}) is exercised ONLY by self-tests '
                    '(T1-T6): none of the three SRP1 cells has a tier-2 retry in W or T (their tier-2 records are C* '
                    'rounds 11 / 13 / 23, pre-tail)'),
}
ACCEPTANCE_CROSS_CHECK = {
    'what': ('the acceptance rule above, applied to every block of EVERY round of a real cell, must reproduce '
             'production\'s own log-based failure-event classification (network_failures_s39_D.jsonl, written by '
             'p515_g_g1_g4_admm_gates from the child stdout): block with an event <-> ladder not (a single accepted '
             'primary); class per ladder: (primary not accepted) -> not_attempted; (primary, recovery accepted) -> '
             'recovered_tier1; (primary, recovery not accepted) -> unrecovered; (primary, recovery, tier2 accepted) -> '
             'recovered_tier2; (primary, recovery, tier2 not accepted) -> unrecovered. Event cycle "init" <-> round 0'),
    'role': ('INTEGRITY on the real cells (a disagreement means the record-derived definition does not match what '
             'production did -> exit 1); reported per cell. Not part of the gate predicate, which self-tests exercise '
             'on altered record populations that have no event file'),
    'known_blind_spot': ('production can discard an accepted IPOPT result when model.solutions.load_from raises; the '
                         'records cannot see that, the events file would (no success print) -- the cross-check is what '
                         'would expose it'),
}
REASON = (
    'v31 ruling 1 judged EVERY applicable terminal-round record on floor status, so a primary that hit max_iter above '
    'the floor and was then successfully retried would fail G6 although the block converged properly (the Worker\'s W88 '
    'question 1; C* alone had 40 retry solves, so the larger 3 x 3 instance could plausibly have one in its terminal '
    'round). The gate\'s question is "did each block\'s terminal solve reach the floor" (Planner W89 step 1). v32 '
    'judges the final accepted attempt per block, keeps every per-record condition on every record, and keeps '
    'non-vacuity at one judged attempt per block and >= B judged applicable records. No threshold is raised or relaxed '
    'elsewhere; the fallback test is unchanged.')

# Self-tests: the v32 gate on the REAL C* records file (4,264 records), deep-copied and altered, with the verdict
# declared here before the run. T1-T5 and T9 are per-record (W88's T1-T5 / T9 carried over); the others evaluate the
# WHOLE altered population through g6_v32_evaluate_records, so the block rules and non-vacuity are exercised end to end.
SELF_TESTS = [
    {'id': 'T1', 'level': 'record', 'base': 'c_star tier-2 record (round 11)',
     'transform': 'round -> min(W); compl_inf_tol_in_force -> 1e-6', 'expect': [],
     'why': 'W88 T1: a legitimate adaptive-mu retry inside the window passes the per-record predicate'},
    {'id': 'T2', 'level': 'record', 'base': 'c_star tier-2 record (round 11)',
     'transform': 'round -> min(W); compl_inf_tol_in_force kept 1e-4', 'expect': ['compl_inf_tol_in_force'],
     'why': 'W88 T2: the tolerance condition still applies to a tier-2 record'},
    {'id': 'T3', 'level': 'record', 'base': 'c_star tier-2 record (round 11)',
     'transform': 'round -> min(W); compl_inf_tol_in_force -> 1e-6; options_list_agrees -> False',
     'expect': ['options_list_agrees'], 'why': 'W88 T3: the options-list condition still applies'},
    {'id': 'T4', 'level': 'record', 'base': 'c_star tier-2 record (round 11)',
     'transform': ('round -> min(W); compl_inf_tol_in_force -> 1e-6; parse_reason += "; no IPOPT options list precedes '
                   'the banner in the attempt segment"'),
     'expect': ['parse_reason_beyond_not_applicable'], 'why': 'W88 T4: another parse problem is not excused'},
    {'id': 'T5', 'level': 'record', 'base': 'c_star tier-2 record (round 11)',
     'transform': 'round -> min(W); compl_inf_tol_in_force -> 1e-6; attempt -> "primary"',
     'expect': ['not_applicable_outside_declared_class'],
     'why': 'W88 T5: a not-applicable declaration outside the tier-2 / adaptive class fails'},
    {'id': 'T6', 'level': 'population', 'base': 'c_star records; the round-11 case33_3/2035/Winter ladder',
     'transform': ('the whole round-11 ladder of that block (primary max_iter above, recovery max_iter above, tier-2 '
                   'Optimal, floor not applicable) transplanted into T in place of the block\'s T primary, round -> T, '
                   'compl_inf_tol_in_force -> 1e-6'),
     'expect_pass': False, 'expect_bad_records': 0, 'expect_bad_blocks': 0,
     'expect_non_vacuity_components_false': ['terminal_judged_applicable_ge_B'],
     'why': ('RECORDED CONSEQUENCE of ">= B judged applicable": a final accepted tier-2 attempt in T cannot be shown at '
             'the floor, so G6 fails through non-vacuity only; every record (tier-2 included, ruling 2) passes')},
    {'id': 'T7', 'level': 'population', 'base': 'c_star records', 'transform': 'none', 'expect_pass': True,
     'expect_bad_records': 0, 'expect_bad_blocks': 0, 'expect_non_vacuity_components_false': [],
     'why': 'positive control: the real cell passes'},
    {'id': 'T8', 'level': 'population', 'base': 'c_star records',
     'transform': 'one T primary (no retry) floor_status -> "above", mu_over_floor -> 7.0', 'expect_pass': False,
     'expect_bad_records': 0, 'expect_block_failures': ['final_accepted_attempt_floor_not_at'],
     'expect_non_vacuity_components_false': [],
     'why': 'W88 T8 at block level: the final (and only) accepted attempt above the floor fails'},
    {'id': 'T9', 'level': 'record', 'base': 'c_star window primary above the floor (round 82, max_iter)',
     'transform': 'none', 'expect': [],
     'why': 'W88 T9: floor status outside T is reported, not judged'},
    {'id': 'T10', 'level': 'count', 'base': 'non-vacuity on a synthetic count',
     'transform': 'B blocks in T, one judged attempt each, B - 1 of them applicable', 'expect_non_vacuous': False,
     'why': 'fewer than B judged applicable attempts in T fails as loudly as a bad record'},
    {'id': 'T11', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': ('PLANNER NEGATIVE CONTROL (a): the T primary -> exit max_iter, floor "above" (mu_over_floor 7.0); a '
                   'recovery attempt appended (copy of it: attempt recovery, exit Optimal, floor "above", '
                   'mu_over_floor 3.0)'),
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['final_accepted_attempt_floor_not_at'],
     'expect_non_vacuity_components_false': [], 'expect_v31_fails': True,
     'why': 'the FINAL ACCEPTED attempt is above the floor: FAIL'},
    {'id': 'T12', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': ('PLANNER NEGATIVE CONTROL (b): the T primary -> exit max_iter, floor "above" (mu_over_floor 7.0); a '
                   'recovery attempt appended (attempt recovery, exit Optimal, floor "at", mu_over_floor 1.0)'),
     'expect_pass': True, 'expect_bad_records': 0, 'expect_bad_blocks': 0, 'expect_non_vacuity_components_false': [],
     'expect_v31_fails': True,
     'why': ('only a SUPERSEDED attempt is above the floor: PASS under v32 (v31 FAILED it -- the defect W89 fixes); the '
             'superseded attempt is counted and reported')},
    {'id': 'T13', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': ('PLANNER NEGATIVE CONTROL (c): the T primary -> exit max_iter above; recovery appended (exit max_iter, '
                   'above); tier-2 appended (copy of the real round-11 tier-2 record: round T, same block, '
                   'compl_inf_tol_in_force 1e-6, exit -> "Maximum Number of Iterations Exceeded.")'),
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['no_accepted_attempt'],
     'expect_non_vacuity_components_false': ['terminal_judged_applicable_ge_B'],
     'why': 'a block of T with NO accepted attempt fails (and its judged attempt, a tier-2, is not applicable)'},
    {'id': 'T14', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': 'the T primary -> exit "Converged to a point of local infeasibility. Problem may be infeasible." (no retry)',
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['no_accepted_attempt'],
     'expect_non_vacuity_components_false': [],
     'why': 'a single-attempt block of T whose only attempt was not accepted fails (no accepted attempt)'},
    {'id': 'T15', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': 'a recovery attempt appended after an ACCEPTED primary (exit Optimal, floor at on both)',
     'expect_pass': False, 'expect_bad_records': 0, 'expect_block_failures': ['superseded_attempt_accepted'],
     'expect_non_vacuity_components_false': [],
     'why': 'production never retries an accepted solve: a superseded accepted attempt means the ladder is not production\'s'},
    {'id': 'T16', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': 'the T primary duplicated (two primary records for one block)', 'expect_pass': False,
     'expect_bad_records': 0, 'expect_block_failures': ['ladder_malformed', 'superseded_attempt_accepted'],
     'expect_non_vacuity_components_false': ['v30_part'],
     'why': ('a malformed ladder fails (the first, accepted, primary is superseded by its duplicate; and the round '
             'holds B + 1 primaries: v30 non-vacuity fails too)')},
    {'id': 'T17', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': 'the T primary deleted (T holds B - 1 blocks)', 'expect_pass': False, 'expect_bad_records': 0,
     'expect_bad_blocks': 0,
     'expect_non_vacuity_components_false': ['v30_part', 'terminal_blocks_eq_B', 'terminal_judged_applicable_ge_B'],
     'why': 'a missing block fails non-vacuity -- the fix cannot hollow the gate out'},
    {'id': 'T18', 'level': 'population', 'base': 'c_star records; one T primary',
     'transform': ('as T12 (superseded primary above, accepted recovery at) but the SUPERSEDED primary\'s '
                   'compl_inf_tol_in_force -> 1e-4'),
     'expect_pass': False, 'expect_bad_records': 1, 'expect_bad_blocks': 0, 'expect_non_vacuity_components_false': [],
     'why': 'superseded attempts are excused only from the floor test: every other per-record condition still applies'},
]

CAVEATS = dict(M88.CAVEATS)   # measured in W88 (reeval_w88.json caveats_measured); reproduced here, not re-derived

PREDICTIONS = {
    'recorded': (
        'BEFORE the recorded re-evaluation runs, and NOT BLIND. Seen first: the committed W88 re-evaluation '
        '(reeval_w88.json) including its REPORTED terminal-round tallies (48 primary records per cell, all "at", 0 not '
        'applicable, 0 non-primary records in T) and window tallies (0 tier-2 records in any W). Before the freeze the '
        'Worker ALSO ran an exploratory read of the three cells\' persisted records (all rounds): ladder shapes -- '
        'c_star 4,187 (primary) / 34 (primary, recovery) / 3 (primary, recovery, recovery_tier2); n7_4h_e1 5,423 / 1 / '
        '0; x0 6,384 / 0 / 0 -- exit tallies per attempt, the ladders of c_star rounds 5 / 11 / 13 / 23, and the '
        'failure-event class counts (c_star 34 recovered_tier1 + 3 recovered_tier2; unit 1 recovered_tier1; x0 none). '
        'That read shows the acceptance cross-check\'s COUNTS agree; the block-by-block cross-check and the per-cell '
        'v32 G6 were NOT executed on the n7_4h_e1 / x0 records before the freeze. Pre-freeze runs otherwise: '
        'py_compile and the self-tests (which read the c_star records file) through a scratchpad import writing '
        'nothing to the repository -- and self-test T7 IS the unaltered c_star population through the v32 gate, so '
        'the c_star v32 G6 verdict (PASS) WAS SEEN before the freeze. The first self-test run failed T18 (2 bad '
        'records, declared 1): the test CONSTRUCTION copied the recovery from the primary after altering the '
        'primary\'s tolerance; the construction was corrected to the declared transform (alter the superseded '
        'primary only), the expectation was not changed; all 18 then as declared. These predictions are a record of '
        'expectation from reported tallies, not a test of an unknown outcome.'),
    'P1_G6_v32': ('PASS on all three cells: population 434 / 432 / 432 (unchanged); T = 87 / 112 / 132 holds exactly 48 '
                  'blocks, 48 judged attempts (all primary, all accepted "Optimal Solution Found.", all applicable, all '
                  '"at"), 0 superseded attempts; 0 bad records, 0 bad blocks; non-vacuous'),
    'P2_cross_check': ('the acceptance rule reproduces production\'s failure-event classification block for block on '
                       'all three cells (c_star 37 events, unit 1, x0 0; 0 mismatches)'),
    'P3_other_gates': 'G1-G5, G7-G9 True on all three cells, identical to W88 / W87 / W86',
    'P4_fallback': 'NOT triggered on any cell (all certified, every v32 gate True, |dQ| < bar_ref)',
    'P5_reproduction': 'W88\'s committed per-cell results, self-tests and R reproduced exactly',
    'P6_self_tests': 'every self-test verdict as declared (T1-T18)',
    'P7_v31_v32_agree_on_real_cells': ('v31 and v32 give the same G6 verdict (PASS) on all three real cells, because '
                                       'no T block has a superseded attempt'),
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


def _roundtrip(obj):
    return json.loads(json.dumps(obj, default=H._json_default))


def guards_verify():
    return {'w89_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)},
            'imported_w88_guard': {'counts': dict(M88.GUARD.counts), 'verify_0_failures': M88.GUARD.verify(0)},
            'imported_w87_guard': {'counts': dict(W.GUARD.counts), 'verify_0_failures': W.GUARD.verify(0)},
            'imported_launcher_guard': {'counts': dict(L.PARENT_GUARD.counts),
                                        'verify_0_failures': L.PARENT_GUARD.verify(0)}}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    """EVERY exit path: verify all four guards at exactly 0, uninstall (LIFO), exit (1 if a guard fails)."""
    g = guards_verify()
    _log(f'[W89] guards {g} {extra_msg}')
    L.PARENT_GUARD.uninstall()
    W.GUARD.uninstall()
    M88.GUARD.uninstall()
    GUARD.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


# ----------------------------------------------------------------------------------------------------------------------
#  the v32 predicate
# ----------------------------------------------------------------------------------------------------------------------
def source_checks():
    """The production lines the definition rests on (the acceptance condition and the ladder), asserted present."""
    with open(_abs(HELPER_PY)) as handle:
        helper = handle.read()
    with open(_abs(M88.NETWORK_PY)) as handle:
        net = handle.read()
    return {'helper_functions': {'path': HELPER_PY, 'sha256': _sha(HELPER_PY), **_git_state(HELPER_PY),
                                 'lines': {s: s in helper for s in ACCEPT_SOURCE_LINES}},
            'network': {'path': M88.NETWORK_PY, 'sha256': _sha(M88.NETWORK_PY), **_git_state(M88.NETWORK_PY),
                        'lines': {s: s in net for s in LADDER_SOURCE_LINES}}}


def _source_ok(sc):
    return all(all(v['lines'].values()) and v['git_tracked'] and v['git_clean'] for v in sc.values())


def block_of(r):
    return (r.get('round'), r.get('network'), r.get('year'), r.get('day'))


def accepted(r):
    return r.get('exit') in ACCEPTED_EXITS


def per_record_failures(r, window, prod):
    """v31's per-record predicate without its terminal floor clause (the floor test moves to the block level)."""
    return M88.v31_failures(r, window, _NEVER, prod)


def judge_block(attempts):
    """attempts: the records of ONE block of T, any order. Returns (final, superseded, failures)."""
    labels = [a.get('attempt') for a in attempts]
    known = all(lab in LADDER for lab in labels)
    ordered = sorted(attempts, key=lambda a: LADDER.index(a.get('attempt'))) if known else list(attempts)
    ordered_labels = [a.get('attempt') for a in ordered]
    failures = []
    if not known or ordered_labels != list(LADDER[:len(ordered_labels)]):
        failures.append('ladder_malformed')
    final = ordered[-1]
    superseded = ordered[:-1]
    if any(accepted(a) for a in superseded):
        failures.append('superseded_attempt_accepted')
    if not accepted(final):
        failures.append('no_accepted_attempt')
    if M88.classify(final)[0] and final.get('floor_status') != 'at':
        failures.append('final_accepted_attempt_floor_not_at')
    return final, superseded, failures


def non_vacuous_v32(per_round, n_blocks_terminal, judged_per_block, n_judged_applicable, population_n, b):
    v30_part = bool(population_n) and all(v['n_primary'] == b and v['n'] >= b for v in per_round.values())
    comp = {'v30_part': v30_part, 'terminal_blocks_eq_B': n_blocks_terminal == b,
            'one_judged_attempt_per_terminal_block': bool(judged_per_block) and all(
                n == 1 for n in judged_per_block.values()),
            'terminal_judged_applicable_ge_B': n_judged_applicable >= b}
    comp['holds'] = all(comp.values())
    return comp


def _brief(r):
    return W._brief(r)


def _short(r):
    return {k: r.get(k) for k in ('round', 'network', 'year', 'day', 'attempt', 'exit', 'floor_status', 'mu_over_floor',
                                  'compl_inf_tol_in_force')}


def g6_v32_evaluate_records(records, window, terminal, prod, b):
    """G6 v32 on a records list (the persisted file, or an altered copy for a self-test)."""
    judged_rounds = set(window) | {terminal}
    pop = [r for r in records if r.get('round') in judged_rounds]
    per_round = {k: {'n': sum(1 for r in pop if r.get('round') == k),
                     'n_primary': sum(1 for r in pop if r.get('round') == k and r.get('attempt') == 'primary')}
                 for k in sorted(judged_rounds)}
    bad_records = []
    for r in pop:
        f = per_record_failures(r, window, prod)
        if f:
            bad_records.append({**_brief(r), 'failures': f})
    term = [r for r in pop if r.get('round') == terminal]
    by_block = defaultdict(list)
    for r in term:
        by_block[block_of(r)].append(r)
    bad_blocks, finals, superseded_all, judged_per_block = [], [], [], {}
    for key in sorted(by_block, key=lambda k: tuple(str(x) for x in k)):
        final, superseded, failures = judge_block(by_block[key])
        judged_per_block[key] = 1
        finals.append(final)
        superseded_all += superseded
        if failures:
            bad_blocks.append({'block': list(key), 'failures': failures,
                               'ladder': [_short(a) for a in by_block[key]]})
    judged_app = [f for f in finals if M88.classify(f)[0]]
    judged_na = [f for f in finals if not M88.classify(f)[0]]
    nv = non_vacuous_v32(per_round, len(by_block), judged_per_block, len(judged_app), len(pop), b)
    floor_recomputed = [((f.get('mu_over_floor') is not None and abs(f['mu_over_floor'] - 1.0) <= 1e-3)
                         == (f.get('floor_status') == 'at')) for f in judged_app]
    pre = [r for r in records if r.get('round') not in judged_rounds]
    win_not_t = [r for r in pop if r.get('round') != terminal]
    return {
        'gate_pass': nv['holds'] and not bad_records and not bad_blocks,
        'B_blocks_per_round': b, 'window_W': sorted(window), 'terminal_round_T': terminal,
        'judged_rounds': sorted(judged_rounds), 'production_compl_inf_tol': prod, 'n_records_total': len(records),
        'population_n': len(pop), 'per_round_counts': per_round, 'non_vacuity': nv,
        'n_bad_records': len(bad_records), 'bad_records': bad_records, 'n_bad_blocks': len(bad_blocks),
        'bad_blocks': bad_blocks,
        'terminal_round': {
            'n_records': len(term), 'n_blocks': len(by_block), 'n_judged': len(finals),
            'n_judged_applicable': len(judged_app), 'n_judged_not_applicable': len(judged_na),
            'judged_attempt_labels': dict(sorted(Counter(f.get('attempt') for f in finals).items())),
            'judged_exit': dict(sorted(Counter(str(f.get('exit')) for f in finals).items())),
            'judged_applicable_floor_status': dict(sorted(Counter(str(f.get('floor_status')) for f in judged_app).items())),
            'judged_not_applicable': [_brief(f) for f in judged_na],
            'judged_floor_status_recomputed_from_mu_over_floor_agrees': all(floor_recomputed),
            'judged_mu_over_floor_range_applicable': (
                [min(f['mu_over_floor'] for f in judged_app), max(f['mu_over_floor'] for f in judged_app)]
                if judged_app and all(f.get('mu_over_floor') is not None for f in judged_app) else None),
            'superseded_counted_reported_never_judged_on_floor': {
                'n': len(superseded_all),
                'attempts': dict(sorted(Counter(s.get('attempt') for s in superseded_all).items())),
                'floor_status': dict(sorted(Counter(str(s.get('floor_status')) for s in superseded_all).items())),
                'exit': dict(sorted(Counter(str(s.get('exit')) for s in superseded_all).items())),
                'records': [_short(s) for s in superseded_all]},
            'ladders': dict(sorted(Counter(tuple(sorted((a.get('attempt') for a in v), key=lambda x: (
                LADDER.index(x) if x in LADDER else 99))) for v in by_block.values()).items())) if by_block else {},
        },
        'window_rounds_before_T_reported': {
            'n': len(win_not_t), 'n_not_applicable': sum(1 for r in win_not_t if not M88.classify(r)[0]),
            'attempts': dict(sorted(Counter(r.get('attempt') for r in win_not_t).items())),
            'floor_status_applicable': dict(sorted(Counter(
                str(r.get('floor_status')) for r in win_not_t if M88.classify(r)[0]).items()))},
        'pre_tail_reported_not_gated': {
            'n': len(pre), 'n_not_applicable': sum(1 for r in pre if not M88.classify(r)[0]),
            'attempts': dict(sorted(Counter(r.get('attempt') for r in pre).items()))},
    }


def _ladders_json(g):
    g = dict(g)
    t = dict(g['terminal_round'])
    t['ladders'] = {'|'.join(k): v for k, v in t['ladders'].items()}
    g['terminal_round'] = t
    return g


def cell_inputs(eval_dir, rec):
    records = _jsonl(os.path.join(eval_dir, 'network_ipopt_solve_records.jsonl'))
    ts = _load(os.path.join(eval_dir, 'convergence_depth_tail_state.json'))
    window = {p['cycle'] for p in (ts.get('per_cycle') or []) if p.get('active')}
    return records, window, rec.get('cycles_run'), L._production_compl_inf_tol(ts.get('baseline'))


def g6_v32_evaluate(eval_dir, rec, b):
    records, window, terminal, prod = cell_inputs(eval_dir, rec)
    return _ladders_json(g6_v32_evaluate_records(records, window, terminal, prod, b))


def expected_event_class(attempts):
    """The class production's event parser assigns, derived from the records by the acceptance rule (None: no event)."""
    ordered = sorted(attempts, key=lambda a: LADDER.index(a.get('attempt')) if a.get('attempt') in LADDER else 99)
    labels = tuple(a.get('attempt') for a in ordered)
    ok = accepted(ordered[-1])
    if labels == ('primary',):
        return None if ok else 'not_attempted'
    if labels == ('primary', 'recovery'):
        return 'recovered_tier1' if ok else 'unrecovered'
    if labels == ('primary', 'recovery', 'recovery_tier2'):
        return 'recovered_tier2' if ok else 'unrecovered'
    return f'malformed:{labels}'


def acceptance_cross_check(eval_dir):
    """ACCEPTANCE_CROSS_CHECK on one real cell, every round."""
    records = _jsonl(os.path.join(eval_dir, 'network_ipopt_solve_records.jsonl'))
    events = _jsonl(os.path.join(eval_dir, FAILURE_EVENTS_FILE))
    by_block = defaultdict(list)
    for r in records:
        by_block[block_of(r)].append(r)
    expected = {k: expected_event_class(v) for k, v in by_block.items()}
    expected = {k: v for k, v in expected.items() if v is not None}
    observed, dup = {}, []
    for e in events:
        if e.get('record_type') != 'network_block':
            continue
        rnd = 0 if e.get('cycle') == 'init' else int(e.get('cycle'))
        key = (rnd, e.get('network_name'), int(e.get('year')), e.get('day'))
        if key in observed:
            dup.append(list(key))
        observed[key] = e.get('class')
    mismatches = [{'block': list(k), 'from_records': expected.get(k), 'production_event': observed.get(k)}
                  for k in sorted(set(expected) | set(observed), key=lambda k: tuple(str(x) for x in k))
                  if expected.get(k) != observed.get(k)]
    return {'events_file': os.path.join(eval_dir, FAILURE_EVENTS_FILE),
            'n_events': len(events), 'n_blocks_with_expected_event': len(expected), 'n_observed': len(observed),
            'expected_class_counts': dict(sorted(Counter(expected.values()).items())),
            'observed_class_counts': dict(sorted(Counter(observed.values()).items())),
            'duplicate_event_blocks': dup, 'n_mismatches': len(mismatches), 'mismatches': mismatches[:20],
            'agrees': not mismatches and not dup and all(e.get('record_type') == 'network_block' for e in events)}


# ----------------------------------------------------------------------------------------------------------------------
#  self-tests
# ----------------------------------------------------------------------------------------------------------------------
def self_tests():
    """The declared self-tests on real C* records (deep copies). Returns (results, all_ok)."""
    entries = W._cells_from_recert_spec()
    d = os.path.join(W.RECERT_ROOT, 'evals', entries['c_star']['eval_dir'])
    rec = _load(os.path.join(d, 'evaluation_record.json'))
    records, window, terminal, prod = cell_inputs(d, rec)
    b = SRP1_BLOCKS_PER_ROUND
    t2 = [r for r in records if r.get('round') == 11 and r.get('attempt') == 'recovery_tier2']
    lad11 = [r for r in records if t2 and block_of(r) == block_of(t2[0])]
    tp_all = [r for r in records if r.get('round') == terminal and r.get('attempt') == 'primary'
              and r.get('floor_status') == 'at' and accepted(r)]
    w82 = [r for r in records if r.get('round') == 82 and r.get('attempt') == 'primary'
           and r.get('floor_status') == 'above']
    found = {'tier2_round11': len(t2), 'round11_ladder_len': len(lad11), 'terminal_primary_at_accepted': len(tp_all),
             'round82_primary_above': len(w82)}
    if len(t2) != 1 or len(lad11) != 3 or len(tp_all) != b or len(w82) != 1:
        return {'base_records_found': found}, False
    first = min(window)
    # the T primary altered in the block-level tests: the case33_3 / 2035 / Winter block (the tier-2 block's network)
    tgt_key = (terminal,) + block_of(t2[0])[1:]
    tgt = next(r for r in tp_all if block_of(r) == tgt_key)

    def replace(pop, old, new_list):
        out = []
        for r in pop:
            if r is old:
                out.extend(new_list)
            else:
                out.append(r)
        return out

    def rec_alter(tid, r):
        r = copy.deepcopy(r)
        if tid in ('T1', 'T2', 'T3', 'T4', 'T5'):
            r['round'] = first
        if tid in ('T1', 'T3', 'T4', 'T5'):
            r['compl_inf_tol_in_force'] = TAIL_TOL
        if tid == 'T3':
            r['options_list_agrees'] = False
        if tid == 'T4':
            r['parse_reason'] = r['parse_reason'] + '; no IPOPT options list precedes the banner in the attempt segment'
        if tid == 'T5':
            r['attempt'] = 'primary'
        return r

    def failed_primary(mu=7.0):
        p = copy.deepcopy(tgt)
        p['exit'] = 'Maximum Number of Iterations Exceeded.'
        p['floor_status'] = 'above'
        p['mu_over_floor'] = mu
        return p

    def retry(base, attempt, exit_, floor, mu):
        r = copy.deepcopy(base)
        r['attempt'] = attempt
        r['warm_start'] = False
        r['exit'] = exit_
        r['floor_status'] = floor
        r['mu_over_floor'] = mu
        return r

    def population(tid):
        pop = list(records)
        if tid == 'T6':
            moved = []
            for r in sorted(lad11, key=lambda a: LADDER.index(a['attempt'])):
                m = copy.deepcopy(r)
                m['round'] = terminal
                m['compl_inf_tol_in_force'] = TAIL_TOL
                moved.append(m)
            return replace(pop, tgt, moved)
        if tid == 'T7':
            return pop
        if tid == 'T8':
            p = copy.deepcopy(tgt)
            p['floor_status'] = 'above'
            p['mu_over_floor'] = 7.0
            return replace(pop, tgt, [p])
        if tid == 'T11':
            p = failed_primary()
            return replace(pop, tgt, [p, retry(p, 'recovery', 'Optimal Solution Found.', 'above', 3.0)])
        if tid in ('T12', 'T18'):
            p = failed_primary()
            r1 = retry(p, 'recovery', 'Optimal Solution Found.', 'at', 1.0)
            if tid == 'T18':   # the SUPERSEDED primary only (the recovery keeps the tolerance in force)
                p['compl_inf_tol_in_force'] = 1e-4
            return replace(pop, tgt, [p, r1])
        if tid == 'T13':
            p = failed_primary()
            r1 = retry(p, 'recovery', 'Maximum Number of Iterations Exceeded.', 'above', 5.0)
            r2 = copy.deepcopy(t2[0])
            r2['round'] = terminal
            r2['compl_inf_tol_in_force'] = TAIL_TOL
            r2['exit'] = 'Maximum Number of Iterations Exceeded.'
            return replace(pop, tgt, [p, r1, r2])
        if tid == 'T14':
            p = copy.deepcopy(tgt)
            p['exit'] = 'Converged to a point of local infeasibility. Problem may be infeasible.'
            return replace(pop, tgt, [p])
        if tid == 'T15':
            return replace(pop, tgt, [tgt, retry(tgt, 'recovery', 'Optimal Solution Found.', 'at', 1.0)])
        if tid == 'T16':
            return replace(pop, tgt, [tgt, copy.deepcopy(tgt)])
        if tid == 'T17':
            return replace(pop, tgt, [])
        raise KeyError(tid)

    out, ok = [], True
    for t in SELF_TESTS:
        tid = t['id']
        if t['level'] == 'record':
            base = w82[0] if tid == 'T9' else t2[0]
            r = rec_alter(tid, base)
            f = per_record_failures(r, window, prod)
            res = {'id': tid, 'level': 'record', 'tested_record': _brief(r), 'observed': f, 'expect': t['expect'],
                   'ok': f == t['expect']}
        elif t['level'] == 'count':
            per_round = {terminal: {'n': b, 'n_primary': b}}
            nv = non_vacuous_v32(per_round, b, {i: 1 for i in range(b)}, b - 1, b, b)
            res = {'id': tid, 'level': 'count', 'observed_non_vacuous': nv['holds'],
                   'expect_non_vacuous': t['expect_non_vacuous'], 'ok': nv['holds'] == t['expect_non_vacuous']}
        else:
            pop = population(tid)
            g = g6_v32_evaluate_records(pop, window, terminal, prod, b)
            nv_false = sorted(k for k, v in g['non_vacuity'].items() if k != 'holds' and not v)
            block_failures = sorted({f for bb in g['bad_blocks'] for f in bb['failures']})
            checks = {'pass': g['gate_pass'] == t['expect_pass'],
                      'bad_records': ('expect_bad_records' not in t or g['n_bad_records'] == t['expect_bad_records']),
                      'non_vacuity_components_false': nv_false == sorted(t['expect_non_vacuity_components_false'])}
            if 'expect_bad_blocks' in t:
                checks['bad_blocks'] = g['n_bad_blocks'] == t['expect_bad_blocks']
            if 'expect_block_failures' in t:
                checks['block_failures'] = block_failures == sorted(t['expect_block_failures'])
            v31 = None
            if 'expect_v31_fails' in t:   # v31's per-record gate on the same terminal round (the defect, shown)
                v31_bad = [r for r in pop if r.get('round') in (set(window) | {terminal})
                           and M88.v31_failures(r, window, terminal, prod)]
                v31 = {'n_bad': len(v31_bad), 'fails': bool(v31_bad),
                       'failures': sorted({f for r in v31_bad for f in M88.v31_failures(r, window, terminal, prod)})}
                checks['v31_verdict'] = v31['fails'] == t['expect_v31_fails']
            res = {'id': tid, 'level': 'population', 'observed_pass': g['gate_pass'], 'expect_pass': t['expect_pass'],
                   'observed_n_bad_records': g['n_bad_records'], 'observed_n_bad_blocks': g['n_bad_blocks'],
                   'observed_block_failures': block_failures, 'observed_non_vacuity_false': nv_false,
                   'observed_terminal': {k: g['terminal_round'][k] for k in (
                       'n_blocks', 'n_judged', 'n_judged_applicable', 'n_judged_not_applicable',
                       'judged_attempt_labels', 'judged_applicable_floor_status')},
                   'observed_superseded': {k: g['terminal_round']['superseded_counted_reported_never_judged_on_floor'][k]
                                           for k in ('n', 'attempts', 'floor_status', 'exit')},
                   'bad_blocks': g['bad_blocks'], 'bad_records_brief': [
                       {k: x.get(k) for k in ('round', 'network', 'year', 'day', 'attempt', 'failures')}
                       for x in g['bad_records']],
                   'v31_on_same_population': v31, 'checks': checks, 'ok': all(checks.values())}
        ok = ok and res['ok']
        out.append(res)
    return out, ok


# ----------------------------------------------------------------------------------------------------------------------
#  freeze v32
# ----------------------------------------------------------------------------------------------------------------------
def _find_v32():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V32_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    return rel, _sha(rel)


def preconditions():
    failures = []
    if _sha(SPEC_V31['path']) != SPEC_V31['sha256']:
        failures.append('v31 sha256 differs from cded3496')
    v31 = _load(SPEC_V31['path'])
    if H.sha256_file(os.path.abspath(M88.__file__)) != v31['script_sha256'] or v31['script_sha256'] != W88_SCRIPT_SHA256:
        failures.append('the imported W88 harness differs from the one v31 pinned')
    if H.sha256_file(os.path.abspath(L.__file__)) != v31['launcher_sha256']:
        failures.append('the imported W86 launcher differs from the one v31 pinned')
    if _sha(W88_OUT['path']) != W88_OUT['sha256']:
        failures.append('reeval_w88.json sha256 differs from its pin')
    if _load(W88_MANIFEST).get(W88_OUT['path']) != W88_OUT['sha256']:
        failures.append('reeval_w88.json pin differs from the W88 manifest')
    for rel in (SPEC_V31['path'], W88_OUT['path'], W88_MANIFEST, os.path.abspath(M88.__file__),
                os.path.abspath(W.__file__), os.path.abspath(L.__file__)):
        st = _git_state(os.path.relpath(rel, REPO) if os.path.isabs(rel) else rel)
        if not (st['git_tracked'] and st['git_clean']):
            failures.append(f'{rel} not committed / not clean')
    sc = source_checks()
    if not _source_ok(sc):
        failures.append(f'production source lines of the definition not found / not committed-clean: {sc}')
    more_failures, evidence, na = M88.preconditions()
    failures += more_failures
    return failures, evidence, na, sc


def v32_content(evidence, na, sc):
    v31 = _load(SPEC_V31['path'])
    return {
        'schema': 'p515_frozen_spec_v32', 'version': 32, 'stage': STAGE_TEXT,
        'authority': ['Planner task W89 step 1 (judge the FINAL ACCEPTED attempt per block in T; superseded attempts '
                      'counted and reported, never judged; non-vacuity preserved; three negative controls; re-evaluate '
                      'from persisted records; zero solves)', 'PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7'],
        'predecessor': {'path': SPEC_V31['path'], 'sha256': _sha(SPEC_V31['path'])},
        'predecessor_not_edited': 'v31 stays as frozen; v32 replaces one of its per-entry gates',
        'reason': REASON,
        'per_entry_gates': {
            **{k: v for k, v in v31['per_entry_gates'].items()
               if k not in ('G6_floor_records_v31', 'unchanged_from_v30')},
            'G6_floor_records_v32': G6_V32,
            'unchanged_from_v31': ('G1-G5, G7-G9, applies_to, comparison: verbatim from v31 (= v30 = v29); computed by '
                                   'the W86 launcher\'s evaluation_checks through the committed W87 harness (by import, '
                                   'via the committed W88 harness)'),
        },
        'v31_G6_verbatim_superseded': v31['per_entry_gates']['G6_floor_records_v31'],
        'acceptance_cross_check': ACCEPTANCE_CROSS_CHECK,
        'production_source_lines': sc,
        'not_applicable_declaration_source': na,
        'fallback_test_operational': v31['fallback_test_operational'],
        'fallback_test_note': ('verbatim from v31 / v30 / v29; "fails to certify" counts a failed per-entry gate, now with '
                               'G6 under v32; "moves materially" (|Q_tail - Q_ref| > bar_ref) unchanged'),
        'self_tests_declared': SELF_TESTS,
        'caveats_travelling_with_the_numbers': CAVEATS,
        'scope_limit_ruling_2': G6_V32['scope_limit'],
        'integrity_checks': ('before any new quantity is read: the evidence base hashes equal the v31 pins and the '
                             'campaign manifest; the committed W88 code re-evaluates and its per-cell results, self-tests '
                             'and R equal the committed reeval_w88.json (cf0309af) exactly; the v32 self-tests give their '
                             'declared verdicts; the acceptance cross-check agrees on every real cell; any mismatch -> '
                             'exit 1'),
        'references': v31['references'], 'reference_R': v31['reference_R'],
        'predictions_recorded_before_the_reevaluation': PREDICTIONS,
        'evidence_base': {**evidence, 'w88_reevaluation': {**W88_OUT, **_git_state(W88_OUT['path'])},
                          'spec_v31': {**SPEC_V31, **_git_state(SPEC_V31['path'])},
                          'w88_harness_sha256': H.sha256_file(os.path.abspath(M88.__file__))},
        'output': {'root': OUT_ROOT, 'file': OUT_FILE, 'manifest': OUT_MANIFEST,
                   'note': 'a NEW root; nothing committed is re-run onto or modified'},
        'solve_claim': ('ZERO SOLVES, guard-verified: SolveProfileGuard(permitted=()) armed at import before any '
                        'project import, plus the imported W88, W87 and W86-launcher permitted=() guards; all four '
                        'verify(0) == [] on every exit path'),
        'script': SCRIPT_NAME, 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'w88_script_sha256': H.sha256_file(os.path.abspath(M88.__file__)),
        'launcher_sha256': H.sha256_file(os.path.abspath(L.__file__)),
        'git_head_at_freeze': H._git(['rev-parse', 'HEAD']), 'frozen_utc': _utc(),
    }


def freeze_spec():
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V32_PREFIX))
    if existing:
        _log(f'[W89-V32 PRECONDITION FAILED] v32 already exists (write-once): {existing}')
        _finish(1)
    failures, evidence, na, sc = preconditions()
    if failures:
        for f in failures:
            _log(f'[W89-V32 PRECONDITION FAILED] {f}')
        _finish(1)
    content = v32_content(evidence, na, sc)
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V32_PREFIX}{sha[:8]}.json')
    with open(_abs(rel), 'x') as handle:
        handle.write(text)
    if _sha(rel) != sha:
        raise RuntimeError('v32 written bytes do not hash to the name')
    _log(f'[W89-V32] {STAGE_TEXT}')
    _log(f"[W89-V32] wrote {rel} sha256={sha} (predecessor {content['predecessor']})")
    for k in ('final_attempt', 'accepted', 'predicate_terminal_block', 'non_vacuity'):
        _log(f'[W89-V32] G6 v32 {k}: {G6_V32[k]}')
    _finish(0, f'-- run with --run --spec-sha256 {sha}')


# ----------------------------------------------------------------------------------------------------------------------
#  run
# ----------------------------------------------------------------------------------------------------------------------
W88_COMPARED_PER_CELL_KEYS = ('gates_v31', 'gates_v31_pass', 'fallback_test_v31', 'status', 'cycles_run',
                              'certification_cycle', 'eval_key', 'candidate_key', 'g6_v31', 'comparison',
                              'caveats_measured')


def reproduce_w88():
    """W88's per-cell results, recomputed by the committed W88 / W87 code, against the committed reeval_w88.json."""
    body = W.reevaluate()
    entries = W._cells_from_recert_spec()
    refs = _load(SPEC_V31['path'])['references']
    mine = {}
    for label in LABELS:
        p87 = body['per_cell'][label]
        eval_dir = os.path.join(W.RECERT_ROOT, 'evals', entries[label]['eval_dir'])
        rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
        g6 = M88.g6_v31_evaluate(eval_dir, rec)
        gates = {k: v for k, v in p87['gates_v30'].items() if k != 'G6_floor_records_tail_window'}
        gates['G6_floor_records_v31'] = g6['gate_pass']
        cmp = p87['comparison']
        fails = rec.get('status') != 'certified' or not all(gates.values())
        moves = cmp['dQ'] is not None and abs(cmp['dQ']) > refs[label]['bar']
        first_window = min(g6['window_W']) if g6['window_W'] else None
        mine[label] = {'gates_v31': gates, 'gates_v31_pass': all(gates.values()),
                       'fallback_test_v31': {'fails_to_certify': fails, 'moves_materially': moves,
                                             'triggered': fails or moves},
                       'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                       'certification_cycle': rec.get('certification_cycle'), 'eval_key': rec.get('eval_key'),
                       'candidate_key': rec.get('candidate_key'), 'g6_v31': g6, 'comparison': cmp,
                       'caveats_measured': M88.caveats_measured(eval_dir, refs[label]['eval_dir'], first_window),
                       '_eval_dir': eval_dir, '_rec': rec}
    st88, st88_ok = M88.self_tests()
    w88 = _load(W88_OUT['path'])
    mine_rt = _roundtrip({k: {kk: mine[k][kk] for kk in W88_COMPARED_PER_CELL_KEYS} for k in LABELS})
    theirs = {k: {kk: w88['per_cell'][k][kk] for kk in W88_COMPARED_PER_CELL_KEYS} for k in LABELS}
    integrity = {
        'w88_per_cell_equal': {k: mine_rt[k] == theirs[k] for k in LABELS},
        'w88_self_tests_equal': _roundtrip(st88) == w88['self_tests'] and st88_ok is True,
        'w88_R_equal': _roundtrip(body['R']) == w88['R_from_w87_reproduced'],
        'w88_fallback_cells_equal': [k for k in LABELS if mine[k]['fallback_test_v31']['triggered']]
        == w88['fallback_cells_v31'],
        'w88_integrity_ok_committed': w88['integrity_ok'] is True,
    }
    ok = (all(integrity['w88_per_cell_equal'].values()) and integrity['w88_self_tests_equal']
          and integrity['w88_R_equal'] and integrity['w88_fallback_cells_equal'] and integrity['w88_integrity_ok_committed'])
    return mine, body, integrity, ok


def run(spec_sha256, started):
    tag = 'W89-REEVAL'
    failures = []
    v32_rel, v32_sha = _find_v32()
    if v32_rel is None or v32_sha != spec_sha256 or not os.path.basename(v32_rel).startswith(
            f'{SPEC_V32_PREFIX}{spec_sha256[:8]}'):
        failures.append(f'v32 not found / sha mismatch: {v32_rel} {v32_sha} vs {spec_sha256}')
    else:
        failures += [f'v32 {k} False' for k, v in _git_state(v32_rel).items() if not v]
    if os.path.exists(_abs(OUT_ROOT)):
        failures.append(f'output root exists (write-once): {OUT_ROOT}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    v32 = _load(v32_rel)
    if v32['script_sha256'] != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script changed since v32 froze')
    if v32['launcher_sha256'] != H.sha256_file(os.path.abspath(L.__file__)):
        failures.append('the W86 launcher changed since v32 froze')
    if v32['w88_script_sha256'] != H.sha256_file(os.path.abspath(M88.__file__)):
        failures.append('the W88 harness changed since v32 froze')
    more, evidence, na, sc = preconditions()
    failures += more
    for label in LABELS:
        for f, v in evidence['cells'][label]['files'].items():
            if v['sha256'] != v32['evidence_base']['cells'][label]['files'][f]['sha256']:
                failures.append(f'{label} {f}: sha256 != v32 pin')
    if evidence['campaign_results']['sha256'] != v32['evidence_base']['campaign_results']['sha256']:
        failures.append('campaign_results sha256 != v32 pin')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)

    # --- integrity 1: W88 reproduced exactly by its own committed code ---
    mine, body, integrity, rep_ok = reproduce_w88()
    # --- integrity 2: self-tests ---
    st, st_ok = self_tests()
    integrity['self_tests_all_as_declared'] = st_ok
    # --- integrity 3: the acceptance rule against production's events, every round of every real cell ---
    xcheck = {label: acceptance_cross_check(mine[label]['_eval_dir']) for label in LABELS}
    integrity['acceptance_cross_check_agrees'] = {label: xcheck[label]['agrees'] for label in LABELS}
    integrity_ok = rep_ok and st_ok and all(integrity['acceptance_cross_check_agrees'].values())
    _log(f'[{tag}] integrity (W88 reproduced, self-tests, cross-check): {integrity_ok} {integrity}')

    refs = v32['references']
    per_cell = {}
    for label in LABELS:
        m = mine[label]
        g6 = g6_v32_evaluate(m['_eval_dir'], m['_rec'], SRP1_BLOCKS_PER_ROUND)
        gates = {k: v for k, v in m['gates_v31'].items() if k != 'G6_floor_records_v31'}
        gates['G6_floor_records_v32'] = g6['gate_pass']
        cmp = m['comparison']
        fails = m['status'] != 'certified' or not all(gates.values())
        moves = cmp['dQ'] is not None and abs(cmp['dQ']) > refs[label]['bar']
        per_cell[label] = {
            'gates_v32': gates, 'gates_v32_pass': all(gates.values()),
            'G6_v31_as_reproduced': m['gates_v31']['G6_floor_records_v31'],
            'fallback_test_v32': {'fails_to_certify': fails, 'moves_materially': moves, 'triggered': fails or moves},
            'fallback_test_v31_w88': m['fallback_test_v31'],
            'status': m['status'], 'cycles_run': m['cycles_run'], 'certification_cycle': m['certification_cycle'],
            'eval_dir': m['_eval_dir'], 'eval_key': m['eval_key'], 'candidate_key': m['candidate_key'],
            'g6_v32': g6, 'acceptance_cross_check': xcheck[label],
            'comparison_objective_convention': cmp.get('objective_convention'), 'comparison': cmp,
            'caveats_measured_reproduced_from_w88': m['caveats_measured'],
        }
    fallback_cells = [k for k in LABELS if per_cell[k]['fallback_test_v32']['triggered']]
    g = guards_verify()
    results = {
        'stage': STAGE_TEXT, 'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'spec_v32': {'path': v32_rel, 'sha256': v32_sha}, 'spec_v31': SPEC_V31, 'evidence_base': evidence,
        'production_source_lines': sc, 'not_applicable_declaration_source': na,
        'objective_convention': 'Q gross_operational_cost (settlement excluded) on every table',
        'g6_v32_definition': G6_V32, 'acceptance_cross_check_definition': ACCEPTANCE_CROSS_CHECK, 'caveats': CAVEATS,
        'integrity': integrity, 'integrity_ok': integrity_ok, 'self_tests': st,
        'per_cell': per_cell, 'all_gates_pass_v32': all(per_cell[k]['gates_v32_pass'] for k in LABELS),
        'fallback_triggered_v32': bool(fallback_cells), 'fallback_cells_v32': fallback_cells,
        'R_from_w87_reproduced': body['R'],
        'solve_claim': 'ZERO SOLVES, guard-verified (four guards permitted=(), verify(0))',
        'guards': g, 'wall_s': time.time() - started,
    }
    os.makedirs(_abs(OUT_ROOT))
    H._write_once_json(_abs(os.path.join(OUT_ROOT, OUT_FILE)), results)
    man = {os.path.join(OUT_ROOT, OUT_FILE): _sha(os.path.join(OUT_ROOT, OUT_FILE))}
    H._write_once_json(_abs(os.path.join(OUT_ROOT, OUT_MANIFEST)), man)
    _log(f'[{tag}] {STAGE_TEXT}; v32 {v32_sha}')
    for t in st:
        _log(f"[{tag}] self-test {t['id']} ({t['level']}) ok={t['ok']} "
             + (f"observed={t['observed']}" if t['level'] == 'record' else
                f"non_vacuous={t['observed_non_vacuous']}" if t['level'] == 'count' else
                f"pass={t['observed_pass']} bad_records={t['observed_n_bad_records']} "
                f"block_failures={t['observed_block_failures']} nv_false={t['observed_non_vacuity_false']} "
                f"superseded={t['observed_superseded']['n']} v31={t['v31_on_same_population']}"))
    for k in LABELS:
        p = per_cell[k]
        gg = p['g6_v32']
        tr = gg['terminal_round']
        _log(f"[{tag}] {k}: status {p['status']} cert {p['certification_cycle']} gates_v32 {p['gates_v32']} "
             f"fallback_v32 {p['fallback_test_v32']}")
        _log(f"[{tag}]   G6 v32: W {gg['window_W'][0]}-{gg['window_W'][-1]} T {gg['terminal_round_T']} population "
             f"{gg['population_n']} bad records {gg['n_bad_records']} bad blocks {gg['n_bad_blocks']} non_vacuity "
             f"{gg['non_vacuity']}; T blocks {tr['n_blocks']} judged {tr['n_judged']} (applicable "
             f"{tr['n_judged_applicable']} {tr['judged_applicable_floor_status']}, labels {tr['judged_attempt_labels']}, "
             f"exits {tr['judged_exit']}), superseded {tr['superseded_counted_reported_never_judged_on_floor']['n']}")
        x = p['acceptance_cross_check']
        _log(f"[{tag}]   acceptance cross-check: agrees {x['agrees']} expected {x['expected_class_counts']} observed "
             f"{x['observed_class_counts']} mismatches {x['n_mismatches']}")
    _log(f'[{tag}] fallback v32 {fallback_cells}; all gates pass {results["all_gates_pass_v32"]}')
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
