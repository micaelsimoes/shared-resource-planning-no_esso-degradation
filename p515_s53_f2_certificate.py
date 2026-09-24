"""
P5.15 Addendum 40 ruling 4 (task W49) -- F2 CERTIFICATE CONTINUATION: the STANDARD UNIT POLL of Addendum 39
ruling 1 (frozen in spec v23 `ruling4_F2_certificate`) at the flexibility price m = 2 under the EUR 1M budget,
continued from the committed incumbent of the s51 F2 Phase B (campaign s51_f2_phase_b, spec 5ce295e1 / c39c4836,
run 0012d85b), which stopped for review when the box completion of that incumbent had 61 feasible points (> 30).

  MODEL VARIANT -- flexibility price x m (m = 2), a FLEXIBILITY-PRICE SCENARIO, never the baseline; m enters the
  evaluation key (p515_s51_f2_phase_b.make_key_of), so no m = 1 record can be a cache entry -- asserted by a scan.
  Ageing: the BASELINE (C2 + phi_cal 0.985 + soh_min 0.70), declared. AA-on case file. Cap 500, 10 cycles,
  concurrency 5, arm s39_D, no overrides, no post-certification. The budget I(x) <= 1e6 is a CONSTRAINT.

AUTHORITY: PLANNER_BRIEF_2026-09-13.md Addendum 39 ruling 1 and Addendum 40 ruling 4;
data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json `ruling4_F2_certificate`; Planner task W49.

================================================================================
WHAT IS REUSED, BY IMPORT (not re-implemented, not re-decided)
================================================================================
  p515_s47_phase_b_record (PB): Lattice (constraints, I(x), canonical form, `neighbourhood`, `completion`),
      householder_columns / project_direction (OrthoMADS H and the lattice rounding, ruling A5), resolution /
      classify (the improvement threshold max(bar_x + bar_inc, sigma_Q), ruling A4), sigma_q_from_tables (A6),
      degradation_clause, merge_cache, initial_incumbent (recorded only), the barrier stop thresholds (A7),
      COMPLETION_CAP = 30, POLL_RECORD_FIELDS / CANDIDATE_RECORD_FIELDS, rule_eleven.
  p515_s51_f2_phase_b (F51): the m = 2 evaluation key (make_key_of / eval_key_of_canonical / spec_label), the pins
      (check_pins), shared_constants, the pinned m = 2 cache sources and their loader (CACHE_SOURCES,
      load_cache_source), the spec-configuration test, the point-record reader (_point_result, flexibility-price
      and ageing read-back, rule ten, solve reconciliation, per-cycle trajectory), _cache_entry_from_point,
      memory_preflight, the guard stack.
  p515_s49_flex_ladder_campaign (L): rule_eleven (harness-side), case-file / memory-rule / solves-per-cycle checks.

================================================================================
WHAT IS NEW HERE (the poll of Addendum 39 ruling 1) -- `run_continuation`
================================================================================
PB.run_mads implements ruling A2 (directions UNION the full box completion at every unit poll, n+1 directions); it
cannot express the new poll, so the continuation loop is written here from PB's primitives and keeps PB's record
schema (every PB poll / candidate field, plus the snap fields). POLL_RULE, SNAP_RULE, ITERATION_RULE and
CERTIFICATE_STATEMENT below are frozen verbatim into the spec.

THE CACHE: every committed CERTIFIED m = 2 record -- s49 ladder (x = 0, E = 1 MWh), s50 marginal (E = 2), s51 F2
ladder (E = 3, 4, 5) through F51.load_cache_source, unchanged; and the s51 F2 Phase B's 20 evaluated points through
`load_phase_b_source` (the same point checks; the source is expected to have STOPPED FOR REVIEW on the completion
cap, which is why it is continued). Each pinned by sha256; eval keys recomputed at m = 2; Q values asserted equal
to pinned literals; the Phase B spec's frozen script sha256s asserted equal to the imported modules on disk.

TWO MODES, attached, both streams captured, never detached, one at a time (the campaign lock):
  --freeze                    ZERO SOLVES: preconditions, pins, cache, exclusions, incumbent, rule eleven, the
                              zero-solve dry run of the first poll (snap table, completion trigger, cache hits,
                              new evaluations), `freeze_campaign_spec` + validation of what is on disk.
  --run --spec-sha256 <sha>   re-checks everything plus the harness / case-file / ESS-params / script sha256s and
                              the memory preflight, takes the campaign lock, runs the continuation, writes
                              campaign_results.json + campaign_manifest_sha256.json; state after every poll.
The parent never solves: this module's SolveProfileGuard(permitted=()) sits on top of F51's, PB's, F's, M's, L's
and W25's; all seven verified at exactly 0.

Exit codes (--run): 0 poll failure at unit mesh with the certificate holding; 2 evaluation cap (60) / MAX_POLLS;
3 STOP_FOR_REVIEW (barrier stop rule or completion cap); 1 harness / guard / read-back / precondition failure, or
a poll failure whose certificate does not hold.

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_f2_certificate.py --freeze \\
      > data/SRP1/Results/P515S53/campaign_s53_f2_certificate_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_f2_certificate.py --run \\
      --spec-sha256 <sha> > data/SRP1/Results/P515S53/campaign_s53_f2_certificate_launch.log 2>&1
"""

import argparse
import glob
import inspect
import json
import math
import os
import re
import sys
import time

from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# F51 imports F (-> M -> L) and PB; each installs its parent guard (permitted=()) at import. Ours goes on top.
import p515_s51_f2_phase_b as F51  # noqa: E402
import p515_s47_phase_b_record as PB  # noqa: E402
import p515_s49_flex_ladder_campaign as L  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S53 F2 certificate parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

LABEL = F51.LABEL
FLEX_LABEL = F51.FLEX_LABEL
SCENARIO_LABEL = F51.SCENARIO_LABEL
STAGE = ('P5.15 Addendum 40 ruling 4 (W49) -- F2 CERTIFICATE CONTINUATION: the standard unit poll (2n OrthoMADS '
         'directions, snap-to-feasible, completion only if fewer than n + 1 feasible poll points) at the flexibility '
         'price m = 2 under the EUR 1M budget, from the committed s51 F2 Phase B incumbent')

_P51 = os.path.join('data', 'SRP1', 'Results', 'P515S51')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
DEFAULT_CAMPAIGN_ID = 's53_f2_certificate'
CAMPAIGN_ID_PATTERN = re.compile(r'^s53_f2_certificate(_r[0-9]+)?$')

M_FLEX = F51.M_FLEX
CASE_FILE_AA = dict(F51.CASE_FILE_AA)
ESS_AGEING_BASELINE = dict(F51.ESS_AGEING_BASELINE)
CAP = 500
CONCURRENCY = 5
REQUIRED_CONSECUTIVE_CYCLES = 10
ARM_LABEL = F51.ARM_LABEL

# ---- the poll of Addendum 39 ruling 1 ----
N_VARS = PB.N_VARS                         # 7
N_DIRECTIONS = 2 * N_VARS                  # 14: +-h_j
MIN_FEASIBLE_POLL_POINTS = N_VARS + 1      # 8: the completion is added only if FEWER than this remain
POLL_DESIGN = 'orthomads_2n'
DELTA_UNIT = PB.DELTA_MIN                  # 1: every poll of this continuation is a unit poll
COMPLETION_CAP = PB.COMPLETION_CAP         # 30: refuses, never truncates
MAX_NEW_EVALUATIONS = 60                   # Addendum 39 ruling 1 / spec v23: cap on NEW evaluations
MAX_POLLS = PB.MAX_POLLS                   # 60 (safety, A8)
FIRST_HALTON_K = 4                         # the s51 poll the completion cap refused (asserted, `continuation_facts`)

SPEC_V23 = {'path': os.path.join(_P53, 'frozen_s53_spec_v23_39a07fd8.json'),
            'sha256': '39a07fd85bd02aaee0f4042d9a7b201cf4df9027191ff70d7939b124061f685c', 'commit': '4ae7ff64',
            'item': 'ruling4_F2_certificate'}

# ---- the committed s51 F2 Phase B: the fourth cache source and the incumbent ----
_PB51 = os.path.join(_P51, 'campaign_s51_f2_phase_b')
PHASE_B_SOURCE = {
    'name': 's51_f2_phase_b',
    'results': {'path': os.path.join(_PB51, 'campaign_results.json'),
                'sha256': '2cdd5d6048919980b6d3d7109709b3441a99b5eb353be3285a09210a9a996eb7', 'commit': '0012d85b',
                'campaign': 's51_f2_phase_b (spec campaign_spec_s51_f2_phase_b_5ce295e1)'},
    'spec': {'path': os.path.join(_PB51, 'campaign_spec_s51_f2_phase_b_5ce295e1.json'),
             'sha256': '5ce295e17c5165ea2148d1240c918ebca05b8aafbeeb22e2ebfc03f9bf120094', 'commit': 'c39c4836'},
    'manifest': {'path': os.path.join(_PB51, 'campaign_manifest_sha256.json'),
                 'sha256': '2d6a41a2e134cf2e788514d2ac6ab407dce1b7ca1326b4f16e153580cb9d4939', 'commit': '0012d85b'},
    'script': {'path': 'p515_s51_f2_phase_b.py', 'commit': '2e2ef65d'},
    'what': 'the s51 F2 Phase B: its 20 evaluated points (the incumbent and the polled neighbours), m = 2',
}
# Asserted, never trusted from the task text: the committed Q (gross, settlement-excluded, m = 2) of every point.
Q_PHASE_B_EXPECTED = {
    'y2025__n7_p0.75_e2.5_m2': 810092671.8248721,
    'y2025__n7_p1_e2.5_m2': 810079546.5286092,
    'y2030__n5_p0.25_e0.5__n7_p0.75_e2.5__n9_p0.25_e0.5_m2': 809921746.4863336,
    'y2030__n5_p0.25_e0.5__n7_p0.75_e2.5_m2': 810078973.9269239,
    'y2030__n5_p0.25_e0.5__n7_p0.75_e3__n9_p0.25_e0.5_m2': 809806167.3731737,
    'y2030__n5_p0.25_e0.5__n7_p0.75_e3_m2': 809965545.0530776,
    'y2030__n5_p0.25_e0.5__n7_p1_e2.5__n9_p0.25_e0.5_m2': 809888874.9916775,
    'y2030__n5_p0.25_e0.5__n7_p1_e2.5_m2': 810052747.5041645,
    'y2030__n5_p0.25_e0.5__n7_p1_e3.5_m2': 809795490.1476665,
    'y2030__n5_p0.25_e0.5__n7_p1_e3_m2': 809916214.2848953,
    'y2030__n7_p0.75_e2.5__n9_p0.25_e0.5_m2': 810084663.4564126,
    'y2030__n7_p0.75_e2.5_m2': 810252769.3495697,
    'y2030__n7_p0.75_e3__n9_p0.25_e0.5_m2': 809972127.2115817,
    'y2030__n7_p0.75_e3_m2': 810139913.559661,
    'y2030__n7_p1_e2.5__n9_p0.25_e0.5_m2': 810058492.8240466,
    'y2030__n7_p1_e2.5_m2': 810219274.8398231,
    'y2030__n7_p1_e3.5__n9_p0.25_e0.5_m2': 809801074.2830775,
    'y2030__n7_p1_e3.5_m2': 809960133.409277,
    'y2030__n7_p1_e3__n9_p0.25_e0.5_m2': 809921835.7487029,
    'y2030__n7_p1_e3_m2': 810081212.8342348}
# The incumbent, from the committed final_incumbent; I and F are the task text's figures, asserted bitwise.
EXPECTED_INCUMBENT = {'label': 'y2030__n5_p0.25_e0.5__n7_p1_e3.5',
                      'eval_key': '5ca4f86c3406c0424d3b63b8f39df4568f1be77f7f2d72642491539c0eb42663',
                      'z': [1, 1, 4, 7, 0, 0, 1], 'I': 992268.1082843702, 'Q': 809795490.1476665,
                      'F': 810787758.2559508, 'bar': 2248.5940684080124}
EXPECTED_BUDGET_SLACK_EUR = 7731.891715629841
EXPECTED_S51_BOX_COMPLETION = 61
N_CACHE_EXPECTED = len(F51.Q_CACHE_EXPECTED) + len(Q_PHASE_B_EXPECTED)  # 6 + 20
BUDGET_INFEASIBLE_CACHE_LABELS = tuple(F51.BUDGET_INFEASIBLE_CACHE_LABELS)  # E = 4, 5 MWh at 2025
ALL_CACHE_SOURCE_PATHS = tuple(s['results']['path'] for s in F51.CACHE_SOURCES) + (PHASE_B_SOURCE['results']['path'],)
BASELINE_EXCLUSION_GLOB = F51.BASELINE_EXCLUSION_GLOB

OBJECTIVE_CONVENTION = F51.OBJECTIVE_CONVENTION
RESOLUTION_RULE = PB.RESOLUTION_RULE
STOP_RULE = PB.STOP_RULE
POLL_RULE = (
    'at unit mesh (Delta = 1; EVERY poll of this continuation) the poll set is: (1) the 2n = 14 OrthoMADS '
    'directions +h_j and -h_j, h_j the columns of H = I - 2 v v^T with v = (2 u_t - 1)/||2 u_t - 1||, u_t the Halton '
    'point (bases = the first 7 primes) of index t = 17 + k, each rounded to the lattice as d = round_half_away(h / '
    '||h||_inf) (PB.project_direction at Delta = 1); (2) every rounded point inc + d that is INFEASIBLE is SNAPPED '
    '(SNAP_RULE); (3) the COMPLETION -- every bound-, rule- and budget-feasible canonical lattice point within l_inf '
    'distance 1 of the incumbent (PB.Lattice.completion) -- is added ONLY IF FEWER THAN n + 1 = 8 DISTINCT feasible '
    'poll points remain after snapping (counted as distinct canonical points != the incumbent, i.e. the cache-hit '
    'plus new-evaluation candidates); the completion cap of 30 applies to it and REFUSES the poll '
    '(STOP_FOR_REVIEW_completion_cap), never truncates; the cap is checked before the evaluation budget. Full poll, '
    'batches of <= 5, cache hits never re-evaluated, duplicate eval keys dropped.')
SNAP_RULE = (
    'an infeasible rounded point r = inc + d is replaced by the NEAREST point of the SNAP SET = every feasible '
    'canonical lattice point z != inc with ||z - inc||_inf <= 1 (the unit frame of the incumbent, '
    'PB.Lattice.neighbourhood -- the same set as the completion). NEAREST = lexicographic minimum of (1) ||z - r||_1, '
    '(2) ||z - r||_inf, (3) ||z - inc||_1, (4) I(z) (lower first), (5) the lattice label (string order). Distances '
    'are in lattice units over the 7 coordinates (zP5, zE5, zP7, zE7, zP9, zE9, zY); the year coordinate is counted '
    'only when z holds storage (it is inactive otherwise, STEP4 5.4). If the snap set is empty the direction is '
    "recorded 'no feasible snap' and contributes no poll point (extreme barrier, not evaluated). Recorded for EVERY "
    'direction: the rounded point, whether it is feasible or the reasons it is not, the snapped point or '
    "'no feasible snap', the snap distances, every point tied at the minimum l1 distance and the tie-break level "
    'that decided.')
ITERATION_RULE = (
    'on SUCCESS (at least one determinate improvement: F(inc) - F(x) > max(bar_x + bar_inc, sigma_Q)) the incumbent '
    'moves to the argmin F among determinate improvers (ties: lower I, then label) and the NEXT POLL IS AGAIN A UNIT '
    'POLL (Delta stays 1: the STEP4 5.2 / ruling A3 doubling is NOT applied in this continuation); the Halton index '
    'advances by one per poll, CONTINUING the s51 F2 Phase B counter (first poll k = 4, t = 21: the poll the '
    'completion cap refused). On FAILURE the continuation terminates with the certificate. At most 60 NEW '
    'evaluations over the whole continuation: a poll whose new evaluations would exceed what remains is NOT launched '
    '(termination evaluation_budget_exhausted; no certificate claim).')
CERTIFICATE_STATEMENT = (
    'POLL FAILURE AT UNIT MESH over the recorded poll set: every point of the final poll set (the 2n rounded '
    'directions after snapping, plus the completion when it was triggered) was evaluated or read from the cache and '
    'none is a determinate improvement (F(inc) - F(x) > max(bar_x + bar_inc, sigma_Q)); indeterminate points are '
    'listed unresolved; barrier points are F = +inf. SCOPE: unless the completion was triggered this is NOT the '
    'full-box certificate of ruling A2 -- the feasible box neighbours outside the poll set are listed as unexamined; '
    'whether the poll displacement set positively spans R^7 is RECORDED (rank and an LP test), not assumed.')
INHERITED_RULING_SCOPE = (
    'Rulings of the Phase B record (PB, W24) as they apply here: A1 (7 variables, one common year) kept; A2 (box '
    'completion at every unit poll) REPLACED for this continuation by Addendum 39 ruling 1 (POLL_RULE; the cap of 30 '
    'kept); A3 (double on success) NOT applied -- every poll is a unit poll (ITERATION_RULE; Worker reading of '
    '"iterate on success", flagged to the Planner); A4 kept (improvement threshold; indeterminate not accepted); A5 '
    'kept for H, the Halton index and the rounding, EXTENDED from n+1 NEG to the 2n directions +-h_j; A6 kept: '
    'sigma_Q = 18,449.66 EUR, C3-era, not re-measured at m = 2, and at m = 2 it is the binding half of the threshold; '
    'A7 kept (barrier stop 2 per poll / 3 overall); A8 REPLACED: 60 new evaluations (Addendum 39 ruling 1), '
    'MAX_POLLS 60 kept as a safety.')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 39 ruling 1 (the standard unit poll, snap-to-feasible, completion only '
    'below n + 1 feasible poll points, cap 30 kept, iterate on success, cap 60) and Addendum 40 ruling 4',
    'data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json ruling4_F2_certificate',
    'Planner task W49 (concurrency 5, cap 500, 10 cycles, AA-on, m = 2, baseline ageing, budget in force; cache = '
    'every committed certified m = 2 record matched by evaluation key and pinned by sha256; baseline excluded)',
    'the s51 F2 Phase B: 2e2ef65d (launcher), c39c4836 (spec 5ce295e1), 0012d85b (run)',
    'Planner rulings A1-A11 (W24) as frozen in p515_s47_phase_b_record.py, with the scope in INHERITED_RULING_SCOPE',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__),) + tuple(F51.EXTRA_CLEAN_FILES)

SNAP_RECORD_FIELDS = ('direction_index', 'direction', 'rounded_z', 'rounded_feasible', 'rounded_label',
                      'rounded_infeasibility_reasons', 'result', 'snapped_z', 'snapped_label', 'snap_l1', 'snap_linf',
                      'snapped_linf_from_incumbent', 'n_tied_at_min_l1', 'tied_at_min_l1', 'decided_by')
POLL_RECORD_FIELDS = tuple(PB.POLL_RECORD_FIELDS) + ('poll_design', 'snap_table', 'n_distinct_feasible_poll_points',
                                                     'completion_triggered', 'poll_set_spanning')
CANDIDATE_RECORD_FIELDS = tuple(PB.CANDIDATE_RECORD_FIELDS) + ('snap',)
CERTIFICATE_FIELDS = ('statement', 'incumbent', 'poll_index', 'halton_t', 'poll_set', 'n_poll_points',
                      'all_poll_points_resolved', 'n_no_improvement', 'n_indeterminate_unresolved', 'n_barrier',
                      'n_improvement', 'completion_triggered', 'box_neighbourhood', 'spanning', 'holds')
POINT_RECORD_FIELDS = tuple(F51.POINT_RECORD_FIELDS)
CAMPAIGN_RESULT_FIELDS = (('termination', 'termination_certificate', 'final_incumbent', 'initial_incumbent',
                           'sigma_Q', 'cache_table_at_start', 'budget_facts', 'baseline_exclusion', 'poll_history',
                           'points', 'n_new_evaluations')
                          + tuple(f'poll_history[].{f}' for f in POLL_RECORD_FIELDS)
                          + tuple(f'poll_history[].candidates[].{f}' for f in CANDIDATE_RECORD_FIELDS)
                          + tuple(f'poll_history[].snap_table[].{f}' for f in SNAP_RECORD_FIELDS)
                          + tuple(f'termination_certificate.{f}' for f in CERTIFICATE_FIELDS)
                          + tuple(f'points[].{f}' for f in POINT_RECORD_FIELDS))

_log = L._log
_banner = L._banner
_load = L._load
make_key_of = F51.make_key_of
spec_label = F51.spec_label


# ======================================================================================================================
#  the poll (pure)
# ======================================================================================================================
def directions_2n(k, delta=DELTA_UNIT):
    """OrthoMADS 2n: +h_j then -h_j (j = 1..n), H from the Halton point t = t0 + k (PB.householder_columns), each
    rounded by PB.project_direction. -h is rounded from the negated column, not by negating a rounded one."""
    t = PB.HALTON_T0 + k
    u, cols = PB.householder_columns(t)
    pos = [PB.project_direction(c, delta) for c in cols]
    neg = [PB.project_direction([-a for a in c], delta) for c in cols]
    return t, u, pos + neg


def _snap_distances(lattice, z, r):
    """(l1, l_inf) in lattice units between a canonical point z and a rounded point r; the year coordinate counts
    only when z holds storage."""
    diffs = [abs(a - b) for a, b in zip(z[:-1], r[:-1])]
    if lattice.has_storage(z):
        diffs.append(abs(z[-1] - r[-1]))
    return sum(diffs), max(diffs) if diffs else 0


def snap_key(lattice, z, r, z_inc):
    l1, linf = _snap_distances(lattice, z, r)
    l1_inc = _snap_distances(lattice, z, z_inc)[0]
    return (l1, linf, l1_inc, lattice.investment_cost(z), lattice.label(z))


SNAP_KEY_LEVELS = ('l1_to_rounded', 'linf_to_rounded', 'l1_to_incumbent', 'lower_I', 'label')


def snap(lattice, r, z_inc, frame):
    """SNAP_RULE. `frame` = lattice.neighbourhood(z_inc). Returns the snap record fields for one rounded point."""
    cands = [z for z in frame if z != tuple(z_inc)]
    if not cands:
        return {'result': 'no_feasible_snap', 'snapped_z': None, 'snapped_label': None, 'snap_l1': None,
                'snap_linf': None, 'snapped_linf_from_incumbent': None, 'n_tied_at_min_l1': 0, 'tied_at_min_l1': [],
                'decided_by': None}
    keyed = sorted((snap_key(lattice, z, r, z_inc), z) for z in cands)
    best_key, best = keyed[0]
    tied = [(k, z) for k, z in keyed if k[0] == best_key[0]]
    decided_by = SNAP_KEY_LEVELS[0]
    if len(tied) > 1:
        runner_key = tied[1][0]
        level = next(i for i in range(len(best_key)) if best_key[i] != runner_key[i])
        decided_by = SNAP_KEY_LEVELS[level]
    return {'result': 'snapped', 'snapped_z': list(best), 'snapped_label': lattice.label(best),
            'snap_l1': best_key[0], 'snap_linf': best_key[1],
            'snapped_linf_from_incumbent': max(abs(a - b) for a, b in zip(best, z_inc)),
            'n_tied_at_min_l1': len(tied),
            'tied_at_min_l1': [{'label': lattice.label(z), 'linf_to_rounded': k[1], 'l1_to_incumbent': k[2],
                                'I_x_eur': k[3]} for k, z in tied],
            'decided_by': decided_by}


def spanning_diagnostic(displacements, n=N_VARS):
    """Rank of the displacement set and whether it POSITIVELY SPANS R^n: D positively spans iff rank(D) = n and
    D lambda = 0 has a solution with every lambda_i > 0 (equivalently lambda_i >= 1), tested by an LP
    (scipy.optimize.linprog, HiGHS -- a 7-row feasibility LP, not an OPF solve; the SolveProfileGuard hooks pyomo
    solvers only). Also lists the signed unit vectors +-e_i NOT in the cone of D. Recorded, never gating."""
    import numpy as np
    from scipy.optimize import linprog
    names = ('zP5', 'zE5', 'zP7', 'zE7', 'zP9', 'zE9', 'zY')
    if not displacements:
        return {'n_vectors': 0, 'rank': 0, 'positively_spans_R7': False, 'zero_strictly_interior': False,
                'signed_unit_vectors_not_in_cone': [f'{s}{v}' for v in names for s in '+-'],
                'method': 'empty set'}
    mat = np.array(displacements, dtype=float).T
    m = mat.shape[1]
    rank = int(np.linalg.matrix_rank(mat))
    lp = linprog(np.zeros(m), A_eq=mat, b_eq=np.zeros(n), bounds=[(1, None)] * m, method='highs')
    missing = []
    for i in range(n):
        for sign in (1, -1):
            e = np.zeros(n)
            e[i] = sign
            r = linprog(np.zeros(m), A_eq=mat, b_eq=e, bounds=[(0, None)] * m, method='highs')
            if r.status != 0:
                missing.append(f"{'+' if sign > 0 else '-'}{names[i]}")
    return {'n_vectors': m, 'rank': rank, 'zero_strictly_interior': lp.status == 0,
            'positively_spans_R7': rank == n and lp.status == 0, 'signed_unit_vectors_not_in_cone': missing,
            'method': ('rank: numpy.linalg.matrix_rank; positive spanning: rank == 7 and linprog(A_eq = D, b_eq = 0, '
                       'lambda >= 1) feasible (HiGHS); cone membership of +-e_i: linprog(A_eq = D, b_eq = e_i, '
                       'lambda >= 0)')}


def _blank_candidate(part, j, d, z, inc_bar, sigma_q):
    return {'poll_part': part, 'direction_index': j, 'direction': list(d), 'z': list(z), 'label': None,
            'eval_key': None, 'canonical': None, 'feasible': None, 'infeasibility_reasons': [], 'I_x_eur': None,
            'disposition': None, 'source': None, 'status': None, 'barrier_cause': None, 'Q_eur': None,
            'F_eur': None, 'bar_eur': None, 'incumbent_bar_eur': inc_bar, 'bar_sum_eur': None,
            'sigma_Q_eur': sigma_q, 'resolution_eur': None, 'F_inc_minus_F_eur': None, 'outcome': None,
            'snap': None}


def build_poll(lattice, cache, key_of, inc, k, sigma_q, completion_cap=COMPLETION_CAP):
    """One unit poll of the continuation, LISTED (nothing evaluated): directions, snap table, candidates with their
    dispositions, the completion trigger. Pure."""
    t, u, dirs = directions_2n(k)
    z_inc = tuple(inc['z'])
    frame = lattice.neighbourhood(z_inc)
    snap_table, cands, seen = [], [], {}

    def _admit(entry, z, tag):
        z = lattice.canonical_z(z)
        entry.update({'z': list(z), 'label': lattice.label(z), 'eval_key': key_of(z), 'feasible': True,
                      'I_x_eur': lattice.investment_cost(z),
                      'canonical': {'investment_year': lattice.year_of(z),
                                    'nodes': {str(n): list(v) for n, v in lattice.nodes_map(z).items()}}})
        if z == z_inc:
            entry.update({'disposition': 'dropped_inactive_only', 'outcome': 'dropped'})
        elif entry['eval_key'] in seen:
            entry.update({'disposition': f"duplicate_of_{seen[entry['eval_key']]}", 'outcome': 'dropped'})
        else:
            seen[entry['eval_key']] = tag
            entry['disposition'] = 'cache_hit' if entry['eval_key'] in cache else 'new_evaluation'

    for j, d in enumerate(dirs):
        r = tuple(a + b for a, b in zip(z_inc, d))
        why = lattice.reasons(r)
        rec = {'direction_index': j, 'direction': list(d), 'rounded_z': list(r), 'rounded_feasible': not why,
               'rounded_label': None if why else lattice.label(r), 'rounded_infeasibility_reasons': why}
        if not why:
            rec.update({'result': 'feasible_as_rounded', 'snapped_z': None, 'snapped_label': None, 'snap_l1': None,
                        'snap_linf': None, 'snapped_linf_from_incumbent': None, 'n_tied_at_min_l1': None,
                        'tied_at_min_l1': [], 'decided_by': None})
            entry = _blank_candidate('direction', j, d, r, inc['bar'], sigma_q)
            entry['snap'] = {'result': 'feasible_as_rounded'}
            _admit(entry, r, f'direction_{j}')
        else:
            rec.update(snap(lattice, r, z_inc, frame))
            entry = _blank_candidate('direction_snapped', j, d, r, inc['bar'], sigma_q)
            entry['snap'] = {k2: rec[k2] for k2 in ('result', 'rounded_z', 'rounded_infeasibility_reasons',
                                                    'snap_l1', 'snap_linf', 'n_tied_at_min_l1', 'decided_by')}
            if rec['result'] == 'no_feasible_snap':
                entry.update({'feasible': False, 'infeasibility_reasons': why,
                              'disposition': 'rejected_no_feasible_snap',
                              'outcome': 'barrier_infeasible_not_evaluated'})
            else:
                _admit(entry, tuple(rec['snapped_z']), f'direction_{j}')
        snap_table.append({f: rec.get(f) for f in SNAP_RECORD_FIELDS})
        cands.append(entry)
    n_distinct = sum(1 for c in cands if c['disposition'] in ('cache_hit', 'new_evaluation'))
    triggered = n_distinct < MIN_FEASIBLE_POLL_POINTS
    completion = None
    if triggered:
        completion = lattice.completion(z_inc)
        completion['over_cap'] = completion['n_feasible'] > completion_cap
        completion['cap'] = completion_cap
        if not completion['over_cap']:
            for z in completion['z']:
                entry = _blank_candidate('completion', None, tuple(a - b for a, b in zip(z, z_inc)), z, inc['bar'],
                                         sigma_q)
                _admit(entry, tuple(z), 'completion')
                cands.append(entry)
    polled = [c for c in cands if c['disposition'] in ('cache_hit', 'new_evaluation')]
    new = [c for c in polled if c['disposition'] == 'new_evaluation']
    record = {'poll_index': k, 'halton_t': t, 'halton_u': u, 'poll_size_delta': DELTA_UNIT,
              'mesh_size': PB.MESH_SIZE, 'poll_design': POLL_DESIGN,
              'incumbent': {kk: inc[kk] for kk in ('label', 'eval_key', 'z', 'I', 'Q', 'F', 'bar')},
              'directions': [list(d) for d in dirs], 'snap_table': snap_table, 'candidates': cands,
              'n_distinct_feasible_poll_points': n_distinct, 'completion_triggered': triggered,
              'n_new_evaluations': len(new), 'n_cache_hits': sum(c['disposition'] == 'cache_hit' for c in cands),
              'batches': [], 'decision': None, 'next_incumbent': None, 'next_poll_size': None, 'unit_poll': True,
              'completion': None if completion is None else {kk: v for kk, v in completion.items() if kk != 'z'},
              'poll_set_spanning': spanning_diagnostic([[a - b for a, b in zip(c['z'], z_inc)] for c in polled])}
    return record, completion


def _certificate(lattice, key_of, cache, inc, record):
    polled = [c for c in record['candidates'] if c['disposition'] in ('cache_hit', 'new_evaluation')]
    outcomes = [c['outcome'] for c in polled]
    nb = lattice.neighbourhood(tuple(inc['z']))
    polled_keys = {c['eval_key'] for c in polled}
    nb_rows = [{'label': lattice.label(z), 'in_poll_set': key_of(z) in polled_keys, 'in_cache': key_of(z) in cache,
                'I_x_eur': lattice.investment_cost(z)} for z in nb]
    cert = {'statement': CERTIFICATE_STATEMENT, 'incumbent': inc['label'], 'poll_index': record['poll_index'],
            'halton_t': record['halton_t'],
            'poll_set': [{'label': c['label'], 'poll_part': c['poll_part'], 'direction_index': c['direction_index'],
                          'disposition': c['disposition'], 'outcome': c['outcome'], 'F_eur': c['F_eur'],
                          'F_inc_minus_F_eur': c['F_inc_minus_F_eur'], 'resolution_eur': c['resolution_eur']}
                         for c in polled],
            'n_poll_points': len(polled),
            'all_poll_points_resolved': all(o in ('no_improvement', 'indeterminate', 'barrier', 'improvement')
                                            for o in outcomes),
            'n_no_improvement': outcomes.count('no_improvement'),
            'n_indeterminate_unresolved': outcomes.count('indeterminate'), 'n_barrier': outcomes.count('barrier'),
            'n_improvement': outcomes.count('improvement'), 'completion_triggered': record['completion_triggered'],
            'box_neighbourhood': {'n_feasible': len(nb), 'n_in_poll_set': sum(r['in_poll_set'] for r in nb_rows),
                                  'n_unexamined': sum(1 for r in nb_rows if not r['in_poll_set']),
                                  'unexamined': [r for r in nb_rows if not r['in_poll_set']],
                                  'is_the_full_box_certificate': all(r['in_poll_set'] for r in nb_rows)},
            'spanning': record['poll_set_spanning']}
    cert['holds'] = cert['all_poll_points_resolved'] and cert['n_improvement'] == 0 and cert['n_poll_points'] > 0
    return cert


def run_continuation(lattice, cache, key_of, incumbent, evaluate_fn, sigma_q, k0=FIRST_HALTON_K,
                     max_new_evaluations=MAX_NEW_EVALUATIONS, max_polls=MAX_POLLS, batch_size=CONCURRENCY,
                     completion_cap=COMPLETION_CAP, log=_log, on_poll=None, on_poll_start=None):
    """The continuation loop (POLL_RULE, SNAP_RULE, ITERATION_RULE). `cache` (eval_key -> entry) is extended in
    place. `evaluate_fn(list[z]) -> list[cache entry]`. Structure, barrier rule, classification and record schema
    follow PB.run_mads; the poll set is `build_poll`'s."""
    inc = dict(incumbent)
    history, n_new, n_barrier_new = [], 0, 0
    termination, certificate = None, None

    def _finish(rec):
        history.append(rec)
        if on_poll:
            on_poll(rec)

    for i in range(max_polls):
        k = k0 + i
        record, completion = build_poll(lattice, cache, key_of, inc, k, sigma_q, completion_cap=completion_cap)
        cands = record['candidates']
        new = [c for c in cands if c['disposition'] == 'new_evaluation']
        if on_poll_start:
            on_poll_start(record)
        if completion is not None and completion['over_cap']:
            record['decision'] = 'stopped_for_review_completion_cap'
            for c in cands:
                if c['disposition'] in ('cache_hit', 'new_evaluation'):
                    c['outcome'] = 'not_evaluated_completion_cap'
            _finish(record)
            termination = {'reason': 'STOP_FOR_REVIEW_completion_cap', 'poll_size_reached': DELTA_UNIT,
                           'completion_n_feasible': completion['n_feasible'], 'cap': completion_cap,
                           'rule': POLL_RULE,
                           'detail': f"poll {k}: {record['n_distinct_feasible_poll_points']} distinct feasible poll "
                                     f"points < {MIN_FEASIBLE_POLL_POINTS} triggered the completion of "
                                     f"{inc['label']}, which has {completion['n_feasible']} feasible points > "
                                     f'{completion_cap}; refused, nothing of this poll evaluated, not truncated'}
            break
        if n_new + len(new) > max_new_evaluations:
            record['decision'] = 'not_launched_evaluation_budget'
            _finish(record)
            termination = {'reason': 'evaluation_budget_exhausted', 'poll_size_reached': DELTA_UNIT,
                           'detail': f'poll {k} needs {len(new)} new evaluations; {n_new} used of '
                                     f'{max_new_evaluations}'}
            break
        stop = None
        for b in range(0, len(new), batch_size):
            batch = new[b:b + batch_size]
            recs = evaluate_fn([tuple(c['z']) for c in batch])
            if len(recs) != len(batch):
                raise RuntimeError('evaluate_fn returned a different number of records')
            for c, rec in zip(batch, recs):
                if rec.get('eval_key') != c['eval_key']:
                    raise RuntimeError(f"evaluation record eval key {rec.get('eval_key')} != {c['eval_key']}")
                cache[c['eval_key']] = dict(rec)
                n_new += 1
                if rec['status'] != 'certified':
                    n_barrier_new += 1
            record['batches'].append([c['label'] for c in batch])
            barrier_this_poll = sum(1 for c in new if c['eval_key'] in cache
                                    and cache[c['eval_key']]['status'] != 'certified')
            if barrier_this_poll >= PB.BARRIER_STOP_PER_POLL or n_barrier_new >= PB.BARRIER_STOP_OVERALL:
                stop = {'reason': 'STOP_FOR_REVIEW_barrier_rule', 'barrier_new_this_poll': barrier_this_poll,
                        'barrier_new_overall': n_barrier_new, 'rule': STOP_RULE}
                break
        for c in cands:
            if c['disposition'] not in ('cache_hit', 'new_evaluation'):
                continue
            rec = cache.get(c['eval_key'])
            if rec is None:
                c['outcome'] = 'not_evaluated_stop_rule'
                continue
            c['source'] = rec.get('source')
            c['status'] = rec['status']
            c['barrier_cause'] = rec.get('barrier_cause')
            if rec['status'] != 'certified' or rec.get('Q') is None:
                c.update({'F_eur': None, 'outcome': 'barrier'})
                continue
            f_x = c['I_x_eur'] + rec['Q']
            res = PB.resolution(rec.get('bar'), inc['bar'], sigma_q)
            c.update({'Q_eur': rec['Q'], 'F_eur': f_x, 'bar_eur': rec.get('bar'),
                      'bar_sum_eur': (rec['bar'] + inc['bar']) if (rec.get('bar') is not None
                                                                     and inc['bar'] is not None) else None,
                      'resolution_eur': res if math.isfinite(res) else None, 'F_inc_minus_F_eur': inc['F'] - f_x,
                      'outcome': PB.classify(inc['F'], f_x, res)})
        if stop:
            record['decision'] = 'stopped_for_review'
            _finish(record)
            termination = dict(stop, poll_size_reached=DELTA_UNIT)
            break
        improvers = [c for c in cands if c['outcome'] == 'improvement']
        if improvers:
            best = min(improvers, key=lambda c: (c['F_eur'], c['I_x_eur'], c['label']))
            inc = {'eval_key': best['eval_key'], 'z': tuple(best['z']), 'label': best['label'], 'I': best['I_x_eur'],
                   'Q': best['Q_eur'], 'F': best['F_eur'], 'bar': best['bar_eur'], 'source': best['source']}
            record.update({'decision': 'success', 'next_incumbent': best['label'], 'next_poll_size': DELTA_UNIT})
            _finish(record)
            log(f"[S53-CERT] poll {k} (t={record['halton_t']}) incumbent={record['incumbent']['label']} "
                f"new={record['n_new_evaluations']} hits={record['n_cache_hits']} -> success, next {best['label']}")
            continue
        record.update({'decision': 'failure_at_unit_mesh', 'next_incumbent': inc['label'], 'next_poll_size': None})
        _finish(record)
        termination = {'reason': 'poll_failure_at_unit_mesh', 'poll_size_reached': DELTA_UNIT}
        certificate = _certificate(lattice, key_of, cache, inc, record)
        break
    if termination is None:
        termination = {'reason': 'max_polls_reached', 'poll_size_reached': DELTA_UNIT}
    last = history[-1] if history else None
    unresolved = [c for c in (last['candidates'] if last else []) if c['outcome'] == 'indeterminate']
    nb = lattice.neighbourhood(tuple(inc['z']))
    return {'termination': termination, 'termination_certificate': certificate,
            'incumbent': {kk: inc[kk] for kk in ('label', 'eval_key', 'z', 'I', 'Q', 'F', 'bar')},
            'n_polls': len(history), 'n_new_evaluations': n_new, 'n_barrier_new_evaluations': n_barrier_new,
            'final_poll_unresolved_indeterminate': [{kk: c[kk] for kk in ('label', 'F_eur', 'F_inc_minus_F_eur',
                                                                          'resolution_eur')} for c in unresolved],
            'final_poll_feasible_points': [c['label'] for c in (last['candidates'] if last else [])
                                           if c['disposition'] in ('cache_hit', 'new_evaluation')],
            'lattice_neighbourhood_of_incumbent': [
                {'label': lattice.label(z), 'eval_key': key_of(z), 'in_cache': key_of(z) in cache,
                 'polled_in_final_poll': any(c['eval_key'] == key_of(z) for c in (last['candidates'] if last else [])),
                 'I_x_eur': lattice.investment_cost(z)} for z in nb],
            'history': history}


class _DryStop(Exception):
    pass


def dry_run(lattice, cache, key_of, inc, sigma_q):
    """ZERO SOLVES: follow the continuation until the first poll that needs a new evaluation; return that poll as
    listed (snap table, completion trigger, cache hits, new evaluations, batch sizes)."""
    started = []

    def _dry_eval(batch):
        raise _DryStop()
    try:
        out = run_continuation(lattice, dict(cache), key_of, dict(inc), _dry_eval, sigma_q, log=lambda m: None,
                               on_poll_start=started.append)
        return {'complete_without_new_evaluations': True, 'termination': out['termination'],
                'termination_certificate': out['termination_certificate'],
                'polls': [_poll_plan(p, cache) for p in out['history']]}
    except _DryStop:
        return {'complete_without_new_evaluations': False, 'first_evaluating_poll': _poll_plan(started[-1], cache),
                'earlier_polls': [_poll_plan(p, cache) for p in started[:-1]]}


def _poll_plan(poll, cache):
    new = [c for c in poll['candidates'] if c['disposition'] == 'new_evaluation']
    return {'poll_index': poll['poll_index'], 'halton_t': poll['halton_t'], 'Delta': poll['poll_size_delta'],
            'incumbent': poll['incumbent']['label'], 'directions': poll['directions'],
            'snap_table': poll['snap_table'],
            'n_distinct_feasible_poll_points': poll['n_distinct_feasible_poll_points'],
            'completion_triggered': poll['completion_triggered'],
            'completion_n_feasible': (poll['completion'] or {}).get('n_feasible'),
            'candidates': [{'part': c['poll_part'], 'j': c['direction_index'], 'label': c['label'],
                            'disposition': c['disposition'], 'I_x_eur': c['I_x_eur'],
                            'cache_source': (((cache.get(c['eval_key']) or {}).get('source') or {}).get('campaign')
                                             if c['disposition'] == 'cache_hit' else None)}
                           for c in poll['candidates']],
            'n_cache_hits': poll['n_cache_hits'], 'n_new_evaluations': len(new),
            'new_evaluations': [{'label': c['label'], 'poll_part': c['poll_part'], 'direction_index':
                                 c['direction_index'], 'I_x_eur': c['I_x_eur']} for c in new],
            'batch_sizes': [len(new[b:b + CONCURRENCY]) for b in range(0, len(new), CONCURRENCY)],
            'poll_set_spanning': poll['poll_set_spanning'], 'decision': poll.get('decision')}


# ======================================================================================================================
#  the fourth cache source: the committed s51 F2 Phase B
# ======================================================================================================================
def _pin_state(pin, need_sha):
    path = os.path.join(REPO, pin['path'])
    got = H.sha256_file(path) if os.path.isfile(path) else None
    tracked, clean = PB._git_state(pin['path'])
    in_head = F51._commit_in_head(pin['commit']) if pin.get('commit') else None
    ok = got is not None and tracked and clean and (in_head is not False) and (not need_sha or got == pin['sha256'])
    return ok, {'path': pin['path'], 'sha256_pinned': pin.get('sha256'), 'sha256_on_disk': got,
                'git_tracked': tracked, 'git_clean': clean, 'commit': pin.get('commit'), 'commit_in_HEAD': in_head}


def load_phase_b_source(case_file_sha256, lattice):
    """The s51 F2 Phase B as a cache source -> (ok, info, entries). The point checks are F51.load_cache_source's,
    transcribed; the run-level checks differ in ONE respect, stated: this source is EXPECTED to have stopped for
    review on the completion cap (that is why it is continued), with no barrier and no non-certified point."""
    src = PHASE_B_SOURCE
    info = {'name': src['name'], 'what': src['what'], 'results': dict(src['results']), 'spec': dict(src['spec']),
            'manifest': dict(src['manifest']), 'script': dict(src['script'])}
    reasons = []
    for what, need_sha in (('results', True), ('spec', True), ('manifest', True), ('script', False)):
        ok, state = _pin_state(src[what], need_sha)
        info[f'{what}_state'] = state
        if not ok:
            reasons.append(f'{what}: {state}')
    if reasons:
        return False, dict(info, reason='; '.join(reasons)), {}
    results = _load(src['results']['path'])
    spec = _load(src['spec']['path'])
    manifest = _load(src['manifest']['path'])
    extra = spec.get('extra') or {}
    checks = {
        'spec_sha256_matches_results': results.get('campaign_spec_sha256') == src['spec']['sha256'],
        'spec_path_matches_results': results.get('campaign_spec_path') == src['spec']['path'],
        'stopped_for_review_on_the_completion_cap': (
            results.get('STOP_FOR_REVIEW') is True
            and (results.get('termination') or {}).get('reason') == 'STOP_FOR_REVIEW_completion_cap'),
        'no_non_certified_points': results.get('non_certified_points') == [],
        'no_barrier_evaluations': results.get('n_barrier_new_evaluations') == 0,
        'no_harness_errors': results.get('harness_errors') == [],
        'no_readback_mismatch': results.get('readback_mismatch_points') == [],
        'no_solve_reconciliation_mismatch': results.get('solve_reconciliation_mismatch_points') == [],
        'flex_multiplier_is_m2': results.get('flex_price_multiplier') == M_FLEX,
        'spec_frozen_this_s51_launcher': (extra.get('campaign_script_sha256')
                                          == H.sha256_file(os.path.join(REPO, src['script']['path']))),
        'spec_frozen_this_pb_module': ((extra.get('phase_b_record_script') or {}).get('sha256')
                                       == H.sha256_file(os.path.join(REPO, PB.__file__))),
        'spec_frozen_this_f2_ladder_launcher': ((extra.get('f2_ladder_script') or {}).get('sha256')
                                                == H.sha256_file(os.path.join(REPO, F51.F2_LADDER_SCRIPT['path']))),
    }
    checks.update(F51._spec_declares_the_same_configuration(spec, case_file_sha256))
    record_files = {k: v for k, v in manifest.items()
                    if os.path.basename(k) in ('evaluation_record.json', 'per_cycle_record.jsonl')}
    bad = sorted(k for k, v in record_files.items() if not os.path.isfile(os.path.join(REPO, k))
                 or H.sha256_file(os.path.join(REPO, k)) != v)
    info['manifest_check'] = {'n_record_files_checked': len(record_files), 'mismatched': bad}
    checks['manifest_record_hashes_hold'] = not bad and bool(record_files)
    entries, seen_labels = {}, []
    for label, point in sorted((results.get('points') or {}).items()):
        seen_labels.append(label)
        canonical = point.get('candidate_canonical')
        ekey = F51.eval_key_of_canonical(canonical)
        pc = {
            'certified': point.get('status') == 'certified',
            'flex_price_multiplier_is_m2': point.get('flex_price_multiplier') == M_FLEX,
            'eval_key_recomputes_at_m2': point.get('eval_key') == ekey,
            'eval_key_differs_from_the_baseline_key': ekey != F51.eval_key_of_canonical(canonical, None),
            'candidate_key_recomputes': point.get('candidate_key') == H.candidate_key(canonical),
            'flex_price_label_in_record': point.get('flex_price_label_in_record') == FLEX_LABEL,
            'flex_price_multiplier_in_record': point.get('flex_price_multiplier_in_record') == M_FLEX,
            'Q_is_the_expected_committed_value': point.get('Q') == Q_PHASE_B_EXPECTED.get(label),
            'bar_present': isinstance(point.get('bar'), float),
            'on_the_lattice': lattice.z_of_canonical(canonical) is not None,
            'flex_readback_all_match': ((point.get('flex_price_readback') or {}).get('all_match') is True),
            'ageing_readback_all_match': ((point.get('ess_ageing_readback_all_match') or {}).get('pre_run') is True
                                          and (point.get('ess_ageing_readback_all_match')
                                               or {}).get('post_run') is True),
            'exit_code_0': point.get('exit_code') == 0,
        }
        rec_rel = os.path.join(point.get('eval_dir') or '', 'evaluation_record.json')
        present = bool(point.get('eval_dir')) and os.path.isfile(os.path.join(REPO, rec_rel))
        pc['evaluation_record_present'] = present
        if present:
            rec = _load(rec_rel)
            pc['record_hash_in_campaign_manifest'] = manifest.get(rec_rel) == H.sha256_file(os.path.join(REPO, rec_rel))
            pc['record_Q_matches'] = rec.get('certified_cost') == point.get('Q')
            pc['record_bar_matches'] = (rec.get('bar') or {}).get('value') == point.get('bar')
            pc['record_cycles_matches'] = rec.get('cycles_run') == point.get('cycles_run')
            pc['record_tracked'] = bool(H._git(['ls-files', '--', rec_rel]).strip())
        traj = point.get('per_cycle_trajectory') or {}
        pc['trajectory_hash_in_campaign_manifest'] = (bool(traj.get('path'))
                                                      and manifest.get(traj['path']) == traj.get('sha256'))
        for key, value in pc.items():
            checks[f'{label}:{key}'] = value
        if ekey in entries:
            reasons.append(f'eval key {ekey[:16]} twice')
        entries[ekey] = {
            'label': label, 'status': point.get('status'), 'Q': point.get('Q'), 'bar': point.get('bar'),
            'canonical': canonical, 'barrier_cause': point.get('barrier_cause'),
            'flex_price_multiplier': point.get('flex_price_multiplier'), 'cycles_run': point.get('cycles_run'),
            'certification_cycle': point.get('certification_cycle'), 'rule_ten': point.get('rule_ten'),
            'eval_dir': point.get('eval_dir'), 'per_cycle_trajectory': traj,
            'source': {'kind': 'committed_certified_record_at_m2', 'campaign': src['name'],
                       'path': src['results']['path'], 'sha256': src['results']['sha256'],
                       'commit': src['results']['commit'], 'label': label, 'eval_dir': point.get('eval_dir')}}
    checks['labels_are_exactly_the_expected_ones'] = sorted(seen_labels) == sorted(Q_PHASE_B_EXPECTED)
    info.update({'checks': checks, 'n_entries': len(entries), 'm2_labels': sorted(seen_labels),
                 'points_at_other_multipliers_not_cached': []})
    failing = sorted(k for k, v in checks.items() if not v)
    if failing or reasons:
        return False, dict(info, reason=f'failing checks {failing}; {reasons}'), {}
    return True, info, entries


def continuation_facts(lattice, cache, key_of):
    """The incumbent and the poll counter are the committed s51 Phase B's, asserted: final_incumbent bitwise, I(z)
    recomputed, in the cache with the same Q and bar; the s51 terminal poll is the refused unit poll k = 4 (t = 21)
    with the 61-point box completion, which reproduces here; its n H-columns equal the first n directions of the 2n
    set at k = 4 (the same H)."""
    res = _load(PHASE_B_SOURCE['results']['path'])
    fin = res['final_incumbent']
    z = tuple(fin['z'])
    last = res['poll_history'][-1]
    t_k0, _u, dirs = directions_2n(FIRST_HALTON_K)
    entry = cache.get(fin['eval_key']) or {}
    checks = {
        'final_incumbent_is_the_expected_one': fin == EXPECTED_INCUMBENT,
        'I_is_992268_1082843702': fin['I'] == 992268.1082843702,
        'F_is_810787758_2559508': fin['F'] == 810787758.2559508,
        'I_recomputes_from_the_lattice': abs(lattice.investment_cost(z) - fin['I']) <= 1e-6,
        'F_equals_I_plus_Q': fin['F'] == fin['I'] + fin['Q'],
        'budget_feasible': not lattice.reasons(z),
        'budget_slack_as_expected': abs((lattice.budget - lattice.investment_cost(z))
                                        - EXPECTED_BUDGET_SLACK_EUR) <= 1e-6,
        'eval_key_recomputes_at_m2': key_of(z) == fin['eval_key'],
        'incumbent_in_the_cache_with_the_same_Q_and_bar': (entry.get('Q') == fin['Q'] and entry.get('bar') == fin['bar']
                                                           and entry.get('status') == 'certified'),
        's51_terminal_poll_is_k4': last['poll_index'] == FIRST_HALTON_K and last['halton_t'] == t_k0 == 21,
        's51_terminal_poll_is_unit': last['poll_size_delta'] == DELTA_UNIT and last['unit_poll'] is True,
        's51_terminal_poll_refused_on_the_cap': last['decision'] == 'stopped_for_review_completion_cap',
        's51_terminal_poll_incumbent_is_ours': last['incumbent']['label'] == fin['label'],
        's51_completion_was_61': (last['completion'] or {}).get('n_feasible') == EXPECTED_S51_BOX_COMPLETION,
        'box_completion_reproduces_61': lattice.completion(z)['n_feasible'] == EXPECTED_S51_BOX_COMPLETION,
        'same_H_first_n_directions': [list(d) for d in dirs[:N_VARS]] == last['directions'][:N_VARS],
        'negatives_are_the_negated_columns': all(list(dirs[N_VARS + j]) == [-a for a in dirs[j]]
                                                 for j in range(N_VARS)),
    }
    inc = {'eval_key': fin['eval_key'], 'z': z, 'label': fin['label'], 'I': fin['I'], 'Q': fin['Q'], 'F': fin['F'],
           'bar': fin['bar'], 'source': entry.get('source')}
    return inc, {'incumbent': dict(fin), 'budget_slack_eur': lattice.budget - lattice.investment_cost(z),
                 's51_terminal_poll': {k: last[k] for k in ('poll_index', 'halton_t', 'poll_size_delta', 'decision',
                                                            'directions')},
                 's51_termination': res['termination'], 'first_halton_k': FIRST_HALTON_K, 'first_halton_t': t_k0,
                 'box_neighbourhood_spanning_non_gating': spanning_diagnostic(
                     [[a - b for a, b in zip(nb, z)] for nb in lattice.neighbourhood(z)]),
                 'checks': checks}


def baseline_exclusion(cache_keys, domain_keys, lattice):
    """F51.baseline_exclusion's scan, transcribed with the FOUR pinned m = 2 sources as the cache. SCOPE (recorded,
    never absolute): every campaign_results.json under data/SRP1/Results (recursive glob) in the working tree."""
    cache_paths = set(ALL_CACHE_SOURCE_PATHS)
    files, collisions, m2_elsewhere = [], [], []
    for path in sorted(glob.glob(os.path.join(REPO, BASELINE_EXCLUSION_GLOB), recursive=True)):
        rel = os.path.relpath(path, REPO)
        try:
            payload = _load(rel)
        except (ValueError, OSError) as error:
            files.append({'file': rel, 'unreadable': f'{type(error).__name__}: {error}'})
            continue
        points = payload.get('points') or {}
        points = list(points.values()) if isinstance(points, dict) else list(points)
        n_m2, hits = 0, []
        for point in points:
            if not isinstance(point, dict):
                continue
            ekey = point.get('eval_key') or point.get('candidate_key')
            if point.get('flex_price_multiplier') == M_FLEX:
                n_m2 += 1
                if rel not in cache_paths:
                    m2_elsewhere.append({'file': rel, 'label': point.get('label'), 'eval_key': ekey})
            if rel in cache_paths:
                continue
            if ekey and (ekey in cache_keys or ekey in domain_keys):
                hits.append({'label': point.get('label'), 'eval_key': ekey,
                             'flex_price_multiplier': point.get('flex_price_multiplier')})
        if hits:
            collisions.append({'file': rel, 'points': hits})
        files.append({'file': rel, 'n_points': len(points), 'n_points_at_m2': n_m2,
                      'is_a_pinned_cache_source': rel in cache_paths})
    baseline_keys = {H.evaluation_key(H.candidate_key(PB.canonical_of(lattice, z)), {}, case_file_aa=CASE_FILE_AA,
                                      ess_ageing_baseline=ESS_AGEING_BASELINE) for z in lattice.domain()}
    overlap = sorted(baseline_keys & (set(domain_keys) | set(cache_keys)))
    return {'scope': ('every campaign_results.json under data/SRP1/Results in the working tree (recursive glob '
                      f'{BASELINE_EXCLUSION_GLOB}); the four pinned m = 2 sources are the cache and are skipped in '
                      'the collision test'),
            'cache_source_paths': list(ALL_CACHE_SOURCE_PATHS),
            'n_files': len(files), 'files': files, 'key_collisions': collisions,
            'm2_points_outside_the_pinned_sources': m2_elsewhere,
            'n_domain_points': len(domain_keys), 'n_baseline_keys_of_the_same_domain': len(baseline_keys),
            'baseline_keys_overlapping_domain_or_cache': overlap, 'm_enters_the_evaluation_key': not overlap,
            'every_cache_entry_is_m2': None, 'ok': not collisions and not m2_elsewhere and not overlap and bool(files)}


def budget_facts(lattice, cache):
    rows, checks = {}, {}
    for ekey, entry in cache.items():
        z = lattice.z_of_canonical(entry['canonical'])
        why = lattice.reasons(z) if z is not None else ['off the lattice']
        i_x = lattice.investment_cost(z) if z is not None else None
        rows[entry['label']] = {'z': list(z) if z is not None else None, 'I_x_eur': i_x,
                                'slack_eur': (lattice.budget - i_x) if i_x is not None else None,
                                'budget_feasible': not why, 'reasons': why, 'eval_key': ekey}
        expect_infeasible = entry['label'] in BUDGET_INFEASIBLE_CACHE_LABELS
        checks[f"{entry['label']}:{'budget_infeasible' if expect_infeasible else 'feasible'}_as_expected"] = (
            (bool(why) and any(r.startswith('budget') for r in why)) if expect_infeasible else not why)
    checks['lattice_budget_is_1e6'] = lattice.budget == 1e6
    return {'budget_eur': lattice.budget,
            'expected_budget_infeasible_cache_labels': list(BUDGET_INFEASIBLE_CACHE_LABELS),
            'constraint': ('I(x) <= B = 1,000,000 EUR is a CONSTRAINT: an infeasible rounded point is snapped '
                           '(SNAP_RULE) or, with no feasible snap, rejected before evaluation (extreme barrier)'),
            'cache_rows': rows, 'checks': checks}


# ======================================================================================================================
#  rule eleven -- every capture path asserted BEFORE the run, through the REAL loop with fake evaluators
# ======================================================================================================================
def rule_eleven(lattice, sigma_q, cache, key_of, incumbent):
    flex_side = L.rule_eleven()
    pb_side = PB.rule_eleven(lattice, sigma_q)
    checks = {f'flex_{k}': v for k, v in flex_side['checks'].items()}
    checks.update({f'pb_{k}': v for k, v in pb_side['checks'].items()})
    point_src = inspect.getsource(L._point_result)
    checks.update({
        'point_record_flex_readback': "'flex_price_readback':" in point_src,
        'point_record_flex_cost': "'flex_cost':" in point_src,
        'point_record_rule_ten': "'rule_ten':" in point_src,
        'point_record_solve_reconciliation': "'solve_reconciliation':" in point_src,
        'point_record_ageing_readback': "'ess_ageing_readback_all_match':" in point_src,
        'point_record_per_cycle_trajectory': "'per_cycle_trajectory':" in point_src,
        'point_record_net_recourse_and_salvage': ("'net_operational_recourse':" in point_src
                                                  and "'terminal_salvage_value':" in point_src),
        'point_reader_is_f51s': inspect.getsource(F51._point_result).count('L._point_result(') == 1,
        'cache_entries_carry_Q_bar_canonical': all(
            isinstance(e.get('Q'), float) and isinstance(e.get('bar'), float) and bool(e.get('canonical'))
            for e in cache.values() if e['status'] == 'certified'),
        'cache_entries_carry_the_multiplier': all(e.get('flex_price_multiplier') == M_FLEX for e in cache.values()),
    })

    def _fake(status, q_of=None):
        def _eval(batch):
            return [{'label': lattice.label(z), 'status': status, 'eval_key': key_of(z),
                     'Q': (q_of(z) if q_of else None), 'bar': (1.0 if q_of else None),
                     'canonical': PB.canonical_of(lattice, z), 'barrier_cause': None if q_of else 'synthetic',
                     'source': 'synthetic'} for z in batch]
        return _eval

    quiet = {'log': lambda m: None}
    # (a) from the REAL incumbent, synthetic barrier records: poll / candidate / snap fields; barrier stop armed
    out_a = run_continuation(lattice, dict(cache), key_of, dict(incumbent), _fake('not_certified'), sigma_q, **quiet)
    pa = out_a['history'][0]
    checks.update({
        'a_poll_fields': all(f in pa for f in POLL_RECORD_FIELDS),
        'a_candidate_fields': all(all(f in c for f in CANDIDATE_RECORD_FIELDS) for c in pa['candidates']),
        'a_snap_table_2n_entries_all_fields': (len(pa['snap_table']) == N_DIRECTIONS
                                               and all(all(f in s for f in SNAP_RECORD_FIELDS)
                                                       for s in pa['snap_table'])),
        'a_every_direction_feasible_or_snapped_or_no_snap': all(
            s['result'] in ('feasible_as_rounded', 'snapped', 'no_feasible_snap') for s in pa['snap_table']),
        'a_unit_poll_at_k4_t21': pa['poll_size_delta'] == 1 and pa['poll_index'] == 4 and pa['halton_t'] == 21,
        'a_barrier_stop_rule_armed': out_a['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule',
        'a_spanning_recorded': all(k in pa['poll_set_spanning'] for k in ('rank', 'positively_spans_R7')),
    })
    # (b) evaluation cap armed: a cap of 0 refuses the first poll, nothing evaluated
    out_b = run_continuation(lattice, dict(cache), key_of, dict(incumbent), _fake('not_certified'), sigma_q,
                             max_new_evaluations=0, **quiet)
    checks['b_evaluation_cap_armed'] = (out_b['termination']['reason'] == 'evaluation_budget_exhausted'
                                        and out_b['n_new_evaluations'] == 0)
    # (c) failure path + certificate: every new point certified and no better -> poll failure, certificate holds
    f_inc = incumbent['F']
    out_c = run_continuation(lattice, dict(cache), key_of, dict(incumbent),
                             _fake('certified', lambda z: f_inc - lattice.investment_cost(z) + 1.0e6), sigma_q,
                             **quiet)
    cert = out_c['termination_certificate'] or {}
    checks.update({
        'c_poll_failure_terminates': out_c['termination']['reason'] == 'poll_failure_at_unit_mesh',
        'c_certificate_fields': all(f in cert for f in CERTIFICATE_FIELDS),
        'c_certificate_holds_on_synthetic_failure': cert.get('holds') is True,
        'c_box_scope_recorded': 'is_the_full_box_certificate' in (cert.get('box_neighbourhood') or {}),
    })
    # (d) success path: the first new point improves by 1e7 -> the incumbent moves, the next poll is a UNIT poll at
    #     k + 1; afterwards nothing improves -> failure with the certificate at the new incumbent
    first = {}

    def _q_success(z):
        key = key_of(z)
        if not first:
            first['key'] = key
        return f_inc - lattice.investment_cost(z) - (1.0e7 if key == first['key'] else -1.0e6)
    out_d = run_continuation(lattice, dict(cache), key_of, dict(incumbent), _fake('certified', _q_success), sigma_q,
                             max_new_evaluations=10 ** 6, **quiet)
    hist = out_d['history']
    checks.update({
        'd_success_moves_the_incumbent': len(hist) >= 2 and hist[0]['decision'] == 'success'
        and hist[1]['incumbent']['label'] == hist[0]['next_incumbent'],
        'd_next_poll_is_unit_at_k_plus_1': len(hist) >= 2 and hist[1]['poll_size_delta'] == 1
        and hist[1]['poll_index'] == hist[0]['poll_index'] + 1,
        'd_terminates_with_a_certificate': out_d['termination']['reason'] == 'poll_failure_at_unit_mesh'
        and (out_d['termination_certificate'] or {}).get('holds') is True,
    })
    # (e) the completion path (trigger, add, cap refusal): at x = 0 every rounded direction is infeasible (ruling
    #     A2's rationale), so fewer than n + 1 feasible poll points remain
    x0 = lattice.x0()
    x0_inc = {'eval_key': key_of(x0), 'z': x0, 'label': 'x0', 'I': 0.0, 'Q': 1.0e8, 'F': 1.0e8, 'bar': 1.0,
              'source': 'synthetic'}
    rec_e, comp_e = build_poll(lattice, {}, key_of, x0_inc, FIRST_HALTON_K, sigma_q)
    rec_f, comp_f = build_poll(lattice, {}, key_of, x0_inc, FIRST_HALTON_K, sigma_q, completion_cap=0)
    out_f = run_continuation(lattice, {}, key_of, x0_inc, _fake('not_certified'), sigma_q, completion_cap=0, **quiet)
    checks.update({
        'e_completion_triggered_below_n_plus_1': (rec_e['n_distinct_feasible_poll_points'] < MIN_FEASIBLE_POLL_POINTS
                                                  and rec_e['completion_triggered'] is True),
        'e_completion_points_added_with_fields': (sum(c['poll_part'] == 'completion' for c in rec_e['candidates'])
                                                  == comp_e['n_feasible']
                                                  and all(f in rec_e['completion'] for f in
                                                          ('n_feasible', 'cap', 'over_cap', 'points'))),
        'e_completion_cap_refuses': (comp_f['over_cap'] is True
                                     and out_f['termination']['reason'] == 'STOP_FOR_REVIEW_completion_cap'
                                     and out_f['n_new_evaluations'] == 0),
        'e_trigger_is_strict_n_plus_1': MIN_FEASIBLE_POLL_POINTS == N_VARS + 1 == 8,
    })
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s53 F2 certificate): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': flex_side['harness_record_capture_checklist']}


# ======================================================================================================================
#  inputs shared by --freeze and --run
# ======================================================================================================================
def shared_constants():
    checks = {f'f51_{k}': v for k, v in F51.shared_constants().items()}
    checks.update({
        'f51_case_file_aa': F51.CASE_FILE_AA == CASE_FILE_AA,
        'f51_ageing': F51.ESS_AGEING_BASELINE == ESS_AGEING_BASELINE,
        'f51_cap': F51.CAP == CAP, 'f51_concurrency': F51.CONCURRENCY == CONCURRENCY == 5,
        'f51_cycles': F51.REQUIRED_CONSECUTIVE_CYCLES == REQUIRED_CONSECUTIVE_CYCLES == 10,
        'm_is_2': M_FLEX == 2.0, 'n_is_7': N_VARS == 7, 'two_n_is_14': N_DIRECTIONS == 14,
        'completion_cap_30': COMPLETION_CAP == 30, 'max_new_60': MAX_NEW_EVALUATIONS == 60,
        'unit_delta_is_pb_delta_min': DELTA_UNIT == PB.DELTA_MIN == 1, 'pb_mesh_1': PB.MESH_SIZE == 1,
        'halton_t0_17': PB.HALTON_T0 == 17,
    })
    return checks


def check_pins():
    out, failures = F51.check_pins()
    ok, state = _pin_state(SPEC_V23, True)
    out['spec_v23'] = state
    if not ok:
        failures.append(f'spec v23: {state}')
    return out, failures


def build_inputs(own_root_rel):
    failures, ev = [], {}
    shared = shared_constants()
    ev['shared_constants_with_reused_modules'] = shared
    failures += [f'constant differs from the module it is reused from: {k}' for k, v in shared.items() if not v]
    pins, more = check_pins()
    ev['pins'] = pins
    failures += more
    case = L._check_case_file_loads_to_declaration()
    ev['case_file_aa'] = case
    if not case['equals_declaration']:
        failures.append(f"case file AA {case['loaded']} != declaration {CASE_FILE_AA}")
    loaded = H.load_ess_ageing_parameters(os.path.join(REPO, H.ESS_PARAMS_FILE_REL))
    ev['ess_file_loads_to_declaration'] = (H.ess_ageing_canonical_text(loaded)
                                           == H.ess_ageing_canonical_text(ESS_AGEING_BASELINE))
    if not ev['ess_file_loads_to_declaration']:
        failures.append(f'ESS parameters file does not load to the declaration: {loaded}')
    memory_rule = L._memory_rule_matches_a0()
    ev['memory_rule_vs_a0_spec'] = memory_rule
    if not memory_rule['match']:
        failures.append(f"memory preflight rule differs from A0's measure: {memory_rule}")
    per_cycle = L.solves_per_cycle_from_case_file()
    ev['solves_per_cycle_from_case_file'] = per_cycle
    if per_cycle != 51:
        failures.append(f'solves per cycle from the case file {per_cycle} != 51')
    v23 = _load(SPEC_V23['path'])
    ev['spec_v23_item'] = v23[SPEC_V23['item']]
    ev['predictions_recorded_before_run'] = v23[SPEC_V23['item']].get('predictions_recorded_before_run')
    if not ev['predictions_recorded_before_run']:
        failures.append('spec v23 ruling4_F2_certificate records no predictions')

    w2 = _load(PB.INVESTMENT_COST_RESULTS['path'])
    costs = PB.unit_costs_from_w2(w2)
    years = tuple(sorted(costs))
    if years != (2025, 2030, 2035):
        failures.append(f'instance years from the W2 table {years} != (2025, 2030, 2035)')
    lattice = PB.Lattice(years, costs)
    ev['I_x_cross_check_vs_W2'] = PB.w2_cross_check(lattice, w2)
    if not ev['I_x_cross_check_vs_W2']['ok']:
        failures.append(f"I(x) closed form != W2 table: {ev['I_x_cross_check_vs_W2']}")
    sq = PB.sigma_q_from_tables(_load(PB.PHASE_A_TABLES['path']))
    ev['sigma_Q'] = dict(sq, inherited_scope=INHERITED_RULING_SCOPE)
    ev['degradation_clause'] = dict(PB.degradation_clause(lattice, sq['sigma_Q_eur']),
                                    **(PB.min_year_step_cost(lattice) or {}))
    if ev['degradation_clause']['triggered']:
        failures.append(f"STEP4 5.2 degradation clause triggers: {ev['degradation_clause']}; STOP")
    key_of = make_key_of(lattice)
    domain = lattice.domain()
    domain_labels = [spec_label(lattice, z) for z in domain]
    ev['domain'] = {'n_points': len(domain), 'n_storage_points': len(domain) - 1,
                    'rule': ('every budget-, bound- and duration-feasible common-year lattice point, x = 0 first; the '
                             'frozen spec holds exactly these, each at m = 2')}

    # ---- the cache: the three s51 sources through F51's loader, the s51 Phase B through ours ----
    case_sha = H.sha256_file(H.CASE_FILE)
    sources, accepted, rejected = [], [], []
    for src in F51.CACHE_SOURCES:
        ok_src, info, entries = F51.load_cache_source(src, case_sha, lattice)
        (accepted if ok_src else rejected).append(info)
        if ok_src:
            sources.append(entries)
    ok_src, info, entries = load_phase_b_source(case_sha, lattice)
    (accepted if ok_src else rejected).append(info)
    if ok_src:
        sources.append(entries)
    ev['cache_sources_accepted'] = accepted
    ev['cache_sources_rejected'] = rejected
    if rejected or len(accepted) != len(F51.CACHE_SOURCES) + 1:
        failures.append(f'cache source(s) refused: {[(r["name"], r.get("reason")) for r in rejected]}')
    cache, duplicates = PB.merge_cache(sources)
    ev['cache_duplicates'] = duplicates
    ev['cache_table'] = {k: {kk: v.get(kk) for kk in ('label', 'status', 'Q', 'bar', 'cycles_run',
                                                      'certification_cycle', 'flex_price_multiplier', 'canonical',
                                                      'source')}
                         for k, v in sorted(cache.items())}
    if len(cache) != N_CACHE_EXPECTED or duplicates:
        failures.append(f'cache holds {len(cache)} entries ({len(duplicates)} duplicates), expected '
                        f'{N_CACHE_EXPECTED} distinct')
    ev['baseline_exclusion'] = baseline_exclusion(set(cache), {key_of(z) for z in domain}, lattice)
    ev['baseline_exclusion']['every_cache_entry_is_m2'] = all(e.get('flex_price_multiplier') == M_FLEX
                                                             for e in cache.values())
    if not (ev['baseline_exclusion']['ok'] and ev['baseline_exclusion']['every_cache_entry_is_m2']):
        failures.append(f"baseline / foreign-record exclusion failed: {ev['baseline_exclusion']['key_collisions']} "
                        f"{ev['baseline_exclusion']['m2_points_outside_the_pinned_sources']} "
                        f"{ev['baseline_exclusion']['baseline_keys_overlapping_domain_or_cache']}")
    ev['budget_facts'] = budget_facts(lattice, cache)
    failures += [f'budget fact failed: {k}' for k, v in ev['budget_facts']['checks'].items() if not v]

    # ---- the incumbent: the committed s51 Phase B final incumbent (asserted), argmin-F recorded beside it ----
    inc, facts = continuation_facts(lattice, cache, key_of)
    ev['continuation_facts'] = facts
    failures += [f'continuation fact failed: {k}' for k, v in facts['checks'].items() if not v]
    _argmin, argmin_record = PB.initial_incumbent(lattice, cache, key_of(lattice.x0()), sq['sigma_Q_eur'])
    ev['argmin_F_over_the_cache'] = {'label': argmin_record['incumbent']['label'],
                                     'F': argmin_record['incumbent']['F'],
                                     'equals_the_continuation_incumbent': argmin_record['incumbent']['eval_key']
                                     == inc['eval_key'],
                                     'margin_vs_runner_up': argmin_record['margin_vs_runner_up'],
                                     'ranked': argmin_record['eligible_ranked'],
                                     'note': ('recorded, not used: the continuation starts from the committed s51 '
                                              'final incumbent (ruling), not from a re-ranking of the cache')}
    ev['initial_incumbent'] = {'rule': 'the committed s51 F2 Phase B final_incumbent (Addendum 39 ruling 1: "continue '
                                       'from the incumbent")',
                               'incumbent': {k: (list(v) if k == 'z' else v) for k, v in inc.items()}}
    try:
        ev['rule_eleven'] = rule_eleven(lattice, sq['sigma_Q_eur'], cache, key_of, inc)
    except AssertionError as error:
        ev['rule_eleven'] = {'checks': {}, 'error': str(error)}
        failures.append(str(error))
    return failures, ev, lattice, key_of, domain, domain_labels, cache, inc


# ======================================================================================================================
#  the frozen spec
# ======================================================================================================================
def campaign_root(campaign_id):
    return os.path.join(REPO, _P53, f'campaign_{campaign_id}')


def _script_pin(path, commit=None):
    return {'path': path, 'commit': commit, 'sha256': H.sha256_file(os.path.join(REPO, path))}


def _extra(ev, lattice, domain, dry, memory):
    return {'campaign_script': os.path.basename(__file__),
            'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
            's51_phase_b_script': _script_pin(PHASE_B_SOURCE['script']['path'], PHASE_B_SOURCE['script']['commit']),
            'phase_b_record_script': _script_pin(F51.PHASE_B_SCRIPT['path'], F51.PHASE_B_SCRIPT['commit']),
            'f2_ladder_script': _script_pin(F51.F2_LADDER_SCRIPT['path'], F51.F2_LADDER_SCRIPT['commit']),
            'label': LABEL, 'flex_label': FLEX_LABEL, 'scenario_label': SCENARIO_LABEL, 'stage': STAGE,
            'flex_price_multiplier': M_FLEX, 'spec_v23': dict(SPEC_V23), 'spec_v23_item': ev['spec_v23_item'],
            'predictions_recorded_before_run': ev['predictions_recorded_before_run'], 'pins': ev['pins'],
            'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE, 'stop_rule': STOP_RULE,
            'poll_rule': POLL_RULE, 'snap_rule': SNAP_RULE, 'iteration_rule': ITERATION_RULE,
            'certificate_statement': CERTIFICATE_STATEMENT, 'inherited_ruling_scope': INHERITED_RULING_SCOPE,
            'planner_rulings': dict(PB.PLANNER_RULINGS), 'open_items': list(PB.OPEN_ITEMS),
            'master_problem': {'nodes': list(PB.ACTIVE_NODES), 'years': list(lattice.years),
                               'granule_P_mva': PB.GRANULE_P_MVA, 'granule_E_mwh': PB.GRANULE_E_MWH,
                               'E_max_mwh': PB.E_MAX_MWH, 'duration_h': [2, 4], 'budget_eur': lattice.budget,
                               'form': 'single-cohort, ONE common investment year (n = 7)',
                               'unit_costs_discounted': {str(y): {'power': lattice.c_p[y], 'energy': lattice.c_e[y]}
                                                         for y in lattice.years}},
            'poll_design': {'design': POLL_DESIGN, 'n_vars': N_VARS, 'n_directions': N_DIRECTIONS,
                            'primes': list(PB.PRIMES[:N_VARS]), 'halton_t0': PB.HALTON_T0,
                            'halton_index': 't = t0 + k; k continues the s51 counter from k = 4',
                            'first_halton_k': FIRST_HALTON_K, 'rounding': 'd = round_half_away(h / ||h||_inf)',
                            'delta': DELTA_UNIT, 'mesh_size': PB.MESH_SIZE, 'on_success': 'Delta stays 1',
                            'min_feasible_poll_points': MIN_FEASIBLE_POLL_POINTS, 'completion_cap': COMPLETION_CAP,
                            'max_new_evaluations': MAX_NEW_EVALUATIONS, 'max_polls': MAX_POLLS,
                            'batch_size': CONCURRENCY, 'barrier_stop': [PB.BARRIER_STOP_PER_POLL,
                                                                        PB.BARRIER_STOP_OVERALL]},
            'sigma_Q': ev['sigma_Q'], 'degradation_clause': ev['degradation_clause'],
            'I_x_cross_check_vs_W2': ev['I_x_cross_check_vs_W2'], 'domain_summary': ev['domain'],
            'domain_I_x_eur': {spec_label(lattice, z): lattice.investment_cost(z) for z in domain},
            'cache_sources_accepted': ev['cache_sources_accepted'],
            'cache_sources_rejected': ev['cache_sources_rejected'], 'cache_duplicates': ev['cache_duplicates'],
            'cache_table': ev['cache_table'], 'baseline_exclusion': ev['baseline_exclusion'],
            'budget_facts': ev['budget_facts'], 'continuation_facts': ev['continuation_facts'],
            'initial_incumbent': ev['initial_incumbent'], 'argmin_F_over_the_cache': ev['argmin_F_over_the_cache'],
            'shared_constants_with_reused_modules': ev['shared_constants_with_reused_modules'],
            'solves_per_cycle_from_case_file': ev['solves_per_cycle_from_case_file'],
            'post_certification': 'none', 'model_variant': 'none (ageing); flexibility-price multiplier per entry',
            'expected_first_poll_dry_run': dry, 'rule_eleven': ev['rule_eleven']['checks'],
            'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS),
            'memory_preflight_rule': L.MEMORY_RULE, 'memory_preflight_rule_rationale': L.MEMORY_RULE_RATIONALE,
            'memory_preflight_refusing_at': '--run (non-gating at --freeze)', 'memory_at_freeze_non_gating': memory}


def validate_spec(spec, lattice, key_of, domain, domain_labels, ev):
    cfg = spec['configuration']
    extra = spec.get('extra') or {}
    entries = spec['candidates']
    pd = extra.get('poll_design') or {}
    checks = {
        'campaign_id': spec.get('campaign_id', '').startswith(DEFAULT_CAMPAIGN_ID),
        'n_entries_equal_domain': len(entries) == len(domain),
        'entries_in_domain_order': [e['label'] for e in entries] == domain_labels,
        'eval_keys_recompute_at_m2': all(e['eval_key'] == key_of(z) for e, z in zip(entries, domain)),
        'no_entry_is_the_baseline_key': all(
            e['eval_key'] != H.evaluation_key(e['key'], {}, case_file_aa=CASE_FILE_AA,
                                              ess_ageing_baseline=ESS_AGEING_BASELINE) for e in entries),
        'every_entry_carries_m2': all(e.get('flex_price_multiplier') == M_FLEX for e in entries),
        'every_entry_carries_the_flex_label': all(e.get('flex_price_label') == FLEX_LABEL for e in entries),
        'flex_label_at_spec_level': spec.get('flex_price_label') == FLEX_LABEL,
        'scenario_label_recorded': extra.get('scenario_label') == SCENARIO_LABEL,
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_pinned': ((cfg.get('ess_params_file') or {}).get('sha256') == L.ESS_PARAMS_FILE['sha256']),
        'overrides_empty': cfg.get('overrides') == {} and all(e.get('overrides') == {} for e in entries),
        'no_ageing_model_variant': ('model_variant_label' not in spec
                                    and not any('model_variant' in e for e in entries)),
        'post_certification_none': all(e.get('post_certification') is None for e in entries),
        'cap_500': spec.get('cap') == CAP, 'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,
        'ten_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_s39_D': cfg.get('arm_label') == ARM_LABEL,
        'budget_1e6_recorded': (extra.get('master_problem') or {}).get('budget_eur') == 1e6 == lattice.budget,
        'cache_table_frozen': extra.get('cache_table') == json.loads(json.dumps(ev['cache_table'])),
        'cache_sources_frozen': ([(a['name'], a['results']['sha256']) for a in extra.get('cache_sources_accepted') or []]
                                 == [(a['name'], a['results']['sha256']) for a in ev['cache_sources_accepted']]),
        'cache_holds_only_m2': all(e.get('flex_price_multiplier') == M_FLEX
                                   for e in (extra.get('cache_table') or {}).values()),
        'cache_size_26': len(extra.get('cache_table') or {}) == N_CACHE_EXPECTED,
        'baseline_exclusion_ok': ((extra.get('baseline_exclusion') or {}).get('ok') is True
                                  and (extra.get('baseline_exclusion') or {}).get('every_cache_entry_is_m2') is True),
        'budget_facts_frozen': all((extra.get('budget_facts') or {}).get('checks', {}).values()),
        'continuation_facts_frozen': all((extra.get('continuation_facts') or {}).get('checks', {}).values()),
        'initial_incumbent_frozen': ((extra.get('initial_incumbent') or {}).get('incumbent')
                                     == json.loads(json.dumps(ev['initial_incumbent']['incumbent']))),
        'sigma_Q_frozen': (extra.get('sigma_Q') or {}).get('sigma_Q_eur') == ev['sigma_Q']['sigma_Q_eur'],
        'poll_design_frozen': (pd.get('design') == POLL_DESIGN and pd.get('n_directions') == N_DIRECTIONS
                               and pd.get('halton_t0') == PB.HALTON_T0 and pd.get('first_halton_k') == FIRST_HALTON_K
                               and pd.get('delta') == DELTA_UNIT
                               and pd.get('min_feasible_poll_points') == MIN_FEASIBLE_POLL_POINTS
                               and pd.get('completion_cap') == COMPLETION_CAP
                               and pd.get('max_new_evaluations') == MAX_NEW_EVALUATIONS),
        'rules_frozen_verbatim': (extra.get('poll_rule') == POLL_RULE and extra.get('snap_rule') == SNAP_RULE
                                  and extra.get('iteration_rule') == ITERATION_RULE
                                  and extra.get('certificate_statement') == CERTIFICATE_STATEMENT
                                  and extra.get('inherited_ruling_scope') == INHERITED_RULING_SCOPE),
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'resolution_rule_recorded': extra.get('resolution_rule') == RESOLUTION_RULE,
        'predictions_recorded': extra.get('predictions_recorded_before_run') == ev['predictions_recorded_before_run'],
        'spec_v23_item_recorded': extra.get('spec_v23_item') == ev['spec_v23_item'],
        'domain_I_x_frozen': all(abs((extra.get('domain_I_x_eur') or {}).get(spec_label(lattice, z), -1.0)
                                     - lattice.investment_cost(z)) <= 1e-6 for z in domain),
        'dry_run_recorded': bool(extra.get('expected_first_poll_dry_run')),
        'memory_rule_recorded': extra.get('memory_preflight_rule') == L.MEMORY_RULE,
        'not_a_stub_spec': not extra.get('test_only_stub'),
    }
    return checks


def _guard_verifications():
    out = {'s53_parent': PARENT_GUARD.verify(0)}
    out.update(F51._guard_verifications())
    return out


def _guard_counts():
    out = {'s53_parent': dict(PARENT_GUARD.counts)}
    out.update(F51._guard_counts())
    return out


def _log_poll_plan(tag, plan):
    _log(f"[{tag}]   poll k={plan['poll_index']} t={plan['halton_t']} Delta={plan['Delta']} "
         f"incumbent={plan['incumbent']}")
    for s in plan['snap_table']:
        if s['result'] == 'feasible_as_rounded':
            what = f"FEASIBLE as rounded -> {s['rounded_label']}"
        elif s['result'] == 'snapped':
            what = (f"INFEASIBLE {s['rounded_infeasibility_reasons']} -> SNAPPED to {s['snapped_label']} "
                    f"(l1={s['snap_l1']}, linf={s['snap_linf']}, linf_from_inc={s['snapped_linf_from_incumbent']}, "
                    f"tied_at_min_l1={s['n_tied_at_min_l1']}, decided_by={s['decided_by']})")
        else:
            what = f"INFEASIBLE {s['rounded_infeasibility_reasons']} -> no feasible snap"
        _log(f"[{tag}]     dir {s['direction_index']:2d} d={s['direction']} rounded={s['rounded_z']}: {what}")
    _log(f"[{tag}]   distinct feasible poll points after snapping: {plan['n_distinct_feasible_poll_points']} "
         f"(threshold n + 1 = {MIN_FEASIBLE_POLL_POINTS}) -> completion triggered: {plan['completion_triggered']}"
         f"{'' if plan['completion_n_feasible'] is None else ' (' + str(plan['completion_n_feasible']) + ' points)'}")
    for c in plan['candidates']:
        _log(f"[{tag}]     [{c['part']}{'' if c['j'] is None else ' ' + str(c['j'])}] {c['label']} I={c['I_x_eur']} "
             f"-> {c['disposition']}{'' if c['cache_source'] is None else ' (cache: ' + c['cache_source'] + ')'}")
    _log(f"[{tag}]   cache hits {plan['n_cache_hits']}; NEW evaluations {plan['n_new_evaluations']} in batches "
         f"{plan['batch_sizes']} at concurrency {CONCURRENCY}")
    sp = plan['poll_set_spanning']
    _log(f"[{tag}]   poll displacement set: {sp['n_vectors']} vectors, rank {sp['rank']}, positively spans R^7: "
         f"{sp['positively_spans_R7']}; signed unit vectors not in its cone: {sp['signed_unit_vectors_not_in_cone']}")


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(campaign_id, started):
    tag = 'S53-F2-CERT'
    root = campaign_root(campaign_id)
    own_rel = os.path.relpath(root, REPO)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, ev, lattice, key_of, domain, domain_labels, cache, inc = build_inputs(own_rel)
    failures += more
    memory = F51.memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    dry = dry_run(lattice, cache, key_of, inc, ev['sigma_Q']['sigma_Q_eur'])
    entries = [(spec_label(lattice, z), lattice.nodes_map(z),
                {'investment_year': lattice.year_of(z), 'flex_price_multiplier': M_FLEX}) for z in domain]
    spec_path, spec_sha, _spec = H.freeze_campaign_spec(
        root, campaign_id, entries,
        configuration={'name': (f'F2 CERTIFICATE CONTINUATION -- {FLEX_LABEL} (m = {M_FLEX:g}) under {LABEL}: the '
                                'standard unit poll of Addendum 39 ruling 1 under the EUR 1,000,000 budget; the case '
                                'file (AA keep_memory in data/SRP1/SRP1_params.json) with the ESS ageing parameters '
                                'declared'),
                       'arm_label': ARM_LABEL, 'overrides': {},
                       'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('candidates = the WHOLE admissible domain (every budget-, bound- and '
                                'duration-feasible common-year lattice point, x = 0 first), each at m = 2, so one '
                                'frozen spec and one campaign lock cover every point the poll may reach; ONLY '
                                'polled, non-cached points are ever evaluated; no overrides; no ageing model '
                                'variant; no post-certification')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra=_extra(ev, lattice, domain, dry, memory))
    with open(spec_path) as handle:
        spec = json.load(handle)  # validate what is ON DISK
    checks = validate_spec(spec, lattice, key_of, domain, domain_labels, ev)
    guards = _guard_verifications()
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f'[{tag}] {STAGE}')
    _log(f'[{tag}] objective convention: {OBJECTIVE_CONVENTION}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha} '
         f'({len(domain)} domain entries, each at m = {M_FLEX:g})')
    _log(f'[{tag}] poll rule: {POLL_RULE}')
    _log(f'[{tag}] snap rule: {SNAP_RULE}')
    _log(f'[{tag}] iteration rule: {ITERATION_RULE}')
    _log(f'[{tag}] certificate: {CERTIFICATE_STATEMENT}')
    for source in ev['cache_sources_accepted']:
        _log(f"[{tag}] cache source {source['name']} ({source['what']}): {source['results']['path']} "
             f"sha256={source['results']['sha256']} commit={source['results'].get('commit')} "
             f"n={source['n_entries']}")
    for ekey, entry in ev['cache_table'].items():
        _log(f"[{tag}]   cache {entry['label']}: Q={entry['Q']} bar={entry['bar']} cycles={entry['cycles_run']} "
             f"m={entry['flex_price_multiplier']} eval_key={ekey[:16]} source={entry['source']['campaign']}")
    be = ev['baseline_exclusion']
    _log(f"[{tag}] baseline exclusion: ok={be['ok']} every_cache_entry_is_m2={be['every_cache_entry_is_m2']} "
         f"files={be['n_files']} collisions={be['key_collisions']} m2_elsewhere="
         f"{be['m2_points_outside_the_pinned_sources']} m_enters_the_key={be['m_enters_the_evaluation_key']}")
    _log(f"[{tag}] sigma_Q={ev['sigma_Q']['sigma_Q_eur']}; degradation clause triggered="
         f"{ev['degradation_clause']['triggered']}")
    _log(f"[{tag}] INCUMBENT (committed s51 final_incumbent): {ev['initial_incumbent']['incumbent']}")
    _log(f"[{tag}] budget slack at the incumbent: {ev['continuation_facts']['budget_slack_eur']}")
    _log(f"[{tag}] continuation facts: {ev['continuation_facts']['checks']}")
    box = ev['continuation_facts']['box_neighbourhood_spanning_non_gating']
    _log(f"[{tag}] the full box neighbourhood ({box['n_vectors']} points): rank {box['rank']}, positively spans R^7: "
         f"{box['positively_spans_R7']}; signed unit vectors not in its cone: {box['signed_unit_vectors_not_in_cone']}")
    _log(f"[{tag}] argmin F over the 26-entry cache (recorded, not used): {ev['argmin_F_over_the_cache']['label']} "
         f"equals the incumbent={ev['argmin_F_over_the_cache']['equals_the_continuation_incumbent']}; margin vs "
         f"runner-up {ev['argmin_F_over_the_cache']['margin_vs_runner_up']}")
    _log(f'[{tag}] DRY RUN (zero solves) -- the poll sequence from the incumbent up to the first poll that needs a '
         f'new evaluation:')
    if dry.get('complete_without_new_evaluations'):
        _log(f"[{tag}]   the whole continuation completes with NO new evaluation: {dry['termination']}")
        for plan in dry['polls']:
            _log_poll_plan(tag, plan)
    else:
        for plan in dry['earlier_polls']:
            _log_poll_plan(tag, plan)
        _log_poll_plan(tag, dry['first_evaluating_poll'])
        _log(f"[{tag}]   evaluation cap = {MAX_NEW_EVALUATIONS}; this poll needs "
             f"{dry['first_evaluating_poll']['n_new_evaluations']}")
    _log(f'[{tag}] DRY RUN (full record, zero solves): {json.dumps(dry, default=str)}')
    _log(f"[{tag}] rule eleven: {ev['rule_eleven']['checks']}")
    _log(f"[{tag}] shared constants: {ev['shared_constants_with_reused_modules']}")
    _log(f"[{tag}] inherited ruling scope: {INHERITED_RULING_SCOPE}")
    _log(f"[{tag}] predictions recorded before the run (spec v23): {ev['predictions_recorded_before_run']}")
    _log(f"[{tag}] memory at freeze (non-gating): {L._memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f'[{tag}] guards: {_guard_counts()}; verify0 failures={guards}; wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not any(guards.values())
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[{tag}] run with: --campaign-id {campaign_id} --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _log_poll(record):
    tag = 'S53-F2-CERT'
    inc = record['incumbent']
    _log(f"[{tag}] POLL k={record['poll_index']} t={record['halton_t']} Delta={record['poll_size_delta']}: "
         f"incumbent={inc['label']} F={inc['F']} I={inc['I']} Q={inc['Q']} bar={inc['bar']}; distinct feasible poll "
         f"points {record['n_distinct_feasible_poll_points']}; completion triggered {record['completion_triggered']}; "
         f"new={record['n_new_evaluations']} cache hits={record['n_cache_hits']}")
    for s in record['snap_table']:
        _log(f"[{tag}]   snap dir {s['direction_index']} d={s['direction']} rounded={s['rounded_z']} "
             f"feasible={s['rounded_feasible']} reasons={s['rounded_infeasibility_reasons']} -> {s['result']} "
             f"{s['snapped_label'] or ''} decided_by={s['decided_by']}")
    for cand in record['candidates']:
        _log(f"[{tag}]   [{cand['poll_part']}{'' if cand['direction_index'] is None else ' ' + str(cand['direction_index'])}] "
             f"label={cand['label']} I={cand['I_x_eur']} disposition={cand['disposition']} status={cand['status']} "
             f"Q={cand['Q_eur']} F={cand['F_eur']} F_inc-F={cand['F_inc_minus_F_eur']} bar_sum={cand['bar_sum_eur']} "
             f"sigma_Q={cand['sigma_Q_eur']} resolution={cand['resolution_eur']} -> {cand['outcome']}")
    _log(f"[{tag}]   decision={record['decision']} next_incumbent={record['next_incumbent']}")


def run(campaign_id, started, spec_sha256):
    tag = 'S53-F2-CERT'
    root = campaign_root(campaign_id)
    own_rel = os.path.relpath(root, REPO)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    root_contents = sorted(os.listdir(root))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, ev, lattice, key_of, domain, domain_labels, cache, inc = build_inputs(own_rel)
    failures += more
    checks = validate_spec(spec, lattice, key_of, domain, domain_labels, ev)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    extra = spec['extra'] or {}
    for what, frozen, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE)),
                              ('ESS params file', spec['configuration']['ess_params_file']['sha256'],
                               H.sha256_file(os.path.join(REPO, H.ESS_PARAMS_FILE_REL))),
                              ('this script', extra.get('campaign_script_sha256'),
                               H.sha256_file(os.path.abspath(__file__))),
                              ('the s51 Phase B launcher', (extra.get('s51_phase_b_script') or {}).get('sha256'),
                               H.sha256_file(os.path.join(REPO, PHASE_B_SOURCE['script']['path']))),
                              ('the Phase B record module', (extra.get('phase_b_record_script') or {}).get('sha256'),
                               H.sha256_file(os.path.join(REPO, F51.PHASE_B_SCRIPT['path']))),
                              ('the F2 ladder launcher', (extra.get('f2_ladder_script') or {}).get('sha256'),
                               H.sha256_file(os.path.join(REPO, F51.F2_LADDER_SCRIPT['path'])))):
        if frozen != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    memory = F51.memory_preflight()
    _log(f"[{tag}] memory preflight: {L._memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}; "
         f"vm_stat pages {memory['vm_stat_pages']}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {L._memory_line(memory)}')
    if failures:
        for failure in failures:
            _log(f'[{tag} PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    label_of_key = {e['eval_key']: e['label'] for e in spec['candidates']}
    entry_of_key = {e['eval_key']: e for e in spec['candidates']}
    sigma_q = ev['sigma_Q']['sigma_Q_eur']
    state_path = os.path.join(root, 'continuation_state.json')
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f'[{tag}] {STAGE}')
    _log(f'[{tag}] objective convention: {OBJECTIVE_CONVENTION}')
    _log(f'[{tag}] resolution rule: {RESOLUTION_RULE}')
    _log(f"[{tag}] incumbent: {ev['initial_incumbent']['incumbent']}")
    _log(f'[{tag}] sigma_Q={sigma_q}; inherited scope: {INHERITED_RULING_SCOPE}')
    _log(f'[{tag}] spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; git HEAD {head} '
         f'(spec frozen at {spec["git_head"]})')
    lock = H.acquire_campaign_lock(campaign_id, spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    new_points, batch_infos, history_so_far = {}, [], []
    ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)

    def evaluate_fn(batch):
        keys = [key_of(z) for z in batch]
        labels = [label_of_key[k] for k in keys]
        _log(f'[{tag}] evaluating {labels} (concurrency {ctx.concurrency}, m = {M_FLEX:g})')
        H.evaluate.last_batch_info = {}
        recs = H.evaluate(labels, ctx)
        batch_infos.append({'labels': labels, **getattr(H.evaluate, 'last_batch_info', {})})
        by_label = {(r or {}).get('candidate_label'): r for r in recs}
        out = []
        for ekey, label in zip(keys, labels):
            point = F51._point_result(label, by_label.get(label), entry_of_key[ekey])
            new_points[label] = point
            _log(f"[{tag}]   {label}: status={point['status']} cycles={point.get('cycles_run')} Q={point.get('Q')} "
                 f"bar={point.get('bar')} m_readback={(point.get('flex_price_readback') or {}).get('all_match')} "
                 f"rule_ten_gross={(point.get('rule_ten') or {}).get('terminal_gross_step_over_threshold')} "
                 f"solves={(point.get('solve_reconciliation') or {}).get('observed')}/"
                 f"{(point.get('solve_reconciliation') or {}).get('expected')}")
            entry = F51._cache_entry_from_point(point, ekey)
            entry['source'] = {'kind': 's53_certificate_new_evaluation_at_m2', 'eval_dir': point.get('eval_dir')}
            out.append(entry)
        return out

    def on_poll(record):
        history_so_far.append(record)
        _log_poll(record)
        H._atomic_write_json(state_path, {'spec_sha256': spec_sha256, 'campaign_id': campaign_id,
                                          'utc': datetime.now(timezone.utc).isoformat(),
                                          'flex_price_multiplier': M_FLEX, 'history': history_so_far,
                                          'new_points': new_points})

    try:
        result = run_continuation(lattice, cache, key_of, inc, evaluate_fn, sigma_q, on_poll=on_poll,
                                  batch_size=CONCURRENCY, log=_log)
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    harness_errors = [lab for lab, p in new_points.items()
                      if p['status'] not in ('certified', 'not_certified') or p.get('exit_code') != 0]
    readback_mismatch = [lab for lab, p in new_points.items() if p['status'] in ('certified', 'not_certified')
                         and not ((p.get('flex_price_readback') or {}).get('all_match')
                                  and (p.get('ess_ageing_readback_all_match') or {}).get('pre_run') is True
                                  and (p.get('ess_ageing_readback_all_match') or {}).get('post_run') is True)]
    solve_mismatch = [lab for lab, p in new_points.items() if p['status'] in ('certified', 'not_certified')
                      and not (p.get('solve_reconciliation') or {}).get('holds')]
    non_certified = [lab for lab, p in new_points.items() if p['status'] != 'certified']
    guards = _guard_verifications()
    reason = result['termination']['reason']
    results = {
        'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL, 'SCENARIO_LABEL': SCENARIO_LABEL, 'stage': STAGE,
        'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'campaign_id': campaign_id, 'flex_price_multiplier': M_FLEX,
        'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE, 'stop_rule': STOP_RULE,
        'poll_rule': POLL_RULE, 'snap_rule': SNAP_RULE, 'iteration_rule': ITERATION_RULE,
        'certificate_statement': CERTIFICATE_STATEMENT, 'inherited_ruling_scope': INHERITED_RULING_SCOPE,
        'STOP_FOR_REVIEW': reason.startswith('STOP_FOR_REVIEW'),
        'termination': result['termination'], 'termination_certificate': result['termination_certificate'],
        'final_incumbent': result['incumbent'], 'initial_incumbent': ev['initial_incumbent'],
        'continuation_facts': ev['continuation_facts'], 'argmin_F_over_the_cache': ev['argmin_F_over_the_cache'],
        'sigma_Q': ev['sigma_Q'], 'budget_facts': ev['budget_facts'], 'baseline_exclusion': ev['baseline_exclusion'],
        'n_polls': result['n_polls'], 'n_new_evaluations': result['n_new_evaluations'],
        'max_new_evaluations': MAX_NEW_EVALUATIONS,
        'n_barrier_new_evaluations': result['n_barrier_new_evaluations'],
        'final_poll_unresolved_indeterminate': result['final_poll_unresolved_indeterminate'],
        'final_poll_feasible_points': result['final_poll_feasible_points'],
        'lattice_neighbourhood_of_incumbent': result['lattice_neighbourhood_of_incumbent'],
        'claim_scope': ('on termination by poll failure the claim is the termination_certificate (CERTIFICATE_'
                        'STATEMENT; holds must be true): poll failure over the RECORDED poll set at unit mesh, NOT '
                        'the full-box certificate unless the completion was triggered; the unexamined box neighbours '
                        'and the spanning facts are recorded with it; on evaluation_budget_exhausted / '
                        'STOP_FOR_REVIEW / max_polls no certificate claim is made. Every figure is a figure of the '
                        'm = 2 FLEXIBILITY-PRICE SCENARIO under the EUR 1M budget.'),
        'poll_history': result['history'],
        'cache_table_at_start': ev['cache_table'], 'cache_sources_accepted': ev['cache_sources_accepted'],
        'points': new_points,
        'non_certified_points': non_certified, 'harness_errors': harness_errors,
        'readback_mismatch_points': readback_mismatch, 'solve_reconciliation_mismatch_points': solve_mismatch,
        'batch_info': batch_infos, 'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: ev[k] for k in ('pins', 'case_file_aa', 'cache_sources_accepted',
                                                'I_x_cross_check_vs_W2', 'degradation_clause',
                                                'shared_constants_with_reused_modules',
                                                'solves_per_cycle_from_case_file')},
        'rule_eleven_asserted_before_run': ev['rule_eleven']['checks'],
        'parent_solve_profile_guard': {'counts': _guard_counts(), 'verify_0_failures': guards},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(root, 'campaign_results.json'), results)
    manifest = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(root, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f"[{tag}] termination: {result['termination']}")
    _log(f"[{tag}] termination certificate: {result['termination_certificate']}")
    _log(f"[{tag}] final incumbent: {result['incumbent']}")
    _log(f"[{tag}] polls {result['n_polls']}; new evaluations {result['n_new_evaluations']} of "
         f"{MAX_NEW_EVALUATIONS}; barrier evaluations {result['n_barrier_new_evaluations']}")
    _log(f"[{tag}] unresolved indeterminate at the final poll: {result['final_poll_unresolved_indeterminate']}")
    if non_certified:
        _log(f'[{tag}] non-certified points (reported with cause): '
             f'{[(l, new_points[l].get("barrier_cause")) for l in non_certified]}')
    if readback_mismatch:
        _log(f'[{tag}] READ-BACK MISMATCH: {readback_mismatch}')
    if solve_mismatch:
        _log(f'[{tag}] SOLVE RECONCILIATION MISMATCH (reported): {solve_mismatch}')
    _log(f'[{tag}] guards: {_guard_counts()}; verify0 failures={guards}')
    if any(guards.values()) or harness_errors or readback_mismatch:
        _log(f'[{tag}] NOT OK guards={guards} harness_errors={harness_errors} readback={readback_mismatch}')
        sys.exit(1)
    if reason == 'poll_failure_at_unit_mesh' and not (result['termination_certificate'] or {}).get('holds'):
        _log(f"[{tag}] NOT OK: the certificate does not hold {result['termination_certificate']}")
        sys.exit(1)
    if reason.startswith('STOP_FOR_REVIEW'):
        _banner([f'STOP_FOR_REVIEW: {reason}', json.dumps(result['termination'], default=str)])
        sys.exit(3)
    if reason != 'poll_failure_at_unit_mesh':
        _log(f'[{tag}] terminated WITHOUT a certificate claim: {reason}')
        sys.exit(2)
    _log(f'[{tag}] OK: poll failure at unit mesh; the certificate holds (scope: CERTIFICATE_STATEMENT)')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: freeze + validate the spec only')
    mode.add_argument('--run', action='store_true', help='run the frozen spec named by --spec-sha256')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--campaign-id', default=DEFAULT_CAMPAIGN_ID)
    args = parser.parse_args()
    if not CAMPAIGN_ID_PATTERN.match(args.campaign_id):
        parser.error(f'--campaign-id must match {CAMPAIGN_ID_PATTERN.pattern}')
    started = time.time()
    os.chdir(REPO)
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        freeze(args.campaign_id, started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(args.campaign_id, started, args.spec_sha256)


if __name__ == '__main__':
    main()
