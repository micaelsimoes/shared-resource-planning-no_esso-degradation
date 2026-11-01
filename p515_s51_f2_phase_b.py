"""
P5.15 Addendum 38 (task W40) -- F2 PHASE B: the STEP4 section 5 MADS poll of the DEMONSTRATION CASE, under the
flexibility-price multiplier m = 2 and the EUR 1M budget, from the best budget-feasible point of the committed
F2 ladder.

  MODEL VARIANT -- flexibility price x m   (m = 2; the DSO `cost_flex` profile, every year / day / hour, growth
  included, multiplied uniformly in the child before the DSO models are built; no workbook edit). m = 2 is a
  FLEXIBILITY-PRICE SCENARIO and is NEVER the baseline; every table of this stage is labelled as such. m enters
  the EVALUATION KEY (`H.evaluation_key(..., flex_price_multiplier=m)`), so no baseline (m = 1) record can ever
  be read as a cache entry here -- asserted, not assumed (`baseline_exclusion`).
  Ageing: the BASELINE (C2 + phi_cal 0.985 + soh_min 0.70), declared (`configuration.ess_ageing_baseline`).

AUTHORITY: PLANNER_BRIEF_2026-09-13.md Addendum 38; `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`
F2 ("phase_B": "from the best budget-feasible ladder point, under the EUR 1M budget, own frozen spec, hull polish
on the incumbent, the unit-poll completion ruling (A2) in force"); Planner task W40; STEP4_DFO_METHOD.md sections
1.2, 1.3, 2.4, 2.6, 3, 5.2-5.5, 6, 8.

================================================================================
WHAT IS REUSED, BY IMPORT
================================================================================
The MADS / OrthoMADS machinery and every Planner ruling A1-A11 frozen with it come from
`p515_s47_phase_b_record.py` (PB) BY IMPORT and are NOT re-implemented or re-decided here:
  PB.Lattice           the 7-variable single-cohort lattice, its constraints (A1) and I(x) (the pinned W2 table)
  PB.poll_directions   OrthoMADS n+1 NEG, t = t0 + k, t0 = p_7 = 17, d = round(Delta h / ||h||_inf)   (A5)
  PB.run_mads          the full poll, Delta_0 = 4, double / halve, never below 1                      (A3)
                       the UNIT-POLL COMPLETION with cap 30 -> STOP FOR REVIEW, never truncated       (A2)
                       the improvement threshold max(bar_x + bar_inc, sigma_Q); indeterminate recorded (A4)
                       the extreme barrier and the barrier stop rule (2 per poll, 3 overall)          (A7)
                       MAX_NEW_EVALUATIONS = 20, MAX_POLLS = 60                                       (A8)
                       the cache-by-EVALUATION-KEY and the termination certificate
  PB.initial_incumbent argmin F over the certified, on-lattice, BUDGET-FEASIBLE cache (ties: lower I, then label)
  PB.expected_first_poll   the zero-solve dry run of the poll sequence
  PB.sigma_q_from_tables / PB.degradation_clause / PB.merge_cache / PB.rule_eleven (record-shape checklist)
A9 (the NOMAD comparison) remains the OPEN ITEM it was; A2 option (b) remains REFERRED TO THE AUTHOR.
The one thing that is NOT PB's is the EVALUATION KEY: here it carries m = 2 (`make_key_of`), so PB's own
`eval_key_of` is never used. Everything else PB decides, PB still decides.

The flexibility-price mechanism comes from `p515_s49_flex_ladder_campaign.py` (L) and the F2 ladder launcher
`p515_s51_f2_ladder_campaign.py` (F) BY IMPORT: the per-entry `flex_price_multiplier` evaluation option and its
MODEL VARIANT labelling (W33, 2e82a23f; two-cycle bitwise gate at m = 1.0, 51e74279), L._point_result (the record
reader: flexibility-price read-back pre- and post-run, flexibility cost per DSO, rule ten, solve reconciliation,
ageing read-back, wall time, peak RSS, AA action counts, per-cycle trajectory), L.memory_preflight (A0's measure
at concurrency 5) and L.rule_eleven (the harness-side capture checklist).

================================================================================
THE CACHE (STEP4 2.6, 5.4) -- every committed CERTIFIED record at m = 2, and nothing else
================================================================================
  x = 0, E = 1 MWh   s49 flexibility-price ladder      dc468ab4   P515S49/campaign_s49_flex_ladder
  E = 2 MWh          s50 marginal-MWh test             0e36ec0a   P515S50/campaign_s50_marginal
  E = 3, 4, 5 MWh    s51 F2 ladder                     a3a9fb47   P515S51/campaign_s51_f2_ladder
Each source is pinned by sha256 (results, spec, manifest, launcher), re-checked at --freeze and --run, its commit
asserted an ancestor of HEAD, its spec asserted to declare the SAME configuration (ageing baseline, AA case file,
ESS params sha256, case-file sha256, cap 500, 10 cycles, arm s39_D, no overrides, no ageing model variant, no
post-certification) and the flexibility-price label, its points cross-checked against their own
`evaluation_record.json` and their campaign manifest, and EVERY eval key RECOMPUTED here from the committed
`candidate_canonical` at m = 2 and asserted equal to the committed one. Baseline (m = 1) records are NOT cache:
`baseline_exclusion` scans every committed `campaign_results.json` under data/SRP1/Results and asserts that no
point outside these three sources has an eval key equal to any cache key or to ANY key of the admissible domain,
and, separately, that the m = 2 key of every domain point differs from its m-absent (baseline) key.

================================================================================
THE BUDGET (this is PHASE B)
================================================================================
I(x) <= B = 1,000,000 EUR is a CONSTRAINT here (PB.Lattice.reasons), not a report column. The F2 ladder's E = 4
(I = 1,271,828.03) and E = 5 (I = 1,589,785.04) MWh points are therefore INFEASIBLE: they are excluded from the
initial incumbent with their budget reason, they are not in the admissible domain and hence not in the frozen
spec at all, and the extreme barrier rejects any polled point of that kind BEFORE evaluation (STEP4 2.4, 5.5).
Asserted before the run (`budget_facts`), never assumed.

INITIAL INCUMBENT: argmin F over the budget-feasible certified cache, F(x) = I(x) + Q(x) (PB.initial_incumbent).
At fixed m this is the same ordering as the ladder's value - I, because F(0) - F(x) = (Q_m(0) - Q_m(x)) - I(x)
and Q_m(0) is common to every rung: the identity is CHECKED against the committed F2 ladder rows to 1e-6 EUR
(`incumbent_matches_committed_ladder`). So the incumbent is the ladder's largest budget-feasible value - I,
E = 3 MWh -- the BUDGET CORNER.

TWO MODES, attached, both streams captured, never detached, one at a time (the campaign lock):
  --freeze                    ZERO SOLVES: preconditions, pins, the cache and its exclusions, the budget facts,
                              sigma_Q and the STEP4 5.2 degradation clause, the initial incumbent, rule eleven,
                              `freeze_campaign_spec` + validation, and the ZERO-SOLVE DRY RUN of the poll
                              sequence (exactly which points the first polls evaluate and how many are new).
  --run --spec-sha256 <sha>   loads THAT spec (the root must hold only it), re-checks everything plus the
                              harness / case-file / ESS-params / script sha256s, the memory preflight (refusing),
                              takes the campaign lock, runs the poll, writes campaign_results.json and
                              campaign_manifest_sha256.json. State is rewritten atomically after every poll.
The parent never solves: this module's SolveProfileGuard(permitted=()) is installed on top of PB's, F's, M's and
L's parent guards and W25's module guard; all six are verified at exactly 0.

Exit codes (--run): 0 terminated by the unit-poll failure with its certificate holding; 2 evaluation budget /
MAX_POLLS reached; 3 STOP_FOR_REVIEW (barrier stop rule or completion cap); 1 harness / guard / read-back /
precondition failure, or a unit-poll termination whose certificate does not hold.

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s51_f2_phase_b.py --freeze \\
      > data/SRP1/Results/P515S51/campaign_s51_f2_phase_b_freeze_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s51_f2_phase_b.py --run \\
      --spec-sha256 <sha> > data/SRP1/Results/P515S51/campaign_s51_f2_phase_b_launch.log 2>&1
"""

import argparse
import glob
import inspect
import json
import os
import re
import subprocess
import sys
import time

from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The F2 ladder launcher imports the marginal launcher, which imports the flexibility-price ladder launcher; each
# installs ITS parent guard (permitted=()) at import, before the harness and hence before any model module. The
# Phase B record module installs its own the same way. This module's guard goes on top of all of them; every one
# is verified at exactly 0.
import p515_s51_f2_ladder_campaign as F  # noqa: E402  (imports M, which imports L)
import p515_s50_marginal_campaign as M  # noqa: E402
import p515_s49_flex_ladder_campaign as L  # noqa: E402
import p515_s47_phase_b_record as PB  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S51 F2 Phase B parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

LABEL = L.LABEL
FLEX_LABEL = H.FLEX_PRICE_LABEL
SCENARIO_LABEL = F.SCENARIO_LABEL
STAGE = ('P5.15 Addendum 38 (W40) -- F2 PHASE B: the STEP4 section 5 MADS poll at the flexibility price m = 2 '
         'under the EUR 1M budget, from the best budget-feasible point of the committed F2 ladder')

_P51 = os.path.join('data', 'SRP1', 'Results', 'P515S51')
DEFAULT_CAMPAIGN_ID = 's51_f2_phase_b'
CAMPAIGN_ID_PATTERN = re.compile(r'^s51_f2_phase_b(_r[0-9]+)?$')

M_FLEX = F.M_FLEX                                   # 2.0 -- the single flexibility-price multiplier of F2
CASE_FILE_AA = dict(L.CASE_FILE_AA)
ESS_AGEING_BASELINE = dict(L.ESS_AGEING_BASELINE)
CAP = 500
CONCURRENCY = 5
REQUIRED_CONSECUTIVE_CYCLES = 10
ARM_LABEL = 's39_D'
CASE_JSON_REL = L.CASE_JSON_REL

# ---- the pinned cache sources: every committed CERTIFIED record at m = 2 ----
F2_LADDER_RESULTS = {'path': os.path.join(_P51, 'campaign_s51_f2_ladder', 'campaign_results.json'),
                     'sha256': 'eb7513dde5d505bc10d6a5ab7c5fdab2ce11859c1a58941d5dc5e1a5b0fdbebd',
                     'commit': 'a3a9fb47',
                     'campaign': 's51_f2_ladder (spec campaign_spec_s51_f2_ladder_c4455767)'}
F2_LADDER_SPEC = {'path': os.path.join(_P51, 'campaign_s51_f2_ladder', 'campaign_spec_s51_f2_ladder_c4455767.json'),
                  'sha256': 'c4455767d58955c523eef59973f54442daac70a0dd7da59e2ef9bd46784fe87e', 'commit': 'a3a9fb47'}
F2_LADDER_MANIFEST = {'path': os.path.join(_P51, 'campaign_s51_f2_ladder', 'campaign_manifest_sha256.json'),
                      'sha256': '6a2ec200c7f620037a3c8d5c9a6f3a62d1a5fb03d35ffe59ba0ce454b3801d32', 'commit': 'a3a9fb47'}
F2_LADDER_SCRIPT = {'path': 'p515_s51_f2_ladder_campaign.py', 'commit': 'a3ad7699'}
PHASE_B_SCRIPT = {'path': 'p515_s47_phase_b_record.py', 'commit': '4b8854b0'}

CACHE_SOURCES = (
    {'name': 's49_flex_ladder', 'results': dict(M.LADDER_RESULTS), 'spec': dict(M.LADDER_SPEC),
     'manifest': dict(M.LADDER_MANIFEST), 'script': dict(M.LADDER_SCRIPT),
     'expected_m2_labels': ('n7_4h_e1_m2', 'x0_m2'),
     'what': 'the flexibility-price ladder: x = 0 and E = 1 MWh at m = 2'},
    {'name': 's50_marginal', 'results': dict(F.MARGINAL_RESULTS), 'spec': dict(F.MARGINAL_SPEC),
     'manifest': dict(F.MARGINAL_MANIFEST), 'script': dict(F.MARGINAL_SCRIPT),
     'expected_m2_labels': ('n7_4h_e2_m2',),
     'what': 'the marginal-MWh test: E = 2 MWh at m = 2'},
    {'name': 's51_f2_ladder', 'results': dict(F2_LADDER_RESULTS), 'spec': dict(F2_LADDER_SPEC),
     'manifest': dict(F2_LADDER_MANIFEST), 'script': dict(F2_LADDER_SCRIPT),
     'expected_m2_labels': ('n7_4h_e3_m2', 'n7_4h_e4_m2', 'n7_4h_e5_m2'),
     'what': 'the F2 ladder: E = 3 (the budget corner), 4 and 5 MWh at m = 2'},
)
# Expected values, ASSERTED (never trusted from the task text): the committed Q at m = 2 of every cache point.
Q_CACHE_EXPECTED = {'x0_m2': 811016062.203051, 'n7_4h_e1_m2': 810649709.6671975,
                    'n7_4h_e2_m2': 810288986.5544469, 'n7_4h_e3_m2': 809936004.2872474,
                    'n7_4h_e4_m2': 809571301.5214276, 'n7_4h_e5_m2': 809202421.3581351}
BUDGET_FEASIBLE_CACHE_LABELS = ('x0_m2', 'n7_4h_e1_m2', 'n7_4h_e2_m2', 'n7_4h_e3_m2')
BUDGET_INFEASIBLE_CACHE_LABELS = ('n7_4h_e4_m2', 'n7_4h_e5_m2')
EXPECTED_INCUMBENT_LABEL = 'y2025__n7_p0.75_e3'        # the budget corner, 0.75 MVA / 3.0 MWh at 2025
LADDER_IDENTITY_TOL_EUR = 1e-6

BASELINE_EXCLUSION_GLOB = os.path.join('data', 'SRP1', 'Results', '**', 'campaign_results.json')

OBJECTIVE_CONVENTION = (
    'Q(x) = certified_cost = gross_operational_cost (settlement-excluded) AT m = 2; F(x) = I(x) + Q(x); I(0) = 0 '
    'so F(0) = Q(0), Q(0) the committed x = 0 record at m = 2 (dc468ab4). Terminal salvage and '
    'net_operational_recourse = gross - salvage are reported by the records and EXCLUDED from F (Addendum 27 item '
    '3). At m = 2 the DSO flexibility cost inside Q is priced at 2 x cost_flex -- every figure here is a figure '
    'of that SCENARIO, not of the baseline.')
RESOLUTION_RULE = PB.RESOLUTION_RULE
STOP_RULE = PB.STOP_RULE
COMPLETION_RULE = PB.COMPLETION_RULE
TERMINATION_CERTIFICATE = PB.TERMINATION_CERTIFICATE
PLANNER_RULINGS = dict(PB.PLANNER_RULINGS)
OPEN_ITEMS = list(PB.OPEN_ITEMS)
METHOD_CHANGES_REFERRED_TO_AUTHOR = list(PB.METHOD_CHANGES_REFERRED_TO_AUTHOR)
INHERITED_RULING_SCOPE = (
    'The rulings A1-A11 are inherited from the Phase B formal record (p515_s47_phase_b_record.py, W24) BY IMPORT '
    'and are not re-decided here. TWO of them are inherited with a stated scope: (i) A6 pins sigma_Q = '
    'phase_a_tables T3 residual_max_abs_eur = 18,449.66 EUR, measured under the C3-era Phase A fit and NOT '
    're-measured at m = 2; at m = 2 the record bars are O(10^3) EUR, so the improvement threshold max(bar_x + '
    'bar_inc, sigma_Q) is sigma_Q for essentially every comparison of this stage -- it is the binding half of the '
    'rule here, and it is an inherited constant, not a measurement of this scenario. (ii) A8 pins '
    'MAX_NEW_EVALUATIONS = 20; the dry run records exactly how far that budget reaches from this incumbent.')
INCUMBENT_ORDERING_NOTE = (
    'F(x) = I(x) + Q(x) and F(0) - F(x) = (Q_m(0) - Q_m(x)) - I(x) = value(x) - I(x) of the committed ladder, '
    'because Q_m(0) is common to every rung at fixed m. So argmin F over the budget-feasible certified cache is '
    'the ladder rung with the largest budget-feasible value - I. The identity is CHECKED here against the '
    'committed F2 ladder rows to 1e-6 EUR (the two differ only by floating-point association).')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 38 (F2, the demonstration case)',
    'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json F2.phase_B',
    'STEP4_DFO_METHOD.md sections 1.2, 1.3, 2.4, 2.6, 3, 5.2-5.5, 6, 8',
    'Planner task W40 (Phase B at m = 2 under the EUR 1M budget; concurrency 5, cap 500, 10 cycles, AA-on case '
    'file, baseline ageing declared, m = 2 declared; the committed m = 2 records are the cache, matched by '
    'evaluation key and pinned by sha256; baseline (m = 1) records are NOT cache and are asserted excluded)',
    'Planner rulings A1-A11 (task W24) as frozen in p515_s47_phase_b_record.py and imported here',
    '2e82a23f (harness flexibility-price option + zero-solve checks), 51e74279 (two-cycle bitwise gate at m = 1.0), '
    '812ee116 / 55ff2cca / dc468ab4 (flexibility-price ladder), 2389d79b / 0e36ec0a (marginal-MWh test), '
    'a3ad7699 / c4455767 / a3a9fb47 (the F2 ladder), 4b8854b0 (the Phase B record machinery)',
]
EXTRA_CLEAN_FILES = (os.path.basename(__file__), PHASE_B_SCRIPT['path'], F2_LADDER_SCRIPT['path'],
                     M.LADDER_SCRIPT['path'], F.MARGINAL_SCRIPT['path'], H.ESS_PARAMS_FILE_REL,
                     'shared_energy_storage_parameters.py', 'shared_energy_storage.py', CASE_JSON_REL)

POLL_RECORD_FIELDS = tuple(PB.POLL_RECORD_FIELDS)
CANDIDATE_RECORD_FIELDS = tuple(PB.CANDIDATE_RECORD_FIELDS)
POINT_RECORD_FIELDS = ('status', 'barrier_cause', 'cycles_run', 'certification_cycle', 'Q', 'bar', 'rule_ten',
                       'flex_price_multiplier', 'flex_price_label_in_record', 'flex_price_readback', 'flex_cost',
                       'solve_reconciliation', 'ess_ageing_readback_all_match', 'wall_time', 'peak_rss',
                       'aa_action_counts', 'per_cycle_trajectory', 'net_operational_recourse',
                       'terminal_salvage_value')
CAMPAIGN_RESULT_FIELDS = (('termination', 'termination_certificate', 'final_incumbent', 'initial_incumbent',
                           'sigma_Q', 'cache_table', 'budget_facts', 'baseline_exclusion', 'poll_history', 'points')
                          + tuple(f'poll_history[].{f}' for f in POLL_RECORD_FIELDS)
                          + tuple(f'poll_history[].candidates[].{f}' for f in CANDIDATE_RECORD_FIELDS)
                          + tuple(f'points[].{f}' for f in POINT_RECORD_FIELDS))

_log = L._log
_banner = L._banner
_load = L._load


# ======================================================================================================================
#  identity: the evaluation key carries m = 2
# ======================================================================================================================
def eval_key_of_canonical(canonical, m=M_FLEX):
    return H.evaluation_key(H.candidate_key(canonical), {}, case_file_aa=CASE_FILE_AA,
                            ess_ageing_baseline=ESS_AGEING_BASELINE, flex_price_multiplier=m)


def make_key_of(lattice, m=M_FLEX):
    memo = {}

    def key_of(z):
        z = lattice.canonical_z(z)
        if z not in memo:
            memo[z] = eval_key_of_canonical(PB.canonical_of(lattice, z), m)
        return memo[z]
    return key_of


def spec_label(lattice, z):
    return f'{lattice.label(z)}_m{L._m_tag(M_FLEX)}'


def _git_state(rel):
    return PB._git_state(rel)


def _commit_in_head(commit):
    return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO,
                          capture_output=True).returncode == 0


# ======================================================================================================================
#  pins
# ======================================================================================================================
def check_pins():
    """This stage's own pins, on top of the F2 ladder launcher's (F._check_pins: spec v21, the s49 / s50 sources,
    the ESS params file, the A0 spec, the W2 table, W25 and the W33 priors) and the Phase B record module's
    (PB._check_pins: spec v17 / v15, the cost file, the A0 x = 0 record, the W2 table, the Phase A tables that
    carry sigma_Q, and the baseline identity / case-file gates)."""
    out, failures = {}, []
    f_pins, more = F._check_pins()
    out['f2_ladder_launcher_pins'] = f_pins
    failures += [f'F2 ladder launcher pin: {m}' for m in more]
    pb_pins, more = PB._check_pins()
    out['phase_b_record_pins'] = pb_pins
    failures += [f'Phase B record pin: {m}' for m in more]
    for name, pin in (('f2_ladder_results', F2_LADDER_RESULTS), ('f2_ladder_spec', F2_LADDER_SPEC),
                      ('f2_ladder_manifest', F2_LADDER_MANIFEST)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked, clean = _git_state(pin['path'])
        in_head = _commit_in_head(pin['commit'])
        entry = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                 'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': clean,
                 'commit': pin['commit'], 'commit_in_HEAD': in_head}
        out[name] = entry
        if not (entry['match'] and tracked and clean and in_head):
            failures.append(f'{name}: {entry}')
    for name, pin in (('f2_ladder_script', F2_LADDER_SCRIPT), ('phase_b_record_script', PHASE_B_SCRIPT)):
        path = os.path.join(REPO, pin['path'])
        tracked, clean = _git_state(pin['path'])
        in_head = _commit_in_head(pin['commit'])
        entry = {'path': pin['path'], 'sha256_on_disk': H.sha256_file(path) if os.path.isfile(path) else None,
                 'git_tracked': tracked, 'git_clean': clean, 'commit': pin['commit'], 'commit_in_HEAD': in_head}
        out[name] = entry
        if not (os.path.isfile(path) and tracked and clean and in_head):
            failures.append(f'{name}: {entry}')
    return out, failures


def shared_constants():
    """Every constant this stage shares with the module it reuses must BE that module's constant."""
    checks = {
        'pb_case_file_aa': PB.CASE_FILE_AA == CASE_FILE_AA,
        'pb_ess_ageing_baseline': PB.ESS_AGEING_BASELINE == ESS_AGEING_BASELINE,
        'pb_cap': PB.CAP == CAP, 'pb_concurrency': PB.CONCURRENCY == CONCURRENCY == 5,
        'pb_required_consecutive_cycles': PB.REQUIRED_CONSECUTIVE_CYCLES == REQUIRED_CONSECUTIVE_CYCLES,
        'pb_arm_label': PB.ARM_LABEL == ARM_LABEL,
        'pb_budget_1e6': PB.BUDGET_EUR == 1e6,
        'pb_active_nodes': PB.ACTIVE_NODES == L.ACTIVE_NODES,
        'pb_delta0_4': PB.DELTA_0 == 4, 'pb_delta_min_1': PB.DELTA_MIN == 1,
        'pb_halton_t0_17': PB.HALTON_T0 == 17, 'pb_poll_design': PB.POLL_DESIGN == 'orthomads_n_plus_1_neg',
        'pb_unit_poll_completion': PB.UNIT_POLL_COMPLETION is True, 'pb_completion_cap_30': PB.COMPLETION_CAP == 30,
        'pb_max_new_evaluations_20': PB.MAX_NEW_EVALUATIONS == 20, 'pb_max_polls_60': PB.MAX_POLLS == 60,
        'pb_barrier_stop': (PB.BARRIER_STOP_PER_POLL, PB.BARRIER_STOP_OVERALL) == (2, 3),
        'l_label': L.LABEL == LABEL, 'l_case_file_aa': L.CASE_FILE_AA == CASE_FILE_AA,
        'l_ess_ageing_baseline': L.ESS_AGEING_BASELINE == ESS_AGEING_BASELINE,
        'l_cap': L.CAP == CAP, 'l_concurrency': L.CONCURRENCY == CONCURRENCY,
        'l_required_consecutive_cycles': L.REQUIRED_CONSECUTIVE_CYCLES == REQUIRED_CONSECUTIVE_CYCLES,
        'l_year_2025': L.YEAR == 2025, 'l_active_nodes': L.ACTIVE_NODES == PB.ACTIVE_NODES,
        'f_m_is_2': F.M_FLEX == M_FLEX == 2.0, 'f_m_is_a_ladder_multiplier': M_FLEX in L.MULTIPLIERS,
        'f_cap': F.CAP == CAP, 'f_scenario_label': F.SCENARIO_LABEL == SCENARIO_LABEL,
        'flex_label_is_the_harness_label': FLEX_LABEL == H.FLEX_PRICE_LABEL,
        'm_enters_the_key': H.flex_price_multiplier_in_key(M_FLEX) == M_FLEX,
        'm_1_does_not_enter_the_key': H.flex_price_multiplier_in_key(1.0) is None,
    }
    return checks


# ======================================================================================================================
#  the cache: the committed CERTIFIED records at m = 2
# ======================================================================================================================
def _spec_declares_the_same_configuration(spec, case_file_sha256):
    cfg = spec.get('configuration') or {}
    entries = spec.get('candidates') or []
    return {
        'ess_ageing_baseline': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'case_file_aa': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_params_sha256': (cfg.get('ess_params_file') or {}).get('sha256') == L.ESS_PARAMS_FILE['sha256'],
        'case_file_sha256': cfg.get('case_file_sha256') == case_file_sha256,
        'overrides_empty': cfg.get('overrides') == {},
        'arm_label': cfg.get('arm_label') == ARM_LABEL,
        'cap': spec.get('cap') == CAP,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'flex_price_label_at_spec_level': spec.get('flex_price_label') == FLEX_LABEL,
        'no_ageing_model_variant': ('model_variant_label' not in spec
                                    and not any('model_variant' in e for e in entries)),
        'no_per_entry_overrides': all((e.get('overrides') or {}) == {} for e in entries),
        'no_post_certification': all(e.get('post_certification') is None for e in entries),
    }


def load_cache_source(src, case_file_sha256, lattice):
    """One pinned source -> (ok, info, entries{eval_key: cache entry}). ZERO SOLVES; reads committed files only."""
    info = {'name': src['name'], 'what': src['what'], 'results': dict(src['results']), 'spec': dict(src['spec']),
            'manifest': dict(src['manifest']), 'script': dict(src['script'])}
    reasons = []
    for what, pin, need_sha in (('results', src['results'], True), ('spec', src['spec'], True),
                                ('manifest', src['manifest'], 'sha256' in src['manifest']),
                                ('script', src['script'], False)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked, clean = _git_state(pin['path'])
        info[f'{what}_state'] = {'sha256_on_disk': got, 'git_tracked': tracked, 'git_clean': clean}
        if got is None or not tracked or not clean:
            reasons.append(f'{what} missing / not tracked-and-clean')
        if need_sha and got != pin.get('sha256'):
            reasons.append(f'{what} sha256 {got} != pinned {pin.get("sha256")}')
    if src['results'].get('commit') and not _commit_in_head(src['results']['commit']):
        reasons.append(f"results commit {src['results']['commit']} is not an ancestor of HEAD")
    if reasons:
        return False, dict(info, reason='; '.join(reasons)), {}
    results = _load(src['results']['path'])
    spec = _load(src['spec']['path'])
    manifest = _load(src['manifest']['path'])
    checks = {'spec_sha256_matches_results': results.get('campaign_spec_sha256') == src['spec']['sha256'],
              'spec_path_matches_results': results.get('campaign_spec_path') == src['spec']['path'],
              'no_stop_for_review': results.get('STOP_FOR_REVIEW') is False,
              'no_non_certified_points': results.get('non_certified_points') == [],
              'no_not_launched_points': results.get('not_launched_points') == [],
              'no_harness_errors': results.get('harness_errors') == [],
              'no_readback_mismatch': results.get('readback_mismatch_points') == [],
              'no_solve_reconciliation_mismatch': results.get('solve_reconciliation_mismatch_points') == []}
    checks.update(_spec_declares_the_same_configuration(spec, case_file_sha256))
    record_files = {k: v for k, v in manifest.items()
                    if os.path.basename(k) in ('evaluation_record.json', 'per_cycle_record.jsonl')}
    bad = sorted(k for k, v in record_files.items() if not os.path.isfile(os.path.join(REPO, k))
                 or H.sha256_file(os.path.join(REPO, k)) != v)
    info['manifest_check'] = {'n_record_files_checked': len(record_files), 'mismatched': bad}
    checks['manifest_record_hashes_hold'] = not bad and bool(record_files)
    entries, seen_labels, other_multipliers = {}, [], []
    for label, point in sorted((results.get('points') or {}).items()):
        m = point.get('flex_price_multiplier')
        if m != M_FLEX:
            other_multipliers.append({'label': label, 'flex_price_multiplier': m, 'eval_key': point.get('eval_key')})
            continue
        seen_labels.append(label)
        canonical = point.get('candidate_canonical')
        ekey = eval_key_of_canonical(canonical)
        point_checks = {
            'certified': point.get('status') == 'certified',
            'eval_key_recomputes_at_m2': point.get('eval_key') == ekey,
            'eval_key_differs_from_the_baseline_key': ekey != eval_key_of_canonical(canonical, None),
            'candidate_key_recomputes': point.get('candidate_key') == H.candidate_key(canonical),
            'flex_price_label_in_record': point.get('flex_price_label_in_record') == FLEX_LABEL,
            'flex_price_multiplier_in_record': point.get('flex_price_multiplier_in_record') == M_FLEX,
            'Q_is_the_expected_committed_value': point.get('Q') == Q_CACHE_EXPECTED.get(label),
            'bar_present': isinstance(point.get('bar'), float),
            'on_the_lattice': lattice.z_of_canonical(canonical) is not None,
            'flex_readback_all_match': ((point.get('flex_price_readback') or {}).get('all_match') is True),
            'ageing_readback_all_match': ((point.get('ess_ageing_readback_all_match') or {}).get('pre_run') is True
                                          and (point.get('ess_ageing_readback_all_match')
                                               or {}).get('post_run') is True),
        }
        rec_rel = os.path.join(point.get('eval_dir') or '', 'evaluation_record.json')
        present = bool(point.get('eval_dir')) and os.path.isfile(os.path.join(REPO, rec_rel))
        point_checks['evaluation_record_present'] = present
        if present:
            rec = _load(rec_rel)
            point_checks['record_hash_in_campaign_manifest'] = (
                manifest.get(rec_rel) == H.sha256_file(os.path.join(REPO, rec_rel)))
            point_checks['record_Q_matches'] = rec.get('certified_cost') == point.get('Q')
            point_checks['record_bar_matches'] = (rec.get('bar') or {}).get('value') == point.get('bar')
            point_checks['record_cycles_matches'] = rec.get('cycles_run') == point.get('cycles_run')
            point_checks['record_tracked'] = bool(H._git(['ls-files', '--', rec_rel]).strip())
        traj = point.get('per_cycle_trajectory') or {}
        point_checks['trajectory_hash_in_campaign_manifest'] = (
            bool(traj.get('path')) and manifest.get(traj['path']) == traj.get('sha256'))
        for key, value in point_checks.items():
            checks[f'{label}:{key}'] = value
        if ekey in entries:
            reasons.append(f'eval key {ekey[:16]} twice in {src["results"]["path"]}')
        entries[ekey] = {
            'label': label, 'status': point.get('status'), 'Q': point.get('Q'), 'bar': point.get('bar'),
            'canonical': canonical, 'barrier_cause': point.get('barrier_cause'),
            'flex_price_multiplier': m, 'cycles_run': point.get('cycles_run'),
            'certification_cycle': point.get('certification_cycle'),
            'rule_ten': point.get('rule_ten'), 'eval_dir': point.get('eval_dir'),
            'per_cycle_trajectory': traj,
            'source': {'kind': 'committed_certified_record_at_m2', 'campaign': src['name'],
                       'path': src['results']['path'], 'sha256': src['results']['sha256'],
                       'commit': src['results'].get('commit'), 'label': label,
                       'eval_dir': point.get('eval_dir')}}
    checks['m2_labels_are_exactly_the_expected_ones'] = tuple(sorted(seen_labels)) == tuple(
        sorted(src['expected_m2_labels']))
    info.update({'checks': checks, 'n_entries': len(entries), 'm2_labels': sorted(seen_labels),
                 'points_at_other_multipliers_not_cached': other_multipliers})
    failing = sorted(k for k, v in checks.items() if not v)
    if failing or reasons:
        return False, dict(info, reason=f'failing checks {failing}; {reasons}'), {}
    return True, info, entries


def baseline_exclusion(cache_keys, domain_keys, lattice, key_of):
    """Baseline (m = 1) and every other committed record are NOT cache here.

    SCOPE (recorded, never absolute): every `campaign_results.json` under data/SRP1/Results (recursive glob) in
    the working tree. For each, every point that is NOT one of the three pinned m = 2 cache sources must have an
    eval key equal to NO cache key and to NO key of the admissible domain at m = 2. Separately, and
    independently of any file, the m = 2 key of every domain point must differ from its m-absent (baseline) key
    -- that is the mechanism, the scan is the audit of it."""
    cache_paths = {s['results']['path'] for s in CACHE_SOURCES}
    files, collisions, m2_elsewhere = [], [], []
    for path in sorted(glob.glob(os.path.join(REPO, BASELINE_EXCLUSION_GLOB), recursive=True)):
        rel = os.path.relpath(path, REPO)
        try:
            payload = _load(rel)
        except (ValueError, OSError) as error:  # recorded, never silently skipped
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
                      f'{BASELINE_EXCLUSION_GLOB}); the three pinned m = 2 sources are the cache and are skipped '
                      'in the collision test'),
            'n_files': len(files), 'files': files, 'key_collisions': collisions,
            'm2_points_outside_the_pinned_sources': m2_elsewhere,
            'n_domain_points': len(domain_keys), 'n_baseline_keys_of_the_same_domain': len(baseline_keys),
            'baseline_keys_overlapping_domain_or_cache': overlap,
            'm_enters_the_evaluation_key': not overlap,
            'ok': not collisions and not m2_elsewhere and not overlap and bool(files)}


# ======================================================================================================================
#  the budget (this is Phase B)
# ======================================================================================================================
def budget_facts(lattice, cache, domain_labels):
    """I(x) <= B is a CONSTRAINT here. Asserted: B = 1e6 = the W2 master fact; the E = 4 / E = 5 cache points are
    budget-infeasible, excluded from the incumbent, absent from the admissible domain and hence from the frozen
    spec; the E <= 3 points are feasible. ZERO SOLVES (closed form)."""
    w2 = _load(PB.INVESTMENT_COST_RESULTS['path'])
    master = w2['master_facts']
    checks = {'lattice_budget_is_1e6': lattice.budget == 1e6,
              'w2_master_budget_is_the_lattice_budget': master['budget_eur'] == lattice.budget,
              'max_capacity_is_5_mwh': master['max_capacity_mwh'] == PB.E_MAX_MWH}
    rows = {}
    for ekey, entry in cache.items():
        z = lattice.z_of_canonical(entry['canonical'])
        why = lattice.reasons(z) if z is not None else ['off the lattice']
        i_x = lattice.investment_cost(z) if z is not None else None
        label = entry['label']
        rows[label] = {'z': list(z) if z is not None else None, 'lattice_label': lattice.label(z) if z else None,
                       'I_x_eur': i_x, 'slack_eur': (lattice.budget - i_x) if i_x is not None else None,
                       'budget_feasible': not why, 'reasons': why,
                       'in_the_admissible_domain': (spec_label(lattice, z) in domain_labels) if z is not None
                       else False,
                       'eval_key': ekey}
        if label in BUDGET_INFEASIBLE_CACHE_LABELS:
            checks[f'{label}:rejected_by_the_budget'] = bool(why) and any(r.startswith('budget') for r in why)
            checks[f'{label}:not_in_the_admissible_domain'] = not rows[label]['in_the_admissible_domain']
        elif label in BUDGET_FEASIBLE_CACHE_LABELS:
            checks[f'{label}:budget_feasible'] = not why
            checks[f'{label}:in_the_admissible_domain'] = rows[label]['in_the_admissible_domain']
    checks['every_cache_label_is_classified'] = (
        sorted(rows) == sorted(BUDGET_FEASIBLE_CACHE_LABELS + BUDGET_INFEASIBLE_CACHE_LABELS))
    return {'budget_eur': lattice.budget, 'max_capacity_mwh': master['max_capacity_mwh'],
            'rule': ('I(x) <= B = 1,000,000 EUR (STEP4 1.2-1.3) is a CONSTRAINT of Phase B: a violating point is '
                     'rejected by the feasibility test BEFORE any evaluation (extreme barrier, STEP4 2.4 / 5.5) '
                     'and is not in the admissible domain, hence not in the frozen spec at all'),
            'cache_rows': rows, 'checks': checks}


# ======================================================================================================================
#  the initial incumbent
# ======================================================================================================================
def incumbent_matches_committed_ladder(lattice, cache, inc_record):
    """F(0) - F(x) must equal the committed F2 ladder's value - I for every cached rung (to 1e-6 EUR): the same
    ordering, checked rather than argued (INCUMBENT_ORDERING_NOTE)."""
    ladder = _load(F2_LADDER_RESULTS['path'])['ladder']
    by_label = {e['label']: e for e in inc_record['eligible_ranked']}
    f0 = next(e['F'] for e in inc_record['eligible_ranked'] if e['label'] == 'x0')
    rows, checks = {}, {}
    for entry in cache.values():
        z = lattice.z_of_canonical(entry['canonical'])
        if z is None or not lattice.has_storage(z):
            continue
        e_mwh = lattice.nodes_map(z)[7][1]
        row = ladder.get(f'{e_mwh:g}')
        if row is None:
            checks[f'{entry["label"]}:committed_ladder_row_present'] = False
            continue
        f_x = lattice.investment_cost(z) + entry['Q']
        diff = (f0 - f_x) - row['value_minus_I_eur']
        rows[entry['label']] = {'E_mwh': e_mwh, 'F_eur': f_x, 'F0_minus_F_eur': f0 - f_x,
                                'committed_value_minus_I_eur': row['value_minus_I_eur'], 'abs_diff_eur': abs(diff),
                                'committed_resolution_eur': row['resolution_eur'],
                                'committed_budget_feasible': row['budget']['budget_feasible'],
                                'is_committed_budget_corner': row['budget']['is_budget_corner'],
                                'eligible_for_the_incumbent_here': entry['label'] in by_label
                                or lattice.label(z) in by_label}
        checks[f'{entry["label"]}:F0_minus_F_equals_committed_value_minus_I'] = abs(diff) <= LADDER_IDENTITY_TOL_EUR
    checks['incumbent_is_the_committed_budget_corner'] = any(
        r['is_committed_budget_corner'] and abs(r['F_eur'] - inc_record['incumbent']['F']) <= LADDER_IDENTITY_TOL_EUR
        for r in rows.values())
    checks['incumbent_label_is_the_expected_one'] = inc_record['incumbent']['label'] == EXPECTED_INCUMBENT_LABEL
    checks['incumbent_is_not_x0'] = inc_record['is_x0'] is False
    return {'note': INCUMBENT_ORDERING_NOTE, 'source': dict(F2_LADDER_RESULTS),
            'tolerance_eur': LADDER_IDENTITY_TOL_EUR, 'rows': rows,
            'ranked_here': inc_record['eligible_ranked'], 'checks': checks}


# ======================================================================================================================
#  rule eleven -- assert every capture path BEFORE the run
# ======================================================================================================================
def rule_eleven(lattice, sigma_q, cache, key_of, incumbent):
    """PB's record-shape checklist (poll / candidate / terminal fields, the unit-poll completion record), L's
    harness-side checklist (flexibility price applied, read back and recorded; status / cycles / Q / bar / rule
    ten / per-cycle trajectory / wall time / RSS / AA), plus this stage's own: the point-record reader's fields,
    the cache entries, and one SYNTHETIC unit poll from the REAL incumbent with THIS stage's key function."""
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
        'cache_entries_carry_Q_bar_canonical': all(
            isinstance(e.get('Q'), float) and isinstance(e.get('bar'), float) and bool(e.get('canonical'))
            for e in cache.values() if e['status'] == 'certified'),
        'cache_entries_carry_the_multiplier': all(e.get('flex_price_multiplier') == M_FLEX for e in cache.values()),
    })
    # A synthetic unit poll from the real incumbent, with THIS key function and synthetic non-certified records:
    # it exercises the completion record, every candidate field, and the barrier stop rule. No solve, no harness.
    def _no_eval(batch):
        return [{'label': lattice.label(z), 'status': 'not_certified', 'eval_key': key_of(z), 'Q': None, 'bar': None,
                 'canonical': PB.canonical_of(lattice, z), 'barrier_cause': 'synthetic', 'source': 'synthetic'}
                for z in batch]
    out = PB.run_mads(lattice, dict(cache), key_of, dict(incumbent), _no_eval, sigma_q, delta0=PB.DELTA_MIN,
                      max_polls=1, log=lambda m: None)
    poll = out['history'][0]
    n_feasible = len(lattice.neighbourhood(tuple(incumbent['z'])))
    checks.update({
        'synthetic_unit_poll_is_a_unit_poll': poll['unit_poll'] is True,
        'synthetic_unit_poll_completion_recorded': (isinstance(poll['completion'], dict)
                                                    and poll['completion']['n_feasible'] == n_feasible
                                                    and poll['completion']['cap'] == PB.COMPLETION_CAP),
        'synthetic_unit_poll_candidate_fields': all(all(f in c for f in CANDIDATE_RECORD_FIELDS)
                                                    for c in poll['candidates']),
        'synthetic_unit_poll_poll_fields': all(f in poll for f in POLL_RECORD_FIELDS),
        'barrier_stop_rule_is_armed': out['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule',
    })
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s51 F2 Phase B): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': flex_side['harness_record_capture_checklist'],
            'completion_size_at_the_incumbent': n_feasible}


# ======================================================================================================================
#  memory preflight (A0's measure, at this campaign's concurrency 5 -- L's own function)
# ======================================================================================================================
def memory_preflight():
    out = L.memory_preflight()
    out['concurrency_matches_this_campaign'] = out.get('concurrency') == CONCURRENCY
    if not out['concurrency_matches_this_campaign']:
        out['pass'] = False
    return out


# ======================================================================================================================
#  inputs shared by --freeze and --run
# ======================================================================================================================
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
        failures.append(f'solves per cycle from the case file {per_cycle} != 51 (the W10/W32/W33 gates)')
    spec21 = _load(F.SPEC_V21['path'])
    ev['spec_v21_item'] = spec21[F.SPEC_V21['item']]
    ev['spec_v21_phase_B'] = spec21[F.SPEC_V21['item']].get('phase_B')
    ev['predictions_recorded_before_run'] = spec21[F.SPEC_V21['item']].get('predictions_recorded_before_run')

    # ---- the lattice, I(x), sigma_Q, the degradation clause (PB's own functions) ----
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
                    'per_year': {y: sum(1 for z in domain if lattice.has_storage(z) and lattice.years[z[-1]] == y)
                                 for y in years},
                    'rule': ('every budget-, bound- and duration-feasible common-year lattice point, x = 0 first; '
                            'the frozen spec holds exactly these, each at m = 2')}

    # ---- the cache ----
    case_sha = H.sha256_file(H.CASE_FILE)
    sources, accepted, rejected = [], [], []
    for src in CACHE_SOURCES:
        if os.path.dirname(src['results']['path']) == own_root_rel:
            continue
        ok_src, info, entries = load_cache_source(src, case_sha, lattice)
        (accepted if ok_src else rejected).append(info)
        if ok_src:
            sources.append(entries)
    ev['cache_sources_accepted'] = accepted
    ev['cache_sources_rejected'] = rejected
    if rejected or len(accepted) != len(CACHE_SOURCES):
        failures.append(f'cache source(s) refused: {[(r["name"], r.get("reason")) for r in rejected]}')
    cache, duplicates = PB.merge_cache(sources)
    ev['cache_duplicates'] = duplicates
    ev['cache_table'] = {k: {kk: v.get(kk) for kk in ('label', 'status', 'Q', 'bar', 'cycles_run',
                                                      'certification_cycle', 'flex_price_multiplier', 'canonical',
                                                      'source')}
                         for k, v in sorted(cache.items())}
    if len(cache) != len(Q_CACHE_EXPECTED):
        failures.append(f'cache holds {len(cache)} entries, expected {len(Q_CACHE_EXPECTED)}')
    ev['baseline_exclusion'] = baseline_exclusion(set(cache), {key_of(z) for z in domain}, lattice, key_of)
    if not ev['baseline_exclusion']['ok']:
        failures.append(f"baseline / foreign-record exclusion failed: "
                        f"{ev['baseline_exclusion']['key_collisions']} "
                        f"{ev['baseline_exclusion']['m2_points_outside_the_pinned_sources']} "
                        f"{ev['baseline_exclusion']['baseline_keys_overlapping_domain_or_cache']}")
    ev['budget_facts'] = budget_facts(lattice, cache, set(domain_labels))
    failures += [f'budget fact failed: {k}' for k, v in ev['budget_facts']['checks'].items() if not v]

    # ---- the initial incumbent ----
    x0_key = key_of(lattice.x0())
    inc, inc_record = PB.initial_incumbent(lattice, cache, x0_key, sq['sigma_Q_eur'])
    inc_record['ordering_note'] = INCUMBENT_ORDERING_NOTE
    ev['initial_incumbent'] = inc_record
    ev['incumbent_vs_committed_ladder'] = incumbent_matches_committed_ladder(lattice, cache, inc_record)
    failures += [f'incumbent / committed-ladder identity failed: {k}'
                 for k, v in ev['incumbent_vs_committed_ladder']['checks'].items() if not v]
    try:
        ev['rule_eleven'] = rule_eleven(lattice, sq['sigma_Q_eur'], cache, key_of, inc)
    except AssertionError as error:
        ev['rule_eleven'] = {'checks': {}, 'error': str(error)}
        failures.append(str(error))
    return failures, ev, lattice, key_of, domain, domain_labels, cache, inc, x0_key


# ======================================================================================================================
#  the frozen spec
# ======================================================================================================================
def campaign_root(campaign_id):
    return os.path.join(REPO, _P51, f'campaign_{campaign_id}')


def _extra(ev, lattice, domain, dry, memory):
    return {'campaign_script': os.path.basename(__file__),
            'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
            'phase_b_record_script': dict(PHASE_B_SCRIPT,
                                          sha256=H.sha256_file(os.path.join(REPO, PHASE_B_SCRIPT['path']))),
            'f2_ladder_script': dict(F2_LADDER_SCRIPT,
                                     sha256=H.sha256_file(os.path.join(REPO, F2_LADDER_SCRIPT['path']))),
            'ladder_script': dict(M.LADDER_SCRIPT,
                                  sha256=H.sha256_file(os.path.join(REPO, M.LADDER_SCRIPT['path']))),
            'marginal_script': dict(F.MARGINAL_SCRIPT,
                                    sha256=H.sha256_file(os.path.join(REPO, F.MARGINAL_SCRIPT['path']))),
            'label': LABEL, 'flex_label': FLEX_LABEL, 'scenario_label': SCENARIO_LABEL, 'stage': STAGE,
            'flex_price_multiplier': M_FLEX, 'spec_v21': dict(F.SPEC_V21), 'spec_v21_item': ev['spec_v21_item'],
            'spec_v21_phase_B': ev['spec_v21_phase_B'],
            'predictions_recorded_before_run': ev['predictions_recorded_before_run'],
            'pins': ev['pins'], 'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE,
            'stop_rule': STOP_RULE, 'completion_rule': COMPLETION_RULE,
            'termination_certificate': TERMINATION_CERTIFICATE, 'planner_rulings': PLANNER_RULINGS,
            'inherited_ruling_scope': INHERITED_RULING_SCOPE, 'open_items': OPEN_ITEMS,
            'method_changes_referred_to_author': METHOD_CHANGES_REFERRED_TO_AUTHOR,
            'master_problem': {'nodes': list(PB.ACTIVE_NODES), 'years': list(lattice.years),
                               'granule_P_mva': PB.GRANULE_P_MVA, 'granule_E_mwh': PB.GRANULE_E_MWH,
                               'E_max_mwh': PB.E_MAX_MWH, 'duration_h': [2, 4], 'budget_eur': lattice.budget,
                               'form': 'single-cohort, ONE common investment year (n = 7)',
                               'unit_costs_discounted': {str(y): {'power': lattice.c_p[y], 'energy': lattice.c_e[y]}
                                                         for y in lattice.years}},
            'poll_design': {'design': PB.POLL_DESIGN, 'n_vars': PB.N_VARS, 'n_directions': PB.N_VARS + 1,
                            'primes': list(PB.PRIMES[:PB.N_VARS]), 'halton_t0': PB.HALTON_T0,
                            'halton_index': 't = t0 + k (k = poll counter from 0)',
                            'rounding': 'd = round_half_away(Delta h / ||h||_inf)', 'delta_0': PB.DELTA_0,
                            'delta_min': PB.DELTA_MIN, 'mesh_size': PB.MESH_SIZE, 'success': 'Delta *= 2',
                            'failure': 'Delta = max(1, Delta // 2)', 'full_poll': True, 'batch_size': CONCURRENCY,
                            'max_new_evaluations': PB.MAX_NEW_EVALUATIONS, 'max_polls': PB.MAX_POLLS,
                            'unit_poll_completion': PB.UNIT_POLL_COMPLETION, 'completion_cap': PB.COMPLETION_CAP,
                            'completion_size_at_the_incumbent': ev['rule_eleven'].get(
                                'completion_size_at_the_incumbent')},
            'sigma_Q': ev['sigma_Q'], 'degradation_clause': ev['degradation_clause'],
            'I_x_cross_check_vs_W2': ev['I_x_cross_check_vs_W2'], 'domain_summary': ev['domain'],
            'domain_I_x_eur': {spec_label(lattice, z): lattice.investment_cost(z) for z in domain},
            'cache_sources_accepted': ev['cache_sources_accepted'],
            'cache_sources_rejected': ev['cache_sources_rejected'], 'cache_duplicates': ev['cache_duplicates'],
            'cache_table': ev['cache_table'], 'baseline_exclusion': ev['baseline_exclusion'],
            'budget_facts': ev['budget_facts'], 'initial_incumbent': ev['initial_incumbent'],
            'incumbent_vs_committed_ladder': ev['incumbent_vs_committed_ladder'],
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
        'cap_500': spec.get('cap') == CAP,
        'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,
        'ten_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_s39_D': cfg.get('arm_label') == ARM_LABEL,
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'budget_1e6_recorded': (extra.get('master_problem') or {}).get('budget_eur') == 1e6 == lattice.budget,
        'cache_table_frozen': extra.get('cache_table') == json.loads(json.dumps(ev['cache_table'])),
        'cache_sources_frozen': ([(a['name'], a['results']['sha256']) for a in extra.get('cache_sources_accepted') or []]
                                 == [(a['name'], a['results']['sha256']) for a in ev['cache_sources_accepted']]),
        'cache_holds_only_m2': all(e.get('flex_price_multiplier') == M_FLEX
                                   for e in (extra.get('cache_table') or {}).values()),
        'baseline_exclusion_ok': (extra.get('baseline_exclusion') or {}).get('ok') is True,
        'budget_facts_frozen': all((extra.get('budget_facts') or {}).get('checks', {}).values()),
        'initial_incumbent_frozen': (extra.get('initial_incumbent') or {}).get('incumbent') == json.loads(
            json.dumps(ev['initial_incumbent']['incumbent'])),
        'incumbent_is_the_budget_corner': all(
            (extra.get('incumbent_vs_committed_ladder') or {}).get('checks', {}).values()),
        'sigma_Q_frozen': (extra.get('sigma_Q') or {}).get('sigma_Q_eur') == ev['sigma_Q']['sigma_Q_eur'],
        'poll_design_frozen': ((extra.get('poll_design') or {}).get('halton_t0') == PB.HALTON_T0
                               and (extra.get('poll_design') or {}).get('delta_0') == PB.DELTA_0
                               and (extra.get('poll_design') or {}).get('max_new_evaluations')
                               == PB.MAX_NEW_EVALUATIONS),
        'completion_frozen': ((extra.get('poll_design') or {}).get('unit_poll_completion') is True
                              and (extra.get('poll_design') or {}).get('completion_cap') == PB.COMPLETION_CAP
                              and extra.get('completion_rule') == COMPLETION_RULE),
        'planner_rulings_frozen': extra.get('planner_rulings') == PLANNER_RULINGS,
        'inherited_ruling_scope_recorded': extra.get('inherited_ruling_scope') == INHERITED_RULING_SCOPE,
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'resolution_rule_recorded': extra.get('resolution_rule') == RESOLUTION_RULE,
        'termination_certificate_recorded': extra.get('termination_certificate') == TERMINATION_CERTIFICATE,
        'predictions_recorded': extra.get('predictions_recorded_before_run') == ev['predictions_recorded_before_run'],
        'domain_I_x_frozen': all(abs((extra.get('domain_I_x_eur') or {}).get(spec_label(lattice, z), -1.0)
                                     - lattice.investment_cost(z)) <= 1e-6 for z in domain),
        'dry_run_recorded': bool(extra.get('expected_first_poll_dry_run')),
        'memory_rule_recorded': extra.get('memory_preflight_rule') == L.MEMORY_RULE,
        'spec_v21_phase_B_recorded': extra.get('spec_v21_phase_B') == ev['spec_v21_phase_B'],
    }
    return checks


def _guard_verifications():
    return {'parent': PARENT_GUARD.verify(0), 'phase_b_record_parent': PB.PARENT_GUARD.verify(0),
            'f2_ladder_parent': F.PARENT_GUARD.verify(0), 'marginal_parent': M.PARENT_GUARD.verify(0),
            'ladder_parent': L.PARENT_GUARD.verify(0), 'w25_module': L._w25().GUARD.verify(0)}


def _guard_counts():
    return {'parent': dict(PARENT_GUARD.counts), 'phase_b_record_parent': dict(PB.PARENT_GUARD.counts),
            'f2_ladder_parent': dict(F.PARENT_GUARD.counts), 'marginal_parent': dict(M.PARENT_GUARD.counts),
            'ladder_parent': dict(L.PARENT_GUARD.counts), 'w25_module': dict(L._w25().GUARD.counts)}


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(campaign_id, started):
    tag = 'S51-F2-PHASE-B'
    root = campaign_root(campaign_id)
    own_rel = os.path.relpath(root, REPO)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, ev, lattice, key_of, domain, domain_labels, cache, inc, _x0 = build_inputs(own_rel)
    failures += more
    memory = memory_preflight()
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    dry = PB._dry_summary(lattice, cache, key_of, inc, ev['sigma_Q']['sigma_Q_eur'])
    entries = [(spec_label(lattice, z), lattice.nodes_map(z),
                {'investment_year': lattice.year_of(z), 'flex_price_multiplier': M_FLEX}) for z in domain]
    spec_path, spec_sha, _spec = H.freeze_campaign_spec(
        root, campaign_id, entries,
        configuration={'name': (f'F2 PHASE B -- {FLEX_LABEL} (m = {M_FLEX:g}) under {LABEL}: the STEP4 section 5 '
                                'MADS poll under the EUR 1,000,000 budget; the case file (AA keep_memory in '
                                'data/SRP1/SRP1_params.json) with the ESS ageing parameters declared'),
                       'arm_label': ARM_LABEL, 'overrides': {},
                       'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'note': ('candidates = the WHOLE admissible domain (every budget-, bound- and '
                                'duration-feasible common-year lattice point, x = 0 first), each at m = 2, so one '
                                'frozen spec and one campaign lock cover every point the poll may reach; ONLY '
                                'polled, non-cached points are ever evaluated; no overrides; no ageing model '
                                'variant; no post-certification; the flexibility-price multiplier is applied per '
                                'entry in the child configuration hook and read back from probe DSO blocks '
                                "(pre-run) and the run's own DSO models (post-run)")},
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
    for source in ev['cache_sources_accepted']:
        _log(f"[{tag}] cache source {source['name']} ({source['what']}): {source['results']['path']} "
             f"sha256={source['results']['sha256']} commit={source['results'].get('commit')} "
             f"labels={source['m2_labels']}")
    for ekey, entry in ev['cache_table'].items():
        _log(f"[{tag}]   cache {entry['label']}: Q={entry['Q']} bar={entry['bar']} cycles={entry['cycles_run']} "
             f"m={entry['flex_price_multiplier']} eval_key={ekey[:16]}")
    _log(f"[{tag}] baseline exclusion: ok={ev['baseline_exclusion']['ok']} "
         f"files={ev['baseline_exclusion']['n_files']} collisions={ev['baseline_exclusion']['key_collisions']} "
         f"m2_elsewhere={ev['baseline_exclusion']['m2_points_outside_the_pinned_sources']} "
         f"m_enters_the_key={ev['baseline_exclusion']['m_enters_the_evaluation_key']}")
    _log(f"[{tag}] budget: {ev['budget_facts']['rule']}")
    for label, row in sorted(ev['budget_facts']['cache_rows'].items()):
        _log(f"[{tag}]   budget {label}: I={row['I_x_eur']} slack={row['slack_eur']} "
             f"feasible={row['budget_feasible']} reasons={row['reasons']} "
             f"in_domain={row['in_the_admissible_domain']}")
    _log(f"[{tag}] sigma_Q={ev['sigma_Q']['sigma_Q_eur']}; degradation clause triggered="
         f"{ev['degradation_clause']['triggered']} (min unit step {ev['degradation_clause']['min_unit_step_cost_eur']})")
    _log(f"[{tag}] INITIAL INCUMBENT: {ev['initial_incumbent']['incumbent']}")
    _log(f"[{tag}] ranked eligible: {ev['initial_incumbent']['eligible_ranked']}")
    _log(f"[{tag}] excluded from the incumbent: {ev['initial_incumbent']['excluded']}")
    _log(f"[{tag}] margin vs runner-up: {ev['initial_incumbent']['margin_vs_runner_up']}")
    _log(f"[{tag}] margin vs x = 0: {ev['initial_incumbent']['margin_vs_x0']}")
    _log(f"[{tag}] incumbent vs the committed ladder: {ev['incumbent_vs_committed_ladder']['rows']}")
    _log(f'[{tag}] DRY RUN (zero solves) -- the poll sequence from the initial incumbent, followed as far as the '
         f'cache and the closed-form feasibility test allow, i.e. up to the first poll that needs a new evaluation:')
    if dry.get('complete_without_new_evaluations'):
        _log(f"[{tag}]   the whole run completes with NO new evaluation: termination {dry['termination']}; "
             f"certificate {dry['termination_certificate']}")
        for poll in dry['polls']:
            _log(f"[{tag}]   poll {poll['poll_index']}: Delta={poll['Delta']} t={poll['halton_t']} "
                 f"incumbent={poll['incumbent']} completion={poll['completion_n_feasible']} -> {poll['decision']}")
    else:
        for poll in dry['earlier_polls']:
            rejected = sum(1 for c in poll['candidates'] if c['disposition'] == 'rejected_infeasible')
            _log(f"[{tag}]   poll {poll['poll_index']}: Delta={poll['Delta']} t={poll['halton_t']} "
                 f"incumbent={poll['incumbent']} completion={poll['completion_n_feasible']}: "
                 f"{len(poll['candidates'])} candidates, {rejected} rejected by the feasibility test, "
                 f"{len(poll['candidates']) - rejected} feasible -> {poll['decision']}")
        first = dry['first_evaluating_poll']
        _log(f"[{tag}]   poll {first['poll_index']}: Delta={first['Delta']} incumbent={first['incumbent']} "
             f"unit_poll={first['unit_poll']} completion={first['completion_n_feasible']} feasible neighbours "
             f"(cap {PB.COMPLETION_CAP}): {first['n_cache_hits']} cache hits, "
             f"{first['n_new_evaluations']} NEW EVALUATIONS in batches {first['batch_sizes']} at concurrency "
             f"{CONCURRENCY}")
        for entry in first['new_evaluations']:
            _log(f"[{tag}]     NEW {entry['label']} ({entry['poll_part']}) I={entry['I_x_eur']}")
        _log(f"[{tag}]   evaluation budget (ruling A8) = {PB.MAX_NEW_EVALUATIONS}; this poll needs "
             f"{first['n_new_evaluations']}; a later poll whose new evaluations would exceed what remains is NOT "
             f"launched (termination 'evaluation_budget_exhausted', no mesh-local claim)")
    _log(f'[{tag}] DRY RUN (full record, zero solves): {json.dumps(dry, default=str)}')
    _log(f"[{tag}] rule eleven: {ev['rule_eleven']['checks']}")
    _log(f"[{tag}] shared constants: {ev['shared_constants_with_reused_modules']}")
    _log(f"[{tag}] inherited ruling scope: {INHERITED_RULING_SCOPE}")
    _log(f"[{tag}] predictions recorded before the run: {ev['predictions_recorded_before_run']}")
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
def _point_result(label, rec, spec_entry):
    """L._point_result (the flexibility-price record reader) for a polled point. Its W25-throughput branch keys on
    instance == 'unit'; this stage polls points that are not that cell, so the instance is the point's own label
    and `throughput` is None by construction (recorded)."""
    spec_point = {'instance': label, 'flex_price_multiplier': M_FLEX}
    out = L._point_result(label, rec, spec_point, L.solves_per_cycle_from_case_file())
    out['spec_eval_key'] = spec_entry.get('eval_key')
    out['throughput_note'] = ('not computed: the W25 cell-side full-cycle throughput is defined for the node-7 '
                              'unit cell of the ladder stages, not for an arbitrary polled point')
    return out


def _cache_entry_from_point(point, ekey):
    certified = point.get('status') == 'certified'
    return {'label': point.get('label'), 'status': point.get('status'),
            'eval_key': point.get('eval_key') or ekey, 'Q': point.get('Q') if certified else None,
            'bar': point.get('bar'), 'canonical': point.get('candidate_canonical'),
            'barrier_cause': point.get('barrier_cause'), 'flex_price_multiplier': point.get('flex_price_multiplier'),
            'cycles_run': point.get('cycles_run'), 'certification_cycle': point.get('certification_cycle'),
            'rule_ten': point.get('rule_ten'), 'eval_dir': point.get('eval_dir'),
            'per_cycle_trajectory': point.get('per_cycle_trajectory'),
            'source': {'kind': 'phase_b_new_evaluation_at_m2', 'eval_dir': point.get('eval_dir')}}


def _log_poll(record):
    tag = 'S51-F2-PHASE-B'
    inc = record['incumbent']
    _log(f"[{tag}] POLL {record['poll_index']}: Delta={record['poll_size_delta']} (mesh {record['mesh_size']}), "
         f"halton t={record['halton_t']}, incumbent={inc['label']} F={inc['F']} I={inc['I']} Q={inc['Q']} "
         f"bar={inc['bar']}, unit_poll={record['unit_poll']}, new={record['n_new_evaluations']}, "
         f"cache hits={record['n_cache_hits']}")
    if record.get('completion'):
        comp = record['completion']
        _log(f"[{tag}]   completion: {comp['n_feasible']} feasible neighbours (cap {comp['cap']}, over_cap="
             f"{comp['over_cap']}); rejected raw offsets by class {comp['rejected_raw_offsets_by_class']}; "
             f"budget-rejected {comp['budget_rejected']}")
    for cand in record['candidates']:
        _log(f"[{tag}]   [{cand['poll_part']}{'' if cand['direction_index'] is None else ' ' + str(cand['direction_index'])}] "
             f"d={cand['direction']} label={cand['label']} feasible={cand['feasible']} "
             f"reasons={cand['infeasibility_reasons']} I={cand['I_x_eur']} disposition={cand['disposition']} "
             f"status={cand['status']} Q={cand['Q_eur']} F={cand['F_eur']} "
             f"F_inc-F={cand['F_inc_minus_F_eur']} bar_sum={cand['bar_sum_eur']} sigma_Q={cand['sigma_Q_eur']} "
             f"resolution={cand['resolution_eur']} -> {cand['outcome']}")
    _log(f"[{tag}]   decision={record['decision']} next_incumbent={record['next_incumbent']} "
         f"next_poll_size={record['next_poll_size']}")


def run(campaign_id, started, spec_sha256):
    tag = 'S51-F2-PHASE-B'
    root = campaign_root(campaign_id)
    own_rel = os.path.relpath(root, REPO)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    root_contents = sorted(os.listdir(root))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, ev, lattice, key_of, domain, domain_labels, cache, inc, _x0 = build_inputs(own_rel)
    failures += more
    checks = validate_spec(spec, lattice, key_of, domain, domain_labels, ev)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, frozen, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE)),
                              ('ESS params file', spec['configuration']['ess_params_file']['sha256'],
                               H.sha256_file(os.path.join(REPO, H.ESS_PARAMS_FILE_REL))),
                              ('this script', (spec['extra'] or {}).get('campaign_script_sha256'),
                               H.sha256_file(os.path.abspath(__file__))),
                              ('the Phase B record module',
                               ((spec['extra'] or {}).get('phase_b_record_script') or {}).get('sha256'),
                               H.sha256_file(os.path.join(REPO, PHASE_B_SCRIPT['path']))),
                              ('the F2 ladder launcher',
                               ((spec['extra'] or {}).get('f2_ladder_script') or {}).get('sha256'),
                               H.sha256_file(os.path.join(REPO, F2_LADDER_SCRIPT['path'])))):
        if frozen != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    memory = memory_preflight()
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
    state_path = os.path.join(root, 'phase_b_state.json')
    _log(f'[{tag}] {SCENARIO_LABEL}')
    _log(f'[{tag}] {STAGE}')
    _log(f'[{tag}] objective convention: {OBJECTIVE_CONVENTION}')
    _log(f'[{tag}] resolution rule: {RESOLUTION_RULE}')
    _log(f"[{tag}] initial incumbent: {ev['initial_incumbent']['incumbent']}")
    _log(f"[{tag}] sigma_Q={sigma_q}; inherited scope: {INHERITED_RULING_SCOPE}")
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
            point = _point_result(label, by_label.get(label), entry_of_key[ekey])
            new_points[label] = point
            _log(f"[{tag}]   {label}: status={point['status']} cycles={point.get('cycles_run')} Q={point.get('Q')} "
                 f"bar={point.get('bar')} m_readback="
                 f"{(point.get('flex_price_readback') or {}).get('all_match')} "
                 f"rule_ten_gross={(point.get('rule_ten') or {}).get('terminal_gross_step_over_threshold')} "
                 f"solves={(point.get('solve_reconciliation') or {}).get('observed')}/"
                 f"{(point.get('solve_reconciliation') or {}).get('expected')}")
            out.append(_cache_entry_from_point(point, ekey))
        return out

    def on_poll(record):
        history_so_far.append(record)
        _log_poll(record)
        H._atomic_write_json(state_path, {'spec_sha256': spec_sha256, 'campaign_id': campaign_id,
                                          'utc': datetime.now(timezone.utc).isoformat(),
                                          'flex_price_multiplier': M_FLEX, 'history': history_so_far,
                                          'new_points': new_points})

    try:
        result = PB.run_mads(lattice, cache, key_of, inc, evaluate_fn, sigma_q, on_poll=on_poll,
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
    results = {
        'LABEL': LABEL, 'FLEX_LABEL': FLEX_LABEL, 'SCENARIO_LABEL': SCENARIO_LABEL, 'stage': STAGE,
        'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'campaign_id': campaign_id, 'flex_price_multiplier': M_FLEX,
        'objective_convention': OBJECTIVE_CONVENTION, 'resolution_rule': RESOLUTION_RULE, 'stop_rule': STOP_RULE,
        'completion_rule': COMPLETION_RULE, 'planner_rulings': PLANNER_RULINGS,
        'inherited_ruling_scope': INHERITED_RULING_SCOPE, 'open_items': OPEN_ITEMS,
        'method_changes_referred_to_author': METHOD_CHANGES_REFERRED_TO_AUTHOR,
        'STOP_FOR_REVIEW': result['termination']['reason'].startswith('STOP_FOR_REVIEW'),
        'termination': result['termination'], 'termination_certificate': result['termination_certificate'],
        'final_incumbent': result['incumbent'], 'initial_incumbent': ev['initial_incumbent'],
        'incumbent_vs_committed_ladder': ev['incumbent_vs_committed_ladder'],
        'sigma_Q': ev['sigma_Q'], 'budget_facts': ev['budget_facts'], 'baseline_exclusion': ev['baseline_exclusion'],
        'n_polls': result['n_polls'], 'n_new_evaluations': result['n_new_evaluations'],
        'n_barrier_new_evaluations': result['n_barrier_new_evaluations'],
        'final_poll_unresolved_indeterminate': result['final_poll_unresolved_indeterminate'],
        'final_poll_feasible_points': result['final_poll_feasible_points'],
        'lattice_neighbourhood_of_incumbent': result['lattice_neighbourhood_of_incumbent'],
        'claim_scope': ('on termination by the unit-poll failure the claim is the termination_certificate (ruling '
                        'A2: the final unit poll = rounded directions UNION the completion of every feasible '
                        '||dz||_inf <= 1 lattice neighbour; certificate.holds must be true); indeterminate '
                        'neighbours are unresolved, not improvements; on any other termination no mesh-local '
                        'claim is made. Every figure is a figure of the m = 2 FLEXIBILITY-PRICE SCENARIO under '
                        'the EUR 1M budget.'),
        'poll_history': result['history'],
        'cache_table_at_start': ev['cache_table'], 'cache_sources_accepted': ev['cache_sources_accepted'],
        'points': new_points,
        'non_certified_points': non_certified, 'harness_errors': harness_errors,
        'readback_mismatch_points': readback_mismatch, 'solve_reconciliation_mismatch_points': solve_mismatch,
        'batch_info': batch_infos, 'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: ev[k] for k in ('pins', 'case_file_aa', 'cache_sources_accepted',
                                                'baseline_exclusion', 'I_x_cross_check_vs_W2',
                                                'degradation_clause', 'shared_constants_with_reused_modules',
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
    _log(f"[{tag}] polls {result['n_polls']}; new evaluations {result['n_new_evaluations']}; "
         f"barrier evaluations {result['n_barrier_new_evaluations']}")
    _log(f"[{tag}] unresolved indeterminate at the final poll: {result['final_poll_unresolved_indeterminate']}")
    if non_certified:
        _log(f'[{tag}] non-certified points (reported with cause): '
             f'{[(l, new_points[l].get("barrier_cause")) for l in non_certified]}')
    if readback_mismatch:
        _log(f'[{tag}] READ-BACK MISMATCH: {readback_mismatch}')
    if solve_mismatch:
        _log(f'[{tag}] SOLVE RECONCILIATION MISMATCH (reported): {solve_mismatch}')
    _log(f'[{tag}] guards: {_guard_counts()}; verify0 failures={guards}')
    reason = result['termination']['reason']
    if any(guards.values()) or harness_errors or readback_mismatch:
        _log(f'[{tag}] NOT OK guards={guards} harness_errors={harness_errors} readback={readback_mismatch}')
        sys.exit(1)
    if reason == 'mesh_local_optimum_unit_poll_failed' and not (result['termination_certificate'] or {}).get('holds'):
        _log(f"[{tag}] NOT OK: the termination certificate does not hold {result['termination_certificate']}")
        sys.exit(1)
    if reason.startswith('STOP_FOR_REVIEW'):
        _banner([f'STOP_FOR_REVIEW: {reason}', json.dumps(result['termination'], default=str)])
        sys.exit(3)
    if reason != 'mesh_local_optimum_unit_poll_failed':
        _log(f'[{tag}] terminated without a mesh-local claim: {reason}')
        sys.exit(2)
    _log(f'[{tag}] OK: terminated by the unit-poll failure; the certificate holds')


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
