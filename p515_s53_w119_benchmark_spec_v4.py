"""P5.15 Addendum 57 Decision 1, Planner task W119 -- BENCHMARK SPEC v4 and its ZERO-SOLVE CHECKS. NO BENCHMARK STAGE RUNS
HERE (no sweep, no NRF arm, no tie-breaker, no report).

A `SolveProfileGuard(permitted=())` is armed BEFORE any production import; importing the v3 script
(`p515_s53_w116_benchmark_spec_v3`, whose checks are re-run here as they apply to v4) arms W116's, W111's and W106's
zero-solve guards on top of it; W114's module (imported by V5) installs a permitting guard at import, which is
uninstalled at once. Every guard is verified at 0 at the end.

WHAT v4 IS: v3 47b37d6e (predecessor, not edited) with exactly the two changes the Planner ruled (W119), and everything
else identical:
  Change 1  per-block EXACT solve accounting for the NRF arm stages and the NRF tie-breaker stages, the scheme W116 built
            for the sweep (`p515_s53_w116_benchmark_nrf.DeclaredBlockAccount`): each phase declares its blocks, every
            executed solve is attributed to a declared block and attempt tier (primary / recovery / recovery_tier2), the
            per-stage total is the sum of the attributed counts; an unattributed or excess solve, or a count short of the
            declared blocks, still raises. Reason: W113's price_taker/perturbed retried DSO|9|2035|Spring 3 times; under
            v3's phase-exact rule that retry alone would fail the stage.
  Change 2  the Planner predictions, recorded before any run, labelled "Planner prediction (W119)" (key
            predictions_recorded_before_any_run_v4); W116's "Worker expectation" entries kept unchanged (v3 key carried).
  Also recorded as Planner rulings (key planner_rulings_w119): the consistency re-evaluation convention (confirmed);
  W116's declared_choices_v3 (confirmed); the reverse-flow count result of the v3 checks with the Addendum 57 caveat.
OUTPUT ROOT: v3's root data/SRP1/Results/P515S53/w116_benchmark_nrf/ with v4-named files -- write-once safe because no
stage ran under v3 (no run directory and no stage log exists under the root; the freeze refuses otherwise and V9 checks
it); the stage harness binds the HIGHEST version under the root and refuses version < 4.

MODE --freeze-spec: writes <root>/frozen_s53_benchmark_spec_v4_<hash8>.json once (refuses unless the highest existing
  benchmark spec version anywhere under data/ is exactly 3, v3 has its committed sha256, the bound files, this script and
  the v3 script are clean in git, and no stage output or stage log exists under the root).
MODE --checks (default): write-once under <root>/w119_zero_solve_checks<suffix>/.
  V0  preconditions snapshot (v3's V0)
  V1  the settled models' hash (v3's V1)
  V2  the NRF rows present / absent (v3's V2, unchanged code path)
  V3  the TSO arm unchanged against v2 (v3's V3)
  V4  the reverse-flow count on the Q181 models recomputed (v3's V4, into the new checks directory) AND equal to the
      committed W116 file the report reads (count block identical)
  V5  the sweep's solve accounting (v3's V5; the sweep is unchanged)
  V6  W100's repository-wide boolean-typing test
  V7  committed eval keys unchanged over ALL campaign specs in the HEAD tree (git ls-tree, not the index snapshot W116's
      V7 used), the ten W118 r2 specs asserted present; v3's V7 recorded beside it
  V8  the frozen spec v4 binds; NEGATIVE: altered pin, name-hash mismatch, v3 and v2 specs refused, v4 selected over v3
  V9  v3 -> v4 key diff: only the declared keys change; every other v3 key identical; the binding differs only in the
      harness pin; stage commands identical; the shared root write-once safe
  V10 no production file changed (v3's V10)
  V11 the report's capture paths (v3's V11, against the committed reverse-flow file the report reads)
  V12 the stage wiring (v3's V12, on the v4 spec)
  V13 Change 1: DeclaredBlockAccount -- NEGATIVE CONTROLS with stubs (a retried block accepted; an unattributed solve
      raises; a count short of the declared total raises; and the other raise paths), W113's committed price_taker/
      perturbed records replayed (the 3-attempt DSO|9|2035|Spring block accepted), attempt tiers read from production's
      source, the declared blocks from the planning, stage_nrf_arm wiring
  V14 Change 2 and the rulings: predictions and rulings recorded as ruled, W116's Worker expectations unchanged, the
      reverse-flow values equal the committed file, the report carries the Addendum 57 caveat and scores the Planner
      predictions (exercised zero-solve on synthetic outcomes)

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w119_benchmark_spec_v4.py --freeze-spec > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w119_freeze_spec_v4.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w119_benchmark_spec_v4.py --checks > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w119_zero_solve_checks.log 2>&1
Exit 0 = done / all checks pass; 1 = a check failed; 2 = refused.
"""

import argparse
import copy
import fnmatch
import gc
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W119 zero-solve').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib at import
import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- stdlib + H + GRIO at import
import p515_s53_w116_benchmark_nrf as NRF  # noqa: E402 -- stdlib + H + GRIO + BENCH at import; no guard
import p515_s53_w116_benchmark_spec_v3 as V3  # noqa: E402 -- arms W116's, W111's and W106's zero-solve guards at import

W111 = V3.W111
W106 = V3.W106
PY = V3.PY
THIS = os.path.basename(__file__)
V3_SCRIPT = V3.THIS
OUT_ROOT_REL = NRF.OUT_ROOT_REL
CHECKS_DIR_REL = os.path.join(OUT_ROOT_REL, 'w119_zero_solve_checks')
LAUNCH_LOGS_REL = NRF.LAUNCH_LOGS_REL
HARNESS = NRF.SCRIPT_NAME
PLANNER_LABEL = 'Planner prediction (W119)'
RULING_LABEL = 'Planner ruling (W119)'
V3_SPEC = {'path': os.path.join(OUT_ROOT_REL, 'frozen_s53_benchmark_spec_v3_47b37d6e.json'),
           'sha256': '47b37d6eadb9d360f2600b6e1b961c93e255dcd6f5cdd7bafb391c15942bbe7a', 'version': 3,
           'committed_in': '79b945f9c12e3da7280a5752c6bbb0a05defd71a',
           'git_head_at_freeze': 'a43ebe24ba6972bfedf15f4b215c0f534e926c51'}
# the committed W116 reverse-flow count the report reads (NRF.REVERSE_FLOW_OUTPUT_REL); sha from w116 manifests
W116_REVERSE_FLOW = {'path': NRF.REVERSE_FLOW_OUTPUT_REL,
                     'sha256': '8706bf286c38bd469aea1e9704686f8de1f74e41db9fb24b985100849de27ce2',
                     'committed_in': '79b945f9c12e3da7280a5752c6bbb0a05defd71a'}
# W113's spec-v2 price_taker/perturbed run: the 3-attempt DSO|9|2035|Spring retry (the reason for Change 1); committed in
# 209f4829, sha256 as in that run's manifest_sha256.json
W113_PT_PERTURBED = {
    'per_solve_record': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'arm_price_taker_perturbed', 'per_solve_record.jsonl'),
                         'sha256': '1c0aa95b812df619b7b465b99d53f3a6c443a8fcc0af682cef9286d8cb58c4ee'},
    'failure': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'arm_price_taker_perturbed', 'failure.json'),
                'sha256': '338d12c141540e07feb1314a9d544f0ed40c1e052bb6f4f6a49064991a5440bf'},
    'committed_in': '209f4829', 'retried_block': 'DSO|9|2035|Spring', 'retried_attempts': 3}
INFORMATIONAL_FILES = V3.INFORMATIONAL_FILES + (THIS,)
# V9: the top-level keys v4 may differ from v3 in (every other v3 key is carried identically and asserted)
V4_CHANGED_OR_ADDED_TOP_KEYS = (
    'version', 'predecessor', 'predecessor_note', 'stage', 'authority', 'frozen_utc', 'git_head',
    'code_sha256_binding', 'code_sha256_informational', 'stages_v3_addendum_57_order', 'solve_counts',
    'zero_solve_checks', 'solve_accounting_v4', 'predictions_recorded_before_any_run_v4', 'planner_rulings_w119')
# within stages_v3_addendum_57_order: the only fields a v4 stage entry may differ in (NRF arm / tie-breaker stages only)
V4_STAGE_FIELDS_CHANGED = ('declared_solves', 'declared_solves_detail', 'guard')
V4_STAGES_CHANGED = ('nrf-arm', 'nrf-passive-tie-breaker')
# the Planner's W119 predictions and the certain blocks behind their lower bounds (W113 cold records)
SWEEP_PREDICTION = {'passive': {'n_range': [1, 4], 'certain_block': 'TSO|-|2035|Summer'},
                    'price_taker': {'n_range': [2, 8], 'certain_block': 'TSO|-|2025|Spring'}}
ADDENDUM_57_CAVEAT = ('if the coordinated solution has any reverse-flow interface-hour, part of the measured benefit is '
                      'the value of allowing reverse flow, and the paper says so (Addendum 57 Decision 1(b))')
REPORT_CAVEAT_TEXT = 'part of the measured benefit is the value of allowing reverse flow'
_T0 = time.time()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W119 +{time.time() - _T0:8.1f}s] {msg}', flush=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _stage_outputs_present():
    """Every v3/v4 stage output directory or stage log that already exists under the shared root (must be none)."""
    run_ids = NRF.SWEEP_RUN_IDS + NRF.NRF_ARM_RUN_IDS + NRF.NRF_VARIANT_RUN_IDS + [NRF.REPORT_RUN_ID]
    present = [os.path.join(OUT_ROOT_REL, r) for r in run_ids if os.path.exists(_abs(os.path.join(OUT_ROOT_REL, r)))]
    present += [os.path.join(LAUNCH_LOGS_REL, f'{r}.log') for r in run_ids
                if os.path.exists(_abs(os.path.join(LAUNCH_LOGS_REL, f'{r}.log')))]
    return present


# ======================================================================================================================
#  the frozen spec v4
# ======================================================================================================================
def _stages_v4(v3_stages):
    d = NRF.SRP1_DECLARED_V3
    m = NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE
    arm_blocks = d['nrf_arm_solves'] + d['nrf_reevaluation_solves']
    arm_blocks_triggered = arm_blocks + d['nrf_sequential_pass_solves']
    stages = copy.deepcopy(v3_stages)
    rule = ('exact per declared block (p515_s53_w116_benchmark_nrf.DeclaredBlockAccount): every record attributed to '
            'one declared block of the open phase and its attempts to the tiers primary / recovery / recovery_tier2 '
            'in order (1..3); guard delta == attributed attempts per block; cumulative verified exactly after every '
            'block; a phase closes only with every declared block settled once and the guard == the sum of the '
            'attributed attempts -- a production retry is that block\'s retries (W119), not a stage failure')
    for st in stages:
        if st['stage'] == 'nrf-arm':
            st['declared_solves'] = {'blocks_exact': arm_blocks, 'blocks_if_triggered': arm_blocks_triggered,
                                     'launch_range': [arm_blocks, m * arm_blocks],
                                     'launch_range_if_triggered': [arm_blocks_triggered, m * arm_blocks_triggered],
                                     'launches': 'the sum of the attempts attributed per block (1..3 each)',
                                     'accounting': 'exact per declared block; see solve_accounting_v4'}
            st['declared_solves_detail'] = {'phase_A_nrf_arm_blocks': d['nrf_arm_solves'],
                                            'phase_B_consistency_reevaluation_blocks': d['nrf_reevaluation_solves'],
                                            'phase_C_sequential_pass_blocks_only_if_triggered':
                                                d['nrf_sequential_pass_solves'],
                                            'total_blocks_if_triggered': arm_blocks_triggered}
            st['guard'] = (f'SolveProfileGuard(permitted={NRF.PERMITTED_ARM_SITES}); verify(0) before solves; phases '
                           f'A ({d["nrf_arm_solves"]} blocks), B ({d["nrf_reevaluation_solves"]}), C '
                           f'({d["nrf_sequential_pass_solves"]}, only if the declared trigger fires): ' + rule)
        elif st['stage'] == 'nrf-passive-tie-breaker':
            st['declared_solves'] = {'blocks_exact': d['nrf_variant_solves'],
                                     'launch_range': [d['nrf_variant_solves'], m * d['nrf_variant_solves']],
                                     'launches': 'the sum of the attempts attributed per block (1..3 each)',
                                     'accounting': 'exact per declared block; see solve_accounting_v4'}
            st['guard'] = (f'SolveProfileGuard(permitted={NRF.PERMITTED_ARM_SITES}); verify(0) before solves; one '
                           f'phase A ({d["nrf_variant_solves"]} blocks): ' + rule)
    return stages


def _solve_counts_v4(v3_counts):
    d = NRF.SRP1_DECLARED_V3
    m = NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE
    arm_blocks = d['nrf_arm_solves'] + d['nrf_reevaluation_solves']
    arm_triggered = arm_blocks + d['nrf_sequential_pass_solves']
    variant = d['nrf_variant_solves']
    counts = {k: copy.deepcopy(v) for k, v in v3_counts.items()
              if k in ('sweep_per_arm', 'nrf_arm_runs', 'nrf_variant_runs', 'report')}
    counts.update({
        'nrf_arm_per_run_blocks_exact': arm_blocks, 'nrf_arm_per_run_blocks_if_triggered': arm_triggered,
        'nrf_arm_per_run_launch_range': [arm_blocks, m * arm_blocks],
        'nrf_arm_per_run_launch_range_if_triggered': [arm_triggered, m * arm_triggered],
        'nrf_variant_per_run_blocks_exact': variant, 'nrf_variant_per_run_launch_range': [variant, m * variant],
        'total_nrf_blocks_exact': 6 * arm_blocks + 2 * variant,
        'total_nrf_blocks_if_every_arm_triggers': 6 * arm_triggered + 2 * variant,
        'total_upper_bound_all_stages': (2 * d['sweep_upper_bound_per_arm'] + 6 * m * arm_triggered + 2 * m * variant),
        'per_stage_rule': ('NRF arms / variants (W119, spec v4): exact per declared block, each block 1..3 attributed '
                           'attempts, the stage total = the sum of the attributed attempts (see solve_accounting_v4); '
                           'sweep: exact per block + upper bound (unchanged from v3); report: 0'),
        'replaces_v3_keys': {'nrf_arm_per_run_exact': v3_counts.get('nrf_arm_per_run_exact'),
                             'nrf_arm_per_run_if_triggered': v3_counts.get('nrf_arm_per_run_if_triggered'),
                             'nrf_variant_per_run_exact': v3_counts.get('nrf_variant_per_run_exact'),
                             'total_nrf_exact': v3_counts.get('total_nrf_exact'),
                             'total_nrf_if_every_arm_triggers': v3_counts.get('total_nrf_if_every_arm_triggers'),
                             'total_upper_bound_all_stages': v3_counts.get('total_upper_bound_all_stages'),
                             'per_stage_rule': v3_counts.get('per_stage_rule'),
                             'note': 'v3 values (launch counts exact at every phase boundary); now counts of BLOCKS'},
    })
    return counts


def solve_accounting_v4():
    d = NRF.SRP1_DECLARED_V3
    return {
        'ruling': ('Planner task W119 Change 1: per-block exact solve accounting for the NRF arm stages and the NRF '
                   'tie-breaker stages, the scheme W116 built for the sweep; exact: every executed solve attributed to '
                   'a declared block and attempt tier, the per-stage total equal to the sum of the attributed counts; '
                   'an unattributed or excess solve still raises'),
        'reason': ('W113 (spec v2) price_taker/perturbed retried DSO|9|2035|Spring 3 times (primary, recovery, '
                   'recovery_tier2; it succeeded at tier 2); under v3\'s phase-exact rule that retry alone would fail '
                   'the stage'),
        'reason_evidence': W113_PT_PERTURBED,
        'implemented_in': ('p515_s53_w116_benchmark_nrf.py: DeclaredBlockAccount (a BlockSolveAccount), '
                           'declared_block_networks, ATTEMPT_TIERS; used by stage_nrf_arm (stages nrf-arm and '
                           'nrf-passive-tie-breaker); failure.json carries the ledger (solve_accounting)'),
        'phases': {
            'A_nrf_arm': {'blocks': d['nrf_arm_solves'], 'labels': '36 DSO (sorted DN, year, day) + 12 TSO',
                          'record_phases': ['<arm>:<start>:dso', '<arm>:<start>:tso']},
            'B_consistency_reevaluation': {'blocks': d['nrf_reevaluation_solves'], 'labels': '36 DSO',
                                           'record_phases': ['consistency:reevaluation'], 'nrf_arm_only': True},
            'C_sequential_pass': {'blocks': d['nrf_sequential_pass_solves'], 'labels': '36 DSO + 12 TSO',
                                  'record_phases': ['sequential_pass:dso', 'sequential_pass:tso'],
                                  'only_if_triggered': True}},
        'attempt_tiers': list(NRF.ATTEMPT_TIERS),
        'attempt_tiers_source': ('network._append_ipopt_solve_record: attempt = log_suffix or \'primary\'; '
                                 'network._run_smopf: log_suffix \'recovery\' then \'recovery_tier2\' (V13 reads both)'),
        'max_attempts_per_block': NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE,
        'per_block_rule': ('the record is attributed to a declared, not-yet-settled block of the open phase with a '
                           'declared record phase; its attempts are ATTEMPT_TIERS[:n], 1 <= n <= 3, each with the '
                           'block\'s network name, year and day; the guard delta (solves and launches) since the '
                           'previous record == n; the cumulative is verified exactly (SolveProfileGuard.verify) after '
                           'every block; cumulative <= the stage upper bound (3 x declared blocks)'),
        'per_phase_rule': ('a phase opens only with the guard exactly at the attributed cumulative and closes only '
                           'with every declared block settled exactly once and the guard == the sum of the '
                           'attributed attempts'),
        'raises_on': ['a record outside an open phase', 'a record for an undeclared block (unattributed)',
                      'a block settled twice', 'a record phase not declared for the phase',
                      'a guard delta above the attempts (an unattributed solve before the record)',
                      'a guard delta below the attempts', 'attempts above three or tiers out of order',
                      'an attempt of another network, year or day',
                      'a phase closed short of its declared blocks',
                      'a solve after the last record of a phase (guard above the attributed sum at close)',
                      'a cumulative above the upper bound'],
        'negative_controls': ('V13, zero solves with stubs: a retried block accepted; an unattributed solve raises; a '
                              'count short of the declared total raises; every other raise path; W113\'s committed '
                              'records replayed'),
        'sweep': 'unchanged: BlockSolveAccount, sweep_addendum_57.solve_accounting',
        'unit_note': ('declared counts are BLOCKS; launches per stage lie in [blocks, 3 x blocks] and equal the sum of '
                      'the attributed attempts'),
    }


def predictions_v4(v3):
    carried = v3['predictions_recorded_before_any_run_v3']['nrf_arms_feasible_at_every_tso_block']
    return {
        'label': PLANNER_LABEL, 'recorded_before_any_run': True,
        'source': 'Planner task W119 (TASKS.md 93f59b04: W119 dispatched)',
        'nrf_claim': {
            'label': PLANNER_LABEL, 'sign': 'positive',
            'statement': ('the coordinated Q181 is below min(passive_NRF, price_taker_NRF): benefit = '
                          'min(Q_passive_NRF, Q_price_taker_NRF) - Q181 > 0'),
            'magnitude': 'not predicted',
            'scored_by': ('report_v3.predictions_scored.planner_predictions_w119.nrf_claim: held iff claim.benefit_eur '
                          '> 0; the verdict and determinacy stated beside it')},
        'sweep_n_of_12_unconstrained_arms_cold': {
            'label': PLANNER_LABEL,
            'passive': {**SWEEP_PREDICTION['passive'], 'of': 12,
                        'statement': 'passive n in [1, 4] of 12 blocks (>= 1 certain: 2035 Summer)'},
            'price_taker': {**SWEEP_PREDICTION['price_taker'], 'of': 12,
                            'statement': 'price-taker n in [2, 8] of 12 blocks (>= 1 certain: 2025 Spring)'},
            'confidence': 'low on the counts',
            'certain_blocks_source': ('W113 cold records (spec v2): the first failing TSO block, passive 2035 Summer, '
                                      'price-taker 2025 Spring, each after 3 attempts'),
            'scored_by': ('report_v3.predictions_scored.planner_predictions_w119.sweep_n_of_12_unconstrained_arms_cold: '
                          'held iff n in the closed range and the certain block among the failing blocks')},
        'nrf_arms_feasible_at_every_tso_block': {
            'label': PLANNER_LABEL, 'prediction': carried['planner_prediction'],
            'carried_from': 'v3 predictions_recorded_before_any_run_v3.nrf_arms_feasible_at_every_tso_block.'
                            'planner_prediction (unchanged)',
            'scored_by': carried['scored_by']},
        'worker_expectations': ('W116\'s "Worker expectation" entries are kept as they are, in '
                                'predictions_recorded_before_any_run_v3 (carried from v3 unchanged; V9 and V14 assert '
                                'it)'),
        'supersedes_note': ('the planner_prediction: None entries of predictions_recorded_before_any_run_v3 '
                            '(nrf_claim_sign, sweep_n_of_12) are W116\'s record that no Planner prediction existed then; '
                            'the Planner predictions are these'),
    }


def planner_rulings_w119(v3, reverse):
    totals = reverse['count']['totals']
    entries = reverse['count']['reverse_entries_most_negative_first']
    choices = v3['declared_choices_v3']['items']
    return {
        'label': RULING_LABEL,
        'consistency_reevaluation': {
            'label': RULING_LABEL, 'status': 'confirmed',
            'ruling': ('the NRF rows are off (deactivated) in the re-evaluation clone and are checked as hard DN '
                       'limits; a violation above 1e-6 p.u. (hard_tol) triggers the one sequential pass'),
            'hard_tol_pu': BENCH.CONSISTENCY_TOL['hard_tol_pu2'],
            'refers_to': ['consistency_convention_v3', 'no_reverse_flow_addendum_57.consistency_reevaluation']},
        'declared_choices_v3': {
            'label': RULING_LABEL, 'status': 'confirmed',
            'confirmed': {'cold_only_sweep': choices[0],
                          'strict_reverse_flow_primary_with_material_count_beside_it': choices[1],
                          'sweep_threshold_1e-6_pu': choices[2],
                          'nrf_rows_hard_limits_in_reevaluation': choices[3]},
            'superseded': {'item': choices[4], 'by': 'solve_accounting_v4 (W119 Change 1)'},
            'note': 'declared_choices_v3 carried from v3 unchanged (its status text is W116\'s, before this ruling)'},
        'reverse_flow_count_v3_checks': {
            'label': RULING_LABEL,
            'result': {'interface_hours_strict': totals['strict']['count'],
                       'interface_hours_material': totals['material']['count'],
                       'nodes': sorted({e['node_id'] for e in entries}), 'years': sorted({e['year'] for e in entries}),
                       'energy_mwh_over_horizon_day_weighted': totals['strict']['energy_mwh_day_weighted'],
                       'energy_mwh_over_horizon_stated': '3,308.30 MWh',
                       'energy_mwh_per_representative_day_sum': totals['strict']['energy_mwh_rep_day'],
                       'entries_mw': [{'node_id': e['node_id'], 'year': e['year'], 'day': e['day'],
                                       'hour': e['hour'], 'p_int_mw': e['p_int_mw']} for e in entries]},
            'source': W116_REVERSE_FLOW,
            'report_must_carry': ADDENDUM_57_CAVEAT,
            'report_statement_text': REPORT_CAVEAT_TEXT,
            'report_implementation': ('stage_report: claim.reverse_flow_caveat and coordinated_reverse_flow_count.'
                                      'statement, both set when the strict count > 0 (V14 checks the source)')},
    }


def build_spec_v4(v3, reverse):
    v4 = copy.deepcopy(v3)
    v4.update({
        'version': 4,
        'predecessor': {'path': V3_SPEC['path'], 'sha256': V3_SPEC['sha256'], 'version': 3,
                        'committed_in': V3_SPEC['committed_in'], 'git_head_at_freeze': V3_SPEC['git_head_at_freeze']},
        'predecessor_note': (
            'v4 = v3 with the two changes the Planner ruled in W119: (1) per-block exact solve accounting for the NRF '
            'arm and tie-breaker stages (solve_accounting_v4; the stage entries\' declared_solves / guard and '
            'solve_counts restated in blocks); (2) the Planner predictions (predictions_recorded_before_any_run_v4); '
            'plus the Planner rulings recorded (planner_rulings_w119). Everything else identical to v3 (V9 asserts '
            'it). Same output root as v3 with v4-named files: write-once safe, no stage ran under v3 (no run '
            'directory and no stage log exists under the root at this freeze). v3 is not edited'),
        'stage': NRF.STAGE, 'authority': NRF.AUTHORITY, 'frozen_utc': _utc(), 'git_head': _git(['rev-parse', 'HEAD']),
        'code_sha256_binding': {name: _sha(name) for name in NRF.FROZEN_SPEC_BOUND_FILES},
        'code_sha256_informational': {name: W106._informational_pin(name) for name in INFORMATIONAL_FILES},
        'stages_v3_addendum_57_order': _stages_v4(v3['stages_v3_addendum_57_order']),
        'solve_counts': _solve_counts_v4(v3['solve_counts']),
        'zero_solve_checks': {'command': (f'set -o noclobber && {PY} -u {THIS} --checks > '
                                          f'{LAUNCH_LOGS_REL}/w119_zero_solve_checks.log 2>&1'),
                              'output': CHECKS_DIR_REL,
                              'note': ('run AFTER this freeze; V8 verifies this spec binds; V9 diffs it against v3; '
                                       'V4 recomputes the reverse-flow count; V13 the per-block accounting controls'),
                              'v3_checks': v3['zero_solve_checks']},
        'solve_accounting_v4': solve_accounting_v4(),
        'predictions_recorded_before_any_run_v4': predictions_v4(v3),
        'planner_rulings_w119': planner_rulings_w119(v3, reverse),
    })
    return v4


def freeze_spec():
    dirty = _git(['status', '--porcelain', '--'] + list(NRF.FROZEN_SPEC_BOUND_FILES) + [THIS, V3_SCRIPT])
    if dirty:
        print(f'REFUSING to freeze: bound files not clean in git:\n{dirty}', flush=True)
        return 2
    highest = W106._highest_existing_version()
    if highest != 3:
        print(f'REFUSING to freeze: highest existing benchmark spec version is {highest}, expected exactly 3', flush=True)
        return 2
    present = _stage_outputs_present()
    if present:
        print(f'REFUSING to freeze: stage outputs exist under the shared root: {present}', flush=True)
        return 2
    v3 = W111._load_verified_json(V3_SPEC)
    reverse = W111._load_verified_json(W116_REVERSE_FLOW)
    v4 = build_spec_v4(v3, reverse)
    text = GRIO.dumps(v4, indent=1, sort_keys=True) + '\n'
    data = text.encode()
    sha = hashlib.sha256(data).hexdigest()
    path = _abs(os.path.join(OUT_ROOT_REL, f'frozen_s53_benchmark_spec_v4_{sha[:8]}.json'))
    with open(path, 'xb') as handle:
        handle.write(data)
    if H.sha256_file(path) != sha:
        raise RuntimeError('frozen spec sha mismatch after write')
    diff = W111.key_diff(v3, json.loads(text))
    failures = _all_guard_failures()
    _log(f'frozen spec v4: {os.path.relpath(path, REPO)} sha256 {sha}; predecessor v3 {V3_SPEC["sha256"]}; git_head '
         f'{v4["git_head"]}; solve counts {v4["solve_counts"]}; guard verify(0) {failures}')
    top = sorted({x['path'].split('.')[0].split('[')[0] for x in diff})
    _log(f'v3 -> v4 key diff: {len(diff)} paths; top-level keys changed / added / removed: {top}')
    for x in diff:
        _log(f"  {x['kind']:8s} {x['path']}")
    return 0 if not any(failures.values()) else 1


def _all_guard_failures():
    return {'w119': _GUARD.verify(0), 'w116_import': V3._GUARD.verify(0), 'w111_import': W111._GUARD.verify(0),
            'w106_import': W106._GUARD.verify(0)}


# ======================================================================================================================
#  checks
# ======================================================================================================================
def _renamed(res, new_id):
    res = dict(res)
    res['id_v3'] = res.get('id')
    res['id'] = new_id
    return res


def _latest_spec():
    latest = NRF._latest_frozen_spec()
    with open(latest[0]) as handle:
        return latest, json.load(handle)


def v4_reverse_flow_count(UB, srp, planning, certified, out_dir):
    res = V3.v4_reverse_flow_count(UB, srp, planning, certified, out_dir)
    committed = W111._load_verified_json(W116_REVERSE_FLOW)
    with open(_abs(res['output'])) as handle:
        recomputed = json.load(handle)
    equal = recomputed['count'] == committed['count']
    res = _renamed(res, 'V4_reverse_flow_count_q181')
    res['committed_w116_file'] = W116_REVERSE_FLOW
    res['count_equals_committed_w116_file'] = equal
    res['passed'] = bool(res['passed'] and equal)
    return res


class _FakeGuard(V3._FakeGuard):
    pass


def _declared_account_unit_test(UB):
    """Zero solves, stubs: the fake guard stands for SolveProfileGuard, synthetic records for production's."""
    nets = {'DSO|5|2025|Spring': ('case33_1', '2025', 'Spring'), 'DSO|5|2025|Summer': ('case33_1', '2025', 'Summer'),
            'TSO|-|2025|Spring': ('case9', '2025', 'Spring')}
    labels = list(nets)
    saved = NRF._GUARD
    results = {}

    def fresh(bound=100):
        g = _FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, bound, nets)
        acc.open_phase('A', labels, ('arm:dso', 'arm:tso'))
        return g, acc

    def rec(label, n, tiers=None, phase=None, network=None):
        name, year, day = nets.get(label, ('elsewhere', '2025', 'Spring'))
        tiers = list(NRF.ATTEMPT_TIERS[:n]) if tiers is None else tiers
        kind = label.split('|')[0]
        return {'block': label, 'kind': kind, 'phase': phase or f'arm:{kind.lower()}', 'n_attempts': n,
                'succeeded': True, 'attempts': [{'attempt': t, 'network': network or name, 'year': int(year),
                                                 'day': day} for t in tiers]}

    def solve(g, acc, label, n, launched=None, **kw):
        g.launch(n if launched is None else launched)
        return acc.settle_record(rec(label, n, **kw))

    def expect_raise(fn, contains=None):
        try:
            fn()
            return False
        except RuntimeError as error:
            return contains is None or contains in str(error)

    try:
        # a retried block is ACCEPTED: three blocks, the middle one at primary, recovery, recovery_tier2
        g, acc = fresh()
        persisted = []
        sink = acc.recorder(persisted.append)
        for label, n in ((labels[0], 1), (labels[1], 3), (labels[2], 1)):
            g.launch(n)
            sink(rec(label, n))
        closed = acc.close_phase('A')
        results['retried_block_accepted'] = (closed['attempts_attributed'] == 5 and acc.cumulative == 5
                                             and closed['retried_blocks'] == {labels[1]: 3}
                                             and closed['n_blocks_settled'] == 3 and len(persisted) == 3
                                             and acc.ledger[1]['attempt_tiers'] == list(NRF.ATTEMPT_TIERS)
                                             and g.verify(5) == [])
        results['summary_totals_close'] = (acc.summary()['attempts_attributed_total'] == acc.cumulative == 5
                                           and acc.summary()['open_phase'] is None)
        # the reason for Change 1: the same retry under v3's phase-exact rule (verify(3 blocks)) fails
        results['retry_fails_v3_phase_exact_rule'] = g.verify(3) != []
        # NEGATIVE: an unattributed solve BEFORE a record (guard delta 2, record 1 attempt)
        g, acc = fresh()
        results['NEGATIVE_unattributed_solve_before_record_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 1, launched=2), 'guard delta')
        # NEGATIVE: an unattributed solve AFTER the last record of a phase, caught at close
        g, acc = fresh()
        for label in labels:
            solve(g, acc, label, 1)
        g.launch(1)
        results['NEGATIVE_unattributed_solve_after_last_record_raises_at_close'] = expect_raise(
            lambda: acc.close_phase('A'), 'SolveProfileGuard')
        # NEGATIVE: a record for an undeclared block
        g, acc = fresh()
        results['NEGATIVE_record_for_undeclared_block_raises'] = expect_raise(
            lambda: solve(g, acc, 'DSO|7|2025|Spring', 1), 'UNATTRIBUTED')
        # NEGATIVE: a record outside an open phase
        g, acc = fresh()
        for label in labels:
            solve(g, acc, label, 1)
        acc.close_phase('A')
        results['NEGATIVE_record_outside_open_phase_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 1), 'outside any declared phase')
        # NEGATIVE: a count SHORT of the declared total (2 of 3 blocks), caught at close
        g, acc = fresh()
        solve(g, acc, labels[0], 1)
        solve(g, acc, labels[1], 3)
        results['NEGATIVE_count_short_of_declared_total_raises'] = expect_raise(lambda: acc.close_phase('A'), 'SHORT')
        # NEGATIVE: a block solved twice
        g, acc = fresh()
        solve(g, acc, labels[0], 1)
        results['NEGATIVE_block_solved_twice_raises'] = expect_raise(lambda: solve(g, acc, labels[0], 1), 'twice')
        # NEGATIVE: an excess attempt (4 > 3), with its tiers
        g, acc = fresh()
        results['NEGATIVE_attempts_above_three_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 4, tiers=list(NRF.ATTEMPT_TIERS) + ['recovery_tier3']))
        # NEGATIVE: tiers out of order
        g, acc = fresh()
        results['NEGATIVE_tiers_out_of_order_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 2, tiers=['primary', 'recovery_tier2']), 'tiers')
        # NEGATIVE: the guard below the attempts (a recorded attempt that did not run through the guard)
        g, acc = fresh()
        results['NEGATIVE_guard_below_attempts_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 2, launched=1), 'guard delta')
        # NEGATIVE: an attempt of another network
        g, acc = fresh()
        results['NEGATIVE_foreign_network_attempt_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 1, network='case33_2'), 'another network')
        # NEGATIVE: a record phase not declared for the phase
        g, acc = fresh()
        results['NEGATIVE_undeclared_record_phase_raises'] = expect_raise(
            lambda: solve(g, acc, labels[0], 1, phase='sequential_pass:dso'), 'record phase')
        # NEGATIVE: over the upper bound
        g, acc = fresh(bound=2)
        results['NEGATIVE_over_upper_bound_raises'] = expect_raise(lambda: solve(g, acc, labels[0], 3), 'upper bound')
        # NEGATIVE: a launch-count (exec) mismatch
        g, acc = fresh()

        def exec_mismatch():
            g.launch(1, exec_n=2)
            acc.settle_record(rec(labels[0], 1))
        results['NEGATIVE_exec_count_mismatch_raises'] = expect_raise(exec_mismatch, 'guard delta')
        # NEGATIVE: a phase opened while the guard holds an unattributed solve
        g = _FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 100, nets)
        g.launch(1)
        results['NEGATIVE_open_phase_with_unattributed_solve_raises'] = expect_raise(
            lambda: acc.open_phase('A', labels, ('arm:dso', 'arm:tso')), 'SolveProfileGuard')
    finally:
        NRF._GUARD = saved
    return results


def _replay_w113(block_networks):
    """W113's committed price_taker/perturbed records (spec v2), replayed zero-solve through the v4 account with a fake
    guard advanced by each record's attempts: (a) the 36 DSO records as their own phase -- the 3-attempt
    DSO|9|2035|Spring is accepted and the phase closes exactly; (b) the whole record under the 48-block phase A -- every
    record settles (the failing TSO block's 3 attempts too) and the phase is SHORT (the stage failed at that TSO block:
    a genuine failure, not the accounting)."""
    path = BENCH._verified_path(W113_PT_PERTURBED['per_solve_record'])
    with open(path) as handle:
        records = [json.loads(line) for line in handle]
    saved = NRF._GUARD
    out = {'source': W113_PT_PERTURBED['per_solve_record'], 'n_records': len(records)}
    dso_labels = [label for label in block_networks if label.startswith('DSO|')]
    tso_labels = [label for label in block_networks if label.startswith('TSO|')]
    try:
        g = _FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 3 * len(block_networks), block_networks)
        acc.open_phase('A_dso_only', dso_labels, ('price_taker:perturbed:dso',))
        for r in records:
            if r['kind'] == 'DSO':
                g.launch(r['n_attempts'])
                acc.settle_record(r)
        closed = acc.close_phase('A_dso_only')
        out['dso_phase'] = {'closed': closed, 'guard_counts': dict(g.counts)}
        out['dso_retry_accepted'] = (closed['retried_blocks'] == {W113_PT_PERTURBED['retried_block']:
                                                                  W113_PT_PERTURBED['retried_attempts']}
                                     and closed['n_blocks_settled'] == len(dso_labels) == 36
                                     and closed['attempts_attributed'] == 38 == g.counts['permitted_solve'])
        g = _FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 3 * len(block_networks), block_networks)
        acc.open_phase('A_nrf_arm', dso_labels + tso_labels, ('price_taker:perturbed:dso', 'price_taker:perturbed:tso'))
        settled_all = True
        for r in records:
            g.launch(r['n_attempts'])
            try:
                acc.settle_record(r)
            except RuntimeError as error:
                settled_all = False
                out['unexpected_settle_error'] = str(error)
                break
        try:
            acc.close_phase('A_nrf_arm')
            short = False
        except RuntimeError as error:
            short = 'SHORT' in str(error)
            out['close_error'] = str(error)[:300]
        out['full_record'] = {'every_record_settled': settled_all, 'cumulative': acc.cumulative,
                              'guard_counts': dict(g.counts), 'phase_short_as_expected': short,
                              'failing_record': next((r['block'] for r in records if not r['succeeded']), None)}
        out['passed'] = bool(out['dso_retry_accepted'] and settled_all and short and acc.cumulative == 41
                             and out['full_record']['failing_record'] == 'TSO|-|2025|Spring')
    finally:
        NRF._GUARD = saved
    return out


def v13_nrf_block_accounting(UB, planning):
    import network as NW
    block_networks = NRF.declared_block_networks(UB, planning)
    declared = UB.declared_solve_count(planning)
    dso = [label for label in block_networks if label.startswith('DSO|')]
    tso = [label for label in block_networks if label.startswith('TSO|')]
    names = {}
    for label, (name, year, day) in block_networks.items():
        names.setdefault((name, year, day), []).append(label)
    blocks = {'n_dso': len(dso), 'n_tso': len(tso), 'equals_declared_solve_count': (
        len(dso) == declared['dso'] and len(tso) == declared['tso'] and len(block_networks) == declared['total']),
        'network_year_day_unique_per_block': all(len(v) == 1 for v in names.values()),
        'example': {k: list(v) for k, v in list(block_networks.items())[:2]}}
    append_src = inspect.getsource(NW._append_ipopt_solve_record)
    smopf_src = inspect.getsource(NW._run_smopf)
    tiers = {'primary_is_default_suffix': "'attempt': log_suffix or 'primary'" in append_src,
             'recovery_suffix': "log_suffix='recovery')" in smopf_src,
             'tier2_suffix': "log_suffix='recovery_tier2')" in smopf_src,
             'recovery_before_tier2': smopf_src.find("log_suffix='recovery')") < smopf_src.find(
                 "log_suffix='recovery_tier2')"),
             'three_attempt_sites': smopf_src.count('_run_smopf_solver_attempt(') == NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE,
             'harness_tiers': list(NRF.ATTEMPT_TIERS) == ['primary', 'recovery', 'recovery_tier2']}
    src = inspect.getsource(NRF.stage_nrf_arm)
    wiring = {
        'phase_A_declared_48': "account.open_phase('A_nrf_arm', dso_labels + tso_labels" in src,
        'phase_A_closed': "phases.append(account.close_phase('A_nrf_arm'))" in src,
        'phase_B_declared_36': "account.open_phase('B_consistency_reevaluation', dso_labels," in src,
        'phase_B_closed': "phases.append(account.close_phase('B_consistency_reevaluation'))" in src,
        'phase_C_declared_48': "account.open_phase('C_sequential_pass', dso_labels + tso_labels," in src,
        'phase_C_closed': "phases.append(account.close_phase('C_sequential_pass'))" in src,
        'records_go_through_the_account': 'sink = account.recorder(solve_sink)' in src,
        'every_production_solve_call_gets_the_sink': src.count('record_callback=sink') == 4,
        'v3_phase_exact_rule_removed': '_check_guard(expected' not in src and 'expected +=' not in src,
        'final_guard_equals_attributed': "_check_guard(account.cumulative, f'{run_id} end')" in src,
        'accounting_in_result': "result['solve_accounting'] = accounting" in src,
        'failure_json_carries_ledger': "'solve_accounting': _ACCOUNT.summary()" in inspect.getsource(NRF.main),
        'tie_breaker_uses_stage_nrf_arm': ("stage_nrf_arm(run_dir, run_id, arm='passive', start='cold'"
                                           in inspect.getsource(NRF.main)),
        'sweep_unchanged_uses_block_solve_account': 'account = BlockSolveAccount(_GUARD, upper)' in inspect.getsource(
            NRF.stage_sweep)}
    unit = _declared_account_unit_test(UB)
    replay = _replay_w113(block_networks)
    passed = (all(blocks[k] for k in ('equals_declared_solve_count', 'network_year_day_unique_per_block'))
              and all(tiers.values()) and all(wiring.values()) and all(unit.values()) and replay['passed'])
    return {'id': 'V13_nrf_per_block_accounting', 'passed': bool(passed), 'declared_blocks': blocks,
            'attempt_tiers_from_production_source': tiers, 'stage_nrf_arm_wiring': wiring,
            'negative_controls_with_stubs': unit, 'w113_replay': replay,
            'account_doc': inspect.getdoc(NRF.DeclaredBlockAccount)}


def v7_eval_keys_head_tree(w106_c6):
    """Committed eval keys unchanged over ALL campaign specs in the HEAD tree (git ls-tree), under evaluation_key's full
    current signature, by the harness at HEAD and in the working tree; the ten W118 r2 specs asserted present. v3's V7
    (index snapshot, W111 informational) recorded beside it."""
    pattern = 'data/*campaign_spec_*.json'
    in_head = sorted(p for p in _git(['ls-tree', '-r', '--name-only', 'HEAD']).splitlines()
                     if fnmatch.fnmatch(p, pattern))
    in_index = sorted(p for p in _git(['ls-files', pattern]).splitlines() if p.strip())
    r2 = [p for p in in_head if '/campaign_s53_w118_resettle_r2_' in p]
    head_module, head_sha = W106._harness_from_git('HEAD')
    at_head = V3._recompute_keys_signature(head_module, in_head)
    in_tree = V3._recompute_keys_signature(H, in_head)
    try:
        v3_v7 = V3.v7_eval_keys(w106_c6)
        v3_v7_summary = {'passed': v3_v7.get('passed'), 'committed_specs': v3_v7.get('committed_specs'),
                         'entries_at_head': (v3_v7.get('harness_at_head') or {}).get('entries'),
                         'w111_c6_informational': v3_v7.get('w111_c6_informational')}
    except Exception as error:  # noqa: BLE001
        v3_v7_summary = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
    disk = _sha('p515_s44_campaign_harness.py')
    passed = (at_head['holds'] and in_tree['holds'] and len(r2) == 10 and set(in_index) == set(in_head)
              and v3_v7_summary.get('passed') is True)
    return {'id': 'V7_committed_eval_keys_unchanged_head_tree', 'passed': bool(passed),
            'git_head': _git(['rev-parse', 'HEAD']), 'committed_specs_head_tree': len(in_head),
            'committed_specs_index': len(in_index), 'index_equals_head_tree': set(in_index) == set(in_head),
            'w118_r2_specs_present': r2, 'n_w118_r2_specs': len(r2),
            'evaluation_key_signature': str(inspect.signature(H.evaluation_key)),
            'harness_at_head': {'sha256': head_sha, **at_head},
            'harness_working_tree': {'sha256': disk, 'differs_from_head': disk != head_sha, **in_tree},
            'v3_v7_as_written': v3_v7_summary, 'w119_modifies_harness': False}


def v8_spec_binding():
    failures = NRF.frozen_spec_binding_failures()
    latest = NRF._latest_frozen_spec()
    negatives = {}
    if latest is not None:
        path, version, _hash8 = latest
        tmp = tempfile.mkdtemp(prefix='w119_spec_neg_')
        try:
            spec = json.load(open(path))
            spec['code_sha256_binding'][HARNESS] = '0' * 64
            data = (GRIO.dumps(spec, indent=1, sort_keys=True) + '\n').encode()
            sha = hashlib.sha256(data).hexdigest()
            name = os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_{sha[:8]}.json')
            with open(name, 'wb') as handle:
                handle.write(data)
            negatives['altered_pin_refused'] = any(HARNESS in f for f in NRF.frozen_spec_binding_failures(tmp))
            os.remove(name)
            bad = os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_00000000.json')
            shutil.copyfile(path, bad)
            negatives['name_hash_mismatch_refused'] = any('does not start with its name hash' in f
                                                          for f in NRF.frozen_spec_binding_failures(tmp))
            os.remove(bad)
            shutil.copyfile(_abs(V3_SPEC['path']), os.path.join(tmp, os.path.basename(V3_SPEC['path'])))
            v3_fail = NRF.frozen_spec_binding_failures(tmp)
            negatives['v3_spec_refused_version_and_harness_pin'] = (any('version 3 < 4' in f for f in v3_fail)
                                                                    and any(HARNESS in f for f in v3_fail))
            shutil.copyfile(path, os.path.join(tmp, os.path.basename(path)))
            both = NRF._latest_frozen_spec(tmp)
            negatives['v4_selected_over_v3_in_shared_root'] = (both is not None and both[1] == 4
                                                               and NRF.frozen_spec_binding_failures(tmp) == [])
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            shutil.copyfile(_abs(V3.V2['path']), os.path.join(tmp, os.path.basename(V3.V2['path'])))
            v2_fail = NRF.frozen_spec_binding_failures(tmp)
            negatives['v2_spec_refused_version_and_root'] = (any('version 2 < 4' in f for f in v2_fail)
                                                             and any('output_root' in f for f in v2_fail))
        finally:
            shutil.rmtree(tmp)
    passed = (latest is not None and latest[1] == 4 and not failures and bool(negatives) and all(negatives.values())
              and NRF.FROZEN_SPEC_MIN_VERSION == 4)
    return {'id': 'V8_frozen_spec_v4_binds', 'passed': bool(passed),
            'spec': None if latest is None else {'path': os.path.relpath(latest[0], REPO), 'version': latest[1],
                                                 'sha256': H.sha256_file(latest[0])},
            'frozen_spec_min_version': NRF.FROZEN_SPEC_MIN_VERSION, 'binding_failures': failures,
            'NEGATIVE': negatives}


def v9_spec_diff():
    v3 = W111._load_verified_json(V3_SPEC)
    latest, v4 = _latest_spec()
    if latest[1] != 4:
        return {'id': 'V9_v3_v4_key_diff', 'passed': False, 'reason': f'latest spec under the root is {latest}'}
    diff = W111.key_diff(v3, v4)
    top = sorted({x['path'].split('.')[0].split('[')[0] for x in diff})
    pins3, pins4 = v3['code_sha256_binding'], v4['code_sha256_binding']
    stage_rows = []
    stages_ok = len(v3['stages_v3_addendum_57_order']) == len(v4['stages_v3_addendum_57_order'])
    for s3, s4 in zip(v3['stages_v3_addendum_57_order'], v4['stages_v3_addendum_57_order']):
        differs = sorted(k for k in set(s3) | set(s4) if s3.get(k) != s4.get(k))
        allowed = V4_STAGE_FIELDS_CHANGED if s3['stage'] in V4_STAGES_CHANGED else ()
        ok = set(differs) <= set(allowed) and (s3['stage'] not in V4_STAGES_CHANGED or 'guard' in differs)
        stages_ok = stages_ok and ok
        stage_rows.append({'run_id': s3['run_id'], 'stage': s3['stage'], 'fields_differing': differs, 'ok': ok})
    checks = {
        'v3_sha256_unchanged': _sha(V3_SPEC['path']) == V3_SPEC['sha256'],
        'v3_file_clean_in_git': _git(['status', '--porcelain', '--', V3_SPEC['path']]) == '',
        'v2_sha256_unchanged': _sha(V3.V2['path']) == V3.V2['sha256'],
        'v4_predecessor_is_v3': (v4.get('predecessor') or {}).get('sha256') == V3_SPEC['sha256'],
        'changed_top_keys_within_declared_set': set(top) <= set(V4_CHANGED_OR_ADDED_TOP_KEYS),
        'every_other_v3_key_identical': all(v3[k] == v4.get(k) for k in v3 if k not in V4_CHANGED_OR_ADDED_TOP_KEYS),
        'no_v3_key_removed': all(k in v4 for k in v3),
        'binding_differs_only_in_harness_pin': (sorted(pins3) == sorted(pins4)
                                                and sorted(k for k in pins3 if pins3[k] != pins4[k]) == [HARNESS]),
        'stage_entries_differ_only_in_declared_fields': stages_ok,
        'stage_commands_and_run_ids_identical': ([(s['run_id'], s['command']) for s in v3['stages_v3_addendum_57_order']]
                                                 == [(s['run_id'], s['command'])
                                                     for s in v4['stages_v3_addendum_57_order']]),
        'sweep_solve_counts_identical': v3['solve_counts']['sweep_per_arm'] == v4['solve_counts']['sweep_per_arm'],
        'worker_expectations_carried_identical': (v3['predictions_recorded_before_any_run_v3']
                                                  == v4['predictions_recorded_before_any_run_v3']),
        'output_root_same': v4.get('output_root') == v3.get('output_root') == OUT_ROOT_REL,
        'shared_root_write_once_safe_no_stage_output': _stage_outputs_present() == [],
    }
    return {'id': 'V9_v3_v4_key_diff', 'passed': all(checks.values()), 'checks': checks,
            'v3': dict(V3_SPEC), 'v4': {'path': os.path.relpath(latest[0], REPO), 'sha256': H.sha256_file(latest[0])},
            'n_paths': len(diff), 'changed_top_keys': top, 'stage_entries': stage_rows,
            'top_level': [x for x in diff if '.' not in x['path'] and '[' not in x['path']], 'diff': diff}


def _synthetic_scoring(spec):
    """NRF._score_predictions on synthetic outcomes (zero solves): all held / all failed / nothing scoreable."""
    def sweep(n, failing):
        return {'sweep': {'n_blocks_tn_cannot_accept': n, 'n_hours_tn_cannot_accept': 6 * n, 'failing_blocks': failing}}
    per_arm = {'passive': {}, 'price_taker': {}}
    held = NRF._score_predictions(
        spec, per_arm, {'computed': True, 'benefit_eur': 1.0e6, 'verdict': 'x', 'determinate': True},
        {'passive': sweep(1, ['TSO|-|2035|Summer']),
         'price_taker': sweep(3, ['TSO|-|2025|Spring', 'TSO|-|2030|Spring', 'TSO|-|2035|Spring'])}, [])
    failed = NRF._score_predictions(
        spec, per_arm, {'computed': True, 'benefit_eur': -5.0, 'verdict': 'x', 'determinate': False},
        {'passive': sweep(5, ['TSO|-|2025|Spring'] * 5), 'price_taker': sweep(1, ['TSO|-|2025|Spring'])},
        ['nrf_arm_passive_cold'])
    missing = NRF._score_predictions(spec, {}, {'computed': False}, {'passive': None, 'price_taker': None}, [])

    def outcomes(scored):
        p = scored.get('planner_predictions_w119') or {}
        return {'nrf_claim': (p.get('nrf_claim') or {}).get('outcome'),
                'sweep_passive': ((p.get('sweep_n_of_12_unconstrained_arms_cold') or {}).get('passive') or {}).get(
                    'outcome'),
                'sweep_price_taker': ((p.get('sweep_n_of_12_unconstrained_arms_cold') or {}).get('price_taker')
                                      or {}).get('outcome'),
                'feasible': (p.get('nrf_arms_feasible_at_every_tso_block') or {}).get('outcome')}
    got = {'held': outcomes(held), 'failed': outcomes(failed), 'missing': outcomes(missing)}
    ok = (set(got['held'].values()) == {'held'} and set(got['failed'].values()) == {'failed'}
          and set(got['missing'].values()) == {'not scoreable'})
    return got, ok


def v14_spec_v4_content():
    v3 = W111._load_verified_json(V3_SPEC)
    latest, spec = _latest_spec()
    reverse = W111._load_verified_json(W116_REVERSE_FLOW)
    p = spec.get('predictions_recorded_before_any_run_v4') or {}
    r = spec.get('planner_rulings_w119') or {}
    totals = reverse['count']['totals']
    entries = reverse['count']['reverse_entries_most_negative_first']
    w113_fail = {arm: NRF._load_verified_json(NRF.W113_COLD[arm]['failure'])['record']['block']
                 for arm in ('passive', 'price_taker')}
    report_src = inspect.getsource(NRF.stage_report)
    score_src = inspect.getsource(NRF._score_predictions)
    scoring, scoring_ok = _synthetic_scoring(spec)
    sweep = p.get('sweep_n_of_12_unconstrained_arms_cold') or {}
    rf = (r.get('reverse_flow_count_v3_checks') or {}).get('result') or {}
    checks = {
        'predictions_labelled': (p.get('label') == PLANNER_LABEL and all(
            (p.get(k) or {}).get('label') == PLANNER_LABEL for k in (
                'nrf_claim', 'sweep_n_of_12_unconstrained_arms_cold', 'nrf_arms_feasible_at_every_tso_block'))),
        'predictions_recorded_before_any_run': p.get('recorded_before_any_run') is True and _stage_outputs_present() == [],
        'nrf_claim_positive_magnitude_not_predicted': ((p.get('nrf_claim') or {}).get('sign') == 'positive'
                                                       and p['nrf_claim'].get('magnitude') == 'not predicted'),
        'sweep_passive_1_4_certain_2035_summer': (sweep.get('passive') or {}).get('n_range') == [1, 4] and (
            sweep['passive'].get('certain_block') == 'TSO|-|2035|Summer'),
        'sweep_price_taker_2_8_certain_2025_spring': (sweep.get('price_taker') or {}).get('n_range') == [2, 8] and (
            sweep['price_taker'].get('certain_block') == 'TSO|-|2025|Spring'),
        'sweep_low_confidence': sweep.get('confidence') == 'low on the counts',
        'certain_blocks_are_w113_cold_failing_blocks': (w113_fail == {'passive': 'TSO|-|2035|Summer',
                                                                      'price_taker': 'TSO|-|2025|Spring'}),
        'feasibility_carries_v3_statement': ((p.get('nrf_arms_feasible_at_every_tso_block') or {}).get('prediction')
                                             == v3['predictions_recorded_before_any_run_v3'][
                                                 'nrf_arms_feasible_at_every_tso_block']['planner_prediction']),
        'worker_expectations_unchanged': (spec['predictions_recorded_before_any_run_v3']
                                          == v3['predictions_recorded_before_any_run_v3']),
        'ruling_consistency_confirmed': ((r.get('consistency_reevaluation') or {}).get('status') == 'confirmed'
                                         and r['consistency_reevaluation'].get('hard_tol_pu') == 1e-6),
        'ruling_declared_choices_confirmed': ((r.get('declared_choices_v3') or {}).get('status') == 'confirmed'
                                              and len(r['declared_choices_v3'].get('confirmed') or {}) == 4),
        'declared_choices_v3_carried_unchanged': spec['declared_choices_v3'] == v3['declared_choices_v3'],
        'reverse_flow_4_interface_hours': (rf.get('interface_hours_strict') == 4 == totals['strict']['count']
                                           and rf.get('interface_hours_material') == 4),
        'reverse_flow_node_7_year_2025': (rf.get('nodes') == [7] and rf.get('years') == ['2025']
                                          and {e['node_id'] for e in entries} == {7}
                                          and {e['year'] for e in entries} == {'2025'} and len(entries) == 4),
        'reverse_flow_3308_30_mwh_over_horizon': (round(totals['strict']['energy_mwh_day_weighted'], 2) == 3308.30
                                                  and rf.get('energy_mwh_over_horizon_day_weighted')
                                                  == totals['strict']['energy_mwh_day_weighted']),
        'reverse_flow_file_is_the_one_the_report_reads': W116_REVERSE_FLOW['path'] == NRF.REVERSE_FLOW_OUTPUT_REL,
        'report_carries_addendum_57_caveat': (REPORT_CAVEAT_TEXT in report_src
                                              and "reverse['count']['totals']['strict']['count'] > 0" in report_src
                                              and "claim['reverse_flow_caveat']" in report_src),
        'report_scores_planner_predictions': "'predictions_recorded_before_any_run_v4'" in score_src,
        'scoring_exercised_synthetic': scoring_ok,
        'solve_accounting_v4_recorded': (spec.get('solve_accounting_v4') or {}).get('attempt_tiers')
        == list(NRF.ATTEMPT_TIERS),
    }
    return {'id': 'V14_spec_v4_predictions_and_rulings', 'passed': all(checks.values()), 'checks': checks,
            'spec': {'path': os.path.relpath(latest[0], REPO), 'sha256': H.sha256_file(latest[0])},
            'synthetic_scoring_outcomes': scoring, 'w113_cold_failing_blocks': w113_fail,
            'reverse_flow_totals_committed': totals}


def run_checks(suffix=''):
    out_dir = _abs(CHECKS_DIR_REL + suffix)
    if os.path.exists(out_dir):
        print(f'REFUSING: output exists (write-once): {out_dir}', flush=True)
        return 2
    os.makedirs(out_dir)
    results, timings, extra = [], {}, {}

    def record(fn, *args, new_id=None):
        t = time.time()
        try:
            res = fn(*args)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res = {'id': new_id or getattr(fn, '__name__', 'check'), 'passed': False,
                   'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        payload = res[0] if isinstance(res, tuple) else res
        if new_id is not None and payload.get('id') != new_id:
            payload = _renamed(payload, new_id)
        timings[payload.get('id', 'check')] = time.time() - t
        _log(f"{payload.get('id')}: {'PASS' if payload.get('passed') is True else 'FAIL'}")
        results.append(payload)
        return res

    record(V3.v0_preconditions, new_id='V0_preconditions')
    record(V3.v1_model_hash, new_id='V1_model_hash')
    record(v8_spec_binding)
    record(v9_spec_diff)
    record(V3.v10_production_unchanged, new_id='V10_no_production_change')
    try:
        w106_c6 = W106.c6_eval_keys()
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        w106_c6 = {'id': 'C6_committed_eval_keys_unchanged', 'passed': False, 'error': f'{type(error).__name__}: {error}'}
    extra['C6_w106_as_written_informational'] = {k: w106_c6.get(k) for k in ('id', 'passed', 'committed_specs', 'error')}
    record(v7_eval_keys_head_tree, w106_c6)
    import shared_resources_planning as srp          # production imports only after the guards (armed at import)
    import uncoordinated_benchmark as UB
    import p56a_oracle as O
    t = time.time()
    planning = O.load_baseline()['planning']
    timings['load_baseline_planning_s'] = time.time() - t
    candidate = BENCH._x0_candidate(srp, planning)
    t = time.time()
    certified = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])
    timings['unpickle_settled_certified_models_s'] = time.time() - t
    reference = UB.coordinated_reference_structure(planning, certified)
    warm = UB.extract_model_values(planning, certified)
    record(v4_reverse_flow_count, UB, srp, planning, certified, out_dir)
    record(v13_nrf_block_accounting, UB, planning)
    v2res = record(V3.v2_nrf_rows, UB, srp, planning, candidate, reference, certified, warm,
                   new_id='V2_nrf_rows_present_and_absent')
    digests = v2res[1] if isinstance(v2res, tuple) else None
    if digests is not None:
        record(V3.v3_tso_unchanged, UB, planning, candidate, reference, certified, digests,
               new_id='V3_tso_arm_unchanged_vs_v2')
    else:
        results.append({'id': 'V3_tso_arm_unchanged_vs_v2', 'passed': False, 'reason': 'V2 did not return digests'})
    del warm
    gc.collect()
    record(V3.v5_sweep_accounting, UB, planning, candidate, certified, new_id='V5_sweep_solve_accounting')
    del certified
    gc.collect()
    record(V3.v11_report_capture, _abs(W116_REVERSE_FLOW['path']), new_id='V11_report_capture_paths')
    record(V3.v12_stage_wiring, new_id='V12_stage_wiring')
    record(v14_spec_v4_content)
    record(V3.v6_typing_test, out_dir, new_id='V6_w100_bool_typing_test')
    guards = _all_guard_failures()
    passed = all(r.get('passed') is True for r in results) and not any(guards.values())
    payload = {'stage': 'P5.15 W119 zero-solve checks (benchmark spec v4)', 'utc': _utc(),
               'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(THIS), 'harness_sha256': _sha(HARNESS),
               'v3_script_sha256': _sha(V3_SCRIPT),
               'solve_profile_guard': {'permitted': [], 'verify_0': guards, 'counts_w119': dict(_GUARD.counts),
                                       'counts_w116_import': dict(V3._GUARD.counts),
                                       'counts_w111_import': dict(W111._GUARD.counts),
                                       'counts_w106_import': dict(W106._GUARD.counts)},
               'passed': bool(passed), 'n_checks': len(results),
               'failed': [r.get('id') for r in results if r.get('passed') is not True],
               'timings_s': timings, 'results': results, 'output_suffix': suffix, **extra}
    with open(os.path.join(out_dir, 'w119_zero_solve_checks.json'), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for name in sorted(files):
            manifest[os.path.relpath(os.path.join(root, name), REPO)] = H.sha256_file(os.path.join(root, name))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    _log(f"W119 checks: {'ALL PASS' if passed else 'FAIL ' + str(payload['failed'])}; guards {guards}; counts "
         f'W119 {dict(_GUARD.counts)}')
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--freeze-spec', action='store_true')
    group.add_argument('--checks', action='store_true')
    parser.add_argument('--output-suffix', default='', choices=('', '_r2', '_r3'),
                        help='checks output directory suffix (write-once; earlier runs are kept as evidence)')
    args = parser.parse_args(argv)
    try:
        return freeze_spec() if args.freeze_spec else run_checks(args.output_suffix)
    finally:
        W106._GUARD.uninstall()        # LIFO: W106 (top), W111, W116 (v3 script), W119
        W111._GUARD.uninstall()
        V3._GUARD.uninstall()
        _GUARD.uninstall()


if __name__ == '__main__':
    sys.exit(main())
