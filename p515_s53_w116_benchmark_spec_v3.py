"""P5.15 Addendum 57 Decision 1, Planner task W116 -- BENCHMARK SPEC v3 and its ZERO-SOLVE CHECKS. NO BENCHMARK STAGE RUNS
HERE (no sweep, no NRF arm, no report).

A `SolveProfileGuard(permitted=())` is armed BEFORE any production import in both modes; W106's and W111's modules,
imported for their checks, arm their own zero-solve guards at import; W114's module (imported by the sweep-accounting
check) installs a permitting guard at import, which is uninstalled at once (it sits on top of the blocking ones and
never sees a call). Every guard is verified at 0 at the end.

WHAT v3 IS (Addendum 57 Decision 1; predecessor v2 bb659122, not edited; v2's `report/` and `arm_*` names are consumed and
v2's tree is not touched): output root data/SRP1/Results/P515S53/w116_benchmark_nrf/ (new, write-once);
  (b) the NO-REVERSE-FLOW arms -- `uncoordinated_benchmark.NO_REVERSE_FLOW_ROW` (pg_adn >= 0, every DSO block, scenario
      and hour), passive and price-taker x three starts + the passive tie-breakers 0.1 / 10, the consistency
      re-evaluation, curtailment per arm (net primary, positive / negative parts, raw MWh); the claim
      min(passive_NRF, price_taker_NRF) - Q181 with the decomposition;
  (a) the SWEEP, report-only -- the unconstrained (v2) arms, cold, continuing past failing TSO blocks, one E3 elastic
      solve per failing block, per block and hour whether the TN can accept the DN schedule and by how much;
  (c) the REVERSE-FLOW COUNT of the coordinated Q181 solution -- zero-solve, computed and written HERE (check V4).
Stage harness: p515_s53_w116_benchmark_nrf.py. v3 = a deep copy of v2 with the changed keys overwritten and the new keys
added; the v2 -> v3 key diff is printed at freeze and recorded by check V9.

MODE --freeze-spec: writes <root>/frozen_s53_benchmark_spec_v3_<hash8>.json once (refuses unless the highest existing
  benchmark spec version anywhere under data/ is exactly 2, v2 has its committed sha256, and the bound files and this
  script are clean in git).
MODE --checks (default): write-once under <root>/w116_zero_solve_checks<suffix>/.
  V0  preconditions snapshot (locks, forbidden processes) -- W106's C0 plus the W116 lock
  V1  the settled models' hash (W106's C1, unchanged)
  V2  the NRF rows: present on every DSO block of freshly built NRF arms (both arms), 24 per block, active, lower 0,
      no upper, body IS pg_adn[s_m, s_o, p]; the structural check passes with them declared; the three starts apply;
      ABSENT from freshly built spec-v2 arms, from the TSO arm, from the coordinated Q181 models, and from every
      production source; the consistency re-evaluation deactivates and evaluates them; NEGATIVE controls: an
      undeclared NRF row and a deactivated NRF row fail the structural check; a planted reverse flow in the
      re-evaluation is reported as a hard 'no_reverse_flow' violation and triggers the sequential pass
  V3  the TSO arm unchanged against v2: `build_tso_arm_model` of the v2-bound uncoordinated_benchmark.py (git blob at v2's
      git_head, sha256 = v2's pin) and of the current one, at the same targets, compared EXHAUSTIVELY per block (every
      Var's bounds / fixed / value / domain, every row's activity / bounds / body, every Param value, every Objective and
      Expression); build and structural records equal; the TSO-path function sources identical. Also the spec-v2 DSO
      arm path unchanged (default no_reverse_flow=False, old vs new code) and the NRF build = the v2 build + exactly the
      NRF rows. NEGATIVE: a planted bound change is detected
  V4  the reverse-flow count on the Q181 models (definition: p515_s53_w116_benchmark_nrf.REVERSE_FLOW_DEFINITION),
      hand-checked on entries recomputed from the raw Vars, and the sign proven from the models (TN: sum pg - sum pc_adn
      = TN losses >= 0 every hour; the TN's only loads are the ADN interfaces); written to reverse_flow_count_q181.json
  V5  the sweep's solve accounting: MAX_ATTEMPTS = 3 from production's source; the declared upper bound 156 per arm;
      `BlockSolveAccount` exercised on a fake guard (correct sequences pass; too few / too many / out-of-range / E3 != 1
      / over the bound all raise); the E3 elastic copy built zero-solve on a freshly built TSO arm block; W114's guard
      handling on import; W113's cold records verified and the expected ranges derived from them
  V6  W100's repository-wide boolean-typing test
  V7  committed eval keys unchanged (W111's C6 with evaluation_key's full argument list; W106's C6 informational)
  V8  the frozen spec v3 binds (`frozen_spec_binding_failures() == []`); NEGATIVE: altered pin, name-hash mismatch, a v2
      spec in the root, all refused
  V9  v2 -> v3 key diff; v2 file unchanged and clean; v2's tree clean in git; changed keys within the declared set
  V10 no production file changed (git status)
  V11 the v3 report's capture paths: every quantity has a producer that writes its key (stage sources), the v2 inputs
      (sha-verified) and the reverse-flow file carry theirs; NEGATIVE: an empty output set is all absent
  V12 the stage wiring: every spec command parses to its declared run id and log name; per-stage permitted sites;
      nrf-arm passes no_reverse_flow=True, the sweep False

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  mkdir -p data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w116_benchmark_spec_v3.py --freeze-spec > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w116_freeze_spec_v3.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w116_benchmark_spec_v3.py --checks > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w116_zero_solve_checks.log 2>&1
Exit 0 = done / all checks pass; 1 = a check failed; 2 = refused.
"""

import argparse
import copy
import gc
import hashlib
import importlib.util
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

_GUARD = SolveProfileGuard((), label='P5.15 W116 zero-solve').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib at import
import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- stdlib + H + GRIO at import
import p515_s53_w116_benchmark_nrf as NRF  # noqa: E402 -- stdlib + H + GRIO + BENCH at import; no guard
import p515_s53_w111_benchmark_spec_v2 as W111  # noqa: E402 -- arms W106's and its own zero-solve guards at import

W106 = W111.W106
PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
THIS = os.path.basename(__file__)
OUT_ROOT_REL = NRF.OUT_ROOT_REL
CHECKS_DIR_REL = NRF.CHECKS_DIR_REL
LAUNCH_LOGS_REL = NRF.LAUNCH_LOGS_REL
HARNESS = NRF.SCRIPT_NAME
V2 = NRF.V2_SPEC
# v2's code state: the uncoordinated_benchmark.py blob at v2's git_head has v2's pinned sha256 (checked by V3)
V2_GIT_HEAD = '15a1d4563a23ddcd3fbe4f2b0712c7395cf20cc7'
V2_UB_SHA256 = 'fd80c93e5b0df9a39c2583e1b6b53df179af7b229276d29fb3611b1c3680360e'
INFORMATIONAL_FILES = W111.INFORMATIONAL_FILES + (THIS, 'p515_s53_w111_benchmark_spec_v2.py')
# V9: the top-level keys v3 may differ from v2 in (every other v2 key is carried identically and asserted)
V3_CHANGED_OR_ADDED_TOP_KEYS = (
    'version', 'predecessor', 'predecessor_note', 'stage', 'authority', 'frozen_utc', 'git_head', 'output_root',
    'code_sha256_binding', 'code_sha256_binding_rule', 'code_sha256_informational', 'stages_in_addendum_49_order',
    'stages_v3_addendum_57_order', 'v2_stages_not_rerun', 'solve_counts', 'wall_estimate', 'launch_rules',
    'preparation_command', 'claim', 'not_permitted', 'report_capture_paths', 'zero_solve_checks',
    'no_reverse_flow_addendum_57', 'sweep_addendum_57', 'reverse_flow_count_addendum_57',
    'predictions_recorded_before_any_run_v3', 'consistency_convention_v3', 'curtailment_reporting_addendum_56',
    'tolerances_v3', 'declared_choices_v3')
V2_KEYS_CARRIED_IDENTICAL = ('schema', 'instance', 'coordinated', 'repointed_inputs', 'tie_breaker', 'perturbation',
                             'arm_network_compl_inf_tol', 'declared_choices_status', 'tolerances',
                             'consistency_convention', 'lambda_t', 'curtailment_reporting',
                             'predictions_recorded_before_any_run', 'objective_convention',
                             'c4_scope_addendum_55', 'curtailment_signed_parts_addendum_55')
_T0 = time.time()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W116 +{time.time() - _T0:8.1f}s] {msg}', flush=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _stage_command(args, log_name):
    return f'set -o noclobber && {PY} -u {HARNESS} {args} > {LAUNCH_LOGS_REL}/{log_name}.log 2>&1'


# ======================================================================================================================
#  evidence read from W113 (zero solves; committed records)
# ======================================================================================================================
def _w113_records(arm, start):
    path = BENCH._verified_path(NRF.W113_COLD[arm]['per_solve_record']) if start == 'cold' else _abs(
        os.path.join(BENCH.OUT_ROOT_REL, f'arm_{arm}_{start}', 'per_solve_record.jsonl'))
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def start_dependence_from_w113():
    """The sweep's start: cold only, unless W113's committed records (all three starts of both v2 arms) show the
    TN-acceptance outcome to be start-dependent."""
    out = {}
    for arm in ('passive', 'price_taker'):
        by_start = {s: _w113_records(arm, s) for s in ('cold', 'warm_from_certified', 'perturbed')}
        first_fail = {s: [r['block'] for r in recs if not r['succeeded']] for s, recs in by_start.items()}
        retried = {s: [(r['block'], r['n_attempts']) for r in recs if r['n_attempts'] != 1]
                   for s, recs in by_start.items()}
        objectives = {s: {r['block']: r['active_objective_value'] for r in recs} for s, recs in by_start.items()}
        common = set.intersection(*(set(v) for v in objectives.values()))
        worst_abs, worst_rel, worst_block = 0.0, 0.0, None
        for block in sorted(common):
            vals = [objectives[s][block] for s in objectives]
            if any(v is None for v in vals):
                continue
            d = max(vals) - min(vals)
            if d > worst_abs:
                worst_abs, worst_block = d, block
            worst_rel = max(worst_rel, d / max(abs(vals[0]), 1e-12))
        out[arm] = {'failing_block_by_start': first_fail, 'retried_by_start': retried,
                    'n_blocks_solved_at_every_start': len(common),
                    'max_abs_objective_spread_eur': worst_abs, 'at_block': worst_block,
                    'max_rel_objective_spread': worst_rel,
                    'first_failing_block_start_independent': len({tuple(v) for v in first_fail.values()}) == 1}
    out['reading'] = (
        'W113 (committed, spec v2, all three starts of both arms): the first failing TSO block is the SAME at every '
        'start (passive 2035 Summer, price-taker 2025 Spring), each after 3 attempts, and the blocks solved at every '
        'start agree in their decision objective to the spreads above (the largest passive spread is on DSO 9 blocks '
        'whose objective is about -0.83; the price-taker spread is ~1e-9 relative). The records show no start '
        'dependence of the TN-acceptance outcome; blocks after the first failure were never solved at any start. '
        'Hence the sweep runs COLD ONLY (Planner task W116: cold only unless start-dependent from W113/W114).')
    out['start_independent_as_far_as_recorded'] = all(out[a]['first_failing_block_start_independent']
                                                      for a in ('passive', 'price_taker'))
    return out


def expected_sweep_counts_from_w113():
    """The sweep's launch count per arm expected from W113's cold records IF they reproduce (informational: the gate is
    the per-block exact accounting and the upper bound). Known blocks at their recorded attempts; the E3 of the known
    failing block (1); every block after it unknown, 1 (primary success) to 4 (3 attempts + E3)."""
    out = {}
    for arm in ('passive', 'price_taker'):
        recs = _w113_records(arm, 'cold')
        dso = [r for r in recs if r['kind'] == 'DSO']
        tso = [r for r in recs if r['kind'] == 'TSO']
        failing = [r for r in tso if not r['succeeded']]
        known = sum(r['n_attempts'] for r in dso) + sum(r['n_attempts'] for r in tso) + len(failing)
        unknown = NRF.SRP1_DECLARED_V3['tso_blocks'] - len(tso)
        out[arm] = {'dso_blocks_recorded': len(dso), 'dso_attempts_recorded': sum(r['n_attempts'] for r in dso),
                    'tso_blocks_recorded': len(tso), 'tso_attempts_recorded': sum(r['n_attempts'] for r in tso),
                    'failing_blocks_recorded': [r['block'] for r in failing], 'e3_launches_known': len(failing),
                    'tso_blocks_after_first_failure_unmeasured': unknown,
                    'expected_range': [known + unknown,
                                       known + unknown * (NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE
                                                          + NRF.E3_LAUNCHES_PER_FAILING_BLOCK)]}
    return out


# ======================================================================================================================
#  the frozen spec v3
# ======================================================================================================================
def stage_list():
    d = NRF.SRP1_DECLARED_V3
    arm_exact = d['nrf_arm_solves'] + d['nrf_reevaluation_solves']
    stages, order = [], 0
    for arm in ('passive', 'price_taker'):
        order += 1
        run_id = f'sweep_{arm}_cold'
        stages.append({'order': order, 'stage': 'sweep', 'arm': arm, 'start': 'cold', 'run_id': run_id,
                       'command': _stage_command(f'--stage sweep --arm {arm}', run_id),
                       'declared_solves': {'upper_bound': d['sweep_upper_bound_per_arm'],
                                           'accounting': 'exact per block (BlockSolveAccount); see sweep_addendum_57'},
                       'guard': f'SolveProfileGuard(permitted={NRF.PERMITTED_SWEEP_SITES}); verify(0) before solves; '
                                'verify(cumulative) after every block; cumulative <= upper bound',
                       'report_only': True, 'dso_decision_tie_breaker': BENCH.TIE_BREAKER['decision'][f'{arm}_dso'],
                       'wall_estimate_min': [1, 5]})
    for arm in ('passive', 'price_taker'):
        for start in ('cold', 'warm_from_certified', 'perturbed'):
            order += 1
            run_id = f'nrf_arm_{arm}_{start}'
            stages.append({'order': order, 'stage': 'nrf-arm', 'arm': arm, 'start': start, 'run_id': run_id,
                           'command': _stage_command(f'--stage nrf-arm --arm {arm} --start {start}', run_id),
                           'declared_solves': arm_exact,
                           'declared_solves_detail': {'phase_A_arm': d['nrf_arm_solves'],
                                                      'phase_B_consistency_reevaluation': d['nrf_reevaluation_solves'],
                                                      'phase_C_sequential_pass_only_if_triggered':
                                                          d['nrf_sequential_pass_solves'],
                                                      'total_if_triggered': arm_exact + d['nrf_sequential_pass_solves']},
                           'guard': f'SolveProfileGuard(permitted={NRF.PERMITTED_ARM_SITES}); exact cumulative verify '
                                    'at every phase boundary: 0, 48, 84 (132 if the declared trigger fires) -- v2\'s '
                                    'convention: a production retry raises the count and fails the stage',
                           'dso_decision_tie_breaker': BENCH.TIE_BREAKER['decision'][f'{arm}_dso'],
                           'no_reverse_flow': True, 'wall_estimate_min': [1, 5]})
    for value, tag in ((0.1, '0p1'), (10.0, '10')):
        order += 1
        run_id = f'nrf_passive_tie_breaker_{tag}'
        stages.append({'order': order, 'stage': 'nrf-passive-tie-breaker', 'value': value, 'run_id': run_id,
                       'command': _stage_command(f'--stage nrf-passive-tie-breaker --value {value:g}', run_id),
                       'declared_solves': d['nrf_variant_solves'],
                       'guard': f'SolveProfileGuard(permitted={NRF.PERMITTED_ARM_SITES}); exact: 0 before, 48 end',
                       'no_reverse_flow': True, 'consistency': False, 'wall_estimate_min': [1, 3]})
    order += 1
    stages.append({'order': order, 'stage': 'report', 'run_id': NRF.REPORT_RUN_ID,
                   'command': _stage_command('--stage report', NRF.REPORT_RUN_ID), 'declared_solves': 0,
                   'guard': 'SolveProfileGuard(permitted=()); verify(0) at the end', 'wall_estimate_min': [0.2, 1]})
    return stages


def predictions_v3(expected_counts):
    searched = ('PLANNER_BRIEF_2026-09-13.md Addendum 57, TASKS.md Addendum 57 order, the W116 task text, '
                'P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md')
    return {
        'recorded_before_any_run': True,
        'nrf_arms_feasible_at_every_tso_block': {
            'planner_prediction': ('the NRF arms are feasible at every TSO block, because NRF removes the export '
                                   'mechanism W114/W115 found (Planner task W116)'),
            'worker_expectation': (
                'agree. With pg_adn >= 0 at every interface the TN load at each ADN bus is non-negative; the TN has no '
                'other load, its generators floor at 0 MW with >= 836 MW headroom (W115), so the TN P balance has a '
                'solution at any non-negative import; W114/W115 found zero slack on Q, voltage and thermal rows at '
                'the failing blocks (only the P balance bound). Residual risk, not seen in any record: an hour in '
                'which all three DNs sit at exactly zero import leaves the TN generating only its losses, with '
                'voltage support from generator Q alone -- expected feasible (Pmin 0)'),
            'scored_by': 'report_v3 predictions_scored: held iff no NRF arm or variant run fails'},
        'nrf_claim_sign': {
            'planner_prediction': None,
            'planner_prediction_note': f'no Planner-recorded sign or magnitude found (searched: {searched})',
            'worker_expectation': {
                'sign': 'positive: min(passive_NRF, price_taker_NRF) - Q181 > 0',
                'verdict': ('coordination beats the best NRF arrangement, determinate (above the larger of the best '
                            'NRF arm band and 0.011 % x Q181 = 71,926 EUR)'),
                'magnitude': 'not predicted',
                'passive_minus_price_taker_sign': 'not predicted (Addendum 49 Ruling 1: not guaranteed)',
                'reasoning': (
                    '(i) each NRF arm point satisfies the interface P/Q consensus by construction (TSO fixed to the '
                    'DSO schedule) and adds the NRF restriction, so up to the interface-voltage mismatch the '
                    'consistency step measures it is a feasible point of the coordinated problem, and at the common '
                    'pricing its Q bounds the coordinated optimum from above; (ii) the coordinated cell is a settled '
                    'local optimum -- a negative sign needs an arm to reach a better basin, which the three-start band '
                    'would expose; (iii) mechanism: at 2035 Summer h11-16 the coordinated DSOs import 166-209 MW where '
                    'the passive DNs export (W114), i.e. coordination moves DN demand into RES-covered TN hours, which '
                    'neither NRF arm can do (passive: no flexibility; price-taker: prices pi_t, and lambda_t != pi_t '
                    'on 848/864 rows, W113)')}},
        'sweep_n_of_12': {
            'planner_prediction': None,
            'planner_prediction_note': f'no Planner-recorded n / 12 found (searched: {searched})',
            'worker_expectation': {
                'passive': {
                    'n_range': [1, 3], 'central': 1, 'h_central': 6,
                    'known_from_records': ('W113 cold: TSO 2025 x 4, 2030 x 4 and 2035 Spring solved (accepted); 2035 '
                                           'Summer not, in hours 11-16 (W114 E3: 18 non-zero interface-P slacks = 3 '
                                           'nodes x 6 hours); 2035 Autumn and Winter never solved'),
                    'reasoning': ('the passive export is the midday PV surplus of the 2035 DNs with flexibility fixed '
                                  'at 0; Spring 2035 was accepted, and Autumn / Winter PV is lower than Summer\'s, so '
                                  'both are expected accepted -- low confidence on Autumn')},
                'price_taker': {
                    'n_range': [1, 12], 'central': 'at least 6', 'h_central': 'not predicted',
                    'known_from_records': ('W113 cold: its FIRST TSO block (2025 Spring) fails, in hours 4-7 (W115 E3: '
                                           '12 non-zero interface-P slacks = 3 nodes x 4 hours); the other 11 blocks '
                                           'were never solved'),
                    'reasoning': ('the price-taker exports wherever selling at pi_t beats its own flexibility cost '
                                  '(c_flex < pi_t in the large majority of hours, W113 lambda look); its exports '
                                  'are not tied to PV, so they are expected in most blocks -- low confidence')}},
            'expected_launch_counts_from_w113_records': expected_counts},
        'nrf_consistency_trigger': {
            'planner_prediction': None,
            'worker_expectation': ('the sequential pass is LIKELY to trigger in at least one NRF arm: wherever the NRF '
                                   'row binds (pg_adn = 0), re-evaluating the DN at the TN\'s actual interface voltage '
                                   'moves the DN losses and can push pg_adn below 0 by more than hard_tol (1e-6 p.u.); '
                                   'the declared counts carry 132 for that case -- low confidence')},
        'carried_from_v2_still_applying': {
            'passive_tie_breaker_value_independence': ('re-solving passive NRF at 0.1 and 10 EUR/MWh leaves the '
                                                       'interface schedule within solver tolerance of the 1 EUR/MWh '
                                                       'run; any residual is the non-unique distribution of '
                                                       'curtailment among units (Addendum 49 clarification)'),
            'q_min_over_starts': BENCH.Q_MIN_OVER_STARTS_NOTE,
            'arm_band': ('each NRF arm\'s band is the spread of its three starts; a difference is claimed only above '
                         'max(arm band, 0.011 % x Q181 = 71,926.11 EUR)')},
        'scope_note': ('the v2 key predictions_recorded_before_any_run is carried unchanged (v2\'s, historical); its '
                       'lambda-look and common-Q-gate items were scored under v2 (W113) and stand (Addendum 57)'),
    }


def build_spec_v3(v2):
    v3 = copy.deepcopy(v2)
    v3.pop('stages_in_addendum_49_order')
    d = NRF.SRP1_DECLARED_V3
    stages = stage_list()
    expected_counts = expected_sweep_counts_from_w113()
    arm_exact = d['nrf_arm_solves'] + d['nrf_reevaluation_solves']
    v3.update({
        'version': 3,
        'predecessor': {'path': V2['path'], 'sha256': V2['sha256'], 'version': 2, 'committed_in': V2['committed_in'],
                        'git_head_at_freeze': V2_GIT_HEAD},
        'predecessor_note': (
            'v3 = v2 with Addendum 57 Decision 1: the NRF arms (b), the report-only sweep (a), the coordinated '
            'reverse-flow count, a new output root (v2\'s report/ and arm_* names are consumed; v2\'s tree is not '
            'touched). Carried from v2 unchanged (V9 asserts it): ' + ', '.join(V2_KEYS_CARRIED_IDENTICAL)
            + '. v2 is not edited'),
        'stage': NRF.STAGE, 'authority': NRF.AUTHORITY, 'frozen_utc': _utc(), 'git_head': _git(['rev-parse', 'HEAD']),
        'output_root': NRF.OUT_ROOT_REL,
        'code_sha256_binding': {name: _sha(name) for name in NRF.FROZEN_SPEC_BOUND_FILES},
        'code_sha256_binding_rule': ('every v3 stage refuses unless each listed file on disk has exactly this sha256 '
                                     '(p515_s53_w116_benchmark_nrf.frozen_spec_binding_failures), the spec is the '
                                     'highest version under output_root, version >= 3, and its name hash matches'),
        'code_sha256_informational': {name: W106._informational_pin(name) for name in INFORMATIONAL_FILES},
        'stages_v3_addendum_57_order': stages,
        'v2_stages_not_rerun': {
            'read_only_inputs': {name: {**entry, 'committed_in': NRF.V2_STAGE_OUTPUTS_COMMITTED_IN,
                                        'status': 'result stands (Addendum 57); read sha-verified by report_v3'}
                                 for name, entry in NRF.V2_STAGE_OUTPUTS.items()},
            'consumed_write_once_names': ['arm_passive_cold', 'arm_passive_warm_from_certified', 'arm_passive_perturbed',
                                          'arm_price_taker_cold', 'arm_price_taker_warm_from_certified',
                                          'arm_price_taker_perturbed', 'report'],
            'never_run_under_v2': ['passive_tie_breaker_0p1', 'passive_tie_breaker_10 (superseded by the NRF variants)'],
        },
        'solve_counts': {
            'sweep_per_arm': {'upper_bound': d['sweep_upper_bound_per_arm'],
                              'upper_bound_formula': '36 DSO x 3 attempts + 12 TSO x (3 attempts + 1 E3)',
                              'rule': 'exact per block against production-recorded attempts; see sweep_addendum_57',
                              'expected_from_w113_records': {a: v['expected_range'] for a, v in expected_counts.items()}},
            'nrf_arm_per_run_exact': arm_exact, 'nrf_arm_per_run_if_triggered': arm_exact + d['nrf_sequential_pass_solves'],
            'nrf_arm_runs': 6, 'nrf_variant_per_run_exact': d['nrf_variant_solves'], 'nrf_variant_runs': 2,
            'report': 0,
            'total_nrf_exact': 6 * arm_exact + 2 * d['nrf_variant_solves'],
            'total_nrf_if_every_arm_triggers': 6 * (arm_exact + d['nrf_sequential_pass_solves'])
            + 2 * d['nrf_variant_solves'],
            'total_upper_bound_all_stages': (2 * d['sweep_upper_bound_per_arm']
                                             + 6 * (arm_exact + d['nrf_sequential_pass_solves'])
                                             + 2 * d['nrf_variant_solves']),
            'per_stage_rule': ('NRF arms / variants: exact at every phase boundary (v2 convention; a retry fails the '
                               'stage); sweep: exact per block + upper bound; report: 0'),
        },
        'wall_estimate': {'total_min': [12, 50],
                          'basis': ('ESTIMATES: W113 (spec v2, same machine) -- arm stages ran 37-46 solves in 13-41 s '
                                    'of solve wall, 44 s end to end incl. planning load (12 s), unpickling (7 s) and '
                                    'builds (5 s for 36 DSO blocks, measured by the W116 probe)')},
        'launch_rules': ('one stage at a time, attached, alone, both streams to a NEW log under output_root/launch_logs '
                         '(noclobber); never detached; the order of stages_v3_addendum_57_order; each stage refuses '
                         'while a campaign / G1-G4 gate / other p515_s53_* process or a W93 / W116 lock is live, or '
                         'while its output or IPOPT log directory exists'),
        'preparation_command': f'mkdir -p {LAUNCH_LOGS_REL}',
        'claim': {
            'definition': 'benefit = min(Q_passive_NRF, Q_price_taker_NRF) - Q181 (Addendum 57 Decision 1(b))',
            'measures': 'dynamic coordination against a static interface limit (no reverse flow at any interface)',
            'decomposition': ['passive_NRF - price_taker_NRF', 'price_taker_NRF - coordinated',
                              'passive_NRF - coordinated'],
            'q_arm': BENCH.Q_MIN_OVER_STARTS_NOTE, 'q_coordinated': v2['claim']['q_coordinated'],
            'resolution': v2['claim']['resolution'],
            'gate_first': ('no arm is reported unless the common-Q gate PASSED -- v2\'s gate output (PASS_BITWISE, '
                           'read-only, sha-verified)'),
            'reverse_flow_caveat': ('if the coordinated Q181 solution has any reverse-flow interface-hour '
                                    '(reverse_flow_count_addendum_57), part of the measured benefit is the value of '
                                    'allowing reverse flow, and the report says so'),
            'verdicts': ['coordination beats the best NRF arrangement', 'inside the band',
                         'the best NRF arrangement beats coordination']},
        'not_permitted': ['any production-file change', 'running any benchmark stage under W116 (freeze and checks only)',
                          'any change to the formulation, the TSO arm or the Addendum 49 definitions other than the '
                          'Addendum 57 NRF row', 'writing into spec v2\'s tree'],
        'report_capture_paths': {k: {'stage_output': s, 'key_path': list(p)}
                                 for k, (s, p) in NRF.REPORT_CAPTURE_PATHS_V3.items()},
        'zero_solve_checks': {'command': (f'set -o noclobber && {PY} -u {THIS} --checks > '
                                          f'{LAUNCH_LOGS_REL}/w116_zero_solve_checks.log 2>&1'),
                              'output': CHECKS_DIR_REL,
                              'note': 'run AFTER this freeze; V8 verifies this spec binds; V9 diffs it against v2; V4 '
                                      'writes the reverse-flow count'},
        'no_reverse_flow_addendum_57': {**NRF.NRF_DEFINITION,
                                        'implemented_in': ('uncoordinated_benchmark.py: NO_REVERSE_FLOW_ROW, '
                                                           '_no_reverse_flow_rule, build_dso_arm_models('
                                                           'no_reverse_flow=True), check_arm_structures (declared rows), '
                                                           'run_operational_planning_uncoordinated(no_reverse_flow), '
                                                           'build_consistency_reevaluation_block / '
                                                           'consistency_violations (re-evaluation)'),
                                        'rows_per_dso_block_srp1': d['nrf_rows_per_dso_block']},
        'sweep_addendum_57': {
            'definition': NRF.SWEEP_DEFINITION, 'arms': ['passive', 'price_taker'],
            'start': 'cold only', 'start_dependence_from_w113': start_dependence_from_w113(),
            'solve_accounting': {
                'why_not_exact_in_advance': ('the number of failing TSO blocks -- each adding production retries and '
                                             'one E3 -- is what the sweep measures'),
                'upper_bound_per_arm': d['sweep_upper_bound_per_arm'],
                'max_attempts_per_network_solve': NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE,
                'max_attempts_source': ('network._run_smopf: primary, recovery (tier 1) and tier-2 attempts, each one '
                                        '_run_smopf_solver_attempt (checked by V5 from the source)'),
                'e3_launches_per_failing_block': NRF.E3_LAUNCHES_PER_FAILING_BLOCK,
                'per_block_rule': ('p515_s53_w116_benchmark_nrf.BlockSolveAccount: the guard delta (solves and process '
                                   'launches) over each block == the attempts production recorded for it '
                                   '(_drain_network_ipopt_solve_records, one record per attempt), 1..3; exactly 1 for '
                                   'an E3; the cumulative total verified exactly (SolveProfileGuard.verify) after every '
                                   'block; > upper bound raises'),
                'expected_from_w113_records': expected_counts,
                'dso_failure': 'a DSO block failing every tier ends the sweep (no schedule to test); recorded'},
            'e3': {'module': NRF.W114_SCRIPT, 'variant': NRF.E3_VARIANT_LABEL, 'rows': list(NRF.E3_ROWS),
                   'functions': ['build_elastic_copy', 'elastic_solve', 'read_slacks', 'interface_moves', 'tn_state'],
                   'evidence': 'W114 b0ed4d14 / W115 568284d1: E3 terminated optimal at both known failing blocks'},
            'output': 'per_block, per_block_hour (every TSO block x 24 hours), n / 12, h, statement, the DSO schedule, '
                      'the unconstrained DSO reverse-flow count, solve ledger, W113 reproduction (reported)'},
        'reverse_flow_count_addendum_57': {
            'definition': NRF.REVERSE_FLOW_DEFINITION,
            'models': {'path': BENCH.COORDINATED['certified_models']['path'],
                       'sha256': BENCH.COORDINATED['certified_models']['sha256'],
                       'cell': 'd110bd1a5977df1e_x0', 'certification_cycle': 181},
            'computed_by': f'{THIS} --checks, check V4 (zero solves)', 'output': NRF.REVERSE_FLOW_OUTPUT_REL,
            'hand_check': 'entries recomputed from the raw pg[ref_gen] and shared_es_pnet Vars and the DSO / TSO '
                          'expected-interface Vars',
            'sign_check': 'TN: sum pg - sum pc_adn = TN losses >= 0 at every block and hour (no TN load but the ADN '
                          'interfaces)'},
        'predictions_recorded_before_any_run_v3': predictions_v3(expected_counts),
        'consistency_convention_v3': ('v2\'s convention, plus: the NRF rows are deactivated in the re-evaluation clone '
                                      '(with the voltage / thermal rows; the reference generator is freed there) and '
                                      'evaluated as hard DN limits (kind no_reverse_flow, excess p.u., hard_tol); a '
                                      'violation triggers the one sequential pass, in which the DSO re-solves with the '
                                      'rows at the TN\'s actual voltage'),
        'curtailment_reporting_addendum_56': ('ruled (Addendum 56): repository and benchmark tables -- the NET '
                                              '(production\'s definition) is the frozen primary, the positive and '
                                              'negative parts and raw MWh beside it; manuscript -- the POSITIVE part '
                                              '(MWh). Every NRF arm, start and variant row, and the coordinated row '
                                              '(v2 common-Q gate output); rows are phase A (arm_cost_source stated)'),
        'tolerances_v3': {'sweep_accept_threshold_pu_tn_base': NRF.SWEEP_ACCEPT_THRESHOLD_PU,
                          'reverse_flow_material_tol_pu_dn_base': NRF.REVERSE_FLOW_MATERIAL_TOL_PU,
                          'consistency': BENCH.CONSISTENCY_TOL,
                          'note': 'the NRF rows are compared against hard_tol_pu2 in the re-evaluation (p.u.)'},
        'declared_choices_v3': {
            'items': ['sweep start cold only (from W113 records, sweep_addendum_57.start_dependence_from_w113)',
                      'reverse-flow count primary strict (p < 0), material count at -1e-6 p.u. beside it',
                      'sweep accept threshold 1e-6 p.u. of the TN base (W114\'s slack threshold)',
                      'NRF rows evaluated as hard limits in the consistency re-evaluation',
                      'NRF arms keep v2\'s exact phase counts (a retry fails the stage)'],
            'status': 'Worker-declared under W116 for the Planner to confirm before the stages run'},
    })
    return v3


def freeze_spec():
    dirty = _git(['status', '--porcelain', '--'] + list(NRF.FROZEN_SPEC_BOUND_FILES) + [THIS])
    if dirty:
        print(f'REFUSING to freeze: bound files not clean in git:\n{dirty}', flush=True)
        return 2
    highest = W106._highest_existing_version()
    if highest != 2:
        print(f'REFUSING to freeze: highest existing benchmark spec version is {highest}, expected exactly 2', flush=True)
        return 2
    v2 = W111._load_verified_json(V2)
    v3 = build_spec_v3(v2)
    text = GRIO.dumps(v3, indent=1, sort_keys=True) + '\n'
    data = text.encode()
    sha = hashlib.sha256(data).hexdigest()
    os.makedirs(_abs(OUT_ROOT_REL), exist_ok=True)
    path = _abs(os.path.join(OUT_ROOT_REL, f'frozen_s53_benchmark_spec_v3_{sha[:8]}.json'))
    with open(path, 'xb') as handle:
        handle.write(data)
    if H.sha256_file(path) != sha:
        raise RuntimeError('frozen spec sha mismatch after write')
    diff = W111.key_diff(v2, json.loads(text))
    failures = _GUARD.verify(0) + W106._GUARD.verify(0) + W111._GUARD.verify(0)
    _log(f'frozen spec v3: {os.path.relpath(path, REPO)} sha256 {sha}; predecessor v2 {V2["sha256"]}; git_head '
         f'{v3["git_head"]}; solve counts {v3["solve_counts"]}; guard verify(0) {failures}')
    top = sorted({x['path'].split('.')[0].split('[')[0] for x in diff})
    _log(f'v2 -> v3 key diff: {len(diff)} paths; top-level keys changed / added / removed: {top}')
    for x in diff:
        if '.' not in x['path'] and '[' not in x['path']:
            _log(f"  {x['kind']:8s} {x['path']}")
    return 0 if not failures else 1


# ======================================================================================================================
#  checks
# ======================================================================================================================
def v0_preconditions():
    res = W106.c0_preconditions()
    res['id'] = 'V0_preconditions'
    res['w116_lock_absent'] = not os.path.exists(NRF.LOCK_PATH)
    res['passed'] = bool(res['passed'] and res['w116_lock_absent'])
    return res


def v1_model_hash():
    res = W106.c1_model_hash()
    res['id'] = 'V1_model_hash'
    return res


def _nrf_row_facts(UB, block):
    comp = block.component(UB.NO_REVERSE_FLOW_ROW)
    if comp is None:
        return {'present': False}
    n_expected = len(block.scenarios_market) * len(block.scenarios_operation) * len(block.periods)
    bad = []
    n = 0
    for s_m in block.scenarios_market:
        for s_o in block.scenarios_operation:
            for p in block.periods:
                n += 1
                cd = comp[s_m, s_o, p]
                ok = (cd.active and cd.has_lb() and not cd.has_ub() and float(cd.lower) == 0.0
                      and cd.body is block.pg_adn[s_m, s_o, p])
                if not ok:
                    bad.append(cd.name)
    return {'present': True, 'n_rows': len(comp), 'n_expected': n_expected, 'n_checked': n,
            'n_active': sum(1 for c in comp.values() if c.active), 'bad_rows': bad[:5],
            'ok': len(comp) == n_expected == n and not bad}


def _blocks(models, planning):
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        for year in dn.years:
            for day in dn.days:
                yield node_id, year, day, dn, models[node_id][year][day]


def model_digest(block, exclude=()):
    """sha256 per category over a canonical listing of the block (sorted component data): Vars (name, lb, ub, fixed,
    value, domain), Constraints (name, active, lower, upper, body text), Params (name, value), Objectives (name, active,
    sense, expression text), Expressions (name, expression text); components named in `exclude` are skipped."""
    import pyomo.environ as pe
    h = {k: hashlib.sha256() for k in ('var', 'con', 'param', 'obj', 'expr')}
    n = {k: 0 for k in h}

    def skip(data):
        return data.parent_component().name in exclude

    for v in block.component_data_objects(pe.Var, descend_into=True, sort=True):
        if skip(v):
            continue
        h['var'].update(repr((v.name, v.lb, v.ub, v.fixed, v.value, str(v.domain))).encode())
        n['var'] += 1
    for c in block.component_data_objects(pe.Constraint, active=None, descend_into=True, sort=True):
        if skip(c):
            continue
        lower = None if c.lower is None else pe.value(c.lower)
        upper = None if c.upper is None else pe.value(c.upper)
        h['con'].update(repr((c.name, c.active, lower, upper, str(c.body))).encode())
        n['con'] += 1
    for comp in block.component_objects(pe.Param, descend_into=True, sort=True):
        if comp.name in exclude:
            continue
        for idx in sorted(comp.keys(), key=repr):
            h['param'].update(repr((comp.name, idx, pe.value(comp[idx], exception=False))).encode())
            n['param'] += 1
    for o in block.component_data_objects(pe.Objective, active=None, descend_into=True, sort=True):
        h['obj'].update(repr((o.name, o.active, o.sense, str(o.expr))).encode())
        n['obj'] += 1
    for e in block.component_data_objects(pe.Expression, descend_into=True, sort=True):
        if skip(e):
            continue
        h['expr'].update(repr((e.name, str(e.expr))).encode())
        n['expr'] += 1
    return {'sha256': {k: v.hexdigest() for k, v in h.items()}, 'counts': n}


def v2_nrf_rows(UB, srp, planning, candidate, reference, certified, warm):
    decision = BENCH.TIE_BREAKER['decision']
    per_arm, digests = {}, {}
    production_sources = sorted(set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) | {'shared_resources_planning.py',
                                                                            'model_construction_helpers.py',
                                                                            'network.py', 'network_data.py',
                                                                            'shared_energy_storage_data.py'})
    for arm in UB.DSO_ARMS:
        rec = {}
        for nrf in (True, False):
            models, build = UB.build_dso_arm_models(planning, candidate['total_capacity'], arm=arm,
                                                    curtailment_penalty=decision[f'{arm}_dso'], no_reverse_flow=nrf)
            structure = UB.check_arm_structures(planning, {'dso': models}, reference, dso_build_record=build)
            facts = {UB.block_label('DSO', n, y, d): _nrf_row_facts(UB, b) for n, y, d, _dn, b in _blocks(models, planning)}
            digests[(arm, nrf)] = {label: model_digest(b, exclude=(UB.NO_REVERSE_FLOW_ROW,) if nrf else ())
                                   for label, b in ((UB.block_label('DSO', n, y, d), b)
                                                    for n, y, d, _dn, b in _blocks(models, planning))}
            entry = {'build_flag': build.get('no_reverse_flow'),
                     'n_blocks': len(facts),
                     'rows_declared_per_block': sorted({b.get('no_reverse_flow_rows_expected', 0)
                                                        for b in build['blocks'].values()}),
                     'structure_passed_all': all(r['passed'] for r in structure.values()),
                     'structure_nrf_rows_per_block': sorted({r['no_reverse_flow_rows'] for r in structure.values()}),
                     'example_arithmetic': next(iter(structure.values()))['arithmetic']}
            if nrf:
                entry['every_block_rows_ok'] = all(f.get('ok') is True for f in facts.values())
                entry['n_rows_total'] = sum(f.get('n_rows', 0) for f in facts.values())
                # the three starts apply to the NRF models (no Var is added: warm values map one to one)
                starts, start_records = {}, {}
                for start in UB.STARTS:
                    m2, _b2 = (models, build) if start == UB.START_COLD else UB.build_dso_arm_models(
                        planning, candidate['total_capacity'], arm=arm, curtailment_penalty=decision[f'{arm}_dso'],
                        no_reverse_flow=True)
                    recs = UB.apply_start(planning, {'tso': None, 'dso': m2}, start=start, warm_values=warm,
                                          perturbation=BENCH.PERTURBATION if start == UB.START_PERTURBED else None,
                                          agents=('DSO',))
                    start_records[start] = recs
                    starts[start] = {'n_blocks': len(recs),
                                     'warm_n_missing_max': max((r.get('warm', {}).get('n_missing', 0)
                                                                for r in recs.values()), default=0),
                                     'warm_n_set_min': min((r.get('warm', {}).get('n_set', 0) for r in recs.values()),
                                                           default=0)}
                    if m2 is not models:
                        del m2
                entry['starts_apply'] = starts
                entry['start_records'] = start_records
                if arm == UB.ARM_PASSIVE:
                    entry['reevaluation_path'] = _reevaluation_path(UB, planning, models, nrf_expected=True)
                    entry['NEGATIVE'] = _nrf_negative_controls(UB, planning, models, build, reference)
            else:
                entry['no_block_has_nrf_rows'] = all(f['present'] is False for f in facts.values())
                # the same three starts on the spec-v2 arm: the NRF rows add no Var, so every per-block start record
                # (warm n_set / n_skipped_fixed / n_missing / n_outside_arm_bounds; perturbation draws) must be equal
                same = {}
                for start in UB.STARTS:
                    m2 = models if start == UB.START_COLD else UB.build_dso_arm_models(
                        planning, candidate['total_capacity'], arm=arm, curtailment_penalty=decision[f'{arm}_dso'],
                        no_reverse_flow=False)[0]
                    recs = UB.apply_start(planning, {'tso': None, 'dso': m2}, start=start, warm_values=warm,
                                          perturbation=BENCH.PERTURBATION if start == UB.START_PERTURBED else None,
                                          agents=('DSO',))
                    same[start] = recs == rec['nrf']['start_records'][start]
                    if m2 is not models:
                        del m2
                entry['starts_apply_identically_to_nrf'] = same
                if arm == UB.ARM_PASSIVE:
                    entry['reevaluation_path'] = _reevaluation_path(UB, planning, models, nrf_expected=False)
            rec['nrf' if nrf else 'v2_type'] = entry
            del models
            gc.collect()
        rec['nrf'].pop('start_records', None)
        per_arm[arm] = rec
    tso_model, _tb = UB.build_tso_arm_model(planning, candidate['total_capacity'],
                                            UB.get_dso_interface_schedule(planning, certified['dso']),
                                            curtailment_penalty=decision['tso'], coupling=UB.TSO_COUPLING_FIXED,
                                            pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
    tso_absent = all(tso_model[y][d].component(UB.NO_REVERSE_FLOW_ROW) is None
                     for y in tso_model for d in tso_model[y])
    del tso_model
    coordinated_absent = (all(b.component(UB.NO_REVERSE_FLOW_ROW) is None for *_x, b in _blocks(certified['dso'], planning))
                          and all(certified['tso'][y][d].component(UB.NO_REVERSE_FLOW_ROW) is None
                                  for y in certified['tso'] for d in certified['tso'][y]))
    source_hits = {name: [s for s in ('uncoord_no_reverse_flow', 'NO_REVERSE_FLOW', 'no_reverse_flow')
                          if s in open(_abs(name)).read()] for name in production_sources if os.path.exists(_abs(name))}
    harness_src = {'stage_nrf_arm': inspect.getsource(NRF.stage_nrf_arm),
                   'stage_sweep': inspect.getsource(NRF.stage_sweep)}
    wiring = {'nrf_arm_passes_no_reverse_flow_true': 'record_callback=sink, no_reverse_flow=True)' in harness_src[
        'stage_nrf_arm'],
              'sweep_builds_no_reverse_flow_false': 'no_reverse_flow=False)' in harness_src['stage_sweep'],
              'run_function_default_false': inspect.signature(
                  UB.run_operational_planning_uncoordinated).parameters['no_reverse_flow'].default is False,
              'build_function_default_false': inspect.signature(
                  UB.build_dso_arm_models).parameters['no_reverse_flow'].default is False}
    nrf_ok = all(per_arm[a]['nrf']['every_block_rows_ok'] and per_arm[a]['nrf']['structure_passed_all']
                 and per_arm[a]['nrf']['build_flag'] is True
                 and per_arm[a]['nrf']['rows_declared_per_block'] == [NRF.SRP1_DECLARED_V3['nrf_rows_per_dso_block']]
                 and per_arm[a]['nrf']['structure_nrf_rows_per_block'] == [NRF.SRP1_DECLARED_V3['nrf_rows_per_dso_block']]
                 and per_arm[a]['nrf']['n_blocks'] == NRF.SRP1_DECLARED_V3['dso_blocks']
                 and per_arm[a]['nrf']['n_rows_total'] == 36 * 24
                 and all(s['n_blocks'] == 36 and s['warm_n_set_min'] > 0
                         for k, s in per_arm[a]['nrf']['starts_apply'].items() if k != UB.START_COLD)
                 and all(per_arm[a]['v2_type']['starts_apply_identically_to_nrf'].values())
                 for a in UB.DSO_ARMS)
    v2_type_ok = all(per_arm[a]['v2_type']['no_block_has_nrf_rows'] and per_arm[a]['v2_type']['structure_passed_all']
                     and per_arm[a]['v2_type']['build_flag'] is False
                     and per_arm[a]['v2_type']['structure_nrf_rows_per_block'] == [0] for a in UB.DSO_ARMS)
    reeval = per_arm['passive']['nrf']['reevaluation_path']
    reeval_v2 = per_arm['passive']['v2_type']['reevaluation_path']
    negative = per_arm['passive']['nrf']['NEGATIVE']
    passed = (nrf_ok and v2_type_ok and tso_absent and coordinated_absent and not any(source_hits.values())
              and all(wiring.values()) and reeval['passed'] and reeval_v2['passed'] and all(negative.values()))
    return {'id': 'V2_nrf_rows_present_and_absent', 'passed': bool(passed), 'per_arm': per_arm,
            'tso_arm_has_no_nrf_rows': tso_absent, 'coordinated_q181_models_have_no_nrf_rows': coordinated_absent,
            'production_source_mentions': source_hits, 'harness_wiring': wiring,
            'definition': NRF.NRF_DEFINITION}, digests


def _reevaluation_path(UB, planning, models, *, nrf_expected):
    """Zero solves: the re-evaluation clone of an NRF block deactivates the NRF rows and records them; a planted reverse
    flow (the reference generator's P set to -0.05 p.u. at hour 4) is reported as a hard 'no_reverse_flow' violation
    and triggers the sequential pass. On a spec-v2 (no-NRF) block the clone has no NRF rows, records none, and the same
    plant produces no 'no_reverse_flow' item (v2's re-evaluation unchanged)."""
    node_id = sorted(planning.distribution_networks)[0]
    dn = planning.distribution_networks[node_id]
    year, day = next(iter(dn.years)), next(iter(dn.days))
    network = dn.network[year][day]
    block = models[node_id][year][day]
    # give every Var a value (cold start leaves some None): the re-evaluation builder needs values to fix
    for v in block.component_data_objects(__import__('pyomo.environ', fromlist=['Var']).Var, descend_into=True):
        if v.value is None:
            v.set_value(0.0, skip_validation=True)
    ref_idx = network.get_node_idx(network.get_reference_node_id())
    v_actual = [0.5 * (block.e[ref_idx, 0, 0, p].lb + block.e[ref_idx, 0, 0, p].ub) for p in block.periods]
    clone, rec = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v_actual)
    comp = clone.component(UB.NO_REVERSE_FLOW_ROW)
    deactivated = comp is not None and all(not c.active for c in comp.values())
    recorded = UB.NO_REVERSE_FLOW_ROW in rec['relaxed_row_families']
    ref_gen = network.get_reference_gen_idx()
    clone.pg[ref_gen, 0, 0, 3].set_value(-0.05, skip_validation=True)   # planted: 5 MW export at 100 MVA, hour 4
    tol = BENCH.CONSISTENCY_TOL
    viol = UB.consistency_violations(clone, rec, block, hard_tol=tol['hard_tol_pu2'],
                                     soft_excess_tol=tol['soft_excess_tol_pu2'], thermal_tol=tol['thermal_tol_pu2'])
    nrf_hits = [h for h in viol['hard'] if h.get('kind') == 'no_reverse_flow']
    planted_found = any(h['row'].endswith('[0,0,3]') for h in nrf_hits)
    del clone
    if nrf_expected:
        ok = deactivated and recorded and planted_found and viol['trigger_sequential_pass']
    else:
        ok = comp is None and not recorded and not nrf_hits
    return {'passed': bool(ok), 'nrf_expected': nrf_expected,
            'nrf_rows_deactivated_in_clone': deactivated, 'nrf_family_recorded_as_relaxed': recorded,
            'NEGATIVE_planted_reverse_flow_reported': planted_found, 'n_nrf_hits': len(nrf_hits),
            'planted_hit': next((h for h in nrf_hits if h['row'].endswith('[0,0,3]')), None),
            'trigger_sequential_pass': viol['trigger_sequential_pass'], 'block': UB.block_label('DSO', node_id, year, day)}


def _nrf_negative_controls(UB, planning, models, build, reference):
    """An undeclared NRF component, and a deactivated NRF row, must each fail the structural check."""
    out = {}
    first = next(iter(build['blocks']))
    undeclared = copy.deepcopy(build)
    undeclared['blocks'][first].pop('no_reverse_flow_rows_expected')
    try:
        UB.check_arm_structures(planning, {'dso': models}, reference, dso_build_record=undeclared)
        out['undeclared_nrf_rows_refused'] = False
    except UB.StructuralCheckError as error:
        out['undeclared_nrf_rows_refused'] = 'undeclared Constraint component' in str(error)
    node_id, year, day = first.split('|')[1], first.split('|')[2], first.split('|')[3]
    dn_key = next(k for k in models if str(k) == node_id)
    y_key = next(k for k in models[dn_key] if str(k) == year)
    d_key = next(k for k in models[dn_key][y_key] if str(k) == day)
    row = models[dn_key][y_key][d_key].component(UB.NO_REVERSE_FLOW_ROW)[0, 0, 0]
    row.deactivate()
    try:
        UB.check_arm_structures(planning, {'dso': models}, reference, dso_build_record=build)
        out['deactivated_nrf_row_refused'] = False
    except UB.StructuralCheckError as error:
        out['deactivated_nrf_row_refused'] = 'fixed rows counted 23 != declared 24' in str(error)
    row.activate()
    return out


def _load_module_from_git(commit, rel, name, expected_sha):
    blob = subprocess.run(['git', 'show', f'{commit}:{rel}'], cwd=REPO, capture_output=True, check=True).stdout
    sha = hashlib.sha256(blob).hexdigest()
    if sha != expected_sha:
        raise RuntimeError(f'{rel} at {commit}: sha256 {sha} != expected {expected_sha}')
    tmp = tempfile.mkdtemp(prefix='w116_')
    path = os.path.join(tmp, f'{name}.py')
    with open(path, 'wb') as handle:
        handle.write(blob)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module       # inspect.getsource of its classes resolves through sys.modules; removed after use
    spec.loader.exec_module(module)
    return module, sha, tmp          # the caller removes `tmp` after its inspect.getsource comparisons


TSO_PATH_FUNCTIONS = ('build_tso_arm_model', '_fixed_interface_p_rule', '_fixed_interface_q_rule',
                      '_fixed_interface_v_rule', '_tso_interface_vmag_targets_pu', 'set_tso_interface_targets',
                      'get_tso_interface_schedule', 'solve_tso_model', '_solve_block', 'read_ipopt_attempt_summary',
                      'check_arm_block_structure', 'block_structure', 'evaluate_common_q', 'common_q_gate',
                      'curtailment_report', 'interface_voltage_mismatch', 'tso_block_metrics', 'apply_start',
                      'apply_warm_values', 'apply_perturbation', 'get_dso_interface_schedule', 'dso_interface_voltage_pin',
                      'pin_dso_interface_voltage', 'reevaluate_dso_at_actual_voltage', 'interface_price_terms',
                      'coordinated_reference_structure', 'declared_solve_count', 'require_single_scenario')


def v3_tso_unchanged(UB, planning, candidate, reference, certified, dso_digests):
    old, old_sha, old_tmp = _load_module_from_git(V2_GIT_HEAD, 'uncoordinated_benchmark.py', '_ub_at_spec_v2',
                                                  V2_UB_SHA256)
    try:
        return _v3_compare(UB, old, old_sha, planning, candidate, reference, certified, dso_digests)
    finally:
        sys.modules.pop('_ub_at_spec_v2', None)
        shutil.rmtree(old_tmp)


def _v3_compare(UB, old, old_sha, planning, candidate, reference, certified, dso_digests):
    targets = UB.get_dso_interface_schedule(planning, certified['dso'])
    kw = dict(curtailment_penalty=BENCH.TIE_BREAKER['decision']['tso'], coupling='fixed_interface',
              pin_interface_voltage=False)
    new_model, new_build = UB.build_tso_arm_model(planning, candidate['total_capacity'], targets, **kw)
    old_model, old_build = old.build_tso_arm_model(planning, candidate['total_capacity'], targets, **kw)
    per_block, all_equal = {}, True
    for year in planning.transmission_network.years:
        for day in planning.transmission_network.days:
            a, b = model_digest(new_model[year][day]), model_digest(old_model[year][day])
            equal = a == b
            all_equal = all_equal and equal
            per_block[UB.block_label('TSO', None, year, day)] = {'equal': equal, 'counts': a['counts'],
                                                                 'differs_in': [k for k in a['sha256']
                                                                                if a['sha256'][k] != b['sha256'][k]]}
    struct_new = UB.check_arm_structures(planning, {'tso': new_model}, reference, tso_build_record=new_build,
                                         tso_coupling='fixed_interface')
    struct_old = UB.check_arm_structures(planning, {'tso': old_model}, reference, tso_build_record=old_build,
                                         tso_coupling='fixed_interface')
    records_equal = (json.dumps(new_build, sort_keys=True, default=str) == json.dumps(old_build, sort_keys=True,
                                                                                      default=str)
                     and json.dumps(struct_new, sort_keys=True, default=str)
                     == json.dumps(struct_old, sort_keys=True, default=str))
    # NEGATIVE: a planted bound change on one TSO Var is detected by the digest
    import pyomo.environ as pe
    first = next(iter(new_model.values()))
    blk = next(iter(first.values()))
    var = next(v for v in blk.component_data_objects(pe.Var, descend_into=True, sort=True)
               if not v.fixed and v.ub is not None)
    saved = var.ub
    var.setub(saved + 1.0)
    negative = model_digest(blk) != model_digest(next(iter(next(iter(old_model.values())).values())))
    var.setub(saved)
    sources = {name: inspect.getsource(getattr(UB, name)) == inspect.getsource(getattr(old, name))
               for name in TSO_PATH_FUNCTIONS}
    changed_functions = sorted(n for n, f in vars(UB).items() if callable(f) and getattr(f, '__module__', None)
                               == UB.__name__ and hasattr(old, n) and callable(getattr(old, n))
                               and inspect.getsource(f) != inspect.getsource(getattr(old, n)))
    added_names = sorted(n for n in vars(UB) if not n.startswith('__') and not hasattr(old, n))
    removed_names = sorted(n for n in vars(old) if not n.startswith('__') and not hasattr(UB, n))
    del new_model, old_model
    gc.collect()
    # the spec-v2 DSO arm path (default no_reverse_flow=False) is unchanged, and the NRF build = the v2 build + the rows
    dso = {}
    for arm in UB.DSO_ARMS:
        models, _build = old.build_dso_arm_models(planning, candidate['total_capacity'], arm=arm,
                                                  curtailment_penalty=BENCH.TIE_BREAKER['decision'][f'{arm}_dso'])
        old_digests = {label: model_digest(b) for label, b in ((UB.block_label('DSO', n, y, d), b)
                                                               for n, y, d, _dn, b in _blocks(models, planning))}
        del models
        gc.collect()
        dso[arm] = {
            'v2_type_new_equals_old_all_blocks': all(dso_digests[(arm, False)][k] == v for k, v in old_digests.items()),
            'nrf_minus_rows_equals_old_all_blocks': all(dso_digests[(arm, True)][k] == v for k, v in old_digests.items()),
            'n_blocks': len(old_digests)}
    passed = (all_equal and records_equal and negative and all(sources.values())
              and all(v['v2_type_new_equals_old_all_blocks'] and v['nrf_minus_rows_equals_old_all_blocks']
                      and v['n_blocks'] == 36 for v in dso.values())
              and set(changed_functions) <= {'build_dso_arm_models', 'check_arm_structures',
                                             'run_operational_planning_uncoordinated',
                                             'build_consistency_reevaluation_block', 'consistency_violations'}
              and set(added_names) <= {'NO_REVERSE_FLOW_ROW', '_no_reverse_flow_rule'} and not removed_names)
    return {'id': 'V3_tso_arm_unchanged_vs_v2', 'passed': bool(passed),
            'v2_code': {'commit': V2_GIT_HEAD, 'uncoordinated_benchmark_sha256': old_sha},
            'tso_per_block': per_block, 'tso_all_blocks_identical': all_equal,
            'build_and_structure_records_equal': records_equal, 'NEGATIVE_planted_bound_change_detected': negative,
            'tso_path_function_sources_identical': sources, 'functions_changed_vs_v2': changed_functions,
            'names_added_vs_v2': added_names, 'names_removed_vs_v2': removed_names,
            'dso_v2_path_unchanged': dso,
            'digest_definition': inspect.getdoc(model_digest)}


def v4_reverse_flow_count(UB, srp, planning, certified, out_dir):
    import pyomo.environ as pe
    from model_construction_helpers import sess_na_scenario
    count = NRF.reverse_flow_count(planning, {'dso': certified['dso'], 'tso': certified['tso']})
    # hand check: the most negative entries (or the smallest imports if there is no reverse entry) and the largest
    rows = []
    for node_id, year, day, dn, block in _blocks(certified['dso'], planning):
        for p in block.periods:
            rows.append((float(pe.value(block.pg_adn[0, 0, p])), node_id, year, day, p))
    rows.sort()
    picks = rows[:3] + rows[-1:]
    tn = planning.transmission_network
    adn = list(tn.active_distribution_network_nodes)
    hand = []
    for _v, node_id, year, day, p in picks:
        dn = planning.distribution_networks[node_id]
        network = dn.network[year][day]
        block = certified['dso'][node_id][year][day]
        ref_gen, ref_node = network.get_reference_gen_idx(), network.get_reference_node_id()
        s_m0, s_o0 = sess_na_scenario(block)
        pg = block.pg[ref_gen, 0, 0, p].value
        shared = sum(block.shared_es_pnet[e, s_m0, s_o0, p].value for e in block.shared_energy_storages
                     if network.shared_energy_storages[e].bus == ref_node)
        by_hand = (pg - shared) * network.baseMVA
        entry = next((e for e in count['reverse_entries_most_negative_first']
                      if (e['node_id'], e['year'], e['day'], e['period']) == (node_id, str(year), str(day), p)), None)
        via_fn = float(pe.value(block.pg_adn[0, 0, p])) * network.baseMVA
        t_block = certified['tso'][year][day]
        tnet = tn.network[year][day]
        dn_idx = adn.index(node_id)
        adn_load_idx = tnet.get_adn_load_idx(node_id)
        tso_by_hand = (t_block.pc[adn_load_idx, 0, 0, p].value + t_block.interface_delta_p[dn_idx, 0, 0, p].value) \
            * tnet.baseMVA
        hand.append({'node_id': node_id, 'year': str(year), 'day': str(day), 'hour': p + 1,
                     'pg_ref_gen_pu': pg, 'shared_ess_at_ref_pu': shared, 'p_int_by_hand_mw': by_hand,
                     'p_int_pg_adn_mw': via_fn, 'abs_diff_mw': abs(by_hand - via_fn),
                     'bitwise': by_hand == via_fn,
                     'in_listed_reverse_entries': entry is not None,
                     'listed_value_mw': None if entry is None else entry['p_int_mw'],
                     'dso_expected_interface_pf_p_mw': float(block.expected_interface_pf_p[p].value) * network.baseMVA,
                     'tso_pc_plus_delta_by_hand_mw': tso_by_hand,
                     'tso_minus_dso_mw': tso_by_hand - via_fn})
    hand_ok = all(h['abs_diff_mw'] <= 1e-9 and abs(h['dso_expected_interface_pf_p_mw'] - h['p_int_pg_adn_mw']) <= 1e-4
                  and (h['p_int_pg_adn_mw'] >= 0.0 or h['in_listed_reverse_entries']) for h in hand)
    # sign proof: TN energy balance per block and hour
    losses, only_adn = [], []
    for year in tn.years:
        for day in tn.days:
            t_block = certified['tso'][year][day]
            tnet = tn.network[year][day]
            base = tnet.baseMVA
            for p in t_block.periods:
                gen = sum(float(pe.value(t_block.pg[g, 0, 0, p])) for g in t_block.generators) * base
                adn_total = sum(float(pe.value(t_block.pc_adn[i, 0, 0, p])) for i in range(len(adn))) * base
                load_total = sum(float(pe.value(t_block.pc_node[i, 0, 0, p])) for i in t_block.nodes) * base
                losses.append({'block': UB.block_label('TSO', None, year, day), 'hour': p + 1, 'gen_mw': gen,
                               'adn_load_mw': adn_total, 'all_loads_mw': load_total, 'losses_mw': gen - load_total})
                only_adn.append(abs(load_total - adn_total))
    min_loss = min(r['losses_mw'] for r in losses)
    max_loss = max(r['losses_mw'] for r in losses)
    settle_src = inspect.getsource(__import__('model_construction_helpers').interface_energy_settlement)
    sign = {'tn_losses_min_mw': min_loss, 'tn_losses_max_mw': max_loss,
            'tn_losses_nonnegative_every_hour': min_loss >= -1e-6,
            'tn_only_loads_are_adn_interfaces_max_abs_mw': max(only_adn),
            'tn_only_loads_are_adn_interfaces': max(only_adn) <= 1e-6,
            'dso_settlement_pays_plus_pi_pg_adn_source': (
                'settlement += probability * c_p[p] * network.baseMVA * model.pg_adn[s_m, s_o, p]' in settle_src
                and '        return settlement' in settle_src and '        return -settlement' in settle_src),
            'consensus_same_sign_max_abs_dso_minus_tso_mw': count['tso_side_secondary']['max_abs_dso_minus_tso_mw'],
            'reading': ('TN generation minus the ADN interface loads is the TN\'s own losses (small, >= 0) at every '
                        'hour, and the TN has no other load: pc_adn > 0 is power the TN delivers to the DN. The DSO copy '
                        'pg_adn agrees with pc_adn to the consensus residual, so pg_adn > 0 is IMPORT (TN -> DN) and '
                        'pg_adn < 0 REVERSE flow -- the NRF rows bound the right sign'),
            'worst_5_hours_by_losses': sorted(losses, key=lambda r: r['losses_mw'])[:5]}
    sign_ok = (sign['tn_losses_nonnegative_every_hour'] and sign['tn_only_loads_are_adn_interfaces']
               and sign['dso_settlement_pays_plus_pi_pg_adn_source']
               and sign['consensus_same_sign_max_abs_dso_minus_tso_mw'] < 1.0)
    payload = {'stage': 'P5.15 W116 -- reverse-flow count of the coordinated Q181 solution (zero solves)',
               'utc': _utc(), 'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(THIS),
               'harness_sha256': _sha(HARNESS),
               'instance': {'problem': 'SRP1', 'label': 'x0', 'candidate_key': BENCH.X0['candidate_key'],
                            'cell': 'd110bd1a5977df1e_x0', 'certification_cycle': 181,
                            'certified_models': BENCH.COORDINATED['certified_models'],
                            'q181': BENCH.COORDINATED['certified_gross']},
               'objective_convention': 'no Q here; counts and energies only (MWh, stated weightings)',
               'count': count, 'hand_check': hand, 'sign_proof': sign,
               'statement': ('part of the measured benefit is the value of allowing reverse flow'
                             if count['totals']['strict']['count'] > 0 else
                             'no reverse-flow interface-hour in the coordinated solution')}
    path = os.path.join(out_dir, os.path.basename(NRF.REVERSE_FLOW_OUTPUT_REL))
    with open(path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    passed = hand_ok and sign_ok and count['n_entries'] == 864
    return {'id': 'V4_reverse_flow_count_q181', 'passed': bool(passed), 'output': os.path.relpath(path, REPO),
            'totals': count['totals'], 'per_node': count['per_node'], 'min_entry': count['min_entry'],
            'n_entries': count['n_entries'], 'tso_side_secondary': count['tso_side_secondary'],
            'n_entries_at_zero_within_material_tol': count['n_entries_at_zero_within_material_tol'],
            'hand_check_ok': hand_ok, 'hand_check': hand, 'sign_ok': sign_ok,
            'sign_proof': {k: v for k, v in sign.items() if k != 'worst_5_hours_by_losses'},
            'statement': payload['statement']}


class _FakeGuard:
    """A stand-in for SolveProfileGuard in the accounting unit test (counts moved by the test itself)."""

    def __init__(self):
        self.counts = {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}
        self.label = 'fake'

    def launch(self, n, exec_n=None):
        self.counts['permitted_solve'] += n
        self.counts['permitted_exec'] += n if exec_n is None else exec_n

    def verify(self, expected_solves, expected_execs=None):
        expected_execs = expected_solves if expected_execs is None else expected_execs
        out = []
        if self.counts['permitted_solve'] != expected_solves:
            out.append('solve')
        if self.counts['permitted_exec'] != expected_execs:
            out.append('exec')
        return out


def _accounting_unit_test(UB):
    saved = NRF._GUARD
    results = {}
    try:
        def fresh(bound=10):
            g = _FakeGuard()
            NRF._GUARD = g
            return g, NRF.BlockSolveAccount(g, bound)

        def solve_ok(g, n):
            def call():
                g.launch(n)
                return None, {'n_attempts': n}
            return call

        def solve_fail(g, n):
            def call():
                g.launch(n)
                raise UB.ArmSolveFailure('planted', {'n_attempts': n})
            return call

        g, acc = fresh()
        acc.network_solve(UB, solve_ok(g, 1), label='b1', kind='DSO')
        acc.network_solve(UB, solve_ok(g, 3), label='b2', kind='TSO')
        _rec, failure = acc.network_solve(UB, solve_fail(g, 3), label='b3', kind='TSO')

        def e3():
            g.launch(1)
            return {'n_attempts': 1}
        acc.elastic_solve(e3, label='b3')
        results['correct_sequence_passes'] = acc.cumulative == 8 and failure is not None and len(acc.ledger) == 4

        def expect_raise(fn):
            try:
                fn()
                return False
            except RuntimeError:
                return True

        g, acc = fresh()
        results['NEGATIVE_too_few_launches_raises'] = expect_raise(
            lambda: acc.network_solve(UB, lambda: (None, {'n_attempts': 1}), label='x', kind='DSO'))
        g, acc = fresh()

        def too_many():
            g.launch(2)
            return None, {'n_attempts': 1}
        results['NEGATIVE_too_many_launches_raises'] = expect_raise(
            lambda: acc.network_solve(UB, too_many, label='x', kind='DSO'))
        g, acc = fresh()
        results['NEGATIVE_attempts_above_three_raises'] = expect_raise(
            lambda: acc.network_solve(UB, solve_ok(g, 4), label='x', kind='TSO'))
        g, acc = fresh()

        def e3_two():
            g.launch(2)
            return {'n_attempts': 2}
        results['NEGATIVE_e3_not_one_raises'] = expect_raise(lambda: acc.elastic_solve(e3_two, label='x'))
        g, acc = fresh(bound=2)
        results['NEGATIVE_over_upper_bound_raises'] = expect_raise(
            lambda: acc.network_solve(UB, solve_ok(g, 3), label='x', kind='TSO'))
        g, acc = fresh()

        def exec_mismatch():
            g.launch(1, exec_n=2)
            return None, {'n_attempts': 1}
        results['NEGATIVE_exec_count_mismatch_raises'] = expect_raise(
            lambda: acc.network_solve(UB, exec_mismatch, label='x', kind='DSO'))
    finally:
        NRF._GUARD = saved
    return results


def v5_sweep_accounting(UB, planning, candidate, certified):
    import network as NW
    src = inspect.getsource(NW._run_smopf)
    n_attempt_sites = src.count('_run_smopf_solver_attempt(')
    tn = planning.transmission_network
    n_dso = len(planning.distribution_networks) * len(tn.years) * len(tn.days)
    n_tso = len(tn.years) * len(tn.days)
    upper = n_dso * NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE + n_tso * (NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE
                                                                  + NRF.E3_LAUNCHES_PER_FAILING_BLOCK)
    unit = _accounting_unit_test(UB)
    W, note = NRF._import_w114(expected_top_guard=W106._GUARD)     # stack: W116, W111, W106 (top), then W114
    # the E3 elastic copy on a freshly built (unconstrained-arm) TSO block, zero solves
    tso_model, _b = UB.build_tso_arm_model(planning, candidate['total_capacity'],
                                           UB.get_dso_interface_schedule(planning, certified['dso']),
                                           curtailment_penalty=BENCH.TIE_BREAKER['decision']['tso'],
                                           coupling=UB.TSO_COUPLING_FIXED,
                                           pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
    year, day = next(iter(tn.years)), next(iter(tn.days))
    network = tn.network[year][day]
    block = tso_model[year][day]
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_variables
    # E3's per-hour reading needs the TSO arm block separable in hours: no active row whose FREE variables span two
    # periods (the period is the last index of every period-indexed Var), on every TSO arm block
    coupling_rows = {}
    for y in tn.years:
        for dd in tn.days:
            for con in tso_model[y][dd].component_data_objects(pe.Constraint, active=True, descend_into=True):
                periods = {v.index()[-1] for v in identify_variables(con.body, include_fixed=False)
                           if isinstance(v.index(), tuple) and v.index() and isinstance(v.index()[-1], int)}
                if len(periods) > 1:
                    key = f'{y}|{dd}|{con.parent_component().name}'
                    coupling_rows[key] = coupling_rows.get(key, 0) + 1
    c, targets, bound_map, _cls = W.build_elastic_copy(block, 'rows', NRF.E3_ROWS)
    slack_block = c.component('_core_add_slack_variables')
    n_slack = sum(1 for _v in slack_block.component_data_objects(pe.Var, descend_into=True))
    slacks = W.read_slacks(c, targets, network, bound_map, allow_none=True)
    sched = {n: UB.get_dso_interface_schedule(planning, certified['dso'])[n][year][day]
             for n in tn.active_distribution_network_nodes}
    moves = W.interface_moves(c, network, sched)
    rows = NRF._hour_rows_from_e3('TSO|-|probe', year, day, moves, block.periods,
                                  NRF.SWEEP_ACCEPT_THRESHOLD_PU * network.baseMVA, True)
    e3_facts = {'target_components': [t.name for t in targets], 'n_slack_vars': n_slack,
                'n_target_rows': slacks['n_target_rows'], 'n_hour_rows': len(rows),
                'original_block_untouched': block.component('_core_add_slack_variables') is None,
                'tso_arm_rows_coupling_two_periods': coupling_rows,
                'tso_arm_blocks_separable_in_hours': not coupling_rows}
    del c, tso_model
    expected = expected_sweep_counts_from_w113()
    starts = start_dependence_from_w113()
    src_sweep = inspect.getsource(NRF.stage_sweep)
    wiring = {'dso_blocks_accounted': "account.network_solve(\n                    UB, lambda: UB._solve_block(" in src_sweep,
              'tso_blocks_accounted_and_continue': ("account.network_solve(\n                UB, lambda: UB._solve_block("
                                                    in src_sweep and 'if failure is None:' in src_sweep),
              'e3_accounted': 'account.elastic_solve(lambda: W.elastic_solve(' in src_sweep,
              'permitted_sites_include_w114_elastic_solve': (NRF.W114_SCRIPT, 'elastic_solve') in NRF.PERMITTED_SWEEP_SITES,
              'permitted_sites_include_solve_block': ('uncoordinated_benchmark.py', '_solve_block') in NRF.PERMITTED_SWEEP_SITES}
    w114_sha_committed = _git(['log', '-1', '--format=%H', '--', NRF.W114_SCRIPT])
    passed = (n_attempt_sites == NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE and upper == 156
              and upper == NRF.SRP1_DECLARED_V3['sweep_upper_bound_per_arm'] and all(unit.values())
              and note['guard_in_force_is_expected'] and note['e3_variant'] == ('rows', NRF.E3_ROWS)
              and e3_facts['target_components'] == ['uncoord_interface_p_fixed']
              and e3_facts['n_target_rows'] == 72 and e3_facts['n_slack_vars'] == 144 and e3_facts['n_hour_rows'] == 24
              and e3_facts['original_block_untouched'] and e3_facts['tso_arm_blocks_separable_in_hours']
              and all(wiring.values())
              and expected['passive']['expected_range'] == [51, 57]
              and expected['price_taker']['expected_range'] == [51, 84]
              and starts['start_independent_as_far_as_recorded'])
    return {'id': 'V5_sweep_solve_accounting', 'passed': bool(passed),
            'max_attempts_from_production_source': n_attempt_sites, 'upper_bound_per_arm': upper,
            'accounting_unit_test': unit, 'w114_import': note, 'w114_last_commit': w114_sha_committed,
            'e3_zero_solve_build': e3_facts, 'expected_from_w113_records': expected,
            'start_dependence_from_w113': starts, 'harness_wiring': wiring}


def v6_typing_test(out_dir):
    res = W106.c5_typing_test(out_dir)
    res['id'] = 'V6_w100_bool_typing_test'
    return res


# evaluation_key's configuration-level arguments and where a campaign spec keeps them (W106 / W111's reading); every
# other argument after (candidate_key_hex, overrides) is an ENTRY-level key of the same name
EVAL_KEY_CONFIG_ARGS = {'case_file_aa': 'case_file_anderson_acceleration', 'ess_ageing_baseline': 'ess_ageing_baseline',
                        'derived_instance': 'derived_instance', 'convergence_depth_tail': 'convergence_depth_tail'}


def _recompute_keys_signature(module, specs):
    """`evaluation_key` of `module` recomputed for every entry of every committed campaign spec with its FULL current
    argument list, read from the signature: the configuration-level arguments as W106/W111 read them, every other one
    from the entry by name (W118 added `settling_resettle` after W111 was written)."""
    params = list(inspect.signature(module.evaluation_key).parameters)[2:]
    n = n_equal = 0
    mismatch, errors = [], []
    for rel in specs:
        spec = json.load(open(_abs(rel)))
        cfg = spec.get('configuration') or {}
        for e in spec.get('candidates') or []:
            n += 1
            overrides = e.get('overrides') if 'overrides' in e else (cfg.get('overrides') or {})
            kw = {name: (cfg.get(EVAL_KEY_CONFIG_ARGS[name]) if name in EVAL_KEY_CONFIG_ARGS else e.get(name))
                  for name in params}
            try:
                key = module.evaluation_key(e['key'], overrides, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if key == H._entry_eval_key(e):
                n_equal += 1
            else:
                mismatch.append({'spec': rel, 'label': e.get('label'), 'recomputed': key[:16],
                                 'frozen': H._entry_eval_key(e)[:16]})
    return {'arguments': params, 'entries': n, 'recomputed_equals_frozen': n_equal, 'mismatches': mismatch[:20],
            'errors': errors[:20], 'holds': n > 0 and n_equal == n and not mismatch and not errors}


def v7_eval_keys(w106_c6):
    """Committed eval keys unchanged, under evaluation_key's full current signature, at HEAD and in the working tree.
    W111's C6 (argument list fixed at W111) is kept INFORMATIONAL: its misses must be exactly the committed entries
    carrying an argument it does not pass (W118's `settling_resettle`)."""
    specs = [p for p in _git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()]
    head_module, head_sha = W106._harness_from_git('HEAD')
    at_head = _recompute_keys_signature(head_module, specs)
    in_tree = _recompute_keys_signature(H, specs)
    try:
        w111 = W111.c6_eval_keys_all_arguments(w106_c6)
    except Exception as error:  # noqa: BLE001
        w111 = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
    with_resettle = sorted((rel, e.get('label'), H._entry_eval_key(e)[:16]) for rel in specs
                           for e in json.load(open(_abs(rel))).get('candidates') or []
                           if e.get('settling_resettle') is not None)
    w111_misses = sorted((m['spec'], m['label'], m['frozen'])
                         for m in ((w111.get('harness_at_head') or {}).get('mismatches') or []))
    resettle_set = set(with_resettle)
    w111_explained = w111.get('passed') is True or (
        bool(w111_misses) and set(w111_misses) <= resettle_set
        and len(w111_misses) == min(20, len(with_resettle))           # W111 lists at most 20 mismatches
        and not ((w111.get('harness_at_head') or {}).get('errors')))
    disk = _sha('p515_s44_campaign_harness.py')
    return {'id': 'V7_committed_eval_keys_unchanged',
            'passed': bool(at_head['holds'] and in_tree['holds'] and w111_explained),
            'committed_specs': len(specs), 'evaluation_key_signature': str(inspect.signature(H.evaluation_key)),
            'harness_at_head': {'sha256': head_sha, **at_head},
            'harness_working_tree': {'sha256': disk, 'differs_from_head': disk != head_sha,
                                     'git_status': _git(['status', '--porcelain', '--',
                                                         'p515_s44_campaign_harness.py']), **in_tree},
            'entries_with_settling_resettle': [list(x) for x in with_resettle],
            'w111_c6_informational': {'passed': w111.get('passed'), 'n_mismatches_listed': len(w111_misses),
                                      'misses_are_exactly_the_settling_resettle_entries': w111_explained},
            'w116_modifies_harness': False}


def v8_spec_binding():
    failures = NRF.frozen_spec_binding_failures()
    latest = NRF._latest_frozen_spec()
    negatives = {}
    if latest is not None:
        path, version, _hash8 = latest
        tmp = tempfile.mkdtemp(prefix='w116_spec_neg_')
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
            shutil.copyfile(path, os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_00000000.json'))
            negatives['name_hash_mismatch_refused'] = any('does not start with its name hash' in f
                                                          for f in NRF.frozen_spec_binding_failures(tmp))
            os.remove(os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_00000000.json'))
            shutil.copyfile(_abs(V2['path']), os.path.join(tmp, os.path.basename(V2['path'])))
            v2_fail = NRF.frozen_spec_binding_failures(tmp)
            negatives['v2_spec_refused_version_and_root'] = (any('version 2 < 3' in f for f in v2_fail)
                                                             and any('output_root' in f for f in v2_fail))
        finally:
            shutil.rmtree(tmp)
    passed = latest is not None and latest[1] == 3 and not failures and negatives and all(negatives.values())
    return {'id': 'V8_frozen_spec_v3_binds', 'passed': bool(passed),
            'spec': None if latest is None else {'path': os.path.relpath(latest[0], REPO), 'version': latest[1],
                                                 'sha256': H.sha256_file(latest[0])},
            'binding_failures': failures, 'NEGATIVE': negatives}


def v9_spec_diff():
    v2 = W111._load_verified_json(V2)
    latest = NRF._latest_frozen_spec()
    if latest is None or latest[1] != 3:
        return {'id': 'V9_v2_v3_key_diff', 'passed': False, 'reason': f'latest v3-root spec is {latest}'}
    with open(latest[0]) as handle:
        v3 = json.load(handle)
    diff = W111.key_diff(v2, v3)
    top = sorted({x['path'].split('.')[0].split('[')[0] for x in diff})
    v2_tree = BENCH.OUT_ROOT_REL
    checks = {
        'v2_sha256_unchanged': _sha(V2['path']) == V2['sha256'],
        'v2_file_clean_in_git': _git(['status', '--porcelain', '--', V2['path']]) == '',
        'v2_tree_tracked_files_clean': _git(['status', '--porcelain', '--untracked-files=no', '--', v2_tree]) == '',
        'v3_predecessor_is_v2': (v3.get('predecessor') or {}).get('sha256') == V2['sha256'],
        'changed_top_keys_within_declared_set': set(top) <= set(V3_CHANGED_OR_ADDED_TOP_KEYS),
        'carried_keys_identical': all(v2.get(k) == v3.get(k) for k in V2_KEYS_CARRIED_IDENTICAL),
        'every_v2_key_present_or_declared': all(k in v3 or k in V3_CHANGED_OR_ADDED_TOP_KEYS for k in v2),
        'v2_stage_outputs_read_only_sha_verified': all(_sha(e['path']) == e['sha256']
                                                       for e in NRF.V2_STAGE_OUTPUTS.values()),
        'v3_output_root_is_new_and_not_v2': v3.get('output_root') == OUT_ROOT_REL != v2_tree,
    }
    return {'id': 'V9_v2_v3_key_diff', 'passed': all(checks.values()), 'checks': checks,
            'v2': dict(V2), 'v3': {'path': os.path.relpath(latest[0], REPO), 'sha256': H.sha256_file(latest[0])},
            'n_paths': len(diff), 'changed_top_keys': top,
            'top_level': [x for x in diff if '.' not in x['path'] and '[' not in x['path']], 'diff': diff}


def v10_production_unchanged():
    harness = 'p515_s44_campaign_harness.py'
    files = sorted(set(NRF.FROZEN_SPEC_BOUND_FILES) | set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) - {harness})
    status = _git(['status', '--porcelain', '--'] + files)
    prod = sorted(set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) - {harness})
    last = {name: _git(['log', '-1', '--format=%h %s', '--', name])[:90] for name in ('shared_resources_planning.py',
                                                                                     'model_construction_helpers.py',
                                                                                     'network.py')}
    return {'id': 'V10_no_production_change', 'passed': status == '', 'git_status': status, 'files_checked': files,
            'production_files': prod, 'harness_git_status_reported': _git(['status', '--porcelain', '--', harness]),
            'last_commit_of_core_production_files': last}


def v11_report_capture(reverse_path):
    producers = {'stage_nrf_arm': inspect.getsource(NRF.stage_nrf_arm), 'stage_sweep': inspect.getsource(NRF.stage_sweep)}
    static = {}
    for quantity, (source, path) in NRF.REPORT_CAPTURE_PATHS_V3.items():
        producer = NRF.REPORT_PRODUCERS_V3[source]
        if producer in producers:
            static[quantity] = f"'{path[0]}'" in producers[producer]
    v2 = {name: W111._load_verified_json(entry) for name, entry in NRF.V2_STAGE_OUTPUTS.items()}
    with open(reverse_path) as handle:
        reverse = json.load(handle)
    dynamic = {}
    for quantity, (source, path) in NRF.REPORT_CAPTURE_PATHS_V3.items():
        if source in v2:
            dynamic[quantity] = BENCH._dig(v2[source], path)[0]
        elif source == 'reverse_flow_count_q181':
            dynamic[quantity] = BENCH._dig(reverse, path)[0]
    negative = NRF.report_capture_check({})
    v2_gate_pass = v2['common_q_gate'].get('status') == 'PASS'
    passed = (all(static.values()) and len(static) + len(dynamic) == len(NRF.REPORT_CAPTURE_PATHS_V3)
              and all(dynamic.values()) and negative['all_present'] is False
              and len(negative['absent']) == len(NRF.REPORT_CAPTURE_PATHS_V3) and v2_gate_pass)
    return {'id': 'V11_report_capture_paths', 'passed': bool(passed),
            'written_by_producer_stage_source': static, 'present_in_read_only_inputs': dynamic,
            'v2_common_q_gate_status': v2['common_q_gate'].get('status'),
            'NEGATIVE_empty_outputs': {'all_present': negative['all_present'], 'n_absent': len(negative['absent'])}}


def v12_stage_wiring():
    latest = NRF._latest_frozen_spec()
    with open(latest[0]) as handle:
        spec = json.load(handle)
    rows = {}
    for st in spec['stages_v3_addendum_57_order']:
        cmd = st['command']
        argv = cmd.split(f' -u {HARNESS} ', 1)[1].split(' > ', 1)[0].split()
        log = cmd.split(' > ', 1)[1].split(' ', 1)[0]
        try:
            args = NRF.parse_args(argv)
        except SystemExit as error:
            rows[st['run_id']] = {'argv': argv, 'ok': False, 'parse_error': f'SystemExit {error.code}'}
            continue
        rows[st['run_id']] = {'argv': argv, 'parsed_run_id': NRF._run_id(args),
                              'log': log, 'ok': (NRF._run_id(args) == st['run_id']
                                                 and log == f"{LAUNCH_LOGS_REL}/{st['run_id']}.log"
                                                 and cmd.startswith('set -o noclobber && ' + PY + ' -u ')
                                                 and cmd.endswith(' 2>&1'))}
    ids = set(rows)
    expected_ids = set(NRF.SWEEP_RUN_IDS + NRF.NRF_ARM_RUN_IDS + NRF.NRF_VARIANT_RUN_IDS + [NRF.REPORT_RUN_ID])
    main_src = inspect.getsource(NRF.main)
    permitted = {"'sweep': PERMITTED_SWEEP_SITES" in main_src, "'nrf-arm': PERMITTED_ARM_SITES" in main_src,
                 "'nrf-passive-tie-breaker': PERMITTED_ARM_SITES" in main_src, "'report': ()" in main_src}
    passed = all(r['ok'] for r in rows.values()) and ids == expected_ids and all(permitted)
    return {'id': 'V12_stage_wiring', 'passed': bool(passed), 'commands': rows,
            'run_ids_match_harness_lists': ids == expected_ids, 'permitted_sites_per_stage_in_main': all(permitted)}


def run_checks(suffix=''):
    out_dir = _abs(CHECKS_DIR_REL + suffix)
    if os.path.exists(out_dir):
        print(f'REFUSING: output exists (write-once): {out_dir}', flush=True)
        return 2
    os.makedirs(out_dir)
    results, timings, extra = [], {}, {}

    def record(fn, *args):
        t = time.time()
        try:
            res = fn(*args)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res = {'id': getattr(fn, '__name__', 'check'), 'passed': False,
                   'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        payload = res[0] if isinstance(res, tuple) else res
        timings[payload.get('id', 'check')] = time.time() - t
        _log(f"{payload.get('id')}: {'PASS' if payload.get('passed') is True else 'FAIL'}")
        results.append(payload)
        return res

    record(v0_preconditions)
    record(v1_model_hash)
    record(v8_spec_binding)
    record(v9_spec_diff)
    record(v10_production_unchanged)
    try:
        w106_c6 = W106.c6_eval_keys()
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        w106_c6 = {'id': 'C6_committed_eval_keys_unchanged', 'passed': False, 'error': f'{type(error).__name__}: {error}'}
    extra['C6_w106_as_written_informational'] = w106_c6
    record(v7_eval_keys, w106_c6)
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
    v2res = record(v2_nrf_rows, UB, srp, planning, candidate, reference, certified, warm)
    digests = v2res[1] if isinstance(v2res, tuple) else None
    if digests is not None:
        record(v3_tso_unchanged, UB, planning, candidate, reference, certified, digests)
    else:
        results.append({'id': 'V3_tso_arm_unchanged_vs_v2', 'passed': False, 'reason': 'V2 did not return digests'})
    del warm
    gc.collect()
    record(v5_sweep_accounting, UB, planning, candidate, certified)
    del certified
    gc.collect()
    record(v11_report_capture, os.path.join(out_dir, os.path.basename(NRF.REVERSE_FLOW_OUTPUT_REL)))
    record(v12_stage_wiring)
    record(v6_typing_test, out_dir)
    guards = {'w116': _GUARD.verify(0), 'w106_import': W106._GUARD.verify(0), 'w111_import': W111._GUARD.verify(0)}
    passed = all(r.get('passed') is True for r in results) and not any(guards.values())
    payload = {'stage': 'P5.15 W116 zero-solve checks (Addendum 57; benchmark spec v3)', 'utc': _utc(),
               'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(THIS), 'harness_sha256': _sha(HARNESS),
               'solve_profile_guard': {'permitted': [], 'verify_0': guards, 'counts_w116': dict(_GUARD.counts),
                                       'counts_w106_import': dict(W106._GUARD.counts),
                                       'counts_w111_import': dict(W111._GUARD.counts)},
               'passed': bool(passed), 'n_checks': len(results),
               'failed': [r.get('id') for r in results if r.get('passed') is not True],
               'timings_s': timings, 'results': results, 'output_suffix': suffix, **extra}
    with open(os.path.join(out_dir, 'w116_zero_solve_checks.json'), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for name in sorted(files):
            manifest[os.path.relpath(os.path.join(root, name), REPO)] = H.sha256_file(os.path.join(root, name))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    _log(f"W116 checks: {'ALL PASS' if passed else 'FAIL ' + str(payload['failed'])}; guards {guards}; counts "
         f'W116 {dict(_GUARD.counts)}')
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
        W106._GUARD.uninstall()        # LIFO: W111 installs its guard before importing W106 (W106 is on top)
        W111._GUARD.uninstall()
        _GUARD.uninstall()


if __name__ == '__main__':
    sys.exit(main())
