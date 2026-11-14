"""
P5.15 Addendum 39 (task W47) -- THE MULTI-SCENARIO PILOT: instance, launcher and freeze (Addendum 36; frozen spec
v22 `data/SRP1/Results/P515S52/frozen_s52_spec_v22_5d8df1e8.json`, `pilot`).

THE INSTANCE (`s52_pilot_2x2`). The paper's structure at 2 market x 2 operation scenarios: the 5 representative
years 2025 / 2028 / 2031 / 2034 / 2037 (each a 3-year block), the 4 days, NumMarketScenarios 2 and
num_operation_scenarios 2 on the TSO and every DSO -- derived by the scale harness's own machinery
(`p515_s44_scale_measurement.derive_case('paper', {num_market_scenarios: 2, num_operation_scenarios: 2})`, which
edits ONLY Years / NumMarketScenarios / num_operation_scenarios of data/SRP1/SRP1.json) and written ONCE to
`INSTANCE_CASE_REL`. Its sha256 must equal W45's pilot case (7ecff44a..., c8f4b4c7) and the combined scenario
checksum production's reader computes for it must equal W45's (53b4bea4...): both asserted at --freeze and --run.
Row 18 ACTIVE at alpha = 0.50 (the author's fixed pilot value, Addendum 39), no premium floor (Addendum 38: none
needed -- no non-positive mean hourly price). Baseline ageing (C2 + phi_cal 0.985 + soh_min 0.70, declared),
case-file AA (declared), the campaign cost file (SRP1_ESS.xlsx, e17bd588..., pinned) and the EUR 1M budget
(reported: I(x), B - I).

THE PATH. Every evaluation goes through the campaign harness (`p515_s44_campaign_harness`, W47 extension:
`configuration.derived_instance` + the entry option `interface_deviation_premium`) and hence through
`p515_g_g1_g4_admm_gates.run_admm_arm` -- NOT `run_s39_arm`, whose `assert_s39_capture_paths` hard-codes
`objective_scale_is_93635360` and the s39 spec hash. That is asserted at --freeze and --run (rule eleven,
`path_is_run_admm_arm`), not assumed. The fixed-sigma calibration assertion the pilot DOES pass through is
production's own (`_resolve_common_admm_objective_scale`, band [1/3, 3] of sigma_fixed = 93,635,360): its inputs
(W45's floor ratio 0.3606 and the paper reference 0.597) are frozen in the spec, and every evaluation records the
ACTUAL sigma_computed / sigma_fixed and the ratio (`sigma_calibration`), printed in a banner.

STAGES (`--stage`), one campaign root and one frozen spec each:
  pilot  `s52_pilot_nopersist`
                            x = 0 and the smallest node-7 4 h unit (0.25 MVA / 1.0 MWh, 2025), ONE wave at
                            concurrency 2, cap 500, 10 consecutive all-pass cycles; post-certification on BOTH:
                            the interval-hull polish only (no reference: no D evaluation of this instance exists;
                            gates (b)/(c) are None). The certified models are NOT persisted (W48 ruling below).
  repro  `s52_pilot_repro_nopersist`
                            the TWO-CYCLE BITWISE REPRODUCTION of the x = 0 evaluation: the same entry (same eval
                            key: candidate x configuration), cap 2, concurrency 1, no post-certification. Either
                            order is valid; run FIRST it is also a ~15-minute smoke of the whole child path on this
                            instance (derived-instance install, row 18, ageing read-back on the 5-year ESSO, the
                            capture hooks, the multi-scenario capture and the workbook) -- everything except the
                            post-certification step.
  --compare (after both): cycle_trajectory[:2] of the repro's x = 0 run against the pilot's x = 0 run, bitwise
                            (every field, type and value), plus the harness trajectory field table; write-once
                            output `COMPARE_ROOT_REL`.

RECORDED PER POINT (campaign_results.json `points`, `table`; objective convention on every table: Q = certified
GROSS operational cost, settlement-contracted part and voltage pin excluded, row 18 and the price-deviation
covariance INCLUDED -- `_get_operational_recourse_components`; salvage reported, excluded): status (non-certified
with cause), cycles, certification cycle, Q, the bar, the rule-ten terminal-step-to-threshold ratios, Boyd terminal
ratios, SIGMA (fixed, computed, ratio, band), the multi-scenario terminal summary (per-DSO max-over-blocks RMS
interface dispersion, aggregate E|d| and sum omega d^2, the row 18 charge and its read-back, the settlement split
and the covariance identity, the per-scenario cost reconciliation), the workbook, the hull polish gate, unrecovered
network failures and the certifying window, the solve reconciliation, ESS dispatch / SoH / floor, AA counts, wall
time, cycle time, peak RSS, per-cycle trajectory path + sha256; and for the unit: value = Q(0) - Q(x), value - I,
the resolution bar_x + bar_0, value per MWh against SRP1's (S2, 79b99b59: 259,427.77) and the R = 0.937 prediction.

W48 RE-FREEZE (Planner ruling, 2026-09-23). `persist_certified_models` is DROPPED from the pilot: it was the W47
Worker's addition, not in spec v22, and the model pickle is the terminal phase's largest cost (+5.1 GiB transient,
+2.3 GiB retained, measured) -- it is what made the memory preflight refuse (19.76 GiB available against 20 GiB
required). The terminal capture (multiscenario_terminal.json) and production's workbook already record what the
manuscript needs; if the models are ever wanted, a single evaluation can be re-run deterministically. With the
pickle gone the terminal-phase transient is the workbook's (+3.73 GiB measured), so the transient budget falls from
6 to 4 GiB: required = concurrency x 7 + 4 GiB (pilot 18 GiB, repro 11 GiB). Both specs are re-frozen under NEW
campaign ids (`s52_pilot_nopersist`, `s52_pilot_repro_nopersist`) because a campaign root is write-once; the W47
specs (`campaign_s52_pilot/campaign_spec_s52_pilot_b62dc2b5.json`,
`campaign_s52_pilot_repro/campaign_spec_s52_pilot_repro_444ee0d0.json`) are left unmodified, each root carries a
SUPERSEDED.md naming its successor, each successor records its predecessor (path + sha256) as
`extra.predecessor_spec`, and --run refuses a predecessor's sha256 explicitly (the A0 -> A0_c7 precedent, 99b81181).
The repro is re-frozen too: its spec pins this script's sha256, which the ruling changes. Eval keys are unchanged
(post-certification is not part of the key).

MODES (attached, both streams captured, never detached):
  --stage <s> --freeze                   ZERO SOLVES. Writes the instance file (write-once; bytes re-derived and
                                         compared if it exists), reads it with production's reader (checksum,
                                         dimensions, block weights), builds ONE TSO and ONE DSO block of SRP1, the
                                         pilot and the paper instance with production's `build_model` (per-block
                                         size), evaluates I(x) on the pilot's own Benders master (no solve), pins,
                                         rule eleven, predictions, `freeze_campaign_spec` + validation. Lock files
                                         are RECORDED, not gating (the freeze needs no lock and makes no solve).
                                         Requires --scratch (the planning reads' plots go there, outside the repo).
  --stage <s> --run --spec-sha256 <sha>  loads THAT spec (its root must hold only it), re-checks everything
                                         cheap (pins, instance hash, harness / case file / ESS params / script
                                         sha256), the memory preflight (refusing), takes the campaign lock,
                                         evaluates, writes campaign_results.json + campaign_manifest_sha256.json.
  --compare                              zero solves; see above.
The parent never solves: SolveProfileGuard(permitted=()) installed before any model import, verified at exactly 0.
Exit codes (--run): 0 every point certified and every capture clean; 2 a non-certified point (harness clean);
1 harness / guard / capture / read-back / precondition failure (a sigma-assertion failure is reported as such).

EXACT COMMANDS (repo root, canonical interpreter):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s52_pilot_campaign.py --stage pilot \\
      --freeze --scratch <scratch dir> > data/SRP1/Results/P515S52/campaign_s52_pilot_nopersist_freeze_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s52_pilot_campaign.py \\
      --stage pilot --run --spec-sha256 <sha> > data/SRP1/Results/P515S52/campaign_s52_pilot_nopersist_launch.log 2>&1
  and the same with `--stage repro` and `campaign_s52_pilot_repro_nopersist_{freeze_launch,launch}.log`; then
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s52_pilot_campaign.py --compare \\
      > data/SRP1/Results/P515S52/pilot_repro_compare_launch.log 2>&1
  One stage at a time, attached, alone, never detached.
"""

import argparse
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S52 pilot campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE_TEXT = ('P5.15 Addendum 39 (W47) -- the multi-scenario pilot (Addendum 36): paper years x 4 days x 2 market x 2 '
              'operation scenarios, row 18 at alpha = 0.50, baseline ageing, AA-on')
LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
_P52 = os.path.join('data', 'SRP1', 'Results', 'P515S52')
_P47 = os.path.join('data', 'SRP1', 'Results', 'P515S47')
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')

# ---- the instance ----------------------------------------------------------------------------------------------
INSTANCE_LABEL = 's52_pilot_2x2'
INSTANCE_DIR_REL = os.path.join(_P52, 'pilot_instance')
INSTANCE_CASE_REL = os.path.join(INSTANCE_DIR_REL, 'SRP1__s52_pilot_2x2.json')
INSTANCE_RECORD_REL = os.path.join(INSTANCE_DIR_REL, 'instance_record.json')
DERIVE_BASE = 'paper'
DERIVE_OVERRIDES = {'num_market_scenarios': 2, 'num_operation_scenarios': 2}
EXPECTED_YEARS = {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3}
EXPECTED_CASE_SHA256 = '7ecff44a874892187d1dd2e3d5ed0664a2f90a4bfa4cdb820a4abcbbd828f949'      # W45 r2 pilot case
EXPECTED_SCENARIO_CHECKSUM = '53b4bea4142001617282a08541079952564903b6d84db066f90d71a650b7d563'  # W45 r2
SOURCE_CASE_REL = os.path.join('data', 'SRP1', 'SRP1.json')

# ---- configuration ---------------------------------------------------------------------------------------------
ALPHA = 0.50
PREMIUM = {'alpha': ALPHA, 'floor': None}
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
ARM_LABEL = 's39_D'
YEAR = 2025
ACTIVE_NODES = (5, 7, 9)
UNIT_NODE, UNIT_S_MVA, UNIT_E_MWH = 7, 0.25, 1.0
BUDGET_EUR = 1e6
REQUIRED_CONSECUTIVE_CYCLES = 10
# W48 (Planner ruling): persistence DROPPED -- declared explicitly False so the spec states it.
POST_CERTIFICATION = {'persist_certified_models': False, 'hull_polish': True}
POST_CERTIFICATION_RATIONALE = (
    'hull polish on BOTH (spec v22 pilot.runs). The certified TSO/DSO models are NOT persisted (Planner ruling, '
    'W48): persistence was the W47 Worker\'s addition, not in spec v22; the model pickle costs a +5.1 GiB transient '
    'and +2.3 GiB retained per child (measured, p515_s52_pilot_checks r1 D3) and is what made the memory preflight '
    'refuse (19.76 GiB available against 20 GiB required); the terminal capture (multiscenario_terminal.json) and '
    'production\'s operational workbook already record what the manuscript needs; if the models are ever wanted, a '
    'single evaluation can be re-run deterministically. No reference: no D evaluation of this instance exists, so '
    'post-certification gates (b)/(c) are None by construction.')
POST_CERTIFICATION_RATIONALE_W47_SUPERSEDED = (
    'hull polish on BOTH (spec v22 pilot.runs). The certified TSO/DSO models are ALSO persisted (before the polish '
    'mutates them): a pilot evaluation costs 7-10 h, and every manuscript quantity must stay recomputable without '
    're-running it (CLAUDE.md rule eleven incident: a quantity the spec required was unrecoverable because the '
    'models were not serialized). Not committed (about 1 GB each, ~6.6 x the 166 MB SRP1 pickle); hash-recorded in '
    'the child and campaign manifests. No reference: no D evaluation of this instance exists, so post-certification '
    'gates (b)/(c) are None by construction.')
STAGES = {
    'pilot': {'campaign_id': 's52_pilot_nopersist', 'cap': 500, 'concurrency': 2,
              'points': ('x0', 'n7_4h_e1'), 'post_certification': dict(POST_CERTIFICATION),
              'description': ('P5.15 Addendum 36 pilot: x = 0 and the node-7 4 h unit (0.25 / 1.0, 2025), one wave '
                              'at concurrency 2, cap 500, 10-cycle bar, hull polish on both (no model persistence, '
                              'W48)')},
    'repro': {'campaign_id': 's52_pilot_repro_nopersist', 'cap': 2, 'concurrency': 1,
              'points': ('x0',), 'post_certification': None,
              'description': ('P5.15 Addendum 36 pilot, two-cycle bitwise reproduction of the x = 0 evaluation '
                              '(same eval key, cap 2, concurrency 1, no post-certification)')},
}
COMPARE_ROOT_REL = os.path.join(_P52, 'pilot_repro_compare')
# W48: the W47 specs, frozen at e52c8135 and never run; recorded in each successor, refused by --run.
PREDECESSOR_SPECS = {
    'pilot': {'path': os.path.join(_P52, 'campaign_s52_pilot', 'campaign_spec_s52_pilot_b62dc2b5.json'),
              'sha256': 'b62dc2b5cd57eacdfd267d583f09e2d61b646006df09836c26d57f71e5499523',
              'campaign_id': 's52_pilot', 'frozen_at_commit': 'e52c8135', 'launcher_commit': '298e58f0',
              'reason': ('superseded before any run (W48, Planner ruling): persist_certified_models dropped; memory '
                         'transient budget 6 -> 4 GiB (required 20 -> 18 GiB)')},
    'repro': {'path': os.path.join(_P52, 'campaign_s52_pilot_repro', 'campaign_spec_s52_pilot_repro_444ee0d0.json'),
              'sha256': '444ee0d014dbeb63b73e268931e7597ec9fb18a43d6301e30e12b660f85391a7',
              'campaign_id': 's52_pilot_repro', 'frozen_at_commit': 'e52c8135', 'launcher_commit': '298e58f0',
              'reason': ('superseded before any run (W48): re-frozen with the pilot because the spec pins this '
                         "launcher's sha256, which the persistence ruling changes; its own configuration is unchanged "
                         '(no post-certification), memory required 13 -> 11 GiB')},
}

# ---- pins ------------------------------------------------------------------------------------------------------
SPEC_V22 = {'path': os.path.join(_P52, 'frozen_s52_spec_v22_5d8df1e8.json'),
            'sha256': '5d8df1e8fe7f3741a3e03167937881ac85c47fd713fe6aa7b938e3113106fe1c'}
ESS_PARAMS_FILE = {'path': H.ESS_PARAMS_FILE_REL,
                   'sha256': '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706',
                   'commit': '2466401d', 'note': 'the ageing BASELINE edit of W21 (Addendum 30)'}
COST_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             'sha256': 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'}
SIGMA_CHECK = {'path': os.path.join(_P52, 'sigma_check', 'r2', 'sigma_check.json'), 'commit': 'c8f4b4c7'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b'}
SRP1_S2_RESULTS = {'path': os.path.join(_P47, 'campaign_s47_recert', 'campaign_results.json'),
                   'sha256': 'e48dad43470829ee1d936cdab5e6433cffd688f9439e86e011b84dad74e19a3e',
                   'commit': '79b99b59', 'label': 'n7_4h_e1'}
CHECKS_RESULT = {'path': os.path.join(_P52, 'pilot_checks', 'r1', 'pilot_checks.json'),
                 'script': 'p515_s52_pilot_checks.py', 'expect': 'all_ok true'}
COMMITTED_BLOCK_COUNTS = {
    'srp1': os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement', 'srp1_cycle_snapoff_r1',
                         'blocks_build.jsonl'),
    'paper': os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement', 'paper_build',
                          'blocks_build.jsonl'),
}
SIZE_PROBE_BLOCKS = (('TSO', None, '2025', 'Spring'), ('DSO', 7, '2025', 'Spring'))
SRP1_VALUE_PER_MWH_EXPECTED = 259427.76775527
SRP1_RESOLUTION_EXPECTED = 34734.63290083408
R_PREDICTION = 0.937

OBJECTIVE_CONVENTION = (
    'Q(x) = certified_cost = gross_operational_cost (_get_operational_recourse_components: the CONTRACTED interface '
    'settlement and the solver-only interface-voltage pin excluded; the row 18 charge and the price-deviation '
    'covariance -- the settlement DEVIATION part -- included, Addendum 38 (C)); value = Q(0) - Q(x); F = I + Q; '
    'terminal salvage and net_operational_recourse = gross - salvage reported, excluded.')
VALUE_DEFINITION = (
    'value = Q(0) - Q(x_unit) on THIS instance (both evaluations of this campaign); resolution = bar_x + bar_0 '
    '(record.bar of each: the max gross step over the last 10 cycles); |value| or |value - I| <= resolution is '
    'INDETERMINATE (CLAUDE.md: the bar bounds stopping slack only). value per MWh = value / E; compared with '
    "SRP1's value per MWh for the same unit (S2 79b99b59: 259,427.77, resolution 34,734.63) as a ratio, and that "
    'ratio against R = 0.937 (Addendum 31 prediction on the market spread; the storage prices at the bus-7 '
    'marginal cost, so a deviation either way is informative -- spec v22 pilot prediction).')
BUDGET_CONVENTION = f'budget_slack_eur = B - I(x) at B = {BUDGET_EUR:g} EUR (the case file budget), REPORTED only'
DISPERSION_CONVENTION = (
    "the Planner's ruling from the alpha sweep: per DSO, the MAX over (year, day) blocks of the block's RMS "
    'interface-P dispersion (MW), and aggregates E|d| (MWh) and sum omega d^2 (MW^2 h) -- d_{s,t} = p_int_{s,t} - '
    "pbar_t against the block's OWN committed schedule; block-local, probability-weighted; reported both summed "
    'over blocks UNWEIGHTED (the W44 convention) and weighted by _get_admm_block_weight (the Q weighting). The '
    'premium convention stated with it: unpriced UPWARD flexibility makes holding the schedule cheaper than '
    'cost_flex suggests (spec v22 ruling2_alpha.manuscript_convention).')
STOP_RULE = ('as a1a (spec v15 execution.barrier), for form: STOP if 2 or more non-certified points fall in ONE '
             'region (node + duration ladder) or 3 or more overall. The pilot is ONE wave of two points in two '
             'regions (x0_y2025, n7_4h_y2025), so the rule cannot fire; spec v22 orders STOP FOR REVIEW after the '
             'pilot regardless.')

# ---- memory ----------------------------------------------------------------------------------------------------
GIB = 1 << 30
MEMORY_PER_CHILD_SUSTAINED_BYTES = 7 * GIB
MEMORY_TERMINAL_TRANSIENT_BYTES = 4 * GIB   # W48: 6 -> 4 GiB (no model pickle; the workbook's +3.73 GiB)
MEMORY_TERMINAL_TRANSIENT_BYTES_W47_SUPERSEDED = 6 * GIB
MEMORY_RULE_TEMPLATE = ('hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {concurrency} x '
                        '{per_child:g} GiB + {transient:g} GiB (one terminal-phase transient at a time: the harness '
                        'terminal-phase lock)')
MEMORY_RULE_RATIONALE = ('non-reclaimable load = wired + anonymous + compressor-occupied pages; file-backed pages '
                         '(active or inactive) are cache the kernel reclaims on demand, so free + inactive '
                         'under-counts what new processes can obtain (free + inactive recorded alongside); the '
                         'measure of A0 / S47, with the budget re-derived for this instance')
MEMORY_BUDGET_DERIVATION = (
    'MEASURED zero-solve on the pilot instance (p515_s52_pilot_checks, memory_by_stage): the 80 ADMM-built network '
    'blocks + 3 ESSO models of the unit candidate hold 3.40 GiB RSS (no pristine clones, no solver state); the '
    'multi-scenario capture adds ~0; production\'s workbook writer a +3.75 GiB TRANSIENT for ~2 min (released by '
    'gc.collect() to +0.55 GiB); the certified-model pickle (0.66 GB on disk) a +5.1 GiB TRANSIENT and +2.3 GiB '
    'retained. NOT measured (need solves): the pristine snapshot clones (lightweight, the default: every TSO block and '
    'every node-7 DSO block, ~1.3 GiB estimated), solver state (multiplier suffixes, SolverResults, AA memory) and the '
    'fixed overhead (SRP1 campaign children peak at 2.32-2.40 GiB for ~0.6 GiB of models). Sustained per child: '
    '~5.5-6.5 GiB -> 7 GiB; the terminal-phase transient -> 6 GiB, counted ONCE because the harness serializes the '
    'children\'s terminal phases (`acquire_terminal_phase_lock` on the campaign root). Without the model pickle the '
    'transient would be the workbook\'s ~4 GiB. '
    'W48 (Planner ruling: persist_certified_models dropped): the terminal-phase transient is now the workbook\'s -- '
    'measured RSS 3,730,014,208 -> peak 7,732,314,112 bytes (+3.727 GiB), retained +0.537 GiB after it -- budgeted '
    'at 4 GiB (was 6 GiB for the pickle); the +2.3 GiB the pickle retained is gone too. Per-child sustained budget '
    'unchanged at 7 GiB. Required: pilot 2 x 7 + 4 = 18 GiB (was 20), repro 1 x 7 + 4 = 11 GiB (was 13). Still '
    'NOT measured: the hull polish\'s own solve-time memory (it re-solves the 80 blocks after the workbook).')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 36 (the pilot) and Addendum 39 (alpha = 0.50 fixed; pilot first)',
    'data/SRP1/Results/P515S52/frozen_s52_spec_v22_5d8df1e8.json pilot',
    'Planner task W47 (instance, harness, freeze; zero solves)',
    'Planner task W48 (ruling 1: persist_certified_models dropped, both specs re-frozen; ruling 2: SRP1 bitwise '
    'gate e75a575e PASS)',
    '700cf13c (row 18 merged), gates 4ce7447b / c9b39bf5 / 59476bff; c8f4b4c7 (W45 sigma check)',
]
SCRIPT_NAME = os.path.basename(__file__)
EXTRA_CLEAN_FILES = (SCRIPT_NAME, CHECKS_RESULT['script'], 'p515_s44_scale_measurement.py', H.ESS_PARAMS_FILE_REL,
                     COST_FILE['path'], SOURCE_CASE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py', 'p513_solve_profile_guard.py')
LOCK_FAILURE_PREFIXES = ('legacy one-run lock exists', 'campaign lock exists')


# ======================================================================================================================
#  small utilities
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _banner(lines):
    _log('!' * 100)
    for line in lines:
        _log(f'!!! {line}')
    _log('!' * 100)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _git_state(rel):
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
    return tracked, not dirty


def _commit_in_head(commit):
    return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO,
                          capture_output=True).returncode == 0


def _nodes_full(partial):
    nodes = {n: (0.0, 0.0) for n in ACTIVE_NODES}
    for node, (s_val, e_val) in partial.items():
        nodes[int(node)] = (float(s_val), float(e_val))
    return nodes


POINT_NODES = {'x0': _nodes_full({}), 'n7_4h_e1': _nodes_full({UNIT_NODE: (UNIT_S_MVA, UNIT_E_MWH)})}


def _key_of(nodes):
    return H.candidate_key(H.canonical_candidate(nodes, investment_year=YEAR))


def _eval_key(nodes, derived):
    return H.evaluation_key(_key_of(nodes), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
                            derived_instance=derived, interface_deviation_premium=PREMIUM)


def _region_key(canonical):
    nz = {int(n): v for n, v in canonical['nodes'].items() if float(v[0]) or float(v[1])}
    if not nz:
        return f"x0_y{canonical['investment_year']}"
    return '+'.join(f'n{n}_{float(nz[n][1]) / float(nz[n][0]):g}h' for n in sorted(nz)) + \
        f"_y{canonical['investment_year']}"


def campaign_root(stage):
    return os.path.join(REPO, _P52, f'campaign_{STAGES[stage]["campaign_id"]}')


def memory_required_bytes(concurrency):
    return concurrency * MEMORY_PER_CHILD_SUSTAINED_BYTES + MEMORY_TERMINAL_TRANSIENT_BYTES


def memory_rule(concurrency):
    return MEMORY_RULE_TEMPLATE.format(concurrency=concurrency, per_child=MEMORY_PER_CHILD_SUSTAINED_BYTES / GIB,
                                       transient=MEMORY_TERMINAL_TRANSIENT_BYTES / GIB)


# ======================================================================================================================
#  pins (zero solves)
# ======================================================================================================================
def check_pins():
    out, failures = {}, []
    for name, pin in (('spec_v22', SPEC_V22), ('ess_params_file', ESS_PARAMS_FILE), ('cost_file', COST_FILE),
                      ('investment_cost_results', INVESTMENT_COST_RESULTS), ('srp1_s2_results', SRP1_S2_RESULTS)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked, clean = _git_state(pin['path'])
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': clean}
        if not (got == pin['sha256'] and tracked and clean):
            failures.append(f'pin {name}: {out[name]}')
    for name, pin in (('sigma_check', SIGMA_CHECK),):
        path = os.path.join(REPO, pin['path'])
        tracked, clean = _git_state(pin['path'])
        out[name] = {'path': pin['path'], 'commit': pin['commit'], 'commit_in_HEAD': _commit_in_head(pin['commit']),
                     'git_tracked': tracked, 'git_clean': clean,
                     'sha256': H.sha256_file(path) if os.path.isfile(path) else None}
        if not (os.path.isfile(path) and tracked and clean and out[name]['commit_in_HEAD']):
            failures.append(f'pin {name}: {out[name]}')
    path = os.path.join(REPO, CHECKS_RESULT['path'])
    tracked, clean = _git_state(CHECKS_RESULT['path'])
    payload = _load(CHECKS_RESULT['path']) if os.path.isfile(path) else {}
    out['pilot_checks'] = {'path': CHECKS_RESULT['path'], 'git_tracked': tracked, 'git_clean': clean,
                           'sha256': H.sha256_file(path) if os.path.isfile(path) else None,
                           'all_ok': payload.get('all_ok'), 'failing': payload.get('failing_checks'),
                           'script_sha256_recorded': payload.get('script_sha256'),
                           'harness_sha256_recorded': payload.get('harness_sha256'),
                           'harness_sha256_now': H.sha256_file(H.HARNESS_PATH),
                           'expect': CHECKS_RESULT['expect']}
    if not (os.path.isfile(path) and tracked and clean and payload.get('all_ok') is True
            and payload.get('harness_sha256') == out['pilot_checks']['harness_sha256_now']):
        failures.append(f"pin pilot_checks (the committed zero-solve checks must pass on THIS harness): "
                        f"{out['pilot_checks']}")
    return out, failures


def sigma_inputs():
    """W45's committed sigma-check inputs (c8f4b4c7), frozen into the spec: the prediction the run will be read
    against. The run's own sigma is recorded by every evaluation (`sigma_calibration`)."""
    d = _load(SIGMA_CHECK['path'])
    pred = d['prediction']
    pilot = d['instances']['pilot_2x2']
    return {'source': dict(SIGMA_CHECK), 'sha256': H.sha256_file(os.path.join(REPO, SIGMA_CHECK['path'])),
            'sigma_definition': d['sigma_definition'],
            'sigma_fixed_case_file': pilot['admm_objective_scale'],
            'assert_factor': pilot['admm_objective_scale_assert_factor'], 'band': pred['band'],
            'formula_floor': pred['formula_floor'], 'sigma_floor': pred['sigma_floor'],
            'ratio_floor': pred['ratio_floor'], 'ratio_floor_over_lower_band_edge': pred['ratio_floor_over_lower_band_edge'],
            'paper_5x5_reference_ratio': pred['paper_5x5_reference']['ratio'],
            'gate_2x2_sigma_exact': pred['gate_sigma_exact'],
            'w45_pilot_case_sha256': pilot['case_sha256'], 'w45_pilot_scenario_checksum': pilot['combined_scenario_checksum'],
            'median_block_weight_production': pilot['median_block_weight_production'],
            'prediction_recorded_before_run': ('sigma ratio in [0.36, 0.60] against the band [1/3, 3] (W45: floor '
                                               '0.3606 = 1.082 x the lower edge, i.e. an 8 % worst-case margin; the '
                                               'paper 5x5 reference 0.597). The pilot passes the assertion.')}


# ======================================================================================================================
#  the instance (zero solves)
# ======================================================================================================================
def derive_instance_text():
    import p515_s44_scale_measurement as S
    case, spec, changes = S.derive_case(DERIVE_BASE, dict(DERIVE_OVERRIDES))
    # the writer the scale harness, the W37 smoke, the 2x2 gate and W45 all use
    text = json.dumps(case, indent='\t')
    return text, case, spec, changes


def ensure_instance_file(write):
    """Writes the derived case once (or re-derives and compares bytes if it exists). Returns the evidence."""
    text, case, spec, changes = derive_instance_text()
    path = os.path.join(REPO, INSTANCE_CASE_REL)
    existed = os.path.exists(path)
    if existed:
        with open(path) as handle:
            on_disk = handle.read()
        if on_disk != text:
            raise RuntimeError(f'{INSTANCE_CASE_REL} exists and differs from the re-derived case (write-once)')
    elif write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'x') as handle:
            handle.write(text)
    else:
        raise RuntimeError(f'{INSTANCE_CASE_REL} is missing')
    sha = H.sha256_file(path)
    return {'case_path': INSTANCE_CASE_REL, 'case_sha256': sha, 'existed_before': existed,
            'equals_w45_pilot_case_sha256': sha == EXPECTED_CASE_SHA256, 'derive_base': DERIVE_BASE,
            'derive_overrides': dict(DERIVE_OVERRIDES), 'changes_vs_source': changes, 'years': case['Years'],
            'years_expected': case['Years'] == EXPECTED_YEARS, 'source_case_path': SOURCE_CASE_REL,
            'source_case_sha256': H.sha256_file(os.path.join(REPO, SOURCE_CASE_REL)),
            'derive_description': spec.get('description')}


def derived_declaration(instance, checksum):
    return H.validate_derived_instance({
        'instance_label': INSTANCE_LABEL, 'case_path': instance['case_path'], 'case_sha256': instance['case_sha256'],
        'scenario_checksum': checksum, 'source_case_path': instance['source_case_path'],
        'source_case_sha256': instance['source_case_sha256'], 'changes_vs_source': instance['changes_vs_source']})


def _read(case_rel, scratch, tag):
    """Production's reader on a case file given REPO-relative (the scale harness's `read_planning_from_derived_case`,
    which resolves it relative to data/SRP1), the read's plots / results / logs redirected into a fresh directory
    under `scratch` (outside the repository)."""
    import p515_s44_scale_measurement as S
    out_dir = tempfile.mkdtemp(prefix=f'w47_{tag}_', dir=scratch)
    planning = S.read_planning_from_derived_case({'derived_case': {'path': case_rel}}, out_dir, H._NoStageLog())
    return planning, out_dir


def _scratch_case(case, scratch, tag):
    """A derived case written for a READ only (the SRP1 / paper size probes) -- to scratch, never into the
    repository; returned REPO-relative (the reader resolves it relative to data/SRP1)."""
    path = os.path.join(tempfile.mkdtemp(prefix=f'w47_case_{tag}_', dir=scratch), f'SRP1__{tag}.json')
    with open(path, 'w') as handle:
        json.dump(case, handle, indent='\t')
    return os.path.relpath(path, REPO), H.sha256_file(path)


def _block_size(planning, kind, node, year, day):
    import pyomo.environ as pe
    import p515_s44_scale_measurement as S
    holder = planning.transmission_network if kind == 'TSO' else planning.distribution_networks[node]
    year_key = next(y for y in holder.years if str(y) == year)
    day_key = next(d for d in holder.days if str(d) == day)
    network = holder.network[year_key][day_key]
    t0 = time.time()
    model = network.build_model(holder.params)
    wall = time.time() - t0
    out = {'agent': kind, 'node': node, 'network': network.name, 'year': year, 'day': day,
           'scenario_combinations': len(model.scenarios_market) * len(model.scenarios_operation),
           'len_counts_at_build_model': S.len_counts(model, pe), 'deep_counts_at_build_model': S.deep_counts(model, pe),
           'build_model_wall_s': wall}
    del model
    return out


def _committed_block_counts(which, kind, node, year, day):
    path = os.path.join(REPO, COMMITTED_BLOCK_COUNTS[which])
    with open(path) as handle:
        for line in handle:
            r = json.loads(line)
            if r['agent'] == kind and r['node'] == node and str(r['year']) == year and r['day'] == day:
                return {'source': COMMITTED_BLOCK_COUNTS[which], 'sha256': H.sha256_file(path),
                        'code': 'pre-Addendum-38 (the scale measurement, P5.15 Addendum 25/27)',
                        'scenario_combinations': r['scenario_combinations'],
                        'len_counts_at_build_model': r['len_counts_at_build_model']}
    return None


def instance_facts(scratch, instance):
    """ZERO SOLVES: read the pilot instance with production's reader; its checksum, dimensions, block weights and
    solves per cycle; the per-block size of SRP1 / the pilot / the paper under the CURRENT code (one TSO and one
    node-7 DSO block each, production `build_model`) beside the committed pre-Addendum-38 counts; I(x) on the
    pilot's own Benders master (no solve) against the SRP1 W2 table."""
    import p56a_oracle as O
    import p515_s44_scale_measurement as S
    import shared_resources_planning as srp
    import pyomo.environ as pe
    facts = {}
    planning, read_dir = _read(instance['case_path'], scratch, 'pilot')
    checksum = planning.scenario_metadata['combined_scenario_checksum']
    facts['scenario_checksum'] = checksum
    facts['scenario_checksum_equals_w45'] = checksum == EXPECTED_SCENARIO_CHECKSUM
    facts['planning_read_redirected_to'] = read_dir
    facts['planning_dimensions'] = S.planning_dimensions(planning)
    facts['expected_block_counts'] = S.expected_block_counts(planning)
    facts['declared_solve_profile_per_cycle'] = S.declared_solve_profile(planning, 1)['solves_per_cycle']
    tn = planning.transmission_network
    facts['block_weights_tso'] = {f'{y}|{d}': srp._get_admm_block_weight(tn, y, d) for y in tn.years for d in tn.days}
    facts['median_block_weight_production'] = srp._compute_median_admm_block_weight(planning)
    admm = planning.params.admm
    facts['case_file_admm'] = {'objective_scale': admm.objective_scale, 'objective_scale_source': admm.objective_scale_source,
                               'objective_scale_assert_factor': admm.objective_scale_assert_factor,
                               'interface_deviation_premium_in_case_file': dict(admm.interface_deviation_premium),
                               'anderson_acceleration': dict(admm.anderson_acceleration),
                               'tso_snapshot_capture_mode': admm.tso_snapshot_capture_mode,
                               'dso_snapshot_capture_mode': admm.dso_snapshot_capture_mode,
                               'parallel_execution': planning.parallel_execution}
    # premium floor: Addendum 38 -- a floor only if some hour's mean price is non-positive
    import model_construction_helpers as MCH
    min_pibar = min(MCH.expected_market_price(tn.network[y][d], p) for y in tn.years for d in tn.days
                    for p in range(planning.num_instants))
    facts['min_hourly_mean_price'] = float(min_pibar)
    facts['premium_floor_needed'] = bool(min_pibar <= 0.0)
    # I(x) on the pilot's own master (production expression, no solve), and the p56a transcription
    sed = planning.shared_ess_data
    master = sed.build_master_problem()
    i_x = {}
    for label, nodes in POINT_NODES.items():
        x = {(n, y): {'s': 0.0, 'e': 0.0} for n in sed.active_distribution_network_nodes for y in sed.years}
        for n, (s_val, e_val) in nodes.items():
            if s_val or e_val:
                ykey = next(y for y in sed.years if int(y) == YEAR)
                x[(n, ykey)] = {'s': s_val, 'e': e_val}
        cand = O.vector_to_candidate(planning, x)
        sed.load_candidate_solution_into_master_model(master, cand)
        i_master = float(pe.value(master.investment_cost))
        i_oracle = O.investment_cost(planning, cand)
        i_x[label] = {'candidate_key': _key_of(nodes), 'I_x_eur_master_expression': i_master,
                      'I_x_eur_p56a_transcription': i_oracle, 'abs_diff': abs(i_master - i_oracle),
                      'budget_eur': BUDGET_EUR, 'budget_slack_eur': BUDGET_EUR - i_master,
                      'case_file_budget_eur': sed.params.budget}
    w2 = _load(INVESTMENT_COST_RESULTS['path'])['candidates']
    unit_key = _key_of(POINT_NODES['n7_4h_e1'])
    w2_vals = sorted({c.get('I_new_eur') for c in w2.values() if c.get('candidate_key') == unit_key}, key=repr)
    i_x['n7_4h_e1']['srp1_w2_table_I_eur'] = w2_vals[0] if len(w2_vals) == 1 else None
    i_x['n7_4h_e1']['equals_srp1_w2_table'] = (len(w2_vals) == 1 and w2_vals[0] == i_x['n7_4h_e1']['I_x_eur_master_expression'])
    facts['investment_cost'] = i_x
    facts['investment_cost_note'] = ('I(x) of THIS instance, production master expression (discount to the first '
                                     'year, 2025, and the ESS workbook scenario weights); the SRP1 W2 value is '
                                     'recorded beside it for comparison')
    # per-block size, current code
    sizes = {'pilot': [_block_size(planning, *b) for b in SIZE_PROBE_BLOCKS]}
    committed, cases = {}, {}
    del planning, master
    for which in ('srp1', 'paper'):
        case, _spec, _changes = S.derive_case(which, {})
        case_rel, case_sha = _scratch_case(case, scratch, which)
        p_other, _rd = _read(case_rel, scratch, which)
        sizes[which] = [_block_size(p_other, *b) for b in SIZE_PROBE_BLOCKS]
        cases[which] = {'derived_to_scratch': os.path.abspath(os.path.join(REPO, case_rel)), 'sha256': case_sha,
                        'scenario_checksum': p_other.scenario_metadata['combined_scenario_checksum']}
        del p_other
        committed[which] = [_committed_block_counts(which, *b) for b in SIZE_PROBE_BLOCKS]
    units = {'srp1': 48 * 1, 'pilot': 80 * 4, 'paper': 80 * 25}
    ratios = {}
    for i, (kind, node, year, day) in enumerate(SIZE_PROBE_BLOCKS):
        v = {w: sizes[w][i]['len_counts_at_build_model']['var'] for w in ('srp1', 'pilot', 'paper')}
        c = {w: sizes[w][i]['len_counts_at_build_model']['constraint'] for w in ('srp1', 'pilot', 'paper')}
        ratios[f'{kind}|{node}|{year}|{day}'] = {
            'vars': v, 'constraints': c,
            'pilot_over_srp1_vars': v['pilot'] / v['srp1'], 'paper_over_pilot_vars': v['paper'] / v['pilot'],
            'pilot_over_srp1_constraints': c['pilot'] / c['srp1'],
            'paper_over_pilot_constraints': c['paper'] / c['pilot']}
    facts['per_block_size'] = {
        'blocks_probed': [list(b) for b in SIZE_PROBE_BLOCKS], 'current_code': sizes, 'ratios': ratios,
        'size_probe_cases': cases, 'committed_pre_addendum38': committed,
        'network_blocks_per_cycle': {'srp1': 48, 'pilot': 80, 'paper': 80},
        'scenario_combinations_per_block': {'srp1': 1, 'pilot': 4, 'paper': 25},
        'block_x_scenario_units': units,
        'note': ('build_model counts (len over constructed component data, as the scale measurement records); the '
                 'ADMM additions (expected-interface Vars, row 18, AL terms) come after build_model and are not in '
                 'these counts')}
    return facts


# ======================================================================================================================
#  rule eleven (zero solves)
# ======================================================================================================================
CAMPAIGN_RESULT_FIELDS = (
    'status', 'barrier_cause', 'cycles_run', 'certification_cycle', 'certified_cost_gross', 'bar', 'rule_ten',
    'sigma_calibration', 'dispersion_per_dso_max_over_blocks_rms', 'dispersion_E_abs_d', 'dispersion_sum_omega_d2',
    'row18_charge_and_readback', 'settlement_split_and_covariance_identity', 'per_scenario_costs_reconciled',
    'per_scenario_interface_profiles', 'ess_dispatch_and_soh', 'penalty_table_components', 'operational_workbook',
    'hull_polish_gate', 'unrecovered_failures_and_certifying_window', 'solve_reconciliation', 'wall_time',
    'cycle_time', 'peak_rss', 'aa_action_counts', 'per_cycle_trajectory', 'value_and_resolution')


def rule_eleven():
    """CLAUDE.md rule eleven: before the run, a capture path exists for EVERY quantity spec v22 `pilot.runs`
    requires (and the Planner's task lists), asserted on the source of the code the children will run."""
    import p515_g_g1_g4_admm_gates as G
    import shared_resources_planning as srp
    import network as network_module
    harness_checks = H.assert_record_capture_paths()
    post_checks = H.assert_post_certification_capture_paths()
    child_src = inspect.getsource(H._child_real)
    capture_src = inspect.getsource(H.multiscenario_terminal_capture)
    hook_src = inspect.getsource(H._config_hook_factory)
    harness_src = inspect.getsource(H)
    arm_src = inspect.getsource(G.run_admm_arm)
    build_src = inspect.getsource(H.build_evaluation_record)
    checks = {
        # the path: run_admm_arm, not run_s39_arm (whose s39 checklist hard-codes objective_scale 93635360)
        'path_is_run_admm_arm': 'G.run_admm_arm(' in child_src,
        'path_never_calls_run_s39_arm': 'run_s39_arm' not in harness_src and 'assert_s39_capture_paths' not in harness_src,
        'run_s39_arm_is_the_one_with_the_hard_coded_sigma': (
            "checklist['objective_scale_is_93635360']" in inspect.getsource(G.assert_s39_capture_paths)
            and 'assert_s39_capture_paths' in inspect.getsource(G.run_s39_arm)),
        # the instance and the premium
        'derived_instance_installed_first': (child_src.find('install_derived_instance(derived, eval_dir)')
                                             < child_src.find('instance_investment_years()')
                                             and 'install_derived_instance(derived, eval_dir)' in child_src),
        'derived_instance_checksum_checked_in_child': "checksum != derived['scenario_checksum']" in inspect.getsource(
            H.install_derived_instance),
        'premium_applied_in_config_hook': "a.interface_deviation_premium = {'alpha': premium['alpha']" in hook_src,
        'premium_threaded_by_production': ('premium_alpha=interface_premium' in inspect.getsource(
            srp._run_operational_planning)),
        # sigma
        'sigma_in_state': "'sigma_computed': sigma_computed" in inspect.getsource(srp._run_operational_planning),
        'sigma_calibration_recorded': ("'sigma_calibration': ms.get('sigma_calibration')" in child_src
                                       and "payload['sigma_calibration'] = sigma_calibration_record(" in inspect.getsource(
                                           H.write_multiscenario_terminal)),
        'sigma_assertion_is_production': 'objective_scale_assert_factor' in inspect.getsource(
            srp._resolve_common_admm_objective_scale),
        # the multi-scenario outputs
        'capture_wired_before_post_certification': (child_src.find('write_multiscenario_terminal(')
                                                    < child_src.find('run_post_certification(')),
        'dispersion_production_function': '_get_operational_interface_dispersion(planning, models)' in capture_src,
        'dispersion_max_over_blocks_E_abs_d_sum_omega_d2': all(t in capture_src for t in (
            "'rms_mw_max_over_blocks'", "'E_abs_d_p_mwh_sum_over_blocks'", "'sum_omega_d2_p_mw2h_sum_over_blocks'")),
        'row18_charge_and_readback': ("'row18_charge_weighted'" in capture_src
                                      and '_row18_readback(model, network' in capture_src),
        'settlement_split_parts': all(f"part='{p}'" in capture_src for p in ('total', 'contracted', 'deviation')),
        'covariance_recomputed_independently': '_covariance_recomputed(model, network)' in capture_src,
        'settlement_identity_reported': "'leftover_identity_residual'" in capture_src,
        'per_scenario_costs_production': 'process_results_summary_detail(model, params)' in inspect.getsource(
            H._per_scenario_costs),
        'per_scenario_costs_reconciled_to_block_recourse': "'block_recourse_rel_diff'" in capture_src,
        'per_scenario_interface_profiles_production': 'process_results_interface(model)' in inspect.getsource(
            H._interface_profiles),
        'shared_ess_schedule_read_at_na_scenario': 'sess_na_scenario(model)' in inspect.getsource(
            H._shared_ess_schedule),
        'voltage_mismatch_reported': '_get_operational_scenario_voltage_mismatch' in capture_src,
        'workbook_production_writer': 'write_operational_planning_results_to_excel(' in inspect.getsource(
            H.write_operational_workbook),
        'workbook_gets_run_results': ("hook_kwargs['optimization_results'] = _results" in arm_src
                                      and "hook_kwargs['primal_evolution'] = _p" in arm_src
                                      and 'optimization_results=None' in child_src),
        'excel_dispersion_reads_na_copy': 'na=sess_na_scenario(model)' in inspect.getsource(
            srp._process_scenario_dispersion_results),
        'network_results_shared_ess_at_na': 'sess_na_scenario(model)' in inspect.getsource(network_module._process_results),
        # ESS dispatch, SoH, penalty table, per-cycle, record
        'ess_dispatch_esso_capture': "report['esso_capture'] = N.capture_esso(models['esso'], sed)" in arm_src,
        'soh_trajectory_declared_ageing': "holder['ageing_trajectory_terminal'] = ageing_trajectory_terminal(" in child_src,
        'penalty_table_component_levels': 'write_component_levels_terminal(' in inspect.getsource(
            G.write_interface_settlement_detail_s31c),
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'record_carries_multiscenario_summary': "'multiscenario_terminal':" in child_src,
        'record_rule_ten': "'terminal_step_over_threshold':" in build_src,
        'hull_polish_gate': 'gate_d_hull_polish' in inspect.getsource(H.run_post_certification),
        'capture_error_exits_2': "outcome.get('multiscenario_capture_error')" in inspect.getsource(H.main_child),
        'terminal_phase_serialized': ('acquire_terminal_phase_lock(terminal_lock_path)' in child_src
                                      and 'release_terminal_phase_lock(terminal_lock_path' in child_src),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (s52 pilot): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': harness_checks,
            'post_certification_capture_checklist': post_checks}


# ======================================================================================================================
#  memory preflight (A0's measure, the pilot's per-child budget)
# ======================================================================================================================
_VM_STAT_KEYS = {'pages_free': 'Pages free', 'pages_active': 'Pages active', 'pages_inactive': 'Pages inactive',
                 'pages_speculative': 'Pages speculative', 'pages_wired_down': 'Pages wired down',
                 'pages_purgeable': 'Pages purgeable', 'file_backed_pages': 'File-backed pages',
                 'anonymous_pages': 'Anonymous pages', 'pages_stored_in_compressor': 'Pages stored in compressor',
                 'pages_occupied_by_compressor': 'Pages occupied by compressor'}
_VM_STAT_REQUIRED = ('pages_free', 'pages_inactive', 'pages_wired_down', 'anonymous_pages',
                     'pages_occupied_by_compressor')


def memory_preflight(concurrency):
    required = memory_required_bytes(concurrency)
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    first = text.splitlines()[0]
    page = int(first.split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    pages = {name: raw.get(label) for name, label in _VM_STAT_KEYS.items()}
    missing = [name for name in _VM_STAT_REQUIRED if pages[name] is None]
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    out = {'utc': _utc(), 'vm_stat_header': first, 'page_size_bytes': page, 'vm_stat_pages': pages,
           'hw_memsize_bytes': total, 'required_bytes': required, 'required_gib': required / GIB,
           'per_child_sustained_budget_gib': MEMORY_PER_CHILD_SUSTAINED_BYTES / GIB,
           'terminal_transient_budget_gib': MEMORY_TERMINAL_TRANSIENT_BYTES / GIB, 'concurrency': concurrency,
           'rule': memory_rule(concurrency), 'rule_rationale': MEMORY_RULE_RATIONALE,
           'budget_derivation': MEMORY_BUDGET_DERIVATION, 'missing_vm_stat_figures': missing}
    if missing:
        out.update({'available_bytes': None, 'available_gib': None, 'pass': False})
        return out
    non_reclaimable = (pages['pages_wired_down'] + pages['anonymous_pages'] + pages['pages_occupied_by_compressor']) * page
    avail = total - non_reclaimable
    free_inactive = (pages['pages_free'] + pages['pages_inactive']) * page
    out.update({'non_reclaimable_gib': non_reclaimable / GIB, 'available_bytes': avail, 'available_gib': avail / GIB,
                'free_plus_inactive_gib': free_inactive / GIB, 'pass': avail >= required})
    return out


def _memory_line(m):
    if m.get('available_gib') is None:
        return f"vm_stat figures missing {m['missing_vm_stat_figures']} (rule {m['rule']})"
    return (f"available (memsize - wired - anonymous - compressor) = {m['available_gib']:.2f} GiB; free+inactive = "
            f"{m['free_plus_inactive_gib']:.2f} GiB (recorded only); required {m['required_gib']:.2f} GiB ({m['rule']})")


# ======================================================================================================================
#  predictions (recorded BEFORE any run; the Worker's, beside spec v22's Planner predictions)
# ======================================================================================================================
WORKER_PREDICTIONS = {
    'recorded_by': 'Worker (W47), 2026-09-23, before any pilot solve',
    'cycle_time': ('3.5-5 min per ADMM cycle per evaluation at concurrency 2 (2x2 limit gate: 118-129 s for init + 2 '
                   'cycles on 16 blocks of 4 scenarios = ~40 s per round, ~2.5 s per network block; the pilot has 80 '
                   'blocks -> ~3.4 min alone, +10-25 % for two concurrent children on 12 cores)'),
    'evaluation_wall_time': ('6-14 h per evaluation, central ~9 h (100-170 cycles to the 10-cycle certification -- '
                             'SRP1 ran 87-170 -- at ~4 min, + ~5 min initialization, + the terminal capture ~10 s and '
                             'the workbook ~2 min (both measured zero-solve), + post-certification ~10 min: model '
                             'pickle ~1 GB, hull polish of 80 blocks); both evaluations run in parallel, so the pilot '
                             'wave ~ the slower of the two'),
    'memory': ('sustained 5.5-6.5 GiB per child through the ADMM run (3.40 GiB of models measured zero-solve, + '
               'lightweight pristine clones, solver state and the SRP1-like fixed overhead; the Planner\'s 3.9 GiB '
               'was extrapolated from the snapshots-OFF 2x2 gate); PEAK ~11-12 GiB per child in the terminal phase '
               '(the model pickle\'s +5.1 GiB transient, measured; the workbook\'s +3.75 GiB before it); the two '
               'children\'s terminal phases are serialized, so the wave peaks at ~17-18 GiB; refusing budget 2 x 7 + '
               '6 = 20 GiB (13 GiB for the repro at concurrency 1)'),
    'repro_stage': '~12-20 min (initialization + 2 cycles, one child)',
    'sigma': 'ratio 0.36-0.60, passes (W45)',
    'dispersion': ('per-DSO max-over-blocks RMS ~0 at alpha = 0.50 under a CONVERGED run (single-block and 2x2 limit '
                   'evidence); the W44 two-cycle arms at alpha 0.5 still read 1.2-3.4 MW, so a nonzero terminal '
                   'dispersion of up to ~1 MW would not be a contradiction of the single-block result'),
}


WORKER_PREDICTIONS_W48_AMENDMENT = {
    'recorded_by': 'Worker (W48), 2026-09-23, before any pilot solve; amends the W47 predictions above, kept verbatim',
    'memory': ('no model pickle: PEAK per child in the terminal phase ~ sustained (5.5-6.5 GiB) + the workbook\'s '
               '+3.73 GiB = ~9.5-10.5 GiB; the two terminal phases serialized, so the wave peaks at ~13-15 GiB; '
               'refusing budget 2 x 7 + 4 = 18 GiB (11 GiB for the repro)'),
    'evaluation_wall_time': 'unchanged central ~9 h, less the model pickle (~45 s + write of ~0.66 GB)',
}


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def _spec_candidates(stage, derived):
    post = STAGES[stage]['post_certification']
    out = []
    for label in STAGES[stage]['points']:
        options = {'investment_year': YEAR, 'interface_deviation_premium': dict(PREMIUM)}
        if post:
            options['post_certification'] = dict(post)
        out.append((label, POINT_NODES[label], options))
    return out


def validate_spec(stage, spec, derived, facts_frozen):
    cfg = spec['configuration']
    entries = spec['candidates']
    extra = spec.get('extra') or {}
    st = STAGES[stage]
    checks = {
        'campaign_id': spec.get('campaign_id') == st['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == list(st['points']),
        'cap': spec.get('cap') == st['cap'], 'concurrency': spec.get('concurrency') == st['concurrency'],
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label': cfg.get('arm_label') == ARM_LABEL,
        'no_campaign_overrides': cfg.get('overrides') == {},
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_file_pinned': (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_FILE['sha256'],
        'derived_instance_declared': cfg.get('derived_instance') == derived,
        'derived_instance_case_sha256_is_w45': (cfg.get('derived_instance') or {}).get('case_sha256') == EXPECTED_CASE_SHA256,
        'derived_instance_checksum_is_w45': (cfg.get('derived_instance') or {}).get('scenario_checksum') == EXPECTED_SCENARIO_CHECKSUM,
        'no_model_variant': 'model_variant_label' not in spec and not any('model_variant' in e for e in entries),
        'no_flex_price_variant': 'flex_price_label' not in spec and not any('flex_price_multiplier' in e for e in entries),
        'extra_stage_recorded': extra.get('stage') == stage,
        'objective_convention_recorded': extra.get('objective_convention') == OBJECTIVE_CONVENTION,
        'facts_frozen': extra.get('instance_facts') == facts_frozen,
        'predecessor_recorded': extra.get('predecessor_spec') == PREDECESSOR_SPECS[stage],
        'persistence_not_requested': not any((e.get('post_certification') or {}).get('persist_certified_models')
                                             for e in entries),
    }
    for e in entries:
        label = e['label']
        nodes = POINT_NODES[label]
        checks[f'{label}:canonical_key'] = e.get('key') == _key_of(nodes)
        checks[f'{label}:eval_key_recomputes'] = e.get('eval_key') == _eval_key(nodes, derived)
        checks[f'{label}:eval_key_not_the_srp1_key'] = e.get('eval_key') != H.evaluation_key(
            _key_of(nodes), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE)
        checks[f'{label}:premium'] = e.get('interface_deviation_premium') == PREMIUM
        checks[f'{label}:no_overrides'] = e.get('overrides') == {}
        checks[f'{label}:effective_aa'] = e.get('effective_anderson_acceleration') == CASE_FILE_AA
        post = st['post_certification']
        checks[f'{label}:post_certification'] = (
            e.get('post_certification') == ({**post, 'reference': None} if post else None))
    return checks


def _common_checks(stage, write_instance, scratch=None, with_facts=True):
    failures, evidence = [], {}
    pins, more = check_pins()
    evidence['pins'] = pins
    failures += more
    instance = ensure_instance_file(write_instance)
    evidence['instance'] = instance
    if not (instance['equals_w45_pilot_case_sha256'] and instance['years_expected']):
        failures.append(f'instance: sha256 {instance["case_sha256"]} (W45 {EXPECTED_CASE_SHA256}) / years '
                        f'{instance["years"]}')
    evidence['sigma_inputs'] = sigma_inputs()
    if evidence['sigma_inputs']['w45_pilot_case_sha256'] != instance['case_sha256']:
        failures.append('the W45 sigma check was computed on a different pilot case file')
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    if with_facts:
        evidence['instance_facts'] = instance_facts(scratch, instance)
        f = evidence['instance_facts']
        if not f['scenario_checksum_equals_w45']:
            failures.append(f"scenario checksum {f['scenario_checksum']} != W45 {EXPECTED_SCENARIO_CHECKSUM}")
        if f['premium_floor_needed']:
            failures.append(f"a non-positive mean hourly price exists ({f['min_hourly_mean_price']}): Addendum 38 "
                            'requires a premium floor, which this spec does not carry')
        if f['declared_solve_profile_per_cycle'] != 83:
            failures.append(f"solves per cycle {f['declared_solve_profile_per_cycle']} != 83")
    return failures, evidence


def freeze(stage, started, scratch):
    tag = f'S52-{stage.upper()}'
    root = campaign_root(stage)
    raw = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    lock_observations = [f for f in raw if f.startswith(LOCK_FAILURE_PREFIXES)]
    failures = [f for f in raw if f not in lock_observations]
    more, evidence = _common_checks(stage, write_instance=True, scratch=scratch)
    failures += more
    memory = memory_preflight(STAGES[stage]['concurrency'])
    if failures:
        for failure in failures:
            _log(f'[{tag} FREEZE PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    facts = evidence['instance_facts']
    derived = derived_declaration(evidence['instance'], facts['scenario_checksum'])
    instance_record_path = os.path.join(REPO, INSTANCE_RECORD_REL)
    instance_record = {'schema': 'p515_s52_pilot_instance_v1', 'instance_label': INSTANCE_LABEL,
                       'derived_instance_declaration': derived, 'instance': evidence['instance'],
                       'facts': facts, 'written_by': SCRIPT_NAME, 'stage_first_frozen': stage, 'utc': _utc()}
    if os.path.exists(instance_record_path):
        existing = _load(INSTANCE_RECORD_REL)
        if existing.get('derived_instance_declaration') != derived:
            raise SystemExit(f'{INSTANCE_RECORD_REL} exists with a different declaration')
    else:
        H._write_once_json(instance_record_path, instance_record)
    st = STAGES[stage]
    extra = {
        'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'stage': stage, 'stage_description': st['description'], 'label': LABEL, 'instance_label': INSTANCE_LABEL,
        'instance_record': {'path': INSTANCE_RECORD_REL, 'sha256': H.sha256_file(instance_record_path)},
        'spec_v22': dict(SPEC_V22), 'cost_file': dict(COST_FILE), 'ess_params_file': dict(ESS_PARAMS_FILE),
        'pins': evidence['pins'], 'instance': evidence['instance'], 'instance_facts': facts,
        'interface_deviation_premium': dict(PREMIUM),
        'alpha_statement': ('alpha = 0.50: the author\'s fixed pilot value (Addendum 39); no premium floor (min hourly '
                            f"mean price {facts['min_hourly_mean_price']:.4f} > 0)"),
        'sigma_check_inputs': evidence['sigma_inputs'],
        'points': {label: {'nodes': {str(n): list(v) for n, v in POINT_NODES[label].items()}, 'investment_year': YEAR,
                           'candidate_key': _key_of(POINT_NODES[label]),
                           'eval_key': _eval_key(POINT_NODES[label], derived),
                           'I_x': facts['investment_cost'][label]} for label in st['points']},
        'post_certification': st['post_certification'], 'post_certification_rationale': POST_CERTIFICATION_RATIONALE,
        'objective_convention': OBJECTIVE_CONVENTION, 'value_definition': VALUE_DEFINITION,
        'budget_convention': BUDGET_CONVENTION, 'dispersion_convention': DISPERSION_CONVENTION,
        'stop_rule': STOP_RULE,
        'srp1_reference_for_value_per_mwh': dict(SRP1_S2_RESULTS, value_per_mwh=SRP1_VALUE_PER_MWH_EXPECTED,
                                                  resolution=SRP1_RESOLUTION_EXPECTED),
        'r_prediction': R_PREDICTION,
        'predictions_recorded_before_run': {'planner_spec_v22': _load(SPEC_V22['path'])['pilot'][
            'predictions_recorded_before_run'], 'worker': WORKER_PREDICTIONS,
            'worker_w48_amendment': WORKER_PREDICTIONS_W48_AMENDMENT},
        'predecessor_spec': dict(PREDECESSOR_SPECS[stage]),
        'w48_persistence_ruling': {
            'ruling': ('Planner, W48 ruling 1: drop persist_certified_models from the pilot; re-freeze both specs '
                       'under new hashes'),
            'why': POST_CERTIFICATION_RATIONALE,
            'post_certification_w47_superseded': {'persist_certified_models': True, 'hull_polish': True},
            'post_certification_rationale_w47_superseded': POST_CERTIFICATION_RATIONALE_W47_SUPERSEDED,
            'memory_budget': {
                'per_child_sustained_gib': MEMORY_PER_CHILD_SUSTAINED_BYTES / GIB,
                'terminal_transient_gib': MEMORY_TERMINAL_TRANSIENT_BYTES / GIB,
                'terminal_transient_gib_w47_superseded': MEMORY_TERMINAL_TRANSIENT_BYTES_W47_SUPERSEDED / GIB,
                'required_gib_this_stage': memory_required_bytes(st['concurrency']) / GIB,
                'required_gib_this_stage_w47_superseded': (
                    st['concurrency'] * MEMORY_PER_CHILD_SUSTAINED_BYTES
                    + MEMORY_TERMINAL_TRANSIENT_BYTES_W47_SUPERSEDED) / GIB,
                'measured_basis': ('p515_s52_pilot_checks r1 (ae951d31) memory_by_stage: workbook RSS 3,730,014,208 '
                                   '-> peak 7,732,314,112 bytes (+3.727 GiB), retained +0.537 GiB; pickle +5.1 GiB '
                                   'transient, +2.3 GiB retained (now not incurred)')},
            'gate_before_the_pilot': ('SRP1 two-cycle bitwise identity for the W47 harness extension: PASS, '
                                      'p515_s52_srp1_bitwise_gate.py, evidence e75a575e')},
        'memory_preflight_rule': memory_rule(st['concurrency']), 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
        'memory_budget_derivation': MEMORY_BUDGET_DERIVATION, 'memory_at_freeze_non_gating': memory,
        'lock_observations_at_freeze_non_gating': lock_observations,
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS),
        'path_statement': ('every evaluation runs through p515_s44_campaign_harness -> p515_g_g1_g4_admm_gates.'
                           'run_admm_arm (asserted: path_is_run_admm_arm, path_never_calls_run_s39_arm)'),
    }
    if stage == 'repro':
        extra['reproduces'] = {'stage': 'pilot', 'campaign_id': STAGES['pilot']['campaign_id'], 'label': 'x0',
                               'eval_key': _eval_key(POINT_NODES['x0'], derived),
                               'compare': ('cycle_trajectory[:2] bitwise (every field, type and value) + '
                                           'H.RECORD_TRAJECTORY_FIELDS table; --compare')}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, st['campaign_id'], _spec_candidates(stage, derived),
        configuration={'name': (f'{LABEL}; the derived pilot instance {INSTANCE_LABEL} (paper years x 4 days x 2 x 2); '
                                'row 18 at alpha = 0.50 per entry; case-file AA declared'),
                       'arm_label': ARM_LABEL, 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': dict(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'derived_instance': derived,
                       'note': ('no overrides; no model variant; no flexibility-price variant; the derived instance is '
                                'installed in the child before anything reads the oracle baseline; the premium is set '
                                'in the configuration hook before any model is built; num_max_iters := cap')},
        cap=st['cap'], concurrency=st['concurrency'], authority=AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES, extra=extra)
    checks = validate_spec(stage, spec, derived, facts)
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    _log(f"[{tag}] instance {INSTANCE_LABEL}: {INSTANCE_CASE_REL} sha256={derived['case_sha256']} "
         f"(W45 {EXPECTED_CASE_SHA256 == derived['case_sha256']}); scenario checksum {derived['scenario_checksum']} "
         f"(W45 {facts['scenario_checksum_equals_w45']})")
    _log(f"[{tag}] dimensions: years {facts['planning_dimensions']['years']} x days "
         f"{facts['planning_dimensions']['days']}; market {facts['planning_dimensions']['num_market_scenarios']} x "
         f"operation {facts['planning_dimensions']['num_operation_scenarios']}; blocks {facts['expected_block_counts']}")
    for key, r in facts['per_block_size']['ratios'].items():
        _log(f"[{tag}] block size {key}: vars {r['vars']} constraints {r['constraints']} (pilot/SRP1 vars "
             f"{r['pilot_over_srp1_vars']:.2f}, paper/pilot vars {r['paper_over_pilot_vars']:.2f})")
    for label in st['points']:
        p = extra['points'][label]
        _log(f"[{tag}]   {label}: key={p['candidate_key'][:16]} eval_key={p['eval_key'][:16]} "
             f"I(x)={p['I_x']['I_x_eur_master_expression']} slack={p['I_x']['budget_slack_eur']}")
    _log(f"[{tag}] sigma inputs: floor ratio {evidence['sigma_inputs']['ratio_floor']:.4f} vs band "
         f"{evidence['sigma_inputs']['band']} (x{evidence['sigma_inputs']['ratio_floor_over_lower_band_edge']:.4f} "
         f"the lower edge); paper reference {evidence['sigma_inputs']['paper_5x5_reference_ratio']:.4f}")
    _log(f"[{tag}] min hourly mean price {facts['min_hourly_mean_price']:.4f} (floor needed: "
         f"{facts['premium_floor_needed']}); snapshots {facts['case_file_admm']['tso_snapshot_capture_mode']} / "
         f"{facts['case_file_admm']['dso_snapshot_capture_mode']}")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f'[{tag}] rule eleven: {evidence["rule_eleven"]["checks"]}')
    _log(f'[{tag}] lock observations (non-gating at --freeze): {lock_observations}')
    _log(f"[{tag}] memory at freeze (non-gating): {_memory_line(memory)} -> would "
         f"{'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}')
    _log(f'[{tag}] run with: --stage {stage} --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _per_cycle_rows(rec):
    path = rec.get('per_cycle_record_path')
    if not path or not os.path.isfile(os.path.join(REPO, path)):
        return {'path': path, 'present': False}, []
    with open(os.path.join(REPO, path)) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {'path': path, 'present': True, 'sha256': H.sha256_file(os.path.join(REPO, path)), 'n_rows': len(rows),
            'n_rows_equals_cycles_run': len(rows) == rec.get('cycles_run')}, rows


def _eval_json(rec, name):
    eval_dir = rec.get('eval_dir')
    path = os.path.join(REPO, eval_dir, name) if eval_dir else None
    if path and os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    return None


def _sigma_from_cause(cause):
    """A run that failed the fixed-sigma calibration assertion raises in production with both numbers."""
    if not cause or 'objective scale (sigma) failed its calibration-range assertion' not in cause:
        return None
    return {'sigma_assertion_failed': True, 'production_message': cause}


def _point_result(label, rec, spec_point):
    rec = rec or {}
    traj, rows = _per_cycle_rows(rec)
    certified = rec.get('status') == 'certified'
    last = rows[-1] if rows else {}
    prev = rows[-2] if len(rows) >= 2 else {}
    gross_step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
                  if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None
                  else None)
    tol = last.get('objective_tolerance')
    window = rows[-REQUIRED_CONSECUTIVE_CYCLES:]
    nfs = (rec.get('network_failures_summary') or {})
    ms = rec.get('multiscenario_terminal') or {}
    boyd = _eval_json(rec, 'boyd_terminal.json') or {}
    sigma = rec.get('sigma_calibration') or {
        'sigma_fixed': boyd.get('sigma_fixed'), 'sigma_computed': boyd.get('sigma_computed'),
        'ratio_computed_over_fixed': (boyd['sigma_computed'] / boyd['sigma_fixed'])
        if boyd.get('sigma_fixed') and boyd.get('sigma_computed') is not None else None,
        'source': 'boyd_terminal.json (the record carries no sigma_calibration)'}
    sigma_failure = _sigma_from_cause(rec.get('barrier_cause'))
    wall = rec.get('wall_time_s') or {}
    run_s = wall.get('run_admm_arm_s')
    pc = rec.get('post_certification') or {}
    return {
        'label': label, 'LABEL': LABEL, 'instance_label': INSTANCE_LABEL, 'alpha': ALPHA,
        'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
        'eval_key': rec.get('eval_key'), 'eval_dir': rec.get('eval_dir'),
        'status': rec.get('status'), 'barrier_cause': rec.get('barrier_cause'),
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'certified_cost_gross': rec.get('certified_cost') if certified else None,
        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'net_operational_recourse': rec.get('terminal_net_operational_recourse'),
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'bar': (rec.get('bar') or {}).get('value'),
        'rule_ten': {'terminal_step_over_threshold_production': (rec.get('rule_ten') or {}).get(
            'terminal_step_over_threshold'),
            'terminal_gross_step_abs': gross_step,
            'terminal_gross_step_over_threshold': (gross_step / tol) if (gross_step is not None and tol) else None,
            'boyd_terminal_ratio_max_per_channel': (rec.get('rule_ten') or {}).get('boyd_terminal_ratio_max_per_channel')},
        'sigma_calibration': sigma, 'sigma_assertion_failure': sigma_failure,
        'multiscenario_terminal': {k: ms.get(k) for k in (
            'status', 'error', 'path', 'sha256', 'alpha_in_force', 'n_blocks', 'n_dso_blocks_row18_wired', 'per_dso',
            'all_dso', 'settlement_identity', 'identity_worst_rel_diffs', 'checks', 'all_checks_pass',
            'voltage_pin_mismatch_max_rms_pu', 'runtime_s')},
        'operational_workbook': rec.get('operational_workbook'),
        'post_certification': {'status': pc.get('status'), 'skip_reason': pc.get('skip_reason'),
                               'error': pc.get('error'), 'persisted_models': pc.get('persisted_models'),
                               'gate_d_hull_polish': pc.get('gate_d')},
        'unrecovered_failures': {'classes': nfs.get('classes'), 'n_blocks': nfs.get('n_blocks'),
                                 'n_esso_recovery_events': nfs.get('n_esso_recovery_events'),
                                 'certifying_window_local_solves_ok': (all(r.get('local_solves_ok') for r in window)
                                                                       if window else None),
                                 'policy': ('continue with the last iterate, count the event, no unrecovered failure '
                                            'inside the 10 certifying cycles (production behaviour, W35)')},
        'solve_reconciliation': {k: (rec.get('solve_profile') or {}).get(k) for k in (
            'observed', 'identity_holds', 'reconciliation_supported', 'expected_solves', 'base_solves',
            'retry_solves_credited', 'solves_per_cycle')},
        'storage_per_node': rec.get('storage_per_node'),
        'ageing_trajectory_terminal': rec.get('ageing_trajectory_terminal'),
        'ess_ageing_readback_terminal_all_match': (rec.get('ess_ageing_readback_terminal') or {}).get('all_match'),
        'ess_ageing_readback_pre_run_all_match': ((rec.get('ess_ageing_verified_pre_run') or {}).get(
            'readback_pre_run') or {}).get('all_match'),
        'derived_instance_installed_in_child': rec.get('derived_instance_installed_in_child'),
        'interface_deviation_premium_applied_in_child': rec.get('interface_deviation_premium_applied_in_child'),
        'wall_time': wall, 'parent_view': rec.get('parent_view'),
        'cycle_time_s_mean_incl_init_round': (run_s / (rec['cycles_run'] + 1)) if (run_s and rec.get('cycles_run')) else None,
        'peak_rss': rec.get('peak_rss'),
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'per_cycle_trajectory': traj, 'I_x': spec_point.get('I_x'),
    }


def _value_block(points):
    x0, unit = points.get('x0') or {}, points.get('n7_4h_e1') or {}
    q0, q1 = x0.get('certified_cost_gross'), unit.get('certified_cost_gross')
    if q0 is None or q1 is None:
        return {'available': False, 'reason': 'both evaluations must be certified'}
    value = q0 - q1
    i_x = (unit.get('I_x') or {}).get('I_x_eur_master_expression')
    resolution = (x0.get('bar') or 0.0) + (unit.get('bar') or 0.0)
    per_mwh = value / UNIT_E_MWH
    ratio = per_mwh / SRP1_VALUE_PER_MWH_EXPECTED
    return {'available': True, 'Q0': q0, 'Q_unit': q1, 'value_eur': value, 'I_x_eur': i_x,
            'value_minus_I_eur': value - i_x if i_x is not None else None, 'F_unit': (i_x + q1) if i_x else None,
            'resolution_eur': resolution, 'value_determinate': abs(value) > resolution,
            'value_minus_I_determinate': (abs(value - i_x) > resolution) if i_x is not None else None,
            'value_per_mwh': per_mwh, 'srp1_value_per_mwh': SRP1_VALUE_PER_MWH_EXPECTED,
            'srp1_resolution': SRP1_RESOLUTION_EXPECTED, 'ratio_pilot_over_srp1': ratio,
            'ratio_resolution': math.hypot(resolution / SRP1_VALUE_PER_MWH_EXPECTED,
                                           per_mwh * SRP1_RESOLUTION_EXPECTED / SRP1_VALUE_PER_MWH_EXPECTED ** 2),
            'r_prediction': R_PREDICTION, 'ratio_minus_r': ratio - R_PREDICTION,
            'within_10pct_of_srp1': abs(ratio - 1.0) <= 0.10, 'definition': VALUE_DEFINITION}


def stop_rule_state(records):
    non_certified = [r for r in records if (r or {}).get('status') != 'certified']
    regions = Counter(_region_key(r['candidate_canonical']) for r in non_certified if (r or {}).get('candidate_canonical'))
    reasons = []
    if any(c >= 2 for c in regions.values()):
        reasons.append(f'(a) 2 or more non-certified points in one region: {dict(regions)}')
    if len(non_certified) >= 3:
        reasons.append('(b) 3 or more non-certified points overall')
    return {'triggered': bool(reasons), 'reasons': reasons,
            'non_certified_labels': [r.get('candidate_label') for r in non_certified]}


def run(stage, started, spec_sha256):
    tag = f'S52-{stage.upper()}'
    root = campaign_root(stage)
    superseded = {v['sha256']: v for v in PREDECESSOR_SPECS.values()}
    if spec_sha256 in superseded:   # W48: never run a superseded spec
        _log(f"[{tag} PRECONDITION FAILED] {spec_sha256} is a superseded predecessor spec "
             f"({superseded[spec_sha256]['path']}; {superseded[spec_sha256]['reason']}); it is never run")
        raise SystemExit(1)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    root_contents = sorted(os.listdir(root))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence = _common_checks(stage, write_instance=False, with_facts=False)
    failures += more
    derived = spec['configuration'].get('derived_instance')
    checks = validate_spec(stage, spec, derived, (spec.get('extra') or {}).get('instance_facts'))
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for rel in (INSTANCE_CASE_REL, INSTANCE_RECORD_REL):
        tracked, clean = _git_state(rel)
        if not (tracked and clean):
            failures.append(f'{rel} must be committed and clean before --run (tracked={tracked}, clean={clean})')
    if (derived or {}).get('case_sha256') != evidence['instance']['case_sha256']:
        failures.append('the instance file no longer hashes to the frozen declaration')
    if os.path.isfile(os.path.join(REPO, INSTANCE_RECORD_REL)):
        if _load(INSTANCE_RECORD_REL).get('derived_instance_declaration') != derived:
            failures.append(f'{INSTANCE_RECORD_REL} declares a different derived instance than the frozen spec')
        if spec['extra'].get('instance_record', {}).get('sha256') != H.sha256_file(os.path.join(REPO, INSTANCE_RECORD_REL)):
            failures.append(f'{INSTANCE_RECORD_REL} sha256 differs from the one the spec recorded')
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    if spec['configuration']['ess_params_file']['sha256'] != H.sha256_file(os.path.join(REPO, H.ESS_PARAMS_FILE_REL)):
        failures.append('ESS params file sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    memory = memory_preflight(STAGES[stage]['concurrency'])
    _log(f"[{tag}] memory preflight: {_memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {_memory_line(memory)}')
    if failures:
        for failure in failures:
            _log(f'[{tag} PRECONDITION FAILED] {failure}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    labels = list(STAGES[stage]['points'])
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; git HEAD {head} '
         f'(spec frozen at {spec["git_head"]}); points {labels}, one wave at concurrency {spec["concurrency"]}, '
         f'cap {spec["cap"]}')
    lock = H.acquire_campaign_lock(STAGES[stage]['campaign_id'], spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(labels, ctx)
        wave_info = dict(getattr(H.evaluate, 'last_batch_info', {}))
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    points = {label: _point_result(label, by_label.get(label), spec['extra']['points'][label]) for label in labels}
    non_certified = [l for l in labels if points[l]['status'] != 'certified']
    harness_errors = [l for l in labels if points[l]['status'] not in ('certified', 'not_certified')
                      or (points[l].get('parent_view') or {}).get('exit_code') != 0]
    capture_problems = [l for l in labels if points[l]['status'] in ('certified', 'not_certified')
                        and not ((points[l]['multiscenario_terminal'] or {}).get('all_checks_pass') is True
                                 and (points[l]['operational_workbook'] or {}).get('status') == 'written'
                                 and points[l]['ess_ageing_readback_terminal_all_match']
                                 and points[l]['ess_ageing_readback_pre_run_all_match'])]
    sigma_failures = [l for l in labels if points[l]['sigma_assertion_failure']]
    stop_state = stop_rule_state(records)
    guard_failures = PARENT_GUARD.verify(0)
    table = [{'label': l, 'status': points[l]['status'], 'cycles': points[l]['cycles_run'],
              'cert_cycle': points[l]['certification_cycle'], 'Q_gross': points[l]['certified_cost_gross'],
              'bar': points[l]['bar'],
              'rule_ten_production': points[l]['rule_ten']['terminal_step_over_threshold_production'],
              'sigma_ratio': (points[l]['sigma_calibration'] or {}).get('ratio_computed_over_fixed'),
              'rms_mw_max_over_all_dso_blocks': ((points[l]['multiscenario_terminal'] or {}).get('all_dso') or {}).get(
                  'rms_mw_max_over_all_dso_blocks'),
              'E_abs_d_mwh_sum_over_blocks': ((points[l]['multiscenario_terminal'] or {}).get('all_dso') or {}).get(
                  'E_abs_d_p_mwh_sum_over_blocks'),
              'row18_charge_weighted': ((points[l]['multiscenario_terminal'] or {}).get('all_dso') or {}).get(
                  'row18_charge_weighted'),
              'hull_polish_pass': ((points[l]['post_certification'] or {}).get('gate_d_hull_polish') or {}).get('pass'),
              'cycle_time_s': points[l]['cycle_time_s_mean_incl_init_round'],
              'wall_s': (points[l]['parent_view'] or {}).get('wall_s'),
              'peak_rss_bytes': (points[l]['parent_view'] or {}).get('wait4_ru_maxrss')} for l in labels]
    results = {
        'STAGE': STAGE_TEXT, 'LABEL': LABEL, 'stage_id': stage, 'stage': STAGES[stage]['description'],
        'STOP_FOR_REVIEW': True if stage == 'pilot' else stop_state['triggered'],
        'stop_for_review_reason': ('spec v22 order: pilot -> STOP FOR REVIEW' if stage == 'pilot' else None),
        'stop_rule': STOP_RULE, 'stop_rule_state': stop_state,
        'non_certified_points': non_certified, 'harness_errors': harness_errors, 'capture_problems': capture_problems,
        'sigma_assertion_failures': sigma_failures, 'authority': AUTHORITY, 'timestamp_utc': _utc(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256, 'objective_convention': OBJECTIVE_CONVENTION,
        'dispersion_convention': DISPERSION_CONVENTION, 'budget_convention': BUDGET_CONVENTION,
        'table_objective_convention': 'Q gross (contracted settlement and voltage pin excluded; row 18 and covariance '
                                      'included); salvage excluded',
        'table': table, 'points': points, 'value': _value_block(points) if stage == 'pilot' else None,
        'memory_preflight_at_run': memory, 'pre_run_evidence': {'pins': evidence['pins'],
                                                                'instance': evidence['instance']},
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'], 'wave_info': wave_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
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
    _log(f'[{tag}] objective convention: {OBJECTIVE_CONVENTION}')
    for row in table:
        _log(f'[{tag}] {row}')
    sigma_lines = []
    for l in labels:
        s = points[l]['sigma_calibration'] or {}
        sigma_lines.append(f"SIGMA {l}: sigma_computed={s.get('sigma_computed')} sigma_fixed={s.get('sigma_fixed')} "
                           f"ratio={s.get('ratio_computed_over_fixed')} band={s.get('band')} "
                           f"ratio/lower_edge={s.get('ratio_over_lower_edge')} within_band={s.get('within_band')}"
                           + (f" -- ASSERTION FAILED: {points[l]['sigma_assertion_failure']}"
                              if points[l]['sigma_assertion_failure'] else ''))
    _banner(sigma_lines)
    if stage == 'pilot':
        _log(f"[{tag}] value: {results['value']}")
    if non_certified:
        _log(f"[{tag}] non-certified points (reported with cause): "
             f"{[(l, points[l].get('barrier_cause')) for l in non_certified]}")
    if capture_problems:
        _log(f'[{tag}] CAPTURE / READ-BACK PROBLEM: {capture_problems}')
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    if stage == 'pilot':
        _banner(['STOP FOR REVIEW (spec v22 order: pilot -> STOP FOR REVIEW)'])
    if guard_failures or harness_errors or capture_problems or sigma_failures:
        _log(f'[{tag}] NOT OK')
        sys.exit(1)
    if non_certified and stage == 'pilot':
        _log(f'[{tag}] harness clean; {len(non_certified)} non-certified point(s)')
        sys.exit(2)
    _log(f'[{tag}] OK')


# ======================================================================================================================
#  --compare (zero solves)
# ======================================================================================================================
def _eval_report(stage, label):
    root = campaign_root(stage)
    results = json.load(open(os.path.join(root, 'campaign_results.json')))
    point = results['points'][label]
    rel = os.path.join(point['eval_dir'], f'g_{ARM_LABEL}.json')
    with open(os.path.join(REPO, rel)) as handle:
        return rel, H.sha256_file(os.path.join(REPO, rel)), json.load(handle), point


def _row_diffs(a_rows, b_rows):
    diffs = []
    for i, (a, b) in enumerate(zip(a_rows, b_rows)):
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key, '<absent>'), b.get(key, '<absent>')
            if not (type(va) is type(vb) and json.dumps(va, sort_keys=True, default=str)
                    == json.dumps(vb, sort_keys=True, default=str)):
                diffs.append({'index': i, 'cycle': a.get('cycle'), 'field': key, 'pilot': va, 'repro': vb})
    return diffs


def compare(started):
    out_root = os.path.join(REPO, COMPARE_ROOT_REL)
    if os.path.exists(out_root):
        raise SystemExit(f'output exists (write-once): {out_root}')
    p_rel, p_sha, p_rep, p_point = _eval_report('pilot', 'x0')
    r_rel, r_sha, r_rep, r_point = _eval_report('repro', 'x0')
    n = 2
    a_rows, b_rows = (p_rep.get('cycle_trajectory') or [])[:n], (r_rep.get('cycle_trajectory') or [])[:n]
    diffs = _row_diffs(a_rows, b_rows)
    table = {}
    for field in H.RECORD_TRAJECTORY_FIELDS:
        mism = [i for i in range(min(len(a_rows), len(b_rows)))
                if not (type(a_rows[i].get(field)) is type(b_rows[i].get(field))
                        and a_rows[i].get(field) == b_rows[i].get(field))]
        table[field] = {'n_mismatch': len(mism), 'indices': mism}
    same_key = p_point.get('eval_key') == r_point.get('eval_key')
    payload = {
        'stage': 'P5.15 W47 pilot: two-cycle bitwise reproduction of the x = 0 evaluation', 'utc': _utc(),
        'pilot_report': {'path': p_rel, 'sha256': p_sha, 'cycles_run': p_rep.get('cycles_run')},
        'repro_report': {'path': r_rel, 'sha256': r_sha, 'cycles_run': r_rep.get('cycles_run')},
        'same_eval_key': same_key, 'eval_key': p_point.get('eval_key'),
        'n_cycles_compared': min(len(a_rows), len(b_rows)), 'n_rows': [len(a_rows), len(b_rows)],
        'n_diffs': len(diffs), 'diffs_first': diffs[:50], 'field_table': table,
        'field_table_total_mismatches': sum(v['n_mismatch'] for v in table.values()),
        'reproduces_bitwise': same_key and len(a_rows) == len(b_rows) == n and not diffs,
        'definition': ('cycle_trajectory[:2] of g_s39_D.json (every field of every row, type and value, JSON-'
                       'canonical) of the repro (cap 2) against the pilot (cap 500), plus the harness trajectory '
                       'field table; the SRP1 precedent (W35 / W39 gates) found cap-2 rows equal to cap-500 rows'),
        'parent_guard': dict(PARENT_GUARD.counts), 'wall_s': time.time() - started,
    }
    os.makedirs(out_root)
    H._write_once_json(os.path.join(out_root, 'compare.json'), payload)
    H._write_once_json(os.path.join(out_root, 'manifest_sha256.json'),
                       {os.path.relpath(os.path.join(out_root, 'compare.json'), REPO):
                        H.sha256_file(os.path.join(out_root, 'compare.json'))})
    guard_failures = PARENT_GUARD.verify(0)
    PARENT_GUARD.uninstall()
    _log(f"[S52-COMPARE] reproduces_bitwise={payload['reproduces_bitwise']} n_diffs={len(diffs)} "
         f"field_table_mismatches={payload['field_table_total_mismatches']} guard_failures={guard_failures}")
    sys.exit(0 if payload['reproduces_bitwise'] and not guard_failures else 1)


def main():
    parser = argparse.ArgumentParser(description=STAGE_TEXT)
    parser.add_argument('--stage', choices=sorted(STAGES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: write the instance, freeze + validate')
    mode.add_argument('--run', action='store_true', help='evaluate the frozen spec named by --spec-sha256')
    mode.add_argument('--compare', action='store_true', help='zero solves: pilot x0 vs repro x0, two cycles')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--scratch', default=None, help='--freeze: directory OUTSIDE the repo for the reads\' plots')
    args = parser.parse_args()
    started = time.time()
    os.chdir(REPO)
    if args.compare:
        compare(started)
        return
    if not args.stage:
        parser.error('--stage is required with --freeze / --run')
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        if not args.scratch or os.path.abspath(args.scratch).startswith(REPO + os.sep):
            parser.error('--freeze requires --scratch <dir outside the repository>')
        os.makedirs(args.scratch, exist_ok=True)
        freeze(args.stage, started, os.path.abspath(args.scratch))
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(args.stage, started, args.spec_sha256)


if __name__ == '__main__':
    main()
