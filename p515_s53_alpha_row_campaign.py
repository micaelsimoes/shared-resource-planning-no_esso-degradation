"""
P5.15 Addendum 40 ruling 1 (task W64) -- THE ALPHA ROW: launcher, frozen spec v24, capture gate. BUILT, NOT LAUNCHED.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 1 (the R3.6 alpha row, settled, 2 x 2) and ruling 2 (the
init fix: the pilot's alpha > 0 artifacts are superseded, the unit is re-run at alpha = 0.5 to restate R); frozen
spec v23 `data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json` `ruling1_alpha_row`; Planner task W64. The
frozen spec of THIS stage is v24 (`--freeze-spec`, predecessor v23), which holds the purpose, every formula, the
capture, the pre-run assertions, the hazard, the smoke gate and every prediction.

PURPOSE (spec v24 `purpose`). NOT a search for a coordinated alpha*. The row is a TWO-MECHANISM COST-OF-COMMITMENT
CURVE: market-arbitrage collapse at low alpha (alpha_arb = price spread / 2 pibar has median 0.022, p90 0.157, max
0.668 on this instance), physical-waste growth at high alpha (curtailment needs alpha of order 4+ to hold schedule
where flexibility is exhausted).

THE INSTANCE: the committed 2 x 2 pilot derived case (`p515_s52_pilot_campaign`: 5 representative years x 4 days x
2 market x 2 operation, `data/SRP1/Results/P515S52/pilot_instance/SRP1__s52_pilot_2x2.json`, sha256 7ecff44a...,
scenario checksum 53b4bea4...), baseline ageing, case-file AA (declared), the campaign cost file, EUR 1M budget
(I(x) reported). CELLS, all CERTIFIED (cap 500, 10 consecutive all-pass cycles), no post-certification step:
x = 0 at alpha in {0, 0.1, 0.25, 0.5, 1.0} (Addendum 40 ruling 1, as ordered) and the smallest node-7 unit
(0.25 MVA / 1.0 MWh, 2025) at alpha = 0.5. Concurrency 2, run by value in three PAIRS:
  pair 1 = {x0_a0p50, n7_4h_e1_a0p50};  pair 2 = {x0_a0p00, x0_a1p00};  pair 3 = {x0_a0p10, x0_a0p25}.

PATH: every evaluation runs through the production campaign harness (`p515_s44_campaign_harness`, by import;
extended in W64 ONLY by capture and per-cycle fields -- `evaluation_key` untouched) and hence
`p515_g_g1_g4_admm_gates.run_admm_arm`. New capture (spec v24 `new_capture`): the activation read-back and the
initialisation identity (at activation, before any ADMM-cycle solve, refusing), the per-cycle response record, the
terminal dual-based curtailment capture and the per-block coordination state (`response_terminal.json`).

HAZARD (spec v24 `hazard`): x0_a0p50 and n7_4h_e1_a0p50 have THE SAME evaluation keys as the SUPERSEDED pre-fix
pilot cells of `campaign_s52_pilot_nopersist` (the key does not carry the code change). This launcher never reads,
reuses or writes any file under a P515S52 campaign root; it snapshots their file metadata before and after every
run and refuses on any change; its campaign roots, eval dirs and working-dir ids are fresh (campaign ids
`s53_alpha_row`, `s53_alpha_row_smoke`).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-spec                           ZERO SOLVES. Writes the frozen spec v24 (write-once, named by its sha256).
  --stage {row,smoke} --freeze --scratch D  ZERO SOLVES. Freezes the campaign spec (harness `freeze_campaign_spec`),
                                          pinning spec v24, the instance, the ESS/cost files, this script.
  --stage smoke --run --spec-sha256 S     THE CAPTURE GATE (task W64): ONE 2-cycle x0 alpha = 0.5 evaluation run
                                          IN THIS PROCESS through the harness child path (`_child_real`: the gate
                                          hooks, the capture), under an armed bounded SolveProfileGuard declared
                                          before the run and verified EXACTLY; then the smoke checks (spec v24
                                          `smoke_gate`). Output: <smoke root>/smoke_gate.json + manifest.
  --stage row --run --pair N --spec-sha256 S   ONE pair (N = 1, 2, 3, in that order) at concurrency 2 through
                                          `H.evaluate` (the parent never solves: guard permitted=() verified 0).
                                          Pair 1 requires the smoke gate committed and PASS.
  --analyse                               ZERO SOLVES, after the three pairs: the spec v24 formulas -> the row.

EXACT COMMANDS (repo root):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py --freeze-spec \\
      > data/SRP1/Results/P515S53/alpha_row/freeze_spec_v24_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py --stage smoke \\
      --freeze --scratch <dir outside the repo> > data/SRP1/Results/P515S53/alpha_row/smoke_freeze_launch.log 2>&1
  (and --stage row --freeze ... > .../row_freeze_launch.log)
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py \\
      --stage smoke --run --spec-sha256 <sha> > data/SRP1/Results/P515S53/alpha_row/smoke_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py \\
      --stage row --run --pair 1 --spec-sha256 <sha> > data/SRP1/Results/P515S53/alpha_row/row_pair1_launch.log 2>&1
Exit codes (--run): 0 all clean (and, for a pair, every point certified); 2 a non-certified point (harness clean);
1 harness / guard / capture / precondition failure.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib-only at import (no model code)

# The thread caps every campaign child runs with, in force in THIS process too before any model import (the smoke
# gate runs the child path in-process; for the parent modes they are what the children get anyway).
os.environ.update(H.THREAD_CAP_ENV)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W64 alpha-row launcher (no solve outside the smoke '
                                                     'gate)').install()

from pyomo.opt.base.solvers import OptSolver  # noqa: E402
from pyomo.opt.solver.shellcmd import SystemCallSolver  # noqa: E402

_GUARD_STATE = (OptSolver.solve, SystemCallSolver._execute_command)
import p515_s52_pilot_campaign as P52  # noqa: E402 -- the pilot instance and its helpers, BY IMPORT
P52.PARENT_GUARD.uninstall()   # its import-time guard; ours (installed first) stays the only armed guard
if (OptSolver.solve, SystemCallSolver._execute_command) != _GUARD_STATE or any(P52.PARENT_GUARD.counts.values()):
    raise RuntimeError('importing p515_s52_pilot_campaign left a guard armed or counted')

STAGE_TEXT = ('P5.15 Addendum 40 ruling 1 (W64) -- the R3.6 alpha row on the 2 x 2 pilot instance: x = 0 at alpha in '
              '{0, 0.1, 0.25, 0.5, 1.0} + the smallest node-7 unit at alpha = 0.5, all certified; a two-mechanism '
              'cost-of-commitment curve')
SCRIPT_NAME = os.path.basename(__file__)
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ALPHA_ROOT_REL = os.path.join(_P53, 'alpha_row')
SPEC_V24_PREFIX = 'frozen_s53_spec_v24_'
SPEC_V23 = {'path': os.path.join(_P53, 'frozen_s53_spec_v23_39a07fd8.json'),
            'sha256': '39a07fd85bd02aaee0f4042d9a7b201cf4df9027191ff70d7939b124061f685c'}
INSTANCE = {'label': P52.INSTANCE_LABEL, 'case_path': P52.INSTANCE_CASE_REL, 'case_sha256': P52.EXPECTED_CASE_SHA256,
            'scenario_checksum': P52.EXPECTED_SCENARIO_CHECKSUM, 'record_path': P52.INSTANCE_RECORD_REL,
            'record_sha256': 'af539cd13d0fde164c5328b8944e903a46bed177a023f8af985288c8df5afb81'}
LABEL = P52.LABEL
YEAR = P52.YEAR
ARM_LABEL = P52.ARM_LABEL
CASE_FILE_AA = dict(P52.CASE_FILE_AA)
ESS_AGEING_BASELINE = json.loads(json.dumps(P52.ESS_AGEING_BASELINE))
REQUIRED_CONSECUTIVE_CYCLES = 10
SOLVES_PER_CYCLE = 83
EXPECTED_DSO_BLOCKS = 60
EXPECTED_TSO_BLOCKS = 20

# ---- the cells ---------------------------------------------------------------------------------------------------
ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)   # Addendum 40 ruling 1, as ordered -- NOT to be changed
UNIT_ALPHA = 0.5
X0 = P52.POINT_NODES['x0']
UNIT = P52.POINT_NODES['n7_4h_e1']


def alpha_tag(alpha):
    return 'a' + f'{alpha:.2f}'.replace('.', 'p')


CELLS = {f'x0_{alpha_tag(a)}': {'nodes': X0, 'alpha': a, 'point': 'x0'} for a in ALPHA_GRID}
CELLS[f'n7_4h_e1_{alpha_tag(UNIT_ALPHA)}'] = {'nodes': UNIT, 'alpha': UNIT_ALPHA, 'point': 'n7_4h_e1'}
PAIRS = {1: ('x0_a0p50', 'n7_4h_e1_a0p50'), 2: ('x0_a0p00', 'x0_a1p00'), 3: ('x0_a0p10', 'x0_a0p25')}
PAIR_RATIONALE = {1: 'by value: the alpha = 0.5 pair restates the pilot pair and R on the fixed initialisation',
                  2: 'by value: the two ends of the grid bound the curve (free deviation, strongest premium)',
                  3: 'by value: the interior of the arbitrage-collapse region'}
STAGES = {
    'row': {'campaign_id': 's53_alpha_row', 'cap': 500, 'concurrency': 2,
            'points': tuple(label for n in sorted(PAIRS) for label in PAIRS[n])},
    'smoke': {'campaign_id': 's53_alpha_row_smoke', 'cap': 2, 'concurrency': 1, 'points': ('x0_a0p50',)},
}
SMOKE_ROUNDS = STAGES['smoke']['cap'] + 1   # the initialisation round + 2 ADMM cycles

# ---- the hazard --------------------------------------------------------------------------------------------------
_P52_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S52')
PROTECTED_ROOTS = tuple(os.path.join(_P52_ROOT, d) for d in (
    'campaign_s52_pilot_nopersist', 'campaign_s52_pilot_repro_nopersist', 'campaign_s52_pilot',
    'campaign_s52_pilot_repro'))
SUPERSEDED_KEYS = {   # the committed s52 spec's eval keys (08790b4d), recorded here so the run never reads that root
    'x0_a0p50': {'superseded_label': 'x0', 'eval_key': '7d53b6f21b686a44db69f0901e03c870d14e15d00bfaaa6a5379822890918a94',
                 'eval_dir': '7d53b6f21b686a44_x0',
                 'working_dir_ids': {'precheck': 'p515s44_s52_pilot_nopersist_7d53b6f21b686a44_precheck',
                                     'run': 'p515s44_s52_pilot_nopersist_7d53b6f21b686a44_run'}},
    'n7_4h_e1_a0p50': {'superseded_label': 'n7_4h_e1',
                       'eval_key': '711fce9aa74d687892b994e332472c7d59745ef74ce9030abc12d54a9aaf83b4',
                       'eval_dir': '711fce9aa74d6878_n7_4h_e1',
                       'working_dir_ids': {'precheck': 'p515s44_s52_pilot_nopersist_711fce9aa74d6878_precheck',
                                           'run': 'p515s44_s52_pilot_nopersist_711fce9aa74d6878_run'}},
}
SUPERSEDED_SPEC = {'path': os.path.join(_P52_ROOT, 'campaign_s52_pilot_nopersist',
                                        'campaign_spec_s52_pilot_nopersist_08790b4d.json'),
                   'sha256': '08790b4db71a8161ac968479f3547f7908942a910dc387e33b93018161e45f54',
                   'note': 'recorded, never read by this launcher'}

# ---- references for the unit restatement (numbers from committed results, recorded) ------------------------------
PILOT_VALUE_EUR = 244321.29150229692          # s52 pilot (f239ee02): 841,964,496.83 - 841,720,175.54
PILOT_R_RATIO = 0.942
R_PREDICTION = P52.R_PREDICTION
SRP1_VALUE_PER_MWH = P52.SRP1_VALUE_PER_MWH_EXPECTED
SRP1_RESOLUTION = P52.SRP1_RESOLUTION_EXPECTED
PILOT_REPRO_PEAK_RSS_BYTES = 12855246848   # s52 repro (bb52aeaf): same configuration, 2 cycles, the smoke's twin

OBJECTIVE_CONVENTION = P52.OBJECTIVE_CONVENTION
EXTRA_CLEAN_FILES = (SCRIPT_NAME, 'p515_s52_pilot_campaign.py', 'p515_s44_scale_measurement.py',
                     'p515_s51_2x2_limit_gate.py', 'p515_s51_single_block_ab.py', 'p515_s51_coordinated_decomposition.py',
                     'p515_s53_curtailment_audit.py', 'p513_solve_profile_guard.py', 'p514_n_instrumented_cstar.py',
                     'p58_rescale.py', 'helper_functions.py', 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py', H.ESS_PARAMS_FILE_REL, P52.COST_FILE['path'], P52.SOURCE_CASE_REL,
                     P52.INSTANCE_CASE_REL, P52.INSTANCE_RECORD_REL)
OWN_PROCESS_SUBSTRING = 'p515_s53_'
LOCK_FAILURE_PREFIXES = ('legacy one-run lock exists', 'campaign lock exists')
GIB = 1 << 30


# ======================================================================================================================
#  THE FROZEN SPEC v24 (content; `--freeze-spec` writes it, named by its own sha256)
# ======================================================================================================================
def spec_v24_content():
    return {
        'schema': 'p515_frozen_spec_v24',
        'version': 24,
        'stage': 'P5.15 Addendum 40 ruling 1 -- the R3.6 alpha row (settled, 2 x 2) + the unit at alpha = 0.5',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 40 rulings 1 and 2',
                      'frozen spec v23 ruling1_alpha_row / ruling2_init_fix', 'Planner task W64'],
        'predecessor': dict(SPEC_V23),
        'purpose': {
            'statement': (
                'The row is NOT a search for a coordinated alpha*. Independent review established there is no single '
                'threshold in range: the committed alpha_arb distribution (price spread / 2 pibar, instance data) has '
                'median 0.022, p90 0.157, max 0.668, so market-arbitrage suppression is essentially complete by '
                'alpha ~ 0.25; while the curtailment mechanism needs alpha of order 4+ to hold schedule where '
                'flexibility is exhausted. The row is a TWO-MECHANISM COST-OF-COMMITMENT CURVE: arbitrage collapse at '
                'low alpha, physical-waste growth at high alpha.'),
            'alpha_arb_recomputed_at_freeze': 'the campaign spec freeze recomputes the distribution from the '
                                              'instance (extra.instance_facts.alpha_arb) and records it beside these',
        },
        'instance': {**INSTANCE, 'structure': '5 representative years (2025/28/31/34/37, 3-year blocks) x 4 days x 2 '
                                              'market x 2 operation', 'ageing': LABEL, 'aa': CASE_FILE_AA,
                     'cost_file': P52.COST_FILE, 'budget_eur': P52.BUDGET_EUR, 'premium_floor': None},
        'cells': {
            'alpha_grid_as_ordered': list(ALPHA_GRID),
            'cells': {label: {'point': c['point'], 'alpha': c['alpha'],
                              'nodes': {str(n): list(v) for n, v in c['nodes'].items()}, 'investment_year': YEAR}
                      for label, c in CELLS.items()},
            'all_certified': True, 'cap': STAGES['row']['cap'], 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES,
            'post_certification': None,
            'post_certification_note': ('none: the ruling asks for certified cells; the hull polish is not requested '
                                        '(Worker question to the Planner)'),
            'concurrency': STAGES['row']['concurrency'],
            'run_order_pairs': {str(n): list(v) for n, v in PAIRS.items()}, 'pair_rationale': PAIR_RATIONALE,
        },
        'objective_convention': OBJECTIVE_CONVENTION,
        'formulas': FORMULAS,
        'reporting_rules': [
            'Q(alpha) is reported ONLY beside its tangent bounds (the envelope), never as a bare system-cost curve.',
            'Every dispersion share is reported with its absolute value and its denominator (the mean committed flow '
            'moves with alpha).',
            'Every difference carries its resolution (bar sum); a difference inside it is INDETERMINATE.',
            'The rule-ten terminal-step-to-threshold ratio is reported for every cell.',
            'State the objective convention on every table.'],
        'new_capture': NEW_CAPTURE,
        'pre_run_assertions': PRE_RUN_ASSERTIONS,
        'hazard': {
            'statement': ('the alpha = 0.5 x = 0 cell (and the alpha = 0.5 unit) have the SAME evaluation key as the '
                          'superseded pre-fix pilot cells in campaign_s52_pilot_nopersist: the key identifies candidate '
                          'x configuration, not code, and the init fix changed code only. The launcher must refuse to '
                          'consult, reuse or write into any artifact of that campaign.'),
            'protected_roots': list(PROTECTED_ROOTS), 'superseded_keys': SUPERSEDED_KEYS,
            'superseded_spec': SUPERSEDED_SPEC,
            'enforcement': [
                'fresh campaign ids (s53_alpha_row, s53_alpha_row_smoke) -> fresh roots, eval dirs and working-dir ids '
                '(asserted different from the superseded ones)',
                'no post_certification.reference (nothing is read from another evaluation)',
                'the harness evaluate() writes only under its own campaign root (asserted on its source)',
                'the file metadata (relpath, size, mtime_ns) of every protected root is snapshotted before and after '
                'every run: any change fails the run; no file content under a protected root is read',
                'the eval-key identity is ASSERTED (the hazard is known, not accidental): x0_a0p50 == s52 x0, '
                'n7_4h_e1_a0p50 == s52 n7_4h_e1'],
        },
        'smoke_gate': SMOKE_GATE,
        'memory': {
            'rule': P52.memory_rule(2), 'required_gib_row': P52.memory_required_bytes(2) / GIB,
            'required_gib_smoke': P52.memory_required_bytes(1) / GIB,
            'claim_to_verify': ('the new capture (1: dual-based curtailment, 2: coordination state) is a few thousand '
                                'rows of ~25 floats, under 10 MB, no pickling, so the 18 GiB budget at concurrency 2 '
                                'stands'),
            'how_verified': ('the smoke gate MEASURES the capture: response_terminal.json size, RSS before/after the '
                             'capture (psutil) and ru_maxrss before/after, per-cycle capture RSS and time; and the '
                             'smoke process peak against the s52 repro twin (same configuration, 2 cycles, '
                             f'{PILOT_REPRO_PEAK_RSS_BYTES} bytes)'),
        },
        'predictions_recorded_before_run': PREDICTIONS,
        'stop': 'DO NOT LAUNCH the row in W64. After the row: STOP FOR REVIEW with the 3x3 prediction attached (v23).',
    }


FORMULAS = {
    'notation': ('b = DSO (node, year, day) block; w_b = admm_block_weight (_get_admm_block_weight: years x days x '
                 'annualization -- the Q weighting); s = (s_m, s_o), omega_s = prob_market[s_m] * prob_operation[s_o]; '
                 't = hour; B_b = baseMVA; pibar_t = sum_m omega_m pi_{m,t} (model_construction_helpers.'
                 'expected_market_price; = row18_premium, no floor); d_p[s,t] = p_int[s,t] - pbar_t (MW), d_q likewise '
                 '(MVAr) -- production _get_local_interface_dispersion per_scenario, captured in multiscenario_terminal'
                 '.json blocks[b].dispersion.per_scenario_d; all money EUR, horizon-weighted unless stated.'),
    'Q': 'Q(alpha) = certified_cost = gross_operational_cost (objective convention above), the cell\'s certified value',
    'charge': ('charge(alpha) = sum_b w_b row18_deviation_charge_b = multiscenario_terminal summary.all_dso.'
               'row18_charge_weighted; row18_deviation_charge_b = sum_{s,t} omega_s alpha pibar_t B_b (dp+ + dp- + dq+ '
               '+ dq-) (add_scenario_commitment_terms); 0 at alpha = 0 (row 18 not constructed)'),
    'C': 'C(alpha) = Q(alpha) - charge(alpha): the comparable system-cost curve; must be NON-DECREASING in alpha',
    'VP': ('VP(alpha) = recourse_components.voltage_pin_total (weighted): the interface-voltage pin, in the SOLVER '
           'objective, excluded from Q'),
    'V': 'V(alpha) = Q(alpha) + VP(alpha): the value of the solver\'s system problem (the function the envelope is about)',
    'A': 'A(alpha) = V(alpha) - charge(alpha) = C(alpha) + VP(alpha): non-decreasing at global optima (the rigorous form)',
    'P_charge': 'P(alpha) = charge(alpha) / alpha for alpha > 0: the pibar-weighted deviation volume (EUR per unit alpha)',
    'P_posthoc': ('P(alpha) |d| form, POST HOC from captured data (the alpha = 0 cell\'s P, so it sits on the same axis): '
                  'P = sum_b w_b sum_s omega_s sum_t pibar_t (|d_p[s,t]| + |d_q[s,t]|), computed by '
                  'p515_s44_campaign_harness.p_posthoc_block from multiscenario_terminal.json blocks[b].dispersion '
                  '{per_scenario_d, probabilities, pibar_by_hour} and blocks[b].admm_block_weight. At alpha > 0 it is '
                  'the minimal-split form of charge / alpha (d+ + d- >= |d|, equal at an exact minimal split): both are '
                  'reported, P_charge is the one used for alpha > 0.'),
    'P_used': 'P(0) = P_posthoc(0); P(alpha > 0) = P_charge(alpha)',
    'envelope': {
        'inequality': ('for adjacent alpha1 < alpha2: (alpha2 - alpha1) P(alpha2) <= V(alpha2) - V(alpha1) <= '
                       '(alpha2 - alpha1) P(alpha1)  (V = Q + VP: the voltage pin is in the solver objective and '
                       'excluded from Q, so it enters the check; the Q-only form dQ is reported beside it)'),
        'validity': ('holds at GLOBAL optima only (V(alpha) = min_x [A_x + alpha P_x] is concave with supergradient P). '
                     'IPOPT is local and ADMM stops at a bar, so a violation WITHIN resolution is "consistent", not a '
                     'proof; a violation BEYOND the bars means the two cells are not comparable optima.'),
        'violation': 'viol = max(LB - dV, dV - UB, 0)',
        'resolution': ('res = bar(alpha1) + bar(alpha2), each the cell\'s record bar (max |gross step| over the last 10 '
                       'cycles). Stated limitation: the bar bounds stopping slack of Q, not of VP or P.'),
        'verdict': '"within" if viol == 0; "consistent" if 0 < viol <= res; "not comparable optima" if viol > res',
    },
    'monotonicity': ('C and A non-decreasing, P non-increasing, each checked per adjacent pair against res (a decrease '
                     'of C beyond res, or an increase of P beyond the P-step equivalent res / (alpha2 - alpha1), is a '
                     'finding)'),
    'dispersion': {
        'E_abs_d': 'sum_b [w_b] sum_s omega_s sum_t |d_p| (MWh), weighted and unweighted (all_dso summaries)',
        'sum_omega_d2': 'sum_b [w_b] sum_s omega_s sum_t d_p^2 (MW^2 h), weighted and unweighted',
        'peak_block_rms': 'max_b rms_mw_b (rms_mw_b = sqrt(sum_s omega_s sum_t d_p^2 / n_hours))',
        'peak_abs_d': 'max_b max_abs_mw_b',
        'worst_block_share': ('max_b rms_mw_b / mean_abs_committed_flow_mw_b, reported WITH that block\'s rms_mw and '
                              'mean_abs_committed_flow_mw (absolute), and the max-absolute-RMS block with its share'),
    },
    'split': ('market / operation parts of sum omega d^2: p515_s51_coordinated_decomposition.block_decomposition '
              '(W46 committed definitions: market_part = sum_t sum_m omega_m dbar_{m,t}^2, operation_part = sum_t '
              'sum_s omega_s (d - dbar_{m(s)})^2) per DSO block, aggregated by .aggregate (unweighted, the W46 '
              'convention) and also weighted by w_b'),
    'covariance': ('the earned price-deviation covariance = production interface_settlement_deviation_total '
                   '(recourse components, weighted; negative = earned), and the W46 formula sum_t sum_m omega_m (pi_m '
                   '- pibar) dbar_m (unweighted and weighted) beside it'),
    'curtailment': ('per network / hour / scenario: c[g,s,t] = (pg_avail - pg) B (production definitional); V_plus = '
                    'sum_g max(c, 0); priced at pi_s: sum_s omega_s sum_t pi_{m(s),t} V_plus[s,t]; weighted by w_b per '
                    'network (response_terminal.json curtailment_by_block / summary.curtailment_per_network); entries '
                    'above TOL = EQUALITY_TOLERANCE x B carry the dual-based cause (W53 audit classes)'),
    'row18_condition_entries': ('DN entries with d_p[s,t] <= +D_TOL_MW (the audit constant) and pi_{m(s),t} < alpha '
                                'pibar_t: count and volume per cell'),
    'flexibility': 'DSO flexibility legs P up / P down / Q up / Q down (omega-weighted, w_b-weighted), per DSO',
    'Q_leg_share': 'charge_q / (charge_p + charge_q), legs from row18_dev_q_* / row18_dev_p_* at the charge coefficients',
    'unit_value': ('value = Q(x0, 0.5) - Q(unit, 0.5); resolution = bar(x0) + bar(unit); value per MWh = value / 1.0; '
                   'ratio to SRP1 (259,427.77 EUR/MWh, res 34,734.63) against R = 0.937 and the pilot\'s 0.942; value '
                   'against the pilot\'s 244,321.29'),
    'code': ('p515_s53_alpha_row_campaign.cell_quantities (per cell) and .row_analysis (across cells); '
             'p515_s44_campaign_harness.p_posthoc_block (the P |d| form)'),
}

NEW_CAPTURE = {
    '1_dual_based_curtailment': (
        'response_terminal.json curtailment_entries: every curtaillable generator-hour-scenario with c > TOL, read at '
        'the TERMINAL (in-memory models, before the workbook and any post-certification step): pg, pg_avail, qg, '
        'sg_avail, sg, sg_capability slack and dual, class (capability_bound / interior, the audit\'s CAP_SLACK_TOL), '
        'zU(pg), the node-balance dual at the generator bus and at the reference bus (raw, and / (B omega_s) as '
        'EUR/MWh), the scenario price, d_p[s,t], d+, d-, the dual of row18_dev_p_def[s,t], zL(d+), zL(d-), the premium '
        'and alpha, the active voltage / branch rows of that network-hour (network_hour_rows, the audit\'s '
        '_network_hour_rows with no transformer set), and the IPOPT termination of the block\'s last solve. Helpers '
        'and tolerances BY IMPORT from p515_s53_curtailment_audit (guard disarmed at import).'),
    '2_coordination_state': (
        'response_terminal.json coordination_by_dso_block: the W44 p515_s51_2x2_limit_gate.coordination_record BY '
        'IMPORT per DSO block (dual_pf_p/q_req as set for the last solve, rho_pf, the TSO request p/q_pf_req, pbar, '
        'the effective objective scale, lambda_AL, penalty_gen_curtailment, interface_settlement_weight, settlement '
        'parts, per-scenario curtailment) + the block\'s termination'),
    '3_per_cycle_response': (
        'per_cycle_response.jsonl, appended every cycle (survives a crash), merged by cycle into per_cycle_record.'
        'jsonl: ' + ', '.join(H.PER_CYCLE_RESPONSE_FIELDS)),
    'also': ('activation_readback.json and initialisation_identity.json (+ <root>/init_identity/<eval dir>.json); '
             'row 18 charge by leg with the zL stationarity sanity identity; DSO flexibility legs; the P |d| form per '
             'block; voltage_pin_total'),
    'scope': 'derived-instance / premium evaluations only; every other evaluation runs exactly as before',
}

PRE_RUN_ASSERTIONS = {
    'before_any_solve_in_the_parent': [
        'campaign spec checks (entries, alphas, cap, concurrency, configuration, instance, no reference, eval keys '
        'recompute; x0_a0p50 / n7_4h_e1_a0p50 keys equal the superseded ones and every other key differs)',
        'pins (spec v24/v23, instance, ESS params, cost file, sigma check) tracked, clean, hashing as pinned',
        'rule eleven: H.assert_record_capture_paths + H.assert_alpha_row_capture_paths + this launcher\'s checklist '
        '(every spec-required quantity incl. the P(0) post-hoc formula and its inputs)',
        'memory preflight at the stage concurrency (P52 rule, refusing)',
        'declared solve profile 83 per cycle (from the instance, at freeze)',
        'fresh campaign root / eval dirs / working-dir ids; protected roots snapshotted',
        'pair order: pair N requires pairs < N complete with verified manifests; pair 1 requires the smoke gate '
        'committed and PASS'],
    'before_any_solve_in_the_child': ['H.assert_alpha_row_capture_paths (rule eleven, fail fast)',
                                      'the configuration hook (premium alpha applied and read back, AA, ageing)'],
    'at_activation_before_any_ADMM_cycle_solve': [
        'alpha read-back on EVERY DSO block == the arm\'s alpha; every row 18 row active and no pair Var fixed where '
        'alpha > 0; row 18 ABSENT where alpha = 0; TSO carries no row 18',
        'penalty_gen_curtailment == 0 and interface_settlement_weight == 1 on EVERY block (TSO and DSO)',
        'the cycle-0 (initialisation) gross cost is BITWISE equal (float.hex) to every earlier record of the same '
        'candidate in <root>/init_identity (the smoke\'s x0 record is placed there before pair 1): the run-time form '
        'of the .nl identity. SCOPE: the five x = 0 cells (and the smoke); the unit is a different candidate and is '
        'compared with nothing.',
        'any failure RAISES in the child: the evaluation stops before its first ADMM-cycle solve'],
    'note': ('the activation-time assertions cannot precede the initialisation solve (activation reads its solution); '
             'they precede every ADMM-cycle solve'),
}

SMOKE_GATE = {
    'what': ('ONE 2-cycle x0 alpha = 0.5 evaluation (campaign s53_alpha_row_smoke, cap 2, concurrency 1, no '
             'post-certification) run in the launcher process through the harness child path H._child_real (the gate '
             'hooks s38/s39, run_admm_arm, the W47 multi-scenario capture, the W64 capture), attached, alone'),
    'declared_solve_profile': (f'SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED) armed around the whole child '
                               f'path; expected = {SOLVES_PER_CYCLE} x {SMOKE_ROUNDS} = {SOLVES_PER_CYCLE * SMOKE_ROUNDS} + '
                               'every retry ATTEMPTED (per-event reconciliation, W35); verified EXACTLY with '
                               'GUARD.verify(expected) (solves and process launches); an unsupported reconciliation '
                               'fails'),
    'checks': {
        'G1_guard_exact': 'GUARD.verify(expected) == [] and the child\'s own solve_profile identity holds',
        'G2_child_ran': ('evaluation_record status not_certified (cap 2) with cycles_run == 2, no child error; '
                         'multiscenario_terminal, operational_workbook and response_terminal all "written"'),
        'G3_activation_readback': ('all_ok; 60 DSO blocks each alpha_on_model == 0.5, 192 rows all active, 0 pair Vars '
                                   'fixed; 20 TSO blocks penalty 0 / weight 1 / no row 18'),
        'G4_init_identity': 'record written with a finite gross cost and its float.hex; compared with 0 records',
        'G5_per_cycle': ('2 response lines, both captured, cycles 1..2 match the trajectory, every numeric response '
                         'field finite; per_cycle_record.jsonl carries every PER_CYCLE_RECORD_FIELDS key'),
        'G6_row18_duals_nonempty_and_sane': (
            'some DSO curtailment entries carry the row 18 fields, all finite; the zL stationarity identity '
            '|zL(d+) + zL(d-)| = 2 omega alpha pibar B holds to rel 1e-4 on >= 99% of the checked indices and to 1e-2 '
            'on all, with 0 missing suffix values (n_checked = 2 x 96 x 60 = 11,520)'),
        'G7_coordination_nonempty_and_sane': ('60 DSO blocks, no capture error, 24 hours each, dual_pf_p_req / p_pf_req_mw '
                                              '/ pbar_mw finite, rho_pf > 0, admm_objective_scale > 0, some dual_pf_p_req '
                                              '!= 0'),
        'G8_curtailment_entries': ('n > 0; c_mw > tol, pg / pg_avail / qg / sg_avail finite, bus dual present, '
                                   'termination available on every entry (LMP quantiles reported, not gated)'),
        'G9_capture_cost': 'response_terminal.json < 10 MB; RSS after - before the capture < 1 GiB',
        'G10_per_cycle_cost': 'per-cycle capture < 60 s per cycle',
        'G11_eval_key_unchanged': ('x0_a0p50 key == the superseded s52 x0 key (computed by the pre-W64 harness) and the '
                                   'row spec\'s n7_4h_e1_a0p50 key == the s52 unit key'),
        'G12_hazard': 'protected-root metadata unchanged before / after',
        'G13_P_formulas': ('cell_quantities runs on the smoke eval dir; the P |d| form from multiscenario_terminal.json '
                           '(post hoc) equals the child-side live-model value to rel 1e-9; charge / alpha >= the |d| '
                           'form and exceeds it by <= 1e-3 relative'),
        'G14_rule_eleven': 'the child\'s alpha-row checklist all true',
        'G1b_no_solve_after_verify': 'the bounded guard\'s counts do not move while the (zero-solve) checks run',
    },
    'reported_not_gated': ['smoke process peak RSS vs the s52 repro twin', 'LMP quantiles', 'Q-leg share',
                           'curtailment counts by class', 'wall time'],
}

PREDICTIONS = {
    'planner_spec_v23_ruling1': [
        'at certified points the market-arbitrage part of sum omega d^2 falls monotonically with alpha and the earned '
        'covariance shrinks toward 0, as the 2-cycle sweep showed for that component',
        'the operation part does NOT fall to zero: it is physical recourse, so dispersion plateaus rather than '
        'vanishing; I expect no alpha in this range to reach the 1e-2 MW tolerance',
        'Q(0) rises with alpha, since the premium is a real cost inside Q: monotone increase across the row',
        'the non-monotonicity seen at 2 cycles does NOT survive certification - if it does, that is a finding about the '
        'formulation rather than the coordination state'],
    'planner_spec_v23_ruling2': [
        'the re-run unit value stays within 10,000 EUR (~4 %) of the pilot\'s 244,321',
        'the restated R ratio stays within 0.01 of 0.942'],
    'worker_smoke': {
        'W-S1': f'solves observed exactly {SOLVES_PER_CYCLE * SMOKE_ROUNDS}, 0 retries (the s52 repro twin: 249, 0)',
        'W-S2': 'activation read-back: 60/60 DSO blocks alpha 0.5 with 192 active rows and 0 fixed pair Vars; 20/20 TSO',
        'W-S3': 'initialisation gross cost finite; the smoke compares against 0 records',
        'W-S4': '2 per-cycle response lines, all fields finite; per-cycle capture 0.5-10 s per cycle',
        'W-S5': 'curtailment entries above tol: 1,500-4,500 (the converged pilot x0 had 2,932), >= 95 % DSO-side',
        'W-S6': 'zL identity at rel <= 1e-4 on >= 99 % of the 11,520 indices (about 70 % confident: acceptable-level '
                'terminations at tol 1e-4 could loosen it)',
        'W-S7': 'charge / alpha exceeds the |d| form of P by <= 1e-4 relative (interior-point complementarity)',
        'W-S8': 'coordination: 60 blocks; every rho_pf in {0.198, 0.132}; some dual_pf_p_req nonzero',
        'W-S9': 'response_terminal.json 2-8 MB; capture 10-120 s; RSS delta < 0.3 GiB',
        'W-S10': 'smoke process peak RSS 12.3-13.5 GB (twin 12.86 GB): the capture does not move the peak',
        'W-S11': 'eval keys equal the superseded s52 keys (certain by construction; the check proves evaluation_key '
                 'unchanged)',
        'W-S12': 'wall 15-25 min (twin 18.5 min)',
        'W-S13': 'Q-leg share of the charge at 2 cycles 5-40 % (low confidence)',
    },
    'worker_row': {
        'W-R1': 'all 6 cells certify inside cap 500 at 60-130 cycles each; each pair 4-6 h wall',
        'W-R2': 'the cycle-0 gross cost is bitwise equal across the five x = 0 cells and the smoke',
        'W-R3': 'P non-increasing: P(0) > P(0.1) >= P(0.25) >= P(0.5) >= P(1.0); P(0.5) ~ 40 M EUR (pilot charge '
                '20.07 M / 0.5), P(0) 1.5-4 x P(0.5)',
        'W-R4': 'C and A non-decreasing within resolution; every adjacent envelope "within" or "consistent"',
        'W-R5': 'market part at alpha >= 0.25 <= 5 % of its alpha = 0 value; operation part within +-30 % across the '
                'row',
        'W-R6': 'earned covariance magnitude falls >= 10x from alpha = 0 to alpha = 0.25',
        'W-R7': 'DSO curtailed RES (V_plus, weighted) larger at alpha = 1.0 than at alpha = 0 beyond resolution of the '
                'priced value; row-18-condition entries grow from alpha 0.25 to 1.0',
        'W-R8': 'V(0.5) - V(0) in [15 M, 60 M] EUR (envelope lower bound 0.5 P(0.5) ~ 20 M)',
        'W-R9': 'Q-leg share of the charge at alpha = 0.5 certified: 2-25 %',
        'W-R10': 'unit value within 10,000 of 244,321 and R within 0.01 of 0.942 (agreeing with the Planner)',
    },
}


# ======================================================================================================================
#  utilities
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W64] {msg}', flush=True)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _git_state(rel):
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
    return tracked, not dirty


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def campaign_root(stage):
    return os.path.join(REPO, ALPHA_ROOT_REL, f'campaign_{STAGES[stage]["campaign_id"]}')


def _key_of(nodes):
    return H.candidate_key(H.canonical_candidate(nodes, investment_year=YEAR))


def _premium(alpha):
    return {'alpha': float(alpha), 'floor': None}


def _eval_key(label, derived):
    c = CELLS[label]
    return H.evaluation_key(_key_of(c['nodes']), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
                            derived_instance=derived, interface_deviation_premium=_premium(c['alpha']))


def derived_declaration():
    """The pilot instance's declaration, from its committed instance record (P515S52/pilot_instance -- the instance,
    not a campaign root), re-validated by the harness."""
    rec = _load(INSTANCE['record_path'])
    return H.validate_derived_instance(rec['derived_instance_declaration'])


def protected_snapshot():
    """relpath -> [size, mtime_ns] of every file under every protected root (metadata only; no content is read)."""
    out = {}
    for rel in PROTECTED_ROOTS:
        root = os.path.join(REPO, rel)
        for directory, _dirs, files in os.walk(root):
            for fname in files:
                path = os.path.join(directory, fname)
                st = os.stat(path)
                out[os.path.relpath(path, REPO)] = [st.st_size, st.st_mtime_ns]
    return out


def own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    lines = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout.splitlines()
    hits = []
    for line in lines:
        fields = line.split()
        if len(fields) > 1 and fields[1] in excluded:
            continue
        if OWN_PROCESS_SUBSTRING in line and 'python' in line:
            hits.append(line.strip())
    return hits


# ======================================================================================================================
#  pins and checks (zero solves)
# ======================================================================================================================
def find_spec_v24():
    hits = sorted(f for f in os.listdir(os.path.join(REPO, _P53)) if f.startswith(SPEC_V24_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one {SPEC_V24_PREFIX}*.json in {_P53}, found {hits}')
    rel = os.path.join(_P53, hits[0])
    sha = H.sha256_file(os.path.join(REPO, rel))
    if not hits[0].startswith(f'{SPEC_V24_PREFIX}{sha[:8]}'):
        raise RuntimeError(f'{rel} is not named by its own sha256 ({sha[:8]})')
    content = _load(rel)
    expected = json.loads(json.dumps(spec_v24_content(), default=str))
    stripped = {k: v for k, v in content.items() if k not in ('frozen_utc', 'git_head_at_freeze')}
    if stripped != expected:
        diff = sorted(k for k in set(stripped) | set(expected) if stripped.get(k) != expected.get(k))
        raise RuntimeError(f'{rel} content differs from this launcher\'s spec v24 in {diff}')
    return {'path': rel, 'sha256': sha}


def check_pins(require_v24=True):
    out, failures = {}, []
    pins = [('spec_v23', SPEC_V23), ('ess_params_file', P52.ESS_PARAMS_FILE), ('cost_file', P52.COST_FILE),
            ('instance_case', {'path': INSTANCE['case_path'], 'sha256': INSTANCE['case_sha256']}),
            ('instance_record', {'path': INSTANCE['record_path'], 'sha256': INSTANCE['record_sha256']})]
    if require_v24:
        try:
            pins.append(('spec_v24', find_spec_v24()))
        except RuntimeError as error:
            failures.append(f'spec v24: {error}')
    for name, pin in pins:
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        tracked, clean = _git_state(pin['path'])
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256'], 'git_tracked': tracked, 'git_clean': clean}
        if not (got == pin['sha256'] and tracked and clean):
            failures.append(f'pin {name}: {out[name]}')
    sig = P52.SIGMA_CHECK
    tracked, clean = _git_state(sig['path'])
    out['sigma_check'] = {'path': sig['path'], 'commit': sig['commit'], 'git_tracked': tracked, 'git_clean': clean,
                          'commit_in_HEAD': P52._commit_in_head(sig['commit'])}
    if not (tracked and clean and out['sigma_check']['commit_in_HEAD']):
        failures.append(f"pin sigma_check: {out['sigma_check']}")
    return out, failures


def launcher_checklist():
    """RULE ELEVEN for this launcher: a capture path exists for EVERY spec v24 quantity, asserted on the code that
    will run -- the harness record, the W47 multi-scenario capture, the W64 capture, and this script's formulas."""
    import inspect
    record_checks = H.assert_record_capture_paths()
    post_checks = H.assert_post_certification_capture_paths()
    alpha_checks = H.assert_alpha_row_capture_paths()
    ms_src = inspect.getsource(H.multiscenario_terminal_capture)
    resp_src = inspect.getsource(H.response_terminal_capture)
    cell_src = inspect.getsource(cell_quantities)
    row_src = inspect.getsource(row_analysis)
    child_src = inspect.getsource(H._child_real)
    checks = {
        'path_is_run_admm_arm': 'G.run_admm_arm(' in child_src,
        'Q_certified_cost': "'certified_cost': report.get('gross_operational_cost') if certified" in inspect.getsource(
            H.build_evaluation_record),
        'charge_weighted_captured': "'row18_charge_weighted'" in ms_src,
        'voltage_pin_total_captured': "'voltage_pin_total'" in ms_src and "'voltage_pin_total'" in resp_src,
        'P0_posthoc_inputs_captured': (all(t in ms_src for t in ("'per_scenario_d'", "'pibar_by_hour'",
                                                                 "'admm_block_weight': weight", '**metrics'))
                                       and "'probabilities': probs" in inspect.getsource(H._dispersion_block_metrics)),
        'P0_posthoc_formula_applied_in_cell': 'H.p_posthoc_block(' in cell_src,
        'P_charge_in_cell': "charge / alpha" in cell_src,
        'envelope_in_row': all(t in row_src for t in ('lb =', 'ub =', 'viol =', "'not comparable optima'")),
        'C_A_monotonicity_in_row': "'C_non_decreasing'" in row_src and "'A_non_decreasing'" in row_src,
        'dispersion_metrics_in_cell': all(t in cell_src for t in (
            "'E_abs_d_p_mwh_weighted'", "'sum_omega_d2_p_mw2h_weighted'", "'peak_block_rms_mw'", "'peak_abs_d_mw'",
            "'worst_block_share'", "'mean_abs_committed_flow_mw'")),
        'split_by_w46_definitions': 'DEC.block_decomposition(' in cell_src and 'DEC.aggregate(' in cell_src,
        'covariance_in_cell': "'covariance_production_weighted'" in cell_src,
        'curtailment_in_capture': "'curtailment_per_network'" in resp_src and "'priced_eur'" in resp_src,
        'row18_duals_in_capture': all(t in inspect.getsource(H._curtailment_block) for t in (
            "'row18_dev_p_def_dual_raw'", "'zL_d_up_raw'", "'zL_d_down_raw'", "'lmp_ref_bus_dual_raw'",
            "'termination_last_solve'", "'sg_capability_dual_raw'", "'pg_zU_raw'")),
        'coordination_in_capture': "'coordination_by_dso_block'" in resp_src,
        'flexibility_in_capture': "'flexibility_per_dso_weighted'" in resp_src,
        'q_leg_share_in_capture': "'row18_q_leg_share_of_charge'" in resp_src,
        'per_cycle_response_fields': set(H.PER_CYCLE_RESPONSE_FIELDS) <= set(H.PER_CYCLE_RECORD_FIELDS),
        'unit_value_in_row': "'value_eur'" in row_src and "'ratio_to_srp1'" in row_src,
        'rule_ten_in_cell': "'terminal_step_over_threshold'" in cell_src,
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (W64 launcher): capture paths missing: {missing}')
    return {'launcher': checks, 'harness_record': record_checks, 'harness_post_certification': post_checks,
            'harness_alpha_row': alpha_checks}


def instance_facts(scratch):
    """ZERO SOLVES: the pilot instance read with production's reader (scratch-redirected plots): checksum,
    dimensions, block weights, solves per cycle, min hourly mean price, and the alpha_arb distribution (instance
    data, the W46 definition) beside the Planner's numbers."""
    import numpy as np
    import p515_s44_scale_measurement as S
    import shared_resources_planning as srp
    import model_construction_helpers as MCH
    DEC = H.import_disarmed_diagnostic(H.DECOMPOSITION_MODULE)
    planning, read_dir = P52._read(INSTANCE['case_path'], scratch, 'w64')
    facts = {'scenario_checksum': planning.scenario_metadata['combined_scenario_checksum'],
             'planning_read_redirected_to': read_dir, 'planning_dimensions': S.planning_dimensions(planning),
             'expected_block_counts': S.expected_block_counts(planning),
             'declared_solve_profile_per_cycle': S.declared_solve_profile(planning, 1)['solves_per_cycle']}
    tn = planning.transmission_network
    facts['block_weights_tso'] = {f'{y}|{d}': srp._get_admm_block_weight(tn, y, d) for y in tn.years for d in tn.days}
    facts['n_dso_blocks'] = sum(len(dn.years) * len(dn.days) for dn in planning.distribution_networks.values())
    facts['n_tso_blocks'] = len(tn.years) * len(tn.days)
    pib, arb = [], []
    for y in tn.years:
        for d in tn.days:
            net = tn.network[y][d]
            pibar = [float(MCH.expected_market_price(net, p)) for p in range(planning.num_instants)]
            pi = [[float(net.cost_energy_p[m][p]) for m in range(planning.num_market_scenarios)]
                  for p in range(planning.num_instants)]
            pib += pibar
            arb += [a for a in DEC.alpha_arb_by_hour(pi, pibar) if a is not None]
    facts['min_hourly_mean_price'] = min(pib)
    facts['premium_floor_needed'] = bool(min(pib) <= 0.0)
    facts['alpha_arb'] = {'n_hours': len(arb), 'median': float(np.median(arb)), 'p90': float(np.percentile(arb, 90)),
                          'max': float(max(arb)), 'planner_statement': {'median': 0.022, 'p90': 0.157, 'max': 0.668},
                          'definition': 'p515_s51_coordinated_decomposition.alpha_arb_by_hour on the TSO block prices '
                                        '(market prices are common to every network), per (year, day, hour)',
                          'fraction_of_hours_below': {str(a): float(np.mean([x < a for x in arb])) for a in ALPHA_GRID}}
    return facts


# ======================================================================================================================
#  --freeze-spec / --freeze
# ======================================================================================================================
def freeze_spec(started):
    got = H.sha256_file(os.path.join(REPO, SPEC_V23['path']))
    tracked, clean = _git_state(SPEC_V23['path'])
    if got != SPEC_V23['sha256'] or not (tracked and clean):
        raise SystemExit(f'predecessor spec v23 not as pinned: sha {got} tracked {tracked} clean {clean}')
    existing = [f for f in os.listdir(os.path.join(REPO, _P53)) if f.startswith(SPEC_V24_PREFIX)]
    if existing:
        raise SystemExit(f'spec v24 already frozen (write-once): {existing}')
    content = spec_v24_content()
    content['frozen_utc'] = _utc()
    content['git_head_at_freeze'] = H._git(['rev-parse', 'HEAD'])
    text = json.dumps(content, indent=1, default=str) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V24_PREFIX}{sha[:8]}.json')
    with open(os.path.join(REPO, rel), 'x') as handle:
        handle.write(text)
    if H.sha256_file(os.path.join(REPO, rel)) != sha:
        raise SystemExit('written spec does not hash to its name')
    check = find_spec_v24()
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'frozen spec v24: {rel} sha256={sha} (predecessor v23 {SPEC_V23["sha256"]}); re-read equal: '
         f'{check["sha256"] == sha}; guard {PARENT_GUARD.counts} verify0={guard_failures}; wall {time.time() - started:.1f}s')
    PARENT_GUARD.uninstall()
    sys.exit(0 if not guard_failures else 1)


def _spec_candidates(stage):
    return [(label, CELLS[label]['nodes'], {'investment_year': YEAR,
                                            'interface_deviation_premium': _premium(CELLS[label]['alpha'])})
            for label in STAGES[stage]['points']]


def validate_spec(stage, spec, derived):
    cfg = spec['configuration']
    entries = spec['candidates']
    extra = spec.get('extra') or {}
    st = STAGES[stage]
    checks = {
        'campaign_id': spec.get('campaign_id') == st['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == list(st['points']),
        'cap': spec.get('cap') == st['cap'], 'concurrency': spec.get('concurrency') == st['concurrency'],
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label': cfg.get('arm_label') == ARM_LABEL, 'no_campaign_overrides': cfg.get('overrides') == {},
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'derived_instance_declared': cfg.get('derived_instance') == derived,
        'derived_instance_is_the_pilot': ((cfg.get('derived_instance') or {}).get('case_sha256') == INSTANCE['case_sha256']
                                          and (cfg.get('derived_instance') or {}).get('scenario_checksum')
                                          == INSTANCE['scenario_checksum']),
        'no_model_variant': 'model_variant_label' not in spec and not any('model_variant' in e for e in entries),
        'no_flex_price_variant': 'flex_price_label' not in spec and not any('flex_price_multiplier' in e for e in entries),
        'no_post_certification': all(e.get('post_certification') is None for e in entries),
        'extra_stage_recorded': extra.get('stage') == stage,
        'alpha_grid_as_ordered': extra.get('alpha_grid_as_ordered') == list(ALPHA_GRID),
        'pairs_recorded': extra.get('pairs') == {str(n): list(v) for n, v in PAIRS.items()},
    }
    for e in entries:
        label = e['label']
        checks[f'{label}:canonical_key'] = e.get('key') == _key_of(CELLS[label]['nodes'])
        checks[f'{label}:eval_key_recomputes'] = e.get('eval_key') == _eval_key(label, derived)
        checks[f'{label}:premium'] = e.get('interface_deviation_premium') == _premium(CELLS[label]['alpha'])
        checks[f'{label}:no_overrides'] = e.get('overrides') == {}
        checks[f'{label}:effective_aa'] = e.get('effective_anderson_acceleration') == CASE_FILE_AA
        sup = SUPERSEDED_KEYS.get(label)
        if sup is not None:   # the hazard: the key IS the superseded one; the dirs and ids must NOT be
            checks[f'{label}:eval_key_equals_superseded_(hazard)'] = e.get('eval_key') == sup['eval_key']
            checks[f'{label}:working_dir_ids_differ_from_superseded'] = (
                not set(e['working_dir_ids'].values()) & set(sup['working_dir_ids'].values()))
        else:
            checks[f'{label}:eval_key_not_superseded'] = e.get('eval_key') not in {
                v['eval_key'] for v in SUPERSEDED_KEYS.values()}
    return checks


def hazard_static_checks():
    import inspect
    ev_src = inspect.getsource(H.evaluate)
    return {'evaluate_writes_only_under_ctx_roots': ('ctx.evals_root' in ev_src and 'RESULTS_ROOT' not in ev_src
                                                     and 'P515S52' not in ev_src),
            'child_reads_no_other_campaign_root': 'P515S52' not in inspect.getsource(H._child_real),
            'protected_roots_exist': all(os.path.isdir(os.path.join(REPO, r)) for r in PROTECTED_ROOTS),
            'own_roots_outside_protected': all(not campaign_root(s).startswith(os.path.join(REPO, _P52_ROOT))
                                               for s in STAGES)}


def freeze(stage, started, scratch):
    tag = f'W64-{stage.upper()}-FREEZE'
    root = campaign_root(stage)
    raw = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    lock_obs = [f for f in raw if f.startswith(LOCK_FAILURE_PREFIXES)]
    failures = [f for f in raw if f not in lock_obs]
    pins, more = check_pins()
    failures += more
    instance = P52.ensure_instance_file(write=False)
    if instance['case_sha256'] != INSTANCE['case_sha256']:
        failures.append(f"instance re-derived sha {instance['case_sha256']} != {INSTANCE['case_sha256']}")
    derived = derived_declaration()
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    facts = instance_facts(scratch)
    if facts['scenario_checksum'] != INSTANCE['scenario_checksum']:
        failures.append(f"scenario checksum {facts['scenario_checksum']}")
    if facts['declared_solve_profile_per_cycle'] != SOLVES_PER_CYCLE:
        failures.append(f"solves per cycle {facts['declared_solve_profile_per_cycle']} != {SOLVES_PER_CYCLE}")
    if facts['n_dso_blocks'] != EXPECTED_DSO_BLOCKS or facts['n_tso_blocks'] != EXPECTED_TSO_BLOCKS:
        failures.append(f"block counts {facts['n_dso_blocks']} / {facts['n_tso_blocks']}")
    if facts['premium_floor_needed']:
        failures.append('a non-positive mean hourly price exists: a premium floor would be required')
    hazard = hazard_static_checks()
    failures += [f'hazard static check failed: {k}' for k, v in hazard.items() if not v]
    memory = P52.memory_preflight(STAGES[stage]['concurrency'])
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    st = STAGES[stage]
    extra = {
        'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'stage': stage, 'stage_text': STAGE_TEXT, 'label': LABEL, 'spec_v24': pins['spec_v24'] if 'spec_v24' in pins
        else None, 'spec_v24_pin': find_spec_v24(), 'spec_v23': dict(SPEC_V23), 'pins': pins, 'instance': instance,
        'instance_facts': facts, 'alpha_grid_as_ordered': list(ALPHA_GRID),
        'pairs': {str(n): list(v) for n, v in PAIRS.items()},
        'cells': {label: {'alpha': CELLS[label]['alpha'], 'point': CELLS[label]['point'],
                          'candidate_key': _key_of(CELLS[label]['nodes']), 'eval_key': _eval_key(label, derived)}
                  for label in st['points']},
        'objective_convention': OBJECTIVE_CONVENTION, 'hazard_static_checks': hazard,
        'superseded_keys': SUPERSEDED_KEYS, 'protected_roots': list(PROTECTED_ROOTS),
        'memory_rule': P52.memory_rule(st['concurrency']), 'memory_at_freeze_non_gating': memory,
        'lock_observations_at_freeze_non_gating': lock_obs, 'rule_eleven_asserted_before_run': rule11,
    }
    if stage == 'smoke':
        extra['smoke_declared_solves_base'] = SOLVES_PER_CYCLE * SMOKE_ROUNDS
        extra['smoke_gate'] = SMOKE_GATE
    else:
        extra['smoke_campaign'] = {'campaign_id': STAGES['smoke']['campaign_id'],
                                   'root': os.path.relpath(campaign_root('smoke'), REPO),
                                   'required': 'smoke_gate.json committed, clean, pass true, before pair 1'}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, st['campaign_id'], _spec_candidates(stage),
        configuration={'name': (f'{LABEL}; the 2x2 pilot instance {INSTANCE["label"]}; row 18 premium per entry '
                                '(the alpha row); case-file AA declared'),
                       'arm_label': ARM_LABEL, 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': json.loads(json.dumps(ESS_AGEING_BASELINE)),
                       'ess_ageing_baseline_label': LABEL, 'derived_instance': derived,
                       'note': 'no overrides, no model variant, no flexibility-price variant, no post-certification'},
        cap=st['cap'], concurrency=st['concurrency'],
        authority=['PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 1', 'frozen spec v24 (this stage)', 'Planner task W64'],
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES, extra=extra)
    checks = validate_spec(stage, spec, derived)
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for label in st['points']:
        c = extra['cells'][label]
        _log(f"[{tag}]   {label}: alpha={c['alpha']} key={c['candidate_key'][:16]} eval_key={c['eval_key'][:16]}")
    _log(f"[{tag}] instance facts: checksum {facts['scenario_checksum'][:16]} blocks {facts['expected_block_counts']} "
         f"solves/cycle {facts['declared_solve_profile_per_cycle']} min pibar {facts['min_hourly_mean_price']:.4f} "
         f"alpha_arb {facts['alpha_arb']['median']:.4f}/{facts['alpha_arb']['p90']:.4f}/{facts['alpha_arb']['max']:.4f} "
         f"(Planner 0.022/0.157/0.668)")
    _log(f'[{tag}] spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f'[{tag}] hazard static {hazard}')
    _log(f"[{tag}] memory at freeze (non-gating): {P52._memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}")
    _log(f'[{tag}] lock observations (non-gating): {lock_obs}')
    _log(f'[{tag}] guard {PARENT_GUARD.counts} verify0={guard_failures} wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}; run with --stage {stage} --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    sys.exit(0 if ok else 1)


# ======================================================================================================================
#  the formulas (spec v24 `formulas`), zero solves, on committed / written artifacts
# ======================================================================================================================
def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cell_quantities(eval_dir_rel):
    """Every spec v24 per-cell quantity, from the evaluation's own artifacts (evaluation_record.json,
    multiscenario_terminal.json, response_terminal.json, per_cycle_record.jsonl). Zero solves."""
    DEC = H.import_disarmed_diagnostic(H.DECOMPOSITION_MODULE)
    CA = H.import_disarmed_diagnostic(H.CURTAILMENT_AUDIT_MODULE)
    ev = os.path.join(REPO, eval_dir_rel)
    rec = json.load(open(os.path.join(ev, 'evaluation_record.json')))
    ms = json.load(open(os.path.join(ev, H.MULTISCENARIO_TERMINAL_FILE)))
    resp = json.load(open(os.path.join(ev, H.RESPONSE_TERMINAL_FILE)))
    rows = _read_jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))
    alpha = float((rec.get('interface_deviation_premium') or {}).get('alpha') or 0.0)
    certified = rec.get('status') == 'certified'
    q = rec.get('certified_cost') if certified else rec.get('terminal_gross_operational_cost')
    summ = ms['summary']
    all_dso = summ['all_dso']
    charge = all_dso['row18_charge_weighted']
    vp = summ['recourse_components']['voltage_pin_total']
    # P |d| form, post hoc (THE P(0) formula), per block from the captured dispersion
    p_post = p_post_p = 0.0
    split_blocks, split_weighted = {}, {k: 0.0 for k in ('sum_omega_d2', 'market_part', 'operation_part', 'E_abs_d',
                                                         'E_abs_d_market', 'covariance')}
    worst_share, worst_abs = None, None
    for key, blk in ms['blocks'].items():
        disp = blk.get('dispersion')
        if blk['kind'] != 'DSO' or disp is None:
            continue
        w = blk['admm_block_weight']
        p_post += w * H.p_posthoc_block(disp['per_scenario_d'], disp['probabilities'], disp['pibar_by_hour'], True)
        p_post_p += w * H.p_posthoc_block(disp['per_scenario_d'], disp['probabilities'], disp['pibar_by_hour'], False)
        dec = DEC.block_decomposition({k: v['d_p_mw'] for k, v in disp['per_scenario_d'].items()},
                                      disp['probabilities'], disp['pi_by_market_by_hour'], disp['pibar_by_hour'])
        split_blocks[f"DSO:{blk['node_id']}:{blk['year']}:{blk['day']}"] = dec
        for k in split_weighted:
            split_weighted[k] += w * dec['totals'][k]
        share = disp['p']['rms_share_of_mean_flow']
        cand = {'block': key, 'rms_mw': disp['p']['rms_mw'], 'share': share,
                'mean_abs_committed_flow_mw': disp['p']['mean_abs_committed_flow_mw']}
        if share is not None and (worst_share is None or share > worst_share['share']):
            worst_share = cand
        if worst_abs is None or disp['p']['rms_mw'] > worst_abs['rms_mw']:
            worst_abs = cand
    agg = DEC.aggregate(split_blocks)
    p_charge = (charge / alpha) if alpha > 0.0 else None
    p_used = p_charge if alpha > 0.0 else p_post
    rs = resp['summary']
    # rule ten: the terminal gross step over its threshold
    last, prev = (rows[-1] if rows else {}), (rows[-2] if len(rows) >= 2 else {})
    step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
            if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None else None)
    tol = last.get('objective_tolerance')
    # entries meeting the row-18 condition (DN side)
    cond_n = cond_mw = 0
    classes = {}
    for e in resp['curtailment_entries']:
        classes[f"{e['network']}|{e['class']}"] = classes.get(f"{e['network']}|{e['class']}", 0) + 1
        if e['network'] != 'TSO' and alpha > 0.0 and e.get('d_p_mw') is not None and e.get('row18_premium_eur_mwh'):
            if e['d_p_mw'] <= CA.D_TOL_MW and e['price_scenario_eur_mwh'] < alpha * e['row18_premium_eur_mwh']:
                cond_n += 1
                cond_mw += e['c_mw']
    lmp_ref = [e['lmp_ref_bus_eur_mwh'] for e in resp['curtailment_entries']
               if e['network'] != 'TSO' and e.get('lmp_ref_bus_eur_mwh') is not None]
    return {
        'eval_dir': eval_dir_rel, 'label': rec.get('candidate_label'), 'candidate_key': rec.get('candidate_key'),
        'eval_key': rec.get('eval_key'), 'alpha': alpha, 'status': rec.get('status'), 'certified': certified,
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'Q': q, 'Q_is_certified': certified, 'bar': (rec.get('bar') or {}).get('value'),
        'rule_ten': {'terminal_step_over_threshold': (step / tol) if (step is not None and tol) else None,
                     'terminal_gross_step_abs': step, 'objective_tolerance': tol,
                     'production': (rec.get('rule_ten') or {}).get('terminal_step_over_threshold')},
        'charge': charge, 'VP': vp, 'C': q - charge if q is not None else None,
        'V': q + vp if q is not None else None, 'A': (q + vp - charge) if q is not None else None,
        'P_charge': p_charge, 'P_posthoc': p_post, 'P_posthoc_p_leg': p_post_p, 'P_used': p_used,
        'P_charge_minus_posthoc_rel': ((p_charge - p_post) / p_post) if (p_charge is not None and p_post) else None,
        'P_posthoc_vs_child_live_rel': H._rel_diff(p_post, rs.get('P_posthoc_weighted'), scale=p_post),
        'dispersion': {'E_abs_d_p_mwh_weighted': all_dso['E_abs_d_p_mwh_weighted'],
                       'E_abs_d_p_mwh_unweighted': all_dso['E_abs_d_p_mwh_sum_over_blocks'],
                       'sum_omega_d2_p_mw2h_weighted': all_dso['sum_omega_d2_p_mw2h_weighted'],
                       'sum_omega_d2_p_mw2h_unweighted': all_dso['sum_omega_d2_p_mw2h_sum_over_blocks'],
                       'peak_block_rms_mw': all_dso['rms_mw_max_over_all_dso_blocks'],
                       'peak_abs_d_mw': all_dso['max_abs_mw_over_all_dso_blocks'],
                       'worst_block_share': worst_share, 'max_abs_rms_block': worst_abs},
        'split_unweighted_w46': {k: agg['all_dso'].get(k) for k in ('sum_omega_d2', 'market_part', 'operation_part',
                                                                    'market_share_of_sum_omega_d2', 'E_abs_d',
                                                                    'E_abs_d_market', 'covariance')},
        'split_weighted': split_weighted,
        'covariance_production_weighted': summ['settlement_identity']['covariance_interface_settlement_deviation_total'],
        'curtailment_per_network': rs['curtailment_per_network'], 'curtailment_entry_classes': classes,
        'row18_condition_entries': {'n': cond_n, 'sum_c_mw_unweighted': cond_mw},
        'flexibility_per_dso_weighted': rs['flexibility_per_dso_weighted'],
        'row18_charge_p_weighted': rs['row18_charge_p_weighted'], 'row18_charge_q_weighted': rs['row18_charge_q_weighted'],
        'q_leg_share': rs['row18_q_leg_share_of_charge'], 'row18_zl_identity': rs['row18_zl_identity'],
        'lmp_ref_dso_quantiles_eur_mwh': (
            {'n': len(lmp_ref), 'min': min(lmp_ref), 'median': statistics.median(lmp_ref), 'max': max(lmp_ref)}
            if lmp_ref else None),
        'initialisation_identity': rec.get('initialisation_identity'),
        'activation_readback_all_ok': (rec.get('activation_readback') or {}).get('all_ok'),
        'per_cycle_rows': len(rows),
        'per_cycle_all_captured': all(r.get('response_captured') for r in rows) if rows else False,
    }


def row_analysis(cells):
    """The spec v24 cross-cell formulas: C / A / P monotonicity and the envelope per adjacent x = 0 pair, each with its
    resolution, and the unit restatement."""
    xs = sorted((c for c in cells.values() if c['label'].startswith('x0_')), key=lambda c: c['alpha'])
    pairs = []
    for c1, c2 in zip(xs, xs[1:]):
        da = c2['alpha'] - c1['alpha']
        res = (c1['bar'] or 0.0) + (c2['bar'] or 0.0)
        dv = c2['V'] - c1['V']
        dq = c2['Q'] - c1['Q']
        lb = da * c2['P_used']
        ub = da * c1['P_used']
        viol = max(lb - dv, dv - ub, 0.0)
        verdict = 'within' if viol == 0.0 else ('consistent' if viol <= res else 'not comparable optima')
        dc = c2['C'] - c1['C']
        d_a = c2['A'] - c1['A']
        dp = c2['P_used'] - c1['P_used']
        pairs.append({
            'alpha1': c1['alpha'], 'alpha2': c2['alpha'], 'resolution': res,
            'envelope': {'dV': dv, 'lb': lb, 'ub': ub, 'violation': viol, 'verdict': verdict,
                         'q_only': {'dQ': dq, 'lb': lb, 'ub': ub}},
            'dC': dc, 'C_non_decreasing': (dc >= -res), 'C_step_determinate': abs(dc) > res,
            'dA': d_a, 'A_non_decreasing': (d_a >= -res),
            'dP': dp, 'P_non_increasing': (dp <= res / da),
        })
    unit = next((c for c in cells.values() if c['label'].startswith('n7_')), None)
    x05 = next((c for c in xs if c['alpha'] == UNIT_ALPHA), None)
    unit_block = None
    if unit and x05 and unit['Q'] is not None and x05['Q'] is not None:
        value = x05['Q'] - unit['Q']
        res = (x05['bar'] or 0.0) + (unit['bar'] or 0.0)
        ratio = value / SRP1_VALUE_PER_MWH
        unit_block = {'value_eur': value, 'resolution': res, 'value_determinate': abs(value) > res,
                      'value_per_mwh': value / P52.UNIT_E_MWH, 'ratio_to_srp1': ratio,
                      'ratio_resolution': math.hypot(res / SRP1_VALUE_PER_MWH,
                                                     value * SRP1_RESOLUTION / SRP1_VALUE_PER_MWH ** 2),
                      'r_prediction': R_PREDICTION, 'pilot_ratio': PILOT_R_RATIO, 'pilot_value_eur': PILOT_VALUE_EUR,
                      'value_minus_pilot': value - PILOT_VALUE_EUR, 'ratio_minus_pilot_ratio': ratio - PILOT_R_RATIO}
    return {'x0_cells_by_alpha': [c['label'] for c in xs], 'adjacent_pairs': pairs, 'unit': unit_block,
            'table': [{'alpha': c['alpha'], 'Q': c['Q'], 'bar': c['bar'], 'charge': c['charge'], 'C': c['C'],
                       'VP': c['VP'], 'V': c['V'], 'A': c['A'], 'P_used': c['P_used'], 'P_posthoc': c['P_posthoc'],
                       'q_leg_share': c['q_leg_share'], 'rule_ten': c['rule_ten']['terminal_step_over_threshold']}
                      for c in xs],
            'objective_convention': OBJECTIVE_CONVENTION, 'formulas': FORMULAS['envelope']}


# ======================================================================================================================
#  the smoke gate (task W64's one gate)
# ======================================================================================================================
def _smoke_checks(entry, eval_dir, outcome, child_error, guard_failures, expected, readback_file, row_spec_entry_keys,
                  hazard_same, alpha_checklist):
    rel = os.path.relpath(eval_dir, REPO)
    rec = json.load(open(os.path.join(eval_dir, 'evaluation_record.json')))
    checks, detail = {}, {}
    sp = rec.get('solve_profile') or {}
    checks['G1_guard_exact'] = (not guard_failures and expected is not None and bool(sp.get('identity_holds')))
    detail['G1'] = {'expected': expected, 'guard_failures': guard_failures, 'child_solve_profile': sp}
    written = {k: (rec.get(k) or {}).get('status') for k in ('multiscenario_terminal', 'operational_workbook',
                                                             'response_terminal')}
    checks['G2_child_ran'] = (child_error is None and rec.get('status') == 'not_certified' and rec.get('cycles_run') == 2
                              and all(v == 'written' for v in written.values())
                              and not (outcome or {}).get('multiscenario_capture_error'))
    detail['G2'] = {'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'), 'written': written,
                    'child_error': child_error, 'outcome': outcome}
    rb = json.load(open(readback_file)) if os.path.isfile(readback_file) else {}
    dso = [v for k, v in (rb.get('per_block') or {}).items() if k.startswith('DSO|')]
    tso = [v for k, v in (rb.get('per_block') or {}).items() if k.startswith('TSO|')]
    checks['G3_activation_readback'] = (bool(rb.get('all_ok')) and len(dso) == EXPECTED_DSO_BLOCKS
                                        and len(tso) == EXPECTED_TSO_BLOCKS
                                        and all(v.get('alpha_on_model') == 0.5 and v.get('n_rows') == 192
                                                and v.get('n_rows_active') == 192 and v.get('n_pair_vars_fixed') == 0
                                                for v in dso)
                                        and all(v['penalty_and_weight_ok'] and v['row18_ok'] for v in tso))
    detail['G3'] = {k: v for k, v in rb.items() if k != 'per_block'}
    ii = rec.get('initialisation_identity') or {}
    checks['G4_init_identity'] = (_finite(ii.get('gross_operational_cost')) and bool(ii.get('gross_operational_cost_hex'))
                                  and ii.get('n_compared') == 0 and ii.get('all_equal') is True)
    detail['G4'] = ii
    resp_lines = H.read_per_cycle_response(eval_dir)
    numeric = [f for f in H.PER_CYCLE_RESPONSE_FIELDS if f not in ('response_cycle', 'response_captured',
                                                                     'response_capture_error', 'non_optimal_blocks')]
    pcr = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    checks['G5_per_cycle'] = (sorted(resp_lines) == [1, 2] and all(v.get('response_captured') for v in resp_lines.values())
                              and all(_finite(v.get(f)) for v in resp_lines.values() for f in numeric)
                              and len(pcr) == 2 and all(set(H.PER_CYCLE_RECORD_FIELDS) <= set(r) for r in pcr)
                              and bool((rec.get('per_cycle_response') or {}).get('cycles_match')))
    detail['G5'] = {'lines': resp_lines, 'per_cycle_record_keys_ok': [set(H.PER_CYCLE_RECORD_FIELDS) <= set(r) for r in pcr]}
    resp = json.load(open(os.path.join(eval_dir, H.RESPONSE_TERMINAL_FILE)))
    ents = resp['curtailment_entries']
    r18 = [e for e in ents if 'row18_dev_p_def_dual_raw' in e]
    # the zL identity distribution over every checked index (recomputed here from the per-block legs)
    zl = resp['summary']['row18_zl_identity']
    legs = [v for v in resp['row18_legs_by_dso_block'].values() if v is not None]
    frac_1e4 = (sum(v['zl_identity']['n_le_1e-4'] for v in legs) / zl['n_checked']) if zl['n_checked'] else 0.0
    checks['G6_row18_duals_nonempty_and_sane'] = (
        len(r18) > 0 and all(_finite(e[k]) for e in r18 for k in ('row18_dev_p_def_dual_raw', 'zL_d_up_raw',
                                                                  'zL_d_down_raw', 'd_up_mw', 'd_down_mw'))
        and zl['n_checked'] == 2 * 96 * EXPECTED_DSO_BLOCKS and zl['n_missing_suffix_values'] == 0
        and frac_1e4 >= 0.99 and zl['worst_rel_zl_sum_vs_2coef'] <= 1e-2)
    detail['G6'] = {'n_entries_with_row18_fields': len(r18), 'zl_identity': zl, 'fraction_le_1e-4': frac_1e4,
                    'samples': r18[:5]}
    coord = resp['coordination_by_dso_block']
    hours_ok = all(len(c.get('hours') or []) == 24 and all(
        _finite(h.get('dual_pf_p_req')) and _finite(h.get('p_pf_req_mw')) and _finite(h.get('pbar_mw'))
        for h in c['hours']) for c in coord.values() if 'capture_error' not in c)
    rhos = sorted({c.get('rho_pf') for c in coord.values()}, key=repr)
    checks['G7_coordination_nonempty_and_sane'] = (
        len(coord) == EXPECTED_DSO_BLOCKS and not any('capture_error' in c for c in coord.values()) and hours_ok
        and all(_finite(c.get('rho_pf')) and c['rho_pf'] > 0 for c in coord.values())
        and all(_finite(c.get('admm_objective_scale')) and c['admm_objective_scale'] > 0 for c in coord.values())
        and any(h['dual_pf_p_req'] != 0.0 for c in coord.values() for h in c['hours']))
    k0 = sorted(coord)[0] if coord else None
    detail['G7'] = {'n_blocks': len(coord), 'rho_pf_values': rhos, 'sample_block': k0,
                    'sample_hours_0_3': (coord[k0]['hours'][:3] if k0 else None)}
    checks['G8_curtailment_entries'] = (len(ents) > 0 and all(
        e['c_mw'] > resp['curtailment_by_block'][e['block']]['tol_mw'] and all(
            _finite(e[k]) for k in ('pg_mw', 'pg_avail_mw', 'qg_mvar', 'sg_avail_mva'))
        and e.get('lmp_bus_dual_raw') is not None and (e.get('termination_last_solve') or {}).get('available')
        for e in ents))
    detail['G8'] = {'n_entries': len(ents), 'n_dso': resp['summary']['n_entries_dso'],
                    'n_tso': resp['summary']['n_entries_tso'], 'samples': ents[:3]}
    cost = resp.get('capture_cost') or {}
    fbytes = os.path.getsize(os.path.join(eval_dir, H.RESPONSE_TERMINAL_FILE))
    rss_delta = ((cost.get('rss_after_bytes') or 0) - (cost.get('rss_before_bytes') or 0))
    checks['G9_capture_cost'] = fbytes < 10 * 10 ** 6 and rss_delta < GIB
    detail['G9'] = {'file_bytes': fbytes, 'rss_delta_bytes': rss_delta, 'capture_cost': cost}
    checks['G10_per_cycle_cost'] = bool(resp_lines) and all((v.get('response_capture_s') or 1e9) < 60.0
                                                            for v in resp_lines.values())
    detail['G10'] = {c: v.get('response_capture_s') for c, v in resp_lines.items()}
    checks['G11_eval_key_unchanged'] = (entry['eval_key'] == SUPERSEDED_KEYS['x0_a0p50']['eval_key']
                                        and row_spec_entry_keys.get('n7_4h_e1_a0p50')
                                        == SUPERSEDED_KEYS['n7_4h_e1_a0p50']['eval_key'])
    detail['G11'] = {'smoke_x0_a0p50': entry['eval_key'], 'superseded_x0': SUPERSEDED_KEYS['x0_a0p50']['eval_key'],
                     'row_unit_key_recomputed': row_spec_entry_keys.get('n7_4h_e1_a0p50'),
                     'superseded_unit': SUPERSEDED_KEYS['n7_4h_e1_a0p50']['eval_key']}
    checks['G12_hazard'] = bool(hazard_same)
    try:
        cell = cell_quantities(rel)
        checks['G13_P_formulas'] = ((cell['P_posthoc_vs_child_live_rel'] or 0.0) <= 1e-9
                                    and cell['P_charge_minus_posthoc_rel'] is not None
                                    and -1e-9 <= cell['P_charge_minus_posthoc_rel'] <= 1e-3)
    except Exception as error:  # noqa: BLE001
        cell = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        checks['G13_P_formulas'] = False
    detail['G13'] = cell
    checks['G14_rule_eleven'] = bool(alpha_checklist) and all(alpha_checklist.values())
    return checks, detail, rec, cell


def run_smoke(started, spec_sha256):
    tag = 'W64-SMOKE'
    stage = 'smoke'
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'smoke root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    others = own_process_alive()
    if others:
        failures.append(f'another {OWN_PROCESS_SUBSTRING}* process is alive: {others}')
    pins, more = check_pins()
    failures += more
    derived = derived_declaration()
    checks_spec = validate_spec(stage, spec, derived)
    failures += [f'spec check failed: {k}' for k, v in checks_spec.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    hazard = hazard_static_checks()
    failures += [f'hazard static check failed: {k}' for k, v in hazard.items() if not v]
    # the row spec's unit key, recomputed now (G11): the row campaign spec is not needed for it
    row_keys = {label: _eval_key(label, derived) for label in STAGES['row']['points']}
    memory = P52.memory_preflight(STAGES[stage]['concurrency'])
    _log(f"[{tag}] memory preflight: {P52._memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {P52._memory_line(memory)}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    snap_before = protected_snapshot()
    entry = spec['candidates'][0]
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f"[{tag}] campaign lock {lock}; entry {entry['label']} eval_key {entry['eval_key'][:16]} -> "
         f'{os.path.relpath(eval_dir, REPO)}')
    base = SOLVES_PER_CYCLE * SMOKE_ROUNDS
    _log(f'[{tag}] DECLARED before the run: solves = {SOLVES_PER_CYCLE} x {SMOKE_ROUNDS} = {base} + every retry '
         'attempted (per-event reconciliation); GUARD.verify(expected) exactly')
    outcome, child_error, smoke_guard, expected, guard_failures = None, None, None, None, None
    try:
        os.makedirs(eval_dir)
        H._write_once_json(os.path.join(eval_dir, 'launch.json'), {
            'mode': 'in-process smoke gate (p515_s53_alpha_row_campaign.run_smoke -> H._child_real)',
            'pid': os.getpid(), 'started_utc': _utc(), 'campaign_spec_sha256': spec_sha256, 'label': entry['label'],
            'eval_key': entry['eval_key'], 'thread_caps_in_env': {k: os.environ.get(k) for k in H.THREAD_CAP_ENV},
            'declared_solves_base': base})
        parent_failures = PARENT_GUARD.verify(0)
        if parent_failures:
            raise RuntimeError(f'the zero-solve guard counted before the smoke: {parent_failures}')
        PARENT_GUARD.uninstall()
        import p514_n_instrumented_cstar as N
        smoke_guard = SolveProfileGuard(tuple(tuple(p) for p in N.PERMITTED),
                                        label='P5.15 W64 smoke gate (bounded, declared)').install()
        env_caps = H._child_verify_env()
        try:
            outcome = H._child_real(SimpleNamespace(spec_sha256=spec_sha256), spec, spec_path, entry, eval_dir, lock,
                                    env_caps, time.time())
        except BaseException as error:  # noqa: BLE001 -- recorded; the gate then FAILS
            child_error = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            print(child_error['traceback'], file=sys.stderr, flush=True)
        rec_path = os.path.join(eval_dir, 'evaluation_record.json')
        sp = (json.load(open(rec_path)).get('solve_profile') or {}) if os.path.isfile(rec_path) else {}
        if sp.get('reconciliation_supported') and sp.get('base_solves') == base:
            expected = base + int(sp.get('retry_solves_credited') or 0)
        guard_failures = (smoke_guard.verify(expected) if expected is not None
                          else [f'reconciliation unsupported or base mismatch: {sp}'])
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    counts_at_verify = dict(smoke_guard.counts) if smoke_guard else None
    snap_after = protected_snapshot()
    hazard_same = snap_before == snap_after
    readback_file = os.path.join(eval_dir, H.ACTIVATION_READBACK_FILE)
    alpha_checklist = None
    if os.path.isfile(os.path.join(eval_dir, 'evaluation_record.json')):
        alpha_checklist = json.load(open(os.path.join(eval_dir, 'evaluation_record.json'))).get(
            'alpha_row_capture_checklist_asserted_before_run')
    try:
        checks, detail, rec, cell = _smoke_checks(entry, eval_dir, outcome, child_error, guard_failures, expected,
                                                  readback_file, row_keys, hazard_same, alpha_checklist)
    except Exception as error:  # noqa: BLE001
        checks, detail, rec, cell = ({'smoke_checks_ran': False},
                                     {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()},
                                     {}, {})
    counts_after_checks = dict(smoke_guard.counts) if smoke_guard else None
    checks['G1b_no_solve_after_verify'] = counts_after_checks == counts_at_verify
    peak = (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss')
    gate = {
        'stage': STAGE_TEXT, 'gate': 'W64 smoke: 2-cycle 2x2 alpha = 0.5 through the harness child path',
        'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']), 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256, 'spec_v24': pins.get('spec_v24'), 'eval_dir': os.path.relpath(eval_dir, REPO),
        'declared_solves_base': base, 'expected_solves': expected,
        'solve_guard': {'counts': dict(smoke_guard.counts) if smoke_guard else None,
                        'permitted_sites': dict(smoke_guard.permitted_sites) if smoke_guard else None,
                        'verify_failures': guard_failures},
        'checks': checks, 'pass': bool(checks) and all(checks.values()),
        'failing': sorted(k for k, v in checks.items() if not v), 'detail': detail,
        'reported_not_gated': {'peak_rss_bytes': peak, 'twin_s52_repro_peak_rss_bytes': PILOT_REPRO_PEAK_RSS_BYTES,
                               'peak_minus_twin_bytes': (peak - PILOT_REPRO_PEAK_RSS_BYTES) if peak else None,
                               'wall_s': time.time() - started, 'q_leg_share': cell.get('q_leg_share'),
                               'lmp_ref_dso_quantiles_eur_mwh': cell.get('lmp_ref_dso_quantiles_eur_mwh'),
                               'curtailment_entry_classes': cell.get('curtailment_entry_classes')},
        'hazard': {'protected_snapshot_unchanged': hazard_same, 'n_files_snapshotted': len(snap_before),
                   'static': hazard},
        'rule_eleven_asserted_before_run': rule11, 'memory_preflight': memory, 'pins': pins,
        'predictions': PREDICTIONS['worker_smoke'],
    }
    H._write_once_json(os.path.join(root, 'smoke_gate.json'), gate)
    manifest = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(root, 'smoke_manifest_sha256.json'), manifest)
    if smoke_guard is not None:
        smoke_guard.uninstall()
    _log(f"[{tag}] solves: counts {gate['solve_guard']['counts']} expected {expected} verify {guard_failures}")
    for k, v in checks.items():
        _log(f'[{tag}]   {k}: {"PASS" if v else "FAIL"}')
    _log(f"[{tag}] GATE {'PASS' if gate['pass'] else 'FAIL'} failing={gate['failing']} wall={time.time() - started:.0f}s")
    sys.exit(0 if gate['pass'] else 1)


# ======================================================================================================================
#  the row, one pair at a time (NOT run in W64)
# ======================================================================================================================
def _pair_files(n):
    return f'pair_{n}_results.json', f'pair_{n}_manifest_sha256.json'


def _verify_manifest(root, fname):
    manifest = json.load(open(os.path.join(root, fname)))
    bad = {p: h for p, h in manifest.items() if not os.path.isfile(os.path.join(REPO, p))
           or H.sha256_file(os.path.join(REPO, p)) != h}
    return manifest, bad


def run_pair(started, spec_sha256, n):
    tag = f'W64-ROW-PAIR{n}'
    stage = 'row'
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    others = own_process_alive()
    if others:
        failures.append(f'another {OWN_PROCESS_SUBSTRING}* process is alive: {others}')
    contents = set(os.listdir(root))
    allowed = {os.path.basename(spec_path)}
    if n > 1:
        allowed |= {'evals', H.INIT_IDENTITY_DIR_NAME, 'campaign_heartbeat.json'}
        for k in range(1, n):
            res_f, man_f = _pair_files(k)
            allowed |= {res_f, man_f}
            if not (os.path.isfile(os.path.join(root, res_f)) and os.path.isfile(os.path.join(root, man_f))):
                failures.append(f'pair {k} is not complete (pairs run in order 1, 2, 3)')
            else:
                _m, bad = _verify_manifest(root, man_f)
                if bad:
                    failures.append(f'pair {k} manifest does not verify: {sorted(bad)[:5]}')
    if contents - allowed:
        failures.append(f'unexpected files in the campaign root: {sorted(contents - allowed)}')
    if os.path.exists(os.path.join(root, _pair_files(n)[0])):
        failures.append(f'pair {n} already ran (write-once)')
    pins, more = check_pins()
    failures += more
    derived = derived_declaration()
    checks_spec = validate_spec(stage, spec, derived)
    failures += [f'spec check failed: {k}' for k, v in checks_spec.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'),
                               H.sha256_file(os.path.abspath(__file__))),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    smoke_gate_rel = os.path.relpath(os.path.join(campaign_root('smoke'), 'smoke_gate.json'), REPO)
    tracked, clean = _git_state(smoke_gate_rel)
    smoke = json.load(open(os.path.join(REPO, smoke_gate_rel))) if os.path.isfile(os.path.join(REPO, smoke_gate_rel)) else {}
    if not (tracked and clean and smoke.get('pass') is True):
        failures.append(f'the smoke gate must be committed, clean and PASS: tracked={tracked} clean={clean} '
                        f"pass={smoke.get('pass')}")
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    hazard = hazard_static_checks()
    failures += [f'hazard static check failed: {k}' for k, v in hazard.items() if not v]
    memory = P52.memory_preflight(STAGES[stage]['concurrency'])
    _log(f"[{tag}] memory preflight: {P52._memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {P52._memory_line(memory)}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    labels = list(PAIRS[n])
    snap_before = protected_snapshot()
    if n == 1:   # the smoke's x0 initialisation record becomes the first reference of the x = 0 identity group
        smoke_ev = smoke['eval_dir']
        src = os.path.join(campaign_root('smoke'), H.INIT_IDENTITY_DIR_NAME, f'{os.path.basename(smoke_ev)}.json')
        dst_dir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
        os.makedirs(dst_dir)
        shutil.copyfile(src, os.path.join(dst_dir, f'smoke_reference__{os.path.basename(src)}'))
        _log(f'[{tag}] placed the smoke initialisation record as the x = 0 identity reference ({H.sha256_file(src)})')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f'[{tag}] {STAGE_TEXT}; pair {n} {labels} at concurrency {spec["concurrency"]}; lock {lock}')
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(labels, ctx)
        wave = dict(getattr(H.evaluate, 'last_batch_info', {}))
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    hazard_same = protected_snapshot() == snap_before
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    points, problems = {}, []
    for label in labels:
        r = by_label.get(label) or {}
        pv = r.get('parent_view') or {}
        cell = None
        if r.get('status') in ('certified', 'not_certified'):
            try:
                cell = cell_quantities(r['eval_dir'])
            except Exception as error:  # noqa: BLE001
                cell = {'error': f'{type(error).__name__}: {error}'}
        points[label] = {'status': r.get('status'), 'barrier_cause': r.get('barrier_cause'), 'exit_code': pv.get('exit_code'),
                         'wall_s': pv.get('wall_s'), 'peak_rss_bytes': pv.get('wait4_ru_maxrss'),
                         'eval_dir': r.get('eval_dir'), 'solve_profile_identity_holds': (r.get('solve_profile') or {}).get(
                             'identity_holds'), 'cell': cell}
        if r.get('status') not in ('certified', 'not_certified') or pv.get('exit_code') not in (0,):
            problems.append(f'{label}: status {r.get("status")} exit {pv.get("exit_code")}')
    ident_dir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
    idents = {f: json.load(open(os.path.join(ident_dir, f))) for f in sorted(os.listdir(ident_dir)) if f.endswith('.json')}
    x0_key = _key_of(X0)
    x0_hex = {f: v['record']['gross_operational_cost_hex'] for f, v in idents.items() if v.get('candidate_key') == x0_key}
    identity_ok = len(set(x0_hex.values())) <= 1
    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'stage': STAGE_TEXT, 'pair': n, 'labels': labels, 'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'objective_convention': OBJECTIVE_CONVENTION, 'points': points, 'problems': problems,
        'init_identity_x0': {'hex_by_record': x0_hex, 'bitwise_equal': identity_ok},
        'hazard': {'protected_snapshot_unchanged': hazard_same, 'static': hazard},
        'wave_info': wave, 'memory_preflight': memory, 'rule_eleven_asserted_before_run': rule11, 'pins': pins,
        'parent_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started}
    res_f, man_f = _pair_files(n)
    H._write_once_json(os.path.join(root, res_f), results)
    manifest = {}
    for label in labels:
        ed = os.path.join(REPO, points[label]['eval_dir']) if points[label]['eval_dir'] else None
        if ed and os.path.isdir(ed):
            for directory, _dirs, files in os.walk(ed):
                for fname in sorted(files):
                    fpath = os.path.join(directory, fname)
                    manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    for f in [os.path.basename(spec_path), res_f]:
        manifest[os.path.relpath(os.path.join(root, f), REPO)] = H.sha256_file(os.path.join(root, f))
    for f in sorted(os.listdir(ident_dir)):
        manifest[os.path.relpath(os.path.join(ident_dir, f), REPO)] = H.sha256_file(os.path.join(ident_dir, f))
    H._write_once_json(os.path.join(root, man_f), manifest)
    PARENT_GUARD.uninstall()
    for label in labels:
        c = points[label]['cell'] or {}
        _log(f"[{tag}] {label}: status={points[label]['status']} cycles={c.get('cycles_run')} Q={c.get('Q')} "
             f"bar={c.get('bar')} charge={c.get('charge')} P={c.get('P_used')} rule10={(c.get('rule_ten') or {}).get('terminal_step_over_threshold')}")
    _log(f'[{tag}] init identity x0 bitwise_equal={identity_ok}; hazard unchanged={hazard_same}; guard {guard_failures}')
    if problems or guard_failures or not identity_ok or not hazard_same:
        _log(f'[{tag}] NOT OK {problems}')
        sys.exit(1)
    if any(points[l]['status'] != 'certified' for l in labels):
        sys.exit(2)
    _log(f'[{tag}] OK')


def analyse(started):
    root = campaign_root('row')
    out = os.path.join(root, 'alpha_row_analysis.json')
    if os.path.exists(out):
        raise SystemExit(f'output exists (write-once): {out}')
    cells = {}
    for n in sorted(PAIRS):
        res_f, man_f = _pair_files(n)
        _m, bad = _verify_manifest(root, man_f)
        if bad:
            raise SystemExit(f'pair {n} manifest does not verify: {sorted(bad)[:5]}')
        res = json.load(open(os.path.join(root, res_f)))
        for label, p in res['points'].items():
            cells[label] = cell_quantities(p['eval_dir'])
    payload = {'stage': STAGE_TEXT, 'utc': _utc(), 'cells': cells, 'row': row_analysis(cells),
               'spec_v24': find_spec_v24(), 'parent_guard': dict(PARENT_GUARD.counts)}
    H._write_once_json(out, payload)
    failures = PARENT_GUARD.verify(0)
    PARENT_GUARD.uninstall()
    _log(f'analysis written: {os.path.relpath(out, REPO)}; guard {failures}; wall {time.time() - started:.1f}s')
    sys.exit(0 if not failures else 1)


def main():
    parser = argparse.ArgumentParser(description=STAGE_TEXT)
    parser.add_argument('--stage', choices=sorted(STAGES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--analyse', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--pair', type=int, choices=sorted(PAIRS), default=None)
    parser.add_argument('--scratch', default=None)
    args = parser.parse_args()
    started = time.time()
    os.chdir(REPO)
    if args.freeze_spec:
        freeze_spec(started)
    elif args.analyse:
        analyse(started)
    elif args.freeze:
        if not args.stage or not args.scratch or os.path.abspath(args.scratch).startswith(REPO + os.sep):
            parser.error('--freeze requires --stage and --scratch <dir outside the repository>')
        os.makedirs(args.scratch, exist_ok=True)
        freeze(args.stage, started, os.path.abspath(args.scratch))
    else:
        if not args.stage or not args.spec_sha256:
            parser.error('--run requires --stage and --spec-sha256')
        if args.stage == 'smoke':
            if args.pair is not None:
                parser.error('--pair is for --stage row')
            run_smoke(started, args.spec_sha256)
        else:
            if args.pair is None:
                parser.error('--stage row --run requires --pair N (1, 2, 3, in order)')
            run_pair(started, args.spec_sha256, args.pair)


if __name__ == '__main__':
    main()
