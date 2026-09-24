"""
P5.15 Addendum 40 ruling 1 (task W64) -- THE ALPHA ROW: launcher, frozen spec v24, capture gate. BUILT, NOT LAUNCHED.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 1 (the R3.6 alpha row, settled, 2 x 2) and ruling 2 (the
init fix: the pilot's alpha > 0 artifacts are superseded, the unit is re-run at alpha = 0.5 to restate R); frozen
spec v23 `data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json` `ruling1_alpha_row`; Planner task W64. The
frozen spec of THIS stage is v24 (`--freeze-spec`, predecessor v23), which holds the purpose, every formula, the
capture, the pre-run assertions, the hazard, the smoke gate and every prediction.

W65 (Planner task W65; frozen spec v25, predecessor v24 3ac8c185, which is NOT edited): the r1 smoke gate (776c2344)
FAILED G9 (response_terminal.json 25.15 MB vs the declared 10 MB) and G13 (charge / alpha 2.99e-6 BELOW the |d| form vs
the declared floor -1e-9); both limits were the Planner's. v25: G13 becomes a SYMMETRIC band DERIVED from IPOPT's primal
tolerances (`p_gap_primal_bound`, evaluated on the r1 evidence before r2 by `g13_bound_on_r1`); G9 becomes a structural
ceiling from measurement over a COMPACT, LOSSLESS layout of the capture (`H.encode_response_payload`; entries are never
capped or subsampled); the recomputed alpha_arb is authoritative; the post-certification hull polish is the s52
pilot's, exactly; the cycle-0 identity is scoped to the five x = 0 cells + the smoke; a zero-solve harness-isolation
check (`--harness-isolation-check`); new campaign ids (s53_alpha_row_smoke_r2, s53_alpha_row_v25) and a gated
reproduction of r1 by r2 (G16).

PURPOSE (spec v24 `purpose`; v25: alpha_arb RECOMPUTED on this instance is authoritative -- median 0.042, p90 0.189,
max 0.877, 94.6 % of hours below 0.25; the figures below are the W46 ones, from a different, 2025-only instance). NOT a search for a coordinated alpha*. The row is a TWO-MECHANISM COST-OF-COMMITMENT
CURVE: market-arbitrage collapse at low alpha (alpha_arb = price spread / 2 pibar has median 0.022, p90 0.157, max
0.668 on this instance), physical-waste growth at high alpha (curtailment needs alpha of order 4+ to hold schedule
where flexibility is exhausted).

THE INSTANCE: the committed 2 x 2 pilot derived case (`p515_s52_pilot_campaign`: 5 representative years x 4 days x
2 market x 2 operation, `data/SRP1/Results/P515S52/pilot_instance/SRP1__s52_pilot_2x2.json`, sha256 7ecff44a...,
scenario checksum 53b4bea4...), baseline ageing, case-file AA (declared), the campaign cost file, EUR 1M budget
(I(x) reported). CELLS, all CERTIFIED (cap 500, 10 consecutive all-pass cycles); v25: post-certification hull polish
configured exactly as the s52 pilot ({'persist_certified_models': False, 'hull_polish': True}, no reference):
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
  --freeze-spec                           ZERO SOLVES. Writes the frozen spec v25 (write-once, named by its sha256;
                                          predecessor v24), after re-verifying the r1 measurements it records
                                          (G9 encoding sizes, G13 bound on r1) and REFUSING if the derived G13 bound
                                          does not admit the r1 observation.
  --harness-isolation-check --scratch D   ZERO SOLVES (guard permitted=() verified 0). W65 ruling 6: a baseline
                                          non-derived SRP1-style configuration through H._child_real up to the run
                                          (run_admm_arm replaced by a probe that stops before any solve) -> none of
                                          the six W64 hooks installed, no W64 capture call, no W64 output; plus a
                                          positive control (the derived / premium configuration) and source checks.
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

EXACT COMMANDS (repo root; W65 names -- the W64 logs freeze_spec_v24_launch.log, smoke_freeze_launch.log,
row_freeze_launch.log and smoke_launch.log are committed evidence and are never written again):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py \\
      --freeze-spec > data/SRP1/Results/P515S53/alpha_row/freeze_spec_v25_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py \\
      --harness-isolation-check --scratch <dir outside the repo> \\
      > data/SRP1/Results/P515S53/alpha_row/harness_isolation_check_w65_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py \\
      --stage smoke --freeze --scratch <dir> > data/SRP1/Results/P515S53/alpha_row/smoke_r2_freeze_launch.log 2>&1
  (and --stage row --freeze ... > .../row_v25_freeze_launch.log)
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_alpha_row_campaign.py \\
      --stage smoke --run --spec-sha256 <sha> > data/SRP1/Results/P515S53/alpha_row/smoke_r2_launch.log 2>&1
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
# W65: v24 is the predecessor of v25 and is NOT edited (its two failed smoke checks stay on the record)
SPEC_V24 = {'path': os.path.join(_P53, 'frozen_s53_spec_v24_3ac8c185.json'),
            'sha256': '3ac8c185261e80bfda29b6b7c53645da1d5984af3d20e5aae1db18f685a14074'}
SPEC_V25_PREFIX = 'frozen_s53_spec_v25_'
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
# W65 ruling 4: post-certification hull polish on every cell, configured EXACTLY as the s52 pilot (by import of the
# s52 launcher's own constant, asserted equal to its pilot stage's; the s52 spec file is never read -- hazard).
POST_CERTIFICATION = dict(P52.POST_CERTIFICATION)
if POST_CERTIFICATION != P52.STAGES['pilot']['post_certification'] or POST_CERTIFICATION != {
        'persist_certified_models': False, 'hull_polish': True}:
    raise RuntimeError(f'post-certification is not the s52 pilot\'s: {POST_CERTIFICATION} vs '
                       f"{P52.STAGES['pilot']['post_certification']}")
POST_CERTIFICATION_ENTRY = {**POST_CERTIFICATION, 'reference': None}   # the harness's resolved form (no reference)
# W65 (ruling 7): NEW campaign ids -- the W64 roots (s53_alpha_row: frozen, never run; s53_alpha_row_smoke: the r1
# evidence) are never written again. The smoke carries the row's post-certification request (skipped at cap 2).
STAGES = {
    'row': {'campaign_id': 's53_alpha_row_v25', 'cap': 500, 'concurrency': 2,
            'points': tuple(label for n in sorted(PAIRS) for label in PAIRS[n]),
            'post_certification': dict(POST_CERTIFICATION)},
    'smoke': {'campaign_id': 's53_alpha_row_smoke_r2', 'cap': 2, 'concurrency': 1, 'points': ('x0_a0p50',),
              'post_certification': dict(POST_CERTIFICATION)},
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

# ---- W65: the W64 campaign specs (superseded, recorded) and the r1 smoke evidence (committed, cited, READ ONLY) -------
W64_CAMPAIGN_SPECS = {
    'row': {'path': os.path.join(ALPHA_ROOT_REL, 'campaign_s53_alpha_row', 'campaign_spec_s53_alpha_row_870f3af8.json'),
            'sha256': '870f3af8cff8c9bb41ecdcaa368f6044723abc557e1ab8dcfea347902fce9266',
            'status': 'frozen at cd8b7cf8 pinning v24, NEVER RUN; superseded by the v25 row spec (W65 ruling 7)'},
    'smoke_r1': {'path': os.path.join(ALPHA_ROOT_REL, 'campaign_s53_alpha_row_smoke',
                                      'campaign_spec_s53_alpha_row_smoke_44bc7931.json'),
                 'sha256': '44bc79315915589a2729af1f1fd0ef83bc143e7d1e96222f5b31baca03baf648',
                 'status': 'ran as r1 (776c2344): FAILED G9 and G13; committed evidence, cited; never re-run onto'},
}
R1_SMOKE = {'root': os.path.join(ALPHA_ROOT_REL, 'campaign_s53_alpha_row_smoke'),
            'eval_dir': os.path.join(ALPHA_ROOT_REL, 'campaign_s53_alpha_row_smoke', 'evals', '7d53b6f21b686a44_x0_a0p50'),
            'gate': {'file': 'smoke_gate.json',
                     'sha256': '62c32adb059694b85e9b81d650e95a8a58b27151467900c9a132f887ed2f9048'},
            'manifest': {'file': 'smoke_manifest_sha256.json',
                         'sha256': '4af0c2dd4b1844acbf2f00114c43cee79c4cae21745e91d57b5ccaeb5047460a'},
            'commit': '776c2344',
            'run_working_dir': os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                            'p515s44_s53_alpha_row_smoke_7d53b6f21b686a44_run'),
            'failed_checks': ('G13_P_formulas', 'G9_capture_cost')}

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
#  W65 -- THE FROZEN SPEC v25 (content; `--freeze-spec` writes it, named by its own sha256). v24 is NOT edited: its
#  objects above (FORMULAS, NEW_CAPTURE, PRE_RUN_ASSERTIONS, SMOKE_GATE, PREDICTIONS) stay byte-identical so
#  `find_spec_v24` still re-derives the committed v24 file; v25's are deep copies with the W65 amendments.
# ======================================================================================================================
def _copy(obj):
    return json.loads(json.dumps(obj, default=str))


# ---- G9 (W65 ruling 2): the size CEILING, justified by measurement ------------------------------------------------------
# Measured by `measure_r1_encoding` (zero solves) on the committed r1 capture re-encoded with the W65 compact layout;
# `--freeze-spec` re-measures and refuses on any difference.
G9_R1_MEASUREMENT = {
    'r1_file': os.path.join(R1_SMOKE['eval_dir'], 'response_terminal.json'),
    'r1_file_bytes_v1_indent1': 25151522,
    'r1_payload_reencoded_bytes_v2': 9443819,
    'reencoded_decodes_to_r1_exactly': True,
    'n_entries_r1': {'DSO': 15751, 'TSO': 72},
    'bytes_per_entry_max': {'DSO': 490, 'TSO': 368},
    'bytes_per_network_hour_key_max': 362,
    'n_network_hour_keys_r1': 5761,
    'fixed_part_bytes_r1': 1181377,
    'entries_header_bytes_r1': 16276,
}
# the structural maxima: EVERY curtaillable generator-hour-scenario of every block above tolerance (the entry count
# cannot exceed it at any alpha) and every network-hour-scenario carrying a row set. Recomputed from the instance at
# campaign freeze (`instance_facts`) and from the r1 capture (`measure_r1_encoding`); both must equal these.
G9_STRUCTURAL_MAX = {'n_entries_dso': 23040, 'n_entries_tso': 11520, 'n_network_hour_scenarios': 7680}
G9_HEADROOM = 1.10                   # on the whole bound
G9_FIXED_PART_FACTOR = 1.25          # the per-block sections (fixed by structure) + the W65 additions to them
G9_HEADER_ALLOWANCE_BYTES = 50000    # block table / column header (16,276 bytes at r1's 68 blocks with entries)
G9_RSS_DELTA_LIMIT_BYTES = GIB       # unchanged from v24


def g9_ceiling_bytes(measurement=None, structural=None):
    """CEILING = ceil_to_1e6( 1.10 x [ N_dso_max x b_dso_max + N_tso_max x b_tso_max + K_max x b_key_max
                                       + 1.25 x F_r1 + 50,000 ] )  bytes."""
    m = measurement or G9_R1_MEASUREMENT
    n = structural or G9_STRUCTURAL_MAX
    raw = G9_HEADROOM * (n['n_entries_dso'] * m['bytes_per_entry_max']['DSO']
                         + n['n_entries_tso'] * m['bytes_per_entry_max']['TSO']
                         + n['n_network_hour_scenarios'] * m['bytes_per_network_hour_key_max']
                         + G9_FIXED_PART_FACTOR * m['fixed_part_bytes_r1'] + G9_HEADER_ALLOWANCE_BYTES)
    return {'raw_bytes': raw, 'ceiling_bytes': int(math.ceil(raw / 1e6) * 1e6)}


G9_CEILING_BYTES = g9_ceiling_bytes()['ceiling_bytes']


# ---- G13 (W65 ruling 1): the SYMMETRIC band derived from IPOPT's primal tolerances --------------------------------------
G13_ROW18_ROW_GRADIENT_INF_NORM = 1.0   # row 18 defining rows: +1 pg_adn, -1 expected, -1 d+, +1 d- (unit coefficients)
G13_CASE_FILES = tuple(os.path.join('data', 'SRP1', c, f'{c}_params.json') for c in ('case33_1', 'case33_2', 'case33_3'))


def g13_eps_beta(options, ipopt_exit):
    """(eps, beta, derivation) for ONE block: eps bounds |r| (the row-18 defining-row residual, per unit) at the
    block's last IPOPT exit; beta bounds the returned point's violation of d+/- >= 0 (per unit). None if the exit is
    neither of IPOPT's two success exits (then no bound is derivable)."""
    o = {k: (v['value'] if isinstance(v, dict) else v) for k, v in options.items()}
    if o.get('nlp_scaling_method') != 'gradient-based':
        return None, None, f"nlp_scaling_method {o.get('nlp_scaling_method')!r}: the row-scale argument does not apply"
    s_row = min(1.0, float(o['nlp_scaling_max_gradient']) / G13_ROW18_ROW_GRADIENT_INF_NORM)
    if ipopt_exit == 'optimal':
        eps = min(float(o['tol']) / s_row, float(o['constr_viol_tol']))
    elif ipopt_exit == 'acceptable':
        eps = min(float(o['acceptable_tol']) / s_row, float(o['acceptable_constr_viol_tol']))
    else:
        return None, None, f'IPOPT exit {ipopt_exit!r} is not a success exit: no primal bound'
    beta = min(float(o['bound_relax_factor']) * max(1.0, 0.0), float(o['constr_viol_tol']))
    return eps, beta, f'row scale {s_row}; exit {ipopt_exit}'


def p_gap_primal_bound(resp):
    """THE v25 G13 BOUND on one evaluation's captured response (decoded): E = sum_b w_b 2 W_b (eps_b + 2 beta_b) with
    W_b = legs.primal_split.bound_weight (sum_s omega_s sum_t pibar_t B_b, ONE leg), eps_b / beta_b from the block's
    last IPOPT exit and the options in force (`g13_eps_beta`). Symmetric band: |P_charge - P_posthoc| <= E.
    Also the realized decomposition (the capture's P_gap_decomposition_weighted) against the observed gap."""
    rs = resp['summary']
    opts = rs.get('ipopt_options_in_force_dso') or {}
    e_tot, problems, per_block = 0.0, [], {}
    for key, legs in (resp.get('row18_legs_by_dso_block') or {}).items():
        split = (legs or {}).get('primal_split')
        if split is None:
            problems.append(f'{key}: no row 18 primal split')
            continue
        term = (resp['curtailment_by_block'].get(key) or {}).get('termination_last_solve') or {}
        net = f"DSO{key.split('|')[1]}"
        if not term.get('succeeded'):
            problems.append(f'{key}: last solve not succeeded ({term})')
            continue
        if net not in opts:
            problems.append(f'{key}: no IPOPT options recorded for {net}')
            continue
        eps, beta, how = g13_eps_beta(opts[net], term.get('ipopt_exit'))
        if eps is None:
            problems.append(f'{key}: {how}')
            continue
        w = resp['curtailment_by_block'][key]['admm_block_weight']
        e_b = w * 2.0 * split['bound_weight'] * (eps + 2.0 * beta)
        e_tot += e_b
        per_block[key] = {'w': w, 'W': split['bound_weight'], 'eps': eps, 'beta': beta, 'E_b': e_b,
                          'ipopt_exit': term.get('ipopt_exit')}
    p_post = rs.get('P_posthoc_weighted')
    p_charge = rs.get('P_charge_over_alpha')
    obs = (p_charge - p_post) if (p_charge is not None and p_post is not None) else None
    dec = rs.get('P_gap_decomposition_weighted') or {}
    return {'derivable': not problems and bool(per_block), 'problems': problems, 'E_eur_per_alpha': e_tot,
            'E_rel_to_P_posthoc': (e_tot / p_post) if p_post else None, 'observed_gap_eur_per_alpha': obs,
            'observed_gap_rel': (obs / p_post) if (obs is not None and p_post) else None,
            'observed_over_E': (abs(obs) / e_tot) if (obs is not None and e_tot) else None,
            'within_symmetric_band': bool(obs is not None and not problems and per_block and abs(obs) <= e_tot),
            'exits': sorted({v['ipopt_exit'] for v in per_block.values()}),
            'eps_values': sorted({v['eps'] for v in per_block.values()}),
            'beta_values': sorted({v['beta'] for v in per_block.values()}),
            'realized_decomposition_weighted': dec,
            'decomposition_gap_minus_observed_abs': (abs(dec['gap'] - obs) if (dec.get('gap') is not None
                                                                              and obs is not None) else None),
            'n_blocks': len(per_block)}


# ---- the r1 re-verification (zero solves; `--freeze-spec` refuses on any difference) ----------------------------------
def _verify_r1_file(rel):
    """sha256 of an r1 file against r1's committed smoke manifest (the evidence read is exactly the committed one)."""
    man = _load(os.path.join(R1_SMOKE['root'], R1_SMOKE['manifest']['file']))
    got = H.sha256_file(os.path.join(REPO, rel))
    if man.get(rel) != got:
        raise RuntimeError(f'r1 evidence {rel} does not verify against the r1 manifest ({got} vs {man.get(rel)})')
    return got


def measure_r1_encoding():
    """G9 inputs: the committed r1 capture re-encoded with the W65 compact layout (in memory; nothing written into the
    repository), its exact decode, the per-entry / per-key maxima, the fixed part, and the structural maxima."""
    rel = os.path.join(R1_SMOKE['eval_dir'], 'response_terminal.json')
    sha = _verify_r1_file(rel)
    r = _load(rel)
    enc = H.encode_response_payload(r)
    text = json.dumps(enc, separators=(',', ':'), default=str)
    dec = H.decode_response_payload(json.loads(text))
    dec['schema'] = r['schema']
    exact = all(json.dumps(r.get(k), sort_keys=True) == json.dumps(dec.get(k), sort_keys=True) for k in set(r) | set(dec))

    def c(v):
        return len(json.dumps(v, separators=(',', ':'), default=str))
    ce = enc['curtailment_entries']
    per = [c(i) + 1 for i in ce['block_index']]
    for col in ce['columns']:
        for j, v in enumerate(col):
            per[j] += c(v) + 1
    kinds = ['TSO' if ce['block_table'][i]['const']['network'] == 'TSO' else 'DSO' for i in ce['block_index']]
    cb = r['curtailment_by_block']
    structural = {'n_entries_dso': sum(len(b['curtaillable_gens']) * len(b['scenarios']) * 24 for b in cb.values()
                                       if b['kind'] == 'DSO'),
                  'n_entries_tso': sum(len(b['curtaillable_gens']) * len(b['scenarios']) * 24 for b in cb.values()
                                       if b['kind'] == 'TSO'),
                  'n_network_hour_scenarios': sum(len(b['scenarios']) * 24 for b in cb.values())}
    measurement = {
        'r1_file': rel, 'r1_file_bytes_v1_indent1': os.path.getsize(os.path.join(REPO, rel)),
        'r1_payload_reencoded_bytes_v2': len(text), 'reencoded_decodes_to_r1_exactly': exact,
        'n_entries_r1': {k: kinds.count(k) for k in ('DSO', 'TSO')},
        'bytes_per_entry_max': {k: max(p for p, kk in zip(per, kinds) if kk == k) for k in ('DSO', 'TSO')},
        'bytes_per_network_hour_key_max': max(c(k) + c(v) + 2 for k, v in r['network_hour_rows'].items()),
        'n_network_hour_keys_r1': len(r['network_hour_rows']),
        'fixed_part_bytes_r1': sum(c(enc[k]) for k in enc if k not in ('curtailment_entries', 'network_hour_rows')),
        'entries_header_bytes_r1': c({k: v for k, v in ce.items() if k not in ('columns', 'block_index')}),
    }
    return {'measurement': measurement, 'structural': structural, 'r1_file_sha256': sha}


def g13_bound_on_r1():
    """W65 ruling 1, BEFORE r2: the derived bound evaluated on the committed r1 evidence. r1 captured neither the
    IPOPT exit messages nor the options in force nor the primal split, so: the exits and the final violations are
    read from the r1 IPOPT logs (the P56A working dir of the r1 run: untracked, hash-recorded here), the options from
    the DSO case files (tracked), W_b from r1's multiscenario_terminal.json (pibar, probabilities, weights) and B from
    r1's response capture. The realized residual / split is sampled on r1's curtailment entries (d, d+, d-)."""
    ms_rel = os.path.join(R1_SMOKE['eval_dir'], 'multiscenario_terminal.json')
    rt_rel = os.path.join(R1_SMOKE['eval_dir'], 'response_terminal.json')
    gate_rel = os.path.join(R1_SMOKE['root'], R1_SMOKE['gate']['file'])
    shas = {rel: _verify_r1_file(rel) for rel in (ms_rel, rt_rel)}
    gate_sha = H.sha256_file(os.path.join(REPO, gate_rel))
    if gate_sha != R1_SMOKE['gate']['sha256']:
        raise RuntimeError(f'r1 smoke_gate.json sha256 {gate_sha} != {R1_SMOKE["gate"]["sha256"]}')
    ms, rt, gate = _load(ms_rel), _load(rt_rel), _load(gate_rel)
    # options in force: the DSO case files (tracked), defaults for what they leave unset
    opts, case_shas = {}, {}
    for rel in G13_CASE_FILES:
        case_shas[rel] = H.sha256_file(os.path.join(REPO, rel))
        configured = (_load(rel).get('solver') or {}).get('options') or {}
        opts[rel] = {k: {'value': configured.get(k, dflt), 'source': 'case_file' if k in configured else 'ipopt_default'}
                     for k, dflt in H.IPOPT_DEFAULTS_3_14_18.items()}
    # the r1 IPOPT logs of the 60 DSO blocks
    logs_dir = os.path.join(REPO, R1_SMOKE['run_working_dir'], 'logs')
    logs = {}
    for fname in sorted(os.listdir(logs_dir)):
        if not (fname.startswith('optim_log_case33_') and fname.endswith('.log')):
            continue
        with open(os.path.join(logs_dir, fname)) as handle:
            lines = handle.read().splitlines()
        exits = [ln for ln in lines if ln.startswith('EXIT:')]
        cv = [float(ln.split()[-1]) for ln in lines if ln.startswith('Constraint violation....')]
        bv = [float(ln.split()[-1]) for ln in lines if ln.startswith('Variable bound violation')]
        logs[fname] = {'sha256': H.sha256_file(os.path.join(logs_dir, fname)), 'n_exit_lines': len(exits),
                       'last_exit': exits[-1] if exits else None,
                       'last_final_unscaled_constraint_violation': cv[-1] if cv else None,
                       'last_final_variable_bound_violation': bv[-1] if bv else None}
    exits = {v['last_exit'] for v in logs.values()}
    all_optimal = len(logs) == EXPECTED_DSO_BLOCKS and exits == {'EXIT: Optimal Solution Found.'}
    by_case = {rel.split(os.sep)[-2]: opts[rel] for rel in G13_CASE_FILES}
    eps_beta = {case: g13_eps_beta(o, 'optimal') for case, o in by_case.items()}
    # W_b from r1's own captured dispersion inputs; B per block from r1's response capture
    w_sum, base_values = 0.0, set()
    for key, blk in ms['blocks'].items():
        if blk['kind'] != 'DSO' or blk.get('dispersion') is None:
            continue
        disp = blk['dispersion']
        rkey = f"DSO|{blk['node_id']}|{blk['year']}|{blk['day']}"
        base = rt['curtailment_by_block'][rkey]['baseMVA']
        base_values.add(base)
        w_one_leg = sum(disp['probabilities'][s] * sum(disp['pibar_by_hour']) for s in disp['per_scenario_d']) * base
        w_sum += blk['admm_block_weight'] * w_one_leg
    # every DSO case file carries the same tol / acceptable_tol and leaves the rest unset (asserted): one (eps, beta)
    distinct = {(e, b) for e, b, _how in eps_beta.values()}
    if len(distinct) != 1:
        raise RuntimeError(f'DSO case files differ in the G13 tolerances: {eps_beta}')
    eps, beta = next(iter(distinct))
    e_tot = 2.0 * w_sum * (eps + 2.0 * beta)
    p_post = gate['detail']['G13']['P_posthoc']
    p_charge = gate['detail']['G13']['P_charge']
    obs = p_charge - p_post
    # the realized residual / split, sampled on r1's DSO curtailment entries (MW -> per unit / B)
    res, low = [], []
    for e in rt['curtailment_entries']:
        if 'd_up_mw' in e:
            res.append(abs(e['d_p_mw'] - (e['d_up_mw'] - e['d_down_mw'])))
            low.append(min(e['d_up_mw'], e['d_down_mw']))
    base = next(iter(base_values)) if len(base_values) == 1 else None
    return {
        'inputs_sha256': {**shas, gate_rel: gate_sha, **case_shas},
        'ipopt_options_in_force_by_case_file': opts, 'eps_beta_by_case_file': {k: list(v) for k, v in eps_beta.items()},
        'eps': eps, 'beta': beta, 'baseMVA_dso': sorted(base_values),
        'W_weighted_sum_eur_per_alpha_per_pu': w_sum,
        'r1_logs': {'dir': R1_SMOKE['run_working_dir'] + '/logs', 'n_dso_logs': len(logs),
                    'all_last_exits_optimal': all_optimal, 'last_exits': sorted(exits),
                    'max_last_final_unscaled_constraint_violation': max(
                        v['last_final_unscaled_constraint_violation'] for v in logs.values()),
                    'max_last_final_variable_bound_violation': max(
                        v['last_final_variable_bound_violation'] for v in logs.values()),
                    'per_file': logs},
        'E_eur_per_alpha': e_tot, 'E_rel_to_P_posthoc': e_tot / p_post,
        'P_charge_r1': p_charge, 'P_posthoc_r1': p_post, 'observed_gap_eur_per_alpha': obs,
        'observed_gap_rel': obs / p_post, 'observed_over_E': abs(obs) / e_tot,
        'bound_relaxation_part_only_4_beta_W': 4.0 * beta * w_sum,
        'row_residual_part_only_2_eps_W': 2.0 * eps * w_sum,
        'admitted': bool(all_optimal and abs(obs) <= e_tot),
        'realized_on_r1_entries': {'n': len(res), 'max_abs_row_residual_mw': max(res),
                                   'max_abs_row_residual_pu': (max(res) / base) if base else None,
                                   'min_split_leg_mw': min(low), 'min_split_leg_pu': (min(low) / base) if base else None,
                                   'max_split_leg_mw': max(low),
                                   'note': 'the P leg on curtailment entries only (a sample, not every index); '
                                           'r2 captures every index and both legs (primal_split)'},
    }


def g13_r1_projection(result):
    """The part of `g13_bound_on_r1` recorded in v25 (every number exact; the 60 r1 DSO IPOPT logs hash-recorded)."""
    keys = ('eps', 'beta', 'baseMVA_dso', 'W_weighted_sum_eur_per_alpha_per_pu', 'E_eur_per_alpha', 'E_rel_to_P_posthoc', 'P_charge_r1', 'P_posthoc_r1', 'observed_gap_eur_per_alpha', 'observed_gap_rel', 'observed_over_E', 'bound_relaxation_part_only_4_beta_W', 'row_residual_part_only_2_eps_W', 'admitted', 'realized_on_r1_entries', 'inputs_sha256', 'eps_beta_by_case_file')
    return {**{k: result[k] for k in keys}, 'r1_logs': dict(result['r1_logs'])}


# The r1 evaluation of the G13 bound, recorded in v25 BEFORE r2, generated from `g13_bound_on_r1` (zero solves) and
# re-computed and compared EXACTLY at --freeze-spec (refusing on any difference, and refusing unless admitted).
G13_R1_EVALUATION = {'eps': 1e-05,
 'beta': 1e-08,
 'baseMVA_dso': [100.0],
 'W_weighted_sum_eur_per_alpha_per_pu': 4058936966.8023076,
 'E_eur_per_alpha': 81341.09681471824,
 'E_rel_to_P_posthoc': 0.002499974690292382,
 'P_charge_r1': 32536670.811121777,
 'P_posthoc_r1': 32536768.12432252,
 'observed_gap_eur_per_alpha': -97.3132007420063,
 'observed_gap_rel': -2.990868680324179e-06,
 'observed_over_E': 0.0011963595839341815,
 'bound_relaxation_part_only_4_beta_W': 162.3574786720923,
 'row_residual_part_only_2_eps_W': 81178.73933604616,
 'admitted': True,
 'realized_on_r1_entries': {'n': 15751,
                            'max_abs_row_residual_mw': 3.448604577008862e-09,
                            'max_abs_row_residual_pu': 3.448604577008862e-11,
                            'min_split_leg_mw': -8.108688182541835e-07,
                            'min_split_leg_pu': -8.108688182541835e-09,
                            'max_split_leg_mw': 5.2878984725565305e-06,
                            'note': 'the P leg on curtailment entries only (a sample, not every index); r2 '
                                    'captures every index and both legs (primal_split)'},
 'inputs_sha256': {'data/SRP1/Results/P515S53/alpha_row/campaign_s53_alpha_row_smoke/evals/7d53b6f21b686a44_x0_a0p50/multiscenario_terminal.json': '497fd88a2f2c28598e0ea35790e38a7460e1e9588e494e4f1c385720405d6bbb',
                   'data/SRP1/Results/P515S53/alpha_row/campaign_s53_alpha_row_smoke/evals/7d53b6f21b686a44_x0_a0p50/response_terminal.json': '51b0c9581d76dc2127c1b8989448b17df85ed8b2bfec68354122fbdb1d8d0cd7',
                   'data/SRP1/Results/P515S53/alpha_row/campaign_s53_alpha_row_smoke/smoke_gate.json': '62c32adb059694b85e9b81d650e95a8a58b27151467900c9a132f887ed2f9048',
                   'data/SRP1/case33_1/case33_1_params.json': '8e9e5a536fd1ef2bca5694274a68795d1099884ef23f2134e7db7d3a58c80b8e',
                   'data/SRP1/case33_2/case33_2_params.json': '31b5fedf87b96724dc5676566fd87f1c21dc02ecfb8d2bd9fdf280a9b7be35ff',
                   'data/SRP1/case33_3/case33_3_params.json': 'a19bd5b9d18a26deff44dc62a351f2e6c193012c3f7075d12cade4f230760c49'},
 'eps_beta_by_case_file': {'case33_1': [1e-05, 1e-08, 'row scale 1.0; exit optimal'],
                           'case33_2': [1e-05, 1e-08, 'row scale 1.0; exit optimal'],
                           'case33_3': [1e-05, 1e-08, 'row scale 1.0; exit optimal']},
 'r1_logs': {'dir': 'data/SRP1/Results/P56A/evals/p515s44_s53_alpha_row_smoke_7d53b6f21b686a44_run/logs',
             'n_dso_logs': 60,
             'all_last_exits_optimal': True,
             'last_exits': ['EXIT: Optimal Solution Found.'],
             'max_last_final_unscaled_constraint_violation': 2.2706200714390362e-06,
             'max_last_final_variable_bound_violation': 9.786098275073977e-09,
             'per_file': {'optim_log_case33_1_2025_Autumn.log': {'sha256': '30ae183e7f4e8538685af4b61790e6c2c20286fe97b11c2a3edb3aad7b61c158',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.29613358829306e-09,
                                                                 'last_final_variable_bound_violation': 9.671822803525877e-09},
                          'optim_log_case33_1_2025_Spring.log': {'sha256': '126fcff4ba25679b7a698583e6a4ebc25bb5ee0be2485a5eaf8c23d1af8b4d57',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.6164758470947902e-08,
                                                                 'last_final_variable_bound_violation': 9.669301295730961e-09},
                          'optim_log_case33_1_2025_Summer.log': {'sha256': '0237471e2faacde2d82986f3f125bbbf41e70f5183e3ddb51fea33288d3e2736',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.860012392526869e-09,
                                                                 'last_final_variable_bound_violation': 9.660903671049066e-09},
                          'optim_log_case33_1_2025_Winter.log': {'sha256': 'd49765bea71d37aed99f8cb397bb70b2d68cf5a0ca154026337282719cb5d76f',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.400785450995535e-08,
                                                                 'last_final_variable_bound_violation': 9.65635109303471e-09},
                          'optim_log_case33_1_2028_Autumn.log': {'sha256': '29c319d9f82825e7f33d9a1087f42b03ae0943efee236566f3769ff58dcd44d9',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.409095311075362e-11,
                                                                 'last_final_variable_bound_violation': 9.663481165268714e-09},
                          'optim_log_case33_1_2028_Spring.log': {'sha256': '9cbd94bd65411c1b0cfc84dc346aba78ceeebfd6e6412ca4980874be837a49ad',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.4019879840398985e-09,
                                                                 'last_final_variable_bound_violation': 9.663663966229705e-09},
                          'optim_log_case33_1_2028_Summer.log': {'sha256': 'd87c77b683b16bb4e3aa75e93c77630be9e1e0fbfcb71e3fa5159fab1fcd725c',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.9498120612814205e-10,
                                                                 'last_final_variable_bound_violation': 9.650428943151246e-09},
                          'optim_log_case33_1_2028_Winter.log': {'sha256': 'c29317d0678cd7f4f970718ec6c2942d0becbcd0027ea6e7dd360fe527806d69',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.1855049164566367e-09,
                                                                 'last_final_variable_bound_violation': 9.669298210628566e-09},
                          'optim_log_case33_1_2031_Autumn.log': {'sha256': '7d9eb91e2a2ea3debfaa0164431aabd67821d5f12bdd3a391e42f47a2441b4cf',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.2956515294557676e-09,
                                                                 'last_final_variable_bound_violation': 9.678094835841603e-09},
                          'optim_log_case33_1_2031_Spring.log': {'sha256': '4b0fdb1fa8f57c41bf59d81eed4ccb3227249fab115f8d20fd63c4cd62163f00',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 5.2762921506385934e-08,
                                                                 'last_final_variable_bound_violation': 9.66653406420775e-09},
                          'optim_log_case33_1_2031_Summer.log': {'sha256': 'e2d31b972165fdc6c64070b3a9ba260ac6eb1682ac5f1b979a3bdb7862cd8823',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 9.500494835279483e-11,
                                                                 'last_final_variable_bound_violation': 9.663593953341902e-09},
                          'optim_log_case33_1_2031_Winter.log': {'sha256': '6842c23c1edeca79cce5e2981e4935d4218974d59c74087c98a784bdd0863c8c',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.6212831077666578e-10,
                                                                 'last_final_variable_bound_violation': 9.677749622739812e-09},
                          'optim_log_case33_1_2034_Autumn.log': {'sha256': '4ea1b945572443ad303e52def0a12ecb5d178636aafa6c2b57787f8a9690ce00',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.972012335706765e-08,
                                                                 'last_final_variable_bound_violation': 9.657034480984611e-09},
                          'optim_log_case33_1_2034_Spring.log': {'sha256': '94465834a29611746fc220b5832cb3cfd00f01243841cb132f0b2632064680c8',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.8068156056014693e-08,
                                                                 'last_final_variable_bound_violation': 9.667394651700312e-09},
                          'optim_log_case33_1_2034_Summer.log': {'sha256': '753f6feef2ef2e474f7ed33d5fad6dfbc83636f1724edb72a10931fc618540d9',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.5294321724186054e-09,
                                                                 'last_final_variable_bound_violation': 9.657058977271273e-09},
                          'optim_log_case33_1_2034_Winter.log': {'sha256': 'c09a981acdbd732ffd984953633039e350af62a978a719ee36372aefe9afe9c6',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.4314538799241795e-10,
                                                                 'last_final_variable_bound_violation': 9.693088098932184e-09},
                          'optim_log_case33_1_2037_Autumn.log': {'sha256': 'c2ba0116a2253fe80be7f7021c36165f5446fdc41b215e7b96c2aa5c56600465',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.9387603461827894e-13,
                                                                 'last_final_variable_bound_violation': 9.68046387334108e-09},
                          'optim_log_case33_1_2037_Spring.log': {'sha256': '8b6b6a418b1c92ab2b9852a314e2cb125677642efbd876b06f3a6b836c084c30',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.865805947051882e-08,
                                                                 'last_final_variable_bound_violation': 9.652025900972701e-09},
                          'optim_log_case33_1_2037_Summer.log': {'sha256': 'e67fde88dcc180fbaeb991d04a7f533d51ec3bd9df6a7036397f9a1b21a42bda',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.3600190124948313e-10,
                                                                 'last_final_variable_bound_violation': 9.669052533989643e-09},
                          'optim_log_case33_1_2037_Winter.log': {'sha256': 'f18f65ab519754010fc9f93893ac49d49dc7d0fc8a4f4f31f8ea8b74dbaff993',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.8616737585428496e-10,
                                                                 'last_final_variable_bound_violation': 9.676314440805246e-09},
                          'optim_log_case33_2_2025_Autumn.log': {'sha256': '8ea80121ae034f40b7ca058ff96043598a5d46db732eb7d69b603872554b3a4c',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.7474669811168297e-07,
                                                                 'last_final_variable_bound_violation': 9.730489382539427e-09},
                          'optim_log_case33_2_2025_Spring.log': {'sha256': '63745d15e2ce81bcc71e2ea1e96e7c4de919b759b56b0bb554b89ddbe95b0c21',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.477085369103406e-12,
                                                                 'last_final_variable_bound_violation': 9.76734836284985e-09},
                          'optim_log_case33_2_2025_Summer.log': {'sha256': '1850f18a4a578390af6cdd212e52c020044180d66958877eb3e184c451e5416f',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.287369381097747e-07,
                                                                 'last_final_variable_bound_violation': 9.701037826114518e-09},
                          'optim_log_case33_2_2025_Winter.log': {'sha256': 'a91d5019549ed1d6196944ae32c9c7ef89a7e99d9ca22e8970859c1d1c22ae93',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.2706200714390362e-06,
                                                                 'last_final_variable_bound_violation': 9.712111875985281e-09},
                          'optim_log_case33_2_2028_Autumn.log': {'sha256': 'fba37e2d4d7b0ddad475ce372eabac6d9ddbb0e488af337c3b5ee0f4252eca40',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 6.203936724347159e-07,
                                                                 'last_final_variable_bound_violation': 9.661536208675935e-09},
                          'optim_log_case33_2_2028_Spring.log': {'sha256': '6ea51cfcead6de12548cb10d01c7ea399d1ae1f5bd174f41a9ce87bbddb9e93d',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.088785782701109e-08,
                                                                 'last_final_variable_bound_violation': 9.750493221260543e-09},
                          'optim_log_case33_2_2028_Summer.log': {'sha256': '94a9296f5ee3fe1e4d7d63e42ae172c4e7a6a6e52ad8c537e433283ded9ef17b',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 7.591227646486232e-10,
                                                                 'last_final_variable_bound_violation': 9.773033000049593e-09},
                          'optim_log_case33_2_2028_Winter.log': {'sha256': '23f738476e02396381a4b592d7df1804bccaf9249bb703f7d3edc7f8f4c04f12',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.0287061313896118e-08,
                                                                 'last_final_variable_bound_violation': 9.732673669936042e-09},
                          'optim_log_case33_2_2031_Autumn.log': {'sha256': '49262ed4834338476f5ca0bc890e597b0f7445015778b909522b1fc99fbcbf27',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.3498628315697943e-09,
                                                                 'last_final_variable_bound_violation': 9.724558077021079e-09},
                          'optim_log_case33_2_2031_Spring.log': {'sha256': 'f2b146743c335d7e4629af739b8f96ce2c5f5cc9399d8114965cf5965014186b',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 5.1705550306807915e-09,
                                                                 'last_final_variable_bound_violation': 9.667162964605882e-09},
                          'optim_log_case33_2_2031_Summer.log': {'sha256': '7edf70159cdec20960ebacb7c385826a16e7d25190ca9c01a5260aa9414d7283',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.8916549982559445e-09,
                                                                 'last_final_variable_bound_violation': 9.769448269450166e-09},
                          'optim_log_case33_2_2031_Winter.log': {'sha256': 'bf58be2decca76f8e5986d3e1b7dd0dbd0b09a31b0fd9f19ba639637679b9b2e',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.5739589476204685e-07,
                                                                 'last_final_variable_bound_violation': 9.735609456043394e-09},
                          'optim_log_case33_2_2034_Autumn.log': {'sha256': 'f23bae98e8429f787a66598368a09be62e68b8bcabd5ad61932d47c4a3f0a4af',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 7.960299086562372e-12,
                                                                 'last_final_variable_bound_violation': 9.760563505744095e-09},
                          'optim_log_case33_2_2034_Spring.log': {'sha256': 'ad2c9f619e8afa4b0c915ae120cec5c2eea7a5c06671fa13b10f6139890566d2',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.725941016612229e-08,
                                                                 'last_final_variable_bound_violation': 9.702890601752684e-09},
                          'optim_log_case33_2_2034_Summer.log': {'sha256': '8b25350411a4c7ab2f25a3e8f63094e8f8a6ccfd00ffad769684ea8dbd5b1df5',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.0190539532068167e-08,
                                                                 'last_final_variable_bound_violation': 9.70685314200694e-09},
                          'optim_log_case33_2_2034_Winter.log': {'sha256': 'fc72d6b9d685b04c96a9c2aba9dfdc75249483de76b622e4a023aa59676faf80',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.4616596821781513e-10,
                                                                 'last_final_variable_bound_violation': 9.786098275073977e-09},
                          'optim_log_case33_2_2037_Autumn.log': {'sha256': '4aeac9263f5d02b67da8e07b18cb94ca5f70bd05bf8e4f54b3b99235f83b12b9',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 6.568190698635873e-08,
                                                                 'last_final_variable_bound_violation': 9.73802652665286e-09},
                          'optim_log_case33_2_2037_Spring.log': {'sha256': '7ab44a5d26668b7216d6c1c31cf6efd482494982fffe277c4ec5b1a7c58cfe7b',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 6.772051530656853e-10,
                                                                 'last_final_variable_bound_violation': 9.690847410957865e-09},
                          'optim_log_case33_2_2037_Summer.log': {'sha256': '6c3b2e65157e15b26f403e22d8097783bdc6b42c39111e221a0d19d97a9fff2a',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.224608194946457e-07,
                                                                 'last_final_variable_bound_violation': 9.72318351246543e-09},
                          'optim_log_case33_2_2037_Winter.log': {'sha256': '61c30eddcfd6ded932f46e51efa5423dd76941577ec24fcf712428a013da1600',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 5.756325416328423e-10,
                                                                 'last_final_variable_bound_violation': 9.721804565651748e-09},
                          'optim_log_case33_3_2025_Autumn.log': {'sha256': 'af7427ff77839cf203b5cb41e319ef496b8993cf9402a0a1fb7b6d5cf620f306',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 6.015742348708386e-10,
                                                                 'last_final_variable_bound_violation': 9.752072721311288e-09},
                          'optim_log_case33_3_2025_Spring.log': {'sha256': 'c2b09ae31f54b3e0604b3029cbbeb2771a534e8376921b1119d97dbcaf229225',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 8.012347176844514e-07,
                                                                 'last_final_variable_bound_violation': 9.67470708129855e-09},
                          'optim_log_case33_3_2025_Summer.log': {'sha256': 'b1846c18d91a6eb20dbe7e968cf6af3711183690e6bc29f6df8e09784e6bf5f2',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 9.358486208199679e-13,
                                                                 'last_final_variable_bound_violation': 9.72043925640278e-09},
                          'optim_log_case33_3_2025_Winter.log': {'sha256': 'd58c6521bc973e440d44a9ff859d7f6065ff6549a350dbb0813ee3cc5ccae5de',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 4.506811590587745e-12,
                                                                 'last_final_variable_bound_violation': 9.72039948210281e-09},
                          'optim_log_case33_3_2028_Autumn.log': {'sha256': '82e8ff349388ce687b6a2a0b74f2232e24687c75da0f2c583f3bdbf8b29ef539',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 2.515580792650475e-07,
                                                                 'last_final_variable_bound_violation': 9.64881324792599e-09},
                          'optim_log_case33_3_2028_Spring.log': {'sha256': '845c0ba0e696348e02db7736550a440bc1f84b0e1e1b7a6d2ebdf4cd69b9173c',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.0372514269718636e-07,
                                                                 'last_final_variable_bound_violation': 9.674505353768889e-09},
                          'optim_log_case33_3_2028_Summer.log': {'sha256': 'a3a72144580fdf147c4cf0f79cbfd68fe01f0f06b750ebf054a84b9ccf1235fe',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 6.425460929992965e-08,
                                                                 'last_final_variable_bound_violation': 9.712800076347283e-09},
                          'optim_log_case33_3_2028_Winter.log': {'sha256': 'a72f6618742c13569bc7f71c8d6b4390c8b6a517bb94109e024c54911f328a09',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.338241905126438e-10,
                                                                 'last_final_variable_bound_violation': 9.761969057988686e-09},
                          'optim_log_case33_3_2031_Autumn.log': {'sha256': 'fb76c6cb17549f88a4f60b796deaa59b1092e5e1a5e16d3923c2308fba737fc7',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 7.353700981482802e-12,
                                                                 'last_final_variable_bound_violation': 9.706995798502531e-09},
                          'optim_log_case33_3_2031_Spring.log': {'sha256': 'fa5563d8c23d3ffe8a02713a82c1755e3781cd6168a69dded66a0a88b87e3466',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.199869925638808e-09,
                                                                 'last_final_variable_bound_violation': 9.702931005089234e-09},
                          'optim_log_case33_3_2031_Summer.log': {'sha256': 'e1f64cfa4a4d2ec3d90ab4e45e19c8878837cc327d1f69c5c564627d5c77cdf4',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.4476203791247146e-08,
                                                                 'last_final_variable_bound_violation': 9.707194786251795e-09},
                          'optim_log_case33_3_2031_Winter.log': {'sha256': '3f03253b04a184311eb088a2c757c0b539fc5ec480a922e43ffc383002e39a2d',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.1212404305016577e-07,
                                                                 'last_final_variable_bound_violation': 9.768298029219119e-09},
                          'optim_log_case33_3_2034_Autumn.log': {'sha256': 'e5a9081d2d30afe398251964f01e623cbb07d82f1b2ba7d81440b669050fb2a6',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.3158811859037556e-06,
                                                                 'last_final_variable_bound_violation': 9.663508954422754e-09},
                          'optim_log_case33_3_2034_Spring.log': {'sha256': '3c7b5387e7774f995568e37e7f681ece7a345f4cc82485f0b9705a0b0aa6d396',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.1312081138470374e-07,
                                                                 'last_final_variable_bound_violation': 9.679739921260326e-09},
                          'optim_log_case33_3_2034_Summer.log': {'sha256': 'd330ddfb2fe0895d7e09065a9aa047a229425c961e43139d531c9e1f9bdd0ee5',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.341022067280484e-07,
                                                                 'last_final_variable_bound_violation': 9.642057244732607e-09},
                          'optim_log_case33_3_2034_Winter.log': {'sha256': '21213adf340367a527263920521b1b9f64e749d582b6c6ba59dd91577959c12f',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.0101211190292076e-08,
                                                                 'last_final_variable_bound_violation': 9.736215446208151e-09},
                          'optim_log_case33_3_2037_Autumn.log': {'sha256': 'f5facc6872964cb750bc8329647c9122d1baf3babe3ec13b0ac48c5aae71a3aa',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 6.619038650512721e-11,
                                                                 'last_final_variable_bound_violation': 9.773024875773158e-09},
                          'optim_log_case33_3_2037_Spring.log': {'sha256': '9bb49f2f929bb16310abf0ae8523b8d90917b669336030ddc0c5cae093a698cd',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.4583361582611557e-06,
                                                                 'last_final_variable_bound_violation': 9.645629576306738e-09},
                          'optim_log_case33_3_2037_Summer.log': {'sha256': '108172efbd3a9b345fdea30e1c181b38de2834cd01262b12debb22bb387980c2',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 3.4363274101223418e-09,
                                                                 'last_final_variable_bound_violation': 9.737364297399642e-09},
                          'optim_log_case33_3_2037_Winter.log': {'sha256': '8ad815c61f92477d67442153fd642bcfda140dfa2dfb47061973fb91bb4332e9',
                                                                 'n_exit_lines': 3,
                                                                 'last_exit': 'EXIT: Optimal Solution Found.',
                                                                 'last_final_unscaled_constraint_violation': 1.0858806565039457e-07,
                                                                 'last_final_variable_bound_violation': 9.77235077673167e-09}}}}


# ---- alpha_arb (W65 ruling 3): the recomputed values GOVERN ----------------------------------------------------------
ALPHA_ARB_V25 = {
    'authoritative': {'median': 0.042183120545718114, 'p90': 0.18890081514072066, 'max': 0.877367699960065,
                      'n_hours': 480,
                      'fraction_of_hours_below': {'0.0': 0.0, '0.1': 0.7270833333333333, '0.25': 0.9458333333333333,
                                                  '0.5': 0.9916666666666667, '1.0': 1.0}},
    'authoritative_provenance': (
        'this instance (the 2 x 2 pilot derived case 7ecff44a..., scenario checksum 53b4bea4...): '
        'p515_s53_alpha_row_campaign.instance_facts -> p515_s51_coordinated_decomposition.alpha_arb_by_hour on the TSO '
        'block prices, per (year, day, hour), 5 representative years x 4 days x 24 h = 480 hours; first recorded in '
        'the W64 campaign specs (870f3af8 / 44bc7931, extra.instance_facts.alpha_arb, cd8b7cf8); recomputed at every v25 '
        'campaign freeze and asserted EQUAL to the values here'),
    'superseded_figures': {'median': 0.022, 'p90': 0.157, 'max': 0.668},
    'superseded_provenance': (
        'the earlier committed decomposition, W46: data/SRP1/Results/P515S51/coordinated_decomposition/w46_r1/'
        'analysis.json (sha256 fa49bf81290fbf72f14e1a94224e3939d22da2e23b820d828e22148aea14296c, commit 74923c25), '
        'alpha_arb_distribution (median 0.02226, 0.9-quantile 0.15725, max 0.66786, n = 288) -- a DIFFERENT instance: '
        'derived case fe054c22..., scenario checksum 9d633584..., 2025 only, 12 DSO blocks (nodes 5 / 7 / 9 x 4 days) x '
        '24 h, DSO-block prices. Quoted by the Planner in the v24 purpose; not wrong for that instance, not this one'),
    'conclusion': (
        'UNCHANGED: arbitrage suppression is essentially complete by alpha ~ 0.25 -- 94.6 % of this instance\'s hours '
        'have alpha_arb < 0.25 and the p90 is 0.189 < 0.25; 99.2 % are below 0.5 and all 480 below 1.0. The tail is '
        'heavier than the W46 figures (max 0.877 vs 0.668): 26 of 480 hours (5.4 %) have alpha_arb at or above 0.25 and '
        '4 (0.8 %) at or above 0.5, so the low-alpha end of the curve may still move between 0.25 and 0.5 on those hours'),
}

AMENDMENTS_V25 = {
    'ruling_1_G13': ('v24 G13 floor (charge / alpha >= the |d| form - 1e-9) REFUTED by r1: -2.99e-6. Replaced by a '
                     'SYMMETRIC band derived from IPOPT\'s primal tolerances (formulas.P_gap_primal_bound); the check '
                     'is fixed, the observation is not. Evaluated on the r1 evidence BEFORE r2 (g13_r1_evaluation): '
                     'ADMITTED, |obs| / E = 1.196e-3.'),
    'ruling_2_G9': ('v24 G9 limit (< 10 MB) REFUTED by r1: 25.15 MB (15,823 entries at ~1.35 kB, indent 1). The capture '
                    'is written in a COMPACT, LOSSLESS layout (new_capture.w65.layout); entries are NEVER capped or '
                    'subsampled; the limit is re-declared as a structural ceiling from measurement (smoke_gate.g9).'),
    'ruling_3_alpha_arb': 'the recomputed alpha_arb governs (alpha_arb); both provenances recorded; conclusion unchanged',
    'ruling_4_hull_polish': ('post-certification hull polish on every cell, configured exactly as the s52 pilot '
                             '({persist_certified_models: False, hull_polish: True}, reference None), asserted equal '
                             'to p515_s52_pilot_campaign\'s pilot stage; Q stays certified_cost (unpolished), as the '
                             'pilot\'s value and R were computed'),
    'ruling_5_init_identity_scope': ('the cycle-0 identity is scoped to the five x = 0 cells plus the (r2) smoke; the '
                                     'unit is a different candidate and is compared with nothing'),
    'ruling_6_isolation': ('a zero-solve harness-isolation check for a baseline non-derived SRP1-style configuration '
                           '(--harness-isolation-check), and the known gaps recorded (known_gaps)'),
    'ruling_7_refreeze_resmoke': ('v25 (predecessor v24 3ac8c185, not edited); new campaign ids s53_alpha_row_smoke_r2 '
                                  '/ s53_alpha_row_v25; the W64 row spec 870f3af8 (never run) and the r1 smoke '
                                  '44bc7931 (committed evidence) are superseded and never written again; r2 predictions '
                                  'recorded here BEFORE the run; r2 must reproduce r1 (smoke_gate G16)'),
    'capture_additions_w65': ('zero-solve, additive: the IPOPT exit message per block (termination record), the '
                              'realized (d+ + d-) - |d| decomposition per DSO block (primal_split), the IPOPT options in '
                              'force per DSO network, and the compact layout with a write-time round-trip check'),
}

V24_PREDICTIONS_SCORED_ON_R1 = {
    'source': 'r1 smoke_gate.json (776c2344, sha256 62c32adb...) and its evaluation record',
    'smoke_checks': {
        'G9_capture_cost': ('REFUTED (FAIL): 25,151,522 bytes vs < 10 MB. Why: the Planner\'s estimate assumed a few '
                            'thousand rows of ~25 floats; r1 wrote 15,823 entries (15,751 DSO) at ~1.35 kB each with '
                            'indent 1 and every field name repeated per entry. RSS delta 12.1 MB passed.'),
        'G13_P_formulas': ('REFUTED (FAIL): charge / alpha = 32,536,670.811 vs the |d| form 32,536,768.124, relative '
                           '-2.99e-6 vs the floor -1e-9. Why: the floor assumed the row-18 defining rows and d+/- >= 0 '
                           'hold EXACTLY; the returned point satisfies them only to IPOPT\'s primal tolerances -- IPOPT '
                           '3.14.18 honor_original_bounds = no leaves the zero leg of a split up to bound_relax_factor '
                           '= 1e-8 pu BELOW zero (r1: min leg -8.11e-9 pu on the entries; logs: max final bound '
                           'violation 9.79e-9), and (d+ + d-) - |d| >= -(|r| + 2 beta) per index. The post-hoc P '
                           'matched the child\'s live value at relative difference 0.0: the P(0) formula is verified.'),
        'all_other_checks': 'PASS (G1-G8, G10-G12, G14, G1b)'},
    'worker_smoke': {
        'W-S1': 'CONFIRMED: 249 solves (240 network, 9 ESSO), 0 retries, GUARD.verify(249) == []',
        'W-S2': 'CONFIRMED: 60/60 DSO alpha 0.5, 192 active rows, 0 fixed pair Vars; 20/20 TSO',
        'W-S3': 'CONFIRMED: 179742.587656685 (0x1.5f0f4b385591bp+17), compared with 0 records',
        'W-S4': ('PARTLY REFUTED: 2 lines, all finite (confirmed); per-cycle capture 0.259 / 0.260 s, BELOW the '
                 'predicted 0.5-10 s'),
        'W-S5': 'REFUTED: 15,823 entries vs 1,500-4,500 predicted; DSO share 99.5 % (>= 95 %, confirmed)',
        'W-S6': 'CONFIRMED: 11,520 / 11,520 at <= 1e-6 (worst 2.51e-7), 0 missing',
        'W-S7': ('REFUTED: charge / alpha is 2.99e-6 BELOW the |d| form, not above it by <= 1e-4 (bound relaxation, '
                 'see G13)'),
        'W-S8': 'CONFIRMED: 60 blocks, every rho_pf 0.132 (in {0.198, 0.132}), some dual_pf_p_req nonzero',
        'W-S9': 'REFUTED on size (25.15 MB vs 2-8 MB); confirmed on capture time (12.5 s) and RSS delta (12.1 MB)',
        'W-S10': ('REFUTED as a range (peak 11.52 GB vs 12.3-13.5 GB); confirmed in substance: below the 12.86 GB '
                  'twin, the capture did not move the peak (ru_maxrss 11.03 GB before = after the capture)'),
        'W-S11': 'CONFIRMED: both keys equal the superseded s52 keys',
        'W-S12': 'CONFIRMED: 1,134.7 s (18.9 min)',
        'W-S13': 'REFUTED: Q-leg share 51.3 % vs 5-40 %',
    },
}

SMOKE_GATE_V25 = _copy(SMOKE_GATE)
SMOKE_GATE_V25['what'] = (
    'ONE 2-cycle x0 alpha = 0.5 evaluation (campaign s53_alpha_row_smoke_r2, cap 2, concurrency 1, post-certification '
    'requested exactly as the row -- the s52 pilot\'s hull polish -- and SKIPPED because a cap-2 run is not certified) '
    'run in the launcher process through the harness child path H._child_real (the gate hooks s38/s39, run_admm_arm, '
    'the W47 multi-scenario capture, the W64 capture with the W65 additions), attached, alone')
SMOKE_GATE_V25['checks']['G9_capture_cost'] = (
    f'response_terminal.json (compact, lossless layout) file_bytes <= {G9_CEILING_BYTES:,} (g9 ceiling); the '
    'write-time round trip verified (the written file decodes EXACTLY to the captured payload, every section) and '
    'n_entries_written == n_entries_captured == summary.n_curtailment_entries_above_tol == len(decoded entries) -- NO '
    'cap; RSS max(after capture, after write-and-verify) - before < 1 GiB')
SMOKE_GATE_V25['checks']['G13_P_formulas'] = (
    '(a) the P |d| form post hoc (multiscenario_terminal.json) equals the child live value to rel 1e-9 (unchanged); '
    '(b) the bound is derivable: alpha > 0, zero retries credited, every DSO block\'s last solve succeeded with an '
    'IPOPT success exit (optimal / acceptable) and the options in force recorded; (c) SYMMETRIC: |P_charge - '
    'P_posthoc| <= E (p_gap_primal_bound). Reported: E, |obs| / E, the realized decomposition (gap, split -/+, '
    'residual) and |captured gap - observed gap|')
SMOKE_GATE_V25['checks']['G15_post_certification_as_s52'] = (
    'the entry requests exactly {persist_certified_models: False, hull_polish: True, reference: None} (== the s52 '
    'pilot\'s); post_certification.json written with status skipped (not certified at cap 2), evaluated False; the '
    'record\'s post_certification summary agrees; its solves are covered by G1 (exactly the declared count)')
SMOKE_GATE_V25['checks']['G16_reproduces_r1'] = (
    '(i) every r1 check other than G9 / G13 has the SAME verdict in r2 (PASS); (ii) canonical-JSON identity (exact '
    'floats) r2 vs r1 of: the solve-guard counts and permitted sites; the initialisation-identity record (gross and '
    'component float.hex); activation_readback.json; every per_cycle_record.jsonl and per_cycle_response.jsonl row less '
    'the four timing / memory fields (cycle_wall_s, rss_bytes, ru_maxrss_bytes, response_capture_s); '
    'multiscenario_terminal.json in full; response_terminal.json decoded and projected onto r1\'s structure (every r1 '
    'key; the W65 additions ignored) less capture_cost and schema; the evaluation-record core (status, cycles_run, '
    'certification_cycle, terminal gross / net, recourse components, bar, rule_ten, eval_key, candidate_key); (iii) '
    'every r1 file compared verifies against r1\'s committed manifest. Any difference FAILS G16.')
SMOKE_GATE_V25['unchanged_from_v24'] = ['G1_guard_exact', 'G2_child_ran', 'G3_activation_readback', 'G4_init_identity',
                                        'G5_per_cycle', 'G6_row18_duals_nonempty_and_sane',
                                        'G7_coordination_nonempty_and_sane', 'G8_curtailment_entries',
                                        'G10_per_cycle_cost', 'G11_eval_key_unchanged', 'G12_hazard', 'G14_rule_eleven',
                                        'G1b_no_solve_after_verify']
SMOKE_GATE_V25['g9'] = {
    'ceiling_bytes': G9_CEILING_BYTES, 'ceiling_raw_bytes': g9_ceiling_bytes()['raw_bytes'],
    'formula': ('ceil_to_1e6( 1.10 x [ N_dso_max x b_dso_max + N_tso_max x b_tso_max + K_max x b_key_max + 1.25 x F_r1 '
                '+ 50,000 ] )'),
    'how_chosen': (
        'from MEASUREMENT, not an estimate: the committed r1 capture re-encoded with the W65 layout (in memory, '
        'decoding back to r1 exactly) gives the per-entry byte maxima (b_dso_max 490, b_tso_max 368), the per '
        'network-hour key maximum (b_key_max 362) and the fixed per-block part (F_r1 1,181,377); the entry count is '
        'bounded STRUCTURALLY -- at most every curtaillable generator-hour-scenario of every block (N_dso_max 23,040, '
        'N_tso_max 11,520; r1 had 15,751 / 72) and every network-hour-scenario (K_max 7,680; r1 5,761) -- so no alpha, '
        'the alpha = 1.0 cell included, can exceed it by COUNT. Headroom: 1.25 on the fixed part (the W65 additions: '
        'messages, primal split, summary keys) and 1.10 overall. A file above the ceiling therefore means a '
        'representation defect (bytes per entry grew), not more curtailment, and FAILS. r1\'s payload re-encoded is '
        '9,443,819 bytes, 43 % of the ceiling; at the structural maximum the entries alone would be 15.5 MB.'),
    'r1_measurement': G9_R1_MEASUREMENT, 'structural_max': G9_STRUCTURAL_MAX,
    'code': 'p515_s53_alpha_row_campaign.measure_r1_encoding / g9_ceiling_bytes; p515_s44_campaign_harness.'
            'encode_response_payload / decode_response_payload / load_response_terminal / write_response_terminal',
}
SMOKE_GATE_V25['reported_not_gated'] = SMOKE_GATE['reported_not_gated'] + [
    'G13 realized decomposition', 'response_terminal.json bytes vs the r1 re-encoding', 'RSS after write-and-verify']

FORMULAS_V25 = _copy(FORMULAS)
FORMULAS_V25['P_gap_primal_bound'] = {
    'statement': ('SYMMETRIC band: |P_charge - P_posthoc| <= E,  E = sum_b w_b x 2 x W_b x (eps_b + 2 beta_b)  (EUR per '
                  'unit alpha)'),
    'symbols': ('w_b = admm_block_weight; W_b = sum_s omega_s sum_t pibar_t B_b (ONE leg; P and Q give the factor 2; '
                'captured as primal_split.bound_weight); eps_b bounds |r|, r = d - (d+ - d-) the row-18 defining-row '
                'residual (per unit) at the block\'s last IPOPT exit; beta_b bounds the returned point\'s violation of '
                'd+/- >= 0 (per unit)'),
    'derivation': [
        'row 18 defines d = d+ - d- with d = pg_adn - expected_interface_pf_p (and the Q twin): the SAME d the |d| form '
        'reads (_get_local_interface_dispersion), so P_charge - P_posthoc = sum_b w_b sum omega_s pibar_t B_b '
        '[(d+ + d-) - |d|] over both legs',
        'the returned point satisfies the row to a residual r and the bounds d+/- >= 0 only to -beta (IPOPT 3.14.18 '
        'honor_original_bounds = no: the final point is NOT projected back into the original bounds)',
        'per index and leg: (d+ + d-) - |d+ - d- + r| lies in [2 min(d+, d-) - |r|, 2 min(d+, d-) + |r|], and d+/- >= '
        '-beta gives 2 min(d+, d-) >= -2 beta, so the LOWER side is >= -(|r| + 2 beta) >= -(eps + 2 beta)',
        'the weights w_b omega_s pibar_t B_b are >= 0 (pibar_t > 0 on this instance: premium_floor_needed False at '
        'freeze), so summing gives P_charge - P_posthoc >= -E',
        'eps: the row-18 rows have unit coefficients (+1 pg_adn, -1 expected, -1 d+, +1 d-), so gradient-based NLP '
        'scaling (nlp_scaling_max_gradient 100) leaves them at scale s = min(1, 100 / 1) = 1; "Optimal Solution '
        'Found" bounds the SCALED overall NLP error (which contains the scaled constraint violation) by tol and the '
        'unscaled violation by constr_viol_tol: eps = min(tol / s, constr_viol_tol); "Solved To Acceptable Level": '
        'eps = min(acceptable_tol / s, acceptable_constr_viol_tol); any other exit: NO bound (the check fails)',
        'beta = min(bound_relax_factor x max(1, |0|), constr_viol_tol): IPOPT relaxes every bound by '
        'bound_relax_factor, capped absolutely by constr_viol_tol (option documentation, ipopt --print-options)',
        'SYMMETRIC as ruled: the upper side carries the same primal term PLUS 2 max(0, min(d+, d-)) -- incomplete '
        'complementarity of the split, NOT a primal-tolerance quantity. It is MEASURED (primal_split.split_positive) '
        'and reported; an upper-side excess beyond E is attributed to it and FAILS the check, never explained away'],
    'values_on_this_instance': ('DSO case files case33_1/2/3_params.json: tol 1e-5, acceptable_tol 1e-4 (set); '
                                'constr_viol_tol 1e-4, acceptable_constr_viol_tol 1e-2, bound_relax_factor 1e-8, '
                                'honor_original_bounds no, nlp_scaling_method gradient-based, nlp_scaling_max_gradient '
                                '100 (IPOPT 3.14.18 defaults, unset) -> eps = 1e-5 (optimal) / 1e-4 (acceptable), '
                                'beta = 1e-8'),
    'retry_caveat': ('the options in force are the case file\'s; a retried last solve\'s option_overrides are not '
                     'reflected -- the smoke requires zero retries; in the row the bound is reported with the retry '
                     'count'),
    'code': 'p515_s53_alpha_row_campaign.p_gap_primal_bound / g13_eps_beta; p515_s44_campaign_harness.'
            '_row18_primal_split_block / ipopt_options_in_force / _termination_record (ipopt_exit)',
}
FORMULAS_V25['P_gap_decomposition'] = (
    'captured per DSO block (primal_split), weighted in summary.P_gap_decomposition_weighted: gap = sum omega pibar B '
    '[(d+ + d-) - |d|] (== P_charge - P_posthoc up to summation order); split = sum omega pibar B 2 min(d+, d-) = '
    'split_positive - split_negative (split_negative: bound violation; split_positive: complementarity); residual = sum '
    'omega pibar B |r|; per block |gap - split| <= residual (asserted, reported)')
FORMULAS_V25['unit_value'] = FORMULAS['unit_value'] + (
    '. v25: Q = certified_cost (UNPOLISHED), exactly as the s52 pilot valued the unit (p515_s52_pilot_campaign.'
    '_value_block, certified_cost_gross) -- so the restated R is comparable with the pilot\'s 0.9418 (report; 0.942 '
    'rounded in code); the hull polish (gate D) is reported beside, never substituted into Q')
FORMULAS_V25['post_certification'] = ('hull polish per certified cell as the s52 pilot: post_certification.json / the '
                                      'record\'s summary (status, gate_d: blocks_solved, relative_pct, threshold_pct, '
                                      'pass) reported per cell in cell_quantities')

NEW_CAPTURE_V25 = _copy(NEW_CAPTURE)
NEW_CAPTURE_V25['w65'] = {
    'termination_record': ('every block\'s last-solve record adds message (the IPOPT .sol exit message) and ipopt_exit '
                           '(optimal / acceptable / other): Pyomo maps both IPOPT success exits to optimal'),
    'primal_split': 'row18_legs_by_dso_block[b].primal_split (_row18_primal_split_block): see formulas.P_gap_decomposition',
    'options_in_force': 'summary.ipopt_options_in_force_dso: the case-file options relevant to G13, IPOPT defaults for unset',
    'exit_counts': 'summary.dso_last_solve_ipopt_exit_counts',
    'layout': ('response_terminal.json schema p515_s44_response_terminal_v2: no indentation; curtailment_entries '
               'COLUMNAR (field names once; block-constant fields once per block with the fields that block lacks; '
               'network_hour_rows_key rebuilt from block|scenario|hour, asserted per entry); coordination hours '
               'columnar; the write-time round trip (decode == captured payload, exact, NaN-safe) or the capture '
               'RAISES; read ONLY through H.load_response_terminal (which reads r1\'s v1 layout unchanged). Nothing is '
               'capped, subsampled or dropped.'),
    'scope': 'as W64: derived-instance / premium evaluations only',
}

PRE_RUN_ASSERTIONS_V25 = _copy(PRE_RUN_ASSERTIONS)
PRE_RUN_ASSERTIONS_V25['before_any_solve_in_the_parent'] = PRE_RUN_ASSERTIONS['before_any_solve_in_the_parent'] + [
    'v25: pins v23 (sha), v24 (sha AND re-derived content: not edited), v25 (sha AND re-derived content)',
    'v25: every entry\'s post_certification == the s52 pilot\'s ({persist False, hull_polish True, reference None}), '
    'asserted at import against p515_s52_pilot_campaign and per entry at freeze and run',
    'v25: at campaign freeze the instance\'s structural curtailment maxima == v25 g9.structural_max and alpha_arb == '
    'v25 alpha_arb.authoritative',
    'v25: pair 1 requires the r2 smoke gate (campaign s53_alpha_row_smoke_r2) committed, clean and PASS; its x0 '
    'initialisation record is the x = 0 identity reference']
PRE_RUN_ASSERTIONS_V25['init_identity_scope'] = (
    'W65 ruling 5: the cycle-0 identity (float.hex) is enforced across the FIVE x = 0 cells (x0_a0p00, x0_a0p10, '
    'x0_a0p25, x0_a0p50, x0_a1p00) plus the r2 smoke\'s x0 record (placed as the reference before pair 1). The unit '
    'n7_4h_e1_a0p50 is a DIFFERENT candidate: its record is written and listed (other_candidates_not_compared), '
    'compared with nothing. r2 vs r1 identity of the smoke\'s own record is checked by G16.')
PRE_RUN_ASSERTIONS_V25['harness_isolation_check'] = (
    'W65 ruling 6, ZERO SOLVES (SolveProfileGuard(permitted=()) verified at exactly 0): --harness-isolation-check '
    '-> data/SRP1/Results/P515S53/alpha_row/harness_isolation_check_w65.json; must PASS before the r2 smoke')

PREDICTIONS_V25 = {
    'planner_spec_v23_ruling1': PREDICTIONS['planner_spec_v23_ruling1'],
    'planner_spec_v23_ruling2': PREDICTIONS['planner_spec_v23_ruling2'],
    'worker_row': dict(PREDICTIONS['worker_row'], **{
        'W-R11': ('the hull polish is evaluated on every certified cell (post_certification status evaluated, gate D '
                  'recorded); no pass / fail prediction beyond "as the pilot"')}),
    'worker_smoke_r2_recorded_before_the_run': {
        'W2-S1': 'G1 as r1: exactly 249 solves (240 network, 9 ESSO), 0 retries, GUARD.verify(249) == []',
        'W2-S2': ('every r1 PASS item (G1-G8, G10-G12, G14, G1b) passes again, and G16 finds r2 IDENTICAL to r1 on every '
                  'compared quantity (the ADMM path is untouched by W65; the s52 repro reproduced its pilot bitwise) -- '
                  'about 90 % confident on bitwise identity; a difference would be reported as a G16 FAIL'),
        'W2-S3': 'initialisation gross cost 0x1.5f0f4b385591bp+17 (= r1), compared with 0 records',
        'W2-S4': ('G13 PASS: P_charge 32,536,670.811121777 and P_posthoc 32,536,768.12432252 exactly as r1, gap '
                  '-97.3132, E = 81,341.10 (all 60 DSO exits optimal, eps 1e-5, beta 1e-8), |gap| / E = 1.2e-3; '
                  'realized: split_negative 97-162 EUR/alpha (it carries the gap), residual term < 1 EUR/alpha, '
                  'split_positive < 10 EUR/alpha; |captured gap - observed gap| < 1e-3 EUR/alpha'),
        'W2-S5': ('G9 PASS: response_terminal.json 9.44-9.56 MB (r1 re-encoded 9,443,819 bytes + the W65 fields) vs the '
                  f'ceiling {G9_CEILING_BYTES:,}; round trip verified; 15,823 entries written == captured; RSS delta incl. '
                  'write-and-verify 0.05-0.5 GiB'),
        'W2-S6': 'G15 PASS: requested {persist False, hull_polish True, reference None}; status skipped; evaluated False',
        'W2-S7': 'smoke process peak RSS within 2 % of r1\'s 11,524,325,376 bytes (the write-verify transient sits '
                 'below the process peak)',
        'W2-S8': 'wall 17-22 min (r1 18.9 min)',
        'W2-S9': ('isolation check PASS: baseline 0 / 6 W64 wrappers installed, 0 W64 capture calls, 0 W64 files; '
                  'positive control 6 / 6 installed, alpha-row checklist and hooks each called once; guard verify(0) == []'),
    },
}

KNOWN_GAPS_V25 = [
    ('No full SRP1 campaign bitwise gate was run for the W64 / W65 harness change (known and ACCEPTED, W65 ruling 6). '
     'What stands in for it: the zero-solve isolation check (harness_isolation_check_w65.json) -- for a baseline '
     'non-derived SRP1-style configuration through H._child_real up to the run, none of the six W64 hooks is '
     'installed, no W64 capture function is called and no W64 output is written; the post-run branches (per-cycle '
     'fields, record keys, terminal capture) by source inspection; the pre-W64 per-cycle tuples by comparison with the '
     'pre-W64 harness (git 70950e8c^).'),
    ('p515_s45_snapshot_off_two_cycle_gate.py writes {k: row.get(k) for k in H.PER_CYCLE_RECORD_FIELDS}: re-run, it '
     'would now write the 22 PER_CYCLE_RESPONSE_FIELDS keys as null in its per-cycle record (W64 finding 6). Not '
     'changed (out of scope); its committed artifacts are unaffected.'),
    ('G13 bound: the options in force are the case file\'s; a retried last solve\'s option_overrides are not '
     'reflected (the smoke requires zero retries; the row reports the retry count beside the bound).'),
    ('The r1 IPOPT logs used to evaluate the G13 bound on r1 (exits, final violations) are in the untracked P56A '
     'working dir; their sha256 are recorded here (g13_r1_evaluation.r1_logs.per_file).'),
]


def spec_v25_content():
    v24 = spec_v24_content()
    cells = _copy(v24['cells'])
    cells['post_certification'] = POST_CERTIFICATION_ENTRY
    cells['post_certification_source'] = ('p515_s52_pilot_campaign.POST_CERTIFICATION == STAGES["pilot"]'
                                          '["post_certification"] (the s52 pilot, campaign_s52_pilot_nopersist, spec '
                                          '08790b4d), by import; resolved by the harness with reference None')
    cells['post_certification_note'] = AMENDMENTS_V25['ruling_4_hull_polish']
    cells['campaign_ids'] = {s: STAGES[s]['campaign_id'] for s in STAGES}
    purpose = _copy(v24['purpose'])
    purpose['alpha_arb'] = ALPHA_ARB_V25
    purpose['v25_note'] = ('the statement above quotes the W46 figures (a different instance); on THIS instance the '
                           'recomputed values govern (alpha_arb.authoritative); the conclusion is unchanged')
    hazard = _copy(v24['hazard'])
    hazard['enforcement'] = [h.replace('fresh campaign ids (s53_alpha_row, s53_alpha_row_smoke)',
                                       'fresh campaign ids (v25: s53_alpha_row_v25, s53_alpha_row_smoke_r2)')
                             for h in hazard['enforcement']]
    hazard['w64_roots'] = ('the W64 roots campaign_s53_alpha_row (spec 870f3af8, never run) and '
                           'campaign_s53_alpha_row_smoke (the r1 evidence) are never written; r1 files are READ (G16, '
                           'the G9 / G13 measurements) only after verifying against r1\'s committed manifest')
    memory = _copy(v24['memory'])
    memory['v25'] = ('the r1 capture measured RSS +12.1 MB and did not move the peak (ru_maxrss 11.03 GB before = '
                     'after); the W65 write-and-verify step is measured in r2 (rss_after_write_verify_bytes) and gated '
                     'with the capture (G9 < 1 GiB); the row\'s hull polish was part of the s52 pilot the memory rule '
                     'was set for (unchanged)')
    return {
        'schema': 'p515_frozen_spec_v25', 'version': 25,
        'stage': v24['stage'] + ' -- W65: G9 / G13 re-derived, hull polish as s52, re-smoke r2',
        'authority': v24['authority'] + ['Planner task W65 (rulings 1-7)'],
        'predecessor': dict(SPEC_V24),
        'predecessor_not_edited': 'v24 stays as frozen; its failed G9 / G13 predictions stay on the record',
        'amendments_from_v24': AMENDMENTS_V25,
        'v24_predictions_scored_on_r1': V24_PREDICTIONS_SCORED_ON_R1,
        'purpose': purpose, 'instance': v24['instance'], 'cells': cells,
        'objective_convention': v24['objective_convention'], 'formulas': FORMULAS_V25,
        'reporting_rules': v24['reporting_rules'], 'new_capture': NEW_CAPTURE_V25,
        'pre_run_assertions': PRE_RUN_ASSERTIONS_V25, 'hazard': hazard,
        'superseded_w64_campaign_specs': W64_CAMPAIGN_SPECS,
        'r1_smoke_evidence': {k: (list(v) if isinstance(v, tuple) else v) for k, v in R1_SMOKE.items()},
        'smoke_gate': SMOKE_GATE_V25,
        'g13_r1_evaluation': G13_R1_EVALUATION,
        'memory': memory,
        'known_gaps': KNOWN_GAPS_V25,
        'predictions_recorded_before_run': PREDICTIONS_V25,
        'stop': 'DO NOT LAUNCH the row in W65. After the row: STOP FOR REVIEW with the 3x3 prediction attached (v23).',
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


def find_spec_v25():
    """W65: the one frozen spec v25, named by its own sha256, whose content equals `spec_v25_content()`."""
    hits = sorted(f for f in os.listdir(os.path.join(REPO, _P53)) if f.startswith(SPEC_V25_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one {SPEC_V25_PREFIX}*.json in {_P53}, found {hits}')
    rel = os.path.join(_P53, hits[0])
    sha = H.sha256_file(os.path.join(REPO, rel))
    if not hits[0].startswith(f'{SPEC_V25_PREFIX}{sha[:8]}'):
        raise RuntimeError(f'{rel} is not named by its own sha256 ({sha[:8]})')
    content = _load(rel)
    expected = json.loads(json.dumps(spec_v25_content(), default=str))
    stripped = {k: v for k, v in content.items() if k not in ('frozen_utc', 'git_head_at_freeze',
                                                                'verified_at_freeze')}
    if stripped != expected:
        diff = sorted(k for k in set(stripped) | set(expected) if stripped.get(k) != expected.get(k))
        raise RuntimeError(f'{rel} content differs from this launcher\'s spec v25 in {diff}')
    return {'path': rel, 'sha256': sha}


def check_pins(require_v25=True):
    out, failures = {}, []
    pins = [('spec_v23', SPEC_V23), ('ess_params_file', P52.ESS_PARAMS_FILE), ('cost_file', P52.COST_FILE),
            ('instance_case', {'path': INSTANCE['case_path'], 'sha256': INSTANCE['case_sha256']}),
            ('instance_record', {'path': INSTANCE['record_path'], 'sha256': INSTANCE['record_sha256']})]
    try:   # v24: the pinned sha AND its content re-derived (proves it was not edited)
        v24 = find_spec_v24()
        if v24 != SPEC_V24:
            failures.append(f'spec v24 {v24} != pinned {SPEC_V24}')
        pins.append(('spec_v24', v24))
    except RuntimeError as error:
        failures.append(f'spec v24: {error}')
    if require_v25:
        try:
            pins.append(('spec_v25', find_spec_v25()))
        except RuntimeError as error:
            failures.append(f'spec v25: {error}')
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
        # W65
        'response_read_through_decoder_in_cell': 'H.load_response_terminal(' in cell_src,
        'g13_bound_in_cell': 'p_gap_primal_bound(' in cell_src,
        'post_certification_in_cell': "'post_certification'" in cell_src,
        'g13_bound_formula_defined': all(t in inspect.getsource(p_gap_primal_bound) for t in (
            "split['bound_weight']", 'g13_eps_beta(', 'abs(obs) <= e_tot')),
        'g13_eps_beta_uses_exit_and_options': all(t in inspect.getsource(g13_eps_beta) for t in (
            "o['tol']", "o['acceptable_tol']", "o['constr_viol_tol']", "o['acceptable_constr_viol_tol']",
            "o['bound_relax_factor']", "o['nlp_scaling_max_gradient']")),
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
    # W65 (G9): the structural maxima of the curtailment capture -- every curtaillable generator-hour-scenario (the
    # capture's own selection: params.rg_curt and generator.is_curtaillable()) and every network-hour-scenario
    n_dso = n_tso = n_hours = 0
    holders = [('TSO', tn)] + [('DSO', dn) for dn in planning.distribution_networks.values()]
    for kind, holder in holders:
        for y in holder.years:
            for d in holder.days:
                net = holder.network[y][d]
                n_scen = len(net.prob_market_scenarios) * len(net.prob_operation_scenarios)
                n_gen = (sum(1 for g in net.generators if g.is_curtaillable()) if holder.params.rg_curt else 0)
                n = n_gen * n_scen * planning.num_instants
                n_hours += n_scen * planning.num_instants
                if kind == 'DSO':
                    n_dso += n
                else:
                    n_tso += n
    facts['curtailment_structural_max'] = {'n_entries_dso': n_dso, 'n_entries_tso': n_tso,
                                           'n_network_hour_scenarios': n_hours}
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
    """W65: freeze spec v25 (write-once, named by its sha256; predecessor v24, which must be tracked, clean, hash as
    pinned and re-derive from this launcher -- i.e. NOT edited). BEFORE writing, the r1 measurements v25 records are
    re-computed from the committed r1 evidence and must equal the recorded constants EXACTLY, and the derived G13 bound
    must ADMIT the r1 observation -- otherwise this REFUSES (W65 ruling 1: STOP and report). Zero solves."""
    failures = []
    for name, pin in (('v23', SPEC_V23), ('v24', SPEC_V24)):
        got = H.sha256_file(os.path.join(REPO, pin['path']))
        tracked, clean = _git_state(pin['path'])
        if got != pin['sha256'] or not (tracked and clean):
            failures.append(f'predecessor spec {name} not as pinned: sha {got} tracked {tracked} clean {clean}')
    try:
        if find_spec_v24() != SPEC_V24:
            failures.append('spec v24 does not re-derive as pinned')
    except RuntimeError as error:
        failures.append(f'spec v24: {error}')
    existing = [f for f in os.listdir(os.path.join(REPO, _P53)) if f.startswith(SPEC_V25_PREFIX)]
    if existing:
        failures.append(f'spec v25 already frozen (write-once): {existing}')
    g9 = measure_r1_encoding()
    if g9['measurement'] != G9_R1_MEASUREMENT or g9['structural'] != G9_STRUCTURAL_MAX:
        failures.append(f"G9 r1 measurement differs from v25's constants: {g9['measurement']} / {g9['structural']}")
    if not g9['measurement']['reencoded_decodes_to_r1_exactly']:
        failures.append('the compact layout does not decode to the r1 capture exactly')
    g13 = g13_bound_on_r1()
    if g13_r1_projection(g13) != G13_R1_EVALUATION:
        failures.append("G13 r1 evaluation differs from v25's recorded constant")
    if not g13['admitted']:
        failures.append(f"STOP (W65 ruling 1): the derived bound E = {g13['E_eur_per_alpha']} does NOT admit the r1 "
                        f"observation {g13['observed_gap_eur_per_alpha']} -- something other than primal tolerance")
    _log(f"G9 on r1: v1 {g9['measurement']['r1_file_bytes_v1_indent1']} B -> v2 "
         f"{g9['measurement']['r1_payload_reencoded_bytes_v2']} B, decodes exactly "
         f"{g9['measurement']['reencoded_decodes_to_r1_exactly']}; structural {g9['structural']}; ceiling "
         f'{g9_ceiling_bytes(g9["measurement"], g9["structural"])}')
    _log(f"G13 on r1: eps {g13['eps']} beta {g13['beta']} W {g13['W_weighted_sum_eur_per_alpha_per_pu']!r} E "
         f"{g13['E_eur_per_alpha']!r} (rel {g13['E_rel_to_P_posthoc']!r}); observed {g13['observed_gap_eur_per_alpha']!r} "
         f"(rel {g13['observed_gap_rel']!r}); |obs|/E {g13['observed_over_E']!r}; 4 beta W "
         f"{g13['bound_relaxation_part_only_4_beta_W']!r}; logs: {g13['r1_logs']['n_dso_logs']} DSO, exits "
         f"{g13['r1_logs']['last_exits']}, max final unscaled constraint violation "
         f"{g13['r1_logs']['max_last_final_unscaled_constraint_violation']!r}, max final bound violation "
         f"{g13['r1_logs']['max_last_final_variable_bound_violation']!r}; ADMITTED {g13['admitted']}")
    if failures:
        for f in failures:
            _log(f'[W65-FREEZE-SPEC PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    content = spec_v25_content()
    content['frozen_utc'] = _utc()
    content['git_head_at_freeze'] = H._git(['rev-parse', 'HEAD'])
    content['verified_at_freeze'] = {'g9_r1_measurement_equal': True, 'g13_r1_evaluation_equal': True,
                                     'g13_admitted': True, 'r1_response_terminal_sha256': g9['r1_file_sha256'],
                                     'v24_rederived': True}
    text = json.dumps(content, indent=1, default=str) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V25_PREFIX}{sha[:8]}.json')
    with open(os.path.join(REPO, rel), 'x') as handle:
        handle.write(text)
    if H.sha256_file(os.path.join(REPO, rel)) != sha:
        raise SystemExit('written spec does not hash to its name')
    check = find_spec_v25()
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'frozen spec v25: {rel} sha256={sha} (predecessor v24 {SPEC_V24["sha256"]}); re-read equal: '
         f'{check["sha256"] == sha}; guard {PARENT_GUARD.counts} verify0={guard_failures}; wall {time.time() - started:.1f}s')
    PARENT_GUARD.uninstall()
    sys.exit(0 if not guard_failures else 1)


def _spec_candidates(stage):
    return [(label, CELLS[label]['nodes'], {'investment_year': YEAR,
                                            'interface_deviation_premium': _premium(CELLS[label]['alpha']),
                                            'post_certification': dict(STAGES[stage]['post_certification'])})
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
        'post_certification_as_s52_pilot': all(e.get('post_certification') == POST_CERTIFICATION_ENTRY
                                               for e in entries),
        'post_certification_equals_p52_pilot_stage': ({**P52.STAGES['pilot']['post_certification'], 'reference': None}
                                                      == POST_CERTIFICATION_ENTRY),
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
    tag = f'W65-{stage.upper()}-FREEZE'
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
    # W65: the instance must reproduce what v25 records (G9 structural maxima; the authoritative alpha_arb)
    if facts['curtailment_structural_max'] != G9_STRUCTURAL_MAX:
        failures.append(f"curtailment structural maxima {facts['curtailment_structural_max']} != v25 {G9_STRUCTURAL_MAX}")
    arb_auth = ALPHA_ARB_V25['authoritative']
    if any(facts['alpha_arb'][k] != arb_auth[k] for k in ('median', 'p90', 'max', 'n_hours', 'fraction_of_hours_below')):
        failures.append(f"alpha_arb {facts['alpha_arb']} != v25 authoritative {arb_auth}")
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
        'stage': stage, 'stage_text': STAGE_TEXT, 'label': LABEL, 'spec_v25': pins.get('spec_v25'),
        'spec_v25_pin': find_spec_v25(), 'spec_v24': dict(SPEC_V24), 'spec_v23': dict(SPEC_V23), 'pins': pins,
        'instance': instance,
        'instance_facts': facts, 'alpha_grid_as_ordered': list(ALPHA_GRID),
        'pairs': {str(n): list(v) for n, v in PAIRS.items()},
        'cells': {label: {'alpha': CELLS[label]['alpha'], 'point': CELLS[label]['point'],
                          'candidate_key': _key_of(CELLS[label]['nodes']), 'eval_key': _eval_key(label, derived)}
                  for label in st['points']},
        'objective_convention': OBJECTIVE_CONVENTION, 'hazard_static_checks': hazard,
        'superseded_keys': SUPERSEDED_KEYS, 'protected_roots': list(PROTECTED_ROOTS),
        'memory_rule': P52.memory_rule(st['concurrency']), 'memory_at_freeze_non_gating': memory,
        'lock_observations_at_freeze_non_gating': lock_obs, 'rule_eleven_asserted_before_run': rule11,
        # W65
        'post_certification': STAGES[stage]['post_certification'], 'post_certification_entry': POST_CERTIFICATION_ENTRY,
        'post_certification_source': 'p515_s52_pilot_campaign.STAGES["pilot"]["post_certification"] (by import)',
        'superseded_w64_campaign_specs': W64_CAMPAIGN_SPECS,
        'init_identity_scope': PRE_RUN_ASSERTIONS_V25['init_identity_scope'],
        'alpha_arb_authoritative': ALPHA_ARB_V25['authoritative'],
        'g9_ceiling_bytes': G9_CEILING_BYTES,
    }
    if stage == 'smoke':
        extra['smoke_declared_solves_base'] = SOLVES_PER_CYCLE * SMOKE_ROUNDS
        extra['smoke_gate'] = SMOKE_GATE_V25
        extra['r1_smoke_evidence'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in R1_SMOKE.items()}
        extra['predictions_r2'] = PREDICTIONS_V25['worker_smoke_r2_recorded_before_the_run']
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
                       'note': ('no overrides, no model variant, no flexibility-price variant; post-certification '
                                'hull polish as the s52 pilot (W65 ruling 4)')},
        cap=st['cap'], concurrency=st['concurrency'],
        authority=['PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 1', 'frozen spec v25 (this stage; predecessor v24)',
                   'Planner tasks W64 and W65'],
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
         f"(v25 authoritative; W46 0.022/0.157/0.668 was another instance); structural "
         f"{facts['curtailment_structural_max']} (v25 {G9_STRUCTURAL_MAX}); G9 ceiling {G9_CEILING_BYTES}")
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
    resp = H.load_response_terminal(os.path.join(ev, H.RESPONSE_TERMINAL_FILE))   # W65: the one reader (v1 / v2)
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
        # W65: the G13 bound (v25 formulas.P_gap_primal_bound) with the retry count it assumes zero of
        'P_gap_primal_bound': (p_gap_primal_bound(resp) if alpha > 0.0 else None),
        'retry_solves_credited': (rec.get('solve_profile') or {}).get('retry_solves_credited'),
        'post_certification': rec.get('post_certification'),
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
    resp = H.load_response_terminal(os.path.join(eval_dir, H.RESPONSE_TERMINAL_FILE))   # W65: the one reader
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
    # W65 (v25 G9): the structural ceiling over the compact lossless layout; nothing capped; RSS incl. write-verify
    cost = (rec.get('response_terminal') or {}).get('capture_cost') or {}
    fbytes = os.path.getsize(os.path.join(eval_dir, H.RESPONSE_TERMINAL_FILE))
    rss_before = cost.get('rss_before_bytes')
    rss_peak_after = max(v for v in (cost.get('rss_after_bytes'), cost.get('rss_after_write_verify_bytes'), 0)
                         if v is not None)
    rss_delta = (rss_peak_after - rss_before) if rss_before is not None else None
    rt = cost.get('roundtrip') or {}
    n_summary = resp['summary']['n_curtailment_entries_above_tol']
    with open(os.path.join(eval_dir, H.RESPONSE_TERMINAL_FILE)) as handle:
        stored_schema = json.load(handle).get('schema')
    checks['G9_capture_cost'] = (fbytes <= G9_CEILING_BYTES and rss_delta is not None and rss_delta < G9_RSS_DELTA_LIMIT_BYTES
                                 and rt.get('verified') is True and stored_schema == H.RESPONSE_TERMINAL_SCHEMA
                                 and rt.get('n_entries_written') == rt.get('n_entries_captured') == n_summary == len(ents))
    detail['G9'] = {'file_bytes': fbytes, 'ceiling_bytes': G9_CEILING_BYTES, 'file_over_ceiling': fbytes / G9_CEILING_BYTES,
                    'r1_reencoded_bytes': G9_R1_MEASUREMENT['r1_payload_reencoded_bytes_v2'],
                    'r1_v1_bytes': G9_R1_MEASUREMENT['r1_file_bytes_v1_indent1'], 'stored_schema': stored_schema,
                    'rss_delta_bytes_incl_write_verify': rss_delta,
                    'rss_delta_bytes_capture_only': ((cost.get('rss_after_bytes') or 0) - (rss_before or 0)),
                    'n_entries': {'summary': n_summary, 'decoded': len(ents), 'written': rt.get('n_entries_written'),
                                  'captured': rt.get('n_entries_captured')},
                    'roundtrip': rt, 'capture_cost': cost}
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
        bound = cell['P_gap_primal_bound'] or {}
        retries = (rec.get('solve_profile') or {}).get('retry_solves_credited')
        # W65 (v25 G13): (a) post hoc == live; (b) derivable, zero retries; (c) SYMMETRIC |gap| <= E
        checks['G13_P_formulas'] = (cell['P_posthoc_vs_child_live_rel'] is not None
                                    and cell['P_posthoc_vs_child_live_rel'] <= 1e-9
                                    and cell['P_charge_minus_posthoc_rel'] is not None
                                    and retries == 0 and bound.get('derivable') is True
                                    and bound.get('within_symmetric_band') is True
                                    # the cell's P_charge (model charge / alpha) against the same band, as well
                                    and abs(cell['P_charge'] - cell['P_posthoc']) <= bound['E_eur_per_alpha'])
        detail['G13_bound'] = {**bound, 'retry_solves_credited': retries,
                               'r1_evaluation_before_r2': {k: G13_R1_EVALUATION[k] for k in (
                                   'E_eur_per_alpha', 'observed_gap_eur_per_alpha', 'observed_over_E', 'admitted')}}
    except Exception as error:  # noqa: BLE001
        cell = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        checks['G13_P_formulas'] = False
    detail['G13'] = cell
    # W65 (v25 G15): the s52 pilot's post-certification requested, skipped at cap 2
    pc_path = os.path.join(eval_dir, H.POST_CERTIFICATION_FILE)
    pc = json.load(open(pc_path)) if os.path.isfile(pc_path) else {}
    pcs = rec.get('post_certification') or {}
    checks['G15_post_certification_as_s52'] = (entry.get('post_certification') == POST_CERTIFICATION_ENTRY
                                               and pc.get('requested') == POST_CERTIFICATION_ENTRY
                                               and pc.get('status') == 'skipped' and pc.get('evaluated') is False
                                               and pcs.get('status') == 'skipped')
    detail['G15'] = {'entry_request': entry.get('post_certification'), 'expected': POST_CERTIFICATION_ENTRY,
                     'post_certification_json': pc, 'record_summary': pcs}
    checks['G14_rule_eleven'] = bool(alpha_checklist) and all(alpha_checklist.values())
    return checks, detail, rec, cell


# ---- W65 (v25 G16): r2 reproduces r1 ----------------------------------------------------------------------------------
G16_TIMING_FIELDS = ('cycle_wall_s', 'rss_bytes', 'ru_maxrss_bytes', 'response_capture_s')
G16_RECORD_CORE = ('status', 'cycles_run', 'certification_cycle', 'terminal_gross_operational_cost',
                   'terminal_net_operational_recourse', 'recourse_components', 'bar', 'rule_ten', 'eval_key',
                   'candidate_key')


def _canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def _project(new, old):
    """`new` restricted to `old`'s structure: dict keys of old (recursively; a key old has and new lacks -> a marker
    that never equals old), lists element-wise (length must match). The W65 additions are thereby ignored."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return {'__type_differs__': type(new).__name__}
        return {k: (_project(new[k], v) if k in new else {'__missing_in_r2__': True}) for k, v in old.items()}
    if isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            return {'__list_differs__': len(new) if isinstance(new, list) else type(new).__name__}
        return [_project(n, o) for n, o in zip(new, old)]
    return new


def _first_difference(new, old, path=''):
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            d = _first_difference(new.get(k), old[k], f'{path}.{k}')
            if d:
                return d
        return None
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return f'{path}: length {len(new)} vs {len(old)}'
        for i, (n, o) in enumerate(zip(new, old)):
            d = _first_difference(n, o, f'{path}[{i}]')
            if d:
                return d
        return None
    return None if _canon(new) == _canon(old) else f'{path}: r2 {new!r} vs r1 {old!r}'


def r1_reproduction(eval_dir_r2, checks_r2, guard_r2):
    """v25 G16. Reads r1 ONLY after verifying each file against r1's committed manifest."""
    r1_ev = R1_SMOKE['eval_dir']
    gate_rel = os.path.join(R1_SMOKE['root'], R1_SMOKE['gate']['file'])
    out, verified = {}, {}
    gate_sha = H.sha256_file(os.path.join(REPO, gate_rel))
    verified[gate_rel] = gate_sha == R1_SMOKE['gate']['sha256']
    gate1 = _load(gate_rel)
    files = ('initialisation_identity.json', 'activation_readback.json', 'per_cycle_record.jsonl',
             'per_cycle_response.jsonl', 'multiscenario_terminal.json', 'response_terminal.json', 'evaluation_record.json')
    for f in files:
        rel = os.path.join(r1_ev, f)
        try:
            _verify_r1_file(rel)
            verified[rel] = True
        except RuntimeError:
            verified[rel] = False
    r2 = eval_dir_r2

    def cmp(name, new, old):
        proj = _project(new, old)
        same = _canon(proj) == _canon(old)
        out[name] = {'identical': same, 'first_difference': None if same else _first_difference(proj, old)}

    # (i) verdicts
    other = {k: v for k, v in gate1['checks'].items() if k not in R1_SMOKE['failed_checks']}
    out['verdicts'] = {'r1': other, 'r2': {k: checks_r2.get(k) for k in other},
                       'identical': all(checks_r2.get(k) == v for k, v in other.items()) and all(other.values())}
    # (ii) values
    cmp('solve_guard', {'counts': guard_r2.get('counts'), 'permitted_sites': guard_r2.get('permitted_sites')},
        {'counts': gate1['solve_guard']['counts'], 'permitted_sites': gate1['solve_guard']['permitted_sites']})
    cmp('initialisation_identity_record', json.load(open(os.path.join(r2, 'initialisation_identity.json')))['record'],
        _load(os.path.join(r1_ev, 'initialisation_identity.json'))['record'])
    cmp('activation_readback', json.load(open(os.path.join(r2, 'activation_readback.json'))),
        _load(os.path.join(r1_ev, 'activation_readback.json')))
    for f in ('per_cycle_record.jsonl', 'per_cycle_response.jsonl'):
        strip = [{k: v for k, v in row.items() if k not in G16_TIMING_FIELDS} for row in _read_jsonl(os.path.join(r2, f))]
        strip1 = [{k: v for k, v in row.items() if k not in G16_TIMING_FIELDS}
                  for row in _read_jsonl(os.path.join(REPO, r1_ev, f))]
        cmp(f, strip, strip1)
    cmp('multiscenario_terminal', json.load(open(os.path.join(r2, 'multiscenario_terminal.json'))),
        _load(os.path.join(r1_ev, 'multiscenario_terminal.json')))
    resp2 = H.load_response_terminal(os.path.join(r2, H.RESPONSE_TERMINAL_FILE))
    resp1 = H.load_response_terminal(os.path.join(REPO, r1_ev, H.RESPONSE_TERMINAL_FILE))
    for section in sorted(set(resp1) - {'capture_cost', 'schema'}):
        cmp(f'response_terminal.{section}', resp2.get(section), resp1[section])
    rec2 = json.load(open(os.path.join(r2, 'evaluation_record.json')))
    rec1 = _load(os.path.join(r1_ev, 'evaluation_record.json'))
    cmp('evaluation_record_core', {k: rec2.get(k) for k in G16_RECORD_CORE}, {k: rec1.get(k) for k in G16_RECORD_CORE})
    values_ok = all(v['identical'] for k, v in out.items() if k != 'verdicts')
    return {'r1_files_verified_against_r1_manifest': verified, 'comparisons': out,
            'identical': bool(all(verified.values()) and out['verdicts']['identical'] and values_ok),
            'excluded': {'per_cycle_fields': list(G16_TIMING_FIELDS), 'response_terminal_sections': ['capture_cost', 'schema'],
                         'w65_additions': 'ignored by projection onto r1\'s structure'}}


def run_smoke(started, spec_sha256):
    tag = 'W65-SMOKE-R2'
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
    _log(f"[{tag}] v25 predictions recorded before the run: {PREDICTIONS_V25['worker_smoke_r2_recorded_before_the_run']}")
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
    # W65 (v25 G16): r2 reproduces r1 -- zero solves (reads files); G1b below covers it
    guard_view = {'counts': counts_at_verify, 'permitted_sites': dict(smoke_guard.permitted_sites) if smoke_guard else None}
    provisional = dict(checks, G1b_no_solve_after_verify=(dict(smoke_guard.counts) if smoke_guard else None)
                       == counts_at_verify)
    try:
        g16 = r1_reproduction(eval_dir, provisional, guard_view)
    except Exception as error:  # noqa: BLE001
        g16 = {'identical': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    detail['G16'] = g16
    counts_after_checks = dict(smoke_guard.counts) if smoke_guard else None
    checks['G1b_no_solve_after_verify'] = counts_after_checks == counts_at_verify
    checks['G16_reproduces_r1'] = bool(g16.get('identical')) and checks['G1b_no_solve_after_verify']
    peak = (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss')
    gate = {
        'stage': STAGE_TEXT, 'gate': 'W65 smoke r2: 2-cycle 2x2 alpha = 0.5 through the harness child path (v25)',
        'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']), 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256, 'spec_v25': pins.get('spec_v25'), 'spec_v24': pins.get('spec_v24'),
        'r1_smoke': {k: (list(v) if isinstance(v, tuple) else v) for k, v in R1_SMOKE.items()},
        'eval_dir': os.path.relpath(eval_dir, REPO),
        'declared_solves_base': base, 'expected_solves': expected,
        'solve_guard': {'counts': dict(smoke_guard.counts) if smoke_guard else None,
                        'permitted_sites': dict(smoke_guard.permitted_sites) if smoke_guard else None,
                        'verify_failures': guard_failures},
        'checks': checks, 'pass': bool(checks) and all(checks.values()),
        'failing': sorted(k for k, v in checks.items() if not v), 'detail': detail,
        'reported_not_gated': {'peak_rss_bytes': peak, 'twin_s52_repro_peak_rss_bytes': PILOT_REPRO_PEAK_RSS_BYTES,
                               'peak_minus_twin_bytes': (peak - PILOT_REPRO_PEAK_RSS_BYTES) if peak else None,
                               'r1_peak_rss_bytes': 11524325376,
                               'peak_minus_r1_bytes': (peak - 11524325376) if peak else None,
                               'g13_realized_decomposition': (detail.get('G13_bound') or {}).get(
                                   'realized_decomposition_weighted'),
                               'wall_s': time.time() - started, 'q_leg_share': cell.get('q_leg_share'),
                               'lmp_ref_dso_quantiles_eur_mwh': cell.get('lmp_ref_dso_quantiles_eur_mwh'),
                               'curtailment_entry_classes': cell.get('curtailment_entry_classes')},
        'hazard': {'protected_snapshot_unchanged': hazard_same, 'n_files_snapshotted': len(snap_before),
                   'static': hazard},
        'rule_eleven_asserted_before_run': rule11, 'memory_preflight': memory, 'pins': pins,
        'predictions': PREDICTIONS_V25['worker_smoke_r2_recorded_before_the_run'],
        'smoke_gate_spec': 'frozen spec v25 smoke_gate (G1-G8, G10-G12, G14, G1b unchanged from v24; G9, G13 re-derived; '
                           'G15, G16 new)',
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
    tag = f'W65-ROW-PAIR{n}'
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
               'spec_v25': find_spec_v25(), 'spec_v24': find_spec_v24(), 'parent_guard': dict(PARENT_GUARD.counts)}
    H._write_once_json(out, payload)
    failures = PARENT_GUARD.verify(0)
    PARENT_GUARD.uninstall()
    _log(f'analysis written: {os.path.relpath(out, REPO)}; guard {failures}; wall {time.time() - started:.1f}s')
    sys.exit(0 if not failures else 1)


# ======================================================================================================================
#  W65 ruling 6: the ZERO-SOLVE harness-isolation check
# ======================================================================================================================
ISOLATION_OUT_REL = os.path.join(ALPHA_ROOT_REL, 'harness_isolation_check_w65.json')
W64_HOOK_NAMES = ('create_transmission_network_model', 'create_shared_energy_storage_model',
                  '_prepare_distribution_objectives_for_admm', '_prepare_transmission_objectives_for_admm',
                  'get_admm_boyd_residual_metrics', '_admm_local_solves_succeeded')
W64_HARNESS_FUNCTIONS = ('assert_alpha_row_capture_paths', 'alpha_row_run_hooks', 'activation_readback',
                         'initialisation_identity_record', 'register_initialisation_identity', 'per_cycle_response_record',
                         'read_per_cycle_response', 'response_terminal_capture', 'write_response_terminal',
                         'write_multiscenario_terminal', 'install_derived_instance')
W64_OUTPUT_FILES = ('RESPONSE_TERMINAL_FILE', 'PER_CYCLE_RESPONSE_FILE', 'ACTIVATION_READBACK_FILE', 'INIT_IDENTITY_FILE')
PRE_W64_HARNESS_REV = '70950e8c^'


class _IsolationProbeStop(Exception):
    pass


def _pre_w64_field_tuples():
    """PER_CYCLE_RECORD_FIELDS and RECORD_TRAJECTORY_FIELDS as the PRE-W64 harness (git 70950e8c^) defines them,
    evaluated from its source (ast; no import of the old module)."""
    import ast
    src = H._git(['show', f'{PRE_W64_HARNESS_REV}:p515_s44_campaign_harness.py'])
    tree = ast.parse(src)
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ('PER_CYCLE_RECORD_FIELDS', 'RECORD_TRAJECTORY_FIELDS'):
            ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), '<pre-w64>', 'eval'),  # noqa: S307
                                          {'__builtins__': {'tuple': tuple, 'sorted': sorted, 'set': set}}, dict(ns))
    return ns


def _isolation_static_checks():
    import inspect
    child_src = inspect.getsource(H._child_real)
    old = _pre_w64_field_tuples()
    after_w47 = child_src[child_src.find('if capture_multiscenario:  # W47: only for derived-instance specs'):
                          child_src.find('record = build_evaluation_record(')]
    terminal = child_src[child_src.find('if capture_multiscenario:  # W47: zero solves, on the terminal models'):
                         child_src.find('        if aa_on:')]
    return {
        'per_cycle_trajectory_fields_equal_pre_w64_record_fields': (
            tuple(H.PER_CYCLE_TRAJECTORY_FIELDS) == tuple(old.get('PER_CYCLE_RECORD_FIELDS') or ())),
        'record_trajectory_fields_equal_pre_w64': (
            tuple(H.RECORD_TRAJECTORY_FIELDS) == tuple(old.get('RECORD_TRAJECTORY_FIELDS') or ())),
        'non_derived_per_cycle_record_writes_trajectory_fields_only': (
            'per_cycle_fields = PER_CYCLE_RECORD_FIELDS if capture_multiscenario else PER_CYCLE_TRAJECTORY_FIELDS'
            in child_src and 'response_by_cycle = read_per_cycle_response(eval_dir) if capture_multiscenario else {}'
            in child_src and '            if capture_multiscenario:\n                merged.update(' in child_src),
        'alpha_row_checklist_only_if_capture_multiscenario': (
            'alpha_row_checklist = assert_alpha_row_capture_paths() if capture_multiscenario else None' in child_src),
        'hooks_nullcontext_otherwise': ('if capture_multiscenario else nullcontext())' in child_src
                                        and child_src.count('alpha_row_run_hooks(') == 1),
        'w64_record_keys_only_in_the_capture_multiscenario_branch': (
            bool(after_w47) and all(k in after_w47 for k in ("'activation_readback'", "'initialisation_identity'",
                                                               "'response_terminal'", "'per_cycle_response'"))
            and child_src.count("'activation_readback': holder.get") == 1),
        'terminal_captures_only_in_the_capture_multiscenario_branch': (
            bool(terminal) and 'write_response_terminal(' in terminal and 'write_multiscenario_terminal(' in terminal
            and child_src.count('write_response_terminal(') == 1),
        'capture_error_only_if_capture_multiscenario': 'capture_error = capture_multiscenario and any(' in child_src,
        'capture_multiscenario_definition': 'capture_multiscenario = derived is not None or premium is not None' in child_src,
        'pre_w64_revision': PRE_W64_HARNESS_REV,
    }


def _isolation_probe(name, scratch, derived, premium):
    """Freeze a probe spec in `scratch` (outside the repository) and run H._child_real on its one entry with
    G.run_admm_arm replaced by a probe that records which production functions are wrapped and stops BEFORE any
    solve; the SoH floor-row precheck (fresh_planning + ESSO builds, zero solves) is stubbed. Returns the evidence."""
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    import p515_s40_polish_gap as PG
    root = os.path.join(scratch, f'isolation_{name}', 'campaign')
    options = {'investment_year': YEAR}
    if premium is not None:
        options['interface_deviation_premium'] = premium
    configuration = {'name': f'W65 isolation probe ({name})', 'arm_label': ARM_LABEL, 'overrides': {},
                     'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                     'ess_ageing_baseline': json.loads(json.dumps(ESS_AGEING_BASELINE)), 'ess_ageing_baseline_label': LABEL,
                     'note': 'zero-solve probe (W65 ruling 6); never run'}
    if derived is not None:
        configuration['derived_instance'] = derived
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, f'w65_isolation_{name}', [('x0', X0, options)], configuration=configuration, cap=2, concurrency=1,
        authority=['Planner task W65 ruling 6 (zero-solve isolation probe)'], required_consecutive_cycles=10,
        extra={'probe': name})
    entry = spec['candidates'][0]
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    os.makedirs(eval_dir)
    originals = {n: getattr(srp, n) for n in W64_HOOK_NAMES}
    calls = {n: 0 for n in W64_HARNESS_FUNCTIONS}
    saved = {n: getattr(H, n) for n in W64_HARNESS_FUNCTIONS}

    def counting(n):
        orig = saved[n]

        def wrapper(*args, **kwargs):
            calls[n] += 1
            return orig(*args, **kwargs)
        return wrapper

    seen = {}

    def probe_run_admm_arm(label, out_dir, **kwargs):
        seen['reached'] = True
        seen['six_production_functions_at_run'] = {
            n: {'qualname': getattr(getattr(srp, n), '__qualname__', None),
                'module': getattr(getattr(srp, n), '__module__', None),
                'is_original': getattr(srp, n) is originals[n],
                'is_w64_wrapper': str(getattr(getattr(srp, n), '__qualname__', '')).startswith('alpha_row_run_hooks.')}
            for n in W64_HOOK_NAMES}
        seen['calls_at_run'] = dict(calls)
        seen['run_admm_arm_kwargs'] = sorted(kwargs)
        raise _IsolationProbeStop('probe: stopped before the first solve')

    orig_run, orig_floor = G.run_admm_arm, PG._build_floor_rows
    stopped, error = False, None
    try:
        for n in W64_HARNESS_FUNCTIONS:
            setattr(H, n, counting(n))
        G.run_admm_arm = probe_run_admm_arm
        PG._build_floor_rows = lambda precheck_eval_id: ({'stubbed_by': 'W65 isolation probe'}, {}, {})
        H._child_real(SimpleNamespace(spec_sha256=spec_sha), spec, spec_path, entry, eval_dir,
                      {'probe': name, 'note': 'no campaign lock: zero-solve probe'}, H._child_verify_env(), time.time())
    except _IsolationProbeStop:
        stopped = True
    except Exception as err:  # noqa: BLE001 -- recorded; the check then fails
        error = {'error': f'{type(err).__name__}: {err}', 'traceback': traceback.format_exc()}
    finally:
        G.run_admm_arm, PG._build_floor_rows = orig_run, orig_floor
        for n in W64_HARNESS_FUNCTIONS:
            setattr(H, n, saved[n])
    files = sorted(os.path.relpath(os.path.join(d, f), root) for d, _dirs, fs in os.walk(root) for f in fs)
    w64_files = [getattr(H, f) for f in W64_OUTPUT_FILES if os.path.exists(os.path.join(eval_dir, getattr(H, f)))]
    return {'probe': name, 'derived_instance': derived is not None, 'premium': premium, 'campaign_spec_sha256': spec_sha,
            'eval_key': entry['eval_key'], 'stopped_by_probe_before_any_solve': stopped, 'error': error,
            'reached_run_admm_arm': bool(seen.get('reached')), 'six_production_functions_at_run':
            seen.get('six_production_functions_at_run'), 'w64_harness_calls_at_run': seen.get('calls_at_run'),
            'w64_harness_calls_total': calls, 'run_admm_arm_kwargs': seen.get('run_admm_arm_kwargs'),
            'files_written_under_probe_root': files, 'w64_output_files_present': w64_files,
            'init_identity_dir_present': os.path.isdir(os.path.join(root, H.INIT_IDENTITY_DIR_NAME)),
            'six_restored_after': all(getattr(srp, n) is originals[n] for n in W64_HOOK_NAMES),
            'harness_functions_restored_after': all(getattr(H, n) is saved[n] for n in W64_HARNESS_FUNCTIONS)}


def harness_isolation_check(started, scratch):
    """W65 ruling 6, ZERO SOLVES: PARENT_GUARD (SolveProfileGuard(permitted=())) armed at import, verified at exactly 0.
    (A) baseline: a non-derived SRP1-style configuration (the pilot's arm / AA / ageing declarations, NO derived
        instance, NO premium) through H._child_real up to the run: expected 0 / 6 W64 wrappers installed, no W64
        harness function called, no W64 output;
    (B) positive control (the probe can see what it looks for): the smoke's derived / premium configuration: expected
        6 / 6 W64 wrappers installed, the alpha-row checklist and hooks called once each (activation not reached, so
        still no W64 output);
    (C) source checks of the post-run branches and the pre-W64 field tuples. Writes ISOLATION_OUT_REL (write-once)."""
    tag = 'W65-ISOLATION'
    out = os.path.join(REPO, ISOLATION_OUT_REL)
    if os.path.exists(out):
        raise SystemExit(f'output exists (write-once): {ISOLATION_OUT_REL}')
    import p56a_oracle as O
    status_before = H._git(['status', '--porcelain', '--untracked-files=no'])
    baseline = _isolation_probe('baseline_srp1', scratch, None, None)
    # a fresh child process starts with no oracle baseline; drop the SRP1 baseline the probe loaded so the positive
    # control can install the derived instance (install_derived_instance refuses when one is present)
    had_baseline = O._BASELINE is not None
    O._BASELINE = None
    positive = _isolation_probe('positive_control_derived_premium', scratch, derived_declaration(),
                                _premium(UNIT_ALPHA))
    static = _isolation_static_checks()
    status_after = H._git(['status', '--porcelain', '--untracked-files=no'])
    expect = {
        'A_baseline_reached_run_and_stopped': baseline['stopped_by_probe_before_any_solve'] and baseline['reached_run_admm_arm'],
        'A_baseline_no_w64_wrapper_installed': bool(baseline['six_production_functions_at_run']) and not any(
            v['is_w64_wrapper'] for v in baseline['six_production_functions_at_run'].values()),
        'A_baseline_no_w64_harness_call': not any(baseline['w64_harness_calls_total'].values()),
        'A_baseline_no_w64_output': not baseline['w64_output_files_present'] and not baseline['init_identity_dir_present'],
        'A_baseline_restored': baseline['six_restored_after'] and baseline['harness_functions_restored_after'],
        'B_positive_reached_run_and_stopped': positive['stopped_by_probe_before_any_solve'] and positive['reached_run_admm_arm'],
        'B_positive_all_six_w64_wrappers_installed': bool(positive['six_production_functions_at_run']) and all(
            v['is_w64_wrapper'] for v in positive['six_production_functions_at_run'].values()),
        'B_positive_checklist_hooks_and_derived_install_called_once': all(
            positive['w64_harness_calls_total'][n] == 1 for n in ('assert_alpha_row_capture_paths', 'alpha_row_run_hooks',
                                                                  'install_derived_instance')),
        'B_positive_restored': positive['six_restored_after'] and positive['harness_functions_restored_after'],
        'C_static_all': all(v for k, v in static.items() if k != 'pre_w64_revision'),
        'no_tracked_file_changed': status_before == status_after,
    }
    guard_failures = PARENT_GUARD.verify(0)
    payload = {'stage': 'P5.15 W65 ruling 6 -- zero-solve harness-isolation check', 'utc': _utc(),
               'git_head': H._git(['rev-parse', 'HEAD']), 'harness_sha256': H.sha256_file(H.HARNESS_PATH),
               'script_sha256': H.sha256_file(os.path.abspath(__file__)), 'scratch': scratch,
               'baseline': baseline, 'positive_control': positive,
               'oracle_baseline_dropped_between_probes': had_baseline, 'static': static, 'expectations': expect,
               'tracked_status_before': status_before, 'tracked_status_after': status_after,
               'solve_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures,
                               'label': PARENT_GUARD.label if hasattr(PARENT_GUARD, 'label') else None},
               'pass': all(expect.values()) and not guard_failures,
               'scope_note': ('the probe stops at run_admm_arm: the post-run branches are covered by (C) only; no '
                              'full SRP1 campaign bitwise gate was run (known, accepted: spec v25 known_gaps)')}
    H._write_once_json(out, payload)
    PARENT_GUARD.uninstall()
    for k, v in expect.items():
        _log(f'[{tag}]   {k}: {"PASS" if v else "FAIL"}')
    _log(f"[{tag}] baseline calls {baseline['w64_harness_calls_total']}; positive calls "
         f"{positive['w64_harness_calls_total']}; static {static}")
    _log(f"[{tag}] guard {payload['solve_guard']['counts']} verify0={guard_failures}; PASS={payload['pass']}; "
         f"{ISOLATION_OUT_REL} sha256={H.sha256_file(out)}; wall {time.time() - started:.1f}s")
    sys.exit(0 if payload['pass'] else 1)


def main():
    parser = argparse.ArgumentParser(description=STAGE_TEXT)
    parser.add_argument('--stage', choices=sorted(STAGES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--analyse', action='store_true')
    mode.add_argument('--harness-isolation-check', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--pair', type=int, choices=sorted(PAIRS), default=None)
    parser.add_argument('--scratch', default=None)
    args = parser.parse_args()
    started = time.time()
    os.chdir(REPO)
    if args.freeze_spec:
        freeze_spec(started)
    elif args.harness_isolation_check:
        if not args.scratch or os.path.abspath(args.scratch).startswith(REPO + os.sep):
            parser.error('--harness-isolation-check requires --scratch <dir outside the repository>')
        os.makedirs(args.scratch, exist_ok=True)
        harness_isolation_check(started, os.path.abspath(args.scratch))
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
