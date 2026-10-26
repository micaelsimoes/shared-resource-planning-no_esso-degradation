"""
P5.15 Addendum 86 ruling 1 (with Addendum 85 ruling 4 (vi)-(xi), W183g (c), Addendum 86 rulings 2-3), Planner task
W183i (b) -- the STEP-7 FIXED-PLAN LAUNCHER: the certified evaluations E3 (1 cell) and E2 stage 1 (8 cells) under the
Step-7 `settling_resettle` declaration of `p515_s55_step7_resettle_hooks.py` (the v6 rule by import; routed by the
one-branch router patch of the campaign harness, sha256 d9a944f2...), the 1x unit-neighbourhood screen (15
production-exit evaluations) and the cross-evaluation phase (declared only; its entries are added after W186).
A NEW FILE, modelled on the W142 extension v6 launcher (`p515_s53_w142_resettle_ext_v6_campaign.py`: configuration,
pre-launch assertion, spec validation, parent capture checklist, launch command, cell report, per-cell freeze, run).
Helpers of the E1 launcher (`p515_s55_e1_launcher.py`, v2 blob b0841b09 at 7cf88b15: pins, committed-holder scan, memory
preflight, the G0 check, the root resolution) are COPIED, not imported: that file is being edited in parallel
(W183i (c)) and importing it would make this launcher depend on its working-tree state and arm its tripwires.

PHASES AND ROOTS (each cell = one write-once campaign root `<phase root>/campaign_s55_step7_<cell>`; one cell per
process; launch order = the table order)
  e3      P515S55/e3_cnrf        1 cell  e3_cnrf_x0_1x: x = 0, 1x, nrf_option True (+ label); the Step-7 certified
                                 protocol (the hooks module's declaration); persist_certified_models ON
  e2      P515S55/e2_rho         8 cells in the hooks module's order (1x,0.5), (1x,2.0), (2x,0.5), (2x,2.0), each pair
                                 x = 0 then the node-7 reference unit; rho0_scale (+ label); flex_price_multiplier 2.0
                                 (+ label) at 2x; certified protocol; persist_certified_models ON
  screen  P515S55/e1_screen_1x  15 cells: x = 0 under A at 1x (evaluated once; its N alias x0_N_1x by G0) and the 14
                                 admissible unit neighbours of x = 0 (the lattice's unit box at x = 0: the smallest unit
                                 0.25 MVA / 0.5 MWh at every non-empty subset of {5, 7, 9}, 2025 or 2030) under N at 1x;
                                 production exit (cap 500, 10 consecutive passing cycles, Addendum 85 ruling 4 (iii),
                                 (vii)); no resettle declaration; no post-certification
  cross   P515S55/e1_2x/cross    declared only (0 entries until after W186): --freeze and --run refuse
  The configuration block is the W142 extension v6 block (spec 10f4aa7a) minus the re-settling declaration (as the E1
  launcher's), checked key by key against the committed ext-v6 specs at every freeze and run.

MODES (repo root, canonical interpreter, attached, ALONE, both streams captured, never detached)
  --freeze --phase P [--retry-of C]
        ZERO SOLVES. Freezes one write-once campaign root per cell of P (`H.freeze_campaign_spec`), then the phase's
        stage spec `frozen_s55_step7_<P>_spec_v1_<sha8>.json` pinning each cell spec's sha256, every code file, the
        inputs, the committed records the report modes read, the solver, and the exact launch commands. With
        --retry-of C (C has results): freezes `campaign_s55_step7_<C>_r2` with `retry_of` and a retry pin file (a retry
        never writes into a write-once root).
  --run --phase P --cell C --spec-sha256 S [--preconditions-only]
        The launch order within P; the stage spec and the cell spec committed and clean; the cell root holds only its
        spec; harness / hooks / launcher / case / ESS / cost pins and every frozen pin; the pre-launch assertion; the
        parent capture checklist; the solver sha256; G0 (screen x = 0); memory preflight at concurrency 1 (gating).
        Then the campaign lock and `H.evaluate`; write-once results and manifest in the cell root.
  --report --phase e3 | e2      ZERO SOLVES; reads committed records only (named, sha256-pinned); write-once output.
  --dry-run --scratch D         ZERO SOLVES (W183i (b)): keys, counts, holders, scratch freezes, refusals, the rho
                                 replication validation, the report modes on planted records.
  Exit codes: 0 ok; 1 any refusal / precondition / guard / harness failure; 3 STOP FOR REVIEW (E3: the settling
  status not certified within the cap, the NRF terminal capture not written or nrf_capture_error, or n_active per copy
  != 864; E2: the rho read-back does not reproduce bitwise -- a harness-fault stop).

THE RHO READ-BACK (Addendum 86 ruling 2): per E2 cell, per cycle, per channel g in {v, pf, ess}: the recorded
`rho_g_after` must equal BITWISE `sum([x] * n_g) / n_g` in the canonical interpreter, where x starts at S39_g x s and
is carried across cycles from the recorded actions: x <- min(max(x * f, 1e-4), 1e4) for 'increased' (f = 1.5) and
'decreased' (f = 1.0 / 1.5, production's `_scale_admm_penalty`), x unchanged for every factor-1 action (held, fixed,
exempt (fixed), held after solver failure, held (frozen ...), and any of them with ' (exemption lifted)'); an unknown
action refuses. n_v = n_pf = TSO + DSO blocks, n_ess = TSO + DSO + ESSO models, from the record's `solve_profile`.

Objective convention on every value: Q = certified gross_operational_cost, contracted interface settlement EXCLUDED
(Addendum 83 ruling 2); salvage reported separately; net = gross - terminal salvage.
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183i (b) Step-7 launcher parent (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('Step-7 launcher: pickle.load called -- the parent never unpickles')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('Step-7 launcher: pickle.loads called -- the parent never unpickles')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)
import p515_s44_campaign_harness as H  # noqa: E402  (stdlib only at import)
import p515_s55_step7_resettle_hooks as S7  # noqa: E402  (stdlib + hooks modules at import)
import settling_criterion_v6 as SC6  # noqa: E402  (pure)
import p515_s47_phase_b_record as PB  # noqa: E402  (pure lattice primitives and pins; arms its own guard)

pickle.load, pickle.loads = _blocked_load, _blocked_loads

import pyomo.environ as pe  # noqa: E402  (already loaded by the solve guard; the model-build tripwire only)

MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}
_ORIGINAL_CONCRETE_INIT = pe.ConcreteModel.__init__


def _blocked_concrete_init(self, *_a, **_k):
    MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
    raise RuntimeError('Step-7 launcher: a pyomo ConcreteModel was constructed in the parent -- no model build')


pe.ConcreteModel.__init__ = _blocked_concrete_init

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s55_step7_launcher',)
STAGE = ('P5.15 Addendum 86 ruling 1, W183i (b) -- the Step-7 fixed-plan launcher: E3 (C_NRF), E2 stage 1 (rho0 x s), '
         'the 1x unit-neighbourhood screen, the cross-evaluations (declared after W186)')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 86 rulings 1-3; W183i order item (b)',
    'PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 4 (iii), (vi)-(xi), ruling 7, ruling 9; W183g order item (c)',
    'STEP7_SIMULATION_PLAN_2026-10-25.md sections 3, 4, 5.3, 11-13',
    'P5_15_W183I_A_ROUTER.md (the Step-7 hooks module 9318ade3, the patched harness d9a944f2)',
]
OBJECTIVE_CONVENTION = ('Q = certified gross_operational_cost, contracted interface settlement EXCLUDED (Addendum 83 '
                        'ruling 2); salvage reported separately; net = gross - terminal salvage')
P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
LOG_RX = re.compile(r'^[A-Za-z0-9_.-]*(launch|preconditions_only)[A-Za-z0-9_.-]*\.log$')
DRY_RUN_FILE = 'w183i_b_step7_dryrun.json'
DRY_RUN_MANIFEST = 'w183i_b_step7_dryrun_manifest_sha256.json'
SOLVER_PATH = '/usr/local/bin/ipopt'
PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
CONCURRENCY = 1
REQUIRED_CONSECUTIVE_CYCLES = 10

# ---- the patched harness and the hooks module (W183i (a), Addendum 86 ruling 1) ----
HARNESS_EXPECTED_SHA256 = 'd9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191'
HOOKS_REL = 'p515_s55_step7_resettle_hooks.py'
HOOKS_EXPECTED_SHA256 = '9318ade3e6d0d5fec25d0c276c4d82fba5bb163c7da61ec10ee6f49aaf2fa1d2'

# ---- the phases ----
PHASES = ('e3', 'e2', 'screen', 'cross')
PHASE_ROOTS = {'e3': os.path.join(P55, 'e3_cnrf'), 'e2': os.path.join(P55, 'e2_rho'),
               'screen': os.path.join(P55, 'e1_screen_1x'), 'cross': os.path.join(P55, 'e1_2x', 'cross')}
STAGE_ROOTS = tuple(PHASE_ROOTS[p] for p in PHASES)
CERTIFIED_PHASES = ('e3', 'e2')
PHASE_TEXT = {
    'e3': 'E3 (C_NRF): x = 0 at the baseline price with the no-reverse-flow option, certified protocol (Addendum 85 '
          'ruling 4 (viii))',
    'e2': 'E2 stage 1: {x = 0, the node-7 reference unit} x {1x, 2x} x {rho0 x 0.5, x 2.0}, certified protocol '
          '(Addendum 85 ruling 4 (ix); all eight cells, Addendum 86 ruling 3 (i))',
    'screen': 'the 1x unit-neighbourhood screen: the 14 admissible unit neighbours of x = 0 under N + x = 0 under A '
              '(its N alias by G0), production exit (Addendum 85 ruling 4 (vii); plan 5.3, section 12)',
    'cross': 'the E1 cross-evaluations x_A, x_N under A and N (Addendum 85 ruling 4 (vi)); entries declared after W186'}
PERSIST = {'persist_certified_models': True, 'hull_polish': False, 'reference': None}   # Addendum 86 ruling 3 (ii)
SCREEN_CAP = 500
STAGE_SPEC_RX = {p: re.compile(rf'^frozen_s55_step7_{p}_spec_v1_([0-9a-f]{{8}})\.json$') for p in PHASES}
RETRY_SUFFIX = '_r2'

# ---- the frozen configuration: the W142 extension v6 block minus the re-settling declaration (as the E1 launcher) --
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
ESS_AGEING_BASELINE_LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
CONVERGENCE_DEPTH_TAIL = {'enabled': True, 'compl_inf_tol': 1e-06}
ARM_LABEL = 's39_D'
EXT_V6_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_ext_v6')
EXT_V6_REFERENCE_SPEC = {
    'path': os.path.join(EXT_V6_ROOT_REL, 'campaign_s53_w142_resettle_ext_v6_e_no_ageing',
                         'campaign_spec_s53_w142_resettle_ext_v6_e_no_ageing_10f4aa7a.json'),
    'sha256': '10f4aa7a05b5022ecdd347d45aea1070dc1557237ad482e17a0ee1645846d022'}
CONFIG_TEXT_KEYS = ('name', 'note')
MODEL_VARIANT_N = {'ageing_enabled': False, 'available_energy_soh_point': 'end', 'calendar_retention_per_year': 1.0,
                   'eol_retention_r': 0.5}
MODEL_DECLARATIONS = {'A': None, 'N': MODEL_VARIANT_N}
WRAPPER_KEYS = ('certification_continuation', 'settling_continuation', 'settling_extension', 'settling_resettle')

# ---- G0 (the screen's x = 0 alias; Addendum 85 ruling 4 (v), (vii); the E1 launcher's convention) ----
G0_ARTIFACT_REL = os.path.join(P55, 'w183_nl_post_w183c_off', 'w183_nl_digest.json.gz')
G0_ARTIFACT_SHA256 = 'ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a'
G0_RECORD = {'path': os.path.join(P55, 'w183d_g0', 'g0.json'),
             'sha256': '8cef45d684bc7e5dc8b719fd504b8e7705933597a9ba07f44349650046b2911c'}
SCREEN_ALIAS = {'alias': 'x0_N_1x', 'reuse_of': 'x0_A_1x'}

# ---- the committed records the key checks and the report modes read (paths + sha256; never typed values) ----
REC = {
    'w183c_nrf_gates': {'path': os.path.join(P55, 'w183c_nrf_gates', 'w183c_nrf_gates.json'),
                        'sha256': 'c4747232e3e0e9de87f5588defa6e9f19fa8cdcda51601ff340b61d8b239899a',
                        'reads': 'gates.G3_on_rows.on_x0 (config x0_A_1x): eval_key_on (E3 base key) / eval_key_off'},
    'step6_tables': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen',
                                          'frozen_step6_tables_v1_590088fe.json'),
                     'sha256': '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4',
                     'reads': ('tables.cells[*] by eval_key (Q, band_width, status): C_OPEN = E.21 row 1 and the '
                               'baseline E2 pairs (rows 1, 2 at 1x; 34, 35 at 2x); tables.claims[*] by the pair of '
                               'eval keys (d_gross, gross_threshold_or_bar, I_other) as cross-checks')},
    'benchmark_report_v3': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w116_benchmark_nrf',
                                                 'report_v3', 'report_v3.json'),
                            'sha256': 'bf6dbf7247a922020c58e7b9502f63539c0bbf048447c3e33bbbf202c152694a',
                            'reads': ('per_arm_nrf.price_taker (Q(P_NRF) = q_best, its three-start spread, best start); '
                                      'coordinated_cell.eval_key / instance.candidate_key (the x = 0 instance and row '
                                      '1); claim.bands_eur (the between-configuration spread at C*); curtailment_table '
                                      'phase A by agent (Table 12 / E.23 columns)')},
    'price_taker_best_arm': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w116_benchmark_nrf',
                                                  'nrf_arm_price_taker_warm_from_certified_r2',
                                                  'nrf_arm_price_taker_warm_from_certified_r2.json'),
                             'sha256': 'b0362c511c8c490bc4f90f8680c62bd6cdf03ab20cf6f19587e65c703ab87c2b',
                             'reads': ('phase_C_sequential_pass.evaluation.block_components (TSO blocks: the TN term '
                                       'of Q(P_NRF), W124 method) and gross_operational_cost; curtailment per_block '
                                       'day_weight (the instance day weights)')},
    'w182f_records': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w182f_records', 'w182f_records.json'),
                      'sha256': '405c79c9e146600421a76443692fb23eb69d0d35d90358de14c899d66ddda085',
                      'reads': 'Z7_row1_reverse_flow.entries_most_negative_first (the C_OPEN reverse-flow hours, H3c)'},
    'c_open_component_levels': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation',
                                                     'campaign_s53_w101_srp1_cont_x0', 'evals', 'd110bd1a5977df1e_x0',
                                                     'component_levels_terminal.json'),
                                'sha256': '033c1e7eac919969c0cdfbfbb8969484b49b2e83b09a36cd171e7c3068a3a3d1',
                                'reads': 'blocks[*].weighted (the decomposition of Q(C_OPEN) by agent)'},
    'c_open_evaluation_record': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation',
                                                      'campaign_s53_w101_srp1_cont_x0', 'evals', 'd110bd1a5977df1e_x0',
                                                      'evaluation_record.json'),
                                 'sha256': 'c08fa4335c7e266d6d61439fe36d28d56cc7ead31b4956d0f2c6d1fe2b0e8956',
                                 'reads': 'eval_key, certified_cost (cross-check of Q(C_OPEN))'},
    'w183f_arm_per_cycle': {'path': os.path.join(P55, 'w183f_srp1_bitwise_gate', 'arm', 'per_cycle_record.jsonl'),
                            'sha256': 'f85069053693b1541a331cfa4ff1748c366cc50640940772ae99838aeea1db86',
                            'reads': 'rho_{v,pf,ess}_after / _action per cycle (the rho replication validation, s = 1)'},
    'w183f_arm_g': {'path': os.path.join(P55, 'w183f_srp1_bitwise_gate', 'arm', 'g_s39_D.json'),
                    'sha256': 'a147a31392e8e4bde56ff582b37448a7fa96d5cc399eb37e5dcab08b592d6318',
                    'reads': 'solve_profile (n_dso, n_years, n_esso_nodes) and rule_eleven_checklist block counts'},
}
C_OPEN_EVAL_KEY = 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546'   # E.21 row 1 (ref:7aa017f0)
BASELINE_PAIRS = {   # E.21 rows 1, 2 (1x) and 34, 35 (2x): (x = 0 eval key, unit eval key); values read from REC
    '1x': {'x0': C_OPEN_EVAL_KEY, 'unit': '3f084f2ffaeef2b78c9c6930d7f2134e66b61fbbefecef19a5ce0db8b1e9d153',
           'rows': (1, 2)},
    '2x': {'x0': '354b1bdae2b86bce8638689641c7f32fee4c8c0aa896f1d50e9be6312ebee10d',
           'unit': 'dca08628128736042fb1d1fa1a933b0b93f6d4cfea9ef8b4279e3c45d5a6139a', 'rows': (34, 35)}}
W183F_EXPECTED_CYCLE1 = {'pf': (0.1980000000000002, 48), 'ess': (0.010000000000000004, 51),
                         'v': (0.011549999999999998, 48)}   # Addendum 86 ruling 2 (the validation target)

# ---- the rho read-back rule (Addendum 86 ruling 2) ----
RHO_CHANNELS = ('v', 'pf', 'ess')
RHO_INCREASE_FACTOR = 1.5
RHO_DECREASE_FACTOR = 1.5
RHO_CLAMP = (1e-4, 1e4)
RHO_FACTOR_ONE_RX = re.compile(r'^(held|fixed|exempt \(fixed\)|held after solver failure|held \(frozen [^)]*\))'
                               r'( \(exemption lifted\))?$')
RHO_RULE_TEXT = (
    'Addendum 86 ruling 2: per E2 cell, per cycle, per channel g in {v, pf, ess}, the recorded rho_g_after equals '
    'BITWISE sum([x] * n_g) / n_g in the canonical interpreter; x starts at S39_g x s and is carried across cycles from '
    'the recorded actions: increased -> x = min(max(x * 1.5, 1e-4), 1e4); decreased -> x = min(max(x * (1.0 / 1.5), '
    '1e-4), 1e4) (production _scale_admm_penalty: value * factor, factor = 1.0 / decrease_factor); every factor-1 '
    'action (held, fixed, exempt (fixed), held after solver failure, held (frozen ...), any of them with " (exemption '
    'lifted)") keeps x; an unknown action refuses; n_v = n_pf = TSO + DSO blocks, n_ess = TSO + DSO + ESSO models (the '
    'record\'s solve_profile). A cell that does not reproduce bitwise is a harness-fault stop.')

# ---- code files pinned (the run path of every phase) ----
SETTLING_MODULES = ('settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v3.py',
                    'settling_criterion_v4.py', 'settling_criterion_v5.py', 'settling_criterion_v6.py')
RUN_FILES = ('p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py', SCRIPT_NAME, HOOKS_REL, 'p56a_oracle.py',
             'p515_s53_w142_resettle_v6_hooks.py', 'p515_s53_w142_resettle_ext_v6_hooks.py',
             'p515_s53_w139_resettle_v5_hooks.py', 'p515_s53_w132_resettle_v3_hooks.py',
             'p515_s53_w118_resettle_hooks.py', 'p515_s53_w101_settling_continuation_hooks.py',
             'p515_s53_w105_settling_extension_hooks.py', 'p515_s47_phase_b_record.py', 'gate_result_io.py',
             'interface_dual_capture.py') + SETTLING_MODULES
EXTRA_CLEAN_FILES = tuple(dict.fromkeys(RUN_FILES))
NAMED_INPUTS = {'case_file': H.CASE_FILE_REL, 'ess_params_file': H.ESS_PARAMS_FILE_REL,
                'cost_file': PB.COST_FILE['path'],
                'network_params_files': [os.path.join('data', 'SRP1', d, f'{d}_params.json')
                                         for d in ('case9', 'case33_1', 'case33_2', 'case33_3')]}
SEARCH_TABLES = {'investment_cost_w2': dict(PB.INVESTMENT_COST_RESULTS), 'phase_a_tables': dict(PB.PHASE_A_TABLES)}
SIGMA_Q_EXPECTED_EUR = 18449.66
PRODUCTION_DEFINITION = 'tracked root-level *.py not matching ^p\\d (the W182f / W183-0 definition)'
GIB = 1 << 30
MEMORY_PER_SLOT_BYTES = 11 * GIB // 4       # 2.75 GiB (the S47 rule)
MEMORY_RULE = ('hw.memsize - (wired + anonymous + compressor-occupied pages) x page size >= concurrency x 2.75 GiB '
               '(the S47 launcher rule)')


# ======================================================================================================================
#  small helpers (copied from p515_s55_e1_launcher.py v2 b0841b09)
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [STEP7] {msg}', flush=True)


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _rel(path):
    return os.path.relpath(path, REPO)


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(_abs(path)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _sha_or_none(rel):
    return H.sha256_file(_abs(rel)) if os.path.isfile(_abs(rel)) else None


def _git_tracked_clean(rel):
    path = os.path.realpath(_abs(rel))
    if not path.startswith(os.path.realpath(REPO) + os.sep):
        return False, False   # outside the repository (a scratch path): never committed
    rel = os.path.relpath(path, os.path.realpath(REPO))
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
    return tracked, not dirty


def _committed_clean(rel):
    tracked, clean = _git_tracked_clean(rel)
    return tracked and clean


def _jsonable(obj):
    return json.loads(GRIO.dumps(obj))


def _write_once_json(path, obj):
    text = GRIO.dumps(obj, indent=1, sort_keys=True)
    with open(path, 'x') as handle:
        handle.write(text)
    return H.sha256_file(path)


def all_guards():
    seen, out = set(), []
    for name, mod in sorted(sys.modules.items()):
        if not getattr(mod, '__file__', None) or not str(mod.__file__).startswith(REPO + os.sep):
            continue
        for attr, value in sorted(vars(mod).items()):
            if isinstance(value, SolveProfileGuard) and id(value) not in seen:
                seen.add(id(value))
                out.append((f'{name}.{attr}', value))
    if id(PARENT_GUARD) not in seen:
        out.append(('p515_s55_step7_launcher.PARENT_GUARD', PARENT_GUARD))
    return out


def guards_report():
    guards = {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in all_guards()}
    return {'solve_profile_guards': guards,
            'solve_guards_ok': all(not v['verify_0_failures'] for v in guards.values()),
            'pickle': {'counts': dict(PICKLE_COUNTS),
                       'blocked': pickle.load is _blocked_load and pickle.loads is _blocked_loads},
            'model_build_tripwire': dict(MODEL_BUILD_COUNTS),
            'all_zero': (all(not v['verify_0_failures'] for v in guards.values())
                         and not any(PICKLE_COUNTS.values()) and not any(MODEL_BUILD_COUNTS.values())
                         and pickle.load is _blocked_load and pickle.loads is _blocked_loads)}


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS) \
                and 'python' in parts[1]:
            hits.append(line.strip())
    return hits


def record_pins_check():
    """Every committed record a key check or a report mode reads: sha256 on disk = pin, git-tracked and clean."""
    out = {}
    for name, pin in REC.items():
        sha = _sha_or_none(pin['path'])
        tracked, clean = _git_tracked_clean(pin['path']) if sha else (False, False)
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': sha,
                     'holds': sha == pin['sha256'] and tracked and clean, 'reads': pin['reads']}
    return out


def _rec(name):
    pin = REC[name]
    if _sha_or_none(pin['path']) != pin['sha256']:
        raise RuntimeError(f'committed record {name} ({pin["path"]}) does not hash to its pin {pin["sha256"]}')
    return _load(pin['path']) if not pin['path'].endswith('.jsonl') else _read_jsonl(pin['path'])


# ======================================================================================================================
#  configuration, the lattice, the cells
# ======================================================================================================================
def configuration(phase):
    """The W142 extension v6 configuration block (spec 10f4aa7a) minus the re-settling declaration (the E1
    launcher's), with this phase's free text."""
    return {'name': (f'STEP-7 {phase.upper()} (W183i (b), Addenda 85-86) -- the current production configuration as '
                     'the W142 extension v6 declares it: the case file (AA keep_memory declared), the ESS ageing '
                     'baseline ' + ESS_AGEING_BASELINE_LABEL + ' declared (minimum SoH 0.70, NOT overridden), the '
                     'convergence-depth tight tail DECLARED ENABLED (compl_inf_tol 1e-6)'),
            'arm_label': ARM_LABEL, 'overrides': {},
            'case_file_anderson_acceleration': dict(CASE_FILE_AA),
            'ess_ageing_baseline': json.loads(json.dumps(ESS_AGEING_BASELINE)),
            'ess_ageing_baseline_label': ESS_AGEING_BASELINE_LABEL,
            'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL),
            'note': PHASE_TEXT[phase] + ('; the entry adds the Step-7 settling_resettle declaration (keyed) and '
                                         'persist_certified_models' if phase in CERTIFIED_PHASES else
                                         '; no re-settling declaration: production exit (10 consecutive passing '
                                         'cycles, cap 500); no post-certification') + '; concurrency 1'}


def configuration_vs_ext_v6(spec_configuration):
    """Every configuration key but the free text equal to the committed ext-v6 e_no_ageing spec's; the seven ext-v6
    specs agree among themselves (copied from the E1 launcher)."""
    ref_sha = _sha_or_none(EXT_V6_REFERENCE_SPEC['path'])
    ref = _load(EXT_V6_REFERENCE_SPEC['path'])['configuration']
    keys = sorted((set(ref) | set(spec_configuration)) - set(CONFIG_TEXT_KEYS))
    diffs = {k: {'step7': spec_configuration.get(k), 'ext_v6': ref.get(k)} for k in keys
             if spec_configuration.get(k) != ref.get(k)}
    ext_specs = sorted(p for p in H._git(['ls-files', EXT_V6_ROOT_REL]).splitlines()
                       if os.path.basename(p).startswith('campaign_spec_'))
    among = {rel: sorted(k for k in keys if _load(rel)['configuration'].get(k) != ref.get(k)) for rel in ext_specs}
    return {'reference': dict(EXT_V6_REFERENCE_SPEC, sha256_on_disk=ref_sha),
            'reference_sha256_holds': ref_sha == EXT_V6_REFERENCE_SPEC['sha256'], 'compared_keys': keys,
            'differing_keys': diffs, 'equal_except_text': not diffs and ref_sha == EXT_V6_REFERENCE_SPEC['sha256'],
            'ext_v6_specs_agree': len(ext_specs) == 7 and not any(among.values())}


def build_lattice():
    """The PB lattice on the pinned W2 / Phase A tables (copied from the E1 launcher)."""
    ev, failures = {}, []
    for name, pin in SEARCH_TABLES.items():
        got = _sha_or_none(pin['path'])
        ev[f'{name}_sha256'] = got
        if got != pin['sha256']:
            failures.append(f"{name}: {pin['path']} sha256 {got} != pinned {pin['sha256']}")
    if failures:
        return None, ev, failures
    w2 = _load(SEARCH_TABLES['investment_cost_w2']['path'])
    lattice = PB.Lattice(tuple(sorted(PB.unit_costs_from_w2(w2))), PB.unit_costs_from_w2(w2))
    ev['I_x_cross_check_vs_W2'] = PB.w2_cross_check(lattice, w2)
    if not ev['I_x_cross_check_vs_W2']['ok']:
        failures.append(f"I(x) closed form != W2 table: {ev['I_x_cross_check_vs_W2']}")
    ev['sigma_Q'] = PB.sigma_q_from_tables(_load(SEARCH_TABLES['phase_a_tables']['path']))
    if round(ev['sigma_Q']['sigma_Q_eur'], 2) != SIGMA_Q_EXPECTED_EUR:
        failures.append(f"sigma_Q {ev['sigma_Q']['sigma_Q_eur']} != {SIGMA_Q_EXPECTED_EUR}")
    return lattice, ev, failures


UNIT_Z = (0, 0, 1, 2, 0, 0, 0)    # node 7 0.25 MVA / 1 MWh, 2025 (the reference unit)


def i_unit_eur(lattice):
    """I(unit) from the pinned W2 table (the lattice's closed form, cross-checked against W2 by build_lattice)."""
    return lattice.investment_cost(UNIT_Z)


def _cdef(cell, phase, nodes, year, model, flex, rho, nrf, decl, extra=None):
    can = H.canonical_candidate(nodes, investment_year=year)
    out = {'cell': cell, 'phase': phase, 'nodes': {int(k): tuple(map(float, v)) for k, v in nodes.items()},
           'investment_year': year, 'canonical': can, 'candidate_key': H.candidate_key(can), 'model': model,
           'model_variant': copy.deepcopy(MODEL_DECLARATIONS[model]), 'flex_price_multiplier': flex,
           'rho0_scale': rho, 'nrf_option': nrf, 'declaration': decl}
    out.update(extra or {})
    return out


def cell_table(phase, lattice=None):
    """The ordered cells of a phase (launch order)."""
    if phase in CERTIFIED_PHASES:
        names = S7.E3_CELLS if phase == 'e3' else S7.E2_CELLS
        out = []
        for cell in names:
            c = S7.CELLS[cell]
            out.append(_cdef(cell, phase, c['nodes'], c['investment_year'], 'A', c['flex_price_multiplier'],
                             c['rho0_scale'], True if c['nrf_option'] else None, S7.declaration_for(cell),
                             {'plan': c['plan'], 'item': c['item'], 'role': c['role']}))
        return out
    if phase == 'screen':
        if lattice is None:
            raise ValueError('the screen needs the lattice')
        x0 = lattice.x0()
        out = [_cdef('x0_A_1x', phase, lattice.nodes_map(x0), lattice.year_of(x0), 'A', None, None, None, None,
                     {'z': list(x0), 'lattice_label': lattice.label(x0),
                      'role': 'x = 0 at 1x on the final code (evaluated once; its N seed is the G0 alias x0_N_1x)'})]
        for z in lattice.neighbourhood(x0):
            out.append(_cdef(f'{lattice.label(z)}_N_1x', phase, lattice.nodes_map(z), lattice.year_of(z), 'N', None,
                             None, None, None,
                             {'z': list(z), 'lattice_label': lattice.label(z), 'I_x_eur': lattice.investment_cost(z),
                              'role': 'an admissible unit neighbour of x = 0 under N at 1x'}))
        return out
    return []   # cross: declared only


def entry_options(cdef):
    opts = {'investment_year': cdef['investment_year']}
    if cdef['phase'] in CERTIFIED_PHASES:
        opts['post_certification'] = dict(PERSIST)
        opts['settling_resettle'] = copy.deepcopy(cdef['declaration'])
    if cdef['model_variant'] is not None:
        opts['model_variant'] = copy.deepcopy(cdef['model_variant'])
    if cdef['flex_price_multiplier'] is not None:
        opts['flex_price_multiplier'] = cdef['flex_price_multiplier']
    if cdef['rho0_scale'] is not None:
        opts['rho0_scale'] = cdef['rho0_scale']
    if cdef['nrf_option'] is not None:
        opts['nrf_option'] = cdef['nrf_option']
    return opts


def entries_of(cdef):
    return [(cdef['cell'], cdef['nodes'], entry_options(cdef))]


def cap_of(cdef):
    return S7.spec_cap(cdef['cell']) if cdef['phase'] in CERTIFIED_PHASES else SCREEN_CAP


# ======================================================================================================================
#  keys
# ======================================================================================================================
def eval_key_of(cdef, with_declaration=True, drop=()):
    """The harness key (`H.evaluation_key`) of a cell in this configuration; `drop` removes options by name."""
    kw = dict(case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
              convergence_depth_tail=CONVERGENCE_DEPTH_TAIL, model_variant=cdef['model_variant'],
              flex_price_multiplier=cdef['flex_price_multiplier'], rho0_scale=cdef['rho0_scale'],
              nrf_option=cdef['nrf_option'])
    for name in drop:
        kw[name] = None
    if with_declaration and cdef['declaration'] is not None:
        kw['settling_resettle'] = cdef['declaration']
    return H.evaluation_key(cdef['candidate_key'], {}, **kw)


def independent_key_payload(cdef):
    """The key payload composed HERE from the documented W183a formula (tail declared enabled -> the tail branch of
    `H.evaluation_key`), so the dry run shows which options the key carries: {candidate_key, overrides,
    effective_anderson_acceleration, [model_variant], ess_ageing_baseline, [flex_price_multiplier if != 1.0],
    convergence_depth_tail, [rho0_scale if != 1.0], [nrf_option if True]}; then, for a certified cell,
    {base_evaluation_key, settling_resettle}."""
    p = {'candidate_key': cdef['candidate_key'], 'overrides': {},
         'effective_anderson_acceleration': H.effective_anderson_acceleration(CASE_FILE_AA, {})}
    if cdef['model_variant'] is not None:
        p['model_variant'] = H.validate_model_variant(cdef['model_variant'])
    p['ess_ageing_baseline'] = H.validate_ess_ageing_baseline(ESS_AGEING_BASELINE)
    if cdef['flex_price_multiplier'] not in (None, 1.0):
        p['flex_price_multiplier'] = cdef['flex_price_multiplier']
    p['convergence_depth_tail'] = H.convergence_depth_tail_in_key(CONVERGENCE_DEPTH_TAIL)
    if cdef['rho0_scale'] not in (None, 1.0):
        p['rho0_scale'] = cdef['rho0_scale']
    if cdef['nrf_option'] is True:
        p['nrf_option'] = True
    base = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    out = {'base_payload': p, 'base_key': base, 'eval_key': base}
    if cdef['declaration'] is not None:
        w = {'base_evaluation_key': base, 'settling_resettle': S7.validate_settling_resettle(cdef['declaration'])}
        out['eval_key'] = hashlib.sha256(json.dumps(w, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return out


def campaign_id_of(cell, retry=False):
    return f's55_step7_{H._sanitize_id(cell)}' + (RETRY_SUFFIX if retry else '')


def cell_root(base_abs, cell, retry=False):
    return os.path.join(base_abs, f'campaign_{campaign_id_of(cell, retry)}')


def committed_eval_keys(exclude_roots=STAGE_ROOTS):
    """eval key -> [git-tracked campaign specs holding it], over every data/*campaign_spec_*.json OUTSIDE the stage
    roots (a pre-run check that scans committed artefacts excludes the run's own roots, CLAUDE.md / Addendum 55)."""
    out, n, excluded = {}, 0, []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        if _excluded(rel, exclude_roots):
            excluded.append(rel)
            continue
        n += 1
        for e in _load(rel).get('candidates') or []:
            out.setdefault(H._entry_eval_key(e), []).append(rel)
    return out, n, excluded


def _excluded(rel, roots):
    return any(rel == r or rel.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def key_checks(cdef, lattice=None):
    """Recompute a cell's keys: the harness key == the independently composed payload's sha256; the key differs
    from the same cell without each declared option; E3's base = W183c's committed x0_A_1x + NRF key."""
    ind = independent_key_payload(cdef)
    k = eval_key_of(cdef)
    base = eval_key_of(cdef, with_declaration=False)
    out = {'eval_key': k, 'base_key': base, 'independent': ind,
           'checks': {'harness_key_equals_independent_payload_key': k == ind['eval_key'],
                      'harness_base_equals_independent_base': base == ind['base_key']}}
    for opt in ('nrf_option', 'rho0_scale', 'flex_price_multiplier', 'model_variant'):
        if cdef[opt] is not None:
            without = eval_key_of(cdef, drop=(opt,))
            out[f'key_without_{opt}'] = without
            out['checks'][f'key_differs_without_{opt}'] = without != k
            out['checks'][f'base_payload_carries_{opt}'] = (opt in ind['base_payload']
                                                            and ind['base_payload'][opt] == (
                                                                H.validate_model_variant(cdef[opt])
                                                                if opt == 'model_variant' else cdef[opt]))
    if cdef['declaration'] is not None:
        out['checks']['key_differs_from_base'] = k != base
    if cdef['phase'] == 'e3':
        g = _rec('w183c_nrf_gates')['gates']['G3_on_rows']['on_x0']
        out['w183c_on_x0'] = {'config': g.get('config'), 'eval_key_on': g.get('eval_key_on'),
                              'eval_key_off': g.get('eval_key_off')}
        out['checks']['e3_base_equals_w183c_committed_x0_A_1x_nrf_key'] = (g.get('config') == 'x0_A_1x'
                                                                          and base == g.get('eval_key_on'))
        out['checks']['e3_base_without_nrf_equals_w183c_off_key'] = (
            eval_key_of(cdef, with_declaration=False, drop=('nrf_option',)) == g.get('eval_key_off'))
        out['checks']['e3_key_carries_nrf_option'] = ind['base_payload'].get('nrf_option') is True
    if cdef['phase'] == 'e2':
        out['checks']['e2_key_carries_rho0_scale'] = ind['base_payload'].get('rho0_scale') == cdef['rho0_scale'] \
            and cdef['rho0_scale'] in (0.5, 2.0)
        out['checks']['e2_key_carries_the_price'] = (
            ind['base_payload'].get('flex_price_multiplier') == 2.0 if cdef['flex_price_multiplier'] == 2.0
            else 'flex_price_multiplier' not in ind['base_payload'])
    if cdef['phase'] == 'screen':
        out['checks']['screen_1x_no_price_in_key'] = 'flex_price_multiplier' not in ind['base_payload']
        out['checks']['screen_model_as_declared'] = (
            ind['base_payload'].get('model_variant') == H.validate_model_variant(MODEL_VARIANT_N)
            if cdef['model'] == 'N' else 'model_variant' not in ind['base_payload'])
    return out


# ======================================================================================================================
#  G0 (copied from the E1 launcher, the price tag a parameter: the screen runs at 1x)
# ======================================================================================================================
def production_modules():
    return sorted(p for p in H._git(['ls-files', '*.py']).splitlines()
                  if p.strip() and '/' not in p and not re.match(r'^p\d', p))


def production_now():
    mods = production_modules()
    return {'definition': PRODUCTION_DEFINITION, 'head': H._git(['log', '-1', '--format=%H', '--'] + mods),
            'modules_sha256': {m: _sha(m) for m in mods}}


def _load_g0(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(_abs(path), 'rt') as handle:
        return json.load(handle)


def g0_check(x0_keys, require_current_production, tag='1x', g0_path=G0_ARTIFACT_REL, g0_sha_pin=G0_ARTIFACT_SHA256):
    """The alias justification at `tag`: the artifact hashes to its pin, is committed, reports NL identity of x = 0
    under A and N at `tag` (ESSO NL every node; network NL both builders), with its x = 0 keys equal to this launcher's
    and its configuration / N declaration equal to this launcher's; with `require_current_production` its production
    modules equal the current production. The W183d G0 record (g0.json) hashes to its pin and reads G0_PASS True."""
    info = {'path': g0_path, 'sha256_pinned': g0_sha_pin, 'exists': os.path.isfile(_abs(g0_path)), 'tag': tag}
    checks = {'exists': info['exists']}
    if not info['exists']:
        return False, dict(info, checks=checks)
    info['sha256_on_disk'] = _sha(g0_path)
    checks['sha256_equals_pin'] = info['sha256_on_disk'] == g0_sha_pin
    checks['git_tracked_and_clean'] = _committed_clean(g0_path)
    if not checks['sha256_equals_pin']:
        return False, dict(info, checks=checks)
    d = _load_g0(g0_path)
    comp = (d.get('summary') or {}).get(f'i_x0_A_{tag}_vs_x0_N_{tag}') or {}
    seq, par = comp.get('sequential') or {}, comp.get('parallel') or {}
    keys = (d.get('summary') or {}).get('eval_keys') or {}
    spec_like = ((d.get('configuration') or {}).get('spec_like') or {}).get('configuration') or {}
    checks.update({
        'artifact_all_ok': d.get('all_ok') is True,
        'artifact_guard_zero_solves': ((d.get('checks') or {}).get('guard_zero_solves_verified') is True),
        f'x0_A_vs_N_{tag}_esso_nl_plain_identical_every_node': (
            bool(seq.get('esso_nl_plain_identical_by_node'))
            and all(v is True for v in seq['esso_nl_plain_identical_by_node'].values())),
        f'x0_A_vs_N_{tag}_esso_nl_all_four_files_identical_every_node': (
            bool(seq.get('esso_nl_all_four_files_identical_by_node'))
            and all(v is True for v in seq['esso_nl_all_four_files_identical_by_node'].values())),
        f'x0_A_vs_N_{tag}_network_nl_identical_sequential': (seq.get('network_nl_plain_identical_all') is True
                                                              and seq.get('network_nl_all_four_files_identical_all')
                                                              is True),
        f'x0_A_vs_N_{tag}_network_nl_identical_parallel': (par.get('network_nl_plain_identical_all') is True
                                                            and par.get('network_nl_all_four_files_identical_all')
                                                            is True),
        f'artifact_x0_A_{tag}_key_equals_alias_target': keys.get(f'x0_A_{tag}') == x0_keys['A'],
        f'artifact_x0_N_{tag}_key_equals_alias_key': keys.get(f'x0_N_{tag}') == x0_keys['N'],
        'artifact_configuration_equals_launcher': (
            spec_like.get('case_file_anderson_acceleration') == CASE_FILE_AA
            and spec_like.get('convergence_depth_tail') == CONVERGENCE_DEPTH_TAIL
            and spec_like.get('ess_ageing_baseline') == ESS_AGEING_BASELINE
            and (spec_like.get('ess_params_file') or {}).get('sha256') == _sha(H.ESS_PARAMS_FILE_REL)),
        'artifact_model_N_equals_launcher': ((d.get('configuration') or {}).get('models') or {}).get('N')
        == MODEL_VARIANT_N,
    })
    g0rec_sha = _sha_or_none(G0_RECORD['path'])
    checks['w183d_g0_record_sha256_equals_pin'] = g0rec_sha == G0_RECORD['sha256']
    checks['w183d_g0_record_G0_PASS'] = bool(g0rec_sha == G0_RECORD['sha256']
                                             and _load(G0_RECORD['path']).get('G0_PASS') is True)
    prod_now = production_now()
    art_prod = (d.get('production') or {}).get('modules_sha256') or {}
    differing = sorted(m for m in set(prod_now['modules_sha256']) | set(art_prod)
                       if prod_now['modules_sha256'].get(m) != art_prod.get(m))
    info['artifact_production'] = {'expected_commit': (d.get('production') or {}).get('expected_commit'),
                                   'git_head_at_run': d.get('git_head_at_run'),
                                   'equals_current_production': not differing,
                                   'modules_differing_from_current': differing,
                                   'current_production_head': prod_now['head']}
    info['artifact_eval_keys'] = {k: keys.get(k) for k in (f'x0_A_{tag}', f'x0_N_{tag}')}
    if require_current_production:
        checks['artifact_production_equals_current_production'] = not differing
    accepted = all(checks.values())
    return accepted, dict(info, checks=checks, accepted=accepted, require_current_production=require_current_production)


def screen_alias(lattice, require_current_production=True):
    x0 = lattice.x0()
    a = _cdef('x0_A_1x', 'screen', lattice.nodes_map(x0), lattice.year_of(x0), 'A', None, None, None, None)
    n = _cdef('x0_N_1x', 'screen', lattice.nodes_map(x0), lattice.year_of(x0), 'N', None, None, None, None)
    keys = {'A': eval_key_of(a), 'N': eval_key_of(n)}
    ok, info = g0_check(keys, require_current_production)
    return ok, {'alias_label': SCREEN_ALIAS['alias'], 'alias_eval_key': keys['N'],
                'reuse_of_label': SCREEN_ALIAS['reuse_of'], 'reuse_of': keys['A'], 'justification': 'G0',
                'g0_artifact_path': G0_ARTIFACT_REL, 'g0_artifact_sha256': info.get('sha256_on_disk'),
                'g0_accepted': ok, 'g0': info,
                'rule': ('x = 0 is evaluated once at 1x, under A; its N cell reuses that evaluation by the G0 identity '
                         '(the model IPOPT receives at x = 0 is byte-identical under A and N); never an evaluation')}


# ======================================================================================================================
#  pins (copied from the E1 launcher, with this launcher's run files and the committed records)
# ======================================================================================================================
def tracked_inputs():
    return sorted(p for p in H._git(['ls-files', 'data/SRP1']).splitlines()
                  if p.strip() and not p.startswith(('data/SRP1/Results/', 'data/SRP1/Diagrams/')))


def imported_repo_modules():
    out = {}
    for _name, mod in sorted(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and str(f).startswith(REPO + os.sep) and str(f).endswith('.py'):
            out[_rel(f)] = _sha(_rel(f))
    return out


def solver_check():
    res = H._resolve_solver_path_from_dotenv()
    path = res.get('NLP_SOLVER_PATH')
    sha = H.sha256_file(path) if path and os.path.isfile(path) else None
    return {'resolved': res, 'expected': SOLVER_PATH, 'sha256': sha, 'ok': path == SOLVER_PATH and sha is not None}


def pins_now():
    prod = production_now()
    run_files = {f: _sha_or_none(f) for f in RUN_FILES}
    imported = imported_repo_modules()
    inputs = {f: _sha(f) for f in tracked_inputs()}
    tables = {name: {'path': pin['path'], 'sha256': _sha_or_none(pin['path'])} for name, pin in SEARCH_TABLES.items()}
    records = {name: {'path': pin['path'], 'sha256': _sha_or_none(pin['path'])} for name, pin in REC.items()}
    every = sorted(set(prod['modules_sha256']) | set(run_files) | set(imported) | set(inputs)
                   | {t['path'] for t in tables.values()} | {r['path'] for r in records.values()}
                   | {G0_ARTIFACT_REL, G0_RECORD['path'], EXT_V6_REFERENCE_SPEC['path']})
    status = H._git(['status', '--porcelain', '--'] + every)
    untracked = [f for f in every if not H._git(['ls-files', '--', f]).strip()]
    return {'git_head_at_pin': H._git(['rev-parse', 'HEAD']), 'production': prod,
            'run_files_sha256': run_files, 'imported_repo_modules_sha256': imported,
            'inputs': {'definition': 'every tracked file under data/SRP1 outside Results/ and Diagrams/',
                       'sha256': inputs},
            'named_inputs': {k: ({'path': v, 'sha256': inputs.get(v)} if isinstance(v, str)
                                 else [{'path': p, 'sha256': inputs.get(p)} for p in v])
                             for k, v in NAMED_INPUTS.items()},
            'search_tables': tables, 'committed_records': records,
            'g0': {'artifact': {'path': G0_ARTIFACT_REL, 'sha256': _sha_or_none(G0_ARTIFACT_REL)},
                   'record': {'path': G0_RECORD['path'], 'sha256': _sha_or_none(G0_RECORD['path'])}},
            'ext_v6_reference_spec': {'path': EXT_V6_REFERENCE_SPEC['path'],
                                      'sha256': _sha_or_none(EXT_V6_REFERENCE_SPEC['path'])},
            'uncommitted_changes': [ln for ln in status.splitlines() if ln.strip()], 'untracked': untracked}


def pins_compare(frozen, now):
    """Every pinned item of `frozen` against `now` (the HEAD itself is not pinned). Empty = identical."""
    diffs = []
    if frozen['production']['head'] != now['production']['head']:
        diffs.append(f"production head {frozen['production']['head']} -> {now['production']['head']}")
    a, b = frozen['production']['modules_sha256'], now['production']['modules_sha256']
    for m in sorted(set(a) | set(b)):
        if a.get(m) != b.get(m):
            diffs.append(f'production module {m}: {a.get(m)} -> {b.get(m)}')
    for group in ('run_files_sha256', 'imported_repo_modules_sha256'):
        for f, sha in sorted(frozen[group].items()):
            if _sha_or_none(f) != sha:
                diffs.append(f'{group} {f}: {sha} -> {_sha_or_none(f)}')
    a, b = frozen['inputs']['sha256'], now['inputs']['sha256']
    for f in sorted(set(a) | set(b)):
        if a.get(f) != b.get(f):
            diffs.append(f'input {f}: {a.get(f)} -> {b.get(f)}')
    for group in ('search_tables', 'committed_records'):
        for name, t in sorted(frozen[group].items()):
            if _sha_or_none(t['path']) != t['sha256']:
                diffs.append(f"{group} {name}: {t['sha256']} -> {_sha_or_none(t['path'])}")
    for name in ('artifact', 'record'):
        t = frozen['g0'][name]
        if _sha_or_none(t['path']) != t['sha256']:
            diffs.append(f"g0 {name}: {t['sha256']} -> {_sha_or_none(t['path'])}")
    return diffs


def pins_recorded_check(pins):
    named = pins.get('named_inputs') or {}
    return {
        'production_head': bool((pins.get('production') or {}).get('head')),
        'production_modules_sha256': bool((pins.get('production') or {}).get('modules_sha256')),
        **{f'run_file:{f}': bool((pins.get('run_files_sha256') or {}).get(f)) for f in RUN_FILES},
        'run_file_harness_is_the_patched_harness': (pins.get('run_files_sha256') or {}).get(
            'p515_s44_campaign_harness.py') == HARNESS_EXPECTED_SHA256,
        'run_file_hooks_is_w183i_a': (pins.get('run_files_sha256') or {}).get(HOOKS_REL) == HOOKS_EXPECTED_SHA256,
        'case_file': bool((named.get('case_file') or {}).get('sha256')),
        'ess_params_file_equals_pb_pin': (named.get('ess_params_file') or {}).get('sha256') == PB.ESS_PARAMS_FILE['sha256'],
        'cost_file_equals_pb_pin': (named.get('cost_file') or {}).get('sha256') == PB.COST_FILE['sha256'],
        'network_params_files': all(e.get('sha256') for e in named.get('network_params_files') or [{}]),
        'search_tables': all(t.get('sha256') for t in (pins.get('search_tables') or {}).values()),
        'committed_records_equal_their_pins': all((pins.get('committed_records') or {}).get(n, {}).get('sha256')
                                                  == REC[n]['sha256'] for n in REC),
        'g0_artifact_equals_pin': ((pins.get('g0') or {}).get('artifact') or {}).get('sha256') == G0_ARTIFACT_SHA256,
        'imported_repo_modules': bool(pins.get('imported_repo_modules_sha256')),
    }


def memory_preflight(concurrency):
    required = int(concurrency) * MEMORY_PER_SLOT_BYTES
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(text.splitlines()[0].split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    need = ('Pages wired down', 'Anonymous pages', 'Pages occupied by compressor')
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    if any(raw.get(k) is None for k in need):
        return {'pass': False, 'missing': [k for k in need if raw.get(k) is None], 'rule': MEMORY_RULE,
                'concurrency': int(concurrency)}
    avail = total - sum(raw[k] for k in need) * page
    return {'utc': _utc(), 'concurrency': int(concurrency), 'per_slot_gib': MEMORY_PER_SLOT_BYTES / GIB,
            'available_gib': avail / GIB, 'required_gib': required / GIB, 'pass': avail >= required,
            'rule': MEMORY_RULE, 'hw_memsize_gib': total / GIB, 'page_size': page,
            'pages': {k: raw.get(k) for k in need + ('Pages free', 'Pages inactive')}}


def code_sha256():
    return {f: _sha_or_none(f) for f in RUN_FILES}


# ======================================================================================================================
#  the capture checklists, the pre-launch assertion, the spec validation
# ======================================================================================================================
def rho_wiring_checklist():
    """E2 (rule eleven): `apply_rho0_scale` is wired (the configuration hook calls it after the S39 checks) and the
    per-cycle rho_{v,pf,ess}_after / _action fields are captured (harness trajectory fields; production writes them)."""
    import inspect
    import shared_resources_planning as srp
    hook_src = inspect.getsource(H._config_hook_factory)
    src = inspect.getsource(srp)
    checks = {'apply_rho0_scale_callable': callable(getattr(H, 'apply_rho0_scale', None)),
              'config_hook_calls_apply_rho0_scale': 'apply_rho0_scale(' in hook_src,
              'config_hook_records_w183a_rho0_scale': "['w183a_rho0_scale']" in hook_src,
              'validate_rho0_scale_callable': callable(getattr(H, 'validate_rho0_scale', None)),
              'production_penalty_summary_is_a_mean_over_models': (
                  'sum(values) / max(len(values), 1)' in inspect.getsource(srp._get_admm_penalty_summary)),
              'production_scale_rule_value_times_factor_clamped': (
                  'min(max(value, params' in inspect.getsource(srp._scale_admm_penalty))}
    for g in RHO_CHANNELS:
        for f in (f'rho_{g}_after', f'rho_{g}_action'):
            checks[f'harness_trajectory_field_{f}'] = f in H.PER_CYCLE_TRAJECTORY_FIELDS
            checks[f'production_writes_{f}'] = f"'{f}':" in src
    return checks


def parent_capture_checklist(cdef, spec):
    """Before each run (and at freeze): H.assert_record_capture_paths(); E3 also H.assert_nrf_capture_path(); E3/E2
    the hooks module's resettle preconditions on the frozen spec; E2 the rho wiring. -> (ok, checks)."""
    out = {}
    try:
        out['record_capture_paths'] = H.assert_record_capture_paths()
    except Exception as error:  # noqa: BLE001
        out['record_capture_paths'] = {'error': f'{type(error).__name__}: {error}'}
    if cdef['phase'] == 'e3':
        try:
            out['nrf_capture_path'] = H.assert_nrf_capture_path()
        except Exception as error:  # noqa: BLE001
            out['nrf_capture_path'] = {'error': f'{type(error).__name__}: {error}'}
    if cdef['phase'] in CERTIFIED_PHASES:
        decl = ((spec or {}).get('candidates') or [{}])[0].get('settling_resettle')
        tail_on = bool(((spec or {}).get('configuration') or {}).get('convergence_depth_tail', {}).get('enabled'))
        aa_on = bool(((spec or {}).get('configuration') or {}).get('case_file_anderson_acceleration', {}).get('enabled'))
        try:
            out['step7_resettle_preconditions'] = S7.assert_resettle_preconditions(
                decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
        except Exception as error:  # noqa: BLE001
            out['step7_resettle_preconditions'] = {'error': f'{type(error).__name__}: {error}'}
    if cdef['phase'] == 'e2':
        try:
            out['rho_wiring'] = rho_wiring_checklist()
        except Exception as error:  # noqa: BLE001
            out['rho_wiring'] = {'error': f'{type(error).__name__}: {error}'}

    def _ok(v):
        if not isinstance(v, dict) or 'error' in v:
            return False
        flat = v['checks'] if isinstance(v.get('checks'), dict) else v
        return bool(flat) and all(x is True for x in flat.values())
    ok = all(_ok(v) for v in out.values())
    return ok, out


def pre_launch_assertion(cdef, base_abs, retry=False, spec=None, holders=None):
    """The eval key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it, its eval dir
    and working dirs); for E3 / E2 it appears in no committed campaign spec OUTSIDE the stage roots (refused
    otherwise); for the screen a committed holder is REPORTED and never reused (Addendum 85 W183g (c)); the working
    dirs are absent."""
    kc = key_checks(cdef)
    k = kc['eval_key']
    if holders is None:
        holders, _n, _ex = committed_eval_keys()
    cid = campaign_id_of(cdef['cell'], retry)
    eval_dir_name = H.eval_dir_name(k, cdef['cell'])
    ids = H.eval_ids(cid, k)
    import p56a_oracle as O   # its WORK_DIR constant; the import builds / solves nothing
    work = O.WORK_DIR
    parts = dict(kc['checks'])
    held_by = holders.get(k, [])
    if cdef['phase'] in CERTIFIED_PHASES:
        parts['eval_key_absent_from_committed_specs_outside_the_stage_roots'] = not held_by
    parts['working_dirs_absent'] = not any(os.path.exists(os.path.join(work, i)) for i in ids.values())
    entry = None
    if spec is not None:
        entry = next((x for x in spec.get('candidates') or [] if x['label'] == cdef['cell']), None)
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    return {'holds': all(parts.values()), 'parts': parts, 'eval_key': k, 'base_key': kc['base_key'],
            'eval_dir_name': eval_dir_name, 'working_dir_ids': ids, 'campaign_id': cid,
            'committed_holders_outside_the_stage_roots': held_by,
            'committed_holder_rule': ('E3 / E2: refused if held (key collision); screen: reported, never reused (no '
                                      'cost is carried across code states)'),
            'excluded_roots': list(STAGE_ROOTS)}


def validate_campaign_spec(cdef, spec, retry=False):
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    vs6 = configuration_vs_ext_v6(cfg)
    k = eval_key_of(cdef)
    cid = campaign_id_of(cdef['cell'], retry)
    extra = spec.get('extra') or {}
    checks = {
        'configuration_equals_ext_v6_minus_resettle': vs6['equal_except_text'],
        'configuration_case_file_sha256': cfg.get('case_file_sha256') == H.sha256_file(H.CASE_FILE),
        'configuration_ess_params_file_pinned': (cfg.get('ess_params_file') or {}).get('sha256')
        == PB.ESS_PARAMS_FILE['sha256'],
        'one_entry': len(ents) == 1, 'entry_label_is_the_cell': e.get('label') == cdef['cell'],
        'entry_eval_key_recomputes': e.get('eval_key') == k,
        'entry_candidate_key': e.get('key') == cdef['candidate_key'],
        'entry_canonical': e.get('canonical') == cdef['canonical'],
        'entry_eval_dir': e.get('eval_dir') == H.eval_dir_name(k, cdef['cell']),
        'entry_working_dir_ids': e.get('working_dir_ids') == H.eval_ids(cid, k),
        'entry_overrides_empty': e.get('overrides') == {} and cfg.get('overrides') == {},
        'campaign_id': spec.get('campaign_id') == cid,
        'cap': spec.get('cap') == cap_of(cdef),
        'concurrency_1': spec.get('concurrency') == CONCURRENCY,
        'ten_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'harness_is_the_patched_harness': (spec.get('harness') or {}).get('sha256') == HARNESS_EXPECTED_SHA256,
        'script_recorded': extra.get('campaign_script') == SCRIPT_NAME,
        'cell_recorded': extra.get('cell') == cdef['cell'] and extra.get('phase') == cdef['phase'],
        'pins_recorded': all(pins_recorded_check(extra.get('pins') or {}).values()),
        'no_early_stop_anywhere': 'early_stop' not in json.dumps(e),
        'entry_model_variant_as_declared': (json.dumps(e.get('model_variant'), sort_keys=True)
                                            == json.dumps(cdef['model_variant'], sort_keys=True)),
    }
    if cdef['phase'] in CERTIFIED_PHASES:
        checks['entry_settling_resettle_is_the_step7_declaration'] = (
            json.dumps(e.get('settling_resettle'), sort_keys=True)
            == json.dumps(S7.declaration_for(cdef['cell']), sort_keys=True))
        checks['entry_post_certification_persist_on'] = e.get('post_certification') == PERSIST
        checks['harness_routes_the_declaration_to_the_step7_module'] = (
            H.resettle_hooks_module(e.get('settling_resettle')) is S7)
        for name, v in S7.spec_entry_checks(S7.declaration_for(cdef['cell']), spec).items():
            checks[f'hooks:{name}'] = bool(v)
        checks['no_other_wrapper'] = not any(x in e for x in WRAPPER_KEYS if x != 'settling_resettle')
    else:
        checks['no_wrapper_declaration'] = not any(x in e for x in WRAPPER_KEYS)
        checks['no_post_certification'] = e.get('post_certification') is None
        checks['no_rho0_scale_nrf_or_price'] = not any(x in e for x in ('rho0_scale', 'nrf_option',
                                                                        'flex_price_multiplier', 'rho0_scale_label',
                                                                        'nrf_option_label', 'flex_price_label'))
        checks['variant_label_iff_N'] = ((cdef['model'] == 'N') == (e.get('model_variant_label')
                                                                    == H.MODEL_VARIANT_LABEL)
                                         and (cdef['model'] == 'N') == (spec.get('model_variant_label')
                                                                        == H.MODEL_VARIANT_LABEL))
    if retry:
        checks['retry_of_recorded'] = bool(extra.get('retry_of'))
    return checks, vs6


# ======================================================================================================================
#  launch commands
# ======================================================================================================================
def launch_command(phase, cell, spec_sha, preconditions_only=False):
    log = f'run_{cell}_launch.log' if not preconditions_only else f'run_{cell}_preconditions_only.log'
    return (f'cd {REPO} && set -o noclobber && NLP_SOLVER_PATH={SOLVER_PATH} {PYTHON} -u {SCRIPT_NAME} --run --phase '
            f'{phase} --cell {cell} --spec-sha256 {spec_sha}' + (' --preconditions-only' if preconditions_only else '')
            + f' > {os.path.join(PHASE_ROOTS[phase], log)} 2>&1')


def report_command(phase):
    return (f'cd {REPO} && set -o noclobber && {PYTHON} -u {SCRIPT_NAME} --report --phase {phase} > '
            f'{os.path.join(PHASE_ROOTS[phase], f"report_{phase}_launch.log")} 2>&1')


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def _stage_spec_files(base_abs, phase):
    if not os.path.isdir(base_abs):
        return []
    return sorted(f for f in os.listdir(base_abs) if STAGE_SPEC_RX[phase].match(f))


SMOKE = {'allow_uncommitted': False}   # --dry-run --allow-uncommitted (a pre-commit smoke into scratch only)


def freeze_phase(phase, base_abs, real=True, log=_log, allow_uncommitted=None):
    """Freeze every cell of `phase` (one write-once campaign root per cell) and the phase's stage spec into
    `base_abs` (the declared phase root when `real`; a scratch directory in the dry run). Returns (ok, summary)."""
    refusals = []
    if phase == 'cross':
        return False, {'refusals': ['phase cross is declared only: its entries are added after W186 (Addendum 86 '
                                    'ruling 1); nothing to freeze']}
    lattice, lev, lfail = build_lattice()
    refusals += lfail
    if lattice is None:
        return False, {'refusals': refusals}
    cells = cell_table(phase, lattice)
    if real and os.path.abspath(base_abs) != os.path.abspath(_abs(PHASE_ROOTS[phase])):
        refusals.append(f'{base_abs} is not the declared root of phase {phase}: {PHASE_ROOTS[phase]}')
    if _stage_spec_files(base_abs, phase):
        refusals.append(f'the stage spec of {phase} already exists (write-once): {_stage_spec_files(base_abs, phase)}')
    for cdef in cells:
        root = cell_root(base_abs, cdef['cell'])
        if os.path.exists(root):
            refusals.append(f'campaign root exists (write-once, never reusable): {root}')
    pre_c = H.check_campaign_preconditions(os.path.join(base_abs, '__never_created__'),
                                          extra_clean_files=EXTRA_CLEAN_FILES)
    if allow_uncommitted is None:
        allow_uncommitted = SMOKE['allow_uncommitted']
    if allow_uncommitted and not real:   # a pre-commit smoke only; the dry run's own assertions still fail
        pre_c = [f for f in pre_c if not f.startswith('production/harness files not clean')]
    refusals += pre_c
    pins = pins_now()
    if real and (pins['uncommitted_changes'] or pins['untracked']):
        refusals.append(f"pinned files not committed and clean: {pins['uncommitted_changes']} {pins['untracked']}")
    if not real:
        log(f'[FREEZE scratch] pinned files uncommitted (informative): {pins["uncommitted_changes"]}')
    recs = record_pins_check()
    refusals += [f'committed record {n} not as pinned: {v}' for n, v in recs.items() if not v['holds']]
    solver = solver_check()
    if not solver['ok']:
        refusals.append(f'solver: {solver}')
    if H.sha256_file(H.HARNESS_PATH) != HARNESS_EXPECTED_SHA256:
        refusals.append(f'harness sha256 {H.sha256_file(H.HARNESS_PATH)} != the patched {HARNESS_EXPECTED_SHA256}')
    if _sha(HOOKS_REL) != HOOKS_EXPECTED_SHA256:
        refusals.append(f'hooks sha256 {_sha(HOOKS_REL)} != {HOOKS_EXPECTED_SHA256}')
    alias = None
    if phase == 'screen':
        g0_ok, alias = screen_alias(lattice, require_current_production=True)
        if not g0_ok:
            refusals.append(f"G0 not accepted for the x = 0 alias: {alias['g0'].get('checks')}")
    holders, n_specs, excluded = committed_eval_keys()
    pres = {c['cell']: pre_launch_assertion(c, base_abs, holders=holders) for c in cells}
    refusals += [f"pre-launch assertion fails for {c}: {[k for k, v in p['parts'].items() if not v]}"
                 for c, p in pres.items() if not p['holds']]
    if refusals:
        for r in refusals:
            log(f'[FREEZE REFUSED] {r}')
        return False, {'refusals': refusals}
    mem = memory_preflight(CONCURRENCY)
    frozen, all_ok = {}, True
    for cdef in cells:
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage': STAGE,
                 'phase': phase, 'phase_root': PHASE_ROOTS[phase], 'phase_text': PHASE_TEXT[phase],
                 'cell': cdef['cell'], 'cell_order': [c['cell'] for c in cells],
                 'role': cdef.get('role'), 'model': cdef['model'], 'objective_convention': OBJECTIVE_CONVENTION,
                 'expected_eval_key': pres[cdef['cell']]['eval_key'],
                 'pre_launch_assertion_at_freeze': pres[cdef['cell']], 'pins': pins,
                 'solver': solver, 'memory_preflight_at_freeze_non_gating': mem,
                 'stage_spec': f'frozen_s55_step7_{phase}_spec_v1_<sha8>.json (frozen AFTER this campaign spec) pins '
                               'this spec by its sha256 and holds the exact launch command',
                 'solve_claim': {'parent': 'never solves (SolveProfileGuard permitted=(), verify(0))',
                                 'child': 'the harness child; reconciled by the record solve_profile'},
                 'dry_run_scratch_freeze': not real}
        if phase in CERTIFIED_PHASES:
            extra['settling_resettle_schema'] = S7.SCHEMA
            extra['hooks_module'] = {'path': HOOKS_REL, 'sha256': _sha(HOOKS_REL)}
        if phase == 'e2':
            extra['rho_read_back_rule'] = RHO_RULE_TEXT
        if phase == 'screen':
            extra['alias'] = alias
            extra['comparator'] = ('compared with Q(x = 0 at 1x on the final code) by max{bar sum, sigma_Q = '
                                   f"{lev['sigma_Q']['sigma_Q_eur']}" + '} (plan 5.3; Addendum 85 ruling 4 (vii)); '
                                   'E.21 row 1 and row 54 reported beside as fixed-plan data, never as the comparator')
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            cell_root(base_abs, cdef['cell']), campaign_id_of(cdef['cell']), entries_of(cdef),
            configuration(phase), cap_of(cdef), CONCURRENCY, AUTHORITY, REQUIRED_CONSECUTIVE_CYCLES, extra)
        checks, _vs6 = validate_campaign_spec(cdef, spec)
        pre_frozen = pre_launch_assertion(cdef, base_abs, spec=spec, holders=holders)
        cap_ok, _cap = parent_capture_checklist(cdef, spec)
        ok = all(checks.values()) and pre_frozen['holds'] and cap_ok
        all_ok = all_ok and ok
        frozen[cdef['cell']] = {'campaign_spec': os.path.relpath(spec_path, base_abs), 'sha256': spec_sha,
                                'eval_key': spec['candidates'][0]['eval_key'], 'campaign_id': spec['campaign_id'],
                                'cap': spec['cap'], 'spec_checks_failing': [k for k, v in checks.items() if not v],
                                'pre_launch_holds': pre_frozen['holds'], 'parent_capture_checklist_ok': cap_ok,
                                'launch_command': launch_command(phase, cdef['cell'], spec_sha),
                                'preconditions_only_command': launch_command(phase, cdef['cell'], spec_sha, True)}
        log(f"[FREEZE] {phase} {cdef['cell']}: {os.path.relpath(spec_path, base_abs)} sha256 {spec_sha} eval_key "
            f"{frozen[cdef['cell']]['eval_key'][:16]} checks_failing {frozen[cdef['cell']]['spec_checks_failing']} "
            f"pre-launch {pre_frozen['holds']} capture {cap_ok}")
    content = {'schema': 'p515_s55_step7_stage_spec_v1', 'stage': STAGE, 'phase': phase,
               'phase_root': PHASE_ROOTS[phase], 'phase_text': PHASE_TEXT[phase], 'authority': AUTHORITY,
               'objective_convention': OBJECTIVE_CONVENTION, 'frozen_utc': _utc(),
               'git_head': H._git(['rev-parse', 'HEAD']), 'cell_order': [c['cell'] for c in cells],
               'cells': frozen, 'n_cells': len(cells), 'concurrency': CONCURRENCY,
               'pins': {'all': pins, 'code_sha256': code_sha256(), 'harness_sha256': H.sha256_file(H.HARNESS_PATH),
                        'hooks_sha256': _sha(HOOKS_REL), 'solver': solver, 'committed_records': recs},
               'committed_holder_scan': {'n_specs_scanned': n_specs, 'excluded_roots': list(STAGE_ROOTS),
                                         'n_excluded_specs': len(excluded),
                                         'holders': {c: pres[c]['committed_holders_outside_the_stage_roots']
                                                     for c in pres}},
               'memory_preflight_at_freeze_non_gating': mem, 'alias': alias,
               'persist_certified_models': phase in CERTIFIED_PHASES,
               'rho_read_back_rule': RHO_RULE_TEXT if phase == 'e2' else None,
               'report_command': report_command(phase) if phase in CERTIFIED_PHASES else None,
               'stops': STOPS.get(phase), 'dry_run_scratch_freeze': not real}
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(base_abs, f'frozen_s55_step7_{phase}_spec_v1_{sha[:8]}.json')
    with open(path, 'x') as handle:
        handle.write(text)
    if H.sha256_file(path) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    log(f'[FREEZE] {phase}: stage spec {os.path.relpath(path, base_abs)} sha256 {sha}; all cells ok {all_ok}')
    return all_ok, {'stage_spec': os.path.relpath(path, base_abs), 'stage_spec_sha256': sha, 'cells': frozen,
                    'all_cells_ok': all_ok}


def freeze_retry(phase, cell, base_abs, real=True, log=_log):
    """A retry of `cell` into a NEW root `campaign_s55_step7_<cell>_r2` (the original root is write-once and holds
    its results), with `retry_of`, plus a write-once retry pin file. Returns (ok, summary)."""
    refusals = []
    lattice, _lev, lfail = build_lattice()
    refusals += lfail
    cells = {c['cell']: c for c in cell_table(phase, lattice)} if lattice is not None else {}
    if cell not in cells:
        return False, {'refusals': refusals + [f'{cell} is not a cell of {phase}']}
    cdef = cells[cell]
    orig_root = cell_root(base_abs, cell)
    new_root = cell_root(base_abs, cell, retry=True)
    res_path = os.path.join(orig_root, RESULTS_FILE)
    if not os.path.isfile(res_path):
        refusals.append(f'{cell} has no results: a retry follows a completed (failed or stopped) run only')
    if os.path.exists(new_root):
        refusals.append(f'retry root exists (write-once; a third attempt needs the Planner): {new_root}')
    st = stage_spec(base_abs, phase, require_git=real)
    if st is None:
        refusals.append('the stage spec of the phase is missing or not unique / not committed')
    pins = pins_now()
    if real and (pins['uncommitted_changes'] or pins['untracked']):
        refusals.append(f"pinned files not committed and clean: {pins['uncommitted_changes']} {pins['untracked']}")
    if real and not _committed_clean(res_path):
        refusals.append('the original results are not committed')
    if refusals:
        for r in refusals:
            log(f'[RETRY FREEZE REFUSED] {r}')
        return False, {'refusals': refusals}
    orig = _load(res_path)
    holders, _n, _ex = committed_eval_keys()
    pre = pre_launch_assertion(cdef, base_abs, retry=True, holders=holders)
    if not pre['holds']:
        return False, {'refusals': [f'pre-launch assertion fails: {pre["parts"]}']}
    extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage': STAGE,
             'phase': phase, 'phase_root': PHASE_ROOTS[phase], 'cell': cell, 'pins': pins, 'solver': solver_check(),
             'objective_convention': OBJECTIVE_CONVENTION, 'expected_eval_key': pre['eval_key'],
             'pre_launch_assertion_at_freeze': pre,
             'retry_of': {'cell': cell, 'campaign_root': os.path.relpath(orig_root, base_abs),
                          'campaign_spec_sha256': orig.get('campaign_spec_sha256'),
                          'results_sha256': H.sha256_file(res_path), 'outcome': orig.get('outcome'),
                          'exit_code': orig.get('exit_code')},
             'dry_run_scratch_freeze': not real}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(new_root, campaign_id_of(cell, retry=True), entries_of(cdef),
                                                       configuration(phase), cap_of(cdef), CONCURRENCY, AUTHORITY,
                                                       REQUIRED_CONSECUTIVE_CYCLES, extra)
    checks, _ = validate_campaign_spec(cdef, spec, retry=True)
    pin = {'schema': 'p515_s55_step7_retry_pin_v1', 'phase': phase, 'cell': cell, 'retry_cell': cell + RETRY_SUFFIX,
           'campaign_spec': os.path.relpath(spec_path, base_abs), 'sha256': spec_sha, 'retry_of': extra['retry_of'],
           'stage_spec': st and {'name': st[0], 'sha256': st[1]},
           'launch_command': launch_command(phase, cell + RETRY_SUFFIX, spec_sha),
           'preconditions_only_command': launch_command(phase, cell + RETRY_SUFFIX, spec_sha, True), 'utc': _utc()}
    text = GRIO.dumps(pin, indent=1, sort_keys=True) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    pin_path = os.path.join(base_abs, f'frozen_s55_step7_{phase}_retry_{H._sanitize_id(cell)}_r2_v1_{sha[:8]}.json')
    with open(pin_path, 'x') as handle:
        handle.write(text)
    ok = all(checks.values())
    log(f'[RETRY FREEZE] {cell}{RETRY_SUFFIX}: {spec_sha} checks_failing {[k for k, v in checks.items() if not v]}')
    return ok, {'campaign_spec': os.path.relpath(spec_path, base_abs), 'sha256': spec_sha,
                'retry_pin': os.path.relpath(pin_path, base_abs), 'retry_pin_sha256': sha,
                'checks_failing': [k for k, v in checks.items() if not v], 'retry_of': extra['retry_of']}


def stage_spec(base_abs, phase, require_git=True):
    """(name, sha256, content) of the phase's unique stage spec whose name carries its own sha256 prefix; None
    otherwise (or when `require_git` and it is not committed and clean)."""
    files = _stage_spec_files(base_abs, phase)
    if len(files) != 1:
        return None
    path = os.path.join(base_abs, files[0])
    sha = H.sha256_file(path)
    if sha[:8] != STAGE_SPEC_RX[phase].match(files[0]).group(1):
        return None
    if require_git and not _committed_clean(path):
        return None
    with open(path) as handle:
        return files[0], sha, json.load(handle)


# ======================================================================================================================
#  the per-cell report, the outcome (stops), the rho read-back
# ======================================================================================================================
STOPS = {
    'e3': ['the settling status is not certified within the cap (the uncertified form is reported)',
           "nrf_terminal.status != 'written' or nrf_capture_error",
           'n_active per copy != 864 (TSO 864; DSO 288 each = 864 total)'],
    'e2': ['the rho read-back does not reproduce bitwise (harness-fault stop, Addendum 86 ruling 2)'],
    'screen': [], 'cross': []}


def _decision(eval_dir):
    path = os.path.join(eval_dir, S7.DECISION_FILE)
    return _load(path) if os.path.isfile(path) else None


def cell_report(cdef, eval_dir, rec):
    """The ext-v6 per-cell report for a Step-7 certified cell (the settling v6 decision), with the uncertified form
    (k_cap, Q_at_cap, t_sum_at_cap, band_window, gap refusals, drift rate), the harness bar and the stopping ratio."""
    rows = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    dec = _decision(eval_dir) or {}
    summ = rec.get('settling_resettle_summary') or {}
    last = max(rows) if rows else None
    certified = dec.get('status') == 'certified'
    end = dec.get('k_star') if certified else dec.get('k_cap')
    end = end if end is not None else last
    q = {k: r.get('gross_operational_cost') for k, r in rows.items()}
    steps = [q[k] - q[k - 1] for k in sorted(q) if (k - 1) in q and q[k] is not None and q[k - 1] is not None]
    q_end = dec.get('Q_k_star') if certified else dec.get('Q_at_cap')
    t_end = dec.get('t_sum_k_star') if certified else dec.get('t_sum_at_cap')
    rt = rec.get('rule_ten') or {}
    rep = {'cell': cdef['cell'], 'phase': cdef['phase'], 'eval_key': rec.get('eval_key'),
           'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
           'nrf_option': cdef['nrf_option'], 'rho0_scale': cdef['rho0_scale'],
           'flex_price_multiplier': cdef['flex_price_multiplier'],
           'criterion_version': dec.get('version'), 'k0_run': summ.get('first_residual_pass_run'),
           'rule_cap': dec.get('cap'), 'cycles_run': rec.get('cycles_run'), 'status': dec.get('status'),
           'record_status': rec.get('status'), 'k_star': dec.get('k_star'), 'end_cycle': end,
           'branch': dec.get('branch'), 'band': dec.get('band'), 'band_width': dec.get('band_width'),
           'range_over_tau': dec.get('range_over_tau'), 'W': dec.get('W'), 'window': dec.get('window'),
           'Q_end': q_end, 't_sum_end': t_end,
           'Q_cc_end': (q_end + t_end) if (q_end is not None and t_end is not None) else None,
           'Q_net_end': (rows.get(end) or {}).get('recourse'),
           'terminal_salvage_value_end': (rows.get(end) or {}).get('terminal_salvage_value'),
           'gap_refusals': dec.get('gap_refusals'), 'non_clean_cycles': dec.get('non_clean_cycles'),
           'vetoes': dec.get('vetoes'), 'lapse_events': dec.get('lapse_events'),
           'harness_bar_eur': (rec.get('bar') or {}).get('value'),
           'terminal_step_abs': abs(steps[-1]) if steps else None,
           'terminal_step_over_EPS0': (abs(steps[-1]) / SC6.EPS0) if steps else None,
           'stopping_ratio_terminal_step_over_threshold': rt.get('terminal_step_over_threshold'),
           'stopping_ratio_definition': ('rule ten: |Q[k] - Q[k-1]| / objective_tolerance at the last cycle (the '
                                         "record's rule_ten.terminal_step_over_threshold)"),
           'objective_convention': OBJECTIVE_CONVENTION}
    if not certified:
        rep.update({k: dec.get(k) for k in ('k_cap', 'reasons', 'band_window', 'drift_rate_mean_dQ_last_25',
                                            'dQ_cc_rate_mean_last_25', 'Q_at_cap', 't_sum_at_cap', 'Q_cc_at_cap',
                                            'gap_clause_refused_at_cap')})
        rep['gap_refused'] = bool(dec.get('gap_refusals'))
        rep['label'] = 'gap-refused (uncertified)' if rep['gap_refused'] else 'uncertified at the cap'
    rep['view'] = view_from_report(rep)
    return rep


def view_from_report(rep):
    """The scorer view (W132's `view_from_report` shape; ungated cells: slack undefined)."""
    certified = rep.get('status') == 'certified'
    q, t = rep.get('Q_end'), rep.get('t_sum_end')
    return {'status': rep.get('status'), 'Q': q, 't': t, 'Q_cc': (q + t) if (q is not None and t is not None) else None,
            'Q_net': rep.get('Q_net_end'), 'salvage': rep.get('terminal_salvage_value_end'),
            'band': rep.get('band_width'), 'gated': False,
            'gap': None if certified or t is None else abs(t), 'slack': None,
            'gap_refused': rep.get('gap_refused'), 'label': rep.get('label'), 'bar': rep.get('harness_bar_eur')}


def resolve_pair(d_q, d_cc, views):
    """Rule v6 for a difference between two ADMM cells: determinate iff both certified and |d| >= max{3 x the larger
    band, 2 tau} (SC6.determinate_certified); otherwise the uncertified form (W132 `resolve`, copied: 3 x max(|gap|,
    |slack|) over the uncertified cells; an ungated uncertified cell has no slack -> indeterminate)."""
    unc = [v for v in views if v.get('status') != 'certified']
    m_cc = abs(d_cc) if (d_cc is not None and (d_q > 0) == (d_cc > 0)) else (-abs(d_cc) if d_cc is not None else None)
    if not unc:
        det, thr, binds = SC6.determinate_certified(d_q, *[v['band'] for v in views])
        return {'rule': 'rule v6 (certified pair)', 'bands': [v['band'] for v in views], 'threshold': thr,
                'binding_term': binds, 'margin_Q': abs(d_q), 'margin_Qcc': m_cc,
                'verdict': 'determinate' if det else 'within resolution', 'margin_over_threshold': abs(d_q) / thr,
                'sign': (1 if d_q > 0 else -1) if det else 0}
    comps = []
    for v in unc:
        if v.get('slack') is None or v.get('gap') is None:
            return {'rule': 'uncertified form', 'bar': None, 'verdict': 'indeterminate (slack undefined)', 'sign': 0,
                    'note': 'an ungated uncertified cell has no s = Q(cap) - Q_N_old (W132 resolve)',
                    'uncertified_cells_harness_bar': [v.get('bar') for v in unc],
                    'uncertified_cells_gap': [v.get('gap') for v in unc]}
        comps += [v['gap'], v['slack']]
    bar = 3.0 * max(comps)
    det = abs(d_q) > bar and m_cc is not None and m_cc > bar
    return {'rule': 'uncertified form', 'bar': bar, 'bar_components': comps, 'margin_Q': abs(d_q), 'margin_Qcc': m_cc,
            'verdict': 'determinate' if det else 'within the uncertified bar', 'sign': (1 if d_q > 0 else -1) if det
            else 0}


def nrf_summary(rec):
    """The terminal NRF capture: status, rows and active rows per copy, both binding measures, per-node minima."""
    t = rec.get('nrf_terminal') or {}
    per_net = t.get('row_counts_per_network') or {}
    dso = {k: v for k, v in per_net.items() if k.startswith('DSO')}
    tso = per_net.get('TSO') or {}
    per_block = t.get('row_counts_per_block') or {}
    blocks = {net: {b: {'min_body_mw': v.get('min_body_mw'), 'n_body_le_tol': v.get('n_body_le_tol'),
                        'n_active': v.get('n_active'), 'n_indices': v.get('n_indices')} for b, v in per.items()}
              for net, per in per_block.items()}
    checks = {'status_written': t.get('status') == 'written',
              'tso_n_active_864': tso.get('n_active') == 864 and tso.get('n_indices') == 864,
              'dso_n_active_288_each': len(dso) == 3 and all(v.get('n_active') == 288 and v.get('n_indices') == 288
                                                             for v in dso.values()),
              'dso_n_active_total_864': sum(v.get('n_active') or 0 for v in dso.values()) == 864}
    return {'status': t.get('status'), 'error': t.get('error'), 'row': t.get('row'),
            'binding_tolerance_mw': t.get('binding_tolerance_mw'),
            'row_counts_per_network': per_net, 'checks': checks, 'n_active_per_copy_ok': all(
                checks[k] for k in ('tso_n_active_864', 'dso_n_active_288_each', 'dso_n_active_total_864')),
            'binding_hours_expected_value_rule': {'definition': t.get('binding_definition'),
                                                  'n_binding_hours': t.get('n_binding_hours'),
                                                  'hours': t.get('binding_hours')},
            'binding_by_block_row_body': {
                'definition': 'per block: the minimum NRF row body (MW) and the number of row indices with body <= '
                              'the binding tolerance (W183c row_counts_per_block)',
                'n_rows_body_le_tol_total': {net: sum((v.get('n_body_le_tol') or 0) for v in per.values())
                                             for net, per in per_block.items()},
                'per_block': blocks},
            'measures_coincide_note': ('at one scenario the expected-value rule and the per-block row-body measure '
                                       'read the same interface quantity, so they coincide (Addendum 85 ruling 4 (x)); '
                                       'both are reported'),
            'per_node_minima': t.get('per_node')}


def _rho_action_factor(action):
    if action == 'increased':
        return RHO_INCREASE_FACTOR
    if action == 'decreased':
        return 1.0 / RHO_DECREASE_FACTOR
    if isinstance(action, str) and RHO_FACTOR_ONE_RX.match(action):
        return None
    raise ValueError(f'unknown rho action {action!r}: the replication refuses (harness fault)')


def model_counts(solve_profile):
    """n_v = n_pf = TSO + DSO blocks; n_ess = TSO + DSO + ESSO models (from a record's solve_profile)."""
    n_years, n_days = solve_profile.get('n_years'), solve_profile.get('n_days')
    n_dso, n_esso = solve_profile.get('n_dso'), solve_profile.get('n_esso_nodes')
    if n_days is None and solve_profile.get('solves_per_cycle') is not None and None not in (n_years, n_dso, n_esso):
        n_days = (solve_profile['solves_per_cycle'] - n_esso) // ((1 + n_dso) * n_years)
    tso = n_years * n_days
    dso = n_dso * n_years * n_days
    return {'v': tso + dso, 'pf': tso + dso, 'ess': tso + dso + n_esso,
            'source': {'n_years': n_years, 'n_days': n_days, 'n_dso': n_dso, 'n_esso_nodes': n_esso,
                       'solves_per_cycle': solve_profile.get('solves_per_cycle')}}


def rho_read_back(rows, s, n, s39=None):
    """Addendum 86 ruling 2. -> {'holds', 'n_cycles', 'mismatches', 'cycle_1', 'per_cycle'}."""
    import p515_g_g1_g4_admm_gates as G
    base = s39 or {'v': G.S39_RHO_V, 'pf': G.S39_RHO_PF, 'ess': G.S39_RHO_ESS}
    x = {g: base[g] * s for g in RHO_CHANNELS}
    lo, hi = RHO_CLAMP
    per, mismatches, errors = [], [], []
    for r in sorted(rows, key=lambda r: r['cycle']):
        line = {'cycle': r['cycle']}
        for g in RHO_CHANNELS:
            action = r.get(f'rho_{g}_action')
            try:
                f = _rho_action_factor(action)
            except ValueError as error:
                errors.append({'cycle': r['cycle'], 'channel': g, 'error': str(error)})
                continue
            if f is not None:
                x[g] = min(max(x[g] * f, lo), hi)
            replicated = sum([x[g]] * n[g]) / n[g]
            recorded = r.get(f'rho_{g}_after')
            eq = isinstance(recorded, float) and replicated == recorded
            line[g] = {'action': action, 'x': x[g], 'replicated': replicated, 'recorded': recorded, 'bitwise': eq}
            if not eq:
                mismatches.append({'cycle': r['cycle'], 'channel': g, 'action': action, 'replicated': replicated,
                                   'recorded': recorded, 'replicated_hex': float(replicated).hex(),
                                   'recorded_hex': float(recorded).hex() if isinstance(recorded, float) else None})
        per.append(line)
    return {'rule': RHO_RULE_TEXT, 's': s, 's39': base, 'n': {g: n[g] for g in RHO_CHANNELS},
            'n_cycles': len(per), 'n_checks': 3 * len(per), 'mismatches': mismatches, 'errors': errors,
            'holds': bool(per) and not mismatches and not errors, 'cycle_1': per[0] if per else None,
            'per_cycle': per}


def rho_validation_w183f():
    """The replication validated zero-solve on the committed W183f arm (s = 1): cycle 1 must reproduce pf
    0.1980000000000002 (n 48), ess 0.010000000000000004 (n 51), v 0.011549999999999998 (n 48) bitwise."""
    rows = _rec('w183f_arm_per_cycle')
    g = _rec('w183f_arm_g')
    n = model_counts(g['solve_profile'])
    counts = (g.get('rule_eleven_checklist') or {}).get('w10_expected_block_counts') or {}
    rb = rho_read_back(rows, 1.0, n)
    c1 = rb['cycle_1'] or {}
    checks = {f'cycle1_{ch}_reproduces_{val!r}_n{nn}_bitwise': (
        (c1.get(ch) or {}).get('replicated') == val and (c1.get(ch) or {}).get('recorded') == val and n[ch] == nn)
        for ch, (val, nn) in W183F_EXPECTED_CYCLE1.items()}
    checks['every_cycle_reproduces_bitwise'] = rb['holds']
    checks['model_counts_equal_the_rule_eleven_block_counts'] = (
        n['v'] == counts.get('network_blocks') and n['ess'] == counts.get('network_blocks', 0)
        + counts.get('esso_models', -99))
    return {'record': REC['w183f_arm_per_cycle'], 'solve_profile_record': REC['w183f_arm_g'], 'model_counts': n,
            'read_back': rb, 'checks': checks, 'holds': all(checks.values())}


def cell_outcome(cdef, rec, eval_dir):
    """The post-run evaluation of one cell (used by --run; exercised on planted records by the dry run):
    -> (exit_code, outcome dict). 0 ok; 3 a declared stop; 1 a missing / barrier record."""
    out = {'cell': cdef['cell'], 'phase': cdef['phase'], 'stops': [], 'faults': []}
    if not rec or rec.get('barrier') or rec.get('status') == 'error':
        out['faults'].append(f"no evaluation record or a barrier: status {(rec or {}).get('status')} cause "
                             f"{(rec or {}).get('barrier_cause')}")
        return 1, out
    out['record_status'] = rec.get('status')
    out['eval_key'] = rec.get('eval_key')
    if rec.get('eval_key') != eval_key_of(cdef):
        out['faults'].append(f"record eval key {rec.get('eval_key')} != the cell's {eval_key_of(cdef)}")
    if cdef['phase'] in CERTIFIED_PHASES:
        rep = cell_report(cdef, eval_dir, rec)
        out['cell_report'] = rep
        if cdef['phase'] == 'e3':
            if rep['status'] != 'certified':
                out['stops'].append(f"E3 settling status {rep['status']!r}: not certified within the cap "
                                    f"(k_cap {rep.get('k_cap')}, Q_at_cap {rep.get('Q_at_cap')}, t_sum_at_cap "
                                    f"{rep.get('t_sum_at_cap')}, harness bar {rep.get('harness_bar_eur')})")
            nrf = nrf_summary(rec)
            out['nrf_terminal'] = nrf
            exit_child = (rec.get('parent_view') or {}).get('exit_code')
            nrf_capture_error = (nrf['status'] != 'written') or exit_child == 2
            out['nrf_capture_error'] = nrf_capture_error
            if nrf_capture_error:
                out['stops'].append(f"nrf_terminal.status {nrf['status']!r} / child exit {exit_child}: "
                                    'nrf_capture_error')
            if not nrf['n_active_per_copy_ok']:
                out['stops'].append(f"n_active per copy != 864: {nrf['row_counts_per_network']}")
        if cdef['phase'] == 'e2':
            rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
            n = model_counts(rec.get('solve_profile') or {})
            eb = cdef['declaration']['expected_blocks']
            n_decl = {'v': eb['tso'] + eb['dso'], 'pf': eb['tso'] + eb['dso'], 'ess': eb['tso'] + eb['dso'] + eb['esso']}
            rb = rho_read_back(rows, cdef['rho0_scale'], n)
            applied = rec.get('rho0_scale_applied_in_child') or {}
            out['rho_read_back'] = rb
            out['rho_model_counts_equal_the_declared_blocks'] = all(n[g] == n_decl[g] for g in RHO_CHANNELS)
            out['rho0_scale_applied_in_child_checks'] = applied.get('checks')
            if not rb['holds'] or not out['rho_model_counts_equal_the_declared_blocks'] or not (
                    applied.get('checks') and all(applied['checks'].values())):
                out['stops'].append(f"rho read-back fails (harness fault): mismatches {rb['mismatches'][:3]} errors "
                                    f"{rb['errors'][:3]} counts {n} applied {applied.get('checks')}")
    else:
        out['screen'] = {'status': rec.get('status'), 'certified_cost': rec.get('certified_cost'),
                         'cycles_run': rec.get('cycles_run'), 'bar_eur': (rec.get('bar') or {}).get('value'),
                         'rule_ten': rec.get('rule_ten'), 'objective_convention': OBJECTIVE_CONVENTION}
    exit_child = (rec.get('parent_view') or {}).get('exit_code')
    if exit_child not in (0, None) and not (cdef['phase'] == 'e3' and out.get('nrf_capture_error')):
        out['faults'].append(f'child exit code {exit_child} (a capture / summary error recorded by the harness)')
    if out['faults']:
        return 1, out
    return (3 if out['stops'] else 0), out


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _cell_done(base_abs, cell):
    return any(os.path.isfile(os.path.join(cell_root(base_abs, cell, r), RESULTS_FILE)) for r in (False, True))


def run_preconditions(phase, cell_arg, spec_sha256, base_abs, real=True):
    """Every --run precondition. -> (failures, context)."""
    failures = []
    retry = cell_arg.endswith(RETRY_SUFFIX)
    cell = cell_arg[:-len(RETRY_SUFFIX)] if retry else cell_arg
    if phase == 'cross':
        return ['phase cross is declared only (entries after W186): --run refuses'], {}
    if real and os.path.abspath(base_abs) != os.path.abspath(_abs(PHASE_ROOTS[phase])):
        failures.append(f'not the declared root of {phase}: {base_abs}')
    lattice, _lev, lfail = build_lattice()
    failures += lfail
    cells = cell_table(phase, lattice) if lattice is not None else []
    names = [c['cell'] for c in cells]
    if cell not in names:
        return failures + [f'{cell} is not a cell of {phase}: {names}'], {}
    cdef = cells[names.index(cell)]
    st = stage_spec(base_abs, phase, require_git=True)
    if st is None:
        failures.append('the stage spec is missing, not unique, not self-hashing or not committed and clean')
    else:
        pinned = (st[2].get('cells') or {}).get(cell) or {}
        if retry:
            pins = [f for f in os.listdir(base_abs) if f.startswith(f'frozen_s55_step7_{phase}_retry_'
                                                                    f'{H._sanitize_id(cell)}_r2_v1_')]
            pin = _load(os.path.join(base_abs, pins[0])) if len(pins) == 1 else {}
            if len(pins) != 1 or pin.get('sha256') != spec_sha256 or not _committed_clean(os.path.join(base_abs,
                                                                                                         pins[0])):
                failures.append(f'the retry pin of {cell} is missing / not unique / not committed or pins another '
                                f'spec: {pins}')
        elif pinned.get('sha256') != spec_sha256:
            failures.append(f"the stage spec pins {pinned.get('sha256')} for {cell}, not {spec_sha256}")
        if st[2]['pins']['code_sha256'] != code_sha256():
            failures.append('code changed since the stage spec froze: '
                            f"{[f for f, s in st[2]['pins']['code_sha256'].items() if _sha_or_none(f) != s]}")
    idx = names.index(cell)
    for prev in names[:idx]:
        if not _cell_done(base_abs, prev):
            failures.append(f'launch order: {prev} (#{names.index(prev) + 1}) has no results yet')
    if _cell_done(base_abs, cell) and not retry:
        failures.append(f'{cell} already has results (a retry goes to {cell}{RETRY_SUFFIX})')
    root = cell_root(base_abs, cell, retry)
    if not os.path.isdir(root):
        return failures + [f'campaign root missing: {root}'], {}
    try:
        spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    except Exception as error:  # noqa: BLE001
        return failures + [f'frozen spec: {error}'], {}
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'the campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(spec_path):
        failures.append('the campaign spec is not committed and clean')
    failures += [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                 if f != f'campaign root already exists (write-once): {root}']
    checks, _vs6 = validate_campaign_spec(cdef, spec, retry=retry)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    extra = spec.get('extra') or {}
    for what, pinned_sha, now in (
            ('harness', (spec.get('harness') or {}).get('sha256'), H.sha256_file(H.HARNESS_PATH)),
            ('harness = the patched harness', HARNESS_EXPECTED_SHA256, H.sha256_file(H.HARNESS_PATH)),
            ('hooks module', HOOKS_EXPECTED_SHA256, _sha(HOOKS_REL)),
            ('this launcher', extra.get('campaign_script_sha256'), _sha(SCRIPT_NAME)),
            ('case file', spec['configuration'].get('case_file_sha256'), H.sha256_file(H.CASE_FILE)),
            ('ESS parameters file', (spec['configuration'].get('ess_params_file') or {}).get('sha256'),
             _sha(H.ESS_PARAMS_FILE_REL)),
            ('cost file', PB.COST_FILE['sha256'], _sha_or_none(PB.COST_FILE['path']))):
        if pinned_sha != now:
            failures.append(f'{what} sha256 differs: frozen {pinned_sha} now {now}')
    failures += [f'PIN DIFFERS: {d}' for d in pins_compare(extra.get('pins') or {}, pins_now())] \
        if extra.get('pins') else ['the spec records no pins']
    pre = pre_launch_assertion(cdef, base_abs, retry=retry, spec=spec)
    if not pre['holds']:
        failures.append(f"pre-launch assertion fails: {[k for k, v in pre['parts'].items() if not v]}")
    cap_ok, cap = parent_capture_checklist(cdef, spec)
    if not cap_ok:
        failures.append(f'parent capture checklist fails: { {k: v for k, v in cap.items() if "error" in v} }')
    solver = solver_check()
    if not solver['ok'] or solver['sha256'] != (extra.get('solver') or {}).get('sha256'):
        failures.append(f'solver path / binary differs from the frozen one: {solver}')
    alias = None
    if phase == 'screen' and cell == SCREEN_ALIAS['reuse_of']:
        ok, alias = screen_alias(lattice, require_current_production=True)
        if not ok or alias.get('g0_artifact_sha256') != ((extra.get('alias') or {}).get('g0_artifact_sha256')):
            failures.append(f"G0 alias not accepted now: {alias['g0'].get('checks')}")
    mem = memory_preflight(CONCURRENCY)
    if not mem['pass']:
        failures.append(f"memory preflight REFUSED (gating at --run): {mem.get('available_gib')} < "
                        f"{mem.get('required_gib')}")
    alive = _own_process_alive()
    if alive:
        failures.append(f'another copy of this launcher is alive: {alive}')
    return failures, {'cdef': cdef, 'spec_path': spec_path, 'spec': spec, 'root': root, 'retry': retry,
                      'pre': pre, 'capture': cap, 'solver': solver, 'memory': mem, 'alias': alias,
                      'stage_spec': st and {'name': st[0], 'sha256': st[1]}, 'index': idx, 'n_cells': len(names)}


def run(args):
    started = time.time()
    base_abs = _abs(PHASE_ROOTS[args.phase])
    failures, ctx = run_preconditions(args.phase, args.cell, args.spec_sha256, base_abs, real=True)
    if failures:
        for f in failures:
            _log(f'[RUN PRECONDITION FAILED] {f}')
        return 1
    cdef, spec, spec_path, root = ctx['cdef'], ctx['spec'], ctx['spec_path'], ctx['root']
    entry = spec['candidates'][0]
    if args.preconditions_only:
        _log(f"[RUN {args.cell}] every precondition holds: spec {_rel(spec_path)} sha256 {args.spec_sha256}; stage "
             f"spec {ctx['stage_spec']}; launch index {ctx['index'] + 1} of {ctx['n_cells']}; eval key "
             f"{entry['eval_key']}; cap {spec['cap']}; harness {H.sha256_file(H.HARNESS_PATH)[:8]}; hooks "
             f"{_sha(HOOKS_REL)[:8]}; solver {ctx['solver']['resolved'].get('NLP_SOLVER_PATH')} "
             f"{ctx['solver']['sha256']}; memory {ctx['memory'].get('available_gib')} GiB")
        _log(f'[RUN {args.cell}] STOPPED before the campaign lock and the child (zero solves)')
        return 0
    head = H._git(['rev-parse', 'HEAD'])
    lock = H.acquire_campaign_lock(spec['campaign_id'], args.spec_sha256)
    _log(f"[RUN {args.cell}] lock {lock}; HEAD {head}; eval key {entry['eval_key']}; cap {spec['cap']}")
    try:
        hctx = H.CampaignContext(root, spec_path, args.spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([entry['label']], hctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('campaign lock released')
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    rec = records[0] if records else None
    try:
        code, outcome = cell_outcome(cdef, rec, eval_dir)
    except Exception as error:  # noqa: BLE001 -- recorded; a harness fault
        code, outcome = 1, {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    g = guards_report()
    results = {'stage': STAGE, 'phase': args.phase, 'cell': cdef['cell'], 'retry': ctx['retry'],
               'launch_index_1_based': ctx['index'] + 1, 'utc': _utc(), 'git_head_at_run': head,
               'campaign_spec_path': _rel(spec_path), 'campaign_spec_sha256': args.spec_sha256,
               'stage_spec': ctx['stage_spec'], 'objective_convention': OBJECTIVE_CONVENTION,
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'], 'outcome': outcome, 'exit_code': code,
               'STOP_FOR_REVIEW': code == 3, 'pre_launch_assertion': ctx['pre'],
               'parent_capture_checklist': ctx['capture'], 'memory_preflight_at_run': ctx['memory'],
               'solver': ctx['solver'], 'alias': ctx['alias'], 'batch_info': batch,
               'parent_view': (rec or {}).get('parent_view'), 'guards': g, 'wall_clock_s': time.time() - started}
    _write_once_json(os.path.join(root, RESULTS_FILE), results)
    manifest = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[_rel(fpath)] = H.sha256_file(fpath)
    _write_once_json(os.path.join(root, MANIFEST_FILE), manifest)
    for s in outcome.get('stops') or []:
        _log(f'[RUN {args.cell}] STOP FOR REVIEW: {s}')
    rep = outcome.get('cell_report') or {}
    _log(f"[RUN {args.cell}] status {rep.get('status', (outcome.get('screen') or {}).get('status'))} k* "
         f"{rep.get('k_star')} band {rep.get('band_width')} Q_end {rep.get('Q_end')} bar {rep.get('harness_bar_eur')} "
         f"stopping ratio {rep.get('stopping_ratio_terminal_step_over_threshold')}; exit {code}; guards {g['all_zero']}")
    if not g['all_zero']:
        return 1
    return code


# ======================================================================================================================
#  the report modes (zero solves; committed records by eval key)
# ======================================================================================================================
def _results_of(base_abs, cell, require_git):
    """(results, eval_dir, record) of the completed run of `cell` (the r2 retry when one exists)."""
    for retry in (True, False):
        root = cell_root(base_abs, cell, retry)
        path = os.path.join(root, RESULTS_FILE)
        if os.path.isfile(path):
            if require_git and not (_committed_clean(path) and _committed_clean(os.path.join(root, MANIFEST_FILE))):
                raise RuntimeError(f'{path} or its manifest is not committed and clean')
            res = _load(path)
            spec = next(f for f in os.listdir(root) if f.startswith('campaign_spec_'))
            entry = _load(os.path.join(root, spec))['candidates'][0]
            eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
            rec = _load(os.path.join(eval_dir, 'evaluation_record.json'))
            return res, eval_dir, rec, {'results': _rel(path) if path.startswith(REPO) else path,
                                        'results_sha256': H.sha256_file(path), 'retry': retry}
    return None, None, None, None


def _step6_cell(eval_key):
    cells = [c for c in _rec('step6_tables')['tables']['cells'].values() if c.get('eval_key') == eval_key]
    if len(cells) != 1:
        raise RuntimeError(f'the frozen step-6 tables hold {len(cells)} cells with eval key {eval_key}')
    return cells[0]


def _agent_terms(levels):
    """By agent: the block-gross sum (generation + flexibility + load curtailment + RES curtailment penalty + ESS
    usage + detector penalties, weighted) and its conventional / flexibility parts; the RES curtailment measure."""
    parts = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
             'ess_usage_cost', 'detector_penalty_total')
    out = {k: {'block_gross_weighted': 0.0, 'generation_cost_weighted': 0.0, 'flexibility_cost_weighted': 0.0,
               'res_curtailment_definitional_unweighted_sum': 0.0, 'n_blocks': 0} for k in ('TSO', 'DSO')}
    per_block_curt = {}
    for name, b in (levels.get('blocks') or {}).items():
        w, u = b['weighted'], b['unweighted']
        a = out[b['kind']]
        a['block_gross_weighted'] += sum(w[p] for p in parts)
        a['generation_cost_weighted'] += w['generation_cost']
        a['flexibility_cost_weighted'] += w['flexibility_cost_internal']
        a['res_curtailment_definitional_unweighted_sum'] += u['res_curtailment_definitional_at_weight_1']
        a['n_blocks'] += 1
        per_block_curt[(b['kind'], str(b['year']), str(b['day']))] = per_block_curt.get(
            (b['kind'], str(b['year']), str(b['day'])), 0.0) + u['res_curtailment_definitional_at_weight_1']
    gross = (levels.get('recourse_components') or {}).get('gross_operational_cost')
    total = out['TSO']['block_gross_weighted'] + out['DSO']['block_gross_weighted']
    return {'by_agent': out, 'gross_operational_cost': gross, 'sum_of_block_gross': total,
            'identity_residual_eur': (total - gross) if gross is not None else None,
            '_per_block_curtailment': per_block_curt}


def _day_weights():
    pb = _rec('price_taker_best_arm')['phase_C_sequential_pass']['curtailment_signed_parts']['per_block']
    out = {}
    for name, v in pb.items():
        parts = name.split('|')
        key = (parts[-2], parts[-1])
        if key in out and out[key] != v['day_weight']:
            raise RuntimeError(f'day weight of {key} differs across blocks')
        out[key] = v['day_weight']
    return out


def _curtailment_gwh(terms, weights):
    out = {}
    for agent in ('TSO', 'DSO'):
        s = sum(v * weights[(y, d)] for (k, y, d), v in terms['_per_block_curtailment'].items() if k == agent)
        out[agent] = s / 1000.0
    return out


def report_e3(base_abs, require_git=True, planted=False):
    cdef = cell_table('e3')[0]
    res, eval_dir, rec, src = _results_of(base_abs, cdef['cell'], require_git)
    if res is None:
        raise RuntimeError(f"{cdef['cell']} has no results")
    rep = cell_report(cdef, eval_dir, rec)
    view = rep['view']
    # C_OPEN = E.21 row 1, by eval key
    c_open = _step6_cell(C_OPEN_EVAL_KEY)
    rv3 = _rec('benchmark_report_v3')
    rec_open = _rec('c_open_evaluation_record')
    xchecks = {'c_open_eval_key_is_row_1': c_open['eval_key'] == C_OPEN_EVAL_KEY,
               'c_open_certified': c_open.get('status') == 'certified',
               'c_open_Q_equals_benchmark_coordinated_q_bitwise': c_open['Q'] == rv3['coordinated']['q'],
               'c_open_Q_equals_its_evaluation_record_certified_cost': c_open['Q'] == rec_open.get('certified_cost'),
               'c_open_record_eval_key': rec_open.get('eval_key') == C_OPEN_EVAL_KEY,
               'benchmark_coordinated_cell_is_row_1': rv3['coordinated_cell']['eval_key'] == C_OPEN_EVAL_KEY,
               'benchmark_instance_is_x0': rv3['instance']['candidate_key'] == S7.X0_CANDIDATE_KEY
               == cdef['candidate_key'],
               'e3_record_eval_key_is_the_cell_key': rec.get('eval_key') == eval_key_of(cdef)}
    v_open = {'status': c_open['status'], 'Q': c_open['Q'], 'Q_cc': c_open.get('Q_cc'), 'band': c_open['band_width'],
              'gap': c_open.get('gap'), 'slack': c_open.get('slack'), 'gated': False}
    pt = rv3['per_arm_nrf']['price_taker']
    q_p = pt['q_best']
    s_static = pt['multimodality_band_eur']
    spread_cstar = rv3['claim']['bands_eur']['coordinated_reproducibility_0.011pct']
    q_c = view['Q']
    # (1) Q(P_NRF) - Q(C_NRF) relative to Q(P_NRF): side-by-side measures (plan section 3)
    d1 = q_p - q_c
    admm_measure = max(x for x in (view.get('band') if view['status'] == 'certified' else view.get('bar'),
                                   spread_cstar) if x is not None)
    diff1 = {'definition': 'Q(P_NRF) - Q(C_NRF), relative to Q(P_NRF) (coordination under NRF)',
             'Q_P_NRF': q_p, 'Q_P_NRF_source': f"{REC['benchmark_report_v3']['path']} per_arm_nrf.price_taker.q_best "
                                               f"(best start {pt['best_start']})",
             'Q_C_NRF': q_c, 'difference_eur': d1, 'relative_to_Q_P_NRF': d1 / q_p,
             'percent_one_decimal': round(100.0 * d1 / q_p, 1),
             'static_three_start_spread_s': s_static,
             'admm_cell_measure': admm_measure,
             'admm_cell_measure_definition': ('max(the C_NRF certification band (or the harness bar if uncertified), '
                                              'the between-configuration spread of Q at C* = '
                                              'claim.bands_eur.coordinated_reproducibility_0.011pct)'),
             'between_configuration_spread_at_cstar': spread_cstar,
             'resolved_under_the_declared_procedure': abs(d1) > 3.0 * s_static and abs(d1) > 3.0 * admm_measure,
             'rule': ('plan section 3 (clarification 5): resolved only if it exceeds three times each of the two '
                      'measures; the two are reported side by side, never combined; "determinate" is reserved for '
                      'ADMM pairs under rule v6')}
    diff1['H3b_90p9_pm_0p1_MEUR_and_12p2_percent'] = {
        'holds': abs(d1 - 90.9e6) <= 0.1e6 and diff1['percent_one_decimal'] == 12.2,
        'statement': 'H3b: Q(P_NRF) - Q(C_NRF) = 90.9 M EUR +- 0.1 M EUR, 12.2 % at one decimal'}
    # (2) Q(C_NRF) - Q(C_OPEN) relative to Q(C_NRF): rule v6
    d2 = q_c - c_open['Q']
    d2_cc = (view['Q_cc'] - c_open['Q_cc']) if (view.get('Q_cc') is not None and c_open.get('Q_cc') is not None) \
        else None
    diff2 = {'definition': 'Q(C_NRF) - Q(C_OPEN), relative to Q(C_NRF) (permitting reverse flow under coordination)',
             'Q_C_NRF': q_c, 'Q_C_OPEN': c_open['Q'],
             'Q_C_OPEN_source': f"{REC['step6_tables']['path']} tables.cells[eval_key {C_OPEN_EVAL_KEY[:16]}] "
                                f"(E.21 row 1, {c_open.get('name')})",
             'difference_eur': d2, 'relative_to_Q_C_NRF': d2 / q_c, 'difference_Qcc_report_only': d2_cc,
             'resolution': resolve_pair(d2, d2_cc, (view, v_open))}
    two_tau = SC6.DETERMINACY_TAU_MULTIPLE * SC6.TAU
    diff2['H3a_positive_and_within_resolution_lt_2tau'] = {
        'holds': d2 > 0 and abs(d2) < two_tau, 'two_tau': two_tau,
        'statement': 'H3a: Q(C_NRF) - Q(C_OPEN) is positive and within resolution (< 2 tau = 9.1 k EUR)'}
    # decomposition (TN conventional / DN flexibility; W124's method for the static arm)
    lev_nrf = _load(os.path.join(eval_dir, 'component_levels_terminal.json'))
    t_nrf, t_open = _agent_terms(lev_nrf), _agent_terms(_rec('c_open_component_levels'))
    arm = _rec('price_taker_best_arm')['phase_C_sequential_pass']['evaluation']
    tso_p = sum(v for k, v in arm['block_components'].items() if k.startswith('TSO|'))
    weights = _day_weights()
    curt = {'C_NRF': _curtailment_gwh(t_nrf, weights), 'C_OPEN': _curtailment_gwh(t_open, weights)}
    ca = rv3['curtailment_table']['arms_and_variants_phase_A'][
        'nrf_arm_price_taker_' + pt['best_start'] + '_r2']['by_agent']
    curt['P_NRF'] = {a: ca[a]['mwh_day_weighted'] / 1000.0 for a in ('TSO', 'DSO')}
    decomposition = {
        'convention': ('Q = gross, settlement excluded; TN = the sum over the 12 TSO blocks of the block gross '
                       '(weighted); DN = the 36 DSO blocks; TN conventional = TSO generation_cost; DN flexibility = '
                       'DSO flexibility_cost_internal; for Q(P_NRF) the TN term is the arm\'s TSO block_components and '
                       'the DN term the residual (the W124 method, Table 12 / Fig. 5(b))'),
        'C_NRF': {k: v for k, v in t_nrf.items() if not k.startswith('_')},
        'C_OPEN': {k: v for k, v in t_open.items() if not k.startswith('_')},
        'P_NRF': {'gross_operational_cost': arm['gross_operational_cost'], 'TSO_block_gross_weighted': tso_p,
                  'DSO_residual': arm['gross_operational_cost'] - tso_p},
        'difference_1_P_NRF_minus_C_NRF': {
            'dTN': tso_p - t_nrf['by_agent']['TSO']['block_gross_weighted'],
            'dDN_residual': d1 - (tso_p - t_nrf['by_agent']['TSO']['block_gross_weighted'])},
        'difference_2_C_NRF_minus_C_OPEN': {
            'dTN_block_gross': t_nrf['by_agent']['TSO']['block_gross_weighted']
            - t_open['by_agent']['TSO']['block_gross_weighted'],
            'dTN_conventional': t_nrf['by_agent']['TSO']['generation_cost_weighted']
            - t_open['by_agent']['TSO']['generation_cost_weighted'],
            'dDN_block_gross': t_nrf['by_agent']['DSO']['block_gross_weighted']
            - t_open['by_agent']['DSO']['block_gross_weighted'],
            'dDN_flexibility': t_nrf['by_agent']['DSO']['flexibility_cost_weighted']
            - t_open['by_agent']['DSO']['flexibility_cost_weighted']},
        'curtailment_gwh': curt,
        'curtailment_definition': (
            'P_NRF: the Table 12 / E.23 column (report_v3 curtailment_table phase A, by agent, mwh_day_weighted / '
            '1000). C_NRF / C_OPEN: the record\'s res_curtailment_definitional_at_weight_1 (component levels, '
            'unweighted per representative day, signed) x the instance day weight (read from the price-taker arm\'s '
            'per-block day_weight) / 1000 -- the coordinated cells\' measure; its equivalence with the phase-A '
            'curtailment definition is NOT confirmed (Table 12 reports dashes for the coordinated row)')}
    # NRF terminal capture and H3c
    nrf = nrf_summary(rec)
    z7 = _rec('w182f_records')['Z7_row1_reverse_flow']
    rf_hours = [{'node': str(e['node_id']), 'year': str(e['year']), 'day': str(e['day']), 'period': int(e['period']),
                 'hour_1_based': e.get('hour'), 'p_int_dso_mw': e['p_int_dso_mw']}
                for e in z7['entries_most_negative_first']]
    bset = {(h['node'], h['year'], h['day'], int(h['hour'])) for h in (nrf['binding_hours_expected_value_rule']['hours']
                                                                      or [])}
    h3c = {'c_open_reverse_flow_hours': rf_hours,
           'c_open_reverse_flow_source': f"{REC['w182f_records']['path']} Z7_row1_reverse_flow (source "
                                         f"{z7.get('source')})",
           'z7_source_is_row_1': C_OPEN_EVAL_KEY[:16] in str(z7.get('source')),
           'matching': ('(node, year, day, period): the capture\'s binding_hours "hour" is the period index of '
                        '_get_interface_reporting_detail (0-based), Z7\'s "period" the same index'),
           'each_among_binding_hours': {f"{h['node']}|{h['year']}|{h['day']}|p{h['period']}":
                                        (h['node'], h['year'], h['day'], h['period']) in bset for h in rf_hours},
           'n_active_864_per_copy': nrf['n_active_per_copy_ok']}
    h3c['holds'] = (len(rf_hours) == 4 and all(h3c['each_among_binding_hours'].values())
                    and h3c['n_active_864_per_copy'] and h3c['z7_source_is_row_1'])
    return {'schema': 'p515_s55_step7_e3_report_v1', 'stage': STAGE, 'utc': _utc(), 'planted': planted,
            'objective_convention': OBJECTIVE_CONVENTION, 'cell': cdef['cell'], 'source': src,
            'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
            'cell_report': rep, 'status': rep['status'], 'Q_C_NRF': q_c,
            'k_star_or_end': rep['k_star'] if rep['status'] == 'certified' else rep['end_cycle'],
            'band_or_gap_slack': ({'band': rep['band'], 'band_width': rep['band_width']} if rep['status'] == 'certified'
                                  else {'gap': view['gap'], 'slack': view['slack'], 'harness_bar': rep['harness_bar_eur'],
                                        'k_cap': rep.get('k_cap'), 'Q_at_cap': rep.get('Q_at_cap'),
                                        't_sum_at_cap': rep.get('t_sum_at_cap'),
                                        'band_window': rep.get('band_window'), 'gap_refusals': rep.get('gap_refusals'),
                                        'drift_rate_mean_dQ_last_25': rep.get('drift_rate_mean_dQ_last_25')}),
            'stopping_ratio': rep['stopping_ratio_terminal_step_over_threshold'],
            'difference_1': diff1, 'difference_2': diff2, 'decomposition': decomposition, 'nrf_terminal': nrf,
            'H3c': h3c, 'cross_checks': xchecks,
            'records_read': {n: {'path': REC[n]['path'], 'sha256': REC[n]['sha256']}
                             for n in ('step6_tables', 'benchmark_report_v3', 'price_taker_best_arm', 'w182f_records',
                                       'c_open_component_levels', 'c_open_evaluation_record')}}


def _baseline(price, lattice):
    """The baseline pair at `price` read by eval key from the frozen step-6 tables: margin = Q(0) - Q(unit) - I(unit)
    and threshold = max(3 x the larger band, 2 tau), cross-checked against the committed claim of that pair."""
    keys = BASELINE_PAIRS[price]
    c0, cu = _step6_cell(keys['x0']), _step6_cell(keys['unit'])
    I = i_unit_eur(lattice)
    margin = c0['Q'] - cu['Q'] - I
    det, thr, binds = SC6.determinate_certified(margin, c0['band_width'], cu['band_width'])
    claims = [c for c in _rec('step6_tables')['tables']['claims']
              if {v['eval_key'] for v in c['instance'].values()} == {keys['x0'], keys['unit']}
              and c.get('claim_type') in ('sign', 'threshold')]
    xc = []
    for c in claims:
        d = c['d_gross']
        m_from_claim = -d if c.get('form') == 'F' else d
        xc.append({'claim_id': c['claim_id'], 'form': c.get('form'), 'd_gross': d,
                   'margin_from_claim': m_from_claim, 'margin_agrees_1e-6': abs(m_from_claim - margin) <= 1e-6,
                   'threshold': c['gross_threshold_or_bar'], 'threshold_agrees': abs(c['gross_threshold_or_bar']
                                                                                     - thr) <= 1e-9,
                   'I_other': c.get('I_other'), 'I_agrees': abs(c.get('I_other') - I) <= 1e-6,
                   'verdict': c.get('gross_verdict')})
    return {'price': price, 'rows': keys['rows'], 'x0_cell': c0.get('cell'), 'unit_cell': cu.get('cell'),
            'x0_eval_key': keys['x0'], 'unit_eval_key': keys['unit'], 'Q_x0': c0['Q'], 'Q_unit': cu['Q'],
            'band_x0': c0['band_width'], 'band_unit': cu['band_width'], 'status': [c0['status'], cu['status']],
            'I_unit': I, 'margin': margin, 'threshold': thr, 'binding_term': binds,
            'verdict': 'determinate' if det else 'within resolution', 'sign': (1 if margin > 0 else -1) if det else 0,
            'claims_cross_check': xc, 'claims_agree': bool(xc) and all(x['margin_agrees_1e-6'] and x['threshold_agrees']
                                                                         and x['I_agrees'] for x in xc)}


def report_e2(base_abs, require_git=True, planted=False):
    lattice, _lev, lfail = build_lattice()
    if lattice is None:
        raise RuntimeError(f'lattice: {lfail}')
    cells = cell_table('e2')
    per_cell, outcomes = {}, {}
    for cdef in cells:
        res, eval_dir, rec, src = _results_of(base_abs, cdef['cell'], require_git)
        if res is None:
            continue
        rep = cell_report(cdef, eval_dir, rec)
        code, oc = cell_outcome(cdef, rec, eval_dir)
        per_cell[cdef['cell']] = {'source': src, 'eval_key': rec.get('eval_key'), 'Q': rep['Q_end'],
                                  'status': rep['status'], 'band': rep['band'], 'band_width': rep['band_width'],
                                  'k_star_or_end': rep['k_star'] if rep['status'] == 'certified' else rep['end_cycle'],
                                  'harness_bar': rep['harness_bar_eur'],
                                  'stopping_ratio': rep['stopping_ratio_terminal_step_over_threshold'],
                                  'view': rep['view'], 'rho_read_back_holds': (oc.get('rho_read_back') or {}).get('holds'),
                                  'rho_read_back_cycle_1': (oc.get('rho_read_back') or {}).get('cycle_1'),
                                  'rho_read_back_n_checks': (oc.get('rho_read_back') or {}).get('n_checks'),
                                  'rho_mismatches': (oc.get('rho_read_back') or {}).get('mismatches'),
                                  'outcome_exit_code': code, 'stops': oc.get('stops')}
    I = i_unit_eur(lattice)
    base = {p: _baseline(p, lattice) for p in BASELINE_PAIRS}
    pairs = []
    for i in range(0, len(cells), 2):
        c0, cu = cells[i], cells[i + 1]
        price = '1x' if c0['flex_price_multiplier'] is None else '2x'
        pr = {'pair': [c0['cell'], cu['cell']], 'price': price, 'rho0_scale': c0['rho0_scale'],
              'complete': c0['cell'] in per_cell and cu['cell'] in per_cell}
        if pr['complete']:
            v0, vu = per_cell[c0['cell']]['view'], per_cell[cu['cell']]['view']
            m = v0['Q'] - vu['Q'] - I
            m_cc = (v0['Q_cc'] - vu['Q_cc'] - I) if (v0.get('Q_cc') is not None and vu.get('Q_cc') is not None) \
                else None
            r = resolve_pair(m, m_cc, (v0, vu))
            b = base[price]
            h2 = {'margin_shift_vs_baseline': m - b['margin'], 'baseline_threshold': b['threshold'],
                  'within_baseline_threshold': abs(m - b['margin']) <= b['threshold'],
                  'verdict_unchanged': r['verdict'] == b['verdict'] and r.get('sign') == b['sign'],
                  'baseline_verdict': b['verdict'], 'baseline_sign': b['sign']}
            h2['holds'] = h2['within_baseline_threshold'] and h2['verdict_unchanged']
            pr.update({'margin': m, 'margin_Qcc_report_only': m_cc, 'I_unit': I, 'resolution': r,
                       'verdict': r['verdict'], 'H2': h2})
        pairs.append(pr)
    complete = [p for p in pairs if p['complete']]
    margins = [p['margin'] for p in complete]
    return {'schema': 'p515_s55_step7_e2_report_v1', 'stage': STAGE, 'utc': _utc(), 'planted': planted,
            'objective_convention': OBJECTIVE_CONVENTION, 'n_cells_complete': len(per_cell), 'cells': per_cell,
            'margin_definition': 'margin = Q(x = 0) - Q(unit) - I(unit) within each pair (plan section 4, clarification 3)',
            'I_unit_source': f"{SEARCH_TABLES['investment_cost_w2']['path']} (PB lattice closed form, W2 cross-checked)",
            'verdict_rule': ('rule v6: determinate iff both certified and |margin| >= max{3 x the larger band, 2 tau}, '
                             f'tau = settling_criterion_v6.TAU = {SC6.TAU}; otherwise the uncertified form'),
            'pairs': pairs, 'baseline': base,
            'baseline_source': {'record': REC['step6_tables']['path'], 'sha256': REC['step6_tables']['sha256'],
                                'read_by_eval_key': BASELINE_PAIRS},
            'H2_rule': ('per pair, after it completes, in the order (1x, 0.5), (1x, 2.0), (2x, 0.5), (2x, 2.0): '
                        '|margin - baseline margin at that price| <= the baseline pair threshold at that price and '
                        'the verdict (and its sign) unchanged; a failed pair does not stop E2 (Addendum 86 ruling 3 '
                        '(i))'),
            'H2_per_pair': {f"{p['price']}_rho{p['rho0_scale']}": (p.get('H2') or {}).get('holds') for p in pairs},
            'margin_range_over_configurations': ({'min': min(margins), 'max': max(margins), 'n': len(margins)}
                                                 if margins else None),
            'rho_read_back_rule': RHO_RULE_TEXT,
            'rho_read_back_holds_per_cell': {c: v['rho_read_back_holds'] for c, v in per_cell.items()},
            'records_read': {'step6_tables': {'path': REC['step6_tables']['path'],
                                              'sha256': REC['step6_tables']['sha256']},
                             'investment_cost_w2': SEARCH_TABLES['investment_cost_w2']}}


def write_report(phase, out_dir_abs, payload, n_done=None):
    name = f'step7_{phase}_report.json' if phase == 'e3' else f'step7_{phase}_report_after_{n_done}of8.json'
    path = os.path.join(out_dir_abs, name)
    sha = _write_once_json(path, payload)
    _write_once_json(os.path.join(out_dir_abs, name.replace('.json', '_manifest_sha256.json')),
                     {_rel(path) if path.startswith(REPO) else path: sha, SCRIPT_NAME: _sha(SCRIPT_NAME),
                      **{REC[n]['path']: REC[n]['sha256'] for n in REC}})
    return path, sha


def report(args):
    base_abs = _abs(PHASE_ROOTS[args.phase])
    if args.phase == 'e3':
        payload = report_e3(base_abs, require_git=True)
        path, sha = write_report('e3', base_abs, payload)
    elif args.phase == 'e2':
        payload = report_e2(base_abs, require_git=True)
        path, sha = write_report('e2', base_abs, payload, n_done=payload['n_cells_complete'])
    else:
        _log(f'no report mode for {args.phase}')
        return 1
    _log(f'wrote {_rel(path)} sha256 {sha}')
    return 0


# ======================================================================================================================
#  --dry-run (W183i (b))
# ======================================================================================================================
def _plant_cell(base_abs, cdef, src_eval_dir, mutate_record=None, mutate_decision=None, per_cycle_rows=None,
                parent_exit=0):
    """Plant a completed run of `cdef` in a scratch-frozen cell root: copies of a committed eval dir's record,
    per-cycle record, settling decision and component levels, mutated as declared, plus a campaign_results.json."""
    root = cell_root(base_abs, cdef['cell'])
    spec_name = next(f for f in os.listdir(root) if f.startswith('campaign_spec_'))
    entry = _load(os.path.join(root, spec_name))['candidates'][0]
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    os.makedirs(eval_dir)
    rec = _load(os.path.join(src_eval_dir, 'evaluation_record.json'))
    rec.update({'eval_key': entry['eval_key'], 'candidate_key': entry['key'],
                'candidate_canonical': entry['canonical'], 'candidate_label': cdef['cell'],
                'parent_view': {'exit_code': parent_exit}, 'PLANTED': 'dry-run planted record (W183i (b))'})
    if mutate_record:
        mutate_record(rec)
    with open(os.path.join(eval_dir, 'evaluation_record.json'), 'x') as h:
        json.dump(rec, h)
    rows = per_cycle_rows if per_cycle_rows is not None else _read_jsonl(os.path.join(src_eval_dir,
                                                                                       'per_cycle_record.jsonl'))
    with open(os.path.join(eval_dir, 'per_cycle_record.jsonl'), 'x') as h:
        for r in rows:
            h.write(json.dumps(r) + '\n')
    dec_src = os.path.join(src_eval_dir, S7.DECISION_FILE)
    if os.path.isfile(dec_src):
        dec = _load(dec_src)
        if mutate_decision:
            mutate_decision(dec)
        with open(os.path.join(eval_dir, S7.DECISION_FILE), 'x') as h:
            json.dump(dec, h)
    shutil.copy(os.path.join(src_eval_dir, 'component_levels_terminal.json'),
                os.path.join(eval_dir, 'component_levels_terminal.json'))
    with open(os.path.join(root, RESULTS_FILE), 'x') as h:
        json.dump({'PLANTED': True, 'cell': cdef['cell'], 'campaign_spec_sha256': H.sha256_file(
            os.path.join(root, spec_name))}, h)
    return eval_dir, rec


def _synthetic_nrf_terminal(binding_hours, tso_active=864, dso_active=288):
    per_block = {'TSO': {f'{y}|{d}': {'present': True, 'n_indices': 72, 'n_active': 72, 'min_body_mw': 0.0,
                                      'n_body_le_tol': 1} for y in ('2025', '2030', '2035')
                         for d in ('Spring', 'Summer', 'Autumn', 'Winter')}}
    for n in (5, 7, 9):
        per_block[f'DSO{n}'] = {f'{y}|{d}': {'present': True, 'n_indices': 24, 'n_active': 24, 'min_body_mw': 0.0,
                                             'n_body_le_tol': 1} for y in ('2025', '2030', '2035')
                                for d in ('Spring', 'Summer', 'Autumn', 'Winter')}
    totals = {'TSO': {'n_blocks': 12, 'n_blocks_with_row': 12, 'n_indices': 864, 'n_active': tso_active}}
    for n in (5, 7, 9):
        totals[f'DSO{n}'] = {'n_blocks': 12, 'n_blocks_with_row': 12, 'n_indices': 288, 'n_active': dso_active}
    return {'status': 'written', 'row': 'PLANTED', 'binding_tolerance_mw': H.NRF_BINDING_TOLERANCE_MW,
            'binding_definition': 'PLANTED', 'row_counts_per_network': totals, 'row_counts_per_block': per_block,
            'n_binding_hours': len(binding_hours), 'binding_hours': binding_hours,
            'per_node': {'7': {'n_hours': 288, 'min_p_int_tso_expected_mw': 0.0, 'min_p_int_dso_expected_mw': 0.0}}}


def _planted_rho_rows(src_rows, s, n, perturb_cycle=None):
    """Per-cycle rows whose rho_*_after are the replicated mean at scale s, from the source rows' actions (planted:
    the positive control of the read-back), optionally with one value moved by one ulp (the negative control)."""
    import p515_g_g1_g4_admm_gates as G
    x = {'v': G.S39_RHO_V * s, 'pf': G.S39_RHO_PF * s, 'ess': G.S39_RHO_ESS * s}
    out = []
    for r in src_rows:
        r = dict(r)
        for g in RHO_CHANNELS:
            f = _rho_action_factor(r[f'rho_{g}_action'])
            if f is not None:
                x[g] = min(max(x[g] * f, RHO_CLAMP[0]), RHO_CLAMP[1])
            r[f'rho_{g}_after'] = sum([x[g]] * n[g]) / n[g]
        if perturb_cycle is not None and r['cycle'] == perturb_cycle:
            import math
            r['rho_pf_after'] = math.nextafter(r['rho_pf_after'], 1.0)
        out.append(r)
    return out


EXT_V6_E_NO_AGEING_EVAL_DIR = os.path.join(EXT_V6_ROOT_REL, 'campaign_s53_w142_resettle_ext_v6_e_no_ageing', 'evals',
                                           '1b10e9e161523f46_e_no_ageing')


def dry_run(args):
    started = time.time()
    out_abs = _abs(args.out)
    if not os.path.abspath(out_abs).startswith(_abs(P55) + os.sep) and not (
            SMOKE['allow_uncommitted'] and not os.path.abspath(out_abs).startswith(REPO + os.sep)):
        raise SystemExit(f'REFUSED: --out must be under {P55} (a smoke: outside the repository)')
    if os.path.isdir(out_abs) and [n for n in os.listdir(out_abs) if not LOG_RX.match(n)]:
        raise SystemExit(f'REFUSED: --out {args.out} is not fresh (only launch logs may be present)')
    os.makedirs(out_abs, exist_ok=True)
    scratch = os.path.abspath(args.scratch)
    if scratch.startswith(REPO + os.sep) or os.path.exists(scratch):
        raise SystemExit(f'REFUSED: --scratch must be a NEW directory outside the repository: {scratch}')
    os.makedirs(scratch)
    res = {'schema': 'p515_s55_step7_launcher_dryrun_v1', 'stage': STAGE, 'authority': AUTHORITY,
           'started_utc': _utc(), 'argv': sys.argv, 'interpreter': sys.executable,
           'git_head': H._git(['rev-parse', 'HEAD']), 'launcher_sha256': _sha(SCRIPT_NAME),
           'launcher_committed_and_clean': _committed_clean(SCRIPT_NAME),
           'objective_convention': OBJECTIVE_CONVENTION + ' (no objective value is computed by the key / freeze '
                                                          'parts; the report parts read committed and PLANTED values)',
           'scratch': scratch}
    A, info = {}, {}
    lattice, lev, lfail = build_lattice()
    if lfail:
        raise SystemExit(f'REFUSED (search tables): {lfail}')
    res['search_tables'] = lev
    A['pins'] = {'harness_is_the_patched_harness': H.sha256_file(H.HARNESS_PATH) == HARNESS_EXPECTED_SHA256,
                 'hooks_is_w183i_a': _sha(HOOKS_REL) == HOOKS_EXPECTED_SHA256,
                 'harness_and_hooks_committed_clean': _committed_clean(H.HARNESS_PATH) and _committed_clean(HOOKS_REL),
                 'launcher_committed_and_clean': res['launcher_committed_and_clean']}
    recs = record_pins_check()
    res['committed_records'] = recs
    A['committed_records'] = {n: v['holds'] for n, v in recs.items()}
    # 1. keys
    holders, n_specs, excluded = committed_eval_keys()
    res['committed_spec_scan'] = {'n_specs_scanned': n_specs, 'excluded_roots': list(STAGE_ROOTS),
                                  'n_excluded_specs': len(excluded), 'excluded_specs': excluded}
    tables, counts = {}, {}
    for phase in ('e3', 'e2', 'screen'):
        rows = []
        for cdef in cell_table(phase, lattice):
            kc = key_checks(cdef)
            rows.append({'cell': cdef['cell'], 'model': cdef['model'], 'candidate_key': cdef['candidate_key'],
                         'canonical': cdef['canonical'], 'nrf_option': cdef['nrf_option'],
                         'rho0_scale': cdef['rho0_scale'], 'flex_price_multiplier': cdef['flex_price_multiplier'],
                         'eval_key': kc['eval_key'], 'base_key': kc['base_key'],
                         'base_payload': kc['independent']['base_payload'],
                         'key_without': {k[len('key_without_'):]: v for k, v in kc.items()
                                         if k.startswith('key_without_')},
                         'w183c_on_x0': kc.get('w183c_on_x0'), 'checks': kc['checks'],
                         'committed_holders_outside_the_stage_roots': holders.get(kc['eval_key'], []),
                         'committed_holders_of_the_base_key': holders.get(kc['base_key'], [])
                         if kc['base_key'] != kc['eval_key'] else None})
            A[f'keys:{phase}:{cdef["cell"]}'] = kc['checks']
        tables[phase] = rows
        counts[phase] = len({r['eval_key'] for r in rows})
    res['key_tables'] = tables
    g0_ok, alias = screen_alias(lattice, require_current_production=True)
    res['screen_alias'] = alias
    A['screen_alias'] = {'g0_accepted_with_current_production': g0_ok,
                         'alias_key_not_evaluated': alias['alias_eval_key'] not in {r['eval_key']
                                                                                    for r in tables['screen']},
                         'alias_target_is_the_screen_x0_cell': alias['reuse_of'] == tables['screen'][0]['eval_key']}
    all_keys = [r['eval_key'] for p in tables for r in tables[p]]
    A['key_counts'] = {'e3_1': counts['e3'] == 1, 'e2_8': counts['e2'] == 8, 'screen_15': counts['screen'] == 15,
                       'all_24_keys_pairwise_distinct': len(set(all_keys)) == len(all_keys) == 24,
                       'screen_14_neighbours_under_N': sum(1 for r in tables['screen'] if r['model'] == 'N') == 14,
                       'screen_neighbours_are_the_lattice_unit_box_at_x0': (
                           lattice.completion(lattice.x0())['n_feasible'] == 14),
                       'e2_order_is_the_hooks_order': [r['cell'] for r in tables['e2']] == list(S7.E2_CELLS)}
    e3e2_holders = {r['cell']: r['committed_holders_outside_the_stage_roots'] for p in ('e3', 'e2')
                    for r in tables[p]}
    res['prediction'] = {
        'source': 'Addendum 86 W183i (b) / Addendum 85 W183g (c): key counts 1/8/15; 0 committed holders of the E3 and '
                  'E2 keys',
        'observed_key_counts': counts,
        'observed_committed_holders_e3_e2': {c: len(h) for c, h in e3e2_holders.items()},
        'observed_committed_holders_screen': {r['cell']: r['committed_holders_outside_the_stage_roots']
                                              for r in tables['screen'] if r['committed_holders_outside_the_stage_roots']},
        'held': (counts == {'e3': 1, 'e2': 8, 'screen': 15} and not any(e3e2_holders.values()))}
    # informative: the base keys against committed configurations (G0 digest keys, W183c)
    g0d = _load_g0(G0_ARTIFACT_REL)['summary']['eval_keys']
    info['base_keys_vs_g0_digest_keys'] = {
        'e2_x0_1x_without_rho = g0 x0_A_1x': eval_key_of(cell_table('e2')[0], with_declaration=False,
                                                          drop=('rho0_scale',)) == g0d['x0_A_1x'],
        'e2_unit_1x_without_rho = g0 unit_A_1x': eval_key_of(cell_table('e2')[1], with_declaration=False,
                                                              drop=('rho0_scale',)) == g0d['unit_A_1x'],
        'e2_x0_2x_without_rho = g0 x0_A_2x': eval_key_of(cell_table('e2')[4], with_declaration=False,
                                                          drop=('rho0_scale',)) == g0d['x0_A_2x'],
        'e2_unit_2x_without_rho = g0 unit_A_2x': eval_key_of(cell_table('e2')[5], with_declaration=False,
                                                              drop=('rho0_scale',)) == g0d['unit_A_2x'],
        'screen_x0_A_1x = g0 x0_A_1x': tables['screen'][0]['eval_key'] == g0d['x0_A_1x'],
        'screen_alias x0_N_1x = g0 x0_N_1x': alias['alias_eval_key'] == g0d['x0_N_1x']}
    # 3. the committed-spec scan excludes the stage roots (the W105 lesson), pure tests
    A['scan_exclusion'] = {
        'stage_root_spec_excluded': _excluded(os.path.join(PHASE_ROOTS['e3'], 'campaign_s55_step7_e3_cnrf_x0_1x',
                                                           'campaign_spec_x.json'), STAGE_ROOTS),
        'cross_root_spec_excluded': _excluded(os.path.join(PHASE_ROOTS['cross'], 'c', 'campaign_spec_x.json'),
                                              STAGE_ROOTS),
        'e1_2x_seeds_spec_not_excluded': not _excluded(os.path.join(P55, 'e1_2x', 'campaign_s55_e1_2x_seeds',
                                                                    'campaign_spec_x.json'), STAGE_ROOTS),
        'sibling_prefix_not_excluded': not _excluded(os.path.join(P55, 'e3_cnrf_other', 'campaign_spec_x.json'),
                                                     STAGE_ROOTS),
        'no_committed_spec_under_a_stage_root_yet': not excluded}
    # 4. the real roots: absent (nothing frozen into them by this task)
    A['real_roots'] = {f'{p}_root_absent_or_empty': (not os.path.exists(_abs(PHASE_ROOTS[p]))
                                                    or not os.listdir(_abs(PHASE_ROOTS[p]))) for p in PHASES}
    # 5. scratch freezes (one root per cell + the stage spec), validation, write-once refusals
    freezes = {}
    for phase in ('e3', 'e2', 'screen'):
        base = os.path.join(scratch, PHASE_ROOTS[phase])
        ok, summ = freeze_phase(phase, base, real=False)
        freezes[phase] = {'ok': ok, **summ}
        A[f'scratch_freeze:{phase}'] = {'all_cells_ok': ok,
                                        'n_cells': len(summ.get('cells') or {}) == {'e3': 1, 'e2': 8, 'screen': 15}[
                                            phase],
                                        'stage_spec_written': bool(summ.get('stage_spec'))}
        st = stage_spec(base, phase, require_git=False)
        A[f'scratch_freeze:{phase}']['stage_spec_self_hashing_and_pins_each_cell'] = bool(
            st and all(st[2]['cells'][c]['sha256'] == H.sha256_file(os.path.join(base, st[2]['cells'][c][
                'campaign_spec'])) for c in st[2]['cells']))
        ok2, again = freeze_phase(phase, base, real=False, log=lambda m: None)
        A[f'scratch_freeze:{phase}']['refreeze_refused_write_once'] = (not ok2 and any(
            'write-once' in r for r in again.get('refusals', [])))
    res['scratch_freezes'] = freezes
    # 2. configuration: the FROZEN (scratch) specs' configuration blocks against the committed ext-v6 specs
    A['configuration'] = {}
    res['configuration_vs_ext_v6'] = {}
    for phase, cell in (('e3', 'e3_cnrf_x0_1x'), ('e2', S7.E2_CELLS[0]), ('screen', 'x0_A_1x')):
        root = cell_root(os.path.join(scratch, PHASE_ROOTS[phase]), cell)
        cfg = _load(os.path.join(root, next(f for f in os.listdir(root) if f.startswith('campaign_spec_'))))[
            'configuration']
        vs6 = configuration_vs_ext_v6(cfg)
        res['configuration_vs_ext_v6'][phase] = vs6
        A['configuration'][f'{phase}_frozen_equals_ext_v6_minus_resettle'] = vs6['equal_except_text']
        A['configuration'][f'{phase}_ext_v6_seven_specs_agree'] = vs6['ext_v6_specs_agree']
    ok_c, cross = freeze_phase('cross', os.path.join(scratch, PHASE_ROOTS['cross']), real=False)
    f_c, _ = run_preconditions('cross', 'any', '0' * 64, os.path.join(scratch, PHASE_ROOTS['cross']), real=False)
    A['cross'] = {'freeze_refused': not ok_c, 'run_refused': bool(f_c) and 'declared only' in f_c[0]}
    res['cross'] = {'freeze': cross, 'run_refusals': f_c}
    # a planted non-empty cell root is refused
    base_p = os.path.join(scratch, 'planted_nonempty', PHASE_ROOTS['e3'])
    os.makedirs(os.path.join(cell_root(base_p, 'e3_cnrf_x0_1x')))
    with open(os.path.join(cell_root(base_p, 'e3_cnrf_x0_1x'), 'foreign.txt'), 'w') as h:
        h.write('planted')
    okp, sp = freeze_phase('e3', base_p, real=False, log=lambda m: None)
    A['write_once'] = {'planted_nonempty_cell_root_refused': not okp and any('write-once' in r
                                                                             for r in sp.get('refusals', [])),
                       'real_mode_refuses_a_non_declared_root': not freeze_phase(
                           'e3', os.path.join(scratch, 'not_declared'), real=True, log=lambda m: None)[0]}
    # --run preconditions on a scratch-frozen cell: they run, and refuse (not committed, not the declared root)
    st_e3 = stage_spec(os.path.join(scratch, PHASE_ROOTS['e3']), 'e3', require_git=False)
    e3_sha = st_e3[2]['cells']['e3_cnrf_x0_1x']['sha256']
    f_run, _ctx = run_preconditions('e3', 'e3_cnrf_x0_1x', e3_sha, os.path.join(scratch, PHASE_ROOTS['e3']),
                                    real=True)
    res['run_preconditions_on_the_scratch_e3_cell'] = f_run
    A['run_preconditions'] = {
        'refuses_non_declared_root': any('declared root' in f for f in f_run),
        'refuses_uncommitted_stage_spec': any('stage spec' in f for f in f_run),
        'refuses_uncommitted_campaign_spec': any('campaign spec is not committed' in f for f in f_run)}
    st_e2 = stage_spec(os.path.join(scratch, PHASE_ROOTS['e2']), 'e2', require_git=False)
    second = S7.E2_CELLS[1]
    f_order, _ = run_preconditions('e2', second, st_e2[2]['cells'][second]['sha256'],
                                   os.path.join(scratch, PHASE_ROOTS['e2']), real=False)
    A['run_preconditions']['launch_order_refuses_cell_2_before_cell_1'] = any('launch order' in f for f in f_order)
    # parent capture checklist on the scratch-frozen specs
    cap = {}
    for phase, cell in (('e3', 'e3_cnrf_x0_1x'), ('e2', S7.E2_CELLS[0]), ('e2', S7.E2_CELLS[7]),
                        ('screen', 'x0_A_1x')):
        base = os.path.join(scratch, PHASE_ROOTS[phase])
        cdef = next(c for c in cell_table(phase, lattice) if c['cell'] == cell)
        root = cell_root(base, cell)
        spec = _load(os.path.join(root, next(f for f in os.listdir(root) if f.startswith('campaign_spec_'))))
        ok, checks = parent_capture_checklist(cdef, spec)
        cap[f'{phase}:{cell}'] = {'ok': ok, 'groups': sorted(checks),
                                  'n_items': {k: len(v) if isinstance(v, dict) else None for k, v in checks.items()}}
        A[f'parent_capture_checklist:{phase}:{cell}'] = {'ok': ok}
    res['parent_capture_checklists'] = cap
    # 6. memory preflight at 1
    mem = memory_preflight(CONCURRENCY)
    res['memory_preflight_at_1'] = mem
    A['memory_preflight'] = {'computed_at_concurrency_1': mem.get('concurrency') == 1,
                             'rule_2_75_gib_per_slot': mem.get('per_slot_gib') == 2.75}
    info['memory_preflight_pass_now'] = mem.get('pass')
    # 7. the rho replication validation (W183f arm, s = 1) + informative on two committed long records
    v = rho_validation_w183f()
    res['rho_validation_w183f'] = v
    A['rho_validation_w183f'] = v['checks']
    longer = {}
    for name, ed in (('ext_v6_e_no_ageing_177_cycles', EXT_V6_E_NO_AGEING_EVAL_DIR),
                     ('row1_x0_181_cycles', os.path.dirname(REC['c_open_evaluation_record']['path']))):
        rows = _read_jsonl(os.path.join(ed, 'per_cycle_record.jsonl'))
        rec = _load(os.path.join(ed, 'evaluation_record.json'))
        rb = rho_read_back(rows, 1.0, model_counts(rec['solve_profile']))
        longer[name] = {'eval_dir': ed, 'per_cycle_sha256': _sha(os.path.join(ed, 'per_cycle_record.jsonl')),
                        'n_cycles': rb['n_cycles'], 'n_checks': rb['n_checks'], 'holds': rb['holds'],
                        'n_mismatches': len(rb['mismatches']), 'actions_seen': sorted({
                            l[g]['action'] for l in rb['per_cycle'] for g in RHO_CHANNELS if g in l})}
    info['rho_replication_on_committed_multi_cycle_records_s1'] = longer
    # 8. the report modes and the stops on PLANTED records
    planted = {}
    e3_base = os.path.join(scratch, PHASE_ROOTS['e3'])
    e3c = cell_table('e3')[0]
    z7 = _rec('w182f_records')['Z7_row1_reverse_flow']['entries_most_negative_first']
    bh = [{'node': str(e['node_id']), 'year': str(e['year']), 'day': str(e['day']), 'hour': int(e['period']),
           'p_int_tso_expected_mw': 0.0, 'p_int_dso_expected_mw': 0.0, 'binding_tso': True, 'binding_dso': True}
          for e in z7]
    ed_e3, _ = _plant_cell(e3_base, e3c, _abs(EXT_V6_E_NO_AGEING_EVAL_DIR),
                           mutate_record=lambda r: r.update({'nrf_option': True,
                                                             'nrf_terminal': _synthetic_nrf_terminal(bh)}))
    rep3 = report_e3(e3_base, require_git=False, planted=True)
    p3, s3 = write_report('e3', scratch, rep3)
    orig_root = cell_root(e3_base, e3c['cell'])
    before = {os.path.relpath(os.path.join(d, f), orig_root): H.sha256_file(os.path.join(d, f))
              for d, _x, fs in os.walk(orig_root) for f in fs}
    ok_r, retry = freeze_retry('e3', e3c['cell'], e3_base, real=False)
    after = {os.path.relpath(os.path.join(d, f), orig_root): H.sha256_file(os.path.join(d, f))
             for d, _x, fs in os.walk(orig_root) for f in fs}
    ok_r2, retry2 = freeze_retry('e3', e3c['cell'], e3_base, real=False, log=lambda m: None)
    r2_root = cell_root(e3_base, e3c['cell'], retry=True)
    r2_spec = _load(os.path.join(r2_root, retry.get('campaign_spec', '').split(os.sep)[-1])) if ok_r else {}
    f_r2, _ = run_preconditions('e3', e3c['cell'] + RETRY_SUFFIX, retry.get('sha256', ''), e3_base, real=False)
    res['retry_freeze'] = {'summary': retry, 'second_attempt': retry2, 'run_preconditions_r2_scratch': f_r2}
    A['retry'] = {'freeze_ok': ok_r, 'new_root_campaign_id_r2': r2_spec.get('campaign_id') == campaign_id_of(
        e3c['cell'], retry=True), 'retry_of_recorded': bool((r2_spec.get('extra') or {}).get('retry_of')),
                  'original_root_untouched': before == after, 'second_retry_refused': not ok_r2,
                  'r2_preconditions_find_the_retry_pin': not any('retry pin' in f for f in f_r2)}
    code_ok, oc_ok = cell_outcome(e3c, _load(os.path.join(ed_e3, 'evaluation_record.json')), ed_e3)
    planted['e3_report'] = {'path_in_scratch': p3, 'sha256': s3, 'content': rep3}
    stops = {'planted_ok_exit_0': code_ok == 0}
    rec_ok = _load(os.path.join(ed_e3, 'evaluation_record.json'))
    for label, mut, exit_child in (
            ('nrf_status_error_exit_3', lambda r: r.update({'nrf_terminal': {'status': 'error', 'error': 'PLANTED'}}), 2),
            ('n_active_863_exit_3', lambda r: r.update({'nrf_terminal': _synthetic_nrf_terminal(bh, tso_active=863)}),
             0),
            ('dso_n_active_287_exit_3', lambda r: r.update({'nrf_terminal': _synthetic_nrf_terminal(bh,
                                                                                                    dso_active=287)}),
             0)):
        r = copy.deepcopy(rec_ok)
        mut(r)
        r['parent_view'] = {'exit_code': exit_child}
        c, o = cell_outcome(e3c, r, ed_e3)
        stops[label] = c == 3 and bool(o['stops'])
    # uncertified E3: a planted decision with status uncertified (the uncertified form reported)
    unc_base = os.path.join(scratch, 'planted_uncertified', PHASE_ROOTS['e3'])
    freeze_phase('e3', unc_base, real=False, log=lambda m: None)
    dec0 = _load(os.path.join(_abs(EXT_V6_E_NO_AGEING_EVAL_DIR), S7.DECISION_FILE))

    def _uncert(d):
        d.update({'status': 'uncertified', 'k_star': None, 'k_cap': dec0['cap'], 'Q_at_cap': dec0['Q_k_star'],
                  't_sum_at_cap': dec0['t_sum_k_star'], 'Q_cc_at_cap': dec0['Q_cc_k_star'],
                  'band_window': dec0['band'], 'drift_rate_mean_dQ_last_25': 12.5, 'reasons': ['PLANTED'],
                  'gap_clause_refused_at_cap': False})
    ed_u, rec_u = _plant_cell(unc_base, e3c, _abs(EXT_V6_E_NO_AGEING_EVAL_DIR),
                              mutate_record=lambda r: r.update({'nrf_option': True,
                                                                'nrf_terminal': _synthetic_nrf_terminal(bh)}),
                              mutate_decision=_uncert)
    c_u, o_u = cell_outcome(e3c, rec_u, ed_u)
    stops['uncertified_exit_3_with_the_uncertified_form'] = (
        c_u == 3 and all(k in o_u['cell_report'] for k in ('k_cap', 'Q_at_cap', 't_sum_at_cap', 'band_window',
                                                           'gap_refusals', 'drift_rate_mean_dQ_last_25',
                                                           'harness_bar_eur',
                                                           'stopping_ratio_terminal_step_over_threshold')))
    rep3u = report_e3(unc_base, require_git=False, planted=True)
    planted['e3_report_uncertified'] = {'difference_2_resolution': rep3u['difference_2']['resolution'],
                                        'band_or_gap_slack': rep3u['band_or_gap_slack']}
    stops['uncertified_report_uses_the_uncertified_form'] = rep3u['difference_2']['resolution']['rule'] == \
        'uncertified form'
    A['e3_report_and_stops'] = dict(stops, **{
        'report_cross_checks_hold': all(rep3['cross_checks'].values()),
        'report_reads_records_by_eval_key': rep3['difference_2']['Q_C_OPEN'] == _step6_cell(C_OPEN_EVAL_KEY)['Q'],
        'decomposition_identity_c_open_below_1e-3_eur': abs(
            rep3['decomposition']['C_OPEN']['identity_residual_eur']) < 1e-3,
        'h3c_reads_the_four_row1_hours': len(rep3['H3c']['c_open_reverse_flow_hours']) == 4,
        'h3c_planted_binding_hours_match': rep3['H3c']['holds'] is True})
    # E2 planted: eight cells, the replicated rho at s (positive control), Q per cell from the baseline cells
    e2_base = os.path.join(scratch, PHASE_ROOTS['e2'])
    src_rows = _read_jsonl(os.path.join(_abs(EXT_V6_E_NO_AGEING_EVAL_DIR), 'per_cycle_record.jsonl'))
    src_rec = _load(os.path.join(_abs(EXT_V6_E_NO_AGEING_EVAL_DIR), 'evaluation_record.json'))
    n_src = model_counts(src_rec['solve_profile'])
    b1, b2 = _baseline('1x', lattice), _baseline('2x', lattice)
    offsets = {}
    for i, cdef in enumerate(cell_table('e2')):
        b = b1 if cdef['flex_price_multiplier'] is None else b2
        q_target = (b['Q_x0'] if cdef['plan'] == 'x0' else b['Q_unit']) + 1000.0 * (i % 3)
        offsets[cdef['cell']] = q_target
        rows = _planted_rho_rows(src_rows, cdef['rho0_scale'], n_src)

        def _mrec(r, s=cdef['rho0_scale']):
            r['rho0_scale'] = s
            r['rho0_scale_applied_in_child'] = {'rho0_scale': s, 'checks': {'PLANTED_every_entry_equals_s39_times_s':
                                                                            True}}

        def _mdec(d, q=q_target):
            d['Q_k_star'] = q
        _plant_cell(e2_base, cdef, _abs(EXT_V6_E_NO_AGEING_EVAL_DIR), mutate_record=_mrec, mutate_decision=_mdec,
                    per_cycle_rows=rows)
    rep2 = report_e2(e2_base, require_git=False, planted=True)
    p2, s2 = write_report('e2', scratch, rep2, n_done=rep2['n_cells_complete'])
    planted['e2_report'] = {'path_in_scratch': p2, 'sha256': s2, 'content': rep2}
    # negative control: one ulp on one cycle of one cell
    cneg = cell_table('e2')[2]
    neg_rows = _planted_rho_rows(src_rows, cneg['rho0_scale'], n_src, perturb_cycle=37)
    neg_dir = os.path.join(scratch, 'rho_negative_control')
    os.makedirs(neg_dir)
    with open(os.path.join(neg_dir, 'per_cycle_record.jsonl'), 'x') as h:
        for r in neg_rows:
            h.write(json.dumps(r) + '\n')
    shutil.copy(os.path.join(_abs(EXT_V6_E_NO_AGEING_EVAL_DIR), S7.DECISION_FILE), neg_dir)
    neg_rec = copy.deepcopy(src_rec)
    neg_rec.update({'eval_key': eval_key_of(cneg), 'rho0_scale_applied_in_child': {'checks': {'PLANTED': True}},
                    'parent_view': {'exit_code': 0}})
    c_neg, o_neg = cell_outcome(cneg, neg_rec, neg_dir)
    unknown_rows = [dict(r) for r in neg_rows]
    unknown_rows[5]['rho_v_action'] = 'PLANTED unknown action'
    rb_unknown = rho_read_back(unknown_rows, cneg['rho0_scale'], n_src)
    A['e2_report_and_stops'] = {
        'eight_cells_complete': rep2['n_cells_complete'] == 8,
        'four_pairs_scored': sum(1 for p in rep2['pairs'] if p['complete']) == 4,
        'rho_read_back_holds_8_of_8_planted_positive_control': all(rep2['rho_read_back_holds_per_cell'].values())
        and len(rep2['rho_read_back_holds_per_cell']) == 8,
        'rho_one_ulp_negative_control_exit_3': c_neg == 3 and len(o_neg['rho_read_back']['mismatches']) == 1,
        'rho_unknown_action_refused': not rb_unknown['holds'] and bool(rb_unknown['errors']),
        'baseline_1x_claims_agree': b1['claims_agree'], 'baseline_2x_claims_agree': b2['claims_agree'],
        'baseline_1x_threshold_13p1_k': round(b1['threshold'] / 1000.0, 1) == 13.1,
        'baseline_2x_threshold_12p3_k': round(b2['threshold'] / 1000.0, 1) == 12.3,
        'h2_evaluated_per_pair': all(p.get('H2') is not None for p in rep2['pairs'])}
    planted['e2_negative_control'] = {'exit_code': c_neg, 'mismatches': o_neg['rho_read_back']['mismatches'],
                                      'unknown_action_errors': rb_unknown['errors']}
    # the screen outcome (production exit): a planted record exits 0, a barrier exits 1
    sc = cell_table('screen', lattice)[0]
    c_s, o_s = cell_outcome(sc, dict(_rec('c_open_evaluation_record'), eval_key=eval_key_of(sc)), None)
    c_b, _o = cell_outcome(sc, {'status': 'error', 'barrier': True, 'barrier_cause': 'PLANTED'}, None)
    A['screen_outcome'] = {'planted_record_exit_0': c_s == 0, 'barrier_exit_1': c_b == 1}
    res['planted'] = planted
    # 9. guards
    res['informative'] = info
    res['repo_modules_imported_sha256_end'] = imported_repo_modules()
    shutil.rmtree(scratch)
    res['scratch_removed'] = not os.path.exists(scratch)
    g = guards_report()
    res['guards'] = g
    A['guards'] = {'zero_solves_pickle_build': g['all_zero'],
                   'parent_guard_verify_0': PARENT_GUARD.verify(0) == []}
    res['assertions'] = A
    failing = sorted(f'{grp}.{k}' for grp, cs in A.items() for k, v in cs.items() if v is not True)
    res['failing_assertions'] = failing
    res['all_assertions_hold'] = not failing
    res['finished_utc'] = _utc()
    res['wall_clock_s'] = time.time() - started
    out_path = os.path.join(out_abs, DRY_RUN_FILE)
    sha = _write_once_json(out_path, res)
    manifest = {_rel(out_path): sha, SCRIPT_NAME: _sha(SCRIPT_NAME), HOOKS_REL: _sha(HOOKS_REL),
                'p515_s44_campaign_harness.py': _sha('p515_s44_campaign_harness.py'),
                **{REC[n]['path']: _sha_or_none(REC[n]['path']) for n in REC},
                G0_ARTIFACT_REL: _sha_or_none(G0_ARTIFACT_REL), G0_RECORD['path']: _sha_or_none(G0_RECORD['path'])}
    _write_once_json(os.path.join(out_abs, DRY_RUN_MANIFEST), manifest)
    for phase in ('e3', 'e2', 'screen'):
        for r in tables[phase]:
            _log(f"  {phase:<6} {r['cell']:<34} eval {r['eval_key'][:16]} base {r['base_key'][:16]} holders "
                 f"{len(r['committed_holders_outside_the_stage_roots'])}")
    _log(f"alias {alias['alias_label']} {alias['alias_eval_key'][:16]} -> {alias['reuse_of'][:16]} G0 {g0_ok}")
    _log(f"prediction key counts 1/8/15, 0 E3/E2 holders: observed {counts} "
         f"{res['prediction']['observed_committed_holders_e3_e2']} -> {'HELD' if res['prediction']['held'] else 'NOT HELD'}")
    _log(f"rho validation W183f: {v['checks']}")
    _log(f"guards {g['all_zero']} {[(n, x['counts']) for n, x in g['solve_profile_guards'].items()]} pickle "
         f"{g['pickle']} build {g['model_build_tripwire']}")
    _log(f'wrote {_rel(out_path)} sha256 {sha}; failing assertions {failing}')
    return 0 if not failing else 1


# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--report', action='store_true')
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--cell')
    parser.add_argument('--spec-sha256')
    parser.add_argument('--preconditions-only', action='store_true')
    parser.add_argument('--retry-of')
    parser.add_argument('--scratch', help='--dry-run only: a NEW directory outside the repository')
    parser.add_argument('--out', default=os.path.join(P55, 'w183i_b_step7_dryrun'), help='--dry-run output dir')
    parser.add_argument('--allow-uncommitted', action='store_true',
                        help='--dry-run smoke only (scratch --out): tolerate the uncommitted launcher in the scratch '
                             'freezes; the launcher_committed_and_clean assertion still fails')
    args = parser.parse_args()
    alive = _own_process_alive()
    if alive:
        raise SystemExit(f'REFUSED: another copy of this launcher is alive: {alive}')
    if args.dry_run:
        if args.phase or args.cell or args.spec_sha256 or not args.scratch or args.retry_of:
            parser.error('--dry-run takes --scratch [--out] [--allow-uncommitted] only')
        SMOKE['allow_uncommitted'] = bool(args.allow_uncommitted)
        code = dry_run(args)
    elif args.freeze:
        if not args.phase or args.cell or args.spec_sha256 or args.scratch:
            parser.error('--freeze takes --phase [--retry-of CELL]')
        base_abs = _abs(PHASE_ROOTS[args.phase])
        if args.retry_of:
            ok, summ = freeze_retry(args.phase, args.retry_of, base_abs, real=True)
        else:
            ok, summ = freeze_phase(args.phase, base_abs, real=True)
        _log(f'freeze summary: {json.dumps(_jsonable(summ))[:4000]}')
        code = 0 if ok else 1
    elif args.run:
        if not args.phase or not args.cell or not args.spec_sha256 or args.scratch or args.retry_of:
            parser.error('--run takes --phase, --cell, --spec-sha256 [--preconditions-only]')
        code = run(args)
    else:
        if args.phase not in ('e3', 'e2') or args.cell or args.spec_sha256 or args.scratch:
            parser.error('--report takes --phase e3 | e2')
        code = report(args)
    g = guards_report()
    _log(f"final guards: all_zero={g['all_zero']} pickle={g['pickle']['counts']} build={g['model_build_tripwire']}")
    for _n, guard in reversed(all_guards()):
        guard.uninstall()
    sys.exit(code if g['all_zero'] else 1)


if __name__ == '__main__':
    main()
