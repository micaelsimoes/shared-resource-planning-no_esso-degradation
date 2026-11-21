"""
P5.15 S56 plan v2, validation step 1 -- GATES of the per-scenario TN-DN interface coupling option (production change,
DEFAULT OFF). Authority: S56_PLAN_v2.md (validation step 1; Expert items X1-X3); S56_SCENARIO_COUPLING_PROPOSAL.md
(F1 and its change list); Planner task "S56 step 1: per-scenario coupling option (default off)"; TASKS.md S56 order.

THE OPTION UNDER TEST (production; read with getattr(params, name, default) off `planning.params.admm`):
  per_scenario_interface_coupling  bool, default False
  per_scenario_voltage_mode        'pin' (V-2: V on the expectation channel, pin kept) | 'per_scenario' (V-1: a
                                   per-scenario V deviation channel, pin not constructed), default 'pin'
  row18_charge_reactive            bool, default True; False -> row 18's Q deviation Vars and rows are not constructed
The harness carries it as the evaluation option `interface_coupling` (keyed only when non-default).

GATES (frozen with predictions in `frozen_coupling_gates_spec_v1_<sha8>.json` before any of them runs):
  G-a  option OFF, zero solves: NL files (production form, labels off; and the labelled copy with .row/.col) of every
       captured TSO / DSO / ESSO block at the ADMM initialisation solve AND at the first ADMM cycle's solve are
       byte-identical before (label 'pre', the pre-change commit) and after (label 'post') the change, for
       (i) SRP1 x0 and the reference unit (every block), (ii) the five-block 3 x 3 x0 at alpha 0.5 (a declared block
       subset), (iii) the SRP1-grid 3 x 3 x0 at alpha 0.5 and at alpha 0 (a declared block subset).
  G-b  option ON (V-1, Q charge off), SRP1 x0, zero solves: every NL identical to the option-off post build; no new
       Param, marker, pin or deviation row on any block.
  G-c  option ON, SRP1-grid 3 x 3 x0 alpha 0.5, zero solves, two arms (V-2 'pin' + Q charge on; V-1 'per_scenario' +
       Q charge off): structural checks at initialisation and after ADMM preparation (see `tso_checks` / `dso_checks`).
  G-d  option OFF: SRP1 reference unit, two ADMM cycles through the campaign harness, pre-change vs post-change, bitwise
       on the per-cycle record (G1 files) and on every IPOPT solve (records minus log_path/log_bytes; final-statistics
       blocks minus timing lines). Run alone, attached, under the harness's campaign lock.
  G-e  every preserved fixture (the W183c gate-1 list: every tracked .pkl + the named untracked anchors +
       FrozenSMOPF/**) unpickles under the new code.
  G-f  option ON (V-1, Q charge off, alpha 0.5), SRP1-grid 3 x 3 x0: a 2-cycle smoke through the harness that runs end
       to end and records the new per-cycle fields (no correctness claim).

ZERO-SOLVE BUILD MECHANISM (G-a, G-b, G-c). One child process per build configuration (the oracle baseline can be
installed once per process). In the child: the derived instance (if any) is installed by the harness's own
`install_derived_instance`; the planning object is built by `p515_g_g1_g4_admm_gates._construct_arm_planning` and
configured by the campaign child's own hook `p515_s44_campaign_harness._config_hook_factory`; then production's own
`run_operational_planning(type='distributed')` runs with every `optimize` replaced ON THE INSTANCE by a stub:
  call 1 (initialisation): the blocks are NL-written / checked, every Var whose value is None is set to 0 clipped to
          its bounds (a stand-in point, identical in every build), and a SolverResults with status ok / optimal is
          returned, so production proceeds through `_prepare_*`, `update_*_to_admm` and the cycle-1 Param updates;
  call 2 (ADMM cycle 1): the ADMM-prepared blocks are NL-written / checked; the network stubs return ok; the ESSO
          stub raises a sentinel that ends the run.
One production function is replaced in the build child only: `_compute_common_admm_objective_scale` returns the
case file's fixed sigma (93635360), because the stand-in point cannot pass sigma's calibration assertion; the ADMM
objectives use the fixed sigma whether or not the computed value is replaced (`_resolve_common_admm_objective_scale`),
so the built models are the ones production builds. Nothing is solved: SolveProfileGuard(permitted=()) is installed
before any project import in every process and verified at exactly 0.

MODES (repo root, canonical interpreter, attached, alone, both streams captured, never detached):
  --freeze                    creates the gate directory (refuses if it exists) and the frozen spec. Zero solves.
  --build --label pre|post|on --spec-sha256 S   the build configurations of that label (one child each).
  --compare-builds --spec-sha256 S              G-a, G-b, G-c verdicts from the build outputs.
  --fixtures --spec-sha256 S                    G-e.
  --run-gd --label pre|post --spec-sha256 S     one G-d root (harness, cap 2, concurrency 1).
  --compare-gd --spec-sha256 S                  the G-d verdict.
  --smoke --spec-sha256 S                       G-f.
  (internal) --child-build --config C --label L --child-out F --scratch D
Objective convention for every value quoted: Q = gross_operational_cost (contracted interface settlement and voltage
pin excluded; salvage separate) unless a field names the other convention ('settlement excluded in full').
"""

import argparse
import gc
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 coupling gates (this process never solves)').install()

import pickle  # noqa: E402

_REAL_PICKLE_LOAD = pickle.load
PICKLE = {'permitted_paths': None, 'load': 0, 'loads': 0, 'refused_load': 0}


def _guarded_load(handle, *args, **kwargs):
    name = os.path.abspath(getattr(handle, 'name', '') or '')
    if PICKLE['permitted_paths'] is None or name not in PICKLE['permitted_paths']:
        PICKLE['refused_load'] += 1
        raise RuntimeError(f'S56 coupling gates: pickle.load of {name!r} refused (G-e declared list only)')
    PICKLE['load'] += 1
    return _REAL_PICKLE_LOAD(handle, *args, **kwargs)


def _blocked_loads(*_a, **_k):
    PICKLE['loads'] += 1
    raise RuntimeError('S56 coupling gates: pickle.loads is not permitted')


pickle.load, pickle.loads = _guarded_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402

# ======================================================================================================================
#  constants (all frozen into the spec)
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
CANONICAL_PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
SOLVER_PATH = '/usr/local/bin/ipopt'
GATE_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'coupling_option_v1')
SPEC_PREFIX = 'frozen_coupling_gates_spec_v1_'
STAGE = ('P5.15 S56 step 1 -- per-scenario TN-DN interface coupling option (default off): gates G-a..G-f')
AUTHORITY = [
    'S56_PLAN_v2.md, validation step 1 (implementation of F1 + V-1/V-2 + Q-charge switch, default off; identity gates)',
    'S56_SCENARIO_COUPLING_PROPOSAL.md (F1, change list)',
    'Planner task: S56 step 1 per-scenario coupling option (default off), gates G-a..G-f',
    'TASKS.md, S56 revised scope',
]
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CONVERGENCE_DEPTH_TAIL = {'compl_inf_tol': 1e-06, 'enabled': True}
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'calendar_retention_per_year': 0.985,
                       'calibration': {'cycles_n': 10000, 'eol_retention_r': 0.8, 'reference_dod_d': 0.8,
                                       'status': 'ACTIVE'},
                       'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8, 'minimum_soh': 0.7}
ESS_AGEING_LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
INVESTMENT_YEAR = 2025
CAP_BUILD = 300
REQUIRED_CONSECUTIVE_CYCLES = 10
X0 = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
UNIT = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
CANDIDATES = {'x0': X0, 'y2025__n7_p0.25_e1': UNIT}
ALPHA_05 = {'alpha': 0.5, 'floor': None}
COUPLING_PIN_Q = {'per_scenario_interface_coupling': True, 'per_scenario_voltage_mode': 'pin',
                  'row18_charge_reactive': True}
COUPLING_V1_NOQ = {'per_scenario_interface_coupling': True, 'per_scenario_voltage_mode': 'per_scenario',
                   'row18_charge_reactive': False}
COUPLING_DEFAULT = {'per_scenario_interface_coupling': False, 'per_scenario_voltage_mode': 'pin',
                    'row18_charge_reactive': True}
DERIVED_SOURCES = {
    'fiveblock': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'campaign_s53_w89_3x3_smoke_r2',
                                       'campaign_spec_s53_w89_3x3_smoke_r2_aa690387.json'),
                  'pointer': 'configuration.derived_instance'},
    'srp1grid': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'instance_record.json'),
                 'pointer': 'derived_instance_declaration'},
}
GRID_SUBSET = ('TSO|2025|Spring', 'TSO|2035|Winter', 'DSO7|2025|Spring', 'DSO5|2030|Summer', 'ESSO|node7')
FIVEBLOCK_SUBSET = ('TSO|2025|Spring', 'DSO7|2025|Spring', 'ESSO|node7')
BUILD_CONFIGS = {
    # G-a (built pre and post)
    'srp1_x0': {'instance': 'srp1', 'candidate': 'x0', 'premium': None, 'coupling': None, 'nl_blocks': 'all',
                'labels': ['pre', 'post'], 'gates': ['G-a', 'G-b reference']},
    'srp1_unit': {'instance': 'srp1', 'candidate': 'y2025__n7_p0.25_e1', 'premium': None, 'coupling': None,
                  'nl_blocks': 'all', 'labels': ['pre', 'post'], 'gates': ['G-a']},
    'fiveblock_x0_a05': {'instance': 'fiveblock', 'candidate': 'x0', 'premium': ALPHA_05, 'coupling': None,
                         'nl_blocks': list(FIVEBLOCK_SUBSET), 'labels': ['pre', 'post'], 'gates': ['G-a']},
    'srp1grid_x0_a05': {'instance': 'srp1grid', 'candidate': 'x0', 'premium': ALPHA_05, 'coupling': None,
                        'nl_blocks': list(GRID_SUBSET), 'labels': ['pre', 'post'], 'gates': ['G-a']},
    'srp1grid_x0_a0': {'instance': 'srp1grid', 'candidate': 'x0', 'premium': None, 'coupling': None,
                       'nl_blocks': list(GRID_SUBSET), 'labels': ['pre', 'post'], 'gates': ['G-a']},
    # G-b (built with label 'on', after the change)
    'srp1_x0_on': {'instance': 'srp1', 'candidate': 'x0', 'premium': None, 'coupling': COUPLING_V1_NOQ,
                   'nl_blocks': 'all', 'labels': ['on'], 'gates': ['G-b']},
    # G-c (label 'on')
    'srp1grid_x0_a05_on_pin': {'instance': 'srp1grid', 'candidate': 'x0', 'premium': ALPHA_05,
                               'coupling': COUPLING_PIN_Q, 'nl_blocks': list(GRID_SUBSET), 'labels': ['on'],
                               'gates': ['G-c']},
    'srp1grid_x0_a05_on_v1_noq': {'instance': 'srp1grid', 'candidate': 'x0', 'premium': ALPHA_05,
                                  'coupling': COUPLING_V1_NOQ, 'nl_blocks': list(GRID_SUBSET), 'labels': ['on'],
                                  'gates': ['G-c']},
}
STAGES = ('init', 'admm_cycle1')
FIXED_SIGMA = 93635360.0
# ---- new production names (the change's own; G-c reads them) ---------------------------------------------------------
DEV_KINDS_PQ = ('p', 'q')
DEV_PARAM = {'req': 'interface_dev_{k}_req', 'dual': 'dual_interface_dev_{k}_req', 'prox': 'prox_interface_dev_{k}_prev'}
WIRED_MARKER = 'per_scenario_interface_coupling_wired'
PER_CYCLE_COUPLING_FIELDS = (
    'interface_coupling_active', 'interface_coupling_voltage_mode', 'interface_coupling_row18_charge_reactive',
    'interface_coupling_aa_forced_off',
    'boyd_pf_r_expectation_part', 'boyd_pf_r_deviation_part', 'boyd_pf_s_expectation_part', 'boyd_pf_s_deviation_part',
    'boyd_v_r_expectation_part', 'boyd_v_r_deviation_part', 'boyd_v_s_expectation_part', 'boyd_v_s_deviation_part',
    'scenario_mismatch_p_max_mw', 'scenario_mismatch_p_rms_mw', 'scenario_mismatch_q_max_mvar',
    'scenario_mismatch_q_rms_mvar', 'scenario_mismatch_v_max_pu', 'scenario_mismatch_v_rms_pu',
    'scenario_mismatch_by_dn', 'priced_residual_pr', 'priced_residual_pr_abs', 'priced_residual_price_part',
    'priced_residual_dual_part', 'gross_operational_cost_settlement_excluded_in_full',
)
# ---- G-d ---------------------------------------------------------------------------------------------------------------
GD_CAP = 2
GD_LABEL = 'y2025__n7_p0.25_e1'
GD_ROOTS = {'pre': 's56cpl_gd_pre', 'post': 's56cpl_gd_post'}
CONFIGURATION_SRP1 = {
    'name': ('S56 step 1 coupling gates -- the published SRP1 single-scenario configuration as current production '
             'declares it (the concurrency gate / W187 x0_A_1x block): case file (AA keep_memory declared), ESS ageing '
             'baseline declared, convergence-depth tail declared enabled (1e-6), 1x flexibility price'),
    'arm_label': 's39_D',
    'case_file_anderson_acceleration': dict(CASE_FILE_AA),
    'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL),
    'ess_ageing_baseline': deepcopy(ESS_AGEING_BASELINE),
    'ess_ageing_baseline_label': ESS_AGEING_LABEL,
    'note': 'no overrides, no model variant, no flex variant, no NRF, no rho0 scale, no post-certification',
    'overrides': {},
}
G1_FILES = ('g_s39_D.json#cycle_trajectory', 'per_cycle_record.jsonl', 'aa_per_cycle.jsonl')
G2_RECORDS_FILE = 'network_ipopt_solve_records.jsonl'
G2_RECORDS_EXCLUDED = {
    '[*].log_path': 'path (the IPOPT log path names the working dir, which carries the campaign id)',
    '[*].log_bytes': 'byte offsets into the log file (the log embeds its own path and the IPOPT timing lines)',
}
IPOPT_FINAL_START = 'Number of Iterations....:'
IPOPT_TIMING_PREFIXES = ('Total seconds in IPOPT', 'Total CPU secs', 'Total CPU seconds', 'Total wall-clock secs',
                         'Total wall clock')
P56A_WORK_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals')
SELFTEST = {   # committed artifacts written by the same harness and production (concurrency gate v1)
    'same_A': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'concurrency_gate_v1',
                           'campaign_s56_cgate_v1_rootA_serial', 'evals', 'ca8927e75d628bd1_y2025__n7_p0_25_e1'),
    'same_B': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'concurrency_gate_v1',
                           'campaign_s56_cgate_v1_rootB_concurrent', 'evals', 'ca8927e75d628bd1_y2025__n7_p0_25_e1'),
    'other_B': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'concurrency_gate_v1',
                            'campaign_s56_cgate_v1_rootB_concurrent', 'evals', '5cfe69a615ae3708_x0'),
}
# ---- G-f ---------------------------------------------------------------------------------------------------------------
GF_CAP = 2
GF_ROOT = 's56cpl_gf_smoke'
GF_LABEL = 'x0'
GF_OPTIONS = {'interface_deviation_premium': dict(ALPHA_05), 'interface_coupling': dict(COUPLING_V1_NOQ)}
GF_WALL_LIMIT_NOTE = ('projected: init + precheck + 2 cycles at ~6-10 min per cycle on 12 blocks (five-block 3x3 '
                      'smoke: 609 / 691 s per cycle at 20 blocks) -> well under 1.5 h; full instance, no block subset')
# ---- G-e ---------------------------------------------------------------------------------------------------------------
NAMED_FIXTURES = (   # p515_s53_w183c_nrf_gates.NAMED_FIXTURES (the W183c gate-1 list), copied
    'data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl',
    'data/SRP1/Results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
    'data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl',
    'data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl',
    'data/SRP1/Results/P512R/cycle21_prepared/snapshot.pkl',
    'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
    'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl',
    'data/SRP1/Results/P515G3F_r2/results/FrozenSMOPF/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl',
    'data/SRP1/Results/P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl',
)
FROZEN_SMOPF_GLOB_DIR = os.path.join('data', 'SRP1', 'Results', 'FrozenSMOPF')
PREDICTIONS = {
    'G-a': 'every captured NL (plain, labelled, .row, .col) of every off configuration is byte-identical pre vs post, '
           'at initialisation and at the first ADMM cycle',
    'G-b': 'SRP1 x0 with the option on: every NL byte-identical to the option-off post build; nothing new on any block',
    'G-c': ('3 x 3 option on: at initialisation the non-alias delta copies are fixed at 0 and every interface '
            'expression references the alias copy only; after ADMM preparation every delta copy is free within '
            '+-rating and the TN interface expression, net load and node-balance rows reference the per-scenario copy; '
            'new req / dual Params = 2 S per (DN, block, t) under pin and 3 S under per_scenario (S = 9), on both '
            'sides; the pin absent under per_scenario and present under pin; row 18 Q Vars / rows absent with the Q '
            'charge off and present with it on'),
    'G-d': 'G1 and G2 hold bitwise (pre vs post, option off, SRP1 unit, 2 cycles)',
    'G-e': 'every declared fixture unpickles',
    'G-f': 'the 2-cycle smoke completes (exit 0, 2 cycles) and every new per-cycle field is present and finite',
}
FAILURE_POLICY = ('a gate failure is recorded and investigated (which block / field / cycle), never re-run or tuned; '
                  'the first differences are written into the result')


# ======================================================================================================================
#  small helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[S56-CPL {datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _rel(path):
    return os.path.relpath(path, REPO)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_or_none(rel):
    p = _abs(rel)
    return _sha_file(p) if os.path.isfile(p) else None


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_once_json(path, obj):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite: {path}')
    text = GRIO.dumps(obj, indent=1, sort_keys=True, default=GRIO.json_default)
    with open(path, 'w') as handle:
        handle.write(text)
    return _sha_file(path)


def _gate_dir():
    return _abs(GATE_DIR_REL)


def _find_spec(spec_sha256):
    d = _gate_dir()
    hits = [f for f in sorted(os.listdir(d)) if f.startswith(SPEC_PREFIX) and f.endswith('.json')] \
        if os.path.isdir(d) else []
    match = [f for f in hits if _sha_file(os.path.join(d, f)) == spec_sha256]
    if len(match) != 1:
        raise RuntimeError(f'frozen spec with sha256 {spec_sha256} not found exactly once in {GATE_DIR_REL}: {hits}')
    path = os.path.join(d, match[0])
    spec = _load(path)
    spec['_path'] = path
    spec['_sha256'] = spec_sha256
    return spec


def production_modules():
    return sorted(set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) | {
        'shared_energy_storage_parameters.py', 'p515_s53_w118_resettle_hooks.py', 'network_parameters.py'})


def provenance():
    files = production_modules() + [SCRIPT_NAME, 'gate_result_io.py', 'p513_solve_profile_guard.py',
                                    H.ESS_PARAMS_FILE_REL, 'data/SRP1/SRP1.json']
    status = _git(['status', '--porcelain', '--'] + files)
    return {'git_head': _git(['rev-parse', 'HEAD']).strip(), 'utc': _utc(),
            'sha256': {f: _sha_or_none(f) for f in files},
            'git_status_porcelain': status, 'interpreter': sys.executable,
            'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH')}


def derived_declaration(instance):
    if instance == 'srp1':
        return None
    src = DERIVED_SOURCES[instance]
    obj = _load(src['path'])
    for part in src['pointer'].split('.'):
        obj = obj[part]
    return H.validate_derived_instance(obj)


def spec_like(derived):
    pin = {'path': H.ESS_PARAMS_FILE_REL, 'sha256': _sha_file(_abs(H.ESS_PARAMS_FILE_REL))}
    cfg = {'overrides': {}, 'arm_label': 's39_D', 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
           'ess_ageing_baseline': deepcopy(ESS_AGEING_BASELINE), 'ess_params_file': pin,
           'ess_ageing_baseline_label': ESS_AGEING_LABEL, 'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL)}
    if derived is not None:
        cfg['derived_instance'] = derived
    return {'configuration': cfg, 'cap': CAP_BUILD, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES}


# ======================================================================================================================
#  the zero-solve build child
# ======================================================================================================================
class _StopCapture(Exception):
    """Raised by the ESSO stub at ADMM cycle 1: every block has been captured; the run ends here."""


def _fake_ok():
    from pyomo.opt import SolverResults, SolverStatus, TerminationCondition
    r = SolverResults()
    r.solver.status = SolverStatus.ok
    r.solver.termination_condition = TerminationCondition.optimal
    return r


def _fill_none(block):
    """Stand-in point: every Var with no value gets 0 clipped to its bounds. Returns the count set."""
    import pyomo.environ as pe
    n = 0
    for v in block.component_data_objects(pe.Var, descend_into=True):
        if v.value is None:
            lb, ub = v.lb, v.ub
            x = 0.0
            if lb is not None and x < lb:
                x = lb
            if ub is not None and x > ub:
                x = ub
            v.set_value(x, skip_validation=True)
            n += 1
    return n


def _strip_nl_comments(text):
    return '\n'.join(line.split('#', 1)[0].rstrip() for line in text.splitlines())


def _drop_name_length_line(text):
    lines = text.splitlines()
    return '\n'.join(line for i, line in enumerate(lines) if not (i < 10 and 'max name lengths' in line))


def _gzip_file(src):
    with open(src, 'rb') as handle:
        data = handle.read()
    with open(src + '.gz', 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as gz:
            gz.write(data)
    os.remove(src)


def write_nl(model, nl_dir, stem):
    """Plain NL (production form: labels off) and the labelled copy (+ .row / .col); sha256 + bytes of each; the two
    must agree after comment stripping (bar the 'max name lengths' header line). Files are gzip'ed in the scratch dir."""
    rec = {'error': None}
    plain = os.path.join(nl_dir, f'{stem}.nl')
    sym = os.path.join(nl_dir, f'{stem}.sym.nl')
    try:
        model.write(plain, format='nl', io_options={'symbolic_solver_labels': False})
        model.write(sym, format='nl', io_options={'symbolic_solver_labels': True})
    except Exception as error:  # noqa: BLE001 -- recorded; the build check fails
        rec['error'] = f'{type(error).__name__}: {error}'
        return rec
    with open(plain) as handle:
        plain_text = handle.read()
    with open(sym) as handle:
        sym_text = handle.read()
    rec['stripped_equal'] = (_strip_nl_comments(_drop_name_length_line(plain_text))
                             == _strip_nl_comments(_drop_name_length_line(sym_text)))
    del plain_text, sym_text
    for name, path in (('plain', plain), ('sym_nl', sym), ('sym_row', os.path.join(nl_dir, f'{stem}.sym.row')),
                       ('sym_col', os.path.join(nl_dir, f'{stem}.sym.col'))):
        rec[name] = {'sha256': _sha_file(path), 'bytes': os.path.getsize(path)}
        _gzip_file(path)
    return rec


def _vars_named(expr, comp_name):
    from pyomo.core.expr.visitor import identify_variables
    return sorted(str(v.index()) for v in identify_variables(expr, include_fixed=True)
                  if v.parent_component().name == comp_name)


def _params_in(expr):
    from pyomo.core.expr.visitor import identify_mutable_parameters
    out = {}
    for prm in identify_mutable_parameters(expr):
        out.setdefault(prm.parent_component().name, set()).add(prm.index())
    return out


def tso_checks(stage, block, planning, year, day, coupling_active, voltage_mode):
    """G-c / G-b structural checks of one TSO block. Returns {check: bool} plus detail."""
    import pyomo.environ as pe
    tn = planning.transmission_network
    net = tn.network[year][day]
    s_base = net.baseMVA
    S = [(sm, so) for sm in block.scenarios_market for so in block.scenarios_operation]
    alias = S[0]
    T = list(block.periods)
    ok, det = {}, {}
    n_dn = len(tn.active_distribution_network_nodes)
    dev_names = {k: {r: DEV_PARAM[r].format(k=k) for r in DEV_PARAM} for k in ('p', 'q', 'v')}
    if not coupling_active:
        ok['no_deviation_param_on_block'] = not any(hasattr(block, dev_names[k][r]) for k in dev_names
                                                    for r in dev_names[k])
        ok['no_wired_marker'] = not getattr(block, WIRED_MARKER, False)
        if len(S) == 1:
            ok['no_pin_at_one_scenario'] = not hasattr(block, 'scenario_voltage_pin')
        return ok, det
    bad_fix, bad_bounds, bad_refs, bad_balance = [], [], [], []
    for dn in range(n_dn):
        node_id = tn.active_distribution_network_nodes[dn]
        rating = planning.distribution_networks[node_id].network[year][day].get_interface_branch_rating() / s_base
        node_idx = net.get_node_idx(node_id)
        for (sm, so) in S:
            for p in T:
                for kind, comp, expr_comp, bal in (('p', 'interface_delta_p', 'pc_adn', 'node_balance_p'),
                                                   ('q', 'interface_delta_q', 'qc_adn', 'node_balance_q')):
                    v = getattr(block, comp)[dn, sm, so, p]
                    is_alias = (sm, so) == alias
                    expect_index = str((dn, sm, so, p)) if stage == 'admm_cycle1' else str((dn,) + alias + (p,))
                    if stage == 'init':
                        if is_alias:
                            if v.fixed or v.lb != -rating or v.ub != rating:
                                bad_bounds.append((kind, dn, sm, so, p))
                        elif not (v.fixed and v.value == 0.0):
                            bad_fix.append((kind, dn, sm, so, p))
                    else:
                        if v.fixed or v.lb != -rating or v.ub != rating:
                            bad_bounds.append((kind, dn, sm, so, p))
                    refs = _vars_named(getattr(block, expr_comp)[dn, sm, so, p].expr, comp)
                    if refs != [expect_index]:
                        bad_refs.append((kind, dn, sm, so, p, refs))
                    bal_refs = _vars_named(getattr(block, bal)[node_idx, sm, so, p].body, comp)
                    if bal_refs != [expect_index]:
                        bad_balance.append((kind, dn, sm, so, p, bal_refs))
    if stage == 'init':
        ok['init_non_alias_delta_fixed_at_0'] = not bad_fix
        ok['init_alias_delta_free_within_rating'] = not bad_bounds
        ok['init_interface_expression_references_alias_only'] = not bad_refs
        ok['init_node_balance_references_alias_only'] = not bad_balance
        ok['init_no_deviation_param_yet'] = not any(hasattr(block, dev_names[k][r]) for k in dev_names
                                                    for r in dev_names[k])
    else:
        ok['admm_every_delta_free_within_rating'] = not bad_bounds
        ok['admm_interface_expression_references_own_scenario_delta'] = not bad_refs
        ok['admm_node_balance_references_own_scenario_delta'] = not bad_balance
        ok['admm_wired_marker'] = getattr(block, WIRED_MARKER, False) is True
        kinds = ('p', 'q', 'v') if voltage_mode == 'per_scenario' else ('p', 'q')
        absent = ('v',) if voltage_mode == 'pin' else ()
        counts = {}
        for k in kinds:
            for r in ('req', 'dual'):
                comp = getattr(block, dev_names[k][r], None)
                counts[dev_names[k][r]] = None if comp is None else len(comp)
        per_dn_t = {r: sum(counts[dev_names[k][r]] or 0 for k in kinds) for r in ('req', 'dual')}
        det['param_counts'] = counts
        expected_total = len(kinds) * len(S) * n_dn * len(T)
        ok['admm_req_params_count_is_kinds_x_S_per_dn_block_t'] = per_dn_t['req'] == expected_total
        ok['admm_dual_params_count_is_kinds_x_S_per_dn_block_t'] = per_dn_t['dual'] == expected_total
        ok['admm_no_v_params_under_pin'] = all(not hasattr(block, dev_names[k][r]) for k in absent for r in DEV_PARAM)
        ok['admm_pin_present_iff_pin_mode'] = hasattr(block, 'scenario_voltage_pin') == (voltage_mode == 'pin')
        pin_obj = _params_in(block.admm_objective.expr)
        ok['admm_objective_references_every_req_and_dual_param'] = all(
            len(pin_obj.get(dev_names[k][r], ())) == expected_total // len(kinds) for k in kinds
            for r in ('req', 'dual'))
        det['n_bad'] = {'bounds': len(bad_bounds), 'refs': len(bad_refs), 'balance': len(bad_balance)}
    det['first_bad'] = {'fix': bad_fix[:3], 'bounds': bad_bounds[:3], 'refs': [str(x) for x in bad_refs[:3]],
                        'balance': [str(x) for x in bad_balance[:3]]}
    det['n_scenarios'] = len(S)
    _ = pe
    return ok, det


def dso_checks(stage, block, coupling, premium):
    ok, det = {}, {}
    S = [(sm, so) for sm in block.scenarios_market for so in block.scenarios_operation]
    T = list(block.periods)
    dev_names = {k: {r: DEV_PARAM[r].format(k=k) for r in DEV_PARAM} for k in ('p', 'q', 'v')}
    on = coupling is not None and coupling.get('per_scenario_interface_coupling') is True and len(S) > 1
    reactive = True if coupling is None else coupling.get('row18_charge_reactive', True)
    mode = 'pin' if coupling is None else coupling.get('per_scenario_voltage_mode', 'pin')
    alpha = (premium or {}).get('alpha', 0.0) or 0.0
    if len(S) > 1 and alpha > 0.0:
        ok['row18_p_rows_present'] = hasattr(block, 'row18_dev_p_def') and hasattr(block, 'row18_dev_p_up')
        q_present = all(hasattr(block, n) for n in ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down'))
        q_absent = not any(hasattr(block, n) for n in ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down'))
        ok['row18_q_rows_present_iff_q_charge_on'] = q_present if reactive else q_absent
    if len(S) > 1:
        ok['pin_present_iff_not_per_scenario_mode'] = (hasattr(block, 'scenario_voltage_pin')
                                                       == (not (on and mode == 'per_scenario')))
    if not on:
        ok['no_deviation_param_on_block'] = not any(hasattr(block, dev_names[k][r]) for k in dev_names
                                                    for r in dev_names[k])
        return ok, det
    if stage == 'admm_cycle1':
        kinds = ('p', 'q', 'v') if mode == 'per_scenario' else ('p', 'q')
        counts = {dev_names[k][r]: (len(getattr(block, dev_names[k][r])) if hasattr(block, dev_names[k][r]) else None)
                  for k in kinds for r in ('req', 'dual')}
        det['param_counts'] = counts
        expected = len(kinds) * len(S) * len(T)
        ok['admm_req_params_count_is_kinds_x_S_per_block_t'] = sum(
            counts[dev_names[k]['req']] or 0 for k in kinds) == expected
        ok['admm_dual_params_count_is_kinds_x_S_per_block_t'] = sum(
            counts[dev_names[k]['dual']] or 0 for k in kinds) == expected
        pin_obj = _params_in(block.admm_objective.expr)
        ok['admm_objective_references_every_req_and_dual_param'] = all(
            len(pin_obj.get(dev_names[k][r], ())) == len(S) * len(T) for k in kinds for r in ('req', 'dual'))
        ok['admm_no_v_params_under_pin'] = (mode == 'per_scenario') or not any(
            hasattr(block, dev_names['v'][r]) for r in DEV_PARAM)
    else:
        ok['init_no_deviation_param_yet'] = not any(hasattr(block, dev_names[k][r]) for k in dev_names
                                                    for r in dev_names[k])
    return ok, det


class Capture:
    def __init__(self, cfg, planning, nl_dir):
        self.cfg = cfg
        self.planning = planning
        self.nl_dir = nl_dir
        self.calls = {}
        self.nl = {s: {} for s in STAGES}
        self.checks = {s: {} for s in STAGES}
        self.details = {s: {} for s in STAGES}
        self.filled = {}
        self.coupling = cfg['coupling']
        tn = planning.transmission_network
        net0 = tn.network[next(iter(tn.years))][next(iter(tn.days))]
        n_s = len(net0.prob_market_scenarios) * len(net0.prob_operation_scenarios)
        self.coupling_active = (self.coupling is not None
                                and self.coupling.get('per_scenario_interface_coupling') is True and n_s > 1)
        self.voltage_mode = (self.coupling or {}).get('per_scenario_voltage_mode', 'pin')

    def _want_nl(self, key):
        return self.cfg['nl_blocks'] == 'all' or key in self.cfg['nl_blocks']

    def _block(self, stage, key, block, kind, year=None, day=None):
        if self._want_nl(key):
            self.nl[stage][key] = write_nl(block, self.nl_dir, f'{stage}__{re.sub(r"[^A-Za-z0-9_.-]+", "_", key)}')
        if kind == 'TSO':
            ok, det = tso_checks(stage, block, self.planning, year, day, self.coupling_active, self.voltage_mode)
        elif kind == 'DSO':
            ok, det = dso_checks(stage, block, self.coupling, self.cfg['premium'])
        else:
            ok, det = {}, {}
        self.checks[stage][key] = ok
        if det:
            self.details[stage][key] = det

    def network_stub(self, tag):
        def stub(model, *args, **kwargs):
            n = self.calls[tag] = self.calls.get(tag, 0) + 1
            if n > 2:
                raise _StopCapture(f'{tag} called a third time')
            stage = STAGES[n - 1]
            results = {}
            for year in model:
                results[year] = {}
                for day in model[year]:
                    key = f'{tag}|{year}|{day}'
                    self._block(stage, key, model[year][day], 'TSO' if tag == 'TSO' else 'DSO', year, day)
                    if stage == 'init':
                        self.filled[key] = _fill_none(model[year][day])
                    results[year][day] = _fake_ok()
            return results
        return stub

    def esso_stub(self, models, *args, **kwargs):
        n = self.calls['ESSO'] = self.calls.get('ESSO', 0) + 1
        stage = STAGES[n - 1]
        for node_id, model in models.items():
            key = f'ESSO|node{node_id}'
            self._block(stage, key, model, 'ESSO')
            if stage == 'init':
                self.filled[key] = _fill_none(model)
        if stage == 'admm_cycle1':
            raise _StopCapture('ESSO cycle-1 capture complete')
        return {node_id: _fake_ok() for node_id in models}


def child_build(args):
    t0 = time.time()
    cfg = BUILD_CONFIGS[args.config]
    out = {'schema': 'p515_s56_coupling_build_child_v1', 'config': args.config, 'label': args.label,
           'declaration': cfg, 'started_utc': _utc(), 'provenance': provenance(), 'error': None}
    try:
        import p515_g_g1_g4_admm_gates as G
        import shared_resources_planning as srp
        derived = derived_declaration(cfg['instance'])
        out['derived_instance'] = derived
        scratch = os.path.abspath(args.scratch)
        os.makedirs(scratch, exist_ok=False)
        if derived is not None:
            out['derived_installed'] = H.install_derived_instance(derived, os.path.join(scratch, 'derived'))
        report, holder = {}, {}
        eval_id = f"p515s56cpl_{args.label}_{args.config}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        planning, sed, candidate = G._construct_arm_planning(
            's39_D', os.path.join(scratch, 'arm'), report, investment_map=CANDIDATES[cfg['candidate']],
            eval_id=eval_id, num_max_iters_override=CAP_BUILD, apply_rho=False, investment_year=INVESTMENT_YEAR)
        hook_kwargs = dict(overrides={}, model_variant=None, investment_year=INVESTMENT_YEAR, expected_floor_rows=None,
                           flex_price_multiplier=None, interface_deviation_premium=cfg['premium'],
                           release_solution_bookkeeping=None)
        if cfg['coupling'] is not None:
            hook_kwargs['interface_coupling'] = cfg['coupling']
        hook = H._config_hook_factory(spec_like(derived), holder, **hook_kwargs)
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        a = planning.params.admm
        out['eval_id'] = eval_id
        out['admm_in_force'] = {'interface_deviation_premium': dict(a.interface_deviation_premium),
                                'per_scenario_interface_coupling': getattr(a, 'per_scenario_interface_coupling',
                                                                           'ABSENT_ATTRIBUTE'),
                                'per_scenario_voltage_mode': getattr(a, 'per_scenario_voltage_mode', 'ABSENT_ATTRIBUTE'),
                                'row18_charge_reactive': getattr(a, 'row18_charge_reactive', 'ABSENT_ATTRIBUTE'),
                                'objective_scale': a.objective_scale,
                                'anderson_acceleration': dict(a.anderson_acceleration)}
        out['hook_checklist'] = report.get('rule_eleven_checklist')
        if a.objective_scale != FIXED_SIGMA:
            raise RuntimeError(f'fixed sigma {a.objective_scale} != {FIXED_SIGMA}: the scale replacement would '
                               f'change the built models')
        original_scale = srp._compute_common_admm_objective_scale
        srp._compute_common_admm_objective_scale = (
            lambda planning_problem, tso_model, dso_models: float(planning_problem.params.admm.objective_scale))
        nl_dir = os.path.join(scratch, 'nl')
        os.makedirs(nl_dir)
        cap = Capture(cfg, planning, nl_dir)
        planning.transmission_network.optimize = cap.network_stub('TSO')
        for node_id, dn in planning.distribution_networks.items():
            dn.optimize = cap.network_stub(f'DSO{node_id}')
        sed.optimize = cap.esso_stub
        stopped = False
        try:
            planning.run_operational_planning(type='distributed', candidate_solution=deepcopy(candidate),
                                              print_results=False, debug_flag=False, return_state=True)
        except _StopCapture as stop:
            stopped = True
            out['stopped_by'] = str(stop)
        finally:
            srp._compute_common_admm_objective_scale = original_scale
        n_dso = len(planning.distribution_networks)
        expected_calls = {'TSO': 2, 'ESSO': 2, **{f'DSO{n}': 2 for n in planning.distribution_networks}}
        n_net_blocks = len(planning.years) * len(planning.days)
        out['stub_calls'] = dict(cap.calls)
        out['nl'] = cap.nl
        out['checks'] = cap.checks
        out['details'] = cap.details
        out['filled_vars'] = cap.filled
        out['coupling_active'] = cap.coupling_active
        n_checked = {s: len(cap.checks[s]) for s in STAGES}
        out['build_checks'] = {
            'stopped_by_esso_cycle1_sentinel': stopped,
            'stub_calls_exact': cap.calls == expected_calls,
            'every_block_checked_both_stages': all(
                n_checked[s] == n_net_blocks * (1 + n_dso) + len(sed.active_distribution_network_nodes)
                for s in STAGES),
            'nl_blocks_as_declared': all(
                set(cap.nl[s]) == (set(cap.checks[s]) if cfg['nl_blocks'] == 'all' else set(cfg['nl_blocks']))
                for s in STAGES),
            'no_nl_write_error': all(r.get('error') is None for s in STAGES for r in cap.nl[s].values()),
            'labelled_equals_plain_after_stripping': all(r.get('stripped_equal') is True for s in STAGES
                                                         for r in cap.nl[s].values()),
            'scale_replacement_restored': srp._compute_common_admm_objective_scale is original_scale,
        }
    except Exception as error:  # noqa: BLE001 -- recorded; the parent fails the gate
        out['error'] = f'{type(error).__name__}: {error}'
        out['traceback'] = traceback.format_exc()
    out['guard'] = guard_report()
    out['wall_s'] = time.time() - t0
    out['finished_utc'] = _utc()
    _write_once_json(args.child_out, out)
    return 0 if out['error'] is None else 1


def guard_report():
    failures = GUARD.verify(0)
    return {'verified_zero': not failures, 'failures': failures, 'counts': dict(GUARD.counts),
            'pickle': {k: (sorted(v) if isinstance(v, set) else v) for k, v in PICKLE.items()}}


# ======================================================================================================================
#  build orchestration and comparison (G-a, G-b, G-c)
# ======================================================================================================================
def build(args):
    spec = _find_spec(args.spec_sha256)
    label = args.label
    configs = [c for c, v in BUILD_CONFIGS.items() if label in v['labels']]
    if args.configs:
        configs = [c for c in configs if c in args.configs.split(',')]
    out_dir = os.path.join(_gate_dir(), f'builds_{label}')
    os.makedirs(out_dir, exist_ok=True)
    scratch_root = os.path.abspath(args.scratch)
    results = {}
    for config in configs:
        child_out = os.path.join(out_dir, f'{config}.json')
        if os.path.exists(child_out):
            raise RuntimeError(f'refusing to overwrite {child_out}')
        log_path = os.path.join(out_dir, f'{config}.log')
        cmd = [sys.executable, '-u', SCRIPT_NAME, '--child-build', '--config', config, '--label', label,
               '--child-out', child_out, '--scratch', os.path.join(scratch_root, f'{label}_{config}')]
        _log(f'build {label}/{config}: {" ".join(cmd[2:])}')
        t0 = time.time()
        with open(log_path, 'x') as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        results[config] = {'exit_code': rc, 'wall_s': time.time() - t0, 'json': _rel(child_out),
                           'log': _rel(log_path)}
        _log(f'build {label}/{config}: exit {rc} in {results[config]["wall_s"]:.0f} s')
    summary = {'schema': 'p515_s56_coupling_build_batch_v1', 'label': label, 'spec': _rel(spec['_path']),
               'spec_sha256': spec['_sha256'], 'configs': results, 'provenance': provenance(),
               'parent_guard': guard_report()}
    _write_once_json(os.path.join(out_dir, f'batch_{label}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json'),
                     summary)
    return 0 if all(r['exit_code'] == 0 for r in results.values()) else 1


def _child(label, config):
    path = os.path.join(_gate_dir(), f'builds_{label}', f'{config}.json')
    return _load(path) if os.path.isfile(path) else None


def _nl_equal(a, b):
    """Per stage and key: every NL hash equal."""
    diffs = []
    for stage in STAGES:
        ka, kb = set((a.get('nl') or {}).get(stage, {})), set((b.get('nl') or {}).get(stage, {}))
        if ka != kb:
            diffs.append({'stage': stage, 'what': 'block sets differ', 'only_a': sorted(ka - kb)[:5],
                          'only_b': sorted(kb - ka)[:5]})
        for key in sorted(ka & kb):
            ra, rb = a['nl'][stage][key], b['nl'][stage][key]
            for part in ('plain', 'sym_nl', 'sym_row', 'sym_col'):
                if (ra.get(part) or {}).get('sha256') != (rb.get(part) or {}).get('sha256'):
                    diffs.append({'stage': stage, 'block': key, 'part': part,
                                  'a': ra.get(part), 'b': rb.get(part)})
    n = sum(len((a.get('nl') or {}).get(s, {})) for s in STAGES)
    return {'identical': not diffs and n > 0, 'n_nl_compared': n, 'n_differences': len(diffs),
            'first_differences': diffs[:10]}


def _child_ok(rec):
    if rec is None:
        return {'present': False}
    bc = rec.get('build_checks') or {}
    return {'present': True, 'error': rec.get('error'), 'build_checks_all': bool(bc) and all(bc.values()),
            'guard_verified_zero': (rec.get('guard') or {}).get('verified_zero') is True,
            'build_checks': bc}


def _checks_all(rec, gate_filter=None):
    failed = []
    for stage in STAGES:
        for key, ok in ((rec or {}).get('checks') or {}).get(stage, {}).items():
            for name, val in ok.items():
                if val is not True:
                    failed.append(f'{stage}|{key}|{name}')
    return failed


def compare_builds(args):
    spec = _find_spec(args.spec_sha256)
    res = {'schema': 'p515_s56_coupling_build_verdicts_v1', 'spec': _rel(spec['_path']), 'spec_sha256': spec['_sha256'],
           'utc': _utc(), 'predictions': PREDICTIONS}
    # G-a
    ga = {}
    for config, v in BUILD_CONFIGS.items():
        if 'G-a' not in v['gates']:
            continue
        pre, post = _child('pre', config), _child('post', config)
        cmp = _nl_equal(pre, post) if (pre and post) else {'identical': False, 'error': 'missing build'}
        ga[config] = {'pre': _child_ok(pre), 'post': _child_ok(post), 'nl': cmp,
                      'off_structural_checks_failed_post': _checks_all(post)[:10],
                      'holds': (cmp.get('identical') is True and _child_ok(pre).get('build_checks_all') is True
                                and _child_ok(post).get('build_checks_all') is True
                                and _child_ok(pre).get('guard_verified_zero') and _child_ok(post).get(
                                    'guard_verified_zero') and not _checks_all(post))}
    res['G-a'] = {'per_config': ga, 'holds': bool(ga) and all(x['holds'] for x in ga.values())}
    # G-b
    on, off_post, off_pre = _child('on', 'srp1_x0_on'), _child('post', 'srp1_x0'), _child('pre', 'srp1_x0')
    gb_post = _nl_equal(off_post, on) if (on and off_post) else {'identical': False, 'error': 'missing build'}
    gb_pre = _nl_equal(off_pre, on) if (on and off_pre) else {'identical': False, 'error': 'missing build'}
    failed_b = _checks_all(on)
    res['G-b'] = {'on_build': _child_ok(on), 'nl_vs_off_post': gb_post, 'nl_vs_off_pre': gb_pre,
                  'coupling_active_in_child': (on or {}).get('coupling_active'),
                  'option_in_force': (on or {}).get('admm_in_force'),
                  'structural_checks_failed': failed_b[:10],
                  'holds': (gb_post.get('identical') is True and gb_pre.get('identical') is True and not failed_b
                            and _child_ok(on).get('build_checks_all') is True
                            and (on or {}).get('coupling_active') is False)}
    # G-c
    gc_ = {}
    for config in ('srp1grid_x0_a05_on_pin', 'srp1grid_x0_a05_on_v1_noq'):
        rec = _child('on', config)
        failed = _checks_all(rec)
        n_checks = sum(len(ok) for s in STAGES for ok in ((rec or {}).get('checks') or {}).get(s, {}).values())
        gc_[config] = {'build': _child_ok(rec), 'coupling_active_in_child': (rec or {}).get('coupling_active'),
                       'option_in_force': (rec or {}).get('admm_in_force'), 'n_checks': n_checks,
                       'n_failed': len(failed), 'failed_first': failed[:15],
                       'tso_param_counts_sample': ((rec or {}).get('details') or {}).get('admm_cycle1', {}).get(
                           'TSO|2025|Spring'),
                       'holds': (rec is not None and _child_ok(rec).get('build_checks_all') is True
                                 and rec.get('coupling_active') is True and n_checks > 0 and not failed)}
    res['G-c'] = {'per_config': gc_, 'holds': all(x['holds'] for x in gc_.values())}
    res['parent_guard'] = guard_report()
    path = os.path.join(_gate_dir(), f'verdict_builds_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    _write_once_json(path, res)
    _log(f"G-a {res['G-a']['holds']} | G-b {res['G-b']['holds']} | G-c {res['G-c']['holds']} -> {_rel(path)}")
    return 0 if (res['G-a']['holds'] and res['G-b']['holds'] and res['G-c']['holds']) else 2


# ======================================================================================================================
#  G-e fixtures
# ======================================================================================================================
def declared_fixture_list():
    tracked = [p for p in _git(['ls-files', '--', '*.pkl']).splitlines() if p]
    globbed = []
    for root, _dirs, files in os.walk(_abs(FROZEN_SMOPF_GLOB_DIR)):
        for f in sorted(files):
            if f.endswith('.pkl'):
                globbed.append(_rel(os.path.join(root, f)))
    return sorted(set(tracked) | set(NAMED_FIXTURES) | set(globbed)), set(tracked)


def fixtures(args):
    spec = _find_spec(args.spec_sha256)
    import shared_resources_planning  # noqa: F401 -- the modules the fixtures resolve names in, loaded first
    import model_construction_helpers  # noqa: F401
    declared, tracked = declared_fixture_list()
    absent = [r for r in declared if not os.path.isfile(_abs(r))]
    PICKLE['permitted_paths'] = {os.path.abspath(_abs(r)) for r in declared}
    loaded, failed = [], []
    before = PICKLE['load']
    try:
        for rel in declared:
            if not os.path.isfile(_abs(rel)):
                continue
            try:
                with open(_abs(rel), 'rb') as handle:
                    payload = pickle.load(handle)
                loaded.append({'path': rel, 'sha256': _sha_file(_abs(rel)), 'type': type(payload).__name__,
                               'tracked': rel in tracked})
                del payload
            except Exception as error:  # noqa: BLE001
                failed.append({'path': rel, 'error': f'{type(error).__name__}: {error}'})
            gc.collect()
    finally:
        PICKLE['permitted_paths'] = None
    n_loads = PICKLE['load'] - before
    checks = {'no_declared_fixture_absent': not absent,
              'every_declared_fixture_unpickles': not failed and len(loaded) == len(declared),
              'pickle_load_count_equals_declared': n_loads == len(declared),
              'named_anchors_included': all(r in declared for r in NAMED_FIXTURES),
              'no_pickle_loads_call': PICKLE['loads'] == 0}
    res = {'schema': 'p515_s56_coupling_ge_v1', 'spec': _rel(spec['_path']), 'spec_sha256': spec['_sha256'],
           'utc': _utc(), 'provenance': provenance(), 'n_declared': len(declared), 'n_tracked': len(tracked),
           'absent': absent, 'failed': failed, 'loaded': loaded, 'checks': checks, 'holds': all(checks.values()),
           'parent_guard': guard_report()}
    path = os.path.join(_gate_dir(), f'verdict_ge_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    _write_once_json(path, res)
    _log(f"G-e holds {res['holds']} ({len(loaded)}/{len(declared)} loaded) -> {_rel(path)}")
    return 0 if res['holds'] else 2


# ======================================================================================================================
#  G-d / G-f -- harness runs, and the bitwise comparator (from p515_s56_concurrency_gate.py, copied)
# ======================================================================================================================
def _leaf_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack('<d', a) == struct.pack('<d', b)
    return a == b


def _show(v):
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return 'str:' + v[:200]
    return 'json:' + json.dumps(v, sort_keys=True, default=str)[:200]


def _norm_path(path):
    return re.sub(r'\[\d+\]', '[*]', path)


def _is_excluded(path, excluded):
    p = _norm_path(path)
    return any(p == ex or p.startswith(ex + '.') or p.startswith(ex + '[') for ex in excluded)


def deep_diff(a, b, excluded=()):
    out, hits = [], []

    def rec(x, y, p):
        sp = p[1:] if p.startswith('.') else p
        if excluded and _is_excluded(sp, excluded):
            if json.dumps(x, sort_keys=True, default=str) != json.dumps(y, sort_keys=True, default=str):
                hits.append(_norm_path(sp))
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in list(x.keys()) + [k for k in y.keys() if k not in x]:
                cp = f'{p}.{k}'
                if k not in x or k not in y:
                    out.append((cp[1:] if cp.startswith('.') else cp, 'MISSING' if k not in x else _show(x[k]),
                                'MISSING' if k not in y else _show(y[k])))
                    continue
                rec(x[k], y[k], cp)
        elif isinstance(x, list) and isinstance(y, list):
            for i in range(min(len(x), len(y))):
                rec(x[i], y[i], f'{p}[{i}]')
            if len(x) != len(y):
                out.append(((sp + '.' if sp else '') + '__len__', len(x), len(y)))
        elif not _leaf_equal(x, y):
            out.append((sp, _show(x), _show(y)))

    rec(a, b, '')
    return out, hits


def per_cycle_files(eval_dir):
    g = _load(os.path.join(eval_dir, 'g_s39_D.json'))
    return {'g_s39_D.json#cycle_trajectory': g['cycle_trajectory'],
            'per_cycle_record.jsonl': _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl')),
            'aa_per_cycle.jsonl': _read_jsonl(os.path.join(eval_dir, 'aa_per_cycle.jsonl'))}


def ipopt_final_blocks(path):
    blocks, exits, starts, cur = [], [], 0, None
    with open(path, 'rb') as handle:
        for raw in handle:
            line = raw.decode('latin-1').rstrip('\r\n')
            if line.startswith('This is Ipopt version'):
                starts += 1
            if line.startswith('EXIT:'):
                exits.append(line)
            if line.startswith(IPOPT_FINAL_START):
                cur = [line]
                continue
            if cur is not None:
                if line.startswith('EXIT:'):
                    cur.append(line)
                    blocks.append(cur)
                    cur = None
                elif line.strip() and not line.startswith(IPOPT_TIMING_PREFIXES):
                    cur.append(line)
    return {'n_solve_starts': starts, 'exit_lines': exits, 'final_blocks': blocks, 'unterminated': cur is not None}


def compare_ipopt_logs(logs_a, logs_b):
    fa = sorted(os.listdir(logs_a)) if os.path.isdir(logs_a) else None
    fb = sorted(os.listdir(logs_b)) if os.path.isdir(logs_b) else None
    out = {'logs_a': _rel(logs_a), 'logs_b': _rel(logs_b), 'files_a': None if fa is None else len(fa),
           'files_b': None if fb is None else len(fb), 'per_file_differences': [], 'n_solves_compared': 0}
    if fa is None or fb is None:
        out['identical'] = False
        return out
    for name in sorted(set(fa) & set(fb)):
        pa, pb = ipopt_final_blocks(os.path.join(logs_a, name)), ipopt_final_blocks(os.path.join(logs_b, name))
        out['n_solves_compared'] += len(pa['final_blocks'])
        diff = [{'what': k} for k in ('n_solve_starts', 'exit_lines', 'unterminated') if pa[k] != pb[k]]
        if len(pa['final_blocks']) != len(pb['final_blocks']):
            diff.append({'what': 'n_final_blocks', 'A': len(pa['final_blocks']), 'B': len(pb['final_blocks'])})
        for i, (ba, bb) in enumerate(zip(pa['final_blocks'], pb['final_blocks'])):
            if ba != bb:
                j = next((k for k in range(min(len(ba), len(bb))) if ba[k] != bb[k]), min(len(ba), len(bb)))
                diff.append({'what': 'final_block', 'solve_index_in_file': i,
                             'A': ba[j] if j < len(ba) else None, 'B': bb[j] if j < len(bb) else None})
                break
        if diff:
            out['per_file_differences'].append({'file': name, 'differences': diff})
    for name in sorted(set(fa) ^ set(fb)):
        out['per_file_differences'].append({'file': name, 'differences': [{'what': 'present in one only'}]})
    out['identical'] = fa == fb and not out['per_file_differences'] and out['n_solves_compared'] > 0
    return out


def compare_eval_dirs(dir_a, dir_b):
    res = {'dir_a': _rel(dir_a), 'dir_b': _rel(dir_b)}
    pa, pb = per_cycle_files(dir_a), per_cycle_files(dir_b)
    g1 = {}
    for name in G1_FILES:
        d, ex = deep_diff(pa[name], pb[name])
        g1[name] = {'identical': not d, 'n_differences': len(d), 'first_differences': d[:15],
                    'n_rows_a': len(pa[name]), 'n_rows_b': len(pb[name])}
    res['G1'] = {'files': g1, 'holds': all(v['identical'] for v in g1.values())}
    ra, rb = _read_jsonl(os.path.join(dir_a, G2_RECORDS_FILE)), _read_jsonl(os.path.join(dir_b, G2_RECORDS_FILE))
    d, ex = deep_diff(ra, rb, excluded=tuple(G2_RECORDS_EXCLUDED))
    rec_a, rec_b = _load(os.path.join(dir_a, 'evaluation_record.json')), _load(os.path.join(dir_b,
                                                                                           'evaluation_record.json'))
    logs = compare_ipopt_logs(_abs(os.path.join(P56A_WORK_DIR_REL, rec_a['working_dir_ids']['run'], 'logs')),
                              _abs(os.path.join(P56A_WORK_DIR_REL, rec_b['working_dir_ids']['run'], 'logs')))
    res['G2'] = {'records': {'identical': not d, 'n_records_a': len(ra), 'n_records_b': len(rb),
                             'first_differences': d[:15], 'excluded_fields_that_differed': sorted(set(ex))},
                 'ipopt_logs': logs, 'holds': (not d) and len(ra) > 0 and logs['identical']}
    d3, _ = deep_diff(rec_a, rec_b)
    res['supplementary_evaluation_record_not_gated'] = {'n_differences': len(d3),
                                                        'differing_paths': sorted({_norm_path(x[0]) for x in d3})[:60]}
    return res


def comparator_selftest():
    out = {}
    try:
        same = compare_eval_dirs(_abs(SELFTEST['same_A']), _abs(SELFTEST['same_B']))
        other = compare_eval_dirs(_abs(SELFTEST['same_B']), _abs(SELFTEST['other_B']))
        out = {'same_G1': same['G1']['holds'], 'same_G2': same['G2']['holds'],
               'other_G1_differs': not other['G1']['holds'], 'other_G2_differs': not other['G2']['holds'],
               'same_n_solves_compared': same['G2']['ipopt_logs']['n_solves_compared']}
        out['all_hold'] = all(v for k, v in out.items() if k != 'same_n_solves_compared') \
            and out['same_n_solves_compared'] > 0
    except Exception as error:  # noqa: BLE001
        out = {'error': f'{type(error).__name__}: {error}', 'all_hold': False}
    return out


def _vm_stat():
    try:
        return subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        return f'unavailable: {error}'


def _run_campaign(spec, root_name, labelled_candidates, configuration, cap, tag):
    root = os.path.join(_gate_dir(), f'campaign_{root_name}')
    failures = H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT_NAME, _rel(spec['_path'])))
    if failures:
        return {'launched': False, 'failures': failures}
    spec_path, spec_sha, cspec = H.freeze_campaign_spec(
        root, root_name, labelled_candidates, configuration, cap, 1, AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra={'gate_spec': {'path': _rel(spec['_path']), 'sha256': spec['_sha256']}, 'role': tag,
               'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha_file(_abs(SCRIPT_NAME)),
               'objective_convention': 'Q = gross_operational_cost (contracted settlement and pin excluded)'})
    lock = H.acquire_campaign_lock(cspec['campaign_id'], spec_sha)
    _log(f'[{tag}] lock {lock}; spec {_rel(spec_path)}; cap {cap}')
    t0 = time.time()
    records, err = None, None
    vm_before = _vm_stat()
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha, cspec, log=_log)
        records = H.evaluate([c[0] for c in labelled_candidates], ctx)
    except BaseException as error:  # noqa: BLE001 -- a harness fault, recorded
        err = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    finally:
        released = H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] lock released: {released}')
    per = {}
    for e in cspec['candidates']:
        ed = os.path.join(root, 'evals', e['eval_dir'])
        rec = next((x for x in (records or []) if (x or {}).get('candidate_label') == e['label']), None) or {}
        per[e['label']] = {'eval_dir': _rel(ed), 'eval_key': e['eval_key'], 'working_dir_run': e['working_dir_ids']['run'],
                           'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                           'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
                           'solve_profile': rec.get('solve_profile'), 'peak_rss': rec.get('peak_rss'),
                           'wall_time_s': rec.get('wall_time_s'),
                           'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost')}
    return {'launched': True, 'campaign_root': _rel(root), 'campaign_spec': {'path': _rel(spec_path), 'sha256': spec_sha},
            'entries': cspec['candidates'], 'harness_error': err, 'per_evaluation': per, 'wall_s': time.time() - t0,
            'vm_stat_before': vm_before, 'provenance': provenance()}


def run_gd(args):
    spec = _find_spec(args.spec_sha256)
    label = args.label
    res = _run_campaign(spec, GD_ROOTS[label], [(GD_LABEL, UNIT, {})], CONFIGURATION_SRP1, GD_CAP, f'G-d {label}')
    res['parent_guard'] = guard_report()
    path = os.path.join(_gate_dir(), f'gd_run_{label}.json')
    _write_once_json(path, res)
    p = (res.get('per_evaluation') or {}).get(GD_LABEL) or {}
    ok = res.get('launched') and not res.get('harness_error') and p.get('exit_code') == 0 and p.get('cycles_run') == GD_CAP
    _log(f'G-d {label}: launched {res.get("launched")} exit {p.get("exit_code")} cycles {p.get("cycles_run")} -> '
         f'{_rel(path)}')
    return 0 if ok else 1


def compare_gd(args):
    spec = _find_spec(args.spec_sha256)
    pre, post = _load(os.path.join(_gate_dir(), 'gd_run_pre.json')), _load(os.path.join(_gate_dir(), 'gd_run_post.json'))
    a = _abs(pre['per_evaluation'][GD_LABEL]['eval_dir'])
    b = _abs(post['per_evaluation'][GD_LABEL]['eval_dir'])
    cmp = compare_eval_dirs(a, b)
    same_key = pre['per_evaluation'][GD_LABEL]['eval_key'] == post['per_evaluation'][GD_LABEL]['eval_key']
    res = {'schema': 'p515_s56_coupling_gd_v1', 'spec': _rel(spec['_path']), 'spec_sha256': spec['_sha256'],
           'utc': _utc(), 'pre_head': pre['provenance']['git_head'], 'post_head': post['provenance']['git_head'],
           'eval_key_identical': same_key, 'comparison': cmp,
           'cycles': {'pre': pre['per_evaluation'][GD_LABEL]['cycles_run'],
                      'post': post['per_evaluation'][GD_LABEL]['cycles_run']},
           'holds': cmp['G1']['holds'] and cmp['G2']['holds'] and same_key, 'parent_guard': guard_report()}
    path = os.path.join(_gate_dir(), f'verdict_gd_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    _write_once_json(path, res)
    _log(f"G-d holds {res['holds']} (G1 {cmp['G1']['holds']}, G2 {cmp['G2']['holds']}) -> {_rel(path)}")
    return 0 if res['holds'] else 2


def smoke(args):
    spec = _find_spec(args.spec_sha256)
    cfg = deepcopy(CONFIGURATION_SRP1)
    cfg['name'] = ('S56 step 1 G-f smoke: SRP1-grid 3 x 3, per-scenario interface coupling ON (V-1, Q charge off), '
                   'row 18 alpha 0.5; otherwise the published configuration')
    cfg['derived_instance'] = derived_declaration('srp1grid')
    res = _run_campaign(spec, GF_ROOT, [(GF_LABEL, X0, deepcopy(GF_OPTIONS))], cfg, GF_CAP, 'G-f smoke')
    p = (res.get('per_evaluation') or {}).get(GF_LABEL) or {}
    checks = {'launched': bool(res.get('launched')), 'no_harness_error': not res.get('harness_error'),
              'exit_0': p.get('exit_code') == 0, 'two_cycles': p.get('cycles_run') == GF_CAP}
    rows = []
    if p.get('eval_dir') and os.path.isfile(_abs(os.path.join(p['eval_dir'], 'per_cycle_record.jsonl'))):
        rows = _read_jsonl(_abs(os.path.join(p['eval_dir'], 'per_cycle_record.jsonl')))
    missing, nonfinite = [], []
    for r in rows:
        for f in PER_CYCLE_COUPLING_FIELDS:
            if f not in r:
                missing.append((r.get('cycle'), f))
                continue
            v = r[f]
            if f.startswith('boyd_v_') and r.get('interface_coupling_voltage_mode') == 'pin':
                continue
            if isinstance(v, float) and not (v == v and abs(v) != float('inf')):
                nonfinite.append((r.get('cycle'), f))
            if v is None:
                nonfinite.append((r.get('cycle'), f, 'None'))
        for f in ('boyd_pf_primal_ratio', 'boyd_pf_dual_ratio', 'boyd_v_primal_ratio', 'boyd_ess_primal_ratio'):
            v = r.get(f)
            if not isinstance(v, float) or v != v or abs(v) == float('inf'):
                nonfinite.append((r.get('cycle'), f))
    checks['per_cycle_rows_2'] = len(rows) == GF_CAP
    checks['every_new_field_present'] = bool(rows) and not missing
    checks['every_new_field_and_boyd_ratio_finite'] = bool(rows) and not nonfinite
    checks['coupling_active_in_rows'] = bool(rows) and all(r.get('interface_coupling_active') is True for r in rows)
    res['checks'] = checks
    res['missing_first'] = missing[:10]
    res['nonfinite_first'] = [list(x) for x in nonfinite[:10]]
    res['rows'] = rows
    res['holds'] = all(checks.values())
    res['parent_guard'] = guard_report()
    path = os.path.join(_gate_dir(), f'verdict_gf_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    _write_once_json(path, res)
    _log(f"G-f holds {res['holds']} {checks} -> {_rel(path)}")
    return 0 if res['holds'] else 2


# ======================================================================================================================
#  freeze
# ======================================================================================================================
def capture_path_checklist():
    import inspect
    import p515_g_g1_g4_admm_gates as G
    checks = {
        'construct_arm_planning_exists': callable(getattr(G, '_construct_arm_planning', None)),
        'config_hook_exists': callable(getattr(H, '_config_hook_factory', None)),
        'install_derived_instance_exists': callable(getattr(H, 'install_derived_instance', None)),
        'harness_campaign_api': all(callable(getattr(H, n, None)) for n in (
            'freeze_campaign_spec', 'check_campaign_preconditions', 'acquire_campaign_lock', 'evaluate',
            'release_campaign_lock')),
        'init_failed_branch_and_scale_call_in_production_loop': all(s in inspect.getsource(
            __import__('shared_resources_planning')._run_operational_planning) for s in (
            '_compute_common_admm_objective_scale(planning_problem, tso_model, dso_models)',
            '_resolve_common_admm_objective_scale(objective_scale_computed, admm_parameters)')),
        'derived_declarations_validate': all(derived_declaration(i) is not None for i in DERIVED_SOURCES),
    }
    return checks


def freeze(_args):
    gate_dir = _gate_dir()
    if os.path.exists(gate_dir):
        _log(f'REFUSING: {GATE_DIR_REL} exists')
        return 1
    checklist = capture_path_checklist()
    selftest = comparator_selftest()
    derived = {i: {'source': DERIVED_SOURCES[i], 'source_sha256': _sha_file(_abs(DERIVED_SOURCES[i]['path'])),
                   'declaration': derived_declaration(i)} for i in DERIVED_SOURCES}
    spec = {
        'schema': 'p515_s56_coupling_gates_spec_v1', 'stage': STAGE, 'authority': AUTHORITY, 'frozen_utc': _utc(),
        'predecessor': None, 'provenance': provenance(), 'script': {'path': SCRIPT_NAME,
                                                                    'sha256': _sha_file(_abs(SCRIPT_NAME))},
        'option_under_test': {'admm_params_attributes': COUPLING_DEFAULT, 'harness_entry_option': 'interface_coupling',
                              'new_names_read_by_G_c': {'params': DEV_PARAM, 'wired_marker': WIRED_MARKER},
                              'per_cycle_fields': list(PER_CYCLE_COUPLING_FIELDS)},
        'build_configs': BUILD_CONFIGS, 'stages': list(STAGES), 'derived_instances': derived,
        'build_mechanism': {
            'stubs': ('NetworkData.optimize (TSO, every DSO) and the ESSO optimize replaced on the instance; call 1 = '
                      'initialisation (NL / checks, None-valued Vars -> 0 clipped to bounds, SolverResults ok/optimal), '
                      'call 2 = ADMM cycle 1 (NL / checks; ESSO raises the sentinel)'),
            'replaced_production_function': ('_compute_common_admm_objective_scale -> the fixed sigma '
                                              f'{FIXED_SIGMA} (asserted equal to the case file); the ADMM objectives '
                                              'use the fixed sigma either way'),
            'nl_forms': 'plain (symbolic_solver_labels False, the production form) + labelled (.nl, .row, .col)'},
        'G-d': {'candidate': GD_LABEL, 'cap': GD_CAP, 'concurrency': 1, 'roots': GD_ROOTS,
                'configuration': CONFIGURATION_SRP1, 'compared': {'G1': list(G1_FILES), 'G2_records': G2_RECORDS_FILE,
                                                                  'G2_ipopt': 'final-statistics blocks'},
                'excluded_in_advance': {'G2_records': G2_RECORDS_EXCLUDED, 'G2_ipopt_timing_line_prefixes':
                                        list(IPOPT_TIMING_PREFIXES),
                                        'G3_evaluation_record': 'NOT gated (supplementary: list of differing paths)'},
                'comparator_selftest_on_committed_artifacts': selftest},
        'G-e': {'named_fixtures': list(NAMED_FIXTURES), 'plus': 'every tracked .pkl + FrozenSMOPF/**'},
        'G-f': {'instance': 'srp1grid', 'candidate': GF_LABEL, 'options': GF_OPTIONS, 'cap': GF_CAP,
                'wall_projection': GF_WALL_LIMIT_NOTE},
        'predictions': PREDICTIONS, 'failure_policy': FAILURE_POLICY,
        'capture_path_checklist': checklist,
        'order': ['freeze + commit (spec, this script)', 'G-a pre builds', 'G-d pre run', 'production change',
                  'G-a post builds', 'G-b / G-c builds (label on)', 'compare builds', 'G-e', 'commit the change',
                  'G-d post run', 'compare G-d', 'G-f smoke'],
        'commands': {
            'build': (f'NLP_SOLVER_PATH={SOLVER_PATH} {CANONICAL_PYTHON} -u {SCRIPT_NAME} --build --label <pre|post|on> '
                      f'--spec-sha256 <S> --scratch <scratchpad>/s56cpl > {GATE_DIR_REL}/build_<label>_launch.log 2>&1'),
            'compare_builds': f'... {SCRIPT_NAME} --compare-builds --spec-sha256 <S>',
            'fixtures': f'... {SCRIPT_NAME} --fixtures --spec-sha256 <S>',
            'run_gd': f'... {SCRIPT_NAME} --run-gd --label <pre|post> --spec-sha256 <S>',
            'compare_gd': f'... {SCRIPT_NAME} --compare-gd --spec-sha256 <S>',
            'smoke': f'... {SCRIPT_NAME} --smoke --spec-sha256 <S>',
            'attached': 'attached, alone, both streams captured, never detached'},
    }
    ok = all(checklist.values()) and selftest.get('all_hold') is True
    spec['freeze_checks_pass'] = ok
    os.makedirs(gate_dir)
    text = GRIO.dumps(spec, indent=1, sort_keys=True, default=GRIO.json_default)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(gate_dir, f'{SPEC_PREFIX}{digest[:8]}.json')
    with open(path, 'x') as handle:
        handle.write(text)
    _log(f'frozen spec {_rel(path)} sha256 {digest}; checklist {checklist}; selftest {selftest}')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ('--freeze', '--build', '--compare-builds', '--fixtures', '--run-gd', '--compare-gd', '--smoke',
                 '--child-build'):
        m.add_argument(flag, action='store_true')
    ap.add_argument('--label')
    ap.add_argument('--spec-sha256')
    ap.add_argument('--scratch')
    ap.add_argument('--configs')
    ap.add_argument('--config')
    ap.add_argument('--child-out')
    args = ap.parse_args()
    if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
        _log(f'REFUSING: NLP_SOLVER_PATH must be {SOLVER_PATH}')
        return 1
    if args.child_build:
        return child_build(args)
    if args.freeze:
        return freeze(args)
    if args.build:
        return build(args)
    if args.compare_builds:
        return compare_builds(args)
    if args.fixtures:
        return fixtures(args)
    if args.run_gd:
        return run_gd(args)
    if args.compare_gd:
        return compare_gd(args)
    if args.smoke:
        return smoke(args)
    return 1


if __name__ == '__main__':
    sys.exit(main())
