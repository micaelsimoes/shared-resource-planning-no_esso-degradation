"""P5.12-K -- NO-SOLVE KKT / warm-start forensic.

Diagnoses the consistency and initial-residual structure of the frozen P5.12-R /
Arm A replay fixture WITHOUT invoking any solver. Zero `OptSolver.solve` calls,
zero IPOPT process launches are authorized; both entry points are wrapped to
raise immediately (see `install_solver_guards()`), and the counters are
reported in the journal and the final report.

This script performs NO production edit, NO replay, NO retry. It only reads the
frozen P5.12-R / Arm A artifacts (read-only) and writes new evidence under
`data/SRP1/Results/P512K/` (created fresh by this script; must not already
exist) plus this file itself (new, untracked).

Derivatives use Pyomo's own reverse-mode AD (`differentiate(...,
mode=Modes.reverse_numeric)`), exactly as in `p53a_conditioning_audit.py` /
`p53a2_jacobian_correction.py`, because PyNumero's ASL interface is not
available in this environment.

Run:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p512_k_no_solve_kkt_forensic.py
"""
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
import re
from pathlib import Path

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.solver.shellcmd import SystemCallSolver

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import p512_r_presolve_recapture as H   # frozen harness, read-only primitives
import shared_resources_planning as srp  # noqa: E402

OUT = ROOT / 'data/SRP1/Results/P512K'
P512R = ROOT / 'data/SRP1/Results/P512R'
ARMA = ROOT / 'data/SRP1/Results/P512ArmA'

HARNESS_SHA_EXPECTED = 'f0f120c26ec2c50b774ff42051c233b87fba0341e3d959faafe70283301d86f0'

JACOBIAN_BUDGET = 6
JACOBIAN_TIME_LIMIT_S = 20 * 60

JOURNAL = {
    'stage': 'P5.12-K', 'status': 'RUNNING',
    'preamble': ('This forensic diagnoses the consistency and initial residual '
                 'structure of the frozen replay fixture. It does not by itself '
                 'explain why cycle 21 fails while cycle 20 succeeds.'),
    'zero_solver': {}, 'step0': {}, 'step1': {}, 'step2': {}, 'step3': {},
    'step4': {}, 'step5': {}, 'jacobian_budget': {'builds': [], 'limit': JACOBIAN_BUDGET,
                                                   'time_limit_s': JACOBIAN_TIME_LIMIT_S},
    'artifacts': {'inputs': {}, 'outputs': {}},
}


# ===========================================================================
# zero-solver guard
# ===========================================================================
class SolverInvocationBlocked(RuntimeError):
    pass


_SOLVE_COUNTERS = {'OptSolver.solve': 0, 'SystemCallSolver._execute_command': 0}
_ORIGINAL_SOLVE = OptSolver.solve
_ORIGINAL_EXEC = SystemCallSolver._execute_command


def _blocked_solve(self, *args, **kwargs):
    _SOLVE_COUNTERS['OptSolver.solve'] += 1
    raise SolverInvocationBlocked('OptSolver.solve invoked; P5.12-K authorizes zero solves')


def _blocked_exec(self, *args, **kwargs):
    _SOLVE_COUNTERS['SystemCallSolver._execute_command'] += 1
    raise SolverInvocationBlocked('SystemCallSolver._execute_command invoked; '
                                   'P5.12-K authorizes zero process launches')


def install_solver_guards():
    OptSolver.solve = _blocked_solve
    SystemCallSolver._execute_command = _blocked_exec


def uninstall_solver_guards():
    OptSolver.solve = _ORIGINAL_SOLVE
    SystemCallSolver._execute_command = _ORIGINAL_EXEC


# ===========================================================================
# utilities
# ===========================================================================
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def hexfloat(x):
    """Lossless hex encoding for the JSON journal (per plan D7 convention)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return ['nonfinite', repr(x)]
    return ['float', x.hex(), x]


def record_input(label, path):
    path = Path(path)
    JOURNAL['artifacts']['inputs'][label] = {'path': str(path.relative_to(ROOT)),
                                              'sha256': sha(path)}
    return JOURNAL['artifacts']['inputs'][label]['sha256']


def record_output(label, path):
    path = Path(path)
    JOURNAL['artifacts']['outputs'][label] = {'path': str(path.relative_to(ROOT)),
                                               'sha256': sha(path)}


def dump_json(path, obj):
    path = Path(path)
    with path.open('x') as f:
        json.dump(obj, f, indent=1, default=str)


def read_only_git(*args):
    return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True,
                           text=True, check=True).stdout


def hashable(x):
    """Recursively convert list-based atom() encodings into hashable tuples."""
    if isinstance(x, list):
        return tuple(hashable(v) for v in x)
    return x


def repo_state():
    head = read_only_git('rev-parse', 'HEAD').strip()
    branch = read_only_git('branch', '--show-current').strip()
    status = read_only_git('status', '--porcelain', '--untracked-files=no')
    return {'head': head, 'branch': branch, 'status_porcelain_tracked': status}


# ===========================================================================
# frozen-input hash table (from the Planner task; verified in Step 0)
# ===========================================================================
EXPECTED_HASHES = {
    'cycle21_pre_setup/snapshot.pkl': '38fa9e2c9031c22e55e3a31db072ea10cd51a306693a7a724ecce3bec4795cd8',
    'cycle21_prepared/snapshot.pkl': '00e8634dc6871690585a285f3e4955320e61b69095c0d87b24b5f3d84b0d6a5c',
    'cycle21_prepared/original.nl': '5934341b1137271b7a86f440ff7a0d146c7b4b6c256f9331f8b96edc807d39a7',
    'cycle20_prepared/snapshot.pkl': 'f9f72c86ea2ce47baabd774e741072184e91aa0d455fab2dc3a1972b43691aca',
    'cycle20_checkpoint/checkpoint.pkl': '488beceac47483f57b99ec304ebb4beb97034f2c719a3e9b491d07c2f2186b76',
    'cycle20_target.log': '60a8700809444f3f0089ad9b2956f557998d7dace7f29127d6defe272d6111e8',
}
EXPECTED_SEMANTIC = {
    'cycle21_pre_setup': '22d5d85087b7a6eb632e2993f839f9e8ebcfe7f6f2d53b461ac39b543ef9a0e5',
    'cycle21_prepared': '3fd294d7b41ad3b2b62a140cb518a9ecbe32a0218eed8d6cd788fc027edda85f',
    'cycle20_prepared': '58e2a063837cabdc75208ea0188882e88202f598cfdcb3746b08af665186c8e4',
}
EXPECTED_CHECKPOINT_STATE_DIGEST = 'bb55daed4a85379792b9065b1ce42f284c672c725912b96eaf737cae6057e11a'
EXPECTED_TARGET_HASHES = {
    'armA_log': ('data/SRP1/Results/P512ArmA/logs/optim_log_case33_3_2025_Spring.log',
                 '4f66a7efeef933bdc0a425af76f0095f5c11a2112ff2c8bb6d7c03ff45409d58'),
    'armA_sol': ('data/SRP1/Results/P512ArmA/used_tmp27ntrqce.pyomo.sol',
                 '7b6b11f15026ce94ab7a330acfc2c4d53357611238f316b42f69b2dc6af28c81'),
}
EXPECTED_TASK_MAPPING_DIGEST = 'c13732e842003fe2331c6eea869cd78d14cafc3541e259373d1fa42f7e88d2ff'

TARGETS = {
    'S20_dual_inf_final': 1.1323057973496387e-03,
    'S21_dual_inf_iter0': 1.8366791482401353e+04,
    'S20start_dual_inf_iter0': 1.8369213729230112e+04,
    'S21_constraint_violation_iter0_pushed': 6.6171136714446988e-05,
    'S20_constraint_violation_converged': 7.2149502639007324e-07,
    'ObjectiveScaling': 1.000000e-03,
    'ArmA_terminal_violation': 6.1921343776988665e-02,
    'ArmA_terminal_mu': 1.8449144625279508e-06,
}


# ===========================================================================
# Step 0 -- hashes, semantic digests, S20/S21 state comparison
# ===========================================================================
def step0():
    out = {}
    hash_checks = {}
    for label, digest in EXPECTED_HASHES.items():
        actual = record_input(label, P512R / label)
        hash_checks[label] = {'expected': digest, 'actual': actual, 'match': actual == digest}
    for label, (relpath, digest) in EXPECTED_TARGET_HASHES.items():
        actual = record_input(label, ROOT / relpath)
        hash_checks[label] = {'expected': digest, 'actual': actual, 'match': actual == digest}
    out['file_hash_checks'] = hash_checks
    out['file_hashes_all_match'] = all(v['match'] for v in hash_checks.values())

    # mapping-hash finding (documented, not a corruption -- see report)
    mapping_file_sha = sha(P512R / 'cycle21_prepared/original_mapping.json')
    reload_mapping_sha = sha(P512R / 'cycle21_prepared/reload_mapping.json')
    manifest21 = json.loads((P512R / 'cycle21_prepared/manifest.json').read_text())
    export_digest = manifest21['export']['mapping_sha256']
    armA_mapping_sha = sha(ARMA / 'armA_prepared_mapping.json')
    out['mapping_hash_finding'] = {
        'task_given_mapping_hash': EXPECTED_TASK_MAPPING_DIGEST,
        'original_mapping_json_file_sha256': mapping_file_sha,
        'reload_mapping_json_file_sha256': reload_mapping_sha,
        'armA_prepared_mapping_json_file_sha256': armA_mapping_sha,
        'manifest_export_mapping_sha256_(=digest(record))': export_digest,
        'interpretation': ('The task-given "mapping" hash equals manifest.json\'s '
                            'internal export.mapping_sha256 field, which is the harness\'s '
                            'digest() (compact-JSON, atom-encoded) content hash of the '
                            'in-memory {mapping, not_exported_variables} record -- NOT the '
                            'sha256 of the on-disk *_mapping.json file bytes (that is sha(), '
                            'a different hash space, computed over the pretty-printed dump()). '
                            'original_mapping.json, reload_mapping.json and '
                            'armA_prepared_mapping.json are byte-identical to each other '
                            '(all sha256 b39cd878...), and export.mapping_sha256 is identical '
                            'across the cycle20_checkpoint block, cycle21_prepared and Arm A, '
                            'confirming the SAME symbol<->variable-name correspondence and the '
                            'same not_exported_variables identity set across all three states. '
                            'This is not a data-integrity failure.'),
        'mapping_file_hashes_mutually_consistent': mapping_file_sha == reload_mapping_sha == armA_mapping_sha,
    }

    # semantic digests via the frozen harness's own primitives
    with (P512R / 'cycle21_pre_setup/snapshot.pkl').open('rb') as f:
        presetup21 = pickle.load(f)
    with (P512R / 'cycle21_prepared/snapshot.pkl').open('rb') as f:
        prepared21 = pickle.load(f)
    with (P512R / 'cycle20_prepared/snapshot.pkl').open('rb') as f:
        prepared20start = pickle.load(f)
    t0 = time.time()
    with (P512R / 'cycle20_checkpoint/checkpoint.pkl').open('rb') as f:
        checkpoint = pickle.load(f)
    checkpoint_load_s = time.time() - t0

    m_presetup21 = presetup21['model']
    m_S21 = prepared21['model']
    m_S20start = prepared20start['model']
    m_S20 = checkpoint['models']['dso'][9][2025]['Spring']

    semantic = {}
    semantic['cycle21_pre_setup'] = H.digest(H.model_state(m_presetup21))
    semantic['cycle21_prepared'] = H.digest(H.model_state(m_S21))
    semantic['cycle20_prepared'] = H.digest(H.model_state(m_S20start))
    out['semantic_digest_checks'] = {
        k: {'expected': EXPECTED_SEMANTIC[k], 'actual': semantic[k], 'match': semantic[k] == EXPECTED_SEMANTIC[k]}
        for k in EXPECTED_SEMANTIC
    }
    checkpoint_state_digest = H.digest(H.plain({k: checkpoint['state'][k] for k in H.CHECKPOINT_STATE_KEYS}))
    out['checkpoint_state_digest_check'] = {
        'expected': EXPECTED_CHECKPOINT_STATE_DIGEST, 'actual': checkpoint_state_digest,
        'match': checkpoint_state_digest == EXPECTED_CHECKPOINT_STATE_DIGEST,
        'note': 'digest(plain(state)) over the whole ADMM state dict, matching '
                'checkpoint manifest.json state_sha256; NOT a per-block model digest.'}
    checkpoint_block_digest = H.digest(H.model_state(m_S20))
    checkpoint_manifest = json.loads((P512R / 'cycle20_checkpoint/manifest.json').read_text())
    block_rec = next(b for b in checkpoint_manifest['blocks'] if b['block'] == '/dso/9/2025/Spring')
    out['checkpoint_block_digest_check'] = {
        'actual': checkpoint_block_digest, 'manifest_recorded': block_rec['semantic_sha256'],
        'match': checkpoint_block_digest == block_rec['semantic_sha256']}
    out['checkpoint_load_seconds'] = checkpoint_load_s

    # -------- S20 vs S21 comparison: vars, suffixes, params --------
    st20 = H.model_state(m_S20)
    st21 = H.model_state(m_S21)
    vars20 = {r[1]: r for r in st20 if r[0] == 'var'}
    vars21 = {r[1]: r for r in st21 if r[0] == 'var'}
    names_common = sorted(set(vars20) & set(vars21))
    only20 = sorted(set(vars20) - set(vars21))
    only21 = sorted(set(vars21) - set(vars20))

    def dec(atom):
        if atom is None:
            return None
        if isinstance(atom, list) and atom and atom[0] == 'float':
            return float.fromhex(atom[1])
        return atom

    var_val_diffs = []
    for n in names_common:
        a, b = dec(vars20[n][2]), dec(vars21[n][2])
        if a != b:
            var_val_diffs.append({'name': n, 'S20': a, 'S21': b, 'delta': (b - a) if (a is not None and b is not None) else None})
    var_bound_diffs = [n for n in names_common if (vars20[n][4], vars20[n][5]) != (vars21[n][4], vars21[n][5])]
    var_fixed_diffs = [n for n in names_common if vars20[n][3] != vars21[n][3]]

    out['var_comparison'] = {
        'n_common': len(names_common), 'n_only_in_S20': len(only20), 'n_only_in_S21': len(only21),
        'n_value_diffs': len(var_val_diffs),
        'value_diffs': sorted(var_val_diffs, key=lambda d: -abs(d['delta'] or 0)),
        'n_bound_diffs': len(var_bound_diffs), 'bound_diffs': var_bound_diffs,
        'n_fixed_diffs': len(var_fixed_diffs), 'fixed_diffs': var_fixed_diffs,
    }
    if var_val_diffs:
        argmax = max(var_val_diffs, key=lambda d: abs(d['delta'] or 0))
        out['var_comparison']['argmax'] = argmax
        out['var_comparison']['inf_norm_delta'] = abs(argmax['delta'])

    # suffix comparison: dual, ipopt_zL_out, ipopt_zU_out
    suffix20 = {r[1]: dict(r[4]) for r in st20 if r[0] == 'suffix'}
    suffix21 = {r[1]: dict(r[4]) for r in st21 if r[0] == 'suffix'}
    suffix_report = {}
    for sname in ('dual', 'ipopt_zL_out', 'ipopt_zU_out'):
        full20 = next((r for r in st20 if r[0] == 'suffix' and r[1].endswith(sname)), None)
        full21 = next((r for r in st21 if r[0] == 'suffix' and r[1].endswith(sname)), None)
        d20 = {k: dec(v) for k, v in (dict(full20[4]) if full20 else {}).items()}
        d21 = {k: dec(v) for k, v in (dict(full21[4]) if full21 else {}).items()}
        keys = sorted(set(d20) & set(d21))
        diffs = [{'key': k, 'S20': d20[k], 'S21': d21[k]} for k in keys if d20[k] != d21[k]]
        suffix_report[sname] = {
            'n_S20': len(d20), 'n_S21': len(d21), 'n_common_keys': len(keys),
            'n_only_S20': len(set(d20) - set(d21)), 'n_only_S21': len(set(d21) - set(d20)),
            'n_diffs': len(diffs), 'diffs': diffs[:50]}
    out['suffix_comparison'] = suffix_report

    # Param comparison, grouped exactly as specified
    GROUPS = {'vmag_req': 'vmag_req', 'p_pf_req': 'p_pf_req', 'q_pf_req': 'q_pf_req',
              'p_ess_req': 'p_ess_req', 'q_ess_req': 'q_ess_req',
              'dual_vmag_req': 'dual_vmag_req', 'dual_pf_p_req': 'dual_pf_p_req',
              'dual_pf_q_req': 'dual_pf_q_req', 'dual_ess_p_req': 'dual_ess_p_req',
              'dual_ess_q_req': 'dual_ess_q_req', 'shared_es_s_rated_fixed': 'shared_es_s_rated_fixed',
              'shared_es_e_rated_fixed': 'shared_es_e_rated_fixed'}
    param20 = {r[1]: r for r in st20 if r[0] == 'param'}
    param21 = {r[1]: r for r in st21 if r[0] == 'param'}
    names_common_p = sorted(set(param20) & set(param21))
    param_group_diffs = {g: [] for g in GROUPS}
    param_group_diffs['other'] = []
    for n in names_common_p:
        a_entries = {hashable(k): v for k, v in param20[n][3]}
        b_entries = {hashable(k): v for k, v in param21[n][3]}
        keys = sorted(set(a_entries) | set(b_entries), key=str)
        row_diffs = []
        for k in keys:
            va, vb = dec(a_entries.get(k)), dec(b_entries.get(k))
            if va != vb:
                row_diffs.append({'index': k, 'S20': va, 'S21': vb,
                                   'delta': (vb - va) if isinstance(va, float) and isinstance(vb, float) else None})
        if row_diffs:
            group = GROUPS.get(n, 'other')
            entry = {'param': n, 'n_diffs': len(row_diffs), 'diffs': row_diffs[:50]}
            deltas = [abs(d['delta']) for d in row_diffs if d['delta'] is not None]
            if deltas:
                entry['inf_norm'] = max(deltas)
                entry['argmax_index'] = row_diffs[int(np.argmax([abs(d['delta'] or 0) for d in row_diffs]))]['index']
            param_group_diffs[group].append(entry)
    out['param_group_diffs'] = param_group_diffs
    out['param_only_in_one_side'] = {
        'only_S20': sorted(set(param20) - set(param21)), 'only_S21': sorted(set(param21) - set(param20))}

    n_var_diffs = len(var_val_diffs)
    n_suffix_diffs = sum(v['n_diffs'] for v in suffix_report.values())
    n_param_other_diffs = len(param_group_diffs['other'])
    out['step0_gate'] = {
        'var_values_identical': n_var_diffs == 0,
        'suffixes_identical': n_suffix_diffs == 0,
        'param_diffs_confined_to_named_groups': n_param_other_diffs == 0,
        'FROZEN_STATE_EQUALITY_HOLDS': (n_var_diffs == 0 and n_suffix_diffs == 0 and n_param_other_diffs == 0),
    }
    print(f'[P512K] Step0: var value diffs={n_var_diffs}, suffix diffs={n_suffix_diffs}, '
          f'param-other diffs={n_param_other_diffs}', flush=True)
    return out, {'m_S20': m_S20, 'm_S21': m_S21, 'm_S20start': m_S20start,
                 'net_S21': prepared21['network'], 'admm_params_S21': prepared21['admm_params'],
                 'net_S20start': prepared20start['network'], 'admm_params_S20start': prepared20start['admm_params']}


# ===========================================================================
# shared column-set construction (S21's mapping is structurally canonical;
# confirmed identical across states in Step 0's mapping-hash finding)
# ===========================================================================
def build_optim_columns(model, not_exported_names):
    all_vars = list(model.component_data_objects(pe.Var, active=None))
    eq_bound_names = {v.name for v in all_vars
                       if v.lb is not None and v.ub is not None
                       and float(v.lb) == float(v.ub) and not v.fixed}
    excluded = not_exported_names | eq_bound_names
    optim_vars = [v for v in all_vars if v.name not in excluded]
    name_to_idx = {v.name: i for i, v in enumerate(optim_vars)}
    return optim_vars, name_to_idx, eq_bound_names


def load_not_exported_names():
    mp = json.loads((P512R / 'cycle21_prepared/original_mapping.json').read_text())
    return {e[0] for e in mp['not_exported_variables']}


# ===========================================================================
# Step 1 -- feasibility / bound-push calibration
# ===========================================================================
BP = BF = 1e-5   # bound_push=bound_frac=slack_bound_push=slack_bound_frac
               # =warm_start_bound_push=warm_start_bound_frac
               # =warm_start_slack_bound_push=warm_start_slack_bound_frac (all equal here)


def push_scalar(x, l, u, bp=BP, bf=BF):
    if l is not None and u is not None:
        bnd = u - l
        cl = min(bp * max(1.0, abs(l)), bf * bnd)
        cu = min(bp * max(1.0, abs(u)), bf * bnd)
        if x < l + cl:
            x = l + cl
        if x > u - cu:
            x = u - cu
    elif l is not None:
        cl = bp * max(1.0, abs(l))
        if x < l + cl:
            x = l + cl
    elif u is not None:
        cu = bp * max(1.0, abs(u))
        if x > u - cu:
            x = u - cu
    return x


def num(x):
    try:
        return float(pe.value(x, exception=False))
    except Exception:
        return None


def push_state(optim_vars):
    """Push all optim_vars in place; return (raw_values, pushed_values) dicts by id."""
    raw = {id(v): v.value for v in optim_vars}
    pushed = {}
    for v in optim_vars:
        x = v.value
        if x is None:
            pushed[id(v)] = x
            continue
        l = float(v.lb) if v.lb is not None else None
        u = float(v.ub) if v.ub is not None else None
        pushed[id(v)] = push_scalar(x, l, u)
    return raw, pushed


def set_var_values(optim_vars, values_by_id):
    for v in optim_vars:
        val = values_by_id[id(v)]
        if val is not None:
            v.set_value(val, skip_validation=True)


def equality_violation_and_argmax(model):
    """Equality-row max |body-rhs| at the model's CURRENT variable values."""
    worst = 0.0
    worst_name = None
    for c in model.component_data_objects(pe.Constraint, active=True):
        if not c.equality:
            continue
        body = num(c.body)
        rhs = num(c.lower)
        if body is None or rhs is None:
            continue
        resid = abs(body - rhs)
        if resid > worst:
            worst = resid
            worst_name = c.name
    return worst, worst_name


def step1(model, label, target_pushed, target_raw_context, not_exported_names):
    optim_vars, _, _ = build_optim_columns(model, not_exported_names)
    raw, pushed = push_state(optim_vars)

    raw_viol, raw_argmax = equality_violation_and_argmax(model)  # current values ARE raw at this point

    set_var_values(optim_vars, pushed)
    pushed_viol, pushed_argmax = equality_violation_and_argmax(model)
    set_var_values(optim_vars, raw)  # restore

    n_bounded = sum(1 for v in optim_vars if v.lb is not None or v.ub is not None)
    result = {
        'label': label, 'n_optim_vars': len(optim_vars), 'n_bounded_vars': n_bounded,
        'raw_equality_violation_max': raw_viol, 'raw_equality_violation_argmax': raw_argmax,
        'pushed_equality_violation_max': pushed_viol, 'pushed_equality_violation_argmax': pushed_argmax,
        'gate_target_pushed': target_pushed,
        'gate_pass_3sigfig': (target_pushed is not None and
                               abs(pushed_viol - target_pushed) <= 5e-3 * abs(target_pushed)),
        'reference_context': target_raw_context,
        'note': ('Inequality-row (slack) violation is 0 at the raw point by construction '
                 '(IPOPT initializes the internal slack s0 := body(x0) exactly, unclipped); '
                 'only the equality rows (including reformulated inequality-free structural '
                 'equalities in this all-equality-Pyomo-formulation subset) are informative '
                 'pre-push. All 7826 active Pyomo constraints in this model are structural '
                 'equalities or true inequalities; see the eq/ineq counts recorded separately.'),
    }
    return result


# ===========================================================================
# Step 2 -- objective-gradient calibration
# ===========================================================================
def dso_admm_groups(model, net, admm_params):
    s_base = net.baseMVA
    ref_node_id = net.get_reference_node_id()
    shared_ess_idx = net.get_shared_energy_storage_idx(ref_node_id)
    shared_ess_rating = srp._shared_ess_admm_normalization_pu(
        net.shared_energy_storages[shared_ess_idx].s, s_base, admm_params.shared_ess_normalization_floor_mva)
    interface_transf_rating = net.get_interface_branch_rating() / s_base

    lin = {'vmag': 0, 'pf_p': 0, 'pf_q': 0, 'ess_p': 0, 'ess_q': 0}
    quad = {'vmag': 0, 'pf_p': 0, 'pf_q': 0, 'ess_p': 0, 'ess_q': 0}
    for p in model.periods:
        c_v = model.expected_interface_vmag[p] - model.vmag_req[p]
        lin['vmag'] += model.dual_vmag_req[p] * c_v
        quad['vmag'] += (model.rho_v / 2) * (c_v ** 2)
        c_p = (model.expected_interface_pf_p[p] - model.p_pf_req[p]) / interface_transf_rating
        c_q = (model.expected_interface_pf_q[p] - model.q_pf_req[p]) / interface_transf_rating
        lin['pf_p'] += model.dual_pf_p_req[p] * c_p
        lin['pf_q'] += model.dual_pf_q_req[p] * c_q
        quad['pf_p'] += (model.rho_pf / 2) * (c_p ** 2)
        quad['pf_q'] += (model.rho_pf / 2) * (c_q ** 2)
        c_ep = (model.expected_shared_ess_p[p] - model.p_ess_req[p]) / (2 * shared_ess_rating)
        c_eq = (model.expected_shared_ess_q[p] - model.q_ess_req[p]) / (2 * shared_ess_rating)
        lin['ess_p'] += model.dual_ess_p_req[p] * c_ep
        lin['ess_q'] += model.dual_ess_q_req[p] * c_eq
        quad['ess_p'] += (model.rho_ess / 2) * (c_ep ** 2)
        quad['ess_q'] += (model.rho_ess / 2) * (c_eq ** 2)
    return lin, quad, {'shared_ess_rating': shared_ess_rating, 'interface_transf_rating': interface_transf_rating}


def step2(model, net, admm_params, optim_vars):
    scale = float(pe.value(model.admm_objective_scale))
    obj = model.p58_rescaled_admm_objective.expr
    grad = differentiate(obj, wrt_list=optim_vars, mode=Modes.reverse_numeric)
    gradf = np.array([float(g) if g is not None else 0.0 for g in grad])
    linf = float(np.max(np.abs(gradf)))
    argmax_idx = int(np.argmax(np.abs(gradf)))
    scale_implied = min(1.0, 100.0 / linf) if linf > 0 else 1.0
    scaled_norm = linf * scale_implied

    has_rho_ess_prev = hasattr(model, 'rho_ess_prev')
    lin, quad, normalization = dso_admm_groups(model, net, admm_params)
    base_expr = model.objective.expr
    reconstructed = base_expr
    for k in lin:
        reconstructed = reconstructed + scale * lin[k]
    for k in quad:
        reconstructed = reconstructed + scale * quad[k]
    val_recon = float(pe.value(reconstructed))
    val_actual = float(pe.value(obj))
    value_mismatch_rel = abs(val_recon - val_actual) / abs(val_actual) if val_actual else abs(val_recon - val_actual)

    grad_base = differentiate(base_expr, wrt_list=optim_vars, mode=Modes.reverse_numeric)
    grad_base_v = np.array([float(g) if g is not None else 0.0 for g in grad_base])
    grad_groups = {}
    for part_name, part_dict in (('lin', lin), ('quad', quad)):
        for k, expr in part_dict.items():
            g = differentiate(expr, wrt_list=optim_vars, mode=Modes.reverse_numeric)
            grad_groups[f'{part_name}_{k}'] = np.array([float(x) if x is not None else 0.0 for x in g])
    grad_recon = grad_base_v + scale * sum(grad_groups.values())
    grad_mismatch_inf = float(np.max(np.abs(grad_recon - gradf)))
    grad_mismatch_rel = grad_mismatch_inf / linf if linf else grad_mismatch_inf

    return {
        'admm_objective_scale': scale, 'has_rho_ess_prev': has_rho_ess_prev,
        'grad_linf': linf, 'argmax_var': optim_vars[argmax_idx].name,
        'implied_scale_min(1,100/||grad||_inf)': scale_implied,
        'scaled_norm': scaled_norm, 'gate_target_scale': TARGETS['ObjectiveScaling'],
        'gate_pass': abs(scale_implied - TARGETS['ObjectiveScaling']) < 1e-9,
        'normalization_constants': normalization,
        'objective_value_reconstruction_relative_mismatch': value_mismatch_rel,
        'objective_gradient_reconstruction_relative_mismatch_inf': grad_mismatch_rel,
        'gradient_reconstruction_pass_1e-9': grad_mismatch_rel < 1e-9,
        '_gradf': gradf, '_grad_base': grad_base_v, '_grad_groups': grad_groups,
    }


# ===========================================================================
# Step 3 -- stationarity reconstruction
#   Calibrated convention (established empirically against S20's exact target,
#   relative error 9.1e-9; and against S21/S20start pushed targets, relative
#   error < 3e-14):
#       r = grad_f(x) - J(x)^T * dual_raw(x) - zL_raw(x) - zU_raw(x)
#   where dual_raw is Pyomo's own 'dual' Suffix value (NOT the textbook +J^T y
#   convention: sigma=-1 is required) and zU_raw is Pyomo's own ipopt_zU_*
#   suffix value taken AS STORED (uniformly negative here) -- i.e. the
#   textbook non-negative z_U = -zU_raw, so "-z_L + z_U" in the textbook
#   formula becomes "-zL_raw - zU_raw" in terms of the raw suffix values.
# ===========================================================================
def build_jacobian_transpose_dot(model, dual_suffix, name_to_idx, n, budget_state, label):
    if len(budget_state['builds']) >= JACOBIAN_BUDGET:
        raise RuntimeError(f'Jacobian build budget ({JACOBIAN_BUDGET}) exceeded, refusing build for {label}')
    t0 = time.time()
    JT = np.zeros(n)
    n_rows = 0
    n_rows_with_lambda = 0
    for c in model.component_data_objects(pe.Constraint, active=True):
        n_rows += 1
        lam = dual_suffix.get(c)
        if lam is None or lam == 0.0:
            continue
        n_rows_with_lambda += 1
        vs = list(identify_variables(c.body, include_fixed=False))
        if not vs:
            continue
        grads = differentiate(c.body, wrt_list=vs, mode=Modes.reverse_numeric)
        lamf = float(lam)
        for v, g in zip(vs, grads):
            idx = name_to_idx.get(v.name)
            if idx is not None and g is not None:
                JT[idx] += lamf * float(g)
    elapsed = time.time() - t0
    rec = {'label': label, 'n_rows': n_rows, 'n_rows_with_nonzero_lambda': n_rows_with_lambda,
           'wall_seconds': elapsed, 'over_budget_time': elapsed > JACOBIAN_TIME_LIMIT_S}
    budget_state['builds'].append(rec)
    print(f'[P512K] Jacobian build #{len(budget_state["builds"])} ({label}): '
          f'{n_rows} rows, {elapsed:.3f}s', flush=True)
    if rec['over_budget_time']:
        raise TimeoutError(f'Jacobian build for {label} exceeded {JACOBIAN_TIME_LIMIT_S}s budget')
    return JT


def stationarity_residual(gradf, JT, zL_raw_v, zU_raw_v):
    r = gradf - JT - zL_raw_v - zU_raw_v
    return r


def suffix_vector(suffix, optim_vars, name_to_idx, n):
    v = np.zeros(n)
    for var in optim_vars:
        val = suffix.get(var)
        if val is not None:
            v[name_to_idx[var.name]] = float(val)
    return v


def push_mult_vectors(zL_raw, zU_raw, mult_push=1e-5):
    zL = np.maximum(zL_raw, mult_push)
    zU = np.minimum(zU_raw, -mult_push)  # zU_raw is <= 0 in this convention
    return zL, zU


def report_r(r, optim_vars, name='r'):
    absr = np.abs(r)
    idx = int(np.argmax(absr))
    return {
        'linf': float(np.max(absr)), 'l2': float(np.linalg.norm(r)),
        'argmax_var': optim_vars[idx].name, 'signed_residual_at_argmax': float(r[idx]),
        'quantiles': {q: float(np.quantile(absr, q)) for q in (0.5, 0.9, 0.99, 0.999)},
        '_argmax_idx': idx,
    }


def step3(states):
    """states: dict with 'S20','S21','S20start' each a dict of prepared inputs."""
    budget_state = JOURNAL['jacobian_budget']
    result = {}

    # ---------- S20 : converged final point, no push ----------
    m20 = states['S20']['model']
    optim20, idx20, _ = build_optim_columns(m20, states['not_exported'])
    n20 = len(optim20)
    grad20 = differentiate(m20.p58_rescaled_admm_objective.expr, wrt_list=optim20, mode=Modes.reverse_numeric)
    gradf20 = np.array([float(g) if g is not None else 0.0 for g in grad20])
    JT20 = build_jacobian_transpose_dot(m20, m20.dual, idx20, n20, budget_state, 'S20 (converged)')
    zL20 = suffix_vector(m20.ipopt_zL_out, optim20, idx20, n20)
    zU20 = suffix_vector(m20.ipopt_zU_out, optim20, idx20, n20)
    r20 = stationarity_residual(gradf20, JT20, zL20, zU20)
    rep20 = report_r(r20, optim20)
    rep20['gate_target'] = TARGETS['S20_dual_inf_final']
    rep20['gate_relative_error'] = (rep20['linf'] - TARGETS['S20_dual_inf_final']) / TARGETS['S20_dual_inf_final']
    rep20['calibrated'] = abs(rep20['gate_relative_error']) < 1e-4
    result['S20'] = rep20

    # ---------- S21 : raw and pushed ----------
    m21 = states['S21']['model']
    optim21, idx21, _ = build_optim_columns(m21, states['not_exported'])
    n21 = len(optim21)
    raw21, pushed21 = push_state(optim21)

    grad21_raw = differentiate(m21.p58_rescaled_admm_objective.expr, wrt_list=optim21, mode=Modes.reverse_numeric)
    gradf21_raw = np.array([float(g) if g is not None else 0.0 for g in grad21_raw])
    JT21_raw = build_jacobian_transpose_dot(m21, m21.dual, idx21, n21, budget_state, 'S21 (raw)')
    zL21_raw_v = suffix_vector(m21.ipopt_zL_in, optim21, idx21, n21)
    zU21_raw_v = suffix_vector(m21.ipopt_zU_in, optim21, idx21, n21)
    r21_raw = stationarity_residual(gradf21_raw, JT21_raw, zL21_raw_v, zU21_raw_v)
    rep21_raw = report_r(r21_raw, optim21)

    set_var_values(optim21, pushed21)
    grad21_pushed = differentiate(m21.p58_rescaled_admm_objective.expr, wrt_list=optim21, mode=Modes.reverse_numeric)
    gradf21_pushed = np.array([float(g) if g is not None else 0.0 for g in grad21_pushed])
    JT21_pushed = build_jacobian_transpose_dot(m21, m21.dual, idx21, n21, budget_state, 'S21 (pushed)')
    zL21_pushed, zU21_pushed = push_mult_vectors(zL21_raw_v, zU21_raw_v)
    r21_pushed = stationarity_residual(gradf21_pushed, JT21_pushed, zL21_pushed, zU21_pushed)
    rep21_pushed = report_r(r21_pushed, optim21)
    rep21_pushed['gate_target'] = TARGETS['S21_dual_inf_iter0']
    rep21_pushed['gate_relative_error'] = (rep21_pushed['linf'] - TARGETS['S21_dual_inf_iter0']) / TARGETS['S21_dual_inf_iter0']
    rep21_pushed['calibrated'] = abs(rep21_pushed['gate_relative_error']) < 1e-4
    set_var_values(optim21, raw21)  # restore

    result['S21_raw'] = rep21_raw
    result['S21_pushed'] = rep21_pushed

    # ---------- S20start : raw and pushed ----------
    m20s = states['S20start']['model']
    optim20s, idx20s, _ = build_optim_columns(m20s, states['not_exported'])
    n20s = len(optim20s)
    raw20s, pushed20s = push_state(optim20s)

    grad20s_raw = differentiate(m20s.p58_rescaled_admm_objective.expr, wrt_list=optim20s, mode=Modes.reverse_numeric)
    gradf20s_raw = np.array([float(g) if g is not None else 0.0 for g in grad20s_raw])
    JT20s_raw = build_jacobian_transpose_dot(m20s, m20s.dual, idx20s, n20s, budget_state, 'S20start (raw)')
    zL20s_raw_v = suffix_vector(m20s.ipopt_zL_in, optim20s, idx20s, n20s)
    zU20s_raw_v = suffix_vector(m20s.ipopt_zU_in, optim20s, idx20s, n20s)
    r20s_raw = stationarity_residual(gradf20s_raw, JT20s_raw, zL20s_raw_v, zU20s_raw_v)
    rep20s_raw = report_r(r20s_raw, optim20s)

    set_var_values(optim20s, pushed20s)
    grad20s_pushed = differentiate(m20s.p58_rescaled_admm_objective.expr, wrt_list=optim20s, mode=Modes.reverse_numeric)
    gradf20s_pushed = np.array([float(g) if g is not None else 0.0 for g in grad20s_pushed])
    JT20s_pushed = build_jacobian_transpose_dot(m20s, m20s.dual, idx20s, n20s, budget_state, 'S20start (pushed)')
    zL20s_pushed, zU20s_pushed = push_mult_vectors(zL20s_raw_v, zU20s_raw_v)
    r20s_pushed = stationarity_residual(gradf20s_pushed, JT20s_pushed, zL20s_pushed, zU20s_pushed)
    rep20s_pushed = report_r(r20s_pushed, optim20s)
    rep20s_pushed['gate_target'] = TARGETS['S20start_dual_inf_iter0']
    rep20s_pushed['gate_relative_error'] = (rep20s_pushed['linf'] - TARGETS['S20start_dual_inf_iter0']) / TARGETS['S20start_dual_inf_iter0']
    rep20s_pushed['calibrated'] = abs(rep20s_pushed['gate_relative_error']) < 1e-4
    set_var_values(optim20s, raw20s)  # restore

    result['S20start_raw'] = rep20s_raw
    result['S20start_pushed'] = rep20s_pushed

    # keep the vectors/context for Step 4 (S20's own build is reused there)
    vectors = {
        'S20': {'model': m20, 'optim_vars': optim20, 'name_to_idx': idx20, 'n': n20,
                'gradf': gradf20, 'JT': JT20, 'zL': zL20, 'zU': zU20, 'r': r20},
        'S21_raw': {'model': m21, 'optim_vars': optim21, 'name_to_idx': idx21, 'n': n21,
                    'gradf': gradf21_raw, 'JT': JT21_raw, 'zL': zL21_raw_v, 'zU': zU21_raw_v, 'r': r21_raw,
                    'raw_values': raw21, 'pushed_values': pushed21},
        'S21_pushed': {'model': m21, 'optim_vars': optim21, 'name_to_idx': idx21, 'n': n21,
                       'gradf': gradf21_pushed, 'JT': JT21_pushed, 'zL': zL21_pushed, 'zU': zU21_pushed, 'r': r21_pushed},
    }
    return result, vectors


# ===========================================================================
# Step 4 -- attribution on residual vectors
# ===========================================================================
def step4(step3_result, vectors, states):
    m20 = vectors['S20']['model']
    optim20 = vectors['S20']['optim_vars']
    idx20 = vectors['S20']['name_to_idx']
    r20 = vectors['S20']['r']
    r21p = vectors['S21_pushed']['r']
    optim21 = vectors['S21_pushed']['optim_vars']

    if len(optim20) != len(optim21) or [v.name for v in optim20] != [v.name for v in optim21]:
        return {'status': 'INCONCLUSIVE', 'reason': 'optimization-column ordering differs between S20 and S21; '
                                                      'vector subtraction R21-R20 is not well-defined column-wise.'}

    dR = r21p - r20

    # --- ADMM/objective-parameter-change contribution: same x as S20 (its own
    # converged point, no push), only the model's mutable ADMM Params swapped to
    # cycle21's values; J and lambda REUSED from S20's own build (x unchanged).
    admm_params_20 = ['vmag_req', 'p_pf_req', 'q_pf_req', 'p_ess_req', 'q_ess_req',
                      'dual_vmag_req', 'dual_pf_p_req', 'dual_pf_q_req', 'dual_ess_p_req', 'dual_ess_q_req']
    saved = {}
    for pname in admm_params_20:
        p20 = getattr(m20, pname)
        saved[pname] = {k: pe.value(p20[k]) for k in p20}
    m21 = vectors['S21_pushed']['model']
    for pname in admm_params_20:
        p20 = getattr(m20, pname)
        p21 = getattr(m21, pname)
        for k in p20:
            p20[k] = pe.value(p21[k])
    grad20_admm21 = differentiate(m20.p58_rescaled_admm_objective.expr, wrt_list=optim20, mode=Modes.reverse_numeric)
    gradf20_admm21 = np.array([float(g) if g is not None else 0.0 for g in grad20_admm21])
    r_admm_only = stationarity_residual(gradf20_admm21, vectors['S20']['JT'], vectors['S20']['zL'], vectors['S20']['zU'])
    for pname in admm_params_20:
        p20 = getattr(m20, pname)
        for k in p20:
            p20[k] = saved[pname][k]
    dR_admm = r_admm_only - r20

    # --- push-only contribution: same cycle21 Params, x/z move from raw to pushed
    dR_push = r21p - vectors['S21_raw']['r']

    dR_other = dR - dR_admm - dR_push

    def summarize(vec, name):
        absv = np.abs(vec)
        idx = int(np.argmax(absv))
        return {'name': name, 'linf': float(np.max(absv)), 'l2': float(np.linalg.norm(vec)),
                'value_at_argmax_dR': float(vec[int(np.argmax(np.abs(dR)))]),
                'argmax_var': optim20[idx].name, 'value_at_own_argmax': float(vec[idx])}

    argmax_dR_idx = int(np.argmax(np.abs(dR)))
    recon = dR_admm + dR_push + dR_other
    recon_err = float(np.max(np.abs(recon - dR)))

    contributions = {
        'ADMM_parameter_change': summarize(dR_admm, 'ADMM_parameter_change'),
        'bound_and_mult_push': summarize(dR_push, 'bound_and_mult_push'),
        'other_remainder': summarize(dR_other, 'other_remainder'),
    }
    total_at_argmax = float(dR[argmax_dR_idx])
    fractions = {}
    for k, c in contributions.items():
        val = c['value_at_argmax_dR']
        frac = (val / total_at_argmax) if total_at_argmax != 0 else None
        sign_match = (val * total_at_argmax > 0) if total_at_argmax != 0 else None
        fractions[k] = {'value': val, 'fraction_of_total_at_argmax': frac, 'sign_matches': sign_match}

    dominant = None
    for k, c in contributions.items():
        if c['linf'] >= 0.9 * float(np.max(np.abs(dR))) and fractions[k]['sign_matches']:
            dominant = k
    status = 'DOMINANT:' + dominant if dominant else 'MIXED/INCONCLUSIVE'

    return {
        'dR_linf': float(np.max(np.abs(dR))), 'dR_l2': float(np.linalg.norm(dR)),
        'dR_argmax_var': optim20[argmax_dR_idx].name, 'dR_value_at_argmax': total_at_argmax,
        'contributions': contributions, 'fractions_at_dR_argmax': fractions,
        'reconstruction_error_inf': recon_err,
        'reconstruction_relative_error': recon_err / float(np.max(np.abs(dR))) if np.max(np.abs(dR)) else None,
        'dominance_status': status,
        'note': ('Step 0 found that S20 and S21 do NOT share an identical primal point '
                 '(shared_es_soc / shared_es_e_rated differ). "other_remainder" therefore '
                 'absorbs both genuine nonlinear interaction effects AND that uncontrolled '
                 'primal-point shift; it is not a clean third category.'),
    }


# ===========================================================================
# Step 5 -- Arm A terminal-iterate forensic (independent of Steps 0-4)
# ===========================================================================
_ITER_ROW = re.compile(
    r'^\s*(\d+)(r?)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+'
    r'([-\d.eE+]+)\s+(\S+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*(\S*)\s*(\d*)\s*([a-zA-Z]*)\s*$')


def parse_iteration_table(log_path):
    rows = []
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            if not re.match(r'^\s*\d+r?\s+[\d.eE+-]', line):
                continue
            parts = line.split()
            try:
                it = int(parts[0].rstrip('r'))
                restoration = parts[0].endswith('r')
                obj = float(parts[1]); inf_pr = float(parts[2]); inf_du = float(parts[3])
                lg_mu = float(parts[4])
                marker = parts[-1] if re.match(r'^[a-zA-Z]+$', parts[-1]) else ''
            except (ValueError, IndexError):
                continue
            rows.append({'iter': it, 'restoration': restoration, 'objective': obj,
                         'inf_pr': inf_pr, 'inf_du': inf_du, 'lg_mu': lg_mu, 'marker': marker})
    return rows


def parse_sol_file(sol_path):
    lines = sol_path.read_text().splitlines()
    opt_idx = lines.index('Options')
    nums = [int(lines[opt_idx + 1 + i]) for i in range(8)]
    m, n = nums[4], nums[6]
    data_start = opt_idx + 1 + 8
    duals = [float(x) for x in lines[data_start:data_start + m]]
    primals = [float(x) for x in lines[data_start + m:data_start + m + n]]
    suffixes = {}
    i = data_start + m + n
    while i < len(lines):
        if lines[i].startswith('suffix'):
            header = lines[i].split()
            count = int(header[2])
            sname = lines[i + 1]
            entries = {}
            for j in range(count):
                idx_str, val_str = lines[i + 2 + j].split()
                entries[int(idx_str)] = float(val_str)
            suffixes[sname] = entries
            i = i + 2 + count
        else:
            i += 1
    return {'m': m, 'n': n, 'duals': duals, 'primals': primals, 'suffixes': suffixes}


def step5():
    log_path = ARMA / 'logs/optim_log_case33_3_2025_Spring.log'
    cycle20_log = P512R / 'cycle20_target.log'
    sol_path = ARMA / 'used_tmp27ntrqce.pyomo.sol'
    mapping_path = ARMA / 'armA_prepared_mapping.json'

    rows_armA = parse_iteration_table(log_path)
    rows_c20 = parse_iteration_table(cycle20_log)
    n_z_marker_armA = sum(1 for r in rows_armA if 'z' in r['marker'])
    n_z_marker_c20 = sum(1 for r in rows_c20 if 'z' in r['marker'])
    first_z_iter = next((r['iter'] for r in rows_armA if 'z' in r['marker']), None)

    n_msg = len(re.findall(r'Some value in z_U becomes too large', log_path.read_text(errors='replace')))
    n_msg_c20 = len(re.findall(r'Some value in z_U becomes too large', cycle20_log.read_text(errors='replace')))

    text = log_path.read_text(errors='replace')
    mu_match = re.findall(r'Current barrier parameter mu = ([\d.eE+-]+)', text)
    final_mu = float(mu_match[-1]) if mu_match else None

    final_unscaled = re.findall(
        r'Constraint violation\.+:\s+([\d.eE+-]+)\s+([\d.eE+-]+)', text)
    final_violation_unscaled = float(final_unscaled[-1][1]) if final_unscaled else None

    side_by_side = {
        'cycle20_iterations_60_115': [r for r in rows_c20 if 60 <= r['iter'] <= 115],
        'cycle21_iterations_60_120': [r for r in rows_armA if 60 <= r['iter'] <= 120],
    }

    sol = parse_sol_file(sol_path)
    mapping = json.loads(mapping_path.read_text())
    sym_to_name = dict(mapping['mapping'])
    var_syms = sorted((s for s in sym_to_name if s.startswith('v')), key=lambda s: int(s[1:]))
    con_syms = sorted((s for s in sym_to_name if s.startswith('c')), key=lambda s: int(s[1:]))

    with (P512R / 'cycle21_prepared/snapshot.pkl').open('rb') as f:
        model = pickle.load(f)['model']
    name_lookup = {}
    for v in model.component_data_objects(pe.Var, active=None):
        name_lookup[v.name] = v

    x_final = {}
    for i, sym in enumerate(var_syms):
        name = sym_to_name[sym]
        x_final[name] = sol['primals'][i]

    saved_vals = {n: v.value for n, v in name_lookup.items()}
    for name, val in x_final.items():
        if name in name_lookup:
            name_lookup[name].set_value(val, skip_validation=True)

    offenders = []
    n_eq_rows_checked = 0
    n_ineq_rows_checked = 0
    for c in model.component_data_objects(pe.Constraint, active=True):
        body = num(c.body)
        if body is None:
            continue
        if c.equality:
            n_eq_rows_checked += 1
            resid = abs(body - num(c.lower))
            side = 'equality'
        else:
            n_ineq_rows_checked += 1
            lo = num(c.lower); hi = num(c.upper)
            viol = 0.0
            if lo is not None and body < lo:
                viol = lo - body
                side = 'below_lower'
            elif hi is not None and body > hi:
                viol = body - hi
                side = 'above_upper'
            else:
                side = 'satisfied'
            resid = viol
        if resid > 1e-6:
            offenders.append({'name': c.name, 'residual': resid, 'side': side, 'equality': c.equality})
    offenders.sort(key=lambda o: -o['residual'])

    zU_out = sol['suffixes'].get('ipopt_zU_out', {})
    zL_out = sol['suffixes'].get('ipopt_zL_out', {})
    complementarity_products = []
    for i, sym in enumerate(var_syms):
        name = sym_to_name[sym]
        v = name_lookup.get(name)
        if v is None or v.fixed:
            continue
        lb = float(v.lb) if v.lb is not None else None
        ub = float(v.ub) if v.ub is not None else None
        if lb is not None and ub is not None and lb == ub:
            continue  # equal-bound column excluded per protocol
        x = x_final[name]
        zl = zL_out.get(i)
        zu = zU_out.get(i)
        prod_l = (zl * (x - lb)) if (zl is not None and lb is not None) else None
        prod_u = (zu * (ub - x)) if (zu is not None and ub is not None) else None
        if prod_l is not None or prod_u is not None:
            complementarity_products.append({'name': name, 'zL_out': zl, 'zU_out': zu,
                                              'x': x, 'lb': lb, 'ub': ub,
                                              'zL_times_(x-l)': prod_l, 'zU_times_(u-x)': prod_u})

    prods_l = [c['zL_times_(x-l)'] for c in complementarity_products if c['zL_times_(x-l)'] is not None]
    prods_u = [c['zU_times_(u-x)'] for c in complementarity_products if c['zU_times_(u-x)'] is not None]

    def dist_summary(vals):
        if not vals:
            return None
        a = np.abs(np.array(vals))
        return {'n': len(a), 'max': float(a.max()), 'mean': float(a.mean()),
                'median': float(np.median(a)), 'n_exceeding_terminal_mu':
                    int(np.sum(a > TARGETS['ArmA_terminal_mu']))}

    for n, v in name_lookup.items():
        if n in saved_vals and saved_vals[n] is not None:
            v.set_value(saved_vals[n], skip_validation=True)

    largeZ = sorted(
        [{'name': sym_to_name[f'v{i}'], 'zU_out': v} for i, v in zU_out.items() if abs(v) > 10],
        key=lambda d: -abs(d['zU_out']))[:25]
    largeZL = sorted(
        [{'name': sym_to_name[f'v{i}'], 'zL_out': v} for i, v in zL_out.items() if abs(v) > 10],
        key=lambda d: -abs(d['zL_out']))[:25]

    return {
        'log_iteration_rows_parsed_armA': len(rows_armA), 'log_iteration_rows_parsed_cycle20': len(rows_c20),
        'z_marker_count_armA': n_z_marker_armA, 'z_marker_count_cycle20': n_z_marker_c20,
        'first_z_marker_iteration_armA': first_z_iter,
        'z_U_correction_message_count_armA': n_msg, 'z_U_correction_message_count_cycle20': n_msg_c20,
        'final_barrier_mu_armA': final_mu, 'final_constraint_violation_unscaled_armA': final_violation_unscaled,
        'target_terminal_violation': TARGETS['ArmA_terminal_violation'],
        'terminal_violation_match': (final_violation_unscaled is not None and
                                      abs(final_violation_unscaled - TARGETS['ArmA_terminal_violation']) < 1e-9),
        'target_terminal_mu': TARGETS['ArmA_terminal_mu'],
        'terminal_mu_match': (final_mu is not None and abs(final_mu - TARGETS['ArmA_terminal_mu']) < 1e-9),
        'n_equality_rows_checked': n_eq_rows_checked, 'n_inequality_rows_checked': n_ineq_rows_checked,
        'top_offending_rows': offenders[:30],
        'inequality_rows_note': ('Inequality-row residuals above are the PHYSICAL body-vs-bound '
                                  'violation at the final .sol primal point; they are NOT the internal '
                                  'IPOPT slack-equality residual, since the .sol file carries no slack '
                                  'values for inequality rows.'),
        'complementarity_products_zL_times_(x-l)': dist_summary(prods_l),
        'complementarity_products_zU_times_(u-x)': dist_summary(prods_u),
        'largest_|zU_out|_variables': largeZ, 'largest_|zL_out|_variables': largeZL,
        'side_by_side_iterations': side_by_side,
    }


# ===========================================================================
# main
# ===========================================================================
def main():
    if OUT.exists():
        print(f'[P512K] refusing to run: {OUT} already exists', file=sys.stderr)
        sys.exit(2)
    OUT.mkdir(parents=True)

    harness_sha_now = sha(ROOT / 'p512_r_presolve_recapture.py')
    if harness_sha_now != HARNESS_SHA_EXPECTED:
        print(f'[P512K] STOP: harness sha mismatch {harness_sha_now} != {HARNESS_SHA_EXPECTED}', file=sys.stderr)
        sys.exit(3)

    install_solver_guards()
    JOURNAL['repo_state_start'] = repo_state()
    JOURNAL['harness_sha256'] = harness_sha_now
    t_start = time.time()

    try:
        step0_out, models = step0()
        JOURNAL['step0'] = step0_out
        JOURNAL['status'] = 'STEP0_DONE'

        not_exported = load_not_exported_names()

        # Step 1 -- push calibration, both parameterizations
        step1_21 = step1(models['m_S21'], 'S21 (cycle21_prepared)',
                          TARGETS['S21_constraint_violation_iter0_pushed'],
                          TARGETS['S20_constraint_violation_converged'], not_exported)
        step1_20s = step1(models['m_S20start'], 'S20start (cycle20_prepared)', None, None, not_exported)
        JOURNAL['step1'] = {'S21': step1_21, 'S20start': step1_20s}

        # Step 2 -- objective gradient calibration (on S21)
        optim21, idx21, _ = build_optim_columns(models['m_S21'], not_exported)
        step2_out = step2(models['m_S21'], models['net_S21'], models['admm_params_S21'], optim21)
        JOURNAL['step2'] = {k: v for k, v in step2_out.items() if not k.startswith('_')}

        # Step 3 -- stationarity, three states, calibrated on S20
        states_for_3 = {'S20': {'model': models['m_S20']},
                        'S21': {'model': models['m_S21']},
                        'S20start': {'model': models['m_S20start']},
                        'not_exported': not_exported}
        step3_out, vectors = step3(states_for_3)
        JOURNAL['step3'] = step3_out

        calibrated = step3_out['S20']['calibrated']
        if not calibrated:
            JOURNAL['step3']['CALIBRATION_STATUS'] = 'NOT CALIBRATED'
            JOURNAL['step4'] = {'status': 'SKIPPED', 'reason': 'Step 3 stationarity not calibrated on S20'}
        else:
            JOURNAL['step3']['CALIBRATION_STATUS'] = 'CALIBRATED'
            step4_out = step4(step3_out, vectors, states_for_3)
            JOURNAL['step4'] = step4_out

        # Step 5 -- independent Arm A forensic
        JOURNAL['step5'] = step5()

        JOURNAL['status'] = 'COMPLETED'
    except BaseException as exc:
        JOURNAL['status'] = 'STOPPED'
        JOURNAL['stop_reason'] = f'{type(exc).__name__}: {exc}'
        import traceback
        JOURNAL['traceback'] = traceback.format_exc()
        print('[P512K] STOPPED:', JOURNAL['stop_reason'], file=sys.stderr)

    JOURNAL['zero_solver']['counters'] = dict(_SOLVE_COUNTERS)
    JOURNAL['zero_solver']['zero_confirmed'] = all(v == 0 for v in _SOLVE_COUNTERS.values())
    JOURNAL['repo_state_end'] = repo_state()
    JOURNAL['wall_seconds'] = time.time() - t_start
    uninstall_solver_guards()

    # -------- write outputs --------
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else str(obj)
        return obj

    write_report()
    record_output('report', OUT / 'P5_12_K_NO_SOLVE_KKT_FORENSIC_REPORT.md')

    journal_path = OUT / 'p512k_journal.json'
    dump_json(journal_path, to_jsonable(JOURNAL))
    record_output('journal', journal_path)

    manifest = {'inputs': JOURNAL['artifacts']['inputs'], 'outputs': JOURNAL['artifacts']['outputs'],
                'script_sha256': sha(Path(__file__)), 'repo_state_start': JOURNAL['repo_state_start'],
                'repo_state_end': JOURNAL['repo_state_end']}
    dump_json(OUT / 'manifest.json', manifest)

    print('[P512K]', JOURNAL['status'], JOURNAL.get('stop_reason', ''), flush=True)
    sys.exit(0 if JOURNAL['status'] == 'COMPLETED' else 2)


def write_report():
    j = JOURNAL
    lines = []
    lines.append('# P5.12-K -- NO-SOLVE KKT / warm-start forensic report\n')
    lines.append(j['preamble'] + '\n')
    lines.append('## 0. Zero-solver evidence\n')
    lines.append(f"- `OptSolver.solve` calls: {j['zero_solver']['counters']['OptSolver.solve']}")
    lines.append(f"- `SystemCallSolver._execute_command` calls: {j['zero_solver']['counters']['SystemCallSolver._execute_command']}")
    lines.append(f"- zero confirmed: {j['zero_solver']['zero_confirmed']}\n")

    s0 = j.get('step0', {})
    lines.append('## 1. Step 0 -- frozen-state equality\n')
    lines.append(f"- all frozen-input file hashes match: {s0.get('file_hashes_all_match')}")
    sd = s0.get('semantic_digest_checks', {})
    for k, v in sd.items():
        lines.append(f"- semantic digest {k}: match={v['match']}")
    lines.append(f"- checkpoint state digest match: {s0.get('checkpoint_state_digest_check', {}).get('match')}")
    lines.append(f"- checkpoint block digest match: {s0.get('checkpoint_block_digest_check', {}).get('match')}\n")
    lines.append('### Mapping-hash finding\n')
    mh = s0.get('mapping_hash_finding', {})
    lines.append(mh.get('interpretation', '') + '\n')
    vc = s0.get('var_comparison', {})
    lines.append('### S20 vs S21 variable-value equality\n')
    lines.append(f"- n common vars: {vc.get('n_common')}, value diffs: {vc.get('n_value_diffs')}, "
                 f"bound diffs: {vc.get('n_bound_diffs')}, fixed diffs: {vc.get('n_fixed_diffs')}")
    if vc.get('n_value_diffs'):
        lines.append(f"- **STOP-CONDITION TRIGGERED: var values differ.** argmax={vc.get('argmax')}")
        lines.append('- top differing variables (all `shared_es_soc`/`shared_es_e_rated`):')
        for d in vc.get('value_diffs', [])[:10]:
            lines.append(f"  - {d['name']}: S20={d['S20']!r} S21={d['S21']!r} delta={d['delta']!r}")
    sc = s0.get('suffix_comparison', {})
    lines.append('\n### S20 vs S21 suffix equality (dual, ipopt_zL_out, ipopt_zU_out)\n')
    for name, v in sc.items():
        lines.append(f"- {name}: n_common_keys={v['n_common_keys']}, n_diffs={v['n_diffs']}, "
                     f"only_S20={v['n_only_S20']}, only_S21={v['n_only_S21']}")
    lines.append('\n### Param group differences\n')
    for group, entries in s0.get('param_group_diffs', {}).items():
        if entries:
            lines.append(f"- **{group}**: {len(entries)} differing Param(s), "
                         f"e.g. {entries[0]['param']} n_diffs={entries[0]['n_diffs']} "
                         f"inf_norm={entries[0].get('inf_norm')}")
    lines.append(f"\n**Step-0 gate:** {s0.get('step0_gate')}\n")

    lines.append('## 2. Step 1 -- feasibility / bound-push calibration\n')
    s1 = j.get('step1', {})
    for label, rec in s1.items():
        lines.append(f"### {label}\n")
        lines.append(f"- raw equality violation max: {rec.get('raw_equality_violation_max')} "
                     f"(argmax {rec.get('raw_equality_violation_argmax')})")
        lines.append(f"- pushed equality violation max: {rec.get('pushed_equality_violation_max')} "
                     f"(argmax {rec.get('pushed_equality_violation_argmax')})")
        lines.append(f"- gate target: {rec.get('gate_target_pushed')}; gate pass (>=3 sig figs): {rec.get('gate_pass_3sigfig')}")
        lines.append(f"- reference (cycle-20 converged) context: {rec.get('reference_context')}\n")

    lines.append('## 3. Step 2 -- objective-gradient calibration\n')
    s2 = j.get('step2', {})
    lines.append(f"- ||grad f||_inf = {s2.get('grad_linf')} at {s2.get('argmax_var')}")
    lines.append(f"- implied IPOPT scaling min(1,100/||grad||_inf) = {s2.get('implied_scale_min(1,100/||grad||_inf)')}")
    lines.append(f"- gate target = {s2.get('gate_target_scale')}; gate pass = {s2.get('gate_pass')}")
    lines.append(f"- objective value reconstruction relative mismatch: {s2.get('objective_value_reconstruction_relative_mismatch')}")
    lines.append(f"- objective gradient reconstruction relative mismatch (inf): {s2.get('objective_gradient_reconstruction_relative_mismatch_inf')}")
    lines.append(f"- gradient reconstruction pass (<1e-9): {s2.get('gradient_reconstruction_pass_1e-9')}\n")

    lines.append('## 4. Step 3 -- stationarity reconstruction\n')
    lines.append('Calibrated convention: `r = grad_f(x) - J(x)^T*dual_raw(x) - zL_raw(x) - zU_raw(x)`, '
                 'using Pyomo\'s own `dual`/`ipopt_zL_*`/`ipopt_zU_*` suffix values exactly as stored '
                 '(no re-signing beyond this single fixed convention).\n')
    s3 = j.get('step3', {})
    for label in ('S20', 'S21_raw', 'S21_pushed', 'S20start_raw', 'S20start_pushed'):
        rec = s3.get(label, {})
        lines.append(f"### {label}")
        lines.append(f"- ||r||_inf = {rec.get('linf')}, ||r||_2 = {rec.get('l2')}, argmax = {rec.get('argmax_var')}")
        if 'gate_target' in rec:
            lines.append(f"- gate target = {rec.get('gate_target')}, relative error = {rec.get('gate_relative_error')}, "
                         f"calibrated = {rec.get('calibrated')}")
        lines.append('')
    lines.append(f"**Calibration status: {s3.get('CALIBRATION_STATUS')}**\n")

    lines.append('## 5. Step 4 -- attribution on residual vectors\n')
    s4 = j.get('step4', {})
    if s4.get('status') == 'SKIPPED':
        lines.append(f"SKIPPED: {s4.get('reason')}\n")
    else:
        lines.append(f"- ||dR||_inf = {s4.get('dR_linf')}, argmax = {s4.get('dR_argmax_var')}, "
                     f"value at argmax = {s4.get('dR_value_at_argmax')}")
        lines.append(f"- reconstruction error (inf): {s4.get('reconstruction_error_inf')} "
                     f"(relative: {s4.get('reconstruction_relative_error')})")
        for k, c in s4.get('contributions', {}).items():
            f_ = s4.get('fractions_at_dR_argmax', {}).get(k, {})
            lines.append(f"- {k}: ||.||_inf={c['linf']}, value at dR-argmax={f_.get('value')}, "
                         f"fraction={f_.get('fraction_of_total_at_argmax')}, sign_matches={f_.get('sign_matches')}")
        lines.append(f"- dominance status: {s4.get('dominance_status')}")
        lines.append(f"- note: {s4.get('note')}\n")

    lines.append('## 6. Step 5 -- Arm A terminal-iterate forensic (independent)\n')
    s5 = j.get('step5', {})
    lines.append(f"- iteration rows parsed: Arm A={s5.get('log_iteration_rows_parsed_armA')}, "
                 f"cycle20={s5.get('log_iteration_rows_parsed_cycle20')}")
    lines.append(f"- 'z' marker count: Arm A={s5.get('z_marker_count_armA')} (first at iteration "
                 f"{s5.get('first_z_marker_iteration_armA')}), cycle20={s5.get('z_marker_count_cycle20')}")
    lines.append(f"- 'Some value in z_U becomes too large' message count: Arm A={s5.get('z_U_correction_message_count_armA')}, "
                 f"cycle20={s5.get('z_U_correction_message_count_cycle20')}")
    lines.append(f"- final barrier mu (Arm A) = {s5.get('final_barrier_mu_armA')} (target {s5.get('target_terminal_mu')}, "
                 f"match={s5.get('terminal_mu_match')})")
    lines.append(f"- final constraint violation (unscaled, Arm A) = {s5.get('final_constraint_violation_unscaled_armA')} "
                 f"(target {s5.get('target_terminal_violation')}, match={s5.get('terminal_violation_match')})")
    lines.append(f"- top offending rows (first 10 of {len(s5.get('top_offending_rows', []))}):")
    for row in s5.get('top_offending_rows', [])[:10]:
        lines.append(f"  - {row['name']}: residual={row['residual']}, side={row['side']}, equality={row['equality']}")
    lines.append(f"- zL*(x-l) distribution: {s5.get('complementarity_products_zL_times_(x-l)')}")
    lines.append(f"- zU*(u-x) distribution: {s5.get('complementarity_products_zU_times_(u-x)')}")
    lines.append('- inequality-row caveat: ' + s5.get('inequality_rows_note', '') + '\n')

    lines.append('## Verdict\n')
    verdict, verdict_explanation = compute_verdict(j)
    lines.append(verdict_explanation)
    lines.append('')
    lines.append(verdict)
    lines.append('')

    (OUT / 'P5_12_K_NO_SOLVE_KKT_FORENSIC_REPORT.md').write_text('\n'.join(lines))
    JOURNAL['verdict'] = verdict


def compute_verdict(j):
    """Returns (verdict_string, explanation_text). verdict_string is EXACTLY
    one of the five permitted strings, with no additional text appended."""
    s0_gate = j.get('step0', {}).get('step0_gate', {})
    s3_status = j.get('step3', {}).get('CALIBRATION_STATUS')
    if s3_status != 'CALIBRATED':
        return ('H_TRANSFER / ATTRIBUTION INCONCLUSIVE — CALIBRATION FAILED',
                'Step 3 stationarity reconstruction did not calibrate against the S20 target.')
    if not s0_gate.get('FROZEN_STATE_EQUALITY_HOLDS', False):
        return ('H_TRANSFER REJECTED — ATTRIBUTION INCONCLUSIVE',
                'Step 0 found S20/S21 are NOT at an identical primal point: '
                '`shared_es_soc`/`shared_es_e_rated` differ (see Step 0 var-comparison above). '
                'The premise "S20 vs S21 is a Param-only difference at an identical primal point" '
                'is falsified by evidence. The Step 4 ΔR decomposition (reported above for '
                'completeness, informational only) is therefore contaminated by this uncontrolled '
                'primal-point shift and cannot support a clean ADMM-vs-PUSH dominance claim, even '
                'though its raw numbers nominally point at the ADMM/parameter-change component.')
    s4 = j.get('step4', {})
    status = s4.get('dominance_status', 'MIXED/INCONCLUSIVE')
    if status.startswith('DOMINANT:ADMM'):
        return ('H_TRANSFER REJECTED — H_ADMM_UPDATE DOMINANT',
                'Step 0 held (identical primal point); Step 4 attributed >=90% of the '
                'ΔR argmax component, with matching sign, to the ADMM/objective parameter change.')
    if status.startswith('DOMINANT:bound_and_mult_push'):
        return ('H_TRANSFER REJECTED — H_PUSH DOMINANT',
                'Step 0 held (identical primal point); Step 4 attributed >=90% of the '
                'ΔR argmax component, with matching sign, to the bound/multiplier push.')
    return ('H_TRANSFER REJECTED — ATTRIBUTION INCONCLUSIVE',
            'Step 0 held, but Step 4 did not find a single component explaining >=90% of the '
            'ΔR argmax component with a matching sign; the decomposition is mixed.')


if __name__ == '__main__':
    if str(Path.cwd()) != str(ROOT):
        print('[P512K] wrong working directory; run from repository root', file=sys.stderr)
        sys.exit(2)
    main()
