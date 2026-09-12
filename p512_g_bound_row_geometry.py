"""P5.12-G -- NO-SOLVE Tier 0 + Tier 1 telemetry and local-geometry forensic.

Zero optimization solves are authorized. Both `OptSolver.solve` and
`SystemCallSolver._execute_command` are wrapped to raise immediately (see
`install_solver_guards()` / `uninstall_solver_guards()`, reused unmodified
from `p512_k_no_solve_kkt_forensic.py`); the final counters are asserted to
be 0 and reported in the journal and report.

This script performs NO production edit, NO replay, NO retry. It only reads
frozen, read-only artifacts (three IPOPT logs, three `.sol` files and the
frozen cycle-21 prepared snapshot) and writes new evidence under
`data/SRP1/Results/P512G/` (created fresh by this script; must not already
exist) plus this file itself (new, untracked).

Derivatives use Pyomo's own reverse-mode AD (`differentiate(...,
mode=Modes.reverse_numeric)`) exactly as in the reused P5.12-K helpers,
because PyNumero's ASL interface is not available in this environment.

Run:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p512_g_bound_row_geometry.py
"""
import hashlib
import json
import math
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reused, validated, READ-ONLY helpers -- do not edit p512_k_*.py.
import p512_k_no_solve_kkt_forensic as K

install_solver_guards = K.install_solver_guards
uninstall_solver_guards = K.uninstall_solver_guards
build_optim_columns = K.build_optim_columns
load_not_exported_names = K.load_not_exported_names
push_scalar = K.push_scalar
equality_violation_and_argmax = K.equality_violation_and_argmax
suffix_vector = K.suffix_vector
parse_sol_file = K.parse_sol_file
sha = K.sha

OUT = ROOT / 'data/SRP1/Results/P512G'
P512R = ROOT / 'data/SRP1/Results/P512R'
ARMA = ROOT / 'data/SRP1/Results/P512ArmA'
P512P = ROOT / 'data/SRP1/Results/P512P'

JACOBIAN_BUDGET = 6
JACOBIAN_TIME_LIMIT_S = 20 * 60

RUNS = {
    'V1': {'label': 'Variant 1 (warm_start_bound_push=1e-6)',
           'log': P512P / 'variant1_1e-6/logs/optim_log_case33_3_2025_Spring.log',
           'sol': P512P / 'variant1_1e-6/used_tmpvuihk_j5.pyomo.sol',
           'expected_log_sha256': '3c5479835849bc859f3bcaf964a2791e89318bae0fe979d2eb7dbb3e155d2bd7',
           'termination': 'Optimal', 'gate_rows': 90, 'gate_factorizations': 89,
           'gate_multitrial': 36},
    'ArmA': {'label': 'Arm A (warm_start_bound_push=1e-5)',
             'log': ARMA / 'logs/optim_log_case33_3_2025_Spring.log',
             'sol': ARMA / 'used_tmp27ntrqce.pyomo.sol',
             'expected_log_sha256': '4f66a7efeef933bdc0a425af76f0095f5c11a2112ff2c8bb6d7c03ff45409d58',
             'termination': 'maxIterations', 'gate_rows': 3001, 'gate_factorizations': 3000,
             'gate_multitrial': 27},
    'V2': {'label': 'Variant 2 (warm_start_bound_push=1e-4)',
           'log': P512P / 'variant2_1e-4/logs/optim_log_case33_3_2025_Spring.log',
           'sol': P512P / 'variant2_1e-4/used_tmpz4j9ez6a.pyomo.sol',
           'expected_log_sha256': 'ccf80716e9b7c7f7346d1404234a2b125d00cc0658f6ec8ba35ad8a63095ed2e',
           'termination': 'Optimal', 'gate_rows': 152, 'gate_factorizations': 151,
           'gate_multitrial': 70},
}

TARGETS = {
    'ArmA_terminal_mu': 1.8449144625279508e-06,
    'ArmA_terminal_violation': 6.1921343776988665e-02,
    'ArmA_safeguard_rows': 2924, 'ArmA_safeguard_first_iter': 77,
    'V2_safeguard_rows': 21, 'V2_safeguard_first_iter': 82, 'V2_safeguard_last_iter': 102,
    'ArmA_accepted_alpha_terminal': 5.42e-05,
}

JOURNAL = {
    'stage': 'P5.12-G', 'status': 'RUNNING',
    'preamble': ('NO-SOLVE Tier 0 (full-log telemetry) + Tier 1 (activity/geometry) forensic '
                 'on the frozen P5.12-P/Arm A trio. Establishes observational facts only; '
                 'no causal mechanism is established by this script alone.'),
    'zero_solver': {}, 'artifacts': {'inputs': {}, 'outputs': {}},
    'jacobian_budget': {'builds': [], 'limit': JACOBIAN_BUDGET, 'time_limit_s': JACOBIAN_TIME_LIMIT_S},
}


# ===========================================================================
# utilities
# ===========================================================================
def record_input(label, path):
    path = Path(path)
    JOURNAL['artifacts']['inputs'][label] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    return JOURNAL['artifacts']['inputs'][label]['sha256']


def record_output(label, path):
    path = Path(path)
    JOURNAL['artifacts']['outputs'][label] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def dump_json(path, obj):
    path = Path(path)
    with path.open('x') as f:
        json.dump(obj, f, indent=1, default=str)


def read_only_git(*args):
    return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True, text=True, check=True).stdout


def repo_state():
    head = read_only_git('rev-parse', 'HEAD').strip()
    branch = read_only_git('branch', '--show-current').strip()
    status = read_only_git('status', '--porcelain', '--untracked-files=no')
    return {'head': head, 'branch': branch, 'status_porcelain_tracked': status}


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


# ===========================================================================
# TIER 0 -- full log telemetry parser
# ===========================================================================
_SUMMARY_LINE = re.compile(r'^\s*(\d+)(r?)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*([a-zA-Z]*)\s*$')
_ALPHA_LETTER = re.compile(r'^([\d.eE+-]+)([a-zA-Z]*)$')


def parse_log_full(log_path):
    """Full-trajectory parser. Returns a dict with per-iteration summary rows,
    per-iteration factorization records, per-iteration line-search records,
    safeguard events, and scalar/summary facts."""
    text = log_path.read_text(errors='replace')
    lines = text.splitlines()

    summary_rows = []          # one per 'Summary of Iteration' table row
    factor_records = {}        # iter -> dict
    linesearch_records = {}    # iter -> dict
    safeguard_events = []      # list of {'iter':, 'kind': 'zL'|'zU', 'correction':}
    beginning_norms = {}       # iter -> dict of curr_* norms + mu + tau
    barrier_records = {}       # iter -> {'optimality_error':, 'mu':}

    n = len(lines)
    cur_iter_for_solve = None
    cur_iter_for_ls = None
    cur_iter_for_begin = None
    cur_iter_for_barrier = None
    i = 0
    while i < n:
        line = lines[i]

        m_hdr = re.match(r'^\*\*\* Summary of Iteration:\s*(\d+):', line)
        if m_hdr:
            # data row is two lines below (blank column header line above it)
            j = i + 1
            while j < n and not re.match(r'^\s*\d+r?\s+[\d.eE+-]', lines[j]):
                j += 1
                if j - i > 6:
                    break
            if j < n and re.match(r'^\s*\d+r?\s+[\d.eE+-]', lines[j]):
                parts = lines[j].split()
                try:
                    it_raw = parts[0]
                    it = int(it_raw.rstrip('r'))
                    restoration = it_raw.endswith('r')
                    objective = float(parts[1]); inf_pr = float(parts[2]); inf_du = float(parts[3])
                    lg_mu = None if parts[4] == '-' else float(parts[4])
                    d_norm = float(parts[5])
                    lg_rg = None if parts[6] == '-' else float(parts[6])
                    alpha_du = float(parts[7])
                    m_alpha = _ALPHA_LETTER.match(parts[8])
                    alpha_pr = float(m_alpha.group(1)); step_type = m_alpha.group(2)
                    ls_count = int(parts[9])
                    marker = parts[10] if len(parts) > 10 else ''
                    summary_rows.append({
                        'iter': it, 'restoration': restoration, 'objective': objective,
                        'inf_pr': inf_pr, 'inf_du': inf_du, 'lg_mu': lg_mu, 'd_norm': d_norm,
                        'lg_rg': lg_rg, 'alpha_du': alpha_du, 'alpha_pr': alpha_pr,
                        'step_type': step_type, 'ls_count': ls_count, 'marker': marker,
                    })
                except (ValueError, IndexError):
                    pass
            i = j + 1
            continue

        m_solve = re.match(r'^\*\*\* Solving the Primal Dual System for Iteration\s*(\d+):', line)
        if m_solve:
            cur_iter_for_solve = int(m_solve.group(1))
            factor_records[cur_iter_for_solve] = {
                'ma97_events': [], 'n_trials': None, 'delta_x': None, 'delta_s': None,
                'delta_c': None, 'delta_d': None, 'residual_ratios': [], 'factorization_failed_count': 0,
            }
            i += 1
            continue

        if cur_iter_for_solve is not None and line.startswith('HSL_MA97: delays'):
            m_ma = re.match(r'HSL_MA97: delays (\d+), nfactor (\d+), nflops (\d+), maxfront (\d+)', line)
            if m_ma:
                factor_records[cur_iter_for_solve]['ma97_events'].append(
                    {'delays': int(m_ma.group(1)), 'nfactor': int(m_ma.group(2)),
                     'nflops': int(m_ma.group(3)), 'maxfront': int(m_ma.group(4))})
            i += 1
            continue

        if cur_iter_for_solve is not None and 'Factorization failed' in line:
            factor_records[cur_iter_for_solve]['factorization_failed_count'] += 1
            i += 1
            continue

        m_trials = re.match(r'^Number of trial factorizations performed:\s*(\d+)', line)
        if m_trials and cur_iter_for_solve is not None:
            factor_records[cur_iter_for_solve]['n_trials'] = int(m_trials.group(1))
            i += 1
            continue

        m_pert = re.match(r'^Perturbation parameters: delta_x=([\d.eE+-]+) delta_s=([\d.eE+-]+)', line)
        if m_pert and cur_iter_for_solve is not None:
            rec = factor_records[cur_iter_for_solve]
            rec['delta_x'] = float(m_pert.group(1))
            rec['delta_s'] = float(m_pert.group(2))
            if i + 1 < n:
                m_pert2 = re.match(r'\s*delta_c=([\d.eE+-]+) delta_d=([\d.eE+-]+)', lines[i + 1])
                if m_pert2:
                    rec['delta_c'] = float(m_pert2.group(1))
                    rec['delta_d'] = float(m_pert2.group(2))
                    i += 1
            i += 1
            continue

        m_resratio = re.match(r'^residual_ratio = ([\d.eE+-]+)', line)
        if m_resratio and cur_iter_for_solve is not None:
            factor_records[cur_iter_for_solve]['residual_ratios'].append(float(m_resratio.group(1)))
            i += 1
            continue

        m_ls_hdr = re.match(r'^\*\*\* Finding Acceptable Trial Point for Iteration\s*(\d+):', line)
        if m_ls_hdr:
            cur_iter_for_ls = int(m_ls_hdr.group(1))
            linesearch_records[cur_iter_for_ls] = {
                'filter_entries': None, 'alpha_min': None, 'first_alpha_primal_check': None,
                'trial_alphas': [], 'n_trials': 0, 'sufficient_reduction_pass': [], 'filter_pass': [],
                'restoration_entered': False, 'watchdog_reset': False,
            }
            i += 1
            continue

        if cur_iter_for_ls is not None:
            m_filt = re.match(r'^The current filter has (\d+) entries\.', line)
            if m_filt:
                linesearch_records[cur_iter_for_ls]['filter_entries'] = int(m_filt.group(1))
                i += 1
                continue
            m_amin = re.match(r'^minimal step size ALPHA_MIN = ([\d.eE+-]+)', line)
            if m_amin:
                linesearch_records[cur_iter_for_ls]['alpha_min'] = float(m_amin.group(1))
                i += 1
                continue
            m_start = re.match(r'^Starting checks for alpha \(primal\) = ([\d.eE+-]+)', line)
            if m_start:
                if linesearch_records[cur_iter_for_ls]['first_alpha_primal_check'] is None:
                    linesearch_records[cur_iter_for_ls]['first_alpha_primal_check'] = float(m_start.group(1))
                i += 1
                continue
            m_trial = re.match(r'^Checking acceptability for trial step size alpha_primal_test=\s*([\d.eE+-]+):', line)
            if m_trial:
                linesearch_records[cur_iter_for_ls]['trial_alphas'].append(float(m_trial.group(1)))
                linesearch_records[cur_iter_for_ls]['n_trials'] += 1
                i += 1
                continue
            if line.startswith('Checking sufficient reduction'):
                # next non-blank line indicates outcome
                j = i + 1
                while j < n and lines[j].strip() == '':
                    j += 1
                ok = j < n and lines[j].strip().startswith('Succeeded')
                linesearch_records[cur_iter_for_ls]['sufficient_reduction_pass'].append(ok)
                i = j + 1
                continue
            if line.startswith('Checking filter acceptability'):
                j = i + 1
                while j < n and lines[j].strip() == '':
                    j += 1
                ok = j < n and lines[j].strip().startswith('Succeeded')
                linesearch_records[cur_iter_for_ls]['filter_pass'].append(ok)
                i = j + 1
                continue
            if 'Mu has changed in line search' in line:
                linesearch_records[cur_iter_for_ls]['watchdog_reset'] = True
                i += 1
                continue
            if 'Starting Restoration Phase' in line or 'Restoration phase' in line:
                linesearch_records[cur_iter_for_ls]['restoration_entered'] = True

        m_safe = re.match(r'^Some value in (z_L|z_U) becomes too large - maximal correction = ([\d.eE+-]+)', line)
        if m_safe:
            kind = 'zL' if m_safe.group(1) == 'z_L' else 'zU'
            it_ctx = cur_iter_for_ls if cur_iter_for_ls is not None else cur_iter_for_solve
            safeguard_events.append({'iter': it_ctx, 'kind': kind, 'correction': float(m_safe.group(2))})
            i += 1
            continue

        m_begin = re.match(r'^\*\*\* Beginning Iteration\s*(\d+) from the following point:', line)
        if m_begin:
            cur_iter_for_begin = int(m_begin.group(1))
            beginning_norms[cur_iter_for_begin] = {}
            i += 1
            continue
        if cur_iter_for_begin is not None:
            m_mu = re.match(r'^Current barrier parameter mu = ([\d.eE+-]+)', line)
            if m_mu:
                beginning_norms[cur_iter_for_begin]['mu'] = float(m_mu.group(1))
                i += 1
                continue
            m_tau = re.match(r'^Current fraction-to-the-boundary parameter tau = ([\d.eE+-]+)', line)
            if m_tau:
                beginning_norms[cur_iter_for_begin]['tau'] = float(m_tau.group(1))
                i += 1
                continue
            m_norm = re.match(r'^\|\|curr_(\w+)\|\|_inf\s*=\s*([\d.eE+-]+)', line)
            if m_norm:
                beginning_norms[cur_iter_for_begin][f'curr_{m_norm.group(1)}_inf'] = float(m_norm.group(2))
                i += 1
                continue

        m_barr_hdr = re.match(r'^\*\*\* Update Barrier Parameter for Iteration\s*(\d+):', line)
        if m_barr_hdr:
            cur_iter_for_barrier = int(m_barr_hdr.group(1))
            barrier_records[cur_iter_for_barrier] = {}
            i += 1
            continue
        if cur_iter_for_barrier is not None:
            m_opt = re.match(r'^Optimality Error for Barrier Sub-problem = ([\d.eE+-]+)', line)
            if m_opt:
                barrier_records[cur_iter_for_barrier]['optimality_error'] = float(m_opt.group(1))
                i += 1
                continue
            m_bp = re.match(r'^Barrier Parameter:\s*([\d.eE+-]+)', line)
            if m_bp:
                barrier_records[cur_iter_for_barrier]['barrier_parameter'] = float(m_bp.group(1))
                i += 1
                continue

        i += 1

    # ---- termination / scalar facts ----
    termination = None
    m_exit = re.search(r'^EXIT: (.+)$', text, re.MULTILINE)
    if m_exit:
        termination = m_exit.group(1).strip()
    n_iters_match = re.search(r'Number of Iterations\.+:\s*(\d+)', text)
    n_iterations = int(n_iters_match.group(1)) if n_iters_match else None

    final_block = re.findall(
        r'Dual infeasibility\.+:\s+([\d.eE+-]+)\s+([\d.eE+-]+)\n'
        r'Constraint violation\.+:\s+([\d.eE+-]+)\s+([\d.eE+-]+)',
        text)
    final_dual_inf_unscaled = float(final_block[-1][1]) if final_block else None
    final_violation_unscaled = float(final_block[-1][3]) if final_block else None

    complementarity_final = re.findall(r'Complementarity\.+:\s+([\d.eE+-]+)\s+([\d.eE+-]+)', text)
    final_complementarity_unscaled = float(complementarity_final[-1][1]) if complementarity_final else None

    return {
        'summary_rows': summary_rows, 'factor_records': factor_records,
        'linesearch_records': linesearch_records, 'safeguard_events': safeguard_events,
        'beginning_norms': beginning_norms, 'barrier_records': barrier_records,
        'termination': termination, 'n_iterations_reported': n_iterations,
        'final_dual_inf_unscaled': final_dual_inf_unscaled,
        'final_violation_unscaled': final_violation_unscaled,
        'final_complementarity_unscaled': final_complementarity_unscaled,
        'n_summary_rows': len(summary_rows), 'n_factor_events': len(factor_records),
        'n_multitrial_events': sum(1 for r in factor_records.values() if (r['n_trials'] or 1) > 1),
    }


def tier0_analysis():
    parsed = {}
    hash_checks = {}
    for key, cfg in RUNS.items():
        actual = record_input(f'{key}_log', cfg['log'])
        hash_checks[key] = {'expected': cfg['expected_log_sha256'], 'actual': actual,
                             'match': actual == cfg['expected_log_sha256']}
        t0 = time.time()
        parsed[key] = parse_log_full(cfg['log'])
        parsed[key]['parse_wall_seconds'] = time.time() - t0
        print(f"[P512G] parsed {key}: {parsed[key]['n_summary_rows']} summary rows, "
              f"{parsed[key]['n_factor_events']} factor events, "
              f"{parsed[key]['parse_wall_seconds']:.2f}s", flush=True)

    gates = {}
    for key, cfg in RUNS.items():
        p = parsed[key]
        gates[key] = {
            'n_summary_rows': {'actual': p['n_summary_rows'], 'expected': cfg['gate_rows'],
                                'match': p['n_summary_rows'] == cfg['gate_rows']},
            'termination': {'actual': p['termination'], 'expected': cfg['termination'],
                             'match': (p['termination'] or '').lower().replace(' ', '') ==
                                      cfg['termination'].lower().replace(' ', '') or
                                      (cfg['termination'] == 'maxIterations' and
                                       'Maximum Number of Iterations' in (p['termination'] or ''))},
            'n_factor_events': {'actual': p['n_factor_events'], 'expected': cfg['gate_factorizations'],
                                 'match': p['n_factor_events'] == cfg['gate_factorizations']},
            'n_multitrial_events': {'actual': p['n_multitrial_events'], 'expected': cfg['gate_multitrial'],
                                     'match': p['n_multitrial_events'] == cfg['gate_multitrial']},
        }

    # Arm A safeguard gate: rows with a 'z' marker in the summary table (matches the
    # historical P5.12-K/T z-marker convention), plus the raw safeguard-message count.
    armA_z_rows = [r for r in parsed['ArmA']['summary_rows'] if 'z' in r['marker']]
    armA_first_z = armA_z_rows[0]['iter'] if armA_z_rows else None
    v2_z_rows = [r for r in parsed['V2']['summary_rows'] if 'z' in r['marker']]
    v1_z_rows = [r for r in parsed['V1']['summary_rows'] if 'z' in r['marker']]
    v2_first_z = v2_z_rows[0]['iter'] if v2_z_rows else None
    v2_last_z = v2_z_rows[-1]['iter'] if v2_z_rows else None

    safeguard_gate = {
        'ArmA_n_z_rows': {'actual': len(armA_z_rows), 'expected': TARGETS['ArmA_safeguard_rows'],
                           'match': len(armA_z_rows) == TARGETS['ArmA_safeguard_rows']},
        'ArmA_first_z_iter': {'actual': armA_first_z, 'expected': TARGETS['ArmA_safeguard_first_iter'],
                               'match': armA_first_z == TARGETS['ArmA_safeguard_first_iter']},
        'V2_n_z_rows': {'actual': len(v2_z_rows), 'expected': TARGETS['V2_safeguard_rows'],
                         'match': len(v2_z_rows) == TARGETS['V2_safeguard_rows']},
        'V2_first_z_iter': {'actual': v2_first_z, 'expected': TARGETS['V2_safeguard_first_iter'],
                             'match': v2_first_z == TARGETS['V2_safeguard_first_iter']},
        'V2_last_z_iter': {'actual': v2_last_z, 'expected': TARGETS['V2_safeguard_last_iter'],
                            'match': v2_last_z == TARGETS['V2_safeguard_last_iter']},
        'V1_n_z_rows': len(v1_z_rows),
    }

    armA_terminal_mu_match = (parsed['ArmA']['beginning_norms'].get(3000, {}).get('mu') is not None and
                               abs(parsed['ArmA']['beginning_norms'][3000]['mu'] - TARGETS['ArmA_terminal_mu']) < 1e-12)
    terminal_gate = {
        'ArmA_final_violation_unscaled': parsed['ArmA']['final_violation_unscaled'],
        'ArmA_final_violation_target': TARGETS['ArmA_terminal_violation'],
        'ArmA_final_violation_match': (parsed['ArmA']['final_violation_unscaled'] is not None and
                                        abs(parsed['ArmA']['final_violation_unscaled'] - TARGETS['ArmA_terminal_violation']) < 1e-9),
        'ArmA_terminal_mu_from_beginning_block': parsed['ArmA']['beginning_norms'].get(3000, {}).get('mu'),
        'ArmA_terminal_mu_target': TARGETS['ArmA_terminal_mu'], 'ArmA_terminal_mu_match': armA_terminal_mu_match,
    }

    # ---- delta_c / delta_x analysis ----
    deltas = {}
    for key in RUNS:
        fr = parsed[key]['factor_records']
        dc_nonzero = [(it, r['delta_c']) for it, r in fr.items() if (r['delta_c'] or 0.0) != 0.0]
        dx_nonzero = [(it, r['delta_x']) for it, r in fr.items() if (r['delta_x'] or 0.0) != 0.0]
        deltas[key] = {
            'n_events_total': len(fr),
            'n_delta_c_nonzero': len(dc_nonzero), 'delta_c_nonzero_iters': sorted(it for it, _ in dc_nonzero),
            'n_delta_x_nonzero': len(dx_nonzero), 'delta_x_nonzero_iters': sorted(it for it, _ in dx_nonzero),
            'delta_x_nonzero_values': sorted(set(v for _, v in dx_nonzero)),
        }
    deltas['ArmA']['delta_c_zero_throughout'] = (deltas['ArmA']['n_delta_c_nonzero'] == 0)
    deltas['ArmA']['delta_x_events_during_stall_iter_ge_77'] = sorted(
        it for it in deltas['ArmA']['delta_x_nonzero_iters'] if it >= 77)
    deltas['V1']['delta_c_zero_throughout'] = (deltas['V1']['n_delta_c_nonzero'] == 0)
    deltas['V2']['delta_c_zero_throughout'] = (deltas['V2']['n_delta_c_nonzero'] == 0)

    # ---- step-truncation / first-trial-acceptance evidence ----
    step_trunc = {}
    for key in RUNS:
        ls = parsed[key]['linesearch_records']
        n_multi_trial = sum(1 for r in ls.values() if r['n_trials'] > 1)
        single_trial_iters = [it for it, r in ls.items() if r['n_trials'] == 1]
        step_trunc[key] = {
            'n_linesearch_events': len(ls), 'n_multi_trial_linesearch_events': n_multi_trial,
            'n_single_trial_linesearch_events': len(single_trial_iters),
        }
    # Arm A terminal (iterations 2990-3000) first-trial-acceptance check
    armA_terminal_ls = {it: parsed['ArmA']['linesearch_records'].get(it) for it in range(2990, 3000)}
    armA_terminal_check = []
    for it, rec in sorted(armA_terminal_ls.items()):
        if rec is None:
            continue
        armA_terminal_check.append({
            'iter': it, 'n_trials': rec['n_trials'],
            'first_alpha_primal_check': rec['first_alpha_primal_check'],
            'accepted_alpha': rec['trial_alphas'][-1] if rec['trial_alphas'] else None,
            'sufficient_reduction_pass_all': all(rec['sufficient_reduction_pass']) if rec['sufficient_reduction_pass'] else None,
            'filter_pass_all': all(rec['filter_pass']) if rec['filter_pass'] else None,
            'alpha_min': rec['alpha_min'],
        })
    step_trunc['ArmA_terminal_detail'] = armA_terminal_check
    last = armA_terminal_check[-1] if armA_terminal_check else {}
    step_trunc['ArmA_first_trial_acceptance_claim_confirmed'] = (
        last.get('n_trials') == 1 and
        last.get('accepted_alpha') is not None and
        abs(last.get('accepted_alpha', 0) - TARGETS['ArmA_accepted_alpha_terminal']) < 1e-6 and
        last.get('sufficient_reduction_pass_all') is True and last.get('filter_pass_all') is True and
        last.get('alpha_min') is not None and last.get('alpha_min') < 1e-8)
    step_trunc['note'] = ('"Starting checks for alpha (primal)" is the FIRST/maximum trial alpha the '
                           'line search attempts (i.e. the fraction-to-boundary-limited maximum primal '
                           'step at that iterate); when n_trials==1 this value equals the accepted alpha, '
                           'so a single line-search trial that succeeds both checks means the step was '
                           'never backtracked -- it was truncated by fraction-to-boundary/step-direction '
                           'magnitude, not rejected by the line search. The actual per-variable blocking '
                           'bound is NOT printed anywhere in these logs; only the aggregate primal alpha '
                           'is available. See Tier 1 for the near-bound candidate set.')

    # ---- safeguard magnitude quantification ----
    safeguard_mag = {}
    for key in RUNS:
        events = parsed[key]['safeguard_events']
        corr = [e['correction'] for e in events]
        safeguard_mag[key] = {
            'n_events': len(events),
            'first_iter': events[0]['iter'] if events else None,
            'last_iter': events[-1]['iter'] if events else None,
            'min_correction': min(corr) if corr else None, 'max_correction': max(corr) if corr else None,
            'mean_correction': (sum(corr) / len(corr)) if corr else None,
            'n_zL': sum(1 for e in events if e['kind'] == 'zL'),
            'n_zU': sum(1 for e in events if e['kind'] == 'zU'),
        }
    armA_corr = [e['correction'] for e in parsed['ArmA']['safeguard_events']]
    safeguard_mag['ArmA_magnitude_in_3.4e-8_to_3.9e-8_band'] = (
        all(3.0e-8 <= c <= 4.0e-8 for c in armA_corr) if armA_corr else None)
    safeguard_mag['materiality_note'] = (
        'Corrections of order 1e-8 applied to z_U/z_L (whose ||.||_inf is O(1e2) in this run) are '
        '~10 orders below the multiplier magnitude and far below mu (1.8e-6) and the accepted primal '
        'step (5.4e-5); they cannot plausibly explain a 1.5e-1 feasibility excursion or a 3000-iteration '
        'stall by direct magnitude. This is a numerical-magnitude argument only, not a proof of full '
        'causal irrelevance.')

    return {
        'hash_checks': hash_checks, 'all_hashes_match': all(v['match'] for v in hash_checks.values()),
        'gates': gates, 'safeguard_gate': safeguard_gate, 'terminal_gate': terminal_gate,
        'deltas': deltas, 'step_truncation': step_trunc, 'safeguard_magnitude': safeguard_mag,
    }, parsed


# ===========================================================================
# TIER 1 -- activity and geometry forensic
# ===========================================================================
ACTIVE_THRESH = 1e-6
NEAR_ACTIVE_THRESH = 1e-4
LADDER = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
DUAL_ACTIVE_REL = 1e-8
ROW_NORM_LADDER = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]  # absolute raw-row-norm ladder, predeclared
SCALED_ROW_NORM_LADDER = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  # scaled-row-norm ladder, predeclared
GRADIENT_SCALING_KAPPA = 100.0  # IPOPT default nlp_scaling_max_gradient


def num(x):
    try:
        return float(pe.value(x, exception=False))
    except Exception:
        return None


def load_state_values(model, mapping, sol_data):
    """Map a parsed .sol primal array onto {var_name: value} using the frozen
    (identical across all 3 runs) symbol map."""
    sym_to_name = dict(mapping['mapping'])
    var_syms = sorted((s for s in sym_to_name if s.startswith('v')), key=lambda s: int(s[1:]))
    values = {}
    for i, sym in enumerate(var_syms):
        values[sym_to_name[sym]] = sol_data['primals'][i]
    return values, var_syms, sym_to_name


def load_state_duals(mapping, sol_data):
    sym_to_name = dict(mapping['mapping'])
    con_syms = sorted((s for s in sym_to_name if s.startswith('c')), key=lambda s: int(s[1:]))
    duals = {}
    for i, sym in enumerate(con_syms):
        duals[sym_to_name[sym]] = sol_data['duals'][i]
    return duals, con_syms


def set_model_values(model, optim_vars, values_by_name):
    for v in optim_vars:
        if v.name in values_by_name:
            v.set_value(values_by_name[v.name], skip_validation=True)


def constraint_activity_snapshot(model):
    """One pass over all active Pyomo constraints: returns per-constraint body,
    lower, upper, equality flag, and (for inequalities) raw slack-to-nearest-bound."""
    rows = {}
    for c in model.component_data_objects(pe.Constraint, active=True):
        body = num(c.body)
        lo = num(c.lower)
        hi = num(c.upper)
        rows[c.name] = {'body': body, 'lower': lo, 'upper': hi, 'equality': c.equality}
    return rows


def classify_activity(rows):
    """Given a constraint-row snapshot, classify each inequality row's slack at
    every LADDER threshold; equality rows are always 'EQUALITY' (treated as
    active by definition)."""
    out = {}
    for name, r in rows.items():
        if r['equality']:
            out[name] = {'kind': 'equality', 'slack': None,
                          'classes': {str(t): 'ACTIVE' for t in LADDER}}
            continue
        body = r['body']
        slacks = []
        if r['lower'] is not None:
            slacks.append(body - r['lower'])
        if r['upper'] is not None:
            slacks.append(r['upper'] - body)
        slack = min(slacks) if slacks else None
        classes = {}
        for t in LADDER:
            if slack is None:
                classes[str(t)] = 'UNBOUNDED'
            elif slack <= t:
                classes[str(t)] = 'ACTIVE'
            elif slack <= NEAR_ACTIVE_THRESH:
                classes[str(t)] = 'NEAR-ACTIVE'
            else:
                classes[str(t)] = 'INACTIVE'
        out[name] = {'kind': 'inequality', 'slack': slack, 'classes': classes,
                     'active_primary': (slack is not None and slack <= ACTIVE_THRESH),
                     'near_active_primary': (slack is not None and ACTIVE_THRESH < slack <= NEAR_ACTIVE_THRESH)}
    return out


def row_jacobian_norms(model, row_names, budget_state, label):
    if len(budget_state['builds']) >= JACOBIAN_BUDGET:
        raise RuntimeError(f'Jacobian build budget ({JACOBIAN_BUDGET}) exceeded, refusing build for {label}')
    t0 = time.time()
    norms = {}
    supports = {}
    name_to_con = {c.name: c for c in model.component_data_objects(pe.Constraint, active=True)}
    n_done = 0
    for name in row_names:
        c = name_to_con.get(name)
        if c is None:
            continue
        vs = list(identify_variables(c.body, include_fixed=False))
        if not vs:
            norms[name] = 0.0
            supports[name] = {'support_size': 0, 'min_abs_coef': None, 'max_abs_coef': None}
            continue
        grads = differentiate(c.body, wrt_list=vs, mode=Modes.reverse_numeric)
        gvals = [abs(float(g)) for g in grads if g is not None]
        norms[name] = max(gvals) if gvals else 0.0
        supports[name] = {'support_size': len(vs),
                           'min_abs_coef': min(gvals) if gvals else None,
                           'max_abs_coef': max(gvals) if gvals else None}
        n_done += 1
    elapsed = time.time() - t0
    rec = {'label': label, 'n_rows_requested': len(row_names), 'n_rows_computed': n_done,
           'wall_seconds': elapsed, 'over_budget_time': elapsed > JACOBIAN_TIME_LIMIT_S}
    budget_state['builds'].append(rec)
    print(f'[P512G] Jacobian build #{len(budget_state["builds"])} ({label}): '
          f'{n_done} rows, {elapsed:.3f}s', flush=True)
    if rec['over_budget_time']:
        raise TimeoutError(f'Jacobian build for {label} exceeded {JACOBIAN_TIME_LIMIT_S}s budget')
    return norms, supports


FAMILIES_OF_INTEREST = ['node_balance_p', 'node_balance_q', 'pij_def', 'pji_def', 'sess_comp',
                         'branch_flow_limit']


def row_family(name):
    return re.split(r'\[', name)[0]


def tier1_analysis(tier0_parsed):
    budget_state = JOURNAL['jacobian_budget']
    result = {}

    # ---- load snapshot model (cycle21_prepared) ----
    snap_sha = record_input('snapshot_pkl', P512R / 'cycle21_prepared/snapshot.pkl')
    ordered_state_sha = record_input('ordered_state_json', P512R / 'cycle21_prepared/ordered_state.json')
    mapping_sha = record_input('original_mapping_json', P512R / 'cycle21_prepared/original_mapping.json')
    result['input_hashes'] = {'snapshot.pkl': snap_sha, 'ordered_state.json': ordered_state_sha,
                               'original_mapping.json': mapping_sha,
                               'snapshot_semantic_digest_expected': '3fd294d7b41ad3b2b62a140cb518a9ecbe32a0218eed8d6cd788fc027edda85f'}

    with (P512R / 'cycle21_prepared/snapshot.pkl').open('rb') as f:
        snap = pickle.load(f)
    model = snap['model']
    semantic_digest = K.H.digest(K.H.model_state(model))
    result['snapshot_semantic_digest_actual'] = semantic_digest
    result['snapshot_semantic_digest_match'] = (
        semantic_digest == result['input_hashes']['snapshot_semantic_digest_expected'])

    mapping = json.loads((P512R / 'cycle21_prepared/original_mapping.json').read_text())
    not_exported = load_not_exported_names()

    optim_vars, name_to_idx, eq_bound_names = build_optim_columns(model, not_exported)
    n_lower_only = sum(1 for v in optim_vars if v.lb is not None and v.ub is None)
    n_two_sided = sum(1 for v in optim_vars if v.lb is not None and v.ub is not None)
    n_upper_only = sum(1 for v in optim_vars if v.lb is None and v.ub is not None)
    result['column_gate'] = {
        'n_optim_vars': len(optim_vars), 'expected_n': 9166, 'match_n': len(optim_vars) == 9166,
        'n_lower_only': n_lower_only, 'expected_lower_only': 124, 'match_lower_only': n_lower_only == 124,
        'n_two_sided': n_two_sided, 'expected_two_sided': 7266, 'match_two_sided': n_two_sided == 7266,
        'n_upper_only': n_upper_only, 'n_equal_bound_excluded': len(eq_bound_names),
        'expected_equal_bound_excluded': 102,
        'match_equal_bound_excluded': len(eq_bound_names) == 102,
    }

    n_ad_failures = {'count': 0, 'events': []}

    # ---- S0 raw values (as loaded, i.e. common frozen start, before any push) ----
    s0_values = {v.name: v.value for v in optim_vars}

    # ---- SA / S1 / S2 from .sol files (identical symbol map, verified) ----
    sol_data = {}
    for key in ('ArmA', 'V1', 'V2'):
        sha_actual = record_input(f'{key}_sol', RUNS[key]['sol'])
        sol_data[key] = parse_sol_file(RUNS[key]['sol'])
        sol_data[key]['_sha256'] = sha_actual

    values_by_state = {'S0': s0_values}
    duals_by_state = {}
    zL_by_state = {}
    zU_by_state = {}
    var_syms_ref = None
    con_syms_ref = None
    for key, label in (('ArmA', 'SA'), ('V1', 'S1'), ('V2', 'S2')):
        vals, var_syms, sym_to_name = load_state_values(model, mapping, sol_data[key])
        duals, con_syms = load_state_duals(mapping, sol_data[key])
        if var_syms_ref is None:
            var_syms_ref = var_syms
            con_syms_ref = con_syms
        values_by_state[label] = vals
        duals_by_state[label] = duals
        zL_out = sol_data[key]['suffixes'].get('ipopt_zL_out', {})
        zU_out = sol_data[key]['suffixes'].get('ipopt_zU_out', {})
        zL_by_state[label] = {sym_to_name[f'v{i}']: v for i, v in zL_out.items()}
        zU_by_state[label] = {sym_to_name[f'v{i}']: v for i, v in zU_out.items()}

    result['state_labels'] = {'S0': 'common frozen cycle-21 start (raw, pre-push)',
                               'SA': 'Arm A terminal stalled iterate (maxIterations)',
                               'S1': 'Variant 1 converged terminal (Optimal)',
                               'S2': 'Variant 2 converged terminal (Optimal)'}

    result['confound_note'] = {
        'SA_final_violation_unscaled': tier0_parsed['ArmA']['final_violation_unscaled'],
        'S1_final_violation_unscaled': tier0_parsed['V1']['final_violation_unscaled'],
        'S2_final_violation_unscaled': tier0_parsed['V2']['final_violation_unscaled'],
        'note': ('SA is INFEASIBLE at termination (unscaled constraint violation '
                 f"{tier0_parsed['ArmA']['final_violation_unscaled']}) while S1 "
                 f"({tier0_parsed['V1']['final_violation_unscaled']}) and S2 "
                 f"({tier0_parsed['V2']['final_violation_unscaled']}) are feasible to solver tolerance. "
                 'Any Arm-A-specific active-set finding below may partly reflect that SA is a failed '
                 'iterate rather than a stationary point; positive geometric results below are '
                 'mechanism CANDIDATES only, not established causes.'),
    }

    # ---- per-state constraint activity + bound geometry ----
    saved_values = {v.name: v.value for v in optim_vars}
    activity_by_state = {}
    row_snapshot_by_state = {}
    for label in ('S0', 'SA', 'S1', 'S2'):
        set_model_values(model, optim_vars, values_by_state[label])
        rows = constraint_activity_snapshot(model)
        row_snapshot_by_state[label] = rows
        activity_by_state[label] = classify_activity(rows)
    set_model_values(model, optim_vars, saved_values)  # restore

    # union of rows active/near-active (primary thresholds) in ANY state, plus all equality rows
    all_eq_rows = [n for n, r in row_snapshot_by_state['S0'].items() if r['equality']]
    active_union = set(all_eq_rows)
    for label in ('S0', 'SA', 'S1', 'S2'):
        for name, a in activity_by_state[label].items():
            if a['kind'] == 'inequality' and (a.get('active_primary') or a.get('near_active_primary')):
                active_union.add(name)

    # ---- full-row Jacobian norm builds, one per state (4 builds; budget 6) ----
    all_row_names = list(row_snapshot_by_state['S0'].keys())
    row_norms_by_state = {}
    row_supports_by_state = {}
    for label in ('S0', 'SA', 'S1', 'S2'):
        set_model_values(model, optim_vars, values_by_state[label])
        try:
            norms, supports = row_jacobian_norms(model, all_row_names, budget_state, f'{label} (all rows)')
        except Exception as exc:
            n_ad_failures['count'] += 1
            n_ad_failures['events'].append({'state': label, 'error': f'{type(exc).__name__}: {exc}'})
            norms, supports = {}, {}
        row_norms_by_state[label] = norms
        row_supports_by_state[label] = supports
    set_model_values(model, optim_vars, saved_values)  # restore

    # ---- approximate IPOPT gradient-based scaling, reconstructed at S0 ----
    scale_by_row = {}
    for name, norm_val in row_norms_by_state['S0'].items():
        scale_by_row[name] = min(1.0, GRADIENT_SCALING_KAPPA / norm_val) if norm_val > 0 else 1.0

    def scaled_norm(state, name):
        raw = row_norms_by_state.get(state, {}).get(name)
        if raw is None:
            return None
        return raw * scale_by_row.get(name, 1.0)

    # ---- anomalously-small-gradient scan (predeclared ladder, no post-hoc tuning) ----
    row_geometry = {}
    for name in active_union:
        entry = {'family': row_family(name), 'kind': row_snapshot_by_state['S0'][name]['equality'] and 'equality' or 'inequality'}
        for label in ('S0', 'SA', 'S1', 'S2'):
            raw = row_norms_by_state[label].get(name)
            sc = scaled_norm(label, name)
            sup = row_supports_by_state[label].get(name, {})
            entry[label] = {
                'raw_row_norm_inf': raw, 'scaled_row_norm_inf_approx': sc,
                'support_size': sup.get('support_size'), 'min_abs_coef': sup.get('min_abs_coef'),
                'max_abs_coef': sup.get('max_abs_coef'),
                'primal_activity': activity_by_state[label].get(name, {}).get('kind') == 'equality'
                    and 'EQUALITY' or activity_by_state[label].get(name, {}).get('classes', {}).get(str(ACTIVE_THRESH)),
                'slack': activity_by_state[label].get(name, {}).get('slack'),
            }
        row_geometry[name] = entry

    raw_below = {}
    scaled_below = {}
    for t in ROW_NORM_LADDER:
        raw_below[str(t)] = {}
        for label in ('S0', 'SA', 'S1', 'S2'):
            names = sorted(n for n in row_geometry if (row_geometry[n][label]['raw_row_norm_inf'] or 0) <= t)
            raw_below[str(t)][label] = {'count': len(names), 'names_sample': names[:20]}
    for t in SCALED_ROW_NORM_LADDER:
        scaled_below[str(t)] = {}
        for label in ('S0', 'SA', 'S1', 'S2'):
            names = sorted(n for n in row_geometry
                            if row_geometry[n][label]['scaled_row_norm_inf_approx'] is not None
                            and row_geometry[n][label]['scaled_row_norm_inf_approx'] <= t)
            scaled_below[str(t)][label] = {'count': len(names), 'names_sample': names[:20]}

    # Arm-A-unique small-gradient rows at the primary scaled threshold 1e-3, vs both controls
    # (full sets, not just the printed samples, are used for this comparison)
    sa_small_full = set(n for n in row_geometry if (row_geometry[n]['SA']['scaled_row_norm_inf_approx'] or 1e9) <= 1e-3)
    s1_small_full = set(n for n in row_geometry if (row_geometry[n]['S1']['scaled_row_norm_inf_approx'] or 1e9) <= 1e-3)
    s2_small_full = set(n for n in row_geometry if (row_geometry[n]['S2']['scaled_row_norm_inf_approx'] or 1e9) <= 1e-3)
    s0_small_full = set(n for n in row_geometry if (row_geometry[n]['S0']['scaled_row_norm_inf_approx'] or 1e9) <= 1e-3)
    armA_unique_small = sorted(sa_small_full - s1_small_full - s2_small_full - s0_small_full)

    result['constraint_row_geometry'] = {
        'n_rows_in_active_union': len(active_union), 'n_equality_rows_total': len(all_eq_rows),
        'families_of_interest_present': {fam: sum(1 for n in row_geometry if row_geometry[n]['family'] == fam)
                                          for fam in FAMILIES_OF_INTEREST},
        'scaling_reconstruction_note': (
            f'Scaled row norms are RECONSTRUCTED (not IPOPT-internal) via the documented default '
            f'gradient-based scaling formula scale_i = min(1, kappa/||row_i||_inf) with kappa='
            f'{GRADIENT_SCALING_KAPPA}, evaluated ONCE at S0 and held fixed across states (consistent '
            'with IPOPT computing NLP scaling once from the initial point). All three logs report '
            '"c scaling provided"/"d scaling provided", confirming scaling IS applied, but the exact '
            'internal factors are not printed and cannot be recovered exactly from these artifacts.'),
        'raw_row_norm_ladder_below_threshold': raw_below,
        'scaled_row_norm_ladder_below_threshold': scaled_below,
        'armA_unique_small_gradient_rows_at_scaled_1e-3': armA_unique_small,
        'armA_unique_small_gradient_rows_note': (
            'Rows with reconstructed scaled ||row||_inf <= 1e-3 in SA and not in S0/S1/S2 at the same '
            'threshold. Threshold-sensitivity is reported via the full ladder above; this single number '
            'must not be read without the ladder.'),
        'per_row_detail_sample': {n: row_geometry[n] for n in list(row_geometry)[:0]},  # placeholder, full detail dumped separately
    }

    # ---- dual activity per state (equality lambda + DUAL-ACTIVE rule) ----
    dual_activity = {}
    for label in ('SA', 'S1', 'S2'):
        d = duals_by_state[label]
        vals = np.array(list(d.values()))
        linf = float(np.max(np.abs(vals))) if len(vals) else 0.0
        thresh = DUAL_ACTIVE_REL * max(1.0, linf)
        dual_active_names = sorted(n for n, v in d.items() if abs(v) > thresh)
        dual_activity[label] = {'lambda_inf_norm': linf, 'dual_active_threshold': thresh,
                                 'n_dual_active_rows': len(dual_active_names)}

    # ---- primal x dual agreement matrix (inequality rows only; equality rows always primal-ACTIVE) ----
    agreement = {}
    for label in ('SA', 'S1', 'S2'):
        d = duals_by_state[label]
        thresh = dual_activity[label]['dual_active_threshold']
        both = primal_only = dual_only = neither = 0
        for name, a in activity_by_state[label].items():
            if a['kind'] != 'inequality':
                continue
            p_active = a.get('active_primary', False)
            lam = d.get(name)
            d_active = (lam is not None and abs(lam) > thresh)
            if p_active and d_active:
                both += 1
            elif p_active and not d_active:
                primal_only += 1
            elif not p_active and d_active:
                dual_only += 1
            else:
                neither += 1
        agreement[label] = {'primal_and_dual_active': both, 'primal_active_dual_inactive': primal_only,
                             'dual_active_primal_inactive': dual_only, 'neither': neither}
    result['dual_activity'] = dual_activity
    result['primal_dual_agreement_matrix'] = agreement

    # ---- bound geometry: distance to bound, z_L/z_U, per state, per ladder ----
    bounded_vars = [v for v in optim_vars if v.lb is not None or v.ub is not None]
    bound_geo = {}
    for v in bounded_vars:
        name = v.name
        entry = {'lb': float(v.lb) if v.lb is not None else None, 'ub': float(v.ub) if v.ub is not None else None}
        for label in ('S0', 'SA', 'S1', 'S2'):
            x = values_by_state[label].get(name)
            if x is None:
                entry[label] = None
                continue
            dist_l = (x - entry['lb']) if entry['lb'] is not None else None
            dist_u = (entry['ub'] - x) if entry['ub'] is not None else None
            dists = [d for d in (dist_l, dist_u) if d is not None]
            min_dist = min(dists) if dists else None
            zl = zL_by_state.get(label, {}).get(name) if label != 'S0' else None
            zu = zU_by_state.get(label, {}).get(name) if label != 'S0' else None
            entry[label] = {'x': x, 'dist_to_lb': dist_l, 'dist_to_ub': dist_u, 'min_dist': min_dist,
                             'zL_out': zl, 'zU_out': zu}
        bound_geo[name] = entry

    near_bound_by_threshold = {}
    for t in LADDER:
        near_bound_by_threshold[str(t)] = {}
        for label in ('S0', 'SA', 'S1', 'S2'):
            names = sorted(n for n, e in bound_geo.items()
                            if e[label] is not None and e[label]['min_dist'] is not None
                            and e[label]['min_dist'] <= t)
            near_bound_by_threshold[str(t)][label] = {'count': len(names)}

    sa_near = set(n for n, e in bound_geo.items()
                  if e['SA'] is not None and e['SA']['min_dist'] is not None and e['SA']['min_dist'] <= NEAR_ACTIVE_THRESH)
    s1_near = set(n for n, e in bound_geo.items()
                  if e['S1'] is not None and e['S1']['min_dist'] is not None and e['S1']['min_dist'] <= NEAR_ACTIVE_THRESH)
    s2_near = set(n for n, e in bound_geo.items()
                  if e['S2'] is not None and e['S2']['min_dist'] is not None and e['S2']['min_dist'] <= NEAR_ACTIVE_THRESH)
    armA_unique_near_bound = sorted(sa_near - s1_near - s2_near)
    shared_all_three = sorted(sa_near & s1_near & s2_near)
    controls_only = sorted((s1_near | s2_near) - sa_near)

    result['bound_geometry'] = {
        'n_bounded_vars': len(bounded_vars), 'expected_n_bounded_vars': 7390,
        'match_n_bounded_vars': len(bounded_vars) == 7390,
        'near_bound_count_by_threshold': near_bound_by_threshold,
        'armA_unique_near_bound_at_1e-4': {'count': len(armA_unique_near_bound),
                                            'sample': armA_unique_near_bound[:30]},
        'shared_near_bound_all_three_at_1e-4': {'count': len(shared_all_three)},
        'controls_only_near_bound_at_1e-4': {'count': len(controls_only)},
        'fraction_to_boundary_candidate_set_note': (
            'NEAR-BOUND CANDIDATE SET ONLY -- BLOCKING VARIABLE NOT IDENTIFIABLE FROM PRESERVED '
            'ARTIFACTS. The preserved logs and .sol files carry the aggregate primal step alpha and '
            'the post-step iterate/multipliers, but NOT the IPOPT internal search direction vector '
            'd_x used at each iteration. Without d_x, no specific variable can be shown to be the '
            'one whose fraction-to-boundary limit produced the observed small aggregate alpha; only '
            'the set of variables near a bound at the terminal iterate can be reported.'),
    }

    # ---- Relation to P5.3 (informational only) ----
    result['p53_relation'] = {
        'sess_snet_def_present_in_current_model': any(row_family(n) == 'sess_snet_def' for n in row_snapshot_by_state['S0']),
        'note': ('The current model uses `sess_pnet_def` (24 rows), not the historical `sess_snet_def` '
                 'removed by the P5.3-B3/P5.4 active-power ESS reformulation. No recurrence of that '
                 'specific historical row is inferred merely because both are ESS active-power-net rows; '
                 'this is recorded per REVISION_CONTEXT housekeeping instruction, not re-litigated.'),
    }

    result['ad_failures'] = n_ad_failures
    result['row_geometry_full'] = row_geometry  # kept for the journal dump (may be large but bounded by active_union size)
    result['jacobian_budget_summary'] = {'n_builds': len(budget_state['builds']), 'builds': budget_state['builds']}
    return result


# ===========================================================================
# main
# ===========================================================================
def main():
    if OUT.exists():
        print(f'[P512G] refusing to run: {OUT} already exists', file=sys.stderr)
        sys.exit(2)
    OUT.mkdir(parents=True)

    install_solver_guards()
    JOURNAL['repo_state_start'] = repo_state()
    t_start = time.time()

    try:
        tier0_gates, tier0_parsed = tier0_analysis()
        JOURNAL['tier0'] = tier0_gates
        JOURNAL['status'] = 'TIER0_DONE'

        tier1_out = tier1_analysis(tier0_parsed)
        JOURNAL['tier1'] = tier1_out
        JOURNAL['status'] = 'COMPLETED'
    except BaseException as exc:
        JOURNAL['status'] = 'STOPPED'
        JOURNAL['stop_reason'] = f'{type(exc).__name__}: {exc}'
        import traceback
        JOURNAL['traceback'] = traceback.format_exc()
        print('[P512G] STOPPED:', JOURNAL['stop_reason'], file=sys.stderr)

    JOURNAL['zero_solver']['counters'] = dict(K._SOLVE_COUNTERS)
    JOURNAL['zero_solver']['zero_confirmed'] = all(v == 0 for v in K._SOLVE_COUNTERS.values())
    JOURNAL['repo_state_end'] = repo_state()
    JOURNAL['wall_seconds'] = time.time() - t_start
    uninstall_solver_guards()

    journal_path = OUT / 'p512g_journal.json'
    dump_json(journal_path, to_jsonable(JOURNAL))
    record_output('journal', journal_path)

    manifest = {'inputs': JOURNAL['artifacts']['inputs'], 'outputs': JOURNAL['artifacts']['outputs'],
                'script_sha256': sha(Path(__file__)), 'repo_state_start': JOURNAL['repo_state_start'],
                'repo_state_end': JOURNAL['repo_state_end']}
    dump_json(OUT / 'manifest.json', manifest)

    print('[P512G]', JOURNAL['status'], JOURNAL.get('stop_reason', ''), flush=True)
    print('[P512G] zero-solver counters:', JOURNAL['zero_solver']['counters'], flush=True)
    sys.exit(0 if JOURNAL['status'] == 'COMPLETED' else 2)


if __name__ == '__main__':
    if str(Path.cwd()) != str(ROOT):
        print('[P512G] wrong working directory; run from repository root', file=sys.stderr)
        sys.exit(2)
    main()
