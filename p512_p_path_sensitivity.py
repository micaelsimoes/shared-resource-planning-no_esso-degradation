"""P5.12-P: bounded numerical path-sensitivity probe on the frozen Arm A
cycle-21 fixture.

Authorized by LOCAL_NLP_STABILITY_PLAN.md, "AUTHORIZED NEXT STAGE -- P5.12-P
path-sensitivity probe (2026-09-12)". Exactly TWO new solves total, one per
variant, each from an independently fresh reload of the frozen P5.12-R
before-setup snapshot. Single varied parameter: `warm_start_bound_push`
(baseline 1e-5 = Arm A, NOT rerun here; variant 1 = 1e-6; variant 2 = 1e-4).

Run once, from the repository root, with the canonical interpreter:

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p512_p_path_sensitivity.py

This script is NOT a modification of any frozen harness or governing
document. All outputs go to a NEW directory data/SRP1/Results/P512P/, which
must not exist beforehand. It imports p512_r_presolve_recapture read-only as
`H` and never edits it.
"""
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path('/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation')
OUT = ROOT / 'data/SRP1/Results/P512P'
TARGET = 'DSO:case33_3|2025|Spring'
PRE_SETUP_SNAPSHOT = ROOT / 'data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl'
ARMA_DIR = ROOT / 'data/SRP1/Results/P512ArmA'
ARMA_LOG = ARMA_DIR / 'logs/optim_log_case33_3_2025_Spring.log'
HARNESS_PATH = ROOT / 'p512_r_presolve_recapture.py'

EXPECTED_HEAD = 'ba202e2b0e937306f3c163de2173951c1d0c24f0'
EXPECTED_TRACKED_COUNT = 1946
EXPECTED_HARNESS_SHA = 'f0f120c26ec2c50b774ff42051c233b87fba0341e3d959faafe70283301d86f0'
EXPECTED_SNAPSHOT_SHA = '38fa9e2c9031c22e55e3a31db072ea10cd51a306693a7a724ecce3bec4795cd8'
EXPECTED_PRE_SETUP_DIGEST = '22d5d85087b7a6eb632e2993f839f9e8ebcfe7f6f2d53b461ac39b543ef9a0e5'
EXPECTED_PREPARED_DIGEST = '3fd294d7b41ad3b2b62a140cb518a9ecbe32a0218eed8d6cd788fc027edda85f'
EXPECTED_NL_SHA = '5934341b1137271b7a86f440ff7a0d146c7b4b6c256f9331f8b96edc807d39a7'
EXPECTED_MAPPING_SHA = 'c13732e842003fe2331c6eea869cd78d14cafc3541e259373d1fa42f7e88d2ff'
EXPECTED_ARMA_LOG_SHA = '4f66a7efeef933bdc0a425af76f0095f5c11a2112ff2c8bb6d7c03ff45409d58'

BASELINE_VALUE = 1e-05          # Arm A -- NOT rerun
VARIANTS = [
    {'name': 'variant1_1e-6', 'value': 1e-6},
    {'name': 'variant2_1e-4', 'value': 1e-4},
]

# Arm A's recorded effective option set, exactly as given in the authorized
# task text (before this probe's two-key override/redirect).
ARMA_OPTIONS_REFERENCE = {
    'tol': 1e-05,
    'acceptable_tol': 0.0001,
    'acceptable_iter': 5,
    'linear_solver': 'ma97',
    'bound_push': 1e-05,
    'bound_frac': 1e-05,
    'slack_bound_frac': 1e-05,
    'slack_bound_push': 1e-05,
    'output_file': '<ARM_A_LOG_PATH>',   # expected to differ (redirected)
    'file_append': 'yes',
    'file_print_level': 6,
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-05,      # expected to differ (the one authorized key)
    'warm_start_bound_frac': 1e-05,
    'warm_start_slack_bound_frac': 1e-05,
    'warm_start_slack_bound_push': 1e-05,
    'warm_start_mult_bound_push': 1e-05,
}

REPORT = {'stage': 'P5.12-P', 'status': 'NOT_STARTED', 'variants': {}}

SOLVE_COUNTER = {'n': 0}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True, text=True, check=True).stdout


def fail_global(msg):
    REPORT['status'] = 'STOPPED'
    REPORT['stop_reason'] = msg
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'p512p_report.json').open('w') as f:
        json.dump(REPORT, f, indent=1, default=str)
    print('P5.12-P STOPPED:', msg, flush=True)
    sys.exit(2)


# ===========================================================================
# log-parsing helpers (self-contained re-implementation of the same read-only
# analysis pattern used by p512_k_no_solve_kkt_forensic.py; that file is not
# imported or modified)
# ===========================================================================
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
                obj = float(parts[1])
                inf_pr = float(parts[2])
                inf_du = float(parts[3])
                lg_mu = float(parts[4])
                dnorm = parts[5]
                lg_rg = parts[6]
                alpha_du = parts[7]
                alpha_pr = parts[8]
                ls = parts[9] if len(parts) > 9 else ''
                marker = parts[-1] if re.match(r'^[a-zA-Z]+$', parts[-1]) else ''
            except (ValueError, IndexError):
                continue
            rows.append({'iter': it, 'restoration': restoration, 'objective': obj,
                         'inf_pr': inf_pr, 'inf_du': inf_du, 'lg_mu': lg_mu,
                         'dnorm': dnorm, 'lg_rg': lg_rg, 'alpha_du': alpha_du,
                         'alpha_pr': alpha_pr, 'ls': ls, 'marker': marker,
                         'raw': line.rstrip('\n')})
    return rows


def parse_iter0_norms(text):
    keys = ['curr_x', 'curr_s', 'curr_y_c', 'curr_y_d', 'curr_z_L', 'curr_z_U', 'curr_v_L', 'curr_v_U']
    out = {}
    for k in keys:
        m = re.findall(r'\|\|' + k + r'\|\|_inf\s*=\s*([\d.eE+-]+)', text)
        out[k] = m[0] if m else None
    return out


def parse_nlp_values(text, first=True):
    """Return (scaled, unscaled) tuples for Objective / Dual infeasibility /
    Constraint violation / Complementarity / Overall NLP error, first or last
    occurrence."""
    fields = ['Objective', 'Dual infeasibility', 'Constraint violation',
              'Complementarity', 'Overall NLP error']
    out = {}
    for name in fields:
        pattern = re.escape(name) + r'\.+:\s+([\d.eE+-]+)\s+([\d.eE+-]+)'
        m = re.findall(pattern, text)
        if not m:
            out[name] = None
            continue
        pair = m[0] if first else m[-1]
        out[name] = {'scaled': float(pair[0]), 'unscaled': float(pair[1])}
    return out


def parse_sol_file(sol_path):
    lines = Path(sol_path).read_text().splitlines()
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


def num(expr):
    try:
        import pyomo.environ as pe
        v = pe.value(expr, exception=False)
        return float(v) if v is not None else None
    except Exception:
        return None


def compute_terminal_offenders(model, sol, mapping):
    import pyomo.environ as pe
    sym_to_name = dict(mapping['mapping'])
    var_syms = sorted((s for s in sym_to_name if s.startswith('v')), key=lambda s: int(s[1:]))
    name_lookup = {}
    for v in model.component_data_objects(pe.Var, active=None):
        name_lookup[v.name] = v
    x_final = {}
    for i, s in enumerate(var_syms):
        name = sym_to_name[s]
        if i < len(sol['primals']):
            x_final[name] = sol['primals'][i]
    saved_vals = {n: v.value for n, v in name_lookup.items()}
    for name, val in x_final.items():
        if name in name_lookup:
            name_lookup[name].set_value(val, skip_validation=True)

    offenders = []
    n_eq, n_ineq = 0, 0
    for c in model.component_data_objects(pe.Constraint, active=True):
        body = num(c.body)
        if body is None:
            continue
        if c.equality:
            n_eq += 1
            resid = abs(body - num(c.lower))
            side = 'equality'
        else:
            n_ineq += 1
            lo = num(c.lower)
            hi = num(c.upper)
            viol = 0.0
            side = 'satisfied'
            if lo is not None and body < lo:
                viol = lo - body
                side = 'below_lower'
            elif hi is not None and body > hi:
                viol = body - hi
                side = 'above_upper'
            resid = viol
        if resid is not None and resid > 1e-6:
            offenders.append({'name': c.name, 'residual': resid, 'side': side, 'equality': c.equality})
    offenders.sort(key=lambda o: -o['residual'])

    zU_out = sol['suffixes'].get('ipopt_zU_out', {})
    zL_out = sol['suffixes'].get('ipopt_zL_out', {})
    complementarity_products = []
    for i, s in enumerate(var_syms):
        name = sym_to_name[s]
        v = name_lookup.get(name)
        if v is None or v.fixed:
            continue
        lb = float(v.lb) if v.lb is not None else None
        ub = float(v.ub) if v.ub is not None else None
        if lb is not None and ub is not None and lb == ub:
            continue
        x = x_final.get(name)
        if x is None:
            continue
        zl = zL_out.get(i)
        zu = zU_out.get(i)
        prod_l = (zl * (x - lb)) if (zl is not None and lb is not None) else None
        prod_u = (zu * (ub - x)) if (zu is not None and ub is not None) else None
        if prod_l is not None or prod_u is not None:
            complementarity_products.append({'name': name, 'zL_out': zl, 'zU_out': zu,
                                              'zL_times_(x-l)': prod_l, 'zU_times_(u-x)': prod_u})

    for n, v in name_lookup.items():
        if n in saved_vals and saved_vals[n] is not None:
            v.set_value(saved_vals[n], skip_validation=True)

    def dist_summary(vals, terminal_mu):
        if not vals:
            return None
        a = np.abs(np.array(vals))
        return {'n': len(a), 'max': float(a.max()), 'mean': float(a.mean()),
                'median': float(np.median(a)),
                'n_exceeding_terminal_mu': int(np.sum(a > terminal_mu)) if terminal_mu else None}

    prods_l = [c['zL_times_(x-l)'] for c in complementarity_products if c['zL_times_(x-l)'] is not None]
    prods_u = [c['zU_times_(u-x)'] for c in complementarity_products if c['zU_times_(u-x)'] is not None]

    return {
        'n_equality_rows_checked': n_eq, 'n_inequality_rows_checked': n_ineq,
        'top_offending_rows': offenders[:30],
        'complementarity_products_zL_times_(x-l)_raw': prods_l,
        'complementarity_products_zU_times_(u-x)_raw': prods_u,
        '_dist_summary_fn': dist_summary,
    }


# ===========================================================================
# main
# ===========================================================================
def main():
    assert Path.cwd() == ROOT, 'wrong cwd: ' + str(Path.cwd())
    assert sys.executable == '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python', sys.executable

    # ------------------------------------------------------------------
    # global pre-checks
    # ------------------------------------------------------------------
    head = git('rev-parse', 'HEAD').strip()
    branch = git('branch', '--show-current').strip()
    staged = git('diff', '--cached')
    status_porcelain = git('status', '--porcelain', '--untracked-files=no')
    tracked_files = sorted(f for f in git('ls-files').splitlines() if f)

    precheck = {
        'head': head, 'expected_head': EXPECTED_HEAD,
        'branch': branch,
        'staged_diff_empty': staged == '',
        'status_porcelain': status_porcelain,
        'tracked_file_count': len(tracked_files),
        'expected_tracked_file_count': EXPECTED_TRACKED_COUNT,
    }
    REPORT['precheck'] = precheck

    if head != EXPECTED_HEAD:
        fail_global('HEAD mismatch: ' + head)
    if staged != '':
        fail_global('git diff --cached is not empty')
    status_lines = sorted(l for l in status_porcelain.splitlines() if l)
    expected_status_lines = sorted([' M LOCAL_NLP_STABILITY_PLAN.md', ' M REVISION_CONTEXT.md'])
    if status_lines != expected_status_lines:
        fail_global('unexpected tracked working-tree status: ' + repr(status_lines))
    if len(tracked_files) != EXPECTED_TRACKED_COUNT:
        fail_global('git ls-files count mismatch: ' + str(len(tracked_files)))

    harness_hash = sha(HARNESS_PATH)
    REPORT['harness_sha256'] = harness_hash
    if harness_hash != EXPECTED_HARNESS_SHA:
        fail_global('harness hash mismatch: ' + harness_hash)

    snapshot_hash = sha(PRE_SETUP_SNAPSHOT)
    REPORT['pre_setup_snapshot_sha256'] = snapshot_hash
    if snapshot_hash != EXPECTED_SNAPSHOT_SHA:
        fail_global('pre_setup snapshot hash mismatch: ' + snapshot_hash)

    arma_log_hash = sha(ARMA_LOG)
    REPORT['arma_log_sha256'] = arma_log_hash
    if arma_log_hash != EXPECTED_ARMA_LOG_SHA:
        fail_global('Arm A reference log hash mismatch: ' + arma_log_hash)

    if OUT.exists():
        fail_global('data/SRP1/Results/P512P already exists')

    REPORT['precheck']['pass'] = True
    REPORT['declared_values'] = {'baseline_not_rerun': BASELINE_VALUE,
                                 'variant1': VARIANTS[0]['value'], 'variant2': VARIANTS[1]['value']}
    print('[P512P] pre-checks passed', flush=True)

    OUT.mkdir(parents=False, exist_ok=False)

    sys.path.insert(0, str(ROOT))
    import p512_r_presolve_recapture as H  # noqa: E402  (read-only import)
    import network as N  # noqa: E402
    from pyomo.opt.base.solvers import OptSolver  # noqa: E402

    if sha(HARNESS_PATH) != EXPECTED_HARNESS_SHA:
        fail_global('harness hash changed after import')

    original_solve = OptSolver.solve

    def counted_solve(self, *args, **kwargs):
        SOLVE_COUNTER['n'] += 1
        if SOLVE_COUNTER['n'] > 2:
            raise RuntimeError('P5.12-P: OptSolver.solve invoked more than twice in this run')
        return original_solve(self, *args, **kwargs)

    OptSolver.solve = counted_solve

    # Pre-parse Arm A's own iteration table / iter-0 norms / NLP values once,
    # for reuse in every variant comparison.
    arma_text = ARMA_LOG.read_text(errors='replace')
    arma_rows = parse_iteration_table(ARMA_LOG)
    arma_iter0_norms = parse_iter0_norms(arma_text[:arma_text.find('Beginning Iteration 1')]
                                         if 'Beginning Iteration 1' in arma_text else arma_text)
    arma_nlp_first = parse_nlp_values(arma_text, first=True)
    arma_nlp_last = parse_nlp_values(arma_text, first=False)
    n_z_marker_armA = sum(1 for r in arma_rows if 'z' in r['marker'])
    first_z_iter_armA = next((r['iter'] for r in arma_rows if 'z' in r['marker']), None)
    n_zU_msg_armA = len(re.findall(r'Some value in z_U becomes too large', arma_text))
    mu_matches_armA = re.findall(r'Current barrier parameter mu = ([\d.eE+-]+)', arma_text)
    arma_final_mu = float(mu_matches_armA[-1]) if mu_matches_armA else None
    arma_exit_line = None
    for ln in arma_text.splitlines():
        if ln.startswith('EXIT:'):
            arma_exit_line = ln.strip()

    REPORT['arma_reference_recomputed'] = {
        'n_iteration_rows': len(arma_rows),
        'z_marker_count': n_z_marker_armA,
        'first_z_marker_iteration': first_z_iter_armA,
        'zU_too_large_message_count': n_zU_msg_armA,
        'final_mu': arma_final_mu,
        'exit_line': arma_exit_line,
        'nlp_values_first': arma_nlp_first,
        'nlp_values_last': arma_nlp_last,
        'iter0_norms': arma_iter0_norms,
    }

    any_variant_stopped = False
    try:
        for spec in VARIANTS:
            try:
                run_variant(H, N, spec, arma_rows, arma_text, arma_nlp_first, arma_nlp_last,
                            n_z_marker_armA, first_z_iter_armA, n_zU_msg_armA, arma_final_mu)
            except VariantStopped:
                any_variant_stopped = True
    finally:
        OptSolver.solve = original_solve

    if any_variant_stopped:
        REPORT['status'] = 'DONE_WITH_STOP'
        REPORT['solve_call_counter_final'] = SOLVE_COUNTER['n']
        with (OUT / 'p512p_report.json').open('w') as f:
            json.dump(REPORT, f, indent=1, default=str)
        print('[P512P] one or more variants stopped before solving; see p512p_report.json', flush=True)
        return

    REPORT['solve_call_counter_final'] = SOLVE_COUNTER['n']
    REPORT['status'] = 'DONE'
    with (OUT / 'p512p_report.json').open('w') as f:
        json.dump(REPORT, f, indent=1, default=str)
    print('[P512P] DONE. solve_call_counter =', SOLVE_COUNTER['n'], flush=True)


def run_variant(H, N, spec, arma_rows, arma_text, arma_nlp_first, arma_nlp_last,
                 n_z_marker_armA, first_z_iter_armA, n_zU_msg_armA, arma_final_mu):
    name = spec['name']
    value = spec['value']
    vdir = OUT / name
    vdir.mkdir(parents=False, exist_ok=False)
    (vdir / 'logs').mkdir()

    vreport = {'name': name, 'value': value, 'status': 'NOT_STARTED'}
    REPORT['variants'][name] = vreport

    def vfail(msg):
        vreport['status'] = 'STOPPED_BEFORE_SOLVE'
        vreport['stop_reason'] = msg
        print(f'[P512P:{name}] STOPPED BEFORE SOLVE:', msg, flush=True)
        with (vdir / 'variant_report.json').open('w') as f:
            json.dump(vreport, f, indent=1, default=str)
        raise VariantStopped(msg)

    # ------------------------------------------------------------------
    # fresh, independent reload
    # ------------------------------------------------------------------
    with PRE_SETUP_SNAPSHOT.open('rb') as f:
        payload = pickle.load(f)

    model = payload['model']
    net = payload['network']
    params = payload['params']
    from_warm_start = payload['from_warm_start']

    reload_checks = {'boundary': payload['boundary'], 'cycle': payload['cycle'],
                      'target': payload['target'], 'from_warm_start': from_warm_start}
    vreport['reload_checks'] = reload_checks
    ok = (payload['boundary'] == 'pre_setup' and payload['cycle'] == 21
          and payload['target'] == TARGET and from_warm_start is True)
    if not ok:
        vfail('reloaded payload metadata mismatch: ' + json.dumps(reload_checks, default=str))

    recomputed_digest = H.digest(H.model_state(model))
    reload_checks['pre_setup_digest_recomputed'] = recomputed_digest
    reload_checks['pre_setup_digest_expected'] = EXPECTED_PRE_SETUP_DIGEST
    if recomputed_digest != EXPECTED_PRE_SETUP_DIGEST:
        vfail('reloaded model_state digest mismatch: ' + recomputed_digest)
    print(f'[P512P:{name}] fresh reload verified', flush=True)

    # ------------------------------------------------------------------
    # redirect logs_dir only
    # ------------------------------------------------------------------
    old_logs_dir = net.logs_dir
    new_logs_dir = str(vdir / 'logs')
    net.logs_dir = new_logs_dir
    vreport['logs_dir_redirect'] = {'old': old_logs_dir, 'new': new_logs_dir}

    # ------------------------------------------------------------------
    # setup exactly once
    # ------------------------------------------------------------------
    solver, solver_log_path, solve_context = N._create_smopf_solver(
        net, model, params, from_warm_start=True)
    vreport['setup'] = {'solver_log_path': solver_log_path, 'solve_context': solve_context}
    print(f'[P512P:{name}] _create_smopf_solver invoked exactly once; log path =', solver_log_path, flush=True)

    # ------------------------------------------------------------------
    # single authorized override, AFTER setup
    # ------------------------------------------------------------------
    before_override = solver.options.get('warm_start_bound_push')
    solver.options['warm_start_bound_push'] = value
    vreport['override'] = {'key': 'warm_start_bound_push',
                            'before': before_override, 'after': value}
    print(f'[P512P:{name}] overrode warm_start_bound_push: {before_override} -> {value}', flush=True)

    # ------------------------------------------------------------------
    # equality gate BEFORE solving
    # ------------------------------------------------------------------
    gate = {}
    prepared_digest = H.digest(H.model_state(model))
    gate['prepared_state_digest'] = prepared_digest
    gate['prepared_state_digest_expected'] = EXPECTED_PREPARED_DIGEST
    gate['prepared_state_digest_equal'] = (prepared_digest == EXPECTED_PREPARED_DIGEST)

    export_result = H.export(model, solver, vdir, f'prepared_{name}')
    gate['export'] = export_result
    gate['nl_sha256_equal'] = (export_result['nl_sha256'] == EXPECTED_NL_SHA)
    gate['mapping_sha256_equal'] = (export_result['mapping_sha256'] == EXPECTED_MAPPING_SHA)

    effective_options_now = dict(solver.options)
    diff_keys = set()
    for k in set(effective_options_now) | set(ARMA_OPTIONS_REFERENCE):
        a = effective_options_now.get(k)
        b = ARMA_OPTIONS_REFERENCE.get(k)
        if k in ('output_file', 'warm_start_bound_push'):
            continue  # authorized-difference keys, checked separately below
        if str(a) != str(b):
            diff_keys.add(k)
    gate['unexpected_option_diff_keys'] = sorted(diff_keys)
    gate['warm_start_bound_push_now'] = effective_options_now.get('warm_start_bound_push')
    gate['warm_start_bound_push_expected_variant_value'] = value
    gate['warm_start_bound_push_matches_variant'] = (
        str(effective_options_now.get('warm_start_bound_push')) == str(value))
    gate['output_file_now'] = effective_options_now.get('output_file')
    gate['output_file_differs_from_armA'] = (
        str(effective_options_now.get('output_file')) != str(ARMA_OPTIONS_REFERENCE['output_file']))
    gate['effective_options_now'] = effective_options_now
    gate['effective_options_armA_reference'] = ARMA_OPTIONS_REFERENCE

    gate_pass = (gate['prepared_state_digest_equal'] and gate['nl_sha256_equal']
                 and gate['mapping_sha256_equal'] and not diff_keys
                 and gate['warm_start_bound_push_matches_variant'])
    vreport['equality_gate'] = gate
    with (vdir / 'equality_gate.json').open('w') as f:
        json.dump(gate, f, indent=1, default=str)

    if not gate_pass:
        vfail('equality gate failed: ' + json.dumps(
            {k: v for k, v in gate.items() if k in (
                'prepared_state_digest_equal', 'nl_sha256_equal', 'mapping_sha256_equal',
                'unexpected_option_diff_keys', 'warm_start_bound_push_matches_variant')}, default=str))

    print(f'[P512P:{name}] equality gate passed (only warm_start_bound_push + output_file differ from Arm A)',
          flush=True)

    # ------------------------------------------------------------------
    # solve exactly once for this variant
    # ------------------------------------------------------------------
    start = time.time()
    result = solver.solve(model, tee=False, load_solutions=False, keepfiles=True)
    wall_seconds = time.time() - start

    vreport['solve'] = {
        'wall_seconds': wall_seconds,
        'status': str(result.solver.status),
        'termination_condition': str(result.solver.termination_condition),
    }
    print(f'[P512P:{name}] solve complete:', vreport['solve'], flush=True)

    temp_nl = getattr(solver, '_problem_files', None)
    temp_sol = getattr(solver, '_soln_file', None)
    vreport['solve']['temp_problem_files'] = str(temp_nl)
    vreport['solve']['temp_soln_file'] = str(temp_sol)

    with (vdir / 'solver_results.pkl').open('wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (vdir / 'solver_results_summary.txt').open('w') as f:
        f.write(str(result))

    copied_temp_files = {}
    used_sol_path = None
    if temp_nl:
        try:
            nl_paths = temp_nl if isinstance(temp_nl, (list, tuple)) else [temp_nl]
            for p in nl_paths:
                p = Path(str(p))
                if p.exists():
                    dest = vdir / ('used_' + p.name)
                    shutil.copy2(p, dest)
                    copied_temp_files[str(p)] = str(dest)
        except Exception as e:
            copied_temp_files['error_nl'] = str(e)
    if temp_sol:
        try:
            p = Path(str(temp_sol))
            if p.exists():
                dest = vdir / ('used_' + p.name)
                shutil.copy2(p, dest)
                copied_temp_files[str(p)] = str(dest)
                used_sol_path = dest
        except Exception as e:
            copied_temp_files['error_sol'] = str(e)
    vreport['copied_temp_files'] = copied_temp_files

    # ------------------------------------------------------------------
    # comparison against Arm A
    # ------------------------------------------------------------------
    variant_log_path = None
    if solver_log_path and os.path.exists(solver_log_path):
        variant_log_path = solver_log_path
    else:
        cand = list((vdir / 'logs').glob('*.log'))
        variant_log_path = str(cand[0]) if cand else None
    vreport['variant_log_path'] = variant_log_path

    comparison = {}
    if variant_log_path and os.path.exists(variant_log_path):
        vtext = Path(variant_log_path).read_text(errors='replace')
        vrows = parse_iteration_table(variant_log_path)
        v_iter0_norms = parse_iter0_norms(
            vtext[:vtext.find('Beginning Iteration 1')] if 'Beginning Iteration 1' in vtext else vtext)
        v_nlp_first = parse_nlp_values(vtext, first=True)
        v_nlp_last = parse_nlp_values(vtext, first=False)
        n_z_marker_v = sum(1 for r in vrows if 'z' in r['marker'])
        first_z_iter_v = next((r['iter'] for r in vrows if 'z' in r['marker']), None)
        n_zU_msg_v = len(re.findall(r'Some value in z_U becomes too large', vtext))
        mu_matches_v = re.findall(r'Current barrier parameter mu = ([\d.eE+-]+)', vtext)
        v_final_mu = float(mu_matches_v[-1]) if mu_matches_v else None
        v_exit_line = None
        for ln in vtext.splitlines():
            if ln.startswith('EXIT:'):
                v_exit_line = ln.strip()

        # row-by-row comparison against Arm A (printed-precision equality of
        # objective / inf_pr / inf_du, matched by iteration index)
        arma_by_iter = {r['iter']: r for r in arma_rows if not r['restoration']}
        first_divergence = None
        n_common = min(len([r for r in vrows if not r['restoration']]), len(arma_by_iter))
        v_ordinary_rows = [r for r in vrows if not r['restoration']]
        for r in v_ordinary_rows:
            a = arma_by_iter.get(r['iter'])
            if a is None:
                first_divergence = {'iter': r['iter'], 'reason': 'no matching Arm A row (Arm A shorter/ended)',
                                     'variant_row': r['raw']}
                break
            if (a['objective'] != r['objective'] or a['inf_pr'] != r['inf_pr']
                    or a['inf_du'] != r['inf_du'] or a['marker'] != r['marker']):
                first_divergence = {'iter': r['iter'], 'arma_row': a['raw'], 'variant_row': r['raw']}
                break

        iter0_identical = (v_iter0_norms == parse_iter0_norms(
            arma_text[:arma_text.find('Beginning Iteration 1')]))

        terminal_analysis = None
        if used_sol_path is not None:
            try:
                mapping = json.loads((vdir / f'prepared_{name}_mapping.json').read_text())
                sol = parse_sol_file(used_sol_path)
                terminal_raw = compute_terminal_offenders(model, sol, mapping)
                dist_fn = terminal_raw.pop('_dist_summary_fn')
                terminal_analysis = {
                    'n_equality_rows_checked': terminal_raw['n_equality_rows_checked'],
                    'n_inequality_rows_checked': terminal_raw['n_inequality_rows_checked'],
                    'top_offending_rows': terminal_raw['top_offending_rows'],
                    'complementarity_zL_times_(x-l)': dist_fn(
                        terminal_raw['complementarity_products_zL_times_(x-l)_raw'], v_final_mu),
                    'complementarity_zU_times_(u-x)': dist_fn(
                        terminal_raw['complementarity_products_zU_times_(u-x)_raw'], v_final_mu),
                }
            except Exception as e:
                terminal_analysis = {'error': str(e), 'traceback': traceback.format_exc()}

        comparison = {
            'n_iteration_rows_variant': len(vrows),
            'n_iteration_rows_armA': len(arma_rows),
            'iter0_norms_variant': v_iter0_norms,
            'iter0_norms_armA_recomputed': parse_iter0_norms(
                arma_text[:arma_text.find('Beginning Iteration 1')]),
            'iter0_norms_identical_17digit': iter0_identical,
            'nlp_values_iter0_variant': v_nlp_first,
            'nlp_values_iter0_armA': arma_nlp_first,
            'first_material_divergence': first_divergence,
            'z_marker_count_variant': n_z_marker_v,
            'z_marker_count_armA': n_z_marker_armA,
            'first_z_marker_iteration_variant': first_z_iter_v,
            'first_z_marker_iteration_armA': first_z_iter_armA,
            'zU_too_large_message_count_variant': n_zU_msg_v,
            'zU_too_large_message_count_armA': n_zU_msg_armA,
            'final_mu_variant': v_final_mu,
            'final_mu_armA': arma_final_mu,
            'nlp_values_final_variant': v_nlp_last,
            'nlp_values_final_armA': arma_nlp_last,
            'exit_line_variant': v_exit_line,
            'terminal_analysis_variant': terminal_analysis,
        }
    else:
        comparison = {'error': 'variant log not found at ' + str(variant_log_path)}

    vreport['comparison_vs_armA'] = comparison

    manifest = {}
    for p in sorted(vdir.rglob('*')):
        if p.is_file():
            try:
                manifest[str(p.relative_to(vdir))] = sha(p)
            except Exception as e:
                manifest[str(p.relative_to(vdir))] = 'ERROR:' + str(e)
    vreport['artifact_manifest'] = manifest

    vreport['status'] = 'SOLVED'
    with (vdir / 'variant_report.json').open('w') as f:
        json.dump(vreport, f, indent=1, default=str)
    print(f'[P512P:{name}] variant complete', flush=True)


class VariantStopped(Exception):
    pass


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        REPORT['status'] = 'ERROR'
        REPORT['traceback'] = traceback.format_exc()
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'p512p_report.json').open('w') as f:
            json.dump(REPORT, f, indent=1, default=str)
        print('[P512P] ERROR', flush=True)
        print(REPORT['traceback'], flush=True)
        sys.exit(1)
