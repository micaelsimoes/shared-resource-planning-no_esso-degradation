"""
Stage P3.5-A -- sess_snet_def-only frozen diagnostic.

Implements LOCAL_NLP_STABILITY_PLAN.md sections 4-10 EXACTLY:
  - loads the two prescribed frozen pickles after SHA-256 verification;
  - for each, solves an untouched baseline (A) and a variant (B) in which
    ONLY the sess_snet_def constraint rows for the one genuinely-installed
    shared-ESS index are deactivated (sess_comp and everything else left
    untouched);
  - reports starting-point structural diagnostics for the deactivated rows,
    the full A/B solver report, and the post-solve residual of the removed
    equation at the B solution.

This script does NOT touch any production file. It only reads the two
frozen .pkl files and writes its own log/result files under
data/SRP1/Results/FrozenSMOPF/P35A/. It must be run inside the project's
real Python/Pyomo/IPOPT (MA97) environment -- the same one used to produce
the frozen captures -- from the repository root:

    cd /Users/micaelsimoes/PycharmProjects/shared-resources-planning
    python p35a_sess_snet_def_diagnostic.py

It intentionally does NOT use the generic placeholder-class unpickler that
was used for the read-only P3 audit (LOCAL_NLP_STABILITY_PLAN.md sec. 5
explicitly forbids that for solver execution). It uses the project's own
`network.py` / `network_parameters.py` / `helper_functions.py` modules.
"""

import os
import re
import sys
import json
import pickle
import hashlib
import statistics
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import pyomo.environ as pe  # noqa: E402

from network import _run_smopf  # noqa: E402  (reuses the exact accepted solver/recovery path)
from network_parameters import NetworkParameters  # noqa: E402
from helper_functions import solver_result_succeeded, solver_result_summary  # noqa: E402


CASES = {
    'DSO': {
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/P3Preserved/frozen_DSO_node7_case33_2_2025_Autumn_cycle8.pkl',
        'sha256': '066117b88085e5d8b20ec4da684555902d57565044d4cf293516796637074711',
        'params_file': 'data/SRP1/case33_2/case33_2_params.json',
        'network_name': 'case33_2',
        'year': 2025,
        'day': 'Autumn',
        'cycle': 8,
        'is_transmission': False,
        'shared_ess_index': 0,
        'from_warm_start': True,
    },
    'TSO': {
        'pkl_path': 'data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Summer_cycle6.pkl',
        'sha256': '51d9097418561612d61367d12600ea3929b622c367a5feeef1ab6efd4b08355e',
        'params_file': 'data/SRP1/case9/case9_params.json',
        'network_name': 'case9',
        'year': 2025,
        'day': 'Summer',
        'cycle': 6,
        'is_transmission': True,
        'shared_ess_index': 1,
        'from_warm_start': True,
    },
}

OUT_ROOT = os.path.join(REPO_ROOT, 'data/SRP1/Results/FrozenSMOPF/P35A')


def sha256_of(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_frozen_model(path):
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    return payload['metadata'], payload['model']


def make_params(params_file):
    params = NetworkParameters()
    params.read_parameters_from_file(os.path.join(REPO_ROOT, params_file))
    return params


def make_network_shim(case_key, case, subdir):
    logs_dir = os.path.join(OUT_ROOT, case_key, subdir, 'logs')
    results_dir = os.path.join(OUT_ROOT, case_key, subdir, 'results')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return SimpleNamespace(
        name=case['network_name'],
        year=case['year'],
        day=case['day'],
        logs_dir=logs_dir,
        results_dir=results_dir,
        is_transmission=case['is_transmission'],
    )


def shared_ess_index_set(model):
    return list(model.shared_energy_storages)


def shared_ess_capacity(model, idx):
    s_rated = pe.value(model.shared_es_s_rated_fixed[idx], exception=False)
    e_rated = pe.value(model.shared_es_e_rated_fixed[idx], exception=False)
    return s_rated, e_rated


def sess_snet_def_rows(model, idx):
    rows = []
    for (e, s_m, s_o, p) in model.sess_snet_def:
        if e == idx:
            rows.append((e, s_m, s_o, p))
    return rows


def row_state(model, idx, s_m, s_o, p):
    sch = pe.value(model.shared_es_sch[idx, s_m, s_o, p], exception=False)
    sdch = pe.value(model.shared_es_sdch[idx, s_m, s_o, p], exception=False)
    pnet = pe.value(model.shared_es_pnet[idx, s_m, s_o, p], exception=False)
    qnet = pe.value(model.shared_es_qnet[idx, s_m, s_o, p], exception=False)
    g = (sch - sdch) ** 2 - pnet ** 2 - qnet ** 2
    grad = {
        'dsch': 2.0 * (sch - sdch),
        'dsdch': -2.0 * (sch - sdch),
        'dpnet': -2.0 * pnet,
        'dqnet': -2.0 * qnet,
    }
    grad_inf_norm = max(abs(v) for v in grad.values())
    return {
        'sch': sch, 'sdch': sdch, 'pnet': pnet, 'qnet': qnet,
        'g': g, 'grad': grad, 'grad_inf_norm': grad_inf_norm,
    }


def summarize_rows(states):
    def stat(key):
        vals = [s[key] for s in states]
        return {'min': min(vals), 'max': max(vals)}

    abs_g = sorted(abs(s['g']) for s in states)
    grad_norms = sorted(s['grad_inf_norm'] for s in states)

    def median(sorted_vals):
        return statistics.median(sorted_vals) if sorted_vals else None

    below = {
        '1e-8': sum(1 for v in grad_norms if v < 1e-8),
        '1e-6': sum(1 for v in grad_norms if v < 1e-6),
        '1e-4': sum(1 for v in grad_norms if v < 1e-4),
    }
    return {
        'n_rows': len(states),
        'sch': stat('sch'),
        'sdch': stat('sdch'),
        'pnet': stat('pnet'),
        'qnet': stat('qnet'),
        'abs_g': {'min': abs_g[0] if abs_g else None, 'median': median(abs_g), 'max': abs_g[-1] if abs_g else None},
        'grad_inf_norm': {
            'min': grad_norms[0] if grad_norms else None,
            'median': median(grad_norms),
            'max': grad_norms[-1] if grad_norms else None,
        },
        'rows_below_grad_threshold': below,
    }


IPOPT_SUMMARY_RE = {
    'iterations': re.compile(r'Number of Iterations\.+:\s*(\d+)'),
    'objective': re.compile(r'Objective\.+:\s*([\-0-9.eE+]+)\s+([\-0-9.eE+]+)'),
    'dual_inf': re.compile(r'Dual infeasibility\.+:\s*([\-0-9.eE+]+)\s+([\-0-9.eE+]+)'),
    'constr_viol': re.compile(r'Constraint violation\.+:\s*([\-0-9.eE+]+)\s+([\-0-9.eE+]+)'),
    'complementarity': re.compile(r'Complementarity\.+:\s*([\-0-9.eE+]+)\s+([\-0-9.eE+]+)'),
    'overall_nlp_error': re.compile(r'Overall NLP error\.+:\s*([\-0-9.eE+]+)\s+([\-0-9.eE+]+)'),
    'cpu_seconds': re.compile(r'Total (?:seconds in IPOPT|CPU secs in IPOPT)\s*(?:\(w/o function evaluations\)\s*)?=\s*([0-9.eE+]+)'),
}


def parse_ipopt_log_tail(log_path, max_bytes=200_000):
    if not log_path or not os.path.exists(log_path):
        return {}
    with open(log_path, 'rb') as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        tail = handle.read().decode('utf-8', errors='replace')
    out = {}
    for key, pattern in IPOPT_SUMMARY_RE.items():
        match = None
        for match in pattern.finditer(tail):
            pass  # keep the LAST match (most recent solve appended to this log)
        if match:
            out[key] = match.groups()
    return out


def solve_case(case_key, case, model, subdir):
    network_shim = make_network_shim(case_key, case, subdir)
    params = make_params(case['params_file'])
    result = _run_smopf(network_shim, model, params, from_warm_start=case['from_warm_start'])
    succeeded = solver_result_succeeded(result)
    summary = solver_result_summary(result)
    log_glob_dir = network_shim.logs_dir
    log_files = sorted(
        (os.path.join(log_glob_dir, f) for f in os.listdir(log_glob_dir)),
        key=os.path.getmtime,
    ) if os.path.isdir(log_glob_dir) else []
    parsed_logs = [parse_ipopt_log_tail(f) for f in log_files]
    return {
        'succeeded': succeeded,
        'summary': summary,
        'status': str(getattr(result.solver, 'status', None)) if result is not None and hasattr(result, 'solver') else None,
        'termination_condition': str(getattr(result.solver, 'termination_condition', None)) if result is not None and hasattr(result, 'solver') else None,
        'log_files': log_files,
        'parsed_logs': parsed_logs,
        'result': result,
    }


def run_case(case_key):
    case = CASES[case_key]
    pkl_path = os.path.join(REPO_ROOT, case['pkl_path'])

    report = {'case': case_key, 'config': case}

    # --- Section A: frozen-file verification -----------------------------
    actual_hash = sha256_of(pkl_path)
    report['hash_check'] = {
        'path': case['pkl_path'],
        'expected_sha256': case['sha256'],
        'actual_sha256': actual_hash,
        'match': actual_hash == case['sha256'],
    }
    if actual_hash != case['sha256']:
        report['stop_reason'] = 'HASH MISMATCH -- STOPPING per plan sec. 4'
        return report

    # --- Load A (untouched baseline) --------------------------------------
    meta_a, model_a = load_frozen_model(pkl_path)
    report['metadata'] = meta_a

    idx = case['shared_ess_index']
    all_indices = shared_ess_index_set(model_a)
    capacities = {i: shared_ess_capacity(model_a, i) for i in all_indices}
    report['shared_ess_capacity_check'] = {
        'expected_index': idx,
        'all_indices_and_capacity_s_e': capacities,
    }
    s_rated, e_rated = capacities.get(idx, (None, None))
    if not s_rated or abs(s_rated) < 1e-10 or not e_rated or abs(e_rated) < 1e-10:
        report['stop_reason'] = (
            f'Expected shared-ESS index {idx} does not have positive installed '
            f'capacity (s_rated={s_rated}, e_rated={e_rated}) -- STOPPING per plan sec. 6.B.4'
        )
        return report

    solve_a = solve_case(case_key, case, model_a, subdir='A_baseline')
    report['A_baseline'] = {k: v for k, v in solve_a.items() if k != 'result'}

    # --- Load B (independent fresh clone) ----------------------------------
    meta_b, model_b = load_frozen_model(pkl_path)

    rows = sess_snet_def_rows(model_b, idx)
    states_pre = [row_state(model_b, *row) for row in rows]
    report['sess_snet_def_starting_point_diagnostics'] = summarize_rows(states_pre)
    report['sess_snet_def_row_count'] = len(rows)
    report['sess_snet_def_capacity_normalization'] = {
        's_rated': s_rated,
        'e_rated': e_rated,
        'max_abs_g_over_s_rated_sqr': (
            max(abs(s['g']) for s in states_pre) / max(s_rated ** 2, 1e-12)
            if states_pre else None
        ),
    }

    # --- Apply the ONLY authorized change: deactivate sess_snet_def for idx
    for row in rows:
        model_b.sess_snet_def[row].deactivate()
    # sess_comp and every other constraint/variable/bound is left untouched.

    solve_b = solve_case(case_key, case, model_b, subdir='B_sess_snet_def_off')
    report['B_variant'] = {k: v for k, v in solve_b.items() if k != 'result'}

    # --- Post-solve residual of the REMOVED equation at the B solution ----
    states_post = [row_state(model_b, *row) for row in rows]
    abs_g_post = [abs(s['g']) for s in states_post]
    if abs_g_post:
        worst_idx = max(range(len(states_post)), key=lambda i: abs_g_post[i])
        worst_row = rows[worst_idx]
        worst_state = states_post[worst_idx]
        report['removed_equation_residual_at_B_solution'] = {
            'max_abs_g': max(abs_g_post),
            'median_abs_g': statistics.median(abs_g_post),
            'max_normalized_violation': max(abs_g_post) / max(s_rated ** 2, 1e-12),
            'worst_row_index_e_sm_so_p': worst_row,
            'worst_row_state': worst_state,
        }
    else:
        report['removed_equation_residual_at_B_solution'] = None

    return report


def main():
    full_report = {}
    for case_key in ('DSO', 'TSO'):
        full_report[case_key] = run_case(case_key)

    out_path = os.path.join(OUT_ROOT, 'p35a_report.json')
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(full_report, handle, indent=2, default=str)
    print(f'[P3.5-A] Wrote full report to {out_path}')
    print(json.dumps(full_report, indent=2, default=str))


if __name__ == '__main__':
    main()
