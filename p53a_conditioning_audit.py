"""
Stage P5.3-A -- quantitative structural SMOPF conditioning audit.

Audits every network SMOPF of the real P5 iteration-2 positive-bootstrap cold
initialization (36 DSO + 12 TSO models) immediately before IPOPT.

Method
------
The production initialization is executed unchanged. `Network.run_smopf` is
wrapped so that, for each model, the audit runs on the exact cold-start state
just before the real solve is delegated to production. The real solve then
proceeds, so DSO results populate the consensus variables and the TSO model is
built and audited at its own true cold start.

Derivatives
-----------
PyNumero's ASL interface is NOT available in this environment
(`AmplInterface.available() == False`), so an exact sparse NLP Jacobian via
`PyomoNLP` cannot be used, and the plan forbids installing a large dependency
for this audit. Instead derivatives are taken with Pyomo's own reverse-mode
automatic differentiation (`pyomo.core.expr.calculus.derivatives.differentiate`,
`Modes.reverse_numeric` / `Modes.reverse_symbolic`), which is analytic and
exact, evaluated at the cold-start point. Second derivatives are obtained by
differentiating the symbolic first derivative again. The limitation is that
this is row-by-row Python-level AD rather than a single sparse ASL callback.

Diagnostic only. No production file is modified.

    python p53a_conditioning_audit.py [--max-models N] [--svd-tags a,b,c]
"""

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import network as network_module  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53A')

GRAD_BINS = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
TOP_N = 25

CURVATURE_FAMILIES = (
    'sess_snet_def', 'ess_snet_def', 'sess_comp', 'ess_comp', 'sg_capability',
    'branch_flow_limit', 'branch_flow_limit_ji', 'branch_current_limit',
    'voltage_mag_sqr_def', 'voltage_mag_def',
    'voltage_product_real_def', 'voltage_product_imag_def', 'r_sqr_def',
)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def safe_value(expr):
    try:
        return float(pe.value(expr, exception=False))
    except Exception:
        return None


def row_kind(con):
    """Classify a ConstraintData."""
    body = con.body
    try:
        degree = body.polynomial_degree()
    except Exception:
        degree = None
    linear = (degree is not None and degree <= 1)
    if con.equality:
        kind = 'linear equality' if linear else 'nonlinear equality'
    elif con.lower is not None and con.upper is not None:
        kind = 'ranged linear inequality' if linear else 'ranged nonlinear inequality'
    else:
        kind = 'linear inequality' if linear else 'nonlinear inequality'
    return kind, degree


def audit_model(network, model, params, tag, want_matrix=False):
    """Cold-start structural/derivative audit of one network SMOPF."""
    tol = 1e-5
    try:
        tol = float(params.solver_params.options.get('tol', 1e-5))
    except Exception:
        pass

    comp_stats = {}
    grad_bin_counts = {f'<{b:g}': 0 for b in GRAD_BINS}
    smallest_rows, largest_rows, tight_margins, worst_ratios = [], [], [], []
    col_sq = defaultdict(float)
    col_count = defaultdict(int)
    var_of_id = {}
    n_active_rows = 0
    zero_grad_rows = 0

    rows_for_matrix = []
    var_index = {}

    for comp in model.component_objects(pe.Constraint, active=True):
        cname = comp.local_name
        stat = comp_stats.setdefault(cname, {
            'component': cname, 'rows_total': 0, 'rows_active': 0,
            'kinds': defaultdict(int), 'degrees': defaultdict(int),
            'n_vars_min': None, 'n_vars_max': None,
            'grad_norm_min': None, 'grad_norm_max': None,
            'grad_norm_sum': 0.0, 'grad_norm_n': 0,
            'zero_gradient_rows': 0,
            'residual_max': 0.0,
            'min_margin_over_tol': None,
            'max_intra_row_ratio': None,
            'coef_abs_min': None, 'coef_abs_max': None,
        })
        for index in comp:
            con = comp[index]
            stat['rows_total'] += 1
            if not con.active:
                continue
            stat['rows_active'] += 1
            n_active_rows += 1

            kind, degree = row_kind(con)
            stat['kinds'][kind] += 1
            stat['degrees'][str(degree)] += 1

            body = con.body
            variables = [v for v in identify_variables(body, include_fixed=False)]
            nv = len(variables)
            stat['n_vars_min'] = nv if stat['n_vars_min'] is None else min(stat['n_vars_min'], nv)
            stat['n_vars_max'] = nv if stat['n_vars_max'] is None else max(stat['n_vars_max'], nv)

            body_val = safe_value(body)
            lower = safe_value(con.lower) if con.lower is not None else None
            upper = safe_value(con.upper) if con.upper is not None else None

            if con.equality and body_val is not None and lower is not None:
                residual = abs(body_val - lower)
                margin = None
            else:
                residual = 0.0
                margins = []
                if body_val is not None and lower is not None:
                    margins.append(body_val - lower)
                if body_val is not None and upper is not None:
                    margins.append(upper - body_val)
                margin = min(margins) if margins else None
                if margin is not None and margin < 0:
                    residual = -margin
            stat['residual_max'] = max(stat['residual_max'], residual)

            grads = []
            if nv:
                try:
                    grads = differentiate(body, wrt_list=variables, mode=Modes.reverse_numeric)
                    grads = [0.0 if g is None else float(g) for g in grads]
                except Exception:
                    grads = []

            if grads:
                absg = [abs(g) for g in grads]
                nz = [a for a in absg if a > 0.0]
                l2 = math.sqrt(sum(a * a for a in absg))
                linf = max(absg)
                cmin = min(nz) if nz else 0.0
                cmax = linf
                ratio = (cmax / cmin) if cmin > 0 else float('inf')

                stat['grad_norm_min'] = l2 if stat['grad_norm_min'] is None else min(stat['grad_norm_min'], l2)
                stat['grad_norm_max'] = l2 if stat['grad_norm_max'] is None else max(stat['grad_norm_max'], l2)
                stat['grad_norm_sum'] += l2
                stat['grad_norm_n'] += 1
                stat['coef_abs_min'] = cmin if stat['coef_abs_min'] is None else min(stat['coef_abs_min'], cmin)
                stat['coef_abs_max'] = cmax if stat['coef_abs_max'] is None else max(stat['coef_abs_max'], cmax)
                if stat['max_intra_row_ratio'] is None or (ratio != float('inf') and ratio > stat['max_intra_row_ratio']):
                    if ratio != float('inf'):
                        stat['max_intra_row_ratio'] = ratio

                for bound in GRAD_BINS:
                    if l2 < bound:
                        grad_bin_counts[f'<{bound:g}'] += 1
                if l2 == 0.0:
                    zero_grad_rows += 1
                    stat['zero_gradient_rows'] += 1

                record = {'tag': tag, 'component': cname, 'index': str(index),
                          'kind': kind, 'grad_l2': l2, 'grad_linf': linf,
                          'coef_min': cmin, 'coef_max': cmax,
                          'ratio': (ratio if ratio != float('inf') else None),
                          'residual': residual, 'n_vars': nv,
                          'body_value': body_val}
                smallest_rows.append((l2, record))
                largest_rows.append((-l2, record))
                if ratio != float('inf'):
                    worst_ratios.append((-ratio, record))

                for var, g in zip(variables, grads):
                    key = id(var)
                    col_sq[key] += g * g
                    col_count[key] += 1
                    if key not in var_of_id:
                        var_of_id[key] = var

                if want_matrix and con.equality:
                    cols = []
                    for var, g in zip(variables, grads):
                        key = id(var)
                        if key not in var_index:
                            var_index[key] = len(var_index)
                        cols.append((var_index[key], g))
                    rows_for_matrix.append(cols)

            if margin is not None:
                over_tol = margin / tol if tol else None
                if over_tol is not None:
                    if stat['min_margin_over_tol'] is None or over_tol < stat['min_margin_over_tol']:
                        stat['min_margin_over_tol'] = over_tol
                    tight_margins.append((over_tol, {
                        'tag': tag, 'component': cname, 'index': str(index),
                        'kind': kind, 'margin': margin, 'margin_over_tol': over_tol,
                        'body_value': body_val}))

        # keep the running top-N small
        smallest_rows = sorted(smallest_rows, key=lambda r: r[0])[:TOP_N]
        largest_rows = sorted(largest_rows, key=lambda r: r[0])[:TOP_N]
        worst_ratios = sorted(worst_ratios, key=lambda r: r[0])[:TOP_N]
        tight_margins = sorted(tight_margins, key=lambda r: r[0])[:TOP_N]

    # finalize component stats
    for stat in comp_stats.values():
        stat['kinds'] = dict(stat['kinds'])
        stat['degrees'] = dict(stat['degrees'])
        stat['grad_norm_mean'] = (stat['grad_norm_sum'] / stat['grad_norm_n']
                                  if stat['grad_norm_n'] else None)
        del stat['grad_norm_sum'], stat['grad_norm_n']

    # column diagnostics
    col_norms = {k: math.sqrt(v) for k, v in col_sq.items()}
    col_items = sorted(col_norms.items(), key=lambda kv: kv[1])
    def vname(key):
        try:
            return var_of_id[key].name
        except Exception:
            return '<unknown>'
    columns = {
        'n_columns_with_derivatives': len(col_norms),
        'n_zero_columns': sum(1 for v in col_norms.values() if v == 0.0),
        'n_columns_below_1e-10': sum(1 for v in col_norms.values() if v < 1e-10),
        'n_columns_below_1e-6': sum(1 for v in col_norms.values() if v < 1e-6),
        'smallest_columns': [{'var': vname(k), 'col_norm': v, 'n_rows': col_count[k]}
                             for k, v in col_items[:TOP_N]],
        'largest_columns': [{'var': vname(k), 'col_norm': v, 'n_rows': col_count[k]}
                            for k, v in col_items[-TOP_N:][::-1]],
    }

    out = {
        'tag': tag, 'ipopt_tol': tol,
        'n_active_rows': n_active_rows,
        'n_zero_gradient_rows': zero_grad_rows,
        'grad_norm_bin_counts': grad_bin_counts,
        'components': list(comp_stats.values()),
        'columns': columns,
        'top_smallest_grad_rows': [r[1] for r in smallest_rows],
        'top_largest_grad_rows': [r[1] for r in largest_rows],
        'top_tightest_margins': [r[1] for r in tight_margins],
        'top_intra_row_ratios': [r[1] for r in worst_ratios],
    }

    # objective gradient scale
    try:
        obj = list(model.component_data_objects(pe.Objective, active=True))[0]
        ovars = list(identify_variables(obj.expr, include_fixed=False))
        ograd = differentiate(obj.expr, wrt_list=ovars, mode=Modes.reverse_numeric)
        oabs = [abs(float(g)) for g in ograd if g is not None]
        nzo = [a for a in oabs if a > 0]
        out['objective'] = {
            'n_vars': len(ovars),
            'grad_l2': math.sqrt(sum(a * a for a in oabs)) if oabs else 0.0,
            'grad_linf': max(oabs) if oabs else 0.0,
            'grad_abs_min_nonzero': min(nzo) if nzo else 0.0,
            'n_zero_partials': sum(1 for a in oabs if a == 0.0),
            'value': safe_value(obj.expr),
        }
    except Exception as error:
        out['objective'] = {'error': str(error)}

    # curvature audit on the named families
    curvature = {}
    for comp in model.component_objects(pe.Constraint, active=True):
        cname = comp.local_name
        if cname not in CURVATURE_FAMILIES:
            continue
        worst = None
        samples = 0
        for index in comp:
            con = comp[index]
            if not con.active:
                continue
            samples += 1
            if samples > 40:      # sample rows; curvature is structurally uniform per family
                break
            body = con.body
            variables = list(identify_variables(body, include_fixed=False))
            if not variables:
                continue
            try:
                sym = differentiate(body, wrt_list=variables, mode=Modes.reverse_symbolic)
                mags = []
                for gi in sym:
                    second = differentiate(gi, wrt_list=variables, mode=Modes.reverse_numeric)
                    mags.extend(abs(float(s)) for s in second if s is not None)
                if mags:
                    m = max(mags)
                    if worst is None or m > worst['max_abs_second_derivative']:
                        worst = {'index': str(index), 'max_abs_second_derivative': m,
                                 'n_vars': len(variables)}
            except Exception:
                continue
        if worst is not None:
            curvature[cname] = worst
    out['curvature'] = curvature

    if want_matrix and rows_for_matrix:
        out['_matrix'] = (rows_for_matrix, len(var_index))
    return out


def matrix_diagnostics(rows_for_matrix, n_cols):
    """Sparse equality-Jacobian extremal singular values / rank estimate."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds
    data, ri, ci = [], [], []
    for i, cols in enumerate(rows_for_matrix):
        for j, g in cols:
            if g != 0.0:
                ri.append(i); ci.append(j); data.append(g)
    n_rows = len(rows_for_matrix)
    J = coo_matrix((data, (ri, ci)), shape=(n_rows, max(n_cols, 1))).tocsr()
    out = {'n_equality_rows': n_rows, 'n_columns': n_cols, 'nnz': int(J.nnz)}

    # exactly-zero equality rows are exact rank deficiency, independent of any SVD
    row_nnz = np.diff(J.indptr)
    out['n_exactly_zero_equality_rows'] = int((row_nnz == 0).sum())
    out['exact_rank_upper_bound'] = int(n_rows - out['n_exactly_zero_equality_rows'])

    try:
        s_large = svds(J, k=3, which='LM', return_singular_vectors=False)
        out['largest_singular_values'] = sorted([float(s) for s in s_large], reverse=True)
    except Exception as error:
        out['largest_singular_value_error'] = str(error)

    out['smallest_singular_values'] = None
    for solver in ('lobpcg', 'propack'):
        try:
            s_small = svds(J, k=3, which='SM', return_singular_vectors=False,
                           solver=solver, maxiter=3000)
            out['smallest_singular_values'] = sorted([float(s) for s in s_small])
            out['smallest_singular_value_solver'] = solver
            break
        except Exception as error:
            out[f'smallest_sv_error_{solver}'] = str(error)[:160]
    if out['smallest_singular_values'] is None:
        out['limitation'] = (
            'Sparse smallest-singular-value iterations did not converge for this '
            'model (ASL/PyomoNLP unavailable, no dense SVD attempted at this size). '
            'Rank deficiency is instead evidenced exactly by zero equality rows and '
            'by the near-collinearity scan below.')
    else:
        smin = min(out['smallest_singular_values'])
        smax = max(out.get('largest_singular_values') or [0.0])
        out['condition_estimate'] = (smax / smin) if smin > 0 else None
        out['rank_deficient_indicator'] = bool(smin < 1e-10)

    # near-collinear equality rows: compare normalized gradients within groups
    # sharing an identical variable support (cheap, catches duplicates/near-dupes)
    groups = defaultdict(list)
    for i, cols in enumerate(rows_for_matrix):
        support = frozenset(j for j, g in cols if g != 0.0)
        if support:
            groups[support].append(i)
    collinear = []
    for support, members in groups.items():
        if len(members) < 2 or len(members) > 60:
            continue
        vecs = {}
        for i in members:
            v = np.zeros(len(support))
            order = {j: k for k, j in enumerate(sorted(support))}
            for j, g in rows_for_matrix[i]:
                if j in order:
                    v[order[j]] = g
            n = np.linalg.norm(v)
            if n > 0:
                vecs[i] = v / n
        keys = list(vecs)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                cos = float(abs(np.dot(vecs[keys[a]], vecs[keys[b]])))
                if cos > 1 - 1e-8:
                    collinear.append({'row_a': keys[a], 'row_b': keys[b],
                                      'abs_cosine': cos, 'support_size': len(support)})
    out['n_near_collinear_equality_row_pairs'] = len(collinear)
    out['near_collinear_examples'] = collinear[:TOP_N]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-models', type=int, default=0,
                        help='audit only the first N models (0 = all)')
    parser.add_argument('--svd-tags', type=str, default='')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    svd_tags = set(t for t in args.svd_tags.split(',') if t)

    report = {'stage': 'P5.3-A', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'derivative_backend': {
                  'pynumero_asl_available': False,
                  'method': 'Pyomo reverse-mode AD (analytic), row-by-row',
                  'limitation': ('No ASL/PyomoNLP sparse Jacobian callback in this '
                                 'environment; no large dependency was installed per plan A2.')},
              'grad_bins': GRAD_BINS}

    audits = []
    matrices = {}
    original_run_smopf = network_module.Network.run_smopf
    counter = {'n': 0}

    def patched_run_smopf(self, model, params, from_warm_start=False, print_header=True):
        counter['n'] += 1
        if args.max_models and counter['n'] > args.max_models:
            return original_run_smopf(self, model, params,
                                      from_warm_start=from_warm_start,
                                      print_header=print_header)
        tag = f'{self.name}/{self.year}/{self.day}'
        want = (not svd_tags) or (tag in svd_tags)
        started = time.time()
        result = audit_model(self, model, params, tag, want_matrix=want)
        result['audit_seconds'] = time.time() - started
        if '_matrix' in result:
            matrices[tag] = result.pop('_matrix')
        audits.append(result)
        sys.__stdout__.write(f"  [audit] {tag}: {result['n_active_rows']} active rows, "
                             f"{result['n_zero_gradient_rows']} zero-gradient, "
                             f"{result['audit_seconds']:.1f}s\n")
        sys.__stdout__.flush()
        return original_run_smopf(self, model, params,
                                  from_warm_start=from_warm_start,
                                  print_header=print_header)

    quiet = io.StringIO()
    started = time.time()
    try:
        network_module.Network.run_smopf = patched_run_smopf
        with redirect_stdout(quiet):
            planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
            planning.read_planning_problem()
            candidate = srp._build_positive_bootstrap_candidate(
                planning, planning.params.benders.positive_bootstrap)
            consensus_vars, _dual = srp.create_admm_variables(planning)
            res = {'tso': dict(), 'dso': dict(), 'esso': dict()}
            _dm, res['dso'] = srp.create_distribution_networks_models(
                planning.distribution_networks, consensus_vars,
                candidate['total_capacity'],
                parallel_execution=planning.parallel_execution)
            _tm, res['tso'] = srp.create_transmission_network_model(
                planning, consensus_vars, candidate['total_capacity'])
            _em, res['esso'] = srp.create_shared_energy_storage_model(
                planning.shared_ess_data, consensus_vars, candidate['investment'])
    finally:
        network_module.Network.run_smopf = original_run_smopf
    report['wall_clock_s'] = time.time() - started
    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]
    report['n_models_audited'] = len(audits)
    report['models'] = audits

    # matrix diagnostics on the selected models
    print('[P5.3-A] computing equality-Jacobian diagnostics ...', flush=True)
    mat = {}
    for tag, (rows, ncols) in matrices.items():
        mat[tag] = matrix_diagnostics(rows, ncols)
        print(f"   {tag}: rows={mat[tag]['n_equality_rows']} cols={mat[tag]['n_columns']} "
              f"smin={mat[tag].get('smallest_singular_values')} "
              f"cond={mat[tag].get('condition_estimate')}", flush=True)
    report['equality_jacobian_diagnostics'] = mat

    out_path = os.path.join(OUT_DIR, 'p53a_audit.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[P5.3-A] report -> {out_path}')
    print(f"[P5.3-A] models audited: {len(audits)}  wall={report['wall_clock_s']:.1f}s")


if __name__ == '__main__':
    main()
