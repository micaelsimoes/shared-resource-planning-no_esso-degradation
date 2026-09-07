"""
Stage P5.3-A2, Parts A and B -- equality-Jacobian singular-value correction and
column-norm validation.

Part A. P5.3-A reported exactly-zero equality rows (24 DSO / 72 TSO) together
with NONZERO sigma_min from sparse `svds`. Those statements are inconsistent: a
matrix with exact zero rows is exactly row-rank deficient, so the full equality
Jacobian has sigma_min = 0. This script separates:

  * the FULL equality Jacobian (sigma_min = 0 exactly, nullity >= #zero rows);
  * the REDUCED Jacobian J_eq_reduced with only the exactly-zero rows removed.

Singular values are obtained exactly from the eigenvalues of J J^T (dense
symmetric eigensolve, tractable at these row counts) rather than from iterative
`svds`, which is what produced the earlier misleading numbers.

Part B. P5.3-A reported ~48 DSO columns with norm < 1e-10 including
`pij[31,...]` / `qij[31,...]`, which cannot be right because `pij_def` supplies
a derivative of exactly +1 w.r.t. `pij`. This script differentiates the actual
production rows to check, and determines the true cause.

Diagnostic only. No production file is modified.

    python p53a2_jacobian_correction.py
"""

import io
import json
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
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53A2')

TARGETS = {
    'case33_1/2025/Winter', 'case33_2/2030/Summer', 'case33_3/2025/Summer',
    'case9/2025/Winter', 'case9/2030/Spring',
}
RANK_TOL_REL = 1e-12          # relative to sigma_max


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def uninitialized_variables(model):
    """Variables entering the NLP with no Pyomo starting value."""
    out = defaultdict(int)
    for var in model.component_objects(pe.Var, active=True):
        for data in var.values():
            if data.value is None and not data.fixed:
                out[var.local_name] += 1
    return dict(out)


def row_gradient(body, variables):
    """(grads, error) -- numeric reverse-AD gradient, or the failure reason."""
    try:
        grads = differentiate(body, wrt_list=variables, mode=Modes.reverse_numeric)
        return [0.0 if g is None else float(g) for g in grads], None
    except Exception as error:
        return None, f'{type(error).__name__}: {str(error)[:120]}'


def analyse_model(model, tag, repair_uninitialized):
    """Assemble the equality Jacobian and column norms, tracking failures."""
    var_index, rows, row_meta = {}, [], []
    failures = defaultdict(int)
    failed_rows = []
    col_sq = defaultdict(float)
    var_of_id = {}

    # optional diagnostic repair: give uninitialized vars a nominal value so the
    # numeric AD can evaluate the rows that reference them
    repaired = {}
    if repair_uninitialized:
        for var in model.component_objects(pe.Var, active=True):
            for data in var.values():
                if data.value is None and not data.fixed:
                    lb, ub = data.lb, data.ub
                    nominal = 1.0
                    if lb is not None and ub is not None:
                        nominal = 0.5 * (lb + ub)
                    elif lb is not None:
                        nominal = lb
                    elif ub is not None:
                        nominal = ub
                    data.set_value(nominal, skip_validation=True)
                    repaired[var.local_name] = repaired.get(var.local_name, 0) + 1

    for comp in model.component_objects(pe.Constraint, active=True):
        cname = comp.local_name
        for index in comp:
            con = comp[index]
            if not con.active:
                continue
            body = con.body
            variables = list(identify_variables(body, include_fixed=False))
            if not variables:
                continue
            grads, error = row_gradient(body, variables)
            if grads is None:
                failures[cname] += 1
                if len(failed_rows) < 20:
                    failed_rows.append({'component': cname, 'index': str(index), 'error': error})
                continue
            for var, g in zip(variables, grads):
                key = id(var)
                col_sq[key] += g * g
                if key not in var_of_id:
                    var_of_id[key] = var
            if con.equality:
                cols = []
                for var, g in zip(variables, grads):
                    key = id(var)
                    if key not in var_index:
                        var_index[key] = len(var_index)
                    if g != 0.0:
                        cols.append((var_index[key], g))
                rows.append(cols)
                row_meta.append({'component': cname, 'index': str(index)})

    col_norms = {k: float(np.sqrt(v)) for k, v in col_sq.items()}
    return {
        'tag': tag,
        'repaired_uninitialized': repaired,
        'derivative_failures_by_component': dict(failures),
        'n_derivative_failures': int(sum(failures.values())),
        'failed_row_examples': failed_rows,
        'rows': rows, 'row_meta': row_meta,
        'n_columns_seen': len(var_index),
        'col_norms': col_norms, 'var_of_id': var_of_id,
    }


def singular_spectrum(rows, n_cols):
    """Exact singular values via eigenvalues of J J^T (dense symmetric)."""
    from scipy.sparse import coo_matrix
    data, ri, ci = [], [], []
    for i, cols in enumerate(rows):
        for j, g in cols:
            ri.append(i); ci.append(j); data.append(g)
    n_rows = len(rows)
    J = coo_matrix((data, (ri, ci)), shape=(n_rows, max(n_cols, 1))).tocsr()
    zero_rows = [i for i in range(n_rows) if J.indptr[i + 1] == J.indptr[i]]

    def spectrum(matrix):
        gram = (matrix @ matrix.T).toarray()
        eig = np.linalg.eigvalsh(gram)
        eig = np.clip(eig, 0.0, None)
        return np.sqrt(eig)[::-1]        # descending

    full = spectrum(J)
    smax = float(full[0]) if full.size else 0.0
    rank_tol = smax * RANK_TOL_REL
    full_rank = int((full > rank_tol).sum())

    keep = [i for i in range(n_rows) if i not in set(zero_rows)]
    Jr = J[keep, :] if keep else J
    red = spectrum(Jr)
    red_smin = float(red[-1]) if red.size else 0.0
    red_smax = float(red[0]) if red.size else 0.0
    red_rank = int((red > (red_smax * RANK_TOL_REL)).sum())

    return {
        'full': {
            'n_rows': n_rows, 'n_columns': n_cols, 'nnz': int(J.nnz),
            'n_exactly_zero_rows': len(zero_rows),
            'structural_minimum_nullity_from_zero_rows': len(zero_rows),
            'sigma_max': smax,
            'sigma_min': float(full[-1]),
            'sigma_min_is_exactly_zero': bool(full[-1] <= rank_tol),
            'numerical_rank': full_rank,
            'nullity_row_space': n_rows - full_rank,
            'condition_number': None if full[-1] <= rank_tol else float(smax / full[-1]),
            'condition_note': ('formally infinite: exact zero rows make the full '
                               'equality Jacobian exactly row-rank deficient'),
            'smallest_5': [float(v) for v in full[-5:]],
        },
        'reduced': {
            'n_rows': Jr.shape[0], 'n_columns': n_cols,
            'rows_removed': len(zero_rows),
            'sigma_max': red_smax,
            'sigma_min_nonzero_subspace': red_smin,
            'condition_number': (red_smax / red_smin) if red_smin > 0 else None,
            'numerical_rank': red_rank,
            'rank_deficient_beyond_zero_rows': bool(red_rank < Jr.shape[0]),
            'extra_nullity_beyond_zero_rows': int(Jr.shape[0] - red_rank),
            'smallest_5': [float(v) for v in red[-5:]],
        },
    }


def part_b_spot_checks(model):
    """Differentiate the actual production rows for the challenged variables."""
    checks = {}
    targets = [
        ('pij_def', 'pij'), ('qij_def', 'qij'),
        ('pji_def', 'pji'), ('qji_def', 'qji'),
    ]
    for comp_name, var_name in targets:
        comp = getattr(model, comp_name, None)
        var = getattr(model, var_name, None)
        if comp is None or var is None:
            checks[comp_name] = {'present': False}
            continue
        entry = {'present': True, 'samples': []}
        for count, index in enumerate(comp):
            if count >= 3:
                break
            con = comp[index]
            try:
                target_var = var[index]
            except Exception:
                continue
            grads, error = row_gradient(con.body, [target_var])
            entry['samples'].append({
                'index': str(index),
                'd_row_d_own_variable': (grads[0] if grads else None),
                'equals_one': (grads is not None and abs(abs(grads[0]) - 1.0) < 1e-12),
                'error': error,
            })
        checks[comp_name] = entry

    extra = {}
    for var_name in ('f', 'shared_es_sch', 'shared_es_sdch'):
        var = getattr(model, var_name, None)
        if var is None:
            extra[var_name] = {'present': False}
            continue
        extra[var_name] = {'present': True, 'n_data': len(list(var.values()))}
    checks['spot_variables'] = extra
    return checks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.3-A2 parts A+B', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'rank_tol_relative': RANK_TOL_REL,
              'method': ('exact singular values from eigenvalues of J J^T '
                         '(dense symmetric eigensolve), replacing the iterative '
                         'svds estimates used in P5.3-A')}

    captured = {}
    original = network_module.Network.run_smopf

    def patched(self, model, params, from_warm_start=False, print_header=True):
        tag = f'{self.name}/{self.year}/{self.day}'
        if tag in TARGETS and tag not in captured:
            captured[tag] = model.clone()
            sys.__stdout__.write(f'  [captured] {tag}\n'); sys.__stdout__.flush()
        return original(self, model, params, from_warm_start=from_warm_start,
                        print_header=print_header)

    quiet = io.StringIO()
    started = time.time()
    try:
        network_module.Network.run_smopf = patched
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
    finally:
        network_module.Network.run_smopf = original
    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]
    report['capture_seconds'] = time.time() - started

    models_out = {}
    for tag, model in captured.items():
        print(f'[A2] analysing {tag} ...', flush=True)
        entry = {'uninitialized_variables_before_repair': uninitialized_variables(model)}

        # (i) exactly as P5.3-A did it -- no repair
        raw = analyse_model(model, tag, repair_uninitialized=False)
        entry['without_repair'] = {
            'n_derivative_failures': raw['n_derivative_failures'],
            'derivative_failures_by_component': raw['derivative_failures_by_component'],
            'failed_row_examples': raw['failed_row_examples'],
        }
        cn = raw['col_norms']
        def named(cn, raw, prefixes):
            out = {}
            for key, value in cn.items():
                name = raw['var_of_id'][key].name
                for pref in prefixes:
                    if name.startswith(pref):
                        out.setdefault(pref, []).append((name, value))
            return {k: sorted(v)[:3] for k, v in out.items()}
        entry['without_repair']['sample_column_norms'] = named(
            cn, raw, ('pij[31', 'qij[31', 'f[0', 'shared_es_sch[', 'shared_es_sdch['))
        entry['without_repair']['n_columns_below_1e-10'] = int(
            sum(1 for v in cn.values() if v < 1e-10))

        # (ii) repaired: give uninitialized variables a nominal value
        model2 = captured[tag].clone()
        fixed = analyse_model(model2, tag, repair_uninitialized=True)
        cn2 = fixed['col_norms']
        entry['with_repair'] = {
            'repaired_uninitialized': fixed['repaired_uninitialized'],
            'n_derivative_failures': fixed['n_derivative_failures'],
            'derivative_failures_by_component': fixed['derivative_failures_by_component'],
            'n_columns_below_1e-10': int(sum(1 for v in cn2.values() if v < 1e-10)),
            'sample_column_norms': named(cn2, fixed, ('pij[31', 'qij[31', 'f[0',
                                                      'shared_es_sch[', 'shared_es_sdch[')),
        }
        entry['part_b_spot_checks'] = part_b_spot_checks(model2)

        print(f'    spectra ({len(fixed["rows"])} equality rows) ...', flush=True)
        entry['equality_jacobian'] = singular_spectrum(fixed['rows'], fixed['n_columns_seen'])
        # which components own the exactly-zero rows
        zero_owners = defaultdict(int)
        for i, cols in enumerate(fixed['rows']):
            if not cols:
                zero_owners[fixed['row_meta'][i]['component']] += 1
        entry['equality_jacobian']['zero_row_components'] = dict(zero_owners)
        models_out[tag] = entry

    report['models'] = models_out
    out_path = os.path.join(OUT_DIR, 'p53a2_jacobian.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[A2] report -> {out_path}')
    for tag, e in models_out.items():
        f, r = e['equality_jacobian']['full'], e['equality_jacobian']['reduced']
        print(f"  {tag:22s} FULL rows={f['n_rows']} zero={f['n_exactly_zero_rows']} "
              f"smin={f['sigma_min']:.3e} rank={f['numerical_rank']} | "
              f"REDUCED rows={r['n_rows']} smin={r['sigma_min_nonzero_subspace']:.3e} "
              f"cond={r['condition_number']:.3e} extraNullity={r['extra_nullity_beyond_zero_rows']}")
        print(f"      derivative failures: no-repair={e['without_repair']['n_derivative_failures']} "
              f"repaired={e['with_repair']['n_derivative_failures']} | "
              f"cols<1e-10: {e['without_repair']['n_columns_below_1e-10']} -> "
              f"{e['with_repair']['n_columns_below_1e-10']}")


if __name__ == '__main__':
    main()
