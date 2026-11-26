"""P5.16 D0 / T0.2-ipopt-interior -- ONE declared IPOPT solve of one exported group relaxation (smooth form).

Question (Planner): does the exported relaxation of the x = 0, variant C, 2025 Spring group (T0.1c LP
`P516D0/t01c/lp/x0/2025_Spring_calendar.lp`, manifest sha256 d2955128...) have a strictly feasible interior and a finite
optimum reachable by a second-order interior-point method -- i.e. is the Clarabel stall (P5_16_D0_T02_CLARABEL_REPORT.md)
a solver / conditioning matter or a well-posedness defect of the relaxation itself?

Zero-solve part (before the one solve):
  1. Redundancy inventory: rhs-0 '=' rows whose columns are all pinned (`p516_d0_clarabel.pinned_zero_columns` /
     `presolve_report`, through `build_conic(data, presolve=True)`, which builds a conic matrix and solves nothing);
     the numerical rank of the equality block (dense SVD per connected component of the row-column graph, scipy
     `svdvals`; numerical rank = #{sigma > sigma_max * max(m_k, n_k) * eps} per component (numpy `matrix_rank`'s
     default), with the singular-value counts below 1e-16 .. 1e-4 sigma_max reported for robustness); the dependent
     rows named by QR with column pivoting of the component's transpose (scipy `qr(pivoting=True)`); done on A_eq and
     on A_eq without the bound-fixed columns (= the conic zero cone [A_eq; E_fixed] up to |fixed| independent rows);
     columns in no row (free ones separately); zero-cost columns appearing only in quadratic rows; all by family.
  2. Quadratic-row inventory (pattern by family via `p516_d0_clarabel.classify_quadratic_row`; circles with r = 0 and
     r < 1e-4; the minimum LB of the bilinear factors; Lorentz cones with a pinned head).
  3. The smooth-form NLP, built by `p516_d0_t02.build_ipopt_model` EXACTLY (its source is extracted from
     `p516_d0_t02.py` with `ast` and executed here; the extracted text's sha256 is pinned and checked before use):
     every linear row; every quadratic row as written (the rotated cone as the bilinear inequality
     WijR^2 + WijI^2 <= Wii Wjj with Wii >= v_min^2 > 0 on the LP bounds; circles p^2 + q^2 <= r); the same bounds
     and linear objective; the initial point = the T0.1c certified mapped point clipped into the bounds. The NL file
     is written once before the solve (to a temporary directory outside the repository) and its header checked
     against the LP dimensions.
  4. The initial infeasibility at that start (LP-row violations by class and family).

Declared solves: exactly ONE IPOPT solve (`_leaf_solve`, the only permitted call site). Guards: `SolveProfileGuard`
(Pyomo OptSolver.solve + SystemCallSolver._execute_command; permitted count 1 + 1, verified exactly);
`p516_d0_t02.GurobiLeafGuard` and `p516_d0_clarabel.ClarabelLeafGuard` with no permitted site (counts 0, verified).
The audit hook of `p516_d0_t02` raises on any process launch outside the leaf and records every write-open.
IPOPT options (task): tol 1e-9, max_iter 3000, mu_strategy adaptive, print_level 5, print_timing_statistics yes,
linear_solver ma97; everything else at IPOPT 3.14.18 defaults. No retry: a failed solve is recorded, not repeated.

Reader: gurobipy reads the LP (never optimized) with HOME pointed at an empty temporary directory (the user licence is
bound to another host; the bundled restricted licence reads models of any size). The parent's preflight warms
matplotlib's font cache inside that HOME (production network.py imports pyplot on unpickling; `fc-list` would be a
process launch in the audited child).

Outputs: `data/SRP1/Results/P516D0/t02_ipopt_interior/` (new; refuses if it exists): predictions.json (first),
interior.json, logs/<tag>.ipopt.log, vectors/<tag>.npz, run.log, provenance files, manifest_sha256.json.

Usage (one process, attached, alone):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_ipopt_interior.py
Rehearsal (zero solves, scratch under /tmp, never evidence; the post-processing runs on the start point, labelled
STAND-IN):
    P516_D0_IPOPT_INTERIOR_REHEARSAL_DIR=/tmp/... /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_ipopt_interior.py

Exit codes: 0 all gates pass; 2 a gate failed (incl. IPOPT not Optimal); 3 a stop fired; 4 refused (root exists);
5 refused (input hash mismatch / reused source changed); 6 refused (reader or IPOPT unavailable, nothing created).
"""

import ast
import collections
import gc
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
import time

import p516_d0_t01b as T1B

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P516D0', 't02_ipopt_interior')
_REHEARSAL = os.environ.get('P516_D0_IPOPT_INTERIOR_REHEARSAL_DIR')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516_D0_IPOPT_INTERIOR_REHEARSAL_DIR must be under /tmp, got {_REHEARSAL!r}')
    OUT_DIR = os.path.abspath(_REHEARSAL)
DRY = bool(_REHEARSAL)

LOG_PATH = os.path.join(OUT_DIR, 'run.log')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
PROVENANCE_PARENT = os.path.join(OUT_DIR, 'provenance_parent.json')

T01C_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01c')
T01C_MANIFEST_SHA256 = 'd29551285683ede902b71188ba6973f33e15d53018f8a77da6e5fd833514afb5'
T01B_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01b')
T01B_MANIFEST_SHA256 = '4e5bbd45234598094593a8daca807a576590eee52f492f5281b63cf621876562'

CELL, YEAR, DAY, MODE, VAR = 'x0', 2025, 'Spring', 'calendar', 'C'
TAG = f'{CELL}_{YEAR}_{DAY}_{VAR}'
DECLARED_SOLVES = 1
IPOPT_OPTIONS = {'tol': 1e-9, 'max_iter': 3000, 'mu_strategy': 'adaptive', 'print_level': 5,
                 'print_timing_statistics': 'yes', 'linear_solver': 'ma97'}
# `p516_d0_t02` closures reused verbatim (sha256 of the dedented source text extracted with ast at HEAD ff3153ff)
REUSED_T02_SOURCE_SHA256 = {
    'build_ipopt_model': '8a6290195c0d08e2b4243b460b1c21e6d3d501e02edeb15b253f902a7f935b5b',
    'parse_ipopt_log': '682f5b6dea26f49089f8e373f7e241eccb0023c50395e71640620c304505c99c',
}

# thresholds, stated before the run
OBJ_IDENTITY_REL = 1e-8          # LP objective at the certified point vs the T0.1c record (as p516_d0_t02)
CONTAINMENT_TOL_REL = 1e-6       # I3: Q_LB <= Q_cert+ + 1e-6 |Q_cert+|
ACTIVE_REL = 1e-6                # a cone / circle is ACTIVE when its relative slack is <= 1e-6
SMALL_CIRCLE_R = 1e-4
SCALE_FLOOR = 1e-12            # a cone whose scale (k u v, e t^2 or r) is <= 1e-12 is reported by its ABSOLUTE slack
I2_MAX_ITER = 500
I4_RANGE = (1e-4, 1e-2)
I5_POINT, I5_FACTOR = 2.45e-2, 2.0
I1_MIN_DEFICIENCY = 216
SLACK_DECADES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

PREDICTIONS = {
    'I1': 'the equality block has numerical rank deficiency >= 216',
    'I2': 'IPOPT converges (Optimal) within 500 iterations without restoration from the certified start',
    'I3': 'containment holds: Q_LB <= Q_cert+ + tol',
    'I4': 'the group gap (Q_cert+ - Q_LB) / Q_cert+ is in [1e-4, 1e-2] relative (H2 point estimate 1e-3)',
    'I5': 'the TN rank residual max is ~ 2.45e-2 relative (within a factor 2)',
}
OPERATIONALISATION = {
    'I1': ('held iff the numerical rank deficiency of A_eq (the LP\'s \'=\' rows, all columns) is >= 216; the '
           'deficiency of A_eq without the bound-fixed columns (the conic zero cone) is reported beside it'),
    'I2': ('held iff IPOPT\'s EXIT line is "Optimal Solution Found." AND the iteration count is <= 500 AND no '
           'restoration iteration (an iteration line numbered "<k>r") appears in the log'),
    'I3': 'held iff Q_LB <= Q_cert+ + 1e-6 |Q_cert+| (Q_LB = IPOPT objective; Q_cert+ from T0.1c objective.json)',
    'I4': 'held iff 1e-4 <= (Q_cert+ - Q_LB) / Q_cert+ <= 1e-2 at an Optimal solve; indeterminate otherwise',
    'I5': ('held iff the TSO max relative rank residual (Wii Wjj - WR^2 - WI^2) / (Wii Wjj) over branch-periods lies '
           'in [2.45e-2 / 2, 2.45e-2 * 2] at an Optimal solve; indeterminate otherwise'),
    'gap_label': ('the gap is a single-group, single-solver, IPOPT-tolerance value (tol 1e-9 on the scaled NLP '
                  'error), NOT a certified bound'),
    'active': 'a cone / circle is active when its relative slack is <= 1e-6 (rotated: (k u v - sum d x^2) / (k u v); '
              'circle: (r - sum d x^2) / r; Lorentz: (e t^2 - sum d x^2) / (e t^2)); a row whose scale (k u v, '
              'e t^2 or r) is <= 1e-12 has no relative slack and is reported by its absolute slack (scale - sum d x^2)',
    'interior': ('read from IPOPT\'s barrier behaviour: an Optimal exit with the final barrier parameter mu -> small '
                 '(lg(mu) of the last iteration) from IPOPT\'s strictly interior slack initialisation; no sampling'),
}

sha256_file = T1B.sha256_file
np = None            # bound in the child after the guards are armed


def rel(path):
    return os.path.relpath(path, REPO)


def _load_dotenv_ipopt():
    path = os.path.join(REPO, '.env')
    value = None
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith('NLP_SOLVER_PATH='):
                    value = line.split('=', 1)[1].strip().strip('"').strip("'")
    return value or os.environ.get('NLP_SOLVER_PATH')


def extract_t02_function(name):
    """The dedented source text of a function defined anywhere in p516_d0_t02.py (closures included), and its sha256."""
    path = os.path.join(REPO, 'p516_d0_t02.py')
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            text = textwrap.dedent(ast.get_source_segment(src, node, padded=True))
            return text, hashlib.sha256(text.encode()).hexdigest(), node.lineno
    raise KeyError(name)


def fam(name):
    from p516_d0_t02 import parse_pyomo_name
    blk, f_, _idx = parse_pyomo_name(name)
    return f'{(blk or "top").split("_")[0]}.{f_}'


# ======================================================================================================================
#  Zero-solve inventories
# ======================================================================================================================
def equality_rank_inventory(data, fixed):
    """Numerical rank of the equality block by dense SVD per connected component of its row-column graph."""
    import scipy.sparse as sp
    import scipy.linalg as sla
    from scipy.sparse.csgraph import connected_components
    eq = np.nonzero(data['sense'] == '=')[0]
    out = {'method': ('connected components of the bipartite row-column graph (scipy.sparse.csgraph); per component '
                      'dense singular values (scipy.linalg.svdvals); numerical rank = #{sigma > sigma_max_k * '
                      'max(m_k, n_k) * eps} (numpy matrix_rank default tolerance, per component); deficiency = '
                      'm_k - rank_k (includes m_k - n_k when a component has more rows than columns); dependent rows '
                      'named as the non-pivot columns of a pivoted QR of the component transpose '
                      '(scipy.linalg.qr(pivoting=True)) -- one admissible choice, not unique'),
           'eps': float(np.finfo(float).eps)}
    for label, colmask in (('A_eq', np.ones(len(data['var_names']), dtype=bool)), ('A_eq_without_fixed_columns',
                                                                                  ~fixed)):
        cols_kept = np.nonzero(colmask)[0]
        M = data['A'][eq][:, cols_kept].tocsr()
        m, n = M.shape
        t0 = time.time()
        empty_rows = np.nonzero(np.diff(M.indptr) == 0)[0]
        B = sp.bmat([[None, M], [M.T, None]]).tocsr()
        B.data[:] = 1.0
        _nc, lab = connected_components(B, directed=False)
        rl, cl = lab[:m], lab[m:]
        rows_by = collections.defaultdict(list)
        for i, k in enumerate(rl):
            rows_by[int(k)].append(i)
        cols_by = collections.defaultdict(list)
        for j, k in enumerate(cl):
            cols_by[int(k)].append(j)
        rank_total, deficiency, structural = 0, 0, 0
        rel_counts = {f'{t:.0e}': 0 for t in (1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)}
        min_rel_sv_kept = math.inf
        max_rel_sv_dropped = 0.0
        deficient_components = collections.Counter()
        dependent_by_family = collections.Counter()
        rows_in_deficient_components_by_family = collections.Counter()
        largest = (0, 0)
        examples = []
        for k, ri in rows_by.items():
            ci = cols_by.get(k, [])
            if not ci:                      # an empty row: rank 0 (counted as deficiency)
                deficiency += len(ri)
                continue
            largest = max(largest, (len(ri), len(ci)))
            D = M[ri][:, ci].toarray()
            sv = sla.svdvals(D)
            tol = sv.max() * max(D.shape) * out['eps']
            r_k = int(np.sum(sv > tol))
            rank_total += r_k
            for key in rel_counts:
                rel_counts[key] += int(np.sum(sv / sv.max() < float(key)))
            kept = sv[sv > tol]
            dropped = sv[sv <= tol]
            min_rel_sv_kept = min(min_rel_sv_kept, float(kept.min() / sv.max()))
            if len(dropped):
                max_rel_sv_dropped = max(max_rel_sv_dropped, float(dropped.max() / sv.max()))
            d_k = len(ri) - r_k
            structural += max(len(ri) - len(ci), 0)
            if d_k > 0:
                deficiency += d_k
                deficient_components[f'{len(ri)} rows x {len(ci)} cols'] += 1
                _R, piv = sla.qr(D.T, mode='r', pivoting=True)
                for p in piv[r_k:]:
                    dependent_by_family[fam(data['con_names'][eq[ri[p]]])] += 1
                for i in ri:
                    rows_in_deficient_components_by_family[fam(data['con_names'][eq[i]])] += 1
                if len(examples) < 3:
                    examples.append({'rows': [data['con_names'][eq[i]] for i in ri],
                                     'columns': [data['var_names'][cols_kept[j]] for j in ci],
                                     'matrix': D.tolist(), 'rank': r_k})
        out[label] = {'rows': m, 'columns': n, 'n_empty_rows': int(len(empty_rows)),
                      'n_components_with_rows': len(rows_by), 'largest_component_rows_cols': list(largest),
                      'numerical_rank': rank_total, 'rank_deficiency': deficiency,
                      'structural_deficiency_rows_minus_cols': structural,
                      'singular_values_below_rel_threshold_count': rel_counts,
                      'min_rel_singular_value_counted_in_rank': min_rel_sv_kept,
                      'max_rel_singular_value_dropped': max_rel_sv_dropped,
                      'deficient_components_by_shape': dict(deficient_components),
                      'dependent_rows_by_family_qr_choice': dict(dependent_by_family),
                      'rows_in_deficient_components_by_family': dict(rows_in_deficient_components_by_family),
                      'examples': examples, 'wall_s': time.time() - t0}
    return out


def column_inventory(data):
    n = len(data['var_names'])
    lin_cnt = np.diff(data['A'].tocsc().indptr)
    q_cnt = np.zeros(n, dtype=int)
    for q in data['qrows']:
        cols = {a for a, _b, _c in q['quad']} | {b_ for _a, b_, _c in q['quad']} | {a for a, _c in q['lin']}
        for c_ in cols:
            q_cnt[c_] += 1
    lb, ub, c = data['lb'], data['ub'], data['c']
    free = ~np.isfinite(lb) & ~np.isfinite(ub)
    unbounded = ~(np.isfinite(lb) & np.isfinite(ub))
    fixed = np.isfinite(lb) & (lb == ub)

    def by_family(mask):
        return dict(sorted(collections.Counter(fam(data['var_names'][k]) for k in np.nonzero(mask)[0]).items()))
    no_row = (lin_cnt == 0) & (q_cnt == 0)
    cone_only_zero_cost = (lin_cnt == 0) & (q_cnt > 0) & (c == 0.0)
    return {'n_columns': n, 'n_fixed_by_bounds': int(fixed.sum()), 'n_fixed_at_zero': int((fixed & (lb == 0)).sum()),
            'n_free': int(free.sum()), 'n_unbounded': int(unbounded.sum()),
            'in_no_row': {'count': int(no_row.sum()), 'by_family': by_family(no_row)},
            'free_in_no_row': {'count': int((no_row & free).sum()), 'by_family': by_family(no_row & free)},
            'zero_cost_in_quadratic_rows_only': {'count': int(cone_only_zero_cost.sum()),
                                                 'by_family': by_family(cone_only_zero_cost),
                                                 'n_free': int((cone_only_zero_cost & free).sum())},
            'objective': {'n_nonzero_cost': int(np.sum(c != 0.0)), 'max_abs_cost': float(np.max(np.abs(c))),
                          'min_abs_nonzero_cost': float(np.min(np.abs(c[c != 0.0]))) if np.any(c != 0.0) else None}}


def quadratic_inventory(data, pinned):
    import p516_d0_clarabel as C
    classified = []
    for j, q in enumerate(data['qrows']):
        classified.append(C.classify_quadratic_row(q, data['lb'], float(data['qc_rhs'][j]), str(data['qc_sense'][j])))
    by = collections.Counter((p, fam(data['qc_names'][j])) for j, (p, _s) in enumerate(classified))
    circles = [(j, s) for j, (p, s) in enumerate(classified) if p == 'circle']
    small = collections.defaultdict(lambda: {'count': 0, 'r_min': math.inf, 'r_max': 0.0})
    for j, s in circles:
        if s['r'] < SMALL_CIRCLE_R:
            rec = small[fam(data['qc_names'][j])]
            rec['count'] += 1
            rec['r_min'] = min(rec['r_min'], s['r'])
            rec['r_max'] = max(rec['r_max'], s['r'])
    factor_lb = collections.defaultdict(lambda: math.inf)
    for j, (p, s) in enumerate(classified):
        if p == 'rotated':
            for k in (s['u'], s['v']):
                f_ = fam(data['var_names'][k])
                factor_lb[f_] = min(factor_lb[f_], float(data['lb'][k]))
    lorentz_pinned = sum(1 for p, s in classified if p == 'lorentz' and pinned[s['t']])
    return classified, {
        'by_pattern_and_family': {f'{p} {f_}': v for (p, f_), v in sorted(by.items())},
        'n_circles': len(circles), 'n_circles_r_eq_0': sum(1 for _j, s in circles if s['r'] == 0.0),
        'n_circles_r_lt_1e-4': sum(v['count'] for v in small.values()),
        'circles_r_lt_1e-4_by_family': {k: dict(v) for k, v in small.items()},
        'min_lb_of_bilinear_factors_by_family': dict(factor_lb),
        'n_lorentz': sum(1 for p, _s in classified if p == 'lorentz'), 'n_lorentz_with_pinned_head': lorentz_pinned,
        'note': ('a Lorentz cone with a pinned head (t = 0 forced by rhs-0 equality chains) is {x = 0}: no strict '
                 'interior, and the smooth row sum d x^2 - e t^2 <= 0 has a vanishing gradient at every feasible '
                 'point (no constraint qualification there); built as written, per p516_d0_t02')}


def violations(data, x, top=8):
    import p516_d0_clarabel as C
    out = C.returned_point_violation(data, x)
    lpv = C.lp_row_violations(data, x)
    qv = np.maximum(lpv['quad_val'], 0.0)
    fams = {}
    for j in np.nonzero(qv > 0.0)[0]:
        f_ = fam(data['qc_names'][j])
        fams[f_] = max(fams.get(f_, 0.0), float(qv[j]))
    out['quadratic']['top_families'] = dict(sorted(fams.items(), key=lambda t: -t[1])[:top])
    return out


def cone_slacks(data, classified, x):
    """Signed relative slack of every quadratic row (>= 0 inside; < 0 violated), by pattern and family."""
    rel_slack = np.full(len(classified), np.nan)
    abs_val = np.zeros(len(classified))
    scale = np.zeros(len(classified))
    for j, (p, s) in enumerate(classified):
        if p == 'rotated':
            sq = sum(d * x[k] ** 2 for k, d in s['x'])
            sc = s['k'] * x[s['u']] * x[s['v']]
        elif p == 'lorentz':
            sq = sum(d * x[k] ** 2 for k, d in s['x'])
            sc = s['e'] * x[s['t']] ** 2
        else:
            sq = sum(d * x[k] ** 2 for k, d in s['x'])
            sc = s['r']
        abs_val[j] = sc - sq
        scale[j] = sc
        if sc > SCALE_FLOOR:
            rel_slack[j] = (sc - sq) / sc
    summary = {}
    for j, (p, _s) in enumerate(classified):
        key = f'{p} {fam(data["qc_names"][j])}'
        rec = summary.setdefault(key, {'count': 0, 'n_active_rel_le_1e-6': 0, 'n_scale_zero': 0, 'min_rel_slack': None,
                                       'max_rel_slack': None, 'min_abs_slack': math.inf,
                                       'rel_slack_le': {f'{t:.0e}': 0 for t in SLACK_DECADES}})
        rec['count'] += 1
        rec['min_abs_slack'] = min(rec['min_abs_slack'], float(abs_val[j]))
        if np.isnan(rel_slack[j]):
            rec['n_scale_zero'] += 1
            continue
        r_ = float(rel_slack[j])
        rec['min_rel_slack'] = r_ if rec['min_rel_slack'] is None else min(rec['min_rel_slack'], r_)
        rec['max_rel_slack'] = r_ if rec['max_rel_slack'] is None else max(rec['max_rel_slack'], r_)
        rec['n_active_rel_le_1e-6'] += int(r_ <= ACTIVE_REL)
        for t in SLACK_DECADES:
            rec['rel_slack_le'][f'{t:.0e}'] += int(r_ <= t)
    finite = rel_slack[~np.isnan(rel_slack)]
    return rel_slack, abs_val, scale, {
        'by_pattern_and_family': summary, 'n_rows': len(classified), 'n_scale_zero': int(np.isnan(rel_slack).sum()),
        'n_active_rel_le_1e-6': int(np.sum(finite <= ACTIVE_REL)),
        'min_rel_slack': float(finite.min()) if len(finite) else None,
        'argmin_row': data['qc_names'][int(np.nanargmin(rel_slack))] if len(finite) else None,
        'max_abs_value_on_scale_zero_rows': float(np.max(np.abs(abs_val[np.isnan(rel_slack)]), initial=0.0)),
        'definition': OPERATIONALISATION['active']}


ITER_RE = re.compile(r'^[ \t]*(\d+)(r?)[ \t]+([-+]?\d\.\d+e[-+]\d+)[ \t]+(\d\.\d+e[-+]\d+)[ \t]+(\d\.\d+e[-+]\d+)'
                     r'[ \t]+(-?[ \t]*\d+\.\d|-[ \t]*inf)[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)'
                     r'(?:[ \t]+(\S+))?[ \t]*$',
                     re.M)


def parse_iterations(path):
    """Every iteration line of an IPOPT print_level-5 log: (iter, restoration flag, objective, inf_pr, inf_du,
    lg(mu), ||d||, lg(rg), alpha_du, alpha_pr, ls), plus the timing-statistics block."""
    text = open(path, errors='replace').read()
    rows = []
    for m_ in ITER_RE.finditer(text):
        def num(s_):
            try:
                return float(re.sub(r'[a-zA-Z]$', '', s_)) if s_ != '-' else None
            except ValueError:
                return None
        rows.append({'iter': int(m_.group(1)), 'restoration': m_.group(2) == 'r', 'objective': float(m_.group(3)),
                     'inf_pr': float(m_.group(4)), 'inf_du': float(m_.group(5)),
                     'lg_mu': float(m_.group(6).replace(' ', '')) if 'inf' not in m_.group(6) else None,
                     'd_norm': num(m_.group(7)), 'lg_rg': m_.group(8), 'alpha_du': num(m_.group(9)),
                     'alpha_pr': m_.group(10), 'ls': m_.group(11), 'info': m_.group(12)})
    timing = None
    if 'Timing Statistics' in text:
        timing = text.split('Timing Statistics', 1)[1].split('EXIT:', 1)[0]
    tot = re.search(r'Total seconds in IPOPT\s*(?:\(w/o function evaluations\))?\s*=\s*([0-9.]+)', text)
    tot_wall = re.findall(r'Total seconds in IPOPT[^\n]*', text)
    final = {}                                  # LAST occurrence (robust to per-iteration summaries)
    for key, label in (('objective', 'Objective'), ('dual_infeasibility', 'Dual infeasibility'),
                       ('constraint_violation', 'Constraint violation'),
                       ('variable_bound_violation', 'Variable bound violation'),
                       ('complementarity', 'Complementarity'), ('overall_nlp_error', 'Overall NLP error')):
        hits = re.findall(rf'^{label}\.*:\s+(\S+)\s+(\S+)', text, re.M)
        final[key] = {'scaled': float(hits[-1][0]), 'unscaled': float(hits[-1][1]), 'n_occurrences': len(hits)} \
            if hits else None
    it_count = re.findall(r'Number of Iterations\.+:\s+(\d+)', text)
    exits = re.findall(r'^EXIT: (.*)$', text, re.M)
    evals = {k: int(v) for k, v in re.findall(r'Number of (\w[\w ]+?) evaluations\s*=\s*(\d+)', text)}
    return {'iterations': rows, 'n_iteration_lines': len(rows), 'final_summary_last_occurrence': final,
            'number_of_iterations_last': int(it_count[-1]) if it_count else None,
            'n_number_of_iterations_lines': len(it_count), 'exit_last': exits[-1].strip() if exits else None,
            'n_restoration_lines': sum(1 for r in rows if r['restoration']),
            'timing_statistics_block_present': 'Timing Statistics' in text, 'timing_statistics_block_text': timing,
            'total_seconds_in_ipopt': float(tot.group(1)) if tot else None, 'total_seconds_lines': tot_wall,
            'evaluations': evals,
            'banner': (re.search(r'This is Ipopt version[^\n]*', text).group(0)
                       if re.search(r'This is Ipopt version[^\n]*', text) else None)}


# ======================================================================================================================
#  Parent
# ======================================================================================================================
def _preflight(home):
    """gurobipy reader starts its Env under the empty HOME (zero solves); matplotlib font cache built inside that HOME;
    IPOPT executable present (existence only; nothing launched)."""
    code = ('import json, gurobipy as gp, matplotlib.font_manager, pyomo.environ\n'
            'env = gp.Env(empty=True); env.setParam("LogToConsole", 0); env.start(); env.dispose()\n'
            'print(json.dumps({"gurobipy": ".".join(str(v) for v in gp.gurobi.version())}))\n')
    env = dict(os.environ, HOME=home)
    proc = subprocess.run([sys.executable, '-c', code], cwd=home, env=env, capture_output=True, text=True)
    ipopt = _load_dotenv_ipopt()
    out = {'ipopt_path': ipopt, 'ipopt_exists': bool(ipopt and os.path.exists(ipopt))}
    if proc.returncode != 0:
        return {'ok': False, 'stderr': proc.stderr[-2000:], **out}
    return {'ok': out['ipopt_exists'], **json.loads(proc.stdout.strip().splitlines()[-1]), **out}


def parent():
    import platform
    home = tempfile.mkdtemp(prefix='p516_ipopt_home_')
    if os.listdir(home):
        print(f'[REFUSED] temporary HOME {home} is not empty')
        return 6
    pre = _preflight(home)
    print(f'[preflight] {json.dumps(pre)} (HOME={home})')
    if not pre['ok']:
        print('[REFUSED] the gurobipy LP reader or the IPOPT executable is unavailable; nothing created, nothing solved.')
        return 6
    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    try:
        os.mkdir(OUT_DIR)                       # atomic: refuses an existing run and a concurrent copy
    except FileExistsError:
        print(f'[REFUSED] {rel(OUT_DIR)} exists; refusing to run (no overwrite, no concurrent copy).')
        return 4
    started = time.time()
    provenance = {'git_head': T1B._git('rev-parse', 'HEAD'),
                  'git_branch': T1B._git('rev-parse', '--abbrev-ref', 'HEAD'),
                  'tracked_status_before': T1B._git('status', '--porcelain', '--untracked-files=no'),
                  'platform': platform.platform(), 'parent_interpreter': sys.executable,
                  'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rehearsal': DRY, 'preflight': pre,
                  'child_HOME': home,
                  'child_HOME_note': 'empty temporary HOME in the child (restricted gurobipy licence, read only)'}
    with open(PROVENANCE_PARENT, 'w') as handle:
        json.dump(provenance, handle, indent=1)
    with open(LOG_PATH, 'w', buffering=1) as log:
        log.write(f'# p516_d0_ipopt_interior parent start {provenance["started"]} interpreter {sys.executable}\n')
        child = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child'], cwd=REPO,
                                 env=dict(os.environ, HOME=home), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = child.wait()
        log.write(f'# child exit code {code}; wall {time.time() - started:.1f} s\n')
    after = {'tracked_status_after': T1B._git('status', '--porcelain', '--untracked-files=no'),
             'child_HOME_contents_after': sorted(os.listdir(home))}
    after['tracked_status_unchanged'] = after['tracked_status_after'] == provenance['tracked_status_before']
    with open(os.path.join(OUT_DIR, 'provenance_parent_after.json'), 'w') as handle:
        json.dump(after, handle, indent=1)
    manifest = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in sorted(files):
            path = os.path.join(root, name)
            if path == MANIFEST_PATH:
                continue
            manifest[os.path.relpath(path, OUT_DIR)] = {'sha256': sha256_file(path), 'bytes': os.path.getsize(path)}
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump({'root': rel(OUT_DIR), 'child_exit_code': code, 'files': dict(sorted(manifest.items()))},
                  handle, indent=1)
    print(f'[parent] child exit code {code}; tracked status unchanged {after["tracked_status_unchanged"]}; '
          f'manifest {len(manifest)} files -> {rel(MANIFEST_PATH)}')
    return code


# ======================================================================================================================
#  THE leaf: the only function from which a solver is reached (Pyomo's IPOPT shell interface).
# ======================================================================================================================
_STATE = {}


def _leaf_solve(payload):
    depth = _STATE['in_leaf']
    depth['depth'] += 1                         # the audit hook permits process launches only inside the leaf
    try:
        solver = _STATE['solver_factory']('ipopt', executable=_STATE['ipopt_path'])
        for key, value in payload['options'].items():
            solver.options[key] = value
        t0 = time.time()
        results = solver.solve(payload['model'], tee=False, load_solutions=False, logfile=payload['log_path'])
        return {'wall_s': time.time() - t0, 'results': results}
    finally:
        depth['depth'] -= 1


# ======================================================================================================================
#  Child
# ======================================================================================================================
def child():
    import p516_d0_t02 as T02
    T02._install_audit_hook()                   # raises on any process launch outside the leaf; records write-opens
    sys.path.insert(0, REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    import p516_d0_clarabel as C
    leaf_site = (('p516_d0_ipopt_interior.py', '_leaf_solve'),)
    no_site = (('p516_d0_ipopt_interior.py', '__no_permitted_site__'),)
    pyomo_guard = SolveProfileGuard(permitted=leaf_site, label='P5.16 D0 T0.2-ipopt-interior (leaf only)').install()
    gurobi_guard = T02.GurobiLeafGuard(permitted=no_site, label='P5.16 D0 T0.2-ipopt-interior: no Gurobi optimize'
                                       ).install()
    clarabel_guard = C.ClarabelLeafGuard(permitted=no_site, label='P5.16 D0 T0.2-ipopt-interior: no Clarabel solve'
                                         ).install()
    t_start = time.time()
    global np
    import numpy
    np = numpy
    T02.np = numpy
    C.np = numpy
    import gurobipy as gp
    import gate_result_io as gio

    def write_json(name, payload, compact=False):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            if compact:
                gio.dump(payload, handle, default=gio.json_default_item, separators=(',', ':'))
            else:
                gio.dump(payload, handle, default=gio.json_default_item, indent=1)
        return path

    write_json('predictions.json', {
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'predictions': PREDICTIONS,
        'operationalisation': OPERATIONALISATION, 'declared_solves': DECLARED_SOLVES, 'ipopt_options': IPOPT_OPTIONS,
        'group': TAG, 'thresholds': {'obj_identity_rel': OBJ_IDENTITY_REL, 'containment_tol_rel': CONTAINMENT_TOL_REL,
                                     'active_rel': ACTIVE_REL, 'small_circle_r': SMALL_CIRCLE_R,
                                     'scale_floor': SCALE_FLOOR,
                                     'I2_max_iter': I2_MAX_ITER, 'I4_range': I4_RANGE,
                                     'I5_point_factor': [I5_POINT, I5_FACTOR], 'I1_min_deficiency': I1_MIN_DEFICIENCY},
        'note': 'recorded by the harness before any input is loaded (task P5.16 D0 / T0.2-ipopt-interior)'})
    print('[write] predictions.json')

    gate_rows, stops = [], []

    def gate(name, scope, achieved, threshold, note=None):
        achieved = float(achieved)
        ok = bool(achieved <= threshold)
        gate_rows.append({'gate': name, 'scope': scope, 'achieved': achieved, 'threshold': threshold, 'pass': ok,
                          'note': note})
        print(f'[gate {"pass" if ok else "FAIL"}] {name} {scope}: {achieved:.6e} <= {threshold:.3e}')
        return ok

    # ---- inputs, hash-checked; the reused source pinned ---------------------------------------------------------
    inputs, mismatch = {}, []

    def check(path, expected, label):
        full = os.path.join(REPO, path)
        actual = sha256_file(full) if os.path.exists(full) else None
        inputs[label] = {'path': path, 'expected': expected, 'actual': actual, 'match': actual == expected}
        if actual != expected:
            mismatch.append(label)
        return actual == expected

    check(os.path.join(T01C_ROOT, 'manifest_sha256.json'), T01C_MANIFEST_SHA256, 't01c/manifest')
    check(os.path.join(T01B_ROOT, 'manifest_sha256.json'), T01B_MANIFEST_SHA256, 't01b/manifest')
    if mismatch:
        write_json('interior.json', {'refused': True, 'inputs': inputs, 'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5
    t01c_files = json.load(open(os.path.join(REPO, T01C_ROOT, 'manifest_sha256.json')))['files']
    t01b_files = json.load(open(os.path.join(REPO, T01B_ROOT, 'manifest_sha256.json')))['files']
    gkey = f'{YEAR}_{DAY}'
    lp_rel = f'lp/{CELL}/{gkey}_{MODE}.lp'
    nm_rel = f'lp/{CELL}/{gkey}_{MODE}.name_map.json'
    for name in ('lp/index.json', 'objective.json', 'capacity.json', 'inventory.json', lp_rel, nm_rel):
        check(os.path.join(T01C_ROOT, name), t01c_files[name]['sha256'], f't01c/{name}')
    for name in (f'captures/{CELL}/blocks_{YEAR}_{DAY}.pkl', f'captures/{CELL}/vectors_{YEAR}_{DAY}.json',
                 f'captures/{CELL}/vectors_esso.json'):
        check(os.path.join(T01B_ROOT, name), t01b_files[name]['sha256'], f't01b/{name}')
    inv = json.load(open(os.path.join(REPO, T01C_ROOT, 'inventory.json')))
    cell_rec = next(c for c in T1B.CELLS if c['label'] == CELL)
    ev_path = os.path.join(cell_rec['eval_dir'], 'evaluation_record.json')
    check(ev_path, inv['cell_records'][CELL]['record_sha256']['evaluation_record.json'], f'{CELL}/evaluation_record.json')
    del inv
    reused = {}
    for name, expected in REUSED_T02_SOURCE_SHA256.items():
        text, digest, lineno = extract_t02_function(name)
        reused[name] = {'sha256': digest, 'expected': expected, 'match': digest == expected,
                        'p516_d0_t02_line': lineno, 'text': text}
        if digest != expected:
            mismatch.append(f'p516_d0_t02.{name} source')
    if mismatch:
        write_json('interior.json', {'refused': True, 'inputs': inputs, 'mismatch': mismatch,
                                     'reused_source': {k: {kk: vv for kk, vv in v.items() if kk != 'text'}
                                                       for k, v in reused.items()}})
        print(f'[REFUSED] input hash / reused-source mismatch: {mismatch}')
        return 5
    ev = json.load(open(os.path.join(REPO, ev_path)))
    instance = {'cell': CELL, 'year': YEAR, 'day': DAY, 'variant': VAR, 'candidate_key': ev['candidate_key'],
                'certification_cycle': ev['certification_cycle'], 'eval_dir': cell_rec['eval_dir'],
                'lp': os.path.join(T01C_ROOT, lp_rel), 'lp_sha256': t01c_files[lp_rel]['sha256'],
                'name_map_sha256': t01c_files[nm_rel]['sha256']}
    del ev
    objective_t01c = json.load(open(os.path.join(REPO, T01C_ROOT, 'objective.json')))
    capacity = json.load(open(os.path.join(REPO, T01C_ROOT, 'capacity.json')))
    cal_factor = {tuple(int(v) for v in k.strip('()').split(',')): rec['factor']
                  for k, rec in capacity['calendar_inputs']['calendar_factor'].items()}
    del capacity
    print(f'[inputs] {len(inputs)} files hash-checked against the T0.1b / T0.1c manifests: all match; reused '
          f'p516_d0_t02 sources {[(k, v["sha256"][:8]) for k, v in reused.items()]} match the pinned hashes')

    # ---- static leaf check; solver state -------------------------------------------------------------------------
    project_modules = {os.path.splitext(f)[0] for f in os.listdir(REPO) if f.endswith('.py')}
    leaf_names = set(_leaf_solve.__code__.co_names)
    leaf_project_refs = sorted(leaf_names & (project_modules - {'p516_d0_ipopt_interior'}))
    if leaf_project_refs:
        raise RuntimeError(f'the leaf names project modules {leaf_project_refs}: production reachable from it')
    from pyomo.opt import SolverFactory
    import pyomo.environ as pe
    _STATE.update({'solver_factory': SolverFactory, 'ipopt_path': _load_dotenv_ipopt(), 'in_leaf': T02._IN_LEAF})
    env = gp.Env(empty=True)
    env.setParam('LogToConsole', 0)
    env.start()
    print(f'[leaf] static check: leaf names {sorted(leaf_names)}; project modules referenced: none; gurobipy '
          f'{".".join(str(v) for v in gp.gurobi.version())} Env started (reader only; optimize blocked); IPOPT '
          f'{_STATE["ipopt_path"]}')
    ns = {'pe': pe, 'np': np, 'math': math, 'os': os, 're': re}
    for name in REUSED_T02_SOURCE_SHA256:
        exec(compile(reused[name]['text'], f'<p516_d0_t02.{name}>', 'exec'), ns)
    build_ipopt_model, parse_ipopt_log = ns['build_ipopt_model'], ns['parse_ipopt_log']

    # ==================================================================================================================
    #  Zero-solve phase
    # ==================================================================================================================
    lp_root = os.path.join(REPO, T01C_ROOT)
    t0 = time.time()
    name_map = json.load(open(os.path.join(lp_root, nm_rel)))['name_map']
    data = T02.read_lp_data(os.path.join(lp_root, lp_rel), name_map, env)
    del name_map
    t_read = time.time() - t0
    import pickle
    with open(os.path.join(REPO, T01B_ROOT, f'captures/{CELL}/blocks_{YEAR}_{DAY}.pkl'), 'rb') as handle:
        blocks = pickle.load(handle)
    vectors = json.load(open(os.path.join(REPO, T01B_ROOT, f'captures/{CELL}/vectors_{YEAR}_{DAY}.json')))
    vectors_esso = json.load(open(os.path.join(REPO, T01B_ROOT, f'captures/{CELL}/vectors_esso.json')))
    networks = {'TSO': blocks['tso'].node_balance_p._rule._fcn.keywords['network']}
    for n_ in (5, 7, 9):
        networks[f'DSO{n_}'] = blocks['dso'][n_].node_balance_p._rule._fcn.keywords['network']
    x_cert = T02.certified_point(data, vectors, vectors_esso, networks, lambda idx: cal_factor[(idx[0], idx[1])])
    del blocks, vectors, vectors_esso
    gc.collect()
    print(f'[read] {TAG}: LP read {t_read:.2f} s; size {json.dumps(data["size"])}')
    obj_cert = float(data['c'] @ x_cert) + data['obj_con']
    ref = objective_t01c['cells'][CELL]['per_group'][gkey][VAR]['group_relaxed_objective_certified']
    q_plus_rec = objective_t01c['q_cert_plus'][CELL]['per_group'][gkey]
    q_cert_plus = float(q_plus_rec['q_cert_plus'])
    gate('LP objective at the certified point == T0.1c group objective (rel)', TAG, abs(obj_cert - ref) / abs(ref),
         OBJ_IDENTITY_REL)
    vmin = min(data['lb'][k] for k, (_b, f_, _i) in enumerate(data['parsed']) if f_ == 'vmag_sqr')
    gate('Wii lower bound > 0 on every vmag_sqr column (min LB; fails if <= 0)', TAG, 0.0 if vmin > 0 else 1.0, 0.0)

    # ---- 1. redundancy inventory ---------------------------------------------------------------------------------
    lb, ub = data['lb'], data['ub']
    fixed = np.isfinite(lb) & (lb == ub)
    conic = C.build_conic(data, presolve=True)          # zero-solve: the presolve report (pinned columns, implied rows)
    pres = conic['presolve']
    pinned, _how = C.pinned_zero_columns(data)
    del conic
    gc.collect()
    rank = equality_rank_inventory(data, fixed)
    cols = column_inventory(data)
    classified, qinv = quadratic_inventory(data, pinned)
    redundancy = {
        'implied_rhs0_equality_rows_all_columns_pinned': pres['not_altered']['implied_equality_rows_already_in_the_lp'],
        'pinned_columns': {'count': pres['n_columns_pinned'], 'by_row_family': pres['pinned_by_row_family'],
                           'rule': pres['rule']},
        'equality_rank': rank, 'columns': cols}
    print(f'[inventory] implied rhs-0 rows {redundancy["implied_rhs0_equality_rows_all_columns_pinned"]["count"]} '
          f'{redundancy["implied_rhs0_equality_rows_all_columns_pinned"]["by_family"]}')
    for key in ('A_eq', 'A_eq_without_fixed_columns'):
        r_ = rank[key]
        print(f'[rank] {key}: {r_["rows"]} x {r_["columns"]}; rank {r_["numerical_rank"]}; deficiency '
              f'{r_["rank_deficiency"]} (structural {r_["structural_deficiency_rows_minus_cols"]}); min rel sigma kept '
              f'{r_["min_rel_singular_value_counted_in_rank"]:.3e}; max rel sigma dropped '
              f'{r_["max_rel_singular_value_dropped"]:.3e}; deficient components '
              f'{r_["deficient_components_by_shape"]}; dependent rows (QR choice) '
              f'{r_["dependent_rows_by_family_qr_choice"]}; {r_["wall_s"]:.1f} s')
    print(f'[columns] {json.dumps({k: v for k, v in cols.items() if k != "objective"})}')
    print(f'[quadratic] {json.dumps({k: v for k, v in qinv.items() if k != "note"})}')
    gate('rank inventory: A_eq rank + deficiency == rows (0 = consistent)', TAG,
         float(abs(rank['A_eq']['numerical_rank'] + rank['A_eq']['rank_deficiency'] - rank['A_eq']['rows'])), 0.0)

    # ---- 2. the smooth-form NLP (p516_d0_t02.build_ipopt_model) and the start ------------------------------------
    t0 = time.time()
    m = build_ipopt_model(data, x_cert)
    t_build = time.time() - t0
    n = len(data['var_names'])
    x_start = np.array([m.x[k].value for k in range(n)], dtype=float)
    clip = np.abs(x_start - x_cert)
    start = {'definition': 'T0.1c certified mapped point (p516_d0_t02.certified_point, variant C calendar factor), '
                           'clipped into the LP bounds by build_ipopt_model',
             'n_clipped': int(np.sum(clip > 0)), 'max_clip': float(clip.max()),
             'worst_clipped_column': data['var_names'][int(np.argmax(clip))] if clip.max() > 0 else None,
             'objective_at_start': float(data['c'] @ x_start) + data['obj_con'],
             'violations': violations(data, x_start)}
    _rs, _as, _ss, start['cone_slacks'] = cone_slacks(data, classified, x_start)
    print(f'[start] clipped {start["n_clipped"]} (max {start["max_clip"]:.3e}); objective '
          f'{start["objective_at_start"]:.6f}; max violation {start["violations"]["max_over_all"]:.3e} '
          f'(eq {start["violations"]["linear_eq"]["max"]:.3e}, ineq {start["violations"]["linear_ineq"]["max"]:.3e}, '
          f'quad {start["violations"]["quadratic"]["max"]:.3e}); active cones/circles at start '
          f'{start["cone_slacks"]["n_active_rel_le_1e-6"]}')
    # NL header vs LP dimensions (zero-solve; written outside the repository and removed)
    nl_dir = tempfile.mkdtemp(prefix='p516_ipopt_nl_')
    nl_path = os.path.join(nl_dir, f'{TAG}.nl')
    t0 = time.time()
    m.write(nl_path)
    with open(nl_path) as handle:
        header = [handle.readline().split() for _ in range(2)][1]
    nl = {'nl_write_s': time.time() - t0, 'nl_bytes': os.path.getsize(nl_path),
          'header_vars_cons_objs_ranges_eqns': [int(v) for v in header[:5]]}
    os.remove(nl_path)
    os.rmdir(nl_dir)
    lp_dims = {'vars': n, 'rows': len(data['rhs']) + len(data['qrows']), 'eq_rows': int(np.sum(data['sense'] == '='))}
    hdr = nl['header_vars_cons_objs_ranges_eqns']
    gate('smooth form: NL header (vars, cons, eqns) == LP dimensions (0 = equal)', TAG,
         0.0 if (hdr[0], hdr[1], hdr[4]) == (lp_dims['vars'], lp_dims['rows'], lp_dims['eq_rows']) else 1.0, 0.0)
    print(f'[nlp] build {t_build:.1f} s; NL header {hdr}; LP dims {lp_dims}; NL write {nl["nl_write_s"]:.1f} s')

    # ==================================================================================================================
    #  The ONE solve
    # ==================================================================================================================
    log_path = os.path.join(OUT_DIR, 'logs', f'{TAG}.ipopt.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    solve = {'tag': TAG, 'options': IPOPT_OPTIONS, 'log': rel(log_path), 'build_s': t_build,
             'ipopt_path': _STATE['ipopt_path']}
    n_solved = 0
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if DRY:
        solve.update({'STAND_IN': True, 'status': 'DRY', 'ObjVal': float(pe.value(m.obj))})
        x = x_start.copy()
        sol = {'X': x, 'Pi': np.zeros(len(data['rhs'])), 'QCPi': np.zeros(len(data['qrows'])), 'RC': data['c'].copy()}
    else:
        print(f'[solve] {TAG}: IPOPT, options {IPOPT_OPTIONS}')
        res = _leaf_solve({'model': m, 'options': IPOPT_OPTIONS, 'log_path': log_path})
        n_solved += 1
        results = res['results']
        solve.update({'wall_s': res['wall_s'], 'termination_condition': str(results.solver.termination_condition),
                      'solver_status': str(results.solver.status), 'solver_message': str(results.solver.message)})
        try:
            m.solutions.load_from(results)
            solve['solution_loaded'] = True
        except Exception as error:      # noqa: BLE001 -- recorded; no retry
            solve['solution_loaded'] = False
            solve['load_error'] = f'{type(error).__name__}: {error}'
        x = np.array([m.x[k].value if m.x[k].value is not None else np.nan for k in range(n)], dtype=float)
        if not solve['solution_loaded']:
            x = np.full(n, np.nan)                  # the model still holds the START values: never read them as x*
        duals_lin = np.array([m.dual.get(m.lin[i + 1], np.nan) for i in range(len(data['rhs']))])
        duals_q = np.array([m.dual.get(m.quad[j + 1], np.nan) for j in range(len(data['qc_rhs']))])
        zl = np.array([m.ipopt_zL_out.get(m.x[k], 0.0) for k in range(n)])
        zu = np.array([m.ipopt_zU_out.get(m.x[k], 0.0) for k in range(n)])
        os.makedirs(os.path.join(OUT_DIR, 'vectors'), exist_ok=True)     # raw, before any post-processing
        np.savez_compressed(os.path.join(OUT_DIR, 'vectors', f'{TAG}.raw.npz'), x=x, dual_lin=duals_lin,
                            dual_quad=duals_q, ipopt_zL_out=zl, ipopt_zU_out=zu)
        print(f'[write] raw solution vectors {TAG}.raw.npz (loaded {solve["solution_loaded"]})')
        try:
            solve['ObjVal'] = float(pe.value(m.obj)) if solve['solution_loaded'] else None
            conv = {}                               # sign convention read off stationarity (as p516_d0_t02.ipopt_solve)
            if solve['solution_loaded']:
                for sgn in (1.0, -1.0):
                    r = data['c'] - data['A'].T @ (sgn * duals_lin)
                    for j, q in enumerate(data['qrows']):
                        _v, qg = T02.quadratic_value_and_grad(q, x)
                        for i, gval in qg.items():
                            r[i] -= sgn * duals_q[j] * gval
                    conv[sgn] = float(np.nanmax(np.abs(r - (zl + zu))))
            sgn = min(conv, key=conv.get) if conv else 1.0
            solve['multiplier_sign_convention'] = {
                'chosen_sign': sgn, 'stationarity_residual_by_sign': {str(k): v for k, v in conv.items()},
                'definition': 'pi_gurobi_convention = sign * pyomo dual; bound multipliers z = ipopt_zL_out + ipopt_zU_out '
                              '(as p516_d0_t02.ipopt_solve)'}
            sol = {'X': x, 'Pi': sgn * duals_lin, 'QCPi': sgn * duals_q, 'RC': zl + zu}
        except Exception as error:      # noqa: BLE001 -- recorded (the solve is never repeated); a STOP
            import traceback
            stops.append({'stop': 'post-processing error (multipliers)', 'error': f'{type(error).__name__}: {error}',
                          'traceback': traceback.format_exc()})
            print(f'[STOP] post-processing error (multipliers): {type(error).__name__}: {error}')
            sol = {'X': x, 'Pi': None, 'QCPi': None, 'RC': None}
    solve['rss_self_max_bytes'] = {'before': rss_before, 'after': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   'children_after': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    vec_path = os.path.join(OUT_DIR, 'vectors', f'{TAG}.npz')
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    np.savez_compressed(vec_path, x_cert=x_cert, x_start=x_start, **{k: v for k, v in sol.items() if v is not None})
    solve['vectors'] = rel(vec_path)
    print(f'[write] {rel(vec_path)}')
    del m
    gc.collect()

    # ---- the log ----
    try:
        if not DRY:
            solve['log_parse'] = parse_ipopt_log(log_path)
            solve['iterations_parse'] = parse_iterations(log_path)
            lp_ = solve['log_parse']
            it_rows = solve['iterations_parse']['iterations']
            solve['summary'] = {
                'exit': lp_.get('exit'), 'iterations': int(lp_['iterations']) if lp_.get('iterations') else None,
                'restoration_iterations': solve['iterations_parse']['n_restoration_lines'],
                'initial_inf_pr': it_rows[0]['inf_pr'] if it_rows else None,
                'initial_inf_du': it_rows[0]['inf_du'] if it_rows else None,
                'initial_objective_ipopt': it_rows[0]['objective'] if it_rows else None,
                'final_lg_mu': it_rows[-1]['lg_mu'] if it_rows else None,
                'final_inf_pr': it_rows[-1]['inf_pr'] if it_rows else None,
                'final_inf_du': it_rows[-1]['inf_du'] if it_rows else None,
                'max_inf_pr_over_run': max((r['inf_pr'] for r in it_rows), default=None),
                'total_seconds_in_ipopt': solve['iterations_parse']['total_seconds_in_ipopt'],
                'exit_last_occurrence': solve['iterations_parse']['exit_last'],
                'iterations_last_occurrence': solve['iterations_parse']['number_of_iterations_last'],
                'final_summary_last_occurrence': solve['iterations_parse']['final_summary_last_occurrence']}
            n_vars_ipopt = lp_.get('n_variables')
            if n_vars_ipopt is not None:
                n_eq = int(lp_['n_equality_constraints'])
                n_in = int(lp_['n_inequality_constraints'])
                gate('IPOPT-reported dimensions: variables == LP columns - bound-fixed columns, eq + ineq == LP rows '
                     '(0 = equal)', TAG,
                     0.0 if (int(n_vars_ipopt) == n - int(fixed.sum()) and n_eq + n_in == lp_dims['rows']) else 1.0, 0.0,
                     note='IPOPT removes bound-fixed variables (fixed_variable_treatment make_parameter)')
            print(f'[solve] {TAG}: {json.dumps(solve["summary"])}; termination {solve["termination_condition"]}; wall '
                  f'{solve["wall_s"]:.1f} s')
            gate('IPOPT exit "Optimal Solution Found." (0 = yes)', TAG,
                 0.0 if (lp_.get('exit') or '').startswith('Optimal Solution Found') else 1.0, 0.0)
    except Exception as error:      # noqa: BLE001 -- recorded (the solve is never repeated); a STOP
        import traceback
        stops.append({'stop': 'post-processing error (log parse)', 'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc()})
        print(f'[STOP] post-processing error (log parse): {type(error).__name__}: {error}')

    # ==================================================================================================================
    #  Measurements at the returned point
    # ==================================================================================================================
    at = {'stand_in': DRY}
    finite_x = bool(np.all(np.isfinite(x)))
    at['x_finite'] = finite_x
    try:
        if finite_x:
            at['objective_recomputed'] = float(data['c'] @ x) + data['obj_con']
            at['violations'] = violations(data, x)
            rel_slack, abs_val, scale, at['cone_slacks'] = cone_slacks(data, classified, x)
            np.savez_compressed(os.path.join(OUT_DIR, 'vectors', f'{TAG}.cone_slacks.npz'), rel_slack=rel_slack,
                                abs_slack=abs_val, scale=scale)
            at['cone_slacks_per_row'] = [[data['qc_names'][j], classified[j][0],
                                          None if np.isnan(rel_slack[j]) else float(rel_slack[j]), float(abs_val[j])]
                                         for j in range(len(classified))]
            at['rank_and_cycles'] = T02.rank_and_cycles(data, x, networks, T02._cycle_basis)
            if not DRY:
                at['lagrangian'] = T02.lagrangian_bound(data, x, sol['Pi'], sol['QCPi'], sol['RC'], solve.get('ObjVal'))
                # multiplier magnitudes by family (Gurobi convention)
                mags = {}
                for names, vals, kind in ((data['con_names'], sol['Pi'], 'linear'), (data['qc_names'], sol['QCPi'],
                                                                                      'quadratic')):
                    for i, v in enumerate(vals):
                        if not np.isfinite(v):
                            continue
                        key = f'{kind} {fam(names[i])}'
                        mags[key] = max(mags.get(key, 0.0), abs(float(v)))
                at['max_abs_multiplier_by_family'] = dict(sorted(mags.items(), key=lambda t: -t[1]))
                at['max_abs_bound_multiplier'] = float(np.max(np.abs(sol['RC'])))
                small = {}
                for j, (p, s) in enumerate(classified):
                    if p == 'circle' and s['r'] < SMALL_CIRCLE_R:
                        key = fam(data['qc_names'][j])
                        rec = small.setdefault(key, {'count': 0, 'n_active': 0, 'min_rel_slack': math.inf,
                                                     'max_abs_qcpi': 0.0})
                        rec['count'] += 1
                        rs = float(rel_slack[j]) if not np.isnan(rel_slack[j]) else math.nan
                        rec['n_active'] += int(rs <= ACTIVE_REL)
                        rec['min_rel_slack'] = min(rec['min_rel_slack'], rs)
                        rec['max_abs_qcpi'] = max(rec['max_abs_qcpi'], abs(float(sol['QCPi'][j])))
                at['small_circles_r_lt_1e-4'] = small
            print(f'[at x*] objective {at["objective_recomputed"]:.6f}; max violation '
                  f'{at["violations"]["max_over_all"]:.3e}; active cones/circles {at["cone_slacks"]["n_active_rel_le_1e-6"]}'
                  f' of {at["cone_slacks"]["n_rows"]}; min rel slack {at["cone_slacks"]["min_rel_slack"]} '
                  f'({at["cone_slacks"]["argmin_row"]})')
            for blk, rec_ in at['rank_and_cycles'].items():
                print(f'[rank] {blk}: max rel rank residual {rec_["max_rel_rank_residual"]}; cycles '
                      f'{rec_.get("n_cycles")}; max cycle angle {rec_.get("max_cycle_angle_residual_rad")}; OLTC '
                      f'{(rec_.get("oltc") or {}).get("max_transformer_cone_rel_residual")}')
    except Exception as error:      # noqa: BLE001 -- recorded (the solve is never repeated); a STOP
        import traceback
        stops.append({'stop': 'post-processing error (measurements)', 'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc()})
        print(f'[STOP] post-processing error (measurements): {type(error).__name__}: {error}')

    # ---- comparison with Q_cert+ ----
    q_lb = solve.get('ObjVal')
    comparison = {'Q_cert_plus': q_cert_plus, 'gross_group': float(q_plus_rec['gross']),
                  'Q_cert_plus_source': 't01c/objective.json q_cert_plus.x0.per_group.2025_Spring.q_cert_plus',
                  'Q_LB': q_lb, 'stand_in': DRY,
                  'label': OPERATIONALISATION['gap_label']}
    optimal = (not DRY) and (solve.get('log_parse', {}).get('exit') or '').startswith('Optimal Solution Found')
    if q_lb is not None and not DRY:
        comparison.update({'gap_abs_eur': q_cert_plus - q_lb, 'gap_abs_keur': (q_cert_plus - q_lb) / 1e3,
                           'gap_rel': (q_cert_plus - q_lb) / q_cert_plus,
                           'containment_tol_eur': CONTAINMENT_TOL_REL * abs(q_cert_plus),
                           'containment_margin_eur': q_lb - q_cert_plus - CONTAINMENT_TOL_REL * abs(q_cert_plus),
                           'containment_held': bool(q_lb <= q_cert_plus + CONTAINMENT_TOL_REL * abs(q_cert_plus)),
                           'solve_optimal': optimal})
        if optimal and not comparison['containment_held']:
            stops.append({'stop': 'G-D0-4 containment violated at an Optimal IPOPT solve', 'comparison': comparison})
        print(f'[gap] Q_LB {q_lb:.6f}; Q_cert+ {q_cert_plus:.6f}; gap {comparison["gap_abs_keur"]:.3f} kEUR '
              f'({comparison["gap_rel"]:.3e} rel) -- {OPERATIONALISATION["gap_label"]}')

    # ---- predictions ----
    preds = {}
    d1 = rank['A_eq']['rank_deficiency']
    preds['I1'] = {'prediction': PREDICTIONS['I1'], 'achieved': {'A_eq': d1, 'A_eq_without_fixed_columns':
                                                                 rank['A_eq_without_fixed_columns']['rank_deficiency']},
                   'verdict': 'HELD' if d1 >= I1_MIN_DEFICIENCY else 'FAILED'}
    if DRY:
        for k in ('I2', 'I3', 'I4', 'I5'):
            preds[k] = {'prediction': PREDICTIONS[k], 'verdict': 'STAND-IN (rehearsal; no solve)'}
    else:
        s_ = solve.get('summary', {})
        i2 = optimal and (s_.get('iterations') or 10 ** 9) <= I2_MAX_ITER and s_.get('restoration_iterations') == 0
        preds['I2'] = {'prediction': PREDICTIONS['I2'], 'achieved': {'exit': s_.get('exit'),
                                                                     'iterations': s_.get('iterations'),
                                                                     'restoration_iterations':
                                                                         s_.get('restoration_iterations')},
                       'verdict': 'HELD' if i2 else 'FAILED'}
        if q_lb is None:
            for k in ('I3', 'I4'):
                preds[k] = {'prediction': PREDICTIONS[k], 'verdict': 'INDETERMINATE (no objective)'}
        else:
            preds['I3'] = {'prediction': PREDICTIONS['I3'], 'achieved': {k: comparison[k] for k in (
                'Q_LB', 'Q_cert_plus', 'containment_tol_eur', 'containment_margin_eur')},
                'verdict': ('HELD' if comparison['containment_held'] else 'FAILED') +
                ('' if optimal else ' (at a NON-optimal point: not a measurement)')}
            g_rel = comparison['gap_rel']
            preds['I4'] = {'prediction': PREDICTIONS['I4'], 'achieved': {'gap_rel': g_rel,
                                                                         'gap_abs_keur': comparison['gap_abs_keur']},
                           'verdict': (('HELD' if I4_RANGE[0] <= g_rel <= I4_RANGE[1] else 'FAILED') if optimal
                                       else 'INDETERMINATE (not Optimal)')}
        tso = at.get('rank_and_cycles', {}).get('TSO', {}).get('max_rel_rank_residual') if finite_x else None
        if tso is None or not optimal:
            preds['I5'] = {'prediction': PREDICTIONS['I5'], 'achieved': tso, 'verdict': 'INDETERMINATE (not Optimal)'}
        else:
            preds['I5'] = {'prediction': PREDICTIONS['I5'], 'achieved': tso,
                           'band': [I5_POINT / I5_FACTOR, I5_POINT * I5_FACTOR],
                           'verdict': 'HELD' if I5_POINT / I5_FACTOR <= tso <= I5_POINT * I5_FACTOR else 'FAILED'}

    # ---- guards, containment, provenance ----
    expected = 0 if DRY else n_solved
    if not DRY and expected != DECLARED_SOLVES:
        stops.append({'stop': f'solved {expected} != declared {DECLARED_SOLVES}'})
    py_fail = pyomo_guard.verify(expected)
    gu_fail = gurobi_guard.verify(0)
    cl_fail = clarabel_guard.verify(0)
    for guard in (pyomo_guard, gurobi_guard, clarabel_guard):
        guard.uninstall()
    env.dispose()
    gate('G-D0-6 leaf guard exact (failures: Pyomo/IPOPT 1 + 1, Gurobi 0, Clarabel 0)', 'process',
         float(len(py_fail) + len(gu_fail) + len(cl_fail)), 0.0)
    if py_fail or gu_fail or cl_fail:
        stops.append({'stop': 'leaf-guard count disagrees', 'pyomo': py_fail, 'gurobi': gu_fail, 'clarabel': cl_fail})
    out_abs, repo_abs = os.path.abspath(OUT_DIR), os.path.abspath(REPO)
    writes_repo = sorted(rel(p) for p in T02._AUDIT['write_opens'] if p.startswith(repo_abs + os.sep) and not
                         p.startswith(out_abs + os.sep) and '__pycache__' not in p.split(os.sep))
    gate('containment: write-opens in the repository outside OUT_DIR (count)', 'process', float(len(writes_repo)), 0.0)
    repo_modules = {}
    for name_, mod in sorted(sys.modules.items()):
        f_ = getattr(mod, '__file__', None)
        if f_ and f_.startswith(repo_abs + os.sep) and '__pycache__' not in f_:
            repo_modules[name_] = {'file': rel(f_), 'sha256': sha256_file(f_)}
    import scipy
    import pyomo
    provenance = json.load(open(PROVENANCE_PARENT))
    provenance.update({'interpreter': sys.executable, 'python': sys.version, 'numpy': np.__version__,
                       'scipy': scipy.__version__, 'pyomo': pyomo.version.version,
                       'gurobipy_reader': '.'.join(str(v) for v in gp.gurobi.version()),
                       'HOME_in_child': os.environ.get('HOME'), 'project_modules_loaded': repo_modules,
                       'harness_sha256': sha256_file(os.path.abspath(__file__)),
                       'spec_sha256': sha256_file(os.path.join(REPO, 'P5_16_D0_SPEC_v2.md')),
                       'reused_source': {k: {kk: vv for kk, vv in v.items() if kk != 'text'} for k, v in reused.items()},
                       'inputs': inputs, 'rehearsal_stand_in': DRY, 'wall_s': time.time() - t_start})
    guard_out = {'leaf': leaf_site, 'declared_solves': DECLARED_SOLVES, 'expected_at_verify': expected,
                 'counted_events': {'pyomo': dict(pyomo_guard.counts), 'gurobi': dict(gurobi_guard.counts),
                                    'clarabel': dict(clarabel_guard.counts)},
                 'verify_failures': {'pyomo': py_fail, 'gurobi': gu_fail, 'clarabel': cl_fail},
                 'process_launch_events': T02._AUDIT['process_launch_events'],
                 'leaf_static_check': {'names': sorted(leaf_names), 'project_modules_referenced': leaf_project_refs}}
    gates_out = {'rows': gate_rows, 'all_pass': all(r['pass'] for r in gate_rows),
                 'failed': [r for r in gate_rows if not r['pass']], 'stops': stops, 'stop_triggered': bool(stops)}
    payload = {'stage': 'P5.16 D0 / T0.2-ipopt-interior (one declared IPOPT solve)', 'rehearsal_stand_in': DRY,
               'instance': instance,
               'objective_convention': ('gross_operational_cost (settlement-EXCLUDED, salvage NOT subtracted) per '
                                        'group; Q_cert+ = gross with every detector slack at max(slack, 0) (spec '
                                        'section 12 ruling 5); gap = Q_cert+ - Q_LB; variant C (calendar-only)'),
               'lp_size': data['size'], 'lp_dimensions': lp_dims, 'nl': nl, 'redundancy': redundancy,
               'quadratic_inventory': qinv, 'start': start, 'solve': solve, 'at_solution': at,
               'comparison': comparison, 'predictions': preds, 'gates': gates_out, 'guard': guard_out,
               'containment': {'write_opens_in_repo_outside_out_dir': writes_repo},
               'gate_obj_identity': {'obj_lp_at_certified': obj_cert, 'obj_t01c_recorded': ref,
                                     'rel': abs(obj_cert - ref) / abs(ref), 'min_vmag_sqr_lb': vmin},
               'thresholds': {'obj_identity_rel': OBJ_IDENTITY_REL, 'containment_tol_rel': CONTAINMENT_TOL_REL,
                              'active_rel': ACTIVE_REL, 'small_circle_r': SMALL_CIRCLE_R, 'scale_floor': SCALE_FLOOR},
               'provenance': provenance}
    write_json('interior.json', payload)
    print('[write] interior.json')
    print(f'[summary] gates {len(gate_rows)} rows, all pass {gates_out["all_pass"]}; failed '
          f'{sorted({(r["gate"], r["scope"]) for r in gates_out["failed"]})}; stops {stops}; guards '
          f'{guard_out["counted_events"]}; repo writes outside OUT_DIR {len(writes_repo)}; wall '
          f'{time.time() - t_start:.1f} s')
    for key, value in preds.items():
        print(f'[prediction] {key}: {value.get("verdict")} {json.dumps(value.get("achieved"), default=str)}')
    if stops:
        return 3
    return 0 if gates_out['all_pass'] else 2


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        sys.exit(child())
    sys.exit(parent())
