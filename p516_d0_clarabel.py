"""P5.16 D0 / T0.2-clarabel -- the open-source conic path (Addendum 87/88 R5 fallback: "Clarabel per block").

A reader + conic builder + Clarabel solver + dual extraction for the T0.1c group LPs (`P516D0/t01c/lp/`, manifest
sha256 d2955128...), so that T0.2's measurements (a), (d), (e), (g) and the containment / P7 gates can run without a
Gurobi licence. Governing spec: `P5_16_D0_SPEC_v2.md` section 4 (G-D0-4 containment, G-D0-5 solver and duals, G-D0-6
guard), section 5, sections 12.5 (Q_cert+), 13 and 14 (ruling 5: Clarabel fallback).

Reader. The exported LP is read with gurobipy (`p516_d0_t02.read_lp_data`; read only, never optimized). On this machine
the user licence `~/gurobi.lic` is bound to another host ID, so the child runs with HOME pointed at an EMPTY temporary
directory: gurobipy then starts its bundled size-restricted licence, which reads models of any size (it only limits
optimize, and optimize is blocked by a guard here).

Conic form (Clarabel: min q'x  s.t.  A x + s = b,  s in K;  P = 0):
  ZeroCone      : every linear '=' row (a, rhs); every fixed column (LB == UB) as e_k x = LB
  Nonnegative   : '<' rows (a, rhs); '>' rows (-a, -rhs); finite UB (e_k, UB); finite LB (-e_k, -LB)
  SecondOrder   : one cone per quadratic row of the LP, three patterns only (anything else is a STOP, row reported):
    rotated  sum_i d_i x_i^2 - k u v <= 0 (d_i, k > 0; LB(u), LB(v) >= 0):  s = (u + v, u - v, 2 sqrt(d_i / k) x_i)
             -- the branch cones WijR^2 + WijI^2 <= Wii Wjj and the transformer cones C^2 + D^2 <= U Wjj
    lorentz  sum_i d_i x_i^2 - e t^2 <= 0 (d_i, e > 0; LB(t) >= 0):  s = (sqrt(e) t, sqrt(d_i) x_i)
             -- the ESSO converter rows es_pnet^2 + es_qnet^2 <= es_s_rated^2
    circle   sum_i d_i x_i^2 <= r (d_i > 0, r >= 0):  s = (sqrt(r), sqrt(d_i) x_i)
             -- generator capability, branch thermal limits, network-copy converter circles (S fixed, possibly 0)
  In every pattern the quadratic row value q(x) - rhs equals alpha * (|s_bar|^2 - s_0^2) exactly (alpha = k / 4 for
  rotated, 1 otherwise) -- the identity gated at the certified point before any solve.

Duals. Clarabel's z (z in K* = K) satisfies q + A' z = 0 at an exact optimum; the reported dual residual is
||q + A' z||_inf in the unscaled data (the conic form of the task's ||A'y + z - c||_inf, bounds being rows here).
Gurobi-convention multipliers (dQ/drhs) for the `p516_d0_t02` post-processing: Pi = -z on '=' and '<' rows, +z on '>'
rows; RC = z_LB - z_UB - z_fix; QCPi = -z_0 / (2 alpha s_0) on a cone with s_0 > 0 (z = mu (s_0, -s_bar) at
complementarity), 0 where s_0 == 0 (counted; the conic residual stays the authoritative check there).

Declared solves: exactly 2 (task T0.2-clarabel smoke; variant C, 2025 Spring, x = 0 and the unit). One leaf,
`_leaf_solve`, is the only function from which `clarabel.DefaultSolver.solve` may be reached: `ClarabelLeafGuard`
replaces `solve` ON THE TYPE and counts two events per solve (leaf entry + the solve call); any other call site raises.
Gurobi optimize and Pyomo solves are blocked (permitted count 0, verified exactly). The audit hook of
`p516_d0_t02` raises on any process launch in the child and records every write-open.

Ablation mode (`--ablation`; task P5.16 D0 / T0.2-clarabel ablation, Planner ruling: option (b) + (a)): exactly 4
declared solves on the same two groups, every one with the PRESOLVE on, max_iter 200 and 5,000 per group; every other
Clarabel setting at its default (verbose on). Output root `t02_clarabel_ablation/` (refuses if it exists), record
`ablation.json`. The smoke mode (no flag) is unchanged: presolve off, 2 solves, `t02_clarabel_smoke/`, `smoke.json`.
  PRESOLVE (equivalence-preserving; `build_conic(data, presolve=True)`): a column is PINNED to 0 when (a) it is fixed at
  LB = UB = 0, or (b) an '=' row with rhs exactly 0 has it as its only unpinned column with a nonzero coefficient
  (a x_k + sum_{pinned} a_j x_j = 0 => x_k = 0 exactly); (b) is iterated to a fixed point. This covers the
  zero-investment chain: es_s_rated_per_unit[k, u] pinned by the single-column row rated_s_capacity_unit, then the
  head es_s_rated[k] pinned by rated_s_capacity (es_s_rated - sum_u es_s_rated_per_unit = 0). Every Lorentz cone
  |sqrt(d) x| <= sqrt(e) t whose head t is pinned is the set {sqrt(d_i) x_i = 0} (the chain rows stay in the model), so
  it is REPLACED by zero-cone rows s_i = sqrt(d_i) x_i = 0 (same set; the empty-interior cone is removed) -- one row per
  component x_i that is NOT itself already pinned by the LP's rows (a row for a pinned component would be a linear
  combination of existing equality rows: same set, but a new rank deficiency of the zero cone; on these LPs every es_pnet
  and some es_qnet are already pinned). Nothing else is altered: no column substituted, no other cone touched.
  Small-radius circles (r < 1e-4) are reported, not altered.
  Gate (ii) identity on a replaced cone: |s_bar|^2 + sum_{pinned i} d_i x_i^2 - e t^2 (s_bar the zero-row slacks; x_i
  and t read from x) equals the LP row value q(x) - rhs to 8 eps; the containment stop applies only at a CONVERGED point
  (status Solved or AlmostSolved).

Stops (exit 3): a quadratic row that fits none of the three patterns (before any solve); G-D0-4 containment violated
(ablation: at a converged point only); the leaf-guard count disagreeing. Refusals: root exists (4); input hash mismatch (5); Clarabel or the gurobipy reader
unavailable (6, nothing created).

Usage (one process, attached, alone):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_clarabel.py
Rehearsal (zero solves; scratch under /tmp, never evidence; the Clarabel solver object is CONSTRUCTED -- data
validation, equilibration and KKT setup -- but `solve` is never called; every "solution" is the certified point with
zero multipliers, labelled STAND-IN):
    P516_D0_CLARABEL_REHEARSAL_DIR=/tmp/... /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_clarabel.py
Ablation (4 declared solves; rehearsal as above with the same flag):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_clarabel.py --ablation

Exit codes: 0 all gates pass; 2 a gate failed (no stop); 3 a stop fired; 4 refused (root exists); 5 refused (input hash
mismatch); 6 refused (solver unavailable).
"""

import gc
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time

import p516_d0_t01b as T1B

REPO = os.path.dirname(os.path.abspath(__file__))
ABLATION = '--ablation' in sys.argv[1:]          # parsed in the parent and passed on to the child
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P516D0',
                       't02_clarabel_ablation' if ABLATION else 't02_clarabel_smoke')
_REHEARSAL = os.environ.get('P516_D0_CLARABEL_REHEARSAL_DIR')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516_D0_CLARABEL_REHEARSAL_DIR must be under /tmp, got {_REHEARSAL!r}')
    OUT_DIR = os.path.abspath(_REHEARSAL)
DRY = bool(_REHEARSAL)

LOG_PATH = os.path.join(OUT_DIR, 'run.log')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
PROVENANCE_PARENT = os.path.join(OUT_DIR, 'provenance_parent.json')

T01C_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01c')
T01C_MANIFEST_SHA256 = 'd29551285683ede902b71188ba6973f33e15d53018f8a77da6e5fd833514afb5'
T01B_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01b')
T01B_MANIFEST_SHA256 = '4e5bbd45234598094593a8daca807a576590eee52f492f5281b63cf621876562'

SMOKE = (('x0', 2025, 'Spring'), ('unit_n7', 2025, 'Spring'))
MODE, VAR = 'calendar', 'C'
PRESOLVE = ABLATION
# the solve plan: (solve label, group tag, max_iter override or None = the default); the smoke labels are the group tags
_TAGS = [f'{c}_{y}_{d}_{VAR}' for c, y, d in SMOKE]
SOLVE_PLAN = ([(f'{_TAGS[0]}.presolve.it200', _TAGS[0], 200), (f'{_TAGS[0]}.presolve.it5000', _TAGS[0], 5000),
               (f'{_TAGS[1]}.presolve.it200', _TAGS[1], 200), (f'{_TAGS[1]}.presolve.it5000', _TAGS[1], 5000)]
              if ABLATION else [(t, t, None) for t in _TAGS])
DECLARED_SOLVES = len(SOLVE_PLAN)
CONVERGED_STATUSES = ('Solved', 'AlmostSolved')   # ablation: the containment STOP applies at these statuses only
SMALL_CIRCLE_R = 1e-4                             # circles reported (not altered) by the ablation
B5_STATIONARITY_REL = 1e-6
# Clarabel settings: every value at the Clarabel 0.11.1 default except `verbose` (the per-solve log). The full
# settings object is recorded per solve.
CLARABEL_OVERRIDES = {'verbose': True}
MIN_COLS_HOME_NOTE = 'HOME is an empty temporary directory in the child (restricted gurobipy licence, read only)'

# Gate thresholds (stated before the run)
IDENTITY_OBJ_REL = 1e-12          # task gate (i)
IPOPT_BOUND_RELAX = 1e-8          # IPOPT 3.14 default bound_relax_factor (production passes none; network.py l.585ff)
T01C_MAPPING_DELTA = 2.1e-9       # T0.1c report: max |Delta| of a retained row at the rank-1 mapped point 2.08e-9
CAPACITY_SHIFT_C = 5e-4           # spec v2 section 14 ruling 1: SoC-row violations of the certified point on C,
#                                   2.3e-4 .. 4.9e-4 pu (the certified SoC belongs to the carried capacity)
CAPACITY_SHIFT_FAMILIES = ('sess_soc_def', 'sess_soc_final', 'sess_soc_limit_lower', 'sess_soc_limit_upper')

PREDICTIONS = {
    'K1': 'Clarabel converges on both groups (status Solved) to a primal-dual gap <= 1e-8 relative',
    'K2': 'containment holds on both groups: Q_LB <= Q_cert+ + solver gap (G-D0-4)',
    'K3': 'the x = 0 group\'s Q_LB lies within 1e-2 relative of Q_cert+ (H2 point estimate); the achieved gap is '
          'stated in k EUR and relative',
}
OPERATIONALISATION = {
    'K1': 'status == Solved and |obj_val - obj_val_dual| / max(1, min(|obj_val|, |obj_val_dual|)) <= 1e-8 '
          '(Clarabel\'s own gap_rel definition), both groups',
    'K2': 'Q_LB - Q_cert+(group) - |obj_val - obj_val_dual| <= 0, both groups; a violation is a STOP',
    'K3': '(Q_cert+ - Q_LB) / Q_cert+ <= 1e-2 on the x = 0 group (and >= 0, else K2 decides)',
    'gate_i': '|q.x_cert + const - (c.x_cert + ObjCon)| / |c.x_cert + ObjCon| <= 1e-12 (LP data vs the Clarabel '
              'model) and |q.x_cert + const - T0.1c group_relaxed_objective_certified| / |T0.1c| <= 1e-12',
    'gate_ii_identity': 'per row, |violation in the Clarabel model - violation in the LP form| <= 4 eps sum|terms| '
                        '(linear and bound rows) and |alpha (|s_bar|^2 - s_0^2) - (q(x) - rhs)| <= 8 eps sum|terms| '
                        '(cones), at the certified mapped point',
    'gate_ii_classes': 'every violated row is in a stated class: coupling rows d0_cpl_* (residual = the recorded '
                       'inter-copy mismatch, T0.1c gate; reported); variant-C capacity shift (section 14.1; SoC rows '
                       'of the unit cell) <= 5e-4; equality rows <= 2.1e-9 (T0.1c mapping delta); inequality, bound '
                       'and quadratic rows <= 1e-8 max(1, |rhs|) + 2.1e-9 + 4 eps sum|terms| (IPOPT '
                       'bound_relax_factor + the mapping delta)',
    'gate_iii': 'K2',
    'gate_iv': 'P7: Q_LB(unit) - Q_LB(x0) - gap(unit) - gap(x0) <= 0',
    'gate_v': 'dual residual ||q + A\'z||_inf and duality gap reported per solve; dual-cone violation reported',
    'gate_vi': 'wall time (setup, solve) and peak RSS per solve reported',
    'projection': '108 x the mean (setup + solve) wall of the two smoke solves, plus 72 group reads + builds at the '
                  'mean measured read + build time',
}

if ABLATION:
    PREDICTIONS = {
        'B1': 'with the presolve, both groups converge (status Solved) within 200 iterations',
        'B2': 'if B1 fails, 5,000 iterations suffice (status Solved) on the group(s) that failed at 200',
        'B3': 'the converged x = 0 Q_LB lies within 1e-2 relative of Q_cert+ and containment holds',
        'B4': 'P7: Q_LB(unit) <= Q_LB(x0) holds between converged values',
        'B5': 'the Gurobi-convention stationarity residual <= 1e-6 relative at a converged point',
    }
    OPERATIONALISATION = {
        'converged': 'status == Solved (Clarabel 0.11.1, full tolerances); AlmostSolved is reported, counts as '
                     'NOT converged for B1-B5, but triggers the containment STOP (conservative)',
        'B1': 'solves x0.presolve.it200 and unit_n7.presolve.it200 both status Solved',
        'B2': 'evaluated only if B1 FAILED: every group whose it200 solve is not Solved has its it5000 solve Solved; '
              'NOT APPLICABLE when B1 held (the it5000 solves are still run and reported)',
        'B3': 'on the converged x0 it5000 solve (else the converged x0 it200 solve; else INDETERMINATE): '
              'Q_LB - Q_cert+ - solver gap <= 0 AND (Q_cert+ - Q_LB) / Q_cert+ <= 1e-2',
        'B4': 'on the it5000 pair if both Solved (else the it200 pair if both Solved; else INDETERMINATE): '
              'Q_LB(unit) - Q_LB(x0) - gap(unit) - gap(x0) <= 0; every both-converged pair is reported',
        'B5': 'at every Solved point: ||c - A\'Pi - RC - sum_j QCPi_j grad q_j(x)||_inf / ||c||_inf <= 1e-6 with the '
              'mapping of `gurobi_convention_duals` (strict: the multipliers of the presolve rows have no image in '
              'the Gurobi convention -- the replaced quadratic row has a zero gradient at its apex -- and are NOT '
              'mapped). HELD iff every Solved point passes; INDETERMINATE if none is Solved. Reported beside it: the '
              'same residual with the presolve-row multipliers folded into RC as implicit fixings (labelled '
              '"folded"), and the residual restricted to columns no presolve row touches',
        'gate_i': 'as in the smoke: |q.x_cert + const - (c.x_cert + ObjCon)| / |c.x_cert + ObjCon| <= 1e-12 and vs '
                  'the T0.1c record <= 1e-12, on the PRESOLVED build',
        'gate_ii_identity': 'as in the smoke on the presolved build; for each replaced cone |alpha (|s_bar|^2 - e t^2) '
                            '- (q(x) - rhs)| <= 8 eps (|s_bar|^2 + e t^2 + sum|terms|) (s_bar the zero-row slacks)',
        'gate_ii_classes': 'unchanged (computed on the LP form)',
        'gate_iii': 'G-D0-4 containment per solve: Q_LB - Q_cert+ - solver gap <= 0; a violation at a converged point '
                    '(Solved / AlmostSolved) is a STOP (no further solve); at an unconverged point it is recorded, '
                    'not a stop',
        'per_solve_record': 'status, iterations, r_prim / r_dual and Clarabel info (res_primal, res_dual, gap_abs, '
                            'gap_rel), objective, wall, the max LP-row violation of the returned point (linear / '
                            'quadratic / bound, worst row), containment vs Q_cert+ (T0.1c objective.json), the dual '
                            'residual ||q + A\'z||_inf, the duality gap (Clarabel and recomputed), the Gurobi-'
                            'convention stationarity residual (strict and folded), the small-circle activity',
        'projection': '108 x the mean (setup + solve) wall of the converged solves of the smallest max_iter that '
                      'converged, plus 72 group reads + builds; stated per iteration as well',
    }

sha256_file = T1B.sha256_file
np = None            # bound in the child after the guards are armed


def rel(path):
    return os.path.relpath(path, REPO)


class ConeMappingError(RuntimeError):
    def __init__(self, row, detail):
        super().__init__(f'quadratic row {row!r} cannot be mapped to a second-order cone: {detail}')
        self.row = row
        self.detail = detail


# ======================================================================================================================
#  Conic build (pure numpy / scipy; no solver)
# ======================================================================================================================
def classify_quadratic_row(q, lb, rhs, sense):
    """Return (pattern, spec) for one LP quadratic row, or raise ConeMappingError with the reason."""
    if sense != '<':
        raise ValueError(f'sense {sense!r}')
    if q['lin'] or q['lin_const'] != 0.0:
        raise ValueError('linear part present')
    diag, off = {}, {}
    for i, j, c in q['quad']:
        if i == j:
            diag[i] = diag.get(i, 0.0) + c
        else:
            key = (min(i, j), max(i, j))
            off[key] = off.get(key, 0.0) + c
    diag = {k: v for k, v in diag.items() if v != 0.0}
    off = {k: v for k, v in off.items() if v != 0.0}
    if off:
        if len(off) != 1:
            raise ValueError(f'{len(off)} off-diagonal terms')
        (u, v), coef = next(iter(off.items()))
        k = -coef
        if k <= 0.0:
            raise ValueError(f'bilinear coefficient {coef} >= 0')
        if u in diag or v in diag:
            raise ValueError('bilinear factor also squared')
        if any(d <= 0.0 for d in diag.values()) or not diag:
            raise ValueError('a squared term has a non-positive coefficient')
        if rhs != 0.0:
            raise ValueError(f'rotated cone with rhs {rhs}')
        if not (lb[u] >= 0.0 and lb[v] >= 0.0):
            raise ValueError(f'bilinear factors not sign-constrained (LB {lb[u]}, {lb[v]})')
        return 'rotated', {'u': u, 'v': v, 'k': k, 'x': sorted(diag.items()), 'alpha': k / 4.0}
    neg = [(i, d) for i, d in diag.items() if d < 0.0]
    pos = sorted((i, d) for i, d in diag.items() if d > 0.0)
    if len(neg) == 1:
        t, e = neg[0][0], -neg[0][1]
        if rhs != 0.0:
            raise ValueError(f'Lorentz cone with rhs {rhs}')
        if not lb[t] >= 0.0:
            raise ValueError(f'cone head not sign-constrained (LB {lb[t]})')
        if not pos:
            raise ValueError('Lorentz cone without squared terms')
        return 'lorentz', {'t': t, 'e': e, 'x': pos, 'alpha': 1.0}
    if neg:
        raise ValueError(f'{len(neg)} negative squared terms')
    if rhs < 0.0:
        raise ValueError(f'circle with rhs {rhs} < 0')
    return 'circle', {'r': rhs, 'x': pos, 'alpha': 1.0}


def pinned_zero_columns(data):
    """Columns pinned to 0 by the LP's own rows (ablation presolve): (a) fixed at LB = UB = 0; (b) the only unpinned
    column with a nonzero coefficient of an '=' row whose rhs is exactly 0, iterated to a fixed point. Exact: the row
    reads a_k x_k + sum_{pinned} a_j x_j = 0, so x_k = 0. Returns (mask, provenance {col: (how, row index, round)})."""
    n = len(data['var_names'])
    lb, ub = data['lb'], data['ub']
    Ar = data['A']
    pinned = np.isfinite(lb) & (lb == ub) & (lb == 0.0)
    how = {int(k): ('fixed at 0', None, 0) for k in np.nonzero(pinned)[0]}
    eq_rows = [int(i) for i in np.nonzero((data['sense'] == '=') & (data['rhs'] == 0.0))[0]]
    row_cols = {}
    col_rows = {}
    for i in eq_rows:
        lo, hi = Ar.indptr[i], Ar.indptr[i + 1]
        cols_ = [int(c_) for c_, v_ in zip(Ar.indices[lo:hi], Ar.data[lo:hi]) if v_ != 0.0]
        row_cols[i] = cols_
        for c_ in cols_:
            col_rows.setdefault(c_, []).append(i)
    work, rnd = list(eq_rows), 0
    while work:
        rnd += 1
        touched = []
        for i in work:
            unp = [c_ for c_ in row_cols[i] if not pinned[c_]]
            if len(unp) == 1:
                k = unp[0]
                pinned[k] = True
                how[k] = ('eq row, rhs 0', i, rnd)
                touched.extend(col_rows.get(k, []))
        work = sorted(set(touched))
    assert n == len(pinned)
    return pinned, how


def build_conic(data, presolve=False):
    """The Clarabel data (P, q, A, b, cone list) for one exported LP read by `p516_d0_t02.read_lp_data`, with a row map
    back to the LP. Raises ConeMappingError naming the first quadratic row that fits no pattern. With `presolve`, every
    Lorentz cone whose head is pinned to 0 (`pinned_zero_columns`) is replaced by zero-cone rows sqrt(d_i) x_i = 0
    (rmap kind 'Q', tag 'presolved_x'; listed in `presolved_cones`, not in `cones_meta`); nothing else changes."""
    import scipy.sparse as sp
    n = len(data['var_names'])
    lb, ub = data['lb'], data['ub']
    Ar = data['A']
    rows, cols, vals, b = [], [], [], []
    rmap = []                                  # (kind, index, tag) per conic row

    def add_row(entries, rhs, kind, idx, tag):
        r = len(b)
        for c_, v_ in entries:
            rows.append(r)
            cols.append(c_)
            vals.append(v_)
        b.append(rhs)
        rmap.append((kind, idx, tag))

    def lin_entries(i, sign):
        lo, hi = Ar.indptr[i], Ar.indptr[i + 1]
        return [(int(c_), sign * float(v_)) for c_, v_ in zip(Ar.indices[lo:hi], Ar.data[lo:hi])]

    sense = data['sense']
    fixed = np.isfinite(lb) & (lb == ub)
    # ---- classification of every quadratic row first (pure; the first misfit is a STOP) ----
    classified = []
    for j, q in enumerate(data['qrows']):
        try:
            classified.append(classify_quadratic_row(q, lb, float(data['qc_rhs'][j]), str(data['qc_sense'][j])))
        except ValueError as error:
            raise ConeMappingError(data['qc_names'][j], str(error)) from None
    replaced = set()
    pinned, how = (pinned_zero_columns(data) if presolve else (None, None))
    if presolve:
        replaced = {j for j, (pattern, spec) in enumerate(classified) if pattern == 'lorentz' and pinned[spec['t']]}
    # ---- zero cone ----
    for i in np.nonzero(sense == '=')[0]:
        add_row(lin_entries(i, 1.0), float(data['rhs'][i]), 'L', int(i), 'eq')
    for k in np.nonzero(fixed)[0]:
        add_row([(int(k), 1.0)], float(lb[k]), 'B', int(k), 'fix')
    presolved_cones = []
    for j in sorted(replaced):
        spec = classified[j][1]
        start = len(b)
        for x_, d in spec['x']:
            if not pinned[x_]:                  # an already-pinned component needs no row (it would be dependent)
                add_row([(x_, -math.sqrt(d))], 0.0, 'Q', j, 'presolved_x')     # s_i = sqrt(d_i) x_i = 0
        presolved_cones.append({'qrow': j, 'pattern': 'lorentz', 'start': start, 'dim': len(b) - start, 'alpha': 1.0,
                                'head_col': spec['t'], 'e': spec['e'],
                                'x_cols': [x_ for x_, _d in spec['x'] if not pinned[x_]],
                                'x_pinned': [(x_, d) for x_, d in spec['x'] if pinned[x_]]})
    n_presolved_rows = sum(c['dim'] for c in presolved_cones)
    n_zero = len(b)
    # ---- nonnegative cone ----
    for i in np.nonzero(sense == '<')[0]:
        add_row(lin_entries(i, 1.0), float(data['rhs'][i]), 'L', int(i), 'le')
    for i in np.nonzero(sense == '>')[0]:
        add_row(lin_entries(i, -1.0), -float(data['rhs'][i]), 'L', int(i), 'ge')
    for k in np.nonzero(np.isfinite(ub) & ~fixed)[0]:
        add_row([(int(k), 1.0)], float(ub[k]), 'B', int(k), 'ub')
    for k in np.nonzero(np.isfinite(lb) & ~fixed)[0]:
        add_row([(int(k), -1.0)], -float(lb[k]), 'B', int(k), 'lb')
    n_nonneg = len(b) - n_zero
    unknown = set(sense.tolist()) - {'=', '<', '>'}
    if unknown:
        raise RuntimeError(f'unknown linear row senses {unknown}')
    # ---- second-order cones ----
    cones_meta = []
    pattern_count = {'rotated': 0, 'lorentz': 0, 'circle': 0}
    for j, (pattern, spec) in enumerate(classified):
        if j in replaced:
            continue
        start = len(b)
        if pattern == 'rotated':
            u, v, k = spec['u'], spec['v'], spec['k']
            add_row([(u, -1.0), (v, -1.0)], 0.0, 'Q', j, 'head')               # s0 = u + v
            add_row([(u, -1.0), (v, 1.0)], 0.0, 'Q', j, 'u-v')                  # s1 = u - v
            for x_, d in spec['x']:
                add_row([(x_, -2.0 * math.sqrt(d / k))], 0.0, 'Q', j, 'x')
        elif pattern == 'lorentz':
            add_row([(spec['t'], -math.sqrt(spec['e']))], 0.0, 'Q', j, 'head')
            for x_, d in spec['x']:
                add_row([(x_, -math.sqrt(d))], 0.0, 'Q', j, 'x')
        else:
            add_row([], math.sqrt(spec['r']), 'Q', j, 'head')
            for x_, d in spec['x']:
                add_row([(x_, -math.sqrt(d))], 0.0, 'Q', j, 'x')
        pattern_count[pattern] += 1
        cones_meta.append({'qrow': j, 'pattern': pattern, 'start': start, 'dim': len(b) - start,
                           'alpha': spec['alpha']})
    m = len(b)
    A = sp.csc_matrix((np.asarray(vals, dtype=float), (np.asarray(rows), np.asarray(cols))), shape=(m, n))
    A.sum_duplicates()
    P = sp.csc_matrix((n, n))
    inventory = {
        'n_columns': n, 'n_conic_rows': m, 'zero_cone_dim': n_zero, 'nonneg_cone_dim': n_nonneg,
        'n_soc': len(cones_meta), 'soc_dims': sorted({c['dim'] for c in cones_meta}), 'soc_patterns': pattern_count,
        'n_lp_eq_rows': int(np.sum(sense == '=')), 'n_lp_le_rows': int(np.sum(sense == '<')),
        'n_lp_ge_rows': int(np.sum(sense == '>')), 'n_lp_quadratic_rows': len(data['qrows']),
        'n_fixed_columns': int(fixed.sum()), 'n_finite_ub_not_fixed': int(np.sum(np.isfinite(ub) & ~fixed)),
        'n_finite_lb_not_fixed': int(np.sum(np.isfinite(lb) & ~fixed)),
    }
    inventory['dims_consistent'] = bool(
        n_zero == inventory['n_lp_eq_rows'] + inventory['n_fixed_columns'] + n_presolved_rows and
        n_nonneg == inventory['n_lp_le_rows'] + inventory['n_lp_ge_rows'] + inventory['n_finite_ub_not_fixed'] +
        inventory['n_finite_lb_not_fixed'] and len(cones_meta) + len(presolved_cones) == len(data['qrows']) and
        m == n_zero + n_nonneg + sum(c['dim'] for c in cones_meta))
    out = {'P': P, 'q': data['c'].astype(float).copy(), 'const': float(data['obj_con']), 'A': A,
           'b': np.asarray(b, dtype=float), 'n_zero': n_zero, 'n_nonneg': n_nonneg, 'cones_meta': cones_meta,
           'rmap': rmap, 'inventory': inventory}
    if presolve:
        out['presolved_cones'] = presolved_cones
        out['presolve'] = presolve_report(data, classified, pinned, how, presolved_cones, cones_meta)
        inventory['presolve'] = {k: out['presolve'][k] for k in (
            'n_columns_pinned', 'n_lorentz_cones', 'n_lorentz_cones_replaced', 'n_zero_rows_added',
            'n_components_already_pinned_no_row', 'n_replaced_cones_with_no_row',
            'n_distinct_pinned_heads', 'pinned_heads', 'n_lorentz_cones_kept', 'kept_heads')}
    return out


def presolve_report(data, classified, pinned, how, presolved_cones, cones_meta):
    """What the presolve did and why (provenance of every pinned head through its chain), plus the empty-interior
    candidates it did NOT alter: rotated cones with a pinned factor, circles with r == 0, and the small circles."""
    from p516_d0_t02 import parse_pyomo_name

    def fam(name):
        blk, f_, _idx = parse_pyomo_name(name)
        return f'{(blk or "top").split("_")[0]}.{f_}'

    def chain(k, depth=0):
        kind, row, rnd = how[k]
        rec = {'column': data['var_names'][k], 'how': kind, 'round': rnd}
        if row is not None:
            rec['row'] = data['con_names'][row]
            if depth < 2:
                lo, hi = data['A'].indptr[row], data['A'].indptr[row + 1]
                rec['inputs'] = [chain(int(c_), depth + 1) for c_, v_ in zip(data['A'].indices[lo:hi],
                                                                              data['A'].data[lo:hi])
                                 if v_ != 0.0 and int(c_) != k]
        return rec

    heads = sorted({c['head_col'] for c in presolved_cones})
    lorentz = [j for j, (p, _s) in enumerate(classified) if p == 'lorentz']
    kept_heads = sorted({classified[c['qrow']][1]['t'] for c in cones_meta if c['pattern'] == 'lorentz'})
    pin_row_fams = {}
    for k, (kind, row, _rnd) in how.items():
        key = 'fixed at 0' if row is None else fam(data['con_names'][row])
        pin_row_fams[key] = pin_row_fams.get(key, 0) + 1
    rot_pinned = [data['qc_names'][j] for j, (p, s) in enumerate(classified)
                  if p == 'rotated' and (pinned[s['u']] or pinned[s['v']])]
    circ_zero = [data['qc_names'][j] for j, (p, s) in enumerate(classified) if p == 'circle' and s['r'] == 0.0]
    used_rows = {row for (_k, row, _r) in how.values() if row is not None}
    implied = {}                                # '=' rows (rhs 0) of the LP whose columns are all pinned by OTHER rows
    Ar = data['A']
    for i in np.nonzero((data['sense'] == '=') & (data['rhs'] == 0.0))[0]:
        lo, hi = Ar.indptr[i], Ar.indptr[i + 1]
        cols_ = [c_ for c_, v_ in zip(Ar.indices[lo:hi], Ar.data[lo:hi]) if v_ != 0.0]
        if cols_ and int(i) not in used_rows and all(pinned[c_] for c_ in cols_):
            f_ = fam(data['con_names'][i])
            implied[f_] = implied.get(f_, 0) + 1
    small = [{'qrow': j, 'row': data['qc_names'][j], 'family': fam(data['qc_names'][j]), 'r': s['r'],
              'radius': math.sqrt(s['r'])}
             for j, (p, s) in enumerate(classified) if p == 'circle' and s['r'] < SMALL_CIRCLE_R]
    small_by_fam = {}
    for c in small:
        rec = small_by_fam.setdefault(c['family'], {'count': 0, 'r_min': math.inf, 'r_max': 0.0})
        rec['count'] += 1
        rec['r_min'] = min(rec['r_min'], c['r'])
        rec['r_max'] = max(rec['r_max'], c['r'])
    return {
        'rule': ('pinned: fixed at LB = UB = 0, or the only unpinned nonzero column of an = row with rhs exactly 0 '
                 '(iterated); a Lorentz cone with a pinned head is replaced by zero rows sqrt(d_i) x_i = 0'),
        'n_columns_pinned': int(pinned.sum()), 'pinned_by_row_family': dict(sorted(pin_row_fams.items())),
        'n_lorentz_cones': len(lorentz), 'n_lorentz_cones_replaced': len(presolved_cones),
        'n_lorentz_cones_kept': len(lorentz) - len(presolved_cones),
        'replaced_by_family': {f_: sum(1 for c in presolved_cones if fam(data['qc_names'][c['qrow']]) == f_)
                               for f_ in sorted({fam(data['qc_names'][c['qrow']]) for c in presolved_cones})},
        'n_zero_rows_added': sum(c['dim'] for c in presolved_cones),
        'n_components_already_pinned_no_row': sum(len(c['x_pinned']) for c in presolved_cones),
        'n_replaced_cones_with_no_row': sum(1 for c in presolved_cones if c['dim'] == 0),
        'n_distinct_pinned_heads': len(heads), 'pinned_heads': [data['var_names'][k] for k in heads],
        'kept_heads': [data['var_names'][k] for k in kept_heads],
        'head_chains': [chain(k) for k in heads],
        'not_altered': {'implied_equality_rows_already_in_the_lp': {
                            'count': sum(implied.values()), 'by_family': implied,
                            'note': ('rhs-0 rows whose every column is pinned by other rows: linearly dependent on '
                                     'them (a rank deficiency of the zero cone present in the LP itself)')},
                        'rotated_cones_with_a_pinned_factor': {'count': len(rot_pinned), 'rows': rot_pinned[:20]},
                        'circles_with_r_eq_0': {'count': len(circ_zero), 'rows': circ_zero[:20]},
                        'small_circles_r_lt_1e-4': {'count': len(small), 'by_family': small_by_fam,
                                                    'rows': small}},
    }


def clarabel_cones(conic):
    import clarabel
    cones = [clarabel.ZeroConeT(conic['n_zero']), clarabel.NonnegativeConeT(conic['n_nonneg'])]
    cones += [clarabel.SecondOrderConeT(c['dim']) for c in conic['cones_meta']]
    return cones


def objective(conic, x):
    return float(conic['q'] @ x) + conic['const']


# ======================================================================================================================
#  Point checks (no solver)
# ======================================================================================================================
def lp_row_violations(data, x):
    """Violation of every LP row (linear, quadratic) and bound at x, with the sum of absolute terms per row."""
    A = data['A']
    ax = A @ x
    r = ax - data['rhs']
    s = data['sense']
    lin = np.where(s == '=', np.abs(r), np.where(s == '<', np.maximum(r, 0.0), np.maximum(-r, 0.0)))
    lin_terms = abs(A) @ np.abs(x) + np.abs(data['rhs'])
    quad_val, quad_terms = np.zeros(len(data['qrows'])), np.zeros(len(data['qrows']))
    for j, q in enumerate(data['qrows']):
        v = q['lin_const'] - data['qc_rhs'][j]
        t = abs(q['lin_const']) + abs(data['qc_rhs'][j])
        for i_, k_, c_ in q['quad']:
            v += c_ * x[i_] * x[k_]
            t += abs(c_ * x[i_] * x[k_])
        for i_, c_ in q['lin']:
            v += c_ * x[i_]
            t += abs(c_ * x[i_])
        quad_val[j], quad_terms[j] = v, t
    lbv = np.where(np.isfinite(data['lb']), np.maximum(data['lb'] - x, 0.0), 0.0)
    ubv = np.where(np.isfinite(data['ub']), np.maximum(x - data['ub'], 0.0), 0.0)
    return {'lin': lin, 'lin_terms': lin_terms, 'lin_resid': r, 'quad_val': quad_val, 'quad_terms': quad_terms,
            'lb': lbv, 'ub': ubv}


def conic_slack(conic, x):
    return conic['b'] - conic['A'] @ x


def gate_identity_and_classes(conic, data, x, cell):
    """Task gate (ii) at the certified mapped point: (a) the Clarabel model's violation of every row equals the LP
    form's to floating-point resolution; (b) every violated row lies in a stated tolerance class."""
    eps = np.finfo(float).eps
    lpv = lp_row_violations(data, x)
    s = conic_slack(conic, x)
    absA = abs(conic['A'])
    terms = absA @ np.abs(x) + np.abs(conic['b'])
    out = {'identity': {}, 'classes': {}}
    nz = conic['n_zero']
    nn = conic['n_nonneg']
    worst = {'linear_or_bound': (0.0, None), 'cone': (0.0, None)}
    n_over = {'linear_or_bound': 0, 'cone': 0}
    presolved = conic.get('presolved_cones')
    n_presolved_rows = sum(c['dim'] for c in presolved) if presolved else 0
    for r in range(nz + nn):
        kind, idx, tag = conic['rmap'][r]
        if kind == 'Q':                         # presolve rows: checked below, per replaced cone
            continue
        v_conic = abs(s[r]) if r < nz else max(-s[r], 0.0)
        if kind == 'L':
            v_lp = lpv['lin'][idx]
            name = data['con_names'][idx]
        else:
            v_lp = {'fix': max(lpv['lb'][idx], lpv['ub'][idx]), 'ub': lpv['ub'][idx], 'lb': lpv['lb'][idx]}[tag]
            name = f'{data["var_names"][idx]} ({tag})'
        diff = abs(v_conic - v_lp)
        tol = 4.0 * eps * terms[r]
        ratio = diff / tol if tol > 0 else (0.0 if diff == 0.0 else math.inf)
        if diff > tol:
            n_over['linear_or_bound'] += 1
        if ratio > worst['linear_or_bound'][0]:
            worst['linear_or_bound'] = (ratio, name)
    cone_rel = []
    for c in conic['cones_meta']:
        sc = s[c['start']:c['start'] + c['dim']]
        val_conic = c['alpha'] * (float(sc[1:] @ sc[1:]) - sc[0] * sc[0])
        val_lp = lpv['quad_val'][c['qrow']]
        scale = c['alpha'] * (float(sc[1:] @ sc[1:]) + sc[0] * sc[0]) + lpv['quad_terms'][c['qrow']]
        diff = abs(val_conic - val_lp)
        tol = 8.0 * eps * scale
        ratio = diff / tol if tol > 0 else (0.0 if diff == 0.0 else math.inf)
        if diff > tol:
            n_over['cone'] += 1
        if ratio > worst['cone'][0]:
            worst['cone'] = (ratio, data['qc_names'][c['qrow']])
        soc_viol = max(float(np.linalg.norm(sc[1:])) - sc[0], 0.0)
        cone_rel.append(soc_viol)
    presolved_check = None
    if presolved:
        worst['presolved_cone'] = (0.0, None)
        n_over['presolved_cone'] = 0
        max_zero_row, max_head, max_lp_quad = 0.0, 0.0, 0.0
        for c in presolved:
            sc = s[c['start']:c['start'] + c['dim']]
            t_val = float(x[c['head_col']])
            s0sq = c['e'] * t_val * t_val
            pin_sq = sum(d_ * float(x[x_]) ** 2 for x_, d_ in c['x_pinned'])
            val_conic = c['alpha'] * (float(sc @ sc) + pin_sq - s0sq)
            val_lp = lpv['quad_val'][c['qrow']]
            scale = c['alpha'] * (float(sc @ sc) + pin_sq + s0sq) + lpv['quad_terms'][c['qrow']]
            diff = abs(val_conic - val_lp)
            tol = 8.0 * eps * scale
            ratio = diff / tol if tol > 0 else (0.0 if diff == 0.0 else math.inf)
            if diff > tol:
                n_over['presolved_cone'] += 1
            if ratio > worst['presolved_cone'][0]:
                worst['presolved_cone'] = (ratio, data['qc_names'][c['qrow']])
            max_zero_row = max(max_zero_row, float(np.max(np.abs(sc), initial=0.0)))
            max_head = max(max_head, abs(t_val))
            max_lp_quad = max(max_lp_quad, float(max(val_lp, 0.0)))
        presolved_check = {
            'n_replaced_cones': len(presolved), 'n_zero_rows': n_presolved_rows,
            'max_abs_zero_row_slack_at_point': max_zero_row, 'max_abs_pinned_head_at_point': max_head,
            'max_lp_quadratic_violation_of_replaced_rows_at_point': max_lp_quad,
            'note': ('identity |s_bar|^2 + sum_pinned d x_i^2 - e t^2 == q(x) - rhs per replaced cone; at the point '
                     'the zero rows are violated by |sqrt(d) x_i| where the LP row is violated by sum d x_i^2 - e t^2 '
                     '(same set, different violation measure)')}
    out['identity'] = {'n_linear_and_bound_rows': nz + nn - n_presolved_rows, 'n_cones': len(conic['cones_meta']),
                       'n_over_tolerance': n_over, 'worst_ratio_to_tolerance': {k: v[0] for k, v in worst.items()},
                       'worst_row': {k: v[1] for k, v in worst.items()}, 'max_soc_violation_conic':
                           max(cone_rel) if cone_rel else 0.0}
    if presolved_check is not None:
        out['identity']['presolved'] = presolved_check
    # ---- (b) classes over the LP-form violations ----
    classes = {}

    def put(cls, fam, viol, thr, name):
        rec = classes.setdefault(cls, {'n_rows_violated': 0, 'max_violation': 0.0, 'max_ratio_to_threshold': 0.0,
                                       'n_over_threshold': 0, 'worst_row': None, 'families': {}})
        rec['n_rows_violated'] += 1
        ratio = viol / thr if thr is not None and thr > 0 else None
        if viol > rec['max_violation']:
            rec['max_violation'] = viol
        if ratio is not None:
            if ratio > rec['max_ratio_to_threshold']:
                rec['max_ratio_to_threshold'] = ratio
                rec['worst_row'] = name
            if ratio > 1.0:
                rec['n_over_threshold'] += 1
        f = rec['families'].setdefault(fam, [0, 0.0])
        f[0] += 1
        f[1] = max(f[1], viol)

    from p516_d0_t02 import parse_pyomo_name
    for i in np.nonzero(lpv['lin'] > 0.0)[0]:
        name = data['con_names'][i]
        blk, fam, _idx = parse_pyomo_name(name)
        famk = f'{(blk or "top").split("_")[0]}.{fam}'
        viol = float(lpv['lin'][i])
        fp = 4.0 * eps * float(lpv['lin_terms'][i])
        if fam.startswith('d0_cpl_'):
            put('coupling (T0.1c: residual = recorded inter-copy mismatch)', famk, viol, None, name)
        elif cell == 'unit_n7' and fam in CAPACITY_SHIFT_FAMILIES:
            put('variant-C capacity shift (section 14.1)', famk, viol, CAPACITY_SHIFT_C, name)
        elif data['sense'][i] == '=':
            put('equality, mapping delta', famk, viol, T01C_MAPPING_DELTA + fp, name)
        else:
            put('inequality, IPOPT bound relaxation + mapping delta', famk, viol,
                IPOPT_BOUND_RELAX * max(1.0, abs(float(data['rhs'][i]))) + T01C_MAPPING_DELTA + fp, name)
    for side in ('lb', 'ub'):
        for k in np.nonzero(lpv[side] > 0.0)[0]:
            name = data['var_names'][k]
            blk, fam, _idx = parse_pyomo_name(name)
            bound = float(data[side][k])
            put('bound, IPOPT bound relaxation + mapping delta', f'{(blk or "top").split("_")[0]}.{fam}.{side}',
                float(lpv[side][k]), IPOPT_BOUND_RELAX * max(1.0, abs(bound)) + T01C_MAPPING_DELTA +
                4.0 * eps * (abs(bound) + abs(x[k])), name)
    for j in np.nonzero(lpv['quad_val'] > 0.0)[0]:
        name = data['qc_names'][j]
        blk, fam, _idx = parse_pyomo_name(name)
        put('quadratic, IPOPT bound relaxation + mapping delta', f'{(blk or "top").split("_")[0]}.{fam}',
            float(lpv['quad_val'][j]), IPOPT_BOUND_RELAX * max(1.0, abs(float(data['qc_rhs'][j]))) +
            T01C_MAPPING_DELTA + 8.0 * eps * float(lpv['quad_terms'][j]), name)
    out['classes'] = classes
    out['n_over_class_threshold'] = sum(v['n_over_threshold'] for v in classes.values())
    return out


# ======================================================================================================================
#  Dual-side measurements on a conic primal-dual point
# ======================================================================================================================
def conic_dual_checks(conic, x, s, z, obj_val, obj_val_dual):
    """Task gate (v): ||q + A'z||_inf (unscaled), dual-cone membership of z, primal residual, complementarity, the
    duality gap (Clarabel's and recomputed: primal q.x + const vs dual -b.z + const)."""
    A, q, b = conic['A'], conic['q'], conic['b']
    rd = q + A.T @ z
    rp = A @ x + s - b
    nz, nn = conic['n_zero'], conic['n_nonneg']
    z_nn = z[nz:nz + nn]
    s_nn = s[nz:nz + nn]
    soc_z, soc_s = 0.0, 0.0
    for c in conic['cones_meta']:
        zc = z[c['start']:c['start'] + c['dim']]
        sc = s[c['start']:c['start'] + c['dim']]
        soc_z = max(soc_z, float(np.linalg.norm(zc[1:])) - zc[0])
        soc_s = max(soc_s, float(np.linalg.norm(sc[1:])) - sc[0])
    primal = float(q @ x) + conic['const']
    dual = -float(b @ z) + conic['const']
    gap_rel_cl = abs(obj_val - obj_val_dual) / max(1.0, min(abs(obj_val), abs(obj_val_dual)))
    return {
        'dual_residual_inf': float(np.max(np.abs(rd))), 'q_inf': float(np.max(np.abs(q))),
        'dual_residual_rel_to_q_inf': float(np.max(np.abs(rd)) / np.max(np.abs(q))),
        'dual_residual_argmax_column': int(np.argmax(np.abs(rd))),
        'primal_residual_inf': float(np.max(np.abs(rp))),
        'zero_cone_slack_inf': float(np.max(np.abs(s[:nz]))) if nz else 0.0,
        'nonneg_min_z': float(z_nn.min()) if nn else 0.0, 'nonneg_min_s': float(s_nn.min()) if nn else 0.0,
        'soc_max_z_violation': soc_z, 'soc_max_s_violation': soc_s,
        'complementarity_s_dot_z': float(s @ z), 'complementarity_max_abs': float(np.max(np.abs(s * z))),
        'objective_primal_clarabel': obj_val + conic['const'], 'objective_dual_clarabel': obj_val_dual + conic['const'],
        'gap_abs_clarabel': abs(obj_val - obj_val_dual), 'gap_rel_clarabel_definition': gap_rel_cl,
        'objective_primal_recomputed': primal, 'objective_dual_recomputed_minus_bz': dual,
        'gap_abs_recomputed': abs(primal - dual),
        'definition': ('rd = q + A\'z (Clarabel: P x + q + A\'z = 0 at optimality, P = 0); rp = A x + s - b; z in K* = '
                       'K: nonneg z >= 0, SOC z0 >= |z_bar|; dual objective -b\'z (+ const); gap_rel as Clarabel: '
                       '|p - d| / max(1, min(|p|, |d|))'),
    }


def gurobi_convention_duals(conic, data, x, s, z):
    """Pi, QCPi, RC in Gurobi's convention (dQ/drhs; RC = c - A'Pi - sum QCPi grad q) for `p516_d0_t02`'s
    post-processing; plus the stationarity residual of the mapped multipliers (equals the conic residual wherever every
    cone with a nonzero multiplier has s_0 > 0)."""
    m_lin, n = len(data['rhs']), len(data['var_names'])
    Pi, RC, QCPi = np.zeros(m_lin), np.zeros(n), np.zeros(len(data['qrows']))
    presolved = conic.get('presolved_cones')
    rc_fold = np.zeros(n) if presolved else None
    for r in range(conic['n_zero'] + conic['n_nonneg']):
        kind, idx, tag = conic['rmap'][r]
        if kind == 'Q':
            # presolve row s_i = sqrt(d_i) x_i = 0 (A entry -sqrt(d_i)): its multiplier contributes +sqrt(d_i) z to c,
            # which the Gurobi convention cannot carry (the replaced row's gradient is 0 at its apex): unmapped in
            # the strict residual; folded into RC as an implicit fixing in the "folded" reading (below)
            continue
        if kind == 'L':
            Pi[idx] = z[r] if tag == 'ge' else -z[r]
        elif tag == 'lb':
            RC[idx] += z[r]
        else:                                   # 'ub', 'fix'
            RC[idx] -= z[r]
    n_degenerate, z_norm_lost = 0, 0.0
    for c in conic['cones_meta']:
        z0 = z[c['start']]
        s0 = s[c['start']]
        if s0 > 0.0:
            QCPi[c['qrow']] = -z0 / (2.0 * c['alpha'] * s0)
        else:
            n_degenerate += 1
            z_norm_lost = max(z_norm_lost, float(np.linalg.norm(z[c['start']:c['start'] + c['dim']])))
    from p516_d0_t02 import quadratic_value_and_grad
    r = data['c'] - data['A'].T @ Pi - RC
    for j, q in enumerate(data['qrows']):
        if QCPi[j] != 0.0:
            _v, g = quadratic_value_and_grad(q, x)
            for i_, gv in g.items():
                r[i_] -= QCPi[j] * gv
    check = {'mapped_stationarity_residual_inf': float(np.max(np.abs(r))),
             'n_cones_with_s0_le_0': n_degenerate, 'max_norm_z_on_those_cones': z_norm_lost,
             'convention': ('Pi = -z on = and < rows, +z on > rows; RC = z_LB - z_UB - z_fix; QCPi = '
                            '-z0 / (2 alpha s0) for s0 > 0, else 0')}
    if presolved:
        touched = np.zeros(n, dtype=bool)
        z_max = 0.0
        for c in presolved:
            zc = z[c['start']:c['start'] + c['dim']]
            z_max = max(z_max, float(np.max(np.abs(zc), initial=0.0)))
            for i_, col in enumerate(c['x_cols']):
                touched[col] = True
                rc_fold[col] += -float(conic['A'][c['start'] + i_, col]) * float(zc[i_])   # +sqrt(d_i) z_i
        r_fold = r - rc_fold
        c_inf = float(np.max(np.abs(data['c'])))
        check.update({
            'c_inf': c_inf,
            'mapped_stationarity_residual_rel_to_c_inf': check['mapped_stationarity_residual_inf'] / c_inf,
            'presolve_rows_max_abs_z': z_max, 'n_columns_touched_by_presolve_rows': int(touched.sum()),
            'folded_stationarity_residual_inf': float(np.max(np.abs(r_fold))),
            'folded_stationarity_residual_rel_to_c_inf': float(np.max(np.abs(r_fold))) / c_inf,
            'untouched_columns_stationarity_residual_inf': float(np.max(np.abs(r[~touched]), initial=0.0)),
            'untouched_columns_stationarity_residual_rel_to_c_inf':
                float(np.max(np.abs(r[~touched]), initial=0.0)) / c_inf,
            'argmax_strict': data['var_names'][int(np.argmax(np.abs(r)))],
            'argmax_folded': data['var_names'][int(np.argmax(np.abs(r_fold)))],
            'presolve_convention': ('strict: the presolve-row multipliers are not mapped (the replaced quadratic row '
                                    'has a zero gradient at its apex, so no QCPi carries them); folded: RC[x_i] += '
                                    'sqrt(d_i) z_i (an implicit fixing x_i = 0); untouched: the strict residual on '
                                    'the columns no presolve row touches')})
    return {'Pi': Pi, 'QCPi': QCPi, 'RC': RC, 'check': check}


def returned_point_violation(data, x):
    """Ablation: the max LP-row violation of a returned point, by row class, with the worst row and the top families
    (the LP form itself, no tolerance class: the returned point is the solver's answer to this LP)."""
    from p516_d0_t02 import parse_pyomo_name
    lpv = lp_row_violations(data, x)

    def fam(name):
        blk, f_, _idx = parse_pyomo_name(name)
        return f'{(blk or "top").split("_")[0]}.{f_}'

    out = {}
    eq = data['sense'] == '='
    for label, mask in (('linear_eq', eq), ('linear_ineq', ~eq)):
        v = np.where(mask, lpv['lin'], 0.0)
        k = int(np.argmax(v))
        fams = {}
        for i in np.nonzero(v > 0.0)[0]:
            f_ = fam(data['con_names'][i])
            fams[f_] = max(fams.get(f_, 0.0), float(v[i]))
        out[label] = {'max': float(v[k]), 'worst_row': data['con_names'][k] if v[k] > 0 else None,
                      'n_rows_over_1e-8': int(np.sum(v > 1e-8)),
                      'top_families': dict(sorted(fams.items(), key=lambda t: -t[1])[:8])}
    qv = np.maximum(lpv['quad_val'], 0.0)
    k = int(np.argmax(qv)) if len(qv) else 0
    out['quadratic'] = {'max': float(qv[k]) if len(qv) else 0.0,
                        'worst_row': data['qc_names'][k] if len(qv) and qv[k] > 0 else None,
                        'n_rows_over_1e-8': int(np.sum(qv > 1e-8))}
    bv = np.maximum(lpv['lb'], lpv['ub'])
    k = int(np.argmax(bv))
    out['bound'] = {'max': float(bv[k]), 'worst_column': data['var_names'][k] if bv[k] > 0 else None,
                    'n_columns_over_1e-8': int(np.sum(bv > 1e-8))}
    out['max_over_all'] = max(out['linear_eq']['max'], out['linear_ineq']['max'], out['quadratic']['max'],
                              out['bound']['max'])
    return out


def small_circle_activity(conic, data, s, z):
    """Ablation: the circles with r < 1e-4 (reported, not altered) at a returned point: slack s0 - |s_bar| (>= 0 inside),
    the share of the radius used, and the multiplier norm, by family."""
    from p516_d0_t02 import parse_pyomo_name
    fams = {}
    for c in conic['cones_meta']:
        if c['pattern'] != 'circle':
            continue
        r0 = float(conic['b'][c['start']])          # head row: s0 = sqrt(r), no column entries
        if r0 * r0 >= SMALL_CIRCLE_R:
            continue
        sc = s[c['start']:c['start'] + c['dim']]
        zc = z[c['start']:c['start'] + c['dim']]
        nb = float(np.linalg.norm(sc[1:]))
        blk, f_, _idx = parse_pyomo_name(data['qc_names'][c['qrow']])
        key = f'{(blk or "top").split("_")[0]}.{f_}'
        rec = fams.setdefault(key, {'count': 0, 'radius_min': math.inf, 'radius_max': 0.0, 'max_used_share': 0.0,
                                    'min_slack': math.inf, 'n_active_used_share_ge_0.999': 0, 'max_norm_z': 0.0,
                                    'max_z0': 0.0})
        rec['count'] += 1
        rec['radius_min'] = min(rec['radius_min'], r0)
        rec['radius_max'] = max(rec['radius_max'], r0)
        share = nb / sc[0] if sc[0] > 0 else math.inf
        rec['max_used_share'] = max(rec['max_used_share'], share)
        rec['min_slack'] = min(rec['min_slack'], float(sc[0] - nb))
        rec['n_active_used_share_ge_0.999'] += int(share >= 0.999)
        rec['max_norm_z'] = max(rec['max_norm_z'], float(np.linalg.norm(zc)))
        rec['max_z0'] = max(rec['max_z0'], float(zc[0]))
    return fams


# ======================================================================================================================
#  Guard + THE leaf
# ======================================================================================================================
class ClarabelSolveBlocked(RuntimeError):
    pass


class ClarabelLeafGuard:
    """`clarabel.DefaultSolver.solve` replaced ON THE TYPE by a wrapper that allows the call only when the declared
    leaf is on the live call stack (counting it) and raises otherwise. Two events per solve: the leaf entry
    (`note_entry`) and the solve call itself."""

    def __init__(self, permitted, label):
        self.permitted = tuple(permitted)
        self.label = label
        self.counts = {'permitted_entry': 0, 'permitted_solve': 0, 'blocked_solve': 0}
        self._cls = None
        self._original = None

    def _site(self):
        frame = sys._getframe(2)
        while frame is not None:
            for suffix, function in self.permitted:
                if frame.f_code.co_name == function and frame.f_code.co_filename.endswith(suffix):
                    return True
            frame = frame.f_back
        return False

    def install(self):
        import clarabel
        self._cls = clarabel.DefaultSolver
        self._original = clarabel.DefaultSolver.solve
        guard = self

        def guarded_solve(solver_self, *args, **kwargs):
            if not guard._site():
                guard.counts['blocked_solve'] += 1
                raise ClarabelSolveBlocked(f'clarabel.DefaultSolver.solve from an undeclared call site; '
                                           f'{guard.label} permits only {guard.permitted}')
            guard.counts['permitted_solve'] += 1
            return guard._original(solver_self, *args, **kwargs)
        clarabel.DefaultSolver.solve = guarded_solve
        if clarabel.DefaultSolver.solve is not guarded_solve:
            raise RuntimeError('ClarabelLeafGuard: the type-level patch did not take')
        return self

    def note_entry(self):
        if not self._site():
            raise ClarabelSolveBlocked('leaf entry noted outside the leaf')
        self.counts['permitted_entry'] += 1

    def uninstall(self):
        if self._cls is not None and self._original is not None:
            self._cls.solve = self._original

    def verify(self, expected):
        failures = []
        if self.counts['permitted_entry'] != expected:
            failures.append(f'Clarabel leaf entries {self.counts["permitted_entry"]} != declared {expected}')
        if self.counts['permitted_solve'] != expected:
            failures.append(f'Clarabel solve calls {self.counts["permitted_solve"]} != declared {expected}')
        if self.counts['blocked_solve']:
            failures.append('a blocked Clarabel call site was reached')
        return failures


_STATE = {}


def _leaf_solve(payload):
    """The only function from which a solver is reached: constructs the Clarabel solver (setup) and solves once."""
    clarabel = _STATE['clarabel']
    _STATE['clarabel_guard'].note_entry()
    settings = clarabel.DefaultSettings()
    for key, value in payload['settings'].items():
        setattr(settings, key, value)
    t0 = time.time()
    solver = clarabel.DefaultSolver(payload['P'], payload['q'], payload['A'], payload['b'], payload['cones'],
                                    settings)
    t_setup = time.time() - t0
    solver.print_to_buffer()
    t1 = time.time()
    solution = solver.solve()
    t_solve = time.time() - t1
    log_text = solver.get_print_buffer()
    info = solver.get_info()
    out = {'setup_s': t_setup, 'solve_wall_s': t_solve, 'log_text': log_text,
           'status': str(solution.status), 'obj_val': float(solution.obj_val),
           'obj_val_dual': float(solution.obj_val_dual), 'iterations': int(solution.iterations),
           'solve_time_reported': float(solution.solve_time), 'r_prim': float(solution.r_prim),
           'r_dual': float(solution.r_dual), 'x': np.asarray(solution.x, dtype=float),
           'z': np.asarray(solution.z, dtype=float), 's': np.asarray(solution.s, dtype=float),
           'settings_in_force': {k: getattr(settings, k) for k in dir(settings)
                                 if not k.startswith('_') and not callable(getattr(settings, k))}}
    out['info'] = {k: (getattr(info, k) if isinstance(getattr(info, k), (int, float, str, bool))
                       else str(getattr(info, k)))
                   for k in dir(info) if not k.startswith('_') and not callable(getattr(info, k))}
    return out


# ======================================================================================================================
#  Parent
# ======================================================================================================================
def _preflight(home):
    """Clarabel importable and the gurobipy reader starting its Env under the empty HOME (zero solves). Also builds
    matplotlib's font cache inside that HOME (`fc-list` is launched here, in the parent's preflight, because the child's
    audit hook refuses every process launch and production's network.py imports pyplot on unpickling)."""
    code = ('import json, clarabel, gurobipy as gp, matplotlib.font_manager\n'
            'env = gp.Env(empty=True); env.setParam("LogToConsole", 0); env.start(); env.dispose()\n'
            'print(json.dumps({"clarabel": clarabel.__version__, '
            '"gurobipy": ".".join(str(v) for v in gp.gurobi.version())}))\n')
    env = dict(os.environ, HOME=home)
    proc = subprocess.run([sys.executable, '-c', code], cwd=home, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return {'ok': False, 'stderr': proc.stderr[-2000:]}
    return {'ok': True, **json.loads(proc.stdout.strip().splitlines()[-1])}


def parent():
    import platform
    home = tempfile.mkdtemp(prefix='p516_clarabel_home_')
    if os.listdir(home):
        print(f'[REFUSED] temporary HOME {home} is not empty')
        return 6
    pre = _preflight(home)
    print(f'[preflight] {json.dumps(pre)} (HOME={home}, empty)')
    if not pre['ok']:
        print('[REFUSED] Clarabel or the gurobipy LP reader is unavailable; nothing created, nothing solved.')
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
                  'child_HOME': home, 'child_HOME_note': MIN_COLS_HOME_NOTE}
    with open(PROVENANCE_PARENT, 'w') as handle:
        json.dump(provenance, handle, indent=1)
    with open(LOG_PATH, 'w', buffering=1) as log:
        log.write(f'# p516_d0_clarabel parent start {provenance["started"]} interpreter {sys.executable}\n')
        child = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child'] +
                                 (['--ablation'] if ABLATION else []), cwd=REPO,
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
#  Child
# ======================================================================================================================
def child():
    import p516_d0_t02 as T02
    T02._install_audit_hook()                   # raises on any process launch (T02's leaf is never entered here)
    sys.path.insert(0, REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    leaf_site = (('p516_d0_clarabel.py', '_leaf_solve'),)
    no_site = (('p516_d0_clarabel.py', '__no_permitted_site__'),)
    pyomo_guard = SolveProfileGuard(permitted=no_site, label='P5.16 D0 T0.2-clarabel: no Pyomo solve').install()
    gurobi_guard = T02.GurobiLeafGuard(permitted=no_site, label='P5.16 D0 T0.2-clarabel: no Gurobi optimize'
                                       ).install()
    clarabel_guard = ClarabelLeafGuard(permitted=leaf_site, label='P5.16 D0 T0.2-clarabel '
                                       f'{"ablation" if ABLATION else "smoke"} (leaf only)').install()
    record_name = 'ablation.json' if ABLATION else 'smoke.json'
    t_start = time.time()
    global np
    import numpy
    np = numpy
    T02.np = numpy                              # T02's readers and measurements use its module-level np
    import clarabel
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

    default_settings = clarabel.DefaultSettings()
    default_settings = {k: getattr(default_settings, k) for k in dir(default_settings)
                        if not k.startswith('_') and not callable(getattr(default_settings, k))}
    write_json('predictions.json', {
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'predictions': PREDICTIONS,
        'operationalisation': OPERATIONALISATION, 'declared_solves': DECLARED_SOLVES,
        'smoke_groups': [f'{c}_{y}_{d}_{VAR}' for c, y, d in SMOKE], 'clarabel_version': clarabel.__version__,
        'clarabel_overrides': CLARABEL_OVERRIDES, 'clarabel_defaults': default_settings,
        'note': 'recorded by the harness before any input is loaded (task P5.16 D0 / T0.2-clarabel)',
        **({'solve_plan': [{'label': l_, 'group': t_, 'max_iter': m_, 'presolve': PRESOLVE}
                           for l_, t_, m_ in SOLVE_PLAN],
            'converged_statuses_for_the_containment_stop': CONVERGED_STATUSES,
            'task': 'P5.16 D0 / T0.2-clarabel ablation (Planner ruling: option (b) + (a))'} if ABLATION else {})})
    print('[write] predictions.json')

    # ---- inputs, hash-checked -----------------------------------------------------------------------------------
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
        write_json(record_name, {'refused': True, 'inputs': inputs, 'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5
    t01c_files = json.load(open(os.path.join(REPO, T01C_ROOT, 'manifest_sha256.json')))['files']
    t01b_files = json.load(open(os.path.join(REPO, T01B_ROOT, 'manifest_sha256.json')))['files']
    for name in ('lp/index.json', 'objective.json', 'capacity.json', 'inventory.json'):
        check(os.path.join(T01C_ROOT, name), t01c_files[name]['sha256'], f't01c/{name}')
    for cell, y, d in SMOKE:
        for name in (f'lp/{cell}/{y}_{d}_{MODE}.lp', f'lp/{cell}/{y}_{d}_{MODE}.name_map.json'):
            check(os.path.join(T01C_ROOT, name), t01c_files[name]['sha256'], f't01c/{name}')
        for name in (f'captures/{cell}/blocks_{y}_{d}.pkl', f'captures/{cell}/vectors_{y}_{d}.json',
                     f'captures/{cell}/vectors_esso.json'):
            check(os.path.join(T01B_ROOT, name), t01b_files[name]['sha256'], f't01b/{name}')
    if mismatch:
        write_json(record_name, {'refused': True, 'inputs': inputs, 'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5
    objective_t01c = json.load(open(os.path.join(REPO, T01C_ROOT, 'objective.json')))
    capacity = json.load(open(os.path.join(REPO, T01C_ROOT, 'capacity.json')))
    lp_index = json.load(open(os.path.join(REPO, T01C_ROOT, 'lp', 'index.json')))['lp_files']
    inv = json.load(open(os.path.join(REPO, T01C_ROOT, 'inventory.json')))
    record_sha = {c['label']: inv['cell_records'][c['label']]['record_sha256']['evaluation_record.json']
                  for c in T1B.CELLS}
    del inv
    gc.collect()
    instances = {}
    for cell in T1B.CELLS:
        ev_path = os.path.join(cell['eval_dir'], 'evaluation_record.json')
        check(ev_path, record_sha[cell['label']], f'{cell["label"]}/evaluation_record.json')
        ev = json.load(open(os.path.join(REPO, ev_path)))
        instances[cell['label']] = {'candidate_key': ev['candidate_key'],
                                    'certification_cycle': ev['certification_cycle'],
                                    'gross_recorded': float(ev['terminal_gross_operational_cost']),
                                    'eval_dir': cell['eval_dir']}
    if mismatch:
        write_json(record_name, {'refused': True, 'inputs': inputs, 'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5
    print(f'[inputs] {len(inputs)} files hash-checked against the T0.1b / T0.1c manifests: all match')
    cal_factor = {tuple(int(v) for v in k.strip('()').split(',')): rec['factor']
                  for k, rec in capacity['calendar_inputs']['calendar_factor'].items()}

    # ---- static leaf check ---------------------------------------------------------------------------------------
    project_modules = {os.path.splitext(f)[0] for f in os.listdir(REPO) if f.endswith('.py')}
    leaf_names = set(_leaf_solve.__code__.co_names)
    leaf_project_refs = sorted(leaf_names & (project_modules - {'p516_d0_clarabel'}))
    if leaf_project_refs:
        raise RuntimeError(f'the leaf names project modules {leaf_project_refs}: production reachable from it')
    _STATE.update({'clarabel': clarabel, 'clarabel_guard': clarabel_guard})
    env = gp.Env(empty=True)
    env.setParam('LogToConsole', 0)
    env.start()
    print(f'[leaf] static check: leaf names {sorted(leaf_names)}; project modules referenced: none; gurobipy '
          f'{".".join(str(v) for v in gp.gurobi.version())} Env started (reader only; optimize blocked); Clarabel '
          f'{clarabel.__version__}')

    gate_rows, stops = [], []

    def gate(name, scope, achieved, threshold, note=None):
        achieved = float(achieved)
        ok = bool(achieved <= threshold)
        gate_rows.append({'gate': name, 'scope': scope, 'achieved': achieved, 'threshold': threshold, 'pass': ok,
                          'note': note})
        print(f'[gate {"pass" if ok else "FAIL"}] {name} {scope}: {achieved:.6e} <= {threshold:.3e}')
        return ok

    import pickle

    # ==================================================================================================================
    #  Phase 1 (zero solves): read, map, build, gates (i) and (ii) on BOTH groups before any solve
    # ==================================================================================================================
    lp_root = os.path.join(REPO, T01C_ROOT)
    groups, prepared = {}, {}
    for cell, y, d in SMOKE:
        tag = f'{cell}_{y}_{d}_{VAR}'
        gkey = f'{y}_{d}'
        print(f'\n========== build {tag} ==========')
        lp_rel = f'lp/{cell}/{gkey}_{MODE}.lp'
        entry = lp_index[lp_rel]
        t0 = time.time()
        name_map = json.load(open(os.path.join(lp_root, entry['name_map'])))['name_map']
        data = T02.read_lp_data(os.path.join(lp_root, lp_rel), name_map, env)
        del name_map
        t_read = time.time() - t0
        with open(os.path.join(REPO, T01B_ROOT, f'captures/{cell}/blocks_{y}_{d}.pkl'), 'rb') as handle:
            blocks = pickle.load(handle)
        vectors = json.load(open(os.path.join(REPO, T01B_ROOT, f'captures/{cell}/vectors_{y}_{d}.json')))
        vectors_esso = json.load(open(os.path.join(REPO, T01B_ROOT, f'captures/{cell}/vectors_esso.json')))
        networks = {'TSO': blocks['tso'].node_balance_p._rule._fcn.keywords['network']}
        for n_ in (5, 7, 9):
            networks[f'DSO{n_}'] = blocks['dso'][n_].node_balance_p._rule._fcn.keywords['network']
        x_cert = T02.certified_point(data, vectors, vectors_esso, networks,
                                     lambda idx: cal_factor[(idx[0], idx[1])])
        del blocks, vectors, vectors_esso
        gc.collect()
        t1 = time.time()
        try:
            conic = build_conic(data, presolve=PRESOLVE)
        except ConeMappingError as error:
            stops.append({'stop': 'a quadratic row cannot be mapped to a second-order cone', 'group': tag,
                          'row': error.row, 'detail': error.detail})
            print(f'[STOP] {error}')
            break
        t_build = time.time() - t1
        print(f'[build] {tag}: read {t_read:.2f} s, conic build {t_build:.2f} s; {json.dumps(conic["inventory"])}')
        gate('conic inventory: dimensions consistent with the LP (0 = consistent)', tag,
             0.0 if conic['inventory']['dims_consistent'] else 1.0, 0.0)
        # ---- gate (i): objective identity at the certified point ----
        obj_lp = float(data['c'] @ x_cert) + data['obj_con']
        obj_cl = objective(conic, x_cert)
        ref = objective_t01c['cells'][cell]['per_group'][gkey][VAR]['group_relaxed_objective_certified']
        gate('(i) model identity: Clarabel objective vs LP objective at the certified point (rel)', tag,
             abs(obj_cl - obj_lp) / abs(obj_lp), IDENTITY_OBJ_REL)
        gate('(i) model identity: Clarabel objective at the certified point vs T0.1c recorded (rel)', tag,
             abs(obj_cl - ref) / abs(ref), IDENTITY_OBJ_REL)
        # ---- gate (ii): feasibility of the mapped point ----
        feas = gate_identity_and_classes(conic, data, x_cert, cell)
        for kind in ('linear_or_bound', 'cone') + (('presolved_cone',) if PRESOLVE else ()):
            gate(f'(ii) identity: rows whose Clarabel-model violation differs from the LP form beyond fp ({kind})',
                 tag, float(feas['identity']['n_over_tolerance'][kind]), 0.0)
        gate('(ii) violated rows outside their stated class threshold (count)', tag,
             float(feas['n_over_class_threshold']), 0.0)
        q_plus = objective_t01c['q_cert_plus'][cell]['per_group'][gkey]
        groups[tag] = {
            'cell': cell, 'year': y, 'day': d, 'variant': VAR, 'instance': instances[cell],
            'lp': os.path.join(T01C_ROOT, lp_rel), 'lp_sha256': t01c_files[lp_rel]['sha256'],
            'name_map_sha256': t01c_files[entry['name_map']]['sha256'], 'lp_size': data['size'],
            'conic_inventory': conic['inventory'], 'read_s': t_read, 'build_s': t_build,
            'gate_i': {'obj_lp_at_certified': obj_lp, 'obj_clarabel_at_certified': obj_cl,
                       'obj_t01c_recorded': ref, 'rel_vs_lp': abs(obj_cl - obj_lp) / abs(obj_lp),
                       'rel_vs_t01c': abs(obj_cl - ref) / abs(ref)},
            'gate_ii': feas, 'gross_group': q_plus['gross'], 'q_cert_plus_group': q_plus['q_cert_plus'],
            'q_cert_plus_correction': q_plus['q_cert_plus'] - q_plus['gross']}
        if PRESOLVE:
            groups[tag]['presolve'] = conic['presolve']
            print(f'[presolve] {tag}: {conic["presolve"]["n_columns_pinned"]} columns pinned '
                  f'{conic["presolve"]["pinned_by_row_family"]}; Lorentz cones replaced '
                  f'{conic["presolve"]["n_lorentz_cones_replaced"]} of {conic["presolve"]["n_lorentz_cones"]} '
                  f'(heads {conic["presolve"]["pinned_heads"]}; kept {conic["presolve"]["kept_heads"]}); not altered: '
                  f'{ {k: v["count"] for k, v in conic["presolve"]["not_altered"].items()} }; small circles by '
                  f'family {conic["presolve"]["not_altered"]["small_circles_r_lt_1e-4"]["by_family"]}; identity '
                  f'at the certified point {json.dumps(feas["identity"].get("presolved"))}')
        prepared[tag] = (data, conic, x_cert, networks)
        print(f'[gate i/ii] {tag}: obj at certified {obj_cl:.6f} (T0.1c {ref:.6f}); identity over-tolerance '
              f'{feas["identity"]["n_over_tolerance"]}; class over-threshold {feas["n_over_class_threshold"]}')

    # ==================================================================================================================
    #  Phase 2: the declared solves (only if no stop), then the measurements
    # ==================================================================================================================
    rss = {}
    solves = {}                                 # ablation: one record per solve label (the smoke records per group)
    n_solved = 0

    def note_rss():
        return {'self_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

    if not stops:
        for label, tag, max_iter in SOLVE_PLAN:
            data, conic, x_cert, networks = prepared[tag]
            if ABLATION:
                g = solves[label] = {'group': tag, 'max_iter': max_iter, 'presolve': PRESOLVE,
                                     'instance': groups[tag]['instance'], 'gross_group': groups[tag]['gross_group'],
                                     'q_cert_plus_group': groups[tag]['q_cert_plus_group']}
            else:
                g = groups[tag]
            print(f'\n========== solve {label} ==========')
            cones = clarabel_cones(conic)
            settings = dict(CLARABEL_OVERRIDES)
            if max_iter is not None:
                settings['max_iter'] = max_iter
            rss_before = note_rss()
            log_path = os.path.join(OUT_DIR, 'logs', f'{label}.clarabel.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            if DRY:
                st = clarabel.DefaultSettings()
                for k_, v_ in settings.items():
                    setattr(st, k_, v_)
                t0 = time.time()
                solver = clarabel.DefaultSolver(conic['P'], conic['q'], conic['A'], conic['b'], cones, st)
                solver.print_to_buffer()
                solver.print_configuration()
                buf = solver.get_print_buffer()
                del solver
                res = {'STAND_IN': True, 'status': 'DRY', 'setup_s': time.time() - t0, 'solve_wall_s': 0.0,
                       'log_text': buf, 'obj_val': objective(conic, x_cert) - conic['const'],
                       'obj_val_dual': objective(conic, x_cert) - conic['const'], 'iterations': 0,
                       'solve_time_reported': 0.0, 'r_prim': None, 'r_dual': None, 'x': x_cert,
                       'z': np.zeros(len(conic['b'])), 's': conic_slack(conic, x_cert), 'info': {},
                       'settings_in_force': {}}
            else:
                payload = {'P': conic['P'], 'q': conic['q'], 'A': conic['A'], 'b': conic['b'], 'cones': cones,
                           'settings': settings}
                res = _leaf_solve(payload)
                n_solved += 1
            rss_after = note_rss()
            with open(log_path, 'w') as handle:
                handle.write(res['log_text'] or '')
            print(res['log_text'] or '[log] (empty print buffer)')
            x, z, s = res['x'], res['z'], res['s']
            vec_path = os.path.join(OUT_DIR, 'vectors', f'{label}.npz')
            os.makedirs(os.path.dirname(vec_path), exist_ok=True)
            np.savez_compressed(vec_path, x=x, z=z, s=s, x_cert=x_cert)    # saved BEFORE any post-processing
            q_lb = res['obj_val'] + conic['const']
            gap_s = abs(res['obj_val'] - res['obj_val_dual'])
            g['solve'] = {k: res[k] for k in ('status', 'obj_val', 'obj_val_dual', 'iterations', 'setup_s',
                                              'solve_wall_s', 'solve_time_reported', 'r_prim', 'r_dual', 'info',
                                              'settings_in_force')}
            g['solve'].update({'STAND_IN': bool(res.get('STAND_IN')), 'log': rel(log_path), 'vectors': rel(vec_path),
                               'wall_s': res['setup_s'] + res['solve_wall_s'], 'rss_before': rss_before,
                               'rss_after': rss_after, 'clarabel_overrides': settings})
            g['Q_LB'] = q_lb
            g['solver_gap_abs'] = gap_s
            g['solver_gap_rel'] = gap_s / max(1.0, min(abs(res['obj_val']), abs(res['obj_val_dual'])))
            g['gap_vs_q_cert_plus_eur'] = g['q_cert_plus_group'] - q_lb
            g['gap_vs_q_cert_plus_keur'] = g['gap_vs_q_cert_plus_eur'] / 1e3
            g['gap_vs_q_cert_plus_rel'] = g['gap_vs_q_cert_plus_eur'] / g['q_cert_plus_group']
            g['gap_vs_gross_eur'] = g['gross_group'] - q_lb
            print(f'[solve] {label}: status {res["status"]}, {res["iterations"]} it, Q_LB {q_lb:.4f}, solver gap '
                  f'{gap_s:.4e} ({g["solver_gap_rel"]:.3e} rel); Q_cert+ {g["q_cert_plus_group"]:.4f}; gap '
                  f'{g["gap_vs_q_cert_plus_keur"]:.3f} kEUR ({g["gap_vs_q_cert_plus_rel"]:.3e}); setup '
                  f'{res["setup_s"]:.2f} s, solve {res["solve_wall_s"]:.2f} s; peak RSS {rss_after}')
            if ABLATION:
                g['converged'] = res['status'] == 'Solved'
                g['containment_stop_applies'] = res['status'] in CONVERGED_STATUSES
                print(f'[solve] {label}: r_prim {res["r_prim"]}, r_dual {res["r_dual"]}; info res_primal '
                      f'{res["info"].get("res_primal")}, res_dual {res["info"].get("res_dual")}, gap_abs '
                      f'{res["info"].get("gap_abs")}, gap_rel {res["info"].get("gap_rel")}; converged '
                      f'{g["converged"]}')
            # ---- gate (iii): containment ----
            if not gate('(iii) G-D0-4 containment: Q_LB - Q_cert+ - solver gap (<= 0)', label,
                        q_lb - g['q_cert_plus_group'] - gap_s, 0.0):
                if not ABLATION:
                    stops.append({'stop': 'G-D0-4 containment violated', 'group': tag, 'Q_LB': q_lb,
                                  'Q_cert_plus': g['q_cert_plus_group'], 'solver_gap': gap_s})
                elif g['containment_stop_applies']:
                    stops.append({'stop': 'G-D0-4 containment violated at a converged point', 'group': tag,
                                  'solve': label, 'status': res['status'], 'Q_LB': q_lb,
                                  'Q_cert_plus': g['q_cert_plus_group'], 'solver_gap': gap_s})
                else:
                    g['containment_note'] = 'violated at an UNCONVERGED point: recorded, not a stop (task rule)'
            # ---- gate (v): duals ----
            try:
                if ABLATION:
                    g['returned_point_lp_violation'] = returned_point_violation(data, x)
                    g['small_circle_activity'] = small_circle_activity(conic, data, s, z)
                    rpv = g['returned_point_lp_violation']
                    worst_ = {k_: (v_['max'], v_.get('worst_row') or v_.get('worst_column'))
                              for k_, v_ in rpv.items() if isinstance(v_, dict)}
                    print(f'[point] {label}: max LP-row violation {rpv["max_over_all"]:.3e} {json.dumps(worst_)}')
                g['duals'] = conic_dual_checks(conic, x, s, z, res['obj_val'], res['obj_val_dual'])
                gm = gurobi_convention_duals(conic, data, x, s, z)
                g['duals']['gurobi_convention'] = gm['check']
                np.savez_compressed(os.path.join(OUT_DIR, 'vectors', f'{label}.gurobi_convention.npz'),
                                    X=x, Pi=gm['Pi'], QCPi=gm['QCPi'], RC=gm['RC'])
                print(f'[duals] {label}: ||q + A\'z||_inf {g["duals"]["dual_residual_inf"]:.3e} (q_inf '
                      f'{g["duals"]["q_inf"]:.3e}); gap Clarabel {g["duals"]["gap_abs_clarabel"]:.4e}, recomputed '
                      f'{g["duals"]["gap_abs_recomputed"]:.4e}; dual-cone viol SOC {g["duals"]["soc_max_z_violation"]:.2e}'
                      f', min z_nonneg {g["duals"]["nonneg_min_z"]:.2e}; mapped stationarity '
                      f'{gm["check"]["mapped_stationarity_residual_inf"]:.3e}')
                if ABLATION:
                    print(f'[duals] {label}: Gurobi-convention stationarity rel to ||c||_inf: strict '
                          f'{gm["check"]["mapped_stationarity_residual_rel_to_c_inf"]:.3e} (argmax '
                          f'{gm["check"]["argmax_strict"]}), folded '
                          f'{gm["check"]["folded_stationarity_residual_rel_to_c_inf"]:.3e} (argmax '
                          f'{gm["check"]["argmax_folded"]}), untouched columns '
                          f'{gm["check"]["untouched_columns_stationarity_residual_rel_to_c_inf"]:.3e}; presolve-row '
                          f'max |z| {gm["check"]["presolve_rows_max_abs_z"]:.3e}; cones with s0 <= 0 '
                          f'{gm["check"]["n_cones_with_s0_le_0"]}')
                # T0.2 measurements on the Clarabel solution, through the T0.2 functions themselves
                g['t02_lagrangian_g'] = T02.lagrangian_bound(data, x, gm['Pi'], gm['QCPi'], gm['RC'], q_lb)
                rk = T02.rank_and_cycles(data, x, networks, T02._cycle_basis)
                g['t02_rank_and_cycles_summary'] = {
                    b_: {k_: v_ for k_, v_ in r_.items() if not k_.startswith('per_')} for b_, r_ in rk.items()}
                for b_, r_ in rk.items():
                    if 'oltc' in r_:
                        g['t02_rank_and_cycles_summary'][b_]['oltc'] = {
                            k_: v_ for k_, v_ in r_['oltc'].items() if k_ != 'per_branch_period'}
            except Exception as error:      # noqa: BLE001 -- recorded; vectors are already saved
                import traceback
                g['postprocessing_error'] = traceback.format_exc()
                print(f'[ERROR] post-processing {label}: {error!r} (vectors saved; recomputable without a solve)')
            gc.collect()
            if stops:
                print('[STOP] containment violated; no further solve')
                break

    # ---- gate (iv): P7 ----
    tags = list(groups)
    t_x0, t_u = tags[0], tags[1] if len(tags) > 1 else None
    readings = {}
    if not ABLATION and t_u and 'Q_LB' in groups[t_x0] and 'Q_LB' in groups[t_u]:
        a, b_ = groups[t_x0], groups[t_u]
        tol = a['solver_gap_abs'] + b_['solver_gap_abs']
        readings['P7'] = {'Q_LB_unit_minus_x0': b_['Q_LB'] - a['Q_LB'], 'tol': tol,
                          'held': bool(b_['Q_LB'] <= a['Q_LB'] + tol),
                          'delta_Q_LB_x0_minus_unit': a['Q_LB'] - b_['Q_LB'],
                          'delta_Q_cert_plus_x0_minus_unit': a['q_cert_plus_group'] - b_['q_cert_plus_group']}
        gate('(iv) P7 Q_LB(unit) - Q_LB(x0) - tolerance (<= 0)', f'{t_u} vs {t_x0}',
             b_['Q_LB'] - a['Q_LB'] - tol, 0.0)
    if ABLATION:
        readings['P7_pairs'] = {}
        for it in (200, 5000):
            lx, lu = f'{_TAGS[0]}.presolve.it{it}', f'{_TAGS[1]}.presolve.it{it}'
            if 'Q_LB' not in solves.get(lx, {}) or 'Q_LB' not in solves.get(lu, {}):
                readings['P7_pairs'][f'it{it}'] = {'available': False}
                continue
            a, b_ = solves[lx], solves[lu]
            tol = a['solver_gap_abs'] + b_['solver_gap_abs']
            both = bool(a['solve']['status'] == 'Solved' and b_['solve']['status'] == 'Solved')
            readings['P7_pairs'][f'it{it}'] = {
                'available': True, 'both_converged': both, 'statuses': [a['solve']['status'], b_['solve']['status']],
                'Q_LB_x0': a['Q_LB'], 'Q_LB_unit': b_['Q_LB'], 'Q_LB_unit_minus_x0': b_['Q_LB'] - a['Q_LB'],
                'tol': tol, 'held': bool(b_['Q_LB'] <= a['Q_LB'] + tol),
                'delta_Q_cert_plus_x0_minus_unit': a['q_cert_plus_group'] - b_['q_cert_plus_group'],
                'note': None if both else 'not both converged: no P7 gate (a reading between iterates is no bound)'}
            if both:
                gate('(iv) P7 Q_LB(unit) - Q_LB(x0) - tolerance (<= 0), converged pair', f'{lu} vs {lx}',
                     b_['Q_LB'] - a['Q_LB'] - tol, 0.0)

    # ---- predictions ----
    preds = {}
    solved = [g for g in groups.values() if 'solve' in g]
    if not ABLATION and len(solved) == 2 and not DRY:
        preds['K1'] = {'per_group': {t: {'status': g['solve']['status'], 'gap_rel': g['solver_gap_rel'],
                                         'gap_abs_eur': g['solver_gap_abs']} for t, g in groups.items()},
                       'verdict': 'HELD' if all(g['solve']['status'] == 'Solved' and g['solver_gap_rel'] <= 1e-8
                                                for g in solved) else 'FAILED'}
        preds['K2'] = {'per_group': {t: g['Q_LB'] - g['q_cert_plus_group'] - g['solver_gap_abs']
                                     for t, g in groups.items()},
                       'verdict': 'HELD' if all(g['Q_LB'] <= g['q_cert_plus_group'] + g['solver_gap_abs']
                                                for g in solved) else 'FAILED'}
        g0 = groups[t_x0]
        preds['K3'] = {'gap_keur': g0['gap_vs_q_cert_plus_keur'], 'gap_rel': g0['gap_vs_q_cert_plus_rel'],
                       'verdict': 'HELD' if 0.0 <= g0['gap_vs_q_cert_plus_rel'] <= 1e-2 else 'FAILED'}
    if ABLATION and not DRY:
        def lab(gi, it):
            return f'{_TAGS[gi]}.presolve.it{it}'

        def status(label_):
            return solves.get(label_, {}).get('solve', {}).get('status')

        per = {l_: {'status': status(l_), 'iterations': solves[l_]['solve']['iterations']}
               for l_ in solves if 'solve' in solves[l_]}
        b1 = all(status(lab(gi, 200)) == 'Solved' for gi in (0, 1))
        preds['B1'] = {'per_solve': per, 'verdict': 'HELD' if b1 else 'FAILED'}
        if b1:
            preds['B2'] = {'verdict': 'NOT APPLICABLE (B1 held)',
                           'it5000_statuses': {lab(gi, 5000): status(lab(gi, 5000)) for gi in (0, 1)}}
        else:
            need = [gi for gi in (0, 1) if status(lab(gi, 200)) != 'Solved']
            st5 = {lab(gi, 5000): status(lab(gi, 5000)) for gi in need}
            preds['B2'] = {'groups_failed_at_200': [_TAGS[gi] for gi in need], 'it5000_statuses': st5,
                           'verdict': ('INDETERMINATE (not run)' if any(v is None for v in st5.values()) else
                                       'HELD' if all(v == 'Solved' for v in st5.values()) else 'FAILED')}
        pick = next((l_ for l_ in (lab(0, 5000), lab(0, 200)) if status(l_) == 'Solved'), None)
        if pick is None:
            preds['B3'] = {'verdict': 'INDETERMINATE (no converged x = 0 solve)'}
        else:
            s0_ = solves[pick]
            cont = s0_['Q_LB'] - s0_['q_cert_plus_group'] - s0_['solver_gap_abs']
            preds['B3'] = {'solve': pick, 'containment_Q_LB_minus_Qcert_plus_minus_gap': cont,
                           'gap_keur': s0_['gap_vs_q_cert_plus_keur'], 'gap_rel': s0_['gap_vs_q_cert_plus_rel'],
                           'verdict': 'HELD' if cont <= 0.0 and s0_['gap_vs_q_cert_plus_rel'] <= 1e-2 else 'FAILED'}
        pair = next((k_ for k_ in ('it5000', 'it200') if readings['P7_pairs'].get(k_, {}).get('both_converged')),
                    None)
        if pair is None:
            preds['B4'] = {'verdict': 'INDETERMINATE (no converged pair)'}
        else:
            p_ = readings['P7_pairs'][pair]
            preds['B4'] = {'pair': pair, 'Q_LB_unit_minus_x0': p_['Q_LB_unit_minus_x0'], 'tol': p_['tol'],
                           'verdict': 'HELD' if p_['held'] else 'FAILED'}
        conv5 = {l_: solves[l_]['duals']['gurobi_convention'] for l_ in solves
                 if status(l_) == 'Solved' and 'duals' in solves[l_]}
        if not conv5:
            preds['B5'] = {'verdict': 'INDETERMINATE (no converged point)'}
        else:
            preds['B5'] = {
                'per_solve': {l_: {k_: c_[k_] for k_ in ('mapped_stationarity_residual_rel_to_c_inf',
                                                         'folded_stationarity_residual_rel_to_c_inf',
                                                         'untouched_columns_stationarity_residual_rel_to_c_inf',
                                                         'argmax_strict', 'argmax_folded',
                                                         'presolve_rows_max_abs_z')}
                              for l_, c_ in conv5.items()},
                'verdict': 'HELD' if all(c_['mapped_stationarity_residual_rel_to_c_inf'] <= B5_STATIONARITY_REL
                                         for c_ in conv5.values()) else 'FAILED',
                'verdict_folded_reading': 'HELD' if all(c_['folded_stationarity_residual_rel_to_c_inf'] <=
                                                        B5_STATIONARITY_REL for c_ in conv5.values()) else 'FAILED'}
    for k_, text in PREDICTIONS.items():
        preds.setdefault(k_, {'verdict': None})['prediction'] = text

    # ---- projection (vi) ----
    if solved and not ABLATION:
        mean_solve = sum(g['solve']['wall_s'] for g in solved) / len(solved)
        mean_prep = sum(g['read_s'] + g['build_s'] for g in groups.values()) / len(groups)
        readings['projection_108'] = {
            'mean_setup_plus_solve_s': mean_solve, 'mean_read_plus_build_s': mean_prep,
            'projected_108_solves_s': 108 * mean_solve, 'projected_72_reads_builds_s': 72 * mean_prep,
            'projected_total_s': 108 * mean_solve + 72 * mean_prep,
            'peak_rss_bytes': max(g['solve']['rss_after']['self_max_rss_bytes'] for g in solved),
            'note': ('solver-side only; excludes T0.2 post-processing (Lagrangian margin, rank, capacity sensitivity, '
                     'block unpickling) and the 24 IPOPT cross-checks; the fixed-mode and freeze solves are assumed '
                     'to cost one group solve each')}
    real = [g for g in solves.values() if 'solve' in g and g['solve']['iterations'] > 0]
    if ABLATION and real:
        mean_prep = sum(g['read_s'] + g['build_s'] for g in groups.values()) / len(groups)
        per_it = sum(g['solve']['solve_wall_s'] / g['solve']['iterations'] for g in real) / len(real)
        first_conv = {}
        for gi in (0, 1):
            l_ = next((lab_ for lab_ in (f'{_TAGS[gi]}.presolve.it200', f'{_TAGS[gi]}.presolve.it5000')
                       if solves.get(lab_, {}).get('solve', {}).get('status') == 'Solved'), None)
            if l_:
                first_conv[_TAGS[gi]] = l_
        proj = {'mean_read_plus_build_s': mean_prep, 'mean_solve_s_per_iteration': per_it,
                'projected_72_reads_builds_s': 72 * mean_prep,
                'peak_rss_bytes': max(g['solve']['rss_after']['self_max_rss_bytes'] for g in real),
                'first_converged_solve_per_group': first_conv,
                'note': ('solver-side only; excludes T0.2 post-processing (Lagrangian margin, rank, capacity '
                         'sensitivity, block unpickling) and the 24 IPOPT cross-checks; the fixed-mode and freeze '
                         'solves are assumed to cost one group solve each; two groups of 72 measured')}
        if len(first_conv) == 2:
            chosen = [solves[l_] for l_ in first_conv.values()]
            mean_solve = sum(g['solve']['wall_s'] for g in chosen) / len(chosen)
            mean_iter = sum(g['solve']['iterations'] for g in chosen) / len(chosen)
            proj.update({'basis': 'converged', 'mean_setup_plus_solve_s': mean_solve,
                         'mean_iterations_to_converge': mean_iter, 'projected_108_solves_s': 108 * mean_solve,
                         'projected_total_s': 108 * mean_solve + 72 * mean_prep})
        else:
            caps = [g for g in real if g['max_iter'] == 5000]
            if caps:
                mean_solve = sum(g['solve']['wall_s'] for g in caps) / len(caps)
                proj.update({'basis': 'NOT converged on both groups: 108 x the mean it5000 wall (a floor at the cap)',
                             'mean_setup_plus_solve_s': mean_solve, 'projected_108_solves_s': 108 * mean_solve,
                             'projected_total_s': 108 * mean_solve + 72 * mean_prep})
        readings['projection_108'] = proj

    # ---- guards, containment, provenance ----
    expected = 0 if DRY else n_solved
    if not DRY and not stops and expected != DECLARED_SOLVES:
        stops.append({'stop': f'solved {expected} != declared {DECLARED_SOLVES}'})
    cl_fail = clarabel_guard.verify(expected)
    py_fail = pyomo_guard.verify(0)
    gu_fail = gurobi_guard.verify(0)
    for guard in (clarabel_guard, pyomo_guard, gurobi_guard):
        guard.uninstall()
    env.dispose()
    gate('G-D0-6 leaf guard exact (failures: Clarabel, Pyomo 0, Gurobi 0)', 'process',
         float(len(cl_fail) + len(py_fail) + len(gu_fail)), 0.0)
    if cl_fail or py_fail or gu_fail:
        stops.append({'stop': 'leaf-guard count disagrees', 'clarabel': cl_fail, 'pyomo': py_fail, 'gurobi': gu_fail})
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
                       'clarabel': clarabel.__version__,
                       'clarabel_buildinfo': ('printed to run.log by clarabel.buildinfo() (returns None): '
                                              f'{clarabel.buildinfo()!r}'),
                       'gurobipy_reader': '.'.join(str(v) for v in gp.gurobi.version()),
                       'HOME_in_child': os.environ.get('HOME'), 'project_modules_loaded': repo_modules,
                       'harness_sha256': sha256_file(os.path.abspath(__file__)),
                       'spec_sha256': sha256_file(os.path.join(REPO, 'P5_16_D0_SPEC_v2.md')),
                       'inputs': inputs, 'rehearsal_stand_in': DRY, 'wall_s': time.time() - t_start})
    guard_out = {'leaf': leaf_site, 'declared_solves': DECLARED_SOLVES, 'expected_at_verify': expected,
                 'counted_events': {'clarabel': dict(clarabel_guard.counts), 'pyomo': dict(pyomo_guard.counts),
                                    'gurobi': dict(gurobi_guard.counts)},
                 'verify_failures': {'clarabel': cl_fail, 'pyomo': py_fail, 'gurobi': gu_fail},
                 'process_launch_events': T02._AUDIT['process_launch_events'],
                 'leaf_static_check': {'names': sorted(leaf_names), 'project_modules_referenced': leaf_project_refs}}
    gates_out = {'rows': gate_rows, 'all_pass': all(r['pass'] for r in gate_rows),
                 'failed': [r for r in gate_rows if not r['pass']], 'stops': stops, 'stop_triggered': bool(stops)}
    payload = {'stage': f'P5.16 D0 / T0.2-clarabel {"ablation" if ABLATION else "smoke"}',
               'rehearsal_stand_in': DRY,
               'objective_convention': ('gross_operational_cost (settlement-EXCLUDED, salvage NOT subtracted) per '
                                        'group; Q_cert+ = gross with every detector slack at max(slack, 0) (spec '
                                        'section 12 ruling 5); gap = Q_cert+ - Q_LB; variant C (calendar-only)'),
               'conic_form': __doc__.split('Conic form', 1)[1].split('Declared solves', 1)[0],
               'groups': groups, 'readings': readings, 'predictions': preds, 'gates': gates_out, 'guard': guard_out,
               'containment': {'write_opens_in_repo_outside_out_dir': writes_repo},
               'thresholds': {'identity_obj_rel': IDENTITY_OBJ_REL, 'ipopt_bound_relax': IPOPT_BOUND_RELAX,
                              't01c_mapping_delta': T01C_MAPPING_DELTA, 'capacity_shift_C': CAPACITY_SHIFT_C},
               'provenance': provenance}
    if ABLATION:
        payload['solve_plan'] = [{'label': l_, 'group': t_, 'max_iter': m_, 'presolve': PRESOLVE}
                                 for l_, t_, m_ in SOLVE_PLAN]
        payload['solves'] = solves
        payload['thresholds'].update({'B5_stationarity_rel': B5_STATIONARITY_REL, 'small_circle_r': SMALL_CIRCLE_R,
                                      'converged_statuses_for_the_containment_stop': CONVERGED_STATUSES})
    write_json(record_name, payload)
    print(f'[write] {record_name}')
    print(f'[summary] gates {len(gate_rows)} rows, all pass {gates_out["all_pass"]}; failed '
          f'{sorted({(r["gate"], r["scope"]) for r in gates_out["failed"]})}; stops {stops}; guards '
          f'{guard_out["counted_events"]}; repo writes outside OUT_DIR {len(writes_repo)}; wall '
          f'{time.time() - t_start:.1f} s')
    for key, value in preds.items():
        print(f'[prediction] {key}: {value.get("verdict")}')
    if stops:
        return 3
    return 0 if gates_out['all_pass'] else 2


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        sys.exit(child())
    sys.exit(parent())
