"""P5.16 D0 / T0.2 -- the measurement: 132 declared solves of the exported relaxations.

Governing spec: `P5_16_D0_SPEC_v2.md` section 1 (decision rule), section 4 G-D0-4 / G-D0-5 / G-D0-6, section 5
measurements (a)-(g), section 6 predictions P1-P9 and the expert's variant predictions, sections 9, 11 (D-1: variant A
is NEVER a bound; D-3: absolute class on C, the mechanism / P6 on A, R reported), 12.5 (Q_cert+) and 13 (declared
count, ESSO separability). Inputs: the T0.1c exports (`P516D0/t01c/`, manifest sha256 d2955128...) and the T0.1b
captures (`P516D0/t01b/`, manifest sha256 4e5bbd45...), every file hash-checked against its manifest before use.

Declared solves -- exactly 132, through ONE leaf function `_leaf_solve` (section 4 G-D0-6):
  Gurobi (108): every exported group LP, 3 variants x 2 cells x 12 groups = 72; variant C at the unit cell: the
                fixed-mode re-solve (12) and the two-way freeze attribution (24).
  IPOPT  (24):  smooth-form cross-check, variant C, both cells, 12 groups.
Two armed guards, both permitting only `p516_d0_t02.py:_leaf_solve`, each counting two events per solve:
  * `p513_solve_profile_guard.SolveProfileGuard` (Pyomo OptSolver.solve + SystemCallSolver._execute_command): 24 + 24;
  * `GurobiLeafGuard` (this file): the leaf's Gurobi entry + `gurobipy.Model.optimize` (patched on the type): 108 + 108.
  Declared count = 2 x 132 = 264 counted events, verified exactly at the end (too few fails too). Any solve reached
  from another call site raises. The audit hook raises on any process launch outside the leaf.
  The leaf references gurobipy and the Pyomo solver factory only; a static check before the first solve asserts that
  its code object names no project module (production entry points unreachable from it).

Gurobi settings (G-D0-5 + spec section 15 ruling 3): QCPDual 1, NonConvex 0, BarQCPConvTol 1e-9, BarConvTol 1e-10,
FeasibilityTol 1e-9, OptimalityTol 1e-9, Threads 4, LogFile per solve, LogToConsole 0; everything else at Gurobi's
defaults. The 24 freeze solves ONLY (both freezes; spec section 17 ruling 2) run with FeasibilityTol 1e-7
(`GUROBI_PARAMS_FREEZE`): their all-fixed rows carry the certified point's own ~1e-8 slack by construction, and they are
attribution diagnostics, never bounds; the 72 group LPs and the 12 fixed-mode solves keep 1e-9. Each solve records the
dict it ran with (`params`, `params_set`). A log containing "failed to compute QCP dual" marks the solve
dual-less (recorded; not a stop). A model Gurobi refuses as nonconvex is a STOP (the row is reported).
IPOPT settings (spec section 15 ruling 2; section 16 rulings 2 and 5): tol 1e-9, max_iter 3000, print_level 5 (the
whole stdout is the per-solve log), bound_relax_factor 0, honor_original_bounds yes, bound_push 1e-10, bound_frac 1e-10,
linear_solver ma97, mu_strategy adaptive (the interior test's linear solver and barrier strategy); everything else at
IPOPT 3.14 defaults. No warm_start_init_point (section 16 ruling 2): the default initializer pushes the primal start
into the bounds by bound_push / bound_frac = 1e-10, so the certified start is kept, and the multipliers take the default
initialisation (bound multipliers bound_mult_init_val = 1; constraint multipliers the least-squares estimate, set to 0
when it exceeds constr_mult_init_max). IPOPT dimension gate (section 16 ruling 3): IPOPT's reported variable count ==
the LP's columns minus the columns fixed by their bounds (lb == ub), which IPOPT removes; both numbers recorded.
Iteration 0 (objective, inf_pr, inf_du) is parsed from the log and recorded
beside the start's own LP-row violation, to show whether the start is kept. The IPOPT model is built from the SAME
exported LP (read with gurobipy, no solve): every linear row,
every quadratic row as written (the rotated cone as the bilinear inequality WijR^2 + WijI^2 <= Wii * Wjj, with
Wii >= v_min^2 > 0 asserted on the LP bounds), the same bounds, the same linear objective; initial point = the rank-1
image of the certified point (T0.1c's mapped point), clipped into the bounds.

Resolution column (spec section 15 ruling 1; record schema v3): after every solve (Gurobi and IPOPT, every variant and
kind) res_LB = sum |multiplier| * max(0, violation) over every linear row, quadratic row and bound at the returned point,
with the solver's own multipliers, and res_cert = the same pricing of the certified mapped point (clipped into the
solve's bounds) at those multipliers (`resolution_price`, definition in RESOLUTION_DEFINITION). Every gap reading
carries (section 16 ruling 1) the FIRM interval for the exact gap [gap - res_LB - res_cert, gap + res_cert]
(`gap_firm_lower`, `gap_firm_upper`) and the first-order point estimate gap - res_LB (`gap_point_estimate`); the section
15 form [gap - res_cert, gap + res_LB] is withdrawn and no longer written. The H class is stated on the raw gap
(`class`), on the point estimate (`class_point_estimate`) and on the firm upper end gap + res_cert (conservative,
`class_firm_upper`); P6 is read on the raw gap and, conservatively, at the firm upper ends. Rank residuals (ruling 4): the
OLTC branch's `d0_cone` is excluded from the rank residual (its WR / WI carry no cost and sit in no balance row) and
reported separately; the transformer is read on `d0_tr_cone` (the `oltc` record).
Freeze attribution (c) (spec section 14.1 option (b), section 16 ruling 4): `TN_W_frozen` freezes the TSO W columns;
`DSO_ESSO_frozen` freezes only the DSO W columns (vmag_sqr, voltage_product_real / imag), the DSO transformer U, C, D
(d0_tr_U / d0_tr_C / d0_tr_D) and every ESSO column except the section 14.2 inert chain es_avg_ch_dch_per_unit ->
es_D_per_unit, which stays FREE (section 17 ruling 1: frozen at clipped values it carried a 3.73e-7 pre-check) -- the
rest of each DSO (dispatch, SoC, copies) is free. The key name is kept from v2; its column set is the section 14.1 (b)
set with the section 17.1 exception.

Stops: an input hash mismatch (refused before anything is solved, exit 5); a containment violation on any group
(G-D0-4, exit 3, after that group's records are written); a nonconvex refusal by Gurobi (exit 3); the leaf-guard
count disagreeing (exit 3); the projected wall time exceeding 3.5 h after the first group of each kind (resumable
stop after whole groups, exit 7). The parent refuses before creating the root when the Gurobi licence does not start,
the IPOPT executable is absent, or (spec section 17 ruling 5) the executable's option listing (`ipopt --print-options`,
zero solves) does not offer ma97 among the `linear_solver` values (exit 6, nothing written); the version banner
(`ipopt -v`) and the listed linear solvers are recorded in provenance_parent.json.

Usage (one process, attached, alone):
    /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_t02.py
Resume after a 3.5 h stop (new root `<root>_resume`, refuses if it exists; the completed groups are read, hash-checked
against the stopped run's manifest, and not re-solved):
    P516_D0_T02_RESUME_FROM=data/SRP1/Results/P516D0/t02 /opt/anaconda3/envs/opf_env_py311/bin/python p516_d0_t02.py
Rehearsal (zero solves, scratch under /tmp, never evidence): P516_D0_T02_REHEARSAL_DIR=/tmp/... [and
P516_D0_T02_REHEARSAL_GROUPS=k]. The rehearsal runs everything except the solves; every "solution" in it is the
certified mapped point with zero multipliers, labelled STAND-IN.

Exit codes: 0 all gates pass; 2 a gate failed (no stop); 3 a stop fired; 4 refused (root exists); 5 refused (input hash
mismatch); 6 refused (solver unavailable); 7 resumable stop (3.5 h projection).
"""

import gc
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import time

import p516_d0_t01b as T1B

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P516D0', 't02')
_RESUME_FROM = os.environ.get('P516_D0_T02_RESUME_FROM')
_REHEARSAL = os.environ.get('P516_D0_T02_REHEARSAL_DIR')
if _REHEARSAL:
    if not os.path.abspath(_REHEARSAL).startswith(('/tmp/', '/private/tmp/')):
        raise SystemExit(f'P516_D0_T02_REHEARSAL_DIR must be under /tmp, got {_REHEARSAL!r}')
    if _RESUME_FROM:
        raise SystemExit('rehearsal and resume are exclusive')
    OUT_DIR = os.path.abspath(_REHEARSAL)
elif _RESUME_FROM:
    OUT_DIR = os.path.join(REPO, _RESUME_FROM.rstrip('/')) + '_resume'
_GROUP_LIMIT = os.environ.get('P516_D0_T02_REHEARSAL_GROUPS')
if _GROUP_LIMIT and not _REHEARSAL:
    raise SystemExit('P516_D0_T02_REHEARSAL_GROUPS is a rehearsal-only setting')
GROUP_LIMIT = int(_GROUP_LIMIT) if _GROUP_LIMIT else None
DRY = bool(_REHEARSAL)                  # the rehearsal never solves

LOG_PATH = os.path.join(OUT_DIR, 'run.log')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
PROVENANCE_PARENT = os.path.join(OUT_DIR, 'provenance_parent.json')

T01C_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01c')
T01C_MANIFEST_SHA256 = 'd29551285683ede902b71188ba6973f33e15d53018f8a77da6e5fd833514afb5'
T01B_ROOT = os.path.join('data', 'SRP1', 'Results', 'P516D0', 't01b')
T01B_MANIFEST_SHA256 = '4e5bbd45234598094593a8daca807a576590eee52f492f5281b63cf621876562'
W183E_EXPORT = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e6_export', 'e6_export.json')
W183E_EXPORT_SHA256 = '4875075728db185e2093e32e90332851d77f7d2195c576886fdafb63f83b25e0'

CELLS = T1B.CELLS
MODES = ('rated', 'available', 'calendar')
VARIANT = {'rated': 'R', 'available': 'A', 'calendar': 'C'}
VARIANT_NOTE = {
    'R': 'rated capacity on the network copies (section 3 item 3); a lower bound of production',
    'A': 'ATTRIBUTION DIAGNOSTIC, NOT a lower bound of production (spec section 11 D-1); never quoted as a bound',
    'C': 'calendar-only available energy (section 11 D-2, section 12 ruling 3); the Phase-2 object; absolute class',
}
HARNESS_VERSION = ('p516_d0_t02 v3 (P5_16_D0_SPEC_v2.md section 15 rulings 1-4 as corrected by section 16 rulings 1-5 '
                   'and section 17 rulings 1, 2, 5, 2026-11-01); v2 = the harness at ff3153ff; the section-15-only '
                   'form of v3 = sha256 7fa39c41; the section-16 form = sha256 9fa3ef1e')
RECORD_SCHEMA = {
    'version': 'v3', 'previous': 'v2 (p516_d0_t02.py at commit ff3153ff; never launched)',
    'spec_reference': ('P5_16_D0_SPEC_v2.md section 15 (rulings 1-4) as corrected by section 16 (rulings 1-5) and '
                       'section 17 (rulings 1, 2, 5)'),
    'changes_from_v2': [
        'ruling 1 (section 15) with section 16 ruling 1: per solve `resolution` {res_LB, res_cert} and `res_LB_eur` / '
        '`res_cert_eur`; per gap reading the firm interval `gap_firm_lower` = gap - res_LB - res_cert, `gap_firm_upper` '
        '= gap + res_cert, the point estimate `gap_point_estimate` = gap - res_LB (each also `_rel`); readings per cell: '
        'sums of both prices and of the three, `class` (raw gap), `class_point_estimate`, `class_firm_upper` '
        '(conservative); P6 at the firm upper ends (`held_at_firm_upper`); IPOPT gap reading. The withdrawn section 15 '
        'form [gap - res_cert, gap + res_LB] (`gap_interval_*`, `class_upper_end`, `D_interval_eur`) is not written',
        'ruling 2 (section 15) with section 16 rulings 2 and 5: IPOPT options bound_relax_factor 0, '
        'honor_original_bounds yes, bound_push 1e-10, bound_frac 1e-10, linear_solver ma97, mu_strategy adaptive; no '
        'warm_start_init_point (start = certified mapped point); `iteration0` parsed from the log',
        'section 16 ruling 3: IPOPT dimension gate on LP columns minus bound-fixed columns '
        '(`lp_dimensions.n_fixed_columns`, `lp_dimensions.vars_expected_by_ipopt`)',
        'section 16 ruling 4: `DSO_ESSO_frozen` = DSO W + transformer U, C, D + every ESSO column (section 14.1 (b))',
        'section 17 ruling 1: `DSO_ESSO_frozen` leaves the inert chain es_avg_ch_dch_per_unit / es_D_per_unit free '
        '(`zero_solve_precheck.n_esso_columns_left_free_by_family`)',
        'ruling 3: Gurobi FeasibilityTol 1e-9, OptimalityTol 1e-9',
        'section 17 ruling 2: the 24 freeze solves run with FeasibilityTol 1e-7 (`GUROBI_PARAMS_FREEZE`); every Gurobi '
        'solve records `params` and `params_set` (standard / freeze)',
        'section 17 ruling 5: preflight refuses (exit 6) unless `ipopt --print-options` lists ma97 among the '
        'linear_solver values; banner recorded in provenance_parent.json (`preflight`)',
        'ruling 4: rank_and_cycles excludes transformer branches\' d0_cone from the rank residual '
        '(`transformer_branches_excluded`, `transformer_d0_cone_excluded`)']}
GUROBI_PARAMS = {'QCPDual': 1, 'NonConvex': 0, 'BarQCPConvTol': 1e-9, 'BarConvTol': 1e-10, 'FeasibilityTol': 1e-9,
                 'OptimalityTol': 1e-9, 'Threads': 4, 'LogToConsole': 0}
# spec section 17 ruling 2: the 24 freeze solves ONLY; identical to GUROBI_PARAMS except FeasibilityTol
GUROBI_PARAMS_FREEZE = {**GUROBI_PARAMS, 'FeasibilityTol': 1e-7}
IPOPT_OPTIONS = {'tol': 1e-9, 'max_iter': 3000, 'print_level': 5, 'bound_relax_factor': 0.0,
                 'honor_original_bounds': 'yes', 'bound_push': 1e-10, 'bound_frac': 1e-10,
                 'linear_solver': 'ma97', 'mu_strategy': 'adaptive'}
RESOLUTION_DEFINITION = (
    'spec v2 section 15 ruling 1. res(x; y) = sum_i |Pi_i| viol_i(x) + sum_j |QCPi_j| viol_j(x) + sum_k |RC_k| '
    'max(lb_k - x_k, x_k - ub_k, 0); linear viol_i = |a_i x - b_i| on =, max(a_i x - b_i, 0) on <=, '
    'max(b_i - a_i x, 0) on >=; quadratic viol_j likewise on q_j(x) - d_j; lb / ub = the solve\'s own bounds (the '
    'fixed-mode and freeze bounds for those solves). Multipliers: the solver\'s own -- Gurobi Pi / QCPi / RC; IPOPT '
    'sign * Pyomo dual on rows (sign read off stationarity, `multiplier_sign_convention`) and ipopt_zL_out + '
    'ipopt_zU_out on bounds. A term is priced only where its violation is > 0 (a missing multiplier there is counted, '
    'and makes the total NaN). res_LB = res(x*; y*) at the returned point; res_cert = res(x_cert clipped into the '
    'solve\'s bounds; y*), x_cert = the T0.1c certified mapped point (the solve\'s stand-in / IPOPT start). '
    'Gap reading (section 16 ruling 1; v(delta) = Q_LB,returned <= v(0) <= Q_LB,returned + res_LB by convexity of the '
    'perturbation function): firm interval for the exact gap [gap - res_LB - res_cert, gap + res_cert], first-order '
    'point estimate gap - res_LB, conservative end gap + res_cert; the section 15 form [gap - res_cert, gap + res_LB] '
    'is withdrawn.')
DECLARED = {'gurobi_group_lp': 72, 'gurobi_fixed_mode': 12, 'gurobi_freeze': 24, 'ipopt_cross_check': 24}
DECLARED_SOLVES = sum(DECLARED.values())          # 132
PROJECTION_LIMIT_S = 3.5 * 3600.0
QCP_DUAL_FAILURE_TEXT = 'failed to compute QCP dual'
FIXED_MODE_TINY = 1e-9                            # P5.5-D4 TIE_RULE (p55d_d4_fixedmode.py)
W_FAMILIES = ('vmag_sqr', 'voltage_product_real', 'voltage_product_imag')
TR_UCD_FAMILIES = ('d0_tr_U', 'd0_tr_C', 'd0_tr_D')   # the OLTC transformer U, C, D (section 14.1 (b) freeze)
ESSO_FREE_FAMILIES = ('es_avg_ch_dch_per_unit', 'es_D_per_unit')   # section 14.2 inert chain; free (section 17.1)
STORAGE_VALUE_NOTE = ('storage value = recorded gross(x0) - recorded gross(unit), both from the cells\' '
                      'evaluation_record.json terminal_gross_operational_cost')

PREDICTIONS = {
    'P1': 'DSO blocks rank-exact <= 1e-5 at the relaxed optimum (P5.5-C): max relative cone residual over DSO branches',
    'P2': 'fixed-mode re-solve cost at the unit <= 0.05 % of group gross (P5.5-D4: 0.049 %), every group',
    'P3': 'sum gap / Q point estimate 1e-3 (H2) for variant R; H1 would falsify the mechanism reading, H3 the TN-gap '
          'reading',
    'P4': 'Gurobi returns duals on all relaxation solves (spec: 48; here the 72 group LPs, and all 108 reported)',
    'P5': 'dual consistency at the unit: -(dual_E * 1 MWh + dual_S * 0.25 MVA) has the sign of Delta Q_LB and lies '
          'within a factor 3 of Delta Q_LB = Q_LB(x0) - Q_LB(unit), per group (duals themselves negative)',
    'P6': '|sum gap(unit) - sum gap(x0)| <= the bar-sum of the two cells (gap plan-independent); point estimate <= 5 k EUR '
          'against the 259,375 EUR storage value -- read on A (mechanism) and C (decision relevance), R reported',
    'P7': 'Q_LB(unit) <= Q_LB(x0) per group to solver tolerance, on R and C',
    'P8': 'OLTC envelope slack <= 1e-5 (transformer-cone relative residual, P5.5-C8 rho_tr)',
    'P9': 'TN rank residual ~ 2.45e-2 relative (P5.5-C8), cycle-angle residual ~ 1.2e-2 rad',
    'E1': 'expert: at x = 0 the variants R, A and C coincide identically',
    'E2': 'expert: unit cell, per group gap_A <= gap_C <= gap_R',
    'E3': 'expert: unit cell, sum over 12 groups gap_R - gap_C in [10, 50] k EUR',
    'E4': 'expert: unit cell, sum over 12 groups gap_C - gap_A in [5, 40] k EUR',
    'E5': 'expert: P6 holds on A; P6 fails on R by about the degradation share (>= 15 k EUR)',
}
OPERATIONALISATION = {
    'P9_approx': 'held iff the achieved maximum lies within a factor 2 of the predicted value (Worker\'s reading of '
                 '"~"; the numbers are reported either way)',
    'P7_tolerance': 'Q_LB(unit) <= Q_LB(x0) + gap_solver(unit) + gap_solver(x0)',
    'E2_tolerance': 'each inequality with the sum of the two solves\' solver gaps as tolerance',
    'E5_reading': 'P6 holds on A iff D_A <= bar-sum; P6 fails on R by >= 15 k EUR iff D_R > bar-sum and D_R - bar-sum '
                  '>= 15,000 EUR (D_R itself reported)',
    'containment_tolerance': 'Q_LB <= Q_cert+(group) + gap_solver(that solve) (G-D0-4)',
    'agreement': '|Q_G - Q_I| / |Q_G| <= 1e-7 or |Q_G - Q_I| <= gap_G + gap_I (Gurobi vs IPOPT, variant C)',
    'H_class': 'sum_groups gap / Q_cert+(cell): H1 <= 1e-4 < H2 <= 1e-2 < H3',
    'lagrangian_check': 'g <= Q_LB <= g + 1e-6 |Q_LB| (informative in D0)',
    'resolution': RESOLUTION_DEFINITION,
    'H_class_firm': ('section 16 ruling 1: per cell sum_groups of the firm interval [gap - res_LB - res_cert, '
                     'gap + res_cert] and of the point estimate gap - res_LB, over Q_cert+(cell); the class is stated on '
                     'the raw gap (`class`, the decision rule\'s reading), on the point estimate '
                     '(`class_point_estimate`) and on the firm upper end (conservative, `class_firm_upper`); '
                     'H1 <= 1e-4 < H2 <= 1e-2 < H3'),
    'P6_firm': ('section 16 ruling 1: conservative P6 read at the firm upper ends gap + res_cert: D_firm_upper = '
                'sum_groups(gap + res_cert)(unit) - sum_groups(gap + res_cert)(x0); held iff |D_firm_upper| <= bar-sum '
                '(`held_at_firm_upper`); raw-gap verdicts unchanged. The interval-arithmetic D range '
                '[lo(unit) - hi(x0), hi(unit) - lo(x0)] of the firm intervals is reported beside it (informative, no '
                'verdict)'),
    'ipopt_dimension_gate': ('section 16 ruling 3: IPOPT-reported variables == LP columns - columns with lb == ub '
                             '(removed by IPOPT as fixed); IPOPT-reported eq + ineq == LP rows'),
    'DSO_freeze_b': ('section 14.1 option (b) / section 16 ruling 4 / section 17 ruling 1: `DSO_ESSO_frozen` fixes the '
                     'DSO W columns (vmag_sqr, voltage_product_real, voltage_product_imag), the DSO transformer '
                     'd0_tr_U / C / D and every ESSO column except es_avg_ch_dch_per_unit and es_D_per_unit (the '
                     'section 14.2 inert chain, left free) at the certified point (clipped into bounds), identified by '
                     'the name map; every other DSO column free'),
    'freeze_feasibility_tol': ('section 17 ruling 2: the 24 freeze solves (both freezes) run with Gurobi FeasibilityTol '
                               '1e-7, every other Gurobi setting as GUROBI_PARAMS; the 72 group LPs and the 12 '
                               'fixed-mode solves keep 1e-9; recorded per solve (`params`, `params_set`)'),
    'P1_transformer_exclusion': ('section 15 ruling 4: the DSO rank residual is read on non-transformer branches; the '
                                 'OLTC branch\'s d0_cone is reported separately and the transformer on d0_tr_cone (P8)'),
}

REQUIRED_CAPTURES = {
    '(a) Q_LB, gap vs Q_cert+, solver gap, status, iterations, wall, peak RSS -- per group/variant/cell': (
        't02.json', 'groups', 'gurobi_solve'),
    '(b) fixed-mode re-solve cost EUR and % of group gross (unit, C)': ('t02.json', 'groups', 'gurobi_solve'),
    '(c) two-way freeze attribution bracket (unit, C)': ('t02.json', 'groups', 'frozen_rows_check'),
    '(c) TN rank residual per branch-period and cycle-angle residual': ('t02.json', 'groups', 'rank_and_cycles'),
    '(d) DSO rank residuals': ('t02.json', 'groups', 'rank_and_cycles'),
    '(e) OLTC envelope slack': ('t02.json', 'groups', 'rank_and_cycles'),
    '(f) capacity duals vs Delta Q_LB and W183e c (unit)': ('t02.json', 'dual_consistency', 'capacity_sensitivity'),
    '(g) Lagrangian-margin dual bound per solve': ('t02.json', 'groups', 'lagrangian_bound'),
    'Gurobi vs IPOPT agreement (C)': ('t02.json', 'groups', 'ipopt_solve'),
    'separability inventory per group LP (section 13)': ('t02.json', 'separability', 'separability_inventory'),
    'readings: H class on C, P6 on A and C (R reported), R4 G and g': ('t02.json', 'readings', 'readings_fn'),
    'predictions vs outcomes': ('t02.json', 'predictions', 'readings_fn'),
    'gates': ('t02.json', 'gates', None),
    'guard counts (both guards), process launches, containment': ('t02.json', 'guard', None),
    'per-solve logs and multiplier vectors': ('t02.json', 'solve_index', '_leaf_solve'),
    'provenance and input hashes': ('t02.json', 'provenance', None),
    '(section 15.1 / 16.1) resolution res_LB and res_cert per solve, firm gap interval and point estimate': (
        't02.json', 'groups', 'resolution_price'),
    '(section 15.2) IPOPT iteration-0 inf_pr beside the start\'s LP violation': ('t02.json', 'groups',
                                                                                 'parse_ipopt_iteration0'),
    'record schema and harness version (v3)': ('t02.json', 'record_schema', None),
}
OUTPUT_FILES = ('predictions.json', 't02.json')

sha256_file = T1B.sha256_file


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


# ======================================================================================================================
#  Parent
# ======================================================================================================================
def _preflight():
    """Solver availability, before anything is created. Zero solves: the Gurobi Env is started and closed, the IPOPT
    executable is only checked for existence."""
    out = {}
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam('LogToConsole', 0)
        env.start()
        env.dispose()
        out['gurobi'] = {'ok': True, 'version': '.'.join(str(v) for v in gp.gurobi.version())}
    except Exception as error:      # noqa: BLE001 -- recorded and refused
        out['gurobi'] = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    ipopt = _load_dotenv_ipopt()
    out['ipopt'] = {'ok': bool(ipopt and os.path.exists(ipopt)), 'path': ipopt}
    out['ipopt_ma97'] = _ipopt_ma97_check(ipopt) if out['ipopt']['ok'] else {'ok': False,
                                                                             'reason': 'executable absent'}
    return out


def _ipopt_ma97_check(ipopt):
    """Spec section 17 ruling 5, zero solves: the IPOPT executable's version banner (`-v`) and its option listing
    (`--print-options`, no stub, nothing solved); ok iff the `linear_solver` entry's possible values include ma97."""
    rec = {'ok': False, 'command_banner': [ipopt, '-v'], 'command_options': [ipopt, '--print-options']}
    try:
        ver = subprocess.run([ipopt, '-v'], capture_output=True, text=True, timeout=60)
        opts = subprocess.run([ipopt, '--print-options'], capture_output=True, text=True, timeout=60)
    except Exception as error:      # noqa: BLE001 -- recorded and refused
        rec['error'] = f'{type(error).__name__}: {error}'
        return rec
    rec['banner'] = (ver.stdout + ver.stderr).strip()
    rec['banner_returncode'] = ver.returncode
    text = opts.stdout + opts.stderr
    rec['print_options_returncode'] = opts.returncode
    rec['print_options_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    m_ = re.search(r'^linear_solver\s.*?(?=^\S)', text + '\n~', re.M | re.S)
    block = m_.group(0) if m_ else ''
    rec['linear_solver_block'] = block.rstrip()
    rec['linear_solvers_listed'] = re.findall(r'^\s+-\s+(\S+)', block, re.M)
    rec['ok'] = 'ma97' in rec['linear_solvers_listed']
    return rec


def parent():
    import platform
    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
    if _REHEARSAL:
        print(f'[REHEARSAL] zero solves; outputs to scratch {OUT_DIR} (not evidence); group limit {GROUP_LIMIT}')
    else:
        pre = _preflight()
        print(f'[preflight] {json.dumps(pre)}')
        if not (pre['gurobi']['ok'] and pre['ipopt']['ok']):
            print('[REFUSED] a declared solver is unavailable; nothing created, nothing solved.')
            return 6
        if not pre['ipopt_ma97']['ok']:
            print('[REFUSED] the IPOPT executable does not list ma97 among its linear solvers (spec section 17 ruling '
                  '5); nothing created, nothing solved.')
            return 6
    if os.path.exists(OUT_DIR):
        print(f'[REFUSED] {rel(OUT_DIR)} exists; refusing to run (no overwrite, no concurrent copy).')
        return 4
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
                  'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rehearsal': bool(_REHEARSAL),
                  'resume_from': _RESUME_FROM, 'preflight': None if _REHEARSAL else pre}
    with open(PROVENANCE_PARENT, 'w') as handle:
        json.dump(provenance, handle, indent=1)
    with open(LOG_PATH, 'w', buffering=1) as log:
        log.write(f'# p516_d0_t02 parent start {provenance["started"]} interpreter {sys.executable}\n')
        child = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child'], cwd=REPO,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = child.wait()
        log.write(f'# child exit code {code}; wall {time.time() - started:.1f} s\n')
    after = {'tracked_status_after': T1B._git('status', '--porcelain', '--untracked-files=no')}
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
#  Child: audit hook (process launches only inside the leaf; writes recorded) -- installed before anything else
# ======================================================================================================================
_AUDIT = {'process_launch_events': [], 'write_opens': {}, 'fs_mutations': []}
_IN_LEAF = {'depth': 0}


def _install_audit_hook():
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def _path(arg):
        if isinstance(arg, bytes):
            return arg.decode(errors='replace')
        return arg if isinstance(arg, str) else None

    def hook(event, args):
        if event == 'open':
            path = _path(args[0]) if args else None
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or \
                      (isinstance(flags, int) and bool(flags & write_flags))
            if writing:
                path = os.path.abspath(path)
                _AUDIT['write_opens'][path] = _AUDIT['write_opens'].get(path, 0) + 1
        elif event in T1B._LAUNCH_EVENTS:
            inside = _IN_LEAF['depth'] > 0
            _AUDIT['process_launch_events'].append({'event': event, 'args': repr(args)[:300], 'inside_leaf': inside})
            if not inside:
                raise RuntimeError(f'P5.16 D0 T0.2: process launch outside the leaf ({event})')
        elif event in T1B._FS_EVENTS:
            paths = [os.path.abspath(p) for p in (_path(a) for a in args) if p]
            _AUDIT['fs_mutations'].append({'event': event, 'paths': paths})
    sys.addaudithook(hook)


class GurobiSolveBlocked(RuntimeError):
    pass


class GurobiLeafGuard:
    """The gurobipy counterpart of `SolveProfileGuard`: `gurobipy.Model.optimize` is replaced ON THE TYPE by a wrapper
    that allows the call only when the declared leaf is on the live call stack (counting it) and raises otherwise.
    Two events per solve: the leaf's own Gurobi entry (`note_entry`) and the optimize call itself."""

    def __init__(self, permitted, label):
        self.permitted = tuple(permitted)
        self.label = label
        self.counts = {'permitted_entry': 0, 'permitted_optimize': 0, 'blocked_optimize': 0}
        self._gp = None
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
        import gurobipy as gp
        self._gp = gp
        self._original = gp.Model.optimize
        guard = self

        def guarded_optimize(model_self, *args, **kwargs):
            if not guard._site():
                guard.counts['blocked_optimize'] += 1
                raise GurobiSolveBlocked(f'gurobipy.Model.optimize from an undeclared call site; {guard.label} '
                                         f'permits only {guard.permitted}')
            guard.counts['permitted_optimize'] += 1
            return guard._original(model_self, *args, **kwargs)
        gp.Model.optimize = guarded_optimize
        if gp.Model.optimize is not guarded_optimize:
            raise RuntimeError('GurobiLeafGuard: the type-level patch did not take')
        return self

    def note_entry(self):
        if not self._site():
            raise GurobiSolveBlocked('leaf entry noted outside the leaf')
        self.counts['permitted_entry'] += 1

    def uninstall(self):
        if self._gp is not None and self._original is not None:
            self._gp.Model.optimize = self._original

    def verify(self, expected):
        failures = []
        if self.counts['permitted_entry'] != expected:
            failures.append(f'Gurobi leaf entries {self.counts["permitted_entry"]} != declared {expected}')
        if self.counts['permitted_optimize'] != expected:
            failures.append(f'Gurobi optimize calls {self.counts["permitted_optimize"]} != declared {expected}')
        if self.counts['blocked_optimize']:
            failures.append('a blocked Gurobi call site was reached')
        return failures


_STATE = {}          # guards, Gurobi Env, IPOPT path: set by the child before the first solve


# ======================================================================================================================
#  THE leaf: the only function from which a solver is reached. It names gurobipy, the Pyomo solver factory and numpy;
#  nothing else (asserted statically before the first solve).
# ======================================================================================================================
def _leaf_solve(kind, payload):
    _IN_LEAF['depth'] += 1
    try:
        if kind == 'gurobi':
            gp = _STATE['gp']
            _STATE['gurobi_guard'].note_entry()
            model = gp.read(payload['lp_path'], _STATE['env'])
            if payload.get('lb') is not None:
                big = gp.GRB.INFINITY
                model.setAttr('LB', model.getVars(), np.where(np.isinf(payload['lb']), -big, payload['lb']).tolist())
                model.setAttr('UB', model.getVars(), np.where(np.isinf(payload['ub']), big, payload['ub']).tolist())
            for key, value in payload['params'].items():
                model.setParam(key, value)
            model.setParam('LogFile', payload['log_path'])
            t0 = time.time()
            error = None
            try:
                model.optimize()
            except gp.GurobiError as exc:
                error = {'errno': getattr(exc, 'errno', None), 'message': str(exc)}
            wall = time.time() - t0
            out = {'wall_s': wall, 'error': error}
            for attr in ('Status', 'ObjVal', 'ObjBound', 'BarIterCount', 'Runtime', 'NumVars', 'NumConstrs',
                         'NumQConstrs', 'NumNZs', 'NumQCNZs'):
                try:
                    out[attr] = model.getAttr(attr)
                except Exception:   # noqa: BLE001 -- an unavailable attribute is recorded as None
                    out[attr] = None
            for attr, getter in (('X', model.getVars), ('RC', model.getVars), ('Pi', model.getConstrs),
                                 ('QCPi', model.getQConstrs)):
                try:
                    out[attr] = np.asarray(model.getAttr(attr, getter()), dtype=float)
                except Exception as exc:    # noqa: BLE001
                    out[attr] = None
                    out[f'{attr}_error'] = f'{type(exc).__name__}: {exc}'
            model.dispose()
            return out
        if kind == 'ipopt':
            solver = _STATE['solver_factory']('ipopt', executable=_STATE['ipopt_path'])
            for key, value in payload['options'].items():
                solver.options[key] = value
            t0 = time.time()
            results = solver.solve(payload['model'], tee=False, load_solutions=True, logfile=payload['log_path'])
            return {'wall_s': time.time() - t0, 'results': results}
        raise ValueError(kind)
    finally:
        _IN_LEAF['depth'] -= 1


np = None            # bound in the child after the guards are armed (the leaf uses it)


# ======================================================================================================================
#  LP data (read with gurobipy; never optimized here)
# ======================================================================================================================
def parse_pyomo_name(name):
    """'dso_7.vmag_sqr[0,0,0,5]' -> ('dso_7', 'vmag_sqr', (0, 0, 0, 5)); top-level rows have block None."""
    if '.' in name.split('[', 1)[0]:
        block, rest = name.split('.', 1)
    else:
        block, rest = None, name
    if '[' in rest:
        fam, idx = rest.split('[', 1)
        parts = idx.rstrip(']').split(',')
        index = tuple(int(p) if re.fullmatch(r'-?\d+', p) else p for p in parts)
    else:
        fam, index = rest, ()
    return block, fam, index


def block_key(block):
    if block == 'tso':
        return 'TSO'
    if block.startswith('dso_'):
        return 'DSO' + block[4:]
    if block.startswith('esso_'):
        return 'ESSO' + block[5:]
    raise KeyError(block)


def read_lp_data(lp_path, name_map, env):
    """Every datum the post-processing needs, from the exported LP read with gurobipy (no optimize)."""
    import gurobipy as gp
    import scipy.sparse as sp
    m = gp.read(lp_path, env)
    vars_ = m.getVars()
    cons = m.getConstrs()
    qcs = m.getQConstrs()
    data = {
        'var_symbols': m.getAttr('VarName', vars_), 'lb': np.asarray(m.getAttr('LB', vars_), dtype=float),
        'ub': np.asarray(m.getAttr('UB', vars_), dtype=float), 'c': np.asarray(m.getAttr('Obj', vars_), dtype=float),
        'grb_infinity': float(gp.GRB.INFINITY),
        'obj_con': float(m.ObjCon), 'sense_obj': int(m.ModelSense),
        'con_symbols': m.getAttr('ConstrName', cons), 'rhs': np.asarray(m.getAttr('RHS', cons), dtype=float),
        'sense': np.asarray(m.getAttr('Sense', cons)), 'A': sp.csr_matrix(m.getA()),
        'qc_symbols': m.getAttr('QCName', qcs), 'qc_rhs': np.asarray(m.getAttr('QCRHS', qcs), dtype=float),
        'qc_sense': np.asarray(m.getAttr('QCSense', qcs)),
        'vtype': set(m.getAttr('VType', vars_)),
    }
    index = {v.index: k for k, v in enumerate(vars_)}
    qrows = []
    for q in qcs:
        qe = m.getQCRow(q)
        quad = [(index[qe.getVar1(i).index], index[qe.getVar2(i).index], float(qe.getCoeff(i)))
                for i in range(qe.size())]
        le = qe.getLinExpr()
        lin = [(index[le.getVar(i).index], float(le.getCoeff(i))) for i in range(le.size())]
        qrows.append({'quad': quad, 'lin': lin, 'lin_const': float(le.getConstant())})
    data['qrows'] = qrows
    data['size'] = {'NumVars': m.NumVars, 'NumConstrs': m.NumConstrs, 'NumQConstrs': m.NumQConstrs,
                    'NumNZs': m.NumNZs, 'NumQCNZs': m.NumQCNZs, 'NumQNZs': m.NumQNZs}
    m.dispose()
    data['lb'][data['lb'] <= -data['grb_infinity']] = -np.inf      # Gurobi writes an infinite bound as +-1e100
    data['ub'][data['ub'] >= data['grb_infinity']] = np.inf
    vmap, cmap = name_map['variables'], name_map['constraints']
    data['var_names'] = [vmap[s] for s in data['var_symbols']]
    data['con_names'] = [cmap.get(s) for s in data['con_symbols']]
    data['qc_names'] = [cmap.get(s) for s in data['qc_symbols']]
    data['col'] = {n: k for k, n in enumerate(data['var_names'])}
    data['parsed'] = [parse_pyomo_name(n) for n in data['var_names']]
    return data


# ======================================================================================================================
#  Certified (mapped) point on the LP's columns
# ======================================================================================================================
def branch_nodes(network, b):
    branch = network.branches[b]
    return network.get_node_idx(branch.fbus), network.get_node_idx(branch.tbus)


def certified_point(data, vectors, vectors_esso, networks, esso_factor):
    """x_cert on the LP's columns: every column from the T0.1b certified Var vector, except the W families
    (rank-1 image of the certified e, f -- T0.1c's mapped point 'ef'), the transformer U, C, D (r^2 Wii(from), r WijR,
    r WijI at that image, r certified) and the ESSO available energy (factor * rated, the variant's row)."""
    x = np.full(len(data['var_names']), np.nan)
    for k, (block, fam, idx) in enumerate(data['parsed']):
        key = block_key(block)
        src = vectors[key]['vars'] if key in vectors else vectors_esso[key]['vars']
        net = networks.get(key)
        if net is not None and fam == 'vmag_sqr':
            e = src[f'e[{",".join(map(str, idx))}]'][0]
            f = src[f'f[{",".join(map(str, idx))}]'][0]
            x[k] = e * e + f * f
        elif net is not None and fam in ('voltage_product_real', 'voltage_product_imag', 'd0_tr_U', 'd0_tr_C',
                                         'd0_tr_D'):
            b, s_m, s_o, p = idx
            fn, tn = branch_nodes(net, b)

            def ef(node):
                key_ = f'{node},{s_m},{s_o},{p}'
                return src[f'e[{key_}]'][0], src[f'f[{key_}]'][0]
            (e_f, f_f), (e_t, f_t) = ef(fn), ef(tn)
            wr, wi = e_f * e_t + f_f * f_t, f_f * e_t - e_f * f_t
            r = src[f'r[{b},{s_m},{s_o},{p}]'][0] if fam.startswith('d0_tr') else None
            x[k] = {'voltage_product_real': lambda: wr, 'voltage_product_imag': lambda: wi,
                    'd0_tr_U': lambda: r * r * (e_f * e_f + f_f * f_f), 'd0_tr_C': lambda: r * wr,
                    'd0_tr_D': lambda: r * wi}[fam]()
        elif key.startswith('ESSO') and fam == 'es_e_available_per_unit':
            rated = src[f'es_e_rated_per_unit[{idx[0]},{idx[1]}]'][0]
            x[k] = rated * esso_factor(idx)
        else:
            name = f'{fam}[{",".join(map(str, idx))}]' if idx else fam
            x[k] = src[name][0]
    return x


# ======================================================================================================================
#  Separability inventory (section 13): no ESSO row of a group LP couples different (year, day) once pinned columns
#  are constants and inert definitional chains are eliminated
# ======================================================================================================================
_ESSO_YD = {'es_pnet': (0, 1), 'es_qnet': (0, 1), 'es_pch_per_unit': (1, 2), 'es_pdch_per_unit': (1, 2)}


def separability_inventory(data):
    A = data['A'].tocsc()
    Ar = data['A']
    n = len(data['var_names'])
    esso_cols = [k for k, (b, _f, _i) in enumerate(data['parsed']) if b and b.startswith('esso_')]
    esso_set = set(esso_cols)
    label = {}
    for k in esso_cols:
        b, fam, idx = data['parsed'][k]
        label[k] = (b, idx[_ESSO_YD[fam][0]], idx[_ESSO_YD[fam][1]]) if fam in _ESSO_YD else None
    rows_lin = sorted({int(i) for k in esso_cols for i in A.indices[A.indptr[k]:A.indptr[k + 1]]})
    rows_q = [j for j, q in enumerate(data['qrows'])
              if any(a in esso_set or b_ in esso_set for a, b_, _c in q['quad']) or
              any(a in esso_set for a, _c in q['lin'])]

    def lin_row(i):
        lo, hi = Ar.indptr[i], Ar.indptr[i + 1]
        return list(zip(Ar.indices[lo:hi].tolist(), Ar.data[lo:hi].tolist()))
    row_cols = {('L', i): lin_row(i) for i in rows_lin}
    for j in rows_q:
        q = data['qrows'][j]
        cols = {a for a, _b, _c in q['quad']} | {b_ for _a, b_, _c in q['quad']} | {a for a, _c in q['lin']}
        row_cols[('Q', j)] = [(c, None) for c in sorted(cols)]
    # 1. pinned columns: LB == UB, or the only non-pinned column of a linear equality row (iterate to a fixpoint)
    pinned = {k for k in esso_cols if data['lb'][k] == data['ub'][k]}
    changed = True
    while changed:
        changed = False
        for (kind, i), cols in row_cols.items():
            if kind != 'L' or data['sense'][i] != '=':
                continue
            free = [c for c, a in cols if c not in pinned and a != 0.0]
            if len(free) == 1 and free[0] in esso_set:
                pinned.add(free[0])
                changed = True
    # 2. inert definitional chains: an unlabelled, unpinned column in exactly one remaining equality row whose image
    #    (b - rest) / a, over the interval of the rest, lies inside the column's OWN bounds -> drop row and column.
    #    The interval of the rest uses bounds tightened by equality rows still present (sound: the tightening row is
    #    in the system when it is used).
    removed_rows, removed_cols = set(), set()
    interval = {k: (data['lb'][k], data['ub'][k]) for k in range(n)}

    def rest_interval(cols, skip):
        lo = hi = 0.0
        for c, a in cols:
            if c == skip or c in pinned or a == 0.0:
                continue
            l_, u_ = interval[c]
            lo += min(a * l_, a * u_)
            hi += max(a * l_, a * u_)
        return lo, hi

    def image(b, lo, hi, a):
        return sorted(((b - hi) / a, (b - lo) / a))

    for _ in range(4):
        for key, cols in row_cols.items():          # tighten unlabelled columns from their defining rows
            if key in removed_rows or key[0] != 'L' or data['sense'][key[1]] != '=':
                continue
            unl = [(c, a) for c, a in cols if c in esso_set and label.get(c) is None and c not in pinned]
            if len(unl) != 1:
                continue
            c, a = unl[0]
            lo, hi = rest_interval(cols, c)
            b = data['rhs'][key[1]]
            cand = image(b, lo, hi, a)
            interval[c] = (max(interval[c][0], cand[0]), min(interval[c][1], cand[1]))
        progress = True
        while progress:
            progress = False
            occ = {}
            for key, cols in row_cols.items():
                if key in removed_rows:
                    continue
                for c, _a in cols:
                    if c in esso_set and label.get(c) is None and c not in pinned and c not in removed_cols:
                        occ.setdefault(c, []).append(key)
            for c, keys in occ.items():
                if len(keys) != 1 or keys[0][0] != 'L' or data['sense'][keys[0][1]] != '=':
                    continue
                key = keys[0]
                cols = row_cols[key]
                a = dict(cols)[c]
                lo, hi = rest_interval(cols, c)
                b = data['rhs'][key[1]]
                img = image(b, lo, hi, a)
                if data['lb'][c] <= img[0] and img[1] <= data['ub'][c]:
                    removed_rows.add(key)
                    removed_cols.add(c)
                    progress = True
    # 3. verdict per remaining row
    literal, coupling_after, unlabelled_after = [], [], []
    for key, cols in row_cols.items():
        labs = {label[c] for c, _a in cols if c in esso_set and c not in pinned and label.get(c) is not None}
        unl = [c for c, _a in cols if c in esso_set and c not in pinned and label.get(c) is None]
        name = data['con_names'][key[1]] if key[0] == 'L' else data['qc_names'][key[1]]
        yd_all = {label[c] for c, _a in cols if c in esso_set and label.get(c) is not None}
        if len(yd_all) > 1:
            literal.append(name)
        if key in removed_rows:
            continue
        if len(labs) > 1:
            coupling_after.append(name)
        if unl:
            unlabelled_after.append({'row': name, 'columns': [data['var_names'][c] for c in unl]})
    return {'n_esso_columns': len(esso_cols), 'n_esso_rows': len(row_cols), 'n_pinned_columns': len(pinned),
            'rows_coupling_yd_literally': literal, 'n_rows_coupling_yd_literally': len(literal),
            'inert_rows_eliminated': sorted(data['con_names'][k[1]] for k in removed_rows),
            'inert_columns_eliminated': sorted(data['var_names'][c] for c in removed_cols),
            'rows_coupling_yd_after_elimination': coupling_after,
            'rows_with_unpinned_unlabelled_columns_after': unlabelled_after,
            'separable': not coupling_after and not unlabelled_after,
            'method': ('ESSO columns labelled by (node, year, day) from their index (es_pnet / es_qnet [y, d, p]; '
                       'es_pch / es_pdch_per_unit [y_inv, y, d, p]); year-level columns are PINNED when LB == UB or '
                       'when they are the only non-pinned column of a linear equality row (fixpoint); an unlabelled '
                       'unpinned column occurring in exactly one equality row is eliminated with that row when the '
                       'row\'s image of it over the interval of its other columns lies inside its own bounds '
                       '(interval arithmetic; tightening only from rows still present). Separable iff no remaining '
                       'row carries two (node, year, day) labels or an unpinned unlabelled column.')}


# ======================================================================================================================
#  Measurements on a primal-dual point
# ======================================================================================================================
def quadratic_value_and_grad(q, x):
    val = q['lin_const']
    grad = {}
    for i, j, c in q['quad']:
        val += c * x[i] * x[j]
        grad[i] = grad.get(i, 0.0) + c * x[j]
        grad[j] = grad.get(j, 0.0) + c * x[i]
    for i, c in q['lin']:
        val += c * x[i]
        grad[i] = grad.get(i, 0.0) + c
    return val, grad


def _classify_qrow(q, x):
    """'psd' (convex quadratic function) or 'hyperbolic' (one negative eigenvalue, no linear part: an SOC set written
    as x'Qx <= 0) with the factors (P, n) of x'Qx = |Px|^2 - (n'x)^2 oriented so that n'x* >= 0."""
    cols = sorted({i for i, _j, _c in q['quad']} | {j for _i, j, _c in q['quad']})
    pos = {c: k for k, c in enumerate(cols)}
    Q = np.zeros((len(cols), len(cols)))
    for i, j, c in q['quad']:
        if i == j:
            Q[pos[i], pos[i]] += c
        else:
            Q[pos[i], pos[j]] += 0.5 * c
            Q[pos[j], pos[i]] += 0.5 * c
    w, V = np.linalg.eigh(Q)
    tol = 1e-12 * max(1.0, float(np.abs(w).max()))
    if w.min() >= -tol:
        return {'kind': 'psd', 'cols': cols}
    if (w < -tol).sum() == 1 and not q['lin'] and q['lin_const'] == 0.0:
        k_neg = int(np.argmin(w))
        n_vec = math.sqrt(-w[k_neg]) * V[:, k_neg]
        P = np.array([math.sqrt(w[k]) * V[:, k] for k in range(len(w)) if k != k_neg and w[k] > tol])
        xs = np.array([x[c] for c in cols])
        if n_vec @ xs < 0:
            n_vec = -n_vec
        return {'kind': 'hyperbolic', 'cols': cols, 'P': P, 'n': n_vec}
    return {'kind': 'nonconvex', 'cols': cols, 'eigs': w.tolist()}


def lagrangian_bound(data, x, pi, qcpi, rc_solver, q_lb, bounds=None):
    """(g) Lagrangian-margin dual bound at (x*, y). Gurobi convention for the multipliers (dQ/drhs): the rows-only
    Lagrangian is c'x - sum pi_i (a_i x - b_i) - sum lambda_j (q_j(x) - d_j); a <= row needs pi <= 0 and a >= row
    pi >= 0 (wrong-sign multipliers are projected to 0). Every quadratic row is written as a CONVEX function before
    the bound is taken: PSD rows as they are; hyperbolic rows (rotated / Lorentz cones written as x'Qx <= 0) as
    |Px| - n'x <= 0 with multiplier |lambda| (|Px*| + n'x*) (the exact gradient match at an active cone).
      g_tight = L_conv(x*, y) + sum_k min(r_k (lb_k - x*_k), r_k (ub_k - x*_k)),   r = grad_x L_conv(x*, y)
      g_spec  = L(x*, y, z) - sum_k |r_k - z_k| w_k   (spec G-D0-5 form; z = bound multipliers from the solver's own
                reduced costs, sign-projected; w_k = ub_k - lb_k)
    Both are valid lower bounds of the relaxation when finite; an unbounded variable with r_k != 0 makes them -inf."""
    lb = data['lb'] if bounds is None else bounds[0]
    ub = data['ub'] if bounds is None else bounds[1]
    c = data['c']
    A = data['A']
    sense = data['sense']
    rhs = data['rhs']
    out = {}
    if pi is None or (data['qrows'] and qcpi is None):
        return {'available': False, 'reason': 'dual-less solve (no Pi / QCPi)'}
    pi_p = pi.copy()
    wrong_lin = int(np.sum((sense == '<') & (pi_p > 0)) + np.sum((sense == '>') & (pi_p < 0)))
    pi_p[(sense == '<') & (pi_p > 0)] = 0.0
    pi_p[(sense == '>') & (pi_p < 0)] = 0.0
    ax_b = A @ x - rhs
    L = float(c @ x) + data['obj_con'] - float(pi_p @ ax_b)
    r = c - A.T @ pi_p
    r_raw = c - A.T @ pi
    kinds = {'psd': 0, 'hyperbolic': 0, 'nonconvex': 0}
    wrong_q = 0
    for j, q in enumerate(data['qrows']):
        lam = float(qcpi[j])
        s = data['qc_sense'][j]
        qv, qg = quadratic_value_and_grad(q, x)
        for i, gval in qg.items():
            r_raw[i] -= lam * gval
        if (s == '<' and lam > 0) or (s == '>' and lam < 0):
            wrong_q += 1
            lam = 0.0
        cls = _classify_qrow(q, x)
        kinds[cls['kind']] += 1
        if cls['kind'] == 'psd' or lam == 0.0:
            L -= lam * (qv - data['qc_rhs'][j])
            for i, gval in qg.items():
                r[i] -= lam * gval
        elif cls['kind'] == 'hyperbolic' and s == '<' and data['qc_rhs'][j] == 0.0:
            xs = np.array([x[cc] for cc in cls['cols']])
            Px = cls['P'] @ xs if len(cls['P']) else np.zeros(0)
            npx = float(np.linalg.norm(Px))
            nx = float(cls['n'] @ xs)
            y_s = -lam * (npx + nx)                   # lam <= 0 for a <= row -> y_s >= 0
            L += y_s * (npx - nx)
            grad = (cls['P'].T @ Px / npx if npx > 0 else np.zeros(len(xs))) - cls['n']
            for k_, cc in enumerate(cls['cols']):
                r[cc] += y_s * grad[k_]
        else:
            return {'available': False, 'reason': f'quadratic row {data["qc_names"][j]} is not convex-representable',
                    'class': cls}
    rc_match = None
    if rc_solver is not None:
        rc_match = {'max_abs_recomputed_minus_solver_rc': float(np.max(np.abs(r_raw - rc_solver))),
                    'max_abs_solver_rc': float(np.max(np.abs(rc_solver)))}
    finite_lb, finite_ub = np.isfinite(lb), np.isfinite(ub)
    lo_term = np.where(r > 0, np.where(finite_lb, r * (np.where(finite_lb, lb, 0.0) - x), -np.inf), 0.0)
    hi_term = np.where(r < 0, np.where(finite_ub, r * (np.where(finite_ub, ub, 0.0) - x), -np.inf), 0.0)
    box_min = lo_term + hi_term
    g_tight = L + float(np.sum(box_min))
    n_unbounded_nonzero = int(np.sum(np.isinf(box_min)))
    out.update({'available': True, 'L_conv': L, 'g_tight': g_tight, 'q_lb': q_lb,
                'n_unbounded_vars_with_nonzero_gradient': n_unbounded_nonzero,
                'max_abs_gradient_on_unbounded_vars': float(np.max(np.abs(r[~(finite_lb & finite_ub)]), initial=0.0)),
                'n_wrong_sign_linear_multipliers_projected': wrong_lin, 'n_wrong_sign_quadratic_multipliers_projected':
                    wrong_q, 'quadratic_row_classes': kinds, 'recomputed_rc_vs_solver_rc': rc_match})
    if rc_solver is not None:
        z_lb = np.maximum(rc_solver, 0.0)
        z_ub = np.maximum(-rc_solver, 0.0)
        bterm = np.zeros(len(x))
        m1, m2 = z_lb > 0, z_ub > 0
        bterm[m1] = z_lb[m1] * (lb[m1] - x[m1])          # -inf when the bound is infinite (z > 0)
        bterm[m2] += z_ub[m2] * (x[m2] - ub[m2])
        resid = r - z_lb + z_ub
        width = ub - lb
        margin = np.zeros(len(x))
        m3 = resid != 0.0
        margin[m3] = np.abs(resid[m3]) * width[m3]       # +inf on an unbounded variable with a nonzero residual
        g_spec = L + float(np.sum(bterm)) - float(np.sum(margin))
        out.update({'g_spec': g_spec, 'max_abs_stationarity_residual': float(np.max(np.abs(resid))),
                    'sum_margin_finite_part': float(np.sum(margin[np.isfinite(margin)]))})
    for key in ('g_tight', 'g_spec'):
        g = out.get(key)
        if g is not None:
            out[f'{key}_check'] = {'g_le_QLB': bool(g <= q_lb),
                                   'QLB_le_g_plus_1e-6': bool(q_lb <= g + 1e-6 * abs(q_lb)),
                                   'held': bool(g <= q_lb <= g + 1e-6 * abs(q_lb)),
                                   'Q_LB_minus_g': (q_lb - g) if math.isfinite(g) else None}
    return out


def _finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def gap_firm_reading(gap, res_lb, res_cert, q_plus):
    """Spec section 16 ruling 1 (correcting section 15.1): the firm interval for the exact gap
    [gap - res_LB - res_cert, gap + res_cert] and the first-order point estimate gap - res_LB, in EUR and relative to
    Q_cert+(group); every field None when the gap or either price is unavailable (dual-less solve, missing multiplier,
    no point). The withdrawn section 15 form [gap - res_cert, gap + res_LB] is not produced."""
    keys = ('gap_firm_lower', 'gap_firm_upper', 'gap_point_estimate')
    if not (_finite(gap) and _finite(res_lb) and _finite(res_cert)):
        return {**{k: None for k in keys}, **{f'{k}_rel': None for k in keys}}
    out = {'gap_firm_lower': gap - res_lb - res_cert, 'gap_firm_upper': gap + res_cert, 'gap_point_estimate': gap - res_lb}
    out.update({f'{k}_rel': out[k] / q_plus for k in keys})
    return out


def _row_family(name):
    block, fam, _idx = parse_pyomo_name(name) if name else (None, str(name), ())
    return f'{(block or "top").split("_")[0]}.{fam}'


def resolution_price(data, x, pi, qcpi, rc, bounds=None, top=6):
    """Spec section 15 ruling 1: the price of a point's violations at a solve's multipliers (RESOLUTION_DEFINITION).
    Called with the returned point (res_LB) and with the certified mapped point clipped into the solve's bounds
    (res_cert); the multipliers are the solve's own in both calls."""
    lb = data['lb'] if bounds is None else bounds[0]
    ub = data['ub'] if bounds is None else bounds[1]
    if x is None or np.any(np.isnan(x)):
        return {'available': False, 'reason': 'no point (X missing or NaN)'}
    if pi is None or rc is None or (len(data['qrows']) and qcpi is None):
        return {'available': False, 'reason': 'dual-less solve (Pi, QCPi or RC missing)'}
    r = data['A'] @ x - data['rhs']
    s = data['sense']
    lin = np.where(s == '=', np.abs(r), np.where(s == '<', np.maximum(r, 0.0), np.maximum(-r, 0.0)))
    qd = np.zeros(len(data['qrows']))
    for j, q in enumerate(data['qrows']):
        v = q['lin_const']
        for i_, k_, c_ in q['quad']:
            v += c_ * x[i_] * x[k_]
        for i_, c_ in q['lin']:
            v += c_ * x[i_]
        qd[j] = v - data['qc_rhs'][j]
    qs = data['qc_sense']
    quad = np.where(qs == '<', np.maximum(qd, 0.0), np.where(qs == '>', np.maximum(-qd, 0.0), np.abs(qd)))
    bnd = np.maximum(np.maximum(np.where(np.isfinite(lb), lb - x, 0.0), np.where(np.isfinite(ub), x - ub, 0.0)), 0.0)
    parts, out = {}, {'available': True}
    fams = {}
    n_nan = 0
    for key, viol, mult, names in (('linear', lin, pi, data['con_names']), ('quadratic', quad, qcpi, data['qc_names']),
                                   ('bounds', bnd, rc, data['var_names'])):
        mult = np.zeros(len(viol)) if mult is None else np.asarray(mult, dtype=float)
        on = viol > 0.0
        price = np.zeros(len(viol))
        price[on] = np.abs(mult[on]) * viol[on]
        n_nan += int(np.sum(np.isnan(price)))
        parts[key] = float(np.sum(price))
        out[f'{key}_max_violation'] = float(np.max(viol, initial=0.0))
        out[f'{key}_n_violated'] = int(np.sum(on))
        out[f'{key}_max_abs_multiplier_on_violated'] = float(np.max(np.abs(mult[on]), initial=0.0)) if on.any() else 0.0
        for k in np.nonzero(on & (price > 0.0))[0]:
            f_ = _row_family(names[k]) if key != 'bounds' else \
                f'{(data["parsed"][k][0] or "top").split("_")[0]}.{data["parsed"][k][1]}'
            fams[(key, f_)] = fams.get((key, f_), 0.0) + float(price[k])
    out.update({'total_eur': parts['linear'] + parts['quadratic'] + parts['bounds'],
                'linear_eur': parts['linear'], 'quadratic_eur': parts['quadratic'], 'bounds_eur': parts['bounds'],
                'n_nan_multipliers_on_violated': n_nan,
                'top_families_eur': [[k_, f_, v_] for (k_, f_), v_ in sorted(fams.items(), key=lambda t: -t[1])[:top]]})
    return out


_IPOPT_ITER0_RE = re.compile(r'^[ \t]*0[ \t]+([-+]?\d\.\d+e[-+]\d+)[ \t]+(\d\.\d+e[-+]\d+)[ \t]+(\d\.\d+e[-+]\d+)'
                             r'[ \t]+(-?[ \t]*\d+\.\d|-[ \t]*inf)', re.M)


def parse_ipopt_iteration0(path):
    """Spec section 15 ruling 2: IPOPT's iteration-0 line (objective, inf_pr, inf_du, lg(mu)) -- the first non-
    restoration line with iteration number 0 -- to show whether the warm start kept the certified point."""
    if not os.path.exists(path):
        return {'present': False}
    text = open(path, errors='replace').read()
    m_ = _IPOPT_ITER0_RE.search(text)
    if not m_:
        return {'present': True, 'found': False}
    lg_mu = m_.group(4).replace(' ', '')
    return {'present': True, 'found': True, 'objective': float(m_.group(1)), 'inf_pr': float(m_.group(2)),
            'inf_du': float(m_.group(3)), 'lg_mu': None if 'inf' in lg_mu else float(lg_mu),
            'note': 'IPOPT inf_pr is the max-norm of the (internally scaled) row residuals including the slacks'}


def rank_and_cycles(data, x, networks, cycle_basis_fn):
    """(c) TN rank residual per branch-period and cycle-angle residual, (d) DSO rank residuals, (e) OLTC envelope
    slack (transformer-cone relative residual = P5.5-C8 rho_tr; plus the U-envelope bound slacks)."""
    col = data['col']
    out = {}
    for block, net in networks.items():
        prefix = {'TSO': 'tso'}.get(block, 'dso_' + block[3:] if block.startswith('DSO') else None)
        rows = []
        periods = sorted({idx[3] for (b_, fam, idx) in data['parsed'] if b_ == prefix and fam == 'vmag_sqr'})
        for b, br in enumerate(net.branches):
            if not br.status:
                continue
            fn, tn = branch_nodes(net, b)
            for p in periods:
                kw = col.get(f'{prefix}.voltage_product_real[{b},0,0,{p}]')
                if kw is None:
                    continue
                wii = x[col[f'{prefix}.vmag_sqr[{fn},0,0,{p}]']]
                wjj = x[col[f'{prefix}.vmag_sqr[{tn},0,0,{p}]']]
                wr = x[kw]
                wi = x[col[f'{prefix}.voltage_product_imag[{b},0,0,{p}]']]
                rows.append({'branch': b, 'period': p, 'rel': (wii * wjj - wr * wr - wi * wi) / max(wii * wjj, 1e-12),
                             'angle': math.atan2(wi, wr), 'from': fn})
        # spec section 15 ruling 4: the OLTC branch's d0_cone (WR / WI inert: no cost, no balance row) is excluded from
        # the rank residual and reported separately; the transformer is read on d0_tr_cone (the 'oltc' record below)
        tr_flagged = {b for b, br in enumerate(net.branches) if br.is_transformer and br.status}
        tr_branches = {b for b in tr_flagged if any(f'{prefix}.d0_tr_U[{b},0,0,{p}]' in col for p in periods)}
        rank_rows = [r for r in rows if r['branch'] not in tr_branches]
        tr_d0 = [r for r in rows if r['branch'] in tr_branches]
        rec = {'n_branch_periods': len(rank_rows),
               'max_rel_rank_residual': max((r['rel'] for r in rank_rows), default=None),
               'min_rel_rank_residual': min((r['rel'] for r in rank_rows), default=None),
               'per_branch_period': [[r['branch'], r['period'], r['rel']] for r in rank_rows],
               'transformer_branches_excluded': sorted({r['branch'] for r in tr_d0}),
               'transformer_flagged_without_oltc_columns_kept': sorted(tr_flagged - tr_branches),
               'transformer_d0_cone_excluded': {
                   'n': len(tr_d0), 'max_rel': max((r['rel'] for r in tr_d0), default=None),
                   'min_rel': min((r['rel'] for r in tr_d0), default=None),
                   'per_branch_period': [[r['branch'], r['period'], r['rel']] for r in tr_d0],
                   'note': ('spec v2 section 15 ruling 4: not a rank reading -- the OLTC branch\'s WR / WI carry no cost '
                            'and sit in no balance row; the live transformer cone is d0_tr_cone (`oltc`)')}}
        cycles = cycle_basis_fn(net, [b for b, br in enumerate(net.branches) if br.status])
        if cycles:
            ang = {(r['branch'], r['period']): (r['angle'], r['from']) for r in rows}
            per = []
            for ci, cyc in enumerate(cycles):
                for p in periods:
                    total = 0.0
                    for b, head, _tail in cyc:
                        a, fnode = ang[(b, p)]
                        total += a if head == fnode else -a
                    per.append([ci, p, abs((total + math.pi) % (2 * math.pi) - math.pi)])
            rec['n_cycles'] = len(cycles)
            rec['max_cycle_angle_residual_rad'] = max(v for _c, _p, v in per)
            rec['per_cycle_period'] = per
        trs = [b for b, br in enumerate(net.branches) if br.is_transformer and br.status]
        if trs:
            tr_rows = []
            for b in trs:
                fn, tn = branch_nodes(net, b)
                for p in periods:
                    k_u = col.get(f'{prefix}.d0_tr_U[{b},0,0,{p}]')
                    if k_u is None:
                        continue
                    u = x[k_u]
                    cc = x[col[f'{prefix}.d0_tr_C[{b},0,0,{p}]']]
                    dd = x[col[f'{prefix}.d0_tr_D[{b},0,0,{p}]']]
                    wii = x[col[f'{prefix}.vmag_sqr[{fn},0,0,{p}]']]
                    wjj = x[col[f'{prefix}.vmag_sqr[{tn},0,0,{p}]']]
                    tr_rows.append([b, p, (u * wjj - cc * cc - dd * dd) / max(u * wjj, 1e-12), u / wii])
            rec['oltc'] = {'n': len(tr_rows), 'max_transformer_cone_rel_residual': max(r[2] for r in tr_rows),
                           'u_over_wii_range': [min(r[3] for r in tr_rows), max(r[3] for r in tr_rows)],
                           'per_branch_period': tr_rows}
        out[block] = rec
    return out


# ======================================================================================================================
#  Child
# ======================================================================================================================
def child():
    _install_audit_hook()
    sys.path.insert(0, REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    leaf_site = (('p516_d0_t02.py', '_leaf_solve'),)
    declared_ipopt = 0 if DRY else None         # set after the resume set is known
    pyomo_guard = SolveProfileGuard(permitted=leaf_site, label='P5.16 D0 T0.2 IPOPT cross-checks (leaf only)').install()
    gurobi_guard = GurobiLeafGuard(permitted=leaf_site, label='P5.16 D0 T0.2 Gurobi relaxation solves (leaf only)'
                                   ).install()
    t_start = time.time()
    global np
    import numpy
    np = numpy
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

    # ---- predictions first, before any input is loaded -----------------------------------------------------------
    write_json('predictions.json', {'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'predictions': PREDICTIONS,
                                    'operationalisation': OPERATIONALISATION, 'declared_solves': DECLARED,
                                    'gurobi_params': GUROBI_PARAMS, 'gurobi_params_freeze': GUROBI_PARAMS_FREEZE,
                                    'ipopt_options': IPOPT_OPTIONS,
                                    'harness_version': HARNESS_VERSION, 'record_schema': RECORD_SCHEMA,
                                    'note': 'recorded by the harness before any input is loaded (task P5.16 D0 / '
                                            'T0.2; spec v2 section 6 P1-P9 and the expert\'s predictions E1-E5)'})
    print('[write] predictions.json')
    print(f'[settings] {HARNESS_VERSION}; record schema {RECORD_SCHEMA["version"]}')
    print(f'[settings] Gurobi params (72 group LPs, 12 fixed-mode) {json.dumps(GUROBI_PARAMS)}')
    print(f'[settings] Gurobi params (24 freeze solves, section 17.2) {json.dumps(GUROBI_PARAMS_FREEZE)}')
    print(f'[settings] IPOPT options {json.dumps(IPOPT_OPTIONS)}')

    # ---- capture-path assertion: a writer key AND a capture function for every required quantity ----------------
    missing = sorted({f for f, _k, _fn in REQUIRED_CAPTURES.values() if f not in OUTPUT_FILES})
    if missing:
        raise RuntimeError(f'capture-path assertion failed: no writer for {missing}')

    # ---- inputs: manifests and every file read, hash-checked; refuse on any mismatch -----------------------------
    mismatch = []
    inputs = {}

    def check(path, expected, label):
        actual = sha256_file(os.path.join(REPO, path)) if os.path.exists(os.path.join(REPO, path)) else None
        inputs[label] = {'path': path, 'expected': expected, 'actual': actual, 'match': actual == expected}
        if actual != expected:
            mismatch.append(label)
        return actual == expected

    check(os.path.join(T01C_ROOT, 'manifest_sha256.json'), T01C_MANIFEST_SHA256, 't01c/manifest')
    check(os.path.join(T01B_ROOT, 'manifest_sha256.json'), T01B_MANIFEST_SHA256, 't01b/manifest')
    check(W183E_EXPORT, W183E_EXPORT_SHA256, 'w183e/e6_export.json')
    t01c_files = json.load(open(os.path.join(REPO, T01C_ROOT, 'manifest_sha256.json')))['files']
    t01b_files = json.load(open(os.path.join(REPO, T01B_ROOT, 'manifest_sha256.json')))['files']
    for name in ('lp/index.json', 'objective.json', 'capacity.json', 'inventory.json'):
        check(os.path.join(T01C_ROOT, name), t01c_files[name]['sha256'], f't01c/{name}')
    check(os.path.join(T01B_ROOT, 'coupling.json'), t01b_files['coupling.json']['sha256'], 't01b/coupling.json')
    if mismatch:
        write_json('t02.json', {'refused': True, 'inputs': inputs, 'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5
    lp_index = json.load(open(os.path.join(REPO, T01C_ROOT, 'lp', 'index.json')))['lp_files']
    objective = json.load(open(os.path.join(REPO, T01C_ROOT, 'objective.json')))
    capacity = json.load(open(os.path.join(REPO, T01C_ROOT, 'capacity.json')))
    t01c_inventory = json.load(open(os.path.join(REPO, T01C_ROOT, 'inventory.json')))
    coupling_const = json.load(open(os.path.join(REPO, T01B_ROOT, 'coupling.json')))['constants']
    del_inv = {'cell_records': t01c_inventory['cell_records'], 'provenance_file_sha256':
               t01c_inventory['provenance']['file_sha256']}
    del t01c_inventory
    gc.collect()
    w183e = json.load(open(os.path.join(REPO, W183E_EXPORT)))
    cell_records = {}
    for cell in CELLS:
        ev_path = os.path.join(cell['eval_dir'], 'evaluation_record.json')
        check(ev_path, del_inv['cell_records'][cell['label']]['record_sha256']['evaluation_record.json'],
              f'{cell["label"]}/evaluation_record.json')
        ev = json.load(open(os.path.join(REPO, ev_path)))
        cell_records[cell['label']] = {'gross': float(ev['terminal_gross_operational_cost']),
                                       'bar': float(ev['bar']['value']), 'bar_definition': ev['bar']['definition'],
                                       'certification_cycle': ev['certification_cycle'],
                                       'candidate_key': ev['candidate_key'], 'eval_dir': cell['eval_dir']}
    if mismatch:
        write_json('t02.json', {'refused': True, 'inputs': inputs, 'mismatch': mismatch})
        print(f'[REFUSED] input hash mismatch: {mismatch}')
        return 5
    storage_value = cell_records['x0']['gross'] - cell_records['unit_n7']['gross']
    bar_sum = cell_records['x0']['bar'] + cell_records['unit_n7']['bar']
    print(f'[inputs] manifests and records match; storage value {storage_value:.2f} EUR; bar-sum {bar_sum:.2f} EUR')

    # ---- groups, resume set --------------------------------------------------------------------------------------
    years = [2025, 2030, 2035]
    days = ['Spring', 'Summer', 'Autumn', 'Winter']
    groups = [(y, d) for y in years for d in days]
    if GROUP_LIMIT:
        groups = groups[:GROUP_LIMIT]
    order = [(cell['label'], y, d) for (y, d) in groups for cell in CELLS]      # interleaved: x0 g, unit g, ...
    resumed = {}
    if _RESUME_FROM:
        prior = os.path.join(REPO, _RESUME_FROM)
        prior_manifest = json.load(open(os.path.join(prior, 'manifest_sha256.json')))['files']
        for name, rec in prior_manifest.items():
            if name.startswith('groups/') and name.endswith('.json'):
                path = os.path.join(prior, name)
                if sha256_file(path) != rec['sha256']:
                    write_json('t02.json', {'refused': True, 'resume_mismatch': name})
                    return 5
                g_rec = json.load(open(path))
                if g_rec.get('record_schema') != RECORD_SCHEMA['version']:
                    write_json('t02.json', {'refused': True, 'resume_schema_mismatch': name,
                                            'found': g_rec.get('record_schema'), 'expected': RECORD_SCHEMA['version']})
                    print(f'[REFUSED] resume: {name} has record schema {g_rec.get("record_schema")!r}, not '
                          f'{RECORD_SCHEMA["version"]!r}')
                    return 5
                if g_rec.get('complete'):
                    resumed[(g_rec['cell'], g_rec['year'], g_rec['day'])] = {'path': rel(path), 'sha256': rec['sha256'],
                                                                             'record': g_rec}
    todo = [k for k in order if k not in resumed]
    n_gurobi = sum((3 + 1 + 2) if c == 'unit_n7' else 3 for c, _y, _d in todo)
    n_ipopt = len(todo)
    declared_now = {'gurobi': 0 if DRY else n_gurobi, 'ipopt': 0 if DRY else n_ipopt}
    if not DRY and not _RESUME_FROM and not GROUP_LIMIT and (n_gurobi, n_ipopt) != (108, 24):
        raise RuntimeError(f'declared count mismatch: {n_gurobi} Gurobi + {n_ipopt} IPOPT != 108 + 24')
    print(f'[declared] this process: Gurobi {declared_now["gurobi"]}, IPOPT {declared_now["ipopt"]} '
          f'(task total {DECLARED_SOLVES}; resumed groups {len(resumed)})')

    # ---- solver state for the leaf; static leaf check -------------------------------------------------------------
    project_modules = {os.path.splitext(f)[0] for f in os.listdir(REPO) if f.endswith('.py')}
    leaf_names = set(_leaf_solve.__code__.co_names)
    for const in _leaf_solve.__code__.co_consts:
        if hasattr(const, 'co_names'):
            leaf_names |= set(const.co_names)
    leaf_project_refs = sorted(leaf_names & (project_modules - {'p516_d0_t02'}))
    if leaf_project_refs:
        raise RuntimeError(f'the leaf names project modules {leaf_project_refs}: production reachable from it')
    from pyomo.opt import SolverFactory
    _STATE.update({'gp': gp, 'gurobi_guard': gurobi_guard, 'solver_factory': SolverFactory,
                   'ipopt_path': _load_dotenv_ipopt()})
    env = gp.Env(empty=True)
    env.setParam('LogToConsole', 0)
    env.start()
    _STATE['env'] = env
    gurobi_licence = 'started'
    print(f'[leaf] static check: leaf names {sorted(leaf_names)}; project modules referenced: none; Gurobi Env '
          f'{gurobi_licence}; IPOPT {_STATE["ipopt_path"]}')

    # ---- network topology + capacity-row objects: the T0.1b capture pickles (hash-checked), unpickled under the
    #      armed guards; project modules loaded by unpickling are hash-recorded ---------------------------------------
    import pickle
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_mutable_parameters

    def load_blocks(label, y, d):
        name = f'captures/{label}/blocks_{y}_{d}.pkl'
        path = os.path.join(REPO, T01B_ROOT, name)
        if sha256_file(path) != t01b_files[name]['sha256']:
            raise RuntimeError(f'{name}: hash differs from the T0.1b manifest')
        with open(path, 'rb') as handle:
            blocks = pickle.load(handle)
        vec_name = f'captures/{label}/vectors_{y}_{d}.json'
        vpath = os.path.join(REPO, T01B_ROOT, vec_name)
        if sha256_file(vpath) != t01b_files[vec_name]['sha256']:
            raise RuntimeError(f'{vec_name}: hash differs from the T0.1b manifest')
        return blocks, json.load(open(vpath))

    vectors_esso = {}
    for cell in CELLS:
        name = f'captures/{cell["label"]}/vectors_esso.json'
        path = os.path.join(REPO, T01B_ROOT, name)
        if sha256_file(path) != t01b_files[name]['sha256']:
            raise RuntimeError(f'{name}: hash differs from the T0.1b manifest')
        vectors_esso[cell['label']] = json.load(open(path))

    import definitions
    import model_construction_helpers as mch
    delta_mode = math.sqrt(float(definitions.ESS_COMPLEMENTARITY_TOLERANCE))
    cycle_basis_fn = _cycle_basis
    cal_factor = {tuple(int(v) for v in k.strip('()').split(',')): rec['factor']
                  for k, rec in capacity['calendar_inputs']['calendar_factor'].items()}

    gate_rows, stops = [], []

    def gate(name, scope, achieved, threshold, rule='<=', note=None):
        achieved = float(achieved)
        ok = bool(achieved <= threshold) if rule == '<=' else bool(achieved == threshold)
        gate_rows.append({'gate': name, 'scope': scope, 'achieved': achieved, 'threshold': threshold, 'rule': rule,
                          'pass': ok, 'note': note})
        if not ok:
            print(f'[gate FAIL] {name} {scope}: achieved {achieved:.6e} {rule} {threshold:.3e}')
        return ok

    solve_index = []
    rss_peak = {'self_max_rss_bytes': 0, 'children_max_rss_bytes': 0}

    def note_rss():
        s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rss_peak['self_max_rss_bytes'] = max(rss_peak['self_max_rss_bytes'], s)
        rss_peak['children_max_rss_bytes'] = max(rss_peak['children_max_rss_bytes'], c)
        return {'self_max_rss_bytes': s, 'children_max_rss_bytes': c}

    def parse_gurobi_log(path):
        if not os.path.exists(path):
            return {'present': False}
        text = open(path, errors='replace').read()
        it = re.findall(r'^\s*(\d+)\*?\s+([-+]?\d\.\d+e[-+]\d+)\s+([-+]?\d\.\d+e[-+]\d+)\s+(\d\.\d+e[-+]\d+)\s+'
                        r'(\d\.\d+e[-+]\d+)\s+(\d\.\d+e[-+]\d+)\s+\d+s', text, re.M)
        out = {'present': True, 'bytes': len(text), 'n_optimize_banners': text.count('Optimize a model with'),
               'qcp_dual_failed': QCP_DUAL_FAILURE_TEXT in text,
               'nonconvex_message': bool(re.search(r'not (positive semi-definite|PSD)|non-?convex', text, re.I))}
        if it:
            last = it[-1]
            primal, dual = float(last[1]), float(last[2])
            out.update({'last_barrier_iter': int(last[0]), 'primal_obj': primal, 'dual_obj': dual,
                        'primal_res': float(last[3]), 'dual_res': float(last[4]), 'compl': float(last[5]),
                        'gap_abs': abs(primal - dual), 'gap_rel': abs(primal - dual) / max(1.0, abs(primal))})
        m_ = re.search(r'Optimal objective\s+([-+0-9.eE]+)', text)
        out['optimal_objective_line'] = float(m_.group(1)) if m_ else None
        return out

    def resolution_record(data, sol, x_cert, bounds):
        """Section 15 ruling 1, per solve: res_LB at the returned point, res_cert at the certified mapped point clipped
        into the solve's bounds, both priced at the solve's own multipliers."""
        lb_s, ub_s = (data['lb'], data['ub']) if bounds is None else bounds
        x_c = np.clip(x_cert, lb_s, ub_s)
        res_lb = resolution_price(data, sol['X'], sol['Pi'], sol['QCPi'], sol['RC'], (lb_s, ub_s))
        res_ct = resolution_price(data, x_c, sol['Pi'], sol['QCPi'], sol['RC'], (lb_s, ub_s))
        return {'resolution': {'res_LB': res_lb, 'res_cert': res_ct,
                               'certified_point': 'T0.1c certified mapped point clipped into this solve\'s bounds '
                                                  f'(max clip {float(np.max(np.abs(x_c - x_cert), initial=0.0)):.3e})',
                               'definition': RESOLUTION_DEFINITION},
                'res_LB_eur': res_lb.get('total_eur'), 'res_cert_eur': res_ct.get('total_eur')}

    def gurobi_solve(tag, lp_path, data, x_stand_in, bounds=None, params=None, params_set='standard'):
        params = GUROBI_PARAMS if params is None else params
        log_path = os.path.join(OUT_DIR, 'logs', f'{tag}.gurobi.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        rec = {'tag': tag, 'solver': 'gurobi', 'lp': rel(lp_path), 'log': rel(log_path), 'params': dict(params),
               'params_set': params_set}
        if DRY:
            rec.update({'STAND_IN': True, 'status': 'DRY', 'ObjVal': float(data['c'] @ x_stand_in) + data['obj_con'],
                        'wall_s': 0.0})
            sol = {'X': x_stand_in, 'Pi': np.zeros(len(data['rhs'])), 'QCPi': np.zeros(len(data['qc_rhs'])),
                   'RC': data['c'].copy()}
        else:
            payload = {'lp_path': lp_path, 'params': params, 'log_path': log_path,
                       'lb': None if bounds is None else bounds[0], 'ub': None if bounds is None else bounds[1]}
            res = _leaf_solve('gurobi', payload)
            sol = {k: res.get(k) for k in ('X', 'Pi', 'QCPi', 'RC')}
            rec.update({k: res.get(k) for k in ('Status', 'ObjVal', 'ObjBound', 'BarIterCount', 'Runtime', 'wall_s',
                                                'error', 'NumVars', 'NumConstrs', 'NumQConstrs')})
            rec['status'] = {2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 12: 'NUMERIC',
                             13: 'SUBOPTIMAL'}.get(res.get('Status'), res.get('Status'))
            for k in ('X', 'Pi', 'QCPi', 'RC'):
                if res.get(f'{k}_error'):
                    rec[f'{k}_error'] = res[f'{k}_error']
        rec['log_parse'] = parse_gurobi_log(log_path)
        rec['dual_less'] = bool(rec['log_parse'].get('qcp_dual_failed')) or sol['Pi'] is None or \
            (len(data['qc_rhs']) > 0 and sol['QCPi'] is None)
        rec['solver_gap_abs'] = rec['log_parse'].get('gap_abs')
        rec.update(resolution_record(data, sol, x_stand_in, bounds))
        rec['rss'] = note_rss()
        if rec.get('error') and re.search(r'not (positive semi-definite|PSD)|non-?convex|Q matrix',
                                          rec['error']['message'], re.I):
            stops.append({'stop': 'Gurobi refused a model as nonconvex', 'tag': tag, 'error': rec['error'],
                          'log': rel(log_path)})
        vec_path = os.path.join(OUT_DIR, 'vectors', f'{tag}.npz')
        os.makedirs(os.path.dirname(vec_path), exist_ok=True)
        arrays = {k: v for k, v in sol.items() if v is not None}
        np.savez_compressed(vec_path, **arrays)
        rec['vectors'] = rel(vec_path)
        rec['lp_sha256'] = sha256_file(lp_path)
        solve_index.append({k: rec[k] for k in ('tag', 'solver', 'log', 'vectors', 'status')})
        return rec, sol

    def build_ipopt_model(data, x0):
        m = pe.ConcreteModel(name='p516_d0_t02_ipopt_smooth_form')
        n = len(data['var_names'])
        lb = [None if not math.isfinite(v) else float(v) for v in data['lb']]
        ub = [None if not math.isfinite(v) else float(v) for v in data['ub']]
        m.x = pe.Var(range(n), bounds=lambda _m, k: (lb[k], ub[k]))
        for k in range(n):
            v = float(x0[k])
            if lb[k] is not None:
                v = max(v, lb[k])
            if ub[k] is not None:
                v = min(v, ub[k])
            m.x[k].set_value(v, skip_validation=True)
        A = data['A']
        from pyomo.core.expr.numeric_expr import LinearExpression
        m.lin = pe.ConstraintList()
        for i in range(A.shape[0]):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            expr = LinearExpression(constant=0.0, linear_coefs=A.data[lo:hi].tolist(),
                                    linear_vars=[m.x[int(j)] for j in A.indices[lo:hi]])
            s, b = data['sense'][i], float(data['rhs'][i])
            m.lin.add(expr == b if s == '=' else (expr <= b if s == '<' else expr >= b))
        m.quad = pe.ConstraintList()
        for j, q in enumerate(data['qrows']):
            expr = q['lin_const'] + sum(c * m.x[a] * m.x[b_] for a, b_, c in q['quad']) + \
                sum(c * m.x[a] for a, c in q['lin'])
            s, b = data['qc_sense'][j], float(data['qc_rhs'][j])
            m.quad.add(expr <= b if s == '<' else (expr >= b if s == '>' else expr == b))
        nz = np.nonzero(data['c'])[0]
        m.obj = pe.Objective(expr=data['obj_con'] + LinearExpression(constant=0.0,
                                                                     linear_coefs=data['c'][nz].tolist(),
                                                                     linear_vars=[m.x[int(k)] for k in nz]),
                             sense=pe.minimize)
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        m.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)
        m.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
        return m

    def parse_ipopt_log(path):
        if not os.path.exists(path):
            return {'present': False}
        text = open(path, errors='replace').read()
        out = {'present': True, 'bytes': len(text), 'n_banners': text.count('This program contains Ipopt')}
        for key, pat in (('iterations', r'Number of Iterations\.+:\s+(\d+)'),
                         ('objective_scaled_unscaled', r'Objective\.+:\s+(\S+)\s+(\S+)'),
                         ('dual_infeasibility', r'Dual infeasibility\.+:\s+(\S+)\s+(\S+)'),
                         ('constraint_violation', r'Constraint violation\.+:\s+(\S+)\s+(\S+)'),
                         ('complementarity', r'Complementarity\.+:\s+(\S+)\s+(\S+)'),
                         ('overall_nlp_error', r'Overall NLP error\.+:\s+(\S+)\s+(\S+)'),
                         ('n_variables', r'Total number of variables\.+:\s+(\d+)'),
                         ('n_equality_constraints', r'Total number of equality constraints\.+:\s+(\d+)'),
                         ('n_inequality_constraints', r'Total number of inequality constraints\.+:\s+(\d+)')):
            m_ = re.search(pat, text)
            out[key] = (m_.groups() if m_ and len(m_.groups()) > 1 else (m_.group(1) if m_ else None))
        m_ = re.findall(r'^EXIT: (.*)$', text, re.M)
        out['exit'] = m_[-1].strip() if m_ else None
        return out

    def ipopt_solve(tag, data, x0):
        log_path = os.path.join(OUT_DIR, 'logs', f'{tag}.ipopt.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        t_build = time.time()
        m = build_ipopt_model(data, x0)
        rec = {'tag': tag, 'solver': 'ipopt', 'log': rel(log_path), 'options': IPOPT_OPTIONS,
               'build_s': time.time() - t_build, 'initial_point': 'certified mapped point (T0.1c ef), clipped to bounds'}
        if DRY:
            nl_path = os.path.join(OUT_DIR, 'scratch_nl', f'{tag}.nl')
            os.makedirs(os.path.dirname(nl_path), exist_ok=True)
            t_nl = time.time()
            m.write(nl_path)
            with open(nl_path) as handle:
                header = [handle.readline().split() for _ in range(2)][1]
            rec.update({'STAND_IN': True, 'status': 'DRY', 'nl_write_s': time.time() - t_nl,
                        'nl_bytes': os.path.getsize(nl_path), 'ObjVal': float(data['c'] @ x0) + data['obj_con'],
                        'nl_header_vars_cons_objs_ranges_eqns': [int(v) for v in header[:5]]})
            os.remove(nl_path)
            sol = {'X': x0, 'Pi': np.zeros(len(data['rhs'])), 'QCPi': np.zeros(len(data['qc_rhs'])),
                   'RC': data['c'].copy()}
        else:
            res = _leaf_solve('ipopt', {'model': m, 'options': IPOPT_OPTIONS, 'log_path': log_path})
            results = res['results']
            rec.update({'wall_s': res['wall_s'], 'termination_condition': str(results.solver.termination_condition),
                        'solver_status': str(results.solver.status)})
            n = len(data['var_names'])
            x = np.array([m.x[k].value if m.x[k].value is not None else np.nan for k in range(n)])
            duals_lin = np.array([m.dual.get(m.lin[i + 1], np.nan) for i in range(len(data['rhs']))])
            duals_q = np.array([m.dual.get(m.quad[j + 1], np.nan) for j in range(len(data['qc_rhs']))])
            zl = np.array([m.ipopt_zL_out.get(m.x[k], 0.0) for k in range(n)])
            zu = np.array([m.ipopt_zU_out.get(m.x[k], 0.0) for k in range(n)])
            rec['ObjVal'] = float(pe.value(m.obj))
            # the multiplier sign convention is read off stationarity (both conventions recorded)
            conv = {}
            for sgn in (1.0, -1.0):
                r = data['c'] - data['A'].T @ (sgn * duals_lin)
                for j, q in enumerate(data['qrows']):
                    _v, qg = quadratic_value_and_grad(q, x)
                    for i, gval in qg.items():
                        r[i] -= sgn * duals_q[j] * gval
                conv[sgn] = float(np.nanmax(np.abs(r - (zl + zu))))
            sgn = min(conv, key=conv.get)
            rec['multiplier_sign_convention'] = {'chosen_sign': sgn, 'stationarity_residual_by_sign':
                                                 {str(k): v for k, v in conv.items()},
                                                 'definition': 'pi_gurobi_convention = sign * pyomo dual; bound '
                                                               'multipliers z = ipopt_zL_out + ipopt_zU_out'}
            sol = {'X': x, 'Pi': sgn * duals_lin, 'QCPi': sgn * duals_q, 'RC': zl + zu}
        rec['log_parse'] = parse_ipopt_log(log_path)
        rec.update(resolution_record(data, sol, x0, None))
        x_start = np.clip(x0, data['lb'], data['ub'])        # = build_ipopt_model's start (max(lb), then min(ub))
        start_v = resolution_price(data, x_start, np.zeros(len(data['rhs'])), np.zeros(len(data['qc_rhs'])),
                                   np.zeros(len(x_start)))
        rec['iteration0'] = parse_ipopt_iteration0(log_path)
        rec['start_kept_evidence'] = {
            'iteration0_inf_pr': rec['iteration0'].get('inf_pr'),
            'start_lp_max_violation_unscaled': {k: start_v[f'{k}_max_violation']
                                                for k in ('linear', 'quadratic', 'bounds')},
            'note': ('section 15 ruling 2: with bound_push / bound_frac 1e-10 the iteration-0 inf_pr should be of the '
                     'order of the start\'s own (scaled) row violation; the default push gave 6.9e-2 '
                     '(P5_16_D0_T02_IPOPT_INTERIOR_REPORT.md)')}
        n_fixed_cols = int(np.sum(data['lb'] == data['ub']))
        rec['lp_dimensions'] = {'vars': len(data['var_names']), 'rows': len(data['rhs']) + len(data['qrows']),
                                'eq_rows': int(np.sum(data['sense'] == '=')), 'n_fixed_columns': n_fixed_cols,
                                'vars_expected_by_ipopt': len(data['var_names']) - n_fixed_cols,
                                'note': 'section 16 ruling 3: IPOPT removes the columns with lb == ub'}
        if DRY:
            hdr = rec['nl_header_vars_cons_objs_ranges_eqns']
            gate('IPOPT smooth form: NL header (vars, cons, eqns) == LP dimensions (0 = equal)', tag,
                 0.0 if (hdr[0], hdr[1], hdr[4]) == (rec['lp_dimensions']['vars'], rec['lp_dimensions']['rows'],
                                                     rec['lp_dimensions']['eq_rows']) else 1.0, 0.0)
        elif rec['log_parse'].get('n_variables') is not None:
            n_ineq = rec['log_parse'].get('n_inequality_constraints')
            n_eq = rec['log_parse'].get('n_equality_constraints')
            rec['lp_dimensions']['vars_reported_by_ipopt'] = int(rec['log_parse']['n_variables'])
            gate('IPOPT smooth form: IPOPT-reported variables == LP columns - bound-fixed columns, eq + ineq == LP rows '
                 '(0 = equal)', tag,
                 0.0 if (int(rec['log_parse']['n_variables']) == rec['lp_dimensions']['vars_expected_by_ipopt'] and
                         int(n_eq) + int(n_ineq) == rec['lp_dimensions']['rows']) else 1.0, 0.0,
                 note=(f'section 16 ruling 3: LP columns {rec["lp_dimensions"]["vars"]}, bound-fixed '
                       f'{rec["lp_dimensions"]["n_fixed_columns"]}, IPOPT-reported {rec["log_parse"]["n_variables"]}; '
                       'a mismatch is investigated, not a stop'))
        rec['rss'] = note_rss()
        vec_path = os.path.join(OUT_DIR, 'vectors', f'{tag}.npz')
        os.makedirs(os.path.dirname(vec_path), exist_ok=True)
        np.savez_compressed(vec_path, **{k: v for k, v in sol.items() if v is not None})
        rec['vectors'] = rel(vec_path)
        solve_index.append({k: rec.get(k) for k in ('tag', 'solver', 'log', 'vectors', 'status',
                                                    'termination_condition')})
        del m
        gc.collect()
        return rec, sol

    def frozen_rows_check(data, lb, ub):
        """Zero-solve: rows whose columns are all fixed (LB == UB) evaluated at the fixed values."""
        fixed = lb == ub
        A = data['A']
        worst, n_rows, worst_row = 0.0, 0, None
        xf = np.where(fixed, lb, 0.0)
        for i in range(A.shape[0]):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            cols = A.indices[lo:hi]
            if len(cols) and fixed[cols].all():
                n_rows += 1
                v = float(A.data[lo:hi] @ xf[cols]) - data['rhs'][i]
                viol = abs(v) if data['sense'][i] == '=' else (max(v, 0.0) if data['sense'][i] == '<' else
                                                               max(-v, 0.0))
                if viol > worst:
                    worst, worst_row = viol, data['con_names'][i]
        for j, q in enumerate(data['qrows']):
            cols = {a for a, _b, _c in q['quad']} | {b_ for _a, b_, _c in q['quad']} | {a for a, _c in q['lin']}
            if all(fixed[c] for c in cols):
                n_rows += 1
                v, _g = quadratic_value_and_grad(q, xf)
                v -= data['qc_rhs'][j]
                viol = max(v, 0.0) if data['qc_sense'][j] == '<' else max(-v, 0.0)
                if viol > worst:
                    worst, worst_row = viol, data['qc_names'][j]
        return {'n_all_fixed_rows': n_rows, 'max_violation': worst, 'worst_row': worst_row}

    # ---- capacity sensitivity (f), unit cell --------------------------------------------------------------------
    _cap_rows_cache = {}

    def capacity_sensitivity(data, sol, blocks, label, y, d, var):
        """dQ_LB/dE (per MWh) and dQ_LB/dS (per MVA) of the 2025 cohort at node 7, from the solve's multipliers:
        ESSO part = Pi of the rated_e / rated_s_capacity_unit rows (rhs = the investment); network part =
        -sum_r pi_r dh_r/dtheta over every LP row of the TSO / DSO7 copies that references shared_es_e/s_rated_fixed
        (h = body - bound; dh/dparam by a central difference on the live block at x*), times dparam/dtheta
        (E: phi^age_end / s_base on C, 1 / s_base on R; S: 1 / s_base), plus min(RC, 0) * 0.05 * dparam/dtheta for the
        capacity-derived slack bounds."""
        if sol['Pi'] is None or sol['QCPi'] is None:
            return {'available': False, 'reason': 'dual-less'}
        yi = years.index(y)
        cons_by_name = {}
        for i, nm in enumerate(data['con_names']):
            cons_by_name.setdefault(nm, []).append(('L', i, data['con_symbols'][i]))
        for j, nm in enumerate(data['qc_names']):
            cons_by_name.setdefault(nm, []).append(('Q', j, data['qc_symbols'][j]))
        esso = {'E': 0.0, 'S': 0.0, 'rows': []}
        A = data['A']
        for i, nm in enumerate(data['con_names']):
            if nm and nm.startswith('esso_7.rated_') and '_capacity_unit[' in nm:
                lo, hi = A.indptr[i], A.indptr[i + 1]
                cols = A.indices[lo:hi]
                if len(cols) != 1:
                    continue
                b_, fam, idx = data['parsed'][cols[0]]
                if idx[0] != 0 or idx[1] != yi:
                    continue
                kind = 'E' if fam == 'es_e_rated_per_unit' else ('S' if fam == 'es_s_rated_per_unit' else None)
                if kind is None:
                    continue
                coef = float(A.data[lo])
                esso[kind] += float(sol['Pi'][i]) / coef
                esso['rows'].append({'row': nm, 'kind': kind, 'pi': float(sol['Pi'][i]), 'coef': coef,
                                     'rhs': float(data['rhs'][i])})
        net = {'E': 0.0, 'S': 0.0, 'rows': 0, 'bounds': 0}
        for prefix, key in (('tso', 'TSO'), ('dso_7', 'DSO7')):
            blk = blocks['tso'] if key == 'TSO' else blocks['dso'][7]
            netw = blk.node_balance_p._rule._fcn.keywords['network']
            sb = float(netw.baseMVA)
            e_idx = coupling_const[label][f'{y}_{d}_7']['e_tso' if key == 'TSO' else 'e_dso']
            caps = capacity['vectors'][label]['per_group'][f'{y}_{d}'][key][var]
            for e_, cap in caps.items():
                blk.shared_es_s_rated_fixed[int(e_)].set_value(cap['s'])
                blk.shared_es_e_rated_fixed[int(e_)].set_value(cap['e'])
            for k, (b_, fam, idx) in enumerate(data['parsed']):
                if b_ == prefix:
                    comp = blk.find_component(f'{fam}[{",".join(map(str, idx))}]' if idx else fam)
                    if comp is not None:
                        comp.set_value(float(sol['X'][k]), skip_validation=True)
            dE = (cal_factor[(0, yi)] if var == 'C' else 1.0) / sb
            dS = 1.0 / sb
            cache_key = (id(blk), e_idx)
            if cache_key not in _cap_rows_cache:
                _cap_rows_cache[cache_key] = [
                    (cd, [p for p in identify_mutable_parameters(cd.expr)
                          if p.parent_component().local_name in ('shared_es_s_rated_fixed', 'shared_es_e_rated_fixed')
                          and p.index() == e_idx])
                    for cd in blk.component_data_objects(pe.Constraint, active=True)]
                _cap_rows_cache[cache_key] = [(cd, ps) for cd, ps in _cap_rows_cache[cache_key] if ps]
            for cd, params in _cap_rows_cache[cache_key]:
                lp_rows = cons_by_name.get(f'{prefix}.{cd.name}')
                if not lp_rows:
                    continue
                for prm in params:
                    base = float(pe.value(prm))
                    h = max(abs(base), 1e-6) * 1e-4
                    vals = {}
                    for sgn in (1, -1):
                        prm.set_value(base + sgn * h)
                        body = float(pe.value(cd.body))
                        vals[sgn] = {'body': body, 'lo': None if cd.lower is None else float(pe.value(cd.lower)),
                                     'up': None if cd.upper is None else float(pe.value(cd.upper))}
                    prm.set_value(base)
                    for kind_, i, sym in lp_rows:
                        side = 'lo' if sym.startswith('c_l_') else 'up'
                        if sym.startswith('c_e_'):
                            side = 'up' if vals[1]['up'] is not None else 'lo'
                        dh = ((vals[1]['body'] - vals[1][side]) - (vals[-1]['body'] - vals[-1][side])) / (2 * h)
                        mult = float(sol['Pi'][i] if kind_ == 'L' else sol['QCPi'][i])
                        contrib = -mult * dh
                        if prm.parent_component().local_name == 'shared_es_e_rated_fixed':
                            net['E'] += contrib * dE
                        else:
                            net['S'] += contrib * dS
                        net['rows'] += 1
            for fam in ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down'):
                for k, (b_, f_, idx) in enumerate(data['parsed']):
                    if b_ == prefix and f_ == fam and idx[0] == e_idx:
                        rc = float(sol['RC'][k])
                        if rc < 0.0:
                            net['E'] += rc * mch.ESS_DAY_BALANCE_SLACK_FRACTION * dE
                        net['bounds'] += 1
        dual_E = esso['E'] + net['E']
        dual_S = esso['S'] + net['S']
        return {'available': True, 'variant': var, 'dual_E_per_MWh': dual_E, 'dual_S_per_MVA': dual_S,
                'esso_part': esso, 'network_part': net,
                'estimate_delta_Q': -(dual_E * 1.0 + dual_S * 0.25),
                'definition': capacity_sensitivity.__doc__}

    # ==================================================================================================================
    #  Main loop
    # ==================================================================================================================
    available_fns = {**globals(), **locals()}
    no_fn = sorted({fn for _f, _k, fn in REQUIRED_CAPTURES.values() if fn and not callable(available_fns.get(fn))})
    if no_fn:
        raise RuntimeError(f'capture-path assertion failed: no capture function {no_fn}')
    print(f'[capture] {len(REQUIRED_CAPTURES)} required quantities: writer keys in {OUTPUT_FILES} and capture '
          f'functions present (asserted before the first solve)')
    group_records = {k: v['record'] for k, v in resumed.items()}
    separability = {}
    t_kind = {}
    stop_resumable = None
    lp_root = os.path.join(REPO, T01C_ROOT)
    for (label, y, d) in todo:
        t_group = time.time()
        gkey = f'{y}_{d}'
        tag_g = f'{label}_{gkey}'
        print(f'\n========== group {label} {gkey} ==========')
        blocks, vectors = load_blocks(label, y, d)
        networks = {'TSO': blocks['tso'].node_balance_p._rule._fcn.keywords['network']}
        for n_ in (5, 7, 9):
            networks[f'DSO{n_}'] = blocks['dso'][n_].node_balance_p._rule._fcn.keywords['network']
        q_plus = objective['q_cert_plus'][label]['per_group'][gkey]
        grec = {'cell': label, 'year': y, 'day': d, 'q_cert_plus_group': q_plus['q_cert_plus'],
                'gross_group': q_plus['gross'], 'variants': {}, 'complete': False,
                'record_schema': RECORD_SCHEMA['version']}
        datas, xcerts, sols = {}, {}, {}
        for mode in MODES:
            var = VARIANT[mode]
            lp_rel = f'lp/{label}/{gkey}_{mode}.lp'
            entry = lp_index[lp_rel]
            lp_path = os.path.join(lp_root, lp_rel)
            nm_path = os.path.join(lp_root, entry['name_map'])
            for p_, key in ((lp_path, lp_rel), (nm_path, entry['name_map'])):
                if sha256_file(p_) != t01c_files[key]['sha256']:
                    raise RuntimeError(f'{key}: hash differs from the T0.1c manifest (refuse)')
            name_map = json.load(open(nm_path))['name_map']
            data = read_lp_data(lp_path, name_map, env)
            del name_map
            datas[var] = data
            fac = (lambda idx: cal_factor[(idx[0], idx[1])]) if var == 'C' else (lambda idx: 1.0)
            x_cert = certified_point(data, vectors, vectors_esso[label], networks, fac)
            xcerts[var] = x_cert
            obj_cert = float(data['c'] @ x_cert) + data['obj_con']
            ref = objective['cells'][label]['per_group'][gkey][var]['group_relaxed_objective_certified']
            gate('LP objective at the certified point == T0.1c group objective (rel)', f'{tag_g} {var}',
                 abs(obj_cert - ref) / abs(ref), 1e-8)
            vmin = min(data['lb'][k] for k, (_b, fam, _i) in enumerate(data['parsed']) if fam == 'vmag_sqr')
            gate('Wii lower bound > 0 on every vmag_sqr column (min LB; fails if <= 0)', f'{tag_g} {var}',
                 0.0 if vmin > 0 else 1.0, 0.0)
            sep = separability_inventory(data)
            separability[f'{tag_g}_{var}'] = sep
            gate('section 13 separability after pinned / inert elimination (0 = separable)', f'{tag_g} {var}',
                 0.0 if sep['separable'] else 1.0, 0.0)
            if var != 'C':
                sep.pop('rows_coupling_yd_literally')      # identical across variants; kept once (C)
            grec['variants'][var] = {'lp': lp_rel, 'lp_sha256': entry['sha256'], 'size': data['size'],
                                     'n_free_columns': int(np.sum(~np.isfinite(data['lb']) & ~np.isfinite(data['ub']))),
                                     'n_unbounded_columns': int(np.sum(~(np.isfinite(data['lb']) &
                                                                         np.isfinite(data['ub'])))),
                                     'obj_at_certified': obj_cert, 'min_vmag_sqr_lb': vmin,
                                     'note': VARIANT_NOTE[var]}
        # ---- the three group LPs ----
        for var in ('R', 'A', 'C'):
            data = datas[var]
            rec, sol = gurobi_solve(f'{tag_g}_{var}', os.path.join(lp_root, grec['variants'][var]['lp']), data,
                                    xcerts[var])
            sols[var] = sol
            q_lb = rec['ObjVal']
            gap_s = rec['solver_gap_abs'] or 0.0
            v = grec['variants'][var]
            v['solve'] = rec
            if q_lb is not None:
                v['Q_LB'] = q_lb
                v['gap_abs'] = grec['q_cert_plus_group'] - q_lb
                v['gap_rel'] = v['gap_abs'] / grec['q_cert_plus_group']
                v['res_LB_eur'], v['res_cert_eur'] = rec.get('res_LB_eur'), rec.get('res_cert_eur')
                v.update(gap_firm_reading(v['gap_abs'], v['res_LB_eur'], v['res_cert_eur'], grec['q_cert_plus_group']))
                v['containment_held'] = bool(q_lb <= grec['q_cert_plus_group'] + gap_s)
                if not gate('G-D0-4 containment: Q_LB - Q_cert+ - solver gap (<= 0)', f'{tag_g} {var}',
                            q_lb - grec['q_cert_plus_group'] - gap_s, 0.0):
                    stops.append({'stop': 'G-D0-4 containment violated', 'group': tag_g, 'variant': var,
                                  'Q_LB': q_lb, 'Q_cert_plus': grec['q_cert_plus_group'], 'solver_gap': gap_s})
            if sol['X'] is not None and not np.any(np.isnan(sol['X'])):
                v['lagrangian'] = lagrangian_bound(data, sol['X'], sol['Pi'], sol['QCPi'], sol['RC'],
                                                   q_lb if q_lb is not None else float('nan'))
                v['rank'] = rank_and_cycles(data, sol['X'], networks, cycle_basis_fn)
        # ---- unit cell: fixed mode (b) and the two freezes (c), all on variant C ----
        if label == 'unit_n7':
            data = datas['C']
            lp_c = os.path.join(lp_root, grec['variants']['C']['lp'])
            cst = coupling_const[label][f'{gkey}_7']
            caps_c = capacity['vectors'][label]['per_group'][gkey]
            schedule, agree = {}, {'dso_copy_disagree': 0, 'esso_disagree': 0, 'tie_or_tiny': 0}
            tv = vectors['TSO']['vars']
            dv = vectors['DSO7']['vars']
            ev = vectors_esso[label]['ESSO7']['vars']
            yi, di = years.index(y), days.index(d)
            for p in range(24):
                pch = abs(tv[f'shared_es_pch[{cst["e_tso"]},0,0,{p}]'][0])
                pdch = abs(tv[f'shared_es_pdch[{cst["e_tso"]},0,0,{p}]'][0])
                if max(pch, pdch) <= FIXED_MODE_TINY:
                    mode_p = 'charge'
                    agree['tie_or_tiny'] += 1
                else:
                    mode_p = 'charge' if pch >= pdch else 'discharge'
                schedule[p] = mode_p
                d_pch = abs(dv[f'shared_es_pch[{cst["e_dso"]},0,0,{p}]'][0])
                d_pdch = abs(dv[f'shared_es_pdch[{cst["e_dso"]},0,0,{p}]'][0])
                if max(d_pch, d_pdch) > FIXED_MODE_TINY and (('charge' if d_pch >= d_pdch else 'discharge') != mode_p):
                    agree['dso_copy_disagree'] += 1
                e_pch = ev[f'es_pch_per_unit[0,{yi},{di},{p}]'][0]
                e_pdch = ev[f'es_pdch_per_unit[0,{yi},{di},{p}]'][0]
                if max(e_pch, e_pdch) > FIXED_MODE_TINY and (('charge' if e_pch >= e_pdch else 'discharge') != mode_p):
                    agree['esso_disagree'] += 1
            lb_fm, ub_fm = data['lb'].copy(), data['ub'].copy()
            applied = 0
            for prefix, key, e_ in (('tso', 'TSO', cst['e_tso']), ('dso_7', 'DSO7', cst['e_dso'])):
                s_cap = caps_c[key]['C'][str(e_)]['s']
                for p, mode_p in schedule.items():
                    off = 'shared_es_pdch' if mode_p == 'charge' else 'shared_es_pch'
                    k = data['col'][f'{prefix}.{off}[{e_},0,0,{p}]']
                    ub_fm[k] = min(ub_fm[k], delta_mode * s_cap)
                    applied += 1
            fm_rec, fm_sol = gurobi_solve(f'{tag_g}_C_fixedmode', lp_c, data, np.clip(xcerts['C'], lb_fm, ub_fm),
                                          bounds=(lb_fm, ub_fm))
            fm = {'schedule': schedule, 'mode_source': 'certified TSO copy (shared_es_pch vs pdch per period; P5.5-D4 '
                                                       'rule: charge iff pch >= pdch, charge when max <= 1e-9 pu)',
                  'agreement': agree, 'delta': delta_mode, 'off_mode_ub': 'delta * S_C of the copy (pu)',
                  'copies_fixed': ['tso', 'dso_7'], 'n_bounds_applied': applied,
                  'esso_not_fixed_reason': ('the ESSO pch / pdch split is non-unique in the relaxation (spec section 3 '
                                            'item 4) and enters no gross term; P5.5-D4 fixed the network copies only'),
                  'solve': fm_rec}
            if fm_rec.get('ObjVal') is not None and grec['variants']['C'].get('Q_LB') is not None:
                fm['cost_eur'] = fm_rec['ObjVal'] - grec['variants']['C']['Q_LB']
                fm['cost_pct_of_group_gross'] = 100.0 * fm['cost_eur'] / grec['gross_group']
            fm['cost_resolution_beside'] = {'res_LB_eur_fixed_mode': fm_rec.get('res_LB_eur'),
                                            'res_LB_eur_C': grec['variants']['C'].get('res_LB_eur'),
                                            'note': 'section 15 ruling 1: each solve\'s resolution, reported beside '
                                                    'the cost (a difference of two solves, not a gap reading)'}
            if fm_sol['X'] is not None and not np.any(np.isnan(fm_sol['X'])):
                fm['lagrangian'] = lagrangian_bound(data, fm_sol['X'], fm_sol['Pi'], fm_sol['QCPi'], fm_sol['RC'],
                                                    fm_rec['ObjVal'] if fm_rec.get('ObjVal') is not None else np.nan,
                                                    bounds=(lb_fm, ub_fm))
            grec['fixed_mode'] = fm
            freeze = {}
            for fz, select in (('TN_W_frozen', lambda b_, fam: b_ == 'tso' and fam in W_FAMILIES),
                               ('DSO_ESSO_frozen', lambda b_, fam: bool(b_) and (
                                   (b_.startswith('dso_') and (fam in W_FAMILIES or fam in TR_UCD_FAMILIES)) or
                                   (b_.startswith('esso_') and fam not in ESSO_FREE_FAMILIES)))):
                lb_f, ub_f = data['lb'].copy(), data['ub'].copy()
                cols = [k for k, (b_, fam, _i) in enumerate(data['parsed']) if select(b_, fam)]
                val = np.clip(xcerts['C'][cols], data['lb'][cols], data['ub'][cols])
                lb_f[cols] = val
                ub_f[cols] = val
                pre = frozen_rows_check(data, lb_f, ub_f)
                pre['n_columns_frozen'] = len(cols)
                fam_count = {}
                for k in cols:
                    b_, fam, _i = data['parsed'][k]
                    fk = f'{b_}.{fam}' if not b_.startswith('esso_') else f'{b_}.*'
                    fam_count[fk] = fam_count.get(fk, 0) + 1
                pre['n_columns_frozen_by_block_family'] = dict(sorted(fam_count.items()))
                if fz == 'DSO_ESSO_frozen':
                    free_count = {}
                    for b_, fam, _i in data['parsed']:
                        if b_ and b_.startswith('esso_') and fam in ESSO_FREE_FAMILIES:
                            free_count[f'{b_}.{fam}'] = free_count.get(f'{b_}.{fam}', 0) + 1
                    pre['n_esso_columns_left_free_by_family'] = dict(sorted(free_count.items()))
                    if not free_count:
                        raise RuntimeError(f'{tag_g}: section 17.1 inert-chain columns not found in the name map')
                pre['n_columns_fixed_by_lp_bounds_before_freeze'] = int(np.sum(data['lb'] == data['ub']))
                pre['max_clip_into_bounds'] = float(np.max(np.abs(val - xcerts['C'][cols]))) if cols else 0.0
                print(f'[freeze] {tag_g} C {fz}: {len(cols)} columns frozen; {pre["n_all_fixed_rows"]} rows with every '
                      f'column fixed, max violation at the certified point {pre["max_violation"]:.3e} '
                      f'({pre["worst_row"]}); max clip {pre["max_clip_into_bounds"]:.3e}')
                f_rec, f_sol = gurobi_solve(f'{tag_g}_C_{fz}', lp_c, data, np.clip(xcerts['C'], lb_f, ub_f),
                                            bounds=(lb_f, ub_f), params=GUROBI_PARAMS_FREEZE, params_set='freeze')
                fr = {'columns': ('tso W (vmag_sqr, voltage_product_real / imag) at the rank-1 image of the certified '
                                  'e, f' if fz == 'TN_W_frozen' else
                                  'section 14.1 (b): the dso_5 / dso_7 / dso_9 W columns (vmag_sqr, voltage_product_real '
                                  '/ imag) and transformer d0_tr_U / C / D at the rank-1 image of the certified e, f, '
                                  'and every esso_5 / esso_7 / esso_9 column except es_avg_ch_dch_per_unit / '
                                  'es_D_per_unit (section 14.2 inert chain, free per section 17.1) at the certified '
                                  'point (ESSO available at the C row); clipped into bounds; every other DSO column '
                                  'free'),
                      'zero_solve_precheck': pre, 'solve': f_rec}
                if f_rec.get('ObjVal') is not None and grec['variants']['C'].get('Q_LB') is not None:
                    fr['Q_frozen_minus_Q_LB'] = f_rec['ObjVal'] - grec['variants']['C']['Q_LB']
                if f_sol['X'] is not None and not np.any(np.isnan(f_sol['X'])):
                    fr['lagrangian'] = lagrangian_bound(data, f_sol['X'], f_sol['Pi'], f_sol['QCPi'], f_sol['RC'],
                                                        f_rec['ObjVal'] if f_rec.get('ObjVal') is not None else np.nan,
                                                        bounds=(lb_f, ub_f))
                freeze[fz] = fr
            g_c = grec['variants']['C'].get('gap_abs')
            a_ = freeze['TN_W_frozen'].get('Q_frozen_minus_Q_LB')
            b_ = freeze['DSO_ESSO_frozen'].get('Q_frozen_minus_Q_LB')
            if None not in (g_c, a_, b_):
                lo_, hi_ = a_, g_c - b_
                freeze['attribution_bracket'] = {
                    'label': 'two-way freeze ATTRIBUTION BRACKET (spec section 5 (c)); not a rank-gap cost',
                    'G_group_C': g_c,
                    'G_group_C_firm': [grec['variants']['C'].get('gap_firm_lower'),
                                       grec['variants']['C'].get('gap_firm_upper')],
                    'G_group_C_point_estimate': grec['variants']['C'].get('gap_point_estimate'),
                    'res_LB_eur_frozen': {k_: freeze[k_]['solve'].get('res_LB_eur') for k_ in freeze},
                    'A_TN_W_frozen_minus_Q_LB': a_, 'B_DSO_ESSO_frozen_minus_Q_LB': b_,
                    'TN_share_bracket': [min(lo_, hi_), max(lo_, hi_)],
                    'DSO_ESSO_share_bracket': [min(b_, g_c - a_), max(b_, g_c - a_)],
                    'interaction_A_plus_B_minus_G': a_ + b_ - g_c}
            grec['freeze'] = freeze
            grec['capacity_sensitivity'] = {var: capacity_sensitivity(datas[var], sols[var], blocks, label, y, d, var)
                                            for var in ('R', 'C')}
        # ---- IPOPT smooth-form cross-check on C ----
        data = datas['C']
        ip_rec, ip_sol = ipopt_solve(f'{tag_g}_C_ipopt', data, xcerts['C'])
        ip = {'solve': ip_rec}
        if ip_sol['X'] is not None and not np.any(np.isnan(ip_sol['X'])):
            ip['lagrangian'] = lagrangian_bound(data, ip_sol['X'], ip_sol['Pi'], ip_sol['QCPi'], ip_sol['RC'],
                                                ip_rec.get('ObjVal', np.nan))
            g_i = ip['lagrangian'].get('g_tight')
            ip['ipopt_gap_abs'] = (ip_rec['ObjVal'] - g_i) if (g_i is not None and math.isfinite(g_i)) else None
        if ip_rec.get('ObjVal') is not None:
            # section 15 ruling 1 / section 16 ruling 1: the IPOPT cross-check's own gap reading, firm interval, point
            ip['gap_abs'] = grec['q_cert_plus_group'] - ip_rec['ObjVal']
            ip['gap_rel'] = ip['gap_abs'] / grec['q_cert_plus_group']
            ip['res_LB_eur'], ip['res_cert_eur'] = ip_rec.get('res_LB_eur'), ip_rec.get('res_cert_eur')
            ip.update(gap_firm_reading(ip['gap_abs'], ip['res_LB_eur'], ip['res_cert_eur'], grec['q_cert_plus_group']))
        q_g = grec['variants']['C'].get('Q_LB')
        if q_g is not None and ip_rec.get('ObjVal') is not None:
            diff = abs(q_g - ip_rec['ObjVal'])
            gap_sum = (grec['variants']['C']['solve'].get('solver_gap_abs') or 0.0) + (ip.get('ipopt_gap_abs') or 0.0)
            ip['agreement'] = {'abs_diff': diff, 'rel_diff': diff / abs(q_g), 'gap_sum': gap_sum,
                               'held': bool(diff / abs(q_g) <= 1e-7 or diff <= gap_sum),
                               'res_LB_eur_beside': {'gurobi': grec['variants']['C'].get('res_LB_eur'),
                                                     'ipopt': ip_rec.get('res_LB_eur')},
                               'note': ('G-D0-5 rule unchanged; each solve\'s resolution (section 15 ruling 1) '
                                        'reported beside, not entered into `held`')}
            gate('Gurobi vs IPOPT on C: rel diff <= 1e-7 or abs diff <= gap sum (0 = held)', tag_g,
                 0.0 if ip['agreement']['held'] else 1.0, 0.0)
        grec['ipopt'] = ip
        grec['wall_s'] = time.time() - t_group
        grec['complete'] = True
        write_json(f'groups/{tag_g}.json', grec, compact=True)
        group_records[(label, y, d)] = grec
        print(f'[group] {tag_g}: ' + '; '.join(
            f'{v_} Q_LB {grec["variants"][v_].get("Q_LB")} gap {grec["variants"][v_].get("gap_rel")}'
            for v_ in ('R', 'A', 'C')) + f'; wall {grec["wall_s"]:.1f} s')
        del blocks, datas, xcerts, sols
        _cap_rows_cache.clear()
        gc.collect()
        if any(s['stop'].startswith(('G-D0-4', 'Gurobi refused')) for s in stops):
            print('[STOP] containment violation or nonconvex refusal; records of completed groups written')
            break
        kind = label
        t_kind.setdefault(kind, grec['wall_s'])
        if len(t_kind) == 2 and not DRY:
            remaining = [k for k in todo if k not in group_records]
            projection = (time.time() - t_start) + sum(t_kind[c] for c, _y, _d in remaining)
            print(f'[projection] elapsed {time.time() - t_start:.0f} s + remaining {len(remaining)} groups -> '
                  f'{projection:.0f} s (limit {PROJECTION_LIMIT_S:.0f} s)')
            if projection > PROJECTION_LIMIT_S and remaining:
                stop_resumable = {'projection_s': projection, 'completed': len(group_records),
                                  'remaining': [f'{c}_{y_}_{d_}' for c, y_, d_ in remaining]}
                print('[STOP] 3.5 h projection exceeded; resumable by group')
                break

    # ==================================================================================================================
    #  Readings (only over complete groups of both cells)
    # ==================================================================================================================
    readings, predictions_out, dual_consistency = readings_fn(group_records, groups, cell_records, storage_value,
                                                              bar_sum, w183e, gate)

    # ---- guards, launches, containment ----------------------------------------------------------------------------
    n_solved_ipopt = sum(1 for s in solve_index if s['solver'] == 'ipopt' and not DRY)
    n_solved_gurobi = sum(1 for s in solve_index if s['solver'] == 'gurobi' and not DRY)
    if stop_resumable or stops:
        exp_g, exp_i = n_solved_gurobi, n_solved_ipopt      # a stopped run declares what it reached; recorded
    else:
        exp_g, exp_i = declared_now['gurobi'], declared_now['ipopt']
    pyomo_fail = pyomo_guard.verify(exp_i)
    gurobi_fail = gurobi_guard.verify(exp_g)
    pyomo_guard.uninstall()
    gurobi_guard.uninstall()
    env.dispose()
    guard_out = {'leaf': leaf_site, 'declared_task_total_solves': DECLARED_SOLVES, 'declared_breakdown': DECLARED,
                 'declared_this_process': declared_now, 'expected_at_verify': {'gurobi': exp_g, 'ipopt': exp_i},
                 'counted_events': {'pyomo': dict(pyomo_guard.counts), 'gurobi': dict(gurobi_guard.counts)},
                 'counted_event_total': (pyomo_guard.counts['permitted_solve'] + pyomo_guard.counts['permitted_exec'] +
                                         gurobi_guard.counts['permitted_entry'] +
                                         gurobi_guard.counts['permitted_optimize']),
                 'verify_failures': {'pyomo': pyomo_fail, 'gurobi': gurobi_fail},
                 'process_launch_events': _AUDIT['process_launch_events'],
                 'leaf_static_check': {'names': sorted(leaf_names), 'project_modules_referenced': leaf_project_refs}}
    gate('G-D0-6 leaf guard exact (failures, both guards)', 'process', float(len(pyomo_fail) + len(gurobi_fail)), 0.0)
    if pyomo_fail or gurobi_fail:
        stops.append({'stop': 'leaf-guard count disagrees', 'pyomo': pyomo_fail, 'gurobi': gurobi_fail})
    out_abs, repo_abs = os.path.abspath(OUT_DIR), os.path.abspath(REPO)
    writes_repo = sorted(rel(p) for p in _AUDIT['write_opens'] if p.startswith(repo_abs + os.sep) and not
                         p.startswith(out_abs + os.sep) and '__pycache__' not in p.split(os.sep))
    writes_outside = sorted(p for p in _AUDIT['write_opens'] if not p.startswith(repo_abs + os.sep))
    containment = {'write_opens_in_repo_outside_out_dir': writes_repo, 'write_opens_outside_repo': writes_outside[:50],
                   'n_write_opens_outside_repo': len(writes_outside),
                   'fs_mutations_outside_out_dir': [e for e in _AUDIT['fs_mutations']
                                                    if any(not p.startswith(out_abs) for p in e['paths'])][:50]}
    gate('containment: write-opens in the repository outside OUT_DIR (count)', 'process', float(len(writes_repo)), 0.0)
    repo_modules = {}
    for name_, mod in sorted(sys.modules.items()):
        f_ = getattr(mod, '__file__', None)
        if f_ and f_.startswith(repo_abs + os.sep) and '__pycache__' not in f_:
            repo_modules[name_] = {'file': rel(f_), 'sha256': sha256_file(f_),
                                   't01c_recorded_sha256': del_inv['provenance_file_sha256'].get(rel(f_))}
    import pyomo
    provenance = json.load(open(PROVENANCE_PARENT))
    provenance.update({'interpreter': sys.executable, 'python': sys.version, 'pyomo': pyomo.version.version,
                       'numpy': np.__version__, 'gurobipy': '.'.join(str(v) for v in gp.gurobi.version()),
                       'ipopt_path': _STATE['ipopt_path'], 'project_modules_loaded': repo_modules,
                       'harness_sha256': sha256_file(os.path.abspath(__file__)),
                       'spec_sha256': sha256_file(os.path.join(REPO, 'P5_16_D0_SPEC_v2.md')),
                       'inputs': inputs, 'rehearsal_stand_in': DRY, 'group_limit': GROUP_LIMIT,
                       'peak_rss': rss_peak, 'wall_s': time.time() - t_start})
    gates_out = {'rows': gate_rows, 'all_pass': all(r['pass'] for r in gate_rows),
                 'failed': [r for r in gate_rows if not r['pass']], 'stops': stops, 'stop_triggered': bool(stops),
                 'resumable_stop': stop_resumable}
    payload = {'stage': 'P5.16 D0 / T0.2', 'rehearsal_stand_in': DRY, 'record_schema': RECORD_SCHEMA,
               'harness_version': HARNESS_VERSION,
               'objective_convention': ('gross_operational_cost (settlement-EXCLUDED, salvage NOT subtracted) per '
                                        'group; Q_cert+ = gross with every detector slack at max(slack, 0) (spec '
                                        'section 12 ruling 5); gap = Q_cert+ - Q_LB'),
               'instances': cell_records, 'storage_value_eur': storage_value, 'storage_value_definition':
                   STORAGE_VALUE_NOTE, 'bar_sum_eur': bar_sum,
               'groups': {f'{c}_{y}_{d}': r for (c, y, d), r in group_records.items()},
               'resumed_groups': {f'{c}_{y}_{d}': {'path': v['path'], 'sha256': v['sha256']}
                                  for (c, y, d), v in resumed.items()},
               'separability': separability, 'dual_consistency': dual_consistency, 'readings': readings,
               'predictions': predictions_out, 'gates': gates_out, 'guard': guard_out, 'containment': containment,
               'solve_index': solve_index, 'provenance': provenance}
    unfilled = [q for q, (f_, k_, _fn) in REQUIRED_CAPTURES.items() if f_ == 't02.json' and not payload.get(k_)
                and k_ not in ('dual_consistency',)]
    if unfilled:
        print(f'[capture] unfilled required quantities: {unfilled}')
        payload['capture_unfilled'] = unfilled
    write_json('t02.json', payload, compact=True)
    print('[write] t02.json')
    print(f'[summary] gates {len(gate_rows)} rows, all pass {gates_out["all_pass"]}; failed kinds '
          f'{sorted({r["gate"] for r in gates_out["failed"]})}; stops {len(stops)}; guards {guard_out["counted_events"]}; '
          f'repo writes outside OUT_DIR {len(writes_repo)}; wall {time.time() - t_start:.1f} s')
    for key, value in predictions_out.items():
        print(f'[prediction] {key}: {value.get("verdict")}')
    if stops:
        return 3
    if stop_resumable:
        return 7
    return 0 if gates_out['all_pass'] else 2


# ======================================================================================================================
#  P5.5-C8 cycle basis -- copied VERBATIM from `p55c_c8_tightness.cycle_basis` (importing that module would import
#  `shared_resources_planning`, which another Worker edits concurrently; nothing of production is reimplemented here)
# ======================================================================================================================
def _cycle_basis(network, branch_ids):
    """Fundamental cycles as oriented branch lists, via a spanning forest."""
    parent, parent_edge = {}, {}
    adjacency = {}
    for b in branch_ids:
        branch = network.branches[b]
        f = network.get_node_idx(branch.fbus)
        t = network.get_node_idx(branch.tbus)
        adjacency.setdefault(f, []).append((t, b))
        adjacency.setdefault(t, []).append((f, b))

    seen, tree_edges = set(), set()
    order = []
    for start in list(adjacency):
        if start in seen:
            continue
        seen.add(start)
        parent[start] = None
        stack = [start]
        while stack:
            node = stack.pop()
            order.append(node)
            for other, b in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    parent[other] = node
                    parent_edge[other] = b
                    tree_edges.add(b)
                    stack.append(other)

    def path_to_root(node):
        path = []
        while parent.get(node) is not None:
            path.append((parent[node], node, parent_edge[node]))
            node = parent[node]
        return path, node

    cycles = []
    for b in branch_ids:
        if b in tree_edges:
            continue
        branch = network.branches[b]
        f = network.get_node_idx(branch.fbus)
        t = network.get_node_idx(branch.tbus)
        path_f, root_f = path_to_root(f)
        path_t, root_t = path_to_root(t)
        if root_f != root_t:
            continue
        nodes_f = {f: 0}
        for i, (up, down, _e) in enumerate(path_f, start=1):
            nodes_f[up] = i
        meet, depth_t = None, None
        if t in nodes_f:
            meet, depth_t = t, 0
        else:
            for i, (up, down, _e) in enumerate(path_t, start=1):
                if up in nodes_f:
                    meet, depth_t = up, i
                    break
        if meet is None:
            continue
        # oriented walk: f -> meet up the tree, then meet -> t down the tree,
        # then the closing chord t -> f
        walk = []
        for up, down, e in path_f[:nodes_f[meet]]:
            walk.append((e, down, up))          # traversed from `down` to `up`
        for up, down, e in reversed(path_t[:depth_t]):
            walk.append((e, up, down))
        walk.append((b, t, f))
        cycles.append(walk)
    return cycles


# ======================================================================================================================
#  Readings and predictions (pure functions of the group records)
# ======================================================================================================================
def readings_fn(group_records, groups, cell_records, storage_value, bar_sum, w183e, gate):
    readings, preds, dual = {}, {}, {}
    cells = ('x0', 'unit_n7')
    complete = {c: [g for g in groups if (c, *g) in group_records] for c in cells}
    both = [g for g in groups if all((c, *g) in group_records for c in cells)]
    readings['n_groups_complete'] = {c: len(v) for c, v in complete.items()}
    readings['all_12_groups_both_cells'] = len(both) == 12

    def val(c, g, var, key):
        return group_records[(c, *g)]['variants'][var].get(key)

    def sgap(c, g, var):
        return group_records[(c, *g)]['variants'][var]['solve'].get('solver_gap_abs') or 0.0

    def h_class(r):
        return None if r is None else ('H1' if r <= 1e-4 else ('H2' if r <= 1e-2 else 'H3'))

    sums = {}
    for c in cells:
        q_plus = sum(group_records[(c, *g)]['q_cert_plus_group'] for g in complete[c])
        gross = sum(group_records[(c, *g)]['gross_group'] for g in complete[c])
        sums[c] = {'Q_cert_plus_over_groups': q_plus, 'gross_over_groups': gross}
        for var in ('R', 'A', 'C'):
            gaps = [val(c, g, var, 'gap_abs') for g in complete[c]]
            if gaps and None not in gaps:
                s = sum(gaps)
                sums[c][var] = {'sum_Q_LB': sum(val(c, g, var, 'Q_LB') for g in complete[c]), 'sum_gap_eur': s,
                                'sum_gap_over_Q_cert_plus': s / q_plus,
                                'sum_solver_gap_eur': sum(sgap(c, g, var) for g in complete[c]),
                                'class': ('H1' if s / q_plus <= 1e-4 else ('H2' if s / q_plus <= 1e-2 else 'H3'))}
                # section 16 ruling 1: summed firm interval [sum(gap - res_LB - res_cert), sum(gap + res_cert)] and
                # summed point estimate sum(gap - res_LB); class on the raw gap (`class`), on the point estimate and on
                # the firm upper end (conservative, `class_firm_upper`)
                rd = {k: [val(c, g, var, k) for g in complete[c]]
                      for k in ('gap_firm_lower', 'gap_firm_upper', 'gap_point_estimate')}
                n_missing = sum(1 for e_ in rd['gap_firm_upper'] if e_ is None)
                fs = {k: (None if n_missing else sum(v_)) for k, v_ in rd.items()}
                sums[c][var].update({
                    'sum_res_LB_eur': None if n_missing else sum(val(c, g, var, 'res_LB_eur') for g in complete[c]),
                    'sum_res_cert_eur': None if n_missing else sum(val(c, g, var, 'res_cert_eur')
                                                                   for g in complete[c]),
                    'n_groups_without_resolution': n_missing,
                    'sum_gap_firm_lower_eur': fs['gap_firm_lower'], 'sum_gap_firm_upper_eur': fs['gap_firm_upper'],
                    'sum_gap_point_estimate_eur': fs['gap_point_estimate'],
                    **{f'sum_{k}_over_Q_cert_plus': (None if fs[k] is None else fs[k] / q_plus) for k in fs},
                    'class_point_estimate': (None if fs['gap_point_estimate'] is None
                                             else h_class(fs['gap_point_estimate'] / q_plus)),
                    'class_firm_upper': (None if fs['gap_firm_upper'] is None
                                         else h_class(fs['gap_firm_upper'] / q_plus))})
    readings['per_cell'] = sums
    readings['absolute_class_on_C'] = {c: sums[c].get('C', {}).get('class') for c in cells}
    readings['absolute_class_on_C_firm_upper'] = {c: sums[c].get('C', {}).get('class_firm_upper') for c in cells}
    readings['absolute_class_on_C_point_estimate'] = {c: sums[c].get('C', {}).get('class_point_estimate')
                                                      for c in cells}
    readings['R4'] = {c: {'G_keur_C': sums[c]['C']['sum_gap_eur'] / 1e3 if 'C' in sums[c] else None,
                          'g_C': sums[c]['C']['sum_gap_over_Q_cert_plus'] if 'C' in sums[c] else None,
                          'stakes_storage_value_keur': storage_value / 1e3, 'bar_sum_keur': bar_sum / 1e3}
                      for c in cells}

    def p6_firm(s_u, s_0):
        """section 16 ruling 1 (OPERATIONALISATION 'P6_firm'): the conservative D read at the firm upper ends
        (gap + res_cert), and the interval-arithmetic D range of the firm intervals beside it (informative)."""
        if s_u.get('sum_gap_firm_upper_eur') is None or s_0.get('sum_gap_firm_upper_eur') is None:
            return {'D_firm_upper_eur': None, 'D_firm_upper_abs_eur': None, 'held_at_firm_upper': None,
                    'D_firm_interval_range_eur_informative': None}
        d_up = s_u['sum_gap_firm_upper_eur'] - s_0['sum_gap_firm_upper_eur']
        d_rng = [s_u['sum_gap_firm_lower_eur'] - s_0['sum_gap_firm_upper_eur'],
                 s_u['sum_gap_firm_upper_eur'] - s_0['sum_gap_firm_lower_eur']]
        return {'D_firm_upper_eur': d_up, 'D_firm_upper_abs_eur': abs(d_up),
                'held_at_firm_upper': bool(abs(d_up) <= bar_sum), 'D_firm_interval_range_eur_informative': d_rng}

    p6 = {}
    if readings['all_12_groups_both_cells']:
        for var in ('R', 'A', 'C'):
            if var in sums['x0'] and var in sums['unit_n7']:
                diff = sums['unit_n7'][var]['sum_gap_eur'] - sums['x0'][var]['sum_gap_eur']
                res = bar_sum + sums['unit_n7'][var]['sum_solver_gap_eur'] + sums['x0'][var]['sum_solver_gap_eur']
                p6[var] = {'sum_gap_unit_minus_sum_gap_x0_eur': diff, 'D_abs_eur': abs(diff), 'bar_sum_eur': bar_sum,
                           'resolution_eur_bar_sum_plus_solver_gaps': res,
                           'held_D_le_bar_sum': bool(abs(diff) <= bar_sum),
                           'D_over_storage_value': abs(diff) / storage_value,
                           'point_estimate_le_5keur': bool(abs(diff) <= 5000.0),
                           **p6_firm(sums['unit_n7'][var], sums['x0'][var]),
                           'role': {'A': 'mechanism (D-3); A is NOT a bound', 'C': 'decision relevance (D-3)',
                                    'R': 'reported; contains the plan-dependent degradation relaxation'}[var]}
    readings['P6'] = p6
    # ---- dual consistency (f) ----
    w_rank = {(r['year'], r['day']): r for r in w183e['day_selection']['ranking']}
    rows = []
    for g in both:
        y, d = g
        ru = group_records[('unit_n7', *g)]
        for var in ('R', 'C'):
            cs = ru.get('capacity_sensitivity', {}).get(var)
            q0, qu = val('x0', g, var, 'Q_LB'), val('unit_n7', g, var, 'Q_LB')
            if cs is None or not cs.get('available') or q0 is None or qu is None:
                continue
            dq = q0 - qu
            est = cs['estimate_delta_Q']
            w = w_rank.get((str(y), d))
            rows.append({'group': f'{y}_{d}', 'variant': var, 'delta_Q_LB': dq, 'dual_estimate': est,
                         'dual_E_per_MWh': cs['dual_E_per_MWh'], 'dual_S_per_MVA': cs['dual_S_per_MVA'],
                         'ratio_estimate_over_delta': est / dq if dq else None,
                         'sign_duals_negative': bool(cs['dual_E_per_MWh'] <= 0 and cs['dual_S_per_MVA'] <= 0),
                         'within_factor_3': bool(dq != 0 and est / dq > 0 and 1 / 3 <= est / dq <= 3),
                         'w183e_c_eur': w['value_contribution_Q0_minus_Qx'] if w else None,
                         'delta_Q_LB_over_w183e_c': (dq / w['value_contribution_Q0_minus_Qx']) if w else None})
    dual['rows'] = rows
    dual['w183e_source'] = {'path': W183E_EXPORT, 'sha256': W183E_EXPORT_SHA256,
                            'note': ('W183e c = Q_row1(x0, W101 continuation cycle 181) - Q_row2(unit, cycle 172) per '
                                     'block; same candidates as the W86 cells, different certified cells')}
    # ---- predictions ----
    def verdict(held):
        return None if held is None else ('HELD' if held else 'FAILED')

    def all_or_none(vals):
        vals = [v for v in vals if v is not None]
        return None if not vals else all(vals)

    def dso_max(var):
        out = []
        for (c, y, d), r in group_records.items():
            rk = r['variants'][var].get('rank') or {}
            out += [v['max_rel_rank_residual'] for k, v in rk.items() if k.startswith('DSO')]
        return max(out) if out else None
    p1 = {v: dso_max(v) for v in ('R', 'A', 'C')}

    def dso_tr_excluded(var):
        mx, branches = [], set()
        for (c, y, d), r in group_records.items():
            for k, v in (r['variants'][var].get('rank') or {}).items():
                if k.startswith('DSO'):
                    branches |= {f'{k}:{b}' for b in v.get('transformer_branches_excluded', [])}
                    t_ = (v.get('transformer_d0_cone_excluded') or {}).get('max_rel')
                    if t_ is not None:
                        mx.append(t_)
        return {'max_rel_excluded_d0_cone': max(mx) if mx else None, 'branches': sorted(branches)}
    preds['P1'] = {'achieved_max_dso_rel_rank_residual': p1,
                   'transformer_d0_cone_excluded': {v: dso_tr_excluded(v) for v in ('R', 'A', 'C')},
                   'reading': OPERATIONALISATION['P1_transformer_exclusion'],
                   'verdict': verdict(all_or_none([(x <= 1e-5) if x is not None else None for x in p1.values()]))}
    fm = [(f'{y}_{d}', group_records[('unit_n7', y, d)].get('fixed_mode', {}).get('cost_pct_of_group_gross'))
          for (y, d) in complete['unit_n7']]
    preds['P2'] = {'per_group_pct': dict(fm), 'max_pct': max((v for _g, v in fm if v is not None), default=None),
                   'verdict': verdict(all_or_none([(v <= 0.05) if v is not None else None for _g, v in fm]))}
    preds['P3'] = {'sum_gap_over_Q_R': {c: sums[c].get('R', {}).get('sum_gap_over_Q_cert_plus') for c in cells},
                   'class_R': {c: sums[c].get('R', {}).get('class') for c in cells},
                   'class_C': readings['absolute_class_on_C'],
                   'class_R_firm_upper': {c: sums[c].get('R', {}).get('class_firm_upper') for c in cells},
                   'class_C_firm_upper': readings['absolute_class_on_C_firm_upper'],
                   'verdict': verdict(all_or_none([(sums[c]['R']['class'] == 'H2') if 'R' in sums[c] else None
                                                   for c in cells]))}
    lp_duals = [not r['variants'][v]['solve'].get('dual_less', True) for r in group_records.values()
                for v in ('R', 'A', 'C') if 'solve' in r['variants'][v]]
    all_g = lp_duals + [not r[k]['solve'].get('dual_less', True) for r in group_records.values()
                        for k in ('fixed_mode',) if k in r] + \
        [not r['freeze'][f]['solve'].get('dual_less', True) for r in group_records.values() if 'freeze' in r
         for f in ('TN_W_frozen', 'DSO_ESSO_frozen')]
    preds['P4'] = {'group_lps_with_duals': f'{sum(lp_duals)} / {len(lp_duals)}',
                   'all_gurobi_with_duals': f'{sum(all_g)} / {len(all_g)}',
                   'verdict': verdict(all(lp_duals) if lp_duals else None)}
    p5 = [r for r in rows if r['variant'] == 'C']
    preds['P5'] = {'rows_C': p5, 'verdict': verdict(all_or_none([r['sign_duals_negative'] and r['within_factor_3']
                                                                 for r in p5]))}
    preds['P6'] = {'per_variant': p6, 'verdict_A': verdict(p6['A']['held_D_le_bar_sum']) if 'A' in p6 else None,
                   'verdict_C': verdict(p6['C']['held_D_le_bar_sum']) if 'C' in p6 else None,
                   'verdict_R': verdict(p6['R']['held_D_le_bar_sum']) if 'R' in p6 else None,
                   'verdict': verdict(p6['C']['held_D_le_bar_sum']) if 'C' in p6 else None,
                   **{f'verdict_{v_}_at_firm_upper': (verdict(p6[v_]['held_at_firm_upper'])
                                                      if v_ in p6 else None) for v_ in ('A', 'C', 'R')}}
    p7 = {}
    for var in ('R', 'C'):
        p7[var] = []
        for g in both:
            q0, qu = val('x0', g, var, 'Q_LB'), val('unit_n7', g, var, 'Q_LB')
            if q0 is None or qu is None:
                continue
            tol = sgap('x0', g, var) + sgap('unit_n7', g, var)
            p7[var].append({'group': f'{g[0]}_{g[1]}', 'Q_LB_unit_minus_x0': qu - q0, 'tol': tol,
                            'held': bool(qu <= q0 + tol)})
            gate('P7 Q_LB(unit) - Q_LB(x0) - tolerance (<= 0)', f'{g[0]}_{g[1]} {var}', qu - q0 - tol, 0.0,
                 note='prediction P7 recorded as a gate (G-D0-4 family)')
    preds['P7'] = {'rows': p7, 'verdict': verdict(all_or_none([r['held'] for v in p7.values() for r in v]))}
    oltc = [v['oltc']['max_transformer_cone_rel_residual'] for r in group_records.values()
            for var in ('R', 'A', 'C') for k, v in (r['variants'][var].get('rank') or {}).items() if 'oltc' in v]
    preds['P8'] = {'max_oltc_envelope_slack': max(oltc) if oltc else None,
                   'verdict': verdict((max(oltc) <= 1e-5) if oltc else None)}
    tn = [r['variants'][var]['rank']['TSO'] for r in group_records.values() for var in ('R', 'A', 'C')
          if r['variants'][var].get('rank')]
    tn_rank = max((t['max_rel_rank_residual'] for t in tn), default=None)
    tn_cyc = max((t.get('max_cycle_angle_residual_rad', 0.0) for t in tn), default=None)
    preds['P9'] = {'max_tn_rel_rank_residual': tn_rank, 'max_cycle_angle_residual_rad': tn_cyc,
                   'predicted': {'rank': 2.45e-2, 'cycle_rad': 1.2e-2},
                   'verdict': verdict(None if tn_rank is None else (0.5 <= tn_rank / 2.45e-2 <= 2 and
                                                                    0.5 <= tn_cyc / 1.2e-2 <= 2))}
    e1 = []
    for g in complete['x0']:
        q = {v: val('x0', g, v, 'Q_LB') for v in ('R', 'A', 'C')}
        if None not in q.values():
            e1.append({'group': f'{g[0]}_{g[1]}', 'R_minus_A': q['R'] - q['A'], 'R_minus_C': q['R'] - q['C'],
                       'identical': q['R'] == q['A'] == q['C']})
    preds['E1'] = {'rows': e1, 'max_abs_difference': max((max(abs(r['R_minus_A']), abs(r['R_minus_C'])) for r in e1),
                                                         default=None),
                   'verdict': verdict(all(r['identical'] for r in e1) if e1 else None)}
    e2 = []
    for g in complete['unit_n7']:
        gp_ = {v: val('unit_n7', g, v, 'gap_abs') for v in ('R', 'A', 'C')}
        if None in gp_.values():
            continue
        tol_ac = sgap('unit_n7', g, 'A') + sgap('unit_n7', g, 'C')
        tol_cr = sgap('unit_n7', g, 'C') + sgap('unit_n7', g, 'R')
        e2.append({'group': f'{g[0]}_{g[1]}', **{f'gap_{v}': gp_[v] for v in gp_},
                   'held': bool(gp_['A'] <= gp_['C'] + tol_ac and gp_['C'] <= gp_['R'] + tol_cr)})
    preds['E2'] = {'rows': e2, 'verdict': verdict(all(r['held'] for r in e2) if e2 else None)}
    if all(v in sums['unit_n7'] for v in ('R', 'A', 'C')) and len(complete['unit_n7']) == 12:
        rc = sums['unit_n7']['R']['sum_gap_eur'] - sums['unit_n7']['C']['sum_gap_eur']
        ca = sums['unit_n7']['C']['sum_gap_eur'] - sums['unit_n7']['A']['sum_gap_eur']
        preds['E3'] = {'sum_gap_R_minus_C_eur': rc, 'verdict': verdict(10000.0 <= rc <= 50000.0)}
        preds['E4'] = {'sum_gap_C_minus_A_eur': ca, 'verdict': verdict(5000.0 <= ca <= 40000.0)}
    else:
        preds['E3'] = {'verdict': None}
        preds['E4'] = {'verdict': None}
    if 'A' in p6 and 'R' in p6:
        holds_a = p6['A']['held_D_le_bar_sum']
        fails_r = (not p6['R']['held_D_le_bar_sum']) and (p6['R']['D_abs_eur'] - bar_sum >= 15000.0)
        preds['E5'] = {'P6_holds_on_A': holds_a, 'P6_fails_on_R_by_ge_15k': fails_r, 'D_R': p6['R']['D_abs_eur'],
                       'D_A': p6['A']['D_abs_eur'], 'verdict': verdict(holds_a and fails_r)}
    else:
        preds['E5'] = {'verdict': None}
    for k_, text in PREDICTIONS.items():
        preds.setdefault(k_, {})['prediction'] = text
    return readings, preds, dual


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        sys.exit(child())
    sys.exit(parent())
