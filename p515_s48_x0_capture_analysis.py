"""
P5.15 Addendum 32 (task W27, Q4 step 2) -- what sets the TSO bus-7 marginal cost at x = 0: ZERO SOLVES, read-only.

Armed `SolveProfileGuard(permitted=())` is installed before any Pyomo / production import; `verify(0)` is checked
at the end (exact count: 0 solves, 0 solver launches). Nothing is built or solved: the certified TSO models
persisted by the s48_x0_capture evaluation (`certified_models.pkl`, post-certification persistence) are unpickled
and their terminal IPOPT primal / dual / bound-multiplier values are read.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 32; frozen spec v18 `Q4_terminal_tso_capture`; Planner task W27.

INSTANCE: x = 0 (candidate key 8435c718..., nodes 5/7/9 empty, 2025 canonical), campaign s48_x0_capture, eval key
d2c96b14... (baseline ageing declaration); the analysis refuses unless the reproduction gate
(`p515_s48_x0_capture_gate.py`, data/SRP1/Results/P515S48/x0_capture_gate/gate.json) PASSED, i.e. the analysed
models belong to an evaluation bitwise identical in trajectory and cost to A0 x0.
W28 (the ONLY change to the W27 script): the gate output reads gate_pass = False (strict FAIL on echoed declared
ageing parameters) and stays so; the precondition accepts it together with the committed Planner ruling
P5_15_S48_X0_CAPTURE_GATE_RULING.md (8f9d67f2, ruled PASS, Step 2 authorized), verified committed and unmodified,
naming this run (s48_x0_capture, spec 4a50c0e2, evidence commit c23a6cd8, in which gate.json is committed and
unmodified). Both the strict gate result and the ruling are recorded in the output. The method is unchanged.

METHOD (formulas are also written into the JSON and the Markdown)
 1. Stationarity (KKT) of each TSO block, recomputed from the model: for every free variable v of the active
    objective / constraints, r_v = df/dv - sum_c lambda_c dc/dv - zL_v - zU_v, with lambda = model.dual,
    zL = ipopt_zL_out (>= 0), zU = ipopt_zU_out (<= 0) (the Pyomo/IPOPT convention, established on these models:
    it is the only sign combination with r ~ 0). Derivatives: pyomo reverse-mode numeric differentiation at the
    stored point (no solve). max|r| is reported per block.
 2. Bus-7 marginal cost LMP7 = lambda(node_balance_p[bus 7]) / baseMVA [EUR/MWh]; sign: the balance row is
    Pg - Pd - Pflow = 0, so lambda = +d(objective)/d(load at the bus): the cost of serving one more MWh at bus 7.
    Cross-checked hour by hour against W25's identity series (b7aca555).
    Market price pi = df/dpg / baseMVA of the CONV generators (all three must agree), cross-checked against W25.
 3. Generator status per hour: pg / qg against their bounds (at-bound when within 1e-4 p.u.), the bound multipliers,
    sg_capability / power-factor rows; a generator is MARGINAL when its pg is strictly inside its bounds and no
    capability row binds -- its bus marginal cost then equals its own marginal cost (pi for CONV; 0 for WIND/PV,
    whose curtailment penalty is 0 in these models: penalty_gen_curtailment read from each block).
 4. EXACT DECOMPOSITION of LMP7 - pi (per hour; reference bus = the case9 reference bus 1):
    the stationarity equations of the NETWORK variables N = {e, f, vmag, vmag_sqr, voltage_product_real,
    voltage_product_imag, slack_v_sqr_*} are linear in the multipliers:
        M lambda_U = sum_g rhs_g,     M[v, c] = dc/dv,
    U = {node_balance_p, node_balance_q, voltage_mag_def, voltage_mag_sqr_def, voltage_product_real_def,
    voltage_product_imag_def} (the balance rows and the definitional equalities among N), and each rhs_g collects
    one group of KNOWN terms: the objective gradient on N, the multipliers of every other row touching N (branch
    flow limits, voltage-magnitude bounds, voltage set-points, angle rows, the ADMM interface-voltage definition),
    and the bound multipliers on N. M has a one-dimensional null space z (rotational invariance of the AC
    equations); with z normalised to z[P, ref] = 1 and each group solved with lambda_P,ref = 0,
        lambda_U = y_ref * z + sum_g x_g     (exact: the IPOPT multipliers satisfy M lambda_U = sum_g rhs_g).
    Hence, per hour, at bus 7 (divide by baseMVA):
        LMP7 - pi = (y_ref/B - pi)                  reference-generator limit (0 when gen 1 is interior)
                  + (y_ref/B) (z_7 - 1)             losses (marginal loss factor of bus 7 relative to bus 1)
                  + sum_g x_g[7]/B                  congestion (branch flow limits), voltage bounds, voltage
                                                    set-points, angle rows, ADMM interface voltage, other
                  + residual                        (KKT tolerance; reported)
    Exactness: exact at the KKT point (residual reported). The split is reference-bus dependent (as every LMP
    decomposition): energy / loss / congestion are defined relative to bus 1; the total is not.
    Per-row attribution: every row with |lambda| >= DUAL_TOL is also solved alone -> its own share of LMP7.
 5. Spread ("flatness") decomposition per (year, day): T, B = the 4 highest / 4 lowest hours of LMP7 (ties by
    hour); market spread S_pi = top-4 minus bottom-4 of pi (its own hours);
        S_LMP = S_pi - H + sum_c [mean_T(c) - mean_B(c)],  H = S_pi - [mean_T(pi) - mean_B(pi)] >= 0 (hour
    selection), c over the components of 4; flatness = S_pi - S_LMP = H - sum_c [...]. Aggregates are day-weighted
    with W25's block weights (w = num_years * num_days, undiscounted and r2).

SCOPE: the 2030 blocks in full detail (spec v18 Q4 / W27); the spread decomposition for all 12 blocks as a
supplementary table (same method, same models).

Output (write-once): data/SRP1/Results/P515S48/x0_capture_analysis/{x0_capture_analysis.json, .md,
manifest_sha256.json, launch.log (shell)}.
EXACT COMMAND (repo root; attached; both streams; noclobber):
    mkdir data/SRP1/Results/P515S48/x0_capture_analysis && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s48_x0_capture_analysis.py \\
        > data/SRP1/Results/P515S48/x0_capture_analysis/launch.log 2>&1
TEST MODE (`--test-pickle PATH --out-dir DIR`, DIR outside the repo): the same analysis on another persisted
pickle, with the campaign / gate inputs not required (development only; never the evidence).
"""
import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W27 x0 capture analysis (zero solves)').install()

import numpy as np  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.common.collections import ComponentSet  # noqa: E402
from pyomo.core.expr.calculus.derivatives import Modes, differentiate  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402

STAGE = 'P5.15 Addendum 32 W27 -- Q4: what sets the TSO bus-7 marginal cost at x = 0 (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 32',
             'data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json Q4_terminal_tso_capture',
             'Planner task W27 step 2']
_P48 = os.path.join('data', 'SRP1', 'Results', 'P515S48')
OUT_REL = os.path.join(_P48, 'x0_capture_analysis')
CAMPAIGN_ROOT_REL = os.path.join(_P48, 'x0_capture')
CAMPAIGN_SPEC_SHA256 = '4a50c0e2277d937409f7471f8361268fea6e36254c1e21bdcf1e6ab8b1dd6a61'
GATE_REL = os.path.join(_P48, 'x0_capture_gate', 'gate.json')
GATE_COMMIT = 'c23a6cd8'           # W28: the evidence commit in which gate.json was committed (strict gate_pass False)
RULING = {'path': 'P5_15_S48_X0_CAPTURE_GATE_RULING.md', 'commit': '8f9d67f2',
          'required_text': ('**PASS.**', 's48_x0_capture', '4a50c0e2', 'c23a6cd8',
                            'Step 2 (the zero-solve bus-7 analysis) is authorized')}
X0_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
W25_JSON = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'tso_marginal_cost', 'tso_marginal_cost.json'),
            'commit': 'b7aca555'}
CASE9_FMT = os.path.join('data', 'SRP1', 'case9', 'case9_{year}.json')
YEARS = (2025, 2030, 2035)
DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')
DETAIL_YEAR = 2030
NODE = 7
ADN_BUSES = (5, 7, 9)
H = 4
N_PERIODS = 24

# Tolerances, stated before the committed run.
GEN_BOUND_TOL_PU = 1e-4        # pg / qg "at a bound" when within 1e-4 p.u. (0.01 MW) of it
DUAL_TOL = 1e-2                # a row's multiplier is "material" when |lambda| >= 1e-2 (raw; = 1e-4 EUR/MWh for a
                               # p.u. power row) -- rows below it are also included in their group, never dropped
CONTRIB_REPORT_TOL = 1e-2      # per-row LMP7 shares listed when |share| >= 0.01 EUR/MWh
W25_LMP_TOL = 1e-3             # |LMP7 (dual) - LMP7 (W25 identity)| per hour, EUR/MWh
PRICE_TOL = 1e-9               # |pi (objective gradient) - pi (W25 / Z4 arrays)|, EUR/MWh
RECON_TOL = 1e-6               # |sum of components - (LMP7 - pi)|, EUR/MWh
RECON_MULT_TOL = 1e-2          # max |lambda_U(reconstructed) - lambda_U(IPOPT)| raw over ALL U rows of an hour: the
                               # multipliers satisfy the network stationarity only to the KKT residual (same scale as
                               # KKT_WARN); set on development data (s44 pickle, max 3.5e-4) before the real run
KKT_WARN = 1e-2                # raw stationarity residual above which a block is flagged
NULL_RATIO_MAX = 1e-9          # sigma_min / sigma_max of M for the rotational null space
NULL_GAP_MIN = 1e-7            # sigma_(n-1) / sigma_max: exactly one null direction

NET_FAMILIES = ('e', 'f', 'vmag', 'vmag_sqr', 'voltage_product_real', 'voltage_product_imag', 'slack_v_sqr_down',
                'slack_v_sqr_up')
U_FAMILIES = ('node_balance_p', 'node_balance_q', 'voltage_mag_def', 'voltage_mag_sqr_def', 'voltage_product_real_def',
              'voltage_product_imag_def')
ROW_GROUP = {'branch_flow_limit': 'congestion', 'branch_flow_limit_ji': 'congestion',
             'voltage_magnitude_lower_cons': 'voltage_bounds', 'voltage_magnitude_upper_cons': 'voltage_bounds',
             'voltage_setpoint_cons': 'voltage_setpoint', 'voltage_product_real_nonnegative': 'angle',
             'branch_angle_difference_lower': 'angle', 'branch_angle_difference_upper': 'angle',
             'expected_interface_vmag_def': 'admm_interface_voltage'}
BOUND_GROUP = {'e': 'voltage_bounds', 'f': 'voltage_bounds', 'vmag': 'voltage_bounds', 'vmag_sqr': 'voltage_bounds',
               'slack_v_sqr_down': 'voltage_bounds', 'slack_v_sqr_up': 'voltage_bounds',
               'voltage_product_real': 'angle', 'voltage_product_imag': 'angle'}
OBJ_GROUP = {'slack_v_sqr_down': 'voltage_bounds', 'slack_v_sqr_up': 'voltage_bounds'}
COMPONENTS = ('reference_generator_limit', 'losses', 'congestion', 'voltage_bounds', 'voltage_setpoint', 'angle',
              'admm_interface_voltage', 'other', 'residual')
COMPONENT_TEXT = {
    'reference_generator_limit': ('y_ref/B - pi: the bus-1 marginal cost minus the market price; = -(zL + zU)/B of '
                                  'pg[gen 1] (plus its capability rows): 0 when gen 1 is interior; negative when gen 1 '
                                  'sits at pmin = 0, i.e. the TN would take less conventional output than 0'),
    'losses': '(y_ref/B)(z_7 - 1): marginal losses of serving bus 7 from bus 1',
    'congestion': 'branch flow limits (branch_flow_limit rows)',
    'voltage_bounds': 'voltage-magnitude bounds (rows, slacks and their penalty, e/f/vmag bounds)',
    'voltage_setpoint': 'PV-bus voltage set-point rows',
    'angle': 'angle rows (voltage_product_real >= 0, angle-difference rows) and their bounds',
    'admm_interface_voltage': 'ADMM interface-voltage definition rows (expected_interface_vmag_def)',
    'other': 'any other row / objective term / bound touching the network variables',
    'residual': '(LMP7 - pi) minus the sum of the components above (KKT tolerance)',
}
SPREAD_FORMULA = ('T, B = the 4 highest / 4 lowest hours of LMP7 (ties by hour); S_pi = mean of the 4 highest pi - mean '
                  'of the 4 lowest pi (own hours); S_LMP = mean_T(LMP7) - mean_B(LMP7); H = S_pi - [mean_T(pi) - '
                  'mean_B(pi)] >= 0 (hour selection); for each component c of LMP7 - pi: D_c = mean_T(c) - mean_B(c); '
                  'identity S_LMP = S_pi - H + sum_c D_c; flatness = S_pi - S_LMP = H - sum_c D_c.')
DECOMP_FORMULA = ('LMP7 - pi = (y_ref/B - pi) + (y_ref/B)(z_7 - 1) + sum_g x_g[P,7]/B + residual, where M lambda_U = '
                  'sum_g rhs_g is the stationarity of the network variables N in the balance / definitional '
                  'multipliers U, z spans null(M) with z[P,ref] = 1, and x_g solves M x = rhs_g with x[P,ref] = 0 '
                  '(least squares on the stacked system; exact when rhs_g is consistent, residuals reported). '
                  'Reference bus = case9 bus 1 (type 3).')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _load_json(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _fam(component_data):
    return component_data.parent_component().name


def _period(component_data):
    idx = component_data.index()
    return idx[-1] if isinstance(idx, tuple) else None


# ======================================================================================================================
#  inputs and the capture-path assertion (rule eleven, before any analysis)
# ======================================================================================================================
def resolve_inputs(test_pickle):
    failures, info = [], {}
    if test_pickle:
        info['mode'] = 'TEST (not evidence)'
        info['pickle'] = {'path': os.path.abspath(test_pickle), 'sha256': _sha(test_pickle)}
        return info, failures
    info['mode'] = 'real'
    root = os.path.join(REPO, CAMPAIGN_ROOT_REL)
    results = _load_json(os.path.join(CAMPAIGN_ROOT_REL, 'campaign_results.json'))
    point = results.get('point') or {}
    persisted = point.get('persisted_models') or {}
    spec_files = [f for f in os.listdir(root) if f.startswith('campaign_spec_')]
    spec_sha = _sha(os.path.join(root, spec_files[0])) if len(spec_files) == 1 else None
    gate = _load_json(GATE_REL) if os.path.isfile(os.path.join(REPO, GATE_REL)) else {}
    ruling = ruling_check(gate)
    ppath = os.path.join(REPO, persisted.get('path') or '__missing__')
    got = _sha(ppath) if os.path.isfile(ppath) else None
    checks = {
        'campaign_spec_is_the_frozen_one': spec_sha == CAMPAIGN_SPEC_SHA256 == results.get('campaign_spec_sha256'),
        'point_certified': point.get('status') == 'certified',
        'point_is_x0': point.get('candidate_key') == X0_KEY,
        'gate_json_present': bool(gate),
        'gate_passed_or_ruled_pass_by_committed_ruling': gate.get('gate_pass') is True or ruling['ruled_pass_ok'],
        'gate_candidate_is_this_point': (gate.get('inputs') or {}).get('candidate_dir') == point.get('eval_dir'),
        'persisted_pickle_present': got is not None,
        'persisted_pickle_sha256_matches_record': got == persisted.get('sha256'),
        'gate_verified_same_pickle_sha256': (((gate.get('comparison') or {}).get('persisted_models') or {})
                                             .get('sha256_on_disk')) == got,
        'w25_json_present_and_committed': (os.path.isfile(os.path.join(REPO, W25_JSON['path']))
                                           and not _git(['status', '--porcelain', '--', W25_JSON['path']])
                                           and subprocess.run(['git', 'merge-base', '--is-ancestor', W25_JSON['commit'],
                                                               'HEAD'], cwd=REPO).returncode == 0),
    }
    failures += [f'input check failed: {k}' for k, v in checks.items() if not v]
    info.update({'campaign_results': os.path.join(CAMPAIGN_ROOT_REL, 'campaign_results.json'),
                 'campaign_results_sha256': _sha(os.path.join(root, 'campaign_results.json')),
                 'campaign_spec_sha256': spec_sha, 'eval_dir': point.get('eval_dir'), 'eval_key': point.get('eval_key'),
                 'candidate_key': point.get('candidate_key'), 'certified_cost_gross': point.get(
                     'certified_cost_gross_settlement_excluded'), 'cycles_run': point.get('cycles_run'),
                 'gate': {'path': GATE_REL, 'sha256': _sha(os.path.join(REPO, GATE_REL)) if gate else None,
                          'gate_pass': gate.get('gate_pass'), 'committed_in': GATE_COMMIT},
                 'gate_ruling': ruling,
                 'pickle': {'path': persisted.get('path'), 'sha256': got, 'size_bytes': persisted.get('size_bytes'),
                            'committed': False, 'note': 'hash-recorded, not committed (W27)'},
                 'checks': checks})
    return info, failures


def ruling_check(gate):
    """W28: the strict gate output (gate_pass False) is accepted only together with the committed Planner ruling."""
    rpath = os.path.join(REPO, RULING['path'])
    text = open(rpath).read() if os.path.isfile(rpath) else ''
    blob_head = lambda commit, rel: _git(['rev-parse', f'{commit}:{rel}'])  # noqa: E731
    items = {
        'ruling_file_present': bool(text),
        'ruling_committed_in_named_commit_and_unmodified': bool(text) and (
            blob_head(RULING['commit'], RULING['path']) == _git(['hash-object', RULING['path']])
            and not _git(['status', '--porcelain', '--', RULING['path']])),
        'ruling_commit_is_ancestor_of_head': subprocess.run(['git', 'merge-base', '--is-ancestor', RULING['commit'],
                                                             'HEAD'], cwd=REPO).returncode == 0,
        'ruling_names_run_spec_evidence_and_rules_pass': all(t in text for t in RULING['required_text']),
        'gate_json_committed_in_evidence_commit_and_unmodified': (
            blob_head(GATE_COMMIT, GATE_REL) == _git(['hash-object', GATE_REL])
            and not _git(['status', '--porcelain', '--', GATE_REL])),
        'gate_spec_is_the_named_one': str((gate.get('inputs') or {}).get('campaign_spec_sha256', '')).startswith('4a50c0e2'),
    }
    return {'path': RULING['path'], 'commit': RULING['commit'], 'sha256': _sha(rpath) if text else None,
            'required_text': list(RULING['required_text']), 'items': items,
            'strict_gate_pass_in_gate_json': gate.get('gate_pass'),
            'ruled_pass_ok': all(items.values()),
            'note': ('W28: gate.json keeps its strict FAIL (not overridden); the committed ruling rules it PASS '
                     '(declared ageing parameters echoed into diagnostic files; no solved quantity differs)')}


def capture_path_checklist(payload, w25):
    """Every quantity the W27 analysis reports, asserted present BEFORE the analysis (fail fast)."""
    items = {}
    tso = payload.get('tso') or {}
    for y in YEARS:
        for d in DAYS:
            m = (tso.get(y) or {}).get(d)
            key = f'{y}_{d}'
            items[f'tso_block_{key}'] = m is not None
            if m is None:
                continue
            items[f'{key}_one_active_objective'] = len(list(m.component_data_objects(pe.Objective, active=True))) == 1
            nb = list(m.node_balance_p.values())
            items[f'{key}_node_balance_p_rows_9x24'] = len(nb) == 9 * N_PERIODS
            items[f'{key}_node_balance_p_duals_all_present'] = all(c in m.dual for c in nb)
            items[f'{key}_bound_multiplier_suffixes_nonempty'] = len(m.ipopt_zL_out) > 0 and len(m.ipopt_zU_out) > 0
            for name in ('pg', 'qg', 'interface_delta_p', 'interface_delta_q', 'pc', 'qc', 'e', 'f', 'vmag_sqr',
                         'voltage_product_real', 'voltage_product_imag', 'branch_flow_limit', 'flow_ij_sqr',
                         'voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons', 'penalty_gen_curtailment'):
                items[f'{key}_has_{name}'] = hasattr(m, name)
            wd = {(str(r['year']), r['day']): r for r in (w25.get('per_day') or [])}.get((str(y), d)) or {}
            items[f'{key}_w25_lmp_and_price_series'] = (len(wd.get('lmp_series') or []) == N_PERIODS
                                                         and len(wd.get('price_series') or []) == N_PERIODS)
            items[f'{key}_w25_weights'] = wd.get('w_undisc') is not None and wd.get('w_r2') is not None
    for y in YEARS:
        items[f'case9_{y}_file'] = os.path.isfile(os.path.join(REPO, CASE9_FMT.format(year=y)))
    return items


# ======================================================================================================================
#  KKT machinery (numeric reverse-mode derivatives at the stored point)
# ======================================================================================================================
def block_kkt(m):
    obj = next(iter(m.component_data_objects(pe.Objective, active=True)))
    cons = [c for c in m.component_data_objects(pe.Constraint, active=True)]
    varset = ComponentSet()
    rows = []
    for c in cons:
        vs = list(identify_variables(c.body, include_fixed=False))
        for v in vs:
            varset.add(v)
        rows.append((c, vs))
    ovars = list(identify_variables(obj.expr, include_fixed=False))
    for v in ovars:
        varset.add(v)
    variables = list(varset)
    vidx = {id(v): i for i, v in enumerate(variables)}
    gf = np.zeros(len(variables))
    for v, dv in zip(ovars, differentiate(obj.expr, wrt_list=ovars, mode=Modes.reverse_numeric)):
        gf[vidx[id(v)]] += dv
    jac = []          # per row: (con, lambda, [(var index, derivative)])
    jt_lam = np.zeros(len(variables))
    n_missing_dual = 0
    for c, vs in rows:
        if not vs:
            continue
        if c not in m.dual:
            n_missing_dual += 1
        lam = float(m.dual.get(c, 0.0))
        ders = differentiate(c.body, wrt_list=vs, mode=Modes.reverse_numeric)
        entries = [(vidx[id(v)], float(dv)) for v, dv in zip(vs, ders)]
        for i, dv in entries:
            jt_lam[i] += lam * dv
        jac.append((c, lam, entries))
    zl = np.array([float(m.ipopt_zL_out.get(v, 0.0)) for v in variables])
    zu = np.array([float(m.ipopt_zU_out.get(v, 0.0)) for v in variables])
    resid = gf - jt_lam - zl - zu
    worst = int(np.argmax(np.abs(resid))) if len(resid) else None
    fam_max = defaultdict(float)
    for i, v in enumerate(variables):
        fam_max[_fam(v)] = max(fam_max[_fam(v)], abs(resid[i]))
    return {'obj': obj, 'variables': variables, 'vidx': vidx, 'gf': gf, 'jac': jac, 'zl': zl, 'zu': zu,
            'resid': resid, 'n_missing_dual': n_missing_dual,
            'kkt': {'n_variables': len(variables), 'n_rows': len(jac), 'n_rows_without_dual': n_missing_dual,
                    'max_abs_residual_raw': float(np.max(np.abs(resid))) if len(resid) else None,
                    'max_abs_gradient_raw': float(np.max(np.abs(gf))) if len(gf) else None,
                    'worst_variable': variables[worst].name if worst is not None else None,
                    'max_abs_residual_per_variable_family': dict(sorted(fam_max.items())),
                    'convention': 'r = df/dv - sum_c lambda_c dc/dv - zL - zU (zL >= 0, zU <= 0)'}}


def _node_order(case9):
    return [int(n['bus_i']) for n in case9['nodes']]


def structure(m, case9):
    """Model-derived structure (generator -> node, branch -> node pair), cross-checked against the case9 file."""
    bus_of_node = _node_order(case9)
    gen_node = {}
    for g in m.generators:
        for i in m.nodes:
            body_vars = {id(v) for v in identify_variables(m.node_balance_p[i, 0, 0, 0].body, include_fixed=True)}
            if id(m.pg[g, 0, 0, 0]) in body_vars:
                gen_node[g] = i
    interface_of_dn = {}  # dn -> (bus, load index), read from the balance rows that carry interface_delta_p / pc
    for i in m.nodes:
        body = list(identify_variables(m.node_balance_p[i, 0, 0, 0].body, include_fixed=True))
        dns = [v.index()[0] for v in body if _fam(v) == 'interface_delta_p']
        loads = [v.index()[0] for v in body if _fam(v) == 'pc']
        for dn in dns:
            interface_of_dn[dn] = (bus_of_node[i], loads[0] if len(loads) == 1 else None)
    br_nodes = {}
    for b in m.branches:
        c = m.voltage_product_real_def[b, 0, 0, 0]
        es = [v.index()[0] for v in identify_variables(c.body, include_fixed=True) if _fam(v) == 'e']
        br_nodes[b] = tuple(es)
    gens = []
    ok = True
    for g in m.generators:
        cg = case9['generators'][g]
        node = gen_node.get(g)
        match = node is not None and bus_of_node[node] == int(cg['bus'])
        ok = ok and match
        gens.append({'index': g, 'gen_id': cg['gen_id'], 'bus': int(cg['bus']), 'type': cg['type'],
                     'Pmax_MW': cg['Pmax'], 'Pmin_MW': cg['Pmin'], 'model_node_index': node, 'bus_matches_model': match})
    branches = []
    for b in m.branches:
        lb = case9['lines'][b]
        pair = br_nodes.get(b) or ()
        buses = {bus_of_node[i] for i in pair}
        match = buses == {int(lb['fbus']), int(lb['tbus'])}
        ok = ok and match
        r, x = float(lb['r']), float(lb['x'])
        den = r * r + x * x
        branches.append({'index': b, 'branch_id': lb.get('branch_id'), 'fbus': int(lb['fbus']), 'tbus': int(lb['tbus']),
                         'r': r, 'x': x, 'g': r / den, 'b': -x / den, 'rating_MVA': lb['rating'],
                         'buses_match_model': match})
    ok = ok and sorted(b for b, _l in interface_of_dn.values()) == sorted(ADN_BUSES) and all(
        l is not None for _b, l in interface_of_dn.values())
    return {'bus_of_node': bus_of_node, 'node_of_bus': {b: i for i, b in enumerate(bus_of_node)},
            'interface_of_dn': interface_of_dn, 'generators': gens, 'branches': branches, 'structure_matches_case9': ok,
            'reference_bus': [int(n['bus_i']) for n in case9['nodes'] if int(n['type']) == 3]}


# ======================================================================================================================
#  per-block analysis
# ======================================================================================================================
def analyse_block(m, case9, w25day, detail):
    B = float(case9['baseMVA'])
    st = structure(m, case9)
    kk = block_kkt(m)
    variables, vidx, gf, jac, zl, zu = kk['variables'], kk['vidx'], kk['gf'], kk['jac'], kk['zl'], kk['zu']
    z_total = zl + zu
    ref_bus = st['reference_bus'][0]
    ref_node = st['node_of_bus'][ref_bus]
    node7 = st['node_of_bus'][NODE]
    dual = m.dual
    pen_curt = pe.value(m.penalty_gen_curtailment)

    # market price from the objective gradient of the CONV generators
    pi = []
    conv = [gg for gg in st['generators'] if gg['type'] in ('CONV', 'REF')]
    pi_spread_across_conv = 0.0
    for p in range(N_PERIODS):
        vals = [gf[vidx[id(m.pg[gg['index'], 0, 0, p])]] / B for gg in conv if id(m.pg[gg['index'], 0, 0, p]) in vidx]
        pi.append(vals[0])
        pi_spread_across_conv = max(pi_spread_across_conv, max(vals) - min(vals))
    lmp = {bus: [float(dual[m.node_balance_p[st['node_of_bus'][bus], 0, 0, p]]) / B for p in range(N_PERIODS)]
           for bus in st['bus_of_node']}
    lmp7 = lmp[NODE]

    # rows touching the network variables, per period
    net_by_p = defaultdict(list)
    for i, v in enumerate(variables):
        if _fam(v) in NET_FAMILIES:
            net_by_p[_period(v)].append(i)
    rows_by_p = defaultdict(list)
    for c, lam, entries in jac:
        rows_by_p[_period(c)].append((c, lam, entries))

    per_hour = []
    null_diag = []
    for p in range(N_PERIODS):
        N = net_by_p[p]
        npos = {vi: k for k, vi in enumerate(N)}
        U, K = [], []
        for c, lam, entries in rows_by_p[p]:
            touches = [(vi, dv) for vi, dv in entries if vi in npos]
            if not touches:
                continue
            (U if _fam(c) in U_FAMILIES else K).append((c, lam, entries, touches))
        upos = {id(c): k for k, (c, _l, _e, _t) in enumerate(U)}
        M = np.zeros((len(N), len(U)))
        for k, (c, lam, entries, touches) in enumerate(U):
            for vi, dv in touches:
                M[npos[vi], k] += dv
        lam_u = np.array([lam for (_c, lam, _e, _t) in U])
        rhs = defaultdict(lambda: np.zeros(len(N)))
        row_rhs = {}
        for k, vi in enumerate(N):
            fam = _fam(variables[vi])
            if gf[vi] != 0.0:
                rhs[OBJ_GROUP.get(fam, 'other')][k] += gf[vi]
            if z_total[vi] != 0.0:
                rhs[BOUND_GROUP.get(fam, 'other')][k] -= z_total[vi]
        for c, lam, entries, touches in K:
            grp = ROW_GROUP.get(_fam(c), 'other')
            vec = np.zeros(len(N))
            for vi, dv in touches:
                vec[npos[vi]] -= lam * dv
            rhs[grp] += vec
            if abs(lam) >= DUAL_TOL:
                row_rhs[c.name] = (grp, lam, vec)
        total_rhs = sum(rhs.values()) if rhs else np.zeros(len(N))
        # rows of M that are identically zero (e.g. slacks: they appear in no U row) -> consistency equations
        live = np.where(np.abs(M).sum(axis=1) > 0)[0]
        dead = np.where(np.abs(M).sum(axis=1) == 0)[0]
        consistency = float(np.max(np.abs(total_rhs[dead]))) if len(dead) else 0.0
        Ml = M[live]
        sv = np.linalg.svd(Ml, compute_uv=False)
        _u, _s, vt = np.linalg.svd(Ml)
        z = vt[-1]
        ref_k = upos[id(m.node_balance_p[ref_node, 0, 0, p])]
        k7 = upos[id(m.node_balance_p[node7, 0, 0, p])]
        z = z / z[ref_k]
        null_diag.append({'period': p, 'n_net_vars': len(N), 'n_live_eq': len(live), 'n_U': len(U),
                          'sigma_min_over_max': float(sv[-1] / sv[0]), 'sigma_2nd_min_over_max': float(sv[-2] / sv[0]),
                          'consistency_dead_rows_max_abs': consistency})
        norm_row = np.zeros((1, len(U)))
        norm_row[0, ref_k] = 1.0
        A = np.vstack([Ml, norm_row])

        def solve(vec):
            b = np.concatenate([vec[live], [0.0]])
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x, float(np.max(np.abs(A @ x - b)))

        comps, group_resid = {}, {}
        x_sum = np.zeros(len(U))
        for grp, vec in rhs.items():
            x, res = solve(vec)
            x_sum += x
            comps[grp] = float(x[k7]) / B
            group_resid[grp] = res
        y_ref = float(dual[m.node_balance_p[ref_node, 0, 0, p]])
        recon_u = y_ref * z + x_sum
        recon_err_u = float(np.max(np.abs(recon_u - lam_u)))
        row_shares = []
        for name, (grp, lam, vec) in row_rhs.items():
            x, res = solve(vec)
            share = float(x[k7]) / B
            if abs(share) >= CONTRIB_REPORT_TOL:
                row_shares.append({'row': name, 'group': grp, 'dual_raw': lam, 'share_of_lmp7_eur_per_mwh': share,
                                   'solve_residual': res})
        row_shares.sort(key=lambda r: -abs(r['share_of_lmp7_eur_per_mwh']))
        c = {k: 0.0 for k in COMPONENTS}
        c['reference_generator_limit'] = y_ref / B - pi[p]
        c['losses'] = (y_ref / B) * (float(z[k7]) - 1.0)
        for grp, val in comps.items():
            c[grp if grp in c else 'other'] += val
        c['residual'] = (lmp7[p] - pi[p]) - sum(v for k, v in c.items() if k != 'residual')
        hour = {'period': p, 'hour': p + 1, 'pi': pi[p], 'lmp7': lmp7[p], 'lmp7_minus_pi': lmp7[p] - pi[p],
                'lmp_by_bus': {str(b): lmp[b][p] for b in st['bus_of_node']}, 'y_ref_over_B': y_ref / B,
                'loss_factor_z7': float(z[k7]), 'components': c, 'group_solve_residuals': group_resid,
                'reconstruction_max_abs_err_raw': recon_err_u, 'row_shares': row_shares}
        hd = hour_detail(m, st, B, p, kk, pen_curt)
        hour['price_setter'] = price_setter(hd['generators'])
        hour['interface_identity_bus7'] = interface_identity(m, st, B, p, kk, pi[p], lmp7[p], rows_by_p[p])
        if detail:
            hour.update(hd)
        else:
            hour['conv_p_status'] = {str(g['gen_id']): g['p_status'] for g in hd['generators']
                                     if g['type'] in ('CONV', 'REF')}
            hour['res_curtailed_MW'] = sum(g['pg_ub_MW'] - g['pg_MW'] for g in hd['generators']
                                           if g['type'] not in ('CONV', 'REF') and g['p_status'].startswith('interior'))
        per_hour.append(hour)

    out = {'baseMVA': B, 'reference_bus': ref_bus, 'structure': {k: st[k] for k in ('bus_of_node', 'generators',
                                                                                    'branches', 'structure_matches_case9')},
           'penalty_gen_curtailment': pen_curt, 'kkt': kk['kkt'], 'pi_spread_across_conv_generators': pi_spread_across_conv,
           'pi': pi, 'lmp7': lmp7, 'null_space': null_diag, 'per_hour': per_hour}
    # cross-checks against W25 (identity series, and the market arrays)
    if w25day:
        out['w25_crosscheck'] = {
            'max_abs_lmp7_dual_minus_w25_identity': max(abs(a - b) for a, b in zip(lmp7, w25day['lmp_series'])),
            'max_abs_pi_objective_minus_w25_price': max(abs(a - b) for a, b in zip(pi, w25day['price_series'])),
            'w25_lmp_spread': w25day.get('lmp_spread'), 'w25_market_spread': w25day.get('market_spread')}
    out['spread'] = spread_decomposition(pi, lmp7, per_hour)
    del kk
    return out


def hour_detail(m, st, B, p, kk, pen_curt):
    vidx, zl, zu, gf = kk['vidx'], kk['zl'], kk['zu'], kk['gf']
    dual = m.dual

    def zmult(v):
        i = vidx.get(id(v))
        return (float(zl[i]), float(zu[i])) if i is not None else (0.0, 0.0)

    gens = []
    for gg in st['generators']:
        g = gg['index']
        pg, qg = m.pg[g, 0, 0, p], m.qg[g, 0, 0, p]
        lb, ub = pg.lb, pg.ub
        pv = pe.value(pg)
        if pg.fixed or (lb is not None and ub is not None and ub - lb <= 2e-5):
            pstat = 'fixed/unavailable'
        elif lb is not None and pv - lb <= GEN_BOUND_TOL_PU:
            pstat = 'at_pmin'
        elif ub is not None and ub - pv <= GEN_BOUND_TOL_PU:
            pstat = 'at_pmax' if gg['type'] in ('CONV', 'REF') else 'at_available'
        else:
            pstat = 'interior' if gg['type'] in ('CONV', 'REF') else 'interior (curtailed)'
        qv = pe.value(qg)
        if qg.lb is not None and qv - qg.lb <= GEN_BOUND_TOL_PU:
            qstat = 'at_qmin'
        elif qg.ub is not None and qg.ub - qv <= GEN_BOUND_TOL_PU:
            qstat = 'at_qmax'
        else:
            qstat = 'interior'
        rows = {}
        for name in ('sg_capability', 'gen_pf_upper', 'gen_pf_lower', 'gen_pf_profile'):
            comp = getattr(m, name, None)
            if comp is not None and (g, 0, 0, p) in comp and comp[g, 0, 0, p].active:
                rows[name] = float(dual.get(comp[g, 0, 0, p], 0.0))
        zlp, zup = zmult(pg)
        zlq, zuq = zmult(qg)
        cap_binding = any(abs(v) >= DUAL_TOL for v in rows.values())
        bus_lmp = float(dual[m.node_balance_p[st['node_of_bus'][gg['bus']], 0, 0, p]]) / B
        own_mc = gf[vidx[id(pg)]] / B if id(pg) in vidx else None
        gens.append({'gen_id': gg['gen_id'], 'bus': gg['bus'], 'type': gg['type'], 'pg_MW': pv * B,
                     'pg_lb_MW': lb * B if lb is not None else None, 'pg_ub_MW': ub * B if ub is not None else None,
                     'p_status': pstat, 'qg_Mvar': qv * B, 'q_status': qstat,
                     'pg_bound_multiplier_eur_per_mwh': (zlp + zup) / B,
                     'qg_bound_multiplier_eur_per_mvarh': (zlq + zuq) / B,
                     'capability_row_duals_raw': rows, 'capability_binding': cap_binding,
                     'own_marginal_cost_eur_per_mwh': own_mc, 'bus_lmp_eur_per_mwh': bus_lmp,
                     'marginal': pstat.startswith('interior') and not cap_binding})
    # branches: loading, flow limit dual, active-power flows and losses (the balance-row branch terms)
    e = {i: pe.value(m.e[i, 0, 0, p]) for i in m.nodes}
    f = {i: pe.value(m.f[i, 0, 0, p]) for i in m.nodes}
    branches, losses = [], 0.0
    for br in st['branches']:
        b = br['index']
        fi, ti = st['node_of_bus'][br['fbus']], st['node_of_bus'][br['tbus']]
        vpr = e[fi] * e[ti] + f[fi] * f[ti]
        vpi = f[fi] * e[ti] - e[fi] * f[ti]
        pij = br['g'] * (e[fi] ** 2 + f[fi] ** 2) - (br['g'] * vpr + br['b'] * vpi)
        pji = br['g'] * (e[ti] ** 2 + f[ti] ** 2) - (br['g'] * vpr - br['b'] * vpi)
        losses += pij + pji
        flow_sqr = pe.value(m.flow_ij_sqr[b, 0, 0, p])
        lim = (br['rating_MVA'] / B) ** 2
        con = m.branch_flow_limit[b, 0, 0, p]
        lam = float(dual.get(con, 0.0))
        branches.append({'branch_id': br['branch_id'], 'from_bus': br['fbus'], 'to_bus': br['tbus'],
                         'P_from_MW': pij * B, 'P_to_MW': pji * B, 'loss_MW': (pij + pji) * B,
                         'loading_sqrt_flow_sqr_over_limit': math.sqrt(max(flow_sqr, 0.0) / lim),
                         'rating_MVA': br['rating_MVA'], 'limit_dual_raw': lam, 'active': abs(lam) >= DUAL_TOL})
    volts = []
    for i in m.nodes:
        bus = st['bus_of_node'][i]
        vm = math.sqrt(pe.value(m.vmag_sqr[i, 0, 0, p]))
        rows = {}
        for name in ('voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons', 'voltage_setpoint_cons'):
            comp = getattr(m, name, None)
            if comp is not None and (i, 0, 0, p) in comp and comp[i, 0, 0, p].active:
                rows[name] = float(dual.get(comp[i, 0, 0, p], 0.0))
        sl = {}
        for name in ('slack_v_sqr_down', 'slack_v_sqr_up'):
            v = getattr(m, name)[i, 0, 0, p]
            sl[name] = pe.value(v)
        volts.append({'bus': bus, 'vmag_pu': vm, 'row_duals_raw': rows, 'slacks': sl,
                      'active': any(abs(v) >= DUAL_TOL for v in rows.values())})
    interface = []
    for dn in sorted({k[0] for k in m.interface_delta_p.keys()}):
        bus, load_idx = st['interface_of_dn'][dn]
        row_pf = m.expected_interface_pf_p_def[dn, p] if hasattr(m, 'expected_interface_pf_p_def') else None
        row_v = m.expected_interface_vmag_def[dn, p] if hasattr(m, 'expected_interface_vmag_def') else None
        dv = m.interface_delta_p[dn, 0, 0, p]
        interface.append({'bus': bus, 'pc_fixed_MW': pe.value(m.pc[load_idx, 0, 0, p]) * B,
                          'qc_fixed_Mvar': pe.value(m.qc[load_idx, 0, 0, p]) * B,
                          'interface_delta_p_MW': pe.value(dv) * B,
                          'interface_delta_q_Mvar': pe.value(m.interface_delta_q[dn, 0, 0, p]) * B,
                          'interface_delta_p_bounds_MW': [dv.lb * B if dv.lb is not None else None,
                                                          dv.ub * B if dv.ub is not None else None],
                          'net_withdrawal_MW': (pe.value(m.pc[load_idx, 0, 0, p]) + pe.value(dv)) * B,
                          'expected_interface_pf_p_def_dual_raw': float(dual.get(row_pf, 0.0)) if row_pf is not None else None,
                          'expected_interface_vmag_def_dual_raw': float(dual.get(row_v, 0.0)) if row_v is not None else None,
                          'bus_lmp_eur_per_mwh': float(dual[m.node_balance_p[st['node_of_bus'][bus], 0, 0, p]]) / B})
    other_active = []
    net_fams = set(NET_FAMILIES)
    listed = {'branch_flow_limit', 'voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons',
              'voltage_setpoint_cons', 'sg_capability', 'gen_pf_upper', 'gen_pf_lower', 'gen_pf_profile'}
    for c, lam, _entries in kk['jac']:
        if _period(c) == p and abs(lam) >= DUAL_TOL and not c.equality and _fam(c) not in listed:
            other_active.append({'row': c.name, 'dual_raw': lam})
    return {'generators': gens, 'marginal_generators': [g['gen_id'] for g in gens if g['marginal']],
            'branches': branches, 'active_branch_limits': [b for b in branches if b['active']],
            'losses_MW': losses * B, 'voltages': volts, 'active_voltage_rows': [v for v in volts if v['active']],
            'interface': interface, 'other_active_inequality_rows': other_active,
            '_net_families': sorted(net_fams)}


PRICE_SETTER_RULE = ('per hour: (a) CONV generator(s) with pg strictly inside [pmin, pmax] and no binding capability '
                     'row -> they are marginal and their bus marginal cost is pi; else (b) TN WIND/PV generator(s) '
                     'curtailed (pg strictly below the available power; curtailment penalty 0) -> marginal at zero '
                     'cost; else (c) no TN generator is marginal (CONV at a limit, TN renewables fully used): the '
                     'marginal MWh is balanced by the ADN interfaces (interface_delta_p, priced by the ADMM consensus '
                     'terms -- see the interface-side identity)')


def price_setter(gens):
    conv = [g for g in gens if g['type'] in ('CONV', 'REF') and g['marginal']]
    res = [g for g in gens if g['type'] not in ('CONV', 'REF') and g['marginal']]
    conv_status = {str(g['gen_id']): g['p_status'] for g in gens if g['type'] in ('CONV', 'REF')}
    if conv:
        kind = 'a_conv_interior'
        text = 'CONV ' + ', '.join(f"G{g['gen_id']}@bus{g['bus']}" for g in conv) + ' interior (marginal at pi)'
    elif res:
        kind = 'b_res_curtailed'
        text = 'curtailed TN RES ' + ', '.join(f"G{g['gen_id']}@bus{g['bus']}" for g in res) + ' (marginal at 0)'
    else:
        kind = 'c_interface'
        text = ('no TN generator marginal (CONV ' + ', '.join(f'G{k} {v}' for k, v in conv_status.items())
                + '; TN RES at available): balanced by the ADN interfaces')
    return {'kind': kind, 'text': text, 'conv_status': conv_status,
            'marginal_conv': [g['gen_id'] for g in conv], 'marginal_res': [g['gen_id'] for g in res]}


INTERFACE_IDENTITY = ('stationarity of interface_delta_p at bus 7 (free; enters node_balance_p[7] with -1): '
                      'LMP7 = -(df/d delta)/B + sum_(c != balance) lambda_c (dc/d delta)/B + (zL + zU)/B, i.e. '
                      'LMP7 - pi = [-(df/d delta)/B - pi] (objective terms on delta other than the settlement at pi; '
                      '0 when the settlement -pi*B*delta is the only one) + [ADMM interface consensus: the '
                      'expected_interface_pf_p_def row] + [delta bound multipliers] + residual')


def interface_identity(m, st, B, p, kk, pi_p, lmp7_p, rows_p):
    dn7 = next(dn for dn, (bus, _l) in st['interface_of_dn'].items() if bus == NODE)
    delta = m.interface_delta_p[dn7, 0, 0, p]
    i = kk['vidx'].get(id(delta))
    if i is None:
        return {'available': False, 'residual': 0.0}
    gf = float(kk['gf'][i])
    bal = m.node_balance_p[st['node_of_bus'][NODE], 0, 0, p]
    j_bal, others = None, defaultdict(float)
    for c, lam, entries in rows_p:
        for vi, dv in entries:
            if vi == i:
                if c is bal:
                    j_bal = dv
                else:
                    others[_fam(c)] += lam * dv
    zsum = float(kk['zl'][i] + kk['zu'][i])
    # stationarity: gf - lambda_bal * j_bal - sum_others - z = 0  ->  lambda_bal = (gf - sum_others - z) / j_bal
    obj_term = (gf / j_bal) / B - pi_p
    row_terms = {k: (-v / j_bal) / B for k, v in others.items()}
    bound_term = (-zsum / j_bal) / B
    resid = (lmp7_p - pi_p) - obj_term - sum(row_terms.values()) - bound_term
    return {'available': True, 'd_balance_d_delta': j_bal, 'objective_terms_minus_pi': obj_term,
            'row_terms': row_terms, 'delta_bound_term': bound_term, 'residual': resid,
            'delta_MW': pe.value(delta) * B}


def _top_bottom(series):
    order = sorted(range(len(series)), key=lambda k: (-series[k], k))
    top = sorted(order[:H])
    order_b = sorted(range(len(series)), key=lambda k: (series[k], k))
    bottom = sorted(order_b[:H])
    return top, bottom


def spread_decomposition(pi, lmp7, per_hour):
    top, bottom = _top_bottom(lmp7)
    ptop, pbot = _top_bottom(pi)
    mean = lambda xs, ks: sum(xs[k] for k in ks) / len(ks)  # noqa: E731
    s_pi = mean(pi, ptop) - mean(pi, pbot)
    s_lmp = mean(lmp7, top) - mean(lmp7, bottom)
    hsel = s_pi - (mean(pi, top) - mean(pi, bottom))
    d = {}
    for comp in COMPONENTS:
        series = [h['components'][comp] for h in per_hour]
        d[comp] = mean(series, top) - mean(series, bottom)
    identity_err = s_lmp - (s_pi - hsel + sum(d.values()))
    return {'lmp7_top4_hours': [k + 1 for k in top], 'lmp7_bottom4_hours': [k + 1 for k in bottom],
            'pi_top4_hours': [k + 1 for k in ptop], 'pi_bottom4_hours': [k + 1 for k in pbot],
            'market_spread': s_pi, 'lmp7_spread': s_lmp, 'flatness_market_minus_lmp7': s_pi - s_lmp,
            'hour_selection_H': hsel, 'component_spreads_D': d, 'identity_error': identity_err,
            'flatness_by_term': dict({'hour_selection_H': hsel}, **{f'minus_D_{k}': -v for k, v in d.items()})}


# ======================================================================================================================
#  report
# ======================================================================================================================
def _f(x, nd=2):
    return '' if x is None else f'{x:,.{nd}f}'


def aggregate(blocks, keys, wkey):
    wsum = sum(blocks[k]['weights'][wkey] for k in keys)
    agg = {'market_spread': 0.0, 'lmp7_spread': 0.0, 'flatness_market_minus_lmp7': 0.0, 'hour_selection_H': 0.0}
    agg.update({f'D_{c}': 0.0 for c in COMPONENTS})
    for k in keys:
        w = blocks[k]['weights'][wkey] / wsum
        s = blocks[k]['spread']
        for f in ('market_spread', 'lmp7_spread', 'flatness_market_minus_lmp7', 'hour_selection_H'):
            agg[f] += w * s[f]
        for c in COMPONENTS:
            agg[f'D_{c}'] += w * s['component_spreads_D'][c]
    agg['ratio_lmp7_over_market'] = agg['lmp7_spread'] / agg['market_spread']
    agg['weights'] = wkey
    agg['blocks'] = list(keys)
    return agg


def write_markdown(path, res):
    L = []
    a = L.append
    a(f"# {res['stage']}")
    a('')
    inp = res['inputs']
    a(f"Instance: x = 0, candidate key `{inp.get('candidate_key')}`; evaluation `{inp.get('eval_dir')}` "
      f"(eval key `{inp.get('eval_key')}`, {inp.get('cycles_run')} cycles, Q gross {_f(inp.get('certified_cost_gross'), 7)}); "
      f"reproduction gate `{inp.get('gate', {}).get('path')}` gate_pass = {inp.get('gate', {}).get('gate_pass')} (strict, "
      f"not overridden); ruled PASS by `{inp.get('gate_ruling', {}).get('path')}` ({inp.get('gate_ruling', {}).get('commit')}, "
      f"sha256 `{inp.get('gate_ruling', {}).get('sha256')}`, verified {inp.get('gate_ruling', {}).get('ruled_pass_ok')}). "
      f"Models: `{inp.get('pickle', {}).get('path')}` sha256 `{inp.get('pickle', {}).get('sha256')}` (not committed). "
      f"Mode: {inp.get('mode')}.")
    a('')
    a(f"Solve profile: armed SolveProfileGuard(permitted=()); counts {res['solve_profile_guard']['counts']}; verify(0) "
      f"failures {res['solve_profile_guard']['verify_0_failures']}. all_checks_pass = {res['all_checks_pass']}; failing: "
      f"{res['failing_checks']}.")
    a('')
    a('Units / sign: LMP_b = dual(node_balance_p[b]) / baseMVA [EUR/MWh]; the row is Pg - Pd - Pflow = 0, so the dual is '
      '+d(objective)/d(load at b) -- the cost of serving one more MWh at b. pi = the market price the TSO pays its CONV '
      'generators (objective gradient / baseMVA). KKT convention: ' + res['method']['kkt_convention'])
    a('')
    a('## Method')
    a('')
    a('Decomposition: ' + DECOMP_FORMULA)
    a('')
    a('Components: ' + '; '.join(f'**{k}** = {v}' for k, v in COMPONENT_TEXT.items()))
    a('')
    a('Spread: ' + SPREAD_FORMULA)
    a('')
    a('## Checks')
    a('')
    a('| check | value |')
    a('|---|---|')
    for k, v in res['checks'].items():
        a(f'| {k} | {v} |')
    a('')
    a('| block | KKT max abs residual (raw) | max abs grad | LMP7 vs W25 identity (EUR/MWh) | pi vs W25 (EUR/MWh) | '
      'max recon err (raw) | null sigma_min/max (max) | null sigma_2nd/max (min) | max |residual comp| (EUR/MWh) |')
    a('|---|---|---|---|---|---|---|---|---|')
    for key, blk in res['blocks'].items():
        w = blk.get('w25_crosscheck') or {}
        a(f"| {key} | {blk['kkt']['max_abs_residual_raw']:.2e} | {blk['kkt']['max_abs_gradient_raw']:.2e} | "
          f"{w.get('max_abs_lmp7_dual_minus_w25_identity', float('nan')):.2e} | "
          f"{w.get('max_abs_pi_objective_minus_w25_price', float('nan')):.2e} | "
          f"{max(h['reconstruction_max_abs_err_raw'] for h in blk['per_hour']):.2e} | "
          f"{max(n['sigma_min_over_max'] for n in blk['null_space']):.1e} | "
          f"{min(n['sigma_2nd_min_over_max'] for n in blk['null_space']):.1e} | "
          f"{max(abs(h['components']['residual']) for h in blk['per_hour']):.2e} |")
    a('')
    a(f'## {DETAIL_YEAR}: top-4 and bottom-4 hours of the bus-7 marginal cost, per day')
    a('')
    for day in DAYS:
        key = f'{DETAIL_YEAR}_{day}'
        blk = res['blocks'][key]
        s = blk['spread']
        a(f"### {key}")
        a('')
        a(f"Market 4 h spread {_f(s['market_spread'])}; LMP7 4 h spread {_f(s['lmp7_spread'])}; flatness "
          f"{_f(s['flatness_market_minus_lmp7'])} EUR/MWh. LMP7 top-4 hours {s['lmp7_top4_hours']}, bottom-4 "
          f"{s['lmp7_bottom4_hours']}; pi top-4 {s['pi_top4_hours']}, bottom-4 {s['pi_bottom4_hours']}. "
          f"penalty_gen_curtailment = {blk['penalty_gen_curtailment']}.")
        a('')
        a('| set | hour | pi | LMP7 | LMP7-pi | price setter | marginal gens (interior) | gens at a bound | curtailed RES (MW) | '
          'active branch limits (loading, dual raw) | active voltage rows (dual raw) | losses MW | '
          'interface dP 5/7/9 MW | withdrawal 5/7/9 MW |')
        a('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for label, hours in (('top', s['lmp7_top4_hours']), ('bottom', s['lmp7_bottom4_hours'])):
            for hr in hours:
                h = blk['per_hour'][hr - 1]
                gens = h['generators']
                marg = ', '.join(f"G{g['gen_id']}@{g['bus']} {g['type']} {g['pg_MW']:.1f}" for g in gens if g['marginal'])
                bnd = ', '.join(f"G{g['gen_id']}@{g['bus']} {g['p_status']} ({g['pg_bound_multiplier_eur_per_mwh']:+.2f})"
                                for g in gens if g['p_status'].startswith('at_'))
                curt = ', '.join(f"G{g['gen_id']}@{g['bus']} {g['pg_ub_MW'] - g['pg_MW']:.1f}" for g in gens
                                 if g['type'] not in ('CONV', 'REF') and g['p_status'].startswith('interior'))
                brs = ', '.join(f"{b['from_bus']}-{b['to_bus']} ({b['loading_sqrt_flow_sqr_over_limit']:.3f}, "
                                f"{b['limit_dual_raw']:.1f})" for b in h['active_branch_limits'])
                vs = ', '.join(f"bus {v['bus']} V={v['vmag_pu']:.4f} " + ' '.join(
                    f"{k.replace('voltage_magnitude_', '').replace('_cons', '')}={d:.1f}" for k, d in v['row_duals_raw'].items()
                    if abs(d) >= DUAL_TOL) for v in h['active_voltage_rows'])
                itf = h['interface']
                a(f"| {label} | {hr} | {_f(h['pi'])} | {_f(h['lmp7'])} | {_f(h['lmp7_minus_pi'])} | "
                  f"{h['price_setter']['kind']} | {marg or '-'} | "
                  f"{bnd or '-'} | {curt or '-'} | {brs or '-'} | {vs or '-'} | {_f(h['losses_MW'])} | "
                  f"{' / '.join(_f(i['interface_delta_p_MW'], 1) for i in itf)} | "
                  f"{' / '.join(_f(i['net_withdrawal_MW'], 1) for i in itf)} |")
        a('')
        a('Decomposition of LMP7 - pi in these hours (EUR/MWh):')
        a('')
        a('| set | hour | LMP7-pi | ' + ' | '.join(COMPONENTS) + ' | loss factor z7 | largest row shares |')
        a('|---|---|---|' + '---|' * len(COMPONENTS) + '---|---|')
        for label, hours in (('top', s['lmp7_top4_hours']), ('bottom', s['lmp7_bottom4_hours'])):
            for hr in hours:
                h = blk['per_hour'][hr - 1]
                shares = '; '.join(f"{r['row']} {r['share_of_lmp7_eur_per_mwh']:+.2f}" for r in h['row_shares'][:3])
                a(f"| {label} | {hr} | {_f(h['lmp7_minus_pi'])} | " + ' | '.join(
                    _f(h['components'][c], 3) for c in COMPONENTS) + f" | {h['loss_factor_z7']:.5f} | {shares or '-'} |")
        a('')
        a('Interface-side identity at bus 7 (EUR/MWh): ' + INTERFACE_IDENTITY)
        a('')
        a('| set | hour | LMP7-pi | objective terms on delta minus pi | ADMM consensus rows | delta bound | residual | '
          'delta_p MW |')
        a('|---|---|---|---|---|---|---|---|')
        for label, hours in (('top', s['lmp7_top4_hours']), ('bottom', s['lmp7_bottom4_hours'])):
            for hr in hours:
                h = blk['per_hour'][hr - 1]
                ii = h['interface_identity_bus7']
                a(f"| {label} | {hr} | {_f(h['lmp7_minus_pi'], 3)} | {_f(ii['objective_terms_minus_pi'], 3)} | "
                  + ', '.join(f'{k} {v:+.3f}' for k, v in ii['row_terms'].items())
                  + f" | {_f(ii['delta_bound_term'], 3)} | {ii['residual']:.1e} | {_f(ii['delta_MW'], 2)} |")
        a('')
        a('Price setter per hour (all 24): ' + '; '.join(f"h{h['hour']} {h['price_setter']['kind']}"
                                                          for h in blk['per_hour']))
        a('')
        a('Spread decomposition (EUR/MWh): ' + ', '.join(f'{k} {v:+.3f}' for k, v in s['flatness_by_term'].items())
          + f"; sum = flatness {s['flatness_market_minus_lmp7']:+.3f} (identity error {s['identity_error']:.1e}).")
        a('')
    a('## Spread decomposition, all blocks (supplementary for 2025 / 2035)')
    a('')
    a('| block | market spread | LMP7 spread | flatness | H (hour selection) | ' + ' | '.join(f'-D {c}' for c in COMPONENTS)
      + ' |')
    a('|---|---|---|---|---|' + '---|' * len(COMPONENTS))
    for key, blk in res['blocks'].items():
        s = blk['spread']
        a(f"| {key} | {_f(s['market_spread'], 3)} | {_f(s['lmp7_spread'], 3)} | {_f(s['flatness_market_minus_lmp7'], 3)} | "
          f"{_f(s['hour_selection_H'], 3)} | " + ' | '.join(_f(-s['component_spreads_D'][c], 3) for c in COMPONENTS) + ' |')
    a('')
    a('Aggregates (day-weighted with W25 block weights; flatness = H - sum_c D_c):')
    a('')
    a('| scope | weights | market | LMP7 | ratio | flatness | H | ' + ' | '.join(f'-D {c}' for c in COMPONENTS) + ' |')
    a('|---|---|---|---|---|---|---|' + '---|' * len(COMPONENTS))
    for name, agg in res['aggregates'].items():
        a(f"| {name} | {agg['weights']} | {_f(agg['market_spread'], 3)} | {_f(agg['lmp7_spread'], 3)} | "
          f"{agg['ratio_lmp7_over_market']:.4f} | {_f(agg['flatness_market_minus_lmp7'], 3)} | {_f(agg['hour_selection_H'], 3)} | "
          + ' | '.join(_f(-agg[f'D_{c}'], 3) for c in COMPONENTS) + ' |')
    a('')
    with open(path, 'x') as handle:
        handle.write('\n'.join(L) + '\n')


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-pickle', default=None)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    if bool(args.test_pickle) != bool(args.out_dir):
        parser.error('--test-pickle and --out-dir go together')
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(REPO, OUT_REL)
    if args.out_dir and os.path.commonpath([out_dir, REPO]) == REPO:
        parser.error('test --out-dir must be outside the repository')
    if not os.path.isdir(out_dir):
        raise SystemExit(f'output directory must exist (created by the launch command): {out_dir}')
    for name in ('x0_capture_analysis.json', 'x0_capture_analysis.md', 'manifest_sha256.json'):
        if os.path.exists(os.path.join(out_dir, name)):
            raise SystemExit(f'refusing to overwrite {name}')
    inputs, failures = resolve_inputs(args.test_pickle)
    w25 = _load_json(W25_JSON['path'])
    w25_t2 = w25['T2'] if 'T2' in w25 else w25
    inputs['w25'] = dict(W25_JSON, sha256=_sha(os.path.join(REPO, W25_JSON['path'])))
    if failures:
        for f in failures:
            _log(f'[W27] REFUSED: {f}')
        raise SystemExit(1)
    ppath = args.test_pickle or os.path.join(REPO, inputs['pickle']['path'])
    _log(f'[W27] loading {ppath}')
    with open(ppath, 'rb') as handle:
        payload = pickle.load(handle)
    checklist = capture_path_checklist(payload, w25_t2)
    missing = sorted(k for k, v in checklist.items() if not v)
    _log(f'[W27] capture-path checklist: {len(checklist)} items, {len(missing)} False: {missing}')
    if missing:
        raise SystemExit('CAPTURE-PATH CHECK FAILED (fail fast, before analysis)')
    w25_by_block = {(str(r['year']), r['day']): r for r in w25_t2['per_day']}
    blocks = {}
    for y in YEARS:
        case9 = _load_json(CASE9_FMT.format(year=y))
        for d in DAYS:
            t0 = time.time()
            wd = w25_by_block[(str(y), d)]
            blk = analyse_block(payload['tso'][y][d], case9, wd, detail=(y == DETAIL_YEAR))
            blk['weights'] = {'w_undisc': wd['w_undisc'], 'w_r2': wd['w_r2']}
            blocks[f'{y}_{d}'] = blk
            s = blk['spread']
            _log(f"[W27] {y} {d}: kkt {blk['kkt']['max_abs_residual_raw']:.2e} w25 "
                 f"{blk['w25_crosscheck']['max_abs_lmp7_dual_minus_w25_identity']:.2e} market {s['market_spread']:.3f} "
                 f"lmp7 {s['lmp7_spread']:.3f} flat {s['flatness_market_minus_lmp7']:.3f} "
                 f"H {s['hour_selection_H']:.3f} D {({k: round(v, 3) for k, v in s['component_spreads_D'].items()})} "
                 f"({time.time() - t0:.1f}s)")
            gc.collect()
    keys2030 = [f'{DETAIL_YEAR}_{d}' for d in DAYS]
    allkeys = list(blocks)
    aggregates = {f'{DETAIL_YEAR}, day-weighted (undiscounted)': aggregate(blocks, keys2030, 'w_undisc'),
                  f'{DETAIL_YEAR}, model block weight (2 %)': aggregate(blocks, keys2030, 'w_r2'),
                  'all years, day-weighted (undiscounted)': aggregate(blocks, allkeys, 'w_undisc'),
                  'all years, model block weight (2 %)': aggregate(blocks, allkeys, 'w_r2')}
    for y in (2025, 2035):
        aggregates[f'{y}, day-weighted (undiscounted)'] = aggregate(blocks, [f'{y}_{d}' for d in DAYS], 'w_undisc')
    w25_aggs = w25_t2.get('aggregates') or {}
    checks = {
        'structure_matches_case9_all_blocks': all(b['structure']['structure_matches_case9'] for b in blocks.values()),
        'kkt_residual_below_warn_all_blocks': all(b['kkt']['max_abs_residual_raw'] <= KKT_WARN for b in blocks.values()),
        'no_rows_without_dual': all(b['kkt']['n_rows_without_dual'] == 0 for b in blocks.values()),
        'pi_equal_across_conv_generators': all(b['pi_spread_across_conv_generators'] <= PRICE_TOL for b in blocks.values()),
        'pi_matches_w25_arrays': all(b['w25_crosscheck']['max_abs_pi_objective_minus_w25_price'] <= PRICE_TOL
                                     for b in blocks.values()),
        'lmp7_matches_w25_identity': all(b['w25_crosscheck']['max_abs_lmp7_dual_minus_w25_identity'] <= W25_LMP_TOL
                                         for b in blocks.values()),
        'rotational_null_space_one_dimensional': all(
            n['sigma_min_over_max'] <= NULL_RATIO_MAX and n['sigma_2nd_min_over_max'] >= NULL_GAP_MIN
            for b in blocks.values() for n in b['null_space']),
        'decomposition_reconstructs_multipliers': all(h['reconstruction_max_abs_err_raw'] <= RECON_MULT_TOL
                                                      for b in blocks.values() for h in b['per_hour']),
        'interface_identity_residual_below_tol': all(abs(h['interface_identity_bus7']['residual']) <= RECON_TOL * 100.0
                                                     for b in blocks.values() for h in b['per_hour']),
        'decomposition_residual_below_tol': all(abs(h['components']['residual']) <= RECON_TOL
                                                for b in blocks.values() for h in b['per_hour']),
        'spread_identity_exact': all(abs(b['spread']['identity_error']) <= 1e-9 for b in blocks.values()),
        'aggregate_2030_matches_w25': (abs(aggregates[f'{DETAIL_YEAR}, day-weighted (undiscounted)']['lmp7_spread']
                                           - ((w25_aggs.get(f'{DETAIL_YEAR}, day-weighted') or {}).get('lmp_x0') or 0))
                                       <= W25_LMP_TOL),
    }
    if args.test_pickle:
        checks['TEST_MODE_not_evidence'] = False
    guard_failures = GUARD.verify(0)
    checks['solve_profile_guard_verified_0'] = not guard_failures
    failing = sorted(k for k, v in checks.items() if not v)
    res = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
           'git_head': _git(['rev-parse', 'HEAD']), 'script': os.path.basename(__file__),
           'script_sha256': _sha(os.path.abspath(__file__)),
           'script_git_status': _git(['status', '--porcelain', '--', os.path.basename(__file__)]) or 'clean',
           'inputs': inputs, 'capture_path_checklist_asserted_before_analysis': checklist,
           'method': {'kkt_convention': 'r = df/dv - sum_c lambda_c dc/dv - zL - zU; lambda = model.dual, zL = '
                                        'ipopt_zL_out >= 0, zU = ipopt_zU_out <= 0',
                      'decomposition': DECOMP_FORMULA, 'components': COMPONENT_TEXT, 'spread': SPREAD_FORMULA,
                      'price_setter_rule': PRICE_SETTER_RULE, 'interface_identity': INTERFACE_IDENTITY,
                      'network_variable_families': list(NET_FAMILIES), 'unknown_multiplier_families': list(U_FAMILIES),
                      'row_groups': ROW_GROUP, 'bound_groups': BOUND_GROUP, 'objective_groups': OBJ_GROUP,
                      'tolerances': {'GEN_BOUND_TOL_PU': GEN_BOUND_TOL_PU, 'DUAL_TOL': DUAL_TOL,
                                     'CONTRIB_REPORT_TOL': CONTRIB_REPORT_TOL, 'W25_LMP_TOL': W25_LMP_TOL,
                                     'PRICE_TOL': PRICE_TOL, 'RECON_TOL': RECON_TOL, 'RECON_MULT_TOL': RECON_MULT_TOL,
                                     'KKT_WARN': KKT_WARN,
                                     'NULL_RATIO_MAX': NULL_RATIO_MAX, 'NULL_GAP_MIN': NULL_GAP_MIN}},
           'objective_convention': ('LMPs and prices in EUR/MWh per representative-day hour; no cost total is reported '
                                    'here (Q(0) = gross operational cost 653,859,461.2279255, settlement-excluded, is '
                                    'the certified value of the evaluation the models belong to)'),
           'checks': checks, 'all_checks_pass': not failing, 'failing_checks': failing,
           'aggregates': aggregates, 'w25_aggregates_for_comparison': w25_aggs, 'blocks': blocks,
           'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
           'wall_clock_s': time.time() - started}
    res = _jsonable(res)
    jpath = os.path.join(out_dir, 'x0_capture_analysis.json')
    with open(jpath, 'x') as handle:
        json.dump(res, handle, indent=1)
    mpath = os.path.join(out_dir, 'x0_capture_analysis.md')
    write_markdown(mpath, res)
    GUARD.uninstall()
    for name, agg in aggregates.items():
        _log(f"[W27] {name}: market {agg['market_spread']:.3f} lmp7 {agg['lmp7_spread']:.3f} ratio "
             f"{agg['ratio_lmp7_over_market']:.4f} flatness {agg['flatness_market_minus_lmp7']:.3f} H "
             f"{agg['hour_selection_H']:.3f} " + ' '.join(f"-D_{c} {-agg[f'D_{c}']:.3f}" for c in COMPONENTS))
    _log(f'[W27] checks: {checks}')
    _log(f"[W27] guard {GUARD.counts} verify0_failures={guard_failures}; wrote {jpath} and {mpath}")
    _log(f"[W27] {'ALL CHECKS PASS' if not failing else 'FAILING CHECKS: ' + str(failing)}")
    if not failing:
        return
    sys.exit(1)


def write_manifest(out_dir):
    man = {}
    for fname in sorted(os.listdir(out_dir)):
        if fname == 'manifest_sha256.json':
            continue
        man[os.path.relpath(os.path.join(out_dir, fname), REPO)] = _sha(os.path.join(out_dir, fname))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        json.dump(man, handle, indent=2)
    return man


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        print(json.dumps(write_manifest(os.path.join(REPO, OUT_REL)), indent=2))
        GUARD.uninstall()
    else:
        main()
