"""
Stage P4.6-B1 -- ordinary/standard ESS load-convention correction validation.

Runs the SAME measurements before and after the production sign correction, so
the two phases are exactly comparable:

    python p46b1_op1_ess_sign_validation.py --phase pre
    python p46b1_op1_ess_sign_validation.py --phase post

What this script does NOT do: it never reimplements the logic under test. It
calls the real production functions/paths only --
  - `SharedResourcesPlanning.read_planning_problem()`
  - `NetworkData.build_model()` / `.optimize()` / `.process_results()`
    (the uncoordinated local-solve recipe documented in `main.py`)
  - `model_construction_helpers.ess_pnet_rule`, `ess_phi_limits_lower/upper`
  - the production `pc_node` / `qc_node` Expressions, which ARE
    `compute_node_load(...)`
and only *observes* the result.

Sections
  A  provenance + verbatim source of every audited sign-bearing function
  B  section-4 direct algebraic sign tests (fresh model, no solve required)
  C  local SMOPF solve of the OP1 / case33_3 / 2025 validation case
  D  ESS trajectories, nodal-balance contribution, interface quantities
  E  raw-model vs processed-result P/Q comparison (section-6 requirement)

Observability note: `params.solver_params.verbose` is toggled ON *in this
script only* (never in a params file or in production code) so IPOPT prints its
iteration summary, which is then parsed. This changes no numerical option --
tol, acceptable_tol, acceptable_iter, linear_solver (ma97) and the exact-Hessian
/ recovery configuration are untouched -- and it is applied identically in both
phases.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import inspect  # noqa: E402

import model_construction_helpers as mch  # noqa: E402
import network as network_module  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

# --- Validation case (plan section 7) ------------------------------------
SPEC_DIR = 'data/OP1'
SPEC_FILE = 'OP1.json'
DSO_CONNECTION_NODE = 9          # case33_3
CASE_YEAR = 2025
CASE_DAY = 'Summer'

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'OP1', 'Results', 'P46B1')

IPOPT_SUMMARY_RE = {
    'iterations': re.compile(r'Number of Iterations\.*:\s*(\d+)'),
    'objective_scaled_unscaled': re.compile(
        r'Objective\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'primal_infeasibility': re.compile(
        r'Constraint violation\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'dual_infeasibility': re.compile(
        r'Dual infeasibility\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'complementarity': re.compile(
        r'Complementarity\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'overall_nlp_error': re.compile(
        r'Overall NLP error\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'exit_status': re.compile(r'EXIT: (.+?)\.'),
}


def git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def parse_ipopt_console(text):
    """Parse the LAST IPOPT summary block present in captured console output."""
    out = {}
    for key, pattern in IPOPT_SUMMARY_RE.items():
        match = None
        for match in pattern.finditer(text):
            pass
        if match:
            out[key] = match.groups() if len(match.groups()) > 1 else match.group(1)
    return out


# ============================================================================
#  A. Provenance -- verbatim source of every sign-bearing function
# ============================================================================
def section_a_provenance(phase):
    def src(obj):
        try:
            return inspect.getsource(obj)
        except Exception as error:  # pragma: no cover
            return f'<unavailable: {error}>'

    node_load_src = src(mch.compute_node_load)
    # the ordinary-ESS stanza inside compute_node_load
    ess_stanza = [
        line.rstrip() for line in node_load_src.splitlines()
        if 'es_pnet' in line or 'es_qnet' in line
    ]

    process_src = src(network_module._process_results)
    result_lines = [
        line.strip() for line in process_src.splitlines()
        if ('p_ess' in line or 'q_ess' in line) and 'shared' not in line
    ]

    return {
        'phase': phase,
        'git_head': git_head(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ess_pnet_rule_source': src(mch.ess_pnet_rule),
        'ess_snet_def_rule_source': src(mch.ess_snet_def_rule),
        'ess_phi_limits_lower_source': src(mch.ess_phi_limits_lower),
        'ess_phi_limits_upper_source': src(mch.ess_phi_limits_upper),
        'ess_comp_rule_source': src(mch.ess_comp_rule),
        'ess_soc_rule_source': src(mch.ess_soc_rule),
        'compute_node_load_ess_lines': ess_stanza,
        'process_results_ess_pq_lines': result_lines,
    }


# ============================================================================
#  Model construction (production path)
# ============================================================================
def build_planning_and_dso():
    planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
    planning.read_planning_problem()
    dso = planning.distribution_networks[DSO_CONNECTION_NODE]
    return planning, dso


def isolate_ordinary_ess(dso):
    """Mirror the uncoordinated local-solve recipe documented in main.py:
    clear shared ESS so the local solve exercises the ordinary-ESS path only.
    This mutates the in-memory network object, never a data file."""
    for year in dso.years:
        for day in dso.days:
            dso.network[year][day].shared_energy_storages = list()


# ============================================================================
#  B. Section-4 direct algebraic sign tests (no solve needed)
# ============================================================================
def section_b_algebra(dso):
    """Fix pch/pdch/qnet on a freshly built model and read the PRODUCTION
    constraint bodies / nodal-load Expressions back."""
    network = dso.network[CASE_YEAR][CASE_DAY]
    model = network.build_model(dso.params)

    s_m, s_o, p = 0, 0, 0
    e = 0
    ess = network.energy_storages[e]
    node_idx = network.get_node_idx(ess.bus)

    def set_ess(pch, pdch, pnet, qnet):
        model.es_pch[e, s_m, s_o, p].value = pch
        model.es_pdch[e, s_m, s_o, p].value = pdch
        model.es_pnet[e, s_m, s_o, p].value = pnet
        model.es_qnet[e, s_m, s_o, p].value = qnet

    def pnet_def_residual(pch, pdch, pnet):
        set_ess(pch, pdch, pnet, 0.0)
        return float(pe.value(model.ess_pnet_def[e, s_m, s_o, p].body))

    def nodal_delta(kind, value):
        """Change in the PRODUCTION net-load Expression caused by the ESS term."""
        expr = model.pc_node if kind == 'p' else model.qc_node
        if kind == 'p':
            set_ess(0.0, 0.0, 0.0, 0.0)
            base = float(pe.value(expr[node_idx, s_m, s_o, p]))
            set_ess(0.0, 0.0, value, 0.0)
        else:
            set_ess(0.0, 0.0, 0.0, 0.0)
            base = float(pe.value(expr[node_idx, s_m, s_o, p]))
            set_ess(0.0, 0.0, 0.0, value)
        return float(pe.value(expr[node_idx, s_m, s_o, p])) - base

    # stay inside the installed rating (s = 0.005 p.u.) so no bound is violated
    x = 0.4 * ess.s
    tol = 1e-12

    # 4.1 charging-only: pch>0, pdch=0 -> pnet must equal +pch
    res_plus = pnet_def_residual(x, 0.0, +x)
    res_minus = pnet_def_residual(x, 0.0, -x)
    charging = {
        'inputs': {'pch': x, 'pdch': 0.0, 'qnet': 0.0},
        'residual_if_pnet_equals_plus_pch': res_plus,
        'residual_if_pnet_equals_minus_pch': res_minus,
        'pnet_equals_plus_pch': abs(res_plus) < tol,
        'nodal_p_delta_for_pnet_plus_x': nodal_delta('p', +x),
    }
    charging['enters_balance_as_additional_load'] = (
        charging['nodal_p_delta_for_pnet_plus_x'] > tol)

    # 4.2 discharging-only: pdch>0, pch=0 -> pnet must equal -pdch
    res_neg = pnet_def_residual(0.0, x, -x)
    res_pos = pnet_def_residual(0.0, x, +x)
    discharging = {
        'inputs': {'pch': 0.0, 'pdch': x, 'qnet': 0.0},
        'residual_if_pnet_equals_minus_pdch': res_neg,
        'residual_if_pnet_equals_plus_pdch': res_pos,
        'pnet_equals_minus_pdch': abs(res_neg) < tol,
        'nodal_p_delta_for_pnet_minus_x': nodal_delta('p', -x),
    }
    discharging['reduces_net_demand'] = (
        discharging['nodal_p_delta_for_pnet_minus_x'] < -tol)

    # 4.3 / 4.4 reactive
    dq_abs = nodal_delta('q', +x)
    dq_inj = nodal_delta('q', -x)
    reactive = {
        'nodal_q_delta_for_qnet_plus_x': dq_abs,
        'qnet_positive_increases_reactive_demand': dq_abs > tol,
        'nodal_q_delta_for_qnet_minus_x': dq_inj,
        'qnet_negative_reduces_reactive_demand': dq_inj < -tol,
    }

    # power-factor limit region, evaluated on the PRODUCTION rules
    tangent_lower, tangent_upper = mch._power_factor_tangents(ess)
    phi = {
        'max_pf': ess.max_pf,
        'min_pf': ess.min_pf,
        'tangent_lower': tangent_lower,
        'tangent_upper': tangent_upper,
        'tangents_symmetric': abs(tangent_lower + tangent_upper) < 1e-12,
        'samples': [],
    }
    qvar = model.es_qnet[e, s_m, s_o, p]

    def bound_of(relational):
        """Return the numeric bound, whichever side of the relation it is on.
        Pyomo stores `q >= rhs` as (rhs, q) and `q <= rhs` as (q, rhs)."""
        left, right = relational.args
        return float(pe.value(right if left is qvar else left))

    for pch, pdch in ((x, 0.0), (0.0, x)):
        set_ess(pch, pdch, pch - pdch, 0.0)
        lo = mch.ess_phi_limits_lower(model, e, s_m, s_o, p, network)
        up = mch.ess_phi_limits_upper(model, e, s_m, s_o, p, network)
        phi['samples'].append({
            'pch': pch, 'pdch': pdch,
            'mode': 'charging' if pch > 0 else 'discharging',
            'qnet_lower_bound': bound_of(lo),
            'qnet_upper_bound': bound_of(up),
        })

    # ess_snet_def invariance under the sign convention
    q_probe = 0.25 * ess.s
    set_ess(x, 0.0, x, q_probe)
    model.es_sch[e, s_m, s_o, p].value = x
    model.es_sdch[e, s_m, s_o, p].value = 0.0
    body_pos = float(pe.value(model.ess_snet_def[e, s_m, s_o, p].body))
    set_ess(x, 0.0, -x, -q_probe)
    body_neg = float(pe.value(model.ess_snet_def[e, s_m, s_o, p].body))
    snet_def = {
        'body_with_pnet_qnet_positive': body_pos,
        'body_with_pnet_qnet_negated': body_neg,
        'invariant_under_sign_flip': abs(body_pos - body_neg) < 1e-12,
    }

    return {
        'ess_index': e, 'bus': ess.bus, 'node_idx': node_idx,
        'charging_only_4_1': charging,
        'discharging_only_4_2': discharging,
        'reactive_4_3_4_4': reactive,
        'power_factor_limits': phi,
        'ess_snet_def_sign_invariance': snet_def,
    }


# ============================================================================
#  C/D/E. Solve the validation case and capture everything
# ============================================================================
def section_cde(dso):
    # NOTE: params.solver_params.verbose is deliberately NOT toggled. In this
    # codebase verbose=True is not observability-only: on an unsuccessful solve
    # _run_smopf() takes a branch that logs infeasible constraints and then
    # calls exit(ERROR_NETWORK_OPTIMIZATION) (network.py:642), killing the
    # process. Solver metrics are therefore taken from the returned result
    # object and from the model itself.
    models = dso.build_model()

    # IPOPT appends to its configured output_file across runs (file_append=yes),
    # so clear the log dir first and parse only what THIS run produced.
    logs_dir = dso.network[CASE_YEAR][CASE_DAY].logs_dir
    if os.path.isdir(logs_dir):
        for name in os.listdir(logs_dir):
            if name.startswith('optim_log_case33_3'):
                try:
                    os.remove(os.path.join(logs_dir, name))
                except OSError:
                    pass

    console = io.StringIO()
    with redirect_stdout(console):
        results = dso.optimize(models)
    console_text = console.getvalue()

    log_text = ''
    log_paths = []
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir)):
            if name.startswith('optim_log_case33_3'):
                path = os.path.join(logs_dir, name)
                log_paths.append(path)
                with open(path, 'rb') as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 400_000))
                    log_text += handle.read().decode('utf-8', errors='replace')

    model = models[CASE_YEAR][CASE_DAY]
    result = results[CASE_YEAR][CASE_DAY]
    network = dso.network[CASE_YEAR][CASE_DAY]

    solver_block = {
        'status': str(getattr(result.solver, 'status', None)),
        'termination_condition': str(getattr(result.solver, 'termination_condition', None)),
        'message': str(getattr(result.solver, 'message', None)),
        'time': getattr(result.solver, 'time', None),
        'ipopt_summary': parse_ipopt_console(log_text or console_text),
        'ipopt_log_paths': log_paths,
        'recovery_used': ('[WARNING] Network primary solve did not converge'
                          in console_text),
        'objective': float(pe.value(model.objective)),
    }

    # --- max primal residual over the ordinary-ESS rows we care about ---
    def max_abs_body(component):
        worst = 0.0
        for index in component:
            worst = max(worst, abs(float(pe.value(component[index].body))))
        return worst

    ess_rows = {}
    for name in ('ess_pnet_def', 'ess_snet_def'):
        if hasattr(model, name):
            ess_rows[f'{name}_max_abs_body'] = max_abs_body(getattr(model, name))

    # --- ESS trajectories + nodal contribution (scenario 0,0) ---
    s_m, s_o = 0, 0
    trajectories = {}
    for e in model.energy_storages:
        ess = network.energy_storages[e]
        node_idx = network.get_node_idx(ess.bus)
        rec = {'es_id': ess.es_id, 'bus': ess.bus,
               's_rated_pu': ess.s, 'e_rated_pu': ess.e,
               'pch': [], 'pdch': [], 'pnet': [], 'qnet': [],
               'sch': [], 'sdch': [], 'soc': [],
               'nodal_p_contribution': [], 'nodal_q_contribution': []}
        for p in model.periods:
            pch = float(pe.value(model.es_pch[e, s_m, s_o, p]))
            pdch = float(pe.value(model.es_pdch[e, s_m, s_o, p]))
            pnet = float(pe.value(model.es_pnet[e, s_m, s_o, p]))
            qnet = float(pe.value(model.es_qnet[e, s_m, s_o, p]))
            rec['pch'].append(pch)
            rec['pdch'].append(pdch)
            rec['pnet'].append(pnet)
            rec['qnet'].append(qnet)
            rec['sch'].append(float(pe.value(model.es_sch[e, s_m, s_o, p])))
            rec['sdch'].append(float(pe.value(model.es_sdch[e, s_m, s_o, p])))
            rec['soc'].append(float(pe.value(model.es_soc[e, s_m, s_o, p])))

            # ESS contribution to the PRODUCTION net-load Expressions
            base_p = float(pe.value(model.pc_node[node_idx, s_m, s_o, p]))
            base_q = float(pe.value(model.qc_node[node_idx, s_m, s_o, p]))
            model.es_pnet[e, s_m, s_o, p].value = 0.0
            model.es_qnet[e, s_m, s_o, p].value = 0.0
            zero_p = float(pe.value(model.pc_node[node_idx, s_m, s_o, p]))
            zero_q = float(pe.value(model.qc_node[node_idx, s_m, s_o, p]))
            model.es_pnet[e, s_m, s_o, p].value = pnet
            model.es_qnet[e, s_m, s_o, p].value = qnet
            rec['nodal_p_contribution'].append(base_p - zero_p)
            rec['nodal_q_contribution'].append(base_q - zero_q)
        trajectories[str(ess.es_id)] = rec

    # --- interface quantities ---
    try:
        interface = dso.process_results_interface(models)[CASE_YEAR][CASE_DAY]
    except Exception as error:
        interface = {'unavailable': str(error)}

    # --- processed results (section 6) ---
    processed = dso.process_results(models, results)
    # NetworkData.process_results returns {'of_value': ..., 'results': {year: {day: ...}}}
    scen = processed['results'][CASE_YEAR][CASE_DAY]['scenarios'][s_m][s_o]['energy_storages']

    base_mva = network.baseMVA
    comparison = {}
    for e in model.energy_storages:
        es_id = network.energy_storages[e].es_id
        raw_p = [float(pe.value(model.es_pnet[e, s_m, s_o, p])) * base_mva
                 for p in model.periods]
        raw_q = [float(pe.value(model.es_qnet[e, s_m, s_o, p])) * base_mva
                 for p in model.periods]
        proc_p = scen['p'][es_id]
        proc_q = scen['q'][es_id]
        dp = [proc_p[i] - raw_p[i] for i in range(len(raw_p))]
        dq = [proc_q[i] - raw_q[i] for i in range(len(raw_q))]
        comparison[str(es_id)] = {
            'raw_model_pnet_MW': raw_p,
            'processed_P_MW': proc_p,
            'raw_model_qnet_MVAr': raw_q,
            'processed_Q_MVAr': proc_q,
            'max_abs_diff_P': max(abs(v) for v in dp),
            'max_abs_diff_Q': max(abs(v) for v in dq),
            'processed_equals_model_P': max(abs(v) for v in dp) < 1e-9,
            'processed_equals_model_Q': max(abs(v) for v in dq) < 1e-9,
            'processed_S_MVA': scen['s'][es_id],
            'processed_SOC_MWh': scen['soc'][es_id],
        }

    return {
        'solver': solver_block,
        'ess_constraint_residuals': ess_rows,
        'trajectories': trajectories,
        'interface': interface,
        'raw_vs_processed': comparison,
        'console_tail': console_text[-4000:],
    }


# ============================================================================
#  F. Reporting-convention injection test (section 6), independent of the solve
# ============================================================================
def section_f_reporting(dso):
    """Inject KNOWN charging/discharging/reactive states into a freshly built
    model and run the real production `process_results`, so the exposed result
    convention is verified even when the local SMOPF does not converge."""
    network = dso.network[CASE_YEAR][CASE_DAY]
    model = network.build_model(dso.params)

    s_m, s_o = 0, 0
    p_charge, p_discharge = 0, 1
    base = network.baseMVA
    injected = {}

    for e in model.energy_storages:
        ess = network.energy_storages[e]
        a = 0.4 * ess.s          # active magnitude, p.u.
        r = 0.25 * ess.s         # reactive magnitude, p.u.

        # period 0: charging + reactive absorption  -> P>0, Q>0
        model.es_pch[e, s_m, s_o, p_charge].value = a
        model.es_pdch[e, s_m, s_o, p_charge].value = 0.0
        model.es_pnet[e, s_m, s_o, p_charge].value = +a
        model.es_qnet[e, s_m, s_o, p_charge].value = +r
        model.es_sch[e, s_m, s_o, p_charge].value = a
        model.es_sdch[e, s_m, s_o, p_charge].value = 0.0

        # period 1: discharging + reactive injection -> P<0, Q<0
        model.es_pch[e, s_m, s_o, p_discharge].value = 0.0
        model.es_pdch[e, s_m, s_o, p_discharge].value = a
        model.es_pnet[e, s_m, s_o, p_discharge].value = -a
        model.es_qnet[e, s_m, s_o, p_discharge].value = -r
        model.es_sch[e, s_m, s_o, p_discharge].value = 0.0
        model.es_sdch[e, s_m, s_o, p_discharge].value = a

        injected[e] = {'es_id': ess.es_id, 'a_pu': a, 'r_pu': r}

    processed = network.process_results(model, dso.params)
    scen = processed['scenarios'][s_m][s_o]['energy_storages']

    out = {}
    for e, meta in injected.items():
        es_id = meta['es_id']
        a, r = meta['a_pu'], meta['r_pu']
        got_p_ch = scen['p'][es_id][p_charge]
        got_q_ch = scen['q'][es_id][p_charge]
        got_p_dis = scen['p'][es_id][p_discharge]
        got_q_dis = scen['q'][es_id][p_discharge]
        out[str(es_id)] = {
            'charging': {
                'model_pnet_MW': +a * base, 'processed_P_MW': got_p_ch,
                'diff_P': got_p_ch - (+a * base),
                'model_qnet_MVAr': +r * base, 'processed_Q_MVAr': got_q_ch,
                'diff_Q': got_q_ch - (+r * base),
                'reported_P_positive': got_p_ch > 0,
                'reported_Q_positive': got_q_ch > 0,
            },
            'discharging': {
                'model_pnet_MW': -a * base, 'processed_P_MW': got_p_dis,
                'diff_P': got_p_dis - (-a * base),
                'model_qnet_MVAr': -r * base, 'processed_Q_MVAr': got_q_dis,
                'diff_Q': got_q_dis - (-r * base),
                'reported_P_negative': got_p_dis < 0,
                'reported_Q_negative': got_q_dis < 0,
            },
        }
        out[str(es_id)]['all_match_model_exactly'] = all(
            abs(out[str(es_id)][k][f'diff_{c}']) < 1e-9
            for k in ('charging', 'discharging') for c in ('P', 'Q'))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pre', 'post'), required=True)
    parser.add_argument('--algebra-only', action='store_true',
                        help='skip the local SMOPF solve (sections C/D/E)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P4.6-B1', 'case': {
        'spec_dir': SPEC_DIR, 'spec_file': SPEC_FILE,
        'dso_connection_node': DSO_CONNECTION_NODE,
        'network': 'case33_3', 'year': CASE_YEAR, 'day': CASE_DAY}}

    report['A_provenance'] = section_a_provenance(args.phase)

    planning, dso = build_planning_and_dso()
    isolate_ordinary_ess(dso)

    report['B_algebra_tests'] = section_b_algebra(dso)
    report['F_reporting_injection'] = section_f_reporting(dso)
    if not args.algebra_only:
        report.update({'C_D_E_solve': section_cde(dso)})

    suffix = '_algebra' if args.algebra_only else ''
    out_path = os.path.join(OUT_DIR, f'p46b1_{args.phase}{suffix}_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[P4.6-B1] phase={args.phase} report written to {out_path}')

    b = report['B_algebra_tests']
    print(f"  4.1 pnet=+pch                 : {b['charging_only_4_1']['pnet_equals_plus_pch']}")
    print(f"  4.1 enters balance as load    : {b['charging_only_4_1']['enters_balance_as_additional_load']}")
    print(f"  4.2 pnet=-pdch                : {b['discharging_only_4_2']['pnet_equals_minus_pdch']}")
    print(f"  4.2 reduces net demand        : {b['discharging_only_4_2']['reduces_net_demand']}")
    print(f"  4.3 qnet>0 increases Q demand : {b['reactive_4_3_4_4']['qnet_positive_increases_reactive_demand']}")
    print(f"  4.4 qnet<0 injects Q          : {b['reactive_4_3_4_4']['qnet_negative_reduces_reactive_demand']}")
    for s_ in b['power_factor_limits']['samples']:
        print(f"  phi {s_['mode']:11s}: qnet in [{s_['qnet_lower_bound']:+.6g}, {s_['qnet_upper_bound']:+.6g}]")
    if 'C_D_E_solve' in report:
        s = report['C_D_E_solve']['solver']
        print(f"  solve: {s['status']} / {s['termination_condition']} | obj={s['objective']:.6f}")
        for es_id, cmp_ in report['C_D_E_solve']['raw_vs_processed'].items():
            print(f"  ES {es_id}: processed==model P {cmp_['processed_equals_model_P']} "
                  f"| Q {cmp_['processed_equals_model_Q']}")
    for es_id, inj in report['F_reporting_injection'].items():
        print(f"  ES {es_id} reporting: exact={inj['all_match_model_exactly']} "
              f"| charge P>0 {inj['charging']['reported_P_positive']}, Q>0 {inj['charging']['reported_Q_positive']}"
              f" | discharge P<0 {inj['discharging']['reported_P_negative']}, Q<0 {inj['discharging']['reported_Q_negative']}")


if __name__ == '__main__':
    main()
