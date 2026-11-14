"""
Stage P4.6-B2 -- ordinary/standard ESS `ess_snet_def` normalization validation.

Runs the SAME measurements before and after the production normalization:

    python p46b2_op1_ess_kappa_validation.py --phase a   # B1-corrected, UNSCALED
    python p46b2_op1_ess_kappa_validation.py --phase b   # B1-corrected, kappa-SCALED

Case (identical to the accepted B1 post-correction solve): OP1 / case33_3 /
2025 / Summer, two ordinary ESS units, RandomSeed 2026, one operation
scenario, BILINEAR_RELAXATION, COST objective, IPOPT + MA97, exact-Hessian
primary with limited-memory recovery.

Discipline (docs/METHODOLOGY.md): this script never reimplements the logic
under test. It calls the real production `build_model` / `optimize` /
`process_results` path and the real rule functions, and only *observes*.

Sections
  A  provenance + verbatim source of the rule and scale helper
  B  construction validation (plan sec. 5)
  C  zero-capacity construction tests (plan sec. 6)
  D  solve + full solver metrics (plan sec. 2 / 7)
  E  scaled and original-unscaled residuals, normalized by S_rated^2
  F  B1 sign-convention re-confirmation (plan sec. 9)
"""

import argparse
import copy
import inspect
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

import model_construction_helpers as mch  # noqa: E402
import network as network_module  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/OP1'
SPEC_FILE = 'OP1.json'
DSO_CONNECTION_NODE = 9          # case33_3
CASE_YEAR = 2025
CASE_DAY = 'Summer'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'OP1', 'Results', 'P46B2')

IPOPT_SUMMARY_RE = {
    'iterations': re.compile(r'Number of Iterations\.*:\s*(\d+)'),
    'objective_scaled_unscaled': re.compile(r'Objective\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'primal_infeasibility': re.compile(r'Constraint violation\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'dual_infeasibility': re.compile(r'Dual infeasibility\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'complementarity': re.compile(r'Complementarity\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'overall_nlp_error': re.compile(r'Overall NLP error\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'exit_status': re.compile(r'EXIT: (.+?)\.'),
}


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def parse_ipopt(text):
    out = {}
    for key, pattern in IPOPT_SUMMARY_RE.items():
        match = None
        for match in pattern.finditer(text):
            pass
        if match:
            out[key] = match.groups() if len(match.groups()) > 1 else match.group(1)
    return out


def src(obj):
    try:
        return inspect.getsource(obj)
    except Exception as error:
        return f'<unavailable: {error}>'


# ============================================================================
#  Model construction (production path)
# ============================================================================
def build_planning_and_dso():
    planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
    planning.read_planning_problem()
    dso = planning.distribution_networks[DSO_CONNECTION_NODE]
    # uncoordinated local-solve recipe documented in main.py
    for year in dso.years:
        for day in dso.days:
            dso.network[year][day].shared_energy_storages = list()
    return planning, dso


# ============================================================================
#  A. Provenance
# ============================================================================
def section_a(phase):
    return {
        'phase': phase,
        'git_head': git_head(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ess_snet_def_rule_source': src(mch.ess_snet_def_rule),
        'ess_pnet_rule_source': src(mch.ess_pnet_rule),
        'ess_comp_rule_source': src(mch.ess_comp_rule),
        'ess_soc_rule_source': src(mch.ess_soc_rule),
        'ess_s_limit_rule_source': src(mch.ess_s_limit_rule),
        'ordinary_scale_helper_present': hasattr(mch, 'ordinary_ess_snet_def_scale'),
        'ordinary_scale_helper_source': (
            src(mch.ordinary_ess_snet_def_scale)
            if hasattr(mch, 'ordinary_ess_snet_def_scale') else None),
        'shared_sync_helper_still_shared_only': hasattr(mch, '_sync_sess_snet_def_scale'),
        'no_ordinary_dual_remapping_helper': not any(
            name for name in dir(mch)
            if 'sync' in name.lower() and 'ess' in name.lower() and 'sess' not in name.lower()),
    }


# ============================================================================
#  B. Construction validation (plan sec. 5)
# ============================================================================
def section_b(dso):
    network = dso.network[CASE_YEAR][CASE_DAY]
    model = network.build_model(dso.params)

    comp = model.ess_snet_def
    rows = list(comp)
    total_con_data = sum(len(list(c.values()))
                         for c in model.component_objects(pe.Constraint, active=None))
    con_names = sorted(c.local_name for c in model.component_objects(pe.Constraint, active=None))

    units = {}
    for e in model.energy_storages:
        ess = network.energy_storages[e]
        entry = {
            'es_id': ess.es_id, 'bus': ess.bus,
            's_rated_pu': ess.s,
            's_rated_MVA': ess.s * network.baseMVA,
            'expected_kappa': (1.0 / ess.s) if ess.s else None,
        }
        if hasattr(model, 'ess_snet_def_scale'):
            component = model.ess_snet_def_scale
            param = component[e]
            entry['kappa_in_model'] = float(pe.value(param))
            entry['kappa_matches_expected'] = abs(
                entry['kappa_in_model'] - entry['expected_kappa']) < 1e-9
            entry['scale_component_is_param'] = isinstance(component, pe.Param)
            entry['scale_is_mutable'] = bool(getattr(component, 'mutable', False))
            # an immutable Pyomo Param resolves to a plain Python float, i.e.
            # the scale is fixed numerical data rather than a model object
            entry['scale_is_plain_numeric_data'] = isinstance(param, float)
            entry['scale_is_not_a_variable'] = not isinstance(param, pe.Var)
        units[str(ess.es_id)] = entry

    # feasible-set equivalence: g == 0  <=>  kappa*g == 0, and sign preservation
    s_m, s_o, p = 0, 0, 0
    e0 = 0
    ess0 = network.energy_storages[e0]
    kappa0 = 1.0 / ess0.s
    a = 0.4 * ess0.s

    def set_state(sch, sdch, pnet, qnet):
        model.es_sch[e0, s_m, s_o, p].value = sch
        model.es_sdch[e0, s_m, s_o, p].value = sdch
        model.es_pnet[e0, s_m, s_o, p].value = pnet
        model.es_qnet[e0, s_m, s_o, p].value = qnet

    def raw_g():
        sch = pe.value(model.es_sch[e0, s_m, s_o, p])
        sdch = pe.value(model.es_sdch[e0, s_m, s_o, p])
        pnet = pe.value(model.es_pnet[e0, s_m, s_o, p])
        qnet = pe.value(model.es_qnet[e0, s_m, s_o, p])
        return (sch - sdch) ** 2 - pnet ** 2 - qnet ** 2

    equivalence = []
    # on-surface, positive-residual, negative-residual
    for label, (sch, sdch, pnet, qnet) in {
        'on_surface': (a, 0.0, a, 0.0),
        'positive_residual': (a, 0.0, 0.5 * a, 0.0),
        'negative_residual': (0.5 * a, 0.0, a, 0.0),
    }.items():
        set_state(sch, sdch, pnet, qnet)
        g = raw_g()
        body = float(pe.value(model.ess_snet_def[e0, s_m, s_o, p].body))
        equivalence.append({
            'point': label,
            'g_es': g,
            'constraint_body': body,
            'body_equals_kappa_times_g': abs(body - kappa0 * g) < 1e-14 * max(1.0, abs(kappa0 * g)),
            'body_equals_g': abs(body - g) < 1e-18,
            'zero_iff_zero': (abs(g) < 1e-18) == (abs(body) < 1e-18),
            'sign_preserved': (g == 0 and body == 0) or (g * body > 0),
        })

    # no symbolic division by a decision variable in the row body
    body_expr = model.ess_snet_def[e0, s_m, s_o, p].body
    body_str = str(body_expr)
    return {
        'component_local_name': comp.local_name,
        'component_is_ess_snet_def': comp.local_name == 'ess_snet_def',
        'row_count': len(rows),
        'index_tuples': [list(r) for r in rows],
        'total_constraint_data_count': total_con_data,
        'constraint_component_names': con_names,
        'has_replacement_component': any(
            n.startswith('ess_snet_def') and n != 'ess_snet_def' for n in con_names),
        'scale_component_present': hasattr(model, 'ess_snet_def_scale'),
        'units': units,
        'equivalence_points': equivalence,
        'body_expression_sample': body_str[:400],
        'body_contains_division_by_variable': '/' in body_str,
        'ess_comp_row_count': len(list(model.ess_comp)) if hasattr(model, 'ess_comp') else 0,
        'ess_s_limit_row_count': len(list(model.ess_s_limit)) if hasattr(model, 'ess_s_limit') else 0,
        'ess_soc_def_row_count': len(list(model.ess_soc_def)) if hasattr(model, 'ess_soc_def') else 0,
        'ess_pnet_def_row_count': len(list(model.ess_pnet_def)) if hasattr(model, 'ess_pnet_def') else 0,
    }


# ============================================================================
#  C. Zero-capacity construction tests (plan sec. 6)
# ============================================================================
def section_c(dso):
    out = {}

    # --- A: network with NO ordinary ESS ---
    net_a = copy.deepcopy(dso.network[CASE_YEAR][CASE_DAY])
    net_a.energy_storages = list()
    try:
        model_a = net_a.build_model(dso.params)
        rows = len(list(model_a.ess_snet_def)) if hasattr(model_a, 'ess_snet_def') else 0
        out['no_ordinary_ess'] = {
            'constructed': True, 'ess_snet_def_rows': rows,
            'rows_are_zero': rows == 0, 'exception': None,
        }
    except Exception as error:
        out['no_ordinary_ess'] = {
            'constructed': False, 'exception': f'{type(error).__name__}: {error}'}

    # --- B: explicit zero / below-tolerance rated ordinary ESS ---
    for label, s_value in (('zero_rated', 0.0), ('below_tolerance_rated', 1e-14)):
        net_b = copy.deepcopy(dso.network[CASE_YEAR][CASE_DAY])
        degenerate = copy.deepcopy(net_b.energy_storages[0])
        degenerate.s = s_value
        net_b.energy_storages = [degenerate]
        try:
            net_b.build_model(dso.params)
            out[label] = {'rejected': False, 'exception': None,
                          'note': 'model constructed -- degenerate row NOT rejected'}
        except Exception as error:
            message = str(error)
            out[label] = {
                'rejected': True,
                'exception_type': type(error).__name__,
                'message': message,
                'mentions_rated_power': any(
                    token in message.lower()
                    for token in ('rated', 'apparent', 's_rated', 'capacity')),
                'identifies_energy_storage': 'energy storage' in message.lower()
                                             or 'ess' in message.lower(),
            }
    return out


# ============================================================================
#  D/E/F. Solve, residuals, sign re-confirmation
# ============================================================================
def section_def(dso):
    models = dso.build_model()

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
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir)):
            if name.startswith('optim_log_case33_3'):
                with open(os.path.join(logs_dir, name), 'rb') as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 400_000))
                    log_text += handle.read().decode('utf-8', errors='replace')

    model = models[CASE_YEAR][CASE_DAY]
    result = results[CASE_YEAR][CASE_DAY]
    network = dso.network[CASE_YEAR][CASE_DAY]
    base = network.baseMVA
    s_m, s_o = 0, 0

    solver_block = {
        'status': str(getattr(result.solver, 'status', None)),
        'termination_condition': str(getattr(result.solver, 'termination_condition', None)),
        'message': str(getattr(result.solver, 'message', None)),
        'time': getattr(result.solver, 'time', None),
        'ipopt_summary': parse_ipopt(log_text or console_text),
        'recovery_used': '[INFO] Retrying network solve once' in console_text,
        'objective': float(pe.value(model.objective)),
    }

    # --- residuals: scaled row body and ORIGINAL unscaled g_es ---
    residuals = {}
    for e in model.energy_storages:
        ess = network.energy_storages[e]
        s_rated = ess.s
        worst_scaled = 0.0
        worst_g = 0.0
        worst_g_norm = 0.0
        for p in model.periods:
            sch = pe.value(model.es_sch[e, s_m, s_o, p])
            sdch = pe.value(model.es_sdch[e, s_m, s_o, p])
            pnet = pe.value(model.es_pnet[e, s_m, s_o, p])
            qnet = pe.value(model.es_qnet[e, s_m, s_o, p])
            g = (sch - sdch) ** 2 - pnet ** 2 - qnet ** 2
            body = float(pe.value(model.ess_snet_def[e, s_m, s_o, p].body))
            worst_scaled = max(worst_scaled, abs(body))
            worst_g = max(worst_g, abs(g))
            worst_g_norm = max(worst_g_norm, abs(g) / (s_rated ** 2))
        residuals[str(ess.es_id)] = {
            's_rated_pu': s_rated,
            'max_abs_scaled_row_body': worst_scaled,
            'max_abs_original_g_es': worst_g,
            'max_abs_original_g_es_normalized_by_s_rated_sq': worst_g_norm,
        }

    # --- trajectories ---
    trajectories = {}
    for e in model.energy_storages:
        ess = network.energy_storages[e]
        node_idx = network.get_node_idx(ess.bus)
        rec = {'es_id': ess.es_id, 'bus': ess.bus, 'pch': [], 'pdch': [],
               'pnet': [], 'qnet': [], 'sch': [], 'sdch': [], 'soc': [],
               'nodal_p_contribution': [], 'nodal_q_contribution': []}
        for p in model.periods:
            pnet = float(pe.value(model.es_pnet[e, s_m, s_o, p]))
            qnet = float(pe.value(model.es_qnet[e, s_m, s_o, p]))
            rec['pch'].append(float(pe.value(model.es_pch[e, s_m, s_o, p])))
            rec['pdch'].append(float(pe.value(model.es_pdch[e, s_m, s_o, p])))
            rec['pnet'].append(pnet)
            rec['qnet'].append(qnet)
            rec['sch'].append(float(pe.value(model.es_sch[e, s_m, s_o, p])))
            rec['sdch'].append(float(pe.value(model.es_sdch[e, s_m, s_o, p])))
            rec['soc'].append(float(pe.value(model.es_soc[e, s_m, s_o, p])))
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

    try:
        interface = dso.process_results_interface(models)[CASE_YEAR][CASE_DAY]
    except Exception as error:
        interface = {'unavailable': str(error)}

    # --- F: sign-convention re-confirmation on processed results ---
    processed = dso.process_results(models, results)
    scen = processed['results'][CASE_YEAR][CASE_DAY]['scenarios'][s_m][s_o]['energy_storages']
    sign_check = {}
    for e in model.energy_storages:
        es_id = network.energy_storages[e].es_id
        raw_p = [float(pe.value(model.es_pnet[e, s_m, s_o, p])) * base for p in model.periods]
        raw_q = [float(pe.value(model.es_qnet[e, s_m, s_o, p])) * base for p in model.periods]
        dp = [scen['p'][es_id][i] - raw_p[i] for i in range(len(raw_p))]
        dq = [scen['q'][es_id][i] - raw_q[i] for i in range(len(raw_q))]
        charging = [i for i in range(len(raw_p)) if raw_p[i] > 1e-9]
        discharging = [i for i in range(len(raw_p)) if raw_p[i] < -1e-9]
        sign_check[str(es_id)] = {
            'max_abs_diff_P': max(abs(v) for v in dp),
            'max_abs_diff_Q': max(abs(v) for v in dq),
            'processed_equals_model_P': max(abs(v) for v in dp) < 1e-9,
            'processed_equals_model_Q': max(abs(v) for v in dq) < 1e-9,
            'num_charging_periods': len(charging),
            'num_discharging_periods': len(discharging),
            'charging_periods_all_report_positive_P': all(
                scen['p'][es_id][i] > 0 for i in charging),
            'discharging_periods_all_report_negative_P': all(
                scen['p'][es_id][i] < 0 for i in discharging),
            'processed_P_MW': scen['p'][es_id],
            'processed_Q_MVAr': scen['q'][es_id],
        }

    return {'solver': solver_block, 'residuals': residuals,
            'trajectories': trajectories, 'interface': interface,
            'sign_reconfirmation': sign_check,
            'console_tail': console_text[-3000:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('a', 'b'), required=True,
                        help='a = unscaled baseline (B2-A), b = kappa-scaled (B2-B)')
    parser.add_argument('--construction-only', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P4.6-B2', 'phase': args.phase, 'case': {
        'spec_dir': SPEC_DIR, 'spec_file': SPEC_FILE,
        'dso_connection_node': DSO_CONNECTION_NODE,
        'network': 'case33_3', 'year': CASE_YEAR, 'day': CASE_DAY}}

    report['A_provenance'] = section_a(args.phase)
    planning, dso = build_planning_and_dso()
    report['B_construction'] = section_b(dso)
    report['C_zero_capacity'] = section_c(dso)
    if not args.construction_only:
        report['D_E_F_solve'] = section_def(dso)

    suffix = '_construction' if args.construction_only else ''
    out_path = os.path.join(OUT_DIR, f'p46b2_phase{args.phase}{suffix}_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[P4.6-B2] phase={args.phase} report written to {out_path}')

    b = report['B_construction']
    print(f"  component={b['component_local_name']} rows={b['row_count']} "
          f"scale_present={b['scale_component_present']} replacement={b['has_replacement_component']}")
    for es_id, u in b['units'].items():
        print(f"  ES {es_id}: s_rated={u['s_rated_pu']} p.u. expected_kappa={u['expected_kappa']} "
              f"in_model={u.get('kappa_in_model')} match={u.get('kappa_matches_expected')}")
    for pt in b['equivalence_points']:
        print(f"  equiv {pt['point']:18s}: g={pt['g_es']:+.6e} body={pt['constraint_body']:+.6e} "
              f"zero_iff_zero={pt['zero_iff_zero']} sign_preserved={pt['sign_preserved']}")
    c = report['C_zero_capacity']
    print(f"  no-ESS network: {c['no_ordinary_ess']}")
    for k in ('zero_rated', 'below_tolerance_rated'):
        print(f"  {k}: rejected={c[k].get('rejected')} {str(c[k].get('message'))[:90]}")
    if 'D_E_F_solve' in report:
        s = report['D_E_F_solve']['solver']
        print(f"  solve: {s['status']}/{s['termination_condition']} obj={s['objective']:.6f} "
              f"iters={s['ipopt_summary'].get('iterations')} recovery={s['recovery_used']}")
        for es_id, r in report['D_E_F_solve']['residuals'].items():
            print(f"  ES {es_id} residual: scaled_body={r['max_abs_scaled_row_body']:.3e} "
                  f"g_es={r['max_abs_original_g_es']:.3e} "
                  f"g/s^2={r['max_abs_original_g_es_normalized_by_s_rated_sq']:.3e}")
        for es_id, sc in report['D_E_F_solve']['sign_reconfirmation'].items():
            print(f"  ES {es_id} sign: dP={sc['max_abs_diff_P']:.2e} dQ={sc['max_abs_diff_Q']:.2e} "
                  f"chg+={sc['charging_periods_all_report_positive_P']} "
                  f"dis-={sc['discharging_periods_all_report_negative_P']}")


if __name__ == '__main__':
    main()
