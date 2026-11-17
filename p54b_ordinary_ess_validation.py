"""
Stage P5.4-B -- ordinary/standard network ESS active-energy parity validation.

The ordinary ESS now uses the same active-energy formulation as the shared ESS
(P5.4-A): `es_sch`/`es_sdch`, `ess_snet_def` and its kappa_es machinery,
`ess_pch_link`, `ess_pdch_link` and the apparent `ess_s_limit` are retired, and
replaced by an active SOC recursion, `ess_converter_capability`,
`ess_active_sum_limit`, and complementarity on `es_pch * es_pdch`.

SRP1 instantiates no ordinary ESS, so the live case is OP1 / case33_3 / 2025 /
Summer (two ordinary ESS units) -- the same case the accepted P4.6-B1/B2 stages
used, so the comparison is like-for-like.

Discipline (docs/METHODOLOGY.md): never reimplements the logic under test. It
calls the real production `build_model` / `optimize` / `process_results` path
and the real rule functions, and only observes.

Sections
  A  provenance + verbatim source of the production rules
  B  construction validation (retired absent, new present, bounds)
  C  physics tests: pure reactive / pure charge / pure discharge
  D  zero-capacity and no-ESS construction behaviour
  E  OP1 solve + full solver metrics
  F  normalized residuals + P5.4-E2 circulating-power instrumentation
  G  load-positive sign-convention re-confirmation on processed results

    python p54b_ordinary_ess_validation.py
"""

import copy
import inspect
import io
import json
import os
import re
import statistics
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
from definitions import ESS_COMPLEMENTARITY_TOLERANCE  # noqa: E402
from model_construction_helpers import period_duration_hours  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/OP1'
SPEC_FILE = 'OP1.json'
DSO_CONNECTION_NODE = 9          # case33_3
CASE_YEAR = 2025
CASE_DAY = 'Summer'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'OP1', 'Results', 'P54B')

RETIRED_VARS = ('es_sch', 'es_sdch')
RETIRED_COMPONENTS = ('ess_snet_def', 'ess_snet_def_scale', 'ess_pch_link',
                      'ess_pdch_link', 'ess_s_limit')
NEW_COMPONENTS = ('ess_converter_capability', 'ess_active_sum_limit',
                  # P5.4-H1: dimensionless complementarity link rows
                  'ess_pch_hat_link', 'ess_pdch_hat_link')

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


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


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
def section_a():
    return {
        'stage': 'P5.4-B',
        'git_head': git_head(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ess_soc_rule_source': src(mch.ess_soc_rule),
        'ess_comp_rule_source': src(mch.ess_comp_rule),
        'ess_converter_capability_rule_source': src(mch.ess_converter_capability_rule),
        'ess_active_sum_limit_rule_source': src(mch.ess_active_sum_limit_rule),
        'ess_pnet_rule_source': src(mch.ess_pnet_rule),
        'retired_helpers_absent': {
            name: not hasattr(mch, name)
            for name in ('ordinary_ess_snet_def_scale', 'ess_snet_def_scale_init',
                         'ess_snet_def_rule', 'ess_pch_link_rule',
                         'ess_pdch_link_rule', 'ess_s_limit_rule', 's_bounds')},
    }


# ============================================================================
#  B. Construction validation
# ============================================================================
def section_b(dso):
    network = dso.network[CASE_YEAR][CASE_DAY]
    model = network.build_model(dso.params)

    con_names = sorted(c.local_name for c in model.component_objects(pe.Constraint, active=None))
    units = {}
    for e in model.energy_storages:
        ess = network.energy_storages[e]
        s_m, s_o, p = 0, 0, 0
        units[str(ess.es_id)] = {
            'es_id': ess.es_id, 'bus': ess.bus,
            's_rated_pu': ess.s, 's_rated_MVA': ess.s * network.baseMVA,
            'e_rated_pu': ess.e,
            'pch_bounds': list(model.es_pch[e, s_m, s_o, p].bounds),
            'pdch_bounds': list(model.es_pdch[e, s_m, s_o, p].bounds),
            'pnet_bounds': list(model.es_pnet[e, s_m, s_o, p].bounds),
            'qnet_bounds': list(model.es_qnet[e, s_m, s_o, p].bounds),
        }

    e0, s_m, s_o, p = 0, 0, 0, 0
    cap_body = str(model.ess_converter_capability[e0, s_m, s_o, p].body)
    sum_body = str(model.ess_active_sum_limit[e0, s_m, s_o, p].body)
    comp_body = str(model.ess_comp[e0, s_m, s_o, p].body)
    soc_body = str(model.ess_soc_def[e0, s_m, s_o, p].body)

    return {
        'period_duration_hours': float(period_duration_hours(model)),
        'retired_vars_absent': {n: not hasattr(model, n) for n in RETIRED_VARS},
        'retired_components_absent': {n: not hasattr(model, n) for n in RETIRED_COMPONENTS},
        'new_components_present': {n: hasattr(model, n) for n in NEW_COMPONENTS},
        'new_component_row_counts': {
            n: len(list(getattr(model, n))) for n in NEW_COMPONENTS if hasattr(model, n)},
        'ess_comp_row_count': len(list(model.ess_comp)),
        'ess_soc_def_row_count': len(list(model.ess_soc_def)),
        'ess_pnet_def_row_count': len(list(model.ess_pnet_def)),
        'constraint_component_names': con_names,
        'units': units,
        'capability_body_sample': cap_body[:300],
        'active_sum_body_sample': sum_body[:300],
        'complementarity_body_sample': comp_body[:300],
        'soc_body_sample': soc_body[:300],
        # P5.4-H1: the complementarity row now acts on the DIMENSIONLESS pair.
        # Check for the hat variables specifically -- 'es_pch' is a substring of
        # 'es_pch_hat', so a substring test on the physical name would pass for
        # the wrong reason.
        'complementarity_acts_on_normalized_powers': (
            'es_pch_hat' in comp_body and 'es_pdch_hat' in comp_body
            and 'es_sch' not in comp_body and 'es_sdch' not in comp_body),
        'complementarity_rhs': float(pe.value(model.ess_comp[e0, s_m, s_o, p].upper)),
        'complementarity_rhs_is_eps': abs(
            float(pe.value(model.ess_comp[e0, s_m, s_o, p].upper))
            - ESS_COMPLEMENTARITY_TOLERANCE) < 1e-18,
        'link_rows_have_unit_coefficient_on_physical': all(
            str(getattr(model, n)[e0, s_m, s_o, p].body).startswith(v)
            for n, v in (('ess_pch_hat_link', 'es_pch['),
                         ('ess_pdch_hat_link', 'es_pdch['))),
        'hat_bounds': [list(model.es_pch_hat[e0, s_m, s_o, p].bounds),
                       list(model.es_pdch_hat[e0, s_m, s_o, p].bounds)],
        'soc_acts_on_active_powers': (
            'es_pch' in soc_body and 'es_pdch' in soc_body
            and 'es_sch' not in soc_body and 'es_sdch' not in soc_body),
        'no_division_by_variable_in_new_rows': not any(
            '/' in b for b in (cap_body, sum_body)),
    }


# ============================================================================
#  C. Physics tests -- pure reactive, pure charge, pure discharge
# ============================================================================
def section_c(dso):
    network = dso.network[CASE_YEAR][CASE_DAY]
    model = network.build_model(dso.params)
    e0, s_m, s_o, p = 0, 0, 0, 0
    ess = network.energy_storages[e0]
    dt = period_duration_hours(model)
    a = 0.4 * ess.s

    # Measure the SOC increment the PRODUCTION row implies, without assuming how
    # Pyomo laid the row out: evaluate the row body at the test dispatch and at
    # zero dispatch, and difference them. Whatever constant terms the row carries
    # cancel, leaving exactly the dispatch-dependent delta.
    def measured_delta(pch, pdch, qnet):
        model.es_pch[e0, s_m, s_o, p].value = pch
        model.es_pdch[e0, s_m, s_o, p].value = pdch
        model.es_qnet[e0, s_m, s_o, p].value = qnet
        model.es_pnet[e0, s_m, s_o, p].value = pch - pdch
        model.es_soc[e0, s_m, s_o, p].value = 0.0
        zero_state = float(pe.value(model.ess_soc_def[e0, s_m, s_o, p].body))
        model.es_pch[e0, s_m, s_o, p].value = 0.0
        model.es_pdch[e0, s_m, s_o, p].value = 0.0
        model.es_qnet[e0, s_m, s_o, p].value = 0.0
        model.es_pnet[e0, s_m, s_o, p].value = 0.0
        baseline = float(pe.value(model.ess_soc_def[e0, s_m, s_o, p].body))
        return baseline - zero_state

    tests = {
        'pure_reactive': {'pch': 0.0, 'pdch': 0.0, 'qnet': a,
                          'expected': 0.0},
        'pure_charging': {'pch': a, 'pdch': 0.0, 'qnet': 0.0,
                          'expected': ess.eff_ch * a * dt},
        'pure_discharging': {'pch': 0.0, 'pdch': a, 'qnet': 0.0,
                             'expected': -a * dt / ess.eff_dch},
    }
    out = {'dt_hours': dt, 's_rated_pu': ess.s,
           'eff_ch': ess.eff_ch, 'eff_dch': ess.eff_dch, 'test_power_pu': a}
    for label, spec in tests.items():
        measured = measured_delta(spec['pch'], spec['pdch'], spec['qnet'])
        out[label] = {
            'measured_delta_soc': measured,
            'expected_delta_soc': spec['expected'],
            'exact': measured == spec['expected'],
            'abs_error': abs(measured - spec['expected']),
        }
    return out


# ============================================================================
#  D. Zero-capacity / no-ESS construction behaviour
# ============================================================================
def section_d(dso):
    out = {}
    net_a = copy.deepcopy(dso.network[CASE_YEAR][CASE_DAY])
    net_a.energy_storages = list()
    try:
        model_a = net_a.build_model(dso.params)
        rows = {n: (len(list(getattr(model_a, n))) if hasattr(model_a, n) else 0)
                for n in NEW_COMPONENTS + ('ess_comp', 'ess_soc_def')}
        out['no_ordinary_ess'] = {'constructed': True, 'row_counts': rows,
                                  'all_zero': all(v == 0 for v in rows.values()),
                                  'exception': None}
    except Exception as error:
        out['no_ordinary_ess'] = {'constructed': False,
                                  'exception': f'{type(error).__name__}: {error}'}

    for label, s_value in (('zero_rated', 0.0), ('below_tolerance_rated', 1e-14)):
        net_b = copy.deepcopy(dso.network[CASE_YEAR][CASE_DAY])
        degenerate = copy.deepcopy(net_b.energy_storages[0])
        degenerate.s = s_value
        net_b.energy_storages = [degenerate]
        try:
            model_b = net_b.build_model(dso.params)
            e0, s_m, s_o, p = 0, 0, 0, 0
            out[label] = {
                'constructed': True, 'exception': None,
                'pch_bounds': list(model_b.es_pch[e0, s_m, s_o, p].bounds),
                'capability_rhs': float(pe.value(
                    model_b.ess_converter_capability[e0, s_m, s_o, p].upper)),
                'note': ('no kappa_es scale exists any more, so a zero rating no '
                         'longer produces a division and no longer needs to be '
                         'rejected at construction; the rows simply collapse'),
            }
        except Exception as error:
            out[label] = {'constructed': False,
                          'exception': f'{type(error).__name__}: {error}'}
    return out


# ============================================================================
#  E/F/G. Solve, residuals + circulating instrumentation, sign check
# ============================================================================
def section_efg(dso):
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
    dt = period_duration_hours(model)
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

    rows, residuals = [], {}
    for e in model.energy_storages:
        ess = network.energy_storages[e]
        s = ess.s
        worst_cap = worst_comp = 0.0
        day_loss = day_legit = 0.0
        for p in model.periods:
            pch = float(pe.value(model.es_pch[e, s_m, s_o, p]))
            pdch = float(pe.value(model.es_pdch[e, s_m, s_o, p]))
            pnet = float(pe.value(model.es_pnet[e, s_m, s_o, p]))
            qnet = float(pe.value(model.es_qnet[e, s_m, s_o, p]))
            cap = max((pnet ** 2 + qnet ** 2 - s ** 2) / s ** 2, 0.0)
            comp = max((pch * pdch - ESS_COMPLEMENTARITY_TOLERANCE * s ** 2) / s ** 2, 0.0)
            worst_cap = max(worst_cap, cap)
            worst_comp = max(worst_comp, comp)
            c = min(pch, pdch)
            loss = c * dt * (1.0 / ess.eff_dch - ess.eff_ch)
            day_loss += loss
            day_legit += ess.eff_ch * pch * dt + pdch * dt / ess.eff_dch
            rows.append({
                'es_id': ess.es_id, 'period': p, 's_rated': s,
                'pch': pch, 'pdch': pdch, 'pnet': pnet, 'qnet': qnet,
                'soc': float(pe.value(model.es_soc[e, s_m, s_o, p])),
                'pch_norm': pch / s, 'pdch_norm': pdch / s,
                'p_circ_norm': c / s, 'p_circ_MW': c * base,
                'p_net_norm': abs(pch - pdch) / s,
                'r_prod': pch * pdch / s ** 2,
                'r_violation': comp,
                'capability_violation': cap,
                'E_circ_loss_MWh': loss * base,
            })
        residuals[str(ess.es_id)] = {
            's_rated_pu': s, 'e_rated_pu': ess.e,
            'max_capability_violation_normalized': worst_cap,
            'max_complementarity_violation_normalized': worst_comp,
            'E_circ_loss_day_MWh': day_loss * base,
            'E_circ_loss_day_over_E_rated': (day_loss / ess.e) if ess.e else None,
            'circ_loss_share_of_throughput': (day_loss / day_legit) if day_legit > 0 else 0.0,
        }

    circ = [r['p_circ_norm'] for r in rows]
    comp_v = [r['r_violation'] for r in rows]
    cap_v = [r['capability_violation'] for r in rows]
    instrumentation = {
        'n_rows': len(rows),
        'p_circ_norm': {'max': max(circ), 'mean': statistics.fmean(circ),
                        'median': statistics.median(circ),
                        'p95': percentile(circ, 0.95), 'p99': percentile(circ, 0.99),
                        'n_nonzero': sum(1 for v in circ if v > 0.0),
                        'above_1e-4_S': sum(1 for v in circ if v > 1e-4),
                        'above_1e-3_S': sum(1 for v in circ if v > 1e-3),
                        'above_1e-2_S': sum(1 for v in circ if v > 1e-2)},
        'p_circ_MW_max': max(r['p_circ_MW'] for r in rows),
        'complementarity_violation': {'max': max(comp_v),
                                      'n_violating': sum(1 for v in comp_v if v > 0.0)},
        'capability_violation': {'max': max(cap_v),
                                 'n_violating': sum(1 for v in cap_v if v > 0.0)},
        'worst_10_by_circulating_power': sorted(rows, key=lambda r: -r['p_circ_norm'])[:10],
    }

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
        reported_s = scen['s'][es_id]
        expected_s = [(raw_p[i] ** 2 + raw_q[i] ** 2) ** 0.5 for i in range(len(raw_p))]
        sign_check[str(es_id)] = {
            'processed_equals_model_P': max(abs(v) for v in dp) < 1e-9,
            'processed_equals_model_Q': max(abs(v) for v in dq) < 1e-9,
            'num_charging_periods': len(charging),
            'num_discharging_periods': len(discharging),
            'charging_periods_all_report_positive_P': all(
                scen['p'][es_id][i] > 0 for i in charging),
            'discharging_periods_all_report_negative_P': all(
                scen['p'][es_id][i] < 0 for i in discharging),
            'reported_s_is_converter_loading_magnitude': max(
                abs(reported_s[i] - expected_s[i]) for i in range(len(expected_s))) < 1e-9,
            'reported_s_is_nonnegative': all(v >= 0.0 for v in reported_s),
        }

    return ({'solver': solver_block, 'residuals': residuals,
             'console_tail': console_text[-3000:]},
            instrumentation, sign_check)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'A_provenance': section_a()}

    print('[P5.4-B] building OP1 planning problem ...', flush=True)
    _planning, dso = build_planning_and_dso()

    print('[P5.4-B] B construction ...', flush=True)
    report['B_construction'] = section_b(dso)
    print('[P5.4-B] C physics ...', flush=True)
    report['C_physics'] = section_c(dso)
    print('[P5.4-B] D zero-capacity ...', flush=True)
    report['D_zero_capacity'] = section_d(dso)
    print('[P5.4-B] E/F/G solve ...', flush=True)
    solve, instrumentation, sign_check = section_efg(dso)
    report['E_solve'] = solve
    report['F_circulating_instrumentation'] = instrumentation
    report['G_sign_reconfirmation'] = sign_check

    out = os.path.join(OUT_DIR, 'p54b_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    b, c, d = report['B_construction'], report['C_physics'], report['D_zero_capacity']
    print(f'\n[P5.4-B] report -> {out}')
    print(f"  dt = {b['period_duration_hours']} h")
    print(f"  retired vars absent      : {b['retired_vars_absent']}")
    print(f"  retired components absent: {b['retired_components_absent']}")
    print(f"  new components present   : {b['new_components_present']} "
          f"rows={b['new_component_row_counts']}")
    print(f"  complementarity on normalized: {b['complementarity_acts_on_normalized_powers']} "
          f"(RHS={b['complementarity_rhs']:.4e}, is_eps={b['complementarity_rhs_is_eps']}); "
          f"SOC on active: {b['soc_acts_on_active_powers']}")
    print(f"  link rows unit coeff on physical: {b['link_rows_have_unit_coefficient_on_physical']}; "
          f"hat bounds {b['hat_bounds']}")
    for label in ('pure_reactive', 'pure_charging', 'pure_discharging'):
        t = c[label]
        print(f"  {label:18s} delta_soc={t['measured_delta_soc']:+.6e} "
              f"expected={t['expected_delta_soc']:+.6e} exact={t['exact']}")
    print(f"  no_ordinary_ess: {d['no_ordinary_ess'].get('all_zero')}  "
          f"zero_rated constructed={d['zero_rated'].get('constructed')}")
    s = solve['solver']
    print(f"  solve: {s['status']} / {s['termination_condition']} "
          f"iters={s['ipopt_summary'].get('iterations')} recovery={s['recovery_used']}")
    f = instrumentation
    print(f"  capability violations: max={f['capability_violation']['max']:.4e} "
          f"n={f['capability_violation']['n_violating']}")
    print(f"  complementarity viol.: max={f['complementarity_violation']['max']:.4e} "
          f"n={f['complementarity_violation']['n_violating']} of {f['n_rows']}")
    pc = f['p_circ_norm']
    print(f"  min(pch,pdch)/S: max={pc['max']:.4e} mean={pc['mean']:.4e} "
          f"median={pc['median']:.4e} p95={pc['p95']:.4e} p99={pc['p99']:.4e}")
    print(f"     >1e-4 S: {pc['above_1e-4_S']}  >1e-3 S: {pc['above_1e-3_S']}  "
          f">1e-2 S: {pc['above_1e-2_S']}  max {f['p_circ_MW_max']:.4e} MW")
    for es_id, r in report['E_solve']['residuals'].items():
        print(f"  ESS {es_id}: circ loss/day {r['E_circ_loss_day_MWh']:.4e} MWh "
              f"= {r['E_circ_loss_day_over_E_rated']:.4e} E_rated, "
              f"{r['circ_loss_share_of_throughput']:.4e} of throughput")
    print(f"  sign/units re-confirmation: {json.dumps(sign_check, default=str)[:400]}")


if __name__ == '__main__':
    main()
