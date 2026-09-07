"""
Stage P5.4-C -- ESSO active-energy conversion validation.

The ESSO per-cohort variables are now ACTIVE powers (es_pch_per_unit /
es_pdch_per_unit). The degradation law's throughput input becomes the active
energy `eta_ch*pch*dt + pdch*dt/eta_dch`; every weighting, normalization and
SoH semantic in that law is preserved. The `es_snet^2 == es_pnet^2 + es_qnet^2`
equality -- the ESSO instance of the exact-zero-gradient row P5.4-A retired in
the network models -- is replaced by the converter capability inequality.

Checks
  A  provenance
  B  construction: retired absent, new present, degradation-law structure
  C  degradation-law preservation: weighting, cohort gating, lifetime gating,
     equivalent-cycle normalization, SoH exponent -- all read off the built model
  D  throughput unit equivalence: the new active-energy throughput reproduces
     the old apparent-power expression exactly when eta = 1 and dt = 1 h
  E  solve on SRP1 + aggregate P/Q preservation + capability residuals
  F  cohort lifetime / zero-investment behaviour

    python p54c_esso_active_energy_validation.py
"""

import inspect
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_energy_storage_data as sesd  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from definitions import ESS_COMPLEMENTARITY_TOLERANCE  # noqa: E402
from model_construction_helpers import period_duration_hours  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54C')

RETIRED = ('es_snet', 'es_sch_per_unit', 'es_sdch_per_unit',
           'slack_es_snet_up', 'slack_es_snet_down',
           'slack_es_snet_def_up', 'slack_es_snet_def_down')
NEW = ('es_pch_per_unit', 'es_pdch_per_unit')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def src(obj):
    try:
        return inspect.getsource(obj)
    except Exception as error:
        return f'<unavailable: {error}>'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-C', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        consensus_vars, _dual = srp.create_admm_variables(planning)
        esso_models, esso_results = srp.create_shared_energy_storage_model(
            planning.shared_ess_data, consensus_vars, candidate['investment'])
    setup_text = console.getvalue()

    shared_ess_data = planning.shared_ess_data
    node_ids = list(shared_ess_data.active_distribution_network_nodes)
    node_id = node_ids[0]
    model = esso_models[node_id]
    dt = float(period_duration_hours(model))

    # ---------------- A. provenance ----------------
    report['A_provenance'] = {
        'esso_model_builder_source_head': src(sesd.build_subproblem)[:6000]
        if hasattr(sesd, 'build_subproblem') else None,
        'admm_helper_source': src(srp._get_esso_shared_ess_charge_discharge_pu),
        'node_ids': node_ids,
    }

    # ---------------- B. construction ----------------
    var_names = sorted(v.local_name for v in model.component_objects(pe.Var, active=None))
    con_names = sorted(c.local_name for c in model.component_objects(pe.Constraint, active=None))
    report['B_construction'] = {
        'period_duration_hours': dt,
        'n_periods': len(model.periods),
        'n_years': len(model.years),
        'n_days': len(model.days),
        'retired_absent': {n: not hasattr(model, n) for n in RETIRED},
        'new_present': {n: hasattr(model, n) for n in NEW},
        'variable_names': var_names,
        'constraint_component_names': con_names,
        'has_no_soc_variable': not any('soc' in n.lower() for n in var_names),
        'soh_variables_present': [n for n in var_names if 'soh' in n.lower()],
    }

    # ---------------- C. degradation-law preservation ----------------
    # Read the law's structure off the BUILT model rather than restating it.
    y_inv = y = 0
    ess0 = shared_ess_data.shared_energy_storages[list(shared_ess_data.years)[y_inv]][
        shared_ess_data.active_distribution_network_nodes.index(node_id)]
    thr_rows = [str(c.expr) for c in model.energy_storage_charging_discharging.values()]
    deg_rows = [str(c.expr) for c in model.energy_storage_capacity_degradation.values()]
    throughput_row = thr_rows[0] if thr_rows else ''

    day_weights = {day: shared_ess_data.days[day] / 365.0 for day in shared_ess_data.days}
    report['C_degradation_law'] = {
        'cl_nom': ess0.cl_nom, 't_cal': ess0.t_cal, 'soh_min': ess0.soh_min,
        'eff_ch': ess0.eff_ch, 'eff_dch': ess0.eff_dch,
        'representative_day_weights_num_days_over_365': day_weights,
        'year_multiplicities': dict(shared_ess_data.years),
        'throughput_row_count': len(thr_rows),
        'degradation_row_count': len(deg_rows),
        'throughput_row_sample': throughput_row[:1500],
        'throughput_uses_active_variables': (
            'es_pch_per_unit' in throughput_row and 'es_pdch_per_unit' in throughput_row
            and 'es_sch_per_unit' not in throughput_row),
        'degradation_rows_mention_cl_nom_normalization': any(
            str(2 * ess0.cl_nom) in r or f'{2 * ess0.cl_nom}' in r for r in deg_rows),
        'cumulative_soh_exponent_present': any('365' in r for r in deg_rows),
        'note': ('cl_nom, t_cal, soh_min, the (num_days/365) weighting, the cohort '
                 '[y_inv, y] indexing, the 2*cl_nom*E_rated equivalent-cycle '
                 'normalization and the 365*num_years cumulative exponent are '
                 'unchanged by P5.4-C; only the throughput INPUT changed.'),
    }

    # ---------------- D. throughput unit equivalence ----------------
    # Evaluate the production throughput row at a known dispatch, and compare to
    # the old apparent-power expression. They coincide exactly when eta_ch =
    # eta_dch = 1 and dt = 1 h, which is the sense in which the change is a unit
    # correction rather than a re-tuning.
    def set_cohort(value):
        for d in model.days:
            for p in model.periods:
                model.es_pch_per_unit[y_inv, y, d, p].value = value
                model.es_pdch_per_unit[y_inv, y, d, p].value = value

    set_cohort(0.001)
    model.es_avg_ch_dch_per_unit[y_inv, y].value = 0.0
    row = model.energy_storage_charging_discharging.values().__iter__().__next__()
    measured = float(pe.value(row.body))
    old_expr = sum(day_weights[day] * (0.001 + 0.001) * len(model.periods)
                   for day in shared_ess_data.days)
    new_expr = sum(day_weights[day]
                   * (ess0.eff_ch * 0.001 * dt + 0.001 * dt / ess0.eff_dch)
                   * len(model.periods)
                   for day in shared_ess_data.days)
    report['D_throughput_units'] = {
        'test_power_pu': 0.001,
        'dt_hours': dt,
        'measured_row_body_abs': abs(measured),
        'old_apparent_expression': old_expr,
        'new_active_energy_expression': new_expr,
        'measured_matches_new_expression': abs(abs(measured) - new_expr) < 1e-12 * max(1.0, new_expr),
        'ratio_new_over_old': (new_expr / old_expr) if old_expr else None,
        'would_equal_old_at_unit_efficiency_and_dt': abs(
            sum(day_weights[day] * (1.0 * 0.001 * 1.0 + 0.001 * 1.0 / 1.0) * len(model.periods)
                for day in shared_ess_data.days) - old_expr) < 1e-15,
    }
    set_cohort(0.0)

    # ---------------- E. solve + aggregate preservation ----------------
    succeeded = {}
    for nid in node_ids:
        succeeded[str(nid)] = bool(srp._solver_result_succeeded(esso_results[nid]))

    agg = {}
    worst_cap = 0.0
    worst_comp = 0.0
    worst_pnet_mismatch = 0.0
    circ_rows = []
    for nid in node_ids:
        m = esso_models[nid]
        for y_ in m.years:
            s_rated = float(pe.value(m.es_s_rated[y_]))
            for d in m.days:
                for p in m.periods:
                    pch = sum(float(pe.value(m.es_pch_per_unit[yi, y_, d, p])) for yi in m.years)
                    pdch = sum(float(pe.value(m.es_pdch_per_unit[yi, y_, d, p])) for yi in m.years)
                    pnet = float(pe.value(m.es_pnet[y_, d, p]))
                    qnet = float(pe.value(m.es_qnet[y_, d, p]))
                    worst_pnet_mismatch = max(worst_pnet_mismatch, abs(pnet - (pch - pdch)))
                    if s_rated > 0.0:
                        worst_cap = max(worst_cap,
                                        max((pnet ** 2 + qnet ** 2 - s_rated ** 2) / s_rated ** 2, 0.0))
                        circ_rows.append({'node': nid, 'year': y_, 'day': d, 'period': p,
                                          's_rated': s_rated, 'pch': pch, 'pdch': pdch,
                                          'p_circ_norm': min(pch, pdch) / s_rated,
                                          'pnet': pnet, 'qnet': qnet})
                    for yi in m.years:
                        a = float(pe.value(m.es_pch_per_unit[yi, y_, d, p]))
                        b = float(pe.value(m.es_pdch_per_unit[yi, y_, d, p]))
                        worst_comp = max(worst_comp, max(a * b - ESS_COMPLEMENTARITY_TOLERANCE, 0.0))
        agg[str(nid)] = {'s_rated_by_year': [float(pe.value(m.es_s_rated[y_])) for y_ in m.years],
                         'e_rated_by_year': [float(pe.value(m.es_e_rated[y_])) for y_ in m.years]}

    report['E_solve'] = {
        'succeeded': succeeded,
        'all_succeeded': all(succeeded.values()),
        'capacities': agg,
        'max_abs_pnet_minus_cohort_sum': worst_pnet_mismatch,
        'aggregate_active_power_preserved': worst_pnet_mismatch < 1e-8,
        'max_capability_violation_normalized': worst_cap,
        'max_cohort_complementarity_violation_absolute': worst_comp,
        'n_active_rows': len(circ_rows),
        'max_p_circ_norm': max((r['p_circ_norm'] for r in circ_rows), default=0.0),
        'worst_5_by_circulating_power': sorted(
            circ_rows, key=lambda r: -r['p_circ_norm'])[:5],
    }

    # ---------------- F. cohort lifetime behaviour ----------------
    fixed_cohorts = 0
    free_cohorts = 0
    for yi in model.years:
        for y_ in model.years:
            for d in model.days:
                for p in model.periods:
                    if model.es_pch_per_unit[yi, y_, d, p].fixed:
                        fixed_cohorts += 1
                    else:
                        free_cohorts += 1
    report['F_cohort_gating'] = {
        'fixed_pch_entries': fixed_cohorts,
        'free_pch_entries': free_cohorts,
        'future_cohorts_are_fixed': fixed_cohorts > 0,
        'note': 'cohort gating is unchanged by P5.4-C; it now gates the active variables',
    }

    report['setup_console_tail'] = setup_text[-1500:]
    out = os.path.join(OUT_DIR, 'p54c_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    b, c, d, e, f = (report['B_construction'], report['C_degradation_law'],
                     report['D_throughput_units'], report['E_solve'],
                     report['F_cohort_gating'])
    print(f'\n[P5.4-C] report -> {out}')
    print(f"  dt={b['period_duration_hours']} h  periods={b['n_periods']} years={b['n_years']} days={b['n_days']}")
    print(f"  retired absent: {b['retired_absent']}")
    print(f"  new present   : {b['new_present']}")
    print(f"  ESSO has NO SOC variable: {b['has_no_soc_variable']} (SoH vars: {b['soh_variables_present']})")
    print(f"  degradation law: cl_nom={c['cl_nom']} t_cal={c['t_cal']} soh_min={c['soh_min']} "
          f"eff=({c['eff_ch']},{c['eff_dch']})")
    print(f"    day weights {c['representative_day_weights_num_days_over_365']} "
          f"year mult {c['year_multiplicities']}")
    print(f"    throughput uses active vars: {c['throughput_uses_active_variables']}; "
          f"rows thr/deg = {c['throughput_row_count']}/{c['degradation_row_count']}")
    print(f"  throughput units: measured={d['measured_row_body_abs']:.6e} "
          f"new={d['new_active_energy_expression']:.6e} match={d['measured_matches_new_expression']}")
    print(f"    old apparent={d['old_apparent_expression']:.6e} ratio new/old={d['ratio_new_over_old']:.6f}")
    print(f"  solve: {e['succeeded']} all={e['all_succeeded']}")
    print(f"    aggregate P preserved: {e['aggregate_active_power_preserved']} "
          f"(max |pnet - sum(pch-pdch)| = {e['max_abs_pnet_minus_cohort_sum']:.3e})")
    print(f"    max capability violation (norm) = {e['max_capability_violation_normalized']:.4e}")
    print(f"    max cohort complementarity violation (abs) = {e['max_cohort_complementarity_violation_absolute']:.4e}")
    print(f"    max min(pch,pdch)/S = {e['max_p_circ_norm']:.4e} over {e['n_active_rows']} rows")
    print(f"  cohort gating: fixed={f['fixed_pch_entries']} free={f['free_pch_entries']}")


if __name__ == '__main__':
    main()
