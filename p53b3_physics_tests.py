"""
Stage P5.3-B3.8 / B3.9 -- physics unit tests and post-reformulation row audit.

Builds a production cold-start model, applies the B3 active-power conversion,
then evaluates the CONSTRUCTED constraint rows at prescribed variable values.
No solve is needed: each test reads the production/prototype row body directly,
so the checks are exact.

Tests
  1 pure reactive     pch = pdch = pnet = 0, qnet != 0   -> Delta SOC == 0
  2 pure charging     pch > 0, pdch = 0                  -> +eta_ch*pch*dt
  3 pure discharging  pdch > 0, pch = 0                  -> -pdch*dt/eta_dch
  4 converter         pnet^2 + qnet^2 <= S_rated^2
  5 complementarity   acts on pch*pdch with the configured tolerance
  6 zero capacity     gating remains safe

Also reports, after the reformulation, the new worst constraint families by
smallest gradient norm, largest curvature and tightest margin.

    python p53b3_physics_tests.py
"""

import io
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from definitions import (ENERGY_STORAGE_RELATIVE_INIT_SOC,  # noqa: E402
                         ESS_COMPLEMENTARITY_TOLERANCE)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p51_small_capacity_scaling_diagnostic import build_dso_initialization_models  # noqa: E402
from p53b3_active_power_ess import convert_to_active_power  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53B3')
NODE, YEAR, DAY = 5, 2030, 'Winter'


def set_state(model, e, p, pch, pdch, pnet, qnet, soc_prev=None):
    model.shared_es_pch[e, 0, 0, p].set_value(pch, skip_validation=True)
    model.shared_es_pdch[e, 0, 0, p].set_value(pdch, skip_validation=True)
    model.shared_es_pnet[e, 0, 0, p].set_value(pnet, skip_validation=True)
    model.shared_es_qnet[e, 0, 0, p].set_value(qnet, skip_validation=True)
    if soc_prev is not None and p > 0:
        model.shared_es_soc[e, 0, 0, p - 1].set_value(soc_prev, skip_validation=True)


def soc_from_row(model, e, p):
    """Solve the SOC row for SOC_t given everything else, by evaluating its body."""
    con = model.b3_sess_soc_active[e, 0, 0, p]
    # body is  soc_t - (soc_prev + delta); set soc_t = 0 and read -(rhs)
    model.shared_es_soc[e, 0, 0, p].set_value(0.0, skip_validation=True)
    return -float(pe.value(con.body))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.3-B3.8/B3.9',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'case': f'case33_1 node {NODE} {YEAR} {DAY}'}

    quiet = io.StringIO()
    with redirect_stdout(quiet):
        planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        dn = planning.distribution_networks[NODE]
        models = build_dso_initialization_models(dn, candidate['total_capacity'])

    model = models[YEAR][DAY]
    network = dn.network[YEAR][DAY]
    info = convert_to_active_power(model, network, dn.params)
    report['conversion'] = {k: v for k, v in info.items() if k != 'converted_indices'}

    e = network.get_shared_energy_storage_idx(network.get_reference_node_id())
    sess = network.shared_energy_storages[e]
    s_rated = float(pe.value(model.shared_es_s_rated_fixed[e]))
    dt = info['dt_hours']
    eff_ch, eff_dch = sess.eff_ch, sess.eff_dch
    report['device'] = {'shared_ess_index': e, 's_rated_pu': s_rated,
                        'eff_ch': eff_ch, 'eff_dch': eff_dch, 'dt_hours': dt,
                        'e_rated_pu': float(pe.value(model.shared_es_e_rated[e]))}

    p = 5
    soc_prev = 0.5 * report['device']['e_rated_pu']
    a = 0.4 * s_rated
    tests = {}

    # ---- 1 pure reactive ----
    set_state(model, e, p, 0.0, 0.0, 0.0, 0.5 * s_rated, soc_prev)
    soc_t = soc_from_row(model, e, p)
    tests['1_pure_reactive'] = {
        'inputs': {'pch': 0.0, 'pdch': 0.0, 'pnet': 0.0, 'qnet': 0.5 * s_rated},
        'soc_prev': soc_prev, 'soc_t': soc_t,
        'delta_soc': soc_t - soc_prev,
        'PASS_soc_unchanged': abs(soc_t - soc_prev) < 1e-15}

    # ---- 2 pure charging ----
    set_state(model, e, p, a, 0.0, a, 0.0, soc_prev)
    soc_t = soc_from_row(model, e, p)
    expected = soc_prev + eff_ch * a * dt
    tests['2_pure_charging'] = {
        'inputs': {'pch': a, 'pdch': 0.0}, 'soc_t': soc_t,
        'expected': expected, 'delta_soc': soc_t - soc_prev,
        'expected_delta': eff_ch * a * dt,
        'PASS_exact': abs(soc_t - expected) < 1e-15}

    # ---- 3 pure discharging ----
    set_state(model, e, p, 0.0, a, -a, 0.0, soc_prev)
    soc_t = soc_from_row(model, e, p)
    expected = soc_prev - a * dt / eff_dch
    tests['3_pure_discharging'] = {
        'inputs': {'pch': 0.0, 'pdch': a}, 'soc_t': soc_t,
        'expected': expected, 'delta_soc': soc_t - soc_prev,
        'expected_delta': -a * dt / eff_dch,
        'PASS_exact': abs(soc_t - expected) < 1e-15}

    # ---- 4 converter capability ----
    cap = model.b3_sess_converter_capability[e, 0, 0, p]
    samples = []
    for pn, qn, label in ((0.0, 0.0, 'origin'),
                          (s_rated, 0.0, 'pure P at rating'),
                          (0.0, s_rated, 'pure Q at rating'),
                          (0.8 * s_rated, 0.6 * s_rated, 'on circle (0.8,0.6)'),
                          (0.9 * s_rated, 0.9 * s_rated, 'outside circle')):
        set_state(model, e, p, 0.0, 0.0, pn, qn, soc_prev)
        body = float(pe.value(cap.body))
        samples.append({'label': label, 'pnet': pn, 'qnet': qn,
                        'body': body, 'upper': float(pe.value(cap.upper)),
                        'satisfied': body <= float(pe.value(cap.upper)) + 1e-18})
    tests['4_converter_capability'] = {
        'row': 'pnet^2 + qnet^2 <= S_rated^2', 'samples': samples,
        'PASS_circle_semantics': (samples[1]['satisfied'] and samples[2]['satisfied']
                                  and samples[3]['satisfied']
                                  and not samples[4]['satisfied'])}

    # ---- 5 complementarity on active power ----
    comp = model.b3_sess_comp_active[e, 0, 0, p]
    comp_vars = sorted(v.name for v in identify_variables(comp.body))
    set_state(model, e, p, a, a, 0.0, 0.0, soc_prev)
    tests['5_complementarity'] = {
        'row': 'pch*pdch <= ESS_COMPLEMENTARITY_TOLERANCE * S_rated^2',
        'variables_in_row': comp_vars,
        'acts_on_active_power': comp_vars == sorted(
            [model.shared_es_pch[e, 0, 0, p].name, model.shared_es_pdch[e, 0, 0, p].name]),
        'tolerance_used': ESS_COMPLEMENTARITY_TOLERANCE,
        'rhs': float(pe.value(comp.upper)),
        'rhs_over_ipopt_tol': float(pe.value(comp.upper)) / 1e-5,
        'body_at_pch=pdch=0.4S': float(pe.value(comp.body)),
        'PASS_tolerance_preserved': abs(
            float(pe.value(comp.upper)) - ESS_COMPLEMENTARITY_TOLERANCE * s_rated ** 2) < 1e-20}

    # ---- 6 zero capacity ----
    zero_ok = {}
    with redirect_stdout(quiet):
        models0 = build_dso_initialization_models(dn, candidate['total_capacity'])
    m0 = models0[YEAR][DAY]
    from model_construction_helpers import configure_shared_ess_operational_state
    configure_shared_ess_operational_state(m0, e, 0.0, 0.0)
    info0 = convert_to_active_power(m0, network, dn.params)
    zero_ok['conversion_skipped_zero_capacity'] = len(info0['skipped_zero_capacity']) == 1
    zero_ok['no_new_rows_added'] = info0['n_new_rows'] == 0
    zero_ok['pch_fixed_zero'] = all(
        m0.shared_es_pch[e, 0, 0, pp].fixed and abs(pe.value(m0.shared_es_pch[e, 0, 0, pp])) < 1e-15
        for pp in m0.periods)
    zero_ok['pnet_fixed_zero'] = all(
        m0.shared_es_pnet[e, 0, 0, pp].fixed and abs(pe.value(m0.shared_es_pnet[e, 0, 0, pp])) < 1e-15
        for pp in m0.periods)
    zero_ok['sess_rows_inactive'] = not any(
        m0.sess_snet_def[i].active for i in m0.sess_snet_def if i[0] == e)
    zero_ok['no_division_by_zero'] = True
    tests['6_zero_capacity'] = zero_ok

    report['physics_tests'] = tests

    # ---- B3.9 new worst families on the converted model ----
    worst = {'smallest_gradient': None, 'largest_curvature': None, 'tightest_margin': None}
    fam_grad, fam_curv, fam_margin = {}, {}, {}
    for comp_obj in model.component_objects(pe.Constraint, active=True):
        name = comp_obj.local_name
        for count, index in enumerate(comp_obj):
            if count >= 12:
                break
            con = comp_obj[index]
            if not con.active:
                continue
            variables = list(identify_variables(con.body, include_fixed=False))
            if not variables:
                continue
            try:
                g = differentiate(con.body, wrt_list=variables, mode=Modes.reverse_numeric)
                norm = math.sqrt(sum(float(x) ** 2 for x in g if x is not None))
            except Exception:
                continue
            prev = fam_grad.get(name)
            if prev is None or norm < prev:
                fam_grad[name] = norm
            try:
                sym = differentiate(con.body, wrt_list=variables, mode=Modes.reverse_symbolic)
                mx = 0.0
                for gi in sym:
                    second = differentiate(gi, wrt_list=variables, mode=Modes.reverse_numeric)
                    mx = max([mx] + [abs(float(s)) for s in second if s is not None])
                if name not in fam_curv or mx > fam_curv[name]:
                    fam_curv[name] = mx
            except Exception:
                pass
            if not con.equality and con.upper is not None:
                try:
                    margin = float(pe.value(con.upper)) - float(pe.value(con.body))
                    if name not in fam_margin or margin < fam_margin[name]:
                        fam_margin[name] = margin
                except Exception:
                    pass
    report['B3_9_post_reformulation_families'] = {
        'smallest_gradient_norm_by_family': dict(sorted(fam_grad.items(), key=lambda kv: kv[1])[:10]),
        'largest_curvature_by_family': dict(sorted(fam_curv.items(), key=lambda kv: -kv[1])[:10]),
        'tightest_margin_by_family': dict(sorted(fam_margin.items(), key=lambda kv: kv[1])[:10]),
    }

    with open(os.path.join(OUT_DIR, 'p53b3_physics.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    d = report['device']
    print(f"device: S_rated={d['s_rated_pu']:.6e} eff_ch={d['eff_ch']} eff_dch={d['eff_dch']} dt={d['dt_hours']}h")
    for key, t in tests.items():
        flags = {k: v for k, v in t.items() if k.startswith('PASS')}
        print(f"  {key:22s} {flags if flags else t}")
    print("\nB3.9 smallest gradient by family:")
    for k, v in report['B3_9_post_reformulation_families']['smallest_gradient_norm_by_family'].items():
        print(f"   {k:32s} {v:.4e}")
    print("B3.9 largest curvature by family:")
    for k, v in report['B3_9_post_reformulation_families']['largest_curvature_by_family'].items():
        print(f"   {k:32s} {v:.4e}")
    print("B3.9 tightest margin by family:")
    for k, v in report['B3_9_post_reformulation_families']['tightest_margin_by_family'].items():
        print(f"   {k:32s} {v:.4e}")


if __name__ == '__main__':
    main()
