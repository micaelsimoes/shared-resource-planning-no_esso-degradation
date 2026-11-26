"""
Stage P5.4-D2-P -- validation of the sensitivity-clean shared-S productionization.

The four capacity-dependent numerical bounds on `shared_es_pch`,
`shared_es_pdch`, `shared_es_pnet` and `shared_es_qnet` are removed at POSITIVE
capacity, so installed power now enters the local NLP only through symbolic rows
and the `shared_es_s_rated` variable. The zero-capacity collapse to [0, 0] and
the explicit fixing at 0 are kept.

Sections
  D2-P.1  exact feasible-set redundancy, tested numerically at solved points
  D2-P.2  zero/positive lifecycle and reused-model identity
  D2-P.3  sensitivity structure across the eight original D2 cases

D2-P.4 (operational regression) is run separately with the existing production
gates: p54h1_gate.py / p54e_production_validation.py / p54f_admm_net_pq.py.

    python p54d2p_validation.py
"""

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

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p54d2_sensitivity_clean_ab import DEFAULT_CASES, solve_variant  # noqa: E402
from p54d2_sensitivity_root_cause import (ESS_VARIABLES, bound_sensitivity_map,  # noqa: E402
                                          build_case, direct_parameter_dependence,
                                          ess_entries, objective_of,
                                          sensitivity_decomposition)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2P')

CLEANED = ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet', 'shared_es_qnet')
TRACKED = CLEANED + ('shared_es_pch_hat', 'shared_es_pdch_hat', 'shared_es_soc',
                     'sess_pnet_def', 'sess_pch_hat_link', 'sess_pdch_hat_link',
                     'sess_converter_capability', 'sess_active_sum_limit',
                     'sess_comp', 'sess_soc_def', 'shared_es_s_rated',
                     'shared_es_s_rated_fixed',
                     'shared_energy_storage_s_sensitivities',
                     'shared_energy_storage_e_sensitivities')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def redundancy_at_solution(model, e):
    """D2-P.1: the removed bounds must still hold at the solved point, and must
    hold because of the retained symbolic rows."""
    s = float(pe.value(model.shared_es_s_rated_fixed[e]))
    worst = {'pch_over_S': 0.0, 'pdch_over_S': 0.0,
             'abs_pnet_over_S': 0.0, 'abs_qnet_over_S': 0.0,
             'pch_plus_pdch_over_S': 0.0, 'capability_over_S2': 0.0,
             'pch_hat': 0.0, 'pdch_hat': 0.0}
    pch = ess_entries(model, 'shared_es_pch', e)
    pdch = ess_entries(model, 'shared_es_pdch', e)
    pnet = ess_entries(model, 'shared_es_pnet', e)
    qnet = ess_entries(model, 'shared_es_qnet', e)
    hc = ess_entries(model, 'shared_es_pch_hat', e)
    hd = ess_entries(model, 'shared_es_pdch_hat', e)
    n_neg_pch = n_neg_pdch = 0
    for i in range(len(pch)):
        a, b = float(pe.value(pch[i])), float(pe.value(pdch[i]))
        p_, q_ = float(pe.value(pnet[i])), float(pe.value(qnet[i]))
        worst['pch_over_S'] = max(worst['pch_over_S'], a / s)
        worst['pdch_over_S'] = max(worst['pdch_over_S'], b / s)
        worst['abs_pnet_over_S'] = max(worst['abs_pnet_over_S'], abs(p_) / s)
        worst['abs_qnet_over_S'] = max(worst['abs_qnet_over_S'], abs(q_) / s)
        worst['pch_plus_pdch_over_S'] = max(worst['pch_plus_pdch_over_S'], (a + b) / s)
        worst['capability_over_S2'] = max(worst['capability_over_S2'],
                                          (p_ ** 2 + q_ ** 2) / s ** 2)
        worst['pch_hat'] = max(worst['pch_hat'], float(pe.value(hc[i])))
        worst['pdch_hat'] = max(worst['pdch_hat'], float(pe.value(hd[i])))
        if a < -1e-12:
            n_neg_pch += 1
        if b < -1e-12:
            n_neg_pdch += 1
    tol = 1e-8
    return {
        's_pu': s, 'n_periods': len(pch), 'worst': worst,
        'n_negative_pch': n_neg_pch, 'n_negative_pdch': n_neg_pdch,
        'pch_le_S_holds': worst['pch_over_S'] <= 1.0 + tol,
        'pdch_le_S_holds': worst['pdch_over_S'] <= 1.0 + tol,
        'abs_pnet_le_S_holds': worst['abs_pnet_over_S'] <= 1.0 + tol,
        'abs_qnet_le_S_holds': worst['abs_qnet_over_S'] <= 1.0 + tol,
        'active_sum_row_holds': worst['pch_plus_pdch_over_S'] <= 1.0 + tol,
        'capability_row_holds': worst['capability_over_S2'] <= 1.0 + tol,
        'hat_bounds_hold': worst['pch_hat'] <= 1.0 + tol and worst['pdch_hat'] <= 1.0 + tol,
        'all_implied_bounds_hold': all([
            worst['pch_over_S'] <= 1.0 + tol, worst['pdch_over_S'] <= 1.0 + tol,
            worst['abs_pnet_over_S'] <= 1.0 + tol, worst['abs_qnet_over_S'] <= 1.0 + tol]),
    }


def lifecycle(network, params, e, s_pu, e_pu):
    """D2-P.2 on ONE reused model."""
    console = io.StringIO()
    with redirect_stdout(console):
        model = network.build_model(params)

    def fingerprint():
        return {n: id(getattr(model, n)) for n in TRACKED if hasattr(model, n)}

    def state(label):
        out = {'label': label,
               's_rated_fixed': float(pe.value(model.shared_es_s_rated_fixed[e]))}
        for name in CLEANED + ('shared_es_pch_hat', 'shared_es_pdch_hat'):
            entry = ess_entries(model, name, e)[0]
            out[name] = {'bounds': list(entry.bounds), 'fixed': bool(entry.fixed),
                         'value': float(pe.value(entry))}
        out['n_unbounded_and_unfixed'] = sum(
            1 for name in CLEANED
            for x in ess_entries(model, name, e)
            if not x.fixed and x.lb is None and x.ub is None)
        out['rows_active'] = {
            n: bool(ess_entries(model, n, e)[0].active)
            for n in ('sess_pch_hat_link', 'sess_pdch_hat_link',
                      'sess_converter_capability', 'sess_active_sum_limit', 'sess_comp')}
        out['suffixes_present'] = {
            n: hasattr(model, n) for n in ('dual', 'ipopt_zL_out', 'ipopt_zU_out',
                                           'ipopt_zL_in', 'ipopt_zU_in')}
        return out

    prints = [fingerprint()]
    steps = []
    for label, sv, ev in (('zero', 0.0, 0.0),
                          ('positive', s_pu, e_pu),
                          ('different_positive', 2.5 * s_pu, 2.5 * e_pu),
                          ('back_to_zero', 0.0, 0.0),
                          ('positive_again', s_pu, e_pu)):
        console = io.StringIO()
        with redirect_stdout(console):
            model.shared_es_s_rated_fixed[e].set_value(sv)
            model.shared_es_e_rated_fixed[e].set_value(ev)
            mch.configure_shared_ess_operational_state(model, e, sv, ev)
        steps.append(state(label))
        prints.append(fingerprint())

    zero_states = [s for s in steps if s['label'] in ('zero', 'back_to_zero')]
    pos_states = [s for s in steps if s['label'] not in ('zero', 'back_to_zero')]
    return {
        'transitions': steps,
        'component_ids_constant': all(f == prints[0] for f in prints),
        'components_tracked': sorted(prints[0].keys()),
        'zero_capacity_all_fixed_at_zero': all(
            all(s[n]['fixed'] and s[n]['value'] == 0.0 and s[n]['bounds'] == [0.0, 0.0]
                for n in CLEANED) for s in zero_states),
        'zero_capacity_rows_deactivated': all(
            not any(s['rows_active'].values()) for s in zero_states),
        'positive_capacity_free': all(
            all(not s[n]['fixed'] for n in CLEANED) for s in pos_states),
        'positive_capacity_rows_active': all(
            all(s['rows_active'].values()) for s in pos_states),
        'positive_capacity_no_S_dependent_bounds': all(
            s['shared_es_pch']['bounds'] == [0.0, None]
            and s['shared_es_pdch']['bounds'] == [0.0, None]
            and s['shared_es_pnet']['bounds'] == [None, None]
            and s['shared_es_qnet']['bounds'] == [None, None] for s in pos_states),
        'hat_bounds_unchanged': all(
            s['shared_es_pch_hat']['bounds'] == [0.0, 1.0]
            and s['shared_es_pdch_hat']['bounds'] == [0.0, 1.0] for s in steps),
        'warm_start_suffixes_intact': all(
            all(s['suffixes_present'].values()) for s in steps),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D2-P', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'lifecycle_bound_table': [
                  list(t) for t in mch._SHARED_ESS_ZERO_GATED_BOUND_VARIABLES],
              'old_table_removed': not hasattr(
                  mch, '_SHARED_ESS_RATED_BOUNDED_VARIABLES'),
              'redundancy_argument': {
                  'pch_le_S': 'pch, pdch >= 0 and pch + pdch <= S; also pch = S*pch_hat, pch_hat <= 1',
                  'pdch_le_S': 'symmetric',
                  'abs_pnet_le_S': 'pnet^2 + qnet^2 <= S^2 for S > 0',
                  'abs_qnet_le_S': 'symmetric'}}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    cases = {}
    for spec in DEFAULT_CASES.split(','):
        agent, ident, year, day = spec.split(':')
        case = build_case(planning, candidate, agent, ident, year, day)
        tag = case['tag']
        print(f'\n[D2-P] {tag}', flush=True)

        m, _r, ok, _i, _l = solve_variant(case['network'], case['params'], case['e'],
                                          case['s_pu'], case['e_pu'], clean=False)
        if not ok:
            cases[tag] = {'solved': False}
            print('    solve FAILED')
            continue

        deriv = bound_sensitivity_map(case['network'], case['params'], case['e'],
                                      case['s_pu'], case['e_pu'])
        decS = sensitivity_decomposition(m, case['e'], 'S', deriv)
        decE = sensitivity_decomposition(m, case['e'], 'E')
        red = redundancy_at_solution(m, case['e'])
        direct = direct_parameter_dependence(m, case['e'])

        entry = {
            'solved': True, 'objective': objective_of(m),
            's_pu': case['s_pu'], 'e_pu': case['e_pu'],
            'D2P_1_redundancy': red,
            'D2P_3_bound_S_dependence': deriv,
            'D2P_3_any_bound_depends_on_S': any(
                v['bound_depends_on_S'] for v in deriv.values()),
            'D2P_3_S_decomposition': {k: decS[k] for k in (
                'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative')},
            'D2P_3_E_decomposition': {k: decE[k] for k in (
                'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative')},
            'D2P_3_bound_contribution_is_zero': abs(decS['bound_contribution']) < 1e-14,
            'D2P_3_direct_parameter_dependence': direct,
            'D2P_3_fixing_dual_is_complete': (
                abs(decS['bound_contribution']) < 1e-14
                and direct['n_active_rows_using_fixed_params'] == 0),
        }
        cases[tag] = entry
        print(f"    redundancy: pch/S<=1 {red['pch_le_S_holds']} pdch/S<=1 {red['pdch_le_S_holds']} "
              f"|pnet|/S<=1 {red['abs_pnet_le_S_holds']} |qnet|/S<=1 {red['abs_qnet_le_S_holds']} "
              f"(worst {red['worst']['pch_over_S']:.4f}/{red['worst']['pdch_over_S']:.4f}/"
              f"{red['worst']['abs_pnet_over_S']:.4f}/{red['worst']['abs_qnet_over_S']:.4f})")
        print(f"    any bound depends on S: {entry['D2P_3_any_bound_depends_on_S']}; "
              f"bound contribution = {decS['bound_contribution']:+.3e}; "
              f"direct rows using S_fixed = {direct['n_active_rows_using_fixed_params']}")
        print(f"    S fixing dual = {decS['fixing_row_dual']:+.6e} -> complete: "
              f"{entry['D2P_3_fixing_dual_is_complete']};  E bound = {decE['bound_contribution']:+.3e}")

    report['cases'] = cases

    # D2-P.2 on one reused model
    case = build_case(planning, candidate, 'dso', '9', '2030', 'Winter')
    print('\n[D2-P.2] lifecycle on a reused model ...', flush=True)
    report['D2P_2_lifecycle'] = lifecycle(case['network'], case['params'], case['e'],
                                          case['s_pu'], case['e_pu'])
    lc = report['D2P_2_lifecycle']
    for k in ('component_ids_constant', 'zero_capacity_all_fixed_at_zero',
              'zero_capacity_rows_deactivated', 'positive_capacity_free',
              'positive_capacity_rows_active', 'positive_capacity_no_S_dependent_bounds',
              'hat_bounds_unchanged', 'warm_start_suffixes_intact'):
        print(f'    {k}: {lc[k]}')
    for t in lc['transitions']:
        print(f"      {t['label']:20s} S={t['s_rated_fixed']:.4e} "
              f"pch={t['shared_es_pch']['bounds']} pnet={t['shared_es_pnet']['bounds']} "
              f"fixed={t['shared_es_pch']['fixed']} "
              f"unbounded_unfixed={t['n_unbounded_and_unfixed']}")

    ok_cases = [v for v in cases.values() if v.get('solved')]
    report['summary'] = {
        'n_cases': len(cases), 'n_solved': len(ok_cases),
        'all_redundant_bounds_hold': all(
            v['D2P_1_redundancy']['all_implied_bounds_hold'] for v in ok_cases),
        'no_bound_depends_on_S_anywhere': not any(
            v['D2P_3_any_bound_depends_on_S'] for v in ok_cases),
        'all_bound_contributions_zero': all(
            v['D2P_3_bound_contribution_is_zero'] for v in ok_cases),
        'all_fixing_duals_complete': all(
            v['D2P_3_fixing_dual_is_complete'] for v in ok_cases),
        'all_E_bound_contributions_zero': all(
            v['D2P_3_E_decomposition']['bound_contribution'] == 0.0 for v in ok_cases),
        'lifecycle_all_pass': all(lc[k] for k in (
            'component_ids_constant', 'zero_capacity_all_fixed_at_zero',
            'zero_capacity_rows_deactivated', 'positive_capacity_free',
            'positive_capacity_rows_active', 'positive_capacity_no_S_dependent_bounds',
            'hat_bounds_unchanged', 'warm_start_suffixes_intact')),
    }
    out = os.path.join(OUT_DIR, 'p54d2p_report.json')
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f"\n[D2-P] summary: {json.dumps(report['summary'], indent=1)}")
    print(f'[D2-P] report -> {out}')


if __name__ == '__main__':
    main()
