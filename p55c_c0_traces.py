"""
Stage P5.5-C0 -- salvage closure (C0.1) and capacity-chain trace (C0.2).

C0.1  Prove V_salvage is maximised at SoH = 1 (E_avail = E_rated) and extract the
      exact affine coefficients gamma[node, cohort] of

          V_salvage_max(x) = sum_{node, cohort} gamma * E_investment.

C0.2  Trace investment S/E -> ESSO rated -> ESSO available -> TSO/DSO
      operational capacity, including the baseMVA conversion.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55c_c0_traces.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_energy_storage_data as sesd  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C0 traces', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C0] ABORTED\n{error}')
        sys.exit(1)

    sed = planning.shared_ess_data
    years = list(sed.years)
    terminal_idx = len(years) - 1
    params = sed.params.salvage_value

    console = io.StringIO()
    with redirect_stdout(console):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        consensus_vars, _ = srp.create_admm_variables(planning)
        esso_models, _ = srp.create_shared_energy_storage_model(
            sed, consensus_vars, candidate['investment'])

    report = {'stage': 'P5.5-C0', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    # ---------------- C0.1 ----------------
    terminal_discount = sesd._get_terminal_discount_factor(sed)
    salvage = {
        'formula': ('V_salvage = sum_{node,cohort} terminal_discount * '
                    'energy_recovery_fraction * expected_unit_cost(y_inv) * '
                    'remaining_life_fraction(y_inv) * E_rated[y_inv,terminal] * '
                    'residual_fraction'),
        'residual_fraction': ('recycling_floor + (1 - recycling_floor) * '
                              '(soh_cumul - soh_min)/(1 - soh_min)'),
        'params': {'enabled': params.enabled,
                   'energy_recovery_fraction': params.energy_recovery_fraction,
                   'recycling_floor_fraction': params.recycling_floor_fraction,
                   'cost_basis': params.cost_basis,
                   'health_basis': params.health_basis,
                   'calendar_life_basis': params.calendar_life_basis},
        'terminal_discount': terminal_discount,
        'cohorts': [],
    }

    total_gamma_check = 0.0
    for node_id in sed.active_distribution_network_nodes:
        model = esso_models[node_id]
        idx = sed.get_shared_energy_storage_idx(node_id)
        for y_inv, year_inv in enumerate(years):
            ess = sed.shared_energy_storages[year_inv][idx]
            age, remaining_life, remaining_life_fraction = sesd._get_remaining_calendar_life(
                sed, y_inv, ess)
            unit_cost = sesd._get_expected_energy_investment_cost(sed, year_inv)
            active = not model.es_e_rated_per_unit[y_inv, terminal_idx].fixed
            eligible = active and remaining_life > 1e-4
            gamma = (terminal_discount * params.energy_recovery_fraction
                     * unit_cost * remaining_life_fraction) if (
                params.enabled and eligible) else 0.0
            # residual_fraction at SoH = 1 and at SoH = soh_min
            rf_at_1 = params.recycling_floor_fraction + (
                1.0 - params.recycling_floor_fraction) * (
                (1.0 - ess.soh_min) / (1.0 - ess.soh_min))
            rf_at_min = params.recycling_floor_fraction
            salvage['cohorts'].append({
                'node_id': node_id, 'investment_year': year_inv,
                'cohort_index': y_inv,
                'soh_min': ess.soh_min, 't_cal': ess.t_cal,
                'age_at_terminal': age, 'remaining_life': remaining_life,
                'remaining_life_fraction': remaining_life_fraction,
                'expected_unit_energy_cost': unit_cost,
                'active_in_terminal_block': active,
                'salvage_eligible': eligible,
                'residual_fraction_at_soh_1': rf_at_1,
                'residual_fraction_at_soh_min': rf_at_min,
                'gamma': gamma,
            })
            total_gamma_check += gamma
    salvage['sum_gamma'] = total_gamma_check
    salvage['max_attained_at'] = 'soh_cumul = 1  <=>  E_available = E_rated'
    salvage['monotone_increasing_in_soh'] = (
        params.recycling_floor_fraction <= 1.0 and params.energy_recovery_fraction >= 0.0)
    salvage['direct_S_coefficient'] = 0.0
    salvage['direct_S_coefficient_note'] = (
        "_get_salvage_value_sensitivities sets sensitivities['s'] = 0.00 for every "
        "node and year and never overwrites it; the salvage expression contains no "
        "S term")
    report['C0_1_salvage'] = salvage

    # ---------------- C0.2 ----------------
    chain = {'steps': [
        "1. es_s_investment[y_inv], es_e_investment[y_inv]  (ESSO, physical MVA / MVAh)",
        "2. rated_s/e_capacity_unit: es_[se]_rated_per_unit[y_inv,y] == es_[se]_investment[y_inv]"
        "   for y in [y_inv, y_inv + tcal_norm)",
        "3. rated_s/e_capacity: es_[se]_rated[y] == sum_{y_inv} es_[se]_rated_per_unit[y_inv,y]",
        "4a. available_s_capacity_unit: es_s_available_per_unit == es_s_rated_per_unit  (EXACT)",
        "4b. available_e_capacity_unit: es_e_available_per_unit == es_e_rated_per_unit * soh_cumul",
        "5. get_available_capacities: s_available = sum_{y_inv} es_s_available_per_unit[y_inv,y]",
        "6. network: shared_es_s_rated_fixed = s_available / baseMVA   (p.u.)",
        "   network: shared_es_e_rated_fixed = e_available / baseMVA   (p.u.)",
    ], 'per_node_year': {}}

    available = sed.get_updated_capacities(esso_models)
    for node_id in sed.active_distribution_network_nodes:
        m = esso_models[node_id]
        base = planning.distribution_networks[node_id].network[
            years[0]][list(planning.days)[0]].baseMVA
        chain['per_node_year'][str(node_id)] = {}
        for y, year in enumerate(years):
            s_rated = float(pe.value(m.es_s_rated[y]))
            e_rated = float(pe.value(m.es_e_rated[y]))
            s_av = available[node_id][year]['s_available']
            e_av = available[node_id][year]['e_available']
            chain['per_node_year'][str(node_id)][str(year)] = {
                'investment_s': float(pe.value(m.es_s_investment[y])),
                'investment_e': float(pe.value(m.es_e_investment[y])),
                'esso_rated_s': s_rated, 'esso_rated_e': e_rated,
                'esso_available_s': s_av, 'esso_available_e': e_av,
                's_available_equals_rated': abs(s_av - s_rated) < 1e-9,
                'e_available_over_rated': (e_av / e_rated) if e_rated else None,
                'baseMVA': base,
                'network_s_rated_pu': s_av / base,
                'network_e_rated_pu': e_av / base,
            }
    report['C0_2_capacity_chain'] = chain

    out = os.path.join(OUT_DIR, 'p55c_c0_traces.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('\n[C0.1] salvage parameters:', salvage['params'])
    print(f"    terminal_discount = {terminal_discount:.9f}")
    print(f"    residual_fraction at SoH=1   : {salvage['cohorts'][0]['residual_fraction_at_soh_1']}")
    print(f"    residual_fraction at SoH=min : {salvage['cohorts'][0]['residual_fraction_at_soh_min']}")
    print(f"    => maximum at {salvage['max_attained_at']}")
    print(f"    direct S coefficient: {salvage['direct_S_coefficient']}")
    print(f"\n{'node':>5} {'cohort':>7} {'elig':>5} {'rem_life_frac':>14} "
          f"{'unit_cost':>14} {'gamma':>16}")
    for c in salvage['cohorts']:
        print(f"{c['node_id']:>5} {c['investment_year']:>7} {str(c['salvage_eligible']):>5} "
              f"{c['remaining_life_fraction']:>14.6f} {c['expected_unit_energy_cost']:>14.4f} "
              f"{c['gamma']:>16.6f}")
    print(f"    sum of gamma = {total_gamma_check:.6f}")

    print('\n[C0.2] capacity chain')
    for step in chain['steps']:
        print('   ', step)
    print(f"\n{'node':>5} {'year':>6} {'inv_E':>10} {'rated_E':>10} {'avail_E':>10} "
          f"{'avail/rated':>12} {'S_av==S_rated':>14} {'net_E_pu':>12}")
    for node, per in chain['per_node_year'].items():
        for year, v in per.items():
            print(f"{node:>5} {year:>6} {v['investment_e']:>10.5f} {v['esso_rated_e']:>10.5f} "
                  f"{v['esso_available_e']:>10.5f} "
                  f"{(v['e_available_over_rated'] if v['e_available_over_rated'] else 0):>12.6f} "
                  f"{str(v['s_available_equals_rated']):>14} {v['network_e_rated_pu']:>12.3e}")
    print(f'\n[C0] report -> {out}')


if __name__ == '__main__':
    main()
