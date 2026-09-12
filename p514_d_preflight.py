"""
Track D pre-flight — five checks, ZERO SOLVES (enforced). Nothing is run afterwards.

The plan under test is specified explicitly and does NOT come from
_build_positive_bootstrap_candidate:
    1.00 MVA / 4.00 MWh per active distribution node, invested in 2025 only,
    zero in 2030 and 2035, single ageing cohort.
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514D')
S_INV, E_INV, INVEST_YEAR = 1.00, 4.00, 2025
K_C3 = 11541.560327111707


def main():
    guard = SolveProfileGuard(permitted=(), label='Track D pre-flight (zero solves)').install()
    try:
        import p56a_oracle as O
        with redirect_stdout(io.StringIO()):
            planning = O.fresh_planning('p514d_preflight')
        sed = planning.shared_ess_data
        years = list(sed.years)
        nodes = list(sed.active_distribution_network_nodes)
        params = sed.params

        # --- check 1: discounted investment cost, by the production formula ---
        per_node = {}
        total = 0.0
        for node_id in nodes:
            cost = 0.0
            discount = 1.0 / ((1.0 + sed.discount_factor) ** (int(INVEST_YEAR) - int(years[0])))
            for scenario, probability in enumerate(sed.prob_market_scenarios):
                c_s = sed.cost_investment['power'][scenario][INVEST_YEAR]
                c_e = sed.cost_investment['energy'][scenario][INVEST_YEAR]
                cost += discount * probability * (c_s * S_INV + c_e * E_INV)
            per_node[node_id] = cost
            total += cost
        unit_costs = {scenario: {'power': sed.cost_investment['power'][scenario][INVEST_YEAR],
                                 'energy': sed.cost_investment['energy'][scenario][INVEST_YEAR]}
                      for scenario in range(len(sed.prob_market_scenarios))}

        # --- check 2: capacity and ratio bounds ---
        ratio = E_INV / S_INV
        bounds = {
            'max_capacity': params.max_capacity,
            'plan_energy_per_node': E_INV,
            'comparison_operator': "candidate check uses `total_capacity['e'] > max_capacity + 1e-8` "
                                   "(shared_resources_planning.py:1631); the master adds "
                                   "`es_e_rated <= max_capacity` (shared_energy_storage_data.py:282)",
            'margin_absolute': params.max_capacity - E_INV,
            'margin_fraction_of_cap': E_INV / params.max_capacity,
            'ratio': ratio,
            'min_energy_to_power_ratio': params.min_energy_to_power_ratio,
            'max_energy_to_power_ratio': params.max_energy_to_power_ratio,
            'ratio_within_bounds': params.min_energy_to_power_ratio <= ratio <= params.max_energy_to_power_ratio,
            'bootstrap_pinned_ratio': planning.params.benders.positive_bootstrap.energy_to_power_ratio,
        }

        # --- check 3: SoH trajectory under C3, single cohort invested 2025 ---
        ess0 = sed.shared_energy_storages[years[0]][0]
        k = getattr(ess0, 'cl_eff', ess0.cl_nom)
        soh_min = ess0.soh_min
        blocks = [(year, sed.years[year]) for year in years]
        trajectory = {}
        for efc_per_day in (0.5, 1.0, 1.5):
            delta = efc_per_day / k
            soh, rows = 1.0, []
            for year, n_years in blocks:
                soh = soh * (1.0 - delta) ** (365.0 * n_years)
                rows.append({'year': year, 'years_in_block': n_years, 'soh_cumulative': soh})
            trajectory[f'{efc_per_day} EFC/day'] = {
                'delta_daily': delta, 'blocks': rows, 'terminal_soh': soh,
                'floor_active': soh < soh_min}
        efc_to_floor = math.log(soh_min) / math.log(1.0 - 1.0 / k)
        binding_rate = efc_to_floor / (365.0 * sum(n for _, n in blocks))

        # --- check 5: which budget paths the evaluation reaches ---
        budget_paths = {
            'hard_constraint': 'shared_energy_storage_data.py:309, investment_cost_total <= budget — '
                               'built inside build_master_problem only',
            'alpha_lower_bound': 'shared_energy_storage_data.py:260, model.alpha.setlb(-budget*1e3) — '
                                 'master only; a recourse evaluation never builds alpha',
            'validity_check': 'shared_resources_planning.py:1634, _check_candidate_first_stage_feasibility '
                              "-> 'investment budget violated'",
            'config_fingerprint': 'p56a_oracle.py:739 includes budget in the config hash',
            'finding': ('a recourse evaluation with a FIXED candidate does not optimize the master, so '
                        'neither the hard constraint nor the alpha bound is built. The binding path is the '
                        'VALIDITY CHECK, plus the config fingerprint. Both are reached with the '
                        'in-memory override, which is why the override must be applied before the '
                        'candidate is validated.'),
        }
    finally:
        guard.uninstall()

    report = {
        'stage': 'Track D pre-flight', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile': {'declared': 0, 'observed': dict(guard.counts), 'failures': guard.verify(0)},
        'plan': {'s_MVA_per_node': S_INV, 'e_MWh_per_node': E_INV, 'invested_year': INVEST_YEAR,
                 'zero_in': [y for y in years if str(y) != str(INVEST_YEAR)], 'nodes': nodes,
                 'single_cohort': True,
                 'source': 'specified explicitly; NOT from _build_positive_bootstrap_candidate'},
        'check1_investment_cost': {
            'discount_factor': sed.discount_factor,
            'market_scenario_probabilities': list(sed.prob_market_scenarios),
            'unit_costs_2025': unit_costs,
            'per_node': per_node, 'total_discounted_expected': total,
            'current_budget': params.budget,
            'exceeds_current_budget': total > params.budget,
            'headroom_at_current_budget': params.budget - total},
        'check2_bounds': bounds,
        'check3_soh': {'k_in_force': k, 'soh_min': soh_min, 'trajectory': trajectory,
                       'efc_to_floor': efc_to_floor,
                       'average_efc_per_day_at_which_floor_binds_by_2035': binding_rate},
        'check4_cell1_solve_mode': {
            'mode': "type='uncoordinated' -> run_without_coordination -> "
                    '_run_operational_planning_without_coordination (shared_resources_planning.py:5816)',
            'structure': 'one distribution_network.optimize per node (:5866) and one '
                         'transmission_network.optimize (:5931). NO consensus loop, NO ESSO solve.',
            'solves': {'dso': f'{len(nodes)} nodes x 12 year-day blocks = {len(nodes)*12}',
                       'tso': '12 year-day blocks', 'total': len(nodes) * 12 + 12},
            'finding': 'cell 1 costs ~48 solves in a single pass, NOT 3,519. The campaign figure '
                       'was a large overestimate.'},
        'check5_budget_paths': budget_paths,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'd_preflight.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != 'check5_budget_paths'},
                     indent=1, default=str)[:3000])
    print(f'\nwritten to {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
