import os
import pandas as pd
import pyomo.opt as po
import pyomo.environ as pe
from shared_energy_storage import SharedEnergyStorage
from shared_energy_storage_parameters import SharedEnergyStorageParameters
from helper_functions import *


# ======================================================================================================================
#  SHARED ENERGY STORAGE Information
# ======================================================================================================================
class SharedEnergyStorageData:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.plots_dir = str()
        self.data_file = str()
        self.params_file = str()
        self.years = list()
        self.days = list()
        self.num_instants = 0
        self.discount_factor = 5e-2
        self.shared_energy_storages = dict()
        self.prob_market_scenarios = dict()         # Probability of market (price) scenarios
        self.cost_investment = dict()
        self.params = SharedEnergyStorageParameters()

    def build_subproblem(self):
        return _build_subproblem_model(self)

    def optimize(self, model, from_warm_start=False):
        print('[INFO] \t\t - Running Shared ESS optimization...')
        return _optimize(model, self.params.solver_params, from_warm_start=from_warm_start)

    def update_model_with_candidate_solution(self, model, candidate_solution):
        _update_model_with_candidate_solution(self, model, candidate_solution)

    def read_shared_energy_storage_data_from_file(self):
        filename = os.path.join(self.data_dir, 'Shared ESS', self.data_file)
        _read_shared_energy_storage_data_from_file(self, filename)

    def read_parameters_from_file(self):
        filename = os.path.join(self.data_dir, 'Shared ESS', self.params_file)
        self.params.read_parameters_from_file(filename)

    def create_shared_energy_storages(self, planning_problem):
        for year in planning_problem.years:
            self.shared_energy_storages[year] = list()
            for node_id in planning_problem.transmission_network.active_distribution_network_nodes:
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = node_id
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].name
                self.shared_energy_storages[year].append(shared_energy_storage)

    def get_shared_energy_storage_idx(self, node_id):
        repr_years = [year for year in self.years]
        for i in range(len(self.shared_energy_storages[repr_years[0]])):
            shared_energy_storage = self.shared_energy_storages[repr_years[0]][i]
            if shared_energy_storage.bus == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Node {node_id} does not have a shared energy storage system! Check network.')
        exit(ERROR_NETWORK_FILE)

    def update_data_with_candidate_solution(self, candidate_solution):
        for year in self.years:
            for shared_ess in self.shared_energy_storages[year]:
                shared_ess.s = candidate_solution[shared_ess.bus][year]['s']
                shared_ess.e = candidate_solution[shared_ess.bus][year]['e']
                shared_ess.e_init = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_RELATIVE_INIT_SOC
                shared_ess.e_min = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MIN_ENERGY_STORED
                shared_ess.e_max = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MAX_ENERGY_STORED


# ======================================================================================================================
#  OPERATIONAL PLANNING functions
# ======================================================================================================================
def _build_subproblem_model(shared_ess_data):

    model = pe.ConcreteModel()
    model.name = 'ESSO, Operational Planning'
    repr_years = [year for year in shared_ess_data.years]
    repr_days = [day for day in shared_ess_data.days]
    total_days = sum([shared_ess_data.days[day] for day in shared_ess_data.days])

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.years = range(len(shared_ess_data.years))
    model.days = range(len(shared_ess_data.days))
    model.periods = range(shared_ess_data.num_instants)
    model.energy_storages = range(len(shared_ess_data.active_distribution_network_nodes))

    # ------------------------------------------------------------------------------------------------------------------
    # Variables
    model.es_s_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Total rated power capacity (considering calendar life)
    model.es_e_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Total rated energy capacity (considering calendar life, not considering degradation)
    model.es_s_investment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Investment in power capacity in year y (complicating variable)
    model.es_e_investment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Invesment in energy capacity in year y (complicating variable)
    model.es_s_investment_fixed = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Benders' -- used to get the dual variables (sensitivities)
    model.es_e_investment_fixed = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # (...)
    model.es_e_capacity_available = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Total Energy capacity available in year y (based on degradation)
    model.es_e_capacity_degradation = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals)  # Energy capacity degradation in year y (based on ESS utilization)
    model.es_e_relative_capacity = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals)  # Relative energy capacity available in year y (based on degradation)
    model.es_soc = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals)
    model.es_pch = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_pdch = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_expected_p = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals, initialize=0.00)
    model.slack_s_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)  # Benders' -- ensures feasibility of the subproblem (numerical issues)
    model.slack_s_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)  # (...)
    model.slack_e_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)  # (...)
    model.slack_e_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)  # (...)
    if shared_ess_data.params.ess_relax_comp:
        model.es_penalty_comp = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if shared_ess_data.params.ess_relax_soc:
        model.es_penalty_soc_up = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_penalty_soc_down = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    if shared_ess_data.params.ess_relax_day_balance:
        model.es_penalty_day_balance_up = pe.Var(model.energy_storages, model.years, model.days, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_penalty_day_balance_down = pe.Var(model.energy_storages, model.years, model.days, domain=pe.NonNegativeReals, initialize=0.00)
    if shared_ess_data.params.ess_relax_capacity_available:
        model.es_penalty_capacity_available_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_penalty_capacity_available_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.00)
    if shared_ess_data.params.ess_relax_installed_capacity:
        model.es_penalty_s_rated_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_penalty_s_rated_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_penalty_e_rated_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_penalty_e_rated_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    if shared_ess_data.params.ess_relax_capacity_degradation:
        model.es_penalty_e_capacity_degradation_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_penalty_e_capacity_degradation_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    if shared_ess_data.params.ess_relax_capacity_relative:
        model.es_penalty_e_relative_capacity_up = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_penalty_e_relative_capacity_down = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    if shared_ess_data.params.ess_interface_relax:
        model.es_penalty_expected_p_up = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_penalty_expected_p_down = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)

    for e in model.energy_storages:
        for y in model.years:
            model.es_e_capacity_degradation[e, y].setub(1.00)
            model.es_e_capacity_degradation[e, y].fix(1.00)
            e_init = shared_ess_data.shared_energy_storages[repr_years[y]][e].e
            for d in model.days:
                for p in model.periods:
                    model.es_soc[e, y, d, p] = e_init * ENERGY_STORAGE_RELATIVE_INIT_SOC
            for x in model.years:
                model.es_e_relative_capacity[e, y, x].fix(0.00)
                if shared_ess_data.params.ess_relax_capacity_relative:
                    model.es_penalty_e_relative_capacity_up[e, y, x].fix(0.00)
                    model.es_penalty_e_relative_capacity_down[e, y, x].fix(0.00)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Yearly Power and Energy ratings as a function of yearly investments
    model.rated_s_capacity = pe.ConstraintList()
    model.rated_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        total_s_capacity_per_year = [0.0 for _ in model.years]
        total_e_capacity_per_year = [0.0 for _ in model.years]
        for y in model.years:
            shared_energy_storage = shared_ess_data.shared_energy_storages[repr_years[y]][e]
            tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y]]))
            max_tcal_norm = min(y + tcal_norm, len(shared_ess_data.years))
            for x in range(y, max_tcal_norm):
                total_s_capacity_per_year[x] += model.es_s_investment[e, y]
                total_e_capacity_per_year[x] += model.es_e_investment[e, y]
        for y in model.years:
            if shared_ess_data.params.ess_relax_installed_capacity:
                model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] == model.es_penalty_s_rated_up[e, y] - model.es_penalty_s_rated_down[e, y])
                model.rated_s_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] == model.es_penalty_e_rated_up[e, y] - model.es_penalty_e_rated_down[e, y])
            else:
                model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] >= -SMALL_TOLERANCE)
                model.rated_s_capacity.add(model.es_s_rated[e, y] - total_s_capacity_per_year[y] <= SMALL_TOLERANCE)
                model.rated_e_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] >= -SMALL_TOLERANCE)
                model.rated_e_capacity.add(model.es_e_rated[e, y] - total_e_capacity_per_year[y] <= SMALL_TOLERANCE)

    # - Energy capacities available in year y (as a function of degradation)
    model.energy_storage_available_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            capacity_e_available_year_y = model.es_e_investment[e, y] * model.es_e_relative_capacity[e, y, y]
            for x in range(y - 1, -1, -1):
                capacity_e_available_year_y += model.es_e_investment[e, x] * model.es_e_relative_capacity[e, x, y]
            if shared_ess_data.params.ess_relax_capacity_available:
                model.energy_storage_available_e_capacity.add(model.es_e_capacity_available[e, y] - capacity_e_available_year_y == model.es_penalty_capacity_available_up[e, y] - model.es_penalty_capacity_available_down[e, y])
            else:
                model.energy_storage_available_e_capacity.add(model.es_e_capacity_available[e, y] - capacity_e_available_year_y >= -SMALL_TOLERANCE)
                model.energy_storage_available_e_capacity.add(model.es_e_capacity_available[e, y] - capacity_e_available_year_y <= SMALL_TOLERANCE)

    # - Yearly degradation
    model.energy_storage_yearly_degradation = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            cl_nom = shared_energy_storage.cl_nom
            dod_nom = shared_energy_storage.dod_nom

            total_ch_dch_day = 0.0
            total_available_capacity = cl_nom * dod_nom * 2 * model.es_e_capacity_available[e, y]
            for d in model.days:
                day = repr_days[d]
                num_days = shared_ess_data.days[day]
                for p in model.periods:
                    pch = model.es_pch[e, y, d, p]
                    pdch = model.es_pdch[e, y, d, p]
                    total_ch_dch_day += (num_days / 365) * (pch + pdch)
            if shared_ess_data.params.ess_relax_capacity_degradation:
                model.energy_storage_yearly_degradation.add(model.es_e_capacity_degradation[e, y] * total_available_capacity - total_ch_dch_day == model.es_penalty_e_capacity_degradation_up[e, y] - model.es_penalty_e_capacity_degradation_down[e, y])
            else:
                model.energy_storage_yearly_degradation.add(model.es_e_capacity_degradation[e, y] * total_available_capacity - total_ch_dch_day >= -SMALL_TOLERANCE)
                model.energy_storage_yearly_degradation.add(model.es_e_capacity_degradation[e, y] * total_available_capacity - total_ch_dch_day <= SMALL_TOLERANCE)

    # - Relative energy capacity
    # - Reflects the degradation of the capacity invested on ESS e in year Y at year X ahead
    model.energy_storage_relative_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            tcal_norm = round(shared_energy_storage.t_cal / shared_ess_data.years[repr_years[y]])
            max_tcal_norm = min(y + tcal_norm, len(shared_ess_data.years))

            # - Relative capacity
            relative_capacity_year_y_in_x = 1.0
            model.es_e_capacity_degradation[e, y].fixed = False
            model.es_e_relative_capacity[e, y, y].fix(relative_capacity_year_y_in_x)
            for x in range(y + 1, max_tcal_norm):
                model.es_e_capacity_degradation[e, x].fixed = False
                model.es_e_relative_capacity[e, y, x].fixed = False
                relative_capacity_year_y_in_x *= (1 - model.es_e_capacity_degradation[e, x - 1]) ** (total_days * shared_ess_data.years[repr_years[y]])          # Relative capacity in year y reflects the accumulated degradation
                if shared_ess_data.params.ess_relax_capacity_relative:
                    model.es_penalty_e_relative_capacity_up[e, y, x].fixed = False
                    model.es_penalty_e_relative_capacity_down[e, y, x].fixed = False
                    model.energy_storage_relative_e_capacity.add(model.es_e_relative_capacity[e, y, x] - relative_capacity_year_y_in_x == model.es_penalty_e_relative_capacity_up[e, y, x] - model.es_penalty_e_relative_capacity_down[e, y, x])
                else:
                    model.energy_storage_relative_e_capacity.add(model.es_e_relative_capacity[e, y, x] - relative_capacity_year_y_in_x >= -SMALL_TOLERANCE)
                    model.energy_storage_relative_e_capacity.add(model.es_e_relative_capacity[e, y, x] - relative_capacity_year_y_in_x <= SMALL_TOLERANCE)

    # - P, Q, S, SoC, Pup and Pdown as a function of available capacities
    model.energy_storage_limits = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            for d in model.days:
                for p in model.periods:
                    model.energy_storage_limits.add(model.es_pch[e, y, d, p] <= model.es_s_rated[e, y])
                    model.energy_storage_limits.add(model.es_pdch[e, y, d, p] <= model.es_s_rated[e, y])
                    model.energy_storage_limits.add(model.es_soc[e, y, d, p] >= model.es_e_capacity_available[e, y] * ENERGY_STORAGE_MIN_ENERGY_STORED)
                    model.energy_storage_limits.add(model.es_soc[e, y, d, p] <= model.es_e_capacity_available[e, y] * ENERGY_STORAGE_MAX_ENERGY_STORED)

    # - Shared ESS operation
    model.energy_storage_operation = pe.ConstraintList()
    model.energy_storage_balance = pe.ConstraintList()
    model.energy_storage_day_balance = pe.ConstraintList()
    model.energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.energy_storage_expected_power = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:

            year = repr_years[y]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            eff_charge, eff_discharge = shared_energy_storage.eff_ch, shared_energy_storage.eff_dch
            soc_init = model.es_e_capacity_available[e, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC
            soc_final = model.es_e_capacity_available[e, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC

            for d in model.days:
                for p in model.periods:

                    pch = model.es_pch[e, y, d, p]
                    pdch = model.es_pdch[e, y, d, p]

                    if p > 0:
                        if shared_ess_data.params.ess_relax_soc:
                            model.energy_storage_balance.add(model.es_soc[e, y, d, p] - model.es_soc[e, y, d, p - 1] - (pch * eff_charge - pdch / eff_discharge) == model.es_penalty_soc_up[e, y, d, p] - model.es_penalty_soc_down[e, y, d, p])
                        else:
                            model.energy_storage_balance.add(model.es_soc[e, y, d, p] - model.es_soc[e, y, d, p - 1] - (pch * eff_charge - pdch / eff_discharge) >= -SMALL_TOLERANCE)
                            model.energy_storage_balance.add(model.es_soc[e, y, d, p] - model.es_soc[e, y, d, p - 1] - (pch * eff_charge - pdch / eff_discharge) <= SMALL_TOLERANCE)
                    else:
                        if shared_ess_data.params.ess_relax_soc:
                            model.energy_storage_balance.add(model.es_soc[e, y, d, p] - soc_init - (pch * eff_charge - pdch / eff_discharge) == model.es_penalty_soc_up[e, y, d, p] - model.es_penalty_soc_down[e, y, d, p])
                        else:
                            model.energy_storage_balance.add(model.es_soc[e, y, d, p] - soc_init - (pch * eff_charge - pdch / eff_discharge) >= -SMALL_TOLERANCE)
                            model.energy_storage_balance.add(model.es_soc[e, y, d, p] - soc_init - (pch * eff_charge - pdch / eff_discharge) <= SMALL_TOLERANCE)

                    # Charging/discharging complementarity constraint
                    if shared_ess_data.params.ess_relax_comp:
                        model.energy_storage_ch_dch_exclusion.add(pch * pdch <= model.es_penalty_comp[e, y, d, p])
                    else:
                        # NLP formulation
                        model.energy_storage_ch_dch_exclusion.add(pch * pdch >= -SMALL_TOLERANCE)
                        model.energy_storage_ch_dch_exclusion.add(pch * pdch <= SMALL_TOLERANCE)

                if shared_ess_data.params.ess_relax_day_balance:
                    model.energy_storage_day_balance.add(model.es_soc[e, y, d, len(model.periods) - 1] - soc_final == model.es_penalty_day_balance_up[e, y, d] - model.es_penalty_day_balance_down[e, y, d])
                else:
                    model.energy_storage_day_balance.add(model.es_soc[e, y, d, len(model.periods) - 1] - soc_final >= -SMALL_TOLERANCE)
                    model.energy_storage_day_balance.add(model.es_soc[e, y, d, len(model.periods) - 1] - soc_final <= SMALL_TOLERANCE)

            # Expected P and Q
            for d in model.days:
                for p in model.periods:
                    expected_p = (model.es_pch[e, y, d, p] - model.es_pdch[e, y, d, p])
                    if shared_ess_data.params.ess_interface_relax:
                        model.energy_storage_expected_power.add(model.es_expected_p[e, y, d, p] - expected_p == model.es_penalty_expected_p_up[e, y, d, p] - model.es_penalty_expected_p_down[e, y, d, p])
                    else:
                        model.energy_storage_expected_power.add(model.es_expected_p[e, y, d, p] - expected_p >= -SMALL_TOLERANCE)
                        model.energy_storage_expected_power.add(model.es_expected_p[e, y, d, p] - expected_p <= SMALL_TOLERANCE)

    # - Sensitivities - Einv and Sinv as a function of Einv_fixed and Sinv_fixed
    model.sensitivities_s = pe.ConstraintList()
    model.sensitivities_e = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            # Note: slack variables added to ensure feasibility (numerical issues)
            model.sensitivities_s.add(model.es_s_investment[e, y] + model.slack_s_up[e, y] - model.slack_s_down[e, y] == model.es_s_investment_fixed[e, y])
            model.sensitivities_e.add(model.es_e_investment[e, y] + model.slack_e_up[e, y] - model.slack_e_down[e, y] == model.es_e_investment_fixed[e, y])

    # ------------------------------------------------------------------------------------------------------------------
    # Objective function
    slack_penalty = 0.0
    for e in model.energy_storages:
        for y in model.years:
            for d in model.days:

                for p in model.periods:

                    if shared_ess_data.params.ess_relax_comp:
                        slack_penalty += PENALTY_ESS_COMPLEMENTARITY * model.es_penalty_comp[e, y, d, p]

                    if shared_ess_data.params.ess_relax_soc:
                        slack_penalty += PENALTY_ESS_SOC * (model.es_penalty_soc_up[e, y, d, p] + model.es_penalty_soc_down[e, y, d, p])

                if shared_ess_data.params.ess_relax_day_balance:
                    slack_penalty += PENALTY_ESS_DAY_BALANCE * (model.es_penalty_day_balance_up[e, y, d] + model.es_penalty_day_balance_down[e, y, d])

            # Slack penalties
            slack_penalty += PENALTY_ESS_SLACK_FEASIBILITY * (model.slack_s_up[e, y] + model.slack_s_down[e, y])
            slack_penalty += PENALTY_ESS_SLACK_FEASIBILITY * (model.slack_e_up[e, y] + model.slack_e_down[e, y])

            # Capacity available penalties
            if shared_ess_data.params.ess_relax_capacity_available:
                slack_penalty += PENALTY_ESS_CAPACITY_AVAILABLE * (model.es_penalty_capacity_available_up[e, y] + model.es_penalty_capacity_available_down[e, y])

            # Installed capacity penalties
            if shared_ess_data.params.ess_relax_installed_capacity:
                slack_penalty += PENALTY_ESS_CAPACITY_INSTALLED * (model.es_penalty_s_rated_up[e, y] + model.es_penalty_s_rated_down[e, y])
                slack_penalty += PENALTY_ESS_CAPACITY_INSTALLED * (model.es_penalty_e_rated_up[e, y] + model.es_penalty_e_rated_down[e, y])

            # Capacity degradation penalties
            if shared_ess_data.params.ess_relax_capacity_degradation:
                slack_penalty += PENALTY_ESS_CAPACITY_DEGRADATION * (model.es_penalty_e_capacity_degradation_up[e, y] + model.es_penalty_e_capacity_degradation_down[e, y])

            # Relative capacity penalties
            if shared_ess_data.params.ess_relax_capacity_relative:
                for x in model.years:
                    slack_penalty += PENALTY_ESS_CAPACITY_RELATIVE * (model.es_penalty_e_relative_capacity_up[e, y, x] + model.es_penalty_e_relative_capacity_down[e, y, x])

            # Expected active power
            if shared_ess_data.params.ess_interface_relax:
                for d in model.days:
                    for p in model.periods:
                        slack_penalty += PENALTY_ESS_EXPECTED_VALUES * (model.es_penalty_expected_p_up[e, y, d, p] + model.es_penalty_expected_p_down[e, y, d, p])

    obj = slack_penalty
    model.objective = pe.Objective(sense=pe.minimize, expr=obj)

    # Define that we want the duals
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    return model


def _optimize(model, params, from_warm_start=False):

    solver = po.SolverFactory(params.solver, executable=params.solver_path)

    if from_warm_start:
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['warm_start_bound_push'] = 1e-9
        solver.options['warm_start_bound_frac'] = 1e-9
        solver.options['warm_start_slack_bound_frac'] = 1e-9
        solver.options['warm_start_slack_bound_push'] = 1e-9
        solver.options['warm_start_mult_bound_push'] = 1e-9
        '''
        solver.options['mu_strategy'] = 'monotone'
        solver.options['mu_init'] = 1e-9
        '''

    if params.verbose:
        solver.options['print_level'] = 6
        solver.options['output_file'] = 'optim_log.txt'

    '''
    if params.solver == 'ipopt':
        solver.options['tol'] = params.solver_tol
        solver.options['acceptable_tol'] = params.solver_tol * 1e3
        solver.options['acceptable_iter'] = 5
        solver.options['max_iter'] = 10000
    '''

    result = solver.solve(model, tee=params.verbose)

    return result


def _update_model_with_candidate_solution(shared_ess_data, model, candidate_solution):
    repr_years = [year for year in shared_ess_data.years]
    for e in model.energy_storages:
        for y in model.years:
            year = repr_years[y]
            node_id = shared_ess_data.shared_energy_storages[year][e].bus
            model.es_s_investment_fixed[e, y].fix(candidate_solution[node_id][year]['s'])
            model.es_e_investment_fixed[e, y].fix(candidate_solution[node_id][year]['e'])


# ======================================================================================================================
#  NETWORK PLANNING read functions
# ======================================================================================================================
def _read_shared_energy_storage_data_from_file(shared_ess_data, filename):

    try:
        num_scenarios, shared_ess_data.prob_market_scenarios = _get_operational_scenarios_info_from_excel_file(filename, 'Scenarios')
        investment_costs = _get_investment_costs_from_excel_file(filename, 'Investment Cost', len(shared_ess_data.years))
        shared_ess_data.cost_investment = investment_costs
    except:
        print(f'[ERROR] File {filename}. Exiting...')
        exit(ERROR_OPERATIONAL_DATA_FILE)


def _get_operational_scenarios_info_from_excel_file(filename, sheet_name):

    num_scenarios = 0
    prob_scenarios = list()

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        for i in range(num_scenarios):
            if is_number(df.iloc[0, i + 2]):
                prob_scenarios.append(float(df.iloc[0, i + 2]))
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)

    if num_scenarios != len(prob_scenarios):
        print('[WARNING] EnergyStorage file. Number of scenarios different from the probability vector!')

    if round(sum(prob_scenarios), 2) != 1.00:
        print('[ERROR] Probability of scenarios does not add up to 100%. Check file {}. Exiting.'.format(filename))
        exit(ERROR_OPERATIONAL_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_investment_costs_from_excel_file(filename, sheet_name, num_years):

    try:

        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        data = {
            'power_capacity': dict(),
            'energy_capacity': dict()
        }

        for i in range(num_years):

            year = int(df.iloc[0, i + 1])

            if is_number(df.iloc[1, i + 1]):
                data['power_capacity'][year] = float(df.iloc[1, i + 1])

            if is_number(df.iloc[2, i + 1]):
                data['energy_capacity'][year] = float(df.iloc[2, i + 1])

        return data

    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(ERROR_MARKET_DATA_FILE)
