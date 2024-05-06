import os
from math import isclose, acos, tan
import pandas as pd
import pyomo.opt as po
import pyomo.environ as pe
from openpyxl import Workbook
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

    def process_results(self, model):
        results = dict()
        results['capacity'] = self.get_investment_and_available_capacities(model)
        results['operation'] = dict()
        results['operation']['aggregated'] = self.process_results_aggregated(model)
        results['operation']['detailed'] = self.process_results_detailed(model)
        results['soh'] = dict()
        results['soh']['aggregated'] = self.process_soh_results_aggregated(model)
        results['soh']['detailed'] = self.process_soh_results_detailed(model)
        if self.params.slacks:
            results['relaxation_variables'] = dict()
            results['relaxation_variables']['operation'] = dict()
            results['relaxation_variables']['operation']['detailed'] = self.process_relaxation_variables_operation_detailed(model)
        return results

    def process_results_aggregated(self, model):
        return _process_results_aggregated(self, model)

    def process_results_detailed(self, model):
        return _process_results_detailed(self, model)

    def process_soh_results_aggregated(self, model):
        return _process_soh_results_aggregated(self, model)

    def process_soh_results_detailed(self, model):
        return _process_soh_results_detailed(self, model)

    def process_relaxation_variables_operation_detailed(self, model):
        return _process_relaxation_variables_operation_detailed(self, model)

    def write_optimization_results_to_excel(self, model):
        results = self.process_results(model)
        _write_optimization_results_to_excel(self, self.results_dir, results)

    def update_data_with_candidate_solution(self, candidate_solution):
        for year in self.years:
            for shared_ess in self.shared_energy_storages[year]:
                shared_ess.s = candidate_solution[shared_ess.bus][year]['s']
                shared_ess.e = candidate_solution[shared_ess.bus][year]['e']
                shared_ess.e_init = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_RELATIVE_INIT_SOC
                shared_ess.e_min = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MIN_ENERGY_STORED
                shared_ess.e_max = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MAX_ENERGY_STORED

    def get_available_capacities(self, model, ess_idx, year_idx):
        s_available = 0.00
        e_available = 0.00
        for y_inv in model.years:
            s_available += pe.value(model.es_s_available_per_unit[ess_idx, y_inv, year_idx])
            e_available += pe.value(model.es_e_available_per_unit[ess_idx, y_inv, year_idx])
        return s_available, e_available

    def get_investment_and_available_capacities(self, model):
        return _get_investment_and_available_capacities(self, model)

    def write_ess_results_to_excel(self, workbook, shared_ess_capacity):
        _write_ess_capacity_investment_to_excel(self, workbook, shared_ess_capacity['investment'], initial_sheet=False)
        _write_ess_capacity_rated_available_to_excel(self, workbook, shared_ess_capacity)

    def write_operation_relaxation_slacks_results_to_excel(self, workbook, results):
        _write_detailed_operation_relaxation_slacks_results_to_excel(self, workbook, results['detailed'])


# ======================================================================================================================
#  OPERATIONAL PLANNING functions
# ======================================================================================================================
def _build_subproblem_model(shared_ess_data):

    model = pe.ConcreteModel()
    model.name = 'ESSO, Operational Planning'
    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.years = range(len(shared_ess_data.years))
    model.days = range(len(shared_ess_data.days))
    model.periods = range(shared_ess_data.num_instants)
    model.energy_storages = range(len(shared_ess_data.active_distribution_network_nodes))

    # ------------------------------------------------------------------------------------------------------------------
    # Variables
    model.es_s_investment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_investment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_investment_fixed = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_investment_fixed = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_investment_slack_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_investment_slack_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_investment_slack_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_investment_slack_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_rated = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_soc = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_snet = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals, initialize=0.0)
    model.es_pnet = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals, initialize=0.0)
    model.es_qnet = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals, initialize=0.0)
    model.slack_es_snet_up = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_es_snet_down = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_es_pnet_up = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_es_pnet_down = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_es_qnet_up = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_es_qnet_down = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)

    model.es_s_rated_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_rated_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_available_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_available_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_rated_per_unit.fix(0.00)
    model.es_e_rated_per_unit.fix(0.00)

    model.es_soc_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_sch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_pch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_qch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.Reals, initialize=0.00)
    model.es_sdch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_pdch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_qdch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.Reals, initialize=0.00)
    model.es_avg_ch_dch_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_soh_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=1.00, bounds=(0.00, 1.00))
    model.es_degradation_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.00, bounds=(0.00, 1.00))
    model.es_soh_per_unit_cumul = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=1.00, bounds=(0.00, 1.00))
    model.es_degradation_per_unit_cumul = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.00, bounds=(0.00, 1.00))
    if shared_ess_data.params.slacks:
        model.slack_es_comp_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_es_sch_up_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_es_sch_down_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_es_sdch_up_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_es_sdch_down_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_soh_per_unit.fix(1.00)
    model.es_degradation_per_unit.fix(0.00)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Sinv and Einv fixing constraints
    model.energy_storage_capacity_fixing = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            model.energy_storage_capacity_fixing.add(model.es_s_investment[e, y] == model.es_s_investment_fixed[e, y] + model.es_s_investment_slack_up[e, y] - model.es_s_investment_slack_down[e, y])
            model.energy_storage_capacity_fixing.add(model.es_e_investment[e, y] == model.es_e_investment_fixed[e, y] + model.es_e_investment_slack_up[e, y] - model.es_e_investment_slack_down[e, y])

    # - Rated capacities of each investment
    model.rated_s_capacity_unit = pe.ConstraintList()
    model.rated_e_capacity_unit = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:
            shared_energy_storage = shared_ess_data.shared_energy_storages[repr_years[y_inv]][e]
            tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y_inv]]))
            max_tcal_norm = min(y_inv + tcal_norm, len(shared_ess_data.years))
            for y in range(y_inv, max_tcal_norm):
                model.es_s_rated_per_unit[e, y_inv, y].fixed = False
                model.es_e_rated_per_unit[e, y_inv, y].fixed = False
                model.rated_s_capacity_unit.add(model.es_s_rated_per_unit[e, y_inv, y] == model.es_s_investment[e, y_inv])
                model.rated_e_capacity_unit.add(model.es_e_rated_per_unit[e, y_inv, y] == model.es_e_investment[e, y_inv])

    # - Rated yearly capacities as a function of yearly investments
    model.rated_s_capacity = pe.ConstraintList()
    model.rated_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            total_s_capacity = 0.00
            total_e_capacity = 0.00
            for y_inv in model.years:
                total_s_capacity += model.es_s_rated_per_unit[e, y_inv, y]
                total_e_capacity += model.es_e_rated_per_unit[e, y_inv, y]
            model.rated_s_capacity.add(model.es_s_rated[e, y] == total_s_capacity)
            model.rated_e_capacity.add(model.es_e_rated[e, y] == total_e_capacity)

    # - Available capacities of each investment
    model.available_s_capacity_unit = pe.ConstraintList()
    model.available_e_capacity_unit = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:
            for y in model.years:
                model.available_s_capacity_unit.add(model.es_s_available_per_unit[e, y_inv, y] == model.es_s_rated_per_unit[e, y_inv, y])
                model.available_e_capacity_unit.add(model.es_e_available_per_unit[e, y_inv, y] == model.es_e_rated_per_unit[e, y_inv, y] * model.es_soh_per_unit_cumul[e, y_inv, y])

    # - Sum of charging and discharging power for the yearly average day (aux, used to estimate degradation of ESSs)
    model.energy_storage_charging_discharging = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:
            for y in model.years:
                avg_ch_dch = 0.0
                for d in model.days:
                    day = repr_days[d]
                    num_days = shared_ess_data.days[day]
                    for p in model.periods:
                        sch = model.es_sch_per_unit[e, y_inv, y, d, p]
                        sdch = model.es_sdch_per_unit[e, y_inv, y, d, p]
                        avg_ch_dch += (num_days / 365.00) * (sch + sdch)
                model.energy_storage_charging_discharging.add(model.es_avg_ch_dch_per_unit[e, y_inv, y] == avg_ch_dch)

    # - Capacity degradation
    model.energy_storage_capacity_degradation = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:
            num_years = shared_ess_data.years[repr_years[y_inv]]
            shared_energy_storage = shared_ess_data.shared_energy_storages[repr_years[y_inv]][e]
            tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y_inv]]))
            max_tcal_norm = min(y_inv + tcal_norm, len(shared_ess_data.years))
            for y in range(y_inv, max_tcal_norm):
                model.es_soh_per_unit.fixed = False
                model.es_degradation_per_unit.fixed = False
                model.energy_storage_capacity_degradation.add(model.es_degradation_per_unit[e, y_inv, y] * (2 * shared_energy_storage.cl_nom * model.es_e_available_per_unit[e, y_inv, y]) == model.es_avg_ch_dch_per_unit[e, y_inv, y])
                model.energy_storage_capacity_degradation.add(model.es_soh_per_unit[e, y_inv, y] == 1.00 - model.es_degradation_per_unit[e, y_inv, y])
                model.energy_storage_capacity_degradation.add(model.es_soh_per_unit[e, y_inv, y] >= shared_energy_storage.soh_min)
                prev_deg = 0.00
                if y > 0:
                    prev_deg = model.es_degradation_per_unit_cumul[e, y_inv, y - 1]
                model.energy_storage_capacity_degradation.add(model.es_degradation_per_unit_cumul[e, y_inv, y] == prev_deg + (model.es_degradation_per_unit[e, y_inv, y]) ** (num_years * 365))
                model.energy_storage_capacity_degradation.add(model.es_soh_per_unit_cumul[e, y_inv, y] == 1.00 - model.es_degradation_per_unit_cumul[e, y_inv, y])

    # - P, Q, S, SoC, per unit as a function of available capacities
    model.energy_storage_limits = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:
            shared_energy_storage = shared_ess_data.shared_energy_storages[repr_years[y_inv]][e]
            max_phi = acos(shared_energy_storage.max_pf)
            min_phi = acos(shared_energy_storage.min_pf)
            for y in model.years:
                s_max = model.es_s_available_per_unit[e, y_inv, y]
                for d in model.days:
                    for p in model.periods:

                        sch = model.es_sch_per_unit[e, y_inv, y, d, p]
                        pch = model.es_pch_per_unit[e, y_inv, y, d, p]
                        qch = model.es_qch_per_unit[e, y_inv, y, d, p]
                        sdch = model.es_sdch_per_unit[e, y_inv, y, d, p]
                        pdch = model.es_pdch_per_unit[e, y_inv, y, d, p]
                        qdch = model.es_qdch_per_unit[e, y_inv, y, d, p]

                        model.energy_storage_limits.add(sch <= s_max)
                        model.energy_storage_limits.add(pch <= s_max)
                        model.energy_storage_limits.add(qch <= s_max * tan(max_phi))
                        model.energy_storage_limits.add(qch >= s_max * tan(min_phi))
                        model.energy_storage_limits.add(qch >= -s_max)
                        model.energy_storage_limits.add(sdch <= s_max)
                        model.energy_storage_limits.add(pdch <= s_max)
                        model.energy_storage_limits.add(qdch <= s_max * tan(max_phi))
                        model.energy_storage_limits.add(qdch >= s_max * tan(min_phi))

                        model.energy_storage_limits.add(model.es_soc_per_unit[e, y_inv, y, d, p] >= model.es_e_available_per_unit[e, y_inv, y] * ENERGY_STORAGE_MIN_ENERGY_STORED)
                        model.energy_storage_limits.add(model.es_soc_per_unit[e, y_inv, y, d, p] <= model.es_e_available_per_unit[e, y_inv, y] * ENERGY_STORAGE_MAX_ENERGY_STORED)

    # - Shared ESS operation, per unit
    model.energy_storage_operation = pe.ConstraintList()
    model.energy_storage_balance = pe.ConstraintList()
    model.energy_storage_day_balance = pe.ConstraintList()
    model.energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.energy_storage_expected_power = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:

            year_inv = repr_years[y_inv]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year_inv][e]
            eff_charge = shared_energy_storage.eff_ch
            eff_discharge = shared_energy_storage.eff_dch

            for y in model.years:

                soc_init = model.es_e_available_per_unit[e, y_inv, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC

                for d in model.days:
                    for p in model.periods:

                        sch = model.es_sch_per_unit[e, y_inv, y, d, p]
                        pch = model.es_pch_per_unit[e, y_inv, y, d, p]
                        qch = model.es_qch_per_unit[e, y_inv, y, d, p]
                        sdch = model.es_sdch_per_unit[e, y_inv, y, d, p]
                        pdch = model.es_pdch_per_unit[e, y_inv, y, d, p]
                        qdch = model.es_qdch_per_unit[e, y_inv, y, d, p]

                        if shared_ess_data.params.slacks:
                            model.energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2 + model.slack_es_sch_up_per_unit[e, y_inv, y, d, p] - model.slack_es_sch_down_per_unit[e, y_inv, y, d, p])
                            model.energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2 + model.slack_es_sdch_up_per_unit[e, y_inv, y, d, p] - model.slack_es_sdch_down_per_unit[e, y_inv, y, d, p])
                        else:
                            model.energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2)
                            model.energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2)

                        if p > 0:
                            model.energy_storage_balance.add(model.es_soc_per_unit[e, y_inv, y, d, p] - model.es_soc_per_unit[e, y_inv, y, d, p - 1] == sch * eff_charge - sdch / eff_discharge)
                        else:
                            model.energy_storage_balance.add(model.es_soc_per_unit[e, y_inv, y, d, p] - soc_init == sch * eff_charge - sdch / eff_discharge)

                        # Charging/discharging complementarity constraint
                        if shared_ess_data.params.slacks:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch <= model.slack_es_comp_per_unit[e, y_inv, y, d, p])
                        else:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch == 0.00)

    # - Shared ESS operation, aggregated
    model.energy_storage_operation_agg = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            for d in model.days:
                for p in model.periods:
                    agg_snet = 0.00
                    agg_pnet = 0.00
                    agg_qnet = 0.00
                    agg_soc = 0.00
                    for y_inv in model.years:
                        agg_snet += (model.es_sch_per_unit[e, y_inv, y, d, p] - model.es_sdch_per_unit[e, y_inv, y, d, p])
                        agg_pnet += (model.es_pch_per_unit[e, y_inv, y, d, p] - model.es_pdch_per_unit[e, y_inv, y, d, p])
                        agg_qnet += (model.es_qch_per_unit[e, y_inv, y, d, p] - model.es_qdch_per_unit[e, y_inv, y, d, p])
                        agg_soc += model.es_soc_per_unit[e, y_inv, y, d, p]
                    model.energy_storage_operation_agg.add(model.es_snet[e, y, d, p] == agg_snet + model.slack_es_snet_up[e, y, d, p] - model.slack_es_snet_down[e, y, d, p])
                    model.energy_storage_operation_agg.add(model.es_pnet[e, y, d, p] == agg_pnet + model.slack_es_pnet_up[e, y, d, p] - model.slack_es_pnet_down[e, y, d, p])
                    model.energy_storage_operation_agg.add(model.es_qnet[e, y, d, p] == agg_qnet + model.slack_es_qnet_up[e, y, d, p] - model.slack_es_qnet_down[e, y, d, p])
                    model.energy_storage_operation_agg.add(model.es_soc[e, y, d, p] == agg_soc)

    # ------------------------------------------------------------------------------------------------------------------
    # Objective function
    slack_penalty = 0.0
    for e in model.energy_storages:
        for y_inv in model.years:

            # Slacks for investment fixing
            slack_penalty += PENALTY_SLACK * (model.es_s_investment_slack_up[e, y_inv] + model.es_s_investment_slack_down[e, y_inv])
            slack_penalty += PENALTY_SLACK * (model.es_e_investment_slack_up[e, y_inv] + model.es_e_investment_slack_down[e, y_inv])

            if shared_ess_data.params.slacks:

                # Operation slacks
                for y in model.years:
                    for d in model.days:
                        for p in model.periods:
                            slack_penalty += PENALTY_SLACK * model.slack_es_comp_per_unit[e, y_inv, y, d, p]
                            slack_penalty += PENALTY_SLACK * (model.slack_es_sch_up_per_unit[e, y_inv, y, d, p] + model.slack_es_sch_down_per_unit[e, y_inv, y, d, p])
                            slack_penalty += PENALTY_SLACK * (model.slack_es_sdch_up_per_unit[e, y_inv, y, d, p] + model.slack_es_sdch_down_per_unit[e, y_inv, y, d, p])

                # Aggregation slacks
                for d in model.days:
                    for p in model.periods:
                        slack_penalty += PENALTY_SLACK * (model.slack_es_snet_up[e, y, d, p] + model.slack_es_snet_down[e, y, d, p])
                        slack_penalty += PENALTY_SLACK * (model.slack_es_pnet_up[e, y, d, p] + model.slack_es_pnet_down[e, y, d, p])
                        slack_penalty += PENALTY_SLACK * (model.slack_es_qnet_up[e, y, d, p] + model.slack_es_qnet_down[e, y, d, p])

    model.objective = pe.Objective(sense=pe.minimize, expr=slack_penalty)

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

    if params.solver == 'ipopt':
        solver.options['tol'] = params.solver_tol
        solver.options['acceptable_tol'] = params.solver_tol * 1e3
        solver.options['acceptable_iter'] = 5
        solver.options['max_iter'] = 10000
        solver.options['linear_solver'] = params.linear_solver

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


# ======================================================================================================================
#   Shared ESS -- Process results
# ======================================================================================================================
def _process_results_aggregated(shared_ess_data, model):

    processed_results = dict()

    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    for y in model.years:
        year = repr_years[y]
        processed_results[year] = dict()
        for d in model.days:
            day = repr_days[d]
            processed_results[year][day] = dict()
            for e in model.energy_storages:
                node_id = shared_ess_data.shared_energy_storages[year][e].bus
                _, capacity = shared_ess_data.get_available_capacities(model, e, y)
                if isclose(capacity, 0.00, abs_tol=SMALL_TOLERANCE):
                    capacity = 1.00
                processed_results[year][day][node_id] = dict()
                processed_results[year][day][node_id]['s'] = list()
                processed_results[year][day][node_id]['soc'] = list()
                processed_results[year][day][node_id]['soc_perc'] = list()
                for p in model.periods:
                    s_net = pe.value(model.es_snet[e, y, d, p])
                    soc = pe.value(model.es_soc[e, y, d, p])
                    soc_perc = soc / capacity
                    processed_results[year][day][node_id]['s'].append(s_net)
                    processed_results[year][day][node_id]['soc'].append(soc)
                    processed_results[year][day][node_id]['soc_perc'].append(soc_perc)

    return processed_results


def _process_results_detailed(shared_ess_data, model):

    processed_results = dict()

    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    for y_inv in model.years:
        year_inv = repr_years[y_inv]
        processed_results[year_inv] = dict()
        for y_curr in model.years:
            year_curr = repr_years[y_curr]
            processed_results[year_inv][year_curr] = dict()
            for d in model.days:
                day = repr_days[d]
                processed_results[year_inv][year_curr][day] = dict()
                for e in model.energy_storages:
                    node_id = shared_ess_data.shared_energy_storages[year_curr][e].bus
                    capacity = pe.value(model.es_e_available_per_unit[e, y_inv, y_curr])
                    if isclose(capacity, 0.00, abs_tol=SMALL_TOLERANCE):
                        capacity = 1.00
                    processed_results[year_inv][year_curr][day][node_id] = dict()
                    processed_results[year_inv][year_curr][day][node_id]['s'] = list()
                    processed_results[year_inv][year_curr][day][node_id]['soc'] = list()
                    processed_results[year_inv][year_curr][day][node_id]['soc_perc'] = list()
                    for p in model.periods:
                        s_net = pe.value(model.es_sch_per_unit[e, y_inv, y_curr, d, p] - model.es_sdch_per_unit[e, y_inv, y_curr, d, p])
                        soc = pe.value(model.es_soc_per_unit[e, y_inv, y_curr, d, p])
                        soc_perc = soc / capacity
                        processed_results[year_inv][year_curr][day][node_id]['s'].append(s_net)
                        processed_results[year_inv][year_curr][day][node_id]['soc'].append(soc)
                        processed_results[year_inv][year_curr][day][node_id]['soc_perc'].append(soc_perc)

    return processed_results


def _process_soh_results_aggregated(shared_ess_data, model):

    processed_results = dict()
    repr_years = [year for year in shared_ess_data.years]

    for y in model.years:
        year = repr_years[y]
        processed_results[year] = {
            's_rated': dict(), 'e_rated': dict(),
            's_available': dict(), 'e_available': dict(),
            'soh': dict(), 'degradation': dict()
        }
        for e in model.energy_storages:
            node_id = shared_ess_data.shared_energy_storages[year][e].bus
            s_rated = pe.value(model.es_s_rated[e, y])
            e_rated = pe.value(model.es_e_rated[e, y])
            s_available, e_available = shared_ess_data.get_available_capacities(model, e, y)
            soh = 0.00
            if not isclose(e_available, 0.00, abs_tol=SMALL_TOLERANCE):
                soh = e_available / e_rated
            degradation = 1 - soh
            processed_results[year]['s_rated'][node_id] = s_rated
            processed_results[year]['e_rated'][node_id] = e_rated
            processed_results[year]['s_available'][node_id] = s_available
            processed_results[year]['e_available'][node_id] = e_available
            processed_results[year]['soh'][node_id] = soh
            processed_results[year]['degradation'][node_id] = degradation

    return processed_results


def _process_soh_results_detailed(shared_ess_data, model):

    processed_results = dict()
    repr_years = [year for year in shared_ess_data.years]

    for y_inv in model.years:
        year_inv = repr_years[y_inv]
        processed_results[year_inv] = dict()
        for y_curr in model.years:
            year_curr = repr_years[y_curr]
            processed_results[year_inv][year_curr] = {
                's_rated': dict(), 'e_rated': dict(),
                's_available': dict(), 'e_available': dict(),
                'soh_unit': dict(), 'degradation_unit': dict(),
                'soh_cumul': dict(), 'degradation_cumul': dict()
            }
            for e in model.energy_storages:
                node_id = shared_ess_data.shared_energy_storages[year_curr][e].bus
                s_rated = pe.value(model.es_s_rated_per_unit[e, y_inv, y_curr])
                e_rated = pe.value(model.es_e_rated_per_unit[e, y_inv, y_curr])
                s_available = pe.value(model.es_s_available_per_unit[e, y_inv, y_curr])
                e_available = pe.value(model.es_e_available_per_unit[e, y_inv, y_curr])
                soh_unit = pe.value(model.es_soh_per_unit[e, y_inv, y_curr])
                degradation_unit = pe.value(model.es_degradation_per_unit[e, y_inv, y_curr])
                soh_cumul = pe.value(model.es_soh_per_unit_cumul[e, y_inv, y_curr])
                degradation_cumul = pe.value(model.es_degradation_per_unit_cumul[e, y_inv, y_curr])
                processed_results[year_inv][year_curr]['s_rated'][node_id] = s_rated
                processed_results[year_inv][year_curr]['e_rated'][node_id] = e_rated
                processed_results[year_inv][year_curr]['s_available'][node_id] = s_available
                processed_results[year_inv][year_curr]['e_available'][node_id] = e_available
                processed_results[year_inv][year_curr]['soh_unit'][node_id] = soh_unit
                processed_results[year_inv][year_curr]['degradation_unit'][node_id] = degradation_unit
                processed_results[year_inv][year_curr]['soh_cumul'][node_id] = soh_cumul
                processed_results[year_inv][year_curr]['degradation_cumul'][node_id] = soh_cumul

    return processed_results


def _process_relaxation_variables_operation_detailed(shared_ess_data, model):

    processed_results = dict()
    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    for y_inv in model.years:
        year_inv = repr_years[y_inv]
        processed_results[year_inv] = dict()
        for y_curr in model.years:
            year_curr = repr_years[y_curr]
            processed_results[year_inv][year_curr] = dict()
            for d in model.days:
                day = repr_days[d]
                processed_results[year_inv][year_curr][day] = dict()
                for e in model.energy_storages:
                    node_id = shared_ess_data.shared_energy_storages[year_curr][e].bus
                    processed_results[year_inv][year_curr][day][node_id] = dict()

                    if shared_ess_data.params.slacks:

                        # - Complementarity
                        processed_results[year_inv][year_curr][day][node_id]['comp'] = list()
                        for p in model.periods:
                            comp = pe.value(model.slack_es_comp_per_unit[e, y_inv, y_curr, d, p])
                            processed_results[year_inv][year_curr][day][node_id]['comp'].append(comp)

                        # - Sch, up
                        processed_results[year_inv][year_curr][day][node_id]['sch_up'] = list()
                        for p in model.periods:
                            sch_up = pe.value(model.slack_es_sch_up_per_unit[e, y_inv, y_curr, d, p])
                            processed_results[year_inv][year_curr][day][node_id]['sch_up'].append(sch_up)

                        # - Sch, down
                        processed_results[year_inv][year_curr][day][node_id]['sch_down'] = list()
                        for p in model.periods:
                            sch_down = pe.value(model.slack_es_sch_down_per_unit[e, y_inv, y_curr, d, p])
                            processed_results[year_inv][year_curr][day][node_id]['sch_down'].append(sch_up)

                        # - Sdch, up
                        processed_results[year_inv][year_curr][day][node_id]['sdch_up'] = list()
                        for p in model.periods:
                            sdch_up = pe.value(model.slack_es_sdch_up_per_unit[e, y_inv, y_curr, d, p])
                            processed_results[year_inv][year_curr][day][node_id]['sdch_up'].append(sdch_up)

                        # - Sdch, down
                        processed_results[year_inv][year_curr][day][node_id]['sdch_down'] = list()
                        for p in model.periods:
                            sdch_down = pe.value(model.slack_es_sdch_down_per_unit[e, y_inv, y_curr, d, p])
                            processed_results[year_inv][year_curr][day][node_id]['sdch_down'].append(sdch_down)

    return processed_results


def _get_investment_and_available_capacities(shared_ess_data, model):

    years = [year for year in shared_ess_data.years]
    ess_capacity = {'investment': dict(), 'rated': dict(), 'available': dict()}

    # - Investment in Power and Energy Capacity (per year)
    # - Power and Energy capacities available (per representative day)
    for e in model.energy_storages:

        node_id = shared_ess_data.shared_energy_storages[years[0]][e].bus
        ess_capacity['investment'][node_id] = dict()
        ess_capacity['rated'][node_id] = dict()
        ess_capacity['available'][node_id] = dict()

        for y in model.years:

            year = years[y]

            ess_capacity['investment'][node_id][year] = dict()
            ess_capacity['investment'][node_id][year]['power'] = pe.value(model.es_s_investment[e, y])
            ess_capacity['investment'][node_id][year]['energy'] = pe.value(model.es_e_investment[e, y])

            ess_capacity['rated'][node_id][year] = dict()
            ess_capacity['rated'][node_id][year]['power'] = pe.value(model.es_s_rated[e, y])
            ess_capacity['rated'][node_id][year]['energy'] = pe.value(model.es_e_rated[e, y])

            s_available, e_available = shared_ess_data.get_available_capacities(model, e, y)
            soh = 0.00
            if not isclose(e_available, 0.00, abs_tol=SMALL_TOLERANCE):
                soh = e_available / pe.value(model.es_e_rated[e, y])
            ess_capacity['available'][node_id][year] = dict()
            ess_capacity['available'][node_id][year]['power'] = s_available
            ess_capacity['available'][node_id][year]['energy'] = e_available
            ess_capacity['available'][node_id][year]['soh'] = soh
            ess_capacity['available'][node_id][year]['degradation_factor'] = 1 - soh

    return ess_capacity


# ======================================================================================================================
#   Shared ESS -- Write Results
# ======================================================================================================================
def _write_optimization_results_to_excel(shared_ess_data, data_dir, results):

    wb = Workbook()

    _write_ess_capacity_investment_to_excel(shared_ess_data, wb, results['capacity']['investment'])
    _write_ess_capacity_rated_available_to_excel(shared_ess_data, wb, results['capacity'])
    _write_aggregated_shared_energy_storage_operation_results_to_excel(shared_ess_data, wb, results['operation']['aggregated'])
    _write_detailed_shared_energy_storage_operation_results_to_excel(shared_ess_data, wb, results['operation']['detailed'])
    _write_aggregated_shared_energy_storage_soh_results_to_excel(shared_ess_data, wb, results['soh']['aggregated'])
    _write_detailed_shared_energy_storage_soh_results_to_excel(shared_ess_data, wb, results['soh']['detailed'])
    _write_detailed_operation_relaxation_slacks_results_to_excel(shared_ess_data, wb, results['relaxation_variables']['operation']['detailed'])

    results_filename = os.path.join(data_dir, f'{shared_ess_data.name}_shared_ess_results.xlsx')
    try:
        wb.save(results_filename)
        print('[INFO] S-MPOPF Results written to {}.'.format(results_filename))
    except:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = os.path.join(data_dir, f'{shared_ess_data.name}_shared_ess_results_{current_time}.xlsx')
        print('[INFO] S-MPOPF Results written to {}.'.format(backup_filename))
        wb.save(backup_filename)


def _write_ess_capacity_investment_to_excel(shared_ess_data, workbook, results, initial_sheet=True):

    if initial_sheet:
        sheet = workbook.worksheets[0]
        sheet.title = 'Capacity Investment'
    else:
        sheet = workbook.create_sheet('Capacity Investment')

    years = [year for year in shared_ess_data.years]

    num_style = '0.00'

    # Write Header
    line_idx = 1
    sheet.cell(row=line_idx, column=1).value = 'Node'
    sheet.cell(row=line_idx, column=2).value = 'Quantity'
    for y in range(len(years)):
        year = years[y]
        sheet.cell(row=line_idx, column=y + 3).value = int(year)

    # Write investment values, power and energy
    for node_id in results:

        # Power capacity
        line_idx = line_idx + 1
        sheet.cell(row=line_idx, column=1).value = node_id
        sheet.cell(row=line_idx, column=2).value = 'S, [MVA]'
        for y in range(len(years)):
            year = years[y]
            sheet.cell(row=line_idx, column=y + 3).value = results[node_id][year]['power']
            sheet.cell(row=line_idx, column=y + 3).number_format = num_style

        # Energy capacity
        line_idx = line_idx + 1
        sheet.cell(row=line_idx, column=1).value = node_id
        sheet.cell(row=line_idx, column=2).value = 'E, [MVAh]'
        for y in range(len(years)):
            year = years[y]
            sheet.cell(row=line_idx, column=y + 3).value = results[node_id][year]['energy']
            sheet.cell(row=line_idx, column=y + 3).number_format = num_style

        # Power capacity cost
        line_idx = line_idx + 1
        sheet.cell(row=line_idx, column=1).value = node_id
        sheet.cell(row=line_idx, column=2).value = 'Cost S, [m.u.]'
        for y in range(len(years)):
            year = years[y]
            cost_s = shared_ess_data.cost_investment['power_capacity'][year] * results[node_id][year]['power']
            sheet.cell(row=line_idx, column=y + 3).value = cost_s
            sheet.cell(row=line_idx, column=y + 3).number_format = num_style

        # Energy capacity cost
        line_idx = line_idx + 1
        sheet.cell(row=line_idx, column=1).value = node_id
        sheet.cell(row=line_idx, column=2).value = 'Cost E, [m.u.]'
        for y in range(len(years)):
            year = years[y]
            cost_e = shared_ess_data.cost_investment['energy_capacity'][year] * results[node_id][year]['energy']
            sheet.cell(row=line_idx, column=y + 3).value = cost_e
            sheet.cell(row=line_idx, column=y + 3).number_format = num_style

        # Total capacity cost
        line_idx = line_idx + 1
        sheet.cell(row=line_idx, column=1).value = node_id
        sheet.cell(row=line_idx, column=2).value = 'Cost Total, [m.u.]'
        for y in range(len(years)):
            year = years[y]
            cost_s = shared_ess_data.cost_investment['power_capacity'][year] * results[node_id][year]['power']
            cost_e = shared_ess_data.cost_investment['energy_capacity'][year] * results[node_id][year]['energy']
            sheet.cell(row=line_idx, column=y + 3).value = cost_s + cost_e
            sheet.cell(row=line_idx, column=y + 3).number_format = num_style


def _write_ess_capacity_rated_available_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Capacity Available')

    num_style = '0.00'
    perc_style = '0.00%'

    # Write Header
    row_idx, col_idx = 1, 1
    sheet.cell(row=row_idx, column=col_idx).value = 'Node'
    col_idx = col_idx + 1
    sheet.cell(row=row_idx, column=col_idx).value = 'Quantity'
    col_idx = col_idx + 1
    for year in shared_ess_data.years:
        sheet.cell(row=row_idx, column=col_idx).value = int(year)
        col_idx = col_idx + 1

    # Write investment values, power and energy
    for node_id in results['rated']:

        # Power, rated
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'Srated, [MVA]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results['rated'][node_id][year]['power']
            sheet.cell(row=row_idx, column=col_idx).number_format = num_style
            col_idx = col_idx + 1

        # Capacity, rated
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'Erated, [MVA]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results['rated'][node_id][year]['energy']
            sheet.cell(row=row_idx, column=col_idx).number_format = num_style
            col_idx = col_idx + 1

        # Power, available
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'Savailable, [MVA]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results['available'][node_id][year]['power']
            sheet.cell(row=row_idx, column=col_idx).number_format = num_style
            col_idx = col_idx + 1

        # Capacity, available
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'Eavailable, [MVAh]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results['available'][node_id][year]['energy']
            sheet.cell(row=row_idx, column=col_idx).number_format = num_style
            col_idx = col_idx + 1

        # SoH
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'SoH, [%]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results['available'][node_id][year]['soh']
            sheet.cell(row=row_idx, column=col_idx).number_format = perc_style
            col_idx = col_idx + 1

        # Degradation factor
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'Degradation factor, [%]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results['available'][node_id][year]['degradation_factor']
            sheet.cell(row=row_idx, column=col_idx).number_format = perc_style
            col_idx = col_idx + 1


def _write_aggregated_shared_energy_storage_operation_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Operation, aggregated')

    row_idx = 1
    decimal_style = '0.00'
    perc_style = '0.00%'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year'
    sheet.cell(row=row_idx, column=3).value = 'Day'
    sheet.cell(row=row_idx, column=4).value = 'Quantity'
    for p in range(shared_ess_data.num_instants):
        sheet.cell(row=row_idx, column=p + 5).value = p
    row_idx = row_idx + 1

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for year in results:
            for day in results[year]:

                # - Apparent Power
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year)
                sheet.cell(row=row_idx, column=3).value = day
                sheet.cell(row=row_idx, column=4).value = 'S, [MVA]'
                for p in range(shared_ess_data.num_instants):
                    pnet = results[year][day][node_id]['s'][p]
                    sheet.cell(row=row_idx, column=p + 5).value = pnet
                    sheet.cell(row=row_idx, column=p + 5).number_format = decimal_style
                row_idx = row_idx + 1

                # - SoC, [MWh]
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year)
                sheet.cell(row=row_idx, column=3).value = day
                sheet.cell(row=row_idx, column=4).value = 'SoC, [MWh]'
                for p in range(shared_ess_data.num_instants):
                    soc = results[year][day][node_id]['soc'][p]
                    sheet.cell(row=row_idx, column=p + 5).value = soc
                    sheet.cell(row=row_idx, column=p + 5).number_format = decimal_style
                row_idx = row_idx + 1

                # - SoC, [%]
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year)
                sheet.cell(row=row_idx, column=3).value = day
                sheet.cell(row=row_idx, column=4).value = 'SoC, [%]'
                for p in range(shared_ess_data.num_instants):
                    soc_perc = results[year][day][node_id]['soc_perc'][p]
                    sheet.cell(row=row_idx, column=p + 5).value = soc_perc
                    sheet.cell(row=row_idx, column=p + 5).number_format = perc_style
                row_idx = row_idx + 1


def _write_detailed_shared_energy_storage_operation_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Operation, Detailed')

    row_idx = 1
    perc_style = '0.00%'
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year Investment'
    sheet.cell(row=row_idx, column=3).value = 'Year Current'
    sheet.cell(row=row_idx, column=4).value = 'Day'
    sheet.cell(row=row_idx, column=5).value = 'Quantity'
    for p in range(shared_ess_data.num_instants):
        sheet.cell(row=row_idx, column=p + 6).value = p
    row_idx = row_idx + 1

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for year_inv in results:
            for year_curr in results[year_inv]:
                for day in results[year_inv][year_curr]:

                    # - Apparent Power
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'S, [MVA]'
                    for p in range(shared_ess_data.num_instants):
                        pnet = results[year_inv][year_curr][day][node_id]['s'][p]
                        sheet.cell(row=row_idx, column=p + 6).value = pnet
                        sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - SoC, [MWh]
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'SoC, [MWh]'
                    for p in range(shared_ess_data.num_instants):
                        soc = results[year_inv][year_curr][day][node_id]['soc'][p]
                        sheet.cell(row=row_idx, column=p + 6).value = soc
                        sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - SoC, [%]
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'SoC, [%]'
                    for p in range(shared_ess_data.num_instants):
                        soc_perc = results[year_inv][year_curr][day][node_id]['soc_perc'][p]
                        sheet.cell(row=row_idx, column=p + 6).value = soc_perc
                        sheet.cell(row=row_idx, column=p + 6).number_format = perc_style
                    row_idx = row_idx + 1


def _write_aggregated_shared_energy_storage_soh_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Degradation, aggregated')

    row_idx = 1
    perc_style = '0.00%'
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year'
    sheet.cell(row=row_idx, column=3).value = 'Quantity'
    sheet.cell(row=row_idx, column=4).value = 'Value'
    row_idx = row_idx + 1

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for year in results:

            s_rated = results[year]['s_rated'][node_id]
            e_rated = results[year]['e_rated'][node_id]
            s_available = results[year]['s_available'][node_id]
            e_available = results[year]['e_available'][node_id]
            soh = results[year]['soh'][node_id]
            degradation = results[year]['degradation'][node_id]

            # - Srated
            sheet.cell(row=row_idx, column=1).value = node_id
            sheet.cell(row=row_idx, column=2).value = int(year)
            sheet.cell(row=row_idx, column=3).value = 'Srated, [MVA]'
            sheet.cell(row=row_idx, column=4).value = s_rated
            sheet.cell(row=row_idx, column=4).number_format = decimal_style
            row_idx = row_idx + 1

            # - Erated
            sheet.cell(row=row_idx, column=1).value = node_id
            sheet.cell(row=row_idx, column=2).value = int(year)
            sheet.cell(row=row_idx, column=3).value = 'Erated, [MVA]'
            sheet.cell(row=row_idx, column=4).value = e_rated
            sheet.cell(row=row_idx, column=4).number_format = decimal_style
            row_idx = row_idx + 1

            # - Savailable
            sheet.cell(row=row_idx, column=1).value = node_id
            sheet.cell(row=row_idx, column=2).value = int(year)
            sheet.cell(row=row_idx, column=3).value = 'Savailable, [MVA]'
            sheet.cell(row=row_idx, column=4).value = s_available
            sheet.cell(row=row_idx, column=4).number_format = decimal_style
            row_idx = row_idx + 1

            # - Eavailable
            sheet.cell(row=row_idx, column=1).value = node_id
            sheet.cell(row=row_idx, column=2).value = int(year)
            sheet.cell(row=row_idx, column=3).value = 'Eavailable, [MVA]'
            sheet.cell(row=row_idx, column=4).value = e_available
            sheet.cell(row=row_idx, column=4).number_format = decimal_style
            row_idx = row_idx + 1

            # - SoH
            sheet.cell(row=row_idx, column=1).value = node_id
            sheet.cell(row=row_idx, column=2).value = int(year)
            sheet.cell(row=row_idx, column=3).value = 'SoH, [%]'
            sheet.cell(row=row_idx, column=4).value = soh
            sheet.cell(row=row_idx, column=4).number_format = perc_style
            row_idx = row_idx + 1

            # - Degradation
            sheet.cell(row=row_idx, column=1).value = node_id
            sheet.cell(row=row_idx, column=2).value = int(year)
            sheet.cell(row=row_idx, column=3).value = 'Degradation, [%]'
            sheet.cell(row=row_idx, column=4).value = degradation
            sheet.cell(row=row_idx, column=4).number_format = perc_style
            row_idx = row_idx + 1


def _write_detailed_shared_energy_storage_soh_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Degradation, detailed')

    row_idx = 1
    perc_style = '0.00%'
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year Investment'
    sheet.cell(row=row_idx, column=3).value = 'Year Current'
    sheet.cell(row=row_idx, column=4).value = 'Quantity'
    sheet.cell(row=row_idx, column=5).value = 'Value'
    row_idx = row_idx + 1

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for year_inv in results:
            for year_curr in results[year_inv]:

                s_rated = results[year_inv][year_curr]['s_rated'][node_id]
                e_rated = results[year_inv][year_curr]['e_rated'][node_id]
                s_available = results[year_inv][year_curr]['s_available'][node_id]
                e_available = results[year_inv][year_curr]['e_available'][node_id]
                soh_unit = results[year_inv][year_curr]['soh_unit'][node_id]
                degradation_unit = results[year_inv][year_curr]['degradation_unit'][node_id]
                soh_cumul = results[year_inv][year_curr]['soh_cumul'][node_id]
                degradation_cumul = results[year_inv][year_curr]['degradation_cumul'][node_id]

                # - Srated, average day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'Srated, [MVA]'
                sheet.cell(row=row_idx, column=5).value = s_rated
                sheet.cell(row=row_idx, column=5).number_format = decimal_style
                row_idx = row_idx + 1

                # - Erated, average day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'Erated, [MVAh]'
                sheet.cell(row=row_idx, column=5).value = e_rated
                sheet.cell(row=row_idx, column=5).number_format = decimal_style
                row_idx = row_idx + 1

                # - Savailable, average day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'Savailable, [MVA]'
                sheet.cell(row=row_idx, column=5).value = s_available
                sheet.cell(row=row_idx, column=5).number_format = decimal_style
                row_idx = row_idx + 1

                # - Eavailable, average day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'Eavailable, [MVAh]'
                sheet.cell(row=row_idx, column=5).value = e_available
                sheet.cell(row=row_idx, column=5).number_format = decimal_style
                row_idx = row_idx + 1

                # - SoH, average day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'SoH day, [%]'
                sheet.cell(row=row_idx, column=5).value = soh_unit
                sheet.cell(row=row_idx, column=5).number_format = perc_style
                row_idx = row_idx + 1

                # - Degradation, average day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'Degradation day, [%]'
                sheet.cell(row=row_idx, column=5).value = degradation_unit
                sheet.cell(row=row_idx, column=5).number_format = perc_style
                row_idx = row_idx + 1

                # - SoH, cumulative
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'SoH cumul., [%]'
                sheet.cell(row=row_idx, column=5).value = soh_cumul
                sheet.cell(row=row_idx, column=5).number_format = perc_style
                row_idx = row_idx + 1

                # - Degradation, cumulative
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'Degradation cumul., [%]'
                sheet.cell(row=row_idx, column=5).value = degradation_cumul
                sheet.cell(row=row_idx, column=5).number_format = perc_style
                row_idx = row_idx + 1


def _write_detailed_operation_relaxation_slacks_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Slacks operation, detailed')

    row_idx = 1
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year Investment'
    sheet.cell(row=row_idx, column=3).value = 'Year Current'
    sheet.cell(row=row_idx, column=4).value = 'Day'
    sheet.cell(row=row_idx, column=5).value = 'Quantity'
    for p in range(shared_ess_data.num_instants):
        sheet.cell(row=row_idx, column=p + 6).value = p
    row_idx = row_idx + 1

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for year_inv in results:
            for year_curr in results[year_inv]:
                for day in results[year_inv][year_curr]:

                    if shared_ess_data.params.slacks:

                        # - Complementarity
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year_inv)
                        sheet.cell(row=row_idx, column=3).value = int(year_curr)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Complementary'
                        for p in range(shared_ess_data.num_instants):
                            comp = results[year_inv][year_curr][day][node_id]['comp'][p]
                            sheet.cell(row=row_idx, column=p + 6).value = comp
                            sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - Sch, up
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year_inv)
                        sheet.cell(row=row_idx, column=3).value = int(year_curr)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Sch, up'
                        for p in range(shared_ess_data.num_instants):
                            sch_up = results[year_inv][year_curr][day][node_id]['sch_up'][p]
                            sheet.cell(row=row_idx, column=p + 6).value = sch_up
                            sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - Sch, down
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year_inv)
                        sheet.cell(row=row_idx, column=3).value = int(year_curr)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Sch, down'
                        for p in range(shared_ess_data.num_instants):
                            sch_down = results[year_inv][year_curr][day][node_id]['sch_down'][p]
                            sheet.cell(row=row_idx, column=p + 6).value = sch_down
                            sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - Sdch, up
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year_inv)
                        sheet.cell(row=row_idx, column=3).value = int(year_curr)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Sdch, up'
                        for p in range(shared_ess_data.num_instants):
                            sdch_up = results[year_inv][year_curr][day][node_id]['sdch_up'][p]
                            sheet.cell(row=row_idx, column=p + 6).value = sdch_up
                            sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - Sdch, down
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year_inv)
                        sheet.cell(row=row_idx, column=3).value = int(year_curr)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Sdch, down'
                        for p in range(shared_ess_data.num_instants):
                            sdch_down = results[year_inv][year_curr][day][node_id]['sdch_down'][p]
                            sheet.cell(row=row_idx, column=p + 6).value = sdch_down
                            sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                        row_idx = row_idx + 1

    return results
