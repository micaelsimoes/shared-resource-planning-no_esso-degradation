import os
from math import isclose
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
        return _process_results(self, model)

    def write_optimization_results_to_excel(self, model):
        shared_ess_capacity = self.get_investment_and_available_capacities(model)
        processed_results = self.process_results(model)
        _write_optimization_results_to_excel(self, self.results_dir, processed_results, shared_ess_capacity)

    def update_data_with_candidate_solution(self, candidate_solution):
        for year in self.years:
            for shared_ess in self.shared_energy_storages[year]:
                shared_ess.s = candidate_solution[shared_ess.bus][year]['s']
                shared_ess.e = candidate_solution[shared_ess.bus][year]['e']
                shared_ess.e_init = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_RELATIVE_INIT_SOC
                shared_ess.e_min = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MIN_ENERGY_STORED
                shared_ess.e_max = candidate_solution[shared_ess.bus][year]['e'] * ENERGY_STORAGE_MAX_ENERGY_STORED

    def get_investment_and_available_capacities(self, model):
        return _get_investment_and_available_capacities(self, model)


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
    model.es_s_investment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_investment = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_rated_total = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_rated_total = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_s_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_s_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_e_up = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.slack_e_down = pe.Var(model.energy_storages, model.years, domain=pe.NonNegativeReals, initialize=0.0)

    model.es_s_rated_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_e_rated_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_soc_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_pch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_pdch_per_unit = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_degradation_per_unit = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.00)
    model.es_avg_ch_dch_day = pe.Var(model.energy_storages, model.years, model.years, domain=pe.NonNegativeReals, initialize=0.00)
    if shared_ess_data.params.ess_relax_comp:
        model.es_penalty_comp = pe.Var(model.energy_storages, model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.es_s_rated_per_unit.fix(0.00)
    model.es_e_rated_per_unit.fix(0.00)
    model.es_avg_ch_dch_day.fix(0.00)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Yearly Power and Energy ratings as a function of yearly investments
    model.rated_s_capacity = pe.ConstraintList()
    model.rated_e_capacity = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            total_s_capacity = 0.00
            total_e_capacity = 0.00
            for y_inv in model.years:
                total_s_capacity += model.es_s_rated_per_unit[e, y_inv, y]
                total_e_capacity += model.es_e_rated_per_unit[e, y_inv, y]
            model.rated_s_capacity.add(model.es_s_rated_total[e, y] == total_s_capacity)
            model.rated_e_capacity.add(model.es_e_rated_total[e, y] == total_e_capacity)

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
                model.es_avg_ch_dch_day[e, y_inv, y].fixed = False
                model.rated_s_capacity_unit.add(model.es_s_rated_per_unit[e, y_inv, y] == model.es_s_investment[e, y_inv])
                model.rated_e_capacity_unit.add(model.es_e_rated_per_unit[e, y_inv, y] == model.es_e_investment[e, y_inv])

    # - P, Q, S, SoC, per unit as a function of available capacities
    model.energy_storage_limits = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:
            for y in model.years:
                for d in model.days:
                    for p in model.periods:
                        model.energy_storage_limits.add(model.es_pch_per_unit[e, y_inv, y, d, p] <= model.es_s_rated_per_unit[e, y_inv, y])
                        model.energy_storage_limits.add(model.es_pdch_per_unit[e, y_inv, y, d, p] <= model.es_s_rated_per_unit[e, y_inv, y])
                        model.energy_storage_limits.add(model.es_soc_per_unit[e, y_inv, y, d, p] >= model.es_e_rated_per_unit[e, y_inv, y] * ENERGY_STORAGE_MIN_ENERGY_STORED)
                        model.energy_storage_limits.add(model.es_soc_per_unit[e, y_inv, y, d, p] <= model.es_e_rated_per_unit[e, y_inv, y] * ENERGY_STORAGE_MAX_ENERGY_STORED)

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
                        pch = model.es_pch_per_unit[e, y_inv, y, d, p]
                        pdch = model.es_pdch_per_unit[e, y_inv, y, d, p]
                        avg_ch_dch += (num_days / 365.00) * (pch + pdch)
                model.energy_storage_charging_discharging.add(model.es_avg_ch_dch_day[e, y_inv, y] == avg_ch_dch)

    # - Capacity degradation
    '''
    model.energy_storage_capacity_degradation = pe.ConstraintList()
    for e in model.energy_storages:
        for y in model.years:
            year = repr_years[y]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year][e]
            tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[year]))
            min_tcal_norm = max(y - tcal_norm + 1, 0)
            eq_cycle_number = shared_energy_storage.cl_nom
            for y_inv in range(y, min_tcal_norm - 1, -1):
                year_inv = repr_years[y_inv]
                num_years = shared_ess_data.years[year_inv]
                rated_capacity = model.es_e_rated_per_unit[e, y_inv, y]
                avg_ch_dch_day = model.es_avg_ch_dch_day[e, y_inv, y]
                previous_year_deg = 0.00
                if y_inv > 0:
                    previous_year_deg = model.es_degradation_per_unit[e, y_inv - 1, y]
                model.energy_storage_capacity_degradation.add((model.es_degradation_per_unit[e, y_inv, y] - previous_year_deg) * (rated_capacity * 2 * eq_cycle_number) == (avg_ch_dch_day))
    '''

    # - Shared ESS operation
    model.energy_storage_operation = pe.ConstraintList()
    model.energy_storage_balance = pe.ConstraintList()
    model.energy_storage_day_balance = pe.ConstraintList()
    model.energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.energy_storage_expected_power = pe.ConstraintList()
    model.secondary_reserve = pe.ConstraintList()
    for e in model.energy_storages:
        for y_inv in model.years:

            year_inv = repr_years[y_inv]
            shared_energy_storage = shared_ess_data.shared_energy_storages[year_inv][e]
            eff_charge, eff_discharge = shared_energy_storage.eff_ch, shared_energy_storage.eff_dch

            for y in model.years:

                soc_init = model.es_e_rated_per_unit[e, y_inv, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC
                soc_final = model.es_e_rated_per_unit[e, y_inv, y] * ENERGY_STORAGE_RELATIVE_INIT_SOC

                for d in model.days:
                    for p in model.periods:

                        pch = model.es_pch_per_unit[e, y_inv, y, d, p]
                        pdch = model.es_pdch_per_unit[e, y_inv, y, d, p]

                        if p > 0:
                            model.energy_storage_balance.add(model.es_soc_per_unit[e, y_inv, y, d, p] - model.es_soc_per_unit[e, y_inv, y, d, p - 1] == pch * eff_charge - pdch / eff_discharge)
                        else:
                            model.energy_storage_balance.add(model.es_soc_per_unit[e, y_inv, y, d, p] - soc_init == pch * eff_charge - pdch / eff_discharge)

                        # Charging/discharging complementarity constraint
                        if shared_ess_data.params.ess_relax_comp:
                            model.energy_storage_ch_dch_exclusion.add(pch * pdch <= model.es_penalty_comp[e, y_inv, y, d, p])
                        else:
                            model.energy_storage_ch_dch_exclusion.add(pch * pdch >= -SMALL_TOLERANCE)
                            model.energy_storage_ch_dch_exclusion.add(pch * pdch <= SMALL_TOLERANCE)

                    model.energy_storage_day_balance.add(model.es_soc_per_unit[e, y_inv, y, d, len(model.periods) - 1] == soc_final)

    # ------------------------------------------------------------------------------------------------------------------
    # Objective function
    slack_penalty = 0.0
    for e in model.energy_storages:
        for y_inv in model.years:

            # Slack penalties
            slack_penalty += PENALTY_ESS_SLACK_FEASIBILITY * (model.slack_s_up[e, y_inv] + model.slack_s_down[e, y_inv])
            slack_penalty += PENALTY_ESS_SLACK_FEASIBILITY * (model.slack_e_up[e, y_inv] + model.slack_e_down[e, y_inv])

            if shared_ess_data.params.ess_relax_comp:
                for y in model.years:
                    for d in model.days:
                        for p in model.periods:
                            slack_penalty += PENALTY_ESS_COMPLEMENTARITY * model.es_penalty_comp[e, y_inv, y, d, p]

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
            model.es_s_investment[e, y].fix(candidate_solution[node_id][year]['s'])
            model.es_e_investment[e, y].fix(candidate_solution[node_id][year]['e'])


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
def _process_results(shared_ess_data, model):

    processed_results = {
        'of_value': 0.00,
        'results': dict()
    }

    repr_days = [day for day in shared_ess_data.days]
    repr_years = [year for year in shared_ess_data.years]

    for y_inv in model.years:
        year_inv = repr_years[y_inv]
        processed_results['results'][year_inv] = dict()
        processed_results['results'][year_inv] = dict()
        for y_curr in model.years:
            year_curr = repr_years[y_curr]
            processed_results['results'][year_inv][year_curr] = dict()
            for d in model.days:
                day = repr_days[d]
                processed_results['results'][year_inv][year_curr][day] = {'p': dict(), 'soc': dict(), 'soc_percent': dict(),
                                                                          'degradation': dict(), 'avg_ch_dch': dict(),
                                                                          'pch': dict(), 'pdch': dict(), 'comp': dict()}
                for e in model.energy_storages:
                    node_id = shared_ess_data.shared_energy_storages[year_curr][e].bus
                    capacity = pe.value(model.es_e_rated_per_unit[e, y_inv, y_curr])
                    if isclose(capacity, 0.0, abs_tol=SMALL_TOLERANCE):
                        capacity = 1.00
                    processed_results['results'][year_inv][year_curr][day]['p'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['pch'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['pdch'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['comp'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['soc'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['soc_percent'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['degradation'][node_id] = []
                    processed_results['results'][year_inv][year_curr][day]['avg_ch_dch'][node_id] = []
                    for p in model.periods:
                        pch = pe.value(model.es_pch_per_unit[e, y_inv, y_curr, d, p])
                        pdch = pe.value(model.es_pdch_per_unit[e, y_inv, y_curr, d, p])
                        comp = pe.value(model.es_penalty_comp[e, y_inv, y_curr, d, p])
                        p_net = pe.value(model.es_pch_per_unit[e, y_inv, y_curr, d, p] - model.es_pdch_per_unit[e, y_inv, y_curr, d, p])
                        soc = pe.value(model.es_soc_per_unit[e, y_inv, y_curr, d, p])
                        soc_perc = soc / capacity
                        degradation = pe.value(model.es_degradation_per_unit[e, y_inv, y_curr])
                        avg_ch_dch = pe.value(model.es_avg_ch_dch_day[e, y_inv, y_curr])
                        processed_results['results'][year_inv][year_curr][day]['p'][node_id].append(p_net)
                        processed_results['results'][year_inv][year_curr][day]['pch'][node_id].append(pch)
                        processed_results['results'][year_inv][year_curr][day]['pdch'][node_id].append(pdch)
                        processed_results['results'][year_inv][year_curr][day]['comp'][node_id].append(comp)
                        processed_results['results'][year_inv][year_curr][day]['soc'][node_id].append(soc)
                        processed_results['results'][year_inv][year_curr][day]['soc_percent'][node_id].append(soc_perc)
                        processed_results['results'][year_inv][year_curr][day]['degradation'][node_id].append(degradation)
                        processed_results['results'][year_inv][year_curr][day]['avg_ch_dch'][node_id].append(avg_ch_dch)

    return processed_results


def _get_investment_and_available_capacities(shared_ess_data, model):

    years = [year for year in shared_ess_data.years]
    ess_capacity = {'investment': dict(), 'available': dict()}

    # - Investment in Power and Energy Capacity (per year)
    # - Power and Energy capacities available (per representative day)
    for e in model.energy_storages:

        node_id = shared_ess_data.shared_energy_storages[years[0]][e].bus
        ess_capacity['investment'][node_id] = dict()
        ess_capacity['available'][node_id] = dict()

        for y in model.years:

            year = years[y]

            ess_capacity['investment'][node_id][year] = dict()
            ess_capacity['investment'][node_id][year]['power'] = pe.value(model.es_s_investment[e, y])
            ess_capacity['investment'][node_id][year]['energy'] = pe.value(model.es_e_investment[e, y])

            ess_capacity['available'][node_id][year] = dict()
            ess_capacity['available'][node_id][year]['power'] = pe.value(model.es_s_rated_total[e, y])
            ess_capacity['available'][node_id][year]['energy'] = pe.value(model.es_e_rated_total[e, y]) # Change later!
            ess_capacity['available'][node_id][year]['degradation_factor'] = 0.00

    return ess_capacity


# ======================================================================================================================
#   Shared ESS -- Write Results
# ======================================================================================================================
def _write_optimization_results_to_excel(shared_ess_data, data_dir, results, shared_ess_capacity):

    wb = Workbook()

    _write_main_info_to_excel(shared_ess_data, wb, results)
    _write_ess_capacity_investment_to_excel(shared_ess_data, wb, shared_ess_capacity['investment'])
    _write_ess_capacity_available_to_excel(shared_ess_data, wb, shared_ess_capacity['available'])
    _write_shared_energy_storage_results_to_excel(shared_ess_data, wb, results['results'])
    #_write_relaxation_slacks_results_to_excel(shared_ess_data, wb, results['results'])
    #if shared_ess_data.params.ess_relax_capacity_relative:
        #_write_relaxation_slacks_yoy_results_to_excel(shared_ess_data, wb, results)

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


def _write_main_info_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.worksheets[0]
    sheet.title = 'Main Info'

    decimal_style = '0.00'
    line_idx = 1

    # Write Header
    col_idx = 2
    for year in shared_ess_data.years:
        for _ in shared_ess_data.days:
            sheet.cell(row=line_idx, column=col_idx).value = year
            col_idx += 1
    col_idx = 2
    line_idx += 1
    for _ in shared_ess_data.years:
        for day in shared_ess_data.days:
            sheet.cell(row=line_idx, column=col_idx).value = day
            col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Total'


def _write_ess_capacity_investment_to_excel(shared_ess_data, workbook, results):

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


def _write_ess_capacity_available_to_excel(shared_ess_data, workbook, results):

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
    for node_id in results:

        # Power capacity
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'S, [MVA]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results[node_id][year]['power']
            sheet.cell(row=row_idx, column=col_idx).number_format = num_style
            col_idx = col_idx + 1

        # Energy capacity
        col_idx = 1
        row_idx = row_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = node_id
        col_idx = col_idx + 1
        sheet.cell(row=row_idx, column=col_idx).value = 'E, [MVAh]'
        col_idx = col_idx + 1
        for year in shared_ess_data.years:
            sheet.cell(row=row_idx, column=col_idx).value = results[node_id][year]['energy']
            sheet.cell(row=row_idx, column=col_idx).number_format = num_style
            col_idx = col_idx + 1

        # Degradation factor
        if "degradation_factor" in results[node_id][year]:
            col_idx = 1
            row_idx = row_idx + 1
            sheet.cell(row=row_idx, column=col_idx).value = node_id
            col_idx = col_idx + 1
            sheet.cell(row=row_idx, column=col_idx).value = 'Degradation factor'
            col_idx = col_idx + 1
            for year in shared_ess_data.years:
                sheet.cell(row=row_idx, column=col_idx).value = results[node_id][year]['degradation_factor']
                sheet.cell(row=row_idx, column=col_idx).number_format = perc_style
                col_idx = col_idx + 1


def _write_shared_energy_storage_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Shared Energy Storage')
    repr_days = [day for day in shared_ess_data.days]

    row_idx = 1
    decimal_style = '0.00'
    perc_style = '0.00%'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year Investment'
    sheet.cell(row=row_idx, column=3).value = 'Year Current'
    sheet.cell(row=row_idx, column=4).value = 'Day'
    sheet.cell(row=row_idx, column=5).value = 'Quantity'
    for p in range(shared_ess_data.num_instants):
        sheet.cell(row=row_idx, column=p + 6).value = p
    row_idx = row_idx + 1

    for year_inv in results:
        for year_curr in results[year_inv]:
            for day in results[year_inv][year_curr]:

                for node_id in results[year_inv][year_curr][day]['p']:

                    # - Active Power
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                    for p in range(shared_ess_data.num_instants):
                        pc = results[year_inv][year_curr][day]['p'][node_id][p]
                        sheet.cell(row=row_idx, column=p + 6).value = pc
                        sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - Active Power, charging
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'Pch, [MW]'
                    for p in range(shared_ess_data.num_instants):
                        pch = results[year_inv][year_curr][day]['pch'][node_id][p]
                        sheet.cell(row=row_idx, column=p + 6).value = pch
                        sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - Active Power, discharging
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'Pdch, [MW]'
                    for p in range(shared_ess_data.num_instants):
                        pdch = results[year_inv][year_curr][day]['pdch'][node_id][p]
                        sheet.cell(row=row_idx, column=p + 6).value = pdch
                        sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - Charging complementarity penalty
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'Comp'
                    for p in range(shared_ess_data.num_instants):
                        comp = results[year_inv][year_curr][day]['comp'][node_id][p]
                        sheet.cell(row=row_idx, column=p + 6).value = comp
                        sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - SoC, [MWh]
                    sheet.cell(row=row_idx, column=1).value = node_id
                    sheet.cell(row=row_idx, column=2).value = int(year_inv)
                    sheet.cell(row=row_idx, column=3).value = int(year_curr)
                    sheet.cell(row=row_idx, column=4).value = day
                    sheet.cell(row=row_idx, column=5).value = 'SoC, [MWh]'
                    for p in range(shared_ess_data.num_instants):
                        soc = results[year_inv][year_curr][day]['soc'][node_id][p]
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
                        soc_perc = results[year_inv][year_curr][day]['soc_percent'][node_id][p]
                        sheet.cell(row=row_idx, column=p + 6).value = soc_perc
                        sheet.cell(row=row_idx, column=p + 6).number_format = perc_style
                    row_idx = row_idx + 1

            for node_id in results[year_inv][year_curr][repr_days[0]]['degradation']:

                # - Degradation
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'N/A'
                sheet.cell(row=row_idx, column=5).value = 'Degradation, [%]'
                for p in range(shared_ess_data.num_instants):
                    degradation = results[year_inv][year_curr][repr_days[0]]['degradation'][node_id][p]
                    sheet.cell(row=row_idx, column=p + 6).value = degradation
                    sheet.cell(row=row_idx, column=p + 6).number_format = perc_style
                row_idx = row_idx + 1

                # - Average charge/discharge per day
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year_inv)
                sheet.cell(row=row_idx, column=3).value = int(year_curr)
                sheet.cell(row=row_idx, column=4).value = 'N/A'
                sheet.cell(row=row_idx, column=5).value = 'Average charge/discharge, [MVAh]'
                for p in range(shared_ess_data.num_instants):
                    avg_ch_dch = results[year_inv][year_curr][repr_days[0]]['avg_ch_dch'][node_id][p]
                    sheet.cell(row=row_idx, column=p + 6).value = avg_ch_dch
                    sheet.cell(row=row_idx, column=p + 6).number_format = decimal_style
                row_idx = row_idx + 1


def _write_relaxation_slacks_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Relaxation Slacks')

    row_idx = 1
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year'
    sheet.cell(row=row_idx, column=3).value = 'Day'
    sheet.cell(row=row_idx, column=4).value = 'Quantity'
    sheet.cell(row=row_idx, column=5).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=6).value = 'Operation Scenario'
    for p in range(shared_ess_data.num_instants):
        sheet.cell(row=row_idx, column=p + 7).value = p
    row_idx = row_idx + 1

    for year in results:
        for day in results[year]:
            for s_m in results[year][day]['scenarios']:
                for s_o in results[year][day]['scenarios'][s_m]:
                    for node_id in results[year][day]['scenarios']['relaxation_slacks']['slack_s_up']:

                        # Slack, Sup
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year)
                        sheet.cell(row=row_idx, column=3).value = day
                        sheet.cell(row=row_idx, column=4).value = 'Investment, slack_s_up'
                        sheet.cell(row=row_idx, column=5).value = s_m
                        sheet.cell(row=row_idx, column=6).value = s_o
                        for p in range(shared_ess_data.num_instants):
                            sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['slack_s_up'][node_id]
                            sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                        row_idx = row_idx + 1

                        # Slack, Sdown
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year)
                        sheet.cell(row=row_idx, column=3).value = day
                        sheet.cell(row=row_idx, column=4).value = 'Investment, slack_s_down'
                        sheet.cell(row=row_idx, column=5).value = s_m
                        sheet.cell(row=row_idx, column=6).value = s_o
                        for p in range(shared_ess_data.num_instants):
                            sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['slack_s_down'][node_id]
                            sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                        row_idx = row_idx + 1

                        # Slack, Eup
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year)
                        sheet.cell(row=row_idx, column=3).value = day
                        sheet.cell(row=row_idx, column=4).value = 'Investment, slack_e_up'
                        sheet.cell(row=row_idx, column=5).value = s_m
                        sheet.cell(row=row_idx, column=6).value = s_o
                        for p in range(shared_ess_data.num_instants):
                            sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['slack_e_up'][node_id]
                            sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                        row_idx = row_idx + 1

                        # Slack, Edown
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = int(year)
                        sheet.cell(row=row_idx, column=3).value = day
                        sheet.cell(row=row_idx, column=4).value = 'Investment, slack_e_down'
                        sheet.cell(row=row_idx, column=5).value = s_m
                        sheet.cell(row=row_idx, column=6).value = s_o
                        for p in range(shared_ess_data.num_instants):
                            sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['slack_e_down'][node_id]
                            sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                        row_idx = row_idx + 1

                        # Slack, Comp
                        if shared_ess_data.params.ess_relax_comp:
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'comp'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['comp'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                        # Slack, SoC
                        if shared_ess_data.params.ess_relax_soc:

                            # Slack, soc, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'soc_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['soc_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, soc, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'soc_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['soc_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                        # Slack, day balance
                        if shared_ess_data.params.ess_relax_day_balance:

                            # Slack, day balance, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'day_balance_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['day_balance_up'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, day balance, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'day_balance_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['day_balance_up'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                        # Capacity available
                        if shared_ess_data.params.ess_relax_capacity_available:

                            # Slack, capacity available, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'capacity_available_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['capacity_available_up'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, capacity available, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'capacity_available_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['capacity_available_down'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                        # Installed capacity
                        if shared_ess_data.params.ess_relax_installed_capacity:

                            # Slack, s_rated, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 's_rated_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['s_rated_up'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, s_rated, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 's_rated_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['s_rated_down'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, e_rated, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'e_rated_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['e_rated_up'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, e_rated, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'e_rated_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['e_rated_down'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                        # Capacity degradation
                        if shared_ess_data.params.ess_relax_capacity_degradation:

                            # Slack, capacity available, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'capacity_degradation_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['capacity_degradation_up'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Slack, capacity available, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'capacity_degradation_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['capacity_degradation_down'][node_id]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                        # Expected values
                        if shared_ess_data.params.ess_interface_relax:

                            # - Expected active power, up
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'expected_p_up'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['expected_p_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1

                            # - Expected active power, down
                            sheet.cell(row=row_idx, column=1).value = node_id
                            sheet.cell(row=row_idx, column=2).value = int(year)
                            sheet.cell(row=row_idx, column=3).value = day
                            sheet.cell(row=row_idx, column=4).value = 'expected_p_down'
                            sheet.cell(row=row_idx, column=5).value = s_m
                            sheet.cell(row=row_idx, column=6).value = s_o
                            for p in range(shared_ess_data.num_instants):
                                sheet.cell(row=row_idx, column=p + 7).value = results[year][day]['scenarios']['relaxation_slacks']['expected_p_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 7).number_format = decimal_style
                            row_idx = row_idx + 1


def _write_relaxation_slacks_yoy_results_to_excel(shared_ess_data, workbook, results):

    sheet = workbook.create_sheet('Relaxation Slacks (2)')

    years = [year for year in shared_ess_data.years]
    days = [day for day in shared_ess_data.days]

    row_idx = 1
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Year (from)'
    sheet.cell(row=row_idx, column=3).value = 'Year (to)'
    sheet.cell(row=row_idx, column=4).value = 'Quantity'
    sheet.cell(row=row_idx, column=5).value = 'Value'
    row_idx = row_idx + 1

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for year in years:
            for year2 in years:

                # Slack, Relative capacity up
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year)
                sheet.cell(row=row_idx, column=3).value = int(year2)
                sheet.cell(row=row_idx, column=4).value = 'Relative capacity, up'
                sheet.cell(row=row_idx, column=5).value = results['results'][year][days[0]]['scenarios']['relaxation_slacks']['relative_capacity_up'][node_id][year2]
                sheet.cell(row=row_idx, column=5).number_format = decimal_style
                row_idx = row_idx + 1

                # Slack, Relative capacity down
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = int(year)
                sheet.cell(row=row_idx, column=3).value = int(year2)
                sheet.cell(row=row_idx, column=4).value = 'Relative capacity, down'
                sheet.cell(row=row_idx, column=5).value = results['results'][year][days[0]]['scenarios']['relaxation_slacks']['relative_capacity_down'][node_id][year2]
                sheet.cell(row=row_idx, column=5).number_format = decimal_style
                row_idx = row_idx + 1
