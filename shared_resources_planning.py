import os
import time
from copy import copy
import pandas as pd
from math import isclose, sqrt
import networkx as nx
import matplotlib.pyplot as plt
import pyomo.opt as po
import pyomo.environ as pe
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from network_data import NetworkData
from load import Load
from shared_energy_storage import SharedEnergyStorage
from planning_parameters import PlanningParameters
from shared_energy_storage_data import SharedEnergyStorageData
from helper_functions import *


# ======================================================================================================================
#   Class SHARED RESOURCES PLANNING
# ======================================================================================================================
class SharedResourcesPlanning:

    def __init__(self, data_dir, filename):
        self.name = filename.replace('.json', '')
        self.data_dir = data_dir
        self.filename = filename
        self.market_data_file = str()
        self.results_dir = os.path.join(data_dir, 'Results')
        self.diagrams_dir = os.path.join(data_dir, 'Diagrams')
        self.params_file = str()
        self.years = dict()
        self.days = dict()
        self.num_instants = int()
        self.discount_factor = float()
        self.cost_energy_p = dict()
        self.cost_flex = dict()
        self.prob_market_scenarios = list()
        self.distribution_networks = dict()
        self.transmission_network = NetworkData()
        self.shared_ess_data = SharedEnergyStorageData()
        self.active_distribution_network_nodes = list()
        self.params = PlanningParameters()

    def run_planning_problem(self):
        print('[INFO] Running PLANNING PROBLEM...')
        _run_planning_problem(self)

    def run_operational_planning(self, candidate_solution=dict(), print_results=False, debug_flag=False):
        print('[INFO] Running OPERATIONAL PLANNING...')
        if not candidate_solution:
            candidate_solution = self.get_initial_candidate_solution()
        results, models, sensitivities, primal_evolution = _run_operational_planning(self, candidate_solution, debug_flag=debug_flag)
        if print_results:
            self.write_operational_planning_results_to_excel(models, results, primal_evolution)
        return results, models, sensitivities, primal_evolution

    def run_without_coordination(self, print_results=False):
        print('[INFO] Running PLANNING PROBLEM WITHOUT COORDINATION...')
        results, models = _run_operational_planning_without_coordination(self)
        if print_results:
            self.write_operational_planning_results_without_coordination_to_excel(models, results)

    def get_upper_bound(self, model):
        return _get_upper_bound(self, model)

    def get_primal_value(self, tso_model, dso_models, esso_model):
        return _get_primal_value(self, tso_model, dso_models, esso_model)

    def add_benders_cut(self, model, upper_bound, sensitivities, candidate_solution):
        _add_benders_cut(self, model, upper_bound, sensitivities, candidate_solution)

    def update_admm_consensus_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, params):
        _update_admm_consensus_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, params)

    def read_planning_problem(self):
        _read_planning_problem(self)

    def read_market_data_from_file(self):
        _read_market_data_from_file(self)

    def read_planning_parameters_from_file(self):
        print(f'[INFO] Reading PLANNING PARAMETERS from file...')
        filename = os.path.join(self.data_dir, self.params_file)
        self.params.read_parameters_from_file(filename)

    def write_planning_results_to_excel(self, operational_planning_models, operational_results=dict(), bound_evolution=dict(), execution_time=float()):
        filename = os.path.join(self.results_dir, self.name + '_planning_results.xlsx')
        processed_results = _process_operational_planning_results(self, operational_planning_models['tso'], operational_planning_models['dso'], operational_planning_models['esso'], operational_results)
        shared_ess_capacity = self.shared_ess_data.get_investment_and_available_capacities(operational_planning_models['esso'])
        _write_planning_results_to_excel(self, processed_results, bound_evolution=bound_evolution, shared_ess_capacity=shared_ess_capacity, filename=filename)

    def write_operational_planning_results_to_excel(self, optimization_models, results, primal_evolution=list()):
        filename = os.path.join(self.results_dir, self.name + '_operational_planning_results.xlsx')
        processed_results = _process_operational_planning_results(self, optimization_models['tso'], optimization_models['dso'], optimization_models['esso'], results)
        shared_ess_capacity = self.shared_ess_data.get_investment_and_available_capacities(optimization_models['esso'])
        _write_operational_planning_results_to_excel(self, processed_results, primal_evolution=primal_evolution, shared_ess_capacity=shared_ess_capacity, filename=filename)

    def write_operational_planning_results_without_coordination_to_excel(self, optimization_models, results):
        filename = os.path.join(self.results_dir, self.name + '_operational_planning_results_no_coordination.xlsx')
        processed_results = _process_operational_planning_results_no_coordination(self, optimization_models['tso'], optimization_models['dso'], results)
        _write_operational_planning_results_no_coordination_to_excel(self, processed_results, filename)

    def get_initial_candidate_solution(self):
        return _get_initial_candidate_solution(self)

    def plot_diagram(self):
        _plot_networkx_diagram(self)


# ======================================================================================================================
#  PLANNING functions
# ======================================================================================================================
def _run_planning_problem(planning_problem):

    shared_ess_data = planning_problem.shared_ess_data
    benders_parameters = planning_problem.params.benders
    lower_level_models = dict()
    operational_results = dict()

    # ------------------------------------------------------------------------------------------------------------------
    # 0. Initialization
    iter = 1
    convergence = False
    from_warm_start = False
    lower_bound = -1e12
    upper_bound = 1e12
    lower_bound_evolution = [lower_bound]
    upper_bound_evolution = [upper_bound]
    candidate_solution = planning_problem.get_initial_candidate_solution()

    start = time.time()
    master_problem_model = planning_problem.shared_ess_data.build_master_problem()
    shared_ess_data.optimize(master_problem_model)

    # Benders' main cycle
    while iter < benders_parameters.num_max_iters and not convergence:

        print(f'=============================================== ITERATION #{iter} ==============================================')
        print(f'[INFO] Iter {iter}. LB = {lower_bound}, UB = {upper_bound}')

        _print_candidate_solution(candidate_solution)

        # 1. Subproblem
        # 1.1. Solve operational planning, with fixed investment variables,
        # 1.2. Get coupling constraints' sensitivities (subproblem)
        # 1.3. Get OF value (upper bound) from the subproblem
        operational_results, lower_level_models, sensitivities, _ = planning_problem.run_operational_planning(candidate_solution, print_results=False)
        upper_bound = planning_problem.get_upper_bound(lower_level_models['tso'])
        upper_bound_evolution.append(upper_bound)

        #  - Convergence check
        if isclose(upper_bound, lower_bound, abs_tol=benders_parameters.tol_abs, rel_tol=benders_parameters.tol_rel):
            lower_bound_evolution.append(lower_bound)
            convergence = True
            break

        # 2. Solve Master problem
        # 2.1. Add Benders' cut, based on the sensitivities obtained from the subproblem
        # 2.2. Run master problem optimization
        # 2.3. Get new capacity values, and the value of alpha (lower bound)
        planning_problem.add_benders_cut(master_problem_model, upper_bound, sensitivities, candidate_solution)
        shared_ess_data.optimize(master_problem_model, from_warm_start=from_warm_start)
        candidate_solution = shared_ess_data.get_candidate_solution(master_problem_model)
        lower_bound = pe.value(master_problem_model.alpha)
        lower_bound_evolution.append(lower_bound)

        #  - Convergence check
        if isclose(upper_bound, lower_bound, abs_tol=benders_parameters.tol_abs, rel_tol=benders_parameters.tol_rel):
            lower_bound_evolution.append(lower_bound)
            convergence = True
            break

        iter += 1
        from_warm_start = True

    if not convergence:
        print('[WARNING] Convergence not obtained!')

    print('[INFO] Final. LB = {}, UB = {}'.format(lower_bound, upper_bound))

    # Write results
    end = time.time()
    total_execution_time = end - start
    print('[INFO] Execution time: {:.2f} s'.format(total_execution_time))
    bound_evolution = {'lower_bound': lower_bound_evolution, 'upper_bound': upper_bound_evolution}
    planning_problem.write_planning_results_to_excel(lower_level_models, operational_results, bound_evolution, execution_time=total_execution_time)


def _get_upper_bound(planning_problem, model):
    upper_bound = 0.00
    years = [year for year in planning_problem.years]
    for year in planning_problem.years:
        num_years = planning_problem.years[year]
        annualization = 1 / ((1 + planning_problem.discount_factor) ** (int(year) - int(years[0])))
        for day in planning_problem.days:
            num_days = planning_problem.days[day]
            network = planning_problem.transmission_network.network[year][day]
            params = planning_problem.transmission_network.params
            obj_repr_day = network.get_primal_value(model[year][day])
            upper_bound += num_days * num_years * annualization * obj_repr_day
    return upper_bound


def _add_benders_cut(planning_problem, model, upper_bound, sensitivities, candidate_solution):
    years = [year for year in planning_problem.years]
    benders_cut = upper_bound
    for e in model.energy_storages:
        node_id = planning_problem.active_distribution_network_nodes[e]
        for y in model.years:
            year = years[y]
            if sensitivities['s'][year][node_id] != 'N/A':
                benders_cut += sensitivities['s'][year][node_id] * (model.es_s_rated[e, y] - candidate_solution['total_capacity'][node_id][year]['s'])
            if sensitivities['e'][year][node_id] != 'N/A':
                benders_cut += sensitivities['e'][year][node_id] * (model.es_e_rated[e, y] - candidate_solution['total_capacity'][node_id][year]['e'])
    model.benders_cuts.add(model.alpha >= benders_cut)


# ======================================================================================================================
#  OPERATIONAL PLANNING functions
# ======================================================================================================================
def _run_operational_planning(planning_problem, candidate_solution, debug_flag=False):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data
    admm_parameters = planning_problem.params.admm
    results = {'tso': dict(), 'dso': dict(), 'esso': dict()}

    # ------------------------------------------------------------------------------------------------------------------
    # 0. Initialization

    print('[INFO]\t\t - Initializing...')

    start = time.time()
    from_warm_start = False
    primal_evolution = list()

    # Create ADMM variables
    consensus_vars, dual_vars = create_admm_variables(planning_problem)

    # Create Operational Planning models
    dso_models, results['dso'] = create_distribution_networks_models(distribution_networks, consensus_vars['interface']['v_sqr']['dso'], consensus_vars['interface']['pf']['dso'], consensus_vars['ess']['dso'], candidate_solution['total_capacity'])
    update_distribution_models_to_admm(distribution_networks, dso_models, consensus_vars['interface']['pf']['dso'], admm_parameters)

    tso_model, results['tso'] = create_transmission_network_model(transmission_network, consensus_vars['interface']['v_sqr'], consensus_vars['interface']['pf'], consensus_vars['ess'], candidate_solution['total_capacity'])
    update_transmission_model_to_admm(transmission_network, tso_model, consensus_vars['interface']['pf'], admm_parameters)

    esso_model = create_shared_energy_storage_model(shared_ess_data, consensus_vars['ess'], candidate_solution['investment'])
    update_shared_energy_storage_model_to_admm(shared_ess_data, esso_model, admm_parameters)

    planning_problem.update_admm_consensus_variables(tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters)
    if debug_flag:
        for node_id in planning_problem.active_distribution_network_nodes:
            print(f"Node {node_id}")
            for year in consensus_vars['interface']['pf']['tso']['current'][node_id]:
                print(f"\tYear {year}")
                for day in consensus_vars['interface']['pf']['tso']['current'][node_id][year]:
                    print(f"\t\tDay {day}")
                    print(f"\t\t\tTSO, V     {consensus_vars['interface']['v_sqr']['tso']['current'][node_id][year][day]}")
                    print(f"\t\t\tDSO, V     {consensus_vars['interface']['v_sqr']['dso']['current'][node_id][year][day]}")
                    print(f"\t\t\tTSO, P     {consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['p']}")
                    print(f"\t\t\tDSO, P     {consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['p']}")
                    print(f"\t\t\tTSO, Q     {consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['q']}")
                    print(f"\t\t\tDSO, Q     {consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['q']}")
                    # print(f"\t\t\tESS, TSO   {consensus_vars['ess']['tso']['current'][node_id][year][day]['p']}")
                    # print(f"\t\t\tESS, DSO   {consensus_vars['ess']['dso']['current'][node_id][year][day]['p']}")
                    # print(f"\t\t\tESS, ESSO  {consensus_vars['ess']['esso']['current'][node_id][year][day]['p']}")


    # ------------------------------------------------------------------------------------------------------------------
    # ADMM -- Main cycle
    # ------------------------------------------------------------------------------------------------------------------
    convergence, iter = False, 1
    for iter in range(iter, admm_parameters.num_max_iters):

        print(f'[INFO]\t - ADMM. Iter {iter}...')

        iter_start = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # 2. Solve TSO problem
        results['tso'] = update_transmission_coordination_model_and_solve(transmission_network, tso_model,
                                                                          consensus_vars['interface']['v_sqr'], dual_vars['v_sqr']['tso'],
                                                                          consensus_vars['interface']['pf'], dual_vars['pf']['tso'],
                                                                          consensus_vars['ess'], dual_vars['ess']['tso'],
                                                                          admm_parameters, from_warm_start=from_warm_start)

        # 2.1 Update ADMM CONSENSUS variables
        planning_problem.update_admm_consensus_variables(tso_model, dso_models, esso_model,
                                                         consensus_vars, dual_vars,
                                                         admm_parameters)
        if debug_flag:
            for node_id in planning_problem.active_distribution_network_nodes:
                print(f"Node {node_id}")
                for year in consensus_vars['interface']['pf']['tso']['current'][node_id]:
                    print(f"\tYear {year}")
                    for day in consensus_vars['interface']['pf']['tso']['current'][node_id][year]:
                        print(f"\t\tDay {day}")
                        print(f"\t\t\tTSO, V     {consensus_vars['interface']['v_sqr']['tso']['current'][node_id][year][day]}")
                        print(f"\t\t\tDSO, V     {consensus_vars['interface']['v_sqr']['dso']['current'][node_id][year][day]}")
                        print(f"\t\t\tTSO, P     {consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['p']}")
                        print(f"\t\t\tDSO, P     {consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['p']}")
                        print(f"\t\t\tTSO, Q     {consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['q']}")
                        print(f"\t\t\tDSO, Q     {consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['q']}")
                        #print(f"\t\t\tESS, TSO   {consensus_vars['ess']['tso']['current'][node_id][year][day]['p']}")
                        #print(f"\t\t\tESS, DSO   {consensus_vars['ess']['dso']['current'][node_id][year][day]['p']}")
                        #print(f"\t\t\tESS, ESSO  {consensus_vars['ess']['esso']['current'][node_id][year][day]['p']}")

        # 2.2 Update primal evolution
        primal_evolution.append(planning_problem.get_primal_value(tso_model, dso_models, esso_model))

        # 2.3 STOPPING CRITERIA evaluation
        convergence = check_admm_convergence(planning_problem, consensus_vars, admm_parameters)
        if convergence:
            break

        # --------------------------------------------------------------------------------------------------------------
        # 3. Solve ESSO problem
        results['esso'] = update_shared_energy_storages_coordination_model_and_solve(planning_problem, esso_model,
                                                                                     consensus_vars['ess']['tso']['current'], dual_vars['ess']['esso']['current']['tso'],
                                                                                     admm_parameters, from_warm_start=from_warm_start)

        # 3.1 Update ADMM CONSENSUS variables
        planning_problem.update_admm_consensus_variables(tso_model, dso_models, esso_model,
                                                         consensus_vars, dual_vars,
                                                         admm_parameters)

        # 3.2 Update primal evolution
        primal_evolution.append(planning_problem.get_primal_value(tso_model, dso_models, esso_model))

        # 4.3 STOPPING CRITERIA evaluation
        convergence = check_admm_convergence(planning_problem, consensus_vars, admm_parameters)
        if convergence:
            break

        # --------------------------------------------------------------------------------------------------------------
        # 3. Solve DSOs problems
        results['dso'] = update_distribution_coordination_models_and_solve(distribution_networks, dso_models,
                                                                           consensus_vars['interface']['v_sqr'], dual_vars['v_sqr']['dso'],
                                                                           consensus_vars['interface']['pf'], dual_vars['pf']['dso'],
                                                                           consensus_vars['ess'], dual_vars['ess']['dso'],
                                                                           admm_parameters, from_warm_start=from_warm_start)

        # 3.1 Update ADMM CONSENSUS variables
        planning_problem.update_admm_consensus_variables(tso_model, dso_models, esso_model,
                                                         consensus_vars, dual_vars,
                                                         admm_parameters)
        if debug_flag:
            for node_id in planning_problem.active_distribution_network_nodes:
                print(f"Node {node_id}")
                for year in consensus_vars['interface']['pf']['tso']['current'][node_id]:
                    print(f"\tYear {year}")
                    for day in consensus_vars['interface']['pf']['tso']['current'][node_id][year]:
                        print(f"\t\tDay {day}")
                        print(f"\t\t\tTSO, V     {consensus_vars['interface']['v_sqr']['tso']['current'][node_id][year][day]}")
                        print(f"\t\t\tDSO, V     {consensus_vars['interface']['v_sqr']['dso']['current'][node_id][year][day]}")
                        print(f"\t\t\tTSO, P     {consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['p']}")
                        print(f"\t\t\tDSO, P     {consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['p']}")
                        print(f"\t\t\tTSO, Q     {consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['q']}")
                        print(f"\t\t\tDSO, Q     {consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['q']}")
                        #print(f"\t\t\tESS, TSO   {consensus_vars['ess']['tso']['current'][node_id][year][day]['p']}")
                        #print(f"\t\t\tESS, DSO   {consensus_vars['ess']['dso']['current'][node_id][year][day]['p']}")
                        #print(f"\t\t\tESS, ESSO  {consensus_vars['ess']['esso']['current'][node_id][year][day]['p']}")

        # 3.2 Update primal evolution
        primal_evolution.append(planning_problem.get_primal_value(tso_model, dso_models, esso_model))

        # 3.3 STOPPING CRITERIA evaluation
        convergence = check_admm_convergence(planning_problem, consensus_vars, admm_parameters)
        if convergence:
            break

        # --------------------------------------------------------------------------------------------------------------
        # 4. Solve ESSO problem
        results['esso'] = update_shared_energy_storages_coordination_model_and_solve(planning_problem, esso_model,
                                                                                     consensus_vars['ess']['dso']['current'], dual_vars['ess']['esso']['current']['dso'],
                                                                                     admm_parameters, from_warm_start=from_warm_start)

        # 4.1 Update ADMM CONSENSUS variables
        planning_problem.update_admm_consensus_variables(tso_model, dso_models, esso_model,
                                                         consensus_vars, dual_vars,
                                                         admm_parameters)

        # 4.2 Update primal evolution
        primal_evolution.append(planning_problem.get_primal_value(tso_model, dso_models, esso_model))

        # 4.3 STOPPING CRITERIA evaluation
        convergence = check_admm_convergence(planning_problem, consensus_vars, admm_parameters)
        if convergence:
            break

        iter_end = time.time()
        print('[INFO] \t - Iter {}: {:.2f} s'.format(iter, iter_end - iter_start))

        from_warm_start = True

    if not convergence:
        print(f'[WARNING] ADMM did NOT converge in {admm_parameters.num_max_iters} iterations!')
    else:
        print(f'[INFO] \t - ADMM converged in {iter} iterations.')

    end = time.time()
    total_execution_time = end - start
    print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))

    optim_models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
    sensitivities = transmission_network.get_sensitivities(tso_model)

    return results, optim_models, sensitivities, primal_evolution


def create_transmission_network_model(transmission_network, interface_v_vars, interface_pf_vars, sess_vars, candidate_solution):

    # Build model, fix candidate solution
    transmission_network.update_data_with_candidate_solution(candidate_solution)
    tso_model = transmission_network.build_model()
    transmission_network.update_model_with_candidate_solution(tso_model, candidate_solution)

    # Update model with expected interface values
    for year in transmission_network.years:
        for day in transmission_network.days:

            # Add expected interface values
            tso_model[year][day].active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))
            tso_model[year][day].expected_interface_vmag_sqr = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=0.00)
            tso_model[year][day].expected_interface_pf_p = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            tso_model[year][day].expected_interface_pf_q = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            tso_model[year][day].expected_shared_ess_p = pe.Var(tso_model[year][day].shared_energy_storages, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            tso_model[year][day].expected_shared_ess_q = pe.Var(tso_model[year][day].shared_energy_storages, tso_model[year][day].periods, domain=pe.Reals, initialize=0.00)

            # Update OF
            obj = tso_model[year][day].objective.expr
            for dn in tso_model[year][day].active_distribution_networks:

                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                adn_node_idx = transmission_network.network[year][day].get_node_idx(adn_node_id)
                adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(adn_node_id)

                # Free Pc, Qc at the interface node
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setlb(None)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setlb(None)

                # Define expected interface values
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:

                            interface_vmag_sqr = (tso_model[year][day].e_actual[adn_node_idx, s_m, s_o, p] ** 2) + (tso_model[year][day].f_actual[adn_node_idx, s_m, s_o, p] ** 2)
                            interface_pf_p = tso_model[year][day].pc[adn_load_idx, s_m, s_o, p]
                            interface_pf_q = tso_model[year][day].qc[adn_load_idx, s_m, s_o, p]

                            obj += PENALTY_SLACK * (tso_model[year][day].expected_interface_vmag_sqr[dn, p] - interface_vmag_sqr) ** 2
                            obj += PENALTY_SLACK * (tso_model[year][day].expected_interface_pf_p[dn, p] - interface_pf_p) ** 2
                            obj += PENALTY_SLACK * (tso_model[year][day].expected_interface_pf_q[dn, p] - interface_pf_q) ** 2

                            interface_ess_p = tso_model[year][day].shared_es_pch[shared_ess_idx, s_m, s_o, p] - tso_model[year][day].shared_es_pdch[shared_ess_idx, s_m, s_o, p]
                            interface_ess_q = tso_model[year][day].shared_es_qch[shared_ess_idx, s_m, s_o, p] - tso_model[year][day].shared_es_qdch[shared_ess_idx, s_m, s_o, p]

                            obj += PENALTY_SLACK * (tso_model[year][day].expected_shared_ess_p[dn, p] - interface_ess_p) ** 2
                            obj += PENALTY_SLACK * (tso_model[year][day].expected_shared_ess_q[dn, p] - interface_ess_q) ** 2

            tso_model[year][day].objective.expr = obj

    # Fix expected interface values
    for year in transmission_network.years:
        for day in transmission_network.days:
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                for p in tso_model[year][day].periods:

                    interface_vmag_sqr_req = interface_v_vars['dso']['current'][adn_node_id][year][day][p]
                    interface_pf_p_req = interface_pf_vars['dso']['current'][adn_node_id][year][day]['p'][p] / s_base
                    interface_pf_q_req = interface_pf_vars['dso']['current'][adn_node_id][year][day]['q'][p] / s_base

                    tso_model[year][day].expected_interface_vmag_sqr[dn, p].fix(interface_vmag_sqr_req)
                    tso_model[year][day].expected_interface_pf_p[dn, p].fix(interface_pf_p_req)
                    tso_model[year][day].expected_interface_pf_q[dn, p].fix(interface_pf_q_req)

                    interface_ess_p = sess_vars['dso']['current'][adn_node_id][year][day]['p'][p] / s_base
                    interface_ess_q = sess_vars['dso']['current'][adn_node_id][year][day]['q'][p] / s_base

                    tso_model[year][day].expected_shared_ess_p[dn, p].fix(interface_ess_p)
                    tso_model[year][day].expected_shared_ess_q[dn, p].fix(interface_ess_q)

    # Run S-MOPF
    results = transmission_network.optimize(tso_model)

    # Get expected values
    for year in transmission_network.years:
        for day in transmission_network.days:
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(adn_node_id)
                for p in tso_model[year][day].periods:

                    # Get initial interface PF values
                    vmag_sqr = pe.value(tso_model[year][day].expected_interface_vmag_sqr[dn, p])
                    interface_pf_p = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    interface_pf_q = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base

                    interface_v_vars['tso']['current'][adn_node_id][year][day][p] = vmag_sqr
                    interface_pf_vars['tso']['current'][adn_node_id][year][day]['p'][p] = interface_pf_p
                    interface_pf_vars['tso']['current'][adn_node_id][year][day]['q'][p] = interface_pf_q

                    # Get initial Shared ESS values
                    shared_ess_p = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base
                    shared_ess_q = pe.value(tso_model[year][day].expected_shared_ess_q[shared_ess_idx, p]) * s_base

                    sess_vars['tso']['current'][adn_node_id][year][day]['p'][p] = shared_ess_p
                    sess_vars['tso']['current'][adn_node_id][year][day]['q'][p] = shared_ess_q

    return tso_model, results


def create_distribution_networks_models(distribution_networks, interface_vars_vmag, interface_vars_pf, sess_vars, candidate_solution):

    dso_models = dict()
    results = dict()

    for node_id in distribution_networks:

        distribution_network = distribution_networks[node_id]

        # Build model, fix candidate solution
        distribution_network.update_data_with_candidate_solution(candidate_solution)
        dso_model = distribution_network.build_model()
        distribution_network.update_model_with_candidate_solution(dso_model, candidate_solution)

        # Add expected interface values
        for year in distribution_network.years:
            for day in distribution_network.days:

                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                ref_node_idx = distribution_network.network[year][day].get_node_idx(ref_node_id)
                ref_gen_idx = distribution_network.network[year][day].get_reference_gen_idx()
                shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)

                # Add interface expectation variables
                dso_model[year][day].expected_interface_vmag_sqr = pe.Var(dso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=0.00)
                dso_model[year][day].expected_interface_pf_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
                dso_model[year][day].expected_interface_pf_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
                dso_model[year][day].expected_shared_ess_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
                dso_model[year][day].expected_shared_ess_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)

                # Update OF
                obj = dso_model[year][day].objective.expr
                for s_m in dso_model[year][day].scenarios_market:
                    for s_o in dso_model[year][day].scenarios_operation:
                        for p in dso_model[year][day].periods:

                            vmag_sqr = dso_model[year][day].e_actual[ref_node_idx, s_m, s_o, p] ** 2
                            interface_pf_p = dso_model[year][day].pg[ref_gen_idx, s_m, s_o, p]
                            interface_pf_q = dso_model[year][day].qg[ref_gen_idx, s_m, s_o, p]
                            obj += PENALTY_SLACK * (dso_model[year][day].expected_interface_vmag_sqr[p] - vmag_sqr) ** 2
                            obj += PENALTY_SLACK * (dso_model[year][day].expected_interface_pf_p[p] - interface_pf_p) ** 2
                            obj += PENALTY_SLACK * (dso_model[year][day].expected_interface_pf_q[p] - interface_pf_q) ** 2

                            interface_ess_p = dso_model[year][day].shared_es_pch[shared_ess_idx, s_m, s_o, p] - dso_model[year][day].shared_es_pdch[shared_ess_idx, s_m, s_o, p]
                            interface_ess_q = dso_model[year][day].shared_es_qch[shared_ess_idx, s_m, s_o, p] - dso_model[year][day].shared_es_qdch[shared_ess_idx, s_m, s_o, p]
                            obj += PENALTY_SLACK * (dso_model[year][day].expected_shared_ess_p[p] - interface_ess_p) ** 2
                            obj += PENALTY_SLACK * (dso_model[year][day].expected_shared_ess_q[p] - interface_ess_q) ** 2

                dso_model[year][day].objective.expr = obj

        # Run SMOPF
        results[node_id] = distribution_network.optimize(dso_model)

        # Get expected values
        for year in distribution_network.years:
            for day in distribution_network.days:
                s_base = distribution_network.network[year][day].baseMVA
                for p in dso_model[year][day].periods:

                    # Get initial interface PF and Vmag values
                    interface_pf_vmag_sqr = pe.value(dso_model[year][day].expected_interface_vmag_sqr[p])
                    interface_pf_p = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    interface_pf_q = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base
                    interface_vars_vmag['current'][node_id][year][day][p] = interface_pf_vmag_sqr
                    interface_vars_pf['current'][node_id][year][day]['p'][p] = interface_pf_p
                    interface_vars_pf['current'][node_id][year][day]['q'][p] = interface_pf_q
                    interface_vars_vmag['prev'][node_id][year][day][p] = interface_pf_vmag_sqr
                    interface_vars_pf['prev'][node_id][year][day]['p'][p] = interface_pf_p
                    interface_vars_pf['prev'][node_id][year][day]['q'][p] = interface_pf_q

                    # Get initial Shared ESS values
                    p_ess = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base
                    q_ess = pe.value(dso_model[year][day].expected_shared_ess_q[p]) * s_base
                    sess_vars['current'][node_id][year][day]['p'][p] = p_ess
                    sess_vars['current'][node_id][year][day]['q'][p] = q_ess
                    sess_vars['prev'][node_id][year][day]['p'][p] = p_ess
                    sess_vars['prev'][node_id][year][day]['q'][p] = q_ess

        dso_models[node_id] = dso_model

    return dso_models, results


def create_shared_energy_storage_model(shared_ess_data, sess_vars, candidate_solution):

    years = [year for year in shared_ess_data.years]
    days = [day for day in shared_ess_data.days]

    shared_ess_data.update_data_with_candidate_solution(candidate_solution)
    esso_model = shared_ess_data.build_subproblem()
    shared_ess_data.update_model_with_candidate_solution(esso_model, candidate_solution)

    for e in esso_model.energy_storages:
        node_id = shared_ess_data.active_distribution_network_nodes[e]
        for y in esso_model.years:
            year = years[y]
            for d in esso_model.days:
                day = days[d]
                for p in esso_model.periods:
                    shared_ess_p = pe.value(esso_model.es_pnet[e, y, d, p])
                    shared_ess_q = pe.value(esso_model.es_qnet[e, y, d, p])
                    sess_vars['esso']['current'][node_id][year][day]['p'][p] = shared_ess_p
                    sess_vars['esso']['current'][node_id][year][day]['q'][p] = shared_ess_q

    return esso_model


def _get_primal_value(planning_problem, tso_model, dso_models, esso_model):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data

    primal_value = 0.0
    primal_value += transmission_network.get_primal_value(tso_model)
    for node_id in distribution_networks:
        primal_value += distribution_networks[node_id].get_primal_value(dso_models[node_id])
    primal_value += shared_ess_data.get_primal_value(esso_model)

    return primal_value


# ======================================================================================================================
#  ADMM functions
# ======================================================================================================================
def create_admm_variables(planning_problem):

    num_instants = planning_problem.num_instants

    consensus_variables = {
        'interface': {
            'v_sqr': {'tso': {'current': dict(), 'prev': dict()},
                      'dso': {'current': dict(), 'prev': dict()}},
            'pf': {'tso': {'current': dict(), 'prev': dict()},
                   'dso': {'current': dict(), 'prev': dict()}}
        },
        'ess': {'tso': {'current': dict(), 'prev': dict()},
                'dso': {'current': dict(), 'prev': dict()},
                'esso': {'current': dict(), 'prev': dict()},
                'capacity': {'s': dict(), 'e': dict()}}
    }

    dual_variables = {
        'v_sqr': {'tso': {'current': dict(), 'prev': dict()},
                  'dso': {'current': dict(), 'prev': dict()}},
        'pf': {'tso': {'current': dict(), 'prev': dict()},
               'dso': {'current': dict(), 'prev': dict()}},
        'ess': {'tso': {'current': dict(), 'prev': dict()},
                'dso': {'current': dict(), 'prev': dict()},
                'esso': {'current': {'tso': dict(), 'dso': dict()},
                         'prev': dict()}}
    }

    for dn in range(len(planning_problem.active_distribution_network_nodes)):

        node_id = planning_problem.active_distribution_network_nodes[dn]

        consensus_variables['interface']['v_sqr']['tso']['current'][node_id] = dict()
        consensus_variables['interface']['v_sqr']['dso']['current'][node_id] = dict()
        consensus_variables['interface']['v_sqr']['tso']['prev'][node_id] = dict()
        consensus_variables['interface']['v_sqr']['dso']['prev'][node_id] = dict()
        consensus_variables['interface']['pf']['tso']['current'][node_id] = dict()
        consensus_variables['interface']['pf']['dso']['current'][node_id] = dict()
        consensus_variables['interface']['pf']['tso']['prev'][node_id] = dict()
        consensus_variables['interface']['pf']['dso']['prev'][node_id] = dict()
        consensus_variables['ess']['tso']['current'][node_id] = dict()
        consensus_variables['ess']['dso']['current'][node_id] = dict()
        consensus_variables['ess']['esso']['current'][node_id] = dict()
        consensus_variables['ess']['tso']['prev'][node_id] = dict()
        consensus_variables['ess']['dso']['prev'][node_id] = dict()
        consensus_variables['ess']['esso']['prev'][node_id] = dict()

        dual_variables['v_sqr']['tso']['current'][node_id] = dict()
        dual_variables['v_sqr']['dso']['current'][node_id] = dict()
        dual_variables['v_sqr']['tso']['prev'][node_id] = dict()
        dual_variables['v_sqr']['dso']['prev'][node_id] = dict()
        dual_variables['pf']['tso']['current'][node_id] = dict()
        dual_variables['pf']['dso']['current'][node_id] = dict()
        dual_variables['pf']['tso']['prev'][node_id] = dict()
        dual_variables['pf']['dso']['prev'][node_id] = dict()
        dual_variables['ess']['tso']['current'][node_id] = dict()
        dual_variables['ess']['dso']['current'][node_id] = dict()
        dual_variables['ess']['esso']['current']['tso'][node_id] = dict()
        dual_variables['ess']['esso']['current']['dso'][node_id] = dict()
        dual_variables['ess']['tso']['prev'][node_id] = dict()
        dual_variables['ess']['dso']['prev'][node_id] = dict()
        dual_variables['ess']['esso']['prev'][node_id] = dict()

        for year in planning_problem.years:

            consensus_variables['interface']['v_sqr']['tso']['current'][node_id][year] = dict()
            consensus_variables['interface']['v_sqr']['tso']['prev'][node_id][year] = dict()
            consensus_variables['interface']['v_sqr']['dso']['current'][node_id][year] = dict()
            consensus_variables['interface']['v_sqr']['dso']['prev'][node_id][year] = dict()
            consensus_variables['interface']['pf']['tso']['current'][node_id][year] = dict()
            consensus_variables['interface']['pf']['tso']['prev'][node_id][year] = dict()
            consensus_variables['interface']['pf']['dso']['current'][node_id][year] = dict()
            consensus_variables['interface']['pf']['dso']['prev'][node_id][year] = dict()
            consensus_variables['ess']['tso']['current'][node_id][year] = dict()
            consensus_variables['ess']['tso']['prev'][node_id][year] = dict()
            consensus_variables['ess']['dso']['current'][node_id][year] = dict()
            consensus_variables['ess']['dso']['prev'][node_id][year] = dict()
            consensus_variables['ess']['esso']['current'][node_id][year] = dict()
            consensus_variables['ess']['esso']['prev'][node_id][year] = dict()

            dual_variables['v_sqr']['tso']['current'][node_id][year] = dict()
            dual_variables['v_sqr']['tso']['prev'][node_id][year] = dict()
            dual_variables['v_sqr']['dso']['current'][node_id][year] = dict()
            dual_variables['v_sqr']['dso']['prev'][node_id][year] = dict()
            dual_variables['pf']['tso']['current'][node_id][year] = dict()
            dual_variables['pf']['tso']['prev'][node_id][year] = dict()
            dual_variables['pf']['dso']['current'][node_id][year] = dict()
            dual_variables['pf']['dso']['prev'][node_id][year] = dict()
            dual_variables['ess']['tso']['current'][node_id][year] = dict()
            dual_variables['ess']['tso']['prev'][node_id][year] = dict()
            dual_variables['ess']['dso']['current'][node_id][year] = dict()
            dual_variables['ess']['dso']['prev'][node_id][year] = dict()
            dual_variables['ess']['esso']['current']['tso'][node_id][year] = dict()
            dual_variables['ess']['esso']['current']['dso'][node_id][year] = dict()
            dual_variables['ess']['esso']['prev'][node_id][year] = dict()

            for day in planning_problem.days:
                consensus_variables['interface']['v_sqr']['tso']['current'][node_id][year][day] = [1.0] * num_instants
                consensus_variables['interface']['v_sqr']['dso']['current'][node_id][year][day] = [1.0] * num_instants
                consensus_variables['interface']['pf']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['interface']['pf']['dso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['dso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['esso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['interface']['v_sqr']['tso']['prev'][node_id][year][day] = [1.0] * num_instants
                consensus_variables['interface']['v_sqr']['dso']['prev'][node_id][year][day] = [1.0] * num_instants
                consensus_variables['interface']['pf']['tso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['interface']['pf']['dso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['tso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['dso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['esso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

                dual_variables['v_sqr']['tso']['current'][node_id][year][day] = [0.0] * planning_problem.num_instants
                dual_variables['v_sqr']['dso']['current'][node_id][year][day] = [0.0] * planning_problem.num_instants
                dual_variables['pf']['tso']['current'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['pf']['dso']['current'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['tso']['current'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['dso']['current'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['esso']['current']['tso'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['esso']['current']['dso'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['v_sqr']['tso']['prev'][node_id][year][day] = [0.0] * planning_problem.num_instants
                dual_variables['v_sqr']['dso']['prev'][node_id][year][day] = [0.0] * planning_problem.num_instants
                dual_variables['pf']['tso']['prev'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['pf']['dso']['prev'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['tso']['prev'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['dso']['prev'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['esso']['prev'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}

    return consensus_variables, dual_variables


def _update_admm_consensus_variables(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, params):
    _update_previous_consensus_variables(planning_problem, consensus_vars)
    _update_interface_power_flow_variables(planning_problem, tso_model, dso_models, consensus_vars['interface'], dual_vars, params)
    _update_shared_energy_storage_variables(planning_problem, tso_model, dso_models, esso_model, consensus_vars['ess'], dual_vars['ess'], params)


def _update_previous_consensus_variables(planning_problem, consensus_vars):
    for dn in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[dn]
        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in range(planning_problem.num_instants):
                    consensus_vars['interface']['v_sqr']['tso']['prev'][node_id][year][day][p] = copy(consensus_vars['interface']['v_sqr']['tso']['current'][node_id][year][day][p])
                    consensus_vars['interface']['v_sqr']['dso']['prev'][node_id][year][day][p] = copy(consensus_vars['interface']['v_sqr']['dso']['current'][node_id][year][day][p])
                    consensus_vars['interface']['pf']['tso']['prev'][node_id][year][day]['p'][p] = copy(consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['p'][p])
                    consensus_vars['interface']['pf']['tso']['prev'][node_id][year][day]['q'][p] = copy(consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['q'][p])
                    consensus_vars['interface']['pf']['dso']['prev'][node_id][year][day]['p'][p] = copy(consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['p'][p])
                    consensus_vars['interface']['pf']['dso']['prev'][node_id][year][day]['q'][p] = copy(consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['q'][p])
                    consensus_vars['ess']['tso']['prev'][node_id][year][day]['p'][p] = copy(consensus_vars['ess']['tso']['current'][node_id][year][day]['p'][p])
                    consensus_vars['ess']['tso']['prev'][node_id][year][day]['q'][p] = copy(consensus_vars['ess']['tso']['current'][node_id][year][day]['q'][p])
                    consensus_vars['ess']['dso']['prev'][node_id][year][day]['p'][p] = copy(consensus_vars['ess']['dso']['current'][node_id][year][day]['p'][p])
                    consensus_vars['ess']['dso']['prev'][node_id][year][day]['q'][p] = copy(consensus_vars['ess']['dso']['current'][node_id][year][day]['q'][p])


def _update_interface_power_flow_variables(planning_problem, tso_model, dso_models, interface_vars, dual_vars, params):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    # Transmission network - Update Vmag and PF at the TN-DN interface
    for dn in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[dn]
        for year in planning_problem.years:
            for day in planning_problem.days:
                s_base = transmission_network.network[year][day].baseMVA
                vmin, vmax = transmission_network.network[year][day].get_node_voltage_limits(node_id)
                s_max = distribution_networks[node_id].network[year][day].get_interface_branch_rating()
                for p in tso_model[year][day].periods:
                    v_req = pe.value(tso_model[year][day].expected_interface_vmag_sqr[dn, p])
                    p_req = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    q_req = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base
                    interface_vars['v_sqr']['tso']['current'][node_id][year][day][p] = min(max(v_req, vmin), vmax)
                    interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p] = min(max(p_req, -s_max), s_max)
                    interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p] = min(max(q_req, -s_max), s_max)

    # Distribution Network - Update PF at the TN-DN interface
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        dso_model = dso_models[node_id]
        for year in planning_problem.years:
            for day in planning_problem.days:
                s_base = distribution_network.network[year][day].baseMVA
                vmin, vmax = transmission_network.network[year][day].get_node_voltage_limits(node_id)
                s_max = distribution_network.network[year][day].get_interface_branch_rating()
                for p in dso_model[year][day].periods:
                    v_req = pe.value(dso_model[year][day].expected_interface_vmag_sqr[p])
                    p_req = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    q_req = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base
                    interface_vars['v_sqr']['dso']['current'][node_id][year][day][p] = min(max(v_req, vmin), vmax)
                    interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p] = min(max(p_req, -s_max), s_max)
                    interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p] = min(max(q_req, -s_max), s_max)

    # Update Lambdas
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in range(planning_problem.num_instants):

                    error_vsqr_req_tso = interface_vars['v_sqr']['tso']['current'][node_id][year][day][p] - interface_vars['v_sqr']['dso']['current'][node_id][year][day][p]
                    error_p_pf_req_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p] - interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p]
                    error_q_pf_req_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p] - interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p]
                    error_vsqr_req_dso = interface_vars['v_sqr']['dso']['current'][node_id][year][day][p] - interface_vars['v_sqr']['tso']['current'][node_id][year][day][p]
                    error_p_pf_req_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p] - interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p]
                    error_q_pf_req_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p] - interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p]
                    dual_vars['v_sqr']['tso']['current'][node_id][year][day][p] += params.rho['v'][transmission_network.name] * error_vsqr_req_tso
                    dual_vars['pf']['tso']['current'][node_id][year][day]['p'][p] += params.rho['pf'][transmission_network.name] * error_p_pf_req_tso
                    dual_vars['pf']['tso']['current'][node_id][year][day]['q'][p] += params.rho['pf'][transmission_network.name] * error_q_pf_req_tso
                    dual_vars['v_sqr']['dso']['current'][node_id][year][day][p] += params.rho['v'][distribution_network.name] * error_vsqr_req_dso
                    dual_vars['pf']['dso']['current'][node_id][year][day]['p'][p] += params.rho['pf'][distribution_network.name] * error_p_pf_req_dso
                    dual_vars['pf']['dso']['current'][node_id][year][day]['q'][p] += params.rho['pf'][distribution_network.name] * error_q_pf_req_dso

                    error_vsqr_prev_tso = interface_vars['v_sqr']['tso']['current'][node_id][year][day][p] - interface_vars['v_sqr']['tso']['prev'][node_id][year][day][p]
                    error_p_pf_prev_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p] - interface_vars['pf']['tso']['prev'][node_id][year][day]['p'][p]
                    error_q_pf_prev_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p] - interface_vars['pf']['tso']['prev'][node_id][year][day]['q'][p]
                    error_vsqr_prev_dso = interface_vars['v_sqr']['dso']['current'][node_id][year][day][p] - interface_vars['v_sqr']['dso']['prev'][node_id][year][day][p]
                    error_p_pf_prev_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p] - interface_vars['pf']['dso']['prev'][node_id][year][day]['p'][p]
                    error_q_pf_prev_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p] - interface_vars['pf']['dso']['prev'][node_id][year][day]['q'][p]
                    dual_vars['v_sqr']['tso']['prev'][node_id][year][day][p] += params.rho['v'][transmission_network.name] * error_vsqr_prev_tso
                    dual_vars['pf']['tso']['prev'][node_id][year][day]['p'][p] += params.rho['pf'][transmission_network.name] * error_p_pf_prev_tso
                    dual_vars['pf']['tso']['prev'][node_id][year][day]['q'][p] += params.rho['pf'][transmission_network.name] * error_q_pf_prev_tso
                    dual_vars['v_sqr']['dso']['prev'][node_id][year][day][p] += params.rho['v'][distribution_network.name] * error_vsqr_prev_dso
                    dual_vars['pf']['dso']['prev'][node_id][year][day]['p'][p] += params.rho['pf'][distribution_network.name] * error_p_pf_prev_dso
                    dual_vars['pf']['dso']['prev'][node_id][year][day]['q'][p] += params.rho['pf'][distribution_network.name] * error_q_pf_prev_dso


def _update_shared_energy_storage_variables(planning_problem, tso_model, dso_models, sess_model, shared_ess_vars, dual_vars, params):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data
    repr_days = [day for day in planning_problem.days]
    repr_years = [year for year in planning_problem.years]

    for node_id in distribution_networks:

        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]

        # Power requested by ESSO
        for y in sess_model.years:
            year = repr_years[y]
            shared_ess_idx = shared_ess_data.get_shared_energy_storage_idx(node_id)
            for d in sess_model.days:
                day = repr_days[d]
                for p in sess_model.periods:
                    p_req = pe.value(sess_model.es_pnet[shared_ess_idx, y, d, p])
                    q_req = pe.value(sess_model.es_qnet[shared_ess_idx, y, d, p])
                    shared_ess_vars['esso']['current'][node_id][year][day]['p'][p] = p_req
                    shared_ess_vars['esso']['current'][node_id][year][day]['q'][p] = q_req

        # Power requested by TSO
        for y in range(len(repr_years)):
            year = repr_years[y]
            for d in range(len(repr_days)):
                day = repr_days[d]
                s_base = transmission_network.network[year][day].baseMVA
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                for p in tso_model[year][day].periods:
                    p_req = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base
                    q_req = pe.value(tso_model[year][day].expected_shared_ess_q[shared_ess_idx, p]) * s_base
                    shared_ess_vars['tso']['current'][node_id][year][day]['p'][p] = p_req
                    shared_ess_vars['tso']['current'][node_id][year][day]['q'][p] = q_req

        # Power requested by DSO
        for y in range(len(repr_years)):
            year = repr_years[y]
            for d in range(len(repr_days)):
                day = repr_days[d]
                s_base = distribution_network.network[year][day].baseMVA
                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)
                for p in dso_model[year][day].periods:
                    p_req = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base
                    q_req = pe.value(dso_model[year][day].expected_shared_ess_q[p]) * s_base
                    shared_ess_vars['dso']['current'][node_id][year][day]['p'][p] = p_req
                    shared_ess_vars['dso']['current'][node_id][year][day]['q'][p] = q_req

        # Update dual variables Shared ESS
        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in range(planning_problem.num_instants):

                    error_p_tso_esso = shared_ess_vars['tso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['esso']['current'][node_id][year][day]['p'][p]
                    error_q_tso_esso = shared_ess_vars['tso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['esso']['current'][node_id][year][day]['q'][p]
                    error_p_dso_esso = shared_ess_vars['dso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['esso']['current'][node_id][year][day]['p'][p]
                    error_q_dso_esso = shared_ess_vars['dso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['tso']['current'][node_id][year][day]['q'][p]
                    error_p_esso_tso = shared_ess_vars['esso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['tso']['current'][node_id][year][day]['p'][p]
                    error_q_esso_tso = shared_ess_vars['esso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['tso']['current'][node_id][year][day]['q'][p]
                    error_p_esso_dso = shared_ess_vars['esso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['dso']['current'][node_id][year][day]['p'][p]
                    error_q_esso_dso = shared_ess_vars['esso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['dso']['current'][node_id][year][day]['q'][p]

                    error_p_tso_prev = shared_ess_vars['tso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['tso']['prev'][node_id][year][day]['p'][p]
                    error_q_tso_prev = shared_ess_vars['tso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['tso']['prev'][node_id][year][day]['q'][p]
                    error_p_dso_prev = shared_ess_vars['dso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['dso']['prev'][node_id][year][day]['p'][p]
                    error_q_dso_prev = shared_ess_vars['dso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['dso']['prev'][node_id][year][day]['q'][p]
                    error_p_esso_prev = shared_ess_vars['esso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['esso']['prev'][node_id][year][day]['p'][p]
                    error_q_esso_prev = shared_ess_vars['esso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['esso']['prev'][node_id][year][day]['q'][p]

                    dual_vars['tso']['current'][node_id][year][day]['p'][p] += params.rho['ess'][transmission_network.name] * error_p_tso_esso
                    dual_vars['tso']['current'][node_id][year][day]['q'][p] += params.rho['ess'][transmission_network.name] * error_q_tso_esso
                    dual_vars['dso']['current'][node_id][year][day]['p'][p] += params.rho['ess'][distribution_network.name] * error_p_dso_esso
                    dual_vars['dso']['current'][node_id][year][day]['q'][p] += params.rho['ess'][distribution_network.name] * error_q_dso_esso
                    dual_vars['esso']['current']['tso'][node_id][year][day]['p'][p] += params.rho['ess']['esso'] * error_p_esso_tso
                    dual_vars['esso']['current']['tso'][node_id][year][day]['q'][p] += params.rho['ess']['esso'] * error_q_esso_tso
                    dual_vars['esso']['current']['dso'][node_id][year][day]['p'][p] += params.rho['ess']['esso'] * error_p_esso_dso
                    dual_vars['esso']['current']['dso'][node_id][year][day]['q'][p] += params.rho['ess']['esso'] * error_q_esso_dso

                    dual_vars['tso']['prev'][node_id][year][day]['p'][p] += params.rho['ess'][transmission_network.name] * error_p_tso_prev
                    dual_vars['tso']['prev'][node_id][year][day]['q'][p] += params.rho['ess'][transmission_network.name] * error_q_tso_prev
                    dual_vars['dso']['prev'][node_id][year][day]['p'][p] += params.rho['ess'][distribution_network.name] * error_p_dso_prev
                    dual_vars['dso']['prev'][node_id][year][day]['q'][p] += params.rho['ess'][distribution_network.name] * error_q_dso_prev
                    dual_vars['esso']['prev'][node_id][year][day]['p'][p] += params.rho['ess']['esso'] * error_p_esso_prev
                    dual_vars['esso']['prev'][node_id][year][day]['q'][p] += params.rho['ess']['esso'] * error_q_esso_prev


def check_admm_convergence(planning_problem, consensus_vars, params):
    if consensus_convergence(planning_problem, consensus_vars, params):
        if stationary_convergence(planning_problem, consensus_vars, params):
            return True
    return False


def consensus_convergence(planning_problem, consensus_vars, params):

    sum_abs = 0.0
    num_elems = 0

    for year in planning_problem.years:
        for day in planning_problem.days:

            # Interface Power Flow
            for node_id in planning_problem.active_distribution_network_nodes:
                for p in range(planning_problem.num_instants):
                    sum_abs += abs(round(consensus_vars['interface']['v_sqr']['tso']['current'][node_id][year][day][p], ERROR_PRECISION) - round(consensus_vars['interface']['v_sqr']['dso']['current'][node_id][year][day][p], ERROR_PRECISION))
                    sum_abs += abs(round(consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += abs(round(consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    num_elems += 6

            # Shared Energy Storage
            for node_id in planning_problem.active_distribution_network_nodes:
                for p in range(planning_problem.num_instants):
                    sum_abs += abs(round(consensus_vars['ess']['tso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['ess']['dso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += abs(round(consensus_vars['ess']['tso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['ess']['dso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    num_elems += 4

    if sum_abs > params.tol['consensus'] * num_elems:
        if not isclose(sum_abs, params.tol['consensus'] * num_elems, rel_tol=ADMM_CONVERGENCE_REL_TOL, abs_tol=params.tol['consensus']):
            print('[INFO]\t\t - Convergence consensus constraints failed. {:.3f} > {:.3f}'.format(sum_abs, params.tol['consensus'] * num_elems))
            return False
        print('[INFO]\t\t - Convergence consensus constraints considered ok. {:.3f} ~= {:.3f}'.format(sum_abs, params.tol['consensus'] * num_elems))
        return True

    print('[INFO]\t\t - Convergence consensus constraints ok. {:.3f} <= {:.3f}'.format(sum_abs, params.tol['consensus'] * num_elems))
    return True


def stationary_convergence(planning_problem, consensus_vars, params):

    rho_esso = params.rho['ess']['esso']
    rho_tso_v = params.rho['v'][planning_problem.transmission_network.name]
    rho_tso_pf = params.rho['pf'][planning_problem.transmission_network.name]
    rho_tso_ess = params.rho['ess'][planning_problem.transmission_network.name]
    sum_abs = 0.0
    num_elems = 0

    # Interface Power Flow
    for node_id in planning_problem.distribution_networks:
        rho_dso_v = params.rho['v'][planning_problem.distribution_networks[node_id].name]
        rho_dso_pf = params.rho['pf'][planning_problem.distribution_networks[node_id].name]
        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in range(planning_problem.num_instants):
                    sum_abs += rho_tso_v * abs(round(consensus_vars['interface']['v_sqr']['tso']['current'][node_id][year][day][p], ERROR_PRECISION) - round(consensus_vars['interface']['v_sqr']['tso']['prev'][node_id][year][day][p], ERROR_PRECISION))
                    sum_abs += rho_tso_pf * abs(round(consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['interface']['pf']['tso']['prev'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += rho_tso_pf * abs(round(consensus_vars['interface']['pf']['tso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['interface']['pf']['tso']['prev'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    sum_abs += rho_dso_v * abs(round(consensus_vars['interface']['v_sqr']['dso']['current'][node_id][year][day][p], ERROR_PRECISION) - round(consensus_vars['interface']['v_sqr']['dso']['prev'][node_id][year][day][p], ERROR_PRECISION))
                    sum_abs += rho_dso_pf * abs(round(consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['interface']['pf']['dso']['prev'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += rho_dso_pf * abs(round(consensus_vars['interface']['pf']['dso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['interface']['pf']['dso']['prev'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    num_elems += 6

    # Shared Energy Storage
    for node_id in planning_problem.distribution_networks:
        distribution_network = planning_problem.distribution_networks[node_id]
        for year in planning_problem.years:
            for day in planning_problem.days:
                rho_dso_ess = params.rho['ess'][distribution_network.network[year][day].name]
                for p in range(planning_problem.num_instants):
                    sum_abs += rho_tso_ess * abs(round(consensus_vars['ess']['tso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['ess']['tso']['prev'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += rho_tso_ess * abs(round(consensus_vars['ess']['tso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['ess']['tso']['prev'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    sum_abs += rho_dso_ess * abs(round(consensus_vars['ess']['dso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['ess']['dso']['prev'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += rho_dso_ess * abs(round(consensus_vars['ess']['dso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['ess']['dso']['prev'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    sum_abs += rho_esso * abs(round(consensus_vars['ess']['esso']['current'][node_id][year][day]['p'][p], ERROR_PRECISION) - round(consensus_vars['ess']['esso']['prev'][node_id][year][day]['p'][p], ERROR_PRECISION))
                    sum_abs += rho_esso * abs(round(consensus_vars['ess']['esso']['current'][node_id][year][day]['q'][p], ERROR_PRECISION) - round(consensus_vars['ess']['esso']['prev'][node_id][year][day]['q'][p], ERROR_PRECISION))
                    num_elems += 6

    if sum_abs > params.tol['stationarity'] * num_elems:
        if not isclose(sum_abs, params.tol['stationarity'] * num_elems, rel_tol=ADMM_CONVERGENCE_REL_TOL, abs_tol=params.tol['stationarity']):
            print('[INFO]\t\t - Convergence stationary constraints failed. {:.3f} > {:.3f}'.format(sum_abs, params.tol['stationarity'] * num_elems))
            return False
        print('[INFO]\t\t - Convergence stationary constraints considered ok. {:.3f} ~= {:.3f}'.format(sum_abs, params.tol['stationarity'] * num_elems))
        return True

    print('[INFO]\t\t - Convergence stationary constraints ok. {:.3f} <= {:.3f}'.format(sum_abs, params.tol['stationarity'] * num_elems))
    return True


def update_transmission_model_to_admm(transmission_network, model, initial_interface_pf, params):

    for year in transmission_network.years:
        for day in transmission_network.days:

            init_of_value = 1.00
            if transmission_network.params.obj_type == OBJ_MIN_COST:
                init_of_value = pe.value(model[year][day].objective)

            s_base = transmission_network.network[year][day].baseMVA

            # Free expected values
            for dn in model[year][day].active_distribution_networks:
                for p in model[year][day].periods:
                    model[year][day].expected_interface_vmag_sqr[dn, p].fixed = False
                    model[year][day].expected_interface_pf_p[dn, p].fixed = False
                    model[year][day].expected_interface_pf_q[dn, p].fixed = False
                    model[year][day].expected_shared_ess_p[dn, p].fixed = False
                    model[year][day].expected_shared_ess_q[dn, p].fixed = False

            # Add ADMM variables
            model[year][day].rho_v = pe.Var(domain=pe.NonNegativeReals)
            model[year][day].rho_v.fix(params.rho['v'][transmission_network.name])
            model[year][day].v_sqr_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.NonNegativeReals)
            model[year][day].dual_v_sqr_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)   # Dual variable - active power requested

            model[year][day].rho_pf = pe.Var(domain=pe.NonNegativeReals)
            model[year][day].rho_pf.fix(params.rho['pf'][transmission_network.name])
            model[year][day].p_pf_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)        # Active power - requested by distribution networks
            model[year][day].q_pf_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)        # Reactive power - requested by distribution networks
            model[year][day].dual_pf_p_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)   # Dual variable - active power requested
            model[year][day].dual_pf_q_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)   # Dual variable - reactive power requested

            model[year][day].rho_ess = pe.Var(domain=pe.NonNegativeReals)
            model[year][day].rho_ess.fix(params.rho['ess'][transmission_network.name])
            model[year][day].p_ess_req = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)             # Shared ESS - Active power requested (DSO)
            model[year][day].q_ess_req = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)             # Shared ESS - Reactive power requested (DSO)
            model[year][day].dual_ess_p_req = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)        # Dual variable - Shared ESS active power
            model[year][day].dual_ess_q_req = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)        # Dual variable - Shared ESS active power

            # Objective function - augmented Lagrangian
            obj = model[year][day].objective.expr / abs(init_of_value)
            for dn in model[year][day].active_distribution_networks:
                node_id = transmission_network.active_distribution_network_nodes[dn]
                for p in model[year][day].periods:

                    init_p = initial_interface_pf['dso']['current'][node_id][year][day]['p'][p] / s_base
                    init_q = initial_interface_pf['dso']['current'][node_id][year][day]['q'][p] / s_base

                    constraint_v_req = (model[year][day].expected_interface_vmag_sqr[dn, p] - model[year][day].v_sqr_req[dn, p])
                    constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p]) / abs(init_p)
                    constraint_q_req = (model[year][day].expected_interface_pf_q[dn, p] - model[year][day].q_pf_req[dn, p]) / abs(init_q)

                    obj += model[year][day].dual_v_sqr_req[dn, p] * constraint_v_req
                    obj += model[year][day].dual_pf_p_req[dn, p] * constraint_p_req
                    obj += model[year][day].dual_pf_q_req[dn, p] * constraint_q_req
                    obj += (model[year][day].rho_pf / 2) * (constraint_v_req ** 2)
                    obj += (model[year][day].rho_pf / 2) * (constraint_p_req ** 2)
                    obj += (model[year][day].rho_pf / 2) * (constraint_q_req ** 2)

            for e in model[year][day].active_distribution_networks:
                rating = transmission_network.network[year][day].shared_energy_storages[e].s
                if isclose(rating, 0.00, abs_tol=SMALL_TOLERANCE):
                    rating = 1.00       # Do not balance residuals
                for p in model[year][day].periods:
                    constraint_ess_p = (model[year][day].expected_shared_ess_p[e, p] - model[year][day].p_ess_req[e, p]) / (2 * rating)
                    constraint_ess_q = (model[year][day].expected_shared_ess_q[e, p] - model[year][day].q_ess_req[e, p]) / (2 * rating)
                    obj += model[year][day].dual_ess_p_req[e, p] * constraint_ess_p
                    obj += model[year][day].dual_ess_q_req[e, p] * constraint_ess_q
                    obj += (model[year][day].rho_ess / 2) * constraint_ess_p ** 2
                    obj += (model[year][day].rho_ess / 2) * constraint_ess_q ** 2

            model[year][day].objective.expr = obj


def update_distribution_models_to_admm(distribution_networks, models, initial_interface_pf, params):

    for node_id in distribution_networks:

        dso_model = models[node_id]
        distribution_network = distribution_networks[node_id]

        # Free voltage at the connection point with the transmission network
        # Free Pg and Qg at the connection point with the transmission network
        for year in distribution_network.years:
            for day in distribution_network.days:

                s_base = distribution_network.network[year][day].baseMVA
                ref_node_id = distribution_network.network[year][day].get_reference_node_id()

                init_of_value = 1.00
                if distribution_network.params.obj_type == OBJ_MIN_COST:
                    init_of_value = pe.value(dso_model[year][day].objective)

                # Free voltage at the interface node
                ref_node_idx = distribution_network.network[year][day].get_node_idx(ref_node_id)
                for s_m in dso_model[year][day].scenarios_market:
                    for s_o in dso_model[year][day].scenarios_operation:
                        for p in dso_model[year][day].periods:
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].fixed = False

                # Add ADMM variables
                dso_model[year][day].rho_v = pe.Var(domain=pe.NonNegativeReals)
                dso_model[year][day].rho_v.fix(params.rho['v'][distribution_network.network[year][day].name])
                dso_model[year][day].v_sqr_req = pe.Var(dso_model[year][day].periods, domain=pe.NonNegativeReals)       # Voltage magnitude - requested by TSO
                dso_model[year][day].dual_v_sqr_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)             # Dual variable - voltage magnitude

                dso_model[year][day].rho_pf = pe.Var(domain=pe.NonNegativeReals)
                dso_model[year][day].rho_pf.fix(params.rho['pf'][distribution_network.network[year][day].name])
                dso_model[year][day].p_pf_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)                   # Active power - requested by TSO
                dso_model[year][day].q_pf_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)                   # Reactive power - requested by TSO
                dso_model[year][day].dual_pf_p_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)              # Dual variable - active power
                dso_model[year][day].dual_pf_q_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)              # Dual variable - reactive power

                dso_model[year][day].rho_ess = pe.Var(domain=pe.NonNegativeReals)
                dso_model[year][day].rho_ess.fix(params.rho['ess'][distribution_network.network[year][day].name])
                dso_model[year][day].p_ess_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)                  # Shared ESS - active power requested (TSO)
                dso_model[year][day].q_ess_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)                  # Shared ESS - reactive power requested (TSO)
                dso_model[year][day].dual_ess_p_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)             # Dual variable - Shared ESS active power
                dso_model[year][day].dual_ess_q_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)             # Dual variable - Shared ESS reactive power

                if params.use_previous_iter:
                    dso_model[year][day].v_sqr_prev = pe.Var(dso_model[year][day].periods, domain=pe.NonNegativeReals)  # Voltage magnitude
                    dso_model[year][day].dual_v_sqr_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)        # Dual variable
                    dso_model[year][day].p_pf_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)              # Active power
                    dso_model[year][day].q_pf_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)              # Reactive power
                    dso_model[year][day].dual_pf_p_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)         # Dual variable - active power
                    dso_model[year][day].dual_pf_q_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)         # Dual variable - reactive power
                    dso_model[year][day].p_ess_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)             # Shared ESS - active power
                    dso_model[year][day].q_ess_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)             # Shared ESS - reactive power
                    dso_model[year][day].dual_ess_p_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)        # Dual variable - Shared ESS active power
                    dso_model[year][day].dual_ess_q_prev = pe.Var(dso_model[year][day].periods, domain=pe.Reals)        # Dual variable - Shared ESS reactive power

                # Objective function - augmented Lagrangian
                obj = dso_model[year][day].objective.expr / max(abs(init_of_value), 1.00)

                # Augmented Lagrangian -- Interface power flow (residual balancing)
                for p in dso_model[year][day].periods:

                    init_pf_p = initial_interface_pf['current'][node_id][year][day]['p'][p] / s_base
                    init_pf_q = initial_interface_pf['current'][node_id][year][day]['q'][p] / s_base

                    constraint_vmag_req = (dso_model[year][day].expected_interface_vmag_sqr[p] - dso_model[year][day].v_sqr_req[p])
                    constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p]) / abs(init_pf_p)
                    constraint_q_req = (dso_model[year][day].expected_interface_pf_q[p] - dso_model[year][day].q_pf_req[p]) / abs(init_pf_q)

                    obj += (dso_model[year][day].dual_v_sqr_req[p]) * constraint_vmag_req
                    obj += (dso_model[year][day].dual_pf_p_req[p]) * constraint_p_req
                    obj += (dso_model[year][day].dual_pf_q_req[p]) * constraint_q_req
                    obj += (dso_model[year][day].rho_v / 2) * (constraint_vmag_req ** 2)
                    obj += (dso_model[year][day].rho_pf / 2) * (constraint_p_req ** 2)
                    obj += (dso_model[year][day].rho_pf / 2) * (constraint_q_req ** 2)

                    if params.use_previous_iter:
                        constraint_vmag_prev = (dso_model[year][day].expected_interface_vmag_sqr[p] - dso_model[year][day].v_sqr_prev[p])
                        constraint_p_prev = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_prev[p]) / abs(init_pf_p)
                        constraint_q_prev = (dso_model[year][day].expected_interface_pf_q[p] - dso_model[year][day].q_pf_prev[p]) / abs(init_pf_q)

                        obj += (dso_model[year][day].dual_v_sqr_prev[p]) * constraint_vmag_prev
                        obj += (dso_model[year][day].dual_pf_p_prev[p]) * constraint_p_prev
                        obj += (dso_model[year][day].dual_pf_q_prev[p]) * constraint_q_prev
                        obj += (dso_model[year][day].rho_v / 2) * (constraint_vmag_prev ** 2)
                        obj += (dso_model[year][day].rho_pf / 2) * (constraint_p_prev ** 2)
                        obj += (dso_model[year][day].rho_pf / 2) * (constraint_q_prev ** 2)

                # Augmented Lagrangian -- Shared ESS (residual balancing)
                sess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)
                sess_rating = pe.value(dso_model[year][day].shared_es_s_rated_fixed[sess_idx])
                if isclose(sess_rating, 0.00, abs_tol=SMALL_TOLERANCE):  # Do not balance residuals
                    sess_rating = 1.00
                for p in dso_model[year][day].periods:

                    constraint_ess_p_req = (dso_model[year][day].expected_shared_ess_p[p] - dso_model[year][day].p_ess_req[p]) / (2 * sess_rating)
                    constraint_ess_q_req = (dso_model[year][day].expected_shared_ess_q[p] - dso_model[year][day].q_ess_req[p]) / (2 * sess_rating)

                    obj += dso_model[year][day].dual_ess_p_req[p] * constraint_ess_p_req
                    obj += dso_model[year][day].dual_ess_q_req[p] * constraint_ess_q_req
                    obj += (dso_model[year][day].rho_ess / 2) * constraint_ess_p_req ** 2
                    obj += (dso_model[year][day].rho_ess / 2) * constraint_ess_q_req ** 2

                    if params.use_previous_iter:
                        constraint_ess_p_prev = (dso_model[year][day].expected_shared_ess_p[p] - dso_model[year][day].p_ess_prev[p]) / (2 * sess_rating)
                        constraint_ess_q_prev = (dso_model[year][day].expected_shared_ess_q[p] - dso_model[year][day].q_ess_prev[p]) / (2 * sess_rating)

                        obj += dso_model[year][day].dual_ess_p_prev[p] * constraint_ess_p_prev
                        obj += dso_model[year][day].dual_ess_q_prev[p] * constraint_ess_q_prev
                        obj += (dso_model[year][day].rho_ess / 2) * constraint_ess_p_prev ** 2
                        obj += (dso_model[year][day].rho_ess / 2) * constraint_ess_q_prev ** 2

                dso_model[year][day].objective.expr = obj


def update_shared_energy_storage_model_to_admm(shared_ess_data, model, params):

    # Add ADMM variables
    model.rho = pe.Var(domain=pe.NonNegativeReals)
    model.rho.fix(params.rho['ess']['esso'])

    # Active and Reactive power requested by TSO and DSOs
    model.p_req = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)            # Active power
    model.q_req = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)            # Reactive power
    model.dual_p_req = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)       # Dual variables
    model.dual_q_req = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)       # Dual variables

    # Objective function - augmented Lagrangian
    obj = model.objective.expr
    for e in model.energy_storages:
        for y in model.years:
            rating_s = pe.value(model.es_s_rated[e, y])
            if isclose(rating_s, 0.0, rel_tol=SMALL_TOLERANCE):
                rating_s = 1.00     # Do not balance residuals
            for d in model.days:
                for p in model.periods:
                    constraint_p_req = (model.es_pnet[e, y, d, p] - model.p_req[e, y, d, p]) / (2 * rating_s)
                    constraint_q_req = (model.es_qnet[e, y, d, p] - model.q_req[e, y, d, p]) / (2 * rating_s)
                    obj += model.dual_p_req[e, y, d, p] * constraint_p_req
                    obj += model.dual_q_req[e, y, d, p] * constraint_q_req
                    obj += (model.rho / 2) * constraint_p_req ** 2
                    obj += (model.rho / 2) * constraint_q_req ** 2

    model.objective.expr = obj

    return model


def update_transmission_coordination_model_and_solve(transmission_network, model, vsqr_req, dual_vsqr, pf_req, dual_pf, ess_req, dual_ess, params, from_warm_start=False):

    print('[INFO] \t\t - Updating transmission network...')

    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA

            rho_v = params.rho['v'][transmission_network.name]
            rho_pf = params.rho['pf'][transmission_network.name]
            rho_ess = params.rho['ess'][transmission_network.name]
            if params.adaptive_penalty:
                rho_v = pe.value(model[year][day].rho_v) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
                rho_pf = pe.value(model[year][day].rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
                rho_ess = pe.value(model[year][day].rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)

            # Update Rho parameter
            model[year][day].rho_v.fix(rho_v)
            model[year][day].rho_pf.fix(rho_pf)
            model[year][day].rho_ess.fix(rho_ess)

            for dn in model[year][day].active_distribution_networks:

                node_id = transmission_network.active_distribution_network_nodes[dn]

                # Update VOLTAGE and POWER FLOW variables at connection point
                for p in model[year][day].periods:
                    model[year][day].dual_v_sqr_req[dn, p].fix(dual_vsqr['current'][node_id][year][day][p] / s_base)
                    model[year][day].dual_pf_p_req[dn, p].fix(dual_pf['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_pf_q_req[dn, p].fix(dual_pf['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].v_sqr_req[dn, p].fix(vsqr_req['dso']['current'][node_id][year][day][p])
                    model[year][day].p_pf_req[dn, p].fix(pf_req['dso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_pf_req[dn, p].fix(pf_req['dso']['current'][node_id][year][day]['q'][p] / s_base)

                # Update shared ESS capacity and power requests
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                for p in model[year][day].periods:
                    model[year][day].dual_ess_p_req[shared_ess_idx, p].fix(dual_ess['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_ess_q_req[shared_ess_idx, p].fix(dual_ess['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_ess_req[shared_ess_idx, p].fix(ess_req['esso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_ess_req[shared_ess_idx, p].fix(ess_req['esso']['current'][node_id][year][day]['q'][p] / s_base)

    # Solve!
    res = transmission_network.optimize(model, from_warm_start=from_warm_start)
    for year in transmission_network.years:
        for day in transmission_network.days:
            if res[year][day].solver.status == po.SolverStatus.error:
                print(f'[ERROR] Network {model[year][day].name} did not converge!')
                # exit(ERROR_NETWORK_OPTIMIZATION)
    return res


def update_distribution_coordination_models_and_solve(distribution_networks, models, vsqr_req, dual_vsqr, pf_req, dual_pf, ess_req, dual_ess, params, from_warm_start=False):

    print('[INFO] \t\t - Updating distribution networks:')
    res = dict()

    for node_id in distribution_networks:

        model = models[node_id]
        distribution_network = distribution_networks[node_id]

        print('[INFO] \t\t\t - Updating active distribution network connected to node {}...'.format(node_id))

        for year in distribution_network.years:
            for day in distribution_network.days:

                s_base = distribution_network.network[year][day].baseMVA

                rho_v = params.rho['v'][distribution_network.name]
                rho_pf = params.rho['pf'][distribution_network.name]
                rho_ess = params.rho['ess'][distribution_network.name]
                if params.adaptive_penalty:
                    rho_v = pe.value(model[year][day].rho_v) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
                    rho_pf = pe.value(model[year][day].rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
                    rho_ess = pe.value(model[year][day].rho_ess) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)

                # Update Rho parameter
                model[year][day].rho_v.fix(rho_v)
                model[year][day].rho_pf.fix(rho_pf)
                model[year][day].rho_ess.fix(rho_ess)

                # Update VOLTAGE and POWER FLOW variables at connection point
                for p in model[year][day].periods:

                    model[year][day].dual_v_sqr_req[p].fix(dual_vsqr['current'][node_id][year][day][p] / s_base)
                    model[year][day].dual_pf_p_req[p].fix(dual_pf['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_pf_q_req[p].fix(dual_pf['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].v_sqr_req[p].fix(vsqr_req['tso']['current'][node_id][year][day][p])
                    model[year][day].p_pf_req[p].fix(pf_req['tso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_pf_req[p].fix(pf_req['tso']['current'][node_id][year][day]['q'][p] / s_base)

                    if params.use_previous_iter:
                        model[year][day].dual_v_sqr_prev[p].fix(dual_vsqr['prev'][node_id][year][day][p] / s_base)
                        model[year][day].dual_pf_p_prev[p].fix(dual_pf['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].dual_pf_q_prev[p].fix(dual_pf['prev'][node_id][year][day]['q'][p] / s_base)
                        model[year][day].v_sqr_prev[p].fix(vsqr_req['dso']['prev'][node_id][year][day][p])
                        model[year][day].p_pf_prev[p].fix(pf_req['dso']['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].q_pf_prev[p].fix(pf_req['dso']['prev'][node_id][year][day]['q'][p] / s_base)

                # Update SHARED ENERGY STORAGE variables (if existent)
                for p in model[year][day].periods:

                    model[year][day].dual_ess_p_req[p].fix(dual_ess['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_ess_q_req[p].fix(dual_ess['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_ess_req[p].fix(ess_req['esso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_ess_req[p].fix(ess_req['esso']['current'][node_id][year][day]['q'][p] / s_base)

                    if params.use_previous_iter:
                        model[year][day].dual_ess_p_prev[p].fix(dual_ess['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].dual_ess_q_prev[p].fix(dual_ess['prev'][node_id][year][day]['q'][p] / s_base)
                        model[year][day].p_ess_prev[p].fix(ess_req['dso']['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].q_ess_prev[p].fix(ess_req['dso']['prev'][node_id][year][day]['q'][p] / s_base)

        # Solve!
        res[node_id] = distribution_network.optimize(model, from_warm_start=from_warm_start)
        for year in distribution_network.years:
            for day in distribution_network.days:
                if res[node_id][year][day].solver.status != po.SolverStatus.ok:
                    print(f'[WARNING] Network {model[year][day].name} did not converge!')
                    #exit(ERROR_NETWORK_OPTIMIZATION)
    return res


def update_shared_energy_storages_coordination_model_and_solve(planning_problem, model, ess_req, dual_ess, params, from_warm_start=False):

    print('[INFO] \t\t - Updating Shared ESS...')
    shared_ess_data = planning_problem.shared_ess_data
    days = [day for day in planning_problem.days]
    years = [year for year in planning_problem.years]

    rho_esso = params.rho['ess']['esso']
    if params.adaptive_penalty:
        rho_esso = pe.value(model.rho) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
    model.rho.fix(rho_esso)

    for e in model.energy_storages:
        for y in model.years:
            year = years[y]
            node_id = shared_ess_data.shared_energy_storages[year][e].bus
            for d in model.days:
                day = days[d]
                for p in model.periods:

                    p_req = ess_req[node_id][year][day]['p'][p]
                    q_req = ess_req[node_id][year][day]['q'][p]
                    dual_p_req = dual_ess[node_id][year][day]['p'][p]
                    dual_q_req = dual_ess[node_id][year][day]['q'][p]

                    model.p_req[e, y, d, p].fix(p_req)
                    model.q_req[e, y, d, p].fix(q_req)
                    model.dual_p_req[e, y, d, p].fix(dual_p_req)
                    model.dual_q_req[e, y, d, p].fix(dual_q_req)

    # Solve!
    res = shared_ess_data.optimize(model, from_warm_start=from_warm_start)
    if res.solver.status != po.SolverStatus.ok:
        print('[WARNING] Shared ESS operational planning did not converge!')

    return res


# ======================================================================================================================
#  OPERATIONAL PLANNING WITHOUT COORDINATION functions
# ======================================================================================================================
def _run_operational_planning_without_coordination(planning_problem):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    results = {'tso': dict(), 'dso': dict(), 'esso': dict()}

    # Do not consider flexible resources
    transmission_network.params.fl_reg = False
    transmission_network.params.es_reg = True
    transmission_network.params.transf_reg = True
    transmission_network.params.rg_curt = True
    transmission_network.params.l_curt = True
    transmission_network.params.slacks = True
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        distribution_network.params.fl_reg = False
        distribution_network.params.es_reg = True
        distribution_network.params.transf_reg = True
        distribution_network.params.rg_curt = True
        distribution_network.params.l_curt = True
        distribution_network.params.slacks = True

    # Shared ESS candidate solution (no shared ESS)
    candidate_solution = dict()
    for e in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[e]
        candidate_solution[node_id] = dict()
        for year in planning_problem.years:
            candidate_solution[node_id][year] = dict()
            candidate_solution[node_id][year]['s'] = 0.00
            candidate_solution[node_id][year]['e'] = 0.00

    # Create interface PF variables
    interface_pf = create_interface_power_flow_variables(planning_problem)

    # Create DSOs' Operational Planning models
    dso_models = dict()
    for node_id in distribution_networks:

        distribution_network = distribution_networks[node_id]
        results['dso'][node_id] = dict()

        # Build model, fix candidate solution, and Run S-MPOPF model
        dso_model = distribution_network.build_model()
        distribution_network.update_model_with_candidate_solution(dso_model, candidate_solution)
        results['dso'][node_id] = distribution_network.optimize(dso_model)

        # Get initial interface PF values
        for year in distribution_network.years:
            for day in distribution_network.days:
                s_base = distribution_network.network[year][day].baseMVA
                for p in dso_model[year][day].periods:
                    interface_pf[node_id][year][day]['p'][p] = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    interface_pf[node_id][year][day]['q'][p] = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base

        dso_models[node_id] = dso_model

    # Create TSO Operational Planning model
    tso_model = transmission_network.build_model()
    transmission_network.update_model_with_candidate_solution(tso_model, candidate_solution)
    for node_id in transmission_network.active_distribution_network_nodes:
        for year in transmission_network.years:
            for day in transmission_network.days:

                node_idx = transmission_network.network[year][day].get_node_idx(node_id)
                s_base = transmission_network.network[year][day].baseMVA

                # - Fix expected interface PF
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            pc = interface_pf[node_id][year][day]['p'][p] / s_base
                            qc = interface_pf[node_id][year][day]['q'][p] / s_base
                            tso_model[year][day].pc[node_idx, s_m, s_o, p].fix(pc)
                            tso_model[year][day].qc[node_idx, s_m, s_o, p].fix(qc)
                            if transmission_network.params.fl_reg:
                                tso_model[year][day].flex_p_up[node_idx, s_m, s_o, p].fix(0.0)
                                tso_model[year][day].flex_p_down[node_idx, s_m, s_o, p].fix(0.0)

    results['tso'] = transmission_network.optimize(tso_model)

    models = {'tso': tso_model, 'dso': dso_models}

    return results, models


def create_interface_power_flow_variables(planning_problem):
    consensus_vars, _ = create_admm_variables(planning_problem)
    return consensus_vars['interface']['pf']['dso']['current']


# ======================================================================================================================
#  PLANNING PROBLEM read functions
# ======================================================================================================================
def _read_planning_problem(planning_problem):

    # Create results folder
    if not os.path.exists(planning_problem.results_dir):
        os.makedirs(planning_problem.results_dir)

    # Create diagrams folder
    if not os.path.exists(planning_problem.diagrams_dir):
        os.makedirs(planning_problem.diagrams_dir)

    # Read specification file
    filename = os.path.join(planning_problem.data_dir, planning_problem.filename)
    planning_data = convert_json_to_dict(read_json_file(filename))

    # General Parameters
    for year in planning_data['Years']:
        planning_problem.years[int(year)] = planning_data['Years'][year]
    planning_problem.days = planning_data['Days']
    planning_problem.num_instants = planning_data['NumInstants']

    # Market Data
    planning_problem.discount_factor = planning_data['DiscountFactor']
    planning_problem.market_data_file = planning_data['MarketData']
    planning_problem.read_market_data_from_file()

    # Distribution Networks
    for distribution_network in planning_data['DistributionNetworks']:

        print('[INFO] Reading DISTRIBUTION NETWORK DATA from file(s)...')

        network_name = distribution_network['name']                         # Network filename
        params_file = distribution_network['params_file']                   # Params filename
        connection_nodeid = distribution_network['connection_node_id']      # Connection node ID

        distribution_network = NetworkData()
        distribution_network.name = network_name
        distribution_network.is_transmission = False
        distribution_network.data_dir = planning_problem.data_dir
        distribution_network.results_dir = planning_problem.results_dir
        distribution_network.diagrams_dir = planning_problem.diagrams_dir
        distribution_network.years = planning_problem.years
        distribution_network.days = planning_problem.days
        distribution_network.num_instants = planning_problem.num_instants
        distribution_network.discount_factor = planning_problem.discount_factor
        distribution_network.prob_market_scenarios = planning_problem.prob_market_scenarios
        distribution_network.cost_energy_p = planning_problem.cost_energy_p
        distribution_network.cost_flex = planning_problem.cost_flex
        distribution_network.params_file = params_file
        distribution_network.read_network_parameters()
        if distribution_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
            distribution_network.prob_market_scenarios = [1.00]
        distribution_network.read_network_data()
        distribution_network.tn_connection_nodeid = connection_nodeid
        planning_problem.distribution_networks[connection_nodeid] = distribution_network
    planning_problem.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]

    # Transmission Network
    print('[INFO] Reading TRANSMISSION NETWORK DATA from file(s)...')
    transmission_network = NetworkData()
    transmission_network.name = planning_data['TransmissionNetwork']['name']
    transmission_network.is_transmission = True
    transmission_network.data_dir = planning_problem.data_dir
    transmission_network.results_dir = planning_problem.results_dir
    transmission_network.diagrams_dir = planning_problem.diagrams_dir
    transmission_network.years = planning_problem.years
    transmission_network.days = planning_problem.days
    transmission_network.num_instants = planning_problem.num_instants
    transmission_network.discount_factor = planning_problem.discount_factor
    transmission_network.prob_market_scenarios = planning_problem.prob_market_scenarios
    transmission_network.cost_energy_p = planning_problem.cost_energy_p
    transmission_network.cost_flex = planning_problem.cost_flex
    transmission_network.params_file = planning_data['TransmissionNetwork']['params_file']
    transmission_network.read_network_parameters()
    if transmission_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        transmission_network.prob_market_scenarios = [1.00]
    transmission_network.read_network_data()
    transmission_network.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]
    for year in transmission_network.years:
        for day in transmission_network.days:
            transmission_network.network[year][day].active_distribution_network_nodes = transmission_network.active_distribution_network_nodes
    planning_problem.transmission_network = transmission_network

    # Shared ESS
    shared_ess_data = SharedEnergyStorageData()
    shared_ess_data.name = planning_problem.name
    shared_ess_data.data_dir = planning_problem.data_dir
    shared_ess_data.results_dir = planning_problem.results_dir
    shared_ess_data.years = planning_problem.years
    shared_ess_data.days = planning_problem.days
    shared_ess_data.num_instants = planning_problem.num_instants
    shared_ess_data.discount_factor = planning_problem.discount_factor
    shared_ess_data.prob_market_scenarios = planning_problem.prob_market_scenarios
    shared_ess_data.cost_energy_p = planning_problem.cost_energy_p
    shared_ess_data.params_file = planning_data['SharedEnergyStorage']['params_file']
    shared_ess_data.read_parameters_from_file()
    shared_ess_data.create_shared_energy_storages(planning_problem)
    shared_ess_data.data_file = planning_data['SharedEnergyStorage']['data_file']
    shared_ess_data.read_shared_energy_storage_data_from_file()
    shared_ess_data.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]
    planning_problem.shared_ess_data = shared_ess_data

    # Planning Parameters
    planning_problem.params_file = planning_data['PlanningParameters']['params_file']
    planning_problem.read_planning_parameters_from_file()

    # Add ADN nodes to Transmission Network
    _add_adn_node_to_transmission_network(planning_problem)

    # Add Shared Energy Storages to Transmission and Distribution Networks
    _add_shared_energy_storage_to_transmission_network(planning_problem)
    _add_shared_energy_storage_to_distribution_network(planning_problem)


# ======================================================================================================================
#  MARKET DATA read functions
# ======================================================================================================================
def _read_market_data_from_file(planning_problem):

    try:
        for year in planning_problem.years:
            filename = os.path.join(planning_problem.data_dir, 'Market Data', f'{planning_problem.market_data_file}_{year}.xlsx')
            num_scenarios, prob_scenarios = _get_market_scenarios_info_from_excel_file(filename, 'Scenarios')
            planning_problem.prob_market_scenarios = prob_scenarios
            planning_problem.cost_energy_p[year] = dict()
            planning_problem.cost_flex[year] = dict()
            for day in planning_problem.days:
                planning_problem.cost_energy_p[year][day] = _get_market_costs_from_excel_file(filename, f'Cp, {day}', num_scenarios)
                planning_problem.cost_flex[year][day] = _get_market_costs_from_excel_file(filename, f'Flex, {day}', num_scenarios)
    except:
        print(f'[ERROR] Reading market data from file(s). Exiting...')
        exit(ERROR_SPECIFICATION_FILE)


def _get_market_scenarios_info_from_excel_file(filename, sheet_name):

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
        exit(ERROR_MARKET_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_market_costs_from_excel_file(filename, sheet_name, num_scenarios):
    data = pd.read_excel(filename, sheet_name=sheet_name)
    _, num_cols = data.shape
    cost_values = dict()
    scn_idx = 0
    for i in range(num_scenarios):
        cost_values_scenario = list()
        for j in range(num_cols - 1):
            cost_values_scenario.append(float(data.iloc[i, j + 1]))
        cost_values[scn_idx] = cost_values_scenario
        scn_idx = scn_idx + 1
    return cost_values


# ======================================================================================================================
#  RESULTS PROCESSING functions
# ======================================================================================================================
def _process_operational_planning_results(operational_planning_problem, tso_model, dso_models, esso_model, optimization_results):

    transmission_network = operational_planning_problem.transmission_network
    distribution_networks = operational_planning_problem.distribution_networks
    shared_ess_data = operational_planning_problem.shared_ess_data

    processed_results = dict()
    processed_results['tso'] = dict()
    processed_results['dso'] = dict()
    processed_results['esso'] = dict()
    processed_results['interface'] = dict()

    processed_results['tso'] = transmission_network.process_results(tso_model, optimization_results['tso'])
    for node_id in distribution_networks:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        processed_results['dso'][node_id] = distribution_network.process_results(dso_model, optimization_results['dso'][node_id])
    processed_results['esso'] = shared_ess_data.process_results(esso_model)
    processed_results['interface'] = _process_results_interface(operational_planning_problem, tso_model, dso_models)

    return processed_results


def _process_operational_planning_results_no_coordination(planning_problem, tso_model, dso_models, optimization_results):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    processed_results = dict()
    processed_results['tso'] = dict()
    processed_results['dso'] = dict()

    processed_results['tso'] = transmission_network.process_results(tso_model, optimization_results['tso'])
    for node_id in distribution_networks:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        processed_results['dso'][node_id] = distribution_network.process_results(dso_model, optimization_results['dso'][node_id])

    return processed_results


def _process_results_interface(planning_problem, tso_model, dso_models):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    processed_results = dict()
    processed_results['tso'] = dict()
    processed_results['dso'] = dict()

    processed_results['tso'] = transmission_network.process_results_interface(tso_model)
    for node_id in distribution_networks:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        processed_results['dso'][node_id] = distribution_network.process_results_interface(dso_model)

    return processed_results


# ======================================================================================================================
#  RESULTS PLANNING - write functions
# ======================================================================================================================
def _write_planning_results_to_excel(planning_problem, results, bound_evolution=dict(), shared_ess_capacity=dict(), filename='planing_results'):

    wb = Workbook()

    _write_operational_planning_main_info_to_excel(planning_problem, wb, results)
    _write_shared_ess_specifications(wb, planning_problem.shared_ess_data)

    if bound_evolution:
        _write_bound_evolution_to_excel(wb, bound_evolution)

    if shared_ess_capacity:
        planning_problem.shared_ess_data.write_ess_results_to_excel(wb, shared_ess_capacity)

    # Interface Power Flow
    _write_interface_results_to_excel(planning_problem, wb, results['interface'])

    # Shared Energy Storages results
    _write_shared_energy_storages_results_to_excel(planning_problem, wb, results)

    #  TSO and DSOs' results
    _write_network_voltage_results_to_excel(planning_problem, wb, results)
    _write_network_consumption_results_to_excel(planning_problem, wb, results)
    _write_network_generation_results_to_excel(planning_problem, wb, results)
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'losses')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'ratio')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'current_perc')
    _write_network_branch_power_flow_results_to_excel(planning_problem, wb, results)
    _write_network_energy_storages_results_to_excel(planning_problem, wb, results)
    planning_problem.shared_ess_data.write_relaxation_slacks_results_to_excel(wb, results['esso'])

    # Save results
    try:
        wb.save(filename)
    except:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = f"{filename.replace('.xlsx', '')}_{current_time}.xlsx"
        print(f"[WARNING] Results saved to file {backup_filename}.xlsx")
        wb.save(backup_filename)


def _write_bound_evolution_to_excel(workbook, bound_evolution):

    sheet = workbook.create_sheet('Convergence Characteristic')

    lower_bound = bound_evolution['lower_bound']
    upper_bound = bound_evolution['upper_bound']
    num_lines = max(len(upper_bound), len(lower_bound))

    num_style = '0.00'

    # Write header
    line_idx = 1
    sheet.cell(row=line_idx, column=1).value = 'Iteration'
    sheet.cell(row=line_idx, column=2).value = 'Lower Bound, [NPV Mm.u.]'
    sheet.cell(row=line_idx, column=3).value = 'Upper Bound, [NPV Mm.u.]'

    # Iterations
    line_idx = 2
    for i in range(num_lines):
        sheet.cell(row=line_idx, column=1).value = i
        line_idx += 1

    # Lower bound
    line_idx = 2
    for value in lower_bound:
        sheet.cell(row=line_idx, column=2).value = value / 1e6
        sheet.cell(row=line_idx, column=2).number_format = num_style
        line_idx += 1

    # Upper bound
    line_idx = 2
    for value in upper_bound:
        sheet.cell(row=line_idx, column=3).value = value / 1e6
        sheet.cell(row=line_idx, column=3).number_format = num_style
        line_idx += 1


# ======================================================================================================================
#  RESULTS OPERATIONAL PLANNING - write functions
# ======================================================================================================================
def _write_operational_planning_results_to_excel(planning_problem, results, primal_evolution=list(), shared_ess_capacity=dict(), filename='operation_planning_results'):

    wb = Workbook()

    _write_operational_planning_main_info_to_excel(planning_problem, wb, results)
    _write_shared_ess_specifications(wb, planning_problem.shared_ess_data)
    if shared_ess_capacity:
        planning_problem.shared_ess_data.write_ess_results_to_excel(wb, shared_ess_capacity)

    if primal_evolution:
        _write_objective_function_evolution_to_excel(wb, primal_evolution)

    # Interface Power Flow
    _write_interface_results_to_excel(planning_problem, wb, results['interface'])

    # Shared Energy Storages results
    _write_shared_energy_storages_results_to_excel(planning_problem, wb, results)

    #  TSO and DSOs' results
    _write_network_voltage_results_to_excel(planning_problem, wb, results)
    _write_network_consumption_results_to_excel(planning_problem, wb, results)
    _write_network_generation_results_to_excel(planning_problem, wb, results)
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'losses')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'ratio')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'current_perc')
    _write_network_branch_power_flow_results_to_excel(planning_problem, wb, results)
    _write_network_energy_storages_results_to_excel(planning_problem, wb, results)
    planning_problem.shared_ess_data.write_relaxation_slacks_results_to_excel(wb, results['esso'])

    # Save results
    try:
        wb.save(filename)
    except:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = f"{filename.replace('.xlsx', '')}_{current_time}.xlsx"
        print(f"[WARNING] Results saved to file {backup_filename}.xlsx")
        wb.save(backup_filename)


def _write_operational_planning_results_no_coordination_to_excel(planning_problem, results, filename='operation_planning_results_no_coordination'):

    wb = Workbook()

    _write_operational_planning_main_info_to_excel(planning_problem, wb, results)

    #  TSO and DSOs' results
    _write_network_voltage_results_to_excel(planning_problem, wb, results)
    _write_network_consumption_results_to_excel(planning_problem, wb, results)
    _write_network_generation_results_to_excel(planning_problem, wb, results)
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'losses')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'ratio')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'current_perc')
    _write_network_branch_power_flow_results_to_excel(planning_problem, wb, results)
    _write_network_energy_storages_results_to_excel(planning_problem, wb, results)
    _write_relaxation_slacks_results_no_coordination_to_excel(planning_problem, wb, results)

    # Save results
    try:
        wb.save(filename)
    except:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = f"{filename.replace('.xlsx', '')}_{current_time}.xlsx"
        print(f"[WARNING] Results saved to file {backup_filename}.xlsx")
        wb.save(backup_filename)


def _write_operational_planning_main_info_to_excel(planning_problem, workbook, results):

    sheet = workbook.worksheets[0]
    sheet.title = 'Main Info'

    decimal_style = '0.00'
    line_idx = 1

    # Write Header
    col_idx = 4
    for year in planning_problem.years:
        for _ in planning_problem.days:
            sheet.cell(row=line_idx, column=col_idx).value = year
            col_idx += 1

    col_idx = 1
    line_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Agent'
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Node ID'
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Value'
    col_idx += 1

    for _ in planning_problem.years:
        for day in planning_problem.days:
            sheet.cell(row=line_idx, column=col_idx).value = day
            col_idx += 1

    # TSO
    line_idx = _write_operational_planning_main_info_per_operator(planning_problem.transmission_network, sheet, 'TSO', line_idx, results['tso']['results'])

    # DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id]
        line_idx = _write_operational_planning_main_info_per_operator(distribution_network, sheet, 'DSO', line_idx, dso_results, tn_node_id=tn_node_id)


def _write_operational_planning_main_info_per_operator(network, sheet, operator_type, line_idx, results, tn_node_id='-'):

    decimal_style = '0.00'

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1

    # - Objective
    obj_string = 'Objective'
    if network.params.obj_type == OBJ_MIN_COST:
        obj_string += ' (cost), [€]'
    elif network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        obj_string += ' (congestion management)'
    sheet.cell(row=line_idx, column=col_idx).value = obj_string
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['obj']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    # Total Load
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Load, [MWh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            load_aux = results[year][day]['total_load']
            if network.params.l_curt:
                load_aux -= results[year][day]['load_curt']
            sheet.cell(row=line_idx, column=col_idx).value = load_aux
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    # Flexibility used
    if network.params.fl_reg:
        line_idx += 1
        col_idx = 1
        sheet.cell(row=line_idx, column=col_idx).value = operator_type
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = 'Flexibility used, [MWh]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['flex_used']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

    # Total Load curtailed
    if network.params.l_curt:
        line_idx += 1
        col_idx = 1
        sheet.cell(row=line_idx, column=col_idx).value = operator_type
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = 'Load curtailed, [MWh]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['load_curt']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

    # Total Generation
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Generation, [MWh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_gen']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    # Total Conventional Generation
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Conventional Generation, [MWh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_conventional_gen']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    # Total Renewable Generation
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Renewable generation, [MWh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_renewable_gen']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    # Renewable Generation Curtailed
    if network.params.rg_curt:
        line_idx += 1
        col_idx = 1
        sheet.cell(row=line_idx, column=col_idx).value = operator_type
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = 'Renewable generation curtailed, [MWh]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['gen_curt']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

    # Losses
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Losses, [MWh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['losses']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    # Number of price (market) scenarios
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Number of market scenarios'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = len(network.network[year][day].prob_market_scenarios)
            col_idx += 1

    # Number of operation (generation and consumption) scenarios
    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Number of operation scenarios'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = len(network.network[year][day].prob_operation_scenarios)
            col_idx += 1

    return line_idx


def _write_shared_ess_specifications(workbook, shared_ess_info):

    sheet = workbook.create_sheet('Shared ESS Specifications')

    decimal_style = '0.000'

    # Write Header
    row_idx = 1
    sheet.cell(row=row_idx, column=1).value = 'Year'
    sheet.cell(row=row_idx, column=2).value = 'Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Sinst, [MVA]'
    sheet.cell(row=row_idx, column=4).value = 'Einst, [MVAh]'

    # Write Shared ESS specifications
    for year in shared_ess_info.years:
        for shared_ess in shared_ess_info.shared_energy_storages[year]:
            row_idx = row_idx + 1
            sheet.cell(row=row_idx, column=1).value = year
            sheet.cell(row=row_idx, column=2).value = shared_ess.bus
            sheet.cell(row=row_idx, column=3).value = shared_ess.s
            sheet.cell(row=row_idx, column=3).number_format = decimal_style
            sheet.cell(row=row_idx, column=4).value = shared_ess.e
            sheet.cell(row=row_idx, column=4).number_format = decimal_style


def _write_objective_function_evolution_to_excel(workbook, primal_evolution):

    sheet = workbook.create_sheet('Primal Evolution')

    decimal_style = '0.000000'
    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Iteration'
    sheet.cell(row=row_idx, column=2).value = 'OF value'
    row_idx = row_idx + 1
    for i in range(len(primal_evolution)):
        sheet.cell(row=row_idx, column=1).value = i
        sheet.cell(row=row_idx, column=2).value = primal_evolution[i]
        sheet.cell(row=row_idx, column=2).number_format = decimal_style
        sheet.cell(row=row_idx, column=2).value = primal_evolution[i]
        sheet.cell(row=row_idx, column=2).number_format = decimal_style
        row_idx = row_idx + 1


def _write_interface_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Interface PF')

    row_idx = 1
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Operator'
    sheet.cell(row=row_idx, column=3).value = 'Year'
    sheet.cell(row=row_idx, column=4).value = 'Day'
    sheet.cell(row=row_idx, column=5).value = 'Quantity'
    sheet.cell(row=row_idx, column=6).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=7).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 8).value = p
    row_idx = row_idx + 1

    # TSO's results
    for year in results['tso']:
        for day in results['tso'][year]:
            for node_id in results['tso'][year][day]:
                expected_vmag = [0.0 for _ in range(planning_problem.num_instants)]
                expected_p = [0.0 for _ in range(planning_problem.num_instants)]
                expected_q = [0.0 for _ in range(planning_problem.num_instants)]
                for s_m in results['tso'][year][day][node_id]:
                    omega_m = planning_problem.transmission_network.network[year][day].prob_market_scenarios[s_m]
                    for s_o in results['tso'][year][day][node_id][s_m]:
                        omega_s = planning_problem.transmission_network.network[year][day].prob_operation_scenarios[s_o]

                        # Voltage magnitude
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Vmag, [p.u.]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            interface_vmag = results['tso'][year][day][node_id][s_m][s_o]['v'][p]
                            sheet.cell(row=row_idx, column=p + 8).value = interface_vmag
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            expected_vmag[p] += interface_vmag * omega_m * omega_s
                        row_idx += 1

                        # Active Power
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            interface_p = results['tso'][year][day][node_id][s_m][s_o]['p'][p]
                            sheet.cell(row=row_idx, column=p + 8).value = interface_p
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            expected_p[p] += interface_p * omega_m * omega_s
                        row_idx += 1

                        # Reactive Power
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            interface_q = results['tso'][year][day][node_id][s_m][s_o]['q'][p]
                            sheet.cell(row=row_idx, column=p + 8).value = interface_q
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            expected_q[p] += interface_q * omega_m * omega_s
                        row_idx += 1

                # Expected Active Power
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Vmag, [p.u.]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_vmag[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                row_idx += 1

                # Expected Active Power
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_p[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                row_idx += 1

                # Expected Reactive Power
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_q[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                row_idx += 1

    # DSOs' results
    for node_id in results['dso']:
        for year in results['dso'][node_id]:
            for day in results['dso'][node_id][year]:
                expected_vmag = [0.0 for _ in range(planning_problem.num_instants)]
                expected_p = [0.0 for _ in range(planning_problem.num_instants)]
                expected_q = [0.0 for _ in range(planning_problem.num_instants)]
                for s_m in results['dso'][node_id][year][day]:
                    omega_m = planning_problem.distribution_networks[node_id].network[year][day].prob_market_scenarios[s_m]
                    for s_o in results['dso'][node_id][year][day][s_m]:
                        omega_s = planning_problem.distribution_networks[node_id].network[year][day].prob_operation_scenarios[s_o]

                        # Voltage magnitude
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Vmag, [p.u.]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            interface_vmag = results['dso'][node_id][year][day][s_m][s_o]['v'][p]
                            sheet.cell(row=row_idx, column=p + 8).value = interface_vmag
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            expected_vmag[p] += interface_vmag * omega_m * omega_s
                        row_idx += 1

                        # Active Power
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            interface_p = results['dso'][node_id][year][day][s_m][s_o]['p'][p]
                            sheet.cell(row=row_idx, column=p + 8).value = interface_p
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            expected_p[p] += interface_p * omega_m * omega_s
                        row_idx += 1

                        # Reactive Power
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(len(results['dso'][node_id][year][day][s_m][s_o]['q'])):
                            interface_q = results['dso'][node_id][year][day][s_m][s_o]['q'][p]
                            sheet.cell(row=row_idx, column=p + 8).value = interface_q
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            expected_q[p] += interface_q * omega_m * omega_s
                        row_idx += 1

                # Expected Voltage magnitude
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Vmag, [p.u.]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_vmag[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                row_idx += 1

                # Expected Active Power
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_p[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                row_idx += 1

                # Expected Reactive Power
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_q[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                row_idx += 1


def _write_shared_energy_storages_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Shared ESS')

    row_idx = 1
    decimal_style = '0.00'
    percent_style = '0.00%'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Node ID'
    sheet.cell(row=row_idx, column=2).value = 'Operator'
    sheet.cell(row=row_idx, column=3).value = 'Year'
    sheet.cell(row=row_idx, column=4).value = 'Day'
    sheet.cell(row=row_idx, column=5).value = 'Quantity'
    sheet.cell(row=row_idx, column=6).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=7).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 8).value = p

    # TSO's results
    for year in results['tso']['results']:
        for day in results['tso']['results'][year]:

            expected_p = dict()
            expected_q = dict()
            expected_s = dict()
            expected_soc = dict()
            expected_soc_percent = dict()
            for node_id in planning_problem.active_distribution_network_nodes:
                expected_p[node_id] = [0.0 for _ in range(planning_problem.num_instants)]
                expected_q[node_id] = [0.0 for _ in range(planning_problem.num_instants)]
                expected_s[node_id] = [0.0 for _ in range(planning_problem.num_instants)]
                expected_soc[node_id] = [0.0 for _ in range(planning_problem.num_instants)]
                expected_soc_percent[node_id] = [0.0 for _ in range(planning_problem.num_instants)]

            for s_m in results['tso']['results'][year][day]['scenarios']:

                omega_m = planning_problem.transmission_network.network[year][day].prob_market_scenarios[s_m]

                for s_o in results['tso']['results'][year][day]['scenarios'][s_m]:

                    omega_s = planning_problem.transmission_network.network[year][day].prob_operation_scenarios[s_o]

                    for node_id in planning_problem.active_distribution_network_nodes:

                        # Active power
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_p = results['tso']['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['p'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_p
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_p != 'N/A':
                                expected_p[node_id][p] += ess_p * omega_m * omega_s
                            else:
                                expected_p[node_id][p] = ess_p

                        # Reactive power
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_q = results['tso']['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['q'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_q
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_q != 'N/A':
                                expected_q[node_id][p] += ess_q * omega_m * omega_s
                            else:
                                expected_q[node_id][p] = ess_q

                        # Apparent power
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_s = results['tso']['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['s'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_s
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_s != 'N/A':
                                expected_s[node_id][p] += ess_s * omega_m * omega_s
                            else:
                                expected_s[node_id][p] = ess_s

                        # State-of-Charge, [MVAh]
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'SoC, [MVAh]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_soc = results['tso']['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_soc
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_soc != 'N/A':
                                expected_soc[node_id][p] += ess_soc * omega_m * omega_s
                            else:
                                expected_soc[node_id][p] = ess_soc

                        # State-of-Charge, [%]
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'TSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'SoC, [%]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_soc_percent = results['tso']['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_soc_percent
                            sheet.cell(row=row_idx, column=p + 8).number_format = percent_style
                            if ess_soc_percent != 'N/A':
                                expected_soc_percent[node_id][p] += ess_soc_percent * omega_m * omega_s
                            else:
                                expected_soc_percent[node_id][p] = ess_soc_percent

            for node_id in planning_problem.active_distribution_network_nodes:

                # Active Power, [MW]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_p[node_id][p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # Reactive Power, [MVAr]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_q[node_id][p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # Apparent Power, [MVA]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_s[node_id][p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # State-of-Charge, [MVAh]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'SoC, [MVAh]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_soc[node_id][p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # State-of-Charge, [%]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'TSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'SoC, [%]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_soc_percent[node_id][p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = percent_style

    # DSO's results
    for node_id in results['dso']:
        for year in results['dso'][node_id]['results']:
            for day in results['dso'][node_id]['results'][year]:

                distribution_network = planning_problem.distribution_networks[node_id].network[year][day]
                ref_node_id = distribution_network.get_reference_node_id()

                expected_p = [0.0 for _ in range(planning_problem.num_instants)]
                expected_q = [0.0 for _ in range(planning_problem.num_instants)]
                expected_s = [0.0 for _ in range(planning_problem.num_instants)]
                expected_soc = [0.0 for _ in range(planning_problem.num_instants)]
                expected_soc_percent = [0.0 for _ in range(planning_problem.num_instants)]

                for s_m in results['dso'][node_id]['results'][year][day]['scenarios']:

                    omega_m = distribution_network.prob_market_scenarios[s_m]

                    for s_o in results['dso'][node_id]['results'][year][day]['scenarios'][s_m]:

                        omega_s = distribution_network.prob_operation_scenarios[s_o]

                        # Active power
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_p = results['dso'][node_id]['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['p'][ref_node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_p
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_p != 'N/A':
                                expected_p[p] += ess_p * omega_m * omega_s
                            else:
                                expected_p[p] = ess_p

                        # Reactive power
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_q = results['dso'][node_id]['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['q'][ref_node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_q
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_q != 'N/A':
                                expected_q[p] += ess_q * omega_m * omega_s
                            else:
                                expected_q[p] = ess_q

                        # Apparent power
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_s = results['dso'][node_id]['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['s'][ref_node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_s
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_s != 'N/A':
                                expected_s[p] += ess_s * omega_m * omega_s
                            else:
                                expected_s[p] = ess_s

                        # State-of-Charge, [MVAh]
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'SoC, [MVAh]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_soc = results['dso'][node_id]['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][ref_node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_soc
                            sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style
                            if ess_soc != 'N/A':
                                expected_soc[p] += ess_soc * omega_m * omega_s
                            else:
                                expected_soc[p] = ess_soc

                        # State-of-Charge, [%]
                        row_idx = row_idx + 1
                        sheet.cell(row=row_idx, column=1).value = node_id
                        sheet.cell(row=row_idx, column=2).value = 'DSO'
                        sheet.cell(row=row_idx, column=3).value = int(year)
                        sheet.cell(row=row_idx, column=4).value = day
                        sheet.cell(row=row_idx, column=5).value = 'SoC, [%]'
                        sheet.cell(row=row_idx, column=6).value = s_m
                        sheet.cell(row=row_idx, column=7).value = s_o
                        for p in range(planning_problem.num_instants):
                            ess_soc_percent = results['dso'][node_id]['results'][year][day]['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][ref_node_id][p]
                            sheet.cell(row=row_idx, column=p + 8).value = ess_soc_percent
                            sheet.cell(row=row_idx, column=p + 8).number_format = percent_style
                            if ess_soc_percent != 'N/A':
                                expected_soc_percent[p] += ess_soc_percent * omega_m * omega_s
                            else:
                                expected_soc_percent[p] = ess_soc_percent

                # Expected values

                # Active Power, [MW]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_p[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # Reactive Power, [MW]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_q[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # Apparent Power, [MW]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_s[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # State-of-Charge, [MVAh]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'SoC, [MVAh]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_soc[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # State-of-Charge, [%]
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'DSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'SoC, [%]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 8).value = expected_soc_percent[p]
                    sheet.cell(row=row_idx, column=p + 8).number_format = percent_style

    # ESSO's results
    for year in results['esso']['operation']['aggregated']:
        for day in results['esso']['operation']['aggregated'][year]:
            for node_id in planning_problem.active_distribution_network_nodes:

                # Active power
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'ESSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    ess_p = results['esso']['operation']['aggregated'][year][day][node_id]['p'][p]
                    sheet.cell(row=row_idx, column=p + 8).value = ess_p
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # Reactive power
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'ESSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    ess_q = results['esso']['operation']['aggregated'][year][day][node_id]['q'][p]
                    sheet.cell(row=row_idx, column=p + 8).value = ess_q
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style

                # Apparent power
                row_idx = row_idx + 1
                sheet.cell(row=row_idx, column=1).value = node_id
                sheet.cell(row=row_idx, column=2).value = 'ESSO'
                sheet.cell(row=row_idx, column=3).value = int(year)
                sheet.cell(row=row_idx, column=4).value = day
                sheet.cell(row=row_idx, column=5).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=6).value = 'Expected'
                sheet.cell(row=row_idx, column=7).value = '-'
                for p in range(planning_problem.num_instants):
                    ess_s = results['esso']['operation']['aggregated'][year][day][node_id]['s'][p]
                    sheet.cell(row=row_idx, column=p + 8).value = ess_s
                    sheet.cell(row=row_idx, column=p + 8).number_format = decimal_style


def _write_network_voltage_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Voltage')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Network Node ID'
    sheet.cell(row=row_idx, column=4).value = 'Year'
    sheet.cell(row=row_idx, column=5).value = 'Day'
    sheet.cell(row=row_idx, column=6).value = 'Quantity'
    sheet.cell(row=row_idx, column=7).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=8).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 9).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    transmission_network = planning_problem.transmission_network.network
    row_idx = _write_network_voltage_results_per_operator(transmission_network, sheet, 'TSO', row_idx, results['tso']['results'])

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        row_idx = _write_network_voltage_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, tn_node_id=tn_node_id)


def _write_network_voltage_results_per_operator(network, sheet, operator_type, row_idx, results, tn_node_id='-'):

    decimal_style = '0.00'

    violation_fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')

    for year in results:
        for day in results[year]:

            ref_node_id = network[year][day].get_reference_node_id()
            expected_vmag = dict()
            expected_vang = dict()
            for node in network[year][day].nodes:
                expected_vmag[node.bus_i] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_vang[node.bus_i] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for node_id in results[year][day]['scenarios'][s_m][s_o]['voltage']['vmag']:

                        v_min, v_max = network[year][day].get_node_voltage_limits(node_id)

                        # Voltage magnitude
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Vmag, [p.u.]'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            v_mag = results[year][day]['scenarios'][s_m][s_o]['voltage']['vmag'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = v_mag
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            if node_id != ref_node_id and (v_mag > v_max + SMALL_TOLERANCE or v_mag < v_min - SMALL_TOLERANCE):
                                sheet.cell(row=row_idx, column=p + 9).fill = violation_fill
                            expected_vmag[node_id][p] += v_mag * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Voltage angle
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Vang, [º]'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            v_ang = results[year][day]['scenarios'][s_m][s_o]['voltage']['vang'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = v_ang
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            expected_vang[node_id][p] += v_ang * omega_m * omega_s
                        row_idx = row_idx + 1

            for node in network[year][day].nodes:

                node_id = node.bus_i
                v_min, v_max = network[year][day].get_node_voltage_limits(node_id)

                # Expected voltage magnitude
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = int(year)
                sheet.cell(row=row_idx, column=5).value = day
                sheet.cell(row=row_idx, column=6).value = 'Vmag, [p.u.]'
                sheet.cell(row=row_idx, column=7).value = 'Expected'
                sheet.cell(row=row_idx, column=8).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 9).value = expected_vmag[node_id][p]
                    sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                    if node_id != ref_node_id and (expected_vmag[node_id][p] > v_max + SMALL_TOLERANCE or expected_vmag[node_id][p] < v_min - SMALL_TOLERANCE):
                        sheet.cell(row=row_idx, column=p + 9).fill = violation_fill
                row_idx = row_idx + 1

                # Expected voltage angle
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = int(year)
                sheet.cell(row=row_idx, column=5).value = day
                sheet.cell(row=row_idx, column=6).value = 'Vang, [º]'
                sheet.cell(row=row_idx, column=7).value = 'Expected'
                sheet.cell(row=row_idx, column=8).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 9).value = expected_vang[node_id][p]
                    sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                row_idx = row_idx + 1

    return row_idx


def _write_network_consumption_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Consumption')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Network Node ID'
    sheet.cell(row=row_idx, column=4).value = 'Load ID'
    sheet.cell(row=row_idx, column=5).value = 'Year'
    sheet.cell(row=row_idx, column=6).value = 'Day'
    sheet.cell(row=row_idx, column=7).value = 'Quantity'
    sheet.cell(row=row_idx, column=8).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=9).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 10).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    tso_results = results['tso']['results']
    transmission_network = planning_problem.transmission_network.network
    tn_params = planning_problem.transmission_network.params
    row_idx = _write_network_consumption_results_per_operator(transmission_network, tn_params, sheet, 'TSO', row_idx, tso_results)

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        dn_params = planning_problem.distribution_networks[tn_node_id].params
        row_idx = _write_network_consumption_results_per_operator(distribution_network, dn_params, sheet, 'DSO', row_idx, dso_results, tn_node_id=tn_node_id)


def _write_network_consumption_results_per_operator(network, params, sheet, operator_type, row_idx, results, tn_node_id='-'):

    decimal_style = '0.00'
    violation_fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')

    for year in results:
        for day in results[year]:

            expected_pc = dict()
            expected_flex_up = dict()
            expected_flex_down = dict()
            expected_pc_curt = dict()
            expected_pnet = dict()
            expected_qc = dict()
            for load in network[year][day].loads:
                expected_pc[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_flex_up[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_flex_down[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pc_curt[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pnet[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qc[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for load in network[year][day].loads:

                        # - Active Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = load.bus
                        sheet.cell(row=row_idx, column=4).value = load.load_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Pc, [MW]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            pc = results[year][day]['scenarios'][s_m][s_o]['consumption']['pc'][load.load_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = pc
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_pc[load.load_id][p] += pc * omega_m * omega_s
                        row_idx = row_idx + 1

                        if params.fl_reg:

                            # - Flexibility, up
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.bus
                            sheet.cell(row=row_idx, column=4).value = load.load_id
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Flex Up, [MW]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                flex = results[year][day]['scenarios'][s_m][s_o]['consumption']['p_up'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = flex
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_flex_up[load.load_id][p] += flex * omega_m * omega_s
                            row_idx = row_idx + 1

                            # - Flexibility, down
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.bus
                            sheet.cell(row=row_idx, column=4).value = load.load_id
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Flex Down, [MW]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                flex = results[year][day]['scenarios'][s_m][s_o]['consumption']['p_down'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = flex
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_flex_down[load.load_id][p] += flex * omega_m * omega_s
                            row_idx = row_idx + 1

                        if params.l_curt:

                            # - Active power curtailment
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.bus
                            sheet.cell(row=row_idx, column=4).value = load.load_id
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Pc_curt, [MW]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                pc_curt = results[year][day]['scenarios'][s_m][s_o]['consumption']['pc_curt'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = pc_curt
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                if pc_curt >= SMALL_TOLERANCE:
                                    sheet.cell(row=row_idx, column=p + 10).fill = violation_fill
                                expected_pc_curt[load.load_id][p] += pc_curt * omega_m * omega_s
                            row_idx = row_idx + 1

                        if params.fl_reg or params.l_curt:

                            # - Active power net consumption
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.bus
                            sheet.cell(row=row_idx, column=4).value = load.load_id
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Pc_net, [MW]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                p_net = results[year][day]['scenarios'][s_m][s_o]['consumption']['pc_net'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = p_net
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_pnet[load.load_id][p] += p_net * omega_m * omega_s
                            row_idx = row_idx + 1

                        # - Reactive power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = load.bus
                        sheet.cell(row=row_idx, column=4).value = load.load_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Qc, [MVAr]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            qc = results[year][day]['scenarios'][s_m][s_o]['consumption']['qc'][load.load_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = qc
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_qc[load.load_id][p] += qc * omega_m * omega_s
                        row_idx = row_idx + 1

            for load in network[year][day].loads:

                # - Active Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = load.bus
                sheet.cell(row=row_idx, column=4).value = load.load_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Pc, [MW]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_pc[load.load_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                if params.fl_reg:

                    # - Flexibility, up
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.bus
                    sheet.cell(row=row_idx, column=4).value = load.load_id
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Flex Up, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_flex_up[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

                    # - Flexibility, down
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.bus
                    sheet.cell(row=row_idx, column=4).value = load.load_id
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Flex Down, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_flex_down[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

                if params.l_curt:

                    # - Load curtailment (active power)
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.bus
                    sheet.cell(row=row_idx, column=4).value = load.load_id
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Pc_curt, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_pc_curt[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                        if expected_pc_curt[load.load_id][p] >= SMALL_TOLERANCE:
                            sheet.cell(row=row_idx, column=p + 9).fill = violation_fill
                    row_idx = row_idx + 1

                if params.fl_reg or params.l_curt:

                    # - Active power net consumption
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.bus
                    sheet.cell(row=row_idx, column=4).value = load.load_id
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Pc_net, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_pnet[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

                # - Reactive power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = load.bus
                sheet.cell(row=row_idx, column=4).value = load.load_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Qc, [MVAr]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_qc[load.load_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

    return row_idx


def _write_network_generation_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Generation')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Network Node ID'
    sheet.cell(row=row_idx, column=4).value = 'Generator ID'
    sheet.cell(row=row_idx, column=5).value = 'Type'
    sheet.cell(row=row_idx, column=6).value = 'Year'
    sheet.cell(row=row_idx, column=7).value = 'Day'
    sheet.cell(row=row_idx, column=8).value = 'Quantity'
    sheet.cell(row=row_idx, column=9).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=10).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 11).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    transmission_network = planning_problem.transmission_network.network
    tn_params = planning_problem.transmission_network.params
    row_idx = _write_network_generation_results_per_operator(transmission_network, tn_params, sheet, 'TSO', row_idx, results['tso']['results'])

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        dn_params = planning_problem.distribution_networks[tn_node_id].params
        row_idx = _write_network_generation_results_per_operator(distribution_network, dn_params, sheet, 'DSO', row_idx, dso_results, tn_node_id=tn_node_id)


def _write_network_generation_results_per_operator(network, params, sheet, operator_type, row_idx, results, tn_node_id='-'):

    decimal_style = '0.00'
    violation_fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')

    for year in results:
        for day in results[year]:

            expected_pg = dict()
            expected_pg_curt = dict()
            expected_pg_net = dict()
            expected_qg = dict()
            for generator in network[year][day].generators:
                expected_pg[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pg_curt[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pg_net[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qg[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for generator in network[year][day].generators:

                        node_id = generator.bus
                        gen_id = generator.gen_id
                        gen_type = network[year][day].get_gen_type(gen_id)

                        # Active Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = gen_id
                        sheet.cell(row=row_idx, column=5).value = gen_type
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Pg, [MW]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            pg = results[year][day]['scenarios'][s_m][s_o]['generation']['pg'][gen_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = pg
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_pg[gen_id][p] += pg * omega_m * omega_s
                        row_idx = row_idx + 1

                        if params.rg_curt:

                            # Active Power curtailment
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = gen_id
                            sheet.cell(row=row_idx, column=5).value = gen_type
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = 'Pg_curt, [MW]'
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                pg_curt = results[year][day]['scenarios'][s_m][s_o]['generation']['pg_curt'][gen_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = pg_curt
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                if pg_curt > SMALL_TOLERANCE:
                                    sheet.cell(row=row_idx, column=p + 11).fill = violation_fill
                                expected_pg_curt[gen_id][p] += pg_curt * omega_m * omega_s
                            row_idx = row_idx + 1

                            # Active Power net
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = gen_id
                            sheet.cell(row=row_idx, column=5).value = gen_type
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = 'Pg_net, [MW]'
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                pg_net = results[year][day]['scenarios'][s_m][s_o]['generation']['pg_net'][gen_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = pg_net
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                expected_pg_net[gen_id][p] += pg_net * omega_m * omega_s
                            row_idx = row_idx + 1

                        # Reactive Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = gen_id
                        sheet.cell(row=row_idx, column=5).value = gen_type
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Qg, [MVAr]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            qg = results[year][day]['scenarios'][s_m][s_o]['generation']['qg'][gen_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = qg
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_qg[gen_id][p] += qg * omega_m * omega_s
                        row_idx = row_idx + 1

            for generator in network[year][day].generators:

                node_id = generator.bus
                gen_id = generator.gen_id
                gen_type = network[year][day].get_gen_type(gen_id)

                # Active Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = gen_id
                sheet.cell(row=row_idx, column=5).value = gen_type
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Pg, [MW]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_pg[gen_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                if params.rg_curt:

                    # Active Power curtailment
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = node_id
                    sheet.cell(row=row_idx, column=4).value = gen_id
                    sheet.cell(row=row_idx, column=5).value = gen_type
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = 'Pg_curt, [MW]'
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_pg_curt[gen_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                        if expected_pg_curt[gen_id][p] > SMALL_TOLERANCE:
                            sheet.cell(row=row_idx, column=p + 11).fill = violation_fill
                    row_idx = row_idx + 1

                    # Active Power net
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = node_id
                    sheet.cell(row=row_idx, column=4).value = gen_id
                    sheet.cell(row=row_idx, column=5).value = gen_type
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = 'Pg_net, [MW]'
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_pg_net[gen_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                    row_idx = row_idx + 1

                # Reactive Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = gen_id
                sheet.cell(row=row_idx, column=5).value = gen_type
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Qg, [MVAr]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_qg[gen_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

    return row_idx


def _write_network_branch_results_to_excel(planning_problem, workbook, results, result_type):

    sheet_name = str()
    if result_type == 'losses':
        sheet_name = 'Branch Losses'
    elif result_type == 'ratio':
        sheet_name = 'Transformer Ratio'
    elif result_type == 'current_perc':
        sheet_name = 'Branch Loading'
    sheet = workbook.create_sheet(sheet_name)

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'From Node ID'
    sheet.cell(row=row_idx, column=4).value = 'To Node ID'
    sheet.cell(row=row_idx, column=5).value = 'Year'
    sheet.cell(row=row_idx, column=6).value = 'Day'
    sheet.cell(row=row_idx, column=7).value = 'Quantity'
    sheet.cell(row=row_idx, column=8).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=9).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 10).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    transmission_network = planning_problem.transmission_network.network
    row_idx = _write_network_branch_results_per_operator(transmission_network, sheet, 'TSO', row_idx, results['tso']['results'], result_type)

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        row_idx = _write_network_branch_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, result_type, tn_node_id=tn_node_id)


def _write_network_branch_results_per_operator(network, sheet, operator_type, row_idx, results, result_type, tn_node_id='-'):

    decimal_style = '0.00'
    perc_style = '0.00%'
    violation_fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')

    aux_string = str()
    if result_type == 'losses':
        aux_string = 'P, [MW]'
    elif result_type == 'ratio':
        aux_string = 'Ratio'
    elif result_type == 'current_perc':
        aux_string = 'I, [%]'

    for year in results:
        for day in results[year]:

            expected_values = dict()
            for branch in network[year][day].branches:
                expected_values[branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for branch in network[year][day].branches:

                        branch_id = branch.branch_id

                        if not(result_type == 'ratio' and not branch.is_transformer):

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = branch.fbus
                            sheet.cell(row=row_idx, column=4).value = branch.tbus
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = aux_string
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                value = results[year][day]['scenarios'][s_m][s_o]['branches'][result_type][branch_id][p]
                                if result_type == 'current_perc':
                                    sheet.cell(row=row_idx, column=p + 10).value = value
                                    sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                                    if value > 1.0 + SMALL_TOLERANCE:
                                        sheet.cell(row=row_idx, column=p + 10).fill = violation_fill
                                else:
                                    sheet.cell(row=row_idx, column=p + 10).value = value
                                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_values[branch_id][p] += value * omega_m * omega_s
                            row_idx = row_idx + 1

            for branch in network[year][day].branches:
                branch_id = branch.branch_id
                if not (result_type == 'ratio' and not branch.is_transformer):

                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = branch.fbus
                    sheet.cell(row=row_idx, column=4).value = branch.tbus
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = aux_string
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        if result_type == 'current_perc':
                            sheet.cell(row=row_idx, column=p + 10).value = expected_values[branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                            if expected_values[branch_id][p] > 1.0:
                                sheet.cell(row=row_idx, column=p + 10).fill = violation_fill
                        else:
                            sheet.cell(row=row_idx, column=p + 10).value = expected_values[branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

    return row_idx


def _write_network_branch_power_flow_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Power Flows')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'From Node ID'
    sheet.cell(row=row_idx, column=4).value = 'To Node ID'
    sheet.cell(row=row_idx, column=5).value = 'Year'
    sheet.cell(row=row_idx, column=6).value = 'Day'
    sheet.cell(row=row_idx, column=7).value = 'Quantity'
    sheet.cell(row=row_idx, column=8).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=9).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 10).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    transmission_network = planning_problem.transmission_network.network
    row_idx = _write_network_power_flow_results_per_operator(transmission_network, sheet, 'TSO', row_idx, results['tso']['results'])

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        row_idx = _write_network_power_flow_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, tn_node_id=tn_node_id)


def _write_network_power_flow_results_per_operator(network, sheet, operator_type, row_idx, results, tn_node_id='-'):

    decimal_style = '0.00'
    perc_style = '0.00%'

    for year in results:
        for day in results[year]:

            expected_values = {'pij': {}, 'pji': {}, 'qij': {}, 'qji': {}, 'sij': {}, 'sji': {}}
            for branch in network[year][day].branches:
                expected_values['pij'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_values['pji'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_values['qij'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_values['qji'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_values['sij'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_values['sji'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for branch in network[year][day].branches:

                        branch_id = branch.branch_id
                        rating = branch.rate
                        if rating == 0.0:
                            rating = BRANCH_UNKNOWN_RATING

                        # Pij, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.fbus
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_values['pij'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Pij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.fbus
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'P, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                        row_idx = row_idx + 1

                        # Pji, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.tbus
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_values['pji'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Pji, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.tbus
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'P, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                        row_idx = row_idx + 1

                        # Qij, [MVAr]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.fbus
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_values['qij'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Qij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.fbus
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Q, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                        row_idx = row_idx + 1

                        # Qji, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.tbus
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_values['qji'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Qji, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.tbus
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Q, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                        row_idx = row_idx + 1

                        # Sij, [MVA]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.fbus
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_values['sij'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Sij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.fbus
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'S, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                        row_idx = row_idx + 1

                        # Sji, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.tbus
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_values['sji'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Sji, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.tbus
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'S, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 10).value = value
                            sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                        row_idx = row_idx + 1

            for branch in network[year][day].branches:

                branch_id = branch.branch_id
                rating = branch.rate
                if rating == 0.0:
                    rating = BRANCH_UNKNOWN_RATING

                # Pij, [MW]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.fbus
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_values['pij'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Pij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.fbus
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'P, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = abs(expected_values['pij'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                row_idx = row_idx + 1

                # Pji, [MW]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.tbus
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_values['pji'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Pji, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.tbus
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'P, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = abs(expected_values['pji'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                row_idx = row_idx + 1

                # Qij, [MVAr]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.fbus
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_values['qij'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Qij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.fbus
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Q, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = abs(expected_values['qij'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                row_idx = row_idx + 1

                # Qji, [MVAr]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.tbus
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_values['qji'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Qji, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.tbus
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Q, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = abs(expected_values['qji'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Sij, [MVA]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.fbus
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_values['sij'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Sij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.fbus
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'S, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = abs(expected_values['sij'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                row_idx = row_idx + 1

                # Sji, [MVA]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.tbus
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_values['sji'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # Sji, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.tbus
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'S, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = abs(expected_values['sji'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 10).number_format = perc_style
                row_idx = row_idx + 1

    return row_idx


def _write_network_energy_storages_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Energy Storage')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Network Node ID'
    sheet.cell(row=row_idx, column=4).value = 'ESS ID'
    sheet.cell(row=row_idx, column=5).value = 'Year'
    sheet.cell(row=row_idx, column=6).value = 'Day'
    sheet.cell(row=row_idx, column=7).value = 'Quantity'
    sheet.cell(row=row_idx, column=8).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=9).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 10).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    tso_results = results['tso']['results']
    transmission_network = planning_problem.transmission_network.network
    if planning_problem.transmission_network.params.es_reg:
        row_idx = _write_network_energy_storages_results_per_operator(transmission_network, sheet, 'TSO', row_idx, tso_results)

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        if planning_problem.distribution_networks[tn_node_id].params.es_reg:
            row_idx = _write_network_energy_storages_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, tn_node_id=tn_node_id)


def _write_network_energy_storages_results_per_operator(network, sheet, operator_type, row_idx, results, tn_node_id='-'):

    decimal_style = '0.00'
    percent_style = '0.00%'

    for year in results:
        for day in results[year]:

            expected_p = dict()
            expected_q = dict()
            expected_s = dict()
            expected_soc = dict()
            expected_soc_percent = dict()
            for energy_storage in network[year][day].energy_storages:
                es_id = energy_storage.es_id
                expected_p[es_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_q[es_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_s[es_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_soc[es_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_soc_percent[es_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for energy_storage in network[year][day].energy_storages:

                        es_id = energy_storage.es_id
                        node_id = energy_storage.bus

                        # - Active Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = es_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            ess_p = results[year][day]['scenarios'][s_m][s_o]['energy_storages']['p'][es_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = ess_p
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_p[es_id][p] += ess_p * omega_m * omega_s
                        row_idx = row_idx + 1

                        # - Reactive Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = es_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            ess_q = results[year][day]['scenarios'][s_m][s_o]['energy_storages']['q'][es_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = ess_q
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_q[es_id][p] += ess_q * omega_m * omega_s
                        row_idx = row_idx + 1

                        # - Apparent Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = es_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            ess_s = results[year][day]['scenarios'][s_m][s_o]['energy_storages']['s'][es_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = ess_s
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            expected_s[es_id][p] += ess_s * omega_m * omega_s
                        row_idx = row_idx + 1

                        # State-of-Charge, [MWh]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = es_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'SoC, [MWh]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            ess_soc = results[year][day]['scenarios'][s_m][s_o]['energy_storages']['soc'][es_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = ess_soc
                            sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                            if ess_soc != 'N/A':
                                expected_soc[es_id][p] += ess_soc * omega_m * omega_s
                            else:
                                expected_soc[es_id][p] = ess_soc
                        row_idx = row_idx + 1

                        # State-of-Charge, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = es_id
                        sheet.cell(row=row_idx, column=5).value = int(year)
                        sheet.cell(row=row_idx, column=6).value = day
                        sheet.cell(row=row_idx, column=7).value = 'SoC, [%]'
                        sheet.cell(row=row_idx, column=8).value = s_m
                        sheet.cell(row=row_idx, column=9).value = s_o
                        for p in range(network[year][day].num_instants):
                            ess_soc_percent = results[year][day]['scenarios'][s_m][s_o]['energy_storages']['soc_percent'][es_id][p]
                            sheet.cell(row=row_idx, column=p + 10).value = ess_soc_percent
                            sheet.cell(row=row_idx, column=p + 10).number_format = percent_style
                            if ess_soc_percent != 'N/A':
                                expected_soc_percent[es_id][p] += ess_soc_percent * omega_m * omega_s
                            else:
                                expected_soc_percent[es_id][p] = ess_soc_percent
                        row_idx = row_idx + 1

            for energy_storage in network[year][day].energy_storages:

                es_id = energy_storage.es_id
                node_id = energy_storage.bus

                # - Active Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = es_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_p[es_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # - Reactive Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = es_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_q[es_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # - Apparent Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = es_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_s[es_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # State-of-Charge, [MWh]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = es_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'SoC, [MWh]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_soc[es_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                # State-of-Charge, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = node_id
                sheet.cell(row=row_idx, column=4).value = es_id
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'SoC, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_soc_percent[es_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = percent_style
                row_idx = row_idx + 1

    return row_idx


def _write_relaxation_slacks_results_to_excel(planning_problem, workbook, results):
    _write_relaxation_slacks_results_network_operators_to_excel(planning_problem, workbook, results)


def _write_relaxation_slacks_results_no_coordination_to_excel(planning_problem, workbook, results):
    _write_relaxation_slacks_results_network_operators_to_excel(planning_problem, workbook, results)


def _write_relaxation_slacks_results_network_operators_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Relaxation Slacks TSO, DSOs')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'Connection Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Network Node ID'
    sheet.cell(row=row_idx, column=4).value = 'Year'
    sheet.cell(row=row_idx, column=5).value = 'Day'
    sheet.cell(row=row_idx, column=6).value = 'Quantity'
    sheet.cell(row=row_idx, column=7).value = 'Market Scenario'
    sheet.cell(row=row_idx, column=8).value = 'Operation Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 9).value = p
    row_idx = row_idx + 1

    # Write results -- TSO
    tso_results = results['tso']['results']
    transmission_network = planning_problem.transmission_network.network
    tn_params = planning_problem.transmission_network.params
    if tn_params.slacks:
        row_idx = _write_relaxation_slacks_results_per_operator(transmission_network, sheet, 'TSO', row_idx, tso_results, tn_params)

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        dn_params = planning_problem.distribution_networks[tn_node_id].params
        if dn_params.slacks:
            row_idx = _write_relaxation_slacks_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, dn_params, tn_node_id=tn_node_id)


def _write_relaxation_slacks_results_per_operator(network, sheet, operator_type, row_idx, results, params, tn_node_id='-'):

    decimal_style = '0.00'

    for year in results:
        for day in results[year]:
            for s_m in results[year][day]['scenarios']:
                for s_o in results[year][day]['scenarios'][s_m]:

                    # Voltage slacks
                    for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e_up']:

                        # - e_up
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Voltage, e_up'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            e_up = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e_up'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = e_up
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - e_down
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Voltage, e_down'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            e_down = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e_down'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = e_down
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - f_up
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Voltage, f_up'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            f_up = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['f_up'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = f_up
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # - f_down
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Voltage, f_down'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            f_down = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['f_down'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = f_down
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                    # Branch current slacks
                    for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['current']['iij_sqr']:

                        # - Charging
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Current, iij_sqr'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            iij_sqr = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['current']['iij_sqr'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = iij_sqr
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # Shared ESS slacks
                        # - Complementarity
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp']:
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Shared ESS, Complementarity'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_shared_es_comp = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_shared_es_comp
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                        # - SoC
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_up']:
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Shared ESS, soc_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_shared_es_soc = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_shared_es_soc
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Shared ESS, soc_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_shared_es_soc = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_shared_es_soc
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                        # - Day balance
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_up']:

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Shared ESS, day_balance_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_shared_es_day_balance = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_shared_es_day_balance
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Shared ESS, day_balance_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_shared_es_day_balance = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_shared_es_day_balance
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # ESS slacks
                    if params.es_reg:

                        # - Complementarity
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp']:
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, Complementarity'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_comp = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_comp
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                        # - Apparent power
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_up']:

                            # Charging, up
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, sch_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_ch_up = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_ch_up
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            # Charging, down
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, sch_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_ch_down = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_ch_down
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            # - Discharging, up
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, sdch_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_dch_up = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_dch_up
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            # - Discharging, down
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, sdch_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_dch_down = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_dch_down
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                        # - SoC
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_up']:
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, soc_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_soc = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_soc
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, soc_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_soc = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_soc
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                        # - Day balance
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_up']:

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, day_balance_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_day_balance = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_day_balance
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'ESS, day_balance_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_es_day_balance = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_es_day_balance
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # - Flexibility day balance slacks
                    if params.fl_reg:
                        for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance_up']:

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Flexibility, balance_up'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_flex_day_balance = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance_up'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_flex_day_balance
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Flexibility, balance_down'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_flex_day_balance = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance_down'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_flex_day_balance
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # - Node balance
                    for node_id in results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_up']:

                        # p_up
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Node Balance, p_up'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            slack_p_up = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_up'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = slack_p_up
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # p_down
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Node Balance, p_down'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            slack_p_down = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_down'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = slack_p_down
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # q_up
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Node Balance, q_up'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            slack_q_up = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_up'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = slack_q_up
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

                        # q_down
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = node_id
                        sheet.cell(row=row_idx, column=4).value = int(year)
                        sheet.cell(row=row_idx, column=5).value = day
                        sheet.cell(row=row_idx, column=6).value = 'Node Balance, q_down'
                        sheet.cell(row=row_idx, column=7).value = s_m
                        sheet.cell(row=row_idx, column=8).value = s_o
                        for p in range(network[year][day].num_instants):
                            slack_q_down = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_down'][node_id][p]
                            sheet.cell(row=row_idx, column=p + 9).value = slack_q_down
                            sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                        row_idx = row_idx + 1

    return row_idx


# ======================================================================================================================
#   NETWORK diagram functions (plot)
# ======================================================================================================================
def _plot_networkx_diagram(planning_problem):

    for year in planning_problem.years:
        for day in planning_problem.days:

            transmission_network = planning_problem.transmission_network.network[year][day]

            node_labels = {}
            ref_nodes, pv_nodes, pq_nodes = [], [], []
            res_pv_nodes = [gen.bus for gen in transmission_network.generators if gen.gen_type == GEN_RES_SOLAR]
            res_wind_nodes = [gen.bus for gen in transmission_network.generators if gen.gen_type == GEN_RES_WIND]
            adn_nodes = planning_problem.active_distribution_network_nodes

            branches = []
            line_list, open_line_list = [], []
            transf_list, open_transf_list = [], []
            for branch in transmission_network.branches:
                if branch.is_transformer:
                    branches.append({'type': 'transformer', 'data': branch})
                else:
                    branches.append({'type': 'line', 'data': branch})

            # Build graph
            graph = nx.Graph()
            for i in range(len(transmission_network.nodes)):
                node = transmission_network.nodes[i]
                graph.add_node(node.bus_i)
                node_labels[node.bus_i] = '{}'.format(node.bus_i)
                if node.type == BUS_REF:
                    ref_nodes.append(node.bus_i)
                elif node.type == BUS_PV:
                    pv_nodes.append(node.bus_i)
                elif node.type == BUS_PQ:
                    if node.bus_i not in (res_pv_nodes + res_wind_nodes + adn_nodes):
                        pq_nodes.append(node.bus_i)
            for i in range(len(branches)):
                branch = branches[i]
                if branch['type'] == 'line':
                    graph.add_edge(branch['data'].fbus, branch['data'].tbus)
                    if branch['data'].status == 1:
                        line_list.append((branch['data'].fbus, branch['data'].tbus))
                    else:
                        open_line_list.append((branch['data'].fbus, branch['data'].tbus))
                if branch['type'] == 'transformer':
                    graph.add_edge(branch['data'].fbus, branch['data'].tbus)
                    if branch['data'].status == 1:
                        transf_list.append((branch['data'].fbus, branch['data'].tbus))
                    else:
                        open_transf_list.append((branch['data'].fbus, branch['data'].tbus))

            # Plot diagram
            pos = nx.spring_layout(graph, k=0.50, iterations=1000)
            fig, ax = plt.subplots(figsize=(12, 8))
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=ref_nodes, node_color='red', node_size=250, label='Reference bus')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=pv_nodes, node_color='lightgreen', node_size=250, label='Conventional generator')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=pq_nodes, node_color='lightblue', node_size=250, label='PQ buses')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=res_pv_nodes, node_color='yellow', node_size=250, label='RES, PV')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=res_wind_nodes, node_color='blue', node_size=250, label='RES, Wind')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=adn_nodes, node_color='orange', node_size=250, label='ADN buses')
            nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=node_labels, font_size=12)
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=line_list, width=1.50, edge_color='black')
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=transf_list, width=2.00, edge_color='blue', label='Transformer')
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_line_list, style='dashed', width=1.50, edge_color='red')
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_transf_list, style='dashed', width=2.00, edge_color='red')
            plt.legend(scatterpoints=1, frameon=False, prop={'size': 12})
            plt.axis('off')

            filename = os.path.join(planning_problem.diagrams_dir, f'{planning_problem.name}_{year}_{day}')
            plt.savefig(f'{filename}.pdf', bbox_inches='tight')
            plt.savefig(f'{filename}.png', bbox_inches='tight')


# ======================================================================================================================
#   Aux functions
# ======================================================================================================================
def _get_initial_candidate_solution(planning_problem):
    candidate_solution = {'investment': {}, 'total_capacity': {}}
    for e in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[e]
        candidate_solution['investment'][node_id] = dict()
        candidate_solution['total_capacity'][node_id] = dict()
        for year in planning_problem.years:
            candidate_solution['investment'][node_id][year] = dict()
            candidate_solution['investment'][node_id][year]['s'] = 0.00
            candidate_solution['investment'][node_id][year]['e'] = 0.00
            candidate_solution['total_capacity'][node_id][year] = dict()
            candidate_solution['total_capacity'][node_id][year]['s'] = 0.00
            candidate_solution['total_capacity'][node_id][year]['e'] = 0.00
            '''
            if year == 2020 or year == 2040:
                candidate_solution['investment'][node_id][year]['s'] = 2.50
                candidate_solution['investment'][node_id][year]['e'] = 2.50
            candidate_solution['total_capacity'][node_id][year]['s'] = 2.50
            candidate_solution['total_capacity'][node_id][year]['e'] = 2.50
            '''''
    return candidate_solution


def _add_adn_node_to_transmission_network(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            for node_id in planning_problem.distribution_networks:
                if planning_problem.transmission_network.network[year][day].adn_load_exists(node_id):
                    adn_load_idx = planning_problem.transmission_network.network[year][day].get_adn_load_idx(node_id)
                    adn_load = planning_problem.transmission_network.network[year][day].loads[adn_load_idx]
                    adn_load.load_id = f'ADN_{node_id}'
                    adn_load.pd = dict()
                    adn_load.qd = dict()
                    for s_o in range(len(planning_problem.transmission_network.network[year][day].prob_operation_scenarios)):
                        adn_load.pd[s_o] = [0.00 for _ in range(planning_problem.num_instants)]
                        adn_load.qd[s_o] = [0.00 for _ in range(planning_problem.num_instants)]
                    adn_load.fl_reg = False
                    adn_load.status = 1
                else:
                    adn_load = Load()
                    adn_load.bus = node_id
                    adn_load.load_id = f'ADN_{node_id}'
                    adn_load.pd = dict()
                    adn_load.qd = dict()
                    for s_o in range(len(planning_problem.transmission_network.network[year][day].prob_operation_scenarios)):
                        adn_load.pd[s_o] = [0.00 for _ in range(planning_problem.num_instants)]
                        adn_load.qd[s_o] = [0.00 for _ in range(planning_problem.num_instants)]
                    adn_load.fl_reg = False
                    adn_load.status = 1
                    planning_problem.transmission_network.network[year][day].loads.append(adn_load)


def _add_shared_energy_storage_to_transmission_network(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            s_base = planning_problem.transmission_network.network[year][day].baseMVA
            for node_id in planning_problem.distribution_networks:
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = node_id
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].name
                shared_energy_storage.s = shared_energy_storage.s / s_base
                shared_energy_storage.e = shared_energy_storage.e / s_base
                planning_problem.transmission_network.network[year][day].shared_energy_storages.append(shared_energy_storage)


def _add_shared_energy_storage_to_distribution_network(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            for node_id in planning_problem.distribution_networks:
                s_base = planning_problem.distribution_networks[node_id].network[year][day].baseMVA
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = planning_problem.distribution_networks[node_id].network[year][day].get_reference_node_id()
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].network[year][day].name
                shared_energy_storage.s = shared_energy_storage.s / s_base
                shared_energy_storage.e = shared_energy_storage.e / s_base
                planning_problem.distribution_networks[node_id].network[year][day].shared_energy_storages.append(shared_energy_storage)


def _print_candidate_solution(candidate_solution):

    print('[INFO] Candidate solution:')

    # Header
    print('\t\t{:3}\t{:10}\t'.format('', 'Capacity'), end='')
    for node_id in candidate_solution['total_capacity']:
        for year in candidate_solution['total_capacity'][node_id]:
            print(f'{year}\t', end='')
        print()
        break

    # Values
    for node_id in candidate_solution['total_capacity']:
        print('\t\t{:3}\t{:10}\t'.format(node_id, 'S, [MVA]'), end='')
        for year in candidate_solution['total_capacity'][node_id]:
            print("{:.3f}\t".format(candidate_solution['total_capacity'][node_id][year]['s']), end='')
        print()
        print('\t\t{:3}\t{:10}\t'.format(node_id, 'E, [MVAh]'), end='')
        for year in candidate_solution['total_capacity'][node_id]:
            print("{:.3f}\t".format(candidate_solution['total_capacity'][node_id][year]['e']), end='')
        print()
