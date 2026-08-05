import gc
import time
from copy import copy, deepcopy
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from math import isclose, isfinite, sqrt
from sklearn.preprocessing import StandardScaler
from copulas.multivariate import GaussianMultivariate
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyomo.opt as po
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from centralized_coordination import combine_networks
from network_data import NetworkData
from load import Load
from shared_energy_storage import SharedEnergyStorage
from planning_parameters import PlanningParameters
from shared_energy_storage_data import SharedEnergyStorageData
from model_construction_helpers import *
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
        self.num_market_scenarios = int()
        self.plot_market_data = bool()
        self.results_dir = os.path.join(data_dir, 'Results')
        self.diagrams_dir = os.path.join(data_dir, 'Diagrams')
        self.logs_dir = os.path.join(self.results_dir, 'Logs')
        self.params_file = str()
        self.years = dict()
        self.days = dict()
        self.num_instants = int()
        self.discount_factor = float()
        self.cost_energy_p = dict()
        self.cost_flex = dict()
        self.prob_market_scenarios = dict()
        self.parallel_execution = bool()
        self.distribution_networks = dict()
        self.transmission_network = NetworkData()
        self.shared_ess_data = SharedEnergyStorageData()
        self.active_distribution_network_nodes = list()
        self.params = PlanningParameters()

    def run_planning_problem(self, debug_flag=False):
        print('[INFO] Running PLANNING PROBLEM...')
        _run_planning_problem(self, debug_flag=debug_flag)

    def run_operational_planning(self, type='distributed', candidate_solution=dict(), num_steps=8,
                                 print_results=False, filename=str(), debug_flag=False,
                                 initial_state=None, return_state=False):

        if type == 'distributed':
            print('[INFO] Running OPERATIONAL PLANNING (DISTRIBUTED)...')
            if not candidate_solution:
                candidate_solution = self.get_initial_candidate_solution()
            convergence, results, models, sensitivities, primal_evolution, execution_time, state = _run_operational_planning(
                self,
                candidate_solution,
                initial_state=initial_state,
                debug_flag=debug_flag,
            )
            if print_results and not state.get('initialization_failed', False):
                if not filename:
                    filename = f'{self.name}_distributed'
                self.write_operational_planning_results_to_excel(
                    models,
                    results,
                    filename=filename,
                    primal_evolution=primal_evolution,
                    admm_diagnostics=state.get('admm_diagnostics', []),
                    execution_time=execution_time,
                )
            elif print_results:
                print('[WARNING] Operational results were not written because initialization failed.')
            output = convergence, results, models, sensitivities, primal_evolution
            if return_state:
                return (*output, state)
            return output

        elif type == 'hierarchical':
            print('[INFO] Running OPERATIONAL PLANNING (HIERARCHICAL)...')
            results, models, execution_time = _run_operational_planning_hierarchical(self, num_steps=num_steps, debug_flag=debug_flag, print_pq_map=print_results)
            if print_results:
                if not filename:
                    filename = f'{self.name}_hierarchical'
                self.write_operational_planning_results_hierarchical_to_excel(models, results, filename=filename, execution_time=execution_time)
            return results, models

        elif type == 'centralized':
            print('[INFO] Running OPERATIONAL PLANNING (CENTRALIZED)...')
            centralized_network, results, model = _run_operational_planning_centralized(self, debug_flag=debug_flag)
            if print_results:
                processed_results = centralized_network.process_results(model, results)
                filename = f'{self.name}_operational_planning_results_centralized'
                centralized_network.write_optimization_results_to_excel(processed_results, filename=filename)
            return results, model

        elif type == 'uncoordinated':
            results, models = self.run_without_coordination(print_results=print_results)
            return results, models

        else:
            print('[ERROR] Unrecognized COORDINATED OPERATIONAL PLANNING TYPE!...')
            exit(ERROR_SPECIFICATION_FILE)

    def run_without_coordination(self, print_results=False):

        print('[INFO] Running PLANNING PROBLEM WITHOUT COORDINATION...')

        # print('[INFO] \t - Transmission Network. Power factor control switched off!')
        # print('[INFO] \t - Transmission Network. Flexible load control switched off!')
        # self.transmission_network.params.fl_reg = False
        # for year in self.transmission_network.years:
        #     for day in self.transmission_network.days:
        #         for generator in self.transmission_network.network[year][day].generators:
        #             if generator.is_curtaillable():
        #                 generator.power_factor_control = False

        # print('[INFO] \t - Distribution Networks. Power factor control switched off!')
        # print('[INFO] \t - Distribution Networks. Flexible load control switched off!')
        # for node_id in self.distribution_networks:
        #     distribution_network = self.distribution_networks[node_id]
        #     distribution_network.params.fl_reg = False
        #     for year in distribution_network.years:
        #         for day in distribution_network.days:
        #             for generator in distribution_network.network[year][day].generators:
        #                 generator.power_factor_control = False

        results, models, execution_time = _run_operational_planning_without_coordination(self)
        if print_results:
            self.write_operational_planning_results_without_coordination_to_excel(models, results, execution_time=execution_time)

        # print('[INFO] \t - Transmission Network. Power factor control switched back on!')
        # print('[INFO] \t - Transmission Network. Flexible load control switched back on!')
        # self.transmission_network.params.fl_reg = True
        # for year in self.transmission_network.years:
        #     for day in self.transmission_network.days:
        #         for generator in self.transmission_network.network[year][day].generators:
        #             if generator.is_curtaillable():
        #                 generator.power_factor_control = True

        # print('[INFO] \t - Distribution Networks. Power factor control switched back on!')
        # print('[INFO] \t - Distribution Networks. Flexible load control switched back on!')
        # for node_id in self.distribution_networks:
        #     distribution_network = self.distribution_networks[node_id]
        #     distribution_network.params.fl_reg = True
        #     for year in distribution_network.years:
        #         for day in distribution_network.days:
        #             for generator in distribution_network.network[year][day].generators:
        #                 generator.power_factor_control = True

        return results, models

    def combine_networks(self):
        transmission_network = self.transmission_network
        distribution_networks = self.distribution_networks
        return combine_networks(transmission_network, distribution_networks)

    def get_operational_recourse_value(self, models):
        return _get_operational_recourse_value(self, models)

    def get_primal_value(self, tso_model, dso_models, esso_model):
        return _get_primal_value(self, tso_model, dso_models, esso_model)

    def add_benders_cut(self, model, recourse_value, sensitivities, candidate_solution):
        return _add_benders_cut(self, model, recourse_value, sensitivities, candidate_solution)

    def update_admm_consensus_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, results, params, update_tn=False, update_dns=False, update_sess=False):
        self.update_interface_power_flow_variables(tso_model, dso_models, consensus_vars, dual_vars, results, params, update_tn=update_tn, update_dns=update_dns)
        self.update_shared_energy_storage_variables(tso_model, dso_models, esso_model, consensus_vars['ess'], dual_vars['ess'], results, params, update_tn=update_tn, update_dns=update_dns, update_sess=update_sess)

    def update_interface_power_flow_variables(self, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=True, update_dns=True):
        _update_interface_power_flow_variables(self, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=update_tn, update_dns=update_dns)

    def update_shared_energy_storage_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, results, params, update_tn=True, update_dns=True, update_sess=True):
        _update_shared_energy_storage_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, results, params, update_tn=update_tn, update_dns=update_dns, update_sess=update_sess)

    def read_planning_problem(self):
        _read_planning_problem(self)

    def read_market_data_from_file(self):
        _read_market_data_from_file(self)

    def read_planning_parameters_from_file(self):
        filename = os.path.join(self.data_dir, self.params_file)
        self.params.read_parameters_from_file(filename)

    def write_planning_results_to_excel(self, master_problem_model, operational_planning_models, operational_results=dict(), bound_evolution=dict(), execution_time=float()):
        filename = os.path.join(self.results_dir, self.name + '_planning_results.xlsx')
        processed_results = _process_operational_planning_results(self, operational_planning_models['tso'], operational_planning_models['dso'], operational_planning_models['esso'], operational_results)
        shared_ess_cost = self.shared_ess_data.get_investment_cost_and_rated_capacity(master_problem_model)
        shared_ess_capacity = self.shared_ess_data.get_available_capacity(operational_planning_models['esso'])
        _write_planning_results_to_excel(self, processed_results, bound_evolution=bound_evolution, shared_ess_cost=shared_ess_cost, shared_ess_capacity=shared_ess_capacity, filename=filename, execution_time=execution_time)

    def write_operational_planning_results_to_excel(self, optimization_models, results, filename=str(),
                                                     primal_evolution=list(), admm_diagnostics=list(),
                                                     execution_time=float()):
        if not filename:
            filename = 'operational_planning_results'
        processed_results = _process_operational_planning_results(self, optimization_models['tso'], optimization_models['dso'], optimization_models['esso'], results)
        shared_ess_capacity = self.shared_ess_data.get_available_capacity(optimization_models['esso'])
        _write_operational_planning_results_to_excel(
            self,
            processed_results,
            primal_evolution=primal_evolution,
            admm_diagnostics=admm_diagnostics,
            shared_ess_capacity=shared_ess_capacity,
            filename=filename,
            execution_time=execution_time,
        )

    def write_operational_planning_results_hierarchical_to_excel(self, optimization_models, results, filename=str(), execution_time=float()):
        if not filename:
            filename = 'operational_planning_results_hierarchical'
        filename = os.path.join(self.results_dir, filename + '.xlsx')
        processed_results = _process_operational_planning_results_hierarchical(self, optimization_models['tso'], optimization_models['dso'], results)
        _write_operational_planning_results_hierarchical_to_excel(self, processed_results, filename, execution_time=execution_time)

    def write_operational_planning_results_without_coordination_to_excel(self, optimization_models, results, filename=str(), execution_time=float()):
        if not filename:
            filename = 'operational_planning_results_hierarchical_no_coordination'
        filename = os.path.join(self.results_dir, self.name + '_operational_planning_results_no_coordination.xlsx')
        processed_results = _process_operational_planning_results_no_coordination(self, optimization_models['tso'], optimization_models['dso'], results)
        _write_operational_planning_results_no_coordination_to_excel(self, processed_results, filename, execution_time=execution_time)

    def plot_market_price_scenarios(self):
        years_to_plot = list(self.years)[0]
        _plot_market_price_scenarios(self, years_to_plot=[years_to_plot], save_dir=self.diagrams_dir)

    def get_initial_candidate_solution(self):
        return _get_initial_candidate_solution(self)

    def get_test_candidate_solution(self, s_inv=1.00, e_inv=2.00):
        return _get_test_candidate_solution(self, s_inv=s_inv, e_inv=e_inv)

    def plot_diagram(self):
        _plot_networkx_diagram(self)


# ======================================================================================================================
#  PLANNING functions
# ======================================================================================================================
def _run_planning_problem(planning_problem, debug_flag=False):

    shared_ess_data = planning_problem.shared_ess_data
    benders_parameters = planning_problem.params.benders
    lower_level_models = dict()
    operational_results = dict()

    # ------------------------------------------------------------------------------------------------------------------
    # 0. Initialization
    iteration = 1
    convergence = False
    from_warm_start = False
    upper_bound = float('inf')
    master_estimate_evolution = list()
    upper_bound_evolution = list()
    investment_cost_evolution = list()
    alpha_evolution = list()
    operational_recourse_evolution = list()
    candidate_total_evolution = list()
    esso_violation_evolution = list()
    gap_signed_evolution = list()
    gap_abs_evolution = list()
    gap_rel_evolution = list()
    finite_difference_results = list()
    admm_diagnostics = list()
    operational_state = None
    print_memory_usage("Start of planning problem", debug_flag)

    start = time.time()
    master_problem_model = planning_problem.shared_ess_data.build_master_problem()
    master_result = shared_ess_data.optimize_master_problem(master_problem_model)
    if not master_result or master_result.solver.termination_condition != po.TerminationCondition.optimal:
        print("[ERROR] Benders-type master problem did not solve to optimality. Exiting planning loop.")
        return
    candidate_solution = shared_ess_data.get_candidate_solution(master_problem_model)

    # Benders-type main cycle
    while iteration <= benders_parameters.num_max_iters and not convergence:

        print(f'=============================================== ITERATION #{iteration} ==============================================')

        _print_candidate_solution(candidate_solution)
        print_memory_usage(f"Before subproblem (iter {iteration})", debug_flag)
        print_results = False
        if iteration == 1 or debug_flag:
            print_results = True

        # 1. Subproblem
        # 1.1. Solve operational planning, with fixed investment variables,
        # 1.2. Get coupling constraints' sensitivities (subproblem)
        # 1.3. Get the economic recourse value and local sensitivities
        operational_convergence, operational_results, lower_level_models, sensitivities, _, operational_state = planning_problem.run_operational_planning(
            candidate_solution=candidate_solution,
            print_results=print_results,
            filename=f'{planning_problem.name}_operational_planning_results_distributed_without ESS',
            return_state=True,
        )
        for diagnostic in operational_state.get('admm_diagnostics', []):
            diagnostic_with_outer_iteration = dict(diagnostic)
            diagnostic_with_outer_iteration['outer_iteration'] = iteration
            admm_diagnostics.append(diagnostic_with_outer_iteration)

        initialization_failed = operational_state.get('initialization_failed', False)
        investment_cost = pe.value(master_problem_model.investment_cost)
        alpha = pe.value(master_problem_model.alpha)
        master_estimate = pe.value(master_problem_model.objective)
        esso_violation = None
        if not initialization_failed:
            esso_violation = shared_ess_data.get_feasibility_violation(lower_level_models['esso'])
        candidate_is_feasible = (
            operational_convergence
            and esso_violation is not None
            and esso_violation <= BENDERS_FEASIBILITY_TOLERANCE
        )

        operational_recourse = None
        candidate_total = None
        if candidate_is_feasible:
            operational_recourse = planning_problem.get_operational_recourse_value(lower_level_models)
            candidate_total = investment_cost + operational_recourse
            upper_bound = min(upper_bound, candidate_total)

        gap_signed = None
        gap_abs = None
        gap_rel = None
        if isfinite(upper_bound):
            gap_signed = upper_bound - master_estimate
            gap_abs = abs(gap_signed)
            gap_rel = gap_abs / max(abs(upper_bound), 1e-6)

        master_estimate_evolution.append(master_estimate)
        upper_bound_evolution.append(upper_bound if isfinite(upper_bound) else None)
        investment_cost_evolution.append(investment_cost)
        alpha_evolution.append(alpha)
        operational_recourse_evolution.append(operational_recourse)
        candidate_total_evolution.append(candidate_total)
        esso_violation_evolution.append(esso_violation)
        gap_signed_evolution.append(gap_signed)
        gap_abs_evolution.append(gap_abs)
        gap_rel_evolution.append(gap_rel)

        recourse_text = f'{operational_recourse:.2f}' if operational_recourse is not None else 'N/A'
        candidate_total_text = f'{candidate_total:.2f}' if candidate_total is not None else 'N/A'
        upper_bound_text = f'{upper_bound:.2f}' if isfinite(upper_bound) else 'N/A'
        gap_text = f'{gap_signed / max(abs(upper_bound), 1e-6) * 100:.2f}%' if gap_signed is not None else 'N/A'
        esso_violation_text = f'{esso_violation:.6f}' if esso_violation is not None else 'N/A'
        print(
            f"[INFO] Iteration #{iteration} | Master = {master_estimate:.2f} | Alpha = {alpha:.2f} | "
            f"Investment = {investment_cost:.2f} | Recourse = {recourse_text} | "
            f"Candidate = {candidate_total_text} | UB = {upper_bound_text} | Gap = {gap_text} | "
            f"ESSO violation = {esso_violation_text}"
        )

        if planning_problem.params.gc:
            gc.collect()
        print_memory_usage(f"After subproblem (iter {iteration})", debug_flag)

        if not operational_convergence:
            if initialization_failed:
                print(
                    '[WARNING] Operational initialization failed. No ADMM cycle or formal Benders '
                    'feasibility cut is available; stopping the outer loop.'
                )
            else:
                print("[WARNING] ADMM did not converge. No formal Benders feasibility cut is available; stopping the outer loop.")
            break
        if esso_violation > BENDERS_FEASIBILITY_TOLERANCE:
            print(
                f"[WARNING] Shared ESS feasibility violation {esso_violation:.6f} exceeds "
                f"{BENDERS_FEASIBILITY_TOLERANCE:.6f}. No formal feasibility cut is available; stopping the outer loop."
            )
            break
        if master_estimate > upper_bound + benders_parameters.tol_abs:
            print(
                "[WARNING] The Benders-type master estimate exceeds the incumbent feasible objective. "
                "The local cuts are not global lower bounds; stopping without claiming optimality."
            )
            break
        if gap_rel < benders_parameters.tol_rel or gap_abs <= benders_parameters.tol_abs:
            convergence = True
            break
        if iteration == benders_parameters.num_max_iters:
            break

        print_memory_usage(f"Before master problem solve (iter {iteration})", debug_flag)

        # 2. Solve Master problem
        # 2.1. Add a local sensitivity cut based on the evaluated recourse value
        # 2.2. Run master problem optimization
        # 2.3. Get the next common investment plan
        cut_added = planning_problem.add_benders_cut(master_problem_model, operational_recourse, sensitivities, candidate_solution)
        if not cut_added:
            print("[WARNING] Sensitivity information is incomplete. Stopping the outer loop without adding a cut.")
            break
        master_result = shared_ess_data.optimize_master_problem(master_problem_model, from_warm_start=from_warm_start)
        if not master_result or master_result.solver.termination_condition != po.TerminationCondition.optimal:
            print("[WARNING] Benders-type master problem did not solve to optimality. Stopping the outer loop.")
            break

        if planning_problem.params.gc:
            gc.collect()
        print_memory_usage(f"After master problem solve (iter {iteration})", debug_flag)

        # Get new candidate solution
        candidate_solution = shared_ess_data.get_candidate_solution(master_problem_model)
        print_memory_usage(f"After GC (iter {iteration})", debug_flag)

        iteration += 1
        from_warm_start = True

    if convergence:
        print(f"[INFO] Benders-type procedure converged at iteration {iteration}.")
    else:
        print('[WARNING] Convergence not obtained!')

    finite_difference_params = benders_parameters.finite_difference
    if convergence and finite_difference_params.enabled:
        finite_difference_results = _validate_local_sensitivities_with_finite_differences(
            planning_problem,
            candidate_solution,
            operational_recourse,
            sensitivities,
            lower_level_models,
            operational_state,
            finite_difference_params,
        )

    # Write results
    end = time.time()
    total_execution_time = end - start
    print('[INFO] Execution time: {:.2f} s'.format(total_execution_time))
    bound_evolution = {
        'master_estimate': master_estimate_evolution,
        'lower_bound': master_estimate_evolution,
        'upper_bound': upper_bound_evolution,
        'investment_cost': investment_cost_evolution,
        'alpha': alpha_evolution,
        'operational_recourse': operational_recourse_evolution,
        'candidate_total': candidate_total_evolution,
        'esso_violation': esso_violation_evolution,
        'gap_signed': gap_signed_evolution,
        'gap_abs': gap_abs_evolution,
        'gap_rel': gap_rel_evolution,
        'finite_difference': finite_difference_results,
        'admm_diagnostics': admm_diagnostics,
    }
    if operational_state and operational_state.get('initialization_failed', False):
        print('[WARNING] Planning results were not written because the final operational initialization failed.')
    else:
        planning_problem.write_planning_results_to_excel(master_problem_model, lower_level_models, operational_results, bound_evolution, execution_time=total_execution_time)


def _get_operational_recourse_value(planning_problem, models):
    recourse_value = planning_problem.transmission_network.get_primal_value(models['tso'])
    for node_id, distribution_network in planning_problem.distribution_networks.items():
        recourse_value += distribution_network.get_primal_value(models['dso'][node_id])
    return recourse_value


def _get_operational_sensitivities(planning_problem, models):
    available_sensitivities = {'s': dict(), 'e': dict()}
    for year in planning_problem.years:
        available_sensitivities['s'][year] = {
            node_id: 0.00 for node_id in planning_problem.active_distribution_network_nodes
        }
        available_sensitivities['e'][year] = {
            node_id: 0.00 for node_id in planning_problem.active_distribution_network_nodes
        }

    local_sensitivities = [
        planning_problem.transmission_network.get_sensitivities(models['tso'])
    ]
    for node_id, distribution_network in planning_problem.distribution_networks.items():
        local_sensitivities.append(distribution_network.get_sensitivities(models['dso'][node_id]))

    for local_values in local_sensitivities:
        for capacity_type in ('s', 'e'):
            for year, node_values in local_values[capacity_type].items():
                for node_id, value in node_values.items():
                    if value is None or available_sensitivities[capacity_type][year][node_id] is None:
                        available_sensitivities[capacity_type][year][node_id] = None
                    else:
                        available_sensitivities[capacity_type][year][node_id] += value

    return planning_problem.shared_ess_data.map_available_capacity_sensitivities_to_investments(
        models['esso'], available_sensitivities
    )


def _validate_local_sensitivities_with_finite_differences(planning_problem, candidate_solution,
                                                          baseline_recourse, sensitivities, baseline_models,
                                                          baseline_state, params):
    selected = _select_finite_difference_investment(candidate_solution, params)
    if selected is None:
        print('[WARNING] Finite-difference validation skipped: no matching positive investment was found.')
        return []
    if baseline_state is None:
        print('[WARNING] Finite-difference validation skipped: the converged operational state is unavailable.')
        return []

    node_id, year = selected
    sensitivity_s = sensitivities['s'][year][node_id]
    sensitivity_e = sensitivities['e'][year][node_id]
    if sensitivity_s is None or sensitivity_e is None:
        print('[WARNING] Finite-difference validation skipped: the selected sensitivity is unavailable.')
        return []

    base_s = candidate_solution['investment'][node_id][year]['s']
    base_e = candidate_solution['investment'][node_id][year]['e']
    if isclose(base_s, 0.00, abs_tol=SMALL_TOLERANCE):
        ratio = planning_problem.shared_ess_data.params.min_energy_to_power_ratio
    else:
        ratio = base_e / base_s

    baseline_soh_margin = _get_investment_soh_margin(
        planning_problem, baseline_models['esso'], node_id, year
    )
    analytic_slope = sensitivity_s + ratio * sensitivity_e
    validation_results = []

    print('[INFO] Running finite-difference validation of the final local sensitivity...')
    print(
        f'[INFO] Selected investment: node {node_id}, year {year}, '
        f'S = {base_s:.6f} MVA, E = {base_e:.6f} MVAh, E/S = {ratio:.6f}.'
    )

    try:
        replay_convergence, _, replay_models, replay_sensitivities, _, _ = planning_problem.run_operational_planning(
            candidate_solution=candidate_solution,
            print_results=False,
            initial_state=baseline_state,
            return_state=True,
        )
        replay_esso_violation = planning_problem.shared_ess_data.get_feasibility_violation(
            replay_models['esso']
        )
        replay_recourse = None
        replay_drift = None
        replay_soh_margin = None
        replay_analytic_slope = None
        sensitivity_relative_drift = None
        replay_active_set_changed = None
        replay_tolerance = max(
            params.replay_absolute_tolerance,
            params.replay_relative_tolerance * max(abs(baseline_recourse), 1.00),
        )
        replay_reasons = []

        if not replay_convergence:
            replay_reasons.append('ADMM replay did not converge')
        else:
            replay_recourse = planning_problem.get_operational_recourse_value(replay_models)
            replay_drift = replay_recourse - baseline_recourse
            replay_soh_margin = _get_investment_soh_margin(
                planning_problem, replay_models['esso'], node_id, year
            )
            replay_sensitivity_s = replay_sensitivities['s'][year][node_id]
            replay_sensitivity_e = replay_sensitivities['e'][year][node_id]
            if replay_sensitivity_s is None or replay_sensitivity_e is None:
                replay_reasons.append('replay sensitivity is unavailable')
            else:
                replay_analytic_slope = replay_sensitivity_s + ratio * replay_sensitivity_e
                sensitivity_relative_drift = abs(replay_analytic_slope - analytic_slope) / max(
                    abs(replay_analytic_slope), abs(analytic_slope), 1.00
                )
                if sensitivity_relative_drift > params.slope_consistency_tolerance:
                    replay_reasons.append('directional sensitivity is not reproducible')

            replay_active_set_changed = _soh_active_state_changed(
                baseline_soh_margin, replay_soh_margin, params.soh_active_tolerance
            )
            if replay_active_set_changed:
                replay_reasons.append('minimum-SoH activity changed during replay')
            if abs(replay_drift) > replay_tolerance:
                replay_reasons.append('recourse replay drift exceeds tolerance')

        if replay_esso_violation > BENDERS_FEASIBILITY_TOLERANCE:
            replay_reasons.append('ESSO replay violation exceeds tolerance')

        replay_status = 'passed' if not replay_reasons else 'inconclusive'
        validation_results.append({
            'run_type': 'replay',
            'status': replay_status,
            'reason': '; '.join(replay_reasons),
            'node_id': node_id,
            'year': year,
            'base_s': base_s,
            'base_e': base_e,
            'energy_to_power_ratio': ratio,
            'step_fraction': 0.00,
            'step_size': 0.00,
            'delta_s': 0.00,
            'delta_e': 0.00,
            'sensitivity_s': sensitivity_s,
            'sensitivity_e': sensitivity_e,
            'analytic_slope': analytic_slope,
            'replay_analytic_slope': replay_analytic_slope,
            'baseline_recourse': baseline_recourse,
            'reference_recourse': replay_recourse,
            'replay_drift': replay_drift,
            'replay_tolerance': replay_tolerance,
            'sensitivity_relative_drift': sensitivity_relative_drift,
            'operational_convergence': replay_convergence,
            'esso_violation': replay_esso_violation,
            'baseline_soh_margin': baseline_soh_margin,
            'reference_soh_margin': replay_soh_margin,
            'active_set_changed': replay_active_set_changed,
            'passed': replay_status == 'passed',
        })

        drift_text = f'{replay_drift:.6f}' if replay_drift is not None else 'N/A'
        print(
            f'[INFO] Finite-difference replay | Recourse drift = {drift_text} | '
            f'Tolerance = {replay_tolerance:.6f} | ESSO violation = {replay_esso_violation:.6f} | '
            f'Status = {replay_status}'
        )
        if replay_status != 'passed':
            print('[WARNING] Finite-difference perturbations skipped: the baseline replay is not reproducible.')
            return validation_results

        noise_floor = max(abs(replay_drift), params.replay_absolute_tolerance)
        previous_observed_slope = None
        step_scale = max(abs(base_s), 1.00)

        for step_fraction in params.relative_step_sizes:
            if step_fraction <= 0.00:
                print(f'[WARNING] Ignoring non-positive relative finite-difference step {step_fraction}.')
                continue

            step_size = step_fraction * step_scale
            delta_s = step_size
            delta_e = ratio * step_size
            perturbed_candidate = deepcopy(candidate_solution)
            perturbed_candidate['investment'][node_id][year]['s'] += delta_s
            perturbed_candidate['investment'][node_id][year]['e'] += delta_e
            _rebuild_candidate_total_capacities(planning_problem, perturbed_candidate)

            predicted_change = analytic_slope * step_size
            operational_convergence, _, perturbed_models, _, _, _ = planning_problem.run_operational_planning(
                candidate_solution=perturbed_candidate,
                print_results=False,
                initial_state=baseline_state,
                return_state=True,
            )
            esso_violation = planning_problem.shared_ess_data.get_feasibility_violation(
                perturbed_models['esso']
            )

            perturbed_recourse = None
            observed_change = None
            absolute_error = None
            observed_slope = None
            absolute_slope_error = None
            relative_error = None
            same_sign = None
            signal_to_noise_ratio = None
            slope_consistency_error = None
            perturbed_soh_margin = None
            active_set_changed = None
            status = 'inconclusive'
            reasons = []

            if not operational_convergence:
                reasons.append('ADMM did not converge')
            else:
                perturbed_recourse = planning_problem.get_operational_recourse_value(perturbed_models)
                observed_change = perturbed_recourse - replay_recourse
                absolute_error = abs(observed_change - predicted_change)
                observed_slope = observed_change / step_size
                absolute_slope_error = abs(observed_slope - analytic_slope)
                relative_error = absolute_slope_error / max(
                    abs(observed_slope), abs(analytic_slope), 1.00
                )
                same_sign = (
                    analytic_slope * observed_slope >= 0.00
                    or (
                        isclose(analytic_slope, 0.00, abs_tol=1.00)
                        and isclose(observed_slope, 0.00, abs_tol=1.00)
                    )
                )
                signal_to_noise_ratio = abs(observed_change) / noise_floor
                perturbed_soh_margin = _get_investment_soh_margin(
                    planning_problem, perturbed_models['esso'], node_id, year
                )
                active_set_changed = _soh_active_state_changed(
                    replay_soh_margin, perturbed_soh_margin, params.soh_active_tolerance
                )
                if previous_observed_slope is not None:
                    slope_consistency_error = abs(observed_slope - previous_observed_slope) / max(
                        abs(observed_slope), abs(previous_observed_slope), 1.00
                    )

                if esso_violation > BENDERS_FEASIBILITY_TOLERANCE:
                    reasons.append('ESSO violation exceeds tolerance')
                if active_set_changed:
                    reasons.append('minimum-SoH activity changed')
                if signal_to_noise_ratio < params.minimum_signal_to_noise_ratio:
                    reasons.append('finite-difference signal is below the noise threshold')

                if reasons:
                    status = 'inconclusive'
                elif not same_sign:
                    status = 'failed'
                    reasons.append('analytic and observed slopes have different signs')
                elif relative_error > params.relative_error_tolerance:
                    status = 'failed'
                    reasons.append('relative slope error exceeds tolerance')
                elif (
                        slope_consistency_error is not None
                        and slope_consistency_error > params.slope_consistency_tolerance):
                    status = 'failed'
                    reasons.append('finite-difference slopes are not consistent across step sizes')
                else:
                    status = 'passed'

                previous_observed_slope = observed_slope

            result = {
                'run_type': 'perturbation',
                'status': status,
                'reason': '; '.join(reasons),
                'node_id': node_id,
                'year': year,
                'base_s': base_s,
                'base_e': base_e,
                'energy_to_power_ratio': ratio,
                'step_fraction': step_fraction,
                'step_size': step_size,
                'delta_s': delta_s,
                'delta_e': delta_e,
                'sensitivity_s': sensitivity_s,
                'sensitivity_e': sensitivity_e,
                'analytic_slope': analytic_slope,
                'replay_analytic_slope': replay_analytic_slope,
                'predicted_change': predicted_change,
                'baseline_recourse': baseline_recourse,
                'reference_recourse': replay_recourse,
                'perturbed_recourse': perturbed_recourse,
                'observed_change': observed_change,
                'absolute_error': absolute_error,
                'observed_slope': observed_slope,
                'absolute_slope_error': absolute_slope_error,
                'relative_error': relative_error,
                'signal_to_noise_ratio': signal_to_noise_ratio,
                'slope_consistency_error': slope_consistency_error,
                'same_sign': same_sign,
                'operational_convergence': operational_convergence,
                'esso_violation': esso_violation,
                'baseline_soh_margin': baseline_soh_margin,
                'reference_soh_margin': replay_soh_margin,
                'perturbed_soh_margin': perturbed_soh_margin,
                'active_set_changed': active_set_changed,
                'replay_drift': replay_drift,
                'replay_tolerance': replay_tolerance,
                'sensitivity_relative_drift': sensitivity_relative_drift,
                'passed': status == 'passed',
            }
            validation_results.append(result)

            observed_text = f'{observed_change:.6f}' if observed_change is not None else 'N/A'
            error_text = f'{relative_error * 100:.2f}%' if relative_error is not None else 'N/A'
            print(
                f'[INFO] Finite difference h = {step_size:.6f} ({step_fraction:.2%}) | '
                f'Predicted change = {predicted_change:.6f} | '
                f'Observed change = {observed_text} | Relative error = {error_text} | '
                f'ESSO violation = {esso_violation:.6f} | Status = {status}'
            )
    finally:
        _restore_candidate_data(planning_problem, candidate_solution)

    return validation_results


def _soh_active_state_changed(baseline_margin, candidate_margin, tolerance):
    if baseline_margin is None or candidate_margin is None:
        return None
    return (baseline_margin <= tolerance) != (candidate_margin <= tolerance)


def _select_finite_difference_investment(candidate_solution, params):
    investments = candidate_solution['investment']

    if params.node_id is not None and params.year is not None:
        node_id = next((value for value in investments if str(value) == str(params.node_id)), None)
        if node_id is None:
            return None
        year = next((value for value in investments[node_id] if str(value) == str(params.year)), None)
        if year is None:
            return None
        return node_id, year

    selected = None
    selected_power = -float('inf')
    for node_id, yearly_investments in investments.items():
        for year, investment in yearly_investments.items():
            if investment['s'] > selected_power and (
                    investment['s'] > SMALL_TOLERANCE or investment['e'] > SMALL_TOLERANCE):
                selected = node_id, year
                selected_power = investment['s']
    return selected


def _rebuild_candidate_total_capacities(planning_problem, candidate_solution):
    shared_ess_data = planning_problem.shared_ess_data
    years = list(shared_ess_data.years)

    for node_id in candidate_solution['investment']:
        candidate_solution['total_capacity'][node_id] = {
            year: {'s': 0.00, 'e': 0.00} for year in years
        }
        shared_ess_idx = shared_ess_data.get_shared_energy_storage_idx(node_id)
        for y_inv, year_inv in enumerate(years):
            shared_energy_storage = shared_ess_data.shared_energy_storages[year_inv][shared_ess_idx]
            num_years = shared_ess_data.years[year_inv]
            tcal_norm = round(shared_energy_storage.t_cal / num_years)
            max_tcal_norm = min(y_inv + tcal_norm, len(years))
            investment = candidate_solution['investment'][node_id][year_inv]
            for y in range(y_inv, max_tcal_norm):
                year = years[y]
                candidate_solution['total_capacity'][node_id][year]['s'] += investment['s']
                candidate_solution['total_capacity'][node_id][year]['e'] += investment['e']


def _get_investment_soh_margin(planning_problem, esso_models, node_id, year_inv):
    years = list(planning_problem.shared_ess_data.years)
    y_inv = years.index(year_inv)
    model = esso_models[node_id]
    shared_ess_idx = planning_problem.shared_ess_data.get_shared_energy_storage_idx(node_id)
    shared_energy_storage = planning_problem.shared_ess_data.shared_energy_storages[year_inv][shared_ess_idx]
    margins = []

    for y in model.years:
        if not model.es_soh_per_unit_cumul[y_inv, y].fixed:
            margins.append(
                pe.value(model.es_soh_per_unit_cumul[y_inv, y]) - shared_energy_storage.soh_min
            )
    return min(margins) if margins else None


def _restore_candidate_data(planning_problem, candidate_solution):
    total_capacity = candidate_solution['total_capacity']
    planning_problem.transmission_network.update_data_with_candidate_solution(total_capacity)
    for distribution_network in planning_problem.distribution_networks.values():
        distribution_network.update_data_with_candidate_solution(total_capacity)
    planning_problem.shared_ess_data.update_data_with_candidate_solution(
        candidate_solution['investment']
    )


def _add_benders_cut(planning_problem, model, recourse_value, sensitivities, candidate_solution):
    years = [year for year in planning_problem.years]
    print("[INFO] Benders-type procedure. Adding local sensitivity cut...")
    benders_cut = recourse_value
    for e in model.energy_storages:
        node_id = planning_problem.active_distribution_network_nodes[e]
        for y in model.years:
            year = years[y]
            sensitivity_s = sensitivities['s'][year][node_id]
            sensitivity_e = sensitivities['e'][year][node_id]
            if sensitivity_s is None or sensitivity_e is None:
                return False
            benders_cut += sensitivity_s * (
                model.es_s_investment[e, y] - candidate_solution['investment'][node_id][year]['s']
            )
            benders_cut += sensitivity_e * (
                model.es_e_investment[e, y] - candidate_solution['investment'][node_id][year]['e']
            )
    model.benders_cuts.add(model.alpha >= benders_cut)
    return True


# ======================================================================================================================
#  OPERATIONAL PLANNING (DISTRIBUTED)
# ======================================================================================================================
def _run_operational_planning(planning_problem, candidate_solution, initial_state=None, debug_flag=False):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data
    admm_parameters = planning_problem.params.admm
    results = {'tso': dict(), 'dso': dict(), 'esso': dict()}

    # ------------------------------------------------------------------------------------------------------------------
    # 0. Initialization

    print('[INFO]\t - Initializing...')

    start = time.time()
    if initial_state is not None and initial_state.get('initialization_failed', False):
        print('[WARNING] Ignoring a previously failed operational state and rebuilding initialization.')
        initial_state = None
    from_warm_start = initial_state is not None
    primal_evolution = list()
    admm_diagnostics = list()
    continuing_same_candidate = (
        initial_state is not None
        and initial_state.get('candidate_solution') == candidate_solution
    )
    previous_recourse = (
        initial_state.get('last_recourse')
        if continuing_same_candidate else None
    )
    consecutive_converged_cycles = (
        initial_state.get('consecutive_converged_cycles', 0)
        if continuing_same_candidate else 0
    )

    if initial_state is None:
        # Create ADMM variables and obtain the initial local solutions.
        consensus_vars, dual_vars = create_admm_variables(planning_problem)
        dso_models, results['dso'] = create_distribution_networks_models(
            distribution_networks,
            consensus_vars,
            candidate_solution['total_capacity'],
            parallel_execution=planning_problem.parallel_execution,
        )
        tso_model, results['tso'] = create_transmission_network_model(
            planning_problem, consensus_vars, candidate_solution['total_capacity']
        )
        esso_model, results['esso'] = create_shared_energy_storage_model(
            shared_ess_data, consensus_vars, candidate_solution['investment']
        )

        if not _admm_local_solves_succeeded(planning_problem, results):
            print(
                '[WARNING] Operational initialization failed because at least one local problem '
                'did not solve successfully. ADMM will not be started.'
            )
            end = time.time()
            total_execution_time = end - start
            print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))
            optim_models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
            state = {
                'models': optim_models,
                'consensus_vars': deepcopy(consensus_vars),
                'dual_vars': deepcopy(dual_vars),
                'candidate_solution': deepcopy(candidate_solution),
                'last_recourse': None,
                'consecutive_converged_cycles': 0,
                'admm_diagnostics': admm_diagnostics,
                'initialization_failed': True,
            }
            return (
                False,
                results,
                optim_models,
                None,
                primal_evolution,
                total_execution_time,
                state,
            )

        update_distribution_models_to_admm(planning_problem, dso_models, admm_parameters)
        update_transmission_model_to_admm(planning_problem, tso_model, admm_parameters)
        update_shared_energy_storage_model_to_admm(planning_problem, esso_model, admm_parameters)

        planning_problem.update_admm_consensus_variables(
            tso_model, dso_models, esso_model,
            consensus_vars, dual_vars, results, admm_parameters,
            update_tn=True, update_dns=True, update_sess=True,
        )
    else:
        models = _clone_operational_models(initial_state['models'])
        tso_model = models['tso']
        dso_models = models['dso']
        esso_model = models['esso']
        consensus_vars = deepcopy(initial_state['consensus_vars'])
        dual_vars = deepcopy(initial_state['dual_vars'])
        _update_operational_models_with_candidate(planning_problem, models, candidate_solution)

    sess_available_capacities = shared_ess_data.get_updated_capacities(esso_model)
    # if debug_flag:
    #     print("[DEBUG] available_capacities:")
    #     for node_id in sess_available_capacities:
    #         print(f"\t{node_id}:")
    #         for year in sess_available_capacities[node_id]:
    #             print(f"\t{year}: {sess_available_capacities[node_id][year]}")

    # ------------------------------------------------------------------------------------------------------------------
    # ADMM -- Main cycle
    # ------------------------------------------------------------------------------------------------------------------
    convergence = False
    for iter in range(1, admm_parameters.num_max_iters + 1):

        print(f'[INFO] \t - ADMM Iteration {iter}')
        log_debug(f"\t - Memory before iteration {iter}", debug_flag)
        print_memory_usage(f"\t - ADMM Iteration {iter} Start", debug_flag)

        iter_start = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # 1. Solve DSOs problems
        results['dso'] = update_distribution_coordination_models_and_solve(
            distribution_networks, dso_models,
            consensus_vars['vmag'], dual_vars['vmag']['dso'],
            consensus_vars['pf'], dual_vars['pf']['dso'],
            consensus_vars['ess'], dual_vars['ess']['dso'],
            admm_parameters,
            sess_available_capacities,
            from_warm_start=from_warm_start, parallel_execution=planning_problem.parallel_execution
        )

        # Update ADMM consensus variables and primal diagnostics.
        update_and_check_convergence(
            planning_problem, tso_model, dso_models, esso_model,
            consensus_vars, dual_vars, results, admm_parameters,
            primal_evolution,
            update_flags={"update_tn": False, "update_dns": True, "update_sess": False},
            debug_flag=debug_flag,
            check_convergence=False,
        )

        # --------------------------------------------------------------------------------------------------------------
        # 2. Solve TSO problem
        results['tso'] = update_transmission_coordination_model_and_solve(
            transmission_network, tso_model,
            consensus_vars['vmag'], dual_vars['vmag']['tso'],
            consensus_vars['pf'], dual_vars['pf']['tso'],
            consensus_vars['ess'], dual_vars['ess']['tso'],
            admm_parameters,
            sess_available_capacities,
            from_warm_start=from_warm_start
        )

        # Update ADMM consensus variables and primal diagnostics.
        update_and_check_convergence(
            planning_problem, tso_model, dso_models, esso_model,
            consensus_vars, dual_vars, results, admm_parameters,
            primal_evolution,
            update_flags={"update_tn": True, "update_dns": False, "update_sess": False},
            debug_flag=debug_flag,
            check_convergence=False,
        )

        # --------------------------------------------------------------------------------------------------------------
        # 3. Solve ESSO problem
        results['esso'] = update_shared_energy_storages_coordination_model_and_solve(
            planning_problem, esso_model,
            consensus_vars['ess']['tso'], dual_vars['ess']['esso'],
            admm_parameters, from_warm_start=from_warm_start
        )

        # Update the final block and evaluate convergence only after a complete cycle.
        update_and_check_convergence(
            planning_problem, tso_model, dso_models, esso_model,
            consensus_vars, dual_vars, results, admm_parameters,
            primal_evolution,
            update_flags={"update_tn": False, "update_dns": False, "update_sess": True},
            debug_flag=debug_flag,
            check_convergence=False,
        )

        residual_metrics = get_admm_residual_metrics(
            planning_problem,
            tso_model,
            dso_models,
            esso_model,
            consensus_vars,
        )
        local_solves_ok = _admm_local_solves_succeeded(planning_problem, results)
        residual_convergence = check_admm_convergence(
            planning_problem,
            consensus_vars,
            residual_metrics,
            admm_parameters,
            debug_flag=debug_flag,
        )
        if not local_solves_ok:
            print('[WARNING]\t\t - At least one local ADMM problem did not solve successfully.')
            residual_convergence = False

        recourse = None
        objective_change_abs = None
        objective_change_rel = None
        objective_tolerance = None
        objective_convergence = False
        if local_solves_ok:
            recourse = _get_operational_recourse_value(
                planning_problem,
                {'tso': tso_model, 'dso': dso_models, 'esso': esso_model},
            )
            if previous_recourse is not None:
                objective_change_abs = abs(recourse - previous_recourse)
                objective_scale = max(abs(recourse), abs(previous_recourse), 1.0)
                objective_change_rel = objective_change_abs / objective_scale
                objective_tolerance = max(
                    admm_parameters.tol['objective']['abs'],
                    admm_parameters.tol['objective']['rel'] * objective_scale,
                )
                objective_convergence = objective_change_abs <= objective_tolerance

        if recourse is None:
            print('[INFO]\t\t - Recourse stationarity unavailable after a failed local solve.')
        elif objective_change_abs is None:
            print('[INFO]\t\t - Recourse stationarity requires one previous successful cycle.')
        elif objective_convergence:
            print('[INFO]\t\t - Recourse stationarity ok!')
        else:
            print(
                f'[INFO]\t\t - Recourse stationarity failed. '
                f'{objective_change_abs:.6f} > {objective_tolerance:.6f}'
            )

        cycle_convergence = residual_convergence and objective_convergence
        if cycle_convergence:
            consecutive_converged_cycles += 1
        else:
            consecutive_converged_cycles = 0
        convergence = (
            consecutive_converged_cycles
            >= admm_parameters.minimum_consecutive_converged_cycles
        )

        penalty_actions, penalties_before, penalties_after = _update_admm_penalties(
            tso_model,
            dso_models,
            esso_model,
            residual_metrics,
            admm_parameters,
            allow_update=local_solves_ok,
        )
        admm_diagnostics.append({
            'cycle': iter,
            'local_solves_ok': local_solves_ok,
            'primal_v': residual_metrics['primal']['v'],
            'primal_pf': residual_metrics['primal']['pf'],
            'primal_ess': residual_metrics['primal']['ess'],
            'primal_v_tolerance': admm_parameters.tol['consensus']['v'],
            'primal_pf_tolerance': admm_parameters.tol['consensus']['pf'],
            'primal_ess_tolerance': admm_parameters.tol['consensus']['ess'],
            'dual_v': residual_metrics['dual']['v'],
            'dual_pf': residual_metrics['dual']['pf'],
            'dual_ess': residual_metrics['dual']['ess'],
            'dual_v_tolerance': admm_parameters.tol['stationarity']['v'],
            'dual_pf_tolerance': admm_parameters.tol['stationarity']['pf'],
            'dual_ess_tolerance': admm_parameters.tol['stationarity']['ess'],
            'primal_v_ratio': residual_metrics['primal']['v'] / admm_parameters.tol['consensus']['v'],
            'primal_pf_ratio': residual_metrics['primal']['pf'] / admm_parameters.tol['consensus']['pf'],
            'primal_ess_ratio': residual_metrics['primal']['ess'] / admm_parameters.tol['consensus']['ess'],
            'dual_v_ratio': residual_metrics['dual']['v'] / admm_parameters.tol['stationarity']['v'],
            'dual_pf_ratio': residual_metrics['dual']['pf'] / admm_parameters.tol['stationarity']['pf'],
            'dual_ess_ratio': residual_metrics['dual']['ess'] / admm_parameters.tol['stationarity']['ess'],
            'recourse': recourse,
            'objective_change_abs': objective_change_abs,
            'objective_change_rel': objective_change_rel,
            'objective_tolerance': objective_tolerance,
            'objective_absolute_tolerance': admm_parameters.tol['objective']['abs'],
            'objective_relative_tolerance': admm_parameters.tol['objective']['rel'],
            'residual_convergence': residual_convergence,
            'objective_convergence': objective_convergence,
            'cycle_convergence': cycle_convergence,
            'consecutive_converged_cycles': consecutive_converged_cycles,
            'required_consecutive_cycles': admm_parameters.minimum_consecutive_converged_cycles,
            'rho_v_before': penalties_before['v'],
            'rho_pf_before': penalties_before['pf'],
            'rho_ess_before': penalties_before['ess'],
            'rho_v_after': penalties_after['v'],
            'rho_pf_after': penalties_after['pf'],
            'rho_ess_after': penalties_after['ess'],
            'rho_v_action': penalty_actions['v'],
            'rho_pf_action': penalty_actions['pf'],
            'rho_ess_action': penalty_actions['ess'],
        })

        objective_change_text = (
            f'{objective_change_abs:.6f}'
            if objective_change_abs is not None else 'N/A'
        )
        objective_tolerance_text = (
            f'{objective_tolerance:.6f}'
            if objective_tolerance is not None else 'N/A'
        )
        print(
            f'[INFO]\t\t - ADMM cycle {iter} | '
            f'Primal (V/PF/ESS) = '
            f'{residual_metrics["primal"]["v"]:.6f}/'
            f'{residual_metrics["primal"]["pf"]:.6f}/'
            f'{residual_metrics["primal"]["ess"]:.6f} | '
            f'Dual (V/PF/ESS) = '
            f'{residual_metrics["dual"]["v"]:.6f}/'
            f'{residual_metrics["dual"]["pf"]:.6f}/'
            f'{residual_metrics["dual"]["ess"]:.6f} | '
            f'Recourse change = {objective_change_text} '
            f'(tol. {objective_tolerance_text}) | '
            f'Stable cycles = {consecutive_converged_cycles}/'
            f'{admm_parameters.minimum_consecutive_converged_cycles} | '
            f'Penalty actions (V/PF/ESS) = '
            f'{penalty_actions["v"]}/{penalty_actions["pf"]}/{penalty_actions["ess"]}'
        )

        previous_recourse = recourse

        sess_available_capacities = shared_ess_data.get_updated_capacities(esso_model)
        if debug_flag:
            print("[DEBUG] available_capacities:")
            for node_id in sess_available_capacities:
                print(f"\t{node_id}:")
                for year in sess_available_capacities[node_id]:
                    print(f"\t{year}: {sess_available_capacities[node_id][year]}")

        # --------------------------------------------------------------------------------------------------------------

        iter_end = time.time()
        print(f"[INFO] \t - Iteration {iter}: {iter_end - iter_start:.2f} s")

        if convergence:
            print(f"[INFO] \t - ADMM converged in {iter} iteration(s).")
            break

        if planning_problem.params.gc:
            gc.collect()
        from_warm_start = True
        log_debug(f"\t - Memory after iteration {iter}", debug_flag)
        print_memory_usage(f"\t - ADMM Iteration {iter} End", debug_flag)

    if not convergence:
        print(f'[WARNING] \t - ADMM did NOT converge in {admm_parameters.num_max_iters} iterations!')

    end = time.time()
    total_execution_time = end - start
    print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))

    optim_models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
    sensitivities = None
    if convergence:
        sensitivities = _get_operational_sensitivities(planning_problem, optim_models)

    state = {
        'models': optim_models,
        'consensus_vars': deepcopy(consensus_vars),
        'dual_vars': deepcopy(dual_vars),
        'candidate_solution': deepcopy(candidate_solution),
        'last_recourse': previous_recourse,
        'consecutive_converged_cycles': consecutive_converged_cycles,
        'admm_diagnostics': admm_diagnostics,
        'initialization_failed': False,
    }
    return convergence, results, optim_models, sensitivities, primal_evolution, total_execution_time, state


def _clone_operational_models(models):
    if isinstance(models, dict):
        return {key: _clone_operational_models(value) for key, value in models.items()}
    return models.clone()


def _update_operational_models_with_candidate(planning_problem, models, candidate_solution):
    total_capacity = candidate_solution['total_capacity']
    investment = candidate_solution['investment']

    transmission_network = planning_problem.transmission_network
    transmission_network.update_data_with_candidate_solution(total_capacity)
    transmission_network.update_model_with_candidate_solution(models['tso'], total_capacity)

    for node_id, distribution_network in planning_problem.distribution_networks.items():
        distribution_network.update_data_with_candidate_solution(total_capacity)
        distribution_network.update_model_with_candidate_solution(models['dso'][node_id], total_capacity)

    planning_problem.shared_ess_data.update_data_with_candidate_solution(investment)
    planning_problem.shared_ess_data.update_model_with_candidate_solution(models['esso'], investment)


def update_and_check_convergence(planning_problem, tso_model, dso_models, esso_model,
                                 consensus_vars, dual_vars, results, admm_parameters,
                                 primal_evolution, update_flags, debug_flag=False,
                                 check_convergence=True):

    planning_problem.update_admm_consensus_variables(
        tso_model, dso_models, esso_model,
        consensus_vars, dual_vars, results, admm_parameters,
        **update_flags
    )
    primal_value = planning_problem.get_primal_value(tso_model, dso_models, esso_model)
    primal_evolution.append(primal_value)

    if not check_convergence:
        return False
    residual_metrics = get_admm_residual_metrics(
        planning_problem,
        tso_model,
        dso_models,
        esso_model,
        consensus_vars,
    )
    return check_admm_convergence(
        planning_problem,
        consensus_vars,
        residual_metrics,
        admm_parameters,
        debug_flag=debug_flag,
    )


def print_debug_info(planning_problem, consensus_vars, print_vmag=False, print_pf=False, print_ess=False):
    for node_id in planning_problem.active_distribution_network_nodes:
        for year in planning_problem.years:
            if any([print_vmag, print_pf, print_ess]):
                print(f"\tYear {year}")
            for day in planning_problem.days:
                if any([print_vmag, print_pf, print_ess]):
                    print(f"\t\tDay {day}")
                if print_vmag:
                    print(f"\t\tNode {node_id}, {year}, {day}, PF, TSO,  V  {[vmag for vmag in consensus_vars['vmag']['tso']['current'][node_id][year][day]]}")
                    print(f"\t\tNode {node_id}, {year}, {day}, PF, DSO,  V  {[vmag for vmag in consensus_vars['vmag']['dso']['current'][node_id][year][day]]}")
                # if print_pf:
                #     print(f"\t\tNode {node_id}, {year}, {day}, PF, TSO,  P {consensus_vars['pf']['tso']['current'][node_id][year][day]['p']}")
                #     print(f"\t\tNode {node_id}, {year}, {day}, PF, DSO,  P {consensus_vars['pf']['dso']['current'][node_id][year][day]['p']}")
                #     print(f"\t\tNode {node_id}, {year}, {day}, PF, TSO,  Q {consensus_vars['pf']['tso']['current'][node_id][year][day]['q']}")
                #     print(f"\t\tNode {node_id}, {year}, {day}, PF, DSO,  Q {consensus_vars['pf']['dso']['current'][node_id][year][day]['q']}")
                # if print_ess:
                #     print(f"\t\tNode {node_id}, {year}, {day}, ESS, TSO,  P {consensus_vars['ess']['tso']['current'][node_id][year][day]['p']}")
                #     print(f"\t\tNode {node_id}, {year}, {day}, ESS, DSO,  P {consensus_vars['ess']['dso']['current'][node_id][year][day]['p']}")
                #     print(f"\t\tNode {node_id}, {year}, {day}, ESS, TSO,  Q {consensus_vars['ess']['tso']['current'][node_id][year][day]['q']}")
                #     print(f"\t\tNode {node_id}, {year}, {day}, ESS, DSO,  Q {consensus_vars['ess']['dso']['current'][node_id][year][day]['q']}")


def create_transmission_network_model(planning_problem, consensus_vars, candidate_solution):

    print(f'[INFO] \t\t - Transmission Network...')

    # Build model, fix candidate solution
    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    transmission_network.update_data_with_candidate_solution(candidate_solution)
    tso_model = transmission_network.build_model()
    transmission_network.update_model_with_candidate_solution(tso_model, candidate_solution)

    # Update model with expected interface values
    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA
            tso_model[year][day].active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))

            # Free Vmag, Pc, Qc at the interface nodes, fix base Pc and Qc profiles
            for dn in tso_model[year][day].active_distribution_networks:

                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                v_min, v_max = transmission_network.network[year][day].get_node_voltage_limits(adn_node_id)
                adn_node_idx = transmission_network.network[year][day].get_node_idx(adn_node_id)
                adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
                distribution_network = distribution_networks[adn_node_id]
                interface_transf_rating = distribution_network.network[year][day].get_interface_branch_rating() / s_base

                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:

                            # Interface voltage, free vmag_adn, remove slacks
                            tso_model[year][day].vmag[adn_node_idx, s_m, s_o, p].setub(v_max + EQUALITY_TOLERANCE)
                            tso_model[year][day].vmag[adn_node_idx, s_m, s_o, p].setlb(v_min - EQUALITY_TOLERANCE)
                            if transmission_network.params.slacks.grid_operation.voltage:
                                tso_model[year][day].slack_e_up[adn_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].slack_e_down[adn_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].slack_f_up[adn_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].slack_f_down[adn_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)

                            # Fix Pc and Qc (base profiles), free pc_adn and qc_adn
                            interface_pf_p = consensus_vars['pf']['dso']['current'][adn_node_id][year][day]['p'][p] / s_base
                            interface_pf_q = consensus_vars['pf']['dso']['current'][adn_node_id][year][day]['q'][p] / s_base

                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setlb(None)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setlb(None)
                            fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, s_o, p], interface_pf_p)
                            fix_or_set(tso_model[year][day].qc[adn_load_idx, s_m, s_o, p], interface_pf_q)

                            tso_model[year][day].flex_p_up[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].flex_p_down[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].flex_q_up[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].flex_q_down[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].flex_p_up[adn_load_idx, s_m, s_o, p].setub(interface_transf_rating)
                            tso_model[year][day].flex_p_down[adn_load_idx, s_m, s_o, p].setub(interface_transf_rating)
                            tso_model[year][day].flex_q_up[adn_load_idx, s_m, s_o, p].setub(interface_transf_rating)
                            tso_model[year][day].flex_q_down[adn_load_idx, s_m, s_o, p].setub(interface_transf_rating)

            # Add expected interface and shared ESS values, and their definition
            tso_model[year][day].expected_interface_vmag = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.NonNegativeReals, bounds=partial(expected_interface_vmag_bounds, network=transmission_network.network[year][day]), initialize=1.0)
            tso_model[year][day].expected_interface_pf_p = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
            tso_model[year][day].expected_interface_pf_q = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
            tso_model[year][day].expected_shared_ess_p = pe.Var(tso_model[year][day].shared_energy_storages, tso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
            tso_model[year][day].expected_shared_ess_q = pe.Var(tso_model[year][day].shared_energy_storages, tso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
            tso_model[year][day].expected_interface_vmag_def = pe.Constraint( tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_vmag_rule, network=transmission_network.network[year][day]))
            tso_model[year][day].expected_interface_pf_p_def = pe.Constraint(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_pf_p_rule, network=transmission_network.network[year][day]))
            tso_model[year][day].expected_interface_pf_q_def = pe.Constraint(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_pf_q_rule, network=transmission_network.network[year][day]))
            tso_model[year][day].expected_shared_ess_p_def = pe.Constraint(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_sess_p_rule, network=transmission_network.network[year][day]))
            tso_model[year][day].expected_shared_ess_q_def = pe.Constraint(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_sess_q_rule, network=transmission_network.network[year][day]))

            # Regularization -- Added to OF to minimize deviations from scenarios to expected values
            obj = copy(tso_model[year][day].objective.expr)
            tso_model[year][day].penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION)
            for s_m in tso_model[year][day].scenarios_market:
                for s_o in tso_model[year][day].scenarios_operation:
                    for p in tso_model[year][day].periods:
                        for dn in tso_model[year][day].active_distribution_networks:
                            obj += tso_model[year][day].penalty_regularization * (tso_model[year][day].vmag_adn[dn, s_m, s_o, p] - tso_model[year][day].expected_interface_vmag[dn, p]) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].pc_adn[dn, s_m, s_o, p] - tso_model[year][day].expected_interface_pf_p[dn, p]) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].qc_adn[dn, s_m, s_o, p] - tso_model[year][day].expected_interface_pf_q[dn, p]) ** 2
                        for e in tso_model[year][day].shared_energy_storages:
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].shared_es_pnet[e, s_m, s_o, p] - tso_model[year][day].expected_shared_ess_p[e, p]) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].shared_es_qnet[e, s_m, s_o, p] - tso_model[year][day].expected_shared_ess_q[e, p]) ** 2
            tso_model[year][day].objective.expr = obj

    # Run SMOPF
    results = transmission_network.optimize(tso_model)

    # Get initial interface and shared ESS values
    for year in transmission_network.years:
        for day in transmission_network.days:
            if not _solver_result_succeeded(results[year][day]):
                continue
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                v_base = transmission_network.network[year][day].get_node_base_kv(adn_node_id)
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(adn_node_id)
                for p in tso_model[year][day].periods:
                    interface_vmag = pe.value(tso_model[year][day].expected_interface_vmag[dn, p]) * v_base
                    interface_pf_p = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    interface_pf_q = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base
                    p_ess = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base
                    q_ess = pe.value(tso_model[year][day].expected_shared_ess_q[shared_ess_idx, p]) * s_base
                    consensus_vars['vmag']['tso']['current'][adn_node_id][year][day][p] = interface_vmag
                    consensus_vars['pf']['tso']['current'][adn_node_id][year][day]['p'][p] = interface_pf_p
                    consensus_vars['pf']['tso']['current'][adn_node_id][year][day]['q'][p] = interface_pf_q
                    consensus_vars['ess']['tso']['current'][adn_node_id][year][day]['p'][p] = p_ess
                    consensus_vars['ess']['tso']['current'][adn_node_id][year][day]['q'][p] = q_ess

    return tso_model, results


def create_distribution_networks_models(distribution_networks, consensus_vars, candidate_solution, parallel_execution=False):
    if parallel_execution:
        return create_distribution_networks_models_parallel(distribution_networks, consensus_vars, candidate_solution)
    else:
        return create_distribution_networks_models_sequential(distribution_networks, consensus_vars, candidate_solution)


def create_distribution_networks_models_sequential(distribution_networks, consensus_vars, candidate_solution):

    dso_models = dict()
    results = dict()

    for node_id in distribution_networks:

        print(f'[INFO] \t\t - Distribution Network node {node_id}...')

        distribution_network = distribution_networks[node_id]

        # Build model, fix candidate solution
        distribution_network.update_data_with_candidate_solution(candidate_solution)
        dso_model = distribution_network.build_model()
        distribution_network.update_model_with_candidate_solution(dso_model, candidate_solution)

        for year in distribution_network.years:
            for day in distribution_network.days:

                s_base = distribution_network.network[year][day].baseMVA
                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                v_min, v_max = distribution_network.network[year][day].get_node_voltage_limits(ref_node_id)
                shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)

                # Add interface expected variables, and definition
                dso_model[year][day].expected_interface_vmag = pe.Var(dso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=1.00, bounds=(v_min - EQUALITY_TOLERANCE, v_max + EQUALITY_TOLERANCE))
                dso_model[year][day].expected_interface_pf_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
                dso_model[year][day].expected_interface_pf_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
                dso_model[year][day].expected_shared_ess_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
                dso_model[year][day].expected_shared_ess_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
                dso_model[year][day].expected_interface_vmag_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_vmag_rule, network=distribution_network.network[year][day]))
                dso_model[year][day].expected_interface_pf_p_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_pf_p_rule, network=distribution_network.network[year][day]))
                dso_model[year][day].expected_interface_pf_q_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_pf_q_rule, network=distribution_network.network[year][day]))
                dso_model[year][day].expected_shared_ess_p_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_sess_p_rule, network=distribution_network.network[year][day], shared_ess_idx=shared_ess_idx))
                dso_model[year][day].expected_shared_ess_q_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_sess_q_rule, network=distribution_network.network[year][day], shared_ess_idx=shared_ess_idx))

                # Regularization -- Added to OF to minimize deviations from scenarios to expected values
                obj = copy(dso_model[year][day].objective.expr)
                dso_model[year][day].penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION)
                for s_m in dso_model[year][day].scenarios_market:
                    for s_o in dso_model[year][day].scenarios_operation:
                        for p in dso_model[year][day].periods:
                            obj += dso_model[year][day].penalty_regularization * (dso_model[year][day].vmag_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_vmag[p]) ** 2
                            obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].pg_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_pf_p[p]) ** 2
                            obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].qg_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_pf_q[p]) ** 2
                            obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].shared_es_pnet[shared_ess_idx, s_m, s_o, p] - dso_model[year][day].expected_shared_ess_p[p]) ** 2
                            obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].shared_es_qnet[shared_ess_idx, s_m, s_o, p] - dso_model[year][day].expected_shared_ess_q[p]) ** 2
                dso_model[year][day].objective.expr = obj

        # Run SMOPF
        results[node_id] = distribution_network.optimize(dso_model)

        # Get initial interface and shared ESS values
        for year in distribution_network.years:
            for day in distribution_network.days:
                if not _solver_result_succeeded(results[node_id][year][day]):
                    continue
                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                s_base = distribution_network.network[year][day].baseMVA
                v_base = distribution_network.network[year][day].get_node_base_kv(ref_node_id)
                for p in dso_model[year][day].periods:

                    interface_vmag = pe.value(dso_model[year][day].expected_interface_vmag[p]) * v_base
                    interface_pf_p = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    interface_pf_q = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base
                    p_ess = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base
                    q_ess = pe.value(dso_model[year][day].expected_shared_ess_q[p]) * s_base

                    consensus_vars['vmag']['dso']['current'][node_id][year][day][p] = interface_vmag
                    consensus_vars['pf']['dso']['current'][node_id][year][day]['p'][p] = interface_pf_p
                    consensus_vars['pf']['dso']['current'][node_id][year][day]['q'][p] = interface_pf_q
                    consensus_vars['ess']['dso']['current'][node_id][year][day]['p'][p] = p_ess
                    consensus_vars['ess']['dso']['current'][node_id][year][day]['q'][p] = q_ess

        dso_models[node_id] = dso_model

    return dso_models, results


def create_distribution_networks_models_parallel(distribution_networks, consensus_vars, candidate_solution):

    results = dict()
    dso_models = dict()
    for node_id in distribution_networks:
        results[node_id] = dict()
        dso_models[node_id] = dict()

    tasks = []
    max_workers = min(os.cpu_count() // 2, len(distribution_networks))  # Note: to limit memory usage
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for node_id in distribution_networks:
            tasks.append(executor.submit(create_distribution_network_model, node_id, distribution_networks[node_id], candidate_solution))

        for future in as_completed(tasks):

            node_id, result, model = future.result()
            results[node_id] = result
            dso_models[node_id] = model

            # Get initial interface and shared ESS values
            for year in distribution_networks[node_id].years:
                for day in distribution_networks[node_id].days:
                    if not _solver_result_succeeded(results[node_id][year][day]):
                        continue
                    ref_node_id = distribution_networks[node_id].network[year][day].get_reference_node_id()
                    s_base = distribution_networks[node_id].network[year][day].baseMVA
                    v_base = distribution_networks[node_id].network[year][day].get_node_base_kv(ref_node_id)
                    for p in dso_models[node_id][year][day].periods:
                        interface_vmag = pe.value(dso_models[node_id][year][day].expected_interface_vmag[p]) * v_base
                        interface_pf_p = pe.value(dso_models[node_id][year][day].expected_interface_pf_p[p]) * s_base
                        interface_pf_q = pe.value(dso_models[node_id][year][day].expected_interface_pf_q[p]) * s_base
                        p_ess = pe.value(dso_models[node_id][year][day].expected_shared_ess_p[p]) * s_base
                        q_ess = pe.value(dso_models[node_id][year][day].expected_shared_ess_q[p]) * s_base

                        consensus_vars['vmag']['dso']['current'][node_id][year][day][p] = interface_vmag
                        consensus_vars['pf']['dso']['current'][node_id][year][day]['p'][p] = interface_pf_p
                        consensus_vars['pf']['dso']['current'][node_id][year][day]['q'][p] = interface_pf_q
                        consensus_vars['ess']['dso']['current'][node_id][year][day]['p'][p] = p_ess
                        consensus_vars['ess']['dso']['current'][node_id][year][day]['q'][p] = q_ess

    return dso_models, results


def create_distribution_network_model(node_id, distribution_network, candidate_solution):

    # Build model, fix candidate solution
    distribution_network.update_data_with_candidate_solution(candidate_solution)
    dso_model = distribution_network.build_model()
    distribution_network.update_model_with_candidate_solution(dso_model, candidate_solution)

    # Update model with expected interface values
    for year in distribution_network.years:
        for day in distribution_network.days:

            ref_node_id = distribution_network.network[year][day].get_reference_node_id()
            shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)
            v_min, v_max = distribution_network.network[year][day].get_node_voltage_limits(ref_node_id)

            # Add interface expected variables, and definition
            dso_model[year][day].expected_interface_vmag = pe.Var(dso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=1.00, bounds=(v_min - EQUALITY_TOLERANCE, v_max + EQUALITY_TOLERANCE))
            dso_model[year][day].expected_interface_pf_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            dso_model[year][day].expected_interface_pf_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            dso_model[year][day].expected_shared_ess_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            dso_model[year][day].expected_shared_ess_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.00)
            dso_model[year][day].interface_expected_values_vmag = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_vmag_rule, network=distribution_network.network[year][day]))
            dso_model[year][day].interface_expected_values_pf_p = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_pf_p_rule, network=distribution_network.network[year][day]))
            dso_model[year][day].interface_expected_values_pf_q = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_pf_q_rule, network=distribution_network.network[year][day]))
            dso_model[year][day].interface_expected_values_sess_p = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_sess_p_rule, network=distribution_network.network[year][day], shared_ess_idx=shared_ess_idx))
            dso_model[year][day].interface_expected_values_sess_q = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_sess_q_rule, network= distribution_network.network[year][day], shared_ess_idx=shared_ess_idx))

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    for year in distribution_network.years:
        for day in distribution_network.days:

            s_base = distribution_network.network[year][day].baseMVA
            ref_node_id = distribution_network.network[year][day].get_reference_node_id()
            shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)

            obj = copy(dso_model[year][day].objective.expr)
            dso_model[year][day].penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION)
            for s_m in dso_model[year][day].scenarios_market:
                for s_o in dso_model[year][day].scenarios_operation:
                    for p in dso_model[year][day].periods:
                        obj += dso_model[year][day].penalty_regularization * (dso_model[year][day].vmag_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_vmag[p]) ** 2
                        obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].pg_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_pf_p[p]) ** 2
                        obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].qg_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_pf_q[p]) ** 2
                        obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].shared_es_pnet[shared_ess_idx, s_m, s_o, p] - dso_model[year][day].expected_shared_ess_p[p]) ** 2
                        obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].shared_es_qnet[shared_ess_idx, s_m, s_o, p] - dso_model[year][day].expected_shared_ess_q[p]) ** 2
            dso_model[year][day].objective.expr = obj

    # Run SMOPF
    res = distribution_network.optimize(dso_model)

    return node_id, res, dso_model


def create_shared_energy_storage_model(shared_ess_data, consensus_vars, candidate_solution):

    years = list(shared_ess_data.years)
    days = list(shared_ess_data.days)

    # Build model, fix candidate solution
    shared_ess_data.update_data_with_candidate_solution(candidate_solution)
    esso_model = shared_ess_data.build_subproblem()
    shared_ess_data.update_model_with_candidate_solution(esso_model, candidate_solution)

    # Fix TSO's request
    for node_id in shared_ess_data.active_distribution_network_nodes:
        for y in esso_model[node_id].years:
            year = years[y]
            for d in esso_model[node_id].days:
                day = days[d]
                for p in esso_model[node_id].periods:
                    p_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['p'][p]
                    q_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['q'][p]
                    fix_or_set(esso_model[node_id].es_pnet[y, d, p], p_req)
                    fix_or_set(esso_model[node_id].es_qnet[y, d, p], q_req)

    # Run optimization
    results = shared_ess_data.optimize(esso_model)

    # Get initial shared ESS values
    for node_id in shared_ess_data.active_distribution_network_nodes:
        if not _solver_result_succeeded(results[node_id]):
            continue
        for y in esso_model[node_id].years:
            year = years[y]
            for d in esso_model[node_id].days:
                day = days[d]
                for p in esso_model[node_id].periods:
                    shared_ess_p = pe.value(esso_model[node_id].es_pnet[y, d, p])
                    shared_ess_q = pe.value(esso_model[node_id].es_qnet[y, d, p])
                    consensus_vars['ess']['esso']['current'][node_id][year][day]['p'][p] = shared_ess_p
                    consensus_vars['ess']['esso']['current'][node_id][year][day]['q'][p] = shared_ess_q
                    consensus_vars['ess']['esso']['prev'][node_id][year][day]['p'][p] = shared_ess_p
                    consensus_vars['ess']['esso']['prev'][node_id][year][day]['q'][p] = shared_ess_q

    return esso_model, results


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
#  OPERATIONAL PLANNING (HIERARCHICAL)
# ======================================================================================================================
def _run_operational_planning_hierarchical(planning_problem, num_steps=8, print_pq_map=False, debug_flag=False):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    results = {'tso': dict(), 'dso': dict()}

    start = time.time()

    # Get initial DN solutions
    dso_models = dict()
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        dso_models[node_id] = distribution_network.get_pq_map(num_steps=num_steps, print_pq_map=print_pq_map)

    pf_requested = dict()
    tso_model = transmission_network.build_model()
    for year in transmission_network.years:
        pf_requested[year] = dict()
        for day in transmission_network.days:

            pf_requested[year][day] = dict()
            for adn_node_id in transmission_network.active_distribution_network_nodes:
                pf_requested[year][day][adn_node_id] = dict()
                for p in tso_model[year][day].periods:
                    pf_requested[year][day][adn_node_id][p] = dict()

            s_base = transmission_network.network[year][day].baseMVA
            tso_model[year][day].active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))

            # TN, Fix Pc, Qc at the interface nodes, free flexibility
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            init_solution = dso_models[adn_node_id][year][day][p]['initial_solution']
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setub(init_solution['Pg'] / s_base + EQUALITY_TOLERANCE)
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setlb(init_solution['Pg'] / s_base - EQUALITY_TOLERANCE)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setub(init_solution['Qg'] / s_base + EQUALITY_TOLERANCE)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setlb(init_solution['Qg'] / s_base - EQUALITY_TOLERANCE)
                            tso_model[year][day].flex_p_up[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].flex_p_down[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].flex_q_up[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].flex_q_down[adn_load_idx, s_m, s_o, p].setub(None)
                            if transmission_network.params.l_curt:
                                tso_model[year][day].pc_curt_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].pc_curt_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].qc_curt_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].qc_curt_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)

            # TN, Add expected interface values
            tso_model[year][day].expected_interface_vmag = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=1.0)
            tso_model[year][day].expected_interface_pf_p = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
            tso_model[year][day].expected_interface_pf_q = pe.Var(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
            tso_model[year][day].expected_interface_vmag_def = pe.Constraint( tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_vmag_rule, network=transmission_network.network[year][day]))
            tso_model[year][day].expected_interface_pf_p_def = pe.Constraint(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_pf_p_rule, network=transmission_network.network[year][day]))
            tso_model[year][day].expected_interface_pf_q_def = pe.Constraint(tso_model[year][day].active_distribution_networks, tso_model[year][day].periods, rule=partial(tn_interface_expected_pf_q_rule, network=transmission_network.network[year][day]))

            # TN, Add ADNs' PQ maps constraints, Fix expected vmag
            tso_model[year][day].pq_maps = pe.ConstraintList()
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                for p in tso_model[year][day].periods:
                    adn_pq_map = dso_models[adn_node_id][year][day][p]
                    # initial_solution = adn_pq_map['initial_solution']
                    # tso_model[year][day].pq_maps.add(tso_model[year][day].expected_interface_vmag[dn, p] <= initial_solution['Vg'] + EQUALITY_TOLERANCE)
                    # tso_model[year][day].pq_maps.add(tso_model[year][day].expected_interface_vmag[dn, p] >= initial_solution['Vg'] - EQUALITY_TOLERANCE)
                    for ineq in adn_pq_map['inequalities']:
                        a = ineq['Pg']
                        b = ineq['Qg']
                        c = ineq['c'] / s_base
                        tso_model[year][day].pq_maps.add(a * tso_model[year][day].expected_interface_pf_p[dn, p] + b * tso_model[year][day].expected_interface_pf_q[dn, p] <= c)

            # Regularization -- Added to OF to minimize deviations from scenarios to expected values
            obj = copy(tso_model[year][day].objective.expr)
            tso_model[year][day].penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION)
            for dn in tso_model[year][day].active_distribution_networks:
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            # obj += tso_model[year][day].penalty_regularization * (tso_model[year][day].vmag_adn[dn, s_m, s_o, p] - tso_model[year][day].expected_interface_vmag[dn, p]) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].pc_adn[dn, s_m, s_o, p] - tso_model[year][day].expected_interface_pf_p[dn, p]) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].qc_adn[dn, s_m, s_o, p] - tso_model[year][day].expected_interface_pf_q[dn, p]) ** 2
            tso_model[year][day].objective.expr = obj

    # Optimize TN, Get resulting interface PFs
    print(f'[INFO] - Running OPF on {transmission_network.name} with hierarchical constraints...')
    results['tso'] = transmission_network.optimize(tso_model)
    for year in transmission_network.years:
        for day in transmission_network.days:
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                for p in tso_model[year][day].periods:
                    vmag = pe.value(tso_model[year][day].expected_interface_vmag[dn, p])
                    pc = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    qc = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base
                    pf_requested[year][day][adn_node_id][p] = {'Pg': pc, 'Qg': qc, 'Vg': vmag}
                    # if print_pq_map:
                    #     init_oper_point = dso_models[adn_node_id][year][day][p]
                    #     final_oper_point = pf_requested[year][day][adn_node_id]
                    #     distribution_network = operational_planning.distribution_networks[adn_node_id]
                    #     distribution_network.pq_map_comparison(t=t, num_steps_max=num_steps, initial_solution=init_oper_point, final_solution=final_oper_point)

    # Run OPF on DNs, considering established power flow (settlement)
    dso_models = dict()
    print('[INFO] - Running settlement on Distribution Networks...')
    for node_id in distribution_networks:

        distribution_network = distribution_networks[node_id]
        print(f'[INFO]\t - Network {distribution_network.name}...')

        dso_model = distribution_network.build_model()
        distribution_network.update_of_to_settlement(dso_model)

        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in dso_model[year][day].periods:
                    dso_model[year][day].interface_vmag_req[p].fix(pf_requested[year][day][node_id][p]['Vg'])
                    dso_model[year][day].interface_pf_p_req[p].fix(pf_requested[year][day][node_id][p]['Pg'] / distribution_network.network[year][day].baseMVA)
                    dso_model[year][day].interface_pf_q_req[p].fix(pf_requested[year][day][node_id][p]['Qg'] / distribution_network.network[year][day].baseMVA)

        results['dso'][node_id] = distribution_network.optimize(dso_model)
        dso_models[node_id] = dso_model

    end = time.time()
    total_execution_time = end - start
    print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))

    optim_models = {'tso': tso_model, 'dso': dso_models}

    return results, optim_models, total_execution_time


# ======================================================================================================================
#  OPERATIONAL PLANNING (CENTRALIZED)
# ======================================================================================================================
def _run_operational_planning_centralized(planning_problem, debug_flag=False):

    # Combined networks
    centralized_network = planning_problem.combine_networks()
    for year in centralized_network.years:
        for day in centralized_network.days:
            if centralized_network.params.print_to_screen:
                centralized_network.network[year][day].print_network_to_screen()
            if centralized_network.params.plot_diagram:
                centralized_network.network[year][day].plot_diagram()

    # Run SMOPF
    centralized_model = centralized_network.build_model()

    print(f'[INFO] - Running SMOPF, Network {centralized_network.name}...')
    results = centralized_network.optimize(centralized_model, print_header=False)

    return centralized_network, results, centralized_model


# ======================================================================================================================
#  ADMM functions
# ======================================================================================================================
def create_admm_variables(planning_problem):

    num_instants = planning_problem.num_instants

    consensus_variables = {
        'vmag': {'tso': {'current': dict(), 'prev': dict()},
                  'dso': {'current': dict(), 'prev': dict()}},
        'pf': {'tso': {'current': dict(), 'prev': dict()},
               'dso': {'current': dict(), 'prev': dict()}},
        'ess': {'tso': {'current': dict(), 'prev': dict()},
                'dso': {'current': dict(), 'prev': dict()},
                'esso': {'current': dict(), 'prev': dict()}}
    }

    dual_variables = {
        'vmag': {'tso': {'current': dict()}, 'dso': {'current': dict()}},
        'pf': {'tso': {'current': dict()}, 'dso': {'current': dict()}},
        'ess': {'tso': {'current': dict()}, 'dso': {'current': dict()}, 'esso': {'current': dict()}}
    }

    if planning_problem.params.admm.previous_iter['ess']:
        dual_variables['ess']['tso']['prev'] = dict()
        dual_variables['ess']['dso']['prev'] = dict()

    for dn in range(len(planning_problem.active_distribution_network_nodes)):

        node_id = planning_problem.active_distribution_network_nodes[dn]

        consensus_variables['vmag']['tso']['current'][node_id] = dict()
        consensus_variables['vmag']['dso']['current'][node_id] = dict()
        consensus_variables['pf']['tso']['current'][node_id] = dict()
        consensus_variables['pf']['dso']['current'][node_id] = dict()
        consensus_variables['ess']['tso']['current'][node_id] = dict()
        consensus_variables['ess']['dso']['current'][node_id] = dict()
        consensus_variables['ess']['esso']['current'][node_id] = dict()

        consensus_variables['vmag']['tso']['prev'][node_id] = dict()
        consensus_variables['vmag']['dso']['prev'][node_id] = dict()
        consensus_variables['pf']['tso']['prev'][node_id] = dict()
        consensus_variables['pf']['dso']['prev'][node_id] = dict()
        consensus_variables['ess']['tso']['prev'][node_id] = dict()
        consensus_variables['ess']['dso']['prev'][node_id] = dict()
        consensus_variables['ess']['esso']['prev'][node_id] = dict()

        dual_variables['vmag']['tso']['current'][node_id] = dict()
        dual_variables['vmag']['dso']['current'][node_id] = dict()
        dual_variables['pf']['tso']['current'][node_id] = dict()
        dual_variables['pf']['dso']['current'][node_id] = dict()
        dual_variables['ess']['tso']['current'][node_id] = dict()
        dual_variables['ess']['dso']['current'][node_id] = dict()
        dual_variables['ess']['esso']['current'][node_id] = dict()

        if planning_problem.params.admm.previous_iter['ess']:
            dual_variables['ess']['tso']['prev'][node_id] = dict()
            dual_variables['ess']['dso']['prev'][node_id] = dict()

        for year in planning_problem.years:

            consensus_variables['vmag']['tso']['current'][node_id][year] = dict()
            consensus_variables['vmag']['dso']['current'][node_id][year] = dict()
            consensus_variables['pf']['tso']['current'][node_id][year] = dict()
            consensus_variables['pf']['dso']['current'][node_id][year] = dict()
            consensus_variables['ess']['tso']['current'][node_id][year] = dict()
            consensus_variables['ess']['dso']['current'][node_id][year] = dict()
            consensus_variables['ess']['esso']['current'][node_id][year] = dict()

            consensus_variables['vmag']['tso']['prev'][node_id][year] = dict()
            consensus_variables['vmag']['dso']['prev'][node_id][year] = dict()
            consensus_variables['pf']['tso']['prev'][node_id][year] = dict()
            consensus_variables['pf']['dso']['prev'][node_id][year] = dict()
            consensus_variables['ess']['tso']['prev'][node_id][year] = dict()
            consensus_variables['ess']['dso']['prev'][node_id][year] = dict()
            consensus_variables['ess']['esso']['prev'][node_id][year] = dict()

            dual_variables['vmag']['tso']['current'][node_id][year] = dict()
            dual_variables['vmag']['dso']['current'][node_id][year] = dict()
            dual_variables['pf']['tso']['current'][node_id][year] = dict()
            dual_variables['pf']['dso']['current'][node_id][year] = dict()
            dual_variables['ess']['tso']['current'][node_id][year] = dict()
            dual_variables['ess']['dso']['current'][node_id][year] = dict()
            dual_variables['ess']['esso']['current'][node_id][year] = dict()

            if planning_problem.params.admm.previous_iter['ess']:
                dual_variables['ess']['tso']['prev'][node_id][year] = dict()
                dual_variables['ess']['dso']['prev'][node_id][year] = dict()

            for day in planning_problem.days:

                node_base_kv = planning_problem.transmission_network.network[year][day].get_node_base_kv(node_id)

                consensus_variables['vmag']['tso']['current'][node_id][year][day] = [node_base_kv] * num_instants
                consensus_variables['vmag']['dso']['current'][node_id][year][day] = [node_base_kv] * num_instants
                consensus_variables['pf']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['pf']['dso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['dso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['esso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

                consensus_variables['vmag']['tso']['prev'][node_id][year][day] = [node_base_kv] * num_instants
                consensus_variables['vmag']['dso']['prev'][node_id][year][day] = [node_base_kv] * num_instants
                consensus_variables['pf']['tso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['pf']['dso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['tso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['dso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['esso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

                dual_variables['vmag']['tso']['current'][node_id][year][day] = [0.0] * planning_problem.num_instants
                dual_variables['vmag']['dso']['current'][node_id][year][day] = [0.0] * planning_problem.num_instants
                dual_variables['pf']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                dual_variables['pf']['dso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['tso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['dso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['esso']['current'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

                if planning_problem.params.admm.previous_iter['ess']:
                    dual_variables['ess']['tso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                    dual_variables['ess']['dso']['prev'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

    return consensus_variables, dual_variables


def update_transmission_model_to_admm(planning_problem, model, params):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA

            # Update costs (penalties) for the coordination procedure
            model[year][day].penalty_ess_usage.set_value(0.00)
            model[year][day].penalty_gen_curtailment.set_value(0.00)
            if transmission_network.params.obj_type == OBJ_MIN_COST:
                model[year][day].cost_load_curtailment.set_value(COST_CONSUMPTION_CURTAILMENT)
            elif transmission_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
                model[year][day].penalty_load_curtailment.set_value(PENALTY_LOAD_CURTAILMENT)
                model[year][day].penalty_flex_usage.set_value(0.00)

            # Add ADMM variables
            model[year][day].rho_v = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['v'][transmission_network.name])
            model[year][day].vmag_req = pe.Param(model[year][day].active_distribution_networks, model[year][day].periods, mutable=True, domain=pe.NonNegativeReals)    # Square of voltage magnitude
            model[year][day].dual_vmag_req = pe.Param(model[year][day].active_distribution_networks, model[year][day].periods, mutable=True, domain=pe.Reals)          # Dual variable - voltage magnitude requested

            model[year][day].rho_pf = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['pf'][transmission_network.name])
            model[year][day].p_pf_req = pe.Param(model[year][day].active_distribution_networks, model[year][day].periods, mutable=True, domain=pe.Reals)                # Active power - requested by distribution networks
            model[year][day].q_pf_req = pe.Param(model[year][day].active_distribution_networks, model[year][day].periods, mutable=True, domain=pe.Reals)                # Reactive power - requested by distribution networks
            model[year][day].dual_pf_p_req = pe.Param(model[year][day].active_distribution_networks, model[year][day].periods, mutable=True, domain=pe.Reals)           # Dual variable - active power requested
            model[year][day].dual_pf_q_req = pe.Param(model[year][day].active_distribution_networks, model[year][day].periods, mutable=True, domain=pe.Reals)           # Dual variable - reactive power requested

            model[year][day].rho_ess = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['ess'][transmission_network.name])
            model[year][day].p_ess_req = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)                     # SharedESS - Active power requested (DSO)
            model[year][day].q_ess_req = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)                     # SharedESS - Reactive power requested (DSO)
            model[year][day].dual_ess_p_req = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)                # Dual variable - SharedESS active power
            model[year][day].dual_ess_q_req = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)                # Dual variable - SharedESS reactive power
            if params.previous_iter['ess']['tso']:
                model[year][day].rho_ess_prev = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho_previous_iter['ess'][transmission_network.name])
                model[year][day].p_ess_prev = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)                # SharedESS - previous iteration active power
                model[year][day].q_ess_prev = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)                # SharedESS - previous iteration reactive power
                model[year][day].dual_ess_p_prev = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)           # Dual variable - previous iteration shared ESS active power
                model[year][day].dual_ess_q_prev = pe.Param(model[year][day].shared_energy_storages, model[year][day].periods, mutable=True, domain=pe.Reals)           # Dual variable - previous iteration shared ESS reactive power

            # Objective function - augmented Lagrangian
            init_of_value = 1.00
            if transmission_network.params.obj_type == OBJ_MIN_COST:
                init_of_value = abs(pe.value(model[year][day].objective))
            if isclose(init_of_value, 0.00, abs_tol=SMALL_TOLERANCE):
                init_of_value = 0.01
            model[year][day].admm_objective_scale = pe.Param(initialize=init_of_value)
            obj = copy(model[year][day].objective.expr) / init_of_value

            for dn in model[year][day].active_distribution_networks:

                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                distribution_network = distribution_networks[adn_node_id]
                interface_transf_rating = distribution_network.network[year][day].get_interface_branch_rating() / s_base

                for p in model[year][day].periods:

                    constraint_v_req = (model[year][day].expected_interface_vmag[dn, p] - model[year][day].vmag_req[dn, p])
                    obj += model[year][day].dual_vmag_req[dn, p] * constraint_v_req
                    obj += (model[year][day].rho_v / 2) * (constraint_v_req ** 2)

                    constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p]) / interface_transf_rating
                    constraint_q_req = (model[year][day].expected_interface_pf_q[dn, p] - model[year][day].q_pf_req[dn, p]) / interface_transf_rating
                    obj += model[year][day].dual_pf_p_req[dn, p] * constraint_p_req
                    obj += model[year][day].dual_pf_q_req[dn, p] * constraint_q_req
                    obj += (model[year][day].rho_pf / 2) * (constraint_p_req ** 2)
                    obj += (model[year][day].rho_pf / 2) * (constraint_q_req ** 2)

            for e in model[year][day].shared_energy_storages:

                shared_ess_rating = abs(transmission_network.network[year][day].shared_energy_storages[e].s)
                if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                    shared_ess_rating = 0.01

                for p in model[year][day].periods:
                    constraint_ess_p_req = (model[year][day].expected_shared_ess_p[e, p] - model[year][day].p_ess_req[e, p]) / (2 * shared_ess_rating)
                    constraint_ess_q_req = (model[year][day].expected_shared_ess_q[e, p] - model[year][day].q_ess_req[e, p]) / (2 * shared_ess_rating)
                    obj += (model[year][day].dual_ess_p_req[e, p]) * constraint_ess_p_req
                    obj += (model[year][day].dual_ess_q_req[e, p]) * constraint_ess_q_req
                    obj += (model[year][day].rho_ess / 2) * constraint_ess_p_req ** 2
                    obj += (model[year][day].rho_ess / 2) * constraint_ess_q_req ** 2
                    if params.previous_iter['ess']['tso']:
                        constraint_ess_p_prev = (model[year][day].expected_shared_ess_p[e, p] - model[year][day].p_ess_prev[e, p]) / (2 * shared_ess_rating)
                        constraint_ess_q_prev = (model[year][day].expected_shared_ess_q[e, p] - model[year][day].q_ess_prev[e, p]) / (2 * shared_ess_rating)
                        obj += (model[year][day].rho_ess_prev / 2) * constraint_ess_p_prev ** 2
                        obj += (model[year][day].rho_ess_prev / 2) * constraint_ess_q_prev ** 2

            # Add ADMM OF, deactivate original OF
            model[year][day].objective.deactivate()
            model[year][day].admm_objective = pe.Objective(sense=pe.minimize, expr=obj)


def update_distribution_models_to_admm(planning_problem, models, params):

    distribution_networks = planning_problem.distribution_networks

    for node_id in distribution_networks:

        dso_model = models[node_id]
        distribution_network = distribution_networks[node_id]

        for year in distribution_network.years:
            for day in distribution_network.days:

                s_base = distribution_network.network[year][day].baseMVA
                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                ref_node_idx = distribution_network.network[year][day].get_node_idx(ref_node_id)
                v_min, v_max = distribution_network.network[year][day].get_node_voltage_limits(ref_node_id)

                # Update Vmag, Pg, Qg limits at the interface node
                for s_m in dso_model[year][day].scenarios_market:
                    for s_o in dso_model[year][day].scenarios_operation:
                        for p in dso_model[year][day].periods:
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].fixed = False
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].setub(v_max + EQUALITY_TOLERANCE)
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].setlb(-v_max - EQUALITY_TOLERANCE)
                            dso_model[year][day].f[ref_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                            dso_model[year][day].f[ref_node_idx, s_m, s_o, p].setlb(-EQUALITY_TOLERANCE)
                            dso_model[year][day].vmag[ref_node_idx, s_m, s_o, p].setub(v_max + EQUALITY_TOLERANCE)
                            dso_model[year][day].vmag[ref_node_idx, s_m, s_o, p].setlb(v_min - EQUALITY_TOLERANCE)
                            if distribution_network.params.slacks.grid_operation.voltage:
                                dso_model[year][day].slack_e_up[ref_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                dso_model[year][day].slack_e_down[ref_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                dso_model[year][day].slack_f_up[ref_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                dso_model[year][day].slack_f_down[ref_node_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)

                # Update costs (penalties) for the coordination procedure
                dso_model[year][day].penalty_ess_usage.set_value(0.00)
                # dso_model[year][day].penalty_gen_curtailment.set_value(0.00)
                if distribution_network.params.obj_type == OBJ_MIN_COST:
                    dso_model[year][day].cost_load_curtailment.set_value(COST_CONSUMPTION_CURTAILMENT)
                elif distribution_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
                    dso_model[year][day].penalty_load_curtailment.set_value(PENALTY_LOAD_CURTAILMENT)
                    dso_model[year][day].penalty_flex_usage.set_value(0.00)

                # Add ADMM variables
                dso_model[year][day].rho_v = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['v'][distribution_network.network[year][day].name])
                dso_model[year][day].vmag_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.NonNegativeReals)       # Voltage magnitude - requested by TSO
                dso_model[year][day].dual_vmag_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)             # Dual variable - voltage magnitude

                dso_model[year][day].rho_pf = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['pf'][distribution_network.network[year][day].name])
                dso_model[year][day].p_pf_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)                   # Active power - requested by TSO
                dso_model[year][day].q_pf_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)                   # Reactive power - requested by TSO
                dso_model[year][day].dual_pf_p_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)              # Dual variable - active power
                dso_model[year][day].dual_pf_q_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)              # Dual variable - reactive power

                dso_model[year][day].rho_ess = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['ess'][distribution_network.network[year][day].name])
                dso_model[year][day].p_ess_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)                  # SharedESS - active power requested (TSO)
                dso_model[year][day].q_ess_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)                  # SharedESS - reactive power requested (TSO)
                dso_model[year][day].dual_ess_p_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)             # Dual variable - SharedESS active power
                dso_model[year][day].dual_ess_q_req = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)             # Dual variable - SharedESS reactive power
                if params.previous_iter['ess']['dso']:
                    dso_model[year][day].rho_ess_prev = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho_previous_iter['ess'][distribution_network.network[year][day].name])
                    dso_model[year][day].p_ess_prev = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)             # SharedESS - previous iteration active power
                    dso_model[year][day].q_ess_prev = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)             # SharedESS - previous iteration reactive power
                    dso_model[year][day].dual_ess_p_prev = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)        # Dual variable - SharedESS previous iteration active power
                    dso_model[year][day].dual_ess_q_prev = pe.Param(dso_model[year][day].periods, mutable=True, domain=pe.Reals)        # Dual variable - SharedESS previous iteration reactive power

                # Objective function - augmented Lagrangian
                init_of_value = 1.00
                if distribution_network.params.obj_type == OBJ_MIN_COST:
                    init_of_value = abs(pe.value(dso_model[year][day].objective))
                if isclose(init_of_value, 0.00, abs_tol=SMALL_TOLERANCE):
                    init_of_value = 0.01
                dso_model[year][day].admm_objective_scale = pe.Param(initialize=init_of_value)
                obj = copy(dso_model[year][day].objective.expr) / init_of_value

                shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)
                shared_ess_rating = abs(distribution_network.network[year][day].shared_energy_storages[shared_ess_idx].s)
                if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                    shared_ess_rating = 0.01

                interface_transf_rating = distribution_network.network[year][day].get_interface_branch_rating() / s_base

                # Augmented Lagrangian -- Interface power flow (residual balancing)
                for p in dso_model[year][day].periods:

                    # Voltage magnitude
                    constraint_vmag_req = (dso_model[year][day].expected_interface_vmag[p] - dso_model[year][day].vmag_req[p])
                    obj += (dso_model[year][day].dual_vmag_req[p]) * constraint_vmag_req
                    obj += (dso_model[year][day].rho_v / 2) * (constraint_vmag_req ** 2)

                    # Interface power flow
                    constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p]) / interface_transf_rating
                    constraint_q_req = (dso_model[year][day].expected_interface_pf_q[p] - dso_model[year][day].q_pf_req[p]) / interface_transf_rating
                    obj += (dso_model[year][day].dual_pf_p_req[p]) * constraint_p_req
                    obj += (dso_model[year][day].dual_pf_q_req[p]) * constraint_q_req
                    obj += (dso_model[year][day].rho_pf / 2) * (constraint_p_req ** 2)
                    obj += (dso_model[year][day].rho_pf / 2) * (constraint_q_req ** 2)

                    # SharedESS
                    constraint_ess_p_req = (dso_model[year][day].expected_shared_ess_p[p] - dso_model[year][day].p_ess_req[p]) / (2 * shared_ess_rating)
                    constraint_ess_q_req = (dso_model[year][day].expected_shared_ess_q[p] - dso_model[year][day].q_ess_req[p]) / (2 * shared_ess_rating)
                    obj += (dso_model[year][day].dual_ess_p_req[p]) * constraint_ess_p_req
                    obj += (dso_model[year][day].dual_ess_q_req[p]) * constraint_ess_q_req
                    obj += (dso_model[year][day].rho_ess / 2) * constraint_ess_p_req ** 2
                    obj += (dso_model[year][day].rho_ess / 2) * constraint_ess_q_req ** 2
                    if params.previous_iter['ess']['dso']:
                        constraint_ess_p_prev = (dso_model[year][day].expected_shared_ess_p[p] - dso_model[year][day].p_ess_prev[p]) / (2 * shared_ess_rating)
                        constraint_ess_q_prev = (dso_model[year][day].expected_shared_ess_q[p] - dso_model[year][day].q_ess_prev[p]) / (2 * shared_ess_rating)
                        obj += (dso_model[year][day].dual_ess_p_prev[p]) * constraint_ess_p_prev
                        obj += (dso_model[year][day].dual_ess_q_prev[p]) * constraint_ess_q_prev
                        obj += (dso_model[year][day].rho_ess_prev / 2) * constraint_ess_p_prev ** 2
                        obj += (dso_model[year][day].rho_ess_prev / 2) * constraint_ess_q_prev ** 2

                # Add ADMM OF, deactivate original OF
                dso_model[year][day].objective.deactivate()
                dso_model[year][day].admm_objective = pe.Objective(sense=pe.minimize, expr=obj)


def update_shared_energy_storage_model_to_admm(planning_problem, models, params):

    shared_ess_data = planning_problem.shared_ess_data
    years = list(shared_ess_data.years)

    for node_id in shared_ess_data.active_distribution_network_nodes:

        shared_ess_idx = shared_ess_data.get_shared_energy_storage_idx(node_id)

        # Add ADMM variables
        models[node_id].rho = pe.Param(mutable=True, domain=pe.NonNegativeReals, initialize=params.rho['ess']['esso'])

        # Free Pnet, Qnet
        for y in models[node_id].years:
            for d in models[node_id].days:
                for p in models[node_id].periods:
                    models[node_id].es_pnet[y, d, p].fixed = False
                    models[node_id].es_qnet[y, d, p].fixed = False

        # Active and Reactive power requested by TSO and DSOs
        models[node_id].p_req = pe.Param(models[node_id].years, models[node_id].days, models[node_id].periods, mutable=True, domain=pe.Reals)
        models[node_id].q_req = pe.Param(models[node_id].years, models[node_id].days, models[node_id].periods, mutable=True, domain=pe.Reals)
        models[node_id].dual_p_req = pe.Param(models[node_id].years, models[node_id].days, models[node_id].periods, mutable=True, domain=pe.Reals)
        models[node_id].dual_q_req = pe.Param(models[node_id].years, models[node_id].days, models[node_id].periods, mutable=True, domain=pe.Reals)

        # Objective function - augmented Lagrangian
        obj = copy(models[node_id].objective.expr)
        for y in models[node_id].years:
            year = years[y]
            shared_ess_rating = shared_ess_data.shared_energy_storages[year][shared_ess_idx].s
            if isclose(shared_ess_rating, 0.00, abs_tol=SMALL_TOLERANCE):
                shared_ess_rating = 1.00
            for d in models[node_id].days:
                for p in models[node_id].periods:
                    constraint_p_req = (models[node_id].es_pnet[y, d, p] - models[node_id].p_req[y, d, p]) / (2 * shared_ess_rating)
                    constraint_q_req = (models[node_id].es_qnet[y, d, p] - models[node_id].q_req[y, d, p]) / (2 * shared_ess_rating)
                    obj += models[node_id].dual_p_req[y, d, p] * constraint_p_req
                    obj += models[node_id].dual_q_req[y, d, p] * constraint_q_req
                    obj += (models[node_id].rho / 2) * constraint_p_req ** 2
                    obj += (models[node_id].rho / 2) * constraint_q_req ** 2

        # Add ADMM OF, deactivate original OF
        models[node_id].admm_objective = pe.Objective(sense=pe.minimize, expr=obj)
        models[node_id].objective.deactivate()

    return models


def update_transmission_coordination_model_and_solve(transmission_network, model, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacities, from_warm_start=False):

    print('[INFO] \t\t - Updating transmission network...')

    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA

            for dn in model[year][day].active_distribution_networks:

                node_id = transmission_network.active_distribution_network_nodes[dn]
                v_base = transmission_network.network[year][day].get_node_base_kv(node_id)
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                sess_estimated_capacity = sess_estimated_capacities[node_id]

                # Update estimated rated power and energy capacity
                model[year][day].shared_es_s_rated_fixed[shared_ess_idx].set_value(max(sess_estimated_capacity[year]['s_available'], EQUALITY_TOLERANCE) / s_base)
                model[year][day].shared_es_e_rated_fixed[shared_ess_idx].set_value(max(sess_estimated_capacity[year]['e_available'], EQUALITY_TOLERANCE) / s_base)

                # Update VOLTAGE and POWER FLOW variables at connection point
                for p in model[year][day].periods:
                    model[year][day].dual_vmag_req[dn, p].set_value(dual_vmag['current'][node_id][year][day][p] / v_base)
                    model[year][day].vmag_req[dn, p].set_value(vmag_req['dso']['current'][node_id][year][day][p] / v_base)
                    model[year][day].dual_pf_p_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_pf_q_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_pf_req[dn, p].set_value(pf_req['dso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_pf_req[dn, p].set_value(pf_req['dso']['current'][node_id][year][day]['q'][p] / s_base)

                # Update shared ESS capacity and power requests
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                for p in model[year][day].periods:
                    model[year][day].dual_ess_p_req[shared_ess_idx, p].set_value(dual_ess['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_ess_q_req[shared_ess_idx, p].set_value(dual_ess['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_ess_req[shared_ess_idx, p].set_value(ess_req['dso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_ess_req[shared_ess_idx, p].set_value(ess_req['dso']['current'][node_id][year][day]['q'][p] / s_base)
                    if params.previous_iter['ess']['tso']:
                        model[year][day].dual_ess_p_prev[shared_ess_idx, p].set_value(dual_ess['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].dual_ess_q_prev[shared_ess_idx, p].set_value(dual_ess['prev'][node_id][year][day]['q'][p] / s_base)
                        model[year][day].p_ess_prev[shared_ess_idx, p].set_value(ess_req['tso']['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].q_ess_prev[shared_ess_idx, p].set_value(ess_req['tso']['prev'][node_id][year][day]['q'][p] / s_base)

    # Solve!
    res = transmission_network.optimize(model, from_warm_start=from_warm_start)
    for year in transmission_network.years:
        for day in transmission_network.days:
            if not _solver_result_succeeded(res[year][day]):
                print(
                    f'[ERROR] Transmission network {model[year][day].name}, '
                    f'year={year}, day={day} did not converge: '
                    f'{solver_result_summary(res[year][day])}'
                )
                # exit(ERROR_NETWORK_OPTIMIZATION)
    return res


def update_distribution_coordination_models_and_solve(distribution_networks, models, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacities, from_warm_start=False, parallel_execution=False):
    if parallel_execution:
        return update_distribution_coordination_models_and_solve_parallel(distribution_networks, models, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacities, from_warm_start=from_warm_start)
    else:
        return update_distribution_coordination_models_and_solve_sequential(distribution_networks, models, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacities, from_warm_start=from_warm_start)


def update_distribution_coordination_models_and_solve_sequential(distribution_networks, models, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacities, from_warm_start=False):

    print('[INFO] \t\t - Updating distribution networks:')
    res = dict()

    for node_id in distribution_networks:

        model = models[node_id]
        distribution_network = distribution_networks[node_id]
        sess_estimated_capacity = sess_estimated_capacities[node_id]

        for year in distribution_network.years:
            for day in distribution_network.days:

                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                v_base = distribution_network.network[year][day].get_node_base_kv(ref_node_id)
                s_base = distribution_network.network[year][day].baseMVA
                shared_ess_idx = distribution_network.network[year][day].get_shared_energy_storage_idx(ref_node_id)

                # Update estimated rated power and energy capacity
                model[year][day].shared_es_s_rated_fixed[shared_ess_idx].set_value(max(sess_estimated_capacity[year]['s_available'], EQUALITY_TOLERANCE) / s_base)
                model[year][day].shared_es_e_rated_fixed[shared_ess_idx].set_value(max(sess_estimated_capacity[year]['e_available'], EQUALITY_TOLERANCE) / s_base)

                # Update VOLTAGE and POWER FLOW variables at connection point
                for p in model[year][day].periods:
                    model[year][day].dual_vmag_req[p].set_value(dual_vmag['current'][node_id][year][day][p] / v_base)
                    model[year][day].vmag_req[p].set_value(vmag_req['tso']['current'][node_id][year][day][p] / v_base)
                    model[year][day].dual_pf_p_req[p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_pf_q_req[p].set_value(dual_pf['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_pf_req[p].set_value(pf_req['tso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_pf_req[p].set_value(pf_req['tso']['current'][node_id][year][day]['q'][p] / s_base)

                # Update SHARED ENERGY STORAGE variables (if existent)
                for p in model[year][day].periods:
                    model[year][day].dual_ess_p_req[p].set_value(dual_ess['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_ess_q_req[p].set_value(dual_ess['current'][node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_ess_req[p].set_value(ess_req['esso']['current'][node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_ess_req[p].set_value(ess_req['esso']['current'][node_id][year][day]['q'][p] / s_base)
                    if params.previous_iter['ess']['dso']:
                        model[year][day].dual_ess_p_prev[p].set_value(dual_ess['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].dual_ess_q_prev[p].set_value(dual_ess['prev'][node_id][year][day]['q'][p] / s_base)
                        model[year][day].p_ess_prev[p].set_value(ess_req['dso']['prev'][node_id][year][day]['p'][p] / s_base)
                        model[year][day].q_ess_prev[p].set_value(ess_req['dso']['prev'][node_id][year][day]['q'][p] / s_base)

        # Solve!
        res[node_id] = distribution_network.optimize(model, from_warm_start=from_warm_start)
        for year in distribution_network.years:
            for day in distribution_network.days:
                if not _solver_result_succeeded(res[node_id][year][day]):
                    print(
                        f'[WARNING] Distribution network node={node_id}, '
                        f'network={model[year][day].name}, year={year}, day={day} '
                        f'did not converge: {solver_result_summary(res[node_id][year][day])}'
                    )
                    #exit(ERROR_NETWORK_OPTIMIZATION)
    return res


def update_distribution_coordination_models_and_solve_parallel(distribution_networks, models, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacities, from_warm_start=False):

    print('[INFO] \t\t - Updating distribution networks in parallel:')
    res = dict()

    tasks = []
    max_workers = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for node_id in distribution_networks:
            sess_estimated_capacity = sess_estimated_capacities[node_id]
            tasks.append(executor.submit(update_and_solve_dso, node_id, distribution_networks[node_id], models[node_id],
                                         vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess,
                                         params, sess_estimated_capacity,
                                         from_warm_start=from_warm_start))

        for future in as_completed(tasks):
            node_id, result, updated_model = future.result()
            res[node_id] = result
            models[node_id] = updated_model

    return res


def update_and_solve_dso(node_id, distribution_network, model, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, params, sess_estimated_capacity, from_warm_start=False):

    for year in distribution_network.years:
        for day in distribution_network.days:

            ref_node_id = distribution_network.network[year][day].get_reference_node_id()
            v_base = distribution_network.network[year][day].get_node_base_kv(ref_node_id)
            s_base = distribution_network.network[year][day].baseMVA

            # Update estimated rated power and energy capacity
            model[year][day].shared_es_s_rated_fixed.set_value(max(sess_estimated_capacity[year]['s_available'], EQUALITY_TOLERANCE) / s_base)
            model[year][day].shared_es_e_rated_fixed.set_value(max(sess_estimated_capacity[year]['e_available'], EQUALITY_TOLERANCE) / s_base)

            # Update VOLTAGE and POWER FLOW variables at connection point
            for p in model[year][day].periods:
                fix_or_set(model[year][day].dual_vmag_req[p], dual_vmag['current'][node_id][year][day][p] / v_base)
                fix_or_set(model[year][day].vmag_req[p], vmag_req['tso']['current'][node_id][year][day][p] / v_base)
                fix_or_set(model[year][day].dual_pf_p_req[p], dual_pf['current'][node_id][year][day]['p'][p] / s_base)
                fix_or_set(model[year][day].dual_pf_q_req[p], dual_pf['current'][node_id][year][day]['q'][p] / s_base)
                fix_or_set(model[year][day].p_pf_req[p], pf_req['tso']['current'][node_id][year][day]['p'][p] / s_base)
                fix_or_set(model[year][day].q_pf_req[p], pf_req['tso']['current'][node_id][year][day]['q'][p] / s_base)

            # Update SHARED ENERGY STORAGE variables (if existent)
            for p in model[year][day].periods:
                fix_or_set(model[year][day].dual_ess_p_req[p], dual_ess['current'][node_id][year][day]['p'][p] / s_base)
                fix_or_set(model[year][day].dual_ess_q_req[p], dual_ess['current'][node_id][year][day]['q'][p] / s_base)
                fix_or_set(model[year][day].p_ess_req[p], ess_req['esso']['current'][node_id][year][day]['p'][p] / s_base)
                fix_or_set(model[year][day].q_ess_req[p], ess_req['esso']['current'][node_id][year][day]['q'][p] / s_base)
                if params.previous_iter['ess']['dso']:
                    fix_or_set(model[year][day].dual_ess_p_prev[p], dual_ess['prev'][node_id][year][day]['p'][p] / s_base)
                    fix_or_set(model[year][day].dual_ess_q_prev[p], dual_ess['prev'][node_id][year][day]['q'][p] / s_base)
                    fix_or_set(model[year][day].p_ess_prev[p], ess_req['dso']['prev'][node_id][year][day]['p'][p] / s_base)
                    fix_or_set(model[year][day].q_ess_prev[p], ess_req['dso']['prev'][node_id][year][day]['q'][p] / s_base)

    # Solve
    res = distribution_network.optimize(model, from_warm_start=from_warm_start)
    for year in distribution_network.years:
        for day in distribution_network.days:
            if not _solver_result_succeeded(res[year][day]):
                print(
                    f'[WARNING] Distribution network node={node_id}, '
                    f'network={model[year][day].name}, year={year}, day={day} '
                    f'did not converge: {solver_result_summary(res[year][day])}'
                )

    return (node_id, res, model)


def update_shared_energy_storages_coordination_model_and_solve(planning_problem, models, ess_req, dual_ess, params, from_warm_start=False):

    print('[INFO] \t\t - Updating SharedESS...')
    shared_ess_data = planning_problem.shared_ess_data
    days = [day for day in planning_problem.days]
    years = [year for year in planning_problem.years]

    for node_id in planning_problem.active_distribution_network_nodes:

        for y in models[node_id].years:
            year = years[y]
            for d in models[node_id].days:
                day = days[d]
                for p in models[node_id].periods:

                    p_req = ess_req['current'][node_id][year][day]['p'][p]
                    q_req = ess_req['current'][node_id][year][day]['q'][p]
                    dual_p_req = dual_ess['current'][node_id][year][day]['p'][p]
                    dual_q_req = dual_ess['current'][node_id][year][day]['q'][p]

                    models[node_id].p_req[y, d, p].set_value(p_req)
                    models[node_id].q_req[y, d, p].set_value(q_req)
                    models[node_id].dual_p_req[y, d, p].set_value(dual_p_req)
                    models[node_id].dual_q_req[y, d, p].set_value(dual_q_req)

    # Solve!
    res = shared_ess_data.optimize(models, from_warm_start=from_warm_start)
    for node_id in planning_problem.active_distribution_network_nodes:
        if not _solver_result_succeeded(res[node_id]):
            print(
                f'[WARNING] SharedESS operational planning node={node_id} did not converge: '
                f'{solver_result_summary(res[node_id])}'
            )

    return res


def get_admm_residual_metrics(planning_problem, tso_model, dso_models, esso_model, consensus_vars):
    sums = {
        'primal': {'v': 0.0, 'pf': 0.0, 'ess': 0.0},
        'dual': {'v': 0.0, 'pf': 0.0, 'ess': 0.0},
    }
    counts = {
        'primal': {'v': 0, 'pf': 0, 'ess': 0},
        'dual': {'v': 0, 'pf': 0, 'ess': 0},
    }

    for node_id in planning_problem.active_distribution_network_nodes:
        dso_model = dso_models[node_id]
        for year in planning_problem.years:
            for day in planning_problem.days:
                network = planning_problem.transmission_network.network[year][day]
                s_base = network.baseMVA
                shared_ess_idx = network.get_shared_energy_storage_idx(node_id)
                interface_v_base = network.get_node_base_kv(node_id)
                interface_rating = planning_problem.distribution_networks[node_id].network[year][day].get_interface_branch_rating()
                shared_ess_rating = max(
                    abs(network.shared_energy_storages[shared_ess_idx].s) * s_base,
                    0.10,
                )

                rho_tso_v = pe.value(tso_model[year][day].rho_v)
                rho_tso_pf = pe.value(tso_model[year][day].rho_pf)
                rho_tso_ess = pe.value(tso_model[year][day].rho_ess)
                rho_dso_v = pe.value(dso_model[year][day].rho_v)
                rho_dso_pf = pe.value(dso_model[year][day].rho_pf)
                rho_dso_ess = pe.value(dso_model[year][day].rho_ess)
                rho_esso_ess = pe.value(esso_model[node_id].rho)

                for p in range(planning_problem.num_instants):
                    tso_v = consensus_vars['vmag']['tso']['current'][node_id][year][day][p]
                    dso_v = consensus_vars['vmag']['dso']['current'][node_id][year][day][p]
                    sums['primal']['v'] += abs(tso_v - dso_v) / interface_v_base
                    counts['primal']['v'] += 1

                    for power_type in ('p', 'q'):
                        tso_pf = consensus_vars['pf']['tso']['current'][node_id][year][day][power_type][p]
                        dso_pf = consensus_vars['pf']['dso']['current'][node_id][year][day][power_type][p]
                        sums['primal']['pf'] += abs(tso_pf - dso_pf) / interface_rating
                        counts['primal']['pf'] += 1

                        tso_ess = consensus_vars['ess']['tso']['current'][node_id][year][day][power_type][p]
                        dso_ess = consensus_vars['ess']['dso']['current'][node_id][year][day][power_type][p]
                        esso_ess = consensus_vars['ess']['esso']['current'][node_id][year][day][power_type][p]
                        sums['primal']['ess'] += (
                            abs(tso_ess - dso_ess)
                            + abs(dso_ess - esso_ess)
                            + abs(esso_ess - tso_ess)
                        ) / shared_ess_rating
                        counts['primal']['ess'] += 3

                    for agent, rho in (('tso', rho_tso_v), ('dso', rho_dso_v)):
                        current = consensus_vars['vmag'][agent]['current'][node_id][year][day][p]
                        previous = consensus_vars['vmag'][agent]['prev'][node_id][year][day][p]
                        sums['dual']['v'] += rho * abs(current - previous) / interface_v_base
                        counts['dual']['v'] += 1

                    for power_type in ('p', 'q'):
                        for agent, rho in (('tso', rho_tso_pf), ('dso', rho_dso_pf)):
                            current = consensus_vars['pf'][agent]['current'][node_id][year][day][power_type][p]
                            previous = consensus_vars['pf'][agent]['prev'][node_id][year][day][power_type][p]
                            sums['dual']['pf'] += rho * abs(current - previous) / interface_rating
                            counts['dual']['pf'] += 1

                        for agent, rho in (
                                ('tso', rho_tso_ess),
                                ('dso', rho_dso_ess),
                                ('esso', rho_esso_ess)):
                            current = consensus_vars['ess'][agent]['current'][node_id][year][day][power_type][p]
                            previous = consensus_vars['ess'][agent]['prev'][node_id][year][day][power_type][p]
                            sums['dual']['ess'] += rho * abs(current - previous) / shared_ess_rating
                            counts['dual']['ess'] += 1

    return {
        residual_type: {
            group: sums[residual_type][group] / max(counts[residual_type][group], 1)
            for group in ('v', 'pf', 'ess')
        }
        for residual_type in ('primal', 'dual')
    }


def check_admm_convergence(planning_problem, consensus_vars, residual_metrics, params, debug_flag=False):
    consensus_convergence = check_consensus_convergence(residual_metrics, params)
    stationary_convergence = check_stationary_convergence(residual_metrics, params)
    if not consensus_convergence and debug_flag:
        print_debug_info(
            planning_problem,
            consensus_vars,
            print_vmag=True,
            print_pf=True,
            print_ess=True,
        )
    return consensus_convergence and stationary_convergence


def _admm_local_solves_succeeded(planning_problem, results):
    for year in planning_problem.years:
        for day in planning_problem.days:
            if not _solver_result_succeeded(results['tso'][year][day]):
                return False
            for node_id in planning_problem.active_distribution_network_nodes:
                if not _solver_result_succeeded(results['dso'][node_id][year][day]):
                    return False
    for node_id in planning_problem.active_distribution_network_nodes:
        if not _solver_result_succeeded(results['esso'][node_id]):
            return False
    return True


def _solver_result_succeeded(result):
    return solver_result_succeeded(result)


def check_consensus_convergence(residual_metrics, params):
    convergence = True
    labels = {'v': 'interface Vmag', 'pf': 'interface PF', 'ess': 'shared ESS'}
    for group in ('v', 'pf', 'ess'):
        residual = residual_metrics['primal'][group]
        tolerance = params.tol['consensus'][group]
        if not _admm_metric_within_tolerance(residual, tolerance):
            print(
                f'[INFO]\t\t - {labels[group]} primal residual failed. '
                f'{residual:.6f} > {tolerance:.6f}'
            )
            convergence = False
    if convergence:
        print('[INFO]\t\t - Primal residuals ok!')
    return convergence


def check_stationary_convergence(residual_metrics, params):
    convergence = True
    labels = {'v': 'interface Vmag', 'pf': 'interface PF', 'ess': 'shared ESS'}
    for group in ('v', 'pf', 'ess'):
        residual = residual_metrics['dual'][group]
        tolerance = params.tol['stationarity'][group]
        if not _admm_metric_within_tolerance(residual, tolerance):
            print(
                f'[INFO]\t\t - {labels[group]} dual residual failed. '
                f'{residual:.6f} > {tolerance:.6f}'
            )
            convergence = False
    if convergence:
        print('[INFO]\t\t - Dual residuals ok!')
    return convergence


def _admm_metric_within_tolerance(value, tolerance):
    return value <= tolerance


def _get_admm_penalty_summary(tso_model, dso_models, esso_model):
    penalties = {'v': [], 'pf': [], 'ess': []}
    for year_models in tso_model.values():
        for model in year_models.values():
            penalties['v'].append(pe.value(model.rho_v))
            penalties['pf'].append(pe.value(model.rho_pf))
            penalties['ess'].append(pe.value(model.rho_ess))
    for node_models in dso_models.values():
        for year_models in node_models.values():
            for model in year_models.values():
                penalties['v'].append(pe.value(model.rho_v))
                penalties['pf'].append(pe.value(model.rho_pf))
                penalties['ess'].append(pe.value(model.rho_ess))
    for model in esso_model.values():
        penalties['ess'].append(pe.value(model.rho))
    return {
        group: sum(values) / max(len(values), 1)
        for group, values in penalties.items()
    }


def _update_admm_penalties(tso_model, dso_models, esso_model, residual_metrics, params,
                           allow_update=True):
    before = _get_admm_penalty_summary(tso_model, dso_models, esso_model)
    actions = dict()
    factors = dict()
    update_params = params.penalty_update

    for group in ('v', 'pf', 'ess'):
        primal = residual_metrics['primal'][group]
        dual = residual_metrics['dual'][group]
        normalized_primal = primal / params.tol['consensus'][group]
        normalized_dual = dual / params.tol['stationarity'][group]
        group_converged = (
            _admm_metric_within_tolerance(primal, params.tol['consensus'][group])
            and _admm_metric_within_tolerance(dual, params.tol['stationarity'][group])
        )
        factor = 1.0
        action = 'held'
        if not params.adaptive_penalty:
            action = 'fixed'
        elif not allow_update:
            action = 'held after solver failure'
        elif not group_converged:
            if normalized_primal > update_params['residual_balance_ratio'] * normalized_dual:
                factor = update_params['increase_factor']
                action = 'increased'
            elif normalized_dual > update_params['residual_balance_ratio'] * normalized_primal:
                factor = 1.0 / update_params['decrease_factor']
                action = 'decreased'
        actions[group] = action
        factors[group] = factor

    if params.adaptive_penalty and allow_update:
        for year_models in tso_model.values():
            for model in year_models.values():
                _scale_admm_penalty(model.rho_v, factors['v'], update_params)
                _scale_admm_penalty(model.rho_pf, factors['pf'], update_params)
                _scale_admm_penalty(model.rho_ess, factors['ess'], update_params)
                if hasattr(model, 'rho_ess_prev'):
                    _scale_admm_penalty(model.rho_ess_prev, factors['ess'], update_params)
        for node_models in dso_models.values():
            for year_models in node_models.values():
                for model in year_models.values():
                    _scale_admm_penalty(model.rho_v, factors['v'], update_params)
                    _scale_admm_penalty(model.rho_pf, factors['pf'], update_params)
                    _scale_admm_penalty(model.rho_ess, factors['ess'], update_params)
                    if hasattr(model, 'rho_ess_prev'):
                        _scale_admm_penalty(model.rho_ess_prev, factors['ess'], update_params)
        for model in esso_model.values():
            _scale_admm_penalty(model.rho, factors['ess'], update_params)

    after = _get_admm_penalty_summary(tso_model, dso_models, esso_model)
    return actions, before, after


def _scale_admm_penalty(penalty, factor, params):
    value = pe.value(penalty) * factor
    penalty.set_value(min(max(value, params['min']), params['max']))


def _update_interface_power_flow_variables(planning_problem, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=True, update_dns=True):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    # Transmission network - Update Vmag and PF at the TN-DN interface
    if update_tn:
        for dn in range(len(planning_problem.active_distribution_network_nodes)):
            node_id = planning_problem.active_distribution_network_nodes[dn]
            for year in planning_problem.years:
                for day in planning_problem.days:
                    v_base = transmission_network.network[year][day].get_node_base_kv(node_id)
                    s_base = transmission_network.network[year][day].baseMVA
                    if _solver_result_succeeded(results['tso'][year][day]):
                        for p in tso_model[year][day].periods:
                            interface_vars['vmag']['tso']['prev'][node_id][year][day][p] = copy(interface_vars['vmag']['tso']['current'][node_id][year][day][p])
                            interface_vars['pf']['tso']['prev'][node_id][year][day]['p'][p] = copy(interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p])
                            interface_vars['pf']['tso']['prev'][node_id][year][day]['q'][p] = copy(interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p])

                            vmag_req = pe.value(tso_model[year][day].expected_interface_vmag[dn, p]) * v_base
                            p_req = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                            q_req = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base
                            interface_vars['vmag']['tso']['current'][node_id][year][day][p] = vmag_req
                            interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p] = p_req
                            interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p] = q_req

    # Distribution Network - Update PF at the TN-DN interface
    if update_dns:
        for node_id in distribution_networks:
            distribution_network = distribution_networks[node_id]
            dso_model = dso_models[node_id]
            for year in planning_problem.years:
                for day in planning_problem.days:
                    ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                    v_base = distribution_network.network[year][day].get_node_base_kv(ref_node_id)
                    s_base = distribution_network.network[year][day].baseMVA
                    if _solver_result_succeeded(results['dso'][node_id][year][day]):
                        for p in dso_model[year][day].periods:
                            interface_vars['vmag']['dso']['prev'][node_id][year][day][p] = copy(interface_vars['vmag']['dso']['current'][node_id][year][day][p])
                            interface_vars['pf']['dso']['prev'][node_id][year][day]['p'][p] = copy(interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p])
                            interface_vars['pf']['dso']['prev'][node_id][year][day]['q'][p] = copy(interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p])

                            vmag_req = pe.value(dso_model[year][day].expected_interface_vmag[p]) * v_base
                            p_req = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                            q_req = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base
                            interface_vars['vmag']['dso']['current'][node_id][year][day][p] = vmag_req
                            interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p] = p_req
                            interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p] = q_req

    # Update Lambdas
    for node_id in distribution_networks:
        for year in planning_problem.years:
            for day in planning_problem.days:
                tso_succeeded = (
                    _solver_result_succeeded(results['tso'][year][day])
                    if update_tn else False
                )
                dso_succeeded = (
                    _solver_result_succeeded(results['dso'][node_id][year][day])
                    if update_tn or update_dns else False
                )
                for p in range(planning_problem.num_instants):

                    if update_tn and tso_succeeded and dso_succeeded:
                        rho_v_tso = pe.value(tso_model[year][day].rho_v)
                        rho_pf_tso = pe.value(tso_model[year][day].rho_pf)
                        error_v_req_tso = interface_vars['vmag']['tso']['current'][node_id][year][day][p] - interface_vars['vmag']['dso']['current'][node_id][year][day][p]
                        error_p_pf_req_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p] - interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p]
                        error_q_pf_req_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p] - interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p]
                        dual_vars['vmag']['tso']['current'][node_id][year][day][p] += rho_v_tso * error_v_req_tso
                        dual_vars['pf']['tso']['current'][node_id][year][day]['p'][p] += rho_pf_tso * error_p_pf_req_tso
                        dual_vars['pf']['tso']['current'][node_id][year][day]['q'][p] += rho_pf_tso * error_q_pf_req_tso

                    if update_dns and dso_succeeded:
                        rho_v_dso = pe.value(dso_models[node_id][year][day].rho_v)
                        rho_pf_dso = pe.value(dso_models[node_id][year][day].rho_pf)
                        error_v_req_dso = interface_vars['vmag']['dso']['current'][node_id][year][day][p] - interface_vars['vmag']['tso']['current'][node_id][year][day][p]
                        error_p_pf_req_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p] - interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p]
                        error_q_pf_req_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['q'][p] - interface_vars['pf']['tso']['current'][node_id][year][day]['q'][p]
                        dual_vars['vmag']['dso']['current'][node_id][year][day][p] += rho_v_dso * error_v_req_dso
                        dual_vars['pf']['dso']['current'][node_id][year][day]['p'][p] += rho_pf_dso * error_p_pf_req_dso
                        dual_vars['pf']['dso']['current'][node_id][year][day]['q'][p] += rho_pf_dso * error_q_pf_req_dso


def _update_shared_energy_storage_variables(planning_problem, tso_model, dso_models, sess_model, shared_ess_vars, dual_vars, results, params, update_tn=True, update_dns=True, update_sess=True):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data
    repr_days = [day for day in planning_problem.days]
    repr_years = [year for year in planning_problem.years]

    for node_id in planning_problem.active_distribution_network_nodes:

        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]

        # Power requested by ESSO
        if update_sess:
            for y in sess_model[node_id].years:
                year = repr_years[y]
                if _solver_result_succeeded(results['esso'][node_id]):
                    for d in sess_model[node_id].days:
                        day = repr_days[d]
                        for p in sess_model[node_id].periods:
                            shared_ess_vars['esso']['prev'][node_id][year][day]['p'][p] = copy(shared_ess_vars['esso']['current'][node_id][year][day]['p'][p])
                            shared_ess_vars['esso']['prev'][node_id][year][day]['q'][p] = copy(shared_ess_vars['esso']['current'][node_id][year][day]['q'][p])

                            p_req = pe.value(sess_model[node_id].es_pnet[y, d, p])
                            q_req = pe.value(sess_model[node_id].es_qnet[y, d, p])
                            shared_ess_vars['esso']['current'][node_id][year][day]['p'][p] = p_req
                            shared_ess_vars['esso']['current'][node_id][year][day]['q'][p] = q_req

        # Power requested by TSO
        if update_tn:
            for y in range(len(repr_years)):
                year = repr_years[y]
                for d in range(len(repr_days)):
                    day = repr_days[d]
                    if _solver_result_succeeded(results['tso'][year][day]):
                        s_base = transmission_network.network[year][day].baseMVA
                        shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                        for p in tso_model[year][day].periods:
                            shared_ess_vars['tso']['prev'][node_id][year][day]['p'][p] = copy(shared_ess_vars['tso']['current'][node_id][year][day]['p'][p])
                            shared_ess_vars['tso']['prev'][node_id][year][day]['q'][p] = copy(shared_ess_vars['tso']['current'][node_id][year][day]['q'][p])

                            p_req = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base
                            q_req = pe.value(tso_model[year][day].expected_shared_ess_q[shared_ess_idx, p]) * s_base
                            shared_ess_vars['tso']['current'][node_id][year][day]['p'][p] = p_req
                            shared_ess_vars['tso']['current'][node_id][year][day]['q'][p] = q_req


        # Power requested by DSO
        if update_dns:
            for y in range(len(repr_years)):
                year = repr_years[y]
                for d in range(len(repr_days)):
                    day = repr_days[d]
                    if _solver_result_succeeded(results['dso'][node_id][year][day]):
                        s_base = distribution_network.network[year][day].baseMVA
                        for p in dso_model[year][day].periods:
                            shared_ess_vars['dso']['prev'][node_id][year][day]['p'][p] = copy(shared_ess_vars['dso']['current'][node_id][year][day]['p'][p])
                            shared_ess_vars['dso']['prev'][node_id][year][day]['q'][p] = copy(shared_ess_vars['dso']['current'][node_id][year][day]['q'][p])

                            p_req = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base
                            q_req = pe.value(dso_model[year][day].expected_shared_ess_q[p]) * s_base
                            shared_ess_vars['dso']['current'][node_id][year][day]['p'][p] = p_req
                            shared_ess_vars['dso']['current'][node_id][year][day]['q'][p] = q_req

        # Update dual variables SharedESS
        for year in planning_problem.years:
            for day in planning_problem.days:
                tso_succeeded = (
                    _solver_result_succeeded(results['tso'][year][day])
                    if update_tn or update_sess else False
                )
                dso_succeeded = (
                    _solver_result_succeeded(results['dso'][node_id][year][day])
                    if update_tn or update_dns else False
                )
                esso_succeeded = (
                    _solver_result_succeeded(results['esso'][node_id])
                    if update_sess else False
                )
                for p in range(planning_problem.num_instants):

                    if update_tn and tso_succeeded and dso_succeeded:
                        rho_ess_tso = pe.value(tso_model[year][day].rho_ess)
                        error_p_tso_dso = shared_ess_vars['tso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['dso']['current'][node_id][year][day]['p'][p]
                        error_q_tso_dso = shared_ess_vars['tso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['dso']['current'][node_id][year][day]['q'][p]
                        dual_vars['tso']['current'][node_id][year][day]['p'][p] += rho_ess_tso * error_p_tso_dso
                        dual_vars['tso']['current'][node_id][year][day]['q'][p] += rho_ess_tso * error_q_tso_dso
                        if params.previous_iter['ess']['tso']:
                            rho_ess_tso_prev = pe.value(tso_model[year][day].rho_ess_prev)
                            error_p_tso_prev = shared_ess_vars['tso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['tso']['prev'][node_id][year][day]['p'][p]
                            error_q_tso_prev = shared_ess_vars['tso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['tso']['prev'][node_id][year][day]['q'][p]
                            dual_vars['tso']['prev'][node_id][year][day]['p'][p] += rho_ess_tso_prev * error_p_tso_prev
                            dual_vars['tso']['prev'][node_id][year][day]['q'][p] += rho_ess_tso_prev * error_q_tso_prev

                    if update_dns and dso_succeeded:
                        rho_ess_dso = pe.value(dso_models[node_id][year][day].rho_ess)
                        error_p_dso_esso = shared_ess_vars['dso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['esso']['current'][node_id][year][day]['p'][p]
                        error_q_dso_esso = shared_ess_vars['dso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['esso']['current'][node_id][year][day]['q'][p]
                        dual_vars['dso']['current'][node_id][year][day]['p'][p] += rho_ess_dso * error_p_dso_esso
                        dual_vars['dso']['current'][node_id][year][day]['q'][p] += rho_ess_dso * error_q_dso_esso
                        if params.previous_iter['ess']['dso']:
                            rho_ess_dso_prev = pe.value(dso_models[node_id][year][day].rho_ess_prev)
                            error_p_dso_prev = shared_ess_vars['dso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['dso']['prev'][node_id][year][day]['p'][p]
                            error_q_dso_prev = shared_ess_vars['dso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['dso']['prev'][node_id][year][day]['q'][p]
                            dual_vars['dso']['prev'][node_id][year][day]['p'][p] += rho_ess_dso_prev * error_p_dso_prev
                            dual_vars['dso']['prev'][node_id][year][day]['q'][p] += rho_ess_dso_prev * error_q_dso_prev

                    if update_sess and esso_succeeded and tso_succeeded:
                        rho_ess_sess = pe.value(sess_model[node_id].rho)
                        error_p_esso_tso = shared_ess_vars['esso']['current'][node_id][year][day]['p'][p] - shared_ess_vars['tso']['current'][node_id][year][day]['p'][p]
                        error_q_esso_tso = shared_ess_vars['esso']['current'][node_id][year][day]['q'][p] - shared_ess_vars['tso']['current'][node_id][year][day]['q'][p]
                        dual_vars['esso']['current'][node_id][year][day]['p'][p] += rho_ess_sess * error_p_esso_tso
                        dual_vars['esso']['current'][node_id][year][day]['q'][p] += rho_ess_sess * error_q_esso_tso


# ======================================================================================================================
#  OPERATIONAL PLANNING WITHOUT COORDINATION functions
# ======================================================================================================================
def _run_operational_planning_without_coordination(planning_problem):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    results = {'tso': dict(), 'dso': dict(), 'esso': dict()}

    # SharedESS candidate solution (no shared ESS)
    candidate_solution = dict()
    for e in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[e]
        candidate_solution[node_id] = dict()
        for year in planning_problem.years:
            candidate_solution[node_id][year] = dict()
            candidate_solution[node_id][year]['s'] = 0.00
            candidate_solution[node_id][year]['e'] = 0.00

    start = time.time()

    # Create interface PF variables
    interface_vmag, interface_pf = create_interface_power_flow_variables(planning_problem)

    # Create DSOs' Operational Planning models, run initial SMOPF
    dso_models = dict()
    for node_id in distribution_networks:

        distribution_network = distribution_networks[node_id]
        results['dso'][node_id] = dict()

        # Build model, fix candidate solution, and Run S-MPOPF model
        distribution_network.update_data_with_candidate_solution(candidate_solution)
        dso_model = distribution_network.build_model()
        distribution_network.update_model_with_candidate_solution(dso_model, candidate_solution)

        # Update model with expected interface values
        # Regularization -- Added to OF to minimize deviations from scenarios to expected values
        for year in distribution_network.years:
            for day in distribution_network.days:

                s_base = distribution_network.network[year][day].baseMVA

                # Add interface expected variables, and their definition
                dso_model[year][day].expected_interface_vmag = pe.Var(dso_model[year][day].periods, domain=pe.NonNegativeReals, initialize=1.00)
                dso_model[year][day].expected_interface_pf_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
                dso_model[year][day].expected_interface_pf_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals, initialize=0.0)
                dso_model[year][day].expected_interface_vmag_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_vmag_rule, network=distribution_network.network[year][day]))
                dso_model[year][day].expected_interface_pf_p_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_pf_p_rule, network=distribution_network.network[year][day]))
                dso_model[year][day].expected_interface_pf_q_def = pe.Constraint(dso_model[year][day].periods, rule=partial(dn_interface_expected_pf_q_rule, network=distribution_network.network[year][day]))

                # Regularization -- decrease variance between scenarios
                obj = copy(dso_model[year][day].objective.expr)
                dso_model[year][day].penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION)
                for s_m in dso_model[year][day].scenarios_market:
                    for s_o in dso_model[year][day].scenarios_operation:
                        for p in dso_model[year][day].periods:
                            obj += dso_model[year][day].penalty_regularization * (dso_model[year][day].vmag_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_vmag[p]) ** 2
                            obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].pg_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_pf_p[p]) ** 2
                            obj += dso_model[year][day].penalty_regularization * s_base * (dso_model[year][day].qg_adn[s_m, s_o, p] - dso_model[year][day].expected_interface_pf_q[p]) ** 2
                dso_model[year][day].objective.expr = obj

        results['dso'][node_id] = distribution_network.optimize(dso_model)

        # Get initial interface PF values
        for year in distribution_network.years:
            for day in distribution_network.days:
                s_base = distribution_network.network[year][day].baseMVA
                for p in dso_model[year][day].periods:
                    interface_vmag[node_id][year][day][p] = pe.value(dso_model[year][day].expected_interface_vmag[p])
                    interface_pf[node_id][year][day]['p'][p] = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    interface_pf[node_id][year][day]['q'][p] = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base

        dso_models[node_id] = dso_model

    # Create TSO Operational Planning model
    transmission_network.update_data_with_candidate_solution(candidate_solution)
    tso_model = transmission_network.build_model()
    transmission_network.update_model_with_candidate_solution(tso_model, candidate_solution)

    # TSO -- Add expected values
    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA
            tso_model[year][day].active_distribution_networks = range(len(transmission_network.active_distribution_network_nodes))

            # Free Pc, Qc at the interface nodes
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                adn_load_idx = transmission_network.network[year][day].get_adn_load_idx(adn_node_id)
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].pc[adn_load_idx, s_m, s_o, p].setlb(None)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].fixed = False
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setub(None)
                            tso_model[year][day].qc[adn_load_idx, s_m, s_o, p].setlb(None)
                            if transmission_network.params.fl_reg:
                                tso_model[year][day].flex_p_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].flex_p_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].flex_q_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].flex_q_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                            if transmission_network.params.l_curt:
                                tso_model[year][day].pc_curt_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].pc_curt_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].qc_curt_down[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)
                                tso_model[year][day].qc_curt_up[adn_load_idx, s_m, s_o, p].setub(EQUALITY_TOLERANCE)

            # "Fix" Pc and Qc (DNs' solutions)
            # Regularization -- decrease variance between scenarios
            obj = copy(tso_model[year][day].objective.expr)
            tso_model[year][day].penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION * 1e6)
            for dn in tso_model[year][day].active_distribution_networks:
                adn_node_id = transmission_network.active_distribution_network_nodes[dn]
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            vmag_req = interface_vmag[adn_node_id][year][day][p]
                            p_req = interface_pf[adn_node_id][year][day]['p'][p] / s_base
                            q_req = interface_pf[adn_node_id][year][day]['q'][p] / s_base
                            obj += tso_model[year][day].penalty_regularization * (tso_model[year][day].vmag_adn[dn, s_m, s_o, p] - vmag_req) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].pc_adn[dn, s_m, s_o, p] - p_req) ** 2
                            obj += tso_model[year][day].penalty_regularization * s_base * (tso_model[year][day].qc_adn[dn, s_m, s_o, p] - q_req) ** 2

            tso_model[year][day].objective.expr = obj

    results['tso'] = transmission_network.optimize(tso_model)

    end = time.time()
    total_execution_time = end - start
    print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))

    models = {'tso': tso_model, 'dso': dso_models}

    return results, models, total_execution_time


def create_interface_power_flow_variables(planning_problem):
    consensus_vars, _ = create_admm_variables(planning_problem)
    return consensus_vars['vmag']['dso']['current'], consensus_vars['pf']['dso']['current']


# ======================================================================================================================
#  PLANNING PROBLEM read functions
# ======================================================================================================================
def _read_planning_problem(planning_problem):

    # Create results and diagrams folder
    os.makedirs(planning_problem.results_dir, exist_ok=True)
    os.makedirs(planning_problem.diagrams_dir, exist_ok=True)
    os.makedirs(planning_problem.logs_dir, exist_ok=True)

    # Read specification file
    filename = os.path.join(planning_problem.data_dir, planning_problem.filename)
    planning_data = convert_json_to_dict(read_json_file(filename))

    # General Parameters
    for year in planning_data['Years']:
        planning_problem.years[int(year)] = planning_data['Years'][year]
    planning_problem.days = planning_data['Days']
    planning_problem.num_instants = planning_data['NumInstants']

    # MarketData
    print('[INFO] Reading MARKET DATA from file(s)...')
    planning_problem.discount_factor = planning_data['DiscountFactor']
    planning_problem.market_data_file = planning_data['MarketData']
    planning_problem.num_market_scenarios = planning_data['NumMarketScenarios']
    planning_problem.plot_market_data = planning_data['PlotMarketData']
    planning_problem.read_market_data_from_file()
    if planning_problem.plot_market_data:
        planning_problem.plot_market_price_scenarios()

    # Distribution Networks
    planning_problem.parallel_execution = planning_data['ParallelExecution']
    for distribution_network in planning_data['DistributionNetworks']:

        print('[INFO] Reading DISTRIBUTION NETWORK DATA from file(s)...')

        network_name = distribution_network['name']                             # Network filename
        operational_data_file = distribution_network['operational_data_file']   # Operational data filename
        num_oper_scenarios = distribution_network['num_operation_scenarios']    # Number of operational scenarios
        plot_oper_data = distribution_network['plot_operational_data']          # Plot operational data
        params_file = distribution_network['params_file']                       # Params filename
        connection_nodeid = distribution_network['connection_node_id']          # Connection node ID

        distribution_network = NetworkData()
        distribution_network.name = network_name
        distribution_network.is_transmission = False
        distribution_network.data_dir = planning_problem.data_dir
        distribution_network.results_dir = planning_problem.results_dir
        distribution_network.diagrams_dir = planning_problem.diagrams_dir
        distribution_network.logs_dir = planning_problem.logs_dir
        distribution_network.years = planning_problem.years
        distribution_network.days = planning_problem.days
        distribution_network.num_oper_scenarios = num_oper_scenarios
        distribution_network.plot_operational_data = plot_oper_data
        distribution_network.num_instants = planning_problem.num_instants
        distribution_network.discount_factor = planning_problem.discount_factor
        distribution_network.cost_energy_p = planning_problem.cost_energy_p
        distribution_network.cost_flex = planning_problem.cost_flex
        distribution_network.params_file = params_file
        distribution_network.read_network_parameters()
        distribution_network.operational_data_file = operational_data_file
        distribution_network.read_network_data()
        if distribution_network.plot_operational_data:
            distribution_network.plot_operational_data_scenarios()
        for year in distribution_network.years:
            for day in distribution_network.days:
                distribution_network.network[year][day].is_transmission = False
                distribution_network.network[year][day].tn_connection_nodeid = connection_nodeid
                if distribution_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
                    distribution_network.network[year][day].prob_market_scenarios = [1.00]
                else:
                    distribution_network.network[year][day].prob_market_scenarios = planning_problem.prob_market_scenarios[year]
                    distribution_network.network[year][day].cost_energy_p = planning_problem.cost_energy_p[year][day]
                    distribution_network.network[year][day].cost_flex = planning_problem.cost_flex[year][day]
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
    transmission_network.logs_dir = planning_problem.logs_dir
    transmission_network.years = planning_problem.years
    transmission_network.days = planning_problem.days
    transmission_network.num_oper_scenarios = planning_data['TransmissionNetwork']['num_operation_scenarios']
    transmission_network.plot_operational_data = planning_data['TransmissionNetwork']['plot_operational_data']
    transmission_network.num_instants = planning_problem.num_instants
    transmission_network.discount_factor = planning_problem.discount_factor
    transmission_network.cost_energy_p = planning_problem.cost_energy_p
    transmission_network.cost_flex = planning_problem.cost_flex
    transmission_network.params_file = planning_data['TransmissionNetwork']['params_file']
    transmission_network.read_network_parameters()
    transmission_network.operational_data_file = planning_data['TransmissionNetwork']['operational_data_file']
    transmission_network.read_network_data()
    if transmission_network.plot_operational_data:
        transmission_network.plot_operational_data_scenarios()
    for year in transmission_network.years:
        for day in transmission_network.days:
            transmission_network.network[year][day].is_transmission = True
            transmission_network.network[year][day].active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]
            if transmission_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
                transmission_network.network[year][day].prob_market_scenarios = [1.00]
            else:
                transmission_network.network[year][day].prob_market_scenarios = planning_problem.prob_market_scenarios[year]
                transmission_network.network[year][day].cost_energy_p = planning_problem.cost_energy_p[year][day]
                transmission_network.network[year][day].cost_flex = planning_problem.cost_flex[year][day]
    transmission_network.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]
    planning_problem.transmission_network = transmission_network

    # SharedESS
    print('[INFO] Reading SHARED ESS DATA from file(s)...')
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
    print(f'[INFO] Reading PLANNING PARAMETERS from file...')
    planning_problem.params_file = planning_data['PlanningParameters']['params_file']
    planning_problem.read_planning_parameters_from_file()

    _check_interface_nodes_base_voltage_consistency(planning_problem)

    # Add ADN nodes to Transmission Network
    _add_adn_node_to_transmission_network(planning_problem)

    # Add Shared Energy Storages to Transmission and Distribution Networks
    _add_shared_energy_storage_to_transmission_network(planning_problem)
    _add_shared_energy_storage_to_distribution_network(planning_problem)


# ======================================================================================================================
#  MARKET DATA read functions
# ======================================================================================================================
def _read_market_data_from_file(planning_problem):

    filename = os.path.join(planning_problem.data_dir, 'MarketData', planning_problem.market_data_file)

    try:
        base_profiles = _read_market_base_profiles(filename)
    except:
        print(f'[ERROR] Reading market data from file(s). Exiting...')
        exit(ERROR_SPECIFICATION_FILE)

    synthetic_profiles = _generate_market_price_scenarios(base_profiles)

    # Update subsequent years
    initial_year = list(planning_problem.years)[0]
    growth_factors = base_profiles['growth_factors']
    energy_growth_factor = float(growth_factors[growth_factors['Growth factors'] == 'Energy']['Value, [%]'].iloc[0])
    flexibility_growth_factor = float(growth_factors[growth_factors['Growth factors'] == 'Flexibility']['Value, [%]'].iloc[0])

    for year in planning_problem.years:

        planning_problem.prob_market_scenarios[year] = [(1 / planning_problem.num_market_scenarios)] * planning_problem.num_market_scenarios
        planning_problem.cost_energy_p[year] = dict()
        planning_problem.cost_flex[year] = dict()

        for day in planning_problem.days:

            energy_growth_cumul = (1 + energy_growth_factor) ** (year - initial_year)
            flexibility_growth_cumul = (1 + flexibility_growth_factor) ** (year - initial_year)

            energy_selected_profiles = synthetic_profiles['energy'][day].sample(n=planning_problem.num_market_scenarios)
            flexibility_selected_profiles = synthetic_profiles['flexibility'][day].sample(n=planning_problem.num_market_scenarios)

            planning_problem.cost_energy_p[year][day] = np.array(energy_selected_profiles * energy_growth_cumul)      # n_scenarios x n_instants
            planning_problem.cost_flex[year][day] = np.array(flexibility_selected_profiles * flexibility_growth_cumul)


def _read_market_base_profiles(filename):

    base_cost_data = {
        'growth_factors': pd.read_excel(filename, sheet_name='Growth Factors'),
        'energy': pd.read_excel(filename, sheet_name='Energy'),
        'flexibility': pd.read_excel(filename, sheet_name='Flexibility')
    }

    return base_cost_data


def _generate_market_price_scenarios(base_profiles, n_samples=100, bandwidth=0.10):

    print('[INFO] \t - Generating market scenarios...')

    energy_df = base_profiles['energy']
    flex_df = base_profiles['flexibility']

    synthetic_profiles = {
        'energy': _generate_market_price_scenarios_per_type(energy_df, n_samples=n_samples, bandwidth=bandwidth),
        'flexibility': _generate_market_price_scenarios_per_type(flex_df, n_samples=n_samples, bandwidth=bandwidth)
    }

    return synthetic_profiles


def _generate_market_price_scenarios_per_type(base_profiles, n_samples=100, bandwidth=0.05):

    seasons = base_profiles['Season'].unique()
    synthetic_profiles = {}

    for season in seasons:

        # Filter
        price_subset = base_profiles[(base_profiles['Season'] == season)]
        if price_subset.empty:
            print(f'[ERROR] No market data provided for season {season}')
            exit(ERROR_MARKET_DATA_FILE)

        # Prepare data
        price_hours = price_subset.iloc[:, 2:].copy()

        # Normalize and fit copula
        scaler = StandardScaler()
        price_scaled = scaler.fit_transform(price_hours)

        model = GaussianMultivariate(distribution=CustomGaussianKDE(bandwidth=bandwidth))
        model.fit(pd.DataFrame(price_scaled))

        # Sample
        samples = model.sample(n_samples)
        samples = scaler.inverse_transform(samples)

        synthetic_profiles[season] = pd.DataFrame(samples)

    return synthetic_profiles


def _plot_market_price_scenarios(planning_problem, years_to_plot, save_dir, save_format='pdf'):

    print('[INFO] \t - Plotting market scenarios...')

    hours = np.arange(planning_problem.num_instants)
    xticks = np.arange(0, planning_problem.num_instants, 4)
    xtick_labels = [f"{h:02d}:00" for h in xticks]

    for year in years_to_plot:
        for season in planning_problem.days:

            cost_energy_p = planning_problem.cost_energy_p[year][season]
            cost_flex = planning_problem.cost_flex[year][season]

            # Calculate statistics
            mean_energy_p = cost_energy_p.mean(axis=0)
            std_energy_p = cost_energy_p.std(axis=0)
            mean_flex = cost_flex.mean(axis=0)
            std_flex = cost_flex.std(axis=0)

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(11, 6), sharex=True)

            color_energy_p = cm.tab10(0)
            color_flex = cm.tab10(1)
            ax.plot(hours, mean_energy_p, label='Energy', color=color_energy_p)
            ax.fill_between(hours, mean_energy_p - std_energy_p, mean_energy_p + std_energy_p, alpha=0.2, color=color_energy_p)
            ax.plot(hours, mean_flex, label='Flexibility', color=color_flex)
            ax.fill_between(hours, mean_flex - std_flex, mean_flex + std_flex, alpha=0.2, color=color_flex)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlim(0, 23)
            ax.set_xticklabels(xtick_labels, fontsize=12)
            ax.set_xlabel("Hour", loc='center', fontsize=14)
            ax.set_ylabel("Market Price, [€/MW]", fontsize=16)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.grid(True)
            ax.legend(loc='lower right', fontsize=11)
            plt.tight_layout()

            filename = os.path.join(save_dir, f"{planning_problem.name}_market_prices_{year}_{season}.{save_format}")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)


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
    processed_results['summary_detail'] = dict()

    processed_results['tso'] = transmission_network.process_results(tso_model, optimization_results['tso'])
    for node_id in distribution_networks:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        processed_results['dso'][node_id] = distribution_network.process_results(dso_model, optimization_results['dso'][node_id])
    processed_results['esso'] = shared_ess_data.process_results(esso_model)
    processed_results['interface'] = _process_results_interface(operational_planning_problem, tso_model, dso_models)
    processed_results['summary_detail'] = _process_results_summary_detail(operational_planning_problem, tso_model, dso_models)

    return processed_results


def _process_operational_planning_results_hierarchical(planning_problem, tso_model, dso_models, optimization_results):
    return _process_operational_planning_results_no_coordination(planning_problem, tso_model, dso_models, optimization_results)


def _process_operational_planning_results_no_coordination(planning_problem, tso_model, dso_models, optimization_results, execution_time=float()):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    processed_results = dict()
    processed_results['tso'] = dict()
    processed_results['dso'] = dict()
    processed_results['summary_detail'] = dict()

    processed_results['tso'] = transmission_network.process_results(tso_model, optimization_results['tso'])
    for node_id in distribution_networks:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        processed_results['dso'][node_id] = distribution_network.process_results(dso_model, optimization_results['dso'][node_id])
    processed_results['summary_detail'] = _process_results_summary_detail(planning_problem, tso_model, dso_models)

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


def _process_results_summary_detail(planning_problem, tso_model, dso_models):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    processed_results = dict()
    processed_results['tso'] = dict()
    processed_results['dso'] = dict()

    processed_results['tso'] = transmission_network.process_results_summary_detail(tso_model)
    for node_id in distribution_networks:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        processed_results['dso'][node_id] = distribution_network.process_results_summary_detail(dso_model)

    return processed_results


# ======================================================================================================================
#  RESULTS PLANNING - write functions
# ======================================================================================================================
def _write_planning_results_to_excel(planning_problem, results, bound_evolution=dict(), shared_ess_cost=dict(), shared_ess_capacity=dict(), filename='planing_results', execution_time=float()):

    wb = Workbook()

    _write_operational_planning_main_info_to_excel(planning_problem, wb, results, execution_time=execution_time)
    _write_operational_planning_main_info_to_excel_detailed(planning_problem, wb, results['summary_detail'])
    _write_shared_ess_specifications(wb, planning_problem.shared_ess_data)
    _write_operational_planning_market_data_to_excel(planning_problem, wb)

    if bound_evolution:
        _write_bound_evolution_to_excel(wb, bound_evolution)
        admm_diagnostics = bound_evolution.get('admm_diagnostics', [])
        if admm_diagnostics:
            _write_admm_diagnostics_to_excel(wb, admm_diagnostics)
        finite_difference_results = bound_evolution.get('finite_difference', [])
        if finite_difference_results:
            _write_finite_difference_validation_to_excel(wb, finite_difference_results)

    if shared_ess_capacity:
        write_investment = True
        if shared_ess_cost:
            write_investment = False
        planning_problem.shared_ess_data.write_ess_capacity_results_to_excel(wb, shared_ess_capacity, write_investment=write_investment)

    if shared_ess_cost:
        planning_problem.shared_ess_data.write_ess_costs_to_excel(wb, shared_ess_cost)

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
    _write_network_branch_loading_results_to_excel(planning_problem, wb, results)
    _write_network_branch_power_flow_results_to_excel(planning_problem, wb, results)
    _write_network_energy_storages_results_to_excel(planning_problem, wb, results)
    _write_relaxation_slacks_results_to_excel(planning_problem, wb, results)
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

    master_estimate = bound_evolution.get('master_estimate', bound_evolution.get('lower_bound', []))
    columns = [
        ('master_estimate', 'Master Estimate (nominal LB), [NPV Mm.u.]', master_estimate, 1e6, '0.00'),
        ('alpha', 'Alpha, [NPV Mm.u.]', bound_evolution.get('alpha', []), 1e6, '0.00'),
        ('investment_cost', 'Investment Cost, [NPV Mm.u.]', bound_evolution.get('investment_cost', []), 1e6, '0.00'),
        ('operational_recourse', 'Operational Recourse, [NPV Mm.u.]', bound_evolution.get('operational_recourse', []), 1e6, '0.00'),
        ('candidate_total', 'Candidate Total Objective, [NPV Mm.u.]', bound_evolution.get('candidate_total', []), 1e6, '0.00'),
        ('upper_bound', 'Incumbent Upper Bound, [NPV Mm.u.]', bound_evolution.get('upper_bound', []), 1e6, '0.00'),
        ('gap_signed', 'Signed Nominal Gap (UB - Master), [NPV Mm.u.]', bound_evolution.get('gap_signed', []), 1e6, '0.00'),
        ('gap_abs', 'Absolute Nominal Gap, [NPV Mm.u.]', bound_evolution.get('gap_abs', []), 1e6, '0.00'),
        ('gap_rel', 'Relative Nominal Gap, [%]', bound_evolution.get('gap_rel', []), 0.01, '0.00'),
        ('esso_violation', 'ESSO Aggregate Feasibility Slack, [N/A]', bound_evolution.get('esso_violation', []), 1.00, '0.000000'),
    ]
    num_lines = max((len(values) for _, _, values, _, _ in columns), default=0)

    sheet.cell(row=1, column=1).value = 'Iteration'
    for column_idx, (_, label, _, _, _) in enumerate(columns, start=2):
        sheet.cell(row=1, column=column_idx).value = label

    for iteration in range(num_lines):
        row_idx = iteration + 2
        sheet.cell(row=row_idx, column=1).value = iteration + 1
        for column_idx, (_, _, values, divisor, number_format) in enumerate(columns, start=2):
            if iteration >= len(values) or values[iteration] is None:
                continue
            sheet.cell(row=row_idx, column=column_idx).value = values[iteration] / divisor
            sheet.cell(row=row_idx, column=column_idx).number_format = number_format


def _write_finite_difference_validation_to_excel(workbook, validation_results):
    sheet = workbook.create_sheet('Sensitivity Validation')
    columns = [
        ('run_type', 'Run Type', 'General'),
        ('status', 'Status', 'General'),
        ('reason', 'Reason', 'General'),
        ('node_id', 'Node ID', '0'),
        ('year', 'Investment Year', '0'),
        ('base_s', 'Base S Investment, [MVA]', '0.000000'),
        ('base_e', 'Base E Investment, [MVAh]', '0.000000'),
        ('energy_to_power_ratio', 'E/S Ratio, [h]', '0.000000'),
        ('step_fraction', 'Relative Power Step, [%]', '0.00%'),
        ('step_size', 'Power Step h, [MVA]', '0.000000'),
        ('delta_s', 'Delta S, [MVA]', '0.000000'),
        ('delta_e', 'Delta E, [MVAh]', '0.000000'),
        ('sensitivity_s', 'Sensitivity S, [NPV m.u./MVA]', '0.000000'),
        ('sensitivity_e', 'Sensitivity E, [NPV m.u./MVAh]', '0.000000'),
        ('analytic_slope', 'Analytic Directional Slope, [NPV m.u./MVA]', '0.000000'),
        ('replay_analytic_slope', 'Replay Directional Slope, [NPV m.u./MVA]', '0.000000'),
        ('predicted_change', 'Predicted Recourse Change, [NPV m.u.]', '0.000000'),
        ('baseline_recourse', 'Baseline Recourse, [NPV m.u.]', '0.000000'),
        ('reference_recourse', 'Replay Reference Recourse, [NPV m.u.]', '0.000000'),
        ('perturbed_recourse', 'Perturbed Recourse, [NPV m.u.]', '0.000000'),
        ('observed_change', 'Observed Recourse Change, [NPV m.u.]', '0.000000'),
        ('absolute_error', 'Absolute Recourse-Change Error, [NPV m.u.]', '0.000000'),
        ('observed_slope', 'Observed Directional Slope, [NPV m.u./MVA]', '0.000000'),
        ('absolute_slope_error', 'Absolute Slope Error, [NPV m.u./MVA]', '0.000000'),
        ('relative_error', 'Relative Slope Error, [%]', '0.00%'),
        ('signal_to_noise_ratio', 'Signal-to-Noise Ratio', '0.00'),
        ('slope_consistency_error', 'Step-to-Step Slope Difference, [%]', '0.00%'),
        ('replay_drift', 'Replay Recourse Drift, [NPV m.u.]', '0.000000'),
        ('replay_tolerance', 'Replay Drift Tolerance, [NPV m.u.]', '0.000000'),
        ('sensitivity_relative_drift', 'Replay Sensitivity Drift, [%]', '0.00%'),
        ('same_sign', 'Same Sign', 'General'),
        ('operational_convergence', 'ADMM Converged', 'General'),
        ('esso_violation', 'ESSO Feasibility Slack, [N/A]', '0.000000'),
        ('baseline_soh_margin', 'Baseline Minimum SoH Margin, [p.u.]', '0.000000'),
        ('reference_soh_margin', 'Replay Minimum SoH Margin, [p.u.]', '0.000000'),
        ('perturbed_soh_margin', 'Perturbed Minimum SoH Margin, [p.u.]', '0.000000'),
        ('active_set_changed', 'Minimum SoH Activity Changed', 'General'),
        ('passed', 'Passed', 'General'),
    ]

    for column_idx, (_, label, _) in enumerate(columns, start=1):
        sheet.cell(row=1, column=column_idx).value = label

    for row_idx, result in enumerate(validation_results, start=2):
        for column_idx, (key, _, number_format) in enumerate(columns, start=1):
            value = result.get(key)
            if value is None:
                continue
            sheet.cell(row=row_idx, column=column_idx).value = value
            sheet.cell(row=row_idx, column=column_idx).number_format = number_format


def _write_admm_diagnostics_to_excel(workbook, diagnostics):
    sheet = workbook.create_sheet('ADMM Convergence')
    columns = [
        ('outer_iteration', 'Planning Iteration', '0'),
        ('cycle', 'ADMM Cycle', '0'),
        ('local_solves_ok', 'Local Solves Successful', 'General'),
        ('primal_v', 'Primal V Residual', '0.000000'),
        ('primal_v_tolerance', 'Primal V Tolerance', '0.000000'),
        ('primal_pf', 'Primal PF Residual', '0.000000'),
        ('primal_pf_tolerance', 'Primal PF Tolerance', '0.000000'),
        ('primal_ess', 'Primal ESS Residual', '0.000000'),
        ('primal_ess_tolerance', 'Primal ESS Tolerance', '0.000000'),
        ('primal_v_ratio', 'Primal V / Tolerance', '0.000'),
        ('primal_pf_ratio', 'Primal PF / Tolerance', '0.000'),
        ('primal_ess_ratio', 'Primal ESS / Tolerance', '0.000'),
        ('dual_v', 'Dual V Residual', '0.000000'),
        ('dual_v_tolerance', 'Dual V Tolerance', '0.000000'),
        ('dual_pf', 'Dual PF Residual', '0.000000'),
        ('dual_pf_tolerance', 'Dual PF Tolerance', '0.000000'),
        ('dual_ess', 'Dual ESS Residual', '0.000000'),
        ('dual_ess_tolerance', 'Dual ESS Tolerance', '0.000000'),
        ('dual_v_ratio', 'Dual V / Tolerance', '0.000'),
        ('dual_pf_ratio', 'Dual PF / Tolerance', '0.000'),
        ('dual_ess_ratio', 'Dual ESS / Tolerance', '0.000'),
        ('recourse', 'Economic Recourse, [NPV m.u.]', '0.000000'),
        ('objective_change_abs', 'Absolute Recourse Change, [NPV m.u.]', '0.000000'),
        ('objective_change_rel', 'Relative Recourse Change, [%]', '0.00%'),
        ('objective_tolerance', 'Applied Recourse Tolerance, [NPV m.u.]', '0.000000'),
        ('objective_absolute_tolerance', 'Absolute Recourse Tolerance, [NPV m.u.]', '0.000000'),
        ('objective_relative_tolerance', 'Relative Recourse Tolerance, [%]', '0.00%'),
        ('residual_convergence', 'Residuals Converged', 'General'),
        ('objective_convergence', 'Recourse Converged', 'General'),
        ('cycle_convergence', 'Cycle Converged', 'General'),
        ('consecutive_converged_cycles', 'Consecutive Converged Cycles', '0'),
        ('required_consecutive_cycles', 'Required Consecutive Cycles', '0'),
        ('rho_v_before', 'Mean Rho V Before', '0.000000'),
        ('rho_v_after', 'Mean Rho V After', '0.000000'),
        ('rho_v_action', 'Rho V Action', 'General'),
        ('rho_pf_before', 'Mean Rho PF Before', '0.000000'),
        ('rho_pf_after', 'Mean Rho PF After', '0.000000'),
        ('rho_pf_action', 'Rho PF Action', 'General'),
        ('rho_ess_before', 'Mean Rho ESS Before', '0.000000'),
        ('rho_ess_after', 'Mean Rho ESS After', '0.000000'),
        ('rho_ess_action', 'Rho ESS Action', 'General'),
    ]

    for column_idx, (_, label, _) in enumerate(columns, start=1):
        sheet.cell(row=1, column=column_idx).value = label
    for row_idx, diagnostic in enumerate(diagnostics, start=2):
        for column_idx, (key, _, number_format) in enumerate(columns, start=1):
            value = diagnostic.get(key)
            if value is None:
                continue
            sheet.cell(row=row_idx, column=column_idx).value = value
            sheet.cell(row=row_idx, column=column_idx).number_format = number_format


# ======================================================================================================================
#  RESULTS OPERATIONAL PLANNING - write functions
# ======================================================================================================================
def _write_operational_planning_results_to_excel(planning_problem, results, primal_evolution=list(),
                                                 admm_diagnostics=list(), shared_ess_capacity=dict(),
                                                 filename='operation_planning', execution_time=float()):

    wb = Workbook()

    _write_operational_planning_main_info_to_excel(planning_problem, wb, results, execution_time=execution_time)
    _write_operational_planning_main_info_to_excel_detailed(planning_problem, wb, results['summary_detail'])
    _write_shared_ess_specifications(wb, planning_problem.shared_ess_data)
    if shared_ess_capacity:
        planning_problem.shared_ess_data.write_ess_capacity_results_to_excel(wb, shared_ess_capacity)
    _write_operational_planning_market_data_to_excel(planning_problem, wb)

    if primal_evolution:
        _write_objective_function_evolution_to_excel(wb, primal_evolution)
    if admm_diagnostics:
        _write_admm_diagnostics_to_excel(wb, admm_diagnostics)

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
    _write_network_branch_loading_results_to_excel(planning_problem, wb, results)
    _write_network_branch_power_flow_results_to_excel(planning_problem, wb, results)
    _write_network_energy_storages_results_to_excel(planning_problem, wb, results)
    _write_relaxation_slacks_results_to_excel(planning_problem, wb, results)
    planning_problem.shared_ess_data.write_relaxation_slacks_results_to_excel(wb, results['esso'])

    # Save results
    results_filename = os.path.join(planning_problem.results_dir, f'{filename}.xlsx')
    try:
        wb.save(results_filename)
        print('[INFO] Operational Planning Results written to {}.'.format(results_filename))
    except:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = os.path.join(planning_problem.results_dir, f"{filename.replace('.xlsx', '')}_{current_time}.xlsx")
        print(f"[WARNING] Results saved to file {backup_filename}.xlsx")
        wb.save(backup_filename)


def _write_operational_planning_results_hierarchical_to_excel(planning_problem, results, filename='operation_planning_results_hierarchical', execution_time=float()):
    _write_operational_planning_results_no_coordination_to_excel(planning_problem, results, filename=filename, execution_time=execution_time)


def _write_operational_planning_results_no_coordination_to_excel(planning_problem, results, filename='operation_planning_results_no_coordination', execution_time=float()):

    wb = Workbook()

    _write_operational_planning_main_info_to_excel(planning_problem, wb, results, execution_time=execution_time)
    _write_operational_planning_main_info_to_excel_detailed(planning_problem, wb, results['summary_detail'])
    _write_operational_planning_market_data_to_excel(planning_problem, wb)

    #  TSO and DSOs' results
    _write_network_voltage_results_to_excel(planning_problem, wb, results)
    _write_network_consumption_results_to_excel(planning_problem, wb, results)
    _write_network_generation_results_to_excel(planning_problem, wb, results)
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'losses')
    _write_network_branch_results_to_excel(planning_problem, wb, results, 'ratio')
    _write_network_branch_loading_results_to_excel(planning_problem, wb, results)
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


def _write_operational_planning_main_info_to_excel(planning_problem, workbook, results, execution_time=float()):

    sheet = workbook.worksheets[0]
    sheet.title = 'Main Info'

    # Write Header
    col_idx = 4
    line_idx = 1
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

    if execution_time:
        line_idx += 1
        sheet.cell(row=line_idx, column=1).value = '-'
        sheet.cell(row=line_idx, column=2).value = '-'
        sheet.cell(row=line_idx, column=3).value = 'Execution time, [s]'
        sheet.cell(row=line_idx, column=4).value = execution_time
        sheet.cell(row=line_idx, column=4).number_format = '0.00'


def _write_operational_planning_main_info_per_operator(network, sheet, operator_type, line_idx, results, tn_node_id='-'):

    decimal_style = '0.00'

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1

    # - Objective
    sheet.cell(row=line_idx, column=col_idx).value = 'Objective function value, [N/A]'
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
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_load']['p']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Load, [MVArh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_load']['q']
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
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['flex_used']['p']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

        line_idx += 1
        col_idx = 1
        sheet.cell(row=line_idx, column=col_idx).value = operator_type
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = 'Flexibility used, [MVArh]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['flex_used']['q']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

        if network.params.obj_type == OBJ_MIN_COST:
            line_idx += 1
            col_idx = 1
            sheet.cell(row=line_idx, column=col_idx).value = operator_type
            col_idx += 1
            sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
            col_idx += 1
            sheet.cell(row=line_idx, column=col_idx).value = 'Flexibility cost, [€]'
            col_idx += 1
            for year in results:
                for day in results[year]:
                    sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['flex_cost']
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
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['load_curt']['p']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

        line_idx += 1
        col_idx = 1
        sheet.cell(row=line_idx, column=col_idx).value = operator_type
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = 'Load curtailed, [MVArh]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['load_curt']['q']
                sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
                col_idx += 1

        if network.params.obj_type == OBJ_MIN_COST:
            line_idx += 1
            col_idx = 1
            sheet.cell(row=line_idx, column=col_idx).value = operator_type
            col_idx += 1
            sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
            col_idx += 1
            sheet.cell(row=line_idx, column=col_idx).value = 'Load curtailment cost, [€]'
            col_idx += 1
            for year in results:
                for day in results[year]:
                    sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['load_curt_cost']
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
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_gen']['p']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Generation, [MVArh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_gen']['q']
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
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_conventional_gen']['p']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Conventional Generation, [MVArh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_conventional_gen']['q']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    if network.params.obj_type == OBJ_MIN_COST:
        line_idx += 1
        col_idx = 1
        sheet.cell(row=line_idx, column=col_idx).value = operator_type
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
        col_idx += 1
        sheet.cell(row=line_idx, column=col_idx).value = 'Conventional generation cost, [€]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['gen_cost']
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
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_renewable_gen']['p']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Renewable generation, [MVArh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_renewable_gen']['q']
            sheet.cell(row=line_idx, column=col_idx).number_format = decimal_style
            col_idx += 1

    line_idx += 1
    col_idx = 1
    sheet.cell(row=line_idx, column=col_idx).value = operator_type
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = tn_node_id
    col_idx += 1
    sheet.cell(row=line_idx, column=col_idx).value = 'Renewable generation, [MVAh]'
    col_idx += 1
    for year in results:
        for day in results[year]:
            sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['total_renewable_gen']['s']
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
        sheet.cell(row=line_idx, column=col_idx).value = 'Renewable generation curtailed, [MVAh]'
        col_idx += 1
        for year in results:
            for day in results[year]:
                sheet.cell(row=line_idx, column=col_idx).value = results[year][day]['gen_curt']['s']
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


def _write_operational_planning_main_info_to_excel_detailed(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Main Info, Detailed')

    # Write Header -- Year
    line_idx = 1
    sheet.cell(row=line_idx, column=1).value = 'Operator'
    sheet.cell(row=line_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=line_idx, column=3).value = 'Year'
    sheet.cell(row=line_idx, column=4).value = 'Day'
    sheet.cell(row=line_idx, column=5).value = 'Market Scenario'
    sheet.cell(row=line_idx, column=6).value = 'Operation Scenario'
    sheet.cell(row=line_idx, column=7).value = 'Probability, [%]'
    sheet.cell(row=line_idx, column=8).value = 'OF Value'
    sheet.cell(row=line_idx, column=9).value = 'Load, [MWh]'
    sheet.cell(row=line_idx, column=10).value = 'Load, [MVArh]'
    sheet.cell(row=line_idx, column=11).value = 'Flexibility used, [MWh]'
    sheet.cell(row=line_idx, column=12).value = 'Flexibility used, [MVArh]'
    sheet.cell(row=line_idx, column=13).value = 'Flexibility Cost, [€]'
    sheet.cell(row=line_idx, column=14).value = 'Generation, [MWh]'
    sheet.cell(row=line_idx, column=15).value = 'Generation, [MVArh]'
    sheet.cell(row=line_idx, column=16).value = 'Conventional Generation, [MWh]'
    sheet.cell(row=line_idx, column=17).value = 'Conventional Generation, [MVArh]'
    sheet.cell(row=line_idx, column=18).value = 'Conventional Generation Cost, [€]'
    sheet.cell(row=line_idx, column=19).value = 'Renewable Generation, [MWh]'
    sheet.cell(row=line_idx, column=20).value = 'Renewable Generation, [MVArh]'
    sheet.cell(row=line_idx, column=21).value = 'Renewable Generation, [MVAh]'
    sheet.cell(row=line_idx, column=22).value = 'Renewable Generation Curtailed, [MVAh]'
    sheet.cell(row=line_idx, column=23).value = 'Losses, [MWh]'

    # TSO
    line_idx += 1
    line_idx = _write_operational_planning_main_info_per_operator_detailed(planning_problem.transmission_network, sheet, 'TSO', line_idx, results['tso'])

    # DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]
        distribution_network = planning_problem.distribution_networks[tn_node_id]
        line_idx = _write_operational_planning_main_info_per_operator_detailed(distribution_network, sheet, 'DSO', line_idx, dso_results, tn_node_id=tn_node_id)


def _write_operational_planning_main_info_per_operator_detailed(network, sheet, operator_type, line_idx, results, tn_node_id='-'):

    decimal_style = '0.00'
    percent_style = '0.00%'

    sheet.cell(row=line_idx, column=1).value = operator_type
    for year in results:
        for day in results[year]:
            for s_m in results[year][day]['scenarios']:
                for s_o in results[year][day]['scenarios'][s_m]:

                    sheet.cell(row=line_idx, column=1).value = operator_type
                    sheet.cell(row=line_idx, column=2).value = tn_node_id
                    sheet.cell(row=line_idx, column=3).value = int(year)
                    sheet.cell(row=line_idx, column=4).value = day
                    sheet.cell(row=line_idx, column=5).value = s_m
                    sheet.cell(row=line_idx, column=6).value = s_o

                    # Probability, [%]
                    sheet.cell(row=line_idx, column=7).value = results[year][day]['scenarios'][s_m][s_o]['probability']
                    sheet.cell(row=line_idx, column=7).number_format = percent_style

                    # OF
                    sheet.cell(row=line_idx, column=8).value = results[year][day]['scenarios'][s_m][s_o]['obj']
                    sheet.cell(row=line_idx, column=8).number_format = decimal_style

                    # Load
                    sheet.cell(row=line_idx, column=9).value = results[year][day]['scenarios'][s_m][s_o]['load']['p']
                    sheet.cell(row=line_idx, column=9).number_format = decimal_style
                    sheet.cell(row=line_idx, column=10).value = results[year][day]['scenarios'][s_m][s_o]['load']['q']
                    sheet.cell(row=line_idx, column=10).number_format = decimal_style

                    # Flexibility, [MWh]
                    sheet.cell(row=line_idx, column=11).value = results[year][day]['scenarios'][s_m][s_o]['flexibility']['p']
                    sheet.cell(row=line_idx, column=11).number_format = decimal_style
                    sheet.cell(row=line_idx, column=12).value = results[year][day]['scenarios'][s_m][s_o]['flexibility']['q']
                    sheet.cell(row=line_idx, column=12).number_format = decimal_style

                    # Flexibility Cost, [€]
                    sheet.cell(row=line_idx, column=13).value = results[year][day]['scenarios'][s_m][s_o]['cost_flexibility']
                    sheet.cell(row=line_idx, column=13).number_format = decimal_style

                    # Generation
                    sheet.cell(row=line_idx, column=14).value = results[year][day]['scenarios'][s_m][s_o]['generation']['p']
                    sheet.cell(row=line_idx, column=14).number_format = decimal_style
                    sheet.cell(row=line_idx, column=15).value = results[year][day]['scenarios'][s_m][s_o]['generation']['q']
                    sheet.cell(row=line_idx, column=15).number_format = decimal_style

                    # Conventional Generation
                    sheet.cell(row=line_idx, column=16).value = results[year][day]['scenarios'][s_m][s_o]['generation_conventional']['p']
                    sheet.cell(row=line_idx, column=16).number_format = decimal_style
                    sheet.cell(row=line_idx, column=17).value = results[year][day]['scenarios'][s_m][s_o]['generation_conventional']['q']
                    sheet.cell(row=line_idx, column=17).number_format = decimal_style

                    # Conventional Generation Cost
                    sheet.cell(row=line_idx, column=18).value = results[year][day]['scenarios'][s_m][s_o]['generation_conventional_cost']
                    sheet.cell(row=line_idx, column=18).number_format = decimal_style

                    # Renewable Generation
                    sheet.cell(row=line_idx, column=19).value = results[year][day]['scenarios'][s_m][s_o]['generation_renewable']['p']
                    sheet.cell(row=line_idx, column=19).number_format = decimal_style
                    sheet.cell(row=line_idx, column=20).value = results[year][day]['scenarios'][s_m][s_o]['generation_renewable']['q']
                    sheet.cell(row=line_idx, column=20).number_format = decimal_style
                    sheet.cell(row=line_idx, column=21).value = results[year][day]['scenarios'][s_m][s_o]['generation_renewable']['s']
                    sheet.cell(row=line_idx, column=21).number_format = decimal_style
                    sheet.cell(row=line_idx, column=22).value = results[year][day]['scenarios'][s_m][s_o]['generation_renewable_curtailed']['s']
                    sheet.cell(row=line_idx, column=22).number_format = decimal_style

                    # Losses
                    sheet.cell(row=line_idx, column=23).value = results[year][day]['scenarios'][s_m][s_o]['losses']
                    sheet.cell(row=line_idx, column=23).number_format = decimal_style

                    line_idx += 1

    return line_idx


def _write_shared_ess_specifications(workbook, shared_ess_info):

    sheet = workbook.create_sheet('SharedESS Specifications')

    decimal_style = '0.000'

    # Write Header
    row_idx = 1
    sheet.cell(row=row_idx, column=1).value = 'Year'
    sheet.cell(row=row_idx, column=2).value = 'Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Sinst, [MVA]'
    sheet.cell(row=row_idx, column=4).value = 'Einst, [MVAh]'

    # Write SharedESS specifications
    for year in shared_ess_info.years:
        for shared_ess in shared_ess_info.shared_energy_storages[year]:
            row_idx = row_idx + 1
            sheet.cell(row=row_idx, column=1).value = year
            sheet.cell(row=row_idx, column=2).value = shared_ess.bus
            sheet.cell(row=row_idx, column=3).value = shared_ess.s
            sheet.cell(row=row_idx, column=3).number_format = decimal_style
            sheet.cell(row=row_idx, column=4).value = shared_ess.e
            sheet.cell(row=row_idx, column=4).number_format = decimal_style


def _write_operational_planning_market_data_to_excel(planning_problem, workbook):

    sheet = workbook.create_sheet('MarketData')

    row_idx = 1
    decimal_style = '0.00'

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Year'
    sheet.cell(row=row_idx, column=2).value = 'Day'
    sheet.cell(row=row_idx, column=3).value = 'Quantity'
    sheet.cell(row=row_idx, column=4).value = 'Market Scenario'
    for p in range(planning_problem.num_instants):
        sheet.cell(row=row_idx, column=p + 5).value = p
    row_idx = row_idx + 1


    for year in planning_problem.years:
        for day in planning_problem.days:

            cost_energy = planning_problem.cost_energy_p[year][day]
            cost_flexibility = planning_problem.cost_flex[year][day]

            # - Energy
            for s_m in range(planning_problem.num_market_scenarios):
                sheet.cell(row=row_idx, column=1).value = int(year)
                sheet.cell(row=row_idx, column=2).value = day
                sheet.cell(row=row_idx, column=3).value = 'Energy'
                sheet.cell(row=row_idx, column=4).value = s_m
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 5).value = cost_energy[s_m][p]
                    sheet.cell(row=row_idx, column=p + 5).number_format = decimal_style
                row_idx += 1

            # - Flexibility
            for s_m in range(planning_problem.num_market_scenarios):
                sheet.cell(row=row_idx, column=1).value = int(year)
                sheet.cell(row=row_idx, column=2).value = day
                sheet.cell(row=row_idx, column=3).value = 'Flexibility'
                sheet.cell(row=row_idx, column=4).value = s_m
                for p in range(planning_problem.num_instants):
                    sheet.cell(row=row_idx, column=p + 5).value = cost_flexibility[s_m][p]
                    sheet.cell(row=row_idx, column=p + 5).number_format = decimal_style
                row_idx += 1


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

    sheet = workbook.create_sheet('SharedESS')

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


def _write_network_voltage_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Voltage')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
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
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Load ID'
    sheet.cell(row=row_idx, column=4).value = 'Node ID'
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
            expected_pc_flex = dict()
            expected_pc_curt = dict()
            expected_pnet = dict()
            expected_qc = dict()
            expected_qc_flex = dict()
            expected_qc_curt = dict()
            expected_qnet = dict()
            for load in network[year][day].loads:
                expected_pc[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pc_flex[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pc_curt[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pnet[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qc[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qc_flex[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qc_curt[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qnet[load.load_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for load in network[year][day].loads:

                        # - Active Power
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = load.load_id
                        sheet.cell(row=row_idx, column=4).value = load.bus
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

                            # - Flexibility, Pc
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.load_id
                            sheet.cell(row=row_idx, column=4).value = load.bus
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Pc_flex, [MW]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                pc_flex = results[year][day]['scenarios'][s_m][s_o]['consumption']['pc_flex'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = pc_flex
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_pc_flex[load.load_id][p] += pc_flex * omega_m * omega_s
                            row_idx = row_idx + 1

                        if params.l_curt:

                            # - Active power curtailment
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.load_id
                            sheet.cell(row=row_idx, column=4).value = load.bus
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
                            sheet.cell(row=row_idx, column=3).value = load.load_id
                            sheet.cell(row=row_idx, column=4).value = load.bus
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
                        sheet.cell(row=row_idx, column=3).value = load.load_id
                        sheet.cell(row=row_idx, column=4).value = load.bus
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

                        if params.fl_reg:

                            # - Flexibility, Qc
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.load_id
                            sheet.cell(row=row_idx, column=4).value = load.bus
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Qc_flex, [MVAr]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                qc_flex = results[year][day]['scenarios'][s_m][s_o]['consumption']['qc_flex'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = qc_flex
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_qc_flex[load.load_id][p] += qc_flex * omega_m * omega_s
                            row_idx = row_idx + 1

                        if params.l_curt:

                            # - Reactive power curtailment
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.load_id
                            sheet.cell(row=row_idx, column=4).value = load.bus
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Qc_curt, [MVAr]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                qc_curt = results[year][day]['scenarios'][s_m][s_o]['consumption']['qc_curt'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = qc_curt
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                if qc_curt >= SMALL_TOLERANCE:
                                    sheet.cell(row=row_idx, column=p + 10).fill = violation_fill
                                expected_qc_curt[load.load_id][p] += qc_curt * omega_m * omega_s
                            row_idx = row_idx + 1

                        if params.fl_reg or params.l_curt:

                            # - Reactive power net consumption
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = load.load_id
                            sheet.cell(row=row_idx, column=4).value = load.bus
                            sheet.cell(row=row_idx, column=5).value = int(year)
                            sheet.cell(row=row_idx, column=6).value = day
                            sheet.cell(row=row_idx, column=7).value = 'Qc_net, [MVAr]'
                            sheet.cell(row=row_idx, column=8).value = s_m
                            sheet.cell(row=row_idx, column=9).value = s_o
                            for p in range(network[year][day].num_instants):
                                q_net = results[year][day]['scenarios'][s_m][s_o]['consumption']['qc_net'][load.load_id][p]
                                sheet.cell(row=row_idx, column=p + 10).value = q_net
                                sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                                expected_qnet[load.load_id][p] += q_net * omega_m * omega_s
                            row_idx = row_idx + 1

            for load in network[year][day].loads:

                # - Active Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = load.load_id
                sheet.cell(row=row_idx, column=4).value = load.bus
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

                    # - Flexibility, active power
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.load_id
                    sheet.cell(row=row_idx, column=4).value = load.bus
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Pc_flex, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_pc_flex[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

                if params.l_curt:

                    # - Load curtailment (active power)
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.load_id
                    sheet.cell(row=row_idx, column=4).value = load.bus
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
                    sheet.cell(row=row_idx, column=3).value = load.load_id
                    sheet.cell(row=row_idx, column=4).value = load.bus
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
                sheet.cell(row=row_idx, column=3).value = load.load_id
                sheet.cell(row=row_idx, column=4).value = load.bus
                sheet.cell(row=row_idx, column=5).value = int(year)
                sheet.cell(row=row_idx, column=6).value = day
                sheet.cell(row=row_idx, column=7).value = 'Qc, [MVAr]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=9).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 10).value = expected_qc[load.load_id][p]
                    sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                row_idx = row_idx + 1

                if params.fl_reg:

                    # - Flexibility, reactive power
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.load_id
                    sheet.cell(row=row_idx, column=4).value = load.bus
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Qc_flex, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_qc_flex[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

                if params.l_curt:

                    # - Load curtailment (reactive power)
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.load_id
                    sheet.cell(row=row_idx, column=4).value = load.bus
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Qc_curt, [MW]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_qc_curt[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                        if expected_pc_curt[load.load_id][p] >= SMALL_TOLERANCE:
                            sheet.cell(row=row_idx, column=p + 9).fill = violation_fill
                    row_idx = row_idx + 1

                if params.fl_reg or params.l_curt:

                    # - Reactive power net consumption
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = load.load_id
                    sheet.cell(row=row_idx, column=4).value = load.bus
                    sheet.cell(row=row_idx, column=5).value = int(year)
                    sheet.cell(row=row_idx, column=6).value = day
                    sheet.cell(row=row_idx, column=7).value = 'Qc_net, [MVAr]'
                    sheet.cell(row=row_idx, column=8).value = 'Expected'
                    sheet.cell(row=row_idx, column=9).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 10).value = expected_qnet[load.load_id][p]
                        sheet.cell(row=row_idx, column=p + 10).number_format = decimal_style
                    row_idx = row_idx + 1

    return row_idx


def _write_network_generation_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Generation')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Generator ID'
    sheet.cell(row=row_idx, column=4).value = 'Node ID'
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
            expected_pg_net = dict()
            expected_qg = dict()
            expected_qg_net = dict()
            expected_sg = dict()
            expected_sg_curt = dict()
            expected_sg_net = dict()
            for generator in network[year][day].generators:
                expected_pg[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qg[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_sg[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_pg_net[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_qg_net[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_sg_curt[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]
                expected_sg_net[generator.gen_id] = [0.0 for _ in range(network[year][day].num_instants)]

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
                        sheet.cell(row=row_idx, column=3).value = gen_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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

                        # Active Power net
                        if generator.is_curtaillable() and params.rg_curt:

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = gen_id
                            sheet.cell(row=row_idx, column=4).value = node_id
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
                        sheet.cell(row=row_idx, column=3).value = gen_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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

                        # Reactive Power net
                        if generator.is_curtaillable() and params.rg_curt:

                            # Reactive Power net
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = gen_id
                            sheet.cell(row=row_idx, column=4).value = node_id
                            sheet.cell(row=row_idx, column=5).value = gen_type
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = 'Qg_net, [MVAr]'
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                qg_net = results[year][day]['scenarios'][s_m][s_o]['generation']['qg_net'][gen_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = qg_net
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                expected_qg_net[gen_id][p] += qg_net * omega_m * omega_s
                            row_idx = row_idx + 1

                        # Apparent Power
                        if generator.is_curtaillable() and params.rg_curt:

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = gen_id
                            sheet.cell(row=row_idx, column=4).value = node_id
                            sheet.cell(row=row_idx, column=5).value = gen_type
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = 'Sg, [MVA]'
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                sg = results[year][day]['scenarios'][s_m][s_o]['generation']['sg'][gen_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = sg
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                expected_sg[gen_id][p] += sg * omega_m * omega_s
                            row_idx = row_idx + 1

                            # Apparent Power curtailment
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = gen_id
                            sheet.cell(row=row_idx, column=4).value = node_id
                            sheet.cell(row=row_idx, column=5).value = gen_type
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = 'Sg_curt, [MVA]'
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                sg_curt = results[year][day]['scenarios'][s_m][s_o]['generation']['sg_curt'][gen_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = sg_curt
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                if not isclose(sg_curt, 0.00, abs_tol=VIOLATION_TOLERANCE):
                                    sheet.cell(row=row_idx, column=p + 11).fill = violation_fill
                                expected_sg_curt[gen_id][p] += sg_curt * omega_m * omega_s
                            row_idx = row_idx + 1

                            # Apparent Power Net
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = gen_id
                            sheet.cell(row=row_idx, column=4).value = node_id
                            sheet.cell(row=row_idx, column=5).value = gen_type
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = 'Sg_net, [MVA]'
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                sg_net = results[year][day]['scenarios'][s_m][s_o]['generation']['sg_net'][gen_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = sg_net
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                expected_sg_net[gen_id][p] += sg_net * omega_m * omega_s
                            row_idx = row_idx + 1

            for generator in network[year][day].generators:

                node_id = generator.bus
                gen_id = generator.gen_id
                gen_type = network[year][day].get_gen_type(gen_id)

                # Active Power
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = gen_id
                sheet.cell(row=row_idx, column=4).value = node_id
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

                # Active Power Net
                if generator.is_curtaillable() and params.rg_curt:

                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = gen_id
                    sheet.cell(row=row_idx, column=4).value = node_id
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
                sheet.cell(row=row_idx, column=3).value = gen_id
                sheet.cell(row=row_idx, column=4).value = node_id
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

                # Reactive Power net
                if generator.is_curtaillable() and params.rg_curt:

                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = gen_id
                    sheet.cell(row=row_idx, column=4).value = node_id
                    sheet.cell(row=row_idx, column=5).value = gen_type
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = 'Qg_net, [MVAr]'
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_qg_net[gen_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                    row_idx = row_idx + 1

                # Apparent Power
                if generator.is_curtaillable() and params.rg_curt:

                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = gen_id
                    sheet.cell(row=row_idx, column=4).value = node_id
                    sheet.cell(row=row_idx, column=5).value = gen_type
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = 'Sg, [MVA]'
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_sg[gen_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                    row_idx = row_idx + 1

                    # Apparent Power curtailment
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = gen_id
                    sheet.cell(row=row_idx, column=4).value = node_id
                    sheet.cell(row=row_idx, column=5).value = gen_type
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = 'Sg_curt, [MVA]'
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_sg_curt[gen_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                        if not isclose(expected_sg_curt[gen_id][p], 0.00, abs_tol=VIOLATION_TOLERANCE):
                            sheet.cell(row=row_idx, column=p + 11).fill = violation_fill
                    row_idx = row_idx + 1

                    # Apparent Power Net
                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = gen_id
                    sheet.cell(row=row_idx, column=4).value = node_id
                    sheet.cell(row=row_idx, column=5).value = gen_type
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = 'Sg_net, [MVA]'
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_sg_net[gen_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                    row_idx = row_idx + 1

    return row_idx


def _write_network_branch_results_to_excel(planning_problem, workbook, results, result_type):

    sheet_name = str()
    if result_type == 'losses':
        sheet_name = 'Branch Losses'
    elif result_type == 'ratio':
        sheet_name = 'Transformer Ratio'

    sheet = workbook.create_sheet(sheet_name)

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Branch ID'
    sheet.cell(row=row_idx, column=4).value = 'From Node ID'
    sheet.cell(row=row_idx, column=5).value = 'To Node ID'
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
    row_idx = _write_network_branch_results_per_operator(transmission_network, sheet, 'TSO', row_idx, results['tso']['results'], result_type)

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        row_idx = _write_network_branch_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, result_type, tn_node_id=tn_node_id)


def _write_network_branch_results_per_operator(network, sheet, operator_type, row_idx, results, result_type, tn_node_id='-'):

    decimal_style = '0.00'

    aux_string = str()
    if result_type == 'losses':
        aux_string = 'P, [MW]'
    elif result_type == 'ratio':
        aux_string = 'Ratio'

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
                            sheet.cell(row=row_idx, column=3).value = branch.branch_id
                            sheet.cell(row=row_idx, column=4).value = branch.fbus
                            sheet.cell(row=row_idx, column=5).value = branch.tbus
                            sheet.cell(row=row_idx, column=6).value = int(year)
                            sheet.cell(row=row_idx, column=7).value = day
                            sheet.cell(row=row_idx, column=8).value = aux_string
                            sheet.cell(row=row_idx, column=9).value = s_m
                            sheet.cell(row=row_idx, column=10).value = s_o
                            for p in range(network[year][day].num_instants):
                                value = results[year][day]['scenarios'][s_m][s_o]['branches'][result_type][branch_id][p]
                                sheet.cell(row=row_idx, column=p + 11).value = value
                                sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                                expected_values[branch_id][p] += value * omega_m * omega_s
                            row_idx = row_idx + 1

            for branch in network[year][day].branches:
                branch_id = branch.branch_id
                if not (result_type == 'ratio' and not branch.is_transformer):

                    sheet.cell(row=row_idx, column=1).value = operator_type
                    sheet.cell(row=row_idx, column=2).value = tn_node_id
                    sheet.cell(row=row_idx, column=3).value = branch.branch_id
                    sheet.cell(row=row_idx, column=4).value = branch.fbus
                    sheet.cell(row=row_idx, column=5).value = branch.tbus
                    sheet.cell(row=row_idx, column=6).value = int(year)
                    sheet.cell(row=row_idx, column=7).value = day
                    sheet.cell(row=row_idx, column=8).value = aux_string
                    sheet.cell(row=row_idx, column=9).value = 'Expected'
                    sheet.cell(row=row_idx, column=10).value = '-'
                    for p in range(network[year][day].num_instants):
                        sheet.cell(row=row_idx, column=p + 11).value = expected_values[branch_id][p]
                        sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                    row_idx = row_idx + 1

    return row_idx


def _write_network_branch_loading_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Branch Loading')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Branch ID'
    sheet.cell(row=row_idx, column=4).value = 'From Node ID'
    sheet.cell(row=row_idx, column=5).value = 'To Node ID'
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
    row_idx = _write_network_branch_loading_results_per_operator(transmission_network, sheet, 'TSO', row_idx, results['tso']['results'])

    # Write results -- DSOs
    for tn_node_id in results['dso']:
        dso_results = results['dso'][tn_node_id]['results']
        distribution_network = planning_problem.distribution_networks[tn_node_id].network
        row_idx = _write_network_branch_loading_results_per_operator(distribution_network, sheet, 'DSO', row_idx, dso_results, tn_node_id=tn_node_id)


def _write_network_branch_loading_results_per_operator(network, sheet, operator_type, row_idx, results, tn_node_id='-'):

    perc_style = '0.00%'
    violation_fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')

    for year in results:
        for day in results[year]:

            expected_values = {'flow_ij': {}}
            for branch in network[year][day].branches:
                expected_values['flow_ij'][branch.branch_id] = [0.0 for _ in range(network[year][day].num_instants)]

            for s_m in results[year][day]['scenarios']:
                omega_m = network[year][day].prob_market_scenarios[s_m]
                for s_o in results[year][day]['scenarios'][s_m]:
                    omega_s = network[year][day].prob_operation_scenarios[s_o]
                    for branch in network[year][day].branches:

                        # flow ij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Flow_ij, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['branch_flow']['flow_ij_perc'][branch.branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                            if value > 1.00 + VIOLATION_TOLERANCE:
                                sheet.cell(row=row_idx, column=p + 11).fill = violation_fill
                            expected_values['flow_ij'][branch.branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

            for branch in network[year][day].branches:

                # flow ij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Flow_ij, [%]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    value = expected_values['flow_ij'][branch.branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).value = value
                    sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                    if value > 1.00 + VIOLATION_TOLERANCE:
                        sheet.cell(row=row_idx, column=p + 11).fill = violation_fill
                row_idx = row_idx + 1

    return row_idx


def _write_network_branch_power_flow_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Power Flows')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Branch ID'
    sheet.cell(row=row_idx, column=4).value = 'From Node ID'
    sheet.cell(row=row_idx, column=5).value = 'To Node ID'
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
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_values['pij'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Pij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'P, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                        row_idx = row_idx + 1

                        # Pji, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = branch.fbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'P, [MW]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_values['pji'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Pji, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = branch.fbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'P, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                        row_idx = row_idx + 1

                        # Qij, [MVAr]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_values['qij'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Qij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Q, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                        row_idx = row_idx + 1

                        # Qji, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = branch.fbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Q, [MVAr]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_values['qji'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Qji, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = branch.fbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'Q, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                        row_idx = row_idx + 1

                        # Sij, [MVA]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_values['sij'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Sij, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.fbus
                        sheet.cell(row=row_idx, column=5).value = branch.tbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'S, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                        row_idx = row_idx + 1

                        # Sji, [MW]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = branch.fbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'S, [MVA]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][branch_id][p]
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                            expected_values['sji'][branch_id][p] += value * omega_m * omega_s
                        row_idx = row_idx + 1

                        # Sji, [%]
                        sheet.cell(row=row_idx, column=1).value = operator_type
                        sheet.cell(row=row_idx, column=2).value = tn_node_id
                        sheet.cell(row=row_idx, column=3).value = branch.branch_id
                        sheet.cell(row=row_idx, column=4).value = branch.tbus
                        sheet.cell(row=row_idx, column=5).value = branch.fbus
                        sheet.cell(row=row_idx, column=6).value = int(year)
                        sheet.cell(row=row_idx, column=7).value = day
                        sheet.cell(row=row_idx, column=8).value = 'S, [%]'
                        sheet.cell(row=row_idx, column=9).value = s_m
                        sheet.cell(row=row_idx, column=10).value = s_o
                        for p in range(network[year][day].num_instants):
                            value = abs(results[year][day]['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][branch_id][p] / rating)
                            sheet.cell(row=row_idx, column=p + 11).value = value
                            sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                        row_idx = row_idx + 1

            for branch in network[year][day].branches:

                branch_id = branch.branch_id
                rating = branch.rate
                if rating == 0.0:
                    rating = BRANCH_UNKNOWN_RATING

                # Pij, [MW]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_values['pij'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Pij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'P, [%]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = abs(expected_values['pij'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                row_idx = row_idx + 1

                # Pji, [MW]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = branch.fbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'P, [MW]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_values['pji'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Pji, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = branch.fbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'P, [%]'
                sheet.cell(row=row_idx, column=8).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = abs(expected_values['pji'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                row_idx = row_idx + 1

                # Qij, [MVAr]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_values['qij'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Qij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Q, [%]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = abs(expected_values['qij'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                row_idx = row_idx + 1

                # Qji, [MVAr]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = branch.fbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Q, [MVAr]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_values['qji'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Qji, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = branch.fbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'Q, [%]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = abs(expected_values['qji'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Sij, [MVA]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_values['sij'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Sij, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.fbus
                sheet.cell(row=row_idx, column=5).value = branch.tbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'S, [%]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = abs(expected_values['sij'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                row_idx = row_idx + 1

                # Sji, [MVA]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = branch.fbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'S, [MVA]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = expected_values['sji'][branch_id][p]
                    sheet.cell(row=row_idx, column=p + 11).number_format = decimal_style
                row_idx = row_idx + 1

                # Sji, [%]
                sheet.cell(row=row_idx, column=1).value = operator_type
                sheet.cell(row=row_idx, column=2).value = tn_node_id
                sheet.cell(row=row_idx, column=3).value = branch.branch_id
                sheet.cell(row=row_idx, column=4).value = branch.tbus
                sheet.cell(row=row_idx, column=5).value = branch.fbus
                sheet.cell(row=row_idx, column=6).value = int(year)
                sheet.cell(row=row_idx, column=7).value = day
                sheet.cell(row=row_idx, column=8).value = 'S, [%]'
                sheet.cell(row=row_idx, column=9).value = 'Expected'
                sheet.cell(row=row_idx, column=10).value = '-'
                for p in range(network[year][day].num_instants):
                    sheet.cell(row=row_idx, column=p + 11).value = abs(expected_values['sji'][branch_id][p]) / rating
                    sheet.cell(row=row_idx, column=p + 11).number_format = perc_style
                row_idx = row_idx + 1

    return row_idx


def _write_network_energy_storages_results_to_excel(planning_problem, workbook, results):

    sheet = workbook.create_sheet('Energy Storage')

    row_idx = 1

    # Write Header
    sheet.cell(row=row_idx, column=1).value = 'Operator'
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'ESS ID'
    sheet.cell(row=row_idx, column=4).value = 'Node ID'
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
                        sheet.cell(row=row_idx, column=3).value = es_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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
                        sheet.cell(row=row_idx, column=3).value = es_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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
                        sheet.cell(row=row_idx, column=3).value = es_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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
                        sheet.cell(row=row_idx, column=3).value = es_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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
                        sheet.cell(row=row_idx, column=3).value = es_id
                        sheet.cell(row=row_idx, column=4).value = node_id
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
                sheet.cell(row=row_idx, column=3).value = es_id
                sheet.cell(row=row_idx, column=4).value = node_id
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
                sheet.cell(row=row_idx, column=3).value = es_id
                sheet.cell(row=row_idx, column=4).value = node_id
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
                sheet.cell(row=row_idx, column=3).value = es_id
                sheet.cell(row=row_idx, column=4).value = node_id
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
                sheet.cell(row=row_idx, column=3).value = es_id
                sheet.cell(row=row_idx, column=4).value = node_id
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
                sheet.cell(row=row_idx, column=3).value = es_id
                sheet.cell(row=row_idx, column=4).value = node_id
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
    sheet.cell(row=row_idx, column=2).value = 'ADN Node ID'
    sheet.cell(row=row_idx, column=3).value = 'Resource ID'
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
                    if params.slacks.grid_operation.voltage:
                        for node in network[year][day].nodes:

                            node_id = node.bus_i

                            # - e
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Voltage, e'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_e = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_e
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                            # - f
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Voltage, f'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_f = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['f'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_f
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # Branch flow slacks
                    if params.slacks.grid_operation.branch_flow:
                        for branch in network[year][day].branches:

                            branch_id = branch.branch_id

                            # - flow_ij
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = branch_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Flow_ij_sqr'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                iij_sqr = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['branch_flow']['flow_ij_sqr'][branch_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = iij_sqr
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # Node balance
                    for node in network[year][day].nodes:

                        node_id = node.bus_i

                        # - p
                        if params.slacks.node_balance.active_power:
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Node balance, p'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_node_balance_p = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_node_balance_p
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                        # - q
                        if params.slacks.node_balance.reactive_power:
                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Node balance, q'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                slack_node_balance_q = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q'][node_id][p]
                                sheet.cell(row=row_idx, column=p + 9).value = slack_node_balance_q
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # SharedESS
                    for shared_energy_storage in network[year][day].shared_energy_storages:

                        node_id = shared_energy_storage.bus

                        # - Day balance
                        if params.slacks.shared_ess.day_balance:

                            sheet.cell(row=row_idx, column=1).value = operator_type
                            sheet.cell(row=row_idx, column=2).value = tn_node_id
                            sheet.cell(row=row_idx, column=3).value = node_id
                            sheet.cell(row=row_idx, column=4).value = int(year)
                            sheet.cell(row=row_idx, column=5).value = day
                            sheet.cell(row=row_idx, column=6).value = 'Shared Energy Storage, soc_final'
                            sheet.cell(row=row_idx, column=7).value = s_m
                            sheet.cell(row=row_idx, column=8).value = s_o
                            for p in range(network[year][day].num_instants):
                                soc_final = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final'][node_id]
                                sheet.cell(row=row_idx, column=p + 9).value = soc_final
                                sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                            row_idx = row_idx + 1

                    # Load flexibility
                    if params.fl_reg:
                        for load in network[year][day].loads:

                            load_id = load.load_id

                            # - Day balance
                            if params.slacks.flexibility.day_balance:

                                sheet.cell(row=row_idx, column=1).value = operator_type
                                sheet.cell(row=row_idx, column=2).value = tn_node_id
                                sheet.cell(row=row_idx, column=3).value = load_id
                                sheet.cell(row=row_idx, column=4).value = int(year)
                                sheet.cell(row=row_idx, column=5).value = day
                                sheet.cell(row=row_idx, column=6).value = 'Flex. balance, p'
                                sheet.cell(row=row_idx, column=7).value = s_m
                                sheet.cell(row=row_idx, column=8).value = s_o
                                for p in range(network[year][day].num_instants):
                                    day_balance_p = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance'][load_id]['p']
                                    sheet.cell(row=row_idx, column=p + 9).value = day_balance_p
                                    sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                                row_idx = row_idx + 1

                                sheet.cell(row=row_idx, column=1).value = operator_type
                                sheet.cell(row=row_idx, column=2).value = tn_node_id
                                sheet.cell(row=row_idx, column=3).value = load_id
                                sheet.cell(row=row_idx, column=4).value = int(year)
                                sheet.cell(row=row_idx, column=5).value = day
                                sheet.cell(row=row_idx, column=6).value = 'Flex. balance, q'
                                sheet.cell(row=row_idx, column=7).value = s_m
                                sheet.cell(row=row_idx, column=8).value = s_o
                                for p in range(network[year][day].num_instants):
                                    day_balance_q = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance'][load_id]['q']
                                    sheet.cell(row=row_idx, column=p + 9).value = day_balance_q
                                    sheet.cell(row=row_idx, column=p + 9).number_format = decimal_style
                                row_idx = row_idx + 1

                    # ESS
                    if params.es_reg:

                        for energy_storage in network[year][day].energy_storages:

                            es_id = energy_storage.es_id

                            # - Day balance
                            if params.slacks.ess.day_balance:

                                sheet.cell(row=row_idx, column=1).value = operator_type
                                sheet.cell(row=row_idx, column=2).value = tn_node_id
                                sheet.cell(row=row_idx, column=3).value = es_id
                                sheet.cell(row=row_idx, column=4).value = int(year)
                                sheet.cell(row=row_idx, column=5).value = day
                                sheet.cell(row=row_idx, column=6).value = 'Energy Storage, soc_final'
                                sheet.cell(row=row_idx, column=7).value = s_m
                                sheet.cell(row=row_idx, column=8).value = s_o
                                for p in range(network[year][day].num_instants):
                                    soc_final = results[year][day]['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final'][es_id]
                                    sheet.cell(row=row_idx, column=p + 9).value = soc_final
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
            plt.close(fig)


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
    return candidate_solution


def _get_test_candidate_solution(planning_problem, s_inv=1.00, e_inv=2.00):
    candidate_solution = {'investment': {}, 'total_capacity': {}}
    for e in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[e]
        candidate_solution['investment'][node_id] = dict()
        candidate_solution['total_capacity'][node_id] = dict()
        for year in planning_problem.years:
            candidate_solution['investment'][node_id][year] = dict()
            candidate_solution['investment'][node_id][year]['s'] = s_inv
            candidate_solution['investment'][node_id][year]['e'] = e_inv
            candidate_solution['total_capacity'][node_id][year] = dict()
            candidate_solution['total_capacity'][node_id][year]['s'] = s_inv
            candidate_solution['total_capacity'][node_id][year]['e'] = e_inv
    return candidate_solution


def _check_interface_nodes_base_voltage_consistency(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            for node_id in planning_problem.distribution_networks:
                tn_node_base_kv = planning_problem.transmission_network.network[year][day].get_node_base_kv(node_id)
                dn_ref_node_id = planning_problem.distribution_networks[node_id].network[year][day].get_reference_node_id()
                dn_node_base_kv = planning_problem.distribution_networks[node_id].network[year][day].get_node_base_kv(dn_ref_node_id)
                if not isclose(tn_node_base_kv, dn_node_base_kv, rel_tol=5e-2):
                    print(f'[ERROR] Distribution Network {planning_problem.distribution_networks[node_id].name}, TN node {node_id}. Inconsistent TN-DN base voltage, year {year}, day {day}! Check network(s). Exiting')
                    exit(ERROR_SPECIFICATION_FILE)


def _add_adn_node_to_transmission_network(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            for node_id in planning_problem.distribution_networks:
                if planning_problem.transmission_network.network[year][day].adn_load_exists(node_id):
                    adn_load_idx = planning_problem.transmission_network.network[year][day].get_adn_load_idx(node_id)
                    adn_load = planning_problem.transmission_network.network[year][day].loads[adn_load_idx]
                    adn_load.load_id = f'ADN_{node_id}'
                    adn_load.pd = np.zeros(adn_load.pd.shape)
                    adn_load.qd = np.zeros(adn_load.qd.shape)
                    adn_load.fl_reg = True
                    adn_load.status = 1
                else:
                    adn_load = Load()
                    adn_load.bus = node_id
                    adn_load.load_id = f'ADN_{node_id}'
                    adn_load.pd = np.zeros((planning_problem.transmission_network.num_oper_scenarios, planning_problem.num_instants))
                    adn_load.qd = np.zeros((planning_problem.transmission_network.num_oper_scenarios, planning_problem.num_instants))
                    adn_load.fl_reg = True
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
