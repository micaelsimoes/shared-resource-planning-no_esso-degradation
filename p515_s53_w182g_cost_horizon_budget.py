"""
P5.15 Addendum 82 v2, W182 order part C, Planner task W182g -- pre-campaign readings C1, C2, C3 (brief section 2).
ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT. Production source is read as TEXT and parsed with `ast`, never
executed; case files are read as JSON / xlsx (pandas, header=None -- the production reader's own indexing); the
manuscript is read as text from the Overleaf clone at the declared HEAD (git object and working copy, both hashed).

C1  The cost boundary, term by term. Every term of the network (TSO / DSO) base objective `objective_function_rule`
    (found by AST: every `obj =` / `obj +=` statement of that function, and the set is asserted complete), every term
    the ADMM path adds (`update_transmission_model_to_admm`, `update_distribution_models_to_admm`,
    `update_shared_energy_storage_model_to_admm`), the ESSO objective, the salvage credit, the investment cost -- each
    with its building line, its weight in the ADMM path, its case-file activation, and its status:
        INSIDE_Q          evaluated by `get_primal_value` (network.py) and not subtracted in
                          `_get_operational_recourse_components`;
        OUTSIDE_Q         subtracted there, or not part of any model `get_primal_value` evaluates;
        ADMM_LOCAL_ONLY   in a local ADMM objective (`admm_objective`) but not in `objective_function_rule`.
    Q is aggregated at shared_resources_planning.py `_get_operational_recourse_components` (located by text). Record
    evidence (committed evaluation records, read-only): the identities gross = including_settlement - contracted -
    voltage_pin and net = gross - salvage on one single-scenario and two 3 x 3 records; deviation settlement == 0 at
    one scenario and != 0 at 3 x 3; the row 18 charge; the closure of the weighted component decomposition.
    The manuscript sentences on the boundary are quoted by exact text with their line numbers.

C2  The horizon end. terminal date = y0 + sum_y Y_y from each case file's "Years" key (SRP1.json; the 3 x 3 instance
    SRP1__s53_3x3.json), with the code lines that compute it, the terminal discount factor, the terminal SoH index and
    the remaining-life rule. Numeric zero-solve checks from committed records: the SoH chain
        SoH_y = SoH_{y-1} * exp(-365 * n * EFC/day_y / k) * phi**n      (k = N D / (-ln R))
    recomputed from the recorded EFC/day of each block and compared with the recorded end-of-block SoH, and the salvage
        C = (1+r)**-(y_End - y0) * cbarE(y_inv) * E * (T_rem / T_cal) * (SoH_term - SoH_min) / (1 - SoH_min)
    recomputed from the recorded terminal SoH and compared with the recorded terminal_salvage_value, for the SRP1 year
    ladder (2030, 2035) and the 3 x 3 reference unit (2025). The uncommitted 3 x 3 results workbook's "Terminal Salvage"
    sheet is read (hash-recorded, status UNCOMMITTED) for its terminal valuation date and discount factor.

C3  Budget admissibility of every plan named in main.tex. The formula (preserved here, `investment_cost`):
        I(x) = sum_n (1 + r)**-(y - y0) * sum_c w_c * (cS_{c,y} * S_n + cE_{c,y} * E_n)
    with w_c from the cost workbook's 'Scenarios' sheet, cS / cE from 'Investment Cost, Power' / 'Energy', r and y0
    from the case file -- the expression of shared_resources_planning.py `_check_candidate_first_stage_feasibility`
    and shared_energy_storage_data.py `_build_master_problem` (both located by text). Verdict: ADMISSIBLE iff
    I(x) <= B + 1e-8 (the production test), B = SRP1_ESS_Params.json "budget"; BOUNDARY flag iff admissible and
    B - I(x) < 1 % of B (declared here); the plan rules (2 <= E/S <= 4, E <= E_max) are reported beside. Plans are
    collected from: Table E.21 (tab:appE_evaluations), the gap table, the neighbours table, Tables value_ladder,
    energy_ladder_m2, year_ladder, and declared prose anchors (each verified by exact text); the families named only
    by count (twelve single-node 2025 plans, fourteen neighbours of x = 0, 61 neighbours of the returned plan) are
    enumerated in the l_inf box of the search (W173's box definition). Cross-checks: I(x) against the committed W2
    table (`investment_cost_results.json` I_new_eur, the production master's value) and the frozen Step-6 JSON
    (I_other / I_ref by candidate), matched by canonical candidate; Table 1 against the workbook; the 2025 expected
    costs printed in the text; the audit's figures (manuscript_review/manuscript_critical_review_2026-10-25.md 3.6).

Negative controls (in memory): an altered code line, an altered manuscript quote, a budget of 0.99e6, a perturbed
probability vector, an altered recorded SoH -- each must change the corresponding verdict or be refused.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0; pickle.load /
loads blocked and counted; no production module imported (asserted at the end); JSON written only through
gate_result_io. Refuses to run unless the output directory exists and holds nothing but launch.log, and when another
copy of this script is running. Read-only on the clone, the case files, the records and manuscript_review/.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w182g_cost_horizon_budget && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182g_cost_horizon_budget.py \\
      > data/SRP1/Results/P515S53/w182g_cost_horizon_budget/launch.log 2>&1
Optional: --out-dir <dir> (scratch trials only).
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W182g cost boundary, horizon end, budget (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W182g: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W182g: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pandas as pd  # noqa: E402  (third-party; the production reader's own library and indexing)
import openpyxl  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

SCRIPT = os.path.basename(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w182g_cost_horizon_budget')

# ------------------------------------------------------------------------------------------------------- identity
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
DECLARED_CLONE_HEAD = 'edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838'
DECLARED_REPO_HEAD = 'a17f6ba7'
LAST_PRODUCTION_COMMIT = '575bfec4'
AUDIT = os.path.join('manuscript_review', 'manuscript_critical_review_2026-10-25.md')

# ------------------------------------------------------------------------------------------------------- sources
SRC = {
    'srp': 'shared_resources_planning.py',
    'mch': 'model_construction_helpers.py',
    'nd': 'network_data.py',
    'net': 'network.py',
    'essd': 'shared_energy_storage_data.py',
    'essp': 'shared_energy_storage_parameters.py',
    'np': 'network_parameters.py',
    'defs': 'definitions.py',
}
PRODUCTION_FILES = tuple(SRC.values())
CASE_SRP1 = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_3X3 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
CASE_3X3_SHA = '2a64e3c5bd06c30c8a015207f41b69257b001094cbd4b185a84394a6765c5871'   # instance_record.json
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
COST_FILE = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')
COST_FILE_SHA = 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'      # p515_s47_phase_b_record.py
NET_PARAMS = {n: os.path.join('data', 'SRP1', n, f'{n}_params.json') for n in ('case9', 'case33_1', 'case33_2', 'case33_3')}

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W2_TABLE = os.path.join('data', 'SRP1', 'Results', 'P515S45', 'investment_cost', 'investment_cost_results.json')
W2_TABLE_SHA = '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b'     # p515_s47_phase_b_record.py
FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
EV_YL2035 = os.path.join(P53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_yl_y2035', 'evals', '3a2387f46448f9db_yl_y2035')
EV_YL2030 = os.path.join(P53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_yl_y2030', 'evals', '6e4f11a95dc3d585_yl_y2030')
EV_3X3_N7 = os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals', 'c82522f470b35b58_n7_4h_e1')
EV_3X3_X0 = os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals', 'f6e9cd53fdbb8ee8_x0')
WB_3X3_N7 = os.path.join(EV_3X3_N7, 'results', 'SRP1_distributed_terminal.xlsx')   # uncommitted (data/ results)

# No repository module other than gate_result_io and p513_solve_profile_guard may be imported (asserted at the end
# over every *.py at the repository root).

# ------------------------------------------------------------------------------------------------------- C1 anchors
# (id, source key, enclosing function, exact stripped line text). The line number is FOUND (unique within the
# function) and reported; nothing here is a line-number literal.
CODE = [
    # --- the network base objective (objective_function_rule) and where its terms are built
    ('obj_rule_def', 'mch', None, 'def objective_function_rule(model, params):'),
    ('obj_core', 'mch', 'objective_function_rule',
     'obj = model.total_gen_cost + model.total_flex_cost + model.total_load_curt_cost + model.total_gen_curt_penalty'),
    ('obj_cm_branch', 'mch', 'objective_function_rule',
     'obj = model.total_gen_curt_penalty + model.total_load_curt_penalty + model.total_flex_penalty'),
    ('obj_ess_usage', 'mch', 'objective_function_rule', 'obj += model.total_ess_utilization_cost_penalty'),
    ('obj_slacks', 'mch', 'objective_function_rule', 'obj += model.total_slack_penalties'),
    ('obj_closure', 'mch', 'objective_function_rule', 'obj += model.total_ess_complementarity_penalties'),
    ('obj_settlement', 'mch', 'objective_function_rule', 'obj += model.interface_settlement_weight * model.interface_settlement'),
    ('obj_row18', 'mch', 'objective_function_rule', 'obj += model.row18_deviation_charge'),
    ('obj_vpin', 'mch', 'objective_function_rule', 'obj += model.scenario_voltage_pin_weight * model.scenario_voltage_pin'),
    ('gen_cost_line', 'mch', 'generation_cost', 'gen_cost_scenario += c_p[p] * network.baseMVA * model.pg[g, s_m, s_o, p]'),
    ('gen_cost_filter', 'mch', 'generation_cost',
     'if network.generators[g].is_controllable() and not (not network.is_transmission and network.generators[g].gen_type == GEN_REFERENCE):'),
    ('flex_cost_line', 'mch', 'flexibility_cost', 'flex_cost += c_flex[p] * network.baseMVA * ('),
    ('flex_cost_tso_adn_skip', 'mch', 'flexibility_cost', 'if load_is_tso_adn_interface(network, load):'),
    ('load_curt_guard', 'mch', 'load_curtailment_cost', 'if params.l_curt:'),
    ('load_curt_line', 'mch', 'load_curtailment_cost', 'load_curt_cost += cost * network.baseMVA * ('),
    ('gen_curt_line', 'mch', 'gen_curtailment_penalty',
     'gen_curt_penalty += penalty * network.baseMVA * (model.pg_avail[g, s_o, p] - model.pg[g, s_m, s_o, p])'),
    ('sess_usage_line', 'mch', 'ess_utilization_cost_penalty',
     'cost += model.penalty_shared_ess_usage * network.baseMVA * (model.shared_es_pch[e, s_m0, s_o0, p] + model.shared_es_pdch[e, s_m0, s_o0, p])'),
    ('v_slack_line', 'mch', 'voltage_slack_penalty',
     'total += PENALTY_VOLTAGE_SQUARED * (model.slack_v_sqr_down[i, s_m, s_o, p] + model.slack_v_sqr_up[i, s_m, s_o, p])'),
    ('nb_slack_line', 'mch', 'node_balance_slack_penalty',
     'total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_p_up[i, s_m, s_o, p] + model.slack_node_balance_p_down[i, s_m, s_o, p])'),
    ('bf_slack_line', 'mch', 'branch_flow_slack_penalty',
     'total += base * PENALTY_CURRENT * (model.slack_flow_ij_sqr[b, s_m, s_o, p])'),
    ('flexdb_slack_line', 'mch', 'flexibility_p_day_balance_slack_penalty',
     'total += base * PENALTY_FLEXIBILITY * (model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o])'),
    ('local_closure_line', 'mch', 'local_ess_day_balance_slack_penalty',
     'total += base * PENALTY_ESS_BALANCE * (model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o])'),
    ('shared_closure_line', 'mch', 'shared_ess_day_balance_slack_penalty',
     'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m0, s_o0] + model.slack_shared_es_soc_final_down[e, s_m0, s_o0])'),
    ('settle_weight_param', 'mch', 'build_objective', 'model.interface_settlement_weight = pe.Param(initialize=0.00, mutable=True)'),
    ('settle_expr', 'mch', 'build_objective', 'model.interface_settlement = pe.Expression(expr=interface_energy_settlement(model, network))'),
    ('settle_contracted_expr', 'mch', 'build_objective',
     'model.interface_settlement_contracted = pe.Expression(expr=interface_energy_settlement_contracted(model, network))'),
    ('settle_deviation_expr', 'mch', 'build_objective',
     'model.interface_settlement_deviation = pe.Expression(expr=model.interface_settlement - model.interface_settlement_contracted)'),
    ('build_objective_line', 'mch', 'build_objective',
     'model.objective = pe.Objective(sense=pe.minimize, rule=partial(objective_function_rule, params=params))'),
    ('commit_single_scenario_guard', 'mch', 'add_scenario_commitment_terms', 'if n_scenarios == 1:'),
    ('row18_charge_term', 'mch', 'add_scenario_commitment_terms',
     'charge += (probability * model.row18_alpha * model.row18_premium[p] * network.baseMVA'),
    ('row18_charge_expr', 'mch', 'add_scenario_commitment_terms', 'model.row18_deviation_charge = pe.Expression(expr=charge)'),
    ('row18_dso_only', 'mch', 'add_scenario_commitment_terms', "if (not network.is_transmission) and premium_alpha > 0.0:"),
    ('vpin_weight', 'mch', 'add_scenario_commitment_terms',
     'model.scenario_voltage_pin_weight = pe.Param(initialize=PENALTY_SCENARIO_DEVIATION, mutable=True)'),
    ('vpin_expr', 'mch', 'add_scenario_commitment_terms', 'model.scenario_voltage_pin = pe.Expression(expr=voltage_deviation)'),
    # --- Q evaluation and aggregation
    ('net_primal_value', 'net', 'get_primal_value', 'primal_value = pe.value(objective_function_rule(model, params))'),
    ('nd_annualization', 'nd', 'get_primal_value', 'annualization = 1 / ((1 + self.discount_factor) ** (int(year) - int(years[0])))'),
    ('nd_weighting', 'nd', 'get_primal_value',
     'obj += self.network[year][day].get_primal_value(model[year][day], self.params) * self.years[year] * self.days[day] * annualization'),
    ('q_tso', 'srp', '_get_operational_recourse_components',
     "gross_operational_cost_including_settlement = planning_problem.transmission_network.get_primal_value(models['tso'])"),
    ('q_dso', 'srp', '_get_operational_recourse_components',
     "gross_operational_cost_including_settlement += distribution_network.get_primal_value(models['dso'][node_id])"),
    ('q_gross_1', 'srp', '_get_operational_recourse_components', 'gross_operational_cost = (gross_operational_cost_including_settlement'),
    ('q_gross_2', 'srp', '_get_operational_recourse_components', '- interface_settlement_contracted_total'),
    ('q_gross_3', 'srp', '_get_operational_recourse_components', '- voltage_pin_total)'),
    ('q_salvage', 'srp', '_get_operational_recourse_components',
     "terminal_salvage_value = planning_problem.shared_ess_data.get_salvage_value(models['esso'])"),
    ('q_net', 'srp', '_get_operational_recourse_components', 'net_operational_recourse = gross_operational_cost - terminal_salvage_value'),
    ('q_called_per_cycle', 'srp', '_run_operational_planning',
     'recourse_components = _get_operational_recourse_components(planning_problem, operational_models)'),
    # --- ADMM path: weights and the augmented local objectives
    ('admm_prepare_dso_call', 'srp', '_run_operational_planning', '_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)'),
    ('admm_prepare_tso_call', 'srp', '_run_operational_planning', '_prepare_transmission_objectives_for_admm(transmission_network, tso_model)'),
    ('admm_tso_sess_usage_0', 'srp', '_prepare_transmission_objectives_for_admm', 'model[year][day].penalty_shared_ess_usage.set_value(0.00)'),
    ('admm_tso_gencurt_0', 'srp', '_prepare_transmission_objectives_for_admm', 'model[year][day].penalty_gen_curtailment.set_value(0.00)'),
    ('admm_tso_loadcurt', 'srp', '_prepare_transmission_objectives_for_admm',
     'model[year][day].cost_load_curtailment.set_value(COST_CONSUMPTION_CURTAILMENT)'),
    ('admm_tso_settle_1', 'srp', '_prepare_transmission_objectives_for_admm', 'model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('admm_dso_sess_usage_0', 'srp', '_prepare_distribution_objectives_for_admm', 'dso_model[year][day].penalty_shared_ess_usage.set_value(0.00)'),
    ('admm_dso_gencurt_0', 'srp', '_prepare_distribution_objectives_for_admm', 'dso_model[year][day].penalty_gen_curtailment.set_value(0.00)'),
    ('admm_dso_loadcurt', 'srp', '_prepare_distribution_objectives_for_admm',
     'dso_model[year][day].cost_load_curtailment.set_value(COST_CONSUMPTION_CURTAILMENT)'),
    ('admm_dso_settle_1', 'srp', '_prepare_distribution_objectives_for_admm', 'dso_model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('admm_dso_row18_on', 'srp', '_prepare_distribution_objectives_for_admm', '_activate_row18_with_settlement(dso_model[year][day])'),
    ('admm_tso_obj_scaled', 'srp', 'update_transmission_model_to_admm', 'obj = copy(model[year][day].objective.expr) / effective_scale'),
    ('admm_tso_dual_v', 'srp', 'update_transmission_model_to_admm', 'obj += model[year][day].dual_vmag_req[dn, p] * constraint_v_req'),
    ('admm_tso_rho_v', 'srp', 'update_transmission_model_to_admm', 'obj += (model[year][day].rho_v / 2) * (constraint_v_req ** 2)'),
    ('admm_tso_prox_v', 'srp', 'update_transmission_model_to_admm', 'obj += (model[year][day].prox_gamma_v / 2) * proximal_v ** 2'),
    ('admm_tso_dual_ess', 'srp', 'update_transmission_model_to_admm', 'obj += (model[year][day].dual_ess_p_req[e, p]) * constraint_ess_p_req'),
    ('admm_tso_deact', 'srp', 'update_transmission_model_to_admm', 'model[year][day].objective.deactivate()'),
    ('admm_tso_obj', 'srp', 'update_transmission_model_to_admm', 'model[year][day].admm_objective = pe.Objective(sense=pe.minimize, expr=obj)'),
    ('admm_dso_obj_scaled', 'srp', 'update_distribution_models_to_admm', 'obj = copy(dso_model[year][day].objective.expr) / effective_scale'),
    ('admm_dso_rho_pf', 'srp', 'update_distribution_models_to_admm', 'obj += (dso_model[year][day].rho_pf / 2) * (constraint_p_req ** 2)'),
    ('admm_dso_obj', 'srp', 'update_distribution_models_to_admm',
     'dso_model[year][day].admm_objective = pe.Objective(sense=pe.minimize, expr=obj)'),
    ('admm_esso_base', 'srp', 'update_shared_energy_storage_model_to_admm', 'obj = copy(models[node_id].objective.expr)'),
    ('admm_esso_dual', 'srp', 'update_shared_energy_storage_model_to_admm', 'obj += models[node_id].dual_p_req[y, d, p] * constraint_p_req'),
    ('admm_esso_obj', 'srp', 'update_shared_energy_storage_model_to_admm',
     'models[node_id].admm_objective = pe.Objective(sense=pe.minimize, expr=obj)'),
    # --- ESSO objective and salvage
    ('esso_slack_pen', 'essd', '_build_subproblem',
     'slack_penalty += PENALTY_ESSO_SLACK * (model.slack_es_pnet_up[y_inv, d, p] + model.slack_es_pnet_down[y_inv, d, p])'),
    ('esso_throughput', 'essd', '_build_subproblem', 'slack_penalty += EPS_ESSO_THROUGHPUT * throughput'),
    ('esso_feas_expr', 'essd', '_build_subproblem', 'model.feasibility_penalty = pe.Expression(expr=slack_penalty)'),
    ('esso_salvage_expr', 'essd', '_build_subproblem', 'model.salvage_value = pe.Expression(expr=salvage_value)'),
    ('esso_objective_expr', 'essd', '_build_subproblem', 'expr=model.feasibility_penalty,'),
    ('salvage_sum', 'essd', 'get_salvage_value', 'salvage_value += pe.value(models[node_id].salvage_value)'),
    # --- investment cost and budget
    ('master_annualization', 'essd', '_build_master_problem',
     'investment_cost_total += annualization * omega_m * model.es_s_investment[e, y] * c_inv_s'),
    ('master_budget_row', 'essd', '_build_master_problem',
     'model.energy_storage_investment.add(investment_cost_total <= shared_ess_data.params.budget)'),
    ('master_objective', 'essd', '_build_master_problem',
     'model.objective = pe.Objective(sense=pe.minimize, expr=model.investment_cost + model.alpha)'),
    ('fs_tolerance', 'srp', '_check_candidate_first_stage_feasibility', 'tolerance = 1e-8'),
    ('fs_discount', 'srp', '_check_candidate_first_stage_feasibility',
     '(1.00 + shared_ess_data.discount_factor) ** (int(year) - int(years[0]))'),
    ('fs_expected', 'srp', '_check_candidate_first_stage_feasibility',
     'expected_investment_cost += year_discount * probability * ('),
    ('fs_terms', 'srp', '_check_candidate_first_stage_feasibility',
     "shared_ess_data.cost_investment['power'][scenario][year] * s_value"),
    ('fs_budget', 'srp', '_check_candidate_first_stage_feasibility', 'if expected_investment_cost > params.budget + tolerance:'),
    ('budget_read', 'essp', '_read_parameters_from_file', "planning_parameters.budget = float(params_data['budget'])"),
    ('cost_probs_read', 'essd', '_read_shared_energy_storage_data_from_file',
     "num_scenarios, shared_ess_data.prob_market_scenarios = _get_operational_scenarios_info_from_excel_file(filename, 'Scenarios')"),
    ('cost_power_read', 'essd', '_read_shared_energy_storage_data_from_file',
     "investment_costs['power'] = _get_investment_costs_from_excel_file(filename, 'Investment Cost, Power', num_scenarios, shared_ess_data.years)"),
    ('ess_prob_from_planning', 'srp', '_read_planning_problem', 'shared_ess_data.prob_market_scenarios = planning_problem.prob_market_scenarios'),
    ('ess_read_costfile', 'srp', '_read_planning_problem', 'shared_ess_data.read_shared_energy_storage_data_from_file()'),
    ('discount_read', 'srp', '_read_planning_problem', "planning_problem.discount_factor = planning_data['DiscountFactor']"),
    # --- horizon end, terminal SoH, salvage (C2)
    ('soh_recursion', 'essd', '_build_subproblem',
     '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))'),
    ('d_row', 'essd', '_build_subproblem', '== 365.00 * num_years * model.es_avg_ch_dch_per_unit[y_inv, y])'),
    ('num_years_inv_block', 'essd', '_build_subproblem', 'num_years = shared_ess_data.years[repr_years[y_inv]]'),
    ('soh_floor', 'essd', '_build_subproblem', 'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'),
    ('salv_terminal_idx', 'essd', '_build_terminal_salvage_value_expression', 'terminal_year_idx = len(shared_ess_data.years) - 1'),
    ('salv_terminal_disc', 'essd', '_build_terminal_salvage_value_expression',
     'terminal_discount = _get_terminal_discount_factor(shared_ess_data)'),
    ('salv_e_rated', 'essd', '_build_terminal_salvage_value_expression', 'e_rated = model.es_e_rated_per_unit[y_inv, terminal_year_idx]'),
    ('salv_e_avail', 'essd', '_build_terminal_salvage_value_expression',
     'e_available = model.es_e_available_per_unit[y_inv, terminal_year_idx]'),
    ('salv_mid_uses_end', 'essd', '_build_terminal_salvage_value_expression',
     'e_available = e_rated * model.es_soh_per_unit_cumul[y_inv, terminal_year_idx]'),
    ('salv_usable', 'essd', '_build_terminal_salvage_value_expression',
     'usable_energy_above_eol = (e_available - min_soh * e_rated) / (1.00 - min_soh)'),
    ('salv_cost_basis', 'essd', '_build_terminal_salvage_value_expression',
     'expected_unit_cost = _get_expected_energy_investment_cost(shared_ess_data, year_inv)'),
    ('term_years', 'essd', '_get_terminal_discount_factor',
     'terminal_years = sum(float(num_years) for num_years in shared_ess_data.years.values())'),
    ('term_disc', 'essd', '_get_terminal_discount_factor',
     'return 1.00 / ((1.00 + shared_ess_data.discount_factor) ** terminal_years)'),
    ('age_at_terminal', 'essd', '_get_remaining_calendar_life',
     'age_at_terminal = sum(float(value) for value in represented_years[investment_year_idx:])'),
    ('remaining_life', 'essd', '_get_remaining_calendar_life',
     'remaining_life = max(float(shared_energy_storage.t_cal) - age_at_terminal, 0.00)'),
    ('remaining_frac', 'essd', '_get_remaining_calendar_life',
     'remaining_life_fraction = min(remaining_life / float(shared_energy_storage.t_cal), 1.00)'),
    ('terminal_horizon', 'essd', '_get_salvage_value_results',
     'terminal_horizon_years = sum(float(value) for value in shared_ess_data.years.values())'),
    ('terminal_date', 'essd', '_get_salvage_value_results', 'terminal_date = int(years[0]) + terminal_horizon_years'),
    ('terminal_soh_report', 'essd', '_get_salvage_value_results',
     'terminal_soh = pe.value(model.es_soh_per_unit_cumul[y_inv, terminal_year_idx])'),
    # --- case-file activation
    ('l_curt_vars_guard', 'net', '_build_model', 'if params.l_curt:'),
    ('flex_day_balance_pinned', 'np', '__init__', 'self.day_balance = True'),
    ('l_curt_read', 'np', '_read_network_parameters_from_file', "parameters.l_curt = bool(params_data['l_curt'])"),
]
DEF_CONSTANTS = ('COST_CONSUMPTION_CURTAILMENT', 'PENALTY_VOLTAGE_SQUARED', 'PENALTY_CURRENT', 'PENALTY_NODE_BALANCE',
                 'PENALTY_FLEXIBILITY', 'PENALTY_ESS_BALANCE', 'PENALTY_SHARED_ESS_BALANCE',
                 'PENALTY_GENERATION_CURTAILMENT', 'PENALTY_LOAD_CURTAILMENT', 'PENALTY_FLEXIBILITY_USAGE',
                 'PENALTY_ESS_USAGE', 'PENALTY_ESSO_SLACK', 'EPS_ESSO_THROUGHPUT', 'PENALTY_SCENARIO_DEVIATION',
                 'PENALTY_SHARED_ESS_SCENARIO_DEVIATION')

# The C1 term table. `rule_attrs` are the Pyomo attributes the term enters objective_function_rule through (checked
# against the AST of that function); `status` is the boundary status; `anchors` the CODE ids that carry it.
TERMS = [
    {'term': 'TN generation cost (controllable generators at the scenario market price; a DSO reference generator '
             'excluded)', 'rule_attrs': ['total_gen_cost'], 'status': 'INSIDE_Q', 'agents': 'TSO, DSO',
     'anchors': ['obj_core', 'gen_cost_line', 'gen_cost_filter'], 'admm_weight': 'unchanged (market price)'},
    {'term': 'Flexibility cost (fl_reg loads, flex_p_down + flex_q_down at cost_flex; TSO ADN-interface loads excluded)',
     'rule_attrs': ['total_flex_cost'], 'status': 'INSIDE_Q', 'agents': 'TSO (non-interface loads), DSO',
     'anchors': ['obj_core', 'flex_cost_line', 'flex_cost_tso_adn_skip'], 'admm_weight': 'unchanged'},
    {'term': 'Load curtailment cost (cost_load_curtailment x curtailment)', 'rule_attrs': ['total_load_curt_cost'],
     'status': 'INSIDE_Q', 'agents': 'TSO, DSO',
     'anchors': ['obj_core', 'load_curt_guard', 'load_curt_line', 'admm_tso_loadcurt', 'admm_dso_loadcurt',
                 'l_curt_vars_guard', 'l_curt_read'],
     'admm_weight': 'COST_CONSUMPTION_CURTAILMENT; term identically 0 when l_curt is false (variables not built)'},
    {'term': 'RES curtailment penalty (pg_avail - pg)', 'rule_attrs': ['total_gen_curt_penalty'], 'status': 'INSIDE_Q',
     'agents': 'TSO, DSO', 'anchors': ['obj_core', 'gen_curt_line', 'admm_tso_gencurt_0', 'admm_dso_gencurt_0'],
     'admm_weight': '0 (set in both _prepare_*_objectives_for_admm): contributes 0 to Q'},
    {'term': 'Storage usage penalty (shared ESS pch + pdch; local ESS when es_reg)',
     'rule_attrs': ['total_ess_utilization_cost_penalty'], 'status': 'INSIDE_Q', 'agents': 'TSO, DSO',
     'anchors': ['obj_ess_usage', 'sess_usage_line', 'admm_tso_sess_usage_0', 'admm_dso_sess_usage_0'],
     'admm_weight': 'shared-ESS weight 0 in ADMM; local-ESS part needs local ESSs (none in SRP1 records: ess_usage_cost 0)'},
    {'term': 'Network-feasibility slack penalties (voltage^2 slack; node balance; branch flow; flexibility P day-balance)',
     'rule_attrs': ['total_slack_penalties'], 'status': 'INSIDE_Q', 'agents': 'TSO, DSO',
     'anchors': ['obj_slacks', 'v_slack_line', 'nb_slack_line', 'bf_slack_line', 'flexdb_slack_line', 'flex_day_balance_pinned'],
     'admm_weight': 'unchanged (constants in definitions.py); case flags decide which families exist'},
    {'term': 'Storage daily energy-closure slack penalties (shared ESS c^Cl; local ESS)',
     'rule_attrs': ['total_ess_complementarity_penalties'], 'status': 'INSIDE_Q', 'agents': 'TSO, DSO',
     'anchors': ['obj_closure', 'shared_closure_line', 'local_closure_line'], 'admm_weight': 'unchanged'},
    {'term': 'Interface energy settlement, CONTRACTED part (expected exchange at the mean price)',
     'rule_attrs': ['interface_settlement_weight', 'interface_settlement'], 'status': 'OUTSIDE_Q',
     'agents': 'TSO, DSO', 'anchors': ['obj_settlement', 'settle_expr', 'settle_contracted_expr', 'settle_weight_param',
                                       'admm_tso_settle_1', 'admm_dso_settle_1', 'q_gross_2'],
     'admm_weight': '1 in the ADMM path (0 otherwise); in every local objective, subtracted from Q'},
    {'term': 'Interface energy settlement, DEVIATION part (price-deviation covariance; the imbalance settlement)',
     'rule_attrs': ['interface_settlement_weight', 'interface_settlement'], 'status': 'INSIDE_Q',
     'agents': 'DSO (TSO part identically 0, its schedule being scenario-free)',
     'anchors': ['obj_settlement', 'settle_deviation_expr', 'q_gross_1'],
     'admm_weight': '1; identically 0 at one market scenario'},
    {'term': 'Row 18 commitment-deviation premium (alpha * pibar_t * baseMVA * (d+ + d-), P and Q)',
     'rule_attrs': ['row18_deviation_charge'], 'status': 'INSIDE_Q', 'agents': 'DSO only',
     'anchors': ['obj_row18', 'row18_charge_term', 'row18_charge_expr', 'row18_dso_only', 'commit_single_scenario_guard',
                 'admm_dso_row18_on'],
     'admm_weight': 'alpha (0.5 in the 3 x 3 campaign); not constructed at one scenario; inactive at the initialisation solve'},
    {'term': 'Interface-voltage dispersion pin (PENALTY_SCENARIO_DEVIATION x E_s (v - vbar)^2)',
     'rule_attrs': ['scenario_voltage_pin_weight', 'scenario_voltage_pin'], 'status': 'OUTSIDE_Q', 'agents': 'TSO, DSO',
     'anchors': ['obj_vpin', 'vpin_weight', 'vpin_expr', 'q_gross_3', 'commit_single_scenario_guard'],
     'admm_weight': 'solver-only; not constructed at one scenario; subtracted from Q'},
    {'term': 'ADMM augmented-Lagrangian terms (duals and rho/2 on interface V, P, Q and shared-ESS P, Q; TSO proximal '
             'terms; previous-iterate ESS terms; base objective divided by the block scale)', 'rule_attrs': [],
     'status': 'ADMM_LOCAL_ONLY', 'agents': 'TSO, DSO',
     'anchors': ['admm_tso_obj_scaled', 'admm_tso_dual_v', 'admm_tso_rho_v', 'admm_tso_prox_v', 'admm_tso_dual_ess',
                 'admm_tso_deact', 'admm_tso_obj', 'admm_dso_obj_scaled', 'admm_dso_rho_pf', 'admm_dso_obj', 'net_primal_value'],
     'admm_weight': 'admm_objective only; get_primal_value evaluates objective_function_rule, not admm_objective'},
    {'term': 'ESSO objective (PENALTY_ESSO_SLACK x pnet slacks + EPS_ESSO_THROUGHPUT x throughput) and its AL terms',
     'rule_attrs': [], 'status': 'ADMM_LOCAL_ONLY', 'agents': 'ESSO',
     'anchors': ['esso_slack_pen', 'esso_throughput', 'esso_feas_expr', 'esso_objective_expr', 'admm_esso_base',
                 'admm_esso_dual', 'admm_esso_obj', 'q_tso', 'q_dso'],
     'admm_weight': 'the ESSO model is not summed into Q (only TSO and DSO get_primal_value are)'},
    {'term': 'Terminal salvage credit (ESSO Expression salvage_value; not in the ESSO objective)', 'rule_attrs': [],
     'status': 'OUTSIDE_Q', 'agents': 'ESSO (reporting)',
     'anchors': ['esso_salvage_expr', 'esso_objective_expr', 'salvage_sum', 'q_salvage', 'q_net'],
     'admm_weight': 'net = Q - salvage; never in any objective'},
    {'term': 'Investment cost I(x) (expected over cost trajectories, discounted to y0)', 'rule_attrs': [],
     'status': 'OUTSIDE_Q', 'agents': 'planning (master / search)',
     'anchors': ['master_annualization', 'master_objective', 'fs_expected'],
     'admm_weight': 'F = I + Q at the planning level; not in any operational model'},
]
RETIRED = [  # retained helpers whose term is not in any objective (unwired), found as defs
    ('adn_interface_flexibility_cost', 'mch', 'row 3 / D1: the TSO ADN-interface flexibility charge (reporting helper only)'),
    ('ess_complementarity_bilinear_value', 'mch', 'row 9: the bilinear pch*pdch penalty (reporting helper only)'),
    ('_add_tso_scenario_deviation_penalty', 'srp', 'the 9e4 scenario-deviation quadratic, replaced by row 18 / the pin'),
    ('_add_dso_scenario_deviation_penalty', 'srp', 'the DSO counterpart, likewise retired'),
]

# ------------------------------------------------------------------------------------------------------- manuscript
MS_QUOTES = {   # id -> exact substring of main.tex at the declared HEAD; every line holding it is reported
    'C1_Q_includes': ('It includes TN generation costs, activated DN flexibility, load curtailment, and penalties on storage '
                      'energy-closure and network-feasibility slacks. In the multi-scenario instance, it also includes the '
                      'commitment charge and the deviation component of interface settlement'),
    'C1_contracted_excluded': ('The contracted component of interface settlement is excluded as a transfer between operators. '
                               'Storage-agent and solver regularisation terms are also excluded. Renewable curtailment is '
                               'assigned no cost and is reported separately as curtailed energy.'),
    'C1_weights': ('The weights $Y_y$ and $D_d$ are the numbers of calendar years and days represented by $y$ and $d$, '
                   'respectively. Operating costs are discounted to $y_0$ using the representative year $y$.'),
    'C1_premium_additional': 'The deviation premium is additional to settlement at the corresponding scenario prices.',
    'C1_no_tso_premium': 'No deviation premium is imposed on the TSO.',
    'C1_single_scenario_omitted': 'With a single scenario, the interface exchange equals its expected value and the deviation-charge term is omitted.',
    'C1_appA_settlement': ('The interface settlement is carried at full weight: its contracted part is removed from the reported '
                           'cost as a transfer, its deviation part is kept. A quadratic regularisation of the interface-voltage '
                           'dispersion across scenarios is added to the TSO and DSO models as a solver-side term and removed '
                           'from the reported cost.'),
    'C1_appA_single': ('At a single scenario the commitment charge and the voltage regularisation are not constructed; the '
                       'interface settlement remains in every local objective at full weight and is excluded from the reported '
                       'cost as a transfer.'),
    'C1_appE_blanket': ('Throughout, $Q$ is the gross operational cost, settlement excluded; the net form subtracts the terminal '
                        'salvage credit~\\eqref{eq:salvage_credit}'),
    'C1_benchmark_same_Q': 'All arrangements are evaluated using the same cost function $Q$, excluding interface settlement.',
    'C1_block_weight': ('Each network block objective is divided by a common scale $\\sigma$ and multiplied by its block weight '
                        '$w_{y,d} = Y_y D_d (1+r)^{-(y-y_0)}$'),
    'C1_F_def': 'F(\\boldsymbol{x}) = I(\\boldsymbol{x}) + Q(\\boldsymbol{x})',
    'C1_no_cycling_cost': 'No explicit monetary cycling cost is included',
    'C1_salvage_after': 'The credit is calculated after the operational evaluation and does not affect dispatch or investment selection.',
    'C1_salvage_limitations': 'Terminal salvage credits are excluded from the planning objective and reported only in net values.',
    'C1_salvage_reported': 'The credit is reported, not optimized: the recourse does not see it, no plan is selected for it',
    'C1_closure_price': 'priced at $c^{\\text{Cl}} = 10^{3}$~\\euro/MWh',
    'C1_passive_rescurt': ('In the passive arrangement, the DN dispatch objective includes a renewable-curtailment coefficient of '
                           '1~\\euro/MWh. This auxiliary term is excluded from the reported operating cost $Q$.'),
    'C1_fig_benchmark': 'Interface settlement is excluded.',
    'C2_horizon_15': 'over a 15-year horizon',
    'C2_two_configs': 'Two configurations represent the 15-year planning horizon.',
    'C2_srp1_years': 'using representative years 2025, 2030 and 2035, each representing five calendar years.',
    'C2_3x3_years': 'using representative years 2025, 2028, 2031, 2034 and 2037, each representing three calendar years.',
    'C2_nomenclature': 'First year and end of the planning horizon',
    'C2_end_of_block': 'Available energy is evaluated using end-of-block state of health',
    'C2_boundary': 'Let $y^{-}$ denote the boundary immediately before the block represented by year $y$.',
    'C2_salvage_discount': '{(1+r)^{\\,y^\\text{End}-y_0}}',
    'C2_salvage_soh': '\\frac{SoH_{e,y^\\text{Inv},y^\\text{End}} - SoH^{\\text{Min}}_e}{1 - SoH^{\\text{Min}}_e}',
    'C2_remaining_life': 'T^\\text{Rem}_{e,y^\\text{Inv}} = \\max\\left\\{0,\\; y^\\text{Inv} + T^\\text{Cal}_{e,y^\\text{Inv}} - y^\\text{End}\\right\\}',
    'C2_yend_def': '$y^\\text{End}$ the end of the planning horizon, $T^\\text{Rem}$ the remaining calendar life of the cohort at $y^\\text{End}$',
    'C2_later_installs': 'Later installations have fewer operating years within the horizon and receive larger terminal salvage credits.',
    'C2_rescale_3x3': 'In the multi-scenario configuration, the 2025 capacities also apply in 2028, the 2030 capacities apply in 2031 and 2034, and the 2035 capacities apply in 2037.',
    'C2_calendar_life': 'The calendar lifetime is $T^{\\text{Cal}} = 15$~years.',
    'C3_budget_text': 'The baseline expected discounted investment budget is 1~M\\euro{}, with a maximum rated energy capacity of 5~MWh per interface node. Costs are discounted at an annual rate of 2\\%.',
    'C3_budget_def': 'is the budget for expected investment expenditure discounted to the initial year $y_0$.',
    'C3_budget_expected': 'It does not require expenditure under every individual trajectory to remain within the budget.',
    'C3_expected_2025': 'The probability-weighted unit costs in 2025 are 256.32~k\\euro{}/MVA for converter rating and 253.88~k\\euro{}/MWh for energy capacity.',
    'C3_increments': 'Capacity additions use increments of $\\Delta^S = 0.25$~MVA and $\\Delta^E = 0.5$~MWh, with energy-to-converter-rating ratios between 2 and 4~MWh/MVA.',
    'C3_I_318_text': 'against an investment cost of 318.0~k\\euro{}',
    'C3_I_318_table': 'Investment cost & 318.0~k\\euro{} \\\\',
    'C3_p_256320': 'with converter cost fixed at $p=256{,}320$~\\euro/MVA',
    'C3_e_253880': 'baseline energy-capacity cost of 253{,}880~\\euro/MWh',
    'C3_returned_plan': ('the planning search returned a two-node plan installed in 2030: 0.25~MVA and 0.5~MWh at node~5, and '
                         '1~MVA and 3.5~MWh at node~7.'),
    'C3_61': 'Of its 61 admissible unit neighbours, 17 were evaluated and 13 were re-evaluated',
    'C3_twelve': 'shows the resulting plans of a single node installed in 2025: twelve are admissible, the smallest being 0.25~MVA / 0.5~MWh',
    'C3_fourteen': ('The plan without storage therefore has fourteen admissible neighbours: the smallest unit at any non-empty '
                    'subset of the three nodes, installed in 2025 or 2030.'),
    'C3_six_others': 'The other six, the smallest unit at the two remaining pairs of nodes and at all three nodes, in 2025 or 2030, were evaluated by the search only',
    'C3_over_budget_seed': 'including ladder points over the budget',
    'C3_two_node_2030_salvage': 'The two-node plan installed in 2030 remains uneconomic after the terminal salvage credit, with a cost increase of 84.0~k\\euro{}.',
}

# Prose plan anchors: (id, exact substring, list of plans). A plan: (year, ((node, S, E), ...)).
REF = (2025, ((7, 0.25, 1.0),))
PROSE_PLANS = [
    ('abstract_reference', 'For a fixed reference 0.25~MVA/1~MWh unit', [REF]),
    ('baseline_reference', 'comprising 0.25~MVA and 1~MWh installed at node~7 in 2025, provides discounted operating savings', [REF]),
    ('baseline_node5_smallest', 'The smallest admissible unit at node~5, comprising 0.25~MVA and 0.5~MWh', [(2025, ((5, 0.25, 0.5),))]),
    ('two_node_2030_baseline', 'The two-node plan installed in 2030 remains uneconomic', [(2030, ((5, 0.25, 0.5), (7, 0.25, 0.5)))]),
    ('returned_plan_m2', 'the planning search returned a two-node plan installed in 2030: 0.25~MVA and 0.5~MWh at node~5, and 1~MVA and 3.5~MWh at node~7',
     [(2030, ((5, 0.25, 0.5), (7, 1.0, 3.5)))]),
    ('ageing_reference', 'compare the reference unit, comprising 0.25~MVA and 1~MWh installed at node~7 in 2025', [REF]),
    ('multiscenario_reference', 'Multi-scenario evaluation of the reference unit (node~7, 0.25~MVA / 1~MWh, installed in 2025)', [REF]),
    ('larger_units_2030_2035', 'The larger units evaluated at node~7, comprising 0.5~MVA / 1~MWh and 0.5~MVA / 2~MWh, also remain more expensive than the plan without storage when installed in either year',
     [(2030, ((7, 0.5, 1.0),)), (2035, ((7, 0.5, 1.0),)), (2030, ((7, 0.5, 2.0),)), (2035, ((7, 0.5, 2.0),))]),
    ('year_ladder_caption', 'Effect of installation year on the discounted cost of the reference unit (node~7, 0.25~MVA / 1~MWh).', [REF]),
    ('discount_caption', 'Effect of discount rate on the reference investment (node~7, 0.25~MVA / 1~MWh, installed in 2025).', [REF]),
    ('conclusions_reference', 'The reference unit, comprising 0.25~MVA and 1~MWh, incurred', [REF]),
    ('lattice_smallest', 'the smallest being 0.25~MVA / 0.5~MWh', [(2025, ((5, 0.25, 0.5),)), (2025, ((7, 0.25, 0.5),)), (2025, ((9, 0.25, 0.5),))]),
]

# Audit 3.6 figures (manuscript_review/manuscript_critical_review_2026-10-25.md), quoted by exact text
AUDIT_QUOTES = {
    'unit_costs': 'The 2025 probability-weighted costs derived from Table 1 are approximately 256.318 k€/MVA and 253.8755 k€/MWh.',
    'row_1': '| 1 MWh / 0.25 MVA | 318.0 k€ |',
    'row_2': '| 2 MWh / 0.50 MVA | 635.9 k€ |',
    'row_3': '| 3 MWh / 0.75 MVA | 953.9 k€ |',
    'row_4': '| 4 MWh / 1.00 MVA | 1,271.8 k€ |',
    'row_5': '| 5 MWh / 1.25 MVA | 1,589.8 k€ |',
    'two_node': 'The returned two-node 2030 plan is a different configuration and has an expected discounted investment of approximately 992.3 k€ from the same table',
}
AUDIT_VALUES = {'unit_power_keur': 256.318, 'unit_energy_keur': 253.8755,
                'ladder_keur': {1: 318.0, 2: 635.9, 3: 953.9, 4: 1271.8, 5: 1589.8}, 'two_node_keur': 992.3}

BOUNDARY_FRACTION = 0.01   # declared: an admissible plan within 1 % of B of the budget is flagged BOUNDARY
FS_TOLERANCE = 1e-8        # the production test's tolerance (anchor fs_tolerance)

# ------------------------------------------------------------------------------------------------------- registry
INPUTS = {}
_TRACKED = None


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(path):
    with open(path, 'rb') as f:
        return _sha_bytes(f.read())


def _git(args, cwd=REPO, binary=False):
    r = subprocess.run(['git'] + list(args), cwd=cwd, check=True, capture_output=True, text=not binary)
    return r.stdout


def _tracked(path):
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = set(_git(['ls-files', '-z']).split('\0'))
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel.startswith(CLONE + os.sep):
        return 'overleaf clone (tracked there; read only)'
    return 'committed' if rel in _TRACKED else 'UNCOMMITTED'


def _reg(path):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': _sha(path), 'status': _tracked(path)}
    return INPUTS[rel]


def _reg_blob(label, data, status):
    INPUTS[label] = {'sha256': _sha_bytes(data), 'status': status}


def _read(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _read_jsonl_last(path):
    lines = [ln for ln in _read(path).split('\n') if ln.strip()]
    return len(lines), json.loads(lines[-1])


def _ck(name, ok, detail=None):
    return {'check': name, 'ok': bool(ok), 'detail': detail}


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ------------------------------------------------------------------------------------------------------- code reading
def _func_spans(tree):
    """{name: [(lineno, end_lineno), ...]} for every function / method definition."""
    spans = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.setdefault(node.name, []).append((node.lineno, node.end_lineno))
    return spans


def load_sources():
    srcs = {}
    for key, path in SRC.items():
        text = _read(path)
        tree = ast.parse(text)
        srcs[key] = {'path': path, 'text': text, 'lines': text.split('\n'), 'tree': tree, 'spans': _func_spans(tree)}
    return srcs


def locate(srcs, key, func, text):
    s = srcs[key]
    hits = [i + 1 for i, ln in enumerate(s['lines']) if ln.strip() == text]
    if func is not None:
        spans = s['spans'].get(func, [])
        hits = [h for h in hits if any(a <= h <= b for a, b in spans)]
    return hits


def code_anchors(srcs, code=CODE):
    out, checks = {}, []
    for cid, key, func, text in code:
        hits = locate(srcs, key, func, text)
        ok = len(hits) >= 1
        # uniqueness is required except for text that legitimately repeats inside one function
        unique = len(hits) == 1
        out[cid] = {'file': SRC[key], 'function': func, 'lines': hits, 'line': hits[0] if hits else None,
                    'text': text, 'unique_in_function': unique}
        checks.append(_ck(f'{cid}: {SRC[key]}:{hits} in {func}', ok, None if ok else text))
    return out, checks


def objective_rule_terms(srcs):
    """Every attribute of `model` read by an `obj =` / `obj +=` statement in objective_function_rule (AST)."""
    tree = srcs['mch']['tree']
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'objective_function_rule')
    stmts = []
    for node in ast.walk(fn):
        tgt, val = None, None
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == 'obj':
            tgt, val = '+=', node.value
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'obj' for t in node.targets):
            tgt, val = '=', node.value
        if val is None:
            continue
        attrs = sorted({a.attr for a in ast.walk(val) if isinstance(a, ast.Attribute)
                        and isinstance(a.value, ast.Name) and a.value.id == 'model'})
        stmts.append({'line': node.lineno, 'op': tgt, 'model_attrs': attrs})
    return sorted(stmts, key=lambda s: s['line'])


def def_constants(srcs):
    tree = srcs['defs']['tree']
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in DEF_CONSTANTS:
                vals[name] = {'value': ast.literal_eval(node.value), 'line': node.lineno}
    return vals


def retired_defs(srcs):
    out = {}
    for name, key, what in RETIRED:
        spans = srcs[key]['spans'].get(name, [])
        # callers outside its own def (any `name(` occurrence not on the def line)
        calls = [i + 1 for i, ln in enumerate(srcs[key]['lines']) if re.search(r'\b' + re.escape(name) + r'\(', ln)
                 and not ln.lstrip().startswith('def ')]
        out[name] = {'file': SRC[key], 'def_lines': spans, 'what': what, 'call_lines_in_file': calls}
    return out


# ------------------------------------------------------------------------------------------------------- manuscript
def identity():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', 'main.tex'], cwd=CLONE).strip()
    blob = _git(['show', f'{DECLARED_CLONE_HEAD}:main.tex'], cwd=CLONE, binary=True)
    _reg_blob(f'{CLONE}@{DECLARED_CLONE_HEAD[:7]}:main.tex', blob, 'overleaf clone git object (read only)')
    with open(MAIN_TEX, 'rb') as fh:
        wc = fh.read()
    _reg(MAIN_TEX)
    repo_head = _git(['rev-parse', 'HEAD']).strip()
    prod_status = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES)).strip()
    last_prod = _git(['log', '-1', '--format=%H', '--'] + list(PRODUCTION_FILES)).strip()
    per_file = {f: _git(['log', '-1', '--format=%H', '--', f]).strip() for f in PRODUCTION_FILES}
    after_prod = _git(['log', '--format=%H', f'{LAST_PRODUCTION_COMMIT}..HEAD', '--'] + list(PRODUCTION_FILES)).strip()
    anc = subprocess.run(['git', 'merge-base', '--is-ancestor', last_prod, LAST_PRODUCTION_COMMIT], cwd=REPO).returncode == 0
    checks = [
        _ck('clone HEAD = edd7a63', head == DECLARED_CLONE_HEAD, head),
        _ck('main.tex unmodified in the clone', status == '', status),
        _ck('main.tex working copy = HEAD blob', wc == blob),
        _ck(f'declared order commit {DECLARED_REPO_HEAD} is an ancestor of repo HEAD (parallel W182 commits allowed)',
            subprocess.run(['git', 'merge-base', '--is-ancestor', DECLARED_REPO_HEAD, 'HEAD'], cwd=REPO).returncode == 0,
            repo_head),
        _ck('production files clean in the working tree', prod_status == '', prod_status),
        _ck(f'no commit after {LAST_PRODUCTION_COMMIT} touches the production files read here',
            after_prod == '' and anc, {'after': after_prod, 'last_touching': last_prod, 'ancestor_of_declared': anc}),
    ]
    out = {'clone': CLONE, 'clone_head': head, 'main_sha256': _sha_bytes(wc), 'main_n_lines': wc.decode('utf-8').count('\n'),
           'repo_head': repo_head, 'last_commit_touching_production_files_read': last_prod,
           'last_commit_per_production_file': per_file, 'declared_last_production_commit': LAST_PRODUCTION_COMMIT,
           'checks': checks, 'ok': all(c['ok'] for c in checks)}
    return out, blob.decode('utf-8').split('\n')


def quote_lines(lines, needle):
    return [i + 1 for i, ln in enumerate(lines) if needle in ln]


def ms_quotes(lines, quotes=MS_QUOTES):
    out, checks = {}, []
    for qid, needle in quotes.items():
        hits = quote_lines(lines, needle)
        out[qid] = {'lines': hits, 'text': needle}
        checks.append(_ck(f'quote {qid} present', len(hits) >= 1, hits))
    return out, checks


def block(lines, start_needle, end_needle):
    s = next(i for i, ln in enumerate(lines) if start_needle in ln)
    e = next(i for i in range(s, len(lines)) if end_needle in lines[i])
    return s, e


# ------------------------------------------------------------------------------------------------------- plans
PLAN_SEG = re.compile(r'^([\d, ]+):\s*([\d.]+)/([\d.]+)(\s+each)?$')
ZERO_PLAN = '$\\boldsymbol{x}=\\boldsymbol{0}$'


def parse_plan(text, year):
    """'5: 0.25/0.5; 7: 1/3.5' -> (year, ((5, .25, .5), (7, 1., 3.5))); x = 0 -> (None, ())."""
    t = re.sub(r'\$\^\{[a-z]\}\$', '', text).replace('(reference unit)', '').strip()
    if t == ZERO_PLAN:
        return (None, ())
    nodes = []
    for seg in [s.strip() for s in t.split(';')]:
        m = PLAN_SEG.match(seg)
        if not m:
            raise ValueError(f'unparsed plan segment {seg!r} in {text!r}')
        for n in [int(x) for x in m.group(1).split(',') if x.strip()]:
            nodes.append((n, float(m.group(2)), float(m.group(3))))
    return (int(year), tuple(sorted(nodes)))


def canon(plan):
    year, nodes = plan
    nodes = tuple(sorted((n, s, e) for n, s, e in nodes if s > 0 or e > 0))
    return (None, ()) if not nodes else (int(year), nodes)


def plan_label(plan):
    year, nodes = canon(plan)
    if not nodes:
        return 'x=0'
    return '; '.join(f'{n}: {s:g}/{e:g}' for n, s, e in nodes) + f' @{year}'


def plan_from_canonical(cc):
    nodes = tuple((int(n), float(v[0]), float(v[1])) for n, v in cc['nodes'].items())
    return canon((int(cc['investment_year']), nodes))


def parse_evaluations(lines):
    s, e = block(lines, '\\label{tab:appE_evaluations}', '\\end{longtable}')
    rows = {}
    rx = re.compile(r'^(\d+) & (.+?) & (---|\d{4}) & ([^&]+?) & ([^&]+?) & ')
    for i in range(s, e):
        m = rx.match(lines[i])
        if m:
            no = int(m.group(1))
            plan = parse_plan(m.group(2), m.group(3) if m.group(3) != '---' else 0)
            rows[no] = {'line': i + 1, 'plan_text': m.group(2), 'year_text': m.group(3), 'condition': m.group(4).strip(),
                        'status': m.group(5).strip(), 'plan': canon(plan)}
    return rows


def parse_claims_evals(lines):
    s, e = block(lines, '\\label{tab:appE_claims}', '\\end{longtable}')
    rx = re.compile(r'^(\d+) & (.+?) & (\d+)(?:\$\^\{[a-z]\}\$)? / (\d+)(?:\$\^\{[a-z]\}\$)? & ')
    rows = {}
    for i in range(s, e):
        m = rx.match(lines[i])
        if m:
            rows[int(m.group(1))] = {'line': i + 1, 'text': m.group(2), 'evals': (int(m.group(3)), int(m.group(4)))}
    return rows


def parse_gap(lines):
    s, e = block(lines, '\\label{tab:appE_gap}', '\\end{tabular}')
    rx = re.compile(r'^\s*(\d+) & (.+?) & (---|\d{4}) & ')
    rows = []
    for i in range(s, e):
        m = rx.match(lines[i])
        if m:
            rows.append({'line': i + 1, 'eval': int(m.group(1)),
                         'plan': canon(parse_plan(m.group(2), m.group(3) if m.group(3) != '---' else 0))})
    return rows


def parse_neighbours(lines):
    s, e = block(lines, '\\label{tab:appE_neighbours}', '\\end{tabular}')
    rx = re.compile(r'^\s*(\d(?: and \d)?) & (\d{4}) & (\d+) & ')
    rows = []
    for i in range(s, e):
        m = rx.match(lines[i])
        if m:
            nodes = tuple((int(n), 0.25, 0.5) for n in m.group(1).split(' and '))
            rows.append({'line': i + 1, 'eval': int(m.group(3)), 'plan': canon((int(m.group(2)), nodes))})
    return rows


def parse_value_ladder(lines):
    s, e = block(lines, '\\label{tab:res_value_ladder}', '\\end{tabular}')
    rx = re.compile(r'^\s*(.+?)\s*& ([\d.]+)( each)? & ([\d.]+)( each)? & (\d) & ([\d.]+) & ([\d.]+) \\\\')
    rows = []
    for i in range(s, e):
        m = rx.match(lines[i])
        if m:
            label = m.group(1)
            year = 2030 if '(2030)' in label else 2025
            node_ids = [int(x) for x in re.findall(r'\d', re.sub(r'\(.*?\)', '', label))]
            nodes = tuple((n, float(m.group(2)), float(m.group(4))) for n in node_ids)
            rows.append({'line': i + 1, 'label': label, 'plan': canon((year, nodes)), 'cost_increase_keur': float(m.group(7))})
    return rows


def parse_energy_ladder_m2(lines):
    s, e = block(lines, '\\label{tab:res_energy_ladder_m2}', '\\end{tabular}')
    rx = re.compile(r'^\s*(\d)(?:\$\^\{[a-z]\}\$)?\s*& ')
    rows = []
    for i in range(s, e):
        m = rx.match(lines[i])
        if m:
            E = float(m.group(1))
            rows.append({'line': i + 1, 'plan': canon((2025, ((7, E / 4.0, E),)))})
    return rows


def parse_year_ladder(lines):
    s, e = block(lines, '\\label{tab:res_year_ladder}', '\\end{tabular}')
    rx = re.compile(r'^\s*(2025|2030|2035)(?: \(reference\))?\s*& ')
    return [{'line': i + 1, 'plan': canon((int(rx.match(lines[i]).group(1)), ((7, 0.25, 1.0),)))}
            for i in range(s, e) if rx.match(lines[i])]


def parse_table1(lines):
    s, e = block(lines, '\\label{tab:investment_cost}', '\\end{table}')
    s = next(i for i in range(s - 40, s) if '\\begin{table}' in lines[i])
    rx = re.compile(r'&\s*(Low|Medium|High)\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)')
    rows = []
    for i in range(s, e + 1):
        m = rx.search(lines[i])
        if m:
            rows.append({'line': i + 1, 'trajectory': m.group(1), 'prob_pct': float(m.group(2)),
                         'v': {2025: float(m.group(3)), 2030: float(m.group(4)), 2035: float(m.group(5))}})
    return rows


# ------------------------------------------------------------------------------------------------------- costs
def read_cost_file(path):
    """Mirror of shared_energy_storage_data._get_operational_scenarios_info_from_excel_file /
    _get_investment_costs_from_excel_file (pandas, header=None): probabilities from row 0 cols 2.., unit costs from
    rows 1..n, year header in row 0."""
    _reg(path)
    with open(path, 'rb') as fh:
        data = fh.read()
    sc = pd.read_excel(io.BytesIO(data), sheet_name='Scenarios', header=None)
    n = int(sc.iloc[0, 1])
    probs = [float(sc.iloc[0, i + 2]) for i in range(n)]
    costs = {}
    for kind, sheet in (('power', 'Investment Cost, Power'), ('energy', 'Investment Cost, Energy')):
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None)
        costs[kind] = {}
        for j in range(1, len(df.columns)):
            y = int(df.iloc[0, j])
            costs[kind][y] = [float(df.iloc[i + 1, j]) for i in range(n)]
    return {'n': n, 'probs': probs, 'costs': costs}


def expected_unit(cf, kind, year, probs=None):
    probs = cf['probs'] if probs is None else probs
    return sum(p * c for p, c in zip(probs, cf['costs'][kind][year]))


def investment_cost(plan, cf, r, y0, probs=None):
    """THE PRESERVED FORMULA (C3):
        I(x) = sum_n (1 + r)**-(y - y0) * sum_c w_c * (cS_{c,y} * S_n + cE_{c,y} * E_n)
    one installation year y per plan (the case study's canonical form)."""
    year, nodes = canon(plan)
    if not nodes:
        return 0.0
    d = 1.0 / ((1.0 + r) ** (year - y0))
    return sum(d * (expected_unit(cf, 'power', year, probs) * s + expected_unit(cf, 'energy', year, probs) * e)
               for _n, s, e in nodes)


def rules_ok(plan, e_max, phi_min, phi_max):
    _y, nodes = canon(plan)
    bad = []
    for n, s, e in nodes:
        if s <= 0 or e <= 0:
            bad.append(f'node {n}: S = 0 xor E = 0')
        elif not (phi_min * s - 1e-9 <= e <= phi_max * s + 1e-9):
            bad.append(f'node {n}: E/S = {e / s:g} outside [{phi_min:g}, {phi_max:g}]')
        if e > e_max + 1e-9:
            bad.append(f'node {n}: E = {e:g} > E_max {e_max:g}')
    return bad


def verdict(I, B):
    adm = I <= B + FS_TOLERANCE
    if not adm:
        return 'INADMISSIBLE'
    return 'ADMISSIBLE (BOUNDARY)' if (B - I) < BOUNDARY_FRACTION * B else 'ADMISSIBLE'


# lattice / box enumeration (W173's box: z = (zP5, zE5, zP7, zE7, zP9, zE9, zY), dS = 0.25, dE = 0.5)
NODES = (5, 7, 9)
YEARS_SRP1 = (2025, 2030, 2035)


def z_to_plan(z):
    nodes = tuple((NODES[i], z[2 * i] * 0.25, z[2 * i + 1] * 0.5) for i in range(3))
    return canon((YEARS_SRP1[z[6]], nodes))


def z_admissible(z, cf, r, y0, B, e_max_lu=10):
    if z[6] not in (0, 1, 2):
        return False
    for i in range(3):
        p, e = z[2 * i], z[2 * i + 1]
        if p < 0 or e < 0 or (p == 0) != (e == 0):
            return False
        if p > 0 and not (p <= e <= 2 * p):
            return False
        if e > e_max_lu:
            return False
    return investment_cost(z_to_plan(z), cf, r, y0) <= B + FS_TOLERANCE


def box(z_inc, cf, r, y0, B):
    feasible, budget_rejected = {}, {}
    inc = z_to_plan(z_inc)
    for d in itertools.product((-1, 0, 1), repeat=7):
        z = tuple(a + b for a, b in zip(z_inc, d))
        if z == tuple(z_inc) or z[6] not in (0, 1, 2) or min(z[:6]) < 0:
            continue
        p = z_to_plan(z)
        if p == inc:
            continue
        rules = all(((z[2 * i] == 0) == (z[2 * i + 1] == 0)) and (z[2 * i] == 0 or z[2 * i] <= z[2 * i + 1] <= 2 * z[2 * i])
                    and z[2 * i + 1] <= 10 for i in range(3))
        if not rules:
            continue
        I = investment_cost(p, cf, r, y0)
        (feasible if I <= B + FS_TOLERANCE else budget_rejected)[p] = I
    return feasible, budget_rejected


# ------------------------------------------------------------------------------------------------------- C1
def c1(srcs, anchors, consts, case_flags, records, ms):
    stmts = objective_rule_terms(srcs)
    rule_attrs_found = sorted({a for s in stmts for a in s['model_attrs']})
    declared = sorted({a for t in TERMS for a in t['rule_attrs']} | {'total_load_curt_penalty', 'total_flex_penalty'})
    checks = [_ck('objective_function_rule term set (AST) == declared term set (incl. the congestion-management branch)',
                  rule_attrs_found == declared, {'found': rule_attrs_found, 'declared': declared})]
    # ADMM-only terms must not appear in objective_function_rule
    admm_attrs = {'dual_vmag_req', 'rho_v', 'rho_pf', 'rho_ess', 'prox_gamma_v', 'prox_gamma_pf', 'prox_gamma_ess',
                  'dual_pf_p_req', 'dual_ess_p_req', 'admm_objective'}
    checks.append(_ck('no ADMM attribute in objective_function_rule', not (admm_attrs & set(rule_attrs_found))))
    terms = []
    for t in TERMS:
        terms.append(dict(t, code=[{'id': a, 'file': anchors[a]['file'], 'lines': anchors[a]['lines']} for a in t['anchors']]))
    rec_checks, rec_values = records_c1(records)
    return {'objective_rule_statements': stmts, 'terms': terms, 'constants_definitions_py': consts,
            'case_flags': case_flags, 'retired_helpers': retired_defs(srcs), 'records': rec_values,
            'manuscript': {k: v for k, v in ms.items() if k.startswith('C1_')}, 'checks': checks + rec_checks}


def records_c1(records):
    checks, vals = [], {}
    for label, rec in records.items():
        rc = rec['recourse_components']
        cdt = rec['component_decomposition_totals_weighted']
        gross = rc['gross_operational_cost']
        recomputed = rc['gross_operational_cost_including_settlement'] - rc['interface_settlement_contracted_total'] - rc['voltage_pin_total']
        net_re = gross - rc['terminal_salvage_value']
        ms = rec.get('multiscenario_terminal') or {}
        row18 = ((ms.get('all_dso') or {}).get('row18_charge_weighted')) if isinstance(ms, dict) else None
        parts = {k: cdt.get(k, 0.0) for k in ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost',
                                               'res_curtailment_penalty', 'ess_usage_cost', 'detector_penalty_total')}
        closure = gross - sum(parts.values()) - rc['interface_settlement_deviation_total'] - (row18 or 0.0)
        vals[label] = {'candidate_canonical': rec['candidate_canonical'], 'candidate_key': rec['candidate_key'],
                       'eval_key': rec['eval_key'], 'cycles_run': rec['cycles_run'],
                       'gross_operational_cost': gross, 'including_settlement': rc['gross_operational_cost_including_settlement'],
                       'contracted_total': rc['interface_settlement_contracted_total'],
                       'deviation_total': rc['interface_settlement_deviation_total'],
                       'voltage_pin_total': rc['voltage_pin_total'], 'terminal_salvage_value': rc['terminal_salvage_value'],
                       'net_operational_recourse': rc['net_operational_recourse'], 'row18_charge_weighted': row18,
                       'decomposition_parts': parts, 'decomposition_closure_residual': closure,
                       'decomposition_closure_rel': closure / gross}
        checks.append(_ck(f'{label}: gross == including_settlement - contracted - voltage_pin',
                          _rel(gross, recomputed) <= 1e-12, gross - recomputed))
        checks.append(_ck(f'{label}: net == gross - salvage', abs(rc['net_operational_recourse'] - net_re) <= 1e-6 * abs(gross),
                          rc['net_operational_recourse'] - net_re))
        checks.append(_ck(f'{label}: decomposition closes within 1e-6 relative', abs(closure) <= 1e-6 * abs(gross), closure))
        if label.startswith('srp1'):
            checks.append(_ck(f'{label}: deviation settlement == 0 and voltage pin == 0 (one scenario)',
                              rc['interface_settlement_deviation_total'] == 0.0 and rc['voltage_pin_total'] == 0.0))
            checks.append(_ck(f'{label}: no row 18 charge recorded', row18 is None))
        else:
            checks.append(_ck(f'{label}: deviation settlement != 0 (3 x 3)', rc['interface_settlement_deviation_total'] != 0.0,
                              rc['interface_settlement_deviation_total']))
            checks.append(_ck(f'{label}: row 18 charge recorded > 0', (row18 or 0.0) > 0.0, row18))
    return checks, vals


# ------------------------------------------------------------------------------------------------------- C2
def k_from_params(ess):
    cal = ess['ageing']['calibration']
    return cal['cycles_n'] * cal['reference_dod_d'] / (-math.log(cal['eol_retention_r']))


def c2(case_srp1, case_3x3, ess, cf, sidecars, ms, anchors, wb):
    out = {}
    for label, case in (('SRP1', case_srp1), ('3x3', case_3x3)):
        years = [int(y) for y in case['Years']]
        widths = [case['Years'][str(y)] for y in years]
        y0, total = years[0], sum(widths)
        last = years[-1]
        out[label] = {'Years': case['Years'], 'y0': y0, 'sum_Y': total, 'terminal_date_code': y0 + total,
                      'last_block': {'year': last, 'width': widths[-1], 'covers': f'{last}..{last + widths[-1] - 1}',
                                     'end_boundary': last + widths[-1]},
                      'terminal_discount': 1.0 / ((1.0 + case['DiscountFactor']) ** total),
                      'all_blocks_equal_width': len(set(widths)) == 1,
                      'blocks_contiguous': all(years[i] + widths[i] == years[i + 1] for i in range(len(years) - 1)),
                      'calendar_life_blocks_of_a_y0_cohort': round(ess['ageing']['calendar_life_years'] / widths[0]),
                      'remaining_life_fraction_by_install_year': {
                          y: max(ess['ageing']['calendar_life_years'] - (y0 + total - y), 0) / ess['ageing']['calendar_life_years']
                          for y in years}}
    checks = [_ck(f'{k}: terminal date = y0 + sum Y_y = 2040 = end boundary of the last block',
                  v['terminal_date_code'] == 2040 == v['last_block']['end_boundary']) for k, v in out.items()]
    checks += [_ck(f'{k}: blocks contiguous and of equal width (the D row reads the investment block width)',
                   v['blocks_contiguous'] and v['all_blocks_equal_width']) for k, v in out.items()]
    k = k_from_params(ess)
    phi = ess['ageing']['calendar_retention_per_year']
    soh_min = ess['ageing']['minimum_soh']
    t_cal = ess['ageing']['calendar_life_years']
    num = {'k': k, 'phi_cal': phi, 'soh_min': soh_min, 't_cal': t_cal, 'cells': {}}
    for label, sc in sidecars.items():
        case = case_srp1 if label.startswith('srp1') else case_3x3
        years = [int(y) for y in case['Years']]
        widths = [case['Years'][str(y)] for y in years]
        r, y0 = case['DiscountFactor'], years[0]
        plan = plan_from_canonical(sc['candidate_canonical'])
        y_inv_year, nodes = plan
        node, S, E = nodes[0]
        y_inv = years.index(y_inv_year)
        ent = {(int(e['y_inv']), int(e['y'])): e for e in sc['soh_last']['entries'] if e['node_id'] == node}
        chain, soh_prev = [], 1.0
        n = widths[y_inv]   # the D row and phi**n read the INVESTMENT block's width (anchor num_years_inv_block)
        for y in range(y_inv, len(years)):
            e = ent[(y_inv, y)]
            soh_re = soh_prev * math.exp(-365.0 * n * e['efc_per_day'] / k) * phi ** n
            chain.append({'block': years[y], 'efc_per_day': e['efc_per_day'], 'soh_recorded': e['es_soh_per_unit_cumul'],
                          'soh_recomputed_from_efc': soh_re, 'rel_diff': _rel(soh_re, e['es_soh_per_unit_cumul']),
                          'floor_active': e['active']})
            soh_prev = e['es_soh_per_unit_cumul']
        soh_term = chain[-1]['soh_recorded']
        y_end = y0 + sum(widths)
        t_rem = max(0, y_inv_year + t_cal - y_end)
        cbar = expected_unit(cf, 'energy', y_inv_year)
        sal_re = ((1.0 + r) ** -(y_end - y0)) * cbar * E * (t_rem / t_cal) * (soh_term - soh_min) / (1.0 - soh_min) \
            if t_rem > 0 else 0.0
        rec_sal = sc['per_cycle_last']['terminal_salvage_value']
        num['cells'][label] = {'eval_dir': sc['eval_dir'], 'candidate_canonical': sc['candidate_canonical'],
                               'candidate_key': sc['candidate_key'], 'eval_key': sc['eval_key'],
                               'cycle': sc['per_cycle_last']['cycle'], 'certification_cycle': sc['certification_cycle'],
                               'plan': plan_label(plan), 'y_end': y_end, 'T_rem': t_rem, 'cbarE_install_year': cbar,
                               'chain': chain, 'terminal_soh': soh_term, 'salvage_recomputed': sal_re,
                               'salvage_recorded': rec_sal, 'abs_diff': sal_re - rec_sal}
        checks.append(_ck(f'{label}: sidecar and per-cycle record end at the same cycle',
                          sc['soh_last']['cycle'] == sc['per_cycle_last']['cycle'], (sc['soh_last']['cycle'], sc['per_cycle_last']['cycle'])))
        checks.append(_ck(f'{label}: SoH chain reproduced from recorded EFC/day (max rel diff <= 1e-6)',
                          max(c['rel_diff'] for c in chain) <= 1e-6, max(c['rel_diff'] for c in chain)))
        checks.append(_ck(f'{label}: salvage reproduced from the terminal (end of last block) SoH, |diff| <= 1e-6 EUR',
                          abs(sal_re - rec_sal) <= 1e-6, sal_re - rec_sal))
    wbchk = {}
    if wb is not None:
        wbchk = wb
        checks.append(_ck('3x3 workbook (uncommitted): terminal valuation date 2040 for every cohort',
                          set(wb['terminal_valuation_dates']) == {2040}, wb['terminal_valuation_dates']))
        checks.append(_ck('3x3 workbook: terminal discount factor = 1.02**-15',
                          abs(wb['terminal_discount_factor'] - out['3x3']['terminal_discount']) <= 1e-12,
                          wb['terminal_discount_factor']))
        checks.append(_ck('3x3 workbook: expected installation energy cost per year = cost-file expectation',
                          all(abs(v - expected_unit(cf, 'energy', y)) <= 1e-6 for y, v in wb['expected_energy_cost_by_year'].items()),
                          wb['expected_energy_cost_by_year']))
    return {'horizon': out, 'numeric': num, 'workbook_3x3': wbchk,
            'code': {a: anchors[a] for a in ('term_years', 'term_disc', 'terminal_horizon', 'terminal_date', 'salv_terminal_idx',
                                             'salv_e_rated', 'salv_e_avail', 'salv_mid_uses_end', 'salv_usable',
                                             'salv_cost_basis', 'age_at_terminal', 'remaining_life', 'remaining_frac',
                                             'soh_recursion', 'd_row', 'num_years_inv_block', 'soh_floor',
                                             'terminal_soh_report')},
            'manuscript': {k: v for k, v in ms.items() if k.startswith('C2_')},
            'main_tex_mentions_of_2040': None, 'checks': checks}


def read_workbook_3x3(path):
    if not os.path.exists(path):
        return None
    _reg(path)
    with open(path, 'rb') as fh:
        data = fh.read()
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb['Terminal Salvage']
    rows = [r for r in ws.iter_rows(values_only=True)]
    head = {r[0]: r[1] for r in rows if r and isinstance(r[0], str) and r[1] is not None and r[2] is None}
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == 'Node ID')
    hdr = list(rows[hdr_i])
    cohorts = [dict(zip(hdr, r)) for r in rows[hdr_i + 1:] if r and r[0] is not None]
    return {'terminal_horizon_years': head.get('Terminal Horizon, [years]'),
            'terminal_discount_factor': head.get('Terminal Discount Factor'),
            'terminal_valuation_dates': sorted({c['Terminal Valuation Date'] for c in cohorts}),
            'terminal_representative_years': sorted({c['Terminal Representative Year'] for c in cohorts}),
            'expected_energy_cost_by_year': {int(c['Investment Year']): c['Expected Installation Energy Cost, [m.u./MVAh]']
                                             for c in cohorts},
            'node7_2025': next(({k: c[k] for k in ('Age at Terminal Date, [years]', 'Remaining Calendar Life, [years]',
                                                    'Terminal SoH, [p.u.]', 'Terminal Salvage Value, [NPV m.u.]')}
                                for c in cohorts if c['Node ID'] == 7 and c['Investment Year'] == 2025), None)}


# ------------------------------------------------------------------------------------------------------- C3
def c3(lines, cf, case_srp1, ess, ms, w2, frozen, audit_lines):
    r, y0 = case_srp1['DiscountFactor'], int(next(iter(case_srp1['Years'])))
    B = float(ess['budget'])
    e_max, phi_min, phi_max = ess['max_capacity'], ess['min_energy_to_power_factor'], ess['max_energy_to_power_factor']
    checks = []
    # unit costs
    unit = {y: {'power': expected_unit(cf, 'power', y), 'energy': expected_unit(cf, 'energy', y),
                'discount': 1.0 / ((1.0 + r) ** (y - y0))} for y in (2025, 2028, 2030, 2031, 2034, 2035, 2037)}
    w2u = w2['expected_unit_costs_per_case_year']['new']
    for y in ('2025', '2030', '2035'):
        checks.append(_ck(f'unit costs {y} == W2 table (production master) within 1e-6',
                          abs(unit[int(y)]['power'] - w2u[y]['power_eur_per_mva_undiscounted']) <= 1e-6
                          and abs(unit[int(y)]['energy'] - w2u[y]['energy_eur_per_mwh_undiscounted']) <= 1e-6
                          and abs(unit[int(y)]['discount'] - w2u[y]['discount_multiplier']) <= 1e-15))
    # Table 1
    t1 = parse_table1(lines)
    t1_out = []
    for idx, row in enumerate(t1):
        kind = 'power' if idx < 3 else 'energy'
        c = idx % 3
        file_vals = {y: cf['costs'][kind][y][c] / 1000.0 for y in (2025, 2030, 2035)}
        match = all(f'{row["v"][y]:.2f}' == f'{file_vals[y]:.2f}' for y in (2025, 2030, 2035))
        pmatch = abs(row['prob_pct'] / 100.0 - cf['probs'][c]) <= 1e-12
        t1_out.append({'line': row['line'], 'kind': kind, 'trajectory': row['trajectory'], 'printed': row['v'],
                       'file_keur': file_vals, 'match_2dp': match, 'prob_printed_pct': row['prob_pct'],
                       'prob_file': cf['probs'][c], 'prob_match': pmatch})
    checks.append(_ck('Table 1: 6 rows parsed, every cell = file value rounded to 0.01 k EUR, every probability = file',
                      len(t1) == 6 and all(x['match_2dp'] and x['prob_match'] for x in t1_out)))
    # Table-1-rounded unit costs (robustness of the verdicts to the printed precision)
    t1u = {}
    for y in (2025, 2030, 2035):
        t1u[y] = {'power': sum(t1[i]['prob_pct'] / 100.0 * t1[i]['v'][y] * 1000.0 for i in range(3)),
                  'energy': sum(t1[3 + i]['prob_pct'] / 100.0 * t1[3 + i]['v'][y] * 1000.0 for i in range(3))}

    def I_t1(plan):
        year, nodes = canon(plan)
        if not nodes:
            return 0.0
        d = 1.0 / ((1.0 + r) ** (year - y0))
        return sum(d * (t1u[year]['power'] * s + t1u[year]['energy'] * e) for _n, s, e in nodes)

    # printed figures
    printed = {
        'expected_2025_text': {'power_keur_2dp': f'{unit[2025]["power"] / 1000:.2f}', 'energy_keur_2dp': f'{unit[2025]["energy"] / 1000:.2f}',
                               'printed': ('256.32', '253.88')},
        'p_fit_eur_10': round(unit[2025]['power'], -1), 'e_baseline_eur_10': round(unit[2025]['energy'], -1),
    }
    checks.append(_ck('text l. "probability-weighted unit costs in 2025" = file (2 dp k EUR)',
                      (printed['expected_2025_text']['power_keur_2dp'], printed['expected_2025_text']['energy_keur_2dp']) == ('256.32', '253.88')))
    checks.append(_ck('p = 256,320 and 253,880 EUR = file rounded to 10 EUR',
                      printed['p_fit_eur_10'] == 256320 and printed['e_baseline_eur_10'] == 253880))
    # plan inventory
    ev = parse_evaluations(lines)
    claims = parse_claims_evals(lines)
    gap = parse_gap(lines)
    nb = parse_neighbours(lines)
    vl = parse_value_ladder(lines)
    el = parse_energy_ladder_m2(lines)
    yl = parse_year_ladder(lines)
    checks.append(_ck('Table E.21: 61 rows numbered 1..61', sorted(ev) == list(range(1, 62)), len(ev)))
    checks.append(_ck('claims table: 58 rows, every Eval. pair in Table E.21',
                      len(claims) == 58 and all(a in ev and b in ev for c in claims.values() for a, b in [c['evals']]), len(claims)))
    checks.append(_ck('gap table rows agree with Table E.21 (plan and year)', len(gap) >= 1 and all(ev[g['eval']]['plan'] == g['plan'] for g in gap),
                      [(g['eval'], plan_label(g['plan'])) for g in gap]))
    checks.append(_ck('neighbours table rows agree with Table E.21', len(nb) == 8 and all(ev[n['eval']]['plan'] == n['plan'] for n in nb)))
    checks.append(_ck('value ladder: 6 rows', len(vl) == 6, [plan_label(v['plan']) for v in vl]))
    checks.append(_ck('energy ladder m2: 5 rows', len(el) == 5))
    checks.append(_ck('year ladder: 3 rows', len(yl) == 3))
    sources = {}

    def add(plan, src):
        p = canon(plan)
        if not p[1]:
            return
        sources.setdefault(p, []).append(src)

    for no, rrow in ev.items():
        add(rrow['plan'], f'Table E.21 row {no} (l. {rrow["line"]}, {rrow["condition"]})')
    for g in gap:
        add(g['plan'], f'gap table eval {g["eval"]} (l. {g["line"]})')
    for n in nb:
        add(n['plan'], f'neighbours table eval {n["eval"]} (l. {n["line"]})')
    for v in vl:
        add(v['plan'], f'Table value_ladder "{v["label"]}" (l. {v["line"]})')
    for e in el:
        add(e['plan'], f'Table energy_ladder_m2 (l. {e["line"]})')
    for y in yl:
        add(y['plan'], f'Table year_ladder (l. {y["line"]})')
    for no, c in claims.items():
        for evn in c['evals']:
            add(ev[evn]['plan'], f'claims table row {no} via eval {evn} (l. {c["line"]})')
    prose = {}
    for pid, needle, plans in PROSE_PLANS:
        hits = quote_lines(lines, needle)
        prose[pid] = {'lines': hits, 'text': needle, 'plans': [plan_label(p) for p in plans]}
        checks.append(_ck(f'prose anchor {pid} present', len(hits) >= 1, hits))
        for p in plans:
            add(p, f'prose {pid} (l. {hits})')
    # six search-only neighbours (named by the prose at C3_six_others)
    six = [canon((y, tuple((n, 0.25, 0.5) for n in ns))) for y, ns in
           ((2025, (5, 7)), (2025, (7, 9)), (2030, (5, 9)), (2030, (7, 9)), (2025, (5, 7, 9)), (2030, (5, 7, 9)))]
    for p in six:
        add(p, f'prose C3_six_others (l. {ms["C3_six_others"]["lines"]})')
    # evaluate every distinct plan
    w2_by_plan = {}
    for lab, c in w2['candidates'].items():
        cc = c.get('candidate_canonical')
        if cc:
            w2_by_plan.setdefault(plan_from_canonical(cc), []).append((lab, c['I_new_eur']))
    frozen_I = frozen_I_by_plan(frozen)
    inventory = []
    for p in sorted(sources, key=lambda q: (q[0], q[1])):
        I = investment_cost(p, cf, r, y0)
        It1 = I_t1(p) if p[0] in t1u else None
        year = p[0]
        stepE = unit[year]['discount'] * unit[year]['energy'] * 0.5
        rec = {'plan': plan_label(p), 'year': year, 'nodes': [list(n) for n in p[1]], 'I_eur': I, 'I_keur_1dp': round(I / 1000.0, 1),
               'slack_eur': B - I, 'slack_pct_of_B': 100.0 * (B - I) / B, 'verdict': verdict(I, B),
               'one_dE_step_fits': (B - I) >= stepE, 'rules_violated': rules_ok(p, e_max, phi_min, phi_max),
               'I_from_table1_rounded_eur': It1, 'verdict_from_table1': verdict(It1, B) if It1 is not None else None,
               'w2_records': [{'label': lab, 'I_new_eur': v, 'abs_diff': I - v} for lab, v in w2_by_plan.get(p, [])],
               'frozen_records': [{'claim_id': cid, 'field': fld, 'I': v, 'abs_diff': I - v} for cid, fld, v in frozen_I.get(p, [])],
               'sources': sources[p]}
        inventory.append(rec)
    n_w2 = sum(len(x['w2_records']) for x in inventory)
    n_fz = sum(len(x['frozen_records']) for x in inventory)
    checks.append(_ck('I(x) == W2 I_new_eur (production master value) within 1e-6 EUR wherever the plan is in W2',
                      n_w2 >= 1 and all(abs(w['abs_diff']) <= 1e-6 for x in inventory for w in x['w2_records']), n_w2))
    checks.append(_ck('I(x) == frozen Step-6 I_ref / I_other within 1e-6 EUR wherever recorded',
                      n_fz >= 1 and all(abs(f['abs_diff']) <= 1e-6 for x in inventory for f in x['frozen_records']), n_fz))
    checks.append(_ck('Table-1-rounded unit costs give the same verdict for every plan',
                      all(x['verdict_from_table1'] in (None, x['verdict']) for x in inventory)))
    # families named by count
    z_ret = (1, 1, 4, 7, 0, 0, 1)
    fam = {}
    single = []
    for zp in range(0, 12):
        for ze in range(0, 12):
            if zp == 0 and ze == 0:
                continue
            p = canon((2025, ((7, zp * 0.25, ze * 0.5),)))
            if zp > 0 and zp <= ze <= 2 * zp and ze <= 10:
                single.append({'S': zp * 0.25, 'E': ze * 0.5, 'I_eur': investment_cost(p, cf, r, y0),
                               'admissible': investment_cost(p, cf, r, y0) <= B + FS_TOLERANCE})
    fam['single_node_2025'] = {'n_rule_feasible': len(single), 'n_admissible': sum(1 for s in single if s['admissible']),
                               'admissible': [f'{s["S"]:g}/{s["E"]:g}' for s in single if s['admissible']],
                               'over_budget': [f'{s["S"]:g}/{s["E"]:g} ({s["I_eur"] / 1000:.1f} k)' for s in single if not s['admissible']]}
    f0, r0 = box((0, 0, 0, 0, 0, 0, 0), cf, r, y0, B)
    f1, r1 = box(z_ret, cf, r, y0, B)
    fam['x0_box'] = {'n_admissible': len(f0), 'n_budget_rejected': len(r0), 'members': sorted(plan_label(p) for p in f0),
                     'max_I_eur': max(f0.values())}
    fam['returned_plan_box'] = {'incumbent': plan_label(z_to_plan(z_ret)), 'I_incumbent_eur': investment_cost(z_to_plan(z_ret), cf, r, y0),
                                'n_admissible': len(f1), 'n_budget_rejected': len(r1),
                                'n_admissible_by_year': {y: sum(1 for p in f1 if p[0] == y) for y in YEARS_SRP1},
                                'min_budget_rejected_I_eur': min(r1.values()) if r1 else None}
    checks.append(_ck('twelve admissible single-node plans in 2025 (text)', fam['single_node_2025']['n_admissible'] == 12,
                      fam['single_node_2025']['n_admissible']))
    checks.append(_ck('fourteen admissible neighbours of x = 0 (text)', len(f0) == 14, len(f0)))
    checks.append(_ck('61 admissible unit neighbours of the returned plan (text)', len(f1) == 61, len(f1)))
    checks.append(_ck('the six search-only neighbours are in the x = 0 box', all(p in f0 for p in six)))
    # audit
    aud = {}
    for k, needle in AUDIT_QUOTES.items():
        aud[k] = {'lines': quote_lines(audit_lines, needle), 'text': needle}
        checks.append(_ck(f'audit quote {k} present', len(aud[k]['lines']) >= 1, aud[k]['lines']))
    ladder = {E: investment_cost((2025, ((7, E / 4.0, float(E)),)), cf, r, y0) for E in range(1, 6)}
    ret = investment_cost((2030, ((5, 0.25, 0.5), (7, 1.0, 3.5))), cf, r, y0)
    aud_cmp = {'unit_power_keur': {'audit': AUDIT_VALUES['unit_power_keur'], 'file': unit[2025]['power'] / 1000.0,
                                   'diff_eur': AUDIT_VALUES['unit_power_keur'] * 1000.0 - unit[2025]['power']},
               'unit_energy_keur': {'audit': AUDIT_VALUES['unit_energy_keur'], 'file': unit[2025]['energy'] / 1000.0,
                                    'diff_eur': AUDIT_VALUES['unit_energy_keur'] * 1000.0 - unit[2025]['energy']},
               'ladder': {E: {'audit_keur': AUDIT_VALUES['ladder_keur'][E], 'file_keur_1dp': round(ladder[E] / 1000.0, 1),
                              'I_eur': ladder[E], 'match': round(ladder[E] / 1000.0, 1) == AUDIT_VALUES['ladder_keur'][E],
                              'verdict': verdict(ladder[E], B)} for E in ladder},
               'two_node_2030': {'audit_keur': AUDIT_VALUES['two_node_keur'], 'I_eur': ret, 'file_keur_1dp': round(ret / 1000.0, 1),
                                 'match': round(ret / 1000.0, 1) == AUDIT_VALUES['two_node_keur'], 'verdict': verdict(ret, B),
                                 'slack_eur': B - ret}}
    checks.append(_ck('audit: two-node 2030 plan = 992.3 k EUR (file, 0.1 k)', aud_cmp['two_node_2030']['match'], ret))
    checks.append(_ck('audit: the five 4 h ladder figures = file (0.1 k)', all(v['match'] for v in aud_cmp['ladder'].values())))
    # printed 318.0
    checks.append(_ck('printed investment cost of the reference unit 318.0 k EUR = file (0.1 k)',
                      round(investment_cost(REF, cf, r, y0) / 1000.0, 1) == 318.0))
    return {'budget': {'B_eur': B, 'source': ESS_PARAMS, 'key': 'budget', 'test': f'I(x) <= B + {FS_TOLERANCE}'},
            'r': r, 'y0': y0, 'E_max': e_max, 'phi': [phi_min, phi_max], 'cost_probs': cf['probs'],
            'unit_costs_by_year': unit, 'table1': t1_out, 'table1_expected': t1u, 'printed': printed,
            'inventory': inventory, 'n_distinct_plans': len(inventory),
            'n_admissible': sum(1 for x in inventory if x['verdict'].startswith('ADMISSIBLE')),
            'n_inadmissible': sum(1 for x in inventory if x['verdict'] == 'INADMISSIBLE'),
            'n_boundary': sum(1 for x in inventory if 'BOUNDARY' in x['verdict']),
            'families': fam, 'prose_anchors': prose, 'audit': {'quotes': aud, 'compare': aud_cmp},
            'manuscript': {k: v for k, v in ms.items() if k.startswith('C3_')}, 'checks': checks}


def frozen_I_by_plan(frozen):
    """Pair each frozen claim's recorded I with a plan without ambiguity: only claims with exactly two instances, one
    of them x = 0 and I_ref == 0, are used, and their I_other is assigned to the non-zero plan."""
    out = {}
    for cl in frozen['tables']['claims']:
        inst = cl.get('instance') or {}
        plans = [plan_from_canonical(c['candidate_canonical']) for c in inst.values()]
        nonzero = [p for p in plans if p[1]]
        if len(plans) == 2 and len(nonzero) == 1 and 'I_other' in cl and cl.get('I_ref') == 0.0:
            out.setdefault(nonzero[0], []).append((cl['claim_id'], 'I_other', cl['I_other']))
    return out


# ------------------------------------------------------------------------------------------------------- negative controls
def negative_controls(srcs, lines, cf, case_srp1, ess, sidecar_yl2035):
    out = []

    def rec(name, refused, detail=None):
        out.append({'control': name, 'refused': bool(refused), 'detail': detail})

    s2 = copy.copy(srcs)
    mch = dict(srcs['mch'])
    mch['lines'] = [ln.replace('obj += model.total_slack_penalties', 'obj += model.total_slack_penaltiez') for ln in mch['lines']]
    s2['mch'] = mch
    _a, chk = code_anchors(s2, [c for c in CODE if c[0] == 'obj_slacks'])
    rec('NC1 an altered objective line is not found', not all(c['ok'] for c in chk))
    alt = [ln.replace('settlement excluded; the net form', 'settlement included; the net form') for ln in lines]
    _q, qchk = ms_quotes(alt, {'C1_appE_blanket': MS_QUOTES['C1_appE_blanket']})
    rec('NC2 an altered manuscript quote is not found', not all(c['ok'] for c in qchk))
    r, y0 = case_srp1['DiscountFactor'], 2025
    ret = (2030, ((5, 0.25, 0.5), (7, 1.0, 3.5)))
    rec('NC3 budget 0.99e6 makes the returned two-node plan INADMISSIBLE',
        verdict(investment_cost(ret, cf, r, y0), 0.99e6) == 'INADMISSIBLE', investment_cost(ret, cf, r, y0))
    rec('NC4 a perturbed probability vector changes I(ref unit)',
        abs(investment_cost(REF, cf, r, y0, probs=[0.30, 0.60, 0.10]) - investment_cost(REF, cf, r, y0)) > 1.0)
    sc = copy.deepcopy(sidecar_yl2035)
    for e in sc['soh_last']['entries']:
        if e['node_id'] == 7 and e['es_soh_per_unit_cumul'] != 1.0:
            e['es_soh_per_unit_cumul'] -= 1e-3
    res = c2(case_srp1, case_srp1, ess, cf, {'srp1_yl_y2035': sc}, {}, ANCHORS_FOR_NC, None)
    rec('NC5 an altered recorded terminal SoH breaks the salvage reproduction',
        not all(c['ok'] for c in res['checks'] if 'salvage reproduced' in c['check']))
    rec('NC6 a rule-violating plan (7: 0.25/1.5, ratio 6) is flagged', bool(rules_ok((2025, ((7, 0.25, 1.5),)), 5.0, 2.0, 4.0)))
    return out


ANCHORS_FOR_NC = {}


# ------------------------------------------------------------------------------------------------------- main
def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def _sidecar(eval_dir, label):
    n_soh, soh_last = _read_jsonl_last(os.path.join(eval_dir, 'soh_floor_sidecar_baseline.jsonl'))
    n_pc, pc_last = _read_jsonl_last(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    er = _read_json(os.path.join(eval_dir, 'evaluation_record.json'))
    return {'eval_dir': eval_dir, 'label': label, 'soh_last': soh_last, 'per_cycle_last': pc_last, 'n_soh': n_soh, 'n_pc': n_pc,
            'candidate_canonical': er['candidate_canonical'], 'candidate_key': er['candidate_key'], 'eval_key': er['eval_key'],
            'certification_cycle': er.get('certification_cycle'), 'record': er}


def main():
    global ANCHORS_FOR_NC
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out_dir = args.out_dir
    started = datetime.now(timezone.utc).isoformat()
    entries = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else None
    if entries is None or any(e != 'launch.log' for e in entries):
        raise SystemExit(f'refusing [fresh-dir check]: {out_dir} must exist and hold nothing but launch.log (found {entries})')
    others = _other_copy_running()
    if others:
        raise SystemExit(f'refusing [single-copy check]: another copy is running: {others}')

    required = [MAIN_TEX, AUDIT, CASE_SRP1, CASE_3X3, ESS_PARAMS, COST_FILE, W2_TABLE, FROZEN] + list(PRODUCTION_FILES) + \
        list(NET_PARAMS.values()) + [os.path.join(d, f) for d in (EV_YL2035, EV_YL2030, EV_3X3_N7, EV_3X3_X0)
                                     for f in ('evaluation_record.json', 'per_cycle_record.jsonl', 'soh_floor_sidecar_baseline.jsonl')]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'refusing [capture-path check]: missing sources {missing}')
    status_req = {p: _tracked(p) for p in required}
    uncommitted = [p for p, s in status_req.items() if s == 'UNCOMMITTED']
    if uncommitted:
        raise SystemExit(f'refusing [committed-input check]: {uncommitted}')

    pins = {'cost_file': (_reg(COST_FILE)['sha256'] == COST_FILE_SHA), 'w2_table': (_reg(W2_TABLE)['sha256'] == W2_TABLE_SHA),
            'frozen': (_reg(FROZEN)['sha256'] == FROZEN_SHA), 'case_3x3': (_reg(CASE_3X3)['sha256'] == CASE_3X3_SHA)}
    if not all(pins.values()):
        raise SystemExit(f'refusing [pinned-hash check]: {pins}')

    ident, lines = identity()
    if not ident['ok']:
        raise SystemExit(f'refusing [identity check]: {[c for c in ident["checks"] if not c["ok"]]}')
    audit_lines = _read(AUDIT).split('\n')
    srcs = load_sources()
    anchors, anchor_checks = code_anchors(srcs)
    ANCHORS_FOR_NC = anchors
    if not all(c['ok'] for c in anchor_checks):
        raise SystemExit(f'refusing [code-anchor check]: {[c for c in anchor_checks if not c["ok"]]}')
    consts = def_constants(srcs)
    ms, ms_checks = ms_quotes(lines)

    case_srp1 = _read_json(CASE_SRP1)
    case_3x3 = _read_json(CASE_3X3)
    ess = _read_json(ESS_PARAMS)
    cf = read_cost_file(COST_FILE)
    w2 = _read_json(W2_TABLE)
    frozen = _read_json(FROZEN)
    net_params = {n: _read_json(p) for n, p in NET_PARAMS.items()}
    # generator / flexible-load scan over every year file the two configurations read
    years_all = sorted({int(y) for y in case_srp1['Years']} | {int(y) for y in case_3x3['Years']})
    scan = {}
    for n in NET_PARAMS:
        for y in years_all:
            d = _read_json(os.path.join('data', 'SRP1', n, f'{n}_{y}.json'))
            types = {}
            for g in d['generators']:
                types[g['type']] = types.get(g['type'], 0) + 1
            flex_buses = sorted(l['bus'] for l in d['loads'] if l.get('fl_reg'))
            scan[f'{n}_{y}'] = {'generator_types': types, 'n_flexible_loads': len(flex_buses),
                                'flexible_load_buses_if_tn': flex_buses if n == 'case9' else None}
    case_flags = {n: {'obj_type': p['obj_type'], 'l_curt': p['l_curt'], 'rg_curt': p['rg_curt'], 'fl_reg': p['fl_reg'],
                      'es_reg': p['es_reg'], 'slacks': p['slacks'],
                      'flexibility_day_balance_effective': True}
                  for n, p in net_params.items()}
    tn_flex_buses = sorted({b for k, v in scan.items() if k.startswith('case9') for b in v['flexible_load_buses_if_tn']})
    adn_nodes = sorted(dn['connection_node_id'] for dn in case_srp1['DistributionNetworks'])
    gen_types_dn = sorted({t for k, v in scan.items() if not k.startswith('case9') for t in v['generator_types']})
    gen_types_tn = sorted({t for k, v in scan.items() if k.startswith('case9') for t in v['generator_types']})
    case_checks = [
        _ck('l_curt is false in all four networks (load curtailment term identically 0, variables not built)',
            all(p['l_curt'] is False for p in net_params.values())),
        _ck('TN flexible loads sit only at the ADN interface buses (excluded from the flexibility cost)',
            tn_flex_buses == adn_nodes, {'tn_flex_buses': tn_flex_buses, 'adn_nodes': adn_nodes}),
        _ck('DN generators are REF + RES only (no controllable non-reference DN generator)',
            set(gen_types_dn) <= {'REF', 'WIND', 'PV'}, gen_types_dn),
        _ck('TN generators: CONV controllable, WIND/PV curtailable', set(gen_types_tn) == {'CONV', 'WIND', 'PV'}, gen_types_tn),
        _ck('both case files name the same ESS params file and cost file',
            case_srp1['SharedEnergyStorage'] == case_3x3['SharedEnergyStorage'], case_3x3['SharedEnergyStorage']),
        _ck('both case files: DiscountFactor 0.02', case_srp1['DiscountFactor'] == case_3x3['DiscountFactor'] == 0.02),
        _ck('ESS probabilities: the planning (market) vector is assigned first and then overwritten by the cost file '
            "'Scenarios' sheet (assignment line < read call line in _read_planning_problem)",
            anchors['ess_prob_from_planning']['line'] < anchors['ess_read_costfile']['line'],
            (anchors['ess_prob_from_planning']['line'], anchors['ess_read_costfile']['line'])),
        _ck('Q evaluates objective_function_rule, not admm_objective (network.py get_primal_value)',
            anchors['net_primal_value']['line'] is not None),
    ]
    sc_yl35 = _sidecar(EV_YL2035, 'srp1_yl_y2035')
    sc_yl30 = _sidecar(EV_YL2030, 'srp1_yl_y2030')
    sc_n7 = _sidecar(EV_3X3_N7, '3x3_n7_4h_e1')
    sc_x0 = _sidecar(EV_3X3_X0, '3x3_x0')
    records = {'srp1_yl_y2035': sc_yl35['record'], '3x3_x0': sc_x0['record'], '3x3_n7_4h_e1': sc_n7['record']}
    C1 = c1(srcs, anchors, consts, {'per_network': case_flags, 'generator_and_flex_scan': scan, 'checks': case_checks},
            records, ms)
    wb = read_workbook_3x3(WB_3X3_N7)
    sidecars = {k: {kk: vv for kk, vv in v.items() if kk != 'record'} for k, v in
                (('srp1_yl_y2030', sc_yl30), ('srp1_yl_y2035', sc_yl35), ('3x3_n7_4h_e1', sc_n7))}
    C2 = c2(case_srp1, case_3x3, ess, cf, sidecars, ms, anchors, wb)
    C2['main_tex_mentions_of_2040'] = quote_lines(lines, '2040')
    C2['main_tex_mentions_of_2039'] = quote_lines(lines, '2039')
    C2['checks'].append(_ck('main.tex states no numeric terminal date (no "2040" / "2039" anywhere in main.tex)',
                            not C2['main_tex_mentions_of_2040'] and not C2['main_tex_mentions_of_2039']))
    C3 = c3(lines, cf, case_srp1, ess, ms, w2, frozen, audit_lines)
    controls = negative_controls(srcs, lines, cf, case_srp1, ess, sidecars['srp1_yl_y2035'])
    controls_ok = all(c['refused'] for c in controls)

    all_checks = {'identity': ident['checks'], 'code_anchors': anchor_checks, 'manuscript_quotes': ms_checks,
                  'case_flags': case_checks, 'C1': C1['checks'], 'C2': C2['checks'], 'C3': C3['checks']}
    failed = {k: [c for c in v if not c['ok']] for k, v in all_checks.items()}
    failed = {k: v for k, v in failed.items() if v}

    repo_modules = {f[:-3] for f in os.listdir(REPO) if f.endswith('.py')} - {'gate_result_io', 'p513_solve_profile_guard',
                                                                               SCRIPT[:-3]}
    bad_mods = sorted(m for m in sys.modules if m.split('.')[0] in repo_modules)
    gfail = GUARD.verify(0)
    if bad_mods or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads'] or not controls_ok:
        raise SystemExit(f'harness fault: production modules {bad_mods}; guard {gfail}; pickle {PICKLE_COUNTS}; '
                         f'controls {[c for c in controls if not c["refused"]]}')
    result = {
        'schema': 'w182g_cost_horizon_budget/1',
        'stage': 'P5.15 W182g (Addendum 82 v2, W182 order part C: C1, C2, C3)',
        'started_utc': started, 'interpreter': sys.executable, 'identity': ident,
        'objective_convention': ('Q = gross_operational_cost = sum TSO/DSO get_primal_value (objective_function_rule, '
                                 'weighted Y_y D_d (1+r)^-(y-y0)) - contracted interface settlement - interface-voltage pin; '
                                 'net = Q - terminal salvage credit; F = I + Q. At one market scenario the deviation '
                                 'settlement is identically 0, so "settlement excluded" is exact there; at 3 x 3 the '
                                 'deviation settlement and the row 18 premium are INSIDE Q.'),
        'formula_I': ('I(x) = sum_n (1+r)^-(y-y0) * sum_c w_c * (cS_{c,y} S_n + cE_{c,y} E_n); verdict ADMISSIBLE iff '
                      f'I(x) <= B + {FS_TOLERANCE}; BOUNDARY iff admissible and B - I(x) < {BOUNDARY_FRACTION} B'),
        'code_anchors': anchors, 'C1': C1, 'C2': C2, 'C3': C3,
        'negative_controls': controls, 'negative_controls_all_refused': controls_ok,
        'failed_checks': failed, 'all_checks_ok': not failed,
        'guards': {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                   'pickle_counts': PICKLE_COUNTS, 'production_modules_imported': bad_mods},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    out_json = os.path.join(out_dir, 'w182g_cost_horizon_budget.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(result, indent=1, sort_keys=False, default=GRIO.json_default_item) + '\n')
    manifest = {'outputs': {out_json: _sha(out_json)}, 'script': {SCRIPT: _sha(SCRIPT)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')

    # ------------------------------------------------------------------------------------------------ console
    print(f'repo HEAD {ident["repo_head"]}; last production commit {ident["last_commit_touching_production_files_read"][:8]}; '
          f'clone HEAD {ident["clone_head"]}; main.tex sha256 {ident["main_sha256"]} ({ident["main_n_lines"]} lines)')
    print(f'code anchors {len(anchors)} found; manuscript quotes {sum(1 for c in ms_checks if c["ok"])}/{len(ms_checks)} present')
    print('--- C1 objective_function_rule statements (AST):')
    for s in C1['objective_rule_statements']:
        print(f'    l. {s["line"]} obj {s["op"]} {s["model_attrs"]}')
    for t in C1['terms']:
        print(f'  [{t["status"]:15s}] {t["term"]}  :: ' + ', '.join(f'{c["file"]}:{c["lines"]}' for c in t['code'][:4]))
    for k, v in C1['records'].items():
        print(f'  record {k}: Q {v["gross_operational_cost"]:.2f}; deviation {v["deviation_total"]:.2f}; row18 {v["row18_charge_weighted"]}; '
              f'pin {v["voltage_pin_total"]:.4f}; contracted {v["contracted_total"]:.2f}; salvage {v["terminal_salvage_value"]:.4f}; '
              f'closure {v["decomposition_closure_residual"]:.6g}')
    print('--- C2')
    for k, v in C2['horizon'].items():
        print(f'  {k}: Years {v["Years"]}; terminal date {v["terminal_date_code"]}; last block {v["last_block"]}; '
              f'disc {v["terminal_discount"]:.12f}; T_rem fraction {v["remaining_life_fraction_by_install_year"]}')
    for k, v in C2['numeric']['cells'].items():
        print(f'  {k}: {v["plan"]} cycle {v["cycle"]}; terminal SoH {v["terminal_soh"]:.10f}; T_rem {v["T_rem"]}; '
              f'salvage re {v["salvage_recomputed"]:.6f} rec {v["salvage_recorded"]:.6f} diff {v["abs_diff"]:.3g}; '
              f'chain max rel {max(c["rel_diff"] for c in v["chain"]):.3g}')
    print(f'  workbook 3x3: {C2["workbook_3x3"]}')
    print(f'  main.tex lines with 2040: {C2["main_tex_mentions_of_2040"]}; with 2039: {C2["main_tex_mentions_of_2039"]}')
    print('--- C3')
    print(f'  B = {C3["budget"]["B_eur"]}; r = {C3["r"]}; probs {C3["cost_probs"]}')
    for y, u in C3['unit_costs_by_year'].items():
        print(f'  {y}: cS {u["power"]:.6f} cE {u["energy"]:.6f} disc {u["discount"]:.10f}')
    for x in C3['inventory']:
        print(f'  {x["plan"]:42s} I {x["I_eur"]:14.2f}  slack {x["slack_eur"]:14.2f} ({x["slack_pct_of_B"]:7.2f} %)  {x["verdict"]:22s} '
              f'rules {x["rules_violated"] or "ok"}  W2 {len(x["w2_records"])} FZ {len(x["frozen_records"])}  n_src {len(x["sources"])}')
    print(f'  distinct {C3["n_distinct_plans"]}; admissible {C3["n_admissible"]} (boundary {C3["n_boundary"]}); '
          f'inadmissible {C3["n_inadmissible"]}')
    print(f'  families: {json.dumps(C3["families"], default=str)[:1500]}')
    print(f'  audit compare: {json.dumps(C3["audit"]["compare"], default=str)}')
    for c in controls:
        print(f'  control {c["control"]}: refused {c["refused"]}')
    print(f'failed checks: {json.dumps(failed, default=str)[:3000]}')
    print(f'inputs {len(INPUTS)} ({manifest["n_uncommitted_inputs"]} uncommitted)')
    print(f'guards: {result["guards"]}')
    print(f'output {out_json} sha256 {_sha(out_json)}')
    if failed:
        sys.exit(3)


if __name__ == '__main__':
    main()
