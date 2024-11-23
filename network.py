import os
import pandas as pd
import pyomo.opt as po
import pyomo.environ as pe
from math import acos, pi, tan, sqrt, atan2, isclose
import networkx as nx
import matplotlib.pyplot as plt
from node import Node
from load import Load
from branch import Branch
from generator import Generator
from energy_storage import EnergyStorage
from helper_functions import *


# ======================================================================================================================
#   Class NETWORK
# ======================================================================================================================
class Network:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.diagrams_dir = str()
        self.year = int()
        self.day = str()
        self.num_instants = 0
        self.operational_data_file = str()
        self.data_loaded = False
        self.is_transmission = False
        self.baseMVA = 100.0
        self.nodes = list()
        self.loads = list()
        self.branches = list()
        self.generators = list()
        self.energy_storages = list()
        self.shared_energy_storages = list()
        self.prob_market_scenarios = list()             # Probability of market (price) scenarios
        self.prob_operation_scenarios = list()          # Probability of operation (generation and consumption) scenarios
        self.cost_energy_p = list()
        self.cost_flex = list()
        self.active_distribution_network_nodes = list()

    def build_model(self, params):
        _pre_process_network(self)
        return _build_model(self, params)

    def run_smopf(self, model, params, from_warm_start=False):
        return _run_smopf(self, model, params, from_warm_start=from_warm_start)

    def get_primal_value(self, model):
        return pe.value(model.objective)

    def compute_objective_function_value(self, model, params):
        return _compute_objective_function_value(self, model, params)

    def get_reference_node_id(self):
        for node in self.nodes:
            if node.type == BUS_REF:
                return node.bus_i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_interface_branch_rating(self):
        if not self.is_transmission:
            ref_node_id = self.get_reference_node_id()
            interface_branch_rating = 0.00
            for branch in self.branches:
                if branch.fbus == ref_node_id or branch.tbus == ref_node_id:
                    interface_branch_rating += branch.rate
            return interface_branch_rating
        else:
            print(f'[ERROR] Network {self.name}. Function get_interface_branch_rating() NOT APPLICABLE to TRANSMISSION NETWORK.')
            exit(ERROR_NETWORK_FILE)

    def get_node_idx(self, node_id):
        for i in range(len(self.nodes)):
            if self.nodes[i].bus_i == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Bus ID {node_id} not found! Check network model.')
        exit(ERROR_NETWORK_FILE)

    def get_node_type(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.type
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_base_kv(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.base_kv
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_node_voltage_limits(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return node.v_min, node.v_max
        print(f'[ERROR] Network {self.name}. Node {node_id} not found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def node_exists(self, node_id):
        for node in self.nodes:
            if node.bus_i == node_id:
                return True
        return False

    def adn_load_exists(self, node_id):
        for load in self.loads:
            if load.bus == node_id:
                return True
        return False

    def get_adn_load_idx(self, node_id):
        for c in range(len(self.loads)):
            if self.loads[c].bus == node_id:
                return c
        print(f'[ERROR] Network {self.name}. Node ID {node_id} does not have an ADN! Check network model.')
        exit(ERROR_NETWORK_FILE)

    def get_branch_idx(self, branch):
        for b in range(len(self.branches)):
            if self.branches[b].branch_id == branch.branch_id:
                return b
        print(f'[ERROR] Network {self.name}. No Branch connecting bus {branch.fbus} and bus {branch.tbus} found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_reference_gen_idx(self):
        ref_node_id = self.get_reference_node_id()
        for i in range(len(self.generators)):
            gen = self.generators[i]
            if gen.bus == ref_node_id:
                return i
        print(f'[ERROR] Network {self.name}. No REF NODE found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_gen_idx(self, node_id):
        for g in range(len(self.generators)):
            gen = self.generators[g]
            if gen.bus == node_id:
                return g
        print(f'[ERROR] Network {self.name}. No Generator in bus {node_id} found! Check network.')
        exit(ERROR_NETWORK_FILE)

    def get_gen_type(self, gen_id):
        description = 'Unknown'
        for gen in self.generators:
            if gen.gen_id == gen_id:
                if gen.gen_type == GEN_REFERENCE:
                    description = 'Reference (TN)'
                elif gen.gen_type == GEN_CONV:
                    description = 'Conventional'
                elif gen.gen_type == GEN_RES_CONTROLLABLE:
                    description = 'RES (Generic, Controllable)'
                elif gen.gen_type == GEN_RES_SOLAR:
                    description = 'RES (Solar)'
                elif gen.gen_type == GEN_RES_WIND:
                    description = 'RES (Wind)'
                elif gen.gen_type == GEN_RES_OTHER:
                    description = 'RES (Generic, Non-controllable)'
                elif gen.gen_type == GEN_INTERCONNECTION:
                    description = 'Interconnection'
        return description

    def get_num_renewable_gens(self):
        num_renewable_gens = 0
        for generator in self.generators:
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                num_renewable_gens += 1
        return num_renewable_gens

    def has_energy_storage_device(self, node_id):
        for energy_storage in self.energy_storages:
            if energy_storage.bus == node_id:
                return True
        return False

    def get_shared_energy_storage_idx(self, node_id):
        for i in range(len(self.shared_energy_storages)):
            shared_energy_storage = self.shared_energy_storages[i]
            if shared_energy_storage.bus == node_id:
                return i
        print(f'[ERROR] Network {self.name}. Node {node_id} does not have a shared energy storage system! Check network.')
        exit(ERROR_NETWORK_FILE)

    def read_network_from_json_file(self):
        filename = os.path.join(self.data_dir, self.name, f'{self.name}_{self.year}.json')
        _read_network_from_json_file(self, filename)
        self.perform_network_check()

    def read_network_operational_data_from_file(self):
        filename = os.path.join(self.data_dir, self.name, self.operational_data_file)
        data = _read_network_operational_data_from_file(self, filename)
        _update_network_with_excel_data(self, data)

    def process_results(self, model, params, results=dict()):
        return _process_results(self, model, params, results=results)

    def process_results_interface(self, model):
        return _process_results_interface(self, model)

    def compute_series_admittance(self):
        for branch in self.branches:
            branch.g = branch.r / (branch.r ** 2 + branch.x ** 2)
            branch.b = -branch.x / (branch.r ** 2 + branch.x ** 2)

    def perform_network_check(self):
        _perform_network_check(self)

    def plot_diagram(self):
        _plot_networkx_diagram(self)


# ======================================================================================================================
#   NETWORK optimization functions
# ======================================================================================================================
def _build_model(network, params):

    network.compute_series_admittance()

    model = pe.ConcreteModel()
    model.name = network.name

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.periods = range(network.num_instants)
    model.scenarios_market = range(len(network.prob_market_scenarios))
    model.scenarios_operation = range(len(network.prob_operation_scenarios))
    model.nodes = range(len(network.nodes))
    model.loads = range(len(network.loads))
    model.generators = range(len(network.generators))
    model.branches = range(len(network.branches))
    model.energy_storages = range(len(network.energy_storages))
    model.shared_energy_storages = range(len(network.shared_energy_storages))

    # ------------------------------------------------------------------------------------------------------------------
    # Decision variables
    # - Voltage
    model.e = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    model.f = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.e_actual = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=1.0)
    model.f_actual = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    if params.slacks.grid_operation.voltage:
        model.slack_e = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.00)
        model.slack_f = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.00)
    for i in model.nodes:
        node = network.nodes[i]
        e_lb, e_ub = -node.v_max, node.v_max
        f_lb, f_ub = -node.v_max, node.v_max
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    if params.slacks.grid_operation.voltage:
                        model.slack_e[i, s_m, s_o, p].setub(VMAG_VIOLATION_ALLOWED * e_ub)
                        model.slack_e[i, s_m, s_o, p].setlb(VMAG_VIOLATION_ALLOWED * e_lb)
                        model.slack_f[i, s_m, s_o, p].setub(VMAG_VIOLATION_ALLOWED * f_ub)
                        model.slack_f[i, s_m, s_o, p].setlb(VMAG_VIOLATION_ALLOWED * f_lb)
                    if node.type == BUS_REF:
                        if network.is_transmission:
                            model.e[i, s_m, s_o, p].setub(e_ub)
                            model.e[i, s_m, s_o, p].setlb(e_lb)
                        else:
                            ref_gen_idx = network.get_gen_idx(node.bus_i)
                            vg = network.generators[ref_gen_idx].vg
                            model.e[i, s_m, s_o, p].setub(vg + SMALL_TOLERANCE)
                            model.e[i, s_m, s_o, p].setlb(vg - SMALL_TOLERANCE)
                            if params.slacks.grid_operation.voltage:
                                model.slack_e[i, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                model.slack_e[i, s_m, s_o, p].setlb(-SMALL_TOLERANCE)
                        model.f[i, s_m, s_o, p].setub(SMALL_TOLERANCE)
                        model.f[i, s_m, s_o, p].setlb(-SMALL_TOLERANCE)
                        if params.slacks.grid_operation.voltage:
                            model.slack_f[i, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.slack_f[i, s_m, s_o, p].setlb(-SMALL_TOLERANCE)
                    else:
                        model.e[i, s_m, s_o, p].setub(e_ub + SMALL_TOLERANCE)
                        model.e[i, s_m, s_o, p].setlb(e_lb - SMALL_TOLERANCE)
                        model.f[i, s_m, s_o, p].setub(f_ub + SMALL_TOLERANCE)
                        model.f[i, s_m, s_o, p].setlb(f_lb - SMALL_TOLERANCE)
    if params.slacks.node_balance:
        model.slack_node_balance_p_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_node_balance_p_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_node_balance_q_up = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.slack_node_balance_q_down = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)

    # - Generation
    model.pg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.qg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    for g in model.generators:
        gen = network.generators[g]
        pg_ub, pg_lb = gen.pmax, gen.pmin
        qg_ub, qg_lb = gen.qmax, gen.qmin
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    if gen.is_controllable():
                        if gen.status[p] == 1:
                            model.pg[g, s_m, s_o, p] = (pg_lb + pg_ub) * 0.50
                            model.qg[g, s_m, s_o, p] = (qg_lb + qg_ub) * 0.50
                            model.pg[g, s_m, s_o, p].setub(pg_ub)
                            model.pg[g, s_m, s_o, p].setlb(pg_lb)
                            model.qg[g, s_m, s_o, p].setub(qg_ub)
                            model.qg[g, s_m, s_o, p].setlb(qg_lb)
                        else:
                            model.pg[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.pg[g, s_m, s_o, p].setlb(-SMALL_TOLERANCE)
                            model.qg[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.qg[g, s_m, s_o, p].setlb(-SMALL_TOLERANCE)
                    else:
                        # Non-conventional generator
                        init_pg = 0.0
                        init_qg = 0.0
                        if gen.status[p] == 1:
                            init_pg = gen.pg[s_o][p]
                            init_qg = gen.qg[s_o][p]
                        model.pg[g, s_m, s_o, p].setub(init_pg + SMALL_TOLERANCE)
                        model.pg[g, s_m, s_o, p].setlb(init_pg - SMALL_TOLERANCE)
                        model.qg[g, s_m, s_o, p].setub(init_qg + SMALL_TOLERANCE)
                        model.qg[g, s_m, s_o, p].setlb(init_qg - SMALL_TOLERANCE)
    if params.rg_curt:
        model.pg_curt_down = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.pg_curt_up = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.qg_curt_down = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.qg_curt_up = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for g in model.generators:
            gen = network.generators[g]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if gen.is_controllable():
                            model.pg_curt_down[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.pg_curt_up[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.qg_curt_down[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.qg_curt_up[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                        else:
                            if gen.is_curtaillable():
                                # - Renewable Generation
                                init_pg = 0.0
                                init_qg = 0.0
                                if gen.status[p] == 1:
                                    init_pg = gen.pg[s_o][p]
                                    init_qg = gen.qg[s_o][p]
                                if init_pg >= 0.00:
                                    model.pg_curt_down[g, s_m, s_o, p].setub(init_pg + SMALL_TOLERANCE)
                                    model.pg_curt_up[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                else:
                                    model.pg_curt_down[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                    model.pg_curt_up[g, s_m, s_o, p].setub(abs(init_pg) + SMALL_TOLERANCE)
                                if init_qg >= 0.00:
                                    model.qg_curt_down[g, s_m, s_o, p].setub(init_qg + SMALL_TOLERANCE)
                                    model.qg_curt_up[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                else:
                                    model.qg_curt_down[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                    model.qg_curt_up[g, s_m, s_o, p].setub(abs(init_qg) + SMALL_TOLERANCE)
                            else:
                                # - Generator is not curtaillable (conventional RES, ref gen, etc.)
                                model.pg_curt_down[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                model.pg_curt_up[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                model.qg_curt_down[g, s_m, s_o, p].setub(SMALL_TOLERANCE)
                                model.qg_curt_up[g, s_m, s_o, p].setub(SMALL_TOLERANCE)

    # - Branch power flows (squared) -- used in branch limits
    model.flow_ij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.grid_operation.branch_flow:
        model.slack_flow_ij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    for b in model.branches:
        if not network.branches[b].status:
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.flow_ij_sqr[b, s_m, s_o, p].setub(SMALL_TOLERANCE)
                        if params.slacks.grid_operation.branch_flow:
                            model.slack_flow_ij_sqr[b, s_m, s_o, p].setub(SMALL_TOLERANCE)
        else:
            if params.slacks.grid_operation.branch_flow:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            rating = network.branches[b].rate / network.baseMVA
                            model.slack_flow_ij_sqr[b, s_m, s_o, p].setub(SIJ_VIOLATION_ALLOWED * rating)

    # - Loads
    model.pc = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals)
    model.qc = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals)
    for c in model.loads:
        load = network.loads[c]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.pc[c, s_m, s_o, p].setub(load.pd[s_o][p] + SMALL_TOLERANCE)
                    model.pc[c, s_m, s_o, p].setlb(load.pd[s_o][p] - SMALL_TOLERANCE)
                    model.qc[c, s_m, s_o, p].setub(load.qd[s_o][p] + SMALL_TOLERANCE)
                    model.qc[c, s_m, s_o, p].setlb(load.qd[s_o][p] - SMALL_TOLERANCE)
    if params.fl_reg:
        model.flex_p_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_p_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        if params.slacks.flexibility.day_balance:
            model.slack_flex_p_balance = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
        for c in model.loads:
            load = network.loads[c]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        if load.fl_reg:
                            flex_up = load.flexibility.upward[p]
                            flex_down = load.flexibility.downward[p]
                            model.flex_p_up[c, s_m, s_o, p].setub(abs(flex_up))
                            model.flex_p_down[c, s_m, s_o, p].setub(abs(flex_down))
                        else:
                            model.flex_p_up[c, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            model.flex_p_down[c, s_m, s_o, p].setub(SMALL_TOLERANCE)
                            if params.slacks.flexibility.day_balance:
                                model.slack_flex_p_balance[c, s_m, s_o].setub(SMALL_TOLERANCE)
                                model.slack_flex_p_balance[c, s_m, s_o].setlb(-SMALL_TOLERANCE)
    if params.l_curt:
        model.pc_curt_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.pc_curt_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        for c in model.loads:
            load = network.loads[c]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:

                        if load.pd[s_o][p] >= 0.00:
                            model.pc_curt_down[c, s_m, s_o, p].setub(abs(load.pd[s_o][p]) + SMALL_TOLERANCE)
                            model.pc_curt_up[c, s_m, s_o, p].setub(SMALL_TOLERANCE)
                        else:
                            model.pc_curt_up[c, s_m, s_o, p].setub(abs(load.pd[s_o][p]) + SMALL_TOLERANCE)
                            model.pc_curt_down[c, s_m, s_o, p].setub(SMALL_TOLERANCE)

                        if load.qd[s_o][p] >= 0.00:
                            model.qc_curt_down[c, s_m, s_o, p].setub(abs(load.qd[s_o][p]) + SMALL_TOLERANCE)
                            model.qc_curt_up[c, s_m, s_o, p].setub(SMALL_TOLERANCE)
                        else:
                            model.qc_curt_up[c, s_m, s_o, p].setub(abs(load.qd[s_o][p]) + SMALL_TOLERANCE)
                            model.qc_curt_down[c, s_m, s_o, p].setub(SMALL_TOLERANCE)

    # - Transformers
    model.r = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=1.0)
    for i in model.branches:
        branch = network.branches[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    if branch.is_transformer:
                        # - Transformer
                        if params.transf_reg and branch.vmag_reg:
                            model.r[i, s_m, s_o, p].setub(TRANSFORMER_MAXIMUM_RATIO)
                            model.r[i, s_m, s_o, p].setlb(TRANSFORMER_MINIMUM_RATIO)
                        else:
                            model.r[i, s_m, s_o, p].setub(branch.ratio + SMALL_TOLERANCE)
                            model.r[i, s_m, s_o, p].setlb(branch.ratio - SMALL_TOLERANCE)
                    else:
                        model.r[i, s_m, s_o, p].setub(1.00 + SMALL_TOLERANCE)
                        model.r[i, s_m, s_o, p].setlb(1.00 - SMALL_TOLERANCE)

    # - Energy Storage devices
    if params.es_reg:
        model.es_soc = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_sch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_qch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
        model.es_sdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_qdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
        for e in model.energy_storages:
            energy_storage = network.energy_storages[e]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        model.es_soc[e, s_m, s_o, p] = energy_storage.e_init
                        model.es_soc[e, s_m, s_o, p].setlb(energy_storage.e_min)
                        model.es_soc[e, s_m, s_o, p].setub(energy_storage.e_max)
                        model.es_sch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_qch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_qch[e, s_m, s_o, p].setlb(-energy_storage.s)
                        model.es_sdch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_pdch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_qdch[e, s_m, s_o, p].setub(energy_storage.s)
                        model.es_qdch[e, s_m, s_o, p].setlb(-energy_storage.s)

        if params.slacks.ess.complementarity:
            model.slack_es_comp = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        if params.slacks.ess.charging:
            model.slack_es_sch_up = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.slack_es_sch_down = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.slack_es_sdch_up = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.slack_es_sdch_down = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        if params.slacks.ess.soc:
            model.slack_es_soc_up = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
            model.slack_es_soc_down = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        if params.slacks.ess.day_balance:
            model.slack_es_soc_final_up = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
            model.slack_es_soc_final_down = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)

    # - Shared Energy Storage devices
    model.shared_es_s_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)
    model.shared_es_e_rated = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)
    model.shared_es_s_rated_fixed = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)          # Benders' -- used to get the dual variables (sensitivities)
    model.shared_es_e_rated_fixed = pe.Var(model.shared_energy_storages, domain=pe.NonNegativeReals, initialize=0.00)          # (...)
    model.shared_es_soc = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.00)
    model.shared_es_sch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_pch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_qch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_sdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_pdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    model.shared_es_qdch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_pnet = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    model.shared_es_qnet = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.Reals, initialize=0.0)
    for e in model.shared_energy_storages:
        shared_energy_storage = network.shared_energy_storages[e]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    model.shared_es_soc[e, s_m, s_o, p] = shared_energy_storage.e * ENERGY_STORAGE_RELATIVE_INIT_SOC
    if params.slacks.shared_ess.complementarity:
        model.slack_shared_es_comp = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.shared_ess.charging:
        model.slack_shared_es_sch_up = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.slack_shared_es_sch_down = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.slack_shared_es_sdch_up = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.slack_shared_es_sdch_down = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.shared_ess.soc:
        model.slack_shared_es_soc_up = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
        model.slack_shared_es_soc_down = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, model.periods, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.shared_ess.day_balance:
        model.slack_shared_es_soc_final_up = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.slack_shared_es_soc_final_down = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)

    # Costs (penalties)
    # Note: defined as variables (bus fixed) so that they can be changed later, if needed
    model.penalty_ess_usage = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_ess_usage.fix(PENALTY_ESS_USAGE)
    if params.obj_type == OBJ_MIN_COST:
        model.cost_res_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.cost_load_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.cost_res_curtailment.fix(COST_GENERATION_CURTAILMENT)
        model.cost_load_curtailment.fix(COST_CONSUMPTION_CURTAILMENT)
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        model.penalty_gen_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.penalty_load_curtailment = pe.Var(domain=pe.NonNegativeReals)
        model.penalty_flex_usage = pe.Var(domain=pe.NonNegativeReals)
        model.penalty_gen_curtailment.fix(PENALTY_GENERATION_CURTAILMENT)
        model.penalty_load_curtailment.fix(PENALTY_LOAD_CURTAILMENT)
        model.penalty_flex_usage.fix(PENALTY_FLEXIBILITY_USAGE)
    else:
        print(f'[ERROR] Unrecognized or invalid objective. Objective = {params.obj_type}. Exiting...')
        exit(ERROR_NETWORK_MODEL)

    # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Voltage
    model.voltage_cons = pe.ConstraintList()
    for i in model.nodes:
        node = network.nodes[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    # e_actual and f_actual definition
                    e_actual = model.e[i, s_m, s_o, p]
                    f_actual = model.f[i, s_m, s_o, p]
                    if params.slacks.grid_operation.voltage:
                        e_actual += model.slack_e[i, s_m, s_o, p]
                        f_actual += model.slack_f[i, s_m, s_o, p]

                    if params.relax_equalities:
                        model.voltage_cons.add(model.e_actual[i, s_m, s_o, p] <= e_actual + EQUALITY_TOLERANCE)
                        model.voltage_cons.add(model.e_actual[i, s_m, s_o, p] >= e_actual - EQUALITY_TOLERANCE)
                        model.voltage_cons.add(model.f_actual[i, s_m, s_o, p] <= f_actual + EQUALITY_TOLERANCE)
                        model.voltage_cons.add(model.f_actual[i, s_m, s_o, p] >= f_actual - EQUALITY_TOLERANCE)
                    else:
                        model.voltage_cons.add(model.e_actual[i, s_m, s_o, p] == e_actual)
                        model.voltage_cons.add(model.f_actual[i, s_m, s_o, p] == f_actual)

                    # voltage magnitude constraints
                    if node.type == BUS_PV:
                        if params.enforce_vg:
                            # - Enforce voltage controlled bus
                            gen_idx = network.get_gen_idx(node.bus_i)
                            vg = network.generators[gen_idx].vg
                            e = model.e[i, s_m, s_o, p]
                            f = model.f[i, s_m, s_o, p]
                            if params.relax_equalities:
                                model.voltage_cons.add(e ** 2 + f ** 2 <= vg[p] ** 2 + EQUALITY_TOLERANCE)
                                model.voltage_cons.add(e ** 2 + f ** 2 >= vg[p] ** 2 - EQUALITY_TOLERANCE)
                            else:
                                model.voltage_cons.add(e ** 2 + f ** 2 == vg[p] ** 2)
                        else:
                            # - Voltage at the bus is not controlled
                            e = model.e[i, s_m, s_o, p]
                            f = model.f[i, s_m, s_o, p]
                            model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min**2)
                            model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max**2)
                    else:
                        e = model.e[i, s_m, s_o, p]
                        f = model.f[i, s_m, s_o, p]
                        model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min**2)
                        model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max**2)

    # - Flexible Loads -- Daily energy balance
    if params.fl_reg:
        model.fl_p_balance = pe.ConstraintList()
        for c in model.loads:
            if network.loads[c].fl_reg:
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        p_up, p_down = 0.0, 0.0
                        for p in model.periods:
                            p_up += model.flex_p_up[c, s_m, s_o, p]
                            p_down += model.flex_p_down[c, s_m, s_o, p]
                        if params.slacks.flexibility.day_balance:
                            model.fl_p_balance.add(p_up == p_down + model.slack_flex_p_balance[c, s_m, s_o])
                        else:
                            if params.relax_equalities:
                                model.fl_p_balance.add(p_up <= p_down + EQUALITY_TOLERANCE)
                                model.fl_p_balance.add(p_up >= p_down - EQUALITY_TOLERANCE)
                            else:
                                model.fl_p_balance.add(p_up == p_down)

    # - Energy Storage constraints
    if params.es_reg:

        model.energy_storage_balance = pe.ConstraintList()
        model.energy_storage_operation = pe.ConstraintList()
        model.energy_storage_day_balance = pe.ConstraintList()
        model.energy_storage_ch_dch_exclusion = pe.ConstraintList()

        for e in model.energy_storages:

            energy_storage = network.energy_storages[e]
            soc_init = energy_storage.e_init
            soc_final = energy_storage.e_init
            eff_charge = energy_storage.eff_ch
            eff_discharge = energy_storage.eff_dch
            max_phi = acos(energy_storage.max_pf)
            min_phi = acos(energy_storage.min_pf)

            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:

                        sch = model.es_sch[e, s_m, s_o, p]
                        pch = model.es_pch[e, s_m, s_o, p]
                        qch = model.es_qch[e, s_m, s_o, p]
                        sdch = model.es_sdch[e, s_m, s_o, p]
                        pdch = model.es_pdch[e, s_m, s_o, p]
                        qdch = model.es_qdch[e, s_m, s_o, p]

                        # ESS operation
                        model.energy_storage_operation.add(qch <= tan(max_phi) * pch)
                        model.energy_storage_operation.add(qch >= tan(min_phi) * pch)
                        model.energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                        model.energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                        if params.slacks.ess.charging:
                            model.energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + model.slack_es_sch_up[e, s_m, s_o, p] - model.slack_es_sch_down[e, s_m, s_o, p])
                            model.energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 + model.slack_es_sch_up[e, s_m, s_o, p] - model.slack_es_sch_down[e, s_m, s_o, p])
                            model.energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + model.slack_es_sdch_up[e, s_m, s_o, p] - model.slack_es_sdch_down[e, s_m, s_o, p])
                            model.energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 + model.slack_es_sdch_up[e, s_m, s_o, p] - model.slack_es_sdch_down[e, s_m, s_o, p])
                        else:
                            if params.relax_equalities:
                                model.energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + EQUALITY_TOLERANCE * 0.10)
                                model.energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 - EQUALITY_TOLERANCE * 0.10)
                                model.energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + EQUALITY_TOLERANCE * 0.10)
                                model.energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 - EQUALITY_TOLERANCE * 0.10)
                            else:
                                model.energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2)
                                model.energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2)

                        # Charging/discharging complementarity constraints
                        if params.slacks.ess.complementarity:
                            model.energy_storage_ch_dch_exclusion.add(sch * sdch <= model.slack_es_comp[e, s_m, s_o, p])
                        else:
                            if params.relax_equalities:
                                model.energy_storage_ch_dch_exclusion.add(sch * sdch <= EQUALITY_TOLERANCE * 0.10)
                            else:
                                model.energy_storage_ch_dch_exclusion.add(sch * sdch == 0.00)

                        # State-of-Charge
                        soc_prev = soc_init
                        if p > 0:
                            soc_prev = model.es_soc[e, s_m, s_o, p - 1]
                        if params.slacks.ess.soc:
                            model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + model.slack_es_soc_up[e, s_m, s_o, p] - model.slack_es_soc_down[e, s_m, s_o, p])
                            model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) + model.slack_es_soc_up[e, s_m, s_o, p] - model.slack_es_soc_down[e, s_m, s_o, p])
                        else:
                            if params.relax_equalities:
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + EQUALITY_TOLERANCE)
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) - EQUALITY_TOLERANCE)
                            else:
                                model.energy_storage_balance.add(model.es_soc[e, s_m, s_o, p] == soc_prev + (sch * eff_charge - sdch / eff_discharge))

                    if params.slacks.ess.day_balance:
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + model.slack_es_soc_final_up[e, s_m, s_o] - model.slack_es_soc_final_down[e, s_m, s_o])
                        model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final + model.slack_es_soc_final_up[e, s_m, s_o] - model.slack_es_soc_final_down[e, s_m, s_o])
                    else:
                        if params.relax_equalities:
                            model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + EQUALITY_TOLERANCE)
                            model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final - EQUALITY_TOLERANCE)
                        else:
                            model.energy_storage_day_balance.add(model.es_soc[e, s_m, s_o, len(model.periods) - 1] == soc_final)

    # - Shared Energy Storage constraints
    model.shared_energy_storage_balance = pe.ConstraintList()
    model.shared_energy_storage_operation = pe.ConstraintList()
    model.shared_energy_storage_day_balance = pe.ConstraintList()
    model.shared_energy_storage_ch_dch_exclusion = pe.ConstraintList()
    model.shared_energy_storage_s_sensitivities = pe.ConstraintList()
    model.shared_energy_storage_e_sensitivities = pe.ConstraintList()
    for e in model.shared_energy_storages:

        shared_energy_storage = network.shared_energy_storages[e]
        eff_charge = shared_energy_storage.eff_ch
        eff_discharge = shared_energy_storage.eff_dch
        max_phi = acos(shared_energy_storage.max_pf)
        min_phi = acos(shared_energy_storage.min_pf)

        s_max = model.shared_es_s_rated[e]
        soc_max = model.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
        soc_min = model.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
        soc_init = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        soc_final = model.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    sch = model.shared_es_sch[e, s_m, s_o, p]
                    pch = model.shared_es_pch[e, s_m, s_o, p]
                    qch = model.shared_es_qch[e, s_m, s_o, p]
                    sdch = model.shared_es_sdch[e, s_m, s_o, p]
                    pdch = model.shared_es_pdch[e, s_m, s_o, p]
                    qdch = model.shared_es_qdch[e, s_m, s_o, p]

                    # ESS operation
                    model.shared_energy_storage_operation.add(sch <= s_max)
                    model.shared_energy_storage_operation.add(pch <= s_max)
                    model.shared_energy_storage_operation.add(qch <= s_max)
                    model.shared_energy_storage_operation.add(qch <= tan(max_phi) * pch)
                    model.shared_energy_storage_operation.add(qch >= -s_max)
                    model.shared_energy_storage_operation.add(qch >= tan(min_phi) * pch)

                    model.shared_energy_storage_operation.add(sdch <= s_max)
                    model.shared_energy_storage_operation.add(pdch <= s_max)
                    model.shared_energy_storage_operation.add(qdch <= s_max)
                    model.shared_energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                    model.shared_energy_storage_operation.add(qdch >= -s_max)
                    model.shared_energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                    # Pnet and Qnet definition
                    if params.relax_equalities:
                        model.shared_energy_storage_operation.add(model.shared_es_pnet[e, s_m, s_o, p] <= pch - pdch + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(model.shared_es_pnet[e, s_m, s_o, p] >= pch - pdch - EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(model.shared_es_qnet[e, s_m, s_o, p] <= qch - qdch + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(model.shared_es_qnet[e, s_m, s_o, p] >= qch - qdch - EQUALITY_TOLERANCE)
                    else:
                        model.shared_energy_storage_operation.add(model.shared_es_pnet[e, s_m, s_o, p] == pch - pdch + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(model.shared_es_qnet[e, s_m, s_o, p] == qch - qdch - EQUALITY_TOLERANCE)

                    model.shared_energy_storage_operation.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_max)
                    model.shared_energy_storage_operation.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_min)

                    if params.slacks.shared_ess.charging:
                        model.shared_energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + model.slack_shared_es_sch_up[e, s_m, s_o, p] - model.slack_shared_es_sch_down[e, s_m, s_o, p] + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 + model.slack_shared_es_sch_up[e, s_m, s_o, p] - model.slack_shared_es_sch_down[e, s_m, s_o, p] - EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + model.slack_shared_es_sdch_up[e, s_m, s_o, p] - model.slack_shared_es_sdch_down[e, s_m, s_o, p] + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 + model.slack_shared_es_sdch_up[e, s_m, s_o, p] - model.slack_shared_es_sdch_down[e, s_m, s_o, p] - EQUALITY_TOLERANCE)
                    else:
                        if params.relax_equalities:
                            model.shared_energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + EQUALITY_TOLERANCE)
                            model.shared_energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 - EQUALITY_TOLERANCE)
                            model.shared_energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + EQUALITY_TOLERANCE)
                            model.shared_energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 - EQUALITY_TOLERANCE)
                        else:
                            model.shared_energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2)
                            model.shared_energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2)

                    # State-of-Charge
                    soc_prev = soc_init
                    if p > 0:
                        soc_prev = model.shared_es_soc[e, s_m, s_o, p - 1]
                    if params.slacks.shared_ess.soc:
                        model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + model.slack_shared_es_soc_up[e, s_m, s_o, p] - model.slack_shared_es_soc_down[e, s_m, s_o, p] + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) + model.slack_shared_es_soc_up[e, s_m, s_o, p] - model.slack_shared_es_soc_down[e, s_m, s_o, p] - EQUALITY_TOLERANCE)
                    else:
                        if params.relax_equalities:
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + EQUALITY_TOLERANCE)
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) - EQUALITY_TOLERANCE)
                        else:
                            model.shared_energy_storage_balance.add(model.shared_es_soc[e, s_m, s_o, p] == soc_prev + (sch * eff_charge - sdch / eff_discharge))

                    # Charging/discharging complementarity constraints
                    if params.slacks.shared_ess.complementarity:
                        model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch <= model.slack_shared_es_comp[e, s_m, s_o, p])
                    else:
                        if params.relax_equalities:
                            model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch <= EQUALITY_TOLERANCE)
                        else:
                            model.shared_energy_storage_ch_dch_exclusion.add(sch * sdch == 0.00)

                # Day balance
                if params.slacks.shared_ess.day_balance:
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + model.slack_shared_es_soc_final_up[e, s_m, s_o] - model.slack_shared_es_soc_final_down[e, s_m, s_o] + EQUALITY_TOLERANCE)
                    model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final + model.slack_shared_es_soc_final_up[e, s_m, s_o] - model.slack_shared_es_soc_final_down[e, s_m, s_o] - EQUALITY_TOLERANCE)
                else:
                    if params.relax_equalities:
                        model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] <= soc_final + EQUALITY_TOLERANCE)
                        model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] >= soc_final - EQUALITY_TOLERANCE)
                    else:
                        model.shared_energy_storage_day_balance.add(model.shared_es_soc[e, s_m, s_o, len(model.periods) - 1] == soc_final)

        model.shared_energy_storage_s_sensitivities.add(model.shared_es_s_rated[e] <= model.shared_es_s_rated_fixed[e])
        model.shared_energy_storage_e_sensitivities.add(model.shared_es_e_rated[e] <= model.shared_es_e_rated_fixed[e])

    # - Node Balance constraints
    model.node_balance_cons_p = pe.ConstraintList()
    model.node_balance_cons_q = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for i in range(len(network.nodes)):

                    node = network.nodes[i]

                    Pd = 0.00
                    Qd = 0.00
                    for c in model.loads:
                        if network.loads[c].bus == node.bus_i:
                            Pd += model.pc[c, s_m, s_o, p]
                            Qd += model.qc[c, s_m, s_o, p]
                            if params.fl_reg and network.loads[c].fl_reg:
                                Pd += (model.flex_p_up[c, s_m, s_o, p] - model.flex_p_down[c, s_m, s_o, p])
                            if params.l_curt:
                                Pd -= (model.pc_curt_down[c, s_m, s_o, p] - model.pc_curt_up[c, s_m, s_o, p])
                                Qd -= (model.qc_curt_down[c, s_m, s_o, p] - model.qc_curt_up[c, s_m, s_o, p])
                    if params.es_reg:
                        for e in model.energy_storages:
                            if network.energy_storages[e].bus == node.bus_i:
                                Pd += (model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p])
                                Qd += (model.es_qch[e, s_m, s_o, p] - model.es_qdch[e, s_m, s_o, p])
                    for e in model.shared_energy_storages:
                        if network.shared_energy_storages[e].bus == node.bus_i:
                            Pd += (model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p])
                            Qd += (model.shared_es_qch[e, s_m, s_o, p] - model.shared_es_qdch[e, s_m, s_o, p])

                    Pg = 0.0
                    Qg = 0.0
                    for g in model.generators:
                        generator = network.generators[g]
                        if generator.bus == node.bus_i:
                            Pg += model.pg[g, s_m, s_o, p]
                            Qg += model.qg[g, s_m, s_o, p]
                            if params.rg_curt:
                                Pg -= (model.pg_curt_down[g, s_m, s_o, p] - model.pg_curt_up[g, s_m, s_o, p])
                                Qg -= (model.qg_curt_down[g, s_m, s_o, p] - model.qg_curt_up[g, s_m, s_o, p])

                    ei = model.e_actual[i, s_m, s_o, p]
                    fi = model.f_actual[i, s_m, s_o, p]

                    Pi = node.gs * (ei ** 2 + fi ** 2)
                    Qi = -node.bs * (ei ** 2 + fi ** 2)
                    for b in range(len(network.branches)):
                        branch = network.branches[b]
                        if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                            rij = model.r[b, s_m, s_o, p]
                            if not branch.is_transformer:
                                rij = 1.00

                            if branch.fbus == node.bus_i:
                                fnode_idx = network.get_node_idx(branch.fbus)
                                tnode_idx = network.get_node_idx(branch.tbus)

                                ei = model.e_actual[fnode_idx, s_m, s_o, p]
                                fi = model.f_actual[fnode_idx, s_m, s_o, p]
                                ej = model.e_actual[tnode_idx, s_m, s_o, p]
                                fj = model.f_actual[tnode_idx, s_m, s_o, p]

                                Pi += branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                                Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                                Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2) * rij ** 2
                                Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))
                            else:
                                fnode_idx = network.get_node_idx(branch.tbus)
                                tnode_idx = network.get_node_idx(branch.fbus)

                                ei = model.e_actual[fnode_idx, s_m, s_o, p]
                                fi = model.f_actual[fnode_idx, s_m, s_o, p]
                                ej = model.e_actual[tnode_idx, s_m, s_o, p]
                                fj = model.f_actual[tnode_idx, s_m, s_o, p]

                                Pi += branch.g * (ei ** 2 + fi ** 2)
                                Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                                Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2)
                                Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

                    if params.slacks.node_balance:
                        model.node_balance_cons_p.add(Pg <= Pd + Pi + model.slack_node_balance_p_up[i, s_m, s_o, p] - model.slack_node_balance_p_down[i, s_m, s_o, p] + EQUALITY_TOLERANCE)
                        model.node_balance_cons_p.add(Pg >= Pd + Pi + model.slack_node_balance_p_up[i, s_m, s_o, p] - model.slack_node_balance_p_down[i, s_m, s_o, p] - EQUALITY_TOLERANCE)
                        model.node_balance_cons_q.add(Qg <= Qd + Qi + model.slack_node_balance_q_up[i, s_m, s_o, p] - model.slack_node_balance_q_down[i, s_m, s_o, p] + EQUALITY_TOLERANCE)
                        model.node_balance_cons_q.add(Qg >= Qd + Qi + model.slack_node_balance_q_up[i, s_m, s_o, p] - model.slack_node_balance_q_down[i, s_m, s_o, p] - EQUALITY_TOLERANCE)
                    else:
                        if params.relax_equalities:
                            model.node_balance_cons_p.add(Pg <= Pd + Pi + EQUALITY_TOLERANCE)
                            model.node_balance_cons_p.add(Pg >= Pd + Pi - EQUALITY_TOLERANCE)
                            model.node_balance_cons_q.add(Qg <= Qd + Qi + EQUALITY_TOLERANCE)
                            model.node_balance_cons_q.add(Qg >= Qd + Qi - EQUALITY_TOLERANCE)
                        else:
                            model.node_balance_cons_p.add(Pg == Pd + Pi)
                            model.node_balance_cons_q.add(Qg == Qd + Qi)

    # - Branch Power Flow constraints (current)
    model.branch_power_flow_cons = pe.ConstraintList()
    model.branch_power_flow_lims = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                for b in model.branches:

                    branch = network.branches[b]
                    rating = branch.rate / network.baseMVA
                    if rating == 0.0:
                        rating = BRANCH_UNKNOWN_RATING
                    fnode_idx = network.get_node_idx(branch.fbus)
                    tnode_idx = network.get_node_idx(branch.tbus)

                    rij = model.r[b, s_m, s_o, p]
                    if not branch.is_transformer:
                        rij = 1.00
                    ei = model.e_actual[fnode_idx, s_m, s_o, p]
                    fi = model.f_actual[fnode_idx, s_m, s_o, p]
                    ej = model.e_actual[tnode_idx, s_m, s_o, p]
                    fj = model.f_actual[tnode_idx, s_m, s_o, p]

                    flow_ij_sqr = 0.00

                    if params.branch_limit_type == BRANCH_LIMIT_CURRENT:

                        bij_sh = branch.b_sh * 0.50

                        iij_sqr = (branch.g ** 2 + branch.b ** 2) * (((rij ** 2) * ei - rij * ej) ** 2 + ((rij ** 2) * fi - rij * fj) ** 2)
                        iij_sqr += bij_sh ** 2 * (ei ** 2 + fi ** 2)
                        iij_sqr += 2 * branch.g * bij_sh * (((rij ** 2) * fi - rij * fj) * ei - ((rij ** 2) * ei - rij * ej) * fi)
                        iij_sqr += 2 * branch.b * bij_sh * (((rij ** 2) * ei - rij * ej) * ei + ((rij ** 2) * fi - rij * fj) * fi)
                        flow_ij_sqr = iij_sqr

                        # Previous (approximation?)
                        # iji_sqr = (branch.g ** 2 + branch.b ** 2) * ((ej - rij * ei) ** 2 + (fj - rij * fi) ** 2)
                        # iji_sqr += bij_sh ** 2 * (ej ** 2 + fj ** 2)
                        # iji_sqr += 2 * branch.g * bij_sh * ((fj - rij * fi) * ej - (ej - rij * ei) * fj)
                        # iji_sqr += 2 * branch.b * bij_sh * ((ej - rij * ei) * ej + (fj - rij * fi) * fj)

                    elif params.branch_limit_type == BRANCH_LIMIT_CURRENT_SIMPLIFIED:

                        iij_sqr = (branch.g ** 2 + branch.b ** 2) * ((ei - ej) ** 2 + (fi - fj) ** 2)
                        flow_ij_sqr = iij_sqr

                    elif params.branch_limit_type == BRANCH_LIMIT_APPARENT_POWER:

                        pij = branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                        pij -= branch.g * (ei * ej + fi * fj) * rij
                        pij -= branch.b * (fi * ej - ei * fj) * rij
                        qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2) * rij ** 2
                        qij += branch.b * (ei * ej + fi * fj) * rij
                        qij -= branch.g * (fi * ej - ei * fj) * rij
                        sij_sqr = pij ** 2 + qij ** 2
                        flow_ij_sqr = sij_sqr

                        # Without rij
                        # pji = branch.g * (ej ** 2 + fj ** 2)
                        # pji -= branch.g * (ej * ei + fj * fi) * rij
                        # pji -= branch.b * (fj * ei - ej * fi) * rij
                        # qji = - (branch.b + branch.b_sh * 0.50) * (ej ** 2 + fj ** 2)
                        # qji += branch.b * (ej * ei + fj * fi) * rij
                        # qji -= branch.g * (fj * ei - ej * fi) * rij
                        # sji_sqr = pji ** 2 + qji ** 2

                    elif params.branch_limit_type == BRANCH_LIMIT_MIXED:

                        if branch.is_transformer:
                            pij = branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                            pij -= branch.g * (ei * ej + fi * fj) * rij
                            pij -= branch.b * (fi * ej - ei * fj) * rij
                            qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2) * rij ** 2
                            qij += branch.b * (ei * ej + fi * fj) * rij
                            qij -= branch.g * (fi * ej - ei * fj) * rij
                            sij_sqr = pij ** 2 + qij ** 2
                            flow_ij_sqr = sij_sqr
                        else:
                            bij_sh = branch.b_sh * 0.50
                            iij_sqr = (branch.g ** 2 + branch.b ** 2) * (((rij ** 2) * ei - rij * ej) ** 2 + ((rij ** 2) * fi - rij * fj) ** 2)
                            iij_sqr += bij_sh ** 2 * (ei ** 2 + fi ** 2)
                            iij_sqr += 2 * branch.g * bij_sh * (((rij ** 2) * fi - rij * fj) * ei - ((rij ** 2) * ei - rij * ej) * fi)
                            iij_sqr += 2 * branch.b * bij_sh * (((rij ** 2) * ei - rij * ej) * ei + ((rij ** 2) * fi - rij * fj) * fi)
                            flow_ij_sqr = iij_sqr

                    # Flow_ij, definition
                    if params.relax_equalities:
                        model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o, p] <= flow_ij_sqr + EQUALITY_TOLERANCE)
                        model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o, p] >= flow_ij_sqr - EQUALITY_TOLERANCE)
                    else:
                        model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o, p] == flow_ij_sqr)

                    # Branch flow limits
                    if params.slacks.grid_operation.branch_flow:
                        model.branch_power_flow_lims.add(model.flow_ij_sqr[b, s_m, s_o, p] <= rating ** 2 + model.slack_flow_ij_sqr[b, s_m, s_o, p])
                    else:
                        model.branch_power_flow_lims.add(model.flow_ij_sqr[b, s_m, s_o, p] <= rating ** 2)

    # ------------------------------------------------------------------------------------------------------------------
    # Objective Function
    obj = 0.0
    if params.obj_type == OBJ_MIN_COST:

        # Cost minimization
        c_p = network.cost_energy_p
        c_flex = network.cost_flex
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0
                omega_oper = network.prob_operation_scenarios[s_o]

                # Generation
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        if (not network.is_transmission) and network.generators[g].gen_type == GEN_REFERENCE:
                            continue
                        for p in model.periods:
                            pg = model.pg[g, s_m, s_o, p]
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pg

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        for p in model.periods:
                            flex_p_up = model.flex_p_up[c, s_m, s_o, p]
                            flex_p_down = model.flex_p_down[c, s_m, s_o, p]
                            obj_scenario += c_flex[s_m][p] * network.baseMVA * (flex_p_down + flex_p_up)

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        for p in model.periods:
                            pc_curt = (model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p])
                            qc_curt = (model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p])
                            obj_scenario += model.cost_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt_down[g, s_m, s_o, p] + model.pg_curt_up[g, s_m, s_o, p]
                            qg_curt = model.qg_curt_down[g, s_m, s_o, p] + model.qg_curt_up[g, s_m, s_o, p]
                            obj_scenario += model.cost_res_curtailment * network.baseMVA * (pg_curt + qg_curt)

                # ESS utilization
                if params.es_reg:
                    for e in model.energy_storages:
                        for p in model.periods:
                            sch = model.es_sch[e, s_m, s_o, p]
                            sdch = model.es_sdch[e, s_m, s_o, p]
                            obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                # Shared ESS utilization
                for e in model.shared_energy_storages:
                    for p in model.periods:
                        sch = model.shared_es_sch[e, s_m, s_o, p]
                        sdch = model.shared_es_sdch[e, s_m, s_o, p]
                        obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                obj += obj_scenario * omega_market * omega_oper
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        # Congestion Management
        for s_m in model.scenarios_market:

            omega_market = network.prob_market_scenarios[s_m]

            for s_o in model.scenarios_operation:

                omega_oper = network.prob_operation_scenarios[s_o]

                obj_scenario = 0.0

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = model.pg_curt_down[g, s_m, s_o, p] + model.pg_curt_up[g, s_m, s_o, p]
                            qg_curt = model.qg_curt_down[g, s_m, s_o, p] + model.qg_curt_up[g, s_m, s_o, p]
                            obj_scenario += model.penalty_gen_curtailment * network.baseMVA * (pg_curt + qg_curt)

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        for p in model.periods:
                            pc_curt = (model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p])
                            qc_curt = (model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p])
                            obj_scenario += model.penalty_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        for p in model.periods:
                            flex_p_up = model.flex_p_up[c, s_m, s_o, p]
                            flex_p_down = model.flex_p_down[c, s_m, s_o, p]
                            obj_scenario += model.penalty_flex_usage * network.baseMVA * (flex_p_down + flex_p_up)

                # ESS utilization
                if params.es_reg:
                    for e in model.energy_storages:
                        for p in model.periods:
                            sch = model.es_sch[e, s_m, s_o, p]
                            sdch = model.es_sdch[e, s_m, s_o, p]
                            obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                # Shared ESS utilization
                for e in model.shared_energy_storages:
                    for p in model.periods:
                        sch = model.shared_es_sch[e, s_m, s_o, p]
                        sdch = model.shared_es_sdch[e, s_m, s_o, p]
                        obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                obj += obj_scenario * omega_market * omega_oper

    # Slacks grid operation
    for s_m in model.scenarios_market:

        omega_market = network.prob_market_scenarios[s_m]

        for s_o in model.scenarios_operation:

            omega_oper = network.prob_operation_scenarios[s_o]

            # Voltage slacks
            if params.slacks.grid_operation.voltage:
                for i in model.nodes:
                    for p in model.periods:
                        slack_e_sqr = model.slack_e[i, s_m, s_o, p] ** 2
                        slack_f_sqr = model.slack_f[i, s_m, s_o, p] ** 2
                        obj += PENALTY_VOLTAGE * network.baseMVA * omega_market * omega_oper * (slack_e_sqr + slack_f_sqr)

            # Branch power flow slacks
            if params.slacks.grid_operation.branch_flow:
                for b in model.branches:
                    for p in model.periods:
                        slack_flow_ij_sqr = (model.slack_flow_ij_sqr[b, s_m, s_o, p])
                        obj += PENALTY_CURRENT * network.baseMVA * omega_market * omega_oper * slack_flow_ij_sqr

    # Operation slacks
    for s_m in model.scenarios_market:

        omega_market = network.prob_market_scenarios[s_m]

        for s_o in model.scenarios_operation:

            omega_oper = network.prob_operation_scenarios[s_o]

            # Node balance
            if params.slacks.node_balance:
                for i in model.nodes:
                    for p in model.periods:
                        slack_p = model.slack_node_balance_p_up[i, s_m, s_o, p] + model.slack_node_balance_p_down[i, s_m, s_o, p]
                        slack_q = model.slack_node_balance_q_up[i, s_m, s_o, p] + model.slack_node_balance_q_down[i, s_m, s_o, p]
                        obj += PENALTY_NODE_BALANCE * network.baseMVA * omega_market * omega_oper * (slack_p + slack_q)

            # Flexibility day balance
            if params.fl_reg:
                for c in model.loads:
                    if params.slacks.flexibility.day_balance:
                        slack_flex_sqr = model.slack_flex_p_balance[c, s_m, s_o] ** 2
                        obj += PENALTY_FLEXIBILITY * network.baseMVA * omega_market * omega_oper * slack_flex_sqr

            # ESS slacks
            if params.es_reg:
                for e in model.energy_storages:
                    for p in model.periods:
                        if params.slacks.ess.complementarity:
                            slack_comp = model.slack_es_comp[e, s_m, s_o, p]
                            obj += PENALTY_ESS * network.baseMVA * omega_market * omega_oper * slack_comp
                        if params.slacks.ess.charging:
                            slack_sch = model.slack_es_sch_up[e, s_m, s_o, p] + model.slack_es_sch_down[e, s_m, s_o, p]
                            slack_sdch = model.slack_es_sdch_up[e, s_m, s_o, p] + model.slack_es_sdch_down[e, s_m, s_o, p]
                            obj += PENALTY_ESS * network.baseMVA * omega_market * omega_oper * (slack_sch + slack_sdch)
                        if params.slacks.ess.soc:
                            slack_soc = model.slack_es_soc_up[e, s_m, s_o, p] + model.slack_es_soc_down[e, s_m, s_o, p]
                            obj += PENALTY_ESS * network.baseMVA * omega_market * omega_oper * slack_soc
                    if params.slacks.ess.day_balance:
                        slack_soc_final = model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o]
                        obj += PENALTY_ESS * network.baseMVA * omega_market * omega_oper * slack_soc_final

            # Shared ESS slacks
            for e in model.shared_energy_storages:
                for p in model.periods:
                    if params.slacks.shared_ess.complementarity:
                        slack_comp = model.slack_shared_es_comp[e, s_m, s_o, p]
                        obj += PENALTY_SHARED_ESS * network.baseMVA * omega_market * omega_oper * slack_comp
                    if params.slacks.shared_ess.charging:
                        slack_sch = model.slack_shared_es_sch_up[e, s_m, s_o, p] + model.slack_shared_es_sch_down[e, s_m, s_o, p]
                        slack_sdch = model.slack_shared_es_sdch_up[e, s_m, s_o, p] + model.slack_shared_es_sdch_down[e, s_m, s_o, p]
                        obj += PENALTY_SHARED_ESS * network.baseMVA * omega_market * omega_oper * (slack_sch + slack_sdch)
                    if params.slacks.shared_ess.soc:
                        slack_soc = model.slack_shared_es_soc_up[e, s_m, s_o, p] + model.slack_shared_es_soc_down[e, s_m, s_o, p]
                        obj += PENALTY_SHARED_ESS * network.baseMVA * omega_market * omega_oper * slack_soc
                if params.slacks.shared_ess.day_balance:
                    slack_soc_final = model.slack_shared_es_soc_final_up[e, s_m, s_o] + model.slack_shared_es_soc_final_down[e, s_m, s_o]
                    obj += PENALTY_SHARED_ESS * network.baseMVA * omega_market * omega_oper * slack_soc_final

    model.objective = pe.Objective(sense=pe.minimize, expr=obj)

    # Model suffixes (used for warm start)
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)  # Obtain dual solutions from previous solve and send to warm start

    return model


def _run_smopf(network, model, params, from_warm_start=False):

    solver = po.SolverFactory(params.solver_params.solver, executable=params.solver_params.solver_path)

    if from_warm_start:
        model.ipopt_zL_in.update(model.ipopt_zL_out)
        model.ipopt_zU_in.update(model.ipopt_zU_out)
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['warm_start_bound_push'] = 1e-9
        solver.options['warm_start_bound_frac'] = 1e-9
        solver.options['warm_start_slack_bound_frac'] = 1e-9
        solver.options['warm_start_slack_bound_push'] = 1e-9
        solver.options['warm_start_mult_bound_push'] = 1e-9

    if params.solver_params.verbose:
        solver.options['print_level'] = 6
        solver.options['output_file'] = 'optim_log.txt'

    if params.solver_params.solver == 'ipopt':
        solver.options['tol'] = params.solver_params.solver_tol
        solver.options['linear_solver'] = params.solver_params.linear_solver
        solver.options['mu_strategy'] = 'adaptive'

    result = solver.solve(model, tee=params.solver_params.verbose)

    '''
    import logging
    from pyomo.util.infeasible import log_infeasible_constraints
    filename = os.path.join(os.getcwd(), 'example.log')
    print(log_infeasible_constraints(model, log_expression=True, log_variables=True))
    #logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO)
    '''

    return result


# ======================================================================================================================
#   NETWORK read functions -- JSON format
# ======================================================================================================================
def _read_network_from_json_file(network, filename):

    network_data = convert_json_to_dict(read_json_file(filename))

    # Network base
    network.baseMVA = float(network_data['baseMVA'])

    # Nodes
    for node_data in network_data['nodes']:
        node = Node()
        node.bus_i = int(node_data['bus_i'])
        node.type = int(node_data['type'])
        node.gs = float(node_data['Gs']) / network.baseMVA
        node.bs = float(node_data['Bs']) / network.baseMVA
        node.base_kv = float(node_data['baseKV'])
        node.v_max = float(node_data['Vmax'])
        node.v_min = float(node_data['Vmin'])
        network.nodes.append(node)

    # Generators
    for gen_data in network_data['generators']:
        generator = Generator()
        generator.gen_id = int(gen_data['gen_id'])
        generator.bus = int(gen_data['bus'])
        if not network.node_exists(generator.bus):
            print(f'[ERROR] Generator {generator.gen_id}. Node {generator.bus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        generator.pmax = float(gen_data['Pmax']) / network.baseMVA
        generator.pmin = float(gen_data['Pmin']) / network.baseMVA
        generator.qmax = float(gen_data['Qmax']) / network.baseMVA
        generator.qmin = float(gen_data['Qmin']) / network.baseMVA
        generator.vg = float(gen_data['Vg'])
        generator.status = float(gen_data['status'])
        gen_type = gen_data['type']
        if gen_type == 'REF':
            generator.gen_type = GEN_REFERENCE
        elif gen_type == 'CONV':
            generator.gen_type = GEN_CONV
        elif gen_type == 'PV':
            generator.gen_type = GEN_RES_SOLAR
        elif gen_type == 'WIND':
            generator.gen_type = GEN_RES_WIND
        elif gen_type == 'INTERCONNECTION':
            generator.gen_type = GEN_INTERCONNECTION
        elif gen_type == 'RES_OTHER':
            generator.gen_type = GEN_RES_OTHER
        elif gen_type == 'RES_CONTROLLABLE':
            generator.gen_type = GEN_RES_CONTROLLABLE
        network.generators.append(generator)

    # Loads
    for load_data in network_data['loads']:
        load = Load()
        load.load_id = int(load_data['load_id'])
        load.bus = int(load_data['bus'])
        if not network.node_exists(load.bus):
            print(f'[ERROR] Load {load.load_id }. Node {load.bus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        load.status = int(load_data['status'])
        load.fl_reg = int(load_data['fl_reg'])
        network.loads.append(load)

    # Lines
    for line_data in network_data['lines']:
        branch = Branch()
        branch.branch_id = int(line_data['branch_id'])
        branch.fbus = int(line_data['fbus'])
        if not network.node_exists(branch.fbus):
            print(f'[ERROR] Line {branch.branch_id }. Node {branch.fbus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        branch.tbus = int(line_data['tbus'])
        if not network.node_exists(branch.tbus):
            print(f'[ERROR] Line {branch.branch_id }. Node {branch.tbus} does not exist! Exiting...')
            exit(ERROR_NETWORK_FILE)
        branch.r = float(line_data['r'])
        branch.x = float(line_data['x'])
        branch.b_sh = float(line_data['b'])
        branch.rate = float(line_data['rating'])
        branch.status = int(line_data['status'])
        network.branches.append(branch)

    # Transformers
    if 'transformers' in network_data:
        for transf_data in network_data['transformers']:
            branch = Branch()
            branch.branch_id = int(transf_data['branch_id'])
            branch.fbus = int(transf_data['fbus'])
            if not network.node_exists(branch.fbus):
                print(f'[ERROR] Transformer {branch.branch_id}. Node {branch.fbus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            branch.tbus = int(transf_data['tbus'])
            if not network.node_exists(branch.tbus):
                print(f'[ERROR] Transformer {branch.branch_id}. Node {branch.tbus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            branch.r = float(transf_data['r'])
            branch.x = float(transf_data['x'])
            branch.b_sh = float(transf_data['b'])
            branch.rate = float(transf_data['rating'])
            branch.ratio = float(transf_data['ratio'])
            branch.status = bool(transf_data['status'])
            branch.is_transformer = True
            branch.vmag_reg = bool(transf_data['vmag_reg'])
            network.branches.append(branch)

    # Energy Storages
    if 'energy_storages' in network_data:
        for energy_storage_data in network_data['energy_storages']:
            energy_storage = EnergyStorage()
            energy_storage.es_id = int(energy_storage_data['es_id'])
            energy_storage.bus = int(energy_storage_data['bus'])
            if not network.node_exists(energy_storage.bus):
                print(f'[ERROR] Energy Storage {energy_storage.es_id}. Node {energy_storage.bus} does not exist! Exiting...')
                exit(ERROR_NETWORK_FILE)
            energy_storage.s = float(energy_storage_data['s']) / network.baseMVA
            energy_storage.e = float(energy_storage_data['e']) / network.baseMVA
            energy_storage.e_init = energy_storage.e * ENERGY_STORAGE_RELATIVE_INIT_SOC
            energy_storage.e_min = energy_storage.e * ENERGY_STORAGE_MIN_ENERGY_STORED
            energy_storage.e_max = energy_storage.e * ENERGY_STORAGE_MAX_ENERGY_STORED
            energy_storage.eff_ch = float(energy_storage_data['eff_ch'])
            energy_storage.eff_dch = float(energy_storage_data['eff_dch'])
            energy_storage.max_pf = float(energy_storage_data['max_pf'])
            energy_storage.min_pf = float(energy_storage_data['min_pf'])
            network.energy_storages.append(energy_storage)


def _get_consumption_from_data(data, node_id, num_instants, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pc'
    else:
        power_label = 'qc'

    for node in data['consumption'][power_label][idx_scenario]:
        if node == node_id:
            return data['consumption'][power_label][idx_scenario][node_id]

    consumption = [0.0 for _ in range(num_instants)]

    return consumption


def _get_flexibility_from_data(data, node_id, num_instants, flex_type):

    flex_label = str()

    if flex_type == DATA_UPWARD_FLEXIBILITY:
        flex_label = 'upward'
    elif flex_type == DATA_DOWNWARD_FLEXIBILITY:
        flex_label = 'downward'
    elif flex_type == DATA_COST_FLEXIBILITY:
        flex_label = 'cost'
    else:
        print('[ERROR] Unrecognized flexibility type in get_flexibility_from_data. Exiting.')
        exit(1)

    for node in data['flexibility'][flex_label]:
        if node == node_id:
            return data['flexibility'][flex_label][node_id]

    flex = [0.0 for _ in range(num_instants)]   # Returns empty flexibility vector

    return flex


def _get_generation_from_data(data, gen_id, idx_scenario, type):

    if type == DATA_ACTIVE_POWER:
        power_label = 'pg'
    else:
        power_label = 'qg'

    return data['generation'][power_label][idx_scenario][gen_id]


# ======================================================================================================================
#   NETWORK OPERATIONAL DATA read functions
# ======================================================================================================================
def _read_network_operational_data_from_file(network, filename):

    data = {
        'consumption': {
            'pc': dict(), 'qc': dict()
        },
        'flexibility': {
            'upward': dict(),
            'downward': dict()
        },
        'generation': {
            'pg': dict(), 'qg': dict(), 'status': list()
        }
    }

    # Scenario information
    num_gen_cons_scenarios, prob_gen_cons_scenarios = _get_operational_scenario_info_from_excel_file(filename, 'Main')
    network.prob_operation_scenarios = prob_gen_cons_scenarios

    # Consumption and Generation data -- by scenario
    for i in range(len(network.prob_operation_scenarios)):

        sheet_name_pc = f'Pc, {network.day}, S{i + 1}'
        sheet_name_qc = f'Qc, {network.day}, S{i + 1}'
        sheet_name_pg = f'Pg, {network.day}, S{i + 1}'
        sheet_name_qg = f'Qg, {network.day}, S{i + 1}'

        # Consumption per scenario (active, reactive power)
        pc_scenario = _get_consumption_flexibility_data_from_excel_file(filename, sheet_name_pc)
        qc_scenario = _get_consumption_flexibility_data_from_excel_file(filename, sheet_name_qc)
        if not pc_scenario:
            print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No active power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        if not qc_scenario:
            print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No reactive power consumption data provided for scenario {i + 1}. Exiting...')
            exit(ERROR_OPERATIONAL_DATA_FILE)
        data['consumption']['pc'][i] = pc_scenario
        data['consumption']['qc'][i] = qc_scenario

        # Generation per scenario (active, reactive power)
        num_renewable_gens = network.get_num_renewable_gens()
        if num_renewable_gens > 0:
            pg_scenario = _get_generation_data_from_excel_file(filename, sheet_name_pg)
            qg_scenario = _get_generation_data_from_excel_file(filename, sheet_name_qg)
            if not pg_scenario:
                print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No active power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            if not qg_scenario:
                print(f'[ERROR] Network {network.name}, {network.year}, {network.day}. No reactive power generation data provided for scenario {i + 1}. Exiting...')
                exit(ERROR_OPERATIONAL_DATA_FILE)
            data['generation']['pg'][i] = pg_scenario
            data['generation']['qg'][i] = qg_scenario

    # Generators status. Note: common to all scenarios
    gen_status = _get_generator_status_from_excel_file(filename, f'GenStatus, {network.day}')
    if not gen_status:
        for g in range(len(network.generators)):
            gen_status.append([network.generators[g].status for _ in range(network.num_instants)])
    data['generation']['status'] = gen_status

    # Flexibility data
    flex_up_p = _get_consumption_flexibility_data_from_excel_file(filename, f'UpFlex, {network.day}')
    if not flex_up_p:
        for load in network.loads:
            flex_up_p[load.load_id] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['upward'] = flex_up_p

    flex_down_p = _get_consumption_flexibility_data_from_excel_file(filename, f'DownFlex, {network.day}')
    if not flex_down_p:
        for load in network.loads:
            flex_down_p[load.load_id] = [0.0 for _ in range(network.num_instants)]
    data['flexibility']['downward'] = flex_down_p

    return data


def _get_operational_scenario_info_from_excel_file(filename, sheet_name):

    num_scenarios = 0
    prob_scenarios = list()

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        for i in range(num_scenarios):
            if is_number(df.iloc[0, i+2]):
                prob_scenarios.append(float(df.iloc[0, i+2]))
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)

    if num_scenarios != len(prob_scenarios):
        print('[WARNING] Workbook {}. Data file. Number of scenarios different from the probability vector!'.format(filename))

    if round(sum(prob_scenarios), 2) != 1.00:
        print('[ERROR] Workbook {}. Probability of scenarios does not add up to 100%.'.format(filename))
        exit(ERROR_OPERATIONAL_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_consumption_flexibility_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = dict()
        for i in range(num_rows):
            node_id = data.iloc[i, 0]
            processed_data[node_id] = [0.0 for _ in range(num_cols - 1)]
        for node_id in processed_data:
            node_values = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == node_id:
                    for j in range(0, num_cols - 1):
                        node_values[j] += data.iloc[i, j + 1]
            processed_data[node_id] = node_values
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _get_generation_data_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        processed_data = dict()
        for i in range(num_rows):
            gen_id = data.iloc[i, 0]
            processed_data[gen_id] = [0.0 for _ in range(num_cols - 1)]
        for gen_id in processed_data:
            processed_data_gen = [0.0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == gen_id:
                    for j in range(0, num_cols - 1):
                        processed_data_gen[j] += data.iloc[i, j + 1]
            processed_data[gen_id] = processed_data_gen
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        processed_data = {}

    return processed_data


def _get_generator_status_from_excel_file(filename, sheet_name):

    try:
        data = pd.read_excel(filename, sheet_name=sheet_name)
        num_rows, num_cols = data.shape
        status_values = dict()
        for i in range(num_rows):
            gen_id = data.iloc[i, 0]
            status_values[gen_id] = [0 for _ in range(num_cols - 1)]
        for node_id in status_values:
            status_values_gen = [0 for _ in range(num_cols - 1)]
            for i in range(0, num_rows):
                aux_node_id = data.iloc[i, 0]
                if aux_node_id == node_id:
                    for j in range(0, num_cols - 1):
                        status_values_gen[j] += data.iloc[i, j + 1]
            status_values[node_id] = status_values_gen
    except:
        print(f'[WARNING] Workbook {filename}. Sheet {sheet_name} does not exist.')
        status_values = list()

    return status_values


def _update_network_with_excel_data(network, data):

    for load in network.loads:

        load_id = load.load_id
        load.pd = dict()         # Note: Changes Pd and Qd fields to dicts (per scenario)
        load.qd = dict()

        for s in range(len(network.prob_operation_scenarios)):
            pc = _get_consumption_from_data(data, load_id, network.num_instants, s, DATA_ACTIVE_POWER)
            qc = _get_consumption_from_data(data, load_id, network.num_instants, s, DATA_REACTIVE_POWER)
            load.pd[s] = [instant / network.baseMVA for instant in pc]
            load.qd[s] = [instant / network.baseMVA for instant in qc]
        flex_up_p = _get_flexibility_from_data(data, load_id, network.num_instants, DATA_UPWARD_FLEXIBILITY)
        flex_down_p = _get_flexibility_from_data(data, load_id, network.num_instants, DATA_DOWNWARD_FLEXIBILITY)
        load.flexibility.upward = [p / network.baseMVA for p in flex_up_p]
        load.flexibility.downward = [q / network.baseMVA for q in flex_down_p]

    for generator in network.generators:

        generator.pg = dict()  # Note: Changes Pg and Qg fields to dicts (per scenario)
        generator.qg = dict()

        # Active and Reactive power
        for s in range(len(network.prob_operation_scenarios)):
            if generator.gen_type in GEN_CURTAILLABLE_TYPES:
                pg = _get_generation_from_data(data, generator.gen_id, s, DATA_ACTIVE_POWER)
                qg = _get_generation_from_data(data, generator.gen_id, s, DATA_REACTIVE_POWER)
                generator.pg[s] = [instant / network.baseMVA for instant in pg]
                generator.qg[s] = [instant / network.baseMVA for instant in qg]
            else:
                generator.pg[s] = [0.00 for _ in range(network.num_instants)]
                generator.qg[s] = [0.00 for _ in range(network.num_instants)]

        # Status
        generator.status = data['generation']['status'][generator.gen_id]

    network.data_loaded = True


# ======================================================================================================================
#   NETWORK RESULTS functions
# ======================================================================================================================
def _process_results(network, model, params, results=dict()):

    s_base = network.baseMVA

    processed_results = dict()
    processed_results['obj'] = _compute_objective_function_value(network, model, params)
    processed_results['gen_cost'] = _compute_generation_cost(network, model)
    processed_results['total_load'] = _compute_total_load(network, model, params)
    processed_results['total_gen'] = _compute_total_generation(network, model, params)
    processed_results['total_conventional_gen'] = _compute_conventional_generation(network, model, params)
    processed_results['total_renewable_gen'] = _compute_renewable_generation(network, model, params)
    processed_results['losses'] = _compute_losses(network, model, params)
    processed_results['gen_curt'] = _compute_generation_curtailment(network, model, params)
    processed_results['load_curt'] = _compute_load_curtailment(network, model, params)
    processed_results['flex_used'] = _compute_flexibility_used(network, model, params)
    if results:
        processed_results['runtime'] = float(_get_info_from_results(results, 'Time:').strip()),

    processed_results['scenarios'] = dict()
    for s_m in model.scenarios_market:

        processed_results['scenarios'][s_m] = dict()

        for s_o in model.scenarios_operation:

            processed_results['scenarios'][s_m][s_o] = {
                'voltage': {'vmag': {}, 'vang': {}},
                'consumption': {'pc': {}, 'qc': {}, 'pc_net': {}, 'qc_net': {}},
                'generation': {'pg': {}, 'qg': {}, 'pg_net': {}, 'qg_net': {}},
                'branches': {'power_flow': {'pij': {}, 'pji': {}, 'qij': {}, 'qji': {}, 'sij': {}, 'sji': {}},
                             'losses': {}, 'ratio': {}, 'branch_flow': {'flow_ij_perc': {}}},
                'energy_storages': {'p': {}, 'q': {}, 's': {}, 'soc': {}, 'soc_percent': {}},
                'shared_energy_storages': {'p': {}, 'q': {}, 's': {}, 'soc': {}, 'soc_percent': {}}
            }

            if params.transf_reg:
                processed_results['scenarios'][s_m][s_o]['branches']['ratio'] = dict()

            if params.fl_reg:
                processed_results['scenarios'][s_m][s_o]['consumption']['p_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['consumption']['p_down'] = dict()

            if params.l_curt:
                processed_results['scenarios'][s_m][s_o]['consumption']['pc_curt'] = dict()
                processed_results['scenarios'][s_m][s_o]['consumption']['qc_curt'] = dict()

            if params.rg_curt:
                processed_results['scenarios'][s_m][s_o]['generation']['pg_curt'] = dict()
                processed_results['scenarios'][s_m][s_o]['generation']['qg_curt'] = dict()

            if params.es_reg:
                processed_results['scenarios'][s_m][s_o]['energy_storages']['p'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['q'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['s'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'] = dict()
                processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'] = dict()

            processed_results['scenarios'][s_m][s_o]['relaxation_slacks'] = dict()
            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage'] = dict()
            if params.slacks.grid_operation.voltage:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['f'] = dict()
            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['branch_flow'] = dict()
            if params.slacks.grid_operation.branch_flow:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['branch_flow']['flow_ij_sqr'] = dict()
            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance'] = dict()
            if params.slacks.node_balance:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_down'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_down'] = dict()
            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages'] = dict()
            if params.slacks.shared_ess.complementarity:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'] = dict()
            if params.slacks.shared_ess.charging:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sch_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sch_down'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sdch_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sdch_down'] = dict()
            if params.slacks.shared_ess.soc:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_down'] = dict()
            if params.slacks.shared_ess.day_balance:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_up'] = dict()
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_down'] = dict()

            if params.fl_reg:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility'] = dict()
                if params.slacks.flexibility.day_balance:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance'] = dict()
            if params.es_reg:
                processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages'] = dict()
                if params.slacks.ess.complementarity:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'] = dict()
                if params.slacks.ess.charging:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_up'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_down'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_up'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_down'] = dict()
                if params.slacks.ess.soc:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_up'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_down'] = dict()
                if params.slacks.ess.day_balance:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_up'] = dict()
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_down'] = dict()

            # Voltage
            for i in model.nodes:
                node_id = network.nodes[i].bus_i
                processed_results['scenarios'][s_m][s_o]['voltage']['vmag'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['voltage']['vang'][node_id] = []
                for p in model.periods:
                    e = pe.value(model.e_actual[i, s_m, s_o, p])
                    f = pe.value(model.f_actual[i, s_m, s_o, p])
                    v_mag = sqrt(e**2 + f**2)
                    v_ang = atan2(f, e) * (180.0 / pi)
                    processed_results['scenarios'][s_m][s_o]['voltage']['vmag'][node_id].append(v_mag)
                    processed_results['scenarios'][s_m][s_o]['voltage']['vang'][node_id].append(v_ang)

            # Consumption
            for c in model.loads:
                load_id = network.loads[c].load_id
                processed_results['scenarios'][s_m][s_o]['consumption']['pc'][load_id] = []
                processed_results['scenarios'][s_m][s_o]['consumption']['qc'][load_id] = []
                processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][load_id] = [0.00 for _ in range(network.num_instants)]
                processed_results['scenarios'][s_m][s_o]['consumption']['qc_net'][load_id] = [0.00 for _ in range(network.num_instants)]
                if params.fl_reg:
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][load_id] = []
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][load_id] = []
                if params.l_curt:
                    processed_results['scenarios'][s_m][s_o]['consumption']['pc_curt'][load_id] = []
                    processed_results['scenarios'][s_m][s_o]['consumption']['qc_curt'][load_id] = []
                for p in model.periods:
                    pc = pe.value(model.pc[c, s_m, s_o, p]) * network.baseMVA
                    qc = pe.value(model.qc[c, s_m, s_o, p]) * network.baseMVA
                    processed_results['scenarios'][s_m][s_o]['consumption']['pc'][load_id].append(pc)
                    processed_results['scenarios'][s_m][s_o]['consumption']['qc'][load_id].append(qc)
                    processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][load_id][p] += pc
                    processed_results['scenarios'][s_m][s_o]['consumption']['qc_net'][load_id][p] += qc
                    if params.fl_reg:
                        pup = pe.value(model.flex_p_up[c, s_m, s_o, p]) * network.baseMVA
                        pdown = pe.value(model.flex_p_down[c, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][load_id].append(pup)
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][load_id].append(pdown)
                        processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][load_id][p] += pup - pdown
                    if params.l_curt:
                        pc_curt = pe.value(model.pc_curt_down[c, s_m, s_o, p] - model.pc_curt_up[c, s_m, s_o, p]) * network.baseMVA
                        qc_curt = pe.value(model.qc_curt_down[c, s_m, s_o, p] - model.qc_curt_up[c, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['consumption']['pc_curt'][load_id].append(pc_curt)
                        processed_results['scenarios'][s_m][s_o]['consumption']['pc_net'][load_id][p] -= pc_curt
                        processed_results['scenarios'][s_m][s_o]['consumption']['qc_curt'][load_id].append(qc_curt)
                        processed_results['scenarios'][s_m][s_o]['consumption']['qc_net'][load_id][p] -= qc_curt

            # Generation
            for g in model.generators:
                gen_id = network.generators[g].gen_id
                processed_results['scenarios'][s_m][s_o]['generation']['pg'][gen_id] = []
                processed_results['scenarios'][s_m][s_o]['generation']['qg'][gen_id] = []
                processed_results['scenarios'][s_m][s_o]['generation']['pg_net'][gen_id] = [0.00 for _ in range(network.num_instants)]
                processed_results['scenarios'][s_m][s_o]['generation']['qg_net'][gen_id] = [0.00 for _ in range(network.num_instants)]
                if params.rg_curt:
                    processed_results['scenarios'][s_m][s_o]['generation']['pg_curt'][gen_id] = []
                    processed_results['scenarios'][s_m][s_o]['generation']['qg_curt'][gen_id] = []
                for p in model.periods:
                    pg = pe.value(model.pg[g, s_m, s_o, p]) * network.baseMVA
                    qg = pe.value(model.qg[g, s_m, s_o, p]) * network.baseMVA
                    processed_results['scenarios'][s_m][s_o]['generation']['pg'][gen_id].append(pg)
                    processed_results['scenarios'][s_m][s_o]['generation']['qg'][gen_id].append(qg)
                    processed_results['scenarios'][s_m][s_o]['generation']['pg_net'][gen_id][p] += pg
                    processed_results['scenarios'][s_m][s_o]['generation']['qg_net'][gen_id][p] += qg
                    if params.rg_curt:
                        pg_curt = pe.value(model.pg_curt_down[g, s_m, s_o, p] - model.pg_curt_up[g, s_m, s_o, p]) * network.baseMVA
                        qg_curt = pe.value(model.qg_curt_down[g, s_m, s_o, p] - model.qg_curt_up[g, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['generation']['pg_curt'][gen_id].append(pg_curt)
                        processed_results['scenarios'][s_m][s_o]['generation']['qg_curt'][gen_id].append(qg_curt)
                        processed_results['scenarios'][s_m][s_o]['generation']['pg_net'][gen_id][p] -= pg_curt
                        processed_results['scenarios'][s_m][s_o]['generation']['qg_net'][gen_id][p] -= qg_curt

            # Branch current, transformers' ratio
            for k in model.branches:

                branch = network.branches[k]
                branch_id = branch.branch_id
                rating = branch.rate / network.baseMVA
                if rating == 0.0:
                    rating = BRANCH_UNKNOWN_RATING

                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['losses'][branch_id] = []
                processed_results['scenarios'][s_m][s_o]['branches']['branch_flow']['flow_ij_perc'][branch_id] = []
                if branch.is_transformer:
                    processed_results['scenarios'][s_m][s_o]['branches']['ratio'][branch_id] = []
                for p in model.periods:

                    # Power flows
                    pij, qij = _get_branch_power_flow(network, params, branch, branch.fbus, branch.tbus, model, s_m, s_o, p)
                    pji, qji = _get_branch_power_flow(network, params, branch, branch.tbus, branch.fbus, model, s_m, s_o, p)
                    sij_sqr = pij**2 + qij**2
                    sji_sqr = pji**2 + qji**2
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pij'][branch_id].append(pij)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['pji'][branch_id].append(pji)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qij'][branch_id].append(qij)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['qji'][branch_id].append(qji)
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sij'][branch_id].append(sqrt(sij_sqr))
                    processed_results['scenarios'][s_m][s_o]['branches']['power_flow']['sji'][branch_id].append(sqrt(sji_sqr))

                    # Losses (active power)
                    p_losses = _get_branch_power_losses(network, params, model, k, s_m, s_o, p)
                    processed_results['scenarios'][s_m][s_o]['branches']['losses'][branch_id].append(p_losses)

                    # Ratio
                    if branch.is_transformer:
                        r_ij = pe.value(model.r[k, s_m, s_o, p])
                        processed_results['scenarios'][s_m][s_o]['branches']['ratio'][branch_id].append(r_ij)

                    # Branch flow (limits)
                    flow_ij_perc = sqrt(abs(pe.value(model.flow_ij_sqr[k, s_m, s_o, p]))) / rating
                    processed_results['scenarios'][s_m][s_o]['branches']['branch_flow']['flow_ij_perc'][branch_id].append(flow_ij_perc)

            # Energy Storage devices
            if params.es_reg:
                for e in model.energy_storages:
                    es_id = network.energy_storages[e].es_id
                    capacity = network.energy_storages[e].e * network.baseMVA
                    if isclose(capacity, 0.0, abs_tol=1e-6):
                        capacity = 1.00
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['p'][es_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['q'][es_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['s'][es_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'][es_id] = []
                    processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'][es_id] = []
                    for p in model.periods:
                        s_ess = pe.value(model.es_sch[e, s_m, s_o, p] - model.es_sdch[e, s_m, s_o, p]) * network.baseMVA
                        p_ess = pe.value(model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p]) * network.baseMVA
                        q_ess = pe.value(model.es_qch[e, s_m, s_o, p] - model.es_qdch[e, s_m, s_o, p]) * network.baseMVA
                        soc_ess = pe.value(model.es_soc[e, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['energy_storages']['p'][es_id].append(p_ess)
                        processed_results['scenarios'][s_m][s_o]['energy_storages']['q'][es_id].append(q_ess)
                        processed_results['scenarios'][s_m][s_o]['energy_storages']['s'][es_id].append(s_ess)
                        processed_results['scenarios'][s_m][s_o]['energy_storages']['soc'][es_id].append(soc_ess)
                        processed_results['scenarios'][s_m][s_o]['energy_storages']['soc_percent'][es_id].append(soc_ess / capacity)

            # Flexible loads
            if params.fl_reg:
                for i in model.loads:
                    load_id = network.loads[i].load_id
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][load_id] = []
                    processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][load_id] = []
                    for p in model.periods:
                        p_up = pe.value(model.flex_p_up[i, s_m, s_o, p]) * network.baseMVA
                        p_down = pe.value(model.flex_p_down[i, s_m, s_o, p]) * network.baseMVA
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_up'][load_id].append(p_up)
                        processed_results['scenarios'][s_m][s_o]['consumption']['p_down'][load_id].append(p_down)

            # Shared Energy Storages
            for e in model.shared_energy_storages:
                node_id = network.shared_energy_storages[e].bus
                capacity = pe.value(model.shared_es_e_rated[e]) * network.baseMVA
                if isclose(capacity, 0.0, abs_tol=1e-6):
                    capacity = 1.00
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['p'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['q'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['s'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][node_id] = []
                processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][node_id] = []
                for p in model.periods:
                    s_ess = pe.value(model.shared_es_sch[e, s_m, s_o, p] - model.shared_es_sdch[e, s_m, s_o, p]) * network.baseMVA
                    p_ess = pe.value(model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p]) * network.baseMVA
                    q_ess = pe.value(model.shared_es_qch[e, s_m, s_o, p] - model.shared_es_qdch[e, s_m, s_o, p]) * network.baseMVA
                    soc_ess = pe.value(model.shared_es_soc[e, s_m, s_o, p]) * network.baseMVA
                    processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['p'][node_id].append(p_ess)
                    processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['q'][node_id].append(q_ess)
                    processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['s'][node_id].append(s_ess)
                    processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc'][node_id].append(soc_ess)
                    processed_results['scenarios'][s_m][s_o]['shared_energy_storages']['soc_percent'][node_id].append(soc_ess / capacity)

            # Voltage slacks
            if params.slacks.grid_operation.voltage:
                for i in model.nodes:
                    node_id = network.nodes[i].bus_i
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['f'][node_id] = []
                    for p in model.periods:
                        slack_e = pe.value(model.slack_e[i, s_m, s_o, p])
                        slack_f = pe.value(model.slack_f[i, s_m, s_o, p])
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['e'][node_id].append(slack_e)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['voltage']['f'][node_id].append(slack_f)

            # Branch current slacks
            if params.slacks.grid_operation.branch_flow:
                for b in model.branches:
                    branch_id = network.branches[b].branch_id
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['branch_flow']['flow_ij_sqr'][branch_id] = []
                    for p in model.periods:
                        slack_flow_ij_sqr = pe.value(model.slack_flow_ij_sqr[b, s_m, s_o, p])
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['branch_flow']['flow_ij_sqr'][branch_id].append(slack_flow_ij_sqr)

            # Slacks
            # - Shared ESS
            for e in model.shared_energy_storages:
                node_id = network.shared_energy_storages[e].bus
                if params.slacks.shared_ess.complementarity:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'][node_id] = []
                if params.slacks.shared_ess.charging:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sch_up'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sch_down'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sdch_up'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sdch_down'][node_id] = []
                if params.slacks.shared_ess.soc:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_up'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_down'][node_id] = []
                if params.slacks.shared_ess.day_balance:
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_up'][node_id] = [0.00 for _ in range(network.num_instants)]
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_down'][node_id] = [0.00 for _ in range(network.num_instants)]
                for p in model.periods:
                    if params.slacks.shared_ess.complementarity:
                        slack_comp = pe.value(model.slack_shared_es_comp[e, s_m, s_o, p]) * (s_base ** 2)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['comp'][node_id].append(slack_comp)
                    if params.slacks.shared_ess.charging:
                        slack_sch_up = pe.value(model.slack_shared_es_sch_up[e, s_m, s_o, p]) * (s_base ** 2)
                        slack_sch_down = pe.value(model.slack_shared_es_sch_down[e, s_m, s_o, p]) * (s_base ** 2)
                        slack_sdch_up = pe.value(model.slack_shared_es_sdch_up[e, s_m, s_o, p]) * (s_base ** 2)
                        slack_sdch_down = pe.value(model.slack_shared_es_sdch_down[e, s_m, s_o, p]) * (s_base ** 2)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sch_up'][node_id].append(slack_sch_up)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sch_down'][node_id].append(slack_sch_down)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sdch_up'][node_id].append(slack_sdch_up)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['sdch_down'][node_id].append(slack_sdch_down)
                    if params.slacks.shared_ess.soc:
                        slack_soc_up = pe.value(model.slack_shared_es_soc_up[e, s_m, s_o, p]) * s_base
                        slack_soc_down = pe.value(model.slack_shared_es_soc_down[e, s_m, s_o, p]) * s_base
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_up'][node_id].append(slack_soc_up)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_down'][node_id].append(slack_soc_down)
                if params.slacks.shared_ess.day_balance:
                    slack_soc_final_up = pe.value(model.slack_shared_es_soc_final_up[e, s_m, s_o]) * s_base
                    slack_soc_final_down = pe.value(model.slack_shared_es_soc_final_down[e, s_m, s_o]) * s_base
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_up'][node_id][network.num_instants - 1] = slack_soc_final_up
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['shared_energy_storages']['soc_final_down'][node_id][network.num_instants - 1] = slack_soc_final_down

            # - Node balance
            if params.slacks.node_balance:
                for i in model.nodes:
                    node_id = network.nodes[i].bus_i
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_up'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_down'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_up'][node_id] = []
                    processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_down'][node_id] = []
                    for p in model.periods:
                        slack_p_up = pe.value(model.slack_node_balance_p_up[i, s_m, s_o, p]) * s_base
                        slack_p_down = pe.value(model.slack_node_balance_p_down[i, s_m, s_o, p]) * s_base
                        slack_q_up = pe.value(model.slack_node_balance_q_up[i, s_m, s_o, p]) * s_base
                        slack_q_down = pe.value(model.slack_node_balance_q_down[i, s_m, s_o, p]) * s_base
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_up'][node_id].append(slack_p_up)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['p_down'][node_id].append(slack_p_down)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_up'][node_id].append(slack_q_up)
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['node_balance']['q_down'][node_id].append(slack_q_down)

            # - Flexibility
            if params.fl_reg:
                for c in model.loads:
                    load_id = network.loads[c].load_id
                    if params.slacks.flexibility.day_balance:
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance'][load_id] = [0.00 for _ in range(network.num_instants)]
                        slack_flex = pe.value(model.slack_flex_p_balance[c, s_m, s_o]) * s_base
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['flexibility']['day_balance'][load_id][network.num_instants-1] = slack_flex

            # - ESS slacks
            if params.es_reg:
                for e in model.energy_storages:
                    es_id = network.energy_storages[e].es_id
                    if params.slacks.ess.complementarity:
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'][es_id] = []
                    if params.slacks.ess.charging:
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_up'][es_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_down'][es_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_up'][es_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_down'][es_id] = []
                    if params.slacks.ess.soc:
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_up'][es_id] = []
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_down'][es_id] = []
                    if params.slacks.ess.day_balance:
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_up'][es_id] = [0.00 for _ in range(network.num_instants)]
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_down'][es_id] = [0.00 for _ in range(network.num_instants)]
                    for p in model.periods:
                        if params.slacks.ess.complementarity:
                            slack_comp = pe.value(model.slack_es_comp[e, s_m, s_o, p]) * (s_base ** 2)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['comp'][es_id].append(slack_comp)
                        if params.slacks.ess.charging:
                            slack_sch_up = pe.value(model.slack_es_sch_up[e, s_m, s_o, p]) * (s_base ** 2)
                            slack_sch_down = pe.value(model.slack_es_sch_down[e, s_m, s_o, p]) * (s_base ** 2)
                            slack_sdch_up = pe.value(model.slack_es_sdch_up[e, s_m, s_o, p]) * (s_base ** 2)
                            slack_sdch_down = pe.value(model.slack_es_sdch_down[e, s_m, s_o, p]) * (s_base ** 2)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_up'][es_id].append(slack_sch_up)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sch_down'][es_id].append(slack_sch_down)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_up'][es_id].append(slack_sdch_up)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['sdch_down'][es_id].append(slack_sdch_down)
                        if params.slacks.ess.soc:
                            slack_soc_up = pe.value(model.slack_es_soc_up[e, s_m, s_o, p]) * s_base
                            slack_soc_down = pe.value(model.slack_es_soc_down[e, s_m, s_o, p]) * s_base
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_up'][es_id].append(slack_soc_up)
                            processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_down'][es_id].append(slack_soc_down)
                    if params.slacks.ess.day_balance:
                        slack_soc_final_up = pe.value(model.slack_es_soc_final_up[e, s_m, s_o]) * s_base
                        slack_soc_final_down = pe.value(model.slack_es_soc_final_down[e, s_m, s_o]) * s_base
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_up'][es_id][network.num_instants-1] = slack_soc_final_up
                        processed_results['scenarios'][s_m][s_o]['relaxation_slacks']['energy_storages']['soc_final_down'][es_id][network.num_instants-1] = slack_soc_final_down

    return processed_results


def _process_results_interface(network, model):

    results = dict()

    if network.is_transmission:
        for dn in model.active_distribution_networks:

            node_id = network.active_distribution_network_nodes[dn]
            node_idx = network.get_node_idx(node_id)
            load_idx = network.get_adn_load_idx(node_id)

            # Power flow results per market and operation scenario
            results[node_id] = dict()
            for s_m in model.scenarios_market:
                results[node_id][s_m] = dict()
                for s_o in model.scenarios_operation:
                    results[node_id][s_m][s_o] = dict()
                    results[node_id][s_m][s_o]['v'] = [1.0 for _ in model.periods]
                    results[node_id][s_m][s_o]['p'] = [0.0 for _ in model.periods]
                    results[node_id][s_m][s_o]['q'] = [0.0 for _ in model.periods]
                    for p in model.periods:
                        vmag = sqrt(pe.value(model.e_actual[node_idx, s_m, s_o, p]**2 + model.f_actual[node_idx, s_m, s_o, p]**2))
                        pf_p = pe.value(model.pc[load_idx, s_m, s_o, p]) * network.baseMVA
                        pf_q = pe.value(model.qc[load_idx, s_m, s_o, p]) * network.baseMVA
                        results[node_id][s_m][s_o]['v'][p] = vmag
                        results[node_id][s_m][s_o]['p'][p] = pf_p
                        results[node_id][s_m][s_o]['q'][p] = pf_q
    else:

        # Power flow results per market and operation scenario
        ref_node_id = network.get_reference_node_id()
        ref_node_idx = network.get_node_idx(ref_node_id)
        ref_gen_idx = network.get_reference_gen_idx()
        for s_m in model.scenarios_market:
            results[s_m] = dict()
            for s_o in model.scenarios_operation:
                results[s_m][s_o] = dict()
                results[s_m][s_o]['v'] = [0.0 for _ in model.periods]
                results[s_m][s_o]['p'] = [0.0 for _ in model.periods]
                results[s_m][s_o]['q'] = [0.0 for _ in model.periods]
                for p in model.periods:
                    vmag = pe.value(model.e_actual[ref_node_idx, s_m, s_o, p])
                    pf_p = pe.value(model.pg[ref_gen_idx, s_m, s_o, p]) * network.baseMVA
                    pf_q = pe.value(model.qg[ref_gen_idx, s_m, s_o, p]) * network.baseMVA
                    results[s_m][s_o]['v'][p] = vmag
                    results[s_m][s_o]['p'][p] = pf_p
                    results[s_m][s_o]['q'][p] = pf_q

    return results


def _compute_objective_function_value(network, model, params):

    obj = 0.0

    if params.obj_type == OBJ_MIN_COST:

        c_p = network.cost_energy_p
        c_flex = network.cost_flex
        cost_res_curt = pe.value(model.cost_res_curtailment)
        cost_load_curt = pe.value(model.cost_load_curtailment)

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0

                # Generation -- paid at market price
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        if (not network.is_transmission) and network.generators[g].gen_type == GEN_REFERENCE:
                            continue
                        for p in model.periods:
                            pg = pe.value(model.pg[g, s_m, s_o, p])
                            obj_scenario += c_p[s_m][p] * network.baseMVA * pg

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        for p in model.periods:
                            flex_up = pe.value(model.flex_p_up[c, s_m, s_o, p])
                            flex_down = pe.value(model.flex_p_down[c, s_m, s_o, p])
                            obj_scenario += c_flex[s_m][p] * network.baseMVA * (flex_down + flex_up)

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        for p in model.periods:
                            pc_curt = pe.value(model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p])
                            qc_curt = pe.value(model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p])
                            obj_scenario += cost_load_curt * network.baseMVA * (pc_curt + qc_curt)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = pe.value(model.pg_curt_down[g, s_m, s_o, p] + model.pg_curt_up[g, s_m, s_o, p])
                            qg_curt = pe.value(model.qg_curt_down[g, s_m, s_o, p] + model.qg_curt_up[g, s_m, s_o, p])
                            obj_scenario += cost_res_curt * network.baseMVA * (pg_curt + qg_curt)

                obj += obj_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        pen_gen_curtailment = pe.value(model.penalty_gen_curtailment)
        pen_load_curtailment = pe.value(model.penalty_load_curtailment)
        pen_flex_usage = pe.value(model.penalty_flex_usage)

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:

                obj_scenario = 0.0

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        for p in model.periods:
                            pg_curt = pe.value(model.pg_curt_down[g, s_m, s_o, p] + model.pg_curt_up[g, s_m, s_o, p])
                            qg_curt = pe.value(model.qg_curt_down[g, s_m, s_o, p] + model.qg_curt_up[g, s_m, s_o, p])
                            obj_scenario += pen_gen_curtailment * network.baseMVA * (pg_curt + qg_curt)

                # Consumption curtailment
                if params.l_curt:
                    for c in model.loads:
                        for p in model.periods:
                            pc_curt = pe.value(model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p])
                            qc_curt = pe.value(model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p])
                            obj_scenario += pen_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        for p in model.periods:
                            flex_p_up = pe.value(model.flex_p_up[c, s_m, s_o, p])
                            flex_p_down = pe.value(model.flex_p_down[c, s_m, s_o, p])
                            obj_scenario += pen_flex_usage * network.baseMVA * (flex_p_down + flex_p_up)

                obj += obj_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return obj


def _compute_generation_cost(network, model):

    gen_cost = 0.0

    c_p = network.cost_energy_p

    for s_m in model.scenarios_market:
        if s_m in c_p:      # Note: Only exists for COST minimization
            for s_o in model.scenarios_operation:
                gen_cost_scenario = 0.0
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        for p in model.periods:
                            gen_cost_scenario += c_p[s_m][p] * network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                gen_cost += gen_cost_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return gen_cost


def _compute_total_load(network, model, params):

    total_load = {'p': 0.00, 'q': 0.00}

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_load_scenario = {'p': 0.00, 'q': 0.00}
            for c in model.loads:
                for p in model.periods:
                    total_load_scenario['p'] += network.baseMVA * pe.value(model.pc[c, s_m, s_o, p])
                    total_load_scenario['q'] += network.baseMVA * pe.value(model.qc[c, s_m, s_o, p])
                    if params.l_curt:
                        total_load_scenario['p'] -= network.baseMVA * pe.value(model.pc_curt_down[c, s_m, s_o, p] - model.pc_curt_up[c, s_m, s_o, p])
                        total_load_scenario['q'] -= network.baseMVA * pe.value(model.qc_curt_down[c, s_m, s_o, p] - model.qc_curt_up[c, s_m, s_o, p])

            total_load['p'] += total_load_scenario['p'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])
            total_load['q'] += total_load_scenario['q'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_load


def _compute_total_generation(network, model, params):

    total_gen = {'p': 0.00, 'q': 0.00}

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_gen_scenario = {'p': 0.00, 'q': 0.00}
            for g in model.generators:
                for p in model.periods:
                    total_gen_scenario['p'] += network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                    total_gen_scenario['q'] += network.baseMVA * pe.value(model.qg[g, s_m, s_o, p])
                    if params.rg_curt:
                        total_gen_scenario['p'] -= network.baseMVA * pe.value(model.pg_curt_down[g, s_m, s_o, p] + model.pg_curt_up[g, s_m, s_o, p])
                        total_gen_scenario['q'] -= network.baseMVA * pe.value(model.qg_curt_down[g, s_m, s_o, p] + model.qg_curt_up[g, s_m, s_o, p])

            total_gen['p'] += total_gen_scenario['p'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])
            total_gen['q'] += total_gen_scenario['q'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_gen


def _compute_conventional_generation(network, model, params):

    total_gen = {'p': 0.00, 'q': 0.00}

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_gen_scenario = {'p': 0.00, 'q': 0.00}
            for g in model.generators:
                if network.generators[g].gen_type == GEN_CONV:
                    for p in model.periods:
                        total_gen_scenario['p'] += network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                        total_gen_scenario['q'] += network.baseMVA * pe.value(model.qg[g, s_m, s_o, p])
                        if params.rg_curt:
                            total_gen_scenario['p'] -= network.baseMVA * pe.value(model.pg_curt_down[g, s_m, s_o, p] - model.pg_curt_up[g, s_m, s_o, p])
                            total_gen_scenario['q'] -= network.baseMVA * pe.value(model.qg_curt_down[g, s_m, s_o, p] - model.qg_curt_up[g, s_m, s_o, p])

            total_gen['p'] += total_gen_scenario['p'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])
            total_gen['q'] += total_gen_scenario['q'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_gen


def _compute_renewable_generation(network, model, params):

    total_renewable_gen = {'p': 0.00, 'q': 0.00}

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_renewable_gen_scenario = {'p': 0.00, 'q': 0.00}
            for g in model.generators:
                if network.generators[g].is_renewable():
                    for p in model.periods:
                        total_renewable_gen_scenario['p'] += network.baseMVA * pe.value(model.pg[g, s_m, s_o, p])
                        total_renewable_gen_scenario['q'] += network.baseMVA * pe.value(model.qg[g, s_m, s_o, p])
                        if params.rg_curt:
                            total_renewable_gen_scenario['p'] -= network.baseMVA * pe.value(model.pg_curt_down[g, s_m, s_o, p] - model.pg_curt_up[g, s_m, s_o, p])
                            total_renewable_gen_scenario['p'] -= network.baseMVA * pe.value(model.qg_curt_down[g, s_m, s_o, p] - model.qg_curt_up[g, s_m, s_o, p])

            total_renewable_gen['p'] += total_renewable_gen_scenario['p'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])
            total_renewable_gen['q'] += total_renewable_gen_scenario['q'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return total_renewable_gen


def _compute_losses(network, model, params):

    power_losses = 0.0

    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            power_losses_scenario = 0.0
            for k in model.branches:
                for p in model.periods:
                    power_losses_scenario += _get_branch_power_losses(network, params, model, k, s_m, s_o, p)

            power_losses += power_losses_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return power_losses


def _compute_generation_curtailment(network, model, params):

    gen_curtailment = {'p': 0.00, 'q': 0.00}

    if params.rg_curt:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                gen_curtailment_scenario = {'p': 0.00, 'q': 0.00}
                for g in model.generators:
                    if network.generators[g].is_curtaillable():
                        for p in model.periods:
                            gen_curtailment_scenario['p'] += pe.value(model.pg_curt_down[g, s_m, s_o, p] - model.pg_curt_up[g, s_m, s_o, p]) * network.baseMVA
                            gen_curtailment_scenario['q'] += pe.value(model.qg_curt_down[g, s_m, s_o, p] - model.qg_curt_up[g, s_m, s_o, p]) * network.baseMVA

                gen_curtailment['p'] += gen_curtailment_scenario['p'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])
                gen_curtailment['q'] += gen_curtailment_scenario['q'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return gen_curtailment


def _compute_load_curtailment(network, model, params):

    load_curtailment = {'p': 0.00, 'q': 0.00}

    if params.l_curt:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                load_curtailment_scenario = {'p': 0.00, 'q': 0.00}
                for c in model.loads:
                    for p in model.periods:
                        load_curtailment_scenario['p'] += pe.value(model.pc_curt_down[c, s_m, s_o, p] - model.pc_curt_up[c, s_m, s_o, p]) * network.baseMVA
                        load_curtailment_scenario['q'] += pe.value(model.qc_curt_down[c, s_m, s_o, p] - model.qc_curt_up[c, s_m, s_o, p]) * network.baseMVA

                load_curtailment['p'] += load_curtailment_scenario['p'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])
                load_curtailment['q'] += load_curtailment_scenario['q'] * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return load_curtailment


def _compute_flexibility_used(network, model, params):

    flexibility_used = 0.0

    if params.fl_reg:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                flexibility_used_scenario = 0.0
                for c in model.loads:
                    for p in model.periods:
                        flexibility_used_scenario += pe.value(model.flex_p_up[c, s_m, s_o, p]) * network.baseMVA
                        flexibility_used_scenario += pe.value(model.flex_p_down[c, s_m, s_o, p]) * network.baseMVA

                flexibility_used += flexibility_used_scenario * (network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o])

    return flexibility_used


# ======================================================================================================================
#   NETWORK diagram functions (plot)
# ======================================================================================================================
def _plot_networkx_diagram(network, data_dir='data'):

    node_labels = {}
    node_voltage_labels = {}
    node_colors = ['lightblue' for _ in network.nodes]

    # Aux - Encapsulated Branch list
    branches = []
    edge_labels = {}
    line_list, open_line_list = [], []
    transf_list, open_transf_list = [], []
    for branch in network.branches:
        if branch.is_transformer:
            branches.append({'type': 'transformer', 'data': branch})
        else:
            branches.append({'type': 'line', 'data': branch})

    # Build graph
    graph = nx.Graph()
    for i in range(len(network.nodes)):
        node = network.nodes[i]
        graph.add_node(node.bus_i)
        node_labels[node.bus_i] = '{}'.format(node.bus_i)
        node_voltage_labels[node.bus_i] = '{} kV'.format(node.base_kv)
        if node.type == BUS_REF:
            node_colors[i] = 'red'
        elif node.type == BUS_PV:
            node_colors[i] = 'green'
        elif network.has_energy_storage_device(node.bus_i):
            node_colors[i] = 'blue'

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
            ratio = '{:.3f}'.format(branch['data'].ratio)
            edge_labels[(branch['data'].fbus, branch['data'].tbus)] = f'1:{ratio}'

    # Plot - coordinates
    pos = nx.spring_layout(graph)
    pos_above, pos_below = {}, {}
    for k, v in pos.items():
        pos_above[k] = (v[0], v[1] + 0.050)
        pos_below[k] = (v[0], v[1] - 0.050)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(graph, ax=ax, pos=pos, node_color=node_colors, node_size=200)
    nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=node_labels, font_size=10)
    nx.draw_networkx_labels(graph, ax=ax, pos=pos_below, labels=node_voltage_labels, font_size=5)
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=line_list, width=1.00, edge_color='black')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=transf_list, width=1.50, edge_color='blue')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_line_list, style='dashed', width=1.00, edge_color='red')
    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_transf_list, style='dashed', width=1.50, edge_color='red')
    nx.draw_networkx_edge_labels(graph, ax=ax, pos=pos, edge_labels=edge_labels, font_size=5, rotate=False)
    plt.axis('off')

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}.pdf')
    plt.savefig(filename, bbox_inches='tight')

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}.png')
    plt.savefig(filename, bbox_inches='tight')


# ======================================================================================================================
#   Other (aux) functions
# ======================================================================================================================
def _perform_network_check(network):

    n_bus = len(network.nodes)
    if n_bus == 0:
        print(f'[ERROR] Reading network {network.name}. No nodes imported.')
        exit(ERROR_NETWORK_FILE)

    n_branch = len(network.branches)
    if n_branch == 0:
        print(f'[ERROR] Reading network {network.name}. No branches imported.')
        exit(ERROR_NETWORK_FILE)


def _pre_process_network(network):

    processed_nodes = []
    for node in network.nodes:
        if node.type != BUS_ISOLATED:
            processed_nodes.append(node)

    processed_gens = []
    for gen in network.generators:
        node_type = network.get_node_type(gen.bus)
        if node_type != BUS_ISOLATED:
            processed_gens.append(gen)

    processed_branches = []
    for branch in network.branches:

        if not branch.is_connected():  # If branch is disconnected for all days and periods, remove
            continue

        if branch.pre_processed:
            continue

        fbus, tbus = branch.fbus, branch.tbus
        fnode_type = network.get_node_type(fbus)
        tnode_type = network.get_node_type(tbus)
        if fnode_type == BUS_ISOLATED or tnode_type == BUS_ISOLATED:
            branch.pre_processed = True
            continue

        parallel_branches = [branch for branch in network.branches if ((branch.fbus == fbus and branch.tbus == tbus) or (branch.fbus == tbus and branch.tbus == fbus))]
        connected_parallel_branches = [branch for branch in parallel_branches if branch.is_connected()]
        if len(connected_parallel_branches) > 1:
            processed_branch = connected_parallel_branches[0]
            r_eq, x_eq, g_eq, b_eq = _pre_process_parallel_branches(connected_parallel_branches)
            processed_branch.r = r_eq
            processed_branch.x = x_eq
            processed_branch.g_sh = g_eq
            processed_branch.b_sh = b_eq
            processed_branch.rate = sum([branch.rate for branch in connected_parallel_branches])
            processed_branch.ratio = branch.ratio
            processed_branch.pre_processed = True
            for branch_parallel in parallel_branches:
                branch_parallel.pre_processed = True
            processed_branches.append(processed_branch)
        else:
            for branch_parallel in parallel_branches:
                branch_parallel.pre_processed = True
            for branch_parallel in connected_parallel_branches:
                processed_branches.append(branch_parallel)

    network.nodes = processed_nodes
    network.generators = processed_gens
    network.branches = processed_branches
    for branch in network.branches:
        branch.pre_processed = False


def _pre_process_parallel_branches(branches):
    branch_impedances = [complex(branch.r, branch.x) for branch in branches]
    branch_shunt_admittance = [complex(branch.g_sh, branch.b_sh) for branch in branches]
    z_eq = 1/sum([(1/impedance) for impedance in branch_impedances])
    ysh_eq = sum([admittance for admittance in branch_shunt_admittance])
    return abs(z_eq.real), abs(z_eq.imag), ysh_eq.real, ysh_eq.imag


def _get_branch_power_losses(network, params, model, branch_idx, s_m, s_o, p):

    # Active power flow, from i to j and from j to i
    branch = network.branches[branch_idx]
    pij, _ = _get_branch_power_flow(network, params, branch, branch.fbus, branch.tbus, model, s_m, s_o, p)
    pji, _ = _get_branch_power_flow(network, params, branch, branch.tbus, branch.fbus, model, s_m, s_o, p)

    return abs(pij + pji)


def _get_branch_power_flow(network, params, branch, fbus, tbus, model, s_m, s_o, p):

    fbus_idx = network.get_node_idx(fbus)
    tbus_idx = network.get_node_idx(tbus)
    branch_idx = network.get_branch_idx(branch)

    rij = pe.value(model.r[branch_idx, s_m, s_o, p])
    ei = pe.value(model.e_actual[fbus_idx, s_m, s_o, p])
    fi = pe.value(model.f_actual[fbus_idx, s_m, s_o, p])
    ej = pe.value(model.e_actual[tbus_idx, s_m, s_o, p])
    fj = pe.value(model.f_actual[tbus_idx, s_m, s_o, p])

    if branch.fbus == fbus:
        pij = branch.g * (ei ** 2 + fi ** 2) * rij ** 2
        pij -= branch.g * (ei * ej + fi * fj) * rij
        pij -= branch.b * (fi * ej - ei * fj) * rij

        qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2) * rij ** 2
        qij += branch.b * (ei * ej + fi * fj) * rij
        qij -= branch.g * (fi * ej - ei * fj) * rij
    else:
        pij = branch.g * (ei ** 2 + fi ** 2)
        pij -= branch.g * (ei * ej + fi * fj) * rij
        pij -= branch.b * (fi * ej - ei * fj) * rij

        qij = - (branch.b + branch.b_sh * 0.50) * (ei ** 2 + fi ** 2)
        qij += branch.b * (ei * ej + fi * fj) * rij
        qij -= branch.g * (fi * ej - ei * fj) * rij

    return pij * network.baseMVA, qij * network.baseMVA


def _get_info_from_results(results, info_string):
    i = str(results).lower().find(info_string.lower()) + len(info_string)
    value = ''
    while str(results)[i] != '\n':
        value = value + str(results)[i]
        i += 1
    return value
