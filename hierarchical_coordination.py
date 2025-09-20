from copy import copy, deepcopy
import numpy as np
from scipy.spatial import ConvexHull
import pyomo.environ as pe
from math import sqrt, acos, tan, atan2
from helper_functions import *
from model_construction_helpers import *


def build_model_single_period(network, t, original_params):

    network.compute_series_admittance()

    # Note: update params to not consider ESSs and FLs
    params = deepcopy(original_params)
    params.fl_reg = False
    params.es_reg = False

    model = pe.ConcreteModel()
    model.name = f'{network.name}_{network.year}_{network.day}_t={t}'

    # ------------------------------------------------------------------------------------------------------------------
    # Sets
    model.scenarios_market = range(len(network.prob_market_scenarios))
    model.scenarios_operation = range(len(network.prob_operation_scenarios))
    model.nodes = range(len(network.nodes))
    model.loads = range(len(network.loads))
    model.generators = range(len(network.generators))
    model.branches = range(len(network.branches))
    model.energy_storages = range(len(network.energy_storages))

    # ------------------------------------------------------------------------------------------------------------------
    # Decision variables
    # - Voltage
    model.e = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=1.0)
    model.f = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    model.e_actual = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=1.0)
    model.f_actual = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    if params.slacks.grid_operation.voltage:
        model.slack_e = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.00)
        model.slack_f = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.00)
    for i in model.nodes:
        node = network.nodes[i]
        e_lb, e_ub = -node.v_max, node.v_max
        f_lb, f_ub = -node.v_max, node.v_max
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                if params.slacks.grid_operation.voltage:
                    model.slack_e[i, s_m, s_o].setub(VMAG_VIOLATION_ALLOWED)
                    model.slack_e[i, s_m, s_o].setlb(-VMAG_VIOLATION_ALLOWED)
                    model.slack_f[i, s_m, s_o].setub(VMAG_VIOLATION_ALLOWED)
                    model.slack_f[i, s_m, s_o].setlb(-VMAG_VIOLATION_ALLOWED)
                if node.type == BUS_REF:
                    if network.is_transmission:
                        model.e[i, s_m, s_o].setub(e_ub)
                        model.e[i, s_m, s_o].setlb(e_lb)
                    else:
                        ref_gen_idx = network.get_gen_idx(node.bus_i)
                        vg = network.generators[ref_gen_idx].vg
                        model.e[i, s_m, s_o].setub(vg + EQUALITY_TOLERANCE)
                        model.e[i, s_m, s_o].setlb(vg - EQUALITY_TOLERANCE)
                        if params.slacks.grid_operation.voltage:
                            model.slack_e[i, s_m, s_o].setub(EQUALITY_TOLERANCE)
                            model.slack_e[i, s_m, s_o].setlb(-EQUALITY_TOLERANCE)
                    model.f[i, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    model.f[i, s_m, s_o].setlb(-EQUALITY_TOLERANCE)
                    if params.slacks.grid_operation.voltage:
                        model.slack_f[i, s_m, s_o].setub(EQUALITY_TOLERANCE)
                        model.slack_f[i, s_m, s_o].setlb(-EQUALITY_TOLERANCE)
                else:
                    model.e[i, s_m, s_o].setub(e_ub)
                    model.e[i, s_m, s_o].setlb(e_lb)
                    model.f[i, s_m, s_o].setub(f_ub)
                    model.f[i, s_m, s_o].setlb(f_lb)
    if params.slacks.node_balance:
        model.slack_node_balance_p = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.00)
        model.slack_node_balance_q = pe.Var(model.nodes, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.00)

    # - Generation
    model.pg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    model.qg = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
    for g in model.generators:
        generator = network.generators[g]
        pg_ub, pg_lb = generator.pmax, generator.pmin
        qg_ub, qg_lb = generator.qmax, generator.qmin
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                if generator.status[t]:
                    model.pg[g, s_m, s_o] = max(pg_lb, 0.00)
                    model.qg[g, s_m, s_o] = max(qg_lb, 0.00)
                    model.pg[g, s_m, s_o].setub(pg_ub)
                    model.pg[g, s_m, s_o].setlb(pg_lb)
                    model.qg[g, s_m, s_o].setub(qg_ub)
                    model.qg[g, s_m, s_o].setlb(qg_lb)
                else:
                    model.pg[g, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    model.pg[g, s_m, s_o].setlb(-EQUALITY_TOLERANCE)
                    model.qg[g, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    model.qg[g, s_m, s_o].setlb(-EQUALITY_TOLERANCE)
    if params.rg_curt:
        model.sg_abs = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.sg_sqr = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.sg_curt = pe.Var(model.generators, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for g in model.generators:
            generator = network.generators[g]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    if generator.is_curtaillable():
                        # - Renewable Generation
                        init_sg = 0.0
                        if generator.status[t]:
                            init_sg = sqrt(generator.pg[s_o][t] ** 2 + generator.qg[s_o][t] ** 2)
                        model.sg_abs[g, s_m, s_o].setub(init_sg)
                        model.sg_sqr[g, s_m, s_o].setub(init_sg ** 2)
                        model.sg_curt[g, s_m, s_o].setub(init_sg)
                    else:
                        model.sg_abs[g, s_m, s_o].setub(EQUALITY_TOLERANCE)
                        model.sg_sqr[g, s_m, s_o].setub(EQUALITY_TOLERANCE)
                        model.sg_curt[g, s_m, s_o].setub(EQUALITY_TOLERANCE)

    # - Branch power flows (squared) -- used in branch limits
    model.flow_ij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
    if params.slacks.grid_operation.branch_flow:
        model.slack_flow_ij_sqr = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
    for b in model.branches:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                if network.branches[b].status:
                    if params.slacks.grid_operation.branch_flow:
                        rating = network.branches[b].rate / network.baseMVA
                        model.slack_flow_ij_sqr[b, s_m, s_o].setub(SIJ_VIOLATION_ALLOWED * rating)
                else:
                    model.flow_ij_sqr[b, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    if params.slacks.grid_operation.branch_flow:
                        model.slack_flow_ij_sqr[b, s_m, s_o].setub(EQUALITY_TOLERANCE)

    # - Loads
    model.pc = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.Reals)
    model.qc = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.Reals)
    for c in model.loads:
        load = network.loads[c]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                model.pc[c, s_m, s_o].setub(load.pd[s_o][t] + EQUALITY_TOLERANCE)
                model.pc[c, s_m, s_o].setlb(load.pd[s_o][t] - EQUALITY_TOLERANCE)
                model.qc[c, s_m, s_o].setub(load.qd[s_o][t] + EQUALITY_TOLERANCE)
                model.qc[c, s_m, s_o].setlb(load.qd[s_o][t] - EQUALITY_TOLERANCE)
    if params.fl_reg:
        model.flex_p_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_p_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_q_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.flex_q_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for c in model.loads:
            load = network.loads[c]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    if load.fl_reg:
                        flex_up = load.flexibility.active_power.upward[s_o][t]
                        flex_down = load.flexibility.active_power.downward[s_o][t]
                        model.flex_p_up[c, s_m, s_o].setub(abs(flex_up))
                        model.flex_p_down[c, s_m, s_o].setub(abs(flex_down))
                    else:
                        model.flex_p_up[c, s_m, s_o].setub(EQUALITY_TOLERANCE)
                        model.flex_p_down[c, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    model.flex_q_up[c, s_m, s_o].setub(EQUALITY_TOLERANCE)  # Note: used for coordinated operation
                    model.flex_q_down[c, s_m, s_o].setub(EQUALITY_TOLERANCE)
    if params.l_curt:
        model.pc_curt_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.pc_curt_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt_down = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.qc_curt_up = pe.Var(model.loads, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        for c in model.loads:
            load = network.loads[c]
            for s_m in network.scenarios_market:
                for s_o in model.scenarios_operation:
                    if load.pd[s_o][t] >= 0.00:
                        model.pc_curt_down[c, s_m, s_o].setub(abs(load.pd[s_o][t]))
                        model.pc_curt_up[c, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    else:
                        model.pc_curt_up[c, s_m, s_o].setub(abs(load.pd[s_o][t]))
                        model.pc_curt_down[c, s_m, s_o].setub(EQUALITY_TOLERANCE)

                    if load.qd[s_o][t] >= 0.00:
                        model.qc_curt_down[c, s_m, s_o].setub(abs(load.qd[s_o][t]))
                        model.qc_curt_up[c, s_m, s_o].setub(EQUALITY_TOLERANCE)
                    else:
                        model.qc_curt_up[c, s_m, s_o].setub(abs(load.qd[s_o][t]))
                        model.qc_curt_down[c, s_m, s_o].setub(EQUALITY_TOLERANCE)

    # - Transformers
    model.r = pe.Var(model.branches, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=1.0)
    for i in model.branches:
        branch = network.branches[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                if branch.is_transformer:
                    # - Transformer
                    if params.transf_reg and branch.vmag_reg:
                        model.r[i, s_m, s_o].setub(TRANSFORMER_MAXIMUM_RATIO)
                        model.r[i, s_m, s_o].setlb(TRANSFORMER_MINIMUM_RATIO)
                    else:
                        model.r[i, s_m, s_o].setub(branch.ratio + EQUALITY_TOLERANCE)
                        model.r[i, s_m, s_o].setlb(branch.ratio - EQUALITY_TOLERANCE)
                else:
                    model.r[i, s_m, s_o].setub(1.00 + EQUALITY_TOLERANCE)
                    model.r[i, s_m, s_o].setlb(1.00 - EQUALITY_TOLERANCE)

    # - Energy Storage devices
    if params.es_reg:
        model.es_soc = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.00)
        model.es_sch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_qch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
        model.es_sdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_pdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)
        model.es_qdch = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.Reals, initialize=0.0)
        for e in model.energy_storages:
            energy_storage = network.energy_storages[e]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    model.es_soc[e, s_m, s_o] = energy_storage.e_init
                    model.es_soc[e, s_m, s_o].setlb(energy_storage.e_min)
                    model.es_soc[e, s_m, s_o].setub(energy_storage.e_max)
                    model.es_sch[e, s_m, s_o].setub(energy_storage.s)
                    model.es_pch[e, s_m, s_o].setub(energy_storage.s)
                    model.es_qch[e, s_m, s_o].setub(energy_storage.s)
                    model.es_qch[e, s_m, s_o].setlb(-energy_storage.s)
                    model.es_sdch[e, s_m, s_o].setub(energy_storage.s)
                    model.es_pdch[e, s_m, s_o].setub(energy_storage.s)
                    model.es_qdch[e, s_m, s_o].setub(energy_storage.s)
                    model.es_qdch[e, s_m, s_o].setlb(-energy_storage.s)
        if params.slacks.ess.complementarity:
            model.slack_es_comp = pe.Var(model.energy_storages, model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)

      # ------------------------------------------------------------------------------------------------------------------
    # Constraints
    # - Voltage
    model.voltage_cons = pe.ConstraintList()
    for i in model.nodes:
        node = network.nodes[i]
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                # e_actual and f_actual definition
                e_actual = model.e[i, s_m, s_o]
                f_actual = model.f[i, s_m, s_o]
                if params.slacks.grid_operation.voltage:
                    e_actual += model.slack_e[i, s_m, s_o]
                    f_actual += model.slack_f[i, s_m, s_o]

                model.voltage_cons.add(model.e_actual[i, s_m, s_o] == e_actual)
                model.voltage_cons.add(model.f_actual[i, s_m, s_o] == f_actual)

                # voltage magnitude constraints
                if node.type == BUS_PV:
                    if params.enforce_vg:
                        # - Enforce voltage controlled bus
                        gen_idx = network.get_gen_idx(node.bus_i)
                        vg = network.generators[gen_idx].vg
                        e = model.e[i, s_m, s_o]
                        f = model.f[i, s_m, s_o]
                        model.voltage_cons.add(e ** 2 + f ** 2 <= vg[t] ** 2 + EQUALITY_TOLERANCE)
                        model.voltage_cons.add(e ** 2 + f ** 2 >= vg[t] ** 2 - EQUALITY_TOLERANCE)
                    else:
                        # - Voltage at the bus is not controlled
                        e = model.e[i, s_m, s_o]
                        f = model.f[i, s_m, s_o]
                        model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min ** 2)
                        model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max ** 2)
                else:
                    e = model.e[i, s_m, s_o]
                    f = model.f[i, s_m, s_o]
                    model.voltage_cons.add(e ** 2 + f ** 2 >= node.v_min ** 2)
                    model.voltage_cons.add(e ** 2 + f ** 2 <= node.v_max ** 2)

    model.generation_apparent_power = pe.ConstraintList()
    model.generation_power_factor = pe.ConstraintList()
    if params.rg_curt:
        for g in model.generators:
            generator = network.generators[g]
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    if generator.is_curtaillable():
                        init_sg = 0.0
                        if generator.status[t]:
                            init_sg = sqrt(generator.pg[s_o][t] ** 2 + generator.qg[s_o][t] ** 2)
                        model.generation_apparent_power.add(model.sg_sqr[g, s_m, s_o] <= model.pg[g, s_m, s_o] ** 2 + model.qg[g, s_m, s_o] ** 2 + EQUALITY_TOLERANCE)
                        model.generation_apparent_power.add(model.sg_sqr[g, s_m, s_o] >= model.pg[g, s_m, s_o] ** 2 + model.qg[g, s_m, s_o] ** 2 - EQUALITY_TOLERANCE)
                        model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o] ** 2 <= model.sg_sqr[g, s_m, s_o] + EQUALITY_TOLERANCE)
                        model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o] ** 2 >= model.sg_sqr[g, s_m, s_o] - EQUALITY_TOLERANCE)
                        model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o] <= init_sg - model.sg_curt[g, s_m, s_o] + EQUALITY_TOLERANCE)
                        model.generation_apparent_power.add(model.sg_abs[g, s_m, s_o] >= init_sg - model.sg_curt[g, s_m, s_o] - EQUALITY_TOLERANCE)
                        if generator.power_factor_control:
                            # Power factor control, variable phi
                            max_phi = acos(generator.max_pf)
                            min_phi = acos(generator.min_pf)
                            model.generation_power_factor.add(model.qg[g, s_m, s_o] <= tan(max_phi) * model.pg[g, s_m, s_o])
                            model.generation_power_factor.add(model.qg[g, s_m, s_o] >= tan(min_phi) * model.pg[g, s_m, s_o])
                        else:
                            # No power factor control, maintain given phi
                            phi = atan2(generator.qg[s_o][t], generator.pg[s_o][t])
                            model.generation_power_factor.add(model.qg[g, s_m, s_o] <= tan(phi) * model.pg[g, s_m, s_o])
                            model.generation_power_factor.add(model.qg[g, s_m, s_o] >= tan(phi) * model.pg[g, s_m, s_o])

    # - Energy Storage constraints
    if params.es_reg:

        model.energy_storage_balance = pe.ConstraintList()
        model.energy_storage_operation = pe.ConstraintList()
        model.energy_storage_ch_dch_exclusion = pe.ConstraintList()

        for e in model.energy_storages:

            energy_storage = network.energy_storages[e]
            soc_init = energy_storage.e_init
            eff_charge = energy_storage.eff_ch
            eff_discharge = energy_storage.eff_dch
            max_phi = acos(energy_storage.max_pf)
            min_phi = acos(energy_storage.min_pf)

            for s_m in model.scenarios_operation:
                for s_o in model.scenarios_operation:

                    sch = model.es_sch[e, s_m, s_o]
                    pch = model.es_pch[e, s_m, s_o]
                    qch = model.es_qch[e, s_m, s_o]
                    sdch = model.es_sdch[e, s_m, s_o]
                    pdch = model.es_pdch[e, s_m, s_o]
                    qdch = model.es_qdch[e, s_m, s_o]

                    # ESS operation
                    model.energy_storage_operation.add(qch <= tan(max_phi) * pch)
                    model.energy_storage_operation.add(qch >= tan(min_phi) * pch)
                    model.energy_storage_operation.add(qdch <= tan(max_phi) * pdch)
                    model.energy_storage_operation.add(qdch >= tan(min_phi) * pdch)

                    if params.slacks.ess.charging:
                        model.energy_storage_operation.add(sch ** 2 == pch ** 2 + qch ** 2 + model.slack_es_ch[e, s_o])
                        model.energy_storage_operation.add(sdch ** 2 == pdch ** 2 + qdch ** 2 + model.slack_es_dch[e, s_o])
                    else:
                        model.energy_storage_operation.add(sch ** 2 <= pch ** 2 + qch ** 2 + EQUALITY_TOLERANCE)
                        model.energy_storage_operation.add(sch ** 2 >= pch ** 2 + qch ** 2 - EQUALITY_TOLERANCE)
                        model.energy_storage_operation.add(sdch ** 2 <= pdch ** 2 + qdch ** 2 + EQUALITY_TOLERANCE)
                        model.energy_storage_operation.add(sdch ** 2 >= pdch ** 2 + qdch ** 2 - EQUALITY_TOLERANCE)

                    # Charging/discharging complementarity constraints
                    if params.slacks.ess.complementarity:
                        model.energy_storage_ch_dch_exclusion.add(sch * sdch == model.slack_es_comp[e, s_m, s_o])
                    else:
                        model.energy_storage_ch_dch_exclusion.add(sch * sdch <= EQUALITY_TOLERANCE)

                    # State-of-Charge
                    soc_prev = soc_init
                    model.energy_storage_balance.add(model.es_soc[e, s_o] <= soc_prev + (sch * eff_charge - sdch / eff_discharge) + EQUALITY_TOLERANCE)
                    model.energy_storage_balance.add(model.es_soc[e, s_o] >= soc_prev + (sch * eff_charge - sdch / eff_discharge) - EQUALITY_TOLERANCE)

    # - Node Balance constraints
    model.node_balance_cons_p = pe.ConstraintList()
    model.node_balance_cons_q = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for i in range(len(network.nodes)):

                node = network.nodes[i]

                Pd = 0.00
                Qd = 0.00
                for c in model.loads:
                    if network.loads[c].bus == node.bus_i:
                        Pd += model.pc[c, s_m, s_o]
                        Qd += model.qc[c, s_m, s_o]
                        if params.fl_reg and network.loads[c].fl_reg:
                            Pd += (model.flex_p_up[c, s_m, s_o] - model.flex_p_down[c, s_m, s_o])
                            Qd += (model.flex_q_up[c, s_m, s_o] - model.flex_q_down[c, s_m, s_o])
                        if params.l_curt:
                            Pd -= (model.pc_curt_down[c, s_m, s_o] - model.pc_curt_up[c, s_m, s_o])
                            Qd -= (model.qc_curt_down[c, s_m, s_o] - model.qc_curt_up[c, s_m, s_o])
                if params.es_reg:
                    for e in model.energy_storages:
                        if network.energy_storages[e].bus == node.bus_i:
                            Pd += (model.es_pch[e, s_m, s_o] - model.es_pdch[e, s_m, s_o])
                            Qd += (model.es_qch[e, s_m, s_o] - model.es_qdch[e, s_m, s_o])

                Pg = 0.0
                Qg = 0.0
                for g in model.generators:
                    generator = network.generators[g]
                    if generator.bus == node.bus_i:
                        Pg += model.pg[g, s_m, s_o]
                        Qg += model.qg[g, s_m, s_o]

                ei = model.e_actual[i, s_m, s_o]
                fi = model.f_actual[i, s_m, s_o]

                Pi = node.gs * (ei ** 2 + fi ** 2)
                Qi = -node.bs * (ei ** 2 + fi ** 2)
                for b in range(len(network.branches)):
                    branch = network.branches[b]
                    if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                        rij = model.r[b, s_m, s_o]
                        if not branch.is_transformer:
                            rij = 1.00

                        if branch.fbus == node.bus_i:
                            fnode_idx = network.get_node_idx(branch.fbus)
                            tnode_idx = network.get_node_idx(branch.tbus)

                            ei = model.e_actual[fnode_idx, s_m, s_o]
                            fi = model.f_actual[fnode_idx, s_m, s_o]
                            ej = model.e_actual[tnode_idx, s_m, s_o]
                            fj = model.f_actual[tnode_idx, s_m, s_o]

                            Pi += branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                            Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                            Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2) * rij ** 2
                            Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))
                        else:
                            fnode_idx = network.get_node_idx(branch.tbus)
                            tnode_idx = network.get_node_idx(branch.fbus)

                            ei = model.e_actual[fnode_idx, s_m, s_o]
                            fi = model.f_actual[fnode_idx, s_m, s_o]
                            ej = model.e_actual[tnode_idx, s_m, s_o]
                            fj = model.f_actual[tnode_idx, s_m, s_o]

                            Pi += branch.g * (ei ** 2 + fi ** 2)
                            Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                            Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2)
                            Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

                if params.slacks.node_balance:
                    model.node_balance_cons_p.add(Pg == Pd + Pi + model.slack_node_balance_p[i, s_m, s_o])
                    model.node_balance_cons_q.add(Qg == Qd + Qi + model.slack_node_balance_q[i, s_m, s_o])
                else:
                    model.node_balance_cons_p.add(Pg <= Pd + Pi + EQUALITY_TOLERANCE)
                    model.node_balance_cons_p.add(Pg >= Pd + Pi - EQUALITY_TOLERANCE)
                    model.node_balance_cons_q.add(Qg <= Qd + Qi + EQUALITY_TOLERANCE)
                    model.node_balance_cons_q.add(Qg >= Qd + Qi - EQUALITY_TOLERANCE)

    # - Branch Power Flow constraints (current)
    model.branch_power_flow_cons = pe.ConstraintList()
    model.branch_power_flow_lims = pe.ConstraintList()
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for b in model.branches:

                branch = network.branches[b]
                rating = branch.rate / network.baseMVA
                if rating == 0.0:
                    rating = BRANCH_UNKNOWN_RATING
                fnode_idx = network.get_node_idx(branch.fbus)
                tnode_idx = network.get_node_idx(branch.tbus)

                rij = model.r[b, s_m, s_o]
                if not branch.is_transformer:
                    rij = 1.00
                ei = model.e_actual[fnode_idx, s_m, s_o]
                fi = model.f_actual[fnode_idx, s_m, s_o]
                ej = model.e_actual[tnode_idx, s_m, s_o]
                fj = model.f_actual[tnode_idx, s_m, s_o]

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
                model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o] <= flow_ij_sqr + EQUALITY_TOLERANCE)
                model.branch_power_flow_cons.add(model.flow_ij_sqr[b, s_m, s_o] >= flow_ij_sqr - EQUALITY_TOLERANCE)

                # Branch flow limits
                if branch.status:
                    if params.slacks.grid_operation.branch_flow:
                        model.branch_power_flow_lims.add(model.flow_ij_sqr[b, s_m, s_o] <= rating ** 2 + model.slack_flow_ij_sqr[b, s_m, s_o])
                    else:
                        model.branch_power_flow_lims.add(model.flow_ij_sqr[b, s_m, s_o] <= rating ** 2)

    # ------------------------------------------------------------------------------------------------------------------
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

    # Objective Function
    obj = 0.0
    if params.obj_type == OBJ_MIN_COST:

        # Cost minimization
        c_p = network.cost_energy_p
        c_flex = network.cost_flex

        for s_m in model.scenarios:

            obj_scenario = 0.00
            omega_market = network.prob_market_scenarios[s_m]

            for s_o in model.scenarios_operation:

                omega_oper = network.prob_operation_scenarios[s_o]

                # Generation
                for g in model.generators:
                    if network.generators[g].is_controllable():
                        if (not network.is_transmission) and network.generators[g].gen_type == GEN_REFERENCE:
                            continue
                        pg = model.pg[g, s_m, s_o]
                        obj_scenario += c_p[t] * network.baseMVA * pg

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        pc_flex = (model.flex_p_up[c, s_m, s_o] + model.flex_p_down[c, s_m, s_o])
                        qc_flex = (model.flex_q_up[c, s_m, s_o] + model.flex_q_down[c, s_m, s_o])
                        obj_scenario += c_flex[t] * network.baseMVA * (pc_flex + qc_flex)

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        pc_curt = (model.pc_curt_down[c, s_m, s_o] + model.pc_curt_up[c, s_m, s_o])
                        qc_curt = (model.qc_curt_down[c, s_m, s_o] + model.qc_curt_up[c, s_m, s_o])
                        obj_scenario += model.cost_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Generation curtailment
                if params.rg_curt:
                    for g in model.generators:
                        if network.generators[g].is_curtaillable():
                            sg_curt = model.sg_curt[g, s_m, s_o]
                            obj_scenario += model.cost_res_curtailment * network.baseMVA * sg_curt

                # ESS utilization
                if params.es_reg:
                    for e in model.energy_storages:
                        sch = model.es_sch[e, s_m, s_o]
                        sdch = model.es_sdch[e, s_m, s_o]
                        obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                obj += obj_scenario * omega_oper * omega_market

    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:

        # Congestion Management
        for s_m in model.scenarios_market:

            obj_scenario = 0.00
            omega_market = network.prob_market_scenarios[s_m]

            for s_o in model.scenarios_operation:

                omega_oper = network.prob_operation_scenarios[s_o]

                # Load curtailment
                if params.l_curt:
                    for c in model.loads:
                        pc_curt = (model.pc_curt_down[c, s_m, s_o] + model.pc_curt_up[c, s_m, s_o])
                        qc_curt = (model.qc_curt_down[c, s_m, s_o] + model.qc_curt_up[c, s_m, s_o])
                        obj_scenario += model.penalty_load_curtailment * network.baseMVA * (pc_curt + qc_curt)

                # Demand side flexibility
                if params.fl_reg:
                    for c in model.loads:
                        pc_flex = (model.flex_p_up[c, s_m, s_o] + model.flex_p_down[c, s_m, s_o])
                        qc_flex = (model.flex_q_up[c, s_m, s_o] + model.flex_q_down[c, s_m, s_o])
                        obj_scenario += model.penalty_flex_usage * network.baseMVA * (pc_flex + qc_flex)

                # ESS utilization
                if params.es_reg:
                    for e in model.energy_storages:
                        sch = model.es_sch[e, s_m, s_o]
                        sdch = model.es_sdch[e, s_m, s_o]
                        obj_scenario += model.penalty_ess_usage * network.baseMVA * (sch + sdch)

                obj += obj_scenario * omega_oper * omega_market

    # Slacks grid operation
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:

            omega_oper = network.prob_operation_scenarios[s_o]

            # Voltage slacks
            if params.slacks.grid_operation.voltage:
                for i in model.nodes:
                    slack_e_sqr = model.slack_e[i, s_m, s_o] ** 2
                    slack_f_sqr = model.slack_f[i, s_m, s_o] ** 2
                    obj += PENALTY_VOLTAGE * 1e3 * network.baseMVA * omega_oper * (slack_e_sqr + slack_f_sqr)

            # Branch power flow slacks
            if params.slacks.grid_operation.branch_flow:
                for b in model.branches:
                    slack_flow_ij_sqr = (model.slack_flow_ij_sqr[b, s_m, s_o])
                    obj += PENALTY_CURRENT * network.baseMVA * omega_oper * slack_flow_ij_sqr

    # Operation slacks
    for s_o in model.scenarios_operation:

        omega_oper = network.prob_operation_scenarios[s_o]

        # Node balance
        if params.slacks.node_balance:
            for i in model.nodes:
                slack_p_sqr = model.slack_node_balance_p[i, s_m, s_o] ** 2
                slack_q_sqr = model.slack_node_balance_q[i, s_m, s_o] ** 2
                obj += PENALTY_NODE_BALANCE * network.baseMVA * omega_oper * (slack_p_sqr + slack_q_sqr)

        # ESS slacks
        if params.es_reg:
            if params.slacks.ess.complementarity:
                slack_comp = model.slack_es_comp[e, s_m, s_o]
                obj += PENALTY_ESS * network.baseMVA * omega_oper * slack_comp

    model.objective = pe.Objective(sense=pe.minimize, expr=obj)

    # Model suffixes (used for warm start)
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)  # Ipopt bound multipliers (obtained from solution)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)  # Ipopt bound multipliers (sent to solver)
    model.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)  # Obtain dual solutions from previous solve and send to warm start

    return model


def get_pq_map(network, params, t=None, num_steps=8, print_pq_map=False):

    pq_map = dict()
    print(f'[INFO] - Determining PQ map, network {network.name}, {network.day}...')

    if t is None:
        for t in range(network.num_instants):
            initial_solution = _get_pq_initial_solution(network, t, params)
            vertices = _get_pq_map_vertices(network, t, num_steps, params)
            hull = ConvexHull(vertices)
            inequalities = _get_pq_map_inequalities(vertices[hull.vertices])
            pq_map[t] = {
                'initial_solution': initial_solution,
                'inequalities': inequalities
            }
    else:
        initial_solution = _get_pq_initial_solution(network, t, params)
        vertices = _get_pq_map_vertices(network, t, num_steps, params)
        hull = ConvexHull(vertices)
        inequalities = _get_pq_map_inequalities(vertices[hull.vertices])
        pq_map[t] = {
            'initial_solution': initial_solution,
            'inequalities': inequalities
        }

    return pq_map


def _build_pq_map_model(network, t, params):

    model = network.build_model_single_period(t, params)
    ref_gen_idx = network.get_reference_gen_idx()

    # Add expected interface power flow variables
    expected_pf_p = 0.00
    expected_pf_q = 0.00
    model.expected_interface_pf_p = pe.Var(domain=pe.Reals, initialize=0.00)
    model.expected_interface_pf_q = pe.Var(domain=pe.Reals, initialize=0.00)
    model.interface_expected_values = pe.ConstraintList()
    for s_m in model.scenarios_market:
        omega_market = network.prob_market_scenarios[s_m]
        for s_o in model.scenarios_operation:
            omega_oper = network.prob_operation_scenarios[s_o]
            expected_pf_p += omega_market * omega_oper * model.pg[ref_gen_idx, s_m, s_o]
            expected_pf_q += omega_market * omega_oper * model.qg[ref_gen_idx, s_m, s_o]
    model.interface_expected_values.add(model.expected_interface_pf_p <= expected_pf_p + EQUALITY_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_p >= expected_pf_p - EQUALITY_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_q <= expected_pf_q + EQUALITY_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_q >= expected_pf_q - EQUALITY_TOLERANCE)

    # New objective function (PQ maps)
    s_base = network.baseMVA
    obj = model.objective.expr
    model.alpha = pe.Var(domain=pe.Reals, initialize=0.00, bounds=(-1.00, 1.00))
    model.beta = pe.Var(domain=pe.Reals, initialize=0.00, bounds=(-1.00, 1.00))
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            omega_oper = network.prob_operation_scenarios[s_o]
            obj += model.alpha * s_base * model.pg[ref_gen_idx, s_m, s_o] * omega_oper
            obj += model.beta * network.baseMVA * model.qg[ref_gen_idx, s_m, s_o] * omega_oper

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    ref_gen_idx = network.get_reference_gen_idx()
    model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_regularization.fix(PENALTY_REGULARIZATION)
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            obj += model.penalty_regularization * s_base * (model.pg[ref_gen_idx, s_m, s_o] - model.expected_interface_pf_p) ** 2
            obj += model.penalty_regularization * s_base * (model.qg[ref_gen_idx, s_m, s_o] - model.expected_interface_pf_q) ** 2

    model.objective.expr = obj

    return model


def _get_pq_map_vertices(network, t, num_steps, params):

    model = _build_pq_map_model(network, t, params)
    vertices = []

    for n in range(num_steps + 1):

        alpha = n/num_steps
        beta = 1 - alpha

        model.alpha.fix(alpha)
        model.beta.fix(beta)
        network.run_smopf(model, params, from_warm_start=True, print_header=False)
        pg = pe.value(model.expected_interface_pf_p) * network.baseMVA
        qg = pe.value(model.expected_interface_pf_q) * network.baseMVA
        vertices.append((pg, qg))

        model.alpha.fix(-alpha)
        model.beta.fix(-beta)
        network.run_smopf(model, params, from_warm_start=True, print_header=False)
        pg = pe.value(model.expected_interface_pf_p) * network.baseMVA
        qg = pe.value(model.expected_interface_pf_q) * network.baseMVA
        vertices.append((pg, qg))

    return np.array(vertices)


def _get_pq_map_inequalities(vertices):

    inequalities = []
    centroid = np.mean(vertices, axis=0)
    n = len(vertices)

    for i in range(n):

        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # Perpendicular vector (normal)
        dx = x2 - x1
        dy = y2 - y1
        a = -dy
        b = dx
        c = a * x1 + b * y1

        # Flip direction if needed
        if a * centroid[0] + b * centroid[1] > c:
            a, b, c = -a, -b, -c

        inequalities.append({'Pg': a, 'Qg': b, 'c': c})

    return inequalities


def _get_pq_initial_solution(network, t, params):

    model = network.build_model_single_period(t, params)
    ref_node_id = network.get_reference_node_id()
    ref_gen_idx = network.get_reference_gen_idx()
    adn_node_idx = network.get_node_idx(ref_node_id)

    # Add expected interface power flow variables
    expected_vmag = 0.00
    expected_pf_p = 0.00
    expected_pf_q = 0.00
    model.expected_interface_vmag = pe.Var(domain=pe.NonNegativeReals, initialize=1.00)
    model.expected_interface_pf_p = pe.Var(domain=pe.Reals, initialize=0.00)
    model.expected_interface_pf_q = pe.Var(domain=pe.Reals, initialize=0.00)
    model.interface_expected_values = pe.ConstraintList()
    for s_m in model.scenarios_market:
        omega_market = network.prob_market_scenarios[s_m]
        for s_o in model.scenarios_operation:
            omega_oper = network.prob_operation_scenarios[s_o]
            expected_vmag += omega_market * omega_oper * model.e[adn_node_idx, s_m, s_o]
            expected_pf_p += omega_market *omega_oper * model.pg[ref_gen_idx, s_m, s_o]
            expected_pf_q += omega_market *omega_oper * model.qg[ref_gen_idx, s_m, s_o]
    model.interface_expected_values.add(model.expected_interface_vmag <= expected_vmag + SMALL_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_vmag >= expected_vmag - SMALL_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_p <= expected_pf_p + EQUALITY_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_p >= expected_pf_p - EQUALITY_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_q <= expected_pf_q + EQUALITY_TOLERANCE)
    model.interface_expected_values.add(model.expected_interface_pf_q >= expected_pf_q - EQUALITY_TOLERANCE)

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    s_base = network.baseMVA
    obj = model.objective.expr
    model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_regularization.fix(PENALTY_REGULARIZATION)
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            obj += model.penalty_regularization * (model.e[adn_node_idx, s_m, s_o] - model.expected_interface_vmag) ** 2
            obj += model.penalty_regularization * s_base * (model.pg[ref_gen_idx, s_m, s_o] - model.expected_interface_pf_p) ** 2
            obj += model.penalty_regularization * s_base * (model.qg[ref_gen_idx, s_m, s_o] - model.expected_interface_pf_q) ** 2
    model.objective.expr = obj

    network.run_smopf(model, params, from_warm_start=True, print_header=False)
    vg = pe.value(model.expected_interface_vmag)
    pg = pe.value(model.expected_interface_pf_p) * network.baseMVA
    qg = pe.value(model.expected_interface_pf_q) * network.baseMVA
    solution = {'Pg': pg, 'Qg': qg, 'Vg': vg}

    return solution


def update_of_to_settlement(network, model, params):

    s_base = network.baseMVA
    ref_node_id = network.get_reference_node_id()
    ref_node_idx = network.get_node_idx(ref_node_id)
    ref_gen_idx = network.get_reference_gen_idx()

    # Add expected interface power flow variables
    expected_vmag = 0.00
    expected_pf_p = 0.00
    expected_pf_q = 0.00
    model.expected_interface_vmag = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.expected_interface_pf_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.expected_interface_pf_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.interface_expected_values = pe.ConstraintList()
    for p in model.periods:
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:
                omega_oper = network.prob_operation_scenarios[s_o]
                expected_vmag += omega_market * omega_oper * model.e[ref_node_idx, s_m, s_o, p]
                expected_pf_p += omega_market * omega_oper * model.pg[ref_gen_idx, s_m, s_o, p]
                expected_pf_q += omega_market * omega_oper * model.qg[ref_gen_idx, s_m, s_o, p]
        model.interface_expected_values.add(model.expected_interface_vmag[p] <= expected_vmag + EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_vmag[p] >= expected_vmag - EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_p[p] <= expected_pf_p + EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_p[p] >= expected_pf_p - EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_q[p] <= expected_pf_q + EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_q[p] >= expected_pf_q - EQUALITY_TOLERANCE)

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    obj = copy(model.objective.expr)
    model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_regularization.fix(PENALTY_REGULARIZATION)
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                obj += model.penalty_regularization * (model.e[ref_node_idx, s_m, s_o, p] - model.expected_interface_vmag[p]) ** 2
                obj += model.penalty_regularization * s_base * (model.pg[ref_gen_idx, s_m, s_o, p] - model.expected_interface_pf_p[p]) ** 2
                obj += model.penalty_regularization * s_base * (model.qg[ref_gen_idx, s_m, s_o, p] - model.expected_interface_pf_q[p]) ** 2

    # New objective function (settlement)
    model.penalty_settlement = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_settlement.fix(PENALTY_SETTLEMENT)
    model.interface_vmag_req = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.interface_pf_p_req = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.interface_pf_q_req = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    for p in model.periods:
        obj += model.penalty_regularization * (model.expected_interface_vmag[p] - model.interface_vmag_req[p]) ** 2
        obj += model.penalty_regularization * s_base * (model.expected_interface_pf_p[p] - model.interface_pf_p_req[p]) ** 2
        obj += model.penalty_regularization * s_base * (model.expected_interface_pf_q[p] - model.interface_pf_q_req[p]) ** 2
    model.objective.expr = obj
