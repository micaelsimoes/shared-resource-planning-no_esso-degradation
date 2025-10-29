import pyomo.environ as pe
from math import tan, atan2, acos
from helper_functions import *
from definitions import *


# Voltage variables, e
def e_bounds(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF and not network.is_transmission:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg
        return (vg - SMALL_TOLERANCE, vg + SMALL_TOLERANCE)
    return (-node.v_max, node.v_max)


# Voltage variables, f
def f_bounds(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)
    return (-node.v_max, node.v_max)


def vmag_bounds(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    v_min = node.v_min
    v_max = node.v_max
    if params.slacks.grid_operation.voltage:
        v_min -= VMAG_VIOLATION_ALLOWED
        v_max += VMAG_VIOLATION_ALLOWED
    return (v_min, v_max)


# Voltage variables, slack bounds
def voltage_slack_bounds(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF:
        return (0.00, SMALL_TOLERANCE)
    return (0.00, VMAG_VIOLATION_ALLOWED)


def node_balance_slack_bounds(m, i, s_m, s_o, p, network):
    return (0.00, NODE_BALANCE_SLACK_LIMIT / network.baseMVA)


# Generation, Pg
def pg_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.status[p]:
        return (gen.pmin, gen.pmax)
    else:
        return (-SMALL_TOLERANCE, SMALL_TOLERANCE)


# Generation, Qg
def qg_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.status[p]:
        return (gen.qmin, gen.qmax)
    else:
        return (-SMALL_TOLERANCE, SMALL_TOLERANCE)


def pg_init(m, g, s_m, s_o, p, network):
    lb, ub = pg_bounds(m, g, s_m, s_o, p, network=network)
    if lb > 0:
        return lb
    else:
        return max(0.0, lb)


def qg_init(m, g, s_m, s_o, p, network):
    lb, ub = qg_bounds(m, g, s_m, s_o, p, network=network)
    if lb > 0:
        return lb
    else:
        return max(0.0, lb)


# Generation, Sg
def sg_init(m, g, s_m, s_o, p, network, params):

    gen = network.generators[g]
    if not gen.is_curtaillable() or not gen.status[p]:
        return 0.00

    # Apparent power for initialization and bound
    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg = max(0.00, (pg ** 2 + qg ** 2) ** 0.5)

    return abs(sg)


def sg_bounds(m, g, s_m, s_o, p, network, params):

    gen = network.generators[g]
    if not gen.is_curtaillable() or not gen.status[p]:
        return (0.0, SMALL_TOLERANCE)

    # Estimated apparent power for bounds
    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg = max(0.00, (pg ** 2 + qg ** 2) ** 0.5)

    return (0.0, sg)


# Branch power flow, Fij
def flow_ij_sqr_bounds(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, SMALL_TOLERANCE)
    rating_sqr = (branch.rate / network.baseMVA)**2
    return (0.0, rating_sqr)


def init_flow_ij_sqr(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return 0.0
    return SMALL_TOLERANCE ** 2


# Branch power flow, Fij slacks
def slack_flow_bounds(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, SMALL_TOLERANCE)
    rating = (branch.rate / network.baseMVA)
    return (0.0, (SIJ_VIOLATION_ALLOWED * rating) ** 2)


# Consumption, Pc
def pc_bounds(m, c, s_m, s_o, p, network):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    return (pd - EQUALITY_TOLERANCE, pd + EQUALITY_TOLERANCE)


# Consumption, Qc
def qc_bounds(m, c, s_m, s_o, p, network):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    return (qd - EQUALITY_TOLERANCE, qd + EQUALITY_TOLERANCE)


def pc_initialize(m, c, s_m, s_o, p, network):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    return pd


def qc_initialize(m, c, s_m, s_o, p, network):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    return qd


# Consumption, flexibility
def pc_flex_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, SMALL_TOLERANCE)
    value = abs(load.flexibility.active_power.upward[s_o][p])
    return (0.0, value)


def pc_flex_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, SMALL_TOLERANCE)
    value = abs(load.flexibility.active_power.downward[s_o][p])
    return (0.0, value)


def qc_flex_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, SMALL_TOLERANCE)
    value = abs(load.flexibility.reactive_power.upward[s_o][p])
    return (0.0, value)


def qc_flex_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, SMALL_TOLERANCE)
    value = abs(load.flexibility.reactive_power.downward[s_o][p])
    return (0.0, value)


# Consumption, curtailment
def pc_curt_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, abs(pd))
    else:
        return (0.0, SMALL_TOLERANCE)


def pc_curt_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, SMALL_TOLERANCE)
    else:
        return (0.0, abs(pd))


def qc_curt_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    if qd >= 0.00:
        return (0.0, abs(qd))
    else:
        return (0.0, SMALL_TOLERANCE)


def qc_curt_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    if qd >= 0.00:
        return (0.0, SMALL_TOLERANCE)
    else:
        return (0.0, abs(qd))

# Transformers
def transformer_ratio_bounds(m, i, s_m, s_o, p, network, params):
    branch = network.branches[i]
    if branch.is_transformer:
        if params.transf_reg and branch.vmag_reg:
            return (TRANSFORMER_MINIMUM_RATIO, TRANSFORMER_MAXIMUM_RATIO)
        else:
            return (branch.ratio - EQUALITY_TOLERANCE, branch.ratio + EQUALITY_TOLERANCE)
    else:
        return (1.00 - EQUALITY_TOLERANCE, 1.00 + EQUALITY_TOLERANCE)


def transformer_ratio_initialize(m, i, s_m, s_o, p, network, params):
    branch = network.branches[i]
    if branch.is_transformer:
        if params.transf_reg and branch.vmag_reg:
            return 1.00
        else:
            return branch.ratio
    else:
        return 1.00


# Energy Storage
def soc_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (ess.e_min, ess.e_max)


def p_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (0.0, ess.s)


def snet_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (-ess.s, ess.s)


def q_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (-ess.s, ess.s)


def s_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (0.0, ess.s)


def slack_es_comp_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (0.0, ess.s * 0.05)


def slack_es_balance_bounds(m, e, s_m, s_o, network):
    ess = network.energy_storages[e]
    return (0.00, ess.e * 0.05)


def soc_initialize(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return ess.e_init


# Shared Energy Storage
def shared_soc_bounds(m, e, s_m, s_o, p, network):
    sess = network.shared_energy_storages[e]
    return (0.0, sess.e)


def shared_q_bounds(m, e, s_m, s_o, p, network):
    sess = network.shared_energy_storages[e]
    return (-sess.s, sess.s)


def shared_s_bounds(m, e, s_m, s_o, p, network):
    sess = network.shared_energy_storages[e]
    return (0.0, sess.s)


def shared_soc_init(m, e, s_m, s_o, p, network):
    sess = network.shared_energy_storages[e]
    return sess.e * ENERGY_STORAGE_RELATIVE_INIT_SOC


# Voltage constraints, e
def voltage_rule_e(m, i, s_m, s_o, p, params):
    e_val = m.e[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        e_val += m.slack_e_up[i, s_m, s_o, p] - m.slack_e_down[i, s_m, s_o, p]
    return m.e_actual[i, s_m, s_o, p] == e_val


# Voltage constraints, f
def voltage_rule_f(m, i, s_m, s_o, p, params):
    f_val = m.f[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        f_val += m.slack_f_up[i, s_m, s_o, p] - m.slack_f_down[i, s_m, s_o, p]
    return m.f_actual[i, s_m, s_o, p] == f_val


# Voltage constraints, magnitude
def voltage_magnitude_def_rule(m, i, s_m, s_o, p):
    e = m.e[i, s_m, s_o, p]
    f = m.f[i, s_m, s_o, p]
    vmag_sq = e ** 2 + f ** 2
    return m.vmag[i, s_m, s_o, p] ** 2 == vmag_sq


# Voltage constraints, magnitude
def voltage_magnitude_cons_rule(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    vmag = m.vmag[i, s_m, s_o, p]
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return pe.inequality(-SMALL_TOLERANCE, vmag - vg, SMALL_TOLERANCE)
    else:
        return pe.inequality(node.v_min, vmag, node.v_max)


def sg_abs_rule(m, g, s_m, s_o, p, network, params):
    generator = network.generators[g]
    if not generator.is_curtaillable() or not generator.status[p]:
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, m.sg_abs[g, s_m, s_o, p] ** 2 - (m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def sg_curt_rule(m, g, s_m, s_o, p, network, params):
    generator = network.generators[g]
    if not generator.is_curtaillable() or not generator.status[p]:
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, m.sg_curt[g, s_m, s_o, p] - (m.sg_init[g, s_m, s_o, p] - m.sg_abs[g, s_m, s_o, p]), EQUALITY_TOLERANCE)


def power_factor_rule_upper(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable() or not generator.status[p]:
        return pe.Constraint.Skip
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]
    if generator.power_factor_control:
        phi = acos(generator.max_pf)
    else:
        phi = atan2(generator.qg[s_o][p], generator.pg[s_o][p])
    return qg <= tan(phi) * pg


def power_factor_rule_lower(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable() or not generator.status[p]:
        return pe.Constraint.Skip
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]
    if generator.power_factor_control:
        phi = acos(generator.min_pf)
    else:
        phi = atan2(generator.qg[s_o][p], generator.pg[s_o][p])
    return qg >= tan(phi) * pg


# Flexible loads
def flex_energy_balance_p_rule(m, c, s_m, s_o, network, params):

    load = network.loads[c]

    if network.is_transmission:
        if load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip

    if load.fl_reg:
        if network.is_transmission and load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip
        p_up = sum(m.flex_p_up[c, s_m, s_o, p] for p in m.periods)
        p_down = sum(m.flex_p_down[c, s_m, s_o, p] for p in m.periods)
        if params.slacks.flexibility.day_balance:
            return p_up == p_down + m.slack_flex_p_balance_up[c, s_m, s_o] - m.slack_flex_p_balance_down[c, s_m, s_o]
        else:
            return pe.inequality(-SMALL_TOLERANCE, p_up - p_down, SMALL_TOLERANCE)
    else:
        return pe.Constraint.Skip


def flex_energy_balance_q_rule(m, c, s_m, s_o, network, params):

    load = network.loads[c]

    if network.is_transmission:
        if load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip

    if load.fl_reg:
        if network.is_transmission and load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip
        q_up = sum(m.flex_q_up[c, s_m, s_o, p] for p in m.periods)
        q_down = sum(m.flex_q_down[c, s_m, s_o, p] for p in m.periods)
        if params.slacks.flexibility.day_balance:
            return q_up == q_down + m.slack_flex_p_balance_up[c, s_m, s_o] - m.slack_flex_p_balance_down[c, s_m, s_o]
        else:
            return pe.inequality(-SMALL_TOLERANCE, q_up - q_down, SMALL_TOLERANCE)
    else:
        return pe.Constraint.Skip


def flex_energy_balance_s_rule(m, c, s_m, s_o, network, params):

    load = network.loads[c]

    if network.is_transmission:
        if load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip

    if load.fl_reg:
        if network.is_transmission and load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip
        s_up_sqr = sum((m.flex_p_up[c, s_m, s_o, p] + m.flex_q_up[c, s_m, s_o, p]) ** 2 for p in m.periods)
        s_down_sqr = sum((m.flex_p_down[c, s_m, s_o, p] + m.flex_q_down[c, s_m, s_o, p]) ** 2 for p in m.periods)
        if params.slacks.flexibility.day_balance:
            return s_up_sqr == s_down_sqr + m.slack_flex_p_balance_up[c, s_m, s_o] - m.slack_flex_p_balance_down[c, s_m, s_o]
        else:
            return pe.inequality(-SMALL_TOLERANCE, s_up_sqr - s_down_sqr, SMALL_TOLERANCE)
    else:
        return pe.Constraint.Skip


# Energy Storage
def ess_sch_def(m, e, s_m, s_o, p, params):
    return pe.inequality(-EQUALITY_TOLERANCE, m.es_sch[e, s_m, s_o, p]**2 - (m.es_pch[e, s_m, s_o, p]**2 + m.es_qch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def ess_sdch_def(m, e, s_m, s_o, p, params):
    return pe.inequality(-EQUALITY_TOLERANCE, m.es_sdch[e, s_m, s_o, p]**2 - (m.es_pdch[e, s_m, s_o, p]**2 + m.es_qdch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def ess_snet_def(m, e, s_m, s_o, p, params):
    return pe.inequality(-EQUALITY_TOLERANCE, m.es_snet[e, s_m, s_o, p] - (m.es_sch[e, s_m, s_o, p] - m.es_sdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


def ess_phi_ch_limits_lower(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    min_phi = acos(ess.min_pf)
    return m.es_qch[e, s_m, s_o, p] >= tan(min_phi) * m.es_pch[e, s_m, s_o, p]


def ess_phi_ch_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    max_phi = acos(ess.max_pf)
    return m.es_qch[e, s_m, s_o, p] <= tan(max_phi) * m.es_pch[e, s_m, s_o, p]


def ess_phi_dch_limits_lower(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    min_phi = acos(ess.min_pf)
    return m.es_qdch[e, s_m, s_o, p] >= tan(min_phi) * m.es_pdch[e, s_m, s_o, p]


def ess_phi_dch_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    max_phi = acos(ess.max_pf)
    return m.es_qdch[e, s_m, s_o, p] <= tan(max_phi) * m.es_pdch[e, s_m, s_o, p]


def ess_comp_rule(m, e, s_m, s_o, p, network, params):
    if params.slacks.ess.complementarity:
        return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] <= m.slack_es_comp[e, s_m, s_o, p]
    else:
        if params.ess_model == ESS_MODEL_EXACT:
            return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] <= SMALL_TOLERANCE
        elif params.ess_model in [ESS_MODEL_LP_SIMPLIFIED, ESS_MODEL_LP_RELAXED, ESS_MODEL_LP_SIMPLIFIED_EXTENDED, ESS_MODEL_FIRST_ORDER]:
            return pe.Constraint.Skip
        else:
            print('[ERROR] Invalid ESS model. Exiting...')
            exit(ERROR_PARAMS_FILE)


def ess_balance_rule(m, e, s_m, s_o, p, network, params):
    es = network.energy_storages[e]
    eff_ch, eff_dch = es.eff_ch, es.eff_dch
    soc_prev = es.e_init if p == 0 else m.es_soc[e, s_m, s_o, p - 1]
    if params.ess_model == ESS_MODEL_FIRST_ORDER:
        return m.es_soc[e, s_m, s_o, p] == (soc_prev + m.es_snet[e, s_m, s_o, p])
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, p] - (soc_prev + m.es_sch[e, s_m, s_o, p] * eff_ch - m.es_sdch[e, s_m, s_o, p] / eff_dch), EQUALITY_TOLERANCE)


def ess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = network.energy_storages[e].e_init
    final_p = m.periods[-1]
    if params.slacks.ess.day_balance:
        return m.es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_es_soc_final_up[e, s_m, s_o] - m.slack_es_soc_final_up[e, s_m, s_o]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, final_p] - final_soc, EQUALITY_TOLERANCE)


# - Linear ESS models -- Relaxed LP formulation
def ess_relaxed_model_ch_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return m.es_sch[e, s_m, s_o, p] <= ess.s * m.es_sch_comp[e, s_m, s_o, p]


def ess_relaxed_model_dch_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return m.es_sdch[e, s_m, s_o, p] <= ess.s * m.es_sdch_comp[e, s_m, s_o, p]


def ess_relaxed_model_comp_rule(m, e, s_m, s_o, p):
    return m.es_sch_comp[e, s_m, s_o, p] + m.es_sdch_comp[e, s_m, s_o, p] <= 1.00


# - Linear ESS models -- Extended simplified formulation
def ess_simplified_model_ch_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    soc_prev = ess.e_init if p == 0 else m.es_soc[e, s_m, s_o, p - 1]
    return m.es_sch[e, s_m, s_o, p] <= (ess.e_max - soc_prev) / ess.eff_ch


def ess_simplified_model_dch_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    soc_prev = ess.e_init if p == 0 else m.es_soc[e, s_m, s_o, p - 1]
    return m.es_sdch[e, s_m, s_o, p] <= (soc_prev - ess.e_min) / ess.eff_dch


def ess_simplified_model_comp_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return m.es_sdch[e, s_m, s_o, p] <= ess.s - m.es_sch[e, s_m, s_o, p]


# Shared Energy Storage
def sess_phi_ch_limits_lower(m, e, s_m, s_o, p, network):
    ess = network.shared_energy_storages[e]
    min_phi = acos(ess.min_pf)
    return m.shared_es_qch[e, s_m, s_o, p] >= tan(min_phi) * m.shared_es_pch[e, s_m, s_o, p]


def sess_phi_ch_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.shared_energy_storages[e]
    max_phi = acos(ess.max_pf)
    return m.shared_es_qch[e, s_m, s_o, p] <= tan(max_phi) * m.shared_es_pch[e, s_m, s_o, p]


def sess_phi_dch_limits_lower(m, e, s_m, s_o, p, network):
    ess = network.shared_energy_storages[e]
    min_phi = acos(ess.min_pf)
    return m.shared_es_qdch[e, s_m, s_o, p] >= tan(min_phi) * m.shared_es_pdch[e, s_m, s_o, p]


def sess_phi_dch_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.shared_energy_storages[e]
    max_phi = acos(ess.max_pf)
    return m.shared_es_qdch[e, s_m, s_o, p] <= tan(max_phi) * m.shared_es_pdch[e, s_m, s_o, p]


def sess_sch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    sch = m.shared_es_sch[e, s_m, s_o, p]
    return sch <= s_max


def sess_sdch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    sdch = m.shared_es_sdch[e, s_m, s_o, p]
    return sdch <= s_max


def sess_sch_def(m, e, s_m, s_o, p, params):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_sch[e, s_m, s_o, p]**2 - (m.shared_es_pch[e, s_m, s_o, p]**2 + m.shared_es_qch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def sess_sdch_def(m, e, s_m, s_o, p, params):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_sdch[e, s_m, s_o, p]**2 - (m.shared_es_pdch[e, s_m, s_o, p]**2 + m.shared_es_qdch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def sess_pch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    pch = m.shared_es_pch[e, s_m, s_o, p]
    return pch <= s_max


def sess_qch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    qch = m.shared_es_qch[e, s_m, s_o, p]
    return qch <= s_max


def sess_pdch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return pdch <= s_max


def sess_qdch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    qdch = m.shared_es_qdch[e, s_m, s_o, p]
    return qdch <= s_max


def sess_soc_lower_limit(m, e, s_m, s_o, p):
    soc_min = m.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] >= soc_min


def sess_soc_upper_limit(m, e, s_m, s_o, p):
    soc_max = m.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] <= soc_max


def sess_comp_rule(m, e, s_m, s_o, p, params):
    if params.slacks.shared_ess.complementarity:
        return m.shared_es_sch[e, s_m, s_o, p] * m.shared_es_sdch[e, s_m, s_o, p] <= m.slack_shared_es_comp[e, s_m, s_o, p]
    else:
        if params.shared_ess_model == ESS_MODEL_EXACT:
            return m.shared_es_sch[e, s_m, s_o, p] * m.shared_es_sdch[e, s_m, s_o, p] <= SMALL_TOLERANCE
        elif params.shared_ess_model in [ESS_MODEL_LP_SIMPLIFIED, ESS_MODEL_LP_RELAXED, ESS_MODEL_LP_SIMPLIFIED_EXTENDED, ESS_MODEL_FIRST_ORDER]:
            return pe.Constraint.Skip
        else:
            print('[ERROR] Invalid ESS model. Exiting...')
            exit(ERROR_PARAMS_FILE)


def sess_balance_rule(m, e, s_m, s_o, p, network, params):
    ses = network.shared_energy_storages[e]
    eff_ch, eff_dch = ses.eff_ch, ses.eff_dch
    soc_prev = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC if p == 0 else m.shared_es_soc[e, s_m, s_o, p - 1]
    if params.shared_ess_model == ESS_MODEL_EXACT:
        return m.shared_es_soc[e, s_m, s_o, p] == soc_prev + (m.shared_es_sch[e, s_m, s_o, p] - m.shared_es_sdch[e, s_m, s_o, p])
    else:
        return m.shared_es_soc[e, s_m, s_o, p] == (soc_prev + m.shared_es_sch[e, s_m, s_o, p] * eff_ch - m.shared_es_sdch[e, s_m, s_o, p] / eff_dch)


def sess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
    final_p = m.periods[-1]
    if params.slacks.shared_ess.day_balance:
        return m.shared_es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_shared_es_soc_final_up[e, s_m, s_o] - m.slack_shared_es_soc_final_down[e, s_m, s_o]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_soc[e, s_m, s_o, final_p] - final_soc, EQUALITY_TOLERANCE)


def sess_pnet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_pnet[e, s_m, s_o, p] - (m.shared_es_pch[e, s_m, s_o, p] - m.shared_es_pdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


def sess_qnet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_qnet[e, s_m, s_o, p] - (m.shared_es_qch[e, s_m, s_o, p] - m.shared_es_qdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


def sess_snet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_snet[e, s_m, s_o, p] - (m.shared_es_sch[e, s_m, s_o, p] - m.shared_es_sdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


def sess_s_sensitivities(m, e):
    return m.shared_es_s_rated[e] <= m.shared_es_s_rated_fixed[e]


def sess_e_sensitivities(m, e):
    return m.shared_es_e_rated[e] <= m.shared_es_e_rated_fixed[e]

# - Linear Shared ESS models -- Relaxed LP formulation
def sess_relaxed_model_ch_rule(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    return m.shared_es_sch[e, s_m, s_o, p] <= s_max * m.shared_es_sch_comp[e, s_m, s_o, p]


def sess_relaxed_model_dch_rule(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    return m.shared_es_sdch[e, s_m, s_o, p] <= s_max * m.shared_es_sdch_comp[e, s_m, s_o, p]


def sess_relaxed_model_comp_rule(m, e, s_m, s_o, p):
    return m.shared_es_sch_comp[e, s_m, s_o, p] + m.shared_es_sdch_comp[e, s_m, s_o, p] <= 1.00


# - Linear Shared ESS models -- Extended simplified formulation
def sess_simplified_model_ch_rule(m, e, s_m, s_o, p, network):
    sess = network.shared_energy_storages[e]
    e_max = m.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
    soc_prev = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC if p == 0 else m.shared_es_soc[e, s_m, s_o, p - 1]
    return m.shared_es_sch[e, s_m, s_o, p] <= (e_max - soc_prev) / sess.eff_ch


def sess_simplified_model_dch_rule(m, e, s_m, s_o, p, network):
    sess = network.shared_energy_storages[e]
    e_min = m.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
    soc_prev = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC if p == 0 else m.shared_es_soc[e, s_m, s_o, p - 1]
    return m.shared_es_sdch[e, s_m, s_o, p] <= (soc_prev - e_min) / sess.eff_dch


def sess_simplified_model_comp_rule(m, e, s_m, s_o, p, network):
    return m.shared_es_sdch[e, s_m, s_o, p] <= m.shared_es_s_rated[e] - m.shared_es_sch[e, s_m, s_o, p]


# Interface power flows and voltage magnitude definition
def interface_vmag_transmission_rule(m, dn, s_m, s_o, p, network):
    adn_node_id = network.active_distribution_network_nodes[dn]
    adn_node_idx = network.get_node_idx(adn_node_id)
    return pe.inequality(-EQUALITY_TOLERANCE, m.vmag_adn[dn, s_m, s_o, p] - m.vmag[adn_node_idx, s_m, s_o, p], EQUALITY_TOLERANCE)


def interface_pf_p_transmission_rule(m, dn, s_m, s_o, p, network, params):
    adn_node_id = network.active_distribution_network_nodes[dn]
    adn_load_idx = network.get_adn_load_idx(adn_node_id)
    if params.l_curt:
        fix_or_set(m.pc_curt_down[adn_load_idx, s_m, s_o, p], SMALL_TOLERANCE)
        fix_or_set(m.pc_curt_up[adn_load_idx, s_m, s_o, p], SMALL_TOLERANCE)
    return pe.inequality(-EQUALITY_TOLERANCE, m.pc_adn[dn, s_m, s_o, p] - (m.pc[adn_load_idx, s_m, s_o, p] + m.flex_p_up[adn_load_idx, s_m, s_o, p] - m.flex_p_down[adn_load_idx, s_m, s_o, p]), EQUALITY_TOLERANCE)


def interface_pf_q_transmission_rule(m, dn, s_m, s_o, p, network, params):
    adn_node_id = network.active_distribution_network_nodes[dn]
    adn_load_idx = network.get_adn_load_idx(adn_node_id)
    if params.l_curt:
        fix_or_set(m.qc_curt_down[adn_load_idx, s_m, s_o, p], SMALL_TOLERANCE)
        fix_or_set(m.qc_curt_up[adn_load_idx, s_m, s_o, p], SMALL_TOLERANCE)
    return pe.inequality(-EQUALITY_TOLERANCE, m.qc_adn[dn, s_m, s_o, p] - (m.qc[adn_load_idx, s_m, s_o, p] + m.flex_q_up[adn_load_idx, s_m, s_o, p] - m.flex_q_down[adn_load_idx, s_m, s_o, p]), EQUALITY_TOLERANCE)


def interface_vmag_distribution_rule(m, s_m, s_o, p, network):
    ref_node_id = network.get_reference_node_id()
    ref_node_idx = network.get_node_idx(ref_node_id)
    return pe.inequality(-EQUALITY_TOLERANCE, m.vmag_adn[s_m, s_o, p] - m.vmag[ref_node_idx, s_m, s_o, p], EQUALITY_TOLERANCE)


def interface_pf_p_distribution_rule(m, s_m, s_o, p, network):
    ref_gen_idx = network.get_reference_gen_idx()
    return pe.inequality(-EQUALITY_TOLERANCE, m.pg_adn[s_m, s_o, p] - m.pg[ref_gen_idx, s_m, s_o, p], EQUALITY_TOLERANCE)


def interface_pf_q_distribution_rule(m, s_m, s_o, p, network):
    ref_gen_idx = network.get_reference_gen_idx()
    return pe.inequality(-EQUALITY_TOLERANCE, m.qg_adn[s_m, s_o, p] - m.qg[ref_gen_idx, s_m, s_o, p], EQUALITY_TOLERANCE)



# Branch limits
def compute_branch_flow_squared(branch, ei, fi, ej, fj, rij, limit_type):
    g = branch.g
    b = branch.b
    bsh = 0.5 * branch.b_sh  # Half-line shunt susceptance for π-model

    if limit_type == BRANCH_LIMIT_CURRENT:
        delta_e = (rij**2) * ei - rij * ej
        delta_f = (rij**2) * fi - rij * fj

        current_squared = (g**2 + b**2) * (delta_e**2 + delta_f**2)
        current_squared += bsh**2 * (ei**2 + fi**2)
        current_squared += 2 * g * bsh * (delta_f * ei - delta_e * fi)
        current_squared += 2 * b * bsh * (delta_e * ei + delta_f * fi)

        # current_squared = (branch.g ** 2 + branch.b ** 2) * ((ei - ej) ** 2 + (fi - fj) ** 2)

        return current_squared

    elif limit_type == BRANCH_LIMIT_APPARENT_POWER or (limit_type == BRANCH_LIMIT_MIXED and branch.is_transformer):
        # Real power flow from i to j
        pij = g * (ei**2 + fi**2) * rij**2
        pij -= g * (ei * ej + fi * fj) * rij
        pij -= b * (fi * ej - ei * fj) * rij

        # Reactive power flow from i to j
        qij = -(b + bsh) * (ei**2 + fi**2) * rij**2
        qij += b * (ei * ej + fi * fj) * rij
        qij -= g * (fi * ej - ei * fj) * rij

        return pij**2 + qij**2

    elif limit_type == BRANCH_LIMIT_MIXED and not branch.is_transformer:
        delta_e = (rij**2) * ei - rij * ej
        delta_f = (rij**2) * fi - rij * fj

        current_squared = (g**2 + b**2) * (delta_e**2 + delta_f**2)
        current_squared += bsh**2 * (ei**2 + fi**2)
        current_squared += 2 * g * bsh * (delta_f * ei - delta_e * fi)
        current_squared += 2 * b * bsh * (delta_e * ei + delta_f * fi)

        # current_squared = (branch.g ** 2 + branch.b ** 2) * ((ei - ej) ** 2 + (fi - fj) ** 2)

        return current_squared

    raise ValueError(f"Unknown branch limit type: {limit_type}")


def compute_node_load(model, i, s_m, s_o, p, network, params):

    node = network.nodes[i]

    Pd, Qd = 0.0, 0.0

    for c in model.loads:
        load = network.loads[c]
        if load.bus == node.bus_i:
            Pd += model.pc[c, s_m, s_o, p]
            Qd += model.qc[c, s_m, s_o, p]
            if params.fl_reg and load.fl_reg:
                Pd += model.flex_p_up[c, s_m, s_o, p] - model.flex_p_down[c, s_m, s_o, p]
                Qd += model.flex_q_up[c, s_m, s_o, p] - model.flex_q_down[c, s_m, s_o, p]
            if params.l_curt:
                Pd -= model.pc_curt_down[c, s_m, s_o, p] - model.pc_curt_up[c, s_m, s_o, p]
                Qd -= model.qc_curt_down[c, s_m, s_o, p] - model.qc_curt_up[c, s_m, s_o, p]

    if params.es_reg:
        for e in model.energy_storages:
            es = network.energy_storages[e]
            if es.bus == node.bus_i:
                Pd += model.es_pnet[e, s_m, s_o, p]
                Qd += model.es_qnet[e, s_m, s_o, p]

    for e in model.shared_energy_storages:
        es = network.shared_energy_storages[e]
        if es.bus == node.bus_i:
            Pd += model.shared_es_pch[e, s_m, s_o, p] - model.shared_es_pdch[e, s_m, s_o, p]
            Qd += model.shared_es_qch[e, s_m, s_o, p] - model.shared_es_qdch[e, s_m, s_o, p]

    return Pd, Qd


def compute_node_gen(model, i, s_m, s_o, p, network):
    Pg, Qg = 0.0, 0.0
    node = network.nodes[i]
    for g in model.generators:
        gen = network.generators[g]
        if gen.bus == node.bus_i:
            Pg += model.pg[g, s_m, s_o, p]
            Qg += model.qg[g, s_m, s_o, p]
    return Pg, Qg


def net_load_p_per_node_rule(model, i, s_m, s_o, p, network, params):
    Pd, _ = compute_node_load(model, i, s_m, s_o, p, network, params)
    return pe.inequality(-EQUALITY_TOLERANCE, model.pc_node[i, s_m, s_o, p] - Pd, EQUALITY_TOLERANCE)


def net_load_q_per_node_rule(model, i, s_m, s_o, p, network, params):
    _, Qd = compute_node_load(model, i, s_m, s_o, p, network, params)
    return pe.inequality(-EQUALITY_TOLERANCE, model.qc_node[i, s_m, s_o, p] - Qd, EQUALITY_TOLERANCE)


def net_gen_p_per_node_rule(model, i, s_m, s_o, p, network):
    Pg, _ = compute_node_gen(model, i, s_m, s_o, p, network)
    return pe.inequality(-EQUALITY_TOLERANCE, model.pg_node[i, s_m, s_o, p] - Pg, EQUALITY_TOLERANCE)


def net_gen_q_per_node_rule(model, i, s_m, s_o, p, network):
    _, Qg = compute_node_gen(model, i, s_m, s_o, p, network)
    return pe.inequality(-EQUALITY_TOLERANCE, model.qg_node[i, s_m, s_o, p] - Qg, EQUALITY_TOLERANCE)


def node_balance_p_rule(model, i, s_m, s_o, p, network, params):

    node = network.nodes[i]

    ei = model.e_actual[i, s_m, s_o, p]
    fi = model.f_actual[i, s_m, s_o, p]
    Pd = model.pc_node[i, s_m, s_o, p]
    Pg = model.pg_node[i, s_m, s_o, p]

    Pi = node.gs * (ei**2 +  fi**2)

    for b in range(len(network.branches)):

        branch = network.branches[b]

        if branch.status:

            if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0

                if branch.fbus == node.bus_i:
                    fnode_idx = network.get_node_idx(branch.fbus)
                    tnode_idx = network.get_node_idx(branch.tbus)

                    ei, fi = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
                    ej, fj = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]

                    Pi += branch.g * (ei ** 2 + fi ** 2) * rij ** 2
                    Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
                else:
                    fnode_idx = network.get_node_idx(branch.tbus)
                    tnode_idx = network.get_node_idx(branch.fbus)

                    ei, fi = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
                    ej, fj = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]

                    Pi += branch.g * (ei ** 2 + fi ** 2)
                    Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))

    if params.slacks.node_balance.active_power:
        return Pg == Pd + Pi + (model.slack_node_balance_p_up[i, s_m, s_o, p] - model.slack_node_balance_p_down[i, s_m, s_o, p])
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, Pg - (Pd + Pi), EQUALITY_TOLERANCE)


def node_balance_q_rule(model, i, s_m, s_o, p, network, params):

    node = network.nodes[i]
    ei = model.e_actual[i, s_m, s_o, p]
    fi = model.f_actual[i, s_m, s_o, p]
    Qd = model.qc_node[i, s_m, s_o, p]
    Qg = model.qg_node[i, s_m, s_o, p]

    Qi = -node.bs * (ei**2 + fi**2)

    for b in range(len(network.branches)):

        branch = network.branches[b]

        if branch.status:

            if branch.fbus == node.bus_i or branch.tbus == node.bus_i:

                rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0

                if branch.fbus == node.bus_i:
                    fnode_idx = network.get_node_idx(branch.fbus)
                    tnode_idx = network.get_node_idx(branch.tbus)

                    ei, fi = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
                    ej, fj = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]

                    Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2) * rij ** 2
                    Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

                else:
                    fnode_idx = network.get_node_idx(branch.tbus)
                    tnode_idx = network.get_node_idx(branch.fbus)

                    ei, fi = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
                    ej, fj = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]

                    Qi -= (branch.b + branch.b_sh * 0.5) * (ei ** 2 + fi ** 2)
                    Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

    if params.slacks.node_balance.reactive_power:
        return Qg == Qd + Qi + (model.slack_node_balance_q_up[i, s_m, s_o, p] - model.slack_node_balance_q_down[i, s_m, s_o, p])
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, Qg - (Qd + Qi), EQUALITY_TOLERANCE)


def branch_flow_equation_rule(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Constraint.Skip

    rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0

    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)

    ei = model.e_actual[fnode_idx, s_m, s_o, p]
    fi = model.f_actual[fnode_idx, s_m, s_o, p]
    ej = model.e_actual[tnode_idx, s_m, s_o, p]
    fj = model.f_actual[tnode_idx, s_m, s_o, p]

    flow_ij_sqr_expr = compute_branch_flow_squared(branch, ei, fi, ej, fj, rij, params.branch_limit_type)

    return model.flow_ij_sqr[b, s_m, s_o, p] == flow_ij_sqr_expr


def branch_flow_limit_rule(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Constraint.Skip

    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)
    rating = branch.rate / network.baseMVA or BRANCH_UNKNOWN_RATING

    rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0
    ei = model.e_actual[fnode_idx, s_m, s_o, p]
    fi = model.f_actual[fnode_idx, s_m, s_o, p]
    ej = model.e_actual[tnode_idx, s_m, s_o, p]
    fj = model.f_actual[tnode_idx, s_m, s_o, p]

    flow_ij_sqr = compute_branch_flow_squared(branch, ei, fi, ej, fj, rij, params.branch_limit_type)

    if params.slacks.grid_operation.branch_flow:
        return flow_ij_sqr <= rating ** 2 + model.slack_flow_ij_sqr[b, s_m, s_o, p]
    else:
        return flow_ij_sqr <= rating ** 2


def setup_cost_parameters(model, params):

    model.penalty_ess_usage = pe.Param(initialize=PENALTY_ESS_USAGE, mutable=True)

    if params.obj_type == OBJ_MIN_COST:
        model.cost_res_curtailment = pe.Param(initialize=COST_GENERATION_CURTAILMENT, mutable=True)
        model.cost_load_curtailment = pe.Param(initialize=COST_CONSUMPTION_CURTAILMENT, mutable=True)

    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        model.penalty_gen_curtailment = pe.Param(initialize=PENALTY_GENERATION_CURTAILMENT, mutable=True)
        model.penalty_load_curtailment = pe.Param(initialize=PENALTY_LOAD_CURTAILMENT, mutable=True)
        model.penalty_flex_usage = pe.Param(initialize=PENALTY_FLEXIBILITY_USAGE, mutable=True)

    else:
        raise ValueError(f"[ERROR] Unrecognized or invalid objective type: {params.obj_type}.")


def build_objective(model, network, params):

    model.total_cost = pe.Expression(expr=0)

    for s_m in model.scenarios_market:
        omega_m = network.prob_market_scenarios[s_m]
        for s_o in model.scenarios_operation:

            omega_o = network.prob_operation_scenarios[s_o]
            weight = omega_m * omega_o

            if params.obj_type == OBJ_MIN_COST:
                scenario_of = (
                    generation_cost(model, network, s_m, s_o, params) +
                    flexibility_cost(model, network, s_m, s_o, params) +
                    load_curtailment_cost(model, network, s_m, s_o, params) +
                    gen_curtailment_cost(model, network, s_m, s_o, params) +
                    ess_utilization_cost_penalty(model, network, s_m, s_o, params)
                )
            elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:
                scenario_of = (
                    gen_curtailment_penalty(model, network, s_m, s_o, params) +
                    load_curtailment_penalty(model, network, s_m, s_o, params) +
                    flexibility_penalty(model, network, s_m, s_o, params) +
                    ess_utilization_cost_penalty(model, network, s_m, s_o, params)
                )
            else:
                raise ValueError(f"Unknown objective function type: {params.obj_type}")

            scenario_of += slack_penalties(model, network, s_m, s_o, params)
            model.total_cost.expr += weight * scenario_of

    model.objective = pe.Objective(sense=pe.minimize, expr=model.total_cost)


def generation_cost(model, network, s_m, s_o, params):
    c_p = network.cost_energy_p
    return sum(
        c_p[s_m][p] * network.baseMVA * model.pg[g, s_m, s_o, p]
        for g in model.generators
        if network.generators[g].is_controllable()
        and not (not network.is_transmission and network.generators[g].gen_type == GEN_REFERENCE)
        for p in model.periods
    )


def flexibility_cost(model, network, s_m, s_o, params):
    if params.fl_reg:
        c_flex = network.cost_flex
        return sum(
            c_flex[s_m][p] * network.baseMVA * (
                model.flex_p_up[c, s_m, s_o, p] + model.flex_p_down[c, s_m, s_o, p] +
                model.flex_q_up[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
            )
            for c in model.loads
            for p in model.periods
        )
    return 0.00


def load_curtailment_cost(model, network, s_m, s_o, params):
    if params.l_curt:
        cost = model.cost_load_curtailment
        return sum(
            cost * network.baseMVA * (
                model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p] +
                model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p]
            )
            for c in model.loads
            for p in model.periods
        )
    return 0.00


def gen_curtailment_cost(model, network, s_m, s_o, params):
    if params.rg_curt:
        cost = model.cost_res_curtailment
        return sum(
            cost * network.baseMVA * (model.sg_curt[g, s_m, s_o, p])
            for g in model.generators if network.generators[g].is_curtaillable()
            for p in model.periods
        )
    return 0.00


def gen_curtailment_penalty(model, network, s_m, s_o, params):
    if params.rg_curt:
        penalty = model.penalty_gen_curtailment
        return sum(
            penalty * network.baseMVA * (model.sg_curt[g, s_m, s_o, p])
            for g in model.generators if network.generators[g].is_curtaillable()
            for p in model.periods
        )
    return 0.00


def load_curtailment_penalty(model, network, s_m, s_o, params):
    if params.l_curt:
        penalty = model.penalty_load_curtailment
        return sum(
            penalty * network.baseMVA * (
                model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p] +
                model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p]
            )
            for c in model.loads
            for p in model.periods
        )
    return 0.00


def flexibility_penalty(model, network, s_m, s_o, params):
    if params.fl_reg:
        penalty = model.penalty_flex_usage
        return sum(
            penalty * network.baseMVA * (
                model.flex_p_up[c, s_m, s_o, p] + model.flex_p_down[c, s_m, s_o, p] +
                model.flex_q_up[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
            )
            for c in model.loads
            for p in model.periods
        )
    return 0.00


def ess_utilization_cost_penalty(model, network, s_m, s_o, params):
    cost = sum(
        model.penalty_ess_usage * network.baseMVA * (
                model.shared_es_sch[e, s_m, s_o, p] + model.shared_es_sdch[e, s_m, s_o, p]
        )
        for e in model.shared_energy_storages
        for p in model.periods
    )
    if params.es_reg:
        cost += sum(
            model.penalty_ess_usage * network.baseMVA * (
                    model.es_sch[e, s_m, s_o, p] + model.es_sdch[e, s_m, s_o, p]
            )
            for e in model.energy_storages
            for p in model.periods
        )
    return cost


def slack_penalties(model, network, s_m, s_o, params):

    total = 0
    base = network.baseMVA

    for i in model.nodes:
        for p in model.periods:
            if params.slacks.grid_operation.voltage:
                total += base * PENALTY_VOLTAGE * (model.slack_e_up[i, s_m, s_o, p] + model.slack_e_down[i, s_m, s_o, p])
                total += base * PENALTY_VOLTAGE * (model.slack_f_up[i, s_m, s_o, p] + model.slack_f_down[i, s_m, s_o, p])
            if params.slacks.node_balance.active_power:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_p_up[i, s_m, s_o, p] + model.slack_node_balance_p_down[i, s_m, s_o, p])
            if params.slacks.node_balance.reactive_power:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_q_up[i, s_m, s_o, p] + model.slack_node_balance_q_down[i, s_m, s_o, p])

    if params.fl_reg and params.slacks.flexibility.day_balance:
        total += base * PENALTY_FLEXIBILITY * sum(model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o] for c in model.loads)
        total += base * PENALTY_FLEXIBILITY * sum(model.slack_flex_q_balance_up[c, s_m, s_o] + model.slack_flex_q_balance_down[c, s_m, s_o] for c in model.loads)

    if params.es_reg:
        for e in model.energy_storages:
            for p in model.periods:
                if params.slacks.ess.complementarity:
                    total += base * PENALTY_ESS_COMP * model.slack_es_comp[e, s_m, s_o, p]
            if params.slacks.ess.day_balance:
                total += base * PENALTY_ESS_BALANCE * (model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o])

    for e in model.shared_energy_storages:
        for p in model.periods:
            if params.slacks.shared_ess.complementarity:
                total += base * PENALTY_SHARED_ESS_COMP * model.slack_shared_es_comp[e, s_m, s_o, p]
        if params.slacks.shared_ess.day_balance:
            total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m, s_o] + model.slack_shared_es_soc_final_down[e, s_m, s_o])

    for b in model.branches:
        for p in model.periods:
            if params.slacks.grid_operation.branch_flow:
                total += base * PENALTY_CURRENT * model.slack_flow_ij_sqr[b, s_m, s_o, p]

    return total


def dn_interface_expected_vmag_rule(m, p, network):
    expected_vmag = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.vmag_adn[s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_interface_vmag[p] - expected_vmag, EQUALITY_TOLERANCE)


def dn_interface_expected_pf_p_rule(m, p, network):
    expected_pf_p = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.pg_adn[s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_interface_pf_p[p] - expected_pf_p, EQUALITY_TOLERANCE)


def dn_interface_expected_pf_q_rule(m, p, network):
    expected_pf_q = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.qg_adn[s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_interface_pf_q[p] - expected_pf_q, EQUALITY_TOLERANCE)


def dn_interface_expected_sess_p_rule(m, p, network, shared_ess_idx):
    expected_ess_p = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.shared_es_pnet[shared_ess_idx, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_shared_ess_p[p] - expected_ess_p, EQUALITY_TOLERANCE)


def dn_interface_expected_sess_q_rule(m, p, network, shared_ess_idx):
    expected_ess_q = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.shared_es_qnet[shared_ess_idx, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_shared_ess_q[p] - expected_ess_q, EQUALITY_TOLERANCE)


def tn_interface_expected_vmag_rule(m, dn, p, network):
    expected_vmag = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.vmag_adn[dn, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_interface_vmag[dn, p] - expected_vmag, EQUALITY_TOLERANCE)


def tn_interface_expected_pf_p_rule(m, dn, p, network):
    expected_pf_p = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.pc_adn[dn, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_interface_pf_p[dn, p] - expected_pf_p, EQUALITY_TOLERANCE)


def tn_interface_expected_pf_q_rule(m, dn, p, network):
    expected_pf_q = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.qc_adn[dn, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_interface_pf_q[dn, p] - expected_pf_q, EQUALITY_TOLERANCE)


def tn_interface_expected_sess_p_rule(m, e, p, network):
    expected_ess_p = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.shared_es_pnet[e, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_shared_ess_p[e, p] - expected_ess_p, EQUALITY_TOLERANCE)


def tn_interface_expected_sess_q_rule(m, e, p, network):
    expected_ess_q = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.shared_es_qnet[e, s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return pe.inequality(-EQUALITY_TOLERANCE, m.expected_shared_ess_q[e, p] - expected_ess_q, EQUALITY_TOLERANCE)
