from functools import partial
from math import tan, acos, sqrt, radians
from helper_functions import *
from definitions import *


# Voltage variables, e
def e_initialize(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF and not network.is_transmission:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg
        return vg
    return 1.00


def f_initialize(m, i, s_m, s_o, p, network):
    return 0.00


def _voltage_magnitude_slack_enabled(node, params):
    return (
        params.slacks.grid_operation.voltage
        and node.type != BUS_REF
        and not (node.type == BUS_PV and params.enforce_vg)
    )


def voltage_numerical_upper_bound(node):
    return node.v_max + VMAG_VIOLATION_ALLOWED + SMALL_TOLERANCE


def e_bounds(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == BUS_REF and not network.is_transmission:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg
        return (vg - SMALL_TOLERANCE, vg + SMALL_TOLERANCE)
    component_max = voltage_numerical_upper_bound(node)
    if node.type == BUS_REF:
        return (0.00, component_max)
    return (-component_max, component_max)


def f_bounds(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == BUS_REF:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)
    component_max = voltage_numerical_upper_bound(node)
    return (-component_max, component_max)


# Squared-voltage slack bounds corresponding to the permitted physical magnitude violation.
def voltage_slack_down_bounds(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if not _voltage_magnitude_slack_enabled(node, params):
        return (0.00, 0.00)
    relaxed_v_min = max(node.v_min - VMAG_VIOLATION_ALLOWED, 0.00)
    return (0.00, node.v_min ** 2 - relaxed_v_min ** 2)


def voltage_slack_up_bounds(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if not _voltage_magnitude_slack_enabled(node, params):
        return (0.00, 0.00)
    relaxed_v_max = node.v_max + VMAG_VIOLATION_ALLOWED
    return (0.00, relaxed_v_max ** 2 - node.v_max ** 2)


def voltage_slack_diagnostics(v_min, v_max, vmag_sqr, slack_down, slack_up):
    effective_down = max(slack_down, 0.00)
    effective_up = max(slack_up, 0.00)
    vmag = sqrt(max(vmag_sqr, 0.00))
    return {
        'squared_down': slack_down,
        'squared_up': slack_up,
        'physical_down': v_min - sqrt(max(v_min ** 2 - effective_down, 0.00)),
        'physical_up': sqrt(v_max ** 2 + effective_up) - v_max,
        'violation_down': max(v_min - vmag, 0.00),
        'violation_up': max(vmag - v_max, 0.00),
    }


def vmag_bounds(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    return (0.00, voltage_numerical_upper_bound(node))


def node_balance_slack_bounds(m, i, s_m, s_o, p, network):
    return (0.00, NODE_BALANCE_SLACK_LIMIT / network.baseMVA)


# Generation, Pg
def renewable_available_apparent_power(generator, s_o, p):
    if not generator.status[p] or not generator.is_curtaillable():
        return 0.0
    return sqrt(generator.pg[s_o][p] ** 2 + generator.qg[s_o][p] ** 2)


def renewable_generation_is_unavailable(generator, s_o, p):
    return renewable_available_apparent_power(generator, s_o, p) <= EQUALITY_TOLERANCE


def _power_factor_tangents(device):
    return sorted((tan(acos(device.min_pf)), tan(acos(device.max_pf))))


def pg_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if not gen.status[p]:
        return (0.0, 0.0)

    if gen.is_curtaillable():
        if renewable_generation_is_unavailable(gen, s_o, p):
            return (0.0, 0.0)
        return (0.0, gen.pg[s_o][p] + EQUALITY_TOLERANCE)
    else:
        return (gen.pmin - EQUALITY_TOLERANCE, gen.pmax + EQUALITY_TOLERANCE)


def qg_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if not gen.status[p]:
        return (0.0, 0.0)
    if gen.is_curtaillable() and renewable_generation_is_unavailable(gen, s_o, p):
        return (0.0, 0.0)
    return (gen.qmin - EQUALITY_TOLERANCE, gen.qmax + EQUALITY_TOLERANCE)


def pg_init(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if not gen.status[p]:
        return 0.0

    if gen.is_curtaillable():
        if renewable_generation_is_unavailable(gen, s_o, p):
            return 0.0
        return max(0.0, gen.pg[s_o][p])
    else:
        lb, ub = pg_bounds(m, g, s_m, s_o, p, network)
        return max(0.0, lb)


def qg_init(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if not gen.status[p]:
        return 0.0

    if gen.is_curtaillable():
        # neutral starting point
        return 0.0
    else:
        lb, ub = qg_bounds(m, g, s_m, s_o, p, network)
        return max(0.0, lb)


def pg_avail_init(m, g, s_o, p, network, params):
    gen = network.generators[g]
    if not gen.is_curtaillable() or renewable_generation_is_unavailable(gen, s_o, p):
        return 0.0
    pg_av = gen.pg[s_o][p]
    return max(0.0, pg_av)


def sg_avail_init(m, g, s_o, p, network, params):
    gen = network.generators[g]
    if not gen.is_curtaillable() or renewable_generation_is_unavailable(gen, s_o, p):
        return 0.0
    return renewable_available_apparent_power(gen, s_o, p)


# Branch power flow, Fij
def flow_ij_sqr_bounds(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, SMALL_TOLERANCE)
    rating_sqr = (branch.rate / network.baseMVA)**2
    return (0.0, rating_sqr + EQUALITY_TOLERANCE)


def init_flow_ij_sqr(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return 0.0
    return EQUALITY_TOLERANCE ** 2


# Branch power flow, Fij slacks
def slack_flow_bounds(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, EQUALITY_TOLERANCE)
    rating = branch.rate / network.baseMVA or BRANCH_UNKNOWN_RATING
    relaxed_rating_sqr = ((1.0 + SIJ_VIOLATION_ALLOWED) * rating) ** 2
    return (0.0, relaxed_rating_sqr - rating ** 2 + EQUALITY_TOLERANCE)


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
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.active_power.upward[s_o][p])
    return (0.0, value + EQUALITY_TOLERANCE)


def pc_flex_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.active_power.downward[s_o][p])
    return (0.0, value + EQUALITY_TOLERANCE)


def qc_flex_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.reactive_power.upward[s_o][p])
    return (0.0, value + EQUALITY_TOLERANCE)


def qc_flex_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.reactive_power.downward[s_o][p])
    return (0.0, value + EQUALITY_TOLERANCE)


# Consumption, curtailment
def pc_curt_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, abs(pd) + EQUALITY_TOLERANCE)
    else:
        return (0.0, EQUALITY_TOLERANCE)


def pc_curt_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, EQUALITY_TOLERANCE)
    else:
        return (0.0, abs(pd) + EQUALITY_TOLERANCE)


def qc_curt_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    if qd >= 0.00:
        return (0.0, abs(qd) + EQUALITY_TOLERANCE)
    else:
        return (0.0, EQUALITY_TOLERANCE)


def qc_curt_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    if qd >= 0.00:
        return (0.0, EQUALITY_TOLERANCE)
    else:
        return (0.0, abs(qd) + EQUALITY_TOLERANCE)

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
def p_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (0.0, ess.s + EQUALITY_TOLERANCE)


def snet_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (-ess.s - EQUALITY_TOLERANCE, ess.s + EQUALITY_TOLERANCE)


def q_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (-ess.s - EQUALITY_TOLERANCE, ess.s + EQUALITY_TOLERANCE)


def s_bounds(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return (0.0, ess.s + EQUALITY_TOLERANCE)


def slack_es_balance_bounds(m, e, s_m, s_o, network):
    ess = network.energy_storages[e]
    return (0.00, ess.e * 0.05 + EQUALITY_TOLERANCE)


def soc_initialize(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return ess.e_init


# Voltage constraints, magnitude
def vmag_sqr_def(m, i, s_m, s_o, p):
    return m.vmag_sqr[i, s_m, s_o, p] == m.e[i, s_m, s_o, p] ** 2 + m.f[i, s_m, s_o, p] ** 2


def vmag_def(m, i, s_m, s_o, p):
    return m.vmag_sqr[i, s_m, s_o, p] == m.vmag[i, s_m, s_o, p] ** 2


def voltage_setpoint_cons_rule(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return pe.inequality(-SMALL_TOLERANCE, m.vmag_sqr[i, s_m, s_o, p] - vg ** 2, SMALL_TOLERANCE)
    return pe.Constraint.Skip


def voltage_magnitude_lower_cons_rule(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == BUS_PV and params.enforce_vg:
        return pe.Constraint.Skip
    slack = m.slack_v_sqr_down[i, s_m, s_o, p] if params.slacks.grid_operation.voltage else 0.00
    return m.vmag_sqr[i, s_m, s_o, p] + slack >= node.v_min ** 2


def voltage_magnitude_upper_cons_rule(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == BUS_PV and params.enforce_vg:
        return pe.Constraint.Skip
    slack = m.slack_v_sqr_up[i, s_m, s_o, p] if params.slacks.grid_operation.voltage else 0.00
    return m.vmag_sqr[i, s_m, s_o, p] - slack <= node.v_max ** 2


def voltage_product_real_rule(m, branch_idx, s_m, s_o, p, network):
    branch = network.branches[branch_idx]
    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)
    return m.voltage_product_real[branch_idx, s_m, s_o, p] == m.e[fnode_idx, s_m, s_o, p] * m.e[tnode_idx, s_m, s_o, p] + m.f[fnode_idx, s_m, s_o, p] * m.f[tnode_idx, s_m, s_o, p]


def voltage_product_imag_rule(m, branch_idx, s_m, s_o, p, network):
    branch = network.branches[branch_idx]
    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)
    return m.voltage_product_imag[branch_idx, s_m, s_o, p] == m.f[fnode_idx, s_m, s_o, p] * m.e[tnode_idx, s_m, s_o, p] - m.e[fnode_idx, s_m, s_o, p] * m.f[tnode_idx, s_m, s_o, p]


def voltage_product_real_nonnegative_rule(m, branch_idx, s_m, s_o, p):
    return (m.voltage_product_real[branch_idx, s_m, s_o, p] >= 0.0)


def branch_angle_difference_lower_rule(m, branch_idx, s_m, s_o, p, network):
    branch = network.branches[branch_idx]
    angle_min_tangent = tan(radians(branch.angle_min))
    return (m.voltage_product_imag[branch_idx, s_m, s_o, p] >= angle_min_tangent * m.voltage_product_real[branch_idx, s_m, s_o, p])


def branch_angle_difference_upper_rule(m, branch_idx, s_m, s_o, p, network):
    branch = network.branches[branch_idx]
    angle_max_tangent = tan(radians(branch.angle_max))
    return (m.voltage_product_imag[branch_idx, s_m, s_o, p] <= angle_max_tangent * m.voltage_product_real[branch_idx, s_m, s_o, p])


def _branch_voltage_products(model, network, branch_idx, terminal_node_idx, s_m, s_o, p):
    branch = network.branches[branch_idx]
    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)

    cross_real = model.voltage_product_real[branch_idx, s_m, s_o, p]
    cross_imag = model.voltage_product_imag[branch_idx, s_m, s_o, p]
    if terminal_node_idx == tnode_idx:
        cross_imag = -cross_imag
    elif terminal_node_idx != fnode_idx:
        raise ValueError(f'Node index {terminal_node_idx} is not incident to branch {branch.branch_id}.')

    return cross_real, cross_imag


def sg_sqr_rule(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if not gen.is_curtaillable() or renewable_generation_is_unavailable(gen, s_o, p):
        return 0.0  # just a scalar
    return m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2


def sg_avail_rule(m, g, s_m, s_o, p, network, params):
    gen = network.generators[g]
    if not gen.is_curtaillable() or renewable_generation_is_unavailable(gen, s_o, p):
        return pe.Constraint.Skip
    return m.sg_sqr[g, s_m, s_o, p] <= m.sg_avail[g, s_o, p] ** 2


def power_factor_rule_upper(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if (
        not generator.is_curtaillable()
        or not generator.power_factor_control
        or renewable_generation_is_unavailable(generator, s_o, p)
    ):
        return pe.Constraint.Skip
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]
    _, tangent_upper = _power_factor_tangents(generator)
    return qg <= tangent_upper * pg


def power_factor_rule_lower(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if (
        not generator.is_curtaillable()
        or not generator.power_factor_control
        or renewable_generation_is_unavailable(generator, s_o, p)
    ):
        return pe.Constraint.Skip
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]
    tangent_lower, _ = _power_factor_tangents(generator)
    return qg >= tangent_lower * pg


def power_factor_profile_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if (
        not generator.is_curtaillable()
        or generator.power_factor_control
        or renewable_generation_is_unavailable(generator, s_o, p)
    ):
        return pe.Constraint.Skip
    pg_available = generator.pg[s_o][p]
    qg_available = generator.qg[s_o][p]
    return qg_available * m.pg[g, s_m, s_o, p] == pg_available * m.qg[g, s_m, s_o, p]


# Flexible loads
def flex_energy_balance_p_rule(m, c, s_m, s_o, network, params):

    load = network.loads[c]

    if load.fl_reg:

        if network.is_transmission:
            if load.bus in network.active_distribution_network_nodes:
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
def ess_pnet_rule(m, e, s_m, s_o, p):
    return m.es_pnet[e,s_m,s_o,p] == m.es_pdch[e,s_m,s_o,p] - m.es_pch[e,s_m,s_o,p]


def ess_snet_def_rule(m, e, s_m, s_o, p):
    snet = m.es_sch[e, s_m, s_o, p] - m.es_sdch[e, s_m, s_o, p]
    return snet ** 2 == m.es_pnet[e, s_m, s_o, p] ** 2 + m.es_qnet[e, s_m, s_o, p] ** 2


def ess_pch_link_rule(m, e, s_m, s_o, p):
    return m.es_pch[e, s_m, s_o, p] <= m.es_sch[e, s_m, s_o, p]


def ess_pdch_link_rule(m, e, s_m, s_o, p):
    return m.es_pdch[e, s_m, s_o, p] <= m.es_sdch[e, s_m, s_o, p]


def ess_s_limit_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return m.es_sch[e, s_m, s_o, p] + m.es_sdch[e, s_m, s_o, p] <= ess.s + EQUALITY_TOLERANCE


def ess_soc_limits_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return pe.inequality(ess.e_min - EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, p], ess.e_max + EQUALITY_TOLERANCE)


def ess_phi_limits_lower(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.es_pch[e, s_m, s_o, p]
    pdch = m.es_pdch[e, s_m, s_o, p]
    return m.es_qnet[e, s_m, s_o, p] >= tangent_lower * pdch - tangent_upper * pch


def ess_phi_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.es_pch[e, s_m, s_o, p]
    pdch = m.es_pdch[e, s_m, s_o, p]
    return m.es_qnet[e, s_m, s_o, p] <= tangent_upper * pdch - tangent_lower * pch


def ess_soc_rule(m, e, s_m, s_o, p, network, params):

    ess = network.energy_storages[e]
    eff_ch = ess.eff_ch
    eff_dch = ess.eff_dch
    if p == 0:
        soc_prev = ess.e_init
    else:
        soc_prev = m.es_soc[e, s_m, s_o, p-1]

    delta = eff_ch * m.es_sch[e, s_m, s_o, p] - (m.es_sdch[e, s_m, s_o, p] / eff_dch)

    return m.es_soc[e, s_m, s_o, p] == soc_prev + delta


def ess_comp_rule(m, e, s_m, s_o, p, network, params):
    if params.ess_model == ESS_MODEL_EXACT:
        return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE
    elif params.ess_model == ESS_MODEL_BILINEAR_RELAXATION:
        return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE * 1e2
    elif params.ess_model == ESS_MODEL_POLYNOMIAL_COMPLEMENTARITY:
        return m.es_sch[e, s_m, s_o, p] ** 2 + m.es_sdch[e, s_m, s_o, p] ** 2 <= (m.es_sch[e, s_m, s_o, p] + m.es_sdch[e, s_m, s_o, p]) ** 2 + EQUALITY_TOLERANCE
    else:
        raise ValueError(f'Unknown ess_model {params.ess_model}')


def ess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = network.energy_storages[e].e_init
    final_p = m.periods[-1]
    if params.slacks.ess.day_balance:
        return m.es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_es_soc_final_up[e, s_m, s_o] - m.slack_es_soc_final_down[e, s_m, s_o]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, final_p] - final_soc, EQUALITY_TOLERANCE)


# Shared Energy Storage
def sess_phi_limits_lower(m, e, s_m, s_o, p, network):
    ess = network.shared_energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.shared_es_pch[e, s_m, s_o, p]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return m.shared_es_qnet[e, s_m, s_o, p] >= tangent_lower * pch - tangent_upper * pdch


def sess_phi_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.shared_energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.shared_es_pch[e, s_m, s_o, p]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return m.shared_es_qnet[e, s_m, s_o, p] <= tangent_upper * pch - tangent_lower * pdch


def sess_pch_link_rule(m, e, s_m, s_o, p):
    return m.shared_es_pch[e, s_m, s_o, p] <= m.shared_es_sch[e, s_m, s_o, p]


def sess_pdch_link_rule(m, e, s_m, s_o, p):
    return m.shared_es_pdch[e, s_m, s_o, p] <= m.shared_es_sdch[e, s_m, s_o, p]


def sess_s_limit_rule(m, e, s_m, s_o, p):
    return m.shared_es_sch[e, s_m, s_o, p] + m.shared_es_sdch[e, s_m, s_o, p] <= m.shared_es_s_rated[e]


def sess_snet_def_rule(m, e, s_m, s_o, p):
    snet = m.shared_es_sch[e, s_m, s_o, p] - m.shared_es_sdch[e, s_m, s_o, p]
    return snet ** 2 == m.shared_es_pnet[e, s_m, s_o, p] ** 2 + m.shared_es_qnet[e, s_m, s_o, p] ** 2


def sess_soc_lower_limit(m, e, s_m, s_o, p):
    soc_min = m.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] >= soc_min


def sess_soc_upper_limit(m, e, s_m, s_o, p):
    soc_max = m.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] <= soc_max


def sess_comp_rule(m, e, s_m, s_o, p, network, params):
    if params.shared_ess_model == ESS_MODEL_EXACT:
        return m.shared_es_sch[e, s_m, s_o, p] * m.shared_es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE
    elif params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:
        return m.shared_es_sch[e, s_m, s_o, p] * m.shared_es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE * 1e2
    elif params.shared_ess_model == ESS_MODEL_POLYNOMIAL_COMPLEMENTARITY:
        return m.shared_es_sch[e, s_m, s_o, p] ** 2 + m.shared_es_sdch[e, s_m, s_o, p] ** 2 <= (m.shared_es_sch[e, s_m, s_o, p] + m.shared_es_sdch[e, s_m, s_o, p]) ** 2 + EQUALITY_TOLERANCE
    else:
        return pe.Constraint.Skip


def sess_soc_rule(m, e, s_m, s_o, p, network, params):

    sess = network.shared_energy_storages[e]
    eff_ch = sess.eff_ch
    eff_dch = sess.eff_dch
    if p == 0:
        soc_prev = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
    else:
        soc_prev = m.shared_es_soc[e, s_m, s_o, p - 1]

    delta = eff_ch * m.shared_es_sch[e, s_m, s_o, p] - (m.shared_es_sdch[e, s_m, s_o, p] / eff_dch)

    return m.shared_es_soc[e, s_m, s_o, p] == soc_prev + delta


def sess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
    final_p = m.periods[-1]
    if params.slacks.shared_ess.day_balance:
        return m.shared_es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_shared_es_soc_final_up[e, s_m, s_o] - m.slack_shared_es_soc_final_down[e, s_m, s_o]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_soc[e, s_m, s_o, final_p] - final_soc, EQUALITY_TOLERANCE)


def sess_pnet_rule(m, e, s_m, s_o, p):
    return m.shared_es_pnet[e, s_m, s_o, p] == m.shared_es_pch[e, s_m, s_o, p] - m.shared_es_pdch[e, s_m, s_o, p]


_SHARED_ESS_OPERATIONAL_VARIABLES = (
    'shared_es_pch',
    'shared_es_pdch',
    'shared_es_sch',
    'shared_es_sdch',
    'shared_es_pnet',
    'shared_es_qnet',
    'shared_es_soc',
    'slack_shared_es_soc_final_up',
    'slack_shared_es_soc_final_down',
)

_SHARED_ESS_OPERATIONAL_CONSTRAINTS = (
    'sess_pnet_def',
    'sess_snet_def',
    'sess_pch_link',
    'sess_pdch_link',
    'sess_s_limit',
    'sess_phi_limit_lower',
    'sess_phi_limit_upper',
    'sess_soc_def',
    'sess_soc_limit_upper',
    'sess_soc_limit_lower',
    'sess_soc_final',
    'sess_comp',
)


def shared_ess_capacity_is_inactive(s_capacity, e_capacity):
    tolerance = SHARED_ESS_ZERO_CAPACITY_TOLERANCE
    return abs(s_capacity) <= tolerance or abs(e_capacity) <= tolerance


def normalize_shared_ess_capacity(capacity):
    tolerance = SHARED_ESS_ZERO_CAPACITY_TOLERANCE
    if capacity < -tolerance:
        raise ValueError(
            f'Shared ESS available capacity cannot be negative: {capacity}.'
        )
    return 0.0 if abs(capacity) <= tolerance else capacity


def _component_entries_for_shared_ess(component, shared_ess_idx):
    for index in component:
        first_index = index[0] if isinstance(index, tuple) else index
        if first_index == shared_ess_idx:
            yield component[index]


def _configure_shared_ess_expected_schedule(model, shared_ess_idx, inactive):

    if not hasattr(model, 'expected_shared_ess_p'):
        return

    is_transmission_model = hasattr(model, 'active_distribution_networks')
    for p in model.periods:
        index = (shared_ess_idx, p) if is_transmission_model else p
        for variable_name in ('expected_shared_ess_p', 'expected_shared_ess_q'):
            variable = getattr(model, variable_name)[index]
            if inactive:
                variable.fix(0.0)
            elif variable.fixed:
                variable.unfix()

    for constraint_name in ('expected_shared_ess_p_def', 'expected_shared_ess_q_def'):

        if not hasattr(model, constraint_name):
            continue

        constraint = getattr(model, constraint_name)
        entries = (_component_entries_for_shared_ess(constraint, shared_ess_idx) if is_transmission_model else constraint.values())
        for entry in entries:
            if inactive:
                entry.deactivate()
            else:
                entry.activate()


def configure_shared_ess_operational_state(
        model, shared_ess_idx, s_capacity, e_capacity):
    s_capacity = normalize_shared_ess_capacity(s_capacity)
    e_capacity = normalize_shared_ess_capacity(e_capacity)
    inactive = shared_ess_capacity_is_inactive(s_capacity, e_capacity)
    model.shared_es_s_rated_fixed[shared_ess_idx].set_value(s_capacity)
    model.shared_es_e_rated_fixed[shared_ess_idx].set_value(e_capacity)
    model.shared_es_s_rated[shared_ess_idx].set_value(s_capacity)
    model.shared_es_e_rated[shared_ess_idx].set_value(e_capacity)

    for variable_name in _SHARED_ESS_OPERATIONAL_VARIABLES:
        if not hasattr(model, variable_name):
            continue
        variable = getattr(model, variable_name)
        for entry in _component_entries_for_shared_ess(variable, shared_ess_idx):
            if inactive:
                entry.fix(0.0)
            else:
                if entry.fixed:
                    entry.unfix()
                if variable_name == 'shared_es_soc':
                    entry.set_value(
                        e_capacity * ENERGY_STORAGE_RELATIVE_INIT_SOC
                    )

    for constraint_name in _SHARED_ESS_OPERATIONAL_CONSTRAINTS:
        constraint = getattr(model, constraint_name)
        for entry in _component_entries_for_shared_ess(
                constraint, shared_ess_idx):
            if inactive:
                entry.deactivate()
            else:
                entry.activate()

    _configure_shared_ess_expected_schedule(
        model, shared_ess_idx, inactive
    )
    return inactive


def sess_s_sensitivities(m, e):
    return m.shared_es_s_rated_fixed[e] == m.shared_es_s_rated[e]


def sess_e_sensitivities(m, e):
    return m.shared_es_e_rated_fixed[e] == m.shared_es_e_rated[e]


# Interface power flows and voltage magnitude definition
def interface_vmag_transmission_def(m, dn, s_m, s_o, p, network):
    adn_node_id = network.active_distribution_network_nodes[dn]
    adn_node_idx = network.get_node_idx(adn_node_id)
    return m.vmag[adn_node_idx, s_m, s_o, p]


def interface_pf_p_transmission_def(m, dn, s_m, s_o, p, network, params):
    adn_node_id = network.active_distribution_network_nodes[dn]
    adn_load_idx = network.get_adn_load_idx(adn_node_id)
    if params.l_curt:
        m.pc_curt_down[adn_load_idx, s_m, s_o, p].fix(EQUALITY_TOLERANCE)
        m.pc_curt_up[adn_load_idx, s_m, s_o, p].fix(EQUALITY_TOLERANCE)
    pc_adn = m.pc[adn_load_idx, s_m, s_o, p]
    if params.fl_reg:
        pc_adn += m.flex_p_up[adn_load_idx, s_m, s_o, p] - m.flex_p_down[adn_load_idx, s_m, s_o, p]
    return pc_adn


def interface_pf_q_transmission_def(m, dn, s_m, s_o, p, network, params):
    adn_node_id = network.active_distribution_network_nodes[dn]
    adn_load_idx = network.get_adn_load_idx(adn_node_id)
    if params.l_curt:
        m.qc_curt_down[adn_load_idx, s_m, s_o, p].fix(EQUALITY_TOLERANCE)
        m.qc_curt_up[adn_load_idx, s_m, s_o, p].fix(EQUALITY_TOLERANCE)
    qc_adn = m.qc[adn_load_idx, s_m, s_o, p]
    if params.fl_reg:
        qc_adn += m.flex_q_up[adn_load_idx, s_m, s_o, p] - m.flex_q_down[adn_load_idx, s_m, s_o, p]
    return qc_adn


def interface_vmag_distribution_def(m, s_m, s_o, p, network):
    ref_node_id = network.get_reference_node_id()
    ref_node_idx = network.get_node_idx(ref_node_id)
    return m.vmag[ref_node_idx, s_m, s_o, p]


def interface_pf_p_distribution_def(m, s_m, s_o, p, network):
    ref_gen_idx = network.get_reference_gen_idx()
    ref_node_id = network.get_reference_node_id()
    shared_ess_p = sum(
        m.shared_es_pnet[e, s_m, s_o, p]
        for e in m.shared_energy_storages
        if network.shared_energy_storages[e].bus == ref_node_id
    )
    return m.pg[ref_gen_idx, s_m, s_o, p] - shared_ess_p


def interface_pf_q_distribution_def(m, s_m, s_o, p, network):
    ref_gen_idx = network.get_reference_gen_idx()
    ref_node_id = network.get_reference_node_id()
    shared_ess_q = sum(
        m.shared_es_qnet[e, s_m, s_o, p]
        for e in m.shared_energy_storages
        if network.shared_energy_storages[e].bus == ref_node_id
    )
    return m.qg[ref_gen_idx, s_m, s_o, p] - shared_ess_q


# Branch limits
def branch_uses_apparent_power_limit(branch, params):
    return (
        params.branch_limit_type == BRANCH_LIMIT_APPARENT_POWER
        or (params.branch_limit_type == BRANCH_LIMIT_MIXED and branch.is_transformer)
    )


def compute_branch_terminal_power(branch, terminal_v_sqr, cross_real, cross_imag,
                                  coupling_ratio=1.0, terminal_ratio_sqr=1.0):
    p_terminal = branch.g * terminal_v_sqr * terminal_ratio_sqr
    p_terminal -= branch.g * cross_real * coupling_ratio
    p_terminal -= branch.b * cross_imag * coupling_ratio

    q_terminal = -(branch.b + 0.5 * branch.b_sh) * terminal_v_sqr * terminal_ratio_sqr
    q_terminal += branch.b * cross_real * coupling_ratio
    q_terminal -= branch.g * cross_imag * coupling_ratio

    return p_terminal, q_terminal


def compute_branch_flow_squared(network, model, branch_idx, fnode_idx, tnode_idx, s_m, s_o, p,
                                limit_type, direction='ij'):

    branch = network.branches[branch_idx]

    if limit_type == BRANCH_LIMIT_CURRENT or (limit_type == BRANCH_LIMIT_MIXED and not branch.is_transformer):

        rij = model.r[branch_idx, s_m, s_o, p] if branch.is_transformer else 1.0
        rij_sqr = model.r_sqr[branch_idx, s_m, s_o, p] if branch.is_transformer else 1.0
        vi_sqr = model.vmag_sqr[fnode_idx, s_m, s_o, p]
        vj_sqr = model.vmag_sqr[tnode_idx, s_m, s_o, p]
        cross_real, _ = _branch_voltage_products(
            model, network, branch_idx, fnode_idx, s_m, s_o, p
        )

        current_squared = (branch.g ** 2 + branch.b ** 2) * (
            vi_sqr + rij_sqr * vj_sqr - 2 * rij * cross_real
        )

        return current_squared

    if limit_type == BRANCH_LIMIT_APPARENT_POWER or (limit_type == BRANCH_LIMIT_MIXED and branch.is_transformer):
        if direction == 'ij':
            p_flow = model.pij[branch_idx, s_m, s_o, p]
            q_flow = model.qij[branch_idx, s_m, s_o, p]
        elif direction == 'ji':
            p_flow = model.pji[branch_idx, s_m, s_o, p]
            q_flow = model.qji[branch_idx, s_m, s_o, p]
        else:
            raise ValueError(f"Unknown branch flow direction: {direction}")
        return p_flow ** 2 + q_flow ** 2

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
                # Ordinary ESS net power follows the generator convention:
                # positive values are injections and therefore reduce net demand.
                Pd -= model.es_pnet[e, s_m, s_o, p]
                Qd -= model.es_qnet[e, s_m, s_o, p]

    for e in model.shared_energy_storages:
        es = network.shared_energy_storages[e]
        if es.bus == node.bus_i:
            # Shared ESS net power follows the load convention: positive values
            # are charging demand and therefore increase net demand.
            Pd += model.shared_es_pnet[e, s_m, s_o, p]
            Qd += model.shared_es_qnet[e, s_m, s_o, p]

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


def net_load_p_per_node_def(model, i, s_m, s_o, p, network, params):
    Pd, _ = compute_node_load(model, i, s_m, s_o, p, network, params)
    return Pd


def net_load_q_per_node_def(model, i, s_m, s_o, p, network, params):
    _, Qd = compute_node_load(model, i, s_m, s_o, p, network, params)
    return Qd


def net_gen_p_per_node_def(model, i, s_m, s_o, p, network):
    Pg, _ = compute_node_gen(model, i, s_m, s_o, p, network)
    return Pg


def net_gen_q_per_node_def(model, i, s_m, s_o, p, network):
    _, Qg = compute_node_gen(model, i, s_m, s_o, p, network)
    return Qg


def node_balance_p_rule(model, i, s_m, s_o, p, network, params):

    node = network.nodes[i]

    Pd = model.pc_node[i, s_m, s_o, p]
    Pg = model.pg_node[i, s_m, s_o, p]

    Pi = node.gs * model.vmag_sqr[i, s_m, s_o, p]

    for b in range(len(network.branches)):

        branch = network.branches[b]

        if not branch.status:
            continue

        if branch.fbus != node.bus_i and branch.tbus != node.bus_i:
            continue

        if branch.fbus == node.bus_i:
            fnode_idx = network.get_node_idx(branch.fbus)
            tnode_idx = network.get_node_idx(branch.tbus)
        else:
            fnode_idx = network.get_node_idx(branch.tbus)
            tnode_idx = network.get_node_idx(branch.fbus)

        rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0
        rij_sqr = model.r_sqr[b, s_m, s_o, p] if branch.is_transformer else 1.0

        vmag_sqr = model.vmag_sqr[fnode_idx, s_m, s_o, p]
        cross_real, cross_imag = _branch_voltage_products(
            model, network, b, fnode_idx, s_m, s_o, p
        )

        if branch.fbus == node.bus_i:
            Pi += branch.g * vmag_sqr * rij_sqr
        else:
            Pi += branch.g * vmag_sqr
        Pi -= rij * (branch.g * cross_real + branch.b * cross_imag)

    if params.slacks.node_balance.active_power:
        return Pg == Pd + Pi + (model.slack_node_balance_p_up[i, s_m, s_o, p] - model.slack_node_balance_p_down[i, s_m, s_o, p])
    else:
        return Pg == Pd + Pi


def node_balance_q_rule(model, i, s_m, s_o, p, network, params):

    node = network.nodes[i]

    Qd = model.qc_node[i, s_m, s_o, p]
    Qg = model.qg_node[i, s_m, s_o, p]

    Qi = -node.bs * model.vmag_sqr[i, s_m, s_o, p]

    for b in range(len(network.branches)):

        branch = network.branches[b]

        if not branch.status:
            continue

        if branch.fbus != node.bus_i and branch.tbus != node.bus_i:
            continue

        if branch.fbus == node.bus_i:
            fnode_idx = network.get_node_idx(branch.fbus)
            tnode_idx = network.get_node_idx(branch.tbus)
        else:
            fnode_idx = network.get_node_idx(branch.tbus)
            tnode_idx = network.get_node_idx(branch.fbus)

        rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0
        rij_sqr = model.r_sqr[b, s_m, s_o, p] if branch.is_transformer else 1.0

        vi_sqr = model.vmag_sqr[fnode_idx, s_m, s_o, p]
        cross_real, cross_imag = _branch_voltage_products(
            model, network, b, fnode_idx, s_m, s_o, p
        )

        if branch.fbus == node.bus_i:
            Qi -= (branch.b + 0.5 * branch.b_sh) * vi_sqr * rij_sqr
            Qi += rij * (branch.b * cross_real - branch.g * cross_imag)
        else:
            Qi -= (branch.b + 0.5 * branch.b_sh) * vi_sqr
            Qi += rij * (branch.b * cross_real - branch.g * cross_imag)

    if params.slacks.node_balance.reactive_power:
        return Qg == Qd + Qi + (model.slack_node_balance_q_up[i, s_m, s_o, p] - model.slack_node_balance_q_down[i, s_m, s_o, p])
    else:
        return Qg == Qd + Qi


def r_sqr_rule(m, b, s_m, s_o, p, network):
    if network.branches[b].is_transformer:
        return m.r_sqr[b, s_m, s_o, p] == m.r[b, s_m, s_o, p] ** 2
    return pe.Constraint.Skip


def _branch_terminal_power_expressions(m, branch_idx, s_m, s_o, p, network, direction):
    branch = network.branches[branch_idx]
    if direction == 'ij':
        terminal_node_idx = network.get_node_idx(branch.fbus)
        terminal_ratio_sqr = m.r_sqr[branch_idx, s_m, s_o, p] if branch.is_transformer else 1.0
    elif direction == 'ji':
        terminal_node_idx = network.get_node_idx(branch.tbus)
        terminal_ratio_sqr = 1.0
    else:
        raise ValueError(f"Unknown branch flow direction: {direction}")

    terminal_v_sqr = m.vmag_sqr[terminal_node_idx, s_m, s_o, p]
    cross_real, cross_imag = _branch_voltage_products(
        m, network, branch_idx, terminal_node_idx, s_m, s_o, p
    )
    coupling_ratio = m.r[branch_idx, s_m, s_o, p] if branch.is_transformer else 1.0

    return compute_branch_terminal_power(
        branch,
        terminal_v_sqr,
        cross_real,
        cross_imag,
        coupling_ratio=coupling_ratio,
        terminal_ratio_sqr=terminal_ratio_sqr,
    )


def pij_rule(m, branch_idx, s_m, s_o, p, network, params):
    pij, _ = _branch_terminal_power_expressions(m, branch_idx, s_m, s_o, p, network, 'ij')
    return m.pij[branch_idx, s_m, s_o, p] == pij


def qij_rule(m, branch_idx, s_m, s_o, p, network, params):
    _, qij = _branch_terminal_power_expressions(m, branch_idx, s_m, s_o, p, network, 'ij')
    return m.qij[branch_idx, s_m, s_o, p] == qij


def pji_rule(m, branch_idx, s_m, s_o, p, network, params):
    pji, _ = _branch_terminal_power_expressions(m, branch_idx, s_m, s_o, p, network, 'ji')
    return m.pji[branch_idx, s_m, s_o, p] == pji


def qji_rule(m, branch_idx, s_m, s_o, p, network, params):
    _, qji = _branch_terminal_power_expressions(m, branch_idx, s_m, s_o, p, network, 'ji')
    return m.qji[branch_idx, s_m, s_o, p] == qji


def branch_flow_def(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Expression.Skip

    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)

    return compute_branch_flow_squared(network, model, b, fnode_idx, tnode_idx, s_m, s_o, p, params.branch_limit_type, direction='ij')


def branch_flow_ji_def(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Expression.Skip

    fnode_idx = network.get_node_idx(branch.tbus)
    tnode_idx = network.get_node_idx(branch.fbus)

    return compute_branch_flow_squared(network, model, b, fnode_idx, tnode_idx, s_m, s_o, p, params.branch_limit_type, direction='ji')


def branch_flow_limit_rule(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Constraint.Skip

    rating = branch.rate / network.baseMVA or BRANCH_UNKNOWN_RATING

    if params.slacks.grid_operation.branch_flow:
        return model.flow_ij_sqr[b, s_m, s_o, p] <= rating ** 2 + model.slack_flow_ij_sqr[b, s_m, s_o, p]
    else:
        return model.flow_ij_sqr[b, s_m, s_o, p] <= rating ** 2 + EQUALITY_TOLERANCE


def branch_flow_limit_ji_rule(model, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return pe.Constraint.Skip

    rating = branch.rate / network.baseMVA or BRANCH_UNKNOWN_RATING

    if params.slacks.grid_operation.branch_flow:
        return model.flow_ji_sqr[b, s_m, s_o, p] <= rating ** 2 + model.slack_flow_ji_sqr[b, s_m, s_o, p]
    return model.flow_ji_sqr[b, s_m, s_o, p] <= rating ** 2 + EQUALITY_TOLERANCE


def setup_cost_parameters(model, params):

    model.penalty_ess_usage = pe.Param(initialize=PENALTY_ESS_USAGE, mutable=True)
    if params.obj_type == OBJ_MIN_COST:
        model.cost_load_curtailment = pe.Param(initialize=COST_CONSUMPTION_CURTAILMENT, mutable=True)
        model.penalty_gen_curtailment = pe.Param(initialize=PENALTY_GENERATION_CURTAILMENT, mutable=True)
    elif params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        model.penalty_load_curtailment = pe.Param(initialize=PENALTY_LOAD_CURTAILMENT, mutable=True)
        model.penalty_flex_usage = pe.Param(initialize=PENALTY_FLEXIBILITY_USAGE, mutable=True)
        model.penalty_gen_curtailment = pe.Param(initialize=PENALTY_GENERATION_CURTAILMENT, mutable=True)
    else:
        raise ValueError(f"[ERROR] Unrecognized or invalid objective type: {params.obj_type}.")


def build_objective(model, network, params):

    if params.obj_type == OBJ_MIN_COST:
        model.gen_cost_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(generation_cost_rule, network=network, params=params))
        model.flex_cost_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(flex_cost_rule, network=network, params=params))
        model.load_curt_cost_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(load_curtailment_cost_rule, network=network, params=params))
        model.total_gen_cost = pe.Expression(rule=partial(total_generation_cost_rule, network=network))
        model.total_flex_cost = pe.Expression(rule=partial(total_flex_cost_rule, network=network))
        model.total_load_curt_cost = pe.Expression(rule=partial(total_load_curtailment_cost_rule, network=network))
    else:
        model.load_curt_penalty_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(load_curtailment_penalty_rule, network=network, params=params))
        model.flex_penalty_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(flexibility_penalty_rule, network=network, params=params))
        model.total_load_curt_penalty = pe.Expression(rule=partial(total_load_curtailment_penalty_rule, network=network))
        model.total_flex_penalty = pe.Expression(rule=partial(total_flex_penalty_rule, network=network))

    model.gen_curt_penalty_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(gen_curtailment_penalty_rule, network=network, params=params))
    model.ess_utilization_cost_penalty_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(ess_utilization_cost_penalty_rule, network=network, params=params))
    model.slack_penalties_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(slack_penalties_rule, network=network, params=params))
    model.ess_complementarity_penalty_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(ess_complementarity_penalties_rule, network=network, params=params))
    model.total_gen_curt_penalty = pe.Expression(rule=partial(total_gen_curtailment_penalty_rule, network=network))
    model.total_ess_utilization_cost_penalty = pe.Expression(rule=partial(total_ess_utilization_cost_penalty_rule, network=network))
    model.total_slack_penalties = pe.Expression(rule=partial(total_slack_penalties_rule, network=network))
    model.total_ess_complementarity_penalties = pe.Expression(rule=partial(total_ess_complementarity_penalties_rule, network=network))

    model.objective = pe.Objective(sense=pe.minimize, rule=partial(objective_function_rule, params=params))


def objective_function_rule(model, params):
    if params.obj_type == OBJ_MIN_COST:
        obj = model.total_gen_cost + model.total_flex_cost + model.total_load_curt_cost + model.total_gen_curt_penalty
    else:
        obj = model.total_gen_curt_penalty + model.total_load_curt_penalty + model.total_flex_penalty
    obj += model.total_ess_utilization_cost_penalty
    obj += model.total_slack_penalties
    obj += model.total_ess_complementarity_penalties
    return obj


def generation_cost(model, network, s_m, s_o, params):
    c_p = network.cost_energy_p[s_m]
    gen_cost_scenario = 0.0
    for g in model.generators:
        if network.generators[g].is_controllable() and not (not network.is_transmission and network.generators[g].gen_type == GEN_REFERENCE):
            for p in model.periods:
                gen_cost_scenario += c_p[p] * network.baseMVA * model.pg[g, s_m, s_o, p]
    return gen_cost_scenario


def generation_cost_rule(model, s_m, s_o, network, params):
    return generation_cost(model, network, s_m, s_o, params)


def total_generation_cost_rule(model, network):
    total_gen_cost = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_gen_cost += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.gen_cost_scenario[s_m, s_o]
    return total_gen_cost


def flexibility_cost(model, network, s_m, s_o, params):
    flex_cost = 0.0
    if params.fl_reg:
        c_flex = network.cost_flex[s_m]
        for c in model.loads:
            if network.loads[c].fl_reg:
                for p in model.periods:
                    flex_cost += c_flex[p] * network.baseMVA * (
                            model.flex_p_down[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
                    )
    return flex_cost


def flex_cost_rule(model, s_m, s_o, network, params):
    return flexibility_cost(model, network, s_m, s_o, params)


def total_flex_cost_rule(model, network):
    total_flex_cost = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_flex_cost += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.flex_cost_scenario[s_m, s_o]
    return total_flex_cost


def load_curtailment_cost(model, network, s_m, s_o, params):
    load_curt_cost = 0.0
    if params.l_curt:
        cost = model.cost_load_curtailment
        for c in model.loads:
            for p in model.periods:
                load_curt_cost += cost * network.baseMVA * (
                        model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p] +
                        model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p]
                    )
    return load_curt_cost


def load_curtailment_cost_rule(model, s_m, s_o, network, params):
    return load_curtailment_cost(model, network, s_m, s_o, params)


def total_load_curtailment_cost_rule(model, network):
    total_load_curtailment_cost = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_load_curtailment_cost += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.load_curt_cost_scenario[s_m, s_o]
    return total_load_curtailment_cost


def gen_curtailment_penalty(model, network, s_m, s_o, params):
    gen_curt_penalty = 0.0
    if params.rg_curt:
        penalty = model.penalty_gen_curtailment
        for g in model.generators:
            if network.generators[g].is_curtaillable():
                for p in model.periods:
                    gen_curt_penalty += penalty * network.baseMVA * (model.pg_avail[g, s_o, p] - model.pg[g, s_m, s_o, p])
    return gen_curt_penalty


def gen_curtailment_penalty_rule(model, s_m, s_o, network, params):
    return gen_curtailment_penalty(model, network, s_m, s_o, params)


def total_gen_curtailment_penalty_rule(model, network):
    total_gen_curtailment_penalty = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_gen_curtailment_penalty += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.gen_curt_penalty_scenario[s_m, s_o]
    return total_gen_curtailment_penalty


def load_curtailment_penalty(model, network, s_m, s_o, params):
    load_curt_penalty = 0.0
    if params.l_curt:
        penalty = model.penalty_load_curtailment
        for c in model.loads:
            for p in model.periods:
                load_curt_penalty += penalty * network.baseMVA * (
                        model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p] +
                        model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p]
                )
    return load_curt_penalty


def load_curtailment_penalty_rule(model, s_m, s_o, network, params):
    return load_curtailment_penalty(model, network, s_m, s_o, params)


def total_load_curtailment_penalty_rule(model, network):
    total_load_curtailment_penalty = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_load_curtailment_penalty += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.load_curt_penalty_scenario[s_m, s_o]
    return total_load_curtailment_penalty


def flexibility_penalty(model, network, s_m, s_o, params):
    flex_penalty = 0.0
    if params.fl_reg:
        penalty = model.penalty_flex_usage
        for c in model.loads:
            for p in model.periods:
                flex_penalty += penalty * network.baseMVA * (
                        model.flex_p_up[c, s_m, s_o, p] + model.flex_p_down[c, s_m, s_o, p] +
                        model.flex_q_up[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
                )
    return flex_penalty


def flexibility_penalty_rule(model, s_m, s_o, network, params):
    return flexibility_penalty(model, network, s_m, s_o, params)


def total_flex_penalty_rule(model, network):
    total_flex_penalty = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_flex_penalty += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.flex_penalty_scenario[s_m, s_o]
    return total_flex_penalty


def ess_utilization_cost_penalty(model, network, s_m, s_o, params):
    cost = 0.0
    for e in model.shared_energy_storages:
        for p in model.periods:
            cost += model.penalty_ess_usage * network.baseMVA * (model.shared_es_sch[e, s_m, s_o, p] + model.shared_es_sdch[e, s_m, s_o, p])
    if params.es_reg:
        for e in model.energy_storages:
            for p in model.periods:
                cost += model.penalty_ess_usage * network.baseMVA * (model.es_sch[e, s_m, s_o, p] + model.es_sdch[e, s_m, s_o, p])
    return cost


def ess_utilization_cost_penalty_rule(model, s_m, s_o, network, params):
    return ess_utilization_cost_penalty(model, network, s_m, s_o, params)


def total_ess_utilization_cost_penalty_rule(model, network):
    total_ess_utilization_cost_penalty = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_ess_utilization_cost_penalty += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.ess_utilization_cost_penalty_scenario[s_m, s_o]
    return total_ess_utilization_cost_penalty


def slack_penalties(model, network, s_m, s_o, params):

    total = 0
    base = network.baseMVA

    for i in model.nodes:
        for p in model.periods:
            if params.slacks.grid_operation.voltage:
                total += PENALTY_VOLTAGE_SQUARED * (
                    model.slack_v_sqr_down[i, s_m, s_o, p]
                    + model.slack_v_sqr_up[i, s_m, s_o, p]
                )
            if params.slacks.node_balance.active_power:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_p_up[i, s_m, s_o, p] + model.slack_node_balance_p_down[i, s_m, s_o, p])
            if params.slacks.node_balance.reactive_power:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_q_up[i, s_m, s_o, p] + model.slack_node_balance_q_down[i, s_m, s_o, p])

    for b in model.branches:
        for p in model.periods:
            if params.slacks.grid_operation.branch_flow:
                total += base * PENALTY_CURRENT * (model.slack_flow_ij_sqr[b, s_m, s_o, p])

    if params.slacks.grid_operation.branch_flow:
        for b in model.apparent_power_limited_branches:
            for p in model.periods:
                total += base * PENALTY_CURRENT * model.slack_flow_ji_sqr[b, s_m, s_o, p]

    if params.fl_reg and params.slacks.flexibility.day_balance:
        for c in model.loads:
            if network.loads[c].fl_reg:
                total += base * PENALTY_FLEXIBILITY * sum(model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o])
                total += base * PENALTY_FLEXIBILITY * sum(model.slack_flex_q_balance_up[c, s_m, s_o] + model.slack_flex_q_balance_down[c, s_m, s_o] )

    return total



def slack_penalties_rule(model, s_m, s_o, network, params):
    return slack_penalties(model, network, s_m, s_o, params)


def ess_complementarity_penalties(model, network, s_m, s_o, p, params):

    total = 0
    base = network.baseMVA

    if params.es_reg:
        for e in model.energy_storages:
            for p in model.periods:
                if params.ess_model == ESS_MODEL_BILINEAR_RELAXATION:
                    total += base * PENALTY_ESS_COMPLEMENTARITY * (model.es_sch[e, s_m, s_o, p] * model.es_sdch[e, s_m, s_o, p])
            if params.slacks.ess.day_balance:
                total += base * PENALTY_ESS_BALANCE * (model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o])

    for e in model.shared_energy_storages:
        for p in model.periods:
            if params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:
                total += base * PENALTY_ESS_COMPLEMENTARITY * (model.shared_es_sch[e, s_m, s_o, p] * model.shared_es_sdch[e, s_m, s_o, p])
        if params.slacks.shared_ess.day_balance:
            total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m, s_o] + model.slack_shared_es_soc_final_down[e, s_m, s_o])

    return total


def ess_complementarity_penalties_rule(model, s_m, s_o, network, params):
    return ess_complementarity_penalties(model, network, s_m, s_o, params, params)


def total_slack_penalties_rule(model, network):
    total_slack_penalties = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_slack_penalties += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.slack_penalties_scenario[s_m, s_o]
    return total_slack_penalties


def total_ess_complementarity_penalties_rule(model, network):
    total_slack_penalties = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_slack_penalties += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.ess_complementarity_penalty_scenario[s_m, s_o]
    return total_slack_penalties


def dn_interface_expected_vmag_def(m, p, network):
    expected_vmag = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.vmag_adn[s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return expected_vmag


def dn_interface_expected_pf_p_def(m, p, network):
    expected_pf_p = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.pg_adn[s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return expected_pf_p


def dn_interface_expected_pf_q_def(m, p, network):
    expected_pf_q = sum(
        network.prob_market_scenarios[s_m] *
        network.prob_operation_scenarios[s_o] *
        m.qg_adn[s_m, s_o, p]
        for s_m in m.scenarios_market
        for s_o in m.scenarios_operation
    )
    return expected_pf_q


def dn_interface_expected_vmag_rule(m, p, network):
    return m.expected_interface_vmag[p] == dn_interface_expected_vmag_def(m, p, network)


def dn_interface_expected_pf_p_rule(m, p, network):
    return m.expected_interface_pf_p[p] == dn_interface_expected_pf_p_def(m, p, network)


def dn_interface_expected_pf_q_rule(m, p, network):
    return m.expected_interface_pf_q[p] == dn_interface_expected_pf_q_def(m, p, network)


def dn_interface_expected_sess_p_def(m, p, network, shared_ess_idx):
    return sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.shared_es_pnet[shared_ess_idx, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)


def dn_interface_expected_sess_q_def(m, p, network, shared_ess_idx):
    return sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.shared_es_qnet[shared_ess_idx, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)


def dn_interface_expected_sess_p_rule(m, p, network, shared_ess_idx):
    return m.expected_shared_ess_p[p] == dn_interface_expected_sess_p_def(m, p, network, shared_ess_idx)


def dn_interface_expected_sess_q_rule(m, p, network, shared_ess_idx):
    return m.expected_shared_ess_q[p] == dn_interface_expected_sess_q_def(m, p, network, shared_ess_idx)


def tn_interface_expected_vmag_def(m, dn, p, network):
    expected_vmag = sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.vmag_adn[dn, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)
    return expected_vmag


def tn_interface_expected_pf_p_def(m, dn, p, network):
    expected_pf_p = sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.pc_adn[dn, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)
    return expected_pf_p


def tn_interface_expected_pf_q_def(m, dn, p, network):
    expected_pf_q = sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.qc_adn[dn, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)
    return expected_pf_q


def tn_interface_expected_vmag_rule(m, dn, p, network):
    return m.expected_interface_vmag[dn, p] == tn_interface_expected_vmag_def(m, dn, p, network)


def tn_interface_expected_pf_p_rule(m, dn, p, network):
    return m.expected_interface_pf_p[dn, p] == tn_interface_expected_pf_p_def(m, dn, p, network)


def tn_interface_expected_pf_q_rule(m, dn, p, network):
    return m.expected_interface_pf_q[dn, p] == tn_interface_expected_pf_q_def(m, dn, p, network)


def tn_interface_expected_sess_p_def(m, e, p, network):
    return sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.shared_es_pnet[e, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)


def tn_interface_expected_sess_q_def(m, e, p, network):
    return sum(network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * m.shared_es_qnet[e, s_m, s_o, p] for s_m in m.scenarios_market for s_o in m.scenarios_operation)


def tn_interface_expected_sess_p_rule(m, e, p, network):
    return m.expected_shared_ess_p[e, p] == tn_interface_expected_sess_p_def(m, e, p, network)


def tn_interface_expected_sess_q_rule(m, e, p, network):
    return m.expected_shared_ess_q[e, p] == tn_interface_expected_sess_q_def(m, e, p, network)
