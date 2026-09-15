from functools import partial
from math import tan, acos, sqrt, radians
from helper_functions import *
from definitions import *


def get_vmag_node_indices(network):
    """Return validated physical-node indices that need explicit ``vmag``."""

    if network.is_transmission:
        bus_ids = list(network.active_distribution_network_nodes)
        source = 'active distribution-network interface'
    else:
        reference_nodes = [
            node.bus_i for node in network.nodes if node.type == BUS_REF
        ]
        if len(reference_nodes) != 1:
            raise ValueError(
                f'Network {network.name} must have exactly one reference node; '
                f'found {len(reference_nodes)}.'
            )
        bus_ids = reference_nodes
        source = 'reference'

    if len(bus_ids) != len(set(bus_ids)):
        raise ValueError(
            f'Network {network.name} has duplicate {source} bus IDs: {bus_ids}.'
        )

    node_indices = []
    for bus_id in bus_ids:
        matches = [
            node_idx
            for node_idx, node in enumerate(network.nodes)
            if node.bus_i == bus_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f'Network {network.name} {source} bus ID {bus_id!r} maps to '
                f'{len(matches)} physical nodes; expected exactly one.'
            )
        node_indices.append(matches[0])

    return node_indices


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

    # PV bus with enforced generator-voltage setpoint
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        v_min = sqrt(max(vg ** 2 - SMALL_TOLERANCE, 0.0))
        v_max = sqrt(vg ** 2 + SMALL_TOLERANCE)
        return (v_min, v_max)

    # Bus where voltage-limit relaxation is allowed
    if _voltage_magnitude_slack_enabled(node, params):
        v_min = max(node.v_min - VMAG_VIOLATION_ALLOWED, 0.0)
        v_max = node.v_max + VMAG_VIOLATION_ALLOWED
        return (v_min, v_max)

    # Hard voltage limits: REF buses, or when voltage slacks are disabled
    return (node.v_min, node.v_max)


def vmag_sqr_bounds(m, i, s_m, s_o, p, network, params):

    node = network.nodes[i]

    # PV bus with enforced generator-voltage setpoint
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return max(vg ** 2 - SMALL_TOLERANCE, 0.0), vg ** 2 + SMALL_TOLERANCE

    # Bus where voltage-limit relaxation is allowed
    if _voltage_magnitude_slack_enabled(node, params):
        v_min = max(node.v_min - VMAG_VIOLATION_ALLOWED, 0.0)
        v_max = node.v_max + VMAG_VIOLATION_ALLOWED
        return (v_min ** 2, v_max ** 2)

    # Hard voltage limits
    return (node.v_min ** 2, node.v_max ** 2)


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


# P5.15 Step 3 (D6, signed table `P5_15_S31_PENALTY_TABLE_DRAFT.md`): the
# local-ESS day-balance slack is bounded to this fraction of the (fixed)
# energy capacity; the shared-ESS one is now bounded the same way (see
# `configure_shared_ess_operational_state`), tracking the CURRENT candidate
# capacity rather than a build-time snapshot.
ESS_DAY_BALANCE_SLACK_FRACTION = 0.05


def slack_es_balance_bounds(m, e, s_m, s_o, network):
    ess = network.energy_storages[e]
    return (0.00, ess.e * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE)


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
    # P5.15-1b Candidate 3' (per-period normalization of this row) was
    # implemented, measured and HELD by the Planner on 2026-09-13: it failed its
    # own gate -- dispatch differed by 1.95e-05 against a 1e-8 requirement, the
    # objective by 2.86e-03 (~0.3%), neither explained by solver tolerance, and
    # cold-start iterations ROSE from 39 to 65, the opposite of the conditioning
    # gain it was proposed for. Evidence: data/SRP1/Results/P5151B/.
    # The un-normalized row below is therefore unchanged and in force.
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
            return q_up == q_down + m.slack_flex_q_balance_up[c, s_m, s_o] - m.slack_flex_q_balance_down[c, s_m, s_o]
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
    # Canonical ordinary-ESS convention (P4.6-B1): the ordinary ESS is
    # represented as a LOAD, so positive net power is consumption from the
    # network (charging) and negative net power is injection (discharging).
    return m.es_pnet[e,s_m,s_o,p] == m.es_pch[e,s_m,s_o,p] - m.es_pdch[e,s_m,s_o,p]


def ess_converter_capability_rule(m, e, s_m, s_o, p, network):
    """Ordinary-ESS converter apparent-power capability (P5.4-B).

    Parity with the shared-ESS `sess_converter_capability` introduced in
    P5.4-A. Replaces the former apparent-charge/discharge geometry
    (`ess_snet_def` and its kappa_es normalization, `ess_pch_link`,
    `ess_pdch_link`). Reactive power is limited by the converter rating but no
    longer participates in the stored battery energy.

    Unlike shared ESS, the ordinary-ESS rating is fixed network data, not a
    decision variable, so `ess.s` enters as a constant.
    """
    ess = network.energy_storages[e]
    return (m.es_pnet[e, s_m, s_o, p] ** 2
            + m.es_qnet[e, s_m, s_o, p] ** 2) <= ess.s ** 2


def ess_active_sum_limit_rule(m, e, s_m, s_o, p, network):
    """Active charge/discharge envelope (P5.4-B).

    Derived from the retired apparent set exactly as its shared-ESS counterpart
    was: `pch <= sch`, `pdch <= sdch` and `sch + sdch <= S + EQUALITY_TOLERANCE`
    together imply `pch + pdch <= S + EQUALITY_TOLERANCE`. The pre-existing
    tolerance is preserved rather than tightened, so the feasible set is not
    narrowed by this stage.
    """
    ess = network.energy_storages[e]
    return (m.es_pch[e, s_m, s_o, p]
            + m.es_pdch[e, s_m, s_o, p]) <= ess.s + EQUALITY_TOLERANCE


def ess_soc_limits_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return pe.inequality(ess.e_min - EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, p], ess.e_max + EQUALITY_TOLERANCE)


def ess_phi_limits_lower(m, e, s_m, s_o, p, network):
    # Re-derived for the load-positive ordinary-ESS convention (P4.6-B1): with
    # es_pnet = pch - pdch, the converter capability region is expressed about
    # the charging (consumption) direction, so pch carries tangent_lower and
    # pdch carries tangent_upper -- the mirror of the previous
    # generation-positive form, under which qnet had the opposite sign.
    ess = network.energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.es_pch[e, s_m, s_o, p]
    pdch = m.es_pdch[e, s_m, s_o, p]
    return m.es_qnet[e, s_m, s_o, p] >= tangent_lower * pch - tangent_upper * pdch


def ess_phi_limits_upper(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.es_pch[e, s_m, s_o, p]
    pdch = m.es_pdch[e, s_m, s_o, p]
    return m.es_qnet[e, s_m, s_o, p] <= tangent_upper * pch - tangent_lower * pdch


def ess_soc_rule(m, e, s_m, s_o, p, network, params):

    # P5.4-B: parity with sess_soc_rule. Stored battery energy is driven by
    # ACTIVE power only, so pure reactive operation leaves the state of charge
    # unchanged. The time step is explicit (period_duration_hours) rather than
    # an implicit assumption; for the standard 24-instant representative day it
    # is exactly 1 h, which reproduces the previous numerical coefficient.
    ess = network.energy_storages[e]
    eff_ch = ess.eff_ch
    eff_dch = ess.eff_dch
    dt = period_duration_hours(m)
    if p == 0:
        soc_prev = ess.e_init
    else:
        soc_prev = m.es_soc[e, s_m, s_o, p-1]

    delta = (eff_ch * m.es_pch[e, s_m, s_o, p] * dt
             - m.es_pdch[e, s_m, s_o, p] * dt / eff_dch)

    return m.es_soc[e, s_m, s_o, p] == soc_prev + delta


def ess_pch_hat_link_rule(m, e, s_m, s_o, p, network):
    """Ordinary-ESS dimensionless link (P5.4-H1), parity with the shared ESS.

    Here `S_rated` is fixed network data rather than a decision variable, so the
    row is linear; the unit coefficient on the physical variable is retained for
    the same structural reason.
    """
    ess = network.energy_storages[e]
    return m.es_pch[e, s_m, s_o, p] - ess.s * m.es_pch_hat[e, s_m, s_o, p] == 0


def ess_pdch_hat_link_rule(m, e, s_m, s_o, p, network):
    ess = network.energy_storages[e]
    return m.es_pdch[e, s_m, s_o, p] - ess.s * m.es_pdch_hat[e, s_m, s_o, p] == 0


def ess_comp_rule(m, e, s_m, s_o, p, network, params):
    # P5.4-H1: the BILINEAR_RELAXATION branch uses the dimensionless pair, in
    # parity with sess_comp_rule. Exact reformulation for positive rating; the
    # tolerance value is unchanged.
    pch = m.es_pch[e, s_m, s_o, p]
    pdch = m.es_pdch[e, s_m, s_o, p]
    if params.ess_model == ESS_MODEL_EXACT:
        return pch * pdch <= EQUALITY_TOLERANCE
    elif params.ess_model == ESS_MODEL_BILINEAR_RELAXATION:
        return (m.es_pch_hat[e, s_m, s_o, p]
                * m.es_pdch_hat[e, s_m, s_o, p]) <= ESS_COMPLEMENTARITY_TOLERANCE
    elif params.ess_model == ESS_MODEL_POLYNOMIAL_COMPLEMENTARITY:
        return pch ** 2 + pdch ** 2 <= (pch + pdch) ** 2 + EQUALITY_TOLERANCE
    else:
        return pe.Constraint.Skip


def ess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = network.energy_storages[e].e_init
    final_p = m.periods[-1]
    if params.slacks.ess.day_balance:
        return m.es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_es_soc_final_up[e, s_m, s_o] - m.slack_es_soc_final_down[e, s_m, s_o]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, final_p] - final_soc, EQUALITY_TOLERANCE)


# Shared Energy Storage
# P5.15-1b (PLANNER_BRIEF_2026-09-13.md, Step 2 Candidate 1, form (a)): the
# power-factor rows `sess_phi_limits_lower`/`sess_phi_limits_upper`, which tied
# `|qnet|` to instantaneous `pch + pdch`, are DELETED. The P5.15-2 audit found
# they created an LICQ-degenerate vertex at every idle period (four active
# inequalities in a 3-dimensional subspace at `pch = pdch = 0`) and disagreed
# with the ESSO subproblem's own reactive feasible set, which imposes only the
# converter circle. `sess_converter_capability` (pnet^2 + qnet^2 <= S_rated^2)
# now carries the reactive limit alone.
#
# PLANNER AMENDMENT (2026-09-13): the two rule FUNCTIONS below are RETAINED,
# unwired and unused, rather than deleted. Deleting them broke unpickling of
# every preserved network fixture -- including
# `data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl`, the cycle-21 anchor
# of the whole P5.12 line, and the FrozenSMOPF comparators -- because the
# pickled models hold `functools.partial` objects that resolve these names at
# load time. Candidate 1 requires the ROWS to be gone, which is achieved by
# removing the wiring in `network.py` and the entries in
# `_SHARED_ESS_OPERATIONAL_CONSTRAINTS`; it does not require deleting the
# callables. This follows the same "deactivate, do not delete" precedent as
# `_add_benders_cut`. Do not call these from any model build.


def sess_phi_limits_lower(m, e, s_m, s_o, p, network):
    """RETIRED by P5.15-1b Candidate 1 -- retained only for fixture unpickling."""
    ess = network.shared_energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.shared_es_pch[e, s_m, s_o, p]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return m.shared_es_qnet[e, s_m, s_o, p] >= tangent_lower * pch - tangent_upper * pdch


def sess_phi_limits_upper(m, e, s_m, s_o, p, network):
    """RETIRED by P5.15-1b Candidate 1 -- retained only for fixture unpickling."""
    ess = network.shared_energy_storages[e]
    tangent_lower, tangent_upper = _power_factor_tangents(ess)
    pch = m.shared_es_pch[e, s_m, s_o, p]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return m.shared_es_qnet[e, s_m, s_o, p] <= tangent_upper * pch - tangent_lower * pdch


def period_duration_hours(m):
    """Duration of one optimization period, in hours.

    Representative days span HOURS_PER_REPRESENTATIVE_DAY hours and are split
    into `len(m.periods)` equal instants, so for the standard 24-instant day
    this is exactly 1 h. Used by the active-energy SOC recursions (P5.4) so the
    time step is explicit rather than an implicit assumption.
    """
    n_periods = len(m.periods)
    if n_periods <= 0:
        raise ValueError('Model has no periods; cannot derive the period duration.')
    return HOURS_PER_REPRESENTATIVE_DAY / n_periods


def sess_converter_capability_rule(m, e, s_m, s_o, p):
    """Shared-ESS converter apparent-power capability (P5.4-A).

    Replaces the former apparent-charge/discharge geometry. Reactive power is
    limited by the converter rating but no longer participates in the stored
    battery energy.

    P5.15-1b (Step 2 Candidate 2): `S_rated` is read from
    `shared_es_s_rated_fixed`, a mutable Param, rather than from the retired
    `shared_es_s_rated` Var -- see the note above `configure_shared_ess_operational_state`.
    This row is therefore a convex SOC constraint, not an indefinite quadratic.
    """
    return (m.shared_es_pnet[e, s_m, s_o, p] ** 2
            + m.shared_es_qnet[e, s_m, s_o, p] ** 2) <= m.shared_es_s_rated_fixed[e] ** 2


def sess_active_sum_limit_rule(m, e, s_m, s_o, p):
    """Active charging/discharging envelope (P5.4-A).

    Derived from the retired production feasible set, which enforced
    `pch <= sch`, `pdch <= sdch` and `sch + sdch <= S_rated`; those together
    imply `pch + pdch <= S_rated`, so this preserves the original active-power
    envelope exactly. `S_rated` is `shared_es_s_rated_fixed` (P5.15-1b
    Candidate 2); this row is linear.
    """
    return (m.shared_es_pch[e, s_m, s_o, p]
            + m.shared_es_pdch[e, s_m, s_o, p]) <= m.shared_es_s_rated_fixed[e]


def sess_soc_lower_limit(m, e, s_m, s_o, p):
    soc_min = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] >= soc_min


def sess_soc_upper_limit(m, e, s_m, s_o, p):
    soc_max = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] <= soc_max


def sess_pch_hat_link_rule(m, e, s_m, s_o, p):
    """Link the dimensionless charging variable to physical power (P5.4-H1).

    Written as `pch - S_rated * pch_hat == 0` rather than `pch_hat == pch/S` so
    that no expression ever divides by the rated-capacity decision variable, and
    so the row keeps a unit coefficient on the physical variable. That unit
    coefficient is what keeps the row from being zero-gradient at zero dispatch
    -- the defect that `sess_snet_def` had.

    P5.15-1b (Step 2 Candidate 2): `S_rated` is now `shared_es_s_rated_fixed`
    (a Param), so this row is LINEAR rather than bilinear.
    """
    return (m.shared_es_pch[e, s_m, s_o, p]
            - m.shared_es_s_rated_fixed[e] * m.shared_es_pch_hat[e, s_m, s_o, p]) == 0


def sess_pdch_hat_link_rule(m, e, s_m, s_o, p):
    return (m.shared_es_pdch[e, s_m, s_o, p]
            - m.shared_es_s_rated_fixed[e] * m.shared_es_pdch_hat[e, s_m, s_o, p]) == 0


def sess_comp_rule(m, e, s_m, s_o, p, network, params):
    # P5.4-H1: the BILINEAR_RELAXATION branch is written on the dimensionless
    # charge/discharge variables. For positive capacity this is an exact
    # reformulation of the previous relative condition
    # `pch * pdch <= eps * S_rated^2` -- substituting the link equalities
    # reproduces it identically -- but the row now sits at O(1) with an RHS of
    # 1e-4 instead of O(S^2) ~ 1e-12, which is what IPOPT can actually resolve.
    # ESS_COMPLEMENTARITY_TOLERANCE is unchanged at 1e-4; this stage rescales
    # the row, it does not tighten the physical tolerance.
    pch = m.shared_es_pch[e, s_m, s_o, p]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    if params.shared_ess_model == ESS_MODEL_EXACT:
        return pch * pdch <= EQUALITY_TOLERANCE
    elif params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:
        return (m.shared_es_pch_hat[e, s_m, s_o, p]
                * m.shared_es_pdch_hat[e, s_m, s_o, p]) <= ESS_COMPLEMENTARITY_TOLERANCE
    elif params.shared_ess_model == ESS_MODEL_POLYNOMIAL_COMPLEMENTARITY:
        return pch ** 2 + pdch ** 2 <= (pch + pdch) ** 2 + EQUALITY_TOLERANCE
    else:
        return pe.Constraint.Skip


def sess_soc_rule(m, e, s_m, s_o, p, network, params):

    # P5.4-A: stored battery energy is driven by ACTIVE power only, so pure
    # reactive operation leaves the state of charge unchanged. The time step is
    # explicit (period_duration_hours) rather than an implicit assumption; for
    # the standard 24-instant representative day it is exactly 1 h, which
    # reproduces the previous numerical coefficient.
    sess = network.shared_energy_storages[e]
    eff_ch = sess.eff_ch
    eff_dch = sess.eff_dch
    dt = period_duration_hours(m)
    if p == 0:
        soc_prev = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
    else:
        soc_prev = m.shared_es_soc[e, s_m, s_o, p - 1]

    delta = (eff_ch * m.shared_es_pch[e, s_m, s_o, p] * dt
             - m.shared_es_pdch[e, s_m, s_o, p] * dt / eff_dch)

    return m.shared_es_soc[e, s_m, s_o, p] == soc_prev + delta


def sess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
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
    # P5.4-H1: the dimensionless pair follows exactly the same zero-capacity
    # gating as the physical pair -- fixed at 0 when inactive, unfixed otherwise.
    'shared_es_pch_hat',
    'shared_es_pdch_hat',
    'shared_es_pnet',
    'shared_es_qnet',
    'shared_es_soc',
    'slack_shared_es_soc_final_up',
    'slack_shared_es_soc_final_down',
)

# P5.4-D2-P: these four variables previously carried numerical bounds rewritten
# from the installed rating (pch, pdch in [0, S]; pnet, qnet in [-S, S]). Those
# bounds are REDUNDANT -- every one of them is implied by a symbolic row that is
# already in the model:
#
#   pch <= S    <-  pch + pdch <= S with pdch >= 0   (sess_active_sum_limit)
#               <-  pch = S*pch_hat with pch_hat <= 1 (P5.4-H1 link)
#   pdch <= S   <-  symmetrically
#   |pnet| <= S <-  pnet^2 + qnet^2 <= S^2           (sess_converter_capability)
#   |qnet| <= S <-  symmetrically
#
# but because they were written from the fixed capacity PARAMETER, they made
# installed power enter the NLP through variable-bound multipliers as well as
# through the rated-capacity variable. The Benders extraction reads only the
# dual of the capacity-fixing row, so that second channel was silently dropped:
# P5.4-D2 measured the omitted term at 28 %-625 % of the true local derivative,
# with the wrong sign in 3 of 8 audited cases.
#
# Removing them is an exact reformulation for positive capacity and routes all
# S-dependence through symbolic rows and `shared_es_s_rated`, which makes the
# fixing-row dual the structurally complete local derivative. The bounds below
# are therefore the CAPACITY-INDEPENDENT ones retained at positive capacity;
# `None` means the symbolic rows carry the bound instead.
#
# The zero-capacity collapse to [0, 0] is a structural gate, not an
# S-dependent bound, and is deliberately kept -- see
# `configure_shared_ess_operational_state`.
_SHARED_ESS_ZERO_GATED_BOUND_VARIABLES = (
    ('shared_es_pch', 0.0, None),      # nonnegativity only; upper bound is symbolic
    ('shared_es_pdch', 0.0, None),
    ('shared_es_pnet', None, None),    # bounded by the capability circle
    ('shared_es_qnet', None, None),
)

_SHARED_ESS_OPERATIONAL_CONSTRAINTS = (
    'sess_pnet_def',
    'sess_pch_hat_link',
    'sess_pdch_hat_link',
    'sess_converter_capability',
    'sess_active_sum_limit',
    # P5.15-1b Candidate 1: 'sess_phi_limit_lower'/'sess_phi_limit_upper' are
    # removed (see the deletion note above `period_duration_hours`); no longer
    # wired in `network.py`, so they are dropped from this activation list too.
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

    # P5.4-A: the shared-ESS rows now depend directly on the rated-capacity
    # parameter (`shared_es_s_rated_fixed`), so a capacity change is carried by
    # the model itself. The former `sess_snet_def` kappa scale and its
    # KKT-consistent multiplier transfer are gone with that row, and no
    # replacement multiplier transformation is required.
    #
    # P5.15-1b (Step 2 Candidate 2): `shared_es_s_rated`/`shared_es_e_rated`
    # (the `Var`s pinned to these Params by the now-deleted
    # `shared_energy_storage_s/e_sensitivities` equalities) are gone --
    # `shared_es_s_rated_fixed`/`shared_es_e_rated_fixed`, set here, are the
    # sole capacity quantities read by the shared-ESS rows.
    model.shared_es_s_rated_fixed[shared_ess_idx].set_value(s_capacity)
    model.shared_es_e_rated_fixed[shared_ess_idx].set_value(e_capacity)

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

    # P5.15 Step 3 (D6, signed table `P5_15_S31_PENALTY_TABLE_DRAFT.md`): the
    # shared-ESS day-balance slack is bounded to the same fraction of energy
    # capacity as the local-ESS one (`slack_es_balance_bounds`,
    # `ESS_DAY_BALANCE_SLACK_FRACTION`). Capacity is candidate-dependent (Step
    # 2 Candidate 2 made it a mutable Param), so a `bounds=` callable evaluated
    # once at Var-construction time cannot track it; the bound is instead
    # re-applied here, every time this function runs (i.e. every time the
    # candidate capacity is set -- `shared_resources_planning.py`, every
    # caller of `configure_shared_ess_operational_state`). Previously
    # unbounded (D6 defect).
    for variable_name in ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down'):
        if not hasattr(model, variable_name):
            continue
        variable = getattr(model, variable_name)
        slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE
        for entry in _component_entries_for_shared_ess(variable, shared_ess_idx):
            entry.setlb(0.0)
            entry.setub(slack_ub)

    for constraint_name in _SHARED_ESS_OPERATIONAL_CONSTRAINTS:
        constraint = getattr(model, constraint_name)
        for entry in _component_entries_for_shared_ess(
                constraint, shared_ess_idx):
            if inactive:
                entry.deactivate()
            else:
                entry.activate()

    # P5.4-D2-P: at zero capacity the box still collapses to [0, 0] -- that is a
    # structural gate and the variables are additionally fixed at 0 above. At
    # POSITIVE capacity the bounds are restored to their capacity-INDEPENDENT
    # values, so installed power no longer enters through bound multipliers.
    # The model stays bounded there via sess_active_sum_limit, the H1 links with
    # pch_hat/pdch_hat in [0, 1], and sess_converter_capability.
    for variable_name, positive_lb, positive_ub in _SHARED_ESS_ZERO_GATED_BOUND_VARIABLES:
        if not hasattr(model, variable_name):
            continue
        variable = getattr(model, variable_name)
        for entry in _component_entries_for_shared_ess(variable, shared_ess_idx):
            if inactive:
                entry.setlb(0.0)
                entry.setub(0.0)
            else:
                entry.setlb(positive_lb)
                entry.setub(positive_ub)

    _configure_shared_ess_expected_schedule(
        model, shared_ess_idx, inactive
    )

    return inactive


# P5.15-1b (PLANNER_BRIEF_2026-09-13.md, Step 2 Candidate 2): `sess_s_sensitivities`
# / `sess_e_sensitivities` -- the equalities pinning the (now-deleted)
# `shared_es_s_rated`/`shared_es_e_rated` Vars to `shared_es_s_rated_fixed`/
# `shared_es_e_rated_fixed` -- are DELETED, not deactivated: the Vars they
# pinned no longer exist. Their Benders capacity-sensitivity duals
# (`network_data.py::_get_sensitivities`) are consumed only by the
# already-retired `_add_benders_cut` path (Step 1 item 5); that consumer is
# separately deactivated with its own comment.


# P5.15-1b Candidate 2: the capacity-pinning ROWS are gone (the Vars they pinned
# are now mutable Params). The two rule functions are RETAINED, unwired and
# unused, for the same reason as `sess_phi_limits_*` above: preserved network
# fixtures hold `functools.partial` objects that resolve these names at unpickle
# time. They reference `m.shared_es_s_rated`/`m.shared_es_e_rated`, which no
# longer exist on newly built models, so they must never be called -- they exist
# only so that historical artifacts remain loadable.


def sess_s_sensitivities(m, e):
    """RETIRED by P5.15-1b Candidate 2 -- retained only for fixture unpickling."""
    return m.shared_es_s_rated_fixed[e] == m.shared_es_s_rated[e]


def sess_e_sensitivities(m, e):
    """RETIRED by P5.15-1b Candidate 2 -- retained only for fixture unpickling."""
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
                # Ordinary ESS net power follows the load convention: positive
                # values are charging/absorption demand and therefore increase
                # net demand; negative values are injections.
                Pd += model.es_pnet[e, s_m, s_o, p]
                Qd += model.es_qnet[e, s_m, s_o, p]

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
    # P5.15 Step 3 (row 8): shared-ESS usage weight, split from the local-ESS
    # one above so each can be zeroed independently by `_prepare_*`.
    model.penalty_shared_ess_usage = pe.Param(initialize=PENALTY_ESS_USAGE, mutable=True)
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


def load_is_tso_adn_interface(network, load):
    # P5.15 Step 3 (row 3 / D1, signed table `P5_15_S31_PENALTY_TABLE_DRAFT.md`):
    # identifies the TSO's ADN-interface loads exactly as
    # `flex_energy_balance_p_rule` does, so the two call sites cannot diverge.
    return network.is_transmission and load.bus in network.active_distribution_network_nodes


def flexibility_cost(model, network, s_m, s_o, params):
    flex_cost = 0.0
    if params.fl_reg:
        c_flex = network.cost_flex[s_m]
        for c in model.loads:
            load = network.loads[c]
            if load.fl_reg:
                # P5.15 Step 3 (row 3 / D1): the TSO's ADN-interface loads are
                # excluded from this charge -- it priced movement of the
                # interface away from a fixed build-time anchor, a double-counted
                # transfer payment (no DSO receipt term); removed per the signed
                # table (Addendum 10). The freeing of flex_* and the fixing of
                # pc/qc to the consensus value (shared_resources_planning.py
                # ~2963-2980) are unchanged.
                if load_is_tso_adn_interface(network, load):
                    continue
                for p in model.periods:
                    flex_cost += c_flex[p] * network.baseMVA * (
                            model.flex_p_down[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
                    )
    return flex_cost


def adn_interface_flexibility_cost(model, network, s_m, s_o, params):
    """P5.15 Step 3, row 3 / D1 -- definitional-decomposition helper only.

    Value the REMOVED TSO ADN-interface flexibility charge would have had at
    the current point, at the pre-signature weight `cost_flex`. NOT part of
    any solver objective (`flexibility_cost` above excludes these loads);
    used only by reporting/harness code (Part 3 of the S31 worker task).
    """
    cost = 0.0
    if params.fl_reg and network.is_transmission:
        c_flex = network.cost_flex[s_m]
        for c in model.loads:
            load = network.loads[c]
            if load.fl_reg and load_is_tso_adn_interface(network, load):
                for p in model.periods:
                    cost += c_flex[p] * network.baseMVA * (
                            model.flex_p_down[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
                    )
    return cost


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


def gen_curtailment_definitional_value(model, network, s_m, s_o, params, weight):
    """P5.15 Step 3, row 5 -- definitional-decomposition helper only.

    Same physical term as `gen_curtailment_penalty` (RES curtailment,
    `pg_avail - pg`), evaluated at an explicit `weight` instead of the model's
    (now zeroed) `penalty_gen_curtailment` Param -- e.g. the pre-signature DSO
    weight `PENALTY_GENERATION_CURTAILMENT`. NOT part of any solver objective;
    used only by reporting/harness code (Part 3 of the S31 worker task).
    """
    total = 0.0
    if params.rg_curt:
        for g in model.generators:
            if network.generators[g].is_curtaillable():
                for p in model.periods:
                    total += weight * network.baseMVA * (model.pg_avail[g, s_o, p] - model.pg[g, s_m, s_o, p])
    return total


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
            # P5.4-A: usage penalty follows ACTIVE charge/discharge power.
            # Coefficients are unchanged; this is part of the active-energy
            # physical correction, not an objective-tuning change.
            # P5.15 Step 3 (row 8, signed table): the shared-ESS usage weight is
            # now its own Param (`penalty_shared_ess_usage`), split from the
            # local-ESS weight (`penalty_ess_usage`) so that `_prepare_*` can
            # zero the shared term for ADMM without silently zeroing the local
            # one too (hygiene for future cases with `es_reg` active; inert on
            # SRP1 -- see `P5_15_S31_PENALTY_TABLE_DRAFT.md` row 8).
            cost += model.penalty_shared_ess_usage * network.baseMVA * (model.shared_es_pch[e, s_m, s_o, p] + model.shared_es_pdch[e, s_m, s_o, p])
    if params.es_reg:
        for e in model.energy_storages:
            for p in model.periods:
                cost += model.penalty_ess_usage * network.baseMVA * (model.es_pch[e, s_m, s_o, p] + model.es_pdch[e, s_m, s_o, p])
    return cost


def ess_utilization_cost_penalty_rule(model, s_m, s_o, network, params):
    return ess_utilization_cost_penalty(model, network, s_m, s_o, params)


def total_ess_utilization_cost_penalty_rule(model, network):
    total_ess_utilization_cost_penalty = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total_ess_utilization_cost_penalty += network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o] * model.ess_utilization_cost_penalty_scenario[s_m, s_o]
    return total_ess_utilization_cost_penalty


def voltage_slack_penalty(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 12, D-category): squared-voltage slacks. Split out of
    # `slack_penalties` so the terminal level can be reported per D-row
    # (`_get_operational_recourse_components`, Part 1 item 6) without
    # duplicating the term.
    total = 0
    if params.slacks.grid_operation.voltage:
        for i in model.nodes:
            for p in model.periods:
                total += PENALTY_VOLTAGE_SQUARED * (model.slack_v_sqr_down[i, s_m, s_o, p] + model.slack_v_sqr_up[i, s_m, s_o, p])
    return total


def node_balance_slack_penalty(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 15, D-category): node-balance slacks.
    total = 0
    base = network.baseMVA
    for i in model.nodes:
        for p in model.periods:
            if params.slacks.node_balance.active_power:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_p_up[i, s_m, s_o, p] + model.slack_node_balance_p_down[i, s_m, s_o, p])
            if params.slacks.node_balance.reactive_power:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_q_up[i, s_m, s_o, p] + model.slack_node_balance_q_down[i, s_m, s_o, p])
    return total


def branch_flow_slack_penalty(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 16, D-category): branch-flow slacks.
    total = 0
    base = network.baseMVA
    if params.slacks.grid_operation.branch_flow:
        for b in model.branches:
            for p in model.periods:
                total += base * PENALTY_CURRENT * (model.slack_flow_ij_sqr[b, s_m, s_o, p])
        for b in model.apparent_power_limited_branches:
            for p in model.periods:
                total += base * PENALTY_CURRENT * model.slack_flow_ji_sqr[b, s_m, s_o, p]
    return total


def flexibility_p_day_balance_slack_penalty(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 13, D-category): flexibility P day-balance slack, of
    # network-INTERNAL fl_reg loads only. The TSO's ADN-interface loads are
    # excluded here (row 14 / D2): their P day-balance constraint is skipped
    # (`flex_energy_balance_p_rule`), so the slack would be orphaned (penalized
    # with no governing constraint); those variables are fixed to 0 at
    # creation (`network.py`) instead of being penalized.
    total = 0
    base = network.baseMVA
    if params.fl_reg and params.slacks.flexibility.day_balance:
        for c in model.loads:
            load = network.loads[c]
            if load.fl_reg and not load_is_tso_adn_interface(network, load):
                # P5.15-1b Candidate 4: this branch is now reachable in
                # production (day_balance defaults True). `slack_flex_*[c, s_m,
                # s_o]` are scalar VarData (no period index); `sum(...)` on a
                # single non-iterable expression raised TypeError at build
                # time. Fixed to plain addition (P5.15-2 audit, "considered
                # and not shortlisted" table, latent-bug entry).
                total += base * PENALTY_FLEXIBILITY * (model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o])
    return total


def slack_penalties(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 14 / D2, signed table): the flexibility Q day-balance
    # slack penalty is REMOVED (its constraint, `flex_energy_balance_q`, is
    # unwired at every load -- `network.py` ~433 -- so the slack was orphaned:
    # penalized with no governing constraint). The P day-balance slacks of the
    # TSO's ADN-interface loads are excluded for the same reason (their P
    # balance constraint is also skipped). Both families of orphaned variables
    # are fixed to 0 at creation in `network.py` so they cannot float free in
    # the NLP.
    return (
        voltage_slack_penalty(model, network, s_m, s_o, params)
        + node_balance_slack_penalty(model, network, s_m, s_o, params)
        + branch_flow_slack_penalty(model, network, s_m, s_o, params)
        + flexibility_p_day_balance_slack_penalty(model, network, s_m, s_o, params)
    )


def slack_penalties_rule(model, s_m, s_o, network, params):
    return slack_penalties(model, network, s_m, s_o, params)


def local_ess_day_balance_slack_penalty(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 10, D-category): local-ESS day-balance slack.
    total = 0
    base = network.baseMVA
    if params.es_reg:
        for e in model.energy_storages:
            if params.slacks.ess.day_balance:
                total += base * PENALTY_ESS_BALANCE * (model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o])
    return total


def shared_ess_day_balance_slack_penalty(model, network, s_m, s_o, params):
    # P5.15 Step 3 (row 11, D-category): shared-ESS day-balance slack.
    total = 0
    base = network.baseMVA
    for e in model.shared_energy_storages:
        if params.slacks.shared_ess.day_balance:
            total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m, s_o] + model.slack_shared_es_soc_final_down[e, s_m, s_o])
    return total


def ess_complementarity_bilinear_value(model, network, s_m, s_o, params):
    """P5.15 Step 3, row 9 -- definitional-decomposition helper only.

    Value the REMOVED bilinear `pch * pdch` complementarity penalty (local and
    shared ESS) would have had at the current point, at the pre-signature
    weight `PENALTY_ESS_COMPLEMENTARITY`. NOT part of any solver objective;
    used only by reporting/harness code (Part 3 of the S31 worker task).
    """
    total = 0
    base = network.baseMVA
    if params.es_reg:
        for e in model.energy_storages:
            for p in model.periods:
                if params.ess_model == ESS_MODEL_BILINEAR_RELAXATION:
                    total += base * PENALTY_ESS_COMPLEMENTARITY * (model.es_pch[e, s_m, s_o, p] * model.es_pdch[e, s_m, s_o, p])
    for e in model.shared_energy_storages:
        for p in model.periods:
            if params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:
                total += base * PENALTY_ESS_COMPLEMENTARITY * (model.shared_es_pch[e, s_m, s_o, p] * model.shared_es_pdch[e, s_m, s_o, p])
    return total


def ess_complementarity_penalties(model, network, s_m, s_o, p, params):
    # P5.15 Step 3 (row 9, signed table): the bilinear `pch * pdch`
    # complementarity penalty (local and shared ESS) is REMOVED from the
    # objective -- the hard relaxed complementarity constraint (`ess_comp` /
    # `sess_comp`, network.py) already enforces feasibility; the penalty was
    # indefinite and redundant (Addendum 10). The day-balance slack terms
    # (rows 10, 11) are KEPT, factored into the two helpers above so they can
    # be reported separately from complementarity (Part 1 item 6 / Part 3
    # split 1). The network's hard complementarity constraints are unchanged
    # by this function.
    return (
        local_ess_day_balance_slack_penalty(model, network, s_m, s_o, params)
        + shared_ess_day_balance_slack_penalty(model, network, s_m, s_o, params)
    )


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
