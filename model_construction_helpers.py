import pyomo.environ as pe
from definitions import *

# Voltage variables, e
def e_bounds(model, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == "BUS_REF" and not network.is_transmission:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg
        return (vg - params.EQUALITY_TOLERANCE, vg + params.EQUALITY_TOLERANCE)
    return (-node.v_max, node.v_max)


# Voltage variables, f
def f_bounds(model, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == "BUS_REF":
        return (-params.EQUALITY_TOLERANCE, params.EQUALITY_TOLERANCE)
    return (-node.v_max, node.v_max)

# Voltage variables, slack bounds
def slack_bounds(model, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    if node.type == "BUS_REF":
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)
    return (-VMAG_VIOLATION_ALLOWED, VMAG_VIOLATION_ALLOWED)


# Generation, Pg
def pg_bounds(model, g, s_m, s_o, p, network, params):
    gen = network.generators[g]
    if gen.status[p]:
        return (max(gen.pmin, 0.0), gen.pmax)
    else:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)

# Generation, Qg
def qg_bounds(model, g, s_m, s_o, p, network, params):
    gen = network.generators[g]
    if gen.status[p]:
        return (max(gen.qmin, 0.0), gen.qmax)
    else:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)


# Generation, Sg
def sg_bounds(model, g, s_m, s_o, p, network, params):

    gen = network.generators[g]
    if not gen.is_curtaillable():
        return (0.0, EQUALITY_TOLERANCE)
    if not gen.status[p]:
        return (0.0, EQUALITY_TOLERANCE)

    # Estimated max apparent power for initialization and bound
    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg = (pg ** 2 + qg ** 2) ** 0.5

    return (0.0, sg)


# Generation, Sg^2
def sg_sqr_bounds(model, g, s_m, s_o, p, network, params):

    gen = network.generators[g]
    if not gen.is_curtaillable() or not gen.status[p]:
        return (0.0, EQUALITY_TOLERANCE)

    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg_sqr = pg**2 + qg**2

    return (0.0, sg_sqr)


# Branch power flow, Fij
def flow_ij_sqr_bounds(model, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, EQUALITY_TOLERANCE)
    return (0.0, None)  # No upper bound unless explicitly constrained elsewhere


def init_flow_ij_sqr(model, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return 0.0
    # use some nominal flow or historical data if available
    return 0.01  # placeholder


# Branch power flow, Fij slacks
def slack_flow_bounds(model, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, EQUALITY_TOLERANCE)
    rating = branch.rate / network.baseMVA
    return (0.0, SIJ_VIOLATION_ALLOWED * rating)


# Consumption, Pc
def pc_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    return (pd - EQUALITY_TOLERANCE, pd + EQUALITY_TOLERANCE)


# Consumption, Qc
def qc_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    return (qd - EQUALITY_TOLERANCE, qd + EQUALITY_TOLERANCE)


# Consumption, flexibility
def pc_flex_up_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.upward[p])
    return (0.0, value)


def pc_flex_down_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.downward[p])
    return (0.0, value)


def qc_flex_up_bounds(model, c, s_m, s_o, p, network, params):
    return (0.0, EQUALITY_TOLERANCE)


def qc_flex_down_bounds(model, c, s_m, s_o, p, network, params):
    return (0.0, EQUALITY_TOLERANCE)


# Consumption, curtailment
def pc_curt_down_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, abs(pd))
    else:
        return (0.0, EQUALITY_TOLERANCE)


def pc_curt_up_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, EQUALITY_TOLERANCE)
    else:
        return (0.0, abs(pd))


def qc_curt_down_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    if qd >= 0.00:
        return (0.0, abs(qd))
    else:
        return (0.0, EQUALITY_TOLERANCE)


def qc_curt_up_bounds(model, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.pd[s_o][p]
    if qd >= 0.00:
        return (0.0, EQUALITY_TOLERANCE)
    else:
        return (0.0, abs(qd))

# Transformers
def transformer_ratio_bounds(model, i, s_m, s_o, p, network, params):
    branch = network.branches[i]
    if branch.is_transformer:
        if params.transf_reg and branch.vmag_reg:
            return (TRANSFORMER_MINIMUM_RATIO, TRANSFORMER_MAXIMUM_RATIO)
        else:
            return (branch.ratio - EQUALITY_TOLERANCE, branch.ratio + EQUALITY_TOLERANCE)
    else:
        return (1.00 - EQUALITY_TOLERANCE, 1.00 + EQUALITY_TOLERANCE)


# Energy Storage
def soc_bounds(e, network):
    es = network.energy_storages[e]
    return (es.e_min, es.e_max)


def q_bounds(e, network):
    es = network.energy_storages[e]
    return (-es.s, es.s)


def s_bounds(e, network):
    return (0.0, network.energy_storages[e].s)


def soc_initialize(model, e, network):
    return network.energy_storages[e].e_init


# Shared Energy Storage
def shared_soc_bounds(model, e, s_m, s_o, p, network):
    ses = network.shared_energy_storages[e]
    return (0.0, ses.e)


def shared_q_bounds(model, e, s_m, s_o, p, network):
    s = network.shared_energy_storages[e].s
    return (-s, s)


def shared_s_bounds(model, e, s_m, s_o, p, network):
    return (0.0, network.shared_energy_storages[e].s)


def shared_soc_init(model, e, s_m, s_o, p, network):
    return network.shared_energy_storages[e].e * ENERGY_STORAGE_RELATIVE_INIT_SOC



# Voltage constraints, e
def voltage_rule_e(model, i, s_m, s_o, p, network, params):
    e_val = model.e[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        e_val += model.slack_e[i, s_m, s_o, p]
    return model.e_actual[i, s_m, s_o, p] == e_val


# Voltage constraints, f
def voltage_rule_f(model, i, s_m, s_o, p, network, params):
    f_val = model.f[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        f_val += model.slack_f[i, s_m, s_o, p]
    return model.f_actual[i, s_m, s_o, p] == f_val


# Voltage constraints, magnitude
def voltage_magnitude_rule(model, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    e = model.e[i, s_m, s_o, p]
    f = model.f[i, s_m, s_o, p]
    vmag_sq = e ** 2 + f ** 2
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return pe.inequality(vg ** 2 - EQUALITY_TOLERANCE, vmag_sq, vg ** 2 + EQUALITY_TOLERANCE)
    else:
        return pe.inequality(node.v_min ** 2, vmag_sq, node.v_max ** 2)



