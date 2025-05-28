import pyomo.environ as pe
from math import sqrt, tan, atan2, acos
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

# Generation, Sg^2
# Apparent power ≈ pg² + qg²
def sg_sqr_upper_bound_rule(m, g, s_m, s_o, p):
    return m.sg_sqr[g, s_m, s_o, p] <= m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2 + EQUALITY_TOLERANCE


def sg_sqr_lower_bound_rule(m, g, s_m, s_o, p):
    return m.sg_sqr[g, s_m, s_o, p] >= m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2 - EQUALITY_TOLERANCE


# sg_abs² ≈ sg_sqr
def sg_abs_upper_bound_rule(m, g, s_m, s_o, p):
    return m.sg_abs[g, s_m, s_o, p]**2 <= m.sg_sqr[g, s_m, s_o, p] + EQUALITY_TOLERANCE


def sg_abs_lower_bound_rule(m, g, s_m, s_o, p):
    return m.sg_abs[g, s_m, s_o, p]**2 >= m.sg_sqr[g, s_m, s_o, p] - EQUALITY_TOLERANCE


# Curtailment: sg_abs = init_sg - sg_curt
def sg_curtailment_upper_rule(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.status[p]:
        init_sg = sqrt(gen.pg[s_o][p]**2 + gen.qg[s_o][p]**2)
    else:
        init_sg = 0.0
    return m.sg_abs[g, s_m, s_o, p] <= init_sg - m.sg_curt[g, s_m, s_o, p] + EQUALITY_TOLERANCE


def sg_curtailment_lower_rule(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.status[p]:
        init_sg = sqrt(gen.pg[s_o][p]**2 + gen.qg[s_o][p]**2)
    else:
        init_sg = 0.0
    return m.sg_abs[g, s_m, s_o, p] >= init_sg - m.sg_curt[g, s_m, s_o, p] - EQUALITY_TOLERANCE


def power_factor_rule_upper(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]

    if gen.power_factor_control:
        phi = acos(gen.max_pf)
    else:
        phi = atan2(gen.qg[s_o][p], gen.pg[s_o][p])

    return qg <= tan(phi) * pg


def power_factor_rule_lower(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]

    if gen.power_factor_control:
        phi = acos(gen.min_pf)
    else:
        phi = atan2(gen.qg[s_o][p], gen.pg[s_o][p])

    return qg >= tan(phi) * pg


# Flexible loads
def flex_energy_balance_rule(model, c, s_m, s_o, p, params):
    p_up = sum(model.flex_p_up[c, s_m, s_o, p] for p in model.periods)
    p_down = sum(model.flex_p_down[c, s_m, s_o, p] for p in model.periods)
    if params.slacks.flexibility.day_balance:
        return p_up == p_down + model.slack_flex_p_balance[c, s_m, s_o]
    else:
        # Soft equality with tolerance
        return pe.inequality(p_down - EQUALITY_TOLERANCE, p_up, p_down + EQUALITY_TOLERANCE)


# Energy Storage
def ess_phi_ch_limits(m, e, s_m, s_o, p, network):
    es = network.energy_storages[e]
    max_phi = acos(es.max_pf)
    min_phi = acos(es.min_pf)
    ineq = pe.inequality(
        tan(min_phi) * m.es_pch[e, s_m, s_o, p],
        m.es_qch[e, s_m, s_o, p],
        tan(max_phi) * m.es_pch[e, s_m, s_o, p]
    )
    return ineq


def ess_phi_dch_limits(m, e, s_m, s_o, p, network):
    es = network.energy_storages[e]
    max_phi = acos(es.max_pf)
    min_phi = acos(es.min_pf)
    ineq = pe.inequality(
        tan(min_phi) * m.es_pdch[e, s_m, s_o, p],
        m.es_qdch[e, s_m, s_o, p],
        tan(max_phi) * m.es_pdch[e, s_m, s_o, p]
    )
    return ineq


def ess_comp_rule(m, e, s_m, s_o, p, params):
    if params.slacks.ess.complementarity:
        return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] == m.slack_es_comp[e, s_m, s_o, p]
    else:
        return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE


def ess_balance_rule(m, e, s_m, s_o, p, network):
    es = network.energy_storages[e]
    eff_ch, eff_dch = es.eff_ch, es.eff_dch
    soc_prev = es.e_init if p == 0 else m.es_soc[e, s_m, s_o, p - 1]
    ineq = pe.inequality(
        soc_prev + (m.es_sch[e, s_m, s_o, p] * eff_ch - m.es_sdch[e, s_m, s_o, p] / eff_dch) - EQUALITY_TOLERANCE,
        m.es_soc[e, s_m, s_o, p],
        soc_prev + (m.es_sch[e, s_m, s_o, p] * eff_ch - m.es_sdch[e, s_m, s_o, p] / eff_dch) + EQUALITY_TOLERANCE
    )
    return ineq


def ess_soc_final_rule(model, e, s_m, s_o, network, params):
    final_soc = network.energy_storages[e].e_init
    final_p = model.periods.last()
    if params.slacks.ess.day_balance:
        return model.es_soc[e, s_m, s_o, final_p] == final_soc + model.slack_es_soc_final[e, s_m, s_o]
    else:
        ineq = pe.inequality(
            final_soc - EQUALITY_TOLERANCE,
            model.es_soc[e, s_m, s_o, final_p],
            final_soc + EQUALITY_TOLERANCE
        )
        return ineq

