from copy import copy
from functools import partial
from math import tan, atan2, acos
from helper_functions import *
from definitions import *


# Voltage variables, e
def e_bounds(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF and not network.is_transmission:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg
        return (vg - EQUALITY_TOLERANCE, vg + EQUALITY_TOLERANCE)
    return (-node.v_max, node.v_max)


# Voltage variables, f
def f_bounds(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)
    return (-node.v_max, node.v_max)


# Voltage variables, slack bounds
def voltage_slack_bounds(m, i, s_m, s_o, p, network):
    node = network.nodes[i]
    if node.type == BUS_REF:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)
    return (-VMAG_VIOLATION_ALLOWED, VMAG_VIOLATION_ALLOWED)


# Generation, Pg
def pg_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.status[p]:
        return (gen.pmin, gen.pmax)
    else:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)


# Generation, Qg
def qg_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.status[p]:
        return (gen.qmin, gen.qmax)
    else:
        return (-EQUALITY_TOLERANCE, EQUALITY_TOLERANCE)


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
    if not gen.is_curtaillable():
        return 0.00
    if not gen.status[p]:
        return 0.00

    # Apparent power for initialization and bound
    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg = (pg ** 2 + qg ** 2) ** 0.5

    return abs(sg)


def sg_bounds(m, g, s_m, s_o, p, network, params):

    gen = network.generators[g]
    if not gen.is_curtaillable():
        return (0.0, EQUALITY_TOLERANCE)
    if not gen.status[p]:
        return (0.0, EQUALITY_TOLERANCE)

    # Estimated apparent power for bounds
    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg = (pg ** 2 + qg ** 2) ** 0.5

    return (0.0, sg)


def sg_slack_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.is_curtaillable() and gen.status[p]:
        sg = (gen.qmin ** 2 + gen.qmax ** 2) ** 0.5
        return (0.00, sg * 0.05)
    return (0.00, EQUALITY_TOLERANCE)


def sg_sqr_slack_bounds(m, g, s_m, s_o, p, network):
    gen = network.generators[g]
    if gen.is_curtaillable() and gen.status[p]:
        sg_sqr = (gen.qmin ** 2 + gen.qmax ** 2)
        return (0.00, sg_sqr * 0.05)
    return (0.00, EQUALITY_TOLERANCE)


# Generation, Sg^2
def sg_sqr_bounds(m, g, s_m, s_o, p, network, params):

    gen = network.generators[g]
    if not gen.is_curtaillable() or not gen.status[p]:
        return (0.0, EQUALITY_TOLERANCE)

    pg = gen.pg[s_o][p]
    qg = gen.qg[s_o][p]
    sg_sqr = pg**2 + qg**2

    return (0.0, sg_sqr)


# Branch power flow, Fij
def flow_ij_sqr_bounds(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, EQUALITY_TOLERANCE)
    rating_sqr = (branch.rate / network.baseMVA)**2
    return (0.0, rating_sqr)


def init_flow_ij_sqr(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return 0.0
    return SMALL_TOLERANCE**2


# Branch power flow, Fij slacks
def slack_flow_bounds(m, b, s_m, s_o, p, network, params):
    branch = network.branches[b]
    if not branch.status:
        return (0.0, EQUALITY_TOLERANCE)
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


# Consumption, flexibility
def pc_flex_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.upward[p])
    return (0.0, value)


def pc_flex_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    if not load.fl_reg:
        return (0.0, EQUALITY_TOLERANCE)
    value = abs(load.flexibility.downward[p])
    return (0.0, value)


def qc_flex_up_bounds(m, c, s_m, s_o, p, network, params):
    return (0.0, EQUALITY_TOLERANCE)


def qc_flex_down_bounds(m, c, s_m, s_o, p, network, params):
    return (0.0, EQUALITY_TOLERANCE)


# Consumption, curtailment
def pc_curt_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, abs(pd))
    else:
        return (0.0, EQUALITY_TOLERANCE)


def pc_curt_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    if pd >= 0.00:
        return (0.0, EQUALITY_TOLERANCE)
    else:
        return (0.0, abs(pd))


def qc_curt_down_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.qd[s_o][p]
    if qd >= 0.00:
        return (0.0, abs(qd))
    else:
        return (0.0, EQUALITY_TOLERANCE)


def qc_curt_up_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    qd = load.pd[s_o][p]
    if qd >= 0.00:
        return (0.0, EQUALITY_TOLERANCE)
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


# Energy Storage
def soc_bounds(m, e, s_m, s_o, p, network):
    es = network.energy_storages[e]
    return (es.e_min, es.e_max)


def p_bounds(m, e, s_m, s_o, p, network):
    return (0.0, network.energy_storages[e].s)


def snet_bounds(m, e, s_m, s_o, p, network):
    return (-network.energy_storages[e].s, network.energy_storages[e].s)


def q_bounds(m, e, s_m, s_o, p, network):
    es = network.energy_storages[e]
    return (-es.s, es.s)


def s_bounds(m, e, s_m, s_o, p, network):
    return (0.0, network.energy_storages[e].s)


def slack_es_comp_bounds(m, e, s_m, s_o, p, network):
    return (0.0, network.energy_storages[e].s * 0.05)


def slack_es_balance_bounds(m, e, s_m, s_o, p, network):
    return (-network.energy_storages[e].e * 0.01, network.energy_storages[e].e * 0.01)


def soc_initialize(m, e, s_m, s_o, p, network):
    return network.energy_storages[e].e_init


# Shared Energy Storage
def shared_soc_bounds(m, e, s_m, s_o, p, network):
    ses = network.shared_energy_storages[e]
    return (0.0, ses.e)


def shared_q_bounds(m, e, s_m, s_o, p, network):
    s = network.shared_energy_storages[e].s
    return (-s, s)


def shared_s_bounds(m, e, s_m, s_o, p, network):
    return (0.0, network.shared_energy_storages[e].s)


def shared_soc_init(m, e, s_m, s_o, p, network):
    return network.shared_energy_storages[e].e * ENERGY_STORAGE_RELATIVE_INIT_SOC


# Voltage constraints, e
def voltage_rule_e(m, i, s_m, s_o, p, params):
    e_val = m.e[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        e_val += m.slack_e[i, s_m, s_o, p]
    return m.e_actual[i, s_m, s_o, p] == e_val


# Voltage constraints, f
def voltage_rule_f(m, i, s_m, s_o, p, params):
    f_val = m.f[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        f_val += m.slack_f[i, s_m, s_o, p]
    return m.f_actual[i, s_m, s_o, p] == f_val


# Voltage constraints, magnitude
def voltage_magnitude_def_rule(m, i, s_m, s_o, p):
    e = m.e[i, s_m, s_o, p]
    f = m.f[i, s_m, s_o, p]
    vmag_sq = e ** 2 + f ** 2
    return pe.inequality(-EQUALITY_TOLERANCE, m.vmag_sqr[i, s_m, s_o, p] - vmag_sq, EQUALITY_TOLERANCE)


# Voltage constraints, magnitude
def voltage_magnitude_cons_rule(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    vmag_sq = m.vmag_sqr[i, s_m, s_o, p]
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return pe.inequality(-EQUALITY_TOLERANCE, vmag_sq - vg ** 2, EQUALITY_TOLERANCE)
    else:
        return pe.inequality(node.v_min ** 2, vmag_sq, node.v_max ** 2)

# Generation, Sg^2
# Apparent power ≈ pg² + qg²
def sg_sqr_rule(m, g, s_m, s_o, p, network, params):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, m.sg_sqr[g, s_m, s_o, p] - (m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


# sg_abs² ≈ sg_sqr
def sg_abs_rule(m, g, s_m, s_o, p, network, params):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return  pe.inequality(-EQUALITY_TOLERANCE, m.sg_abs[g, s_m, s_o, p]**2 - m.sg_sqr[g, s_m, s_o, p], EQUALITY_TOLERANCE)


# Curtailment: sg_abs = init_sg - sg_curt
def sg_curtailment_rule(m, g, s_m, s_o, p, network, params):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, m.sg_abs[g, s_m, s_o, p] - (m.sg_init[g, s_m, s_o, p] - m.sg_curt[g, s_m, s_o, p]), EQUALITY_TOLERANCE)


def power_factor_rule_upper(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
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
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    pg = m.pg[g, s_m, s_o, p]
    qg = m.qg[g, s_m, s_o, p]
    if generator.power_factor_control:
        phi = acos(generator.min_pf)
    else:
        phi = atan2(generator.qg[s_o][p], generator.pg[s_o][p])
    return qg >= tan(phi) * pg


# Flexible loads
def flex_energy_balance_rule(m, c, s_m, s_o, network, params):
    load = network.loads[c]
    if load.fl_reg:
        if network.is_transmission and load.bus in network.active_distribution_network_nodes:
            return pe.Constraint.Skip
        p_up = sum(m.flex_p_up[c, s_m, s_o, p] for p in m.periods)
        p_down = sum(m.flex_p_down[c, s_m, s_o, p] for p in m.periods)
        if params.slacks.flexibility.day_balance:
            return p_up == p_down + m.slack_flex_p_balance[c, s_m, s_o]
        else:
            return pe.inequality(-EQUALITY_TOLERANCE, p_up == p_down, EQUALITY_TOLERANCE)
    else:
        return pe.Constraint.Skip


# Energy Storage
def ess_sch_def(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.es_sch[e, s_m, s_o, p]**2 - (m.es_pch[e, s_m, s_o, p]**2 + m.es_qch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def ess_sdch_def(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.es_sdch[e, s_m, s_o, p]**2 - (m.es_pdch[e, s_m, s_o, p]**2 + m.es_qdch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


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
        return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] == m.slack_es_comp[e, s_m, s_o, p]
    else:
        if params.ess_model == ESS_MODEL_EXACT:
            return m.es_sch[e, s_m, s_o, p] * m.es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE
        elif params.ess_model in [ESS_MODEL_LP_SIMPLIFIED, ESS_MODEL_LP_RELAXED, ESS_MODEL_LP_SIMPLIFIED_EXTENDED]:
            return pe.Constraint.Skip
        else:
            print('[ERROR] Invalid ESS model. Exiting...')
            exit(ERROR_PARAMS_FILE)


def ess_balance_rule(m, e, s_m, s_o, p, network):
    es = network.energy_storages[e]
    eff_ch, eff_dch = es.eff_ch, es.eff_dch
    soc_prev = es.e_init if p == 0 else m.es_soc[e, s_m, s_o, p - 1]
    return pe.inequality(-EQUALITY_TOLERANCE, m.es_soc[e, s_m, s_o, p] - (soc_prev + m.es_sch[e, s_m, s_o, p] * eff_ch - m.es_sdch[e, s_m, s_o, p] / eff_dch), EQUALITY_TOLERANCE)


def ess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = network.energy_storages[e].e_init
    final_p = m.periods[-1]
    if params.slacks.ess.day_balance:
        return m.es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_es_soc_final[e, s_m, s_o]
    else:
        return pe.inequality(-SMALL_TOLERANCE, m.es_soc[e, s_m, s_o, final_p] - final_soc, SMALL_TOLERANCE)


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


def sess_sch_def(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_sch[e, s_m, s_o, p]**2 - (m.shared_es_pch[e, s_m, s_o, p]**2 + m.shared_es_qch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def sess_sdch_def(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_sdch[e, s_m, s_o, p]**2 - (m.shared_es_pdch[e, s_m, s_o, p]**2 + m.shared_es_qdch[e, s_m, s_o, p]**2), EQUALITY_TOLERANCE)


def sess_pch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    pch = m.shared_es_pch[e, s_m, s_o, p]
    return pch <= s_max


def sess_pdch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return pdch <= s_max


def sess_soc_lower_limit(m, e, s_m, s_o, p):
    soc_min = m.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] >= soc_min


def sess_soc_upper_limit(m, e, s_m, s_o, p):
    soc_max = m.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
    return m.shared_es_soc[e, s_m, s_o, p] <= soc_max


def sess_comp_rule(m, e, s_m, s_o, p, params):
    if params.slacks.shared_ess.complementarity:
        return m.shared_es_sch[e, s_m, s_o, p] * m.shared_es_sdch[e, s_m, s_o, p] == m.slack_shared_es_comp[e, s_m, s_o, p]
    else:
        if params.shared_ess_model == ESS_MODEL_EXACT:
            return m.shared_es_sch[e, s_m, s_o, p] * m.shared_es_sdch[e, s_m, s_o, p] <= EQUALITY_TOLERANCE
        elif params.shared_ess_model in [ESS_MODEL_LP_SIMPLIFIED, ESS_MODEL_LP_RELAXED, ESS_MODEL_LP_SIMPLIFIED_EXTENDED]:
            return pe.Constraint.Skip
        else:
            print('[ERROR] Invalid ESS model. Exiting...')
            exit(ERROR_PARAMS_FILE)


def sess_balance_rule(m, e, s_m, s_o, p, network):
    ses = network.shared_energy_storages[e]
    eff_ch, eff_dch = ses.eff_ch, ses.eff_dch
    soc_prev = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC if p == 0 else m.shared_es_soc[e, s_m, s_o, p - 1]
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_soc[e, s_m, s_o, p] - (soc_prev + m.shared_es_sch[e, s_m, s_o, p] * eff_ch - m.shared_es_sdch[e, s_m, s_o, p] / eff_dch), EQUALITY_TOLERANCE)


def sess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = m.shared_es_e_rated[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
    final_p = m.periods[-1]
    if params.slacks.ess.day_balance:
        return m.shared_es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_shared_es_soc_final[e, s_m, s_o]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_soc[e, s_m, s_o, final_p] - final_soc, EQUALITY_TOLERANCE)


def sess_pnet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_pnet[e, s_m, s_o, p] - (m.shared_es_pch[e, s_m, s_o, p] - m.shared_es_pdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


def sess_qnet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_qnet[e, s_m, s_o, p] - (m.shared_es_qch[e, s_m, s_o, p] - m.shared_es_qdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


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


# Branch limits
def compute_branch_flow_squared(branch, ei, fi, ej, fj, rij, limit_type):
    """
    Computes the squared branch flow expression depending on the limit type:
    - current (I²)
    - apparent power (S² = P² + Q²)
    - mixed (based on whether branch is a transformer)
    All inputs should be Pyomo expressions or variables.

    Parameters:
    - branch: an object with electrical parameters (g, b, b_sh, is_transformer, etc.)
    - ei, fi: real and imaginary voltage components at sending node
    - ej, fj: real and imaginary voltage components at receiving node
    - rij: tap ratio (symbolic for transformer, 1.0 otherwise)
    - limit_type: one of 'current', 'apparent', or 'mixed'

    Returns:
    - A Pyomo expression representing the squared flow
    """
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
        return current_squared

    raise ValueError(f"Unknown branch limit type: {limit_type}")


# Objective function
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
    if isinstance(Pd, float):
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, model.pc_node[i, s_m, s_o, p] - Pd, EQUALITY_TOLERANCE)


def net_load_q_per_node_rule(model, i, s_m, s_o, p, network, params):
    _, Qd = compute_node_load(model, i, s_m, s_o, p, network, params)
    if isinstance(Qd, float):
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, model.qc_node[i, s_m, s_o, p] - Qd, EQUALITY_TOLERANCE)


def net_gen_p_per_node_rule(model, i, s_m, s_o, p, network):
    Pg, _ = compute_node_gen(model, i, s_m, s_o, p, network)
    if isinstance(Pg, float):
        return pe.Constraint.Skip
    return pe.inequality(-EQUALITY_TOLERANCE, model.pg_node[i, s_m, s_o, p] - Pg, EQUALITY_TOLERANCE)


def net_gen_q_per_node_rule(model, i, s_m, s_o, p, network):
    _, Qg = compute_node_gen(model, i, s_m, s_o, p, network)
    if isinstance(Qg, float):
        return pe.Constraint.Skip
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

    if params.slacks.node_balance:
        return Pg == Pd + Pi + model.slack_node_balance_p[i, s_m, s_o, p]
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

    if params.slacks.node_balance:
        return Qg == Qd + Qi + model.slack_node_balance_q[i, s_m, s_o, p]
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

    return pe.inequality(-EQUALITY_TOLERANCE, model.flow_ij_sqr[b, s_m, s_o, p] - flow_ij_sqr_expr, EQUALITY_TOLERANCE)


def branch_flow_limit_rule(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Constraint.Skip

    rating = branch.rate / network.baseMVA or BRANCH_UNKNOWN_RATING
    flow_var = model.flow_ij_sqr[b, s_m, s_o, p]

    if params.slacks.grid_operation.branch_flow:
        slack = model.slack_flow_ij_sqr[b, s_m, s_o, p]
        return flow_var - slack <= rating ** 2
    else:
        return flow_var <= rating ** 2


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
    if not params.rg_curt:
        cost = model.cost_res_curtailment
        return sum(
            cost * network.baseMVA * model.sg_curt[g, s_m, s_o, p]
            for g in model.generators if network.generators[g].is_curtaillable()
            for p in model.periods
        )
    return 0.00


def gen_curtailment_penalty(model, network, s_m, s_o, params):
    if not params.rg_curt:
        penalty = model.penalty_gen_curtailment
        return sum(
            penalty * network.baseMVA * model.sg_curt[g, s_m, s_o, p]
            for g in model.generators
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
    cost = model.penalty_ess_usage
    return sum(
        cost * network.baseMVA * (
            model.es_sch[e, s_m, s_o, p] + model.es_sdch[e, s_m, s_o, p]
        )
        for e in model.energy_storages
        for p in model.periods
    ) + sum(
        cost * network.baseMVA * (
            model.shared_es_sch[e, s_m, s_o, p] + model.shared_es_sdch[e, s_m, s_o, p]
        )
        for e in model.shared_energy_storages
        for p in model.periods
    )


def slack_penalties(model, network, s_m, s_o, params):

    total = 0
    base = network.baseMVA

    for i in model.nodes:
        for p in model.periods:
            if params.slacks.grid_operation.voltage:
                total += base * PENALTY_VOLTAGE * (model.slack_e[i, s_m, s_o, p]**2 + model.slack_f[i, s_m, s_o, p]**2)
            if params.slacks.node_balance:
                total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_p[i, s_m, s_o, p]**2 + model.slack_node_balance_q[i, s_m, s_o, p]**2)

    if params.fl_reg and params.slacks.flexibility.day_balance:
        total += base * PENALTY_FLEXIBILITY * sum(
            model.slack_flex_p_balance[c, s_m, s_o]**2 for c in model.loads
        )

    if params.es_reg:
        for e in model.energy_storages:
            for p in model.periods:
                if params.slacks.ess.complementarity:
                    total += base * PENALTY_ESS * model.slack_es_comp[e, s_m, s_o, p]
            if params.slacks.ess.day_balance:
                total += base * PENALTY_ESS * model.slack_es_soc_final[e, s_m, s_o]**2

    for e in model.shared_energy_storages:
        for p in model.periods:
            if params.slacks.shared_ess.complementarity:
                total += base * PENALTY_SHARED_ESS * model.slack_shared_es_comp[e, s_m, s_o, p]
        if params.slacks.shared_ess.day_balance:
            total += base * PENALTY_SHARED_ESS * model.slack_shared_es_soc_final[e, s_m, s_o]**2

    for b in model.branches:
        for p in model.periods:
            if params.slacks.grid_operation.branch_flow:
                total += base * PENALTY_CURRENT * model.slack_flow_ij_sqr[b, s_m, s_o, p]

    return total


# ADMM Models

# - TSO
def define_tso_interface_variables(model):
    model.expected_interface_vmag_sqr = pe.Var(model.active_distribution_networks, model.periods, domain=pe.NonNegativeReals, initialize=1.0)
    model.expected_interface_pf_p = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=0.0)
    model.expected_interface_pf_q = pe.Var(model.active_distribution_networks, model.periods, domain=pe.Reals, initialize=0.0)
    model.expected_shared_ess_p = pe.Var(model.shared_energy_storages, model.periods, domain=pe.Reals, initialize=0.0)
    model.expected_shared_ess_q = pe.Var(model.shared_energy_storages, model.periods, domain=pe.Reals, initialize=0.0)


def define_tso_expected_value_constraints(model, network):
    model.interface_expected_values = pe.ConstraintList()
    for dn in model.active_distribution_networks:
        adn_id = network.active_distribution_network_nodes[dn]
        adn_idx = network.get_node_idx(adn_id)
        for p in model.periods:
            evs = ep_p = ep_q = 0.0
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    weight = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                    evs += weight * model.vmag_sqr[adn_idx, s_m, s_o, p]
                    ep_p += weight * model.pc_node[adn_idx, s_m, s_o, p]
                    ep_q += weight * model.qc_node[adn_idx, s_m, s_o, p]
            model.interface_expected_values.add(model.expected_interface_vmag_sqr[dn, p] == evs)
            model.interface_expected_values.add(model.expected_interface_pf_p[dn, p] == ep_p)
            model.interface_expected_values.add(model.expected_interface_pf_q[dn, p] == ep_q)
    for e in model.shared_energy_storages:
        for p in model.periods:
            ess_p = ess_q = 0.0
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    weight = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                    ess_p += weight * model.shared_es_pnet[e, s_m, s_o, p]
                    ess_q += weight * model.shared_es_qnet[e, s_m, s_o, p]
            model.interface_expected_values.add(model.expected_shared_ess_p[e, p] == ess_p)
            model.interface_expected_values.add(model.expected_shared_ess_q[e, p] == ess_q)


def add_regularization_to_tso_objective(model, network):
    s_base = network.baseMVA
    penalty = pe.Param(initialize=PENALTY_REGULARIZATION, mutable=True)
    model.penalty_regularization = penalty
    expr = copy(model.objective.expr)
    for dn in model.active_distribution_networks:
        adn_id = network.active_distribution_network_nodes[dn]
        adn_idx = network.get_node_idx(adn_id)
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    expr += penalty * (
                            (model.vmag_sqr[adn_idx, s_m, s_o, p] - model.expected_interface_vmag_sqr[dn, p]) ** 2 +
                            s_base * (model.pc_node[adn_idx, s_m, s_o, p] - model.expected_interface_pf_p[dn, p]) ** 2 +
                            s_base * (model.qc_node[adn_idx, s_m, s_o, p] - model.expected_interface_pf_q[dn, p]) ** 2
                    )
    for e in model.shared_energy_storages:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    expr += penalty * s_base * (
                            (model.shared_es_pnet[e, s_m, s_o, p] - model.expected_shared_ess_p[e, p]) ** 2 +
                            (model.shared_es_qnet[e, s_m, s_o, p] - model.expected_shared_ess_q[e, p]) ** 2
                    )
    model.regularization = pe.Expression(expr=expr)
    model.objective.expr += model.regularization

# - DSO
def define_dso_interface_variables(model):
    model.expected_interface_vmag_sqr = pe.Var(model.periods, domain=pe.NonNegativeReals, initialize=1.0)
    model.expected_interface_pf_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
    model.expected_interface_pf_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
    model.expected_shared_ess_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)
    model.expected_shared_ess_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.0)


def define_dso_expected_value_constraints(model, network, ref_node_idx, shared_ess_idx):
    model.interface_expected_vmag_values = pe.Constraint(model.periods, rule=partial(dso_interface_expected_vmag_rule, net=network, ref_node_idx=ref_node_idx))
    model.interface_expected_p_values = pe.Constraint(model.periods, rule=partial(dso_interface_expected_p_rule, net=network, ref_node_idx=ref_node_idx))
    model.interface_expected_q_values = pe.Constraint(model.periods, rule=partial(dso_interface_expected_q_rule, net=network, ref_node_idx=ref_node_idx))
    model.shared_ess_expected_p_values = pe.Constraint(model.periods, rule=partial(dso_shared_expected_p_rule, net=network, shared_ess_idx=shared_ess_idx))
    model.shared_ess_expected_q_values = pe.Constraint(model.periods, rule=partial(dso_shared_expected_q_rule, net=network, shared_ess_idx=shared_ess_idx))


def add_regularization_to_dso_objective(model, network, ref_node_idx, ess_idx):
    s_base = network.baseMVA
    penalty = pe.Param(initialize=PENALTY_REGULARIZATION, mutable=True)
    model.penalty_regularization = penalty
    expr = sum(
        penalty * (
            (model.vmag_sqr[ref_node_idx, s_m, s_o, p] - model.expected_interface_vmag_sqr[p]) ** 2 +
            s_base * (model.pg_node[ref_node_idx, s_m, s_o, p] - model.expected_interface_pf_p[p]) ** 2 +
            s_base * (model.qg_node[ref_node_idx, s_m, s_o, p] - model.expected_interface_pf_q[p]) ** 2 +
            s_base * (model.shared_es_pnet[ess_idx, s_m, s_o, p] - model.expected_shared_ess_p[p]) ** 2 +
            s_base * (model.shared_es_qnet[ess_idx, s_m, s_o, p] - model.expected_shared_ess_q[p]) ** 2
        )
        for s_m in model.scenarios_market
        for s_o in model.scenarios_operation
        for p in model.periods
    )
    model.regularization = pe.Expression(expr=expr)
    model.objective.expr += model.regularization


def dso_interface_expected_rule(m, p, net, ref_node_idx, ess_idx):
    evs = ep_p = ep_q = ess_p = ess_q = 0.0
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            weight = net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o]
            evs += weight * m.vmag_sqr[ref_node_idx, s_m, s_o, p]
            ep_p += weight * m.pg_node[ref_node_idx, s_m, s_o, p]
            ep_q += weight * m.qg_node[ref_node_idx, s_m, s_o, p]
            ess_p += weight * m.shared_es_pnet[ess_idx, s_m, s_o, p]
            ess_q += weight * m.shared_es_qnet[ess_idx, s_m, s_o, p]
    return (
        m.expected_interface_vmag_sqr[p] == evs,
        m.expected_interface_pf_p[p] == ep_p,
        m.expected_interface_pf_q[p] == ep_q,
        m.expected_shared_ess_p[p] == ess_p,
        m.expected_shared_ess_q[p] == ess_q,
    )


def dso_interface_expected_vmag_rule(m, p, net, ref_node_idx):
    vmag_sqr = 0.00
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            weight = net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o]
            vmag_sqr += weight * m.vmag_sqr[ref_node_idx, s_m, s_o, p]
    return m.expected_interface_vmag_sqr[p] == vmag_sqr


def dso_interface_expected_p_rule(m, p, net, ref_node_idx):
    interface_p = 0.00
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            weight = net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o]
            interface_p += weight * m.pg_node[ref_node_idx, s_m, s_o, p]
    return m.expected_interface_pf_p[p] == interface_p


def dso_interface_expected_q_rule(m, p, net, ref_node_idx):
    interface_q = 0.00
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            weight = net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o]
            interface_q += weight * m.qg_node[ref_node_idx, s_m, s_o, p]
    return m.expected_interface_pf_q[p] == interface_q


def dso_shared_expected_p_rule(m, p, net, shared_ess_idx):
    shared_ess_p = 0.0
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            weight = net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o]
            shared_ess_p += weight * m.shared_es_pnet[shared_ess_idx, s_m, s_o, p]
    return m.expected_shared_ess_p[p] == shared_ess_p


def dso_shared_expected_q_rule(m, p, net, shared_ess_idx):
    shared_ess_q = 0.0
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            weight = net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o]
            shared_ess_q += weight * m.shared_es_pnet[shared_ess_idx, s_m, s_o, p]
    return m.expected_shared_ess_q[p] == shared_ess_q

