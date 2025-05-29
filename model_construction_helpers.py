import pyomo.environ as pe
from math import sqrt, tan, atan2, acos
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
    # Initialize at lower bound if positive, else zero (or lb if lb <= 0)
    if lb > 0:
        return lb
    else:
        return max(0.0, lb)  # Just in case lb < 0


def qg_init(m, g, s_m, s_o, p, network):
    lb, ub = qg_bounds(m, g, s_m, s_o, p, network=network)
    if lb > 0:
        return lb
    else:
        return max(0.0, lb)


# Generation, Sg
def sg_bounds(m, g, s_m, s_o, p, network, params):

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
    rating_sqr = (branch.rate / network.baseMVA) ** 2
    return (0.0, SIJ_VIOLATION_ALLOWED * rating_sqr)


# Consumption, Pc
def pc_bounds(m, c, s_m, s_o, p, network, params):
    load = network.loads[c]
    pd = load.pd[s_o][p]
    return (pd - EQUALITY_TOLERANCE, pd + EQUALITY_TOLERANCE)


# Consumption, Qc
def qc_bounds(m, c, s_m, s_o, p, network, params):
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
def soc_bounds(e, network):
    es = network.energy_storages[e]
    return (es.e_min, es.e_max)


def q_bounds(e, network):
    es = network.energy_storages[e]
    return (-es.s, es.s)


def s_bounds(e, network):
    return (0.0, network.energy_storages[e].s)


def soc_initialize(m, e, network):
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
def voltage_magnitude_rule(m, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    e = m.e[i, s_m, s_o, p]
    f = m.f[i, s_m, s_o, p]
    vmag_sq = e ** 2 + f ** 2
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return pe.inequality(vg ** 2 - EQUALITY_TOLERANCE, vmag_sq, vg ** 2 + EQUALITY_TOLERANCE)
    else:
        return pe.inequality(node.v_min ** 2, vmag_sq, node.v_max ** 2)

# Generation, Sg^2
# Apparent power ≈ pg² + qg²
def sg_sqr_upper_bound_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return m.sg_sqr[g, s_m, s_o, p] <= m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2 + EQUALITY_TOLERANCE


def sg_sqr_lower_bound_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return m.sg_sqr[g, s_m, s_o, p] >= m.pg[g, s_m, s_o, p]**2 + m.qg[g, s_m, s_o, p]**2 - EQUALITY_TOLERANCE


# sg_abs² ≈ sg_sqr
def sg_abs_upper_bound_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return m.sg_abs[g, s_m, s_o, p]**2 <= m.sg_sqr[g, s_m, s_o, p] + EQUALITY_TOLERANCE


def sg_abs_lower_bound_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    return m.sg_abs[g, s_m, s_o, p]**2 >= m.sg_sqr[g, s_m, s_o, p] - EQUALITY_TOLERANCE


# Curtailment: sg_abs = init_sg - sg_curt
def sg_curtailment_upper_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    init_sg = 0.0
    if generator.status[p]:
        init_sg = sqrt(generator.pg[s_o][p]**2 + generator.qg[s_o][p]**2)
    return m.sg_abs[g, s_m, s_o, p] <= init_sg - m.sg_curt[g, s_m, s_o, p] + EQUALITY_TOLERANCE


def sg_curtailment_lower_rule(m, g, s_m, s_o, p, network):
    generator = network.generators[g]
    if not generator.is_curtaillable():
        return pe.Constraint.Skip
    init_sg = 0.0
    if generator.status[p]:
        init_sg = sqrt(generator.pg[s_o][p]**2 + generator.qg[s_o][p]**2)
    return m.sg_abs[g, s_m, s_o, p] >= init_sg - m.sg_curt[g, s_m, s_o, p] - EQUALITY_TOLERANCE


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
        p_up = sum(m.flex_p_up[c, s_m, s_o, p] for p in m.periods)
        p_down = sum(m.flex_p_down[c, s_m, s_o, p] for p in m.periods)
        if params.slacks.flexibility.day_balance:
            return p_up == p_down + m.slack_flex_p_balance[c, s_m, s_o]
        else:
            return pe.inequality(-EQUALITY_TOLERANCE, p_up - p_down, EQUALITY_TOLERANCE)
    else:
        return pe.Constraint.Skip


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


def ess_soc_final_rule(m, e, s_m, s_o, network, params):
    final_soc = network.energy_storages[e].e_init
    final_p = m.periods.last()
    if params.slacks.ess.day_balance:
        return m.es_soc[e, s_m, s_o, final_p] == final_soc + m.slack_es_soc_final[e, s_m, s_o]
    else:
        ineq = pe.inequality(
            final_soc - EQUALITY_TOLERANCE,
            m.es_soc[e, s_m, s_o, final_p],
            final_soc + EQUALITY_TOLERANCE
        )
        return ineq


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


def sess_pch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    pch = m.shared_es_pch[e, s_m, s_o, p]
    return pch <= s_max


def sess_pdch_limit(m, e, s_m, s_o, p):
    s_max = m.shared_es_s_rated[e]
    pdch = m.shared_es_pdch[e, s_m, s_o, p]
    return pdch <= s_max


def sess_soc_limits(m, e, s_m, s_o, p):
    soc_min = m.shared_es_e_rated[e] * ENERGY_STORAGE_MIN_ENERGY_STORED
    soc_max = m.shared_es_e_rated[e] * ENERGY_STORAGE_MAX_ENERGY_STORED
    return pe.inequality(soc_min, m.shared_es_soc[e, s_m, s_o, p], soc_max)


def sess_pnet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_pnet[e, s_m, s_o, p] - (m.shared_es_pch[e, s_m, s_o, p] - m.shared_es_pdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


def sess_qnet_rule(m, e, s_m, s_o, p):
    return pe.inequality(-EQUALITY_TOLERANCE, m.shared_es_qnet[e, s_m, s_o, p] - (m.shared_es_qch[e, s_m, s_o, p] - m.shared_es_qdch[e, s_m, s_o, p]), EQUALITY_TOLERANCE)


# Node balance
def compute_branch_power(branch, ei, fi, ej, fj, rij):
    vi_sq = ei ** 2 + fi ** 2
    Pi = branch.g * vi_sq * rij ** 2
    Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))
    Qi = -(branch.b + branch.b_sh * 0.5) * vi_sq * rij ** 2
    Qi += rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))
    return Pi, Qi


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
def compute_weight(network, omega_market, omega_oper):
    return network.baseMVA * omega_market * omega_oper


def add_flex_cost(model, s_m, s_o, p, c, unit_cost, baseMVA):
    flex_pc = model.flex_p_up[c, s_m, s_o, p] + model.flex_p_down[c, s_m, s_o, p]
    flex_qc = model.flex_q_up[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]
    return unit_cost * baseMVA * (flex_pc + flex_qc)


def add_load_curtailment(model, s_m, s_o, p, c, cost, baseMVA):
    pc = model.pc_curt_down[c, s_m, s_o, p] + model.pc_curt_up[c, s_m, s_o, p]
    qc = model.qc_curt_down[c, s_m, s_o, p] + model.qc_curt_up[c, s_m, s_o, p]
    return cost * baseMVA * (pc + qc)


def add_gen_curtailment(model, s_m, s_o, p, g, cost, baseMVA):
    return cost * baseMVA * model.sg_curt[g, s_m, s_o, p]


def add_ess_usage(model, s_m, s_o, p, e, sch_var, sdch_var, penalty, baseMVA):
    return penalty * baseMVA * (sch_var[e, s_m, s_o, p] + sdch_var[e, s_m, s_o, p])


def add_slack_squared(var):
    return var**2


def compute_node_load(model, i, s_m, s_o, p, network, params):
    Pd, Qd = 0.0, 0.0
    node = network.nodes[i]

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
                Pd += model.es_pch[e, s_m, s_o, p] - model.es_pdch[e, s_m, s_o, p]
                Qd += model.es_qch[e, s_m, s_o, p] - model.es_qdch[e, s_m, s_o, p]

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


def node_balance_p_rule(model, i, s_m, s_o, p, network, params):

    Pd, _ = compute_node_load(model, i, s_m, s_o, p, network, params)
    Pg, _ = compute_node_gen(model, i, s_m, s_o, p, network)

    node = network.nodes[i]
    ei = model.e_actual[i, s_m, s_o, p]
    fi = model.f_actual[i, s_m, s_o, p]

    Pi = node.gs * (ei**2 + fi**2)

    for b in range(len(network.branches)):

        branch = network.branches[b]
        if branch.fbus != node.bus_i and branch.tbus != node.bus_i:
            continue

        rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0

        # define from and to node indices
        fnode_idx = network.get_node_idx(branch.fbus)
        tnode_idx = network.get_node_idx(branch.tbus)
        if branch.fbus == node.bus_i:
            ei, fi = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
            ej, fj = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]
        else:
            ei, fi = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]
            ej, fj = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]

        Pi += branch.g * (ei**2 + fi**2) * rij**2
        Pi -= rij * (branch.g * (ei * ej + fi * fj) + branch.b * (fi * ej - ei * fj))

    if params.slacks.node_balance:
        return Pg == Pd + Pi + model.slack_node_balance_p[i, s_m, s_o, p]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, Pg - Pd - Pi, EQUALITY_TOLERANCE)


def node_balance_q_rule(model, i, s_m, s_o, p, network, params):

    _, Qd = compute_node_load(model, i, s_m, s_o, p, network, params)
    _, Qg = compute_node_gen(model, i, s_m, s_o, p, network)

    node = network.nodes[i]
    ei = model.e_actual[i, s_m, s_o, p]
    fi = model.f_actual[i, s_m, s_o, p]

    # Shunt reactive power at bus
    Qi = -node.bs * (ei**2 + fi**2)

    for b in range(len(network.branches)):
        branch = network.branches[b]
        if branch.fbus != node.bus_i and branch.tbus != node.bus_i:
            continue

        rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0

        fnode_idx = network.get_node_idx(branch.fbus)
        tnode_idx = network.get_node_idx(branch.tbus)

        if branch.fbus == node.bus_i:
            ei, fi = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
            ej, fj = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]
            shunt_term = (branch.b + branch.b_sh * 0.5) * (ei**2 + fi**2) * rij**2
            flow_term = rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))
        else:
            ei, fi = model.e_actual[tnode_idx, s_m, s_o, p], model.f_actual[tnode_idx, s_m, s_o, p]
            ej, fj = model.e_actual[fnode_idx, s_m, s_o, p], model.f_actual[fnode_idx, s_m, s_o, p]
            shunt_term = (branch.b + branch.b_sh * 0.5) * (ei**2 + fi**2)
            flow_term = rij * (branch.b * (ei * ej + fi * fj) - branch.g * (fi * ej - ei * fj))

        Qi -= shunt_term
        Qi += flow_term

    if params.slacks.node_balance:
        return Qg == Qd + Qi + model.slack_node_balance_q[i, s_m, s_o, p]
    else:
        return pe.inequality(-EQUALITY_TOLERANCE, Qg - Qd - Qi, EQUALITY_TOLERANCE)


def branch_flow_equation_rule(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    rij = model.r[b, s_m, s_o, p] if branch.is_transformer else 1.0

    fnode_idx = network.get_node_idx(branch.fbus)
    tnode_idx = network.get_node_idx(branch.tbus)

    ei = model.e_actual[fnode_idx, s_m, s_o, p]
    fi = model.f_actual[fnode_idx, s_m, s_o, p]
    ej = model.e_actual[tnode_idx, s_m, s_o, p]
    fj = model.f_actual[tnode_idx, s_m, s_o, p]

    flow_expr = compute_branch_flow_squared(branch, ei, fi, ej, fj, rij, params.branch_limit_type)
    flow_var = model.flow_ij_sqr[b, s_m, s_o, p]

    return pe.inequality(-EQUALITY_TOLERANCE, flow_var - flow_expr, EQUALITY_TOLERANCE)


def branch_flow_limit_rule(model, b, s_m, s_o, p, network, params):

    branch = network.branches[b]
    if not branch.status:
        return pe.Constraint.Skip

    rating = branch.rate / network.baseMVA or BRANCH_UNKNOWN_RATING
    flow_var = model.flow_ij_sqr[b, s_m, s_o, p]

    if params.slacks.grid_operation.branch_flow:
        slack = model.slack_flow_ij_sqr[b, s_m, s_o, p]
        return flow_var <= rating ** 2 + slack
    else:
        return flow_var <= rating ** 2

