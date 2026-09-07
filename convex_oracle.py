"""
P5.5-C — centralized convex lower-bound oracle for the shared-resources planning
problem.

This is a NEW code path. It does not modify, wrap or replace the nonlinear
production SMOPF, which remains the upper-bound / feasibility model.

Design principle: reuse production rule functions wherever they do not touch the
rectangular voltage variables. Production's objective terms and most operational
rules depend only on `pg`, `qg`, `pc`, `flex_*`, `shared_es_*` and the voltage
slacks, so a block that declares those variables under the SAME names can call
the production rules directly. Only the families that reference `e`, `f` or the
tap `r` are rewritten in W-space:

  * nodal P/Q balance
  * branch terminal powers and thermal limits
  * the rank coupling, relaxed to a rotated second-order cone

Uniform branch treatment (P5.5-B2). For every branch define (U_b, C_b, D_b):

    line        : U_b == Wii[from]          C_b == WijR       D_b == WijI
    transformer : r_min^2 Wii <= U_b <= r_max^2 Wii,  C_b = r*WijR,  D_b = r*WijI

so every branch expression is affine in (U_b, C_b, D_b, W_jj) and the coupling is

    C_b^2 + D_b^2 <= U_b * W_jj          (rotated SOC; equality in the exact model)

The tap variables `r` and `r_sqr` never exist in this model.
"""

import math
import os
import sys

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
from definitions import (BUS_REF, ENERGY_STORAGE_MAX_ENERGY_STORED,  # noqa: E402
                         ENERGY_STORAGE_MIN_ENERGY_STORED,
                         ENERGY_STORAGE_RELATIVE_INIT_SOC, EQUALITY_TOLERANCE,
                         TRANSFORMER_MAXIMUM_RATIO, TRANSFORMER_MINIMUM_RATIO)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _voltage_bounds(node):
    lo = max(node.v_min ** 2 - 0.5, 0.0)
    return lo, node.v_max ** 2 + 0.5


# ---------------------------------------------------------------------------
#  one AC block  (agent, year, day, scenario)
# ---------------------------------------------------------------------------
def build_ac_block(network, params, s_m=0, s_o=0):
    """W-space convex relaxation of one production SMOPF."""
    blk = pe.Block(concrete=True)
    blk.network = network
    blk.params = params

    nodes = range(len(network.nodes))
    branches = [b for b, br in enumerate(network.branches) if br.status]
    periods = range(network.num_instants)

    blk.nodes = pe.Set(initialize=list(nodes), ordered=True)
    blk.branches = pe.Set(initialize=branches, ordered=True)
    blk.periods = pe.Set(initialize=list(periods), ordered=True)
    blk.loads = pe.Set(initialize=list(range(len(network.loads))), ordered=True)
    blk.generators = pe.Set(initialize=list(range(len(network.generators))), ordered=True)
    blk.shared_energy_storages = pe.Set(
        initialize=list(range(len(network.shared_energy_storages))), ordered=True)
    blk.energy_storages = pe.Set(
        initialize=list(range(len(network.energy_storages))), ordered=True)
    blk.scenarios_market = pe.Set(initialize=[s_m], ordered=True)
    blk.scenarios_operation = pe.Set(initialize=[s_o], ordered=True)

    # ---------------- W-space voltage variables ----------------
    def w_bounds(m, i, sm, so, p):
        return _voltage_bounds(network.nodes[i])
    blk.vmag_sqr = pe.Var(blk.nodes, blk.scenarios_market, blk.scenarios_operation,
                          blk.periods, domain=pe.NonNegativeReals,
                          bounds=w_bounds, initialize=1.0)

    vmax = max(n.v_max for n in network.nodes)
    cd_cap = (TRANSFORMER_MAXIMUM_RATIO * vmax) ** 2 + 1.0
    # C >= 0 is production's `voltage_product_real_nonnegative` (class A)
    blk.Cb = pe.Var(blk.branches, blk.periods, domain=pe.NonNegativeReals,
                    bounds=(0.0, cd_cap), initialize=1.0)
    blk.Db = pe.Var(blk.branches, blk.periods, bounds=(-cd_cap, cd_cap), initialize=0.0)
    blk.Ub = pe.Var(blk.branches, blk.periods, domain=pe.NonNegativeReals,
                    bounds=(0.0, cd_cap), initialize=1.0)

    # ---------------- operational variables (production names) ----------------
    blk.pg = pe.Var(blk.generators, blk.scenarios_market, blk.scenarios_operation,
                    blk.periods, bounds=lambda m, g, sm, so, p: mch.pg_bounds(
                        m, g, sm, so, p, network),
                    initialize=0.0)
    blk.qg = pe.Var(blk.generators, blk.scenarios_market, blk.scenarios_operation,
                    blk.periods, bounds=lambda m, g, sm, so, p: mch.qg_bounds(
                        m, g, sm, so, p, network),
                    initialize=0.0)
    if network.is_transmission:
        # Production keeps pc/qc as decision variables on the transmission
        # network (network.py:308) because the ADN interface powers are set by
        # the coordination procedure.  Mirror that exactly, bounds included.
        blk.pc = pe.Var(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                        blk.periods, domain=pe.Reals,
                        initialize=lambda m, c, sm, so, p: mch.pc_initialize(
                            m, c, sm, so, p, network),
                        bounds=lambda m, c, sm, so, p: mch.pc_bounds(
                            m, c, sm, so, p, network))
        blk.qc = pe.Var(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                        blk.periods, domain=pe.Reals,
                        initialize=lambda m, c, sm, so, p: mch.qc_initialize(
                            m, c, sm, so, p, network),
                        bounds=lambda m, c, sm, so, p: mch.qc_bounds(
                            m, c, sm, so, p, network))
    else:
        blk.pc = pe.Param(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                          blk.periods, mutable=True,
                          initialize=lambda m, c, sm, so, p: mch.pc_initialize(
                              m, c, sm, so, p, network))
        blk.qc = pe.Param(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                          blk.periods, mutable=True,
                          initialize=lambda m, c, sm, so, p: mch.qc_initialize(
                              m, c, sm, so, p, network))
    if params.fl_reg:
        blk.flex_p_up = pe.Var(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                               blk.periods, domain=pe.NonNegativeReals,
                               bounds=lambda m, c, sm, so, p: mch.pc_flex_up_bounds(
                                   m, c, sm, so, p, network, params), initialize=0.0)
        blk.flex_p_down = pe.Var(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                                 blk.periods, domain=pe.NonNegativeReals,
                                 bounds=lambda m, c, sm, so, p: mch.pc_flex_down_bounds(
                                     m, c, sm, so, p, network, params), initialize=0.0)
        blk.flex_q_up = pe.Var(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                               blk.periods, domain=pe.NonNegativeReals,
                               bounds=lambda m, c, sm, so, p: mch.qc_flex_up_bounds(
                                   m, c, sm, so, p, network, params), initialize=0.0)
        blk.flex_q_down = pe.Var(blk.loads, blk.scenarios_market, blk.scenarios_operation,
                                 blk.periods, domain=pe.NonNegativeReals,
                                 bounds=lambda m, c, sm, so, p: mch.qc_flex_down_bounds(
                                     m, c, sm, so, p, network, params), initialize=0.0)
    if params.slacks.grid_operation.voltage:
        blk.slack_v_sqr_up = pe.Var(blk.nodes, blk.scenarios_market, blk.scenarios_operation,
                                    blk.periods, domain=pe.NonNegativeReals,
                                    bounds=lambda m, i, sm, so, p: mch.voltage_slack_up_bounds(
                                        m, i, sm, so, p, network, params), initialize=0.0)
        blk.slack_v_sqr_down = pe.Var(blk.nodes, blk.scenarios_market, blk.scenarios_operation,
                                      blk.periods, domain=pe.NonNegativeReals,
                                      bounds=lambda m, i, sm, so, p: mch.voltage_slack_down_bounds(
                                          m, i, sm, so, p, network, params), initialize=0.0)

    # shared ESS (capacity comes from the parent, so bounds stay generous here)
    big = 10.0
    for name, dom in (('shared_es_pch', pe.NonNegativeReals),
                      ('shared_es_pdch', pe.NonNegativeReals),
                      ('shared_es_pnet', pe.Reals), ('shared_es_qnet', pe.Reals),
                      ('shared_es_soc', pe.NonNegativeReals)):
        setattr(blk, name, pe.Var(blk.shared_energy_storages, blk.scenarios_market,
                                  blk.scenarios_operation, blk.periods, domain=dom,
                                  bounds=(None if dom is pe.Reals else 0.0, big),
                                  initialize=0.0))
    if params.slacks.shared_ess.day_balance:
        blk.slack_shared_es_soc_final_up = pe.Var(
            blk.shared_energy_storages, blk.scenarios_market, blk.scenarios_operation,
            domain=pe.NonNegativeReals, initialize=0.0)
        blk.slack_shared_es_soc_final_down = pe.Var(
            blk.shared_energy_storages, blk.scenarios_market, blk.scenarios_operation,
            domain=pe.NonNegativeReals, initialize=0.0)

    # capacity handles; the parent ties these to the ESSO block
    blk.S_av = pe.Var(blk.shared_energy_storages, domain=pe.NonNegativeReals,
                      bounds=(0.0, big), initialize=0.0)
    blk.E_av = pe.Var(blk.shared_energy_storages, domain=pe.NonNegativeReals,
                      bounds=(0.0, big), initialize=0.0)

    # RES availability parameters, exactly as production declares them
    if params.rg_curt:
        blk.pg_avail = pe.Param(
            blk.generators, blk.scenarios_operation, blk.periods,
            domain=pe.NonNegativeReals, mutable=False,
            initialize=lambda m, g, so, p: mch.pg_avail_init(m, g, so, p, network, params))
        blk.sg_avail = pe.Param(
            blk.generators, blk.scenarios_operation, blk.periods,
            domain=pe.NonNegativeReals, mutable=False,
            initialize=lambda m, g, so, p: mch.sg_avail_init(m, g, so, p, network, params))
        blk.sg_sqr = pe.Expression(
            blk.generators, blk.scenarios_market, blk.scenarios_operation, blk.periods,
            rule=lambda m, g, sm, so, p: mch.sg_sqr_rule(m, g, sm, so, p, network))

    # ---------------- nodal aggregation (production semantics) ----------------
    def pc_node(i, p):
        node = network.nodes[i]
        Pd = 0.0
        for c in blk.loads:
            if network.loads[c].bus == node.bus_i:
                Pd += blk.pc[c, s_m, s_o, p]
                if params.fl_reg and network.loads[c].fl_reg:
                    Pd += blk.flex_p_up[c, s_m, s_o, p] - blk.flex_p_down[c, s_m, s_o, p]
        for e in blk.shared_energy_storages:
            if network.shared_energy_storages[e].bus == node.bus_i:
                Pd += blk.shared_es_pnet[e, s_m, s_o, p]
        return Pd

    def qc_node(i, p):
        node = network.nodes[i]
        Qd = 0.0
        for c in blk.loads:
            if network.loads[c].bus == node.bus_i:
                Qd += blk.qc[c, s_m, s_o, p]
                if params.fl_reg and network.loads[c].fl_reg:
                    Qd += blk.flex_q_up[c, s_m, s_o, p] - blk.flex_q_down[c, s_m, s_o, p]
        for e in blk.shared_energy_storages:
            if network.shared_energy_storages[e].bus == node.bus_i:
                Qd += blk.shared_es_qnet[e, s_m, s_o, p]
        return Qd

    def pg_node(i, p):
        node = network.nodes[i]
        return sum(blk.pg[g, s_m, s_o, p] for g in blk.generators
                   if network.generators[g].bus == node.bus_i)

    def qg_node(i, p):
        node = network.nodes[i]
        return sum(blk.qg[g, s_m, s_o, p] for g in blk.generators
                   if network.generators[g].bus == node.bus_i)

    # ---------------- branch U/C/D structure ----------------
    def ub_rule(m, b, p):
        branch = network.branches[b]
        f_idx = network.get_node_idx(branch.fbus)
        if branch.is_transformer and params.transf_reg and branch.vmag_reg:
            return pe.Constraint.Skip          # handled by the tap box below
        if branch.is_transformer:
            ratio_sqr = branch.ratio ** 2
            return m.Ub[b, p] == ratio_sqr * m.vmag_sqr[f_idx, s_m, s_o, p]
        return m.Ub[b, p] == m.vmag_sqr[f_idx, s_m, s_o, p]
    blk.ub_def = pe.Constraint(blk.branches, blk.periods, rule=ub_rule)

    tap_branches = [b for b in branches
                    if network.branches[b].is_transformer
                    and params.transf_reg and network.branches[b].vmag_reg]
    blk.tap_branches = pe.Set(initialize=tap_branches, ordered=True)

    def tap_lower(m, b, p):
        f_idx = network.get_node_idx(network.branches[b].fbus)
        return m.Ub[b, p] >= TRANSFORMER_MINIMUM_RATIO ** 2 * m.vmag_sqr[f_idx, s_m, s_o, p]

    def tap_upper(m, b, p):
        f_idx = network.get_node_idx(network.branches[b].fbus)
        return m.Ub[b, p] <= TRANSFORMER_MAXIMUM_RATIO ** 2 * m.vmag_sqr[f_idx, s_m, s_o, p]
    blk.tap_lower = pe.Constraint(blk.tap_branches, blk.periods, rule=tap_lower)
    blk.tap_upper = pe.Constraint(blk.tap_branches, blk.periods, rule=tap_upper)

    # rank coupling, relaxed  (rotated second-order cone)
    def soc_rule(m, b, p):
        t_idx = network.get_node_idx(network.branches[b].tbus)
        return (m.Cb[b, p] ** 2 + m.Db[b, p] ** 2
                <= m.Ub[b, p] * m.vmag_sqr[t_idx, s_m, s_o, p])
    blk.rank_soc = pe.Constraint(blk.branches, blk.periods, rule=soc_rule)

    # ---------------- terminal powers (affine in U, C, D, Wjj) ----------------
    def terminal_pq(b, p, head_idx):
        """Production terminal power at `head_idx`, affine in (U, C, D, W).

        Verified against production in P5.5-B2 to 5.9e-15. When the head is the
        'to' bus the imaginary part flips sign (production's
        `_branch_voltage_products` convention) and the head voltage term uses
        W_jj rather than U.
        """
        branch = network.branches[b]
        g, bb, bsh = branch.g, branch.b, branch.b_sh
        f_idx = network.get_node_idx(branch.fbus)
        t_idx = network.get_node_idx(branch.tbus)
        Cv = blk.Cb[b, p]
        if head_idx == f_idx:
            head_w = blk.Ub[b, p]
            Dv = blk.Db[b, p]
        elif head_idx == t_idx:
            head_w = blk.vmag_sqr[t_idx, s_m, s_o, p]
            Dv = -blk.Db[b, p]
        else:
            raise ValueError(f'node {head_idx} not incident to branch {b}')
        P = g * head_w - g * Cv - bb * Dv
        Q = -(bb + 0.5 * bsh) * head_w + bb * Cv - g * Dv
        return P, Q

    # ---------------- node balance ----------------
    def node_balance_p(m, i, p):
        node = network.nodes[i]
        Pi = node.gs * m.vmag_sqr[i, s_m, s_o, p]
        for b in blk.branches:
            branch = network.branches[b]
            if branch.fbus != node.bus_i and branch.tbus != node.bus_i:
                continue
            P, _ = terminal_pq(b, p, i)
            Pi += P
        return pg_node(i, p) == pc_node(i, p) + Pi
    blk.node_balance_p = pe.Constraint(blk.nodes, blk.periods, rule=node_balance_p)

    def node_balance_q(m, i, p):
        node = network.nodes[i]
        Qi = -node.bs * m.vmag_sqr[i, s_m, s_o, p]
        for b in blk.branches:
            branch = network.branches[b]
            if branch.fbus != node.bus_i and branch.tbus != node.bus_i:
                continue
            _, Q = terminal_pq(b, p, i)
            Qi += Q
        return qg_node(i, p) == qc_node(i, p) + Qi
    blk.node_balance_q = pe.Constraint(blk.nodes, blk.periods, rule=node_balance_q)

    # ---------------- branch limits ----------------
    apparent = [b for b in branches
                if mch.branch_uses_apparent_power_limit(network.branches[b], params)]
    blk.apparent_branches = pe.Set(initialize=apparent, ordered=True)

    def flow_ij_rule(m, b, p):
        branch = network.branches[b]
        rating = branch.rate / network.baseMVA or 999.99
        f_idx = network.get_node_idx(branch.fbus)
        t_idx = network.get_node_idx(branch.tbus)
        if b in apparent:
            P, Q = terminal_pq(b, p, f_idx)
            return P ** 2 + Q ** 2 <= rating ** 2 + EQUALITY_TOLERANCE
        # production: (g^2+b^2)(Wii + r^2 Wjj - 2 r WijR); with r == 1 on lines
        return ((branch.g ** 2 + branch.b ** 2)
                * (m.vmag_sqr[f_idx, s_m, s_o, p] + m.vmag_sqr[t_idx, s_m, s_o, p]
                   - 2 * m.Cb[b, p])
                <= rating ** 2 + EQUALITY_TOLERANCE)
    blk.branch_flow_limit = pe.Constraint(blk.branches, blk.periods, rule=flow_ij_rule)

    def flow_ji_rule(m, b, p):
        branch = network.branches[b]
        rating = branch.rate / network.baseMVA or 999.99
        t_idx = network.get_node_idx(branch.tbus)
        P, Q = terminal_pq(b, p, t_idx)
        return P ** 2 + Q ** 2 <= rating ** 2 + EQUALITY_TOLERANCE
    blk.branch_flow_limit_ji = pe.Constraint(blk.apparent_branches, blk.periods,
                                             rule=flow_ji_rule)

    # ---------------- voltage limits / reference gauge ----------------
    def v_lower(m, i, p):
        node = network.nodes[i]
        slack = (m.slack_v_sqr_down[i, s_m, s_o, p]
                 if params.slacks.grid_operation.voltage
                 and mch._voltage_magnitude_slack_enabled(node, params) else 0.0)
        return m.vmag_sqr[i, s_m, s_o, p] + slack >= node.v_min ** 2

    def v_upper(m, i, p):
        node = network.nodes[i]
        slack = (m.slack_v_sqr_up[i, s_m, s_o, p]
                 if params.slacks.grid_operation.voltage
                 and mch._voltage_magnitude_slack_enabled(node, params) else 0.0)
        return m.vmag_sqr[i, s_m, s_o, p] - slack <= node.v_max ** 2
    blk.v_lower = pe.Constraint(blk.nodes, blk.periods, rule=v_lower)
    blk.v_upper = pe.Constraint(blk.nodes, blk.periods, rule=v_upper)

    # DSO reference bus: production pins e to vg and f to ~0, i.e. W_ref = vg^2
    if not network.is_transmission:
        ref_id = network.get_reference_node_id()
        ref_idx = network.get_node_idx(ref_id)
        vg = network.generators[network.get_gen_idx(ref_id)].vg

        def ref_gauge(m, p):
            v = vg[p] if hasattr(vg, '__len__') else vg
            return pe.inequality((v - 1e-4) ** 2, m.vmag_sqr[ref_idx, s_m, s_o, p],
                                 (v + 1e-4) ** 2 + 1e-10)
        blk.ref_gauge = pe.Constraint(blk.periods, rule=ref_gauge)

    # ---------------- generation capability / PF (production rules) ----------
    blk.sg_capability = pe.Constraint(
        blk.generators, blk.scenarios_market, blk.scenarios_operation, blk.periods,
        rule=lambda m, g, sm, so, p: mch.sg_avail_rule(m, g, sm, so, p, network, params))

    blk.gen_pf_upper = pe.Constraint(
        blk.generators, blk.scenarios_market, blk.scenarios_operation, blk.periods,
        rule=lambda m, g, sm, so, p: mch.power_factor_rule_upper(m, g, sm, so, p, network))
    blk.gen_pf_lower = pe.Constraint(
        blk.generators, blk.scenarios_market, blk.scenarios_operation, blk.periods,
        rule=lambda m, g, sm, so, p: mch.power_factor_rule_lower(m, g, sm, so, p, network))

    # ---------------- flexibility day balance (production rule) --------------
    if params.fl_reg:
        blk.flex_energy_balance_p = pe.Constraint(
            blk.loads, blk.scenarios_market, blk.scenarios_operation,
            rule=lambda m, c, sm, so: mch.flex_energy_balance_p_rule(
                m, c, sm, so, network, params))

    # ---------------- shared ESS (convex LB form) ----------------
    dt = mch.HOURS_PER_REPRESENTATIVE_DAY / network.num_instants

    def sess_pnet(m, e, p):
        return (m.shared_es_pnet[e, s_m, s_o, p]
                == m.shared_es_pch[e, s_m, s_o, p] - m.shared_es_pdch[e, s_m, s_o, p])
    blk.sess_pnet_def = pe.Constraint(blk.shared_energy_storages, blk.periods, rule=sess_pnet)

    def sess_sum(m, e, p):
        return (m.shared_es_pch[e, s_m, s_o, p]
                + m.shared_es_pdch[e, s_m, s_o, p] <= m.S_av[e])
    blk.sess_active_sum_limit = pe.Constraint(blk.shared_energy_storages, blk.periods,
                                              rule=sess_sum)

    def sess_cap(m, e, p):
        # NORM form: jointly convex in (pnet, qnet, S) -- exact, not a relaxation
        return (m.shared_es_pnet[e, s_m, s_o, p] ** 2
                + m.shared_es_qnet[e, s_m, s_o, p] ** 2 <= m.S_av[e] ** 2)
    blk.sess_converter_capability = pe.Constraint(blk.shared_energy_storages, blk.periods,
                                                  rule=sess_cap)

    blk.sess_phi_limit_lower = pe.Constraint(
        blk.shared_energy_storages, blk.scenarios_market, blk.scenarios_operation,
        blk.periods, rule=lambda m, e, sm, so, p: mch.sess_phi_limits_lower(
            m, e, sm, so, p, network))
    blk.sess_phi_limit_upper = pe.Constraint(
        blk.shared_energy_storages, blk.scenarios_market, blk.scenarios_operation,
        blk.periods, rule=lambda m, e, sm, so, p: mch.sess_phi_limits_upper(
            m, e, sm, so, p, network))

    def sess_soc(m, e, p):
        ess = network.shared_energy_storages[e]
        prev = (m.E_av[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC if p == 0
                else m.shared_es_soc[e, s_m, s_o, p - 1])
        delta = (ess.eff_ch * m.shared_es_pch[e, s_m, s_o, p] * dt
                 - m.shared_es_pdch[e, s_m, s_o, p] * dt / ess.eff_dch)
        return m.shared_es_soc[e, s_m, s_o, p] == prev + delta
    blk.sess_soc_def = pe.Constraint(blk.shared_energy_storages, blk.periods, rule=sess_soc)

    blk.sess_soc_upper = pe.Constraint(
        blk.shared_energy_storages, blk.periods,
        rule=lambda m, e, p: m.shared_es_soc[e, s_m, s_o, p]
        <= m.E_av[e] * ENERGY_STORAGE_MAX_ENERGY_STORED)
    blk.sess_soc_lower = pe.Constraint(
        blk.shared_energy_storages, blk.periods,
        rule=lambda m, e, p: m.shared_es_soc[e, s_m, s_o, p]
        >= m.E_av[e] * ENERGY_STORAGE_MIN_ENERGY_STORED)

    def sess_final(m, e):
        final = m.E_av[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC
        last = list(blk.periods)[-1]
        if params.slacks.shared_ess.day_balance:
            return (m.shared_es_soc[e, s_m, s_o, last] == final
                    + m.slack_shared_es_soc_final_up[e, s_m, s_o]
                    - m.slack_shared_es_soc_final_down[e, s_m, s_o])
        return pe.inequality(-EQUALITY_TOLERANCE,
                             m.shared_es_soc[e, s_m, s_o, last] - final,
                             EQUALITY_TOLERANCE)
    blk.sess_soc_final = pe.Constraint(blk.shared_energy_storages, rule=sess_final)

    # production cost parameters, so the production objective rules apply verbatim
    mch.setup_cost_parameters(blk, params)

    blk.s_m, blk.s_o, blk.dt = s_m, s_o, dt
    return blk


def block_objective(blk):
    """Retained production objective terms, evaluated on the convex block.

    Dropped (all non-negative, hence lower-bound safe): the ESS complementarity
    penalty. Load curtailment is disabled in SRP1.
    """
    network, params = blk.network, blk.params
    s_m, s_o = blk.s_m, blk.s_o
    prob = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]

    class _Shim:
        """Presents the block under the names production's rules expect."""
        def __init__(self, b):
            self._b = b
        def __getattr__(self, name):
            return getattr(self._b, name)

    shim = _Shim(blk)
    total = 0.0
    total += prob * mch.generation_cost(shim, network, s_m, s_o, params)
    total += prob * mch.flexibility_cost(shim, network, s_m, s_o, params)
    total += prob * mch.gen_curtailment_penalty(shim, network, s_m, s_o, params)
    total += prob * mch.ess_utilization_cost_penalty(shim, network, s_m, s_o, params)
    total += prob * mch.slack_penalties(shim, network, s_m, s_o, params)
    return total
