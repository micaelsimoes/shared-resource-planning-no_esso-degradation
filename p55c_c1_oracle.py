"""
Stage P5.5-C1/C5 -- the centralized convex lower-bound oracle.

One parent Pyomo model holding

  * one W-space AC block per (agent, year, day, scenario) -- TSO plus DSOs 5/7/9,
    built by `convex_oracle.build_ac_block` (C1/C2/C3/C4);
  * one convex ESSO/investment block (C5) carrying, per (node, year), the rated
    and available S/E capacities;
  * exact affine couplings tying every network block's capacity handle to the
    ESSO available capacity, with the explicit baseMVA conversion (C0.2).

There is no ADMM.  Every quantity production reconciles by consensus is instead
an exact affine equality in this model:

  * capacity   -- TSO and DSO blocks reference the SAME (node, year) capacity
                  variables, so `S_network^TSO == S_network^DSO == S_available`
                  holds by construction.  Capacity is indexed by (node, year)
                  only, hence shared across the representative days of a year.
  * interface  -- the three ADMM consensus families, written with production's
                  own interface definitions and production's unit conversions:

      vmag : vmag_TSO[adn node] * kV_base_TSO == vmag_DSO[ref node] * kV_base_DSO
             which in W-space is exactly
             W_TSO[adn] * kV_base_TSO^2 == W_DSO[ref] * kV_base_DSO^2

      pf   : (pc + flex_p_up - flex_p_down)_TSO[adn load] * MVA_TSO
             == (pg[ref gen] - sum shared_es_pnet)_DSO * MVA_DSO      (and q)

      ess  : shared_es_pnet_TSO[e] * MVA_TSO == shared_es_pnet_DSO[e'] * MVA_DSO
             (and qnet)

Without these the TSO sees no ADN demand at all -- its interface loads are the
data-file defaults, which are zero -- and the oracle would bound a different,
far cheaper problem than the one production solves.

The ESSO block is the convex relaxation described in C5: degradation, SoH,
minimum-SoH, complementarity and their slacks are dropped, and the physical
identity `E_available == E_rated * soh_cumul` (soh_cumul <= 1) is relaxed to the
interval `0 <= E_available <= E_rated`, which contains it.  `S_available` stays
an exact equality to `S_rated` (verified in C0.2).

Salvage is NOT part of this model.  Per C9 the lower bound is recovered as
`LB_rec(x) = ObjBound_R(x) - V_salvage_max(x)` using the affine maximum-salvage
credit proven in C0.1.
"""

import os
import sys

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from convex_oracle import block_objective, build_ac_block  # noqa: E402


def block_weight(network_data, year, day):
    """Production's year/day weighting, identical to `_get_admm_block_weight`."""
    years = list(network_data.years)
    annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - int(years[0])))
    return float(network_data.years[year]) * float(network_data.days[day]) * annualization


def _agents(planning):
    """(tag, node_id_or_None, network_data) for every agent, TSO first."""
    agents = [('TSO', None, planning.transmission_network)]
    for node_id in sorted(planning.distribution_networks):
        agents.append((f'DSO{node_id}', node_id, planning.distribution_networks[node_id]))
    return agents


def _sess_node_map(network, holder_node_id):
    """Map each shared-ESS index of `network` to the ESSO node it represents.

    On the transmission network every shared ESS sits at the ADN connection bus,
    so the bus id is the ESSO node id directly.  On a distribution network the
    single shared ESS sits at that network's reference bus and represents the
    ESSO node the DSO is attached to.
    """
    if holder_node_id is None:
        return {e: ess.bus for e, ess in enumerate(network.shared_energy_storages)}
    ref = network.get_reference_node_id()
    idx = network.get_shared_energy_storage_idx(ref)
    return {idx: holder_node_id}


def build_centralized_relaxation(planning, candidate, objective_scale=1.0,
                                 only_years=None, only_days=None, only_agents=None,
                                 widen_narrow_bounds=None):
    """Assemble the parent convex model for one investment candidate.

    `candidate` is a production candidate solution; its `total_capacity` entries
    (MVA / MVAh, per node and year) are the parametrization x on which the cut
    contract of C10 is defined.

    `objective_scale` divides the objective by a positive constant.  This is a
    pure reformulation: the feasible set and the argmin are untouched, and both
    the optimal value and every dual multiplier scale back by the same constant.
    It exists because production's penalty constants span 1e-1 to 1e6 and the
    year/day weights multiply them by ~4.6e2, which leaves the barrier unable to
    certify a dual bound at the default scaling.
    """
    if objective_scale <= 0.0:
        raise ValueError('objective_scale must be positive')
    # `only_years` / `only_days` / `only_agents` restrict the model to a
    # sub-problem.  They are a diagnostic device only: a restricted model bounds
    # a restricted problem, never the full one.  (The names carry the `only_`
    # prefix because `years` is already a local holding the ESSO horizon.)
    parent = pe.ConcreteModel(name='p55c_convex_lower_bound_oracle')

    esso = planning.shared_ess_data
    nodes = list(esso.active_distribution_network_nodes)
    years = list(esso.years)

    # ------------------------------------------------------------------ C5
    parent.esso_nodes = pe.Set(initialize=nodes, ordered=True)
    parent.esso_years = pe.Set(initialize=years, ordered=True)
    ny = parent.esso_nodes * parent.esso_years

    parent.S_rated = pe.Var(ny, domain=pe.NonNegativeReals, initialize=0.0)
    parent.E_rated = pe.Var(ny, domain=pe.NonNegativeReals, initialize=0.0)
    parent.S_available = pe.Var(ny, domain=pe.NonNegativeReals, initialize=0.0)
    parent.E_available = pe.Var(ny, domain=pe.NonNegativeReals, initialize=0.0)

    # the capacity-fixing rows: these are the rows whose duals form g_k in C10
    def fix_s(m, n, y):
        return m.S_rated[n, y] == abs(candidate['total_capacity'][n][y]['s'])

    def fix_e(m, n, y):
        return m.E_rated[n, y] == abs(candidate['total_capacity'][n][y]['e'])
    parent.capacity_fix_s = pe.Constraint(ny, rule=fix_s)
    parent.capacity_fix_e = pe.Constraint(ny, rule=fix_e)

    # C0.2: S_available == S_rated exactly; E_available in [0, E_rated]
    parent.s_available_def = pe.Constraint(
        ny, rule=lambda m, n, y: m.S_available[n, y] == m.S_rated[n, y])
    parent.e_available_ub = pe.Constraint(
        ny, rule=lambda m, n, y: m.E_available[n, y] <= m.E_rated[n, y])

    # ------------------------------------------------------------------ C1
    parent.blocks = {}                      # (tag, year, day, s_m, s_o) -> block
    parent.block_weights = {}
    link_s, link_e = [], []

    for tag, node_id, holder in _agents(planning):
        if only_agents is not None and tag not in only_agents:
            continue
        for year in holder.years:
            if only_years is not None and year not in only_years:
                continue
            for day in holder.days:
                if only_days is not None and day not in only_days:
                    continue
                network = holder.network[year][day]
                sess_nodes = _sess_node_map(network, node_id)
                if node_id is None:
                    # production's own interface transformer ratings, per ADN,
                    # in p.u. on the transmission base (shared_resources_planning.py:2899)
                    ratings = {
                        adn: (planning.distribution_networks[adn].network[year][day]
                              .get_interface_branch_rating() / network.baseMVA)
                        for adn in network.active_distribution_network_nodes}
                    coordination = {'role': 'tso', 'interface_ratings': ratings}
                else:
                    coordination = {'role': 'dso'}
                for s_m in range(len(network.prob_market_scenarios)):
                    for s_o in range(len(network.prob_operation_scenarios)):
                        key = (tag, year, day, s_m, s_o)
                        name = f'ac_{tag}_{year}_{day}_{s_m}_{s_o}'
                        blk = build_ac_block(network, holder.params, s_m=s_m, s_o=s_o,
                                             coordination=coordination)
                        parent.add_component(name, blk)
                        parent.blocks[key] = blk
                        parent.block_weights[key] = block_weight(holder, year, day)
                        for e, esso_node in sess_nodes.items():
                            link_s.append((name, e, esso_node, year, network.baseMVA))
                            link_e.append((name, e, esso_node, year, network.baseMVA))

    # ------------------------------------------------------------- couplings
    parent.link_index = pe.Set(initialize=[(a, b) for a, b, _, _, _ in link_s],
                               dimen=2, ordered=True)
    _s_meta = {(a, b): (n, y, base) for a, b, n, y, base in link_s}

    def couple_s(m, name, e):
        node, year, base = _s_meta[(name, e)]
        return getattr(m, name).S_av[e] * base == m.S_available[node, year]

    def couple_e(m, name, e):
        node, year, base = _s_meta[(name, e)]
        return getattr(m, name).E_av[e] * base == m.E_available[node, year]
    parent.couple_s = pe.Constraint(parent.link_index, rule=couple_s)
    parent.couple_e = pe.Constraint(parent.link_index, rule=couple_e)

    # ------------------------------------------------------------- objective
    parent.objective_scale = objective_scale
    parent.objective = pe.Objective(
        expr=sum(parent.block_weights[k] * block_objective(b)
                 for k, b in parent.blocks.items()) / objective_scale,
        sense=pe.minimize)

    # ------------------------------------------------- TSO/DSO interface (C1)
    _add_interface_couplings(parent, planning)

    if widen_narrow_bounds:
        parent.widened_variables = _widen_narrow_bounds(parent, widen_narrow_bounds)

    parent.link_meta = _s_meta
    return parent


def _widen_narrow_bounds(parent, width):
    """Widen every box narrower than `width` to exactly `width`.

    Production boxes quantities it wants to be effectively zero into
    [0, EQUALITY_TOLERANCE] = [0, 1e-5] instead of fixing them.  In the
    distribution networks that is ~19% of all variables (the reactive
    flexibility of loads that have no reactive flexibility), and a log-barrier
    term on a 1e-5-wide corridor is what stops Gurobi producing a QCP dual.

    Widening ENLARGES the feasible set.  The relaxation therefore stays a
    relaxation and any dual bound obtained on the widened model is still a valid
    bound on the original -- it is simply weaker.  The looseness this buys is
    bounded by (objective coefficient) x (width) x (number widened) and is
    measured, not assumed.

    Returns the number of variables widened.
    """
    n = 0
    for var in parent.component_objects(pe.Var, active=None, descend_into=True):
        for index in var:
            data = var[index]
            lo, hi = data.bounds
            if lo is None or hi is None or hi <= lo:
                continue
            if (hi - lo) >= width:
                continue
            data.setub(lo + width)
            n += 1
    return n


def _add_interface_couplings(parent, planning):
    """Exact affine replacements for the ADMM vmag / pf / ess consensus."""
    tso = planning.transmission_network
    rows_v, rows_p, rows_q, rows_ep, rows_eq = [], [], [], [], []

    for node_id, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net = tso.network[year][day]
                d_net = dso.network[year][day]
                t_key = ('TSO', year, day, 0, 0)
                d_key = (f'DSO{node_id}', year, day, 0, 0)
                if t_key not in parent.blocks or d_key not in parent.blocks:
                    continue
                t_blk, d_blk = parent.blocks[t_key], parent.blocks[d_key]

                adn_idx = t_net.get_node_idx(node_id)
                adn_load = t_net.get_adn_load_idx(node_id)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                ref_gen = d_net.get_reference_gen_idx()
                kv_t = t_net.get_node_base_kv(node_id)
                kv_d = d_net.get_node_base_kv(ref_id)
                sess_t = [e for e, ess in enumerate(t_net.shared_energy_storages)
                          if ess.bus == node_id]
                sess_d = [e for e, ess in enumerate(d_net.shared_energy_storages)
                          if ess.bus == ref_id]

                for p in t_blk.periods:
                    meta = (t_blk, d_blk, adn_idx, adn_load, ref_idx, ref_gen,
                            kv_t, kv_d, t_net, d_net, sess_t, sess_d, p)
                    rows_v.append((node_id, year, day, p, meta))
                for p in t_blk.periods:
                    rows_p.append((node_id, year, day, p))
                    rows_q.append((node_id, year, day, p))
                for e_t in sess_t:
                    for e_d in sess_d:
                        for p in t_blk.periods:
                            rows_ep.append((node_id, year, day, p))
                            rows_eq.append((node_id, year, day, p))

    meta = {(n, y, d, p): m for n, y, d, p, m in rows_v}
    parent.interface_index = pe.Set(initialize=sorted(meta), dimen=4, ordered=True)

    def _tso_pf(t_blk, t_net, adn_load, p, kind):
        """Production's interface_pf_[pq]_transmission_def, W-space safe."""
        params = t_blk.params
        base = t_blk.pc if kind == 'p' else t_blk.qc
        value = base[adn_load, t_blk.s_m, t_blk.s_o, p]
        if params.fl_reg:
            up = t_blk.flex_p_up if kind == 'p' else t_blk.flex_q_up
            down = t_blk.flex_p_down if kind == 'p' else t_blk.flex_q_down
            value = value + (up[adn_load, t_blk.s_m, t_blk.s_o, p]
                             - down[adn_load, t_blk.s_m, t_blk.s_o, p])
        return value

    def _dso_pf(d_blk, ref_gen, sess_d, p, kind):
        """Production's interface_pf_[pq]_distribution_def."""
        gen = d_blk.pg if kind == 'p' else d_blk.qg
        net = d_blk.shared_es_pnet if kind == 'p' else d_blk.shared_es_qnet
        value = gen[ref_gen, d_blk.s_m, d_blk.s_o, p]
        for e in sess_d:
            value = value - net[e, d_blk.s_m, d_blk.s_o, p]
        return value

    def vmag_rule(m, n, y, d, p):
        (t, dd, adn, _l, ref, _g, kv_t, kv_d, _tn, _dn, _st, _sd, _p) = meta[(n, y, d, p)]
        return (t.vmag_sqr[adn, t.s_m, t.s_o, p] * kv_t ** 2
                == dd.vmag_sqr[ref, dd.s_m, dd.s_o, p] * kv_d ** 2)

    def pf_rule(m, n, y, d, p, kind):
        (t, dd, _a, load, _r, gen, _kt, _kd, tn, dn, _st, sd, _p) = meta[(n, y, d, p)]
        return (_tso_pf(t, tn, load, p, kind) * tn.baseMVA
                == _dso_pf(dd, gen, sd, p, kind) * dn.baseMVA)

    def ess_rule(m, n, y, d, p, kind):
        (t, dd, _a, _l, _r, _g, _kt, _kd, tn, dn, st, sd, _p) = meta[(n, y, d, p)]
        var_t = t.shared_es_pnet if kind == 'p' else t.shared_es_qnet
        var_d = dd.shared_es_pnet if kind == 'p' else dd.shared_es_qnet
        lhs = sum(var_t[e, t.s_m, t.s_o, p] for e in st) * tn.baseMVA
        rhs = sum(var_d[e, dd.s_m, dd.s_o, p] for e in sd) * dn.baseMVA
        return lhs == rhs

    parent.interface_vmag = pe.Constraint(parent.interface_index, rule=vmag_rule)
    parent.interface_pf_p = pe.Constraint(
        parent.interface_index, rule=lambda m, n, y, d, p: pf_rule(m, n, y, d, p, 'p'))
    parent.interface_pf_q = pe.Constraint(
        parent.interface_index, rule=lambda m, n, y, d, p: pf_rule(m, n, y, d, p, 'q'))
    parent.interface_ess_p = pe.Constraint(
        parent.interface_index, rule=lambda m, n, y, d, p: ess_rule(m, n, y, d, p, 'p'))
    parent.interface_ess_q = pe.Constraint(
        parent.interface_index, rule=lambda m, n, y, d, p: ess_rule(m, n, y, d, p, 'q'))


def model_size(parent):
    n_vars = sum(len(list(v.values())) for v in parent.component_objects(pe.Var, active=None,
                                                                        descend_into=True))
    n_cons = sum(len(list(c.values())) for c in parent.component_objects(pe.Constraint,
                                                                        active=True,
                                                                        descend_into=True))
    return {'variables': n_vars, 'constraints': n_cons, 'ac_blocks': len(parent.blocks)}
