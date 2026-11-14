import sys
import json
import pyomo.environ as pe

sys.path.insert(0, '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation')

import network as network_module
import model_construction_helpers as mch
from shared_resources_planning import SharedResourcesPlanning


def old_sg_avail_rule(m, g, s_m, s_o, p, network, params):
    gen = network.generators[g]
    if not gen.is_curtaillable() or mch.renewable_generation_is_unavailable(gen, s_o, p):
        return pe.Constraint.Skip
    return m.sg_sqr[g, s_m, s_o, p] <= m.sg_avail[g, s_o, p] ** 2


def build_and_solve(use_new_rule):
    if use_new_rule:
        network_module.sg_avail_rule = mch.sg_avail_rule
    else:
        network_module.sg_avail_rule = old_sg_avail_rule

    pp = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    pp.read_planning_problem()
    dso = pp.distribution_networks[7]
    dso.years = {2025: dso.years[2025]}
    dso.days = {'Autumn': dso.days['Autumn']}

    model = dso.build_model()
    m = model[2025]['Autumn']
    s_base = dso.network[2025]['Autumn'].baseMVA
    for e in m.shared_energy_storages:
        s_cap_pu = 0.97 / s_base
        e_cap_pu = 1.94 / s_base
        m.shared_es_s_rated_fixed[e].set_value(s_cap_pu)
        m.shared_es_e_rated_fixed[e].set_value(e_cap_pu)
        mch.configure_shared_ess_operational_state(m, e, s_cap_pu, e_cap_pu)

    dso.optimize(model)

    net = dso.network[2025]['Autumn']
    per_gen_max_diff = {}
    curtailable = {}
    for g in m.generators:
        curtailable[g] = net.generators[g].is_curtaillable()
    return m, curtailable


if __name__ == '__main__':
    m_before, curt = build_and_solve(False)
    m_after, _ = build_and_solve(True)

    rows = []
    for g in m_before.generators:
        for p in m_before.periods:
            pg_b = pe.value(m_before.pg[g, 0, 0, p])
            qg_b = pe.value(m_before.qg[g, 0, 0, p])
            pg_a = pe.value(m_after.pg[g, 0, 0, p])
            qg_a = pe.value(m_after.qg[g, 0, 0, p])
            diff = max(abs(pg_a - pg_b), abs(qg_a - qg_b))
            rows.append((g, p, curt[g], diff, pg_b, pg_a))

    rows.sort(key=lambda r: -r[3])
    for r in rows[:10]:
        print(f'gen={r[0]} curtailable={r[2]} period={r[1]} max_diff={r[3]:.6e} pg_before={r[4]:.6e} pg_after={r[5]:.6e}')
