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

    results = dso.optimize(model)
    r = results[2025]['Autumn']

    g, p = 0, 17
    diag = {}
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            pg = pe.value(m.pg[g, s_m, s_o, p])
            qg = pe.value(m.qg[g, s_m, s_o, p])
            pg_avail = pe.value(m.pg_avail[g, s_o, p])
            sg_avail = pe.value(m.sg_avail[g, s_o, p])
            margin = pg_avail**2 - (pg**2 + qg**2)
            diag[(s_m, s_o)] = dict(pg=pg, qg=qg, pg_avail=pg_avail, sg_avail=sg_avail, margin=margin)
            for gg in m.generators:
                is_curt = dso.network[2025]['Autumn'].generators[gg].is_curtaillable()
                if is_curt:
                    diag.setdefault('curtailable_gens', []).append(gg)
    return diag, pe.value(m.objective)


if __name__ == '__main__':
    d_before, obj_before = build_and_solve(False)
    d_after, obj_after = build_and_solve(True)
    print('BEFORE', json.dumps({str(k): v for k, v in d_before.items()}, indent=2, default=str))
    print('AFTER', json.dumps({str(k): v for k, v in d_after.items()}, indent=2, default=str))
