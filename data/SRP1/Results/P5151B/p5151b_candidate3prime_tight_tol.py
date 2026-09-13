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


def build_and_solve_tight(use_new_rule, tol):
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

    network_obj = dso.network[2025]['Autumn']
    result, log_path = network_module._run_smopf_solver_attempt(
        network_obj, m, dso.params, from_warm_start=False,
        option_overrides={'tol': tol, 'acceptable_tol': tol, 'max_iter': 3000},
    )
    m.solutions.load_from(result)

    dispatch = {}
    for g in m.generators:
        for s_m in m.scenarios_market:
            for s_o in m.scenarios_operation:
                for p in m.periods:
                    dispatch[('pg', g, s_m, s_o, p)] = pe.value(m.pg[g, s_m, s_o, p])
                    dispatch[('qg', g, s_m, s_o, p)] = pe.value(m.qg[g, s_m, s_o, p])
    return {
        'termination': str(result.solver.termination_condition),
        'objective': pe.value(m.objective),
        'dispatch': dispatch,
    }


if __name__ == '__main__':
    tol = 1e-9
    before = build_and_solve_tight(False, tol)
    after = build_and_solve_tight(True, tol)
    max_abs_diff = 0.0
    worst_key = None
    for key in before['dispatch']:
        diff = abs(before['dispatch'][key] - after['dispatch'][key])
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_key = key
    report = {
        'tol': tol,
        'before_termination': before['termination'],
        'after_termination': after['termination'],
        'before_objective': before['objective'],
        'after_objective': after['objective'],
        'objective_diff': after['objective'] - before['objective'],
        'max_abs_dispatch_diff': max_abs_diff,
        'worst_key': str(worst_key),
    }
    print(json.dumps(report, indent=2, default=str))
    with open('/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151B/candidate3prime_tight_tol_diagnostic.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
