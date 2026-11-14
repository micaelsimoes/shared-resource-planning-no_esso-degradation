import sys
import re
import json
import glob
import os
import pyomo.environ as pe

sys.path.insert(0, '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation')

import network as network_module
import model_construction_helpers as mch
from shared_resources_planning import SharedResourcesPlanning


def old_sg_avail_rule(m, g, s_m, s_o, p, network, params):
    """Pre-Candidate-3' form, reconstructed for the A/B only (not written to
    any production file): pg^2 + qg^2 <= sg_avail^2, un-normalized."""
    gen = network.generators[g]
    if not gen.is_curtaillable() or mch.renewable_generation_is_unavailable(gen, s_o, p):
        return pe.Constraint.Skip
    return m.sg_sqr[g, s_m, s_o, p] <= m.sg_avail[g, s_o, p] ** 2


def latest_iteration_count(log_glob):
    candidates = sorted(glob.glob(log_glob), key=os.path.getmtime)
    if not candidates:
        return None, None
    path = candidates[-1]
    with open(path) as f:
        content = f.read()
    matches = re.findall(r'Number of Iterations\.*\s*:\s*(\d+)', content)
    if not matches:
        return None, path
    return int(matches[-1]), path


def build_and_solve(tag, use_new_rule):
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
    ok = str(r.solver.termination_condition) == 'optimal'

    log_glob = os.path.join(dso.network[2025]['Autumn'].logs_dir, 'optim_log_case33_2_2025_Autumn.log')
    iters, log_path = latest_iteration_count(log_glob)

    dispatch = {}
    for g in m.generators:
        for s_m in m.scenarios_market:
            for s_o in m.scenarios_operation:
                for p in m.periods:
                    dispatch[('pg', g, s_m, s_o, p)] = pe.value(m.pg[g, s_m, s_o, p])
                    dispatch[('qg', g, s_m, s_o, p)] = pe.value(m.qg[g, s_m, s_o, p])

    return {
        'tag': tag,
        'termination': str(r.solver.termination_condition),
        'objective': pe.value(m.objective),
        'iterations': iters,
        'log_path': log_path,
        'dispatch': dispatch,
    }


if __name__ == '__main__':
    before = build_and_solve('before_candidate3prime', use_new_rule=False)
    after = build_and_solve('after_candidate3prime', use_new_rule=True)

    max_abs_diff = 0.0
    worst_key = None
    for key in before['dispatch']:
        diff = abs(before['dispatch'][key] - after['dispatch'][key])
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_key = key

    report = {
        'before': {k: v for k, v in before.items() if k != 'dispatch'},
        'after': {k: v for k, v in after.items() if k != 'dispatch'},
        'max_abs_dispatch_diff': max_abs_diff,
        'worst_key': str(worst_key),
        'objective_diff': after['objective'] - before['objective'],
        'gate_1e-8_pass': max_abs_diff <= 1e-8,
    }
    print(json.dumps(report, indent=2, default=str))
    with open('/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151B/candidate3prime_ab.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
