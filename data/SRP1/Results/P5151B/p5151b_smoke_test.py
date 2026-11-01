import sys
import json
import pyomo.environ as pe
from copy import deepcopy

sys.path.insert(0, '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation')

from shared_resources_planning import SharedResourcesPlanning
from model_construction_helpers import configure_shared_ess_operational_state


def solver_ok(result):
    try:
        term = result['results'].solver.termination_condition
    except Exception:
        term = None
    return term


def run(tag):
    pp = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    pp.read_planning_problem()

    dso = pp.distribution_networks[7]
    tso = pp.transmission_network

    # Narrow to the preserved comparator fixtures to keep the smoke test cheap:
    # DSO case33_2 node 7, 2025 Autumn (converged comparator);
    # TSO case9, 2025 Summer (matches the preserved frozen TSO fixture).
    dso.years = {2025: dso.years[2025]}
    dso.days = {'Autumn': dso.days['Autumn']}
    tso.years = {2025: tso.years[2025]}
    tso.days = {'Summer': tso.days['Summer']}

    report = {'tag': tag}

    # ---- DSO ----
    dso_model = dso.build_model()
    m = dso_model[2025]['Autumn']
    # Exercise the shared-ESS families (Candidate 1 / Candidate 2) at positive capacity.
    s_base = dso.network[2025]['Autumn'].baseMVA
    for e in m.shared_energy_storages:
        s_cap_pu = 0.97 / s_base
        e_cap_pu = 1.94 / s_base
        m.shared_es_s_rated_fixed[e].set_value(s_cap_pu)
        m.shared_es_e_rated_fixed[e].set_value(e_cap_pu)
        configure_shared_ess_operational_state(m, e, s_cap_pu, e_cap_pu)
    report['dso_has_sess_phi_limit_lower'] = hasattr(m, 'sess_phi_limit_lower')
    report['dso_has_sess_phi_limit_upper'] = hasattr(m, 'sess_phi_limit_upper')
    report['dso_has_shared_es_s_rated_var'] = hasattr(m, 'shared_es_s_rated') and isinstance(getattr(m, 'shared_es_s_rated', None), pe.Var)
    report['dso_has_shared_energy_storage_s_sensitivities'] = hasattr(m, 'shared_energy_storage_s_sensitivities')
    report['dso_has_shared_energy_storage_e_sensitivities'] = hasattr(m, 'shared_energy_storage_e_sensitivities')
    dso_results = dso.optimize(dso_model)
    res = dso_results[2025]['Autumn']
    report['dso_termination'] = str(res.solver.termination_condition)
    report['dso_status'] = str(res.solver.status)
    report['dso_message'] = str(res.solver.message)
    report['dso_objective'] = pe.value(m.objective)
    report['dso_iterations'] = None
    try:
        report['dso_iterations'] = res.solver.statistics.iterations if hasattr(res.solver, 'statistics') else None
    except Exception:
        pass
    report['dso_has_scenario_deviation_penalty'] = hasattr(m, 'scenario_deviation_penalty')
    report['dso_num_scenarios_market'] = len(list(m.scenarios_market))
    report['dso_num_scenarios_operation'] = len(list(m.scenarios_operation))

    # ---- TSO ----
    tso_model = tso.build_model()
    mt = tso_model[2025]['Summer']
    s_base_t = tso.network[2025]['Summer'].baseMVA
    for e in mt.shared_energy_storages:
        s_cap_pu = 0.97 / s_base_t
        e_cap_pu = 1.94 / s_base_t
        mt.shared_es_s_rated_fixed[e].set_value(s_cap_pu)
        mt.shared_es_e_rated_fixed[e].set_value(e_cap_pu)
        configure_shared_ess_operational_state(mt, e, s_cap_pu, e_cap_pu)
    report['tso_has_sess_phi_limit_lower'] = hasattr(mt, 'sess_phi_limit_lower')
    report['tso_has_sess_phi_limit_upper'] = hasattr(mt, 'sess_phi_limit_upper')
    report['tso_has_shared_es_s_rated_var'] = hasattr(mt, 'shared_es_s_rated') and isinstance(getattr(mt, 'shared_es_s_rated', None), pe.Var)
    report['tso_has_shared_energy_storage_s_sensitivities'] = hasattr(mt, 'shared_energy_storage_s_sensitivities')
    report['tso_has_shared_energy_storage_e_sensitivities'] = hasattr(mt, 'shared_energy_storage_e_sensitivities')
    tso_results = tso.optimize(tso_model)
    rest = tso_results[2025]['Summer']
    report['tso_termination'] = str(rest.solver.termination_condition)
    report['tso_status'] = str(rest.solver.status)
    report['tso_message'] = str(rest.solver.message)
    report['tso_objective'] = pe.value(mt.objective)
    report['tso_has_scenario_deviation_penalty'] = hasattr(mt, 'scenario_deviation_penalty')
    report['tso_num_scenarios_market'] = len(list(mt.scenarios_market))
    report['tso_num_scenarios_operation'] = len(list(mt.scenarios_operation))

    return report


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
    rep = run(tag)
    outpath = f'/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151B/smoke_{tag}.json'
    with open(outpath, 'w') as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
