"""
P5.15-1 -- SoH-floor stress probe (diagnostic, not a gate).

While building `p515_1_esso_reform_smoke.py`, a synthetic +/-50% duty-cycle
request (every period, every representative day, all 3 years) was tried first
and produced a striking result: the model satisfies the CHARGE half of the
cycle with real `pch` throughput, then falls back almost entirely to the
(heavily penalized) `slack_es_pnet_down` for the DISCHARGE half, rather than
using `pdch`. This script isolates and records that finding as evidence,
separate from the "clean" smoke test (which deliberately uses a gentler 10%
amplitude that stays inside the region where the floor does not bind).

Diagnosis (recorded here, not asserted as proven): the log-domain SoH floor
(`es_soh_per_unit_cumul[y_inv, y] >= soh_min`) constrains TOTAL throughput
(`avg_ch_dch_per_unit`, to which BOTH pch and pdch contribute positively --
throughput, not net energy). At a sustained +/-50% duty cycle, the charge half
alone already drives the cohort's cumulative SoH down to soh_min by the last
representative year; the model has no remaining "throughput budget" to also
serve the discharge half without violating the floor, so it satisfies the
discharge request via the (expensive but always-feasible) operation slack
instead. Given `slack_es_pnet_up` / `slack_es_pnet_down` are symmetric in cost,
and the achievable split of "how much of the total throughput budget goes to
charge vs. discharge" is otherwise unconstrained by anything else in the model,
there may be a FLAT direction (multiple equally-optimal splits) -- this probe
does not attempt to resolve that; it is offered as an unexpected finding for
the Planner, not a claim of a bug in the reformulation (the SoH floor doing
exactly this -- refusing throughput once soh_min would be violated, and
falling back to the always-feasible penalty slack -- is the intended,
authorized design per PLANNER_BRIEF_2026-09-13.md's Governing decisions: "the
only intertemporal trade-off enforced in operation is the SoH floor").

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_1_floor_stress_probe.py
"""

import json
import os

import pyomo.environ as pe

import p56a_oracle as oracle
from shared_resources_planning import create_admm_variables, create_shared_energy_storage_model

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P5151')
NODE_ID = 7
S_CANDIDATE_MVA = 1.00
E_CANDIDATE_MVAH = 2.00
AMPLITUDE_MW = 0.50 * S_CANDIDATE_MVA  # 50% duty cycle, deliberately aggressive


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    planning = oracle.fresh_planning('p5151_floor_stress')
    shared_ess_data = planning.shared_ess_data
    years = list(planning.years)

    consensus_vars, _ = create_admm_variables(planning)
    candidate_solution = {
        node_id: {year: {'s': 0.00, 'e': 0.00} for year in years}
        for node_id in shared_ess_data.active_distribution_network_nodes
    }
    candidate_solution[NODE_ID][years[0]] = {'s': S_CANDIDATE_MVA, 'e': E_CANDIDATE_MVAH}

    for year in years:
        for day in planning.days:
            p_req = consensus_vars['ess']['tso']['current'][NODE_ID][year][day]['p']
            for p in range(planning.num_instants):
                p_req[p] = AMPLITUDE_MW if p < planning.num_instants // 2 else -AMPLITUDE_MW

    esso_models, esso_results = create_shared_energy_storage_model(
        shared_ess_data, consensus_vars, candidate_solution
    )
    model = esso_models[NODE_ID]
    result = esso_results[NODE_ID]

    report = {
        'node_id': NODE_ID,
        's_candidate_mva': S_CANDIDATE_MVA,
        'e_candidate_mvah': E_CANDIDATE_MVAH,
        'amplitude_mw': AMPLITUDE_MW,
        'termination_condition': str(result.solver.termination_condition),
        'objective_feasibility_penalty': pe.value(model.objective),
        'soh_chain': [
            {
                'y': y,
                'D': pe.value(model.es_D_per_unit[0, y]),
                'soh_cumul': pe.value(model.es_soh_per_unit_cumul[0, y]),
                'avg_ch_dch_per_unit': pe.value(model.es_avg_ch_dch_per_unit[0, y]),
            }
            for y in model.years if not model.es_s_rated_per_unit[0, y].fixed
        ],
        'operation_sample_y0_d0': [
            {
                'p': p,
                'pnet_requested': pe.value(model.es_pnet[0, 0, p]),
                'pch': pe.value(model.es_pch_per_unit[0, 0, 0, p]),
                'pdch': pe.value(model.es_pdch_per_unit[0, 0, 0, p]),
                'slack_pnet_up': pe.value(model.slack_es_pnet_up[0, 0, p]),
                'slack_pnet_down': pe.value(model.slack_es_pnet_down[0, 0, p]),
            }
            for p in model.periods
        ],
        'complementarity_detector_max_violation': shared_ess_data.get_complementarity_violation(esso_models),
    }

    out_path = os.path.join(OUT_DIR, 'p5151_floor_stress_probe.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
