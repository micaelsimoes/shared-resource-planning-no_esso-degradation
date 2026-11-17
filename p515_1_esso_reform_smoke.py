"""
P5.15-1 -- ESSO reformulation smoke test (Step 1 of PLANNER_BRIEF_2026-09-13.md,
"Before reporting" instruction).

The P5.14-N pickles (`data/SRP1/Results/P514N/esso_models_k10000.pkl`) hold ESSO
models built by the OLD (pre-reformulation) `_build_subproblem`: they contain
`es_soh_per_unit`, `es_pch_hat_per_unit`, `slack_es_s_investment_up`, etc., none
of which exist on the reformulated model. Unpickling them and trying to solve
would either crash on load (Pyomo model components are reconstructed from the
pickled class' current code, not the code that built them) or silently test the
WRONG model. Per the task's explicit instruction, this harness therefore
REBUILDS the ESSO from a live planning problem via `p56a_oracle.fresh_planning`
(read-only import of production data) instead of unpickling.

What this smoke test exercises, concretely:
  - `SharedEnergyStorageData.build_subproblem()` on the full SRP1 case
    (production `_build_subproblem`, exercising every changed constraint family:
    investments-as-Params, the log-domain D/exp SoH chain, the linear SoH floor,
    the retired complementarity block, the new throughput regularization).
  - `create_shared_energy_storage_model(...)` (production function, unmodified),
    called with a real (non-pickled) `consensus_vars`/`candidate_solution` pair
    built from `create_admm_variables` + a hand-set nonzero charge/discharge
    request profile, so that pch/pdch are not trivially zero (a zero-request
    solve would not exercise the D-equation or the throughput penalty
    meaningfully). Candidates: node 7 and node 9 (the two Step-0 failing
    instances) get a non-zero (S, E) investment in the first representative
    year; all other active nodes get zero investment (inactive cohorts).
  - The post-solve complementarity detector
    (`SharedEnergyStorageData.get_complementarity_violation`).

This is a BUILD-AND-SOLVE smoke test, not a reproduction of the Step 0/1a
failing instances (those no longer exist in this model family) and not the
brief's G1-G5 gate runs (separate, not-yet-authorized task per the Planner).

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_1_esso_reform_smoke.py
"""

import json
import math
import os

import pyomo.environ as pe

import p56a_oracle as oracle
from shared_resources_planning import create_admm_variables, create_shared_energy_storage_model

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P5151')
os.makedirs(OUT_DIR, exist_ok=True)

SMOKE_NODES_WITH_INVESTMENT = (7, 9)
S_CANDIDATE_MVA = 1.00
E_CANDIDATE_MVAH = 2.00


def _zero_candidate(planning):
    years = list(planning.years)
    candidate = {}
    for node_id in planning.shared_ess_data.active_distribution_network_nodes:
        candidate[node_id] = {year: {'s': 0.00, 'e': 0.00} for year in years}
    return candidate


def _set_nonzero_charge_discharge_request(planning, consensus_vars, node_id, amplitude_mw):
    years = list(planning.years)
    days = list(planning.days)
    num_instants = planning.num_instants
    for year in years:
        for day in days:
            p_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['p']
            for p in range(num_instants):
                # First half of the day charges, second half discharges -- a
                # non-trivial throughput profile that forces the D-equation and
                # the throughput penalty to do real work, instead of the
                # trivially-feasible all-zero request `create_admm_variables`
                # initializes to.
                p_req[p] = amplitude_mw if p < num_instants // 2 else -amplitude_mw


def main():
    report = {}

    planning = oracle.fresh_planning('p5151_smoke')
    shared_ess_data = planning.shared_ess_data

    # ---- 1. Build (no candidate solution yet: exercises _build_subproblem alone)
    esso_models_fresh_build = shared_ess_data.build_subproblem()
    report['build_ok'] = True
    report['nodes_built'] = list(esso_models_fresh_build.keys())
    report['smoke_nodes'] = list(SMOKE_NODES_WITH_INVESTMENT)
    assert all(node in esso_models_fresh_build for node in SMOKE_NODES_WITH_INVESTMENT), (
        'Expected smoke nodes not present among the active distribution network nodes.'
    )

    # ---- 2. Build a real candidate solution + consensus_vars pair (production
    #         shapes, not reimplemented) and solve node 7 and node 9 through the
    #         production entry point `create_shared_energy_storage_model`.
    consensus_vars, dual_vars = create_admm_variables(planning)
    candidate_solution = _zero_candidate(planning)
    first_year = list(planning.years)[0]
    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        candidate_solution[node_id][first_year] = {'s': S_CANDIDATE_MVA, 'e': E_CANDIDATE_MVAH}
        # A gentle amplitude (10% of s_max, not 50%) is used deliberately: a
        # sustained +/-50% duty cycle every period of every representative day
        # for the whole 3-year horizon was tried first and drove the SoH floor
        # (soh_cumul >= soh_min) to bind by y=2, at which point the model
        # correctly falls back to the (heavily penalized) pnet slack rather
        # than violate the floor -- see P5_15_1_REPORT.md for that finding.
        # 10% keeps this smoke test inside the region where the floor does not
        # bind, so it demonstrates the ordinary (non-floor-limited) operating
        # regime cleanly.
        _set_nonzero_charge_discharge_request(
            planning, consensus_vars, node_id, amplitude_mw=0.10 * S_CANDIDATE_MVA
        )

    esso_models, esso_op_results = create_shared_energy_storage_model(
        shared_ess_data, consensus_vars, candidate_solution
    )

    report['solves'] = {}
    for node_id in shared_ess_data.active_distribution_network_nodes:
        result = esso_op_results.get(node_id)
        termination = str(result.solver.termination_condition) if result is not None else None
        objective = None
        max_throughput = None
        if node_id in esso_models:
            model = esso_models[node_id]
            try:
                objective = pe.value(model.objective)
            except Exception as exc:  # noqa: BLE001 -- report, don't hide
                objective = f'ERROR: {exc}'
        report['solves'][str(node_id)] = {
            'termination_condition': termination,
            'objective_feasibility_penalty': objective,
            'has_investment': node_id in SMOKE_NODES_WITH_INVESTMENT,
        }

    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        result = esso_op_results.get(node_id)
        assert result is not None, f'No solver result recorded for node {node_id}.'
        assert str(result.solver.termination_condition) in ('optimal', 'TerminationCondition.optimal'), (
            f'Node {node_id} did not solve to optimality: {result.solver.termination_condition}'
        )

    # ---- 3. Post-solve complementarity detector (Step 1 item 3)
    max_violation = shared_ess_data.get_complementarity_violation(esso_models)
    report['complementarity_detector_max_violation'] = max_violation

    # ---- 4. Report a few SoH-chain values for node 7 and node 9 (sanity, not a gate)
    report['soh_chain_sample'] = {}
    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        model = esso_models[node_id]
        y_inv = 0
        sample = []
        for y in model.years:
            if not model.es_s_rated_per_unit[y_inv, y].fixed:
                sample.append({
                    'y': y,
                    'D': pe.value(model.es_D_per_unit[y_inv, y]),
                    'soh_cumul': pe.value(model.es_soh_per_unit_cumul[y_inv, y]),
                    'avg_ch_dch_per_unit': pe.value(model.es_avg_ch_dch_per_unit[y_inv, y]),
                })
        report['soh_chain_sample'][str(node_id)] = sample

    with open(os.path.join(OUT_DIR, 'p5151_smoke_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print('\n[SMOKE] PASS' if all(
        str(esso_op_results[n].solver.termination_condition) in ('optimal', 'TerminationCondition.optimal')
        for n in SMOKE_NODES_WITH_INVESTMENT
    ) else '\n[SMOKE] FAIL')


if __name__ == '__main__':
    main()
