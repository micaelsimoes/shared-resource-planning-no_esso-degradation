"""
P5.15 Addendum 3 item 2 (H3 rule) -- multi-cohort SOLVE check
(PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Verification required" item 3:
"If a multi-cohort instance can be constructed AND solved cheaply, one solve
is authorized to confirm the split is pinned pro-rata.").

Reuses the EXACT same two-cohort node-7 instance as
`p515_h3_multicohort_construction_test.py` (S=1.00 MVA/E=2.00 MVAh at the
first representative year, S=0.50 MVA/E=1.00 MVAh at the second), through the
SAME production entry point every other P5.15 harness uses
(`create_shared_energy_storage_model`), with a non-trivial +/-10% duty-cycle
request (`_set_nonzero_charge_discharge_request`, reused from
`p515_1_esso_reform_smoke.py`) so pch/pdch are not trivially zero.

"One solve" here means one end-to-end construction+solve invocation of the
multi-cohort instance. `create_shared_energy_storage_model` solves EVERY
active node (5, 7, 9) in that one call -- nodes 5 and 9 carry zero investment
in this instance and solve trivially; node 7 is the multi-cohort instance of
interest. This is declared and GUARD-VERIFIED (bounded `SolveProfileGuard`,
expected count 3), not asserted, per CLAUDE.md's solve-claim rule.

CHECK. At the solved point, for node 7 at calendar years y=1 and y=2 (the
two-cohort years), confirm for EVERY (d, p):
    pnet_cohort[y_inv, y, d, p] == share[y_inv, y] * agg_pnet[y, d, p]
for BOTH cohorts (y_inv=0, whose row is ACTIVE, and y_inv=1, the omitted
cohort whose value is only IMPLIED by the aggregate identity, never directly
constrained). Confirming the omitted cohort's identity holds too is the
actual test of "the split is pinned pro-rata" -- it is exactly what the
N_active-1 row count is supposed to buy back from N.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_h3_multicohort_solve_check.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import p56a_oracle as oracle  # noqa: E402
from shared_resources_planning import (  # noqa: E402
    create_admm_variables,
    create_shared_energy_storage_model,
)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_1_esso_reform_smoke import _zero_candidate, _set_nonzero_charge_discharge_request  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151')
os.makedirs(OUT_DIR, exist_ok=True)

PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES = 3  # create_shared_energy_storage_model solves every active node (5, 7, 9)

TEST_NODE = 7
COHORT_0_S_MVA, COHORT_0_E_MVAH = 1.00, 2.00
COHORT_1_S_MVA, COHORT_1_E_MVAH = 0.50, 1.00


def main():
    with redirect_stdout(io.StringIO()):
        planning = oracle.fresh_planning('p515h3_multicohort_solve')
    shared_ess_data = planning.shared_ess_data
    years = list(planning.years)

    consensus_vars, dual_vars = create_admm_variables(planning)
    candidate_solution = _zero_candidate(planning)
    candidate_solution[TEST_NODE][years[0]] = {'s': COHORT_0_S_MVA, 'e': COHORT_0_E_MVAH}
    candidate_solution[TEST_NODE][years[1]] = {'s': COHORT_1_S_MVA, 'e': COHORT_1_E_MVAH}
    _set_nonzero_charge_discharge_request(
        planning, consensus_vars, TEST_NODE, amplitude_mw=0.10 * COHORT_0_S_MVA
    )

    arm_dir = os.path.join(OUT_DIR, 'h3_multicohort_solve_logs')
    os.makedirs(arm_dir, exist_ok=True)

    guard = SolveProfileGuard(PERMITTED, label='P5.15 H3 multi-cohort solve check').install()
    cwd_before = os.getcwd()
    try:
        os.chdir(arm_dir)
        with redirect_stdout(io.StringIO()):
            esso_models, esso_op_results = create_shared_energy_storage_model(
                shared_ess_data, consensus_vars, candidate_solution
            )
    finally:
        os.chdir(cwd_before)
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)

    model = esso_models[TEST_NODE]
    solver_result = esso_op_results.get(TEST_NODE)
    termination = str(solver_result.solver.termination_condition) if solver_result else None

    checks = []
    max_abs_error = 0.0
    for y in model.years:
        active_cohorts = [
            y_inv for y_inv in model.years
            if (not model._esso_cohort_inactive.get(y_inv, False))
        ]
        import shared_energy_storage_data as SED
        active_cohorts = [
            y_inv for y_inv in model.years
            if (not model._esso_cohort_inactive.get(y_inv, False))
            and SED._esso_cohort_pair_is_within_lifetime(model, y_inv, y)
        ]
        if len(active_cohorts) <= 1:
            continue
        for d in model.days:
            for p in model.periods:
                agg_pnet = sum(
                    pe.value(model.es_pch_per_unit[y_inv, y, d, p]) - pe.value(model.es_pdch_per_unit[y_inv, y, d, p])
                    for y_inv in model.years
                )
                for y_inv in active_cohorts:
                    share = pe.value(model.es_pnet_cohort_share_h3[y_inv, y])
                    pnet_cohort = (
                        pe.value(model.es_pch_per_unit[y_inv, y, d, p])
                        - pe.value(model.es_pdch_per_unit[y_inv, y, d, p])
                    )
                    predicted = share * agg_pnet
                    error = abs(pnet_cohort - predicted)
                    max_abs_error = max(max_abs_error, error)
                    checks.append({
                        'y_inv': y_inv, 'y': y, 'd': d, 'p': p,
                        'row_is_active_constraint': y_inv != max(active_cohorts),
                        'pnet_cohort': pnet_cohort,
                        'share': share,
                        'agg_pnet': agg_pnet,
                        'predicted_pnet_cohort': predicted,
                        'abs_error': error,
                    })

    report = {
        'stage': 'P5.15 Addendum 3 item 2 (H3 rule) -- multi-cohort SOLVE check',
        'authority': (
            'PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Verification required" item 3'
        ),
        'instance': {
            'node': TEST_NODE,
            'cohort_0_investment_year': years[0],
            'cohort_0_s_mva': COHORT_0_S_MVA,
            'cohort_0_e_mvah': COHORT_0_E_MVAH,
            'cohort_1_investment_year': years[1],
            'cohort_1_s_mva': COHORT_1_S_MVA,
            'cohort_1_e_mvah': COHORT_1_E_MVAH,
            'amplitude_mw_formula': '0.10 * COHORT_0_S_MVA',
        },
        'guard': {
            'permitted_call_sites': PERMITTED,
            'expected_solves_declared_in_advance': EXPECTED_SOLVES,
            'observed_counts': guard.counts,
            'permitted_sites_hit': guard.permitted_sites,
            'verify_failures': guard_failures,
        },
        'termination_condition_node_7': termination,
        'objective_node_7': pe.value(model.objective),
        'n_pro_rata_identity_checks': len(checks),
        'max_abs_error_pro_rata_identity': max_abs_error,
        'pro_rata_identity_holds_to_1e-9': max_abs_error < 1e-9,
        'checks_sample': checks[:8],
    }

    summary_path = os.path.join(OUT_DIR, 'h3_multicohort_solve_check_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    full_path = os.path.join(OUT_DIR, 'h3_multicohort_solve_check_full.json')
    with open(full_path, 'w') as fh:
        json.dump({'checks': checks}, fh, indent=2, default=str)

    print(json.dumps({k: v for k, v in report.items() if k != 'checks_sample'}, indent=2, default=str))
    print(f'\n[P5.15-H3-solve] summary written to {summary_path}')
    print(f'[P5.15-H3-solve] full per-check data written to {full_path}')
    print(f'[P5.15-H3-solve] guard verify failures: {guard_failures}')


if __name__ == '__main__':
    main()
