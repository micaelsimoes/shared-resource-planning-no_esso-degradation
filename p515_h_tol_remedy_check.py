"""
P5.15 Addendum 3 item 1 -- ESSO `tol`/`acceptable_tol` remedy (h) verification
(PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Authorized" 1, "Verification
required").

Two direct ESSO solves of the SAME reformulated instance the P5.15-1b eps
check used, differing ONLY in the ESSO's IPOPT `tol`/`acceptable_tol`:

    | arm        | tol   | acceptable_tol |
    |------------|-------|----------------|
    | control    | 1e-6  | 1e-5           |
    | remedy_h   | 1e-8  | 1e-7           |

INSTANCE. Reuses VERBATIM the construction path of `p515_1b_eps_sensitivity_check.py`
(itself reusing `p515_1_esso_reform_smoke.py`, commit `b03c9b14`): a fresh
production planning problem (`p56a_oracle.fresh_planning`), a non-zero
investment candidate at nodes 7 and 9 (`SMOKE_NODES_WITH_INVESTMENT`,
S=1.00 MVA / E=2.00 MVAh in the first representative year), and the same
non-trivial +/-10% duty-cycle `p_req` profile
(`_set_nonzero_charge_discharge_request`, amplitude = 0.10 * S). Every one of
these building blocks is IMPORTED from `p515_1_esso_reform_smoke.py` /
`p515_1b_eps_sensitivity_check.py`, not reimplemented.

EPS_ESSO_THROUGHPUT is left at the case-file default (1e-3, the value G1-G5
are gated on per Addendum 2/3) for BOTH arms -- this task varies `tol` only.

TOL OVERRIDE. `ESSO_TOL_OVERRIDES` (Item 1's production default, applied in
`shared_energy_storage_data.py:optimize()`) is monkeypatched on the module
before each arm's solve and restored immediately afterward (verified: an
assertion re-reads `SED.ESSO_TOL_OVERRIDES` after every arm and fails loudly
if it is not the shipped production default {'tol': 1e-8, 'acceptable_tol':
1e-7}). `shared_energy_storage_data.py` is never edited on disk by this
harness.

DIAGNOSTICS. Uses the PRODUCTION functions added by this same task
(`shared_energy_storage_data._get_esso_complementarity_diagnostics`,
`._complementarity_ratio_for_model`, `._parse_ipopt_barrier_terms`) to compute
the ratio detector, parse mu_final/s_obj from each arm's own IPOPT log, and
compute the closed-form spurious-throughput bound -- not reimplemented here,
per CLAUDE.md ("use real production functions rather than reimplementing them
in diagnostics").

GUARD. A `SolveProfileGuard` is armed for the whole run, permitting solves
only from `shared_energy_storage_data.py:_run_solver_attempt`. Declared count
in advance: 6 (2 arms x 3 active nodes -- 5, 7, 9). Node 5 carries zero
investment (trivial solve); nodes 7 and 9 are the instances of interest. If a
recovery retry fires on either arm, the observed count will exceed 6; that is
reported as an extra, explicitly declared solve per node, not hidden.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_h_tol_remedy_check.py
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import definitions  # noqa: E402
import p56a_oracle as oracle  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
from shared_resources_planning import (  # noqa: E402
    create_admm_variables,
    create_shared_energy_storage_model,
)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_1_esso_reform_smoke import (  # noqa: E402
    SMOKE_NODES_WITH_INVESTMENT,
    S_CANDIDATE_MVA,
    E_CANDIDATE_MVAH,
    _zero_candidate,
    _set_nonzero_charge_discharge_request,
)
from p515_1b_eps_sensitivity_check import parse_ipopt_log, sha256_of  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151')
LOG_ROOT = os.path.join(OUT_DIR, 'tol_check_logs')
os.makedirs(OUT_DIR, exist_ok=True)

PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES = 6  # 2 arms x 3 active nodes, declared in advance (see docstring)

ARMS = (
    ('control', {'tol': 1e-6, 'acceptable_tol': 1e-5}),
    ('remedy_h', {'tol': 1e-8, 'acceptable_tol': 1e-7}),
)

PRODUCTION_DEFAULT_TOL_OVERRIDES = {'tol': 1e-8, 'acceptable_tol': 1e-7}
CASE_DEFAULT_EPS = 1e-3  # unchanged in both arms -- must be unchanged after every arm

REPORT = {
    'stage': 'P5.15 Addendum 3 item 1',
    'objective': (
        'Remedy (h) verification: two direct ESSO solves of the same reformulated '
        'node-7/9 instance the P5.15-1b eps check used, differing only in the ESSO '
        'IPOPT tol/acceptable_tol (control 1e-6/1e-5 vs remedy (h) 1e-8/1e-7).'
    ),
    'authority': 'PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Authorized" item 1 / "Verification required"',
    'construction_commit': 'b03c9b14',
    'construction_source': (
        'p515_1_esso_reform_smoke.py + p515_1b_eps_sensitivity_check.py (imported, not reimplemented)'
    ),
    'capture_path_notes': {
        'gross_operational_cost': (
            'NOT CAPTURED, same reasoning as P5.15-1b: create_shared_energy_storage_model '
            'never builds the TSO/DSO SMOPFs; only the ESSO objective (== feasibility_penalty) '
            'is reported.'
        ),
    },
}


def sha256_of_dict(payload_obj):
    return sha256_of(payload_obj)


def run_arm(tag, tol_overrides):
    arm_dir = os.path.join(LOG_ROOT, tag)
    os.makedirs(arm_dir, exist_ok=True)

    with redirect_stdout(io.StringIO()):
        planning = oracle.fresh_planning(f'p515h_tol_{tag}')
    shared_ess_data = planning.shared_ess_data

    consensus_vars, dual_vars = create_admm_variables(planning)
    candidate_solution = _zero_candidate(planning)
    first_year = list(planning.years)[0]
    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        candidate_solution[node_id][first_year] = {'s': S_CANDIDATE_MVA, 'e': E_CANDIDATE_MVAH}
        _set_nonzero_charge_discharge_request(
            planning, consensus_vars, node_id, amplitude_mw=0.10 * S_CANDIDATE_MVA
        )

    # Harness override, NOT a production-code edit: reassigns the module-level
    # constant `_optimize` reads through `optimize()`. Restored in `finally`.
    original_overrides = SED.ESSO_TOL_OVERRIDES
    SED.ESSO_TOL_OVERRIDES = dict(tol_overrides)

    # EPS_ESSO_THROUGHPUT is left untouched (case default 1e-3) in both arms.
    assert SED.EPS_ESSO_THROUGHPUT == CASE_DEFAULT_EPS

    cwd_before = os.getcwd()
    started = time.time()
    try:
        os.chdir(arm_dir)
        with redirect_stdout(io.StringIO()):
            esso_models, esso_op_results = create_shared_energy_storage_model(
                shared_ess_data, consensus_vars, candidate_solution
            )
    finally:
        os.chdir(cwd_before)
        SED.ESSO_TOL_OVERRIDES = original_overrides
    wall = time.time() - started

    assert SED.ESSO_TOL_OVERRIDES == PRODUCTION_DEFAULT_TOL_OVERRIDES, (
        f'SED.ESSO_TOL_OVERRIDES not restored after arm {tag}: {SED.ESSO_TOL_OVERRIDES}'
    )
    assert definitions.EPS_ESSO_THROUGHPUT == CASE_DEFAULT_EPS

    arm = {
        'tag': tag,
        'tol_overrides_in_force': tol_overrides,
        'wall_seconds': wall,
        'nodes': {},
    }

    for node_id in shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node_id]
        result = esso_op_results.get(node_id)
        termination = str(result.solver.termination_condition) if result is not None else None
        status = str(result.solver.status) if result is not None else None

        primary_log = os.path.join(arm_dir, f'optim_log_node_{node_id}.txt')
        recovery_log = os.path.join(arm_dir, f'optim_log_node_{node_id}_recovery.txt')
        recovery_fired = os.path.exists(recovery_log)
        log_tail = parse_ipopt_log(primary_log)
        recovery_log_tail = parse_ipopt_log(recovery_log) if recovery_fired else None
        diagnostics_log_path = recovery_log if recovery_fired else primary_log

        # PRODUCTION function, not reimplemented (CLAUDE.md).
        diagnostics = SED._get_esso_complementarity_diagnostics(model, node_id, diagnostics_log_path)

        try:
            objective = pe.value(model.objective)
        except Exception as exc:  # noqa: BLE001 -- report, do not hide
            objective = f'ERROR: {exc}'

        arm['nodes'][node_id] = {
            'termination_condition': termination,
            'status': status,
            'objective': objective,
            'log_tail_primary': log_tail,
            'recovery_fired': recovery_fired,
            'log_tail_recovery': recovery_log_tail,
            'diagnostics': diagnostics,
        }

    arm['recovery_diagnostics'] = list(shared_ess_data.solver_recovery_diagnostics)
    arm['esso_complementarity_diagnostics_sink'] = list(shared_ess_data.esso_complementarity_diagnostics)
    arm['total_objective_all_active_nodes'] = shared_ess_data.get_primal_value(esso_models)
    arm['total_feasibility_penalty_all_active_nodes'] = shared_ess_data.get_feasibility_penalty(esso_models)
    arm['complementarity_detector_absolute_max_violation'] = shared_ess_data.get_complementarity_violation(esso_models)
    arm['complementarity_detector_ratio_max_violation'] = shared_ess_data.get_complementarity_violation_ratio(esso_models)

    return arm


def _serializable(arm):
    out = dict(arm)
    out['nodes'] = {str(k): v for k, v in arm['nodes'].items()}
    return out


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)

    baseline = oracle.load_baseline()
    instance_descriptor = {
        'construction_commit': 'b03c9b14',
        'construction_module': 'p515_1_esso_reform_smoke.py + p515_1b_eps_sensitivity_check.py',
        'baseline_checksum': baseline['checksum'],
        'smoke_nodes_with_investment': list(SMOKE_NODES_WITH_INVESTMENT),
        's_candidate_mva': S_CANDIDATE_MVA,
        'e_candidate_mvah': E_CANDIDATE_MVAH,
        'amplitude_mw_formula': '0.10 * S_CANDIDATE_MVA',
        'eps_esso_throughput_both_arms': CASE_DEFAULT_EPS,
        'primary_nodes_for_this_task': [7, 9],
    }
    instance_hash = sha256_of_dict(instance_descriptor)
    REPORT['instance'] = instance_descriptor
    REPORT['instance_hash_sha256'] = instance_hash

    guard = SolveProfileGuard(PERMITTED, label='P5.15 Addendum 3 item 1 tol remedy check').install()
    try:
        arm_reports = {}
        for tag, tol_overrides in ARMS:
            print(f'[P5.15-H] running arm {tag} (tol_overrides={tol_overrides}) ...', flush=True)
            arm_reports[tag] = run_arm(tag, tol_overrides)
            for node_id, node_data in arm_reports[tag]['nodes'].items():
                diag = node_data['diagnostics']
                print(
                    f'  node {node_id}: {node_data["termination_condition"]} '
                    f'iters={node_data["log_tail_primary"].get("n_iterations")} '
                    f'recovery_fired={node_data["recovery_fired"]} '
                    f'ratio_max={diag["complementarity_ratio_max"]:.6e} '
                    f'mu_final={diag["mu_final"]} s_obj={diag["s_obj"]} '
                    f'bound={diag["spurious_throughput_bound"]} '
                    f'measured={diag["spurious_throughput_measured"]:.6e}',
                    flush=True,
                )
    finally:
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)
    REPORT['guard'] = {
        'permitted_call_sites': PERMITTED,
        'expected_solves_declared_in_advance': EXPECTED_SOLVES,
        'observed_counts': guard.counts,
        'permitted_sites_hit': guard.permitted_sites,
        'verify_failures': guard_failures,
    }

    arm_control = arm_reports['control']
    arm_h = arm_reports['remedy_h']

    def _node_table(arm):
        table = {}
        for node_id, node_data in arm['nodes'].items():
            diag = node_data['diagnostics']
            table[str(node_id)] = {
                'termination_condition': node_data['termination_condition'],
                'n_iterations_primary': node_data['log_tail_primary'].get('n_iterations'),
                'recovery_fired': node_data['recovery_fired'],
                'n_iterations_recovery': (node_data['log_tail_recovery'] or {}).get('n_iterations'),
                'mu_final': diag['mu_final'],
                's_obj': diag['s_obj'],
                'parse_reason': diag['parse_reason'],
                'complementarity_ratio_max': diag['complementarity_ratio_max'],
                'n_active_cohort_periods': diag['n_active_cohort_periods'],
                'spurious_throughput_bound': diag['spurious_throughput_bound'],
                'spurious_throughput_measured': diag['spurious_throughput_measured'],
                'measured_to_bound_ratio': (
                    diag['spurious_throughput_measured'] / diag['spurious_throughput_bound']
                    if diag['spurious_throughput_bound'] not in (None, 0.0) else None
                ),
                'objective': node_data['objective'],
            }
        return table

    control_table = _node_table(arm_control)
    remedy_table = _node_table(arm_h)

    predeclared = {
        'expected_ratio_at_tol_1e-8': '~5e-6',
        'expected_spurious_throughput_fraction_at_tol_1e-8': '~0.01% of fixture total',
    }

    bound_falsified = []
    for node_id, node_data in remedy_table.items():
        bound = node_data['spurious_throughput_bound']
        measured = node_data['spurious_throughput_measured']
        if bound is not None and measured > bound:
            bound_falsified.append({'node_id': node_id, 'measured': measured, 'bound': bound})
    for node_id, node_data in control_table.items():
        bound = node_data['spurious_throughput_bound']
        measured = node_data['spurious_throughput_measured']
        if bound is not None and measured > bound:
            bound_falsified.append({'node_id': node_id, 'arm': 'control', 'measured': measured, 'bound': bound})

    REPORT['results'] = {
        'control_tol_1e-6_acceptable_1e-5': control_table,
        'remedy_h_tol_1e-8_acceptable_1e-7': remedy_table,
        'predeclared_expectation': predeclared,
        'bound_falsified_nodes': bound_falsified,
        'total_objective_all_active_nodes': {
            'control': arm_control['total_objective_all_active_nodes'],
            'remedy_h': arm_h['total_objective_all_active_nodes'],
        },
        'recovery_diagnostics': {
            'control': arm_control['recovery_diagnostics'],
            'remedy_h': arm_h['recovery_diagnostics'],
        },
    }

    full_path = os.path.join(OUT_DIR, 'tol_remedy_check_full.json')
    with open(full_path, 'w') as fh:
        json.dump({
            'meta': {k: v for k, v in REPORT.items() if k not in ('results',)},
            'arms_full': {tag: _serializable(arm_reports[tag]) for tag in arm_reports},
        }, fh, indent=2, default=str)

    summary_path = os.path.join(OUT_DIR, 'tol_remedy_check_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(REPORT, fh, indent=2, default=str)

    print(json.dumps(REPORT, indent=2, default=str))
    print(f'\n[P5.15-H] full per-node data written to {full_path}')
    print(f'[P5.15-H] summary written to {summary_path}')
    print(f'\n[P5.15-H] guard verify failures: {guard_failures}')
    print(f'[P5.15-H] bound_falsified_nodes: {bound_falsified}')


if __name__ == '__main__':
    main()
