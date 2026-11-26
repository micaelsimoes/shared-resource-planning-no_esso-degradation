"""
P5.15 Gate G5 -- reformulated-ESSO A1/A3/A4 agreement gate.

Authority: `PLANNER_BRIEF_2026-09-13.md`, Addendum 1 ("Gate G5 (new)"):

    "on the reformulated ESSO, re-solve the node-7 and node-9 instances under
    A1, A3 and A4. All three must agree to 1e-6 relative. Disagreement means
    the reformulation left a nonconvexity and Step 1 does not close until it
    is identified."

amended by Addendum 3 (remedy (h): ESSO `tol = 1e-8`, `acceptable_tol = 1e-7`,
applied via `SED.ESSO_TOL_OVERRIDES` -- already the production default in the
current working tree, see `WORKER_REPORT_H.md`) and by the H3 cohort-split
rule (`WORKER_REPORT_H3.md`, inert on this single-cohort fixture).

INSTANCE. Reuses the SAME reformulated node-7/9 fixture every P5.15 stage-1
diagnostic since the smoke test has used (`p515_1_esso_reform_smoke.py`,
`p515_1b_eps_sensitivity_check.py`, `p515_h_tol_remedy_check.py`): a fresh
production planning problem (`p56a_oracle.fresh_planning`), a non-zero
investment candidate at nodes 7 and 9 (`SMOKE_NODES_WITH_INVESTMENT`,
S=1.00 MVA / E=2.00 MVAh in the first representative year, all other active
nodes zero), and the same non-trivial +/-10% duty-cycle `p_req` profile
(`_set_nonzero_charge_discharge_request`, amplitude = 0.10 * S). This is the
right instance for G5 because it is the only reformulated ESSO fixture this
stage has exercised and validated (Step 1 smoke test, the eps check, the
tol-remedy check): reusing it means G5 tests the SAME model/candidate the
rest of P5.15 has been measuring, not a new one. Construction is IMPORTED
from `p515_1_esso_reform_smoke.py`, not reimplemented.

WHY THREE INDEPENDENT SOLVES PER NODE, NOT A CHAIN. Each arm (A1, A3, A4)
solves an independent `copy.deepcopy` of the SAME freshly-built,
NEVER-YET-SOLVED per-node model (built once via the production
`build_subproblem` / `update_model_with_candidate_solution` / `fix_or_set`
sequence, mirroring `create_shared_energy_storage_model`'s own pre-solve
steps). Only the solver policy differs between arms (`from_warm_start` and
`option_overrides`), per the brief's own arm table ("only solver options
differ"). Chaining A1/A3 from A4's own converged output was considered and
rejected: it would warm-start from an already-optimal point, which converges
trivially under any policy and would not test whether the reformulated ESSO
has multiple local optima reachable by different solver policies -- the
actual question G5 asks.

ARMS (verbatim from `PLANNER_BRIEF_2026-09-13.md` Step 0's table, reused by
Addendum 1's G5 dispatch):
  A1 -- warm start; `warm_start_mult_bound_push=1e-3`, `warm_start_bound_push=1e-3`,
        `warm_start_slack_bound_push=1e-3`; `_frac` options at IPOPT defaults.
  A3 -- A1 + `mu_strategy=adaptive`.
  A4 -- cold start (`warm_start_init_point=no`, no suffix export).
`SED.ESSO_TOL_OVERRIDES` (remedy (h), `tol=1e-8`/`acceptable_tol=1e-7`) is
applied to EVERY arm (it is the shipped production default in this working
tree, applied through `option_overrides` exactly as production's own
`optimize()` does -- this harness calls `SED._optimize` directly instead of
`optimize()` only because `optimize()` hardcodes ONE option_overrides dict
per call and offers no per-arm hook; the tol values are held IDENTICAL across
arms so tol is not a confound between arms).

DISCRIMINATOR (predeclared in the dispatching task). For every arm pair on
every node: is the observed objective difference explained by the
already-diagnosed barrier-complementarity leak, propagated through
`EPS_ESSO_THROUGHPUT * delta(spurious_throughput_measured)`? The
`feasibility_penalty` term this bears on is
`EPS_ESSO_THROUGHPUT * sum(pch + pdch)` over active cohort-periods, and
`sum(pch+pdch) = true_throughput + spurious_throughput_measured` (the leak
IS the extra throughput the barrier admits at the interior point,
`spurious_throughput_measured = 2 * sum(min(pch,pdch))`,
`SED._complementarity_ratio_for_model` / `_get_esso_complementarity_diagnostics`,
already computed per solve). So for a fixed dispatch/investment (both held
fixed by construction: candidate solution and `p_req`/`q_req` are identical
across arms), `predicted_diff = EPS_ESSO_THROUGHPUT * (spurious_i - spurious_j)`
is the leak's own prediction for `objective_i - objective_j`. Reported for
every pair, per node, alongside the observed difference.

GUARD. `SolveProfileGuard` armed for the whole run, permitting solves only
from `shared_energy_storage_data.py:_run_solver_attempt`. Declared count in
advance: 6 (2 nodes x 3 arms). If a recovery retry fires on any arm, the
observed count exceeds 6 -- reported explicitly (see "maxIterations /
recovery events" in the summary), not hidden or silently re-declared.

OUTPUT. Only new files under `data/SRP1/Results/P5151/` (asserted before
writing that none of them pre-exist -- this harness must not be re-run
without renaming, matching the "no clobbering a cited artifact" rule):
  - g5_gate_summary.json
  - g5_gate_full.json
  - g5_gate_logs/node{7,9}_{A1,A3,A4}[_recovery].txt

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_g5_reformulated_gate.py
"""

import copy
import hashlib
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

import p56a_oracle as oracle  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
from helper_functions import fix_or_set, solver_result_succeeded  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_0_esso_warmstart_ab import max_constraint_violation, parse_ipopt_log_tail  # noqa: E402 -- reused, not reimplemented
from p515_1_esso_reform_smoke import (  # noqa: E402
    SMOKE_NODES_WITH_INVESTMENT,
    S_CANDIDATE_MVA,
    E_CANDIDATE_MVAH,
    _zero_candidate,
    _set_nonzero_charge_discharge_request,
)
from p515_1b_eps_sensitivity_check import sha256_of  # noqa: E402
from shared_resources_planning import create_admm_variables  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151')
LOG_ROOT = os.path.join(OUT_DIR, 'g5_gate_logs')

SUMMARY_PATH = os.path.join(OUT_DIR, 'g5_gate_summary.json')
FULL_PATH = os.path.join(OUT_DIR, 'g5_gate_full.json')

GATE_NODES = (7, 9)
ARM_NAMES = ('A1', 'A3', 'A4')

ARM_SPECS = {
    'A1': dict(
        from_warm_start=True,
        extra_overrides={
            'warm_start_mult_bound_push': 1e-3,
            'warm_start_bound_push': 1e-3,
            'warm_start_slack_bound_push': 1e-3,
        },
        description='warm start, pushes=1e-3, frac at IPOPT defaults',
    ),
    'A3': dict(
        from_warm_start=True,
        extra_overrides={
            'warm_start_mult_bound_push': 1e-3,
            'warm_start_bound_push': 1e-3,
            'warm_start_slack_bound_push': 1e-3,
            'mu_strategy': 'adaptive',
        },
        description='A1 + mu_strategy=adaptive',
    ),
    'A4': dict(
        from_warm_start=False,
        extra_overrides={'warm_start_init_point': 'no'},
        description='cold start',
    ),
}

PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES = len(GATE_NODES) * len(ARM_NAMES)  # 2 nodes x 3 arms = 6, declared in advance

RELATIVE_AGREEMENT_TOLERANCE = 1e-6  # the gate's own threshold, per the brief


def _assert_capture_paths_exist():
    """Rule-eight preflight: fail fast if a quantity the gate requires has no
    capture path, BEFORE any solve runs."""
    problems = []
    if not hasattr(SED, 'ESSO_TOL_OVERRIDES'):
        problems.append('SED.ESSO_TOL_OVERRIDES missing (remedy (h) not in working tree)')
    elif SED.ESSO_TOL_OVERRIDES != {'tol': 1e-8, 'acceptable_tol': 1e-7}:
        problems.append(f'SED.ESSO_TOL_OVERRIDES unexpected: {SED.ESSO_TOL_OVERRIDES}')
    if not hasattr(SED, '_configure_esso_cohort_pnet_share_rows'):
        problems.append('SED._configure_esso_cohort_pnet_share_rows missing (H3 rule not in working tree)')
    if not hasattr(SED, '_get_esso_complementarity_diagnostics'):
        problems.append('SED._get_esso_complementarity_diagnostics missing (mu_final/s_obj/bound capture path)')
    if not hasattr(SED, '_optimize'):
        problems.append('SED._optimize missing')
    # Capture-path check for mu_final/s_obj parsing: a prior committed log from
    # this same stage must parse cleanly with the production parser, proving
    # the log format the new solves will produce is the one the parser reads.
    reference_log = os.path.join(OUT_DIR, 'tol_check_logs', 'remedy_h', 'optim_log_node_7.txt')
    if not os.path.exists(reference_log):
        problems.append(f'reference IPOPT log for capture-path check not found: {reference_log}')
    else:
        mu_final, s_obj, reason = SED._parse_ipopt_barrier_terms(reference_log)
        if mu_final is None or s_obj is None:
            problems.append(f'mu_final/s_obj capture path FAILED on reference log: {reason}')
    for path in (SUMMARY_PATH, FULL_PATH):
        if os.path.exists(path):
            problems.append(f'output path already exists, refusing to overwrite: {path}')
    if os.path.isdir(LOG_ROOT) and os.listdir(LOG_ROOT):
        problems.append(f'log directory already exists and is non-empty: {LOG_ROOT}')
    if problems:
        raise RuntimeError('G5 capture-path preflight FAILED:\n' + '\n'.join(f'  - {p}' for p in problems))


def _build_fresh_node_models():
    """Mirrors `create_shared_energy_storage_model`'s pre-solve steps exactly
    (shared_resources_planning.py:3169-3192), reusing the production methods
    (`update_data_with_candidate_solution`, `build_subproblem`,
    `update_model_with_candidate_solution`) and the same `fix_or_set` glue the
    production function itself uses -- stopping short of that function's own
    `shared_ess_data.optimize(...)` call, since G5 needs three independently
    solved copies per node rather than one production-default solve.

    Returns (shared_ess_data, {node_id: unsolved_model}) for GATE_NODES only.
    """
    with redirect_stdout(io.StringIO()):
        planning = oracle.fresh_planning('p515_g5_gate')
    shared_ess_data = planning.shared_ess_data

    consensus_vars, dual_vars = create_admm_variables(planning)
    candidate_solution = _zero_candidate(planning)
    first_year = list(planning.years)[0]
    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        candidate_solution[node_id][first_year] = {'s': S_CANDIDATE_MVA, 'e': E_CANDIDATE_MVAH}
        _set_nonzero_charge_discharge_request(
            planning, consensus_vars, node_id, amplitude_mw=0.10 * S_CANDIDATE_MVA
        )

    years = list(shared_ess_data.years)
    days = list(shared_ess_data.days)

    shared_ess_data.update_data_with_candidate_solution(candidate_solution)
    esso_model = shared_ess_data.build_subproblem()
    shared_ess_data.update_model_with_candidate_solution(esso_model, candidate_solution)

    for node_id in shared_ess_data.active_distribution_network_nodes:
        for y in esso_model[node_id].years:
            year = years[y]
            for d in esso_model[node_id].days:
                day = days[d]
                for p in esso_model[node_id].periods:
                    p_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['p'][p]
                    q_req = consensus_vars['ess']['tso']['current'][node_id][year][day]['q'][p]
                    fix_or_set(esso_model[node_id].es_pnet[y, d, p], p_req)
                    fix_or_set(esso_model[node_id].es_qnet[y, d, p], q_req)

    fresh_models = {node_id: esso_model[node_id] for node_id in GATE_NODES}
    return shared_ess_data, candidate_solution, fresh_models


def _recovery_log_path(primary_log_path):
    stem, ext = os.path.splitext(primary_log_path)
    return f'{stem}_recovery{ext}'


def _run_arm(shared_ess_data, node_id, arm_name, fresh_model):
    spec = ARM_SPECS[arm_name]
    model = copy.deepcopy(fresh_model)

    option_overrides = dict(SED.ESSO_TOL_OVERRIDES)
    option_overrides.update(spec['extra_overrides'])
    primary_log_path = os.path.join(LOG_ROOT, f'node{node_id}_{arm_name}.txt')
    option_overrides['output_file'] = primary_log_path

    label = f'{node_id}_{arm_name}'
    t0 = time.time()
    result = SED._optimize(
        model,
        shared_ess_data.params.solver_params,
        from_warm_start=spec['from_warm_start'],
        node_id=label,
        diagnostic_sink=shared_ess_data.solver_recovery_diagnostics,
        option_overrides=option_overrides,
        complementarity_diagnostics_sink=shared_ess_data.esso_complementarity_diagnostics,
    )
    wall_seconds = time.time() - t0

    recovery_log_path = _recovery_log_path(primary_log_path)
    recovery_fired = os.path.exists(recovery_log_path)
    diagnostics_log_path = recovery_log_path if recovery_fired else primary_log_path

    termination_condition = str(result.solver.termination_condition) if result is not None else None
    status = str(result.solver.status) if result is not None else None

    objective = None
    max_viol = None
    worst_constraint = None
    n_constraints_checked = None
    load_error = None
    # NOTE: `SED._optimize` (called above) already loads the solution into
    # `model` itself on success (`model.solutions.load_from(result)`,
    # `shared_energy_storage_data.py` ~line 1152) -- calling `load_from` a
    # SECOND time on the same `result` object here raised
    # `KeyError` (Pyomo's symbol map is single-use per `SolverResults`
    # object). The model already carries the loaded solution; only read it.
    if solver_result_succeeded(result):
        try:
            objective = pe.value(model.objective)
            max_viol, worst_constraint, n_constraints_checked = max_constraint_violation(model)
        except Exception as exc:  # noqa: BLE001 -- report, do not hide
            load_error = f'{type(exc).__name__}: {exc}'
    else:
        load_error = f'solver did not succeed: {termination_condition}'

    diagnostics = SED._get_esso_complementarity_diagnostics(model, label, diagnostics_log_path)
    primary_log_tail = parse_ipopt_log_tail(primary_log_path)
    recovery_log_tail = parse_ipopt_log_tail(recovery_log_path) if recovery_fired else None

    return {
        'node_id': node_id,
        'arm': arm_name,
        'description': spec['description'],
        'from_warm_start': spec['from_warm_start'],
        'option_overrides': option_overrides,
        'wall_seconds': wall_seconds,
        'termination_condition': termination_condition,
        'status': status,
        'recovery_fired': recovery_fired,
        'objective': objective,
        'load_solutions_error': load_error,
        'max_constraint_violation': max_viol,
        'worst_constraint': worst_constraint,
        'n_constraints_checked': n_constraints_checked,
        'diagnostics': diagnostics,
        'primary_log_tail': primary_log_tail,
        'recovery_log_tail': recovery_log_tail,
        'primary_log_path': primary_log_path,
        'recovery_log_path': recovery_log_path if recovery_fired else None,
    }


def _rule_ten_ratio(arm_result):
    """Terminal-step-to-threshold ratio: unscaled constraint violation at the
    solve that actually produced the loaded solution (recovery if it fired,
    else primary), against the tol in force for THAT solve. Recovery merges
    SED.ESSO_TOL_OVERRIDES on top of params.recovery_options (see
    `_optimize`), so tol=1e-8 applies to both primary and recovery here."""
    tail = arm_result['recovery_log_tail'] if arm_result['recovery_fired'] else arm_result['primary_log_tail']
    if not tail or tail.get('error'):
        return None
    constraint_violation = (tail.get('nlp_error_summary_unscaled') or {}).get('Constraint violation')
    tol_in_force = arm_result['option_overrides'].get('tol', SED.ESSO_TOL_OVERRIDES.get('tol'))
    if constraint_violation is None or not tol_in_force:
        return None
    return constraint_violation / tol_in_force


def _pairwise_discriminator(node_id, arms_by_name):
    eps = SED.EPS_ESSO_THROUGHPUT
    pairs = []
    names = list(arms_by_name.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            obj_a = arms_by_name[a]['objective']
            obj_b = arms_by_name[b]['objective']
            spur_a = arms_by_name[a]['diagnostics']['spurious_throughput_measured']
            spur_b = arms_by_name[b]['diagnostics']['spurious_throughput_measured']
            observed_diff = None
            relative_diff = None
            if obj_a is not None and obj_b is not None:
                observed_diff = obj_a - obj_b
                denom = max(abs(obj_a), abs(obj_b), 1e-300)
                relative_diff = abs(observed_diff) / denom
            predicted_diff = None
            residual = None
            if spur_a is not None and spur_b is not None:
                predicted_diff = eps * (spur_a - spur_b)
            if observed_diff is not None and predicted_diff is not None:
                residual = observed_diff - predicted_diff
            pairs.append({
                'node_id': node_id,
                'pair': f'{a}_vs_{b}',
                'objective_a': obj_a, 'objective_b': obj_b,
                'observed_diff': observed_diff,
                'relative_diff': relative_diff,
                'agree_to_1e-6_relative': (relative_diff is not None and relative_diff <= RELATIVE_AGREEMENT_TOLERANCE),
                'spurious_throughput_a': spur_a, 'spurious_throughput_b': spur_b,
                'eps_esso_throughput': eps,
                'predicted_diff_from_leak': predicted_diff,
                'residual_observed_minus_predicted': residual,
                'residual_over_observed': (
                    abs(residual) / abs(observed_diff) if (residual is not None and observed_diff) else None
                ),
            })
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _assert_capture_paths_exist()
    os.makedirs(LOG_ROOT, exist_ok=True)

    baseline = oracle.load_baseline()
    shared_ess_data, candidate_solution, fresh_models = _build_fresh_node_models()

    instance_descriptor = {
        'construction_source': 'p515_1_esso_reform_smoke.py (imported, not reimplemented)',
        'baseline_checksum': baseline['checksum'],
        'head_commit': '8af3e242286172efda0d02a67c55f586af94cd8c',
        'shared_energy_storage_data_py_sha256_working_tree': hashlib.sha256(
            open(os.path.join(REPO, 'shared_energy_storage_data.py'), 'rb').read()
        ).hexdigest(),
        'working_tree_note': (
            'shared_energy_storage_data.py carries UNCOMMITTED changes on top of '
            'head_commit: remedy (h) ESSO_TOL_OVERRIDES (WORKER_REPORT_H.md) and the '
            'H3 cohort-split rule (WORKER_REPORT_H3.md). The sha256 above is of the '
            'exact file this run executed against.'
        ),
        'smoke_nodes_with_investment': list(SMOKE_NODES_WITH_INVESTMENT),
        's_candidate_mva': S_CANDIDATE_MVA,
        'e_candidate_mvah': E_CANDIDATE_MVAH,
        'amplitude_mw_formula': '0.10 * S_CANDIDATE_MVA',
        'eps_esso_throughput_in_force': SED.EPS_ESSO_THROUGHPUT,
        'esso_tol_overrides_in_force': SED.ESSO_TOL_OVERRIDES,
        'gate_nodes': list(GATE_NODES),
        'candidate_solution': {str(k): v for k, v in candidate_solution.items()},
    }
    instance_hash = sha256_of(instance_descriptor)

    report = {
        'stage': 'P5.15 Gate G5',
        'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 1 ("Gate G5 (new)") + Addendum 3',
        'gate_requirement': 'A1, A3, A4 must agree to 1e-6 relative on node 7 AND node 9',
        'objective_convention': (
            'model.objective == model.feasibility_penalty (ESSO subproblem objective only; '
            'this harness never builds/solves the TSO/DSO SMOPFs, so gross_operational_cost '
            'and net_operational_recourse have NO capture path here and are not reported -- '
            'same convention as p515_h_tol_remedy_check.py and p515_1b_eps_sensitivity_check.py.'
        ),
        'instance': instance_descriptor,
        'instance_hash_sha256': instance_hash,
        'arm_specs': {name: {k: v for k, v in spec.items()} for name, spec in ARM_SPECS.items()},
    }

    guard = SolveProfileGuard(PERMITTED, label='P5.15 Gate G5').install()
    all_arms = {}
    try:
        for node_id in GATE_NODES:
            all_arms[node_id] = {}
            for arm_name in ARM_NAMES:
                print(f'[G5] node={node_id} arm={arm_name} solving...', flush=True)
                arm_result = _run_arm(shared_ess_data, node_id, arm_name, fresh_models[node_id])
                arm_result['rule_ten_ratio'] = _rule_ten_ratio(arm_result)
                all_arms[node_id][arm_name] = arm_result
                diag = arm_result['diagnostics']
                print(
                    f'  -> {arm_result["termination_condition"]} '
                    f'iters={arm_result["primary_log_tail"].get("n_iterations")} '
                    f'recovery_fired={arm_result["recovery_fired"]} '
                    f'objective={arm_result["objective"]} '
                    f'max_viol={arm_result["max_constraint_violation"]} '
                    f'mu_final={diag["mu_final"]} s_obj={diag["s_obj"]} '
                    f'ratio_max={diag["complementarity_ratio_max"]:.6e} '
                    f'rule_ten={arm_result["rule_ten_ratio"]}',
                    flush=True,
                )
    finally:
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)
    report['guard'] = {
        'permitted_call_sites': PERMITTED,
        'expected_solves_declared_in_advance': EXPECTED_SOLVES,
        'observed_counts': guard.counts,
        'permitted_sites_hit': guard.permitted_sites,
        'verify_failures': guard_failures,
    }

    maxiter_or_recovery_events = []
    for node_id in GATE_NODES:
        for arm_name in ARM_NAMES:
            arm = all_arms[node_id][arm_name]
            if arm['recovery_fired'] or arm['termination_condition'] not in ('optimal', 'TerminationCondition.optimal'):
                maxiter_or_recovery_events.append({
                    'node_id': node_id, 'arm': arm_name,
                    'termination_condition': arm['termination_condition'],
                    'recovery_fired': arm['recovery_fired'],
                })
    report['maxiterations_or_recovery_events'] = {
        'count': len(maxiter_or_recovery_events),
        'events': maxiter_or_recovery_events,
        'expected': 'zero',
    }

    node_tables = {}
    discriminator = {}
    node_pass = {}
    for node_id in GATE_NODES:
        table = {}
        for arm_name in ARM_NAMES:
            arm = all_arms[node_id][arm_name]
            diag = arm['diagnostics']
            table[arm_name] = {
                'termination_condition': arm['termination_condition'],
                'n_iterations_primary': arm['primary_log_tail'].get('n_iterations'),
                'recovery_fired': arm['recovery_fired'],
                'n_iterations_recovery': (arm['recovery_log_tail'] or {}).get('n_iterations'),
                'objective': arm['objective'],
                'load_solutions_error': arm['load_solutions_error'],
                'max_constraint_violation': arm['max_constraint_violation'],
                'worst_constraint': arm['worst_constraint'],
                'rule_ten_ratio': arm['rule_ten_ratio'],
                'mu_final': diag['mu_final'],
                's_obj': diag['s_obj'],
                'parse_reason': diag['parse_reason'],
                'complementarity_ratio_max': diag['complementarity_ratio_max'],
                'n_active_cohort_periods': diag['n_active_cohort_periods'],
                'spurious_throughput_bound': diag['spurious_throughput_bound'],
                'spurious_throughput_measured': diag['spurious_throughput_measured'],
            }
        node_tables[node_id] = table
        pairs = _pairwise_discriminator(node_id, all_arms[node_id])
        discriminator[node_id] = pairs
        node_pass[node_id] = all(p['agree_to_1e-6_relative'] for p in pairs)

    gate_pass_strict = all(node_pass.values())

    # Leak-explained-disagreement check: for every disagreeing pair, is the
    # residual (observed - predicted-from-leak) small relative to the
    # observed difference? No fixed threshold is prescribed by the brief for
    # "explained" -- reported as a ratio for the Planner to read, per
    # instructions not to adjudicate what counts as small.
    leak_explains = {}
    for node_id in GATE_NODES:
        leak_explains[node_id] = []
        for p in discriminator[node_id]:
            if p['agree_to_1e-6_relative']:
                continue
            leak_explains[node_id].append({
                'pair': p['pair'],
                'observed_diff': p['observed_diff'],
                'predicted_diff_from_leak': p['predicted_diff_from_leak'],
                'residual': p['residual_observed_minus_predicted'],
                'residual_over_observed': p['residual_over_observed'],
            })

    report['results_by_node'] = {str(k): v for k, v in node_tables.items()}
    report['discriminator_by_node'] = {str(k): v for k, v in discriminator.items()}
    report['leak_explained_disagreements'] = {str(k): v for k, v in leak_explains.items()}
    report['node_pass'] = {str(k): v for k, v in node_pass.items()}
    report['gate_pass_strict_1e-6'] = gate_pass_strict

    with open(FULL_PATH, 'w') as fh:
        json.dump({
            'meta': {k: v for k, v in report.items() if k not in ('results_by_node', 'discriminator_by_node')},
            'arms_full': {
                str(node_id): {arm_name: all_arms[node_id][arm_name] for arm_name in ARM_NAMES}
                for node_id in GATE_NODES
            },
        }, fh, indent=2, default=str)

    with open(SUMMARY_PATH, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f'\n[G5] full per-arm data written to {FULL_PATH}')
    print(f'[G5] summary written to {SUMMARY_PATH}')
    print(f'[G5] guard verify failures: {guard_failures}')
    print(f'[G5] STRICT 1e-6 gate pass: {gate_pass_strict}')
    return 0 if gate_pass_strict else 1


if __name__ == '__main__':
    sys.exit(main())
