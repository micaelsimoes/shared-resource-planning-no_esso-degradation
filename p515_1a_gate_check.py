"""
P5.15-1a -- gate check for the ESSO/network warm-start production fix
(Step 1a of PLANNER_BRIEF_2026-09-13.md, Addendum 1).

Verifies, on the production code as modified by this task (no algorithm
change, only the option-merge/recovery-policy fix in
`shared_energy_storage_data.py::_create_solver` /
`_is_recoverable_shared_ess_failure` / `_optimize` and
`network.py::_create_smopf_solver` / `_is_recoverable_network_failure` /
`_run_smopf`):

  1. The Step 0 `clobber_probe` (reused unmodified from
     `p515_0_esso_warmstart_ab.py`) now shows caller `option_overrides`
     RESPECTED on the ESSO path (previously clobbered back to 1e-9).
  2. Passing the OLD policy (all five `warm_start_*` at 1e-9) via
     `option_overrides` on ESSO node 7 STILL reproduces `maxIterations` --
     i.e. the reproduction path is intact and overrides genuinely reach the
     solver (this is what makes (1) a meaningful fix rather than a
     no-op/dead code path).
  3. Passing A1's policy (1e-3) via `option_overrides` alone converges in
     ~30 iterations on node 7.
  4. The same three checks repeated on node 9 (the second failing instance,
     per Addendum 1's correction of the Step 0 brief).

Uses the same preserved instance and production solve path as Step 0:
`data/SRP1/Results/P514N/esso_models_k10000.pkl`, loaded fresh (deep copy)
per arm, solved through `shared_energy_storage_data._create_solver` +
`solver.solve(...)` directly (same technique Step 0 used to realise an
option combination that the production entry point does not expose as a
single call).

Nothing under data/ is modified. No production code is modified by this
harness (it imports and calls it). Not committed (per instructions).

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_1a_gate_check.py
"""

import copy
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import shared_energy_storage_data as SED  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p515_0_esso_warmstart_ab as P5150  # noqa: E402  (reuse esso_preflight / esso_clobber_probe / parse_ipopt_log_tail)

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151A')
LOG_DIR = os.path.join(OUT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

ESSO_PICKLE = P5150.ESSO_PICKLE

OLD_POLICY_OVERRIDES = {
    'warm_start_mult_bound_push': 1e-9,
    'warm_start_bound_push': 1e-9,
    'warm_start_bound_frac': 1e-9,
    'warm_start_slack_bound_push': 1e-9,
    'warm_start_slack_bound_frac': 1e-9,
}
A1_POLICY_OVERRIDES = {
    'warm_start_mult_bound_push': 1e-3,
    'warm_start_bound_push': 1e-3,
    'warm_start_bound_frac': 1e-3,
    'warm_start_slack_bound_push': 1e-3,
    'warm_start_slack_bound_frac': 1e-3,
}

REPORT = {
    'stage': 'P5.15-1a',
    'objective': 'Gate check for the option_overrides-clobbering fix and the new '
                 'primary warm-start policy in shared_energy_storage_data.py '
                 'and network.py (PLANNER_BRIEF_2026-09-13.md Step 1a).',
}


def run_arm(node_id, arm_name, option_overrides, models, params):
    model = copy.deepcopy(models[node_id])
    log_path = os.path.join(LOG_DIR, f'node{node_id}_{arm_name}.txt')
    overrides = dict(option_overrides)
    overrides['output_file'] = log_path

    solver, created_log_path = SED._create_solver(
        model, params, from_warm_start=True, node_id=f'{node_id}_{arm_name}',
        option_overrides=overrides,
    )
    observed_after_create = {k: solver.options[k] for k in solver.options if 'warm_start' in k}

    result = solver.solve(model, tee=params.verbose, load_solutions=False)
    log_tail = P5150.parse_ipopt_log_tail(log_path)

    admm_objective = None
    max_viol = None
    try:
        model.solutions.load_from(result)
        admm_objective = pe.value(model.admm_objective)
        max_viol, _, _ = P5150.max_constraint_violation(model)
    except Exception as exc:  # noqa: BLE001 -- report, do not hide
        admm_objective = f'load_solutions_error: {type(exc).__name__}: {exc}'

    evidence = {
        'node_id': node_id,
        'arm': arm_name,
        'option_overrides_requested': option_overrides,
        'solver_options_observed_after_create_solver': observed_after_create,
        'termination_condition': str(result.solver.termination_condition),
        'status': str(result.solver.status),
        'n_iterations': log_tail.get('n_iterations'),
        'admm_objective': admm_objective,
        'max_constraint_violation': max_viol,
        'log_path': created_log_path,
    }
    print(f'[P5.15-1a][node={node_id} arm={arm_name}] '
          f'{evidence["termination_condition"]} iters={evidence["n_iterations"]} '
          f'admm_objective={evidence["admm_objective"]} max_viol={evidence["max_constraint_violation"]}',
          flush=True)
    return evidence


def main():
    with open(ESSO_PICKLE, 'rb') as f:
        models = pickle.load(f)
    REPORT['pickle_path'] = ESSO_PICKLE
    REPORT['nodes_in_pickle'] = list(models.keys())

    preflight = P5150.esso_preflight(models)
    REPORT['preflight'] = preflight
    if not preflight['all_pass']:
        REPORT['status'] = 'STOPPED -- preflight check failed'
        _write_report()
        print('P5.15-1a STOPPED: preflight failed.', flush=True)
        return 2

    planning = O.fresh_planning('p515_1a_gate_check_params')
    params = planning.shared_ess_data.params.solver_params
    REPORT['solver_params'] = {
        'solver': params.solver, 'solver_path': params.solver_path,
        'options': params.options, 'recovery_options': params.recovery_options,
    }

    # Gate 1: clobber_probe -- reused unmodified from Step 0's harness.
    clobber = P5150.esso_clobber_probe(models[7], params)
    REPORT['clobber_probe'] = clobber
    gate1_pass = clobber['clobbered'] is False
    REPORT['gate1_overrides_respected'] = {
        'required': 'clobber_probe.clobbered must now be False (overrides reach solver.options)',
        'observed': clobber,
        'pass': gate1_pass,
    }
    print(f'[P5.15-1a] GATE 1 (clobber fixed): {"PASS" if gate1_pass else "FAIL"} '
          f'-- observed={clobber["solver_options_observed_after_create_solver"]}', flush=True)

    arms = {}
    gates = {}
    for node_id in (7, 9):
        arms[f'node{node_id}_old_policy'] = run_arm(node_id, 'old_policy', OLD_POLICY_OVERRIDES, models, params)
        arms[f'node{node_id}_a1_policy'] = run_arm(node_id, 'a1_policy', A1_POLICY_OVERRIDES, models, params)

        old = arms[f'node{node_id}_old_policy']
        a1 = arms[f'node{node_id}_a1_policy']
        gate_old_pass = old['termination_condition'] == 'maxIterations'
        gate_a1_pass = (a1['termination_condition'] == 'optimal') and isinstance(a1['n_iterations'], int) and a1['n_iterations'] <= 60
        gates[f'node{node_id}'] = {
            'old_policy_reproduces_maxIterations': {
                'required': 'termination_condition == maxIterations',
                'observed': old['termination_condition'],
                'n_iterations': old['n_iterations'],
                'pass': gate_old_pass,
            },
            'a1_policy_converges_fast': {
                'required': 'termination_condition == optimal, roughly ~30 iterations (<=60 as a loose gate bound)',
                'observed': a1['termination_condition'],
                'n_iterations': a1['n_iterations'],
                'pass': gate_a1_pass,
            },
        }
        print(f'[P5.15-1a] GATE node={node_id} old_policy->maxIterations: '
              f'{"PASS" if gate_old_pass else "FAIL"} (iters={old["n_iterations"]}); '
              f'a1_policy->fast optimal: {"PASS" if gate_a1_pass else "FAIL"} (iters={a1["n_iterations"]})',
              flush=True)

    REPORT['arms'] = arms
    REPORT['gates'] = gates
    all_pass = gate1_pass and all(
        gates[f'node{n}']['old_policy_reproduces_maxIterations']['pass']
        and gates[f'node{n}']['a1_policy_converges_fast']['pass']
        for n in (7, 9)
    )
    REPORT['all_gates_pass'] = all_pass
    REPORT['status'] = 'DONE'
    _write_report()
    print(f'P5.15-1a DONE. all_gates_pass={all_pass}', flush=True)
    return 0 if all_pass else 1


def _write_report():
    path = os.path.join(OUT, 'p5151a_gate_report.json')
    with open(path, 'w') as f:
        json.dump(REPORT, f, indent=1, default=str)
    print('[P5.15-1a] report written to', path, flush=True)


if __name__ == '__main__':
    sys.exit(main())
