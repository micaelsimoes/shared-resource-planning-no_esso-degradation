"""
P5.15-1 Part A -- validate the Step 1a network warm-start policy change by
solving (PLANNER_BRIEF_2026-09-13.md, Step 1 task, "Part A" pre-flight gate).

Step 1a (see P5_15_1A_REPORT.md) changed `network.py::_create_smopf_solver`
(and `_is_recoverable_network_failure` / `_run_smopf`) but only
compile/import/grep-checked the network-family side; it was never solved
against the preserved cycle-21 DSO fixture. This harness closes that gap
before Part B (the ESSO reformulation) proceeds.

Fixture: `data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl`, target
`DSO:case33_3|2025|Spring` (same fixture Step 0/1a's Addendum 1 identifies as
the genuine cycle-21 failure; case33_2/node7/Autumn/cycle7 is the converged
comparator, not used here).

Arms (all through the production `network._create_smopf_solver` +
`solver.solve(...)` path, on an independent fresh load of the pickle per arm):

  1. default  -- from_warm_start=True, no option_overrides (new production
     policy: warm_start_* pushes/fracs left at IPOPT compiled default unless
     explicitly set).
  2. old_policy -- from_warm_start=True, option_overrides sets all five
     warm_start_* keys to 1e-6 (the value the former hardcoded derivation in
     network.py:534 used to produce, per the brief). Must still reproduce
     `maxIterations`, and now at `max_iter=500` (item 3 of Step 1a), not 3000.
  3. cold -- from_warm_start=False, option_overrides={'warm_start_init_point':
     'no'}, plus the warm-start suffixes (ipopt_zL_in, ipopt_zU_in, dual)
     explicitly cleared beforehand so nothing stale is exported to the .nl.

Additional check (no solve): build a solver with option_overrides={'bound_push':
1e-5} only (the generic key, not any warm_start_* key) and confirm
`warm_start_mult_bound_push` (and the other four warm_start_* keys) are
UNSET on the assembled `solver.options` -- i.e. the former derivation
(`warm_start_mult_bound_push <- bound_push`) is gone, not merely overridable.

Gate: arms 1 and 3 (default vs cold) must agree on the objective to 1e-6
relative -- this is the sanity check that the new warm-start default policy
converges to the same point as a cold solve, on the real network fixture (the
ESSO-side equivalent of this was already shown in P5_15_1A_REPORT.md Sec 5.3a,
this closes the DSO-side gap named in that report's Sec 8/Q1).

Nothing under data/ is modified. No production code is modified by this
harness. Not committed.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_1_partA_dso_check.py
"""

import copy
import json
import os
import sys
import time
import pickle

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import network as N  # noqa: E402
from helper_functions import solver_result_summary  # noqa: E402
import p515_0_esso_warmstart_ab as P5150  # noqa: E402  (reuse parse_ipopt_log_tail / max_constraint_violation / active_objective / dso_preflight)

DSO_SNAPSHOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P512R', 'cycle21_pre_setup', 'snapshot.pkl')
# Evidence lives under P5_15_1A's evidence dir (data/SRP1/Results/P5151A/),
# per the task instruction to append Part A's results there, not under a
# separate P5151 dir (Part B / Step 1 proper was not reached -- see report).
OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151A', 'partA')
LOG_DIR = os.path.join(OUT, 'partA_logs')
os.makedirs(LOG_DIR, exist_ok=True)

OLD_POLICY_OVERRIDES = {
    'warm_start_mult_bound_push': 1e-6,
    'warm_start_bound_push': 1e-6,
    'warm_start_bound_frac': 1e-6,
    'warm_start_slack_bound_push': 1e-6,
    'warm_start_slack_bound_frac': 1e-6,
}
# Diagnostic-only, added after the literal 1e-6 arm above did NOT reproduce
# maxIterations (see report): the value the pre-1a network.py derivation
# block actually produced on THIS fixture was 1e-5, not the 1e-6 fallback
# default, because case33_3_params.json sets the generic bound_push /
# bound_frac / slack_bound_push / slack_bound_frac keys to 1e-5 each, and the
# old derivation read those generic keys before falling back to 1e-6. This
# arm reproduces that historical value directly, as evidence for the report;
# it is not a substitute for the brief's literal "1e-6" arm above.
OLD_POLICY_OVERRIDES_1E5_DIAGNOSTIC = {
    'warm_start_mult_bound_push': 1e-5,
    'warm_start_bound_push': 1e-5,
    'warm_start_bound_frac': 1e-5,
    'warm_start_slack_bound_push': 1e-5,
    'warm_start_slack_bound_frac': 1e-5,
}

REPORT = {
    'stage': 'P5.15-1 Part A',
    'objective': 'Validate the Step 1a network.py warm-start policy change by solving on the '
                 'preserved cycle-21 DSO fixture (DSO:case33_3|2025|Spring), through the '
                 'production network.py path.',
    'snapshot_path': DSO_SNAPSHOT,
}


def _load_fixture():
    with open(DSO_SNAPSHOT, 'rb') as f:
        payload = pickle.load(f)
    return payload


def run_arm(arm_name, from_warm_start, option_overrides, clear_suffixes=False):
    payload = _load_fixture()
    model = payload['model']
    net = payload['network']
    params = payload['params']
    net.logs_dir = LOG_DIR

    if clear_suffixes:
        model.ipopt_zL_in.clear()
        model.ipopt_zU_in.clear()
        model.dual.clear()

    overrides = dict(option_overrides or {})

    t0 = time.time()
    solver, created_log_path, solve_context = N._create_smopf_solver(
        net, model, params, from_warm_start=from_warm_start,
        option_overrides=overrides, log_suffix=arm_name,
    )
    observed_warm_start_options = {k: solver.options[k] for k in solver.options if 'warm_start' in k}

    result = None
    error = None
    try:
        result = solver.solve(model, tee=params.solver_params.verbose, load_solutions=False)
    except (ValueError, RuntimeError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    wall_seconds = time.time() - t0

    evidence = {
        'arm': arm_name, 'from_warm_start': from_warm_start,
        'option_overrides_requested': option_overrides,
        'solve_context': solve_context,
        'solver_options_observed_warm_start_keys': observed_warm_start_options,
        'solver_log_path': created_log_path,
        'wall_seconds': wall_seconds, 'solver_error': error,
    }
    if result is not None:
        evidence['termination_condition'] = str(result.solver.termination_condition)
        evidence['status'] = str(result.solver.status)
        evidence['result_summary'] = solver_result_summary(result)
    else:
        evidence['termination_condition'] = None
        evidence['status'] = None

    log_tail = P5150.parse_ipopt_log_tail(created_log_path) if created_log_path else {'error': 'no log path'}
    evidence['ipopt_log_tail'] = log_tail

    evidence['load_solutions_error'] = None
    evidence['objective_name'] = None
    evidence['objective_value'] = None
    evidence['max_constraint_violation'] = None
    evidence['worst_constraint'] = None
    evidence['n_constraints_checked'] = None
    if result is not None:
        try:
            model.solutions.load_from(result)
            obj = P5150.active_objective(model)
            evidence['objective_name'] = obj.name if obj is not None else None
            evidence['objective_value'] = pe.value(obj) if obj is not None else None
            max_viol, worst, n_checked = P5150.max_constraint_violation(model)
            evidence['max_constraint_violation'] = max_viol
            evidence['worst_constraint'] = worst
            evidence['n_constraints_checked'] = n_checked
        except Exception as exc:  # noqa: BLE001 -- report, do not hide
            evidence['load_solutions_error'] = f'{type(exc).__name__}: {exc}'

    print(f'[P5.15-1 Part A][arm={arm_name}] {evidence["termination_condition"]} '
          f'iters={log_tail.get("n_iterations")} wall={wall_seconds:.2f}s '
          f'objective={evidence["objective_value"]} max_viol={evidence["max_constraint_violation"]}',
          flush=True)
    return evidence


def derivation_gone_probe():
    """Confirm: option_overrides={'bound_push': 1e-5} ALONE (the generic key,
    not any warm_start_* key) leaves warm_start_mult_bound_push (and the other
    four warm_start_* keys) UNSET on the assembled solver.options -- i.e. the
    former derivation from the generic key is gone, not merely overridable."""
    payload = _load_fixture()
    model = payload['model']
    net = payload['network']
    params = payload['params']
    net.logs_dir = LOG_DIR
    solver, _, _ = N._create_smopf_solver(
        net, model, params, from_warm_start=True,
        option_overrides={'bound_push': 1e-5, 'output_file': 'derivation_gone_probe.txt'},
        log_suffix='derivation_gone_probe',
    )
    observed = {k: solver.options[k] for k in solver.options}
    warm_start_keys = ['warm_start_mult_bound_push', 'warm_start_bound_push',
                        'warm_start_bound_frac', 'warm_start_slack_bound_push',
                        'warm_start_slack_bound_frac']
    unset = {k: (k not in observed) for k in warm_start_keys}
    return {
        'option_overrides_requested': {'bound_push': 1e-5},
        'bound_push_observed': observed.get('bound_push'),
        'warm_start_keys_unset': unset,
        'all_warm_start_keys_unset': all(unset.values()),
        'full_observed_options': observed,
    }


def main():
    with open(DSO_SNAPSHOT, 'rb') as f:
        payload = pickle.load(f)
    preflight = P5150.dso_preflight(payload)
    REPORT['preflight'] = preflight
    if not preflight['pass']:
        REPORT['status'] = 'STOPPED -- preflight check failed'
        _write_report()
        print('P5.15-1 Part A STOPPED: preflight failed.', flush=True)
        return 2
    print('[P5.15-1 Part A] DSO fixture preflight PASSED.', flush=True)

    REPORT['derivation_gone_probe'] = derivation_gone_probe()

    arms = {}
    arms['default'] = run_arm('default', from_warm_start=True, option_overrides=None)
    arms['old_policy'] = run_arm('old_policy', from_warm_start=True, option_overrides=OLD_POLICY_OVERRIDES)
    arms['cold'] = run_arm('cold', from_warm_start=False,
                            option_overrides={'warm_start_init_point': 'no'},
                            clear_suffixes=True)
    arms['old_policy_1e5_diagnostic'] = run_arm(
        'old_policy_1e5_diagnostic', from_warm_start=True,
        option_overrides=OLD_POLICY_OVERRIDES_1E5_DIAGNOSTIC,
    )
    REPORT['arms'] = arms

    gates = {}
    gates['old_policy_reproduces_maxIterations'] = {
        'required': 'termination_condition == maxIterations (literal brief value: all five warm_start_* at 1e-6)',
        'observed': arms['old_policy']['termination_condition'],
        'n_iterations': arms['old_policy']['ipopt_log_tail'].get('n_iterations'),
        'pass': arms['old_policy']['termination_condition'] == 'maxIterations',
    }
    gates['old_policy_1e5_diagnostic_reproduces_maxIterations'] = {
        'required': 'diagnostic only, not a brief gate -- reproduces the value the pre-1a code actually '
                    'derived on this fixture (case33_3_params.json bound_push/bound_frac/'
                    'slack_bound_push/slack_bound_frac = 1e-5)',
        'observed': arms['old_policy_1e5_diagnostic']['termination_condition'],
        'n_iterations': arms['old_policy_1e5_diagnostic']['ipopt_log_tail'].get('n_iterations'),
        'pass': arms['old_policy_1e5_diagnostic']['termination_condition'] == 'maxIterations',
    }

    default_obj = arms['default']['objective_value']
    cold_obj = arms['cold']['objective_value']
    rel_diff = None
    agree_pass = False
    if isinstance(default_obj, (int, float)) and isinstance(cold_obj, (int, float)):
        denom = max(abs(default_obj), abs(cold_obj), 1e-30)
        rel_diff = abs(default_obj - cold_obj) / denom
        agree_pass = rel_diff <= 1e-6
    gates['default_and_cold_objective_agree'] = {
        'required': 'default and cold objective agree to 1e-6 relative',
        'default_objective': default_obj,
        'cold_objective': cold_obj,
        'relative_difference': rel_diff,
        'pass': agree_pass,
    }

    gates['derivation_gone'] = {
        'required': 'setting only bound_push in an override leaves warm_start_mult_bound_push (and siblings) UNSET',
        'observed': REPORT['derivation_gone_probe']['warm_start_keys_unset'],
        'pass': REPORT['derivation_gone_probe']['all_warm_start_keys_unset'],
    }
    REPORT['gates'] = gates
    required_gate_names = ['old_policy_reproduces_maxIterations', 'default_and_cold_objective_agree', 'derivation_gone']
    all_pass = all(gates[name]['pass'] for name in required_gate_names)
    REPORT['required_gates'] = required_gate_names
    REPORT['all_gates_pass'] = all_pass
    REPORT['status'] = 'DONE'
    _write_report()
    print(f'P5.15-1 Part A DONE. all_gates_pass={all_pass}', flush=True)
    return 0 if all_pass else 1


def _write_report():
    path = os.path.join(OUT, 'p5151_partA_report.json')
    os.makedirs(OUT, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(REPORT, f, indent=1, default=str)
    print('[P5.15-1 Part A] report written to', path, flush=True)


if __name__ == '__main__':
    sys.exit(main())
