"""
P5.15-0 -- ESSO / DSO warm-start policy A/B (Step 0 of PLANNER_BRIEF_2026-09-13.md).

Tests hypothesis H-WS: the ESSO subproblem's `maxIterations` failures are
caused by the warm-start policy in `shared_energy_storage_data.py:_create_solver`
(imported bound multipliers, all five `warm_start_*_push/frac` at 1e-9), which
throttles the dual step through the fraction-to-boundary rule. See
`EXPERT_REVIEW_2_ACTION_PLAN.md` section 1 for the diagnosis this harness acts
on.

Diagnostic only. Nothing in production is modified. Every solve goes through
the production solve path:
    - ESSO: shared_energy_storage_data._create_solver / ._run_solver_attempt
    - DSO:  network._create_smopf_solver / ._run_smopf_solver_attempt

RECORDED PRODUCTION QUIRK (not fixed here; reported to the Planner).
`shared_energy_storage_data._create_solver` hardcodes the five
`warm_start_*_push/frac` options to 1e-9 *after* merging `option_overrides`,
whenever `from_warm_start=True` (lines ~898-911 as of this writing). Passing
warm-start push/frac values through `option_overrides` is therefore silently
clobbered back to 1e-9 for the ESSO -- verified directly (see
`esso_clobber_probe` in the report). `network._create_smopf_solver` does NOT
have this specific defect (its warm-start block reads from the same merged
`options` dict that already includes `option_overrides`, via `.get(...)`), but
neither function offers a way to *omit* a warm-start option once
`from_warm_start=True` (both always assign concrete values to all five keys).
To realise an "IPOPT compiled defaults" arm (no explicit warm-start push/frac
values at all) for either solver family, this harness calls the production
`_create_solver` / `_create_smopf_solver` unmodified, then deletes or
reassigns entries directly on the returned `solver.options` object before
calling `solver.solve(...)` -- the same post-creation-override technique
`p512_p_path_sensitivity.py` already used, authorized, for a single key on the
DSO fixture. Every arm's pre- and post-adjustment option dict is recorded.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_0_esso_warmstart_ab.py
"""

import copy
import hashlib
import json
import os
import pickle
import re
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402
import pyomo.opt as po  # noqa: E402

import p56a_oracle as O  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
import network as N  # noqa: E402
from helper_functions import solver_result_summary  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5150')
ESSO_LOG_DIR = os.path.join(OUT, 'esso', 'logs')
DSO_LOG_DIR = os.path.join(OUT, 'dso', 'logs')
os.makedirs(ESSO_LOG_DIR, exist_ok=True)
os.makedirs(DSO_LOG_DIR, exist_ok=True)

ESSO_PICKLE = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514N', 'esso_models_k10000.pkl')
DSO_SNAPSHOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P512R', 'cycle21_pre_setup', 'snapshot.pkl')

REPORT = {
    'stage': 'P5.15-0',
    'objective': 'Test H-WS (ESSO maxIterations caused by 1e-9 warm-start pushes) '
                 'and repeat the same A/B on the preserved cycle-21 DSO fixture.',
}


# ===========================================================================
# generic helpers
# ===========================================================================
def sha256_of(payload_obj):
    blob = json.dumps(payload_obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def parse_ipopt_log_tail(log_path):
    """Extract exit line, iteration count, wall time and the unscaled NLP
    error summary from an IPOPT log file. The FULL iteration table remains in
    the log file itself (not duplicated into JSON)."""
    if not os.path.exists(log_path):
        return {'error': 'log file not found'}
    text = open(log_path, 'r', errors='replace').read()
    out = {}
    exit_lines = [ln.strip() for ln in text.splitlines() if ln.startswith('EXIT:')]
    out['exit_line'] = exit_lines[-1] if exit_lines else None
    m = re.findall(r'Number of Iterations\.+:\s*(\d+)', text)
    out['n_iterations'] = int(m[-1]) if m else None
    m = re.findall(r'Total seconds in IPOPT\s*=\s*([\d.eE+-]+)', text)
    out['total_seconds_in_ipopt'] = float(m[-1]) if m else None
    fields = ['Objective', 'Dual infeasibility', 'Constraint violation', 'Complementarity', 'Overall NLP error']
    unscaled = {}
    for name in fields:
        matches = re.findall(re.escape(name) + r'\.+:\s+([\d.eE+-]+)\s+([\d.eE+-]+)', text)
        unscaled[name] = float(matches[-1][1]) if matches else None
    out['nlp_error_summary_unscaled'] = unscaled
    out['n_iteration_table_rows'] = len(re.findall(r'^\s*\d+r?\s+[\d.eE+-]', text, flags=re.MULTILINE))
    return out


def max_constraint_violation(model):
    """Max violation over every active constraint, evaluated directly from the
    loaded solution (equality: |body - rhs|; inequality: distance beyond the
    violated bound). Returns (max_violation, worst_constraint_name, n_checked)."""
    max_viol = 0.0
    worst = None
    n_checked = 0
    for c in model.component_data_objects(pe.Constraint, active=True):
        body = pe.value(c.body, exception=False)
        if body is None:
            continue
        n_checked += 1
        if c.equality:
            lo = pe.value(c.lower, exception=False)
            viol = abs(body - lo) if lo is not None else 0.0
        else:
            lo = pe.value(c.lower, exception=False) if c.has_lb() else None
            hi = pe.value(c.upper, exception=False) if c.has_ub() else None
            viol = 0.0
            if lo is not None and body < lo:
                viol = lo - body
            if hi is not None and body > hi:
                viol = max(viol, body - hi)
        if viol > max_viol:
            max_viol = viol
            worst = c.name
    return max_viol, worst, n_checked


def active_objective(model):
    for o in model.component_objects(pe.Objective, active=True):
        return o
    return None


# ===========================================================================
# PART 1 -- ESSO A/B
# ===========================================================================
ESSO_FRAC_KEYS = ['warm_start_bound_frac', 'warm_start_slack_bound_frac']
ESSO_PUSH_OVERRIDES = {
    'warm_start_mult_bound_push': 1e-3,
    'warm_start_bound_push': 1e-3,
    'warm_start_slack_bound_push': 1e-3,
}

ESSO_ARM_SPECS = {
    'A0': dict(from_warm_start=True, description='current production policy'),
    'A1': dict(from_warm_start=True, post_delete=list(ESSO_FRAC_KEYS),
               post_overrides=dict(ESSO_PUSH_OVERRIDES),
               description='warm start, pushes=1e-3, frac at IPOPT defaults'),
    'A2': dict(from_warm_start=True, post_delete=list(ESSO_FRAC_KEYS),
               post_overrides=dict(ESSO_PUSH_OVERRIDES), clear_dual_suffixes=True,
               description='warm start primal only (bound multipliers not exported), pushes as A1'),
    'A3': dict(from_warm_start=True, post_delete=list(ESSO_FRAC_KEYS),
               post_overrides=dict(ESSO_PUSH_OVERRIDES, mu_strategy='adaptive'),
               description='A1 + mu_strategy=adaptive'),
    'A4': dict(from_warm_start=False, option_overrides={'warm_start_init_point': 'no'},
               description='cold start'),
    'A5': dict(from_warm_start=True, option_overrides={'max_iter': 500},
               description='A0 + max_iter=500 (cost-of-failure check only)'),
}
ESSO_NODE_ARMS = {
    7: ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
    5: ['A0', 'A4'],
    9: ['A0', 'A4'],
}


def esso_preflight(models):
    checks = {}
    for node_id, model in models.items():
        node_checks = {}
        has_admm_obj = hasattr(model, 'admm_objective')
        node_checks['has_admm_objective'] = has_admm_obj
        node_checks['admm_objective_active'] = bool(model.admm_objective.active) if has_admm_obj else False
        req_attrs = ('p_req', 'q_req', 'dual_p_req', 'dual_q_req', 'rho')
        node_checks['has_all_request_attrs'] = all(hasattr(model, a) for a in req_attrs)
        keys = sorted(model.p_req.keys()) if hasattr(model, 'p_req') else []
        p = [pe.value(model.p_req[k]) for k in keys] if keys else []
        q = [pe.value(model.q_req[k]) for k in keys] if keys else []
        dp = [pe.value(model.dual_p_req[k]) for k in keys] if keys else []
        dq = [pe.value(model.dual_q_req[k]) for k in keys] if keys else []
        rho_val = pe.value(model.rho) if hasattr(model, 'rho') else None
        node_checks['n_periods'] = len(keys)
        node_checks['p_req_nonzero_count'] = sum(1 for v in p if v != 0)
        node_checks['q_req_nonzero_count'] = sum(1 for v in q if v != 0)
        node_checks['dual_p_req_nonzero_count'] = sum(1 for v in dp if v != 0)
        node_checks['dual_q_req_nonzero_count'] = sum(1 for v in dq if v != 0)
        node_checks['rho'] = rho_val
        node_checks['non_trivial'] = (
            node_checks['p_req_nonzero_count'] > 0 and node_checks['q_req_nonzero_count'] > 0
            and node_checks['dual_p_req_nonzero_count'] > 0 and node_checks['dual_q_req_nonzero_count'] > 0
            and (rho_val or 0) > 0
        )
        node_checks['request_parameter_hash_sha256'] = sha256_of(
            {'keys': [list(k) for k in keys], 'p': p, 'q': q, 'dp': dp, 'dq': dq, 'rho': rho_val}
        )
        node_checks['pass'] = (
            node_checks['has_admm_objective'] and node_checks['admm_objective_active']
            and node_checks['has_all_request_attrs'] and node_checks['non_trivial']
        )
        checks[str(node_id)] = node_checks
    checks['all_pass'] = all(checks[str(n)]['pass'] for n in models)
    return checks


def esso_clobber_probe(pristine_model, params):
    """One-off, read-only demonstration that option_overrides on the five
    warm-start keys is clobbered by _create_solver's hardcoded block when
    from_warm_start=True. Solves nothing."""
    model = copy.deepcopy(pristine_model)
    solver, _ = SED._create_solver(
        model, params, from_warm_start=True, node_id='clobber_probe',
        option_overrides={
            'warm_start_mult_bound_push': 1e-3, 'warm_start_bound_push': 1e-3,
            'warm_start_slack_bound_push': 1e-3, 'output_file': os.path.join(ESSO_LOG_DIR, 'clobber_probe.txt'),
        })
    observed = {k: solver.options[k] for k in solver.options if 'warm_start' in k}
    return {
        'option_overrides_requested': {'warm_start_mult_bound_push': 1e-3, 'warm_start_bound_push': 1e-3,
                                        'warm_start_slack_bound_push': 1e-3},
        'solver_options_observed_after_create_solver': observed,
        'clobbered': observed.get('warm_start_mult_bound_push') == 1e-9,
    }


def run_esso_arm(node_id, arm_name, spec, pristine_models, params):
    model = copy.deepcopy(pristine_models[node_id])
    log_path = os.path.join(ESSO_LOG_DIR, f'node{node_id}_{arm_name}.txt')
    option_overrides = dict(spec.get('option_overrides') or {})
    option_overrides['output_file'] = log_path

    t0 = time.time()
    solver, created_log_path = SED._create_solver(
        model, params, from_warm_start=spec['from_warm_start'], node_id=f'{node_id}_{arm_name}',
        option_overrides=option_overrides,
    )
    pre_adjust_options = dict(solver.options)
    for key in spec.get('post_delete') or []:
        if key in solver.options:
            del solver.options[key]
    for key, value in (spec.get('post_overrides') or {}).items():
        solver.options[key] = value
    if spec.get('clear_dual_suffixes'):
        model.ipopt_zL_in.clear()
        model.ipopt_zU_in.clear()
    post_adjust_options = dict(solver.options)

    result = None
    error = None
    try:
        result = solver.solve(model, tee=params.verbose, load_solutions=False)
    except (ValueError, RuntimeError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    wall_seconds = time.time() - t0

    evidence = {
        'node_id': node_id, 'arm': arm_name, 'description': spec.get('description'),
        'from_warm_start': spec['from_warm_start'],
        'solver_log_path': created_log_path,
        'pre_adjust_options': pre_adjust_options,
        'post_adjust_options': post_adjust_options,
        'wall_seconds': wall_seconds,
        'solver_error': error,
    }
    if result is not None:
        evidence['termination_condition'] = str(result.solver.termination_condition)
        evidence['status'] = str(result.solver.status)
        evidence['result_summary'] = solver_result_summary(result)
    else:
        evidence['termination_condition'] = None
        evidence['status'] = None

    log_tail = parse_ipopt_log_tail(log_path)
    evidence['ipopt_log_tail'] = log_tail

    evidence['load_solutions_error'] = None
    evidence['admm_objective'] = None
    evidence['max_constraint_violation'] = None
    evidence['worst_constraint'] = None
    evidence['n_constraints_checked'] = None
    if result is not None:
        try:
            model.solutions.load_from(result)
            evidence['admm_objective'] = pe.value(model.admm_objective)
            max_viol, worst, n_checked = max_constraint_violation(model)
            evidence['max_constraint_violation'] = max_viol
            evidence['worst_constraint'] = worst
            evidence['n_constraints_checked'] = n_checked
        except Exception as exc:  # noqa: BLE001 -- report, do not hide
            evidence['load_solutions_error'] = f'{type(exc).__name__}: {exc}'

    print(f'[P5.15-0][ESSO node={node_id} arm={arm_name}] '
          f'{evidence["termination_condition"]} iters={log_tail.get("n_iterations")} '
          f'wall={wall_seconds:.2f}s admm_objective={evidence["admm_objective"]} '
          f'max_viol={evidence["max_constraint_violation"]}', flush=True)
    return evidence


def run_esso_part():
    section = {'pickle_path': ESSO_PICKLE}
    with open(ESSO_PICKLE, 'rb') as f:
        pristine_models = pickle.load(f)
    section['nodes_in_pickle'] = list(pristine_models.keys())

    preflight = esso_preflight(pristine_models)
    section['preflight'] = preflight
    if not preflight['all_pass']:
        section['status'] = 'STOPPED -- preflight check failed'
        return section

    print('[P5.15-0] ESSO preflight PASSED for all nodes.', flush=True)

    planning = O.fresh_planning('p515_0_esso_params')
    params = planning.shared_ess_data.params.solver_params
    section['solver_params'] = {
        'solver': params.solver, 'solver_path': params.solver_path,
        'options': params.options, 'recovery_options': params.recovery_options,
        'verbose': params.verbose,
    }

    section['clobber_probe'] = esso_clobber_probe(pristine_models[7], params)

    arms = {}
    for node_id, arm_names in ESSO_NODE_ARMS.items():
        for arm_name in arm_names:
            key = f'node{node_id}_{arm_name}'
            arms[key] = run_esso_arm(node_id, arm_name, ESSO_ARM_SPECS[arm_name], pristine_models, params)
    section['arms'] = arms

    a0_node7 = arms.get('node7_A0', {})
    section['reproduction_gate'] = {
        'required': "A0 on node 7 must hit maxIterations",
        'observed_termination_condition': a0_node7.get('termination_condition'),
        'pass': a0_node7.get('termination_condition') == 'maxIterations',
    }
    section['status'] = 'DONE'
    return section


# ===========================================================================
# PART 2 -- DSO A/B on the preserved cycle-21 fixture
# ===========================================================================
DSO_FRAC_KEYS = ['warm_start_bound_frac', 'warm_start_slack_bound_frac']
DSO_ARM_SPECS = {
    'A0': dict(from_warm_start=True, description='current production policy (network.py:521-534)'),
    'A1': dict(from_warm_start=True, post_delete=list(DSO_FRAC_KEYS + ['warm_start_bound_push',
                'warm_start_slack_bound_push', 'warm_start_mult_bound_push']),
               description='IPOPT compiled defaults for all five warm-start push/frac options'),
    'A2': dict(from_warm_start=True, post_delete=list(DSO_FRAC_KEYS + ['warm_start_bound_push',
                'warm_start_slack_bound_push', 'warm_start_mult_bound_push']),
               clear_dual_suffixes=True,
               description='warm start primal only (bound multipliers not exported), pushes as A1'),
    'A3': dict(from_warm_start=True, post_delete=list(DSO_FRAC_KEYS + ['warm_start_bound_push',
                'warm_start_slack_bound_push', 'warm_start_mult_bound_push']),
               post_overrides={'mu_strategy': 'adaptive'},
               description='A1 + mu_strategy=adaptive'),
    'A4': dict(from_warm_start=False, option_overrides={'warm_start_init_point': 'no'},
               description='cold start'),
}


def dso_preflight(payload):
    checks = {
        'boundary': payload.get('boundary'), 'cycle': payload.get('cycle'),
        'target': payload.get('target'), 'from_warm_start': payload.get('from_warm_start'),
    }
    model = payload['model']
    active_objs = [o.name for o in model.component_objects(pe.Objective, active=True)]
    checks['active_objectives'] = active_objs
    checks['n_active_objectives'] = len(active_objs)
    checks['pass'] = (
        checks['boundary'] == 'pre_setup' and checks['cycle'] == 21
        and checks['target'] == 'DSO:case33_3|2025|Spring' and len(active_objs) == 1
    )
    return checks


def run_dso_arm(arm_name, spec):
    with open(DSO_SNAPSHOT, 'rb') as f:
        payload = pickle.load(f)
    model = payload['model']
    net = payload['network']
    params = payload['params']
    net.logs_dir = DSO_LOG_DIR

    option_overrides = dict(spec.get('option_overrides') or {})

    t0 = time.time()
    solver, created_log_path, solve_context = N._create_smopf_solver(
        net, model, params, from_warm_start=spec['from_warm_start'],
        option_overrides=option_overrides, log_suffix=arm_name,
    )
    pre_adjust_options = dict(solver.options)
    for key in spec.get('post_delete') or []:
        if key in solver.options:
            del solver.options[key]
    for key, value in (spec.get('post_overrides') or {}).items():
        solver.options[key] = value
    if spec.get('clear_dual_suffixes'):
        model.ipopt_zL_in.clear()
        model.ipopt_zU_in.clear()
    post_adjust_options = dict(solver.options)

    result = None
    error = None
    try:
        result = solver.solve(model, tee=params.solver_params.verbose, load_solutions=False)
    except (ValueError, RuntimeError) as exc:
        error = f'{type(exc).__name__}: {exc}'
    wall_seconds = time.time() - t0

    evidence = {
        'arm': arm_name, 'description': spec.get('description'), 'solve_context': solve_context,
        'from_warm_start': spec['from_warm_start'], 'solver_log_path': created_log_path,
        'pre_adjust_options': pre_adjust_options, 'post_adjust_options': post_adjust_options,
        'wall_seconds': wall_seconds, 'solver_error': error,
    }
    if result is not None:
        evidence['termination_condition'] = str(result.solver.termination_condition)
        evidence['status'] = str(result.solver.status)
        evidence['result_summary'] = solver_result_summary(result)
    else:
        evidence['termination_condition'] = None
        evidence['status'] = None

    log_tail = parse_ipopt_log_tail(created_log_path) if created_log_path else {'error': 'no log path'}
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
            obj = active_objective(model)
            evidence['objective_name'] = obj.name if obj is not None else None
            evidence['objective_value'] = pe.value(obj) if obj is not None else None
            max_viol, worst, n_checked = max_constraint_violation(model)
            evidence['max_constraint_violation'] = max_viol
            evidence['worst_constraint'] = worst
            evidence['n_constraints_checked'] = n_checked
        except Exception as exc:  # noqa: BLE001
            evidence['load_solutions_error'] = f'{type(exc).__name__}: {exc}'

    print(f'[P5.15-0][DSO arm={arm_name}] {evidence["termination_condition"]} '
          f'iters={log_tail.get("n_iterations")} wall={wall_seconds:.2f}s '
          f'objective={evidence["objective_value"]} max_viol={evidence["max_constraint_violation"]}', flush=True)
    return evidence


def run_dso_part():
    section = {'snapshot_path': DSO_SNAPSHOT}
    with open(DSO_SNAPSHOT, 'rb') as f:
        payload = pickle.load(f)
    section['preflight'] = dso_preflight(payload)
    if not section['preflight']['pass']:
        section['status'] = 'STOPPED -- preflight check failed'
        return section
    print('[P5.15-0] DSO fixture preflight PASSED.', flush=True)

    arms = {}
    for arm_name, spec in DSO_ARM_SPECS.items():
        arms[arm_name] = run_dso_arm(arm_name, spec)
    section['arms'] = arms

    a0 = arms.get('A0', {})
    section['reproduction_gate'] = {
        'required': 'A0 must hit maxIterations (matches P5.12-P Arm A on this fixture)',
        'observed_termination_condition': a0.get('termination_condition'),
        'pass': a0.get('termination_condition') == 'maxIterations',
    }
    section['status'] = 'DONE'
    return section


# ===========================================================================
# main
# ===========================================================================
def main():
    REPORT['esso'] = run_esso_part()
    if REPORT['esso']['status'] != 'DONE':
        REPORT['status'] = 'STOPPED_AT_ESSO'
        _write_report()
        print('P5.15-0 STOPPED at ESSO part:', REPORT['esso']['status'], flush=True)
        return 2
    if not REPORT['esso']['reproduction_gate']['pass']:
        REPORT['status'] = 'STOPPED -- ESSO A0 did not reproduce maxIterations'
        _write_report()
        print('P5.15-0 HARD STOP:', REPORT['status'], flush=True)
        return 2

    REPORT['dso'] = run_dso_part()
    if REPORT['dso']['status'] != 'DONE':
        REPORT['status'] = 'STOPPED_AT_DSO'
        _write_report()
        print('P5.15-0 STOPPED at DSO part:', REPORT['dso']['status'], flush=True)
        return 2

    REPORT['status'] = 'DONE'
    _write_report()
    print('P5.15-0 DONE.', flush=True)
    return 0


def _write_report():
    path = os.path.join(OUT, 'p5150_report.json')
    with open(path, 'w') as f:
        json.dump(REPORT, f, indent=1, default=str)
    print('[P5.15-0] report written to', path, flush=True)


if __name__ == '__main__':
    sys.exit(main())
