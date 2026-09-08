"""
P5.6-B -- oracle policy layer.

P5.6-A settled the physics and the purity of a candidate evaluation.  What it did
not settle is POLICY: which interface anchor the polish uses, which start the
search uses, and how large a difference in the returned objective is meaningful.
This module composes the P5.6-A building blocks -- unchanged -- into the policy
variants B1-B3 have to compare, and adds nothing to the physics.

Two things it provides that `p56a_oracle.evaluate_planning_candidate` cannot:

  * an explicit template argument, so a template can be CHAINED (B1) rather than
    always rebuilt from the canonical cold solution;
  * a split between the operational stage and the polish, so the same ADMM result
    can be polished twice under different anchors (B2).  Polishing mutates the
    models, so each polish runs on `_clone_operational_models` copy; the ADMM is
    solved once.

Every evaluation still runs against its own deep copy of the baseline, so the
P5.6-A3 purity property is preserved.
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56B')

# Anchor policies compared in B2.
ANCHOR_MIDPOINT = 'midpoint'
ANCHOR_DSO = 'dso'
POLICY_MIDPOINT_ONLY = 'midpoint-only'
POLICY_DSO_ALWAYS = 'dso-always'
POLICY_MIDPOINT_THEN_DSO = 'midpoint-first-dso-fallback'


# ===========================================================================
#  operational stage  (the expensive part, run once per candidate)
# ===========================================================================
def run_operational(planning, candidate, template_state=None):
    """The accepted nonlinear operational ADMM, nothing else."""
    started = time.time()
    try:
        with redirect_stdout(io.StringIO()):
            convergence, _, models, _, _, state = \
                planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False,
                    initial_state=template_state, return_state=True)
    except Exception as error:
        return None, None, {'status': O.STATUS_SOLVER_CRASH,
                            'error': f'{type(error).__name__}: {error}',
                            'runtime_s': time.time() - started}
    info = {
        'status': None,
        'converged': bool(convergence),
        'initialization_failed': bool(state.get('initialization_failed', False)),
        'cycles': len(state.get('admm_diagnostics', [])),
        'runtime_s': time.time() - started,
        'n_recovery_diagnostics': len(state.get('solver_recovery_diagnostics', [])),
    }
    if info['initialization_failed']:
        info['status'] = O.STATUS_LOCAL_SOLVE_FAILURE
        return models, state, info
    components = planning.get_operational_recourse_components(models)
    info.update({
        'gross_operational_cost': components['gross_operational_cost'],
        'net_operational_recourse': components['net_operational_recourse'],
    })
    return models, state, info


# ===========================================================================
#  polish stage  (cheap, run once per anchor)
# ===========================================================================
def polish_and_audit(planning, models, state, candidate, anchor):
    """P5.6-A's pipeline from the common values through the audits.

    `models` is mutated, so the caller passes a clone when polishing more than
    once from the same operational result.
    """
    out = {'interface_anchor': anchor}
    started = time.time()
    common = O.common_coordinated_values(planning, models, state['consensus_vars'],
                                         interface_anchor=anchor)
    request = O.esso_request_from_common(planning, state['consensus_vars'], common)
    try:
        esso_models, _, esso_solved, available = O.solve_physical_esso(
            planning, candidate, request)
    except Exception as error:
        out.update({'status': O.STATUS_SOLVER_CRASH, 'failed_stage': 'physical ESSO',
                    'error': f'{type(error).__name__}: {error}',
                    'polish_runtime_s': time.time() - started})
        return out
    out['esso_solved'] = esso_solved
    out['available_capacity'] = {
        f'{node}|{year}': dict(available[node][year])
        for node in available for year in available[node]}
    if not all(esso_solved.values()):
        out.update({'status': O.STATUS_ESSO_SOLVE_FAILURE,
                    'polish_runtime_s': time.time() - started})
        return out

    try:
        out['capacity_shift_into_networks'] = O.apply_physical_capacities(
            planning, models, available)
        O.apply_common_values(planning, models, common)
        blocks, all_solved = O.polish_networks(planning, models)
    except Exception as error:
        out.update({'status': O.STATUS_SOLVER_CRASH,
                    'failed_stage': 'exact-consensus polish',
                    'error': f'{type(error).__name__}: {error}',
                    'polish_runtime_s': time.time() - started})
        return out
    out['polish_all_solved'] = all_solved
    out['failed_blocks'] = [f"{b['agent']}|{b['year']}|{b['day']}"
                            for b in blocks if not b['solved']]
    if not all_solved:
        out.update({'status': O.STATUS_POLISH_FAILURE,
                    'polish_runtime_s': time.time() - started})
        return out

    residuals, _ = O.coordination_residuals(planning, models, esso_models,
                                            common, available)
    out['coordination_residuals'] = residuals
    out['esso_audit'] = O.audit_esso(planning, esso_models)
    out['network_audit'] = O.audit_networks(planning, models)

    out['per_block_polished'] = O.per_block_base_objectives(planning, models)
    polished = planning.get_operational_recourse_components(models)
    physical_salvage = float(planning.shared_ess_data.get_salvage_value(esso_models))
    out['gross_operational_cost'] = polished['gross_operational_cost']
    out['physical_salvage'] = physical_salvage
    out['net_operational_recourse'] = (polished['gross_operational_cost']
                                       - physical_salvage)
    out['investment_cost'] = O.investment_cost(planning, candidate)
    out['total_objective'] = (out['investment_cost']
                              + out['net_operational_recourse'])
    out['polish_runtime_s'] = time.time() - started

    if not out['esso_audit']['production_feasible']:
        out['status'] = O.STATUS_ESSO_FEASIBILITY_FAILURE
    elif residuals['max_coordinated'] > O.COORDINATION_TARGET:
        out['status'] = O.STATUS_COUPLING_FAILURE
    else:
        out['status'] = O.STATUS_VALID
    return out


# ===========================================================================
#  the policy-level evaluation
# ===========================================================================
def evaluate(x, template_state=None, anchor_policy=POLICY_MIDPOINT_THEN_DSO,
             eval_id=None, keep_state=False, keep_models=False):
    """One evaluation under one declared anchor policy.

    `midpoint-first-dso-fallback` attempts the midpoint polish and, only if it
    fails, retries the POLISH ALONE with the DSO anchor from the SAME ADMM
    result.  That is deterministic and does not pay for two successful polishes
    at every candidate.
    """
    started = time.time()
    eval_id = eval_id or f'b_{int(time.time() * 1e6)}'
    result = {'eval_id': eval_id, 'anchor_policy': anchor_policy,
              'oracle_version': O.ORACLE_VERSION,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    planning = O.fresh_planning(eval_id)
    candidate = O.vector_to_candidate(planning, x)
    feasible, reason = O.check_master_feasibility(planning, candidate)
    result['master_feasible'] = feasible
    result['master_reason'] = reason
    if not feasible:
        result['status'] = O.STATUS_INVALID_INVESTMENT
        result['wall_clock_s'] = time.time() - started
        return result
    result['investment_cost'] = O.investment_cost(planning, candidate)

    models, state, admm = run_operational(planning, candidate, template_state)
    result['admm'] = admm
    if admm.get('status'):
        result['status'] = admm['status']
        result['wall_clock_s'] = time.time() - started
        return result

    order = {POLICY_MIDPOINT_ONLY: [ANCHOR_MIDPOINT],
             POLICY_DSO_ALWAYS: [ANCHOR_DSO],
             POLICY_MIDPOINT_THEN_DSO: [ANCHOR_MIDPOINT, ANCHOR_DSO]}[anchor_policy]

    attempts, chosen = [], None
    for anchor in order:
        clone = srp._clone_operational_models(models)
        attempt = polish_and_audit(planning, clone, state, candidate, anchor)
        attempts.append({k: v for k, v in attempt.items()
                         if k not in ('esso_audit', 'network_audit')})
        if attempt['status'] == O.STATUS_VALID:
            chosen = attempt
            if keep_models:
                result['_models'] = clone
            break
    result['polish_attempts'] = attempts
    result['anchors_attempted'] = [a['interface_anchor'] for a in attempts]
    result['fallback_used'] = len(attempts) > 1 and chosen is not None

    if chosen is None:
        last = attempts[-1]
        result['status'] = last['status']
        result['failed_blocks'] = last.get('failed_blocks')
        result['wall_clock_s'] = time.time() - started
        return result

    result.update({k: v for k, v in chosen.items() if k != 'interface_anchor'})
    result['interface_anchor'] = chosen['interface_anchor']
    result['wall_clock_s'] = time.time() - started
    if keep_state:
        result['_state'] = state
        result['_planning'] = planning
    return result


def certificate_summary(result):
    """The B0.1 certificate fields, in one place."""
    if result.get('status') != O.STATUS_VALID:
        return {'status': result.get('status'),
                'failed_blocks': result.get('failed_blocks')}
    return {
        'status': result['status'],
        'master_feasible': result['master_feasible'],
        'interface_anchor': result['interface_anchor'],
        'fallback_used': result['fallback_used'],
        'max_coordinated_residual':
            result['coordination_residuals']['max_coordinated'],
        'esso_max_violation': result['esso_audit']['max_violation'],
        'esso_production_violation':
            result['esso_audit']['production_feasibility_violation'],
        'esso_production_feasible': result['esso_audit']['production_feasible'],
        'network_max_violation': result['network_audit']['max_violation'],
        'network_h1_violation':
            result['network_audit']['max_h1_complementarity_violation'],
        'converter_capability_violation':
            result['network_audit']['max_converter_capability_violation'],
        'physical_salvage': result['physical_salvage'],
        'gross_operational_cost': result['gross_operational_cost'],
        'net_operational_recourse': result['net_operational_recourse'],
        'investment_cost': result['investment_cost'],
        'total_objective': result['total_objective'],
        'admm_cycles': result['admm']['cycles'],
        'admm_runtime_s': result['admm']['runtime_s'],
        'polish_runtime_s': result['polish_runtime_s'],
        'wall_clock_s': result['wall_clock_s'],
    }
