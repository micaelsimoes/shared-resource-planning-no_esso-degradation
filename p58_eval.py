"""
P5.8 -- one evaluation, with ADMM tolerance overrides and/or the rescaled
augmented objective, plus the per-cycle ADMM diagnostics P5.8 needs.

Same pipeline as the accepted P5.6-D / P5.7 evaluation: production's nonlinear
ADMM, the original nonlinear ESSO, the midpoint-anchor exact-consensus polish and
the full feasibility audit.  With `tolerance_overrides=None` and
`rescale_admm=False` it IS that evaluation, and `p58_a0_tolerances.py` checks
that by reproducing the accepted base chain before it varies anything.

Both knobs act on a per-evaluation deep copy or on a private copy of the
template.  `data/SRP1/SRP1_params.json` is never written and no production
function is edited; the rescaling is a process-local wrapper that
`p58_rescale.patched_admm_objectives` installs and removes.
"""

import io
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_policy as P  # noqa: E402
import p57_fingerprint as FP  # noqa: E402
import p58_rescale as R  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P58')
ARCHIVE_DIR = os.path.join(OUT_DIR, 'archives')


def _cycle_diagnostics(state):
    """The per-cycle residual record production already keeps."""
    keep = ('cycle', 'local_solves_ok', 'primal_v', 'primal_v_mean',
            'primal_pf', 'primal_pf_mean', 'primal_ess', 'primal_ess_mean',
            'dual_v_mean', 'dual_pf_mean', 'dual_ess_mean',
            'recourse', 'gross_operational_cost',
            'objective_change_abs', 'objective_change_rel',
            'objective_tolerance', 'objective_convergence',
            'residual_convergence', 'cycle_convergence',
            'consecutive_converged_cycles', 'required_consecutive_cycles')
    out = []
    for entry in state.get('admm_diagnostics', []):
        out.append({k: entry.get(k) for k in keep if k in entry})
    return out


def evaluate(x, template_state=None, eval_id=None, archive_label=None,
             tolerance_overrides=None, capture_admm=True, capture_polished=True):
    """One midpoint-anchor evaluation; returns (record, state)."""
    started = time.time()
    eval_id = eval_id or f'p58_{int(time.time() * 1e6)}'
    label = archive_label or eval_id
    out = {'eval_id': eval_id, 'archive_label': label,
           'oracle_version': O.ORACLE_VERSION, 'anchor': P.ANCHOR_MIDPOINT,
           'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    planning = O.fresh_planning(eval_id)
    if tolerance_overrides:
        out['tolerance_overrides'] = tolerance_overrides
        out['tolerances_replaced'] = R.apply_tolerance_overrides(
            planning, **tolerance_overrides)
    out['tolerances_in_force'] = R.tolerance_snapshot(planning)

    candidate = O.vector_to_candidate(planning, x)
    feasible, reason = O.check_master_feasibility(planning, candidate)
    out['master_feasible'], out['master_reason'] = feasible, reason
    if not feasible:
        out['status'] = O.STATUS_INVALID_INVESTMENT
        out['wall_clock_s'] = time.time() - started
        return out, None
    out['investment_cost'] = O.investment_cost(planning, candidate)

    models, state, admm = P.run_operational(planning, candidate, template_state)
    out['admm'] = admm
    if admm.get('status'):
        out['status'] = admm['status']
        out['wall_clock_s'] = time.time() - started
        return out, None
    out['admm_cycles_detail'] = _cycle_diagnostics(state)
    out['admm_base_objective_before_polish'] = admm.get('gross_operational_cost')
    out['admm_net_recourse_before_polish'] = admm.get('net_operational_recourse')

    if capture_admm:
        out['fingerprint_admm'] = FP.fingerprint(planning, models,
                                                 f'{label}__admm', ARCHIVE_DIR)

    # ---------------- polish, exactly as the accepted policy layer does ------
    clone = srp._clone_operational_models(models)
    out['rescaled_objectives_removed_for_polish'] = R.restore_for_polish(
        planning, clone)
    polish_started = time.time()
    common = O.common_coordinated_values(planning, clone, state['consensus_vars'],
                                         interface_anchor=P.ANCHOR_MIDPOINT)
    request = O.esso_request_from_common(planning, state['consensus_vars'], common)
    esso_models, _, esso_solved, available = O.solve_physical_esso(
        planning, candidate, request)
    out['esso_solved'] = esso_solved
    if not all(esso_solved.values()):
        out['status'] = O.STATUS_ESSO_SOLVE_FAILURE
        out['wall_clock_s'] = time.time() - started
        return out, state

    out['capacity_shift_into_networks'] = O.apply_physical_capacities(
        planning, clone, available)
    O.apply_common_values(planning, clone, common)
    blocks, all_solved = O.polish_networks(planning, clone)
    out['polish_all_solved'] = all_solved
    out['failed_blocks'] = [f"{b['agent']}|{b['year']}|{b['day']}"
                            for b in blocks if not b['solved']]
    if not all_solved:
        # the ADMM stage itself succeeded, so the state is still a valid input
        # for the next refinement; the polish is a separate downstream stage
        out['status'] = O.STATUS_POLISH_FAILURE
        out['wall_clock_s'] = time.time() - started
        return out, state

    residuals, _ = O.coordination_residuals(planning, clone, esso_models,
                                            common, available)
    out['coordination_residuals'] = residuals
    out['esso_audit'] = O.audit_esso(planning, esso_models)
    out['network_audit'] = O.audit_networks(planning, clone)

    polished = planning.get_operational_recourse_components(clone)
    salvage = float(planning.shared_ess_data.get_salvage_value(esso_models))
    out['gross_operational_cost'] = polished['gross_operational_cost']
    out['physical_salvage'] = salvage
    out['net_operational_recourse'] = polished['gross_operational_cost'] - salvage
    out['total_objective'] = out['investment_cost'] + out['net_operational_recourse']
    out['polish_runtime_s'] = time.time() - polish_started
    if out['admm_net_recourse_before_polish'] is not None:
        out['admm_to_polish_improvement'] = (
            out['net_operational_recourse'] - out['admm_net_recourse_before_polish'])

    if capture_polished:
        out['fingerprint_polished'] = FP.fingerprint(
            planning, clone, f'{label}__polished', ARCHIVE_DIR,
            esso_models=esso_models)

    if not out['esso_audit']['production_feasible']:
        out['status'] = O.STATUS_ESSO_FEASIBILITY_FAILURE
    elif residuals['max_coordinated'] > O.COORDINATION_TARGET:
        out['status'] = O.STATUS_COUPLING_FAILURE
    else:
        out['status'] = O.STATUS_VALID
    out['wall_clock_s'] = time.time() - started
    return out, state


def summarize(record):
    """The row P5.8 tabulates for every case."""
    audit_n = record.get('network_audit') or {}
    audit_e = record.get('esso_audit') or {}
    residuals = record.get('coordination_residuals') or {}
    admm = record.get('admm') or {}
    cycles = record.get('admm_cycles_detail') or []
    last = cycles[-1] if cycles else {}
    return {
        'status': record.get('status'),
        'polish_all_solved': record.get('polish_all_solved'),
        'failed_blocks': record.get('failed_blocks'),
        'admm_cycles': admm.get('cycles'),
        'admm_converged': admm.get('converged'),
        'admm_runtime_s': admm.get('runtime_s'),
        'primal_v': last.get('primal_v'), 'primal_pf': last.get('primal_pf'),
        'primal_ess': last.get('primal_ess'),
        'primal_v_mean': last.get('primal_v_mean'),
        'primal_pf_mean': last.get('primal_pf_mean'),
        'primal_ess_mean': last.get('primal_ess_mean'),
        'dual_ess_mean': last.get('dual_ess_mean'),
        'objective_change_abs': last.get('objective_change_abs'),
        'objective_convergence': last.get('objective_convergence'),
        'residual_convergence': last.get('residual_convergence'),
        'objective_tolerance': last.get('objective_tolerance'),
        'admm_net_recourse_before_polish':
            record.get('admm_net_recourse_before_polish'),
        'polished_net_operational_recourse':
            record.get('net_operational_recourse'),
        'admm_to_polish_improvement': record.get('admm_to_polish_improvement'),
        'total_objective': record.get('total_objective'),
        'max_coordinated_residual': residuals.get('max_coordinated'),
        'esso_production_feasible': audit_e.get('production_feasible'),
        'network_max_violation': audit_n.get('max_violation'),
        'network_h1_violation': audit_n.get('max_h1_complementarity_violation'),
        'wall_clock_s': record.get('wall_clock_s'),
    }
