"""
P5.9 -- ADMM penalty (rho) overrides and the extra per-cycle diagnostics the
stage needs.

WHY RHO HAS TO BE REVALIDATED, STATED PRECISELY.  The rescaled objective is a
positive constant multiple of the current one:

    RESCALED = effective_scale * CURRENT
             = base_objective + effective_scale * (rho/2) * c^2 + ...

so the RATIO between the base objective and the consensus penalty is IDENTICAL
in the two formulations -- that is exactly why P5.8-B could prove the argmin
unchanged.  Rescaling does not reweight the penalty.

What changes is which point IPOPT actually returns.  Under CURRENT the
subproblems stop far short of base-optimality (P5.7-A4: 8.89e6 of base objective
left on the table), and that under-solving acted as unintended regularization --
the agents never moved far enough apart to strain the consensus.  Under RESCALED
each agent pursues its own base objective properly and therefore AGREES LESS:
P5.8-C measured `primal_pf` rising to 1.017e-02, at the 1e-2 tolerance boundary,
and the exact-consensus polish failed on 1-4 DSO blocks in 7 of 8 generations.
`rho` is the only remaining lever that buys agreement back, and its production
value of 1.0 was calibrated against the under-solved operating point.

NOTHING HERE IS A PRODUCTION CHANGE.  `data/SRP1/SRP1_params.json` is never
written.  Overrides are applied to a per-evaluation deep copy of the planning
problem and to a PRIVATE copy of the T0 template state, exactly as
`p58_rescale.rescale_state_models` does, and they live only for that evaluation.

    imported by p59_a_sweep.py, p59_b_adaptive.py, p59_d_replay.py, p59_e_anchor.py
"""

import os
import sys

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GROUPS = ('v', 'pf', 'ess')


# ===========================================================================
#  writing rho into an already-built model set
# ===========================================================================
def _set_block_rho(model, rho):
    """Set the mutable rho Params on one built network subproblem."""
    applied = {}
    for group in GROUPS:
        if group not in rho:
            continue
        component = getattr(model, f'rho_{group}', None)
        if component is None:
            continue
        applied[group] = (float(pe.value(component)), float(rho[group]))
        component.set_value(float(rho[group]))
    # `rho_ess_prev` is scaled by the same factor as `rho_ess` in production's
    # own adaptive update, so it is kept proportional here rather than pinned.
    if 'ess' in rho and hasattr(model, 'rho_ess_prev'):
        before, after = applied.get('ess', (None, None))
        if before not in (None, 0.0):
            previous = float(pe.value(model.rho_ess_prev))
            model.rho_ess_prev.set_value(previous * (after / before))
    return applied


def apply_rho_to_state(state, rho):
    """Write rho into every network subproblem carried inside a saved state.

    Production's warm-start path clones `initial_state['models']` and never
    rebuilds the augmented objectives, so writing rho into a private copy of the
    template makes every warm run started from it use these penalties -- from an
    identical primal starting point, which is what keeps the sweep controlled.
    """
    touched = {}
    models = state['models']
    for year, days in models['tso'].items():
        for day, model in days.items():
            touched[f'TSO|{year}|{day}'] = _set_block_rho(model, rho)
    for node_id, years in models['dso'].items():
        for year, days in years.items():
            for day, model in days.items():
                touched[f'DSO{node_id}|{year}|{day}'] = _set_block_rho(model, rho)
    return touched


def apply_rho_to_params(planning, rho):
    """Mirror the override into the deep copy's ADMM parameters.

    The warm path does not rebuild from these, but keeping them consistent means
    a rebuild inside the same evaluation cannot silently use production values,
    and the recorded `rho_snapshot` is then honest about what was in force.
    """
    admm = planning.params.admm
    replaced = {}
    for group in GROUPS:
        if group not in rho:
            continue
        replaced[group] = dict(admm.rho[group])
        for network_name in admm.rho[group]:
            admm.rho[group][network_name] = float(rho[group])
    return replaced


def set_adaptive_penalty(planning, enabled):
    """Turn production's adaptive rho balancing on/off on the deep copy."""
    admm = planning.params.admm
    before = admm.adaptive_penalty
    admm.adaptive_penalty = bool(enabled)
    return before


def rho_snapshot(planning):
    admm = planning.params.admm
    return {'rho': {g: dict(admm.rho[g]) for g in admm.rho},
            'adaptive_penalty': admm.adaptive_penalty,
            'penalty_update': dict(admm.penalty_update)}


def observed_rho(state):
    """Rho actually in force on the returned models, read back from Pyomo."""
    models = state['models']
    out = {}
    for year, days in models['tso'].items():
        for day, model in days.items():
            out[f'TSO|{year}|{day}'] = {
                g: float(pe.value(getattr(model, f'rho_{g}')))
                for g in GROUPS if hasattr(model, f'rho_{g}')}
    for node_id, years in models['dso'].items():
        for year, days in years.items():
            for day, model in days.items():
                out[f'DSO{node_id}|{year}|{day}'] = {
                    g: float(pe.value(getattr(model, f'rho_{g}')))
                    for g in GROUPS if hasattr(model, f'rho_{g}')}
    return out


# ===========================================================================
#  the per-cycle record P5.9 tabulates
# ===========================================================================
CYCLE_KEYS = (
    'cycle', 'local_solves_ok',
    # consensus (primal) residuals and how close they are to their tolerances
    'primal_v', 'primal_v_mean', 'primal_pf', 'primal_pf_mean',
    'primal_ess', 'primal_ess_mean',
    'primal_v_ratio', 'primal_pf_ratio', 'primal_ess_ratio',
    'primal_v_mean_ratio', 'primal_pf_mean_ratio', 'primal_ess_mean_ratio',
    # stationarity (dual) residuals
    'dual_v', 'dual_v_mean', 'dual_pf', 'dual_pf_mean',
    'dual_ess', 'dual_ess_mean',
    'dual_v_mean_ratio', 'dual_pf_mean_ratio', 'dual_ess_mean_ratio',
    # objective evolution and the criterion that actually terminates the loop
    'gross_operational_cost', 'recourse',
    'objective_change_abs', 'objective_change_rel', 'objective_tolerance',
    'objective_convergence', 'residual_convergence', 'cycle_convergence',
    'consecutive_converged_cycles', 'required_consecutive_cycles',
    # adaptive penalty behaviour (stage B)
    'rho_v_before', 'rho_pf_before', 'rho_ess_before',
    'rho_v_after', 'rho_pf_after', 'rho_ess_after',
    'rho_v_action', 'rho_pf_action', 'rho_ess_action',
    # where the worst disagreement sits
    'worst_pf_primal_node', 'worst_pf_primal_year', 'worst_pf_primal_day',
    'worst_pf_primal_type', 'worst_pf_primal_difference',
    'worst_v_primal_node', 'worst_v_primal_difference',
)


def cycle_diagnostics(state):
    return [{k: entry.get(k) for k in CYCLE_KEYS if k in entry}
            for entry in state.get('admm_diagnostics', [])]


def terminating_criterion(cycles):
    """Which test actually stopped the ADMM on the last recorded cycle.

    `cycle_convergence = residual_convergence AND objective_convergence`, so the
    BINDING criterion is the one that was the last to become true -- or, when
    both are true on the first cycle, the one that would have failed first had
    it been even slightly tighter.  Reported factually per cycle; stage C
    interprets it.
    """
    if not cycles:
        return None
    last = cycles[-1]
    return {
        'residual_convergence': last.get('residual_convergence'),
        'objective_convergence': last.get('objective_convergence'),
        'objective_change_abs': last.get('objective_change_abs'),
        'objective_tolerance': last.get('objective_tolerance'),
        'objective_slack_factor': (
            (last['objective_tolerance'] / last['objective_change_abs'])
            if last.get('objective_tolerance') and last.get('objective_change_abs')
            else None),
        'worst_primal_ratio': max(
            [v for v in (last.get('primal_v_ratio'), last.get('primal_pf_ratio'),
                         last.get('primal_ess_ratio')) if v is not None],
            default=None),
        'worst_dual_mean_ratio': max(
            [v for v in (last.get('dual_v_mean_ratio'),
                         last.get('dual_pf_mean_ratio'),
                         last.get('dual_ess_mean_ratio')) if v is not None],
            default=None),
    }
