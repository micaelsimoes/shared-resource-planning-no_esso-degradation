"""
P5.9 -- one evaluation under the RESCALED objective with a chosen rho.

This does NOT reimplement the evaluation.  It calls the accepted
`p58_eval.evaluate` -- which is itself the accepted P5.6-D / P5.7 pipeline:
production's nonlinear ADMM, the original nonlinear ESSO, the midpoint-anchor
exact-consensus polish and the full feasibility audit -- and adds exactly two
controlled knobs on top:

  * the RESCALED augmented objective, applied to a PRIVATE copy of the frozen
    T0 template via `p58_rescale.rescale_state_models`, exactly as P5.8-C did;
  * the rho override, written into that same private copy and mirrored into the
    per-evaluation deep copy's ADMM parameters.

Every case therefore starts from the identical T0 primal state and differs only
in the penalties, which is what makes the A sweep a controlled experiment rather
than a set of unrelated runs.

`data/SRP1/SRP1_params.json` is never written and no production function is
edited.  The rho override is installed by patching `p56a_oracle.fresh_planning`
for the duration of one call and removing the patch afterwards.

    imported by p59_a_sweep.py, p59_b_adaptive.py, p59_d_replay.py, p59_e_anchor.py
"""

import os
import pickle
import sys
from contextlib import contextmanager

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p57_eval as E7  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p58_eval as E8  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P59')
ARCHIVE_DIR = os.path.join(OUT_DIR, 'archives')


# ===========================================================================
#  the template, rescaled and penalised, one private copy per case
# ===========================================================================
def load_template(rescale=True, rho=None):
    """A PRIVATE copy of frozen T0, optionally rescaled and re-penalised.

    Loaded from the pickle every time rather than shared, so no case can mutate
    the template another case will use.
    """
    with open(E7.T0_CACHE, 'rb') as handle:
        state = pickle.load(handle)
    info = {'template': 'T0 (a81f7f5191dd42dbf50d1726149b8909)',
            'rescaled': bool(rescale), 'rho_requested': rho}
    if rescale:
        applied = R.rescale_state_models(state)
        scales = [v for v in applied.values() if v]
        info['blocks_rescaled'] = len(scales)
        info['effective_scale_range'] = [min(scales), max(scales)] if scales else None
    if rho:
        RH.apply_rho_to_state(state, rho)
    info['rho_in_template'] = RH.observed_rho(state).get('TSO|2025|Spring')
    return state, info


@contextmanager
def planning_overrides(rho=None, adaptive_penalty=None):
    """Apply rho / adaptive-penalty overrides to each fresh deep copy."""
    original = O.fresh_planning
    captured = {}

    def patched(eval_id, *args, **kwargs):
        planning = original(eval_id, *args, **kwargs)
        if rho:
            captured['rho_replaced'] = RH.apply_rho_to_params(planning, rho)
        if adaptive_penalty is not None:
            captured['adaptive_before'] = RH.set_adaptive_penalty(
                planning, adaptive_penalty)
        captured['snapshot'] = RH.rho_snapshot(planning)
        return planning

    O.fresh_planning = patched
    try:
        yield captured
    finally:
        O.fresh_planning = original


@contextmanager
def captured_cost_families():
    """Record the per-family cost decomposition the polish computes.

    `p58_eval.evaluate` keeps only `gross_operational_cost`, but stage E needs
    `flexibility_cost` per generation to see whether the rescaled formulation
    changes behaviour against the frozen T0 interface anchor.  Rather than
    duplicate the polish, the production accessor is wrapped for the duration of
    one call and its own return value is recorded unchanged.
    """
    original = srp.SharedResourcesPlanning.get_operational_recourse_components
    seen = []

    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            seen.append({k: float(v) for k, v in result.items()
                         if isinstance(v, (int, float))})
        except Exception:
            pass
        return result

    srp.SharedResourcesPlanning.get_operational_recourse_components = patched
    try:
        yield seen
    finally:
        srp.SharedResourcesPlanning.get_operational_recourse_components = original


# ===========================================================================
#  one case
# ===========================================================================
def evaluate(x, template_state, case_id, rho=None, adaptive_penalty=None,
             archive_label=None, capture_admm=False, capture_polished=False):
    """One RESCALED evaluation at the given rho; returns (record, state)."""
    with planning_overrides(rho=rho, adaptive_penalty=adaptive_penalty) as captured:
        with captured_cost_families() as families:
            record, state = E8.evaluate(
                x, template_state=template_state, eval_id=case_id,
                archive_label=archive_label or case_id,
                capture_admm=capture_admm, capture_polished=capture_polished)
    record['cost_families'] = families[-1] if families else None
    record['cost_families_all_calls'] = families
    record['case_id'] = case_id
    record['rho_requested'] = rho
    record['adaptive_penalty_requested'] = adaptive_penalty
    record['rho_snapshot_in_force'] = captured.get('snapshot')
    if state is not None:
        record['cycle_detail'] = RH.cycle_diagnostics(state)
        record['terminating_criterion'] = RH.terminating_criterion(
            record['cycle_detail'])
        observed = RH.observed_rho(state)
        record['rho_observed_final'] = observed.get('TSO|2025|Spring')
        record['rho_observed_spread'] = {
            g: sorted({round(v[g], 8) for v in observed.values() if g in v})
            for g in RH.GROUPS}
    return record, state


def row(record):
    """The row every P5.9 table is built from."""
    base = E8.summarize(record)
    cycles = record.get('cycle_detail') or []
    last = cycles[-1] if cycles else {}
    base.update({
        'case_id': record.get('case_id'),
        'rho_requested': record.get('rho_requested'),
        'adaptive_penalty': record.get('adaptive_penalty_requested'),
        'rho_observed_final': record.get('rho_observed_final'),
        'n_failed_polish_blocks': len(record.get('failed_blocks') or []),
        'primal_pf_ratio': last.get('primal_pf_ratio'),
        'primal_v_ratio': last.get('primal_v_ratio'),
        'primal_ess_ratio': last.get('primal_ess_ratio'),
        'dual_pf_mean_ratio': last.get('dual_pf_mean_ratio'),
        'dual_v_mean_ratio': last.get('dual_v_mean_ratio'),
        'terminating_criterion': record.get('terminating_criterion'),
        'cost_families': record.get('cost_families'),
        'rho_actions': [
            {'cycle': c.get('cycle'), 'v': c.get('rho_v_action'),
             'pf': c.get('rho_pf_action'), 'ess': c.get('rho_ess_action')}
            for c in cycles],
    })
    return base
