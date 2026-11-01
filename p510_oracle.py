"""
P5.10-A -- an operational oracle whose result is a function of the candidate and
an EXPLICIT configuration, not of execution history.

WHAT WAS IMPLICIT, AND WHERE IT ENTERS
--------------------------------------
Production's warm-start path clones `initial_state['models']`
(`shared_resources_planning.py:2173-2179`) and never rebuilds the augmented
objectives, and it restores several scalars from `initial_state` whenever the
evaluated candidate equals the template's candidate
(`continuing_same_candidate`, 2068-2095).  Four distinct channels result:

  1. **rho** -- carried inside the cloned models.  The frozen T0 template holds
     `rho_v = 1.5`, `rho_pf = 2.25`, not the parameter file's 1.0 (P5.9-A).
  2. **objective scaling** -- also carried inside the cloned models, so whether
     a run is CURRENT or RESCALED is a property of the template, not of a flag.
  3. **`consecutive_converged_cycles`** -- restored from `initial_state`
     (2093-2094), so a persistence requirement binds only on a chain's first
     run (P5.8-A0).
  4. **`last_recourse`** and the block-level previous values (2072-2091) -- these
     seed the objective-convergence test.

CHANNEL 4 IS NOT SYMMETRIC ACROSS CANDIDATES, AND THAT IS THE SERIOUS ONE.
`objective_convergence` is initialised False and is only computed when
`previous_recourse is not None` (2325-2328, 2375-2380).  `previous_recourse` is
inherited only when `continuing_same_candidate`, i.e. when
`initial_state['candidate_solution'] == candidate_solution`.  T0's candidate IS
the base candidate.  Therefore, evaluating from T0:

    base candidate      -> continuing_same_candidate TRUE  -> objective test live
                           on cycle 1, so the run may converge in one cycle
    any other candidate -> continuing_same_candidate FALSE -> objective test dead
                           on cycle 1, so the run needs at least two

**The base candidate is evaluated under a different stopping rule from every
other candidate.**  The planning problem consumes `Delta(x) = Q(x) - Q(base)`,
so this asymmetry sits directly inside the quantity being ranked.

WHAT THIS MODULE DOES
---------------------
`OracleConfig` names every one of these explicitly.  `prepare_template` applies
the scaling mode and rho to a PRIVATE copy of the template and then neutralises
the history channels, so that the same declared configuration produces the same
stopping rule for every candidate including the base.  Neutralisation is done by
setting the derived keys directly AND by clearing `candidate_solution`, rather
than by relying on either alone.

`evaluate` returns a provenance record carrying the configuration, its hash, rho
at start and at end, every adaptive action taken, the convergence counters and
the template identifier.

NOTHING HERE IS A PRODUCTION CHANGE.  `data/SRP1/SRP1_params.json` is never
written; no production function is edited.  The scaling wrapper is P5.8's
(`p58_rescale`), removed before every polish, and the rho/adaptive overrides act
on a per-evaluation deep copy and on a private copy of the template.

    imported by p510_a_state.py, p510_b_fixedrho.py, p510_c_endpoint.py,
                p510_f_replay.py, p510_g_anchor.py
"""

import copy
import hashlib
import json
import os
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p57_eval as E7  # noqa: E402
import p58_eval as E8  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P510')
ARCHIVE_DIR = os.path.join(OUT_DIR, 'archives')

T0_ID = 'a81f7f5191dd42dbf50d1726149b8909'
SCALING_CURRENT = 'CURRENT'
SCALING_RESCALED = 'RESCALED'

# the history keys `_run_operational_planning` restores from `initial_state`
HISTORY_KEYS = ('last_recourse', 'last_recourse_blocks',
                'last_objective_component_blocks', 'last_slack_component_blocks',
                'last_tso_voltage_slack_state', 'consecutive_converged_cycles')


@dataclass(frozen=True)
class OracleConfig:
    """Everything that determines Q(x) besides x itself."""
    scaling_mode: str = SCALING_RESCALED
    rho_v: float = 1.5
    rho_pf: float = 1000.0
    rho_ess: float = 1.0
    adaptive_penalty: bool = False
    neutralize_history: bool = True
    template_id: str = T0_ID
    initialization_policy: str = 'warm from frozen T0'
    anchor: str = 'MIDPOINT'
    notes: str = ''

    @property
    def rho(self):
        return {'v': self.rho_v, 'pf': self.rho_pf, 'ess': self.rho_ess}

    def as_dict(self):
        return asdict(self)

    @property
    def config_hash(self):
        payload = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def label(self):
        return (f'{self.scaling_mode}_v{self.rho_v:g}_pf{self.rho_pf:g}'
                f'_ess{self.rho_ess:g}_ad{int(self.adaptive_penalty)}'
                f'_nh{int(self.neutralize_history)}')


# ===========================================================================
#  the template, fully specified
# ===========================================================================
def prepare_template(config):
    """A PRIVATE template copy carrying exactly what `config` declares."""
    with open(E7.T0_CACHE, 'rb') as handle:
        state = pickle.load(handle)

    info = {'template_id': config.template_id,
            'scaling_mode': config.scaling_mode,
            'rho_requested': config.rho,
            'neutralized': {}}

    if config.scaling_mode == SCALING_RESCALED:
        applied = R.rescale_state_models(state)
        scales = [v for v in applied.values() if v]
        info['blocks_rescaled'] = len(scales)
        info['effective_scale_range'] = [min(scales), max(scales)] if scales else None
    elif config.scaling_mode != SCALING_CURRENT:
        raise ValueError(f'unknown scaling_mode {config.scaling_mode!r}')

    info['rho_in_template_before'] = RH.observed_rho(state).get('TSO|2025|Spring')
    RH.apply_rho_to_state(state, config.rho)
    info['rho_in_template_after'] = RH.observed_rho(state).get('TSO|2025|Spring')

    if config.neutralize_history:
        # Explicitly, not by relying on `candidate_solution` alone: clear each
        # key `_run_operational_planning` would restore, then clear the
        # candidate so `continuing_same_candidate` is False for EVERY candidate
        # including the base.  Both, so the neutralisation survives either the
        # keys or the comparison changing.
        for key in HISTORY_KEYS:
            info['neutralized'][key] = _describe(state.get(key))
            state[key] = 0 if key == 'consecutive_converged_cycles' else None
        info['neutralized']['candidate_solution'] = (
            'cleared; was the base candidate, which made '
            'continuing_same_candidate True for the base and False for every '
            'other candidate')
        state['candidate_solution'] = None
    state['initialization_failed'] = False
    return state, info


def _describe(value):
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return f'<{type(value).__name__}>'


# ===========================================================================
#  overrides on the per-evaluation deep copy
# ===========================================================================
@contextmanager
def planning_overrides(config):
    original = O.fresh_planning
    captured = {}

    def patched(eval_id, *args, **kwargs):
        planning = original(eval_id, *args, **kwargs)
        captured['rho_replaced'] = RH.apply_rho_to_params(planning, config.rho)
        captured['adaptive_before'] = RH.set_adaptive_penalty(
            planning, config.adaptive_penalty)
        captured['snapshot'] = RH.rho_snapshot(planning)
        return planning

    O.fresh_planning = patched
    try:
        yield captured
    finally:
        O.fresh_planning = original


@contextmanager
def captured_cost_families():
    """Record the per-family decomposition production itself computes."""
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


@contextmanager
def captured_admm_models():
    """Grab the ADMM output models before the polish clones them.

    `p58_eval.evaluate` calls `srp._clone_operational_models(models)` on the
    ADMM result immediately after the coordination loop and before any polish
    step touches it, so wrapping that call is a clean, read-only way to reach
    the UNPOLISHED solution -- which is exactly what Oracle A is (stage C).
    """
    original = srp._clone_operational_models
    seen = []

    def patched(models, *args, **kwargs):
        seen.append(models)
        return original(models, *args, **kwargs)

    srp._clone_operational_models = patched
    try:
        yield seen
    finally:
        srp._clone_operational_models = original


# ===========================================================================
#  one evaluation
# ===========================================================================
def evaluate(x, config, case_id, template_state=None, capture_polished=False,
             capture_admm=False, audit_unpolished=False):
    """One fully specified evaluation. Returns (record, state).

    With `audit_unpolished`, the ADMM output is audited with production's own
    `audit_networks` BEFORE the polish, which is what stage C needs to compare
    Oracle A against Oracle B.
    """
    if template_state is None:
        template_state, template_info = prepare_template(config)
    else:
        template_info = {'template_id': config.template_id,
                         'supplied_by_caller': True}

    with planning_overrides(config) as captured:
        with captured_cost_families() as families:
            with captured_admm_models() as admm_models:
                record, state = E8.evaluate(
                    x, template_state=template_state, eval_id=case_id,
                    archive_label=case_id, capture_admm=capture_admm,
                    capture_polished=capture_polished)

    if audit_unpolished and admm_models:
        planning_for_audit = O.fresh_planning(f'{case_id}__audit')
        try:
            record['unpolished_network_audit'] = O.audit_networks(
                planning_for_audit, admm_models[0])
        except Exception as error:
            record['unpolished_network_audit'] = {
                'error': f'{type(error).__name__}: {error}'}

    record['case_id'] = case_id
    record['config'] = config.as_dict()
    record['config_hash'] = config.config_hash
    record['config_label'] = config.label
    record['template_info'] = template_info
    record['rho_snapshot_in_force'] = captured.get('snapshot')
    record['cost_families'] = families[-1] if families else None

    if state is not None:
        cycles = RH.cycle_diagnostics(state)
        record['cycle_detail'] = cycles
        record['terminating_criterion'] = RH.terminating_criterion(cycles)
        observed = RH.observed_rho(state)
        record['rho_observed_final'] = observed.get('TSO|2025|Spring')
        record['adaptive_actions'] = [
            {'cycle': c.get('cycle'), 'v': c.get('rho_v_action'),
             'pf': c.get('rho_pf_action'), 'ess': c.get('rho_ess_action')}
            for c in cycles]
        record['n_adaptive_updates'] = sum(
            1 for a in record['adaptive_actions']
            if a['v'] in ('increased', 'decreased')
            or a['pf'] in ('increased', 'decreased'))
        record['consecutive_converged_cycles_out'] = state.get(
            'consecutive_converged_cycles')
        record['consecutive_converged_cycles_in'] = (
            0 if config.neutralize_history else 'inherited')
    return record, state


def row(record):
    """The row every P5.10 table is built from."""
    base = E8.summarize(record)
    cycles = record.get('cycle_detail') or []
    last = cycles[-1] if cycles else {}
    base.update({
        'case_id': record.get('case_id'),
        'config_label': record.get('config_label'),
        'config_hash': record.get('config_hash'),
        'scaling_mode': (record.get('config') or {}).get('scaling_mode'),
        'rho_requested': {k: (record.get('config') or {}).get(f'rho_{k}')
                          for k in ('v', 'pf', 'ess')},
        'rho_observed_final': record.get('rho_observed_final'),
        'n_adaptive_updates': record.get('n_adaptive_updates'),
        'n_failed_polish_blocks': len(record.get('failed_blocks') or []),
        'primal_pf_ratio': last.get('primal_pf_ratio'),
        'primal_v_ratio': last.get('primal_v_ratio'),
        'primal_ess_ratio': last.get('primal_ess_ratio'),
        'dual_pf_mean_ratio': last.get('dual_pf_mean_ratio'),
        'terminating_criterion': record.get('terminating_criterion'),
        'cost_families': record.get('cost_families'),
        'consecutive_converged_cycles_out':
            record.get('consecutive_converged_cycles_out'),
    })
    return base
