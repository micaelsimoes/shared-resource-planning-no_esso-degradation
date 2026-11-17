"""
P5.8 -- the diagnostic ADMM objective rescaling, and IPOPT solver-log readout.

Production builds every network ADMM subproblem objective as

    admm_objective = base_objective / effective_scale  +  ADMM terms          (CURRENT)

with `effective_scale = objective_scale / block_weight`
(`shared_resources_planning.py:3592-3596` for the TSO, `3733-3737` for the DSOs).
P5.7 measured `effective_scale` at 9.41e4 .. 1.16e5.  The formulation P5.8 has to
validate is

    effective_scale * admm_objective
        = base_objective  +  effective_scale * ADMM terms                     (RESCALED)

which is a POSITIVE CONSTANT MULTIPLE of the current objective.  Feasible set,
argmin and KKT point are therefore identical by construction; only what IPOPT
sees changes.  That is the whole hypothesis, and it is why the comparison is
meaningful: any difference in the returned solution is a solver-path effect.

NOTHING HERE IS A PRODUCTION CHANGE.  `patched_admm_objectives()` is a context
manager that wraps `update_transmission_model_to_admm` and
`update_distribution_models_to_admm` for the duration of one diagnostic run in
this process only, and restores them on exit.  The ESSO subproblem is NOT
rescaled: production does not divide it by `objective_scale`
(`update_shared_energy_storage_model_to_admm` takes no scale argument), so there
is nothing to undo there.

    imported by p58_a0_tolerances.py, p58_b_rescale.py, p58_c_replay.py
"""

import os
import re
import sys
from contextlib import contextmanager

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

RESCALED_OBJECTIVE = 'p58_rescaled_admm_objective'


# ===========================================================================
#  rescaling one already-built subproblem
# ===========================================================================
def effective_scale(model):
    if not hasattr(model, 'admm_objective_scale'):
        return None
    return float(pe.value(model.admm_objective_scale))


def rescale_block(model):
    """Replace `admm_objective` by `effective_scale * admm_objective`.

    Returns the multiplier, or None if the block carries no ADMM objective.
    """
    scale = effective_scale(model)
    if scale is None or not hasattr(model, 'admm_objective'):
        return None
    if model.component(RESCALED_OBJECTIVE) is not None:
        model.del_component(RESCALED_OBJECTIVE)
    expr = scale * model.admm_objective.expr
    model.admm_objective.deactivate()
    model.add_component(RESCALED_OBJECTIVE,
                        pe.Objective(sense=pe.minimize, expr=expr))
    return scale


def rescale_all(planning, tso_model, dso_models):
    """Rescale every network subproblem of a built operational model set."""
    applied = {}
    for year in planning.transmission_network.years:
        for day in planning.transmission_network.days:
            applied[f'TSO|{year}|{day}'] = rescale_block(tso_model[year][day])
    for node_id, dso in sorted(planning.distribution_networks.items()):
        for year in dso.years:
            for day in dso.days:
                applied[f'DSO{node_id}|{year}|{day}'] = rescale_block(
                    dso_models[node_id][year][day])
    return applied


# ===========================================================================
#  running a whole ADMM under the rescaled objective
# ===========================================================================
@contextmanager
def patched_admm_objectives(record=None):
    """Run production's own ADMM, with every network subproblem rescaled.

    The two production functions that BUILD the augmented objectives are wrapped
    for the duration of the block: each is called unchanged, and its result is
    then multiplied through by that block's own `effective_scale`.  Both are
    restored on exit, including on an exception.
    """
    original_tso = srp.update_transmission_model_to_admm
    original_dso = srp.update_distribution_models_to_admm
    applied = {} if record is None else record

    def patched_tso(planning_problem, model, params, objective_scale):
        original_tso(planning_problem, model, params, objective_scale)
        network = planning_problem.transmission_network
        for year in network.years:
            for day in network.days:
                applied[f'TSO|{year}|{day}'] = rescale_block(model[year][day])

    def patched_dso(planning_problem, models, params, objective_scale):
        original_dso(planning_problem, models, params, objective_scale)
        for node_id, dso in sorted(planning_problem.distribution_networks.items()):
            for year in dso.years:
                for day in dso.days:
                    applied[f'DSO{node_id}|{year}|{day}'] = rescale_block(
                        models[node_id][year][day])

    srp.update_transmission_model_to_admm = patched_tso
    srp.update_distribution_models_to_admm = patched_dso
    try:
        yield applied
    finally:
        srp.update_transmission_model_to_admm = original_tso
        srp.update_distribution_models_to_admm = original_dso


# ===========================================================================
#  IPOPT solver-log readout
# ===========================================================================
_ITERATIONS = re.compile(r'Number of Iterations\.*:\s*(\d+)')
_SUMMARY_ROW = re.compile(
    r'^(Objective|Dual infeasibility|Constraint violation|'
    r'Variable bound violation|Complementarity|Overall NLP error)'
    r'\.*:\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*$')
_STATUS = re.compile(r'^EXIT: (.+?)\.?$', re.M)


def block_log_path(network, params):
    """Where production writes this block's IPOPT log."""
    options = params.solver_params.options or {}
    name = options.get('output_file')
    if not name:
        return None
    stem, extension = os.path.splitext(os.path.join(network.logs_dir, name))
    day = ''.join(c if c.isalnum() or c in ('-', '_') else '_'
                  for c in str(network.day))
    return f'{stem}_{network.year}_{day}{extension}'


def log_offset(path):
    """Byte offset to read from after the next solve (production appends)."""
    if not path or not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def read_log_since(path, offset):
    """Parse the LAST IPOPT summary in the chunk appended since `offset`.

    IPOPT reports its final error measures twice, `(scaled)` and `(unscaled)`.
    The unscaled column is the one that matters here: it is the stationarity of
    the problem as written, and the scaling hypothesis predicts the CURRENT
    formulation terminates with a much larger unscaled dual infeasibility than
    the RESCALED one for the same physical solve.
    """
    out = {'iterations': None, 'exit': None, 'scaled': {}, 'unscaled': {}}
    if not path or not os.path.exists(path):
        return out
    with open(path, 'r', errors='replace') as handle:
        handle.seek(offset)
        chunk = handle.read()
    if not chunk:
        return out
    iterations = _ITERATIONS.findall(chunk)
    if iterations:
        out['iterations'] = int(iterations[-1])
    exits = _STATUS.findall(chunk)
    if exits:
        out['exit'] = exits[-1].strip()
    for line in chunk.splitlines():
        match = _SUMMARY_ROW.match(line.strip())
        if not match:
            continue
        key = match.group(1).lower().replace(' ', '_')
        out['scaled'][key] = float(match.group(2))
        out['unscaled'][key] = float(match.group(3))
    return out


def restore_for_polish(planning, models):
    """Remove the rescaled objective before the exact-consensus polish.

    `p56a_oracle.restore_base_objective` deactivates `admm_objective` and
    activates `objective`.  It knows nothing about the diagnostic objective added
    here, so without this the block would carry two active objectives and Pyomo
    would refuse to write the NL file.
    """
    removed = 0
    for tag, holder in O._tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                if model.component(RESCALED_OBJECTIVE) is not None:
                    model.del_component(RESCALED_OBJECTIVE)
                    removed += 1
    return removed


# ===========================================================================
#  diagnostic ADMM tolerance overrides  (A0)
# ===========================================================================
def apply_tolerance_overrides(planning, consensus=None,
                              minimum_consecutive_converged_cycles=None):
    """Override ADMM tolerances on ONE per-evaluation deep copy.

    Diagnostic only.  `data/SRP1/SRP1_params.json` is never written, and the
    override applies to the copy handed to this evaluation and to nothing else.
    Returns what was replaced, so the report can state the before/after exactly.
    """
    admm = planning.params.admm
    replaced = {'consensus': {}, 'minimum_consecutive_converged_cycles': None}
    for key, value in (consensus or {}).items():
        replaced['consensus'][key] = admm.tol['consensus'][key]
        admm.tol['consensus'][key] = value
    if minimum_consecutive_converged_cycles is not None:
        replaced['minimum_consecutive_converged_cycles'] = \
            admm.minimum_consecutive_converged_cycles
        admm.minimum_consecutive_converged_cycles = \
            minimum_consecutive_converged_cycles
    return replaced


def tolerance_snapshot(planning):
    admm = planning.params.admm
    return {'consensus': dict(admm.tol['consensus']),
            'stationarity': dict(admm.tol['stationarity']),
            'objective': dict(admm.tol['objective']),
            'num_max_iters': admm.num_max_iters,
            'minimum_consecutive_converged_cycles':
                admm.minimum_consecutive_converged_cycles}


def rescale_state_models(state):
    """Rescale every network subproblem carried inside a saved ADMM state.

    Production's warm-start path clones `initial_state['models']` and never
    rebuilds the augmented objectives, so rescaling the template's models once
    makes every warm run started from it use the RESCALED formulation -- with the
    identical primal starting point, which is what makes C a controlled
    comparison rather than two unrelated runs.
    """
    applied = {}
    models = state['models']
    for year, days in models['tso'].items():
        for day, model in days.items():
            applied[f'TSO|{year}|{day}'] = rescale_block(model)
    for node_id, years in models['dso'].items():
        for year, days in years.items():
            for day, model in days.items():
                applied[f'DSO{node_id}|{year}|{day}'] = rescale_block(model)
    return applied
