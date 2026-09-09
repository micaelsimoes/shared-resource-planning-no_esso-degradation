"""
P5.7 -- an instrumented copy of the P5.6-D evaluation, with fingerprints.

The pipeline below is deliberately the SAME sequence of P5.6-A primitives that
`p56b_policy.evaluate` runs, in the same order, with the same midpoint anchor.
It is duplicated rather than imported-and-patched for one reason: P5.7 needs the
pre-polish ADMM solution and the physical ESSO models, and the accepted P5.6-B/D
code is evidence that should not be edited to obtain them.

Because it is the same sequence, its objectives must reproduce the accepted
P5.6-D base chain exactly.  `p57_d1_chain.py` checks that against the published
values and stops if it does not -- a re-implementation that drifts is worthless
as a diagnosis of the thing it is re-implementing.
"""

import io
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_policy as P  # noqa: E402
import p57_fingerprint as FP  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P57')
ARCHIVE_DIR = os.path.join(OUT_DIR, 'archives')


def evaluate_capture(x, template_state=None, eval_id=None, archive_label=None,
                     capture_admm=True, capture_polished=True,
                     solver_option_overrides=None, objective_rescale=None):
    """One midpoint-anchor evaluation, returning the state and the fingerprints.

    `solver_option_overrides` and `objective_rescale` exist only for the P5.7
    controlled hypotheses; both default to off, in which case this function is
    the accepted P5.6-D evaluation verbatim.
    """
    started = time.time()
    eval_id = eval_id or f'p57_{int(time.time() * 1e6)}'
    label = archive_label or eval_id
    out = {'eval_id': eval_id, 'archive_label': label,
           'oracle_version': O.ORACLE_VERSION, 'anchor': P.ANCHOR_MIDPOINT,
           'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    planning = O.fresh_planning(eval_id)

    if solver_option_overrides:
        # applied to this evaluation's own deep copy only; production untouched
        applied = {}
        for tag, holder in O._tagged_holders(planning):
            options = holder.params.solver_params.options
            applied[tag] = {k: options.get(k) for k in solver_option_overrides}
            options.update(solver_option_overrides)
        out['solver_option_overrides'] = solver_option_overrides
        out['solver_options_replaced'] = applied

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

    if capture_admm:
        out['fingerprint_admm'] = FP.fingerprint(
            planning, models, f'{label}__admm', ARCHIVE_DIR)

    # ---------------- polish, exactly as the accepted policy layer does ------
    clone = srp._clone_operational_models(models)
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
        return out, None

    out['capacity_shift_into_networks'] = O.apply_physical_capacities(
        planning, clone, available)
    O.apply_common_values(planning, clone, common)

    if objective_rescale is None:
        blocks, all_solved = O.polish_networks(planning, clone)
    else:
        blocks, all_solved, factors = polish_networks_rescaled(
            planning, clone, objective_rescale)
        out['objective_rescale'] = objective_rescale
        out['objective_rescale_factors'] = factors
    out['polish_all_solved'] = all_solved
    out['polish_solve_count'] = len(blocks)
    out['esso_solve_count'] = len(esso_solved)
    out['failed_blocks'] = [f"{b['agent']}|{b['year']}|{b['day']}"
                            for b in blocks if not b['solved']]
    if not all_solved:
        out['status'] = O.STATUS_POLISH_FAILURE
        out['wall_clock_s'] = time.time() - started
        return out, None

    residuals, _ = O.coordination_residuals(planning, clone, esso_models,
                                            common, available)
    out['coordination_residuals'] = residuals
    out['esso_audit'] = O.audit_esso(planning, esso_models)
    out['network_audit'] = O.audit_networks(planning, clone)
    out['per_block_polished'] = O.per_block_base_objectives(planning, clone)

    polished = planning.get_operational_recourse_components(clone)
    salvage = float(planning.shared_ess_data.get_salvage_value(esso_models))
    out['gross_operational_cost'] = polished['gross_operational_cost']
    out['physical_salvage'] = salvage
    out['net_operational_recourse'] = polished['gross_operational_cost'] - salvage
    out['total_objective'] = out['investment_cost'] + out['net_operational_recourse']
    out['polish_runtime_s'] = time.time() - polish_started

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


def block_rescale_factor(model, factor_kind):
    """The positive constant this block's polish objective is multiplied by.

      'admm_scale'          -- 1 / effective_scale, i.e. put the base objective
                               back on the magnitude the ADMM subproblem gave it
                               (production builds the ADMM objective as
                               base / effective_scale);
      'inv_admm_scale'      -- effective_scale, the opposite direction;
      ('admm_scale_pow', t) -- (1 / effective_scale) ** t, so t = 1 is the ADMM
                               magnitude and t = 0 is production's own polish;
                               this is the penalty/scaling homotopy of P5.7-D;
      float                 -- that constant, for every block.
    """
    if isinstance(factor_kind, (tuple, list)) and factor_kind[0] == 'admm_scale_pow':
        if not hasattr(model, 'admm_objective_scale'):
            return 1.0
        return (1.0 / float(pe.value(model.admm_objective_scale))) ** float(
            factor_kind[1])
    if factor_kind in ('admm_scale', 'inv_admm_scale'):
        if not hasattr(model, 'admm_objective_scale'):
            return 1.0
        scale = float(pe.value(model.admm_objective_scale))
        return 1.0 / scale if factor_kind == 'admm_scale' else scale
    return float(factor_kind)


def polish_networks_rescaled(planning, models, factor_kind):
    """`O.polish_networks`, but each block minimizes `factor * base objective`.

    A positive constant multiple has an identical argmin and an identical
    feasible set, so this changes only what IPOPT sees -- never what the problem
    is.  Any difference in the returned solution is therefore, by construction,
    a solver-path effect and not an economic one.
    """
    blocks, factors, all_solved = [], {}, True
    for tag, holder in O._tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                key = f'{tag}|{year}|{day}'
                O.restore_base_objective(model)
                factor = block_rescale_factor(model, factor_kind)
                factors[key] = factor
                if factor != 1.0:
                    model.objective.deactivate()
                    if model.component('p57_scaled_objective') is not None:
                        model.del_component('p57_scaled_objective')
                    model.add_component('p57_scaled_objective', pe.Objective(
                        sense=pe.minimize, expr=factor * model.objective.expr))
                with redirect_stdout(io.StringIO()):
                    result = network.run_smopf(model, holder.params,
                                               print_header=False)
                ok = bool(srp._solver_result_succeeded(result))
                all_solved &= ok
                blocks.append({'agent': tag, 'year': year, 'day': day,
                               'solved': ok})
                # restore the base objective so every downstream reader --
                # per_block_base_objectives, get_operational_recourse_components,
                # the audits -- sees the production objective, not the scaled one
                if factor != 1.0:
                    model.del_component('p57_scaled_objective')
                    model.objective.activate()
    return blocks, all_solved, factors


T0_CACHE = os.path.join(OUT_DIR, 't0_state.pkl')


def t0_template(verbose=True):
    """The frozen T0 template, cached to disk across P5.7 harness processes.

    `O.build_fixed_template` caches per PROCESS, and rebuilding costs ~500 s.
    P5.7 runs several separate harnesses against the same T0, so it is cached to
    disk here.  The cache is only ever a speed-up: every harness that uses it
    also checks its step-1 objective against the accepted P5.6-D value
    828021090.360850, so a stale or corrupted template cannot pass silently.
    """
    import pickle
    if os.path.exists(T0_CACHE):
        try:
            with open(T0_CACHE, 'rb') as handle:
                state = pickle.load(handle)
            if verbose:
                print(f'[p57] T0 loaded from {T0_CACHE}', flush=True)
            return state
        except Exception as error:
            print(f'[p57] T0 cache unusable ({error}); rebuilding', flush=True)
    state = O.build_fixed_template(verbose=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        with open(T0_CACHE, 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        print(f'[p57] T0 could not be cached ({error}); continuing', flush=True)
    return state
