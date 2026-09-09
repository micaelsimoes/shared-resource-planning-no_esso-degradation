"""
Stage P5.7 -- the controlled hypotheses A, B and D.

  A) OBJECTIVE SCALING.  Solve selected local NLPs with equivalent base-objective
     scaling and see whether the solution branch changes.  Two levels:
       A1 -- the 48 polish NLPs, under a positive constant multiple of their own
             objective (identical argmin, identical feasible set);
       A2 -- the 48 ADMM subproblems, re-solved in place from their converged
             point, as-is and multiplied by their own effective scale.
     A2 is the decisive one: production builds the ADMM objective as
     `base / effective_scale + consensus terms`, and P5.6-A2.1 measured that
     scale at ~1.1e5.

  B) INITIALIZATION SENSITIVITY.  The SAME polish NLP, at the same investment,
     the same fixed consensus and the same capacities, solved from four starts:
     its own ADMM state, the previous continuation state, the best known state,
     and a cold state.

  D) CONTINUATION PATH.  Direct solve vs capacity continuation vs self-
     refinement at fixed capacity vs penalty (objective-scaling) continuation,
     all to the same target investment.

Nothing production is modified.  Every rescale is a positive constant multiple
applied to a per-evaluation deep copy, and every initial point is transferred
into non-fixed variables only.

    /opt/anaconda3/envs/opf_env_py311/bin/python p57_hypotheses.py
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p56b_policy as P  # noqa: E402
import p56d_oracle as D  # noqa: E402
import p57_eval as E  # noqa: E402
import p57_fingerprint as FP  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p57_hypotheses.json')
CHAIN_PATH = os.path.join(E.OUT_DIR, 'p57_d1_chain.json')
D_TARGET = 'se|node9|2025|-10%'


# ===========================================================================
#  shared machinery
# ===========================================================================
def prepare(x, template_state, eval_id):
    """Run the ADMM and the physical ESSO once; return everything a polish needs."""
    planning = O.fresh_planning(eval_id)
    candidate = O.vector_to_candidate(planning, x)
    feasible, reason = O.check_master_feasibility(planning, candidate)
    if not feasible:
        raise RuntimeError(f'{eval_id}: first-stage infeasible: {reason}')
    models, state, admm = P.run_operational(planning, candidate, template_state)
    if admm.get('status'):
        raise RuntimeError(f"{eval_id}: operational stage returned {admm['status']}")
    common = O.common_coordinated_values(planning, models, state['consensus_vars'],
                                         interface_anchor=P.ANCHOR_MIDPOINT)
    request = O.esso_request_from_common(planning, state['consensus_vars'], common)
    esso_models, _, esso_solved, available = O.solve_physical_esso(
        planning, candidate, request)
    if not all(esso_solved.values()):
        raise RuntimeError(f'{eval_id}: physical ESSO did not solve')
    return {'planning': planning, 'candidate': candidate, 'models': models,
            'state': state, 'admm': admm, 'common': common,
            'esso_models': esso_models, 'available': available}


def polish_variant(ctx, label, rescale=None, init_values=None,
                   archive=True):
    """One polish of a clone of the same ADMM result, under one variant."""
    started = time.time()
    planning, models = ctx['planning'], ctx['models']
    clone = srp._clone_operational_models(models)
    O.apply_physical_capacities(planning, clone, ctx['available'])
    O.apply_common_values(planning, clone, ctx['common'])

    out = {'variant': label, 'rescale': str(rescale) if rescale is not None
           else None}
    if init_values is not None:
        transfer = FP.apply_values(planning, clone, init_values)
        out['initial_point_transfer'] = transfer
        if transfer['n_set'] == 0:
            # a start that silently transferred nothing would masquerade as the
            # default ADMM start and quietly invalidate hypothesis B
            raise RuntimeError(f'{label}: initial point transferred 0 values '
                               f'({transfer})')
    if rescale is None:
        blocks, all_solved = O.polish_networks(planning, clone)
    else:
        blocks, all_solved, factors = E.polish_networks_rescaled(
            planning, clone, rescale)
        out['rescale_factor_min'] = min(factors.values())
        out['rescale_factor_max'] = max(factors.values())
    out['polish_all_solved'] = all_solved
    out['failed_blocks'] = [f"{b['agent']}|{b['year']}|{b['day']}"
                            for b in blocks if not b['solved']]
    if not all_solved:
        out['status'] = O.STATUS_POLISH_FAILURE
        out['runtime_s'] = time.time() - started
        return out, None

    residuals, _ = O.coordination_residuals(planning, clone, ctx['esso_models'],
                                            ctx['common'], ctx['available'])
    audit = O.audit_networks(planning, clone)
    esso_audit = O.audit_esso(planning, ctx['esso_models'])
    polished = planning.get_operational_recourse_components(clone)
    salvage = float(planning.shared_ess_data.get_salvage_value(ctx['esso_models']))
    out.update({
        'max_coordinated_residual': residuals['max_coordinated'],
        'network_max_violation': audit['max_violation'],
        'network_worst_block': audit['worst_block'],
        'network_h1_violation': audit['max_h1_complementarity_violation'],
        'converter_capability_violation':
            audit['max_converter_capability_violation'],
        'esso_production_feasible': esso_audit['production_feasible'],
        'gross_operational_cost': polished['gross_operational_cost'],
        'physical_salvage': salvage,
        'net_operational_recourse': polished['gross_operational_cost'] - salvage,
        'investment_cost': O.investment_cost(planning, ctx['candidate']),
        'runtime_s': time.time() - started,
    })
    out['total_objective'] = out['investment_cost'] + out['net_operational_recourse']
    out['status'] = (O.STATUS_VALID
                     if esso_audit['production_feasible']
                     and residuals['max_coordinated'] <= O.COORDINATION_TARGET
                     else O.STATUS_COUPLING_FAILURE)
    if archive:
        out['fingerprint'] = FP.fingerprint(planning, clone, label,
                                            E.ARCHIVE_DIR)
    values = FP.capture_values(planning, clone)
    return out, values


def block_base_objective(model, network, params, weight):
    return weight * float(pe.value(mch.objective_function_rule(model, params)))


# ===========================================================================
#  A2 -- the ADMM subproblems, re-solved as-is and rescaled
# ===========================================================================
def hypothesis_a2(ctx):
    """Re-solve every ADMM subproblem from its own converged point, twice.

    Variant 1 is production's augmented objective unchanged; variant 2 is the
    SAME objective multiplied by that block's effective scale, which restores
    the base objective to a unit coefficient.  Same argmin, same feasible set.
    """
    planning = ctx['planning']
    rows = []
    for variant, rescale in (('as-is', False), ('x effective_scale', True)):
        clone = srp._clone_operational_models(ctx['models'])
        for tag, holder in O._tagged_holders(planning):
            node_of = None if tag == 'TSO' else int(tag[3:])
            for year in holder.years:
                for day in holder.days:
                    network = holder.network[year][day]
                    model = (clone['tso'][year][day] if tag == 'TSO'
                             else clone['dso'][node_of][year][day])
                    weight = srp._get_admm_block_weight(holder, year, day)
                    if not hasattr(model, 'admm_objective'):
                        continue
                    scale = float(pe.value(model.admm_objective_scale))
                    before = block_base_objective(model, network, holder.params,
                                                  weight)
                    if rescale:
                        model.admm_objective.deactivate()
                        model.add_component('p57_admm_scaled', pe.Objective(
                            sense=pe.minimize,
                            expr=scale * model.admm_objective.expr))
                    with redirect_stdout(io.StringIO()):
                        result = network.run_smopf(model, holder.params,
                                                   print_header=False)
                    ok = bool(srp._solver_result_succeeded(result))
                    after = block_base_objective(model, network, holder.params,
                                                 weight)
                    rows.append({'variant': variant,
                                 'block': f'{tag}|{year}|{day}',
                                 'effective_scale': scale, 'solved': ok,
                                 'weighted_base_before': before,
                                 'weighted_base_after': after,
                                 'delta': after - before})
        del clone
    summary = {}
    for variant in ('as-is', 'x effective_scale'):
        subset = [r for r in rows if r['variant'] == variant]
        summary[variant] = {
            'n_blocks': len(subset),
            'n_solved': sum(1 for r in subset if r['solved']),
            'total_weighted_base_before': sum(r['weighted_base_before']
                                              for r in subset),
            'total_weighted_base_after': sum(r['weighted_base_after']
                                             for r in subset),
            'total_delta': sum(r['delta'] for r in subset),
            'worst_block': min(subset, key=lambda r: r['delta'])['block']
            if subset else None,
            'worst_delta': min((r['delta'] for r in subset), default=None),
        }
    return {'per_block': rows, 'summary': summary}


# ===========================================================================
def main():
    os.makedirs(E.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.7 hypotheses', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.7] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    x0 = population['base']
    if D_TARGET not in population:
        print(f'[P5.7] ABORTED: population has no {D_TARGET}')
        sys.exit(1)
    x_target = population[D_TARGET]

    report = {'stage': 'P5.7 hypotheses', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'fixed_candidate': 'canonical base x0',
              'D_target_candidate': D_TARGET,
              'A': {}, 'B': {}, 'D': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.7] obtaining T0 ...', flush=True)
    t0 = E.t0_template()

    # ================================================================== A ===
    print('\n[P5.7] === hypothesis A: objective scaling ===', flush=True)
    ctx = prepare(x0, t0, 'p57_A_gen1')
    report['A']['generation'] = 1
    report['A']['admm'] = ctx['admm']
    persist()

    print('[P5.7] A2: re-solving the 48 ADMM subproblems, as-is and rescaled',
          flush=True)
    report['A']['A2_admm_subproblems'] = hypothesis_a2(ctx)
    persist()
    for variant, entry in report['A']['A2_admm_subproblems']['summary'].items():
        print(f"      {variant:20s} solved {entry['n_solved']}/{entry['n_blocks']}"
              f"  total base delta {entry['total_delta']:+.4f}", flush=True)

    print('\n[P5.7] A1: the 48 polish NLPs under equivalent scalings', flush=True)
    a1 = {}
    for tag, label, rescale in (
            ('production_scale', 'polish, production scale', None),
            ('inv_effective_scale', 'polish x 1/effective_scale', 'admm_scale'),
            ('x1e3', 'polish x 1e3', 1e3),
            ('x1e-3', 'polish x 1e-3', 1e-3)):
        result, _ = polish_variant(ctx, f'A1__{tag}', rescale=rescale)
        a1[label] = result
        persist()
        print(f"      {label:30s} {result['status']:16s} "
              f"Q={result.get('total_objective')}", flush=True)
    report['A']['A1_polish_scalings'] = a1
    reference = a1['polish, production scale'].get('total_objective')
    report['A']['A1_deltas_vs_production_scale'] = {
        label: (entry.get('total_objective') - reference)
        if entry.get('total_objective') is not None and reference is not None
        else None for label, entry in a1.items()}
    persist()
    del ctx

    # ================================================================== B ===
    print('\n[P5.7] === hypothesis B: initialization sensitivity ===', flush=True)
    print('[P5.7] B: generation 1 (to supply the previous continuation state)',
          flush=True)
    gen1, gen1_state = E.evaluate_capture(x0, template_state=t0,
                                          eval_id='p57_B_gen1',
                                          archive_label='B_gen1',
                                          capture_admm=False)
    if gen1['status'] != O.STATUS_VALID:
        print(f"[P5.7] B ABORTED: generation 1 returned {gen1['status']}")
        report['B']['error'] = gen1['status']
        persist()
        gen1_state = None
    report['B']['generation_1'] = {'status': gen1['status'],
                                   'total_objective': gen1.get('total_objective')}
    persist()

    if gen1_state is not None:
        previous_values = FP.load_archive_values(
            os.path.join(REPO_ROOT, gen1['fingerprint_polished']['archive']))
        ctx2 = prepare(x0, gen1_state, 'p57_B_gen2')
        report['B']['generation'] = 2
        report['B']['admm'] = ctx2['admm']

        starts = [('ADMM state (production default)', None)]
        starts.append(('previous continuation state', previous_values))
        best_archive = os.path.join(E.ARCHIVE_DIR, 'chain_j12__polished.npz')
        if os.path.exists(best_archive):
            starts.append(('best known state (chain step 12)',
                           FP.load_archive_values(best_archive)))
        else:
            report['B']['best_known_state'] = 'chain step 12 archive not found'
        print('[P5.7] B: building cold initial values ...', flush=True)
        starts.append(('cold state', FP.cold_values(ctx2['planning'])))

        variants = {}
        for label, values in starts:
            result, _ = polish_variant(
                ctx2, f'B__{label.split("(")[0].strip().replace(" ", "_")}',
                init_values=values)
            variants[label] = result
            persist()
            print(f"      {label:36s} {result['status']:16s} "
                  f"Q={result.get('total_objective')}", flush=True)
        report['B']['variants'] = variants
        reference = variants['ADMM state (production default)'].get(
            'total_objective')
        report['B']['deltas_vs_admm_start'] = {
            label: (entry.get('total_objective') - reference)
            if entry.get('total_objective') is not None and reference is not None
            else None for label, entry in variants.items()}
        persist()
        del ctx2

    # ================================================================== D ===
    print('\n[P5.7] === hypothesis D: continuation path ===', flush=True)
    report['D']['target'] = D_TARGET

    print('[P5.7] D1: direct solve (K=1)', flush=True)
    direct, direct_state = E.evaluate_capture(
        x_target, template_state=t0, eval_id='p57_D_direct',
        archive_label='D_direct', capture_admm=False)
    report['D']['direct'] = {'status': direct['status'],
                             'total_objective': direct.get('total_objective'),
                             'wall_clock_s': direct.get('wall_clock_s')}
    persist()
    print(f"      {direct['status']}  Q={direct.get('total_objective')}",
          flush=True)

    print('[P5.7] D2: capacity continuation (K=4)', flush=True)
    capacity, _ = D.run_H(x_target, 4, x0, t0, tag='p57_D_capacity')
    report['D']['capacity_continuation'] = {
        'status': capacity['status'],
        'total_objective': capacity.get('total_objective'),
        'lambdas': capacity['lambdas'],
        'step_objectives': [s.get('total_objective') for s in capacity['steps']],
        'total_runtime_s': capacity['total_runtime_s']}
    persist()
    print(f"      {capacity['status']}  Q={capacity.get('total_objective')}",
          flush=True)

    print('[P5.7] D3: self-refinement at fixed capacity (4 solves)', flush=True)
    self_ref, state = {'objectives': []}, direct_state
    if direct['status'] == O.STATUS_VALID:
        self_ref['objectives'].append(direct['total_objective'])
        for k in range(2, 5):
            step, state = E.evaluate_capture(
                x_target, template_state=state, eval_id=f'p57_D_self{k}',
                archive_label=f'D_self{k}', capture_admm=False)
            self_ref['objectives'].append(step.get('total_objective'))
            self_ref[f'status_{k}'] = step['status']
            report['D']['self_refinement'] = self_ref
            persist()
            print(f"      solve {k}: {step['status']}  "
                  f"Q={step.get('total_objective')}", flush=True)
            if step['status'] != O.STATUS_VALID:
                break
    report['D']['self_refinement'] = self_ref
    persist()

    print('[P5.7] D4: penalty (objective-scaling) continuation', flush=True)
    ctx_t = prepare(x_target, t0, 'p57_D_penalty')
    penalty, values = {'steps': []}, None
    for t in (1.0, 2 / 3, 1 / 3, 0.0):
        result, values = polish_variant(
            ctx_t, f'D_penalty_t{t:.3f}', rescale=('admm_scale_pow', t),
            init_values=values)
        penalty['steps'].append({
            'exponent_t': t, 'status': result['status'],
            'rescale_factor_min': result.get('rescale_factor_min'),
            'rescale_factor_max': result.get('rescale_factor_max'),
            'total_objective': result.get('total_objective'),
            'runtime_s': result['runtime_s']})
        report['D']['penalty_continuation'] = penalty
        persist()
        print(f"      t={t:.3f}  {result['status']:16s} "
              f"Q={result.get('total_objective')}", flush=True)
        if result['status'] != O.STATUS_VALID:
            break
    report['D']['penalty_continuation'] = penalty
    persist()

    print(f'\n[P5.7] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
