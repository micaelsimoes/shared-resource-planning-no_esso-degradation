"""
Stage P5.8-B -- objective rescaling validation.

    CURRENT   base_objective / effective_scale  +  ADMM terms
    RESCALED  base_objective  +  effective_scale * ADMM terms

The second is the first multiplied by `effective_scale > 0`, so the feasible set,
the argmin and the KKT point are identical by construction.  Everything else is
held fixed: the same converged ADMM state as the initial point, the same
consensus variables, the same rho, the same proximal terms, the same IPOPT
settings, the same model objects up to a Pyomo clone.

B1 -- one-block experiments on three representative blocks.
B2 -- the full 48-block base replay.

Recorded per block: base objective, augmented objective (reported on the CURRENT
scale for both variants so they are comparable), IPOPT iterations and exit,
IPOPT's own scaled AND unscaled dual infeasibility / constraint violation /
overall NLP error, and the block's consensus residual.

Nothing production is modified: the rescale is applied to a Pyomo clone.

    /opt/anaconda3/envs/opf_env_py311/bin/python p58_b_rescale.py
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
import p57_eval as E7  # noqa: E402
import p57_hypotheses as H  # noqa: E402
import p58_eval as E  # noqa: E402
import p58_rescale as R  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p58_b_rescale.json')

# B1 blocks: the two P5.6-A2.1 used, plus P5.7's largest single recovery
B1_BLOCKS = ['DSO5|2035|Autumn', 'TSO|2025|Spring', 'DSO7|2035|Autumn']


def block_consensus_residual(model, is_tso):
    """Max |expected - required| over this block's consensus families, in p.u."""
    worst = {'vmag': 0.0, 'pf': 0.0, 'ess': 0.0}
    pairs = (('vmag', 'expected_interface_vmag', 'vmag_req'),
             ('pf', 'expected_interface_pf_p', 'p_pf_req'),
             ('pf', 'expected_interface_pf_q', 'q_pf_req'),
             ('ess', 'expected_shared_ess_p', 'p_ess_req'),
             ('ess', 'expected_shared_ess_q', 'q_ess_req'))
    for group, expected_name, required_name in pairs:
        expected = getattr(model, expected_name, None)
        required = getattr(model, required_name, None)
        if expected is None or required is None:
            continue
        for idx in expected:
            try:
                delta = abs(float(pe.value(expected[idx]))
                            - float(pe.value(required[idx])))
            except Exception:
                continue
            worst[group] = max(worst[group], delta)
    worst['max'] = max(worst.values())
    return worst


def measure(model, network, params, weight, is_tso):
    """Everything read off a block without solving it."""
    base = weight * float(pe.value(mch.objective_function_rule(model, params)))
    augmented = None
    for name in ('admm_objective', R.RESCALED_OBJECTIVE):
        component = model.component(name)
        if component is not None:
            value = float(pe.value(component.expr))
            if name == R.RESCALED_OBJECTIVE:
                value /= R.effective_scale(model)   # back onto the CURRENT scale
            augmented = value
            break
    return {'weighted_base_objective': base,
            'augmented_objective_current_scale': augmented,
            'consensus_residual': block_consensus_residual(model, is_tso)}


def run_block(models, planning, tag, year, day, rescaled):
    """Solve one block from its converged point, CURRENT or RESCALED."""
    holder = (planning.transmission_network if tag == 'TSO'
              else planning.distribution_networks[int(tag[3:])])
    network = holder.network[year][day]
    model = (models['tso'][year][day] if tag == 'TSO'
             else models['dso'][int(tag[3:])][year][day])
    weight = srp._get_admm_block_weight(holder, year, day)
    is_tso = tag == 'TSO'

    scale = R.effective_scale(model)
    before = measure(model, network, holder.params, weight, is_tso)
    if rescaled:
        R.rescale_block(model)

    log = R.block_log_path(network, holder.params)
    offset = R.log_offset(log)
    started = time.time()
    with redirect_stdout(io.StringIO()):
        result = network.run_smopf(model, holder.params, print_header=False)
    runtime = time.time() - started
    solved = bool(srp._solver_result_succeeded(result))
    after = measure(model, network, holder.params, weight, is_tso)
    ipopt = R.read_log_since(log, offset)

    return {'block': f'{tag}|{year}|{day}',
            'variant': 'RESCALED' if rescaled else 'CURRENT',
            'effective_scale': scale, 'solved': solved, 'runtime_s': runtime,
            'weighted_base_before': before['weighted_base_objective'],
            'weighted_base_after': after['weighted_base_objective'],
            'base_delta': (after['weighted_base_objective']
                           - before['weighted_base_objective']),
            'augmented_before_current_scale':
                before['augmented_objective_current_scale'],
            'augmented_after_current_scale':
                after['augmented_objective_current_scale'],
            'consensus_residual_before': before['consensus_residual'],
            'consensus_residual_after': after['consensus_residual'],
            'ipopt_iterations': ipopt['iterations'], 'ipopt_exit': ipopt['exit'],
            'ipopt_scaled': ipopt['scaled'], 'ipopt_unscaled': ipopt['unscaled']}


def main():
    os.makedirs(E.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.8-B rescaling validation', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.8] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    report = {'stage': 'P5.8-B', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'candidate': 'canonical positive-bootstrap base x0',
              'formulations': {
                  'CURRENT': 'base_objective / effective_scale + ADMM terms',
                  'RESCALED': 'base_objective + effective_scale * ADMM terms'},
              'identical': ['initialization (the same converged ADMM state)',
                            'consensus variables', 'rho', 'proximal terms',
                            'IPOPT settings', 'model objects up to a clone'],
              'B1_one_block': [], 'B2_full_replay': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.8] obtaining T0 ...', flush=True)
    t0 = E7.t0_template()
    print('[P5.8] running the ADMM once at x0 ...', flush=True)
    ctx = H.prepare(x0, t0, 'p58_B_base')
    planning, models = ctx['planning'], ctx['models']
    report['admm'] = ctx['admm']
    persist()

    # ------------------------------------------------------------------ B1 --
    print('\n[P5.8] B1: one-block experiments\n', flush=True)
    for key in B1_BLOCKS:
        tag, year, day = key.split('|')
        year = int(year)
        for rescaled in (False, True):
            clone = srp._clone_operational_models(models)
            row = run_block(clone, planning, tag, year, day, rescaled)
            report['B1_one_block'].append(row)
            persist()
            print(f"      {row['block']:22s} {row['variant']:9s} "
                  f"iters={str(row['ipopt_iterations']):>5s} "
                  f"base delta {row['base_delta']:14.2f}  "
                  f"unscaled dual inf "
                  f"{row['ipopt_unscaled'].get('dual_infeasibility')}",
                  flush=True)
            del clone

    # ------------------------------------------------------------------ B2 --
    print('\n[P5.8] B2: full 48-block base replay\n', flush=True)
    for variant, rescaled in (('CURRENT', False), ('RESCALED', True)):
        clone = srp._clone_operational_models(models)
        rows = []
        for tag, holder in O._tagged_holders(planning):
            for year in holder.years:
                for day in holder.days:
                    rows.append(run_block(clone, planning, tag, year, day,
                                          rescaled))
        summary = {
            'n_blocks': len(rows),
            'n_solved': sum(1 for r in rows if r['solved']),
            'total_weighted_base_before': sum(r['weighted_base_before']
                                              for r in rows),
            'total_weighted_base_after': sum(r['weighted_base_after']
                                             for r in rows),
            'total_base_delta': sum(r['base_delta'] for r in rows),
            'total_ipopt_iterations': sum(r['ipopt_iterations'] or 0
                                          for r in rows),
            'max_unscaled_dual_infeasibility': max(
                (r['ipopt_unscaled'].get('dual_infeasibility', 0.0) or 0.0)
                for r in rows),
            'max_unscaled_constraint_violation': max(
                (r['ipopt_unscaled'].get('constraint_violation', 0.0) or 0.0)
                for r in rows),
            'max_scaled_dual_infeasibility': max(
                (r['ipopt_scaled'].get('dual_infeasibility', 0.0) or 0.0)
                for r in rows),
            'max_consensus_residual_after': max(
                r['consensus_residual_after']['max'] for r in rows),
            'total_runtime_s': sum(r['runtime_s'] for r in rows),
        }
        report['B2_full_replay'][variant] = {'summary': summary,
                                             'per_block': rows}
        persist()
        print(f"      {variant:9s} solved {summary['n_solved']}/48  "
              f"total base delta {summary['total_base_delta']:+16.2f}  "
              f"iters {summary['total_ipopt_iterations']}  "
              f"max unscaled dual inf "
              f"{summary['max_unscaled_dual_infeasibility']:.4e}", flush=True)
        del clone

    current = report['B2_full_replay']['CURRENT']['summary']
    rescaled_summary = report['B2_full_replay']['RESCALED']['summary']
    report['B2_gap_reduction'] = {
        'current_total_base_delta': current['total_base_delta'],
        'rescaled_total_base_delta': rescaled_summary['total_base_delta'],
        'additional_base_objective_recovered_by_rescaling':
            rescaled_summary['total_base_delta'] - current['total_base_delta'],
    }
    persist()
    print(f'\n[P5.8] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
