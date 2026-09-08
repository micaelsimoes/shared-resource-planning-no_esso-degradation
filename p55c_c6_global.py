"""
Stage P5.5-C6 (global) -- the outer-relaxation check on the FULL centralized
convex model, not one block at a time.

Runs the production distributed operational planning on the canonical base
candidate, then maps the converged nonlinear state of every (agent, year, day)
model into the matching block of the centralized convex relaxation and asks two
questions:

  1. Is that mapped point feasible for the whole parent -- including the
     interface rows that replace the ADMM vmag / pf / ess consensus, and the
     capacity couplings that replace the ESSO consensus?
  2. Does the parent objective at the mapped point account exactly for the
     production gross operational cost, once the explicitly dropped
     non-negative complementarity penalties are added back?

Question 1 is the only real test of the interface couplings: an ADMM-converged
point satisfies the consensus only to the ADMM tolerance, so the residual there
is a measurement of that tolerance, not of a modelling error -- both numbers are
reported so they can be told apart.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55c_c6_global.py
"""

import io
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from convex_oracle import block_objective  # noqa: E402
from p55c_c6_mapping import constraint_violation, map_state  # noqa: E402

OBJECTIVE_FAMILIES = ('generation_cost', 'flexibility_cost', 'load_curtailment_cost',
                      'gen_curtailment_penalty', 'ess_utilization_cost_penalty',
                      'slack_penalties')


def objective_families(model, network, params, s_m, s_o):
    values = {}
    for name in OBJECTIVE_FAMILIES:
        try:
            values[name] = float(pe.value(
                getattr(mch, name)(model, network, s_m, s_o, params)))
        except Exception as error:
            values[name] = f'{type(error).__name__}: {error}'
    return values

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')


def violations(component_owner, descend):
    fams = defaultdict(lambda: {'n': 0, 'max_violation': 0.0, 'worst_index': None})
    for con in component_owner.component_objects(pe.Constraint, active=True,
                                                 descend_into=descend):
        name = con.local_name
        for idx in con:
            value = constraint_violation(con[idx])
            if value is None:
                continue
            fams[name]['n'] += 1
            if value > fams[name]['max_violation']:
                fams[name]['max_violation'] = value
                fams[name]['worst_index'] = str(idx)
    return {k: dict(v) for k, v in fams.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C6 global', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C6g] ABORTED\n{error}')
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    print('[C6g] running production distributed operational planning ...', flush=True)
    started = time.time()
    with redirect_stdout(io.StringIO()):
        convergence, results, models, sensitivities, primal_evolution, state = \
            planning.run_operational_planning(
                type='distributed', candidate_solution=deepcopy(candidate),
                print_results=False, debug_flag=False, return_state=True)
    runtime = time.time() - started
    diagnostics = state.get('admm_diagnostics', [])
    components = planning.get_operational_recourse_components(models)
    print(f'[C6g] converged={bool(convergence)} cycles={len(diagnostics)} '
          f'runtime={runtime:.1f}s', flush=True)
    print(f"[C6g] gross operational cost = {components['gross_operational_cost']:.6f}")
    print(f"[C6g] terminal salvage       = {components['terminal_salvage_value']:.6f}")
    print(f"[C6g] net recourse           = {components['net_operational_recourse']:.6f}")

    print('\n[C6g] building the centralized convex relaxation ...', flush=True)
    parent = build_centralized_relaxation(planning, candidate)
    size = model_size(parent)
    print(f"[C6g] {size['ac_blocks']} blocks, {size['variables']} vars, "
          f"{size['constraints']} cons", flush=True)

    print('[C6g] mapping the converged nonlinear state into every block ...', flush=True)
    per_block = {}
    dropped_total = 0.0
    q_mapped = 0.0
    for key, blk in parent.blocks.items():
        tag, year, day, s_m, s_o = key
        if tag == 'TSO':
            model = models['tso'][year][day]
            holder = planning.transmission_network
        else:
            node_id = int(tag[3:])
            model = models['dso'][node_id][year][day]
            holder = planning.distribution_networks[node_id]
        network = holder.network[year][day]
        params = holder.params
        weight = parent.block_weights[key]

        map_state(blk, model, network, s_m=s_m, s_o=s_o, clip=False)
        prob = (network.prob_market_scenarios[s_m]
                * network.prob_operation_scenarios[s_o])
        dropped_total += weight * prob * float(pe.value(
            mch.ess_complementarity_penalties_rule(model, s_m, s_o,
                                                   network=network, params=params)))
        # per-block objective accounting, family by family
        q_prod = float(pe.value(mch.objective_function_rule(model, params)))
        q_blk = float(pe.value(block_objective(blk)))
        comp = prob * float(pe.value(mch.ess_complementarity_penalties_rule(
            model, s_m, s_o, network=network, params=params)))
        prod_families = objective_families(model, network, params, s_m, s_o)
        blk_families = objective_families(blk, network, params, s_m, s_o)
        family_diff = {k: (prod_families[k] - blk_families[k])
                       for k in OBJECTIVE_FAMILIES
                       if isinstance(prod_families[k], float)
                       and isinstance(blk_families[k], float)
                       and abs(prod_families[k] - blk_families[k]) > 1e-12}

        block_fams = violations(blk, descend=False)
        worst = max((v['max_violation'] for v in block_fams.values()), default=0.0)
        prod_fams = violations(model, descend=False)
        prod_worst = max((v['max_violation'] for v in prod_fams.values()), default=0.0)
        per_block['|'.join(str(k) for k in key)] = {
            'weight': weight,
            'worst_violation': worst,
            'nonlinear_worst_violation': prod_worst,
            'excess_over_nonlinear': worst - prod_worst,
            'worst_family': max(block_fams.items(),
                                key=lambda kv: kv[1]['max_violation'])[0]
            if block_fams else None,
            'production_objective': q_prod,
            'block_objective': q_blk,
            'dropped_complementarity': comp,
            'objective_residual': q_prod - q_blk - comp,
            'weighted_objective_residual': weight * (q_prod - q_blk - comp),
            'objective_family_differences': family_diff,
        }

    # capacity handles: production's own available capacities
    available = planning.shared_ess_data.get_updated_capacities(models['esso'])
    for (node, year) in parent.S_rated:
        parent.S_rated[node, year].value = abs(candidate['total_capacity'][node][year]['s'])
        parent.E_rated[node, year].value = abs(candidate['total_capacity'][node][year]['e'])
        parent.S_available[node, year].value = available[node][year]['s_available']
        parent.E_available[node, year].value = available[node][year]['e_available']

    q_mapped = float(pe.value(parent.objective))

    parent_fams = violations(parent, descend=False)
    coupling = {k: v for k, v in parent_fams.items()}
    worst_coupling = max((v['max_violation'] for v in coupling.values()), default=0.0)
    worst_block = max(v['worst_violation'] for v in per_block.values())
    worst_excess = max(v['excess_over_nonlinear'] for v in per_block.values())

    residual = components['gross_operational_cost'] - q_mapped - dropped_total
    report = {
        'stage': 'P5.5-C6 global', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'production_run': {
            'converged': bool(convergence), 'cycles': len(diagnostics),
            'runtime_s': runtime,
            'gross_operational_cost': components['gross_operational_cost'],
            'terminal_salvage_value': components['terminal_salvage_value'],
            'net_operational_recourse': components['net_operational_recourse'],
        },
        'model_size': size,
        'objective_accounting': {
            'production_gross_operational_cost': components['gross_operational_cost'],
            'parent_objective_at_mapped_point': q_mapped,
            'dropped_complementarity_weighted': dropped_total,
            'residual': residual,
            'exact': abs(residual) < 1e-6 * max(
                abs(components['gross_operational_cost']), 1.0),
        },
        'worst_block_violation': worst_block,
        'worst_block_excess_over_nonlinear': worst_excess,
        'worst_coupling_violation': worst_coupling,
        'coupling_violations': coupling,
        'per_block': per_block,
    }

    out = os.path.join(OUT_DIR, 'p55c_c6_global.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('\n[C6g] objective accounting')
    print(f"    production gross operational cost = "
          f"{components['gross_operational_cost']:.6f}")
    print(f"    parent objective at mapped point  = {q_mapped:.6f}")
    print(f"    dropped complementarity (weighted)= {dropped_total:.6f}")
    print(f"    residual                          = {residual:.6e}  "
          f"exact={report['objective_accounting']['exact']}")
    print('\n[C6g] feasibility')
    print(f"    worst violation inside blocks       = {worst_block:.3e}")
    print(f"    worst excess over nonlinear's own   = {worst_excess:.3e}")
    print(f"    worst violation on coupling rows    = {worst_coupling:.3e}")
    for name, entry in sorted(coupling.items(),
                              key=lambda kv: -kv[1]['max_violation'])[:8]:
        print(f"      {name:24s} {entry['max_violation']:.3e}  n={entry['n']:<6d} "
              f"worst={entry['worst_index']}")
    worst_obj = sorted(per_block.items(),
                       key=lambda kv: -abs(kv[1]['weighted_objective_residual']))
    print('\n[C6g] objective residual by block (worst 6)')
    for name, entry in worst_obj[:6]:
        print(f"      {name:26s} weighted={entry['weighted_objective_residual']:14.4f} "
              f"raw={entry['objective_residual']:.6e}")
        for fam, diff in entry['objective_family_differences'].items():
            print(f"          {fam:32s} production - block = {diff:.6e}")

    print(f'\n[C6g] report -> {out}')


if __name__ == '__main__':
    main()
