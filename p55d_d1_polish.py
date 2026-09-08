"""
Stage P5.5-D1 -- exact-consensus nonlinear polish.

P5.5-C6's global audit mapped an ADMM-CONVERGED point into the centralized convex
parent.  Such a point satisfies the consensus only to the ADMM tolerance, so the
interface rows showed residuals (worst 14.16 kV^2 on voltage, 1.341 MW on
interface active power) and the audit could not be called a proof that an exactly
feasible centralized nonlinear point maps into the parent.

This script removes that caveat.  Starting from the canonical converged ADMM
state it

  1. forms ONE common value for every coordinated physical quantity
     (interface P, interface Q, interface voltage, shared-ESS P, shared-ESS Q;
     available S/E are already common because both sides are set from the same
     candidate);
  2. fixes those common values in the ORIGINAL nonlinear local models and
     re-solves each with its own base objective;
  3. maps the polished point into the centralized convex parent and re-runs the
     C6 mapped-point audit.

Production ADMM and its convergence logic are NOT modified.  The models returned
by `run_operational_planning` are post-processed here as ordinary Pyomo objects:
the ADMM objective is deactivated and the base objective reactivated, because the
quantity being polished is the recourse, which `get_primal_value` measures on the
base objective.

Because both sides are fixed to the same number, the interface mismatch of the
polished point is zero by construction; what is actually being tested is whether
the local nonlinear models remain FEASIBLE at the common values, and what the
resulting rigorous upper-bound incumbent costs.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55d_d1_polish.py
"""

import io
import json
import os
import pickle
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
from convex_oracle import block_objective  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from p55c_c6_mapping import constraint_violation, map_state  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55D')
TARGET = 1e-7


def interface_values(planning, models):
    """Read each side's interface quantities from the converged ADMM models."""
    values = defaultdict(dict)
    tso = planning.transmission_network
    for node_id, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net = tso.network[year][day]
                d_net = dso.network[year][day]
                t_model = models['tso'][year][day]
                d_model = models['dso'][node_id][year][day]
                dn = list(t_net.active_distribution_network_nodes).index(node_id)
                adn_idx = t_net.get_node_idx(node_id)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                t_sess = [e for e, s in enumerate(t_net.shared_energy_storages)
                          if s.bus == node_id]
                d_sess = [e for e, s in enumerate(d_net.shared_energy_storages)
                          if s.bus == ref_id]
                for p in t_model.periods:
                    t_p = float(pe.value(t_model.pc_adn[dn, 0, 0, p]))
                    t_q = float(pe.value(t_model.qc_adn[dn, 0, 0, p]))
                    d_p = float(pe.value(d_model.pg_adn[0, 0, p]))
                    d_q = float(pe.value(d_model.qg_adn[0, 0, p]))
                    t_v = float(pe.value(t_model.vmag_sqr[adn_idx, 0, 0, p])) ** 0.5
                    d_v = float(pe.value(d_model.vmag_sqr[ref_idx, 0, 0, p])) ** 0.5
                    t_sp = sum(float(pe.value(t_model.shared_es_pnet[e, 0, 0, p]))
                               for e in t_sess)
                    t_sq = sum(float(pe.value(t_model.shared_es_qnet[e, 0, 0, p]))
                               for e in t_sess)
                    d_sp = sum(float(pe.value(d_model.shared_es_pnet[e, 0, 0, p]))
                               for e in d_sess)
                    d_sq = sum(float(pe.value(d_model.shared_es_qnet[e, 0, 0, p]))
                               for e in d_sess)
                    values[(node_id, year, day, p)] = {
                        'tso_p': t_p, 'dso_p': d_p, 'tso_q': t_q, 'dso_q': d_q,
                        'tso_v': t_v, 'dso_v': d_v,
                        'tso_sess_p': t_sp, 'dso_sess_p': d_sp,
                        'tso_sess_q': t_sq, 'dso_sess_q': d_sq,
                        # the common value: the midpoint, which minimises the
                        # largest shift asked of either side
                        'common_p': 0.5 * (t_p + d_p), 'common_q': 0.5 * (t_q + d_q),
                        'common_v': 0.5 * (t_v + d_v),
                        'common_sess_p': 0.5 * (t_sp + d_sp),
                        'common_sess_q': 0.5 * (t_sq + d_sq),
                    }
    return values


def apply_common_values(planning, models, values):
    """Fix every coordinated quantity to its common value, both sides."""
    tso = planning.transmission_network
    for node_id, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net = tso.network[year][day]
                d_net = dso.network[year][day]
                t_model = models['tso'][year][day]
                d_model = models['dso'][node_id][year][day]
                adn_idx = t_net.get_node_idx(node_id)
                adn_load = t_net.get_adn_load_idx(node_id)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                ref_gen = d_net.get_reference_gen_idx()
                t_sess = [e for e, s in enumerate(t_net.shared_energy_storages)
                          if s.bus == node_id]
                d_sess = [e for e, s in enumerate(d_net.shared_energy_storages)
                          if s.bus == ref_id]
                for p in t_model.periods:
                    entry = values[(node_id, year, day, p)]
                    # --- transmission side -------------------------------
                    # pc_adn = pc + flex_up - flex_down; pin the flex to zero so
                    # the fixed pc IS the interface power
                    t_model.pc[adn_load, 0, 0, p].fix(entry['common_p'])
                    t_model.qc[adn_load, 0, 0, p].fix(entry['common_q'])
                    for name in ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down'):
                        if hasattr(t_model, name):
                            getattr(t_model, name)[adn_load, 0, 0, p].fix(0.0)
                    # voltage is pinned through the squared magnitude, which
                    # leaves the angle free
                    t_model.vmag_sqr[adn_idx, 0, 0, p].fix(entry['common_v'] ** 2)
                    # --- distribution side -------------------------------
                    d_model.vmag_sqr[ref_idx, 0, 0, p].fix(entry['common_v'] ** 2)
                    # pg_adn = pg[ref_gen] - sum shared_es_pnet
                    d_model.pg[ref_gen, 0, 0, p].fix(
                        entry['common_p'] + entry['common_sess_p'])
                    d_model.qg[ref_gen, 0, 0, p].fix(
                        entry['common_q'] + entry['common_sess_q'])
                    # --- the shared ESS itself, common on both sides -----
                    for e in t_sess:
                        t_model.shared_es_pnet[e, 0, 0, p].fix(entry['common_sess_p'])
                        t_model.shared_es_qnet[e, 0, 0, p].fix(entry['common_sess_q'])
                    for e in d_sess:
                        d_model.shared_es_pnet[e, 0, 0, p].fix(entry['common_sess_p'])
                        d_model.shared_es_qnet[e, 0, 0, p].fix(entry['common_sess_q'])


def restore_base_objective(model):
    """Polish against the base objective, which is what the recourse measures."""
    if hasattr(model, 'admm_objective') and model.admm_objective.active:
        model.admm_objective.deactivate()
    model.objective.activate()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-D1 polish', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D1] ABORTED\n{error}')
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    print('[D1] running the canonical production distributed planning ...', flush=True)
    started = time.time()
    with redirect_stdout(io.StringIO()):
        convergence, results, models, sensitivities, primal_evolution, state = \
            planning.run_operational_planning(
                type='distributed', candidate_solution=deepcopy(candidate),
                print_results=False, debug_flag=False, return_state=True)
    admm_runtime = time.time() - started
    admm_components = planning.get_operational_recourse_components(models)
    print(f'[D1] ADMM converged={bool(convergence)} '
          f"cycles={len(state.get('admm_diagnostics', []))} "
          f'runtime={admm_runtime:.1f}s', flush=True)
    print(f"[D1] ADMM gross={admm_components['gross_operational_cost']:.6f} "
          f"net recourse={admm_components['net_operational_recourse']:.6f}", flush=True)

    values = interface_values(planning, models)
    before = {
        'p': max(abs(v['tso_p'] - v['dso_p']) for v in values.values()),
        'q': max(abs(v['tso_q'] - v['dso_q']) for v in values.values()),
        'v': max(abs(v['tso_v'] - v['dso_v']) for v in values.values()),
        'sess_p': max(abs(v['tso_sess_p'] - v['dso_sess_p']) for v in values.values()),
        'sess_q': max(abs(v['tso_sess_q'] - v['dso_sess_q']) for v in values.values()),
    }
    print('\n[D1] ADMM-converged interface mismatch (p.u.)')
    for key, val in before.items():
        print(f'    {key:8s} {val:.6e}')

    # The shared available S/E are the remaining coordinated quantity.  The
    # networks were given their capacities earlier in the ADMM loop, while the
    # ESSO's final available capacity moved slightly afterwards; making them
    # common is part of "one common value for every coordinated quantity".
    available = planning.shared_ess_data.get_updated_capacities(models['esso'])
    capacity_shift = 0.0
    for tag, holder in ([('TSO', planning.transmission_network)]
                        + [(f'DSO{n}', h)
                           for n, h in sorted(planning.distribution_networks.items())]):
        node_of = (None if tag == 'TSO' else int(tag[3:]))
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = models['tso'][year][day] if tag == 'TSO' \
                    else models['dso'][node_of][year][day]
                if node_of is None:
                    pairs = [(e, ess.bus)
                             for e, ess in enumerate(network.shared_energy_storages)]
                else:
                    ref = network.get_reference_node_id()
                    pairs = [(network.get_shared_energy_storage_idx(ref), node_of)]
                for e_idx, esso_node in pairs:
                    s_pu = available[esso_node][year]['s_available'] / network.baseMVA
                    e_pu = available[esso_node][year]['e_available'] / network.baseMVA
                    capacity_shift = max(
                        capacity_shift,
                        abs(float(pe.value(model.shared_es_e_rated_fixed[e_idx]))
                            - e_pu) * network.baseMVA)
                    model.shared_es_s_rated_fixed[e_idx].set_value(s_pu)
                    model.shared_es_e_rated_fixed[e_idx].set_value(e_pu)
                    mch.configure_shared_ess_operational_state(model, e_idx,
                                                               s_pu, e_pu)
    print(f'\n[D1] shared available S/E made common; largest capacity shift '
          f'{capacity_shift:.6e} MVAh', flush=True)

    print('\n[D1] fixing common values and re-solving every local model ...', flush=True)
    apply_common_values(planning, models, values)

    polish = {'blocks': [], 'all_solved': True}
    started = time.time()
    for year in planning.transmission_network.years:
        for day in planning.transmission_network.days:
            model = models['tso'][year][day]
            network = planning.transmission_network.network[year][day]
            restore_base_objective(model)
            with redirect_stdout(io.StringIO()):
                result = network.run_smopf(model, planning.transmission_network.params,
                                           print_header=False)
            ok = bool(srp._solver_result_succeeded(result))
            polish['all_solved'] &= ok
            polish['blocks'].append({'agent': 'TSO', 'year': year, 'day': day,
                                     'solved': ok})
            print(f'    TSO  {year} {day:8s} solved={ok}', flush=True)
    for node_id, dso in sorted(planning.distribution_networks.items()):
        for year in dso.years:
            for day in dso.days:
                model = models['dso'][node_id][year][day]
                network = dso.network[year][day]
                restore_base_objective(model)
                with redirect_stdout(io.StringIO()):
                    result = network.run_smopf(model, dso.params, print_header=False)
                ok = bool(srp._solver_result_succeeded(result))
                polish['all_solved'] &= ok
                polish['blocks'].append({'agent': f'DSO{node_id}', 'year': year,
                                         'day': day, 'solved': ok})
                print(f'    DSO{node_id} {year} {day:8s} solved={ok}', flush=True)
    polish['runtime_s'] = time.time() - started

    after_values = interface_values(planning, models)
    after = {
        'p': max(abs(v['tso_p'] - v['dso_p']) for v in after_values.values()),
        'q': max(abs(v['tso_q'] - v['dso_q']) for v in after_values.values()),
        'v': max(abs(v['tso_v'] - v['dso_v']) for v in after_values.values()),
        'sess_p': max(abs(v['tso_sess_p'] - v['dso_sess_p'])
                      for v in after_values.values()),
        'sess_q': max(abs(v['tso_sess_q'] - v['dso_sess_q'])
                      for v in after_values.values()),
    }
    print('\n[D1] polished interface mismatch (p.u.)')
    for key, val in after.items():
        flag = 'OK' if val < TARGET else 'ABOVE TARGET'
        print(f'    {key:8s} {val:.6e}   {flag}')

    polished_components = planning.get_operational_recourse_components(models)
    print(f"\n[D1] polished gross         = "
          f"{polished_components['gross_operational_cost']:.6f}")
    print(f"[D1] polished net recourse  = "
          f"{polished_components['net_operational_recourse']:.6f}")

    # ------------------------------------------------------------- C6 re-audit
    print('\n[D1] mapping the polished point into the convex parent ...', flush=True)
    parent = build_centralized_relaxation(planning, candidate)
    dropped_total, per_block = 0.0, {}
    for key, blk in parent.blocks.items():
        tag, year, day, s_m, s_o = key
        if tag == 'TSO':
            model, holder = models['tso'][year][day], planning.transmission_network
        else:
            node_id = int(tag[3:])
            model = models['dso'][node_id][year][day]
            holder = planning.distribution_networks[node_id]
        network, params = holder.network[year][day], holder.params
        weight = parent.block_weights[key]
        map_state(blk, model, network, s_m=s_m, s_o=s_o, clip=False)
        prob = (network.prob_market_scenarios[s_m]
                * network.prob_operation_scenarios[s_o])
        dropped_total += weight * prob * float(pe.value(
            mch.ess_complementarity_penalties_rule(model, s_m, s_o,
                                                   network=network, params=params)))
        worst_blk, worst_prod = 0.0, 0.0
        for con in blk.component_objects(pe.Constraint, active=True,
                                         descend_into=False):
            for idx in con:
                v = constraint_violation(con[idx])
                if v is not None:
                    worst_blk = max(worst_blk, v)
        for con in model.component_objects(pe.Constraint, active=True,
                                           descend_into=False):
            for idx in con:
                v = constraint_violation(con[idx])
                if v is not None:
                    worst_prod = max(worst_prod, v)
        per_block['|'.join(str(k) for k in key)] = {
            'worst_violation': worst_blk, 'nonlinear_worst_violation': worst_prod,
            'excess_over_nonlinear': worst_blk - worst_prod,
            'weight': weight,
            'weighted_block_objective_at_polished_point':
                weight * float(pe.value(block_objective(blk))),
            'production_objective':
                float(pe.value(mch.objective_function_rule(model, params))),
        }

    for (node, year) in parent.S_rated:
        parent.S_rated[node, year].value = abs(candidate['total_capacity'][node][year]['s'])
        parent.E_rated[node, year].value = abs(candidate['total_capacity'][node][year]['e'])
        parent.S_available[node, year].value = available[node][year]['s_available']
        parent.E_available[node, year].value = available[node][year]['e_available']

    q_mapped = float(pe.value(parent.objective))
    coupling = {}
    for con in parent.component_objects(pe.Constraint, active=True,
                                        descend_into=False):
        worst, worst_idx = 0.0, None
        n = 0
        for idx in con:
            v = constraint_violation(con[idx])
            if v is None:
                continue
            n += 1
            if v > worst:
                worst, worst_idx = v, str(idx)
        coupling[con.local_name] = {'n': n, 'max_violation': worst,
                                    'worst_index': worst_idx}
    worst_coupling = max((v['max_violation'] for v in coupling.values()), default=0.0)
    worst_block = max(v['worst_violation'] for v in per_block.values())
    worst_excess = max(v['excess_over_nonlinear'] for v in per_block.values())
    residual = polished_components['gross_operational_cost'] - q_mapped - dropped_total

    tolerance_limited = not all(after[k] < TARGET for k in ('p', 'q', 'v'))
    verdict = ('PASS' if (polish['all_solved'] and not tolerance_limited
                          and worst_coupling < 1e-6) else
               'tolerance-limited' if polish['all_solved'] else 'FAIL')

    report = {
        'stage': 'P5.5-D1', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'admm': {'converged': bool(convergence),
                 'cycles': len(state.get('admm_diagnostics', [])),
                 'runtime_s': admm_runtime,
                 'gross_operational_cost': admm_components['gross_operational_cost'],
                 'net_operational_recourse':
                     admm_components['net_operational_recourse']},
        'shared_available_capacity_made_common': True,
        'largest_capacity_shift_MVAh': capacity_shift,
        'interface_mismatch_before': before,
        'interface_mismatch_after': after,
        'target': TARGET,
        'polish': polish,
        'polished_gross_operational_cost':
            polished_components['gross_operational_cost'],
        'polished_net_operational_recourse':
            polished_components['net_operational_recourse'],
        'polished_terminal_salvage': polished_components['terminal_salvage_value'],
        'rigorous_feasible_UB_incumbent': (
            polished_components['net_operational_recourse']
            if verdict != 'FAIL' else None),
        'model_size': model_size(parent),
        'c6_global_exact': {
            'parent_objective_at_mapped_point': q_mapped,
            'dropped_complementarity_weighted': dropped_total,
            'residual': residual,
            'exact': abs(residual) < 1e-6 * max(
                abs(polished_components['gross_operational_cost']), 1.0),
            'worst_block_violation': worst_block,
            'worst_block_excess_over_nonlinear': worst_excess,
            'worst_coupling_violation': worst_coupling,
            'coupling_violations': coupling,
            'per_block': per_block,
        },
        'C6_local_mapping': 'PASS (P5.5-C, unchanged)',
        'C6_global_exact_coupling_mapping': verdict,
    }

    with open(os.path.join(OUT_DIR, 'p55d_d1_polish.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # the polished ESS schedule is what D4 needs for its mode assignment
    schedule = {}
    for node_id, dso in sorted(planning.distribution_networks.items()):
        for year in dso.years:
            for day in dso.days:
                d_net = dso.network[year][day]
                d_model = models['dso'][node_id][year][day]
                ref_id = d_net.get_reference_node_id()
                e_idx = d_net.get_shared_energy_storage_idx(ref_id)
                for p in d_model.periods:
                    schedule[f'{node_id}|{year}|{day}|{p}'] = {
                        'pch': float(pe.value(d_model.shared_es_pch[e_idx, 0, 0, p])),
                        'pdch': float(pe.value(d_model.shared_es_pdch[e_idx, 0, 0, p])),
                        'pnet': float(pe.value(d_model.shared_es_pnet[e_idx, 0, 0, p])),
                    }
    with open(os.path.join(OUT_DIR, 'p55d_d1_ess_schedule.json'), 'w') as handle:
        json.dump(schedule, handle, indent=1)

    print('\n[D1] C6 re-audit on the polished point')
    print(f"    parent objective at mapped point = {q_mapped:.6f}")
    print(f"    accounting residual              = {residual:.6e} "
          f"exact={report['c6_global_exact']['exact']}")
    print(f"    worst block violation            = {worst_block:.3e}")
    print(f"    worst excess over nonlinear      = {worst_excess:.3e}")
    print(f"    worst coupling violation         = {worst_coupling:.3e}")
    for name, entry in sorted(coupling.items(),
                              key=lambda kv: -kv[1]['max_violation'])[:8]:
        print(f"      {name:24s} {entry['max_violation']:.3e}  n={entry['n']}")
    print(f"\n[D1] C6-local mapping                     : PASS")
    print(f"[D1] C6-global exact-coupling mapping     : {verdict}")
    print(f"\n[D1] report -> {os.path.join(OUT_DIR, 'p55d_d1_polish.json')}")


if __name__ == '__main__':
    main()
