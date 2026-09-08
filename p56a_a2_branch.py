"""
Stage P5.6-A2.1 / A2.2 -- base versus augmented objective, and branch fingerprint.

A2 attributed the ADMM-to-polished change by objective family.  What it cannot
say on its own is WHY a local model moves at all once every coordinated variable
is fixed.  Two explanations are possible and they have different consequences:

  (a) the ADMM's augmented objective genuinely prefers a different dispatch, in
      which case the polish is an economic effect; or
  (b) after fixing the consensus the two objectives differ by a constant (and a
      positive scale), so they have the SAME optimizer set, and any difference
      the solver returns is local-NLP branch / solver-path selection.

A2.1 decides between them numerically rather than by argument.  At exactly the
same fixed consensus values and exactly the same initial point, one representative
affected local problem is solved twice -- once with the production augmented
objective, once with the base recourse objective -- and the two objectives are
evaluated at three common points to test whether

        augmented  =  alpha * base + beta,     alpha > 0 constant

holds on the fixed-consensus set.

A2.2 fingerprints the dominant blocks so the recovered branch can be compared
with the P5.4-D4 evidence.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56a_a2_branch.py
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import p56a_candidates as C  # noqa: E402
import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = O.OUT_DIR
# the dominant block under the corrected polish, plus the historically
# problematic TSO Spring block
PROBE_BLOCKS = [('DSO5', 2035, 'Autumn'), ('TSO', 2025, 'Spring')]


def snapshot(model):
    return {id(v): v.value for v in model.component_data_objects(pe.Var,
                                                                 active=True)}


def restore(model, state):
    for v in model.component_data_objects(pe.Var, active=True):
        if id(v) in state:
            v.value = state[id(v)]


def block_fingerprint(model, network, params):
    def total(name):
        try:
            return float(pe.value(getattr(mch, name)(model, network, 0, 0, params)))
        except Exception:
            return None

    def var_sum(name):
        if not hasattr(model, name):
            return None
        return float(sum(abs(v.value) for v in getattr(model, name).values()
                         if v.value is not None))

    return {
        'base_objective': float(pe.value(
            mch.objective_function_rule(model, params))),
        'generation_cost': total('generation_cost'),
        'flexibility_cost': total('flexibility_cost'),
        'gen_curtailment_penalty': total('gen_curtailment_penalty'),
        'slack_penalties': total('slack_penalties'),
        'sum_abs_pg': var_sum('pg'),
        'sum_abs_qg': var_sum('qg'),
        'sum_abs_flex_p_down': var_sum('flex_p_down'),
        'sum_abs_flex_q_down': var_sum('flex_q_down'),
        'sum_abs_slack_v_sqr_up': var_sum('slack_v_sqr_up'),
        'sum_abs_slack_v_sqr_down': var_sum('slack_v_sqr_down'),
        'sum_abs_shared_es_pch': var_sum('shared_es_pch'),
        'sum_abs_shared_es_pdch': var_sum('shared_es_pdch'),
        'sum_abs_vmag_sqr': var_sum('vmag_sqr'),
    }


def solver_stats(result):
    stats = {}
    try:
        stats['termination'] = str(result.solver.termination_condition)
        stats['status'] = str(result.solver.status)
    except Exception:
        pass
    for key in ('iterations', 'Number of Iterations'):
        try:
            stats['iterations'] = int(result.solver[key])
            break
        except Exception:
            continue
    for key, attr in (('dual_infeasibility', 'Dual infeasibility'),
                      ('primal_infeasibility', 'Primal infeasibility')):
        try:
            stats[key] = float(result.solver[attr])
        except Exception:
            pass
    return stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-A2.1/A2.2 branch', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A2.1] ABORTED\n{error}')
        sys.exit(1)

    planning = O.fresh_planning('a2_branch')
    x = C.base_vector(planning)
    candidate = O.vector_to_candidate(planning, x)

    print('[A2.1] running the canonical ADMM to obtain the fixed consensus ...',
          flush=True)
    started = time.time()
    with redirect_stdout(io.StringIO()):
        _, _, models, _, _, state = planning.run_operational_planning(
            type='distributed', candidate_solution=deepcopy(candidate),
            print_results=False, debug_flag=False, return_state=True)
    print(f'[A2.1] ADMM done in {time.time() - started:.1f}s', flush=True)

    common = O.common_coordinated_values(planning, models,
                                         state['consensus_vars'])
    request = O.esso_request_from_common(planning, state['consensus_vars'],
                                         common)
    esso_models, _, esso_solved, available = O.solve_physical_esso(
        planning, candidate, request)
    O.apply_physical_capacities(planning, models, available)
    O.apply_common_values(planning, models, common)

    report = {'stage': 'P5.6-A2.1/A2.2', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'esso_solved': esso_solved, 'blocks': {}}

    for tag, year, day in PROBE_BLOCKS:
        node_of = None if tag == 'TSO' else int(tag[3:])
        holder = (planning.transmission_network if tag == 'TSO'
                  else planning.distribution_networks[node_of])
        network = holder.network[year][day]
        model = (models['tso'][year][day] if tag == 'TSO'
                 else models['dso'][node_of][year][day])
        key = f'{tag}|{year}|{day}'
        print(f'\n[A2.1] === {key}', flush=True)

        has_augmented = hasattr(model, 'admm_objective')
        start_state = snapshot(model)
        entry = {'has_augmented_objective': has_augmented,
                 'common_initial_point': 'the ADMM-converged local iterate'}

        # ---- A : the production augmented / local objective ----
        if has_augmented:
            model.objective.deactivate()
            model.admm_objective.activate()
            restore(model, start_state)
            with redirect_stdout(io.StringIO()):
                result_a = network.run_smopf(model, holder.params,
                                             print_header=False)
            entry['A_augmented'] = {
                'solved': bool(srp._solver_result_succeeded(result_a)),
                'solver': solver_stats(result_a),
                'fingerprint': block_fingerprint(model, network, holder.params),
                'augmented_objective': float(pe.value(model.admm_objective)),
            }
            state_a = snapshot(model)
        else:
            entry['A_augmented'] = None
            state_a = None

        # ---- B : the base recourse objective, same initial point ----
        if has_augmented:
            model.admm_objective.deactivate()
        model.objective.activate()
        restore(model, start_state)
        with redirect_stdout(io.StringIO()):
            result_b = network.run_smopf(model, holder.params,
                                         print_header=False)
        entry['B_base'] = {
            'solved': bool(srp._solver_result_succeeded(result_b)),
            'solver': solver_stats(result_b),
            'fingerprint': block_fingerprint(model, network, holder.params),
        }
        state_b = snapshot(model)
        if has_augmented:
            entry['B_base']['augmented_objective'] = float(
                pe.value(model.admm_objective))

        # ---- is augmented = alpha * base + beta on the fixed set? ----
        if has_augmented and state_a is not None:
            points = {}
            for name, st in (('admm_iterate', start_state),
                             ('solution_A', state_a),
                             ('solution_B', state_b)):
                restore(model, st)
                points[name] = {
                    'base': float(pe.value(
                        mch.objective_function_rule(model, holder.params))),
                    'augmented': float(pe.value(model.admm_objective)),
                }
            names = list(points)
            alphas = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    db = points[names[j]]['base'] - points[names[i]]['base']
                    da = (points[names[j]]['augmented']
                          - points[names[i]]['augmented'])
                    alphas.append({'pair': f'{names[i]}->{names[j]}',
                                   'delta_base': db, 'delta_augmented': da,
                                   'ratio': (da / db) if abs(db) > 1e-12 else None})
            ratios = [a['ratio'] for a in alphas if a['ratio'] is not None]
            entry['affine_equivalence'] = {
                'points': points, 'pairs': alphas,
                'ratios': ratios,
                'ratio_spread': (max(ratios) - min(ratios)) if len(ratios) > 1
                                else None,
                'consistent_positive_scale': (
                    bool(ratios) and all(r > 0 for r in ratios)
                    and (max(ratios) - min(ratios)) <= 1e-6 * max(
                        abs(max(ratios)), 1.0)),
            }
            entry['objective_difference_A_minus_B'] = (
                entry['A_augmented']['fingerprint']['base_objective']
                - entry['B_base']['fingerprint']['base_objective'])
        report['blocks'][key] = entry

        print(f"      augmented objective present : {has_augmented}")
        if has_augmented:
            ae = entry['affine_equivalence']
            print(f"      ratios d(augmented)/d(base) : "
                  + ', '.join(f'{r:.9f}' for r in ae['ratios']))
            print(f"      consistent positive scale   : "
                  f"{ae['consistent_positive_scale']}")
            print(f"      base objective, A (augmented) : "
                  f"{entry['A_augmented']['fingerprint']['base_objective']:.6f}")
        print(f"      base objective, B (base)      : "
              f"{entry['B_base']['fingerprint']['base_objective']:.6f}")
        if has_augmented:
            print(f"      A - B                         : "
                  f"{entry['objective_difference_A_minus_B']:.6e}")
        for label in ('A_augmented', 'B_base'):
            if entry.get(label):
                print(f"      {label:12s} solver: {entry[label]['solver']}")

    out = os.path.join(OUT_DIR, 'p56a_a2_branch.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[A2.1/A2.2] report -> {out}')


if __name__ == '__main__':
    main()
