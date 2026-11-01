"""
Stage P5.4-D4 -- recourse branch recovery and oracle hardening.

D3 showed that candidates with SMALLER shared-ESS capacity converge to a recourse
~11.5e6 below the base candidate's own cold solve. Since a smaller capacity gives
a SUBSET feasible set, those operating points are feasible at the base candidate
too. D4 asks whether the base candidate can actually be made to attain them.

The lift (D4.2) uses the production warm-start path: `run_operational_planning`
clones `initial_state['models']`, restores the ADMM consensus/dual variables and
then calls `_update_operational_models_with_candidate` to move the capacities to
the target candidate. That preserves every physical primal value.

One thing production does NOT do is recompute the P5.4-H1 dimensionless
variables. After a capacity change the stale `pch_hat` corresponds to the SOURCE
rating, so the link row `pch - S*pch_hat = 0` is violated at the transferred
point. This script repairs them in the INITIALIZATION TRANSFER ONLY:

    pch_hat  = pch  / S_base      (clipped into [0, 1])
    pdch_hat = pdch / S_base

No division by S is introduced into the optimization model itself.

Because the source capacity is smaller than the base capacity, pch <= S_source <
S_base, so the repaired hats are automatically inside [0, 1].

    python p54d4_branch_recovery.py --source s:9:-0.05 --policies A,B
"""

import argparse
import io
import json
import os
import subprocess
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
import shared_resources_planning as srp  # noqa: E402
from definitions import (ESS_COMPLEMENTARITY_TOLERANCE,  # noqa: E402
                         SHARED_ESS_ZERO_CAPACITY_TOLERANCE)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D4')
Q_BASE_COLD = 848258809.8141167

MULTIPLIER_SUFFIXES = ('ipopt_zL_in', 'ipopt_zU_in', 'dual')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def run_candidate(planning, candidate, initial_state=None, keep_state=True):
    console = io.StringIO()
    started = time.time()
    with redirect_stdout(console):
        convergence, results, models, sensitivities, primal_evolution, state = \
            planning.run_operational_planning(
                type='distributed', candidate_solution=deepcopy(candidate),
                print_results=False, debug_flag=False,
                initial_state=initial_state, return_state=True)
    runtime = time.time() - started
    diags = state.get('admm_diagnostics', [])
    last = diags[-1] if diags else {}
    out = {
        'converged': bool(convergence), 'runtime_s': runtime,
        'initialization_failed': bool(state.get('initialization_failed', False)),
        'n_cycles': len(diags),
        'all_local_solves_ok': all(d.get('local_solves_ok') for d in diags) if diags else None,
        'n_recovery_diagnostics': len(state.get('solver_recovery_diagnostics', [])),
        'recourse': last.get('recourse'),
        'gross_operational_cost': last.get('gross_operational_cost'),
        'terminal_salvage_value': last.get('terminal_salvage_value'),
        'final_primal': {k: last.get(k) for k in ('primal_v', 'primal_pf', 'primal_ess')},
        'final_dual': {k: last.get(k) for k in ('dual_v', 'dual_pf', 'dual_ess')},
        'sensitivities': sensitivities,
    }
    if keep_state:
        out['_state'] = state
        out['_models'] = models
    return out


def repair_hat_variables(models, planning):
    """D4.2: recompute the H1 dimensionless variables for the CURRENT rating.

    Initialization transfer only -- the model equations are untouched.
    """
    repaired = {'n_entries': 0, 'max_hat': 0.0, 'n_clipped': 0, 'n_zero_capacity': 0}

    def fix_model(model, indices):
        for e in indices:
            s = float(pe.value(model.shared_es_s_rated_fixed[e]))
            for entry_ch, entry_hat in (('shared_es_pch', 'shared_es_pch_hat'),
                                        ('shared_es_pdch', 'shared_es_pdch_hat')):
                phys = getattr(model, entry_ch)
                hat = getattr(model, entry_hat)
                for idx in mch._component_entries_for_shared_ess(phys, e):
                    key = idx.index()
                    h = hat[key]
                    if h.fixed:
                        continue
                    if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
                        h.set_value(0.0)
                        repaired['n_zero_capacity'] += 1
                        continue
                    value = float(pe.value(idx)) / s
                    if value > 1.0:
                        value = 1.0
                        repaired['n_clipped'] += 1
                    if value < 0.0:
                        value = 0.0
                        repaired['n_clipped'] += 1
                    h.set_value(value)
                    repaired['n_entries'] += 1
                    repaired['max_hat'] = max(repaired['max_hat'], value)

    for year, per_day in models['tso'].items():
        for day, model in per_day.items():
            fix_model(model, list(model.shared_energy_storages))
    for node_id, per_year in models['dso'].items():
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                fix_model(model, list(model.shared_energy_storages))
    return repaired


def transferred_point_residuals(models, planning):
    """D4.2: does the transferred physical point satisfy the BASE candidate's
    ESS constraints before any solve?"""
    worst = {'active_sum': 0.0, 'capability': 0.0, 'complementarity': 0.0,
             'soc_upper': 0.0, 'soc_lower': 0.0, 'link_ch': 0.0, 'link_dch': 0.0}
    n_rows = 0

    def check(model, e):
        nonlocal n_rows
        s = float(pe.value(model.shared_es_s_rated_fixed[e]))
        e_cap = float(pe.value(model.shared_es_e_rated_fixed[e]))
        if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
            return
        for p in model.periods:
            key = (e, 0, 0, p)
            pch = float(pe.value(model.shared_es_pch[key]))
            pdch = float(pe.value(model.shared_es_pdch[key]))
            pnet = float(pe.value(model.shared_es_pnet[key]))
            qnet = float(pe.value(model.shared_es_qnet[key]))
            hc = float(pe.value(model.shared_es_pch_hat[key]))
            hd = float(pe.value(model.shared_es_pdch_hat[key]))
            soc = float(pe.value(model.shared_es_soc[key]))
            worst['active_sum'] = max(worst['active_sum'], (pch + pdch - s) / s)
            worst['capability'] = max(worst['capability'],
                                      (pnet ** 2 + qnet ** 2 - s ** 2) / s ** 2)
            worst['complementarity'] = max(worst['complementarity'],
                                           hc * hd - ESS_COMPLEMENTARITY_TOLERANCE)
            worst['link_ch'] = max(worst['link_ch'], abs(pch - s * hc) / s)
            worst['link_dch'] = max(worst['link_dch'], abs(pdch - s * hd) / s)
            if e_cap > 0:
                worst['soc_upper'] = max(worst['soc_upper'], (soc - 0.9 * e_cap) / e_cap)
                worst['soc_lower'] = max(worst['soc_lower'], (0.1 * e_cap - soc) / e_cap)
            n_rows += 1

    for year, per_day in models['tso'].items():
        for day, model in per_day.items():
            for e in model.shared_energy_storages:
                check(model, e)
    for node_id, per_year in models['dso'].items():
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                for e in model.shared_energy_storages:
                    check(model, e)
    worst['n_rows_checked'] = n_rows
    worst['max_violation'] = max(v for k, v in worst.items() if k != 'n_rows_checked')
    return worst


def clear_multipliers(models):
    n = 0
    def strip(model):
        nonlocal n
        for name in MULTIPLIER_SUFFIXES:
            if hasattr(model, name):
                suffix = getattr(model, name)
                n += len(suffix)
                suffix.clear()
    for year, per_day in models['tso'].items():
        for day, model in per_day.items():
            strip(model)
    for node_id, per_year in models['dso'].items():
        for year, per_day in per_year.items():
            for day, model in per_day.items():
                strip(model)
    for node_id, model in models['esso'].items():
        strip(model)
    return n


def lift_state(planning, source_state, base_candidate, policy):
    """Build a base-candidate initial_state from a converged smaller-S state."""
    models = srp._clone_operational_models(source_state['models'])
    srp._update_operational_models_with_candidate(planning, models, base_candidate)
    repaired = repair_hat_variables(models, planning)
    residuals = transferred_point_residuals(models, planning)
    n_cleared = clear_multipliers(models) if policy == 'A' else 0
    state = {
        'models': models,
        'consensus_vars': deepcopy(source_state['consensus_vars']),
        'dual_vars': deepcopy(source_state['dual_vars']),
        'candidate_solution': deepcopy(base_candidate),
        'last_recourse': None, 'last_recourse_blocks': None,
        'last_objective_component_blocks': None,
        'last_slack_component_blocks': None,
        'last_tso_voltage_slack_state': None,
        'consecutive_converged_cycles': 0,
        'admm_diagnostics': [], 'solver_recovery_diagnostics': [],
        'initialization_failed': False,
    }
    return state, {'hat_repair': repaired, 'transferred_residuals': residuals,
                   'multiplier_policy': policy, 'n_multipliers_cleared': n_cleared}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='s:9:-0.05',
                        help='kind:node:rel, e.g. s:9:-0.05')
    parser.add_argument('--policies', default='A,B')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--year', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    kind, node, rel = args.source.split(':')
    node, rel = int(node), float(rel)
    tag = f'{kind}{node}_{rel:+.3f}'.replace('.', 'p')
    out_path = os.path.join(OUT_DIR, f'p54d4_{tag}.json')

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        base_candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    year = args.year if args.year is not None else list(planning.years)[0]

    report = {'stage': 'P5.4-D4', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'source': {'kind': kind, 'node': node, 'rel': rel, 'year': year},
              'Q_base_cold_reference': Q_BASE_COLD,
              'method': ('production warm-start lift; H1 hats recomputed for the base '
                         'rating in the transfer only; no model equation changed')}

    # ---- D4.1: reproduce the source branch ----
    source_candidate = deepcopy(base_candidate)
    source_candidate['investment'][node][year][kind] *= (1.0 + rel)
    srp._rebuild_candidate_total_capacities(planning, source_candidate)
    print(f'[D4.1] solving source {kind}|node{node}|{year}|{rel:+.1%} (cold) ...', flush=True)
    source = run_candidate(planning, source_candidate, keep_state=True)
    print(f"    converged={source['converged']} cycles={source['n_cycles']} "
          f"Q_source={source['recourse']} rt={source['runtime_s']:.0f}s", flush=True)
    report['D4_1_source'] = {k: v for k, v in source.items()
                             if not k.startswith('_') and k != 'sensitivities'}
    with open(out_path, 'w') as h:
        json.dump(report, h, indent=1, default=str)

    if not source['converged']:
        print('    source did not converge -- stopping')
        return

    # ---- D4.2-D4.4: lift to the exact base candidate ----
    lifts = {}
    for policy in args.policies.split(','):
        for rep in range(args.repeats):
            key = f'policy_{policy}_rep{rep + 1}'
            print(f'\n[D4.2/D4.4] lift to base, {key} ...', flush=True)
            state, info = lift_state(planning, source['_state'], base_candidate, policy)
            res = info['transferred_residuals']
            print(f"    hats repaired: {info['hat_repair']['n_entries']} entries, "
                  f"max_hat={info['hat_repair']['max_hat']:.4f}, "
                  f"clipped={info['hat_repair']['n_clipped']}")
            print(f"    transferred-point max residual = {res['max_violation']:.4e} "
                  f"(active_sum {res['active_sum']:.2e}, capability {res['capability']:.2e}, "
                  f"comp {res['complementarity']:.2e}, link {res['link_ch']:.2e}, "
                  f"soc_up {res['soc_upper']:.2e})")
            run = run_candidate(planning, base_candidate, initial_state=state,
                                keep_state=False)
            print(f"    -> converged={run['converged']} cycles={run['n_cycles']} "
                  f"Q_base={run['recourse']} rt={run['runtime_s']:.0f}s")
            if run['recourse'] is not None:
                print(f"       delta vs cold = {Q_BASE_COLD - run['recourse']:+.2f}")
            lifts[key] = {'transfer': info,
                          'run': {k: v for k, v in run.items()
                                  if not k.startswith('_') and k != 'sensitivities'},
                          'sensitivities': run.get('sensitivities'),
                          'delta_vs_cold': (Q_BASE_COLD - run['recourse'])
                          if run['recourse'] is not None else None}
            report['D4_lifts'] = lifts
            with open(out_path, 'w') as h:
                json.dump(report, h, indent=1, default=str)

    best = min((v['run']['recourse'] for v in lifts.values()
                if v['run'].get('converged') and v['run'].get('recourse') is not None),
               default=None)
    report['D4_5_summary'] = {
        'Q_base_cold': Q_BASE_COLD,
        'Q_base_best_observed_from_this_source': best,
        'delta_Q': (Q_BASE_COLD - best) if best is not None else None,
        'relative_improvement': ((Q_BASE_COLD - best) / abs(Q_BASE_COLD))
        if best is not None else None,
        'source_recourse': source['recourse'],
    }
    with open(out_path, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f"\n[D4.5] {json.dumps(report['D4_5_summary'], indent=1, default=str)}")
    print(f'[D4] report -> {out_path}')


if __name__ == '__main__':
    main()
