"""
Stage P5.4-D3 -- distributed cut-consistency / local-branch audit.

Tests the quantity the planning algorithm actually uses: the fully aggregated
distributed recourse `Q0` and the production investment-space sensitivity vector
`g0` returned by `run_operational_planning(type='distributed', ...)` -- the same
object `_add_benders_cut` consumes -- not an isolated raw local dual.

The candidate cut

    L(x) = Q0 + g0^T (x - x0)

is CONSTRUCTED but never added to a master problem, and `run_planning_problem()`
is never invoked.

Each perturbed candidate is evaluated from
  A -- production initialization (cold), and
  B -- continuation from the converged neighbouring candidate, via the
       production `initial_state` warm-start path,
and `Q_best_observed = min(Q_A, Q_B)` over converged runs is used as a
falsification benchmark. It is never claimed to be the global optimum.

Results are checkpointed to JSON after every candidate, so a long batch can be
inspected or resumed.

    python p54d3_cut_consistency.py --group base
    python p54d3_cut_consistency.py --group s9
    ...
"""

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D3')

FULL_STEPS = (-0.10, -0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05, 0.10)
COARSE_STEPS = (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)

GROUPS = {
    'base': [],
    's9': [('s', 9, r) for r in FULL_STEPS],
    'e9': [('e', 9, r) for r in FULL_STEPS],
    's5': [('s', 5, r) for r in COARSE_STEPS],
    's7': [('s', 7, r) for r in COARSE_STEPS],
    'e5': [('e', 5, r) for r in COARSE_STEPS],
    'e7': [('e', 7, r) for r in COARSE_STEPS],
}


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def flatten_investment(candidate, nodes, years):
    """x as an ordered dict keyed by (capacity_type, node, year)."""
    out = {}
    for kind in ('s', 'e'):
        for node in nodes:
            for year in years:
                out[(kind, node, year)] = candidate['investment'][node][year][kind]
    return out


def flatten_sensitivities(sens, nodes, years):
    out = {}
    for kind in ('s', 'e'):
        for node in nodes:
            for year in years:
                value = sens[kind][year][node] if sens is not None else None
                out[(kind, node, year)] = value
    return out


def run_candidate(planning, candidate, initial_state=None):
    """One production distributed operational-planning solve."""
    console = io.StringIO()
    started = time.time()
    with redirect_stdout(console):
        convergence, results, models, sensitivities, primal_evolution, state = \
            planning.run_operational_planning(
                type='distributed',
                candidate_solution=deepcopy(candidate),
                print_results=False,
                debug_flag=False,
                initial_state=initial_state,
                return_state=True)
    runtime = time.time() - started
    diags = state.get('admm_diagnostics', [])
    last = diags[-1] if diags else {}
    return {
        'converged': bool(convergence),
        'initialization_failed': bool(state.get('initialization_failed', False)),
        'runtime_s': runtime,
        'n_cycles': len(diags),
        'all_local_solves_ok': all(d.get('local_solves_ok') for d in diags) if diags else None,
        'n_recovery_diagnostics': len(state.get('solver_recovery_diagnostics', [])),
        'recourse': last.get('recourse'),
        'gross_operational_cost': last.get('gross_operational_cost'),
        'terminal_salvage_value': last.get('terminal_salvage_value'),
        'final_primal': {k: last.get(k) for k in ('primal_v', 'primal_pf', 'primal_ess')},
        'final_dual': {k: last.get(k) for k in ('dual_v', 'dual_pf', 'dual_ess')},
        'sensitivities': sensitivities,
        'state': state,
    }


def branch_signature(run, decimals=6):
    """A coarse label for which local solution branch the run settled on."""
    q = run.get('recourse')
    if q is None:
        return None
    return round(q, decimals)


def make_candidate(base, kind, node, year, rel, planning):
    cand = deepcopy(base)
    cand['investment'][node][year][kind] *= (1.0 + rel)
    srp._rebuild_candidate_total_capacities(planning, cand)
    return cand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default='base')
    parser.add_argument('--year', type=int, default=None,
                        help='investment year to perturb; defaults to the first')
    parser.add_argument('--with-continuation', action='store_true',
                        help='also evaluate start B (continuation warm start)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'p54d3_{args.group}.json')

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        base_candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    nodes = list(planning.active_distribution_network_nodes)
    years = list(planning.years)
    year = args.year if args.year is not None else years[0]

    report = {'stage': 'P5.4-D3', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'group': args.group, 'perturbed_year': year,
              'nodes': nodes, 'years': years,
              'coefficient_path': (
                  'g0 = the `sensitivities` returned by '
                  'run_operational_planning(type="distributed"), i.e. the object '
                  '_add_benders_cut consumes: local fixing-row duals scaled by '
                  'objective_scale/baseMVA, weighted by annualization*num_years*'
                  'num_days, summed over TSO+DSOs, mapped available->investment '
                  'capacity, plus the salvage sensitivity'),
              'cut_form': 'L(x) = Q0 + g0^T (x - x0); constructed, never added to a master'}

    # ---------------- D3.1: base run ----------------
    print('[D3.1] base distributed run at the positive-bootstrap candidate ...', flush=True)
    base_run = run_candidate(planning, base_candidate)
    q0 = base_run['recourse']
    g0 = flatten_sensitivities(base_run['sensitivities'], nodes, years)
    x0 = flatten_investment(base_candidate, nodes, years)
    base_state = base_run.pop('state')
    print(f"    converged={base_run['converged']} cycles={base_run['n_cycles']} "
          f"Q0={q0:.6f} runtime={base_run['runtime_s']:.0f}s", flush=True)

    report['D3_1_base'] = {k: v for k, v in base_run.items() if k != 'sensitivities'}
    report['D3_1_base']['Q0'] = q0
    report['D3_2_cut'] = {
        'x0': {f'{k[0]}|node{k[1]}|{k[2]}': v for k, v in x0.items()},
        'g0': {f'{k[0]}|node{k[1]}|{k[2]}': v for k, v in g0.items()},
        'n_coefficients': len(g0),
        'n_none_coefficients': sum(1 for v in g0.values() if v is None),
        'g0_sign_summary': {
            's_negative': sum(1 for k, v in g0.items() if k[0] == 's' and v is not None and v < 0),
            's_positive': sum(1 for k, v in g0.items() if k[0] == 's' and v is not None and v > 0),
            'e_negative': sum(1 for k, v in g0.items() if k[0] == 'e' and v is not None and v < 0),
            'e_positive': sum(1 for k, v in g0.items() if k[0] == 'e' and v is not None and v > 0)},
    }

    # repeatability at the identical candidate -> the empirical part of tol_cut
    print('[D3.6] repeatability probe at the identical candidate ...', flush=True)
    repeat = run_candidate(planning, base_candidate)
    repeat.pop('state', None)
    spread = abs((repeat['recourse'] or 0.0) - (q0 or 0.0))
    report['D3_6_repeatability'] = {
        'Q_repeat': repeat['recourse'], 'Q0': q0,
        'absolute_spread': spread,
        'relative_spread': spread / max(abs(q0), 1e-30),
        'converged': repeat['converged'], 'n_cycles': repeat['n_cycles']}
    print(f"    Q_repeat={repeat['recourse']:.6f} spread={spread:.6e} "
          f"({spread / max(abs(q0), 1e-30):.3e} relative)", flush=True)

    with open(out_path, 'w') as h:
        json.dump(report, h, indent=1, default=str)

    # ---------------- D3.3-D3.7: perturbed candidates ----------------
    candidates = []
    prev_state = base_state
    for kind, node, rel in GROUPS[args.group]:
        cand = make_candidate(base_candidate, kind, node, year, rel, planning)
        label = f'{kind}|node{node}|{year}|{rel:+.3%}'
        print(f'\n[D3.4] {label} ...', flush=True)

        run_a = run_candidate(planning, cand)
        run_a.pop('state', None)
        print(f"    A: converged={run_a['converged']} cycles={run_a['n_cycles']} "
              f"Q={run_a['recourse']} rt={run_a['runtime_s']:.0f}s", flush=True)

        run_b = None
        if args.with_continuation and prev_state is not None:
            try:
                run_b = run_candidate(planning, cand, initial_state=prev_state)
                if run_b.get('converged'):
                    prev_state = run_b.pop('state')
                else:
                    run_b.pop('state', None)
                print(f"    B: converged={run_b['converged']} cycles={run_b['n_cycles']} "
                      f"Q={run_b['recourse']} rt={run_b['runtime_s']:.0f}s", flush=True)
            except Exception as error:
                run_b = {'converged': False, 'error': f'{type(error).__name__}: {error}'}
                print(f'    B: failed -- {run_b["error"]}', flush=True)

        observed = [r['recourse'] for r in (run_a, run_b)
                    if r is not None and r.get('converged') and r.get('recourse') is not None]
        q_best = min(observed) if observed else None

        x = flatten_investment(cand, nodes, years)
        dx = {k: x[k] - x0[k] for k in x}
        predicted_delta = sum(
            (g0[k] or 0.0) * dx[k] for k in dx if g0.get(k) is not None)
        n_missing = sum(1 for k in dx if abs(dx[k]) > 0 and g0.get(k) is None)
        l_x = (q0 + predicted_delta) if q0 is not None else None

        rec = {
            'label': label, 'kind': kind, 'node': node, 'year': year, 'rel': rel,
            'delta_x_nonzero': {f'{k[0]}|node{k[1]}|{k[2]}': v
                                for k, v in dx.items() if abs(v) > 0},
            'n_perturbed_entries_without_coefficient': n_missing,
            'A': {k: v for k, v in run_a.items() if k != 'sensitivities'},
            'B': ({k: v for k, v in run_b.items() if k != 'sensitivities'}
                  if run_b is not None else None),
            'Q_best_observed': q_best,
            'predicted_delta_g0T_dx': predicted_delta,
            'L_x': l_x,
            'cut_gap': (q_best - l_x) if (q_best is not None and l_x is not None) else None,
            'observed_delta_Q': (q_best - q0) if q_best is not None else None,
            'branch_A': branch_signature(run_a),
            'branch_B': branch_signature(run_b) if run_b else None,
            'branch_base': branch_signature({'recourse': q0}),
        }
        if rec['observed_delta_Q'] is not None:
            rec['linearity_abs_error'] = abs(rec['observed_delta_Q'] - predicted_delta)
            rec['linearity_rel_error'] = (
                rec['linearity_abs_error'] / max(abs(rec['observed_delta_Q']), 1e-30))
            rec['same_branch_as_base'] = (rec['branch_A'] == rec['branch_base'])
        candidates.append(rec)

        report['D3_candidates'] = candidates
        with open(out_path, 'w') as h:
            json.dump(report, h, indent=1, default=str)

    report['D3_candidates'] = candidates
    if candidates:
        gaps = [c['cut_gap'] for c in candidates if c['cut_gap'] is not None]
        report['D3_summary'] = {
            'n_candidates': len(candidates),
            'n_converged_A': sum(1 for c in candidates if c['A'].get('converged')),
            'min_cut_gap': min(gaps) if gaps else None,
            'n_negative_cut_gap': sum(1 for g in gaps if g < 0),
            'worst_candidate': min(candidates, key=lambda c: c['cut_gap']
                                   if c['cut_gap'] is not None else float('inf'))['label']
            if gaps else None,
        }
    with open(out_path, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f'\n[D3] report -> {out_path}')
    if candidates:
        print(json.dumps(report['D3_summary'], indent=1, default=str))


if __name__ == '__main__':
    main()
