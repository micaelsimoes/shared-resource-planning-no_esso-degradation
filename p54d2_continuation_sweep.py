"""
Stage P5.4-D2.5c/D2.9 -- matched-branch derivative validation by continuation.

Two corrections to the earlier D2 experiments:

1. The cold-start base solve and the seeded perturbed solves land on DIFFERENT
   local optima (base Q ~ -0.801702, seeded Q ~ -0.799075). Comparing a dual
   extracted on one branch against finite differences taken on another is
   apples-to-oranges. Here the reference dual is extracted from a solve reached
   by the SAME seeding procedure as the finite differences, so both belong to
   the same branch.

2. A pair of independent perturbed solves can still straddle a branch boundary.
   A continuation sweep -- stepping capacity monotonically and seeding each
   solve from the PREVIOUS solution -- tracks one branch explicitly, and makes a
   branch change visible as a jump in the recorded objective.

At every sweep point the fixing-row dual, the bound contribution and the
corrected envelope derivative are recorded, so the derivative predictions can be
compared against a difference taken along the same branch.

Solver options, tolerances, MA97/exact-Hessian policy and the recovery path are
unchanged throughout; only the starting point varies.

    python p54d2_continuation_sweep.py [--cases dso:9:2030:Winter]
"""

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p54d2_branch_controlled_fd import snapshot_primal, solve_seeded  # noqa: E402
from p54d2_sensitivity_root_cause import (active_set_signature, build_case,  # noqa: E402
                                          objective_of,
                                          sensitivity_decomposition, solve_at)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def sweep(case, theta, half_width, n_steps):
    """Continuation sweep across theta0*(1-half_width) .. theta0*(1+half_width).

    Each solve is seeded from the previous solve's solution, so the sequence
    follows one branch until it demonstrably jumps.
    """
    network, params, e = case['network'], case['params'], case['e']
    s0, e0 = case['s_pu'], case['e_pu']
    theta0 = s0 if theta == 'S' else e0

    # start from the cold-start solution at the lower end, then walk upward
    lo = theta0 * (1.0 - half_width)
    if theta == 'S':
        m0, _r, ok = solve_at(network, params, e, lo, e0)
    else:
        m0, _r, ok = solve_at(network, params, e, s0, lo)
    if not ok:
        return {'ok': False, 'reason': 'cold start at lower end failed'}
    snap = snapshot_primal(m0)

    values = [theta0 * (1.0 - half_width + 2.0 * half_width * i / (n_steps - 1))
              for i in range(n_steps)]
    points = []
    for value in values:
        if theta == 'S':
            m, r, ok, _info = solve_seeded(network, params, e, value, e0, snap)
        else:
            m, r, ok, _info = solve_seeded(network, params, e, s0, value, snap)
        if not ok:
            points.append({'theta': value, 'solved': False})
            continue
        snap = snapshot_primal(m)          # continuation: carry the solution forward
        dec = sensitivity_decomposition(m, e, theta)
        points.append({
            'theta': value, 'solved': True,
            'Q': objective_of(m),
            'fixing_row_dual': dec['fixing_row_dual'],
            'bound_contribution': dec['bound_contribution'],
            'corrected_total_derivative': dec['corrected_total_derivative'],
            'ess_active_set': json.dumps(active_set_signature(m, e), sort_keys=True),
            'termination': str(getattr(r.solver, 'termination_condition', None)),
        })

    # differentiate along the sweep and compare with the predictions at the
    # interior points; flag any jump as a branch change
    solved = [p for p in points if p.get('solved')]
    comparisons = []
    for i in range(1, len(solved) - 1):
        a, b, c = solved[i - 1], solved[i], solved[i + 1]
        dtheta = c['theta'] - a['theta']
        if dtheta <= 0:
            continue
        central = (c['Q'] - a['Q']) / dtheta
        fix = b['fixing_row_dual']
        cor = b['corrected_total_derivative']
        comparisons.append({
            'theta': b['theta'],
            'central_difference': central,
            'fixing_row_dual': fix,
            'corrected_total_derivative': cor,
            'rel_err_vs_fixing_dual': abs(central - fix) / max(abs(fix), 1e-30),
            'rel_err_vs_corrected': abs(central - cor) / max(abs(cor), 1e-30),
            'same_active_set_as_neighbours': (a['ess_active_set'] == b['ess_active_set']
                                              == c['ess_active_set']),
        })

    qs = [p['Q'] for p in solved]
    steps = [abs(qs[i + 1] - qs[i]) for i in range(len(qs) - 1)]
    median_step = sorted(steps)[len(steps) // 2] if steps else 0.0
    jumps = [i for i, s in enumerate(steps) if s > max(20.0 * median_step, 1e-6)]

    return {'ok': True, 'theta': theta, 'theta0': theta0,
            'half_width': half_width, 'n_steps': n_steps,
            'points': points, 'comparisons': comparisons,
            'objective_step_median': median_step,
            'n_detected_jumps': len(jumps), 'jump_indices': jumps,
            'monotone_branch': len(jumps) == 0,
            'n_distinct_active_sets': len({p['ess_active_set'] for p in solved})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='dso:9:2030:Winter')
    parser.add_argument('--half-width', type=float, default=0.05)
    parser.add_argument('--n-steps', type=int, default=21)
    parser.add_argument('--out', default='p54d2_continuation.json')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D2.5c continuation sweep', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'half_width': args.half_width, 'n_steps': args.n_steps}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    results = {}
    for spec in args.cases.split(','):
        agent, ident, year, day = spec.split(':')
        case = build_case(planning, candidate, agent, ident, year, day)
        print(f"\n[D2.5c] {case['tag']}", flush=True)
        entry = {}
        for theta in ('S', 'E'):
            print(f'    sweeping {theta} ...', flush=True)
            res = sweep(case, theta, args.half_width, args.n_steps)
            entry[theta] = res
            if not res.get('ok'):
                print(f"      FAILED: {res.get('reason')}")
                continue
            print(f"      {len([p for p in res['points'] if p.get('solved')])}"
                  f"/{args.n_steps} solved, {res['n_detected_jumps']} jumps, "
                  f"{res['n_distinct_active_sets']} distinct ESS active sets, "
                  f"monotone_branch={res['monotone_branch']}")
            print(f"      {'theta/theta0':>13} {'Q':>22} {'central':>13} "
                  f"{'fix_dual':>13} {'corrected':>13} {'err_cor':>10}")
            for cmp_ in res['comparisons']:
                print(f"      {cmp_['theta']/res['theta0']:>13.5f} "
                      f"{'':>22} {cmp_['central_difference']:>13.5e} "
                      f"{cmp_['fixing_row_dual']:>13.5e} "
                      f"{cmp_['corrected_total_derivative']:>13.5e} "
                      f"{cmp_['rel_err_vs_corrected']:>10.3e}")
        results[case['tag']] = entry

    report['cases'] = results
    out = os.path.join(OUT_DIR, args.out)
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f'\n[D2.5c] report -> {out}')


if __name__ == '__main__':
    main()
