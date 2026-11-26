"""
Stage P5.4-D2.5b/D2.9 -- branch-controlled finite differences.

The cold-start finite differences in `p54d2_sensitivity_root_cause.py` are
invalid as derivative estimates: the perturbed objectives cluster into a few
discrete values whose separation (~5e-3) is INDEPENDENT of the step size, so
`dQ` measures which local optimum the solver happened to land in, not the
capacity effect. At the smallest tested step the capacity effect predicted by
the envelope derivative is ~2e-6, three orders of magnitude below that gap.

This script does two things:

  1. quantifies the basin structure -- solves the SAME unperturbed problem from
     the production cold start repeatedly and clusters the resulting objectives,
     establishing the basin gap independently of any perturbation;

  2. re-runs the finite differences on a SINGLE branch, by seeding every
     perturbed solve with the base solution's variable values.

Only the starting point changes. Solver options, MA97/exact-Hessian policy,
tolerances and the recovery path are identical to production -- `from_warm_start`
is deliberately left False so the perturbed solves use exactly the same IPOPT
configuration as the base solve.

    python p54d2_branch_controlled_fd.py [--cases dso:9:2030:Winter]
"""

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p54d2_sensitivity_root_cause import (active_set_signature,  # noqa: E402
                                          build_case, objective_of,
                                          sensitivity_decomposition, solve_at)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2')

FD_STEPS = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def snapshot_primal(model):
    """All variable values, keyed by fully qualified name."""
    out = {}
    for var in model.component_objects(pe.Var, active=None):
        for index in var:
            data = var[index]
            try:
                out[data.name] = None if data.value is None else float(data.value)
            except Exception:
                pass
    return out


def apply_primal(model, snapshot, skip_prefixes=()):
    """Seed a model with a previous solution. Fixed variables are left alone so
    the capacity lifecycle is not disturbed."""
    applied = missing = 0
    for var in model.component_objects(pe.Var, active=None):
        for index in var:
            data = var[index]
            if data.fixed:
                continue
            name = data.name
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            if name in snapshot and snapshot[name] is not None:
                value = snapshot[name]
                lo, hi = data.bounds
                if lo is not None:
                    value = max(value, lo)
                if hi is not None:
                    value = min(value, hi)
                data.set_value(value)
                applied += 1
            else:
                missing += 1
    return {'applied': applied, 'missing': missing}


def solve_seeded(network, params, e, s_pu, e_pu, snapshot):
    """Same production build/configure/solve, but seeded from `snapshot`.

    The capacity-dependent variables are re-seeded too, then clipped into their
    new bounds by apply_primal, which is what keeps the perturbed solve on the
    base branch.
    """
    console = io.StringIO()
    with redirect_stdout(console):
        m = network.build_model(params)
        m.shared_es_s_rated_fixed[e].set_value(s_pu)
        m.shared_es_e_rated_fixed[e].set_value(e_pu)
        mch.configure_shared_ess_operational_state(m, e, s_pu, e_pu)
        seed_info = apply_primal(m, snapshot)
        # the rated-capacity variables must start at the PERTURBED capacity
        m.shared_es_s_rated[e].set_value(s_pu)
        m.shared_es_e_rated[e].set_value(e_pu)
        r = network.run_smopf(m, params, print_header=False)
    ok = bool(srp._solver_result_succeeded(r))
    return m, r, ok, seed_info


def cluster(values, tol=1e-5):
    """Group objective values into basins separated by more than `tol`."""
    clusters = []
    for v in sorted(values):
        if clusters and abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [{'n': len(c), 'mean': statistics.fmean(c),
             'min': min(c), 'max': max(c)} for c in clusters]


def basin_probe(case, n_repeats):
    """Solve the UNPERTURBED problem repeatedly from the production cold start.

    Any spread here is basin structure, not a capacity effect, because the
    problem is bit-identical every time.
    """
    objectives, sigs = [], []
    for _ in range(n_repeats):
        m, _r, ok = solve_at(case['network'], case['params'], case['e'],
                             case['s_pu'], case['e_pu'])
        if not ok:
            continue
        objectives.append(objective_of(m))
        sigs.append(json.dumps(active_set_signature(m, case['e']), sort_keys=True))
    return {
        'n_solves': len(objectives), 'objectives': objectives,
        'n_distinct_active_sets': len(set(sigs)),
        'spread': (max(objectives) - min(objectives)) if objectives else None,
        'clusters': cluster(objectives),
        'note': ('identical problem solved repeatedly from the production cold '
                 'start; any spread is basin structure, not a capacity effect'),
    }


def branch_controlled_fd(case, theta, steps, snapshot, decomposition, q0, base_sig):
    network, params, e = case['network'], case['params'], case['e']
    s0, e0 = case['s_pu'], case['e_pu']
    theta0 = s0 if theta == 'S' else e0
    rows = []
    for rel in steps:
        h = rel * theta0
        if theta == 'S':
            up = solve_seeded(network, params, e, s0 + h, e0, snapshot)
            dn = solve_seeded(network, params, e, s0 - h, e0, snapshot)
        else:
            up = solve_seeded(network, params, e, s0, e0 + h, snapshot)
            dn = solve_seeded(network, params, e, s0, e0 - h, snapshot)
        (mu, ru, oku, su), (md, rd, okd, sd) = up, dn
        rec = {'relative_step': rel, 'step': h,
               'up_solved': oku, 'down_solved': okd, 'seed_info': su}
        if not (oku and okd):
            rec['usable'] = False
            rows.append(rec)
            continue
        qu, qd = objective_of(mu), objective_of(md)
        sig_u = json.dumps(active_set_signature(mu, e), sort_keys=True)
        sig_d = json.dumps(active_set_signature(md, e), sort_keys=True)
        base = json.dumps(base_sig, sort_keys=True)
        central = (qu - qd) / (2 * h)
        fix = decomposition['fixing_row_dual']
        cor = decomposition['corrected_total_derivative']
        rec.update({
            'Q_up': qu, 'Q_down': qd, 'Q_base': q0, 'delta_Q': qu - qd,
            'central_difference': central,
            'fixing_dual_prediction': fix,
            'corrected_prediction': cor,
            'rel_err_vs_fixing_dual': abs(central - fix) / max(abs(fix), 1e-30),
            'rel_err_vs_corrected': abs(central - cor) / max(abs(cor), 1e-30),
            'same_ess_active_set_up': sig_u == base,
            'same_ess_active_set_down': sig_d == base,
            'iterations_note': 'solver options identical to base; only the start point differs',
            'termination_up': str(getattr(ru.solver, 'termination_condition', None)),
            'termination_down': str(getattr(rd.solver, 'termination_condition', None)),
            'usable': True,
        })
        rows.append(rec)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='dso:9:2030:Winter')
    parser.add_argument('--steps', default=','.join(str(s) for s in FD_STEPS))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--out', default='p54d2_branch_controlled.json')
    args = parser.parse_args()
    steps = tuple(float(x) for x in args.steps.split(','))

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D2.5b branch-controlled FD', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'method': ('perturbed solves seeded from the base solution; solver '
                         'options, tolerances and recovery policy unchanged')}

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
        print(f"\n[D2.5b] {case['tag']}  S={case['s_pu']:.6e} E={case['e_pu']:.6e}", flush=True)

        print('    basin probe (identical problem, repeated cold starts) ...', flush=True)
        probe = basin_probe(case, args.repeats)
        print(f"      {probe['n_solves']} solves, spread={probe['spread']:.4e}, "
              f"{len(probe['clusters'])} objective clusters, "
              f"{probe['n_distinct_active_sets']} distinct ESS active sets")

        base_model, _r, ok = solve_at(case['network'], case['params'], case['e'],
                                      case['s_pu'], case['e_pu'])
        if not ok:
            results[case['tag']] = {'base_solved': False, 'basin_probe': probe}
            continue
        q0 = objective_of(base_model)
        base_sig = active_set_signature(base_model, case['e'])
        snapshot = snapshot_primal(base_model)

        entry = {'base_solved': True, 'Q_base': q0, 'basin_probe': probe,
                 's_pu': case['s_pu'], 'e_pu': case['e_pu'],
                 'n_snapshot_variables': len(snapshot)}

        for theta in ('S', 'E'):
            dec = sensitivity_decomposition(base_model, case['e'], theta)
            fd = branch_controlled_fd(case, theta, steps, snapshot, dec, q0, base_sig)
            entry[theta] = {'decomposition': dec, 'finite_differences': fd}
            print(f"    {theta}: fixing_dual={dec['fixing_row_dual']:+.6e} "
                  f"bound={dec['bound_contribution']:+.6e} "
                  f"corrected={dec['corrected_total_derivative']:+.6e}")
            print(f"      {'rel':>8} {'dQ':>13} {'central':>14} {'err_dual':>11} {'err_corr':>11}")
            for r in fd:
                if not r.get('usable'):
                    print(f"      {r['relative_step']:>8g} solve failed")
                    continue
                print(f"      {r['relative_step']:>8g} {r['delta_Q']:>13.4e} "
                      f"{r['central_difference']:>14.6e} "
                      f"{r['rel_err_vs_fixing_dual']:>11.3e} "
                      f"{r['rel_err_vs_corrected']:>11.3e}")
        results[case['tag']] = entry

    report['cases'] = results
    out = os.path.join(OUT_DIR, args.out)
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f'\n[D2.5b] report -> {out}')


if __name__ == '__main__':
    main()
