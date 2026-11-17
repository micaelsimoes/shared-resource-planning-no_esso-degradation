"""
Stage P5.4-R2 -- canonical D3 distributed cut-consistency audit.

Everything is recomputed from scratch in the canonical environment. Nothing is
loaded from disk: models, ADMM consensus/dual states, warm starts and
sensitivities are all built in-process by the production code during this run,
so no `srp_env` artefact can leak in. Output goes to P54R_D3/ so the earlier
noncanonical evidence under P54D3/ is preserved untouched.

Candidate set is identical to the completed prior D3 population, for a
like-for-like comparison.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p54r_d3.py --group s9
"""

import argparse
import io
import json
import os
import statistics
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from p54d3_cut_consistency import (branch_signature, flatten_investment,  # noqa: E402
                                   flatten_sensitivities, make_candidate,
                                   run_candidate)
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54R_D3')

# exactly the candidates that completed in the prior (noncanonical) D3
GROUPS = {
    's9': [('s', 9, -0.05), ('s', 9, -0.10)],
    'e9': [('e', 9, -0.10), ('e', 9, -0.05), ('e', 9, -0.02), ('e', 9, -0.01)],
    's5': [('s', 5, -0.10), ('s', 5, -0.05)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=sorted(GROUPS))
    parser.add_argument('--with-continuation', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'p54r_d3_{args.group}.json')

    try:
        provenance, planning = gate(f'P5.4-R2 canonical D3 [{args.group}]', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[R2] ABORTED\n{error}')
        sys.exit(1)

    console = io.StringIO()
    with redirect_stdout(console):
        base_candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    nodes = list(planning.active_distribution_network_nodes)
    years = list(planning.years)
    year = years[0]

    report = {'stage': 'P5.4-R2', 'group': args.group, 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'nodes': nodes, 'years': years, 'perturbed_year': year,
              'freshness': ('all models, consensus/dual states, warm starts and '
                            'sensitivities are constructed in-process during this run; '
                            'no srp_env artefact is read from disk')}

    # ---- D3.1 base ----
    print('\n[R2/D3.1] canonical base distributed run ...', flush=True)
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

    # ---- repeatability, the empirical half of tol_cut ----
    print('[R2/D3.6] canonical repeatability probe ...', flush=True)
    repeat = run_candidate(planning, base_candidate)
    repeat.pop('state', None)
    spread = abs((repeat['recourse'] or 0.0) - (q0 or 0.0))
    admm_drift = None
    diags = base_run.get('admm_diagnostics')
    report['D3_6_repeatability'] = {
        'Q_repeat': repeat['recourse'], 'Q0': q0,
        'absolute_spread': spread,
        'relative_spread': spread / max(abs(q0), 1e-30),
        'converged': repeat['converged'], 'n_cycles': repeat['n_cycles']}
    print(f"    Q_repeat={repeat['recourse']:.6f} spread={spread:.6e}", flush=True)

    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # ---- D3.3-D3.7 candidates ----
    candidates = []
    prev_state = base_state
    for kind, node, rel in GROUPS[args.group]:
        cand = make_candidate(base_candidate, kind, node, year, rel, planning)
        label = f'{kind}|node{node}|{year}|{rel:+.3%}'
        print(f'\n[R2/D3.4] {label} ...', flush=True)

        run_a = run_candidate(planning, cand)
        run_a.pop('state', None)
        print(f"    A: converged={run_a['converged']} cycles={run_a['n_cycles']} "
              f"Q={run_a['recourse']} rt={run_a['runtime_s']:.0f}s", flush=True)

        run_b = None
        if args.with_continuation and prev_state is not None:
            run_b = run_candidate(planning, cand, initial_state=prev_state)
            if run_b.get('converged'):
                prev_state = run_b.pop('state')
            else:
                run_b.pop('state', None)
            print(f"    B: converged={run_b['converged']} cycles={run_b['n_cycles']} "
                  f"Q={run_b['recourse']} rt={run_b['runtime_s']:.0f}s", flush=True)

        observed = [r['recourse'] for r in (run_a, run_b)
                    if r is not None and r.get('converged') and r.get('recourse') is not None]
        q_best = min(observed) if observed else None

        x = flatten_investment(cand, nodes, years)
        dx = {k: x[k] - x0[k] for k in x}
        predicted = sum((g0[k] or 0.0) * dx[k] for k in dx if g0.get(k) is not None)
        l_x = (q0 + predicted) if q0 is not None else None

        rec = {
            'label': label, 'kind': kind, 'node': node, 'year': year, 'rel': rel,
            'delta_x_nonzero': {f'{k[0]}|node{k[1]}|{k[2]}': v
                                for k, v in dx.items() if abs(v) > 0},
            'A': {k: v for k, v in run_a.items() if k != 'sensitivities'},
            'B': ({k: v for k, v in run_b.items() if k != 'sensitivities'}
                  if run_b is not None else None),
            'Q_best_observed': q_best,
            'predicted_delta_g0T_dx': predicted,
            'L_x': l_x,
            'cut_gap': (q_best - l_x) if (q_best is not None and l_x is not None) else None,
            'observed_delta_Q': (q_best - q0) if q_best is not None else None,
            'branch_A': branch_signature(run_a),
            'branch_B': branch_signature(run_b) if run_b else None,
            'branch_base': branch_signature({'recourse': q0}),
        }
        if rec['observed_delta_Q'] is not None:
            rec['linearity_abs_error'] = abs(rec['observed_delta_Q'] - predicted)
            rec['linearity_rel_error'] = (
                rec['linearity_abs_error'] / max(abs(rec['observed_delta_Q']), 1e-30))
            rec['same_branch_as_base'] = (rec['branch_A'] == rec['branch_base'])
        candidates.append(rec)

        report['D3_candidates'] = candidates
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    print(f'\n[R2] {args.group} -> {out_path}')


if __name__ == '__main__':
    main()
