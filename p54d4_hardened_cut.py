"""
Stage P5.4-D4.6/D4.7/D4.8/D4.9/D4.10 -- branch fingerprint, deterministic
multi-start oracle, and the hardened empirical cut test.

D4.4 established that the exact base candidate attains a recourse ~1.5 % below
its production cold value when initialized from a converged smaller-capacity
state. This script:

  1. reproduces the best known base branch deterministically and keeps its state;
  2. fingerprints it against the cold base branch (D4.6);
  3. treats the recovered base state as an archived branch template and lifts it
     onto each previously probed candidate, giving a HARDENED
     Q_best_observed(x) for each (D4.7);
  4. rebuilds the cut from the recovered branch and re-runs the D3 falsification
     test (D4.9).

`L_best(x) = Q_base_best + g_best^T (x - x0)`.

The cut is constructed, never added to a master, and `run_planning_problem()` is
never invoked.

    python p54d4_hardened_cut.py
"""

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p54d4_branch_recovery import (Q_BASE_COLD, lift_state,  # noqa: E402
                                   run_candidate)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D4')

# the D3 candidates already probed, plus the D4.10 expansion (positive side)
D3_CANDIDATES = [('s', 9, -0.05), ('s', 9, -0.10), ('s', 5, -0.05),
                 ('s', 5, -0.10), ('e', 9, -0.01), ('e', 9, -0.10)]
EXPANSION = [('s', 9, 0.05), ('s', 9, 0.10), ('s', 5, 0.05),
             ('e', 9, 0.05), ('s', 7, 0.05)]

# best deterministic source found in D4.4
BEST_SOURCE = ('s', 9, -0.10)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def flat(sens, nodes, years):
    out = {}
    for kind in ('s', 'e'):
        for node in nodes:
            for year in years:
                out[(kind, node, year)] = sens[kind][year][node] if sens else None
    return out


def make_candidate(base, kind, node, year, rel, planning):
    cand = deepcopy(base)
    cand['investment'][node][year][kind] *= (1.0 + rel)
    srp._rebuild_candidate_total_capacities(planning, cand)
    return cand


def stringify_keys(obj):
    """Recourse/objective blocks are keyed by tuples; JSON needs string keys."""
    if isinstance(obj, dict):
        return {(str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k):
                stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [stringify_keys(v) for v in obj]
    return obj


def fingerprint(state, run):
    """D4.6: objective decomposition of a converged branch."""
    return {
        'recourse': run.get('recourse'),
        'gross_operational_cost': run.get('gross_operational_cost'),
        'terminal_salvage_value': run.get('terminal_salvage_value'),
        'recourse_blocks': stringify_keys(state.get('last_recourse_blocks')),
        'objective_component_blocks': stringify_keys(
            state.get('last_objective_component_blocks')),
        'slack_component_blocks': stringify_keys(state.get('last_slack_component_blocks')),
        'tso_voltage_slack_state': stringify_keys(state.get('last_tso_voltage_slack_state')),
    }


def diff_blocks(a, b, path=''):
    """Recursively difference two nested numeric block dicts."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            out.extend(diff_blocks(a.get(key), b.get(key), f'{path}/{key}'))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out.append({'path': path, 'cold': a, 'recovered': b, 'delta': b - a})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expand', action='store_true')
    parser.add_argument('--out', default='p54d4_hardened.json')
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, args.out)

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        base = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    nodes = list(planning.active_distribution_network_nodes)
    years = list(planning.years)
    year = years[0]
    x0 = {(k, n, y): base['investment'][n][y][k]
          for k in ('s', 'e') for n in nodes for y in years}

    report = {'stage': 'P5.4-D4.6-D4.10', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'Q_base_cold': Q_BASE_COLD, 'best_source': list(BEST_SOURCE)}

    # ---- cold base, kept for the D4.6 fingerprint ----
    print('[D4.6] cold base run (fingerprint reference) ...', flush=True)
    cold = run_candidate(planning, base, keep_state=True)
    print(f"    Q_cold={cold['recourse']} cycles={cold['n_cycles']}", flush=True)

    # ---- reproduce the best base branch deterministically ----
    kind, node, rel = BEST_SOURCE
    src_cand = make_candidate(base, kind, node, year, rel, planning)
    print(f'[D4.7] source {kind}|node{node}|{rel:+.0%} (cold) ...', flush=True)
    source = run_candidate(planning, src_cand, keep_state=True)
    print(f"    Q_source={source['recourse']} cycles={source['n_cycles']}", flush=True)

    print('[D4.7] lift source -> base (policy A) ...', flush=True)
    state, info = lift_state(planning, source['_state'], base, 'A')
    best = run_candidate(planning, base, initial_state=state, keep_state=True)
    q_best = best['recourse']
    g_best = flat(best['sensitivities'], nodes, years)
    print(f"    Q_base_best={q_best} cycles={best['n_cycles']} "
          f"delta_vs_cold={Q_BASE_COLD - q_best:+.2f}", flush=True)

    report['D4_7_base_branch'] = {
        'transfer': info,
        'run': {k: v for k, v in best.items() if not k.startswith('_') and k != 'sensitivities'},
        'Q_base_best': q_best,
        'delta_vs_cold': Q_BASE_COLD - q_best,
        'relative_improvement': (Q_BASE_COLD - q_best) / abs(Q_BASE_COLD),
        'g_best': {f'{k[0]}|node{k[1]}|{k[2]}': v for k, v in g_best.items()},
        'g_cold': {f'{k[0]}|node{k[1]}|{k[2]}': v
                   for k, v in flat(cold['sensitivities'], nodes, years).items()},
    }

    # ---- D4.6 fingerprints ----
    fp_cold = fingerprint(cold['_state'], cold)
    fp_best = fingerprint(best['_state'], best)
    deltas = diff_blocks(stringify_keys(cold['_state'].get('last_recourse_blocks')),
                         stringify_keys(best['_state'].get('last_recourse_blocks')))
    deltas += diff_blocks(
        stringify_keys(cold['_state'].get('last_objective_component_blocks')),
        stringify_keys(best['_state'].get('last_objective_component_blocks')))
    deltas.sort(key=lambda d: -abs(d['delta']))
    report['D4_6_fingerprint'] = {
        'cold': fp_cold, 'recovered': fp_best,
        'gross_cost_delta': (fp_best['gross_operational_cost'] or 0)
        - (fp_cold['gross_operational_cost'] or 0),
        'salvage_delta': (fp_best['terminal_salvage_value'] or 0)
        - (fp_cold['terminal_salvage_value'] or 0),
        'largest_component_deltas': deltas[:40],
    }
    print(f"[D4.6] gross cost delta = {report['D4_6_fingerprint']['gross_cost_delta']:+.2f}; "
          f"salvage delta = {report['D4_6_fingerprint']['salvage_delta']:+.4f}")
    for d in deltas[:10]:
        print(f"    {d['path']:60s} cold={d['cold']:.4e} best={d['recovered']:.4e} "
              f"delta={d['delta']:+.4e}")

    with open(out_path, 'w') as h:
        json.dump(report, h, indent=1, default=str)

    # ---- D4.9: hardened cut vs hardened candidates ----
    todo = list(D3_CANDIDATES) + (EXPANSION if args.expand else [])
    rows = []
    for kind, node, rel in todo:
        cand = make_candidate(base, kind, node, year, rel, planning)
        label = f'{kind}|node{node}|{year}|{rel:+.2%}'
        print(f'\n[D4.9] {label}: lifting the recovered base branch ...', flush=True)
        cstate, cinfo = lift_state(planning, best['_state'], cand, 'A')
        run = run_candidate(planning, cand, initial_state=cstate, keep_state=False)
        x = {(k, n, y): cand['investment'][n][y][k]
             for k in ('s', 'e') for n in nodes for y in years}
        dx = {k: x[k] - x0[k] for k in x}
        pred = sum((g_best[k] or 0.0) * dx[k] for k in dx)
        l_best = q_best + pred
        q_obs = run['recourse'] if run['converged'] else None
        gap = (q_obs - l_best) if q_obs is not None else None
        print(f"    Q_hardened={q_obs} cycles={run['n_cycles']} "
              f"L_best={l_best:.2f} cut_gap_best={gap if gap is None else f'{gap:+.2f}'}")
        rows.append({
            'label': label, 'kind': kind, 'node': node, 'rel': rel,
            'transfer_max_residual': cinfo['transferred_residuals']['max_violation'],
            'run': {k: v for k, v in run.items()
                    if not k.startswith('_') and k != 'sensitivities'},
            'Q_hardened': q_obs, 'predicted_delta': pred, 'L_best': l_best,
            'cut_gap_best': gap,
        })
        report['D4_9_hardened_cut'] = rows
        with open(out_path, 'w') as h:
            json.dump(report, h, indent=1, default=str)

    gaps = [r['cut_gap_best'] for r in rows if r['cut_gap_best'] is not None]
    report['D4_9_summary'] = {
        'n_candidates': len(rows), 'n_evaluated': len(gaps),
        'min_cut_gap_best': min(gaps) if gaps else None,
        'n_negative': sum(1 for g in gaps if g < 0),
        'max_abs_predicted': max((abs(r['predicted_delta']) for r in rows), default=None),
    }
    with open(out_path, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f"\n[D4.9] {json.dumps(report['D4_9_summary'], indent=1, default=str)}")
    print(f'[D4] report -> {out_path}')


if __name__ == '__main__':
    main()
