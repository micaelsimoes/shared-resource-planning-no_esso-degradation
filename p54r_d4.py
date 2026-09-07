"""
Stage P5.4-R3 -- canonical D4 branch recovery and oracle hardening.

Triggered because canonical R2 found decisive cut violations and a clearly lower
feasible branch (835 829 512 against a base of 838 496 831).

Every branch state is regenerated canonically in this process. No srp_env
archived state, model, warm start or sensitivity is read. Output goes to
P54R_D4/ so the noncanonical P54D4/ evidence is preserved.

    /opt/anaconda3/envs/opf_env_py311/bin/python p54r_d4.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from p54d4_branch_recovery import lift_state, run_candidate  # noqa: E402
from p54d4_hardened_cut import diff_blocks, fingerprint, stringify_keys  # noqa: E402
from p54d3_cut_consistency import flatten_investment, flatten_sensitivities, make_candidate  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54R_D4')

# deterministic cold source: the candidate whose canonical cold solve sat lowest
BEST_SOURCE = ('s', 9, -0.10)
# the same 8 candidates as canonical R2
CANDIDATES = [('s', 9, -0.05), ('s', 9, -0.10), ('s', 5, -0.05), ('s', 5, -0.10),
              ('e', 9, -0.01), ('e', 9, -0.02), ('e', 9, -0.05), ('e', 9, -0.10)]
TOL_CUT = 716410.898236          # canonical, from p54r_d3_analysis.json


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'p54r_d4_report.json')
    try:
        provenance, planning = gate('P5.4-R3 canonical D4', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[R3] ABORTED\n{error}')
        sys.exit(1)

    console = io.StringIO()
    with redirect_stdout(console):
        base = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    nodes = list(planning.active_distribution_network_nodes)
    years = list(planning.years)
    year = years[0]
    x0 = flatten_investment(base, nodes, years)

    report = {'stage': 'P5.4-R3', 'provenance': provenance, 'tol_cut': TOL_CUT,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'freshness': 'all branch states regenerated in-process; no srp_env artefact reused'}

    # ---- cold base (fingerprint reference) ----
    print('\n[R3/D4.6] canonical cold base ...', flush=True)
    cold = run_candidate(planning, base, keep_state=True)
    q_cold = cold['recourse']
    print(f"    Q_cold={q_cold:.6f} cycles={cold['n_cycles']}", flush=True)

    # ---- source ----
    kind, node, rel = BEST_SOURCE
    src_cand = make_candidate(base, kind, node, year, rel, planning)
    print(f'[R3/D4.1] canonical source {kind}|node{node}|{rel:+.0%} (cold) ...', flush=True)
    source = run_candidate(planning, src_cand, keep_state=True)
    print(f"    Q_source={source['recourse']:.6f} cycles={source['n_cycles']}", flush=True)

    # ---- D4.3 multiplier policies + D4.4/D4.5 ----
    lifts = {}
    for policy in ('A', 'B'):
        print(f'[R3/D4.4] lift source -> base, policy {policy} ...', flush=True)
        state, info = lift_state(planning, source['_state'], base, policy)
        run = run_candidate(planning, base, initial_state=state, keep_state=(policy == 'A'))
        lifts[policy] = {
            'transfer': info,
            'run': {k: v for k, v in run.items()
                    if not k.startswith('_') and k != 'sensitivities'},
            'delta_vs_cold': q_cold - run['recourse'] if run['recourse'] else None,
        }
        print(f"    policy {policy}: Q_base={run['recourse']:.6f} cycles={run['n_cycles']} "
              f"delta_vs_cold={q_cold - run['recourse']:+.2f}", flush=True)
        print(f"      transferred-point max residual = "
              f"{info['transferred_residuals']['max_violation']:.4e}")
        if policy == 'A':
            best = run
    q_best = best['recourse']
    g_best = flatten_sensitivities(best['sensitivities'], nodes, years)

    report['D4_1_source'] = {k: v for k, v in source.items()
                             if not k.startswith('_') and k != 'sensitivities'}
    report['D4_4_lifts'] = lifts
    report['D4_5_summary'] = {
        'Q_base_cold': q_cold,
        'Q_base_best_observed': q_best,
        'delta_Q': q_cold - q_best,
        'relative_improvement': (q_cold - q_best) / abs(q_cold),
        'source_recourse': source['recourse'],
        'policy_A_vs_B_spread': abs(lifts['A']['run']['recourse'] - lifts['B']['run']['recourse']),
    }

    # ---- D4.6 fingerprint ----
    deltas = diff_blocks(stringify_keys(cold['_state'].get('last_recourse_blocks')),
                         stringify_keys(best['_state'].get('last_recourse_blocks')))
    deltas += diff_blocks(
        stringify_keys(cold['_state'].get('last_objective_component_blocks')),
        stringify_keys(best['_state'].get('last_objective_component_blocks')))
    deltas.sort(key=lambda d: -abs(d['delta']))
    report['D4_6_fingerprint'] = {
        'gross_cost_delta': (best.get('gross_operational_cost') or 0)
        - (cold.get('gross_operational_cost') or 0),
        'salvage_delta': (best.get('terminal_salvage_value') or 0)
        - (cold.get('terminal_salvage_value') or 0),
        'largest_component_deltas': deltas[:40],
        'cold': fingerprint(cold['_state'], cold),
        'recovered': fingerprint(best['_state'], best),
    }
    print(f"\n[R3/D4.6] gross cost delta = {report['D4_6_fingerprint']['gross_cost_delta']:+.2f}; "
          f"salvage delta = {report['D4_6_fingerprint']['salvage_delta']:+.4f}")
    for d in deltas[:8]:
        print(f"    {d['path']:58s} cold={d['cold']:.4e} best={d['recovered']:.4e} "
              f"delta={d['delta']:+.4e}")
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # ---- D4.9 hardened cut test ----
    rows = []
    for kind, node, rel in CANDIDATES:
        cand = make_candidate(base, kind, node, year, rel, planning)
        label = f'{kind}|node{node}|{year}|{rel:+.2%}'
        print(f'\n[R3/D4.9] {label} ...', flush=True)
        cstate, cinfo = lift_state(planning, best['_state'], cand, 'A')
        run = run_candidate(planning, cand, initial_state=cstate, keep_state=False)
        x = flatten_investment(cand, nodes, years)
        dx = {k: x[k] - x0[k] for k in x}
        pred = sum((g_best[k] or 0.0) * dx[k] for k in dx if g_best.get(k) is not None)
        l_best = q_best + pred
        q_obs = run['recourse'] if run['converged'] else None
        gap = (q_obs - l_best) if q_obs is not None else None
        rows.append({
            'label': label, 'kind': kind, 'node': node, 'rel': rel,
            'transfer_max_residual': cinfo['transferred_residuals']['max_violation'],
            'run': {k: v for k, v in run.items()
                    if not k.startswith('_') and k != 'sensitivities'},
            'Q_hardened': q_obs, 'predicted_delta': pred, 'L_best': l_best,
            'cut_gap_best': gap,
            'decisive': (gap is not None and gap < -TOL_CUT),
        })
        print(f"    Q={q_obs:.2f} cycles={run['n_cycles']} L_best={l_best:.2f} "
              f"gap={gap:+.2f} decisive={rows[-1]['decisive']}")
        report['D4_9_hardened_cut'] = rows
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    gaps = [r['cut_gap_best'] for r in rows if r['cut_gap_best'] is not None]
    report['D4_9_summary'] = {
        'n_candidates': len(rows), 'n_evaluated': len(gaps),
        'min_cut_gap_best': min(gaps) if gaps else None,
        'n_negative': sum(1 for g in gaps if g < 0),
        'n_decisive': sum(1 for r in rows if r['decisive']),
        'max_abs_predicted': max((abs(r['predicted_delta']) for r in rows), default=None),
        'all_below_hardened_base': all(
            r['Q_hardened'] is not None and r['Q_hardened'] < q_best for r in rows),
        'tol_cut': TOL_CUT,
    }
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f"\n[R3/D4.5] {json.dumps(report['D4_5_summary'], indent=1, default=str)}")
    print(f"[R3/D4.9] {json.dumps(report['D4_9_summary'], indent=1, default=str)}")
    print(f'[R3] report -> {out_path}')


if __name__ == '__main__':
    main()
