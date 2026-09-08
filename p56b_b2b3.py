"""
Stage P5.6-B2 / B3 -- lock the interface-anchor policy and the recurring start
policy, and re-run the purity test against T_STAR.

B1 found that the template chain does not converge, so T_STAR cannot be frozen
by stabilization.  It is therefore frozen by RULE instead, and the rule is the
one P5.6-A already used and can be reproduced exactly:

    T_STAR  :=  T0  =  the cold solution of the canonical base candidate

That is a DECLARED template, not a converged fixed point, and the report says so.
It is sufficient for B2 and B3, which compare policies at a fixed template; it is
not a claim that the objective level it produces is the best available.

B2 measures the anchor convention with the ADMM solved ONCE per candidate and the
polish run twice from clones, so the comparison isolates the polish convention.
B3 then adds cold evaluations on a small strategic subset only.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56b_b2b3.py
"""

import hashlib
import io
import json
import os
import statistics
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as B  # noqa: E402
import p56b_policy as P  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = P.OUT_DIR
PURITY_TOLERANCE = 1.0
MAX_COLD = 5


def spearman(a, b):
    """Rank correlation without pulling in SciPy."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den = (sum((x - mean_a) ** 2 for x in ra)
           * sum((y - mean_b) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def worst_reversal(labels, a, b):
    """Largest pairwise ordering reversal between two objective vectors."""
    worst = None
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                size = min(abs(a[i] - a[j]), abs(b[i] - b[j]))
                if worst is None or size > worst['min_gap_reversed']:
                    worst = {'pair': [labels[i], labels[j]],
                             'midpoint_gap': a[i] - a[j],
                             'dso_gap': b[i] - b[j],
                             'min_gap_reversed': size}
    return worst


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-B2/B3 policy', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[B2] ABORTED\n{error}')
        sys.exit(1)

    report = {'stage': 'P5.6-B2/B3', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    print('[B2] building T_STAR := T0 (cold solution of the canonical base) ...',
          flush=True)
    started = time.time()
    t_star = O.build_fixed_template(verbose=False)
    build_s = time.time() - started
    t_star_meta = {
        'definition': 'T0 = cold solution of the canonical base candidate',
        'frozen_by': 'rule, not stabilization (B1 did not converge)',
        'config_hash': O._config_hash(planning_gate),
        'canonical_checksum': O.load_baseline()['checksum'],
        'oracle_version': O.ORACLE_VERSION,
        'build_runtime_s': build_s,
    }
    t_star_meta['id'] = hashlib.sha256(
        json.dumps(t_star_meta, sort_keys=True).encode()).hexdigest()[:32]
    report['T_STAR'] = t_star_meta
    print(f"[B2] T_STAR id {t_star_meta['id']} built in {build_s:.1f}s\n",
          flush=True)

    population = B.population(planning_gate)

    # ------------------------------------------------------- purity vs T_STAR
    print('[B2] re-running the A3 purity test against T_STAR ...', flush=True)
    purity = []
    for tag, x in (('A1', population[0][1]),
                   ('B', population[-2][1]),
                   ('A2', population[0][1])):
        r = P.evaluate(x, template_state=t_star,
                       anchor_policy=P.POLICY_MIDPOINT_THEN_DSO,
                       eval_id=f'b2_purity_{tag}')
        purity.append({'tag': tag, 'status': r['status'],
                       'total_objective': r.get('total_objective'),
                       'wall_clock_s': r['wall_clock_s']})
        print(f"      {tag}: {r['status']} Q={r.get('total_objective')}",
              flush=True)
    valid = [p for p in purity if p['tag'].startswith('A')
             and p['status'] == O.STATUS_VALID]
    delta = (abs(valid[1]['total_objective'] - valid[0]['total_objective'])
             if len(valid) == 2 else None)
    report['purity_vs_T_STAR'] = {
        'runs': purity, 'declared_tolerance': PURITY_TOLERANCE,
        'objective_delta': delta,
        'bit_identical': delta == 0.0 if delta is not None else None,
        'pass': delta is not None and delta <= PURITY_TOLERANCE}
    print(f"      delta = {delta}   PASS = {report['purity_vs_T_STAR']['pass']}\n",
          flush=True)

    # -------------------------------------------------------------------- B2
    print('[B2] anchor comparison: one ADMM per candidate, two polishes\n',
          flush=True)
    rows = []
    for label, x in population:
        entry = {'label': label, 'anchors': {}}
        planning = O.fresh_planning(f'b2_{label[:18].replace("|", "_")}')
        candidate = O.vector_to_candidate(planning, x)
        feasible, reason = O.check_master_feasibility(planning, candidate)
        entry['master_feasible'] = feasible
        if not feasible:
            entry['status'] = O.STATUS_INVALID_INVESTMENT
            entry['reason'] = reason
            rows.append(entry)
            print(f'[B2] {label:32s} INVALID_INVESTMENT (never solved)', flush=True)
            continue
        models, state, admm = P.run_operational(planning, candidate, t_star)
        entry['admm'] = admm
        if admm.get('status'):
            entry['status'] = admm['status']
            rows.append(entry)
            print(f"[B2] {label:32s} {admm['status']}", flush=True)
            continue
        for anchor in (P.ANCHOR_MIDPOINT, P.ANCHOR_DSO):
            clone = srp._clone_operational_models(models)
            out = P.polish_and_audit(planning, clone, state, candidate, anchor)
            entry['anchors'][anchor] = {
                'status': out['status'],
                'total_objective': out.get('total_objective'),
                'polish_runtime_s': out.get('polish_runtime_s'),
                'failed_blocks': out.get('failed_blocks'),
                'max_coordinated_residual':
                    (out.get('coordination_residuals') or {}).get('max_coordinated'),
                'esso_max_violation':
                    (out.get('esso_audit') or {}).get('max_violation'),
                'network_max_violation':
                    (out.get('network_audit') or {}).get('max_violation'),
                'network_h1_violation':
                    (out.get('network_audit') or {}).get(
                        'max_h1_complementarity_violation'),
            }
        mid = entry['anchors'][P.ANCHOR_MIDPOINT]
        dso = entry['anchors'][P.ANCHOR_DSO]
        if mid['status'] == O.STATUS_VALID and dso['status'] == O.STATUS_VALID:
            entry['anchor_delta'] = (dso['total_objective']
                                     - mid['total_objective'])
        print(f"[B2] {label:32s} mid={mid['status']:16s} "
              f"dso={dso['status']:16s} "
              f"delta={entry.get('anchor_delta')}", flush=True)
        rows.append(entry)
    report['B2_rows'] = rows

    # ------------------------------------------------------ B2 aggregates
    both = [r for r in rows if r.get('anchor_delta') is not None]
    mid_ok = sum(1 for r in rows if r.get('anchors', {}).get(
        P.ANCHOR_MIDPOINT, {}).get('status') == O.STATUS_VALID)
    dso_ok = sum(1 for r in rows if r.get('anchors', {}).get(
        P.ANCHOR_DSO, {}).get('status') == O.STATUS_VALID)
    n_operational = sum(1 for r in rows if r.get('anchors'))
    deltas = [r['anchor_delta'] for r in both]
    labels = [r['label'] for r in both]
    mid_q = [r['anchors'][P.ANCHOR_MIDPOINT]['total_objective'] for r in both]
    dso_q = [r['anchors'][P.ANCHOR_DSO]['total_objective'] for r in both]
    aggregate = {
        'n_candidates': len(rows),
        'n_operationally_evaluated': n_operational,
        'midpoint_success': mid_ok, 'dso_success': dso_ok,
        'midpoint_success_rate': mid_ok / n_operational if n_operational else None,
        'dso_success_rate': dso_ok / n_operational if n_operational else None,
        'n_both_valid': len(both),
    }
    if deltas:
        aggregate.update({
            'anchor_delta_mean': statistics.fmean(deltas),
            'anchor_delta_max_abs': max(abs(d) for d in deltas),
            'anchor_delta_std': statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            'anchor_delta_min': min(deltas), 'anchor_delta_max': max(deltas),
            'spearman_rank_correlation': spearman(mid_q, dso_q),
            'ranking_changes': [labels[i] for i in range(len(labels))
                                if sorted(range(len(mid_q)),
                                          key=lambda k: mid_q[k]).index(i)
                                != sorted(range(len(dso_q)),
                                          key=lambda k: dso_q[k]).index(i)],
            'worst_pairwise_reversal': worst_reversal(labels, mid_q, dso_q),
            'investment_objective_spread_midpoint': max(mid_q) - min(mid_q),
            'investment_objective_spread_dso': max(dso_q) - min(dso_q),
        })
    report['B2_aggregate'] = aggregate

    print('\n[B2] aggregate')
    for key, value in aggregate.items():
        if key in ('ranking_changes', 'worst_pairwise_reversal'):
            continue
        print(f'      {key:38s} {value}')
    print(f"      ranking_changes                        "
          f"{aggregate.get('ranking_changes')}")
    print(f"      worst_pairwise_reversal                "
          f"{aggregate.get('worst_pairwise_reversal')}")

    # -------------------------------------------------------------------- B3
    print('\n[B3] cold evaluations on a small strategic subset\n', flush=True)
    valid_rows = [r for r in rows
                  if r.get('anchors', {}).get(P.ANCHOR_MIDPOINT, {}).get('status')
                  == O.STATUS_VALID]
    by_q = sorted(valid_rows,
                  key=lambda r: r['anchors'][P.ANCHOR_MIDPOINT]['total_objective'])
    subset, seen = [], set()
    def add(label):
        if label and label not in seen:
            seen.add(label)
            subset.append(label)
    add('base')
    if by_q:
        add(by_q[0]['label'])
        add(by_q[-1]['label'])
    add('se|ALL|x19 (budget boundary)')
    fallback = [r['label'] for r in rows
                if r.get('anchors', {}).get(P.ANCHOR_MIDPOINT, {}).get('status')
                not in (O.STATUS_VALID, None)]
    if fallback:
        add(fallback[0])
    subset = subset[:MAX_COLD]
    report['B3_subset'] = subset
    print(f'[B3] subset ({len(subset)}): {subset}\n', flush=True)

    lookup = dict(population)
    b3_rows = []
    for label in subset:
        x = lookup[label]
        r = P.evaluate(x, template_state=None,
                       anchor_policy=P.POLICY_MIDPOINT_THEN_DSO,
                       eval_id=f'b3_cold_{label[:16].replace("|", "_")}')
        row = {'label': label, 'cold': P.certificate_summary(r)}
        t_row = next((q for q in rows if q['label'] == label), None)
        t_valid = (t_row or {}).get('anchors', {}).get(
            P.ANCHOR_MIDPOINT, {}) if t_row else {}
        row['t_star_total_objective'] = t_valid.get('total_objective')
        if (r['status'] == O.STATUS_VALID
                and row['t_star_total_objective'] is not None):
            row['cold_minus_tstar'] = (r['total_objective']
                                       - row['t_star_total_objective'])
        b3_rows.append(row)
        print(f"[B3] {label:32s} cold={r['status']:16s} "
              f"Q={r.get('total_objective')} "
              f"cold-T*={row.get('cold_minus_tstar')}", flush=True)
    report['B3_rows'] = b3_rows
    diffs = [r['cold_minus_tstar'] for r in b3_rows
             if r.get('cold_minus_tstar') is not None]
    report['B3_aggregate'] = {
        'n': len(diffs),
        'cold_minus_tstar_min': min(diffs) if diffs else None,
        'cold_minus_tstar_max': max(diffs) if diffs else None,
        'cold_ever_better': any(d < 0 for d in diffs) if diffs else None,
        'worst_cold_advantage': min(diffs) if diffs else None,
    }
    print(f"\n[B3] aggregate {report['B3_aggregate']}")

    out = os.path.join(OUT_DIR, 'p56b_b2b3_policy.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[B2/B3] report -> {out}')


if __name__ == '__main__':
    main()
