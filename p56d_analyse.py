"""
Stage P5.6-D5/D7/D8/D11/D12 -- landscape-stability metrics for the uniformly
refined oracle.

Every delta is taken against the SAME-DEPTH base, as D2 requires:

    Delta_K(x) = H_K(x) - H_K(x0)

Comparing H_K(x) with H_J(x0) for K != J is forbidden and is not done anywhere
here.  The question is not whether objective LEVELS agree across depths -- they
will not, because each refinement lowers everything -- but whether the RELATIVE
investment effects stabilise as the same refinement is applied uniformly.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56d_analyse.py
"""

import json
import os
import statistics
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56d_oracle as D  # noqa: E402

IN_PATH = os.path.join(D.OUT_DIR, 'p56d_depths.json')
OUT_PATH = os.path.join(D.OUT_DIR, 'p56d_metrics.json')
TAU_NUMERICAL = 10.0
P56C_TEMPLATE_LANDSCAPE_UNCERTAINTY = 4.25e5


def rank_of(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = position
    return ranks


def spearman(a, b):
    ra, rb = rank_of(a), rank_of(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def kendall(a, b):
    n, c, d = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return ((c - d) / (c + d)) if (c + d) else None


def reversals(labels, a, b):
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                out.append({'pair': [labels[i], labels[j]],
                            'gap_first': a[i] - a[j], 'gap_second': b[i] - b[j],
                            'min_abs_gap': min(abs(a[i] - a[j]),
                                               abs(b[i] - b[j]))})
    out.sort(key=lambda r: -r['min_abs_gap'])
    return out


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[k]


def main():
    if not os.path.exists(IN_PATH):
        print(f'[D5] ABORTED: {IN_PATH} not found -- run p56d_run.py first.')
        sys.exit(1)
    with open(IN_PATH) as handle:
        data = json.load(handle)

    base_ref = data.get('base_reference') or {}
    runs = data.get('runs', {})
    others = [c for c in data['core_population'] if c != 'base']

    report = {'stage': 'P5.6-D5/D7/D8', 'provenance': data.get('provenance'),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'base_reference': base_ref,
              'tau_numerical': TAU_NUMERICAL,
              'P56C_TEMPLATE_LANDSCAPE_UNCERTAINTY':
                  P56C_TEMPLATE_LANDSCAPE_UNCERTAINTY}

    # -------------------------------------------------- levels and same-depth deltas
    depths = [('H_2', 'H_2'), ('H_4', 'H_4'), ('H_8', 'H_8'),
              ('H_8plus', 'H_8plus')]
    levels, deltas, status = {}, {}, {}
    for key, _ in depths:
        levels[key], deltas[key], status[key] = {}, {}, {}
        ref = base_ref.get(key)
        if ref is not None:
            levels[key]['base'] = ref
            deltas[key]['base'] = 0.0
            status[key]['base'] = 'VALID'
        for label in others:
            run = runs.get(label, {}).get(key)
            if not run:
                status[key][label] = 'NOT RUN'
                continue
            status[key][label] = run.get('status')
            q = run.get('total_objective')
            if run.get('status') == 'VALID' and q is not None and ref is not None:
                levels[key][label] = q
                deltas[key][label] = q - ref
    report['levels'] = levels
    report['deltas_same_depth'] = deltas
    report['status'] = status

    # -------------------------------------------------------------- coverage (D11)
    coverage = {}
    for key, _ in depths:
        total = len(data['core_population'])
        ok = sum(1 for l in data['core_population']
                 if status[key].get(l) == 'VALID')
        coverage[key] = {'valid': ok, 'total': total, 'rate': ok / total,
                         'failures': [l for l in data['core_population']
                                      if status[key].get(l) not in
                                      ('VALID', None)]}
    report['coverage'] = coverage

    # ------------------------------------------------------- stability (D5)
    core_keys = ['H_2', 'H_4', 'H_8']
    joint = [l for l in data['core_population']
             if all(l in deltas[k] for k in core_keys)]
    report['jointly_valid_K2_K4_K8'] = joint

    per_candidate = {}
    for label in joint:
        values = [deltas[k][label] for k in core_keys]
        entry = {f'Delta_{k}': deltas[k][label] for k in core_keys}
        entry['spread_K2_K4_K8'] = max(values) - min(values)
        entry['sign_consistent'] = (all(v > 0 for v in values)
                                    or all(v < 0 for v in values)
                                    or all(v == 0 for v in values))
        if label in deltas['H_8plus']:
            entry['Delta_H_8plus'] = deltas['H_8plus'][label]
            entry['terminal_movement'] = abs(deltas['H_8plus'][label]
                                             - deltas['H_8'][label])
        entry['depth_movement_K4_to_K8'] = abs(deltas['H_8'][label]
                                               - deltas['H_4'][label])
        per_candidate[label] = entry
    report['per_candidate'] = per_candidate

    pairwise = {}
    for a in range(len(core_keys)):
        for b in range(a + 1, len(core_keys)):
            ka, kb = core_keys[a], core_keys[b]
            va = [deltas[ka][l] for l in joint]
            vb = [deltas[kb][l] for l in joint]
            rev = reversals(joint, va, vb)
            pairwise[f'{ka}_vs_{kb}'] = {
                'spearman': spearman(va, vb), 'kendall': kendall(va, vb),
                'n_reversals': len(rev),
                'max_rank_displacement': max(
                    abs(x - y) for x, y in zip(rank_of(va), rank_of(vb))),
                'worst_reversal': rev[0] if rev else None,
                'max_abs_delta_change': max(abs(x - y) for x, y in zip(va, vb)),
            }
    report['pairwise'] = pairwise

    report['ranking'] = {
        k: sorted([l for l in joint], key=lambda l: deltas[k][l])
        for k in core_keys}
    best = {}
    for k in core_keys:
        ordered = sorted(joint, key=lambda l: deltas[k][l])
        non_base = [l for l in ordered if l != 'base']
        best[k] = {
            'best_candidate': ordered[0],
            'best_delta': deltas[k][ordered[0]],
            'second_best': ordered[1] if len(ordered) > 1 else None,
            'gap_best_to_second': (deltas[k][ordered[1]] - deltas[k][ordered[0]])
                                  if len(ordered) > 1 else None,
            'best_improvement_over_same_depth_base':
                min(deltas[k][l] for l in non_base) if non_base else None,
        }
    report['best_per_depth'] = best

    sign_flips = [l for l in joint if not per_candidate[l]['sign_consistent']]
    report['n_sign_reversals_K2_K4_K8'] = len(sign_flips)
    report['sign_reversal_candidates'] = sign_flips

    # -------------------------------------------------- refined uncertainty (D7)
    u_depth = {}
    for label in joint:
        if label == 'base':
            continue
        parts = [per_candidate[label]['depth_movement_K4_to_K8']]
        if 'terminal_movement' in per_candidate[label]:
            parts.append(per_candidate[label]['terminal_movement'])
        u_depth[label] = max(parts)
    if u_depth:
        values = list(u_depth.values())
        report['u_depth'] = u_depth
        report['u_depth_stats'] = {
            'max': max(values), 'p95': percentile(values, 0.95),
            'median': statistics.median(values), 'min': min(values),
            'n': len(values)}
        report['tau_planning_refined'] = max(values)
        report['improvement_over_P56C'] = (
            P56C_TEMPLATE_LANDSCAPE_UNCERTAINTY / max(values)
            if max(values) > 0 else None)

    # ------------------------------------------------------------- cost (D12)
    cost = {}
    for key, _ in depths:
        valid, failed, solves = [], [], []
        for label in data['core_population']:
            run = (data['base_chain'] if label == 'base' and key == 'H_8'
                   else runs.get(label, {}).get(key))
            if not run or 'steps' not in run:
                continue
            total = run.get('total_runtime_s')
            if total is None:
                continue
            (valid if run.get('status') == 'VALID' else failed).append(total)
            solves.append(sum(s.get('polish_solve_count') or 0
                              for s in run['steps']))
        cost[key] = {
            'n_valid': len(valid), 'n_failed': len(failed),
            'median_valid_runtime_s': statistics.median(valid) if valid else None,
            'p95_valid_runtime_s': percentile(valid, 0.95) if valid else None,
            'median_failed_runtime_s': statistics.median(failed) if failed else None,
            'median_polish_solves': statistics.median(solves) if solves else None,
        }
    report['cost'] = cost

    with open(OUT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # ------------------------------------------------------------------ print
    print(f"[D5] base reference (same-depth): {base_ref}")
    print(f"[D5] jointly VALID at K=2,4,8: {len(joint)} of "
          f"{len(data['core_population'])}  {joint}\n")
    print('%-32s %14s %14s %14s %13s %8s' % ('candidate', 'Delta_2', 'Delta_4',
                                             'Delta_8', 'spread', 'sign ok'))
    for label in joint:
        v = per_candidate[label]
        print('%-32s %14.2f %14.2f %14.2f %13.2f %8s' % (
            label, v['Delta_H_2'], v['Delta_H_4'], v['Delta_H_8'],
            v['spread_K2_K4_K8'], v['sign_consistent']))
    print(f"\n[D5] improvement sign reversals across K: "
          f"{report['n_sign_reversals_K2_K4_K8']}  {sign_flips}")
    print('\n[D5] pairwise')
    for key, v in pairwise.items():
        print(f"      {key}: spearman={v['spearman']:.4f} "
              f"kendall={v['kendall']:.4f} reversals={v['n_reversals']} "
              f"max_rank_disp={v['max_rank_displacement']} "
              f"max|dDelta|={v['max_abs_delta_change']:.2f}")
    print('\n[D5] best per depth')
    for k in core_keys:
        b = best[k]
        print(f"      {k}: best={b['best_candidate']} "
              f"delta={b['best_delta']:.2f} "
              f"gap_to_second={b['gap_best_to_second']}")
    if 'u_depth' in report:
        print('\n[D7] u_depth per candidate')
        for label, value in sorted(u_depth.items(), key=lambda kv: -kv[1]):
            print(f'      {label:32s} {value:14.2f}')
        st = report['u_depth_stats']
        print(f"      max {st['max']:.2f}   p95 {st['p95']:.2f}   "
              f"median {st['median']:.2f}")
        print(f"[D7] tau_planning_refined = {report['tau_planning_refined']:.2f}")
        print(f"     versus P56C_TEMPLATE_LANDSCAPE_UNCERTAINTY 4.25e5 "
              f"-> improvement factor {report['improvement_over_P56C']:.2f}")
    print('\n[D11] coverage')
    for key, v in coverage.items():
        print(f"      {key:9s} {v['valid']}/{v['total']} = {100*v['rate']:.1f} %"
              f"   failures {v['failures']}")
    print('\n[D12] cost')
    for key, v in cost.items():
        print(f"      {key:9s} median VALID {v['median_valid_runtime_s']} s   "
              f"p95 {v['p95_valid_runtime_s']} s   "
              f"median failed {v['median_failed_runtime_s']} s")
    print(f'\n[D5] metrics -> {OUT_PATH}')


if __name__ == '__main__':
    main()
