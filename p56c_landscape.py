"""
Stage P5.6-C2 -- cross-template landscape test.

P5.6-B1 showed the template chain does not converge: T0..T4 differ by ~2.06e6 at
the base candidate.  That is a statement about LEVELS.  A derivative-free search
does not care about levels; it cares whether the ORDERING of investment
candidates is the same.  This measures that directly.

Templates are regenerated deterministically in a single process -- T0 is the cold
base solution, and Tk+1 is the operational state produced by evaluating the SAME
base candidate from Tk -- and the benchmark population is then evaluated under
T0, T2 and T4 with the locked midpoint anchor.

Everything reported per candidate is the delta relative to that template's own
base value, because that is the quantity a search actually follows.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56c_landscape.py
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
import p56b_candidates as BC  # noqa: E402
import p56b_policy as P  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56C')
TEMPLATES_TO_TEST = (0, 2, 4)
MAX_GENERATION = max(TEMPLATES_TO_TEST)
ANCHOR = P.ANCHOR_MIDPOINT
POLICY = P.POLICY_MIDPOINT_ONLY          # locked recurring convention


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
    n, concordant, discordant = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return ((concordant - discordant) / total) if total else None


def reversals(labels, a, b):
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                out.append({'pair': [labels[i], labels[j]],
                            'gap_first': a[i] - a[j],
                            'gap_second': b[i] - b[j],
                            'min_abs_gap': min(abs(a[i] - a[j]),
                                               abs(b[i] - b[j]))})
    out.sort(key=lambda r: -r['min_abs_gap'])
    return out


def linear_fit(x, y):
    """y = a + b x, least squares, plus R^2 -- a diagnostic only."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((u - mx) * (v - my) for u, v in zip(x, y))
    if sxx == 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((v - (a + b * u)) ** 2 for u, v in zip(x, y))
    ss_tot = sum((v - my) ** 2 for v in y)
    return {'a': a, 'b': b, 'r_squared': (1 - ss_res / ss_tot) if ss_tot else None,
            'max_abs_residual': max(abs(v - (a + b * u)) for u, v in zip(x, y))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-C2 landscape', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C2] ABORTED\n{error}')
        sys.exit(1)

    population = BC.population(planning_gate)
    base_x = population[0][1]
    report = {'stage': 'P5.6-C2', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'anchor': ANCHOR, 'policy': POLICY,
              'templates_tested': list(TEMPLATES_TO_TEST),
              'templates': {}, 'evaluations': {}}
    out_path = os.path.join(OUT_DIR, 'p56c_landscape.json')

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    # ------------------------------------------------ deterministic templates
    print('[C2] regenerating the deterministic template chain T0..T'
          f'{MAX_GENERATION} ...', flush=True)
    started = time.time()
    templates = {0: O.build_fixed_template(verbose=False)}
    report['templates']['0'] = {'definition': 'cold solution of the canonical base',
                                'build_runtime_s': time.time() - started}
    print(f'[C2] T0 built in {time.time() - started:.1f}s', flush=True)
    for k in range(MAX_GENERATION):
        started = time.time()
        r = P.evaluate(base_x, template_state=templates[k], anchor_policy=POLICY,
                       eval_id=f'c2_chain_T{k}', keep_state=True)
        if r['status'] != O.STATUS_VALID:
            report['templates'][str(k + 1)] = {'status': r['status']}
            print(f"[C2] chain broke at T{k + 1}: {r['status']}", flush=True)
            persist()
            sys.exit(2)
        templates[k + 1] = r['_state']
        report['templates'][str(k + 1)] = {
            'definition': f'operational state of the base candidate evaluated from T{k}',
            'base_total_objective': r['total_objective'],
            'build_runtime_s': time.time() - started}
        print(f"[C2] T{k + 1} built in {time.time() - started:.1f}s   "
              f"base Q = {r['total_objective']:.6f}", flush=True)
    persist()

    # --------------------------------------------- evaluate under each template
    for k in TEMPLATES_TO_TEST:
        print(f'\n[C2] === evaluating the population under T{k}', flush=True)
        report['evaluations'][f'T{k}'] = {}
        for label, x in population:
            eid = f'c2_T{k}_' + label[:18].replace('|', '_').replace(' ', '')
            r = P.evaluate(x, template_state=templates[k], anchor_policy=POLICY,
                           eval_id=eid)
            entry = {'status': r['status'],
                     'total_objective': r.get('total_objective'),
                     'wall_clock_s': r['wall_clock_s'],
                     'admm_cycles': (r.get('admm') or {}).get('cycles'),
                     'failed_blocks': r.get('failed_blocks')}
            if r['status'] == O.STATUS_VALID:
                entry['max_coordinated_residual'] = \
                    r['coordination_residuals']['max_coordinated']
                entry['esso_max_violation'] = r['esso_audit']['max_violation']
                entry['network_max_violation'] = r['network_audit']['max_violation']
            report['evaluations'][f'T{k}'][label] = entry
            print(f"      {label:32s} {r['status']:16s} "
                  f"Q={r.get('total_objective')} t={r['wall_clock_s']:.0f}s",
                  flush=True)
            persist()

    # ------------------------------------------------------------- statistics
    tags = [f'T{k}' for k in TEMPLATES_TO_TEST]
    joint = [label for label, _ in population
             if all(report['evaluations'][t][label]['status'] == O.STATUS_VALID
                    for t in tags)]
    report['jointly_valid'] = joint
    print(f'\n[C2] jointly VALID under all tested templates: {len(joint)} of '
          f'{len(population)}')

    levels = {t: [report['evaluations'][t][l]['total_objective'] for l in joint]
              for t in tags}
    base_level = {t: report['evaluations'][t]['base']['total_objective']
                  for t in tags}
    deltas = {t: [v - base_level[t] for v in levels[t]] for t in tags}
    report['base_level_per_template'] = base_level
    report['levels'] = {t: dict(zip(joint, levels[t])) for t in tags}
    report['deltas_vs_own_base'] = {t: dict(zip(joint, deltas[t])) for t in tags}

    per_candidate = {}
    for i, label in enumerate(joint):
        values = [deltas[t][i] for t in tags]
        per_candidate[label] = {
            **{f'delta_{t}': deltas[t][i] for t in tags},
            'spread': max(values) - min(values),
            'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
            'sign_consistent': (all(v > 0 for v in values)
                                or all(v < 0 for v in values)
                                or all(v == 0 for v in values)),
        }
    report['per_candidate_delta'] = per_candidate

    pairwise = {}
    for a in range(len(tags)):
        for b in range(a + 1, len(tags)):
            ta, tb = tags[a], tags[b]
            key = f'{ta}_vs_{tb}'
            rev = reversals(joint, deltas[ta], deltas[tb])
            ranks_a, ranks_b = rank_of(levels[ta]), rank_of(levels[tb])
            pairwise[key] = {
                'spearman_levels': spearman(levels[ta], levels[tb]),
                'kendall_levels': kendall(levels[ta], levels[tb]),
                'spearman_deltas': spearman(deltas[ta], deltas[tb]),
                'kendall_deltas': kendall(deltas[ta], deltas[tb]),
                'max_rank_displacement': max(abs(x - y) for x, y in
                                             zip(ranks_a, ranks_b)),
                'n_pairwise_reversals': len(rev),
                'worst_reversal': rev[0] if rev else None,
                'affine_fit_level': linear_fit(levels[ta], levels[tb]),
            }
    report['pairwise'] = pairwise

    report['ranking'] = {t: [joint[i] for i in sorted(range(len(joint)),
                                                      key=lambda j: levels[t][j])]
                         for t in tags}
    report['max_delta_spread_over_candidates'] = max(
        v['spread'] for v in per_candidate.values()) if per_candidate else None
    report['n_sign_inconsistent'] = sum(1 for v in per_candidate.values()
                                        if not v['sign_consistent'])
    persist()

    print('\n[C2] deltas relative to each template\'s own base')
    header = '      %-32s' % 'candidate' + ''.join(f'{t:>16s}' for t in tags) \
             + f'{"spread":>14s}{"sign ok":>9s}'
    print(header)
    for label in joint:
        v = per_candidate[label]
        print('      %-32s' % label
              + ''.join(f"{v[f'delta_{t}']:16.2f}" for t in tags)
              + f"{v['spread']:14.2f}{str(v['sign_consistent']):>9s}")
    print('\n[C2] pairwise template comparison')
    for key, v in pairwise.items():
        print(f"      {key}: spearman(levels)={v['spearman_levels']:.4f} "
              f"kendall(levels)={v['kendall_levels']:.4f} "
              f"reversals={v['n_pairwise_reversals']} "
              f"max_rank_disp={v['max_rank_displacement']}")
        fit = v['affine_fit_level']
        if fit:
            print(f"          affine fit Q_b = {fit['a']:.2f} + "
                  f"{fit['b']:.6f} Q_a   R^2={fit['r_squared']:.6f} "
                  f"max|resid|={fit['max_abs_residual']:.2f}")
    print(f"\n[C2] max delta spread across templates: "
          f"{report['max_delta_spread_over_candidates']}")
    print(f"[C2] candidates whose improvement CHANGES SIGN: "
          f"{report['n_sign_inconsistent']}")
    print(f'\n[C2] report -> {out_path}')


if __name__ == '__main__':
    main()
