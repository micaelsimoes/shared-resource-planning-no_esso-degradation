"""
Stage P5.6-A6 / A7 -- reproducibility and cost benchmark, and the cache demo.

This is NOT an investment search.  It is a small fixed population -- the
canonical base candidate plus three P5.4-R/D3 perturbations already defined and
solved under the canonical environment -- evaluated to answer one question:

    how expensive is ONE trustworthy nonlinear planning evaluation?

That number sets the derivative-free search budget, so it is measured rather
than estimated.  Both declared start policies are evaluated for every candidate,
because A5 defines Q_oracle as the best VALID total objective over the fixed
start set and the benefit of the second start has to be paid for to be judged.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56a_a6_benchmark.py
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_candidates as C  # noqa: E402
import p56a_oracle as O  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = O.OUT_DIR
START_SET = (O.START_COLD, O.START_TEMPLATE)
TOL_CUT = 7.164e5


def summarise(result):
    if result['status'] != O.STATUS_VALID:
        return {'status': result['status'],
                'wall_clock_s': result.get('wall_clock_s')}
    return {
        'status': result['status'],
        'start_policy': result['start_policy'],
        'investment_cost': result['investment_cost'],
        'admm_recourse': result['admm']['net_operational_recourse'],
        'admm_cycles': result['admm']['cycles'],
        'admm_runtime_s': result['admm']['runtime_s'],
        'admm_recovery_solves': result['admm']['n_recovery_diagnostics'],
        'polished_gross': result['gross_operational_cost'],
        'physical_salvage': result['physical_salvage'],
        'polished_recourse': result['net_operational_recourse'],
        'total_objective': result['total_objective'],
        'polish_benefit': (result['admm']['net_operational_recourse']
                           - result['net_operational_recourse']),
        'esso_solve_count': result['esso']['solve_count'],
        'esso_runtime_s': result['esso']['runtime_s'],
        'polish_solve_count': result['polish']['solve_count'],
        'polish_runtime_s': result['polish']['runtime_s'],
        'n_nonlinear_local_solves': (result['admm']['cycles'] * 48
                                     + result['polish']['solve_count']),
        'wall_clock_s': result['wall_clock_s'],
        'max_coordinated_residual':
            result['coordination_residuals']['max_coordinated'],
        'esso_max_violation': result['esso_audit']['max_violation'],
        'esso_production_violation':
            result['esso_audit']['production_feasibility_violation'],
        'network_max_violation': result['network_audit']['max_violation'],
        'network_h1_violation':
            result['network_audit']['max_h1_complementarity_violation'],
        'consistency_iteration': result['consistency_iteration'],
        'cache_hit': result.get('cache_hit', False),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.6-A6 benchmark', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A6] ABORTED\n{error}')
        sys.exit(1)

    population = C.benchmark_population(planning)
    print(f'[A6] benchmark population: {len(population)} candidates, '
          f'start set {START_SET}\n', flush=True)

    report = {'stage': 'P5.6-A6/A7', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'start_set': list(START_SET), 'candidates': []}

    for label, x in population:
        print(f'[A6] === {label}', flush=True)
        entry = {'label': label, 'runs': {}}
        for policy in START_SET:
            started = time.time()
            result = O.evaluate_planning_candidate(
                x, start_policy=policy,
                eval_id=f'a6_{label.split()[0].replace("|", "_")}_{policy}',
                use_cache=True)
            entry['runs'][policy] = summarise(result)
            print(f"      {policy:24s} status={result['status']:12s} "
                  f"Q={result.get('total_objective')} "
                  f"t={time.time() - started:.1f}s", flush=True)
        valid = {p: r for p, r in entry['runs'].items()
                 if r['status'] == O.STATUS_VALID}
        if valid:
            best = min(valid.items(), key=lambda kv: kv[1]['total_objective'])
            entry['Q_oracle'] = best[1]['total_objective']
            entry['best_start'] = best[0]
            if len(valid) == 2:
                cold = valid[O.START_COLD]['total_objective']
                tmpl = valid[O.START_TEMPLATE]['total_objective']
                entry['second_start_benefit'] = cold - min(cold, tmpl)
                entry['second_start_cost_s'] = \
                    valid[O.START_TEMPLATE]['wall_clock_s']
            print(f"      -> Q_oracle = {entry['Q_oracle']} "
                  f"(best start {entry['best_start']})")
        else:
            entry['Q_oracle'] = None
            entry['best_start'] = None
        report['candidates'].append(entry)

    # ---- START-2 purity: the template is shared across evaluations, so it
    # ---- must not be mutated by the candidates that consume it.
    if any(r['runs'].get(O.START_TEMPLATE, {}).get('status') == O.STATUS_VALID
           for r in report['candidates']):
        label0, x0 = population[0]
        first = report['candidates'][0]['runs'].get(O.START_TEMPLATE, {})
        print('\n[A6] START-2 template purity: re-evaluating the first candidate '
              'after every other candidate has consumed the template',
              flush=True)
        repeat = O.evaluate_planning_candidate(
            x0, start_policy=O.START_TEMPLATE, eval_id='a6_template_repeat',
            use_cache=False)
        delta = (None if repeat['status'] != O.STATUS_VALID
                 or first.get('status') != O.STATUS_VALID
                 else repeat['total_objective'] - first['total_objective'])
        report['template_purity'] = {
            'candidate': label0,
            'first_total_objective': first.get('total_objective'),
            'repeat_total_objective': repeat.get('total_objective'),
            'repeat_status': repeat['status'],
            'delta': delta,
            'pass': delta is not None and abs(delta) <= 1.0,
        }
        print(f"      first={first.get('total_objective')} "
              f"repeat={repeat.get('total_objective')} delta={delta}")

    # ---------------------------------------------------------- A7 cache demo
    print('\n[A7] cache demonstration on the first benchmark candidate',
          flush=True)
    label, x = population[0]
    started = time.time()
    hit = O.evaluate_planning_candidate(x, start_policy=O.START_COLD,
                                        eval_id='a7_cache_hit', use_cache=True)
    hit_time = time.time() - started
    report['cache_demo'] = {
        'candidate': label,
        'cache_hit': hit.get('cache_hit', False),
        'wall_clock_s': hit_time,
        'total_objective': hit.get('total_objective'),
        'cache_key': hit.get('cache_key'),
        'cache_payload': hit.get('cache_payload'),
        'cached_entries': len(O.load_cache()),
    }
    print(f"      cache_hit={hit.get('cache_hit')} "
          f"t={hit_time:.3f}s Q={hit.get('total_objective')}")
    print(f"      cache holds {len(O.load_cache())} VALID entries")

    # ------------------------------------------------------------- aggregates
    runtimes, spreads = [], []
    for entry in report['candidates']:
        for policy, run in entry['runs'].items():
            if run['status'] == O.STATUS_VALID and not run.get('cache_hit'):
                runtimes.append(run['wall_clock_s'])
        valid = [r['total_objective'] for r in entry['runs'].values()
                 if r['status'] == O.STATUS_VALID]
        if len(valid) > 1:
            spreads.append(max(valid) - min(valid))
    if runtimes:
        runtimes_sorted = sorted(runtimes)
        p95 = runtimes_sorted[min(len(runtimes_sorted) - 1,
                                  int(0.95 * len(runtimes_sorted)))]
        report['runtime'] = {
            'n': len(runtimes), 'median_s': statistics.median(runtimes),
            'p95_s': p95, 'min_s': min(runtimes), 'max_s': max(runtimes),
            'mean_s': statistics.fmean(runtimes)}
    polish_benefits = [r['polish_benefit'] for e in report['candidates']
                       for r in e['runs'].values()
                       if r['status'] == O.STATUS_VALID]
    if polish_benefits:
        report['polish_benefit'] = {
            'median': statistics.median(polish_benefits),
            'min': min(polish_benefits), 'max': max(polish_benefits)}
    if spreads:
        report['objective_spread_across_starts'] = {
            'median': statistics.median(spreads), 'max': max(spreads),
            'max_over_tol_cut': max(spreads) / TOL_CUT}
    second = [e.get('second_start_benefit') for e in report['candidates']
              if e.get('second_start_benefit') is not None]
    if second:
        report['second_start_benefit'] = {
            'median': statistics.median(second), 'max': max(second),
            'max_over_tol_cut': max(second) / TOL_CUT,
            'n_candidates_improved': sum(1 for v in second if v > 0.0)}

    out = os.path.join(OUT_DIR, 'p56a_a6_benchmark.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('\n[A6] cost of one trustworthy nonlinear planning evaluation')
    if 'runtime' in report:
        r = report['runtime']
        print(f"      n={r['n']}  median {r['median_s']:.1f}s  "
              f"p95 {r['p95_s']:.1f}s  range {r['min_s']:.1f}-{r['max_s']:.1f}s")
    if 'polish_benefit' in report:
        b = report['polish_benefit']
        print(f"      polish benefit (ADMM recourse - polished): "
              f"median {b['median']:.2f}, range {b['min']:.2f}..{b['max']:.2f}")
    if 'second_start_benefit' in report:
        s = report['second_start_benefit']
        print(f"      second start improved {s['n_candidates_improved']} of "
              f"{len(second)} candidates; max benefit {s['max']:.2f} "
              f"({s['max_over_tol_cut']:.4f} x tol_cut)")
    print(f'\n[A6/A7] report -> {out}')


if __name__ == '__main__':
    main()
