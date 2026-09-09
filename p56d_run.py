"""
Stage P5.6-D4/D6 -- run the uniformly refined oracle at K = 2, 4, 8 and the
terminal self-refinement, over the fixed core population.

Work order puts the decisive comparison first: the base refinement chain (which
supplies the same-depth reference for every K), then K=8 with its terminal
self-refinement, then K=2, then the two K=4 completions.  Results are persisted
after every evaluation so a partial run is still usable.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56d_run.py
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_candidates as AC  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p56d_oracle as D  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = D.OUT_DIR
OUT_PATH = os.path.join(OUT_DIR, 'p56d_depths.json')

# D3 core population, by label in p56b_candidates
CORE = ['base',
        'se|node5|2025|-10%',
        'se|node9|2025|-10%',
        'se|ALL|-10%',
        'e|node5|2025|+25%',
        'se|node9|2025|+10%',
        'se|node7|2030|-10%',
        'se|ALL|x19 (budget boundary)']

# K=4 evidence that P5.6-C4 already produced under a definition identical to D1
# (lambda = 0.25/0.50/0.75/1.00 from T0, midpoint anchor, full audit).
C4_REUSABLE = ['se|node9|2025|-10%', 'se|ALL|-10%',
               'e|node5|2025|+25%', 'se|node9|2025|+10%',
               'se|ALL|x19 (budget boundary)']
# re-run at K=4 rather than reuse:
#   se|node5|2025|-10%  -- C4 did not persist converter capability or the
#                          per-block network detail, and D0.1 requires them
#   se|node7|2030|-10%  -- not in the C4 target set
C4_RERUN = ['se|node5|2025|-10%', 'se|node7|2030|-10%']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-D uniform oracle', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    missing = [c for c in CORE if c not in population]
    if missing:
        print(f'[D] ABORTED: population is missing {missing}')
        sys.exit(1)
    x0 = population['base']

    report = {'stage': 'P5.6-D', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'origin': 'canonical positive-bootstrap base candidate',
              'template': D.T0_ID, 'anchor': 'midpoint', 'policy': D.POLICY,
              'core_population': CORE,
              'first_stage': {}, 'base_chain': None, 'runs': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    # ---- every candidate must pass first-stage feasibility before any solve --
    for label in CORE:
        candidate = O.vector_to_candidate(planning_gate, population[label])
        ok, reason = O.check_master_feasibility(planning_gate, candidate)
        report['first_stage'][label] = {
            'master_feasible': ok, 'reason': reason,
            'investment_cost': O.investment_cost(planning_gate, candidate)}
        if not ok:
            print(f'[D] ABORTED: {label} is first-stage infeasible: {reason}')
            sys.exit(1)
    print(f'[D] all {len(CORE)} core candidates pass first-stage feasibility\n',
          flush=True)
    persist()

    print('[D] building the frozen T0 template ...', flush=True)
    started = time.time()
    t0 = O.build_fixed_template(verbose=False)
    report['t0_build_runtime_s'] = time.time() - started
    print(f"[D] T0 built in {report['t0_build_runtime_s']:.1f}s\n", flush=True)
    persist()

    # ---------------------------------------------------------------- phase 1
    # For x = x0 every continuation point IS x0, so the lambda schedule is
    # immaterial and one 9-step chain supplies H_2(x0), H_4(x0), H_8(x0) and
    # H_8plus(x0) as the outputs of steps 2, 4, 8 and 9.
    print('[D] phase 1: base refinement chain (9 steps)', flush=True)
    chain, _ = D.run_H(x0, 9, x0, t0, tag='d_base')
    report['base_chain'] = chain
    persist()
    base_at = {}
    for step in chain['steps']:
        if step['status'] == O.STATUS_VALID:
            base_at[step['j']] = step['total_objective']
        print(f"      step {step['j']}  {step['status']:16s} "
              f"Q={step.get('total_objective')}", flush=True)
    report['base_reference'] = {'H_2': base_at.get(2), 'H_4': base_at.get(4),
                                'H_8': base_at.get(8), 'H_8plus': base_at.get(9)}
    persist()
    print(f"[D] base reference: {report['base_reference']}\n", flush=True)

    others = [c for c in CORE if c != 'base']

    # ---------------------------------------------------------------- phase 2
    print('[D] phase 2: K=8 and the terminal self-refinement\n', flush=True)
    for label in others:
        print(f'[D] === {label}  K=8', flush=True)
        run, final_state = D.run_H(population[label], 8, x0, t0,
                                   tag='d_' + label[:14].replace('|', '_')
                                       .replace(' ', ''))
        report['runs'].setdefault(label, {})['H_8'] = run
        persist()
        print(f"      status={run['status']} Q={run.get('total_objective')} "
              f"t={run['total_runtime_s']:.0f}s", flush=True)
        if final_state is not None:
            plus, _ = D.run_H(population[label], 1, x0, t0, tag='d_plus_'
                              + label[:12].replace('|', '_').replace(' ', ''),
                              start_state=final_state,
                              start_label='output of H_8 step 8')
            report['runs'][label]['H_8plus'] = plus
            persist()
            print(f"      H_8plus status={plus['status']} "
                  f"Q={plus.get('total_objective')}", flush=True)
        del final_state

    # ---------------------------------------------------------------- phase 3
    print('\n[D] phase 3: K=2\n', flush=True)
    for label in others:
        run, _ = D.run_H(population[label], 2, x0, t0,
                         tag='d2_' + label[:14].replace('|', '_').replace(' ', ''))
        report['runs'].setdefault(label, {})['H_2'] = run
        persist()
        print(f"[D] {label:32s} K=2 {run['status']:16s} "
              f"Q={run.get('total_objective')}", flush=True)

    # ---------------------------------------------------------------- phase 4
    print('\n[D] phase 4: K=4 completions\n', flush=True)
    for label in C4_RERUN:
        run, _ = D.run_H(population[label], 4, x0, t0,
                         tag='d4_' + label[:14].replace('|', '_').replace(' ', ''))
        report['runs'].setdefault(label, {})['H_4'] = run
        persist()
        print(f"[D] {label:32s} K=4 {run['status']:16s} "
              f"Q={run.get('total_objective')}", flush=True)

    # reuse the P5.6-C4 evidence where the definition matches exactly
    c4_path = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56C',
                           'p56c_continuation.json')
    with open(c4_path) as handle:
        c4 = json.load(handle)
    for label in C4_REUSABLE:
        entry = c4['targets'].get(label)
        if entry is None:
            continue
        report['runs'].setdefault(label, {})['H_4'] = {
            'K': 4, 'schedule_kind': 'uniform',
            'lambdas': [s['lambda'] for s in entry['steps']],
            'source': 'P5.6-C4 persisted evidence (definition identical to D1)',
            'status': ('VALID' if entry.get('rescued') else
                       entry['steps'][-1]['status']),
            'total_objective': entry.get('final_total_objective'),
            'total_runtime_s': entry['total_runtime_s'],
            'steps': entry['steps']}
        persist()
        print(f"[D] {label:32s} K=4 reused from P5.6-C4 "
              f"Q={entry.get('final_total_objective')}")

    persist()
    print(f'\n[D] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
