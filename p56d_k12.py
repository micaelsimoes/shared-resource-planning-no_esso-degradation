"""
Stage P5.6-D9 -- conditional K=12 escalation.

D9 is entered only because K4 versus K8 is materially unstable: Spearman 0.5714,
Kendall 0.4286, six pairwise reversals, and the best candidate changing identity
between the two depths.  It is run on the reduced diagnostic subset D9 prescribes,
not on the whole population, and K=12 is this stage's hard maximum.

The question is narrow: is the movement of the RELATIVE deltas decaying with
depth?  The terminal self-refinement (D6) moved them by at most 88 374 while
K4 -> K8 moved them by up to 811 438, which is consistent either with decay or
with the terminal step simply being one step rather than four.  K=12 separates
those.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56d_k12.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p56d_oracle as D  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(D.OUT_DIR, 'p56d_k12.json')

# D9's reduced subset: the base, the best K=8 candidate, the one candidate whose
# improvement sign changed (which is also the budget-boundary control), and the
# candidate with the second largest depth movement.
SUBSET = ['base',
          'se|node9|2025|-10%',            # best at K=8
          'se|ALL|x19 (budget boundary)',  # sign change + remote control
          'se|node9|2025|+10%']            # second largest u_depth


def main():
    os.makedirs(D.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-D9 K12', D.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D9] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    x0 = population['base']

    report = {'stage': 'P5.6-D9', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'subset': SUBSET, 'K': 12, 'hard_maximum_K': 12, 'runs': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    print('[D9] building the frozen T0 template ...', flush=True)
    started = time.time()
    t0 = O.build_fixed_template(verbose=False)
    print(f'[D9] T0 built in {time.time() - started:.1f}s\n', flush=True)

    for label in SUBSET:
        print(f'[D9] === {label}  K=12', flush=True)
        run, _ = D.run_H(population[label], 12, x0, t0,
                         tag='d9_' + label[:14].replace('|', '_').replace(' ', ''))
        report['runs'][label] = run
        persist()
        print(f"      status={run['status']} Q={run.get('total_objective')} "
              f"t={run['total_runtime_s']:.0f}s", flush=True)

    # ---- compare relative deltas at K=8 and K=12 ---------------------------
    metrics_path = os.path.join(D.OUT_DIR, 'p56d_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        base12 = report['runs'].get('base', {}).get('total_objective')
        comparison = {}
        if base12 is not None:
            for label in SUBSET:
                if label == 'base':
                    continue
                run = report['runs'].get(label, {})
                if run.get('status') != 'VALID':
                    comparison[label] = {'status': run.get('status')}
                    continue
                delta12 = run['total_objective'] - base12
                delta8 = metrics['deltas_same_depth']['H_8'].get(label)
                comparison[label] = {
                    'Delta_8': delta8, 'Delta_12': delta12,
                    'movement_K8_to_K12': (abs(delta12 - delta8)
                                           if delta8 is not None else None),
                    'sign_consistent': (delta8 is not None
                                        and (delta8 < 0) == (delta12 < 0)),
                }
        report['base_H_12'] = base12
        report['comparison_K8_vs_K12'] = comparison
        moves = [v['movement_K8_to_K12'] for v in comparison.values()
                 if v.get('movement_K8_to_K12') is not None]
        if moves:
            report['max_movement_K8_to_K12'] = max(moves)
            report['prior_max_movement_K4_to_K8'] = metrics.get(
                'tau_planning_refined')
            report['movement_decay_factor'] = (
                metrics['tau_planning_refined'] / max(moves)
                if max(moves) > 0 else None)
        persist()

        print('\n[D9] relative deltas, K=8 versus K=12')
        for label, v in comparison.items():
            if 'Delta_8' not in v:
                print(f"      {label:32s} {v.get('status')}")
                continue
            print(f"      {label:32s} Delta_8={v['Delta_8']:14.2f} "
                  f"Delta_12={v['Delta_12']:14.2f} "
                  f"movement={v['movement_K8_to_K12']:12.2f} "
                  f"sign_ok={v['sign_consistent']}")
        if moves:
            print(f"\n[D9] max movement K8->K12 : "
                  f"{report['max_movement_K8_to_K12']:.2f}")
            print(f"[D9] prior max K4->K8     : "
                  f"{report['prior_max_movement_K4_to_K8']:.2f}")
            print(f"[D9] decay factor         : "
                  f"{report['movement_decay_factor']:.2f}")

    persist()
    print(f'\n[D9] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
