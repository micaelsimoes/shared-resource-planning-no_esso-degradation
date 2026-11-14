"""
P5.15-1 -- EPS_ESSO_THROUGHPUT vs. AL gradient-scale comparison (Step 1 item 3,
"Required measurement").

Loads the preserved P5.14-N CONTROL-arm ESSO models
(`data/SRP1/Results/P514N/esso_models_control.pkl`, the OLD pre-reformulation
model structure -- used here read-only, purely to read out converged
`es_pnet` / `p_req` / `dual_p_req` / `rho` values and the `admm_objective`
expression; nothing is rebuilt or re-solved from it) and computes, at every
(year, day, period) of nodes 7 and 9:

  AL gradient  = d(admm_objective)/d(es_pnet[y,d,p])
               = [dual_p_req + rho * (es_pnet - p_req)/(2*rating)] / (2*rating)

  via Pyomo's exact symbolic differentiation (`differentiate`, reverse-symbolic
  mode), evaluated at the pickle's stored (converged) values -- NOT
  re-derived by hand, so it is exactly what production's AL term produces.

`es_pnet = sum_{y_inv} (pch[y_inv,y,d,p] - pdch[y_inv,y,d,p])`, and this sum is
linear with unit coefficients, so d(es_pnet)/d(pch) = d(es_pnet)/d(pdch) = +/-1
for every active cohort -- the AL gradient w.r.t. es_pnet IS the AL gradient
w.r.t. any individual active pch/pdch (apples-to-apples with EPS_ESSO_THROUGHPUT,
whose gradient w.r.t. pch/pdch is the constant EPS_ESSO_THROUGHPUT itself, since
the throughput term is linear).

Not a gate, not a frozen artifact (Steps 0-2 do not require the ceremony per
the brief); this IS a number that will be reused (the EPS/AL ratio), so the
instance, convention and both magnitudes are recorded here per CLAUDE.md.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_1_eps_al_gradient_comparison.py
"""

import json
import os
import pickle
import statistics

import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import differentiate, Modes

from definitions import EPS_ESSO_THROUGHPUT

PICKLE_PATH = os.path.join('data', 'SRP1', 'Results', 'P514N', 'esso_models_control.pkl')
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P5151')
NODES = (7, 9)


def al_gradient_stats(model):
    values = []
    for y in model.years:
        for d in model.days:
            for p in model.periods:
                grad_expr = differentiate(
                    model.admm_objective.expr, wrt=model.es_pnet[y, d, p], mode=Modes.reverse_symbolic
                )
                values.append(pe.value(grad_expr))
    abs_values = [abs(v) for v in values]
    return {
        'n': len(values),
        'max_abs': max(abs_values),
        'mean_abs': sum(abs_values) / len(abs_values),
        'median_abs': statistics.median(abs_values),
        'min_abs': min(abs_values),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PICKLE_PATH, 'rb') as fh:
        models = pickle.load(fh)

    report = {
        'instance': PICKLE_PATH,
        'convention': (
            'AL gradient = d(admm_objective)/d(es_pnet[y,d,p]) via Pyomo exact '
            'symbolic differentiation, evaluated at the pickled (converged) '
            'point. EPS gradient = EPS_ESSO_THROUGHPUT (constant; the '
            'throughput regularization term is linear in pch/pdch). Both are '
            'gradients w.r.t. the same physical quantity (a cohort pch/pdch, '
            'since d(es_pnet)/d(pch) = 1 for an active cohort).'
        ),
        'eps_esso_throughput': EPS_ESSO_THROUGHPUT,
        'nodes': {},
    }

    for node_id in NODES:
        model = models[node_id]
        stats = al_gradient_stats(model)
        stats['ratio_eps_over_al_max'] = EPS_ESSO_THROUGHPUT / stats['max_abs']
        stats['ratio_eps_over_al_mean'] = EPS_ESSO_THROUGHPUT / stats['mean_abs']
        stats['ratio_eps_over_al_median'] = EPS_ESSO_THROUGHPUT / stats['median_abs']
        report['nodes'][str(node_id)] = stats

    out_path = os.path.join(OUT_DIR, 'eps_al_gradient_comparison.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
