"""
P5.6-D -- the uniformly refined nonlinear oracle  H_K(x).

P5.6-C showed that deterministic template generations reorder the investment
landscape (Spearman T0 vs T4 = -0.033, 6 of 9 improvement signs reversed), and
that continuation rescues every failure but shifts even direct-valid solutions by
~2.9e6.  So continuation cannot be a failure fallback; it has to be the oracle
itself, applied identically to every candidate.  This module defines that oracle.

    x0     = the canonical positive-bootstrap base investment, FIXED FOREVER
    T0     = the frozen cold-base template, id a81f7f5191dd42dbf50d1726149b8909
    anchor = MIDPOINT, always

    lambda_j = j / K,   x_j = (1 - lambda_j) * x0 + lambda_j * x,   j = 1..K

    state_0 = T0
    for j = 1..K:  evaluate x_j from state_{j-1}; on VALID, state_j = its state
    H_K(x) = total planning objective at x_K = x

The origin is ALWAYS x0.  Never the incumbent, never the previous candidate,
never the nearest cached point, never search history -- any of those would make
the objective depend on search order, which is the defect the whole P5.6 line has
been removing.

For x = x0 every continuation point is x0, so H_K(x0) is K deterministic repeated
refinements of the base.  That is intentional: it is the only fair reference,
because a candidate at depth K must be compared against a base that has received
exactly the same amount of refinement.  Comparing H_K(x) with H_J(x0) for K != J
is forbidden.

Generation labelling, to remove the off-by-one ambiguity of the earlier reports:

    INPUT state of step j   = state_{j-1}   (what initializes the solve)
    OUTPUT state of step j  = state_j       (what the solve produced)

    so H_K(x0) is the OUTPUT of the K-th evaluation, and the template called
    "T4" in P5.6-C is the OUTPUT of the 4th base evaluation.
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

import p56a_oracle as O  # noqa: E402
import p56b_policy as P  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56D')
POLICY = P.POLICY_MIDPOINT_ONLY
T0_ID = 'a81f7f5191dd42dbf50d1726149b8909'


def blend(x0, x, lam):
    """The continuation point; the first-stage set is a polyhedron, so it is feasible."""
    return {key: {'s': (1.0 - lam) * x0[key]['s'] + lam * x[key]['s'],
                  'e': (1.0 - lam) * x0[key]['e'] + lam * x[key]['e']}
            for key in x0}


def schedule(K, kind='uniform'):
    """lambda_j for j = 1..K.  'uniform' is j/K; 'squared' is (j/K)^2."""
    if kind == 'uniform':
        return [j / K for j in range(1, K + 1)]
    if kind == 'squared':
        return [(j / K) ** 2 for j in range(1, K + 1)]
    raise ValueError(f'unknown schedule {kind}')


def certificate(result):
    """Everything D0.1 requires, captured at every step rather than a subset."""
    if result.get('status') != O.STATUS_VALID:
        return {'status': result.get('status'),
                'failed_blocks': result.get('failed_blocks'),
                'master_feasible': result.get('master_feasible')}
    audit_n = result.get('network_audit') or {}
    audit_e = result.get('esso_audit') or {}
    residuals = result.get('coordination_residuals') or {}
    admm = result.get('admm') or {}
    return {
        'status': result.get('status'),
        'master_feasible': result.get('master_feasible'),
        'total_objective': result.get('total_objective'),
        'investment_cost': result.get('investment_cost'),
        'net_operational_recourse': result.get('net_operational_recourse'),
        'gross_operational_cost': result.get('gross_operational_cost'),
        'physical_salvage': result.get('physical_salvage'),
        'coordination_residuals': residuals,
        'max_coordinated_residual': residuals.get('max_coordinated'),
        'esso_max_violation': audit_e.get('max_violation'),
        'esso_production_violation':
            audit_e.get('production_feasibility_violation'),
        'esso_production_feasible': audit_e.get('production_feasible'),
        'esso_nodes_audited': sorted((audit_e.get('per_node') or {}).keys()),
        'network_max_violation': audit_n.get('max_violation'),
        'network_worst_block': audit_n.get('worst_block'),
        'network_blocks_audited': len(audit_n.get('per_block') or {}),
        'network_h1_violation': audit_n.get('max_h1_complementarity_violation'),
        'converter_capability_violation':
            audit_n.get('max_converter_capability_violation'),
        'admm_cycles': admm.get('cycles'),
        'admm_converged': admm.get('converged'),
        'admm_runtime_s': admm.get('runtime_s'),
        'polish_runtime_s': result.get('polish_runtime_s'),
        'polish_solve_count': result.get('polish_solve_count'),
        'esso_solve_count': result.get('esso_solve_count'),
        'wall_clock_s': result.get('wall_clock_s'),
    }


def run_H(x, K, x0, t0_state, kind='uniform', tag='H', start_state=None,
          start_label=None):
    """Walk the continuation and return the full record.

    `start_state` / `start_label` allow a chain to be EXTENDED (the terminal
    self-refinement of D6) without redoing the walk.
    """
    lambdas = schedule(K, kind)
    state = t0_state if start_state is None else start_state
    record = {'K': K, 'schedule_kind': kind, 'lambdas': lambdas,
              'origin': 'canonical base x0', 'template': T0_ID,
              'anchor': P.ANCHOR_MIDPOINT, 'policy': POLICY,
              'input_state_of_step_1': start_label or 'T0',
              'steps': [], 'total_runtime_s': 0.0}
    final, final_state = None, None
    for j, lam in enumerate(lambdas, start=1):
        x_j = blend(x0, x, lam)
        eid = f'{tag}_K{K}_{kind}_j{j}'
        result = P.evaluate(x_j, template_state=state, anchor_policy=POLICY,
                            eval_id=eid, keep_state=True)
        step = {'j': j, 'lambda': lam, **certificate(result)}
        step['input_state'] = f'output of step {j - 1}' if j > 1 else \
            (start_label or 'T0')
        record['steps'].append(step)
        record['total_runtime_s'] += result['wall_clock_s']
        if result['status'] != O.STATUS_VALID:
            record['status'] = result['status']
            record['stopped_at_step'] = j
            record['stopped_at_lambda'] = lam
            record['total_objective'] = None
            return record, None
        state = result['_state']
        final, final_state = result, state
    record['status'] = O.STATUS_VALID
    record['total_objective'] = final['total_objective']
    record['certificate'] = certificate(final)
    return record, final_state
