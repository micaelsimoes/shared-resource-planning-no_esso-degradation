"""
Stage P5.6-A3 -- purity regression for the candidate-evaluation oracle.

P5.5-A12 recorded that production mutates persistent planning data in place, so
the recourse returned for a candidate depended on which candidate had been
evaluated before it -- P5.4-D4 saw base recourse values 1.5e5 apart from call
order alone.  A derivative-free planner cannot optimise an objective like that.

`p56a_oracle.fresh_planning` gives every evaluation its own deep copy of the
baseline, its own solver-log directory and its own diagnostics accumulators, and
never mutates the baseline.  This script tests whether that is enough, by
evaluating the same candidate from two different call histories:

    sequence 1 :  x_A , x_B , x_A
    sequence 2 :  x_A , x_C , x_A

All four x_A results must agree.  Caching is DISABLED here: a cache hit would
return the first result and hide exactly the defect being tested.

DECLARED TOLERANCE, fixed before the run
----------------------------------------
    objective agreement            |dQ| <= 1.0        (absolute)

against a total objective of order 7.3e8 and a canonical Benders cut tolerance
tol_cut = 7.164e5, so the requirement is 1.4e-9 of the objective and 1.4e-6 of
the investment signal -- materially below it, as required.  Bit-identical
agreement is reported separately but is not the pass criterion, and solver logs
are not required to match.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56a_a3_purity.py
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

import p56a_candidates as C  # noqa: E402
import p56a_oracle as O  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = O.OUT_DIR
OBJECTIVE_TOLERANCE = 1.0
TOL_CUT = 7.164e5


def fingerprint(result):
    """Everything the regression compares, objective and physics alike."""
    if result['status'] != O.STATUS_VALID:
        return {'status': result['status']}
    return {
        'status': result['status'],
        'total_objective': result['total_objective'],
        'net_operational_recourse': result['net_operational_recourse'],
        'gross_operational_cost': result['gross_operational_cost'],
        'physical_salvage': result['physical_salvage'],
        'investment_cost': result['investment_cost'],
        'admm_cycles': result['admm']['cycles'],
        'admm_recourse': result['admm']['net_operational_recourse'],
        'coordination_residuals': result['coordination_residuals'],
        'esso_max_violation': result['esso_audit']['max_violation'],
        'esso_production_violation':
            result['esso_audit']['production_feasibility_violation'],
        'network_max_violation': result['network_audit']['max_violation'],
        'network_h1_violation':
            result['network_audit']['max_h1_complementarity_violation'],
        'consistency_iteration': result['consistency_iteration'],
    }


def compare(reference, other):
    diffs = {}
    for key, value in reference.items():
        if key not in other:
            diffs[key] = 'missing'
            continue
        peer = other[key]
        if isinstance(value, (int, float)) and isinstance(peer, (int, float)):
            diffs[key] = float(peer) - float(value)
        elif isinstance(value, dict) and isinstance(peer, dict):
            diffs[key] = {k: (peer.get(k, 0.0) - v) for k, v in value.items()
                          if isinstance(v, (int, float))}
        else:
            diffs[key] = 'same' if value == peer else f'{value} -> {peer}'
    return diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.6-A3 purity', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A3] ABORTED\n{error}')
        sys.exit(1)

    base = C.base_vector(planning)
    x_a = base
    x_b = C.perturbed(base, 's', 5, 2025, -0.10)
    x_c = C.perturbed(base, 'e', 9, 2025, -0.10)

    print(f'[A3] declared objective tolerance |dQ| <= {OBJECTIVE_TOLERANCE} '
          f'({OBJECTIVE_TOLERANCE / TOL_CUT:.2e} x tol_cut)')
    print('[A3] caching DISABLED for this test\n', flush=True)

    plan = [('A1', x_a), ('B', x_b), ('A2', x_a), ('C', x_c), ('A3', x_a)]
    runs, order = {}, []
    for tag, x in plan:
        started = time.time()
        print(f'[A3] evaluating {tag} ...', flush=True)
        result = O.evaluate_planning_candidate(
            x, start_policy=O.START_COLD, eval_id=f'a3_{tag}', use_cache=False)
        runs[tag] = result
        order.append(tag)
        print(f"      status={result['status']} "
              f"Q={result.get('total_objective')} "
              f"t={time.time() - started:.1f}s", flush=True)

    prints = {tag: fingerprint(runs[tag]) for tag in order}
    reference = prints['A1']
    report = {
        'stage': 'P5.6-A3', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'declared_objective_tolerance': OBJECTIVE_TOLERANCE,
        'tol_cut': TOL_CUT,
        'tolerance_as_fraction_of_tol_cut': OBJECTIVE_TOLERANCE / TOL_CUT,
        'sequence': [t for t, _ in plan],
        'fingerprints': prints,
        'comparisons': {tag: compare(reference, prints[tag])
                        for tag in ('A2', 'A3')},
    }

    valid = all(prints[t].get('status') == O.STATUS_VALID
                for t in ('A1', 'A2', 'A3'))
    deltas = {}
    if valid:
        for tag in ('A2', 'A3'):
            deltas[tag] = abs(prints[tag]['total_objective']
                              - reference['total_objective'])
    report['objective_deltas'] = deltas
    report['bit_identical'] = valid and all(d == 0.0 for d in deltas.values())
    report['pass'] = valid and all(d <= OBJECTIVE_TOLERANCE
                                   for d in deltas.values())

    out = os.path.join(OUT_DIR, 'p56a_a3_purity.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('\n[A3] x_A evaluated from three different call histories')
    for tag in ('A1', 'A2', 'A3'):
        entry = prints[tag]
        print(f"      {tag}: status={entry.get('status')} "
              f"Q={entry.get('total_objective')}")
    if deltas:
        print('\n[A3] objective differences against the first evaluation')
        for tag, delta in deltas.items():
            print(f'      {tag}: {delta:.6e}  '
                  f'({delta / TOL_CUT:.2e} x tol_cut)')
    print(f"\n[A3] bit-identical : {report['bit_identical']}")
    print(f"[A3] PASS          : {report['pass']}  "
          f"(tolerance {OBJECTIVE_TOLERANCE})")
    print(f'\n[A3] report -> {out}')


if __name__ == '__main__':
    main()
