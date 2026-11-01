"""
Stage P5.6-A6 (failure diagnosis) -- why `s+e|node9|2025|-10%` polishes to
POLISH_FAILURE under both start policies.

A6's summary keeps only the status for a failed run, which is not enough to act
on.  This re-runs that one candidate and records which of the 48 network SMOPFs
failed and what the ADMM did beforehand.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56a_a6_failure.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_candidates as C  # noqa: E402
import p56a_oracle as O  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402


def main():
    try:
        provenance, planning = gate('P5.6-A6 failure diagnosis', O.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A6f] ABORTED\n{error}')
        sys.exit(1)

    base = C.base_vector(planning)
    x = C.perturbed(base, 'se', 9, 2025, -0.10)
    print('[A6f] re-running s+e|node9|2025|-10% cold, full diagnostics ...',
          flush=True)
    result = O.evaluate_planning_candidate(
        x, start_policy=O.START_COLD, eval_id='a6_failure', use_cache=False)

    report = {'stage': 'P5.6-A6 failure', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'candidate': 's+e|node9|2025|-10%',
              'status': result['status'],
              'admm': result.get('admm'),
              'esso': result.get('esso'),
              'admm_coordination_residuals':
                  result.get('admm_coordination_residuals'),
              'capacity_shift_into_networks':
                  result.get('capacity_shift_into_networks'),
              'available_capacity': result.get('available_capacity'),
              'polish': result.get('polish')}
    out = os.path.join(O.OUT_DIR, 'p56a_a6_failure.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f"\n[A6f] status {result['status']}")
    if result.get('admm'):
        a = result['admm']
        print(f"[A6f] ADMM converged={a['converged']} cycles={a['cycles']} "
              f"recovery={a['n_recovery_diagnostics']} "
              f"runtime={a['runtime_s']:.1f}s")
    if result.get('admm_coordination_residuals'):
        print('[A6f] ADMM coordinated residuals:',
              {k: f'{v:.3e}' for k, v in
               result['admm_coordination_residuals'].items()})
    if result.get('polish'):
        failed = [b for b in result['polish']['blocks'] if not b['solved']]
        print(f"[A6f] polish: {len(failed)} of "
              f"{len(result['polish']['blocks'])} network solves FAILED")
        for b in failed:
            print(f"        {b['agent']:6s} {b['year']} {b['day']}")
    print(f'\n[A6f] report -> {out}')


if __name__ == '__main__':
    main()
