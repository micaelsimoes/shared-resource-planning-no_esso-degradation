"""
P5.13-F — re-audit of P5.12-T's zero-solve claim under ARMED GUARDS.

P5.12-T's committed report states "Zero solves were performed." That claim was
asserted, not enforced: T installs no guards. The claim is very probably true — T
parses three preserved IPOPT logs — but by this project's own standard an argument is
not evidence, which is the distinction the P5.12-R first-repair rejection turned on.

This re-runs T's own harness unchanged, with every solve path blocked, and lets an
enforced zero replace the asserted one. Recorded as the same defect found in an earlier
stage, not as a separate incident.

The original `data/SRP1/Results/P512T` artifacts are NOT touched; output is redirected
to a re-audit directory.
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p513_solve_profile_guard import SolveProfileGuard, SolverInvocationBlocked  # noqa: E402
import p512_t_trajectory_forensic as t  # noqa: E402

REAUDIT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P512T_REAUDIT')


def main():
    original_out = t.OUT_DIR
    t.OUT_DIR = REAUDIT_DIR
    guard = SolveProfileGuard(permitted=(), label='P5.12-T re-audit (zero solves)').install()
    error = None
    try:
        t.main()
    except SolverInvocationBlocked as exc:
        error = f'SOLVE ATTEMPTED: {exc}'
    except SystemExit as exc:
        error = None if exc.code in (0, None) else f'harness exited with {exc.code}'
    except Exception as exc:  # noqa: BLE001 - the verdict must record any failure
        error = f'{type(exc).__name__}: {exc}'
    finally:
        guard.uninstall()
        t.OUT_DIR = original_out

    failures = []
    if guard.counts['permitted_solve'] or guard.counts['blocked_solve']:
        failures.append(f'solves observed: {guard.counts}')
    if guard.counts['permitted_exec'] or guard.counts['blocked_exec']:
        failures.append(f'solver process launches observed: {guard.counts}')
    if error:
        failures.append(error)

    verdict = {
        'stage': 'P5.13-F', 'audited_stage': 'P5.12-T',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'claim_audited': 'P5_12_T_TRAJECTORY_FORENSIC_REPORT.md: "Zero solves were performed."',
        'defect': 'the claim was asserted, not enforced; P5.12-T installs no solver guards',
        'method': 'P5.12-T harness re-run unchanged with every solve path blocked; output '
                  'redirected to P512T_REAUDIT so the original artifacts are untouched',
        'observed_counts': dict(guard.counts),
        'harness_error': error,
        'failures': failures,
        'verdict': 'PASS — zero solves ENFORCED, not asserted' if not failures
                   else 'FAIL — the asserted zero does not hold under guards',
    }
    os.makedirs(os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P513D'), exist_ok=True)
    out = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P513D', 't_reaudit_verdict.json')
    with open(out, 'w') as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
