"""
Stage P5.6-A6 (remedy test) -- does a DSO-anchored interface remove the
POLISH_FAILURE?

The only failure in the benchmark was one distribution block (DSO9 2030 Spring)
that could not reach the MIDPOINT interface value.  A midpoint asks both sides to
move; the DSO's own achieved value asks only the transmission side to move, and
the transmission side has interface flexibility for exactly that purpose.  This
tests that alternative on the failing candidate and, as a control, on the base
candidate.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56a_a6_anchor.py
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


def summarise(result):
    out = {'status': result['status'], 'wall_clock_s': result.get('wall_clock_s'),
           'interface_anchor': result.get('interface_anchor')}
    if result['status'] == O.STATUS_VALID:
        out.update({
            'total_objective': result['total_objective'],
            'net_operational_recourse': result['net_operational_recourse'],
            'max_coordinated_residual':
                result['coordination_residuals']['max_coordinated'],
            'esso_max_violation': result['esso_audit']['max_violation'],
            'network_max_violation': result['network_audit']['max_violation'],
            'network_h1_violation':
                result['network_audit']['max_h1_complementarity_violation'],
        })
    elif result.get('polish'):
        out['failed_blocks'] = [f"{b['agent']}|{b['year']}|{b['day']}"
                                for b in result['polish']['blocks']
                                if not b['solved']]
    return out


def main():
    try:
        provenance, planning = gate('P5.6-A6 interface anchor', O.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A6a] ABORTED\n{error}')
        sys.exit(1)

    base = C.base_vector(planning)
    # The base control is run separately by p56a_a6_anchor_control.py so that a
    # crash there cannot cost the failing candidate's evidence.
    cases = [('s+e|node9|2025|-10% (the failing candidate)',
              C.perturbed(base, 'se', 9, 2025, -0.10))]

    report = {'stage': 'P5.6-A6 interface anchor', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'cases': []}
    for label, x in cases:
        print(f'\n[A6a] === {label}', flush=True)
        entry = {'label': label, 'anchors': {}}
        for anchor in ('midpoint', 'dso'):
            result = O.evaluate_planning_candidate(
                x, start_policy=O.START_COLD,
                eval_id=f'a6a_{anchor}_{label.split("|")[0]}',
                use_cache=False, interface_anchor=anchor)
            entry['anchors'][anchor] = summarise(result)
            print(f"      {anchor:10s} status={result['status']:16s} "
                  f"Q={result.get('total_objective')} "
                  f"t={result['wall_clock_s']:.1f}s", flush=True)
            if entry['anchors'][anchor].get('failed_blocks'):
                print(f"                 failed: "
                      f"{entry['anchors'][anchor]['failed_blocks']}")
        report['cases'].append(entry)
        # write after every case: a crash in a later case must not destroy the
        # evidence already gathered
        out = os.path.join(O.OUT_DIR, 'p56a_a6_anchor.json')
        with open(out, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    print(f"\n[A6a] report -> {os.path.join(O.OUT_DIR, 'p56a_a6_anchor.json')}")


if __name__ == '__main__':
    main()
