"""P5.6-A6 -- controlled midpoint-versus-DSO anchor comparison on the base candidate."""
import json, os, sys
from datetime import datetime, timezone
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import p56a_candidates as C  # noqa: E402
import p56a_oracle as O  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

def main():
    try:
        provenance, planning = gate('P5.6-A6 anchor control', O.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A6c] ABORTED\n{error}'); sys.exit(1)
    x = C.base_vector(planning)
    report = {'stage': 'P5.6-A6 anchor control', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'anchors': {}}
    for anchor in ('midpoint', 'dso'):
        r = O.evaluate_planning_candidate(x, start_policy=O.START_COLD,
                                          eval_id=f'a6c_{anchor}',
                                          use_cache=False, interface_anchor=anchor)
        report['anchors'][anchor] = {
            'status': r['status'], 'wall_clock_s': r.get('wall_clock_s'),
            'total_objective': r.get('total_objective'),
            'net_operational_recourse': r.get('net_operational_recourse'),
            'max_coordinated_residual':
                (r.get('coordination_residuals') or {}).get('max_coordinated'),
            'error': r.get('error'), 'failed_stage': r.get('failed_stage')}
        print(f"[A6c] {anchor:10s} status={r['status']:16s} "
              f"Q={r.get('total_objective')} t={r['wall_clock_s']:.1f}s", flush=True)
    a, b = report['anchors']['midpoint'], report['anchors']['dso']
    if a['status'] == 'VALID' and b['status'] == 'VALID':
        report['dso_anchor_cost'] = b['total_objective'] - a['total_objective']
        print(f"[A6c] DSO anchor costs {report['dso_anchor_cost']:.2f} "
              f"({report['dso_anchor_cost'] / 7.164e5:.3f} x tol_cut)")
    out = os.path.join(O.OUT_DIR, 'p56a_a6_anchor_control.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[A6c] report -> {out}')

if __name__ == '__main__':
    main()
