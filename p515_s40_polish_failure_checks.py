"""P5.15 Step 3.5 -- zero-solve checks of two hypotheses for the polish failures (Planner).

The v2 polish-gap run (`data/SRP1/Results/P515S40/polish_gap_v2/`) reproduced arm D bitwise and then
failed to polish 17 of 48 blocks at fixed consensus: all 12 TSO blocks ('Converged to a point of local
infeasibility') and 5 DSO blocks. The polish fixes each interface quantity at the MIDPOINT of the two
sides' own achieved values (p56a_oracle design). This script tests, from the committed terminal
artifacts of that run, whether the midpoint lands outside a bound that is active at the certified point.

H-V (voltage): for every interface entry in `interface_voltage_terminal.json` (TSO and DSO side values
  in pu, with the TSO node's bounds), midpoint m = (tso_pu + dso_pu) / 2. Count entries with
  m > v_max_pu or m < v_min_pu. Also report entries with either side within 1e-6 pu of v_max, and the
  largest |tso_pu - dso_pu|.

H-S (interface rating): from the LAST row of `pf_entry_stride_<label>.jsonl` (the certified cycle), for
  every (node, year, day, period), midpoint P and Q are ((x_dso + z_tso_current) / 2) per power type, in
  the units the stride records (the same units as `interface_rating`, MVA). S = hypot(Pm, Qm);
  utilization u = S / interface_rating. Count entries with u > 1, grouped by (node, year, day), and the
  number of distinct TSO (year, day) blocks they touch.

Zero solves: SolveProfileGuard armed for the whole script, verify(0). Write-once output.
"""
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'polish_gap_v2')
OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'polish_failure_checks.json')


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s40 polish failure checks').install()
    try:
        with open(os.path.join(RUN, 'interface_voltage_terminal.json')) as h:
            ventries = json.load(h)['entries']
        outside, near = [], []
        for x in ventries:
            m = (x['tso_pu'] + x['dso_pu']) / 2.0
            if m > x['v_max_pu'] or m < x['v_min_pu']:
                outside.append(x)
            if x['v_max_pu'] - max(x['tso_pu'], x['dso_pu']) < 1e-6:
                near.append(x)
        h_v = {
            'n_entries': len(ventries),
            'n_midpoint_outside_bounds': len(outside),
            'n_either_side_within_1e-6_of_vmax': len(near),
            'near_vmax_by_node_year_day': sorted(
                ([list(k), v] for k, v in Counter((x['node_id'], x['year'], x['day']) for x in near).items())),
            'max_abs_tso_minus_dso_pu': max(abs(x['tso_pu'] - x['dso_pu']) for x in ventries),
            'verdict': 'falsified' if not outside else 'supported',
        }

        stride = glob.glob(os.path.join(RUN, 'pf_entry_stride_*.jsonl'))[0]
        last = None
        with open(stride) as h:
            for line in h:
                last = line
        row = json.loads(last)
        by = defaultdict(dict)
        for e in row['entries']:
            by[(e['node_id'], e['year'], e['day'], e['period'])][e['power_type']] = e
        over = Counter()
        max_u = defaultdict(float)
        for k, v in by.items():
            if 'p' not in v or 'q' not in v:
                continue
            pm = (v['p']['x_dso'] + v['p']['z_tso_current']) / 2.0
            qm = (v['q']['x_dso'] + v['q']['z_tso_current']) / 2.0
            u = math.hypot(pm, qm) / v['p']['interface_rating']
            key = (k[0], k[1], k[2])
            max_u[key] = max(max_u[key], u)
            if u > 1.0:
                over[key] += 1
        h_s = {
            'stride_path': os.path.relpath(stride, REPO),
            'certified_cycle': row['cycle'],
            'midpoint_over_rating_by_node_year_day': sorted([list(k), v] for k, v in over.items()),
            'n_tso_year_day_blocks_touched': len({(y, d) for (_n, y, d) in over}),
            'max_midpoint_utilization_by_node': {
                str(n): max(u for (nn, _y, _d), u in max_u.items() if nn == n)
                for n in sorted({k[0] for k in max_u})},
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(failures)
    out = {'stage': 'P5.15 Step 3.5 polish-failure hypothesis checks (zero solves)',
           'run': os.path.relpath(RUN, REPO), 'H_V_voltage_midpoint': h_v,
           'H_S_interface_rating_midpoint': h_s,
           'solve_profile_guard': {'counts': dict(guard.counts), 'verify_failures': failures},
           'definitions': __doc__}
    with open(OUT, 'w') as h:
        json.dump(out, h, indent=1, default=str)
    print(json.dumps({'H_V': {k: h_v[k] for k in ('n_midpoint_outside_bounds', 'max_abs_tso_minus_dso_pu', 'verdict')},
                      'H_S': {k: h_s[k] for k in ('midpoint_over_rating_by_node_year_day', 'n_tso_year_day_blocks_touched')}},
                     indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
