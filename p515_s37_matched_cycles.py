"""P5.15 Addendum 19 - rho_ess arms: matched-cycle channel comparison against run 1 (s35ref). Zero solves.

Reads the per-cycle trajectories (`cycle_trajectory` in each run's g_*.json) of run 1 and both s37 arms and
reports, at matched cycles, every Boyd channel primal/dual ratio, the PF dual-residual rho/proximal parts,
EFC/day max, rho_pf and its action; and per run the first cycle each channel (and the PF dual test alone)
passes, and the number of passing cycles per channel.

Definitions (all read verbatim from the trajectory, nothing recomputed):
  channel ratio      = boyd_<ch>_<primal|dual>_ratio   (r/eps_pri, s/eps_dual; pass iff <= 1)
  channel pass       = boyd_<ch>_channel_pass           (primal and dual pass in the same cycle)
  first pass         = smallest cycle with channel pass True (None if never within the run)
  PF-dual spread     = (max - min) / min of boyd_pf_dual_ratio across the three runs at a matched cycle

Writes data/SRP1/Results/P515S37/matched_cycles_vs_run1.json (write-once). SolveProfileGuard armed, 0 solves.
"""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
RUNS = {
    'run1_s35ref_rho_ess_balanced': os.path.join(RES, 'P515S35_REF_run', 'g_baseline.json'),
    's37_rho0p01': os.path.join(RES, 'P515S37_RHO0P01_run', 'g_s37_rho0p01.json'),
    's37_rho0p001': os.path.join(RES, 'P515S37_RHO0P001_run', 'g_s37_rho0p001.json'),
}
MATCH = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 226, 300, 400, 477)
CH = ('v', 'pf', 'ess')
FIELDS = ['boyd_v_primal_ratio', 'boyd_v_dual_ratio', 'boyd_pf_primal_ratio', 'boyd_pf_dual_ratio',
          'boyd_ess_primal_ratio', 'boyd_ess_dual_ratio', 'boyd_pf_s_rho_part', 'boyd_pf_s_proximal_part',
          'boyd_v_channel_pass', 'boyd_pf_channel_pass', 'boyd_ess_channel_pass', 'boyd_pf_dual_pass',
          'efc_per_day_max', 'rho_pf_after', 'rho_pf_action', 'rho_ess_after', 'gross_operational_cost']
OUT = os.path.join(RES, 'P515S37', 'matched_cycles_vs_run1.json')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s37 matched cycles').install()
    try:
        traj = {}
        for name, path in RUNS.items():
            with open(path) as handle:
                traj[name] = {e['cycle']: e for e in json.load(handle)['cycle_trajectory']}
        per_run = {}
        for name, t in traj.items():
            cycles = sorted(t)
            per_run[name] = {
                'source': os.path.relpath(RUNS[name], REPO), 'sha256': _sha256(RUNS[name]),
                'cycles_run': len(cycles),
                'first_channel_pass': {ch: next((c for c in cycles if t[c].get(f'boyd_{ch}_channel_pass')), None)
                                       for ch in CH},
                'first_pf_dual_pass': next((c for c in cycles if t[c].get('boyd_pf_dual_pass')), None),
                'n_channel_pass_cycles': {ch: sum(1 for c in cycles if t[c].get(f'boyd_{ch}_channel_pass'))
                                          for ch in CH},
            }
        matched = []
        for c in MATCH:
            row = {'cycle': c}
            for name, t in traj.items():
                if c in t:
                    row[name] = {f: t[c].get(f) for f in FIELDS}
            pf_d = [row[n]['boyd_pf_dual_ratio'] for n in traj if n in row and row[n]['boyd_pf_dual_ratio']]
            if len(pf_d) == 3:
                row['pf_dual_ratio_spread_across_runs'] = (max(pf_d) - min(pf_d)) / min(pf_d)
            matched.append(row)
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out = {'stage': 'P5.15 Addendum 19 - rho_ess arms, matched-cycle comparison with run 1',
           'definitions': __doc__, 'per_run': per_run, 'matched_cycles': matched,
           'solve_profile_guard': {'counts': dict(guard.counts), 'verify_failures': failures}}
    if failures:
        raise RuntimeError(failures)
    with open(OUT, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(json.dumps(per_run, indent=1))
    for row in matched:
        print(row['cycle'], {n: (round(row[n]['boyd_pf_dual_ratio'], 3) if n in row else None) for n in traj},
              'spread', round(row.get('pf_dual_ratio_spread_across_runs', float('nan')), 4))
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
