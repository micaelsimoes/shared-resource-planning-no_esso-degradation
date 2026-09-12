"""
P5.14-G — (a) retroactive candidate recovery for AB1, (b) the rho-free trajectory
comparison AB1 should have had.

Zero solves, ENFORCED in blocking mode. Existing artifacts plus one deterministic
re-derivation of the bootstrap candidate.

(a) Neither ab1_*.json nor the frozen design records the problem instance (eighth
    evidence rule). The candidate is nonetheless recoverable: both AB1 and P5.9-B build
    it from the SAME deterministic constructor -- AB1 via
    srp._build_positive_bootstrap_candidate, P5.9-B via BC.population -> A.base_vector,
    which calls that same function (p56a_candidates.py:39-43). Re-deriving it and
    cross-checking against the candidate P5.12-R recorded verbatim settles the identity.

(b) Both arms' per-cycle dual_pf_mean and rho are stored, so |dz|/base = dual/rho can be
    evaluated against the SINGLE fixed physical threshold tol/rho_ref. This is the
    rho-free comparison; it costs nothing.
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P514AB1')
TOL_STATIONARITY_PF = 0.01
RHO_REF = 300.0
THRESHOLD = TOL_STATIONARITY_PF / RHO_REF          # 3.333e-05, the physical standard


def recover_candidate():
    import p56a_oracle as O
    import shared_resources_planning as srp
    with redirect_stdout(io.StringIO()):
        planning = O.fresh_planning('p514g_candidate_recovery')
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    return candidate


def main():
    guard = SolveProfileGuard(permitted=(), label='P5.14-G (zero solves)').install()
    try:
        candidate = recover_candidate()
        with open(os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P512R', 'run.json')) as handle:
            r_candidate = json.load(handle)['configuration']['candidate']['investment']

        inv = candidate['investment'] if 'investment' in candidate else candidate
        normalized = {str(node): {str(year): {k: float(v) for k, v in value.items()}
                                  for year, value in years.items()}
                      for node, years in inv.items()}
        r_normalized = {str(node): {str(year): {k: float(v) for k, v in value.items()}
                                    for year, value in years.items()}
                        for node, years in r_candidate.items()}
        identical = normalized == r_normalized

        arms = {}
        for arm in ('treatment', 'control'):
            with open(os.path.join(OUT, f'ab1_{arm}.json')) as handle:
                data = json.load(handle)
            rows = []
            rho_by_cycle = {r['cycle']: (r['rho_pf_before'] or r['rho_pf_after'])
                            for r in data['rho_trace']}
            for cycle in data['cycles']:
                dual = cycle.get('dual_pf_mean')
                rho = rho_by_cycle.get(cycle['cycle'])
                if dual is None or not rho:
                    continue
                step = dual / rho
                rows.append({'cycle': cycle['cycle'], 'rho_pf': rho, 'dual_pf_mean': dual,
                             'step_dz_over_base': step,
                             'multiple_of_physical_threshold': step / THRESHOLD,
                             'primal_pf': cycle.get('primal_pf'),
                             'recourse': cycle.get('recourse')})
            best = min(rows, key=lambda r: r['step_dz_over_base'])
            arms[arm] = {
                'cycles': len(rows),
                'first_step': rows[0]['step_dz_over_base'],
                'final_step': rows[-1]['step_dz_over_base'],
                'final_multiple_of_threshold': rows[-1]['multiple_of_physical_threshold'],
                'best_step': best['step_dz_over_base'],
                'best_step_at_cycle': best['cycle'],
                'reached_physical_threshold': any(
                    r['step_dz_over_base'] <= THRESHOLD for r in rows),
                'trajectory': rows,
            }
    finally:
        guard.uninstall()

    treatment, control = arms['treatment'], arms['control']
    report = {
        'stage': 'P5.14-G', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile': {'declared': 0, 'observed': dict(guard.counts),
                          'failures': guard.verify(0)},
        'candidate_recovery': {
            'defect': 'AB1 recorded its settings exhaustively and not its problem instance',
            'recoverable': True,
            'method': 'deterministic re-derivation of srp._build_positive_bootstrap_candidate, '
                      'the same constructor P5.9-B reaches through BC.population -> '
                      'p56a_candidates.base_vector (:39-43)',
            'cross_check_against_p512r_recorded_candidate': identical,
            'candidate_investment': normalized,
        },
        'rho_free_comparison': {
            'physical_threshold_dz_over_base': THRESHOLD,
            'definition': 'step = dual_pf_mean / rho_pf, i.e. the mean interface increment '
                          'with rho divided out; the single fixed standard is tol/rho_ref '
                          f'= {TOL_STATIONARITY_PF}/{RHO_REF}',
            'treatment': {k: v for k, v in treatment.items() if k != 'trajectory'},
            'control': {k: v for k, v in control.items() if k != 'trajectory'},
            'neither_reached_threshold': not (treatment['reached_physical_threshold']
                                              or control['reached_physical_threshold']),
            'which_approaches_faster': (
                'control' if control['best_step'] < treatment['best_step'] else 'treatment'),
            'ratio_treatment_over_control_at_best': treatment['best_step'] / control['best_step'],
        },
        'trajectories': {'treatment': treatment['trajectory'], 'control': control['trajectory']},
    }
    out = os.path.join(OUT, 'p514g_candidate_and_rhofree.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1)
    printable = {k: v for k, v in report.items() if k != 'trajectories'}
    printable['candidate_recovery'] = {k: v for k, v in report['candidate_recovery'].items()
                                       if k != 'candidate_investment'}
    print(json.dumps(printable, indent=1)[:2200])
    print(f'\nwritten to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
