"""
Track B — configuration reconciliation. Zero solves, ENFORCED. Artifacts and git only.

Answers three questions from evidence alone:
  1. which configuration actually governed each preserved stage
  2. the mechanism of the divergence from data/SRP1/SRP1_params.json
  3. what a plain production run would do today

Tolerances are derived from each stage's own numbers where possible
(tolerance = residual / ratio), rather than assumed from the case file, because the
case file's tolerances changed several times over the period the evidence spans.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514B')


def sh(args):
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def load(path):
    with open(os.path.join(REPO, path)) as handle:
        return json.load(handle)


def derive_tol(residual, ratio):
    return (residual / ratio) if (residual is not None and ratio) else None


def main():
    guard = SolveProfileGuard(permitted=(), label='Track B (zero solves)').install()
    try:
        stages = {}

        # --- P5.9-B (warm, adaptive A/B) ---
        p59 = load('data/SRP1/Results/P59/p59_b_adaptive.json')
        g = p59['arms']['adaptive_on']['generations'][0]
        c1 = (g.get('cycle_detail') or [{}])[0]
        stages['P5.9-B'] = {
            'date': p59.get('timestamp_utc', '')[:10],
            'rho_requested': p59['arms']['adaptive_on'].get('rho_requested'),
            'adaptive_penalty': {'arm_on': True, 'arm_off': False},
            'num_max_iters': 'not serialized',
            'tol_stationarity_pf_derived': derive_tol(c1.get('dual_pf_mean'), c1.get('dual_pf_mean_ratio')),
            'tol_consensus_pf_derived': derive_tol(c1.get('primal_pf'), c1.get('primal_pf_ratio')),
            'objective_tolerance_observed': g.get('objective_tolerance'),
            'source': 'data/SRP1/Results/P59/p59_b_adaptive.json',
        }

        # --- P5.10-B (warm, fixed-rho sweep) ---
        p510 = load('data/SRP1/Results/P510/p510_b_fixedrho_pf300.json')
        row = p510['rows'][0]
        c1 = (row.get('cycle_detail') or [{}])[0]
        stages['P5.10-B'] = {
            'date': p510.get('timestamp_utc', '')[:10],
            'rho_requested': {'v': 1.5, 'pf': row.get('rho_pf'), 'ess': 1.0},
            'adaptive_penalty': p510.get('adaptive_penalty'),
            'config_label': row.get('config_label'),
            'num_max_iters': 'not serialized',
            'tol_stationarity_pf_derived': derive_tol(c1.get('dual_pf_mean'), c1.get('dual_pf_mean_ratio')),
            'tol_consensus_pf_derived': derive_tol(c1.get('primal_pf'), c1.get('primal_pf_ratio')),
            'objective_tolerance_observed': row.get('objective_tolerance'),
            'source': 'data/SRP1/Results/P510/p510_b_fixedrho_pf300.json',
        }

        # --- P5.12-R (cold, capped) ---
        r = load('data/SRP1/Results/P512R/run.json')
        stages['P5.12-R'] = {
            'date': '2026-09-11',
            'rho_requested': r['configuration']['rho'],
            'adaptive_penalty': r['configuration']['adaptive_penalty'],
            'num_max_iters': r['configuration']['cap'],
            'source': 'data/SRP1/Results/P512R/run.json (configuration block)',
        }

        # --- AB1 cells and X22 cells (this project's own runs) ---
        for label, path in (('AB1 control', 'data/SRP1/Results/P514AB1/ab1_control.json'),
                            ('AB1 treatment', 'data/SRP1/Results/P514AB1/ab1_treatment.json')):
            d = load(path)
            c1 = d['cycles'][0]
            stages[label] = {
                'date': d['timestamp_utc'][:10],
                'rho_requested': d['rho_requested'],
                'adaptive_penalty': d['adaptive_penalty'],
                'num_max_iters': d['cap'],
                'tol_stationarity_pf_derived': derive_tol(c1.get('dual_pf_mean'), c1.get('dual_pf_mean_ratio')),
                'tol_consensus_pf_derived': derive_tol(c1.get('primal_pf'), c1.get('primal_pf_ratio')),
                'objective_tolerance_observed': d['cycles'][-1].get('objective_tolerance'),
                'penalty_update': d.get('penalty_update_in_force'),
                'source': path,
            }
        for label, path in (('X22 warm_fixed', 'data/SRP1/Results/P514X22/x22_warm_fixed.json'),
                            ('X22 warm_adaptive', 'data/SRP1/Results/P514X22/x22_warm_adaptive.json')):
            d = load(path)
            c1 = d['cycle_detail'][0]
            stages[label] = {
                'date': d['timestamp_utc'][:10],
                'rho_requested': {'v': d['config']['rho_v'], 'pf': d['config']['rho_pf'],
                                  'ess': d['config']['rho_ess']},
                'adaptive_penalty': d['adaptive_penalty'],
                'num_max_iters': d['cap'],
                'tol_stationarity_pf_derived': derive_tol(c1.get('dual_pf_mean'), c1.get('dual_pf_mean_ratio')),
                'tol_consensus_pf_derived': derive_tol(c1.get('primal_pf'), c1.get('primal_pf_ratio')),
                'objective_tolerance_observed': d['terminating_criterion'].get('objective_tolerance'),
                'source': path,
            }

        # --- the case file on disk today ---
        case = load('data/SRP1/SRP1_params.json')['admm']
        today = {
            'rho': case['rho'], 'adaptive_penalty': case['adaptive_penalty'],
            'num_max_iters': case['num_max_iters'], 'tol': case['tol'],
            'proximal_regularization': case.get('proximal_regularization'),
            'penalty_update': case.get('penalty_update'),
        }

        # --- mechanism: history of the file, and the programmatic overrides ---
        history = []
        for commit in sh(['git', '--no-optional-locks', 'log', '--full-history',
                          '--format=%H', '--reverse', '--',
                          'data/SRP1/SRP1_params.json']).split():
            meta = sh(['git', '--no-optional-locks', 'log', '-1', '--format=%h|%ad|%s',
                       '--date=short', commit])
            try:
                payload = json.loads(sh(['git', '--no-optional-locks', 'show',
                                         f'{commit}:data/SRP1/SRP1_params.json']))
            except Exception:
                continue
            admm = payload.get('admm', {})
            rho = admm.get('rho', {})
            first = lambda x: (list(x.values())[0] if isinstance(x, dict) and x else x)  # noqa: E731
            history.append({'commit': meta.split('|')[0], 'date': meta.split('|')[1],
                            'subject': meta.split('|', 2)[2],
                            'rho_v': first(rho.get('v')), 'rho_pf': first(rho.get('pf')),
                            'adaptive_penalty': admm.get('adaptive_penalty'),
                            'num_max_iters': admm.get('num_max_iters'),
                            'tol': admm.get('tol')})
        overrides = sh(['grep', '-rln', 'apply_rho_to_params', '--include=*.py', '.']).splitlines()
    finally:
        guard.uninstall()

    report = {
        'stage': 'Track B — configuration reconciliation',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile': {'declared': 0, 'observed': dict(guard.counts),
                          'failures': guard.verify(0)},
        'q1_governing_configuration_per_stage': stages,
        'q2_mechanism': {
            'case_file_history': history,
            'finding': ('data/SRP1/SRP1_params.json has carried rho.{v,pf,ess} = 1.0 and '
                        'adaptive_penalty = true CONTINUOUSLY since 784346d7 (2025-12-15), '
                        'which PRE-DATES every preserved stage. The file was therefore NOT '
                        'edited after those runs. The divergence is PROGRAMMATIC OVERRIDE: '
                        'the harnesses call p59_rho.apply_rho_to_params and '
                        'p59_rho.set_adaptive_penalty on the in-memory planning problem.'),
            'harnesses_that_override': overrides,
            'what_did_change_in_the_file': ('the TOLERANCES moved repeatedly — stationarity '
                                            '0.05 -> 0.001 -> 0.01 and objective rel '
                                            '0.0005 -> 0.005 -> 0.001 -> 0.01 -> 0.001 — and '
                                            'num_max_iters 50 -> 25 on 2026-09-04. Tolerances '
                                            'are NOT overridden by the harnesses, so each stage '
                                            'ran with whatever the file held at its commit; the '
                                            'per-stage derived values above are the authority.'),
        },
        'q3_plain_production_run_today': {
            'effective_configuration': today,
            'statement': ('A plain production run from this case file today would use '
                          'rho = 1.0 on every family, adaptive_penalty = True, num_max_iters = 25, '
                          'stationarity tolerances 0.01, objective (abs 1000, rel 0.001), '
                          'TSO proximal on and DSO proximal off, and C3 active. '
                          'That configuration matches NO preserved stage: every stage in the '
                          'evidence base used an overridden rho and, except P5.9-B\'s treatment '
                          'arm, adaptive_penalty = False.'),
        },
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'b_config_reconciliation.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('q1_governing_configuration_per_stage', 'q2_mechanism')},
                     indent=1)[:1400])
    print(f'\nwritten to {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
