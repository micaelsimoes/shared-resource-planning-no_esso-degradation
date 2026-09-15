"""P5.15 s32 gate - supplementary zero-solve extraction (SolveProfileGuard armed) to discriminate why the Boyd dual test
never passed. Reads data/SRP1/Results/P515S32_run/g_baseline.json (and s31c's for comparison). Write-once output:
data/SRP1/Results/P515S32_run/s32_supplementary.json.

Quantities (definitions preserved here):
  legacy_stop_cycle: first cycle k (k >= 2) with residual_convergence AND objective_convergence AND local_solves_ok,
      i.e. the composite rule in force for s31c (consecutive cycles 1), recorded by production each cycle;
  rms_step_per_entry[c] = s_proximal_part_c / (gamma_c * sqrt(n_c)) - RMS per-entry motion of the TSO copy (V/PF) or of
      the TSO ESS copy (ESS proximal block) in the channel's normalized units; gamma = 1 (case file), n from
      eps_dual = sqrt(n)*eps_abs + eps_rel*||y||, i.e. n = ((eps_dual - eps_rel*||y||)/eps_abs)^2;
  eps_dual_floor_share[c] = sqrt(n)*eps_abs / eps_dual;
  dual_ratio_if_rho_part_only = s_rho_part / eps_dual (= dual_ratio_balance);
  counterfactual pass cycles: cycles where each channel would pass with the proximal part excluded from s;
  drift: sign counts and sums of recourse changes over windows; monotone-run lengths.
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32_run')
OUT = os.path.join(RUN, 's32_supplementary.json')
C = ('v', 'pf', 'ess')
WINDOWS = ((2, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 150))


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s32 supplementary').install()
    try:
        g = json.load(open(os.path.join(RUN, 'g_baseline.json')))
        rows = g['cycle_trajectory']
        params = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')))['admm']
        gamma = params['proximal_regularization']['tso']['gamma']
        eps_abs = rows[0]['boyd_eps_abs']
        eps_rel = rows[0]['boyd_eps_rel']

        legacy = [r['cycle'] for r in rows if r.get('residual_convergence') and r.get('objective_convergence')
                  and r.get('local_solves_ok')]
        legacy_first = legacy[0] if legacy else None
        legacy_block = {
            'rule': 'residual_convergence AND objective_convergence AND local_solves_ok, consecutive 1 (s31c rule)',
            'first_cycle': legacy_first,
            'n_cycles_satisfying': len(legacy),
            'cycles_satisfying_first_20': legacy[:20],
            'gross_at_first_cycle': rows[legacy_first - 1]['gross_operational_cost'] if legacy_first else None,
            'gross_at_150': rows[-1]['gross_operational_cost'],
            'first_residual_convergence_cycle': next((r['cycle'] for r in rows if r.get('residual_convergence')), None),
            'first_objective_convergence_cycle': next((r['cycle'] for r in rows if r.get('objective_convergence')), None),
        }

        per_channel = {}
        for c in C:
            series = []
            for r in rows:
                eps_dual = r[f'boyd_{c}_eps_dual']
                norm_y = r[f'boyd_{c}_norm_y']
                sqrt_n = (eps_dual - eps_rel * norm_y) / eps_abs
                s_prox = r[f's_placeholder'] if False else r[f'boyd_{c}_s_proximal_part']
                series.append({
                    'cycle': r['cycle'],
                    'sqrt_n': sqrt_n,
                    'rms_step_per_entry': (s_prox / (gamma[c] * sqrt_n)) if gamma[c] > 0 and sqrt_n > 0 else None,
                    'eps_dual_floor_share': (sqrt_n * eps_abs) / eps_dual if eps_dual > 0 else None,
                    'norm_y': norm_y,
                    'rho_before': r[f'rho_{c}_before'],
                    'dual_ratio': r[f'boyd_{c}_dual_ratio'],
                    'dual_ratio_rho_part_only': r[f'boyd_{c}_dual_ratio_balance'],
                    'primal_ratio': r[f'boyd_{c}_primal_ratio'],
                    'proximal_share': r[f'boyd_{c}_proximal_share'],
                })
            cf_pass = [e['cycle'] for e in series if e['primal_ratio'] <= 1.0 and e['dual_ratio_rho_part_only'] <= 1.0]
            sample = [e for e in series if e['cycle'] in (1, 10, 25, 50, 75, 100, 125, 150)]
            per_channel[c] = {
                'n_entries': round(series[0]['sqrt_n'] ** 2),
                'rms_step_per_entry_sampled': [{k: e[k] for k in ('cycle', 'rms_step_per_entry', 'norm_y', 'rho_before',
                                                                    'eps_dual_floor_share', 'dual_ratio',
                                                                    'dual_ratio_rho_part_only', 'primal_ratio')}
                                               for e in sample],
                'rms_step_per_entry_terminal_over_eps_abs': (series[-1]['rms_step_per_entry'] / eps_abs)
                if series[-1]['rms_step_per_entry'] is not None else None,
                'counterfactual_channel_pass_cycles_without_proximal_part': {
                    'n': len(cf_pass), 'first': cf_pass[:1], 'last': cf_pass[-5:]},
            }
        cf_all = [r['cycle'] for r in rows
                  if all(r[f'boyd_{c}_primal_ratio'] <= 1.0 and r[f'boyd_{c}_dual_ratio_balance'] <= 1.0 for c in C)]

        changes = [(r['cycle'], r['gross_operational_cost'] - rows[i - 1]['gross_operational_cost'])
                   for i, r in enumerate(rows) if i > 0]
        drift = []
        for lo, hi in WINDOWS:
            sel = [d for k, d in changes if lo <= k <= hi]
            drift.append({'cycles': [lo, hi], 'n_negative': sum(1 for d in sel if d < 0),
                          'n_positive': sum(1 for d in sel if d > 0), 'sum_change': sum(sel),
                          'sum_abs_change': sum(abs(d) for d in sel),
                          'net_over_abs': (sum(sel) / sum(abs(d) for d in sel)) if sel else None})
        v_period3 = all(rows[k - 1]['rho_v_action'] == rows[k - 4]['rho_v_action'] for k in range(51, 151))

        out = {
            'stage': 'P5.15 s32 gate - supplementary zero-solve extraction',
            'source': os.path.relpath(os.path.join(RUN, 'g_baseline.json'), REPO),
            'instance': g.get('instance'),
            'eps_abs': eps_abs, 'eps_rel': eps_rel, 'gamma_tso': gamma,
            'legacy_composite_rule_on_s32': legacy_block,
            'per_channel': per_channel,
            'counterfactual_all_channels_pass_without_proximal_part': {'n': len(cf_all), 'first': cf_all[:1],
                                                                        'last': cf_all[-5:]},
            'recourse_drift_windows': drift,
            'rho_v_action_period3_from_cycle_48': v_period3,
            'efc_note': 'EFC/day max 0.1038 (threshold 1.4612) printed by the harness at exit',
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    with open(OUT, 'w') as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
