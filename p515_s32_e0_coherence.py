"""P5.15 s32 gate - E0 zero-solve verification of the drift reading (SolveProfileGuard armed).

Per channel c and cycle k, from data/SRP1/Results/P515S32_run/g_baseline.json:
  dz_norm(k)    = s_proximal_part / gamma_c         (norm of the per-cycle TSO-copy step; gamma = 1 on every channel)
  coherence(k)  = (||z||(k) - ||z||(k-1)) / dz_norm(k)   (1 = every step in the direction of the norm's growth;
                                                          ~0 = zero-mean motion)
  lag_ratio(k)  = r / dz_norm                        (primal residual relative to one TSO step)
  ||z||: V/PF norm_z (TSO copy); ESS norm_z (weighted consensus z).
Around each rho_ess decrease cycle d: ESS dz_norm at d-2..d+3.
Write-once output: data/SRP1/Results/P515S32_run/s32_e0_coherence.json
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32_run')
OUT = os.path.join(RUN, 's32_e0_coherence.json')
C = ('v', 'pf', 'ess')
WIN = ((2, 25), (26, 50), (51, 75), (76, 100), (101, 125), (126, 150))


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s32 E0').install()
    try:
        rows = json.load(open(os.path.join(RUN, 'g_baseline.json')))['cycle_trajectory']
        gamma = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')))['admm']['proximal_regularization']['tso']['gamma']
        out = {'stage': 'P5.15 s32 E0 coherence / lag verification', 'gamma_tso': gamma, 'channels': {}}
        for c in C:
            series = []
            for i, r in enumerate(rows):
                dz = r[f'boyd_{c}_s_proximal_part'] / gamma[c]
                nz = r[f'boyd_{c}_norm_z']
                coh = ((nz - rows[i - 1][f'boyd_{c}_norm_z']) / dz) if i > 0 and dz > 0 else None
                series.append({'cycle': r['cycle'], 'norm_z': nz, 'norm_x': r[f'boyd_{c}_norm_x'], 'dz_norm': dz,
                               'coherence': coh, 'r': r[f'boyd_{c}_r'],
                               'lag_ratio': (r[f'boyd_{c}_r'] / dz) if dz > 0 else None, 'rho': r[f'rho_{c}_before']})
            windows = []
            for lo, hi in WIN:
                sel = [e for e in series if lo <= e['cycle'] <= hi]
                cohs = [e['coherence'] for e in sel if e['coherence'] is not None]
                lags = [e['lag_ratio'] for e in sel if e['lag_ratio'] is not None]
                windows.append({'cycles': [lo, hi],
                                'norm_z_start_end': [sel[0]['norm_z'], sel[-1]['norm_z']],
                                'norm_z_mean_increment': (sel[-1]['norm_z'] - sel[0]['norm_z']) / (len(sel) - 1),
                                'dz_norm_min_max': [min(e['dz_norm'] for e in sel), max(e['dz_norm'] for e in sel)],
                                'coherence_mean': sum(cohs) / len(cohs), 'coherence_min': min(cohs), 'coherence_max': max(cohs),
                                'n_positive_norm_increments': sum(1 for x in cohs if x > 0),
                                'lag_ratio_mean': sum(lags) / len(lags), 'lag_ratio_max': max(lags)})
            out['channels'][c] = {'windows': windows,
                                  'norm_z_c1_c150': [series[0]['norm_z'], series[-1]['norm_z']],
                                  'norm_x_c1_c150': [series[0]['norm_x'], series[-1]['norm_x']],
                                  'lag_ratio_max_all_cycles_ge_10': max(e['lag_ratio'] for e in series if e['cycle'] >= 10)}
        dec = [r['cycle'] for r in rows if r['rho_ess_action'] == 'decreased']
        out['ess_dz_around_rho_decreases'] = [
            {'decrease_cycle': d, 'dz_norm_d-2..d+3': [rows[k - 1]['boyd_ess_s_proximal_part'] / gamma['ess']
                                                      for k in range(d - 2, d + 4) if 1 <= k <= len(rows)]}
            for d in dec]
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    json.dump(out, open(OUT, 'w'), indent=1)
    for c in C:
        ch = out['channels'][c]
        print(c, 'norm_z c1->c150', [round(x, 4) for x in ch['norm_z_c1_c150']], 'norm_x', [round(x, 4) for x in ch['norm_x_c1_c150']], 'lag max(>=10)', round(ch['lag_ratio_max_all_cycles_ge_10'], 3))
        for w in ch['windows']:
            print('  ', w['cycles'], 'dnorm/cyc %.3g' % w['norm_z_mean_increment'], 'dz %.3g-%.3g' % tuple(w['dz_norm_min_max']),
                  'coh mean %.2f [%.2f,%.2f]' % (w['coherence_mean'], w['coherence_min'], w['coherence_max']),
                  'pos %d' % w['n_positive_norm_increments'], 'lag mean %.2f max %.2f' % (w['lag_ratio_mean'], w['lag_ratio_max']))
    for e in out['ess_dz_around_rho_decreases']:
        print('ess dec', e['decrease_cycle'], ['%.3g' % x for x in e['dz_norm_d-2..d+3']])
    print('guard', out['solve_profile_guard'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
