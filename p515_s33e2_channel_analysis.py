"""P5.15 Step 3.2 E2 gate (s33e2) - per-channel motion analysis (zero solves, SolveProfileGuard armed).

For each channel c and cycle k, from g_baseline.json (s33e2 and, for comparison, s32):
  dz_norm(k)   = s_rho_part / rho_in_force          (norm of the per-cycle consensus step; gamma = tau*rho here, so
                                                     s_proximal_part/gamma gives the same quantity for V/PF)
  rms_step(k)  = dz_norm / sqrt(n)                  (per-entry step in the channel's normalized units; n from the
                                                     eps_dual identity n = ((eps_dual - eps_rel*||y||)/eps_abs)^2)
  coherence(k) = (||z||(k) - ||z||(k-1)) / dz_norm   (1 = every step in the direction of the norm's growth)
  dual_ratio decay: ratio over the last 50 cycles and the implied cycles to reach 1.0 at that geometric rate.
Also: per-entry step against eps_abs (1e-5) and against the E4 measured floor delta_c.
Write-once output: data/SRP1/Results/P515S33_E2_run/s33e2_channel_analysis.json
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
RUN = os.path.join(RES, 'P515S33_E2_run')
S32 = os.path.join(RES, 'P515S32_run', 'g_baseline.json')
E4 = os.path.join(RES, 'P515S33', 'E4', 'e4_noise_floor_results.json')
OUT = os.path.join(RUN, 's33e2_channel_analysis.json')
C = ('v', 'pf', 'ess')
WIN = ((3, 30), (31, 60), (61, 90), (91, 120), (121, 150))


def series(rows, c, eps_abs, eps_rel):
    out = []
    for i, r in enumerate(rows):
        rho = r[f'rho_{c}_before']
        s_rho = r[f'boyd_{c}_s_rho_part']
        dz = (s_rho / rho) if rho else None
        nz = r[f'boyd_{c}_norm_z']
        sqrt_n = (r[f'boyd_{c}_eps_dual'] - eps_rel * r[f'boyd_{c}_norm_y']) / eps_abs
        out.append({
            'cycle': r['cycle'], 'rho': rho, 'gamma': r.get(f'gamma_{c}_before'), 'norm_z': nz,
            'dz_norm': dz, 'sqrt_n': sqrt_n,
            'rms_step_per_entry': (dz / sqrt_n) if dz is not None and sqrt_n > 0 else None,
            'coherence': ((nz - rows[i - 1][f'boyd_{c}_norm_z']) / dz) if i > 0 and dz else None,
            'dual_ratio': r[f'boyd_{c}_dual_ratio'], 'primal_ratio': r[f'boyd_{c}_primal_ratio'],
            'proximal_share': r[f'boyd_{c}_proximal_share'],
        })
    return out


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s33e2 channel analysis').install()
    try:
        g = json.load(open(os.path.join(RUN, 'g_baseline.json')))
        rows = g['cycle_trajectory']
        s32_rows = json.load(open(S32))['cycle_trajectory']
        e4 = json.load(open(E4))['derivation']
        eps_abs, eps_rel = rows[0]['boyd_eps_abs'], rows[0]['boyd_eps_rel']
        out = {'stage': 'P5.15 s33e2 per-channel motion analysis', 'eps_abs': eps_abs, 'eps_rel': eps_rel,
               'e4_delta_c': e4['delta_c'], 'channels': {}}
        for c in C:
            s = series(rows, c, eps_abs, eps_rel)
            s32s = series(s32_rows, c, eps_abs, eps_rel)
            windows = []
            for lo, hi in WIN:
                sel = [e for e in s if lo <= e['cycle'] <= hi]
                cohs = [e['coherence'] for e in sel if e['coherence'] is not None]
                windows.append({
                    'cycles': [lo, hi],
                    'norm_z_start_end': [sel[0]['norm_z'], sel[-1]['norm_z']],
                    'norm_z_increment_per_cycle': (sel[-1]['norm_z'] - sel[0]['norm_z']) / (len(sel) - 1),
                    'rms_step_per_entry_start_end': [sel[0]['rms_step_per_entry'], sel[-1]['rms_step_per_entry']],
                    'coherence_mean': (sum(cohs) / len(cohs)) if cohs else None,
                    'dual_ratio_start_end': [sel[0]['dual_ratio'], sel[-1]['dual_ratio']],
                })
            d100, d150 = s[99]['dual_ratio'], s[149]['dual_ratio']
            decay = (d150 / d100) if d100 else None
            if decay and 0 < decay < 1 and d150 > 1:
                cycles_to_one = math.log(1.0 / d150) / math.log(decay) * 50.0
            else:
                cycles_to_one = None
            term, term32 = s[-1], s32s[-1]
            out['channels'][c] = {
                'windows': windows,
                'terminal': {k: term[k] for k in ('cycle', 'rho', 'gamma', 'norm_z', 'dz_norm', 'rms_step_per_entry',
                                                  'dual_ratio', 'primal_ratio', 'proximal_share')},
                's32_terminal_for_comparison': {k: term32[k] for k in ('rho', 'norm_z', 'dz_norm', 'rms_step_per_entry',
                                                                       'dual_ratio', 'primal_ratio')},
                'rms_step_over_eps_abs_terminal': term['rms_step_per_entry'] / eps_abs if term['rms_step_per_entry'] else None,
                'rms_step_over_e4_delta_terminal': (term['rms_step_per_entry'] / e4['delta_c'][c.upper()])
                if term['rms_step_per_entry'] and e4['delta_c'].get(c.upper()) else None,
                'dual_ratio_decay_c100_to_c150': decay,
                'projected_cycles_to_dual_ratio_1_at_that_rate': cycles_to_one,
                'first_primal_pass_cycle': next((r['cycle'] for r in rows if r[f'boyd_{c}_primal_pass']), None),
                'first_dual_pass_cycle': next((r['cycle'] for r in rows if r[f'boyd_{c}_dual_pass']), None),
                'n_channel_pass_cycles': sum(1 for r in rows if r[f'boyd_{c}_channel_pass']),
            }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    json.dump(out, open(OUT, 'w'), indent=1)
    for c in C:
        ch = out['channels'][c]
        print(c, 'terminal', {k: ('%.4g' % v if isinstance(v, float) else v) for k, v in ch['terminal'].items()})
        print('   s32 terminal', {k: ('%.4g' % v if isinstance(v, float) else v) for k, v in ch['s32_terminal_for_comparison'].items()})
        print('   rms/eps_abs %.3g | rms/E4 delta %.3g | dual decay(100->150) %s | cycles to 1: %s | first dual pass %s | channel pass %d' % (
            ch['rms_step_over_eps_abs_terminal'] or float('nan'), ch['rms_step_over_e4_delta_terminal'] or float('nan'),
            ('%.3f' % ch['dual_ratio_decay_c100_to_c150']) if ch['dual_ratio_decay_c100_to_c150'] else None,
            ('%.0f' % ch['projected_cycles_to_dual_ratio_1_at_that_rate']) if ch['projected_cycles_to_dual_ratio_1_at_that_rate'] else None,
            ch['first_dual_pass_cycle'], ch['n_channel_pass_cycles']))
        for w in ch['windows']:
            print('    ', w['cycles'], 'z %.6g->%.6g (%.3g/cyc)' % (w['norm_z_start_end'][0], w['norm_z_start_end'][1], w['norm_z_increment_per_cycle']),
                  'rms %.3g->%.3g' % tuple(x or float('nan') for x in w['rms_step_per_entry_start_end']),
                  'coh %.2f' % (w['coherence_mean'] if w['coherence_mean'] is not None else float('nan')),
                  'dual %.3g->%.3g' % tuple(w['dual_ratio_start_end']))
    print('guard', out['solve_profile_guard'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
