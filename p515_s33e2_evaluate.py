"""P5.15 Step 3.2 E2 gate (s33e2, spec v3) - evaluation (zero solves, SolveProfileGuard armed).

Addendum 14 report fields: cycles; terminal residuals per channel (primal / dual, with the proximal share of the dual);
rho and gamma trajectories; system cost vs s31c and s32 at matched cycles; failures; E4 noise floor.

Conventions:
  * system cost = gross_operational_cost (settlements excluded; salvage 0 in all three runs, gross = net);
  * binding test = the (channel, primal|dual) pair with the largest ratio to its threshold at the terminal cycle;
  * objective-change test is diagnostic only (rule ten = objective_change_abs / objective_tolerance);
  * matched-cycle and terminal differences against s31c / s32 are not limit statements unless both runs settled:
    s31c (cap 90) and s32 (cap 150) did not, so every bar is reported as not valid;
  * identity check: cycles 1-2 must reproduce s32 (spec v3 prediction).
Usage: python p515_s33e2_evaluate.py [RUN_DIR] [--dry-run]; writes RUN_DIR/s33e2_evaluation.json (write-once).
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

CH = ('v', 'pf', 'ess')
RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
S31C_G = os.path.join(RES, 'P515S31C_run', 'g_baseline.json')
S32_G = os.path.join(RES, 'P515S32_run', 'g_baseline.json')
E4 = os.path.join(RES, 'P515S33', 'E4', 'e4_noise_floor_results.json')
MATCH = (1, 2, 3, 5, 10, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150)
BOYD_FIELDS = ('r', 's', 's_rho_part', 's_proximal_part', 'proximal_share', 'eps_pri', 'eps_dual', 'norm_x', 'norm_z',
               'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance', 'primal_pass', 'dual_pass', 'channel_pass')


def _one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError(f'expected one file for {pattern}, found {hits}')
    return hits[0]


def _traj(path):
    return {t['cycle']: t for t in json.load(open(path))['cycle_trajectory']}


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run = os.path.abspath(args[0]) if args else os.path.join(RES, 'P515S33_E2_run')
    out_path = os.path.join(run, 's33e2_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s33e2 evaluation').install()
    try:
        g = json.load(open(_one(os.path.join(run, 'g_*.json'))))
        bt = json.load(open(os.path.join(run, 'boyd_terminal.json')))
        rows = g['cycle_trajectory']
        last = rows[-1]
        n = len(rows)
        s31c, s32 = _traj(S31C_G), _traj(S32_G)
        e4 = json.load(open(E4))['derivation']

        # identity with s32 at cycles 1-2 (spec v3 prediction)
        identity = {}
        for k in (1, 2):
            if k > n:
                continue
            a, b = rows[k - 1], s32[k]
            diffs = {}
            for key, val in a.items():
                if not (key.startswith('boyd_') or key.startswith('rho_') or key == 'gross_operational_cost'):
                    continue
                if key not in b:
                    continue
                if isinstance(val, bool) or isinstance(b[key], bool) or isinstance(val, str) or isinstance(b[key], str):
                    if val != b[key]:
                        diffs[key] = [val, b[key]]
                elif isinstance(val, (int, float)) and isinstance(b[key], (int, float)):
                    d = abs(val - b[key])
                    if d != 0.0:
                        diffs[key] = d
            identity[k] = {'n_fields_differing': len(diffs), 'differences': diffs}

        term = {}
        binding = None
        for c in CH:
            e = {f: last.get(f'boyd_{c}_{f}') for f in BOYD_FIELDS}
            e['gamma_in_force'] = last.get(f'gamma_{c}_before')
            e['rho_in_force'] = last.get(f'rho_{c}_before')
            term[c] = e
            for test in ('primal', 'dual'):
                val = e[f'{test}_ratio']
                if val is not None and (binding is None or val > binding[2]):
                    binding = (c, test, val)

        traj = {}
        for c in CH:
            seq = [(r['cycle'], r.get(f'rho_{c}_before'), r.get(f'rho_{c}_after'), r.get(f'rho_{c}_action'),
                    r.get(f'gamma_{c}_before'), r.get(f'gamma_{c}_after')) for r in rows]
            counts = {}
            for s in seq:
                counts[s[3]] = counts.get(s[3], 0) + 1
            traj[c] = {
                'actions': counts,
                'change_cycles': [s[0] for s in seq if s[3] in ('increased', 'decreased')],
                'rho_initial': seq[0][1], 'rho_terminal_after': seq[-1][2],
                'gamma_initial': seq[0][4], 'gamma_terminal_after': seq[-1][5],
                'gamma_equals_tau_rho_every_cycle': all(
                    s[4] is not None and s[1] is not None and abs(s[4] - (last.get('gamma_tau') or 1.0) * s[1]) <= 1e-12 * max(1.0, abs(s[1]))
                    for s in seq),
                'sampled': [{'cycle': s[0], 'rho_before': s[1], 'rho_after': s[2], 'action': s[3],
                             'gamma_before': s[4], 'gamma_after': s[5]} for s in seq if s[0] in MATCH or s[0] == n],
                'proximal_share_terminal': last.get(f'boyd_{c}_proximal_share'),
            }
        pass_counts = {c: {t: sum(1 for r in rows if r.get(f'boyd_{c}_{t}_pass')) for t in ('primal', 'dual')} for c in CH}
        all_pass = [r['cycle'] for r in rows if r.get('boyd_all_pass')]
        freeze_check = {
            'freeze_after_cycle': last.get('freeze_after_cycle'),
            'changes_after_freeze': {c: [r['cycle'] for r in rows if r['cycle'] > (last.get('freeze_after_cycle') or 10 ** 9)
                                         and r.get(f'rho_{c}_action') in ('increased', 'decreased')] for c in CH},
        }

        matched = []
        for k in MATCH:
            if k > n:
                continue
            row = {'cycle': k, 's33e2_gross': rows[k - 1]['gross_operational_cost']}
            row['s31c_gross'] = s31c[k]['gross_operational_cost'] if k in s31c else None
            row['s32_gross'] = s32[k]['gross_operational_cost'] if k in s32 else None
            row['minus_s31c'] = (row['s33e2_gross'] - row['s31c_gross']) if row['s31c_gross'] is not None else None
            row['minus_s32'] = (row['s33e2_gross'] - row['s32_gross']) if row['s32_gross'] is not None else None
            matched.append(row)
        terminal_cmp = {
            's33e2': {'cycle': n, 'gross': last['gross_operational_cost'], 'terminal_step': last.get('objective_change_abs')},
            's32': {'cycle': max(s32), 'gross': s32[max(s32)]['gross_operational_cost'], 'terminal_step': s32[max(s32)]['objective_change_abs']},
            's31c': {'cycle': max(s31c), 'gross': s31c[max(s31c)]['gross_operational_cost'], 'terminal_step': s31c[max(s31c)]['objective_change_abs']},
            'bars_valid': False,
            'note': 's31c and s32 did not settle; differences bound stopping slack only and are not limit statements'}

        vt_path = os.path.join(run, 'interface_voltage_terminal.json')
        vt = json.load(open(vt_path))['summary'] if os.path.exists(vt_path) else None

        out = {
            'stage': 'P5.15 Step 3.2 E2 gate (s33e2) - evaluation',
            'run_dir': os.path.relpath(run, REPO), 'instance': g.get('instance'),
            'spec': {'file': bt.get('spec_file'), 'sha256': bt.get('spec_file_sha256')},
            'cycles': {'n': n, 'stopped_by': bt.get('stopped_by'), 'converged_at_cycle': g.get('converged_at_cycle'),
                       'consecutive_converged_at_stop': bt.get('consecutive_converged_at_stop'), 'cap': 150,
                       'all_pass_cycles': all_pass},
            'identity_with_s32_cycles_1_2': identity,
            'terminal_residuals_per_channel': term,
            'binding_test': {'channel': binding[0], 'test': binding[1], 'ratio': binding[2]} if binding else None,
            'pass_counts': pass_counts,
            'rho_gamma_trajectory': traj,
            'freeze_check': freeze_check,
            'objective_diagnostic': {'terminal_step': last.get('objective_change_abs'),
                                     'terminal_tolerance': last.get('objective_tolerance'),
                                     'rule_ten': last.get('objective_change_ratio')},
            'system_cost_matched_cycles': matched,
            'system_cost_terminal': terminal_cmp,
            'network_failures_summary': g.get('network_failures_summary'),
            'local_solve_failures': g.get('local_solve_failures'),
            'solves': g.get('solve_profile'),
            'wall_clock_s': g.get('wall_clock_s'),
            'interface_voltage_terminal_summary': vt,
            'e4_noise_floor': {'delta_c': e4['delta_c'], 'eps_abs': e4['eps_abs']},
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    print(json.dumps(out, indent=1, default=str))
    if not dry:
        with open(out_path, 'w') as handle:
            json.dump(out, handle, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
