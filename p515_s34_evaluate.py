"""P5.15 Step 3.4 gate (s34, frozen spec v4) - evaluation (zero solves, SolveProfileGuard armed).

Reports, per Addendum 15 and spec v4:
  * cycles, stop cause, consecutive converged cycles at stop;
  * terminal Boyd residuals per channel (primal / dual, with the proximal share of the dual), rho and gamma in force;
  * rho and gamma trajectories, per-channel freeze cycle, unchanged-streak, and the rho_at_clamp GATE FAILURE flag;
  * scaling constants actually in force: sigma_fixed vs sigma_computed, al_scale_esso, S_ref;
  * EFC/day per cycle and at terminal, read against the price-taker benchmark EFC* = 1.8520 (2025 max across nodes)
    and the SoH-binding threshold 1.4612 - never against the retired 1.1;
  * the storage-response falsifier: terminal per-entry ESS step vs s33e2's 4.26e-5 at matched normalization. Spec v4
    states that removing an ~sigma stiffness should raise it by orders of magnitude; a rise < 3x with the step cosine
    still at 1.000 falsifies the Addendum 15 stiffness reading on its own terms;
  * the S_ref prediction, recorded separately from the pass test: ESS dual ratio x2.58 versus s33e2 at matched
    conditions (3.245 -> ~8.4), eps_dual moving < 1 %;
  * the cycle-30 early read (storage fraction of rating and step cosine);
  * system cost at matched cycles against s31c, s32 and s33e2, with every bar marked not valid (no reference settled);
  * failures by tier, solve identity, settlement cancellation, per-entry terminal voltages vs bounds.
Pass test is the Boyd stop ALONE (3 consecutive cycles, all channels); EFC, cost and agreement with earlier runs are
diagnostics. Usage: python p515_s34_evaluate.py [RUN_DIR] [--dry-run]; writes RUN_DIR/s34_evaluation.json (write-once).
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
REFS = {'s31c': os.path.join(RES, 'P515S31C_run', 'g_baseline.json'),
        's32': os.path.join(RES, 'P515S32_run', 'g_baseline.json'),
        's33e2': os.path.join(RES, 'P515S33_E2_run', 'g_baseline.json')}
EFC_STAR_2025_MAX = 1.8520
EFC_STAR_RANGE = [0.7964, 1.8520]
SOH_THRESHOLD = 1.4612
S33E2_ESS_STEP_PER_ENTRY = 4.26e-5
S33E2_ESS_DUAL_RATIO = 3.245
MATCH = (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 75, 90, 100, 125, 150)
BOYD = ('r', 's', 's_rho_part', 's_proximal_part', 'proximal_share', 'eps_pri', 'eps_dual', 'norm_x', 'norm_z',
        'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance', 'primal_pass', 'dual_pass', 'channel_pass')


def _one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError(f'expected one file for {pattern}, found {hits}')
    return hits[0]


def _traj(path):
    return {t['cycle']: t for t in json.load(open(path))['cycle_trajectory']} if os.path.exists(path) else {}


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run = os.path.abspath(args[0]) if args else os.path.join(RES, 'P515S34_run')
    out_path = os.path.join(run, 's34_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s34 evaluation').install()
    try:
        g = json.load(open(_one(os.path.join(run, 'g_*.json'))))
        bt_path = os.path.join(run, 'boyd_terminal.json')
        bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}
        rows = g['cycle_trajectory']
        last, n = rows[-1], len(rows)
        refs = {k: _traj(v) for k, v in REFS.items()}

        term, binding = {}, None
        for c in CH:
            e = {f: last.get(f'boyd_{c}_{f}') for f in BOYD}
            e['rho_in_force'] = last.get(f'rho_{c}_before')
            e['gamma_in_force'] = last.get(f'gamma_{c}_before')
            e['frozen'] = last.get(f'rho_frozen_{c}')
            e['rho_at_clamp'] = last.get(f'rho_at_clamp_{c}')
            term[c] = e
            for t in ('primal', 'dual'):
                v = e[f'{t}_ratio']
                if v is not None and (binding is None or v > binding[2]):
                    binding = (c, t, v)

        traj = {}
        for c in CH:
            seq = [(r['cycle'], r.get(f'rho_{c}_before'), r.get(f'rho_{c}_after'), r.get(f'rho_{c}_action'),
                    r.get(f'gamma_{c}_before'), r.get(f'rho_frozen_{c}'), r.get(f'rho_unchanged_streak_{c}')) for r in rows]
            acts = {}
            for s in seq:
                acts[s[3]] = acts.get(s[3], 0) + 1
            traj[c] = {'actions': acts, 'change_cycles': [s[0] for s in seq if s[3] in ('increased', 'decreased')],
                       'rho_initial': seq[0][1], 'rho_terminal': seq[-1][2], 'gamma_terminal': seq[-1][4],
                       'freeze_cycle': next((s[0] for s in seq if s[5]), None),
                       'rho_at_clamp_any': any(r.get(f'rho_at_clamp_{c}') for r in rows),
                       'sampled': [{'cycle': s[0], 'rho': s[1], 'action': s[3], 'gamma': s[4], 'frozen': s[5],
                                    'streak': s[6]} for s in seq if s[0] in MATCH or s[0] == n]}

        efc = [(r['cycle'], r.get('efc_per_day_max')) for r in rows if r.get('efc_per_day_max') is not None]
        efc_terminal = efc[-1][1] if efc else None
        ess_step = (last.get('boyd_ess_s_rho_part') / last.get('boyd_ess_rho_in_force')
                    if last.get('boyd_ess_rho_in_force') else None)
        n_ess = None
        if last.get('boyd_ess_eps_dual') and last.get('boyd_ess_norm_y') is not None:
            sqrt_n = (last['boyd_ess_eps_dual'] - last['boyd_eps_rel'] * last['boyd_ess_norm_y']) / last['boyd_eps_abs']
            n_ess = sqrt_n ** 2
            if ess_step is None:
                rho = last.get(f'rho_ess_before')
                ess_step = (last['boyd_ess_s_rho_part'] / rho / sqrt_n) if rho else None
            else:
                ess_step = ess_step / sqrt_n

        cyc30 = next((r for r in rows if r['cycle'] == 30), None)
        matched = []
        for k in MATCH:
            if k > n:
                continue
            row = {'cycle': k, 's34_gross': rows[k - 1]['gross_operational_cost']}
            for name, ref in refs.items():
                row[f'{name}_gross'] = ref[k]['gross_operational_cost'] if k in ref else None
                row[f'minus_{name}'] = (row['s34_gross'] - row[f'{name}_gross']) if row.get(f'{name}_gross') else None
            matched.append(row)

        out = {
            'stage': 'P5.15 Step 3.4 gate (s34, spec v4) - evaluation',
            'run_dir': os.path.relpath(run, REPO), 'instance': g.get('instance'),
            'spec': {'file': bt.get('spec_file'), 'sha256': bt.get('spec_file_sha256')},
            'scaling_in_force': {k: last.get(k) for k in ('sigma_fixed', 'sigma_computed', 'al_scale_esso',
                                                          'shared_ess_reference_rating_mva', 'boyd_eps_abs', 'boyd_eps_rel')},
            'PASS_TEST_boyd_stop': {'cycles': n, 'stopped_by': bt.get('stopped_by'),
                                    'converged_at_cycle': g.get('converged_at_cycle'),
                                    'consecutive_at_stop': bt.get('consecutive_converged_at_stop'),
                                    'all_pass_cycles': [r['cycle'] for r in rows if r.get('boyd_all_pass')],
                                    'rho_at_clamp_any_channel': any(traj[c]['rho_at_clamp_any'] for c in CH),
                                    'verdict': ('PASS' if bt.get('stopped_by') == 'boyd' and
                                                not any(traj[c]['rho_at_clamp_any'] for c in CH) else 'FAIL')},
            'terminal_residuals_per_channel': term,
            'binding_test': {'channel': binding[0], 'test': binding[1], 'ratio': binding[2]} if binding else None,
            'pass_counts': {c: {t: sum(1 for r in rows if r.get(f'boyd_{c}_{t}_pass')) for t in ('primal', 'dual')} for c in CH},
            'rho_gamma_freeze': traj,
            'efc_diagnostic': {'terminal': efc_terminal, 'sampled': [e for e in efc if e[0] in MATCH or e[0] == n],
                               'EFC_star_2025_max': EFC_STAR_2025_MAX, 'EFC_star_range_all_cells': EFC_STAR_RANGE,
                               'soh_binding_threshold': SOH_THRESHOLD,
                               'fraction_of_EFC_star': (efc_terminal / EFC_STAR_2025_MAX) if efc_terminal else None,
                               'fraction_of_soh_threshold': (efc_terminal / SOH_THRESHOLD) if efc_terminal else None,
                               's33e2_terminal_for_comparison': 0.0671,
                               'reading': 'equilibrium near the SoH threshold with the constraint active is the expected success case'},
            'storage_response_falsifier': {
                's34_ess_step_per_entry': ess_step, 's33e2_ess_step_per_entry': S33E2_ESS_STEP_PER_ENTRY,
                'ratio': (ess_step / S33E2_ESS_STEP_PER_ENTRY) if ess_step else None,
                'n_ess_entries': n_ess,
                'criterion': 'spec v4: a rise < 3x together with a persistent step cosine of 1.000 falsifies the Addendum 15 stiffness reading'},
            's_ref_prediction': {'s33e2_ess_dual_ratio': S33E2_ESS_DUAL_RATIO,
                                 's34_ess_dual_ratio_terminal': term['ess']['dual_ratio'],
                                 'predicted_multiplier': 2.58,
                                 'note': 'recorded separately from the pass test; x1.0 means the residual is not gradient-limited, x0.39 means the mechanism is misidentified'},
            'cycle_30_early_read': {k: (cyc30 or {}).get(k) for k in
                                    ('boyd_ess_norm_z', 'boyd_ess_r', 'boyd_ess_s', 'rho_ess_before', 'efc_per_day_max')},
            'system_cost_matched_cycles': matched,
            'system_cost_terminal': {'s34': last['gross_operational_cost'],
                                     **{k: (ref[max(ref)]['gross_operational_cost'] if ref else None) for k, ref in refs.items()},
                                     'bars_valid': False,
                                     'note': 'no reference run settled; differences bound stopping slack only'},
            'objective_diagnostic': {'terminal_step': last.get('objective_change_abs'),
                                     'terminal_tolerance': last.get('objective_tolerance'),
                                     'rule_ten': last.get('objective_change_ratio')},
            'network_failures_summary': g.get('network_failures_summary'),
            'local_solve_failures': g.get('local_solve_failures'),
            'solves': g.get('solve_profile'), 'wall_clock_s': g.get('wall_clock_s'),
        }
        vt = os.path.join(run, 'interface_voltage_terminal.json')
        out['interface_voltage_terminal_summary'] = json.load(open(vt))['summary'] if os.path.exists(vt) else None
        det = glob.glob(os.path.join(run, 'interface_settlement_detail_*.json'))
        if det:
            d = json.load(open(det[0]))
            out['cancellation'] = {'t_sum': d['t_tso_plus_t_dso_terminal'],
                                   'priced_residual_total': sum(v['sum_pi_baseMVA_residual_weighted']
                                                                for v in d['interface_consensus_residual_per_dso'].values())}
            out['cancellation']['closure'] = out['cancellation']['t_sum'] + out['cancellation']['priced_residual_total']
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
