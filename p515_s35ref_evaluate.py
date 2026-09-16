"""P5.15 Addendum 16 run 1 - reference equilibrium s35ref (frozen spec v5) - evaluation (zero solves, guard armed).

Reference established iff: Boyd stop on all three channels with 3 consecutive converged cycles within 500 cycles, and
no channel frozen at a rho clamp (spec v5 gates.G_reference). Everything else is REPORTED, not gated:
  * stop cycle: the first cycle of the terminal 3-consecutive all-pass run, and the cycle it completed;
  * terminal Boyd residuals per channel with the proximal share, rho/gamma in force, freeze cycles;
  * EFC/day: terminal max, per node, per cohort-year where recorded; read against
      - the revised pre-registered expectation (P5_15_Z2_FLOOR_SLACK_NOTE.md, commit a993b088): approach to the
        price-taker harness-definition value, max over cohort-years 1.1918 (2025), from below, floor slack;
      - the SoH-binding threshold 1.4612 (spec v5's original prediction, now expected to fail);
      - EFC* 1.8520 (single-day price-taker maximum, for context only);
  * SoH floor block from soh_floor_sidecar_baseline.jsonl: per (node, y_inv, y) the floor-row dual (Pyomo sign
    convention, no flip, as recorded), SoH, soh_min, active flag; the maximum |dual| over ACTIVE rows is the reported
    floor multiplier (0 if no row is active);
  * EFC trajectory and its late slope; storage dual-ratio trajectory;
  * system cost at matched cycles vs s34, s33e2, s32, s31c; bars valid only if both runs settled (none of the
    references did), so every bar is marked not valid;
  * failures by tier, solve identity, cancellation closure, terminal interface voltages vs bounds.
Usage: python p515_s35ref_evaluate.py [RUN_DIR] [--dry-run]; writes RUN_DIR/s35ref_evaluation.json (write-once).
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
REFS = {'s34': os.path.join(RES, 'P515S34_run', 'g_baseline.json'),
        's33e2': os.path.join(RES, 'P515S33_E2_run', 'g_baseline.json'),
        's32': os.path.join(RES, 'P515S32_run', 'g_baseline.json'),
        's31c': os.path.join(RES, 'P515S31C_run', 'g_baseline.json')}
PRICE_TAKER_EFC = {'2025': 1.1918, '2030': 1.1888, '2035': 0.9647}
PRICE_TAKER_TERMINAL_SOH = 0.589209
SOH_THRESHOLD = 1.4612
EFC_STAR = 1.8520
MATCH = (1, 2, 5, 10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
BOYD = ('primal_ratio', 'dual_ratio', 'dual_ratio_balance', 'proximal_share', 'primal_pass', 'dual_pass', 'channel_pass')


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
    run = os.path.abspath(args[0]) if args else os.path.join(RES, 'P515S35_REF_run')
    out_path = os.path.join(run, 's35ref_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s35ref evaluation').install()
    try:
        g = json.load(open(_one(os.path.join(run, 'g_*.json'))))
        bt_path = os.path.join(run, 'boyd_terminal.json')
        bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}
        rows = g['cycle_trajectory']
        last, n = rows[-1], len(rows)

        allpass = [r['cycle'] for r in rows if r.get('boyd_all_pass')]
        run_start = None
        if bt.get('stopped_by') == 'boyd' and allpass:
            k = allpass[-1]
            run_start = k
            while run_start - 1 in allpass:
                run_start -= 1
        clamp_any = any(r.get(f'rho_at_clamp_{c}') for r in rows for c in CH)

        term = {}
        for c in CH:
            e = {f: last.get(f'boyd_{c}_{f}') for f in BOYD}
            e['rho_in_force'] = last.get(f'rho_{c}_before')
            e['gamma_in_force'] = last.get(f'gamma_{c}_before')
            e['freeze_cycle'] = next((r['cycle'] for r in rows if r.get(f'rho_frozen_{c}')), None)
            e['cycles_channel_pass'] = sum(1 for r in rows if r.get(f'boyd_{c}_channel_pass'))
            e['first_channel_pass'] = next((r['cycle'] for r in rows if r.get(f'boyd_{c}_channel_pass')), None)
            term[c] = e

        efc = [(r['cycle'], r.get('efc_per_day_max')) for r in rows if r.get('efc_per_day_max') is not None]
        efc_t = efc[-1][1] if efc else None
        slope = None
        if len(efc) > 31:
            slope = (efc[-1][1] - efc[-31][1]) / (efc[-1][0] - efc[-31][0])

        floor = {'available': False}
        side = glob.glob(os.path.join(run, 'soh_floor_sidecar_*.jsonl'))
        if side:
            recs = [json.loads(l) for l in open(side[0]) if l.strip()]
            if recs:
                fin = recs[-1]
                entries = fin.get('entries', [])
                active = [e for e in entries if e.get('active')]
                duals = [abs(e['dual']) for e in active if e.get('dual') is not None]
                efc_cy = {}
                for e in entries:
                    if e.get('efc_per_day') is not None:
                        efc_cy[f"node{e['node_id']}_yinv{e['y_inv']}_y{e['y']}"] = e['efc_per_day']
                floor = {'available': True, 'cycle': fin.get('cycle'), 'dual_sign_convention': fin.get('dual_sign_convention'),
                         'n_rows': len(entries), 'n_active': len(active),
                         'floor_multiplier_max_abs_over_active_rows': (max(duals) if duals else 0.0),
                         'min_soh': min((e['es_soh_per_unit_cumul'] for e in entries if e.get('es_soh_per_unit_cumul') is not None), default=None),
                         'soh_min': entries[0].get('soh_min') if entries else None,
                         'rows': entries, 'efc_per_day_per_cohort_year': efc_cy}

        matched = []
        refs = {k: _traj(v) for k, v in REFS.items()}
        for k in MATCH:
            if k > n:
                continue
            row = {'cycle': k, 's35ref_gross': rows[k - 1]['gross_operational_cost'],
                   's35ref_efc': rows[k - 1].get('efc_per_day_max')}
            for name, ref in refs.items():
                if k in ref:
                    row[f'{name}_gross'] = ref[k]['gross_operational_cost']
                    row[f'minus_{name}'] = row['s35ref_gross'] - ref[k]['gross_operational_cost']
            matched.append(row)

        out = {
            'stage': 'P5.15 Addendum 16 run 1 (s35ref, spec v5) - evaluation',
            'run_dir': os.path.relpath(run, REPO), 'instance': g.get('instance'),
            'spec': {'file': bt.get('spec_file'), 'sha256': bt.get('spec_file_sha256')},
            'REFERENCE_ESTABLISHED': {
                'stopped_by': bt.get('stopped_by'), 'cycles': n, 'converged_at_cycle': g.get('converged_at_cycle'),
                'stop_run_first_cycle': run_start, 'rho_at_clamp_any': clamp_any,
                'verdict': 'ESTABLISHED' if (bt.get('stopped_by') == 'boyd' and not clamp_any) else 'NOT ESTABLISHED'},
            'terminal_per_channel': term,
            'efc': {'terminal_max': efc_t, 'late_slope_per_cycle_last_30': slope,
                    'sampled': [e for e in efc if e[0] in MATCH or e[0] == n],
                    'price_taker_harness_definition': PRICE_TAKER_EFC,
                    'fraction_of_price_taker_2025': (efc_t / PRICE_TAKER_EFC['2025']) if efc_t else None,
                    'soh_threshold': SOH_THRESHOLD, 'fraction_of_threshold': (efc_t / SOH_THRESHOLD) if efc_t else None,
                    'efc_star_single_day': EFC_STAR,
                    'expectation': 'revised, pre-registered (a993b088): approach ~1.19 from below with the floor slack'},
            'soh_floor': floor,
            'price_taker_terminal_soh_for_comparison': PRICE_TAKER_TERMINAL_SOH,
            'storage_dual_ratio_sampled': [(r['cycle'], r.get('boyd_ess_dual_ratio')) for r in rows if r['cycle'] in MATCH or r['cycle'] == n],
            'system_cost_matched_cycles': matched,
            'system_cost_terminal': last['gross_operational_cost'],
            'objective_diagnostic': {'terminal_step': last.get('objective_change_abs'), 'tolerance': last.get('objective_tolerance'),
                                     'rule_ten': last.get('objective_change_ratio')},
            'network_failures_summary': g.get('network_failures_summary'),
            'local_solve_failures': g.get('local_solve_failures'),
            'solves': g.get('solve_profile'), 'wall_clock_s': g.get('wall_clock_s'),
            'scaling_in_force': {k: last.get(k) for k in ('sigma_fixed', 'sigma_computed', 'al_scale_esso', 'shared_ess_reference_rating_mva')},
        }
        vt = os.path.join(run, 'interface_voltage_terminal.json')
        out['interface_voltage_terminal_summary'] = json.load(open(vt))['summary'] if os.path.exists(vt) else None
        det = glob.glob(os.path.join(run, 'interface_settlement_detail_*.json'))
        if det:
            d = json.load(open(det[0]))
            t = d['t_tso_plus_t_dso_terminal']
            p = sum(v['sum_pi_baseMVA_residual_weighted'] for v in d['interface_consensus_residual_per_dso'].values())
            out['cancellation'] = {'t_sum': t, 'priced_residual_total': p, 'closure': t + p}
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    print(json.dumps({k: v for k, v in out.items() if k not in ('soh_floor',)}, indent=1, default=str)[:3000])
    print('soh_floor summary:', {k: v for k, v in out['soh_floor'].items() if k not in ('rows', 'efc_per_day_per_cohort_year')})
    if not dry:
        with open(out_path, 'w') as handle:
            json.dump(out, handle, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
