"""P5.15 Step 3.2 + 3.3(a) gate s32 - evaluation (zero solves, SolveProfileGuard armed).

Report order required by Addendum 13: (A) the s31c rho/residual extraction, (B) the s32 gate cycle count; then terminal
residuals per channel, rho trajectory per channel, system cost vs s31c at matched cycles and at the terminal point,
failures, settlement cancellation, per-DSO settlement, D rows.

Conventions:
  * system cost = gross_operational_cost (settlements excluded by construction; salvage 0 in both runs, so gross = net);
  * binding test = the (channel, primal|dual) pair with the largest ratio to its threshold at the terminal cycle;
  * rule ten (objective diagnostic) = objective_change_abs / objective_tolerance at the terminal cycle; the objective
    test is NOT a stopping criterion under spec v2;
  * difference bar = sum of the two terminal objective steps; valid only if BOTH runs settled - s31c did not (cap 90,
    rule ten 1.96, still descending), so the terminal comparison is reported with the bar marked not valid;
  * matched-cycle differences are not attributable to the stopping rule (initial rho and balancing rule differ).
Usage: python p515_s32_evaluate.py [RUN_DIR] [--dry-run]; writes RUN_DIR/s32_evaluation.json (write-once).
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

CH = ('v', 'pf', 'ess')
S31C_G = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C_run', 'g_baseline.json')
S31C_EXTRACT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 's31c_rho_residual_extraction.json')
MATCH = (1, 2, 5, 10, 12, 18, 20, 30, 40, 50, 60, 70, 80, 90)


def _one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError(f'expected one file for {pattern}, found {hits}')
    return hits[0]


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run = os.path.abspath(args[0]) if args else os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32_run')
    out_path = os.path.join(run, 's32_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s32 evaluation').install()
    try:
        g = json.load(open(_one(os.path.join(run, 'g_*.json'))))
        bt = json.load(open(os.path.join(run, 'boyd_terminal.json')))
        levels = json.load(open(os.path.join(run, 'component_levels_terminal.json')))
        detail = json.load(open(_one(os.path.join(run, 'interface_settlement_detail_*.json'))))
        ext = json.load(open(S31C_EXTRACT))
        s31c = {t['cycle']: t for t in json.load(open(S31C_G))['cycle_trajectory']}
        rows = g['cycle_trajectory']
        last = rows[-1]
        n = len(rows)

        # (A) s31c extraction summary
        crawl = ext['summary_crawl']
        lead_a = {
            'initial_rho_s31c_harness': ext['initial_rho_harness'],
            'rho_change_cycles': ext['rho_change_cycles'],
            'rho_terminal_in_force': ext['rho_terminal_in_force'],
            'crawl_window': ext['rule_definitions']['crawl_window'],
            'crawl_hold_reasons': {c: crawl[c]['reasons'] for c in CH},
            'crawl_frozen_cycles_where_dead_band_alone_would_act': {
                c: {k: len(v) for k, v in crawl[c]['frozen_cycles_where_rule_would_act_without_freeze'].items()} for c in CH},
        }

        # (B) cycles
        lead_b = {'cycles': n, 'stopped_by': bt.get('stopped_by'), 'converged_at_cycle': g.get('converged_at_cycle'),
                  'cap': 150, 'spec_file': bt.get('spec_file'), 'spec_sha256': bt.get('spec_file_sha256')}

        # terminal residuals
        term = {}
        binding = None
        for c in CH:
            e = {k: last.get(f'boyd_{c}_{k}') for k in (
                'r', 's', 's_rho_part', 's_proximal_part', 'proximal_share', 'eps_pri', 'eps_dual', 'norm_x', 'norm_z',
                'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance', 'primal_pass', 'dual_pass', 'channel_pass')}
            e['legacy_primal_max_ratio'] = last.get(f'primal_{c}_ratio')
            e['legacy_primal_mean_ratio'] = last.get(f'primal_{c}_mean_ratio')
            e['legacy_dual_mean_ratio'] = last.get(f'dual_{c}_mean_ratio')
            term[c] = e
            for test in ('primal', 'dual'):
                val = e[f'{test}_ratio']
                if val is not None and (binding is None or val > binding[2]):
                    binding = (c, test, val)

        # rho trajectory
        rho = {}
        for c in CH:
            seq = [(r['cycle'], r.get(f'rho_{c}_before'), r.get(f'rho_{c}_after'), r.get(f'rho_{c}_action')) for r in rows]
            counts = {}
            for _, _, _, a in seq:
                counts[a] = counts.get(a, 0) + 1
            rho[c] = {
                'actions': counts,
                'change_cycles': [k for k, _, _, a in seq if a in ('increased', 'decreased')],
                'min_in_force': min(b for _, b, _, _ in seq), 'max_in_force': max(b for _, b, _, _ in seq),
                'initial': seq[0][1], 'terminal_after': seq[-1][2],
                'sampled': [{'cycle': k, 'rho_before': b, 'rho_after': a2, 'action': a}
                            for k, b, a2, a in seq if k in MATCH or k == n],
                'proximal_share_terminal': last.get(f'boyd_{c}_proximal_share'),
                'proximal_share_max': max((r.get(f'boyd_{c}_proximal_share') or 0.0) for r in rows),
            }

        # objective diagnostic
        steps = [r.get('objective_change_abs') for r in rows if r.get('objective_change_abs') is not None]
        tols = [r.get('objective_tolerance') for r in rows if r.get('objective_tolerance') is not None]
        objective = {
            'terminal_step': last.get('objective_change_abs'), 'terminal_tolerance': last.get('objective_tolerance'),
            'rule_ten': last.get('objective_change_ratio'),
            'recent_mean_step_last5': (sum(steps[-5:]) / len(steps[-5:])) if steps else None,
            'recent_mean_rule_ten_last5': ((sum(steps[-5:]) / len(steps[-5:])) / tols[-1]) if steps and tols else None,
        }

        # system cost vs s31c
        matched = []
        for k in sorted(set(MATCH) | {n}):
            if k <= n and k in s31c:
                a, b = rows[k - 1]['gross_operational_cost'], s31c[k]['gross_operational_cost']
                matched.append({'cycle': k, 's32_gross': a, 's31c_gross': b,
                                'difference': (a - b) if a is not None and b is not None else None})
        s31c_last = s31c[max(s31c)]
        s32_step = last.get('objective_change_abs')
        bar = (s32_step + s31c_last['objective_change_abs']) if s32_step is not None else None
        diff = last['gross_operational_cost'] - s31c_last['gross_operational_cost']
        terminal_cmp = {
            's32_cycle': n, 's32_gross': last['gross_operational_cost'], 's31c_cycle': max(s31c),
            's31c_gross': s31c_last['gross_operational_cost'], 'difference': diff,
            'relative_difference': diff / s31c_last['gross_operational_cost'],
            's32_terminal_step': s32_step, 's31c_terminal_step': s31c_last['objective_change_abs'],
            'bar_sum_of_terminal_steps': bar,
            'bar_valid': False,
            'bar_note': 's31c did not settle (cap 90, rule ten 1.96, descending); the bar bounds stopping slack only',
        }

        rc = levels['recourse_components']
        out = {
            'stage': 'P5.15 Step 3.2 + 3.3(a) gate s32 - evaluation',
            'run_dir': os.path.relpath(run, REPO), 'instance': g.get('instance'),
            'A_s31c_rho_residual_extraction': lead_a,
            'B_s32_cycles': lead_b,
            'terminal_residuals_per_channel': term,
            'binding_test': {'channel': binding[0], 'test': binding[1], 'ratio': binding[2]} if binding else None,
            'rho_trajectory': rho,
            'objective_diagnostic': objective,
            'system_cost_vs_s31c': {'matched_cycles': matched, 'terminal': terminal_cmp,
                                    'attribution_note': 'initial rho and balancing rule differ from s31c'},
            'network_failures_summary': g.get('network_failures_summary'),
            'local_solve_failures': g.get('local_solve_failures'),
            'solves': g.get('solve_profile'),
            'wall_clock_s': g.get('wall_clock_s'),
            'gap_proxy_terminal': {'G_over_Q': last.get('gap_proxy_G_over_Q'), 'reason': last.get('gap_proxy_G_reason')},
            'cancellation': {'t_tso': detail['t_tso_total'], 't_dso_by_node': detail['t_dso_by_node'],
                             't_sum': detail['t_tso_plus_t_dso_terminal'],
                             'priced_residual_total': sum(v['sum_pi_baseMVA_residual_weighted']
                                                          for v in detail['interface_consensus_residual_per_dso'].values())},
            'per_dso': {nid: {'settlement_weighted': v['dso_settlement_sum_pi_p_int_weighted'],
                              'flexibility_volumes': detail['flexibility_volumes_per_dso'].get(nid)}
                        for nid, v in detail['interface_consensus_residual_per_dso'].items()},
            'recourse_components': {k: rc.get(k) for k in (
                'gross_operational_cost', 'gross_operational_cost_including_settlement', 'detector_penalty_total',
                'detector_components', 'economic_recourse_all_D_excluded')},
            'economic_components_weighted': {k: levels['totals_weighted'].get(k) for k in (
                'generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
                'ess_usage_cost')},
        }
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
