"""P5.15 Step 3.2/3.3(a) item (1): zero-solve extraction of the rho trajectory and per-channel residual ratios from
the s31c reference trajectory (Addendum 13).

Sources (read-only): data/SRP1/Results/P515S31C_run/stdout_baseline.log and g_baseline.json; the ADMM penalty_update
block of data/SRP1/SRP1_params.json; the harness initial rho p514_n_instrumented_cstar.RHO (read by regex).

Per cycle k and channel c in {v, pf, ess}:
  * the production [ADMM RHO] line printed by _update_admm_penalties at the end of cycle k: primal max ratio, primal
    mean ratio, primal selected = max of the two, dual mean ratio, thresholds, action;
  * reason for the action, using the production rule:
      held_adaptation_converged  when selected <= 1 and dual <= 1 (freeze clause),
      held_dead_band             when held otherwise,
      held_after_solver_failure  when production printed that action;
  * rho_in_force(k): rho used during cycle k. rho_in_force(1) = harness RHO; rho_in_force(k+1) = clamp(rho_in_force(k)
    * factor(action_k), [min, max]), factor 1.5 up / 1/1.5 down from the case file.
Reasons and the would-act test use the FULL-PRECISION ratios from g_baseline.json cycle_trajectory (the log prints
3 decimals, so a printed 0.000 would make 'dual > 5*primal' trivially true); the logged action is checked against the
production rule recomputed at full precision.
Cross-checks: (a) ratios from the log vs g_baseline.json cycle_trajectory (primal_*_ratio / primal_*_mean_ratio /
dual_*_mean_ratio); (b) reconstructed rho vs every rho value production printed inside a cycle
('[WARNING] ADMM penalties ...' and '[DIAG][PF MAX] ... rho(TSO/DSO)').
Guard: SolveProfileGuard armed with zero permitted solves for the whole script.
Output (write-once): data/SRP1/Results/P515S32/s31c_rho_residual_extraction.json
"""
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C_run')
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32')
OUT = os.path.join(OUT_DIR, 's31c_rho_residual_extraction.json')
CHANNELS = ('v', 'pf', 'ess')
CRAWL_FROM = 18
RE_RHO = re.compile(r'\[ADMM RHO\] (V|PF|ESS) \| primal max ratio=([0-9.eE+-]+) \| primal mean ratio=([0-9.eE+-]+) \| '
                    r'primal selected=([0-9.eE+-]+) \| dual mean ratio=([0-9.eE+-]+) \| increase threshold=([0-9.]+) \| '
                    r'decrease threshold=([0-9.]+) \| action=(.+?)\s*$')
RE_CYCLE = re.compile(r'ADMM cycle (\d+) \|')
RE_WARN = re.compile(r'\[WARNING\] ADMM penalties for (\S+), year=(\d+), day=(\w+): rho_v=([0-9.eE+-]+), '
                     r'rho_pf=([0-9.eE+-]+), rho_ess=([0-9.eE+-]+)')
RE_DIAG = re.compile(r'\[DIAG\]\[PF MAX\].*rho\(TSO/DSO\)=([0-9.eE+-]+)/([0-9.eE+-]+)')


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 S32 s31c rho/residual extraction').install()
    try:
        params = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')))['admm']
        upd = params['penalty_update']
        harness_src = open(os.path.join(REPO, 'p514_n_instrumented_cstar.py')).read()
        m = re.search(r"^RHO = (\{.*\})\s*$", harness_src, re.M)
        rho0 = {k: float(v) for k, v in eval(m.group(1), {}).items()}  # literal dict of floats in the harness

        pending, cycles, in_cycle_rho = {}, {}, []
        with open(os.path.join(RUN, 'stdout_baseline.log')) as handle:
            for line in handle:
                r = RE_RHO.search(line)
                if r:
                    pending[r.group(1).lower()] = {
                        'primal_max_ratio': float(r.group(2)), 'primal_mean_ratio': float(r.group(3)),
                        'primal_selected_ratio': float(r.group(4)), 'dual_mean_ratio': float(r.group(5)),
                        'increase_threshold': float(r.group(6)), 'decrease_threshold': float(r.group(7)),
                        'action': r.group(8)}
                    continue
                w = RE_WARN.search(line)
                if w:
                    in_cycle_rho.append({'cycle': len(cycles) + 1, 'source': 'warning', 'network': w.group(1),
                                         'v': float(w.group(4)), 'pf': float(w.group(5)), 'ess': float(w.group(6))})
                    continue
                d = RE_DIAG.search(line)
                if d:
                    in_cycle_rho.append({'cycle': len(cycles) + 1, 'source': 'diag_pf_max',
                                         'pf': float(d.group(1)), 'pf_dso': float(d.group(2))})
                    continue
                c = RE_CYCLE.search(line)
                if c and '[INFO]' in line:
                    k = int(c.group(1))
                    if set(pending) != set(CHANNELS):
                        raise RuntimeError(f'cycle {k}: [ADMM RHO] block incomplete: {sorted(pending)}')
                    cycles[k] = pending
                    pending = {}
        n = max(cycles)
        if sorted(cycles) != list(range(1, n + 1)):
            raise RuntimeError('cycle sequence has gaps')

        traj = {t['cycle']: t for t in json.load(open(os.path.join(RUN, 'g_baseline.json')))['cycle_trajectory']}
        tol = params['tol']
        rho = dict(rho0)
        rows = []
        action_mismatches = []
        for k in range(1, n + 1):
            row = {'cycle': k}
            for ch in CHANNELS:
                e = {f'log_{key}': val for key, val in cycles[k][ch].items()}
                t = traj[k]
                e['primal_max_ratio'] = t[f'primal_{ch}_ratio']
                e['primal_mean_ratio'] = t[f'primal_{ch}_mean_ratio']
                e['primal_selected_ratio'] = max(e['primal_max_ratio'], e['primal_mean_ratio'])
                e['dual_mean_ratio'] = t[f'dual_{ch}_mean_ratio']
                e['increase_threshold'] = e['log_increase_threshold']
                e['decrease_threshold'] = e['log_decrease_threshold']
                e['action'] = e['log_action']
                e['rho_in_force'] = rho[ch]
                frozen = e['primal_selected_ratio'] <= 1.0 and e['dual_mean_ratio'] <= 1.0
                up = e['primal_selected_ratio'] > e['increase_threshold'] * e['dual_mean_ratio']
                down = e['dual_mean_ratio'] > e['decrease_threshold'] * e['primal_selected_ratio']
                e['rule_would_act_without_freeze'] = 'increase' if up else ('decrease' if down else None)
                if t['local_solves_ok']:
                    expected = 'held' if frozen else ('increased' if up else ('decreased' if down else 'held'))
                else:
                    expected = 'held after solver failure'
                if expected != e['action']:
                    action_mismatches.append({'cycle': k, 'channel': ch, 'logged': e['action'], 'recomputed': expected})
                act = e['action']
                if act == 'held':
                    e['reason'] = 'held_adaptation_converged' if frozen else 'held_dead_band'
                    factor = 1.0
                elif act == 'increased':
                    e['reason'], factor = 'increased', upd['increase_factor']
                elif act == 'decreased':
                    e['reason'], factor = 'decreased', 1.0 / upd['decrease_factor']
                elif act == 'held after solver failure':
                    e['reason'], factor = 'held_after_solver_failure', 1.0
                else:
                    raise RuntimeError(f'cycle {k} {ch}: unknown action {act!r}')
                rho[ch] = min(max(rho[ch] * factor, upd['min']), upd['max'])
                e['rho_after_update'] = rho[ch]
                row[ch] = e
            rows.append(row)

        # cross-check (a): ratios vs g_baseline.json trajectory
        worst = 0.0
        for row in rows:
            for ch in CHANNELS:
                for key in ('primal_max_ratio', 'primal_mean_ratio', 'dual_mean_ratio'):
                    worst = max(worst, abs(row[ch][f'log_{key}'] - row[ch][key]))
        if worst > 5.0e-4:
            raise RuntimeError(f'log ratios differ from trajectory beyond 3-decimal rounding: {worst}')
        # cross-check (b): reconstructed rho vs rho printed inside cycles
        rho_mismatch = []
        for obs in in_cycle_rho:
            row = rows[obs['cycle'] - 1]
            for ch in CHANNELS:
                if ch in obs:
                    rec = row[ch]['rho_in_force']
                    if abs(rec - obs[ch]) > 5e-4 * max(1.0, abs(obs[ch])) + 5e-5:
                        rho_mismatch.append({'cycle': obs['cycle'], 'channel': ch, 'observed': obs[ch], 'reconstructed': rec,
                                             'source': obs['source']})

        def window(lo, hi):
            out = {}
            sel = [r for r in rows if lo <= r['cycle'] <= hi]
            for ch in CHANNELS:
                reasons = {}
                for r in sel:
                    reasons[r[ch]['reason']] = reasons.get(r[ch]['reason'], 0) + 1
                out[ch] = {
                    'reasons': reasons,
                    'rho_in_force_min': min(r[ch]['rho_in_force'] for r in sel),
                    'rho_in_force_max': max(r[ch]['rho_in_force'] for r in sel),
                    'primal_selected_ratio_min_max': [min(r[ch]['primal_selected_ratio'] for r in sel),
                                                      max(r[ch]['primal_selected_ratio'] for r in sel)],
                    'dual_mean_ratio_min_max': [min(r[ch]['dual_mean_ratio'] for r in sel),
                                                max(r[ch]['dual_mean_ratio'] for r in sel)],
                    'frozen_cycles_where_rule_would_act_without_freeze': {
                        direction: [r['cycle'] for r in sel if r[ch]['reason'] == 'held_adaptation_converged'
                                    and r[ch]['rule_would_act_without_freeze'] == direction]
                        for direction in ('increase', 'decrease')},
                }
            return out

        change_cycles = {ch: [r['cycle'] for r in rows if r[ch]['action'] in ('increased', 'decreased')] for ch in CHANNELS}
        out = {
            'stage': 'P5.15 Step 3.2/3.3(a) item (1) - s31c rho and residual extraction (zero solves)',
            'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 13',
            'source_run': os.path.relpath(RUN, REPO),
            'instance': json.load(open(os.path.join(RUN, 'g_baseline.json')))['instance'],
            'rule_definitions': {
                'ratios': 'production _update_admm_penalties: primal_selected = max(primal_max/tol_consensus[c], '
                          'primal_mean/tol_consensus[c_mean]); dual = dual_mean/tol_stationarity[c]',
                'freeze': 'held when primal_selected <= 1 and dual <= 1 (adaptation_converged)',
                'increase': 'primal_selected > increase_threshold * dual', 'decrease': 'dual > decrease_threshold * primal_selected',
                'rho_reconstruction': 'rho_in_force(1)=harness RHO; rho(k+1)=clamp(rho(k)*factor(action_k),[min,max])',
                'crawl_window': f'cycles {CRAWL_FROM}..{n} (first residual convergence at cycle {CRAWL_FROM})',
            },
            'initial_rho_harness': rho0,
            'initial_rho_case_file': {ch: params['rho'][ch] for ch in CHANNELS},
            'penalty_update_case_file': upd,
            'rho_change_cycles': change_cycles,
            'rho_terminal_in_force': {ch: rows[-1][ch]['rho_in_force'] for ch in CHANNELS},
            'summary_cycles_1_to_17': window(1, CRAWL_FROM - 1),
            'summary_crawl': window(CRAWL_FROM, n),
            'cross_check_log_ratios_vs_trajectory_max_abs_diff': worst,
            'cross_check_log_ratio_rounding_bound': 5.0e-4,
            'cross_check_logged_action_vs_recomputed_rule_mismatches': action_mismatches,
            'cross_check_rho_in_cycle_observations': len(in_cycle_rho),
            'cross_check_rho_mismatches': rho_mismatch,
            'per_cycle': rows,
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(f'guard verify failed: {failures}')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, 'w') as handle:
        json.dump(out, handle, indent=1)
    show = {k: out[k] for k in ('initial_rho_harness', 'initial_rho_case_file', 'rho_change_cycles', 'rho_terminal_in_force',
                                'summary_cycles_1_to_17', 'summary_crawl', 'cross_check_log_ratios_vs_trajectory_max_abs_diff',
                                'cross_check_logged_action_vs_recomputed_rule_mismatches',
                                'cross_check_rho_in_cycle_observations', 'cross_check_rho_mismatches', 'solve_profile_guard')}
    print(json.dumps(show, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
