"""
Stage P5.9-B -- adaptive penalty audit at the best A configuration.

NOTHING IS TUNED HERE.  Production's `_update_admm_penalties`
(`shared_resources_planning.py:5440-5520`) is left exactly as it is.  This stage
records what it DOES after rescaling and answers one question: does adaptive rho
balancing still behave correctly?

The control is `adaptive_penalty = False` at the same rho, which is the only way
to separate "the initial penalty was right" from "the adaptation found a good
penalty".  Both arms run the same number of generations from the same frozen T0.

THE STRUCTURAL POINT THE DATA HAS TO SETTLE.  Production measures the dual
residual as

    dual = rho * |z_current - z_prev| / base                (lines 4897, 4938)

i.e. LINEAR IN RHO, and the update rule compares
`primal_ratio = primal/primal_tol` against `dual_ratio = dual_mean/dual_tol`,
increasing rho when `primal_ratio > 5 * dual_ratio` and decreasing it when
`dual_ratio > 5 * primal_ratio` (3 for pf).  Raising rho therefore raises the
measured dual residual proportionally and pushes the rule toward DECREASING it
again.  Whether that negative feedback actually cancels a deliberately raised
penalty is measured below, not asserted.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_b_adaptive.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_b_adaptive.json')
A_PATH = os.path.join(EV.OUT_DIR, 'p59_a_sweep.json')
GENERATIONS = 4


def run_arm(x, label, rho, adaptive, generations=GENERATIONS):
    state, info = EV.load_template(rescale=True, rho=rho)
    out = {'template_info': info, 'adaptive_penalty': adaptive,
           'rho_requested': rho, 'generations': []}
    previous = None
    for j in range(1, generations + 1):
        case_id = f'{label}_j{j}'
        record, new_state = EV.evaluate(
            x, template_state=state, case_id=case_id, rho=rho,
            adaptive_penalty=adaptive, archive_label=case_id)
        row = EV.row(record)
        row['generation'] = j
        objective = row.get('total_objective')
        row['step_delta'] = (objective - previous) if (
            objective is not None and previous is not None) else None
        row['cycle_detail'] = record.get('cycle_detail')
        out['generations'].append(row)
        print(f"      gen {j}: {row['status']:16s} cycles={row['admm_cycles']} "
              f"pre-polish={row['admm_net_recourse_before_polish']} "
              f"Q={objective} fail={row['n_failed_polish_blocks']} "
              f"rho_final={row['rho_observed_final']}", flush=True)
        if objective is not None:
            previous = objective
        if new_state is None:
            row['chain_stopped_here'] = True
            break
        state = new_state
    return out


def rho_trace(arm):
    """Every rho update the run performed, cycle by cycle."""
    trace = []
    for row in arm['generations']:
        for cycle in (row.get('cycle_detail') or []):
            trace.append({
                'generation': row['generation'], 'cycle': cycle.get('cycle'),
                'rho_v': (cycle.get('rho_v_before'), cycle.get('rho_v_after')),
                'rho_pf': (cycle.get('rho_pf_before'), cycle.get('rho_pf_after')),
                'rho_ess': (cycle.get('rho_ess_before'), cycle.get('rho_ess_after')),
                'action_v': cycle.get('rho_v_action'),
                'action_pf': cycle.get('rho_pf_action'),
                'action_ess': cycle.get('rho_ess_action'),
                'primal_v_ratio': cycle.get('primal_v_ratio'),
                'primal_pf_ratio': cycle.get('primal_pf_ratio'),
                'dual_v_mean_ratio': cycle.get('dual_v_mean_ratio'),
                'dual_pf_mean_ratio': cycle.get('dual_pf_mean_ratio'),
            })
    return trace


def arm_summary(arm):
    rows = arm['generations']
    objectives = [r['total_objective'] for r in rows
                  if r.get('total_objective') is not None]
    trace = rho_trace(arm)
    actions = [t for t in trace if t['action_v'] not in (None, 'fixed')
               or t['action_pf'] not in (None, 'fixed')]
    triggered = [t for t in trace
                 if t['action_v'] in ('increased', 'decreased')
                 or t['action_pf'] in ('increased', 'decreased')]
    return {
        'n_generations': len(rows),
        'n_polish_failures': sum(1 for r in rows
                                 if (r.get('n_failed_polish_blocks') or 0) > 0),
        'first_objective': objectives[0] if objectives else None,
        'last_objective': objectives[-1] if objectives else None,
        'total_drift': (objectives[-1] - objectives[0]) if len(objectives) > 1 else None,
        'prepolish': [r.get('admm_net_recourse_before_polish') for r in rows],
        'n_cycles_recorded': len(trace),
        'n_cycles_with_an_update': len(triggered),
        'update_actions_seen': sorted({a for t in trace
                                       for a in (t['action_v'], t['action_pf'])
                                       if a}),
        'rho_first': (trace[0]['rho_v'][0], trace[0]['rho_pf'][0]) if trace else None,
        'rho_last': (trace[-1]['rho_v'][1], trace[-1]['rho_pf'][1]) if trace else None,
        'rho_trace': trace,
        'n_actions_non_fixed': len(actions),
    }


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.9-B adaptive penalty audit', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    with open(A_PATH) as handle:
        a_report = json.load(handle)
    best_row = a_report.get('best_case_row')
    if not best_row:
        print('[P5.9-B] stage A has no selected configuration; not continuing.')
        sys.exit(1)
    best_rho = best_row['rho_requested']
    print(f"[P5.9-B] best A configuration: {a_report['best_case']}  rho={best_rho}\n",
          flush=True)

    x0 = dict(BC.population(planning_gate))['base']
    report = {
        'stage': 'P5.9-B', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'best_configuration_from_A': a_report['best_case'],
        'rho': best_rho,
        'generations_per_arm': GENERATIONS,
        'nothing_tuned': ('production _update_admm_penalties is unmodified; '
                          'the only variable is whether it is enabled'),
        'penalty_update_rules_in_force': a_report['cases'][0].get(
            'rho_snapshot_in_force', {}).get('penalty_update')
        if a_report.get('cases') else None,
        'dual_residual_is_linear_in_rho': (
            'dual = rho * |z_current - z_prev| / base '
            '(shared_resources_planning.py:4897, 4938)'),
        'arms': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.9-B] arm 1: adaptive_penalty = True (production)', flush=True)
    arm_on = run_arm(x0, 'b_adaptive_on', best_rho, adaptive=True)
    report['arms']['adaptive_on'] = arm_on
    report['arms']['adaptive_on_summary'] = arm_summary(arm_on)
    persist()

    print('\n[P5.9-B] arm 2: adaptive_penalty = False (rho pinned)', flush=True)
    arm_off = run_arm(x0, 'b_adaptive_off', best_rho, adaptive=False)
    report['arms']['adaptive_off'] = arm_off
    report['arms']['adaptive_off_summary'] = arm_summary(arm_off)
    persist()

    on, off = report['arms']['adaptive_on_summary'], report['arms']['adaptive_off_summary']
    report['comparison'] = {
        'objective_last_on': on['last_objective'],
        'objective_last_off': off['last_objective'],
        'adaptation_worth': (on['last_objective'] - off['last_objective'])
        if (on['last_objective'] and off['last_objective']) else None,
        'polish_failures_on': on['n_polish_failures'],
        'polish_failures_off': off['n_polish_failures'],
        'rho_moved_under_adaptation': on['rho_first'] != on['rho_last'],
        'rho_first_on': on['rho_first'], 'rho_last_on': on['rho_last'],
    }
    persist()

    print('\n[P5.9-B] comparison')
    for key, value in report['comparison'].items():
        print(f'      {key:32} {value}')
    print(f'\n[P5.9-B] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
