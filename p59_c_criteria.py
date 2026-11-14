"""
Stage P5.9-C -- convergence criterion audit.

NOTHING IS CHANGED.  No stopping criterion, tolerance or parameter is modified
anywhere in this stage.  It reads the per-cycle records the A/B/D runs already
produced and reports what production's existing criteria actually do once the
objective is rescaled.

The criteria, as implemented:

    cycle_convergence = residual_convergence AND objective_convergence
    residual_convergence = check_consensus_convergence AND check_stationary_convergence
                           (max and mean primal per family vs tol['consensus'],
                            mean dual per family vs tol['stationarity'])
    objective_convergence = |change in net operational recourse| <= objective_tolerance
    objective_tolerance   = max(tol.objective.abs, tol.objective.rel * recourse)
                          = max(1e3, 1e-3 * recourse)

"Binding" is defined here without ambiguity: on a cycle where every test passes,
the BINDING test is the one with the least relative slack -- the one that would
have failed first had the thresholds been tightened uniformly.  Slack is
measured as threshold/observed, so a slack of 1.0 is exactly at the boundary and
larger is looser.  This makes "which criterion terminates the ADMM" a measured
quantity rather than a reading of the code.

The question is only meaningful on the cycle that actually ENDED the loop, so
the headline counts are restricted to cycles with `cycle_convergence = True`.
Statistics over all recorded cycles are reported separately and are a different
quantity: they mix terminating cycles with the intermediate ones, where the
tests are legitimately unsatisfied.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_c_criteria.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p59_eval as EV  # noqa: E402

OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_c_criteria.json')
SOURCES = ('p59_a_sweep.json', 'p59_a_refine.json', 'p59_b_adaptive.json',
           'p59_d_replay.json', 'p59_d2_signal_base.json',
           'p59_d3_largesignal_base.json', 'p59_e2_flexibility.json')
PLANNING_SIGNAL = 33031.0        # P5.6-D best-to-second candidate gap
TAU_NUMERICAL = 10.0


def _walk_cycles(node, label, out):
    """Collect every recorded cycle from an arbitrarily nested stage report."""
    if isinstance(node, dict):
        if 'cycle_detail' in node and isinstance(node['cycle_detail'], list):
            tag = node.get('case_id') or label
            for cycle in node['cycle_detail']:
                if isinstance(cycle, dict) and 'cycle' in cycle:
                    out.append({'case': tag, **cycle})
        for key, value in node.items():
            if key != 'cycle_detail':
                _walk_cycles(value, node.get('case_id') or label, out)
    elif isinstance(node, list):
        for item in node:
            _walk_cycles(item, label, out)


def slack_profile(cycle):
    """Relative slack of each criterion on one cycle. Larger = looser."""
    slacks = {}
    for group in ('v', 'pf', 'ess'):
        ratio = cycle.get(f'primal_{group}_ratio')
        if ratio:
            slacks[f'consensus_{group}'] = 1.0 / ratio
        mean_ratio = cycle.get(f'primal_{group}_mean_ratio')
        if mean_ratio:
            slacks[f'consensus_{group}_mean'] = 1.0 / mean_ratio
        dual = cycle.get(f'dual_{group}_mean_ratio')
        if dual:
            slacks[f'stationarity_{group}'] = 1.0 / dual
    change = cycle.get('objective_change_abs')
    tolerance = cycle.get('objective_tolerance')
    if change and tolerance:
        slacks['objective'] = tolerance / change
    return slacks


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.9-C',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'nothing_changed': ('no stopping criterion, tolerance or '
                                  'parameter was modified in this stage'),
              'criteria_as_implemented': {
                  'cycle_convergence': 'residual_convergence AND objective_convergence',
                  'residual_convergence': ('consensus (max and mean primal per '
                                           'family) AND stationarity (mean dual '
                                           'per family)'),
                  'objective_tolerance': 'max(1e3, 1e-3 * net operational recourse)'},
              'planning_signal': PLANNING_SIGNAL,
              'tau_numerical': TAU_NUMERICAL,
              'sources': [], 'cycles_analysed': 0}

    cycles = []
    for name in SOURCES:
        path = os.path.join(EV.OUT_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            data = json.load(handle)
        collected = []
        _walk_cycles(data, name, collected)
        report['sources'].append({'file': name, 'cycles': len(collected)})
        cycles.extend(collected)

    if not cycles:
        print('[P5.9-C] no per-cycle evidence found; run A/B/D first.')
        sys.exit(1)
    report['cycles_analysed'] = len(cycles)

    # ---- 1. which criterion actually terminates the ADMM -------------------
    binding_counts, binding_counts_all, rows = {}, {}, []
    for cycle in cycles:
        slacks = slack_profile(cycle)
        if not slacks:
            continue
        binding = min(slacks, key=slacks.get)
        binding_counts_all[binding] = binding_counts_all.get(binding, 0) + 1
        if cycle.get('cycle_convergence'):
            binding_counts[binding] = binding_counts.get(binding, 0) + 1
        rows.append({
            'case': cycle.get('case'), 'cycle': cycle.get('cycle'),
            'binding_criterion': binding,
            'binding_slack': slacks[binding],
            'objective_slack': slacks.get('objective'),
            'objective_change_abs': cycle.get('objective_change_abs'),
            'objective_tolerance': cycle.get('objective_tolerance'),
            'residual_convergence': cycle.get('residual_convergence'),
            'objective_convergence': cycle.get('objective_convergence'),
            'cycle_convergence': cycle.get('cycle_convergence'),
            'slacks': slacks})
    report['binding_criterion_counts_on_terminating_cycles'] = dict(
        sorted(binding_counts.items(), key=lambda kv: -kv[1]))
    report['binding_criterion_counts_all_cycles'] = dict(
        sorted(binding_counts_all.items(), key=lambda kv: -kv[1]))
    report['n_terminating_cycles'] = sum(binding_counts.values())
    report['per_cycle'] = rows

    # ---- 2. is each criterion meaningful after rescaling -------------------
    terminating = [r for r in rows if r.get('cycle_convergence')]

    def stat(key, subset=None):
        source = terminating if subset == 'terminating' else rows
        values = [r['slacks'][key] for r in source if key in r['slacks']]
        if not values:
            return None
        values.sort()
        return {'n': len(values), 'min': values[0], 'median': values[len(values) // 2],
                'max': values[-1],
                'n_at_or_inside_boundary': sum(1 for v in values if v <= 1.0)}
    report['slack_statistics_terminating_cycles'] = {
        key: stat(key, 'terminating') for key in
        ('consensus_v', 'consensus_pf', 'consensus_ess', 'consensus_v_mean',
         'consensus_pf_mean', 'consensus_ess_mean', 'stationarity_v',
         'stationarity_pf', 'stationarity_ess', 'objective')}
    report['slack_statistics_all_cycles'] = {
        key: stat(key) for key in
        ('consensus_v', 'consensus_pf', 'consensus_ess', 'consensus_v_mean',
         'consensus_pf_mean', 'consensus_ess_mean', 'stationarity_v',
         'stationarity_pf', 'stationarity_ess', 'objective')}

    # ---- 3. the objective tolerance against the decision it feeds ----------
    tolerances = [c['objective_tolerance'] for c in cycles
                  if c.get('objective_tolerance')]
    changes = [c['objective_change_abs'] for c in cycles
               if c.get('objective_change_abs')]
    report['objective_tolerance_vs_signal'] = {
        'tolerance_min': min(tolerances) if tolerances else None,
        'tolerance_max': max(tolerances) if tolerances else None,
        'planning_signal': PLANNING_SIGNAL,
        'tolerance_over_planning_signal': (
            (min(tolerances) / PLANNING_SIGNAL) if tolerances else None),
        'observed_recourse_changes_min': min(changes) if changes else None,
        'observed_recourse_changes_max': max(changes) if changes else None,
        'n_changes_larger_than_planning_signal': sum(
            1 for c in changes if c > PLANNING_SIGNAL),
        'n_changes_accepted_as_converged': sum(
            1 for c in cycles if c.get('objective_convergence') and
            (c.get('objective_change_abs') or 0) > PLANNING_SIGNAL),
        'interpretation': ('a cycle counted in the last row was declared '
                           'objective-converged while the recourse was still '
                           'moving by more than the gap the planning decision '
                           'has to resolve')}

    report['recommendations'] = [
        {'option': 'absolute floor tied to the planning signal',
         'form': 'objective_tolerance = max(tau_planning_floor, rel * recourse)',
         'note': ('the current relative term scales with the recourse LEVEL '
                  '(8.3e8) rather than with the decision resolution (3.3e4), so '
                  'it is insensitive to the only quantity the criterion feeds; '
                  'a floor makes the test mean what the planner needs. Values '
                  'deliberately not selected here.')},
        {'option': 'per-subproblem local KKT residual in base-objective units',
         'form': 'read IPOPT unscaled dual infeasibility from the solver logs',
         'note': ('already produced by every solve and already parsed by '
                  'p58_rescale.read_log_since; P5.8-B measured it 390-1488x '
                  'tighter under RESCALED, so after rescaling it is a '
                  'meaningful optimality signal rather than a scaling artefact')},
        {'option': 'base-objective improvement between cycles',
         'form': 'track the base objective separately from the augmented one',
         'note': ('the augmented objective mixes economics with consensus '
                  'penalties; the planning problem only cares about the base '
                  'term')},
        {'option': 'consecutive objective stabilization, made explicit',
         'form': 'reset consecutive_converged_cycles on a warm start',
         'note': ('P5.8-A0 showed the counter is restored from initial_state '
                  '(shared_resources_planning.py:2093-2094), so a persistence '
                  'requirement binds only on the first run of a chain; P5.9-A '
                  'found rho is inherited through the same channel')},
    ]

    with open(OUT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('[P5.9-C] convergence criterion audit')
    print(f"      cycles analysed: {report['cycles_analysed']}")
    print(f"      terminating cycles: {report['n_terminating_cycles']}")
    print('\n      binding criterion ON TERMINATING CYCLES (least relative slack):')
    for key, count in report['binding_criterion_counts_on_terminating_cycles'].items():
        print(f'        {key:26} {count:4d}')
    print('\n      slack statistics on TERMINATING cycles '
          '(threshold/observed; 1.0 = at the boundary):')
    for key, value in report['slack_statistics_terminating_cycles'].items():
        if value:
            print(f"        {key:26} n={value['n']:4d}  min={value['min']:.3g}  "
                  f"median={value['median']:.3g}  max={value['max']:.3g}")
    ot = report['objective_tolerance_vs_signal']
    print(f"\n      objective tolerance {ot['tolerance_min']:.0f}..{ot['tolerance_max']:.0f} "
          f"vs planning signal {PLANNING_SIGNAL:.0f} "
          f"({ot['tolerance_over_planning_signal']:.1f}x)")
    print(f"      cycles declared converged while still moving > signal: "
          f"{ot['n_changes_accepted_as_converged']}")
    print(f'\n[P5.9-C] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
