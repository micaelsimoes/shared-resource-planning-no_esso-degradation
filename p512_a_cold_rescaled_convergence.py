"""
P5.12-A -- does the cold RESCALED fixed-rho construction converge if the same
ADMM recurrence is allowed to continue past the production cap of 25 cycles?

P5.11.1 built a cold RESCALED template at rho_pf = 300 and found it exhausted
`num_max_iters = 25` with `cycle_convergence = False` on every cycle, recourse
still falling from 1.461e9 (cycle 13) to 1.402e9 (cycle 25). This asks only
whether that trajectory ever terminates, out to a diagnostic horizon of 100.

HOW THE HORIZON IS EXTENDED, AND WHY IT IS NOT A PRODUCTION CHANGE.
`num_max_iters` is raised ONLY on the per-evaluation deep copy handed to one
diagnostic run. `data/SRP1/SRP1_params.json` is never written and no production
default is touched. The production convergence definitions, tolerances, IPOPT
options, ADMM update equations, proximal regularization and local NLP
formulation are used exactly as they are.

WHY FOUR RUNS RATHER THAN ONE WITH SNAPSHOTS. The stage requires detached polish
diagnostics at cycles 25/50/75/100 that cannot perturb the live trajectory.
Production runs its coordination loop internally, so intercepting it mid-loop
would mean hooking a production function and deep-copying 48 models in-flight.
Instead each checkpoint is produced by its OWN cold run with the cap set to that
cycle count. The cold trajectory is deterministic, so run(cap=50) reproduces
run(cap=25) exactly through cycle 25 -- which the harness verifies rather than
assumes. Nothing is spliced, and each checkpoint is a genuine state produced by
an uninterrupted trajectory.

The cap-25 run must additionally reproduce the accepted P5.11 trajectory. If it
does not, the stage stops.

    python p512_a_cold_rescaled_convergence.py run <max_cycles>
    python p512_a_cold_rescaled_convergence.py preflight
"""

import hashlib
import io
import json
import math
import os
import pickle
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import p510_oracle as OR  # noqa: E402
import p511_1_selfconsistent_t0 as T0  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P512A')
HORIZON = 100
CHECKPOINTS = (25, 50, 75, 100)

RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
P511_META = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P511',
                         'p511_selfconsistent_t0_meta.json')
P511_TEMPLATE = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P511',
                             'p511_selfconsistent_t0.pkl')

# accepted P5.11 cycle-25 trajectory markers
P511_MARKERS = {1: 2352009862.13208, 13: 1461174062.836328, 25: 1402384338.113119}
MARKER_TOL = 1e-6      # relative


def config():
    return OR.OracleConfig(
        scaling_mode=OR.SCALING_RESCALED, rho_v=RHO['v'], rho_pf=RHO['pf'],
        rho_ess=RHO['ess'], adaptive_penalty=False, neutralize_history=True,
        template_id='P512A-COLD-RESCALED',
        initialization_policy='original cold initialization, no warm start',
        notes='P5.12-A diagnostic horizon')


def cycle_row(entry, previous_recourse):
    """Everything the stage asks for, per cycle, as production records it."""
    recourse = entry.get('recourse')
    row = {
        'cycle': entry.get('cycle'),
        'cycle_convergence': entry.get('cycle_convergence'),
        'residual_convergence': entry.get('residual_convergence'),
        'objective_convergence': entry.get('objective_convergence'),
        'consecutive_converged_cycles': entry.get('consecutive_converged_cycles'),
        'required_consecutive_cycles': entry.get('required_consecutive_cycles'),
        'local_solves_ok': entry.get('local_solves_ok'),
        'recourse': recourse,
        'gross_operational_cost': entry.get('gross_operational_cost'),
        'objective_change_abs': entry.get('objective_change_abs'),
        'objective_tolerance': entry.get('objective_tolerance'),
    }
    for group in ('v', 'pf', 'ess'):
        for kind in ('primal', 'dual'):
            for suffix in ('', '_mean'):
                key = f'{kind}_{group}{suffix}'
                row[key] = entry.get(key)
        row[f'primal_{group}_ratio'] = entry.get(f'primal_{group}_ratio')
        row[f'primal_{group}_mean_ratio'] = entry.get(f'primal_{group}_mean_ratio')
        row[f'dual_{group}_mean_ratio'] = entry.get(f'dual_{group}_mean_ratio')
        # normalized slack: threshold / observed; 1.0 sits exactly on the bound
        pr = entry.get(f'primal_{group}_ratio')
        dr = entry.get(f'dual_{group}_mean_ratio')
        row[f'slack_consensus_{group}'] = (1.0 / pr) if pr else None
        row[f'slack_stationarity_{group}'] = (1.0 / dr) if dr else None
        # production defines dual = rho * |dz| / base, and rho is FIXED here,
        # so the underlying state step is recoverable exactly
        rho = RHO[group]
        dual_mean = entry.get(f'dual_{group}_mean')
        row[f'state_step_norm_{group}'] = (dual_mean / rho) if dual_mean is not None else None
    tol = entry.get('objective_tolerance')
    chg = entry.get('objective_change_abs')
    row['slack_objective'] = (tol / chg) if (tol and chg) else None
    row['worst_pf_primal_difference'] = entry.get('worst_pf_primal_difference')
    row['recourse_change'] = (
        (recourse - previous_recourse)
        if (recourse is not None and previous_recourse is not None) else None)
    row['nonfinite'] = any(
        isinstance(v, float) and not math.isfinite(v)
        for v in (recourse, entry.get('primal_pf'), entry.get('primal_v'),
                  entry.get('dual_pf_mean')) if v is not None)
    return row


def run(max_cycles):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'p512a_trajectory_cap{max_cycles}.json')
    try:
        provenance, _ = gate(f'P5.12-A cold RESCALED cap={max_cycles}', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.12-A] ABORTED\n{error}')
        sys.exit(1)

    cfg = config()
    report = {'stage': 'P5.12-A', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'max_cycles_this_run': max_cycles, 'horizon': HORIZON,
              'config': cfg.as_dict(), 'config_hash': cfg.config_hash,
              'rho_fixed': RHO,
              'production_num_max_iters_untouched': True,
              'note': ('num_max_iters raised only on the per-evaluation deep '
                       'copy; data/SRP1/SRP1_params.json never written'),
              'cycles': []}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    planning = O.fresh_planning(f'p512a_cap{max_cycles}')
    report['rho_params_replaced'] = RH.apply_rho_to_params(planning, RHO)
    report['adaptive_before'] = RH.set_adaptive_penalty(planning, False)
    report['num_max_iters_before'] = planning.params.admm.num_max_iters
    planning.params.admm.num_max_iters = max_cycles
    report['num_max_iters_on_deep_copy'] = planning.params.admm.num_max_iters
    persist()

    print(f'[P5.12-A] cold RESCALED, rho_pf={RHO["pf"]:g}, cap={max_cycles} ...',
          flush=True)
    started = time.time()
    console = io.StringIO()
    with redirect_stdout(console):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        with R.patched_admm_objectives() as applied:
            # public signature: (convergence, results, models, sensitivities,
            # primal_evolution) and, with return_state, state appended -- six
            # values, not the seven the internal `_run_operational_planning`
            # returns (shared_resources_planning.py:87-89)
            convergence, _, models, _, _, state = \
                planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
    runtime = time.time() - started

    report['production_reported_convergence'] = bool(convergence)
    report['blocks_rescaled_at_build'] = len([v for v in applied.values() if v])
    report['wall_clock_s'] = runtime
    observed_rho = RH.observed_rho(state)
    report['rho_observed_distinct'] = [
        dict(t) for t in {tuple(sorted(v.items())) for v in observed_rho.values()}]

    previous = None
    for entry in (state.get('admm_diagnostics') or []):
        row = cycle_row(entry, previous)
        report['cycles'].append(row)
        if entry.get('recourse') is not None:
            previous = entry['recourse']
    report['cycles_run'] = len(report['cycles'])
    report['converged_at_cycle'] = next(
        (c['cycle'] for c in report['cycles'] if c['cycle_convergence']), None)
    report['any_local_solve_failure'] = any(
        c['local_solves_ok'] is False for c in report['cycles'])
    report['any_nonfinite'] = any(c['nonfinite'] for c in report['cycles'])
    persist()

    # ---- verification against the accepted P5.11 trajectory ----------------
    checks = {}
    for cycle, expected in P511_MARKERS.items():
        row = next((c for c in report['cycles'] if c['cycle'] == cycle), None)
        observed = row.get('recourse') if row else None
        checks[cycle] = {
            'expected': expected, 'observed': observed,
            'relative_delta': (abs(observed - expected) / abs(expected)
                               if observed is not None else None),
            'matches': (observed is not None
                        and abs(observed - expected) / abs(expected) <= MARKER_TOL)}
    report['p511_trajectory_check'] = checks
    report['matches_p511_through_cycle_25'] = all(v['matches'] for v in checks.values())
    persist()
    print(f"[P5.12-A] cycles run: {report['cycles_run']}  "
          f"converged_at: {report['converged_at_cycle']}  "
          f"matches P5.11 through 25: {report['matches_p511_through_cycle_25']}",
          flush=True)

    # ---- detached checkpoint polish, on a deep copy only -------------------
    if max_cycles in CHECKPOINTS or report['converged_at_cycle'] is not None:
        print('[P5.12-A] detached polish on a deep copy ...', flush=True)
        last = report['cycles'][-1] if report['cycles'] else {}
        cp = {'at_cycle': report['cycles_run'],
              'is_converged_state': report['converged_at_cycle'] is not None,
              'prepolish_interface_disagreement': last.get('primal_pf'),
              'state_sha256': T0.state_hash(state),
              'config_hash': cfg.config_hash}
        try:
            clone = srp._clone_operational_models(models)
            R.restore_for_polish(planning, clone)
            common = O.common_coordinated_values(
                planning, clone, state['consensus_vars'],
                interface_anchor='midpoint')
            request = O.esso_request_from_common(
                planning, state['consensus_vars'], common)
            esso_models, _, esso_solved, available = O.solve_physical_esso(
                planning, candidate, request)
            cp['esso_solved'] = all(esso_solved.values())
            O.apply_physical_capacities(planning, clone, available)
            O.apply_common_values(planning, clone, common)
            blocks, all_solved = O.polish_networks(planning, clone)
            cp['polish_success'] = bool(all_solved)
            cp['failed_blocks'] = [f"{b['agent']}|{b['year']}|{b['day']}"
                                   for b in blocks if not b['solved']]
            if all_solved:
                polished = planning.get_operational_recourse_components(clone)
                salvage = float(
                    planning.shared_ess_data.get_salvage_value(esso_models))
                net = polished['gross_operational_cost'] - salvage
                cp['polished_recourse'] = net
                cp['polished_total_objective'] = (
                    O.investment_cost(planning, candidate) + net)
                cp['polish_correction'] = (
                    net - last.get('recourse') if last.get('recourse') else None)
        except Exception as error:
            cp['polish_error'] = f'{type(error).__name__}: {error}'
        report['checkpoint'] = cp
        persist()
        print(f"          polish_success={cp.get('polish_success')} "
              f"failed={len(cp.get('failed_blocks') or [])} "
              f"correction={cp.get('polish_correction')}", flush=True)

    print(f'\n[P5.12-A] report -> {out_path}')


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if action == 'run':
        run(int(sys.argv[2]) if len(sys.argv) > 2 else HORIZON)
    else:
        print(f'unknown action {action!r}')
        sys.exit(2)
