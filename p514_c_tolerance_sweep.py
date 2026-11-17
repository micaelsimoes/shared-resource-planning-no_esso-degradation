"""
Track C1 — objective-tolerance decade sweep, rel 1e-3 -> 1e-4, cold and warm.

Frozen spec: data/SRP1/Results/P514C/frozen_c0_tolerance_sweep_v1_43da5cb6.json

The tolerance is overridden IN THE HARNESS on the in-memory planning problem.
data/SRP1/SRP1_params.json is never edited: every preserved result must stay
reproducible from the tree.

Configuration is inherited from the 2x2's ADAPTIVE cells (rho_pf 300, adaptive on), so
the tolerance is the only varied setting and the 2x2 cells are the 1e-3 baseline.

    python p514_c_tolerance_sweep.py <cell>      # cell in {cold, warm}
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p510_oracle as OR  # noqa: E402
import p512_a_cold_rescaled_convergence as A  # noqa: E402
import p56a_candidates as CAND  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514C')
SPEC = 'data/SRP1/Results/P514C/frozen_c0_tolerance_sweep_v1_43da5cb6.json'
CAP = 90
REL = 1e-4
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
SOLVES_PER_CYCLE = 51
UPPER_BOUND = CAP * SOLVES_PER_CYCLE + SOLVES_PER_CYCLE
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]


def patch_planning():
    """Cap and objective-relative tolerance, applied to every planning problem built."""
    original = O.fresh_planning

    def patched(eval_id, *args, **kwargs):
        planning = original(eval_id, *args, **kwargs)
        planning.params.admm.num_max_iters = CAP
        planning.params.admm.tol['objective']['rel'] = REL
        return planning

    O.fresh_planning = patched
    return original


def main(cell):
    if cell not in ('cold', 'warm'):
        raise SystemExit("cell must be 'cold' or 'warm'")
    os.makedirs(OUT, exist_ok=True)
    guard = SolveProfileGuard(PERMITTED, label=f'C1 {cell}').install()
    original_fresh = patch_planning()
    started = time.time()
    effective = {}
    try:
        with redirect_stdout(io.StringIO()):
            if cell == 'cold':
                planning = O.fresh_planning('c1_cold')
                RH.apply_rho_to_params(planning, RHO)
                RH.set_adaptive_penalty(planning, True)
                effective = {'num_max_iters': planning.params.admm.num_max_iters,
                             'tol_objective': dict(planning.params.admm.tol['objective']),
                             'tol_stationarity': dict(planning.params.admm.tol['stationarity']),
                             'adaptive_penalty': planning.params.admm.adaptive_penalty}
                candidate = srp._build_positive_bootstrap_candidate(
                    planning, planning.params.benders.positive_bootstrap)
                with R.patched_admm_objectives():
                    _conv, _r, _m, _s, _p, state = planning.run_operational_planning(
                        type='distributed', candidate_solution=deepcopy(candidate),
                        print_results=False, debug_flag=False, return_state=True)
                rows = [A.cycle_row(e, None) for e in (state.get('admm_diagnostics') or [])]
                cycles = rows
                recourse = rows[-1].get('recourse') if rows else None
                terminal_change = rows[-1].get('objective_change_abs') if rows else None
                terminal_tol = rows[-1].get('objective_tolerance') if rows else None
                converged = next((r['cycle'] for r in rows if r['cycle_convergence']), None)
                rho_final = (state.get('admm_diagnostics') or [{}])[-1].get('rho_pf_after')
            else:
                planning = O.fresh_planning('c1_warm_candidate')
                x = CAND.base_vector(planning)
                effective = {'num_max_iters': planning.params.admm.num_max_iters,
                             'tol_objective': dict(planning.params.admm.tol['objective']),
                             'tol_stationarity': dict(planning.params.admm.tol['stationarity']),
                             'adaptive_penalty_before_config': planning.params.admm.adaptive_penalty}
                config = OR.OracleConfig(
                    scaling_mode=OR.SCALING_RESCALED, rho_v=RHO['v'], rho_pf=RHO['pf'],
                    rho_ess=RHO['ess'], adaptive_penalty=True, neutralize_history=True,
                    notes='Track C1 warm cell, objective rel 1e-4')
                template_state, _info = OR.prepare_template(config)
                record, _state = OR.evaluate(x, config, case_id=f'c1_{cell}',
                                             template_state=template_state)
                cycles = record.get('cycle_detail') or []
                recourse = record.get('admm_net_recourse_before_polish')
                term = record.get('terminating_criterion') or {}
                terminal_change = term.get('objective_change_abs')
                terminal_tol = term.get('objective_tolerance')
                converged = len(cycles)
                rho_final = (record.get('rho_observed_final') or {}).get('pf')
    finally:
        O.fresh_planning = original_fresh
        guard.uninstall()

    n = len(cycles)
    changes = [c.get('objective_change_abs') for c in cycles if c.get('objective_change_abs')]
    ratios = [changes[i] / changes[i - 1] for i in range(len(changes) - 4, len(changes))
              if len(changes) > 5 and changes[i - 1]]
    report = {
        'stage': 'Track C1', 'cell': cell, 'spec': SPEC,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'varied_setting': {'tol_objective_rel': REL, 'baseline_was': 1e-3},
        'effective_configuration': effective,
        'instance': 'recovered AB1 candidate (positive bootstrap), C3 active',
        'rho_requested': RHO, 'adaptive_penalty': True, 'cap': CAP,
        'cycles_run': n, 'converged_at_cycle': converged,
        'hit_cap': converged is None and n >= CAP,
        'recourse': recourse,
        'terminal_objective_change_abs': terminal_change,
        'terminal_objective_tolerance': terminal_tol,
        'final_rho_pf': rho_final,
        'realised_decrement_ratio_last4': (sum(ratios) / len(ratios)) if ratios else None,
        'cycles': cycles,
        'solve_profile': {
            'observed': dict(guard.counts), 'upper_bound': UPPER_BOUND,
            'identity_expected': SOLVES_PER_CYCLE * n + SOLVES_PER_CYCLE,
            'identity_holds': guard.counts['permitted_solve'] == SOLVES_PER_CYCLE * n + SOLVES_PER_CYCLE,
            'hard_failures': (['blocked solves'] if guard.counts['blocked_solve'] else [])
                             + (['exceeded upper bound'] if guard.counts['permitted_solve'] > UPPER_BOUND else []),
        },
        'wall_clock_s': time.time() - started,
    }
    path = os.path.join(OUT, f'c1_{cell}_rel1e-4.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[C1 {cell}] cycles={n} converged={converged} hit_cap={report["hit_cap"]} '
          f'recourse={recourse} terminal_change={terminal_change} rho_final={rho_final} '
          f'solves={guard.counts["permitted_solve"]} ratio={report["realised_decrement_ratio_last4"]} '
          f'wall={report["wall_clock_s"]:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'warm'))
