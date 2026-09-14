"""
WORKER T1 -- two-cycle probe for the P5.15-F ESSO log-handling fix.
PLANNER_BRIEF_2026-09-13.md Addendum 5 / P5_15_G1_G4_BLOCKED.md.

Runs a REAL run_operational_planning(type='distributed') with num_max_iters=2
on the C* control configuration (imported from p514_n_instrumented_cstar.py:
S_INV, E_INV, INVEST_YEAR, BUDGET, RHO, adaptive penalty), via
p56a_oracle.fresh_planning. Diagnostic-only harness (not committed); imports
production functions/constants rather than reimplementing them.

Run from the repo root (T1 does not require a cwd change -- that is T2).
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p514_n_instrumented_cstar as N  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515F'
os.makedirs(OUT, exist_ok=True)

PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]


def main():
    guard = SolveProfileGuard(PERMITTED, label='P5.15-F T1 two-cycle probe').install()
    started = time.time()
    try:
        with redirect_stdout(io.StringIO()):
            planning = O.fresh_planning('p515f_t1_two_cycle')
            planning.params.admm.num_max_iters = 2
            planning.params.admm.tol['objective']['rel'] = N.REL
            planning.shared_ess_data.params.budget = N.BUDGET
            RH.apply_rho_to_params(planning, N.RHO)
            RH.set_adaptive_penalty(planning, True)
            sed = planning.shared_ess_data

            candidate = planning.get_initial_candidate_solution()
            for node_id in sed.active_distribution_network_nodes:
                candidate['investment'][node_id][N.INVEST_YEAR]['s'] = N.S_INV
                candidate['investment'][node_id][N.INVEST_YEAR]['e'] = N.E_INV
            srp._rebuild_candidate_total_capacities(planning, candidate)

            logs_dir_in_force = {
                'planning.logs_dir': planning.logs_dir,
                'shared_ess_data.logs_dir': sed.logs_dir,
                'transmission_network.logs_dir': planning.transmission_network.logs_dir,
            }

            with R.patched_admm_objectives():
                _c, _results, models, _s, _p, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
    finally:
        guard.uninstall()
        wall = time.time() - started

    rows = [N.A.cycle_row(e, None) for e in (state.get('admm_diagnostics') or [])]

    # Per-solve ESSO log files: list the logs_dir.
    logs_dir = sed.logs_dir
    esso_logs = sorted(f for f in os.listdir(logs_dir) if 'esso' in f.lower())

    diagnostics = sed.esso_complementarity_diagnostics
    report = {
        'stage': 'P5.15-F T1',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'wall_clock_s': wall,
        'logs_dir_in_force': logs_dir_in_force,
        'esso_log_files': esso_logs,
        'cycles_run': len(rows),
        'guard_counts': guard.counts,
        'guard_permitted_sites': guard.permitted_sites,
        'esso_complementarity_diagnostics': diagnostics,
    }

    with open(os.path.join(OUT, 't1_two_cycle_probe.json'), 'w') as handle:
        json.dump(report, handle, indent=2, default=str)

    print('WALL_CLOCK_S', wall)
    print('LOGS_DIR', logs_dir)
    print('ESSO_LOG_FILES')
    for f in esso_logs:
        print(' ', f)
    print('GUARD_COUNTS', guard.counts)
    print('N_DIAGNOSTICS_ENTRIES', len(diagnostics))
    for entry in diagnostics:
        print('DIAG', {k: entry.get(k) for k in
                        ('node_id', 'log_path', 'mu_final', 's_obj', 'parse_reason')})


if __name__ == '__main__':
    main()
