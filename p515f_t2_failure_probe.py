"""
WORKER T2 -- deliberately triggered network-failure snapshot + no-abort probe,
for the P5.15-F ESSO log-handling fix (PLANNER_BRIEF_2026-09-13.md Addendum 5 /
P5_15_G1_G4_BLOCKED.md).

Part A: force exactly one TSO block to fail (max_iter=1, injected via a
TEST-HARNESS monkeypatch of network._run_smopf_solver_attempt -- NOT a
production change, and not applied to any other block) inside a 1-cycle
run_operational_planning(type='distributed') launched from a DIFFERENT cwd
(os.chdir to a temp dir AFTER the planning object is constructed/loaded at the
repo root, exactly reproducing the original crash sequence: construct with an
absolute results_dir -- item 4 of this fix -- then chdir, then hit a failure).
Required: a FrozenSMOPF pickle is written under the absolute results dir, and
the run completes the cycle without an exception.

Part B: unit-test the no-abort path directly: call
shared_resources_planning._save_frozen_network_block with an unwritable
save_dir (a path that walks through a plain FILE, guaranteeing
NotADirectoryError from os.makedirs) and confirm it returns None with a
[WARNING], not an exception.

FrozenSMOPF output is redirected to data/SRP1/Results/P515F/t2_results (a new
result path under the authorized P515F directory), not the production
data/SRP1/Results/FrozenSMOPF tree.
"""

import io
import os
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import network as net_mod  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515F')
os.makedirs(OUT, exist_ok=True)
T2_RESULTS_DIR = os.path.join(OUT, 't2_results')
os.makedirs(T2_RESULTS_DIR, exist_ok=True)

PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]


def run_part_a():
    print('=== T2 Part A: deliberately triggered TSO block failure, different cwd ===')
    # 1. Construct/load the planning object AT THE REPO ROOT (results_dir/logs_dir
    #    become absolute at construction per item 4 -- this is the ordering that
    #    matters: construction happens before any chdir, exactly like production).
    with redirect_stdout(io.StringIO()):
        planning = O.fresh_planning('p515f_t2_failure_probe')
        planning.params.admm.num_max_iters = 1
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

        # Redirect FrozenSMOPF output to the authorized P515F evidence directory
        # instead of the production data/SRP1/Results/FrozenSMOPF tree.
        planning.results_dir = T2_RESULTS_DIR
        planning.transmission_network.results_dir = T2_RESULTS_DIR
        for node_id in planning.distribution_networks:
            planning.distribution_networks[node_id].results_dir = T2_RESULTS_DIR
        sed.results_dir = T2_RESULTS_DIR

    tso_results_dir_before_chdir = planning.transmission_network.results_dir
    tso_results_dir_is_absolute = os.path.isabs(tso_results_dir_before_chdir)

    # 2. Monkeypatch (TEST HARNESS ONLY, not production): force exactly ONE TSO
    #    block (the first (year, day) encountered) to fail via max_iter=1,
    #    applied on both the primary and the recovery attempt so the failure is
    #    not rescued. No other block is affected.
    original_attempt = net_mod._run_smopf_solver_attempt
    target = {}
    forced_calls = {'count': 0}

    def _in_admm_cycle_tso_solve():
        # Only the per-cycle TSO solve (update_transmission_coordination_model_and_solve),
        # never the pre-ADMM initialization solve (create_transmission_network_model,
        # which has NO failure_snapshot_callback wired at all -- forcing a failure
        # there would never exercise _save_frozen_network_block and would instead
        # just abort initialization early).
        frame = sys._getframe(2)
        while frame is not None:
            if (frame.f_code.co_name == 'update_transmission_coordination_model_and_solve'
                    and frame.f_code.co_filename.endswith('shared_resources_planning.py')):
                return True
            frame = frame.f_back
        return False

    def patched_attempt(network, model, params, from_warm_start=False, option_overrides=None, log_suffix=None):
        if network.is_transmission and _in_admm_cycle_tso_solve():
            key = (network.year, network.day)
            if 'block' not in target:
                target['block'] = key
            if key == target['block']:
                overrides = dict(option_overrides or {})
                overrides['max_iter'] = 1
                forced_calls['count'] += 1
                return original_attempt(network, model, params, from_warm_start=from_warm_start,
                                         option_overrides=overrides, log_suffix=log_suffix)
        return original_attempt(network, model, params, from_warm_start=from_warm_start,
                                 option_overrides=option_overrides, log_suffix=log_suffix)

    net_mod._run_smopf_solver_attempt = patched_attempt

    guard = SolveProfileGuard(PERMITTED, label='P5.15-F T2 failure probe').install()

    # 3. NOW change cwd -- reproduces the original crash sequence (a chdir
    #    window opened AFTER construction, inside which a local solve failed and
    #    the failure handler resolved a then-relative results_dir against the
    #    changed cwd).
    tmp_cwd = tempfile.mkdtemp(prefix='p515f_t2_cwd_')
    os.chdir(tmp_cwd)

    completed_without_exception = False
    error = None
    started = time.time()
    try:
        with redirect_stdout(io.StringIO()):
            with R.patched_admm_objectives():
                _c, _results, models, _s, _p, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
        completed_without_exception = True
    except Exception as exc:  # noqa: BLE001 -- this IS the crash/no-crash probe
        error = f'{type(exc).__name__}: {exc}'
    finally:
        wall = time.time() - started
        net_mod._run_smopf_solver_attempt = original_attempt
        guard.uninstall()
        os.chdir(REPO)
        shutil.rmtree(tmp_cwd, ignore_errors=True)

    frozen_dir = os.path.join(T2_RESULTS_DIR, 'FrozenSMOPF')
    frozen_files = sorted(os.listdir(frozen_dir)) if os.path.isdir(frozen_dir) else []
    failure_files = [f for f in frozen_files if f.startswith('failure_TSO')]

    print('cwd_at_chdir_time', tmp_cwd)
    print('tso_results_dir_before_chdir', tso_results_dir_before_chdir)
    print('tso_results_dir_is_absolute', tso_results_dir_is_absolute)
    print('target_block_forced', target.get('block'))
    print('forced_solver_attempt_calls', forced_calls['count'])
    print('completed_without_exception', completed_without_exception)
    print('error', error)
    print('wall_clock_s', wall)
    print('guard_counts', guard.counts)
    print('frozen_smopf_dir', frozen_dir)
    print('frozen_smopf_files', frozen_files)
    print('failure_snapshot_files', failure_files)

    return {
        'tso_results_dir_before_chdir': tso_results_dir_before_chdir,
        'tso_results_dir_is_absolute': tso_results_dir_is_absolute,
        'target_block_forced': target.get('block'),
        'forced_solver_attempt_calls': forced_calls['count'],
        'completed_without_exception': completed_without_exception,
        'error': error,
        'wall_clock_s': wall,
        'guard_counts': guard.counts,
        'frozen_smopf_files': frozen_files,
        'failure_snapshot_files': failure_files,
    }


def run_part_b():
    print('=== T2 Part B: _save_frozen_network_block against an unwritable save_dir ===')
    tmp_root = tempfile.mkdtemp(prefix='p515f_t2_partb_')
    blocker_file = os.path.join(tmp_root, 'blocker')
    with open(blocker_file, 'w') as handle:
        handle.write('this is a FILE, not a directory -- os.makedirs on a path through it must raise')
    unwritable_save_dir = os.path.join(blocker_file, 'FrozenSMOPF')  # walks through a file -> NotADirectoryError

    class _FakeSolverResult:
        class solver:
            termination_condition = 'maxIterations'
            status = 'warning'

    result = srp._save_frozen_network_block(
        model=object(),  # never dereferenced if the exception fires before pickling
        save_dir=unwritable_save_dir,
        agent='TSO',
        network_name='unit_test_network',
        year=2025,
        day='UnitTest',
        cycle=1,
        from_warm_start=False,
        result=_FakeSolverResult(),
        label='failure',
    )

    print('unwritable_save_dir', unwritable_save_dir)
    print('_save_frozen_network_block return value', result)
    ok = (result is None)
    print('no_abort_path_ok (returned None, no exception propagated)', ok)

    shutil.rmtree(tmp_root, ignore_errors=True)
    return {'unwritable_save_dir': unwritable_save_dir, 'return_value': result, 'ok': ok}


if __name__ == '__main__':
    import json
    a = run_part_a()
    b = run_part_b()
    report = {'stage': 'P5.15-F T2', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'part_a_failure_probe': a, 'part_b_unit_test': b}
    with open(os.path.join(OUT, 't2_failure_probe.json'), 'w') as handle:
        json.dump(report, handle, indent=2, default=str)
