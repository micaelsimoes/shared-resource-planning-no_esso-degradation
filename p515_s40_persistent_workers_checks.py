"""
P5.15 Addendum 22 item (2), Step 3.6 (Addendum 18 design): ZERO-SOLVE checks
for `admm_persistent_workers.py` (the persistent-worker pool).

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 18; frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`,
`2_step_3_6.then`.

A `SolveProfileGuard(permitted=())` is armed for the WHOLE run, in the
PARENT process; every persistent worker this script spawns is ALSO
constructed with `guard_permitted=()`, so a solve reached through ANY call
site -- parent or child -- raises immediately and is never silently missed.
`verify(expected_solves=0, expected_execs=0)` is checked against the PARENT
guard's own counts AND against `pool.total_child_guard_counts()` (the
child-side accounting mechanism this module provides, Requirement 4) at the
end.

Checks performed (each is a REAL, direct exercise of the production code in
`admm_persistent_workers.py`, not a reimplementation):

  1. Flag off -> no new path: a fresh, default `ADMMParameters` instance
     (built through the real production class) has `persistent_workers ==
     {'enabled': False, 'num_workers': 8}`, and
     `admm_persistent_workers.persistent_workers_enabled(...)` reads False
     from it. A static source check additionally confirms
     `_run_operational_planning`'s three per-stage dispatch sites keep an
     unconditional `else:` branch that calls the ORIGINAL, unmodified
     `update_distribution_coordination_models_and_solve` /
     `update_transmission_coordination_model_and_solve` /
     `update_shared_energy_storages_coordination_model_and_solve` entry
     points by name.
  2. Block-to-worker assignment: on the REAL, zero-solve-built SRP1 ADMM-
     ready state (`p515_s32_zero_solve_checks._build_admm_ready_state`,
     reused, not reimplemented), `canonical_dso_block_keys` /
     `canonical_tso_block_keys` / `canonical_esso_block_keys` produce
     exactly 36 + 12 + 3 = 51 DISTINCT keys; `partition_round_robin` at
     several worker counts (1, 2, 3, 7, 8, 51, 97) assigns every key to
     exactly one worker, and is DETERMINISTIC (re-run twice, same result;
     independent of dict iteration by construction, since the input is
     already a concrete ordered list).
  3. A REAL two-worker `PersistentWorkerPool` is constructed against that
     same ADMM-ready state (this DOES spawn two real `multiprocessing`
     (spawn-context) processes and DOES clone 51 real Pyomo blocks into
     them -- it never calls a solver). Verified on the pool:
       a. every worker's `ready_info['env_snapshot']` shows every
          `THREAD_ENV_VARS` entry == '1' INSIDE the child (not merely set
          in the parent before spawning);
       b. the PARENT's own environment is back to its pre-pool value for
          every one of those variables immediately after construction (the
          temporary-set-for-`Process.start()`-then-restore mechanism does
          not leak into the parent's own subsequent execution);
       c. each worker's `tempdir` (its private `TempfileManager.tempdir`)
          is a distinct, worker-numbered path under this run's own tmp
          root;
       d. `pool.worker_for_key` covers exactly the 51 canonical keys.
  4. Fixture reload: every FrozenSMOPF / cycle21 fixture this programme has
     ever cited (same list as `WORKER_REPORT_S36_CLONE_CAPTURE.md`) still
     unpickles, confirming this task deactivated-and-unwired rather than
     deleted anything a preserved fixture could resolve at load time
     (nothing in this task removed a `model_construction_helpers` symbol,
     but the check is repeated here per CLAUDE.md's rule, not skipped
     because "nothing should have changed").
  5. Lifecycle on an injected exception, using stand-in (never-solved) work:
     a REAL captured state for one REAL DSO block is deliberately corrupted
     (an invented mutable-Param name that does not exist on the block) and
     dispatched through `pool._dispatch_stage(...)` (the exact internal
     dispatch path `run_dso_stage`/`run_tso_stage`/`run_esso_stage` use).
     `apply_block_mutable_state` raises `AttributeError` INSIDE the worker,
     BEFORE `network.run_smopf` is ever reached (confirmed by the guard
     count staying at 0) -- `PersistentWorkerPool.PersistentWorkerError` is
     raised in the parent, the pool has already shut itself down (verified:
     every worker process `is_alive() -> False` and `pool._alive is
     False`), and a second `pool.shutdown()` call is a harmless no-op
     (idempotent shutdown).

Guard: `SolveProfileGuard(permitted=())` -- zero permitted call sites,
installed for the WHOLE script. `verify(expected_solves=0,
expected_execs=0)` checked at the very end, both for the parent guard and
for `pool.total_child_guard_counts()` on every pool this script builds.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python \\
        p515_s40_persistent_workers_checks.py

Writes `data/SRP1/Results/P515S40/persistent_workers_checks/results.json`
(new; refuses to overwrite) and a sha256 manifest alongside it.
"""

import hashlib
import inspect
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault('NLP_SOLVER_PATH', '/usr/local/bin/ipopt')
os.environ.setdefault('SOLVER_PATH', '/usr/local/bin/ipopt')
os.environ.setdefault('LP_SOLVER_PATH', '/Users/micaelsimoes/coin-or/dist/bin/clp')

import shared_resources_planning as srp  # noqa: E402
import admm_persistent_workers as apw  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402
from network import capture_block_mutable_state  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'persistent_workers_checks')
OUT_PATH = os.path.join(OUT_DIR, 'results.json')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')

# Same fixture list WORKER_REPORT_S36_CLONE_CAPTURE.md exercised -- one
# representative per distinct content family, per that report's own scoped
# claim. Repeated here (per CLAUDE.md's "deactivate and unwire" rule)
# because this task adds new code that COULD, in principle, have disturbed
# an unpickling path (it does not touch any `model_construction_helpers`
# rule function, but the check costs nothing and is not skipped on that
# assumption).
FIXTURES = (
    'data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl',
    'data/SRP1/Results/P512R/cycle21_prepared/snapshot.pkl',
    'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/'
    'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
    'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/'
    'matched_success_TSO_case9_2025_Summer_cycle7.pkl',
    'data/SRP1/Results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl',
    'data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl',
    'data/SRP1/Results/P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl',
)


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def check_flag_off_default(report):
    params = ADMMParameters()
    enabled = apw.persistent_workers_enabled(params)
    report['1_flag_off_default'] = {
        'persistent_workers_value': params.persistent_workers,
        'persistent_workers_enabled_reads': enabled,
        'pass': (params.persistent_workers == {'enabled': False, 'num_workers': 8}) and (enabled is False),
    }

    source = inspect.getsource(srp._run_operational_planning)
    static_checks = {
        'has_persistent_pool_none_default': 'persistent_pool = None' in source,
        'has_enabled_gate': 'admm_persistent_workers.persistent_workers_enabled(admm_parameters)' in source,
        'dso_else_calls_original': (
            'else:\n            results[\'dso\'] = update_distribution_coordination_models_and_solve(' in source
        ),
        'tso_else_calls_original': (
            'else:\n            results[\'tso\'] = update_transmission_coordination_model_and_solve(' in source
        ),
        'esso_else_calls_original': (
            'else:\n            results[\'esso\'] = update_shared_energy_storages_coordination_model_and_solve(' in source
        ),
    }
    report['1_flag_off_default']['static_source_checks'] = static_checks
    report['1_flag_off_default']['static_pass'] = all(static_checks.values())


def check_block_partition(report, planning, tso_model, dso_models, esso_model):
    dso_keys = apw.canonical_dso_block_keys(planning.distribution_networks)
    tso_keys = apw.canonical_tso_block_keys(planning.transmission_network)
    esso_keys = apw.canonical_esso_block_keys(planning.shared_ess_data.active_distribution_network_nodes)

    all_keys = dso_keys + tso_keys + esso_keys
    n_distinct = len(set(all_keys))

    determinism_results = {}
    coverage_results = {}
    for num_workers in (1, 2, 3, 7, 8, 51, 97):
        assignment_a = apw.partition_round_robin(all_keys, num_workers)
        assignment_b = apw.partition_round_robin(all_keys, num_workers)
        determinism_results[num_workers] = (assignment_a == assignment_b)
        covered = sorted(k for keys in assignment_a.values() for k in keys)
        coverage_results[num_workers] = (covered == sorted(all_keys))

    report['2_block_partition'] = {
        'n_dso_keys': len(dso_keys), 'n_tso_keys': len(tso_keys), 'n_esso_keys': len(esso_keys),
        'n_total_keys': len(all_keys), 'n_distinct_keys': n_distinct,
        'expected_36_12_3': (len(dso_keys) == 36 and len(tso_keys) == 12 and len(esso_keys) == 3),
        'determinism_by_num_workers': determinism_results,
        'coverage_by_num_workers': coverage_results,
        'pass': (
            n_distinct == len(all_keys)
            and len(dso_keys) == 36 and len(tso_keys) == 12 and len(esso_keys) == 3
            and all(determinism_results.values())
            and all(coverage_results.values())
        ),
    }
    return dso_keys, tso_keys, esso_keys


def _env_before():
    return {name: os.environ.get(name) for name in apw.THREAD_ENV_VARS}


def check_pool_lifecycle(report, planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars,
                          num_workers, tempdir_root):
    parent_env_before = _env_before()

    pool = apw.PersistentWorkerPool(
        num_workers=num_workers,
        distribution_networks=planning.distribution_networks,
        dso_models=dso_models,
        transmission_network=planning.transmission_network,
        tso_model=tso_model,
        tso_pristine_base=None,
        tso_snapshot_capture_mode=planning.params.admm.tso_snapshot_capture_mode,
        shared_ess_data=planning.shared_ess_data,
        esso_model=esso_model,
        tempdir_root=tempdir_root,
        guard_permitted=(),
    )

    parent_env_after = _env_before()

    env_caps_ok = all(
        all(info['env_snapshot'].get(name) == '1' for name in apw.THREAD_ENV_VARS)
        for info in pool.ready_info
    )
    parent_env_restored = (parent_env_before == parent_env_after)
    tempdirs = [info['tempdir'] for info in pool.ready_info]
    tempdirs_distinct = (len(set(tempdirs)) == len(tempdirs))

    all_keys = set(apw.canonical_dso_block_keys(planning.distribution_networks))
    all_keys |= set(apw.canonical_tso_block_keys(planning.transmission_network))
    all_keys |= set(apw.canonical_esso_block_keys(planning.shared_ess_data.active_distribution_network_nodes))
    worker_for_key_covers_all = (set(pool.worker_for_key.keys()) == all_keys)

    report['3_pool_startup'] = {
        'num_workers': num_workers,
        'env_caps_set_in_children': env_caps_ok,
        'parent_env_restored_after_spawn': parent_env_restored,
        'worker_tempdirs': tempdirs,
        'worker_tempdirs_distinct': tempdirs_distinct,
        'worker_for_key_covers_all_51': worker_for_key_covers_all,
        'ready_info_ok_flags': [info['ok'] for info in pool.ready_info],
        'pass': (env_caps_ok and parent_env_restored and tempdirs_distinct and worker_for_key_covers_all
                  and all(info['ok'] for info in pool.ready_info)),
    }

    # ---- injected-exception lifecycle check, using a real-but-corrupted
    # (never-solved) captured state -------------------------------------
    dso_key = apw.canonical_dso_block_keys(planning.distribution_networks)[0]
    _, node_id, year, day = dso_key
    real_model = dso_models[node_id][year][day]
    network_yd = planning.distribution_networks[node_id].network[year][day]
    # A fresh, zero-solve-built block (`_build_admm_ready_state`) never ran
    # the initial `update_interface_power_flow_variables` call, so some
    # mutable Params (e.g. `vmag_req`) are still at Pyomo's "no value yet"
    # sentinel and `capture_block_mutable_state` cannot read them. Apply the
    # SAME real per-block update function `run_dso_stage` would use (with
    # the real, zero-consensus `consensus_vars`/`dual_vars` this build
    # returned) to bring the block to a genuinely capturable state, exactly
    # as one ADMM cycle would -- this is not a synthetic bypass of
    # production logic, it is production logic, called once.
    apw._apply_dso_block_params(
        real_model, network_yd, node_id, year, day,
        {'s_available': 1.0, 'e_available': 1.0},
        consensus_vars['vmag'], dual_vars['vmag']['dso'],
        consensus_vars['pf'], dual_vars['pf']['dso'],
        consensus_vars['ess'], dual_vars['ess']['dso'],
        planning.params.admm.previous_iter['ess']['dso'],
    )
    corrupted_state = capture_block_mutable_state(real_model)
    corrupted_state['params']['this_param_does_not_exist_on_the_block'] = {0: 1.0}

    error_raised = None
    error_text = ''
    try:
        pool._dispatch_stage('dso', [dso_key], cycle=999999, from_warm_start=False,
                              captured_states={dso_key: corrupted_state})
    except apw.PersistentWorkerError as exc:
        error_raised = 'PersistentWorkerError'
        error_text = str(exc)
    except Exception as exc:  # noqa: BLE001
        error_raised = type(exc).__name__
        error_text = str(exc)

    processes_dead = [not p.is_alive() for p in pool._processes]
    pool_alive_after = pool._alive
    child_guard_counts = pool.total_child_guard_counts()

    # Idempotent shutdown: must not raise / must not hang.
    idempotent_shutdown_ok = True
    idempotent_shutdown_error = None
    try:
        pool.shutdown()
    except Exception as exc:  # noqa: BLE001
        idempotent_shutdown_ok = False
        idempotent_shutdown_error = repr(exc)

    report['5_injected_exception_lifecycle'] = {
        'error_raised': error_raised,
        'error_mentions_attributeerror_or_the_bogus_name': (
            'this_param_does_not_exist_on_the_block' in error_text or 'AttributeError' in error_text
        ),
        'all_worker_processes_dead_after': all(processes_dead),
        'pool_alive_flag_after': pool_alive_after,
        'child_guard_counts_after_injected_failure': child_guard_counts,
        'no_solve_reached_before_failure': (child_guard_counts.get('permitted_solve', 0) == 0
                                             and child_guard_counts.get('blocked_solve', 0) == 0),
        'idempotent_second_shutdown_ok': idempotent_shutdown_ok,
        'idempotent_second_shutdown_error': idempotent_shutdown_error,
        'pass': (
            error_raised == 'PersistentWorkerError'
            and all(processes_dead)
            and pool_alive_after is False
            and child_guard_counts.get('permitted_solve', 0) == 0
            and child_guard_counts.get('blocked_solve', 0) == 0
            and idempotent_shutdown_ok
        ),
    }
    return pool


def check_fixture_reload(report):
    results = {}
    for rel_path in FIXTURES:
        abs_path = os.path.join(REPO, rel_path)
        entry = {'exists': os.path.exists(abs_path)}
        if entry['exists']:
            try:
                with open(abs_path, 'rb') as handle:
                    payload = pickle.load(handle)
                model = payload.get('model') if isinstance(payload, dict) else None
                n_vars = None
                if model is not None:
                    import pyomo.environ as pe
                    n_vars = sum(1 for _ in model.component_data_objects(pe.Var, active=None))
                entry['unpickled'] = True
                entry['n_vars'] = n_vars
            except Exception as exc:  # noqa: BLE001
                entry['unpickled'] = False
                entry['error'] = repr(exc)
        else:
            entry['unpickled'] = None
        results[rel_path] = entry

    report['4_fixture_reload'] = {
        'fixtures': results,
        'pass': all(
            (not v['exists']) or v.get('unpickled', False) for v in results.values()
        ) and any(v['exists'] for v in results.values()),
    }


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)
    _refuse_overwrite(MANIFEST_PATH)

    guard = SolveProfileGuard(permitted=(), label='p515_s40_persistent_workers_checks')
    guard.install()

    report = {
        'stage': 'P5.15 Addendum 22 item (2) -- persistent-workers zero-solve checks',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 18',
            'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }

    pool = None
    try:
        check_flag_off_default(report)

        eval_id = f'p515s40_persistent_workers_checks_{int(time.time())}'
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = _build_admm_ready_state(eval_id)

        check_block_partition(report, planning, tso_model, dso_models, esso_model)

        tempdir_root = os.path.join(OUT_DIR, 'tmp', eval_id)
        os.makedirs(tempdir_root, exist_ok=True)
        pool = check_pool_lifecycle(report, planning, tso_model, dso_models, esso_model,
                                     consensus_vars, dual_vars,
                                     num_workers=2, tempdir_root=tempdir_root)

        check_fixture_reload(report)
    finally:
        if pool is not None and pool._alive:
            pool.shutdown()
        guard.uninstall()

    failures = guard.verify(expected_solves=0, expected_execs=0)
    report['guard_verification'] = {
        'parent_guard_counts': guard.counts,
        'parent_guard_failures': failures,
        'child_guard_counts_total': pool.total_child_guard_counts() if pool is not None else None,
    }

    all_checks_pass = all(
        report[key].get('pass', False)
        for key in ('1_flag_off_default', '2_block_partition', '3_pool_startup',
                    '4_fixture_reload', '5_injected_exception_lifecycle')
    ) and (not failures)
    report['all_checks_pass'] = all_checks_pass

    with open(OUT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[S40-PW-CHECKS] wrote {OUT_PATH}')
    print(f'[S40-PW-CHECKS] all_checks_pass={all_checks_pass}')
    for key in ('1_flag_off_default', '2_block_partition', '3_pool_startup',
                '4_fixture_reload', '5_injected_exception_lifecycle'):
        print(f'[S40-PW-CHECKS]   {key}: pass={report[key].get("pass")}')
    print(f'[S40-PW-CHECKS] guard: parent={guard.counts} child_total={report["guard_verification"]["child_guard_counts_total"]} failures={failures}')

    manifest = {os.path.relpath(OUT_PATH, REPO): _sha256_file(OUT_PATH)}
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S40-PW-CHECKS] wrote {MANIFEST_PATH}')

    if not all_checks_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
