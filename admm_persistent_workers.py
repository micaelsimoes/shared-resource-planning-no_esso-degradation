"""
P5.15 Addendum 22 item (2), Step 3.6 (Addendum 18 design): persistent,
single-threaded worker processes for within-cycle ADMM block parallelism.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 18 ("Persistent worker
processes, each owning a fixed subset of blocks; models built once; only
consensus parameters, duals and interface results exchanged per cycle. Each
worker single-threaded... Private Pyomo temp directory and unique log paths
per worker... Results assembled in a fixed block order; ADMM updates
unchanged."), frozen spec v11 (`data/SRP1/Results/P515S40/
frozen_s40_closure_spec_v11_0e9a37be.json`, `2_step_3_6.then`).

DEFAULT OFF. `admm_parameters.persistent_workers['enabled']` (default False,
set in `ADMMParameters.__init__`) is the ONLY switch. With the flag off, this
module is never imported by `shared_resources_planning.py` at call time (the
import is a plain module-level import, but `PersistentWorkerPool` is never
constructed and no function in this module ever runs) -- the existing serial
call sites in `_run_operational_planning` are byte-for-byte unchanged.

--------------------------------------------------------------------------
DESIGN (see WORKER_REPORT_S40_PERSISTENT_WORKERS.md "Design summary" for the
full write-up; this is the short version for a reader of the code only):

- Block granularity: one worker "block" = one (agent, node_id, year, day)
  DSO solve, one (agent, year, day) TSO solve, or one (agent, node_id) ESSO
  solve -- 36 + 12 + 3 = 51 blocks for SRP1. Blocks are assigned to workers
  by a STATIC, DETERMINISTIC round-robin over a fixed canonical ordering
  (`partition_round_robin`), computed ONCE per `_run_operational_planning`
  call and never revisited at runtime (no load-based rebalancing -- that
  would make the assignment a function of wall-clock timing, which is not
  deterministic).
- Models are cloned ONCE per block, at pool construction (mirroring the
  existing `tso_pristine_base` pattern, `shared_resources_planning.py`),
  and live in the OWNING WORKER's process for the life of the pool. They are
  never rebuilt and never fully re-transmitted after that.
- Per cycle, per block: the PARENT applies the EXISTING per-block
  consensus/dual/rho/gamma-driven mutations to ITS OWN resident copy of the
  block (a narrow, faithful re-statement of the mutation loops already in
  `update_distribution_coordination_models_and_solve_sequential` /
  `update_transmission_coordination_model_and_solve` /
  `update_shared_energy_storages_coordination_model_and_solve` -- see
  `_apply_dso_block_params` etc. below), THEN takes a generic, structural
  snapshot of "everything that changed" via the ALREADY-PRODUCTION,
  ALREADY-EQUIVALENCE-TESTED `network.capture_block_mutable_state`
  (P5.15 Step 3.6 TSO clone-capture work, `WORKER_REPORT_S36_CLONE_CAPTURE.md`)
  and sends ONLY that (a plain dict of Param/Var/Suffix values -- no Pyomo
  component objects, no expression trees) to the owning worker. This is the
  "only consensus parameters, duals... exchanged" leg of the Addendum 18
  design, generalized past a hand-enumerated field list (the Step 3.6 report
  found a hand-curated list misses fields -- the generic sweep does not).
- The worker applies `network.apply_block_mutable_state` onto its OWN
  resident (structurally-frozen) clone, solves it with the EXACT SAME
  production solve primitive the serial path uses for that agent type
  (`network.run_smopf` for DSO/TSO -- recovery tiers included, unchanged;
  `shared_energy_storage_data._optimize` for ESSO -- recovery tiers and the
  complementarity-diagnostics parse included, unchanged), replays the
  node-7 / TSO FrozenSMOPF snapshot logic (see `_solve_dso_block_in_worker`
  / `_solve_tso_block_in_worker`), and sends back its OWN now-solved,
  now-loaded resident MODEL OBJECT (not just the SolverResults) plus the
  `SolverResults` and any diagnostics. The parent REPLACES its own
  dictionary entry (`dso_models[node][year][day] = returned_model`, an
  in-place dict-value update, so every OTHER function in
  `_run_operational_planning` that already reads `dso_models`/`tso_model`/
  `esso_model` needs no change and no reordering risk -- Python dicts
  preserve key insertion order regardless of which key is updated when).
  This "ship the solved block back whole" design choice is a considered,
  documented trade (see the Worker report): it costs the same per-cycle
  pickling class as the EXISTING (audited) `*_parallel` path's own return
  leg, in exchange for zero risk of missing a downstream consumer of the
  model that a smaller, hand-picked return payload might silently omit.
- Block-to-worker assembly is always by the FIXED canonical block order
  (never by IPC arrival order), so the returned per-block results
  dictionaries are built identically to the serial path's own nested
  dictionaries, regardless of which worker finishes first.
- Thread caps (`OMP_NUM_THREADS=1` and siblings) are applied by temporarily
  setting them in the PARENT's environment for the duration of
  `Process.start()` (POSIX `spawn` children inherit the parent's
  environment AT PROCESS-CREATION TIME, before any Python/BLAS/OpenMP
  library in the child has run), then restored in the parent immediately
  afterwards -- so only the children are capped; the parent's own process is
  unaffected once workers are up. On this machine/build, `libcoinhsl`
  (MA57/MA97) links no OpenMP runtime (checked via `otool -L`), so this cap
  is a documented precaution, not a measured behaviour change.
- Pyomo `TempfileManager.tempdir` is set to a private, worker-numbered
  directory inside the worker (Addendum 18's explicit ask; the prior audit
  found Pyomo's own per-call temp-file naming already collision-safe, so
  this is defence in depth, not a fix for an observed collision).
- IPOPT log paths need no extra isolation: `_create_smopf_solver` /
  `_create_solver` derive the log filename from (case name, year, day[,
  suffix]), and the STATIC block partition guarantees no two workers ever
  own the same (case, year, day) block -- so no two processes ever write the
  same log file, with or without extra per-worker stamping.
- `SolveProfileGuard` child-side accounting: each worker, if constructed
  with a non-None `guard_permitted`, installs its OWN `SolveProfileGuard`
  (imported fresh in the child) at startup and returns its current
  `.counts` with every response message; `PersistentWorkerPool` exposes
  `total_child_guard_counts()` (summed across all workers, always current)
  for a harness to ADD to its own parent-side guard's counts before calling
  `verify()`.
- Lifecycle: `PersistentWorkerPool` is used as a context manager. `__exit__`
  always calls `shutdown()` (send STOP, join with a timeout, escalate to
  `terminate()`/`kill()` for a straggler) -- covers the ordinary success
  path. `_run_operational_planning` ADDITIONALLY wraps each of the three
  per-stage pool-dispatch call sites in a narrow `try/except Exception:
  pool.shutdown(); raise`, so a failure AT a dispatch call shuts the pool
  down before the exception propagates (belt-and-braces alongside the
  context-manager's own `__exit__`, which fires regardless via CPython's
  normal `with`-statement semantics on any exception, including ones raised
  from code between dispatch calls such as residual/penalty updates).
  `PersistentWorkerPool.dispatch_and_collect` raises immediately (never
  hangs, never falls back to serial) if a worker process dies while a task
  is outstanding for it.
"""

import os
import sys
import time
import traceback
import queue as _queue_module
import multiprocessing as mp


MP_CONTEXT = mp.get_context('spawn')

# Addendum 18: "Each worker single-threaded (OMP_NUM_THREADS=1; MA57, or
# MA97 single-thread): bitwise reproducibility and no core oversubscription.
# Multithreaded factorization is NOT used." VECLIB_MAXIMUM_THREADS caps
# Apple's Accelerate framework (macOS BLAS/LAPACK backend, linked by
# libcoinhsl on this machine per `otool -L`), which does not read
# OMP_NUM_THREADS.
THREAD_ENV_VARS = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
)

# Generous but bounded -- a single DSO/TSO/ESSO block including up to two
# recovery retries is expected to take well under a minute; this is a
# liveness-poll granularity, not a hard per-task deadline (a live worker is
# never killed while still processing; only a DEAD worker with an
# outstanding task raises).
_POLL_INTERVAL_S = 1.0
# Overall bound on how long dispatch_and_collect waits for ALL outstanding
# results for one stage before giving up on a worker that is alive but
# never responds (e.g. wedged in a native call). Chosen well above any
# plausible per-cycle solve time (production cycles run tens of seconds
# total across 51 blocks; one worker's whole batch is a small fraction of
# that).
_STAGE_TIMEOUT_S = 1800.0


class PersistentWorkerCrashed(RuntimeError):
    pass


class PersistentWorkerError(RuntimeError):
    pass


def partition_round_robin(canonical_keys, num_workers):
    """Deterministic, static block -> worker assignment: round-robin over
    `canonical_keys` in the given (caller-fixed) order. Covers every key
    exactly once; depends only on the input order and `num_workers`, never
    on runtime timing -- the requirement this exists to satisfy
    (`p515_s40_persistent_workers_checks.py`: "block-to-worker assignment
    deterministic and covering every block exactly once")."""
    if num_workers < 1:
        raise ValueError(f'num_workers must be >= 1, got {num_workers}')
    assignment = {worker_id: [] for worker_id in range(num_workers)}
    for index, key in enumerate(canonical_keys):
        assignment[index % num_workers].append(key)
    return assignment


def canonical_dso_block_keys(distribution_networks):
    keys = []
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        for year in distribution_network.years:
            for day in distribution_network.days:
                keys.append(('dso', node_id, year, day))
    return keys


def canonical_tso_block_keys(transmission_network):
    keys = []
    for year in transmission_network.years:
        for day in transmission_network.days:
            keys.append(('tso', None, year, day))
    return keys


def canonical_esso_block_keys(active_distribution_network_nodes):
    return [('esso', node_id, None, None) for node_id in active_distribution_network_nodes]


def _set_thread_env_vars(value='1'):
    """Returns the previous values (for restoration)."""
    previous = {}
    for name in THREAD_ENV_VARS:
        previous[name] = os.environ.get(name)
        os.environ[name] = value
    return previous


# ==========================================================================
# Cross-process-safe state capture/apply. `network.capture_block_mutable_
# state` / `apply_block_mutable_state` are reused for the Param/Var/
# Constraint/Objective legs (already plain (name, index) -> value data, safe
# to pickle) -- ONLY the warm-start-suffix leg is replaced.
# `network._snapshot_multiplier_suffixes` (which `capture_block_mutable_
# state` uses internally) keys its captured entries by the LIVE Pyomo
# component OBJECT itself, which is safe only within the SAME process (its
# one existing production use, the TSO snapshot rebuild, always clones a
# pristine base and applies IN THE SAME PROCESS that captured the state).
# Pickling a raw component reference to ANOTHER process does not give a
# reference into that process's structurally-identical model: Pyomo pickles
# the referenced component's ENTIRE containing block, and the unpickled
# object belongs to that embedded, otherwise-discarded copy. Applying it
# onto a DIFFERENT (worker) resident model produces `dual`/`ipopt_z*_in`
# suffix entries keyed by objects that do not belong to that model (Pyomo:
# "model contains export suffix ... that contains N keys that are not Var,
# Constraint, Objective, or the model. Skipping.") -- harmless for the NL
# write itself (Pyomo's writer already detects and skips them), but a LATER
# `model.clone()` on the resulting object can raise
# (`pyomo.common.collections.component_map._rehash_keys`,
# `AttributeError: 'NoneType' object has no attribute 'values'`) -- this was
# found by this task's own two-cycle preflight (worker 4, a DSO node-7
# block, whose legacy snapshot path clones after applying received state),
# not by design review; see WORKER_REPORT_S40_PERSISTENT_WORKERS.md
# "Unexpected findings". `capture_block_state_for_ipc` /
# `apply_block_state_for_ipc` below replace the suffix leg with a
# (component name, index) -> value capture, resolved back to the TARGET
# model's own component via `getattr(model, name)[index]` on apply -- safe
# across a pickle round-trip, and used for EVERY per-cycle state transfer in
# this module, including the worker's own same-process TSO snapshot-rebuild
# call (for uniformity: `state` is always in this format once captured).
# ==========================================================================

def capture_block_state_for_ipc(model):
    from network import capture_block_mutable_state

    state = capture_block_mutable_state(model)
    ipc_suffixes = {}
    for suffix_name in ('ipopt_zL_in', 'ipopt_zU_in', 'dual'):
        if hasattr(model, suffix_name):
            entries = []
            for component, value in getattr(model, suffix_name).items():
                parent = component.parent_component()
                entries.append((parent.name, component.index(), value))
            ipc_suffixes[suffix_name] = entries
    state['suffixes'] = ipc_suffixes
    return state


def _mark_all_vars_stale(model, pe_module):
    """Cosmetic-only bookkeeping fix, found by this task's own two-cycle
    preflight (a real, if non-numerical, divergence -- reported, not
    silently patched over without explanation): serial production's ESSO
    subproblem model, dumped whole into `esso_models_pickle` by
    `p515_g_g1_g4_admm_gates.py`'s own reporting code, shows EVERY Var's
    Pyomo `.stale` flag True after the ADMM loop's last cycle (checked
    directly, both arms, by this task -- 2949/2949 for a representative
    ESSO node). `.stale` is a pure bookkeeping flag Pyomo sets/clears on
    `Var.set_value()` and `model.solutions.load_from()`; it is READ BY NO
    production computation (confirmed by search) -- only by Pyomo's own
    `pprint()`/warning machinery. This worker's per-cycle protocol calls
    `Var.set_value()` on every Var while applying the received captured
    state (`apply_block_state_for_ipc`, reusing `network.
    apply_block_mutable_state`'s generic Var-restore loop), which clears
    `.stale` for whichever Vars that touches -- a side effect serial's own
    flow never triggers (its per-cycle Param-update loops never call
    `Var.set_value()` on a decision variable; only `model.solutions.
    load_from(result)`, inside the solve primitive itself, does). Setting
    every Var stale=True here, right after the solve (which is exactly
    when serial's own model is observed to be in this state), reproduces
    serial's bookkeeping state exactly, with NO effect on any Var's VALUE,
    bound, or fixed flag (unaffected by this call), and NO effect on the
    solve itself (already complete by this point)."""
    for var_data in model.component_data_objects(pe_module.Var, active=None):
        var_data.stale = True


def apply_block_state_for_ipc(model, state):
    from network import apply_block_mutable_state

    safe_state = dict(state)
    ipc_suffixes = safe_state.get('suffixes', {})
    safe_state['suffixes'] = {}
    apply_block_mutable_state(model, safe_state)
    for suffix_name, entries in ipc_suffixes.items():
        if not hasattr(model, suffix_name):
            continue
        suffix = getattr(model, suffix_name)
        for comp_name, index, value in entries:
            comp = getattr(model, comp_name)
            suffix[comp[index]] = value
    return model


def _restore_env_vars(previous):
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


# ==========================================================================
# PARENT-SIDE: narrow, faithful re-statements of the per-block consensus/
# dual parameter-update loops already in shared_resources_planning.py
# (`update_distribution_coordination_models_and_solve_sequential`,
# `update_transmission_coordination_model_and_solve`,
# `update_shared_energy_storages_coordination_model_and_solve`). These are
# the ONLY new "logic" this module introduces for the mutation step itself;
# everything downstream of the mutation (state transfer, solve, snapshot,
# recovery) reuses existing production functions unchanged. Applied by the
# PARENT to its OWN resident block objects, exactly where the existing
# functions would have mutated them, before capturing state to ship to the
# owning worker.
# ==========================================================================

def _apply_dso_block_params(model_yd, network_yd, node_id, year, day, sess_capacity_year,
                             vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess,
                             previous_iter_ess_dso):
    import pyomo.environ as pe  # noqa: E402 (deferred: parent already has pyomo loaded by this point)
    from model_construction_helpers import configure_shared_ess_operational_state

    ref_node_id = network_yd.get_reference_node_id()
    v_base = network_yd.get_node_base_kv(ref_node_id)
    s_base = network_yd.baseMVA
    shared_ess_idx = network_yd.get_shared_energy_storage_idx(ref_node_id)

    model_yd.shared_es_s_rated_fixed[shared_ess_idx].set_value(sess_capacity_year['s_available'] / s_base)
    model_yd.shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_capacity_year['e_available'] / s_base)
    configure_shared_ess_operational_state(
        model_yd, shared_ess_idx,
        pe.value(model_yd.shared_es_s_rated_fixed[shared_ess_idx]),
        pe.value(model_yd.shared_es_e_rated_fixed[shared_ess_idx]),
    )

    for p in model_yd.periods:
        model_yd.dual_vmag_req[p].set_value(dual_vmag['current'][node_id][year][day][p] / v_base)
        model_yd.vmag_req[p].set_value(vmag_req['tso']['current'][node_id][year][day][p] / v_base)
        model_yd.dual_pf_p_req[p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)
        model_yd.dual_pf_q_req[p].set_value(dual_pf['current'][node_id][year][day]['q'][p] / s_base)
        model_yd.p_pf_req[p].set_value(pf_req['tso']['current'][node_id][year][day]['p'][p] / s_base)
        model_yd.q_pf_req[p].set_value(pf_req['tso']['current'][node_id][year][day]['q'][p] / s_base)

    for p in model_yd.periods:
        model_yd.dual_ess_p_req[p].set_value(dual_ess['current'][node_id][year][day]['p'][p])
        model_yd.dual_ess_q_req[p].set_value(dual_ess['current'][node_id][year][day]['q'][p])
        model_yd.p_ess_req[p].set_value(ess_req['z']['current'][node_id][year][day]['p'][p] / s_base)
        model_yd.q_ess_req[p].set_value(ess_req['z']['current'][node_id][year][day]['q'][p] / s_base)
        if previous_iter_ess_dso:
            model_yd.dual_ess_p_prev[p].set_value(dual_ess['prev'][node_id][year][day]['p'][p] / s_base)
            model_yd.dual_ess_q_prev[p].set_value(dual_ess['prev'][node_id][year][day]['q'][p] / s_base)
            model_yd.p_ess_prev[p].set_value(ess_req['dso']['prev'][node_id][year][day]['p'][p] / s_base)
            model_yd.q_ess_prev[p].set_value(ess_req['dso']['prev'][node_id][year][day]['q'][p] / s_base)


def _apply_tso_block_params(model_yd, network_yd, transmission_network, year, day, sess_estimated_capacities,
                             vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess, previous_iter_ess_tso):
    import pyomo.environ as pe
    from model_construction_helpers import configure_shared_ess_operational_state

    s_base = network_yd.baseMVA

    for dn in model_yd.active_distribution_networks:
        node_id = transmission_network.active_distribution_network_nodes[dn]
        v_base = network_yd.get_node_base_kv(node_id)
        shared_ess_idx = network_yd.get_shared_energy_storage_idx(node_id)
        sess_estimated_capacity = sess_estimated_capacities[node_id]

        model_yd.shared_es_s_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['s_available'] / s_base)
        model_yd.shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['e_available'] / s_base)
        configure_shared_ess_operational_state(
            model_yd, shared_ess_idx,
            pe.value(model_yd.shared_es_s_rated_fixed[shared_ess_idx]),
            pe.value(model_yd.shared_es_e_rated_fixed[shared_ess_idx]),
        )

        for p in model_yd.periods:
            model_yd.dual_vmag_req[dn, p].set_value(dual_vmag['current'][node_id][year][day][p] / v_base)
            model_yd.vmag_req[dn, p].set_value(vmag_req['dso']['current'][node_id][year][day][p] / v_base)
            model_yd.dual_pf_p_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)
            model_yd.dual_pf_q_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['q'][p] / s_base)
            model_yd.p_pf_req[dn, p].set_value(pf_req['dso']['current'][node_id][year][day]['p'][p] / s_base)
            model_yd.q_pf_req[dn, p].set_value(pf_req['dso']['current'][node_id][year][day]['q'][p] / s_base)

        shared_ess_idx = network_yd.get_shared_energy_storage_idx(node_id)
        for p in model_yd.periods:
            model_yd.dual_ess_p_req[shared_ess_idx, p].set_value(dual_ess['current'][node_id][year][day]['p'][p])
            model_yd.dual_ess_q_req[shared_ess_idx, p].set_value(dual_ess['current'][node_id][year][day]['q'][p])
            model_yd.p_ess_req[shared_ess_idx, p].set_value(ess_req['z']['current'][node_id][year][day]['p'][p] / s_base)
            model_yd.q_ess_req[shared_ess_idx, p].set_value(ess_req['z']['current'][node_id][year][day]['q'][p] / s_base)
            if previous_iter_ess_tso:
                model_yd.dual_ess_p_prev[shared_ess_idx, p].set_value(dual_ess['prev'][node_id][year][day]['p'][p] / s_base)
                model_yd.dual_ess_q_prev[shared_ess_idx, p].set_value(dual_ess['prev'][node_id][year][day]['q'][p] / s_base)
                model_yd.p_ess_prev[shared_ess_idx, p].set_value(ess_req['tso']['prev'][node_id][year][day]['p'][p] / s_base)
                model_yd.q_ess_prev[shared_ess_idx, p].set_value(ess_req['tso']['prev'][node_id][year][day]['q'][p] / s_base)


def _apply_esso_block_params(model_node, node_id, years_list, days_list, ess_req, dual_ess):
    for y in model_node.years:
        year = years_list[y]
        for d in model_node.days:
            day = days_list[d]
            for p in model_node.periods:
                model_node.p_req[y, d, p].set_value(ess_req['current'][node_id][year][day]['p'][p])
                model_node.q_req[y, d, p].set_value(ess_req['current'][node_id][year][day]['q'][p])
                model_node.dual_p_req[y, d, p].set_value(dual_ess['current'][node_id][year][day]['p'][p])
                model_node.dual_q_req[y, d, p].set_value(dual_ess['current'][node_id][year][day]['q'][p])


# ==========================================================================
# WORKER-SIDE: block solve, reusing production solve primitives unchanged.
# ==========================================================================

def _solve_dso_block_in_worker(model, network_yd, network_params, node_id, year, day, cycle,
                                from_warm_start, results_dir, dso_snapshot_capture_mode,
                                pristine_snapshot_base, state_for_snapshot):
    """Reproduces, for exactly ONE (node, year, day) block, the solve +
    snapshot contract `network_data.NetworkData.optimize` gives the DSO
    sequential path (`shared_resources_planning.py`,
    `update_distribution_coordination_models_and_solve_sequential`) --
    node-7-only failure snapshot, node-7 cycle-7 success comparator. Solve
    itself: `network.run_smopf`, unchanged (recovery tiers included).

    P5.15 Addendum 23/24, Step 3.6 persistent-worker bounded task item 3
    (same design as `_solve_tso_block_in_worker` below, `WORKER_REPORT_
    S36_CLONE_CAPTURE.md` Q2): when `dso_snapshot_capture_mode !=
    'legacy_clone'` and a `pristine_snapshot_base` is available (node 7
    only -- every other node passes `None`), a snapshot is rebuilt ON
    DEMAND from `state_for_snapshot` (the SAME captured-state payload this
    block was just brought up to date with) instead of paying an
    unconditional per-cycle `model.clone()`. Any other node, or the
    `'legacy_clone'` override, falls back to the exact pre-task behaviour."""
    import shared_resources_planning as srp
    from helper_functions import solver_result_succeeded

    is_node7 = (node_id == 7)
    needs_failure_snapshot_capability = is_node7
    needs_comparator_capability = is_node7 and cycle == 7

    if is_node7 and dso_snapshot_capture_mode != 'legacy_clone' and pristine_snapshot_base is not None:
        result = network_yd.run_smopf(model, network_params, from_warm_start=from_warm_start, print_header=True)
        needs_failure_snapshot = needs_failure_snapshot_capability and not solver_result_succeeded(result)
        needs_comparator_snapshot = (
            needs_comparator_capability and str(year) == '2025' and str(day) == 'Autumn'
            and solver_result_succeeded(result)
        )
        if needs_failure_snapshot or needs_comparator_snapshot:
            rebuilt_block = apply_block_state_for_ipc(pristine_snapshot_base.clone(), state_for_snapshot)
            if needs_failure_snapshot:
                srp._save_frozen_smopf_block(
                    rebuilt_block, os.path.join(results_dir, 'FrozenSMOPF'),
                    node_id=node_id, network_name=network_yd.name, year=year, day=day,
                    cycle=cycle, from_warm_start=from_warm_start,
                )
            if needs_comparator_snapshot:
                srp._save_frozen_network_block(
                    rebuilt_block, os.path.join(results_dir, 'FrozenSMOPF'),
                    agent='DSO', node_id=node_id, network_name=network_yd.name, year=year, day=day,
                    cycle=cycle, from_warm_start=from_warm_start, result=result, label='matched_success',
                )
    else:
        pre_solve_model = None
        if needs_failure_snapshot_capability or needs_comparator_capability:
            pre_solve_model = model.clone()

        result = network_yd.run_smopf(model, network_params, from_warm_start=from_warm_start, print_header=True)

        if needs_failure_snapshot_capability and not solver_result_succeeded(result):
            srp._save_frozen_smopf_block(
                pre_solve_model, os.path.join(results_dir, 'FrozenSMOPF'),
                node_id=node_id, network_name=network_yd.name, year=year, day=day,
                cycle=cycle, from_warm_start=from_warm_start,
            )
        if needs_comparator_capability and str(year) == '2025' and str(day) == 'Autumn' and solver_result_succeeded(result):
            srp._save_frozen_network_block(
                pre_solve_model, os.path.join(results_dir, 'FrozenSMOPF'),
                agent='DSO', node_id=node_id, network_name=network_yd.name, year=year, day=day,
                cycle=cycle, from_warm_start=from_warm_start, result=result, label='matched_success',
            )
    return result


def _solve_tso_block_in_worker(model, network_yd, network_params, year, day, cycle, from_warm_start,
                                results_dir, tso_snapshot_capture_mode, pristine_snapshot_base, state_for_snapshot):
    """Reproduces, for exactly ONE (year, day) TSO block, the solve +
    snapshot contract `update_transmission_coordination_model_and_solve`
    gives (`shared_resources_planning.py`) -- both the lightweight (default)
    and legacy_clone capture modes, unchanged in substance (see
    `network.capture_block_mutable_state` / `apply_block_mutable_state`,
    P5.15 Step 3.6). `state_for_snapshot` is the SAME captured-state payload
    this block was just brought up to date with (`apply_block_mutable_state`
    was already applied to `model` before this call) -- reused directly as
    the lightweight mode's "pre-solve" capture, avoiding a redundant second
    `capture_block_mutable_state` call."""
    import shared_resources_planning as srp
    from helper_functions import solver_result_succeeded
    if tso_snapshot_capture_mode != 'legacy_clone' and pristine_snapshot_base is not None:
        result = network_yd.run_smopf(model, network_params, from_warm_start=from_warm_start, print_header=True)
        needs_failure_snapshot = not solver_result_succeeded(result)
        needs_comparator_snapshot = (
            cycle == 7 and str(year) == '2025' and str(day) == 'Summer' and solver_result_succeeded(result)
        )
        if needs_failure_snapshot or needs_comparator_snapshot:
            rebuilt_block = apply_block_state_for_ipc(pristine_snapshot_base.clone(), state_for_snapshot)
            if needs_failure_snapshot:
                srp._save_frozen_network_block(
                    rebuilt_block, os.path.join(results_dir, 'FrozenSMOPF'),
                    agent='TSO', network_name=network_yd.name, year=year, day=day,
                    cycle=cycle, from_warm_start=from_warm_start, result=result, label='failure',
                )
            if needs_comparator_snapshot:
                srp._save_frozen_network_block(
                    rebuilt_block, os.path.join(results_dir, 'FrozenSMOPF'),
                    agent='TSO', network_name=network_yd.name, year=year, day=day,
                    cycle=cycle, from_warm_start=from_warm_start, result=result, label='matched_success',
                )
    else:
        pre_solve_model = model.clone()
        result = network_yd.run_smopf(model, network_params, from_warm_start=from_warm_start, print_header=True)
        if not solver_result_succeeded(result):
            srp._save_frozen_network_block(
                pre_solve_model, os.path.join(results_dir, 'FrozenSMOPF'),
                agent='TSO', network_name=network_yd.name, year=year, day=day,
                cycle=cycle, from_warm_start=from_warm_start, result=result, label='failure',
            )
        if cycle == 7 and str(year) == '2025' and str(day) == 'Summer' and solver_result_succeeded(result):
            srp._save_frozen_network_block(
                pre_solve_model, os.path.join(results_dir, 'FrozenSMOPF'),
                agent='TSO', network_name=network_yd.name, year=year, day=day,
                cycle=cycle, from_warm_start=from_warm_start, result=result, label='matched_success',
            )
    return result


def _worker_main(worker_id, block_specs, task_queue, result_queue, guard_permitted, tempdir_root):
    """Persistent worker process entry point. Runs until it receives a
    `{'cmd': 'stop'}` message. EVERYTHING import-heavy (pyomo, network,
    shared_resources_planning, shared_energy_storage_data, numpy-linked
    libraries transitively) is imported HERE, inside the function body,
    AFTER the thread-cap environment variables have already been inherited
    from the parent at process-creation time (see `PersistentWorkerPool`'s
    `_set_thread_env_vars`/`Process.start()` sequencing) -- this module's
    OWN top-level imports are deliberately limited to `os`/`sys`/`time`/
    `traceback`/`queue`/`multiprocessing`, so importing `admm_persistent_
    workers` itself (unavoidable: `spawn` must import the module that
    defines the target function before calling it) never imports numpy or
    pyomo as a side effect."""
    try:
        import pyomo.environ as pe  # noqa: F401
        from pyomo.common.tempfiles import TempfileManager
        import shared_resources_planning as srp  # noqa: F401
        from shared_energy_storage_data import _optimize as esso_optimize, ESSO_TOL_OVERRIDES
        from helper_functions import solver_result_summary

        worker_tempdir = os.path.join(tempdir_root, f'worker_{worker_id}')
        os.makedirs(worker_tempdir, exist_ok=True)
        TempfileManager.tempdir = worker_tempdir

        guard = None
        if guard_permitted is not None:
            from p513_solve_profile_guard import SolveProfileGuard
            guard = SolveProfileGuard(guard_permitted, label=f'persistent-worker-{worker_id}')
            guard.install()

        resident = {}
        for spec in block_specs:
            resident[spec['key']] = spec

        # Verifiable, not asserted (CLAUDE.md): report the thread-cap
        # environment this worker actually sees at startup, so a zero-solve
        # check can confirm the cap took effect INSIDE the child process,
        # rather than only trusting that the parent set it before spawning.
        env_snapshot = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
        result_queue.put({'cmd': 'ready', 'worker_id': worker_id, 'ok': True, 'n_blocks': len(resident),
                           'env_snapshot': env_snapshot, 'tempdir': TempfileManager.tempdir})
    except Exception:  # noqa: BLE001 -- startup failure must reach the parent, never hang.
        result_queue.put({'cmd': 'ready', 'worker_id': worker_id, 'ok': False,
                           'traceback': traceback.format_exc()})
        return

    while True:
        msg = task_queue.get()
        if msg['cmd'] == 'stop':
            counts = dict(guard.counts) if guard is not None else None
            result_queue.put({'cmd': 'stopped', 'worker_id': worker_id, 'guard_counts': counts})
            return

        if msg['cmd'] != 'solve_batch':
            result_queue.put({'cmd': 'error', 'worker_id': worker_id,
                               'traceback': f'unknown command {msg["cmd"]!r}'})
            continue

        cycle = msg['cycle']
        from_warm_start = msg['from_warm_start']
        block_results = []
        try:
            for task in msg['blocks']:
                key = task['key']
                spec = resident[key]
                model = spec['pristine_model']
                state = task['state']
                apply_block_state_for_ipc(model, state)

                kind = spec['kind']
                if kind == 'dso':
                    result = _solve_dso_block_in_worker(
                        model, spec['network'], spec['network_params'], spec['node_id'],
                        key[2], key[3], cycle, from_warm_start, spec['results_dir'],
                        spec['dso_snapshot_capture_mode'], spec.get('dso_pristine_snapshot_base'), state,
                    )
                    _mark_all_vars_stale(model, pe)
                    block_results.append({
                        'key': key, 'model': model, 'result': result,
                        'result_summary': solver_result_summary(result),
                    })
                elif kind == 'tso':
                    result = _solve_tso_block_in_worker(
                        model, spec['network'], spec['network_params'], key[2], key[3], cycle,
                        from_warm_start, spec['results_dir'], spec['tso_snapshot_capture_mode'],
                        spec.get('tso_pristine_snapshot_base'), state,
                    )
                    _mark_all_vars_stale(model, pe)
                    block_results.append({
                        'key': key, 'model': model, 'result': result,
                        'result_summary': solver_result_summary(result),
                    })
                elif kind == 'esso':
                    local_diag_sink = []
                    local_complementarity_sink = []
                    result = esso_optimize(
                        model, spec['esso_solver_params'], from_warm_start=from_warm_start,
                        node_id=spec['node_id'], diagnostic_sink=local_diag_sink,
                        option_overrides=ESSO_TOL_OVERRIDES,
                        complementarity_diagnostics_sink=local_complementarity_sink,
                        cycle=cycle, logs_dir=spec['esso_logs_dir'],
                    )
                    _mark_all_vars_stale(model, pe)
                    block_results.append({
                        'key': key, 'model': model, 'result': result,
                        'result_summary': solver_result_summary(result),
                        'recovery_diagnostics': local_diag_sink,
                        'complementarity_diagnostics': local_complementarity_sink,
                    })
                else:
                    raise ValueError(f'unknown block kind {kind!r} for key {key!r}')

            counts = dict(guard.counts) if guard is not None else None
            result_queue.put({'cmd': 'solve_batch_done', 'worker_id': worker_id,
                               'blocks': block_results, 'guard_counts': counts})
        except Exception:  # noqa: BLE001 -- must reach the parent, never hang, never swallow.
            result_queue.put({'cmd': 'error', 'worker_id': worker_id,
                               'traceback': traceback.format_exc()})


class PersistentWorkerPool:
    """Parent-side handle. Construct once per `_run_operational_planning`
    call (right where the existing `tso_pristine_base` is built); use as a
    context manager so `shutdown()` always runs."""

    def __init__(self, num_workers, distribution_networks, dso_models, transmission_network, tso_model,
                 tso_pristine_base, tso_snapshot_capture_mode, shared_ess_data, esso_model,
                 tempdir_root, guard_permitted=None, dso_pristine_base=None,
                 dso_snapshot_capture_mode='lightweight'):
        self.num_workers = num_workers
        self.distribution_networks = distribution_networks
        self.dso_models = dso_models
        self.transmission_network = transmission_network
        self.tso_model = tso_model
        self.shared_ess_data = shared_ess_data
        self.esso_model = esso_model
        self.guard_permitted = guard_permitted
        self._processes = []
        self._task_queues = []
        self._result_queues = []
        self._alive = False
        self._child_guard_counts_by_worker = {}

        dso_keys = canonical_dso_block_keys(distribution_networks)
        tso_keys = canonical_tso_block_keys(transmission_network)
        esso_keys = canonical_esso_block_keys(shared_ess_data.active_distribution_network_nodes)
        all_keys = dso_keys + tso_keys + esso_keys
        self.canonical_dso_keys = dso_keys
        self.canonical_tso_keys = tso_keys
        self.canonical_esso_keys = esso_keys

        assignment = partition_round_robin(all_keys, num_workers)
        self.worker_for_key = {}
        for worker_id, keys in assignment.items():
            for key in keys:
                self.worker_for_key[key] = worker_id

        block_specs_by_worker = {worker_id: [] for worker_id in range(num_workers)}
        for key in dso_keys:
            _, node_id, year, day = key
            distribution_network = distribution_networks[node_id]
            network_yd = distribution_network.network[year][day]
            spec = {
                'kind': 'dso', 'key': key, 'node_id': node_id,
                'pristine_model': dso_models[node_id][year][day].clone(),
                'network': network_yd,
                'network_params': distribution_network.params,
                'results_dir': distribution_network.results_dir,
                'dso_snapshot_capture_mode': dso_snapshot_capture_mode,
                'dso_pristine_snapshot_base': (
                    dso_pristine_base[year][day].clone()
                    if (node_id == 7 and dso_pristine_base is not None) else None
                ),
            }
            block_specs_by_worker[self.worker_for_key[key]].append(spec)
        for key in tso_keys:
            _, _, year, day = key
            network_yd = transmission_network.network[year][day]
            spec = {
                'kind': 'tso', 'key': key,
                'pristine_model': tso_model[year][day].clone(),
                'network': network_yd,
                'network_params': transmission_network.params,
                'results_dir': transmission_network.results_dir,
                'tso_snapshot_capture_mode': tso_snapshot_capture_mode,
                'tso_pristine_snapshot_base': (
                    tso_pristine_base[year][day].clone() if tso_pristine_base is not None else None
                ),
            }
            block_specs_by_worker[self.worker_for_key[key]].append(spec)
        for key in esso_keys:
            _, node_id, _, _ = key
            spec = {
                'kind': 'esso', 'key': key, 'node_id': node_id,
                'pristine_model': esso_model[node_id].clone(),
                'esso_solver_params': shared_ess_data.params.solver_params,
                'esso_logs_dir': shared_ess_data.logs_dir,
            }
            block_specs_by_worker[self.worker_for_key[key]].append(spec)

        previous_env = _set_thread_env_vars('1')
        try:
            for worker_id in range(num_workers):
                task_q = MP_CONTEXT.Queue()
                result_q = MP_CONTEXT.Queue()
                process = MP_CONTEXT.Process(
                    target=_worker_main,
                    args=(worker_id, block_specs_by_worker[worker_id], task_q, result_q,
                          guard_permitted, tempdir_root),
                    daemon=True,
                )
                process.start()
                self._processes.append(process)
                self._task_queues.append(task_q)
                self._result_queues.append(result_q)
        finally:
            _restore_env_vars(previous_env)

        self._alive = True
        self.ready_info = []
        # Fail fast on a startup crash / import error, never hang.
        for worker_id in range(num_workers):
            ready = self._blocking_get(worker_id, timeout_s=_STAGE_TIMEOUT_S)
            if ready['cmd'] != 'ready' or not ready.get('ok', False):
                self.shutdown()
                raise PersistentWorkerError(
                    f'worker {worker_id} failed to start: '
                    f'{ready.get("traceback", ready)}'
                )
            self.ready_info.append(ready)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    def _blocking_get(self, worker_id, timeout_s):
        deadline = time.time() + timeout_s
        process = self._processes[worker_id]
        result_q = self._result_queues[worker_id]
        while True:
            try:
                return result_q.get(timeout=_POLL_INTERVAL_S)
            except _queue_module.Empty:
                if not process.is_alive():
                    raise PersistentWorkerCrashed(
                        f'worker {worker_id} died unexpectedly (exitcode={process.exitcode}) '
                        f'while a task was outstanding'
                    )
                if time.time() > deadline:
                    raise PersistentWorkerCrashed(
                        f'worker {worker_id} did not respond within {timeout_s}s '
                        f'(process alive={process.is_alive()})'
                    )

    def _dispatch_stage(self, kind, canonical_keys, cycle, from_warm_start, captured_states):
        by_worker = {}
        for key in canonical_keys:
            worker_id = self.worker_for_key[key]
            by_worker.setdefault(worker_id, []).append({'key': key, 'state': captured_states[key]})

        for worker_id, blocks in by_worker.items():
            self._task_queues[worker_id].put({
                'cmd': 'solve_batch', 'cycle': cycle, 'from_warm_start': from_warm_start,
                'blocks': blocks,
            })

        responses_by_key = {}
        for worker_id in by_worker:
            response = self._blocking_get(worker_id, timeout_s=_STAGE_TIMEOUT_S)
            if response['cmd'] == 'error':
                self.shutdown()
                raise PersistentWorkerError(
                    f'worker {worker_id} raised while solving a {kind} batch:\n'
                    f'{response["traceback"]}'
                )
            if response['cmd'] != 'solve_batch_done':
                self.shutdown()
                raise PersistentWorkerError(
                    f'worker {worker_id} sent an unexpected message: {response}'
                )
            if response.get('guard_counts') is not None:
                self._child_guard_counts_by_worker[worker_id] = response['guard_counts']
            for block_result in response['blocks']:
                responses_by_key[block_result['key']] = block_result

        # Assemble in the FIXED canonical order, never IPC arrival order.
        return [responses_by_key[key] for key in canonical_keys]

    def run_dso_stage(self, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess,
                       admm_parameters, sess_estimated_capacities, from_warm_start, cycle):
        from helper_functions import solver_result_succeeded, solver_result_summary

        captured_states = {}
        for key in self.canonical_dso_keys:
            _, node_id, year, day = key
            model_yd = self.dso_models[node_id][year][day]
            distribution_network = self.distribution_networks[node_id]
            network_yd = distribution_network.network[year][day]
            _apply_dso_block_params(
                model_yd, network_yd, node_id, year, day,
                sess_estimated_capacities[node_id][year],
                vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess,
                admm_parameters.previous_iter['ess']['dso'],
            )
            captured_states[key] = capture_block_state_for_ipc(model_yd)

        ordered_results = self._dispatch_stage('dso', self.canonical_dso_keys, cycle, from_warm_start, captured_states)

        res = {}
        for key, block_result in zip(self.canonical_dso_keys, ordered_results):
            _, node_id, year, day = key
            self.dso_models[node_id][year][day] = block_result['model']
            res.setdefault(node_id, {}).setdefault(year, {})[day] = block_result['result']
            if not solver_result_succeeded(block_result['result']):
                print(
                    f'[WARNING] Distribution network node={node_id}, '
                    f'network={block_result["model"].name}, year={year}, day={day} '
                    f'did not converge: {solver_result_summary(block_result["result"])}'
                )
        return res

    def run_tso_stage(self, vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess,
                       admm_parameters, sess_estimated_capacities, from_warm_start, cycle):
        from helper_functions import solver_result_succeeded, solver_result_summary

        captured_states = {}
        for key in self.canonical_tso_keys:
            _, _, year, day = key
            model_yd = self.tso_model[year][day]
            network_yd = self.transmission_network.network[year][day]
            _apply_tso_block_params(
                model_yd, network_yd, self.transmission_network, year, day,
                sess_estimated_capacities,
                vmag_req, dual_vmag, pf_req, dual_pf, ess_req, dual_ess,
                admm_parameters.previous_iter['ess']['tso'],
            )
            captured_states[key] = capture_block_state_for_ipc(model_yd)

        ordered_results = self._dispatch_stage('tso', self.canonical_tso_keys, cycle, from_warm_start, captured_states)

        res = {}
        for key, block_result in zip(self.canonical_tso_keys, ordered_results):
            _, _, year, day = key
            self.tso_model[year][day] = block_result['model']
            res.setdefault(year, {})[day] = block_result['result']
            if not solver_result_succeeded(block_result['result']):
                print(
                    f'[ERROR] Transmission network {block_result["model"].name}, '
                    f'year={year}, day={day} did not converge: '
                    f'{solver_result_summary(block_result["result"])}'
                )
        return res

    def run_esso_stage(self, ess_req, dual_ess, admm_parameters, from_warm_start, cycle):
        from helper_functions import solver_result_succeeded, solver_result_summary

        years_list = [year for year in self.shared_ess_data.years]
        days_list = [day for day in self.shared_ess_data.days]

        captured_states = {}
        for key in self.canonical_esso_keys:
            _, node_id, _, _ = key
            model_node = self.esso_model[node_id]
            _apply_esso_block_params(model_node, node_id, years_list, days_list, ess_req, dual_ess)
            captured_states[key] = _capture_esso_state(model_node)

        ordered_results = self._dispatch_stage('esso', self.canonical_esso_keys, cycle, from_warm_start, captured_states)

        res = {}
        for key, block_result in zip(self.canonical_esso_keys, ordered_results):
            _, node_id, _, _ = key
            self.esso_model[node_id] = block_result['model']
            res[node_id] = block_result['result']
            self.shared_ess_data.solver_recovery_diagnostics.extend(block_result.get('recovery_diagnostics', []))
            self.shared_ess_data.esso_complementarity_diagnostics.extend(
                block_result.get('complementarity_diagnostics', []))
            if not solver_result_succeeded(block_result['result']):
                print(
                    f'[WARNING] SharedESS operational planning node={node_id} did not converge: '
                    f'{solver_result_summary(block_result["result"])}'
                )
        return res

    def total_child_guard_counts(self):
        totals = {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}
        for counts in self._child_guard_counts_by_worker.values():
            if counts is None:
                continue
            for k in totals:
                totals[k] += counts.get(k, 0)
        return totals

    def shutdown(self, timeout_s=30.0):
        if not self._alive:
            return
        self._alive = False
        for worker_id, process in enumerate(self._processes):
            if process.is_alive():
                try:
                    self._task_queues[worker_id].put({'cmd': 'stop'})
                except Exception:  # noqa: BLE001 -- best-effort; join/terminate below is the real guarantee.
                    pass
        for worker_id, process in enumerate(self._processes):
            process.join(timeout=timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        for q in self._task_queues + self._result_queues:
            try:
                q.close()
            except Exception:  # noqa: BLE001
                pass


def _capture_esso_state(model_node):
    return capture_block_state_for_ipc(model_node)


def persistent_workers_enabled(admm_parameters):
    settings = getattr(admm_parameters, 'persistent_workers', None)
    if not settings:
        return False
    return bool(settings.get('enabled', False))
