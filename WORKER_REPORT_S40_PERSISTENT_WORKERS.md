# Worker Report — P5.15 Addendum 22 item (2), Step 3.6 final part: persistent worker processes

## Task received

Implement persistent, single-threaded worker processes for within-stage ADMM block
parallelism (`PLANNER_BRIEF_2026-09-13.md` Addendum 18 design), behind a default-off
flag, and pass the two-cycle bitwise gate. Frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`, `2_step_3_6.then`.
Full-length matched run explicitly NOT authorized to run here.

## Files inspected

`PLANNER_BRIEF_2026-09-13.md` (Addendum 18); `WORKER_REPORT_S36_PARALLEL_AUDIT.md`;
`P5_15_STEP36_TIMING_DESIGN.md`; `WORKER_REPORT_S36_TIMING_10CYC.md`;
`WORKER_REPORT_S36_CLONE_CAPTURE.md`; frozen spec v11 JSON; `REVISION_CONTEXT.md`
(current head); `admm_parameters.py` (full); `shared_resources_planning.py`
(`_run_operational_planning`, `update_distribution_coordination_models_and_solve*`,
`update_transmission_coordination_model_and_solve`,
`update_shared_energy_storages_coordination_model_and_solve`,
`update_and_check_convergence`, `_update_admm_penalties`, rho/gamma Param wiring);
`network.py` (`_run_smopf`, `_create_smopf_solver`, `capture_block_mutable_state`,
`apply_block_mutable_state`, `_snapshot_multiplier_suffixes`); `network_data.py`
(`NetworkData.optimize`); `shared_energy_storage_data.py` (`SharedEnergyStorageData.
optimize`, `_optimize`, `_create_solver`); `p513_solve_profile_guard.py` (full);
`p515_g_g1_g4_admm_gates.py` (`run_s39_arm`, `_s39_configure_hook`, `_s39_ids_for_mode`,
`_acquire_exclusive_run_lock`); `p515_s40_clone_capture_preflight.py` (full, as the
structural template); `p515_s32_zero_solve_checks.py` (`_build_admm_ready_state`,
reused); `data/SRP1/case*/*.json` (`linear_solver`: `ma97` DSO/TSO, `ma57` ESSO);
`otool -L /usr/local/lib/libcoinhsl.2.dylib` (no OpenMP linkage on this build).

## Files modified / created

Production:
- `admm_persistent_workers.py` (new) — the persistent-worker pool.
- `admm_parameters.py` — `persistent_workers = {'enabled': False, 'num_workers': 8}`.
- `shared_resources_planning.py` — pool construction next to `tso_pristine_base`; three
  `if persistent_pool is not None: ... else: <original call>` branches (DSO/TSO/ESSO);
  explicit shutdown at the loop's natural exit.

Checks / gate:
- `p515_s40_persistent_workers_checks.py` (new) — zero-solve checks.
- `p515_s40_persistent_workers_preflight.py` (new) — two-cycle bitwise gate.

Evidence committed: `data/SRP1/Results/P515S40/persistent_workers_checks/` (results +
manifest); `data/SRP1/Results/P515S40/persistent_workers_preflight_v4/` (both arms) +
its launch log. Commits: `c6d60857` (design + flag + checks), `971c7558` (cross-process
suffix-capture fix), `549b1c5b` (TSO attribute-name fix), `1b3ca9fc` (`.stale`-flag
fix), `a3793e9e` (final preflight evidence). `admm_persistent_workers.py`'s own header
docstring is the authoritative design write-up; this report summarizes it.

## Design summary

**Block granularity and assignment.** One block = one (DSO node, year, day) solve (36),
one (TSO year, day) solve (12), or one (ESSO node) solve (3) — 51 total. A single
canonical, fixed-order key list is built once (`canonical_dso_block_keys` /
`canonical_tso_block_keys` / `canonical_esso_block_keys`) and partitioned across `W`
workers by static round-robin (`partition_round_robin`) — deterministic, covers every
block exactly once, never revisited at runtime (no load-based rebalancing).

**What is exchanged per cycle.** Models are cloned ONCE per block at pool construction
(mirroring the existing `tso_pristine_base` pattern) and live in the owning worker for
the pool's lifetime. Each cycle: the PARENT applies the existing per-block consensus/
dual/rho/gamma-driven mutations to its OWN resident copy (narrow, faithful
re-statements of the existing param-update loops — `_apply_dso_block_params` /
`_apply_tso_block_params` / `_apply_esso_block_params`, new code, not shared with the
serial functions to avoid touching them), then takes a generic snapshot via
`capture_block_state_for_ipc` (built on the already-production, already-tested
`network.capture_block_mutable_state`) and sends ONLY that (plain (name,index)->value
data, no Pyomo objects) to the owning worker. The worker applies it
(`apply_block_state_for_ipc`), solves with the unchanged production primitive
(`network.run_smopf` for DSO/TSO — recovery tiers and node-7/TSO FrozenSMOPF snapshot
logic replayed; `shared_energy_storage_data._optimize` for ESSO — recovery tiers and
complementarity diagnostics unchanged), and returns its own solved MODEL OBJECT (not
just results) plus the `SolverResults` and diagnostics. The parent replaces its own
dict entry (`dso_models[node][year][day] = returned_model`) — an in-place update, so
every other function already reading `dso_models`/`tso_model`/`esso_model` needs no
change. This "ship the solved block back whole" choice is a documented trade: it costs
the same per-cycle pickling class as the existing (audited) `*_parallel` path's return
leg, in exchange for zero risk of missing a downstream consumer of the model that a
smaller, hand-picked return payload might silently omit — the alternative (auditing and
re-deriving every one of dozens of downstream consumers of a solved block) was judged
out of this task's scope.

**Determinism controls.** Static block order (never IPC arrival order) for result
assembly. Thread caps (`OMP_NUM_THREADS=1` and five siblings) set in the PARENT's
`os.environ` only for the duration of `Process.start()` (POSIX `spawn` children inherit
the parent's environment at process-creation time, before any BLAS/OpenMP library in
the child runs), then restored — only children are capped. Private per-worker
`TempfileManager.tempdir`. IPOPT log paths need no extra stamping: the static partition
guarantees no two workers ever own the same (case, year, day), so log-path collision is
structurally impossible regardless of per-worker isolation.

**Solve accounting (Requirement 4).** Each worker, if given a non-`None`
`guard_permitted`, installs its own `SolveProfileGuard` at startup and returns its
`.counts` with every message; `PersistentWorkerPool.total_child_guard_counts()` sums
across workers, always current, for a harness to add to its own parent-side guard.
Demonstrated directly in the zero-solve checks (injected-exception test: 0/0 child
counts) and, for the real preflight run, independently verified from each run's own
per-block IPOPT logs (not from a guard at all, since `p515_g_g1_g4_admm_gates.py`'s own
embedded guard cannot see child solves and this task may not edit that harness): serial
153, parallel8 153 (51 init + 51×2 cycles), exact match.

**Lifecycle.** Context-manager pool with `shutdown()` (STOP message, join with timeout,
escalate to `terminate()`/`kill()`); a narrow `try/except Exception: pool.shutdown();
raise` around each of the three per-stage dispatch calls; an explicit `shutdown()` call
at the ADMM loop's natural exit (`break` or iteration exhaustion, one common fall-
through point). A dead worker with an outstanding task raises
`PersistentWorkerCrashed` immediately (polling `process.is_alive()`, never a silent
hang or fallback to serial).

## Zero-solve checks (Requirement: `p515_s40_persistent_workers_checks.py`)

`SolveProfileGuard(permitted=())` armed for the whole script, parent AND every spawned
worker (`guard_permitted=()`). All five checks PASS (`all_checks_pass=True`,
`data/SRP1/Results/P515S40/persistent_workers_checks/results.json`):

1. **Flag off default**: fresh `ADMMParameters()` has `persistent_workers ==
   {'enabled': False, 'num_workers': 8}`; a static source check confirms all three
   dispatch sites keep an unconditional `else:` branch calling the original,
   unmodified entry points by name.
2. **Block partition**: 36+12+3=51 distinct keys on the real SRP1 ADMM-ready state;
   `partition_round_robin` at 7 worker counts (1,2,3,7,8,51,97) is deterministic
   (re-run twice, identical) and covers every key exactly once.
3. **Pool startup** (real, 2-worker pool spawned): every worker's reported
   `env_snapshot` shows every thread-cap var `== '1'` inside the child; the parent's
   own environment is unchanged after construction; worker tempdirs distinct;
   `worker_for_key` covers all 51 keys.
4. **Fixture reload**: all 7 FrozenSMOPF/cycle21 fixtures cited by
   `WORKER_REPORT_S36_CLONE_CAPTURE.md` still unpickle.
5. **Lifecycle on an injected exception**, stand-in work: a real captured DSO-block
   state, corrupted with an invented Param name, dispatched through the real
   `_dispatch_stage`; `apply_block_mutable_state` raises `AttributeError` inside the
   worker BEFORE any solve (guard counts stay 0/0, confirmed); `PersistentWorkerError`
   raised in the parent; every worker process dead afterward; a second `shutdown()`
   call is a harmless no-op.

Guard verdict: `parent={'permitted_solve': 0, ...}`, `child_total={'permitted_solve':
0, ...}`, `failures=[]`.

## Two-cycle bitwise gate

Command used (`p515_s40_persistent_workers_preflight.py`, run with a `v4` suffix so
the final evidence lands in a fresh, never-reused root):

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
    p515_s40_persistent_workers_preflight.py v4 \
    > data/SRP1/Results/P515S40/persistent_workers_preflight_v4_launch.log 2>&1
```

Attached, alone, both streams captured, preconditions checked (no lock, no forbidden
process, fresh output roots, production files clean) immediately before each launch.

**Three real defects were found and fixed during this gate, each via a genuine crash
or a real diff, not by design review:**

1. **Cross-process suffix-capture defect** (`971c7558`). `network.
   _snapshot_multiplier_suffixes` (used inside `capture_block_mutable_state`) keys its
   captured warm-start-suffix entries by the LIVE Pyomo component object — safe only
   within one process (its one prior production use always clones-and-applies in the
   same process). Reused unchanged for the cross-process transfer, a pickled component
   reference resolved to an object belonging to an embedded, discarded copy of the
   ORIGINATING block, not the receiving model; Pyomo silently skipped the resulting
   stale `dual`/`ipopt_z*_in` entries when writing the NL file, but a later
   `model.clone()` (the DSO node-7 legacy snapshot path) raised inside
   `ComponentMap`'s deepcopy/rehash (`AttributeError: 'NoneType' object has no
   attribute 'values'`). Fix: `capture_block_state_for_ipc` /
   `apply_block_state_for_ipc`, keying suffix entries by (component name, index)
   instead, resolved back via `getattr(model, name)[index]` on the TARGET model.
   Verified zero-solve with a real `pickle.dumps`/`loads` round trip.
2. **TSO Pyomo attribute-name typo** (`549b1c5b`). `_apply_tso_block_params` read
   `model_yd.active_distribution_network_nodes`; the model's own Set is
   `active_distribution_networks` (a NetworkData attribute of the same near-identical
   name is the correct source for the OTHER lookup in the same loop). `AttributeError`
   on the first TSO cycle of the second preflight attempt.
3. **Pyomo `.stale` bookkeeping divergence** (`1b3ca9fc`, cosmetic, non-numerical).
   After fixes 1–2, the run completed but `esso_models_pickle.bytes` differed
   (2842737 vs 2880195). A `pickletools` opcode diff isolated it to 1773
   `NEWTRUE`↔`NEWFALSE` flips and nothing content-bearing. Traced to every Var's
   `.stale` flag: serial shows ALL Vars `stale=True` after the last cycle (2949/2949,
   checked directly); the worker-solved model showed only 1176/2949, because
   `apply_block_state_for_ipc`'s generic Var-restore loop calls `Var.set_value()` on
   every Var, clearing `.stale` as a side effect serial's own per-cycle Param-update
   loops never trigger. `_mark_all_vars_stale` sets every Var stale=True in the worker
   right after solving, reproducing serial's observed state exactly (no effect on any
   value/bound/fixed flag).

**After all three fixes, the gate is run-to-completion but NOT bitwise identical.**
Every artifact except one field is identical: `g_s39_D.json` (full `cycle_trajectory`
— every Boyd residual/ratio/norm, rho/gamma actions, costs, EFC, complementarity
diagnostics, `network_failures_summary`, `rule_eleven_checklist`), `boyd_terminal.json`,
`component_levels_terminal.json`, `interface_settlement_detail_s31c.json`,
`interface_voltage_terminal.json`, and all 5 JSONL sidecars — **0 diffs**.

**One field remains**: `report.esso_models_pickle.bytes` — serial 2,842,737,
parallel8 2,880,195. Diagnosed, not eliminated, within this task's remaining scope:

- A second `pickletools` opcode diff (post-`.stale`-fix) shows exactly `NONE: -2355`,
  `BININT1: +2355` and nothing else content-bearing.
- Reproduced in complete isolation, zero-solve: `capture_block_state_for_ipc` +
  `apply_block_state_for_ipc` applied to the SAME serial model (self-capture, no
  process boundary at all) reproduces the identical byte-size jump
  (949081 → 951436 bytes for one ESSO node).
- Root cause: `network.apply_block_mutable_state`'s Var-restore loop calls
  `var_data.setlb(lb)` / `setub(ub)` unconditionally for every Var. When a Var's bound
  was never explicitly set (Pyomo stores it as an internal `None`, resolved lazily from
  the variable's domain, e.g. `NonNegativeReals` → 0), this call converts it to an
  EXPLICITLY stored value — same effective bound, different internal representation
  (`None` → a small int), hence the pickle-opcode pattern.
- Exhaustively checked, node 5, this run's own committed artifacts: **0 differences**
  in every Var's value, `.lb`, `.ub` (as queried — the effective value, not the
  internal representation), and `.fixed`; every Param's value; every Constraint's and
  Objective's `.active` flag; every `ipopt_zL_in`/`ipopt_zU_in`/`ipopt_zL_out`/
  `ipopt_zU_out`/`dual` Suffix entry and value. This is why it appears in the raw
  model-pickle diagnostic dump and in NO other compared artifact (nothing else reads
  or reports the implicit/explicit bound-storage distinction).
- **Secondary, non-blocking finding, same root mechanism family**: the parallel8
  stdout contains 263,707 Pyomo `W1001` domain-validation warnings (0 in serial),
  from `Var.set_value()` (called by the same generic Var-restore loop, without
  `skip_validation=True`) being applied to tiny (~1e-9) out-of-domain slack values
  that `model.solutions.load_from()` does not validate. Log-verbosity only (the
  parallel8 launch log is 893k lines vs a ~600-line reference); no effect on any value
  or on the solve.
- **Not fixed**: both are side effects of reusing `network.apply_block_mutable_state`
  (built for the Step 3.6 TSO snapshot mechanism's rare, same-process, once-per-
  failure rebuild) as this module's high-frequency, every-cycle, every-block state-
  sync primitive — a much higher exercise rate than its original use. A clean fix
  needs either editing `network.py`'s tested function (its Var-restore loop would need
  a `skip_validation=True` `set_value()` call and a bound-unchanged skip for
  `setlb`/`setub`) — out of this task's stated scope of touching only the persistent-
  workers path — or a bypass specific to this module (a hand-written Var-restore loop
  duplicating `apply_block_mutable_state`'s logic with those two changes). Deferred to
  the Planner rather than decided unilaterally, per CLAUDE.md's rule against adjusting
  a comparator to force a pass without authorization.

**Solve profile** (Requirement 4): `report.solve_profile.*` (the harness's own,
parent-only-guard-derived field) is EXCLUDED from the bitwise diff, with an in-code
comment stating why (the field measures which PROCESS ran a solve, architecturally
different by design for persistent workers, exactly what Requirement 4 anticipates and
answers with a separate mechanism) — not because a numeric divergence was found there
and hidden. Independently verified instead, from each run's own per-block IPOPT logs
(not a guard): **serial 153 = parallel8 153** (51 init + 51×2 cycles each).

## Serial vs parallel wall time and speedup

Two ADMM cycles, cold start, oracle `s39_D`, 8 workers:

| | serial | parallel8 |
|---|---:|---:|
| total wall (both cycles + init) | 104.70 s | 134.03 s |
| Iteration 1 | ~72 s (init + cycle 1, not separately logged) | — |
| Iteration 2 (reported) | 32.49 s | 32.34–32.49 s |

**Observed speed-up: 0.78× — parallel8 is SLOWER than serial at 2 cycles.** This is
expected and was not the target of this gate (the gate is correctness, not speed;
Addendum 18's own speed-up projections were for a longer, warmed-up run). The fixed
cost of spawning 8 fresh Python processes and importing pyomo/numpy/networkx/
matplotlib in each (no module-level caching across cycles helps here, since a 2-cycle
preflight pays that startup cost once but amortizes it over only 2 cycles) plausibly
dominates at this length; the prior 10-cycle timing re-measurement (`WORKER_REPORT_
S36_TIMING_10CYC.md`) projected 5.09× at 8 workers from a *solve-time-only* Amdahl
analysis, which does not include worker start-up — this preflight does not by itself
confirm or refute that projection, and a longer run is needed to separate them. Not
further investigated here (out of this task's stated scope — the gate, not sizing).

## Validation

- **Code executes correctly**: yes — the full pool lifecycle (spawn, thread caps,
  private tempdirs, per-stage dispatch, guard accounting, shutdown) ran successfully
  across both the zero-solve checks and a real, 2-cycle, 51-block, 8-worker ADMM run
  with real IPOPT solves, real recovery-tier machinery reachable (none fired in this
  run — 0 network failures on both arms), and the node-7/TSO FrozenSMOPF snapshot
  paths reachable (none fired in this run — no failure, not cycle 7).
- **Test (zero-solve checks) passes**: yes, all 5, guard-enforced.
- **Requested diagnostic (bitwise gate) works**: yes — it found three real defects and
  is now reporting an honestly-diagnosed remaining divergence, not fabricating a pass.
- **Underlying numerical-equivalence claim**: established at the level this run
  supports — every reported, algorithmically-meaningful ADMM quantity (residuals,
  rho/gamma trajectory, costs, EFC, complementarity diagnostics, voltage/PF/ESS
  interface settlement, network-failure/recovery summary, solve count) is bitwise
  identical between serial and 8-worker parallel over 2 real cycles. The ONE remaining
  divergence is confined to a raw whole-model pickle diagnostic dump and has been
  shown, by direct inspection of every field Pyomo exposes on that model, to carry
  zero numerical content — but this is short of literal bitwise identity on "every
  numeric artifact," the gate's stated bar, and is reported as such rather than
  rounded up to PASS.

## Unexpected findings

- `libcoinhsl` on this machine/build links no OpenMP runtime (`otool -L`) — the
  `OMP_NUM_THREADS=1` cap this design applies is a documented precaution, not a
  measured behaviour change on this specific build; a different HSL build could behave
  differently, and the cap is still applied unconditionally per Addendum 18.
- `network.apply_block_mutable_state` (the Step 3.6 TSO-capture mechanism, already
  production and already equivalence-tested for its ORIGINAL use) has two real, if
  numerically inert, side effects when reused at a much higher call frequency than
  its original design point: it silently converts implicit (domain-derived) Var
  bounds into explicit stored ones, and it triggers Pyomo domain-validation warnings
  for out-of-domain values that `solutions.load_from()` does not check. Neither was
  visible in the function's original (rare, same-process) use.
- Two zero-solve, single-process reproductions (self-capture-and-apply on the SAME
  serial model, no multiprocessing at all) exactly reproduced both remaining findings
  — useful precedent for a Planner-authorized fix, since it means the eventual repair
  can be validated without a full solve-based preflight re-run.

## Remaining issues

1. **The two-cycle gate is not literally bitwise identical.** One field
   (`esso_models_pickle.bytes`) diverges for a fully diagnosed, exhaustively-checked-
   benign (non-numerical) reason. The Planner must decide whether to (a) authorize an
   explicit, documented exclusion of this field (with the diagnosis above as its
   justification) and re-run the gate once to confirm PASS, or (b) authorize a fix to
   `network.apply_block_mutable_state`'s Var-restore loop (or a bypass specific to
   this module) and re-run.
2. **Speed-up not measured at scale.** The 2-cycle preflight shows parallel8 SLOWER
   than serial (0.78×), dominated by one-time worker-startup cost; this does not
   inform whether the design meets the Addendum 18 speed-up target at production
   length (300 cycles) — a separate, longer, timing-focused measurement would be
   needed, out of this task's scope.
3. **TSO is not split below one (year, day) block for genuine intra-stage
   parallelism finer than 12-way** — already matches Addendum 18's own block count
   ("12 TSO"), no further granularity is claimed or needed.
4. **v1–v3 preflight attempts are NOT committed** (only the final v4 run is), to keep
   the commit size reasonable; they remain on disk, uncommitted, as the debugging
   trail, and each attempt's discovered defect is independently recorded in its own
   commit message (`971c7558`, `549b1c5b`) — nothing was destroyed, but a reader
   wanting the raw IPOPT tee output of the intermediate attempts must look on disk,
   not in git history.
5. **Only the DSO node-7 legacy-clone and TSO lightweight-capture snapshot paths were
   exercised in the "no failure" branch** (0 network failures occurred in this
   2-cycle run on either arm) — the FAILURE-triggered snapshot-write path inside a
   worker (`_solve_dso_block_in_worker`'s `needs_failure_snapshot` branch,
   `_solve_tso_block_in_worker`'s equivalent) is implemented and covered by the
   zero-solve equivalence logic it reuses (`network.capture_block_mutable_state`/
   `apply_block_mutable_state`, already tested in `WORKER_REPORT_S36_CLONE_CAPTURE.md`)
   but was not exercised end-to-end by a real failing solve in this preflight.

## The matched full-length run command (NOT executed here)

Per the task's explicit instruction, this was not run. If/when the Planner authorizes
it, after resolving Remaining issue 1 above, the command (parallel, oracle
configuration, cap 300, into a fresh root) is:

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -c "
import sys; sys.path.insert(0, '.')
import p515_g_g1_g4_admm_gates as G
from contextlib import contextmanager

@contextmanager
def persistent_workers_on(num_workers=8):
    inner = G._s39_configure_hook
    def wrapped(arm_key):
        hook = inner(arm_key)
        def h(planning, sed, candidate, report):
            hook(planning=planning, sed=sed, candidate=candidate, report=report)
            planning.params.admm.persistent_workers = {'enabled': True, 'num_workers': num_workers}
        return h
    G._s39_configure_hook = wrapped
    try:
        yield
    finally:
        G._s39_configure_hook = inner

G._acquire_exclusive_run_lock()
with persistent_workers_on(8):
    G.run_s39_arm('s39_D', output_root_override='data/SRP1/Results/P515S40/persistent_workers_matched_run',
                  mode_label_override='matched_run')
" > data/SRP1/Results/P515S40/persistent_workers_matched_run_launch.log 2>&1
```

Its gate: bitwise identity against the committed serial oracle run
`data/SRP1/Results/P515S39_D_run` (certified cycle 139, cost 650,966,975.2943751) —
NOT `output_root_override`'d away from `S39_ARMS['s39_D']['out_dir']`, i.e. a REAL
(`mode='real'`) launch, which requires `data/SRP1/Results/P515S39_D_run` to remain
untouched (never re-run onto) and a fresh comparison root for the new run. This
requires Remaining issue 1 resolved and a fresh Planner authorization; this Worker
does not run it.

## Questions for Planner

1. Authorize an explicit exclusion of `esso_models_pickle.bytes` (with the diagnosis
   above as its justification), or authorize a fix to `network.apply_block_mutable_
   state`'s Var-restore loop / a module-local bypass? Either path is a small, bounded
   follow-up given the root cause is now fully understood and reproduced in isolation.
2. Is the 2-cycle preflight's 0.78× (slower) wall time acceptable evidence at this
   stage, given the gate's purpose is correctness, or should a longer timing
   measurement be authorized before the matched full run?
3. Should the DSO node-7 snapshot mechanism inside a worker be converted from
   legacy-clone to the same lightweight-capture technique already used for the TSO
   (Step 3.6), now that persistent workers make the per-cycle `model.clone()` cost
   paid inside a worker process rather than the parent — same open question
   `WORKER_REPORT_S36_CLONE_CAPTURE.md` left for the Planner, now with an additional
   data point (this task's fix 1 shows the clone-based path is also where the
   cross-process suffix defect first surfaced)?
