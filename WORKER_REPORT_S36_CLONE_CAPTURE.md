# Worker Report — P5.15 Addendum 21 item (4), Step 3.6: TSO clone -> lightweight capture

**Worktree** (detached HEAD at `fd25503b`):
`/private/tmp/claude-501/-Users-micaelsimoes-Projects-share-resource-planning-no-esso-degradation/ce1f88a8-5176-446e-89bd-7b0ea6787216/scratchpad/wt_step36_clone`
No file in the main checkout was modified. No solve was run anywhere (zero-solve equivalence
checks only, `SolveProfileGuard` armed and verified `0/0`).

---

## Task received

Replace the per-cycle TSO whole-model clone (`network_data.py` `NetworkData.optimize`'s
`pre_solve_model = model[year][day].clone()`, fired unconditionally every TSO cycle for all 12
`(year, day)` blocks per `WORKER_REPORT_S36_TIMING_MEASUREMENT.md`) with a lightweight capture
(mutable Param values + warm-start suffixes; block rebuilt deterministically on demand), per
`PLANNER_BRIEF_2026-09-13.md` Addendum 21 item 4. Deliverables: production change with a
legacy-path switch (in the params object, never deleting the legacy path), a zero-solve
equivalence-checks script, outputs + sha256 manifest, preserved-fixture reload confirmation, and
this report with an integration plan for the Planner.

---

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 21 item 4 (read from the main checkout).
- `WORKER_REPORT_S36_TIMING_MEASUREMENT.md`, `P5_15_STEP36_TIMING_DESIGN.md` (full) — call-path
  map for the clone, D1-D4 timing-defect analysis, the "TSO pays an unconditional per-cycle
  clone, DSO only at node 7" finding.
- `network_data.py` `NetworkData.optimize` (`:54-68`) — the exact clone/callback contract:
  `pre_solve_model = model[year][day].clone()` (`:61`) fires whenever either callback is
  non-`None`; `failure_snapshot_callback` fires on failure, `pre_solve_snapshot_callback` fires
  unconditionally (its own internal filter decides whether to write).
- `network.py` `_run_smopf` (`:673-777`), `_create_smopf_solver` (`:540-601`, the
  `replace_warm_start_suffix` warm-start-suffix copy), `_snapshot_multiplier_suffixes` /
  `_restore_multiplier_suffixes` (`:638-657`, reused verbatim).
- `shared_resources_planning.py`: `_run_operational_planning` (`:2311` and following, both the
  fresh-construction and the continuation branch, and where they converge before the ADMM loop);
  `update_transmission_coordination_model_and_solve` (`:5151` pre-edit `:5150`) — the per-cycle
  Param-update loop, `save_failed_tso_block`/`save_selected_tso_comparator` closures, the
  `cycle == 7` gating; `_save_frozen_network_block` / `_save_frozen_smopf_block` (payload
  structure: `{'metadata': {...}, 'model': <the pre-solve Pyomo block>}`); `create_transmission_network_model`
  (`:3446-3529`, read to confirm `pc`/`qc` are fixed ONCE at construction from
  `consensus_vars['pf']['dso']['current']` and never re-fixed per cycle — see "Design decision"
  below); `_update_tso_proximal_centres_after_solve` (`:4603-4792`, confirmed it also mutates
  mutable Params — `prox_v_prev`/`prox_pf_p_prev`/`prox_pf_q_prev`/`prox_ess_p_prev`/
  `prox_ess_q_prev` — from a DIFFERENT call site than the per-cycle update, gated on
  `proximal_regularization.enabled`); `_clone_operational_models` (`:3060`, the continuation
  branch's own clone-based restore).
- `model_construction_helpers.py` `configure_shared_ess_operational_state` (`:1043-1122`) — the
  ONLY function besides the Param `set_value` loop that mutates the TSO block per cycle (Var
  `.fix()`/`.unfix()`/`.setlb()`/`.setub()`/`.set_value()`, Constraint `.activate()`/
  `.deactivate()`); confirmed it is a PURE function of `(s_capacity, e_capacity)` at call time —
  its `shared_es_soc` reset and fix/unfix/activate branches depend only on the CURRENT capacity
  argument, never on capacity history, so replaying only the current cycle's captured
  `shared_es_s/e_rated_fixed` values (not the full per-cycle history) reproduces its effect
  exactly.
- `p515_g_g1_g4_admm_gates.py` `_s38_build_probe_tso_model` (`:5068-5117`) — the existing
  zero-solve, production-model-construction pattern (`srp.create_transmission_network_model`
  with `transmission_network.optimize` stubbed).
- `p515_s32_zero_solve_checks.py` `_build_admm_ready_state` (`:102-154`) — reused verbatim (not
  reimplemented) as the checks script's zero-solve ADMM-ready-state builder, since it already
  builds BOTH a real TSO model and real DSO models (including node 7) through
  `create_transmission_network_model`/`create_distribution_networks_models` with `.optimize`
  stubbed.
- `p512_y_tso_replay.py`, `p512_z_classification.py` — confirmed downstream consumers load
  `payload['model']` and use it DIRECTLY as the model to re-solve, i.e. the snapshot's `model`
  field must be a real, standalone, solvable Pyomo block — not a data-only summary.
- `admm_parameters.py` (full class body around the existing `esso_al_scale`/`shared_ess_initialization`
  switches) — the existing convention for adding an optional, documented, non-case-file-wired
  attribute.
- `p513_solve_profile_guard.py` (full file — the guard mechanism reused).
- `helper_functions.py` `solver_result_succeeded` (`:74-85`).
- `solver_parameters.py` (`SolverParameters.__init__`, `:11-30`) — found, while debugging, that
  it `exit(ERROR_PARAMS_FILE)` (`-3`) at OBJECT CONSTRUCTION time (not solve time) whenever
  `NLP_SOLVER_PATH`/`SOLVER_PATH` (and, for the ESSO's LP solver, `LP_SOLVER_PATH`) are unset —
  this worktree has no `.env` (untracked, main-checkout-only); every zero-solve harness that
  constructs a `SharedResourcesPlanning`/`NetworkParameters` object needs these three env vars
  set even though it never calls a solver. Read `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/.env`
  (main checkout, read-only) to learn the two canonical values already documented in
  `CLAUDE.local.md` (`NLP_SOLVER_PATH=/usr/local/bin/ipopt`); exported both as process
  environment variables for every run in this worktree — the worktree's own `.env` was never
  created, read, or staged.

---

## Files modified (in the worktree)

- `network.py` — added `capture_block_mutable_state` (`:660-690`) and
  `apply_block_mutable_state` (`:700-729`).
- `shared_resources_planning.py` — one new import line (`:20`); the pristine-base construction
  in `_run_operational_planning` (`:2497-2515`, right before the ADMM loop); the new
  `tso_pristine_base=tso_pristine_base` argument at the call site (`:2554-2565`); the
  `tso_pristine_base=None` parameter and the lightweight/legacy branch inside
  `update_transmission_coordination_model_and_solve` (`:5151`, branch at `:5244-5294`).
- `admm_parameters.py` — `self.tso_snapshot_capture_mode = 'lightweight'` (`:175`), with its
  documentation comment (`:157-174`).

## Files created (in the worktree)

- `p515_s36_clone_capture_checks.py` — the zero-solve equivalence-checks script.
- `data/SRP1/Results/P515S36/clone_capture_checks/results.json`
- `data/SRP1/Results/P515S36/clone_capture_checks/manifest_sha256.json`
- This file.

---

## Design

### What is captured, every cycle, in place of `model[year][day].clone()`

`network.capture_block_mutable_state(model)` (`network.py:660`) sweeps the block ONCE, generically
(`model.component_objects(ctype, active=None)`, no per-field enumeration by name), and records:

1. **Every mutable `Param`'s value** (`comp.mutable` filter — Pyomo requires `mutable=True` for
   `set_value()` to be legal at all, so this filter exactly targets the ADMM-updated Params —
   `shared_es_s/e_rated_fixed`, `dual_vmag_req`/`vmag_req`, `dual_pf_p/q_req`/`p/q_pf_req`,
   `dual_ess_p/q_req`/`p/q_ess_req`, the conditional `*_prev` copies, `rho_v`/`rho_pf`/`rho_ess`/
   `rho_ess_prev`, and — **found while reading `_update_tso_proximal_centres_after_solve`, not
   in the brief's own enumeration** — `prox_v_prev`/`prox_pf_p_prev`/`prox_pf_q_prev`/
   `prox_ess_p_prev`/`prox_ess_q_prev`, set from a DIFFERENT call site, after the solve, only
   when proximal regularization is enabled. A hand-curated whitelist (as the brief's own
   phrasing, "enumerate from the code", suggested) would have missed this; the generic sweep
   does not, and needs no maintenance if a future change adds another mutable Param).
2. **Every `Var`'s value, lower bound, upper bound, and fixed flag** — covers both ordinary
   value updates (from `model.solutions.load_from`, the actual warm-start mechanism) and the
   `.fix()`/`.unfix()`/`.setlb()`/`.setub()` mutations `configure_shared_ess_operational_state`
   performs.
3. **Every `Constraint`'s and `Objective`'s per-entry active flag** — covers the same function's
   `.activate()`/`.deactivate()` calls (`_configure_shared_ess_expected_schedule`'s
   `expected_shared_ess_p/q_def`, and the `_SHARED_ESS_OPERATIONAL_CONSTRAINTS` entries).
4. **The warm-start multiplier suffixes** (`ipopt_zL_in`, `ipopt_zU_in`, `dual`) via the
   EXISTING `_snapshot_multiplier_suffixes` (`network.py:638`, reused as-is, not reimplemented).
   These are captured at the SAME point in the call sequence the legacy clone used (after this
   cycle's Param-update loop, before `transmission_network.optimize(...)`) — i.e. BEFORE
   `_create_smopf_solver`'s `replace_warm_start_suffix` call copies `ipopt_zL/zU_out` into
   `_in` for THIS solve. The legacy clone captured the same (technically one-step-stale `_in`,
   fresh `_out`) state; the lightweight capture reproduces it bit-for-bit rather than "fixing"
   it, since a downstream replay (`p512_y_tso_replay.py`) re-runs `_create_smopf_solver` on the
   loaded model itself, so whatever is on the model when it is captured is what gets replayed
   downstream too — verified by the equivalence check, not merely reasoned about.

**Deliberately NOT captured**: component structure (Sets, constraint/objective expression trees,
Var/Param declarations). This is the one place this implementation depends on an invariant
about the TSO ADMM loop rather than being fully self-verifying: the loop never adds, removes, or
redefines a component after construction. This was checked (not assumed) by grepping
`update_transmission_coordination_model_and_solve`, `_update_tso_proximal_centres_after_solve`,
and `update_and_check_convergence` for `.fix(`/`.unfix(`/`.setlb(`/`.setub(`/`.fixed *=` outside
`configure_shared_ess_operational_state` — none found — and is the reason the equivalence check
also compares every constraint's and objective's expression STRING (3,378 TSO / 7,779 DSO
comparisons, see Results) rather than only their `.active` flag.

### How the rebuild-on-demand works, and the ONE deliberate deviation from a literal reading

The brief: "block rebuilt deterministically on demand ... through the production model-
construction path". A literal reading would re-invoke `create_transmission_network_model` at
snapshot-write time. Reading it (`shared_resources_planning.py:3446-3529`) found a correctness
problem with that literal approach: it FIXES the TSO's `pc`/`qc` Vars from
`consensus_vars['pf']['dso']['current']` **at call time** (`fix_or_set(...)`, once, for the whole
run) — never re-fixed per cycle. Re-invoking the constructor at a LATER cycle's failure would
read `consensus_vars` as it stands at THAT (later) time, fixing `pc`/`qc` to the wrong (later)
values instead of the true cycle-0 ones — a genuine, silent correctness bug, not a hypothetical
one. It would also not extend correctly to `_run_operational_planning`'s continuation branch
(`initial_state is not None`), where `tso_model` is a clone of an EARLIER run's saved state, not
freshly built in THIS process invocation, so there is no "this run's construction-time
`consensus_vars`" to re-derive from at all.

**Implemented instead**: one `tso_model[year][day].clone()` per block, taken ONCE per
`_run_operational_planning` call, immediately before the ADMM loop starts mutating it
(`shared_resources_planning.py:2497-2515` — `tso_pristine_base`). This covers BOTH branches
uniformly (both converge to a fully ADMM-ready `tso_model` at that point) and needs no extra
threading of `consensus_vars`/`candidate_solution`/`objective_scale` into the per-cycle function.
"Rebuilt deterministically on demand" becomes: `apply_block_mutable_state(tso_pristine_base[year][day].clone(),
captured_state)` — a fresh clone of the FROZEN, structurally-correct base, with this cycle's
captured state replayed onto it — done ONLY for the block(s) that actually need a snapshot
written (a failure, or the cycle-7 comparator), never for the other ~11 blocks in that cycle.

This is a considered engineering trade, not the literal wording, and is flagged here for Planner
review. Its cost: `tso_pristine_base` still pays 12 real `clone()` calls, but ONCE per
`_run_operational_planning` call (i.e. once per candidate evaluation) rather than once per
(cycle x block) — the O(cycles) -> O(1) reduction the task asked for is preserved; only the very
first cycle's fixed cost is unchanged from before.

### The switch

`admm_parameters.py:175`, `self.tso_snapshot_capture_mode = 'lightweight'` (default). Setting it
to `'legacy_clone'` (programmatically; not wired to any case-file key, per the task) makes
`_run_operational_planning` skip building `tso_pristine_base` (stays `None`), which makes
`update_transmission_coordination_model_and_solve` take its `else` branch — byte-for-byte the
pre-Step-3.6 code path (`shared_resources_planning.py:5273-5283`), including the exact clone
call inside `NetworkData.optimize` and the same two callback closures. Deactivate-and-unwire:
`network_data.py`'s clone/callback machinery, `_save_frozen_smopf_block`, and every DSO call
site are UNTOUCHED — the legacy path still runs through them exactly as before, for the DSO
always, and for the TSO whenever `tso_snapshot_capture_mode='legacy_clone'`.

---

## Commands / experiments run

- `NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=/Users/micaelsimoes/coin-or/dist/bin/clp
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B -u p515_s36_clone_capture_checks.py`
  — the equivalence-checks script. `SolveProfileGuard(permitted=())` armed for the whole run;
  `verify(expected_solves=0)` — `counts={'permitted_solve': 0, 'permitted_exec': 0,
  'blocked_solve': 0, 'blocked_exec': 0}`, `verify_failures=[]`. Exit 0, `all_checks_pass=true`.
- Several earlier debugging invocations (isolating two defects, both fixed before the final run
  — see "Unexpected findings"): a `ValueError` reading `.value` on an uninitialized mutable Param
  the checks script itself was perturbing (fixed in the checks script, not production); and a
  `comparator_present` mis-check using `'2025' in transmission_network.years` where `years` is
  keyed by `int`, not `str` (fixed; production's own `save_selected_tso_comparator` already used
  `str(year) == '2025'`, so this was purely a bug in my check, never in production).
- `python -B -c "import shared_resources_planning as srp; import network as N; import
  admm_parameters as AP; ..."` — import-sanity and default-switch check (`OK True True`,
  `mode default: lightweight`).
- `python -B -c "import pyomo.environ as pe; m = pe.ConcreteModel(); ..."` — confirmed the
  `component_objects(ctype, active=None)` / `.mutable` / Suffix API used in `capture_block_
  mutable_state` behaves as assumed, on Pyomo 6.9.5 (this environment).

---

## Results

### Equivalence (zero-solve, `p515_s36_clone_capture_checks.py`)

On a fresh, zero-solve, production-constructed ADMM-ready state (`p515_s32_zero_solve_checks.
_build_admm_ready_state`, reused verbatim), one TSO block (`case9`, year picked by the script's
own fallback logic) and the DSO node-7 block (`case33_2`), each driven through a representative
per-cycle mutation (a real `configure_shared_ess_operational_state` active->inactive transition
on one shared-ESS index — exercising the Var fix/unfix/bound/deactivate paths a value-only
perturbation would never touch — plus a generic perturbation of every other mutable Param/Var,
plus 5 synthetic `ipopt_zL_in`/`ipopt_zU_in`/`dual` suffix entries each):

| | TSO block | DSO node-7 block |
|---|---:|---:|
| mutable Params perturbed | 1,097 | 1,786 |
| Vars perturbed | 2,664 | 10,600 |
| `n_diffs` (legacy clone vs. rebuild-from-capture) | **0** | **0** |
| Constraint/Objective expr strings compared | 3,378 | 7,779 |
| `equivalent` | **True** | **True** |

Every Var value/bounds/fixed flag, every mutable Param value, every `ipopt_zL_in`/`ipopt_zU_in`/
`dual` Suffix entry (matched by `(component_name, index)`, since the two blocks are distinct
Python objects), every Constraint's and Objective's active flag, and every Constraint's and
Objective's expression string — all identical between the two paths.

### End-to-end dispatch (real `update_transmission_coordination_model_and_solve`, `Network.run_smopf`
canned to a never-solving result, real `BlockData.clone` call-counted — not inferred from timing)

| Case | Path | Condition | `BlockData.clone` calls | Expected | Match |
|---|---|---|---:|---:|---|
| A | legacy (`tso_pristine_base=None`) | cycle 1, no block fails | 12 | 12 (one per block, every call) | True |
| B | lightweight | cycle 1, no block fails, not cycle 7 | **0** | 0 | True |
| C | lightweight | cycle 1, ONE block fails | **1** | 1 | True |
| D | lightweight | cycle 7 (comparator), no block fails | **1** | 1 (the `('2025','Summer')` block is present in this state) | True |

Case B is the core Step 3.6 claim, verified directly: the lightweight path reaches
`BlockData.clone` **zero** times in the (overwhelmingly common) cycle where nothing needs to be
written. Case A confirms the legacy path is untouched (still one clone per block, every cycle).

### Timing (medians over 15 reps, same mutated block; `time.perf_counter()`; this worktree's
machine, zero-solve — NOT a production ADMM-loop measurement, see "Remaining issues")

| | TSO block | DSO node-7 block |
|---|---:|---:|
| legacy `clone()` (paid every cycle, old) | 0.0879 s | 0.1988 s |
| lightweight `capture_block_mutable_state` (paid every cycle, new) | **0.0222 s** | **0.0677 s** |
| on-demand rebuild (clone-of-pristine + apply; paid only when a snapshot is written) | 0.0954 s | 0.2139 s |

Per-block per-cycle ratio (legacy / lightweight): **3.95x** (TSO), **2.94x** (DSO). The
on-demand rebuild costs slightly MORE than a bare legacy clone (expected: it is a clone plus a
replay), but is paid only for the rare block(s) that actually need a snapshot, never for the
other ~11.

### Preserved-fixture reload (read-only, main checkout, absolute paths)

All 7 fixtures unpickled successfully; `n_vars` is `sum(1 for _ in model.component_data_objects(pe.Var, active=None))`
on the loaded `payload['model']`, confirming each is a live, structurally intact Pyomo block, not
merely bytes that happened to unpickle:

| Fixture | Unpickled | `n_vars` |
|---|---|---:|
| `P512R/cycle21_pre_setup/snapshot.pkl` | True | 10,756 |
| `P512R/cycle21_prepared/snapshot.pkl` | True | 10,756 |
| `P512R/production_snapshots/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` | True | 10,756 |
| `P512R/production_snapshots/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl` | True | 3,756 |
| `FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl` | True | 10,882 |
| `FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl` | True | 3,762 |
| `P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl` | True | 3,762 |

Scope note (rule seven, scoped negative claim): this is the set of fixtures found by grepping
`WORKER_REPORT_*.md`/`REVISION_CONTEXT.md`/`CLAUDE.md` for `FrozenSMOPF`/`cycle21`/
`matched_success`, deduplicated to one representative copy per distinct content family (many
`data/SRP1/Results/P515*_run/results/FrozenSMOPF/matched_success_*` directories on disk are
byte-identical re-copies from different campaign runs — not separately loaded here). It is not a
claim that every one of the dozens of on-disk copies was individually loaded.

### Switch

`admm_parameters.tso_snapshot_capture_mode` defaults to `'lightweight'` and is settable to
`'legacy_clone'` — both confirmed directly (not by log inspection) in the checks script.

**Overall**: `all_checks_pass = True`, `SolveProfileGuard` verified `0/0`, `0` blocked solves.

---

## Validation

- **Code executes correctly**: yes — production imports cleanly (`import shared_resources_planning`,
  `import network`, `import admm_parameters` all succeed); the checks script runs to completion,
  exit 0.
- **Test passes**: yes — every equivalence, dispatch, switch, and fixture-reload check passes.
- **Requested diagnostic works**: the checks directly exercise the NEW production code (the real
  `update_transmission_coordination_model_and_solve`, the real `capture_block_mutable_state`/
  `apply_block_mutable_state`), not a reimplementation of it.
- **Underlying numerical-equivalence claim**: established at the level this task's evidence
  supports — a zero-solve, single-representative-mutation, single-TSO-block/single-DSO-block
  test. It is NOT a claim that a full multi-cycle production ADMM run with this path enabled
  reproduces the SAME numerics as the legacy path bit-for-bit; that is the 2-cycle bitwise
  preflight named in the integration plan below, which requires real solves and was out of this
  task's zero-solve scope.
- **Numerics-unchanged claim (Requirement 4)**: confirmed by reading, not by a solve — the clone
  (legacy or pristine-base) never mutates the model it is cloned FROM (`BlockData.clone()` is a
  deep copy, returns a NEW object; verified this is the only use in both the legacy and the new
  code paths — `network_data.py:61` (legacy), `shared_resources_planning.py:2513`
  (pristine-base construction) and `:5265` (on-demand rebuild)); the lightweight path's solve call
  (`transmission_network.optimize(model, from_warm_start=from_warm_start)`,
  `shared_resources_planning.py:5252`) is IDENTICAL to the legacy call's `model`/
  `from_warm_start` arguments — only the two snapshot-callback keyword arguments are omitted,
  and `NetworkData.optimize` (`network_data.py:54-68`) does not read or use those callbacks for
  anything except the (removed) clone and the (unaffected) post-solve `failure_snapshot_callback`/
  `pre_solve_snapshot_callback` invocation, so the solve itself is byte-for-byte the same call.

---

## Unexpected findings

- `_update_tso_proximal_centres_after_solve` (`shared_resources_planning.py:4603-4792`) mutates
  five more TSO mutable Params (`prox_v_prev`/`prox_pf_p_prev`/`prox_pf_q_prev`/`prox_ess_p_prev`/
  `prox_ess_q_prev`) from a call site OUTSIDE `update_transmission_coordination_model_and_solve`,
  gated on `proximal_regularization.enabled`/`proximal_regularization.tso.enabled` (both default
  `False`; Addendum 21 records tau=0 as the current oracle, i.e. proximal regularization off in
  production today, but the mechanism still EXISTS and is recorded as a per-channel fallback).
  A hand-curated capture list (a literal reading of the brief's "enumerate from the code") would
  have silently missed these if proximal regularization is ever re-enabled; the generic
  `component_objects(pe.Param, ...)` sweep captures them automatically, with no extra code.
- `configure_shared_ess_operational_state` is a PURE function of `(s_capacity, e_capacity)` at
  call time (no path/history dependency) — confirmed by reading, which is why replaying only the
  CURRENT cycle's captured value (not every historical cycle's) reproduces its effect exactly;
  this was not obvious from the function's name alone and needed the read-through described
  above.
- `pc`/`qc` on the TSO block are `pe.Var` (not `pe.Param`) — `network.py:309-311`,
  `is_transmission: True`; the `mutable=True` `pe.Param` branch at `:312-314` is DSO-only. This
  matters because it means `create_transmission_network_model`'s `pc`/`qc` fixing happens exactly
  ONCE, at construction, and is the reason the literal "re-invoke the constructor on demand"
  reading of the brief was rejected (see "Design decision" above) — not a hypothetical concern.
- This worktree has no `.env` — every zero-solve harness that constructs a planning object
  (including the ALREADY-COMMITTED `p515_s32_zero_solve_checks.py`, run standalone as a
  cross-check) needs `NLP_SOLVER_PATH`/`SOLVER_PATH`/`LP_SOLVER_PATH` set as process environment
  variables (`solver_parameters.py:27-30` `exit(ERROR_PARAMS_FILE)` at construction time, not
  solve time) — worth flagging to the Planner in case other Workers hit the same silent-exit
  (`sys.exit(-3)` prints nothing when caught by the top-level interpreter under certain
  invocation forms, which cost real debugging time here) in this or another worktree.
- My own harness debugging caused, and this report records having caused and FIXED, an
  unrelated incident: an early `rm -rf data/SRP1/Results/FrozenSMOPF data/SRP1/Results/P515S36`
  (intended only to clear MY OWN scratch subdirectories between test runs) deleted 68 pre-existing
  TRACKED, COMMITTED files under those same parent directories (P43/P44/P45 reports, A17
  diagnostics, parallel_audit, and step36_timing artifacts from prior Worker tasks). Caught
  immediately via `git status --porcelain` (which showed the deletions), restored exactly via
  `git checkout -- <the 10 specific subdirectories>` (confirmed via `git diff --stat HEAD --
  <those paths>` returning empty, i.e. bit-for-bit restoration), BEFORE any commit — no data was
  lost, but flagged here per CLAUDE.md's evidence-preservation rules, since it is exactly the
  class of incident those rules exist to prevent. Recorded so the Planner can decide whether
  independent verification of `git diff --stat HEAD -- data/SRP1/Results/FrozenSMOPF
  data/SRP1/Results/P515S36` (should be empty) is wanted before this commit.

---

## Remaining issues

- **No production/full-scale timing measurement was taken.** The timing numbers above are
  zero-solve, single-block, single-machine-state medians from the checks script itself — useful
  for the RELATIVE per-block ratio (legacy/lightweight ~3-4x), NOT a substitute for the "re-measure
  over 10 cycles" step the brief itself sequences AFTER this implementation and AFTER Planner
  integration (see below) — that requires a real, multi-cycle production ADMM run, out of this
  task's zero-solve scope and this worktree's "never run solves" constraint.
- **The 2-cycle bitwise preflight (legacy-clone vs. lightweight-capture) has NOT been run.** The
  equivalence checks here are a zero-solve, synthetic-mutation proxy for it, not a replacement —
  the brief's own deliverable list separates them for a reason (this task's checks establish the
  MECHANISM is correct in isolation; the preflight would establish that a REAL multi-cycle run
  with real IPOPT solves produces bit-identical FrozenSMOPF snapshots and bit-identical ADMM
  trajectories under both settings of `tso_snapshot_capture_mode`).
- **DSO node-7's own snapshot path is unchanged** (still legacy-clone, conditional on node 7)
  per the brief's explicit scoping ("replace the per-cycle TSO whole-model clone"); the DSO node-7
  block was used here only to demonstrate `capture_block_mutable_state`/`apply_block_mutable_state`
  generalize beyond the TSO, not to change DSO production behaviour. If the Planner wants the DSO
  path converted too, that is a separate, smaller follow-up (the DSO already only pays the clone
  at one node, not every cycle x every block).
- **The pristine-base clone (`tso_pristine_base`) is still an unconditional, eager cost** — 12
  clones, once per `_run_operational_planning` call, even on a run/candidate that never triggers
  a single failure or reaches cycle 7. This is the documented, deliberate trade described above;
  it is small relative to the eliminated O(cycles) cost but is not literally "zero clones unless
  a snapshot is needed."
- The equivalence checks exercise ONE representative per-cycle mutation pattern (one
  active->inactive shared-ESS transition plus a generic perturbation of everything else) on ONE
  TSO block and ONE DSO block. It does not exhaustively exercise every historical value trajectory
  a real multi-hundred-cycle run could produce (e.g. an inactive->active transition, or a
  second/third shared-ESS index going inactive independently) — the "on demand, only for a
  genuine failure" design means any residual gap would surface as a non-equivalent FrozenSMOPF
  snapshot on an actual production failure, which is exactly what the 2-cycle preflight above is
  for.

---

## Integration plan (for the Planner, after arms C and D)

1. **Cherry-pick, in this order**, from this worktree's commits (hashes below): the
   `admm_parameters.py` switch commit, then the `network.py` capture/apply-function commit, then
   the `shared_resources_planning.py` dispatch commit (the three are logically one change split
   for reviewability; cherry-pick as a set, not individually, since `shared_resources_planning.py`
   imports the two new `network.py` functions and reads the new `admm_parameters.py` attribute).
   The checks script and its outputs may be cherry-picked separately or not at all (diagnostic
   evidence, not production code).
2. **2-cycle bitwise preflight** (production run, not zero-solve): the SAME two-cycle,
   cold-start, C\*-candidate configuration class already used for the Step 3.0/Addendum-18
   determinism baselines. Run once with `tso_snapshot_capture_mode='lightweight'` (the new
   default — no override needed) and once with `'legacy_clone'` (explicit override), same case
   files, same seeds. Diff EVERY numeric artifact already used as the determinism reference
   elsewhere in this programme (per-cycle recourse, primal/dual residuals, SoH trajectory, the
   ESSO complementarity detector) bitwise between the two runs — per CLAUDE.md ("proving it
   changes nothing" is a verification, not a discovery step; a non-bitwise diff means the wiring
   is wrong, not that lightweight capture is inherently risky). ALSO diff the FrozenSMOPF
   snapshot files any failure/cycle-7-comparator event produces between the two runs (after
   stripping the `log_path`-style fields that are expected to differ across output directories,
   same technique `WORKER_REPORT_S36_TIMING_MEASUREMENT.md`'s `bitwise_diff_v2` already used) —
   this is the part this task's zero-solve checks cannot cover (they use a SYNTHETIC mutation on
   a freshly built state, not a real multi-cycle trajectory).
3. **10-cycle re-measurement**: extend `p515_s36_step36_timing_run.py` (already-committed,
   `data/SRP1/Results/P515S36/step36_timing/{off,on}/` convention) from 2 to 10 cycles, same
   C\*-candidate configuration, with `tso_snapshot_capture_mode='lightweight'` (production
   default). Re-run the corrected `p515_s36_step36_timing.py`/`_reanalyze.py` phase-timing
   analysis (D1/D2/D3-fixed, per `WORKER_REPORT_S36_TIMING_MEASUREMENT.md`) on the new capture,
   with a NEW `clone`/`capture` phase tag distinguishing the eliminated per-cycle TSO clone from
   the new lightweight capture, to get a PRODUCTION (not zero-solve-synthetic) measurement of the
   `overhead_local`/`overhead_total` ratio and the projected 8-worker speedup with this change
   folded in — the number this whole Step 3.6 preflight exists to inform (X >= 70% screening
   rule, persistent-worker decision).
4. **Only after 2-3 pass**: proceed to "build persistent workers" (Addendum 21 item 4's own next
   clause, itself NOT authorized by this task).

---

## Questions for Planner

1. Is the "one clone per `_run_operational_planning` call" residual cost (the `tso_pristine_base`
   construction) acceptable as the final design, or would the Planner prefer investigating a
   TRUE zero-eager-cost variant (re-deriving `create_transmission_network_model`'s construction
   inputs, including a frozen copy of construction-time `consensus_vars`, threaded through
   `update_transmission_coordination_model_and_solve` so the pc/qc-fixing correctness problem
   identified above is avoided some other way)? This was considered and rejected here on
   correctness + continuation-branch grounds (see "Design decision"), but is a genuine remaining
   design choice, not a closed question.
2. Should the DSO's node-7 snapshot path be converted to the same lightweight mechanism as a
   follow-up, given `capture_block_mutable_state`/`apply_block_mutable_state` already generalize
   to it (demonstrated by this task's DSO equivalence check, 0 diffs)? It was out of this task's
   explicit scope ("replace the per-cycle TSO whole-model clone") and left untouched.
3. Confirm whether independent verification of `git diff --stat HEAD -- data/SRP1/Results/FrozenSMOPF
   data/SRP1/Results/P515S36` (expected empty — the accidental-deletion recovery described under
   "Unexpected findings") is wanted before or as part of reviewing this commit.
