# Worker Report — P5.15 Addendum 23 item 4, Step 3.7: Anderson acceleration

Worktree: `wt_step37_aa`, detached HEAD, base commit `a099c7c5`.
Commits made (this worktree only): `c5d18add` (production code),
`eacbdfcc` (zero-solve checks + outputs). No solves were run at any point
in this task; `p515_s41_aa_checks.py` is CODE + zero-solve validation only.

## Task received

Implement type-II Anderson acceleration (AA) for the ADMM loop behind a
default-off flag, with zero-solve checks, per `PLANNER_BRIEF_2026-09-13.md`
Addendum 22 (Step 3.7 bullet) and Addendum 23 (amendments (i)–(vi)), and
frozen spec `data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json`
(`item4_step_3_7_anderson`). Code only — no solves, no runs.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (full file, Addendum 1–23) — authority.
- `data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json` — binding spec.
- `data/SRP1/SRP1_params.json` (`admm` block) — production oracle configuration (rho, `shared_ess_reference_rating_mva=2.5`, `proximal_regularization.tso.tau=0.0`).
- `shared_resources_planning.py`: `_run_operational_planning` (the ADMM cycle loop, DSO→TSO→ESSO), `get_admm_boyd_residual_metrics`, `_update_admm_penalties`, `_scale_admm_penalty`, `_update_interface_power_flow_variables`, `_update_shared_energy_storage_variables`, `create_admm_variables`, `update_and_check_convergence`, `_admm_shared_ess_reference_mva`, `_shared_ess_admm_normalization_mva`.
- `admm_parameters.py` (full file) — `ADMMParameters.__init__`/`_read_parameters_from_file` conventions, in particular `persistent_workers`/`tso_snapshot_capture_mode` (programmatic-only flags, not case-file-wired) as the precedent followed for `anderson_acceleration`.
- `admm_persistent_workers.py` (module docstring, `persistent_workers_enabled`) — style/gating precedent.
- `p513_solve_profile_guard.py` — `SolveProfileGuard` API used by the checks script.
- `p515_s40_polish_failure_checks.py`, `p515_s32_zero_solve_checks.py` (`_build_admm_ready_state`, `_perturb_single_entry`) — conventions for P5.15 zero-solve checks; `_build_admm_ready_state` reused (not reimplemented) for Check D.
- `p515_g_g1_g4_admm_gates.py` (`S39_ARMS`, `_s39_ids_for_mode`, `_s39_working_dir_ids`, `run_s39_arm`) — the arm structure Step 3.7's own harness task (Planner-authorized, not built by me) must mirror.
- `p56a_oracle.py` (`fresh_planning`, `load_baseline`, `WORK_DIR`) — reused via `_build_admm_ready_state`.

## Files modified

- `admm_parameters.py` — new `self.anderson_acceleration` field.
- `shared_resources_planning.py` — import, construction of `aa_layout`/`aa_state`, four `if aa_enabled:` call sites inside `_run_operational_planning`, two new module-level helper functions, peak-RSS capture.

## Files added

- `admm_anderson_acceleration.py` — the AA module (layout/collect/write-back, `AndersonAccelerationState`, `combined_scaled_residual`).
- `p515_s41_aa_checks.py` — zero-solve checks.
- `WORKER_REPORT_S41_AA.md` — this report.

## Design: w, F, g, scaling, write-back (with file:line)

**F, one ADMM cycle.** `shared_resources_planning.py:_run_operational_planning`,
the `for iter in range(1, admm_parameters.num_max_iters + 1):` loop
(starts at the line just above `shared_resources_planning.py:2568` in the
committed diff): DSO solve + consensus/dual update (`update_tn=False,
update_dns=True`) → TSO solve + proximal-centre update + consensus/dual
update (`update_tn=True`) → ESSO solve + consensus/dual update
(`update_sess=True`). One iteration of this loop, restricted to its effect
on the consensus/dual stores, is `F`.

**w = (z, u), per channel, file:line:**

| channel | z (consensus) | u (scaled dual) |
|---|---|---|
| V | `consensus_vars['vmag']['tso']['current'][node][year][day][p]`, scaled by `v_base` (matches `get_admm_boyd_residual_metrics:shared_resources_planning.py:6163-6164`, now at that line number in this worktree's committed HEAD before my edits — see `git show a099c7c5:shared_resources_planning.py` for the pre-edit numbering, `6163` `r_v = (x_dso_v - z_tso_v) / v_base`) | `dual_vars['vmag']['dso']['current'][node][year][day][p] / v_base / rho_v` — the DSO's stored dual (`y_v` in the boyd docstring, pre-edit `6161,6168`), divided by the channel's rho to convert the codebase's UNSCALED multiplier into Boyd's scaled dual |
| PF | `consensus_vars['pf']['tso']['current'][node][year][day][pt][p]`, scaled by `interface_rating` (pre-edit `6192-6193`) | `dual_vars['pf']['dso']['current'][node][year][day][pt][p] / s_base_dso / rho_pf` (pre-edit `6197`) |
| ESS | `consensus_vars['ess']['z']['current'][node][year][day][pt][p]`, scaled by `2*S_ref` (`S_ref = admm_parameters.shared_ess_reference_rating_mva`; see `_admm_shared_ess_reference_mva`, pre-edit `4287-4297`, and `_shared_ess_admm_normalization_mva`, pre-edit `4300-4303`) | THREE per-agent duals: `dual_vars['ess'][agent]['current'][node][year][day][pt][p] / rho_ess` for `agent in ('tso','dso','esso')` — RAW, no v_base/s_base factor, matching `y_agent` at pre-edit `6218,6228-6229` |

This is implemented in `admm_anderson_acceleration.py::build_iterate_layout`
(entry ordering/scale, static per run) and `::collect_w`/`::write_back_w`
(the physical↔scaled conversion, called fresh every cycle since rho
changes over the run). `collect_w`/`write_back_w` are pure Python/NumPy —
no Pyomo model access — so they are directly unit-testable (Check D).

**Why u = y/rho, not y directly.** The codebase's own dual-ascent update
is the classical UNSCALED form: `_update_interface_power_flow_variables`
(pre-edit `7257-7259` for the TSO copy, `7264-7269` for the DSO copy — in
this worktree's committed diff these are now at `7444`/`7454` after my
~90-line insertion earlier in the file; see
`grep -n "dual_vars\['vmag'\]\['tso'\]\['current'\].*+=" shared_resources_planning.py`)
does `dual_vars[...] += rho * error`, i.e. `y^{k+1} = y^k + rho*(x-z)`
(Boyd 2011 eq. 3.3, unscaled form), not the scaled-dual recursion
`u^{k+1} = u^k + (x-z)`. AA's fixed-point-map theory (Zhang/O'Donoghue/Boyd
2020; Fu/Zhang/Boyd 2020) is stated on the SCALED iterate `(z,u)`, and the
task instructions require stating "how the scaled dual u relates to the
stored duals (y/ρ convention)" explicitly — this table, and the module
docstring, are that statement. Because AA's memory is cleared on every rho
change (Addendum 23 amendment (i)), rho is constant within any window the
memory spans, so `u = y/rho` is a well-defined, consistent rescaling
throughout that window.

**TSO/DSO dual mirror invariant.** `_update_interface_power_flow_variables`
updates BOTH `dual_vars[...]['tso']['current']` and `dual_vars[...]['dso']['current']`
in the SAME call, with opposite-signed errors (`error_v_req_tso =
tso_v-dso_v`, `error_v_req_dso = dso_v-tso_v = -error_v_req_tso`) and the
SAME rho (rho is shared across every agent within a channel — see below),
so `dual_vars[...]['tso']['current'] == -dual_vars[...]['dso']['current']`
exactly, by induction from the shared zero initial value
(`create_admm_variables`, pre-edit `4147-4148`). `collect_w` treats the
DSO copy as THE canonical `u` (only one independent u per V/PF entry,
matching Boyd two-block ADMM's single dual) and ASSERTS the mirror
relation on every read (`admm_anderson_acceleration._antisymmetry_check`)
rather than assuming it holds forever; `write_back_w` writes BOTH copies
(`dso = y`, `tso = -y`) explicitly so the TSO's own next-cycle read of its
copy stays consistent.

**Single rho per channel.** `_update_admm_penalties`'s "Apply common
group-wise scaling factors" block (pre-edit `7099-7122`, unchanged by this
task) applies the SAME multiplicative `factors[group]` to every TSO/DSO
model's `rho_v`/`rho_pf`/`rho_ess` AND to every ESSO model's `rho`, every
cycle; the case file (`SRP1_params.json`, `admm.rho`) seeds them equal
across agents. `shared_resources_planning._get_admm_rho_channel_scalars_for_aa`
(new, `shared_resources_planning.py:6413`) reads the raw per-agent values
and RAISES if they are ever non-uniform, rather than silently averaging or
picking one — this is the guard the `u = y/rho` construction depends on.

**Proximal centres.** Checked, not touched. The oracle configuration
(`SRP1_params.json`: `proximal_regularization.tso.tau = 0.0`,
`gamma_policy = 'tied_to_rho'` ⇒ `prox_gamma_c = tau*rho_c = 0`) makes the
TSO proximal term `(gamma/2)*(x-prox_prev)**2` multiply by `gamma == 0`,
so it is inert regardless of `prox_prev`'s value — verified by reading
`update_transmission_model_to_admm`'s construction of that term (only ever
scaled by `prox_gamma_*`). Documented in `write_back_w`'s docstring; a
future run with `tau != 0` would need this reconsidered (flagged as an
open item below).

**g = F(w) − w.** Computed by the caller
(`shared_resources_planning._anderson_acceleration_cycle_step`,
`shared_resources_planning.py:6459`) as `w_after - w_before`: `w_before`
is snapshotted at the TOP of the cycle (before the DSO solve,
`shared_resources_planning.py:2581-2585`, guarded by `if aa_enabled:`),
`w_after` is collected right after the ESSO consensus/dual update and
`get_admm_boyd_residual_metrics` have both run (same rho throughout, since
rho only changes once per cycle, at the very end).

## Type-II AA and the safeguard (the implemented choice)

`admm_anderson_acceleration.AndersonAccelerationState` (module docstring
has the full derivation). Memory `m=5`; with history
`w_{k-m_k},...,w_k`, `g_{k-m_k},...,g_k`:

```
DeltaW = [w_{k-m_k+1}-w_{k-m_k}, ..., w_k-w_{k-1}]
DeltaG = [g_{k-m_k+1}-g_{k-m_k}, ..., g_k-g_{k-1}]
gamma* = solve( (DeltaG^T DeltaG + 1e-10 I) gamma = DeltaG^T g_k )
w_hat  = w_k + g_k - (DeltaW + DeltaG) @ gamma*
```

With `m_k=0` (cycle 1, or immediately after any memory clear) `w_hat`
reduces identically to the plain iterate — this is how "AA active from
cycle 1" holds without a special case.

**Safeguard (frozen spec's literal ratchet form, chosen over the
Fu-Zhang-Boyd envelope, which the spec states is "acceptable but not
required"):** `last_accepted_residual` starts at `+inf`. At cycle `k`,
`combined_residual_k = sqrt(Σ_{v,pf,ess} r_channel^2 + s_channel^2)` (the
Boyd primal+dual residual norms `get_admm_boyd_residual_metrics` already
computes every cycle — no extra evaluation of `F`,
`admm_anderson_acceleration.combined_scaled_residual`). If
`combined_residual_k < last_accepted_residual`: ACCEPT `w_hat`, set
`last_accepted_residual = combined_residual_k`. Else: REJECT (keep the
already-in-place plain iterate — no write-back call at all) and CLEAR the
memory. **Recorded design choice:** `last_accepted_residual` is left
UNCHANGED on a rejection — a genuine ratchet, not reset to `+inf` every
time, so AA does not fire again until the plain-ADMM trajectory has made
real progress past the last acceleration's mark. This is stated explicitly
in the module docstring as the interpretation chosen where the frozen
spec's English left it ambiguous.

**Certificate independence (amendment iv).** The caller skips the
extrapolation attempt entirely whenever `boyd_metrics['all_boyd_pass']` is
True (`AndersonAccelerationState.step`'s `boyd_all_pass` branch); the
`(w_k, g_k)` pair is still pushed to history in that case — a recorded
design choice (ordinary, valid ADMM data regardless of whether an
extrapolation is attempted from it; keeps AA's memory "warm" so it resumes
immediately, not from empty, the cycle a channel leaves tolerance again).

**Memory clear on rho change (amendment i).** After `_update_admm_penalties`
runs, `shared_resources_planning.py:2889-2907` compares its returned
`penalties_before`/`penalties_after` dicts PER CHANNEL with `!=` (exact
float comparison — `_scale_admm_penalty`'s `factor == 1.0` path is an
IEEE-754 no-op, so a channel that is merely held/frozen/exempt this cycle
has `before[g] == after[g]` bit-for-bit) and clears the AA memory on any
channel that changed. This is state-based, not string-based, per the
task's instruction.

**Local-solve-failure handling (not explicitly specified by the frozen
spec; a necessary, documented extension).** If any local solve fails this
cycle, `consensus_vars`/`dual_vars` are NOT fully updated for the failed
agent(s) (`_update_interface_power_flow_variables`/
`_update_shared_energy_storage_variables` both gate their writes on
`_solver_result_succeeded`), so `g_k` would mix a stale sub-block with
fresh ones and is not a valid secant pair. `AndersonAccelerationState.
skip_on_failure` clears the memory and records the cycle distinctly
(`'skipped (local solve failure this cycle)'`), leaving
`last_accepted_residual` unchanged. Flagged under Questions for Planner.

## Per-cycle record and peak RSS

Every cycle's AA action/safeguard/reset outcome is added to the SAME
per-cycle dict the existing `admm_diagnostics` list already accumulates
(`shared_resources_planning.py:3120-3137`, fields prefixed `aa_`) — `None`
on every field when the flag is off, so an existing harness reading that
list is unaffected until it opts into the new keys. Peak RSS
(`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`, with
`peak_rss_platform_units` stating the bytes-on-macOS/BSD-vs-kilobytes-on-
Linux convention) is added to the returned `state` dict unconditionally
(`shared_resources_planning.py:3222-3231`, `:3251-3252`) — not gated by
the AA flag, since it is a general resource record for the run, not part
of the algorithm.

## What the harness must add to run arm "s41_aa" (NOT implemented; spec only, per the task's "you may")

Given the size of this task, I chose not to add the harness arm to
`p515_g_g1_g4_admm_gates.py` (explicitly optional — "you may also add...
but DO NOT RUN IT"); the mandatory deliverable is this specification.
Mirror the `S39_ARMS`/`_s39_ids_for_mode`/`run_s39_arm` structure
(`p515_g_g1_g4_admm_gates.py:5728-6229`):

1. **Arm config.** `S41_AA_ARM = {'out_dir': .../P515S41/aa_arm, 'run_id_stub': 'p515s41_aa_arm', 'precheck_id_stub': 'p515s41_aa_precheck'}` — the D-oracle's rho/tau/exemption/freeze values (`S39_RHO_V/PF/ESS`, `S39_TAU=0.0`, D's `balancing_exempt_until` for ESS, `freeze_after_unchanged_cycles`, `freeze_backstop_cycle`, `minimum_consecutive_converged_cycles=10`) exactly as `run_s39_arm('s39_D')` sets them, PLUS one new line: `precheck_admm_params.anderson_acceleration['enabled'] = True` (and, if the frozen spec's memory/regularization ever need overriding from their `ADMMParameters` defaults, set `precheck_admm_params.anderson_acceleration['memory']`/`['regularization']` here too — the defaults already equal the frozen spec's values, 5 and 1e-10, so no override is needed for the spec as written).
2. **Working-dir ids.** `_s39_ids_for_mode`-style helper for `'s41_aa'`, mode-derived (`'real'` vs `'preflight'`), producing THREE ids (precheck/probe/run) per mode, so a preflight and the real launch never collide — copy `_s39_ids_for_mode`/`_s39_working_dir_ids` verbatim with the arm key changed.
3. **Cap.** 300 (Addendum 23 amendment (v): "certification ≤ 80 cycles at C*" is the SUCCESS criterion, not the cap — run to the standard budget cap like every other s39/s41 arm).
4. **Captures.** All s39-standard captures (`s38_pf_capture_hooks`, `s39_exempt_until_capture_hooks`, `write_boyd_terminal_s35ref`) UNCHANGED, PLUS:
   - the per-cycle AA sidecar: every `aa_*` field already added to `admm_diagnostics` (see above) is already present in each row if the harness's row-writer already serializes the full `admm_diagnostics` entry (`full_diagnostics_in_rows=True`, as `run_s39_arm` already passes) — verify this BEFORE the run (rule eleven: assert the capture path exists), rather than discovering afterwards that a hand-picked field list omitted the `aa_*` keys;
   - the run's peak RSS: `state['peak_rss_ru_maxrss']`/`state['peak_rss_platform_units']` from the returned `state` dict (`run_operational_planning`'s return value) — add both to the arm's `report` dict, the same way `s39_ess_exempt_until_state_sidecar_path` etc. are added in `_s39_hook`.
5. **Preflight.** Two-cycle bitwise-identity preflight with the flag OFF first (reproduces D exactly — the frozen spec's gate item (v) first bullet), THEN a short (e.g. 5–10 cycle) preflight with the flag ON to confirm the AA code path executes without exception and produces sane `aa_*` records, before the full 300-cycle launch.
6. **Dispatcher.** `elif gate == 's41_aa': run_s41_aa_arm()` in the `if __name__ == '__main__':` block, one exact command documented in the docstring, per the `CLAUDE.md` "one gate per task" rule.

## Commits (for the Planner to cherry-pick)

In this worktree (`wt_step37_aa`, detached HEAD at `a099c7c5`):

1. `c5d18add` — production code: `admm_anderson_acceleration.py` (new),
   `admm_parameters.py`, `shared_resources_planning.py`.
2. `eacbdfcc` — `p515_s41_aa_checks.py` and its outputs
   (`data/SRP1/Results/P515S41/aa_checks/aa_checks.json`,
   `aa_checks_manifest_sha256.json`).

`WORKER_REPORT_S41_AA.md` (this file) is a separate, third commit.

## Commands / experiments run

All commands used the canonical interpreter
(`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`), run from
inside the worktree only.

- `python -m py_compile shared_resources_planning.py admm_parameters.py admm_anderson_acceleration.py p515_s41_aa_checks.py` — syntax/import check.
- `python -c "import shared_resources_planning as srp; import admm_anderson_acceleration as aa; ..."` — confirmed the flag defaults False and the new helper functions are present.
- `NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=/Users/micaelsimoes/coin-or/dist/bin/clp python -u p515_s41_aa_checks.py` — the zero-solve checks (see Results). `NLP_SOLVER_PATH`/`LP_SOLVER_PATH` are CLAUDE.local.md's canonical values, supplied as environment variables (not a committed `.env`, which does not exist in a detached-HEAD worktree and must never be committed) because `SharedResourcesPlanning`'s constructor validates the solver path is configured even on a path that performs zero solves.
- `git -C <worktree> stash` / `stash pop` (temporary, used ONLY to confirm a crash was pre-existing on unmodified HEAD, not introduced by this task's diff — see Unexpected Findings; nothing was left stashed).

No `git commit --amend`, no destructive git operations, no changes to the main checkout at any point.

## Results

`data/SRP1/Results/P515S41/aa_checks/aa_checks.json`:
`"all_checks_passed": true` — `check_A_flag_off`, `check_B_linear_map`,
`check_C_state_machine`, `check_D_round_trip` all `"passed": true`.
`solve_profile_guard.counts` = `{"permitted_solve": 0, "permitted_exec": 0,
"blocked_solve": 0, "blocked_exec": 0}`, `verify_failures: []` — zero
solves, confirmed by an armed (not asserted) guard.

Check D's dual (u) round-trip, reported honestly as a TOLERANCE check (not
bitwise — see the module/check docstrings for why the double-rounding
through two non-power-of-two factors, rho and scale, makes bitwise
unattainable in general): `max_abs_diff = 8.88e-16`, `max_rel_diff =
2.72e-16` (machine epsilon), well inside the `1e-9` bar used.

## Validation

- **Code executes correctly:** yes — `py_compile` clean; the module
  imports and the flag/helpers behave as expected in isolation.
- **Zero-solve checks pass:** yes, all four, with an armed
  `SolveProfileGuard(permitted=())` verifying `expected_solves=0` for the
  whole check script (not merely asserted).
- **Underlying claim ("AA off ⇒ byte-identical") — evidence level:** the
  flag-off no-new-path claim is established by (a) the actual gating
  predicate (`anderson_acceleration_enabled`) reading `False` from a
  fresh `ADMMParameters`, and (b) a static, `ast`-based source check that
  every call into `admm_anderson_acceleration` (and the two orchestration
  helpers) from `_run_operational_planning` sits inside an `if
  aa_enabled:` guard, except the one call that defines `aa_enabled`
  itself. This is NOT a full dynamic run of `_run_operational_planning`
  with the flag off (that would need real or heavily stubbed DSO/TSO/ESSO
  solves and was judged disproportionate for this specific claim — see
  Remaining issues). The static check is precise (it walks the actual
  function's AST, not a hand-copied excerpt) but is a source-level
  guarantee, not a runtime one.
- **The 2-cycle bitwise-identity gate against the oracle (frozen spec gate
  item (v), "flag off") and the ≤80-cycle/cost-within-1.5e-4 gates (flag
  on) are NOT evaluated by this task** — they require real solves and are
  explicitly out of scope ("CODE ONLY — no solves, no runs").

## Unexpected findings

- **Pre-existing worktree/environment issue, unrelated to this task's
  diff:** `p56a_oracle.load_baseline()` (and therefore
  `p515_s32_zero_solve_checks._build_admm_ready_state`, needed for Check
  D) exits with `SystemExit(-3)` and no traceback unless `NLP_SOLVER_PATH`
  (and `LP_SOLVER_PATH`) are set in the environment — the detached-HEAD
  worktree has no `.env` (by design: `.env` must never be committed).
  Confirmed via `git stash` that this reproduces on the UNMODIFIED
  worktree HEAD (`a099c7c5`) with none of this task's changes present, so
  it is an environment-provisioning gap in how the worktree was set up,
  not a defect introduced here. Worked around by supplying
  `NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=/Users/micaelsimoes/coin-or/dist/bin/clp`
  (CLAUDE.local.md's canonical values) as environment variables for the
  one invocation of `p515_s41_aa_checks.py`; nothing was written to the
  main checkout to fix this. Flagged so the Planner is aware future
  worktree tasks needing `p56a_oracle`/`SharedResourcesPlanning` will hit
  the same thing.
- **A math correction made mid-task, kept as evidence, not hidden:** the
  first version of Check D claimed the dual (u) round-trip would ALSO be
  bitwise exact under a power-of-two multiplier, by the same reasoning as
  the z (single-scale) case. Running it found `dual_bitwise_match: false`.
  The reasoning error (two chained non-power-of-two factors — rho and
  scale — do not compose exactly even when the third factor is a power of
  two) is now documented in both `admm_anderson_acceleration.write_back_w`'s
  docstring and `check_d_round_trip`'s docstring, and the check reports
  the dual round-trip honestly as a tight-tolerance (not bitwise) result.

## Remaining issues

- The harness arm "s41_aa" is specified (above) but not implemented —
  scoped out per the task's "you may also add... but DO NOT RUN IT"
  (optional) given this task's overall size.
- The flag-off "no new code path" claim rests on a static AST check, not
  a full dynamic run of `_run_operational_planning` with stubbed solves —
  see Validation.
- `write_back_w` does not update TSO proximal centres; correct only under
  `tau == 0` (verified for the current oracle configuration). Not
  reconsidered for a hypothetical `tau != 0` run, since none is currently
  authorized.
- The local-solve-failure handling (`skip_on_failure`) is a documented
  extension beyond the frozen spec's literal text (which does not mention
  solver failures); flagged below for Planner confirmation.

## Questions for Planner

1. Is the local-solve-failure handling (`AndersonAccelerationState.
   skip_on_failure` — clear memory, leave `last_accepted_residual`
   unchanged, distinct record label) the intended behaviour, or should a
   failed cycle be treated identically to a rejected safeguard step (same
   clear, but also update the baseline to something), or some other rule?
2. Confirm the ratchet-safeguard interpretation (`last_accepted_residual`
   left UNCHANGED on rejection, never reset to `+inf`) is the intended
   reading of "an AA step is taken only while the combined scaled residual
   at the current cycle is below the residual recorded before the last
   accepted AA step" — the frozen spec's English supports this reading but
   does not rule out resetting the bound on every rejection instead.
3. Should the harness arm (spec above) be implemented as a follow-up task,
   and should it launch as-is, or does the Planner want the preflight
   (flag-off 2-cycle bitwise identity, flag-on short smoke test) run
   first as its own bounded task before the full 300-cycle launch?
