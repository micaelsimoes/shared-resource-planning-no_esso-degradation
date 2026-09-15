# Worker Report -- P5.15 Step 3.2 E4: local-solver noise floor

## Task received

Bounded diagnostic task E4 (P5.15 Step 3.2, Addendum 14 of `PLANNER_BRIEF_2026-09-13.md`):
measure the local-solver noise floor at fixed inputs, using exactly 8 IPOPT solves across
two preserved pre-solve fixtures (one TSO block, one DSO block) x four variants (W1 warm,
W2 warm repeat, C1 cold, T1 warm with tol/acceptable_tol /10), and derive `eps_abs` from
the measured floor by the rule frozen in
`data/SRP1/Results/P515S33/frozen_e4_noise_floor_spec_v1_35f54dfe.json` (commit
`254048f4`). No production-code changes.

## Files inspected

- `data/SRP1/Results/P515S33/frozen_e4_noise_floor_spec_v1_35f54dfe.json` (binding spec)
- `p512_y_tso_replay.py`, `p512_x_comparator_replay.py` (precedent to follow)
- `p513_solve_profile_guard.py` (guard mechanism)
- `p515_s33_e1_zero_solve.py` (concurrent E1 task; read-only, not modified, not reused --
  answers different questions, zero-solve)
- `network.py` (`_create_smopf_solver`, `_run_smopf`, solution-load path
  `model.solutions.load_from(result)` gated on `solver_result_succeeded`)
- `helper_functions.py` (`solver_result_succeeded`, `solver_result_summary`)
- `shared_resources_planning.py` (`get_admm_boyd_residual_metrics` -- channel mapping and
  normalization definitions for V/PF/ESS; `_shared_ess_admm_normalization_mva`;
  `expected_interface_vmag`/`_pf_p`/`_pf_q`/`shared_ess_p`/`_q` variable definitions and
  their production display-side scaling by `s_base`/`v_base`)
- `p56a_oracle.py` (`fresh_planning`, `_holders`)
- `p58_rescale.py` (`read_log_since` -- reused for IPOPT iteration-count/exit parsing)
- `admm_parameters.py`, `planning_parameters.py` (`shared_ess_normalization_floor_mva`
  default 0.10, current `boyd.eps_abs` default 1e-5)
- `solver_parameters.py`, `.env` (`NLP_SOLVER_PATH=/usr/local/bin/ipopt`, loaded via
  `dotenv` on import -- confirms production resolves the canonical machine IPOPT, not the
  conda one, with no harness override needed)
- `data/SRP1/case9/case9_params.json`, `data/SRP1/case33_2/case33_2_params.json` (current
  case-file solver options: `tol=1e-5`, `acceptable_tol=1e-4` on both, matching the spec's
  `reference_numbers_from_s32`)
- The two fixture pickles themselves (`metadata`, active-objective inventory,
  `shared_es_s_rated_fixed` parameter, `periods`/`active_distribution_networks`/
  `shared_energy_storages` index structure, `ipopt_zL_in/out`/`ipopt_zU_in/out` suffix
  contents) via direct Python introspection, no solve.
- `pyomo/opt/plugins/sol.py`, `pyomo/solvers/plugins/solvers/IPOPT.py` (installed conda
  Pyomo 3.14.18 AMPL solve_result_num mapping -- confirmed IPOPT's "Solved To Acceptable
  Level" and "Optimal Solution Found" both map to `TerminationCondition.optimal` with
  `status=ok`, i.e. `solver_result_succeeded` already implements the spec's "optimal or
  acceptable" test; there is no separate pyomo-level "acceptable" termination condition to
  check against).

## Files modified

None (no existing file was edited).

## Files created

- `p515_s33_e4_noise_floor.py` -- the harness.
- `data/SRP1/Results/P515S33/E4/e4_noise_floor_results.json` -- fixture verification, guard
  record, per-solve table, per-pair per-channel statistics, validity checks, derivation.
- `data/SRP1/Results/P515S33/E4/e4_channel_detail.json` -- full per-entry normalized
  channel arrays and per-node/period detail for all 8 solves (evidence base for the
  statistics above).
- `data/SRP1/Results/P515S33/E4/e4_manifest_sha256.json` -- sha256 + byte count of every
  file written under `E4/` (results JSON, channel-detail JSON, all 8 IPOPT logs).
- `data/SRP1/Results/P515S33/E4/logs/*.log` (8 files, ~88 KB-153 KB each, ~970 KB total --
  committed directly, not hash-only, since small) -- raw IPOPT logs, one per solve,
  distinguished by `log_suffix=<variant>` passed to the real `_create_smopf_solver`.
- `WORKER_REPORT_S33_E4.md` (this file).

## Changes made

Wrote one new harness script, `p515_s33_e4_noise_floor.py`. No production, case-file, or
spec file was edited.

Design decisions, each tied to a spec clause or a concrete check performed first:

1. **Fixture identity.** `data/SRP1/Results/P515S32_run/results/FrozenSMOPF/...` (the exact
   path the spec names) was hashed with `shasum -a 256` *before* writing any solve logic;
   both sha256 values matched the spec exactly (see Validation). The other, differently
   sized copies of files with the same basenames elsewhere in `data/SRP1/Results/` (e.g.
   `data/SRP1/Results/FrozenSMOPF/...`, 4,097,647 and 1,581,844 bytes) were NOT used --
   they do not match the spec's `bytes`/`sha256` fields and are a different capture.
2. **Objective read from the model's own ACTIVE Pyomo `Objective`,** not
   `network.compute_objective_function_value` (which hard-codes `model.objective`). Both
   fixtures have exactly three `Objective` components (`objective`, `admm_objective`,
   `p58_rescaled_admm_objective`) with only `p58_rescaled_admm_objective` active -- verified
   by direct introspection before writing the extraction code. Using the inactive
   `model.objective` would have reported a quantity IPOPT never optimized.
3. **Shared-ESS rating `S` read from the model's own `shared_es_s_rated_fixed[e]`
   parameter** (in the frozen pre-solve model), not from
   `network.shared_energy_storages[e].s` on a network object rebuilt by
   `p56a_oracle.fresh_planning`. Checked first: `fresh_planning` returns the *baseline*,
   pre-investment network (`shared_energy_storages[*].s == 0.0` for all three TSO-side
   nodes), whereas `shared_es_s_rated_fixed[e]` on both frozen models reads `0.0096875` pu
   (0.96875 MVA), the actual candidate-derived ESS rating in effect when this fixture was
   captured. `get_admm_boyd_residual_metrics` reads `S` from the *live* mutated network
   object during a real ADMM run; that live object is not recoverable from a fresh replay,
   so the model's own embedded parameter -- which mirrors it at model-build time
   (`shared_resources_planning.py:3322` etc. set `shared_es_s_rated_fixed` directly from
   the same estimated/candidate capacity) -- was used instead, per the "preserved pre-solve
   model is authoritative for all numerical state" principle stated in the precedent
   scripts' own docstrings.
4. **DSO interface rating confirmed day-invariant** before using nodes 5/7/9's DSO network
   objects from `fresh_planning` (structural, candidate-independent) for the TSO block's PF
   normalization: `get_interface_branch_rating()` returned identically 200.0 / 100.0 / 150.0
   MVA for nodes 5/7/9 across all four days for year 2025, confirmed by direct query before
   writing the extraction code.
5. **T1's tol/acceptable_tol base values are read from the solver actually built for that
   exact solve** (`solver.options['tol']`/`['acceptable_tol']` immediately after
   `_create_smopf_solver`, before the /10 override), not hard-coded from the case files --
   this matches the spec's "divide tol and acceptable_tol by 10 relative to the options
   production actually set" literally, and is recorded per solve
   (`t1_tolerance_override` in each record).
6. **One process, 8 solves,** each preceded by a fresh `pickle.load` from disk and a fresh
   `network._create_smopf_solver` call (fresh `SolverFactory`/`.options`). The script's own
   docstring records the one-process justification: the actual NLP solve is external-process
   work regardless (`SystemCallSolver._execute_command` launches a new `ipopt` binary per
   `solver.solve()` call), confirmed by the guard's `permitted_exec` count equalling
   `permitted_solve` (8 == 8, see Results) -- no in-process solver state could have leaked
   between solves. No fallback to separate processes was needed.
7. **Guard:** `SolveProfileGuard` with `PERMITTED = [('p515_s33_e4_noise_floor.py',
   '_do_solve')]`, installed before the first solve, uninstalled in a `finally`, verified
   with `guard.verify(8)` after uninstall (same pattern as `p515_g5b_physical_variable_capture.py`).
8. **Failures are recorded, not masked.** Rather than raising on the first non-optimal
   solve (which would have destroyed the partial evidence of the remaining solves), the
   harness records `succeeded=False` per solve, adds it to `invalid_reasons`, and continues
   through all 8 solves before deciding whether the derivation rule may run. In this run all
   8 solves succeeded, so this path was not exercised.

## Commands / experiments run

```
shasum -a 256 data/SRP1/Results/P515S32_run/results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl \
              data/SRP1/Results/P515S32_run/results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl
```
(fixture verification, before any harness code was written)

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_s33_e4_noise_floor.py
```
(the one authorized run -- exactly 8 solves)

Two zero-solve dry runs preceded the real run and consumed none of the 8-solve budget:
(a) a syntax/import check (`py_compile` + `importlib.import_module`), and (b) a direct call
of `extract_tso_channels`/`extract_dso_channels` on the two UNSOLVED pre-solve models (valid
since `pe.value()` on an unsolved `Var` just reads its `initialize` value; no `OptSolver.solve`
or `_execute_command` call occurred in that check).

## Results

### Fixture verification

| fixture | expected bytes | actual bytes | expected sha256 | actual sha256 | match |
|---|---|---|---|---|---|
| DSO node7 case33_2 2025 Autumn cycle7 | 4,090,848 | 4,090,848 | `85afdd2d5153dff943d568f97f16183999f2c114089fef0bb25e86f8731c9c25` | identical | YES |
| TSO case9 2025 Summer cycle7 | 1,588,345 | 1,588,345 | `21695017d48d5997218fb3cda877f064e3f788d850050c8aee83b9c51672639c` | identical | YES |

(Full 64-hex-char values recorded in `e4_noise_floor_results.json['fixture_verification']`;
both matched exactly, so the run proceeded per spec's "Stop on mismatch" clause never
firing.)

### Per-solve table

All 8 solves terminated `optimal` / `status=ok` (`solver_result_succeeded` -> True for all
8; equivalently `solve_result_num` in [0,99] per the installed Pyomo ASL sol-reader, i.e.
production's own optimal-or-acceptable test, satisfied by all 8).

| fixture | variant | from_warm_start | termination | iterations | objective (`p58_rescaled_admm_objective`) | wall (s) |
|---|---|---|---|---|---|---|
| DSO node7 | W1 | True | optimal | 46 | 58751.882589337234 | 0.47 |
| DSO node7 | W2 | True | optimal | 46 | 58751.882589337234 | 0.45 |
| DSO node7 | C1 | False | optimal | 40 | 58751.88258934214  | 0.43 |
| DSO node7 | T1 | True | optimal | 47 | 58751.882578604695 | 0.46 |
| TSO case9 | W1 | True | optimal | 29 | -73682.36599525607 | 0.18 |
| TSO case9 | W2 | True | optimal | 29 | -73682.36599525607 | 0.13 |
| TSO case9 | C1 | False | optimal | 26 | -73682.36599525608 | 0.13 |
| TSO case9 | T1 | True | optimal | 29 | -73682.36599525607 | 0.14 |

T1 tol/acceptable_tol (base -> applied, both fixtures identical): `tol` 1e-05 -> 1e-06,
`acceptable_tol` 1e-04 -> 1e-05 (base values read from the solver actually constructed for
that solve, confirming both fixtures share the case-file default `tol=1e-5`,
`acceptable_tol=1e-4`, matching the spec's `reference_numbers_from_s32`).

**Objective convention** (per CLAUDE.md's reporting-conventions rule): the value above is
the model's ACTIVE Pyomo objective, `p58_rescaled_admm_objective` (confirmed the unique
active objective on both fixtures before any solve; `objective` and `admm_objective` are
present but inactive). It is a rescaled ADMM-cycle objective, not
`gross_operational_cost`/`net_operational_recourse` -- not comparable to those conventions
without further work, which this task did not do.

### Per-pair, per-channel statistics (RMS per entry / max abs per entry / entry count)

**DSO node7 (V: n=24, PF: n=48, ESS: n=48):**

| pair | V rms | V max | PF rms | PF max | ESS rms | ESS max | obj abs diff | obj rel diff |
|---|---|---|---|---|---|---|---|---|
| W1 vs W2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| W1 vs C1 | 6.80e-15 | 1.43e-14 | 6.69e-14 | 3.25e-13 | 1.45e-17 | 2.22e-17 | -4.90e-09 | -8.35e-14 |
| W1 vs T1 | 1.04e-14 | 2.19e-14 | 1.25e-08 | 8.05e-08 | 6.60e-08 | 1.77e-07 | 1.07e-05 | 1.83e-10 |

**TSO case9 (V: n=72, PF: n=144, ESS: n=144):**

| pair | V rms | V max | PF rms | PF max | ESS rms | ESS max | obj abs diff | obj rel diff |
|---|---|---|---|---|---|---|---|---|
| W1 vs W2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| W1 vs C1 | 3.93e-17 | 2.22e-16 | 1.20e-17 | 6.94e-17 | 2.24e-18 | 5.37e-18 | 1.46e-11 | 1.97e-16 |
| W1 vs T1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The TSO block reproduced BIT-IDENTICAL iterates for W1 vs T1 (tol/10 changed nothing at
all, not even at float-noise level) and near-machine-epsilon differences for W1 vs C1
(~1e-16 to 1e-17, i.e. floating-point round-off, not solver-path noise). The DSO block shows
a real, small, above-machine-epsilon floor, dominated by the T1 comparison
(tolerance-sensitivity, not warm/cold).

### Validity checks

- Every one of the 8 solves terminated `optimal`/`status=ok`: **PASS** (0 failures).
- W1 vs W2 bitwise-identical on every measured entry, both fixtures, all 3 channels:
  **PASS** (`w1_w2_bitwise_identical` all `true`).
- Fixture hash match: **PASS** (see Fixture verification).
- Guard: `permitted_solve=8`, `permitted_exec=8`, `blocked_solve=0`, `blocked_exec=0`,
  `guard.verify(8) -> []` (no failures): **PASS**.

`derivation.valid = true`; no invalidity condition fired.

### delta_c and eps_abs derivation (exact arithmetic from the frozen rule)

```
delta_c['V']   = max( max(RMS(W1-C1),RMS(W1-T1)) over {DSO,TSO} )
              = max( max(6.8028e-15, 1.0429e-14),  max(3.9252e-17, 0.0) )
              = max( 1.0429e-14, 3.9252e-17 )
              = 1.0429401450e-14

delta_c['PF']  = max( max(6.6919e-14, 1.2525e-08),  max(1.1972e-17, 0.0) )
              = max( 1.2525028628e-08, 1.1972e-17 )
              = 1.2525028628e-08

delta_c['ESS'] = max( max(1.4467e-17, 6.6037e-08),  max(2.2350e-18, 0.0) )
              = max( 6.6037060098e-08, 2.2350e-18 )
              = 6.6037060098e-08

max_c delta_c  = max(1.0429e-14, 1.2525e-08, 6.6037e-08) = 6.6037060098e-08  (ESS channel)

3 * max_c delta_c = 1.98111180e-07

pre_round = max(1e-05, 1.98111180e-07) = 1e-05          (the floor already dominates)

eps_abs = round_up_1_or_3(1e-05) = 1e-05                (already of the form 1eK, K=-5)
```

**Derived `eps_abs = 1e-5`.** This equals the current production default
(`admm_parameters.py`, `ADMMParameters.tol['boyd']['eps_abs'] = 1e-5`) -- the measured
noise floor does NOT require loosening `eps_abs`; it stays at 1e-5 per the spec's own
predicted no-loosening branch ("eps_abs stays 1e-5 unless the measured floor requires
more").

### Comparison with s32 per-entry reference steps

| channel | delta_c (measured floor) | s32 reference step | ratio (delta_c / reference) |
|---|---|---|---|
| V | 1.0429e-14 | 9.2e-05 | 1.13e-10 |
| PF | 1.2525e-08 | 1.7e-04 | 7.37e-05 |
| ESS | 6.6037e-08 | 5.5e-05 | 1.20e-03 |

The measured local-solver noise floor is 4 to 10 orders of magnitude SMALLER than the s32
per-entry cycle-to-cycle step size on every channel. Per CLAUDE.md's "report a difference
with its resolution" rule, this licenses the conclusion that s32-scale per-cycle steps (all
>= 5.5e-05) are NOT explained by local-solver noise -- they are real ADMM trajectory
movement, not solver-resolution artifacts -- for the specific cycle-7 fixed-point instances
tested here. It does not by itself establish anything about steps measured near
convergence (small steps late in a run), which this task did not test.

### Guard counts

`permitted_solve=8`, `permitted_exec=8`, `blocked_solve=0`, `blocked_exec=0`,
`guard.verify(8)` returned `[]` (no failures; the required exact-count check passed both
for `OptSolver.solve` invocations and for `SystemCallSolver._execute_command`, i.e. actual
`ipopt` process launches).

## Validation

- Code executes correctly: yes -- `py_compile` clean, module imports cleanly, ran to
  completion with no exceptions.
- Requested diagnostic works: yes -- fixture hashes verified before any solve; exactly 8
  solves executed and guard-counted; per-solve, per-pair, per-channel statistics computed
  and written; derivation rule applied with exact arithmetic reproduced above; all outputs
  hash-manifested.
- The underlying numerical question (does the local solver alone produce noise large
  enough to require loosening `eps_abs`) is answered NEGATIVELY for these two fixtures at
  cycle 7: no. This is evidence from ONE cycle (cycle 7) of TWO blocks (one TSO, one DSO);
  it is not a claim about every cycle of every block in a full run.
- `git status --short` after the run shows only the new script, the new `E4/` directory,
  and this report as additions; no tracked file was modified, `git diff --stat` against
  tracked files is unchanged by this task (the diffs shown by `git diff --stat` predate
  this task, per the session's initial `gitStatus` snapshot: `.claude/agents/planner.md`,
  `.claude/agents/worker.md`, `PLANNER_BRIEF_2026-09-13.md`).
- `data/SRP1/Results/P515S33/E1/` untouched (verified by directory listing/mtimes
  unchanged from before this run).

## Unexpected findings

- **The TSO block's W1-vs-T1 iterate is bit-identical** (not merely close): dividing
  `tol`/`acceptable_tol` by 10 changed nothing about the converged point at cycle-7 float
  precision. This is consistent with the TSO block converging to a very sharp KKT point
  well inside both tolerance bands, but it does mean the TSO fixture alone contributes
  essentially nothing to `delta_c` on any channel; the derived `eps_abs` is driven entirely
  by the DSO fixture.
- The `shared_es_s_rated_fixed` discovery (finding #3 above) is worth flagging to the
  Planner even though it did not block this task: any future replay harness that reads `S`
  from `network.shared_energy_storages[*].s` via `p56a_oracle.fresh_planning` (rather than
  from the frozen model's own parameter) on one of these `FrozenSMOPF` fixtures will
  silently get `S=0`, which would blow up the ESS normalization (denominator collapses to
  the floor, 0.1 MVA, rather than the true ~0.97 MVA) and produce channel values ~10x too
  large. No such bug is currently present in `get_admm_boyd_residual_metrics` itself
  (it runs during a live ADMM cycle with a correctly mutated network object); the risk is
  specific to REPLAY harnesses built on these frozen fixtures.

## Remaining issues

- None within this task's bound. The spec's own stated `limitation` applies unchanged:
  "cycle-7 states, not terminal; one TSO block and one DSO block; the ESSO subproblem (tol
  1e-10) is not measured" -- this task did not attempt to extend coverage beyond that scope.

## Questions for Planner

None -- the spec fully determined the procedure and no ambiguity required a judgment call
outside it. The three design decisions in "Changes made" items 2-4 above (active-objective
selection, `S` source, DSO-rating day-invariance) were resolved by direct inspection of the
fixtures/production code rather than by asking, since the spec's `measured_quantities` text
underdetermines them only at the level of "which existing production value satisfies this
description," not at the level of new judgment calls; they are reported here for the
Planner's/Advisor's review rather than left implicit.

## Final summary

Derived from 8 solves (2 fixtures x {W1,W2,C1,T1}), all `optimal`, guard-verified exactly
8/8, W1 vs W2 bitwise-identical, fixture hashes verified:

- `delta_c['V']   = 1.0429e-14`  (s32 reference step 9.2e-05; ratio 1.13e-10)
- `delta_c['PF']  = 1.2525e-08`  (s32 reference step 1.7e-04; ratio 7.37e-05)
- `delta_c['ESS'] = 6.6037e-08`  (s32 reference step 5.5e-05; ratio 1.20e-03)
- `max_c delta_c = 6.6037e-08` (ESS channel, driven by the DSO fixture's W1-vs-T1 pair)
- `3 * max_c delta_c = 1.9811e-07`
- **Derived `eps_abs = max(1e-5, 1.9811e-07)` rounded up to the next `1eK`/`3eK` = `1e-5`**
  -- unchanged from the current production default; the measured local-solver noise floor
  does not require loosening it.
