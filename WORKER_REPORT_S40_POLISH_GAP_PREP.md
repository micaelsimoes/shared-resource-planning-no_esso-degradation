# Worker Report — P5.15 Addendum 22 item (4): Step 3.5 polish-gap gate — build, checks, smoke test

## Task received

Build `p515_s40_polish_gap.py`, the Step 3.5 polish-gap gate on D's certified point (48
network solves, no ADMM; gate: recourse change < 0.1 %). Determine precisely which of
"P5.7's test" vs `p55d_d1_polish.py` "fixed consensus" means; build zero-solve checks
(`p515_s40_polish_gap_checks.py`); commit the scripts; run ONLY the hidden
`--smoke-cycles 2` smoke test; commit its evidence + this report. Do **not** run the
full 300-cycle run — the Planner launches that.

Authority: `PLANNER_BRIEF_2026-09-13.md` Step 3.5 (~line 516) and Addendum 22 item
`4_step_3_5`; frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Step 3.5, Addendum 22, frozen-spec-v11 authority block)
- `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`
- `p57_eval.py`, `p57_hypotheses.py` ("P5.7's test")
- `p56a_oracle.py` (`common_coordinated_values`, `apply_common_values`,
  `restore_base_objective`, `polish_networks`, `per_block_base_objectives`,
  `_tagged_holders`, `audit_networks`) — P5.7's own primitives
- `p55d_d1_polish.py` (P5.5-D1's `interface_values`/`apply_common_values`)
- `p515_g_g1_g4_admm_gates.py` (`run_admm_arm`, `_construct_arm_planning`,
  `assert_g_capture_paths`, `write_boyd_terminal_s35ref`, `s38_pf_capture_hooks`,
  `s39_exempt_until_capture_hooks`, `_acquire_exclusive_run_lock`, `S = p58_rescale`
  import as `R`)
- `p515_s40_case_file_repro.py`, `p515_s40_clone_capture_preflight.py` (comparator
  conventions, precondition-check conventions, reused by import)
- `p515_s32_zero_solve_checks.py` (`_build_admm_ready_state`, the zero-solve
  ADMM-ready-state production construction sequence)
- `shared_resources_planning.py` (`_get_operational_recourse_components`,
  `get_admm_boyd_residual_metrics`, `update_transmission_model_to_admm`,
  `create_admm_variables`, `_get_admm_block_weight`, the main ADMM cycle loop)
- `model_construction_helpers.py` (`objective_function_rule`,
  `configure_shared_ess_operational_state`, the `expected_interface_*`/
  `expected_shared_ess_*` def-constraint rules)
- `network.py` (`run_smopf`, `_run_smopf`, `_run_smopf_solver_attempt`,
  `_is_recoverable_network_failure`)
- `network_data.py` (`get_primal_value`, confirming the per-block weight formula
  matches `_get_admm_block_weight` exactly)
- `p58_rescale.py` (`patched_admm_objectives`, `rescale_block`, `RESCALED_OBJECTIVE`)
- `p513_solve_profile_guard.py`
- `data/SRP1/SRP1_params.json` (confirmed `admm.num_max_iters=300`,
  `minimum_consecutive_converged_cycles=10`, matching D's oracle config)
- `data/SRP1/Results/P515S39_D_run/g_s39_D.json` (D's committed full 139-cycle
  reference: `cycles_run=139`, `gross_operational_cost=650966975.2943751`,
  `converged_at_cycle=125`)

## Files modified / created

- `p515_s40_polish_gap.py` (new)
- `p515_s40_polish_gap_checks.py` (new)
- `data/SRP1/Results/P515S40/polish_gap_checks/polish_gap_checks.json` (new, evidence)
- `data/SRP1/Results/P515S40/polish_gap_smoke/` (new, evidence — smoke-test run)
- `data/SRP1/Results/P515S40/polish_gap_smoke_launch.log`,
  `polish_gap_smoke_launch_v1_failed.log` (new, evidence)
- `WORKER_REPORT_S40_POLISH_GAP_PREP.md` (this file)

No production files touched. `p56a_oracle.py`, `p55d_d1_polish.py`, `p57_eval.py`,
`p57_hypotheses.py` not edited.

## Reading: "fixed consensus" (citations)

Step 3.5's own text: *"re-solve every network block with the unscaled base objective at
fixed consensus (P5.7's test)"*. "P5.7's test" (`p57_eval.py`/`p57_hypotheses.py`) runs
its polish step by **calling `p56a_oracle.py`'s primitives verbatim**:
`common_coordinated_values`, `apply_common_values`, `restore_base_objective`,
`polish_networks`. Reading those four functions (`p56a_oracle.py:233-528`):

- **Interface voltage / power flow**: no explicit ADMM "z" for these two channels, in
  either the P5.6-era model or the current P5.15 Boyd-residual mapping —
  `get_admm_boyd_residual_metrics`'s own docstring (Advisor findings F1-F4, Planner-
  accepted) calls the TSO's own copy "z" and the DSO's own copy "x": an asymmetric
  Gauss-Seidel target, not a genuine two-sided average. `common_coordinated_values`
  instead takes the **midpoint** of the two sides' own achieved values, read directly
  off the solved Pyomo models — "the choice that minimises the largest correction asked
  of either side" (its own docstring). This is line-for-line **the same design** as
  `p55d_d1_polish.py`'s (P5.5-D1's) `interface_values`/`apply_common_values` — P5.7's
  helpers are a direct descendant of that stage, not an independent one. **The two
  "designs" the task asks to distinguish are therefore the same mechanism**; P5.7
  (`p56a_oracle.py`) is named as the authority because it is the one Step 3.5 cites and
  because it is **already a dependency of `p515_g_g1_g4_admm_gates.py`** (`import
  p56a_oracle as O`), confirming it is structure-compatible with the current
  planning/model objects, not a stale interface.
- **Shared-ESS dispatch**: `consensus_vars['ess']['z']['current']` **is** an explicit,
  genuine three-way consensus variable in the current ADMM (`create_admm_variables`,
  `shared_resources_planning.py` ~4041-4046) and is used directly, unchanged, by
  `common_coordinated_values`.

Fixing at this common value (`apply_common_values`) pins the model's own
`expected_interface_vmag`/`expected_interface_pf_p/q`/`expected_shared_ess_p/q` Vars
(the actual Boyd-residual "x", tied by hard equality constraints —
`tn_interface_expected_*_rule`/`dn_interface_expected_*_rule`,
`shared_resources_planning.py` ~3615-3693 — to the physical variables
`apply_common_values` fixes directly) at a single number on both sides: the primal
residual is set to exactly zero **by construction**, not merely driven below the Boyd
tolerance. The base objective is then reactivated (no consensus penalty at all, since
consensus is now a hard constraint), and every block is re-solved with
`network.run_smopf`, production's own SMOPF entry point.

Available shared-ESS S/E capacity is **not** separately reconciled (unlike
`p55d_d1_polish.py`'s explicit capacity-shift step): Step 3.5's text names only
"network block[s] ... at fixed consensus" (the three augmented-Lagrangian channels),
and the current ADMM republishes the same `sess_available_capacities` into every block
every cycle (unlike the P5.5-era design `p55d` was correcting). Noted for the Planner
as an intentional scope decision, not an oversight.

`common_coordinated_values`/`apply_common_values`/`per_block_base_objectives`/
`_tagged_holders` are imported **by import** from `p56a_oracle.py` and called verbatim.

## Finding: `restore_base_objective`/`polish_networks` are stale for the current harness

`p56a_oracle.restore_base_objective` only knows about two Objective components
(`objective`, `admm_objective`) — correct for P5.6/P5.7-era code. **Step 3.4** (frozen
spec v4) made objective rescaling **production** behaviour: `run_admm_arm` wraps every
ADMM cycle unconditionally in `p58_rescale.patched_admm_objectives()`
(`p515_g_g1_g4_admm_gates.py`'s own `R = p58_rescale` import), which deactivates
`admm_objective` and activates a **third** component, `p58_rescale.RESCALED_OBJECTIVE`
(`'p58_rescaled_admm_objective'` = `effective_scale * admm_objective`) — the objective
IPOPT actually solves on every real cycle.

Calling only `p56a_oracle.restore_base_objective` after a real `run_admm_arm` run
therefore leaves **both** `objective` and `p58_rescaled_admm_objective` active
simultaneously. The first smoke-test attempt (using `O.polish_networks` verbatim, as
originally built) failed on **all 48/48 blocks**; direct single-block reproduction
(`tee=True`) showed IPOPT/AMPL raising:

```
There is more than one objective function in the AMPL model, but
AmplTNLP::set_active_objective has not been called.
Exception of type: INVALID_TNLP ...
EXIT: Some uncaught Ipopt exception encountered.
```

`termination_condition='other'` is **not** one of the three "recoverable" conditions
(`internalSolverError`, `maxIterations`, `infeasible`) `network.py:
_is_recoverable_network_failure` checks, so no retry ever fired — consistent with
`solve_profile.permitted_solve == 48` (no retries) on that first attempt.

**Fix (in `p515_s40_polish_gap.py` only — no edit to `p56a_oracle.py` or any production
file):** `_switch_to_base_objective(model)` generalizes the switch — deactivates
**every** Objective component except `objective`, activates `objective`, and asserts
exactly one Objective is active afterward (fails loudly if the generalization is ever
itself incomplete). `_polish_networks_fixed_consensus` is `p56a_oracle.polish_networks`
with that one substitution, otherwise identical (same `_tagged_holders` loop, same
`network.run_smopf` call).

Re-run after the fix: **35/48 blocks now solve successfully**; the remaining 13
failures (all 12 TSO blocks + 1 DSO) are genuine IPOPT terminations —
`EXIT: Converged to a point of local infeasibility. Problem may be infeasible.` — a
legitimate outcome of fixing both sides to a midpoint at a **2-cycle, far-from-Boyd-
tolerance** point (V max primal residual 0.050 vs `eps_pri` 0.0034; PF max 0.194 vs
`eps_pri` 0.0013 at that cycle), not a code defect. This is exactly what the task
predicted: *"The smoke gate value is not meaningful (2 cycles is not a converged
point)."*

## Commands / experiments run

1. Zero-solve checks (guard armed, `permitted=()`):
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s40_polish_gap_checks.py
   ```
2. Diagnostic single-block probe (outside `redirect_stdout`, `tee=True`) to identify the
   first smoke-test attempt's root cause — ad hoc, not committed as a script (evidence
   is the IPOPT log excerpt above and this report).
3. Smoke test (the one the task authorizes the Worker to run), attached, both streams
   captured:
   ```
   /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s40_polish_gap.py \
       --smoke-cycles 2 > data/SRP1/Results/P515S40/polish_gap_smoke_launch.log 2>&1
   ```
   Preconditions checked immediately beforehand (matching
   `p515_s39_preflight.py`/`p515_s40_case_file_repro.py`): no `.p515_g_gate.lock`; `ps
   aux` clear of `p515_g_g1_g4_admm_gates.py`/`p515_s39_`/`p515_s40_`/
   `p515_s36_step36_timing_run.py` (excluding this process's own ancestor chain);
   `data/SRP1/Results/P515S40/polish_gap_smoke/` did not exist; `git status --porcelain`
   clean on `shared_resources_planning.py`, `network.py`, `network_data.py`,
   `shared_energy_storage_data.py`, `admm_parameters.py`, `p515_g_g1_g4_admm_gates.py`,
   `data/SRP1/SRP1_params.json`; D's committed reference report present.

## Results

### Zero-solve checks (`polish_gap_checks.json`)

Guard: `permitted_solve=0, permitted_exec=0, blocked_solve=0, blocked_exec=0`,
`verify_failures=[]`. `all_checks_pass=True`.

- **(i) objective switching** — `_switch_to_base_objective` tested on 4 states (TSO/DSO
  × two-objective and three-objective-after-`p58_rescale.rescale_block`): every Var's
  value/`.fixed` flag identical before/after; exactly `['objective']` active afterward
  in every case. Confirms the fix against the **real** post-`run_admm_arm` situation
  (three active objectives at some point in the lifecycle), not just the stale
  two-objective assumption.
- **(ii) fixing step** — `n_expected_fixed_targets=6912` (=`864 × 8`: 3 nodes × 12
  year/day × 24 periods × {vmag_sqr(tso), vmag_sqr(dso), pg, qg, t_sess_p, t_sess_q,
  d_sess_p, d_sess_q}); `n_expected_covered_and_correct=6912`; `n_unexpected_touched=0`;
  `n_value_mismatches=0`; `n_missing_expected=0`; `p56a_interface_rows` count
  `1728 == 2 × 864` expected, max residual `0.0`.
  Design note (documented in the script): shared-ESS operational Vars were **already**
  `fixed=True` at the zero-investment build-default probe state
  (`configure_shared_ess_operational_state` fixes them to 0 when capacity is
  "inactive"), so a naive False→True transition test would silently miss
  `apply_common_values` correctly re-fixing them to a *new* value. The check instead
  verifies, per expected key, that `.fixed` is True **and** the value matches the
  target — and, for every other key, that neither `.fixed` nor the value changed at
  all.
- **(iii) recourse aggregation** — after perturbing 15 dispatch Vars directly
  (`.set_value`, never `.fix`/solve): `sum(per_block_base_objectives) ==
  gross_operational_cost_including_settlement` exactly (`2609858.251721687` both sides,
  `absolute_difference=0.0`), confirming `per_block_base_objectives` sums the
  settlement-**included** convention (matches `_get_operational_recourse_components`'s
  own comment), distinct from the settlement-**excluded** `gross_operational_cost`
  (`1631161.4073260545` at that same perturbed state) the certified cost/gate use.

### Smoke test (`polish_gap_smoke/polish_gap_results.json`)

- Reproduction: **PASS** — `reproduces=True`, `n_diffs=0`, mode = truncated
  (`cycle_trajectory[:2]` + rows-derived top-level fields vs D's committed first 2
  rows). `admm_cycles_run=2`, `admm_gross_operational_cost=740742829.5748339`.
- Polish dispatched: **48/48 blocks** (`n_blocks=48`).
- Solve accounting: `permitted_solve=74 = 48 + 26 retries` (`retries_beyond_one_per_block:
  26`), `permitted_exec=74` (matches — no retry left mid-launch), `blocked_solve=0`,
  `blocked_calls=0`. **All accounted for**, exactly as the task specifies.
- `all_solved=False`; **13 failed blocks** (all 12 TSO year/day blocks + `DSO5|2025|
  Spring`), each `EXIT: Converged to a point of local infeasibility` — a legitimate
  IPOPT outcome at this deeply-unconverged point (see Finding above), not a mechanism
  defect.
- Gate: **not evaluated** (`gate=None`), per build item 3 ("the gate is not evaluated
  if any block fails") — correct behaviour, not a failure of the harness.
- Per-block records for the 35 blocks that **did** solve show nonzero, plausible
  `weighted_base_objective` deltas; the 13 failed blocks show `delta=0.0`
  (before==after — `network.py:_run_smopf` only calls `model.solutions.load_from(result)`
  on success, so a failed solve leaves every Var exactly where it was fixed, correctly
  reflected).

## Validation

- Code executes correctly: yes (both scripts compile, import, and run to completion
  without an unhandled exception; smoke test exits 1, as intended, because the gate was
  correctly withheld — this is documented exit-code behaviour, not a crash).
- Zero-solve claim: **enforced**, not asserted (`SolveProfileGuard(permitted=())`,
  `verify_failures=[]`).
- Reproduction (τ=0 determinism) at 2 cycles: **verified**, bitwise, against D's own
  committed first-2-cycle rows.
- Polish mechanics (dispatch count, solve accounting, per-block reporting, gate
  withholding on failure): **verified** by direct inspection of `polish_gap_results.json`
  and IPOPT logs, not merely inferred from a clean exit.
- The 0.1 % recourse-change **gate value itself** was not, and could not be, validated
  by the smoke test (2 cycles is not a converged point; the task states this explicitly).
  That requires the full 300-cycle run, which the Planner runs.

## Unexpected findings

1. **The objective-switching bug** described above — `p56a_oracle.restore_base_objective`
   is incomplete for any model that has been through `run_admm_arm`'s
   unconditionally-applied `p58_rescale.patched_admm_objectives()` wrapper (i.e. every
   real arm run since Step 3.4). Fixed locally in `p515_s40_polish_gap.py`
   (`_switch_to_base_objective`); `p56a_oracle.py` itself is untouched, per scope.
   Worth flagging to the Planner: any **other** future stage that reuses
   `p56a_oracle.restore_base_objective`/`polish_networks` against a real
   `run_admm_arm`-produced model set (rather than P5.6-era `p56b_policy.run_operational`
   output) will hit the same bug.
2. Shared-ESS operational Vars are fixed at construction time when the candidate's
   installed capacity is zero (`configure_shared_ess_operational_state`,
   `shared_ess_capacity_is_inactive`) — a `_build_admm_ready_state` zero-solve probe
   (zero-investment candidate) therefore starts with `shared_es_pnet`/`shared_es_qnet`
   already `fixed=True`, unlike the real D candidate (positive investment), where they
   are genuinely free going into the ADMM cycle. Purely a testing-environment note
   (documented in `p515_s40_polish_gap_checks.py`), not a production concern.

## Remaining issues

- The 0.1 % gate is untested at a genuinely converged point — that is the full run's
  job, not this task's.
- 13/48 blocks failing to locally converge at the 2-cycle smoke point is expected and
  not something to "fix"; it is reported, not rationalized away, per the task's own
  framing.

## Questions for Planner

None — the reading of "fixed consensus" is directly supported by the cited code, and
the smoke test's outcome matches the task's own stated expectation once the
objective-switching bug (found and fixed within this script's own scope) was resolved.

## Full-run launch command (Planner launches; Worker did NOT run this)

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s40_polish_gap.py \
    > data/SRP1/Results/P515S40/polish_gap_launch.log 2>&1
```

Preconditions (checked automatically by the script before writing anything, matching
`p515_s39_preflight.py`/`p515_s40_case_file_repro.py`): `.p515_g_gate.lock` absent; no
`p515_g_g1_g4_admm_gates.py`/`p515_s39_`/`p515_s40_`/`p515_s36_step36_timing_run.py`
process alive (excluding this process's own ancestor chain); output root
`data/SRP1/Results/P515S40/polish_gap/` does not yet exist (write-once); production
files (`shared_resources_planning.py`, `network.py`, `network_data.py`,
`shared_energy_storage_data.py`, `admm_parameters.py`, `p515_g_g1_g4_admm_gates.py`,
`data/SRP1/SRP1_params.json`) clean in git; D's committed reference report
(`data/SRP1/Results/P515S39_D_run/g_s39_D.json`) present. Run attached, alone (no
`screen`/`nohup`/backgrounding), both streams captured via the shell redirection above.
Expected runtime: on the order of D's own certification run (~85 minutes, per D's
committed timestamps) plus 48 polish solves.
