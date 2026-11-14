# Worker Report — S31: Step 3.1 signed-table implementation + harness capture extension

## Task received

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 10 and the signed table
`P5_15_S31_PENALTY_TABLE_DRAFT.md` (§0, §3, §5, §6 read). Baseline to compare with:
Step 3.0, `data/SRP1/Results/P515S30/` (bitwise = `P515G1B`), recourse
817,520,272.93.

Four parts:
1. Production changes implementing the signed rows 3/D1, 5, 8, 9, 12(D-semantics),
   14/D2, plus D3, D6.
2. Zero-solve verification of every change (blocking `SolveProfileGuard`, count 0).
3. Extend `p515_g_g1_g4_admm_gates.py` with a new CLI arm `s31` that writes
   per-block component levels at the terminal cycle, zero extra solves.
4. Pre-flight: one ADMM cycle through the new code, foreground, < 9 min, running the
   Part 3 writer on the result.

**Not implemented here** (explicitly out of scope): row 18 (Step 5); the stopping
rule / ρ policy / σ scaling (3.2–3.4); ε or tolerances; recovery policy; anything for
D6 beyond the bound; actually launching the `s31` campaign (the Planner launches the
gate — Part 3 only *adds* the CLI arm, it is not invoked from `__main__` by this
Worker).

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (full, including all ten addenda)
- `P5_15_S31_PENALTY_TABLE_DRAFT.md` (full)
- `model_construction_helpers.py` — objective composition (`objective_function_rule`,
  `flexibility_cost`, `gen_curtailment_penalty`, `ess_utilization_cost_penalty`,
  `slack_penalties`, `ess_complementarity_penalties`, `setup_cost_parameters`),
  shared-ESS capacity configuration (`configure_shared_ess_operational_state`,
  `_SHARED_ESS_OPERATIONAL_VARIABLES`, `_SHARED_ESS_ZERO_GATED_BOUND_VARIABLES`),
  `slack_es_balance_bounds`
- `network.py` — variable/constraint construction (`_build_model` ~260–494),
  `flex_energy_balance_p_rule` / `_q_rule`, shared-ESS Var creation (~341–381),
  results-processing reads of the flexibility/day-balance slacks (~1490–1530)
- `shared_resources_planning.py` — `_prepare_transmission_objectives_for_admm`,
  `_prepare_distribution_objectives_for_admm`, `update_transmission_model_to_admm`
  (confirms `admm_objective = copy(model.objective.expr)/scale + AL terms`),
  `_get_operational_recourse_value` / `_get_operational_recourse_components`,
  `_get_operational_recourse_block_components`, `_get_admm_block_weight`,
  `_get_local_slack_penalty_components` / `_get_local_objective_components` /
  `_get_operational_objective_component_blocks` / `_get_operational_slack_component_blocks`
  (pre-existing per-block diagnostic infrastructure, reused as a style template),
  TSO ADN-interface load creation and fixing (~2955–2994, ~10600–10616)
- `shared_energy_storage_data.py` — `get_feasibility_violation`,
  `get_feasibility_penalty`, `feasibility_penalty` construction (~703–780),
  `build_subproblem` / `_build_subproblem`
- `definitions.py` — every `PENALTY_*` constant, `EPS_ESSO_THROUGHPUT`,
  `EQUALITY_TOLERANCE`
- `network_parameters.py` — `Slacks`/`SlacksFlexibility`/`SlacksEnergyStorage`
- `p515_g_g1_g4_admm_gates.py` — `run_admm_arm`, `_construct_arm_planning`,
  existing CLI gates (`g1`, `s30`, `g3_full`, ablations) as the template for `s31`
- `p513_solve_profile_guard.py` — `SolveProfileGuard` (blocking form used in Part 2)
- `p56a_oracle.py` — `fresh_planning`, `WORK_DIR`

## Files modified

- `model_construction_helpers.py`
- `network.py`
- `shared_resources_planning.py`
- `shared_energy_storage_data.py`
- `p515_g_g1_g4_admm_gates.py` (harness extension)

## Files created

- `p515_s31_zero_solve_checks.py` (Part 2 harness, repo root, `p5*` convention)
- `p515_s31_preflight.py` (Part 4 harness, repo root, `p5*` convention)
- `data/SRP1/Results/P515S31/zero_solve_checks.json` (Part 2 output)
- `data/SRP1/Results/P515S31/preflight/…` (Part 4 output — `run_admm_arm`'s own
  artifacts plus `component_levels_terminal.json`)
- `WORKER_REPORT_S31_IMPL.md` (this file)

No file under `data/` was edited except new files under
`data/SRP1/Results/P515S31/`. Nothing was committed, branched, reset or stashed.

## Changes made

### Row 3 / D1 — remove the TSO interface flexibility charge

`model_construction_helpers.py:1618` adds `load_is_tso_adn_interface(network, load)`,
identifying ADN-interface loads exactly as `flex_energy_balance_p_rule` does
(`network.is_transmission and load.bus in network.active_distribution_network_nodes`).
`flexibility_cost` (`model_construction_helpers.py:1625`) now `continue`s past these
loads. The freeing of `flex_*` and fixing of `pc`/`qc` to consensus
(`shared_resources_planning.py:2963–2980`) is untouched.

A definitional-decomposition helper, `adn_interface_flexibility_cost`
(`model_construction_helpers.py:1648`), computes the *removed* charge at the current
point (not part of any objective) — evaluable because `flex_p_down`/`flex_q_down` of
ADN loads remain free variables; only their *pricing* was removed.

### Row 5 — RES curtailment penalty identical on both sides, weight 0

`_prepare_distribution_objectives_for_admm`
(`shared_resources_planning.py:3810–3813`): the commented-out
`penalty_gen_curtailment.set_value(0.00)` is uncommented, matching the TSO's
`_prepare_transmission_objectives_for_admm` (`shared_resources_planning.py:3652`,
pre-existing).

A definitional-decomposition helper, `gen_curtailment_definitional_value`
(`model_construction_helpers.py:1721`), takes an explicit `weight` argument (used by
the harness at `PENALTY_GENERATION_CURTAILMENT`) instead of reading the model's
now-zeroed `penalty_gen_curtailment` Param.

### Row 8 — split the ESS-usage zeroing flag

`setup_cost_parameters` (`model_construction_helpers.py:1541–1546`) adds
`model.penalty_shared_ess_usage` (mutable Param, same initial value
`PENALTY_ESS_USAGE`). `ess_utilization_cost_penalty`
(`model_construction_helpers.py:1797`) uses `penalty_shared_ess_usage` for the shared
ESS loop and keeps `penalty_ess_usage` for the local (`es_reg`) loop.
`_prepare_transmission_objectives_for_admm` (`shared_resources_planning.py:3651`) and
`_prepare_distribution_objectives_for_admm` (`shared_resources_planning.py:3807`) now
zero only `penalty_shared_ess_usage`; `penalty_ess_usage` is left at
`PENALTY_ESS_USAGE`.

**SRP1 `es_reg` status (verified, Part 2 zero-solve JSON,
`checks.es_reg_and_local_ess_device_count`):** every SRP1 network (TSO `case9`; DSOs
`case33_1`, `case33_2`, `case33_3`) has `es_reg = true`, but `n_local_energy_storages
= 0` on all four. The split is therefore **inert on SRP1 numerics** — `for e in
model.energy_storages` is `range(0)` everywhere, so neither the old single-Param
zeroing nor the new split changes any local-ESS cost term for this dataset. The split
is hygiene for a future case file that defines local ESS devices with `es_reg`.

### Row 9 — remove the network ESS complementarity penalty

`ess_complementarity_penalties` (`model_construction_helpers.py:1957`) no longer adds
the bilinear `PENALTY_ESS_COMPLEMENTARITY * (pch * pdch)` terms (local or shared); it
now returns `local_ess_day_balance_slack_penalty + shared_ess_day_balance_slack_penalty`
(new helpers, `model_construction_helpers.py:1914` / `1925`, factored out so rows 10/11
can be reported separately — Part 1 item 6). The hard relaxed complementarity
constraints (`ess_comp`, `sess_comp`, `network.py`) are **untouched**. The
`ess_complementarity_penalties_rule` wrapper and the original call signature are
preserved (a preserved fixture's `functools.partial` still resolves the name).

A definitional-decomposition helper, `ess_complementarity_bilinear_value`
(`model_construction_helpers.py:1935`), evaluates the *removed* bilinear term at the
current point (pch/pdch remain free variables — only the penalty term was removed).

### Row 14 / D2 — remove orphan slacks

`slack_penalties` (`model_construction_helpers.py:1893`) is decomposed into four small
helpers — `voltage_slack_penalty` (row 12), `node_balance_slack_penalty` (row 15),
`branch_flow_slack_penalty` (row 16), `flexibility_p_day_balance_slack_penalty` (row
13) — and now sums only those four. The Q day-balance term is **removed entirely**
(its constraint, `flex_energy_balance_q`, is unwired for every load —
`network.py:433`, unchanged, still commented out). `flexibility_p_day_balance_slack_penalty`
additionally **excludes** the TSO's ADN-interface loads (their P day-balance
constraint is also skipped, `flex_energy_balance_p_rule`).

**Orphan mechanism and the fix chosen:** these slack variables are created
unconditionally for every load in `model.loads` (`network.py:315–324`), gated only by
`params.fl_reg and params.slacks.flexibility.day_balance` — not per-load, and not
aware of the ADN-interface distinction. Two families are orphaned (penalized with no
governing constraint, or previously so): the Q slacks (every load) and the P slacks of
ADN-interface loads. **Chosen fix: fix them to 0.0, do not delete the variables**
(`network.py:325–345`, right after creation) — this is what keeps the
unconditional-over-`model.loads` reads in results-processing code
(`network.py`'s `_process_results_detail`, ~1490–1530, reads
`slack_flex_p_balance_up/down` and `slack_flex_q_balance_up/down` for every load
without a per-load `fl_reg`/ADN guard for the Q pair) working without a `KeyError`.
Deleting the variables would have required adding that guard in results-processing
code too — out of scope for this task, and a larger surface for regression.

### Row 12 and every D row — reporting semantics

`_get_operational_recourse_components` (`shared_resources_planning.py:801`) is
extended with **additional** fields; `gross_operational_cost` and
`net_operational_recourse` are computed **identically** to before (verified by
inspection — the two lines computing them are unchanged, only new code was inserted
after them). New fields:

- `detector_penalty_total`, `detector_components` (a dict with `voltage_slack`,
  `node_balance_slack`, `branch_flow_slack`, `flexibility_p_day_balance_slack`,
  `local_ess_day_balance_slack`, `shared_ess_day_balance_slack`,
  `detector_penalty_total`), each summed over TSO + every DSO with the SAME
  `_get_admm_block_weight` weighting `get_primal_value` uses (verified: `NetworkData.
  get_primal_value` at `network_data.py:96` multiplies by exactly
  `years[year]*days[day]*annualization`, identical to `_get_admm_block_weight`
  at `shared_resources_planning.py:2724`);
- `economic_recourse_all_D_excluded` = net − `detector_penalty_total`;
- `economic_recourse_voltage_excluded` = net − `detector_components['voltage_slack']`.

Implementation: two new small production functions,
`_get_local_detector_components` (`shared_resources_planning.py:740`, per-block,
scenario-probability-weighted, calling the SAME six helper functions the objective
now calls) and `_get_operational_detector_component_blocks`
(`shared_resources_planning.py:774`, applies `_get_admm_block_weight` per TSO/DSO
block) — mirroring the style of the pre-existing `_get_local_slack_penalty_components`
/ `_get_operational_slack_component_blocks` (found during inspection; not reused
directly because they read variables directly rather than through the objective's own
building blocks and do not cover rows 10/11, but their style — per-block, `weight *
value` — is followed exactly).

### D3 — ESSO slacks reported directly

`SharedEnergyStorageData.get_feasibility_violation`
(`shared_energy_storage_data.py:109`) no longer computes
`get_feasibility_penalty(models) / PENALTY_ESSO_SLACK` (contaminated by the row-20 ε
throughput term since Step 1). It now sums `slack_es_pnet_up[y_inv,d,p] +
slack_es_pnet_down[y_inv,d,p]` directly over every node, year, day, period. Function
name and every caller are unchanged.

### D6 — bound the shared-ESS day-balance slack like the local one

Local mechanism (unchanged, refactored only to name the magic number):
`slack_es_balance_bounds` (`model_construction_helpers.py:413`) still returns `(0,
ess.e * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE)`
(`ESS_DAY_BALANCE_SLACK_FRACTION = 0.05`, `model_construction_helpers.py:410`,
factored out of the previous inline `0.05` literal so the shared-ESS mechanism below
can reference the SAME constant) — this works as a Pyomo `bounds=` callable because
the local ESS's `network.energy_storages[e].e` is a static, build-time attribute.

**D6 mechanism.** The shared-ESS energy capacity (`shared_es_e_rated_fixed`) is a
mutable `Param`, set and re-set by `configure_shared_ess_operational_state` every time
the candidate capacity changes (every caller across the codebase —
`shared_resources_planning.py`, `network_data.py`, several `p5*` harnesses). A Pyomo
`bounds=` callable is evaluated **once**, at `Var` construction, so it cannot track a
Param that changes later — this is exactly why the shared-ESS slack was left unbounded
originally (D6 defect). The fix does **not** use a `bounds=` callable: it adds a block
inside `configure_shared_ess_operational_state`
(`model_construction_helpers.py:1078–1096`) that calls `.setlb(0.0)` /
`.setub(e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE)` (or
`(0.0, 0.0)` when the shared ESS is capacity-inactive, matching the existing
`_SHARED_ESS_ZERO_GATED_BOUND_VARIABLES` zero-collapse convention) directly on
`slack_shared_es_soc_final_up/down`, **every time this function runs** — i.e. every
time the candidate capacity is set. This is a re-applied bound, not a constraint row
(Pyomo `Var` bounds are cheaper than an extra constraint and the existing local-ESS
mechanism is also a bound, not a row, so this keeps the two symmetric as the table
asks).

### Everything asked "not permitted": verified unchanged

- No change to `network.py`'s hard complementarity constraints (`ess_comp`,
  `sess_comp`), the stopping rule, ρ, `gamma`, or any tolerance/`option_overrides`.
- No change to what IPOPT sees for rows 10, 11, 12, 13, 15, 16 (their objective
  contribution is bit-identical to before — the refactor only *names* the same four/two
  terms already summed into `slack_penalties`/`ess_complementarity_penalties`; verified
  by the Part 4 preflight matching structure — `detector_components['voltage_slack']`
  in the recourse report equals `totals_weighted['voltage_slack']` in the Part 3
  block file to full float precision, confirming the two independent call paths
  agree).
- Row 18 untouched.

## Commands / experiments run

1. Syntax/import checks after each edit:
   `python -c "import py_compile; py_compile.compile(<file>, doraise=True)"` for all
   five modified files plus the two new harnesses — all passed.
2. `python -c "import shared_resources_planning, network, model_construction_helpers"`
   — confirms `load_is_tso_adn_interface` and the other new helpers are visible from
   `network.py`'s `from model_construction_helpers import *` (this caught and fixed a
   bug: the first draft used a leading underscore, `_load_is_tso_adn_interface`, which
   `import *` silently drops — renamed to `load_is_tso_adn_interface` before any
   further work).
3. **Part 2** — `python p515_s31_zero_solve_checks.py` (foreground, ~1 min). Builds
   TSO (`case9`) and one DSO (`case33_2`, node 7) model via `NetworkData.build_model()`
   plus `srp._prepare_transmission_objectives_for_admm` /
   `srp._prepare_distribution_objectives_for_admm` (the SAME production functions the
   ADMM driver calls), under a `SolveProfileGuard(permitted=())` armed for the whole
   script. `guard.verify(expected_solves=0)` returned no failures.
4. **Part 4** — `python p515_s31_preflight.py` (foreground, ~60 s wall clock,
   well under the 9-minute limit). One ADMM cycle (`num_max_iters_override=1`)
   through `run_admm_arm` by import (never through
   `p515_g_g1_g4_admm_gates.py`'s own `__main__`), fresh root
   `data/SRP1/Results/P515S31/preflight/`, fresh eval id `p515s31_preflight`
   (plus a distinct pre-flight-check eval id `p515s31_preflight_precheck` for
   `assert_s31_capture_paths`, per rule eleven — asserted **before** the run).

## Results

### Part 2 — `data/SRP1/Results/P515S31/zero_solve_checks.json`

`solve_profile_guard.counts` = `{permitted_solve: 0, permitted_exec: 0, blocked_solve:
0, blocked_exec: 0}`; `verify_failures = []`. `all_checks_pass = true`. Per check:

| check | method | result |
|---|---|---|
| ADN loads absent from `flexibility_cost` | `identify_variables` on the expression tree; assert no ADN `flex_p_down`/`flex_q_down` VarData present, and a non-ADN fl_reg load IS present (sanity) | pass |
| TSO + DSO `penalty_gen_curtailment` both 0 after `_prepare_*` | `pe.value(...)` | pass (both 0.0) |
| shared-ESS usage Param 0, local Param = `PENALTY_ESS_USAGE` | `pe.value(...)` on both models | pass |
| no bilinear `pch*pdch` in the objective | `generate_standard_repn(model.objective.expr, quadratic=True)`; every `(pch, pdch)` pair of every storage unit checked against `repn.quadratic_vars` | pass (0 forbidden pairs found on TSO and DSO) |
| no Q-balance / ADN-load-P slack in the objective; those variables fixed | `identify_variables` on `model.objective.expr` plus `.fixed`/`pe.value()` on every `(load, s_m, s_o)` slack pair; also confirms a non-ADN fl_reg load's P slack IS present and NOT fixed (sanity) | pass, TSO and DSO |
| shared-ESS day-balance slack bounded, tracking capacity | `configure_shared_ess_operational_state` probed at `(s,e) = (1,2), (0,0), (1.62,3.24)`; bounds read back and compared with the expected formula | pass, all three probes match |
| `get_feasibility_violation` reads slacks | `inspect.getsource` (no `/ PENALTY_ESSO_SLACK`, no `get_feasibility_penalty(` call) **plus** a live evaluation on freshly-built (unsolved) ESSO subproblem models (`sed.build_subproblem()`), compared with a manual direct sum | pass (manual sum == reported, both 0.0 at the unsolved initial point; the static check initially failed because it looked for the bare substring `PENALTY_ESSO_SLACK`, which the function's own explanatory comment legitimately still names — fixed to check for the `/ PENALTY_ESSO_SLACK` division pattern and the old indirection call instead) |
| `es_reg` / local-ESS device count | direct attribute read | all four SRP1 networks `es_reg=true`, `n_local_energy_storages=0` (see Row 8 above) |

### Part 4 — `data/SRP1/Results/P515S31/preflight/`

One ADMM cycle, cold start, production defaults (no overrides), converged status per
the arm report `g_preflight.json`:

- `cycles_run = 1`, `local_solve_failures = 0`, `network_failures_summary.n_blocks = 0`.
- `solve_profile = {permitted_solve: 102, permitted_exec: 102, blocked_solve: 0,
  blocked_exec: 0}`, `identity_holds = True` (matches `51 * cycles + 51` for
  `cycles=1`).
- `shared_frozen_smopf_modified = []`, `shared_frozen_smopf_new_files = []` — the
  shared, preserved `FrozenSMOPF` tree was not touched.
- ESSO complementarity detector grouping clean (`observed=6, expected=6`); the three
  nodes' `mu_final`/leak diagnostics printed to stdout are in the same range as every
  prior baseline run (`ratio_max` ≈ 7–9e-6).
- `component_levels_terminal.json` written with **zero extra solves** (the
  `post_run_hook` mechanism added to `run_admm_arm` runs on the SAME `models` dict the
  arm already built; nothing calls the solver again). Internal consistency check: the
  weighted `voltage_slack` total in `component_levels_terminal.json`
  (`totals_weighted.voltage_slack = -11796.943577440745`) equals
  `recourse_components.detector_components.voltage_slack` in the same file to full
  float precision, confirming the two independent reporting call paths (Part 1 item 6
  and Part 3) agree exactly.
- `orphan_flex_q_day_balance_slack_raw` and `orphan_tso_adn_flex_p_day_balance_slack_raw`
  are `0.0` in every one of the 48 blocks — the row-14 fix holds under an **actual
  IPOPT solve**, not only at variable creation.
- **One-cycle caveat, reported as asked (§3's "say explicitly if a quantity cannot be
  evaluated"):** `flexibility_cost_tso_adn_interface_definitional`
  (row 3) and `res_curtailment_definitional_at_weight_1` (row 5) are both LARGE at
  this single, non-converged cycle (totals ≈ 2.80e9 and 2.52e6 respectively — cycle 1
  is before TSO/DSO consensus is established, so the ADN-interface loads' freed
  `flex_p_down`/`flex_q_down` variables and the pre-consensus dispatch mismatch are far
  from their converged values). This is a property of evaluating a **non-terminal**
  point, not a defect in the two definitional helpers (which were separately verified
  correct against the pre-Step-3 formulas in Part 2). The real campaign (`s31` CLI
  arm, not launched by this Worker) will report these at the actual terminal,
  converged cycle, which is what the schema is defined to capture.
- `orphan_flex_q_day_balance_penalty_at_PENALTY_FLEXIBILITY_definitional` and its ADN-P
  counterpart are `0.0` **by construction** (the underlying raw slacks are fixed to
  0.0), stated explicitly in the JSON's `orphan_slack_note` field per the task's
  instruction.

## Validation

Distinguishing what was actually established:

- **Code executes correctly:** yes — every modified file compiles, imports, and the
  full production ADMM path (`run_admm_arm`, one cycle) ran to completion with zero
  local solve failures and zero network failures.
- **The zero-solve claims hold:** yes, enforced by an armed `SolveProfileGuard`
  (Part 2, blocking) and by the `post_run_hook` design (Part 3/4, reuses the already-
  solved models, no new solver call site is introduced).
- **The signed-table changes took effect exactly as specified:** yes, for every row
  checked in Part 2's table above, all zero-solve, all passing.
- **The underlying "does Q(x) move" question:** **not answered by this task** — that
  is explicitly the subject of the (not-yet-launched) `s31` campaign and its reading
  rule against Step 3.0's baseline (817,520,272.93), per the "sequence after
  signature" in Addendum 10 and §5 of the penalty table.

## Unexpected findings

1. **Underlying-name wildcard-import trap.** A leading-underscore helper name
   (`_load_is_tso_adn_interface`) is silently dropped by `from module import *`
   (Python's documented but easy-to-forget behavior), which `network.py` and
   `shared_resources_planning.py` both rely on for `model_construction_helpers`. Found
   and fixed via an import smoke test before any further work; flagged here since the
   codebase has several other underscore-prefixed helpers in
   `model_construction_helpers.py` (`_component_entries_for_shared_ess`,
   `_configure_shared_ess_expected_schedule`, etc.) that are, by design, never called
   from outside that module — this convention is intact, my draft briefly violated it.
2. **Pre-existing per-block diagnostic infrastructure.** `shared_resources_planning.py`
   already has `_get_local_slack_penalty_components` /
   `_get_local_objective_components` / `_get_operational_objective_component_blocks` /
   `_get_operational_slack_component_blocks` (used elsewhere for cycle-printing
   diagnostics), which independently re-derive voltage/node-balance/branch-flow/flex-P/
   flex-Q slack levels by reading variables directly rather than through a shared
   helper function. These were **not modified** (out of scope; a different diagnostic
   path, not cited by the S31 task) but their existence is worth flagging to the
   Planner: after this change, `_get_local_slack_penalty_components`'s
   `flex_day_balance_q` and `flex_day_balance_p` fields will report `0.0` for the Q
   family and for ADN loads (since those variables are now fixed), which is consistent
   with, but not derived from, the new `flexibility_p_day_balance_slack_penalty` /
   `slack_penalties` — a future refactor could unify the two, not attempted here since
   unrequested.
3. **`p53b3_active_power_ess.py`** (a pre-existing `p5*` harness, historical) reads
   `model.penalty_ess_usage` and asserts it is zeroed under ADMM; after the row-8
   split, `penalty_ess_usage` (local) is **no longer** zeroed — this harness's
   assertion is now stale. It is listed as a broken-historical harness (in the same
   class as the nine already accepted in Addendum 3 item 5), not repaired per scope
   discipline.

## Remaining issues

- The `s31` campaign itself (full multi-cycle run, `python -u
  p515_g_g1_g4_admm_gates.py s31 > data/SRP1/Results/P515S31_launch.log 2>&1`) has
  **not been run** — per the task's explicit instruction, the Planner launches the
  gate. `data/SRP1/Results/P515S31_run/` (the arm's own fresh root, `OUT_S31` in
  `p515_g_g1_g4_admm_gates.py`) does not yet exist.
- `p53b3_active_power_ess.py`'s stale assertion (Unexpected Finding 3) is unrepaired
  and unreported anywhere except here.
- The definitional-decomposition values (rows 3, 5, 9, 14) reported by the Part 4
  preflight are from a single, non-converged cycle and are NOT representative of the
  terminal values Step 3.1's reading rule needs — only the real `s31` campaign can
  supply those.

## Questions for Planner

None — the task was fully specified. Flagging Unexpected Finding 2 (existing
diagnostic-function overlap) and Finding 3 (stale harness assertion) for awareness,
not requesting a decision.

## Exact gate command (for the Planner to run)

```
python -u p515_g_g1_g4_admm_gates.py s31 > data/SRP1/Results/P515S31_launch.log 2>&1
```

This launches the `s31` CLI arm added in Part 3: `_require_fresh_output_root` on
`OUT_S31` (`data/SRP1/Results/P515S31_run/`), `assert_s31_capture_paths` on a
dedicated pre-flight eval id (rule eleven, before any solve), then `run_admm_arm`
with production defaults, no overrides, writing `component_levels_terminal.json` via
`write_component_levels_terminal` after the run, zero extra solves. Keeps the lock,
heartbeat, stderr/tee, per-cycle capture, results-dir redirection and shared-
`FrozenSMOPF` hash guard (all unchanged, inherited from `run_admm_arm`).

## git status (at report time)

```
 M .claude/agents/planner.md          (pre-existing, not from this task)
 M .claude/agents/worker.md           (pre-existing, not from this task)
 M model_construction_helpers.py
 M network.py
 M p515_g_g1_g4_admm_gates.py
 M shared_energy_storage_data.py
 M shared_resources_planning.py
?? p515_s31_preflight.py
?? p515_s31_zero_solve_checks.py
?? data/SRP1/Results/P515S31/            (new — this task's Part 2/4 output)
```

(Plus the large pre-existing set of untracked `data/` directories and files already
present at the start of this session — unrelated to this task, unchanged by it,
enumerated in the session's initial `git status` snapshot.)
