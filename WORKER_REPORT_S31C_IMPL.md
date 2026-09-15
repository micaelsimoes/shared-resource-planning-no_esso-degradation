# Worker Report — S31C: signed interface reparametrization + interface energy settlement

**Task authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 12, with `P5_15_S31B_ROW3PRIME_REVIEW.md`
(diagnosis) and `P5_15_S31_PENALTY_TABLE_DRAFT.md` (signed table) as context. HEAD at task start:
`fc200780` (signed-table rows) + `b2c24371` (level-capture arm `s31`).

## Task received

Production change (Part 1: signed interface reparametrization, interface energy settlement, Q(x)
exclusion, reporting helper) + Part 2 (blocking zero-solve verification) + Part 3 (harness arm
`s31c`) + Part 4 (one-cycle pre-flight, foreground). No commit. No edits under `data/` except new
files in `data/SRP1/Results/P515S31C/`.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Addenda 10, 11, 12), `P5_15_S31B_ROW3PRIME_REVIEW.md`,
  `P5_15_S31_PENALTY_TABLE_DRAFT.md`
- `shared_resources_planning.py`: `create_transmission_network_model` (~3139-3230),
  `_run_operational_planning_hierarchical` (~3358-3420), `_run_operational_planning_without_coordination`
  (~6070-6100), `_get_operational_recourse_components` / `_get_operational_recourse_block_components`
  (~801-1005 pre-edit), `_prepare_transmission_objectives_for_admm` /
  `_prepare_distribution_objectives_for_admm`, `create_admm_variables`, `_get_admm_block_weight`,
  `get_initial_candidate_solution`, `_rebuild_candidate_total_capacities`
- `model_construction_helpers.py`: `interface_pf_p_transmission_def` /
  `interface_pf_q_transmission_def` (~1167-1188 pre-edit), `interface_pf_p_distribution_def` /
  `interface_pf_q_distribution_def`, `compute_node_load` (~1275-1312 pre-edit), `load_is_tso_adn_interface`,
  `build_objective` / `objective_function_rule` (~1558-1614 pre-edit), `generation_cost`,
  `flexibility_cost`
- `network.py`: `_build_model` (~250-420, `adn_nodes`, `pc_adn`/`qc_adn`/`pg_adn`/`qg_adn` Expressions,
  loads/flex-leg Vars), `get_primal_value`, `get_interface_branch_rating`, `get_adn_load_idx`
- `p513_solve_profile_guard.py` (`SolveProfileGuard`)
- `p515_s31_zero_solve_checks.py`, `p515_s31_preflight.py`, `p515_g_g1_g4_admm_gates.py`
  (`assert_s31_capture_paths`, `write_component_levels_terminal`, `run_admm_arm`, gate dispatch)
  for the existing `s31` conventions this task extends

## Files modified

- `network.py`
- `model_construction_helpers.py`
- `shared_resources_planning.py`
- `p515_g_g1_g4_admm_gates.py`

## Files created

- `p515_s31c_zero_solve_checks.py` (Part 2 harness)
- `p515_s31c_preflight.py` (Part 4 harness)
- `data/SRP1/Results/P515S31C/zero_solve_checks.json` (Part 2 evidence)
- `data/SRP1/Results/P515S31C/preflight/*` (Part 4 evidence: `g_preflight.json`,
  `component_levels_terminal.json`, `interface_settlement_detail_s31c.json`, `heartbeat_preflight.json`,
  `stdout_preflight.log`, `leak_classification_preflight.jsonl`, `network_failures_preflight.jsonl`,
  `esso_capture/`)

## Changes made

### Part 1.1 — Signed interface reparametrization (`interface_delta_p`/`interface_delta_q`)

- `network.py:405-421` (`_build_model`, guarded by `if network.is_transmission:`): new
  `model.interface_delta_p` / `model.interface_delta_q`, `Reals`, indexed
  `(adn_nodes, scenarios_market, scenarios_operation, periods)`, **fixed at 0** immediately at
  construction — the default in every TSO model (hierarchical, uncoordinated, and the pre-fix state
  of the ADMM build).
- `model_construction_helpers.py:1167-1198` (`interface_pf_p_transmission_def` /
  `interface_pf_q_transmission_def`): `pc_adn`/`qc_adn` now **add** `interface_delta_p[dn,...]` /
  `interface_delta_q[dn,...]` to the existing `pc + legs` expression — additively, not by
  replacement. With δ fixed at 0 (every path except ADMM) the expression is exactly `pc + legs` as
  before; with the legs fixed at 0 (ADMM path only, below) it is exactly `pc + δ`. No path-specific
  branch is needed inside this shared rule function.
- `model_construction_helpers.py:1275-1310` (`compute_node_load`): for a load that is a TSO
  ADN-interface load (`network.is_transmission and load_is_tso_adn_interface(...)`), the node
  balance additionally adds `interface_delta_p[dn,...]` / `interface_delta_q[dn,...]`, on top of
  the unchanged `fl_reg` legs branch — so the interface expression (`pc_adn`) and the node-balance
  contribution can never diverge (verified in Part 2, `node_balance_matches_interface`, abs
  difference `0.0`).
- `shared_resources_planning.py:3196-3212` (`create_transmission_network_model`, ADMM path only):
  the four ADN-load legs (`flex_p_up/down`, `flex_q_up/down`) are now **fixed at 0.00** (previously
  freed with `setub(interface_transf_rating)`); `interface_delta_p`/`interface_delta_q` are freed
  (`fixed = False`) and bounded `[-interface_transf_rating, +interface_transf_rating]` — the same
  rating the legs used. Interface flexibility is **not** fixed at zero: δ is the free channel.

The hierarchical (`_run_operational_planning_hierarchical`, `shared_resources_planning.py`
~3358-3420) and uncoordinated (`_run_operational_planning_without_coordination`, ~6070-6100) paths
are **untouched** — `git diff` contains zero hunks inside either function (checked directly, see
Validation). They call the shared rule functions above, but since δ stays fixed at 0 there (the
default from `network.py`) and the legs keep their pre-existing free/fixed state, the added `+ δ`
term evaluates to `+ 0` and both paths behave exactly as before.

### Part 1.2 — Interface energy settlement

- `model_construction_helpers.py:1558-1567,1603-1650` (`build_objective`, new function
  `interface_energy_settlement`): mutable `Param model.interface_settlement_weight` (default
  `0.00`) and `Expression model.interface_settlement`, built for **every** network model (TSO and
  DSO) in the same `build_objective` call every network already goes through.
  `interface_energy_settlement(model, network)`:
  - TSO (`network.is_transmission`): `T_TSO = -Σ_dn Σ_{s_m,s_o} prob · Σ_p π[s_m][p]·baseMVA·pc_adn[dn,s_m,s_o,p]`
  - DSO: `T_DSO = +Σ_{s_m,s_o} prob · Σ_p π[s_m][p]·baseMVA·pg_adn[s_m,s_o,p]`
  - `π = network.cost_energy_p`, the same array `generation_cost` uses; no anchor.
- `model_construction_helpers.py:1651-1657` (`objective_function_rule`): `obj +=
  model.interface_settlement_weight * model.interface_settlement`, added to the **physical**
  objective (before the ADMM path's `effective_scale` division at
  `shared_resources_planning.py:3732` area — `obj = copy(model[year][day].objective.expr) /
  effective_scale`), so cancellation holds in currency units.
- `shared_resources_planning.py:3813-3815` (`_prepare_transmission_objectives_for_admm`) and
  `:3977-3979` (`_prepare_distribution_objectives_for_admm`): `interface_settlement_weight.set_value(1.00)`,
  ADMM path only. Every other build keeps the `0.00` default.

### Part 1.3 — Q(x) excludes settlement by construction

- `shared_resources_planning.py:801-830` (`_get_local_interface_settlement`,
  `_get_operational_interface_settlement_blocks`, new): per-block settlement contribution, weighted
  (`_get_admm_block_weight`) the same way `_get_operational_detector_component_blocks` already is.
- `shared_resources_planning.py:891-951` (`_get_operational_recourse_components`, rewritten): computes
  `gross_operational_cost_including_settlement` exactly as the pre-existing `gross_operational_cost`
  was computed (`get_primal_value`, which now includes the settlement at its current weight); then
  subtracts `interface_settlement_total` (`interface_settlement_tso + Σ interface_settlement_dso`)
  to obtain the **settlement-excluded** `gross_operational_cost`; `net_operational_recourse` is
  `gross_operational_cost - terminal_salvage_value` as before, now on the settlement-excluded
  value. New fields: `interface_settlement_tso`, `interface_settlement_dso` (dict keyed by node
  id), `interface_settlement_total`, `gross_operational_cost_including_settlement`. The D-row
  fields (`detector_penalty_total`, `detector_components`, `economic_recourse_all_D_excluded`,
  `economic_recourse_voltage_excluded`) are kept, now derived from the settlement-excluded value.
  **The stopping test in `_run_operational_planning` (the recourse-stationarity / objective-change
  check, ~line 2600 area) reads `net_operational_recourse` unchanged — its mechanism is untouched;
  its input is now the settlement-excluded recourse**, stated explicitly in the code comment
  (`shared_resources_planning.py:919-926`).
- **In-scope consequential fix**, `shared_resources_planning.py:967-1013`
  (`_get_operational_recourse_block_components`): this function's own docstring promises
  `sum(blocks.values()) == net_operational_recourse`. Since it independently calls
  `network.get_primal_value(...)` (which now includes the settlement), leaving it unmodified broke
  that promise by exactly the settlement total — reproduced live as `[WARNING][RECOURSE JUMP] Block
  decomposition does not reconcile ... difference=-8.400107e+08` on the Part 4 pre-flight (see
  Unexpected findings). Fixed by subtracting `_get_local_interface_settlement(model)` from each
  block's `local_value` before weighting, restoring the reconciliation (warning no longer appears
  on the re-run). This is the only change outside the literal Part 1 item list; it is a same-class,
  same-mechanism, minimal fix to a function the settlement change silently broke.

### Part 1.4 — Reporting helper

- `shared_resources_planning.py:837-889` (`_get_interface_reporting_detail`, new; public wrapper
  `SharedResourcesPlanning.get_interface_reporting_detail` at `:193-194`): per DSO node, per
  year/day/period — `p_int_tso_expected_mw`/`q_int_tso_expected_mvar` (the AL-coupled
  `expected_interface_pf_p/q` on the TSO side), `p_int_dso_expected_mw`/`q_int_dso_expected_mvar`
  (same, DSO side), `delta_p_mw`/`delta_q_mvar` (per market/operation scenario — SRP1 has exactly
  one of each), `anchor_p_mw`/`anchor_q_mvar` (the TSO's fixed `pc`/`qc`, per scenario), the market
  price `price_per_mwh`, and per (year, day) `dso_settlement_sum_pi_p_int` (Σ_t π_t·p_int,t,
  unweighted — reporting granularity, not the recourse total).

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` on all four modified production files and the two new
   harness scripts — syntax OK (run twice, after each further edit).
2. `python p515_s31c_zero_solve_checks.py` (canonical interpreter) — three times (once to discover
   the sequencing gap described below, twice more after fixes); final run:
   `all_checks_pass=True solve_guard_failures=[]`.
3. `python p515_s31c_preflight.py` (canonical interpreter, foreground) — three times: once (found
   the block-reconciliation bug via the recourse-jump warning and the units mismatch in the
   consensus-residual weighting), once after the `_get_operational_recourse_block_components` fix
   (warning gone, sign/magnitude mismatch in the consensus-residual comparison found), once more
   after the weighting fix in `_s31c_interface_detail` (final, reported below). Each run < 70 s
   wall clock.
4. `git diff --stat` / `git diff -- shared_resources_planning.py | grep '^@@'` to confirm every hunk
   is inside the intended functions and that `_run_operational_planning_hierarchical` /
   `_run_operational_planning_without_coordination` have zero diff.

No solve of the actual `s31c` campaign gate was run (`python -u p515_g_g1_g4_admm_gates.py s31c`
was **not** executed — the Planner launches the gate, per the task's execution rules).

## Results

### Part 2 — zero-solve verification (`data/SRP1/Results/P515S31C/zero_solve_checks.json`)

Blocking `SolveProfileGuard(permitted=())`: `counts={'permitted_solve': 0, 'permitted_exec': 0,
'blocked_solve': 0, 'blocked_exec': 0}`, `verify_failures=[]`. All 12 checks pass
(`all_checks_pass: true`):

| check | result |
|---|---|
| `delta_fixed_zero_by_default` | δ_P/δ_Q fixed, value 0.0, on a bare `build_model()` |
| `hierarchical_uncoordinated_delta_untouched` | static: neither path function's source references `interface_delta` |
| `price_identical_tso_dso` | TSO and DSO `cost_energy_p` exactly equal, every scenario/period (0 mismatches) |
| `delta_free_bounds_admm` | δ not fixed, bounds = ±`interface_transf_rating`, every period, in the ADMM-built model |
| `legs_fixed_zero_admm` | all four legs fixed, value 0.0, every period, ADMM-built model |
| `interface_expression_pc_delta_no_legs` | `identify_variables(include_fixed=True)` shows pc, δ **and** the (fixed) legs present in `pc_adn`; `include_fixed=False` shows δ as the **sole** free variable |
| `node_balance_matches_interface` | node-balance `Pd` == `pc_adn` value, abs difference `0.0` |
| `settlement_weight` | raw builds (TSO, DSO): `0.0`; ADMM-prepared (TSO, DSO): `1.0` |
| `cancellation_identity` | matching `p_int` → `T_TSO+T_DSO = 0.0` exactly; differing `p_int` (30 MW vs 18 MW) → sum matches `prob·π·baseMVA·(p_DSO-p_TSO)` to `<1e-12` |
| `no_quadratic_terms_from_settlement` | quadratic-term count in `model.objective.expr` unchanged between weight-0 and weight-1 builds (TSO and DSO) |
| `gross_operational_cost_excludes_settlement` | `gross_operational_cost + interface_settlement_total == gross_operational_cost_including_settlement` to `<1e-6` |
| `fixture_unpickling` | all three named fixtures (`P512R/cycle21_pre_setup/snapshot.pkl`, `P512R/production_snapshots/.../matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl`, `P515G3F_r2/.../frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl`) `pickle.load` without error |

### Part 3 — harness arm `s31c` (schema)

`p515_g_g1_g4_admm_gates.py`:
- `OUT_S31C = data/SRP1/Results/P515S31C_run/`
- `assert_s31c_capture_paths` — extends `assert_s31_capture_paths` with 8 new checks (δ Vars,
  `interface_settlement`/`interface_settlement_weight` present on TSO and DSO, the three new `srp`
  reporting functions are callable); raises before any solve if any is missing.
- `_s31c_interface_detail` — zero-solve, reads only `_get_interface_reporting_detail` /
  `_get_operational_interface_settlement_blocks` / `_get_local_interface_settlement`: per-block
  `interface_settlement` (unweighted and weighted); `t_tso_total`, `t_dso_by_node`,
  `t_tso_plus_t_dso_terminal`; `interface_consensus_residual_per_dso` (TSO − DSO expected `p_int`,
  MW, per period, both unweighted and block-weighted, plus the priced residual in both
  conventions); `flexibility_volumes_per_dso` (Σ|δ_P|, max|δ_P|, Σ|δ_Q|, max|δ_Q|, MW/MVAr and pu,
  with the anchor already carried in `interface_reporting_detail`).
- `write_interface_settlement_detail_s31c` — calls `write_component_levels_terminal` (the `s31`
  writer, **unmodified**, reused verbatim) first, then writes the extension to a companion file
  `interface_settlement_detail_s31c.json` in the same `out_dir` (rather than mutating the `s31`
  file's bytes), carrying a `reading_rule` field stating the exact sign identity (below).
- Gate dispatch `elif gate == 's31c':` — fresh root, fresh eval id `p515s31c_baseline`, no
  overrides, pre-flight capture-path assertion, `run_admm_arm('baseline', OUT_S31C, ...,
  post_run_hook=_s31c_hook)`. Lock (`_acquire_exclusive_run_lock` at the top of `__main__`),
  heartbeat, stdout tee, per-cycle capture, results-dir redirection and the shared-FrozenSMOPF hash
  guard are all inherited unchanged from `run_admm_arm` (not touched).

### Part 4 — one-cycle pre-flight (`data/SRP1/Results/P515S31C/preflight/`)

`num_max_iters_override=1`, fresh eval id `p515s31c_preflight`, fresh smoke root. Result (final
run, reproduced identically on the run before it):

- `cycles_run=1`, `local_solve_failures=0`, `network_failures={n_blocks: 0, all classes 0}`.
- `solve_profile={'observed': {'permitted_solve': 102, 'permitted_exec': 102, 'blocked_solve': 0,
  'blocked_exec': 0}, 'identity_holds': True}` — 102 = 2 rounds (initialization + the one ADMM
  cycle) × (48 network SMOPFs [12 TSO year/day + 3 DSOs × 12 year/day] + 3 ESSO solves); every one
  went through a permitted call site; nothing raised.
- `recourse (net_operational_recourse, settlement-excluded) = 2,359,311,749.284464`.
- `t_tso_total = -2,141,667,840.885775`; `t_dso_by_node = {'5': 429,493,343.111, '7':
  417,668,120.206, '9': 454,495,703.199}`; `t_tso_plus_t_dso_terminal = -840,010,674.369292`
  (block-weighted).
- `total_priced_consensus_residual_weighted = 840,010,674.369291` (block-weighted, TSO−DSO
  ordering); `total_priced_consensus_residual_unweighted = 2,013,488.469101`.
- **Identity check**: `t_tso_plus_t_dso_terminal == -1 × total_priced_consensus_residual_weighted`
  to 6 significant figures (`-840,010,674.369292` vs `-840,010,674.369291`) — the sign is negative
  because the residual is defined **TSO − DSO** (as the task text specifies) while
  `T_TSO+T_DSO = prob·π·baseMVA·(p_DSO−p_TSO)` (**DSO − TSO**); documented in the
  `interface_settlement_detail_s31c.json` `reading_rule` field so the Planner reads the two numbers
  correctly. This large a residual is expected at cycle 1 of a cold start — the S31B discriminator
  already showed the interface consensus has not formed early in the run (PF max primal residual
  `0.364822` p.u. here, far above the `0.01` tolerance); nothing about this run contradicts that.
- `EFC/day max across nodes = 0.000286` (threshold `1.4612`) — sane at one cycle.

## Validation

- Syntax: `ast.parse` clean on all four production files and both new harness scripts.
- Diff scope: `git diff --stat` touches exactly `network.py`, `model_construction_helpers.py`,
  `shared_resources_planning.py`, `p515_g_g1_g4_admm_gates.py`; `git diff -- shared_resources_planning.py
  | grep '^@@'` shows every hunk inside `SharedResourcesPlanning` (one new method),
  `_get_operational_detector_component_blocks`→`_get_operational_recourse_block_components` (the
  recourse/settlement block, one contiguous region), `create_transmission_network_model`, and the
  two `_prepare_*_objectives_for_admm` functions — **zero hunks** in
  `_run_operational_planning_hierarchical` or `_run_operational_planning_without_coordination`
  (grep for both names against the diff returns nothing).
- Part 2's 12 checks executed under a **blocking** guard (`permitted=()`), so "zero solves" is
  enforced, not asserted; `guard.verify(expected_solves=0)` returned no failures.
- Part 4 executed one full ADMM cycle through the **real** production path (`run_admm_arm` →
  `planning.run_operational_planning(type='distributed', ...)`), with `SolveProfileGuard` counting
  (not blocking) — 102/102 solves through permitted call sites, 0 blocked, confirming no stray
  solve path was introduced.
- I distinguish: the code **executes** (syntax + live run, confirmed); the zero-solve **diagnostics
  pass** (12/12, confirmed under a blocking guard); the one-cycle **pre-flight runs cleanly**
  (confirmed, zero failures). I do **not** claim the campaign **converges** or that `T_TSO+T_DSO`
  "vanishes" at C\* — that is what the (not-yet-run) `s31c` gate campaign is for.

## Unexpected findings

1. **`_get_operational_recourse_block_components` reconciliation bug (fixed, in scope).** Described
   above under Part 1.3 — caught live via `[WARNING][RECOURSE JUMP] ... difference=-8.400107e+08`
   on the first Part 4 pre-flight run, fixed, re-verified clean.
2. **Units mismatch in the Part 3 consensus-residual field (fixed, harness-only).** My first
   `_s31c_interface_detail` implementation summed the priced residual **without** the
   `_get_admm_block_weight` (year-count × day-count × discount annualization) factor that
   `t_tso_plus_t_dso_terminal` carries, so the two were off by orders of magnitude even though the
   underlying identity held. Fixed by adding a `_weighted`/`_unweighted` pair of fields
   consistently, documented in `reading_rule`.
3. **`p56a_oracle.py`'s `per_block_base_objectives` (NOT modified, out of scope).** This function
   (used by the derivative-free-planner candidate-evaluation pipeline, not authorized for this
   task) decomposes each block's objective into named families (`generation_cost`,
   `flexibility_cost`, …) that do **not** include `interface_settlement`. It is therefore silent
   about the settlement rather than double-counting it, so it will not sum to `get_primal_value`'s
   block total once the settlement weight is 1 on any model it inspects — the same class of issue
   item 1 above fixed in `_get_operational_recourse_block_components`, but in a module outside this
   task's authorized scope. `evaluate_planning_candidate`'s own `polished['gross_operational_cost']`
   (which calls `planning.get_operational_recourse_components`, i.e. my fixed function) is
   unaffected. Flagging for the Planner; not fixed here.
4. **Sign convention of the consensus residual.** The task text specifies "TSO expected p_int −
   DSO expected p_int"; the algebraic identity from Part 1's settlement definitions is
   `T_TSO+T_DSO = prob·π·baseMVA·(p_DSO−p_TSO)`. I kept the task's literal ordering and documented
   the resulting sign flip explicitly (`reading_rule` field) rather than silently reordering it.

## Remaining issues

- The `s31c` campaign gate itself has not been run (by design — the Planner launches it). The exact
  command is below.
- `_get_local_objective_components` (`shared_resources_planning.py:4461` area, used only by the
  recourse-jump-diagnostic printer, not by any pass/fail check) now has a nonzero `unclassified`
  field whenever the settlement weight is 1, since it reads `model.objective.expr` directly but
  does not enumerate the settlement among its named families. No warning consumes this field
  (checked: only `_get_local_slack_penalty_components`'s `unclassified` is asserted against, and
  that one is unaffected — it reads `model.total_slack_penalties`, not `model.objective.expr`). Left
  unmodified as informational-diagnostic-only and out of the Part 1 item list; noted for the
  Planner in case the jump-diagnostic printout is read closely during the campaign.
- Item 3 above (`p56a_oracle.py`) is unresolved and flagged, not fixed.

## Exact gate command

```
python -u p515_g_g1_g4_admm_gates.py s31c > data/SRP1/Results/P515S31C_launch.log 2>&1
```

## git status (at report time)

```
 M model_construction_helpers.py
 M network.py
 M p515_g_g1_g4_admm_gates.py
 M shared_resources_planning.py
?? data/SRP1/Results/P515S31C/
?? p515_s31c_preflight.py
?? p515_s31c_zero_solve_checks.py
```

(plus the pre-existing untracked/modified files listed in the session's initial `git status`,
unrelated to this task and not touched by it.)

## Questions for Planner

1. Confirm the sign convention for `interface_consensus_residual_per_dso` (TSO − DSO, per the task
   text) is what the campaign report should use, given `T_TSO+T_DSO` is `−1×` that quantity
   (documented in `reading_rule`) — or whether the field should be redefined DSO − TSO to read
   positively alongside `t_tso_plus_t_dso_terminal`.
2. Should `_get_local_objective_components` / `per_block_base_objectives` (`p56a_oracle.py`) be
   folded into a future settlement-consistency pass, given they are diagnostic-only today?
3. Confirm `run_admm_arm`'s existing "solves" accounting (102 for one cycle: 2 rounds ×
   (48 network SMOPFs + 3 ESSO)) is the expected count before the campaign is launched at scale.
