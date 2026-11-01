# Worker Report — P5.15 Addendum 16 item 2, PHASE 1 (price-taker initialization)

## Task received

Bounded implementation task, PHASE 1 ONLY of Addendum 16 item 2: build a production
price-taker LP module and an ADMM-consensus initialization wrapper for the shared ESS, plus
zero-solve checks, per frozen spec v6
(`data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json`, commit `a993b088`). No
Pyomo/IPOPT solves of any kind; the concurrent `s35ref` reference run (PID 21426, harness
lock `.p515_g_gate.lock`) was not to be disturbed, and `data/SRP1/SRP1_params.json` /
`p515_g_g1_g4_admm_gates.py` were not to be modified. Z3/Z4, the case-file flag, the
`s35pt` harness arm and the IPOPT half of Z7 are explicitly deferred to PHASE 2.

## Files inspected

- `data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json` (binding spec, read in full).
- `P5_15_Z2_FLOOR_SLACK_NOTE.md`.
- `p515_s34_efc_benchmark.py` + `data/SRP1/Results/P515S34/EFC_benchmark/efc_benchmark_results.json`.
- `p515_s35_z2_floor_slackness.py` + `data/SRP1/Results/P515S35/Z2/z2_floor_slackness_results.json`.
- `WORKER_REPORT_S35_Z2.md`.
- `shared_resources_planning.py`: `_run_operational_planning` setup (≈2309–2470, post-edit
  ≈2309–2565), `_initialize_shared_ess_consensus` (≈4057), `update_transmission_model_to_admm`
  (proximal centre Params/assignment, ≈4094–4150 pre-edit), `create_admm_variables` (≈3894),
  `create_shared_energy_storage_model` (≈3679), `_update_shared_energy_storage_variables` /
  the ESS z/λ update (≈6825–7013 pre-edit), `_rebuild_candidate_total_capacities`,
  `_get_initial_candidate_solution`, `_update_model_with_candidate_solution`.
- `shared_energy_storage_data.py`: `_build_subproblem` (≈397–720: variable/constraint
  definitions for `es_s/e_investment_fixed`, `es_s/e_rated_per_unit`, `es_s/e_available_per_unit`,
  `es_avg_ch_dch_per_unit`, `es_D_per_unit`, `es_soh_per_unit_cumul`, the floor row), `get_candidate_solution`,
  `update_data_with_candidate_solution`, `_update_model_with_candidate_solution`,
  `get_updated_capacities`/`get_available_capacities`.
- `model_construction_helpers.py` (`sess_active_sum_limit_rule`, `sess_soc_lower/upper_limit`,
  `sess_soc_rule`, `sess_soc_final_rule`, `sess_pnet_rule`, `period_duration_hours`,
  `configure_shared_ess_operational_state`).
- `shared_energy_storage.py`, `shared_energy_storage_parameters.py` (ageing constant
  provenance, `cl_eff`/`phi_cal` attributes).
- `network.py` (`prob_market_scenarios`/`prob_operation_scenarios`, `get_shared_energy_storage_idx`).
- `network_data.py` (`NetworkData` structure, `.network[year][day]`).
- `definitions.py` (confirmed no imports — the only production module `shared_ess_price_taker.py`
  imports from).
- `admm_parameters.py` (existing optional-key pattern: `shared_ess_reference_rating_mva`,
  `esso_al_scale`, used as the template for the new key).
- `p513_solve_profile_guard.py` (guard API).
- `p56a_oracle.py` (`load_baseline`, read-only, zero-solve on import).

## Files modified

- `shared_resources_planning.py`
- `admm_parameters.py`

## Files created

- `shared_ess_price_taker.py` (new production module)
- `p515_s35pt_phase1_checks.py` (new zero-solve check harness)
- `data/SRP1/Results/P515S35/pt_phase1_checks/phase1_checks_results.json`
- `data/SRP1/Results/P515S35/pt_phase1_checks/sha256_manifest.json`
- this report

## Changes made

### 1. `shared_ess_price_taker.py` (new, no Pyomo import)

`solve_price_taker_schedule(planning_problem, candidate_investment, node_ids=None,
efficiency_on=True, wear_on=True, outer_iterations=40, damping=0.5,
convergence_rel_tol=1e-9)`. Per active node, one joint LP across (year, day, period) with
scipy `linprog` (HiGHS), day-weighted objective `max Σ (num_days[d]/365) · π · (pdch−pch)`,
rows [1]–[5] exactly reproducing `shared_es_pch/pdch` limits, `sess_active_sum_limit_rule`,
`sess_soc_rule`, `sess_soc_lower/upper_limit`, `sess_soc_final_rule` (file:line citations in
the module docstring), and, when `wear_on=True`, the exact linear SoH floor row
`Σ_{y≤Y} D_y ≤ −ln(soh_min) + (Σ_{y≤Y} n_y)·ln(phi_cal)` with `D_y` the production throughput
expression (`shared_energy_storage_data.py:652-655`). Every physical parameter (`eff_ch`,
`eff_dch`, `cl_eff`, `phi_cal`, `soh_min`, `t_cal`, day weights, year widths, market prices,
candidate `s`/`e`) is read from `planning_problem.shared_ess_data` /
`candidate_investment[node_id][year]`, never hard-coded; the four SoC-band/init-SoC constants
come from `definitions.py`.

**Guards** (`NotImplementedError`, checked before any LP call): more than one
market/operation scenario in any network/year/day (`prob_market_scenarios`/
`prob_operation_scenarios` on the TSO and every DSO `network[year][day]`); more than one
active investment cohort at a node; a node's active cohort's calendar window not covering
every modelled year (reproduces the production window formula,
`shared_energy_storage_data.py:636-637`); market prices differing between any network's own
`cost_energy_p` and the shared-ESS reference copy.

**Capacity wear**: `outer_iterations=40` damped (`damping=0.5`) fixed-point iterations —
always run in full, no early exit; convergence (`max_y |E_new−E_guess|/E_nameplate <
1e-9`) is checked AFTER the fixed budget and a `RuntimeError` is raised if not met (never
silently accepted). Empirically (see Results) the fixed point reaches `rel_change ≈ 6e-15`
well inside the 40-iteration budget for the C* candidate.

**A numerical finding during implementation, corrected before commit**: at
`efficiency_on=False` (eff_ch=eff_dch=1.0), simultaneous same-period charge/discharge is a
genuine zero-cost degenerate LP direction (buying and selling the same MWh at the same price
nets to zero, and cancels in the SoC recursion too). Solving the WHOLE multi-year joint LP
in one call let HiGHS select a different, equally-optimal vertex from that degenerate face
than the small per-day LPs `p515_s34_efc_benchmark.py`/`p515_s35_z2_floor_slackness.py` used,
giving a spurious 0.30 max abs diff against the committed `EFC*` cells on first attempt. Fix
(documented in `_solve_node`'s comment): when `wear_on=False` (no floor rows couple years/
days), the module solves one INDEPENDENT LP per (year, day) instead of one combined LP —
mathematically identical (fully separable objective/constraints) but numerically reproduces
the committed benchmark to `0.0` max abs diff. `wear_on=True` is unaffected (real
efficiencies make simultaneous charge/discharge strictly loss-making, so no such degeneracy
exists there — confirmed by the diff already being effectively 0 on the first attempt, see
Results).

The LP call point is a single module-level function `_solve_lp` incrementing
`_LP_CALL_COUNTER['count']`, read via `get_lp_call_count()`/`reset_lp_call_count()`.

### 2. `admm_parameters.py`

New optional key `admm.shared_ess_initialization` (`'standalone'` default/absent,
`'price_taker'`), validated against exactly those two values (`ValueError` otherwise),
with `shared_ess_initialization_source` recording `'default'` or `'case_file'`. Added
following the exact pattern of the existing `shared_ess_reference_rating_mva` /
`esso_al_scale` optional keys (same file, same function `_read_parameters_from_file`).

### 3. `shared_resources_planning.py`

- New import: `import shared_ess_price_taker` (after `shared_energy_storage_data`, before
  `model_construction_helpers`).
- `_run_operational_planning`: reads `shared_ess_initialization_mode` once, right after the
  only point `initial_state` can still be reset to `None` (the `initialization_failed`
  branch); raises `ValueError` if the mode is `'price_taker'` and `initial_state is not
  None` (continuation) — spec v6 `initialization.scope`.
- Call site: immediately after `_initialize_shared_ess_consensus(planning_problem,
  consensus_vars)` (line ≈4172 pre-numbering shift; see diff), inside the `if initial_state
  is None:` branch, before `update_interface_power_flow_variables` and well before
  `sess_available_capacities = shared_ess_data.get_updated_capacities(esso_model)`. Calls
  `_initialize_shared_ess_from_price_taker(...)` only when the mode is `'price_taker'`.
- New function `_initialize_shared_ess_from_price_taker(planning_problem,
  candidate_solution, tso_model, esso_model, consensus_vars)`, implementing spec v6
  `initialization.what_is_set` items 1–5 exactly:
  1. `z` current/prev `p` = LP `p` (MW, load convention).
  2. `q` = existing `z_q` (0.0 pre-call) projected onto `sqrt(s_avail² − p²)`; clipped-cell
     count returned.
  3. The three agent copies (tso/dso/esso), current/prev, set = `z`.
  4. TSO proximal centres `prox_ess_p_prev`/`prox_ess_q_prev` = `z/s_base` via `.set_value()`
     (never `.fix()`/`.unfix()`), for every (year, day, period).
  5. ESSO state (`es_soh_per_unit_cumul` when unfixed, `es_e_available_per_unit`, `es_pnet`,
     `es_pch_per_unit`/`es_pdch_per_unit` for the active cohort row) set via `.set_value()`
     before the capacity-publishing step.
  Never touches network primal values (`shared_es_pch`/`pdch`/`pnet` on the TSO/DSO models)
  or `dual_vars` — the ESS duals stay exactly zero because this function never writes to
  `dual_vars` and `create_admm_variables` already zero-initializes them.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -c "import ast; ast.parse(...)"   # syntax check, all 4 files
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s35pt_phase1_checks.py
```
One run of the checks harness (output directory did not exist beforehand; the script refuses
otherwise). `SolveProfileGuard([])` armed for the whole run; `guard.verify(0)` passed.

## Results

All seven Phase-1 checks passed (`data/SRP1/Results/P515S35/pt_phase1_checks/phase1_checks_results.json`):

- **Solve profile**: `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0,
  'blocked_exec': 0}` — zero Pyomo/IPOPT solves anywhere in the run.
- **Z1** (efficiency toggled, wear OFF, vs committed `EFC*` benchmark, 36 cells × 2 variants):
  `max_abs_diff_variant1_efficiency_on_wear_off = 0.0`,
  `max_abs_diff_variant3_efficiency_off_wear_off = 0.0`. LP calls: 72 (36 + 36, one LP per
  (node, year, day) per variant — see the degeneracy fix above for why this granularity was
  used instead of one combined LP).
- **Z2p** (efficiency + wear ON, vs committed Z2): `max_abs_efc_diff_vs_committed_Z2 =
  5.77e-15`, `max_abs_terminal_soh_diff_vs_committed_Z2 = 6.66e-16`, floor slack at every
  node, floor multiplier exactly `0.0` at every node/year, all three nodes converged
  (`final_rel_change ≈ 6.19e-15`, well inside `outer_iterations=40`). Per-year EFC/day
  (node 5, identical at 7/9): 2025→1.191777…, 2030→1.188810…, 2035→0.964719… (committed:
  1.1918/1.1888/0.9647); terminal SoH 0.589209… (committed 0.589209). LP calls: 120 (3 nodes
  × 40 outer iterations × 1 joint LP).
- **Z5**: 36 (node, year, day) cells checked (3 nodes × 3 years × 4 days); 0 violations —
  charge-weighted price ≤ day mean, discharge-weighted price ≥ day mean, `|p| ≤ s_avail`,
  SoC inside `[E_avail·0.10, E_avail·0.90]` for every cell.
- **Z6**: all four synthetic-input guards fired `NotImplementedError` as expected
  (`scenario_guard`, `cohort_count_guard`, `prices_differ_guard`, `cohort_window_guard`), each
  with **0** `linprog` calls before raising.
- **Z7 (LP part)**: LP call counts read from `shared_ess_price_taker.get_lp_call_count()`
  matched the counts declared above exactly (72 for Z1, 120 for Z2p); all four Z6 guard calls
  made exactly 0 LP calls before raising.
- **Injection unit test**: passed with 0 problems, `clipped_q_cells = 0` (z_q starts at 0.0,
  always inside the converter circle for this candidate). Route used: **synthetic** — see
  below.
- **Flag-absent identity**: loading `data/SRP1/SRP1_params.json`'s and `data/CS1/CS1_params.json`'s
  `admm` blocks through the modified `ADMMParameters.read_parameters_from_file` yields
  `shared_ess_initialization == 'standalone'`, `source == 'default'` for both, with
  `num_max_iters` and `rho['v']` parsed identically to the raw JSON (spot-checking that
  nothing else in parsing changed).

## Validation

- `ast.parse` succeeded on all four changed/new production files before running anything.
- `git diff` on `shared_resources_planning.py` and `admm_parameters.py` is additive-only —
  no existing line was altered, only new lines inserted (verified by reading the full diff,
  reproduced in this report's "Changes made" section and available via `git diff --cached`
  before commit).
- The checks harness independently re-derives the committed EFC*/Z2 reference numbers from
  the committed JSON artifacts (`data/SRP1/Results/P515S34/EFC_benchmark/efc_benchmark_results.json`,
  `data/SRP1/Results/P515S35/Z2/z2_floor_slackness_results.json`) rather than re-typing
  literals, so Z2p's near-zero diffs are a genuine cross-check, not a tautology.
- Confirmed the `s35ref` reference process (PID 21426) and an active `ipopt` child process
  were still running after the checks harness completed; `.p515_g_gate.lock` and
  `data/SRP1/Results/P515S35_REF_run/` untouched (`git status --short` shows no changes to
  either); `data/SRP1/SRP1_params.json` and `p515_g_g1_g4_admm_gates.py` show no modification
  (`git status --short`, mtimes unchanged from before this task).

### Injection unit-test route

**Synthetic**, as anticipated by the task's fallback instruction. Building REAL unsolved
Pyomo ADMM models is not possible without a solve in this codebase:
`create_shared_energy_storage_model` (`shared_resources_planning.py:3679`) calls
`shared_ess_data.optimize(esso_model)` unconditionally as part of model construction (the
"initial shared ESS values" step), and the TSO/DSO model constructors solve a standalone
SMOPF similarly. The test therefore uses: (a) the REAL `create_admm_variables` and
`_initialize_shared_ess_consensus` (both pure Python, no Pyomo model, no solve) against the
REAL `planning_problem`/candidate; (b) a minimal synthetic `tso_model[year][day]` object
exposing only `prox_ess_p_prev`/`prox_ess_q_prev` as settable fake-Param dicts, and a
minimal synthetic `esso_model[node_id]` object exposing `years`/`days`/`periods` (plain
ranges, matching production's own `range(...)` convention) and
`es_soh_per_unit_cumul`/`es_e_available_per_unit`/`es_pnet`/`es_pch_per_unit`/
`es_pdch_per_unit` as settable fake-Var dicts (with a `.fixed` attribute defaulting to
`False`); (c) the REAL `_initialize_shared_ess_from_price_taker` applied to these. This
exercises the wrapper's actual logic and every assertion the task lists (z=LP p, copies=z,
proximal centres=z/s_base, q inside the circle, ESS duals untouched, V/PF consensus and
duals bit-identical) without any Pyomo/IPOPT solve.

## Unexpected findings

- The degenerate-LP vertex-selection issue described above (efficiency_on=False creates a
  genuine zero-cost charge/discharge direction) is a property of `scipy.optimize.linprog`
  (HiGHS) applied to a large combined LP versus many small ones, not a defect in the row
  algebra; it only affects the (economically unrealistic) `eff=1` diagnostic configuration,
  never the production-realistic `efficiency_on=True` path. Flagging it here since it was
  not anticipated by the spec text and is worth the Planner/Advisor knowing about if the LP
  module is ever extended to solve a single combined multi-node/multi-year LP at `eff=1`.
- `outer_iterations=40`/`damping=0.5` are generous relative to what C* needed
  (`final_rel_change ≈ 6e-15`, converged in far fewer than 40 iterations); left at 40/0.5 as
  a declared, fixed, conservative budget per spec's "FIXED, declared number K" instruction,
  not tuned down to the minimum that happens to work for this one candidate.

## Remaining issues

- PHASE 2 (deferred, per task): the SRP1 case-file flag, the `s35pt` harness arm, Z3/Z4 (need
  the standalone-solve comparison), the IPOPT-count half of Z7, and the two-cycle preflight.
  None of these were started.
- The injection unit test's synthetic route does not exercise the REAL
  `configure_shared_ess_operational_state`/`create_shared_energy_storage_model` code paths
  end-to-end; Phase 2's two-cycle preflight (explicitly deferred) is the point at which the
  wrapper's effect on an actual solved ADMM cycle gets checked.

## Deviations from spec v6

None identified. One clarification for the record: spec v6's `initialization.lp.formulation`
describes the LP as "coupled across (year, day)" without qualifying the `wear_on=False`
case; PHASE 1 implements `wear_on=False` as separable independent per-(year,day) LPs (see
"Changes made" §1) since no floor row couples years/days in that mode, which is
mathematically identical to one combined LP but was needed to reproduce the committed `Z1`
benchmark exactly rather than land on an alternate (still-optimal) LP vertex.

## Questions for Planner

- None blocking. Ready for Planner/Advisor review before Phase 2 is authorized.
