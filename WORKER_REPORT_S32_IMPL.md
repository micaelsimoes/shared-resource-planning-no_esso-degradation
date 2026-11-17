# Worker Report -- P5.15 Step 3.2 + 3.3(a) implementation (s32)

## Task received

Implement the Boyd (2011 Sec. 3.3.1) stopping rule (Step 3.2) and residual
balancing (Step 3.3(a)) per `PLANNER_BRIEF_2026-09-13.md` Addendum 9 sections
3.2/3.3(a) and Addendum 13, bound by the frozen specification
`data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json`; add the `s32`
gate harness arm; write and run zero-solve checks and a one-cycle preflight.
**Not** authorized to launch the s32 gate campaign.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` Addendum 9 (secs. 3.0-3.5), Addendum 13.
- `data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json` (hash verified,
  see below).
- `data/SRP1/Results/P515S32/s31c_rho_residual_extraction.json`.
- `shared_resources_planning.py`: cycle loop (~2380-2900), `get_admm_residual_metrics`,
  `_update_admm_penalties`, `_update_interface_power_flow_variables`,
  `_update_shared_energy_storage_variables`, `create_admm_variables`,
  `_prepare_transmission_objectives_for_admm`/`_prepare_distribution_objectives_for_admm`,
  `update_transmission_model_to_admm`/`update_distribution_models_to_admm`/
  `update_shared_energy_storage_model_to_admm`, `_update_tso_proximal_centres_after_solve`,
  `update_transmission_coordination_model_and_solve`,
  `update_distribution_coordination_models_and_solve_sequential`.
- `admm_parameters.py`.
- `data/SRP1/SRP1_params.json` (`admm` block).
- `p515_g_g1_g4_admm_gates.py` (`_construct_arm_planning`, `run_admm_arm`,
  `assert_s31c_capture_paths`, `write_interface_settlement_detail_s31c`,
  `write_component_levels_terminal`, the `s31c` CLI gate).
- `p515_s31c_zero_solve_checks.py`, `p515_s31c_preflight.py`,
  `p515_s31c_evaluate.py` (style/pattern precedents).

## Files modified

- `admm_parameters.py`
- `shared_resources_planning.py`
- `data/SRP1/SRP1_params.json`
- `p515_g_g1_g4_admm_gates.py`
- New: `p515_s32_zero_solve_checks.py`
- New: `p515_s32_preflight.py`
- New (results): `data/SRP1/Results/P515S32/zero_solve_checks/zero_solve_checks.json`,
  `.../manifest_sha256.json` (`synthetic_admm_snapshot.pkl` hash-recorded, not
  committed -- see below)
- New (results): `data/SRP1/Results/P515S32/preflight/*` (18 files) +
  `manifest_sha256.json`

## Verification of F1 solve order and F4 conversions (file:line, current working tree)

**F1 -- solve order.** `shared_resources_planning.py:2459` DSO solves first
(`results['dso'] = update_distribution_coordination_models_and_solve(...)`),
then the consensus/primal update at `update_flags={"update_dns": True}`
(:2470); **then** the TSO solves second (:2483,
`results['tso'] = update_transmission_coordination_model_and_solve(...)`)
against the DSO's just-updated copy, followed by
`update_flags={"update_tn": True}` (:2494) and the dual (lambda) update,
which is embedded in `_update_interface_power_flow_variables` (:6242, dual-update block :6295-6325)
and runs **only** when `update_tn=True` -- i.e. once per cycle, immediately
after the TSO solve, confirming the two-block mapping x=DSO, z=TSO, with the
DSO-copy-first / TSO-copy-second order the Advisor described. **Matches F1
exactly; no discrepancy found.**

**F4 -- lambda unit conversions.**
- V: `update_distribution_coordination_models_and_solve_sequential`
  (`shared_resources_planning.py:4975`) --
  `model[year][day].dual_vmag_req[p].set_value(dual_vmag['current'][node_id][year][day][p] / v_base)`.
- PF (P and Q): same function, `:4977` (P) / `:4978` (Q) --
  `dual_pf['current'][...]['p'][p] / s_base` / `.../'q'[p] / s_base`, where
  `s_base = distribution_network.network[year][day].baseMVA` (**the DSO's
  own** s_base, confirmed by reading the `s_base` assignment at `:4965`
  inside the same function, not the TSO's).
- ESS: same function, `:4982-4983` -- `dual_ess['current'][...]` loaded with
  **no division at all** (`ESS: as stored`).
- `get_admm_boyd_residual_metrics` reproduces the identical divisions
  (`lambda_dso_v / v_base`, `lambda_dso_pf / s_base_dso`, ESS undivided) --
  asserted textually and numerically in zero-solve check 4.
  **Matches F4 exactly; no discrepancy found.**

## Exact formulas as implemented (file:line, current working tree)

- `get_admm_boyd_residual_metrics` -- new function,
  `shared_resources_planning.py:5519-5750` (docstring at :5520-5555). Per
  channel, per coordinate:
  - V (`:5616-5626`): `r_v = (x_DSO_v - z_TSO_v) / v_base`;
    `dz_v = (z_TSO_v^k - z_TSO_v^{k-1}) / v_base`;
    `s_rho_v = rho_tso_v * dz_v`; `s_prox_v = gamma_v * dz_v`;
    `s_v = sqrt(rho_tso_v^2 + gamma_v^2) * dz_v`; `y_v = lambda_DSO_v / v_base`.
  - PF (`:5628-5644`), P and Q: identical structure, normalized by
    `interface_rating` (r, s) and `s_base_dso` (y).
  - ESS (`:5648-5675`): `r_agent = a_agent * (x_agent - z)`,
    `a_agent = 1/(2*S_agent)`; `s_agent_rho = rho_agent * a_agent * dz_ess`
    (one entry per agent in {tso, dso, esso}); PLUS one extra TSO-proximal
    entry per coordinate (`:5678-5686`, gated on
    `proximal_regularization.enabled and .tso.enabled`):
    `s_prox_ess = gamma_ess * a_tso * (x_TSO^k - x_TSO^{k-1})`.
  - Aggregation (`:5688-5718`): Euclidean norm (`sqrt(sum of squares)`) over
    all normalized entries per channel; `n = p` = the base entry count
    (nodes x years x days x periods [x p,q] [x agents]) -- the ESS
    TSO-proximal entries contribute to `||s||` but **not** to `n`, per the
    frozen spec's literal "p = n = entry count per channel" (documented
    deviation from a literal per-vector count, stated in the function
    docstring `:5551-5555`, entry-count comment at `:5692`).
  - `eps_pri = sqrt(n)*eps_abs + eps_rel*max(||x||,||z||)`,
    `eps_dual = sqrt(n)*eps_abs + eps_rel*||y||` (`:5713-5714`).
- Stopping test: `:2534` (call), `:2684` (gate) -- `boyd_all_pass = boyd_metrics['all_boyd_pass']`;
  `cycle_convergence = boyd_all_pass and local_solves_ok`. The recourse
  objective-change test and the legacy consensus/stationarity test remain
  computed and recorded (`objective_convergence`, `residual_convergence`)
  but do **not** appear in this expression -- verified by zero-solve check 8
  (source-text presence of the exact line, plus a check that
  `objective_convergence` is absent from it).
- `_update_admm_penalties` -- rewritten, `:6093-6234`. Per channel:
  `primal_ratio = boyd_metrics[group]['primal_ratio']` (= r/eps_pri),
  `dual_ratio = boyd_metrics[group]['dual_ratio']` (= s/eps_dual);
  `increase` if `primal_ratio > residual_balance_ratio * dual_ratio` (5.0);
  `decrease` if `dual_ratio > (residual_balance_ratio_pf_decrease (3.0) if
  group=='pf' else residual_balance_ratio) * primal_ratio`; **no
  `adaptation_converged` gate** (freeze clause removed, `:6162-6170`);
  `held after solver failure` kept (`allow_update=False`); factors/clamp
  unchanged (one common factor per channel, applied uniformly to TSO, every
  DSO and ESSO, `:6176-6203`). `[ADMM RHO]` (legacy) and `[ADMM RHO BOYD]`
  (new: r, s, eps_pri, eps_dual, ratios, rho before/after at `%.6e`) are
  both printed, after the scaling is applied (`:6205-6234`, `[ADMM RHO]` at :6212, `[ADMM RHO BOYD]` at :6222).
- `admm_diagnostics.append(...)` -- extended, `:2744-2861`, with every
  `boyd_<group>_<field>` key listed in the frozen spec's
  `report_per_cycle_channel`, `objective_change_ratio`, and
  `gap_proxy_G`/`gap_proxy_Q`/`gap_proxy_G_over_Q`/`gap_proxy_G_reason`.

## Gap proxy G/Q -- recorded as `None` with a reason, not approximated

`gap_proxy_G` is `None` on every cycle (`:2858-2861`). Reason: G's
definition (`Sum_blocks sigma_b*(...)`) requires `sigma_b` (the block's
`admm_objective_scale`) for **every** block including the ESSO's; the ESSO
objective is **not** divided by an effective scale (F5, confirmed:
`update_shared_energy_storage_model_to_admm`,
`shared_resources_planning.py:4206`, `obj = copy(models[node_id].objective.expr)`,
no `/effective_scale` term, no `admm_objective_scale` Param on the ESSO
model). Summing the TSO/DSO blocks alone would silently omit the ESSO block,
which the evidence rules ("never approximate") forbid. This is a **documented
deviation from computing G**, not a silent omission -- `gap_proxy_G_reason`
states it on every cycle. `gap_proxy_Q` (= `gross_operational_cost`) is
always populated.

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` on all four modified/new
   production/harness files -- syntax OK.
2. `python -c "import p515_g_g1_g4_admm_gates as G; ..."` -- module imports
   cleanly; `run_admm_arm`/`_construct_arm_planning` signatures confirmed.
3. `assert_s32_capture_paths(planning)` run standalone on a freshly built,
   never-solved planning object under a zero-permitted `SolveProfileGuard` --
   **passed** (0 solves).
4. `python p515_s32_zero_solve_checks.py` -- zero-solve verification, 11
   checks, `SolveProfileGuard(permitted=())`.
5. `python p515_s32_preflight.py` -- one real ADMM cycle through the s32 arm
   machinery (`num_max_iters_override=1`, `apply_rho=False`,
   `full_diagnostics_in_rows=True`).

## Results

### Zero-solve checks (`p515_s32_zero_solve_checks.py`)

All 11 pass; `solve_profile_guard.counts = {permitted_solve: 0, permitted_exec: 0,
blocked_solve: 0, blocked_exec: 0}`, `verify_failures = []`.

| # | check | result |
|---|---|---|
| 1 | legacy metrics bit-identical (new vs. `git show HEAD` pre-change module, on a constructed-and-pickled synthetic ADMM snapshot -- see deviation note) | **pass** -- primal/dual dicts identical to the last float |
| 2 | Boyd ESS primal entries == legacy entries (isolated coordinate); V/PF dual differs from legacy only by the DSO-change term | **pass** -- V: `tso_term_v=0.02`, `dso_term_v=0.10`, `legacy_dual_v_max=0.10`, `boyd_s_rho_part_v=0.02` (legacy picks the larger DSO term, Boyd excludes it exactly); PF: `tso_term_pf=0.015`, `dso_term_pf=0.05`, same pattern; ESS: `boyd_ess_r == sqrt(sum of the 3 legacy per-agent terms squared)` |
| 3 | lambda_TSO = -lambda_DSO, model units, machine precision (no non-trivial-lambda fixture found -> synthetic, exercised through the real `_update_interface_power_flow_variables`) | **pass** -- `lambda_v_tso=27.6`, `lambda_v_dso=-27.6` exactly; PF P/Q likewise; model-units `y` antisymmetric too |
| 4 | ||y|| unit conversions vs. model-loading expressions (source-text + numeric) | **pass** |
| 5 | `_update_admm_penalties` touches only rho; consensus/dual dicts bit-identical before/after | **pass** (`consensus_vars_unchanged=True`, `dual_vars_unchanged=True`) |
| 6 | AL objective / proximal-centre functions structurally unchanged (exact source-text equality, pre- vs. post-change) | **pass** -- all 6 functions byte-identical (sha256 matches) |
| 7 | balancing unit test: increase / decrease / dead band / no-freeze / failure-hold | **pass**, all 5 sub-cases |
| 8 | stop-rule unit test: objective test not gating; each of 6 Boyd tests blocks individually | **pass** |
| 9 | other case params (CS1) still load, `boyd_eps_source='default'`; SRP1 `='case_file'` | **pass** |
| 10 | every `p515_s31c_zero_solve_checks.py` fixture still unpickles | **pass** (3/3) |
| 11 | hierarchical/uncoordinated paths contain no hunks | **pass** -- both functions byte-identical pre/post; `git diff` on `shared_resources_planning.py` has 8 hunks total, all inside `_run_operational_planning`'s distributed cycle loop, the new function, and `_update_admm_penalties` |

**Deviation noted for check 1:** no pre-existing full multi-agent
(planning_problem + tso_model + dso_models + esso_model + consensus_vars)
pickled fixture was found in the repository for this comparison. One was
constructed via the production zero-solve pipeline (same
monkeypatched-`.optimize` technique as `p515_s31c_zero_solve_checks.py`),
perturbed at one synthetic coordinate, and pickled to
`data/SRP1/Results/P515S32/zero_solve_checks/synthetic_admm_snapshot.pkl`
before being reloaded and compared -- satisfying "on a pickled snapshot"
literally, though not from a pre-existing artifact.

### Preflight (`p515_s32_preflight.py`, one real ADMM cycle, C\* candidate)

`cycles_run=1`, `local_solve_failures=0`, `solve_profile.identity_holds=True`
(102 solves), `network_failures.n_blocks=0`. Every required
`report_per_cycle_channel` field populated and finite; `gap_proxy_G` /
`gap_proxy_G_over_Q` are `None` with the stated reason (as designed);
`recourse = gross_operational_cost = 968,494,925.09`.

| channel | r | s | eps_pri | eps_dual | primal_ratio | dual_ratio | proximal_share | action | rho before -> after |
|---|---|---|---|---|---|---|---|---|---|
| V | 1.0117e-01 | 5.3027e-02 | 3.2282e-03 | 2.9769e-04 | 31.34 | 178.13 | **0.7071** | decreased | 1.0 -> 0.6667 |
| PF | 3.6407e+00 | 7.1274e+00 | 1.4325e-03 | 9.2154e-04 | 2541.61 | 7734.20 | **0.7071** | decreased | 1.0 -> 0.6667 |
| ESS | 5.2877e-03 | 5.8753e-03 | 7.3001e-04 | 7.2053e-04 | 7.24 | 8.15 | **0.9497** | held | 1.0 -> 1.0 |

`boyd_all_pass=False`, `boyd_stop=False`, `stopped_by='cap'` (1/1 cycles, as
expected for a one-cycle smoke test). The 3.3(a) balancing recheck
(independently recomputed from the logged ratios, penalty_update thresholds
and clamp) matches production exactly on all three channels
(`balancing_recheck_all_match=True`).

## Validation

- Code executes correctly: confirmed (zero-solve checks + one real-solve
  preflight cycle, both exit 0).
- Requested diagnostic works: confirmed -- every per-cycle field the frozen
  spec's `report_per_cycle_channel` requires is populated on a real cycle;
  `boyd_terminal.json`'s `report_terminal` fields (`stopped_by`, binding test
  per channel, rho trajectory, system-cost-vs-s31c with terminal steps,
  network failures by tier, cancellation residual, per-DSO
  settlement/flexibility, D rows) are all present and populated.
- Underlying numerical problem (ADMM convergence under 3.2/3.3(a)) is
  **not** assessed here -- that requires the full 150-cycle campaign, which
  this task explicitly does not launch.
- `git diff` hunk inspection (shared_resources_planning.py: 8 hunks, all
  inside the distributed cycle loop / the new function / `_update_admm_penalties`;
  admm_parameters.py: 2 hunks, both inside `__init__`/`_read_parameters_from_file`;
  SRP1_params.json: 1 line added) confirms no unintended modification.
- Existing harness arms (g1, g2, g3_*, g4b, s31, s31c, ablation_*) call
  `run_admm_arm`/`_construct_arm_planning` with no new arguments, so
  `apply_rho=True`/`full_diagnostics_in_rows=False` defaults preserve their
  prior behavior exactly.

## Unexpected findings

- **Proximal share exceeds 50% on every channel at cycle 1** -- flagged per
  the task's explicit instruction. V and PF sit at exactly `1/sqrt(2) ≈
  0.7071` (mathematically forced: at cycle 1, `rho == gamma == 1.0` for both,
  so `s = sqrt(2)*rho*dz`, split evenly in quadrature between the rho part
  and the proximal part -- this ratio will only change once rho itself moves
  away from `gamma=1.0`). ESS is higher still, `0.9497`.
- **The dead-band decision at cycle 1 (V and PF) is `decreased`, driven by
  the proximal-inflated dual residual**: `dual_ratio` (178 for V, 7734 for
  PF) is roughly 5-6x `primal_ratio` (31, 2542), so both channels' rho are
  cut by the decrease factor on the very first cycle. This is exactly the
  frozen spec's own flagged concern ("proximal term may floor s and drive
  rho_pf down") materializing immediately, not merely a theoretical risk --
  worth the Planner's attention before/alongside interpreting the 150-cycle
  campaign's rho trajectory.
- `_compute_common_admm_objective_scale` raises on a never-solved (build
  -default) model set (every block's raw objective is exactly 0.0), which is
  unrelated to this stage's production code but required a workaround
  (`objective_scale = 1.0`) inside the zero-solve check's own model-building
  helper; noted in code comments there, not a production change.

## Remaining issues

- The s32 gate campaign itself (150 cycles) has **not** been run; per the
  task, only the Planner launches it, with the exact command already wired
  into the `s32` CLI dispatch (`_require_fresh_output_root(OUT_S32)`,
  `boyd_eps_source`/spec-hash/initial-rho asserted before any solve).
- `system_cost_vs_s31c`'s `matched_cycles`/`terminal` comparison in
  `boyd_terminal.json` is only meaningful once the s32 campaign has run past
  cycle 1; on the preflight it correctly reports `s32_terminal_step_objective_change_abs=None`
  (no previous-cycle recourse yet on a 1-cycle run) and
  `error_bar_sum_of_terminal_steps=None` accordingly -- not a bug, the
  campaign result will populate it properly.

## Questions for Planner

- None blocking. The proximal-share and decrease-on-cycle-1 findings above
  are handed up as evidence for the 150-cycle campaign's interpretation, per
  "Distinguish observations, hypotheses, evidence, and conclusions" --  no
  redesign was made in response to them.

## Commit hashes

Recorded after the two commits below are created (see final reply).
