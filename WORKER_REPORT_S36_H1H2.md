# Worker Report — P5.15 Addendum 17 H1/H2 dual-identifiability diagnostic

## Task received

Bounded ZERO-SOLVE diagnostic (Planner, 2026-09-17): decompose run 1's
reconstructed per-agent shared-ESS consensus duals lambda_agent(c), c=1..477,
at the terminal (cycle 477) schedule, into a component orthogonal to the span
of the storage constraints active there (I, "identifiable" -- price pins this
down) and a component in that span (J, "in-span" -- free, price-independent).
Determine whether the slow build-up documented in run 1's per-agent dual norms
(TSO 8.075->6.268, DSO 4.037->5.341, ESSO 4.039e-3->2.152e-3 from cycle 1 to
477, total norm flat over cycles 346-477) is the identifiable component (H1)
or the in-span split (H2), per a pre-registered verdict rule. No Pyomo/IPOPT
solves; no production, case-file or harness edits. `SolveProfileGuard([], ...)`
armed for the whole script.

## Files inspected

- `p515_s36_a17_parta_dual_reconstruction.py` and its output
  `data/SRP1/Results/P515S36/A17_partA_dual_reconstruction/` (commit `b63e4d94`)
  -- the exact ADMM dual-update rule replayed here, reused via `import`.
- `WORKER_REPORT_S36_A17_CAPTURE.md` -- PART A's documented cross-validation and
  the "dual_p_req is lagged one cycle" timing subtlety (not relevant here, since
  this script never reads `esso_models_baseline.pkl`'s `dual_p_req`/`dual_q_req`
  Params, only its `es_e_available_per_unit`/`es_s_available_per_unit` Vars,
  which ARE solved, not lagged, at the pickled cycle).
- `data/SRP1/Results/P515S35_REF_run/`: `g_baseline.json`,
  `ess_entry_stride_baseline.jsonl` (stride 1, cycles 1-477, confirmed complete),
  `esso_capture/baseline/node{5,7,9}_cycle477.jsonl` (checked field list: `pch`,
  `pdch`, `pnet`, `s_max`, slacks, `zL_pch`/`zL_pdch`/`zU_pch`/`zU_pdch`, a
  `duals` list over `energy_storage_limits`/`energy_storage_operation_agg`/
  `energy_storage_cohort_pnet_share_h3`/`energy_storage_capacity_degradation` --
  **no SoC or energy field present**), `esso_models_baseline.pkl`.
- `data/SRP1/Results/P515S35_PT_run/`: `ess_entry_stride_baseline.jsonl` (stride
  5, 30 rows, cycles 1-146) and `g_baseline.json` (150 cycles) -- gate 3.
- `shared_energy_storage_data.py`: `_build_subproblem` (~397-760, confirms the
  ESSO's own subproblem has NO SoC/energy-state variable at all -- only cohort
  power/degradation rows: `energy_storage_limits`, `energy_storage_operation_agg`,
  `energy_storage_capacity_degradation`, the charging/discharging throughput
  row), `get_available_capacities` (256-262, `e_available = sum_y_inv
  es_e_available_per_unit[y_inv, year]`).
- `model_construction_helpers.py`: `sess_soc_rule` (893-912), `sess_soc_final_rule`
  (915-921), `sess_soc_lower_limit`/`sess_soc_upper_limit` (840-847),
  `sess_active_sum_limit_rule` (827-837), `period_duration_hours` (797-808) --
  the TSO/DSO local SoC recursion, day-balance row and bound rows this script
  reconstructs.
- `shared_energy_storage.py:14-15` (`eff_ch=0.97`, `eff_dch=0.96` class defaults).
- `definitions.py:38-40,94` (`ENERGY_STORAGE_{MIN,MAX}_ENERGY_STORED`,
  `ENERGY_STORAGE_RELATIVE_INIT_SOC`, `HOURS_PER_REPRESENTATIVE_DAY`).
- `shared_resources_planning.py:6962-7150` (`_update_shared_energy_storage_variables`,
  the consensus/dual update, including the unit conversion at 7004/7021/6986
  confirming x_tso/x_dso/x_esso/z all live in the SAME physical MW/MWh space,
  not each agent's own per-unit system) and `:5138-5147` etc. (`shared_es_e_rated_fixed`
  set from `sess_estimated_capacity[year]['e_available']`, confirming the
  physical-unit E_avail this script reads from the pickle is exactly what the
  TSO/DSO local SoC bound/day-balance rows use).
- `p515_s34_efc_benchmark.py` (precedent for the zero-solve efficiency-uniformity
  read via `p56a_oracle.fresh_planning`, and for `E_INV=3.875 MWh`,
  `S_INV=0.96875 MVA` matching this run's node-5 nameplate values).
- `p56a_oracle.py` (`fresh_planning`, isolated per-`eval_id` log dir).
- `p513_solve_profile_guard.py` (guard mechanism; confirmed nested
  `install()`/`uninstall()` with the same empty `permitted` list composes safely
  if ever called, though this script uses a single top-level guard only).

## Files modified / created

- Created `p515_s36_h1h2_dual_identifiability.py`.
- Created `data/SRP1/Results/P515S36/H1H2_identifiability/`:
  `h1h2_identifiability_results.json`, `h1h2_per_cycle_series.json`,
  `sha256_manifest.json`.
- Created this report.
- No production, case-file, or other-arm harness code modified. No file under
  `P515S35_REF_run`/`P515S35_PT_run` (or any other previously-committed result
  directory) was written to. No file the concurrent cycle-0-LMP worker owns
  (`data/SRP1/Results/P515S36/cycle0_lmp/` and related) was touched (verified:
  `git status --short` before writing shows only my two new paths).

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_h1h2_dual_identifiability.py
```
Exit 0. Before running: verified no `p515_g_g1_g4_admm_gates.py`/`p515_s36_*`
process running (`ps aux`) and `.p515_g_gate.lock` absent.

## Results

**Preconditions** (both verified from source/data, not assumed, reusing PART A's
own verified helpers via `import p515_s36_a17_parta_dual_reconstruction as parta`):
zero-initial-dual: `True`. No-skip: `True` (zero cycles skip the dual update in
run 1).

**Replay cross-validation** vs `g_baseline.json`'s `boyd_ess_norm_y_{tso,dso,esso}`
and `boyd_ess_norm_y` (all 477 cycles): max relative error 2.7e-15 (dso) to
1.5e-14 (esso), total 5.0e-15 -- all well under the 1e-9 pass threshold. This
independently confirms the replay in this script reproduces PART A's own
already-validated reconstruction.

**Terminal-schedule cross-check**: ESSO capture's own `pnet = pch - pdch` vs the
terminal stride file's `z`, max abs diff over all 36 P-channel blocks x 24
periods = 3.53e-4 (small relative to typical |pnet| ~0.25 -- the two have not
fully converged bit-for-bit, as expected of an ADMM consensus schedule that is
still drifting, consistent with the "neither settles" finding below).

**E_avail (post-SoH, MWh)**, read from the terminal ESSO pickle's
`es_e_available_per_unit` (a solved Var, not lagged): node 5/2025 = 3.2783,
node 5/2035 = 2.5552 (vs nameplate `E_INV=3.875`); nodes 7 and 9 within 0.03%
of node 5 at each year (SoH tracks are near-identical across nodes).

**dim S** (P-channel, 36 blocks = 3 nodes x 3 years x 4 days), both span
variants:

| variant | min | max | mean | blocks with dim >= 20 |
|---|---|---|---|---|
| with_idle | 5 | 14 | 8.19 | 0 / 36 |
| without_idle | 5 | 14 | 8.00 | 0 / 36 |

Well below the vacuous threshold (20 of 24, or >18/36 blocks) in every block --
the decomposition is NOT vacuous under either span variant.

**Settling** (window 50 cycles, threshold 1% relative change, both span
variants give materially the same numbers; `without_idle` shown):

| series | settling cycle | rel. change, last 50 cycles (427->477) |
|---|---|---|
| ‖I‖ total | **not settled by 477** | 2.00% |
| ‖J‖ total | **not settled by 477** | 2.64% |
| ‖I‖ TSO | not settled | 1.82% |
| ‖J‖ TSO | not settled | 2.98% |
| ‖I‖ DSO | not settled | 5.29% |
| ‖J‖ DSO | not settled | 1.95% |
| ‖I‖ ESSO | not settled | 11.10% |
| ‖J‖ ESSO | not settled | 9.46% |

Neither ‖I‖ nor ‖J‖ has settled by cycle 477 under the pre-registered 1%/50-cycle
rule, at the total level or for any individual agent. The trajectory is
monotone and slow: ‖I‖_total falls from 0.006657 (cycle 300) to 0.006134
(cycle 477); ‖J‖_total rises from 0.005399 to 0.005897 over the same span --
i.e. the two are still converging toward each other, not yet at a plateau.

**Identifiable fraction** ‖I‖²/‖λ_P‖² (total, P channel only, `without_idle`
variant -- `with_idle` agrees to <0.1 pp at every checkpoint):

| cycle | 1 | 50 | 150 | 300 | 477 |
|---|---|---|---|---|---|
| ‖I‖²/‖λ_P‖² | 0.685 | 0.743 | 0.694 | 0.603 | 0.520 |

The identifiable fraction is falling over the run (0.685 -> 0.520), i.e. the
in-span (price-free) component J is growing as a SHARE of the total P-channel
dual, even though in absolute terms it is J that is still slowly rising while
I is still slowly falling (see settling table above) -- consistent with, but
not proof of, an H2-leaning trend that has not yet reached a verdict-qualifying
verdict under the pre-registered rule (see below).

**Q dual share** (Q has no LP price; reported as a fraction of that agent's
total dual norm, not decomposed against a span): total_q_share_sq at cycle 1 =
4.47e-4 (i.e. Q carries <0.05% of squared dual mass at cycle 1); by cycle 150
it is 4.79e-4 (ESSO's own Q share individually rises to 0.41% at cycle 150,
`q_share_sq=0.004126`, notably higher than TSO's 0.062% and DSO's 0.017% at
the same cycle) -- Q remains a small fraction of the total dual mass throughout,
for every agent, at every checkpoint cycle inspected (1/50/150/300/477; full
values in the committed JSON).

**Verdict** (pre-registered rule, both span variants agree):

```
INCONCLUSIVE -- neither vacuous-span (dim S max 14/24, well under the 20/24
threshold) nor "both settle together" in the literal sense, but "both series
still drifting above the 1%/50-cycle threshold at cycle 477" -- which this
script's docstring documents as falling under the same INCONCLUSIVE bucket as
"both settle together" (see verdict-rule docstring section: "both None/
not-settled" is treated as the same qualitative outcome the pre-registered rule
calls "settle together", since neither H1's nor H2's specific trigger --
"the other has settled while this one has not" -- holds).
```

This modelling extension of the pre-registered rule (not explicitly anticipated
by the task text, which described the rule assuming at least one series would
settle) is stated explicitly in the script's docstring and repeated here rather
than silently forcing a verdict.

## Gate 3

**Not computable**, per the task's own fallback instruction. Evidence:
`data/SRP1/Results/P515S35_PT_run/ess_entry_stride_baseline.jsonl` has 30 rows
at declared stride 5, covering cycles 1, 6, 11, ..., 146, while
`g_baseline.json`'s `cycle_trajectory` for that run spans cycles 1-150 in full.
The per-cycle dual-update recursion needs every cycle's own (x, rho) to advance
lambda(c-1) -> lambda(c); cycles 2-5, 7-10, ..., 141-145 and 147-150 have no
recorded x in this run's stride file, so the recursion cannot be replayed past
cycle 1 without inventing intermediate values. Recorded in the committed JSON's
`gate3` field with this exact evidence; no reconstruction was attempted.

## Validation

- Script executes end-to-end, exit 0.
- Zero-solve claim ENFORCED (not asserted): `SolveProfileGuard([], ...)`
  installed for the whole script; `guard.verify(0, 0)` returns no failures
  (checked in `main()`'s `finally` block, which raises `AssertionError` if it
  ever does).
- Replay cross-validated against `g_baseline.json`'s independently-computed
  `boyd_ess_norm_y*` trajectory at all 477 cycles to <=1.5e-14 relative error
  (see Results).
- Preconditions (zero initial dual, no skipped cycles) verified from source/data,
  not assumed, by direct reuse of PART A's own verified helper functions.
- `s_max` constancy across periods within each (node,year,day) block asserted
  (`AssertionError` if violated) -- confirmed holds (no assertion fired).
- Output directory did not pre-exist; script refuses if it does (checked: ran
  once, succeeded).
- `git status --short` before writing this report shows only the two new paths
  this task owns; no file under any other worker's or run's output directory
  was touched.

## Unexpected findings

- The ESSO's own capture files (`esso_capture/baseline/node*_cycle*.jsonl`)
  record NO SoC or energy-state variable at all -- confirmed by enumerating
  every key present across a full cycle-477 file for node 5 (`pch`, `pdch`,
  `pnet`, `s_max`, slacks, bound multipliers, a `duals` list over four named
  ConstraintLists). This is a structural fact about the ESSO's own subproblem
  (`shared_energy_storage_data.py::_build_subproblem`), which genuinely has NO
  SoC/energy-state Pyomo variable -- the ESSO's degradation model tracks only
  annual aggregate throughput (`es_avg_ch_dch_per_unit`), not a per-period SoC
  trajectory. The day-balance/SoC-bound rows this script needed for the span
  construction live only in the TSO/DSO LOCAL copies of the shared storage
  (`network.py`'s `shared_es_soc`/`sess_soc_rule` family), which are per-agent,
  not per the "one physical schedule" this diagnostic needed -- SoC was
  therefore reconstructed from the ESSO capture's own `pch`/`pdch` (the task's
  own documented fallback), not read from any committed artifact.
- Both ‖I‖ and ‖J‖ are STILL monotonically drifting toward each other at cycle
  477 (I falling, J rising), a genuinely unsettled state that the pre-registered
  verdict rule did not explicitly anticipate (it implicitly assumed at least one
  series would settle). This is itself informative: it says the per-agent
  dual-norm build-up documented in the task's "Why" section (ESSO share rising
  36% while the total stayed flat) has NOT finished redistributing by cycle 477,
  for either the identifiable or the in-span component.
- ESSO's own I and J components are the LEAST settled of the three agents
  (10-11% relative change over the last 50 cycles, vs 2-5% for TSO/DSO) --
  worth flagging since ESSO's absolute dual magnitude is much smaller than
  TSO/DSO's (per the task's own cited cycle-1/477 norms), so this large
  relative drift corresponds to a small absolute one, but it is the LARGEST
  relative drift of any agent/component pair measured.

## Remaining issues

- The verdict is INCONCLUSIVE under the pre-registered rule as literally
  stated, extended (documented) to cover the "neither settles" case as
  equivalent to "both settle together". If the Planner wants a directional
  read despite the inconclusive verdict, the identifiable-fraction trend
  (falling from 0.685 to 0.520 over 477 cycles) and the sign of each series'
  drift (I falling, J rising, i.e. converging toward each other, with J's
  relative rate the larger of the two at the total level) both lean toward the
  in-span component J still having further to move -- an H2-leaning
  DIRECTIONAL trend, not a settled conclusion.
- This diagnostic covers gate 1 (run 1, `P515S35_REF_run`) only; gate 3 is not
  computable from committed artifacts (see above). No other gate/arm was in
  scope per the task.

## Questions for Planner

1. Given neither series settles by cycle 477, is a longer replay (if a
   longer-cycle run becomes available) or a different settling-window/threshold
   choice wanted, or should the "neither settles" finding stand as the reported
   result?
2. Is the identifiable-fraction trend (0.685 -> 0.520, falling) sufficient
   informally to prioritize an H2-leaning remedy (e.g. price-informed
   initialization would not by itself resolve the still-growing in-span
   component), or should this remain purely INCONCLUSIVE pending further
   evidence?

## Evidence / paths

- Script: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_h1h2_dual_identifiability.py`
- Output: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/H1H2_identifiability/`
- Reference run: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35_REF_run/`
- Gate 3 run (not computable): `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35_PT_run/`
