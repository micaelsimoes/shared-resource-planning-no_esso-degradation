# Worker Report — P5.15 Addendum 17: cycle-0 LMP unit calibration

## Task received

Bounded ZERO-SOLVE task (Planner, 2026-09-17): calibrate the units of the captured
cycle-0 node-balance duals from `p515_s36_cycle0_lmp_capture.py`/`WORKER_REPORT_S36_A17_
CAPTURE.md` (commit `b8f129e0`), whose captured cycle-0 LMPs came out ~1e-6 to 1e-8
$/MWh — not economically plausible against the scenario market price π (~85-105 $/MWh in
this configuration). Redo Addendum 17 diagnostic 1(c) ONLY if calibration succeeds. No
Pyomo/IPOPT solves (scipy LPs permitted, declared and counted). No production, case-file
or harness edits.

## Files inspected

- `WORKER_REPORT_S36_A17_CAPTURE.md`, `p515_s36_cycle0_lmp_capture.py` (commit `b8f129e0`)
  and its output `data/SRP1/Results/P515S36/cycle0_lmp/cycle0_lmp_capture_results.json`.
- `model_construction_helpers.py`: `node_balance_p_rule`/`node_balance_q_rule`
  (1366-1456), `pc_bounds`/`pc_initialize` (272-295), `compute_node_load` (1286-1327),
  `build_objective` (1579-1614), `interface_energy_settlement` (1617-1644),
  `objective_function_rule` (1647-1660), `generation_cost`/`generation_cost_rule`/
  `total_generation_cost_rule` (1663-1682), `load_is_tso_adn_interface` (1685-1689),
  `interface_pf_p_transmission_def`/`interface_pf_q_transmission_def` (1165-1197).
- `shared_resources_planning.py`: the `_run_operational_planning` initialization branch
  (2330-2495, in particular the standalone-solve success check at 2402 and the ordering
  of `_prepare_distribution_objectives_for_admm`/`_prepare_transmission_objectives_for_
  admm`/`update_distribution_models_to_admm`/`update_transmission_model_to_admm` at
  2440-2448, all AFTER that check); `create_transmission_network_model` (3440-3547, in
  particular the interface-node free/fix logic at 3458-3503); `update_transmission_model_
  to_admm` (4231-4320) / `update_distribution_models_to_admm` (4412-4472) (where
  `admm_objective`, `admm_objective_scale`, `admm_block_weight`, `admm_common_objective_
  scale` and `effective_scale` are FIRST created); `_get_admm_block_weight` (3136);
  `_run_operational_planning_without_coordination` (7156-7270) and
  `_add_tso_scenario_tracking_penalty` (3403-3436) — read in full to check and rule out an
  alternative explanation (see "Unexpected findings").
- `network_data.py:191` (`prob_operation_scenarios` default for a single scenario).
- `definitions.py:75` (`PENALTY_SCENARIO_DEVIATION = 9e4`).
- `p56a_oracle.py` (`fresh_planning`/`load_baseline` — confirmed zero-solve: reads case
  data via `read_planning_problem()` only, no `.build_model()`/`.optimize()`).
- `p513_solve_profile_guard.py` (`SolveProfileGuard`, reused unmodified).
- `data/SRP1/case9/case9_2025.json`, `data/SRP1/case33_2/case33_2_2025.json` (generator
  bus/type data, read directly, zero-solve).
- `data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl` (an
  UNRELATED, already-solved, already-pickled snapshot — loaded read-only, not modified,
  not a solve).

## Files modified

- Created `p515_s36_cycle0_lmp_calibration.py`.
- Created `data/SRP1/Results/P515S36/cycle0_lmp_calibration/` (`cycle0_lmp_calibration_
  results.json`, `sha256_manifest.json`).
- Created this report.
- `data/SRP1/Results/P515S36/cycle0_lmp/` (the earlier capture) was **not** modified —
  verified: `sha256sum` of `cycle0_lmp_capture_results.json` after this task still equals
  the value recorded in its own committed `sha256_manifest.json`
  (`3b9c34703520719e489c662e5116e35e0a744a06695c7d9c0256bca6f62ec9b8`).

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s36_cycle0_lmp_calibration.py
```
Exit 0. Guard counts `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0,
'blocked_exec': 0}` (zero Pyomo/IPOPT solves, verified via `guard.verify(0)` — empty
failure list). scipy price-taker LP call count: declared 0, observed 0 (`sept.get_lp_call_
count()` delta). No `p515_g_g1_g4_admm_gates.py` process was running and no
`.p515_g_gate.lock` existed before or during this run (checked; a different Worker's
concurrent task, `H1H2_identifiability`, writes to a disjoint output directory and was not
touched).

Several throwaway, uncommitted inspection commands (inline `python -c`/heredocs, not
part of the repository) were also run to explore the FrozenSMOPF pickle's components and
to cross-check numeric outputs against the committed JSON — all zero-solve (pickle
loads and `pe.value()` reads only).

## Results

### Step 1 — active objective and scale factors (source-verified, zero-solve)

At the moment the 51 standalone construction solves run, **only `model.objective`
exists** — `model.admm_objective` is a *separate* Objective component created later
(`shared_resources_planning.py:4382`/`:4518`/`:4592`), reached only from `update_
transmission_model_to_admm`/`update_distribution_models_to_admm`/`update_shared_energy_
storage_model_to_admm`, which are called at `shared_resources_planning.py:2446-2448` —
strictly **after** the standalone-solve success check at line 2402 that immediately
follows the 51 solves. So the active objective at capture time is `model.objective`,
**unscaled**: no `/effective_scale`, no `admm_objective_scale`/`admm_block_weight`/
`admm_common_objective_scale` Param (these do not exist on the model yet — confirmed by
a positive control: the unrelated FrozenSMOPF cycle-7 pickle, from *after* `update_
transmission_model_to_admm` ran, **does** have them, populated). `model.interface_
settlement_weight` is created at `model_construction_helpers.py:1611` with `initialize=
0.00` unconditionally and is set to 1 only by `_prepare_transmission_objectives_for_
admm`/`_prepare_distribution_objectives_for_admm` — also called after line 2402, so
still 0.00 at capture time (confirms the prior report's claim, now cited independently).
The only factors multiplying a controllable generator's cost gradient at capture time are
`baseMVA` (=100.0, both TSO and DSO) and `prob(s_m)*prob(s_o)` (=1.0 exactly, for this
single-market/single-operation-scenario configuration — `shared_resources_planning.py:
7418`, `network_data.py:191`). No day weight, no 365/years factor, no `admm_block_weight`
apply at capture time (those live only inside `update_transmission_model_to_admm`/
`update_distribution_models_to_admm`, `shared_resources_planning.py:4314-4320`/
`:4467-4472`).

### Step 2 — identity check

**Buses 5, 7, 9 host no controllable, cost-bearing generator** in case9 (confirmed from
`case9_2025.json`, read via `O.fresh_planning`, zero-solve): the 3 CONV generators sit at
buses 1, 2, 3; buses 4, 6, 8 host WIND/PV generators (`is_controllable()=False`, excluded
from `generation_cost`). The Planner's literal identity check (an unconstrained
cost-bearing generator *at a captured bus*) cannot be performed — this is stated, not
worked around.

Per the Planner's own fallback ("use any bus where the identity holds, e.g. the TSO slack
generator"), and because the calibration script must perform zero new solves, I used an
**already-pickled, already-solved** fixture — `data/SRP1/Results/FrozenSMOPF/matched_
success_TSO_case9_2025_Summer_cycle7.pkl` (cycle 7 of an unrelated run; loading and
reading it is not a solve) — at bus 1 (gen_id=1, index 0), strictly interior at every one
of its 24 periods (`pg` between `lb=-1e-5` and `ub=2.50001` p.u.). I extracted the
objective's own symbolic coefficient of `pg[0,0,0,p]` (via `generate_standard_repn`) and
compared it to `model.dual.get(node_balance_p[0,0,0,p])` for all 24 periods:

| | value |
|---|---|
| ratio `dual / objective_coefficient`, mean | **0.9999999952** |
| ratio, min | 0.999999986026382 |
| ratio, max | 0.999999999169788 |
| sign | **no flip** (ratio ≈ +1.0, not −1.0) |

This confirms, on real IPOPT-exported output in this codebase, that `model.dual` equals
exactly `+1 ×` the *active* objective's own local coefficient of the isolated variable —
no missing scale factor, no sign flip — validating the **mechanism** the Planner's
proposed identity relies on (though not at cycle 0, and not at a bus this diagnostic
captured).

**Cross-bus comparison** (same fixture, p=0): bus-1 (generator) dual = 12174.29; buses
5, 7, 9's own node-balance-P duals (node_idx 4, 6, 8) = 12314.69 / 12226.72 / 12297.56 —
within 0.4%–1.2% of the generator-bus value (ordinary loss-driven LMP spread), **not**
orders of magnitude smaller. This shows that in a *properly cost-driven* instance of this
exact model, a load bus's dual is comparable in magnitude to a nearby interior
generator's — the captured cycle-0 magnitude (~1e-9× smaller than a plausible price) is
therefore **not** a generic property of these buses' topology.

### Step 2c — the identity that actually governs buses 5/7/9 at capture time

`compute_node_load` (`model_construction_helpers.py:1286-1327`) builds `Pd` at the ADN
interface buses as `pc[adn_load_idx]` (fixed to the current consensus interface value,
`shared_resources_planning.py:3477-3484` — the *same* constructor the 51-solve
standalone-init phase calls) **plus** `interface_delta_p[dn,...]`, which is **freed**
(not fixed) at construction, bounded by ± the DSO's own interface-transformer rating in
p.u. (`shared_resources_planning.py:3498-3503`; numerically 2.0 / 1.0 / 1.5 p.u. at nodes
5/7/9 — not tiny). `interface_delta_p` has **zero** direct objective sensitivity at
capture time: it enters only the interface-settlement term, whose weight is 0.00 (step
1), and no other term in `objective_function_rule` references it. So, for any period
where `interface_delta_p` is strictly interior of its bounds, KKT stationarity for that
free variable forces `dual[node] = 0` **exactly** (mod solver numerical tolerance),
**independent of `c_p[p]`**. The observed 1e-5 to 1e-8 p.u. magnitudes are consistent
with this being IPOPT's own KKT-residual/duality-gap noise floor at termination, not a
real (even tiny) priced quantity.

**An alternative explanation was checked and ruled out**: `_add_tso_scenario_tracking_
penalty` (`shared_resources_planning.py:3403-3436`) *would* give the interface variable a
large (`PENALTY_SCENARIO_DEVIATION*1e6 = 9e10`) quadratic tracking cost — but that
function is reached only from `_run_operational_planning_without_coordination`
(`:7156-7270`), the separate "no_coordination" baseline run mode, **not** the ADMM
initialization path this capture exercises. Confirmed by reading both functions in full;
recorded here so this dead end is not re-walked.

### Step 3 — corrected conversion and calibrated LMPs vs π

**No conversion factor changes**: `LMP[$/MWh] = dual_p_pu / base_mva` is confirmed
correct (steps 1, 2, 2c find no missing probability/day-weight/`admm_block_weight`/σ
factor at capture time). Recomputing with this unchanged formula:

| node | calibrated LMP mean ($/MWh) | π mean ($/MWh, 2025) | ratio (LMP/π) | plausible (within 2x)? | Pearson corr (LMP vs π shape) |
|---|---|---|---|---|---|
| 5 | 9.17e-08 | 95.41 | 9.61e-10 | **no** | 0.780 |
| 7 | 1.14e-07 | 95.41 | 1.18e-09 | **no** | 0.712 |
| 9 | 1.06e-07 | 95.41 | 1.11e-09 | **no** | 0.817 |

None is remotely plausible (ratio ~1e-9, not within a factor of 2). The moderate
correlation with π's diurnal shape (0.71-0.82) does **not** contradict the "solver-noise
floor" explanation — it is reported, not over-interpreted, as a secondary observation;
both π and the residual/conditioning that shapes IPOPT's noise floor share the same
diurnal load pattern, and DSO node 7's WORKER_REPORT_S36_A17_CAPTURE.md already recorded
that price-taker LP behavior driven by this series' *shape* is what its 1(c) run used.

**DSO reference-node duals** (task step 4, reported, **not** used as a storage price):
calibrated to ~1e-6 to 1e-7 $/MWh at all three nodes — also at the numerical floor, and
consistent with (not contradicting) the independent finding that the storage term
cancels at the DSO reference node (its own generator is `type=REF`, structurally
excluded from `generation_cost` — confirmed directly from `case33_2_2025.json`: the only
generator at the reference bus (bus 1) is `gen_id=1, type='REF'`; the feeder's other
generators are non-controllable WIND/PV elsewhere on the feeder).

### Calibration verdict

**FAILS**, in the specific sense the Planner's task defines success/failure: the
identity `dual = ±(scale)×baseMVA×c_p[p]` does **not** hold at the captured buses (5, 7,
9) — not because a scale factor is missing, but because the locally-relevant free
variable there (`interface_delta_p`) carries zero objective weight at capture time, so
the true (analytic) dual is ~0 there **regardless of `c_p[p]`/π**. No scale factor
converts an analytically-zero quantity into a plausible market price. Per the task's own
branch ("If calibration fails ... stop after step 3. Do not redo 1(c)."), **diagnostic
1(c) was not redone.**

## Validation

- Script executes, exit 0; guard-verified zero Pyomo/IPOPT solves (`guard.verify(0)` →
  `[]`); declared/observed scipy LP calls both 0.
- Step 2's identity-mechanism check is exact to 8-9 significant figures across all 24
  periods on a real, independently-pickled, already-solved model in this codebase — not
  a theoretical claim alone.
- Step 3's numeric LMP means/ratios were independently recomputed by hand (outside the
  script, via ad hoc `python -c` checks) before writing the script and match the script's
  own output exactly (e.g. node 5 mean 9.166534702886068e-08, ratio 9.607514580229839e-10
  in both).
- `data/SRP1/Results/P515S36/cycle0_lmp/cycle0_lmp_capture_results.json` confirmed
  byte-identical to its committed manifest hash after this task (not overwritten).
- `git diff --cached --name-only` checked before committing (below) to confirm only this
  task's files are staged.

## Unexpected findings

- The prior report's magnitude explanation ("`interface_settlement_weight` defaults to
  0.00 ... so the standalone DSO objective at the reference bus has essentially no priced
  economic driver") was correct for the *DSO reference bus* but **incomplete/imprecise**
  as a general explanation of the *TSO-side* near-zero magnitude at buses 5/7/9: the
  decisive local mechanism there is the free, zero-weighted `interface_delta_p` variable
  (step 2c), not merely the settlement weight per se (interface_settlement_weight being
  0.00 is one of the two facts step 2c's argument needs, but not on its own sufficient —
  the *freeing* of `interface_delta_p`, `shared_resources_planning.py:3498-3499`, matters
  equally).
- `_add_tso_scenario_tracking_penalty`/`_run_operational_planning_without_coordination`
  looked, on first reading, like they might apply to the standalone-init construction
  phase (both operate on the same interface variables). They do not — confirmed by
  reading the call graph; recorded above so a future reader does not re-walk this dead
  end.
- The correlation (0.71-0.82) between the near-zero captured duals and π's diurnal shape
  is real in the data but is not evidence against the "numerical noise floor"
  explanation — it is reported as an open, secondary observation, not resolved further
  (would require a new solve with primal capture of `interface_delta_p`/`pc_adn` values
  to investigate, out of this task's zero-solve scope).

## Remaining issues

- The "numerical noise floor" claim for the captured cycle-0 dual magnitudes is
  well-supported (interior-stationarity argument, source-verified) but not itself
  solve-verified for cycle 0 specifically (that would require capturing `interface_
  delta_p`'s own bound status at cycle 0, which the original capture did not record and
  this zero-solve task cannot add without a new solve).
- Diagnostic 1(c) (price-taker LP re-solved with calibrated cycle-0 LMPs) was **not**
  redone, per the task's own stop condition. The existing 1(c) result from `WORKER_
  REPORT_S36_A17_CAPTURE.md` (node 5 EFC/day 1.413 at default budget; nodes 7, 9
  non-convergent) stands as reported there, understood now to be a price-taker LP driven
  by a numerically-near-zero (not economically meaningful) price series' shape.

## Questions for Planner

1. Given the identity fails structurally (not from a missing scale factor), does the
   Planner want a **new, non-zero-solve** capture that additionally records `interface_
   delta_p`'s primal value/bound status at cycle 0 (to fully close the "interior vs.
   saturated" question left open above), or is the analytic argument in step 2c
   sufficient to retire the cycle-0-LMP-as-storage-price approach entirely?
2. Should `WORKER_REPORT_S36_A17_CAPTURE.md`'s magnitude-finding paragraph be
   superseded/corrected in a follow-up note to point at this report's more complete
   mechanism (step 2c), given CLAUDE.md's evidence rules on reports whose explanations
   turn out to be incomplete?

## Evidence / paths

- Script: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s36_cycle0_lmp_calibration.py`
- Output: `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/cycle0_lmp_calibration/`
- Input (unmodified): `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S36/cycle0_lmp/cycle0_lmp_capture_results.json`
- Input (read-only, unrelated fixture): `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl`
