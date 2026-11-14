# Worker Report — P5.15 Step 3.5, diagnostic Z2 (floor active/slack at C*)

## Task received

Bounded ZERO-SOLVE diagnostic Z2 (Planner instruction, 2026-09-16): determine
whether the SoH floor `es_soh_per_unit_cumul[y_inv,y] >= soh_min` (0.50) is
active or slack at candidate C\* under the price-taker schedule, measured
with the PRODUCTION EFC-per-day and degradation formulas, without any
Pyomo/IPOPT solve (a small, declared number of `scipy.optimize.linprog`
(HiGHS) calls permitted). Establish the formulas from code with file:line
citations, compute the harness-definition EFC/day and SoH trajectory per
(node, cohort-year) under the price-taker schedule (iterating capacity/SoH to
a fixed point with a declared iteration count), check floor activity, confirm
the 1.4612 threshold algebraically, and — only if the floor binds — re-solve
a coupled price-taker LP with the linear floor row added and report its
shadow price.

## Files inspected

- `p514_n_instrumented_cstar.py` (harness EFC/day formula, line 133; capture
  logic lines 118-144; `EFC_BINDING_THRESHOLD = 1.4612` at line 46).
- `p515_s34_efc_benchmark.py` (committed single-day price-taker benchmark;
  reused for the LP row provenance and as the cross-check reference; not
  imported, reimplemented per the task's read-only-reuse instruction).
- `shared_energy_storage_data.py` lines 401-720, 1780-1835 (degradation
  chain `energy_storage_capacity_degradation`, `available_e_capacity_unit`,
  `get_updated_capacities`/`get_available_capacities`, `rated_e_capacity_unit`,
  `energy_storage_charging_discharging`).
- `shared_energy_storage_parameters.py` (ageing constants, `cl_eff`
  derivation, `phi_cal` default).
- `model_construction_helpers.py` lines 790-935 (`sess_soc_lower_limit`,
  `sess_soc_upper_limit`, `sess_active_sum_limit_rule`, `sess_soc_rule`,
  `period_duration_hours`) — confirms which capacity Param the physical
  network dispatch bounds actually read.
- `network.py` lines 388-389 (`shared_es_s_rated_fixed`/`shared_es_e_rated_fixed`
  Param declarations).
- `shared_resources_planning.py` lines 2465, 4987-5330 (where
  `sess_available_capacities`/`sess_estimated_capacity[...]['e_available']`
  is computed and pushed into the network Params).
- `data/SRP1/SRP1.json` (Years=5/5/5, Days=Spring 92/Summer 91/Autumn 91/
  Winter 91).
- `data/SRP1/SharedESS/SRP1_ESS_Params.json` (ageing block: `calibration.status
  = ACTIVE`, `cycles_n=10000`, `reference_dod_d=0.80`, `eol_retention_r=0.50`,
  `minimum_soh=0.50`, `calendar_life_years=15`; no `calendar_retention_per_year`
  key, so `phi_cal` = class default 1.00).
- `p513_solve_profile_guard.py` (guard API, `verify(0)` pattern).
- `p56a_oracle.py` module docstring (confirms `load_baseline()` is read-only,
  as already established/used by the S34 script).
- `data/SRP1/Results/P515S34/EFC_benchmark/efc_benchmark_results.json`
  (committed cross-check reference; read, not modified).

A zero-solve interactive read (guard-armed, `p56a_oracle.load_baseline()`,
same pattern as S34) was used to confirm parameter uniformity
(`eff_ch=0.97`, `eff_dch=0.96`, `cl_eff=11541.560327111707`, `phi_cal=1.00`,
`soh_min=0.50`, `t_cal=15`) across all three active nodes (5, 7, 9) and all
three modelled years, and that prices (`cost_energy_p[year][day][0]`) are
identical across nodes 5/7/9 (single-market SRP1 confirmed by the S34
committed JSON showing the same 2025-Winter value, 1.851977663230241, for all
three nodes).

## Files modified

None (production, case-file, or harness files). New files only.

## Files created

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_s35_z2_floor_slackness.py`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35/Z2/z2_floor_slackness_results.json`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P515S35/Z2/sha256_manifest.json`
- this report

## Changes made

New diagnostic script only. Every formula (harness EFC/day, the degradation
law, the log-domain SoH chain, the linear floor equivalent, the capacity-wear
asymmetry between S and E, the LP rows) is preserved verbatim in the script's
module docstring with file:line citations — see the script itself for the
full derivation, not repeated here except where needed for this report's
Results section.

**Key facts established from code (see script docstring for full citations):**

1. Harness EFC/day (`p514_n_instrumented_cstar.py:133`) =
   `es_avg_ch_dch_per_unit[y_inv,y] / (2 * es_e_rated_per_unit[y_inv,y])`,
   where `es_avg_ch_dch_per_unit` is day-weighted (`num_days/365`,
   92/91/91/91) and efficiency-weighted (`eff_ch*pch*dt + pdch*dt/eff_dch`),
   and `es_e_rated_per_unit` is the NAMEPLATE investment (never SoH-derated) —
   `shared_energy_storage_data.py:552,584,614-622`.
2. Degradation law: `D[y_inv,y]*(2*cl_eff*E_inv) == 365*num_years*avg_ch_dch[y_inv,y]`;
   `soh_cumul[y_inv,y] == prev_soh*exp(-D[y_inv,y])*phi_cal**num_years`;
   floor `soh_cumul[y_inv,y] >= soh_min` for EVERY y in the cohort window, not
   only the terminal one (`shared_energy_storage_data.py:652-676`). Since
   `D>=0`, the cumulative sum is non-decreasing, so the terminal-year (2035)
   row is the binding one to check.
3. Linear floor equivalent, checked algebraically:
   `sum_{y<=Y} D[y] <= -ln(soh_min) + (sum_{y<=Y} num_years[y]) * ln(phi_cal)`.
4. Capacity wear: **S never derates; E for the physical SOC/dispatch bound
   DOES derate by SoH** in every later block
   (`shared_energy_storage_data.py:584`, `get_available_capacities:256-262`,
   pushed into `shared_es_e_rated_fixed` at
   `shared_resources_planning.py:5005/5202/5327`), while the **degradation
   law's own denominator stays nameplate** — confirmed as a deliberate,
   pre-existing asymmetry (comment at `shared_energy_storage_data.py:309`).
   Within one ESSO solve this same-year self-consistency is genuinely
   simultaneous (all in one IPOPT NLP); this zero-solve diagnostic
   approximates it by fixed-point iteration on the available capacity, as the
   task instructs, and states this as an explicit approximation, not a claim
   of reproducing the coupled NLP exactly.
5. `cl_eff = 11541.560327111707` (C3 calibration ACTIVE), `phi_cal = 1.00`
   (default, absent from the case's ageing block), `soh_min = 0.50`,
   `t_cal = 15` yr, cohort window = all three modelled years (2025/2030/2035,
   since `round(15/5) = 3 = len(years)`).

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s35_z2_floor_slackness.py
```

Single run. Output directory `data/SRP1/Results/P515S35/Z2/` did not exist
beforehand (script refuses if it does); it was created fresh by this run.

## Results

**Declared scipy LP count (checked exactly before/after run):**
Base = 3 nodes × 3 years × 4 days × 20 fixed-point iterations (Part A, 720) +
3 nodes × 3 years × 4 days (eff=1 cross-check, 36) = **756**, unconditional.
Part D (coupled floor re-solve) = 1 per node whose Part-A unconstrained
terminal SoH is at/below `soh_min` — **0 nodes bound**, so declared Part D = 0.
**Declared total = 756. Observed = 756. Exact match: True.**
Pyomo/IPOPT `SolveProfileGuard`: armed with an empty permitted list for the
whole script; `permitted_solve=0, permitted_exec=0, blocked_solve=0,
blocked_exec=0`; `guard.verify(0)` returned no failures.

**Cross-check vs the committed `P515S34` benchmark** (efficiency on / wear
off, and efficiency off / wear off, both at nameplate capacity, uncoupled
across years — the same conditions S34 used): **max abs diff = 0.0 for both
variants**, over all 3 nodes × 3 years × 4 days = 36 cells each. This
confirms the reimplemented LP (rows [1]-[5]) and the harness's own
`avg / (2*rated)` formula agree bit-for-bit with the committed benchmark
under matching assumptions, before the fixed-point/derating machinery is
switched on.

**Part A — harness-definition EFC/day (day-weighted, efficiency-weighted,
per cohort-year), all three nodes identical (prices are identical across
nodes 5/7/9):**

| Year | EFC/day | D_y | soh_cumul (cumulative, end of year) | E_available (MWh) |
|------|---------|-----|--------------------------------------|--------------------|
| 2025 | 1.191777 | 0.188449 | 0.828243 | 3.209441 |
| 2030 | 1.188810 | 0.187980 | 0.686308 | 2.659444 |
| 2035 | 0.964719 | 0.152545 | 0.589209 | 2.283186 |

Fixed-point convergence: every (node, year) block ran the full 20 declared
iterations (no early exit) and reached `final_rel_change = 0.0` well before
iteration 20 (e.g. node 5 / 2035: iteration 0 guess=3.875 MWh → rel_change
0.448; iteration 19 guess=2.283186 MWh → rel_change 0.0, i.e. the guess
exactly reproduces itself). All 27 (node, year) fixed points converged
(`fixed_point_converged: true` for all).

**Part B — floor activity:** terminal (2035) SoH = **0.589209** for every
node, vs `soh_min = 0.50`. **Margin = +0.089209** (SoH sits about 17.8% of
`soh_min` above the floor). **Floor is SLACK for all three nodes (5, 7, 9).**
No node's unconstrained terminal SoH falls at or below `soh_min`.

**Part C — equivalent constant EFC/day threshold:** computed
`cl_eff*(-ln(soh_min))/(365*sum(num_years)) = 1.461187214611872`, vs the
reference `EFC_BINDING_THRESHOLD = 1.4612`. **Abs diff = 1.28e-5** — the
threshold is confirmed correct to 4 decimal places by the production
formula (`cl_eff = k = N*D/(-ln R)`, `soh_min = 0.50`, cohort spans 3×5 = 15
years).

**Part D — coupled floor re-solve:** not performed (floor provably slack
under the unconstrained Part-A schedule for every node — adding a
constraint that's already satisfied cannot change the optimum). **Floor
multiplier = 0.0 for every node**, reported without a re-solve, exactly as
the task specifies for the slack case.

## Validation

- Script correctly refused to run had `data/SRP1/Results/P515S35/Z2/`
  already existed (verified the directory absent beforehand).
- `SolveProfileGuard([]).verify(0)` passed: zero Pyomo/IPOPT solves anywhere
  in the run.
- Declared scipy LP count matched the observed count exactly (756 = 756);
  the counter is a single global incremented inside `solve_price_taker`
  (called from both Part A's per-day fixed-point loop and the eff=1
  cross-check loop) and inside `solve_coupled_price_taker_with_floor`
  (never called here, since Part D did not trigger).
- Cross-check against the committed S34 benchmark reproduced all 36+36
  per-cell EFC values to `0.0` absolute difference under matching
  assumptions (nameplate capacity, no cross-year coupling) — this validates
  both the LP reimplementation and the harness-formula reimplementation
  independently of the new fixed-point/derating machinery.
- Part C's algebraic check reproduced the frozen `EFC_BINDING_THRESHOLD =
  1.4612` to 1.28e-5, an independent confirmation that the degradation-law
  and floor-equivalence algebra in the docstring is correct.
- Fixed-point iteration genuinely converged (verified via the recorded
  `history` showing `rel_change` decaying from ~0.45 at iteration 0 to
  exactly `0.0` well before the fixed iteration budget of 20 was exhausted,
  for every (node, year) cell) rather than merely being declared converged.
- I did **not** touch `p515_g_g1_g4_admm_gates.py` (per the coordination
  note) and did not stage or modify `data/SRP1/SRP1_params.json`, which
  showed as pre-existing staged (`M `) state unrelated to this task — `git
  diff data/SRP1/SRP1_params.json` shows no unstaged content and I made no
  edits to it.

## Unexpected findings

- The Planner's "Why" section cites an independent review's estimate of
  ~1.335 (2025), 1.465 (2030), 1.366 (2035), mean ≈1.39 for the
  harness-definition EFC/day. This run's production-formula figures are
  **lower and more front-loaded**: 1.192 (2025), 1.189 (2030), 0.965 (2035),
  mean ≈1.115. The **conclusion direction agrees** (all three years below
  the 1.4612 threshold, floor slack) but the **magnitude and the year-3 drop
  are more pronounced** here. A plausible driver, visible directly in this
  run's own output: **EFC/day itself falls in 2035** (0.965 vs ~1.19 in the
  earlier years) because the fixed-point capacity derating is a genuine
  negative-feedback loop — E_available drops from 3.875 (nameplate) to 2.283
  MWh by 2035 (soh_cumul 0.589), which shrinks the SOC band available for
  arbitrage and therefore throughput, which in turn caps further degradation.
  The review's estimate may not have modelled this capacity-derating
  feedback (or modelled it differently); this is an observation about a
  possible source of the discrepancy, not a claim about which estimate is
  more correct in an absolute sense — the review's own numbers/method were
  not available to this Worker for a side-by-side comparison.
- The self-limiting feedback above means that, on this production formula,
  a price-taker schedule at C\*'s nameplate rating is comfortably inside the
  SoH floor (margin +0.089, ~17.8% of `soh_min`) even under an aggressive
  unconstrained arbitrage schedule — i.e., not merely "close to slack" but
  slack with a material margin. The Planner's stated success criterion
  ("equilibrium at the SoH threshold with the degradation constraint
  active") is therefore **not reached by the unconstrained price-taker
  benchmark at C\*'s current nameplate rating**, on the harness's own EFC
  definition and the production degradation chain, consistent with what
  Addendum 16 flagged as a risk.

## Remaining issues

- This diagnostic approximates the ESSO's genuinely simultaneous (same-year)
  NLP self-consistency between dispatch, D, SoH and available capacity via
  external fixed-point iteration on a decoupled LP. It converges cleanly
  here (rel_change → 0.0), but it is not a proof that the true joint NLP
  optimum (which ADMM/IPOPT would find, coupled with the TSO/DSO networks
  and other cost terms, not just the ESS's own arbitrage profit) produces
  the same trajectory — the price-taker schedule is, by construction, an
  UPPER BOUND on achievable EFC/day (no coordination, no network
  constraints, no degradation cost in the objective), so if even this upper
  bound sits comfortably below the threshold, that is informative, but the
  reverse inference (unconstrained-above-threshold implying the true
  coordinated schedule also crosses it) would not have followed either way.
- Part D's coupled-LP machinery (`solve_coupled_price_taker_with_floor`) was
  written and is available in the script but never exercised in this run
  (0 binding nodes) — it has not been validated by an actual solve; if the
  Planner wants it exercised end-to-end (e.g., under a tighter candidate or
  lower `soh_min`), that would need a further bounded run.

## Questions for Planner

- The magnitude gap between this run's EFC/day figures (mean ≈1.115) and the
  review's cited estimate (mean ≈1.39) is large enough (~20%) that it may be
  worth reconciling explicitly — in particular, whether the review's
  estimate applied the same same-year capacity-derating feedback this script
  applies, since that feedback is what drives 2035's EFC/day down the most
  in this run. I did not have access to the review's own computation to
  check this directly.
- Given the floor is slack with a ~17.8%-of-`soh_min` margin under the
  unconstrained price-taker upper bound, does this settle the "reachability
  of Addendum 16's success target at C\*" question, or does the Planner want
  a follow-up bounded diagnostic (e.g., Part D exercised under a
  parametrically tighter case, or a check of whether some OTHER candidate
  in the neighbourhood of C\* would bind the floor)?

## Commit

Explicit pathspec only (verified `git diff --cached --name-only` before
committing lists exactly these four paths — not `p515_g_g1_g4_admm_gates.py`
or any other file):
- `p515_s35_z2_floor_slackness.py`
- `data/SRP1/Results/P515S35/Z2/z2_floor_slackness_results.json`
- `data/SRP1/Results/P515S35/Z2/sha256_manifest.json`
- `WORKER_REPORT_S35_Z2.md`
