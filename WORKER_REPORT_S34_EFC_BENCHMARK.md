# Worker Report — P5.15 Step 3.4 EFC* price-taking storage arbitrage benchmark

## Task received

Bounded diagnostic task (Planner, 2026-09-16): compute, per active node (5, 7, 9), per year
(2025, 2030, 2035) and per representative day, the profit-maximising price-taking storage
schedule for candidate C\* and report its EFC/day, to serve as the diagnostic benchmark for
Step 3.4's gate ("EFC/day rises from 0.067 to order 1") instead of the retired 1.1 figure.
Zero production solves; no production-code changes; new output directory
`data/SRP1/Results/P515S34/EFC_benchmark/`.

## Files inspected

- `p514_n_instrumented_cstar.py` (C\* instance: `S_INV=0.96875`, `E_INV=3.875`,
  `INVEST_YEAR=2025`; EFC block formula `value = avg / (2.0 * rated)`, line 133)
- `shared_energy_storage_data.py` (`energy_storage_charging_discharging` /
  `es_avg_ch_dch_per_unit`, lines 586-622; `energy_storage_capacity_degradation`,
  lines 624-676; `create_shared_energy_storages`, lines 159-171)
- `shared_energy_storage.py` (class defaults `eff_ch=0.97`, `eff_dch=0.96`, lines 14-15)
- `model_construction_helpers.py`: `sess_soc_rule` (893-912), `sess_soc_final_rule`
  (915-921, day balance), `sess_active_sum_limit_rule` (827-836, power envelope),
  `sess_soc_lower_limit`/`sess_soc_upper_limit` (840-847, energy bounds),
  `period_duration_hours` (797-808), `flexibility_cost` (1687-1701),
  `load_is_tso_adn_interface` (1682-1685), `adn_interface_flexibility_cost` (1712-1732,
  row-3 counterfactual, "NOT part of any solver objective")
- `network.py`: `shared_es_pch`/`shared_es_pdch` domain (390-391),
  `shared_es_s_rated_fixed`/`shared_es_e_rated_fixed` Params (388-389),
  `interface_energy_settlement` not inspected here (irrelevant to row-3; see below),
  `_compute_cost_flexibility_per_scenario` (1945-1958, not used)
- `definitions.py`: `ENERGY_STORAGE_MAX_ENERGY_STORED=0.90`,
  `ENERGY_STORAGE_MIN_ENERGY_STORED=0.10`, `ENERGY_STORAGE_RELATIVE_INIT_SOC=0.50`
  (lines 38-40); `HOURS_PER_REPRESENTATIVE_DAY=24.0` (line 94);
  `EQUALITY_TOLERANCE=1e-5` (line 90)
- `network_parameters.py` (day-balance slack default `True`, lines 75-89)
- `shared_resources_planning.py`: `cost_energy_p` construction (7099-7119) and copy onto
  `shared_ess_data` (6976); `cost_flex` construction (7119) and copy onto TSO/DSO network
  objects (6889, 6905, 6932, 6948) — NOT copied onto `shared_ess_data` (verified: `sed` has
  no `cost_flex` attribute; only `planning.cost_flex` does)
- `p56a_oracle.py`: `load_baseline` (95-110, pure data read, no Pyomo model built),
  `fresh_planning` (122-144)
- `p513_solve_profile_guard.py` (full file; `SolveProfileGuard`, armed with `permitted=[]`)
- `data/SRP1/SRP1.json` (`NumMarketScenarios: 1`, line 17)
- `data/SRP1/Results/P515S33_E2_run/interface_settlement_detail_s31c.json`
  (`interface_reporting_detail[node][year][day]['periods'][p]['price_per_mwh']`, used as
  the cross-check source)
- `REVISION_CONTEXT.md` (Amendment 2026-09-15 row-3 removal / Addendum 10-11; Amendment
  2026-09-16 Addendum 15, Step 3.4 gate and "no shared-ESS usage price" decision)

## Files modified

None (production code untouched). New files only:

- `p515_s34_efc_benchmark.py` (new diagnostic script, repo root)
- `data/SRP1/Results/P515S34/EFC_benchmark/efc_benchmark_results.json` (new)
- `data/SRP1/Results/P515S34/EFC_benchmark/sha256_manifest.json` (new)
- `WORKER_REPORT_S34_EFC_BENCHMARK.md` (this file, new)

## Changes made

Wrote `p515_s34_efc_benchmark.py`, a standalone diagnostic (not touching Pyomo/IPOPT) that:

1. Installs `SolveProfileGuard([])` (empty permitted list) before importing any production
   module, so any `OptSolver.solve` / `SystemCallSolver._execute_command` call anywhere in
   the process raises `SolverInvocationBlocked` immediately.
2. Loads the canonical planning data via `p56a_oracle.load_baseline()`
   (`SharedResourcesPlanning.read_planning_problem()` — a data read, builds no Pyomo model).
3. Reads, per (node, year), the shared-ESS efficiencies (`eff_ch`, `eff_dch`) directly from
   `sed.shared_energy_storages[year][idx]` and asserts they are uniform across node/year
   (they are: 0.97 / 0.96 everywhere, the class defaults, un-overridden for C\*).
4. Reads the market price series `sed.cost_energy_p[year][day][0]` (single market scenario)
   for every (node, year, day) — the array is identical across nodes 5/7/9 by construction
   (one global price series copied onto every network object,
   `shared_resources_planning.py:6889/6905/6932/6948/6976`).
5. Cross-checks every read price against `price_per_mwh` in
   `interface_settlement_detail_s31c.json` — exact match (max abs diff = 0.0 over all
   3×3×4×24 = 864 cells checked).
6. Solves, per (node, year, day), a linear program with `scipy.optimize.linprog` (`method='highs'`)
   for the profit-maximising price-taking schedule, exactly reproducing constraints [1]-[5]
   in the script's module docstring (power limits, active-power envelope, SoC recursion with
   `eff_ch`/`eff_dch`, SoC bounds at 10%/90% of `E_INV`, hard day balance at 50% of `E_INV`).
7. Reports `EFC*_energy` (variant 1), `EFC*_efficiency1` (variant 3, `eff_ch=eff_dch=1.0`),
   and explicitly SKIPS variant 2 (`EFC*_row3`) with a documented reason (see Unexpected
   Findings).
8. Refuses to run if `data/SRP1/Results/P515S34/EFC_benchmark/` already exists
   (`os.makedirs(OUT_DIR)` without `exist_ok`, after an explicit existence check).
9. Verifies `guard.verify(0)` (zero permitted, zero blocked) before writing any output, and
   raises if the guard was ever triggered.
10. Writes a sha256 manifest of the results JSON and the script itself.

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s34_efc_benchmark.py
```

Output:
```
EFC*_energy (2025, max across nodes 5/7/9) = 1.8520 (node 5, Winter) -> order 1.
solve_profile: {'observed': {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}, 'permitted_declared': 0, 'guard_verify_failures': []}
price cross-check max abs diff vs s31c JSON: 0.0
wrote: .../data/SRP1/Results/P515S34/EFC_benchmark/efc_benchmark_results.json
wrote: .../data/SRP1/Results/P515S34/EFC_benchmark/sha256_manifest.json
```

A second, throwaway (not saved to any result path) interactive python session was run to
sanity-check the LP's per-period schedule for node 5 / 2025 / Winter (guard armed, verified
zero solves): confirms bang-bang behaviour, no simultaneous charge+discharge in any period,
SoC touches both the 10% floor and the 90% ceiling, day balance holds exactly
(`soc[23] = 1.9375 = soc_init`), and `EFC = (7.4321+6.9208)/(2*3.875) = 1.8520`, matching the
saved result exactly.

## Results

**Instance** (recorded in every artifact): `s_mva=0.96875`, `e_mwh=3.875`, `invest_year=2025`,
`eff_ch=0.97`, `eff_dch=0.96`, `soc_min_frac=0.10`, `soc_max_frac=0.90`,
`soc_init_frac=0.50`, `dt=1.0 h`.

**EFC\*\_energy and EFC\*\_efficiency1, all node/year/day cells** (all three nodes are
numerically identical — the price series is node-invariant; shown once):

| Year | Day    | mean π ($/MWh) | max−min | MAD   | EFC\*\_energy | EFC\*\_eff1 | binding (v1)  |
|------|--------|-----------------|---------|-------|----------------|-------------|---------------|
| 2025 | Spring | 83.60           | 154.83  | 47.10 | 1.0369         | 2.3500      | power+energy  |
| 2025 | Summer | 90.61           | 110.62  | 31.74 | 0.7964         | 2.1500      | power+energy  |
| 2025 | Autumn | 104.32          | 81.25   | 15.20 | 1.6587         | 2.4750      | power+energy  |
| 2025 | Winter | 103.12          | 56.88   | 14.52 | **1.8520**     | **2.4750**  | power+energy  |
| 2030 | Spring | 94.59           | 175.17  | 53.29 | 1.0369         | 2.3500      | power+energy  |
| 2030 | Summer | 111.43          | 140.49  | 37.56 | 1.2970         | 2.1500      | power+energy  |
| 2030 | Autumn | 119.91          | 82.92   | 19.36 | 1.6774         | 2.3500      | power+energy  |
| 2030 | Winter | 113.96          | 69.02   | 17.23 | 1.8520         | 2.4750      | power+energy  |
| 2035 | Spring | 137.64          | 195.91  | 69.58 | 1.2792         | 2.3500      | power+energy  |
| 2035 | Summer | 115.28          | 143.29  | 41.27 | 0.7964         | 2.1500      | power+energy  |
| 2035 | Autumn | 89.70           | 134.60  | 27.00 | 1.5384         | 2.4250      | power+energy  |
| 2035 | Winter | 136.89          | 56.17   | 13.83 | 1.8520         | 2.4750      | power+energy  |

**Year-2025, max across nodes 5/7/9** (comparable to the harness's reported EFC/day):
- `EFC*_energy = 1.8520` (node 5, and identically nodes 7/9, Winter)
- `EFC*_row3` — not computed (skipped; see below)
- `EFC*_efficiency1 (sensitivity) = 2.4750` (Winter, and identically Autumn)

**Binding constraint.** Every one of the 3×3×4 = 36 (node, year, day) cells binds
`power+energy`: on each representative day the optimal schedule saturates the converter
rating (`pch` or `pdch` = `S_INV` = 0.96875 MW) on 5-16 periods AND saturates the SoC range
(10%/90% of `E_INV`) on 4-16 periods (day-dependent; e.g. node 5 / 2025 / Winter: 13
power-bound periods, 8 energy-bound periods; node 5 / 2025 / Summer: 5 power-bound, 16
energy-bound). The day-balance row is active by construction on every day (that is what
closes the schedule, not a separate economic limiter). No day in the 2025-2035 / 4-season
grid is bound by power alone or energy alone.

**Price spread actually available.** Mean price ranges 83.60-137.64 $/MWh across the grid;
max−min ranges 56.17-195.91 $/MWh; mean absolute deviation ranges 13.83-69.58 $/MWh (all
values in the table above; every cell computed directly from
`sed.cost_energy_p[year][day][0]`, cross-checked exactly against the serialized
`price_per_mwh`).

**Verdict.** `EFC*_energy` is **order 1** (0.7964-1.8520 across the full 2025-2035 grid; the
year-2025 max across nodes is 1.8520), not order 0.1. This is materially ABOVE the retired
"1.1" figure and roughly 28× the current run's measured 0.067 (`s33e2`) and 65× the 0.0285
measured in `s31c`.

## Validation

- **Zero-solve claim armed, not asserted**: `SolveProfileGuard([])` installed for the whole
  script before any production import; `guard.verify(0)` checked before any output is
  written; observed counts `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0,
  'blocked_exec': 0}` — the guard was never triggered in either direction, confirming the
  entire pipeline (data load + 36 LP solves) touched no Pyomo/IPOPT solve path.
- **Price provenance cross-checked**: `sed.cost_energy_p[year][day][0]` vs.
  `interface_settlement_detail_s31c.json`'s `price_per_mwh`, max abs diff = 0.0 over all 864
  (node, year, day, period) cells checked — the two sources are bitwise identical for this
  quantity, confirming the LP is priced against the same market data production settles
  against.
- **Physical sanity of one LP solution inspected by hand** (node 5 / 2025 / Winter): no
  simultaneous charge+discharge in any period (round-trip loss never wasted), SoC touches
  both bounds, day balance holds exactly, `res.success = True` (HiGHS) for all 36×2 = 72 LP
  solves (variant 1 and variant 3 per cell) — no solver failure recorded.
- **Instance recorded**: `s_mva`, `e_mwh`, `invest_year`, `eff_ch`, `eff_dch` and the SoC
  fraction constants are written into the top-level `instance` block of every output
  artifact, per the evidence rule on recording the problem instance alongside every value.
- **Distinguishing what was validated**: the LP code executes correctly and reproduces the
  cited production constraints; the requested diagnostic (EFC\* benchmark) was produced for
  all 36 cells with zero solver failures. This is NOT a claim that the coordinated ADMM
  problem would itself reach this benchmark — EFC\* is by construction an UPPER BOUND (a
  price-taker ignores network constraints, ADMM coordination stiffness, and every other
  network's simultaneous dispatch), so "storage should reach order-1 EFC/day" is a necessary
  condition the Step 3.4 gate can now be checked against, not a guarantee the re-scaled ADMM
  run will attain it.

## Unexpected findings

- **`EFC*_energy` (order 1, 0.80-1.85) is not merely order-1 but is ABOVE the retired 1.1
  figure** in 8 of 12 year/day cells for the nameplate schedule (`eff_ch=0.97`,
  `eff_dch=0.96`), and above it in every cell under the efficiency-1.0 sensitivity. This
  means the "EFC/day rises from 0.067 to order 1" prediction in Addendum 15 is, if anything,
  CONSERVATIVE relative to what an unconstrained price-taker with C\*'s own rating/capacity
  would choose — the gate's target is not an overshoot of the physically achievable
  benchmark.
- **Every day binds BOTH power and energy**, never one alone. This is a mechanical
  consequence of C\*'s own geometry: filling the usable 80% SoC band (0.8 × 3.875 = 3.1 MWh)
  at the rated 0.96875 MW takes ≈3.3 hours, short relative to the 24-period day and the
  observed price cycle lengths, so every profitable price swing the LP exploits both ramps
  at the power limit AND then sits at the SoC cap/floor waiting for the next price
  reversal. This is a useful diagnostic fact for interpreting any future ADMM trajectory
  that saturates SoC but not power, or vice versa — under the price-taking benchmark neither
  should be expected alone.
- **Row-3 mapping is genuinely undefined, not merely inconvenient.** Confirmed by direct
  inspection that (a) the TSO's ADN-interface flexibility charge is excluded from every
  current solver objective (`flexibility_cost`'s explicit `continue` over
  `load_is_tso_adn_interface` loads) and (b) its counterfactual value
  (`adn_interface_flexibility_cost`) is computed ONLY for reporting, with its own docstring
  stating this; (c) the shared ESS's dispatch variables and the TSO's interface-load
  flexibility variables are physically distinct quantities related only through the coupled
  AC power flow of both networks plus the ADMM consensus — no committed linear or fixed-rate
  identity exists between them. Per the task's explicit instruction, this variant is skipped
  rather than approximated; full reasoning and the located `cost_flex` values are recorded
  in the script's `ROW3_SKIP_REASON` constant and in `report['row3_variant']`.
- **The day-balance slack.** Production allows the shared ESS's day-balance row a small,
  penalized slack by default (`params.slacks.shared_ess.day_balance = True`,
  `network_parameters.py:76`, "Candidate 4 pins `day_balance = True` above,
  unconditionally"). This benchmark enforces the row as a hard equality instead (no slack
  variable in the LP). This is a deliberate, documented simplification (see script
  docstring) rather than an oversight: a price-taker with a correctly priced arbitrage
  objective would never want to leave the day-balance row slack in a way that costs it
  future-day flexibility, since the LP already prices every unit of throughput; the
  production slack exists for FEASIBILITY under network coupling the price-taker LP does not
  have. If the Planner wants the exact production slack tolerance (`EQUALITY_TOLERANCE =
  1e-5` p.u. ≈ 0.001 MWh) reproduced instead, that is a small, identified follow-up.

## Remaining issues

- `EFC*_row3` is not computed (by design; see above). If the Planner determines a specific,
  committed mapping should govern this variant (e.g., a defined pass-through rate agreed
  with the Advisor), that would need to be specified before a follow-up computes it — this
  Worker did not invent one.
- The benchmark uses C\*'s NAMEPLATE rating/capacity for every year (2025/2030/2035), per
  the task's explicit instruction ("uniform across nodes 5/7/9"); it does not apply an SoH
  derating even though production's own ESSO model tracks `es_soh_per_unit_cumul`. This
  matches Amendment 2026-09-16's own statement that degradation is not currently priced as a
  benchmark cost, but the Planner should be aware the reported EFC\* is not adjusted for any
  year-over-year available-capacity shrinkage.
- Full per-period `pch`/`pdch`/`soc` schedules are NOT persisted to the output JSON (only
  aggregate quantities: `efc_star_energy`, `profit`, binding-period counts, price stats).
  If the Planner wants the full 24-period schedule per cell for a subsequent plot or
  cross-ADMM comparison, that is a small addition (the LP already computes them; only the
  JSON writer would need to grow).

## Questions for Planner

1. Should `EFC*_row3` be attempted under a specific, Planner-authorized mapping (e.g., a
   defined displacement identity), or is the skip (with reason recorded) sufficient for the
   gate?
2. Should a follow-up persist full per-period schedules for the highest-EFC\* cells (2025
   Winter/Autumn) to support a direct visual comparison against a future re-scaled ADMM
   trajectory?

## Provenance

- Script: `p515_s34_efc_benchmark.py` (repo root)
- Results: `data/SRP1/Results/P515S34/EFC_benchmark/efc_benchmark_results.json`
- Manifest: `data/SRP1/Results/P515S34/EFC_benchmark/sha256_manifest.json`
- Interpreter: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`
- Solver: `scipy.optimize.linprog` (`method='highs'`), scipy 1.17.0 — no Pyomo, no IPOPT.
