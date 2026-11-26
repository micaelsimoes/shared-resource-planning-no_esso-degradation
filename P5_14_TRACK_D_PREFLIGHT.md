# Track D — pre-flight checks

**Zero solves, enforced (0 / 0 / 0 / 0). Nothing has been run. Stopping for the budget
value and your approval.**

Evidence: `data/SRP1/Results/P514D/d_preflight.json`. Plan under test, specified
explicitly and **not** from `_build_positive_bootstrap_candidate`: **1.00 MVA / 4.00 MWh
per active distribution node (5, 7, 9), invested in 2025 only**, zero in 2030 and 2035,
single ageing cohort.

## Check 1 — investment cost, and the budget proposal

By the production formula (`shared_resources_planning.py:1620-1626`), discount factor
**0.02**, market-scenario probabilities **[0.35, 0.55, 0.10]**, 2025 unit costs:

| scenario | power | energy |
|---|---|---|
| 0 | 214,171.33 | 169,706.26 |
| 1 | 267,528.72 | 211,985.89 |
| 2 | 342,165.62 | 271,127.09 |

**Per node: 1,068,725.89. Total discounted expected: 3,206,177.68.**

Against the current `budget = 1.0e6` the plan is **short by 2,206,177.68** — the plan is
3.21x the present budget.

**Proposed override value: `budget = 4.0e6`.**
Headroom **793,822.32 above the plan, i.e. 24.8%**. Alternatives, so the choice is yours:

| value | headroom over 3,206,177.68 | comment |
|---|---|---|
| 3.5e6 | 293,822 (9.2%) | tight; a later capacity tweak could breach it |
| **4.0e6** | **793,822 (24.8%)** | proposed — round, comfortable, still of the plan's order |
| 5.0e6 | 1,793,822 (55.9%) | generous; further from the plan's scale |

**I am not choosing. Awaiting your value.**

### And a finding that lowers the stakes of that choice

**On the cold path the budget is never consulted.** Cells 2 and 3 run through
`run_operational_planning(type='distributed', candidate_solution=...)`, which goes directly
to `_run_operational_planning` (`:64-72`) without calling
`_check_candidate_first_stage_feasibility`. The validity check is reached only from the
Benders/oracle paths (`:1254`, `:1707`, `:1816`, and `p56a_oracle.check_master_feasibility`).
The hard constraint (`shared_energy_storage_data.py:309`) and the `alpha` lower bound
(`:260`) are built **only inside the master problem**, which a fixed-candidate recourse
evaluation never builds.

So for these three cells the override is **inert** — no code path reads it. I recommend
applying it anyway: it costs nothing, it keeps the recorded configuration consistent with a
plan that would be admissible, and it protects any later step that does reach the gate (a
warm cross-check through `p56a_oracle.evaluate` would). Its inertness is recorded as a
measured fact rather than assumed either way.

Your third point is confirmed and worth keeping: the budget enters the oracle's config
fingerprint at `p56a_oracle.py:739-741`, so an override changes that hash for anything
built through the oracle. The cold cells build no template, so nothing preserved is
affected.

## Check 2 — capacity and ratio bounds

| quantity | value | bound | margin |
|---|---|---|---|
| energy per node | 4.00 | `max_capacity` 5.00 | **1.00 absolute, 80.0% of cap** |
| E/S ratio | 4.00 | `[2.00, 10.00]` | mid-range, **within bounds** |

Operators, as they matter: the candidate check is `total_capacity['e'] > max_capacity +
1e-8` (`shared_resources_planning.py:1631`), and the master adds `es_e_rated <=
max_capacity` (`shared_energy_storage_data.py:282`). **Strictly inside on both**, so no
boundary multiplier is active — your intended property holds.

`benders.positive_bootstrap.energy_to_power_ratio` is pinned at **2.0**, confirming that
the plan's ratio of 4 is a deliberate departure from the two-hour floor.

## Check 3 — SoH trajectory, and a risk you should see

`k = 11541.56` (C3 active), floor `soh_min = 0.50`, three five-year blocks, one cohort
invested 2025:

| cycling | 2025 block | 2030 block | 2035 block | floor active? |
|---|---|---|---|---|
| 0.5 EFC/day | 0.9240 | 0.8537 | **0.7888** | no |
| **1.0 EFC/day** | 0.8537 | 0.7289 | **0.6223** | **no** |
| 1.5 EFC/day | 0.7888 | 0.6223 | **0.4909** | **YES** |

Your ~62% at one equivalent full cycle per day is confirmed exactly: **0.6223**.

**But the floor is not unconditionally inactive.** It binds at an average of **1.4611
EFC/day** or above (8,000 EFC to the floor over 15 years). A 4 MWh / 1 MVA unit is a
four-hour device, so 1.46 equivalent full cycles a day is physically reachable, and the
arbitrage role is precisely what would drive it there. **This cannot be excluded a priori
and must be read off cell 3's output.** If it binds, it adds the floor constraint and its
penalized slack to cell 3 only — exactly the asymmetry you flagged, which would make the
cell-3-minus-cell-2 difference an artifact rather than a cost.

## Check 4 — cell 1 is not an ADMM evaluation

`type='uncoordinated'` reaches `_run_operational_planning_without_coordination`
(`shared_resources_planning.py:5816`): one `distribution_network.optimize` per node
(`:5866`) and one `transmission_network.optimize` (`:5931`). **No consensus loop and no
ESSO solve.**

| | solves |
|---|---|
| DSO | 3 nodes x 12 year-day blocks = **36** |
| TSO | 12 year-day blocks = **12** |
| **total** | **48, single pass** |

**Cell 1 costs ~48 solves, not 3,519** — a 73x overestimate in the campaign figure. Revised
campaign estimate: `48 + cell 2 + cell 3`, with cells 2 and 3 unknown until measured at
real capacity (the transfer caveat), against 3,519 at negligible capacity as the reference.

## Check 5 — which budget paths the evaluation reaches

| path | reached by a fixed-candidate recourse evaluation? |
|---|---|
| hard constraint `investment_cost_total <= budget` (`shared_energy_storage_data.py:309`) | **no** — master only |
| `alpha.setlb(-budget * 1e3)` (`:260`) | **no** — master only |
| validity check -> `'investment budget violated'` (`shared_resources_planning.py:1634`) | **no** on the cold path; **yes** via the oracle/Benders paths |
| config fingerprint (`p56a_oracle.py:739`) | only for oracle-built artifacts |

## Summary

Checks 2, 4 and 5 pass cleanly. Check 1 gives a cost of **3,206,177.68** and a proposed
override of **4.0e6** for your approval. Check 3 passes at the expected cycling rate but
carries a **conditional risk**: the SoH floor binds at ≥1.46 EFC/day, which this plan can
reach.

**Nothing runs until you approve the budget value.**
