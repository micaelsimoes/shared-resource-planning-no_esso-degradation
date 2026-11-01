# P5.15 Addendum 36 W37 -- 2x2 multi-scenario smoke (w37_x0_r1)

> **SMOKE UNDER THE CURRENT QUADRATIC SCENARIO-DEVIATION PENALTY (definitions.PENALTY_SCENARIO_DEVIATION = 9e4 on voltage + interface power, definitions.PENALTY_SHARED_ESS_SCENARIO_DEVIATION = 1e4 on the shared ESS, added to model.objective.expr and therefore OUTSIDE Q(x)); NOT THE PILOT INSTANCE; THE ROW-18 / alpha DECISION IS PENDING. Branch-execution and sanity evidence only -- not a result, not comparable with any SRP1 (1x1) figure.**

- stage: P5.15 Addendum 36 W37 -- 2x2 multi-scenario SMOKE of the >1x1 hull-polish and settlement branches (quadratic scenario-deviation penalty active; NOT the pilot)
- instance label: `w37_2x2_smoke`  (see `launch.json` for the derived case file, its sha256, and the scenario checksum)
- cycles per arm: 2  (certification NOT attempted)
- arms: x0
- wall clock: 172.1 s

## Solve profile (armed bounded guard, verified exactly)

- declared strict total: **73**
- ADMM retries credited per failure event: 0
- polish retries beyond one per block: 0
- expected (gated): **73**, observed: **73**
- verify failures: `[]`
- identity: observed == declared_total_strict + sum over arms of [ADMM retries credited per failure event] + [polish inner-guard surplus over one solve per block]

## Checks

| arm | C1 multi-scenario | C1 expectation mode | C2 hull | C3 polish solved | C4 decomposition | C5 S31C | C6 prices | C7 probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x0 | True | True | True | True | True | True | True | True |

## Arm `x0`

- investment map (MVA, MWh) at year 2025: `{'5': [0.0, 0.0], '7': [0.0, 0.0], '9': [0.0, 0.0]}`
- cycles run: 2, converged at cycle: None (certification not attempted)
- objective convention: recourse = net_operational_recourse (settlement-excluded, salvage-netted); gross_operational_cost = settlement-excluded gross; both as shared_resources_planning._get_operational_recourse_components defines them
- recourse (net_operational_recourse): 386177025.5497756
- gross_operational_cost (settlement-excluded): 386177025.5497756
- arm wall clock: 160.38337182998657 s; solves in arm incl. polish: 73
- block size `TSO|2025|Spring`: {'n_var_data': 14544, 'n_constraint_data': 12432, 'n_expression_data': 6088, 'n_market_scenarios': 2, 'n_operation_scenarios': 2, 'n_periods': 24}
- block size `DSO5|2025|Spring`: {'n_var_data': 43168, 'n_constraint_data': 30744, 'n_expression_data': 16648, 'n_market_scenarios': 2, 'n_operation_scenarios': 2, 'n_periods': 24}
- C2 hull: pass=True, descriptors checked=2880, violations=0, descriptor mismatches=0
  - V: n=288, agents=2, degenerate=0, width in [0.00028280067378405604, 0.06220037906355458], ESSO binding=None, ESSO interior=None
  - PF_P: n=288, agents=2, degenerate=0, width in [3.3935062104806235e-06, 0.05452176673568547], ESSO binding=None, ESSO interior=None
  - PF_Q: n=288, agents=2, degenerate=0, width in [4.880114817529257e-07, 0.00960847971971579], ESSO binding=None, ESSO interior=None
  - ESS_P: n=288, agents=3, degenerate=0, width in [3.2856887046555475e-27, 3.285689037810342e-27], ESSO binding=288, ESSO interior=0
  - ESS_Q: n=288, agents=3, degenerate=288, width in [0.0, 0.0], ESSO binding=288, ESSO interior=0
- C4 decomposition: pass=True, worst relative=2.648202043638781e-16, worst family-expectation relative=0.0, tolerance=1e-09
- C5 S31C: pass=True, lhs=-6102119.563154459, rhs=-6102119.563154436, relative=3.8155700039005636e-15, tolerance=1e-09
- C6 prices: pass=True, entries=288, bad=0, worst relative=0.0, market spread={'min': np.float64(0.0735437352917927), 'median': np.float64(4.784916890998446), 'max': np.float64(52.77731134287096), 'n_zero': 0}
- C3 polish: blocks=16, all_solved=True, failed=[], solve profile={'observed': {'permitted_solve': 16, 'permitted_exec': 16, 'blocked_solve': 0, 'blocked_exec': 0}, 'n_blocks_dispatched': 16, 'retries_beyond_one_per_block': 0, 'blocked_calls': 0}
  - gate caveat: the s41 gate threshold and FLAG_ABS are SRP1 C* constants (1e-6 x 650,966,975.29); they are reported here for provenance and are NOT a meaningful gate on this instance. Delta <= 0 also need not hold above 1 x 1: the polish minimises model.objective, whose .expr carries the quadratic scenario-deviation penalty, while Delta is measured on objective_function_rule, which does not.
- C7 apply_common_values probe: raised=NotImplementedError, pass=True
- solve times (cumulative, all phases so far): {'n': 64, 'total_s': 145.83605909347534, 'mean_s': 2.278688423335552, 'min_s': 0.34202003479003906, 'median_s': 2.4579031467437744, 'max_s': 6.844059705734253}

## Timing and memory (C8) -- preliminary, quadratic active, 1 year x 4 days

- per-solve wall time, all phases: `{'n': 64, 'total_s': 145.83605909347534, 'mean_s': 2.278688423335552, 'min_s': 0.34202003479003906, 'median_s': 2.4579031467437744, 'max_s': 6.844059705734253}`
  - `x0: hull polish (apply_hull_bounds + re-solve every block)`: {'n': 16, 'total_s': 44.77162218093872, 'mean_s': 2.79822638630867, 'min_s': 0.34202003479003906, 'median_s': 2.8701171875, 'max_s': 6.844059705734253}
  - `x0: run_admm_arm (initialization + 2 cycles)`: {'n': 48, 'total_s': 101.06443691253662, 'mean_s': 2.105509102344513, 'min_s': 0.3866868019104004, 'median_s': 2.4144582748413086, 'max_s': 4.7579851150512695}
- peak memory: `{'watchdog_peak': {'measure': 2363686912, 'rss_tree': 2363686912, 'rss_self': 2208890880, 'footprint_self': 1659767256, 'stage_at_peak_measure': 'x0: run_admm_arm (initialization + 2 cycles)', 't_at_peak_measure': 119.313, 'gating_measure': 'rss_tree', 'gating_value': 2363686912, 'stage_at_peak_gating_value': 'x0: run_admm_arm (initialization + 2 cycles)', 't_at_peak_gating_value': 119.313, 'swap_used': 643760128, 'stage_at_peak_swap_used': 'derive case'}, 'memory_final': {'t': 172.08, 'stage': 'rule eleven checklist', 'rss_self': 2068217856, 'rss_children': 0, 'n_children': 0, 'rss_tree': 2068217856, 'footprint_self': 1511672304, 'measure': 2068217856, 'sys_available': 15285321728, 'sys_used_pct': 55.5, 'swap_used': 643760128, 'gating_measure': 'rss_tree', 'gating_value': 2068217856}, 'ru_maxrss_self_bytes': 2232729600, 'ru_maxrss_children_ipopt_bytes': 173686784, 'units_note': 'ru_maxrss is in BYTES on macOS'}`

## What this does and does not establish

- It establishes that the W35 item 3 >1 x 1 branches EXECUTE and produce internally consistent quantities on a real 2 x 2 instance, under the tolerances stated above.
- It does NOT establish any economic quantity, any convergence property, or anything about the pilot: the formulation it runs is the CURRENT quadratic scenario-deviation penalty, the row-18 / alpha decision is pending, and every number is at 1 representative year with 2 cycles and no certification.
