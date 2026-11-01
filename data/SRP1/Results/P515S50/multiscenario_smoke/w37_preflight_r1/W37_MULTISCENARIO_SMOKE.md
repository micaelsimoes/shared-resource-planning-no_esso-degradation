# P5.15 Addendum 36 W37 -- 2x2 multi-scenario smoke (w37_preflight_r1)

> **SMOKE UNDER THE CURRENT QUADRATIC SCENARIO-DEVIATION PENALTY (definitions.PENALTY_SCENARIO_DEVIATION = 9e4 on voltage + interface power, definitions.PENALTY_SHARED_ESS_SCENARIO_DEVIATION = 1e4 on the shared ESS, added to model.objective.expr and therefore OUTSIDE Q(x)); NOT THE PILOT INSTANCE; THE ROW-18 / alpha DECISION IS PENDING. Branch-execution and sanity evidence only -- not a result, not comparable with any SRP1 (1x1) figure.**

- stage: P5.15 Addendum 36 W37 -- 2x2 multi-scenario SMOKE of the >1x1 hull-polish and settlement branches (quadratic scenario-deviation penalty active; NOT the pilot)
- instance label: `w37_2x2_smoke`  (see `launch.json` for the derived case file, its sha256, and the scenario checksum)
- cycles per arm: 2  (certification NOT attempted)
- arms: 
- wall clock: 11.8 s

## Solve profile (armed bounded guard, verified exactly)

- declared strict total: **0**
- ADMM retries credited per failure event: 0
- polish retries beyond one per block: 0
- expected (gated): **0**, observed: **0**
- verify failures: `[]`
- identity: observed == declared_total_strict + sum over arms of [ADMM retries credited per failure event] + [polish inner-guard surplus over one solve per block]

## Checks

| arm | C1 multi-scenario | C1 expectation mode | C2 hull | C3 polish solved | C4 decomposition | C5 S31C | C6 prices | C7 probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Timing and memory (C8) -- preliminary, quadratic active, 1 year x 4 days

- per-solve wall time, all phases: `{'n': 0}`
- peak memory: `{'watchdog_peak': {'measure': 428310528, 'rss_tree': 428310528, 'rss_self': 428310528, 'footprint_self': 299631864, 'stage_at_peak_measure': 'rule eleven checklist', 't_at_peak_measure': 11.817, 'gating_measure': 'rss_tree', 'gating_value': 428310528, 'stage_at_peak_gating_value': 'rule eleven checklist', 't_at_peak_gating_value': 11.817, 'swap_used': 643760128, 'stage_at_peak_swap_used': 'derive case'}, 'memory_final': {'t': 11.817, 'stage': 'rule eleven checklist', 'rss_self': 428310528, 'rss_children': 0, 'n_children': 0, 'rss_tree': 428310528, 'footprint_self': 299631864, 'measure': 428310528, 'sys_available': 16106258432, 'sys_used_pct': 53.1, 'swap_used': 643760128, 'gating_measure': 'rss_tree', 'gating_value': 428310528}, 'ru_maxrss_self_bytes': 428343296, 'ru_maxrss_children_ipopt_bytes': 13418496, 'units_note': 'ru_maxrss is in BYTES on macOS'}`

## What this does and does not establish

- It establishes that the W35 item 3 >1 x 1 branches EXECUTE and produce internally consistent quantities on a real 2 x 2 instance, under the tolerances stated above.
- It does NOT establish any economic quantity, any convergence property, or anything about the pilot: the formulation it runs is the CURRENT quadratic scenario-deviation penalty, the row-18 / alpha decision is pending, and every number is at 1 representative year with 2 cycles and no certification.
