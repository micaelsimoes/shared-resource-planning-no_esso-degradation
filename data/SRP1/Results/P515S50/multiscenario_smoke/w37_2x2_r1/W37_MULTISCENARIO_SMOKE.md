# P5.15 Addendum 36 W37 -- 2x2 multi-scenario smoke (w37_2x2_r1)

> **SMOKE UNDER THE CURRENT QUADRATIC SCENARIO-DEVIATION PENALTY (definitions.PENALTY_SCENARIO_DEVIATION = 9e4 on voltage + interface power, definitions.PENALTY_SHARED_ESS_SCENARIO_DEVIATION = 1e4 on the shared ESS, added to model.objective.expr and therefore OUTSIDE Q(x)); NOT THE PILOT INSTANCE; THE ROW-18 / alpha DECISION IS PENDING. Branch-execution and sanity evidence only -- not a result, not comparable with any SRP1 (1x1) figure.**

- stage: P5.15 Addendum 36 W37 -- 2x2 multi-scenario SMOKE of the >1x1 hull-polish and settlement branches (quadratic scenario-deviation penalty active; NOT the pilot)
- instance label: `w37_2x2_smoke`  (see `launch.json` for the derived case file, its sha256, and the scenario checksum)
- cycles per arm: 2  (certification NOT attempted)
- arms: x0
- wall clock: 11.6 s

## Solve profile (armed bounded guard, verified exactly)

- declared strict total: **73**
- ADMM retries credited per failure event: 0
- polish retries beyond one per block: 0
- expected (gated): **None**, observed: **0**
- verify failures: `['event-level reconciliation UNSUPPORTED: [{"arm": "x0", "reconciliation": {}}]']`
- identity: observed == declared_total_strict + sum over arms of [ADMM retries credited per failure event] + [polish inner-guard surplus over one solve per block]

## Checks

| arm | C1 multi-scenario | C1 expectation mode | C2 hull | C3 polish solved | C4 decomposition | C5 S31C | C6 prices | C7 probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x0 | None | None | None | None | None | None | None | None |

## Arm `x0`

- investment map (MVA, MWh) at year 2025: `{'5': [0.0, 0.0], '7': [0.0, 0.0], '9': [0.0, 0.0]}`
- cycles run: None, converged at cycle: None (certification not attempted)
- objective convention: None
- recourse (net_operational_recourse): None
- gross_operational_cost (settlement-excluded): None
- arm wall clock: None s; solves in arm incl. polish: None
- C2 hull: pass=None, descriptors checked=None, violations=None, descriptor mismatches=None
- C4 decomposition: pass=None, worst relative=None, worst family-expectation relative=None, tolerance=None
- C5 S31C: pass=None, lhs=None, rhs=None, relative=None, tolerance=None
- C6 prices: pass=None, entries=None, bad=None, worst relative=None, market spread=None
- C3 polish: blocks=None, all_solved=None, failed=None, solve profile=None
  - gate caveat: None
- C7 apply_common_values probe: raised=None, pass=None
- solve times (cumulative, all phases so far): None

## Timing and memory (C8) -- preliminary, quadratic active, 1 year x 4 days

- per-solve wall time, all phases: `{'n': 0, 'by_agent': {'TSO': {'n': 0}, 'DSO': {'n': 0}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}`
- peak memory: `{'watchdog_peak': {'measure': 419430400, 'rss_tree': 419430400, 'rss_self': 419430400, 'footprint_self': 308446432, 'stage_at_peak_measure': 'rule eleven checklist', 't_at_peak_measure': 11.576, 'gating_measure': 'rss_tree', 'gating_value': 419430400, 'stage_at_peak_gating_value': 'rule eleven checklist', 't_at_peak_gating_value': 11.576, 'swap_used': 643760128, 'stage_at_peak_swap_used': 'derive case'}, 'memory_final': {'t': 11.576, 'stage': 'rule eleven checklist', 'rss_self': 419430400, 'rss_children': 0, 'n_children': 0, 'rss_tree': 419430400, 'footprint_self': 308446432, 'measure': 419430400, 'sys_available': 16314433536, 'sys_used_pct': 52.5, 'swap_used': 643760128, 'gating_measure': 'rss_tree', 'gating_value': 419430400}, 'ru_maxrss_self_bytes': 419463168, 'ru_maxrss_children_ipopt_bytes': 13418496, 'units_note': 'ru_maxrss is in BYTES on macOS'}`

## What this does and does not establish

- It establishes that the W35 item 3 >1 x 1 branches EXECUTE and produce internally consistent quantities on a real 2 x 2 instance, under the tolerances stated above.
- It does NOT establish any economic quantity, any convergence property, or anything about the pilot: the formulation it runs is the CURRENT quadratic scenario-deviation penalty, the row-18 / alpha decision is pending, and every number is at 1 representative year with 2 cycles and no certification.
