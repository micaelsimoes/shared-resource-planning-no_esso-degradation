# P5.15 Addendum 36 W37 -- 2x2 multi-scenario smoke (w37_2x2_r2)

> **SMOKE UNDER THE CURRENT QUADRATIC SCENARIO-DEVIATION PENALTY (definitions.PENALTY_SCENARIO_DEVIATION = 9e4 on voltage + interface power, definitions.PENALTY_SHARED_ESS_SCENARIO_DEVIATION = 1e4 on the shared ESS, added to model.objective.expr and therefore OUTSIDE Q(x)); NOT THE PILOT INSTANCE; THE ROW-18 / alpha DECISION IS PENDING. Branch-execution and sanity evidence only -- not a result, not comparable with any SRP1 (1x1) figure.**

- stage: P5.15 Addendum 36 W37 -- 2x2 multi-scenario SMOKE of the >1x1 hull-polish and settlement branches (quadratic scenario-deviation penalty active; NOT the pilot)
- instance label: `w37_2x2_smoke`  (see `launch.json` for the derived case file, its sha256, and the scenario checksum)
- cycles per arm: 2  (certification NOT attempted)
- arms: x0, unit
- wall clock: 378.1 s

## Solve profile (armed bounded guard, verified exactly)

- declared strict total: **146**
- ADMM retries credited per failure event: 0
- polish retries beyond one per block: 0
- expected (gated): **146**, observed: **146**
- verify failures: `[]`
- identity: observed == declared_total_strict + sum over arms of [ADMM retries credited per failure event] + [polish inner-guard surplus over one solve per block]

## Checks

| arm | C1 multi-scenario | C1 expectation mode | C2 hull | C3 polish solved | C4 decomposition | C5 S31C | C6 prices | C7 probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x0 | True | True | True | True | True | True | True | True |
| unit | True | True | True | True | True | True | True | True |

## Arm `x0`

- investment map (MVA, MWh) at year 2025: `{'5': [0.0, 0.0], '7': [0.0, 0.0], '9': [0.0, 0.0]}`
- cycles run: 2, converged at cycle: None (certification not attempted)
- objective convention: recourse = net_operational_recourse (settlement-excluded, salvage-netted); gross_operational_cost = settlement-excluded gross; both as shared_resources_planning._get_operational_recourse_components defines them
- recourse (net_operational_recourse): 386177025.5497756
- gross_operational_cost (settlement-excluded): 386177025.5497756
- arm wall clock: 160.88740134239197 s; solves in arm incl. polish: 73
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
- C6 prices: pass=True, entries=288, bad=0, worst relative=0.0, market spread={'min': 0.0735437352917927, 'median': 4.784916890998446, 'max': 52.77731134287096, 'n_zero': 0}
- C3 polish: blocks=16, all_solved=True, failed=[], solve profile={'observed': {'permitted_solve': 16, 'permitted_exec': 16, 'blocked_solve': 0, 'blocked_exec': 0}, 'n_blocks_dispatched': 16, 'retries_beyond_one_per_block': 0, 'blocked_calls': 0}
  - gate caveat: the s41 gate threshold and FLAG_ABS are SRP1 C* constants (1e-6 x 650,966,975.29); they are reported here for provenance and are NOT a meaningful gate on this instance. Delta <= 0 also need not hold above 1 x 1: the polish minimises model.objective, whose .expr carries the quadratic scenario-deviation penalty, while Delta is measured on objective_function_rule, which does not.
- C7 apply_common_values probe: raised=NotImplementedError, pass=True
- solve times (cumulative, all phases so far): {'n': 64, 'total_s': 146.39632177352905, 'mean_s': 2.2874425277113914, 'min_s': 0.33987879753112793, 'median_s': 2.4519710540771484, 'max_s': 6.974586248397827, 'by_agent': {'TSO': {'n': 16, 'total_s': 6.587923526763916, 'mean_s': 0.41174522042274475, 'min_s': 0.33987879753112793, 'median_s': 0.41472887992858887, 'max_s': 0.4806079864501953}, 'DSO': {'n': 48, 'total_s': 139.80839824676514, 'mean_s': 2.9126749634742737, 'min_s': 1.5865087509155273, 'median_s': 2.6189188957214355, 'max_s': 6.974586248397827}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}

## Arm `unit`

- investment map (MVA, MWh) at year 2025: `{'5': [0.0, 0.0], '7': [0.25, 1.0], '9': [0.0, 0.0]}`
- cycles run: 2, converged at cycle: None (certification not attempted)
- objective convention: recourse = net_operational_recourse (settlement-excluded, salvage-netted); gross_operational_cost = settlement-excluded gross; both as shared_resources_planning._get_operational_recourse_components defines them
- recourse (net_operational_recourse): 387759644.2431016
- gross_operational_cost (settlement-excluded): 387861865.13254035
- arm wall clock: 205.54023909568787 s; solves in arm incl. polish: 73
- block size `TSO|2025|Spring`: {'n_var_data': 14544, 'n_constraint_data': 12432, 'n_expression_data': 6088, 'n_market_scenarios': 2, 'n_operation_scenarios': 2, 'n_periods': 24}
- block size `DSO5|2025|Spring`: {'n_var_data': 43168, 'n_constraint_data': 30744, 'n_expression_data': 16648, 'n_market_scenarios': 2, 'n_operation_scenarios': 2, 'n_periods': 24}
- C2 hull: pass=True, descriptors checked=2880, violations=0, descriptor mismatches=0
  - V: n=288, agents=2, degenerate=0, width in [0.00017829086061937893, 0.06272321739785558], ESSO binding=None, ESSO interior=None
  - PF_P: n=288, agents=2, degenerate=0, width in [7.795382514719229e-06, 0.056653307807875496], ESSO binding=None, ESSO interior=None
  - PF_Q: n=288, agents=2, degenerate=0, width in [4.935798141036951e-08, 0.0160659854774359], ESSO binding=None, ESSO interior=None
  - ESS_P: n=288, agents=3, degenerate=0, width in [3.2856887046555475e-27, 0.0008106241929040513], ESSO binding=235, ESSO interior=53
  - ESS_Q: n=288, agents=3, degenerate=192, width in [0.0, 4.375296956358981e-05], ESSO binding=279, ESSO interior=9
- C4 decomposition: pass=True, worst relative=4.670365183243514e-16, worst family-expectation relative=0.0, tolerance=1e-09
- C5 S31C: pass=True, lhs=-6368661.318145931, rhs=-6368661.3181457985, relative=2.0765400920063125e-14, tolerance=1e-09
- C6 prices: pass=True, entries=288, bad=0, worst relative=0.0, market spread={'min': 0.0735437352917927, 'median': 4.784916890998446, 'max': 52.77731134287096, 'n_zero': 0}
- C3 polish: blocks=16, all_solved=True, failed=[], solve profile={'observed': {'permitted_solve': 16, 'permitted_exec': 16, 'blocked_solve': 0, 'blocked_exec': 0}, 'n_blocks_dispatched': 16, 'retries_beyond_one_per_block': 0, 'blocked_calls': 0}
  - gate caveat: the s41 gate threshold and FLAG_ABS are SRP1 C* constants (1e-6 x 650,966,975.29); they are reported here for provenance and are NOT a meaningful gate on this instance. Delta <= 0 also need not hold above 1 x 1: the polish minimises model.objective, whose .expr carries the quadratic scenario-deviation penalty, while Delta is measured on objective_function_rule, which does not.
- C7 apply_common_values probe: raised=NotImplementedError, pass=True
- solve times (cumulative, all phases so far): {'n': 128, 'total_s': 333.18700671195984, 'mean_s': 2.6030234899371862, 'min_s': 0.33987879753112793, 'median_s': 2.509152889251709, 'max_s': 14.398983716964722, 'by_agent': {'TSO': {'n': 32, 'total_s': 17.123730421066284, 'mean_s': 0.5351165756583214, 'min_s': 0.33987879753112793, 'median_s': 0.48718905448913574, 'max_s': 0.9701199531555176}, 'DSO': {'n': 96, 'total_s': 316.06327629089355, 'mean_s': 3.292325794696808, 'min_s': 1.5865087509155273, 'median_s': 2.8670222759246826, 'max_s': 14.398983716964722}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}

## Timing and memory (C8) -- preliminary, quadratic active, 1 year x 4 days

- per-solve wall time, all phases: `{'n': 128, 'total_s': 333.18700671195984, 'mean_s': 2.6030234899371862, 'min_s': 0.33987879753112793, 'median_s': 2.509152889251709, 'max_s': 14.398983716964722, 'by_agent': {'TSO': {'n': 32, 'total_s': 17.123730421066284, 'mean_s': 0.5351165756583214, 'min_s': 0.33987879753112793, 'median_s': 0.48718905448913574, 'max_s': 0.9701199531555176}, 'DSO': {'n': 96, 'total_s': 316.06327629089355, 'mean_s': 3.292325794696808, 'min_s': 1.5865087509155273, 'median_s': 2.8670222759246826, 'max_s': 14.398983716964722}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}`
  - `unit: hull polish (apply_hull_bounds + re-solve every block)`: {'n': 16, 'total_s': 61.12107753753662, 'mean_s': 3.820067346096039, 'min_s': 0.48718905448913574, 'median_s': 3.6217288970947266, 'max_s': 14.398983716964722, 'by_agent': {'TSO': {'n': 4, 'total_s': 2.015944004058838, 'mean_s': 0.5039860010147095, 'min_s': 0.48718905448913574, 'median_s': 0.5034818649291992, 'max_s': 0.5371739864349365}, 'DSO': {'n': 12, 'total_s': 59.10513353347778, 'mean_s': 4.925427794456482, 'min_s': 2.6253578662872314, 'median_s': 3.889150857925415, 'max_s': 14.398983716964722}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}
  - `unit: run_admm_arm (initialization + 2 cycles)`: {'n': 48, 'total_s': 125.66960740089417, 'mean_s': 2.6181168208519616, 'min_s': 0.5217020511627197, 'median_s': 2.473784923553467, 'max_s': 6.872145891189575, 'by_agent': {'TSO': {'n': 12, 'total_s': 8.51986289024353, 'mean_s': 0.7099885741869608, 'min_s': 0.5217020511627197, 'median_s': 0.7666330337524414, 'max_s': 0.9701199531555176}, 'DSO': {'n': 36, 'total_s': 117.14974451065063, 'mean_s': 3.2541595697402954, 'min_s': 1.6449220180511475, 'median_s': 2.9557549953460693, 'max_s': 6.872145891189575}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}
  - `x0: hull polish (apply_hull_bounds + re-solve every block)`: {'n': 16, 'total_s': 46.1797776222229, 'mean_s': 2.8862361013889313, 'min_s': 0.33987879753112793, 'median_s': 2.9144339561462402, 'max_s': 6.974586248397827, 'by_agent': {'TSO': {'n': 4, 'total_s': 1.4549586772918701, 'mean_s': 0.36373966932296753, 'min_s': 0.33987879753112793, 'median_s': 0.3634788990020752, 'max_s': 0.3981132507324219}, 'DSO': {'n': 12, 'total_s': 44.72481894493103, 'mean_s': 3.727068245410919, 'min_s': 2.2635867595672607, 'median_s': 3.035456895828247, 'max_s': 6.974586248397827}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}
  - `x0: run_admm_arm (initialization + 2 cycles)`: {'n': 48, 'total_s': 100.21654415130615, 'mean_s': 2.087844669818878, 'min_s': 0.3851630687713623, 'median_s': 2.4038658142089844, 'max_s': 4.752590894699097, 'by_agent': {'TSO': {'n': 12, 'total_s': 5.132964849472046, 'mean_s': 0.42774707078933716, 'min_s': 0.3851630687713623, 'median_s': 0.43280911445617676, 'max_s': 0.4806079864501953}, 'DSO': {'n': 36, 'total_s': 95.0835793018341, 'mean_s': 2.6412105361620584, 'min_s': 1.5865087509155273, 'median_s': 2.584804058074951, 'max_s': 4.752590894699097}}, 'scope_note': 'NETWORK solves only -- wrapped at network.Network.run_smopf. The ESSO solves (3 per round) go through shared_energy_storage_data._run_solver_attempt and are NOT in this table; they ARE in the guard counts.'}
- peak memory: `{'watchdog_peak': {'measure': 2920841216, 'rss_tree': 2920841216, 'rss_self': 2242363392, 'footprint_self': 1693665680, 'stage_at_peak_measure': 'unit: hull polish (apply_hull_bounds + re-solve every block)', 't_at_peak_measure': 371.015, 'gating_measure': 'rss_tree', 'gating_value': 2920841216, 'stage_at_peak_gating_value': 'unit: hull polish (apply_hull_bounds + re-solve every block)', 't_at_peak_gating_value': 371.015, 'swap_used': 643760128, 'stage_at_peak_swap_used': 'derive case'}, 'memory_final': {'t': 378.082, 'stage': 'rule eleven checklist', 'rss_self': 2130018304, 'rss_children': 0, 'n_children': 0, 'rss_tree': 2130018304, 'footprint_self': 1555450376, 'measure': 2130018304, 'sys_available': 15361703936, 'sys_used_pct': 55.3, 'swap_used': 643760128, 'gating_measure': 'rss_tree', 'gating_value': 2130018304}, 'ru_maxrss_self_bytes': 2263449600, 'ru_maxrss_children_ipopt_bytes': 778502144, 'units_note': 'ru_maxrss is in BYTES on macOS'}`

## What this does and does not establish

- It establishes that the W35 item 3 >1 x 1 branches EXECUTE and produce internally consistent quantities on a real 2 x 2 instance, under the tolerances stated above.
- It does NOT establish any economic quantity, any convergence property, or anything about the pilot: the formulation it runs is the CURRENT quadratic scenario-deviation penalty, the row-18 / alpha decision is pending, and every number is at 1 representative year with 2 cycles and no certification.
