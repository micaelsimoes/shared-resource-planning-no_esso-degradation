# P5.15 Addendum 40 ruling 3 W52 -- 3 x 3 selection rule and R prediction (spec v23; zero solves)

Started 2026-09-24T08:57:29.109408+00:00; HEAD e1f98be8360697125fbd9c71e54f6483dd5e3f81; script sha256 f129a01e9e6492c3ab668cc8b075e4048c5fd7f7a0ff9269ec73899babb607d8.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 3; data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json ruling3_3x3 (spec sha256 39a07fd85bd02aaee0f4042d9a7b201cf4df9027191ff70d7939b124061f685c).

**Zero solves** (armed SolveProfileGuard(permitted=()), verify(0) failures: []; counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}). **No model construction** (blocked calls: []; blockers ['Network.build_model', 'NetworkData.build_model', 'SharedEnergyStorageData.build_subproblem', 'SharedEnergyStorageData.build_master_problem']). Stop point hits: 2 (declared 2).

Instances: srp1: data/SRP1/SRP1.json checksum 5a02b77ccbbbbbb8... match True; paper: data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json checksum 1e8bdd3e5233442a... match True

## 1. Market scenarios (3 of 5)

Spread definition: per representative (year, day) and price vector c (24 hourly EUR/MWh): max_minus_min = max_p c_p - min_p c_p; top4_mean_minus_bottom4_mean = mean of the 4 highest c_p minus mean of the 4 lowest (W19 Z4 definition, p515_s46_zero_solve_reports.py @ dd3afa6e). Horizon average = sum_(y,d) w * spread / sum_(y,d) w with w = num_years[y] * num_days[d] (undiscounted) or w = num_years[y] * num_days[d] / 1.02^(y - 2025) (model block weight, production DiscountFactor 0.02).

Renormalization: pm_S[s] = pm[s] / sum_(s in S) pm[s] (= 1/3 each, pm = [0.2] x 5).

Rule: rank the 10 three-subsets S of the market-scenario indices {1..5} by |spread_u(S) - TARGET| ascending, spread_u = horizon (undiscounted day-weighted) 4 h spread of the probability-renormalized mean price profile, TARGET = W22 five-scenario mean-profile spread (undiscounted); ties (|diff| equal within 1e-09) broken by |spread_r2(S) - TARGET_r2|, then by the lexicographic order of S; select the first.

TARGET (W22 five-scenario mean profile, undiscounted) = 91.72775939691898; TARGET_r2 = 90.9307562006712.

| rank | subset | 4 h spread undisc. | spread - target | 4 h spread r2 | growth-norm. | avg of scen. spreads | R_u | R_r2 | R_gn |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [2, 3, 4] | 91.8964 | +0.1687 | 91.1141 | 78.8328 | 93.6401 | 0.9378 | 0.9391 | 0.9169 |
| 2 | [1, 3, 5] | 91.9169 | +0.1891 | 91.1130 | 78.8276 | 94.7270 | 0.9380 | 0.9391 | 0.9169 |
| 3 | [1, 3, 4] | 91.4068 | -0.3209 | 90.6063 | 78.3892 | 93.8005 | 0.9328 | 0.9339 | 0.9118 |
| 4 | [1, 2, 3] | 91.2994 | -0.4284 | 90.5314 | 78.3307 | 94.0026 | 0.9317 | 0.9331 | 0.9111 |
| 5 | [1, 2, 4] | 92.2034 | +0.4756 | 91.4174 | 79.0959 | 94.7888 | 0.9409 | 0.9423 | 0.9200 |
| 6 | [1, 2, 5] | 92.4831 | +0.7553 | 91.7130 | 79.3557 | 95.7153 | 0.9438 | 0.9453 | 0.9230 |
| 7 | [1, 4, 5] | 92.5043 | +0.7766 | 91.7042 | 79.3418 | 95.5132 | 0.9440 | 0.9452 | 0.9229 |
| 8 | [3, 4, 5] | 92.7045 | +0.9768 | 91.8862 | 79.4948 | 94.3646 | 0.9461 | 0.9471 | 0.9246 |
| 9 | [2, 3, 5] | 92.8715 | +1.1438 | 92.0601 | 79.6470 | 94.5667 | 0.9478 | 0.9489 | 0.9264 |
| 10 | [2, 4, 5] | 93.3382 | +1.6105 | 92.4772 | 79.9987 | 95.3528 | 0.9525 | 0.9532 | 0.9305 |
| (all 5) | [1, 2, 3, 4, 5] | 91.7278 | +0.0000 | 90.9308 | 78.6712 | 94.6472 | 0.9361 | 0.9373 | 0.9151 |

**Selected market scenarios: [2, 3, 4].** Tie at the top: none (tie-break used: False); margin to runner-up in |diff|: 0.0205 EUR/MWh.

## 2. Operation scenarios (3 of 5)

Rule: rank the five operation-scenario indices by the PRIMARY metric (system horizon RES availability, MWh/day, undiscounted day weights, TSO + 3 DSOs, PV + Wind pg * baseMVA summed over 24 h) ascending; select (lowest, median = 3rd of 5, highest). Secondary metrics are reported, not used.

Network counts: {"TSO_case9": {"n_pv": 3, "n_wind": 3, "n_loads": 3, "baseMVA": 100.0, "num_instants": 24}, "DSO_node5_case33_1": {"n_pv": 3, "n_wind": 1, "n_loads": 32, "baseMVA": 100.0, "num_instants": 24}, "DSO_node7_case33_2": {"n_pv": 2, "n_wind": 2, "n_loads": 32, "baseMVA": 100.0, "num_instants": 24}, "DSO_node9_case33_3": {"n_pv": 3, "n_wind": 1, "n_loads": 32, "baseMVA": 100.0, "num_instants": 24}}

| rank | index | RES MWh/day (u) | RES r2 | PV | Wind | load MWh/day (u) | RES share of load | net load |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 3,594.58 | 3,545.90 | 1,174.72 | 2,419.86 | 11,655.75 | 0.3084 | 8,061.17 |
| 2 | 3 | 3,598.38 | 3,548.86 | 1,177.17 | 2,421.21 | 11,633.86 | 0.3093 | 8,035.48 |
| 3 | 5 | 3,608.25 | 3,554.63 | 1,177.20 | 2,431.05 | 11,654.44 | 0.3096 | 8,046.20 |
| 4 | 2 | 3,608.38 | 3,556.21 | 1,180.23 | 2,428.15 | 11,629.26 | 0.3103 | 8,020.88 |
| 5 | 1 | 3,636.83 | 3,584.03 | 1,179.47 | 2,457.36 | 11,676.41 | 0.3115 | 8,039.58 |

**Selected operation scenarios (low, mid, high RES): [4, 5, 1].**

Orders (ascending) by metric: {"system_res_MWh_per_day_u": [4, 3, 5, 2, 1], "system_res_MWh_per_day_r2": [4, 3, 5, 2, 1], "system_res_share_of_load_u": [4, 3, 5, 2, 1], "system_net_load_MWh_per_day_u": [2, 3, 1, 5, 4], "system_load_MWh_per_day_u": [2, 3, 5, 4, 1], "system_pv_MWh_per_day_u": [4, 3, 5, 1, 2], "system_wind_MWh_per_day_u": [4, 3, 2, 5, 1]}

Per-network RES orders (ascending): {"TSO_case9": [4, 5, 3, 1, 2], "DSO_node5_case33_1": [2, 1, 3, 4, 5], "DSO_node7_case33_2": [4, 3, 5, 2, 1], "DSO_node9_case33_3": [3, 2, 5, 1, 4]}

Relative range (max - min) / mean across the five: {"system_res_MWh_per_day_u": 0.01171, "system_load_MWh_per_day_u": 0.00405, "system_res_share_of_load_u": 0.00992, "system_net_load_MWh_per_day_u": 0.00501}

Per-block ranks on system RES: Kendall W = 0.0140 (W = 1: every block ranks the indices identically; W ~ 0: no agreement. Under independent per-block draws the expectation is about 1/m = 0.050.); rank sums [60, 64, 60, 62, 54]; consistency of the selection: {"blocks_where_selected_lowest_is_lowest": 4, "blocks_where_selected_highest_is_highest": 4, "blocks_where_selected_median_is_3rd": 7, "blocks_where_selected_low_below_selected_high": 9, "num_blocks": 20}

| block | ranks of indices 1..5 (1 = lowest RES) |
|---|---|
| 2025/Spring | [2, 5, 4, 1, 3] |
| 2025/Summer | [1, 4, 2, 3, 5] |
| 2025/Autumn | [3, 4, 5, 2, 1] |
| 2025/Winter | [1, 4, 3, 5, 2] |
| 2028/Spring | [2, 5, 1, 4, 3] |
| 2028/Summer | [2, 4, 5, 3, 1] |
| 2028/Autumn | [4, 1, 3, 5, 2] |
| 2028/Winter | [2, 1, 5, 4, 3] |
| 2031/Spring | [4, 1, 3, 5, 2] |
| 2031/Summer | [4, 2, 1, 5, 3] |
| 2031/Autumn | [5, 1, 2, 4, 3] |
| 2031/Winter | [5, 1, 2, 3, 4] |
| 2034/Spring | [3, 5, 4, 1, 2] |
| 2034/Summer | [1, 2, 5, 3, 4] |
| 2034/Autumn | [5, 4, 3, 1, 2] |
| 2034/Winter | [4, 5, 3, 2, 1] |
| 2037/Spring | [1, 5, 2, 4, 3] |
| 2037/Summer | [4, 2, 5, 1, 3] |
| 2037/Autumn | [2, 4, 1, 3, 5] |
| 2037/Winter | [5, 4, 1, 3, 2] |

Distinguishability finding: Relative range across the five indices: RES 1.1706%, load 0.4047%, RES share of load 0.9919%, net load 0.5012%. Kendall W of the per-block RES ranks = 0.0140 (about 0.050 expected with no agreement). Primary-metric order [4, 3, 5, 2, 1]; RES-share order [4, 3, 5, 2, 1]; net-load order [2, 3, 1, 5, 4]; load order [2, 3, 5, 4, 1]. The selected (low, mid, high) = [4, 5, 1] is the same as the (low, mid, high) of the RES-share order ([4, 5, 1]). Operation-scenario indices are not coherent RES states: production draws every load and every generator row with its own seed per network and per (year, day) block (network.py _update_network_with_operational_data; network_data.py realization seed per block), so an index is a bundle of independent per-element draws and its horizon RES level is an aggregate of those draws.

## 3. R prediction for the selected 3 x 3 set -- PREDICTION, recorded before any 3 x 3 run

R_u = spread_u(S*) / 97.99086039739976 (SRP1 Z4, undiscounted day weights); R_r2 = spread_r2(S*) / 97.01788674187569 (model block weight w / 1.02^(y-2025)); R_gn = [sum w_u top4_mean_minus_bottom4_mean(y,d) / (1+g)^(y-2025) / sum w_u](S*) / same for SRP1, g = 0.025. spread(S*) is the horizon 4 h spread of the probability-renormalized mean price profile of the selected market scenarios. The operation-scenario selection does not enter R (R is a price-only measure).

| form | selected set | five-scenario set (W22) | selected / five - 1 |
|---|---|---|---|
| R_u | 0.9378 | 0.9361 | +0.1839% |
| R_r2 | 0.9391 | 0.9373 | +0.2016% |
| R_gn | 0.9169 | 0.9151 | +0.2053% |

PREDICTION: for the 3 x 3 instance (market [2, 3, 4], operation [4, 5, 1]), the first-order arbitrage value per MWh relative to SRP1 is R_u = 0.9378, R_r2 = 0.9391 (the form of the confirmed R = 0.937), R_gn = 0.9169; recorded before any 3 x 3 run exists.

## 4. Comparison with the five-scenario mean profile

Selected-set renormalized mean-profile 4 h spread 91.8964 vs five-scenario 91.7278 EUR/MWh (undiscounted; difference +0.1687, +0.1839%); r2 91.1141 vs 90.9308 (+0.2016%); growth-normalized 78.8328 vs 78.6712 (+0.2053%). On the first-order (spread-proportional) model the reduction is expected to bias the arbitrage value UP relative to the full 5 x 5 instance by about 0.20% (r2 form). Bracket: average of the selected scenarios' own spreads 93.6401 vs five-scenario 94.6472 (-1.0640%). This covers price only; the operation-scenario reduction (and any interaction with network constraints) is not captured by R.

## 5. Realization check (how a 3 x 3 case is drawn by production)

{
 "note": "index-level only, dummy frame of the pool size; seeds recomputed with production's derive_random_seed and the labels of _read_market_data_from_file and _update_network_with_operational_data (network.random_seed read from the production objects). Checks whether DataFrame.sample(n=3) is the first 3 rows of DataFrame.sample(n=5) for the same seed. No seed depends on the scenario count.",
 "market_seeds_checked": 40,
 "market_prefix_true": 40,
 "operation_seeds_checked": 10620,
 "operation_prefix_true": 10620,
 "first_failures": [],
 "implication": "market prefix 40/40, operation prefix 10620/10620. If all true, a case derived with NumMarketScenarios = 3 and num_operation_scenarios = 3 (the p515_s44_scale_measurement.derive_case route used for the 2 x 2 pilot) holds exactly indices [1, 2, 3] in both dimensions, NOT the selected market [2, 3, 4] x operation [4, 5, 1]. Indices [1, 2, 3] rank 4 of 10 on the market rule (spread_u 91.2994, R_r2 0.9331). Realizing the selected set needs a selection mechanism the derive route does not have."
}

## Checks

| check | result |
|---|---|
| solve_guard_verify_0 | True |
| no_model_construction_call | True |
| stop_point_hits_exactly_declared | True |
| no_curtailable_generator_other_than_pv_wind | True |
| srp1_scenario_checksum_matches | True |
| srp1_prices_bound_to_every_block_are_planning_object | True |
| srp1_network_pm_is_planning_prob_market_scenarios | True |
| srp1_no_congestion_management_network | True |
| paper_scenario_checksum_matches | True |
| paper_prices_bound_to_every_block_are_planning_object | True |
| paper_network_pm_is_planning_prob_market_scenarios | True |
| paper_no_congestion_management_network | True |
| paper_pm_all_0_2 | True |
| srp1_recomputed_spread_equals_Z4_97.99_to_1e-9 | True |
| srp1_recomputed_spread_r2_equals_Z4_97.02_to_1e-9 | True |
| five_set_spread_u_reproduces_W22_to_1e-9 | True |
| five_set_R_u_reproduces_W22_to_1e-9 | True |
| five_set_R_r2_reproduces_W22_to_1e-9 | True |
| five_set_R_gn_reproduces_W22_to_1e-9 | True |
| ten_market_subsets_ranked | True |
| five_operation_indices_ranked | True |
| operation_selection_three_distinct | True |
| prefix_check_ran_market | True |
| prefix_check_ran_operation | True |

ok = True
