# P5.15 Addendum 31 W22 -- paper-scale market-scenario spreads (spec v17 Z31; zero solves)

Started 2026-09-21T14:37:28.535618+00:00; HEAD abf0cb303882b3aa8ccfecf51a152682c29e852e; script sha256 7524f32e4a9eea642833baef410b942bb949ff6db7a557d5dd47fbbe89863cc1.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 31; data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json step Z31.

**Zero solves** (armed SolveProfileGuard(permitted=()), verify(0) failures: []; counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}). **No model construction** (blocked calls: []; blockers ['Network.build_model', 'NetworkData.build_model', 'SharedEnergyStorageData.build_subproblem', 'SharedEnergyStorageData.build_master_problem']). Declared stop point hits: 2 (declared 2).

## Instances (production reader)

| instance | case | case sha256 | scenario checksum | expected | match | years (num_years) | market scen. | pm (network) |
|---|---|---|---|---|---|---|---|---|
| srp1 | data/SRP1/SRP1.json | 61a794a7ce7a | 5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358 | 5a02b77c... (p56a_oracle.CANONICAL_CHECKSUM) | True | {'2025': 5, '2030': 5, '2035': 5} | 1 | [(1.0,)] |
| paper | data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json | d726307c1949 | 1e8bdd3e5233442a87fbe44b78281c8ae61b09c09700388fed20ab9684d18aef | 1e8bdd3e... (data/SRP1/Results/P515S44/scale_measurement/paper_build/build_child_stdout.log:33 ("[INFO] Scenario checksum: ...")) | True | {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3} | 5 | [(0.2, 0.2, 0.2, 0.2, 0.2)] |

pm source: network.prob_market_scenarios of every TSO/DSO block (shared_resources_planning.py:8174, shared_resources_planning.py:8131) = planning.prob_market_scenarios[year] = [1/NumMarketScenarios] x NumMarketScenarios (shared_resources_planning.py:8326); identity checked above. Consistent with the W3 probability audit (P515S45/probability_audit, [0.2]x5).

Binding checks: {"srp1": {"tso_and_dso_price_array_is_planning_object": true, "pm_is_planning_object": true, "obj_types": {"TSO": 1, "DSO_5": 1, "DSO_7": 1, "DSO_9": 1}}, "paper": {"tso_and_dso_price_array_is_planning_object": true, "pm_is_planning_object": true, "obj_types": {"TSO": 1, "DSO_5": 1, "DSO_7": 1, "DSO_9": 1}}}

SRP1 prices from this read vs Z4 committed prices: max |diff| = 0.000e+00.

## Spread definition

per representative (year, day) and price vector c (24 hourly EUR/MWh): max_minus_min = max_p c_p - min_p c_p; top4_mean_minus_bottom4_mean = mean of the 4 highest c_p minus mean of the 4 lowest (W19 Z4 definition, p515_s46_zero_solve_reports.py @ dd3afa6e). Horizon average = sum_(y,d) w * spread / sum_(y,d) w with w = num_years[y] * num_days[d] (undiscounted) or w = num_years[y] * num_days[d] / 1.02^(y - 2025) (model block weight, production DiscountFactor 0.02).

Z4 reproduction (definition identity): max per-day |diff| 0.000e+00; recomputed horizon (undiscounted) 97.99086039739976 vs committed 97.99086039739976; ok = True.

## Paper scale: horizon-weighted spreads (EUR/MWh)

Rows: scenario s (its row in every (year, day) block), the pm-weighted mean price profile, and the pm-weighted average of the scenario spreads. Weight = num_years x num_days (undiscounted) and model block weight (/1.02^(y-2025)).

| profile | 4 h spread undisc. | 4 h spread r2 | max-min undisc. | max-min r2 | mean price undisc. | 4 h 2025 | 4 h 2028 | 4 h 2031 | 4 h 2034 | 4 h 2037 | max-min 2025 | max-min 2028 | max-min 2031 | max-min 2034 | max-min 2037 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 95.31 | 94.48 | 113.09 | 112.06 | 113.32 | 83.22 | 86.57 | 90.85 | 108.80 | 107.12 | 101.04 | 99.32 | 104.80 | 131.05 | 129.25 |
| s2 | 94.83 | 94.01 | 108.35 | 107.41 | 119.72 | 82.87 | 86.37 | 93.04 | 101.88 | 109.99 | 95.83 | 99.11 | 104.72 | 113.44 | 128.63 |
| s3 | 91.87 | 91.11 | 107.79 | 107.00 | 111.10 | 79.99 | 82.10 | 93.75 | 101.49 | 102.00 | 94.98 | 97.50 | 111.31 | 116.21 | 118.96 |
| s4 | 94.22 | 93.38 | 111.07 | 110.16 | 112.69 | 81.02 | 87.23 | 92.35 | 100.02 | 110.50 | 97.96 | 100.62 | 112.69 | 115.04 | 129.04 |
| s5 | 97.00 | 96.12 | 112.09 | 111.08 | 113.02 | 86.46 | 83.73 | 96.39 | 104.87 | 113.56 | 97.81 | 101.92 | 107.35 | 123.63 | 129.72 |
| mean_profile | 91.73 | 90.93 | 104.93 | 104.01 | 113.97 | 80.68 | 81.80 | 91.12 | 99.48 | 105.56 | 91.87 | 94.02 | 103.95 | 114.05 | 120.76 |
| avg_of_scenario_spreads | 94.65 | 93.82 | 110.48 | 109.54 | 113.97 | 82.71 | 85.20 | 93.28 | 103.41 | 108.63 | 97.52 | 99.69 | 108.17 | 119.87 | 127.12 |

SRP1 (single scenario, recomputed here): 4 h undiscounted 97.99, r2 97.02; max-min undiscounted 116.92; per year 4 h 2025: 83.22, 2030: 98.01, 2035: 112.74.

## Paper scale: per (year, day) 4 h spread

| year/day | s1 | s2 | s3 | s4 | s5 | mean_profile | avg_of_scenario_spreads |
|---|---|---|---|---|---|---|---|
| 2025/Spring | 133.37 | 142.40 | 129.03 | 128.86 | 147.11 | 131.88 | 136.15 |
| 2025/Summer | 96.41 | 87.67 | 81.36 | 89.86 | 90.04 | 88.66 | 89.07 |
| 2025/Autumn | 56.26 | 55.42 | 67.94 | 60.98 | 60.72 | 57.58 | 60.26 |
| 2025/Winter | 46.28 | 45.34 | 41.09 | 43.87 | 47.30 | 44.02 | 44.78 |
| 2028/Spring | 138.14 | 153.29 | 140.54 | 169.65 | 135.70 | 142.67 | 147.46 |
| 2028/Summer | 90.56 | 90.87 | 79.17 | 79.57 | 86.23 | 84.17 | 85.28 |
| 2028/Autumn | 73.18 | 60.85 | 61.49 | 60.76 | 65.34 | 59.96 | 64.32 |
| 2028/Winter | 43.82 | 39.77 | 46.55 | 38.02 | 47.10 | 39.72 | 43.05 |
| 2031/Spring | 151.39 | 157.54 | 154.67 | 146.37 | 159.13 | 152.77 | 153.82 |
| 2031/Summer | 88.39 | 92.85 | 102.03 | 100.83 | 104.21 | 97.65 | 97.66 |
| 2031/Autumn | 72.06 | 69.03 | 66.57 | 84.45 | 80.30 | 67.76 | 74.48 |
| 2031/Winter | 50.90 | 52.04 | 51.08 | 37.16 | 41.26 | 45.64 | 46.49 |
| 2034/Spring | 166.34 | 177.84 | 168.11 | 174.00 | 159.92 | 167.15 | 169.24 |
| 2034/Summer | 122.65 | 101.60 | 100.94 | 103.93 | 116.55 | 107.73 | 109.14 |
| 2034/Autumn | 87.11 | 75.59 | 81.15 | 71.47 | 85.79 | 71.63 | 80.22 |
| 2034/Winter | 58.48 | 51.65 | 55.01 | 49.87 | 56.62 | 50.68 | 54.33 |
| 2037/Spring | 179.34 | 189.97 | 172.26 | 184.54 | 191.43 | 178.10 | 183.51 |
| 2037/Summer | 108.28 | 122.34 | 111.21 | 125.86 | 132.08 | 117.66 | 119.96 |
| 2037/Autumn | 77.20 | 74.53 | 78.85 | 80.67 | 86.61 | 75.54 | 79.57 |
| 2037/Winter | 62.84 | 52.22 | 44.92 | 50.12 | 43.26 | 50.15 | 50.67 |

## SRP1 correspondence

Metric: a = SRP1 price row, b = paper profile (same day/season). max_abs_diff = max_p |a_p - b_p| (EUR/MWh; "identical" iff <= 1e-09); ls_scale_k = <a,b>/<b,b> (least-squares a ~ k b); pearson_r = correlation over the 24 hours. Growth-normalized: every profile divided by (1+g)^(y-2025), g = production energy growth factor, i.e. brought to 2025 price level.

Same-year comparison ([2025], the only year both instances share):

| day | s1 max|diff| / k / r | s2 max|diff| / k / r | s3 max|diff| / k / r | s4 max|diff| / k / r | s5 max|diff| / k / r | mean_profile max|diff| / k / r |
|---|---|---|---|---|---|---|
| 2025/Spring | 0.000 / 1.0000 / 1.0000 | 52.777 / 0.8029 / 0.9767 | 30.203 / 0.9466 / 0.9843 | 32.518 / 1.1492 / 0.9882 | 56.983 / 0.7913 / 0.9754 | 18.137 / 0.9265 / 0.9956 |
| 2025/Summer | 0.000 / 1.0000 / 1.0000 | 11.120 / 0.9834 / 0.9930 | 23.327 / 0.9509 / 0.9672 | 7.889 / 0.9882 / 0.9968 | 9.952 / 1.0220 / 0.9954 | 8.869 / 0.9900 / 0.9951 |
| 2025/Autumn | 0.000 / 1.0000 / 1.0000 | 11.481 / 1.0089 / 0.9764 | 51.413 / 1.1627 / 0.6693 | 34.773 / 1.1017 / 0.8042 | 32.950 / 1.1447 / 0.8557 | 25.498 / 1.0856 / 0.8935 |
| 2025/Winter | 0.000 / 1.0000 / 1.0000 | 7.430 / 0.9820 / 0.9829 | 7.162 / 0.9824 / 0.9938 | 2.844 / 0.9943 / 0.9987 | 10.281 / 1.0104 / 0.9735 | 2.695 / 0.9942 / 0.9991 |

Growth-normalized (2025 level) comparison over all SRP1 years:

| SRP1 year/day | nearest paper profile | max|diff| | identical paper profiles | vs paper all-year mean profile max|diff| / k / r |
|---|---|---|---|---|
| 2025/Spring | 2025/s1 | 0.000 | ['2025/s1', '2031/s3'] | 16.718 / 0.9300 / 0.9965 |
| 2025/Summer | 2025/s1 | 0.000 | ['2025/s1'] | 15.811 / 0.9731 / 0.9850 |
| 2025/Autumn | 2025/s1 | 0.000 | ['2025/s1'] | 19.256 / 1.0481 / 0.9264 |
| 2025/Winter | 2025/s1 | 0.000 | ['2025/s1'] | 9.434 / 0.9764 / 0.9879 |
| 2030/Spring | 2025/s1 | 0.000 | ['2025/s1', '2031/s3'] | 16.718 / 0.9300 / 0.9965 |
| 2030/Summer | 2037/s2 | 14.152 | [] | 27.492 / 1.0534 / 0.9670 |
| 2030/Autumn | 2028/s4 | 5.549 | [] | 12.537 / 1.0668 / 0.9812 |
| 2030/Winter | 2025/s5 | 4.275 | [] | 15.574 / 0.9552 / 0.9530 |
| 2035/Spring | 2037/s2 | 1.461 | [] | 39.334 / 1.1524 / 0.9836 |
| 2035/Summer | 2025/s1 | 1.773 | [] | 17.585 / 0.9691 / 0.9824 |
| 2035/Autumn | 2034/s3 | 25.029 | [] | 51.715 / 0.7212 / 0.8105 |
| 2035/Winter | 2037/s4 | 3.396 | [] | 2.978 / 1.0071 / 0.9979 |

Mechanism check (index level): {"note": "index-level check only, on a dummy 100-row frame (the synthetic pool size n_samples=100 of _generate_market_price_scenarios); no prices are generated here", "same_random_seed_both_cases": true, "same_market_file_both_cases": true, "per_day": {"Spring": {"seed": 2125893541, "n1_rows": [50], "n5_rows": [50, 53, 65, 79, 59], "n1_is_prefix_of_n5": true}, "Summer": {"seed": 462913175, "n1_rows": [52], "n5_rows": [52, 29, 92, 88, 61], "n1_is_prefix_of_n5": true}, "Autumn": {"seed": 3536255137, "n1_rows": [27], "n5_rows": [27, 48, 18, 10, 49], "n1_is_prefix_of_n5": true}, "Winter": {"seed": 2895842244, "n1_rows": [88], "n5_rows": [88, 33, 59, 96, 94], "n1_is_prefix_of_n5": true}}}

Finding: Same year ([2025]): paper profiles identical (max|diff| <= 1e-09) to SRP1 per day: {'2025/Spring': ['s1'], '2025/Summer': ['s1'], '2025/Autumn': ['s1'], '2025/Winter': ['s1']}. Growth-normalized, SRP1 (year/day) identical to paper (year/scenario): {'2025/Spring': ['2025/s1', '2031/s3'], '2025/Summer': ['2025/s1'], '2025/Autumn': ['2025/s1'], '2025/Winter': ['2025/s1'], '2030/Spring': ['2025/s1', '2031/s3'], '2030/Summer': [], '2030/Autumn': [], '2030/Winter': [], '2035/Spring': [], '2035/Summer': [], '2035/Autumn': [], '2035/Winter': []}. Scenario labels are per (year, day) block draws, not trajectories across blocks (each block selects its rows with its own seed).

## Ratios -- PREDICTION for the paper-scale evaluations

R = [sum_(y,d) w * top4_mean_minus_bottom4_mean(mean profile)] / [sum w] (paper, w = num_years x num_days, undiscounted) / 97.99086039739976 (SRP1, Z4, same weighting); the average of scenario spreads replaces the mean-profile spread by sum_s pm_s * top4_mean_minus_bottom4_mean(scenario s) per (y, d); the r2 variants use the model block weight and SRP1 r2 = 97.01788674187569; the growth-normalized supplementary ratios divide every (y, d) spread by (1+g)^(y-2025) before weighting, g = 0.025, removing the effect of the two instances' different representative years.

| quantity | value |
|---|---|
| SRP1_4h_spread_undiscounted (Z4) | 97.99086039739976 |
| paper_mean_profile_4h_spread_undiscounted | 91.72775939691898 |
| paper_avg_of_scenario_4h_spreads_undiscounted | 94.64715634846279 |
| R = mean_profile / SRP1 (undiscounted) | 0.9360848453102575 |
| avg_of_scenario_spreads / SRP1 (undiscounted) | 0.9658773886117884 |
| flattening = mean_profile / avg_of_scenario_spreads (undiscounted) | 0.9691549427983294 |
| SRP1_4h_spread_r2 (Z4) | 97.01788674187569 |
| R_r2 = mean_profile_r2 / SRP1_r2 | 0.9372576465471794 |
| avg_of_scenario_spreads_r2 / SRP1_r2 | 0.9670257076166785 |
| like-for-like 2025 (only common year): paper mean_profile 2025 / SRP1 2025 (day-weighted) | 0.9694648709978263 |
| like-for-like 2025 (only common year): paper avg_of_scenario_spreads 2025 / SRP1 2025 (day-weighted) | 0.9939286226892993 |
| supplementary: growth-normalized (2025 level) R = paper mean_profile / SRP1, undiscounted weights | 0.9150641848892984 |
| supplementary: growth-normalized avg_of_scenario_spreads / SRP1, undiscounted weights | 0.94411047518174 |

PREDICTION: first-order estimate of the paper-scale arbitrage value per MWh relative to SRP1: R = 0.9361 (mean-profile 4 h spread 91.73 vs SRP1 97.99 EUR/MWh); the average of the scenario spreads would give 0.9659, the difference being the flattening of the non-anticipative mean price profile.

Expert prediction (spec v17 Z31_expert): the mean-profile spread is at most the average of the scenario spreads (misaligned peaks flatten it); uncertainty cannot improve the case in this model -- check: True (every (year, day)), True (horizon).

## Checks

| check | result |
|---|---|
| solve_guard_verify_0 | True |
| no_model_construction_call | True |
| stop_point_hits_exactly_declared | True |
| srp1_scenario_checksum_matches | True |
| srp1_prices_bound_to_every_block_are_planning_object | True |
| srp1_network_pm_is_planning_prob_market_scenarios | True |
| srp1_no_congestion_management_network | True |
| srp1_discount_factor_0_02 | True |
| paper_scenario_checksum_matches | True |
| paper_prices_bound_to_every_block_are_planning_object | True |
| paper_network_pm_is_planning_prob_market_scenarios | True |
| paper_no_congestion_management_network | True |
| paper_discount_factor_0_02 | True |
| paper_pm_all_0_2 | True |
| z4_definition_reproduced_to_1e-9 | True |
| srp1_prices_equal_z4_committed_prices | True |
| srp1_recomputed_spread_equals_97.99_to_1e-9 | True |
| mean_profile_spread_le_avg_of_scenario_spreads_every_day | True |
| mean_profile_spread_le_avg_of_scenario_spreads_horizon | True |

ok = True
