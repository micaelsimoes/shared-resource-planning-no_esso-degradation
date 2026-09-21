# P5.15 Addendum 28 W19 -- zero-solve reports Z1-Z4

Instance: node 7, 0.25 MVA / 1.0 MWh, 2025; nodes 5, 9 empty. candidate_key `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`; x = 0 key `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57`.
Records: A1a `data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1`, A0 duplicate `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7eb1ce62c2509f54_n7_p0_25_e1_0`, x0 `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0` (all certified, frozen oracle).
Solve profile: armed SolveProfileGuard(permitted=()); counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}; verify(0) failures [].
Git HEAD 912af3889ef8f19899b0fc710d13c3c0fa519f1b. all_pass = True; failed checks: [].

Objective convention: Q(x) = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); terminal salvage is ~0 on these runs (gross = net). V = Q(0) - Q(x). EUR.

## Z1 -- branch endpoints

Node-7 DN (case33_2): get_interface_branch_rating sums ONE branch -- branch_id 1 (index 31), from bus 1 (the DN reference bus) to bus 2, transformer, 100 MVA; the binding row is its apparent-power limit at the bus-1 end, branch_flow_limit[31,*] (pij^2 + qij^2 <= 1 + 1e-5 p.u.^2); the shared storage injects at DN bus 1 (the reference bus, the from-bus of that branch) in the DSO model and at TN bus 7 in the TSO model.

**Nuance.** There is no multi-branch sum: bus 1 has exactly one in-service branch, no load and no shunt, so the DSO interface flow pg_adn = pg_ref - shared_es_pnet equals, by the bus-1 node balance, the from-end flow of branch 1-2 identically. The storage term appears in the bus-1 balance and in pg_adn and cancels between them; it appears in no row of bus 2 or beyond and not in the branch row. The injection therefore sits AT the upstream terminal of the single constrained branch (the TN side of the cut), not upstream of a separate element: the constrained flow is the DN's own net draw, which the storage cannot change -- only DN-side resources (flexibility, DERs) can. In the TSO the storage is a separate injection at TN bus 7, the same bus as the ADN interface load; TN branches at bus 7 were NOT evaluated here (no TN branch flow is captured in the records used).

**Verdict.** CONFIRMED, with the precise wording: the binding element (branch 1-2, 100 MVA, its bus-1 end) is on the DN side of the storage's injection bus (bus 1 = its from-bus), and the interface flow is by construction net of the storage, so the storage cannot relieve it.

Evidence (terminal, certified A1a record): node 7 at or above 99% of 100 MVA in 23/288 slots; max |S| = 100.0005003 MVA against the row bound sqrt(1 + 1e-5) x 100 = 100.0005000 MVA (max deviation in the at-rating slots 1.57e-05 MVA; minimum utilization among them 1.000005; 23 slots within 0.001 MVA of the bound). Nodes 5/9 max utilization 0.510 / 0.672. Same at-rating slot set at x = 0: True (max |dP| on those slots 7.97e-05 MW). Reference generator (pmax 100 MW) is not at its bound: max pg_ref in those slots 99.8248 MW. Storage in those slots (DSO copy): at >= 99 % of its own 0.25 MVA rating in 2, at <= 1 % in 4 of 23 (per-slot values in the JSON).

Citations: get_interface_branch_rating network.py:78; selection_rule network.py:84; rating_sum network.py:85; branch_flow_limit_rule model_construction_helpers.py:1536; branch_flow_limit_ji_rule model_construction_helpers.py:1550; apparent_power_limit_for_transformers_under_MIXED model_construction_helpers.py:1231; branch_flow_limit_row_wired network.py:520; branch_flow_limit_ji_row_wired network.py:521; dso_interface_p_def_storage_at_reference_bus model_construction_helpers.py:1208; dso_interface_p_returns_pg_minus_storage model_construction_helpers.py:1216; shared_storage_in_node_balance model_construction_helpers.py:1329; tso_adn_interface_load_def model_construction_helpers.py:1167; get_interface_branch_rating_uses_in_srp ['shared_resources_planning.py:3750', 'shared_resources_planning.py:4611', 'shared_resources_planning.py:4768', 'shared_resources_planning.py:4936', 'shared_resources_planning.py:6130', 'shared_resources_planning.py:6481', 'shared_resources_planning.py:7689']

## Z2 -- ageing trajectory (n7 0.25/1.0, 2025)

phi_cal as consumed = 1.0 (shared_energy_storage_data.py:638, applied at shared_energy_storage_data.py:668; ageing.calendar_retention_per_year is ABSENT from data/SRP1/SharedESS/SRP1_ESS_Params.json; the default 1.0 applies). k = cl_eff = 11541.560327112 (C3: 10000 x 0.80 / -ln 0.50; shared_energy_storage_data.py:654). soh_min = 0.5.

| cohort | block | EFC/day | D | SoH start | SoH used (end of block) | recomputed | E_avail MWh | floor active |
|---|---|---|---|---|---|---|---|---|
| 2025 | 2025 | 1.144513 | 0.180975 | 1.000000 | 0.834456 | 0.834456 | 0.834456 | False |
| 2025 | 2030 | 1.027014 | 0.162396 | 0.834456 | 0.709375 | 0.709375 | 0.709375 | False |
| 2025 | 2035 | 0.826472 | 0.130685 | 0.709375 | 0.622472 | 0.622472 | 0.622472 | False |

EFC over the horizon = 5471.35 full cycles. Cohorts 2030/2035 carry zero capacity (SoH fixed 1.0, not a result).

Formula: EFC/day[y_inv,y] = avg_ch_dch / (2 E_rated), avg_ch_dch = sum_d (n_d/365) sum_p (eta_ch pch + pdch/eta_dch) dt (throughput on the CELL side, E_rated = nameplate); D[y_inv,y] = 365 * num_years * avg_ch_dch / (2 k E) = 365 * num_years * EFC/day / k; SoH[y_inv,y] = SoH[y_inv,y-1] * exp(-D[y_inv,y]) * phi_cal^num_years (SoH[y_inv,y_inv-1] := 1); E_available[y] = sum_{y_inv} E_rated[y_inv,y] * SoH[y_inv,y], published to the TSO and DSO blocks of year y as shared_es_e_rated_fixed.

**End-of-block:** YES -- available energy in block y uses the END-of-block SoH: SoH[0,y] already includes exp(-D[0,y]), the degradation caused by block y's own throughput (for 2025: 0.8345, not 1.0).

## Z3 -- discount re-weighting

Production convention: block weight = num_years[y] * num_days[d] / (1 + r)^(y - y_first), r = DiscountFactor = 0.02: ONE discount factor per representative year, applied to all num_years (5) years of that block (NOT an annuity over the 5 years); y_first = 2025 so the 2025 block is undiscounted. (shared_resources_planning.py:3421-3424).

Formula: Q_y = sum over the TSO block and all DSO blocks of year y of the gross settlement-excluded per-block cost (sum of generation_cost + flexibility_cost_internal + load_curtailment_cost + res_curtailment_penalty + ess_usage_cost + detector_penalty_total); undiscounted Q_y = sum of unweighted * num_years * num_days; V_y = Q_y(0) - Q_y(x); V(r) = sum_y V_y_undiscounted / (1 + r)^(y - 2025); value_to_cost(r) = V(r) / I(x).

| year | Q_y(0) weighted 2 % | Q_y(x) weighted 2 % | V_y weighted 2 % | V_y undiscounted |
|---|---|---|---|---|
| 2025 | 306,232,429.60 | 306,129,057.65 | 103,371.95 | 103,371.95 |
| 2030 | 170,760,980.25 | 170,685,591.93 | 75,388.32 | 83,234.79 |
| 2035 | 176,866,051.38 | 176,783,004.14 | 83,047.24 | 101,234.12 |

Reconstruction check: V(2 %) from blocks = 261,807.504426, re-weighted from undiscounted = 261,807.504426; certified 261,807.504426; |diff| 8.94e-08 / 6.88e-08 EUR (tolerance 1.0 EUR).

I(x) = 317,957.01 EUR (2025, held fixed). Resolution of V(2 %): bar(x) + bar(x0) = 20,970.30 EUR.

| r | V(r) | V(r)/V(2 %) | change | value-to-cost | bar (conservative) | expert |
|---|---|---|---|---|---|---|
| 0% | 287,840.86 | 1.0994 | +9.94 % | 0.9053 | 25,562.68 | + at most ~10 % |
| 2% | 261,807.50 | 1.0000 | +0.00 % | 0.8234 | 20,970.30 | - |
| 5% | 230,737.56 | 0.8813 | -11.87 % | 0.7257 | 20,970.30 | ~ -12 % |
| 8% | 206,911.14 | 0.7903 | -20.97 % | 0.6508 | 20,970.30 | ~ -21 % |

resolution of V(2 %) = bar(x) + bar(x0) (max |gross step| over the last 10 cycles of each run); the per-year split of the bar is not recorded, so V(r)'s bar is bounded conservatively by bar_sum * max_y ((1.02/(1+r))^(y-2025)).

## Z4 -- price spread

Formula: per representative (year, day): max_p c_p - min_p c_p, and mean of the top-4 hourly prices minus mean of the bottom-4 (h = E/P = 4 h); averaged with weight num_years * num_days (undiscounted) and, separately, with the model block weight num_years * num_days / 1.02^(y-2025).

| scope | max - min | top-4 - bottom-4 | mean price |
|---|---|---|---|
| all years, day-weighted (undiscounted) | 116.92 | 97.99 | 108.41 |
| all years, model block weight (2 %) | 115.88 | 97.02 | 107.60 |
| 2025, day-weighted | 101.04 | 83.22 | 95.38 |
| 2030, day-weighted | 117.06 | 98.01 | 109.93 |
| 2035, day-weighted | 132.67 | 112.74 | 119.93 |

| year | day | days | min | max | max - min | top-4 - bottom-4 |
|---|---|---|---|---|---|---|
| 2025 | Spring | 92 | 3.26 | 158.09 | 154.83 | 133.37 |
| 2025 | Summer | 91 | 33.64 | 144.26 | 110.62 | 96.41 |
| 2025 | Autumn | 91 | 80.55 | 161.80 | 81.25 | 56.26 |
| 2025 | Winter | 91 | 84.13 | 141.01 | 56.88 | 46.28 |
| 2030 | Spring | 92 | 3.69 | 178.86 | 175.17 | 150.90 |
| 2030 | Summer | 91 | 42.76 | 183.24 | 140.49 | 118.99 |
| 2030 | Autumn | 91 | 92.15 | 175.07 | 82.92 | 63.16 |
| 2030 | Winter | 91 | 92.37 | 161.38 | 69.02 | 58.41 |
| 2035 | Spring | 92 | 23.43 | 219.34 | 195.91 | 180.92 |
| 2035 | Summer | 91 | 41.53 | 184.82 | 143.29 | 125.03 |
| 2035 | Autumn | 91 | 41.00 | 175.60 | 134.60 | 96.43 |
| 2035 | Winter | 91 | 115.34 | 171.51 | 56.17 | 47.85 |

Captured spread. captured spread [EUR/MWh per full cycle] = (V / E) / EFC_horizon, EFC_horizon = sum_y 365 * num_years * EFC/day_y (full equivalent cycles of the nameplate E over the 15 years; Z2 EFC, cell-side throughput / (2 E))

- EFC_horizon = 5471.35 cycles (discounted at 2 %: 5023.69)
- (a) V(2 %) / EFC_horizon = 47.85 EUR/MWh per full cycle
- (b) V(0 %) / EFC_horizon = 52.61
- (c) V(2 %) / EFC_horizon discounted = 52.11
- 2025: V_y undiscounted 103,371.95 / 2088.7 cycles = 49.49 EUR/MWh per cycle (SoH end of block 0.8345)
- 2030: V_y undiscounted 83,234.79 / 1874.3 cycles = 44.41 EUR/MWh per cycle (SoH end of block 0.7094)
- 2035: V_y undiscounted 101,234.12 / 1508.3 cycles = 67.12 EUR/MWh per cycle (SoH end of block 0.6225)
- expert: captured ~55-60 EUR/MWh per full cycle

Citations: market_reader shared_resources_planning.py:8302; price_array shared_resources_planning.py:8346; tso_prices_bound shared_resources_planning.py:8175; dso_prices_bound shared_resources_planning.py:8132; generation_cost_uses_price model_construction_helpers.py:1669; market_file data/SRP1/MarketData/SRP1_market_data.xlsx (base profiles; SRP1.json "MarketData"), synthetic scenario selection seeded by SRP1.json "RandomSeed", growth factor from its "Growth Factors" sheet

## Checks

- a0_c7_record_key_matches: True
- a1a_record_key_matches: True
- all_three_records_certified: True
- candidate_key_recomputed_equals_expected: True
- inputs_all_tracked: True
- x0_record_key_matches: True
- z1_dso_rows_structure_ok_every_block_every_period: True
- z1_dso_shared_storage_at_reference_bus: True
- z1_ess_stride_sidecar_hash_matches_committed_child_manifest: True
- z1_no_load_no_shunt_at_reference_bus: True
- z1_node7_has_slots_on_the_branch_row_bound: True
- z1_node7_never_exceeds_the_branch_row_bound_by_more_than_1e-5_mva: True
- z1_node7_only_node_at_rating: True
- z1_rating_equals_that_branch_rate_every_block: True
- z1_reference_generator_bound_inactive: True
- z1_same_reference_bus_and_branch_every_block: True
- z1_single_adjacent_branch_every_block: True
- z1_tso_storage_at_bus7_with_adn_load_every_block: True
- z2_available_equals_rated_times_soh: True
- z2_k_equals_C3_calibration: True
- z2_no_floor_row_active: True
- z2_phi_cal_is_1: True
- z2_sidecar_matches_record: True
- z2_soh_chain_recomputes_within_1e-9: True
- z3_T1_row_candidate_key_matches: True
- z3_a1a_and_a0_c7_candidate_blocks_bit_identical: True
- z3_a1a_block_weights_match_formula: True
- z3_a1a_blocks_reconcile_to_gross_within_0.01_eur: True
- z3_a1a_undiscounted_times_factor_equals_weighted: True
- z3_model_discount_factor_is_0_02: True
- z3_r2_reconstruction_within_tolerance: True
- z3_x0_block_weights_match_formula: True
- z3_x0_blocks_reconcile_to_gross_within_0.01_eur: True
- z3_x0_undiscounted_times_factor_equals_weighted: True
- z4_one_market_scenario: True
- z4_prices_equal_those_recorded_by_the_certified_run: True
- z4_tso_and_dso_price_arrays_are_the_planning_object: True
