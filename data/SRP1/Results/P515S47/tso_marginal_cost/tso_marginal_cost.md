# P5.15 Addendum 31 item (2) W25 -- TSO bus-7 marginal cost, value attribution, captured spread (zero solves)

Instances (candidate keys): x = 0 `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57`; node 7 0.25 MVA / 1.0 MWh, 2025, nodes 5/9 empty `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`.
- x = 0 (a0_c7:x0): `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0` (status certified, cycles 132, Q gross 653,859,461.23, bar 9,629.98, terminal step / threshold 0.0034)
- BASELINE (s47_recert:n7_4h_e1; C2 + phi_cal 0.985 + soh_min 0.70): `data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1` (status certified, cycles 112, Q gross 653,600,033.46, bar 25,104.65, terminal step / threshold 0.0567)
- SENSITIVITY SET (s45_a1a:n7_4h_e1; C3, soh_min 0.50): `data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1` (status certified, cycles 125, Q gross 653,597,653.72, bar 11,340.32, terminal step / threshold 0.0185)

Solve profile: armed SolveProfileGuard(permitted=()); counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}; verify(0) failures []. Git HEAD f6f868882134cbbb2dd607f1a2c951d3a643b3ae. all_pass = True; failed checks: [].

Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage is 0 on these records (gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) unless stated.

## T1 -- how the market price enters the TSO objective

- The TSO pays the market price on its generation: generation_cost = sum over controllable generators (CONV, REF, controllable RES) of c_p[s_m][p] * baseMVA * pg, with c_p = network.cost_energy_p[s_m] (the per-(year, day) market price array of scenario s_m); total_generation_cost weights each scenario by prob_market[s_m] * prob_operation[s_o]. On SRP1 there is one market and one operation scenario (weight 1). In case9 (2025 file) the priced generators are CONV gen 1 at bus 1, CONV gen 2 at bus 2, CONV gen 3 at bus 3; unpriced: WIND gen 4 at bus 4, WIND gen 5 at bus 6, WIND gen 6 at bus 8, PV gen 7 at bus 4, PV gen 8 at bus 6, PV gen 9 at bus 8. Generators at bus 7: [].
- The TSO earns the market price on the interface energy it delivers: interface_energy_settlement = -sum prob * c_p[p] * baseMVA * pc_adn[dn] for every ADN bus (5, 7, 9), weight 1 in the ADMM path (snapshot: interface_settlement_weight = 1.0); the DSO pays +c_p * pg_adn. The settlement is excluded from the reported Q(x).
- Bus 7 connects to it through node_balance_p[7]: load = pc (FIXED at the build-time consensus) + interface_delta_p (FREE within +/- the interface rating in p.u.) + shared_es_pnet (load convention). The TSO's only loads are the three ADN interfaces.
- The storage does NOT face the market price directly: the ESSO objective is its feasibility penalty (plus ADMM terms) and shared_energy_storage_data.py contains no reference to cost_energy_p. Its economic signal is the consensus duals of the TSO and DSO copies. In the TSO its injection enters node_balance_p[7] only, so the TSO values it at the bus-7 dual y_7; in the DSO it enters the reference-bus balance and pg_adn and cancels (W19 Z1), so the DSO copy carries no price. The storage therefore sees the bus-7 marginal cost, and y_7 = pi + (interface consensus correction) by the identity of T2 -- the market price reaches it only through the bus-7 nodal balance.

Citations: generation_cost_prices_every_controllable_generator_at_c_p model_construction_helpers.py:1663; controllable_types definitions.py:21; total_generation_cost_probability_weights model_construction_helpers.py:1677; interface_energy_settlement_tso_minus_pi_pc_adn model_construction_helpers.py:1617; settlement_in_objective model_construction_helpers.py:1659; settlement_weight_set_1_admm_tso shared_resources_planning.py:4496; tso_prices_bound_per_block shared_resources_planning.py:8175; tso_market_probabilities_bound shared_resources_planning.py:8174; market_probabilities_uniform shared_resources_planning.py:8326; price_array_built shared_resources_planning.py:8346; compute_node_load_adn_delta_and_shared_storage model_construction_helpers.py:1286; node_balance_p_rule model_construction_helpers.py:1366; tso_interface_pc_adn_def model_construction_helpers.py:1167; tso_pc_fixed_delta_freed_admm shared_resources_planning.py:3761; tso_admm_objective_pf_terms shared_resources_planning.py:4516; tso_dual_pf_param_set_from_dual_vars_over_s_base shared_resources_planning.py:5439; pf_dual_update_rule shared_resources_planning.py:7630; aa_pf_dual_antisymmetry admm_anderson_acceleration.py:358; admm_block_weight shared_resources_planning.py:3421; esso_operational_objective_feasibility_penalty_only shared_energy_storage_data.py:853; dso_interface_p_def_storage_at_reference_bus model_construction_helpers.py:1208; objective_scale_fixed_in_case_file data/SRP1/SRP1_params.json:49

Case9 generators: gen 1 bus 1 CONV (priced at pi); gen 2 bus 2 CONV (priced at pi); gen 3 bus 3 CONV (priced at pi); gen 4 bus 4 WIND (no cost term); gen 5 bus 6 WIND (no cost term); gen 6 bus 8 WIND (no cost term); gen 7 bus 4 PV (no cost term); gen 8 bus 6 PV (no cost term); gen 9 bus 8 PV (no cost term). Loads: bus 5, bus 7, bus 9 (exactly the three ADN interfaces).

## T2 -- TSO bus-7 nodal marginal cost at x = 0

**Recoverability.** The terminal bus-7 duals are NOT persisted by any of the three records (searched, W25 exploration: the file listing of the three record directories incl. results/ and esso_capture/, and the key names of every *.json / *.jsonl / *.log of the x0 directory for dual / lmp / lambda / marginal / shadow; the persisted pickles are the ESSO models (esso_models_s39_D.pkl) and the cycle-7 2025-Summer TSO (Summer) and DSO-node-7 (Autumn) pre-solve snapshots; esso_capture/ holds ESSO rows only, empty at x = 0). They ARE recoverable without a solve through an exact first-order identity of the TSO ADMM block (formula below), verified on that snapshot in every run to the stated tolerance, and applied to the terminal PF duals recorded every cycle in pf_entry_stride_s39_D.jsonl (hash-settled by the committed child manifest).

Formula: LMP_b[y,d,p] [EUR/MWh] = y_b / baseMVA = pi[y,d,p] + sigma * lambda_dso_b[y,d,p] / (w_block[y,d] * s_base * R_b * baseMVA); y_b = IPOPT dual of node_balance_p at TSO bus b (Pyomo sign: +objective coefficient of an interior injection, per p.u.-hour of the active objective, which is the PHYSICAL per-representative-day objective plus sigma/w_block times the ADMM terms -- p58_rescaled_admm_objective); lambda_dso = dual_vars[pf][dso] after the cycle's update (pf_entry_stride, captured before Anderson acceleration); R_b = interface rating / s_base (p.u.); sigma = admm_common_objective_scale; w_block = admm_block_weight. Derivation: stationarity of the TSO block in interface_delta_p (free; enters node_balance_p[b] with -1 via compute_node_load, the settlement with -pi*baseMVA, expected_interface_pf_p_def with -1) and in expected_interface_pf_p (free; ADMM terms), with lambda_tso = -lambda_dso (production antisymmetry, admm_anderson_acceleration.py) and the post-update identity lambda_tso(c)/s_base = lambda_param + rho*(E - p_req)/R (p_req = this cycle's DSO value: DSO solves first). Single market and operation scenario: the scenario-deviation penalty has zero gradient; TSO proximal gamma_pf = 0 (checked on the snapshot).

Verification on the persisted solved TSO block of each run (2025 Summer; duals of the cycle-6 solve carried by the cycle-7 pre-solve snapshot; buses 5, 7, 9 x 24 periods):

| run | max abs(identity error) EUR/MWh | antisymmetry max dev (raw) | delta free & interior | pc fixed | sigma | w_block | CONV interior rows | max abs(dual - pi) interior CONV |
|---|---|---|---|---|---|---|---|---|
| x0 | 7.071e-07 | 6.939e-18 | True | True | 93,635,360 | 455.0000 | 63/72 | 3.301e-05 |
| baseline | 7.067e-07 | 1.857e-03 | True | True | 93,635,360 | 455.0000 | 63/72 | 3.893e-05 |
| sensitivity_c3 | 7.061e-07 | 1.825e-03 | True | True | 93,635,360 | 455.0000 | 63/72 | 3.830e-05 |

Terminal validity: interface_delta_p margin to its bound, minimum over every block/period/node of the x = 0 record = 25.893 MW (required >= 0.001 MW). Prices in the run vs Z4 arrays: max abs dev 0.000e+00.

Spread definition: per representative (year, day) and hourly series c (24 values, EUR/MWh): top4_minus_bottom4 = mean of the 4 highest c_p - mean of the 4 lowest (W19 Z4, p515_s46_zero_solve_reports.py @ dd3afa6e); horizon average = sum w * spread / sum w with w = num_years * num_days (undiscounted) or w = num_years * num_days / 1.02^(y - 2025) (r2).

Units / sign: the dual is EUR per p.u.-hour of the representative-day objective; divided by baseMVA = 100 it is EUR/MWh; positive = the cost of serving one more MWh of load at bus 7 (equivalently the value of one more MWh injected there). The storage (load convention: charging positive) sees -LMP per MWh injected.

| year | day | market 4 h spread | bus-7 LMP 4 h spread (x = 0) | difference | market mean | LMP mean | max abs(LMP - pi) | LMP spread change over the last cycle |
|---|---|---|---|---|---|---|---|---|
| 2025 | Spring | 133.372 | 127.243 | +6.129 | 83.601 | 80.542 | 16.894 | +2.53e-07 |
| 2025 | Summer | 96.408 | 93.652 | +2.756 | 90.608 | 90.097 | 10.951 | +1.42e-05 |
| 2025 | Autumn | 56.260 | 55.957 | +0.303 | 104.315 | 104.443 | 1.342 | +3.33e-05 |
| 2025 | Winter | 46.276 | 42.643 | +3.634 | 103.116 | 102.511 | 10.829 | +1.14e-05 |
| 2030 | Spring | 150.898 | 91.610 | +59.287 | 94.587 | 70.528 | 80.503 | +1.33e-04 |
| 2030 | Summer | 118.994 | 91.067 | +27.927 | 111.426 | 103.177 | 40.179 | +3.38e-04 |
| 2030 | Autumn | 63.161 | 71.028 | -7.867 | 119.907 | 113.707 | 41.494 | -4.22e-09 |
| 2030 | Winter | 58.414 | 42.412 | +16.002 | 113.958 | 111.065 | 26.508 | -3.55e-05 |
| 2035 | Spring | 180.918 | 110.723 | +70.195 | 137.642 | 97.085 | 94.995 | +7.10e-03 |
| 2035 | Summer | 125.033 | 111.930 | +13.103 | 115.279 | 109.767 | 34.375 | -1.95e-05 |
| 2035 | Autumn | 96.430 | 81.246 | +15.184 | 89.704 | 84.100 | 49.843 | +2.83e-05 |
| 2035 | Winter | 47.846 | 46.839 | +1.007 | 136.893 | 132.444 | 39.486 | -4.91e-06 |

| scope | market 4 h spread | bus-7 LMP 4 h spread (x = 0) | ratio LMP / market |
|---|---|---|---|
| all years, day-weighted (undiscounted) | 97.991 | 80.610 | 0.82262 |
| all years, model block weight (2 %) | 97.018 | 80.365 | 0.82835 |
| 2025, day-weighted | 83.217 | 80.003 | 0.96139 |
| 2030, day-weighted | 98.012 | 74.078 | 0.75580 |
| 2035, day-weighted | 112.744 | 87.748 | 0.77829 |

Bus-7 LMP 4 h spread with the storage in place (same identity, terminal cycle of each record), horizon average: baseline: undiscounted 80.596, r2 80.351; sensitivity_c3: undiscounted 80.592, r2 80.347

**Reading.** At x = 0 the 4 h spread of the bus-7 marginal cost is 80.610 EUR/MWh against the market's 97.991 (ratio 0.8226, undiscounted day weights; per year 2025 0.9614, 2030 0.7558, 2035 0.7783). The largest hourly deviation |LMP - pi| on a representative day is 94.99 EUR/MWh; the largest change of a daily LMP spread over the last ADMM cycle is 7.10e-03 EUR/MWh (settled). With the storage in place the spread is 80.596 (baseline) / 80.592 (C3), undiscounted. Which TSO rows make the bus-7 marginal cost depart from pi (branch limits, voltage limits, losses) is NOT identifiable from the persisted data: the only persisted solved TSO block is 2025 Summer at cycle 6 (max |LMP_7 - pi| there 0.898 EUR/MWh).

## T3 -- attribution of value = Q(0) - Q(x): BASELINE

Record x `data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1` (candidate_key `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`) against x = 0 `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0` (candidate_key `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57`). Q(0) = 653,859,461.23, Q(x) = 653,600,033.46, value = 259,427.77; sum of parts = 259,427.77 (|diff| 6.71e-08 EUR). Resolution bar(x) + bar(x0) = 34,734.63.

Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage is 0 on these records (gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) unless stated. Weighted at 2 % (the model block weights; this is the convention of Q).

| agent | year | generation_cost | flexibility_cost_internal | load_curtailment | res_curtailment | ess_usage | detector_penalty_total | gross total |
|---|---|---|---|---|---|---|---|---|
| TSO | 2025 | 93,948.79 | 0.00 | 0.00 | 0.00 | 0.00 | 3.48 | 93,952.27 |
| TSO | 2030 | 20,766.33 | 0.00 | 0.00 | 0.00 | 0.00 | 3.15 | 20,769.49 |
| TSO | 2035 | 41,780.31 | 0.00 | 0.00 | 0.00 | 0.00 | 2.86 | 41,783.16 |
| TSO | all | 156,495.43 | 0.00 | 0.00 | 0.00 | 0.00 | 9.49 | 156,504.92 |
| DSO5 | 2025 | 0.00 | 2,348.79 | 0.00 | 0.00 | 0.00 | 0.00 | 2,348.79 |
| DSO5 | 2030 | 0.00 | 14,075.73 | 0.00 | 0.00 | 0.00 | -0.00 | 14,075.73 |
| DSO5 | 2035 | 0.00 | 8,528.17 | 0.00 | 0.00 | 0.00 | -0.00 | 8,528.17 |
| DSO5 | all | 0.00 | 24,952.69 | 0.00 | 0.00 | 0.00 | -0.00 | 24,952.69 |
| DSO7 | 2025 | 0.00 | 6,172.17 | 0.00 | 0.00 | 0.00 | 3.62 | 6,175.79 |
| DSO7 | 2030 | 0.00 | 23,529.07 | 0.00 | 0.00 | 0.00 | 3.28 | 23,532.35 |
| DSO7 | 2035 | 0.00 | 8,730.69 | 0.00 | 0.00 | 0.00 | 2.97 | 8,733.66 |
| DSO7 | all | 0.00 | 38,431.93 | 0.00 | 0.00 | 0.00 | 9.86 | 38,441.79 |
| DSO9 | 2025 | 0.00 | 2,390.39 | 0.00 | 0.00 | 0.00 | 0.00 | 2,390.39 |
| DSO9 | 2030 | 0.00 | 19,495.61 | 0.00 | 0.00 | 0.00 | -0.00 | 19,495.61 |
| DSO9 | 2035 | 0.00 | 17,642.36 | 0.00 | 0.00 | 0.00 | -0.00 | 17,642.36 |
| DSO9 | all | 0.00 | 39,528.36 | 0.00 | 0.00 | 0.00 | -0.00 | 39,528.36 |
| ALL | 2025 | 93,948.79 | 10,911.35 | 0.00 | 0.00 | 0.00 | 7.10 | 104,867.24 |
| ALL | 2030 | 20,766.33 | 57,100.41 | 0.00 | 0.00 | 0.00 | 6.43 | 77,873.17 |
| ALL | 2035 | 41,780.31 | 34,901.22 | 0.00 | 0.00 | 0.00 | 5.82 | 76,687.35 |
| ALL | all | 156,495.43 | 102,912.99 | 0.00 | 0.00 | 0.00 | 19.35 | 259,427.77 |

Detector subcomponents (all agents, all years, r2): voltage_slack -0.00, node_balance_slack 0.00, branch_flow_slack 0.00, flexibility_p_day_balance_slack -0.00, local_ess_day_balance_slack 0.00, shared_ess_day_balance_slack 19.35. Undiscounted rows are in the JSON.

## T3 -- attribution of value = Q(0) - Q(x): SENSITIVITY SET (C3; never compared in one table with the baseline)

Record x `data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1` (candidate_key `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`) against x = 0 `data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0` (candidate_key `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57`). Q(0) = 653,859,461.23, Q(x) = 653,597,653.72, value = 261,807.50; sum of parts = 261,807.50 (|diff| 3.71e-08 EUR). Resolution bar(x) + bar(x0) = 20,970.30.

Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage is 0 on these records (gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) unless stated. Weighted at 2 % (the model block weights; this is the convention of Q).

| agent | year | generation_cost | flexibility_cost_internal | load_curtailment | res_curtailment | ess_usage | detector_penalty_total | gross total |
|---|---|---|---|---|---|---|---|---|
| TSO | 2025 | 92,582.63 | 0.00 | 0.00 | 0.00 | 0.00 | 3.48 | 92,586.11 |
| TSO | 2030 | 15,006.85 | 0.00 | 0.00 | 0.00 | 0.00 | 3.15 | 15,010.01 |
| TSO | 2035 | 46,234.03 | 0.00 | 0.00 | 0.00 | 0.00 | 2.86 | 46,236.88 |
| TSO | all | 153,823.51 | 0.00 | 0.00 | 0.00 | 0.00 | 9.49 | 153,833.00 |
| DSO5 | 2025 | 0.00 | 2,342.69 | 0.00 | 0.00 | 0.00 | 0.00 | 2,342.69 |
| DSO5 | 2030 | 0.00 | 15,222.44 | 0.00 | 0.00 | 0.00 | -0.00 | 15,222.44 |
| DSO5 | 2035 | 0.00 | 10,337.45 | 0.00 | 0.00 | 0.00 | -0.00 | 10,337.45 |
| DSO5 | all | 0.00 | 27,902.58 | 0.00 | 0.00 | 0.00 | -0.00 | 27,902.58 |
| DSO7 | 2025 | 0.00 | 5,911.87 | 0.00 | 0.00 | 0.00 | 3.62 | 5,915.49 |
| DSO7 | 2030 | 0.00 | 23,750.46 | 0.00 | 0.00 | 0.00 | 3.28 | 23,753.74 |
| DSO7 | 2035 | 0.00 | 10,744.89 | 0.00 | 0.00 | 0.00 | 2.97 | 10,747.86 |
| DSO7 | all | 0.00 | 40,407.23 | 0.00 | 0.00 | 0.00 | 9.86 | 40,417.09 |
| DSO9 | 2025 | 0.00 | 2,527.66 | 0.00 | 0.00 | 0.00 | 0.00 | 2,527.66 |
| DSO9 | 2030 | 0.00 | 21,402.12 | 0.00 | 0.00 | 0.00 | -0.00 | 21,402.12 |
| DSO9 | 2035 | 0.00 | 15,725.05 | 0.00 | 0.00 | 0.00 | -0.00 | 15,725.05 |
| DSO9 | all | 0.00 | 39,654.83 | 0.00 | 0.00 | 0.00 | -0.00 | 39,654.83 |
| ALL | 2025 | 92,582.63 | 10,782.22 | 0.00 | 0.00 | 0.00 | 7.10 | 103,371.95 |
| ALL | 2030 | 15,006.85 | 60,375.03 | 0.00 | 0.00 | 0.00 | 6.43 | 75,388.32 |
| ALL | 2035 | 46,234.03 | 36,807.39 | 0.00 | 0.00 | 0.00 | 5.82 | 83,047.24 |
| ALL | all | 153,823.51 | 107,964.64 | 0.00 | 0.00 | 0.00 | 19.35 | 261,807.50 |

Detector subcomponents (all agents, all years, r2): voltage_slack -0.00, node_balance_slack 0.00, branch_flow_slack 0.00, flexibility_p_day_balance_slack -0.00, local_ess_day_balance_slack 0.00, shared_ess_day_balance_slack 19.35. Undiscounted rows are in the JSON.

## T4 -- captured spread decomposition: BASELINE

Record `data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1` (candidate_key `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`), ESSO terminal capture `data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/esso_capture/s39_D/node7_cycle112.jsonl`. Efficiencies in force: eff_ch = 0.97, eff_dch = 0.96 (round trip 0.9312); confirmed by recomputing the recorded EFC/day (max rel dev 7.0e-16).

Formula: Throughput T = sum_b w_b sum_p (eff_ch*pch + pdch/eff_dch)*dt / 2 (cell-side MWh of full cycles; W19 Z4 denominator E*EFC); captured = value / T. Grid injection at bus 7 = pdch - pch (ESSO terminal capture). A(LMP) = sum w * LMP*(pdch - pch)*dt; A_lossless = sum w * LMP*(pdch/eff_dch - eff_ch*pch)*dt; efficiency loss = A_lossless - A = sum w*LMP*[pch*(1-eff_ch) + pdch*(1/eff_dch-1)]*dt. Ideal = sum_b w_b * T_b * top4_minus_bottom4(series_b), T_b the block's daily throughput. value = dQ_TSO + dQ_DSO (T3); system remainder (V - A)/T = (dQ_TSO - A)/T + dQ_DSO/T. Terms telescope: day-weighted market spread - weighting - flatness - timing/depth - efficiency + system remainder = value / T exactly.

**Weighting r2** (Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); per MWh of full cell cycle). T = 4296.19 MWh-cycles, value 259,427.77, captured 60.386 EUR/MWh-cycle; resolution (bars / T) 8.085.

| term | EUR/MWh-cycle |
|---|---|
| available: market 4 h spread, day-weighted | +97.018 |
| seasonal/throughput weighting: day-weighted minus throughput-weighted market spread | -11.927 |
| (b) marginal-cost flatness: market minus bus-7 LMP(x=0) 4 h spread, throughput-weighted | -12.326 |
| dispatch timing & depth: ideal top-4/bottom-4 at LMP(x=0) minus the lossless value of the realized schedule | -6.534 |
| (a) round-trip efficiency losses valued at LMP(x=0) at the hours they occur | -7.475 |
| (d) system remainder: value minus the realized arbitrage at LMP(x=0) (price response of the system to the storage, ADMM stopping); split below into TSO and DN-side parts | +1.630 |
| **= captured (value / T)** | **60.386** |
|    of which TSO: TSO cost reduction minus the realized arbitrage at LMP(x=0) (not added again) | -22.326 |
|    of which (c) DN-side: DSO cost change (value contributions of DSO 5/7/9; + = DSO costs fall) (not added again) | +23.957 |

Four-way grouping: (a) efficiency -7.475; (b) marginal-cost flatness -12.326; (c) DN-side (part of the system remainder) +23.957; (d) residual = weighting + timing/depth + TSO part of the system remainder -40.787. Charge-weighted LMP 68.04, discharge-weighted LMP 134.28 EUR/MWh; analytic efficiency loss per cell cycle at these prices p_ch (1/eff_ch - 1) + p_dch (1 - eff_dch) = 7.476 EUR/MWh. System remainder with LMP(x) +1.649, with the trapezoid +1.640. value - A(LMP x=0) = 7,004.75 EUR = +0.202 x the resolution bar. dQ_TSO 156,504.92 (generation cost 156,495.43); A at LMP(x=0) 252,423.02 (TSO copy 252,409.54); dQ_DSO 102,922.85.

**Weighting undiscounted** (Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); per MWh of full cell cycle). T = 4659.62 MWh-cycles, value 284,326.97, captured 61.019 EUR/MWh-cycle; resolution (bars / T) 7.454.

| term | EUR/MWh-cycle |
|---|---|
| available: market 4 h spread, day-weighted | +97.991 |
| seasonal/throughput weighting: day-weighted minus throughput-weighted market spread | -11.961 |
| (b) marginal-cost flatness: market minus bus-7 LMP(x=0) 4 h spread, throughput-weighted | -12.949 |
| dispatch timing & depth: ideal top-4/bottom-4 at LMP(x=0) minus the lossless value of the realized schedule | -6.384 |
| (a) round-trip efficiency losses valued at LMP(x=0) at the hours they occur | -7.497 |
| (d) system remainder: value minus the realized arbitrage at LMP(x=0) (price response of the system to the storage, ADMM stopping); split below into TSO and DN-side parts | +1.819 |
| **= captured (value / T)** | **61.019** |
|    of which TSO: TSO cost reduction minus the realized arbitrage at LMP(x=0) (not added again) | -23.185 |
|    of which (c) DN-side: DSO cost change (value contributions of DSO 5/7/9; + = DSO costs fall) (not added again) | +25.004 |

Four-way grouping: (a) efficiency -7.497; (b) marginal-cost flatness -12.949; (c) DN-side (part of the system remainder) +25.004; (d) residual = weighting + timing/depth + TSO part of the system remainder -41.530. Charge-weighted LMP 68.08, discharge-weighted LMP 134.79 EUR/MWh; analytic efficiency loss per cell cycle at these prices p_ch (1/eff_ch - 1) + p_dch (1 - eff_dch) = 7.497 EUR/MWh. System remainder with LMP(x) +1.838, with the trapezoid +1.829. value - A(LMP x=0) = 8,476.42 EUR = +0.244 x the resolution bar. dQ_TSO 167,816.91 (generation cost 167,806.46); A at LMP(x=0) 275,850.56 (TSO copy 275,829.43); dQ_DSO 116,510.07.

## T4 -- captured spread decomposition: SENSITIVITY SET (C3)

Record `data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1` (candidate_key `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a`), ESSO terminal capture `data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1/esso_capture/s39_D/node7_cycle125.jsonl`. Efficiencies in force: eff_ch = 0.97, eff_dch = 0.96 (round trip 0.9312); confirmed by recomputing the recorded EFC/day (max rel dev 5.8e-16).

Formula: Throughput T = sum_b w_b sum_p (eff_ch*pch + pdch/eff_dch)*dt / 2 (cell-side MWh of full cycles; W19 Z4 denominator E*EFC); captured = value / T. Grid injection at bus 7 = pdch - pch (ESSO terminal capture). A(LMP) = sum w * LMP*(pdch - pch)*dt; A_lossless = sum w * LMP*(pdch/eff_dch - eff_ch*pch)*dt; efficiency loss = A_lossless - A = sum w*LMP*[pch*(1-eff_ch) + pdch*(1/eff_dch-1)]*dt. Ideal = sum_b w_b * T_b * top4_minus_bottom4(series_b), T_b the block's daily throughput. value = dQ_TSO + dQ_DSO (T3); system remainder (V - A)/T = (dQ_TSO - A)/T + dQ_DSO/T. Terms telescope: day-weighted market spread - weighting - flatness - timing/depth - efficiency + system remainder = value / T exactly.

**Weighting r2** (Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); per MWh of full cell cycle). T = 5023.69 MWh-cycles, value 261,807.50, captured 52.115 EUR/MWh-cycle; resolution (bars / T) 4.174.

| term | EUR/MWh-cycle |
|---|---|
| available: market 4 h spread, day-weighted | +97.018 |
| seasonal/throughput weighting: day-weighted minus throughput-weighted market spread | -13.779 |
| (b) marginal-cost flatness: market minus bus-7 LMP(x=0) 4 h spread, throughput-weighted | -11.648 |
| dispatch timing & depth: ideal top-4/bottom-4 at LMP(x=0) minus the lossless value of the realized schedule | -15.440 |
| (a) round-trip efficiency losses valued at LMP(x=0) at the hours they occur | -7.551 |
| (d) system remainder: value minus the realized arbitrage at LMP(x=0) (price response of the system to the storage, ADMM stopping); split below into TSO and DN-side parts | +3.514 |
| **= captured (value / T)** | **52.115** |
|    of which TSO: TSO cost reduction minus the realized arbitrage at LMP(x=0) (not added again) | -17.979 |
|    of which (c) DN-side: DSO cost change (value contributions of DSO 5/7/9; + = DSO costs fall) (not added again) | +21.493 |

Four-way grouping: (a) efficiency -7.551; (b) marginal-cost flatness -11.648; (c) DN-side (part of the system remainder) +21.493; (d) residual = weighting + timing/depth + TSO part of the system remainder -47.197. Charge-weighted LMP 74.78, discharge-weighted LMP 130.97 EUR/MWh; analytic efficiency loss per cell cycle at these prices p_ch (1/eff_ch - 1) + p_dch (1 - eff_dch) = 7.552 EUR/MWh. System remainder with LMP(x) +3.538, with the trapezoid +3.526. value - A(LMP x=0) = 17,655.13 EUR = +0.842 x the resolution bar. dQ_TSO 153,833.00 (generation cost 153,823.51); A at LMP(x=0) 244,152.38 (TSO copy 244,318.21); dQ_DSO 107,974.50.

**Weighting undiscounted** (Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); per MWh of full cell cycle). T = 5471.35 MWh-cycles, value 287,840.86, captured 52.609 EUR/MWh-cycle; resolution (bars / T) 3.833.

| term | EUR/MWh-cycle |
|---|---|
| available: market 4 h spread, day-weighted | +97.991 |
| seasonal/throughput weighting: day-weighted minus throughput-weighted market spread | -13.944 |
| (b) marginal-cost flatness: market minus bus-7 LMP(x=0) 4 h spread, throughput-weighted | -12.218 |
| dispatch timing & depth: ideal top-4/bottom-4 at LMP(x=0) minus the lossless value of the realized schedule | -15.560 |
| (a) round-trip efficiency losses valued at LMP(x=0) at the hours they occur | -7.571 |
| (d) system remainder: value minus the realized arbitrage at LMP(x=0) (price response of the system to the storage, ADMM stopping); split below into TSO and DN-side parts | +3.911 |
| **= captured (value / T)** | **52.609** |
|    of which TSO: TSO cost reduction minus the realized arbitrage at LMP(x=0) (not added again) | -18.446 |
|    of which (c) DN-side: DSO cost change (value contributions of DSO 5/7/9; + = DSO costs fall) (not added again) | +22.356 |

Four-way grouping: (a) efficiency -7.571; (b) marginal-cost flatness -12.218; (c) DN-side (part of the system remainder) +22.356; (d) residual = weighting + timing/depth + TSO part of the system remainder -47.950. Charge-weighted LMP 74.99, discharge-weighted LMP 131.30 EUR/MWh; analytic efficiency loss per cell cycle at these prices p_ch (1/eff_ch - 1) + p_dch (1 - eff_dch) = 7.571 EUR/MWh. System remainder with LMP(x) +3.935, with the trapezoid +3.923. value - A(LMP x=0) = 21,397.42 EUR = +1.020 x the resolution bar. dQ_TSO 165,520.87 (generation cost 165,510.43); A at LMP(x=0) 266,443.44 (TSO copy 266,625.14); dQ_DSO 122,319.99.

## T5 -- planning-signal magnitude (baseline points)

Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage is 0 on these records (gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) unless stated.

| quantity | EUR | / Q(0) | log10 |
|---|---|---|---|
| value (baseline smallest node-7 unit) | 259,427.77 | 3.968e-04 | -3.401 |
| sigma_Q = s (Phase A T3 regression residual std, dof 13) | 11,410.62 | 1.745e-05 | -4.758 |
| sigma_Q = residual rms (Phase A T3) | 10,285.39 | 1.573e-05 | -4.803 |
| sigma_Q = residual max abs (Phase A T3) | 18,449.66 | 2.822e-05 | -4.549 |
| bar(x0) | 9,629.98 | 1.473e-05 | -4.832 |
| bar(x) baseline | 25,104.65 | 3.839e-05 | -4.416 |
| bar(x) + bar(x0) (resolution of the baseline value) | 34,734.63 | 5.312e-05 | -4.275 |

value / Q(0) = 3.968e-04 = 10^-3.40: the planning signal is 3.40 orders of magnitude below the recourse, i.e. between 3 and 4 (1 part in 2,520); sigma_Q / Q(0) = 10^-4.80 to 10^-4.55; bar(x)+bar(x0) / Q(0) = 10^-4.27; the value exceeds sigma_Q by 14.1x to 25.2x.

## Paper-scale R on the marginal-cost spread

The storage sees the bus-7 marginal cost (T1), whose SRP1 4 h spread is f_SRP1 = 0.8226 of the market spread (per year 2025 0.9614, 2030 0.7558, 2035 0.7783). Recomputing W22's paper-scale R on the marginal-cost spread needs the paper-scale bus-7 duals at an ADMM equilibrium; no paper-scale ADMM evaluation has reached one (searched: data/SRP1/Results/P515S44/scale_measurement -- paper_cycle_snapoff_r1/r2 are one-cycle timing runs, both watchdog-aborted, exit 97; the "paper_plan" records under P515S44 are SRP1-scale runs of the paper's plan), so it CANNOT be computed without solves and is not computed. R on the market spread (W22 @ 046d4d00: 0.9361) equals R on the marginal-cost spread only if f_paper = f_SRP1. Restated prediction: R_LMP = 0.9361 x f_paper / 0.8226. Conditional range, ASSUMING f_paper lies within SRP1's per-year range [0.7558, 0.9614]: R_LMP in [0.8600, 1.0940] (an assumption, not a measurement). Cheapest solve-bearing route: none extra -- the planned paper-scale x = 0 evaluation, run with the PF entry-stride capture (lambda_dso per cycle) and one TSO snapshot for the identity check, yields f_paper at zero additional solves; the identity must be re-derived there for 5 x 5 scenarios (probability weights and the scenario-deviation penalty enter the stationarity conditions).

## Checks

- T1_esso_model_file_has_no_market_price_reference: True
- T1_no_priced_generator_at_bus7: True
- T1_tso_only_loads_are_the_three_adn_interfaces: True
- T2_baseline_pf_stride_last_cycle_is_terminal: True
- T2_baseline_run_prices_equal_z4_arrays: True
- T2_baseline_sigma_equals_case_file_objective_scale: True
- T2_baseline_snapshot_delta_free_interior_pc_fixed: True
- T2_baseline_snapshot_eff_equals_sigma_over_w: True
- T2_baseline_snapshot_identity_within_0.0001: True
- T2_baseline_snapshot_settlement_weight_1_proximal_off: True
- T2_baseline_terminal_interface_delta_interior_every_block_period: True
- T2_sensitivity_c3_pf_stride_last_cycle_is_terminal: True
- T2_sensitivity_c3_run_prices_equal_z4_arrays: True
- T2_sensitivity_c3_sigma_equals_case_file_objective_scale: True
- T2_sensitivity_c3_snapshot_delta_free_interior_pc_fixed: True
- T2_sensitivity_c3_snapshot_eff_equals_sigma_over_w: True
- T2_sensitivity_c3_snapshot_identity_within_0.0001: True
- T2_sensitivity_c3_snapshot_settlement_weight_1_proximal_off: True
- T2_sensitivity_c3_terminal_interface_delta_interior_every_block_period: True
- T2_x0_pf_stride_last_cycle_is_terminal: True
- T2_x0_run_prices_equal_z4_arrays: True
- T2_x0_sigma_equals_case_file_objective_scale: True
- T2_x0_snapshot_delta_free_interior_pc_fixed: True
- T2_x0_snapshot_eff_equals_sigma_over_w: True
- T2_x0_snapshot_identity_within_0.0001: True
- T2_x0_snapshot_settlement_weight_1_proximal_off: True
- T2_x0_terminal_interface_delta_interior_every_block_period: True
- T3_baseline_block_weights_match_formula: True
- T3_baseline_detector_subcomponents_sum_to_detector_total: True
- T3_baseline_gross_equals_net_both_records: True
- T3_baseline_parts_sum_to_value_within_0.01_eur: True
- T3_baseline_value_matches_task_text_2dp: True
- T3_sensitivity_c3_block_weights_match_formula: True
- T3_sensitivity_c3_detector_subcomponents_sum_to_detector_total: True
- T3_sensitivity_c3_gross_equals_net_both_records: True
- T3_sensitivity_c3_parts_sum_to_value_within_0.01_eur: True
- T3_sensitivity_c3_value_matches_task_text_2dp: True
- T4_baseline_efc_recomputed_with_production_efficiencies: True
- T4_baseline_esso_capture_288_rows_single_cohort: True
- T4_baseline_esso_capture_index_mapping_matches_ess_stride: True
- T4_baseline_r2_terms_telescope_to_captured: True
- T4_baseline_undiscounted_terms_telescope_to_captured: True
- T4_sensitivity_c3_efc_recomputed_with_production_efficiencies: True
- T4_sensitivity_c3_esso_capture_288_rows_single_cohort: True
- T4_sensitivity_c3_esso_capture_index_mapping_matches_ess_stride: True
- T4_sensitivity_c3_r2_terms_telescope_to_captured: True
- T4_sensitivity_c3_reproduces_W19_captured_52.11: True
- T4_sensitivity_c3_undiscounted_terms_telescope_to_captured: True
- Z4_market_spread_reproduced_r2: True
- Z4_market_spread_reproduced_undiscounted: True
- baseline_candidate_key_as_expected: True
- baseline_certified: True
- baseline_child_manifest_tracked: True
- baseline_component_levels_terminal_exists: True
- baseline_component_levels_terminal_settled_by_git_or_committed_manifest_hash: True
- baseline_ess_entry_stride_exists: True
- baseline_ess_entry_stride_settled_by_git_or_committed_manifest_hash: True
- baseline_esso_capture_terminal_exists: True
- baseline_esso_capture_terminal_settled_by_git_or_committed_manifest_hash: True
- baseline_evaluation_record_exists: True
- baseline_evaluation_record_settled_by_git_or_committed_manifest_hash: True
- baseline_interface_settlement_detail_exists: True
- baseline_interface_settlement_detail_settled_by_git_or_committed_manifest_hash: True
- baseline_pf_entry_stride_exists: True
- baseline_pf_entry_stride_settled_by_git_or_committed_manifest_hash: True
- baseline_tso_snapshot_cycle7_exists: True
- baseline_tso_snapshot_cycle7_settled_by_git_or_committed_manifest_hash: True
- case_params_file_is_the_one_every_record_ran_with: True
- sensitivity_c3_candidate_key_as_expected: True
- sensitivity_c3_certified: True
- sensitivity_c3_child_manifest_tracked: True
- sensitivity_c3_component_levels_terminal_exists: True
- sensitivity_c3_component_levels_terminal_settled_by_git_or_committed_manifest_hash: True
- sensitivity_c3_ess_entry_stride_exists: True
- sensitivity_c3_ess_entry_stride_settled_by_git_or_committed_manifest_hash: True
- sensitivity_c3_esso_capture_terminal_exists: True
- sensitivity_c3_esso_capture_terminal_settled_by_git_or_committed_manifest_hash: True
- sensitivity_c3_evaluation_record_exists: True
- sensitivity_c3_evaluation_record_settled_by_git_or_committed_manifest_hash: True
- sensitivity_c3_interface_settlement_detail_exists: True
- sensitivity_c3_interface_settlement_detail_settled_by_git_or_committed_manifest_hash: True
- sensitivity_c3_pf_entry_stride_exists: True
- sensitivity_c3_pf_entry_stride_settled_by_git_or_committed_manifest_hash: True
- sensitivity_c3_tso_snapshot_cycle7_exists: True
- sensitivity_c3_tso_snapshot_cycle7_settled_by_git_or_committed_manifest_hash: True
- tracked::data/SRP1/Results/P515S45/phase_a_tables/phase_a_tables.json: True
- tracked::data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json: True
- tracked::data/SRP1/Results/P515S47/market_spreads/market_spreads.json: True
- tracked::data/SRP1/SRP1.json: True
- tracked::data/SRP1/SRP1_params.json: True
- tracked::data/SRP1/case9/case9_2025.json: True
- x0_candidate_key_as_expected: True
- x0_certified: True
- x0_child_manifest_tracked: True
- x0_component_levels_terminal_exists: True
- x0_component_levels_terminal_settled_by_git_or_committed_manifest_hash: True
- x0_evaluation_record_exists: True
- x0_evaluation_record_settled_by_git_or_committed_manifest_hash: True
- x0_interface_settlement_detail_exists: True
- x0_interface_settlement_detail_settled_by_git_or_committed_manifest_hash: True
- x0_pf_entry_stride_exists: True
- x0_pf_entry_stride_settled_by_git_or_committed_manifest_hash: True
- x0_tso_snapshot_cycle7_exists: True
- x0_tso_snapshot_cycle7_settled_by_git_or_committed_manifest_hash: True
