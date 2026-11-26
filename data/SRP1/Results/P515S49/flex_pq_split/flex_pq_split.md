# P5.15 Addenda 33-34 W29 -- flexibility (congestion relief / economic shifting) and P/Q splits (zero solves)

Instances: x = 0 `8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57` (`data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0`, s48_x0_capture, status certified, 132 cycles, Q gross 653,859,461.23, bar 9,629.98, terminal step / threshold 0.0034; persisted models `data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/certified_models.pkl` sha256 03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce); UNIT node 7 0.25 MVA / 1.0 MWh, 2025, nodes 5/9 empty `db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a` (`data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1`, s47_recert, status certified, 112 cycles, Q gross 653,600,033.46, bar 25,104.65, terminal step / threshold 0.0567; RECORDS ONLY, no persisted models). Baseline = C2 + phi_cal 0.985 + soh_min 0.70.

Solve profile: armed SolveProfileGuard(permitted=()); counts {'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}; verify(0) failures []. Git HEAD 8f9d67f2c6c4e9233c79532eb8807983643d8e9f. all_pass = True; failed checks: [].

Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage 0 on both records (gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) (r2) unless stated.

## Z33 (a) -- the cost_flex profile actually applied

Applied c_flex = (coefficient of flex_p_down[c, p] in flex_cost_scenario[0, 0]) / baseMVA, read from every persisted x = 0 DSO block (3 DSOs x 12 blocks); identical across the three DSOs (max dev 0.0e+00) and across all priced loads (max dev 0.0e+00). Production builds it as cost_flex[year][day] = (one sampled synthetic flexibility-price profile per (year, day), market scenario count 1) x (1 + g_flex)^(year - 2025) (shared_resources_planning.py:8347; sampling shared_resources_planning.py:8339; cumul shared_resources_planning.py:8333; g read at shared_resources_planning.py:8322 from data/SRP1/MarketData/SRP1_market_data.xlsx sheet "Growth Factors": Flexibility 0.02, Energy 0.025) and binds it to every DSO block (shared_resources_planning.py:8133). Range 21.28 to 92.70 EUR/MWh; day-weighted mean 48.37 (2025 43.51, 2030 48.04, 2035 53.56). P and Q carry the SAME price: flexibility_cost (model_construction_helpers.py:1692) charges c_flex[p] * baseMVA * (flex_p_down + flex_q_down) (model_construction_helpers.py:1709); confirmed on the built objective: coefficient(flex_p_down[c,p]) = coefficient(flex_q_down[c,p]) for every priced load and period (max abs dev 0.0e+00 EUR per p.u.). flex_p_up and flex_q_up are NOT priced. BUT the reactive flexibility is structurally absent: its bounds are set to EQUALITY_TOLERANCE (1e-5 p.u.) for every load (network.py:1276, network.py:1277), i.e. ub(flex_q_*) = 2e-5 p.u. = 2 kVAr per load in the built models (checked: max ub 2.0e-05 p.u.), and the Q day-balance row is not wired (network.py:476); the P day balance is (network.py:475).

Citations: market_workbook_read_growth_factors_sheet shared_resources_planning.py:8353; flexibility_growth_factor_read shared_resources_planning.py:8322; flexibility_growth_cumul shared_resources_planning.py:8333; flexibility_profile_sampled_per_year_day shared_resources_planning.py:8339; cost_flex_built shared_resources_planning.py:8347; cost_flex_bound_to_dso_blocks shared_resources_planning.py:8133; flexibility_cost_def model_construction_helpers.py:1692; flexibility_cost_term_p_down_plus_q_down_same_c_flex model_construction_helpers.py:1709; tso_adn_interface_loads_excluded model_construction_helpers.py:1700; flex_p_day_balance_rule model_construction_helpers.py:564; flex_q_day_balance_NOT_wired network.py:476; flex_p_day_balance_wired network.py:475; reactive_flex_up_bound_is_EQUALITY_TOLERANCE network.py:1276; reactive_flex_down_bound_is_EQUALITY_TOLERANCE network.py:1277; qc_flex_down_bounds model_construction_helpers.py:322; EQUALITY_TOLERANCE definitions.py:90; shared_storage_in_node_balance_load_convention model_construction_helpers.py:1330; dso_interface_q_def_storage_at_reference_bus model_construction_helpers.py:1219; voltage_upper_row model_construction_helpers.py:448; tso_adn_bus_voltage_slacks_fixed_0 shared_resources_planning.py:3759; sess_converter_capability_pq_circle model_construction_helpers.py:811; esso_objective_feasibility_penalty_only shared_energy_storage_data.py:853

| year | day | growth cumul (1+g)^(y-2025) | min | max | mean | top-4 mean | bottom-4 mean | base (applied / cumul) mean |
|---|---|---|---|---|---|---|---|---|
| 2025 | Spring | 1.000000 | 21.278 | 55.334 | 39.205 | 52.290 | 22.892 | 39.205 |
| 2025 | Summer | 1.000000 | 33.378 | 61.717 | 47.102 | 58.160 | 35.970 | 47.102 |
| 2025 | Autumn | 1.000000 | 44.032 | 76.381 | 54.759 | 71.570 | 44.997 | 54.759 |
| 2025 | Winter | 1.000000 | 21.432 | 44.135 | 33.029 | 41.015 | 22.480 | 33.029 |
| 2030 | Spring | 1.104081 | 25.030 | 60.502 | 43.267 | 56.277 | 25.962 | 39.188 |
| 2030 | Summer | 1.104081 | 38.188 | 69.644 | 51.786 | 65.528 | 39.222 | 46.905 |
| 2030 | Autumn | 1.104081 | 48.615 | 84.331 | 60.458 | 79.019 | 49.681 | 54.759 |
| 2030 | Winter | 1.104081 | 23.897 | 49.934 | 36.708 | 45.715 | 24.938 | 33.248 |
| 2035 | Spring | 1.218994 | 26.281 | 66.464 | 47.850 | 63.075 | 28.208 | 39.254 |
| 2035 | Summer | 1.218994 | 41.114 | 75.711 | 57.347 | 71.273 | 43.688 | 47.045 |
| 2035 | Autumn | 1.218994 | 53.671 | 92.699 | 67.127 | 86.418 | 54.688 | 55.067 |
| 2035 | Winter | 1.218994 | 27.439 | 55.031 | 41.988 | 52.184 | 28.600 | 34.445 |

| year | day-weighted mean c_flex (EUR/MWh) | min | max |
|---|---|---|---|
| 2025 | 43.512 | 21.278 | 76.381 |
| 2030 | 48.042 | 23.897 | 84.331 |
| 2035 | 53.562 | 26.281 | 92.699 |
| all (undiscounted day weights) | 48.372 | 21.278 | 92.699 |

Hourly profiles (24 values per block) are in the JSON (`Z33a.per_block[*].hourly`).

## Z33 (b) -- congestion relief (node-7 binding slots) vs economic shifting

Binding set S = the 23 of 288 (year, day, hour) slots where the node-7 DSO interface apparent flow sqrt(P^2 + Q^2) (interface_settlement_detail, DSO side) is >= 99% of its 100 MVA rating -- W19 Z1's slot set (dd3afa6e), recomputed here on the x = 0 record (equal: True) and on the unit record (equal: True). S lies in 7 blocks: 2025 Spring, 2025 Summer, 2030 Spring, 2030 Summer, 2035 Autumn, 2035 Spring, 2035 Summer; hours 9-13. The same slots are applied to DSOs 5 and 9 (whose own interfaces are never near rating: W19 max utilisation 0.51 / 0.67), so for them "congestion relief" means "flexibility bought in the node-7 binding hours".

**x = 0 (exact, per slot from the persisted DSO blocks; weighted r2, EUR; volumes MWh / MVArh, weighted by the same block weights).** P-up and Q-up carry no cost (the objective prices only flex_p_down and flex_q_down).

| DSO | group | slots | P-up MWh | P-down MWh | Q-up MVArh | Q-down MVArh | cost P-down | cost Q-down | cost total | share of DSO flex cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | congestion relief | 23 | 568,877.29 | -0.25 | 56.8905 | -0.2236 | -8.99 | -7.9827 | -16.97 | -0.0000 |
| 5 | economic shifting | 265 | 622,389.20 | 1,191,266.75 | 4887.0069 | -2.8132 | 65,158,558.37 | -141.2515 | 65,158,417.12 | 1.0000 |
| 7 | congestion relief | 23 | 555,725.35 | -0.23 | 2.0157 | -0.2187 | -8.09 | -7.8302 | -15.92 | -0.0000 |
| 7 | economic shifting | 265 | 656,243.87 | 1,211,969.44 | 4756.0293 | -2.8078 | 66,076,978.67 | -141.0077 | 66,076,837.66 | 1.0000 |
| 9 | congestion relief | 23 | 662,355.18 | -0.26 | 602.8203 | -0.2260 | -9.00 | -8.0694 | -17.07 | -0.0000 |
| 9 | economic shifting | 265 | 745,720.39 | 1,408,075.82 | 6223.1653 | -2.8127 | 76,970,246.38 | -141.2026 | 76,970,105.18 | 1.0000 |

x = 0: the flexibility cost in the node-7 binding slots is -16.97, -15.92, -17.07 EUR (DSO 5, 7, 9; r2), i.e. zero to within IPOPT's bound relaxation (|.| <= eps_S = 21.65, 21.65, 21.65 EUR over the 23 slots, the values being slightly negative P-down / Q-down at the relaxed lower bound). No DSO buys downward flexibility in those hours. In the binding slots the DSOs move load UP (unpriced P-up): P-up in S = 568,877, 555,725, 662,355 MWh (weighted) = 0.478, 0.459, 0.470 of each DSO's total P-up; the priced P-down leg of the same day-balanced shift lies entirely outside S. Under the stated definition the whole flexibility cost at x = 0 is economic shifting.

**UNIT (block totals only -- no per-slot flexibility is recorded for it; weighted r2, EUR).** CR is exact (= 0) in the blocks without a binding slot; in the 7 blocks with binding slots only the interval stated in the formula is available.

| DSO | FC(0) | FC(x) | saving FC(0) - FC(x) | CR(0) exact | CR(x) interval | saving on CR interval | ES(0) exact | ES(x) interval | saving on ES interval | ES(x) exact part (blocks without S) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 65,158,400.15 | 65,133,447.46 | 24,952.69 | -16.97 | [-220.19, 25,504,718.85] | [-25,504,735.83, 203.22] | 65,158,417.12 | [39,628,728.60, 65,133,667.64] | [24,749.47, 25,529,688.52] | 5,922,033.45 |
| 7 | 66,076,821.74 | 66,038,389.81 | 38,431.93 | -15.92 | [-220.19, 29,794,745.28] | [-29,794,761.20, 204.27] | 66,076,837.66 | [36,243,644.53, 66,038,610.00] | [38,227.67, 29,833,193.13] | 5,136,563.61 |
| 9 | 76,970,088.11 | 76,930,559.75 | 39,528.36 | -17.07 | [-220.19, 29,750,715.87] | [-29,750,732.94, 203.12] | 76,970,105.18 | [47,179,843.88, 76,930,779.94] | [39,325.24, 29,790,261.30] | 6,862,996.41 |

Saving restricted to the blocks WITHOUT a binding slot (exact, all economic shifting, r2): DSO 5 13,424.66; DSO 7 26,393.79; DSO 9 18,505.36. Saving in the 7 blocks WITH binding slots (block totals, r2): DSO 5 11,528.03; DSO 7 12,038.15; DSO 9 21,023.00.

Resolution: value and savings are differences of two ADMM runs; bar(x0) + bar(unit) = 34,734.63 EUR (x0 bar 9,629.98, terminal step / threshold 0.0034; unit bar 25,104.65, terminal step / threshold 0.0567). The bar bounds the TOTAL recourse; its split across agents/blocks is not recorded, so a per-DSO or per-group saving smaller than the bar is not shown to be determinate by this bar (each per-DSO saving here is of the same order as the bar).

NOT COMPUTABLE for the unit without its terminal DSO models: the per-slot flexibility (hence the exact CR/ES split inside the 7 blocks that contain binding slots, and the per-slot P-up / P-down / Q volumes). Searched: the unit record directory (every *.json / *.jsonl key containing "flex": only component_levels_terminal / evaluation_record block totals, interface_settlement_detail flexibility_volumes_per_dso = TSO interface_delta totals, recourse_jump_sidecar per-block totals), results/FrozenSMOPF (cycle-7 pre-solve snapshots only, not terminal), esso_capture (ESSO rows only). What it would take: the unit's terminal DSO blocks -- a re-evaluation of the unit with persist_certified_models (one ~1 h evaluation; solves, not authorized here).

## Z34 (a) -- P/Q split of the storage value and of each DSO's flexibility saving

Storage: value = Q(0) - Q(x) (exact, records). Attributed split at the x = 0 marginal values (the exact terminal duals of the persisted x = 0 TSO blocks at bus 7, where the storage injects in the TSO model; in the DSO model its P and Q enter the reference-bus balance and the interface definition and cancel -- model_construction_helpers.py:1219, W19 Z1): A_P = sum_b w_b sum_p LMP_P,7 * (pdch - pch) (ESSO terminal capture; W25's A0), A_Q = sum_b w_b sum_p LMP_Q,7 * (-qnet) (qnet = the TSO copy of the ESS Q consensus in the unit's terminal ess stride, load convention, MVAr; sign/units verified against the unit's persisted cycle-7 TSO snapshot). LMP_P,7 = dual(node_balance_p[7]) / baseMVA, LMP_Q,7 = dual(node_balance_q[7]) / baseMVA (EUR per MWh / MVArh of load at bus 7). remainder = value - A_P - A_Q. DSO flexibility saving: saving = FC(0) - FC(x) (exact, records); FC_P(0) and FC_Q(0) exact from the x = 0 models; at the unit FC_Q(x) is bounded by the structural reactive-flexibility bound (formula in the header), which bounds saving_Q and saving_P = saving - saving_Q. EXACT: value, saving, FC_P(0), FC_Q(0), the duals. ATTRIBUTED: A_P, A_Q (first-order pricing of the dispatch at fixed marginal values). BOUNDED (not exact): the unit's P/Q flexibility split.

| quantity | r2 (EUR) | undiscounted (EUR) | status |
|---|---|---|---|
| value = Q(0) - Q(x) (records) | 259,427.77 | 284,326.97 | exact |
| A_P: P dispatch (ESSO -pnet) at LMP_P,7(x=0) (terminal TSO duals) | 252,423.02 | 275,850.56 | attributed (first order) |
| A_Q: Q dispatch (TSO copy -qnet) at LMP_Q,7(x=0) (terminal TSO duals) | 309.15 | 342.22 | attributed (first order) |
| remainder = value - A_P - A_Q (system response incl. the DSO saving; ADMM stopping) | 6,695.59 | 8,134.19 | residual |
|   check: A_P with LMP_P(x) (identity on the unit record) | 252,341.75 | 275,761.53 | attributed |
|   check: A_Q with LMP_Q(x) (identity on the unit record) | 308.92 | 341.97 | attributed |
|   check: A_Q at LMP_Q(x=0) with the consensus z / DSO copy / ESSO copy | 309.15 | 342.22 | DSO copy 309.1510, ESSO copy 309.1478 (r2) |
|   |Q| throughput of the storage, MVArh (TSO copy) | 1,346.29 | 1,481.25 | signed (absorbing +) 1346.291 MVArh (r2) |

| DSO | saving (exact, records) | FC_P(0) exact | FC_Q(0) exact | FC_Q(x) interval | saving_Q interval | saving_P interval |
|---|---|---|---|---|---|---|
| 5 | 24,952.69 | 65,158,549.38 | -149.2342 | [-183.5599, 367119.7302] | [-367268.9644, 34.3257] | [24,918.37, 392,221.66] |
| 7 | 38,431.93 | 66,076,970.58 | -148.8379 | [-183.5599, 367119.7302] | [-367268.5681, 34.7220] | [38,397.21, 405,700.50] |
| 9 | 39,528.36 | 76,970,237.38 | -149.2719 | [-183.5599, 367119.7302] | [-367269.0022, 34.2879] | [39,494.07, 406,797.36] |

Storage: A_P = 252,423.02 (0.9730 of the value; W25 committed A0 252,423.02), A_Q = 309.1525 EUR (1.19e-03 of the value), remainder 6,695.59. The reactive marginal value at bus 7 at x = 0 is -0.4396 to -0.0038 EUR/MVArh (mean |.| 0.1836) against LMP_P,7 mean 99.96 EUR/MWh. DSOs: FC_Q(0) = -149.2342, -148.8379, -149.2719 EUR (DSO 5, 7, 9) and |FC_Q(x)| <= 367119.73, 367119.73, 367119.73 EUR by the structural bound (sum 1101359.19). Because FC_Q(0) is already zero (at the relaxed bound), the storage cannot SAVE reactive flexibility cost: saving_Q <= 34.33, 34.72, 34.29 EUR, hence saving_P >= saving - that, i.e. at least 24,918.37, 38,397.21, 39,494.07 EUR of the DSO savings (DSO 5, 7, 9) is active-power (P-down) flexibility. The other side of the interval (the unit buying MORE reactive flexibility, saving_Q < 0) is bounded only by the structural bound, which is wide (2e-5 p.u. per load-hour, priced over the horizon) and so uninformative; without the unit's DSO models the Q part is not determined beyond: saving_Q <= ~0.

## Z34 (b) -- the storage's Q dispatch as a fraction of its 0.25 MVA rating

Terminal cycle 112 of the unit's ess_entry_stride_baseline.jsonl, node 7, power_type "q" (MVAr, load convention: + = absorbing; per copy x.tso / x.dso / x.esso and the consensus z); fraction = |q| / 0.25 MVA, per slot (288 = 12 blocks x 24 h). esso_capture holds no Q field (pch, pdch, pnet only), so the stride is the only per-slot Q source. The unit's S3 duplicate gives identical terminal z (max dev 0.0e+00).

| series | min | median | p90 | p99 | max | slots > 1 % | slots > 10 % | slots > 50 % | absorbing (q > 0) | injecting (q < 0) |
|---|---|---|---|---|---|---|---|---|---|---|
| |q| / S (consensus z) | 0.00101 | 0.04967 | 0.07573 | 0.08449 | 0.08606 | 254 | 0 | 0 | 288 | 0 |
| |q| / S (TSO copy) | 0.00101 | 0.04967 | 0.07573 | 0.08449 | 0.08606 | 254 | 0 | 0 | 288 | 0 |
| |q| / S (DSO copy) | 0.00101 | 0.04967 | 0.07573 | 0.08449 | 0.08606 | 254 | 0 | 0 | 288 | 0 |
| |q| / S (ESSO copy) | 0.00101 | 0.04967 | 0.07573 | 0.08449 | 0.08606 | 254 | 0 | 0 | 288 | 0 |
| |q| / S (z) in the 23 binding slots | 0.00403 | 0.00786 | 0.01134 | 0.01297 | 0.01306 | 5 | 0 | 0 | 23 | 0 |
| sqrt(p^2+q^2) / S (z, converter loading) | 0.00412 | 0.07969 | 0.99998 | 1.00004 | 1.00053 | 276 | 126 | 77 |  |  |

Q is NOT idle: median |q|/S = 0.0497, max 0.0861 (2035 Spring h2), 254 of 288 slots above 1 % of rating, 288 absorbing / 0 injecting; converter loading max 1.0005; 43 slots at >= 99 % converter loading (|q|/S there 0.0010 to 0.0611: Q shares the P-Q circle with P). Its value at the bus-7 reactive marginal cost is A_Q = 309.1525 EUR (r2) over the horizon.

## Z34 (c) -- interface voltage-bound activity (entries at the 1.1 pu bound)

At the bound = v_max - v <= 1e-06 pu (the convention of the production summary "n_entries_at_bound_within_1e-6_pu" in interface_voltage_terminal.json), v = TSO-side voltage magnitude at the ADN bus; also counted at 1e-5 / 1e-4 / 1e-3 pu. The TSO voltage rows at the ADN buses are HARD (their slacks are fixed at 0: shared_resources_planning.py:3759; checked on the models). x = 0 from the persisted models (the vmag Var at the ADN buses; sqrt(vmag_sqr) elsewhere) and, as a cross-check, from the x = 0 record (max dev 0.0e+00 pu; the model's vmag Var is used, max |vmag - sqrt(vmag_sqr)| 1.9e-08 pu); the unit from its record (S3 duplicate identical).

| run | bus | entries (of 288) within 1e-6 pu of v_max | 1e-5 | 1e-4 | 1e-3 | min distance to v_max (pu) | within 1e-6 of v_min |
|---|---|---|---|---|---|---|---|
| x0_models | 5 | 6 | 59 | 63 | 77 | 3.315e-07 | 0 |
| x0_models | 7 | 0 | 2 | 5 | 10 | 3.940e-06 | 0 |
| x0_models | 9 | 37 | 130 | 139 | 147 | 3.598e-07 | 0 |
| x0_record | 5 | 6 | 59 | 63 | 77 | 3.315e-07 | 0 |
| x0_record | 7 | 0 | 2 | 5 | 10 | 3.940e-06 | 0 |
| x0_record | 9 | 37 | 130 | 139 | 147 | 3.598e-07 | 0 |
| unit_record | 5 | 6 | 59 | 63 | 77 | 3.306e-07 | 0 |
| unit_record | 7 | 0 | 2 | 5 | 10 | 3.796e-06 | 0 |
| unit_record | 9 | 35 | 132 | 138 | 145 | 3.980e-07 | 0 |

At the bound (1e-6 pu): x = 0 -- bus 5 6; bus 7 0; bus 9 37 ; unit -- bus 5 6; bus 7 0; bus 9 35. With the unit 0 entries enter and 2 leave the at-bound set. Bus-7 voltage change x = 0 -> unit: -5.75e-03 to +9.17e-03 pu (mean |.| 4.05e-04); bus-7 minimum distance to 1.1 pu: x = 0 3.94e-06, unit 3.80e-06 pu. Per-slot lists are in the JSON.

Multipliers at x = 0 on the 43 at-bound ADN entries: |dual(voltage_magnitude_upper_cons)| 18.6 to 61.3 (EUR per p.u.^2 of v^2 per representative-day hour); the complementarity product s * |z| is 3.62e-05 to 4.52e-05 (median 4.51e-05) -- a near-common value across rows, consistent with rows held at the interior-point barrier distance from the bound; their activity is resolved only to that level. Economically: the reactive marginal value at those entries is -1.2681 to -0.1715 EUR/MVArh (dual(node_balance_q) / baseMVA at the same bus and hour). Bus 7 (the storage bus): 5 entries within 1e-4 pu of 1.1, with |dual| 1.05 to 5.24 and s * |z| 2.61e-05 to 7.86e-05.

Other TSO buses (non-ADN) within 1e-6 pu of v_max at x = 0: 0 of 1728 entries (buses []).

Context (x = 0, all DSO buses, soft voltage rows): 1095 of 28512 DSO bus-hour entries within 1e-4 pu of v_max (DSO 5 360, DSO 7 301, DSO 9 434), 0 within 1e-4 pu of v_min; sum of DSO voltage slacks -5.429e-04 p.u.^2. In the DSO model the storage's Q enters only the reference-bus balance and the interface definition, where it cancels (W19 Z1 for P; the Q rows are the same form), so it reaches DN voltages only through the TSO bus-7 voltage, whose value to the system is priced by LMP_Q,7.

## Z34 (d) -- Addendum 34 conclusion

Q idle: False (max |q|/S 0.0861). Bus-7 1.1 pu bound binds: False (x = 0: 0 entries, unit: 0). The 1.1 pu bound is reached only at the OTHER two interfaces (bus 5: x = 0 6, unit 6; bus 9: x = 0 37, unit 35), where there is no storage in this instance. At bus 7 the reactive marginal value is -0.4396 to -0.0038 EUR/MVArh (vs LMP_P,7 mean 99.96 EUR/MWh), so the storage's Q -- dispatched, not idle -- is worth A_Q = 309.15 EUR over the horizon (1.2e-03 of the value, below the resolution bar 34,734.63). The DSOs' reactive flexibility cannot be displaced: it is structurally absent (bounds = EQUALITY_TOLERANCE) and unused at x = 0, so saving_Q <= ~0 and the saving is P flexibility. Reading (criteria above): the storage's voltage support is effectively unmonetized because no voltage constraint binds at its bus (bus 7 never within 1e-6 pu of 1.1; the reactive price there is at most 0.44 EUR/MVArh in magnitude), and the DN side has no reactive flexibility for it to displace -- network value exists only where a constraint binds.

## Checks

- A_P_at_x0_duals_reproduces_W25_A0_within_0.1_eur: True
- a0_x0_child_manifest_tracked: True
- binding_set_unit_288_slots: True
- binding_set_unit_equals_W19: True
- binding_set_x0_288_slots: True
- binding_set_x0_equals_W19: True
- dso5_saving_equals_W25_T3_within_0.01: True
- dso7_saving_equals_W25_T3_within_0.01: True
- dso9_saving_equals_W25_T3_within_0.01: True
- dso_blocks_have_flex_cost_expression_flex_vars_single_scenario: True
- dso_nodes_5_7_9: True
- ess_stride_q_is_tso_shared_es_qnet_x_baseMVA_load_convention: True
- esso_capture_288_rows_single_cohort: True
- esso_capture_pnet_equals_ess_stride_esso_p: True
- esso_capture_s_max_is_0p25: True
- exists::data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/component_levels_terminal.json: True
- exists::data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/evaluation_record.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/component_levels_terminal.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/ess_entry_stride_baseline.jsonl: True
- exists::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/evaluation_record.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/interface_voltage_terminal.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/component_levels_terminal.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/ess_entry_stride_baseline.jsonl: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/esso_capture/s39_D/node7_cycle112.jsonl: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/evaluation_record.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/interface_settlement_detail_s31c.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/interface_voltage_terminal.json: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/pf_entry_stride_s39_D.jsonl: True
- exists::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/certified_models.pkl: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/component_levels_terminal.json: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/ess_entry_stride_baseline.jsonl: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/evaluation_record.json: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/interface_settlement_detail_s31c.json: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/interface_voltage_terminal.json: True
- exists::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/pf_entry_stride_s39_D.jsonl: True
- flex_cost_P_and_Q_coefficients_equal_every_load_period: True
- flex_cost_coefficient_uniform_across_loads: True
- flex_cost_only_flex_p_down_and_flex_q_down: True
- flex_cost_per_slot_sum_equals_model_expression_and_total: True
- flex_cost_per_slot_sum_reproduces_record_component_level: True
- flex_cost_profile_identical_across_the_three_DSOs: True
- flex_values_not_below_relaxed_bound: True
- payload_has_tso_and_dso: True
- pf_stride_unit_last_cycle_is_terminal: True
- pf_stride_x0_last_cycle_is_terminal: True
- pickle_sha256_in_committed_campaign_manifest: True
- pickle_sha256_is_03b62593: True
- q_flex_bounds_are_2e-5_pu: True
- settled::data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/component_levels_terminal.json: True
- settled::data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0/evaluation_record.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/component_levels_terminal.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/ess_entry_stride_baseline.jsonl: True
- settled::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/evaluation_record.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1/interface_voltage_terminal.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/component_levels_terminal.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/ess_entry_stride_baseline.jsonl: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/esso_capture/s39_D/node7_cycle112.jsonl: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/evaluation_record.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/interface_settlement_detail_s31c.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/interface_voltage_terminal.json: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/pf_entry_stride_s39_D.jsonl: True
- settled::data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1/results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/certified_models.pkl: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/component_levels_terminal.json: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/ess_entry_stride_baseline.jsonl: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/evaluation_record.json: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/interface_settlement_detail_s31c.json: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/interface_voltage_terminal.json: True
- settled::data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/pf_entry_stride_s39_D.jsonl: True
- spec_sha256_prefix_f8adcc97: True
- tracked::data/SRP1/MarketData/SRP1_market_data.xlsx: True
- tracked::data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json: True
- tracked::data/SRP1/Results/P515S47/tso_marginal_cost/tso_marginal_cost.json: True
- tracked::data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json: True
- tracked::data/SRP1/SRP1.json: True
- tracked::data/SRP1/case9/case9_2025.json: True
- tracked::data/SRP1/case9/case9_2030.json: True
- tracked::data/SRP1/case9/case9_2035.json: True
- tso_blocks_have_duals_balances_voltage_rows: True
- tso_identity_P_on_every_persisted_x0_block: True
- tso_identity_Q_on_every_persisted_x0_block: True
- tso_interface_delta_interior_x0_models: True
- tso_sigma_and_w_equal_record_weights: True
- unit_candidate_key: True
- unit_certified: True
- unit_child_manifest_tracked: True
- unit_duplicate_S3_flex_levels_identical: True
- unit_duplicate_S3_terminal_q_identical: True
- unit_duplicate_S3_voltage_entries_identical: True
- unit_duplicate_candidate_key: True
- unit_duplicate_certified: True
- unit_duplicate_child_manifest_tracked: True
- unit_duplicate_same_cost_and_cycles: True
- unit_interface_delta_interior_every_block_period: True
- unit_is_node7_0p25_1p0_2025_nodes_5_9_empty: True
- value_from_component_levels_equals_record_difference: True
- value_is_259427.77_2dp: True
- x0_all_nodes_empty: True
- x0_campaign_manifest_tracked: True
- x0_candidate_key: True
- x0_certified: True
- x0_child_manifest_tracked: True
- x0_model_voltages_equal_x0_record_interface_voltage_terminal: True
- x0_s48_flex_levels_equal_a0_x0_record_W25: True
