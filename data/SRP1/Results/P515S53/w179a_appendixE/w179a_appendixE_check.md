# W179a — Appendix E against the frozen tables (Overleaf 48099ab)

main.tex sha256 `28008ab0d79b477c4e8d86a0c3a4d1acf9c95ed9c4921f4757f1600cda1c60af`; frozen JSON `590088fe`; script `p515_s53_w179a_appendixE_check.py`. Zero solves; guards 0/0 verified.

## Table entries

| table | rows | entries | MATCH | MISMATCH | not a frozen field, agrees | not a frozen field, disagrees |
|---|---|---|---|---|---|---|
| E.20 | 58 | 639 | 581 | 0 | 58 | 0 |
| E.21 | 61 | 854 | 732 | 0 | 122 | 0 |
| E.22 | 8 | 18 | 18 | 0 | 0 | 0 |
| E.23 | 7 | 49 | 49 | 0 | 0 | 0 |
| E.24 | 9 | 63 | 63 | 0 | 0 | 0 |

### Every MISMATCH

- none

## Prose

| id | section | written | status | value | source |
|---|---|---|---|---|---|
| I1 | intro (l. 2388) | tau = 4,539.07 EUR | MATCH | 4539.0682750000005 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json constant… |
| I2 | intro (l. 2388) | amounts rounded to 0.1 k EUR, ratios to two decimals | MISMATCH | {'k_EUR_dp': [1], 'ratio_dp_E20_E22': [2], 'n_k': 284, 'n_ratio': 171, 'E24_resid_ratio_dp': [3], 'n_E24_resi… | decimal places of every printed k EUR amount and every ratio column of E.20-E.24 (parsed … |
| E1.1 | E.1 (l. 2395) | 58 distinct comparisons | MATCH | 58 | E.20 rows parsed; each mapped to a distinct frozen claim |
| E1.2 | E.1 (l. 2395) | Fifty-five reported both gross and net | MATCH | 55 | E.20 rows with a T1 claim (net printed) |
| E1.3 | E.1 (l. 2395) | the remaining three (m = 1.75 unit; two at the 0.50 floor) gross only | MATCH | {'22': 'H:m1.75:value_minus_I', '52': 'E:soh050:value_minus_I', '58': 'E:soh050:delta_value_vs_070'} | tables.a64.rows (T10); no d_net field |
| E1.4 | E.1 (l. 2395) | 1.75 times the base flexibility price; the 0.50 end-of-life floor | MATCH | {'h_unit_m175 m': True, 'e_soh050 floor 0.5': True} | data/SRP1/Results/P515S53/w155_a64_cells/frozen_s53_a64_cells_spec_v1_44a2dce8.json cells |
| E1.5 | E.1 (l. 2395) | determinate if \|Delta\| >= max{3b, 2 tau}, b the larger band | DERIVED | {'certified_claims': 41, 'threshold_reproduced': 41, 'not_reproduced': [], 'verdict_inconsistent': []} | max(3 x max(band_width ref, other), 2 TAU) recomputed for every certified-certified claim |
| E1.6 | E.1 (l. 2395) | uncertified: \|Delta\| > 3 x the largest consensus gap and settling slack involved, gross… | DERIVED | {'uncertified_claims': 23, 'bar_reproduced': 23, 'not_reproduced': []} | 3 x max(\|gap\|, \|slack\|) over the uncertified cells involved recomputed for every unce… |
| E1.7 | E.1 (l. 2395) | two certificates on a non-clean turning point: evaluations 4 and 38 | MATCH | [4, 38] | cells with flag_non_clean_turning_point true, numbered by the E.21 map |
| E1.8 | E.1 (l. 2395) | none changes verdict (uncertified form of the comparisons using 4 and 38) | MATCH | {'B:n7_4h_e3': ['determinate', 'determinate'], 'J:e3_to_e4': ['determinate', 'determinate'], 'J:e4_to_e5': ['… | claims.uncertified_form_beside_A64_ruling4 verdict vs gross_verdict |
| N1 | E.20 note (l. 2481) | row 3 bar 70.7 k EUR (2.94x), determinate | MATCH | {'bar_k': 70.7, 'x': 2.94, 'verdict': 'determinate', 'row_maps_to_claim': True} | claims[B:n7_4h_e3].uncertified_form_beside_A64_ruling4 |
| N2 | E.20 note (l. 2481) | row 26 bar 15.9 k EUR (11.12x), determinate | MATCH | {'bar_k': 15.9, 'x': 11.12, 'verdict': 'determinate', 'row_maps_to_claim': True} | claims[J:e4_value_minus_I].uncertified_form_beside_A64_ruling4 |
| N3 | E.20 note (l. 2481) | rows 30 and 31 already use the uncertified form | MATCH | {'30': 'uncertified form', '31': 'uncertified form'} | claims gross_rule of J:e3_to_e4, J:e4_to_e5 |
| N4 | E.20 note (l. 2481) | evaluation 56 reproduces evaluation 2 bitwise (row 48 repeats row 1 with the sign of valu… | MATCH | {'row48_d_gross': -64417.38662493462, 'row1_d_gross': 64417.38662493229, 'twin_evidence': {'field': 'gross_op… | at_or_above_0_95_tau.twins.evidence.e_c2_calfade (reproduced through 172) and claims d_gr… |
| N5 | E.20 note (l. 2481) | row 58 against evaluation 61 (bitwise through 172) gives the same figures | MATCH | {'equal_at_print': {'d_gross': True, 'threshold_at_print': True, 'multiple_at_print': True, 'verdict': True},… | tables.a64.rows E:soh050:delta_value_vs_g070_REPORT_BESIDE vs E:soh050:delta_value_vs_070… |
| N6 | E.20 note (l. 2481) | ^n: energy capacity at node 7 differs by two lattice steps (row 45) | DERIVED | 2.0 | \|3.5 - 2.5\| MWh / 0.5 MWh step (ref:5ca4f86c vs l_df1a5525 candidate_canonical) |
| N7 | E.20 note (l. 2481) | rows 17-20 each recorded once gross and once net | MATCH | {'17': ['G:n7_2h_e1_y2030:gross', 'G:n7_2h_e1_y2030:net'], '18': ['G:n7_2h_e1_y2035:gross', 'G:n7_2h_e1_y2035… | claims G:*:gross / G:*:net pairs |
| E2.1 | E.2 (l. 2488) | 61 recourse evaluations | MATCH | [61, 61] | E.21 rows parsed and mapped one-to-one onto the 49 + 12 frozen cells |
| E2.2 | E.21 note (l. 2580) | evaluations 1, 2, 40 and 41 continue earlier runs; first residual passes not tabulated | MATCH | {'1': None, '2': None, '40': None, '41': None} | cells ref:* k0_run None |
| E2.3 | E.21 note (l. 2580) | for evaluation 1, k0 = 123 (Fig. res_q_cycle) | MATCH | 123 | data/SRP1/Results/P515S53/w168_x0_v6_replay/w168_x0_v6_replay_decision.json decision_prim… |
| E2.4 | E.21 note (l. 2580) | evaluation 16 superseded: rested on a cycle whose local solve did not end optimal; re-run… | MATCH | {'superseded': True, 'v6_on_record': 'not decided within the recorded cycles 1..167', 'v6_vetoes': 1, 'rerun_… | cells_appended_w160.pb_y2025_n5 superseded; T5 note; W142 item 1 (v6 not decided, 1 veto) |
| E3.1 | E.3 (l. 2588) | all fourteen admissible unit neighbours evaluated by the search | MATCH | {'b_box': {'n_feasible': 14, 'n_evaluated': 14, 'n_not_evaluated': 0, 'box_equals_final_poll_set': True}, 'ce… | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.x0 (b_box, a_f… |
| E3.2 | E.3 (l. 2588) | eight of them re-evaluated under the certification rule | MATCH | [8, 8] | E.22 rows; W173 item2.x0.c_resettled_in_frozen_tables |
| E3.3 | E.3 (l. 2588) | the smallest unit 0.25 MVA / 0.5 MWh, each node in 2025 or 2030, and two pairs of nodes | MATCH | {'15': '5: 0.25/0.5 2025', '17': '7: 0.25/0.5 2025', '18': '9: 0.25/0.5 2025', '19': '5: 0.25/0.5 2030', '20'… | E.22 cells candidate_canonical |
| E3.4 | E.3 (l. 2588) | every one costs determinately more | MATCH | {'15': ['determinate', 'determinate'], '17': ['determinate'], '18': ['determinate'], '19': ['determinate'], '… | phase_b v6_verdict / C claims gross_verdict |
| E3.5 | E.3 (l. 2588) | at its resolution sigma_Q | DERIVED | {'search_resolution_eur_distinct': [18449.66, 22128.84, 22145.02, 40188.72, 43788.42, 44086.65, 47205.25], 'm… | W173 item2.x0.rows search_resolution_eur (not a frozen field): the search accepted at max… |
| E4.1 | E.4 (l. 2623) | each static arrangement solved from three starts | MATCH | [3, 3] | benchmark.w160_additions.arms_in_full.*.n_starts |
| E4.2 | E.4 (l. 2623) | the best is reported in Table res_benchmark; the spread over the starts is its band | MATCH | {'passive': [25804.990407824516, 25804.990407824516], 'price_taker': [0.004241466522216797, 0.004241466522216… | arms.*.multimodality_band == max - min of Q_by_start |
| E5.1 | E.5 (l. 2661) | refused at m = 2 and evaluation 42: gap -8.8 to -9.4 k EUR | MATCH | {'39': -8.8, '40': -9.2, '41': -9.3, '42': -9.3, '45': -9.4, '50': -9.2, '51': -9.2} | dead_zone t_sum_at_cap of evals 39, 40, 41, 42, 45, 50, 51 |
| E5.2 | E.5 (l. 2661) | split about 55 %, 14 %, 31 % among nodes 5, 7, 9 | MATCH | {'39': [55.2, 13.7, 31.0], '42': [55.3, 13.7, 31.1], '45': [55.3, 13.7, 31.0], '50': [55.3, 13.7, 31.0], '51'… | dead_zone share_by_node of the rows that record it (39, 42, 45, 50, 51); evaluations 40 a… |
| E5.3 | E.5 (l. 2661) | power-flow primal residual 0.68-0.70 of its tolerance, last cycle | MATCH | {'39': 0.682, '42': 0.692, '45': 0.7, '50': 0.69, '51': 0.69} | dead_zone pf_primal_last of the rows that record it (39, 42, 45, 50, 51); 40 and 41 not r… |
| E5.4 | E.5 (l. 2661) | evaluation 42 reset by two transmission solver recoveries before the gap clause | MATCH | {'cause': 'lapse reset', 'lapse_cycles': [218, 222], 'entry': 'by its signature, cause stated: two TSO recove… | dead_zone l_0ee93aca cause / lapse_events / entry |
| E5.5 | E.5 (l. 2661) | reference unit at 1.5x: gap -2.5 k EUR, same node split, residual ratio 0.027 | MATCH | {'eval': 31, 'gap_k': -2.5, 'share': [54.6, 14.5, 30.9], 'pf_primal_last': 0.027, 'm': 1.5} | dead_zone h_f9eae48f |
| E5.6 | E.5 (l. 2661) | 1.25 MVA / 5 MWh at base price: growth test after repeated TSO recoveries, small gap of o… | MATCH | {'eval': 6, 'cause': 'growth test after five TSO recoveries (not the dual dead zone; Addendum 65)', 'gap_k': … | dead_zone d_36686489; cells.d_36686489 non_clean_after_N |
| E5.7 | E.24 caption (l. 2666) | refused evaluations and two others | MATCH | {'gap_refused_rows': 7, 'others': [42, 6]} | E.24 rows: gap_refused true vs others |
| E6.1 | E.6 (l. 2692) | 42 = evaluations 3-11, 15, 22, 23, 26-31, 34-39, 42-59 | MATCH | {'n_listed': 42, 'equals_W153_42': True, 'arith': '49 T2 cells (tables.cells) - 3 B cells - 4 references = 42… | E.21 map vs data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json … |
| E6.2 | E.6 (l. 2692) | 32 certified, 24 oscillatory, 8 monotone | MATCH | [32, 24, 8] | frozen cells status / branch over the 42 (= W153 totals) |
| E6.3 | E.6 (l. 2692) | 10 uncertified: 5 gap clause, 2 lapse reset, 3 growth test | MATCH | [10, 5, 2, 3] | frozen cells cause_uncertified over the 42 |
| E6.4 | E.6 (l. 2692) | median certification cycle 174 | DERIVED | 174.0 | median of k_star over the 32 certified of the 42 (frozen cells) |
| E6.5 | E.6 (l. 2692) | k* - k0 median 65 (range 21-89) | DERIVED | [65.0, 21, 89] | k_star - k0_run over the 32 (frozen cells); W153 cycles_after_k0_k_star_minus_N beside |
| E6.6 | E.6 (l. 2692) | median range / tau at certification 0.86 | DERIVED | 0.8602808955777403 | median of range_over_tau over the 32 |
| E6.7 | E.6 (l. 2692) | four evaluations added later (32, 33, 60, 61) all certified | MATCH | {'32': ['h_x0_m175', 'certified'], '33': ['h_unit_m175', 'certified'], '60': ['e_soh050', 'certified'], '61':… | cells_appended_w160 A64 cells |
| E6.8 | E.6 (l. 2692) | Ten of the 46 certificates the tables use | DERIVED | {'registry': 49, 'certified_cells_in_T2': 49, 'superseded': 1, 'twins': 2, 'arith': '49 - 1 - 2 = 46', 'n_cou… | at_or_above_0_95_tau: registry 49 certified - 1 superseded - 2 twins; n_counted 10 |
| E6.9 | E.6 (l. 2692) | stop within 5 % of tau: evaluations 2, 3, 5, 10, 11, 12, 14, 17, 20, 23 | MATCH | [2, 3, 5, 10, 11, 12, 14, 17, 20, 23] | at_or_above_0_95_tau.counted_cells numbered by the E.21 map |
| E6.10 | E.6 (l. 2692) | evaluation 36 at 0.94 on the monotone branch, just below | MATCH | {'eval': 36, 'branch': 'monotone', 'range_over_tau': 0.93890671105718, 'beside': 'monotone, 0.939 (below 0.95… | at_or_above_0_95_tau.beside i_5a6a88b4; cells.i_5a6a88b4 |
| E6.11 | E.6 (l. 2692) | the two evaluations continued past their certificates, 14 and 3, moved 0.74 tau and 0.78-… | MISMATCH | {'W142_item3': {'b_4649234b (cell 3)': 0.738, 'd_c52e1670 at its v6 certificate': 0.781, "d_c52e1670 at the l… | data/SRP1/Results/P515S53/w142_resettle_v6/v6_from_records/w142_v6_from_records.json item… |
| E6.12 | E.6 (l. 2692) | 21 cycles in 10 evaluations not clean after the first residual pass, all in transmission … | MATCH | {'cycles': 21, 'evaluations': 10, 'by_family_W153': {'TSO': 21, 'DSO': 0, 'ESSO': 0}, 'evals': [3, 4, 6, 8, 9… | frozen non_clean_after_N over the 42; W153 totals by family |
| E6.13 | E.6 (l. 2692) | 119 exits at the acceptable level within 10x, all in distribution blocks, counted clean | MATCH | {'after_N_total': 119, 'by_family': {'TSO': 0, 'DSO': 119, 'ESSO': 0}, 'cells': {'d_f759dd48': 114, 'c_156ce2… | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json totals (not … |
| R1 | E.6 reproducibility (l. 2699) | multi-scenario x = 0 replayed its 72 recorded cycles bitwise | MATCH | {'w163_status': 'match', 'w163_value': 72, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R2 | E.6 reproducibility (l. 2699) | the three single-scenario reference continuations replayed bitwise to their certification… | MISMATCH | {'w163_status': 'match', 'w163_value': {'c_star': {'G19': True, 'N': 87, 'through': 87}, 'n7_4h_e1': {'G19': … | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R3 | E.6 reproducibility (l. 2699) | Apple M2 Max | MATCH | {'w163_status': 'match', 'w163_value': 'Apple M2 Max', 'note': 'read at this run; the campaign records carry … | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R4 | E.6 reproducibility (l. 2699) | 32 GiB of memory | MATCH | {'w163_status': 'match', 'w163_value': {'now': 34359738368, 'v6_spec': 34359738368}, 'note': '34,359,738,368 … | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R5 | E.6 reproducibility (l. 2699) | macOS 27.0 | MATCH | {'w163_status': 'match', 'w163_value': {'now': '27.0', 'w159': '27.0'}, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R6 | E.6 reproducibility (l. 2699) | thread caps at one thread | MATCH | {'w163_status': 'match', 'w163_value': True, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R7 | E.6 reproducibility (l. 2699) | HSL built without OpenMP; MA97 serial | MATCH | {'w163_status': 'match', 'w163_value': True, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R8 | E.6 reproducibility (l. 2699) | two distribution blocks (node 7 2025 Winter, node 5 2035 Winter) end some primary solves … | MATCH | {'w163_status': 'match', 'w163_value': {'DSO\|5\|2035\|Winter': 109, 'DSO\|7\|2025\|Winter': 10}, 'note': 'sc… | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R9 | E.6 reproducibility (l. 2699) | within ten times the tight-tail tolerances | MATCH | {'w163_status': 'match', 'w163_value': 10.0, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R10 | E.6 reproducibility (l. 2699) | tail lowered each of the three references by 1.0-1.2 x 10^-6 relative, same sign | MATCH | {'w163_status': 'approximate', 'w163_value': {'c_star': -1.1998393015853015e-06, 'n7_4h_e1': -1.0361353232586… | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R11 | E.6 reproducibility (l. 2699) | complementarity tolerance 10^-6 from 10^-4 (DN) and 5 x 10^-4 (TN) | MATCH | {'w163_status': 'match', 'w163_value': {'dso_case_files_set_it': False, 'ipopt_print_options_default': 0.0001… | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R12 | E.6 reproducibility (l. 2699) | 10^-5 p.u. slack on the renewable availability bound | MATCH | {'w163_status': 'match', 'w163_value': 1e-05, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R13 | E.6 reproducibility (l. 2699) | lowers the objective by about 7.8 k EUR at first order | MATCH | {'w163_status': 'match', 'w163_value': -7848.084606834722, 'note': '-7,848.08 € = 7.8481 k€'} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R14 | E.6 reproducibility (l. 2699) | 1.2 x 10^-5 of it | MATCH | {'w163_status': 'match', 'w163_value': 1.2002447231900346e-05, 'note': None} | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json c… |
| R15 | E.6 reproducibility (l. 2699) | three single-scenario reference continuations (count) | MATCH | {'W163_N49': {'c_star': {'G19': True, 'N': 87, 'through': 87}, 'n7_4h_e1': {'G19': True, 'N': 112, 'through':… | W163 N49 / Addendum 53 report: x0 (eval 1), unit (eval 2) and C* (not in Appendix E) |
| S1 | sec. 4.6 (task list) (l. 1681) | 55 comparisons reported both gross and net | MATCH | 55 | E.20 rows mapped to T1 claims (net form present) |
| S2 | sec. 4.6 (task list) (l. 1681) | the salvage credit changes no sign | DERIVED | [] | sign(d_gross) vs sign(d_net) over the 55 (net from the G :net claim for rows 17-20) |
| S3 | sec. 4.6 (task list) (l. 1681) | one verdict change: row 34, within the bar gross (0.95x), determinate net (1.22x), in the… | DERIVED | [{'row': 34, 'claim': 'L:y2030__n5_p0.25_e0.5__n7_p1_e3', 'gross': [0.95, 'within the uncertified bar'], 'net… | gross_verdict vs net_verdict_W153 over the 55; d_net > 0 = the returned plan costs less (… |

## Interpretations (i)-(ix)

| item | outcome | evidence |
|---|---|---|
| (i) 60 -> 55 + 3 mapping | confirmed | {'n_T1': 60, 'n_T1_rows': 55, 'unused_T1': ['CHECK:headline_V_minus_I_settled', 'G:n7_2h_e1_y2030:net', 'G:n7_2h_e1_y2035:net', 'G:n7_4h_e2_y2030:net', 'G:n7_4h_e2_y2035:net'], 'net_dups': ['G:n7_2h_e1_y2030:net', 'G:n7_2h_e1_y2035:net', 'G:n7_4h_e2_y2030:net', 'G:n7_4h_e2_y2035:net'], 'n_T10': 4, 'n_T10_rows': 3, 'unused_T10': ['E:soh050:delta_value_vs_g070_REPORT_BESIDE'], 'CHECK_equals_row1_wi… |
| (ii) plan behind every cell of E.21 | confirmed | {'all_61_rows_plan_year_match_candidate_canonical': True, 'named_cells': {'yl_y2030': ['7: 0.25/1', 2030], 'yl_y2035': ['7: 0.25/1', 2035], 'pb_*': {'pb_y2025_n5_v6': ['5: 0.25/0.5', 2025], 'pb_y2025_n5': ['5: 0.25/0.5', 2025], 'pb_y2025_n7': ['7: 0.25/0.5', 2025], 'pb_y2025_n9': ['9: 0.25/0.5', 2025], 'pb_y2030_n5': ['5: 0.25/0.5', 2030], 'pb_y2030_n7': ['7: 0.25/0.5', 2030], 'pb_y2030_n9': ['9:… |
| (iii) Rule column | confirmed, with one qualification (row 16) | {'rule_column_matches_certifying_spec': True, 'record_rows_v6_on_record_W142_item1': {'1': {'cell': 'ref:7aa017f0', 'w142_record': 'x0', 'v6_status': 'certified', 'v6_k_star': 181, 'T2_k_star': 181, 'T2_end': 181, 'v6_k_cap': None}, '2': {'cell': 'ref:bd504ecf', 'w142_record': 'unit_n7_4h_e1', 'v6_status': 'certified', 'v6_k_star': 172, 'T2_k_star': 172, 'T2_end': 172, 'v6_k_cap': None}, '3': {'c… |
| (iv) the six other box neighbours evaluated by the search only | confirmed (scoped) | {'six_labels_W173': ['y2025__n7_p0.25_e0.5__n9_p0.25_e0.5', 'y2030__n7_p0.25_e0.5__n9_p0.25_e0.5', 'y2030__n5_p0.25_e0.5__n9_p0.25_e0.5', 'y2025__n5_p0.25_e0.5__n7_p0.25_e0.5', 'y2025__n5_p0.25_e0.5__n7_p0.25_e0.5__n9_p0.25_e0.5', 'y2030__n5_p0.25_e0.5__n7_p0.25_e0.5__n9_p0.25_e0.5'], 'equal_to_the_six_named': True, 'searched': {'y2025__n7_p0.25_e0.5__n9_p0.25_e0.5': {'in_frozen_cells': False, 'i… |
| (v) T6: unit of NRF violations, consistency pass, warm | corrected (the reading of the consistency pass); the unit and "warm" confirmed | {'unit': 'one entry per violated row uncoord_no_reverse_flow[s_m, s_o, p] of a DSO block, i.e. per DN interface and hour of a year-season block (the single-scenario instance has one market and one operation scenario: \'every DSO arm block (3 DNs x 3 years x 4 days), every market / operation scenario, every period; 24 rows per block at SRP1; declared per block in the build record and counted by th… |
| (vi) 119 exits at the acceptable level, all in distribution blocks | confirmed | {'w153': {'after_N_total': 119, 'by_family': {'TSO': 0, 'DSO': 119, 'ESSO': 0}, 'cells': {'d_f759dd48': 114, 'c_156ce2d1': 5}, 'definition': 'cycle records acceptable_clean_blocks (a primary Acceptable within 10x: clean under v5/v6)'}, 'unit': 'an event is one (cycle, block) whose final IPOPT exit was Acceptable on the primary attempt within 10x (reason acceptable_primary_within_factor), summed o… |
| (vii) continued cells and movements | corrected (the set): the movements of 14 (0.74 tau) and 3 (0.78-0.88 tau) are confirmed, but a third evaluation, 12, was also continued past its certificate (0.004 tau); "the two evaluations" should read three, and the sec. 4.7 bound "at most 0.9 tau" still holds | {'W142_item3': {'b_4649234b (cell 3)': 0.738, 'd_c52e1670 at its v6 certificate': 0.781, "d_c52e1670 at the last-pair reading's 148 (W141 V1)": 0.882}, 'W163_N25_over_tau': {'b_2a0ba8b2 (v3 run, Q(213) - Q(173); v4 k* 173)': 0.0039, 'b_4649234b (v4 run, Q(193) - Q(148); v5 k* 148)': 0.7383, 'd_c52e1670 (v5 run, Q(198) - Q(148); last-pair reading k* 148)': 0.8821, 'd_c52e1670 (v5 run, Q(198) - Q(1… |
| (viii) the three single-scenario reference continuations | corrected (wording): the three are evaluations 1 and 2 and C*, which is not in Appendix E; their "—" in the Replay column agrees with T2, but they replayed to the cycles at which the residual-based rule had certified them (132, 112, 87), not to the certification cycles k* that Appendix E prints (181, 172) | {'cells': 'x0 (evaluation 1), the unit n7_4h_e1 (evaluation 2) and C* (not in Appendix E; new certification none, cap 187, per the Addendum 53 report)', 'replayed_through': {'c_star': 87, 'n7_4h_e1': 112, 'x0': 132}, 'T2_replay_field': {'1': None, '2': None}, 'meaning': 'they replayed bitwise to the cycles at which the residual-based rule had certified them (x0 132, unit 112, C* 87; "old certific… |
| (ix) k0 = 123 for x = 0 | confirmed | {'W168_decision_primary_N_k0_kstar': [123, 123, 181], 'T2_k0_run': None, 'figure_caption_line': 1700, 'derived': "k* - k0 = 181 - 123 = 58 (sec. 4.7 l. 1694: '58 cycles after')"} |

## Predictions

- (a) 0 MISMATCH in table entries: **HELD** — 0 MISMATCH over 1623 entries (1443 MATCH on frozen fields; 180 not-a-frozen-field entries agree with their named source, 0 disagree); mapping problems 0
- at most two of (i)-(ix) corrected, most likely (v) and (vi): **FAILED** — 3 corrected: ['(v) T6: unit of NRF violations, consistency pass, warm', '(vii) continued cells and movements', '(viii) the three single-scenario reference continuations']

## Integrity checks

- main_tex_lines: True
- e21_61_rows_numbered_1_61: True
- e21_map_one_to_one_onto_frozen_cells: True
- e21_mapping_problems_none: True
- e20_mapping_problems_none: True
- e20_58_rows_numbered_1_58: True
- every_needle_found: True
