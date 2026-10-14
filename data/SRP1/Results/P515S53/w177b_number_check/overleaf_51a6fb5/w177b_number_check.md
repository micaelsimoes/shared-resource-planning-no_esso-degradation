# W177b-1 -- Parts 0 and 1 at Overleaf 51a6fb5 (main.tex sha256 61fe630c)

Zero solves; script `p515_s53_w177b_number_check.py`; frozen tables `data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json` (sha256 590088fe); clone `manuscript/6a67305f25e8348fb71380c3` (read only). Objective convention: Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR

## Predictions (expert, Addendum 76)

- **Part 1: zero MISMATCH in numbers carried from the tables** -- FAILED as worded: 1 MISMATCH among 730 in-scope tokens (0 in table bodies / captions / notes; 0 against a frozen-table field, the rest against a named record) -- l. 1488 'two' -> 1 (data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d))
- **Part 1: the derived numbers hold** -- HELD: 16/16 derived tokens hold at the written precision; named: 24 {'line': 1227, 'holds': True}, 33 {'line': 1227, 'holds': True}, 7 [{'line': 1411, 'holds': True}, {'line': 1553, 'holds': True}], 9 [{'line': 1411, 'holds': True}, {'line': 1553, 'holds': True}], 90.9 {'line': 1553, 'holds': True}, 654 {'line': 1557, 'holds': True}, 9.3 {'line': 1320, 'holds': True}, 42.1 {'line': 1320, 'holds': True}, 28.5 {'line': 1320, 'holds': True}

## Part 0 -- W177a findings at this HEAD

| id | W177a finding | status | where | evidence |
|---|---|---|---|---|
| Blocked 1 / C1 | eq:salvage_credit needs two corrections (energy cost only; health factor (SoH - SoH^Min)/(1 - SoH^Min)); items (a)-(d) hold | still present | l. 1151-1164 | {"equation_line": 1157, "cohort_cost_with_power_term": [{"line": 1165, "text": "where $I_{e,y^\\text{Inv}} = \\sum_{c \\in \\Omega_C} \\omega_c \\big( c^{\\text{Inv,S}}_{y^\\text{Inv},c} S^{\\text{Inv}}_{e,y^\\text{Inv}} + c^{\\text{Inv,E}}_{y^\\text{Inv},c} E^{\\text{Inv}}_{e,y^\\text{Inv}} \\big)$ is the expected undiscounted cost of the cohort installed at node $e$ in year $y^\\text{Inv}$, "}], "factor_I_over_discount": [{"line": 1154, "text": "\\frac{I_{e,y^\\text{Inv}}}{(1+r)^{\\,y^\\text{End}-y_0}}"}], "factor_SoH_unnormalized": [{"line": 1156, "text": "\\; SoH_{e,y^\\text{Inv},y^\\text{End}},"}], "block_equals_frame_lines": "main.tex l. 1150-1168 = frame l. 27-45: True", "recomputed_C |
| Blocked 2 / C4 | the F2 plan misdescribed: node 7 is 1 MVA / 3.5 MWh, not 3 MWh (frame l. 131) | no longer applicable | l. 1284 | {"plan_sizes_printed_in_4.2": [], "F2_sentence": [{"line": 1284, "quote": "1.0--51.8~k\\euro{}. The planning search under this price assumption returned a two-node investment plan installed in 2030. Its operational evaluation remained uncertified because of the consensus gap, and only seventeen of its sixty-one adm"}], "note": "the pasted text names the plan as \"a two-node investment plan installed in 2030\" without sizes; the two nodes and 2030 match the frozen candidate of ref:5ca4f86c (Part 1, l. 1284)", "recomputed_C4": {"outcome": "differs", "summary": "recorded: 2030, node 5 0.25 MVA / 0.5 MWh, node 7 1 MVA / 3.5 MWh; W173 label y2030__n5_p0.25_e0.5__n7_p1_e3.5 (z [1, 1, 4, 7, 0, 0, 1 |
| Blocked 3 | Addendum 64 ruling 4 (J 4 MWh at its uncertified bar) vs tab:res_energy_ladder_m2: the 4 MWh row prints the flagged certified threshold 9.7 / 18.14 x; the uncertified bar 15.9 / 11.12 x is beside it | still present | l. 1300, note l. 1309 | {"row": [{"line": 1300, "text": "4$^{f}$   & 176.6 & 9.7  & 18.14 & 42.8 & 27.7 & 1.55 \\\\"}], "note": [{"line": 1309, "quote": "\\textit{Note:} Treating the flagged evaluation as uncertified gives a comparison bar of 15.9~k\\euro{} ($11.12\\times$) for its investment margin. The increments to 4 and 5~MWh have bars of 27.7 and 26.5~k\\euro{} ($1.55\\times$ and $1.95\\"}], "main_columns_certified_threshold": true, "uncertified_bar_in_note": true, "remark": "the uncertified bar moved from the caption (frame) to the table note; no verdict depends on the choice"} |
| Blocked 4 | two of three Addendum 73 predictions failed (scoring record) | no longer applicable |  | {"note": "a scoring record, not manuscript text; scored in Addendum 74/76 by the Planner"} |
| Blocked 5 | literal "Section~4.x" references (l. 377, 415, 445, 631, 635, 671, 1066, 1119 at 8b76423) | fixed |  | {"literal_section_4_references": [], "refs_to_section_4_labels_outside_section_4": [{"line": 359, "refs": ["sec:results"]}, {"line": 426, "refs": ["sec:results"]}, {"line": 495, "refs": ["subsec:res_baseline", "subsec:res_breakeven"]}, {"line": 674, "refs": ["subsec:res_certification"]}, {"line": 678, "refs": ["subsec:res_certification"]}, {"line": 714, "refs": ["subsec:res_degradation"]}, {"line": 1097, "refs": ["subsec:res_degradation"]}, {"line": 1136, "refs": ["subsec:res_coordination"]}, {"line": 2187, "refs": ["sec:results"]}]} |
| Blocked 6 (l. 58) | "every evaluated plan ... 4.39 x to 45.94 x" spans the B rows only (table range 3.94-45.94 x) | no longer applicable |  | {"range_in_4.1_prose": [], "note": "no multiple range is stated in the 4.1 prose; the table prints every row"} |
| Blocked 6 (l. 63) | "Two-node plans lose more than their parts would" is not resolved by the records | fixed | l. 1219 | {"old_claim_found": [], "new_text": [{"line": 1219, "quote": "ed investment cost. The evaluated two-node plans likewise remain uneconomic: the cost excess is 102.1~k\\euro{} for nodes~5 and~9 in 2025 and 93.9~k\\euro{} for nodes~5 and~7 in 2030. Applying the terminal salvage credit reduces the latter to"}]} |
| Blocked 6 (l. 65) | "identical d net, since all cohorts reach y^End with the same remaining life" (the reason is that the 2025 cohort's remaining life is zero) | no longer applicable |  | {"old_claim_found": [], "note": "not written; tab:res_year_ladder prints the 2025 salvage as 0.0"} |
| Blocked 6 (l. 381) | "Deferring the reference unit changes little gross and nothing net" contradicts the figures | fixed | l. 1443 | {"old_claim_found": [], "new_text": [{"line": 1443, "quote": "Deferring the reference unit does not make it pay (Table~\\ref{tab:res_year_ladder}). Against the plan without storage, the unit installed in 2030 increases the total cost by 68.5~k\\euro{} and the unit installed in 2035 by 111.0~k\\euro{}. A later installation shortens the part of the horizon over which the unit operates, while the terminal salvage credit returns its remaining calendar life: 11.8~k\\"}], "note": "the new text makes no 2025-vs-2030/2035 net resolution claim; \"net of salvage, the unit costs 54.5--64.4 kEUR more than the plan without storage in every investment year\" is checked in Part 1"} |
| Blocked 6 (l. 484) | 4.9 (i) "with the datasheet calibration and the 0.70 floor" ambiguous (baseline vs C4 row) | no longer applicable |  | {"found_now": [], "main_tex_lines_removed_6ff8f3c_to_51a6fb5": 19, "removed_key_insights_heading": true, "removed_datasheet_clause": true, "note": "section 4.9 Key Insights was deleted at 51a6fb5 (git show 6ff8f3c:main.tex in the clone, read only)"} |
| Blocked 7 / C5 | optional fill of the 0.50-floor row: value 258.5 kEUR, AE 0.778, EFC/day 1.08, floor never (W155, not in the frozen T1-T11) | still open (row not filled; author's option) | l. 1346 | {"row": [{"line": 1346, "text": "Baseline, 0.50 floor & --- & $-$59.5 & 12.6 & 4.71 & never & --- & --- \\\\"}], "cells": ["Baseline, 0.50 floor", "---", "$-$59.5", "12.6", "4.71", "never", "---", "---"], "W177a_values": {"Value (kEUR)": "258.5", "AE": "0.778", "EFC/day": "1.08", "Floor binds": "never"}, "note": "value, AE and EFC/day print \"---\" (\"not reported\", per the note); value - I, threshold, multiple and \"never\" are checked in Part 1"} |
| C2 | S / E / duration of each B and C cell confirmed (16/16) | holds | l. 1189-1204 | {"plan_tokens_rows_1189_1204": {"n": 67, "match": 67}, "recomputed_C2": "16 of 16 rows: node, S, E, E/S and year equal the frozen candidate; evaluation records found for 16, equal to the frozen candidate, candidate key and eval key for 16"} |
| C3 | I/J cells = node-7 4 h plans of 2-5 MWh (keys confirmed) | holds | l. 1290-1301 | {"plan_tokens_caption_and_rows_1290_1301": {"n": 8, "match": 8}, "recomputed_C3": "i_5a6a88b4 cc4eb5df = d_c52e1670 ([0.5, 2.0] at node 7, 2025); j_f3aa335e fb21d822 = d_4a82a64a ([0.75, 3.0] at node 7, 2025); j_a11d7966 11725e45 = d_3632b0ae ([1.0, 4.0] at node 7, 2025); j_5f3cccb4 cab98853 = d_36686489 ([1.25, 5.0] at node 7, 2025)"} |
| C6 | the 3 x 3 "16 cycles" confirmed; wording: the evaluation, not the fit, was continued | fixed | l. 1411, l. 1417 | {"16_cycles_printed": [], "prose": [{"line": 1411, "quote": "stopping criterion, the evaluation of the plan without storage was continued beyond its stopping point. A subsequent fit to the objective trajectory estimated a further cost decrease of 6.2~k\\euro{}. Accounting for this decrease changes the ratio of the reference unit's multi-sce"}], "caption": [{"line": 1417, "quote": "he production exit; $D$ is the post-hoc settled descent of the $\\boldsymbol{x} = \\boldsymbol{0}$ evaluation continued past it. Gross operational cost, settlement excluded.}}"}], "recomputed_C6": "W98 r2 continuation: replay bitwise through N = 72 (the production exit, cycle 72), continuation record cycles 1-88 -> |
| C7 | W153: 0 sign changes in 60 claims; ONE verdict changes category (the F2 neighbour) | changed (the text now says two verdicts) | l. 1488 | {"text": [{"line": 1488, "quote": "Across the 60 pairwise comparisons reported in this section, applying the salvage credit changes no sign. It changes two verdicts: the investment-year comparison above, and the comparison of the plan returned at twice the base flexibility price with one of its unit neighbours, which is within the uncertified bar in gross terms (0.95 times) and determinate net of salvage (1.22 times), in the returned plan's favour."}], "recomputed_C7": "W153 totals: n_claims 60, n_sign_differs 0, n_verdict_differs 1 (['L:y2030__n5_p0.25_e0.5__n7_p1_e3']: within the uncertified bar gross -> determinate net); frozen T1: 60 claims, 0 sign changes; the claim at 0.9472 x gross (w |
| C8 | the 2025 cohort's salvage is zero (the row may print 0.0) | holds | l. 1458 | {"row": [{"line": 1458, "text": "2025 (reference) & 64.4  & 0.0  & 64.4 \\\\"}], "recomputed_C8": "code: age_at_terminal(2025) = 15, T_rem = max(15 - 15, 0) = 0 -> remaining-life fraction 0, the 2025 term is exactly 0; records: 30 cells with a 2025 plan, max /salvage/ 5.76e-33 EUR (numerical residue of the inactive cohorts' terms, not the 2025 cohort), ref:bd504ecf salvage 0.0; T1: d_net = d_gros"} |
| C9 | 41 new recourse evaluations (14, 20, 7 against 20, 20, 60) confirmed | holds | l. 1516 | {"text": [{"line": 1516, "quote": "_ess_planning_mads} made 41 new recourse evaluations in all (14, 20 and 7 against budgets of 20, 20 and 60), each stopped by the production"}], "recomputed_C9": "W173 item1: s47 14/20 (mesh_local_optimum_unit_poll_failed), s51 20/20 (STOP_FOR_REVIEW_completion_cap), s53 7/60 (poll_failure_at_unit_mesh); sum 41"} |

### [CONFIRM -- W177] comments in main.tex

- l. 1166 (salvage equation items (a)-(d)): items (a)-(d) hold as worded (recomputed C1: (a) confirmed, (b) confirmed, (c) confirmed, (d) confirmed); the equation above them still needs the two corrections (Blocked 1)
- l. 1445 (G rows of T1 (node 7, 0.5 MVA / 1 and 2 MWh, 2030 / 2035) gross / net, all determinate): confirmed: 17/17 tokens of l. 1445-1446 match the frozen T1 G claims; every G claim determinate gross and net (W153 and recomputed): True
- l. 1490 (W153 totals; the neighbour at 0.95 x gross / 1.22 x net; the 2035 - 2030 pair): 60 claims, 0 sign changes, 1 verdict change (['L:y2030__n5_p0.25_e0.5__n7_p1_e3']) -- recomputed C7 confirmed; the 2035 - 2030 investment-year pair is NOT among the 60 (no W153 claim has a yl_y2030 / yl_y2035 cell; it is T4 year_ladder): gross determinate 3.46 x -> net within resolution 0.19 x, sign + -> -
- l. 1518 (W173 realized use 14/20, 20/20, 7/60; 41 is their sum): recomputed C9 confirmed: W173 item1: s47 14/20 (mesh_local_optimum_unit_poll_failed), s51 20/20 (STOP_FOR_REVIEW_completion_cap), s53 7/60 (poll_failure_at_unit_mesh); sum 41

W177a CONFIRM outcomes recomputed now vs committed: C1 = (differs), C2 = (confirmed), C3 = (confirmed), C4 = (differs), C5 = (confirmed (values available)), C6 = (confirmed), C7 = (confirmed), C8 = (confirmed), C9 = (confirmed)

## Part 1 -- counts by scope

| scope | lines | match | approximate | derived | MISMATCH | no counterpart | unchecked | excluded | total |
|---|---|---|---|---|---|---|---|---|---|
| abstract | 138-143 | 7 | 6 | 1 | 0 | 0 | 2 | 0 | 16 |
| highlights (main.tex block) | 155-161 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| highlights.tex | 1-17 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 3 |
| 4 preamble (reported separately) | 1144-1171 | 5 | 0 | 0 | 0 | 0 | 7 | 1 | 13 |
| 4.1 | 1172-1222 | 150 | 0 | 0 | 0 | 0 | 4 | 1 | 155 |
| 4.2 | 1223-1314 | 126 | 0 | 3 | 0 | 0 | 0 | 5 | 134 |
| 4.3 | 1315-1363 | 79 | 0 | 3 | 0 | 0 | 1 | 2 | 85 |
| 4.4 | 1364-1402 | 44 | 3 | 0 | 0 | 0 | 0 | 2 | 49 |
| 4.5 | 1403-1438 | 33 | 0 | 2 | 0 | 0 | 12 | 1 | 48 |
| 4.6 | 1439-1495 | 106 | 0 | 2 | 1 | 0 | 5 | 13 | 127 |
| 4.7 | 1496-1526 | 92 | 0 | 0 | 0 | 0 | 9 | 5 | 106 |
| 4.8 (reported separately) | 1527-1541 | 31 | 4 | 0 | 0 | 0 | 2 | 0 | 37 |
| Conclusions | 1545-1563 | 26 | 3 | 5 | 0 | 0 | 3 | 0 | 37 |

In scope (Addendum 76 Part 1): {'unchecked': 36, 'match': 665, 'derived': 16, 'approximate': 12, 'excluded': 31, 'MISMATCH': 1}; reported separately (section 4 preamble and 4.8): {'unchecked': 9, 'match': 36, 'excluded': 1, 'approximate': 4}

### MISMATCH

- main.tex l. 1488:122 (4.6, prose) written `two` -> 1 [data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d)] as written ("Across the 60 pairwise comparisons ... It changes two verdicts") the count is of the 60: W153 records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3). The second verdict the sentence lists, the 2035 - 2030 investment-year comparison, is NOT one of the 60 (T4 year_ladder, outside T1 / W153): its gross verdict determinate (3.46 x) becomes within resolution net (0.19 x) and its sign changes (D_gross +42,458.65 EUR, D_net -2,286.25 EUR). "Two" holds for the 60 + 1 comparisons, not for the 60; the "changes no sign" clause then fails for that pair (within resolution net, so its net sign is indeterminate). Part 2 (c), W177b-2

### DERIVED (arithmetic)

- main.tex l. 142:75 (abstract) `108` -- derived (holds True): derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108
- main.tex l. 1227:905 (4.2) `24` -- derived (holds True): derived: 1 - e*/e_cost = 1 - 192,557.46 / 253,877.68 = 61,320.21 / 253,877.68 = 24.1534 % (T3 break_even.conservative_A64.breakeven_range[1], margin_to_cost_range[0], e_cost_eur_per_MWh)
- main.tex l. 1227:909 (4.2) `33` -- derived (holds True): derived: 1 - e*/e_cost = 1 - 171,309.08 / 253,877.68 = 82,568.60 / 253,877.68 = 32.5230 % (T3 break_even.conservative_A64.breakeven_range[0], margin_to_cost_range[1], e_cost_eur_per_MWh)
- main.tex l. 1284:814 (4.2) `six` -- derived (holds True): derived: W173 F2 L neighbours 13 - determinate 7 = 6 (= within the bar, plan better 5 + neighbour better within the bar 1); frozen T1 L box-neighbour rows 13, determinate and positive 7, the rest 6
- main.tex l. 1320:126 (4.3) `9.3` -- derived (holds True): derived: T8 value(C2_calfade) 253,539.62 - value(C3_unit) 244,210.35 = 9,329.27 EUR = T1 E:n7_4h_e1_C2_calfade:vs_C3 d_gross 9,329.27 EUR (within resolution, 0.71 x); from the printed T8 values 253.5 - 244.2 = 9.3
- main.tex l. 1320:255 (4.3) `42.1` -- derived (holds True): derived: T8 value(C2) 286,349.77 - value(C3_unit) 244,210.35 = 42,139.42 EUR = T1 E:n7_4h_e1_C2:vs_C3 d_gross 42,139.42 EUR (determinate, 3.28 x); from the printed T8 values 286.3 - 244.2 = 42.1
- main.tex l. 1320:264 (4.3) `28.5` -- derived (holds True): derived: T8 value(C4) 272,760.31 - value(C3_unit) 244,210.35 = 28,549.96 EUR = T1 E:n7_4h_e1_C4:vs_C3 d_gross 28,549.96 EUR (determinate, 2.22 x); from the printed T8 values 272.8 - 244.2 = 28.6
- main.tex l. 1411:742 (4.5) `7` -- derived (holds True): derived: 1 - R(upper) = 1 - 0.933563 = 6.6437 % (T11 three_by_three.rows[7].value)
- main.tex l. 1411:745 (4.5) `9` -- derived (holds True): derived: 1 - R(lower) = 1 - 0.908972 = 9.1028 % (T11 three_by_three.rows[8].value)
- main.tex l. 1443:821 (4.6) `54.5` -- derived (holds True): derived: min of M net over the investment years {2025: 64,417.39 (T1 CHECK:headline_V_minus_I_settled d_net, sign M = -(value - I)), 2030: 56,739.66, 2035: 54,453.40} EUR (T4) = 54,453.40 EUR
- main.tex l. 1443:827 (4.6) `64.4` -- derived (holds True): derived: max of M net over the investment years {2025: 64,417.39 (T1 CHECK:headline_V_minus_I_settled d_net, sign M = -(value - I)), 2030: 56,739.66, 2035: 54,453.40} EUR (T4) = 64,417.39 EUR
- main.tex l. 1553:22 (Conclusions) `108` -- derived (holds True): derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108
- main.tex l. 1553:973 (Conclusions) `7` -- derived (holds True): derived: 1 - R(upper) = 1 - 0.933563 = 6.6437 % (T11 three_by_three.rows[7].value)
- main.tex l. 1553:976 (Conclusions) `9` -- derived (holds True): derived: 1 - R(lower) = 1 - 0.908972 = 9.1028 % (T11 three_by_three.rows[8].value)
- main.tex l. 1553:1050 (Conclusions) `90.9` -- derived (holds True): derived: benefit 90,896,608.40 EUR = 90,896.6 kEUR = 90.8966 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit)
- main.tex l. 1557:615 (Conclusions) `654` -- derived (holds True): derived: coordinated Q (x = 0, cycle 181) 653,873,702.19 EUR = 653,873.7 kEUR = 653.8737 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_Q)

### Per-number table (every assigned token that is not excluded)

| file | line | scope | part | written | status | value at written precision | source / category | note |
|---|---|---|---|---|---|---|---|---|
| main.tex | 141 | abstract | prose | two | unchecked |  | compound adjective | "two-stage" (the model class, not a count) |
| main.tex | 142 | abstract | prose | 15 | match | 15 | SRP1.json Years: sum of the block lengths |  |
| main.tex | 142 | abstract | prose | 108 | derived | 108 | derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108 | buses of the integrated system |
| main.tex | 142 | abstract | prose | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| main.tex | 142 | abstract | prose | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 142 | abstract | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 142 | abstract | prose | 64 | approximate | 64 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].d_gross |  |
| main.tex | 142 | abstract | prose | 1.75 | match | 1.75 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 142 | abstract | prose | 171 | approximate | 171 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[0] |  |
| main.tex | 142 | abstract | prose | 193 | approximate | 193 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[1] |  |
| main.tex | 142 | abstract | prose | 254 | approximate | 254 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.e_cost_eur_per_MWh |  |
| main.tex | 142 | abstract | prose | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |  |
| main.tex | 142 | abstract | prose | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json TransmissionNetwork.num_operation_scenarios (sha256 2a64e3c5) |  |
| main.tex | 142 | abstract | prose | 0.91 | approximate | 0.91 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[8].value |  |
| main.tex | 142 | abstract | prose | 0.93 | approximate | 0.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[7].value |  |
| main.tex | 142 | abstract | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 160 | highlights (main.tex block) | prose | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| main.tex | 1148 | 4 preamble | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1148 | 4 preamble | prose | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |  |
| main.tex | 1148 | 4 preamble | prose | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json TransmissionNetwork.num_operation_scenarios (sha256 2a64e3c5) |  |
| main.tex | 1148 | 4 preamble | prose | two | unchecked |  | language | a difference between two evaluations (definition) |
| main.tex | 1148 | 4 preamble | prose | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1148 | 4 preamble | prose | 15 | match | 15 | SRP1.json Years: sum of the block lengths |  |
| main.tex | 1148 | 4 preamble | prose | 2 | match | 2 | data/SRP1/SRP1.json DiscountFactor (sha256 61a794a7) |  |
| main.tex | 1154 | 4 preamble | equation | 1 | unchecked |  | notation | the 1 of (1 + r) in eq:salvage_credit (formula; checked by CONFIRM C1) |
| main.tex | 1160 | 4 preamble | equation | 0 | unchecked |  | notation | max{0, ...} in eq:salvage_remaining_life (formula; CONFIRM C1: _get_remaining_calendar_life max(..., 0.00)) |
| main.tex | 1165 | 4 preamble | prose | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1167 | 4 preamble | comment | 2035 | match | 2035 | SRP1.json: the last representative year |  |
| main.tex | 1168 | 4 preamble | comment | one | unchecked |  | language | pronoun ("the expected ... one") |
| main.tex | 1176 | 4.1 | prose | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1176 | 4.1 | prose | fourteen | match | 14 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.x0.b_box.n_feasible (sha256 0aa337fc) |  |
| main.tex | 1177 | 4.1 | prose | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1177 | 4.1 | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1177 | 4.1 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1177 | 4.1 | prose | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1177 | 4.1 | prose | 253.5 | match | 253.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].value |  |
| main.tex | 1177 | 4.1 | prose | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].I_other |  |
| main.tex | 1177 | 4.1 | prose | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].d_gross |  |
| main.tex | 1177 | 4.1 | prose | 4.93 | match | 4.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].gross_multiple |  |
| main.tex | 1177 | 4.1 | prose | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (node node 5) |  |
| main.tex | 1177 | 4.1 | prose | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (S node 5) |  |
| main.tex | 1177 | 4.1 | prose | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (E node 5) |  |
| main.tex | 1177 | 4.1 | prose | 49.7 | match | 49.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5].d_gross |  |
| main.tex | 1182 | 4.1 | table caption | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1182 | 4.1 | table caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1187 | 4.1 | table body | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1189 | 4.1 | table body | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (node node 5) |  |
| main.tex | 1189 | 4.1 | table body | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (S node 5) |  |
| main.tex | 1189 | 4.1 | table body | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (E node 5) |  |
| main.tex | 1189 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[pb_y2025_n5_v6].candidate_canonical (dur node 5) |  |
| main.tex | 1189 | 4.1 | table body | 49.7 | match | 49.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5].d_gross |  |
| main.tex | 1189 | 4.1 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5].gross_threshold_or_bar |  |
| main.tex | 1189 | 4.1 | table body | 3.94 | match | 3.94 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5].gross_multiple |  |
| main.tex | 1190 | 4.1 | table body | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_2a0ba8b2].candidate_canonical (node node 5) |  |
| main.tex | 1190 | 4.1 | table body | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_2a0ba8b2].candidate_canonical (S node 5) |  |
| main.tex | 1190 | 4.1 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_2a0ba8b2].candidate_canonical (E node 5) |  |
| main.tex | 1190 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_2a0ba8b2].candidate_canonical (dur node 5) |  |
| main.tex | 1190 | 4.1 | table body | 58.0 | match | 58.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n5_4h_e1].d_gross |  |
| main.tex | 1190 | 4.1 | table body | 13.2 | match | 13.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n5_4h_e1].gross_threshold_or_bar |  |
| main.tex | 1190 | 4.1 | table body | 4.39 | match | 4.39 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n5_4h_e1].gross_multiple |  |
| main.tex | 1191 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1191 | 4.1 | table body | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1191 | 4.1 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1191 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1191 | 4.1 | table body | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].d_gross |  |
| main.tex | 1191 | 4.1 | table body | 13.1 | match | 13.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].gross_threshold_or_bar |  |
| main.tex | 1191 | 4.1 | table body | 4.93 | match | 4.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].gross_multiple |  |
| main.tex | 1192 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c52e1670].candidate_canonical (node node 7) |  |
| main.tex | 1192 | 4.1 | table body | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c52e1670].candidate_canonical (S node 7) |  |
| main.tex | 1192 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c52e1670].candidate_canonical (E node 7) |  |
| main.tex | 1192 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c52e1670].candidate_canonical (dur node 7) |  |
| main.tex | 1192 | 4.1 | table body | 135.5 | match | 135.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e2].d_gross |  |
| main.tex | 1192 | 4.1 | table body | 13.4 | match | 13.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e2].gross_threshold_or_bar |  |
| main.tex | 1192 | 4.1 | table body | 10.13 | match | 10.13 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e2].gross_multiple |  |
| main.tex | 1193 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_4a82a64a].candidate_canonical (node node 7) |  |
| main.tex | 1193 | 4.1 | table body | 0.75 | match | 0.75 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_4a82a64a].candidate_canonical (S node 7) |  |
| main.tex | 1193 | 4.1 | table body | 3 | match | 3.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_4a82a64a].candidate_canonical (E node 7) |  |
| main.tex | 1193 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_4a82a64a].candidate_canonical (dur node 7) |  |
| main.tex | 1193 | 4.1 | table body | 208.3 | match | 208.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e3].d_gross |  |
| main.tex | 1193 | 4.1 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e3].gross_threshold_or_bar |  |
| main.tex | 1193 | 4.1 | table body | 16.49 | match | 16.49 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e3].gross_multiple |  |
| main.tex | 1194 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_3632b0ae].candidate_canonical (node node 7) |  |
| main.tex | 1194 | 4.1 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_3632b0ae].candidate_canonical (S node 7) |  |
| main.tex | 1194 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_3632b0ae].candidate_canonical (E node 7) |  |
| main.tex | 1194 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_3632b0ae].candidate_canonical (dur node 7) |  |
| main.tex | 1194 | 4.1 | table body | 274.0 | match | 274.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e4].d_gross |  |
| main.tex | 1194 | 4.1 | table body | 13.6 | match | 13.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e4].gross_threshold_or_bar |  |
| main.tex | 1194 | 4.1 | table body | 20.19 | match | 20.19 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e4].gross_multiple |  |
| main.tex | 1195 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (node node 7) |  |
| main.tex | 1195 | 4.1 | table body | 1.25 | match | 1.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (S node 7) |  |
| main.tex | 1195 | 4.1 | table body | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (E node 7) |  |
| main.tex | 1195 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (dur node 7) |  |
| main.tex | 1195 | 4.1 | table body | 347.2 | match | 347.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e5].d_gross |  |
| main.tex | 1195 | 4.1 | table body | 23.9 | match | 23.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e5].gross_threshold_or_bar |  |
| main.tex | 1195 | 4.1 | table body | 14.52 | match | 14.52 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e5].gross_multiple |  |
| main.tex | 1196 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical (node node 7) |  |
| main.tex | 1196 | 4.1 | table body | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical (S node 7) |  |
| main.tex | 1196 | 4.1 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical (E node 7) |  |
| main.tex | 1196 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical (dur node 7) |  |
| main.tex | 1196 | 4.1 | table body | 114.1 | match | 114.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e1].d_gross |  |
| main.tex | 1196 | 4.1 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e1].gross_threshold_or_bar |  |
| main.tex | 1196 | 4.1 | table body | 9.03 | match | 9.03 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e1].gross_multiple |  |
| main.tex | 1197 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_a12d95a2].candidate_canonical (node node 7) |  |
| main.tex | 1197 | 4.1 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_a12d95a2].candidate_canonical (S node 7) |  |
| main.tex | 1197 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_a12d95a2].candidate_canonical (E node 7) |  |
| main.tex | 1197 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_a12d95a2].candidate_canonical (dur node 7) |  |
| main.tex | 1197 | 4.1 | table body | 231.7 | match | 231.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e2].d_gross |  |
| main.tex | 1197 | 4.1 | table body | 24.2 | match | 24.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e2].gross_threshold_or_bar |  |
| main.tex | 1197 | 4.1 | table body | 9.56 | match | 9.56 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e2].gross_multiple |  |
| main.tex | 1198 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_f759dd48].candidate_canonical (node node 7) |  |
| main.tex | 1198 | 4.1 | table body | 1.5 | match | 1.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_f759dd48].candidate_canonical (S node 7) |  |
| main.tex | 1198 | 4.1 | table body | 3 | match | 3.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_f759dd48].candidate_canonical (E node 7) |  |
| main.tex | 1198 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_f759dd48].candidate_canonical (dur node 7) |  |
| main.tex | 1198 | 4.1 | table body | 361.7 | match | 361.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e3].d_gross |  |
| main.tex | 1198 | 4.1 | table body | 41.4 | match | 41.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e3].gross_threshold_or_bar |  |
| main.tex | 1198 | 4.1 | table body | 8.74 | match | 8.74 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e3].gross_multiple |  |
| main.tex | 1199 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c7fee8be].candidate_canonical (node node 7) |  |
| main.tex | 1199 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c7fee8be].candidate_canonical (S node 7) |  |
| main.tex | 1199 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c7fee8be].candidate_canonical (E node 7) |  |
| main.tex | 1199 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_c7fee8be].candidate_canonical (dur node 7) |  |
| main.tex | 1199 | 4.1 | table body | 494.4 | match | 494.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e4].d_gross |  |
| main.tex | 1199 | 4.1 | table body | 13.4 | match | 13.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e4].gross_threshold_or_bar |  |
| main.tex | 1199 | 4.1 | table body | 36.91 | match | 36.91 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e4].gross_multiple |  |
| main.tex | 1200 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_9246ed01].candidate_canonical (node node 7) |  |
| main.tex | 1200 | 4.1 | table body | 2.5 | match | 2.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_9246ed01].candidate_canonical (S node 7) |  |
| main.tex | 1200 | 4.1 | table body | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_9246ed01].candidate_canonical (E node 7) |  |
| main.tex | 1200 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_9246ed01].candidate_canonical (dur node 7) |  |
| main.tex | 1200 | 4.1 | table body | 616.6 | match | 616.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e5].d_gross |  |
| main.tex | 1200 | 4.1 | table body | 13.4 | match | 13.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e5].gross_threshold_or_bar |  |
| main.tex | 1200 | 4.1 | table body | 45.94 | match | 45.94 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e5].gross_multiple |  |
| main.tex | 1201 | 4.1 | table body | 9 | match | 9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_0dd237f0].candidate_canonical (node node 9) |  |
| main.tex | 1201 | 4.1 | table body | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_0dd237f0].candidate_canonical (S node 9) |  |
| main.tex | 1201 | 4.1 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_0dd237f0].candidate_canonical (E node 9) |  |
| main.tex | 1201 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_0dd237f0].candidate_canonical (dur node 9) |  |
| main.tex | 1201 | 4.1 | table body | 61.1 | match | 61.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n9_4h_e1].d_gross |  |
| main.tex | 1201 | 4.1 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n9_4h_e1].gross_threshold_or_bar |  |
| main.tex | 1201 | 4.1 | table body | 4.84 | match | 4.84 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n9_4h_e1].gross_multiple |  |
| main.tex | 1202 | 4.1 | table body | 9 | match | 9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_4649234b].candidate_canonical (node node 9) |  |
| main.tex | 1202 | 4.1 | table body | 0.75 | match | 0.75 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_4649234b].candidate_canonical (S node 9) |  |
| main.tex | 1202 | 4.1 | table body | 3 | match | 3.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_4649234b].candidate_canonical (E node 9) |  |
| main.tex | 1202 | 4.1 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[b_4649234b].candidate_canonical (dur node 9) |  |
| main.tex | 1202 | 4.1 | table body | 199.1 | match | 199.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n9_4h_e3].d_gross |  |
| main.tex | 1202 | 4.1 | table body | 13.5 | match | 13.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n9_4h_e3].gross_threshold_or_bar |  |
| main.tex | 1202 | 4.1 | table body | 14.78 | match | 14.78 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n9_4h_e3].gross_multiple |  |
| main.tex | 1203 | 4.1 | table body | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (node node 5) |  |
| main.tex | 1203 | 4.1 | table body | 9 | match | 9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (node node 9) |  |
| main.tex | 1203 | 4.1 | table body | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (S node 5,9) |  |
| main.tex | 1203 | 4.1 | table body | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (E node 5,9) |  |
| main.tex | 1203 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (dur node 5,9) |  |
| main.tex | 1203 | 4.1 | table body | 102.1 | match | 102.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5].d_gross |  |
| main.tex | 1203 | 4.1 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5].gross_threshold_or_bar |  |
| main.tex | 1203 | 4.1 | table body | 8.09 | match | 8.09 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5].gross_multiple |  |
| main.tex | 1204 | 4.1 | table body | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (node node 5) |  |
| main.tex | 1204 | 4.1 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (node node 7) |  |
| main.tex | 1204 | 4.1 | table body | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (year) |  |
| main.tex | 1204 | 4.1 | table body | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (S node 5,7) |  |
| main.tex | 1204 | 4.1 | table body | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (E node 5,7) |  |
| main.tex | 1204 | 4.1 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (dur node 5,7) |  |
| main.tex | 1204 | 4.1 | table body | 93.9 | match | 93.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5].d_gross |  |
| main.tex | 1204 | 4.1 | table body | 13.4 | match | 13.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5].gross_threshold_or_bar |  |
| main.tex | 1204 | 4.1 | table body | 7.00 | match | 7.00 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5].gross_multiple |  |
| main.tex | 1213 | 4.1 | table note | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1213 | 4.1 | table note | 70.7 | match | 70.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e3].uncertified_form_beside_A64_ruling4.bar |  |
| main.tex | 1213 | 4.1 | table note | 2.94 | match | 2.94 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e3].uncertified_form_beside_A64_ruling4.multiple_Q |  |
| main.tex | 1219 | 4.1 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1219 | 4.1 | prose | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].d_gross |  |
| main.tex | 1219 | 4.1 | prose | 347.2 | match | 347.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e5].d_gross |  |
| main.tex | 1219 | 4.1 | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1219 | 4.1 | prose | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (E node 7) |  |
| main.tex | 1219 | 4.1 | prose | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (dur node 7) |  |
| main.tex | 1219 | 4.1 | prose | 114.1 | match | 114.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e1].d_gross |  |
| main.tex | 1219 | 4.1 | prose | 616.6 | match | 616.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_2h_e5].d_gross |  |
| main.tex | 1219 | 4.1 | prose | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_9246ed01].candidate_canonical (dur node 7) |  |
| main.tex | 1219 | 4.1 | prose | two | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (nnodes) |  |
| main.tex | 1219 | 4.1 | prose | 102.1 | match | 102.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5].d_gross |  |
| main.tex | 1219 | 4.1 | prose | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (node node 5) |  |
| main.tex | 1219 | 4.1 | prose | 9 | match | 9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (node node 9) |  |
| main.tex | 1219 | 4.1 | prose | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_156ce2d1].candidate_canonical (year) |  |
| main.tex | 1219 | 4.1 | prose | 93.9 | match | 93.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5].d_gross |  |
| main.tex | 1219 | 4.1 | prose | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (node node 5) |  |
| main.tex | 1219 | 4.1 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (node node 7) |  |
| main.tex | 1219 | 4.1 | prose | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (year) |  |
| main.tex | 1219 | 4.1 | prose | 84.0 | match | 84.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5].d_net |  |
| main.tex | 1227 | 4.2 | prose | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical (dur node 7) |  |
| main.tex | 1227 | 4.2 | prose | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1227 | 4.2 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1227 | 4.2 | prose | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1227 | 4.2 | prose | 4 | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.objective_convention: "e* = b + c/4 - p_cost/4" |  |
| main.tex | 1227 | 4.2 | prose | 4 | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.objective_convention: "e* = b + c/4 - p_cost/4" |  |
| main.tex | 1227 | 4.2 | prose | 256,320 | match | 256,320 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.p_cost_eur_per_MVA |  |
| main.tex | 1227 | 4.2 | prose | six | match | 6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.certified_only.n |  |
| main.tex | 1227 | 4.2 | prose | 183,080 | match | 183,080 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost |  |
| main.tex | 1227 | 4.2 | prose | ten | match | 10 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.banded_mid.n |  |
| main.tex | 1227 | 4.2 | prose | 171,310 | match | 171,310 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[0] |  |
| main.tex | 1227 | 4.2 | prose | 192,560 | match | 192,560 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[1] |  |
| main.tex | 1227 | 4.2 | prose | 61,320 | match | 61,320 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.margin_to_cost_range[0] |  |
| main.tex | 1227 | 4.2 | prose | 253,880 | match | 253,880 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.e_cost_eur_per_MWh |  |
| main.tex | 1227 | 4.2 | prose | 24 | derived | 24 | derived: 1 - e*/e_cost = 1 - 192,557.46 / 253,877.68 = 61,320.21 / 253,877.68 = 24.1534 % (T3 break_even.conservative_A64.breakeven_range[1], margin_to_cost_range[0], e_cost_eur_per_MWh) | energy-cost reduction to pay, lower end (upper break-even) |
| main.tex | 1227 | 4.2 | prose | 33 | derived | 33 | derived: 1 - e*/e_cost = 1 - 171,309.08 / 253,877.68 = 82,568.60 / 253,877.68 = 32.5230 % (T3 break_even.conservative_A64.breakeven_range[0], margin_to_cost_range[1], e_cost_eur_per_MWh) | energy-cost reduction to pay, upper end (lower break-even) |
| main.tex | 1227 | 4.2 | prose | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1227 | 4.2 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1234 | 4.2 | figure caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1234 | 4.2 | figure caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1234 | 4.2 | figure caption | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1234 | 4.2 | figure caption | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical (dur node 7) |  |
| main.tex | 1234 | 4.2 | figure caption | 2 | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/captions.md: "bar + 2 tau" | the interval of the banded fit (captions.md, W165) |
| main.tex | 1240 | 4.2 | table caption | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1240 | 4.2 | table caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1240 | 4.2 | table caption | 256,320 | match | 256,320 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.p_cost_eur_per_MVA |  |
| main.tex | 1241 | 4.2 | table caption | 253,880 | match | 253,880 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.e_cost_eur_per_MWh |  |
| main.tex | 1249 | 4.2 | table body | 4 | match | 4 | T3 conservative A64: intervals |  |
| main.tex | 1249 | 4.2 | table body | 6 | match | 6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.certified_only.n |  |
| main.tex | 1249 | 4.2 | table body | 183,080 | match | 183,080 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost |  |
| main.tex | 1249 | 4.2 | table body | 171,310 | match | 171,310 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[0] |  |
| main.tex | 1249 | 4.2 | table body | 192,560 | match | 192,560 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[1] |  |
| main.tex | 1249 | 4.2 | table body | 61,320 | match | 61,320 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.margin_to_cost_range[0] |  |
| main.tex | 1249 | 4.2 | table body | 82,570 | match | 82,570 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.margin_to_cost_range[1] |  |
| main.tex | 1250 | 4.2 | table body | 3 | match | 3 | T3 committed W145: intervals |  |
| main.tex | 1250 | 4.2 | table body | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.committed_W145.certified_only.n |  |
| main.tex | 1250 | 4.2 | table body | 182,700 | match | 182,700 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.committed_W145.certified_only.breakeven_marginal_4h_energy_cost |  |
| main.tex | 1250 | 4.2 | table body | 173,490 | match | 173,490 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.committed_W145.breakeven_range[0] |  |
| main.tex | 1250 | 4.2 | table body | 190,380 | match | 190,380 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.committed_W145.breakeven_range[1] |  |
| main.tex | 1250 | 4.2 | table body | 63,500 | match | 63,500 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.committed_W145.margin_to_cost_range[0] |  |
| main.tex | 1250 | 4.2 | table body | 80,390 | match | 80,390 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.committed_W145.margin_to_cost_range[1] |  |
| main.tex | 1256 | 4.2 | prose | 14.2 | match | 14.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[H:m1.75:value_minus_I].d_gross |  |
| main.tex | 1256 | 4.2 | prose | 1.75 | match | 1.75 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1256 | 4.2 | prose | 46.3 | match | 46.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].d_gross |  |
| main.tex | 1256 | 4.2 | prose | 1.5 | match | 1.5 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1256 | 4.2 | prose | 1.75 | match | 1.75 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1256 | 4.2 | prose | 1.63 | match | 1.63 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[S1c] (sha256 34f53a2c; committed check, status match): derived: 1.5 + 0.25·(−d(1.5)) / (d(1.75) − d(1.5)) from T1 H:m1. | linear interpolation of value - I between m = 1.5 and 1.75 |
| main.tex | 1256 | 4.2 | prose | 1.5 | match | 1.5 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1256 | 4.2 | prose | 14.5 | match | 14.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m1.5:value_minus_I].d_gross | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1256 | 4.2 | prose | 18.8 | match | 18.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m1.5:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1256 | 4.2 | prose | two | match | 2 | T1 / T10 H rows: positive and determinate margins (m = 1.75, 2) |  |
| main.tex | 1262 | 4.2 | figure caption | 1.5 | match | 1.5 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1275 | 4.2 | table body | 1.00 | match | 1.00 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[CHECK:headline_V_minus_I_settled.other_cell.ref:bd504ecf.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1275 | 4.2 | table body | 64.4 | match | -64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].d_gross |  |
| main.tex | 1275 | 4.2 | table body | 13.1 | match | 13.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].gross_threshold_or_bar |  |
| main.tex | 1275 | 4.2 | table body | 4.93 | match | 4.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].gross_multiple |  |
| main.tex | 1276 | 4.2 | table body | 1.50 | match | 1.50 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1276 | 4.2 | table body | 14.5 | match | -14.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m1.5:value_minus_I].d_gross |  |
| main.tex | 1276 | 4.2 | table body | 18.8 | match | 18.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m1.5:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1276 | 4.2 | table body | 0.77 | match | 0.77 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m1.5:value_minus_I].gross_multiple |  |
| main.tex | 1277 | 4.2 | table body | 1.75 | match | 1.75 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1277 | 4.2 | table body | 14.2 | match | 14.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[H:m1.75:value_minus_I].d_gross |  |
| main.tex | 1277 | 4.2 | table body | 10.8 | match | 10.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[H:m1.75:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1277 | 4.2 | table body | 1.32 | match | 1.32 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[H:m1.75:value_minus_I].gross_multiple |  |
| main.tex | 1278 | 4.2 | table body | 2.00 | match | 2.00 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1278 | 4.2 | table body | 46.3 | match | 46.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].d_gross |  |
| main.tex | 1278 | 4.2 | table body | 12.3 | match | 12.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1278 | 4.2 | table body | 3.77 | match | 3.77 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].gross_multiple |  |
| main.tex | 1284 | 4.2 | prose | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (dur node 7) |  |
| main.tex | 1284 | 4.2 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (node node 7) |  |
| main.tex | 1284 | 4.2 | prose | 46.3 | match | 46.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].d_gross |  |
| main.tex | 1284 | 4.2 | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[h_74eda68d].candidate_canonical (E node 7) |  |
| main.tex | 1284 | 4.2 | prose | 228.4 | match | 228.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e5_value_minus_I].d_gross |  |
| main.tex | 1284 | 4.2 | prose | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (E node 7) |  |
| main.tex | 1284 | 4.2 | prose | 1 | match | 1 | the E step between consecutive cells of the m = 2 ladder (1-5 MWh at node 7) |  |
| main.tex | 1284 | 4.2 | prose | 41.0 | match | 41.0 | T1 I/J increments: smallest (J:e2_to_e3) |  |
| main.tex | 1284 | 4.2 | prose | 51.8 | match | 51.8 | T1 I/J increments: largest (J:e4_to_e5) |  |
| main.tex | 1284 | 4.2 | prose | two | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:5ca4f86c].candidate_canonical (nnodes) |  |
| main.tex | 1284 | 4.2 | prose | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:5ca4f86c].candidate_canonical (year) |  |
| main.tex | 1284 | 4.2 | prose | seventeen | match | 17 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box.n_evaluated_in_search (sha256 0aa337fc) |  |
| main.tex | 1284 | 4.2 | prose | sixty | match | 61 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box.n_feasible (sha256 0aa337fc) | W173 F2 admissible unit neighbours (sixty-one) |
| main.tex | 1284 | 4.2 | prose | one | match | 61 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.b_box.n_feasible (sha256 0aa337fc) | the units part of "sixty-one" |
| main.tex | 1284 | 4.2 | prose | thirteen | match | 13 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.c_L_claims_box_neighbours.n (sha256 0aa337fc) |  |
| main.tex | 1284 | 4.2 | prose | seven | match | 7 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.F2.c_L_claims_box_neighbours.n_positive_determinate (sha256 0aa337fc) |  |
| main.tex | 1284 | 4.2 | prose | six | derived | 6 | derived: W173 F2 L neighbours 13 - determinate 7 = 6 (= within the bar, plan better 5 + neighbour better within the bar 1); frozen T1 L box-neighbour rows 13, determinate and positive 7, the rest 6 | "within resolution of the other six" |
| main.tex | 1290 | 4.2 | table caption | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (dur node 7) |  |
| main.tex | 1290 | 4.2 | table caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (node node 7) |  |
| main.tex | 1290 | 4.2 | table caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (year) |  |
| main.tex | 1297 | 4.2 | table body | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[h_74eda68d].candidate_canonical (E node 7) |  |
| main.tex | 1297 | 4.2 | table body | 46.3 | match | 46.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].d_gross |  |
| main.tex | 1297 | 4.2 | table body | 12.3 | match | 12.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1297 | 4.2 | table body | 3.77 | match | 3.77 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[H:m2:value_minus_I].gross_multiple |  |
| main.tex | 1298 | 4.2 | table body | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[i_5a6a88b4].candidate_canonical (E node 7) |  |
| main.tex | 1298 | 4.2 | table body | 92.8 | match | 92.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[I:m2:e2_value_minus_I].d_gross |  |
| main.tex | 1298 | 4.2 | table body | 12.8 | match | 12.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[I:m2:e2_value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1298 | 4.2 | table body | 7.26 | match | 7.26 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[I:m2:e2_value_minus_I].gross_multiple |  |
| main.tex | 1298 | 4.2 | table body | 46.5 | match | 46.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[I:m2:second_MWh].d_gross |  |
| main.tex | 1298 | 4.2 | table body | 12.8 | match | 12.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[I:m2:second_MWh].gross_threshold_or_bar |  |
| main.tex | 1298 | 4.2 | table body | 3.64 | match | 3.64 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[I:m2:second_MWh].gross_multiple |  |
| main.tex | 1299 | 4.2 | table body | 3 | match | 3.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_f3aa335e].candidate_canonical (E node 7) |  |
| main.tex | 1299 | 4.2 | table body | 133.8 | match | 133.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_value_minus_I].d_gross |  |
| main.tex | 1299 | 4.2 | table body | 27.7 | match | 27.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1299 | 4.2 | table body | 4.83 | match | 4.83 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_value_minus_I].gross_multiple |  |
| main.tex | 1299 | 4.2 | table body | 41.0 | match | 41.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e2_to_e3].d_gross |  |
| main.tex | 1299 | 4.2 | table body | 27.7 | match | 27.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e2_to_e3].gross_threshold_or_bar |  |
| main.tex | 1299 | 4.2 | table body | 1.48 | match | 1.48 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e2_to_e3].gross_multiple |  |
| main.tex | 1300 | 4.2 | table body | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_a11d7966].candidate_canonical (E node 7) |  |
| main.tex | 1300 | 4.2 | table body | 176.6 | match | 176.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].d_gross |  |
| main.tex | 1300 | 4.2 | table body | 9.7 | match | 9.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1300 | 4.2 | table body | 18.14 | match | 18.14 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].gross_multiple |  |
| main.tex | 1300 | 4.2 | table body | 42.8 | match | 42.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].d_gross |  |
| main.tex | 1300 | 4.2 | table body | 27.7 | match | 27.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].gross_threshold_or_bar |  |
| main.tex | 1300 | 4.2 | table body | 1.55 | match | 1.55 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].gross_multiple |  |
| main.tex | 1301 | 4.2 | table body | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (E node 7) |  |
| main.tex | 1301 | 4.2 | table body | 228.4 | match | 228.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e5_value_minus_I].d_gross |  |
| main.tex | 1301 | 4.2 | table body | 26.5 | match | 26.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e5_value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1301 | 4.2 | table body | 8.60 | match | 8.60 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e5_value_minus_I].gross_multiple |  |
| main.tex | 1301 | 4.2 | table body | 51.8 | match | 51.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].d_gross |  |
| main.tex | 1301 | 4.2 | table body | 26.5 | match | 26.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].gross_threshold_or_bar |  |
| main.tex | 1301 | 4.2 | table body | 1.95 | match | 1.95 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].gross_multiple |  |
| main.tex | 1309 | 4.2 | table note | 15.9 | match | 15.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].uncertified_form_beside_A64_ruling4.bar |  |
| main.tex | 1309 | 4.2 | table note | 11.12 | match | 11.12 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].uncertified_form_beside_A64_ruling4.multiple_Q |  |
| main.tex | 1309 | 4.2 | table note | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_a11d7966].candidate_canonical (E node 7) |  |
| main.tex | 1309 | 4.2 | table note | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical (E node 7) |  |
| main.tex | 1309 | 4.2 | table note | 27.7 | match | 27.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].uncertified_form_beside_A64_ruling4.bar |  |
| main.tex | 1309 | 4.2 | table note | 26.5 | match | 26.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].uncertified_form_beside_A64_ruling4.bar |  |
| main.tex | 1309 | 4.2 | table note | 1.55 | match | 1.55 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].uncertified_form_beside_A64_ruling4.multiple_Q |  |
| main.tex | 1309 | 4.2 | table note | 1.95 | match | 1.95 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].uncertified_form_beside_A64_ruling4.multiple_Q |  |
| main.tex | 1309 | 4.2 | table note | three | match | 3 | the three comparisons of the note, each >= 1 x in the uncertified form |  |
| main.tex | 1319 | 4.3 | prose | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1319 | 4.3 | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1319 | 4.3 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1319 | 4.3 | prose | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1319 | 4.3 | prose | 4.1 | match | 4.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[no_ageing].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1319 | 4.3 | prose | 31.6 | match | 31.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1319 | 4.3 | prose | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1319 | 4.3 | prose | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| main.tex | 1319 | 4.3 | prose | 31.6 | match | 31.6 | T8 aged arms: smallest loss (C2) | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1319 | 4.3 | prose | 73.7 | match | 73.7 | T8 aged arms: largest loss (C3_unit) | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1320 | 4.3 | prose | one | unchecked |  | language | "from one another" |
| main.tex | 1320 | 4.3 | prose | 50 | match | 50 | T8 k of C3_unit: R = exp(-N D / k), N D = 8,000 |  |
| main.tex | 1320 | 4.3 | prose | 9.3 | derived | 9.3 | derived: T8 value(C2_calfade) 253,539.62 - value(C3_unit) 244,210.35 = 9,329.27 EUR = T1 E:n7_4h_e1_C2_calfade:vs_C3 d_gross 9,329.27 EUR (within resolution, 0.71 x); from the printed T8 values 253.5 - 244.2 = 9.3 | baseline - retention 50 % (O3) |
| main.tex | 1320 | 4.3 | prose | 42.1 | derived | 42.1 | derived: T8 value(C2) 286,349.77 - value(C3_unit) 244,210.35 = 42,139.42 EUR = T1 E:n7_4h_e1_C2:vs_C3 d_gross 42,139.42 EUR (determinate, 3.28 x); from the printed T8 values 286.3 - 244.2 = 42.1 | cycling only - retention 50 % (O3) |
| main.tex | 1320 | 4.3 | prose | 28.5 | derived | 28.5 | derived: T8 value(C4) 272,760.31 - value(C3_unit) 244,210.35 = 28,549.96 EUR = T1 E:n7_4h_e1_C4:vs_C3 d_gross 28,549.96 EUR (determinate, 2.22 x); from the printed T8 values 272.8 - 244.2 = 28.6 | datasheet - retention 50 % (O3) |
| main.tex | 1326 | 4.3 | figure caption | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| main.tex | 1332 | 4.3 | table caption | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1332 | 4.3 | table caption | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1332 | 4.3 | table caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1332 | 4.3 | table caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1339 | 4.3 | table body | 253.5 | match | 253.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].value |  |
| main.tex | 1339 | 4.3 | table body | 64.4 | match | -64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].value_minus_I |  |
| main.tex | 1339 | 4.3 | table body | 13.1 | match | 13.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C2_calfade:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1339 | 4.3 | table body | 4.93 | match | 4.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C2_calfade:value_minus_I].gross_multiple |  |
| main.tex | 1339 | 4.3 | table body | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].floor_year_070 |  |
| main.tex | 1339 | 4.3 | table body | 0.793 | match | 0.793 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].AE |  |
| main.tex | 1339 | 4.3 | table body | 0.86 | match | 0.86 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].EFC |  |
| main.tex | 1340 | 4.3 | table body | 286.3 | match | 286.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].value |  |
| main.tex | 1340 | 4.3 | table body | 31.6 | match | -31.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].value_minus_I |  |
| main.tex | 1340 | 4.3 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C2:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1340 | 4.3 | table body | 2.50 | match | 2.50 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C2:value_minus_I].gross_multiple |  |
| main.tex | 1340 | 4.3 | table body | 0.888 | match | 0.888 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].AE |  |
| main.tex | 1340 | 4.3 | table body | 1.19 | match | 1.19 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].EFC |  |
| main.tex | 1341 | 4.3 | table body | 50 | match | 50 | T8 k of C3_unit: R = exp(-N D / k) |  |
| main.tex | 1341 | 4.3 | table body | 244.2 | match | 244.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_unit].value |  |
| main.tex | 1341 | 4.3 | table body | 73.7 | match | -73.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_unit].value_minus_I |  |
| main.tex | 1341 | 4.3 | table body | 12.8 | match | 12.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:C3_unit_value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1341 | 4.3 | table body | 5.74 | match | 5.74 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:C3_unit_value_minus_I].gross_multiple |  |
| main.tex | 1341 | 4.3 | table body | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_unit].floor_year_070 |  |
| main.tex | 1341 | 4.3 | table body | 0.786 | match | 0.786 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_unit].AE |  |
| main.tex | 1341 | 4.3 | table body | 0.76 | match | 0.76 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_unit].EFC |  |
| main.tex | 1342 | 4.3 | table body | 50 | match | 50 | T8 k of C3_midblock: R = exp(-N D / k) |  |
| main.tex | 1342 | 4.3 | table body | 252.1 | match | 252.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_midblock].value |  |
| main.tex | 1342 | 4.3 | table body | 65.9 | match | -65.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_midblock].value_minus_I |  |
| main.tex | 1342 | 4.3 | table body | 12.9 | match | 12.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C3_midblock:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1342 | 4.3 | table body | 5.12 | match | 5.12 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C3_midblock:value_minus_I].gross_multiple |  |
| main.tex | 1342 | 4.3 | table body | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_midblock].floor_year_070 |  |
| main.tex | 1342 | 4.3 | table body | 0.836 | match | 0.836 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_midblock].AE |  |
| main.tex | 1342 | 4.3 | table body | 0.76 | match | 0.76 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_midblock].EFC |  |
| main.tex | 1343 | 4.3 | table body | 8,000 | match | 8000 | T8 k of C4 x (-ln 0.70) |  |
| main.tex | 1343 | 4.3 | table body | 272.8 | match | 272.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C4].value |  |
| main.tex | 1343 | 4.3 | table body | 45.2 | match | -45.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C4].value_minus_I |  |
| main.tex | 1343 | 4.3 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C4:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1343 | 4.3 | table body | 3.58 | match | 3.58 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_C4:value_minus_I].gross_multiple |  |
| main.tex | 1343 | 4.3 | table body | 0.834 | match | 0.834 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C4].AE |  |
| main.tex | 1343 | 4.3 | table body | 1.13 | match | 1.13 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C4].EFC |  |
| main.tex | 1344 | 4.3 | table body | 313.8 | match | 313.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[no_ageing].value |  |
| main.tex | 1344 | 4.3 | table body | 4.1 | match | -4.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[no_ageing].value_minus_I |  |
| main.tex | 1344 | 4.3 | table body | 12.9 | match | 12.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_no_ageing:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1344 | 4.3 | table body | 0.32 | match | 0.32 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[E:n7_4h_e1_no_ageing:value_minus_I].gross_multiple |  |
| main.tex | 1344 | 4.3 | table body | 1.000 | match | 1.000 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[no_ageing].AE |  |
| main.tex | 1344 | 4.3 | table body | 1.29 | match | 1.29 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[no_ageing].EFC |  |
| main.tex | 1346 | 4.3 | table body | 0.50 | match | 0.50 | data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json reports.e_soh050.minimum_soh (sha256 3c71d71d) |  |
| main.tex | 1346 | 4.3 | table body | 59.5 | match | -59.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:value_minus_I].d_gross |  |
| main.tex | 1346 | 4.3 | table body | 12.6 | match | 12.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:value_minus_I].gross_threshold_or_bar |  |
| main.tex | 1346 | 4.3 | table body | 4.71 | match | 4.71 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:value_minus_I].gross_multiple |  |
| main.tex | 1354 | 4.3 | table note | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].I |  |
| main.tex | 1354 | 4.3 | table note | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| main.tex | 1354 | 4.3 | table note | 0.50 | match | 0.50 | data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json reports.e_soh050.minimum_soh (sha256 3c71d71d) |  |
| main.tex | 1360 | 4.3 | prose | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| main.tex | 1360 | 4.3 | prose | 2035 | match | 2035 | T8 floor binds (0.70): baseline, C3_unit, C3_midblock (all 2035; the other three never) |  |
| main.tex | 1360 | 4.3 | prose | 50 | match | 50 | T8 k of C3_unit |  |
| main.tex | 1360 | 4.3 | prose | 8,000 | match | 8000 | T8 k of C4 x (-ln 0.70) |  |
| main.tex | 1360 | 4.3 | prose | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| main.tex | 1360 | 4.3 | prose | 0.50 | match | 0.50 | data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json reports.e_soh050.minimum_soh (sha256 3c71d71d) |  |
| main.tex | 1360 | 4.3 | prose | 0.18 | match | 0.18 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N70] (sha256 34f53a2c; committed check, status match): T10 a64.scored.B.efc_per_day[2025] (0.50) − W153 w153_discount_r | EFC/day at 0.50 - at 0.70, 2025 |
| main.tex | 1360 | 4.3 | prose | 0.20 | match | 0.20 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N71] (sha256 34f53a2c; committed check, status match): T10 a64.scored.B.efc_per_day[2030] (0.50) − W153 w153_discount_r | EFC/day at 0.50 - at 0.70, 2030 |
| main.tex | 1360 | 4.3 | prose | 0.27 | match | 0.27 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N72] (sha256 34f53a2c; committed check, status match): T10 a64.scored.B.efc_per_day[2035] (0.50) − W153 w153_discount_r | EFC/day at 0.50 - at 0.70, 2035 |
| main.tex | 1360 | 4.3 | prose | 2025 | match | 2025 | a64.scored.B.efc_per_day: the block-year keys |  |
| main.tex | 1360 | 4.3 | prose | 2030 | match | 2030 | a64.scored.B.efc_per_day: the block-year keys |  |
| main.tex | 1360 | 4.3 | prose | 2035 | match | 2035 | a64.scored.B.efc_per_day: the block-year keys |  |
| main.tex | 1360 | 4.3 | prose | 4.9 | match | 4.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:delta_value_vs_070].d_gross |  |
| main.tex | 1360 | 4.3 | prose | 59.5 | match | 59.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:value_minus_I].d_gross | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1368 | 4.4 | prose | three | match | 3 | T6: coordinated + the two NRF arms |  |
| main.tex | 1368 | 4.4 | prose | 653,873.7 | match | 653873.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_Q |  |
| main.tex | 1368 | 4.4 | prose | 744,770.3 | match | 744770.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.arms.price_taker.Q_best |  |
| main.tex | 1368 | 4.4 | prose | 895,049.9 | match | 895049.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.arms.passive.Q_best |  |
| main.tex | 1368 | 4.4 | prose | two | match | 2 | T6: two NRF arms |  |
| main.tex | 1368 | 4.4 | prose | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| main.tex | 1368 | 4.4 | prose | 36.9 | match | 36.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative |  |
| main.tex | 1368 | 4.4 | prose | 90,896.6 | match | 90896.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit |  |
| main.tex | 1368 | 4.4 | prose | 1,264 | approximate | 1264 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.multiple |  |
| main.tex | 1373 | 4.4 | table caption | three | match | 3 | T6 n_starts (both arms) |  |
| main.tex | 1380 | 4.4 | table body | 653,873.7 | match | 653873.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_Q |  |
| main.tex | 1380 | 4.4 | table body | 71.9 | match | 71.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_reproducibility_band |  |
| main.tex | 1381 | 4.4 | table body | 744,770.3 | match | 744770.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.arms.price_taker.Q_best |  |
| main.tex | 1381 | 4.4 | table body | 0.0 | match | 0.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.arms.price_taker.multimodality_band |  |
| main.tex | 1381 | 4.4 | table body | 90,896.6 | match | 90896.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.decomposition.price_taker_NRF_minus_coordinated_eur |  |
| main.tex | 1381 | 4.4 | table body | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| main.tex | 1381 | 4.4 | table body | 807.6 | match | 807.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted |  |
| main.tex | 1381 | 4.4 | table body | 0.6 | match | 0.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted |  |
| main.tex | 1382 | 4.4 | table body | 895,049.9 | match | 895049.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.arms.passive.Q_best |  |
| main.tex | 1382 | 4.4 | table body | 25.8 | match | 25.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.arms.passive.multimodality_band |  |
| main.tex | 1382 | 4.4 | table body | 241,176.2 | match | 241176.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.decomposition.passive_NRF_minus_coordinated_eur |  |
| main.tex | 1382 | 4.4 | table body | 36.9 | match | 36.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative |  |
| main.tex | 1382 | 4.4 | table body | 781.7 | match | 781.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.passive.curtailment_phase_A_by_start.perturbed.by_agent.TSO.mwh_day_weighted |  |
| main.tex | 1382 | 4.4 | table body | 201.2 | match | 201.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.passive.curtailment_phase_A_by_start.perturbed.by_agent.DSO.mwh_day_weighted |  |
| main.tex | 1390 | 4.4 | figure caption | two | match | 2 | T6: two NRF arms |  |
| main.tex | 1390 | 4.4 | figure caption | three | match | 3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.passive.n_starts |  |
| main.tex | 1390 | 4.4 | figure caption | 90,896.6 | match | 90896.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit |  |
| main.tex | 1390 | 4.4 | figure caption | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| main.tex | 1390 | 4.4 | figure caption | 1,264 | approximate | 1264 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.multiple |  |
| main.tex | 1390 | 4.4 | figure caption | 71.9 | match | 71.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.larger_band |  |
| main.tex | 1390 | 4.4 | figure caption | 90,896.6 | match | 90896.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit |  |
| main.tex | 1390 | 4.4 | figure caption | 77.2 | match | 77.2 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig5_benchmark_data.json computed[share_TSO].value (sha256 82d98c41) |  |
| main.tex | 1390 | 4.4 | figure caption | 0.81 | approximate | 0.81 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted |  |
| main.tex | 1390 | 4.4 | figure caption | 22.8 | match | 22.8 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig5_benchmark_data.json computed[share_DSO].value (sha256 82d98c41) |  |
| main.tex | 1390 | 4.4 | figure caption | 1 | match | 1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.sweep.sweep_passive_cold.n_blocks |  |
| main.tex | 1390 | 4.4 | figure caption | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.sweep.sweep_price_taker_cold.n_blocks |  |
| main.tex | 1390 | 4.4 | figure caption | 12 | match | 12 | SRP1.json: 12 blocks |  |
| main.tex | 1390 | 4.4 | figure caption | 4 | match | 4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.reverse_flow_interface_hours |  |
| main.tex | 1395 | 4.4 | prose | 77.2 | match | 77.2 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig5_benchmark_data.json computed[share_TSO].value (sha256 82d98c41) |  |
| main.tex | 1395 | 4.4 | prose | 22.8 | match | 22.8 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig5_benchmark_data.json computed[share_DSO].value (sha256 82d98c41) |  |
| main.tex | 1395 | 4.4 | prose | 0.6 | match | 0.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted |  |
| main.tex | 1395 | 4.4 | prose | 807.6 | match | 807.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted |  |
| main.tex | 1399 | 4.4 | prose | four | match | 4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.reverse_flow_interface_hours |  |
| main.tex | 1399 | 4.4 | prose | one | match | 1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.sweep.sweep_passive_cold.n_blocks |  |
| main.tex | 1399 | 4.4 | prose | twelve | match | 12 | SRP1.json: 3 years x 4 days = 12 TN blocks |  |
| main.tex | 1399 | 4.4 | prose | eight | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.sweep.sweep_price_taker_cold.n_blocks |  |
| main.tex | 1399 | 4.4 | prose | twelve | match | 12 | SRP1.json: 12 blocks |  |
| main.tex | 1407 | 4.5 | prose | three | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |  |
| main.tex | 1407 | 4.5 | prose | three | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json TransmissionNetwork.num_operation_scenarios (sha256 2a64e3c5) |  |
| main.tex | 1407 | 4.5 | prose | five | match | 5 | 3 x 3 case: Years |  |
| main.tex | 1407 | 4.5 | prose | three | match | 3 | 3 x 3 case: block length |  |
| main.tex | 1407 | 4.5 | prose | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1407 | 4.5 | prose | 236.7 | match | 236.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[0].value |  |
| main.tex | 1407 | 4.5 | prose | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[1].value |  |
| main.tex | 1407 | 4.5 | prose | 81.3 | match | 81.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[2].value | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1407 | 4.5 | prose | 4.52 | match | 4.52 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[4].value |  |
| main.tex | 1407 | 4.5 | prose | 18.0 | match | 18.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[3].value |  |
| main.tex | 1407 | 4.5 | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1407 | 4.5 | prose | ten | match | 10 | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json required_consecutive_cycles (sha256 231558f0) |  |
| main.tex | 1411 | 4.5 | prose | 6.2 | match | 6.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[5].value |  |
| main.tex | 1411 | 4.5 | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1411 | 4.5 | prose | 0.934 | match | 0.934 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[7].value |  |
| main.tex | 1411 | 4.5 | prose | 0.909 | match | 0.909 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[8].value |  |
| main.tex | 1411 | 4.5 | prose | 0.9331 | match | 0.9331 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[9].value |  |
| main.tex | 1411 | 4.5 | prose | 7 | derived | 7 | derived: 1 - R(upper) = 1 - 0.933563 = 6.6437 % (T11 three_by_three.rows[7].value) | value lower in the 3 x 3 instance, lower end |
| main.tex | 1411 | 4.5 | prose | 9 | derived | 9 | derived: 1 - R(lower) = 1 - 0.908972 = 9.1028 % (T11 three_by_three.rows[8].value) | value lower in the 3 x 3 instance, upper end |
| main.tex | 1412 | 4.5 | prose | two | unchecked |  | language | "the two evaluated instances" |
| main.tex | 1417 | 4.5 | table caption | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |  |
| main.tex | 1417 | 4.5 | table caption | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json TransmissionNetwork.num_operation_scenarios (sha256 2a64e3c5) |  |
| main.tex | 1417 | 4.5 | table caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1417 | 4.5 | table caption | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1417 | 4.5 | table caption | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1417 | 4.5 | table caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1417 | 4.5 | table caption | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1417 | 4.5 | table caption | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1424 | 4.5 | table body | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1424 | 4.5 | table body | 72 | match | 72 | data/SRP1/Results/P515S53/w98_continuation/campaign_s53_w98_x0_continuation_r2/campaign_results.json replay_gate.N (sha256 b0d58764) |  |
| main.tex | 1424 | 4.5 | table body | 236.7 | match | 236.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[0].value |  |
| main.tex | 1425 | 4.5 | table body | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[1].value |  |
| main.tex | 1426 | 4.5 | table body | 81.3 | match | -81.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[2].value |  |
| main.tex | 1427 | 4.5 | table body | 18.0 | match | 18.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[3].value |  |
| main.tex | 1427 | 4.5 | table body | 4.52 | match | 4.52 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[4].value |  |
| main.tex | 1428 | 4.5 | table body | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1428 | 4.5 | table body | 6.2 | match | 6.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[5].value |  |
| main.tex | 1429 | 4.5 | table body | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1429 | 4.5 | table body | 253.5 | match | 253.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[6].value |  |
| main.tex | 1430 | 4.5 | table body | 0.934 | match | 0.934 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[7].value |  |
| main.tex | 1430 | 4.5 | table body | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1431 | 4.5 | table body | 0.909 | match | 0.909 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[8].value |  |
| main.tex | 1431 | 4.5 | table body | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1432 | 4.5 | table body | 0.9331 | match | 0.9331 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[9].value |  |
| main.tex | 1433 | 4.5 | table body | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1433 | 4.5 | table body | 72 | match | 72 | data/SRP1/Results/P515S53/w98_continuation/campaign_s53_w98_x0_continuation_r2/campaign_results.json replay_gate.N (sha256 b0d58764) |  |
| main.tex | 1433 | 4.5 | table body | 72 | match | 72 | W98 r2 replay gate: cycles equal (bitwise) |  |
| main.tex | 1443 | 4.6 | prose | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1443 | 4.6 | prose | 68.5 | match | 68.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2030.M_gross_w153_form |  |
| main.tex | 1443 | 4.6 | prose | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1443 | 4.6 | prose | 111.0 | match | 111.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2035.M_gross_w153_form |  |
| main.tex | 1443 | 4.6 | prose | 11.8 | match | 11.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2030.salvage_last |  |
| main.tex | 1443 | 4.6 | prose | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1443 | 4.6 | prose | 56.5 | match | 56.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2035.salvage_last |  |
| main.tex | 1443 | 4.6 | prose | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1443 | 4.6 | prose | two | match | 2 | T4: the two deferral years |  |
| main.tex | 1443 | 4.6 | prose | 42.5 | match | 42.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.D_gross |  |
| main.tex | 1443 | 4.6 | prose | 3.46 | match | 3.46 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.gross_v6.multiple |  |
| main.tex | 1443 | 4.6 | prose | 12.3 | match | 12.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.gross_v6.threshold |  |
| main.tex | 1443 | 4.6 | prose | -2.3 | match | -2.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.D_net |  |
| main.tex | 1443 | 4.6 | prose | 0.19 | match | 0.19 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.net_v6.multiple |  |
| main.tex | 1443 | 4.6 | prose | 54.5 | derived | 54.5 | derived: min of M net over the investment years {2025: 64,417.39 (T1 CHECK:headline_V_minus_I_settled d_net, sign M = -(value - I)), 2030: 56,739.66, 2035: 54,453.40} EUR (T4) = 54,453.40 EUR | net cost over the investment years, lower end |
| main.tex | 1443 | 4.6 | prose | 64.4 | derived | 64.4 | derived: max of M net over the investment years {2025: 64,417.39 (T1 CHECK:headline_V_minus_I_settled d_net, sign M = -(value - I)), 2030: 56,739.66, 2035: 54,453.40} EUR (T4) = 64,417.39 EUR | net cost over the investment years, upper end |
| main.tex | 1445 | 4.6 | comment | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_37b5c499].candidate_canonical (node node 7) |  |
| main.tex | 1446 | 4.6 | comment | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_37b5c499].candidate_canonical (S node 7) |  |
| main.tex | 1446 | 4.6 | comment | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_37b5c499].candidate_canonical (E node 7) |  |
| main.tex | 1446 | 4.6 | comment | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_37b5c499].candidate_canonical (year) |  |
| main.tex | 1446 | 4.6 | comment | 99.0 | match | 99.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_2h_e1_y2030:gross].d_gross |  |
| main.tex | 1446 | 4.6 | comment | 88.3 | match | 88.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_2h_e1_y2030:gross].d_net |  |
| main.tex | 1446 | 4.6 | comment | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_47dce43c].candidate_canonical (year) |  |
| main.tex | 1446 | 4.6 | comment | 139.7 | match | 139.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_2h_e1_y2035:gross].d_gross |  |
| main.tex | 1446 | 4.6 | comment | 85.1 | match | 85.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_2h_e1_y2035:gross].d_net |  |
| main.tex | 1446 | 4.6 | comment | 0.5 | match | 0.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_48749148].candidate_canonical (S node 7) |  |
| main.tex | 1446 | 4.6 | comment | 2 | match | 2.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_48749148].candidate_canonical (E node 7) |  |
| main.tex | 1446 | 4.6 | comment | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_48749148].candidate_canonical (year) |  |
| main.tex | 1446 | 4.6 | comment | 137.4 | match | 137.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_4h_e2_y2030:gross].d_gross |  |
| main.tex | 1446 | 4.6 | comment | 113.1 | match | 113.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_4h_e2_y2030:gross].d_net |  |
| main.tex | 1446 | 4.6 | comment | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[g_9abf31d4].candidate_canonical (year) |  |
| main.tex | 1446 | 4.6 | comment | 223.6 | match | 223.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_4h_e2_y2035:gross].d_gross |  |
| main.tex | 1446 | 4.6 | comment | 109.1 | match | 109.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[G:n7_4h_e2_y2035:gross].d_net |  |
| main.tex | 1451 | 4.6 | table caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (node node 7) |  |
| main.tex | 1451 | 4.6 | table caption | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (S node 7) |  |
| main.tex | 1451 | 4.6 | table caption | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (E node 7) |  |
| main.tex | 1451 | 4.6 | table caption | 0 | unchecked |  | notation | the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure) |
| main.tex | 1451 | 4.6 | table caption | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1451 | 4.6 | table caption | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1451 | 4.6 | table caption | 3.46 | match | 3.46 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.gross_v6.multiple |  |
| main.tex | 1451 | 4.6 | table caption | 12.3 | match | 12.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.gross_v6.threshold |  |
| main.tex | 1451 | 4.6 | table caption | 0.19 | match | 0.19 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.net_v6.multiple |  |
| main.tex | 1458 | 4.6 | table body | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1458 | 4.6 | table body | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].d_gross | M = I + Q(x) - Q(0) = -(value - I); sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1458 | 4.6 | table body | 0.0 | match | 0.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.cells[ref:bd504ecf].salvage | the 2025 cohort: T^Rem = 0 (W177a CONFIRM C8) |
| main.tex | 1458 | 4.6 | table body | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].d_net | M net = -(value - I) net; sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1459 | 4.6 | table body | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1459 | 4.6 | table body | 68.5 | match | 68.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2030.M_gross_w153_form |  |
| main.tex | 1459 | 4.6 | table body | 11.8 | match | 11.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2030.salvage_last |  |
| main.tex | 1459 | 4.6 | table body | 56.7 | match | 56.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2030.M_net_w153_form |  |
| main.tex | 1460 | 4.6 | table body | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1460 | 4.6 | table body | 111.0 | match | 111.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2035.M_gross_w153_form |  |
| main.tex | 1460 | 4.6 | table body | 56.5 | match | 56.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2035.salvage_last |  |
| main.tex | 1460 | 4.6 | table body | 54.5 | match | 54.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.per_year.2035.M_net_w153_form |  |
| main.tex | 1461 | 4.6 | table body | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1461 | 4.6 | table body | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1461 | 4.6 | table body | 42.5 | match | 42.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.D_gross |  |
| main.tex | 1461 | 4.6 | table body | 2.3 | match | -2.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.D_net |  |
| main.tex | 1467 | 4.6 | prose | 2025 | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/export/T7.tex: "I paid in 2025" | T7 caption |
| main.tex | 1467 | 4.6 | prose | 277.1 | match | 277.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].V |  |
| main.tex | 1467 | 4.6 | prose | 203.4 | match | 203.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].V |  |
| main.tex | 1467 | 4.6 | prose | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].rate |  |
| main.tex | 1467 | 4.6 | prose | 0 | match | 0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].rate |  |
| main.tex | 1467 | 4.6 | prose | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].rate |  |
| main.tex | 1467 | 4.6 | prose | 40.8 | match | 40.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1467 | 4.6 | prose | 2.57 | match | 2.57 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].multiple |  |
| main.tex | 1467 | 4.6 | prose | 114.6 | match | 114.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1467 | 4.6 | prose | 8.78 | match | 8.78 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].multiple |  |
| main.tex | 1472 | 4.6 | table caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1472 | 4.6 | table caption | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1472 | 4.6 | table caption | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1472 | 4.6 | table caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (year) |  |
| main.tex | 1472 | 4.6 | table caption | one | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/export/T7.tex: "one discount factor per representative year" | T7 caption |
| main.tex | 1472 | 4.6 | table caption | five | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/export/T7.tex: "applied to the five years of its block" | T7 caption |
| main.tex | 1472 | 4.6 | table caption | 2025 | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/export/T7.tex: "I paid in 2025" | T7 caption |
| main.tex | 1479 | 4.6 | table body | 0 | match | 0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].rate |  |
| main.tex | 1479 | 4.6 | table body | 277.1 | match | 277.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].V |  |
| main.tex | 1479 | 4.6 | table body | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].I |  |
| main.tex | 1479 | 4.6 | table body | 40.8 | match | -40.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].value_minus_I |  |
| main.tex | 1479 | 4.6 | table body | 15.9 | match | 15.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].threshold_a61_conservative |  |
| main.tex | 1479 | 4.6 | table body | 2.57 | match | 2.57 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].multiple |  |
| main.tex | 1480 | 4.6 | table body | 2 | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].rate |  |
| main.tex | 1480 | 4.6 | table body | 253.5 | match | 253.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].V |  |
| main.tex | 1480 | 4.6 | table body | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].I |  |
| main.tex | 1480 | 4.6 | table body | 64.4 | match | -64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].value_minus_I |  |
| main.tex | 1480 | 4.6 | table body | 13.1 | match | 13.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].threshold_a61_conservative |  |
| main.tex | 1480 | 4.6 | table body | 4.93 | match | 4.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].multiple |  |
| main.tex | 1481 | 4.6 | table body | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].rate |  |
| main.tex | 1481 | 4.6 | table body | 225.2 | match | 225.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].V |  |
| main.tex | 1481 | 4.6 | table body | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].I |  |
| main.tex | 1481 | 4.6 | table body | 92.7 | match | -92.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].value_minus_I |  |
| main.tex | 1481 | 4.6 | table body | 13.1 | match | 13.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].threshold_a61_conservative |  |
| main.tex | 1481 | 4.6 | table body | 7.10 | match | 7.10 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].multiple |  |
| main.tex | 1482 | 4.6 | table body | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].rate |  |
| main.tex | 1482 | 4.6 | table body | 203.4 | match | 203.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].V |  |
| main.tex | 1482 | 4.6 | table body | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].I |  |
| main.tex | 1482 | 4.6 | table body | 114.6 | match | -114.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].value_minus_I |  |
| main.tex | 1482 | 4.6 | table body | 13.1 | match | 13.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].threshold_a61_conservative |  |
| main.tex | 1482 | 4.6 | table body | 8.78 | match | 8.78 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].multiple |  |
| main.tex | 1488 | 4.6 | prose | 60 | match | 60 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_claims (sha256 215fd21d) |  |
| main.tex | 1488 | 4.6 | prose | two | MISMATCH | 1 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d) | as written ("Across the 60 pairwise comparisons ... It changes two verdicts") the count is of the 60: W153 records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3). The second verdict the sentence lists, the 20 |
| main.tex | 1488 | 4.6 | prose | one | unchecked |  | language | "one of its unit neighbours" |
| main.tex | 1488 | 4.6 | prose | 0.95 | match | 0.95 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[L:y2030__n5_p0.25_e0.5__n7_p1_e3].gross_multiple |  |
| main.tex | 1488 | 4.6 | prose | 1.22 | match | 1.22 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[L:y2030__n5_p0.25_e0.5__n7_p1_e3].net_multiple |  |
| main.tex | 1490 | 4.6 | comment | 0 | match | 0 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_sign_differs (sha256 215fd21d) |  |
| main.tex | 1490 | 4.6 | comment | 60 | match | 60 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_claims (sha256 215fd21d) |  |
| main.tex | 1491 | 4.6 | comment | 4.6 | unchecked |  | cross-reference | the section 4.6 placeholder (Overleaf cfee0ba) |
| main.tex | 1491 | 4.6 | comment | one | match | 1 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d) |  |
| main.tex | 1492 | 4.6 | comment | two | unchecked |  | quotation | the letter's R3.1 count, quoted in the comment; the letter is outside this scope (the count is Part 2 (c), W177b-2; W153 gives one among the 60, see l. 1488) |
| main.tex | 1492 | 4.6 | comment | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1492 | 4.6 | comment | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1493 | 4.6 | comment | 60 | match | 60 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_claims (sha256 215fd21d) |  |
| main.tex | 1493 | 4.6 | comment | one | unchecked |  | language | "is one of the 60" |
| main.tex | 1493 | 4.6 | comment | 60 | match | 60 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_claims (sha256 215fd21d) |  |
| main.tex | 1500 | 4.7 | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1501 | 4.7 | prose | 42 | match | 42 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_cells (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1501 | 4.7 | prose | 32 | match | 32 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_certified (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 24 | match | 24 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.branch_counts_certified.oscillatory (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 8 | match | 8 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.branch_counts_certified.monotone (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 10 | match | 10 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_uncertified (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 5 | match | 5 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.uncertified_by_cause.gap clause.n (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 2 | match | 2 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.uncertified_by_cause.lapse reset.n (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 3 | match | 3 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.uncertified_by_cause.growth test.n (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 174 | match | 174 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.k_star.median (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 65 | match | 65 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.cycles_after_k0_k_star_minus_N.median (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 21 | match | 21 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.cycles_after_k0_k_star_minus_N.min (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 89 | match | 89 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.cycles_after_k0_k_star_minus_N.max (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | 0.86 | match | 0.86 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.range_over_tau_at_k_star.median (sha256 ff037985) |  |
| main.tex | 1501 | 4.7 | prose | four | match | 4 | frozen cells_appended_w160 (T10 rows), certified |  |
| main.tex | 1501 | 4.7 | prose | Ten | match | 10 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.at_or_above_0_95_tau.n_counted |  |
| main.tex | 1501 | 4.7 | prose | 46 | match | 46 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N23] (sha256 34f53a2c; committed check, status match): frozen tables.at_or_above_0_95_tau.registry: certified (status i | certificates the tables use (49 certified cells, 1 superseded, 2 bitwise twins) |
| main.tex | 1501 | 4.7 | prose | 5 | match | 5 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N24] (sha256 34f53a2c; committed check, status match): frozen constants.FLAG_RANGE_OVER_TAU / registry threshold 0.95:  | 1 - 0.95 (the >= 0.95 tau flag) |
| main.tex | 1501 | 4.7 | prose | 0.9 | match | max 0.882 τ | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N25] (sha256 34f53a2c; committed check, status match): records: Q(last) − Q(k*) of the runs that continued past a settl | largest movement past a settling-rule certificate: 0.882 tau |
| main.tex | 1501 | 4.7 | prose | 15 | match | 15 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N4] (sha256 34f53a2c; committed check, status match): W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€), | x0: 14,970.62 EUR after the residual-rule certificate N = 132 |
| main.tex | 1501 | 4.7 | prose | 21 | match | 21 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N5] (sha256 34f53a2c; committed check, status match): W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed | unit: 20,806.32 EUR after N = 112 |
| main.tex | 1505 | 4.7 | prose | two | unchecked |  | language | the two outcomes (certified / refused) |
| main.tex | 1505 | 4.7 | prose | 181 | match | 181 | W165 fig4 data a.k_star |  |
| main.tex | 1505 | 4.7 | prose | 58 | match | 58 | fig4 data: k* - k0 |  |
| main.tex | 1505 | 4.7 | prose | 123 | match | 123 | fig4 data a.k0 |  |
| main.tex | 1505 | 4.7 | prose | 0.93 | match | 0.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.cells[ref:7aa017f0].range_over_tau |  |
| main.tex | 1505 | 4.7 | prose | 132 | match | 132 | fig4 data a.N |  |
| main.tex | 1505 | 4.7 | prose | 15.0 | match | 15.0 | fig4 data check a.N_and_slack_equal_summary: Q(N) - Q(k*) |  |
| main.tex | 1505 | 4.7 | prose | 1.5 | match | 1.5 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig4_q_versus_cycle_data.json values[b.h_f9eae48f.flex_price_multiplier].value (sha256 aae0e1a7) |  |
| main.tex | 1505 | 4.7 | prose | 2 | match | 2 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N12] (sha256 34f53a2c; committed check, status match): settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND) | GAP_BOUND = tau / 2 |
| main.tex | 1505 | 4.7 | prose | 2.5 | match | 2.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.cells[h_f9eae48f].gap |  |
| main.tex | 1505 | 4.7 | prose | 6.3 | match | 6.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.cells[h_f9eae48f].slack |  |
| main.tex | 1511 | 4.7 | figure caption | two | unchecked |  | language | two recorded evaluations (panels a and b-c) |
| main.tex | 1511 | 4.7 | figure caption | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1511 | 4.7 | figure caption | 123 | match | 123 | fig4 data a.k0 |  |
| main.tex | 1511 | 4.7 | figure caption | 181 | match | 181 | fig4 data a.k_star |  |
| main.tex | 1511 | 4.7 | figure caption | 32 | match | 32 | fig4 data a.window [150, 181] |  |
| main.tex | 1511 | 4.7 | figure caption | 0.93 | match | 0.93 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.cells[ref:7aa017f0].range_over_tau |  |
| main.tex | 1511 | 4.7 | figure caption | 132 | match | 132 | fig4 data a.N |  |
| main.tex | 1511 | 4.7 | figure caption | 15.0 | match | 15.0 | fig4 data: Q(N) - Q(k*) |  |
| main.tex | 1511 | 4.7 | figure caption | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[h_f9eae48f].candidate_canonical (node node 7) |  |
| main.tex | 1511 | 4.7 | figure caption | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[h_f9eae48f].candidate_canonical (S node 7) |  |
| main.tex | 1511 | 4.7 | figure caption | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[h_f9eae48f].candidate_canonical (E node 7) |  |
| main.tex | 1511 | 4.7 | figure caption | 2025 | match | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[h_f9eae48f].candidate_canonical (year) |  |
| main.tex | 1511 | 4.7 | figure caption | 1.5 | match | 1.5 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig4_q_versus_cycle_data.json values[b.h_f9eae48f.flex_price_multiplier].value (sha256 aae0e1a7) |  |
| main.tex | 1511 | 4.7 | figure caption | 85 | match | 85 | fig4 data b.k0 |  |
| main.tex | 1511 | 4.7 | figure caption | 194 | match | 194 | fig4 data b.cap |  |
| main.tex | 1511 | 4.7 | figure caption | 2 | match | 2 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N12] (sha256 34f53a2c; committed check, status match): settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND) | GAP_BOUND = tau / 2 |
| main.tex | 1511 | 4.7 | figure caption | 59 | match | 59 | fig4 data: gap refusals (T2 contributing "59 refusals") |  |
| main.tex | 1516 | 4.7 | prose | three | match | 3 | W173 item1: s47, s51, s53 |  |
| main.tex | 1516 | 4.7 | prose | 41 | match | 41 | W173 sum |  |
| main.tex | 1516 | 4.7 | prose | 14 | match | 14 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s47.n_new_evaluations_total (sha256 0aa337fc) |  |
| main.tex | 1516 | 4.7 | prose | 20 | match | 20 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s51.n_new_evaluations_total (sha256 0aa337fc) |  |
| main.tex | 1516 | 4.7 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s53.n_new_evaluations_total (sha256 0aa337fc) |  |
| main.tex | 1516 | 4.7 | prose | 20 | match | 20 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s47.max_new_evaluations (sha256 0aa337fc) |  |
| main.tex | 1516 | 4.7 | prose | 20 | match | 20 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s51.max_new_evaluations (sha256 0aa337fc) |  |
| main.tex | 1516 | 4.7 | prose | 60 | match | 60 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s53.max_new_evaluations (sha256 0aa337fc) |  |
| main.tex | 1517 | 4.7 | prose | Two | match | 2 | W173 item1 termination: unit poll failed (s47, s53) |  |
| main.tex | 1517 | 4.7 | prose | one | match | 1 | W173 item1 termination: STOP_FOR_REVIEW_completion_cap (s51) |  |
| main.tex | 1517 | 4.7 | prose | two | match | 2 | "the first two" (s47, s53) |  |
| main.tex | 1517 | 4.7 | prose | third | unchecked |  | language | ordinal ("the third") |
| main.tex | 1518 | 4.7 | comment | 14 | match | 14 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s47.n_new_evaluations_total (sha256 0aa337fc) |  |
| main.tex | 1518 | 4.7 | comment | 20 | match | 20 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s47.max_new_evaluations (sha256 0aa337fc) |  |
| main.tex | 1518 | 4.7 | comment | 20 | match | 20 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s51.n_new_evaluations_total (sha256 0aa337fc) |  |
| main.tex | 1518 | 4.7 | comment | 20 | match | 20 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s51.max_new_evaluations (sha256 0aa337fc) |  |
| main.tex | 1518 | 4.7 | comment | 7 | match | 7 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s53.n_new_evaluations_total (sha256 0aa337fc) |  |
| main.tex | 1518 | 4.7 | comment | 60 | match | 60 | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item1.s53.max_new_evaluations (sha256 0aa337fc) |  |
| main.tex | 1518 | 4.7 | comment | 41 | match | 41 | W173 sum |  |
| main.tex | 1519 | 4.7 | prose | 42 | match | 42 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_cells (sha256 ff037985) |  |
| main.tex | 1519 | 4.7 | prose | 220,876 | match | 220876 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.wall_s_total (sha256 ff037985) |  |
| main.tex | 1519 | 4.7 | prose | 8,570 | match | 8570 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.cycles_total (sha256 ff037985) |  |
| main.tex | 1519 | 4.7 | prose | 25.6 | match | 25.6 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.wall_s_per_cycle.median (sha256 ff037985) |  |
| main.tex | 1519 | 4.7 | prose | 32 | match | 32 GiB | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N57] (sha256 34f53a2c; committed check, status match): sysctl hw.memsize now; v6 spec memory_preflight hw_memsize_bytes | 34,359,738,368 bytes = 32 GiB |
| main.tex | 1519 | 4.7 | prose | 27.0 | match | 27.0 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N55] (sha256 34f53a2c; committed check, status match): W159 c3 sw_vers ProductVersion; sw_vers now | macOS ProductVersion |
| main.tex | 1519 | 4.7 | prose | one | match | 1 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N44] (sha256 34f53a2c; committed check, status match): v6 spec configuration.concurrency | one evaluation at a time |
| main.tex | 1519 | 4.7 | prose | single | match | thread caps 1 on every table evaluation | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N45] (sha256 34f53a2c; committed check, status match): W159 c1: harness THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VE | single-threaded (thread caps) |
| main.tex | 1519 | 4.7 | prose | one | match | thread caps 1 on every table evaluation | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N45] (sha256 34f53a2c; committed check, status match): W159 c1: harness THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VE | one thread (OMP / MKL / OpenBLAS / vecLib / NumExpr) |
| main.tex | 1523 | 4.7 | prose | 72 | match | 72/72 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N47] (sha256 34f53a2c; committed check, status match): W98 r2 campaign_results.json (4689475e) replay_gate: bitwise_thr | 3 x 3 x = 0 replay 72/72 |
| main.tex | 1523 | 4.7 | prose | three | match | 3/3 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N49] (sha256 34f53a2c; committed check, status match): W101 campaign_results (x0, unit, C*): gates.G19_replay_bitwise_1 | three SRP1 reference continuations |
| main.tex | 1523 | 4.7 | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1523 | 4.7 | prose | Two | match | DSO/5/2035/Winter 109, DSO/7/2025/Winter | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N58] (sha256 34f53a2c; committed check, status match): W153 cells[].acceptable_clean_after_N_detail blocks, tallied; to | two distribution blocks |
| main.tex | 1523 | 4.7 | prose | 7 | match | DSO/5/2035/Winter 109, DSO/7/2025/Winter | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N58] (sha256 34f53a2c; committed check, status match): W153 cells[].acceptable_clean_after_N_detail blocks, tallied; to | node 7 (DSO/7/2025/Winter) |
| main.tex | 1523 | 4.7 | prose | 2025 | match | DSO/5/2035/Winter 109, DSO/7/2025/Winter | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N58] (sha256 34f53a2c; committed check, status match): W153 cells[].acceptable_clean_after_N_detail blocks, tallied; to | DSO/7/2025/Winter |
| main.tex | 1523 | 4.7 | prose | 5 | match | DSO/5/2035/Winter 109, DSO/7/2025/Winter | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N58] (sha256 34f53a2c; committed check, status match): W153 cells[].acceptable_clean_after_N_detail blocks, tallied; to | node 5 (DSO/5/2035/Winter) |
| main.tex | 1523 | 4.7 | prose | 2035 | match | DSO/5/2035/Winter 109, DSO/7/2025/Winter | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N58] (sha256 34f53a2c; committed check, status match): W153 cells[].acceptable_clean_after_N_detail blocks, tallied; to | DSO/5/2035/Winter |
| main.tex | 1523 | 4.7 | prose | ten | match | 10 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N59] (sha256 34f53a2c; committed check, status match): settling_criterion_v5.CLEAN_FACTOR (W153 acceptable_clean = prim | CLEAN_FACTOR 10 |
| main.tex | 1523 | 4.7 | prose | Two | unchecked |  | language | two offsets (the tight tail and the RES slack), enumerated in the sentence |
| main.tex | 1523 | 4.7 | prose | three | match | 3 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N62] (sha256 34f53a2c; committed check, status match): W86 tail recert campaign_results.json per_cell (x0, unit, C*) | the three SRP1 references |
| main.tex | 1523 | 4.7 | prose | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| main.tex | 1523 | 4.7 | prose | 1.0 | match | 1.0 | W163 N63 (W86 per_cell dQ_relative x 1e6): smallest 1.036 |  |
| main.tex | 1523 | 4.7 | prose | 1.2 | match | 1.2 | W163 N63: largest 1.200 |  |
| main.tex | 1523 | 4.7 | prose | 10^{-6} | match |  | the unit of the 1.0-1.2 range (W163 N63) |  |
| main.tex | 1523 | 4.7 | prose | 10^{-6} | match | 1e-06 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N61] (sha256 34f53a2c; committed check, status match): v6 spec convergence_depth_tail.compl_inf_tol; admm_parameters.py | tail compl_inf_tol |
| main.tex | 1523 | 4.7 | prose | 10^{-4} | match | 0.0001 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N60] (sha256 34f53a2c; committed check, status match): IPOPT default compl_inf_tol: `/usr/local/bin/ipopt --print-optio | DSO compl_inf_tol before the tail |
| main.tex | 1523 | 4.7 | prose | 5 | match | parent_of_tail_commit 0.0005, w86_campai | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N74] (sha256 34f53a2c; committed check, status match): data/SRP1/case9/case9_params.json solver.options.compl_inf_tol r | TSO 5 x 10^-4 |
| main.tex | 1523 | 4.7 | prose | 10^{-4} | match | parent_of_tail_commit 0.0005, w86_campai | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N74] (sha256 34f53a2c; committed check, status match): data/SRP1/case9/case9_params.json solver.options.compl_inf_tol r | TSO 5 x 10^-4 |
| main.tex | 1523 | 4.7 | prose | 10^{-5} | match | 1e-05 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N33] (sha256 34f53a2c; committed check, status match): definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_hel | RES availability slack |
| main.tex | 1523 | 4.7 | prose | 7.8 | match | 7.8 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N64] (sha256 34f53a2c; committed check, status match): W109 (f6e3533f) tso + dso priced_neg_part_eur_block_weighted (k€ | 7,848.08 EUR |
| main.tex | 1523 | 4.7 | prose | 1.2 | match | 1.2e-5 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N65] (sha256 34f53a2c; committed check, status match): W109 priced negative parts / Q of the settled x0 (T2 ref:7aa017f | 1.2002e-5 of Q |
| main.tex | 1523 | 4.7 | prose | 10^{-5} | match | 1.2e-5 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N65] (sha256 34f53a2c; committed check, status match): W109 priced negative parts / Q of the settled x0 (T2 ref:7aa017f | 1.2 x 10^-5 |
| main.tex | 1531 | 4.8 | prose | 0 | unchecked |  | notation | lambda in [0, c^flex] (interval endpoint) |
| main.tex | 1531 | 4.8 | prose | one | unchecked |  | language | one fingerprint |
| main.tex | 1531 | 4.8 | prose | -8.8 | match | -8.8 | T9 dead-zone cells: t_sum at cap, largest |  |
| main.tex | 1531 | 4.8 | prose | -9.4 | match | -9.4 | T9: smallest |  |
| main.tex | 1531 | 4.8 | prose | 55 | approximate | ['55'] | T9 node-5 share (five cells) |  |
| main.tex | 1531 | 4.8 | prose | 5 | match | 5 | T9 share_by_node key 5 |  |
| main.tex | 1531 | 4.8 | prose | 31 | approximate | ['31'] | T9 node-9 share |  |
| main.tex | 1531 | 4.8 | prose | 9 | match | 9 | T9 share_by_node key 9 |  |
| main.tex | 1531 | 4.8 | prose | 14 | approximate | ['14'] | T9 node-7 share |  |
| main.tex | 1531 | 4.8 | prose | 7 | match | 7 | T9 share_by_node key 7 |  |
| main.tex | 1531 | 4.8 | prose | 0.69 | approximate | 0.682-0.700 | T9 pf_primal last (0.682-0.700) |  |
| main.tex | 1531 | 4.8 | prose | 5 | match | 5.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_36686489].candidate_canonical (E node 7) |  |
| main.tex | 1531 | 4.8 | prose | +1.1 | match | 1.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.cells[d_36686489].t_sum_end |  |
| main.tex | 1535 | 4.8 | prose | 21 | match | 21 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.non_clean_cycles_after_N_total (sha256 ff037985) |  |
| main.tex | 1535 | 4.8 | prose | 10 | match | 10 | W153: cells with non-clean cycles after N |  |
| main.tex | 1535 | 4.8 | prose | 2035 | match | ['TSO/2035/Spring'] (6 of 21) | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N32] (sha256 34f53a2c; committed check, status match): W153 certification statistics (48e76c9f) cells[].non_clean_after | recurring block TSO/2035/Spring (6 of 21) |
| main.tex | 1535 | 4.8 | prose | 10^{-5} | match | 1e-05 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N33] (sha256 34f53a2c; committed check, status match): definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_hel | RES slack |
| main.tex | 1535 | 4.8 | prose | 62 | match | 62 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N35] (sha256 34f53a2c; committed check, status match): W109 (f6e3533f) tso + dso neg_part_eur_at_1_block_weighted (MWh- | 61.91 MWh-equivalent |
| main.tex | 1535 | 4.8 | prose | 3 | match | 3 (3.36) | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N36] (sha256 34f53a2c; committed check, status match): P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md (ce96d492): "≈ 3 τ left | 3.36 tau estimate |
| main.tex | 1535 | 4.8 | prose | 44 | match | 44 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N37] (sha256 34f53a2c; committed check, status match): frozen spec v41 (fcea4b38, the C* extension W110) report_only_ru | L_MONO = 44 (spec v41) |
| main.tex | 1539 | 4.8 | prose | one | match | 1 | W173 box_definition: one year coordinate zY |  |
| main.tex | 1539 | 4.8 | prose | three | match | 3 | SRP1.json Years |  |
| main.tex | 1539 | 4.8 | prose | five | match | 5 | SRP1.json block length |  |
| main.tex | 1539 | 4.8 | prose | five | match | 5 | 3 x 3 case Years |  |
| main.tex | 1539 | 4.8 | prose | three | match | 3 | 3 x 3 block length |  |
| main.tex | 1539 | 4.8 | prose | four | match | 4 | SRP1.json Days |  |
| main.tex | 1539 | 4.8 | prose | one | match | 1 | SRP1.json TransmissionNetwork (one) |  |
| main.tex | 1539 | 4.8 | prose | 9 | match | 9 | case9_2025.json nodes |  |
| main.tex | 1539 | 4.8 | prose | three | match | 3 | SRP1.json DistributionNetworks |  |
| main.tex | 1539 | 4.8 | prose | 33 | match | 33 | case33_*_2025.json nodes |  |
| main.tex | 1539 | 4.8 | prose | 0.70 | match | 0.70 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N38] (sha256 34f53a2c; committed check, status match): v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the | baseline floor |
| main.tex | 1539 | 4.8 | prose | 1.04 | match | 1.04 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[L9] (sha256 34f53a2c; committed check, status match): T8 ε_AE band over resolvable arms (0.70) |  |
| main.tex | 1539 | 4.8 | prose | 1.85 | match | 1.85 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[L10] (sha256 34f53a2c; committed check, status match): T8 |  |
| main.tex | 1539 | 4.8 | prose | 0.41 | match | 0.41 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[L11a] (sha256 34f53a2c; committed check, status match): T8 ε_AE (0.50, superseded), min over the resolvable arms (eps_A |  |
| main.tex | 1539 | 4.8 | prose | 0.62 | match | 0.62 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[L11b] (sha256 34f53a2c; committed check, status match): T8 ε_AE (0.50, superseded), max over the resolvable arms |  |
| main.tex | 1539 | 4.8 | prose | three | match | three | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[L11c] (sha256 34f53a2c; committed check, status match): T8 count of resolvable arms (the arms of the 0.70 band 1.04–1.8 |  |
| main.tex | 1539 | 4.8 | prose | 0.50 | match | 0.50 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N39] (sha256 34f53a2c; committed check, status match): T8 column label eps_AE_050_superseded ("soh_min 0.50") |  |
| main.tex | 1549 | Conclusions | prose | two | unchecked |  | compound adjective | "two-stage" (the model class) |
| main.tex | 1549 | Conclusions | prose | single | unchecked |  | language | "a single, discrete investment plan" (one plan; not the instance name) |
| main.tex | 1553 | Conclusions | prose | 108 | derived | 108 | derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108 | buses of the integrated system |
| main.tex | 1553 | Conclusions | prose | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| main.tex | 1553 | Conclusions | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| main.tex | 1553 | Conclusions | prose | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].d_gross | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1553 | Conclusions | prose | 318.0 | match | 318.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].I_other |  |
| main.tex | 1553 | Conclusions | prose | 0.5 | match | 0.5 | smallest per-node E over the evaluated base-price plans other than the unit (T1 B / C / G other cells) |  |
| main.tex | 1553 | Conclusions | prose | 5 | match | 5 | largest per-node E over the same plans |  |
| main.tex | 1553 | Conclusions | prose | one | match | 1 | fewest nodes over the same plans |  |
| main.tex | 1553 | Conclusions | prose | two | match | 2 | most nodes over the same plans |  |
| main.tex | 1553 | Conclusions | prose | 31.6 | match | 31.6 | T8 aged arms: smallest loss | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1553 | Conclusions | prose | 73.7 | match | 73.7 | T8 aged arms: largest loss | sign not written; magnitude compared (the prose / column states the direction) |
| main.tex | 1553 | Conclusions | prose | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| main.tex | 1553 | Conclusions | prose | 0.50 | match | 0.50 | data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json reports.e_soh050.minimum_soh (sha256 3c71d71d) |  |
| main.tex | 1553 | Conclusions | prose | 4 | match | 4.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (dur node 7) |  |
| main.tex | 1553 | Conclusions | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| main.tex | 1553 | Conclusions | prose | 171 | approximate | 171 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[0] |  |
| main.tex | 1553 | Conclusions | prose | 193 | approximate | 193 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[1] |  |
| main.tex | 1553 | Conclusions | prose | 254 | approximate | 254 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.e_cost_eur_per_MWh |  |
| main.tex | 1553 | Conclusions | prose | 1.75 | match | 1.75 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1553 | Conclusions | prose | 2 | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1553 | Conclusions | prose | 2 | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/fig2_flexibility_ladder_data.json values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value (sha256 5eae35d9) |  |
| main.tex | 1553 | Conclusions | prose | two | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:5ca4f86c].candidate_canonical (nnodes) |  |
| main.tex | 1553 | Conclusions | prose | 0 | match | 0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].rate |  |
| main.tex | 1553 | Conclusions | prose | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].rate |  |
| main.tex | 1553 | Conclusions | prose | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| main.tex | 1553 | Conclusions | prose | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| main.tex | 1553 | Conclusions | prose | 7 | derived | 7 | derived: 1 - R(upper) = 1 - 0.933563 = 6.6437 % (T11 three_by_three.rows[7].value) | storage value lower at the 3 x 3 instance, lower end |
| main.tex | 1553 | Conclusions | prose | 9 | derived | 9 | derived: 1 - R(lower) = 1 - 0.908972 = 9.1028 % (T11 three_by_three.rows[8].value) | storage value lower at the 3 x 3 instance, upper end |
| main.tex | 1553 | Conclusions | prose | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| main.tex | 1553 | Conclusions | prose | 90.9 | derived | 90.9 | derived: benefit 90,896,608.40 EUR = 90,896.6 kEUR = 90.8966 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit) | value of coordination |
| main.tex | 1557 | Conclusions | prose | Two | unchecked |  | language | "Two implications" |
| main.tex | 1557 | Conclusions | prose | 15 | match | 15 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N4] (sha256 34f53a2c; committed check, status match): W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€), | x0: 14,970.62 EUR after the residual-rule certificate N = 132 |
| main.tex | 1557 | Conclusions | prose | 21 | match | 21 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N5] (sha256 34f53a2c; committed check, status match): W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed | unit: 20,806.32 EUR after N = 112 |
| main.tex | 1557 | Conclusions | prose | 654 | derived | 654 | derived: coordinated Q (x = 0, cycle 181) 653,873,702.19 EUR = 653,873.7 kEUR = 653.8737 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_Q) | operating cost |
| main.tex | 1561 | Conclusions | prose | one | match | 1 | W173 box_definition: one year coordinate zY |  |
| highlights.tex | 14 | highlights.tex | prose | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |

## Flag and verdict checks (tables)

- l. 1189 tab:res_value_ladder C:y2025__n5_p0.25_e0.5 (caption: "All reported differences are determinate": determinate): ok
- l. 1190 tab:res_value_ladder B:n5_4h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1191 tab:res_value_ladder B:n7_4h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1192 tab:res_value_ladder B:n7_4h_e2 (caption: "All reported differences are determinate": determinate): ok
- l. 1193 tab:res_value_ladder B:n7_4h_e3 (caption: "All reported differences are determinate": determinate): ok
- l. 1194 tab:res_value_ladder B:n7_4h_e4 (caption: "All reported differences are determinate": determinate): ok
- l. 1195 tab:res_value_ladder B:n7_4h_e5 (caption: "All reported differences are determinate": determinate): ok
- l. 1196 tab:res_value_ladder B:n7_2h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1197 tab:res_value_ladder B:n7_2h_e2 (caption: "All reported differences are determinate": determinate): ok
- l. 1198 tab:res_value_ladder B:n7_2h_e3 (caption: "All reported differences are determinate": determinate): ok
- l. 1199 tab:res_value_ladder B:n7_2h_e4 (caption: "All reported differences are determinate": determinate): ok
- l. 1200 tab:res_value_ladder B:n7_2h_e5 (caption: "All reported differences are determinate": determinate): ok
- l. 1201 tab:res_value_ladder B:n9_4h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1202 tab:res_value_ladder B:n9_4h_e3 (caption: "All reported differences are determinate": determinate): ok
- l. 1203 tab:res_value_ladder C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5 (caption: "All reported differences are determinate": determinate): ok
- l. 1204 tab:res_value_ladder C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5 (caption: "All reported differences are determinate": determinate): ok
- l. 1275 tab:res_flex_ladder CHECK:headline_V_minus_I_settled: ok
- l. 1276 tab:res_flex_ladder H:m1.5:value_minus_I: ok
- l. 1277 tab:res_flex_ladder H:m1.75:value_minus_I: ok
- l. 1278 tab:res_flex_ladder H:m2:value_minus_I: ok
- l. 1297 tab:res_energy_ladder_m2 H:m2:value_minus_I / None (caption: "All differences are determinate"): ok
- l. 1298 tab:res_energy_ladder_m2 I:m2:e2_value_minus_I / I:m2:second_MWh (caption: "All differences are determinate"): ok
- l. 1299 tab:res_energy_ladder_m2 J:e3_value_minus_I / J:e2_to_e3 (caption: "All differences are determinate"): ok
- l. 1300 tab:res_energy_ladder_m2 J:e4_value_minus_I / J:e3_to_e4 (caption: "All differences are determinate"): ok
- l. 1301 tab:res_energy_ladder_m2 J:e5_value_minus_I / J:e4_to_e5 (caption: "All differences are determinate"): ok
- l. 1339 tab:res_ageing C2_calfade (determinate; floor 2035): ok
- l. 1340 tab:res_ageing C2 (determinate; floor never): ok
- l. 1341 tab:res_ageing C3_unit (determinate; floor 2035): ok
- l. 1342 tab:res_ageing C3_midblock (determinate; floor 2035): ok
- l. 1343 tab:res_ageing C4 (determinate; floor never): ok
- l. 1344 tab:res_ageing no_ageing (within resolution; floor n/a): ok
- l. 1346 tab:res_ageing 0.50 floor (determinate; floor never binds): ok
- l. 1451 tab:res_year_ladder caption: "determinate gross ... within resolution net": ok
- l. 1472 tab:res_discount caption: "every difference is determinate": ok

## Integrity

checks: 19/19 True; failed []; guards ok True; exit 0

