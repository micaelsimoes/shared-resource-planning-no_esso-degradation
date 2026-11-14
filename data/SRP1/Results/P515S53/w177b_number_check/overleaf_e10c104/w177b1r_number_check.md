# W177b-1' -- Parts 0 and 1 at Overleaf e10c104 (main.tex sha256 1b532035), section 3 opening, letter

Zero solves; script `p515_s53_w177b1r_number_check.py` (imports `p515_s53_w177b_number_check.py`, not edited); frozen tables `data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json` (sha256 590088fe); clone `manuscript/6a67305f25e8348fb71380c3` (read only). Objective convention: Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR

## Predictions (expert, Addendum 76)

- **Part 1: zero MISMATCH in numbers carried from the tables** -- FAILED as worded: 1 MISMATCH among 721 in-scope tokens; 0 among the 388 non-excluded tokens in table bodies / captions / notes / figure captions; 0 against a frozen-table field, the rest against a named record -- l. 1520 'two' -> 1 (data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d))
- **Part 1: the derived numbers hold** -- HELD: 16/16 derived tokens (Part 1 scopes) hold at the written precision; named: 24 {'line': 1259, 'holds': True}, 33 {'line': 1259, 'holds': True}, 7 [{'line': 1443, 'holds': True}, {'line': 1585, 'holds': True}], 9 [{'line': 1443, 'holds': True}, {'line': 1585, 'holds': True}], 90.9 {'line': 1585, 'holds': True}, 654 {'line': 1589, 'holds': True}, 9.3 {'line': 1352, 'holds': True}, 42.1 {'line': 1352, 'holds': True}, 28.5 {'line': 1352, 'holds': True}; all scopes incl. section 3 opening and the letter: 26/26 hold

## Part 0 -- W177a findings at this HEAD

| id | W177a finding | status | where | evidence |
|---|---|---|---|---|
| Blocked 1 / C1 | eq:salvage_credit needs two corrections (energy cost only; health factor (SoH - SoH^Min)/(1 - SoH^Min)); items (a)-(d) hold | still present | l. 1183-1196 | {"equation_line": 1189, "cohort_cost_with_power_term": [{"line": 1197, "text": "where $I_{e,y^\\text{Inv}} = \\sum_{c \\in \\Omega_C} \\omega_c \\big( c^{\\text{Inv,S}}_{y^\\text{Inv},c} S^{\\text{Inv}}_{e,y^\\text{Inv}} + c^{\\text{Inv,E}}_{y^\\text{Inv},c} E^{\\text{Inv}}_{e,y^\\text{Inv}} \\big)$ is the expected undiscounted cost of the cohort installed at node $e$ in year $y^\\text{Inv}$, "}], "factor_I_over_discount": [{"line": 1186, "text": "\\frac{I_{e,y^\\text{Inv}}}{(1+r)^{\\,y^\\text{End}-y_0}}"}], "factor_SoH_unnormalized": [{"line": 1188, "text": "\\; SoH_{e,y^\\text{Inv},y^\\text{End}},"}], "block_equals_frame_lines": "main.tex l. 1182-1200 = frame l. 27-45: True", "recomputed_C |
| Blocked 2 / C4 | the F2 plan misdescribed: node 7 is 1 MVA / 3.5 MWh, not 3 MWh (frame l. 131) | no longer applicable | l. 1316 | {"plan_sizes_printed_in_4.2": [], "F2_sentence": [{"line": 1316, "quote": "1.0--51.8~k\\euro{}. The planning search under this price assumption returned a two-node investment plan installed in 2030. Its operational evaluation remained uncertified because of the consensus gap, and only seventeen of its sixty-one adm"}], "note": "the text names the plan as \"a two-node investment plan installed in 2030\" without sizes; the two nodes and 2030 match the frozen candidate of ref:5ca4f86c (Part 1, l. 1316)", "recomputed_C4": {"outcome": "differs", "summary": "recorded: 2030, node 5 0.25 MVA / 0.5 MWh, node 7 1 MVA / 3.5 MWh; W173 label y2030__n5_p0.25_e0.5__n7_p1_e3.5 (z [1, 1, 4, 7, 0, 0, 1], eval |
| Blocked 3 | Addendum 64 ruling 4 (J 4 MWh at its uncertified bar) vs tab:res_energy_ladder_m2: the 4 MWh row prints the flagged certified threshold 9.7 / 18.14 x; the uncertified bar 15.9 / 11.12 x is beside it | still present | l. 1332, note l. 1341 | {"row": [{"line": 1332, "text": "4$^{f}$   & 176.6 & 9.7  & 18.14 & 42.8 & 27.7 & 1.55 \\\\"}], "note": [{"line": 1341, "quote": "\\textit{Note:} Treating the flagged evaluation as uncertified gives a comparison bar of 15.9~k\\euro{} ($11.12\\times$) for its investment margin. The increments to 4 and 5~MWh have bars of 27.7 and 26.5~k\\euro{} ($1.55\\times$ and $1.95\\"}], "main_columns_certified_threshold": true, "uncertified_bar_in_note": true, "remark": "the uncertified bar is in the table note (W177b-1: moved there from the frame caption); no verdict depends on the choice"} |
| Blocked 4 | two of three Addendum 73 predictions failed (scoring record) | no longer applicable |  | {"note": "a scoring record, not manuscript text; scored in Addendum 74/76 by the Planner"} |
| Blocked 5 | literal "Section~4.x" references (l. 377, 415, 445, 631, 635, 671, 1066, 1119 at 8b76423) | fixed |  | {"literal_section_4_references": [], "refs_to_section_4_labels_outside_section_4": [{"line": 391, "refs": ["sec:results"]}, {"line": 698, "refs": ["subsec:res_certification"]}, {"line": 702, "refs": ["subsec:res_certification"]}, {"line": 738, "refs": ["subsec:res_degradation"]}, {"line": 957, "refs": ["sec:results", "subsec:res_multiscenario"]}, {"line": 990, "refs": ["sec:results"]}, {"line": 1129, "refs": ["subsec:res_degradation"]}, {"line": 1168, "refs": ["subsec:res_coordination"]}, {"line": 2220, "refs": ["sec:results"]}], "any_literal_numbered_pointer_in_main_tex (wider search, this run)": [{"line": 985, "text": "Three ESS investment-cost trajectories are considered at the planning l |
| Blocked 6 (l. 58) | "every evaluated plan ... 4.39 x to 45.94 x" spans the B rows only (table range 3.94-45.94 x) | no longer applicable |  | {"range_in_4.1_prose": [], "note": "no multiple range is stated in the 4.1 prose; the table prints every row"} |
| Blocked 6 (l. 63) | "Two-node plans lose more than their parts would" is not resolved by the records | fixed | l. 1251 | {"old_claim_found": [], "new_text": [{"line": 1251, "quote": "ed investment cost. The evaluated two-node plans likewise remain uneconomic: the cost excess is 102.1~k\\euro{} for nodes~5 and~9 in 2025 and 93.9~k\\euro{} for nodes~5 and~7 in 2030. Applying the terminal salvage credit reduces the latter to"}]} |
| Blocked 6 (l. 65) | "identical d net, since all cohorts reach y^End with the same remaining life" (the reason is that the 2025 cohort's remaining life is zero) | no longer applicable |  | {"old_claim_found": [], "note": "not written; tab:res_year_ladder prints the 2025 salvage as 0.0"} |
| Blocked 6 (l. 381) | "Deferring the reference unit changes little gross and nothing net" contradicts the figures | fixed | l. 1475 | {"old_claim_found": [], "new_text": [{"line": 1475, "quote": "Deferring the reference unit does not make it pay (Table~\\ref{tab:res_year_ladder}). Against the plan without storage, the unit installed in 2030 increases the total cost by 68.5~k\\euro{} and the unit installed in 2035 by 111.0~k\\euro{}. A later installation shortens the part of the horizon over which the unit operates, while the terminal salvage credit returns its remaining calendar life: 11.8~k\\"}], "note": "\"net of salvage, the unit costs 54.5--64.4 kEUR more than the plan without storage in every investment year\" is checked in Part 1 (derived)"} |
| Blocked 6 (l. 484) | 4.9 (i) "with the datasheet calibration and the 0.70 floor" ambiguous (baseline vs C4 row) | no longer applicable |  | {"found_now": [], "main_tex_lines_removed_6ff8f3c_to_e10c104": 118, "removed_key_insights_heading": true, "removed_datasheet_clause": true, "note": "section 4.9 Key Insights was deleted at 51a6fb5 (W177b-1) and is still absent at e10c104"} |
| Blocked 7 / C5 | optional fill of the 0.50-floor row: value 258.5 kEUR, AE 0.778, EFC/day 1.08, floor never (W155, not in the frozen T1-T11) | still open (row not filled; author's option) | l. 1378 | {"row": [{"line": 1378, "text": "Baseline, 0.50 floor & --- & $-$59.5 & 12.6 & 4.71 & never & --- & --- \\\\"}], "cells": ["Baseline, 0.50 floor", "---", "$-$59.5", "12.6", "4.71", "never", "---", "---"], "W177a_values": {"Value (kEUR)": "258.5", "AE": "0.778", "EFC/day": "1.08", "Floor binds": "never"}, "note": "value, AE and EFC/day print \"---\"; value - I, threshold, multiple and \"never\" are checked in Part 1"} |
| C2 | S / E / duration of each B and C cell confirmed (16/16) | holds | l. 1221-1236 | {"plan_tokens_rows": {"lines": [1221, 1236], "n": 67, "match": 67}, "recomputed_C2": "16 of 16 rows: node, S, E, E/S and year equal the frozen candidate; evaluation records found for 16, equal to the frozen candidate, candidate key and eval key for 16"} |
| C3 | I/J cells = node-7 4 h plans of 2-5 MWh (keys confirmed) | holds | l. 1322-1333 | {"plan_tokens_caption_and_rows": {"lines": [1322, 1333], "n": 8, "match": 8}, "recomputed_C3": "i_5a6a88b4 cc4eb5df = d_c52e1670 ([0.5, 2.0] at node 7, 2025); j_f3aa335e fb21d822 = d_4a82a64a ([0.75, 3.0] at node 7, 2025); j_a11d7966 11725e45 = d_3632b0ae ([1.0, 4.0] at node 7, 2025); j_5f3cccb4 cab98853 = d_36686489 ([1.25, 5.0] at node 7, 2025)"} |
| C6 | the 3 x 3 "16 cycles" confirmed; wording: the evaluation, not the fit, was continued | fixed | l. 1443, l. 1449 | {"16_cycles_printed": [], "prose": [{"line": 1443, "quote": "stopping criterion, the evaluation of the plan without storage was continued beyond its stopping point. A subsequent fit to the objective trajectory estimated a further cost decrease of 6.2~k\\euro{}. Accounting for this decrease changes the ratio of the reference unit's multi-sce"}], "caption": [{"line": 1449, "quote": "he production exit; $D$ is the post-hoc settled descent of the $\\boldsymbol{x} = \\boldsymbol{0}$ evaluation continued past it. Gross operational cost, settlement excluded.}}"}], "recomputed_C6": "W98 r2 continuation: replay bitwise through N = 72 (the production exit, cycle 72), continuation record cycles 1-88 -> |
| C7 | W153: 0 sign changes in 60 claims; ONE verdict changes category (the F2 neighbour) | changed (the text says two verdicts) | l. 1520 | {"text": [{"line": 1520, "quote": "Across the 60 pairwise comparisons reported in this section, applying the salvage credit changes no sign. It changes two verdicts: the investment-year comparison above, and the comparison of the plan returned at twice the base flexibility price with one of its unit neighbours, which is within the uncertified bar in gross terms (0.95 times) and determinate net of salvage (1.22 times), in the returned plan's favour."}], "recomputed_C7": "W153 totals: n_claims 60, n_sign_differs 0, n_verdict_differs 1 (['L:y2030__n5_p0.25_e0.5__n7_p1_e3']: within the uncertified bar gross -> determinate net); frozen T1: 60 claims, 0 sign changes; the claim at 0.9472 x gross (w |
| C8 | the 2025 cohort's salvage is zero (the row may print 0.0) | holds | l. 1490 | {"row": [{"line": 1490, "text": "2025 (reference) & 64.4  & 0.0  & 64.4 \\\\"}], "recomputed_C8": "code: age_at_terminal(2025) = 15, T_rem = max(15 - 15, 0) = 0 -> remaining-life fraction 0, the 2025 term is exactly 0; records: 30 cells with a 2025 plan, max /salvage/ 5.76e-33 EUR (numerical residue of the inactive cohorts' terms, not the 2025 cohort), ref:bd504ecf salvage 0.0; T1: d_net = d_gros"} |
| C9 | 41 new recourse evaluations (14, 20, 7 against 20, 20, 60) confirmed | holds | l. 1548 | {"text": [{"line": 1548, "quote": "_ess_planning_mads} made 41 new recourse evaluations in all (14, 20 and 7 against budgets of 20, 20 and 60), each stopped by the production"}], "recomputed_C9": "W173 item1: s47 14/20 (mesh_local_optimum_unit_poll_failed), s51 20/20 (STOP_FOR_REVIEW_completion_cap), s53 7/60 (poll_failure_at_unit_mesh); sum 41"} |

### [CONFIRM -- W177] comments in main.tex

- l. 1198 (salvage equation items (a)-(d)): items (a)-(d) hold as worded (recomputed C1: (a) confirmed, (b) confirmed, (c) confirmed, (d) confirmed); the equation above them still needs the two corrections (Blocked 1)
- l. 1477 (G rows of T1 (node 7, 0.5 MVA / 1 and 2 MWh, 2030 / 2035) gross / net, all determinate): confirmed: 17/17 tokens of l. 1477-1478 match the frozen T1 G claims; every G claim determinate gross and net (W153 and recomputed): True
- l. 1522 (W153 totals; the neighbour at 0.95 x gross / 1.22 x net; the 2035 - 2030 pair): 60 claims, 0 sign changes, 1 verdict change (['L:y2030__n5_p0.25_e0.5__n7_p1_e3']) -- recomputed C7 confirmed; the 2035 - 2030 investment-year pair is NOT among the 60 (no W153 claim has a yl_y2030 / yl_y2035 cell; it is T4 year_ladder): gross determinate 3.46 x -> net within resolution 0.19 x, sign + -> -
- l. 1550 (W173 realized use 14/20, 20/20, 7/60; 41 is their sum): recomputed C9 confirmed: W173 item1: s47 14/20 (mesh_local_optimum_unit_poll_failed), s51 20/20 (STOP_FOR_REVIEW_completion_cap), s53 7/60 (poll_failure_at_unit_mesh); sum 41

W177a CONFIRM outcomes recomputed now vs committed: C1 = (differs), C2 = (confirmed), C3 = (confirmed), C4 = (differs), C5 = (confirmed (values available)), C6 = (confirmed), C7 = (confirmed), C8 = (confirmed), C9 = (confirmed)

## Part 1 -- counts by scope (main.tex, highlights.tex)

| scope | lines | match | approximate | derived | MISMATCH | no counterpart | unchecked | excluded | total |
|---|---|---|---|---|---|---|---|---|---|
| abstract | 140-150 | 4 | 1 | 1 | 0 | 0 | 1 | 1 | 8 |
| highlights (main.tex block) | 162-168 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| highlights.tex | 1-17 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 3 |
| 4 preamble (reported separately) | 1176-1203 | 5 | 0 | 0 | 0 | 0 | 7 | 1 | 13 |
| 4.1 | 1204-1254 | 150 | 0 | 0 | 0 | 0 | 4 | 1 | 155 |
| 4.2 | 1255-1346 | 126 | 0 | 3 | 0 | 0 | 0 | 5 | 134 |
| 4.3 | 1347-1395 | 79 | 0 | 3 | 0 | 0 | 1 | 2 | 85 |
| 4.4 | 1396-1434 | 44 | 3 | 0 | 0 | 0 | 0 | 2 | 49 |
| 4.5 | 1435-1470 | 33 | 0 | 2 | 0 | 0 | 12 | 1 | 48 |
| 4.6 | 1471-1527 | 106 | 0 | 2 | 1 | 0 | 5 | 13 | 127 |
| 4.7 | 1528-1558 | 92 | 0 | 0 | 0 | 0 | 9 | 5 | 106 |
| 4.8 (reported separately) | 1559-1573 | 31 | 4 | 0 | 0 | 0 | 2 | 0 | 37 |
| Conclusions | 1577-1596 | 26 | 3 | 5 | 0 | 0 | 3 | 0 | 37 |

In scope: {'excluded': 32, 'unchecked': 35, 'derived': 16, 'match': 662, 'approximate': 7, 'MISMATCH': 1}; reported separately: {'unchecked': 9, 'match': 36, 'excluded': 1, 'approximate': 4}

### Lines changed since 51a6fb5 inside the Part 1 scopes

- replace 51a6fb5 l. 139-142 -> e10c104 l. 141-149 (abstract): tokens removed ['0.91', '0.93', '1.75', '171', '193', '254', '3', '3', 'single'], added [], kept ['0.25', '1', '108', '13.9', '15', '64', 'two']
- replace 51a6fb5 l. 1561-1561 -> e10c104 l. 1593-1594 (Conclusions): tokens removed [], added [], kept ['one']

### New or changed tokens (in scope)

- main.tex l. 145:158 (abstract, prose) `two` -- unchecked: compound adjective
- main.tex l. 146:53 (abstract, prose) `108` -- derived: derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108
- main.tex l. 146:107 (abstract, prose) `15` -- match: SRP1.json Years: sum of the block lengths
- main.tex l. 146:216 (abstract, prose) `13.9` -- match: data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative
- main.tex l. 146:391 (abstract, prose) `0.25` -- match: data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7)
- main.tex l. 146:400 (abstract, prose) `1` -- match: data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7)
- main.tex l. 146:450 (abstract, prose) `64` -- approximate: data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[B:n7_4h_e1].d_gross
- main.tex l. 1593:36 (Conclusions, prose) `one` -- match: W173 box_definition: one year coordinate zY

### MISMATCH (all scopes)

- main.tex l. 1520:122 (4.6, prose) `two` -- MISMATCH: data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d) [1] as written ("Across the 60 pairwise comparisons ... It changes two verdicts") the count is of the 60: W153 records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3). The second verdict the sentence lists, the 2035 - 2030 investment-year comparison, is NOT one of the 60 (T4 year_ladder, outside T1 / W153): its gross verdict determinate (3.46 x) becomes within resolution net (0.19 x) and its sign changes (D_gross +42,458.65 EUR, D_net -2,286.25 EUR). "Two" holds for the 60 + 1 comparisons, not for the 60; the "changes no sign" clause then fails for that pair (within resolution net, so its net sign is indeterminate). Part 2 (c), W177b-2
- response_to_reviewers_draft.tex l. 221:55 (letter, response) `two` -- MISMATCH: data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d) [1] as written ("It changes no sign in the 60 reported comparisons and two verdicts") the count is of the 60: W153 records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3, 0.95 x gross -> 1.22 x net). The second verdict the sentence names, the 2035 - 2030 investment-year comparison, is NOT one of the 60 (T4 year_ladder): determinate gross (3.46 x) -> within resolution net (0.19 x), and its sign changes (+42,458.65 -> -2,286.25 EUR). The same count as main.tex section 4.6 (W177b-1 l. 1488; here l. 1520)

### DERIVED (arithmetic, all scopes)

- main.tex l. 146:53 (abstract) `108` -- derived (holds True): derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108
- main.tex l. 957:575 (3 opening) `nine` -- derived (holds True): derived: SRP1__s53_3x3.json NumMarketScenarios 3 x num_operation_scenarios 3 (TN and every DN) = 9; instance_record.json derive_overrides {'num_market_scenarios': 3, 'num_operation_scenarios': 3}
- main.tex l. 1259:905 (4.2) `24` -- derived (holds True): derived: 1 - e*/e_cost = 1 - 192,557.46 / 253,877.68 = 61,320.21 / 253,877.68 = 24.1534 % (T3 break_even.conservative_A64.breakeven_range[1], margin_to_cost_range[0], e_cost_eur_per_MWh)
- main.tex l. 1259:909 (4.2) `33` -- derived (holds True): derived: 1 - e*/e_cost = 1 - 171,309.08 / 253,877.68 = 82,568.60 / 253,877.68 = 32.5230 % (T3 break_even.conservative_A64.breakeven_range[0], margin_to_cost_range[1], e_cost_eur_per_MWh)
- main.tex l. 1316:814 (4.2) `six` -- derived (holds True): derived: W173 F2 L neighbours 13 - determinate 7 = 6 (= within the bar, plan better 5 + neighbour better within the bar 1); frozen T1 L box-neighbour rows 13, determinate and positive 7, the rest 6
- main.tex l. 1352:126 (4.3) `9.3` -- derived (holds True): derived: T8 value(C2_calfade) 253,539.62 - value(C3_unit) 244,210.35 = 9,329.27 EUR = T1 E:n7_4h_e1_C2_calfade:vs_C3 d_gross 9,329.27 EUR (within resolution, 0.71 x); from the printed T8 values 253.5 - 244.2 = 9.3
- main.tex l. 1352:255 (4.3) `42.1` -- derived (holds True): derived: T8 value(C2) 286,349.77 - value(C3_unit) 244,210.35 = 42,139.42 EUR = T1 E:n7_4h_e1_C2:vs_C3 d_gross 42,139.42 EUR (determinate, 3.28 x); from the printed T8 values 286.3 - 244.2 = 42.1
- main.tex l. 1352:264 (4.3) `28.5` -- derived (holds True): derived: T8 value(C4) 272,760.31 - value(C3_unit) 244,210.35 = 28,549.96 EUR = T1 E:n7_4h_e1_C4:vs_C3 d_gross 28,549.96 EUR (determinate, 2.22 x); from the printed T8 values 272.8 - 244.2 = 28.6
- main.tex l. 1443:742 (4.5) `7` -- derived (holds True): derived: 1 - R(upper) = 1 - 0.933563 = 6.6437 % (T11 three_by_three.rows[7].value)
- main.tex l. 1443:745 (4.5) `9` -- derived (holds True): derived: 1 - R(lower) = 1 - 0.908972 = 9.1028 % (T11 three_by_three.rows[8].value)
- main.tex l. 1475:821 (4.6) `54.5` -- derived (holds True): derived: min of M net over the investment years {2025: 64,417.39 (T1 CHECK:headline_V_minus_I_settled d_net, sign M = -(value - I)), 2030: 56,739.66, 2035: 54,453.40} EUR (T4) = 54,453.40 EUR
- main.tex l. 1475:827 (4.6) `64.4` -- derived (holds True): derived: max of M net over the investment years {2025: 64,417.39 (T1 CHECK:headline_V_minus_I_settled d_net, sign M = -(value - I)), 2030: 56,739.66, 2035: 54,453.40} EUR (T4) = 64,417.39 EUR
- main.tex l. 1585:22 (Conclusions) `108` -- derived (holds True): derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108
- main.tex l. 1585:973 (Conclusions) `7` -- derived (holds True): derived: 1 - R(upper) = 1 - 0.933563 = 6.6437 % (T11 three_by_three.rows[7].value)
- main.tex l. 1585:976 (Conclusions) `9` -- derived (holds True): derived: 1 - R(lower) = 1 - 0.908972 = 9.1028 % (T11 three_by_three.rows[8].value)
- main.tex l. 1585:1050 (Conclusions) `90.9` -- derived (holds True): derived: benefit 90,896,608.40 EUR = 90,896.6 kEUR = 90.8966 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit)
- main.tex l. 1589:615 (Conclusions) `654` -- derived (holds True): derived: coordinated Q (x = 0, cycle 181) 653,873,702.19 EUR = 653,873.7 kEUR = 653.8737 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_Q)
- response_to_reviewers_draft.tex l. 53:40 (letter) `90.9` -- derived (holds True): derived: benefit 90,896,608.40 EUR = 90,896.6 kEUR = 90.8966 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit)
- response_to_reviewers_draft.tex l. 78:325 (letter) `108` -- derived (holds True): derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108
- response_to_reviewers_draft.tex l. 125:692 (letter) `48` -- derived (holds True): derived: SRP1.json Years 3 x Days 4 x network operators (1 TN + 3 DN) 4 = 48
- response_to_reviewers_draft.tex l. 182:86 (letter) `7` -- derived (holds True): derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.1109/ISGTEUROPE62998.2024.10863557', '10.1016/j.apenergy.2022.120569'] -> bib keys ['aghdam_2023', 'aghdam_2024', 'aghdam_2025'] (bibliography.bib doi fields); "rented to renewable producers" cites ['aghdam_2025'], "congestion management" cites ['aghdam_2023', 'aghdam_2024']; static order, not a compile
- response_to_reviewers_draft.tex l. 182:91 (letter) `9` -- derived (holds True): derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.1109/ISGTEUROPE62998.2024.10863557', '10.1016/j.apenergy.2022.120569'] -> bib keys ['aghdam_2023', 'aghdam_2024', 'aghdam_2025'] (bibliography.bib doi fields); "rented to renewable producers" cites ['aghdam_2025'], "congestion management" cites ['aghdam_2023', 'aghdam_2024']; static order, not a compile
- response_to_reviewers_draft.tex l. 182:420 (letter) `7` -- derived (holds True): derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.1109/ISGTEUROPE62998.2024.10863557', '10.1016/j.apenergy.2022.120569'] -> bib keys ['aghdam_2023', 'aghdam_2024', 'aghdam_2025'] (bibliography.bib doi fields); "rented to renewable producers" cites ['aghdam_2025'], "congestion management" cites ['aghdam_2023', 'aghdam_2024']; static order, not a compile
- response_to_reviewers_draft.tex l. 182:517 (letter) `8` -- derived (holds True): derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.1109/ISGTEUROPE62998.2024.10863557', '10.1016/j.apenergy.2022.120569'] -> bib keys ['aghdam_2023', 'aghdam_2024', 'aghdam_2025'] (bibliography.bib doi fields); "rented to renewable producers" cites ['aghdam_2025'], "congestion management" cites ['aghdam_2023', 'aghdam_2024']; static order, not a compile
- response_to_reviewers_draft.tex l. 182:520 (letter) `9` -- derived (holds True): derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.1109/ISGTEUROPE62998.2024.10863557', '10.1016/j.apenergy.2022.120569'] -> bib keys ['aghdam_2023', 'aghdam_2024', 'aghdam_2025'] (bibliography.bib doi fields); "rented to renewable producers" cites ['aghdam_2025'], "congestion management" cites ['aghdam_2023', 'aghdam_2024']; static order, not a compile
- response_to_reviewers_draft.tex l. 287:189 (letter) `1.25` -- derived (holds True): derived: W167 question4_cost_file.json: submitted (predecessor 072b1310) energy formula divisor 5 / corrected (7ce1d1ab) divisor 4 = 1.25

## Amendment (1) -- section 3 opening

Scope l. 951-992; counts {'match': 33, 'unchecked': 6, 'derived': 1, 'excluded': 4}; M1 texts at the HEAD: {'M1 (a)': True, 'M1 (b)': True, 'M1 (c)': True, 'M2 (anywhere in main.tex, outside comments)': False, 'M3 (anywhere in main.tex, outside comments)': True}; submitted sentences gone: {'represented by five years': True, 'five load/RES scenarios': True, '25 operating scenarios': True, 'used as a baseline to quantify the overall benefits': True}

| line | written | status | value | source |
|---|---|---|---|---|
| 954 | 9 | match | 9 | case9_2025.json nodes (IEEE 9-bus) |
| 954 | three | match | 3 | SRP1.json DistributionNetworks (count) |
| 954 | 33 | match | 33 | case33_1/2/3_2025.json nodes (each) |
| 954 | single | unchecked |  | compound adjective |
| 955 | three | match | 3 | SRP1.json DistributionNetworks (count) = the interface nodes |
| 957 | 15 | match | 15 | sum of the block lengths: SRP1.json Years and SRP1__s53_3x3.json Years (both 15) |
| 957 | two | match | 2 | the instances with results: SRP1.json (T1-T10) and its derived 3 x 3 instance (T11); the 3 x 3 campaign spec derived_instance names both, with their sha256 (checked against the files now) |
| 957 | single | unchecked |  | compound adjective |
| 957 | three | match | 3 | SRP1.json Years (count) |
| 957 | 2025 | match | 2025 | SRP1.json Years |
| 957 | 2030 | match | 2030 | SRP1.json Years |
| 957 | 2035 | match | 2035 | SRP1.json Years |
| 957 | five | match | 5 | SRP1.json Years: block length (each) |
| 957 | one | match | 1 | data/SRP1/SRP1.json NumMarketScenarios (sha256 61a794a7) |
| 957 | one | match | 1 | SRP1.json num_operation_scenarios (TN and every DN) |
| 957 | five | match | 5 | SRP1__s53_3x3.json Years (count) |
| 957 | 2025 | match | 2025 | SRP1__s53_3x3.json Years |
| 957 | 2028 | match | 2028 | SRP1__s53_3x3.json Years |
| 957 | 2031 | match | 2031 | SRP1__s53_3x3.json Years |
| 957 | 2034 | match | 2034 | SRP1__s53_3x3.json Years |
| 957 | 2037 | match | 2037 | SRP1__s53_3x3.json Years |
| 957 | three | match | 3 | SRP1__s53_3x3.json Years: block length (each) |
| 957 | three | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |
| 957 | three | match | 3 | SRP1__s53_3x3.json num_operation_scenarios (TN and every DN) |
| 957 | nine | derived | 9 | derived: SRP1__s53_3x3.json NumMarketScenarios 3 x num_operation_scenarios 3 (TN and every DN) = 9; instance_record.json derive_overrides {'num_market_scenarios': 3, 'num_operation_scenarios': 3} |
| 957 | four | match | 4 | Days (count) of both instances (Spring, Summer, Autumn, Winter) |
| 961 | single | match | 1 | W173 box_definition: one year coordinate zY common to all nodes |
| 961 | one | match | 1 | every frozen cell's candidate_canonical: one investment_year and one [S, E] pair per node (no staged additions) |
| 967 | Single | unchecked |  | comment (not typeset) |
| 973 | 9 | match | 9 | case9_2025.json nodes |
| 977 | 33 | match | 33 | case33_1/2/3_2025.json nodes (each) |
| 985 | Three | match | 3 | SRP1_ESS.xlsx sheet Scenarios: the investment-cost trajectories (probabilities) |
| 985 | 3.1 | unchecked |  | pointer, checked by W177b-3' |
| 987 | 1 | match | 1 | SRP1_ESS_Params.json budget (EUR) / 1e6 |
| 987 | 5.00 | match | 5.00 | data/SRP1/SharedESS/SRP1_ESS_Params.json max_capacity (sha256 39106f93) |
| 987 | 15 | match | 15 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.calendar_life_years (sha256 39106f93) |
| 987 | 2.00 | match | 2.00 | DiscountFactor of SRP1.json and SRP1__s53_3x3.json (equal) |
| 990 | Two | unchecked |  | language |
| 990 | 0 | unchecked |  | notation |
| 990 | two | match | 2 | T6: the two NRF arms (passive, price-taker) |

Case files: SRP1_JSON data/SRP1/SRP1.json sha256 61a794a7 blob 9a6d7373 (campaign spec records 61a794a7, equal True); CASE_3X3 data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json sha256 2a64e3c5 blob dd8dd781 (campaign spec records 2a64e3c5, equal True)

## Amendment (2) -- the letter

counts {'excluded': 39, 'unchecked': 162, 'match': 101, 'derived': 9, 'reviewer quotation, verified': 62, 'approximate': 1, 'MISMATCH': 1, 'no counterpart': 1}; by rule {'identifier (date)': 1, 'identifier (hash)': 1, 'structure: package/class option or argument': 4, 'structure: macro definition': 3, 'structure: macro parameter': 3, 'declared': 140, 'pointer rule (W171a XREF_RX)': 135, 'identifier (label)': 27, "reviewer quotation (W171a quotation check against the reviewers' document)": 62}; letter = response_to_reviewers_v2.tex: True

| line | part | written | status | value | source / category | note |
|---|---|---|---|---|---|---|
| 24 | prose | 1 | unchecked |  | enumerator | revision number (title block) |
| 33 | prose | three | match | 3 | manuscript_review/Reviewers Comments.docx: distinct "Reviewer #n" |  |
| 35 | prose | two | match | 2 | the letter itself: \paragraph{} blocks before \section*{Reviewer 1} |  |
| 38 | prose | 2 | unchecked |  | enumerator | reviewer number ('Reviewers 2 and 3') |
| 38 | prose | 3 | unchecked |  | enumerator | reviewer number ('Reviewers 2 and 3') |
| 40 | prose | 0.50 | match | 0.50 | manuscript_submitted/main.tex (sha256 ca07d7db) lines [866]: 'relative optimality gap of 0.50' | submitted figure (the gap withdrawn) |
| 42 | prose | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~2.1' |
| 42 | prose | 2.2 | unchecked |  | pointer, checked by W177b-3' | numbered pointer (Section / Table / Figure / Eq. / Appendix) the pointer regex does not catch |
| 43 | prose | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 45 | prose | 3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3.5)' |
| 48 | prose | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| 48 | prose | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| 48 | prose | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| 48 | prose | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[CHECK:headline_V_minus_I_settled].d_gross | "loses 64.4 kEUR" = -(value - I) of the reference unit; sign not written; magnitude compared (the prose / column states the direction) |
| 50 | prose | 4.2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.2)' |
| 51 | prose | 4.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.3)' |
| 51 | prose | 18.25 | match | 18.25 | manuscript_submitted/main.tex (sha256 ca07d7db) lines [140, 1122]: 'reduces operating costs by up to 18.25' | submitted abstract figure (withdrawn) |
| 53 | prose | 90.9 | derived | 90.9 | derived: benefit 90,896,608.40 EUR = 90,896.6 kEUR = 90.8966 MEUR (data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit) | value of coordination (MEUR) |
| 53 | prose | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| 54 | prose | 4.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.4)' |
| 58 | prose | Two | unchecked |  | language | "Two requests were not met in full" |
| 58 | prose | 1 | unchecked |  | enumerator | reviewer number ('Reviewer~1)') |
| 58 | prose | 3 | unchecked |  | enumerator | reviewer number ('Reviewer~3') |
| 58 | prose | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8)' |
| 58 | prose | five | match | five | manuscript_submitted/main.tex (sha256 ca07d7db) lines [664]: 'represented by five years' | the submitted instance: five representative years |
| 58 | prose | twenty-five | match | 25 | manuscript_submitted/main.tex (sha256 ca07d7db) lines [673]: 'resulting in 25 operating scenarios' | the submitted instance: 5 x 5 scenarios |
| 58 | prose | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |  |
| 58 | prose | 3 | match | 3 | 3 x 3 instance: operation scenarios |  |
| 58 | prose | 4.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.5)' |
| 61 | prose | 1 | unchecked |  | enumerator | reviewer number ('Reviewer 1') |
| 68 | reviewer quotation | 18.25 | reviewer quotation, verified | 18.25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 92.16 | reviewer quotation, verified | 92.16 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 18.25 | reviewer quotation, verified | 18.25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 92.16 | reviewer quotation, verified | 92.16 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 8th | reviewer quotation, verified | 8th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 10th | reviewer quotation, verified | 10th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 13th | reviewer quotation, verified | 13th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 68 | reviewer quotation | 14th | reviewer quotation, verified | 14th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 70 | reviewer quotation | one | reviewer quotation, verified | one | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 9; letter quotation 0 segment 5 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 75 | response | 3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3)' |
| 77 | response | 3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3.5)' |
| 77 | response | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| 77 | response | 3 | match | 3 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json NumMarketScenarios (sha256 2a64e3c5) |  |
| 77 | response | 3 | match | 3 | 3 x 3 instance: operation scenarios |  |
| 78 | response | five | match | 5 | SRP1__s53_3x3.json Years (count): "five-block horizon" |  |
| 78 | response | 0.909 | match | 0.909 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[8].value |  |
| 78 | response | 0.934 | match | 0.934 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[7].value |  |
| 78 | response | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| 78 | response | 0.933 | approximate | 0.933 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[9].value | R predicted from the mean-profile spread (0.9331) |
| 78 | response | 4.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.5)' |
| 78 | response | 9 | match | 9 | case9_2025.json nodes |  |
| 78 | response | three | match | 3 | SRP1.json DistributionNetworks (count) |  |
| 78 | response | 33 | match | 33 | case33_1/2/3_2025.json nodes (each) |  |
| 78 | response | 108 | derived | 108 | derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108 | buses of the integrated system |
| 79 | response | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8)' |
| 80 | response | 18.25 | match | 18.25 | manuscript_submitted/main.tex (sha256 ca07d7db) lines [140, 1122]: 'reduces operating costs by up to 18.25' | submitted figure (withdrawn) |
| 80 | response | 92.16 | match | 92.16 | manuscript_submitted/main.tex (sha256 ca07d7db) lines [140, 1122]: 'RES curtailment by up to 92.16' | submitted figure (withdrawn) |
| 80 | response | 13.9 | match | 13.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.benefit_relative |  |
| 80 | response | 15 | match | 15 | SRP1.json Years: sum of the block lengths |  |
| 81 | response | three | match | 3 | SRP1.json Years (count) |  |
| 81 | response | five | match | 5 | SRP1.json Years: block length (each) |  |
| 81 | response | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| 81 | response | five | match | 5 | SRP1__s53_3x3.json Years (count) |  |
| 81 | response | three | match | 3 | SRP1__s53_3x3.json Years: block length |  |
| 81 | response | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 81 | response | 4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4' |
| 81 | response | 7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Tables~7--15)' |
| 81 | response | 15 | unchecked |  | pointer, checked by W177b-3' | pointer 'Tables~7--15)' |
| 82 | response | 2.3.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.4' |
| 82 | response | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8)' |
| 82 | response | 5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~5)' |
| 85 | changes | 3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3' |
| 85 | changes | 4.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.5' |
| 86 | changes | 2.3.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.4' |
| 86 | changes | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8' |
| 86 | changes | 5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~5' |
| 86 | changes | 7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Tables~7--15' |
| 86 | changes | 15 | unchecked |  | pointer, checked by W177b-3' | pointer 'Tables~7--15' |
| 97 | response | 0 | match | 0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].rate |  |
| 97 | response | 2 | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].rate |  |
| 97 | response | 5 | match | 5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].rate |  |
| 97 | response | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].rate |  |
| 97 | response | -40.8 | match | -40.8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.0].value_minus_I |  |
| 97 | response | -64.4 | match | -64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.02].value_minus_I |  |
| 97 | response | -92.7 | match | -92.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.05].value_minus_I |  |
| 97 | response | -114.6 | match | -114.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.discount[0.08].value_minus_I |  |
| 97 | response | 15 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~15)' |
| 97 | response | 2 | match | 2 | data/SRP1/SRP1.json DiscountFactor (sha256 61a794a7) |  |
| 98 | changes | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~1' |
| 98 | changes | 4.6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.6' |
| 98 | changes | 15 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~15' |
| 100 | prose | 2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure 2' |
| 101 | reviewer quotation | 2 | reviewer quotation, verified | 2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 2 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 102 | response | 2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~2' |
| 102 | response | 3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~3' |
| 102 | response | two | match | 2 | main.tex at the HEAD: \subfloat panels of fig:tso_dso_coordination_cs1_network_diagram |  |
| 102 | response | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~1)' |
| 103 | changes | 3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~3' |
| 103 | changes | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~1' |
| 106 | prose | 2 | unchecked |  | enumerator | reviewer number ('Reviewer 2') |
| 110 | reviewer quotation | two | reviewer quotation, verified | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 110 | reviewer quotation | two | reviewer quotation, verified | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 110 | reviewer quotation | two | reviewer quotation, verified | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 110 | reviewer quotation | single | reviewer quotation, verified | single | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 111 | reviewer quotation | one | reviewer quotation, verified | one | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 3 segment 1 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 113 | response | two | unchecked |  | compound adjective | "two-stage" |
| 114 | response | one | unchecked |  | language | "selects one scenario-independent multi-year plan" (one plan) |
| 116 | response | single | unchecked |  | language | "a single vector" (article sense) |
| 116 | response | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.1)' |
| 118 | response | single | unchecked |  | language | "that single plan" (article sense) |
| 121 | changes | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.1' |
| 121 | changes | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~1, 2.1, 2.2, 5' |
| 121 | changes | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~1, 2.1, 2.2, 5' |
| 121 | changes | 2.2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~1, 2.1, 2.2, 5' |
| 121 | changes | 5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~1, 2.1, 2.2, 5' |
| 125 | response | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Algorithm~1' |
| 125 | response | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.1' |
| 125 | response | 0.25 | match | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP (MVA) |  |
| 125 | response | 0.5 | match | 0.5 | p515_s45_a1_campaign.py LATTICE_E_STEP (MWh) |  |
| 125 | response | 2 | match | 2 | data/SRP1/SharedESS/SRP1_ESS_Params.json min_energy_to_power_factor (sha256 39106f93) |  |
| 125 | response | 4 | match | 4 | data/SRP1/SharedESS/SRP1_ESS_Params.json max_energy_to_power_factor (sha256 39106f93) |  |
| 125 | response | 2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~2' |
| 125 | response | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eq.~(11)' |
| 125 | response | One | unchecked |  | language | "One recourse evaluation is made per candidate plan" (method statement) |
| 125 | response | one | unchecked |  | language | "one local problem per network block" (method statement) |
| 125 | response | three | match | 3 | SRP1.json Years (count) |  |
| 125 | response | four | match | 4 | SRP1.json Days (count) |  |
| 125 | response | four | match | 4 | SRP1.json: 1 TN + the DistributionNetworks |  |
| 125 | response | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| 125 | response | 48 | derived | 48 | derived: SRP1.json Years 3 x Days 4 x network operators (1 TN + 3 DN) 4 = 48 | network blocks per ADMM cycle at the single-scenario instance |
| 125 | response | one | unchecked |  | language | "one per shared-ESS agent" (method statement) |
| 125 | response | three | match | 3 | SRP1.json DistributionNetworks connection_node_id (distinct) = the interface nodes = the shared-ESS agents |  |
| 125 | response | one | unchecked |  | language | "one per interface node" (method statement) |
| 125 | response | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 125 | response | 3.6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3.6)' |
| 125 | response | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7)' |
| 126 | changes | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.1' |
| 126 | changes | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Algorithm~1' |
| 126 | changes | 2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~2' |
| 126 | changes | 2.2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2' |
| 126 | changes | 3.6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3.6' |
| 126 | changes | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7' |
| 135 | response | 2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~2.1' |
| 135 | response | 2.2 | unchecked |  | pointer, checked by W177b-3' | numbered pointer (Section / Table / Figure / Eq. / Appendix) the pointer regex does not catch |
| 136 | response | 2.3.2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.2)' |
| 137 | response | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7)' |
| 141 | changes | 2.2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2' |
| 141 | changes | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | the submitted version's section number ("former 2.2.7 Benders' cuts") |
| 142 | changes | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | "new 2.2.7" (the revised section number) |
| 143 | changes | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7' |
| 153 | response | three | match | 3 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N7] (sha256 34f53a2c; committed check, status match): settling_criterion_v2.SettlingRuleV2.evaluate | at_least_3_turning_points |
| 155 | response | one | match | 1 | W163 N9 (W_MIN 20) and N10 (W_FACTOR 1.1), both committed checks status match: W = max(20, ceil(1.1 P_hat)) >= 1.1 P_hat > P_hat, i.e. a window at least one measured period long (as W176b) |  |
| 156 | response | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 157 | response | 42 | match | 42 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_cells (sha256 ff037985) |  |
| 157 | response | 32 | match | 32 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_certified (sha256 ff037985) |  |
| 158 | response | 10 | match | 10 | data/SRP1/Results/P515S53/w153_step5_rows/w153_certification_statistics.json result.totals.n_uncertified (sha256 ff037985) |  |
| 159 | response | 0.9 | match | max 0.882 τ | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N25] (sha256 34f53a2c; committed check, status match): records: Q(last) − Q(k*) of the runs that co | largest movement past a settling-rule certificate: 0.882 tau |
| 160 | response | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7' |
| 163 | response | two | unchecked |  | compound adjective | "two-phase schedule" |
| 171 | response | three | match | 3 | T6 n_starts (both arms) |  |
| 171 | response | 12 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~12)' |
| 173 | changes | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 173 | changes | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7' |
| 173 | changes | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8' |
| 178 | reviewer quotation | 10.1016/j.est.2024.114911 | reviewer quotation, verified | 10.1016/j.est.2024.114911 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 178 | reviewer quotation | 10.1109/ISGTEUROPE62998.2024.10863557 | reviewer quotation, verified | 10.1109/ISGTEUROPE62998.2024.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 178 | reviewer quotation | 10.1016/j.apenergy.2022.120569 | reviewer quotation, verified | 10.1016/j.apenergy.2022.120569 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 182 | response | three | match | 3 | the letter itself: DOIs quoted in the R2.8 comment |  |
| 182 | response | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~1' |
| 182 | response | 7 | derived | 7 | derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.110 | references [7]--[9]: the lowest citation number of the three |
| 182 | response | 9 | derived | 9 | derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.110 | references [7]--[9]: the highest citation number of the three |
| 182 | response | 7 | derived | 7 | derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.110 | "[7]": the key cited in main.tex after "rented to renewable producers" |
| 182 | response | 8 | derived | 8 | derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.110 | "[8, 9]": the keys cited in main.tex after "congestion management" |
| 182 | response | 9 | derived | 9 | derived: citation numbers by first \cite in main.tex at the HEAD (elsarticle-num, unsorted): {'aghdam_2023': 8, 'aghdam_2024': 9, 'aghdam_2025': 7}; reviewer DOIs ['10.1016/j.est.2024.114911', '10.110 | "[8, 9]": the keys cited in main.tex after "congestion management" |
| 182 | response | three | match | 3 | the letter itself: DOIs quoted in the R2.8 comment |  |
| 185 | changes | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~1' |
| 194 | response | 20 | unchecked |  | pointer, checked by W177b-3' | pointer 'Equations~(20)--(26)' |
| 194 | response | 26 | unchecked |  | pointer, checked by W177b-3' | pointer 'Equations~(20)--(26)' |
| 195 | changes | 2.3.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.4' |
| 205 | response | two | unchecked |  | language | "couples the two" (the network models and the agent) |
| 205 | response | 2.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3' |
| 210 | changes | 2.3.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.3' |
| 210 | changes | 15 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eqs.~(15)--(19)' |
| 210 | changes | 19 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eqs.~(15)--(19)' |
| 210 | changes | 2.3.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.4' |
| 214 | prose | 3 | unchecked |  | enumerator | reviewer number ('Reviewer 3') |
| 218 | reviewer quotation | 2.2.1 | reviewer quotation, verified | 2.2.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 1 | reviewer quotation, verified | 1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 2034 | reviewer quotation, verified | 2034 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 2037 | reviewer quotation, verified | 2037 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 15 | reviewer quotation, verified | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 2037 | reviewer quotation, verified | 2037 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 2.4 | reviewer quotation, verified | 2.4 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 2 | reviewer quotation, verified | 2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 218 | reviewer quotation | 4.1 | reviewer quotation, verified | 4.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 220 | response | 29 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eqs.~(29)--(30)' |
| 220 | response | 30 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eqs.~(29)--(30)' |
| 220 | response | 14 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~14)' |
| 220 | response | two | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[c_6597a79d].candidate_canonical (nnodes) |  |
| 220 | response | 4.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.1' |
| 221 | response | 60 | match | 60 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_claims (sha256 215fd21d) |  |
| 221 | response | two | MISMATCH | 1 | data/SRP1/Results/P515S53/w153_step5_rows/w153_salvage_addback.json result.totals.n_verdict_differs (sha256 215fd21d) | as written ("It changes no sign in the 60 reported comparisons and two verdicts") the count is of the 60: W153 records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3, 0. |
| 221 | response | two | match | 2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[l_0ee93aca].candidate_canonical (nnodes) |  |
| 221 | response | 2035 | match | 2035 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2035].candidate_canonical (year) |  |
| 221 | response | 2030 | match | 2030 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[yl_y2030].candidate_canonical (year) |  |
| 221 | response | +42.5 | match | 42.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.D_gross |  |
| 221 | response | -2.3 | match | -2.3 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.year_ladder.D_net |  |
| 221 | response | 4.6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.6)' |
| 221 | response | 3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3.5' |
| 221 | response | 10 | match | 10 | definitions.py ENERGY_STORAGE_MIN_ENERGY_STORED (SoC^Min) |  |
| 221 | response | 90 | match | 90 | definitions.py ENERGY_STORAGE_MAX_ENERGY_STORED (SoC^Max) |  |
| 221 | response | 16 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eq.~(16)' |
| 222 | changes | 4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4' |
| 222 | changes | 29 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eqs.~(29)--(30)' |
| 222 | changes | 30 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eqs.~(29)--(30)' |
| 222 | changes | 3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~3.5, 4.1 and 4.6' |
| 222 | changes | 4.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~3.5, 4.1 and 4.6' |
| 222 | changes | 4.6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~3.5, 4.1 and 4.6' |
| 222 | changes | 14 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~14' |
| 224 | prose | 0.50 | match | 0.50 | manuscript_submitted/main.tex (sha256 ca07d7db) lines [866]: 'relative optimality gap of 0.50' | submitted figure quoted in the heading |
| 226 | reviewer quotation | 2.2.7 | reviewer quotation, verified | 2.2.7 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 11 | reviewer quotation, verified | 11 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 4.3 | reviewer quotation, verified | 4.3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 26 | reviewer quotation, verified | 26 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 27 | reviewer quotation, verified | 27 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 29 | reviewer quotation, verified | 29 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 11 | reviewer quotation, verified | 11 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 0.50 | reviewer quotation, verified | 0.50 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 3 | reviewer quotation, verified | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 15 | reviewer quotation, verified | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 4.4 | reviewer quotation, verified | 4.4 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 226 | reviewer quotation | 15 | reviewer quotation, verified | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 230 | response | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7)' |
| 231 | response | 4 | match | 4 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N18] (sha256 34f53a2c; committed check, status match): settling_criterion.TAU = DELTA_R * R_REF / 4 | TAU = DELTA_R * R_REF / 4 |
| 231 | response | 0.07 | match | 0.07 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N19] (sha256 34f53a2c; committed check, status match): settling_criterion.DELTA_R; v6 spec stop_rul | DELTA_R |
| 231 | response | single | unchecked |  | compound adjective | "single-scenario" (the instance name, not a count) |
| 232 | response | two | match | 2 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N22] (sha256 34f53a2c; committed check, status match): T11: R = V / V_SRP1_settled (a ratio of two  | R = V / V_SRP1: a ratio of two values |
| 232 | response | 4.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.4' |
| 232 | response | 15 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~15' |
| 235 | changes | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 235 | changes | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7' |
| 235 | changes | 4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4' |
| 239 | reviewer quotation | 21 | reviewer quotation, verified | 21 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 239 | reviewer quotation | 31 | reviewer quotation, verified | 31 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 239 | reviewer quotation | 2.3.2 | reviewer quotation, verified | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) | the token lies in a part of the quotation that matches the document verbatim |
| 241 | response | 2.3.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.3' |
| 241 | response | one | unchecked |  | language | "one representative year" (the block definition) |
| 241 | response | one | unchecked |  | language | "one representative day" (the block definition) |
| 241 | response | single | unchecked |  | language | "a single ... schedule" (article) |
| 241 | response | one | unchecked |  | language | "one scenario-free variable per hour" (model statement) |
| 241 | response | one | unchecked |  | language | "this one schedule" (model statement) |
| 241 | response | 2.3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.5)' |
| 243 | changes | 2.3.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~2.3.3--2.3.5' |
| 243 | changes | 2.3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~2.3.3--2.3.5' |
| 247 | reviewer quotation | 2.3.2 | reviewer quotation, verified | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 247 | reviewer quotation | 22 | reviewer quotation, verified | 22 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 247 | reviewer quotation | 23 | reviewer quotation, verified | 23 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 250 | changes | 2.3.4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.3.4' |
| 254 | reviewer quotation | 2.3.2 | reviewer quotation, verified | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 254 | reviewer quotation | 22 | reviewer quotation, verified | 22 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 254 | reviewer quotation | 25 | reviewer quotation, verified | 25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 254 | reviewer quotation | 15 | reviewer quotation, verified | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 254 | reviewer quotation | 3 | reviewer quotation, verified | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 254 | reviewer quotation | two | reviewer quotation, verified | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 254 | reviewer quotation | 15 | reviewer quotation, verified | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 256 | response | 0.985 | match | 0.985 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.calendar_retention_per_year (sha256 39106f93) |  |
| 256 | response | 6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~6)' |
| 256 | response | 31.6 | match | 31.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| 256 | response | 64.4 | match | 64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].value_minus_I | sign not written; magnitude compared (the prose / column states the direction) |
| 256 | response | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~11)' |
| 257 | changes | 3.5 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3.5' |
| 257 | changes | 6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~6' |
| 257 | changes | 4.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.3' |
| 257 | changes | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~11' |
| 261 | reviewer quotation | 4.6 | reviewer quotation, verified | 4.6 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 28; letter quotation 15 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 261 | reviewer quotation | 5 | reviewer quotation, verified | 5 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 28; letter quotation 15 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 263 | response | 0.25 | match | 0.25 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (S node 7) |  |
| 263 | response | 1 | match | 1.0 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (E node 7) |  |
| 263 | response | 7 | match | 7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical (node node 7) |  |
| 263 | response | -4.1 | match | -4.1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[no_ageing].value_minus_I |  |
| 263 | response | -31.6 | match | -31.6 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2].value_minus_I |  |
| 263 | response | -64.4 | match | -64.4 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].value_minus_I |  |
| 263 | response | 8,000 | match | 8000 | T8 k of C4 x (-ln 0.70) |  |
| 263 | response | -45.2 | match | -45.2 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C4].value_minus_I |  |
| 263 | response | 50 | match | 50 | T8 k of C3_unit: R = exp(-N D / k), N D = 8,000 |  |
| 263 | response | -73.7 | match | -73.7 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_unit].value_minus_I |  |
| 263 | response | -65.9 | match | -65.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C3_midblock].value_minus_I |  |
| 263 | response | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~11' |
| 263 | response | 7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~7)' |
| 264 | response | 0.70 | match | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |  |
| 264 | response | 0.50 | match | 0.50 | data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json reports.e_soh050.minimum_soh (sha256 3c71d71d) |  |
| 264 | response | +4.9 | match | 4.9 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:delta_value_vs_070].d_gross |  |
| 264 | response | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~11' |
| 264 | response | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8)' |
| 264 | response | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~11' |
| 264 | response | 2035 | match | 2035 | T8 floor binds (0.70): baseline, C3_unit, C3_midblock (all 2035; the other aged arms never) |  |
| 264 | response | 50 | match | 50 | T8 k of C3_unit and C3_midblock: R = exp(-8,000 / k), both 0.50 |  |
| 265 | changes | 4.3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.3' |
| 265 | changes | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~11' |
| 265 | changes | 7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~7' |
| 269 | reviewer quotation | 3 | reviewer quotation, verified | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 269 | reviewer quotation | 8 | reviewer quotation, verified | 8 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 269 | reviewer quotation | 0.50 | reviewer quotation, verified | 0.50 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 270 | reviewer quotation | 3 | reviewer quotation, verified | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 270 | reviewer quotation | 8 | reviewer quotation, verified | 8 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 270 | reviewer quotation | 4.4.1 | reviewer quotation, verified | 4.4.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 270 | reviewer quotation | four | reviewer quotation, verified | four | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 270 | reviewer quotation | 8760 | reviewer quotation, verified | 8760 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) | the token lies in a part of the quotation that matches the document verbatim |
| 272 | response | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eq.~(11)' |
| 272 | response | one | match |  | data/SRP1/Results/P515S53/w160_step6_frozen/export/T7.tex: "one discount factor per representative year" | T7 caption |
| 274 | response | 2.2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.1)' |
| 276 | changes | 2.2.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.7' |
| 276 | changes | 11 | unchecked |  | pointer, checked by W177b-3' | pointer 'Eq.~(11)' |
| 276 | changes | 2.2.1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~2.2.1' |
| 276 | changes | 3 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~3' |
| 276 | changes | 4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4' |
| 282 | prose | five | match | 5 | SRP1.json Years: block length (each) |  |
| 283 | prose | 1 | match | 1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.sweep.sweep_passive_cold.n_blocks |  |
| 283 | prose | 12 | match | 12 | SRP1.json: 3 years x 4 days = 12 TN blocks |  |
| 283 | prose | 8 | match | 8 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.sweep.sweep_price_taker_cold.n_blocks |  |
| 283 | prose | 12 | match | 12 | SRP1.json: 12 blocks |  |
| 283 | prose | 3.6 | unchecked |  | pointer, checked by W177b-3' | pointer 'Sections~3.6' |
| 283 | prose | 4.4 | unchecked |  | pointer, checked by W177b-3' | numbered pointer (Section / Table / Figure / Eq. / Appendix) the pointer regex does not catch |
| 284 | prose | 10^{-5} | match | 1e-05 | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N33] (sha256 34f53a2c; committed check, status match): definitions.py EQUALITY_TOLERANCE = 1e-5; mo | RES availability slack (EQUALITY_TOLERANCE) |
| 284 | prose | 4.8 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.8)' |
| 285 | prose | single | no counterpart |  |  | "single machine": no committed record states the machine count (W176b: the author's attestation); the machine is named in section 4.7 (W163 N55-N57) |
| 285 | prose | single | match | thread caps 1 on every table e | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N45] (sha256 34f53a2c; committed check, status match): W159 c1: harness THREAD_CAP_ENV lines (OMP / | single-threaded (thread caps) |
| 285 | prose | 4.7 | unchecked |  | pointer, checked by W177b-3' | pointer 'Section~4.7' |
| 286 | prose | 49 | match | 49 | data/SRP1/Results/P515S53/w160_step6_frozen/export/paragraphs_v2.md section (v) prediction scorecard: rows numbered 1..n |  |
| 287 | prose | 4 | match | 4 | W167 question4_cost_file.json formula_recompute_7ce1d1ab (the corrected energy formula divides the 4-h system cost by 4) |  |
| 287 | prose | 5 | match | 5 | W167 question4_cost_file.json predecessor_072b1310_selected (submitted: /5) |  |
| 287 | prose | 4 | match | 4 | W167: the corrected divisor |  |
| 287 | prose | 1.25 | derived | 1.25 | derived: W167 question4_cost_file.json: submitted (predecessor 072b1310) energy formula divisor 5 / corrected (7ce1d1ab) divisor 4 = 1.25 | the corrected values are x 5/4 |
| 287 | prose | 1 | unchecked |  | pointer, checked by W177b-3' | pointer 'Table~1' |
| 288 | prose | two | match | 2 | main.tex at the HEAD: fig:search_generic and fig:investment_lattice present, neither in the submitted main.tex |  |
| 288 | prose | 2 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~2)' |
| 288 | prose | 4 | unchecked |  | pointer, checked by W177b-3' | pointer 'Figure~4)' |

### Reviewer quotations (W171a check)

| quote | segment | letter lines | chars | result | ratio | docx paragraph | differences |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 64 | 51 | verbatim | 1.000 | 0 |  |
| 0 | 1 | 66 | 246 | verbatim | 1.000 | 0 |  |
| 0 | 2 | 67 | 116 | verbatim | 1.000 | 3 |  |
| 0 | 3 | 68 | 476 | differs | 0.994 | 5 | delete '"' -> '' (l. 68); delete '."' -> '' (l. 68) |
| 0 | 4 | 69 | 84 | verbatim | 1.000 | 7 |  |
| 0 | 5 | 70 | 294 | verbatim | 1.000 | 9 |  |
| 0 | 6 | 71 | 97 | verbatim | 1.000 | 11 |  |
| 1 | 0 | 90 | 197 | verbatim | 1.000 | 13 |  |
| 1 | 1 | 92 | 88 | verbatim | 1.000 | 13 |  |
| 1 | 2 | 93 | 97 | verbatim | 1.000 | 15 |  |
| 2 | 0 | 101 | 47 | verbatim | 1.000 | 16 |  |
| 3 | 0 | 110 | 797 | verbatim | 1.000 | 16 |  |
| 3 | 1 | 111 | 638 | verbatim | 1.000 | 22 |  |
| 4 | 0 | 124 | 613 | verbatim | 1.000 | 22 |  |
| 5 | 0 | 130 | 648 | verbatim | 1.000 | 22 |  |
| 5 | 1 | 131 | 508 | verbatim | 1.000 | 22 |  |
| 5 | 2 | 132 | 534 | verbatim | 1.000 | 22 |  |
| 6 | 0 | 148 | 603 | verbatim | 1.000 | 22 |  |
| 6 | 1 | 149 | 474 | verbatim | 1.000 | 22 |  |
| 7 | 0 | 178 | 227 | verbatim | 1.000 | 22 |  |
| 8 | 0 | 189 | 684 | verbatim | 1.000 | 22 |  |
| 9 | 0 | 199 | 612 | verbatim | 1.000 | 22 |  |
| 9 | 1 | 200 | 576 | verbatim | 1.000 | 22 |  |
| 10 | 0 | 218 | 1099 | verbatim | 1.000 | 23 |  |
| 11 | 0 | 226 | 1254 | verbatim | 1.000 | 24 |  |
| 12 | 0 | 239 | 924 | differs | 0.999 | 24 | replace '.' -> '⋅' (l. 239) |
| 13 | 0 | 247 | 1137 | verbatim | 1.000 | 26 |  |
| 14 | 0 | 254 | 940 | verbatim | 1.000 | 27 |  |
| 15 | 0 | 261 | 801 | verbatim | 1.000 | 28 |  |
| 16 | 0 | 269 | 820 | verbatim | 1.000 | 29 |  |
| 16 | 1 | 270 | 865 | verbatim | 1.000 | 30 |  |

## Flag and verdict checks (tables)

- l. 1221 (51a6fb5 l. 1189) tab:res_value_ladder C:y2025__n5_p0.25_e0.5 (caption: "All reported differences are determinate": determinate): ok
- l. 1222 (51a6fb5 l. 1190) tab:res_value_ladder B:n5_4h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1223 (51a6fb5 l. 1191) tab:res_value_ladder B:n7_4h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1224 (51a6fb5 l. 1192) tab:res_value_ladder B:n7_4h_e2 (caption: "All reported differences are determinate": determinate): ok
- l. 1225 (51a6fb5 l. 1193) tab:res_value_ladder B:n7_4h_e3 (caption: "All reported differences are determinate": determinate): ok
- l. 1226 (51a6fb5 l. 1194) tab:res_value_ladder B:n7_4h_e4 (caption: "All reported differences are determinate": determinate): ok
- l. 1227 (51a6fb5 l. 1195) tab:res_value_ladder B:n7_4h_e5 (caption: "All reported differences are determinate": determinate): ok
- l. 1228 (51a6fb5 l. 1196) tab:res_value_ladder B:n7_2h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1229 (51a6fb5 l. 1197) tab:res_value_ladder B:n7_2h_e2 (caption: "All reported differences are determinate": determinate): ok
- l. 1230 (51a6fb5 l. 1198) tab:res_value_ladder B:n7_2h_e3 (caption: "All reported differences are determinate": determinate): ok
- l. 1231 (51a6fb5 l. 1199) tab:res_value_ladder B:n7_2h_e4 (caption: "All reported differences are determinate": determinate): ok
- l. 1232 (51a6fb5 l. 1200) tab:res_value_ladder B:n7_2h_e5 (caption: "All reported differences are determinate": determinate): ok
- l. 1233 (51a6fb5 l. 1201) tab:res_value_ladder B:n9_4h_e1 (caption: "All reported differences are determinate": determinate): ok
- l. 1234 (51a6fb5 l. 1202) tab:res_value_ladder B:n9_4h_e3 (caption: "All reported differences are determinate": determinate): ok
- l. 1235 (51a6fb5 l. 1203) tab:res_value_ladder C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5 (caption: "All reported differences are determinate": determinate): ok
- l. 1236 (51a6fb5 l. 1204) tab:res_value_ladder C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5 (caption: "All reported differences are determinate": determinate): ok
- l. 1307 (51a6fb5 l. 1275) tab:res_flex_ladder CHECK:headline_V_minus_I_settled: ok
- l. 1308 (51a6fb5 l. 1276) tab:res_flex_ladder H:m1.5:value_minus_I: ok
- l. 1309 (51a6fb5 l. 1277) tab:res_flex_ladder H:m1.75:value_minus_I: ok
- l. 1310 (51a6fb5 l. 1278) tab:res_flex_ladder H:m2:value_minus_I: ok
- l. 1329 (51a6fb5 l. 1297) tab:res_energy_ladder_m2 H:m2:value_minus_I / None (caption: "All differences are determinate"): ok
- l. 1330 (51a6fb5 l. 1298) tab:res_energy_ladder_m2 I:m2:e2_value_minus_I / I:m2:second_MWh (caption: "All differences are determinate"): ok
- l. 1331 (51a6fb5 l. 1299) tab:res_energy_ladder_m2 J:e3_value_minus_I / J:e2_to_e3 (caption: "All differences are determinate"): ok
- l. 1332 (51a6fb5 l. 1300) tab:res_energy_ladder_m2 J:e4_value_minus_I / J:e3_to_e4 (caption: "All differences are determinate"): ok
- l. 1333 (51a6fb5 l. 1301) tab:res_energy_ladder_m2 J:e5_value_minus_I / J:e4_to_e5 (caption: "All differences are determinate"): ok
- l. 1371 (51a6fb5 l. 1339) tab:res_ageing C2_calfade (determinate; floor 2035): ok
- l. 1372 (51a6fb5 l. 1340) tab:res_ageing C2 (determinate; floor never): ok
- l. 1373 (51a6fb5 l. 1341) tab:res_ageing C3_unit (determinate; floor 2035): ok
- l. 1374 (51a6fb5 l. 1342) tab:res_ageing C3_midblock (determinate; floor 2035): ok
- l. 1375 (51a6fb5 l. 1343) tab:res_ageing C4 (determinate; floor never): ok
- l. 1376 (51a6fb5 l. 1344) tab:res_ageing no_ageing (within resolution; floor n/a): ok
- l. 1378 (51a6fb5 l. 1346) tab:res_ageing 0.50 floor (determinate; floor never binds): ok
- l. 1483 (51a6fb5 l. 1451) tab:res_year_ladder caption: "determinate gross ... within resolution net": ok
- l. 1504 (51a6fb5 l. 1472) tab:res_discount caption: "every difference is determinate": ok

## Integrity

checks: 27/27 True; failed []; guards ok True; exit 0

