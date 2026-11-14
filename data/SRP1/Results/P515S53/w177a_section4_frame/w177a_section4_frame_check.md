# W177a -- section 4 frame check (section4_expert_draft.tex sha256 e73b6d91, commit 4f941b29)

Zero solves; script `p515_s53_w177a_section4_frame_check.py`; frozen tables `data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json` (sha256 590088fe); clone manuscript/6a67305f25e8348fb71380c3 HEAD 8b764234 (main.tex sha256 7aa9e105). Objective convention: Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum (first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR

## Predictions (expert, Addendum 73)

- **zero MISMATCH in the composed tables and captions** -- HELD: 0 MISMATCH among 466 table / caption tokens ({'unchecked': 19, 'match': 431, 'excluded': 14, 'approximate': 2})
- **the salvage equation needs one correction among its four points** -- FAILED as worded: items (a)-(d) all hold as worded, and the equation needs TWO corrections -- the cohort cost is the expected ENERGY cost only (the definition of I, adjacent to (c)) and the health factor is (SoH - SoH^Min)/(1 - SoH^Min) (adjacent to (b)); the frame's equation gives 4.22 x and 1.86 x the recorded salvage of yl_y2030 / yl_y2035
- **the four other structural CONFIRMs (S/E, I/J keys, F2 plan, 16 cycles) hold** -- FAILED: C2 confirmed, C3 confirmed, C4 differs, C6 confirmed (C4: node 7 is 1 MVA / 3.5 MWh, not 3 MWh)

## CONFIRM points

| id | lines | point | outcome | summary |
|---|---|---|---|---|
| C1 | 28-45 (comment 43-45) | eq:salvage_credit, eq:salvage_remaining_life and items (a)-(d) against _get_terminal_salvage_value_* | differs | no function named _get_terminal_salvage_value_* exists in the tracked *.py files; the reported net uses _build_terminal_salvage_value_expression (via get_salvage_value). Items (a), (b), (c), (d) hold as worded; eq:salvage_remaining_life holds; eq:salvage_credit needs TWO corrections: the cohort cost is the expected ENERGY cost only (no c^{Inv,S} S term), and the health factor is (SoH - SoH^Min)/(1 - SoH^Min), not SoH. The production form reproduces the recorded salvage of yl_y2030 / yl_y2035 to 0.0e+00 relative; the frame's equation gives 49,624.20 / 104,914.88 EUR against the recorded 11,759.26 / 56,504.17 EUR |
| C2 | 83-98 (comment 103-105) | S / E / duration of each B and C cell against its candidate record (S = E / duration) | confirmed | 16 of 16 rows: node, S, E, E/S and year equal the frozen candidate; evaluation records found for 16, equal to the frozen candidate, candidate key and eval key for 16 |
| C3 | 208-209 | the I/J cells are the node-7 4 h plans of 2-5 MWh (keys cc4eb5df, fb21d822, 11725e45, cab98853 = the B cells n7_4h_e2 ... e5) | confirmed | i_5a6a88b4 cc4eb5df = d_c52e1670 ([0.5, 2.0] at node 7, 2025); j_f3aa335e fb21d822 = d_4a82a64a ([0.75, 3.0] at node 7, 2025); j_a11d7966 11725e45 = d_3632b0ae ([1.0, 4.0] at node 7, 2025); j_5f3cccb4 cab98853 = d_36686489 ([1.25, 5.0] at node 7, 2025) |
| C4 | 130-131 | the F2 plan behind ref:5ca4f86c (year, nodes, P, E) | differs | recorded: 2030, node 5 0.25 MVA / 0.5 MWh, node 7 1 MVA / 3.5 MWh; W173 label y2030__n5_p0.25_e0.5__n7_p1_e3.5 (z [1, 1, 4, 7, 0, 0, 1], eval key 5ca4f86c); the frame writes node 7 1 MVA / 3 MWh |
| C5 | 262, 269 | the value and AE / EFC of the 0.50-floor cell (e_soh050) | confirmed (values available) | value 258,456.29 EUR (258.5 kEUR) = value - I -59,500.72 + I 317,957.01 (frozen A64 row) = Q(0) - Q(e_soh050) 258,456.29 (W155 summary); AE 0.7781 (0.778), EFC/day 1.0763 (1.08) PV-weighted (W155 summary reports.e_soh050.AE / EFC; the T8 definition M46A.pv of the ageing series); floor binds: never (floor_year None, minimum_soh 0.5, |dual| <= 5.1e-10) |
| C6 | 342 | the 3 x 3 "16 cycles" of the post-hoc continuation | confirmed | W98 r2 continuation: replay bitwise through N = 72 (the production exit, cycle 72), continuation record cycles 1-88 -> 16 cycles past the exit (73-88), stopped by the frozen |dQ| < 500 EUR x 3 rule; W99 damped-cosine fit window [72, 88] (17 points), L = -6,234.94 EUR (= D, T11 6.2 kEUR) |
| C7 | 390-392 | W153 totals: "0 sign changes in 60 claims"; one verdict changes category (the F2 neighbour y2030__n5_p0.25_e0.5__n7_p1_e3, 0.95 x gross, 1.22 x net) | confirmed | W153 totals: n_claims 60, n_sign_differs 0, n_verdict_differs 1 (['L:y2030__n5_p0.25_e0.5__n7_p1_e3']: within the uncertified bar gross -> determinate net); frozen T1: 60 claims, 0 sign changes; the claim at 0.9472 x gross (within the uncertified bar) and 1.2229 x net (determinate) |
| C8 | 403, 410-411 | the 2025 cohort's salvage is zero (T^Rem = 0 at y^End for T^Cal = 15 and y^Inv = 2025) | confirmed | code: age_at_terminal(2025) = 15, T_rem = max(15 - 15, 0) = 0 -> remaining-life fraction 0, the 2025 term is exactly 0; records: 30 cells with a 2025 plan, max |salvage| 5.76e-33 EUR (numerical residue of the inactive cohorts' terms, not the 2025 cohort), ref:bd504ecf salvage 0.0; T1: d_net = d_gross on every claim whose cells are all 2025 plans (none differ); the row may print 0.0 |
| C9 | 452 | "41 new recourse evaluations (14, 20 and 7 against budgets of 20, 20 and 60)" = W173 realized use, summed | confirmed | W173 item1: s47 14/20 (mesh_local_optimum_unit_poll_failed), s51 20/20 (STOP_FOR_REVIEW_completion_cap), s53 7/60 (poll_failure_at_unit_mesh); sum 41 |

## Number check

Tokens 1057: unchecked 64, excluded 89, match 888, MISMATCH 2, approximate 14

Composed tables and captions (body tokens in table / figure environments): {'unchecked': 19, 'match': 431, 'excluded': 14, 'approximate': 2}

| line | scope | written | status | value at written precision | source | note |
|---|---|---|---|---|---|---|
| 3 | comment | 1448 | MISMATCH | 1445 | main.tex at the clone HEAD: the line before the CONCLUSIONS banner (the header's own rule; Addendum 73: 1126-1445) |  |
| 131 | comment | 3 | MISMATCH | 3.5 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:5ca4f86c].candidate_canonical (E node 7) |  |
| 281 | comment | 1,264 | approximate | 1264 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.multiple |  |
| 284 | comment | 0.81 | approximate | 0.81 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by |  |
| 305 | body | 1,264 | approximate | 1264 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.multiple |  |
| 305 | body | 0.81 | approximate | 0.81 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.w160_additions.arms_in_full.price_taker.curtailment_phase_A_by |  |
| 349 | comment | 7 | approximate | 7 | 1 - R(upper) = 6.6 % |  |
| 349 | comment | 9 | approximate | 9 | 1 - R(lower) = 9.1 % |  |
| 465 | body | 55 | approximate | ['55'] | T9 node-5 share (five cells) |  |
| 465 | body | 31 | approximate | ['31'] | T9 node-9 share |  |
| 465 | body | 14 | approximate | ['14'] | T9 node-7 share |  |
| 465 | body | 0.69 | approximate | 0.682-0.700 | T9 pf_primal last (0.682-0.700) |  |
| 485 | body | 171 | approximate | 171 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[0] |  |
| 485 | body | 193 | approximate | 193 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.breakeven_range[1] |  |
| 485 | body | 254 | approximate | 254 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.e_cost_eur_per_MWh |  |
| 487 | body | 654 | approximate | 654 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.benchmark.coordinated_Q |  |

## Text checks

- l. 83 tab:res_value_ladder C:y2025__n5_p0.25_e0.5: ok
- l. 84 tab:res_value_ladder B:n5_4h_e1: ok
- l. 85 tab:res_value_ladder B:n7_4h_e1: ok
- l. 86 tab:res_value_ladder B:n7_4h_e2: ok
- l. 87 tab:res_value_ladder B:n7_4h_e3: ok
- l. 88 tab:res_value_ladder B:n7_4h_e4: ok
- l. 89 tab:res_value_ladder B:n7_4h_e5: ok
- l. 90 tab:res_value_ladder B:n7_2h_e1: ok
- l. 91 tab:res_value_ladder B:n7_2h_e2: ok
- l. 92 tab:res_value_ladder B:n7_2h_e3: ok
- l. 93 tab:res_value_ladder B:n7_2h_e4: ok
- l. 94 tab:res_value_ladder B:n7_2h_e5: ok
- l. 95 tab:res_value_ladder B:n9_4h_e1: ok
- l. 96 tab:res_value_ladder B:n9_4h_e3: ok
- l. 97 tab:res_value_ladder C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5: ok
- l. 98 tab:res_value_ladder C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5: ok
- l. 181 tab:res_flex_ladder CHECK:headline_V_minus_I_settled: ok
- l. 182 tab:res_flex_ladder H:m1.5:value_minus_I: ok
- l. 183 tab:res_flex_ladder H:m1.75:value_minus_I: ok
- l. 184 tab:res_flex_ladder H:m2:value_minus_I: ok
- l. 199 tab:res_energy_ladder_m2 H:m2:value_minus_I / None: ok
- l. 200 tab:res_energy_ladder_m2 I:m2:e2_value_minus_I / I:m2:second_MWh: ok
- l. 201 tab:res_energy_ladder_m2 J:e3_value_minus_I / J:e2_to_e3: ok
- l. 202 tab:res_energy_ladder_m2 J:e4_value_minus_I / J:e3_to_e4: ok
- l. 203 tab:res_energy_ladder_m2 J:e5_value_minus_I / J:e4_to_e5: ok
- l. 256 tab:res_ageing C2_calfade: ok
- l. 257 tab:res_ageing C2: ok
- l. 258 tab:res_ageing C3_unit: ok
- l. 259 tab:res_ageing C3_midblock: ok
- l. 260 tab:res_ageing C4: ok
- l. 261 tab:res_ageing no_ageing: ok
- l. 262 tab:res_ageing 0.50 floor (floor never binds: a64 scored B floor_never_binds_as_predicted): ok
- l. 406 tab:res_year_ladder 2035 - 2030: ok
- l. 423 tab:res_discount 0.0: ok
- l. 424 tab:res_discount 0.02: ok
- l. 425 tab:res_discount 0.05: ok
- l. 426 tab:res_discount 0.08: ok
- l. 362 tab:res_multiscenario value - I (T11 label "determinate True"): ok
- l. accepted sentences paragraphs_v5.md accepted sentences 1-4, verbatim: ok
- l. source comments every claim / row named in the frame's comments exists in the frozen tables (claim ids exact or a unique prefix; T8 arm names; T6 / T3 row names in the export CSV): ok
- l. 202 Addendum 64 ruling 4: "the manuscript uses the conservative figures (61.3 kEUR/MWh; J at its uncertified bar)" -- the 4 MWh row of tab:res_energy_ladder_m2 in its main columns: NOT OK
- l. 383 T4 caption sentence quoted "verbatim" (l. 383-384): ok

## Labels, figures, references

- labels 26, duplicates [], header list equal True (header - labels [], labels - header []), collide with main.tex outside the range []
- unresolved refs []
- main.tex refs outside the range to old section-4 labels: [{'line': 308, 'target': 'sec:results', 'kept_by_frame': True}, {'line': 481, 'target': 'sec:results', 'kept_by_frame': True}, {'line': 2031, 'target': 'sec:results', 'kept_by_frame': True}]
- figures [('fig1_breakeven.pdf', True, True), ('fig2_flexibility_ladder.pdf', True, True), ('fig3_ageing_arms.pdf', True, True), ('fig5_benchmark.pdf', True, True), ('fig4_q_versus_cycle.pdf', True, True)]; set equals export True

### Literal references in main.tex at the clone HEAD (outside the replaced range)

- l. 377: `(Section~4)` -- state of health and the remaining calendar life of each installation (Section~4).}
- l. 415: `Section~4` -- the budget as the sensitivity study of Section~4, and their combinations);
- l. 445: `Section~4.7` -- determinacy threshold; the resolution the search used is reported in Section~4.7. What the search certifies is
- l. 631: `Section~4.5` -- to measure the remaining movement of the objective, and the ratio of Section~4.5 is reported as a band that al
- l. 635: `(Section~4.7)` -- evaluations continued past certification moved by at most $0.9\,\tau$ (Section~4.7).
- l. 671: `Section~4` -- vative convention; the mid-block evaluation point is a sensitivity in Section~4 ($\forall e$, $y^\text{Inv}$,
- l. 1066: `Section~4.3` -- is $SoH^{\text{Min}} = 0.70$. The other rows are the sensitivities of Section~4.3: the same count without calen
- l. 1119: `(Section~4.4)` -- The value of coordination (Section~4.4) is measured against two stati

Addendum 73 list: l. 415 Section~4 -> found there; l. 445 Section~4.7 -> found there; l. 631 Section~4.5 -> found there; l. 1066 Section~4.3 -> found there; l. 1116 (Section~4.4) -> NOT at that line; found at [1119]
Not on the Addendum 73 list: l. 377 (Section~4); l. 635 (Section~4.7); l. 671 Section~4

## Integrity

checks: 13/13 True; failed []; guards ok True; exit 0

