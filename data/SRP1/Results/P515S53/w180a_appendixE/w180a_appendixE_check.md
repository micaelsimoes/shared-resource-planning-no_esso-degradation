# W180a — Appendix E regenerated and re-audited (Overleaf e5322f8)

main.tex sha256 `b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492`; frozen JSON `590088fe`; script `p515_s53_w180a_appendixE_check.py`. Zero solves; guards 0/0 verified.

## Predictions (Addendum 79, item (a))

- (a1) the regenerated block equals main.tex bitwise: **HELD** — 1 occurrence(s) of the generator output in main.tex at e5322f8, lines [(2375, 2694)], line-aligned [(True, True)]; also = committed appendixE_evaluation_record.tex: True; = verifier's own regeneration: True
- (task) verify_appE.py: 33 PASS, 0 FAIL: **HELD** — 33 PASS, 0 FAIL, exception None
- (a2) 0 MISMATCH in 1,623 entries: **HELD** — 1623 entries; 1443 MATCH on frozen fields, 0 MISMATCH; 180 not-a-frozen-field entries agree with their named source, 0 disagree; mapping problems 0
- (a3) exactly two entries changed from 48099ab (row 16 Status -> superseded^s, Rule -> ---): **HELD** — 2 of 1637 table cells changed: [('E.21', '16', 'Status', 'cert., osc.$^{s}$', 'superseded$^{s}$'), ('E.21', '16', 'Rule', 'record', '---')]; outside the cells (reported separately, not entries): {'row_set_changes': 0, 'captions': 2, 'notes': 0, 'group_headers': 0, 'other_lines': 0}
- (a4) all five prose sentences MATCH (intro, E.3, E.4, E.5, E.6): **HELD** — P1 MATCH; P2a DERIVED; P2b MATCH; P3a MATCH; P3b MATCH; P3c MATCH; P3d MATCH; P3e MATCH; P4 MATCH; P5a MATCH; P5b MATCH

## Generator / verifier / bitwise

- generator stdout `ok 58 61`; output `data/SRP1/Results/P515S53/w180a_appendixE/appE_generated_w180a.tex` sha256 `79ff8094d1559f564850ca10957f565833097c5a3460a70a8675fd1f97329444` (320 lines)
- verifier: 33 PASS, 0 FAIL; exception None; tempdir output sha256 `79ff8094d1559f564850ca10957f565833097c5a3460a70a8675fd1f97329444`
- in main.tex at e5322f8: [{'byte_start': 188555, 'byte_end': 222479, 'start_line': 2375, 'end_line': 2694, 'n_lines': 320, 'starts_at_line_start': True, 'ends_at_line_end': True}]
- = committed manuscript_review/appendixE_evaluation_record.tex: True; = verifier regeneration: True
- Appendix E block at e5322f8 (comment rule to the line before Declaration): lines [2375, 2695]; at 48099ab: lines [2374, 2702]

## Table entries at e5322f8 vs the frozen JSON

| table | rows | entries | MATCH | MISMATCH | not a frozen field, agrees | not a frozen field, disagrees |
|---|---|---|---|---|---|---|
| E.20 | 58 | 639 | 581 | 0 | 58 | 0 |
| E.21 | 61 | 854 | 732 | 0 | 122 | 0 |
| E.22 | 8 | 18 | 18 | 0 | 0 | 0 |
| E.23 | 7 | 49 | 49 | 0 | 0 | 0 |
| E.24 | 9 | 63 | 63 | 0 | 0 | 0 |

Every MISMATCH / disagreement:

- none

Cross-check: 48099ab rated under the same (round-6) convention: non-agreeing entries [{'table': 'E.21', 'row': 16, 'key': 'status', 'status': 'MISMATCH', 'printed': 'cert., osc.', 'frozen_or_derived': 'superseded'}, {'table': 'E.21', 'row': 16, 'key': 'rule', 'status': 'NOT_FROZEN_DISAGREES', 'printed': 'record', 'frozen_or_derived': '---'}]

## Entry diff 48099ab -> e5322f8

Table cells compared: 1637; changed: 2

- cell E.21 row 16 col 4 (Status), l. 2525 -> 2525: `cert., osc.$^{s}$` -> `superseded$^{s}$`
- cell E.21 row 16 col 13 (Rule), l. 2525 -> 2525: `record` -> `---`
- row_set_changes: 0
- captions: 2
  - E.21 l. [2494] -> [2494]: [{'op': 'delete', 'old': 'or', 'new': ''}, {'op': 'replace', 'old': 'test).', 'new': 'test), or superseded.'}, {'op': 'replace', 'old': 'evaluation~15.', 'new': 'evaluation~15; certified under an earlier version of the rule, whose cycle, range and band are shown, and not decided under the rule of \\ref{app:admm_updated_certification} within its recorded cycles.'}]
  - E.23 l. [2630] -> [2626]: [{'op': 'replace', 'old': 'excluded.', 'new': 'excluded, after the consistency pass.'}, {'op': 'replace', 'old': 'horizon.}}', 'new': 'horizon, from the first phase (before the pass).}}'}]
- notes: 0
- group_headers: 0
- other_lines: 0

## Appendix E block line diff (hunks)

| # | kind | section | lines old | lines new | token changes |
|---|---|---|---|---|---|
| 1 | comment | block header | [2379, 2381] | [2380, 2381] | [{'op': 'replace', 'old': '[CONFIRM -- W179] mapping for', 'new': 'Audited against'}, {'op': 'replace', 'old': 'audit: Table E.1 holds', 'new': 'frozen JSON by W179a (0 MISMATCH in 1,623 table entries; P5_15_W179A_APPENDIX_E.md). Round 6 % (Addendum 79): row 16 relabelled, E.3/E.4/E.5/E.6 prose corrected;'}, {'op': 'delete', 'old': "55 distinct comparisons of T1 (the record's 60 claims less the %… |
| 2 | prose | \section{\textcolor{blue}{Evaluation Record}} | [2388, 2388] | [2388, 2388] | [{'op': 'replace', 'old': 'decimals.', 'new': 'decimals, except the residual ratio of Table~\\ref{tab:appE_gap}, given to three.'}] |
| 3 | table E.21 | \subsection{\textcolor{blue}{Evaluations}} | [2494, 2494] | [2494, 2494] | [{'op': 'delete', 'old': 'or', 'new': ''}, {'op': 'replace', 'old': 'test).', 'new': 'test), or superseded.'}, {'op': 'replace', 'old': 'evaluation~15.', 'new': 'evaluation~15; certified under an earlier version of the rule, whose cycle, range and band are shown, and not decided under the rule of \\ref{app:admm_updated_certification} within its recorded cycles.'}] |
| 4 | table E.21 | \subsection{\textcolor{blue}{Evaluations}} | [2525, 2525] | [2525, 2525] | [{'op': 'replace', 'old': 'cert., osc.$^{s}$', 'new': 'superseded$^{s}$'}, {'op': 'replace', 'old': 'record', 'new': '---'}] |
| 5 | prose | \subsection{\textcolor{blue}{Neighbours of the Plan without… | [2588, 2588] | [2588, 2588] | [{'op': 'insert', 'old': '', 'new': 'own'}, {'op': 'replace', 'old': '$\\sigma_Q$ (Subsection~\\ref{subsec:res_baseline}).', 'new': "($\\sigma_Q$, or the sum of the two evaluations' bars where larger; Subsection~\\ref{subsec:res_baseline})."}, {'op': 'insert', 'old': '', 'new': 'The other six, the smallest unit at the two remaining pairs of nodes and at all three nodes, in 2025 or 2030, were eval… |
| 6 | comment | \subsection{\textcolor{blue}{Neighbours of the Plan without… | [2590, 2591] | [2590, 2589] | [{'op': 'delete', 'old': '% [CONFIRM -- W179] that the other six box neighbours (two-node 2025 {5,7},{7,9}; 2030 {5,9},{7,9}; three-node 2025, 2030) % were evaluated by the search only (production exit, sigma_Q), not under the certification rule.', 'new': ''}] |
| 7 | prose | \subsection{\textcolor{blue}{Static Arrangements: Solver St… | [2623, 2623] | [2621, 2621] | [{'op': 'replace', 'old': 'The consistency pass', 'new': 'Each arrangement'}, {'op': 'insert', 'old': '', 'new': 'solved in three phases (Section~\\ref{sec:case_settings}):'}, {'op': 'delete', 'old': 'second', 'new': ''}, {'op': 'replace', 'old': 'solve', 'new': "networks' dispatch and the transmission network's dispatch against it, from which the curtailment columns are taken; a re-solve of each… |
| 8 | comment | \subsection{\textcolor{blue}{Static Arrangements: Solver St… | [2625, 2626] | [2623, 2622] | [{'op': 'delete', 'old': "% [CONFIRM -- W179] (i) the unit of 'NRF violations (n)' (interface-hours? node-hours?) and the reading of the consistency % re-evaluation in the sentence above; (ii) 'warm' = started from the solution of the certified coordinated evaluation.", 'new': ''}] |
| 9 | table E.23 | \subsection{\textcolor{blue}{Static Arrangements: Solver St… | [2630, 2630] | [2626, 2626] | [{'op': 'replace', 'old': 'excluded.', 'new': 'excluded, after the consistency pass.'}, {'op': 'replace', 'old': 'horizon.}}', 'new': 'horizon, from the first phase (before the pass).}}'}] |
| 10 | prose | \subsection{\textcolor{blue}{Evaluations Refused by the Gap… | [2661, 2661] | [2657, 2657] | [{'op': 'replace', 'old': 'cycle.', 'new': 'cycle (for evaluations~40 and~41 the split and the residual were not recorded).'}] |
| 11 | prose | \subsection{\textcolor{blue}{Certification Statistics and R… | [2692, 2692] | [2688, 2688] | [{'op': 'replace', 'old': 'two', 'new': 'three'}, {'op': 'insert', 'old': '', 'new': '12,'}, {'op': 'insert', 'old': '', 'new': '0.004$\\tau$,'}] |
| 12 | comment | \subsection{\textcolor{blue}{Certification Statistics and R… | [2694, 2696] | [2690, 2689] | [{'op': 'delete', 'old': '% [CONFIRM -- W179] the 46 certificates = the 49 certified rows of Table E.2 less evaluation 16 (superseded) and the two % bitwise twins (56, 61); the ten >= 0.95 tau = at_or_above_0.95_tau_counted; continued cells b_4649234b (0.738 tau), % d_c52e1670 (0.781 / 0.882 tau) per paragraphs_v2 (ii); 119 acceptable-clean events after N, all DSO (W153).', 'new': ''}] |
| 13 | prose | \subsection{\textcolor{blue}{Certification Statistics and R… | [2699, 2699] | [2692, 2692] | [{'op': 'insert', 'old': '', 'new': '(the plan without storage, the reference unit and a larger plan not tabulated here)'}, {'op': 'replace', 'old': 'to their certification cycles;', 'new': 'through the stopping cycles of the residual-based runs they continue, cycles 132, 112 and 87;'}] |

## Mutation test

| id | edit | caught by frozen check | caught by diff | detail |
|---|---|---|---|---|
| M1 | E.20 row 3 col 3 (Delta gross (kEUR)): '208.3' -> '208.4' | True | True | {'frozen_check_new_non_agreeing': [['E.20', '3', 'd_gross_k']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.20', '3', 3]], 'diff_vs_48099ab_symmetric_change': [['E.20', '3… |
| M2 | E.21 row 1 col 8 (Band): '4.2' -> '4.3' | True | True | {'frozen_check_new_non_agreeing': [['E.21', '1', 'band_k']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.21', '1', 8]], 'diff_vs_48099ab_symmetric_change': [['E.21', '1', … |
| M3 | E.21 row 16 col 4 (Status): 'superseded$^{s}$' -> 'cert., osc.$^{s}$' | True | True | {'frozen_check_new_non_agreeing': [['E.21', '16', 'status']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.21', '16', 4]], 'diff_vs_48099ab_symmetric_change': [['E.21', '16… |
| M4 | E.21 row 16 col 13 (Rule): '---' -> 'record' | True | True | {'frozen_check_new_non_agreeing': [['E.21', '16', 'rule']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.21', '16', 13]], 'diff_vs_48099ab_symmetric_change': [['E.21', '16'… |
| M5 | E.22 row 19 col 5 (x): '3.40' -> '3.41' | True | True | {'frozen_check_new_non_agreeing': [['E.22', '19', 'phase_b']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.22', '19', 5]], 'diff_vs_48099ab_symmetric_change': [['E.22', '1… |
| M6 | E.23 row Passive NRF\|cold col 3 (NRF n): '29' -> '30' | True | True | {'frozen_check_new_non_agreeing': [['E.23', 'passive/cold', 'nrf_n']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.23', 'Passive NRF\|cold', 3]], 'diff_vs_48099ab_symmetri… |
| M7 | E.24 row 42 col 9 (Resid.): '0.692' -> '0.693' | True | True | {'frozen_check_new_non_agreeing': [['E.24', '42', 'pf_primal_last']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.24', '42', 9]], 'diff_vs_48099ab_symmetric_change': [['E.… |
| M8 | E.21 row 40 col 13 (Rule): 'record' -> 'run' | True | True | {'frozen_check_new_non_agreeing': [['E.21', '40', 'rule']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.21', '40', 13]], 'diff_vs_48099ab_symmetric_change': [['E.21', '40'… |
| M9 | E.20 row 34 col 9 (Verdict net): 'det.' -> 'w.r.' | True | True | {'frozen_check_new_non_agreeing': [['E.20', '34', 'verdict_net']], 'frozen_check_vanished_non_agreeing': [], 'diff_vs_e5322f8_cells': [['E.20', '34', 9]], 'diff_vs_48099ab_symmetric_change': [['E.20'… |
| M10 | E.23 caption: 'after the consistency pass.' -> 'before the consistency pass.' | not applicable (caption) | True | {'diff_vs_e5322f8': [{'table': 'E.23', 'op': 'replace', 'lines_old': [2626], 'lines_new': [2626], 'old': '\\caption{\\textcolor{blue}{Static no-reverse-flow (NRF) arrangements at the plan without sto… |
| M11 | E.21 caption: 'within its recorded cycles' -> 'within its cycles' | not applicable (caption) | True | {'diff_vs_e5322f8': [{'table': 'E.21', 'op': 'replace', 'lines_old': [2494], 'lines_new': [2494], 'old': '\\caption{\\textcolor{blue}{Recourse evaluations. Plan: node: MVA/MWh. Condition: flexibility… |
| M12 | E.22 notes: '51.3~k\\euro{}' -> '51.4~k\\euro{}' | True | True | {'diff_vs_e5322f8': [{'table': 'E.22', 'op': 'replace', 'lines_old': [2613], 'lines_new': [2613], 'old': '\\textit{Note:} The superseded first certificate of the 2025 node-5 neighbour (evaluation~16)… |

## Prose (round 6)

| id | section | line | status | written | source |
|---|---|---|---|---|---|
| P1 | intro | 2388 | MATCH | Amounts are rounded to 0.1~k\euro{} and ratios to two decimals, except the residual ratio of Table~\ref{tab:appE_gap}, … | decimal places of every printed k EUR amount and every ratio column of E.20-E.24 (parsed here, W179a census); E.24 Resi… |
| P2a | E.3 | 2588 | DERIVED | at its own resolution ($\sigma_Q$, or the sum of the two evaluations' bars where larger; | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.x0.rows.search_resolution_eur, recomputed as… |
| P2b | E.3 | 2588 | MATCH | The other six, the smallest unit at the two remaining pairs of nodes and at all three nodes, in 2025 or 2030, were eval… | data/SRP1/Results/P515S53/w173_search_counts/w173_search_counts.json item2.x0.rows (frozen_table_rows empty; search_F_m… |
| P3a | E.4 | 2621 | MATCH | the distribution networks' dispatch and the transmission network's dispatch against it, from which the curtailment colu… | p515_s53_w116_benchmark_nrf.py phase A; uncoordinated_benchmark.py run_operational_planning_uncoordinated docstring |
| P3b | E.4 | 2621 | MATCH | a re-solve of each distribution network at the transmission network's interface voltage with its no-reverse-flow rows s… | p515_s53_w116_benchmark_nrf.py phase B; uncoordinated_benchmark.py build_consistency_reevaluation_block; data/SRP1/Resu… |
| P3c | E.4 | 2621 | MATCH | and the consistency pass, a re-solve of each distribution network at that voltage with the rows on, followed by the tra… | p515_s53_w116_benchmark_nrf.py phase C; data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.jso… |
| P3d | E.4 | 2621 | MATCH | The consistency column is the change of $Q$ from the first phase to the pass. | p515_s53_w116_benchmark_nrf.py effect_on_q_eur = Q(phase C) - Q(phase A) |
| P3e | E.23 caption | 2626 | MATCH | $Q$: gross operational cost, settlement excluded, after the consistency pass. Curtailment: day-weighted, over the horiz… | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json benchmark.objective_convention, arms_i… |
| P4 | E.5 | 2657 | MATCH | (for evaluations~40 and~41 the split and the residual were not recorded) | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json dead_zone.cells share_by_node / pf_pri… |
| P5a | E.6 | 2688 | MATCH | The three evaluations continued past their certificates, 12, 14 and 3, moved by a further 0.004$\tau$, 0.74$\tau$ and 0… | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json N25 (b_2a0ba8b2 +17.67 EUR past… |
| P5b | E.6 | 2692 | MATCH | the three single-scenario reference continuations (the plan without storage, the reference unit and a larger plan not t… | data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/campaign_results.json, data/SRP1/Result… |
| P6 | E.6 | 2688 | DERIVED | Ten of the 46 certificates the tables use | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json at_or_above_0_95_tau.registry (49 cert… |
| P7 | E.21 caption (legend) | 2494 | MATCH | $^{s}$: superseded by evaluation~15; certified under an earlier version of the rule, whose cycle, range and band are sh… | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells_appended_w160.pb_y2025_n5 (super… |

Decimal tokens of >= 3 places in the appendix prose (beside P1):

- l. 2688: `0.004` (3 dp) … d past their certificates, 12, 14 and 3, moved by a further 0.004$\tau$, 0.74$\tau$ and 0.78--0.88$\tau$.

## Integrity checks

- main_tex_lines: True
- prev_main_tex_lines: True
- generator_ran: True
- verifier_ran_without_exception: True
- bitwise_checks_computed: True
- e21_61_rows_numbered_1_61: True
- e21_map_one_to_one_onto_frozen_cells: True
- e20_58_rows_numbered_1_58: True
- mapping_problems_none: True
- e21_map_same_at_48099ab: True
- mutation_test_every_edit_caught: True
- every_needle_found: True
- every_prose_sentence_found_verbatim: True
