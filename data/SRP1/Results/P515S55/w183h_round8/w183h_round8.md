# W183h — round 8 at Overleaf a87e195 (zero solves)

HEAD `a87e1950fbdbb3667f37e64a43c298ba675e6cff` (parent `edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838`), 2026-10-25T15:48:48+00:00. Objective convention: no objective is computed; numbers are rated against the frozen JSON and named records (Q = gross_operational_cost, contracted interface settlement excluded).

## Predictions

- (a) the four files identical to the round-8 deliverables; clone diff = expert diffs: **HELD** — {"main.tex": {"byte_identical": true, "md5_addendum_85": true, "hunks_clone_expert": [31, 31], "every_hunk_pm_equal": true, "ranges_equal": true, "context_equal": true}, "response_to_reviewers_draft.tex": {"byte_identical": true, "md5_addendum_85": true, "hunks_clone_expert": [12, 12], "every_hunk_pm_equal": true, "ranges_equal": true, "context_equal": true}, "highlights.tex": {"byte_identical": true, "md5_addendum_85": true, "hunks_clone_expert": [1, 1], "every_hunk_pm_equal": true, "ranges_equal": true, "context_equal": true}, "cover_letter.tex": {"byte_identical": true, "md5_addendum_85": t
- (b) Appendix E bitwise; verify 33 PASS; 0 MISMATCH; 0 entries changed; b-marks exactly as listed: **HELD** — {"bitwise": true, "verify_33_pass": true, "entries": 1623, "mismatch": 0, "not_frozen_disagrees": 0, "entries_changed_vs_edd7a63_marks_removed": 0, "cells_changed_vs_edd7a63_raw": 16, "raw_changes_are_mark_insertions_only": true, "marks_as_stated": true}
- (c) letter v10 pointers: 0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED; [42]/[43]: **HELD** — {"counts": {"MATCH": 167, "SUBMITTED": 5}, "citations": {"naumann_2018": 42, "eve_mb31_datasheet": 43, "boyd_admm_2011": 44}}
- (d) 0 of the 16 A7 sentences remaining: **HELD** — {"RESOLVED": 16}
- (d) every over-budget point labelled at every table / figure occurrence: **NOT HELD** — {"table_or_figure": 25, "labelled_or_covered": 23, "not": ["P12a", "P12b"]}
- (e) 0 MISMATCH in the numbers of the 21 edits: **HELD** — {"NOT RATED": 14, "MATCH": 93, "DERIVED": 7}
- (f) 0 [CONFIRM, 0 label-word prefixes, blue on the 23 A5 lines: **NOT HELD** — {"confirm": 0, "label_word_prefixes": 7, "a5_lines_blue": true, "a5_captions_blue": true}

## (a) identity

| file | sha256 | md5 | = review copy | hunks clone/expert | +/- equal | ranges | context |
|---|---|---|---|---|---|---|---|
| main.tex | `cec2e2982dfa` | `39f6fd86` | True | [31, 31] | True | True | True |
| response_to_reviewers_draft.tex | `982482a1e070` | `2aeb33b2` | True | [12, 12] | True | True | True |
| highlights.tex | `13e37f6d452d` | `4cedd45c` | True | [1, 1] | True | True | True |
| cover_letter.tex | `de7a91748074` | `88eee8ce` | True | [2, 2] | True | True | True |

## (b) Appendix E

- outcome: `{"bitwise": true, "verify_33_pass": true, "entries": 1623, "mismatch": 0, "not_frozen_disagrees": 0, "entries_changed_vs_edd7a63_marks_removed": 0, "cells_changed_vs_edd7a63_raw": 16, "raw_changes_are_mark_insertions_only": true, "marks_as_stated": true}`
- E.21 marked rows [5, 6, 9, 10, 11, 38, 39]; E.20 marked rows [4, 5, 8, 9, 10, 26, 27, 30, 31]; legends True/True
- block hunks edd7a63 -> HEAD: 16
  - ['comment'] [block header] [2560, 2560] -> [2563, 2565]: % (Addendum 80): E.4 second phase and the E.21 note on evaluations 40 and 41. Round 8 (Addendum 83/84): the author's six % hand-edited hunks
  - ['prose'] [\section{\textcolor{blue}{Evaluation Record}}] [2567, 2567] -> [2572, 2572]: This appendix gives the record behind the comparisons of Section~\ref{sec:results}: every pairwise comparison (Table~\ref{tab:appE_claims});
  - ['prose'] [\subsection{\textcolor{blue}{Terminal Salvage Cred] [2596, 2596] -> [2601, 2601]: where $\bar{c}^{\text{Inv,E}}_{y^\text{Inv}} = \sum_{c \in \Omega_C} \omega_c\, c^{\text{Inv,E}}_{y^\text{Inv},c}$ is the expected undiscoun
  - ['E.20 caption'] [\subsection{\textcolor{blue}{Comparisons}}] [2609, 2609] -> [2614, 2614]: \caption{\textcolor{blue}{Every pairwise comparison of the record. Eval.: the two evaluations of Table~\ref{tab:appE_evaluations} (reference
  - ['E.20 cell'] [\subsection{\textcolor{blue}{Comparisons}}] [2628, 2629] -> [2633, 2634]: 4 & 7: 1/4$^{b}$ & 1 / 5 & 274.0 & 13.6 & 20.19 & det. & 274.0 & 20.19 & det. \\ 5 & 7: 1.25/5$^{b}$ & 1 / 6$^{u}$ & 347.2 & 23.9 & 14.52 & 
  - ['E.20 cell'] [\subsection{\textcolor{blue}{Comparisons}}] [2632, 2634] -> [2637, 2639]: 8 & 7: 1.5/3$^{b}$ & 1 / 9$^{u}$ & 361.7 & 41.4 & 8.74 & det. & 361.7 & 8.74 & det. \\ 9 & 7: 2/4$^{b}$ & 1 / 10 & 494.4 & 13.4 & 36.91 & de
  - ['E.20 cell'] [\subsection{\textcolor{blue}{Comparisons}}] [2652, 2653] -> [2657, 2658]: 26 & 7: 1/4$^{b}$, $m=2$ & 34 / 38$^{f}$ & 176.6 & 9.7 & 18.14 & det. & 176.6 & 18.14 & det. \\ 27 & 7: 1.25/5$^{b}$, $m=2$ & 34 / 39$^{u}$ 
  - ['E.20 cell'] [\subsection{\textcolor{blue}{Comparisons}}] [2656, 2657] -> [2661, 2662]: 30 & 7: 0.75/3 $\to$ 1/4$^{b}$, $m=2$$^{\Delta}$ & 37$^{u}$ / 38$^{f}$ & 42.8 & 27.7 & 1.55 & det. & 42.8 & 1.55 & det. \\ 31 & 7: 1/4$^{b}$
  - ['E.21 caption'] [\subsection{\textcolor{blue}{Evaluations}}] [2703, 2703] -> [2708, 2708]: \caption{\textcolor{blue}{Recourse evaluations. Plan: node: MVA/MWh. Condition: flexibility-price multiplier $m$ (base: $m = 1$) or ageing c
  - ['E.21 cell'] [\subsection{\textcolor{blue}{Evaluations}}] [2721, 2722] -> [2726, 2727]: 5 & 7: 1/4$^{b}$ & 2025 & base & cert., osc. & 86 & 153 & 1.00 & 4.5 & --- & --- & 0 & 86 & run \\ 6 & 7: 1.25/5$^{b}$ & 2025 & base & growt
  - ['E.21 cell'] [\subsection{\textcolor{blue}{Evaluations}}] [2725, 2727] -> [2730, 2732]: 9 & 7: 1.5/3$^{b}$ & 2025 & base & growth test & 89 & 198 & --- & --- & 0.8 & 13.8 & 6 & 89 & run \\ 10 & 7: 2/4$^{b}$ & 2025 & base & cert.
  - ['E.21 cell'] [\subsection{\textcolor{blue}{Evaluations}}] [2760, 2761] -> [2765, 2766]: 38 & 7: 1/4$^{b}$ & 2025 & $m=2$ & cert., osc.$^{f}$ & 127 & 158 & 0.71 & 3.2 & --- & --- & 1 & 127 & run \\ 39 & 7: 1.25/5$^{b}$ & 2025 & $
  - ['prose'] [\subsection{\textcolor{blue}{Neighbours of the Pla] [2803, 2803] -> [2808, 2808]: \caption{\textcolor{blue}{Admissible single-node investment plans for installation in 2025 under the baseline constraints. The investment bu
  - ['E.22 caption'] [\subsection{\textcolor{blue}{Neighbours of the Pla] [2813, 2813] -> [2818, 2818]: \caption{\textcolor{blue}{Neighbours of the plan without storage re-evaluated under the certification rule: the smallest unit (0.25~MVA / 0.
  - ['E.23 caption'] [\subsection{\textcolor{blue}{Static Arrangements: ] [2853, 2853] -> [2858, 2858]: \caption{\textcolor{blue}{Static no-reverse-flow (NRF) arrangements at the plan without storage, by solver start. Cold: default start; pertu
  - ['prose'] [\subsection{\textcolor{blue}{Certification Statist] [2919, 2919] -> [2924, 2924]: The acceptance tolerance~\eqref{eq:acceptance_tolerance} uses $\delta_R = 0.07$ on the reference value $\mathcal{V} = 259{,}375.33$~\euro{},

## (c) letter v10 pointers

- counts {'MATCH': 167, 'SUBMITTED': 5}; by extractor {'W177b-3prime letter_pointers': 169, 'W183h lettered': 3}
- citations {'naumann_2018': 42, 'eve_mb31_datasheet': 43, 'boyd_admm_2011': 44}; v9 at HEAD {'MATCH': 103, 'MISMATCH': 65, 'SUBMITTED': 5}
- v9 pointers dropped in v10: [(125, 'Figure~2'), (126, 'Figure~2'), (230, 'Section~2.2.7)'), (288, 'Figure~2)')]

| line | pointer | verdict | expected at HEAD | note |
|---|---|---|---|---|
| 43 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 45 | Section~3.4) | MATCH | 3.4 |  |
| 58 | Section~4.3 | MATCH | 4.3 | new in v10 (no v9 counterpart) |
| 77 | [43] | MATCH | 43 |  |
| 77 | Section~3.4 | MATCH | 3.4 |  |
| 81 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 81 | Tables~4--11) | MATCH | 4--11 | range member(s) removed since 8f051a2: ['tab:res_flex_ladder'] |
| 82 | Section~2.3.3 | MATCH | 2.3.3 |  |
| 86 | Section~2.3.3 | MATCH | 2.3.3 |  |
| 86 | Tables~4--11 | MATCH | 4--11 | range member(s) removed since 8f051a2: ['tab:res_flex_ladder'] |
| 97 | Section~3.3) | MATCH | 3.3 | v9 target heading removed; resolved by its content needle (subsection 3.3) |
| 97 | Table~11) | MATCH | 11 |  |
| 98 | Table~11 | MATCH | 11 |  |
| 100 | Figure 2 | SUBMITTED | 2 | the R1.5 heading quotes the reviewer's figure number (submitted Figure 2, Addendum 75) |
| 102 | Figure~2 | SUBMITTED | 2 | "Figure 2 of the submitted version" |
| 102 | Figure~2 | MATCH | 2 |  |
| 103 | Figure~2 | MATCH | 2 |  |
| 125 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 125 | Section~3.5) | MATCH | 3.5 |  |
| 126 | Section~3.5 | MATCH | 3.5 |  |
| 136 | Sections~2.2 and~2.3.1) | MATCH | 2.2 | v9 target target removed (heading and needle text no longer typeset); declared target |
| 136 | Sections~2.2 and~2.3.1) | MATCH | 2.3.1 |  |
| 137 | Section~2.2.6) | MATCH | 2.2.6 |  |
| 141 | former 2.2.7 | SUBMITTED | 2.2.7 | "former 2.2.7 Benders' cuts" |
| 142 | new 2.2.6 | MATCH | 2.2.6 |  |
| 152 | Section~2.2.6) | MATCH | 2.2.6 |  |
| 156 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 160 | Appendix~E.7) | MATCH | E.7 |  |
| 171 | Table~8) | MATCH | 8 |  |
| 173 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 194 | Equations~(29)--(37) | MATCH | 29--37 |  |
| 195 | Section~2.3.3 | MATCH | 2.3.3 |  |
| 210 | Section~2.3.2 | MATCH | 2.3.2 |  |
| 210 | Section~2.3.3 | MATCH | 2.3.3 |  |
| 220 | Eqs.~(E.1)--(E.2) | MATCH | E.1--E.2 |  |
| 220 | Table~10) | MATCH | 10 |  |
| 221 | Section~3.4 | MATCH | 3.4 |  |
| 222 | Eqs.~(E.1)--(E.2) | MATCH | E.1--E.2 |  |
| 222 | Sections~3.4, 4.1 and 4.6 | MATCH | 3.4 |  |
| 222 | Table~10 | MATCH | 10 |  |
| 229 | Section~2.2.6 | MATCH | 2.2.6 | new in v10 (no v9 counterpart) |
| 231 | Appendix~A.5) | MATCH | A.5 | new in v10 (no v9 counterpart) |
| 232 | Figure~15 | SUBMITTED | 15 | the reviewer's Figure 15 (Addendum 75: no such figure in the submitted source) |
| 232 | Section~4.4 | SUBMITTED | 4.4 | the reviewer's Section 4.4 (submitted version) |
| 235 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 241 | Section~2.3.2 | MATCH | 2.3.2 |  |
| 241 | Section~2.3.4) | MATCH | 2.3.4 |  |
| 243 | Sections~2.3.2--2.3.4 | MATCH | 2.3.2--2.3.4 |  |
| 249 | [43] | MATCH | 43 |  |
| 249 | Section~3.4 | MATCH | 3.4 |  |
| 249 | Table~3) | MATCH | 3 |  |
| 250 | Section~2.3.3 | MATCH | 2.3.3 |  |
| 250 | Section~3.4 | MATCH | 3.4 |  |
| 250 | Table~3 | MATCH | 3 |  |
| 256 | [42] | MATCH | 42 |  |
| 256 | Table~3) | MATCH | 3 |  |
| 256 | Table~7) | MATCH | 7 |  |
| 257 | Section~3.4 | MATCH | 3.4 |  |
| 257 | Table~3 | MATCH | 3 |  |
| 257 | Table~7 | MATCH | 7 |  |
| 263 | Figure~5) | MATCH | 5 |  |
| 263 | Table~7 | MATCH | 7 |  |
| 264 | Table~7 | MATCH | 7 |  |
| 264 | Table~7 | MATCH | 7 |  |
| 265 | Figure~5 | MATCH | 5 |  |
| 265 | Table~7 | MATCH | 7 |  |
| 275 | Section~2.2.6 | MATCH | 2.2.6 |  |
| 282 | Sections~3.5 and~4.4) | MATCH | 3.5 |  |
| 284 | Appendices~E.7 and~A.8) | MATCH | E.7 |  |
| 288 | Fig.~E.19) | MATCH | E.19 |  |

## (d) A7

- ratings {'RESOLVED': 16}; remaining []

| id | file:line HEAD | rating |
|---|---|---|
| M-ABS-1 | main.tex:[145] | RESOLVED |
| M-ABS-2 | main.tex:[145] | RESOLVED |
| M-HL-1 | main.tex:[163] | RESOLVED |
| M-CON-1 | main.tex:[387] | RESOLVED |
| H-3 | highlights.tex:[12] | RESOLVED |
| H-4 | highlights.tex:[13] | RESOLVED |
| L-01 | response_to_reviewers_draft.tex:[42, 43] | RESOLVED |
| L-04 | response_to_reviewers_draft.tex:[78] | RESOLVED |
| L-05 | response_to_reviewers_draft.tex:[102] | RESOLVED |
| L-07 | response_to_reviewers_draft.tex:[137] | RESOLVED |
| L-08 | response_to_reviewers_draft.tex:[168, 169] | RESOLVED |
| L-09 | response_to_reviewers_draft.tex:[229, 230] | RESOLVED |
| L-10 | response_to_reviewers_draft.tex:[230, 231, 232] | RESOLVED |
| L-12 | response_to_reviewers_draft.tex:[263] | RESOLVED |
| L-13 | response_to_reviewers_draft.tex:[264] | RESOLVED |
| C-1 | cover_letter.tex:[None] | RESOLVED |

## (d) A8

- ratings {'LABELLED: probe / budget text in the same paragraph': 2, 'LABELLED: probe / budget text at the place': 5, 'COVERED BY THE SECTION 3.5 SENTENCE (it names the break-even fit of this subsection); no mark at the place': 1, 'LABELLED: the table marks its over-budget rows $^{b}$ and carries the legend': 1, 'LABELLED: $^{b}$ at the place': 18, 'NOT LABELLED AT THE PLACE (text)': 1, 'LABELLED: every over-budget row under the header carries $^{b}$': 1, 'NOT MARKED: a table row without $^{b}$; only the Section 3.5 sentence names the plan': 2, 'PARTIAL (unchanged): the lattice figure legend marks over-budget lattice points': 1, 'WITHDRAWN: the affine-ladder argument is no longer in the letter line': 2}
- table/figure occurrences 25, labelled or covered 23

| id | line HEAD | element | rating |
|---|---|---|---|
| P01 | 1426 | text, Section 4.2 | LABELLED: probe / budget text in the same paragraph |
| P02 | 1426 | text, Section 4.2 | LABELLED: probe / budget text in the same paragraph |
| P03 | 1427 | text, Section 4.2 | LABELLED: probe / budget text at the place |
| P04 | 1433 | Figure 1 caption (fig1_breakeven.pdf) | LABELLED: probe / budget text at the place |
| P05 | 1439 | Table tab:res_breakeven caption | COVERED BY THE SECTION 3.5 SENTENCE (it names the break-even fit of this subsection); no mark at the place |
| P06 | 1469 | text, Section 4.2 | LABELLED: probe / budget text at the place |
| P07 | 1469 | text, Section 4.2 | LABELLED: probe / budget text at the place |
| P08 | 1474 | Table tab:res_energy_ladder_m2 caption | LABELLED: the table marks its over-budget rows $^{b}$ and carries the legend |
| P08a | 1489 | Table tab:res_energy_ladder_m2 row | LABELLED: $^{b}$ at the place |
| P08b | 1490 | Table tab:res_energy_ladder_m2 row | LABELLED: $^{b}$ at the place |
| P09 | 1806 | text, Section 5 | NOT LABELLED AT THE PLACE (text) |
| P10a | 2633 | tab:appE_claims (a) row 4 | LABELLED: $^{b}$ at the place |
| P10b | 2634 | tab:appE_claims (a) row 5 | LABELLED: $^{b}$ at the place |
| P10c | 2637 | tab:appE_claims (a) row 8 | LABELLED: $^{b}$ at the place |
| P10d | 2638 | tab:appE_claims (a) row 9 | LABELLED: $^{b}$ at the place |
| P10e | 2639 | tab:appE_claims (a) row 10 | LABELLED: $^{b}$ at the place |
| P10f | 2657 | tab:appE_claims (b) row 26 | LABELLED: $^{b}$ at the place |
| P10g | 2658 | tab:appE_claims (b) row 27 | LABELLED: $^{b}$ at the place |
| P10h | 2661 | tab:appE_claims (b) row 30 | LABELLED: $^{b}$ at the place |
| P10i | 2662 | tab:appE_claims (b) row 31 | LABELLED: $^{b}$ at the place |
| P11a | 2721 | tab:appE_evaluations group (a) header | LABELLED: every over-budget row under the header carries $^{b}$ |
| P11d | 2726 | tab:appE_evaluations row 5 | LABELLED: $^{b}$ at the place |
| P11e | 2727 | tab:appE_evaluations row 6 | LABELLED: $^{b}$ at the place |
| P11f | 2730 | tab:appE_evaluations row 9 | LABELLED: $^{b}$ at the place |
| P11g | 2731 | tab:appE_evaluations row 10 | LABELLED: $^{b}$ at the place |
| P11h | 2732 | tab:appE_evaluations row 11 | LABELLED: $^{b}$ at the place |
| P11b | 2765 | tab:appE_evaluations row 38 | LABELLED: $^{b}$ at the place |
| P11c | 2766 | tab:appE_evaluations row 39 | LABELLED: $^{b}$ at the place |
| P12a | 2904 | tab:appE_gap row | NOT MARKED: a table row without $^{b}$; only the Section 3.5 sentence names the plan |
| P12b | 2911 | tab:appE_gap row | NOT MARKED: a table row without $^{b}$; only the Section 3.5 sentence names the plan |
| P13 | 2926 | text, Appendix E.7 | LABELLED: probe / budget text at the place |
| P14 | 2802 | text + fig_lattice_case.pdf, Appendix E.3 | PARTIAL (unchanged): the lattice figure legend marks over-budget lattice points |
| P15 | 58 | letter, opening | WITHDRAWN: the affine-ladder argument is no longer in the letter line |
| P16 | 264 | letter, R3.6 | WITHDRAWN: the affine-ladder argument is no longer in the letter line |

## (e) numbers

- main: {'NOT RATED': 14, 'MATCH': 93, 'DERIVED': 7}; supplement {'MATCH': 4, 'NOT RATED': 1}; negative control 104/104 refused

| line | written | rating | source |
|---|---|---|---|
| 145 | two | NOT RATED | compound adjective |
| 145 | one | NOT RATED | model structure |
| 145 | single | MATCH | SRP1.json (the instance of the planning searches): 1 market scenario x 1 operation scenario |
| 163 | two | NOT RATED | compound adjective |
| 387 | two | NOT RATED | compound adjective |
| 387 | one | NOT RATED | model structure |
| 387 | single | MATCH | SRP1.json (the instance of the planning searches): 1 market scenario x 1 operation scenario |
| 390 | 108 | DERIVED | derived: case9_2025.json nodes 9 + case33_1/2/3_2025.json nodes [33, 33, 33] = 108 |
| 1291 | 32 | MATCH | case33_1/2/3_2025.json: loads per ADN, each with fl_reg = 1 |
| 1291 | 1.25 | MATCH | case33_1/2/3_operational_data.xlsx sheet Growth Factors, Load (x 100) |
| 1291 | 3.00 | MATCH | the same workbooks, Flexible Load (x 100) |
| 1291 | 1.00 | MATCH | case9_operational_data.xlsx sheet Growth Factors, Load (x 100) |
| 1291 | 2.50 | MATCH | case9_operational_data.xlsx, Flexible Load (x 100) |
| 1347 | Five | MATCH | W182d A8 table: over-budget evaluations at the base price (2025) |
| 1347 | single | NOT RATED | compound adjective |
| 1347 | 7 | MATCH | W182d: the node of every over-budget base evaluation |
| 1347 | 2025 | MATCH | W182d: the installation year of every over-budget base evaluation |
| 1347 | 4 | MATCH | W182d over-budget base evaluations: the longer duration E/S (h) |
| 1347 | 4 | MATCH | W182d over-budget base evaluations at 4 h: E (MWh), 1. smallest |
| 1347 | 5 | MATCH | W182d over-budget base evaluations at 4 h: E (MWh), 2. smallest |
| 1347 | 2 | MATCH | W182d over-budget base evaluations: the shorter duration E/S (h) |
| 1347 | 3 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 1. smallest |
| 1347 | 4 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 2. smallest |
| 1347 | 5 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 3. smallest |
| 1347 | 1.15 | MATCH | W182d A8 table: smallest I(x) of the five, MEUR (each above the budget) |
| 1347 | 1.91 | MATCH | W182d A8 table: largest I(x) of the five, MEUR |
| 1347 | 1 | MATCH | SRP1_ESS_Params.json budget (EUR) / 1e6 |
| 1352 | three | MATCH | T6 n_starts (both arms) |
| 1382 | single | NOT RATED | compound adjective |
| 1427 | ten | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.band |
| 1427 | 171,310 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.brea |
| 1427 | 192,560 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.brea |
| 1427 | 24 | DERIVED | derived: 1 - e*/e_cost = 1 - 192,557.46 / 253,877.68 = 61,320.21 / 253,877.68 = 24.1534 % (T3 break_even.conservative_A6 |
| 1427 | 33 | DERIVED | derived: 1 - e*/e_cost = 1 - 171,309.08 / 253,877.68 = 82,568.60 / 253,877.68 = 32.5230 % (T3 break_even.conservative_A6 |
| 1427 | 253,880 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.e_cost_eur_per_MWh |
| 1427 | five | MATCH | W182d: over-budget evaluations among the fit points (evaluations 2-11) |
| 1427 | ten | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.break_even.conservative_A64.band |
| 1427 | 4 | MATCH | W182d over-budget base evaluations at 4 h: E (MWh), 1. smallest |
| 1427 | 5 | MATCH | W182d over-budget base evaluations at 4 h: E (MWh), 2. smallest |
| 1427 | 4 | MATCH | W182d over-budget base evaluations: the longer duration E/S (h) |
| 1427 | 3 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 1. smallest |
| 1427 | 4 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 2. smallest |
| 1427 | 5 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 3. smallest |
| 1427 | 2 | MATCH | W182d over-budget base evaluations: the shorter duration E/S (h) |
| 1433 | 7 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical |
| 1433 | 2025 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical |
| 1433 | 4 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[ref:bd504ecf].candidate_canonical |
| 1433 | 2 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[d_d3709599].candidate_canonical ( |
| 1433 | 2 | DERIVED | derived: T3 conservative_A64.intervals: (half_width - bar) / tau = [2.0] for ['n7_2h_e2', 'n7_2h_e3', 'n7_4h_e3', 'n7_4h |
| 1433 | 1 | MATCH | SRP1_ESS_Params.json budget (EUR) / 1e6 |
| 1433 | 4 | MATCH | W182d over-budget base evaluations at 4 h: E (MWh), 1. smallest |
| 1433 | 5 | MATCH | W182d over-budget base evaluations at 4 h: E (MWh), 2. smallest |
| 1433 | 4 | MATCH | W182d over-budget base evaluations: the longer duration E/S (h) |
| 1433 | 3 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 1. smallest |
| 1433 | 5 | MATCH | W182d over-budget base evaluations at 2 h: E (MWh), 3. smallest |
| 1433 | 2 | MATCH | W182d over-budget base evaluations: the shorter duration E/S (h) |
| 1469 | five | MATCH | m = 2 ladder (H:m2, I:m2:e2, J:e3, J:e4, J:e5; 4 h at node 7): positive margins |
| 1469 | 4 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1469 | 7 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1469 | 2025 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1469 | 4 | MATCH | W182d over-budget evaluations at m = 2 (4 h): E, smallest |
| 1469 | 5 | MATCH | W182d over-budget evaluations at m = 2 (4 h): E, largest |
| 1469 | 1 | MATCH | the E step between consecutive cells of the m = 2 ladder |
| 1469 | 41.0 | MATCH | T1 I/J increments: smallest |
| 1469 | 51.8 | MATCH | T1 I/J increments: largest |
| 1474 | 4 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1474 | 7 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1474 | 2025 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1489 | 4 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_a11d7966].candidate_canonical ( |
| 1489 | 176.6 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].d_gro |
| 1489 | 15.9 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_value_minus_I].uncer |
| 1489 | 42.8 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].d_gross |
| 1489 | 27.7 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e3_to_e4].gross_thresho |
| 1490 | 5 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json cells[j_5f3cccb4].candidate_canonical ( |
| 1490 | 228.4 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e5_value_minus_I].d_gro |
| 1490 | 26.5 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e5_value_minus_I].gross |
| 1490 | 51.8 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].d_gross |
| 1490 | 26.5 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.claims[J:e4_to_e5].gross_thresho |
| 1502 | 1 | MATCH | SRP1_ESS_Params.json budget (EUR) / 1e6 |
| 1568 | 318.0 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.ageing.rows[C2_calfade].I |
| 1576 | single | NOT RATED | compound adjective |
| 1576 | 0.70 | MATCH | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |
| 1576 | 2035 | MATCH | T8 floor binds (0.70): baseline, C3_unit, C3_midblock (all 2035; the other three never) |
| 1576 | 50 | MATCH | T8 k of C3_unit |
| 1576 | 8,000 | MATCH | T8 k of C4 x (-ln 0.70) |
| 1576 | three | MATCH | SRP1__s53_3x3.json Years: block length (each) |
| 1576 | 2037 | DERIVED | derived: W182g C2 cell 3x3_n7_4h_e1 (7: 0.25/1 @2025, eval c82522f4, cycle 69): first block with the 0.70 floor active = |
| 1576 | 0.70 | MATCH | data/SRP1/SharedESS/SRP1_ESS_Params.json ageing.minimum_soh (sha256 39106f93) |
| 1576 | 0.50 | MATCH | data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json reports.e_soh050.minimum_soh (sha256 3c71d7 |
| 1576 | 0.18 | MATCH | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N70] (sha256 34f53a |
| 1576 | 0.20 | MATCH | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N71] (sha256 34f53a |
| 1576 | 0.27 | MATCH | data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json checks_in_v4[N72] (sha256 34f53a |
| 1576 | 2025 | MATCH | a64.scored.B.efc_per_day: the block-year keys |
| 1576 | 2030 | MATCH | a64.scored.B.efc_per_day: the block-year keys |
| 1576 | 2035 | MATCH | a64.scored.B.efc_per_day: the block-year keys |
| 1576 | 4.9 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:delta_value_vs |
| 1576 | 59.5 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.a64.rows[E:soh050:value_minus_I] |
| 1585 | three | MATCH | T6: coordinated + the two NRF arms |
| 1639 | nine | DERIVED | derived: SRP1__s53_3x3.json NumMarketScenarios 3 x num_operation_scenarios 3 (TN and every DN) = 9; instance_record.json |
| 1639 | three | MATCH | 3 x 3 NumMarketScenarios |
| 1639 | three | MATCH | 3 x 3 num_operation_scenarios |
| 1639 | five | MATCH | 3 x 3 Years (count) |
| 1639 | three | MATCH | 3 x 3 block length |
| 1639 | 3 | DERIVED | derived: W182g records 3x3_x0 (eval f6e9cd53, cycle 72): (deviation_total -748,745.77 + row18_charge_weighted 25,082,401 |
| 1639 | single | NOT RATED | compound adjective |
| 1639 | 236.7 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[0].value |
| 1639 | 318.0 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[1].value |
| 1639 | 81.3 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[2].value |
| 1639 | 18.0 | MATCH | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json tables.three_by_three.rows[3].value |
| 1835 | Three | NOT RATED | model structure |
| 1835 | one | NOT RATED | model structure |
| 1835 | one | NOT RATED | model structure |
| 1835 | three | NOT RATED | model structure |
| 1835 | one | NOT RATED | model structure |
| 2601 | 2040 | MATCH | W182g C2 horizon SRP1: y0 + sum Y = terminal date |
| 2601 | 1 | NOT RATED | date |
| 2601 | 2040 | MATCH | W182g C2: terminal date of both configurations (equal) |
| 2601 | 2035 | MATCH | W182g C2: last representative block, single-scenario |
| 2601 | 2037 | MATCH | W182g C2: last representative block, 3 x 3 |

## (f)

- CONFIRM 0; general "<word>~\ref{app:" {'Section': 2, 'in': 14, 'and': 1, 'Appendix': 5, 'of': 2}; label-word prefixes 7; A5 lines all blue True; A5 captions all blue True

## Prediction-record bullet (quoted only)

- l. 286: "\item The prediction record of the revision (49 predictions, with outcomes) is provided as supplementary material."

Negative controls: [('a main.tex: expert diff + line altered', True), ('a response_to_reviewers_draft.tex: expert diff + l', True), ('a highlights.tex: expert diff + line altered', True), ('a cover_letter.tex: expert diff + line altered', True), ('b1 entry: E.21 row 1 band altered', True), ('b2 mark removed from E.21 row 39', True), ('b3 mark added to E.21 row 4 (an admissible plan)', True), ('b4 E.20: mark dropped from row 26 and misplaced on', True), ('b5 generated block with one byte changed is not lo', True), ('c1 l. 45 "Section~3.4" -> "Section~3.5"', True), ('c2 l. 256 "[42]" -> "[43]"', True), ('c3 l. 288 "Fig.~E.19" -> "Fig.~E.18"', True), ('c4 l. 231 "Appendix~A.5" -> "Appendix~A.4" (declar', True), ('d1 the L-05 quote planted back into the letter', True), ('f1 a planted "[CONFIRM" is counted', True), ('f2 the A5 caption at HEAD l. 1317 unwrapped in a c', True), ('e every rated number perturbed (104)', True)]

Guards ok True (57 guards); check failures []
