# W181a — identity and Appendix E at Overleaf cd57efc

main.tex sha256 `ff7b045edd17136fce7a37d9ef7ec16c8515ab4f8df69f3c33cc021127bbed35`; letter sha256 `06cc24604c54955964c6000edf48de2aba34408d6a6e499e55122ac705d5cb2c`; frozen JSON `590088fe`; script `p515_s53_w181a_identity_appendixE.py`. Zero solves; guards 0/0 verified.

## Predictions

- (a1) main.tex byte-identical to main_e5322f8_round7.tex and the letter to v8 (HEAD cd57efc): **HELD** — HEAD cd57efcb54399cb652a0ac2dcd64e5d3a78d89fc; main.tex sha256 ff7b045edd17136fce7a37d9ef7ec16c8515ab4f8df69f3c33cc021127bbed35, 2713 lines, = round7 copy True; letter sha256 06cc24604c54955964c6000edf48de2aba34408d6a6e499e55122ac705d5cb2c, 292 lines, = v8 True
- (a2) clone diff e5322f8 -> HEAD = main_e5322f8_round7.diff (6 hunks) + letter_v7_v8.diff (2 hunks), +/- lines in order: **HELD** — files ['main.tex', 'response_to_reviewers_draft.tex']; main.tex: hunks 6/6, +/- per hunk True, in order True, ranges True, context True; response_to_reviewers_draft.tex: hunks 2/2, +/- per hunk True, in order True, ranges True, context True
- (b1) Appendix E regenerated bitwise from the patched gen_appE.py: **HELD** — 1 occurrence(s) in main.tex at cd57efc, lines [(2377, 2697)], line-aligned [(True, True)]; = committed appendixE_evaluation_record.tex True; = verifier regeneration True
- (b2) verify_appE.py: 33 PASS, 0 FAIL: **HELD** — 33 PASS, 0 FAIL, exception None
- (b3) 0 MISMATCH in 1,623 entries: **HELD** — 1623 entries; 1443 MATCH on frozen fields, 0 MISMATCH; 180 not-a-frozen-field entries agree, 0 disagree; mapping problems 0
- (b4) 0 table cells changed e5322f8 -> HEAD: **HELD** — 0 of 1637 cells changed; row-set changes 0; outside the cells: captions 0, notes 1, group headers 0, other table lines 0
- (b5, task context) only the E.4 paragraph and the E.21 note differ in the appendix block: **HELD for every non-comment line; FAILED as worded (the block also differs in a comment)** — 3 block hunk(s): ['comment'] in 'block header' new l. [2383, 2384]; ['E.21 note'] in '\\subsection{\\textcolor{blue}{Evaluations}}' new l. [2583, 2583]; ['prose'] in '\\subsection{\\textcolor{blue}{Static Arrangements: Solver Sta' new l. [2624, 2624]

## (a) Identity

| revision | file | sha256 | lines | equals |
|---|---|---|---|---|
| cd57efc | main.tex | `ff7b045edd17136f…` | 2713 | {'manuscript_review/main_e5322f8_round7.tex': True} |
| cd57efc | response_to_reviewers_draft.tex | `06cc24604c549559…` | 292 | {'manuscript_review/response_to_reviewers_v8.tex': True} |
| e5322f8 | main.tex | `b3282295ece4cc3b…` | 2710 | {'manuscript_review/main_48099ab_round6.tex': True} |
| e5322f8 | response_to_reviewers_draft.tex | `34a5754ba6ccb53e…` | 292 | {'manuscript_review/response_to_reviewers_v7.tex': True} |

Files in `git diff e5322f8 cd57efc`: ['main.tex', 'response_to_reviewers_draft.tex']

- main.tex: clone 6 hunks, expert 6 hunks; every hunk +/- equal True; concatenated +/- in order True; ranges equal True; context equal True
  - hunk 1: @@ -966,7 +966,7 @@ The total rated energy capacity at each… | -1 +1 | +/- equal True, range equal True, context equal True
  - hunk 2: @@ -1304,6 +1304,8 @@ Within each ADN, all 32 loads are mod… | -0 +2 | +/- equal True, range equal True, context equal True
  - hunk 3: @@ -1616,7 +1618,7 @@ Within each ADN, all 32 loads are mod… | -1 +1 | +/- equal True, range equal True, context equal True
  - hunk 4: @@ -2378,7 +2380,8 @@ Figure~\ref{fig:cs3_adn_node_9_res_sc… | -1 +2 | +/- equal True, range equal True, context equal True
  - hunk 5: @@ -2577,7 +2580,7 @@ No. & Plan & Year & Condition & Statu… | -1 +1 | +/- equal True, range equal True, context equal True
  - hunk 6: @@ -2618,7 +2621,7 @@ No. & Plan & Year & Condition & Statu… | -1 +1 | +/- equal True, range equal True, context equal True
- response_to_reviewers_draft.tex: clone 2 hunks, expert 2 hunks; every hunk +/- equal True; concatenated +/- in order True; ranges equal True; context equal True
  - hunk 1: @@ -1,4 +1,4 @@ | -1 +1 | +/- equal True, range equal True, context equal True
  - hunk 2: @@ -122,7 +122,7 @@ reused.} | -1 +1 | +/- equal True, range equal True, context equal True

## (b) Generator / verifier / bitwise

- generator stdout `ok 58 61`; output `data/SRP1/Results/P515S53/w181a_identity_appendixE/appE_generated_w181a.tex` sha256 `623ddbbdcaa684b4f7d70d525c5519d91466f3fd7ebff735dc6349c483d94986` (321 lines)
- verifier: 33 PASS, 0 FAIL; exception None; tempdir output sha256 `623ddbbdcaa684b4f7d70d525c5519d91466f3fd7ebff735dc6349c483d94986`
- occurrences in main.tex at cd57efc: [{'byte_start': 188871, 'byte_end': 223017, 'start_line': 2377, 'end_line': 2697, 'n_lines': 321, 'starts_at_line_start': True, 'ends_at_line_end': True}]
- = committed manuscript_review/appendixE_evaluation_record.tex: True; = verifier regeneration: True
- Appendix E block (comment rule to the line before Declaration): at cd57efc lines [2377, 2698]; at e5322f8 lines [2375, 2695]

## Table entries at cd57efc vs the frozen JSON

| table | rows | entries | MATCH | MISMATCH | not a frozen field, agrees | not a frozen field, disagrees |
|---|---|---|---|---|---|---|
| E.20 | 58 | 639 | 581 | 0 | 58 | 0 |
| E.21 | 61 | 854 | 732 | 0 | 122 | 0 |
| E.22 | 8 | 18 | 18 | 0 | 0 | 0 |
| E.23 | 7 | 49 | 49 | 0 | 0 | 0 |
| E.24 | 9 | 63 | 63 | 0 | 0 | 0 |

Every MISMATCH / disagreement:

- none

Cross-check: e5322f8 under the same convention, non-agreeing entries: []

## Entry diff e5322f8 -> cd57efc

Table cells compared: 1637; changed: 0

- row_set_changes: 0
- captions: 0
- notes: 1
  - E.21 l. [2580] -> [2583]: [{'op': 'replace', 'old': 'Evaluations~1, 2, 40', 'new': 'Evaluations~1 and~2 continue earlier runs, and evaluations~40'}, {'op': 'replace', 'old': 'continue', 'new': 'are re-decided from the records of'}]
- group_headers: 0
- other_lines: 0

## Appendix E block line diff (hunks, classified)

| # | class | section | lines old | lines new | token changes |
|---|---|---|---|---|---|
| 1 | comment | block header | [2381, 2381] | [2383, 2384] | [{'op': 'insert', 'old': '', 'new': 'Round 7 % (Addendum 80): E.4 second phase and the E.21 note on evaluations 40 and 41.'}] |
| 2 | E.21 note | \subsection{\textcolor{blue}{Evaluations}} | [2580, 2580] | [2583, 2583] | [{'op': 'replace', 'old': 'Evaluations~1, 2, 40', 'new': 'Evaluations~1 and~2 continue earlier runs, and evaluations~40'}, {'op': 'replace', 'old': 'continue', 'new': 'are re-decided from the records of'}] |
| 3 | prose | \subsection{\textcolor{blue}{Static Arrangements: Solver St… | [2621, 2621] | [2624, 2624] | [{'op': 'replace', 'old': 're-solve', 'new': 're-evaluation'}, {'op': 'replace', 'old': 'voltage', 'new': 'voltage,'}, {'op': 'insert', 'old': '', 'new': 'decisions held at the first phase and its voltage, thermal and'}] |

Generator output W180a -> W181a: 3 hunk(s), same +/- lines as the block diff: True

## Mutation test

| id | edit | caught by frozen check | caught by diff |
|---|---|---|---|
| M1 | E.20 row 3 col 3 (Delta gross (kEUR)): '208.3' -> '208.4' | True | True |
| M2 | E.21 row 1 col 8 (Band): '4.2' -> '4.3' | True | True |
| M3 | E.21 row 16 col 4 (Status): 'superseded$^{s}$' -> 'cert., osc.$^{s}$' | True | True |
| M4 | E.21 row 16 col 13 (Rule): '---' -> 'record' | True | True |
| M5 | E.22 row 19 col 5 (x): '3.40' -> '3.41' | True | True |
| M6 | E.23 row Passive NRF\|cold col 3 (NRF n): '29' -> '30' | True | True |
| M7 | E.24 row 42 col 9 (Resid.): '0.692' -> '0.693' | True | True |
| M8 | E.21 row 40 col 13 (Rule): 'record' -> 'run' | True | True |
| M9 | E.20 row 34 col 9 (Verdict net): 'det.' -> 'w.r.' | True | True |
| M10 | E.23 caption: 'after the consistency pass.' -> 'before the consistency pass.' | not applicable (not a frozen entry) | True |
| M11 | E.21 caption: 'within its recorded cycles' -> 'within its cycles' | not applicable (not a frozen entry) | True |
| M12 | E.22 notes: '51.3~k\\euro{}' -> '51.4~k\\euro{}' | True | True |
| M13 | E.21 notes: 'are re-decided from the records of earlier runs' -> 'continue earlier runs' | not applicable (not a frozen entry) | True |

## Integrity checks

- main_tex_read_equals_git_show_HEAD: True
- letter_read_equals_git_show_HEAD: True
- diff_comparison_negative_control_caught: True
- diff_parse_counts_consistent: True
- main_tex_lines_2713: True
- prev_main_tex_lines_2710: True
- generator_ran: True
- verifier_ran_without_exception: True
- e21_61_rows_numbered_1_61: True
- e21_map_one_to_one_onto_frozen_cells: True
- e20_58_rows_numbered_1_58: True
- mapping_problems_none: True
- e21_map_same_at_prev: True
- mutation_test_every_edit_caught: True
- mutation_test_at_least_five_single_entry_edits: True
