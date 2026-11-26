# W164 -- number check of the manuscript .tex files

Overleaf clone `manuscript/6a67305f25e8348fb71380c3` at commit `6191c6cc805cffd27122986d885a8714ef0540e4` (declared `6191c6c`); files: `cover_letter.tex` sha256 `363ad821`, `highlights.tex` sha256 `d568f329`, `main.tex` sha256 `3cedbb6e`, `response_to_reviewers_draft.tex` sha256 `8df94963`.
Frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe`). Script `p515_s53_w164_manuscript_number_check.py` (imports `p515_s53_w163_paragraphs_v4_check.py`, not edited). ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.

Statuses: match (declared / rule against a named record / auto-unique / auto-ambiguous), MISMATCH, approximate, no table counterpart, submitted-version figure (main.tex only, cites the revision-map line), unchecked (with the reason). Excluded LaTeX structure is counted separately and is not in scope.

## Counts per file

| file | tokens | excluded (structure) | in scope (body) | comments | match declared | match rule | match auto-unique | match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | unchecked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cover_letter.tex | 5 | 3 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| highlights.tex | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| main.tex | 2057 | 219 | 1809 | 29 | 22 | 83 | 0 | 0 | 0 | 0 | 2 | 610 | 1121 |
| response_to_reviewers_draft.tex | 299 | 10 | 287 | 2 | 86 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 201 |

Every token assigned: **True**; stale declarations: []; tokens claimed twice: [].

## MISMATCH (all files)

- `response_to_reviewers_draft.tex` line 256 col 48: written `one` -- counterpart '2' (T1 tables.claims (gross_verdict != net verdict) and T4 tables.year_ladder (gross_v6.verdict != net_v6.verdict): the comparisons whose verdict depends on the convention). the letter calls the investment-year comparison "the one result that depends on the convention"; T1 carries a second: L:y2030__n5_p0.25_e0.5__n7_p1_e3 (gross within the uncertified bar, net determinate), the verdict the preceding sentence counts. Reading: "result" = a reported comparison

## Approximate

none

## No table counterpart

- `main.tex` line 1058: `60` -- the frozen tables carry minimum SoH 0.70 (T8) and 0.50 (T10 soh050 row) only; no 0.60 / 0.80 evaluation exists in them. This red note (main.tex l. 1058) is not cited by the revision map
- `main.tex` line 1058: `80` -- the frozen tables carry minimum SoH 0.70 (T8) and 0.50 (T10 soh050 row) only; no 0.60 / 0.80 evaluation exists in them. This red note (main.tex l. 1058) is not cited by the revision map
- `response_to_reviewers_draft.tex` line 360: `single` -- no committed record states the machine count: W163 listed "one machine" as unchecked, NO SOURCE FOUND (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json unchecked_tokens); the v6 spec records concurrency 1 (one evaluation at a time, W163 N44) and W159 the environment of one host

## Response letter: every number with its source

| line | written | scope | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|---|
| 1 | 2026-10-06 | comment | match | LC1 | 2026-10-06 | STEP6_REVISION_MAP.md line 3 (header: "Expert's plan for the author, 2026-10-06") |
| 5 | 590088fe | comment | match | LC2 | 590088fe | sha256 of data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json |
| 24 | 1 | body | unchecked | enumerator |  | revision number (title block) |
| 33 | three | body | match | L01 | 3 | the letter itself: \section*{Reviewer n} blocks |
| 35 | two | body | match | L02 | 2 | the letter itself: \paragraph blocks before Reviewer 1 |
| 38 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 38 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 40 | 0.50 | body | match | L03 | 0.50 | main.tex at the declared Overleaf commit, line 1140: 'to a relative optimality gap of 0.50\\%' (the submitted text; main.tex is still the submitted version) |
| 42 | 2.2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 44 | 2.2.7 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 45 | 3.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 48 | 64.4 | body | match | L04 | 64.4 | T1 tables.claims[claim_id=CHECK:headline_V_minus_I_settled].d_gross (k EUR, absolute value) |
| 49 | 4.2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.2 |
| 50 | 4.3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 50 | 18.25 | body | match | L05 | 18.25 | main.tex at the declared Overleaf commit, line 144: 'by 18.25\\% and 92.16\\%, respectively' (the submitted text; main.tex is still the submitted version) |
| 52 | 90.9 | body | match | L06 | 90.9 | T6 tables.benchmark.benefit (M EUR) |
| 52 | 13.9 | body | match | L06 | 13.9 | T6 tables.benchmark.benefit_relative (%) |
| 53 | 4.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.4 |
| 57 | Two | body | unchecked | L07 |  | count of the requests the paragraph names (Reviewer 1 attribution; Reviewer 3 re-optimisation); the third sentence (the 5 x 5 instance) is not a reviewer request |
| 58 | 1 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 60 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 65 | five | body | match | L08 | five | main.tex at the declared Overleaf commit, line 915: 'is represented by five years (2025, 2028, 2031, 2034, and 2037)' (the submitted text; main.tex is still the submitted version) |
| 66 | twenty-five | body | match | L08 | 25 | main.tex at the declared Overleaf commit, line 937: 'resulting in 25 operating scenarios per representative day' (the submitted text; main.tex is still the submitted version) |
| 67 | 3 | body | match | L09 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 67 | 3 | body | match | L09 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 68 | 4.5 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 71 | 1 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 73 | R1.2 | body | unchecked | enumerator |  | reviewer item label |
| 78 | 18.25 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C); the figure appears in submitted main.tex lines [144, 1179, 1212, 1397] |
| 78 | 92.16 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C); the figure appears in submitted main.tex lines [144, 1185, 1227, 1397] |
| 78 | 18.25 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C); the figure appears in submitted main.tex lines [144, 1179, 1212, 1397] |
| 78 | 92.16 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C); the figure appears in submitted main.tex lines [144, 1185, 1227, 1397] |
| 78 | 8th | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 78 | 10th | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 78 | 13th | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 78 | 14th | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 80 | one | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 86 | R1.4 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 87 | 3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 89 | 3.4 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 91 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 91 | 3 | body/rresponse | match | L10 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 91 | 3 | body/rresponse | match | L10 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 92 | 0.909 | body/rresponse | match | L11 | 0.909 | T11 tables.three_by_three.R_range_derived[0] (lower end, post-hoc settled descent) |
| 92 | 0.934 | body/rresponse | match | L11 | 0.934 | T11 tables.three_by_three.R_range_derived[1] (upper end, at certification) |
| 92 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 93 | 0.933 | body/rresponse | match | L12 | 0.933 | T11 tables.three_by_three.rows[quantity=R predicted from the mean-profile spread].value (0.9331) |
| 93 | 4.5 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 93 | 9 | body/rresponse | match | L13 | 9 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case9_2025': 9, 'case9_2030': 9, 'case9_2035': 9} |
| 94 | three | body/rresponse | match | L13 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DistributionNetworks (count) |
| 94 | 33 | body/rresponse | match | L13 | 33 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case33_1_2025': 33, 'case33_1_2030': 33, 'case33_1_2035': 33, 'case33_2_2025': 33, 'case33_2_2030': 33, 'case33_2_2035': 33, 'case33_3_ |
| 94 | 108 | body/rresponse | match | L13 | 108 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) 9 + 3 x 33 per year {'2025': 108, '2030': 108, '2035': 108} |
| 96 | three | body/rresponse | match | L14 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} (count of keys) |
| 97 | five | body/rresponse | match | L14 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} (every block 5 years) |
| 98 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 98 | T1 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 98 | T6 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 101 | 4.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 103 | 3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 103 | 4.5 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 104 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 106 | R1.4 | body | unchecked | enumerator |  | reviewer item label |
| 120 | 0 | body/rresponse | match | L15 | 0 | T7 tables.discount[0].rate (%) |
| 120 | 2 | body/rresponse | match | L15 | 2 | T7 tables.discount[1].rate (%) |
| 120 | 5 | body/rresponse | match | L15 | 5 | T7 tables.discount[2].rate (%) |
| 120 | 8 | body/rresponse | match | L15 | 8 | T7 tables.discount[3].rate (%) |
| 121 | -40.8 | body/rresponse | match | L16 | -40.8 | T7 tables.discount[0].value_minus_I (k EUR) at rate 0.0 |
| 121 | -64.4 | body/rresponse | match | L16 | -64.4 | T7 tables.discount[1].value_minus_I (k EUR) at rate 0.02 |
| 121 | -92.7 | body/rresponse | match | L16 | -92.7 | T7 tables.discount[2].value_minus_I (k EUR) at rate 0.05 |
| 121 | -114.6 | body/rresponse | match | L16 | -114.6 | T7 tables.discount[3].value_minus_I (k EUR) at rate 0.08 |
| 122 | T7 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 122 | 2 | body/rresponse | match | L17 | 2 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DiscountFactor (%) |
| 125 | 4.6 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 125 | T7 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 127 | R1.5 | body | unchecked | enumerator |  | reviewer item label |
| 127 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 128 | 2 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 132 | 2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 135 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 137 | R2.1 | body | unchecked | enumerator |  | reviewer item label |
| 137 | R2.9 | body | unchecked | enumerator |  | reviewer item label |
| 139 | two | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 139 | two | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 139 | two | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 139 | single | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 140 | one | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 142 | two | body/rresponse | unchecked | not a figure |  | compound adjective (two-stage) |
| 143 | one | body/rresponse | unchecked | L18 |  | article sense ("one ... plan" = a single plan) |
| 145 | single | body/rresponse | unchecked | not a figure |  | article sense ("a single ...") |
| 145 | 2.1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 147 | single | body/rresponse | unchecked | not a figure |  | article sense ("a single ...") |
| 150 | 2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 150 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 150 | 2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 150 | 2.2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 150 | 5 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 152 | R2.2 | body | unchecked | enumerator |  | reviewer item label |
| 154 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 154 | 0.25 | body/rresponse | match | L19 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 155 | 0.5 | body/rresponse | match | L19 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 155 | 2 | body/rresponse | match | L19 | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 155 | 4 | body/rresponse | match | L19 | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 157 | One | body/rresponse | unchecked | L20 |  | one evaluation per candidate (the evaluation cache keyed on the candidate); an algorithm statement, audited by the equation/algorithm audit (map section B 2.1), not a table figure |
| 158 | one | body/rresponse | unchecked | L21 |  | one local problem per network block per cycle (algorithm statement; audited by the equation/algorithm audit, map section B Appendix A) |
| 158 | three | body/rresponse | match | L21 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 159 | four | body/rresponse | match | L21 | 4 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 159 | four | body/rresponse | match | L21 | 4 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 159 | single | body/rresponse | unchecked | L21 |  | compound adjective (single-scenario) |
| 160 | 48 | body/rresponse | match | L21 | 48 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks (3 x 4 x 4) |
| 161 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 162 | 4.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 163 | 2.2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 163 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 163 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 165 | R2.3 | body | unchecked | enumerator |  | reviewer item label |
| 165 | R2.4 | body | unchecked | enumerator |  | reviewer item label |
| 165 | R2.7 | body | unchecked | enumerator |  | reviewer item label |
| 173 | 2.2 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 176 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 178 | 2.2 | body/rchanges | unchecked | L22 |  | revised-manuscript section number (map section B 2.2) |
| 178 | 2.2.7 | body/rchanges | match | L22 | 2.2.7 | main.tex section structure (subsubsection "Benders' Cuts" at line 684; numbered by its position) |
| 178 | 2.2.7 | body/rchanges | match | L23 | 2.2.7 | STEP6_REVISION_MAP.md line 96: "replace by \"2.2.7 Recourse evaluation and certification\"" |
| 179 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 181 | R2.5 | body | unchecked | enumerator |  | reviewer item label |
| 181 | R2.6 | body | unchecked | enumerator |  | reviewer item label |
| 188 | three | body/rresponse | match | L24 | 3 | W163 N7 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.e |
| 190 | one | body/rresponse | match | L25 | >= 1 period | W163 N9 (W_MIN 20) and N10 (W_FACTOR 1.1): W = max(20, ceil(1.1 P_hat)) >= 1.1 P_hat > P_hat, i.e. at least one measured period |
| 191 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 192 | 42 | body/rresponse | match | L26 | 42 | W163 C5 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 cells with certification_stats_included; W163 status match, written '42', value at written precision '42'; paragraphs_v5.md line 65 ( |
| 192 | 32 | body/rresponse | match | L26 | 32 | W163 C6 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2; W163 status match, written '32', value at written precision '32'; paragraphs_v5.md line 65 (097421f8) |
| 193 | 10 | body/rresponse | match | L26 | 10 | W163 N30 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 cells with certification_stats_included, status uncertified (W153 totals.n_uncertified); W163 status match, written '10', value at w |
| 194 | 0.9 | body/rresponse | match | L27 | max 0.882 τ | W163 N25 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run p |
| 195 | 4.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 195 | T2 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 198 | two | body/rresponse | unchecked | not a figure |  | compound adjective (two-phase) |
| 203 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 206 | three | body/rresponse | match | L28 | 3 | T6 tables.benchmark.w160_additions.arms_in_full.{passive,price_taker}.Q_by_start (count of starts) |
| 206 | T6 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 208 | 2.2.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 208 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 208 | T2 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 210 | R2.8 | body | unchecked | enumerator |  | reviewer item label |
| 211 | 10.1016/j.est.2024.114911 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 211 | 10.1109/ISGTEUROPE62998.2024.10863557 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 211 | 10.1016/j.apenergy.2022.120569 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 212 | three | body/rresponse | match | L29 | 3 | the letter itself: DOIs in the R2.8 comment |
| 212 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 214 | one | body/rresponse | unchecked | L30 |  | [AUTHOR: ...] instruction, not manuscript text |
| 215 | 10.1016/j.est.2024.114911 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 215 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 216 | 10.1109/ISGTEUROPE62998.2024.10863557 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 216 | 10.1016/j.apenergy.2022.120569 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 217 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 219 | R2.10 | body | unchecked | enumerator |  | reviewer item label |
| 226 | 21 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 226 | 28 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 227 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 229 | R2.11 | body | unchecked | enumerator |  | reviewer item label |
| 229 | R2.12 | body | unchecked | enumerator |  | reviewer item label |
| 242 | two | body/rresponse | unchecked | L31 |  | pronoun ("the two" = the network models and the storage agent) |
| 242 | 2.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 243 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 246 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 248 | R3.1 | body | unchecked | enumerator |  | reviewer item label |
| 250 | 2.2.1 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 1 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 2034 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 2037 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 15 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 2037 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 2.4 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 2 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 250 | 4.1 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 254 | T1 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 254 | T4 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 254 | 60 | body/rresponse | match | L32 | 60 | T1 tables.claims (count) |
| 255 | one | body/rresponse | match | L32 | 1 | T1 tables.claims: gross_verdict != net verdict (count) |
| 256 | one | body/rresponse | MISMATCH | L33 | 2 | T1 tables.claims (gross_verdict != net verdict) and T4 tables.year_ladder (gross_v6.verdict != net_v6.verdict): the comparisons whose verdict depends on the convention |
| 256 | 2035 | body/rresponse | match | L34 | 2035 | T4 tables.year_ladder.per_year; data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 257 | 2030 | body/rresponse | match | L34 | 2030 | T4 tables.year_ladder.per_year; data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 257 | +42.5 | body/rresponse | match | L34 | 42.5 | T4 tables.year_ladder.D_gross (k EUR, 2035 - 2030) |
| 258 | -2.3 | body/rresponse | match | L35 | -2.3 | T4 tables.year_ladder.D_net (k EUR) |
| 258 | 4.6 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 259 | 2 | body/rresponse | unchecked | L36 |  | Table [2] of the revised manuscript (placeholder; the reviewer cites "Table 2 in Section 4.1" of the submitted version) |
| 260 | 2.2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 260 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 260 | 4.6 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 260 | T1 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 260 | T4 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 262 | R3.2 | body | unchecked | enumerator |  | reviewer item label |
| 262 | 0.50 | body | match | L37 | 0.50 | main.tex at the declared Overleaf commit, line 1140: 'to a relative optimality gap of 0.50\\%' (the submitted text; main.tex is still the submitted version) |
| 264 | 2.2.7 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 11 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 4.3 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 26 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 27 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 29 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 11 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 0.50 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C); the figure appears in submitted main.tex lines [1140] |
| 264 | 3 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 15 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 4.4 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 264 | 15 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 268 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 269 | 4 | body/rresponse | match | L38 | 4 | W163 N18 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula); W163 status match, written '4', value at written precision '4'; p |
| 269 | 0.07 | body/rresponse | match | L38 | 0.07 | W163 N19 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R; W163 status match, written '0.07', value at written precision '0.07';  |
| 269 | single | body/rresponse | unchecked | L38 |  | compound adjective (single-scenario) |
| 270 | two | body/rresponse | match | L39 | 2 | W163 N22 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T11: R = V / V_SRP1_settled (a ratio of two values); W163 status match, written 'two', value at written precision '2'; paragraphs_v5.md |
| 270 | 15 | body/rresponse | unchecked | L40 |  | figure number of the submitted version: main.tex at this commit has 18 figure environments (outside comments); its figure 15 by environment order is at line 1912; the submitted numbering is not re-derived here |
| 273 | 2.2.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 273 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 273 | 4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 275 | R3.3 | body | unchecked | enumerator |  | reviewer item label |
| 277 | 21 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 277 | 31 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 277 | 2.3.2 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 283 | R2.11 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 283 | R2.12 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 287 | 2.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 294 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 296 | R3.4 | body | unchecked | enumerator |  | reviewer item label |
| 298 | 2.3.2 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 298 | 22 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 298 | 23 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 300 | R2.10 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 301 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 303 | R3.5 | body | unchecked | enumerator |  | reviewer item label |
| 305 | 2.3.2 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 305 | 22 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 305 | 25 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 305 | 15 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 305 | 3 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 305 | two | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 305 | 15 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 307 | 0.985 | body/rresponse | match | L41 | 0.985 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) ageing.calendar_retention_per_year; data/SRP1/Results/P515S53/frozen_s53_spec_v41_fcea4b38.json (sha256 fcea4b38, last commit 477dba3e) configuration.identical_to_w104.ess_ageing_ |
| 308 | C2 | body/rresponse | unchecked | enumerator |  | ageing calibration name (T8 arm labels C2 / C2_calfade) |
| 308 | C2 | body/rresponse | unchecked | enumerator |  | ageing calibration name (T8 arm labels C2 / C2_calfade) |
| 309 | 31.6 | body/rresponse | match | L42 | 31.6 | T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR, absolute value) |
| 309 | 64.4 | body/rresponse | match | L42 | 64.4 | T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR, absolute value) |
| 310 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 310 | 0.5 | body/rresponse | match | L43 | 0.5 | main.tex at the declared Overleaf commit, line 1062: 'annual rates of 0.5\\% and 2.0\\% are additionally considered' (the submitted text; main.tex is still the submitted version) |
| 310 | 2 | body/rresponse | match | L43 | 2.0 | main.tex at the declared Overleaf commit, line 1062: 'annual rates of 0.5\\% and 2.0\\% are additionally considered' (the submitted text; main.tex is still the submitted version) |
| 312 | 3.4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 312 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 312 | T8 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 314 | R3.6 | body | unchecked | enumerator |  | reviewer item label |
| 316 | 4.6 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 316 | 5 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 319 | 7 | body/rresponse | match | L44 | 7 | T1 tables.claims[claim_id=E:n7_4h_e1_C2_calfade:value_minus_I].instance (the non-zero node of the unit) |
| 320 | -4.1 | body/rresponse | match | L45 | -4.1 | T8 tables.ageing.rows[arm=no_ageing].value_minus_I (k EUR) |
| 320 | -31.6 | body/rresponse | match | L45 | -31.6 | T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR) |
| 321 | -64.4 | body/rresponse | match | L45 | -64.4 | T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR) |
| 321 | -73.7 | body/rresponse | match | L45 | -73.7 | T8 tables.ageing.rows[arm=C3_unit].value_minus_I (k EUR) |
| 321 | -45.2 | body/rresponse | match | L45 | -45.2 | T8 tables.ageing.rows[arm=C4].value_minus_I (k EUR) |
| 322 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 322 | 0.70 | body/rresponse | match | L46 | 0.70 | W163 N38 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label "soh_min 0.70"); W163 status match, written '0.70', val |
| 322 | 0.50 | body/rresponse | match | L46 | 0.50/0.50 | W163 N41 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): A64 e_soh050 r2 campaign spec (28c9478d) candidates[0].settling_resettle.minimum_soh and extra.minimum_soh (the override; configuration |
| 323 | +4.9 | body/rresponse | match | L46 | 4.9 | T10 tables.a64.rows[claim_id=E:soh050:delta_value_vs_070].d_gross (k EUR) |
| 324 | T10 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 327 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 331 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 331 | T8 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 331 | T10 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 333 | R3.7 | body | unchecked | enumerator |  | reviewer item label |
| 333 | R3.8 | body | unchecked | enumerator |  | reviewer item label |
| 335 | 3 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 335 | 8 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 335 | 0.50 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C); the figure appears in submitted main.tex lines [1140] |
| 336 | 3 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 336 | 8 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 336 | 4.4.1 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 336 | four | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 336 | 8760 | body/rcomment | unchecked | reviewer quotation |  | reviewer quotation (the reviewers' documents are not in the repository, map section C) |
| 338 | one | body/rresponse | match | L47 | 1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.md T7 caption ("one discount factor per representative year applied to the five years of its block") |
| 339 | five | body/rresponse | match | L47 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 340 | 2025 | body/rresponse | match | L48 | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.md T7 caption ("I paid in 2025"); T7 tables.discount[*].I identical at every rate |
| 342 | 2.2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 342 | 3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 342 | 4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 349 | five | body | match | L49 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 355 | 1 | body | match | L50 | 1 | T6 tables.benchmark.sweep.sweep_passive_cold.n_blocks |
| 355 | 12 | body | match | L50 | 12 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): years x days |
| 355 | 8 | body | match | L50 | 8 | T6 tables.benchmark.sweep.sweep_price_taker_cold.n_blocks |
| 355 | 12 | body | match | L50 | 12 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): years x days |
| 356 | 4.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.4 |
| 358 | 10^{-5} | body | match | L51 | 1e-05 | W163 N33 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_helpers.py curtaillable pg upper bound pg_avail + EQUALITY_TOLERANCE; W109 |
| 360 | single | body | no table counterpart | L52 |  | no committed record states the machine count: W163 listed "one machine" as unchecked, NO SOURCE FOUND (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json unchecked_tokens); the v6 spec records concurrency 1 (one evaluation  |
| 360 | single | body | match | L52 | thread caps 1 on every table evaluation | W163 N45 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W159 c1: harness THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VECLIB / NUMEXPR = 1); all_table_evaluations_ran_with_OMP_NUM_THREADS_1;  |
| 362 | 49 | body | match | L53 | 49 | data/SRP1/Results/P515S53/w160_step6_frozen/export/paragraphs_v2.md (sha256 73b42c49, last commit 2a1d7f92) section (v) prediction scorecard: rows numbered 1..n |

## Response letter: findings (sentences against a table, a record or the letter itself)

- **F1** (line 254-256, wording against a table): "one verdict (a Phase~B neighbour becomes determinate in the certificate's favour)": the one T1 claim whose verdict changes gross -> net is L:y2030__n5_p0.25_e0.5__n7_p1_e3 (family L: "F2 certificate: F(y2030__n5_p0.25_e0.5__n7_p1_e3) - F(incumbent) (positive = worse than th..."), an F2-certificate neighbour of the m = 2 Phase B incumbent, both cells uncertified; T1 labels "Phase B certificate" its C-family rows (against x = 0), which do not change. Count and direction match; the label is ambiguous.
- **F2** (line 256, claim against a table): "The investment-year comparison is the one result that depends on the convention": T4 2035 - 2030 (gross determinate, net within resolution) is one; T1 carries a second, L:y2030__n5_p0.25_e0.5__n7_p1_e3 (gross within the uncertified bar, net determinate), which the preceding sentence itself counts. Token status MISMATCH (count 2 against "one").
- **F3** (line 327-328, claim about a table): "Table [T8] gives, per calibration, the year in which the end-of-life floor binds and the terminal available-energy fraction, together with the equivalent full cycles per day": T8 carries "floor binds (0.70)", "AE (PV-weighted)" and "EFC/day (PV-weighted)" -- the AE column is present-value-weighted, not terminal (no terminal available-energy column in T8).
- **F4** (line 212-216, internal contradiction): "The three references were added to the literature review" against the bracketed status in the same response: 10.1109/ISGTEUROPE62998.2024.10863557 and 10.1016/j.apenergy.2022.120569 "not yet added" (author placeholder).
- **F5** (line 320-322, wording against a table): "harsher calibrations -73.7 and -45.2 k EUR": C4 (-45.2) loses less than the baseline C2_calfade (-64.4); it is harsher than C2 (cycling only, -31.6), not than the baseline. T8's sixth arm C3_midblock (-65.9 k EUR, determinate) is not named. Every named value matches T8.
- **F6** (line 158-160, wording against a record): "one local problem per network block (three representative years x four representative days x four agents ..., 48 blocks)": 3 x 4 x (1 TSO + 3 DSO) = 48 network blocks matches SRP1.json; the storage-operator (ESSO) problems solved in the same cycle are not counted (TASKS.md W132 / W133 lines: 51 exits per cycle including the ESSO; not re-read from the solve records here).
- **F7** (line 127-132, not verifiable here): "Figure 2" (R1.5) is answered as the framework figure. In main.tex at this commit the framework figure is the first figure environment (line 387, blue caption) and the second is the network diagram (line 924); the numbering of the submitted PDF the reviewer read cannot be re-derived from this file.
- **F8** (line 24, 99-100, internal inconsistency): the title block says "revision 1" while R1.2 says "we respectfully maintain the position of our first reply"; one of the two is wrong unless an earlier reply exists.
- **F9** (line 360, no record): "single machine": no committed record states it (W163 listed "one machine" as NO SOURCE FOUND).
- **K1** (line 195-197, claim checked, consistent): "the limitations section explains the mechanism behind most of them (a set-valued interface dual ...)": 6 of the 10 uncertified SRP1 evaluations are T9 dead-zone entries (gap-refused or by signature: ['h_f9eae48f', 'j_5f3cccb4', 'l_0ee93aca', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5']); the others ['d_36686489', 'd_a12d95a2', 'd_f759dd48', 'j_f3aa335e'] are not.
- **K2** (line 253-255, claim checked, consistent): "It changes no sign in the 60 reported comparisons": sign changes gross -> net in T1: 0.

## highlights.tex: every number with its source

no in-scope numeric token (structure only: `12pt` (package/class option or argument), `1in` (package/class option or argument))

## cover_letter.tex: every number with its source

- line 16 `four`: match -- the cover letter itself: \item entries of the list that follows
- line 24 `15`: match -- data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years (sum of block lengths)

## main.tex: submitted-version figures by section

| section | n | map lines cited |
|---|---:|---|
| Front matter | 3 | 117, 23 |
| 3 Case Study | 77 | 114, 117, 120, 126 |
| 4 Results | 245 | 137, 159, 165, 169 |
| 5 Conclusions | 2 | 23 |
| E Results | 283 | 193 |

- **Front matter** (3): l.144 2037, l.144 18.25, l.144 92.16
- **3 Case Study** (77): l.915 five, l.915 2028, l.915 2031, l.915 2034, l.915 2037, l.915 three, l.937 Three, l.937 five, l.937 five, l.937 25, l.954 2025, l.954 2028, l.954 2031, l.954 2034, l.954 2037, l.956 1, l.956 35.00, l.956 214.17, l.956 186.90, l.956 165.76, l.956 156.88, l.956 148.01, l.957 2, l.957 55.00, l.957 267.53, l.957 241.56, l.957 220.79, l.957 210.42, l.957 200.04, l.958 3, l.958 10.00, l.958 342.17, l.958 303.72, l.958 276.19, l.958 270.48, l.958 264.77, l.960 1, l.960 35.00, l.960 169.71, l.960 148.10, l.960 131.35, l.960 124.31, l.960 117.28, l.961 2, l.961 55.00, l.961 211.99, l.961 191.41, l.961 174.95, l.961 166.73, l.961 158.51, l.962 3, l.962 10.00, l.962 271.13, l.962 240.67, l.962 218.85, l.962 214.33, l.962 209.80, l.984 2028, l.984 2031, l.984 2034, l.984 2037, l.1008 2028, l.1008 2031, l.1008 2034, l.1008 2037, l.1025 2028, l.1025 2031, l.1025 2034, l.1025 2037, l.1042 2028, l.1042 2031, l.1042 2034, l.1042 2037, l.1062 1, l.1062 14, l.1062 0.5, l.1062 2.0
- **4 Results** (245): l.1071 two, l.1076 three, l.1083 2025, l.1083 2028, l.1083 2031, l.1083 2034, l.1083 2037, l.1085 7, l.1085 1, l.1085 2.50, l.1086 2, l.1086 1.36, l.1087 3, l.1088 1.62, l.1089 1, l.1089 5.00, l.1090 2, l.1090 2.71, l.1091 3, l.1092 3.24, l.1094 1.00, l.1094 0.00, l.1094 0.00, l.1094 0.00, l.1094 0.00, l.1100 2025, l.1101 7, l.1104 7, l.1108 7, l.1112 7, l.1115 2025, l.1115 2028, l.1115 2031, l.1115 2034, l.1115 2037, l.1117 2.62, l.1117 2.27, l.1117 1.99, l.1117 1.76, l.1117 1.61, l.1118 80.90, l.1118 69.88, l.1118 61.24, l.1118 54.41, l.1118 49.62, l.1124 7, l.1124 19.10, l.1124 three, l.1124 49.62, l.1125 2, l.1140 four, l.1140 0.50, l.1140 30,560, l.1147 two, l.1151 three, l.1151 four, l.1160 2025, l.1160 2028, l.1160 2031, l.1160 2034, l.1160 2037, l.1163 228,160.22, l.1163 259,520.80, l.1163 211,683.68, l.1163 247,772.49, l.1163 273,955.22, l.1164 917.65, l.1164 962.18, l.1164 1,348.91, l.1164 1,381.60, l.1164 1,314.22, l.1165 1.30, l.1165 2.77, l.1165 81.23, l.1165 54.34, l.1165 409.29, l.1168 222,309.25, l.1168 250,720.96, l.1168 202,001.02, l.1168 236,819.08, l.1168 228,540.87, l.1169 -2.56, l.1169 -3.39, l.1169 -4.57, l.1169 -4.42, l.1169 -16.58, l.1171 907.86, l.1171 947.99, l.1171 1,415.01, l.1171 1,422.13, l.1171 1,677.43, l.1172 -1.07, l.1172 -1.47, l.1172 4.90, l.1172 2.93, l.1172 27.64, l.1174 12.95, l.1174 17.82, l.1174 15.24, l.1174 13.80, l.1174 46.89, l.1175 894.07, l.1175 543.90, l.1175 -81.24, l.1175 -74.60, l.1175 -88.54, l.1178 222,393.22, l.1178 249,768.49, l.1178 202,309.71, l.1178 239,441.31, l.1178 223,948.57, l.1179 -2.53, l.1179 -3.76, l.1179 -4.43, l.1179 -3.36, l.1179 -18.25, l.1181 907.86, l.1181 947.97, l.1181 1,414.95, l.1181 1,410.36, l.1181 1,692.24, l.1182 -1.07, l.1182 -1.48, l.1182 4.90, l.1182 2.08, l.1182 28.76, l.1184 12.94, l.1184 17.81, l.1184 15.28, l.1184 25.57, l.1184 32.08, l.1185 893.89, l.1185 543.59, l.1185 -81.19, l.1185 -52.95, l.1185 -92.16, l.1192 2031, l.1198 2025, l.1198 3, l.1198 2025, l.1203 3, l.1203 2025, l.1203 three, l.1212 2037, l.1212 16.58, l.1212 18.25, l.1227 2031, l.1227 2037, l.1227 88.54, l.1227 92.16, l.1286 two, l.1286 3, l.1288 5, l.1289 1, l.1289 three, l.1289 four, l.1289 three, l.1292 1, l.1294 5, l.1294 single, l.1294 7, l.1294 two, l.1294 2025, l.1294 7, l.1294 1.54, l.1294 3.09, l.1294 one, l.1294 9, l.1294 0.10, l.1294 0.20, l.1298 5, l.1302 2025, l.1302 2030, l.1302 2035, l.1304 7, l.1304 1.54, l.1305 3.09, l.1307 9, l.1307 0.10, l.1308 0.20, l.1310 1.00, l.1310 0.00, l.1310 0.00, l.1316 two, l.1316 7, l.1316 50.80, l.1316 49.62, l.1316 3, l.1316 9, l.1320 5, l.1323 2025, l.1323 2030, l.1323 2035, l.1325 7, l.1325 2.27, l.1325 1.83, l.1325 1.57, l.1326 73.48, l.1326 59.15, l.1326 50.80, l.1328 9, l.1328 0.13, l.1328 0.10, l.1328 0.07, l.1329 60.76, l.1329 50.91, l.1329 38.43, l.1335 1, l.1335 single, l.1335 1.62, l.1335 3.24, l.1335 7, l.1335 2025, l.1339 1, l.1342 2025, l.1342 2026, l.1342 2027, l.1342 2039, l.1344 7, l.1344 1.62, l.1345 3.24, l.1347 1.00, l.1347 0.00, l.1347 0.00, l.1347 0.00, l.1355 1, l.1359 2025, l.1359 2026, l.1359 2027, l.1359 2037, l.1359 2038, l.1359 2039, l.1361 3.16, l.1361 3.07, l.1361 2.96, l.1361 2.13, l.1361 2.06, l.1361 1.99, l.1362 97.56, l.1362 94.56, l.1362 91.42, l.1362 65.83, l.1362 63.65, l.1362 61.49, l.1375 four
- **5 Conclusions** (2): l.1397 18.25, l.1397 92.16
- **E Results** (283): l.1985 2025, l.1985 2028, l.1985 2031, l.1985 2034, l.1985 2037, l.1994 269671.54, l.1994 240359.85, l.1994 194686.35, l.1994 207466.96, l.1995 278745.92, l.1995 271843.10, l.1995 239972.27, l.1995 247310.67, l.1996 266821.42, l.1996 241056.10, l.1996 178945.77, l.1996 159305.51, l.1997 313102.26, l.1997 288405.93, l.1997 179448.50, l.1997 209415.34, l.1998 329528.87, l.1998 294022.70, l.1998 195895.18, l.1998 275763.44, l.1999 814.83, l.1999 875.19, l.1999 907.46, l.1999 1074.27, l.2000 799.59, l.2000 871.78, l.2000 1038.93, l.2000 1140.19, l.2001 1170.76, l.2001 1106.34, l.2001 1547.14, l.2001 1573.34, l.2002 1227.64, l.2002 1206.71, l.2002 1462.45, l.2002 1631.29, l.2003 1177.36, l.2003 927.30, l.2003 1831.44, l.2003 1322.29, l.2004 0.00, l.2004 5.22, l.2004 0.00, l.2004 0.00, l.2005 10.98, l.2005 0.00, l.2005 0.00, l.2005 0.00, l.2006 84.56, l.2006 240.34, l.2006 0.00, l.2006 0.00, l.2007 57.84, l.2007 159.49, l.2007 0.00, l.2007 0.00, l.2008 316.18, l.2008 697.68, l.2008 0.00, l.2008 624.33, l.2012 244253.20, l.2012 235518.58, l.2012 200807.07, l.2012 208417.00, l.2013 248518.89, l.2013 259801.27, l.2013 245425.14, l.2013 249162.75, l.2014 231386.32, l.2014 229083.91, l.2014 184500.25, l.2014 162710.69, l.2015 274169.86, l.2015 272988.69, l.2015 187791.61, l.2015 211915.70, l.2016 270509.42, l.2016 247965.68, l.2016 205275.47, l.2016 189951.70, l.2017 -9.43, l.2017 -2.01, l.2017 3.14, l.2017 0.46, l.2018 -10.84, l.2018 -4.43, l.2018 2.27, l.2018 0.75, l.2019 -13.28, l.2019 -4.97, l.2019 3.10, l.2019 2.14, l.2020 -12.43, l.2020 -5.35, l.2020 4.65, l.2020 1.19, l.2021 -17.91, l.2021 -15.66, l.2021 4.79, l.2021 -31.12, l.2023 795.04, l.2023 867.14, l.2023 894.16, l.2023 1076.33, l.2024 773.76, l.2024 856.18, l.2024 1025.98, l.2024 1137.93, l.2025 1218.67, l.2025 1335.05, l.2025 1538.91, l.2025 1569.59, l.2026 1259.00, l.2026 1353.06, l.2026 1451.31, l.2026 1626.96, l.2027 1405.13, l.2027 1537.88, l.2027 1825.09, l.2027 1944.63, l.2028 -2.43, l.2028 -0.92, l.2028 -1.47, l.2028 0.19, l.2029 -3.23, l.2029 -1.79, l.2029 -1.25, l.2029 -0.20, l.2030 4.09, l.2030 20.67, l.2030 -0.53, l.2030 -0.24, l.2031 2.56, l.2031 12.13, l.2031 -0.76, l.2031 -0.27, l.2032 19.35, l.2032 65.85, l.2032 -0.35, l.2032 47.07, l.2035 19.76, l.2035 13.29, l.2035 13.25, l.2035 5.41, l.2036 36.80, l.2036 15.56, l.2036 12.90, l.2036 5.79, l.2037 36.66, l.2037 12.18, l.2037 8.19, l.2037 3.70, l.2038 26.48, l.2038 13.21, l.2038 11.10, l.2038 4.29, l.2039 88.47, l.2039 87.80, l.2039 7.13, l.2039 3.72, l.2040 154.45, l.2041 235.22, l.2042 -56.65, l.2042 -94.93, l.2043 -54.22, l.2043 -91.72, l.2044 -72.02, l.2044 -87.42, l.2044 -99.40, l.2048 244140.90, l.2048 234804.95, l.2048 200439.69, l.2048 209948.35, l.2049 246055.36, l.2049 258220.48, l.2049 245809.26, l.2049 249029.65, l.2050 233458.63, l.2050 227967.81, l.2050 186566.71, l.2050 160903.39, l.2051 284765.82, l.2051 272862.03, l.2051 187353.93, l.2051 212285.38, l.2052 269546.16, l.2052 233501.93, l.2052 203314.34, l.2052 188930.78, l.2053 -9.47, l.2053 -2.31, l.2053 2.96, l.2053 1.20, l.2054 -11.73, l.2054 -5.01, l.2054 2.43, l.2054 0.70, l.2055 -12.50, l.2055 -5.43, l.2055 4.26, l.2055 1.00, l.2056 -9.05, l.2056 -5.39, l.2056 4.41, l.2056 1.37, l.2057 -18.20, l.2057 -20.58, l.2057 3.79, l.2057 -31.49, l.2059 795.03, l.2059 867.15, l.2059 894.17, l.2059 1076.34, l.2060 773.74, l.2060 856.20, l.2060 1025.95, l.2060 1137.90, l.2061 1218.68, l.2061 1334.90, l.2061 1538.84, l.2061 1569.55, l.2062 1212.29, l.2062 1353.10, l.2062 1451.27, l.2062 1626.96, l.2063 1405.29, l.2063 1597.11, l.2063 1825.09, l.2063 1944.64, l.2064 -2.43, l.2064 -0.92, l.2064 -1.47, l.2064 0.19, l.2065 -3.23, l.2065 -1.79, l.2065 -1.25, l.2065 -0.20, l.2066 4.09, l.2066 20.66, l.2066 -0.54, l.2066 -0.24, l.2067 -1.25, l.2067 12.13, l.2067 -0.76, l.2067 -0.27, l.2068 19.36, l.2068 72.23, l.2068 -0.35, l.2068 47.07, l.2070 19.77, l.2070 13.28, l.2070 13.25, l.2070 5.41, l.2071 36.82, l.2071 15.55, l.2071 12.87, l.2071 5.78, l.2072 36.63, l.2072 12.30, l.2072 8.21, l.2072 3.75, l.2073 73.17, l.2073 13.16, l.2073 11.14, l.2073 4.29, l.2074 88.31, l.2074 28.58, l.2074 7.12, l.2074 3.70, l.2075 154.19, l.2076 235.43, l.2077 -56.68, l.2077 -94.88, l.2078 26.49, l.2078 -91.75, l.2079 -72.07, l.2079 -95.90, l.2079 -99.41

## main.tex: statuses by section

| section | status | n |
|---|---|---:|
| (comments) | unchecked | 29 |
| Front matter / preamble | unchecked | 6 |
| Front matter | unchecked | 4 |
| Front matter | match (declared) | 5 |
| Front matter | submitted-version figure | 3 |
| 1 Introduction | unchecked | 8 |
| 2 Shared ESS Planning Framework | unchecked | 39 |
| 3 Case Study | match (declared) | 17 |
| 3 Case Study | unchecked | 135 |
| 3 Case Study | submitted-version figure | 77 |
| 3 Case Study | match (rule (named record)) | 16 |
| 3 Case Study | no table counterpart | 2 |
| 4 Results | submitted-version figure | 245 |
| 5 Conclusions | unchecked | 2 |
| 5 Conclusions | submitted-version figure | 2 |
| A TSO--DSO Coordinated Operational Planning | unchecked | 22 |
| B Market Data | match (rule (named record)) | 7 |
| C Transmission Network | match (rule (named record)) | 7 |
| D Distribution Networks | unchecked | 876 |
| D Distribution Networks | match (rule (named record)) | 53 |
| E Results | submitted-version figure | 283 |

## main.tex: auto-index matches


Tokens assigned another status by precedence although the auto index has >= 1 candidate: {'match': 143, 'unchecked': 729, 'submitted-version figure': 129, 'no table counterpart': 2}

## Unchecked: by category (all files)

| file | category | n |
|---|---|---:|
| main.tex | comment (not typeset) | 29 |
| main.tex | address | 6 |
| main.tex | not a figure | 17 |
| main.tex | enumerator | 2 |
| main.tex | descriptive count | 7 |
| main.tex | notation | 52 |
| main.tex | literature value | 3 |
| main.tex | network/data parameter | 1005 |
| response_to_reviewers_draft.tex | enumerator | 48 |
| response_to_reviewers_draft.tex | cross-reference | 74 |
| response_to_reviewers_draft.tex | structural | 1 |
| response_to_reviewers_draft.tex | reviewer quotation | 62 |
| response_to_reviewers_draft.tex | not a figure | 10 |
| response_to_reviewers_draft.tex | method statement | 2 |
| response_to_reviewers_draft.tex | author placeholder | 1 |
| response_to_reviewers_draft.tex | reference identifier | 3 |

## Excluded LaTeX structure

| file | category | n | tokens |
|---|---|---:|---|
| cover_letter.tex | package/class option or argument | 3 | 12pt, utf8, T1 |
| highlights.tex | package/class option or argument | 2 | 12pt, 1in |
| main.tex | package/class option or argument | 8 | 12pt, 1p, 3p, 5p |
| main.tex | figure option/filename | 44 | 1.00, 0.50, 0.65, 0.90, 0.95, 0.475 |
| main.tex | table structure | 145 | 2, 5, 4, 8, 3, 1, 7, 9 |
| main.tex | label/ref/cite key | 2 | two |
| main.tex | length/layout | 16 | 0.5em, 1.00, 5pt, 0.85, 1.40 |
| main.tex | environment option/argument | 4 | 0.90 |
| response_to_reviewers_draft.tex | package/class option or argument | 4 | 11pt, 1in, utf8, T1 |
| response_to_reviewers_draft.tex | macro definition | 3 | 1 |
| response_to_reviewers_draft.tex | macro parameter | 3 | 1 |

## Integrity checks

- clone_head_is_declared_commit: True
- file_set_equals_declared: True
- every_sha_equals_declared: True
- every_file_equals_its_blob_at_commit: True
- no_tex_modified_or_untracked_in_clone: True
- frozen_json_unchanged: True
- every_declaration_found_and_evaluated: True
- no_token_claimed_twice: True
- every_token_assigned: True
- every_map_citation_found_once: True
- statuses_in_vocabulary: True
- main_line_rules_pinned_to_this_main: True
