# W171a -- number check of the manuscript .tex files at Overleaf 42794d4 (declarations v2)

Overleaf clone `manuscript/6a67305f25e8348fb71380c3` at commit `42794d403fe701d6cbc606b2cedb948e17db1b1d` (declared `42794d4`); files: `cover_letter.tex` sha256 `363ad821`, `highlights.tex` sha256 `d568f329`, `main.tex` sha256 `7effd898`, `response_to_reviewers_draft.tex` sha256 `1af28d7a`, `section2_expert_draft.tex` sha256 `7e347dc8`.
Frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe`). Script `p515_s53_w171_manuscript_number_check.py` (imports `p515_s53_w164_manuscript_number_check.py` @ 77fcf136, not edited). Submitted source `manuscript_submitted/main.tex` (sha256 `ca07d7db`). Reviewers' document `manuscript_review/Reviewers Comments.docx` (sha256 `c7afdc8b`; zipfile + xml.etree (word/document.xml w:p / w:t / w:tab / w:br; python-docx not installed)). ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.

Statuses: match (declared / rule against a named record / auto-unique / auto-ambiguous), MISMATCH, approximate, no table counterpart, submitted-version figure (main.tex, cites the revision-map line), reviewer quotation, verified (letter quotations), unchecked (with the reason). Excluded LaTeX structure is counted separately.

## Counts per file -- manuscript

| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, verified | unchecked | unassigned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cover_letter.tex | 5 | 3 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| highlights.tex | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| main.tex | 2132 | 219 | 1877 | 36 | 63 | 83 | 0 | 0 | 3 | 0 | 2 | 610 | 0 | 1152 | 0 |
| response_to_reviewers_draft.tex | 311 | 10 | 299 | 2 | 89 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 62 | 147 | 0 |

## Counts -- draft, not compiled (`section2_expert_draft.tex`, not \input into main.tex; apart from the manuscript counts)

| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, verified | unchecked | unassigned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| section2_expert_draft.tex | 231 | 0 | 92 | 139 | 60 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 167 | 0 |

Every token assigned: **True**; stale declarations: []; tokens claimed twice: [].

## MISMATCH (all files)

| file | line | col | written | counterpart at written precision | source | note |
|---|---:|---:|---|---|---|---|
| main.tex | 426 | 23 | 2n | n+1 | p515_s47_phase_b_record.py:199 (sha256 97f573e5); p515_s47_phase_b_record.py:509 (sha256 97f573e5); p515_s51_f2_phase_b.py:803 (sha256 87ab3b2d) | Algorithm 1 writes "the 2n OrthoMADS directions"; both planning searches as run (Phase B and the F2 Phase B) poll n + 1 directions (the n Householder columns and minus their sum; STEP4 section 3 allows "2n ... or the n+1 minimal positive basis") |
| main.tex | 431 | 34 | 1 | unit poll | p515_s47_phase_b_record.py:607 (sha256 97f573e5); p515_s47_phase_b_record.py:204 (sha256 97f573e5); p515_s47_phase_b_record.py:205 (sha256 97f573e5) | Algorithm 1 adds the completion set "If /P/ < n + 1"; the code adds it at every poll of unit size (UNIT_POLL_COMPLETION and delta == DELTA_MIN), refusing (stop for review) above 30 points |
| main.tex | 625 | 37 | third | third-last to last | settling_criterion_v2.py:166 (sha256 3db13b0e) (inherited by settling_criterion_v6); W163 N11 written 'the three most recent turning points: from the third-last to the last' | "measured between the first and third turning points": the code measures P_hat between the third-last and the last turning point (the three most recent); the two agree only while exactly three turning points have occurred since k0 |
| response_to_reviewers_draft.tex | 251 | 81 | 0.5 |  | manuscript_submitted/main.tex (sha256 ca07d7db): fragment 'annual rates of 0.5\\% and 2.0\\% are additionally considered' not found | "submitted main.tex fragment 'annual rates of 0.5\\\\% and 2.0\\\\% are additionally considered': 0 hits". the letter says the SUBMITTED version announced these cases; the sentence is in the Overleaf main.tex (line 1053, first-reply text), not in the submitted source |
| response_to_reviewers_draft.tex | 251 | 93 | 2 |  | manuscript_submitted/main.tex (sha256 ca07d7db): fragment 'annual rates of 0.5\\% and 2.0\\% are additionally considered' not found | "submitted main.tex fragment 'annual rates of 0.5\\\\% and 2.0\\\\% are additionally considered': 0 hits". as the 0.5 token |
| section2_expert_draft.tex (draft) | 74 | 23 | 2n | n+1 | p515_s47_phase_b_record.py:199 (sha256 97f573e5); p515_s47_phase_b_record.py:509 (sha256 97f573e5); p515_s51_f2_phase_b.py:803 (sha256 87ab3b2d) | Algorithm 1 writes "the 2n OrthoMADS directions"; both planning searches as run (Phase B and the F2 Phase B) poll n + 1 directions (the n Householder columns and minus their sum; STEP4 section 3 allows "2n ... or the n+1 minimal positive basis") |
| section2_expert_draft.tex (draft) | 79 | 34 | 1 | unit poll | p515_s47_phase_b_record.py:607 (sha256 97f573e5); p515_s47_phase_b_record.py:204 (sha256 97f573e5); p515_s47_phase_b_record.py:205 (sha256 97f573e5) | Algorithm 1 adds the completion set "If /P/ < n + 1"; the code adds it at every poll of unit size (UNIT_POLL_COMPLETION and delta == DELTA_MIN), refusing (stop for review) above 30 points |
| section2_expert_draft.tex (draft) | 233 | 37 | third | third-last to last | settling_criterion_v2.py:166 (sha256 3db13b0e) (inherited by settling_criterion_v6); W163 N11 written 'the three most recent turning points: from the third-last to the last' | "measured between the first and third turning points": the code measures P_hat between the third-last and the last turning point (the three most recent); the two agree only while exactly three turning points have occurred since k0 |
| section2_expert_draft.tex (draft) | 496 | 31 | 22,430 | 22429 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C4 (= T8 C4 k 22429.386) | 8,000 x 1.0 / -ln 0.70 = 22,429.39 |

## Approximate / no table counterpart

- `main.tex` line 1049: `60` (no table counterpart) -- the frozen tables carry minimum SoH 0.70 (T8) and 0.50 (T10 soh050 row) only; no 0.60 / 0.80 evaluation exists in them. This red note (main.tex l. 1058) is not cited by the revision map
- `main.tex` line 1049: `80` (no table counterpart) -- the frozen tables carry minimum SoH 0.70 (T8) and 0.50 (T10 soh050 row) only; no 0.60 / 0.80 evaluation exists in them. This red note (main.tex l. 1058) is not cited by the revision map
- `response_to_reviewers_draft.tex` line 290: `single` (no table counterpart) -- no committed record states the machine count; stands on the author's attestation (Addendum 67; Addendum 68 Decision 5 "Single machine stands on the author's attestation")

## W164 declarations re-located (version 1 -> version 2)

| v1 id | file | fragment found (times) | outcome | v2 declaration | reason / what replaced it |
|---|---|---:|---|---|---|
| LC1 | response_to_reviewers_draft.tex | 1 | carried | LC1 | fragment found once; evaluated unchanged |
| LC2 | response_to_reviewers_draft.tex | 1 | carried | LC2 | fragment found once; evaluated unchanged |
| L01 | response_to_reviewers_draft.tex | 1 | carried | L01 | fragment found once; evaluated unchanged |
| L02 | response_to_reviewers_draft.tex | 1 | carried | L02 | fragment found once; evaluated unchanged |
| L03 | response_to_reviewers_draft.tex | 1 | superseded | L03v2 | a figure quoted from the submitted version: checked against the committed submitted source |
| L04 | response_to_reviewers_draft.tex | 1 | carried | L04 | fragment found once; evaluated unchanged |
| L05 | response_to_reviewers_draft.tex | 1 | superseded | L05v2 | as L03; the fragment of the submitted abstract differs from the Overleaf abstract |
| L06 | response_to_reviewers_draft.tex | 1 | carried | L06 | fragment found once; evaluated unchanged |
| L07 | response_to_reviewers_draft.tex | 1 | carried | L07 | fragment found once; evaluated unchanged |
| L08 | response_to_reviewers_draft.tex | 1 | superseded | L08v2 | as L03 |
| L09 | response_to_reviewers_draft.tex | 1 | carried | L09 | fragment found once; evaluated unchanged |
| L10 | response_to_reviewers_draft.tex | 1 | carried | L10 | fragment found once; evaluated unchanged |
| L11 | response_to_reviewers_draft.tex | 1 | carried | L11 | fragment found once; evaluated unchanged |
| L12 | response_to_reviewers_draft.tex | 1 | carried | L12 | fragment found once; evaluated unchanged |
| L13 | response_to_reviewers_draft.tex | 1 | carried | L13 | fragment found once; evaluated unchanged |
| L14 | response_to_reviewers_draft.tex | 1 | carried | L14 | fragment found once; evaluated unchanged |
| L15 | response_to_reviewers_draft.tex | 1 | carried | L15 | fragment found once; evaluated unchanged |
| L16 | response_to_reviewers_draft.tex | 1 | carried | L16 | fragment found once; evaluated unchanged |
| L17 | response_to_reviewers_draft.tex | 1 | carried | L17 | fragment found once; evaluated unchanged |
| L18 | response_to_reviewers_draft.tex | 1 | carried | L18 | fragment found once; evaluated unchanged |
| L19 | response_to_reviewers_draft.tex | 1 | carried | L19 | fragment found once; evaluated unchanged |
| L20 | response_to_reviewers_draft.tex | 1 | carried | L20 | fragment found once; evaluated unchanged |
| L21 | response_to_reviewers_draft.tex | 0 | removed | L21v2 | B.3: "four agents" -> "four network operators" and "and one per shared-ESS agent (three, one per interface node, ...)" added |
| L22 | response_to_reviewers_draft.tex | 1 | superseded | L22v2 | the former section number is read from the submitted source |
| L23 | response_to_reviewers_draft.tex | 1 | carried | L23 | fragment found once; evaluated unchanged |
| L24 | response_to_reviewers_draft.tex | 1 | carried | L24 | fragment found once; evaluated unchanged |
| L25 | response_to_reviewers_draft.tex | 1 | carried | L25 | fragment found once; evaluated unchanged |
| L26 | response_to_reviewers_draft.tex | 1 | carried | L26 | fragment found once; evaluated unchanged |
| L27 | response_to_reviewers_draft.tex | 1 | carried | L27 | fragment found once; evaluated unchanged |
| L28 | response_to_reviewers_draft.tex | 1 | carried | L28 | fragment found once; evaluated unchanged |
| L29 | response_to_reviewers_draft.tex | 1 | carried | L29 | fragment found once; evaluated unchanged |
| L30 | response_to_reviewers_draft.tex | 1 | carried | L30 | fragment found once; evaluated unchanged |
| L31 | response_to_reviewers_draft.tex | 1 | carried | L31 | fragment found once; evaluated unchanged |
| L32 | response_to_reviewers_draft.tex | 0 | removed | L32v2 | B.4: "and one verdict (a Phase~B neighbour ...)" -> "and two verdicts: a two-node plan evaluated under the doubled flexibility price ..." |
| L33 | response_to_reviewers_draft.tex | 0 | removed | L32v2 | B.4: "The investment-year comparison is the one result that depends on the convention" -> "and the investment-year comparison is the other comparison that depends on the convention" (no number token left; the count is checked by L32v2) |
| L34 | response_to_reviewers_draft.tex | 1 | carried | L34 | fragment found once; evaluated unchanged |
| L35 | response_to_reviewers_draft.tex | 1 | carried | L35 | fragment found once; evaluated unchanged |
| L36 | response_to_reviewers_draft.tex | 1 | carried | L36 | fragment found once; evaluated unchanged |
| L37 | response_to_reviewers_draft.tex | 1 | superseded | L37v2 | as L03 |
| L38 | response_to_reviewers_draft.tex | 1 | carried | L38 | fragment found once; evaluated unchanged |
| L39 | response_to_reviewers_draft.tex | 1 | carried | L39 | fragment found once; evaluated unchanged |
| L40 | response_to_reviewers_draft.tex | 1 | superseded | L40v2 | the figure order of the submitted source is now available |
| L41 | response_to_reviewers_draft.tex | 1 | carried | L41 | fragment found once; evaluated unchanged |
| L42 | response_to_reviewers_draft.tex | 1 | carried | L42 | fragment found once; evaluated unchanged |
| L43 | response_to_reviewers_draft.tex | 1 | superseded | L43v2 | as L03 |
| L44 | response_to_reviewers_draft.tex | 1 | carried | L44 | fragment found once; evaluated unchanged |
| L45 | response_to_reviewers_draft.tex | 0 | removed | L45v2 | B.5: "harsher calibrations -73.7 and -45.2 k EUR; all determinate except the first" -> the five aged arms named in order, with -65.9 added |
| L46 | response_to_reviewers_draft.tex | 1 | carried | L46 | fragment found once; evaluated unchanged |
| L47 | response_to_reviewers_draft.tex | 1 | carried | L47 | fragment found once; evaluated unchanged |
| L48 | response_to_reviewers_draft.tex | 1 | carried | L48 | fragment found once; evaluated unchanged |
| L49 | response_to_reviewers_draft.tex | 1 | carried | L49 | fragment found once; evaluated unchanged |
| L50 | response_to_reviewers_draft.tex | 1 | carried | L50 | fragment found once; evaluated unchanged |
| L51 | response_to_reviewers_draft.tex | 1 | carried | L51 | fragment found once; evaluated unchanged |
| L52 | response_to_reviewers_draft.tex | 1 | superseded | L52v2 | Addendum 68 Decision 5: "single machine" stands on the author's attestation |
| L53 | response_to_reviewers_draft.tex | 1 | carried | L53 | fragment found once; evaluated unchanged |
| CL1 | cover_letter.tex | 1 | carried | CL1 | fragment found once; evaluated unchanged |
| CL2 | cover_letter.tex | 1 | carried | CL2 | fragment found once; evaluated unchanged |
| M01 | main.tex | 1 | carried | M01 | fragment found once; evaluated unchanged |
| M02 | main.tex | 1 | carried | M02 | fragment found once; evaluated unchanged |
| M03 | main.tex | 1 | carried | M03 | fragment found once; evaluated unchanged |
| M04 | main.tex | 1 | carried | M04 | fragment found once; evaluated unchanged |
| M05 | main.tex | 1 | carried | M05 | fragment found once; evaluated unchanged |
| M06 | main.tex | 1 | carried | M06 | fragment found once; evaluated unchanged |
| M07 | main.tex | 1 | carried | M07 | fragment found once; evaluated unchanged |
| M17 | main.tex | 1 | carried | M17 | fragment found once; evaluated unchanged |
| M08 | main.tex | 1 | carried | M08 | fragment found once; evaluated unchanged |
| M09 | main.tex | 1 | carried | M09 | fragment found once; evaluated unchanged |
| M10 | main.tex | 1 | carried | M10 | fragment found once; evaluated unchanged |
| M11 | main.tex | 1 | carried | M11 | fragment found once; evaluated unchanged |
| M12 | main.tex | 1 | carried | M12 | fragment found once; evaluated unchanged |
| M13 | main.tex | 1 | carried | M13 | fragment found once; evaluated unchanged |
| M14 | main.tex | 1 | carried | M14 | fragment found once; evaluated unchanged |
| M15 | main.tex | 1 | carried | M15 | fragment found once; evaluated unchanged |
| M16 | main.tex | 1 | carried | M16 | fragment found once; evaluated unchanged |

## Reviewer quotations against the reviewers' document

Document: `manuscript_review/Reviewers Comments.docx` sha256 `c7afdc8b`, word/document.xml sha256 `c054751b`, 32 paragraphs. Normalisation on both sides: LaTeX quotes and Word curly quotes -> straight, \% -> %, ~ and runs of white space -> one space, math delimiters and command names dropped; enumerate items and \\ start a new segment (the document interleaves the authors' earlier replies).

| quote | segment | letter lines | chars | result | match ratio | docx paragraph | start | differences (letter -> document) |
|---:|---:|---|---:|---|---:|---:|---|---|
| 0 | 0 | 63 | 51 | verbatim | 1.000 | 0 | Regarding the Abstract, the following are relevant |  |
| 0 | 1 | 65 | 246 | verbatim | 1.000 | 0 | The Authors could tell the reader a little more ab |  |
| 0 | 2 | 66 | 116 | verbatim | 1.000 | 3 | The size of the test systems on which the framewor |  |
| 0 | 3 | 67 | 476 | differs | 0.994 | 5 | The lines below could also be improved: "Relative  | delete l.67: ...d also be improved: ['"' -> '']Relative to uncoordi...; delete l.67: ...g voltage violations['."' -> ''] Does 18.25% refer t... |
| 0 | 4 | 68 | 84 | verbatim | 1.000 | 7 | What type of battery ESS is the reference? This co |  |
| 0 | 5 | 69 | 294 | verbatim | 1.000 | 9 | The battery ESS is shared by both the Transmission |  |
| 0 | 6 | 70 | 97 | verbatim | 1.000 | 11 | Improving the Abstract taking into consideration t |  |
| 1 | 0 | 87 | 197 | verbatim | 1.000 | 13 | Several uncertainties will accompany the evolution |  |
| 1 | 1 | 89 | 88 | verbatim | 1.000 | 13 | It could be good for this type of Paper to elabora |  |
| 1 | 2 | 90 | 97 | verbatim | 1.000 | 15 | Additionally, could some sensitivity analysis be u |  |
| 2 | 0 | 98 | 47 | verbatim | 1.000 | 16 | Figure 2 could be improved in size for clarity. |  |
| 3 | 0 | 107 | 797 | verbatim | 1.000 | 16 | The claimed "bi-level" formulation is not clearly  |  |
| 3 | 1 | 108 | 638 | verbatim | 1.000 | 22 | The scenario treatment is conceptually problematic |  |
| 4 | 0 | 121 | 613 | verbatim | 1.000 | 22 | The interaction between ADMM, Benders decompositio |  |
| 5 | 0 | 127 | 648 | verbatim | 1.000 | 22 | The Benders cuts are not mathematically justified. |  |
| 5 | 1 | 128 | 508 | verbatim | 1.000 | 22 | The source of the sensitivity coefficients is uncl |  |
| 5 | 2 | 129 | 534 | verbatim | 1.000 | 22 | The feasibility cuts are not valid as written. The |  |
| 6 | 0 | 143 | 603 | verbatim | 1.000 | 22 | The ADMM convergence claim is weak for the stated  |  |
| 6 | 1 | 144 | 474 | verbatim | 1.000 | 22 | The adaptive ADMM penalty update is not justified. |  |
| 7 | 0 | 171 | 227 | verbatim | 1.000 | 22 | The literature review must include foundational re |  |
| 8 | 0 | 181 | 684 | verbatim | 1.000 | 22 | The degradation model is too simplified for the st |  |
| 9 | 0 | 191 | 612 | verbatim | 1.000 | 22 | The ESS active/reactive/apparent power relationshi |  |
| 9 | 1 | 192 | 576 | verbatim | 1.000 | 22 | The shared ESS formulation does not visibly contai |  |
| 10 | 0 | 210 | 1099 | verbatim | 1.000 | 23 | Regarding the master-problem objective in Section  |  |
| 11 | 0 | 218 | 1254 | verbatim | 1.000 | 24 | Regarding the Benders optimality cut in Section 2. |  |
| 12 | 0 | 231 | 924 | differs | 0.999 | 24 | Regarding Eqs. (21)-(31) in Section 2.3.2 and Appe | replace l.231: ... generic function h(['.' -> '⋅']), which is insuffic... |
| 13 | 0 | 239 | 1137 | verbatim | 1.000 | 26 | Regarding the degradation formulation in Section 2 |  |
| 14 | 0 | 246 | 940 | verbatim | 1.000 | 27 | Regarding the degradation model in Section 2.3.2,  |  |
| 15 | 0 | 257 | 801 | verbatim | 1.000 | 28 | Regarding insight (iii) in Section 4.6 and the rel |  |
| 16 | 0 | 265 | 820 | verbatim | 1.000 | 29 | Fig. 3 reports the Benders objective in , whereas  |  |
| 16 | 1 | 266 | 865 | verbatim | 1.000 | 30 | Regarding the representative-day setting in Sectio |  |

Quotation tokens: {'reviewer quotation, verified': 62}

## Revised section 2 of main.tex (l. 315-898): every number with its source

| line | written | status | check | counterpart (at written precision) | source / reason |
|---:|---|---|---|---|---|
| 319 | two | unchecked | not a figure |  | compound adjective (two-stage) |
| 319 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 379 | 4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 4 exists ("Results") |
| 388 | two | unchecked | not a figure |  | compound adjective (two-stage) |
| 393 | two | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 412 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 412 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 414 | -1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 416 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 419 | single | unchecked | not a figure |  | compound adjective (single-node) |
| 426 | 2n | MISMATCH | S219 | n+1 | p515_s47_phase_b_record.py:199 (sha256 97f573e5); p515_s47_phase_b_record.py:509 (sha256 97f573e5); p515_s51_f2_phase_b.py:803 (sha256 87ab3b2d) |
| 431 | 1 | MISMATCH | S225 | unit poll | p515_s47_phase_b_record.py:607 (sha256 97f573e5); p515_s47_phase_b_record.py:204 (sha256 97f573e5); p515_s47_phase_b_record.py:205 (sha256 97f573e5) |
| 432 | one | match | S220 | one | p515_s47_phase_b_record.py:46 (sha256 97f573e5) (COMPLETION_RULE) |
| 441 | 2 | match | S226 | 2 | p515_s47_phase_b_record.py:719 (sha256 97f573e5); STEP4_DFO_METHOD.md:190 (sha256 d05bae1e) |
| 443 | 1 | match | S227 | 1 | p515_s47_phase_b_record.py:195 (sha256 97f573e5); STEP4_DFO_METHOD.md:123 (sha256 d05bae1e) |
| 444 | 2 | match | S228 | 2 | p515_s47_phase_b_record.py:730 (sha256 97f573e5); STEP4_DFO_METHOD.md:190 (sha256 d05bae1e) |
| 451 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 451 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 458 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 472 | 1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 496 | +1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 503 | +1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 542 | 2 | unchecked | M07 |  | cited typical duration range (\cite{nrel_ess_costs}); the model bounds are E/P in [2.0, 4.0] h (SRP1_ESS_Params) |
| 542 | 10 | unchecked | M07 |  | cited typical duration range (\cite{nrel_ess_costs}); the model bounds are E/P in [2.0, 4.0] h (SRP1_ESS_Params) |
| 544 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 544 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 544 | 0.25 | match | S201 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 544 | 0.5 | match | S201 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 544 | 2 | match | S201 | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 544 | 4 | match | S201 | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 544 | 3 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 3 exists ("Case Study") |
| 555 | 1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 607 | 1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 620 | 10^{-5} | match | S202 | 1e-05 | W163 N2 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): data/SRP1/SRP1_params.json admm.tol.boyd.eps_abs; W163 status match, written '10⁻⁵', value at written precision '1e-05' |
| 620 | 10^{-4} | match | S202 | 0.0001 | W163 N3 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): data/SRP1/SRP1_params.json admm.tol.boyd.eps_rel; W163 status match, written '10⁻⁴', value at written precision '0.0001' |
| 620 | 15 | match | S203 | 15 | W163 N4 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€), measured from the old residual-rule certificate N; W163 status match,  |
| 620 | 21 | match | S203 | 21 | W163 N5 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed (k€), measured from the old residual-rule certificate N; W163 status m |
| 624 | three | match | S204 | 3 | W163 N7 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.e |
| 625 | third | MISMATCH | S205 | third-last to last | settling_criterion_v2.py:166 (sha256 3db13b0e) (inherited by settling_criterion_v6); W163 N11 written 'the three most recent turning points: from the third-last to the last' |
| 626 | 10 | match | S206 | 10 | W163 N8 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.SWING_FLOOR = TAU / 10 (= GROWTH_TEST_FLOOR = TURNING_POINT_FLOOR); v6 spec stop_rule.swing_floor.F; W163 status m |
| 628 | 20 | match | S207 | 20 | W163 N9 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.W_MIN; v6 spec stop_rule.W.oscillatory; W163 status match, written '20', value at written precision '20' |
| 628 | 1.1 | match | S207 | 1.1 | W163 N10 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.W_FACTOR; W163 status match, written '1.1', value at written precision '1.1' |
| 631 | 2 | match | S208 | 2 | W163 N12 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND); W163 status match, written '2', value at written precision '2' |
| 633 | four | match | S209 | 4 | W163 N13 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v5.METRICS (v6) and v6 spec stop_rule.clean_rule.metric_table; W163 status match, written 'four', value at written p |
| 634 | ten | match | S209 | 10 | W163 N14 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v5.CLEAN_FACTOR (v6); v6 spec stop_rule.clean_rule.factor; PRIMARY_ATTEMPT; W163 status match, written '10', value a |
| 638 | 2P_ | match | S210 | 60 | W163 N15 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec stop_rule.p_max.L and stop_rule.W.monotone ("L = L_MONO = 2 * P_MAX = 60"); W163 status match, written '2 P_max = 60', value at |
| 638 | 2P_ | match | S211 | 60 | W163 N17 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec stop_rule.constants.MONOTONE_LAST_STEP_CLAUSE "abs(dQ_k) * L_MONO <= TAU" with L_MONO 60; settling_criterion_v2 last_step_times |
| 639 | single | unchecked | not a figure |  | compound adjective (single-scenario) |
| 639 | ten | match | S230 | 10 | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0) = 10; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles = 10; shared_resources_planning.py:3390 (sha256 0610d74 |
| 639 | 4.5 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 4.5 exists ("Impact of Planning Horizon Discretization") |
| 641 | 4 | match | S212 | 4 | W163 N18 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula); W163 status match, written '4', value at written precision '4' |
| 641 | 0.07 | match | S212 | 0.07 | W163 N19 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R; W163 status match, written '0.07', value at written precision '0.07' |
| 641 | 259,375.33 | match | S213 | 259375.33 / 259375.33 | W163 N20 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.R_REF; W101 summary expert_P2.V_old (the SRP1 value Q(x0, N) − Q(unit, N) at the certificates in force when v39 froz |
| 641 | 4,539.07 | match | S213 | 4539.07 | W163 C1 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): constants.TAU; W163 status match, written '4,539.07', value at written precision '4539.07' |
| 641 | two | match | S214 | 2 | W163 N22 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T11: R = V / V_SRP1_settled (a ratio of two values); W163 status match, written 'two', value at written precision '2' |
| 641 | four | match | S214 | 4 | W163 N21 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): TAU divisor 4 (N18); T11 rows: V = Q(0) − Q(unit) (two evaluations per value) and R = V / V_SRP1 (two values): 2 × 2 = 4 evaluations; W |
| 641 | ten | match | S215 | 10 | W163 C4 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 at_or_above_0.95_tau_counted (count of true); W163 status match, written '10', value at written precision '10' |
| 641 | 5 | match | S215 | 5 | W163 N24 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): frozen constants.FLAG_RANGE_OVER_TAU / registry threshold 0.95: 1 − 0.95; W163 status match, written '5', value at written precision '5 |
| 641 | 0.9 | match | S216 | max 0.882 τ | W163 N25 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run p |
| 641 | 4.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 4.7 DOES NOT EXIST |
| 643 | two | match | S217 | 2 bars | W163 N73 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the difference of two certified cells, one bar each; W163 status mat |
| 643 | 3 | match | S217 | 3 | W163 N28 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.DETERMINACY_BAR_FACTOR; determinacy_threshold behaviour; W163 status match, written '3', value at written precisi |
| 643 | 2 | match | S217 | 2 | W163 N29 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.DETERMINACY_TAU_MULTIPLE; determinacy_threshold(100, 200) = 2 TAU; W163 status match, written '2', value at writt |
| 643 | two | match | S217 | 2 bars | W163 N73 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the difference of two certified cells, one bar each; W163 status mat |
| 643 | three | match | S218 | 3 × max(1000, 2000) = 6000 | W163 N26 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): p515_s53_w132_resettle_v3_campaign.resolve (the scorer): bar = 3.0 × max(gap, slack) over the uncertified cell(s); behaviour on a synth |
| 648 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 656 | two | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 670 | 4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 4 exists ("Results") |
| 684 | +1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 686 | +1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 691 | R2.11 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 691 | R2.12 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 691 | R3.1 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 691 | R3.3 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 701 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 701 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 705 | 1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 720 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 721 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 722 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 730 | 2 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 730 | 2 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 730 | 2 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 744 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 745 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 745 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 745 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 745 | 3.5 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 3.5 DOES NOT EXIST |
| 747 | W169 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 758 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 763 | single | match | S229 | single | T1 tables.claims[*].instance -> candidate_canonical.investment_year (instance dict, or tables.cells / cells_appended_w160 for a list instance) |
| 780 | 365 | match | S221 | 365 | shared_energy_storage_data.py:665 (sha256 9acd095f) |
| 795 | 365 | match | S222 | 365 | shared_energy_storage_data.py:703 (sha256 9acd095f) |
| 796 | 2 | match | S223 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 802 | 2 | match | S224 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 802 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 806 | 1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 808 | -1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 808 | 1 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 830 | 3.4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 3.4 exists ("Active Distribution Networks") |
| 854 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 854 | 2 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 854 | 2 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 854 | 2 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 854 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 854 | 3.5 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 3.5 DOES NOT EXIST |
| 857 | W169 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 858 | 3 | unchecked | comment (not typeset) |  | [CONFIRM] / instruction comment in revised section 2; comments are not manuscript text |
| 873 | 0 | unchecked | notation |  | formula constant in revised section 2 (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 897 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 897 | one | unchecked | method statement |  | number word in revised section 2 describing structure (no value to check; the statement is audited against the code by W171b) |
| 897 | single | unchecked | not a figure |  | article sense ("a single ...") |
| 897 | single | unchecked | not a figure |  | compound adjective (single-scenario) |
| 897 | 3.5 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; main.tex at this commit: section 3.5 DOES NOT EXIST |

## Parameter values in force (Addendum 68 list; section 2 defers them to "Section 3.5", not yet in main.tex) -- each checked against its source

| id | quantity | stated (Addendum 68) | found | status | sources | note |
|---|---|---|---|---|---|---|
| P01 | eta_ch (shared ESS, network models and agent) | 0.97 | 0.97 | **match** | shared_energy_storage.py:14 (sha256 d45c72d0); model_construction_helpers.py:969 (sha256 a20a224f); shared_energy_storage_data.py:655 (sha256 9acd095f) | class default; assignments to .eff_ch/.eff_dch in production modules: [('network.py', 1186), ('network.py', 1187)] (local-ESS loader only: True) |
| P02 | eta_dch | 0.96 | 0.96 | **match** | shared_energy_storage.py:15 (sha256 d45c72d0) |  |
| P03 | SoC^Min (fraction of E^Av) | 0.1 | 0.1 | **match** | definitions.py:39 (sha256 7e5719af); model_construction_helpers.py:894 (sha256 a20a224f) | shared_es_e_rated_fixed = e_available / s_base (shared_resources_planning, every cycle): the fraction applies to the degraded available energy |
| P04 | SoC^Max | 0.9 | 0.9 | **match** | definitions.py:38 (sha256 7e5719af); model_construction_helpers.py:902 (sha256 a20a224f) |  |
| P05 | SoC^0 (initial = closure target) | 0.5 | 0.5 | **match** | definitions.py:40 (sha256 7e5719af); model_construction_helpers.py:973 (sha256 a20a224f); model_construction_helpers.py:987 (sha256 a20a224f) |  |
| P06 | closure slack bound eps^Cl (fraction of E^Av) | 0.05 | 0.05 | **match** | model_construction_helpers.py:410 (sha256 a20a224f); model_construction_helpers.py:1166 (sha256 a20a224f); data/SRP1/case9/case9_params.json:22,26 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:22,26 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:22,26 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:22,26 (sha256 a19bd5b9) | slacks.shared_ess.day_balance per case file: {'case9': True, 'case33_1': True, 'case33_2': True, 'case33_3': True} |
| P07 | closure slack numerical term | 1e-05 | 1e-05 | **match** | definitions.py:90 (sha256 7e5719af) | in model units: per unit at baseMVA 100 (the slack and e_capacity are e_available / s_base), i.e. 1e-3 MWh -- STEP6_ROUND1_CORRECTIONS A.3(e) writes "10^-5 p.u." |
| P08 | closure slack penalty c^Cl (EUR/MWh) | 1000.0 | 1000.0 | **match** | definitions.py:58 (sha256 7e5719af); model_construction_helpers.py:2341 (sha256 a20a224f) | objective term base * 1e3 * slack_pu with base = network.baseMVA (100): 1e3 per MWh of slack |
| P09 | network complementarity eps^C (normalised) | 0.0001 | 0.0001 | **match** | definitions.py:91 (sha256 7e5719af); definitions.py:92 (sha256 7e5719af); model_construction_helpers.py:951 (sha256 a20a224f) | shared_ess_model per case file: {'case9': 'BILINEAR_RELAXATION', 'case33_1': 'BILINEAR_RELAXATION', 'case33_2': 'BILINEAR_RELAXATION', 'case33_3': 'BILINEAR_RELAXATION'}; hat = P / S_rated_fixed (hat-link rows) |
| P10 | eps^E (ESSO throughput regularisation) | 1e-05 | 1e-05 | **match** | definitions.py:74 (sha256 7e5719af); shared_energy_storage_data.py:844 (sha256 9acd095f) |  |
| P11 | c^sigma (ESSO P-net slack penalty) | 1000.0 | 1000.0 | **match** | definitions.py:63 (sha256 7e5719af); shared_energy_storage_data.py:829 (sha256 9acd095f); data/SRP1/SharedESS/SRP1_ESS_Params.json:6 (sha256 39106f93) | ESS params slacks = True |
| P12 | S_ref (MVA); normalisation 2 S_ref | 2.5 (2 S_ref = 5) | 2.5 | **match** | data/SRP1/SRP1_params.json:50 (sha256 dbfdb2a0); shared_resources_planning.py:5028 (sha256 0610d745); shared_resources_planning.py: 12 lines divide by (2 * shared_ess_rating) | floor 0.1 MVA inactive while S_ref is set (_shared_ess_admm_normalization_mva returns reference_mva) |
| P13 | rho initial v / pf / ess | 0.0077 / 0.198 / 0.01 | {'v': {'case9': 0.0077, 'case33_1': 0.0077, 'case33_2': 0.0077, 'case33_3': 0.0077}, 'pf': {'case9': 0.198, 'case33_1': 0.198, 'case33_2': 0.198, 'case33_3': 0.198}, 'ess': {'case9': 0.01, 'case33_1': 0.01, 'case33_2': 0.01, 'case33_3': 0.01, 'esso': 0.01}} | **match** | data/SRP1/SRP1_params.json:67 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:73 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:83 (sha256 dbfdb2a0) | every agent (case9, case33_1..3; ess also the ESSO) |
| P14 | residual balancing: ratio 5 (pf decrease 3), x/÷1.5, clamp [1e-4, 1e4] | 5 / 3 / 1.5 / 1.5 / 1e-4 / 1e4 | {'residual_balance_ratio': 5.0, 'residual_balance_ratio_pf_decrease': 3.0, 'increase_factor': 1.5, 'decrease_factor': 1.5, 'min': 0.0001, 'max': 10000.0} | **match** | data/SRP1/SRP1_params.json:53 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:54 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:55 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:56 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:57 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:58 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:48 (sha256 dbfdb2a0) |  |
| P15 | per-channel freeze after 10 unchanged cycles; backstop 200; ESS exempt until dual ratio < 1 on 5 cycles | 10 / 200 / (< 1, 5) | {'freeze_after_unchanged_cycles': 10, 'freeze_backstop_cycle': 200, 'ess_exempt': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}} | **match** | data/SRP1/SRP1_params.json:59 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:60 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:62 (sha256 dbfdb2a0) |  |
| P16 | Boyd eps_abs / eps_rel | 1e-5 / 1e-4 | {'eps_abs': 1e-05, 'eps_rel': 0.0001} | **match** | data/SRP1/SRP1_params.json:42 (sha256 dbfdb2a0); admm_parameters.py:363 (sha256 40d0e972) |  |
| P17 | production exit: consecutive passing cycles | 10 | {'SRP1': 10, '3x3_spec': 10} | **match** | data/SRP1/SRP1_params.json:45 (sha256 dbfdb2a0); shared_resources_planning.py:3390 (sha256 0610d745); data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) required_consecutive_cycles |  |
| P18 | cycle caps: 3x3 500; SRP1 gated N_old + 100, ungated min(k0 + 109, 300) | 500 / N_old + 100 / min(k0 + 109, 300) | {'3x3_cap': 500, 'v6_gated_cells': 30, 'v6_ungated_cells': 4, 'CAP_AFTER_K0': 109} | **match** | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) cap; data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells[*].cap_rule (gated: formula, cap = N_old + 100, per-cell ceiling; ungated: after_first_k0, ceiling); settling_criterion_v2.py:76 (sha256 3db13b0e) | gated caps are N_old + 100 with a per-cell ceiling (v6 configuration note "gated N_old + 100 (per-cell ceiling), ungated 300"); gated cells above 300: {'l_2ab0ce2d': 437, 'l_b2251bc5': 320} |
| P19 | sigma (common objective scale) | 93635360 | 93635360.0 | **match** | data/SRP1/SRP1_params.json:49 (sha256 dbfdb2a0) |  |
| P20 | kappa_ESSO = sigma / median w_b | 227,210.997 (SRP1); 386,258.694 (3x3) | {'SRP1': 227210.996652, 'SRP1_median_w_b': 412.10751847261173, 'SRP1_n_blocks': 48, '3x3': 386258.694309, '3x3_median_w_b': 242.4161873368304, '3x3_n_blocks': 80} | **match** | data/SRP1/SRP1_params.json:51 (sha256 dbfdb2a0); shared_resources_planning.py:4062 (sha256 0610d745); shared_resources_planning.py:4037 (sha256 0610d745); data/SRP1/SRP1.json (sha256 61a794a7); data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) | recomputed here from the instance files with the code's block set (TSO + 3 DSOs) and weight |
| P21 | w_b = Y_y D_d 1.02^-(y - y0) | Y_y D_d 1.02^-(y-y0) | {'SRP1_DiscountFactor': 0.02, '3x3_DiscountFactor': 0.02} | **match** | shared_resources_planning.py:3872 (sha256 0610d745); shared_resources_planning.py:3873 (sha256 0610d745); data/SRP1/SRP1.json (sha256 61a794a7); data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) |  |
| P22 | AA: type-II, memory 5, Tikhonov 1e-10, ratchet safeguard, keep_memory, cleared on rho change and on a solve failure, off when every channel passes | II / 5 / 1e-10 / ratchet / keep_memory / clears / off | {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory', 'type_II': True, 'ratchet_safeguard': True, 'cleared_on_rho_change': True, 'cleared_on_solve_failure': True, 'off_when_every_channel_passes': True} | **match** | data/SRP1/SRP1_params.json:123 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:124 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:125 (sha256 dbfdb2a0); admm_anderson_acceleration.py:3 (sha256 24cf3bea); admm_anderson_acceleration.py:398 (sha256 24cf3bea); admm_anderson_acceleration.py:442 (sha256 24cf3bea); admm_anderson_acceleration.py:477 (sha256 24cf3bea) |  |
| P23 | tail compl_inf_tol 1e-6 (production: TSO 5e-4, DSO 1e-4) | 1e-6 / 5e-4 / 1e-4 | {'v6_tail': {'compl_inf_tol': 1e-06, 'enabled': True}, '3x3_tail': {'compl_inf_tol': 1e-06, 'enabled': True}, 'tso_case_file': 0.0005, 'dso_case_files': {'case33_1': None, 'case33_2': None, 'case33_3': None}, 'ipopt_default_in_network_py': 0.0001} | **match** | data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) inputs_in_force_now.configuration_now.convergence_depth_tail; data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) configuration.convergence_depth_tail; data/SRP1/case9/case9_params.json:41 (sha256 f3eff050); network.py:632 (sha256 18acaa84) | DSO production value = the IPOPT default (no case33 file sets it) |
| P24 | ESSO IPOPT tol / acceptable_tol / linear solver | 1e-10 / 1e-9 / MA57 | {'override': '1e-10 / 1e-9', 'file_tol': 1e-06, 'file_acceptable_tol': 1e-05, 'linear_solver': 'ma57'} | **match** | shared_energy_storage_data.py:1082 (sha256 9acd095f); shared_energy_storage_data.py:108 (sha256 9acd095f); shared_energy_storage_data.py:1104 (sha256 9acd095f); data/SRP1/SharedESS/SRP1_ESS_Params.json:37 (sha256 39106f93) | the override replaces the file tol 1e-6 / acceptable 1e-5 for every ESSO solve (applied after the file options) |
| P25 | networks IPOPT tol / acceptable / MA97 / max_iter / recovery acceptable_tol & acceptable_iter | 1e-5 / 1e-4 / MA97 / 500 / 1e-4 & 1 | {'options': {'case9': {'tol': 1e-05, 'acceptable_tol': 0.0001, 'linear_solver': 'ma97'}, 'case33_1': {'tol': 1e-05, 'acceptable_tol': 0.0001, 'linear_solver': 'ma97'}, 'case33_2': {'tol': 1e-05, 'acceptable_tol': 0.0001, 'linear_solver': 'ma97'}, 'case33_3': {'tol': 1e-05, 'acceptable_tol': 0.0001,  | **partial** | data/SRP1/case9/case9_params.json:37 (sha256 f3eff050); data/SRP1/case33_1/case33_1_params.json:37 (sha256 8e9e5a53); data/SRP1/case33_2/case33_2_params.json:37 (sha256 31b5fedf); data/SRP1/case33_3/case33_3_params.json:37 (sha256 a19bd5b9); network.py:559 (sha256 18acaa84); network.py:972 (sha256 18acaa84) | recovery acceptable_tol 1e-4 / acceptable_iter 1 is set for the TSO only (case9_params.json); case33_1 has no recovery_options and case33_2 / case33_3 an empty one: a DSO recovery is the cold restart with the primary options (network.py recovery_options built from the case file) |
| P26 | proximal gamma (TSO; DSO off) | 0 | {'tso': {'enabled': True, 'gamma_policy': 'tied_to_rho', 'tau': 0.0}, 'dso_enabled': False} | **match** | data/SRP1/SRP1_params.json:95 (sha256 dbfdb2a0); data/SRP1/SRP1_params.json:96 (sha256 dbfdb2a0); shared_resources_planning.py:5081 (sha256 0610d745) | gamma = tau * rho with tau = 0 -> gamma = 0 on every channel |
| P27 | row 18 alpha (3x3 only) | 0.5 | ['{"alpha": 0.5, "floor": null}'] | **match** | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) candidates[*].interface_deviation_premium |  |
| P28 | interface-voltage pin weight (3x3 only; solver-only) | 90000.0 | 90000.0 | **match** | definitions.py:75 (sha256 7e5719af); model_construction_helpers.py:1940 (sha256 a20a224f); shared_resources_planning.py:849 (sha256 0610d745) |  |
| P29 | baseMVA (TN and DNs) | 100 | [100.0] | **match** | case files data/SRP1/case9/case9_<y>.json and data/SRP1/case33_<n>/case33_<n>_<y>.json at 2025/2030/2035: baseMVA | 12 files |
| P30 | lattice units and duration bounds | 0.25 MVA / 0.5 MWh / 2-4 h | {'p_step': 0.25, 'e_step': 0.5, 'step4_p': True, 'step4_e': True, 'step4_duration': True, 'ep_min': 2.0, 'ep_max': 4.0} | **match** | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) min/max_energy_to_power_factor |  |
| P31 | storage agents per cycle (one per interface node, all years and days) | 3 | 3 | **match** | shared_energy_storage_data.py:96 (sha256 9acd095f); data/SRP1/SRP1.json (sha256 61a794a7) DistributionNetworks |  |

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
| 40 | 0.50 | body | match | L03v2 | 0.50 | manuscript_submitted/main.tex (sha256 ca07d7db) line 866: 'to a relative optimality gap of 0.50\\%' |
| 42 | 2.2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 44 | 2.2.7 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 45 | 3.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 48 | 64.4 | body | match | L04 | 64.4 | T1 tables.claims[claim_id=CHECK:headline_V_minus_I_settled].d_gross (k EUR, absolute value) |
| 49 | 4.2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.2 |
| 50 | 4.3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 50 | 18.25 | body | match | L05v2 | 18.25 | manuscript_submitted/main.tex (sha256 ca07d7db) line 140: 'Relative to uncoordinated operation, coordinated operation with shared ESSs reduces operating costs by up to 18.25\\%' |
| 52 | 90.9 | body | match | L06 | 90.9 | T6 tables.benchmark.benefit (M EUR) |
| 52 | 13.9 | body | match | L06 | 13.9 | T6 tables.benchmark.benefit_relative (%) |
| 53 | 4.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.4 |
| 57 | Two | body | unchecked | L07 |  | count of the requests the paragraph names (Reviewer 1 attribution; Reviewer 3 re-optimisation); the third sentence (the 5 x 5 instance) is not a reviewer request |
| 57 | 1 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 57 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 57 | five | body | match | L08v2 | five | manuscript_submitted/main.tex (sha256 ca07d7db) line 664: 'is represented by five years (2025, 2028, 2031, 2034, and 2037)' |
| 57 | twenty-five | body | match | L08v2 | 25 | manuscript_submitted/main.tex (sha256 ca07d7db) line 673: 'resulting in 25 operating scenarios per representative day' |
| 57 | 3 | body | match | L09 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 57 | 3 | body | match | L09 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 57 | 4.5 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 60 | 1 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 62 | R1.2 | body | unchecked | enumerator |  | reviewer item label |
| 67 | 18.25 | body/rcomment | reviewer quotation, verified |  | 18.25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 92.16 | body/rcomment | reviewer quotation, verified |  | 92.16 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 18.25 | body/rcomment | reviewer quotation, verified |  | 18.25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 92.16 | body/rcomment | reviewer quotation, verified |  | 92.16 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 8th | body/rcomment | reviewer quotation, verified |  | 8th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 10th | body/rcomment | reviewer quotation, verified |  | 10th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 13th | body/rcomment | reviewer quotation, verified |  | 13th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 67 | 14th | body/rcomment | reviewer quotation, verified |  | 14th | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 5; letter quotation 0 segment 3 (differs) |
| 69 | one | body/rcomment | reviewer quotation, verified |  | one | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 9; letter quotation 0 segment 5 (verbatim) |
| 74 | R1.4 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 74 | 3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 76 | 3.4 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 76 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 76 | 3 | body/rresponse | match | L10 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 76 | 3 | body/rresponse | match | L10 | 3x3 / x0 | W163 N48 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W98 r2 replay reference data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/evals/f6e9cd53fdbb8ee8_x0/per_cycle_record.jsonl (t |
| 77 | five | body/rresponse | match | L54 | 5 | data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json (sha256 2a64e3c5) Years {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3} (count); data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha |
| 77 | 0.909 | body/rresponse | match | L11 | 0.909 | T11 tables.three_by_three.R_range_derived[0] (lower end, post-hoc settled descent) |
| 77 | 0.934 | body/rresponse | match | L11 | 0.934 | T11 tables.three_by_three.R_range_derived[1] (upper end, at certification) |
| 77 | single | body/rresponse | unchecked | not a figure |  | compound adjective (single-scenario) |
| 77 | 0.933 | body/rresponse | match | L12 | 0.933 | T11 tables.three_by_three.rows[quantity=R predicted from the mean-profile spread].value (0.9331) |
| 77 | 4.5 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 77 | 9 | body/rresponse | match | L13 | 9 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case9_2025': 9, 'case9_2030': 9, 'case9_2035': 9} |
| 77 | three | body/rresponse | match | L13 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DistributionNetworks (count) |
| 77 | 33 | body/rresponse | match | L13 | 33 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) {'case33_1_2025': 33, 'case33_1_2030': 33, 'case33_1_2035': 33, 'case33_2_2025': 33, 'case33_2_2030': 33, 'case33_2_2035': 33, 'case33_3_ |
| 77 | 108 | body/rresponse | match | L13 | 108 | case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at 2025/2030/2035: len(nodes) 9 + 3 x 33 per year {'2025': 108, '2030': 108, '2035': 108} |
| 78 | three | body/rresponse | match | L14 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} (count of keys) |
| 78 | five | body/rresponse | match | L14 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} (every block 5 years) |
| 78 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 78 | T1 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 78 | T6 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 79 | 4.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 82 | 3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 82 | 4.5 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.5 |
| 83 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 85 | R1.4 | body | unchecked | enumerator |  | reviewer item label |
| 94 | 0 | body/rresponse | match | L15 | 0 | T7 tables.discount[0].rate (%) |
| 94 | 2 | body/rresponse | match | L15 | 2 | T7 tables.discount[1].rate (%) |
| 94 | 5 | body/rresponse | match | L15 | 5 | T7 tables.discount[2].rate (%) |
| 94 | 8 | body/rresponse | match | L15 | 8 | T7 tables.discount[3].rate (%) |
| 94 | -40.8 | body/rresponse | match | L16 | -40.8 | T7 tables.discount[0].value_minus_I (k EUR) at rate 0.0 |
| 94 | -64.4 | body/rresponse | match | L16 | -64.4 | T7 tables.discount[1].value_minus_I (k EUR) at rate 0.02 |
| 94 | -92.7 | body/rresponse | match | L16 | -92.7 | T7 tables.discount[2].value_minus_I (k EUR) at rate 0.05 |
| 94 | -114.6 | body/rresponse | match | L16 | -114.6 | T7 tables.discount[3].value_minus_I (k EUR) at rate 0.08 |
| 94 | T7 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 94 | 2 | body/rresponse | match | L17 | 2 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DiscountFactor (%) |
| 95 | 4.6 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 95 | T7 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 97 | R1.5 | body | unchecked | enumerator |  | reviewer item label |
| 97 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 98 | 2 | body/rcomment | reviewer quotation, verified |  | 2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 2 segment 0 (verbatim) |
| 99 | 1 | body/rresponse | match | L55 | 1 | main.tex at the declared Overleaf commit: the first figure environment (line 385, ['framework_v2.pdf'], label ['fig:two-stage_tool_framework']) |
| 100 | 2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 103 | 2 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 105 | R2.1 | body | unchecked | enumerator |  | reviewer item label |
| 105 | R2.9 | body | unchecked | enumerator |  | reviewer item label |
| 107 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 107 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 107 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 107 | single | body/rcomment | reviewer quotation, verified |  | single | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 16; letter quotation 3 segment 0 (verbatim) |
| 108 | one | body/rcomment | reviewer quotation, verified |  | one | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 3 segment 1 (verbatim) |
| 110 | two | body/rresponse | unchecked | not a figure |  | compound adjective (two-stage) |
| 111 | one | body/rresponse | unchecked | L18 |  | article sense ("one ... plan" = a single plan) |
| 113 | single | body/rresponse | unchecked | not a figure |  | article sense ("a single ...") |
| 113 | 2.1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 115 | single | body/rresponse | unchecked | not a figure |  | article sense ("a single ...") |
| 118 | 2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 118 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 118 | 2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.1 |
| 118 | 2.2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 118 | 5 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 120 | R2.2 | body | unchecked | enumerator |  | reviewer item label |
| 122 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 122 | 0.25 | body/rresponse | match | L19 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | 0.5 | body/rresponse | match | L19 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | 2 | body/rresponse | match | L19 | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | 4 | body/rresponse | match | L19 | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) min/max_energy_to_power_factor |
| 122 | One | body/rresponse | unchecked | L20 |  | one evaluation per candidate (the evaluation cache keyed on the candidate); an algorithm statement, audited by the equation/algorithm audit (map section B 2.1), not a table figure |
| 122 | one | body/rresponse | unchecked | L21v2 |  | one local problem per network block per cycle (algorithm statement) |
| 122 | three | body/rresponse | match | L21v2 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 122 | four | body/rresponse | match | L21v2 | 4 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 122 | four | body/rresponse | match | L21v2 | 4 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks |
| 122 | single | body/rresponse | unchecked | L21v2 |  | compound adjective (single-scenario) |
| 122 | 48 | body/rresponse | match | L21v2 | 48 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): Years ['2025', '2030', '2035'], Days ['Autumn', 'Spring', 'Summer', 'Winter'], TransmissionNetwork + 3 DistributionNetworks (3 x 4 x 4) |
| 122 | one | body/rresponse | unchecked | L21v2 |  | one ESSO problem per storage agent per cycle |
| 122 | three | body/rresponse | match | L21v2 | 3 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) DistributionNetworks (count) = the interface nodes; shared_energy_storage_data.py:96 (sha256 9acd095f) (one ESSO solve per active distribution-network node per cycle; each ESSO model spans every year  |
| 122 | one | body/rresponse | unchecked | L21v2 |  | one agent per interface node (structure) |
| 122 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 122 | 4.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 123 | 2.2 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 123 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 123 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 125 | R2.3 | body | unchecked | enumerator |  | reviewer item label |
| 125 | R2.4 | body | unchecked | enumerator |  | reviewer item label |
| 125 | R2.7 | body | unchecked | enumerator |  | reviewer item label |
| 133 | 2.2 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2 |
| 136 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 138 | 2.2 | body/rchanges | unchecked | L22v2 |  | revised-manuscript section number (map section B 2.2) |
| 138 | 2.2.7 | body/rchanges | match | L22v2 | 2.2.7 | manuscript_submitted/main.tex section structure (subsubsection "Benders' Cuts" at line 440; numbered by its position) |
| 138 | 2.2.7 | body/rchanges | match | L23 | 2.2.7 | STEP6_REVISION_MAP.md line 96: "replace by \"2.2.7 Recourse evaluation and certification\"" |
| 139 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 141 | R2.5 | body | unchecked | enumerator |  | reviewer item label |
| 141 | R2.6 | body | unchecked | enumerator |  | reviewer item label |
| 148 | three | body/rresponse | match | L24 | 3 | W163 N7 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.e |
| 150 | one | body/rresponse | match | L25 | >= 1 period | W163 N9 (W_MIN 20) and N10 (W_FACTOR 1.1): W = max(20, ceil(1.1 P_hat)) >= 1.1 P_hat > P_hat, i.e. at least one measured period |
| 151 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 152 | 42 | body/rresponse | match | L26 | 42 | W163 C5 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 cells with certification_stats_included; W163 status match, written '42', value at written precision '42'; paragraphs_v5.md line 65 ( |
| 152 | 32 | body/rresponse | match | L26 | 32 | W163 C6 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2; W163 status match, written '32', value at written precision '32'; paragraphs_v5.md line 65 (097421f8) |
| 153 | 10 | body/rresponse | match | L26 | 10 | W163 N30 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 cells with certification_stats_included, status uncertified (W153 totals.n_uncertified); W163 status match, written '10', value at w |
| 154 | 0.9 | body/rresponse | match | L27 | max 0.882 τ | W163 N25 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): records: Q(last) − Q(k*) of the runs that continued past a settling-rule certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run p |
| 155 | 4.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 155 | T2 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 158 | two | body/rresponse | unchecked | not a figure |  | compound adjective (two-phase) |
| 163 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 166 | three | body/rresponse | match | L28 | 3 | T6 tables.benchmark.w160_additions.arms_in_full.{passive,price_taker}.Q_by_start (count of starts) |
| 166 | T6 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 168 | 2.2.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 168 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 168 | T2 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 170 | R2.8 | body | unchecked | enumerator |  | reviewer item label |
| 171 | 10.1016/j.est.2024.114911 | body/rcomment | reviewer quotation, verified |  | 10.1016/j.est.2024.114911 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) |
| 171 | 10.1109/ISGTEUROPE62998.2024.10863557 | body/rcomment | reviewer quotation, verified |  | 10.1109/ISGTEUROPE62998.2024.10863557 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) |
| 171 | 10.1016/j.apenergy.2022.120569 | body/rcomment | reviewer quotation, verified |  | 10.1016/j.apenergy.2022.120569 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 22; letter quotation 7 segment 0 (verbatim) |
| 172 | three | body/rresponse | match | L29 | 3 | the letter itself: DOIs in the R2.8 comment |
| 172 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 174 | one | body/rresponse | unchecked | L30 |  | [AUTHOR: ...] instruction, not manuscript text |
| 175 | 10.1016/j.est.2024.114911 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 175 | 1 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 176 | 10.1109/ISGTEUROPE62998.2024.10863557 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 176 | 10.1016/j.apenergy.2022.120569 | body/rresponse | unchecked | reference identifier |  | DOI (reference-list identifier) |
| 177 | 1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 179 | R2.10 | body | unchecked | enumerator |  | reviewer item label |
| 186 | 21 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 186 | 28 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 187 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 189 | R2.11 | body | unchecked | enumerator |  | reviewer item label |
| 189 | R2.12 | body | unchecked | enumerator |  | reviewer item label |
| 202 | two | body/rresponse | unchecked | L31 |  | pronoun ("the two" = the network models and the storage agent) |
| 202 | 2.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 203 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 206 | 3 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 208 | R3.1 | body | unchecked | enumerator |  | reviewer item label |
| 210 | 2.2.1 | body/rcomment | reviewer quotation, verified |  | 2.2.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 1 | body/rcomment | reviewer quotation, verified |  | 1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2034 | body/rcomment | reviewer quotation, verified |  | 2034 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2037 | body/rcomment | reviewer quotation, verified |  | 2037 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2037 | body/rcomment | reviewer quotation, verified |  | 2037 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2.4 | body/rcomment | reviewer quotation, verified |  | 2.4 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 2 | body/rcomment | reviewer quotation, verified |  | 2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 210 | 4.1 | body/rcomment | reviewer quotation, verified |  | 4.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 23; letter quotation 10 segment 0 (verbatim) |
| 212 | T1 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 212 | T4 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 213 | 60 | body/rresponse | match | L32v2 | 60 | T1 tables.claims (count) |
| 213 | two | body/rresponse | match | L32v2 | 2 | T1 tables.claims (gross_verdict != net verdict) + T4 tables.year_ladder (gross_v6.verdict != net_v6.verdict) |
| 213 | two | body/rresponse | match | L32v2 | 2 | T1 claim L:y2030__n5_p0.25_e0.5__n7_p1_e3: other_cell l_0ee93aca candidate_canonical.nodes (non-zero nodes); data/SRP1/Results/P515S53/w142_resettle_v6/frozen_s53_resettle_spec_v6_96c23404.json (sha256 96c23404) cells.l_0ee93aca.m_flex_price_multiplier |
| 213 | 2035 | body/rresponse | match | L34 | 2035 | T4 tables.year_ladder.per_year; data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 213 | 2030 | body/rresponse | match | L34 | 2030 | T4 tables.year_ladder.per_year; data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years |
| 213 | +42.5 | body/rresponse | match | L34 | 42.5 | T4 tables.year_ladder.D_gross (k EUR, 2035 - 2030) |
| 213 | -2.3 | body/rresponse | match | L35 | -2.3 | T4 tables.year_ladder.D_net (k EUR) |
| 213 | 4.6 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 213 | 2 | body/rresponse | unchecked | L36 |  | Table [2] of the revised manuscript (placeholder; the reviewer cites "Table 2 in Section 4.1" of the submitted version) |
| 214 | 2.2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 214 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 214 | 4.6 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.6 |
| 214 | T1 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 214 | T4 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 216 | R3.2 | body | unchecked | enumerator |  | reviewer item label |
| 216 | 0.50 | body | match | L37v2 | 0.50 | manuscript_submitted/main.tex (sha256 ca07d7db) line 866: 'to a relative optimality gap of 0.50\\%' |
| 218 | 2.2.7 | body/rcomment | reviewer quotation, verified |  | 2.2.7 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 11 | body/rcomment | reviewer quotation, verified |  | 11 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 4.3 | body/rcomment | reviewer quotation, verified |  | 4.3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 26 | body/rcomment | reviewer quotation, verified |  | 26 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 27 | body/rcomment | reviewer quotation, verified |  | 27 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 29 | body/rcomment | reviewer quotation, verified |  | 29 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 11 | body/rcomment | reviewer quotation, verified |  | 11 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 0.50 | body/rcomment | reviewer quotation, verified |  | 0.50 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 4.4 | body/rcomment | reviewer quotation, verified |  | 4.4 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 218 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 11 segment 0 (verbatim) |
| 222 | 2.2.7 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 223 | 4 | body/rresponse | match | L38 | 4 | W163 N18 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula); W163 status match, written '4', value at written precision '4'; p |
| 223 | 0.07 | body/rresponse | match | L38 | 0.07 | W163 N19 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R; W163 status match, written '0.07', value at written precision '0.07';  |
| 223 | single | body/rresponse | unchecked | L38 |  | compound adjective (single-scenario) |
| 224 | two | body/rresponse | match | L39 | 2 | W163 N22 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T11: R = V / V_SRP1_settled (a ratio of two values); W163 status match, written 'two', value at written precision '2'; paragraphs_v5.md |
| 224 | 15 | body/rresponse | unchecked | L40v2 |  | figure number of the submitted version. In manuscript_submitted/main.tex the figure environments in order give Figure 15 = line 1637, D Distribution Networks > D.2 ADN Connected to TN Node~7, ['case33_2_flexibility_scenarios_2025_Spring.pdf']; Section 4.4 hold |
| 227 | 2.2.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.2.7 |
| 227 | 4.7 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.7 |
| 227 | 4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 229 | R3.3 | body | unchecked | enumerator |  | reviewer item label |
| 231 | 21 | body/rcomment | reviewer quotation, verified |  | 21 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) |
| 231 | 31 | body/rcomment | reviewer quotation, verified |  | 31 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) |
| 231 | 2.3.2 | body/rcomment | reviewer quotation, verified |  | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 24; letter quotation 12 segment 0 (differs) |
| 233 | 2.3.3 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 233 | R2.11 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 233 | R2.12 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 233 | one | body/rresponse | unchecked | L56 |  | a network block = one (representative year, representative day) pair (48 = 3 x 4 x 4, L21v2); structure, no value |
| 233 | one | body/rresponse | unchecked | L56 |  | as the first "one" |
| 233 | single | body/rresponse | unchecked | L56 |  | one scenario-free schedule (model_construction_helpers.sess_na_scenario / sess_row_is_duplicate); a model statement audited by W171b |
| 233 | one | body/rresponse | unchecked | L56 |  | as "a single schedule" |
| 233 | one | body/rresponse | unchecked | L57 |  | demonstrative ("this one schedule") |
| 233 | 2.3.6 | body/rresponse | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 235 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 237 | R3.4 | body | unchecked | enumerator |  | reviewer item label |
| 239 | 2.3.2 | body/rcomment | reviewer quotation, verified |  | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) |
| 239 | 22 | body/rcomment | reviewer quotation, verified |  | 22 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) |
| 239 | 23 | body/rcomment | reviewer quotation, verified |  | 23 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 26; letter quotation 13 segment 0 (verbatim) |
| 241 | R2.10 | body/rresponse | unchecked | enumerator |  | reviewer item label |
| 242 | 2.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 2.3 |
| 244 | R3.5 | body | unchecked | enumerator |  | reviewer item label |
| 246 | 2.3.2 | body/rcomment | reviewer quotation, verified |  | 2.3.2 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 22 | body/rcomment | reviewer quotation, verified |  | 22 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 25 | body/rcomment | reviewer quotation, verified |  | 25 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | two | body/rcomment | reviewer quotation, verified |  | two | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 246 | 15 | body/rcomment | reviewer quotation, verified |  | 15 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 27; letter quotation 14 segment 0 (verbatim) |
| 248 | 0.985 | body/rresponse | match | L41 | 0.985 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d) ageing.calendar_retention_per_year; data/SRP1/Results/P515S53/frozen_s53_spec_v41_fcea4b38.json (sha256 fcea4b38, last commit 477dba3e) configuration.identical_to_w104.ess_ageing_ |
| 249 | C2 | body/rresponse | unchecked | enumerator |  | ageing calibration name (T8 arm labels C2 / C2_calfade) |
| 249 | C2 | body/rresponse | unchecked | enumerator |  | ageing calibration name (T8 arm labels C2 / C2_calfade) |
| 250 | 31.6 | body/rresponse | match | L42 | 31.6 | T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR, absolute value) |
| 250 | 64.4 | body/rresponse | match | L42 | 64.4 | T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR, absolute value) |
| 251 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 251 | 0.5 | body/rresponse | MISMATCH | L43v2 |  | manuscript_submitted/main.tex (sha256 ca07d7db): fragment 'annual rates of 0.5\\% and 2.0\\% are additionally considered' not found |
| 251 | 2 | body/rresponse | MISMATCH | L43v2 |  | manuscript_submitted/main.tex (sha256 ca07d7db): fragment 'annual rates of 0.5\\% and 2.0\\% are additionally considered' not found |
| 253 | 3.4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 3.4 |
| 253 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 253 | T8 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 255 | R3.6 | body | unchecked | enumerator |  | reviewer item label |
| 257 | 4.6 | body/rcomment | reviewer quotation, verified |  | 4.6 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 28; letter quotation 15 segment 0 (verbatim) |
| 257 | 5 | body/rcomment | reviewer quotation, verified |  | 5 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 28; letter quotation 15 segment 0 (verbatim) |
| 259 | 7 | body/rresponse | match | L44 | 7 | T1 tables.claims[claim_id=E:n7_4h_e1_C2_calfade:value_minus_I].instance (the non-zero node of the unit) |
| 259 | -4.1 | body/rresponse | match | L45v2 | -4.1 | T8 tables.ageing.rows[arm=no_ageing].value_minus_I (k EUR) |
| 259 | -31.6 | body/rresponse | match | L45v2 | -31.6 | T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR) |
| 259 | -64.4 | body/rresponse | match | L45v2 | -64.4 | T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR) |
| 259 | -45.2 | body/rresponse | match | L45v2 | -45.2 | T8 tables.ageing.rows[arm=C4].value_minus_I (k EUR) |
| 259 | -65.9 | body/rresponse | match | L45v2 | -65.9 | T8 tables.ageing.rows[arm=C3_midblock].value_minus_I (k EUR) |
| 259 | -73.7 | body/rresponse | match | L45v2 | -73.7 | T8 tables.ageing.rows[arm=C3_unit].value_minus_I (k EUR) |
| 259 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 260 | 0.70 | body/rresponse | match | L46 | 0.70 | W163 N38 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label "soh_min 0.70"); W163 status match, written '0.70', val |
| 260 | 0.50 | body/rresponse | match | L46 | 0.50/0.50 | W163 N41 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): A64 e_soh050 r2 campaign spec (28c9478d) candidates[0].settling_resettle.minimum_soh and extra.minimum_soh (the override; configuration |
| 260 | +4.9 | body/rresponse | match | L46 | 4.9 | T10 tables.a64.rows[claim_id=E:soh050:delta_value_vs_070].d_gross (k EUR) |
| 260 | T10 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 260 | T8 | body/rresponse | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 261 | 4.3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.3 |
| 261 | T8 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 261 | T10 | body/rchanges | unchecked | enumerator |  | table identifier of the frozen tables (placeholder [T..] in the letter) |
| 263 | R3.7 | body | unchecked | enumerator |  | reviewer item label |
| 263 | R3.8 | body | unchecked | enumerator |  | reviewer item label |
| 265 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) |
| 265 | 8 | body/rcomment | reviewer quotation, verified |  | 8 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) |
| 265 | 0.50 | body/rcomment | reviewer quotation, verified |  | 0.50 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 29; letter quotation 16 segment 0 (verbatim) |
| 266 | 3 | body/rcomment | reviewer quotation, verified |  | 3 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 266 | 8 | body/rcomment | reviewer quotation, verified |  | 8 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 266 | 4.4.1 | body/rcomment | reviewer quotation, verified |  | 4.4.1 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 266 | four | body/rcomment | reviewer quotation, verified |  | four | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 266 | 8760 | body/rcomment | reviewer quotation, verified |  | 8760 | manuscript_review/Reviewers Comments.docx (sha256 c7afdc8b) paragraph 30; letter quotation 16 segment 1 (verbatim) |
| 268 | one | body/rresponse | match | L47 | 1 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.md T7 caption ("one discount factor per representative year applied to the five years of its block") |
| 269 | five | body/rresponse | match | L47 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 270 | 2025 | body/rresponse | match | L48 | 2025 | data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.md T7 caption ("I paid in 2025"); T7 tables.discount[*].I identical at every rate |
| 272 | 2.2.1 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 272 | 3 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 272 | 4 | body/rchanges | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 279 | five | body | match | L49 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 285 | 1 | body | match | L50 | 1 | T6 tables.benchmark.sweep.sweep_passive_cold.n_blocks |
| 285 | 12 | body | match | L50 | 12 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): years x days |
| 285 | 8 | body | match | L50 | 8 | T6 tables.benchmark.sweep.sweep_price_taker_cold.n_blocks |
| 285 | 12 | body | match | L50 | 12 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a): years x days |
| 286 | 4.4 | body | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose; the revision map names section 4.4 |
| 288 | 10^{-5} | body | match | L51 | 1e-05 | W163 N33 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_helpers.py curtaillable pg upper bound pg_avail + EQUALITY_TOLERANCE; W109 |
| 290 | single | body | no table counterpart | L52v2 |  | no committed record states the machine count; stands on the author's attestation (Addendum 67; Addendum 68 Decision 5 "Single machine stands on the author's attestation") |
| 290 | single | body | match | L52v2 | thread caps 1 on every table evaluation | W163 N45 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W159 c1: harness THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VECLIB / NUMEXPR = 1); all_table_evaluations_ran_with_OMP_NUM_THREADS_1;  |
| 292 | 49 | body | match | L53 | 49 | data/SRP1/Results/P515S53/w160_step6_frozen/export/paragraphs_v2.md (sha256 73b42c49, last commit 2a1d7f92) section (v) prediction scorecard: rows numbered 1..n |

## Findings -- letter

- **G1** (line 260, claim against a table): "the year in which the end-of-life floor binds (within the horizon it never does)": T8 "floor binds (0.70)" (tables.ageing.rows[*].floor_year_070) = {'C3_unit': 2035, 'C2': None, 'C4': None, 'C2_calfade': 2035, 'C3_midblock': 2035, 'no_ageing': None}. The floor binds in 2035 for ['C2_calfade', 'C3_midblock', 'C3_unit'] -- the baseline C2_calfade among them -- and never for ['C2', 'C4', 'no_ageing']. The sentence (STEP6_ROUND1_CORRECTIONS B.5; Addendum 68 Decision 5 "the floor never binds within the horizon") contradicts T8 for three of the five aged arms. No number token: reported here only.
- **G2** (line 251, claim against the submitted source): "The sentence of the submitted version announcing 0.5 % and 2 % cases that had not been run was removed": the sentence is not in the submitted source (manuscript_submitted/main.tex, sha256 ca07d7db; substrings found in it: {'calendar ageing': False, 'calendar aging': False, 'annual rates of 0.5': False}); it is first-reply text of the Overleaf main.tex, still present at this commit, line 1053 (Addendum 68 Decision 6: removed in the section 3.4 rewrite, not yet made). Tokens 0.5 / 2: MISMATCH (L43v2).
- **G3** (line 99, 100, figure numbering): R1.5 answers for the framework figure ("Figure~1 of the revised manuscript": main.tex first figure environment, line 385, ['framework_v2.pdf'] -- match). (a) The \rchanges line still says "Figure~2 and the graphical abstract". (b) In the submitted source the second figure environment -- the reviewer's "Figure 2" by environment order -- is ['network_diagram.pdf'] (line 666, 3 Case Study); the framework figure is Figure 1 (['framework.pdf']). The reviewers' document carries the author's note "Placeholder figure" after the comment: True. The PDF numbering is not re-derived here.
- **G4** (line 224, figure numbering): "On Figure 15 of the submitted version: the figure and its discussion belonged to the superseded results and were removed": by environment order the submitted Figure 15 is ['case33_2_flexibility_scenarios_2025_Spring.pdf'] (D Distribution Networks > D.2 ADN Connected to TN Node~7, line 1637), a data figure kept in the revision; Section 4.4 holds Figures [4, 5, 6, 7]. The reviewer's "Figure 15 in Section 4.4" matches no figure of the source; the letter's claim cannot be confirmed from it.
- **G5** (line [77], typographical): R1.2: a stray double quote follows "(Section~4.5)" -- `(Section~4.5)".` found on letter lines [77] -- left by the B.6 paste.
- **G6** (line 78, 122, correction not applied): STEP6_ROUND1_CORRECTIONS B.6 second part: "wherever 'three representative years standing for five-year blocks' is stated as the instance, add '(single-scenario instance; the multi-scenario instance uses five three-year blocks)'". Qualifier present anywhere in the letter: False. R1.2 reads "the horizon (three representative years standing for five-year blocks)" (R2.2 states the single-scenario instance explicitly).
- **G7** (line Further changes, correction not applied): Addendum 68 Decision 3: "the letter's 'Further changes' names the correction (÷4 h, not ÷5)". The "Further changes" list names no cost-file correction (searched the section for ['cost file', 'cost-file', 'cost input', 'investment-cost', 'divided', '\\div', '÷', '/5', '4~h']: found []); the only mention is the opening paragraph's "We also corrected the investment-cost input file".
- **G8** (line 172, internal contradiction (unchanged)): "The three references were added" against the bracketed status in the same response (two "not yet added"); B.8 keeps the status note until the references are in.
- **K1** (line 213, claim checked, consistent): "two verdicts": T1 ['L:y2030__n5_p0.25_e0.5__n7_p1_e3'] (gross within the uncertified bar -> net determinate) and T4 2035 - 2030 (gross determinate, net within resolution). The T4 comparison is not among T1's 60 claims; the sentence reads "no sign in the 60 ... and two verdicts", so the count spans T1 and T4. Sign changes in T1: 0. "a two-node plan evaluated under the doubled flexibility price": checked (L32v2).
- **K2** (line 259, claim checked, consistent): R3.6: every named value matches T8 (L45v2); arms in order no_ageing, C2, C2_calfade, C4 ("datasheet-exact"), C3_midblock ("mid-block evaluation point"), C3_unit ("unit-retention reading"); all aged arms determinate, no_ageing within resolution. C3_midblock is the C3 calibration (k 11542, eol retention 0.5) evaluated mid-block, not the baseline calibration evaluated mid-block.
- **K3** (line 122, claim checked, consistent): R2.2: 48 network blocks (3 x 4 x 4) plus one ESSO problem per interface node (3) per cycle (shared_energy_storage_data.optimize loops over the active distribution-network nodes).
- **K4** (line 156, claim checked, consistent (as W164 K1)): 6 of the 10 uncertified SRP1 evaluations are T9 dead-zone entries (['h_f9eae48f', 'j_5f3cccb4', 'l_0ee93aca', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5']).
- **K5** (line rcomment blocks, quotation check): 29 of 31 quotation segments verbatim, 2 differ, 0 not found (table in the MD).

## Findings -- main.tex section 2

- **M1** (line [745, 854, 897], cross-reference to a section not yet written): Section 2 defers the values of eta, SoC^Min/Max/0, eps^Cl, c^Cl, eps^C, c^sigma, eps^E and alpha to "Section~3.5"; main.tex at this commit has sections 3.1-3.4 only (['3.1 Investment Costs', '3.2 Market Data', '3.3 Transmission Network', '3.4 Active Distribution Networks']). The values are checked against the code in the parameter table, for when 3.5 is written.
- **M2** (line [830], cross-reference to unrevised text): "The calibrations used, the calendar retention and the end-of-life floor are given in Section~3.4": 3.4 is "Active Distribution Networks" -- the submitted text, which carries the 1 %/yr calendar sentence and the 60 % / 80 % floors (line 1053).
- **M3** (line 625, definition against code): P_hat "measured between the first and third turning points": the code takes the three most recent (T[-1] - T[-3]); W163 N11 and paragraphs_v5 wrote "from the third-last to the last". Token "third" MISMATCH (S205; draft d205 likewise).
- **M4** (line 426, 431, algorithm against code): Algorithm 1: "the 2n OrthoMADS directions" and the completion "If |P| < n + 1"; the planning searches as run poll n + 1 directions (p515_s47_phase_b_record.py:199 (sha256 97f573e5)) and add the completion at every unit poll (p515_s47_phase_b_record.py:607 (sha256 97f573e5)). Tokens MISMATCH (S219, S225). Doubling / halving / unit termination / l_inf-1 completion match. Whether "improves by a determinate margin (Subsection certification)" equals the code's improvement test (Phase B: resolution = max(bar(x) + bar(inc), sigma_Q)) is an algorithm question for W171b, not checked here.
- **M5** (line 639, claim with a pending record): "The rule above, with its holds, produced every single-scenario evaluation reported in this paper": the x = 0 reference ref:7aa017f0 was certified under rule v1 (Addendum 68 Decision 4); W168 (commit 96037a16, not read by this run) reports its v6 replay certifying at the same cycle. Not checked here.

## highlights.tex: every number with its source

no in-scope numeric token (structure only: `12pt` (package/class option or argument), `1in` (package/class option or argument))

## cover_letter.tex: every number with its source

- line 16 `four`: match -- the cover letter itself: \item entries of the list that follows
- line 24 `15`: match -- data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years (sum of block lengths)

## Draft (not compiled): every number with its source

| line | scope | written | status | check | counterpart | source / reason |
|---:|---|---|---|---|---|---|
| 2 | comment | 2 | unchecked | dC01 |  | section number in a title comment |
| 2 | comment | 2026-10-07 | match | dC01 | 2026-10-07 | STEP6_ROUND1_CORRECTIONS.md line 1 header date (the corrections the draft precedes are of the same date) |
| 4 | comment | Three | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 4 | comment | 6cb4492 | unchecked | comment (draft instruction) |  | hash in a draft instruction |
| 5 | comment | 1 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 6 | comment | 394 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 6 | comment | 397 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 6 | comment | 540 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 6 | comment | 2.2.1 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 6 | comment | 550 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 6 | comment | 577 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 7 | comment | 2.2.2 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 7 | comment | 2.2.5 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 7 | comment | 2.2.6 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 7 | comment | 656 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 7 | comment | 659 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 8 | comment | 2.2.7 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 9 | comment | 2.2.7 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 9 | comment | 684 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 9 | comment | 722 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 10 | comment | 2.3 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 12 | comment | 735 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 12 | comment | 906 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 12 | comment | 2.3.2 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 12 | comment | 2.3 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 13 | comment | 724 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 13 | comment | 733 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 13 | comment | two | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 17 | comment | v1 | unchecked | dC02 |  | version label of the frozen tables |
| 17 | comment | 590088fe | match | dC02 | 590088fe | sha256 of data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json |
| 18 | comment | 4 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 23 | comment | 1 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 26 | comment | 394 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 26 | comment | two | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 29 | body | two | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 32 | body | 2.2.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 40 | comment | 397 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 40 | comment | 540 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 41 | comment | 0.25 | match | dC03 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d)  |
| 41 | comment | 0.5 | match | dC03 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d)  |
| 42 | comment | 3.5 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 54 | body | 2.2.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 60 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 60 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 62 | body | -1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 64 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 67 | body | single | unchecked | not a figure |  | compound adjective (single-node) |
| 74 | body | 2n | MISMATCH | d219 | n+1 | p515_s47_phase_b_record.py:199 (sha256 97f573e5); p515_s47_phase_b_record.py:509 (sha256 97f573e5); p515_s51_f2_phase_b.py:803 (sha256 87ab3b2d) |
| 79 | body | 1 | MISMATCH | d225 | unit poll | p515_s47_phase_b_record.py:607 (sha256 97f573e5); p515_s47_phase_b_record.py:204 (sha256 97f573e5); p515_s47_phase_b_record.py:205 (sha256 97f573e5) |
| 80 | body | one | match | d220 | one | p515_s47_phase_b_record.py:46 (sha256 97f573e5) (COMPLETION_RULE) |
| 88 | body | 2.2.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 89 | body | 2 | match | d226 | 2 | p515_s47_phase_b_record.py:719 (sha256 97f573e5); STEP4_DFO_METHOD.md:190 (sha256 d05bae1e) |
| 91 | body | 1 | match | d227 | 1 | p515_s47_phase_b_record.py:195 (sha256 97f573e5); STEP4_DFO_METHOD.md:123 (sha256 d05bae1e) |
| 92 | body | 2 | match | d228 | 2 | p515_s47_phase_b_record.py:730 (sha256 97f573e5); STEP4_DFO_METHOD.md:190 (sha256 d05bae1e) |
| 100 | comment | 1 | match | dC04 | 1 | p515_s47_phase_b_record.py:195 (sha256 97f573e5); STEP4_DFO_METHOD.md:123 (sha256 d05bae1e) |
| 101 | comment | 30 | match | dC05 | 30 | p515_s47_phase_b_record.py:205 (sha256 97f573e5) |
| 101 | comment | 2.2.7 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 102 | comment | single | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 103 | comment | one | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 104 | comment | 4.7 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 111 | body | 2.2 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 112 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 114 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 125 | comment | 17 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 125 | comment | 1 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 125 | comment | 188 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 125 | comment | 217 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 125 | comment | 2006 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 127 | comment | 20 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 127 | comment | 2 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 127 | comment | 948 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 127 | comment | 966 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 127 | comment | 2009 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 129 | comment | 29 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 129 | comment | 2 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 129 | comment | 1164 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 129 | comment | 1189 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 129 | comment | 2019 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 131 | comment | 2.2.1 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 131 | comment | 550 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 131 | comment | 577 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 142 | body | 1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 157 | body | 2.2.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 160 | body | 4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 162 | comment | 2.1 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 162 | comment | 1 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 163 | comment | R3.1 | unchecked | comment (draft instruction) |  | label in a draft instruction |
| 166 | comment | 2.2.2 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 166 | comment | 2.2.5 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 166 | comment | one | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 166 | comment | 629 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 168 | comment | 2.2.2 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 168 | comment | 2.2.5 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 170 | comment | 629 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 173 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 174 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 174 | body | 0.25 | match | d201 | 0.25 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d)  |
| 175 | body | 0.5 | match | d201 | 0.5 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d)  |
| 175 | body | 2 | match | d201 | 2.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d)  |
| 175 | body | 4 | match | d201 | 4.0 | p515_s45_a1_campaign.py LATTICE_P_STEP / LATTICE_E_STEP; STEP4_DFO_METHOD.md lines 8-9 (author's decision 2026-09-19); data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93, last commit 2466401d)  |
| 175 | body | 3 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 177 | comment | 2.2.6 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 177 | comment | 656 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 177 | comment | 659 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 186 | comment | 2.2.7 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 188 | comment | three | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 188 | comment | 684 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 188 | comment | 722 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 199 | body | 1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 212 | body | 2.3 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 222 | body | 10^{-5} | match | d202 | 1e-05 | W163 N2 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): data/SRP1/SRP1_params.json admm.tol.boyd.eps_abs; W163 status match, written |
| 223 | body | 10^{-4} | match | d202 | 0.0001 | W163 N3 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): data/SRP1/SRP1_params.json admm.tol.boyd.eps_rel; W163 status match, written |
| 225 | body | 15 | match | d203 | 15 | W163 N4 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€), measured f |
| 225 | body | 21 | match | d203 | 21 | W163 N5 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed (k€), meas |
| 232 | body | three | match | d204 | 3 | W163 N7 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points = l |
| 233 | body | third | MISMATCH | d205 | third-last to last | settling_criterion_v2.py:166 (sha256 3db13b0e) (inherited by settling_criterion_v6); W163 N11 written 'the three most recent turning points: from the third-last to the last' |
| 234 | body | 10 | match | d206 | 10 | W163 N8 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.SWING_FLOOR = TAU / 10 (= GROWTH_TEST_FLOOR = TURNING_ |
| 236 | body | 20 | match | d207 | 20 | W163 N9 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.W_MIN; v6 spec stop_rule.W.oscillatory; W163 status match |
| 236 | body | 1.1 | match | d207 | 1.1 | W163 N10 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.W_FACTOR; W163 status match, written '1.1', value at wri |
| 239 | body | 2 | match | d208 | 2 | W163 N12 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND); W163 status match |
| 241 | body | four | match | d209 | 4 | W163 N13 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v5.METRICS (v6) and v6 spec stop_rule.clean_rule.metric_ |
| 242 | body | ten | match | d209 | 10 | W163 N14 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v5.CLEAN_FACTOR (v6); v6 spec stop_rule.clean_rule.facto |
| 247 | body | 2P_ | match | d210 | 60 | W163 N15 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec stop_rule.p_max.L and stop_rule.W.monotone ("L = L_MONO = 2 * P_MAX |
| 248 | body | 2P_ | match | d211 | 60 | W163 N17 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): v6 spec stop_rule.constants.MONOTONE_LAST_STEP_CLAUSE "abs(dQ_k) * L_MONO < |
| 254 | body | 4 | match | d212 | 4 | W163 N18 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula); W163  |
| 254 | body | 0.07 | match | d212 | 0.07 | W163 N19 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R; W163 statu |
| 255 | body | 259,375.33 | match | d213 | 259375.33 / 259375.33 | W163 N20 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion.R_REF; W101 summary expert_P2.V_old (the SRP1 value Q(x0 |
| 255 | body | 4,539.07 | match | d213 | 4539.07 | W163 C1 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): constants.TAU; W163 status match, written '4,539.07', value at written preci |
| 256 | body | two | match | d214 | 2 | W163 N22 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T11: R = V / V_SRP1_settled (a ratio of two values); W163 status match, wri |
| 256 | body | four | match | d214 | 4 | W163 N21 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): TAU divisor 4 (N18); T11 rows: V = Q(0) − Q(unit) (two evaluations per valu |
| 258 | body | ten | match | d215 | 10 | W163 C4 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): T2 at_or_above_0.95_tau_counted (count of true); W163 status match, written  |
| 258 | body | 5 | match | d215 | 5 | W163 N24 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): frozen constants.FLAG_RANGE_OVER_TAU / registry threshold 0.95: 1 − 0.95; W |
| 259 | body | 0.9 | match | d216 | max 0.882 τ | W163 N25 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): records: Q(last) − Q(k*) of the runs that continued past a settling-rule ce |
| 259 | body | 4.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 261 | body | two | match | d217 | 2 bars | W163 N73 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the diff |
| 262 | body | 3 | match | d217 | 3 | W163 N28 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.DETERMINACY_BAR_FACTOR; determinacy_threshold behavio |
| 262 | body | 2 | match | d217 | 2 | W163 N29 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.DETERMINACY_TAU_MULTIPLE; determinacy_threshold(100,  |
| 262 | body | two | match | d217 | 2 bars | W163 N73 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the diff |
| 264 | body | three | match | d218 | 3 × max(1000, 2000) = 6000 | W163 N26 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): p515_s53_w132_resettle_v3_campaign.resolve (the scorer): bar = 3.0 × max(ga |
| 272 | comment | 66 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 272 | comment | ten | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 272 | comment | 4.7 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 274 | comment | 3 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 274 | comment | 1 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 274 | comment | 2011 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 274 | comment | 3.3 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 274 | comment | 3.4.1 | unchecked | comment (draft instruction) |  | bibliographic data of a reference (volume, issue, pages, year) |
| 278 | comment | 2.3 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 280 | comment | Two | unchecked | comment (draft instruction) |  | number word in a draft instruction (structure) |
| 280 | comment | 2.3 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 280 | comment | 727 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 280 | comment | 731 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 281 | comment | 727 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 283 | comment | 731 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 287 | body | two | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 290 | comment | 735 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 290 | comment | 737 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 290 | comment | 790 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 312 | body | 1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 317 | comment | 1 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 333 | comment | R2.11 | unchecked | comment (draft instruction) |  | label in a draft instruction |
| 333 | comment | R2.12 | unchecked | comment (draft instruction) |  | label in a draft instruction |
| 333 | comment | R3.1 | unchecked | comment (draft instruction) |  | label in a draft instruction |
| 333 | comment | R3.3 | unchecked | comment (draft instruction) |  | label in a draft instruction |
| 338 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 339 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 346 | body | 1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 361 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 370 | body | 2 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 370 | body | 2 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 370 | body | 2 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 377 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 384 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 392 | body | single | unchecked | not a figure |  | article sense ("a single ...") |
| 393 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 394 | body | 2.3.4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 396 | comment | 0 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 399 | comment | 0 | unchecked | comment (draft instruction) |  | other integer in a draft instruction (algorithm / equation / addendum number, part counter or formula index) |
| 399 | comment | 3 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 401 | comment | 792 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 401 | comment | 842 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 419 | comment | H3 | unchecked | comment (draft instruction) |  | label in a draft instruction |
| 429 | body | 365 | match | d221 | 365 | shared_energy_storage_data.py:665 (sha256 9acd095f) |
| 444 | body | 365 | match | d222 | 365 | shared_energy_storage_data.py:703 (sha256 9acd095f) |
| 445 | body | 2 | match | d223 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 451 | body | 2 | match | d224 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 452 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 458 | body | 1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 460 | body | -1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 460 | body | 1 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 486 | body | 3.4 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 492 | comment | 2 | match | dC06 | 2 | shared_energy_storage_data.py:702 (sha256 9acd095f) |
| 492 | comment | 365 | match | dC06 | 365 | shared_energy_storage_data.py:703 (sha256 9acd095f) |
| 495 | comment | 5 | match | dC07 | 5 | data/SRP1/SRP1.json (sha256 61a794a7, last commit 9568835a) Years {'2025': 5, '2030': 5, '2035': 5} |
| 495 | comment | 3.4 | unchecked | dC07 |  | section number in a draft instruction |
| 495 | comment | C2 | unchecked | dC07 |  | ageing calibration name |
| 495 | comment | 10,000 | match | dC07 | 10000 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.cycles_n |
| 495 | comment | 0.80 | match | dC07 | 0.80 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.reference_dod_d |
| 495 | comment | 0.80 | match | dC07 | 0.80 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calibration.eol_retention_r |
| 495 | comment | 35,851 | match | dC07 | 35851 | T8 tables.ageing.rows[arm=C2].k |
| 496 | comment | C4 | unchecked | dC08 |  | ageing calibration name |
| 496 | comment | 8,000 | match | dC08 | 8000 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant: C4 runs with the file's (N, D) = (10000, 0.8) and eol_retention_r 0.7; N x |
| 496 | comment | 1.0 | match | dC08 |  | as the 8,000 token |
| 496 | comment | 0.70 | match | dC08 | 0.70 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.arms.C4.eol_retention_r |
| 496 | comment | 22,430 | MISMATCH | dC08 | 22429 | data/SRP1/Results/P515S53/w142_resettle_ext_v6/frozen_s53_resettle_ext_spec_v3_84775dc4.json (sha256 84775dc4) model_variant.k_closed_form_by_arm.C4 (= T8 C4 k 22429.386) |
| 496 | comment | 0.985 | match | dC08 | 0.985 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.calendar_retention_per_year |
| 496 | comment | 0.70 | match | dC08 | 0.70 | data/SRP1/SharedESS/SRP1_ESS_Params.json (sha256 39106f93) ageing.minimum_soh |
| 496 | comment | 0.50 | match | dC08 | 0.50/0.50 | W163 N41 (data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_paragraphs_v4_figure_check.json, results commit ff66c75f): A64 e_soh050 r2 campaign spec (28c9478d) candidates[0].settling_resettle.mi |
| 498 | comment | 844 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 498 | comment | 900 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 521 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 524 | body | 2.2.7 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 524 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 532 | comment | 3.5 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 533 | comment | 1e | match | dC10 | 1 | shared_energy_storage_data.py:1970 (sha256 9acd095f) |
| 533 | comment | 6 | unchecked | dC10 |  | exponent of 1e-6 (checked with the 1e token against the code line) |
| 535 | comment | 905 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 535 | comment | 906 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 550 | body | 0 | unchecked | notation |  | formula constant in the draft (an index offset, exponent, bound or coefficient of the printed formula; the equations and Algorithm 1 are audited against the code by W171b) |
| 572 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 573 | body | 2.3.2 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 573 | body | one | unchecked | method statement |  | number word in the draft describing structure (no value to check; the statement is audited against the code by W171b) |
| 573 | body | single | unchecked | not a figure |  | article sense ("a single ...") |
| 574 | body | single | unchecked | not a figure |  | compound adjective (single-scenario) |
| 575 | body | 3.5 | unchecked | cross-reference |  | section / table / figure / equation / algorithm / reviewer number written in prose |
| 577 | comment | 0.50 | match | dC09 | 0.50 | data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_spec_s53_w91_3x3_pair_231558f0.json (sha256 231558f0) candidates[*].interface_deviation_premium.alpha |
| 577 | comment | 3x3 | unchecked | dC09 |  | instance label (3 x 3) |
| 579 | comment | 2.3.2 | unchecked | comment (draft instruction) |  | section number in a draft instruction (of the manuscript or of a cited reference) |
| 582 | comment | 902 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 583 | comment | 902 | unchecked | comment (draft instruction) |  | line number of the Overleaf clone at 6cb4492 the draft refers to |
| 583 | comment | R1.2 | unchecked | comment (draft instruction) |  | label in a draft instruction |

## main.tex: submitted-version figures by section (line rules re-pinned to 7effd898)

| section | n | map lines cited |
|---|---:|---|
| Front matter | 3 | 117, 23 |
| 3 Case Study | 77 | 114, 117, 120, 126 |
| 4 Results | 245 | 137, 159, 165, 169 |
| 5 Conclusions | 2 | 23 |
| E Results | 283 | 193 |

## main.tex: statuses by section

| section | status | n |
|---|---|---:|
| (comments) | unchecked | 36 |
| Front matter / preamble | unchecked | 6 |
| Front matter | unchecked | 4 |
| Front matter | match (declared) | 5 |
| Front matter | submitted-version figure | 3 |
| 1 Introduction | unchecked | 8 |
| 2 Shared ESS Planning Framework | unchecked | 63 |
| 2 Shared ESS Planning Framework | MISMATCH | 3 |
| 2 Shared ESS Planning Framework | match (declared) | 41 |
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

Auto index (4025 leaves; main.tex outside section 2 only): matches 0; tokens with >= 1 candidate assigned another status by precedence: {'match': 203, 'unchecked': 778, 'MISMATCH': 4, 'submitted-version figure': 129, 'no table counterpart': 2, 'reviewer quotation, verified': 27}

## Unchecked: by category (all files)

| file | category | n |
|---|---|---:|
| main.tex | comment (not typeset) | 36 |
| main.tex | address | 6 |
| main.tex | not a figure | 22 |
| main.tex | enumerator | 2 |
| main.tex | descriptive count | 5 |
| main.tex | cross-reference | 9 |
| main.tex | method statement | 11 |
| main.tex | notation | 53 |
| main.tex | literature value | 3 |
| main.tex | network/data parameter | 1005 |
| response_to_reviewers_draft.tex | enumerator | 48 |
| response_to_reviewers_draft.tex | cross-reference | 75 |
| response_to_reviewers_draft.tex | structural | 1 |
| response_to_reviewers_draft.tex | not a figure | 11 |
| response_to_reviewers_draft.tex | method statement | 8 |
| response_to_reviewers_draft.tex | author placeholder | 1 |
| response_to_reviewers_draft.tex | reference identifier | 3 |
| section2_expert_draft.tex | comment (draft instruction) | 111 |
| section2_expert_draft.tex | identifier | 2 |
| section2_expert_draft.tex | method statement | 10 |
| section2_expert_draft.tex | cross-reference | 15 |
| section2_expert_draft.tex | notation | 23 |
| section2_expert_draft.tex | not a figure | 4 |
| section2_expert_draft.tex | enumerator | 2 |

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
- no_duplicate_declaration_id: True
- every_token_assigned: True
- every_map_citation_found_once: True
- statuses_in_vocabulary: True
- main_line_rules_pinned_to_this_main: True
- every_v1_declaration_carried_superseded_or_replaced: True
- parameter_table_evaluated: True
- every_quotation_segment_located: True
- docx_sha_equals_declared: True
