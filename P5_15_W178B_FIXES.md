# P5.15 W178b — fixes A1–A20 / B1–B4 at Overleaf `9fd19e7`; W177(b) rows re-scored; the 0.50-floor row (Addendum 77, W178 items (b) and (c); zero-solve)

Manuscript: `manuscript/6a67305f25e8348fb71380c3/main.tex` at clone HEAD `9fd19e7` (sha256 `d18384d6…`, 2,511 lines,
byte-identical to `manuscript_review/main_820ee1b_fixed.tex`; letter = `response_to_reviewers_v4.tex`). The clone was
read only. Line numbers are at `9fd19e7` unless marked "e10c104".

Objective convention, every figure: Q = certified `gross_operational_cost`, settlement excluded; net = Q − terminal
salvage credit; value = Q(0) − Q(x); F = Q + I. Amounts in k€ unless stated.

## Blocked on Planner

Nothing blocks. Three items need a reading, and one is the author's:

1. **Score the prediction "(b) all applied".** Every item is present. Three are applied with changes, and none of the
   changes alters meaning:
   - **A8:** the `% [CONFIRM — W177]` salvage comment (l. 1359–1361) was not deleted. It is a comment and is not typeset.
   - **A11:** the pointer keeps a "Section~" prefix (below).
   - **B1:** the cohort-rating equation is split in two. This is the author's split at `820ee1b`, with the expert's
     follow-up hunk 2.

   B4 was declined by the author (Addendum 77). Whether A8 and A11 count as "applied" is the Planner's call. My
   reading: HELD.
2. **A11, l. 1603 — a new L row.** The text reads `Section~\ref{app:admm_updated_settings}`; the fix wrote
   `\ref{app:admm_updated_settings}`. The target is right (A.8). But under elsarticle's `\appendix`
   (`\gdef\thesection{\appendixname~\@Alph\c@section}`, `elsarticle.dtx` l. 1465 in the clone) the line renders
   "Section Appendix A.8" (static reading; nothing was compiled).
   - W178a's R2 is the same class: l. 755 `Appendix~\ref`.
   - The author fix is to drop "Section~".
3. **(c) The 0.50-floor row (author item):** AE **0.778**, EFC/day **1.08** (W155; section (c)). main.tex l. 1540 still
   prints `--- & ---` for these two cells.
   - Row as it would read: `Baseline, 0.50 floor & 258.5 & $-$59.5 & 12.6 & 4.71 & never & 0.778 & 1.08 \\`.
   - The note's legend "---: not reported" then has no other use in T8.
4. **Rows with a new difference, for the Planner's rating:**
   - **4.6-2 → M.** The count is now right, but the new sentence cites "the full claims table (\ref{app:results})", and
     Appendix E does not hold that table yet. This is the same gap as LM-9, which closes when the expert composes
     Appendix E (ruling 3).
   - **C-1 → L.**
   - **A-1 → L.**

## Prediction vs outcome

| prediction (expert, Addendum 77) | outcome |
|---|---|
| (b) all applied | **HELD.** A1–A20 and B1–B3 are all present: 20 verbatim, 3 with changes (A8, A11, B1), none of which alters meaning; 0 absent. B4 declined by the author, as Addendum 77 records. Located by content, whitespace-insensitive. Every old text a fix replaces is absent from the typeset text. |

## (b) The fixes at `9fd19e7`

Method (script `p515_s53_w178b_number_check.py`). The fixes file is parsed by block. The uncommented lines of each
`% A<n> --` / `% B<n> --` block are the new text. Each fragment is found in order inside the item's pinned line ranges,
whitespace-insensitive, with comment text kept apart from typeset text.

Three further checks back this:
- **Insertions:** inside a multi-line fix, the typeset text between consecutive lines must be only the
  `\textcolor{blue}{…}` wrapper, so text inserted inside a fix cannot pass unseen.
- **Old text:** the old text each block names must be absent from the typeset text.
- **Comment-only instructions** are declared and checked: pointers, a comment to delete, a counter line after each
  longtable.

A difference is declared with the text observed at the HEAD, which must be found there; an undeclared difference would
fail an integrity check.

| item | where | outcome | lines (`9fd19e7`) | difference / judgement |
|---|---|---|---|---|
| A1 | Abstract | verbatim | 146 | — |
| A2 | Nomenclature | verbatim | 224, 287, 345 (`\addtocounter{table}{-1}` directly after each of the 3 `\end{longtable}`); 273 (π̄_t row); 342 (t_sum row, s dropped); 313 (C^Op_s) | the daily C^Op_{y,d,m,o} row is kept at l. 312 (follow-up hunk 3) |
| A3 | Introduction | verbatim | 381, directly after "…market prices." (l. 380) | — |
| A4 | §2.2.7 | verbatim | 733 (first sentence, directly after `\paragraph{Stopping and certification}`); 734 ("priced at the expected wholesale price, must be within") | — |
| A5 | §3 opening / §3.1 | verbatim | 1136 (`Subsection~\ref{subsec:case_investment_costs}`); 1145 (label directly under `\subsection{Investment Costs}`, l. 1144) | — |
| A6 | §3.6 | verbatim | 1315 (cap clause); 1316 (cache clause) | — |
| A7 | §4 opening | verbatim | 1333; the rest of the sentence is kept (1334) | — |
| A8 | §4 opening, salvage | **with changes** | 1338–1345 (equation, verbatim); 1350 (eq:salvage_remaining_life kept); 1357 (where-clause, verbatim); 1358 ("The credit is reported…" kept) | The three-line `% [CONFIRM — W177]` comment (l. 1359–1361) was **not deleted**. It is a comment and is not typeset, so it does not alter meaning. Its points (a)–(d) are W177a C1, re-confirmed by this run. |
| A9 | `tab:res_energy_ladder_m2` | verbatim | 1493 (row `4$^{f}$ & 176.6 & 15.9 & 11.12 & 42.8 & 27.7 & 1.55`); 1502 (note); its second sentence kept (1503) | — |
| A10 | `tab:res_ageing` | verbatim | 1540 | the fix itself leaves AE / EFC "---" pending W155; values in (c) |
| A11 | §4.5 | **with changes** | 1602 (sentence, verbatim); 1603 (pointer) | `Section~\ref{app:admm_updated_settings}` where the fix wrote `\ref{…}`. The target is right; it renders "Section Appendix A.8" (static). **L**, wording. Every other appendix pointer of the fixes is written without a prefix. |
| A12 | §4.6 + T7 caption | verbatim | 1664, directly after "…every difference is determinate." (1663); 1670 (caption clause after "$I$ paid in 2025;") | — |
| A13 | §4.6 salvage | verbatim | 1686; the `[CONFIRM — W177] W153 totals` comment deleted (e10c104 l. 1522–1525) | — |
| A14 | §4.7 | verbatim | 1717 | — |
| A15 | §4.8 | verbatim | 1734, directly after "…applied at block boundaries." (1733) | — |
| A16 | Conclusions | verbatim | 1745 (a); 1749 (b); 1753 (c), continuing "For the method, …" unchanged (1753–1754) | — |
| A17 | App. A.3 | verbatim | 1908 | — |
| A18 | App. A.4 | verbatim | 1925 (`(\ref{app:admm_updated_settings});`) | — |
| A19 | App. A.5 | verbatim | 1932 (hold origin); 1941 (C4) | — |
| A20 | Algorithm 2 | verbatim | 1962 | — |
| B1 | §2.3.2 | **with changes** | 767 (heading, follow-up hunk 1); 770–820 | The cohort-rating equation is printed as two numbered equations, S (l. 774–777, `\label{eq:cohort_rated}`) then E (l. 781–784, new `\label{eq:cohort_rated_energy}`). The fix had one equation with `\qquad`. The definitions are the same, with one more equation number. A.2's range `\eqref{eq:cohort_rated}--\eqref{eq:available_capacity}` (l. 1845) now starts at the S equation and covers both ratings. `eq:cohort_rated_energy` is defined once and cited nowhere. Typeset text has 0 duplicate labels and 0 undefined references. Each of eq:cohort_rated, eq:cohort_available, eq:cohort_available_energy, eq:available_power and eq:available_capacity is defined once. **The fix's range end:** at `9a697cc` (the HEAD the fixes were written against) the paragraph "The storage agent publishes S^Av and E^Av …" (l. 825) came after the boundary comment `% ---- C.2 New` (l. 822), so it lay outside B1's range. It is retained at l. 823–826; the comment line itself was removed (not typeset). Alters meaning: no. |
| B2 | §2.3.4 | verbatim | 1051 (follow-up hunk 4 completed the sentence) | — |
| B3 | §2.1 | verbatim | 524 | — |
| B4 | §2 opening | **declined** (author; Addendum 77) | 411 | "%The decomposition is computational; it is not a leader--follower structure." is still commented out |

**Changes since `e10c104` outside the fixes**, from the difflib map of main.tex (82 hunks):
- every hunk from l. 1051 on maps to a fix;
- two nomenclature hunks are not fixes: a new 𝒴^Act_{e,y} row (l. 221) and the φ^Min/φ^Max row reworded (l. 257);
- the §2.1–2.3 rewrite (l. 537–967) is W178a's (`4504ebf4`).

## Re-scored rows: W177(b) at `e10c104` → `9fd19e7`

Sources: `P5_15_W177B_PART01_E10C104.md` (Parts 0–1) and `P5_15_W177B_PART2_E10C104.md` (Part 2). Added: the manuscript
side of the letter rows that describe main.tex text, and the decision-6 Part 3 rows the fixes address. A "still" status
requires the `e10c104` text at the HEAD; "resolved" requires it gone and the new text present. Each `e10c104` text was
verified present at `e10c104`. All 31 rows are consistent with their texts (integrity check).

| row | e10c104 | rating then | status at `9fd19e7` | rating now | evidence |
|---|---|---|---|---|---|
| P1-M1 / P0-C7 "two verdicts" | 1520 | MISMATCH (H in Part 2) | **resolved** — l. 1686 "changes no sign and one verdict" | — | token "one" = W153 `result.totals.n_verdict_differs` 1 (`215fd21d`) |
| P0-C1 salvage equation | 1186 | expert item (printed form 4.22× / 1.86× the recorded credit) | **resolved** — l. 1341–1343 | — | the printed form reproduces the recorded credit exactly (salvage, below) |
| P0-B3 J 4 MWh row | 1332 | author item | **resolved** — l. 1493 `176.6 & 15.9 & 11.12`; note l. 1502 "certified threshold would be 9.7 (18.14×)" | — | all four tokens match frozen `claims[J:e4_value_minus_I]`: `uncertified_form_beside_A64_ruling4.bar` / `.multiple_Q`, `gross_threshold_or_bar` / `gross_multiple` |
| P0-C5 0.50-floor row | 1378 | author item (row unfilled) | **changed** — value 258.5 filled (derived, holds); AE / EFC/day still "---" | author item | (c) gives 0.778 / 1.08 |
| P0-B5 literal "Subsection~3.1" | 985 | pointer | **resolved** — l. 1136 `\ref{subsec:case_investment_costs}`, label l. 1145 | — | — |
| 4.2-1 "every evaluated 4 h configuration … exceeds its investment cost" | 1316 | L | still (l. 1477) | L | not addressed by A/B |
| 4.2-2 "better than seven … within resolution of the other six" | 1316 | L | still (l. 1477) | L | not addressed |
| 4.3-1 "persists across the tested ageing calibrations and both floor settings" | 1392 | L | still (l. 1554) | L | not addressed |
| 4.5-1 "The planning search again returned x = 0" | 1439 | **H** | **resolved** — l. 1602 "The planning search was not repeated at this instance; the reference unit was evaluated against the plan without storage." | — | W177b-2′ (e): only the pair and the x = 0 continuation ran on the 3 × 3 |
| 4.6-1 discount rows not re-solved | 1499 | M | **resolved** — l. 1664 "…re-weight the per-year costs of the 2 % evaluations rather than re-solving them; this is exact at the plan without storage and first-order at the reference unit…"; T7 caption l. 1670 | — | The rate enters Q only through the block weights (`_get_admm_block_weight`, W177b-2′ LM-7). At x = 0 no state links the years. At the unit the 2 %-optimal schedule is held, so the error is first-order-small (envelope argument, local optimum). Its size is unmeasured; the margins are 2.57–8.78×. |
| 4.6-2 "It changes two verdicts" | 1520 | **H** | **changed** — the count is resolved (one verdict among the 60; the year pair "not among the 60"). New: the sentence cites "the full claims table (\ref{app:results})", and Appendix E (l. 2384) still holds only the submitted per-block table (`tab:cs3_results_summary_years_days`, l. 2493) | **M** (until Appendix E is rebuilt) | The year pair 3.46× → 0.19× is "to within resolution" as written. Not stated: its sign also flips net of salvage (+42.5 → −2.3). F2 neighbour: `gross_verdict` "within the uncertified bar", 0.947×; net determinate 1.223× (frozen claims; W153). |
| C-1 "each evaluation is certified" | 1581 | M | **changed** — l. 1745 "…either certified by an explicit stopping rule or reported in an uncertified form with its error bar" | **L** | The uncertified cells are now covered. Remaining: the 3 × 3 pair behind §4.5's comparison (production exit; resolution 18.0 k€) and the search evaluations behind §4.1's "beyond the search resolution" are neither. The §4 opening (A7) names that third status; this sentence does not. No number or verdict changes. |
| C-2 "every other evaluated plan … loses determinately" | 1585 | M | **resolved** — l. 1749 "every other plan of Table~\ref{tab:res_value_ladder}, from 0.5 to 5 MWh and at one or two nodes, also loses determinately" | — | The table's 15 other plans span per-node E 0.5–5 MWh at 1–2 nodes (tokens derived from the frozen cells), and all 15 lose determinately (frozen gross verdicts; B:n7_4h_e3 also determinate in its uncertified form, 2.94×). The §4 table is unchanged since `e10c104` (no hunk). |
| C-3 "cannot be assessed apart from its ageing or from the coordination" | 1589 | M | **resolved** — l. 1753 "ageing decides the value of shared storage on this system, the same unit being at break-even without it and losing with it; and the coordinated operation … is itself the larger benefit…" | — | T8: no ageing −4.1 within resolution (0.32×); every aged calibration determinate (2.50–5.74×). T6: 90.9 M€. The unit pays at 1.75× / 2× the flexibility price (l. 1449). |
| §4 preamble flag "Every reported number is a certified recourse evaluation…" | 1180 | flag (expert) | **resolved** — l. 1333 "…its status is stated: certified, uncertified with its bar, or, at the multi-scenario instance, stopped by the production exit" | — | — |
| S3-12 "Every candidate plan is compared with the plan without shared storage" | 990 | L | still (l. 1141) | L | not addressed |
| LM-4 5 × 5 future work (manuscript side) | — | M | **resolved** — l. 1734 | — | memory fact: REVISION_CONTEXT.md l. 446 / 464; Addendum 44 |
| LM-5 abstract chemistry | — | M | **resolved** — l. 146 "utility-scale lithium-ion storage" | — | — |
| LM-6 introduction uncertainty scope (M2) | — | M | **resolved** — l. 381 | — | — |
| LM-7 R1.4 "exact" (manuscript side) | — | M | **resolved** — l. 1664 | — | as 4.6-1 |
| LM-8 no convergence guarantee | — | M | **resolved** — l. 733 | — | the letter's wording is W178c's |
| LM-9 Appendix E claims table | — | M (until the rebuild) | still — l. 2384–2493 hold the submitted table | M (until the rebuild) | now also cited by l. 1686 (row 4.6-2) |
| LM-10 "representative-day-weighted annual" | — | M | moot in main.tex — ruling 3 trims the letter (L4) | — (letter side W178c) | "representative-day-weighted" absent from main.tex, typeset and comments |
| LL-8 penalty-schedule limitation in §4.8 | — | L | still | L | "different local solution", "penalty schedule": 0 hits |
| LL-9 salvage equation vs production | — | L | **resolved** — l. 1357 "the converter is not credited" | — | as P0-C1 |
| A-1 hold origin / cap for continued evaluations | 1767 | M | **changed** — A.5 l. 1932, A.3 l. 1908, §3.6 l. 1315 name the three cases | **L** | Re-runs: hold from k₀ + 1 and cap N_old + 100 (every fixed-cap cell in specs v6 `96c23404`, ext v3 `84775dc4`, A64 `44a2dce8`); these now match. Remaining: for the three reference continuations the text holds "from that run's stopping cycle", while the hooks hold "for every cycle > N" (`p515_s53_w101_settling_continuation_hooks.py:19`). One cycle's wording, unchanged since `e10c104`. |
| A-3 C4 "at the interface price" | 1775 | M | **resolved** — l. 1941 "…active-power schedules at the expected wholesale price of each hour, block-weighted"; l. 734 | — | `p515_s53_w118_resettle_hooks.py:337–369`: t_sum = Σ w·π̄·(p_DSO − p_TSO), active power only. **Correction to the committed JSON row text:** it says "'the interface price' no longer occurs in main.tex". It does occur, at §4.8 l. 1725 ("The interface price λ, the consensus dual of the interface power") and l. 1758 ("degenerate interface prices"). Both name λ, the dual; C4 no longer uses the phrase, so the status is unaffected. |
| nomenclature (π_t, s, C^Op_s) | 271 | author item | **resolved** — l. 273, 342, 313 | — | — |
| stale pointers (A.5 / A.8) | 1555 | author item | **resolved** — l. 1717, 1962, 1925, 1603 | — (A11 keeps "Section~", L) | — |
| §3.6 cache clause | 1164 | author item | **resolved** — l. 1316 | — | s47: 30 single-node ladder points, 15 over budget; s51 / s53 hold only committed certified records at m = 2; s53's 13 multi-node plans are all 2030 (tokens checked). Not stated, unchanged: s47's cache also held c_star and x = 0 (L at most). |
| table numbering | — | author item | **resolved** — 3 × `\addtocounter{table}{-1}` | — (static) | the pointers are W178c's |

**Net at `9fd19e7`, main.tex rows of W177(b):**
- **H 0.** Both H rows are resolved or changed.
- **M 2.** 4.6-2 and LM-9, both waiting on the Appendix E claims table.
- **L:** 4.2-1, 4.2-2, 4.3-1, S3-12 and LL-8 (not addressed); C-1 and A-1 (changed); new from the fixes, the A11 pointer.

### Number check over the addressed passages

**Scope:** the 9fd19e7 lines each fix lands on (144 lines, pinned; each range must hold a located fragment). Every token
on those lines is assigned:
- 37 new declarations;
- 3 declarations relocated unchanged from W177b-1′ (lines equal to `e10c104`);
- carried specs, by `e10c104` line, built by W177b-1′'s own functions. Those specs are evaluated now, not copied.

| tokens | match | derived | approximate | MISMATCH | no counterpart | unchecked | excluded |
|---|---|---|---|---|---|---|---|
| 207 | 168 | 7 | 4 | **0** | 0 | 26 | 2 |

- **Derived, 7 of 7 hold:**
  - 108 (l. 146 and l. 1749);
  - **258.5** (l. 1540): I + (value − I) = 317,957.01 + (−59,500.72) = 258,456.29 €. This is consistent with W155
    `scored.claims.E:soh050:value_minus_I` ref.Q − other.Q;
  - 7 % and 9 % (l. 1749): 1 − 0.933563 and 1 − 0.908972;
  - 90.9 M€ (l. 1749);
  - 654 M€ (l. 1754).
- **Approximate, 4** (printed roundings, unchanged from W177b-1′): 64 (abstract), 171 / 193 / 254 (Conclusions).
- **Unchecked, 26:** language (one, two, single), compound adjectives, notation (the 1 of (1 + r) and of
  (1 − SoH^Min); the 0 of max{0,·} and of Q(0)), and the `\addtocounter` argument −1 (×3).
- **Excluded, 2:** the label "W177" in the retained comment, and "(C4)".
- **New sentences:**
  - §2.2.7 l. 734: τ/2, matching the gap bound.
  - §4.5 l. 1601–1603: three / three / five / three (3 × 3 case); 236.7, 318.0, 81.3, 4.52, 18.0 (T11); ten (the pair
    campaign spec).
  - §4.6 l. 1664 / 1670: the 2 % entries match `SRP1.json` DiscountFactor. l. 1686: 60 and one (W153); 0.95 and 1.22
    (frozen claims).
  - §4.8 l. 1734: 3 × 3 from `SRP1__s53_3x3.json`; 5 × 5 from the submitted main.tex l. 673.
  - Conclusions l. 1749: 0.5 / 5 / one / two are derived from the plans of tab:res_value_ladder.
- **§3.6 l. 1315–1316:**
  - 100 = cap − N_old for every fixed-cap cell of the three frozen re-settling specs;
  - 109 and 300 come from the dynamic-cap cells;
  - 4 / 20 / two / 60 / 30 / 18,449.66 / ten / 500 / two / three / 60 come from the W177b-3′ run settings;
  - two caches at m = 2; 2030.
- **J row (A9):** l. 1493 and l. 1502 print 15.9 / 11.12× (the uncertified bar, Addendum 64 ruling 4) and 9.7 / 18.14×
  (the certified threshold, flagged). All match the frozen fields above.

### Salvage equation (A8) against the code

`shared_energy_storage_data.py:866–905` (`_build_terminal_salvage_value_expression`, sha256 `9acd095f…`, committed
clean). The 17 cited code lines were checked by content.

| printed (l. 1338–1345, 1357) | code |
|---|---|
| Σ_e Σ_{y^Inv} | one ESSO model per node; `for y_inv in model.years` (l. 876) |
| c̄^{Inv,E}_{y^Inv} = Σ_c ω_c c^{Inv,E}_{y^Inv,c} | `_get_expected_energy_investment_cost` (l. 895, 925–929): probability-weighted **energy** cost of the installation year |
| E^Inv | `e_rated` (l. 883) |
| 1/(1+r)^{y^End − y_0} | `_get_terminal_discount_factor` (l. 872, 910–912): 1/1.02^15 = 0.743015 |
| T^Rem/T^Cal, T^Rem = max{0, y^Inv + T^Cal − y^End} | `remaining_life_fraction` (l. 896–898, 915–922) |
| (SoH − SoH^Min)/(1 − SoH^Min) | `usable_energy_above_eol = (e_available − min_soh·e_rated)/(1 − min_soh)` (l. 890), with terminal end-of-block SoH (l. 884 / 889) |
| converter not credited | no power-cost term in l. 866–905 |
| (not printed) `energy_recovery_fraction`, `recycling_floor_fraction` | 1.0 and 0.0 in `SRP1_ESS_Params.json` `salvage_value`, so the printed form equals the coded form at these parameters |

**Reproduction by the printed form:**

| cell | inputs | printed form | recorded | rel. diff | old printed / recorded |
|---|---|---|---|---|---|
| yl_y2030 | node 7, 0.25 / 1 MWh; c̄ 208,198.72 €/MWh (ω 0.35 / 0.55 / 0.10); T_rem 5 of 15; SoH 0.768414 → health 0.228048 | 11,759.257933 € | 11,759.257933 € (frozen `year_ladder.per_year.2030.salvage_last`) | 0 | 4.220 |
| yl_y2035 | c̄ 192,708.12 €/MWh; T_rem 10 of 15; SoH 0.877581 → health 0.591935 | 56,504.166416 € | 56,504.166416 € (`…2035.salvage_last`) | 0 | 1.857 |

**Independent check:** I also recomputed both values by hand, outside the script. c̄ came from `SRP1_ESS.xlsx` sheets
"Investment Cost, Energy" and "Scenarios", the floor 0.70 from `SRP1_ESS_Params.json`, and the SoH from each cell's
`boyd_terminal.json`. Both values were identical to the last digit.

## (c) The 0.50-floor row

- **Value:** 258.5 k€ = 318.0 − 59.5. Exactly 258,456.29 € = I 317,957.01 + (value − I) −59,500.72 (frozen
  `tables.a64.rows[E:soh050:value_minus_I]`). The 258.5 at l. 1540 holds.
- **Record:** `data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json`
  - sha256 `3c71d71dd3266880a18c14acc0ed3d477afe01c28f4d71f5883faccefad3122a`, last commit `52d416cb`;
  - fields `reports.e_soh050.AE` = **0.7781072297862642** and `reports.e_soh050.EFC` = **1.076295478054417**;
  - cell e_soh050: arm C2_calfade, `minimum_soh` 0.5, certified at k* 184, plan node 7 0.25 / 1 MWh 2025;
    `floor_year` null, hence "never".
- **Recomputed from the primary record:**
  `…/campaign_s53_w155_a64_r2_e_soh050/evals/3f01b9eaaab13c82_e_soh050/evaluation_record.json` (sha256
  `ab3573fc…`, commit `5fb81cc4`), with the same reader and formula. The values are bitwise equal.
  - soh_used = 0.872631 / 0.765059 / 0.677290;
  - efc_per_day = 1.191924 / 1.099939 / 0.909240 (2025 / 2030 / 2035);
  - weights 1 / 1.02^(y−2025) = 1 / 0.905731 / 0.820348.
- **The definition matches T8:**
  - The frozen JSON (`590088fe`) holds T8's columns as `tables.ageing.rows[].AE` / `.EFC`. Its `column_labels` carries
    no AE / EFC label, only `eps_AE_050_superseded`.
  - Chain: W157 builds T8 from the W142-ext summary (`p515_s53_w157_step6_tables.py:662`, source l. 129). W142-ext
    computes `'AE': M46A.pv(series['soh_used'])` and `'EFC': M46A.pv(series['efc_per_day'])`
    (`p515_s53_w142_resettle_ext_v6_campaign.py:1096–1097`). Its definition text: "pv(SoH_used per block): Σ w_y
    SoH_used[y] / Σ w_y, w_y = 1/1.02^(y−2025); SoH_used = soh_used_for_available_energy of node 7, cohort 2025".
  - W155 computes the identical expression through the same reader `LX.ageing_series`
    (`p515_s53_w155_a64_campaign.py:837–839`).
  - `pv` = `sum(w*v)/sum(w)` (`p515_s46_ageing_mechanism.py:45, 54`).
  - **Numerical identity:** W155's 0.70 neutrality cell gives AE 0.7927159115133644 / EFC 0.8635062982587755, exactly
    T8's baseline row (C2_calfade).
  - The manuscript note matches: "$AE$ and EFC/day are the present-value-weighted available-energy fraction and
    equivalent full cycles per day".
- **At T8's rounding:** **AE 0.778, EFC/day 1.08.** T8 prints 3 dp / 2 dp, e.g. baseline 0.793 / 0.86 and no ageing
  1.000 / 1.29.
- **main.tex at `9fd19e7`:** **does not print them.** l. 1540 reads
  `Baseline, 0.50 floor & 258.5 & $-$59.5 & 12.6 & 4.71 & never & --- & ---`.

## Not confirmed

- **Rendering.** Nothing was compiled; there is no TeX here. The A11 "Section Appendix A.8" reading is static, from
  `elsarticle.dtx` l. 1458–1465. So is the table-counter effect of A2 (the expert's compile in ruling 5 reports
  Tables 1–15).
- **The size of the first-order error in A12 / 4.6-1.** Re-solves at 0 / 5 / 8 % would be needed; none was run.
- **Number-check scope.** The 144 addressed lines only; the rest of main.tex was not re-checked at `9fd19e7`. W178a
  covers §2.1–2.3, and W178c covers the letter.
- **B1's range end** rests on `9a697cc` being the HEAD the fixes were written against (the fixes file header). It was
  read with `git show` in the clone.
- **Terminal SoH of the salvage cells.** Read through W177a's reader from `boyd_terminal.json`. It is corroborated by
  the exact reproduction of the recorded credit, not by an independent parser.
- **Rows not re-assessed:** M/L ratings carried from W177b-2′ / W177b-3′ for rows the fixes do not touch.

---

## Changed

| commit | content |
|---|---|
| `e894e109` | `p515_s53_w178b_number_check.py` (sha256 `17b5c8ce…`), a new file. It imports `p515_s53_w177b1r_number_check.py`, `p515_s53_w142_resettle_ext_v6_campaign.py` and `p515_s46_ageing_mechanism.py`, none of them edited. |
| this commit | outputs in `data/SRP1/Results/P515S53/w178b_number_check/overleaf_9fd19e7/` + this report |

The script started as the previous Worker's uncommitted draft. I added two things:
- the residue check, so that text inserted inside a multi-line fix fails unless declared (the draft matched lines in
  order with arbitrary gaps);
- the B1 range-end check against `9a697cc`, the LM-10 row, and a fuller cache-clause evidence string.

It was trial-run three times into the session scratchpad before the commit.

**Outputs** (written once into a new directory):

| file | sha256 |
|---|---|
| `w178b_number_check.json` | `b807c5b959a674e9…` |
| `w178b_number_check.md` | `8e9f06773e5395d9…` |
| `manifest_sha256.json` (93 entries: outputs, script, 84 inputs, the clone's *.tex + .bib) | `02e70a87f61ec144…` |
| `launch.log` (stdout + stderr) | `a433da05cc3ceebc…` |
| `w178b_bool_typing_test.json` / `.log` (W100: **PASS**, 129,368 files) | `3725cf94…` / `46eb0a6a…` |
| `manifest_post_run_sha256.json` | `9286d6378399133f…` |

**Guards.**
- **Zero solves:** `SolveProfileGuard(permitted=())`, together with all 46 guards the import chain arms, verified at
  exactly 0.
- **No model loads:** `pickle.load` / `pickle.loads` were blocked for the whole run, with 0 calls.
- **Integrity:** 29 of 29 checks are True, and the exit code is 0.
- **Read-only:** git reads only; the clone was not pulled or modified. No production file, spec, case, frozen JSON,
  manuscript, `manuscript_review/` file or earlier script changed.

**Unexpected findings** (not acted on):
1. Two other `% [CONFIRM — W177]` comments remain, at l. 1641 (G rows) and l. 1712 (W173). No fix addresses them; they
   are comments, not typeset.
2. Two nomenclature edits at `9fd19e7` are not part of A/B: the new 𝒴^Act row (l. 221) and the reworded φ row
   (l. 257). They were not scored here.
3. A false statement went into the committed JSON row evidence for A-3. It says "the interface price" no longer occurs
   in main.tex, but it does (l. 1725, 1758). Corrected in the A-3 row above. The status and every check are unaffected.
   I did not re-run, because the output directory is write-once.
