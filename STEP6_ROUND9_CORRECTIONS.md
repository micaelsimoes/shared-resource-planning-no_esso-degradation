# Round 9b — trimmed manuscript, letter v12, cover letter v3, Figure 6 (expert, 2026-11-10; Addenda 89 ruling 7 and 90)

**Round 9b supersedes round 9 / letter v11 (never pasted).** It adds, from the W184/W185/W188 records (Addendum 90): the §4.4 NRF sentence, the §4.7 penalty-sensitivity paragraph, the Appendix B.4 slack sentence, the ruling-5 wording ("the rows of inactive cohorts are deactivated"), and the new Figure 6 (stored energy and state of health of the reference unit) with its paragraph in §4.3; letter v12 adds the same in R2.6, R3.2 and the further-changes list, discloses the Appendix D ±100 correction, and fixes v11's benchmark pointer (Table 7).

Base: Overleaf `a87e195` (`main.tex` md5 `39f6fd86…` = `main_edd7a63_round8.tex`), letter v10, cover letter v2,
highlights v3 (unchanged this round). No number was typed: the two break-even figures moved from Table 8 into prose and
the rows of the merged year/discount table are copied verbatim from the removed tables (checked by script); every other
number is untouched.

## Deliverables (all in `manuscript_review/`)

| File | md5 | What |
|---|---|---|
| `main_a87e195_round9b.tex` | `49139cc2…` | full manuscript, paste over `main.tex` |
| `main_a87e195_round9b.diff` | — | unified diff vs `a87e195` (34 hunks) |
| `response_to_reviewers_v12.tex` | `ed8f5b13…` | paste over `response_to_reviewers_draft.tex` |
| `letter_v10_v12.diff` | — | 10 hunks |
| `fig_soc_soh_reference_unit.pdf`, `.py` | — | new Figure 6; upload the PDF to Overleaf; the script reads the three W188 (d) CSVs |
| `supplement_README_prose_v1.md` | — | README prose for the supplement generator (W189a) |
| `cover_letter_v3.tex` | `0b9e4c21…` | paste over `cover_letter.tex` |
| `cover_letter_v2_v3.diff` | — | 1 hunk |
| `main_round9b_preview_demo_graphics.pdf` | — | container compile, figures as demo boxes (layout check only) |

Compile (container, elsarticle rebuilt from `manuscript_submitted/elsarticle.ins`, `\PassOptionsToPackage{demo}{graphicx}`,
pdflatex → bibtex → pdflatex ×2): **0 errors, 0 undefined references, 0 multiply-defined labels; 63 pages (was 76).**
Letter v12: 0 errors, 13 pages. Cover letter v3: 0 errors.

## Numbering changes (intended; everything else identical to `a87e195`)

Equations (1)–(40) of the main text: unchanged. Figures 1–5: unchanged; **new Figure 6** (`fig:res_soc_soh`, §4.3); `fig:res_benchmark` 6 → 7; `fig:res_q_cycle` 7 → 8. Tables 1–4: unchanged.

| Label | `a87e195` | round 9 |
|---|---|---|
| `tab:res_breakeven` | Table 5 | removed (two numbers moved into the §4.2 prose) |
| `tab:res_energy_ladder_m2` | 6 | 5 |
| `tab:res_ageing` | 7 | 6 |
| `tab:res_benchmark` | 8 | 7 |
| `tab:res_multiscenario` | 9 | 8 |
| `tab:res_year_ladder` | 10 | 9 — now two panels, (a) installation year, (b) discount rate |
| `tab:res_discount` | 11 | removed (merged into Table 9(b); refs retargeted) |
| §4.8 Limitations | 4.8 | merged into §4.7 "Numerical Performance, Scope and Limitations" |
| Appendix A.7 Scenario mechanics | A.7 | folded into A.1 (one paragraph) |
| Appendix A.8 Settings | A.8 | A.7 |
| Appendices B, C, D (market, TN, DNs) | B–D | removed → supplementary material S1–S3 |
| Appendix E Evaluation Record | E | **B** (E.1 → B.1 salvage, eqs. (E.1)–(E.2) → (B.1)–(B.2); E.2 → B.2 comparisons, Table E.20 → B.10; E.4 → B.3 neighbours, Fig. E.19 → B.8, Table E.22 → B.11; E.7 → B.4 statistics) |
| Tables E.21, E.23, E.24; sections E.3, E.5, E.6 | printed | removed → supplementary Tables S4, S5, S6 |

Note on the compile count: Appendix B's two tables print as B.10 and B.11 because the table counter continues from the
main text (9) and the nomenclature's `\addtocounter{table}{-1}` convention is kept.

## Hunks (main text), by Addendum 89 ruling 7

1. **Round-9 queue (Addendum 86 ruling 4).** (i) The seven `Appendix~\ref{app:…}`/`Section~\ref{app:…}` prefixes → bare
   `\ref` (l. 530, 1631, 1639, 1742, 1763, 1780, 2002 of `a87e195`). (iv) "within numerical resolution of break-even"
   → "within the numerical resolution of zero, so its sign is not resolved" in §4.3 and in the Conclusions; the same in
   cover letter v3. (ii)/(iii) The E.23 `$^{s}$` and E.24 over-budget marks now belong to supplementary Tables S5/S6 —
   transferred to the supplement generator (W188 (b)). (vi) The letter's prediction-record bullet is dropped (author's
   decision of 2026-10-26).
2. **G1 — zero-capacity conventions.** Eq. (28) printed in product form, `P^Ch · P^Dch ≤ ε^C (S^Av)²` (equivalent for
   S^Av > 0, vacuous at zero with eq. (26)); one sentence after it; one sentence after eq. (32) stating that (32)–(34)
   are defined for active cohorts with positive rated energy and that storage rows without an active cohort are
   deactivated. **W188 (c) confirms against the code before this is pasted** (Addendum 89 ruling 5).
3. **G2 — first-order characterization of an accepted point.** New paragraph at the end of Appendix A.4 (fixed point of
   the sweep + block KKT ⇒ KKT of the coupled recourse with exact consensus; approximate first-order stationary,
   AC-feasible within tolerances; no global claim; sequential nonconvex scheme outside the convex/two-block theory).
   One sentence in §2.2.7 points to it. Letter v11 R2.5 and R3.2 cite it.
4. **Disclaimers 7 → 3.** "Global optimality" is now disclaimed at Algorithm 1 (§2.1), in the limitations, and once in
   A.4; the sentences at §2.1 (recourse value), §2.2.7, §4.1 and §4.7 are rewritten without it; A.5's definition of
   "certified" is shortened.
5. **§4.1** gains the sentence that Q varies almost linearly with capacity over the evaluated plans, so the ladders carry
   the planning information and the search returns a corner of the admissible set (refocus analysis §3.1, item 1).
6. **G9** — block-level discounting at the representative year stated as an approximation common to every plan (§2.2.7).
7. **G11** — one sentence after Algorithm 1: the search threshold is a cheap screen; reported comparisons are
   re-evaluated under §2.2.7's criteria.
8. **G14** — §3's restatement of the search restrictions replaced by a pointer to §2.1.
9. **Table 8 → prose; Tables 13/14 → one two-panel table; §4.7 + §4.8 merged** (headings, labels, cross-references).
10. **Appendix A compacted**: A.7 (scenario mechanics) folded into A.1 as one paragraph; the unused proximal-term
    sentence removed; the "continued from an earlier run" clause stated once (A.5) instead of three times.
11. **Appendices B–D removed**; the six main-text references (§3.2, §3.3 ×4) point to supplementary Sections S1–S3.
12. **Appendix E → B, compacted**: E.3 (61 evaluations), E.5 (static-arrangement starts) and E.6 (gap-refused) removed;
    the opening paragraph rewritten; every reference to the moved tables (main text ×9, Appendix B ×6) points to
    Tables S4–S6 / Sections S4–S6 of the supplementary material.
13. **Conclusions**: the future-work sentence names terminal value inside the planning objective and a convex-relaxation
    lower bound on the recourse (the next paper), without claiming either here.

## Letter v12 (10 hunks vs v10)

Pointers renumbered per the table above (Section 4.8 → 4.7 ×9; Tables 7/8/10/11 → 6/7/9(a)/9(b); Tables 4–11 → 4–9;
Appendix A.8 → A.7; Appendix E → B, E.7 → B.4, Fig. E.19 → B.8, Eqs. (E.1)–(E.2) → (B.1)–(B.2)); R2.5 and R3.2 carry the
first-order characterization with its location; the "Appendix E rebuilt" bullet now describes Appendix B and the
supplementary material; the prediction-record bullet removed.

## Added in 9b from the records (Addendum 90)

14. §4.4: coordinated operation re-evaluated with NRF imposed (W184): +20.5 k€, 1.62× threshold, 0.02 % of the saving; 12.2 % either way.
15. §4.7: penalty sensitivity (W185): eight cells certified; decisive difference moves 1.6 / 1.5 k€ (≈ 0.12 of threshold); signs and verdicts unchanged; seven cells inside their frozen windows, one 0.7 k€ above.
16. Appendix B.4: every penalised slack at its zero bound at every certified evaluation behind the tables; common relaxation offset −13.0 k€ cancelling to within 40 €; the two exceptions (≤ 2.2 €, ≤ 171 €).
17. §4.3: Figure 6 and its paragraph (usable band 0.70 → 0.56 MWh in 2035; floor reached in 2035 by the baseline and both 50 % arms; 0.68 under the 0.50 floor) — numbers read from the W188 (d) CSVs by the figure script and checked.
18. Ruling-5 wording after eq. (32).

## Deferred, with reason

- **G8** ("certificate" noun → "acceptance record"): the term "flagged certificate" is defined in the generator, the
  tables and the letter; a global rename this close to the freeze adds inconsistency risk for a cosmetic gain. Left as is.
- **Deeper main-text cut** (merging §4.6 into §4.5, shortening §2.3.3 prose): author's call; the main text is 6.5 k prose
  words, 9 tables + 9 figures, 40 equations. The page count is now 61 with full appendices of 3 floats.
- **Cover letter v3** keeps one bracketed paragraph; it is filled in round 10 (one sentence on W184/W185 and, if run, W187).
- **Highlights v3** unchanged.
- **Supplement** (S1–S6): generated by W188 (b); the expert writes its README. Until it exists, the manuscript's
  "supplementary material" pointers are forward references — do not submit before W188 (b) has landed.

## Author steps

1. W188 (c) confirmed the zero-capacity wording (Addendum 90 ruling 5): nothing to wait for.
2. Upload `fig_soc_soh_reference_unit.pdf` to the Overleaf project; paste `main_a87e195_round9b.tex` over `main.tex`,
   `response_to_reviewers_v12.tex` over `response_to_reviewers_draft.tex`, `cover_letter_v3.tex` over
   `cover_letter.tex`; compile on Overleaf; expect 0 errors, 0 undefined, 63 pages ± layout, and the numbering above.
3. Pull the Overleaf clone on the Air; commit the four deliverables and this file by filename; push; pull on the
   Studio; forward Addendum 90 (W189a, W187 under spec v1.1, W189 at the new HEAD).

## Round 9c (expert, 2026-11-12; Addendum 91 ruling 4) — on Overleaf `46c9d77`/`52a607d` = round 9b

| File | md5 | What |
|---|---|---|
| `main_46c9d77_round9c.tex` | `cd1433d7…` | paste over `main.tex` (8 hunks vs 9b: `main_round9b_round9c.diff`) |
| `response_to_reviewers_v13.tex` | `74e88c78…` | paste over the response (1 hunk: Fig. B.8 → B.9) |
| `cover_letter_v4.tex` | `63b7ddee…` | paste over `cover_letter.tex` (bracket filled with W184/W185; manuscript number; delay sentence; one bracket left for W187) |
| `gen_appE.py`, `verify_appE.py` | — | `--reduced` mode prints Appendix B bitwise; verifier 40 PASS (`gen_appE_round9c.diff`) |
| `supplement_README_prose_v2.md` | — | for supplement v3 |

Edits: B.4 net-power pair "penalised in the agent's own objective"; Figure 6 prose (90 % limit every day, 10 % on six of
eight, Winter 14–15 %); evaluation 16 "171 €, almost all voltage slack"; "Section S5" → "Table S5" ×2; blue on the five
lines (§3.2 pointer; Table 9 panel headings and note). Compile 0/0/0, 63 pages, numbering identical to 9b.
