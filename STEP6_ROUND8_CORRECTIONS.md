# Step 6, round 8 — claims, consistency and submission files after W182 (expert, 2026-10-26)

Rulings: `PLANNER_BRIEF_2026-09-13.md` Addendum 83 ruling 5 and Addendum 84. Base: Overleaf `edd7a63` (`main.tex` md5
`137e8095…`, 2,956 lines; letter = v9, md5 `35d95269…`; verified byte-identical to the clone on the Air). Independent of
the Step 7 campaign; the results-dependent text (§4.3 E1 table, §4.4 E3, §4.7 E2, letter R2.5/R2.6/R3.6, cover-letter
paragraph, marked copy) is round 9.

**Deliverables (`manuscript_review/`):** `main_edd7a63_round8.tex` (full `main.tex`; + `main_edd7a63_round8.diff`, 31
hunks), `response_to_reviewers_v10.tex` (+ `letter_v9_v10.diff`, 12 hunks), `highlights_v3.tex` (+ `highlights_v2_v3.diff`),
`cover_letter_v2.tex` (+ `cover_letter_v1_v2.diff`), `gen_appE.py` (patched), `appendixE_evaluation_record.tex`
(regenerated), `verify_appE.py` unchanged (33 PASS).

**Compile check (demo graphics, `elsarticle.cls` rebuilt):** `edd7a63` baseline 0 errors, 0 undefined, 75 pages, 136
labels; round 8: 0 errors, 0 undefined, 0 multiply-defined, 76 pages, every label number identical to `edd7a63`
(verified from the `.aux`, not from the static map); citations unchanged ([42] Naumann, [43] EVE, [44] Boyd). Letter v10,
highlights v3 and cover letter v2 compile alone. The generator reproduces the `edd7a63` block bitwise before the round-8
edits, and the new block equals the patched generator's output.

## A. Appendix E (generator; Addendum 83 ruling 1)

1. The author's six hand-edited hunks of `edd7a63` ported into `gen_appE.py`: the new E.1 "Terminal Salvage Credit"
   (`app:salvage_credit`, Eqs. (E.1)–(E.2)), the E.21 group-(d) header (`Fig.~\ref{fig:res_flex_ladder}`), the lattice
   paragraph and `fig:investment_lattice` in E.4, the E.5 passive-arrangement sentence and three-phase paragraph, the
   four E.7 paragraphs (settings, search statistics, closure slacks, acceptance statistics), the trailing blank line.
   Bitwise reproduction verified.
2. Convention: "gross operational cost, **contracted interface settlement** excluded" in the intro, the E.20, E.22 and
   E.23 captions.
3. Superscript $^{b}$ on the seven over-budget 2025 evaluations: E.21 rows 5, 6, 9, 10, 11, 38, 39; E.20 rows 4, 5, 8,
   9, 10, 26, 27, 30, 31; legends in the E.20 and E.21 captions ("investment above the 1 M€ budget, a valuation probe
   outside the admissible set (Section~\ref{sec:case_settings})"). 0 table entries changed.
4. E.1 salvage paragraph: $y^{\text{End}} = y_0 + \sum_y Y_y = 2040$ (1 January 2040, both configurations) and the
   terminal SoH defined as the end-of-block SoH of the last representative block (2035; 2037 at 3 × 3) (W182g C2).
5. E.7: "Certification (Subsection~\ref{subsubsec:certification}) uses δ_R…" → "The acceptance
   tolerance~\eqref{eq:acceptance_tolerance} uses δ_R…" (W182b).
6. Lattice-figure caption re-wrapped in blue.

## B. `main.tex` outside Appendix E (21 edits, all blue)

| id | place | edit |
|---|---|---|
| M1 | abstract l. 145 | sentences 1–2 rewritten: degradation-aware framework for siting, sizing and installation year of shared **lithium-ion** ESSs, formulated as a two-stage stochastic program with one scenario-independent plan; cost uncertainty = probability-weighted trajectories of future converter and battery costs; the search runs in a single operating scenario, fixed plans are re-evaluated under scenarios (W182d M-ABS-1/2; audit §3.1, §6) |
| M2 | highlight 1 l. 163 | "A two-stage framework plans shared lithium-ion storage at TN--DN interface nodes" (M-HL-1) |
| M3a | contribution 1 l. 387 | one scenario-independent plan; searches on a single operating scenario; fixed plans under scenarios (M-CON-1, D4) |
| M3b | contribution 4 l. 390 | "the role of degradation in the investment decision" → "the effect of degradation on storage value and available capacity" (audit 3.1; revisited in round 9 if E1 resolves) |
| M4 | §2.2.6 l. 724 | "load curtailment" removed from the $C^{\text{Op}}$ list (never built: `l_curt` false in every network file, W182g) |
| M6a | §3.5 l. 1350 | "excluding interface settlement" → "excluding the contracted interface settlement" |
| M6b | §3.5 | new sentence: the five over-budget 2025 rungs (4 h: 4, 5 MWh; 2 h: 3, 4, 5 MWh; 1.15–1.91 M€) are valuation probes outside the admissible set, marked $^{b}$; no returned plan or certified comparator is among them (W182d A8) |
| M7 | §4.2 l. 1425 | the fit's support: five of ten configurations over budget, the fit describes marginal value, not an admissible plan |
| M8 | Fig. 5 caption l. 1431 | over-budget configurations named as probes |
| M9a–c | §4.2 l. 1467; Table 6 | "the 4 and 5 MWh configurations exceed the budget and are valuation probes"; rows $4^{f,b}$, $5^{u,b}$; note legend $^{b}$ |
| M10a–j | Tables 3, 4, 6, 7 | `\textcolor{blue}` restored on the four captions and the three notes (23 lines of W182c A5) |
| M11 | §4.3 l. 1573 | floor binding year qualified: single-scenario 2035; 2037 in the multi-scenario evaluation (W182g) |
| M12 | §4.5 l. 1636 | one sentence: at 3 × 3, $Q$ includes the deviation settlement and the commitment premium (about 3 % of $Q$ at x = 0); the contracted settlement is excluded (Addendum 83 ruling 2; W182 decision 2) |
| M13a–b | Fig. 8 caption l. 1619; §4.4 l. 1582 | "contracted interface settlement" |
| M14 | Appendix A.1 l. 1832 | $P^{\text{I}}$ positive from the TN to DN $i$, the DN's import net of the co-located shared ESS, whose schedule has its own channel (W182a) |
| M17 | §3.3 l. 1290 | ADN growth 1.25 %/3.00 % made explicit; TN 1.00 %/2.50 % added (W182b, `case9_operational_data.xlsx`) |
| M18 | §3.3 | the lost sentence restored: "Conventional generation in the TN is priced at the wholesale energy price of the scenario; no separate generator cost curves are used." (verified against code in W178; the letter's R1.4 target) |

Not changed: §4.4's 13.9 % (correct as a cost increase); the "two-stage stochastic" wording where it describes the
formulation (D4); the per-year EFC/day (Addendum 78 ruling 2).

## C. Letter v10 (80 line edits; 12 hunks)

- **Pointers** renumbered from the W182c label map and verified against the round-8 `.aux`: §2.2.7 → 2.2.6 (11), §3.5
  → 3.4 (7), §3.6 → 3.5 (3), §2.3.4 → 2.3.3 (5), §2.3.3 → 2.3.2, §2.3.5 → 2.3.4, "Sections 2.3.3–2.3.5" → 2.3.2–2.3.4,
  "Sections 2.2.6 and 2.3.2" → "Sections 2.2 and 2.3.1" (the coupling subsubsection is gone), Tables 7–15 → 4–11,
  Table 15 → 11, 12 → 8, 14 → 10, 6 → 3, 11 → 7, Figures 3 → 2 and 7 → 5, Appendix E.6 → E.7, Eqs. (29)–(35) →
  (29)–(37), Eqs. (38)–(39) → (E.1)–(E.2), [43]/[44] → [42]/[43]; the removed search figure dropped from R2.2 and the
  "Further changes" bullet; Figure 4 → Fig. E.19; "new 2.2.7 ``Recourse evaluation and certification''" → "new 2.2.6
  ``Recourse Evaluation and Numerical Acceptance''". R1.4's "(Section 3.3)" now has its target again (M18).
- **12.2 % / 13.9 %:** l. 53 and l. 80 state the saving as 12.2 % of the price-taker arrangement's cost and 13.9 % as
  that arrangement's cost increase (W182d A8).
- **Abstract descriptions** (l. 76, 77, 80) made true of the round-8 abstract: it names lithium-ion, says what the
  cost uncertainty is, and reports 12.2 %.
- **Convention** l. 81: "contracted interface settlement excluded".
- **The nine A7 sentences:** L-01 (search under the residual exit, reported comparisons under the acceptance rule);
  L-04 (3 × 3: "by more than the reported numerical resolution … stopped by the production exit and not certified";
  "0.933 predicted" removed, it is no longer in the paper); L-05, L-07, L-09 ("certified" → declared acceptance
  status / threshold or bar); L-08 (the rule assesses stopping behaviour; neither bounds the distance to an optimum nor
  selects the basin); L-10 (τ is a declared tolerance, not a bound on the error of a ratio); L-12, L-13 and l. 58
  ("within the numerical resolution of zero, so its sign is not resolved"); L-14 ("no effect of the floor on the value
  is resolved"); L-00 ("every other plan assessed under the acceptance rule").
- **R3.6 scope** (audit 3.2): the argument that the affine ladder makes re-optimisation unnecessary is withdrawn at
  l. 58 and l. 264; the fixed-plan comparison "establishes the effect of degradation on storage value and available
  capacity, not on the selected plan; the planning part of this request is met only in part". Round 9 adds E1.
- **R3.4 correction** of v9's own error: the datasheet calibration loses *less* than the baseline (−45.2 vs −64.4 k€);
  "it loses more" → "it also loses under the datasheet calibration (−45.2 k€ against −64.4 k€)".
- Left as is: the factual uses of "certified/certification" for the rule's defined status (l. 158–167, 285, 288); the
  supplementary prediction-record bullet (l. 286) — author's call whether to keep it (audit §4 suggests an internal
  log).

## D. `highlights_v3.tex` and `cover_letter_v2.tex`

- Highlights: the five bullets of `main.tex` (12.2 %; no "every evaluation is certified"; highlight 1 as M2); replaces
  the stale standalone file (H-3, H-4).
- Cover letter: rewritten for the revision — what changed and why (invalid cuts, battery physics, corrected inputs:
  the ÷5/÷4 energy-cost error stated plainly, the transfer term), that the conclusions changed in kind, the 12.2 %
  figure with its restriction, the two partial responses, and one bracketed paragraph for the E1–E3 results (round 9).

## E. Author's steps

1. Overleaf: paste `main_edd7a63_round8.tex` over `main.tex`, `response_to_reviewers_v10.tex` over
   `response_to_reviewers_draft.tex`, `highlights_v3.tex` over `highlights.tex`, `cover_letter_v2.tex` over
   `cover_letter.tex`; compile (0 errors; table/figure/equation numbers unchanged).
2. Pull the clone; stage by filename the files listed under Deliverables plus `STEP6_ROUND8_CORRECTIONS.md`,
   `PLANNER_BRIEF_2026-09-13.md`, `manuscript_review/W177B_CHANGE_LOG.md`; commit
   (`Round 8: claims and consistency after W182; letter v10; highlights v3; cover letter v2; Appendix E generator port (Addendum 84)`);
   push; pull on the Studio.
3. From now on, Appendix E prose is edited through the generator (Addendum 83 ruling 1): send me the sentence.
4. The W183h check (Addendum 84) can run alongside W183a–g; forward its line when you forward the next prompt.
