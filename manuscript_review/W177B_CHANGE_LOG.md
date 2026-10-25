# Change log for the W177(b) order (manuscript changes since round 5 / Overleaf 8b76423)

Kept by the expert during the author's whole-document read (2026-10-09 onward); the W177(b) order is written from it.

## Already in Overleaf (verified by the expert by diff)
- §4 frame pasted (Addendum 73), nine tables refit (`section4_tables_refit.tex`), benchmark table reduced to five columns.
- Abstract rewritten (13.9 %; 64 k€; break-even 171–193 k€/MWh; 1.75×; 0.91–0.93).
- Introduction tightened; new citations geoffrion_1972, li_tomasgard_barton_2011, rios_sahinidis_2013, gerard_2018,
  givisiez_2020, pearson_2022, aghdam_2023, aghdam_2024, aghdam_2025; letter R2.8 reply rewritten (three references).
- §2.1: ex-ante, computational-decomposition and nonanticipativity sentences restored; salvage pointer; footnote on
  the scenario pair; Figure 1 redrawn (framework_v3.pdf: no sensitivities; certified value and resolution upward).
- §2.1: plain-language search paragraph + new Fig. `fig:search_generic` (toy, no case numbers); MADS paragraph
  rewritten (OrthoMADS by reference; settling bar defined; scope sentences on per-node timing/staging);
  "heuristic" paragraph shortened; the variant-use and three-runs sentences MOVED to §4.7 (wording "Two runs").
- Algorithm 1 compacted: run settings moved to §3.6 (production exit 10 cycles / cap 500; review stops: completion
  cap, 2 barrier points per poll or 3 in all, at most 60 polls; cache seeding); input/output lines shortened.
- §2.2: index convention y^Inv in I(x), budget, duration bounds, §2.2.6; lattice sentence moved to §2.2.2;
  §2.2.5 rewritten (budget = I(x) in expectation); closing sentence on X; nomenclature v3.
- §3.5: new Fig. `fig:investment_lattice` (single node, 2025: 12 admissible plans; budget line at 256.32 k€/MVA,
  253.88 k€/MWh) and the fourteen-neighbour sentence (7 node subsets × {2025, 2030}; W173 E2.1(b)).
- §2.2.7 / C^Op paragraph simplified by the author (571df96): tolerances and the 15–21 k€ evidence removed from
  §2.2.7 (both stated in §3.6 / §4.7); C^Op composition shortened — expert asks to restore composition and exclusions
  in compact form and the successful-solve clause of the residual pass (2026-10-13).

- §2.2.7 restructured as a method statement (expert draft `section227_certification_v2.tex`, 2026-10-13): clauses
  named (C1)–(C5) and (M1)–(M3), noise and counting details in one sentence after the list, no clause dropped;
  case-specific sentences moved: "Every certificate reported … decided, or re-decided …" → §4.7; the multi-scenario
  production-exit sentence → §4.5; δR / 𝒱 / τ values and "kept frozen" → §3.6; "ten certificates within 5 % of τ,
  ≤ 0.9τ" dropped from §2.2.7 (already in §4.7).
- §2.2.7 SPLIT (expert draft `section227_concise_and_appendix.tex`, 2026-10-13; supersedes the v2 list in the main
  text): §2.2.7 keeps a concise Stopping-and-certification paragraph (residual pass opens the certifying regime;
  oscillatory branch with growth test, τ settling, gap clause τ/2, clean evidence; monotone branch; uncertified form)
  plus the Resolution and Determinacy paragraphs. The full rule (holds, k₀ reset, C1–C5, noise/counting sentence,
  M1–M3, uncertified-form definition) moves to new Appendix A subsection "Certification Rule"
  (`app:admm_updated_certification`), after "Stopping Test, Acceleration and Tail". Letter R3.2 changes line to add
  the new appendix subsection at the final pass.
- Overleaf 4c56901 (author pasted the split): the paste also deleted the 2.2.7 heading + `subsubsec:certification`
  (15 refs) and eq. `eq:recourse_value` with the C^Op paragraph (2 refs), plus the "re-decided" and multi-scenario
  production-exit sentences and the 𝒱-provenance/"kept frozen" clause. Expert restore `section227_restore.tex`
  (2026-10-13): heading/label/eq./C^Op verbatim from dce05d0; 𝒱 provenance + "kept frozen" → §3.6; "re-decided" → §4.7
  opening; production-exit sentence → author's §4.5 paragraph.
- Overleaf a8a61e9 (author): restore applied (no undefined refs, no duplicate labels by static check); §3.6 retitled
  "Planning and Certification Settings"; solver + ADMM settings moved to new Appendix A subsection "Solver and
  Coordination Settings". Expert corrections (`section235_move.tex`, 2026-10-13): label `app:admm_updated_settings`;
  frozen-configuration/re-evaluation sentence and α = 0.5 back in §3.6 (§2.3.6 points there for α); stale §3.4 line
  "All simulations were executed on a Mac Studio …" deleted (machine stated in §4.7).
- §2.3.5 "Objective of the Shared-ESS Agent" removed from the main text (expert draft `section235_move.tex`): two-sentence
  summary closes §2.3.4; eq. `eq:esso_objective` becomes f_E (slack penalty + throughput regularisation, AL terms now only
  in `eq:admm_esso_local`) in Appendix A.2 with the agent's exact constraint list and the regularisation paragraph;
  nomenclature row 𝓛^{E,P}, 𝓛^{E,Q} deleted; "investment-fixing slacks / apparent-power aggregation retired" sentence
  moved to the letter (Benders reply; R2.11–R2.12). §2.3.6 becomes §2.3.5.
- Letter final pass: l. 45 and R3.5 changes line cite "Section 3.4" for the cycling calibration; it is §3.5
  (Shared Energy Storage Parameters).
- Overleaf 3ad9ae8 (author): §2.3.5 heading removed (done); letter R2.11–R2.12 reply fixed; author wrote §4.1–4.5 prose
  (placeholders deleted; facts kept in git cfee0ba). Expert check (`section4_review_fixes.tex`, 2026-10-13): numbers
  in 4.1–4.5 correct; fixes F1 blue highlighting of new §4 prose, F2 `e^{}`→`e^{*}`, F3 §4.4 ref → `sec:case_settings`,
  F4 §4.4 ref → `subsec:res_baseline`, F5 locational-signal clause restored in §4.4 (accepted sentence 4), F6 the
  multi-scenario paragraphs had replaced the §4.6 placeholder → moved to §4.5, Table ref → `tab:res_multiscenario`,
  "production exit" wording, F7 §4.6 drafted (investment year, discount rate, salvage across 60 claims), F8 Fig.
  `fig:res_ageing` redrawn (`fig3_ageing_arms_v2.pdf`): frozen figure labelled rows with code names (no_ageing, C2,
  C2_calfade, C3_unit, C3_midblock, C4). Optional O1–O3 (24–33 % energy-cost reduction; 7 determinate / 6 within
  resolution among the 13 re-evaluated neighbours at 2×; aged arms not all resolved from one another).
- Appendix E (`app:results`) still holds the submitted results table (`tab:cs3_results_summary_years_days`); the
  Determinacy paragraph promises the bands/gaps/slacks of every evaluation there → rebuild Appendix E (per-evaluation
  record; destination of the §4.7 breakdown and reproducibility paragraph).

- Overleaf b47492a (author): F1–F4 applied (blue highlighting, e*, two refs). Still open: F5 locational clause, F6
  multi-scenario text still under §4.6 with Table ref `tab:res_year_ladder` and "7--9,\%", F7 §4.6 body, F8 figure v2.
- Highlights and keywords still describe the submitted method (bi-level, Benders, voltage violations, horizon
  discretization). Expert draft `highlights_keywords_v2.tex` (2026-10-13): five bullets ≤ 85 characters; keyword
  "Benders' decomposition" → "Derivative-free optimization".

- Overleaf b81ed9a (author): F6, F7, F8 (figure v2 uploaded), O1–O3 applied; highlights and keywords replaced in
  main.tex, but the `highlights` environment was changed to `itemize` (restore `\begin{highlights}`); separate
  highlights.tex still has the submitted bullets. F5 (locational clause) not applied — author's call.
- Conclusions rewritten (expert draft `conclusions_v2.tex`, 2026-10-13): framework and certification in one paragraph;
  findings (baseline x = 0, degradation decides, break-even, robustness, coordination 13.9 %); two implications
  (planning; residual test insufficient, 15–21 k€); limitations and future work. Replaces the submitted
  18.25 % / 92.16 % / voltage-violation / discretization text.

- Overleaf 6ff8f3c (author): `highlights` environment restored in main.tex; highlights.tex updated with the same five
  bullets; Conclusions v2 pasted verbatim. Static check: no undefined refs, braces balanced; no remaining "18.25",
  "92.16", "Benders", "voltage violation" or "non-wire" in the text (one "bi-level" in the literature review, correct).
  Open: §4.9 Key Insights now overlaps the conclusions (author to shorten or drop); F5 declined.

- Proposed (2026-10-14): delete §4.9 Key Insights (duplicated by Conclusions v2; no \ref to `subsec:res_insights`; the
  letter's R3.6 points to §4.3); keep its one unique point by adding "on an operating cost of 654 M€" to the
  Conclusions' implications paragraph.
- Letter final pass, R3.6 reply: "the smallest admissible unit at node 7" → the reference unit (0.25 MVA / 1 MWh, node 7);
  "under every aged calibration the optimal plan is no investment" overclaims — align with §4.8 (inferred from the
  determinate loss and the affine ladders; the search was not rerun per calibration); calibration names to Table
  `tab:res_ageing` (datasheet 8,000 cycles; retention 50 %; retention 50 %, mid-block).
- W177(b) ordered in Addendum 76 (Planner brief), incl. Part 4: push the W160 export for Appendix E.

- Letter final pass (expert draft `response_to_reviewers_v2.tex`, 2026-10-14; numbers from a compile of `6ff8f3c`:
  Fig. 1 framework, 2 search, 3 test system, 4 lattice, 5–9 results; Tables 1 costs, 6 calibrations, 7–15 results;
  Eq. (11) Q(x), (15)–(19) SOC, (20)–(26) agent/degradation, (29)–(30) salvage; App. A.5 rule, A.8 settings):
  "reference unit" throughout; R3.6 "optimal plan" → inference per §4.8; R1.5/R3.2 per Addendum 75; R3.3 §2.3.6 → 2.3.5;
  R3.5 code names (C2) → table names; R2.2 search evaluations end at the production exit; R3.7 investment discounted
  from installation year; R2.8 \cite → [7]–[9] (the letter has no bibliography); "price-responsive" → "price-taker";
  further-changes bullets for Appendix A moves, Figs. 2 and 4, highlights/keywords; all [T…] placeholders filled.
- Manuscript fixes the letter relies on (`section3_intro_conclusions_fixes.tex`): M1 §3 opening still described the
  submitted instance (2025–2037 five years, 5 × 5 scenarios, two old reference cases) → both instances as run;
  M2 Introduction l. 343: "%" had commented out the uncertainty clause R1.4 relies on and left the sentence without a
  full stop → restored, shortened; M3 Conclusions: attribution of degradation added to future work (letter says so).
- Addendum 76 amended: §3 opening and the letter join W177(b).

- W177(b) received (`4bef2b2c`, audited Overleaf `e10c104`; Part 4 export `829e92e5` under `export_W160/`). Author
  continued editing to `9a697cc` (§2.1–2.3 rewritten, Q* vs Q, nomenclature longtables, abstract, introduction).
  Expert rulings: Addendum 77. Fixes: `w177b_fixes.tex` (A1–A20 from the report; B1–B4 new in §2.1–2.3: duplicate
  eq:available_power/capacity labels, eq:cohort_rated dropped → undefined ref in A.2, wrong ref to the energy product,
  completion-cap stop missing). Letter v3 (`response_to_reviewers_v3.tex`): one salvage verdict; 3×3 wording; R1.4
  re-weighting; R3.7 no annual figures; equation numbers from a compile with all fixes applied.

- Reconciled with Overleaf `54246b7` (author had applied A1, A2 counters, A2a, A2b): remaining A2c–A20 and B1–B4
  applied by the expert to a copy of `54246b7` → `main_54246b7_fixed.tex` (+ `main_54246b7_fixes.diff`, 25 hunks);
  also the Table 13 note for A9 and the two resolved [CONFIRM] comments removed. Compile: 0 errors, 0 undefined,
  0 duplicate labels; numbering identical to the letter v3's.

- Overleaf `820ee1b` (author applied the fixes with line-break refinements; B4 declined; letter not yet replaced).
  Four slips → `main_820ee1b_fixed.tex` (4 hunks): §2.3.2 heading restored; label `eq:cohort_rated` on the S equation,
  `eq:cohort_rated_energy` on the E one; nomenclature row C^Op_{y,d,m,o} restored beside C^Op_s; §2.3.4 sentence
  completed. Letter v4 = v3 with equation numbers +1 after (13) (split equation). Compile: 0 errors, 0 undefined.

- W178 received (`5a7d663e`, audited Overleaf `9fd19e7` = `main_820ee1b_fixed.tex`; letter = v4). All predictions held:
  §2.1–2.3 storage block and cohort equations match code (0 M); A1–A20/B1–B3 applied; 151/156 letter pointers MATCH,
  0 MISMATCH. Expert follow-up (`main_9fd19e7_fixed.tex`, 14 hunks; letter v5): G1–G2 "Appendix Appendix" /
  "Section Appendix" prefixes; G3 0.50-floor row AE 0.778, EFC/day 1.08 (W155); G4 day weights 92/91/91/91 in §3;
  G5 conclusions stopping status incl. production exit (C-1); G6 holds "from the cycle after" the stopping cycle (A-1);
  G7 4.2-1; G8 4.3-1; G9 S3-12; G10 §4.8 local-solution limitation (letter LL-8); G11 two resolved CONFIRM comments
  removed; G12 nomenclature Ŝ/Ê. Letter v5: R1.4 fuel via wholesale price (§3.3) + A.1 pointer; R2.10 converter
  limits (26)–(28); R2.3 §2.2.6; LM-3 "inference"; LL-8 pointer §4.8. Numbering unchanged.
- Applied (2026-10-15): Overleaf `e926fb7` main.tex byte-identical to `main_9fd19e7_fixed.tex`; `response_to_reviewers_draft.tex` byte-identical to v5. `e926fb7` is the base of the Appendix E edit below.
- Appendix E composed (expert, 2026-10-15; Addendum 78): `appendixE_evaluation_record.tex`, generated by `gen_appE.py`
  from `export_W160/*.csv`; full manuscript `main_e926fb7_appE.tex` (+ `.diff`, 5 hunks); letter v6. Appendix E
  "Evaluation Record" (label `app:results` kept) replaces the submitted per-block table: E.1 comparisons (Table E.20,
  58 rows = T1's 55 distinct + 3 T10), E.2 evaluations (E.21, 61 rows, numbered), E.3 neighbours of x = 0 (E.22),
  E.4 static arrangements by start (E.23), E.5 gap-refused (E.24), E.6 statistics + reproducibility (from §4.7).
  §4.6: "55 comparisons reported gross and net", row 34, year pair changes sign and verdict; G-rows sentence added
  (85.1–223.6 k€), CONFIRM removed. §4.7 trimmed (branch split, medians, reproducibility → E.6); §4.8 node split and
  residual ratio → E.5. Letter v6: R2.2, R3.1 (55), reproducibility → E.6 + A.8, new bullet on Appendix E. Compile
  0 errors, 0 undefined; numbering outside Appendix E unchanged. Closes LM-9, 4.6-2, R3.7(a) subject to W179.
- Still open: W179 (Addendum 78) — Appendix E audit against `590088fe`, the nine interpretations, closure-slack
  inventory 45 → 51 of 51 (§2.3.3), the §4.8 Scope wording for the six search-only neighbours of x = 0.
- W179 received (`43c72e06`, audited Overleaf `48099ab` = `main_e926fb7_appE.tex` + letter v6; zero solves). Held: 0 table
  MISMATCH (1,623 entries), 0 letter pointer MISMATCH, §4.8 Scope qualifier. Failed: three of (i)–(ix) corrected
  (v, vii, viii); 18 of 20 W178 rows closed (C-1, R14-2 narrowed L); closure slack 50 of 51 — E.21 row 16
  (`pb_y2025_n5`, the superseded W118 v2 certificate) has a positive closure sum +1.76e-6 MWh and is "not decided"
  under v6. Rulings: Addendum 79 (expert, 2026-10-15; new account).
- Round 6 delivered (expert, 2026-10-15; `STEP6_ROUND6_CORRECTIONS.md`): `main_48099ab_round6.tex` (+ `.diff`, 16
  hunks), letter v7 (+ `letter_v6_v7.diff`, 5 hunks), `gen_appE.py` / `verify_appE.py` patched (33 PASS),
  `appendixE_evaluation_record.tex` regenerated. Row 16 kept, relabelled "superseded", Rule "---"; §2.3.3 closure
  sentence at the bound within the solver's bound relaxation, row 16's figure quoted, x = 0 "row not built"; §4.7
  "every certificate used in a reported comparison" + superseded sentence; §2.2.7 Determinacy "single-scenario" +
  Table 13, "every incumbent"; §4.7 closing and §4.8 "wherever the split was recorded"; §4.8 Scope "determinately for
  the eight … beyond the search resolution for the other six"; A.3 l. 1899 prefix; §2.2.6 Ŝ/Ê wording (+ label
  `subsubsec:capacity`); conclusions C-1; Appendix E: E.4 three phases (count in network-hours of 864), E.6 three
  continued (12, 14, 3), replays through 132/112/87, intro three-decimal residual, E.3 resolution + six search-only
  neighbours (3.47–7.17×), E.5 rows 40/41 unrecorded; four CONFIRM comments removed. Letter v7: R1.4 fuel via the
  wholesale price, not "the introduction states"; R2.2, R2.5, Appendix E bullet scoped. Compile 0 errors, 0
  undefined; every label number unchanged, one label added.
- Overleaf `e5322f8` (author): `main.tex` = `main_48099ab_round6.tex`, letter = v7 (md5 by the expert; sha256 and cmp by
  the Planner). Repo: round-6 files committed in `c831ea46` (message stale: says Addendum 78) and `465da61a`.
- W180 received (`a1ab9e99`; zero solves). Held: Appendix E regenerated bitwise, 0 MISMATCH in 1,623, 2 cells changed,
  5/5 prose; row 16 the only superseded certificate; 0 pointer MISMATCH, 0 CONFIRM, 0 prefixes. Failed: R22-b still L
  (R2.2 lacks "single-scenario"); closure count is 50 of 50, not 52 (T11 pair double-counted by the expert). Also:
  "closure row is not built" → built then deactivated; Table 13 "cycle 72" (unit stopped at 69); E.4 phase B omits
  fixed DN decisions and voltage/thermal rows off; E.21 note on rows 40/41; the `% [AUTHOR]` ageing-citation note
  was never in `main.tex` and §3.5 carries no datasheet citations. Rulings: Addendum 80.
- Round 7 delivered (expert, 2026-10-16; `STEP6_ROUND7_CORRECTIONS.md`): `main_e5322f8_round7.tex` (+ `.diff`, 6
  hunks), letter v8 (+ `letter_v7_v8.diff`, 2 hunks: R2.2 "single-scenario"), `gen_appE.py` patched (E.4 second
  phase; E.21 note), `appendixE_evaluation_record.tex` regenerated, `verify_appE.py` unchanged (33 PASS). §2.3.3
  "deactivated and its slacks are fixed at zero"; Table 13 "(cycle 72 for the plan without storage, 69 for the
  unit)"; `% [AUTHOR]` citation reminder restored under Table 6. Compile 0 errors, 0 undefined; label numbers
  identical to `e5322f8`. Accepted L: 0.004τ, search-only neighbours' bands, W153 cause rule for rows 39/41.
- W181 received (`6931c781`, Overleaf `cd57efc`): every Addendum 80 prediction held; rounds 6–7 closed.
- Ageing citations (author, Overleaf `fa7dcb6`, bib fix `8f051a2`; Addendum 81): the 10,000-cycle statement has no
  datasheet (case-file input) → §3.5 restated as an assumption "at the favourable end of current LFP cell datasheets";
  calendar 0.985/yr cited to Naumann et al. 2018 [43]; 8,000 cycles to 70 % cited to the EVE MB31 specification
  PBRI-MB31-D06-01 [44]; Table 6 caption reworded; `% [AUTHOR]` comment removed. Expert compile 0 errors, numbering
  unchanged, [1]–[42] unchanged.
- Marked copy (expert, 2026-10-16): `main_latexdiff.tex` = latexdiff 1.4.0 of `manuscript_submitted/main.tex` vs
  `8f051a2` with the 381 `\textcolor{blue}{}` wrappers stripped, `--math-markup=whole`, tables/figures/algorithms as
  whole blocks; post-processed (emergencystretch; 18 refs to removed labels print the label; display-math markup
  colour-only). First delivery failed on Overleaf ("Missing $ inserted", l. 746: `\DIFadd` around a wrapped equation);
  fixed as above. Compiles 0 errors / 0 undefined with a ulem stand-in (98 pages); final PDF on Overleaf.
  `main_latexdiff_v2..v5.tex`, `rev_nocolor_scratch.tex` and `latexdiff-so` in `manuscript_review/` are scratch (do not
  stage; delete).
- Letter v9 (+ `letter_v8_v9.diff`, 4 hunks): opening, R1.2(iv), R3.4, R3.5 with [43]/[44].
- 2026-10-25: author's simulation brief E1–E6 after an external critical audit (`manuscript_review/manuscript_critical_review_2026-10-25.md`);
  Overleaf at `edd7a63` (1,021 changed lines, four new equations, one table added, one table and two figures removed,
  headline 12.2 % vs the price-taker benchmark). Expert plan v1 → reviewed (`STEP7_PLAN_REVIEW_2026-10-25.md`) →
  author's D1–D5 and six clarifications → **plan v2** (`STEP7_SIMULATION_PLAN_2026-10-25.md`): matched E1 arms (both
  on the final code, variant B, common seeds, N^max 30 each, 2× primary, 14-point unit-neighbourhood screen at 1×),
  staged E2 (penalty pilot 8, start stage conditional), E3 with a declared mixed-comparison reporting rule, E4/E5
  deferred with narrowed claims, E6 zero-solve; hypotheses H1–H3 recorded; counts reconciled (E1 ≤ 72 + 14; serial
  ≈ 84–89 h). Addendum 82 rewritten (v2): W182 = re-audit of `edd7a63` (A1–A8) + inventory Z1–Z7 + readings C1–C4;
  no run authorized.
- W182 received (`dfd09ca5`; Overleaf `edd7a63`, 67 commits after `8f051a2`; code `a17f6ba7`/production `575bfec4`;
  seven zero-solve parts). Held: A1 0 M; A2 0 MISMATCH (701 MATCH, 31 DERIVED); A3 65/173 letter pointers MISMATCH;
  33 PASS; A7 16 sentences; Z1 by code reading; Z2; Z6. Failed: A6 bitwise (Appendix E hand-edited, six hunks, 0
  entries changed); Z7 favourably (reverse-flow hours in committed JSON). E1–E3 need harness/production changes
  (no-ageing launcher, keyed ρ₀ option, NRF option on both copies); no perturbed coordinated start exists.
  Rulings: Addendum 83 — generator port; "contracted interface settlement excluded"; eol_retention_r 0.50; E2 stage
  2 deferred; NRF on the consensus exchange; concurrency gate inside E1; gate G0; E6 pickle read; round 8 scope;
  counts E1 ≤ 75 + 14. W183 ordered (code changes, gates, spec v1 freeze). §3.3 generator-pricing sentence found
  missing at `edd7a63` (letter R1.4 target) — restored in round 8.
- Author committed the two E6 JSONL files and forwarded W183 (Addendum 83).
- Round 8 delivered (expert, 2026-10-26; `STEP6_ROUND8_CORRECTIONS.md`; Addendum 84): `main_edd7a63_round8.tex`
  (+ `.diff`, 31 hunks; 21 main-text edits + regenerated Appendix E), letter v10 (+ `letter_v9_v10.diff`; 65 pointers
  renumbered, [42]/[43], 12.2 %/13.9 %, nine A7 sentences, R3.6 narrowed, R3.4 v9 error fixed), `highlights_v3.tex`,
  `cover_letter_v2.tex` (revision account; E1–E3 paragraph bracketed), `gen_appE.py` (six edd7a63 hunks ported +
  round-8 edits), `appendixE_evaluation_record.tex`. Compile 0 errors, 0 undefined; label numbers identical to
  `edd7a63`; 76 pages. W183h check ordered.
- Round 8 on Overleaf `a87e195` (author, 2026-10-26); repo `22e19b56` (the four tracked files) + `bd3a9f35` (the nine
  new files, after the expert found them unstaged). Expert verified the four Overleaf files byte-identical to the
  deliverables (main 39f6fd86…, letter v10 2aeb33b2…, highlights v3 4cedd45c…, cover letter v2 88eee8ce…).
- W183 received (`ca56713c`; production `6f43479e`, harness `c34ebdc5`, E1 launcher `320d503c`; 153 solves, all in the
  two-cycle gate). Held: G0 identical at x = 0; NRF-off NL identity 696/696 (on: 864 rows per copy); 7 seed keys (+1
  G0 alias). Failed as frozen: the two-cycle gate, one item — 22 W64 record keys absent in the W48 reference and null
  in the arm, 29 shared keys bitwise equal, per-cycle record byte-identical to the W83 arm. Unpredicted: variant B
  refuses the first poll for every storage seed at k = 0 (6–7 distinct points; completion 31/34/61 > cap 30). E6
  exported (declared day 2025 Spring, 40,181.22 € = 15.8 % of 253,539.62 €; panel (b) in €). W183h not run.
  Rulings: Addendum 85 — gate read as field-list drift (zero-solve comparator vs the W83 arm; standing rule
  `GATE_PASS_RULED`; W48 reference kept; sequential-builder scope accepted); E1 poll rule **B+** (when the completion
  would exceed the cap, advance the Halton index up to 10 times until n + 1 distinct feasible points; else as frozen;
  both arms, every poll, declared before any E1 solve; launcher-owned loop, F2 module unedited); A7 barrier rule
  confirmed; spec v1 declarations (batch seeds unit/n7 0.5/2 × A/N; cross-evaluations and E3/E2 under the ext-v6
  certified protocol; screen 14 + x = 0 at 1× = 15; NRF tolerance 1e-3 MW; G0 pointer `ca195bd3`); E6 € measure and
  the three guarded reads accepted; W185 no longer waits for the §4.4 text; counts E3 1, E2 8, E1 ≤ 75, screen 15,
  ≈ 81 h serial. W183g ordered (comparator run, launcher v2 + dry-run gate, fixed-plan launcher, spec v1 freeze,
  W183h); W184 (E3) and W185 (E2) authorized conditionally on W183g's gates, no further addendum. Plan amended
  (`STEP7_SIMULATION_PLAN_2026-10-25.md` §12, v2.2).
- Still open: W183g (+ W183h); W184–W187 runs; round 9 (results text incl. the one-sentence B+ statement, the E6
  figure, letter R2.5/R2.6/R3.6, cover-letter paragraph, marked copy); author's call on the prediction-record bullet
  (letter l. 286).
