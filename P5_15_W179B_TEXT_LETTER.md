# P5.15 W179b — main-text and letter changes of Addendum 78, letter v6 pointers, G1–G12 / letter v5 (zero-solve)

**Authority:** Addendum 78, W179 items (b) and (c).

**HEAD read: Overleaf `48099ab`**, the only commit after `e926fb7`, with the clone clean.
- main.tex sha256 `28008ab0…`, 2,717 lines; byte-identical to `manuscript_review/main_e926fb7_appE.tex`.
- The letter `6ecd0ee7…`, 292 lines; byte-identical to `response_to_reviewers_v6.tex`.
- At `e926fb7`, main.tex = `main_9fd19e7_fixed.tex` and the letter = v5. At `9fd19e7`, both equal the W178 hashes, and
  the letter = v4.
- The clone's `9fd19e7 → e926fb7 → 48099ab` diffs carry the same +/− lines as the committed
  `main_9fd19e7_followup.diff` (14 hunks), `main_e926fb7_appE.diff` (5 hunks) and `letter_v5_v6.diff`.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit;
F = I + Q.

**Runs:** zero solves. Everything here is a static reading; nothing was compiled.
- `SolveProfileGuard(permitted=())` was armed for the whole run (own guard plus the imported W178c / W177b-3′ /
  W177b-3 / W167 / Phase B guards). It was verified at exactly 0, with all counts 0.
- `pickle.load` / `loads` were blocked (0 calls), and no production module was imported.
- 0 check failures; exit 0.
- W100 typing test: PASS.

## Blocked on Planner

1. **Prediction (c) did not hold.** Two of the 20 addressed rows are **still L, narrowed**; neither is M or H. Each
   needs a ruling: close as worded, or one more wording edit (author). Details are in the G table below.
   - **C-1 (G5).** The conclusions now cover the 3 × 3 pair. Still uncovered:
     - the search evaluations (production exit, single-scenario) behind §4.1's "none reduced … beyond the search
       resolution";
     - "with its remaining movement measured": only the x = 0 evaluation was continued, and its 6.2 k€ is a post-hoc
       damped-cosine fit (§4.5, Table 13). The unit's movement was not measured.
   - **R14-2, fuel (letter v5).** The reply still opens "the introduction now states …". The Introduction (l. 381) names
     market prices but not fuel. "fuel" occurs 0 times in main.tex. The new "(Section~3.3)" lands on the mechanism
     (l. 1183: "Conventional generation in the TN is priced at the wholesale energy price of the scenario").
2. **One remaining appendix prefix that W178 missed (L, author).**
   - A.3 l. 1899 reads `Subsection~\ref{app:admm_updated_stopping}`.
   - By the same elsarticle reading as G1–G2 (`\thesection` = "Appendix A", and `\thesubsection` is not redefined), it
     prints "Subsection Appendix A.4".
   - It has been present since at least `9fd19e7` (l. 1908). W178 listed l. 755 and l. 1603 only.
   - **Fix:** drop "Subsection~".
   - This is the only remaining `<word>~\ref{app:…}` prefix. There are 0 "Appendix~\ref" and 0
     "Section~\ref{app:…" left.
3. **New L (G12 wording).**
   - The new nomenclature entry reserves "fixed parameters of the operational subproblem" for Ŝ^Inv / Ê^Inv.
   - §2.2.6 l. 697 still calls the un-hatted S^Inv, E^Inv "fixed parameters of the operational subproblem".
   - §2.3.2 uses the hats (l. 771, 776, 783).

## Prediction vs outcome (expert, Addendum 78)

| prediction | outcome |
|---|---|
| (b) 0 pointer MISMATCH | **HELD.** Letter v6 has 165 numbered pointers: **160 MATCH, 0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED** (the same five as W178c) |
| (c) all W178 rows addressed close | **NOT HELD.** Of the 20 rows addressed, 18 are closed (one with an accepted L residue, one with a new L) and 2 are still L, narrowed (C-1, R14-2). Every fix is applied verbatim |

## (b) Main text, §4.6–§4.8

**Numbers.** Every numeric token of the six changed paragraphs was declared and checked: l. 1637, 1681, 1690, 1707,
1716 and 1720. A token that is not declared fails the run, and the token list must equal the declarations.
- **70 tokens:** 60 match the frozen JSON (`590088fe`) or a named record (W153, W163, the F2 spec); 5 carried and
  matched by W163 (N33, N35–N37); 5 language. **0 MISMATCH.**
- **New in Addendum 78:**
  - **two, 7, 2030, 2035, 0.5/1, 0.5/2.** Distinct node-7 plans and years of the four G claims.
  - **85.1 / 223.6.** The min and max over the eight G `d_gross` / `d_net` values: 85.1 is `G:n7_2h_e1_y2035` net; 223.6
    is `G:n7_4h_e2_y2035` gross.
  - **rows 17–20.** Parsed from Table E.20.
  - **55.** The distinct frozen comparisons with a net figure: the 60 claims less the four G `:net` duplicates and
    `CHECK:headline`.
  - **row 34.** `L:y2030__n5_p0.25_e0.5__n7_p1_e3`.
  - **42.5 / −2.3.** `year_ladder` `D_gross` / `D_net`.
- **Negative control** (in-memory text altered to 85.2 and to row 35): both were refused.

**Claims (W174b scale).**

| id | sentence | rating | evidence |
|---|---|---|---|
| C46-1 | G-rows sentence: two larger node-7 plans, 2030/2035, 85.1–223.6 k€ more, gross or net, all determinate (rows 17–20) | consistent | Rows 17–20 are the four G claims. Δ, threshold, × and the two verdicts all equal the frozen claims; 8 of 8 verdicts det. The G CONFIRM comment is gone |
| C46-2 | "55 comparisons of Table E.20 reported both gross and net … no sign and one verdict" | consistent | 55 distinct comparisons in the frozen JSON; Table E.20 prints net on 55 rows (gross only: 22, 52, 58); 0 sign changes; 1 determinacy change (row 34). W153 agrees (`n_sign_differs` 0; `claims_verdict_differs` = row 34) |
| C46-3 | row 34 within the uncertified bar gross (0.95×), determinate net (1.22×), in the returned plan's favour | consistent | Row 34 prints 26.4 / 27.9 / 0.95 w.b. and 34.1 / 1.22 det., equal to the frozen values. Δ = F(neighbour) − F(returned) > 0 |
| C46-4 | year pair "changes both the sign and the verdict" (+42.5 → −2.3) | consistent | No row of Table E.20 is the year pair. Gross: +42,458.65, 3.46×, det. Net: −2,286.25, 0.19×, within resolution |
| C47-1 | "Every comparison and every evaluation behind this section is listed in Appendix E" | **L** | E.20 holds 58 comparisons and E.21 the 61 evaluations. The 41 search evaluations named in the same section (production exit, compared at σ_Q) are not in Appendix E |
| C47-2 | "one evaluation at a time on a single thread of an Apple M2 Max (E.6)" | consistent | E.6 carries the machine, the thread caps and serial MA97 verbatim |
| C48-1 | fingerprint "−8.8 to −9.4 k€ split in the same proportions among the three interface nodes (Table E.24)" | **L** | Table E.24 prints the split for 5 of the 7 m = 2 evaluations (39, 50, 51, 45, 42: 55.2–55.3 / 13.7 / 31.0–31.1 %). For the F2 references 40 and 41 it prints "---" (no `share_by_node` in the frozen `dead_zone`). The gap range itself holds for all 7 |
| C48-2 | "21 cycles in 10 evaluations … (Table E.21)" | consistent | Column N.c. is positive on 10 rows (3, 4, 6, 8, 9, 15, 37, 38, 39, 42) and sums to 21 |

**What moved and what stayed (all verified).**
- **Moved to E.6** (each absent from §4.7 at HEAD, present in E.6; the e926fb7 text was present):
  - the branch split 24/8 (reworded);
  - the median certification cycle 174;
  - k* − k₀: median 65, range 21–89;
  - range/τ median 0.86;
  - "the four further evaluations" (now with IDs 32, 33, 60, 61);
  - the machine and memory, the thread caps, and HSL without OpenMP;
  - the whole Reproducibility paragraph (every sentence is in E.6, none in §4.7).
- **Moved to E.5:**
  - the node split, reworded as 55/14/31 % for nodes 5/7/9;
  - the residual ratio, restated from "near 0.69" to "0.68–0.70".
- **Kept in §4.7:**
  - the counts by cause (32/10; 5/2/3);
  - the τ statement (10 of 46);
  - 0.9τ and 15–21 k€;
  - the Fig. 9 paragraph;
  - the search paragraph (41; 14/20/7 of 20/20/60);
  - the wall time (220,876 s, 8,570 cycles, 25.6 s).
- **Kept in §4.8:** the −8.8 to −9.4 k€ fingerprint and the 5 MWh evaluation (+1.1 k€).
- The numbers in the E.5/E.6 destinations are W179a's scope and are not re-checked here.

## (b) Letter v6: R2.2, R3.1 and the two "Further changes" bullets

**Numbers.** 41 tokens on the 10 lines changed since v4 (l. 58, 97, 125, 126, 136, 172, 193, 221, 284, 285):
- 1 inserted and checked: "55" (l. 221);
- 36 carried, each matched to its W178c row (30 match, 2 derived and holding, 4 language);
- 4 language (2 reviewer labels, 2 "single").
- **0 MISMATCH.**

**Claims.**

| id | text | rating | evidence |
|---|---|---|---|
| R22-a | "The number of candidates evaluated is reported in Section 4.7" | consistent | §4.7: "41 new recourse evaluations in all (14, 20 and 7 …)" |
| R22-b | "and the cycles of every evaluation in Appendix E" | **L** | E.21 gives k*/end for all 61 evaluations it lists, and k₀ for all but 1, 2, 40, 41 (the note gives k₀ = 123 for 1). Not in Appendix E: the 41 search evaluations, which are the "candidates evaluated" of the same sentence, and the 3 × 3 pair (Table 13, cycle 72) |
| R22-c | rchanges adds "Appendix E" | consistent | — |
| R31-a | "Among the 55 comparisons of the full claims table that are reported both gross and net, it changes no sign and one verdict" | consistent | As C46-2 |
| FC-1 | reproducibility note "(Appendices E.6 and A.8)" | consistent | E.6: one machine, single-threaded, bitwise replays. A.8: IPOPT 3.14.18, Pyomo 6.9.5, Python 3.11.11 (no HSL version is printed) |
| FC-2a | Appendix E bullet: the submitted Appendix E "held per-block operating results of the submitted instance" | consistent | `manuscript_submitted/main.tex` holds `\label{app:results}` with `tab:cs3_results_summary_years_days`; that table is absent at HEAD |
| FC-2b | "every pairwise comparison, with its threshold or bar and its verdict, gross and, where evaluated, net" | consistent | Table E.20 (gross only on rows 22, 52, 58: "---: not evaluated net") |
| FC-2c | "every recourse evaluation, with its stopping status, band, gap and slack" | **L** | Same scope point as R22-b / C47-1: E.21 lists the evaluations behind the comparisons, and the E introduction says so ("every recourse evaluation those comparisons use"); the bullet drops the qualifier |
| FC-2d | the neighbours, the static arrangements, the gap-refused evaluations, the statistics | consistent | Tables E.22, E.23, E.24 and E.6, in that order |

## (b) Letter v6 pointer table at `48099ab`

- **Method:** W178c's static numbering (`numbering_adjusted`, imported unedited). One addition: a level-qualified title
  target for "Section 3.3", because "Transmission Network" titles both §3.3 and Appendix C. In the first trial run that
  pointer was AMBIGUOUS for exactly this reason.
- **Labels:** 119 outside Appendix E are compared with `e926fb7`, and **0 changed number**. The 11 new labels are
  E.1–E.6 and Tables **E.20–E.24**. One label is removed (`tab:cs3_results_summary_years_days`). 0 undefined refs, 0
  duplicate labels.
- **elsarticle.dtx l. 1459–1469:** `\appendix` sets `\thetable` to `\@Alph\c@section.\arabic{table}` and never resets
  the table counter, so Appendix E's tables continue the global count from 20.

| kind | MATCH | SUBMITTED | MISMATCH | AMBIGUOUS |
|---|---|---|---|---|
| section | 91 | 1 | 0 | 0 |
| appendix | 26 | 0 | 0 | 0 |
| table | 16 | 0 | 0 | 0 |
| figure | 10 | 3 | 0 | 0 |
| equation | 10 | 0 | 0 | 0 |
| algorithm / reference / "new" / "former" | 2 / 4 / 1 / 0 | 0 / 0 / 0 / 1 | 0 | 0 |
| **total (165)** | **160** | **5** | **0** | **0** |

- **Pointers on lines changed since v4 (34), all MATCH:**
  - l. 58: §4.8 ×2, §4.5.
  - l. 97: §3.3, Table 15, §4.6, A.1.
  - l. 125–126: R2.2, including Appendix E.
  - l. 136: §2.2.6, §2.3.2.
  - l. 172: §4.8.
  - l. 193: Eqs. (26)–(28) = `eq:storage_active_power_limit`, `eq:converter_capability`,
    `eq:network_complementarity`.
  - l. 221: §4.6, §3.5, Eq. (22).
  - l. 284: Appendices E.6 and A.8.
  - l. 285: Section 4, Appendix E.
- **Unchanged lines:** every verdict equals W178c's at `9fd19e7` (0 changes).
- **Content needles:** carried on 60 pointers; **0 absent**. The two needles W178c found absent now land: l. 136
  §2.2.6 "are fixed parameters of the operational subproblem", and l. 160 Appendix E "settling slack" / "bitwise".
- **SUBMITTED (5, deliberate):** l. 100 "Figure 2" (R1.5 heading); l. 102 "Figure~2 of the submitted version"; l. 141
  "former 2.2.7"; l. 232 "Section~4.4" and "Figure~15" (the reviewer's).
- The per-pointer table is in `w179b_checks.md` / `.json` (`b_pointers.pointers`).

## (c) G1–G12 and letter v5 at `e926fb7`, still present at `48099ab`

**All 14 hunks of `main_9fd19e7_followup.diff` are applied verbatim**, at `e926fb7` and at HEAD:
- every "+" line is present verbatim;
- every "−" line was present at `9fd19e7` and is absent at HEAD.

Inserted numbers:

| G | inserted value | source |
|---|---|---|
| G3 | 0.778 / 1.08 | W155 `reports.e_soh050` AE 0.77811 / EFC 1.07630 |
| G4 | 92/91/91/91 | `SRP1.json` and the 3 × 3 instance file, both Spring 92, Summer 91, Autumn 91, Winter 91, matching "in both instances" |
| G7 | 2025 | — |
| G10 | "three" | 3 recorded starts per NRF arrangement in `export_W160/T6.csv` |

**Letter v5:** all six edits present verbatim at HEAD (R1.4 fuel, R1.4 A.1, R2.10, R2.3, LM-3, LL-8 pointer). The old
"assumption (Section 4.8)" and "only through the P–Q–S capability inequality" are gone.

| W178 row (prior) | by | status at 48099ab | evidence |
|---|---|---|---|
| W178a R2, l. 755 prefixes (L) | G1 | closed | "detailed in \ref{…implementation}", "(\ref{…algorithm})" |
| W178b A11, l. 1603 prefix (L) | G2 | closed | "…successful; \ref{app:admm_updated_settings})" |
| 0.50-floor AE/EFC (Decision 2) | G3 | closed | T11 row "… never & 0.778 & 1.08"; "---: not reported" legend removed |
| R3.7(b) day weights (M) | G4 | closed | §3 l. 1109; letter R3.7 "(Section~3)" now holds |
| C-1 conclusions (L) | G5 | **still, narrowed (L)** | 3 × 3 covered; search evaluations not; "movement measured" is a fit on x = 0 only |
| A-1 hold origin (L) | G6 | closed | A.3 l. 1899 and A.5 l. 1923 "from the cycle after that run's stopping cycle"; hooks: "for every cycle > N" |
| 4.2-1 (L) | G7 | closed | Table 10 (node 7, 2025) value − I = 46.3, 92.8, 133.8, 176.6, 228.4, all > 0 |
| 4.3-1 (L) | G8 | closed | 0.50 floor at the baseline only (one 0.50 row in Table 11) |
| S3-12 (L) | G9 | closed | "or with one another" covers E.20 (b)–(d) plan-to-plan comparisons |
| LL-8 (L; manuscript + R2.5) | G10 + v5 | closed | §4.8 states the local-solution limit and points to Table 12's spread (25.8, 0.0); letter "(Section~4.8)" |
| CONFIRM comments (note) | G11 + A78 | closed | 0 `[CONFIRM — W177]` at HEAD (2 removed at e926fb7, the G-rows one at 48099ab) |
| Ŝ/Ê nomenclature (L) | G12 | closed; **new L** | §2.2.6 l. 697 wording (Blocked 3) |
| R14-2 fuel (L) | v5 | **still, narrowed (L)** | Blocked 1 |
| R14-6 rate in weights (L) | v5 | closed | "(Appendix~A.1)": the A.1 block weight carries (1+r)^−(y−y₀); "the paper says so (Section 4.6)" now attaches to the exact/first-order statement §4.6 makes (l. 1659) |
| R210-2 "only through" (L) | v5 | closed | (26), (27) and (28) all carry S^Av. The A.2 agent bound is also a converter limit but is not cited; that does not make the statement false |
| R2.3 pointer (L) | v5 | closed | §2.2.6 and §2.3.2 both land |
| LM-3 "assumption" (M) | v5 | closed | §4.8: "… but this was not run" — an inference |
| LM-9 Appendix E content (M) | A78 App. E | closed | E holds "settling slack", "bitwise" and the salvage claims table |
| 4.6-2 claims table (M) | A78 App. E | closed | C46-2 / C46-4 |
| R3.7(a) "no annual figures" (M) | A78 App. E | closed, L residue accepted (ruling 2) | per-block table gone; §4.3 per-year EFC/day remains |

**Remaining CONFIRM comments at HEAD:** 4 × `[CONFIRM -- W179]`, at l. 2379, 2590, 2625 and 2694. All are in Appendix E,
all are comments (none typeset), and all are W179a's scope.
- Two still use the composition's working numbers: l. 2379 "Table E.1 … row 1", and l. 2694 "the 49 certified rows of
  Table E.2".
- The printed tables are E.20 / E.21. Not typeset; noted for the author's final pass.

## Changed

- `5b7ffcbc`: `p515_s53_w179b_checks.py`, sha256 `69e24631…`.
  - It started from the interrupted draft, which I reviewed.
  - Fixes, made before the commit: the §3.3 title ambiguity; G9 "one" declared as language; G10 "three" taken from
    T6; G4 checked on both instance files; the CONFIRM row read on raw lines; two carried letter tokens declared; a
    vacuous W153 fallback removed.
  - Trial runs went to the session scratchpad only.
- `data/SRP1/Results/P515S53/w179b_text_letter/` (write-once): `w179b_checks.json` (`88221b8d…`), `w179b_checks.md`
  (`b804c256…`), `launch.log` (`34739217…`), `manifest_sha256.json` (`3ad8fc23…`), `w179b_bool_typing_test.json` /
  `.log` (PASS), `manifest_post_run_sha256.json`.
- No manuscript, `manuscript_review/`, frozen, spec or earlier-script file changed.

## Not confirmed

- **Typeset output.** Nothing was compiled. The "Subsection Appendix A.4" rendering and the table numbering E.20–E.24
  are static readings of `elsarticle.dtx` in the clone. Overleaf's installed `elsarticle.cls` was not inspected.
- **Pointer numbers.** These are checked by label: each letter number is mapped to the label it is meant to cite, and
  that label's number at HEAD is compared. A pointer to the wrong section with a correct number is caught only where a
  content needle exists (60 of 165).
- **E.5 / E.6 destination numbers** (32/33/60/61, 0.68–0.70, 55/14/31 %, the reproducibility figures). These were
  checked only as moved text, not against the record; that is W179a.
- **R2.5 l. 159–160, "bitwise replays of every gated evaluation"**, which this task did not rate. E.21's Replay column
  is "---" for several evaluations (e.g. 1, 2, 54–61). Whether "gated evaluation" means every one of them was not
  resolved; I looked at E.6 and E.21 only.
- **4.2-2 and the other unchanged L rows** (LL-1, LL-4–LL-7; W178a R1). No G fix addresses them, and they were not
  re-scored.
