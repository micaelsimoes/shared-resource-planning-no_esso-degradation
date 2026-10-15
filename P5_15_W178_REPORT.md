# P5.15 W178 — the manuscript at Overleaf `9fd19e7` (Addendum 77; zero-solve)

**Authority:** Addendum 77 (W178 order and its follow-up at `820ee1b`).

**HEAD audited: Overleaf `9fd19e7`** (2026-10-14 20:18 UTC).
- It descends from `820ee1b`.
- main.tex sha256 `d18384d669b7a4fb…`, 2,511 lines. It is byte-identical to `manuscript_review/main_820ee1b_fixed.tex`, so
  it carries the expert's four follow-up hunks.
- The letter (`18992d8f…`) is byte-identical to `manuscript_review/response_to_reviewers_v4.tex`.
- No later Overleaf commit existed when the tasks ran.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit.

**Runs:** zero solves in every part (guards armed and verified at 0, pickle blocked). Nothing else ran.
- W178a–c were each restarted after session-limit interruptions.
- The first interruption left nothing; the second left W178b's draft script. The final Worker reviewed it, fixed it
  (it now checks for insertions inside multi-line fixes) and committed it before its real run.

**Part reports:**

| item | report | commits |
|---|---|---|
| (a) §2.1–§2.3 vs code | `P5_15_W178A_SECTION2.md` | `4504ebf4` |
| (b) fixes applied, (c) 0.50-floor row | `P5_15_W178B_FIXES.md` | `e894e109`, `2542da63` |
| (d) letter v4 | `P5_15_W178C_LETTER.md` | `45e04e6c`, `6680e6f4` |

## Decisions needed

All are wording, and none needs a run. The predictions held, so nothing here is a stop.

1. **Two appendix pointers print "Appendix" twice (author; L).**
   - elsarticle's `\appendix` sets `\thesection` to `\appendixname~\Alph` (`elsarticle.dtx` l. 1465; the Planner read
     it). So `\ref{app:…}` already prints "Appendix A.x".
   - l. 755: `Appendix~\ref{app:admm_updated_implementation}` and `Appendix~\ref{app:admm_updated_algorithm}` print
     "Appendix Appendix A.2 / A.6".
   - l. 1603: `Section~\ref{app:admm_updated_settings}` prints "Section Appendix A.8".
   - These are static readings; nothing was compiled.
   - **Recommendation:** drop the "Appendix~" and "Section~" prefixes.

2. **0.50-floor row (l. 1540) — fill the two cells (author).** AE **0.778**, EFC/day **1.08**.
   - Source: `data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json` (sha256 `3c71d71d…`,
     `reports.e_soh050.AE` 0.77811 and `.EFC` 1.07630). Recomputing from the evaluation record gives bitwise-equal
     values.
   - The definition is the same `M46A.pv` over SoH and EFC/day as T8. W155's 0.70 cell reproduces T8's baseline row
     exactly.
   - The row's value 258.5 = 317,957.01 − 59,500.72 € holds. The cells still print "---", and letter R3.6 says
     Table 11 gives AE and EFC/day per calibration.

3. **Letter R3.7 / R3.8 (l. 274) — two M rows from L4 (expert).**
   - **(a) "The revised paper reports no annual figures".** Appendix E still holds the submitted per-year table
     (l. ~2391). Once the expert composes Appendix E (Addendum 77 ruling 3), only the non-monetary per-year EFC
     figures remain (Table 11; §4.3 l. 1554), and those are L.
   - **(b) "the representative-day weights … are stated (Section 3)".** The block lengths are stated, but the day
     counts (92 / 91 / 91 / 91, `SRP1.json`) appear nowhere in main.tex; the Planner checked.
   - **Recommendation:** add the four day counts to §3 and compose Appendix E. Then the reply holds as written.

4. **Ratings for the expert to confirm (W178b).** These are reclassifications, not new findings.
   - **4.6-2, H → M:** the count is now right ("one verdict among the 60"), but the sentence cites "the full claims
     table (Appendix E)", which does not exist yet. This is the same gap as letter LM-9 (R2.5 l. 159, R3.1 l. 220).
   - **C-1, M → L:** "certified or uncertified" does not cover the 3 × 3 production-exit evaluations or the search
     evaluations.
   - **A-1, M → L:** for the three reference continuations the text says "from that run's stopping cycle". The hooks
     hold for cycles > N, one cycle later.

5. **Small items (author, final pass).**
   - **Remaining comments:** three `% [CONFIRM — W177]` comments remain (l. 1359–1361, the salvage block A8 said to
     delete; l. 1641; l. 1712). They are not typeset.
   - **W178a R1 (L):** eq. (7) counts cohort life in years; the code counts it in blocks. The two give identical sets
     in both instances.
   - **W178c L rows:**
     - R1.4 says §4.6 states that the discount rate enters the coordination weights; A.1 says so, §4.6 does not;
     - R2.10 says the converter rating enters "only through" the capability inequality; it also enters eq. (26), the
       normalisation in (28) and the A.2 agent bounds;
     - R1.4's "fuel" is not named in the Introduction;
     - R2.3's pointer to §2.3.2 lands on text that B1 rewrote. The substance is still at l. 770 and §2.2.6.
   - **Letter LM-3:** "which we state as an assumption (Section 4.8)". §4.8 states an inference, and the word
     "assumption" does not occur there.
   - **Unchanged L rows from earlier rounds:** 4.2-1, 4.2-2, 4.3-1, S3-12, LL-8.
   - **Nomenclature:** Ŝ^Inv and Ê^Inv (l. 770) have no entry. Two nomenclature edits outside A/B (𝒴^Act at l. 221,
     φ reworded at l. 257) were not scored.

## Blocked on the author

- **Push:** everything is committed locally. The terminal cannot authenticate to GitHub, so push from PyCharm.

## Changed

| commit | content |
|---|---|
| `3312c468` | TASKS.md (order) |
| `4504ebf4` | W178a: `p515_s53_w178a_section2_checks.py`; `data/SRP1/Results/P515S53/w178a_section2/`; `P5_15_W178A_SECTION2.md` |
| `e894e109`, `2542da63` | W178b: `p515_s53_w178b_number_check.py`; `…/w178b_number_check/overleaf_9fd19e7/`; `P5_15_W178B_FIXES.md` |
| `45e04e6c`, `6680e6f4` | W178c: `p515_s53_w178c_letter_checks.py`; `…/w178c_letter/`; `P5_15_W178C_LETTER.md` |

- No production, spec, case, frozen or manuscript file changed.
- The production code equals all 46 campaign heads, checked by W178a with a positive control.
- **Erratum, W178b's committed JSON:** the A-3 evidence string says "'the interface price' no longer occurs in
  main.tex". It does occur, at l. 1725 and l. 1758, both naming λ. The status (resolved) is unaffected. The output
  directory is write-once, so the correction is recorded in `P5_15_W178B_FIXES.md` and here.

## Found — predictions against outcomes (expert, Addendum 77)

| prediction | outcome |
|---|---|
| (a) the network storage block and the cohort equations hold against code; at most one M row, in the closure-slack or power-limit wording | **held** — of 42 changed statements, 40 match and 2 are L rows (R1 eq. (7); R2 l. 755); 0 M. Both wordings match |
| (b) all applied | **held** — A1–A20 and B1–B3 present: 20 verbatim, 3 with changes that do not alter meaning (A8: CONFIRM comment not deleted; A11: "Section~" prefix; B1: split equation per the expert's hunk). B4 declined by the author |
| (d) 0 pointer MISMATCH | **held** — 156 pointers: 151 MATCH, 5 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS. The 16 table mismatches of `e10c104` are fixed by `\addtocounter{table}{-1}`, and the static numbering agrees with the expert's compile on all 7 items of ruling 5 |

**(a) §2.1–§2.3 against the code** (W178a).
- Every equation of the network storage block (20)–(28) matches `model_construction_helpers.py`:
  - net power at :999;
  - SoC recursion at :977–980;
  - limits 0.10 / 0.90 at :890–903;
  - initial state 0.5 at :973;
  - closure at :987–990;
  - closure slack bound 0.05 E^Av + 10⁻⁵ at :1166, penalty at :2341;
  - P^Ch + P^Dch ≤ S^Av at :887;
  - circle at :871;
  - complementarity 10⁻⁴ at :951.
- The cohort and available-capacity equations (14)–(19) match `shared_energy_storage_data.py:569–570, 619, 628,
  274–280`.
- The A.2 ranges resolve to (14)–(19) and (29)–(34).
- Labels: 121, 0 duplicates, 0 undefined references.
- Earlier rows now resolved: W174b T2-11, T3-7, NEW-8 and NEW-10; W166 S9; W177b A-3; B1–B3.

**(b) Re-scored W177(b) rows** (W178b, 31 rows).
- **Resolved:** both earlier H rows (4.5-1; 4.6-2's count, now M), the "two verdicts" MISMATCH, the salvage
  equation, the J row, 4.6-1, C-2, C-3, the §4 preamble flag, LM-4–LM-8, LL-9, A-3, nomenclature, the stale pointers,
  the cache clause, table numbering and "Subsection 3.1".
- **Reclassified M → L:** C-1 and A-1.
- **Still open:** LM-9 (M), plus L 4.2-1, 4.2-2, 4.3-1, S3-12 and LL-8.
- **Number check over the 144 addressed lines:** 207 tokens; 168 match; 7 derived, all holding; 4 approximate
  (carried roundings); **0 MISMATCH**.
- **Salvage:** the printed equation equals the code at recovery 1 / floor 0. It reproduces the recorded credits of
  `yl_y2030` (11,759.257933 €) and `yl_y2035` (56,504.166416 €) exactly.
- **J 4 MWh row:** printed at its uncertified bar, 15.9 / 11.12×, with the certified threshold 9.7 / 18.14× beside
  it. Both match the frozen fields.

**(c)** See decision 2.

**(d) Letter v4** (W178c).
- **Changed replies:** 10 hunks in 9 replies.
  - Numbers: 67 tokens; 55 match, 5 derived holding, 0 MISMATCH.
  - Claims: 58; 43 consistent; H 0, M 5, L 10. On the edited lines alone: M 2 (decision 3) and L 1.
- **Earlier letter rows:** resolved LH-1, LM-4–LM-8, LL-2 and LL-9. LM-10 was replaced by the L4 rows. LM-3, LM-9
  and LL-1, LL-4–LL-8 are unchanged.
- **L1–L4:** applied. They live in Addendum 77's rulings, not in `w177b_fixes.tex`.

## Not confirmed

- **Compilation:** nothing was compiled; there is no TeX here. The equation and table numbers and the
  "Appendix Appendix" rendering are static readings. The numbering agrees with the expert's compile where both exist.
- **Closure slacks:** the l. 967 claim that both closure slacks were zero rests on W169's inventory, which scored 45
  of 51 certified points.
- **Discount re-weighting:** the size of A12's first-order error at the unit plan needs re-solves.
- **Scope:** unchanged letter replies were checked for pointers only. Rows the fixes do not touch carry their earlier
  ratings.
