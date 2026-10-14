# P5.15 W177(b) — the revised manuscript at Overleaf `e10c104` (Addendum 76 and its amendment; zero-solve)

**Authority:** Addendum 76, its same-day Amendment (`f4f58d10`), and `manuscript_review/W177B_CHANGE_LOG.md`.

**HEAD audited: Overleaf `e10c104`** (2026-10-14 13:59 UTC).
- It is a descendant of `bfc5009`, as the order requires.
- main.tex sha256 `1b532035abc65a7f…`, 2,345 lines.
- The letter `response_to_reviewers_draft.tex` (`bc270fe2…`) is byte-identical to
  `manuscript_review/response_to_reviewers_v2.tex`.
- Parts 0–3 were first run at `51a6fb5`, then repeated at `e10c104`. §4.1–4.7 are byte-identical between the two
  (shifted by 32 lines); 213 lines of main.tex changed elsewhere.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit; value
= Q(0) − Q(x).

**Runs:** zero solves in every part (guards armed and verified at 0, pickle blocked). Nothing else ran.

**Interruption:** the three `e10c104` Workers were cut off by a session limit after their runs.
- The Planner committed Parts 0–1 and Part 2 from their complete outputs.
- A follow-up Worker verified the Part 3 run intact (script, input and output hashes) and wrote its summary. Nothing
  was re-run.

**Part reports:**

| part | at `51a6fb5` | at `e10c104` |
|---|---|---|
| 0–1 numbers | `P5_15_W177B_PART01.md` (`404e9181`, `128b99e6`) | `P5_15_W177B_PART01_E10C104.md` (`f78836a9`, `87fecf8a`) |
| 2 statements | `P5_15_W177B_PART2.md` (`99fdfabb`) | `P5_15_W177B_PART2_E10C104.md` (`559ef967`) |
| 3 structure + letter pointers | `P5_15_W177B_PART3.md` (`be583a4d`) | `P5_15_W177B_PART3_E10C104.md` (`87db6a83`) |
| 4 Appendix E inputs | `manuscript_review/export_W160/` (`829e92e5`) | — |

**Stop.** Two Addendum 76 predictions failed as worded, and the addendum names no fallback. Both are text
predictions, and no run depends on them.

## Decisions needed

1. **§4.6 l. 1520 and letter R3.1 l. 221: "two verdicts" among the 60 — the record has one (expert; H).**
   - W153 (`215fd21d`): 60 claims, 0 sign changes, **1** verdict change, `L:y2030__n5_p0.25_e0.5__n7_p1_e3`
     (0.95× → 1.22×).
   - The 2035 − 2030 comparison is T4, not one of the 60. On it the salvage credit changes the verdict (3.46× → 0.19×)
     and also the sign (+42,458.65 € → −2,286.25 €; the net value is within resolution).
   - **Recommendation**, in both places: "Across the 60 comparisons of Table T1, the salvage credit changes no sign and
     one verdict (…the F2 neighbour…). Separately, the investment-year comparison is determinate in gross terms and
     within resolution net of salvage."

2. **§4.5 l. 1439: "The planning search again returned x = 0" at the multi-scenario instance — no search was run there
   (expert; H).**
   - Only x = 0, the reference unit, and the x = 0 continuation were evaluated on the 3 × 3 instance. The search
     covered all campaign specs, results, evaluation records, scripts and `git log --all`.
   - **Recommendation:** "At the multi-scenario instance the reference unit again loses to x = 0 (−81.3 k€, 4.5×); the
     planning search was not repeated there."

3. **The letter describes text that is not in `e10c104` (expert/author; M).** Five replies:
   - **R1.4 (i):** the introduction does not state which uncertainties are represented and which are not. The
     amendment's M2 was not applied: the "To address this gap" paragraph now ends "…of load, RES generation and market
     prices.", and "cost of capital" appears nowhere in main.tex (the letter's l. 98 points to Section 1 for it).
   - **R1.2:** the abstract does not name the chemistry.
   - **Opening:** no statement of 5 × 5 as future work.
   - **R2.5:** no statement that no convergence theorem exists.
   - **R3.7 / R3.8:** no "representative-day-weighted annual" wording.

   Two more replies (l. 160, l. 220) cite Appendix E for content it does not hold yet; the Part 4 composition is
   pending. **Recommendation:** apply M2, and either add the four statements or trim the replies.

4. **R1.4: "the sensitivity is exact at a fixed plan" (expert; M, H if read as a model statement).**
   - The discount rate is inside the recourse objective (`_get_admm_block_weight`,
     `shared_resources_planning.py:3870–3873`), and the ageing chain couples the years at the unit plan. Per-year
     costs are rate-independent only at x = 0.
   - The verdicts hold (margins 2.57–8.78×), but the 0/5/8 % rows re-weight the 2 % evaluations rather than re-solve
     them. §4.6 should say so (M row, unchanged since `51a6fb5`).
   - **Recommendation:** "the re-weighting is exact at x = 0 and first-order at the unit plan".

5. **Letter table numbers: all 16 pointer MISMATCHes have one cause (author).**
   - Overleaf `1c644d0` replaced the nomenclature's three `tabular` blocks with three caption-less `longtable`s
     (l. 200, 232, 294). `longtable` steps the table counter even without a caption, so every body table moves up by
     3 (static numbering; no TeX here).
   - The letter's numbers equal `6ff8f3c`'s. No section, figure, equation, appendix or algorithm number changed.
   - **Recommendation:** add `\addtocounter{table}{-1}` after each nomenclature `longtable`. All 16 then match again;
     no letter edit is needed.
   - The 5 "SUBMITTED" pointers (l. 100, 102, 141, 232) refer to the submitted version and are correct as such.

6. **Unchanged since `51a6fb5` (decisions carried from the draft at that HEAD).**
   - **Salvage equation `eq:salvage_credit` (l. 1183–1196) (expert):** still not the coded form, which uses the energy
     cost only and the health factor (SoH − SoH^Min)/(1 − SoH^Min). The printed form is 4.22× and 1.86× too high on
     the recorded cells. W177(a)'s corrected equation is ready.
   - **Appendix A.5, two M rows (expert):**
     - A-1, the hold origin for continued evaluations: N + 1 for the three references; k₀ + 1 with cap N_old + 100
       for the 30 gated re-runs, two of them above 300.
     - A-3, C4 priced "at the interface price": the code uses the expected wholesale price
       (`p515_s53_w118_resettle_hooks.py:337–369`); §4.8 calls λ "the interface price".
   - **Conclusions M rows (expert):**
     - "every other evaluated plan … loses determinately": the verdicts hold, but the set includes two three-node
       plans, and some cells hold only at search resolution;
     - "each evaluation is certified";
     - "cannot be assessed apart from coordination".
   - **§4 preamble l. 1148 equivalent** ("every reported number is a certified recourse evaluation…") (expert):
     discount re-weightings, 3 × 3 values, D and uncertified cells are not.
   - **Author items:**
     - the J 4 MWh row (Addendum 64 ruling 4);
     - the 0.50-floor row (W155 values available);
     - nomenclature: π_t and s unused, C^Op_s used but undefined, 74 used-undefined keys;
     - stale pointers: l. 1555 and l. 1796 point to content now only in A.5; l. 1760 and l. 1439 point to §3.6 for
       values in A.8;
     - §3.6 cache-seeding clause: "with and without the budget" is s47's, but "their combinations" is s53's only;
     - "Subsection~3.1" at l. 985 is hard-coded rather than a `\ref`.

## Blocked on the author

- **Push:** everything is committed locally. The terminal cannot authenticate to GitHub, so push from PyCharm.

## Changed

| commit | content |
|---|---|
| `829e92e5` | Part 4: `manuscript_review/export_W160/` — T1, T2, T5, T6 (all starts), T9, T10 (.tex/.csv), README.md (frozen-JSON hash 590088fe), README_v2_note.md, paragraphs_v2.md (statistics (ii), scorecard (v)), MANIFEST.md (each sha256 equal to its source) |
| `404e9181`, `128b99e6`, `99fdfabb`, `be583a4d` | Parts 0–3 at `51a6fb5` |
| `f78836a9`, `87fecf8a` | Parts 0–1 at `e10c104` (`p515_s53_w177b1r_number_check.py`; outputs; summary) |
| `559ef967` | Part 2 at `e10c104` (`p515_s53_w177b2r_checks.py`; outputs, with the `production_read/` plots hashed in the manifest but not committed; summary) |
| `87db6a83` | Part 3 at `e10c104` (`p515_s53_w177b3r_checks.py`; outputs; summary) |
| `c9e6f4c6`, `dbe92713` | TASKS.md |

No production, spec, case, frozen or manuscript file changed.

## Found — predictions against outcomes (expert, Addendum 76), at `e10c104`

| prediction | outcome |
|---|---|
| Part 1: zero MISMATCH in numbers carried from the tables | **held for table cells** — 0 among 388 tokens of table bodies, captions and notes; none against a frozen field. One prose MISMATCH in scope: "two verdicts" (decision 1) |
| Part 1: the derived numbers hold | **held**, 16/16 — 24–33 %, 7–9 % (§4.5 and Conclusions), 90.9 M€, 654 M€, O3 9.3 / 42.1 / 28.5 k€, 108, "six", 54.5–64.4 |
| Part 2: (c) is two verdict changes | **failed on the 60 scope** — one; "two" holds only with T4's pair beside the 60, and on that pair the sign also changes |
| Part 2: nothing contradicted | **failed** — manuscript H 2 (decisions 1–2); letter H 1 (R3.1) plus 5 M rows describing absent text (decision 3) and R1.4 (decision 4) |
| Part 3: no undefined reference | **held** — 0 undefined (93 ref keys, 185 uses, 111 labels; 43 cite keys, 48 bib keys); 0 duplicate labels |
| Part 3: at least one nomenclature symbol defined but no longer used | **held on the count** (π_t, s); **not by the mechanism** — neither was orphaned by the moves, and the `e10c104` edits orphaned nothing |

**Amendment item (1), the §3 opening (M1):** in, and correct.
- The single-scenario instance carries the first draw of production's per-block scenario sampling, for market and for
  operation. In 2025 it is bit-for-bit scenario 1 of the 3 × 3 (1,784 arrays, 0 mismatches). The 2030 and 2035 draws
  are SRP1's own, because the 3 × 3 instance has no such years.
- The prefix draw [1,2,3] × [1,2,3] is confirmed (W89 / W52).
- The new `\ref{sec:case_settings}` at l. 990 points to the no-reverse-flow rule, which §3.6 states.

**Amendment item (2), the letter:**
- Numbers: every number traced, except the "two verdicts" count. 62 quotation tokens verified against the reviewers'
  document.
- Claims: decisions 1, 3 and 4.
- Numbered pointers: 151 in all — 130 MATCH, 16 MISMATCH (decision 5), 5 SUBMITTED.

**Part 0 (W177(a) carry-over):**
- Still present: the salvage equation, the J 4 MWh row, and the 0.50-floor row.
- Fixed: the literal "Section 4" references and the C6 wording; slot texts l. 63 and l. 381.
- No longer applicable: the F2 plan sizes, slot texts l. 58 and l. 65, and l. 484 (§4.9 deleted).

**Part 3 checklist (change log):**
- `fig:investment_lattice` matches the lattice exactly: 12 admissible, 23 over budget, 108 outside limits, with the
  budget and limit lines exact. The fourteen neighbours are the 7 node subsets × {2025, 2030}.
- The §3.6 run settings match the search specs, 9 of 9 mechanical items.
- Of the 21 references to `subsubsec:certification`, 19 find their content in §2.2.7; the other two are the stale
  pointers above.
- Appendix A.5 has all 20 rule-v6 clause anchors. All 22 prior Part 3 rows remain open.

## Not confirmed

- **Numbering:** all numbering is static; nothing was compiled. `longtable`'s counter behaviour is assumed from the
  package, not observed.
- **Scopes not re-checked at `e10c104`:** §1–§3 outside the §3 opening, and Appendix A text outside Part 3. Many lines
  changed there (nomenclature, §2.1, introduction).
- **"No 3 × 3 search":** scoped to this working tree's `data/`, the root scripts, `git log --all` subjects and the
  brief.
- **The lattice figure:** its PDF was produced with Matplotlib 3.10.9 and its script was not executed. The v2 ageing
  figure was not reproduced bytewise.
- **The M ratings of Part 3:** carried from W177b-3, not re-assessed.
