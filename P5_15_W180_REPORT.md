# P5.15 W180 — round 6 at Overleaf `e5322f8` (Addendum 79; zero-solve)

**Authority:** Addendum 79 (W180 order).

**HEAD audited: Overleaf `e5322f8`** (`e5322f85dced…`, 2026-10-15 16:56 UTC), one commit after `48099ab`.
- The clone was fast-forwarded from `48099ab` to `e5322f8`; it has no local changes.
- **main.tex** (sha256 `b3282295…`, 2,710 lines) is **byte-identical to `manuscript_review/main_48099ab_round6.tex`**
  (`cmp`, Planner; re-verified in all four harnesses).
- **The letter** (`34a5754b…`, 292 lines) is byte-identical to `response_to_reviewers_v7.tex`.
- The clone's diff `48099ab → e5322f8` equals the expert's `main_48099ab_round6.diff` and `letter_v6_v7.diff` line
  for line.

**Delivery commits.** The round-6 generator, verifier and regenerated appendix arrived in `c831ea46`, together with
Addendum 79; that commit's message says "Addendum 78". The rest arrived in `465da61a`.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit.

**Runs:** zero solves in every part.
- `SolveProfileGuard(permitted=())` was armed before any project import and verified at exactly 0; pickle was
  blocked, with 0 calls; no production module was imported.
- The four parts ran in parallel. They are static readings with separate scripts and output directories.
- Two Workers' scratch trials collided in one shared scratchpad directory. No committed output is affected; each
  committed run went into a fresh `data/` directory.

**Part reports:**

| item | report | commits |
|---|---|---|
| (a) Appendix E regenerated; entry check; prose | `P5_15_W180A_APPENDIX_E.md` | `7b6ba5fe`, `995f0628` |
| (b) row 16 class | `P5_15_W180B_ROW16_CLASS.md` | `6520188b` |
| (c) main-text edits, letter v7, pointers, rows | `P5_15_W180C_TEXT_LETTER.md` | `e924814f`, `b4920f5c` |
| (d) §2.3.3 closure sentence | `P5_15_W180D_CLOSURE_SENTENCE.md` | `d1e6ee19` |

**Stop.** Two recorded predictions failed — (c) "all eight re-scored rows close" and (d) "52 of 52" — and Addendum
79 names no fallback. Both are wording or count predictions. No number, verdict or run depends on them. The steps
after W180 (latexdiff, cover letter, submission) wait for the rulings below.

## Decisions needed

1. **Letter R2.2 (row R22-b; expert).** This row is still L.
   - **The finding.** R2.2 splits the evaluations into two classes: the search's own evaluations (production exit),
     and those "behind a reported comparison" (certification rule or cap).
   - The 3 × 3 pair is behind a reported comparison (Table 13, −81.3 k€, determinate) but fits neither class. Its
     cycle, 72, is printed in Table 13 and not in Appendix E.
   - Of the four sentences round 6 scoped, R2.2 is the only one without "single-scenario". The other three are
     §2.2.7, §4.7 and the Further-changes bullet.
   - **Options:**
     - (a) Close it as worded and accept the residue as L, as ruling 4 did for Determinacy. Costs nothing.
     - (b) The author inserts one word: "every single-scenario evaluation behind a reported comparison".
   - **Recommendation: (b).** It makes R2.2 read like the other three. Only the letter changes; no pointer moves.

2. **§2.3.3, "Without storage the closure row is not built" (expert; L).**
   - The row is built for every unit (`network.py:511`, unconditionally in `_build_model`).
   - For a unit with zero capacity, `configure_shared_ess_operational_state` then does three things:
     - it fixes both slacks at 0 (`model_construction_helpers.py:1143`);
     - it bounds them to [0, 0] (`:1166`);
     - it deactivates the row (`:1176`).

     The Planner read 1140–1144 and 1174–1177.
   - The substance holds: at all five x = 0 points the closure is exactly 0.
   - **Recommendation:** "Without storage the closure row is deactivated and its slacks are fixed at zero." This rests
     on the code only. "…is not part of the solved problem" would rest on Pyomo's NL writer, which was not tested.

3. **The `% [AUTHOR]` citation note that "stays" is not in the source (author).**
   - STEP6 "Not changed" and Addendum 79 ruling 6 say the `% [AUTHOR]` note under the ageing table stays until the
     citations are in.
   - It is in none of these: `main.tex` at `e5322f8` or `48099ab`, `main_48099ab_round6.tex`,
     `main_e926fb7_appE.tex`, `main_9fd19e7_fixed.tex`, or the letter.
   - The clone's history of main.tex shows one `[AUTHOR` note: the "four insights" note, added before `51a6fb5` and
     removed at `51a6fb5`.
   - Across the Overleaf project, the only `[AUTHOR]` is in `section2_expert_draft.tex` l. 162 (the salvage sign).
     That file is not `\input` by main.tex.
   - **Needed:** the author confirms whether the ageing-table datasheet citations are in. If they are not, they must
     be added before submission. A reminder comment is optional.

4. **Wording residues for the expert (all L; no number changes).** Recommendation for each in brackets.
   - **Table 13, l. 1619:** "Q(0) − Q(unit) at operational stopping (cycle 72)". The unit evaluation stopped at
     **69**; only x = 0 stopped at 72 (W180d, from `campaign_results.json` `n7_4h_e1`). The frozen JSON's label also
     says "cycle 72"; the value 236.7 k€ is not in question. [Print "(cycles 72 and 69)", or name both evaluations.
     The frozen artefact is not touched.]
   - **E.4, phase B:** "a re-solve … with its no-reverse-flow rows switched off". In the W116 benchmark, phase B also
     fixes every distribution-network decision at the arm solution, and switches off the voltage and thermal rows
     too (W180a's reading of the benchmark code). [Add "with its decisions held at the first phase and its network
     limits switched off", or accept as L.]
   - **E.6, "0.004τ" (l. 2688):** three decimals. The intro names only E.24's residual ratio as a three-decimal
     exception. [Accept as L: the intro sentence is about the tables, and 0.00τ would lose the figure.]
   - **§2.2.7 vs E.3 / §4.8 Scope:** the six search-only neighbours are single-scenario evaluations used in a
     reported comparison ("beyond the search resolution"), but their bands are not tabulated. [Accept as L, with the
     residues ruling 4 accepted.]
   - **E.21 note, rows 40 and 41:** the note calls these F2 references continuations of earlier runs, but W163 N49's
     reference continuations are x = 0, unit and C\* only. Both rows are "record" rows uncertified under v2 and v6
     (W180b). [Reword to "re-decided from the records of earlier runs".]
   - **Rows 39 and 41, cause "gap clause":** this rests only on the W153 first-match cause formula. The W142
     floor-replay records, at the cap, the reasons `insufficient_turning_points` / `monotone_window_not_reached` for
     row 41 (W180b). [Accept as L; the cause rule is W153's, as stated.]

5. **After the rulings (Planner recommendation).** Items 1–4 are one-word to one-sentence edits.
   - If the expert issues them as a round 7, the check needs only:
     - identity of the round-7 files;
     - the clone diff equal to the expert's diff;
     - the touched sentences re-rated;
     - Appendix E regenerated bitwise, if E.4 or the E.21 note changes.
   - A full re-audit is not needed.
   - Then the order's latexdiff against `manuscript_submitted/`, the cover letter and submission.

## Blocked on the author

- **Decision 3:** confirm the ageing-table citations.
- **Push:** see `Changed`. If the terminal push failed, push from PyCharm.

## Changed

| commit | content |
|---|---|
| `09eb590c` | Planner: TASKS.md W180 order, REVISION_CONTEXT head (Addendum 79 received) |
| `7b6ba5fe`, `995f0628` | W180a: `p515_s53_w180a_appendixE_check.py` (committed before the run); `data/SRP1/Results/P515S53/w180a_appendixE/` (generator output `appE_generated_w180a.tex`, verifier output, check JSON/MD, manifests, launch log, bool-typing test); `P5_15_W180A_APPENDIX_E.md` |
| `6520188b` | W180b: `p515_s53_w180b_row16_class.py`; `…/w180b_row16_class/` (JSON, manifests, launch log); `P5_15_W180B_ROW16_CLASS.md` |
| `e924814f`, `b4920f5c` | W180c: `p515_s53_w180c_text_letter.py`; `…/w180c_text_letter/`; `P5_15_W180C_TEXT_LETTER.md` |
| `d1e6ee19` | W180d: `p515_s53_w180d_closure_sentence.py`; `…/w180d_closure_sentence/`; `P5_15_W180D_CLOSURE_SENTENCE.md` |
| this commit | `P5_15_W180_REPORT.md`, TASKS.md, REVISION_CONTEXT.md |

- No production, spec, case, frozen, `manuscript_review/` or manuscript file changed.
- No W179 script or output was re-run or overwritten.
- `gen_appE.py` wrote only into `w180a_appendixE/`; no `appE.tex` exists anywhere.

## Found — predictions against outcomes (expert, Addendum 79)

| prediction | outcome |
|---|---|
| (a) the regenerated block equals main.tex bitwise | **held** — the generator output occurs once, at l. 2375–2694 (line-aligned). It also equals the committed `appendixE_evaluation_record.tex`. The Planner re-checked both with `cmp` |
| (a) `verify_appE.py` 33 PASS | **held** — 33 PASS, 0 FAIL |
| (a) 0 MISMATCH in 1,623 entries | **held** — 1,443 MATCH on frozen fields, 0 MISMATCH; 180 non-frozen entries all agree with their named sources |
| (a) exactly two entries changed (row 16 Status, Rule) | **held** — 2 of 1,637 cells: `cert., osc.^s` → `superseded^s`, `record` → `---`. Captions changed separately: the E.21 legend and the E.23 caption |
| (a) all five prose sentences MATCH | **held** — intro, E.4, E.5 and both E.6 sentences MATCH; E.3 "own resolution" DERIVED (8 at σ_Q, 6 at the bar sum), E.3 six-neighbour sentence MATCH. Notes in decision 4 |
| (b) 0 further rows like row 16 | **held** — rows flagged = {16}. All 61 E.21 rows are classified, with a 9-case mutation self-test |
| (c) 0 pointer MISMATCH | **held** — 165 = 160 MATCH, 5 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS. 131 labels unchanged; `subsubsec:capacity` is the only label added (→ 2.3.2) |
| (c) all eight re-scored rows close | **failed** — 7 of 8: C-1, R14-2, C47-1, C48-1, FC-2c, the G12 follow-on and the A.3 prefix close; R22-b is still L, narrowed (decision 1) |
| (c) 0 CONFIRM, 0 prefixes | **held** — 0 `[CONFIRM`; 0 label-word prefixes (the 8 `in~`/`of~` matches are prepositions). The expected `% [AUTHOR]` note is absent (decision 3) |
| (d) the sentence holds at 52 of 52 points | **failed on the count** — it holds at **50 of 50**: 48 current E.21 certificates + 2 multi-scenario points. W179c's 51 were 49 E.21 certified rows + the 2 T11 points. Less row 16 that gives 50, so "50 + 2 T11" counts the T11 pair twice (the Planner checked W179c §(d) Coverage). No point fails |

**(c) detail** (W180c).
- §A.1–A.9 are verbatim. Applying STEP6's replacement pairs to `48099ab` outside Appendix E reproduces the HEAD
  exactly, so no other main-text line changed. §A.10 lies wholly inside the Appendix E block.
- Letter §B is verbatim. Only lines 1, 97, 125, 159, 160 and 285 changed; the one pointer change is "Appendix E" →
  "Appendix E.6".
- R14-2 closes. "Fuel" appears 0 times in main.tex. §3.3 l. 1184 says "Conventional generation in the TN is priced at
  the wholesale energy price of the scenario"; the letter's new sentence restates it, plus an inference consistent
  with it.
- Bookkeeping: STEP6 lists nine main-text edits (A.1–A.9; A.10 is Appendix E), where Addendum 79 says ten, and A.9
  precedes A.8 in file order. No effect.

**(d) detail** (W180d).
- **The 10⁻⁶ MWh figure is correct.**
  - IPOPT 3.14.18 runs with `bound_relax_factor` at its default. No case file, production source or option list in
    the 49 logs sets it.
  - The relaxation is 10⁻⁸ p.u. on lb = 0, and 10⁻⁸ × baseMVA 100 × 1 h = 10⁻⁶ MWh.
  - Observed: every slack sits 0.995–1.000 × 10⁻⁶ MWh below zero. The distance to the relaxed bound is at most
    3.76 × 10⁻¹⁰ MWh over the 44 E.21 certificates with storage, and 5.01 × 10⁻⁹ at `t11_3x3_unit`.
  - W179c's "5.1 × 10⁻¹⁰" was looser than its data. That figure is not in the manuscript.
- **Row 16** (evaluation 16 = `pb_y2025_n5`, k\* 167): +1.7637 × 10⁻⁶ MWh, S/E^Av = 5.039 × 10⁻⁶. "1.8 × 10⁻⁶" and
  "5 × 10⁻⁶" MATCH.
- **The multi-scenario pair:**
  - x = 0 at cycle 72: exactly 0 in all 80 blocks.
  - The unit at cycle 69: all 40 active blocks at the bound.
- **Twins.** Rows 56 and 61 are bitwise twins of evaluation 2. Counted as distinct certificates the set is 48 of 48,
  consistent with E.6's 46.

**(b) detail** (W180b).
- **Earlier-rule certificates with a v6 decision on record.** Every "record" row with an earlier-rule certificate has
  v6 certifying it on its record at the same k\*:
  - v1 (rows 1, 2; W168 also gives 181 for row 1);
  - v4 (rows 12, 13);
  - v5 (row 14);
  - v2 (rows 17–21, 24, 25).
- **Row 3** was certified by v6 from its records (W142, k\* 150).
- **The 45 "run" rows** are v6 runs; their G15 gate is the v6 decision.
- **Rows 40 and 41** (the F2 references) are uncertified under both v2 and v6 at the same end cycle. They print no
  certificate, so prediction (b) is unaffected; they do not fit its parenthetical gloss literally.
- **Row 16:** "superseded^s", Rule "---", k\* 167; v6 on its record gives "not decided within 1..167", with one veto at
  cycle 167.

## Not confirmed

- **Compilation:** nothing was compiled; there is no TeX here. The numbering is static (the clone's `elsarticle.dtx`
  plus W178c's method). The expert's compile results were matched, not reproduced.
- **Pointer semantics** are content-checked for 60 of the 165 pointers.
- **The multi-scenario stopping cycles** (72 and 69) were read from `campaign_results.json` by W180d. The section 4
  tables were not parsed cell by cell.
- **Not tested or not reproduced:**
  - Pyomo's NL writer behaviour for deactivated rows and fixed variables (decision 2's alternative wording).
  - IPOPT's `relax_bounds` code was read from the fetched 3.14.18 source; the binary was not run.
  - The completeness of the "three continued evaluations" rests on W163 N25. No new search was made, and the
  movements were not recomputed from per-cycle records.
- **The `[AUTHOR]` search** covered main.tex, the letter, the clone's history of main.tex, every file in the clone and
  the three `manuscript_review/main_*.tex` versions. It did not cover Overleaf project files outside the clone.
- **W180b did not search:** the W118 r1 campaigns, the v3 runs, the 3 × 3 evaluations or pickles. Its full scope is
  in its JSON under `search_scope`.
