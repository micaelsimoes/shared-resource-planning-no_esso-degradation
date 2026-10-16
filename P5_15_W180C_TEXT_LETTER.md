# P5.15 W180c — round-6 main-text edits A.1–A.9 and letter v7 at Overleaf `e5322f8`; letter pointers; W179b rows re-scored (zero-solve)

**Authority:** Addendum 79, W180 item (c).

**HEAD read: Overleaf `e5322f8`** (`e5322f85dced8034312c9afb97c19001bce6bfa0`), the only commit after `48099ab`. The
clone is clean.
- main.tex: sha256 `b3282295…`, 2,710 lines. It is byte-identical to `manuscript_review/main_48099ab_round6.tex`.
- The letter: sha256 `34a5754b…`, 292 lines. It is byte-identical to `response_to_reviewers_v7.tex`.
- At `48099ab` the blobs are W179b's: main.tex = `main_e926fb7_appE.tex` (`28008ab0…`), and the letter = v6 (`6ecd0ee7…`).
- The clone's own `git diff 48099ab HEAD` matches the expert's diffs line for line, in the same order:
  - main.tex: 28 − / 21 + lines, equal to `main_48099ab_round6.diff` (16 hunks);
  - letter: 6 − / 6 +, equal to `letter_v6_v7.diff` (5 hunks).
  - A difflib diff aligns some lines differently but agrees on the net change.

**Objective convention:** Q = `gross_operational_cost`, settlement excluded; net = Q − terminal salvage credit;
F = I + Q. No objective value is computed here. The values quoted are the manuscript's.

**Runs:** zero solves. Everything here is a static reading of the text; nothing was compiled.
- `SolveProfileGuard(permitted=())` was armed for the whole run, with the guards of the imported W179b / W178c /
  W177b-3′ / W177b-3 / W167 / Phase B modules. It was verified at exactly 0, with all counts 0.
- `pickle.load` / `loads` were blocked (0 calls), and no production module was imported.
- 0 check failures; exit 0.
- Six negative controls, all refused. W100 bool-typing test: PASS.

## Blocked on Planner

1. **Prediction (c) "all eight re-scored rows close" did not hold.** Seven rows close. **R22-b** (letter R2.2, l. 125)
   is **still L, narrowed**. A ruling is needed; options:
   - **What R2.2 says now:** "the cycles of every evaluation behind a reported comparison in Appendix E".
   - **Narrowed:** R2.2's own sentence sorts evaluations into two classes. Search evaluations end at the production exit.
     Evaluations "behind the reported comparisons" end by the certification rule or its cap. Under that split the 41
     search evaluations drop out, and every certification-rule evaluation has k*/end in Table E.21.
   - **Residue (L):** the 3 × 3 pair is behind a reported comparison (Table 13: −81.3 k€, determinate) but fits neither
     class.
     - Its stopping cycle, 72, is in Table 13, which the Appendix E introduction points to.
     - E.6 gives the x = 0 evaluation's 72 replayed cycles, but not the unit's.
   - **Secondary, arguable:** the six neighbours evaluated by the search only enter the comparison of §4.1 / §4.8 Scope
     / E.3 at the search resolution. No cycles are printed for them anywhere.
   - Of the four scoped sentences, R2.2 alone has no "single-scenario" in its cycles clause. The other three are
     §2.2.7, §4.7 and the "Further changes" Appendix E bullet. No number or verdict depends on this.
   - **Options:**
     - (a) Close as worded, accepting the L residue. This is how ruling 4 treated the Determinacy sentence's residues.
     - (b) One word for the author: "every **single-scenario** evaluation behind a reported comparison". This aligns
       R2.2 with the bullet and §2.2.7 and removes the 3 × 3 residue. The six search-only neighbours stay under the
       secondary reading.
   - Either way, the letter's text is the author's to change. I changed nothing.
2. **Expected one `% [AUTHOR]` comment; found 0.** This is a fact for the author, not a manuscript defect.
   - main.tex at `e5322f8` has no `[AUTHOR` anywhere, comment or text. The letter has none either.
   - The same holds at `48099ab`, which has 0.
   - Under the ageing table (l. 1286–1307) there is no comment line at all.
   - Addendum 79 ruling 6 and STEP6 "Not changed" both say "the `% [AUTHOR]` note under the ageing table stays until the
     citations are in".
   - Clone history (`git log -S'[AUTHOR' -- main.tex`): main.tex had 0 at `c71ca2e`, which agrees with Addendum 72.
     It had 1 at `51a6fb5^`: the "four insights" note (`% [AUTHOR — edit] …`), not a citation note. `51a6fb5`
     (2026-10-14) removed it, and main.tex has had 0 since.
   - So the reminder about the datasheet citations is not in the source. If it is still wanted, the author has to
     re-add it.

## Changed

- `<commit>` (this commit): `p515_s53_w180c_text_letter.py` (sha256 `52dfd0f2…`), this report, and the output directory
  `data/SRP1/Results/P515S53/w180c_text_letter/` (write-once; the post-run manifest lists every file):

  | file | sha256 |
  |---|---|
  | `w180c_text_letter.json` | `88e1d1d9…` |
  | `w180c_text_letter.md` | `250dcdca…` |
  | `launch.log` (stdout + stderr) | `6a81bc72…` |
  | `manifest_sha256.json` | `24b8354b…` |
  | `w180c_bool_typing_test.json` / `.log` (PASS) | `42211d74…` / `5051fd3c…` |
  | `manifest_post_run_sha256.json` | `2946308c…` |

- `p515_s53_w179b_checks.py` is imported unedited. W179b's committed JSON (`88221b8d…`) is read only.
- Trial runs went to the session scratchpad only.
- No other file changed. That includes manuscript_review/, the clone, production code, frozen files, specs, W179
  files, TASKS.md, REVISION_CONTEXT.md and the brief.

## Found — item-(c) predictions against outcomes

| prediction (expert, Addendum 79) | outcome |
|---|---|
| 0 pointer MISMATCH | **HELD.** 165 pointers: **160 MATCH, 5 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS**. The five SUBMITTED are the same ones as W179b |
| all eight re-scored rows close | **NOT HELD.** 7 closed; R22-b still (narrowed), L (Blocked 1) |
| 0 `[CONFIRM` | **HELD.** 0 in main.tex and 0 in the letter, comment lines included |
| 0 `<word>~\ref{app:` prefixes | **HELD, with a caveat on the pattern.** The general pattern also matches prepositions (below). There are 0 label-word prefixes |
| (task) one `% [AUTHOR]` comment | **0 found** (Blocked 2) |

### (1) Identity: 22 of 22 checks hold

The 22 checks are:
- HEAD, short and full hash;
- sha256 and line count of main.tex and of the letter;
- byte-identity with the round-6 files;
- the HEAD blobs equal the working copy;
- the `48099ab` blobs are W179b's;
- exactly one commit after `48099ab`, and the clone is clean;
- the frozen JSON, W179b's JSON and W179b's script are unchanged;
- hunk counts 16 and 5;
- the git and difflib +/− equalities.

### (2) Main text A.1–A.9: all verbatim, nothing else changed outside Appendix E

**Method.** Each edit is stated as exact (old → new) string pairs copied from STEP6 §A, with its markdown line wraps
joined by one space.
- A.3's old sentence is taken from the expert's diff, because STEP6 gives only "Replace the sentence by".
- Every old string occurs exactly once outside Appendix E at `48099ab`, and every new string exactly once at HEAD.
- **Reconstruction:** applying the pairs plus A.1's label line to `48099ab` outside Appendix E gives HEAD outside
  Appendix E **exactly**. So no other main-text change exists.
- The text after Appendix E is unchanged.

| edit | where | STEP6 l. (= old text at `48099ab`) | l. at HEAD | verbatim |
|---|---|---|---|---|
| A.1 | §2.2.6 "held fixed … parameters Ŝ, Ê of Subsection~\ref{subsubsec:capacity}" | 697 | 697 | yes |
| A.1b | `\label{subsubsec:capacity}` on the line after the §2.3.2 heading | 768 (heading) | 769 | yes |
| A.2 | §2.2.7 Determinacy, "every incumbent"; "single-scenario … Table~\ref{tab:res_multiscenario}" | 743 | 743 | yes |
| A.3 | §2.3.3 closure sentence | 968 | 969 | yes |
| A.4 | §4.7 "Every certificate used in a reported comparison … used in no comparison." | 1689 | 1690 | yes |
| A.5 | §4.7 last sentence | 1690 | 1691 | yes |
| A.6 | §4.8 "wherever the split was recorded" | 1716 | 1717 | yes |
| A.7 | §4.8 Scope "worse, determinately for the eight … other six." | 1724 | 1725 | yes |
| A.8 | A.3 "(\ref{app:admm_updated_stopping})" | 1899 | 1900 | yes |
| A.9 | Conclusions | 1736 | 1737 | yes |

**Hunks of `git diff -U0`.**
- There are 22 hunks. 9 are main-text hunks, one per edit; A.4 and A.5 share one hunk (l. 1689–1690).
- The other 13 lie wholly inside the Appendix E block, both at `48099ab` (l. 2374–2701) and at HEAD (l. 2375–2694).
  These ranges equal STEP6's.
- In the expert's 16 hunks with context, that is 9 + 7.
- **A.10 is wholly inside the Appendix E block.** Its content is W180a's scope and is not checked here.

### (3) Letter v7, section B: all verbatim

Checked on whitespace-collapsed text, because the letter wraps lines and R2.5 crosses a line break. The reconstruction
v6 + pairs = v7 holds exactly. Changed lines: 1, 97, 125, 159, 160 and 285, and no others.

| edit | line | verbatim | source of the strings |
|---|---|---|---|
| R1.4, fuel out of the "introduction now states" clause | 97 | yes; "fuel" no longer in the clause | old/new from the diff (STEP6 describes it) |
| R1.4, new sentence "Fuel costs are not represented separately: … market-price scenarios." | 97 | yes | STEP6 B |
| R2.2 "every evaluation behind a reported comparison in Appendix~E" | 125 | yes | STEP6 B |
| R2.5 "every evaluation repeated from an earlier record (Section~4.7 and Appendix~E.6)" | 159–160 | yes | STEP6 B |
| Further changes, "every single-scenario recourse evaluation those comparisons use" | 285 | yes | STEP6 B |
| header comment dated v7 | 1 | yes | diff (STEP6: "Header comment dated v7") |

**Pointer multiset v6 → v7.** The only difference is l. 160 "Appendix~E" → "Appendix~E.6". STEP6 says the same: "No
pointer changes beyond Appendix~E.6". The count stays at 165.

### (4) Numbering and the letter's pointers at `e5322f8`

**Labels.** The static numbering uses W178c's `numbering_adjusted` and the clone's `elsarticle.dtx` reading; nothing
was compiled.
- 131 labels are compared with `48099ab`: 119 outside Appendix E and 12 in it.
- **0 changed number, 0 removed.**
- The **only added label** is `subsubsec:capacity`, at l. 769. It resolves to **2.3.2**, the subsubsection "Installed and
  Available Capacity" (l. 768).
- 0 undefined refs, 0 duplicate labels.

**Pointers.**

| kind | MATCH | SUBMITTED | MISMATCH | AMBIGUOUS |
|---|---|---|---|---|
| section | 91 | 1 | 0 | 0 |
| appendix | 26 | 0 | 0 | 0 |
| table | 16 | 0 | 0 | 0 |
| figure | 10 | 3 | 0 | 0 |
| equation | 10 | 0 | 0 | 0 |
| algorithm / reference / "new" / "former" | 2 / 4 / 1 / 0 | 0 / 0 / 0 / 1 | 0 | 0 |
| **total (165)** | **160** | **5** | **0** | **0** |

- **Compared with W179b:** every pointer has the same verdict and number as at `48099ab`, except the new one.
  - **The new pointer:** l. 160 "Appendix~E.6" (`app:results_statistics`) = E.6, MATCH. Its needle is "every evaluation
    repeated from an earlier record replayed that record bitwise", and it is present.
- **Content needles:** 60 pointers carry needles; **0 absent**.
  - W179b's needles were carried over, with one replaced. W179b's needle for l. 136 §2.2.6 quoted the sentence that A.1
    rewrote ("are fixed parameters of the operational subproblem", now absent).
  - It is replaced by "are held fixed in the operational subproblem", which is present. The letter's claim there, "the
    investment being a fixed parameter of the recourse", still holds.
  - The first trial run, before this replacement, failed on exactly this needle.
  - Added for l. 97 §3.3: "no separate generator cost curves are used", present.
- **SUBMITTED (5, deliberate, unchanged):**
  - l. 100 "Figure 2" and l. 102 "Figure~2" (R1.5);
  - l. 141 "former 2.2.7";
  - l. 232 "Section~4.4" and "Figure~15" (the reviewer's).

### (5) The eight W179b rows

| row | at W179b | now | evidence (all checks located and passing) |
|---|---|---|---|
| C-1 (Conclusions, A.9) | still, L: search evaluations not covered; "movement measured" is a fit on x = 0 only | **closed** | See (a) below |
| R14-2 (letter R1.4) | still, L: fuel attributed to the introduction; "fuel" occurs 0 times in main.tex | **closed** | See (b) below |
| C47-1 (§4.7 last sentence, A.5) | L: the 41 search evaluations are not in Appendix E | **closed** | See (c) below |
| C48-1 (§4.8, A.6) | L: E.24 prints the split for 5 of 7 | **closed** | See (d) below |
| R22-b (letter R2.2) | L: the 41 search evaluations and the 3 × 3 pair are not in Appendix E | **still (narrowed), L** | Blocked 1 |
| FC-2c (letter bullet, l. 285) | L: the bullet drops "those comparisons use" | **closed** | See (e) below |
| G12 follow-on (§2.2.6, A.1) | new L: the un-hatted S/E^Inv were called "fixed parameters of the operational subproblem" | **closed** | See (f) below |
| A.3 prefix (A.8) | L: "Subsection~\ref{app:…}" prints "Subsection Appendix A.4" | **closed** | See (g) below |

**(a) C-1.**
- "with the remaining movement of the plan without storage estimated from a continued run" (l. 1737) is exactly what
  §4.5 and Table 13 report:
  - §4.5 l. 1606: "the evaluation of the plan without storage was continued … A subsequent fit … 6.2 k€";
  - Table 13: D is a "damped-cosine fit".
- "the search's own evaluations at the search resolution" is supported by §4.1 l. 1368 ("none reduced … beyond the
  search resolution").
- §4.7 l. 1706 states their exit: "each stopped by the production exit … compared at the search resolution".
- "with its remaining movement measured" is gone.

**(b) R14-2.**
- "Fuel" is removed from the "introduction now states" clause.
- What the introduction actually says:
  - l. 381: "alternative trajectories of future ESS investment costs and operational scenarios of load, RES generation
    and market prices".
  - l. 382: "Demand growth and RES capacity follow predefined trajectories, while policy changes, infrastructure
    retirements and the cost of capital as a stochastic quantity are outside the present formulation and left to future
    work".
  - So every source the clause attributes to the introduction is there as listed.
- "Technology maturation" is the letter's gloss. The introduction says "future ESS investment costs", and §3.2 l. 1149
  ties the trajectories to "technological progress".
- What §3.3 actually says (l. 1184, the subsection "Transmission Network", number 3.3): "Conventional generation in the
  TN is priced at the wholesale energy price of the scenario; no separate generator cost curves are used."
- The new letter sentence restates that line. Its "so fuel-cost uncertainty enters only through the market-price
  scenarios" is the letter's inference, not a manuscript claim, and it is consistent with "no separate generator cost
  curves".
- "fuel" still occurs 0 times in main.tex.

**(c) C47-1.**
- l. 1691 now scopes Appendix E to "every single-scenario evaluation behind it".
- It sends the search evaluations to their counts: §4.7 l. 1706, "41 new recourse evaluations in all (14, 20 and 7 …)".
- It sends the multi-scenario pair to Table 13 (`tab:res_multiscenario` = 13), which holds "(cycle 72)".

**(d) C48-1.**
- l. 1717 now reads "wherever the split was recorded".
- In Table E.24, the share columns are "---" exactly for evaluations 40 and 41.
- E.5 l. 2657 says "(for evaluations~40 and~41 the split and the residual were not recorded)".
- The five recorded splits are 55.2–55.3 / 13.7 / 31.0–31.1.

**(e) FC-2c.**
- l. 285 now carries the Appendix E introduction's qualifier ("those comparisons use") plus "single-scenario".
- "Those comparisons" = the bullet's "every pairwise comparison" = Table E.20. The Eval. column of E.20 indexes E.21.
- The multi-scenario pair is outside E.21; the Appendix E introduction sends it to Table 13.

**(f) G12 follow-on.**
- l. 697 now reads "are held fixed in the operational subproblem, where they enter as the parameters Ŝ, Ê of
  Subsection~\ref{subsubsec:capacity}". The label resolves to 2.3.2.
- The nomenclature (l. 316–317) separates the two:
  - S/E^Inv: "Converter rating and energy capacity installed …";
  - Ŝ/Ê: "The same additions as fixed parameters of the operational subproblem".
- §2.3.2 uses the hats ("whose ratings are the fixed additions Ŝ … Ê … of the candidate plan").

**(g) A.3 prefix.**
- l. 1900 now reads "their tolerances (\ref{app:admm_updated_stopping})".
- 0 label-word prefixes remain.

### (6) Counts at HEAD

- **`[CONFIRM`:** 0 in main.tex and 0 in the letter. Comment lines are counted too.
- **General pattern `[A-Za-z]+~\\ref\{app:`:** 8 hits on typeset lines, all prepositions, printing "in Appendix X" or
  "of Appendix X":
  - "in~" at l. 743, 1178, 1182, 1209, 1691 and 2692;
  - "of~" at l. 2395 and 2692.
  - The same 8 hits appear on raw lines.
- **Label-word prefixes** (Appendix / Section / Subsection / Table / … followed by `~` or a space, then `\ref{app:`):
  **0**. There are also 0 cases of "Appendix~\ref{…}" before any label.
- **`% [AUTHOR]`:** 0 (Blocked 2).

### (7) Negative controls (all refused)

All six run in memory, or in a temporary directory outside the repository.

| control | result |
|---|---|
| NC1: "1.8 × 10⁻⁶" → "1.9" in A.3 | A.3 not verbatim; reconstruction fails |
| NC2: one word altered in the Introduction (l. 381), outside every edit | every edit still verbatim, but the reconstruction fails at exactly l. 381 |
| NC3: a subsection inserted before §3.3 | §3.3 → 3.4; the l. 97 "Section~3.3" pointer is MISMATCH, with 9 MISMATCH in all |
| NC4: the §3.3 needle sentence removed | needle reported absent |
| NC5: letter "Appendix~E.6" → "E.9" | UNDECLARED, not MATCH |
| NC6: one word altered in the R1.4 sentence | letter check refused |

## Unexpected findings

1. **Scratchpad collision with the W180a Worker (trial run only; no committed artifact affected).**
   - My first trial run used the session scratchpad sub-directory `trial1`. The W180a Worker shares the same scratchpad
     and wrote into the same directory, including to the same `launch.log` (interleaved) and a `manifest_sha256.json`.
   - My trial outputs there, and possibly W180a's, are therefore mixed. I re-ran into uniquely named directories
     (`w180c_only_trial_02`, `_03`).
   - The committed run is in `data/SRP1/Results/P515S53/w180c_text_letter/`, which was created fresh, and its log holds
     only W180c lines.
   - W180a should not rely on anything in `scratchpad/trial1`.
2. **A tension that A.7 makes visible (L, not in the eight rows; scoped to §2.2.7, §4.8 Scope and E.3).**
   - §2.2.7 now says "the bands, gaps and slacks of every single-scenario evaluation used in a reported comparison are
     tabulated in Appendix E".
   - §4.8 Scope (A.7) and E.3 report a comparison of six single-scenario search-only neighbours ("beyond the search
     resolution for the other six"; 3.47–7.17×). Their bands, gaps and slacks are not tabulated: they ended at the
     production exit, on an earlier code state.
   - The same six are the secondary reading under R22-b. Whether "reported comparison" in §2.2.7 is meant to exclude
     comparisons at the search resolution is a wording question for the expert.
3. **STEP6 bookkeeping (no effect on the text).**
   - §A is headed "the edits, in file order", but A.9 (l. 1736) comes before A.8 (l. 1899) in the file.
   - Addendum 79's W180 order says "the ten main-text edits (§A.1–A.9)". There are nine items; the tenth is A.10 =
     Appendix E, or A.1's second hunk.

## Not confirmed

- **Typeset output.** Nothing was compiled; no compilation is available here.
  - All numbers (2.3.2, 3.3, E.6, Tables 13 and E.20–E.24) and the "in Appendix X" rendering are static readings, using
    W178c's `numbering_adjusted` and the clone's `elsarticle.dtx`.
  - Overleaf's installed `elsarticle.cls` was not inspected.
  - The expert's compile claims (0 errors, 0 undefined, 0 multiply-defined, one label added) are matched statically,
    not reproduced.
- **Pointer semantics.** Each letter number is mapped to the label it is meant to cite, and that label's number at HEAD
  is checked. A pointer to the wrong place with a correct number is caught only where a needle exists: 60 of 165.
- **Appendix E content (A.10).** Only located here: every non-A.1–A.9 hunk lies inside the block. Its entries and prose
  are W180a's scope. §2.3.3's figures (A.3: 10⁻⁶, 1.8 × 10⁻⁶, 5 × 10⁻⁶, evaluation 16) are checked as verbatim text
  only; their rating against W179c is W180d.
- **The `% [AUTHOR]` history.** Searched: main.tex and the letter at `e5322f8` and `48099ab`, and `git log -S'[AUTHOR'`
  over the clone's main.tex. Branch `main` is the clone's only branch. Not searched: other files of the Overleaf
  project.
- **R22-b, secondary reading.** Whether the six search-only neighbours count as "behind a reported comparison" is a
  reading of the text. I looked at §4.1 l. 1368, §4.8 Scope l. 1725 and E.3 l. 2588.
