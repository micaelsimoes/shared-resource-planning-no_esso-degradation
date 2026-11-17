# P5.15 W189 part 2: round-9b audit of identity, labels, pointers, structure and counts at Overleaf 46c9d77 (Addenda 89–90)

**Worker, 2026-11-10.** Zero solves, zero model builds, zero pickle loads. The harness read the Overleaf clone,
`manuscript_review/`, `manuscript_submitted/main.tex` and committed artifacts only. It ran no fetch, pull or checkout and
edited nothing in the clone, `manuscript_review/`, production, earlier harnesses, TASKS.md, REVISION_CONTEXT.md or the
brief. Numbers and claim sentences are W189 part 1's task, and supplement v2 is W189a's; neither is done here.

**State.** The clone `manuscript/6a67305f25e8348fb71380c3` is at HEAD `46c9d779743b40a42da73c0050d8923e6af6fb55`
(2026-11-10 18:13:45 UTC), whose parent is `a87e195`. Its tracked files are clean; the only untracked entry is
`.DS_Store`. `a87e195 → HEAD` changes exactly `main.tex`, `response_to_reviewers_draft.tex`, `cover_letter.tex` and the
new `fig_soc_soh_reference_unit.pdf`. The repository HEAD at run time was `8c931b0b` (the harness commit). The only
modified tracked file was `TASKS.md`, which is the Planner's and was not staged.

**No TeX is installed on this machine.** Every section, table, figure and equation number below is a static number
(W182c method, the clone's `elsarticle.dtx`). Nothing was compiled.

## Blocked on Planner

None of these changes what runs next. Items 1 and 3 are wording or numbering points for the expert or the author. Item 2
is a reporting point against Addendum 89 decision 4.

1. **One letter pointer is MISMATCH: letter v12 l. 285 "Fig.~B.8" should read B.9 (static).**
   - `fig:investment_lattice` is the only label whose static number differs from the numbering STEP6 states (STEP6
     says E.19 → B.8).
   - The reason: `elsarticle.dtx` l. 1459–1469 shows that `\appendix` resets section, subsection and equation, and
     redefines `\thefigure` as `<letter>.<figure counter>`. It does not reset the figure counter.
   - The main text now holds 8 figures (1–5, the new 6, 7, 8). `algorithm2e` is loaded as `[ruled,vlined]`, so it
     uses its own counter and does not step the figure counter. The lattice figure is therefore the 9th figure: B.9.
   - STEP6 uses the same counter reasoning for the tables ("print as B.10 and B.11 because the table counter continues
     from the main text").
   - This is static only. The expert's compiled preview `main_round9b_preview_demo_graphics.pdf` is listed in STEP6 but
     is not in the repository.
   - **Action:** read the figure number off the Overleaf compile. If it is B.9, letter l. 285 and STEP6's table need
     the fix.
2. **The trimming target of Addendum 89 decision 4 is not met, measured statically.**
   - The main text holds **17 floats: 9 tables + 8 figures**, plus Algorithm 1. The target is ≤ 10. At `a87e195` it
     held 18 (11 + 7).
   - Main-text prose went **from 6,130 to 6,692 words**, and Appendix A from 2,497 to 2,605 words.
   - Appendix B (the reduced Appendix E) has 1,550 prose words, the longtable B.10, Table B.11 and Fig. B.9, over 206
     source lines. Whether that is one page cannot be decided statically.
   - STEP6 "Deferred" leaves the deeper cut to the author. This is recorded for the decision, not acted on.
3. **Five lines changed in round 9b have no `\textcolor{blue}` wrapper** (Found 5): l. 1250, l. 1685, l. 1695,
   l. 1709 and l. 1710 of `main.tex`. Every caption that is new or changed is blue.

## Changed

- **`8c931b0b`** adds the harness `p515_s55_w189_p2_structure.py` (sha256 `39aa205d…`). It was committed **before** the
  real run; the run records that HEAD and `script_committed_clean: true`.
  - It imports W183h `p515_s53_w183h_round8.py` without editing it, and through it W182c, W181a, W179b, W178c,
    W177b-3′, W177b-3, W167, the S47 record, W182b's chain and W180c.
  - None of those harnesses was run, and none of their output directories was written.
- **This commit** adds this report and `data/SRP1/Results/P515S55/w189_p2_structure/`.
  - The directory was created fresh with `mkdir`, and every file was opened `'x'`.
  - The run was attached, with stdout and stderr in `launch.log`. Exit code 0; 0 check failures.
  - Guards: `SolveProfileGuard(permitted=())` was installed before any project import. All 58 distinct guards of the
    chain verified at exactly 0. `pickle.load` / `pickle.loads` were blocked, and every counter was 0. No production
    module was imported.
  - JSON was written only through `gate_result_io`. The W100 bool-typing test gave PASS.
  - Four scratch trials ran first under the session scratchpad (`w189p2_trial_1..4`). They exposed my own faults, all
    fixed before the commit:
    - a JSON tuple key;
    - an 8-word control sentence that was meant to have 7;
    - SUBMITTED carry-over by key instead of by occurrence (l. 102 has two "Figure~2");
    - hunk-23 placement (see Found 1);
    - majority-of-characters blue detection on tokens like `\caption{Storage`.

| file | sha256 |
|---|---|
| `w189_p2_structure.json` | `1d0071561bb377b1c1a0bdc178e4d8d6e7b3ea22a32a4cf7131df417e5b54993` |
| `w189_p2_structure.md` | `5062b85e08f5f142a3cdd5088287f704f581710388220b5093b1dfe9b55ffc9c` |
| `manifest_sha256.json` | `313a6d363f550ba3854d408d5c6f00fd58ec87d6f88e560bd7ba8a9eb9934a0e` |
| `launch.log` | `784c7de54b79205a561ad38903ee2e7d24b89c66ed14d5245e860f1ca80ecde1` |
| `w189p2_bool_typing_test.json` / `.log` | `469e6846…` / `c671e90c…` (VERDICT: PASS) |
| `manifest_post_run_sha256.json` | `aeac5491955c358d2512a0c32693ea3d4bd88812aa81b17af05993daf5f42bb7` |

`manifest_sha256.json` hashes:
- every input: the four clone files, both Figure 6 PDFs, `elsarticle.dtx`, the four round-9b and the four round-8
  review copies, the three expert diffs, letters v9 and v10, `manuscript_submitted/main.tex`, the W183h, W182c and W180c
  JSONs, `STEP6_ROUND9_CORRECTIONS.md`, the brief and every imported script;
- the `git show` blobs of the four files at `a87e195` and at HEAD.

Commands (repo root, canonical interpreter, run attached):

```
mkdir data/SRP1/Results/P515S55/w189_p2_structure && \
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p2_structure.py \
    > data/SRP1/Results/P515S55/w189_p2_structure/launch.log 2>&1                                  # exit 0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \
    --out data/SRP1/Results/P515S55/w189_p2_structure/w189p2_bool_typing_test.json \
    > data/SRP1/Results/P515S55/w189_p2_structure/w189p2_bool_typing_test.log 2>&1                 # exit 0, PASS
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p2_structure.py --post-run  # exit 0
```

## Found

No objective value is computed in this stage.

### 1 Identity — **held**

| file at 46c9d77 | md5 (declared) | = review copy | clone / expert hunks | +/− lines | `a87e195` blob = |
|---|---|---|---|---|---|
| `main.tex` | `49139cc2` ✓ (Addendum 90) | `main_a87e195_round9b.tex` ✓ | 34 / 34 | +74 / −699 both; 33 of 34 hunks equal; see note | `main_edd7a63_round8.tex` ✓ |
| `response_to_reviewers_draft.tex` | `ed8f5b13` ✓ | `response_to_reviewers_v12.tex` ✓ | 10 / 10 | +31 / −34; every hunk equal, ranges and context too | `response_to_reviewers_v10.tex` ✓ |
| `highlights.tex` | `4cedd45c` ✓ | `highlights_v3.tex` ✓ | 0 (unchanged since `a87e195`) | — | `highlights_v3.tex` ✓ |
| `cover_letter.tex` | `0b9e4c21` ✓ (STEP6) | `cover_letter_v3.tex` ✓ | 1 / 1 | +2 / −2; equal | `cover_letter_v2.tex` ✓ |

- **Note on `main.tex` hunk 23.** It is a pure 4-line insertion, and the expert's diff places it one line later
  (`@@ -1836,6 +1818,10` against git's `@@ -1835,6 +1817,10`). The block's first line, `\textcolor{blue}{`, equals the
  line after it, so the two placements are the same edit: the '+' lines are a rotation of each other.
  - The test that does not depend on placement: the expert diff, applied to the `a87e195` blob with every context and
    '−' line verified, gives `46c9d77` byte for byte. The same holds for the letter and the cover letter.
  - The hunk-23 placement difference appears under all four git diff algorithms (myers, minimal, patience,
    histogram); the apply test does not depend on the algorithm.
- `fig_soc_soh_reference_unit.pdf` in the clone is byte-identical to the `manuscript_review/` copy (sha256
  `287bccf7…`). It is absent at `a87e195`.
- **Cover letter v3 still carries the bracketed round-10 paragraph** at l. 21 ("[Paragraph on the additional
  experiments of the revision …"). This is expected per STEP6 "Deferred", and is recorded so it is not submitted as is.
- Negative control: one '+' line of each expert diff altered in memory is flagged at exactly that hunk (plus hunk 23
  for `main.tex`), and the patched result no longer equals HEAD. All three were refused.

### 2 Label map `a87e195 → 46c9d77` — **1 disagreement with the stated numbering**

- **Labels: 136 → 102.** 83 unchanged, 18 renumbered, 35 removed, 1 added. The full per-label table, with caption or
  title at each revision, is in `w189_p2_structure.md`.
- **The stated numbering** (STEP6 "Numbering changes" and Addendum 90) is encoded label by label, and every other
  label is expected unchanged. **101 of 102 HEAD labels and every removed label agree.**
  - The one exception is `fig:investment_lattice`: stated B.8, static **B.9** (Blocked 1).
- **Confirmed as stated:**
  - Figures 1–5 unchanged; `fig:res_soc_soh` new as 6; `fig:res_benchmark` 6 → 7; `fig:res_q_cycle` 7 → 8.
  - Tables 1–4 unchanged; `tab:res_energy_ladder_m2` 6 → 5, `res_ageing` 7 → 6, `res_benchmark` 8 → 7,
    `res_multiscenario` 9 → 8, `res_year_ladder` 10 → 9.
  - `tab:res_breakeven` (5) and `tab:res_discount` (11) removed.
  - §4.8 `subsec:res_limitations` removed, and §4.7 retitled "Numerical Performance, Scope and Limitations" (title
    matches STEP6). The shingle hint puts 0.934 of the old §4.8 text in the new §4.7.
  - A.7 `app:admm_updated_scenarios` removed, with a hint of 0.392 of its text in A.1 ("folded into A.1", rewritten);
    A.8 → A.7.
  - Appendices B, C, D removed (10 appendix headings, 11 figures and 8 tables).
  - E → B: E.1 → B.1, E.2 → B.2, E.4 → B.3, E.7 → B.4; Eqs. (E.1)–(E.2) → (B.1)–(B.2); Tables E.20 → B.10 and
    E.22 → B.11.
  - E.3, E.5, E.6 and Tables E.21, E.23, E.24 removed.
- **Equations.** Main text: every equation label is unchanged. The static count is **(1)–(41)**, 42 numbered rows
  with (3a)/(3b), at both revisions.
  - STEP6's "(1)–(40) unchanged" is true as far as it goes, but there are 41 numbers.
  - STEP6's "40 equations" is one short of the static 41.
- **Appendix A compaction.** A.1 "Agents, Blocks and Consensus Variables" and A.6 "Algorithm" are both still present,
  with unchanged numbers.
  - What was folded is A.7 into A.1, as STEP6 describes. The "A.1 and A.6 folded into A.2" of Addendum 89 ruling 7
    (and "folded A.1/A.6" in the task) did not happen in that form.
  - At HEAD, Appendix A runs A.1–A.7.
- **Citations.** The static citation order is identical at both revisions (0 renumbered).

### 3 Letter v12 pointers — **171 MATCH, 8 SUBMITTED, 1 MISMATCH, 0 AMBIGUOUS** (180 pointers)

**Method.** The ratings are chained through letter v10:
- **(i) v10 at `a87e195`.** W183h's (c) was re-derived in memory with W183h's and W182c's own functions: 167 MATCH and
  5 SUBMITTED. This equals the committed W183h JSON on all 172 pointers.
- **(ii) v10 at HEAD.** Each v10 pointer was re-rated at HEAD, with its target taken as the `a87e195` object that
  carried the printed number: 130 MATCH, 37 MISMATCH, 5 SUBMITTED.
  - The 37 MISMATCH are the renumberings v12 was meant to make.
  - The merges STEP6 declares are applied as declared successors: 4.8 → 4.7, A.7 → A.1, and Table 11 → Table 9(b).
- **(iii) v12 pairing.** Each v12 pointer was paired with its v10 pointer through the letter line map (W183h
  `pair_and_rate`).
  - 16 new v12 pointers took a **declared target**. Each declaration cites STEP6 or the letter's sentence, and is
    checked rather than asserted:
    - the label is defined at HEAD;
    - for 7 of them, a content needle sits in the target's region at HEAD (for example "KKT" in A.4, "halved and
      doubled" in §4.7, "three solver starts" in Table 7, "20.5~k" in §4.4);
    - for 11 of them, the v10 counterpart line resolves to the same label.

    All 16 checks hold, and no declaration is stale.
  - 3 new pointers to the submitted version are **declared SUBMITTED**:
    - l. 282 "submitted Appendix~E", which denotes "Results" in `manuscript_submitted/main.tex`;
    - l. 282 "submitted Appendices~B--D" (Market Data … Distribution Networks);
    - l. 283 "submitted Appendix~D" (Distribution Networks).
  - The other 5 SUBMITTED are W183h's: l. 100, 102 "Figure 2", l. 141 "former 2.2.7", and l. 230 "Section~4.4" and
    "Figure~15".

**MISMATCH (the only one):**

| line | printed | should be (static) | v10 counterpart | why |
|---|---|---|---|---|
| 285 | Fig.~B.8 | **B.9** | l. 288 "Fig.~E.19" (`fig:investment_lattice`) | the figure counter continues from 8 main-text figures (Blocked 1) |

**Table 7 — expectation "Table 7 at the benchmark bullet and at R2.6": held, with a reading.**
- The letter has exactly **one** "Table~7", at l. 170 in R2.5/R2.6: "The spread across independent starts is reported
  for the static benchmark arms (three-start bands, Table~7)". It is **MATCH** against `tab:res_benchmark`.
  - Its v10 counterpart, l. 171 "Table~8", was `tab:res_benchmark` at `a87e195`.
  - This sentence is the "benchmark pointer" Addendum 90 names (v11's "three-start bands, Table 6").
- The further-changes bullet on the benchmark (l. 279) carries only "Sections~3.5 and~4.4" and the new "(Section~4.4)"
  for the NRF sentence. All three are MATCH.

**The other renumberings are all MATCH:**
- Section 4.8 → 4.7 (×9 v10 pointers; those on the rewritten l. 172–173 through declared targets);
- Tables 7 → 6 (×6);
- Table 10 → 9(a) and Table 11 → 9(b) (the extractor reads "9");
- Tables 4–11 → 4–9 (members `tab:res_breakeven` and `tab:res_discount` gone; endpoints correct);
- Appendix A.8 → A.7 (×4);
- Appendix E → B (×6);
- E.7 → B.4 (×2);
- Eqs. (E.1)–(E.2) → (B.1)–(B.2) (×2).

Citations [42] and [43] are MATCH.

**v10 pointers with no v12 counterpart:** v10 l. 171–173 and l. 235. These are the rewritten R2.6 and R3.2 lines, and
their v12 replacements are among the declared targets.

**Negative controls, all refused:**
- l. 170 "Table~7" → "Table~8", with the declared target kept;
- a paired "Section~4.7" → "Section~4.8";
- a declared "Appendix~A.4" → "A.5".

### 4 Dangling references — **held**

- **`main.tex` at HEAD.** Statically there are 0 undefined `\ref`/`\eqref` keys (151 uses, 62 keys, 102 labels) and 0
  duplicate labels.
  - 0 uses of any of the 35 removed labels, either typeset or in commented lines.
  - 0 kind-word mismatches (a "Table/Figure/Section" word before a `\ref` of another kind).
  - 0 typed (literal) numbered pointers.
- **Prose mentions of the removed objects, in `main.tex`:**
  - 0 typed "Appendix B–E" and 0 lettered C./D./E. numbers;
  - 0 "scenario mechanics".
  - The rest is informational, and every hit is consistent with round 9b:
    - "Market Data" (l. 1248) is the §3.2 heading.
    - "the break-even fit of Subsection~\ref{subsec:res_breakeven}" (l. 1348) points to the section that holds it,
      not to the removed Table 5.
    - "discount-rate sensitivities" is in the Table 9 caption.
    - "limitations" hits are l. 364 (literature), the §4.7 heading and l. 1728.
    - "evaluation record(s)" hits are l. 748, l. 1571, l. 2030, the Appendix B heading and l. 2120.
- **Supplementary pointers in `main.tex`:**
  - S1, S2, S3, S1–S3, S4, S5, S6: all inside S1–S6.
  - S5 is cited twice as a *Section* (l. 1353 "evaluation procedure and voltage-consistency checks are detailed in
    Section~S5", and l. 1627), while l. 2120 names "Table~S5" (solver starts). Whether S5 is a section holding that
    material is a W189a question about the supplement's content; it was not checked here.
  - The letter cites S1–S6 and S3.
- **Letter:** "Appendix C/D/E" appears only in the three declared-SUBMITTED pointers (l. 282, 283).
- Controls refused: an injected `\ref{tab:res_discount}` is reported undefined, and a duplicated
  `\label{tab:res_ageing}` is reported.

### 5 Hygiene — `[CONFIRM` and prefixes **held**; blue **not held on 5 lines**

- **`[CONFIRM`:** 0 in each of the four files.
- **Tightened pattern `\b(Appendix|Section)~\\ref\{app:`** (Addendum 86 ruling 4 (i)): **0** in `main.tex`, both raw
  and typeset. It is also 0 in the letter, the highlights and the cover letter. The seven prefixes W183h found at
  `a87e195` are gone.
- **General `[A-Za-z]+~\\ref\{app:`:** 11 matches, all prepositions: "in" 9, "of" 2.
- **Blue.**
  - **Captions:** every caption that is new or changed in round 9b is fully blue (share 1.0): `fig:res_soc_soh`,
    `tab:res_year_ladder` and `tab:appE_claims`.
  - **Changed lines with content outside any wrapper** (on round-9b '+' lines):

| line | class | text not blue | written / moved verbatim |
|---|---|---|---|
| 1250 | prose (§3.2) | "the supplementary material (Section~S1)." (replaces `~\ref{app:market_data}`; the surrounding original paragraph was never blue) | 5 / 0 |
| 1685 | Table 9 panel header | `\multicolumn{4}{l}{(a) Installation year, 2\,\% discount rate}` | 6 / 0 |
| 1695 | Table 9 panel header | `\multicolumn{4}{l}{(b) Discount rate, 2025}` | 4 / 0 |
| 1709 | Table 9 note (a) | "(a) $M = I + Q(x) − Q(0)$ is the cost increase …" (the merged note) | 18 / 26 |
| 1710 | Table 9 note (b) | "(b) Investment cost $I=318.0$~k€, paid in 2025; all investment margins are determinate …" | 5 / 0 |

  - Wrapper count (raw `\textcolor{blue}` occurrences): 410 at `a87e195` and 400 at HEAD. Most of the difference is
    the text removed with Appendices B–D and E.3/E.5/E.6.
  - Controls refused: a planted `[CONFIRM` is counted; a planted `Appendix~\ref{app:results}` is matched; and the
    Figure 6 caption, unwrapped in a copy, is detected.

### 6 Counts

**Counting rule** (also in the JSON):
- **Source:** comment-stripped.
- **Regions:**
  - main text = `\section{Introduction}` to `\appendix`;
  - each appendix = its `\section` to the next one;
  - Appendix B = its `\section` to the unnumbered declaration section;
  - the abstract is counted apart.
- **Removed before counting:**
  - floats (`table`, `figure`, `longtable`, `algorithm`, starred forms), with caption words counted apart;
  - display math (`equation`, `align`, `gather`, `multline`, `eqnarray`, `displaymath`, `subequations`);
  - inline math (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`);
  - heading titles, counted apart;
  - the arguments of `\ref`, `\eqref`, `\cite*`, `\label`, `\url`, `\includegraphics`, `\begin`, `\end`, `\hspace`
    and `\vspace`, and the colour argument of `\textcolor`.
- **Other control words** are removed, but the text of their arguments is kept.
- **A word** is a whitespace-separated token holding a letter or digit, so a hyphenated compound counts once.

| revision | region | prose words | heading words | caption words | tables | figures | algorithms | numbered equations (rows / numbers) | source lines |
|---|---|---|---|---|---|---|---|---|---|
| 46c9d77 | abstract | 246 | — | — | — | — | — | — | 11 |
| 46c9d77 | **main text** | **6,692** | 95 | 491 | **9** | **8** | 1 | 42 / 41: (1)–(41) | 1,446 |
| 46c9d77 | **Appendix A** | **2,605** | 31 | 11 | 0 | 0 | 1 | 8 / 8: (A.1)–(A.8) | 315 |
| 46c9d77 | **Appendix B** (reduced E) | **1,550** | 16 | 132 | 2 (B.10 longtable, B.11) | 1 (B.9) | 0 | 2 / 2: (B.1)–(B.2) | 206 |
| a87e195 | main text | 6,130 | 96 | 471 | 11 | 7 | 1 | 42 / 41 | 1,464 |
| a87e195 | Appendix A | 2,497 | 37 | 11 | 0 | 0 | 1 | 8 / 8 | 301 |
| a87e195 | Appendices B / C / D | 49 / 37 / 194 | | 8 / 22 / 173 | 0 / 1 / 7 | 1 / 1 / 9 | | 0 | 28 / 47 / 373 |
| a87e195 | Appendix E | 1,898 | 27 | 369 | 5 | 1 | 0 | 2 / 2 | 379 |

- **Main-text floats (tables + figures): 17**, against a target of ≤ 10. That is not met (Blocked 2). Algorithm 1 is
  counted apart.
- **Appendices at HEAD: 2** (A and B). There were 5 at `a87e195`.
- **Compared with STEP6's statement** ("6.5 k prose words, 9 tables + 9 figures, 40 equations"):
  - static count 6,692 words (rule-dependent);
  - **9 tables + 8 figures**, plus 1 algorithm;
  - 41 equation numbers.
- **Pages: not estimable statically.** There is no TeX here and no `.aux`, `.log` or round-9b PDF in the repository or
  the clone. The expert's "63 pages (was 76)" is not confirmed. Source lines and characters are given only as
  proxies, not as pages.
- Controls refused:
  - a 7-word sentence with inline math adds exactly 7 prose words;
  - one added numbered equation adds one numbered row.

### Negative controls: 14 of 14 refused

- identity: one per file with an expert diff (×3);
- the label shift;
- pointers (×3);
- dangling (×2);
- hygiene (×3);
- counts (×2).

## Not confirmed

- **Compilation.** Nothing was compiled; there is no TeX and no `.aux`, `.bbl`, `.log` or PDF of round 9b.
  - Every label, citation and page figure is static. The expert's "0 / 0 / 0, 63 pages" was not reproduced.
  - The B.8 against B.9 question (Blocked 1) rests on the `elsarticle.dtx` reading and on counting 8 main-text figure
    floats. `main_round9b_preview_demo_graphics.pdf`, which STEP6 lists, is not in `manuscript_review/`; I searched
    the repository by name.
- **The remote.** The clone's HEAD is `46c9d77` as present on disk. Nothing was fetched.
- **Declared pointer targets.** The 16 declared targets and 3 declared SUBMITTED readings rest on STEP6 and the letter's
  own sentences. The checks above (needle, v10 counterpart, label defined) support them but do not prove intent.
  - The counterpart check is set-based: a pointer of the same kind on the v10 line resolves to the declared label.
  - "Table~9(a)/(b)" is read as Table 9; the panel letter is not checked.
- **Supplementary pointers** (S1–S6) were inventoried only. Whether each S-number names the material the sentence
  says, for example "Section~S5" against "Table~S5", is W189a's scope.
- **Word counts** follow the stated rule. Other rules, such as counting captions, table notes or headings as prose,
  give different totals, and all components are reported so another rule can be applied. Table notes inside float
  environments are not in the prose count.
- **The blue check** is token-level. Text moved verbatim from `a87e195`, which was not blue there, is reported as
  "moved" and still listed when it is not blue now. Lines whose only change is structural, such as `\label`, `\begin`
  or `\includegraphics`, are excluded by rule.
