# P5.15 W182c: label map, letter v9 pointers, dangling references, dropped blue and Appendix E at Overleaf edd7a63 (Addendum 82 v2, W182 items A3–A6)

**Worker, 2026-10-25.** Zero solves, zero model builds, zero pickle loads. The harness only read the Overleaf clone,
`manuscript_review/` and committed artifacts. It ran no fetch, pull or checkout, and it changed nothing in the
manuscript or the letter.

**State.** Overleaf clone HEAD is `edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838` (2026-10-24 15:54 UTC), and its tracked
files are clean. `main.tex` has 2,956 lines, sha256 `b08a3365607e56f2…`, which matches the critical review's
fingerprint `b08a3365607e56f2`. At run time the repository HEAD was `2f4df40c`, and the only modified tracked file was
`TASKS.md`, the Planner's.

**No TeX is installed on this machine** (`pdflatex`, `latex`, `latexmk`, `lualatex` and `xelatex` are all absent), and
the clone holds no `.aux` or `.bbl` file. So every number below is a **static** number, and nothing was compiled. The
static rules are those of W177b-3′/W178c (`numbering_adjusted`) and the clone's `elsarticle.dtx`. `\appendix` resets
section, subsection and equation. It does not reset the figure or table counters, which print as `<letter>.<n>`.
Citation numbers assume elsarticle-num's citation order (unsorted).

## Blocked on Planner

1. **The A6 bitwise prediction failed.** The `gen_appE.py` output occurs **0 times** in `main.tex` at `edd7a63`. The
   control at `8f051a2` finds it exactly once (byte 188,848).
   - The author edited the Appendix E block by hand. The edits form six hunks: a new E.1 "Terminal Salvage Credit"
     (30 lines), the E.21 group-(d) header, the lattice figure and its paragraph moved into E.4 (11 lines), the E.5
     paragraph (1 line → 8 lines), the E.7 paragraphs (1 line → 20 lines), and one blank line.
   - **0 of 1,637** table entries in E.20–E.24 changed, and the tables are still numbered E.20–E.24.
   - `verify_appE.py` gives 33 PASS, 0 FAIL, so that prediction held. Note that the verifier never reads `main.tex`, so
     this result says nothing about the manuscript block.
   - Addendum 82 names no fallback. **Decision needed (expert):** either update `gen_appE.py` to the author's block, or
     declare the block hand-maintained and drop the bitwise gate.
   - If the appendix is regenerated as it stands, it **reintroduces an undefined reference**. Generated line 169 still
     emits `Tables~\ref{tab:res_flex_ladder}`, a label that is gone at `edd7a63`. The author's block reads
     `Fig.~\ref{fig:res_flex_ladder} and Table~\ref{tab:res_energy_ladder_m2}`.
2. **Letter v9 needs renumbering throughout.** 65 pointers are MISMATCH, listed in Found A3. Fixing them is an
   author or expert edit, not mine.
   - One pointer is **AMBIGUOUS** and needs an intended target: l. 97 "Section~3.3". §3.3 "Transmission Network" was
     merged away. Its table moved to Appendix C (C.12), and one of the two W180c needles ("no separate generator cost
     curves are used") now sits in the new §3.3 "Transmission and Distribution Networks". The other needle is no
     longer typeset anywhere.
3. **Citation numbers moved, so the [43]/[44] check in Addendum 81's W182 (b) would now fail.** `boyd_admm_2011` is
   first cited later at `edd7a63`, so it moves from [36] to [44] and [37]–[44] each move down by one.
   - `naumann_2018` moves from [43] to [42], and `eve_mb31_datasheet` from [44] to [43].
   - The letter's [44] (l. 77, l. 249) and [43] (l. 256) are therefore MISMATCH.
   - This is static citation order, not a compiled `.bbl`.
4. **Counts that differ from the task text.**
   - There are **67** commits in `8f051a2..edd7a63` (`git rev-list --count`). There are 66 in `d2ebb0d..edd7a63`.
     TASKS.md says 66 and Addendum 82 says "eight".
   - Of the "two removed figures", **one is removed**: `fig:search_generic`, whose figure environment is commented
     out. **The other moved**: `fig:investment_lattice` went from Figure 4 to E.19 in Appendix E. Body figures go from 9
     to 7.

## Changed

- **`179c9355`** adds the new harness `p515_s53_w182c_labels_pointers_appE.py` (sha256 `a933d3e8…`). That is the
  file the run recorded.
  - It was committed *after* the run. The JSON records the script's sha256, and the committed file matches it.
  - It imports, without editing, W179b (and through it W178c, W177b-3′, W177b-3, W167 and the S47 record) and W181a:
    `run_generator`, `run_verifier`, `locate_bytes`, `appendix_block`, `raw_tables`, `diff_tables` and `block_diff`.
- **This commit** adds this report and `data/SRP1/Results/P515S53/w182c_labels_pointers_appE/`.
  - The directory was created fresh with `mkdir`, and every file in it was opened `'x'`.
  - The run was attached, with stdout and stderr captured in `launch.log`.
  - `SolveProfileGuard(permitted=())` was verified at 0 for its own guard and the eight imported ones. Pickle was
    0/0. No production module was imported. There were 0 check failures, and the exit code was 0.
  - The repository-wide W100 bool-typing test gave PASS.
  - The generator wrote only to its explicit path inside this directory. No `appE.tex` exists in the repository root
    or in `manuscript_review/`.

| file | sha256 |
|---|---|
| `w182c_labels_pointers_appE.json` | `3d680d8a93de22e8c8f0b3961619b416428a865aaf35c3e4511fcb2808319559` |
| `w182c_labels_pointers_appE.md` | `31085ded777b2608dcab0d40b6b556e8d6ff2150d5722711ed5e0afd9cdaf43a` |
| `appE_generated_w182c.tex` | `623ddbbdcaa684b4f7d70d525c5519d91466f3fd7ebff735dc6349c483d94986` |
| `verify_appE_output.txt` | `7a0c47e9eee3b365075f0b46bdc6d20e7ec0ff40eab7f68c8213695861d213a7` |
| `manifest_sha256.json` | `6910f92a404b7a694e0fec761e37cf313855b2298ccff73537272e406a20858c` |
| `manifest_post_run_sha256.json` | `4562b225177bb0237b344a15608bb3d3ec8b47a6414d85d58cdcd2937c4c6eba` |
| `launch.log` | `6d776b3f47ab8337c9fb369fbcfdd0ffc854317b0c71396f4450250c781ffdf9` |
| `w182c_bool_typing_test.json` / `.log` | `97cc8a8b…` / `86fb9b69…` |

`manifest_sha256.json` hashes every input: the clone files, the letter v8, `gen_appE.py`, `verify_appE.py`, the
committed appendix, all of `export_W160/`, the W180c JSON, the brief, the review and every imported script. It also
hashes the `git show` blobs of `main.tex` and the letter at `8f051a2`, `cd57efc`, `e5322f8`, `d2ebb0d` and HEAD.

## Found

**Letter identity (zero-solve).** No `response_to_reviewers_v9.tex` exists in `manuscript_review/`. The letter's
identity therefore comes from the clone:
- The letter at HEAD equals its `d2ebb0d` blob (its last change), and its header line reads "— v9 …".
- The letter at `cd57efc` is byte-identical to `response_to_reviewers_v8.tex`.
- From v8 to HEAD the diff has **4 hunks** at default context and 5 at `-U0`.
- All three Addendum 81 v9 sentences are present: "cycle-life calibration bracketed by a manufacturer datasheet", "one
  of them read from a manufacturer datasheet [44]", and "… range reported for LFP cells [43]".
- The `main.tex` diff from `8f051a2` to HEAD has 49 hunks. The files changed are `main.tex`, `main_latexdiff.tex` and
  the letter.

### A3 label map (8f051a2 → edd7a63)

- **Labels.** There are 132 → 136. Of these, 71 are unchanged, 59 renumbered, 2 removed and 6 added. The full
  per-label table, with the caption or title at each revision, is in `w182c_labels_pointers_appE.md`.
- **Cross-check.** The static numbering at `8f051a2` equals that at `cd57efc` and at `e5322f8` for all 132 labels.
  This confirms Addendum 81's "every label number identical to round 7".
- **Removed.**
  - `tab:res_flex_ladder` (Table 9).
  - `fig:search_generic` (Figure 2, commented out).
- **Added.**
  - `tab:res_capacity_summary` (Table 2).
  - `eq:soh_initial` (35).
  - `eq:interface_deviation` (39).
  - `eq:interface_deviation_nonnegative` (40).
  - `eq:acceptance_tolerance` (A.8).
  - `app:salvage_credit` (E.1).
- **Renumbered, by class.**

| class | renumbered (8f051a2 → edd7a63) |
|---|---|
| tables | 6→3 `ageing_calibrations`, 7→4 `res_value_ladder`, 8→5 `res_breakeven`, 10→6 `res_energy_ladder_m2`, 11→7 `res_ageing`, 12→8 `res_benchmark`, 13→9 `res_multiscenario`, 14→10 `res_year_ladder`, 15→11 `res_discount`; 2→C.12 `cs3_tn_res_generators`; 3,4,5→D.17,D.18,D.19 (ADN RES tables moved to Appendix D); D.16–D.19→D.13–D.16 (`cs1_ieee33_*`). Table 1 unchanged |
| figures | 3→2 network diagram, 4→E.19 `investment_lattice`, 5→3 `res_breakeven`, 6→4 `res_flex_ladder`, 7→5 `res_ageing`, 8→6 `res_benchmark`, 9→7 `res_q_cycle`; B.10→B.8, C.11→C.9, D.12–D.20→D.10–D.18 |
| equations | 30→31 `cohort_allocation`, 31→32, 32→33, 33→34, 34→36 `soh_floor`, 35→37 `cycle_life_calibration`, 36→38 `commitment_deviation`, 37→41 `commitment_charge`, 38→E.1 `salvage_credit`, 39→E.2 `salvage_remaining_life`, A.5→A.6 `admm_interface_duals` (new unlabelled (30) and (A.5)). 30 equation labels unchanged, including (13) `recourse_value` and (20)–(29) |
| sections | 2.2.7→2.2.6 `certification`, 2.3.2→2.3.1 `capacity`, 2.3.3→2.3.2 `network_storage`, 2.3.5→2.3.4 `commitment`, 3.4→3.3 `case_study_adns` (now "Transmission and Distribution Networks"), 3.5→3.4 `case_ess_params`, 3.6→3.5 `case_settings` |
| appendix | E.1–E.6 → E.2–E.7 (`app:results_claims` … `app:results_statistics`). A.1–A.8, B, C, D.1–D.3 and E unchanged |

- **Unlabelled headings.** The successor rule is: a label of the heading, else the exact title in the same part, else
  at least two thirds of at least two labels defined inside the heading.
  - 2.3.4 "Degradation Chain of the Shared-ESS Agent" → **2.3.3 "Battery Degradation"**. All seven of the old
    heading's equation labels are now under it.
  - **Not located: 2.2.6 "Coupling Between Planning and Operational Recourse".** The heading line is commented out at
    l. 697.
  - **Not located: 2.3.1 "Shared ESS Operation Subproblem".** It has no successor. Its text was rewritten into the
    §2.3 opening, and the shingle hint is 0.01.
  - **Not located: 3.3 "Transmission Network".** The section was split: the table went to C.12, and the prose went to
    the new §3.3. The shingle hint points 0.50 to Appendix C.
  - Five headings are retitled under unchanged numbers: 2.3, 4.3, 4.5, A.5 and D.3.

### A3 letter v9 pointers at edd7a63

**Prediction (expert) "≥ 10 letter pointers MISMATCH": HELD.** The extractor (W177b-3′'s `letter_pointers`,
unchanged) finds **173** pointers:

| verdict | count |
|---|---|
| MATCH | 102 |
| MISMATCH | **65** (64 distinct line/text/number) |
| AMBIGUOUS | 1 |
| SUBMITTED | 5 |

- **How targets were derived.** Each pointer's intended target is the object that carried the printed number at
  `8f051a2`. For all 146 pointers on lines unchanged since v7, the derived target agrees with W180c's declared target
  (146 of 146).
- **SUBMITTED.** These are W180c's five, carried through the v7 → v9 line map: l. 100 and l. 102 Figure 2, l. 141
  "former 2.2.7", and l. 232 Section 4.4 and Figure 15.

| printed | should be (static) | letter lines | why |
|---|---|---|---|
| Section 2.2.7 (and "new 2.2.7") | 2.2.6 | 43, 81, 125, 137, 142, 152, 156, 173, 230, 235, 275 | 2.2.6 heading removed |
| Section 2.2.6 | — (target removed; heading commented out, needle text not typeset) | 136 | |
| Section 2.3.2 | 2.3.1 | 136 | |
| Section 2.3.3 | 2.3.2 | 210, 241 | |
| Section 2.3.4 | 2.3.3 | 82, 86, 195, 210, 250 | |
| Section 2.3.5 | 2.3.4 | 241 | |
| Sections 2.3.3–2.3.5 | 2.3.2–2.3.4 | 243 | |
| Section 3.5 | 3.4 | 45, 77, 221, 222, 249, 250, 257 | 3.3/3.4 merged |
| Section 3.6 | 3.5 | 125, 126, 282 | |
| Appendix E.6 | E.7 | 160, 284 | E.1 inserted |
| Table 6 | 3 | 249, 250, 256, 257 | |
| Table 11 | 7 | 256, 257, 263, 264 (×2), 265 | |
| Table 12 | 8 | 171 | |
| Table 14 | 10 | 220, 222 | |
| Table 15 | 11 | 97, 98 | |
| Tables 7–15 | 4–11 | 81, 86 | member `tab:res_flex_ladder` removed |
| Figure 2 | — (target removed: `fig:search_generic`) | 125, 126, 288 | |
| Figure 3 | 2 | 102, 103 | |
| Figure 4 | E.19 | 288 | lattice moved to Appendix E |
| Figure 7 | 5 | 263, 265 | |
| Eqs. (29)–(35) | (29)–(37) | 194 | (30) and (35) inserted inside the span |
| Eqs. (38)–(39) | (E.1)–(E.2) | 220, 222 | salvage equations moved to Appendix E.1 |
| [44] | [43] | 77, 249 | citation order |
| [43] | [42] | 256 | citation order |

Unchanged and MATCH: Sections 1, 2, 2.1, 2.2, 2.2.1, 2.3, 3, 4, 4.1–4.8 and 5; Appendices A, A.1–A.8 and E; Table 1;
Figure 1; Algorithm 1; Eqs. (13), (20)–(28), (22) and (26)–(28); [7]–[9].

### A4 dangling or re-pointed references and prose mentions

- **main.tex at edd7a63.** Statically there are 0 undefined `\ref`/`\eqref` keys (also 0 at `8f051a2`) and 0
  duplicate labels.
  - No typeset line references `tab:res_flex_ladder` or `fig:search_generic`, and no commented line names them in a
    `\ref`.
  - No "Table/Figure/Section/Eq./Algorithm" word stands before a `\ref` of another kind.
  - **0 literal (typed) numbered pointers** ("Table 12", "Figure 7" and the like) occur outside `\ref`.
  - The prose has no "flexibility ladder" or "flexibility-price ladder". "Ladder(s)" occurs twice, both on l. 2921
    (E.7: "seeded with the single-node ladders of the initial design, including ladder points over the budget"), and
    neither points at the removed table. There is no "shaded region".
- **The generator** would reintroduce one dangling reference: generated l. 169, `\ref{tab:res_flex_ladder}`. See
  Blocked 1.
- **Letter hard-coded numbers that now resolve to a different object** (static, at `edd7a63`):

| printed | now denotes |
|---|---|
| Table 6 | `tab:res_energy_ladder_m2` |
| Table 11 | `tab:res_discount` |
| Tables 12, 14, 15 | nothing |
| Figure 2 | the network diagram |
| Figure 3 | `fig:res_breakeven` |
| Figure 4 | `fig:res_flex_ladder` |
| Figure 7 | `fig:res_q_cycle` |
| Section 2.2.6 | `subsubsec:certification` |
| Section 2.2.7 | nothing |
| Section 2.3.2 | `network_storage` |
| Section 2.3.3 | "Battery Degradation" |
| Section 2.3.4 | `subsubsec:commitment` |
| Section 2.3.5 | nothing |
| Section 3.5 | `sec:case_settings` |
| Section 3.6 | nothing |
| Appendix E.6 | `app:results_gap` |
| Eq. (38) | `eq:commitment_deviation` |
| Eq. (39) | `eq:interface_deviation` |
| Eq. (35), the end of (29)–(35) | `eq:soh_initial` |
| [44] | `boyd_admm_2011` |
| [43] | `eve_mb31_datasheet` |

- **Letter prose mentions of the removed figure.** l. 125: "Figure~2 illustrates the search". l. 288: "an
  illustration of the planning search (Figure~2)". l. 264 "evaluated ladders" is generic.
- **Kept labels whose object's caption or title changed (informational).** There are 25; the list is in the stage MD.
  The lowest similarity is `tab:res_benchmark`, 0.14: "Static no-reverse-flow (NRF) arrangements against coordination"
  became "Operating costs and RES curtailment without shared storage".
- **Content needles (informational).** 26 of W180c's content needles for letter pointers no longer occur anywhere in
  typeset `main.tex`. Most sit under pointers that still MATCH by number, for example l. 58 "the value is affine in
  size" (§4.8), l. 78 "0.9331" (§4.5), l. 81 "The objective is the gross operational cost" (§4), l. 160 "Of the 42
  single-scenario evaluations" (§4.7), and l. 282 "1 (passive) and 8 (price-taker)". Three more needles now sit outside
  the target's region. The list is in the JSON, `pointers.needles_absent_from_document`.

### A5 text whose `\textcolor{blue}` the author dropped (list only)

There are 381 wrappers at `8f051a2` and 399 at `edd7a63` (typeset), or 402 raw.

**Captions, by label** (blue share of the caption argument goes from 1.0 to 0.0):

| label | edd7a63 l. | caption starts |
|---|---|---|
| `fig:investment_lattice` (4 → E.19) | 2800 | "Admissible single-node investment plans for installation in 2025 …" |
| `tab:ageing_calibrations` (6 → 3) | 1314 | "Battery ageing calibrations for the baseline and sensitivity analyses." |
| `tab:res_value_ladder` (7 → 4) | 1377 | "Discounted cost increases for selected storage plans …" |
| `tab:res_energy_ladder_m2` (10 → 6) | 1470 | "Investment margins and incremental gains for 4~h storage at node~7 in 2025, …" (same words, wrapper removed) |
| `tab:res_ageing` (11 → 7) | 1529 | "Investment margins and operating indicators of the reference unit …" |

No numbered heading title lost its wrapper.

**Lines carrying text that was blue at `8f051a2` and is not blue at `edd7a63`** (23 lines; blue share now 0.0 on
every one):

| lines | what | first words |
|---|---|---|
| 1316 | Table 3 caption | "\caption{Battery ageing calibrations for the baseline …" |
| 1379–1380 | Table 4 caption | "\caption{Discounted cost increases for selected storage …" |
| 1404–1408 | Table 4 note | "\textit{Note:} Ratings and capacities are specified per node. …" |
| 1472 | Table 6 caption | "\caption{Investment margins and incremental gains …" |
| 1494–1502 | Table 6 note | "\textit{Note:} Increment denotes the change in investment margin …" |
| 1531–1532 | Table 7 caption | "\caption{Investment margins and operating indicators …" |
| 1565–1567 | Table 7 note | "\textit{Note:} $I=318.0$~k\euro{}. $^{w}$: margin within numerical resolution …" |

The token classes behind these rows are 64 unwrapped (same words) and 114 rewritten (replaced text that had been
blue). No moved text was found unwrapped.

### A6 Appendix E at edd7a63

- **Prediction "bitwise": FAILED.** The generated output occurs 0 times. The block runs over l. 2553–2941 (389 lines,
  against 321 generated).
- **Prediction "verify_appE.py 33 PASS": HELD**, with 0 FAIL.
- The generator output is byte-identical to the committed `appendixE_evaluation_record.tex` and to W181a's output
  (`623ddbbd…`).

| hunk | generated l. | main.tex l. | kind | content |
|---|---|---|---|---|
| 1 insert | after 15 | 2568–2597 | prose | new `\subsection{Terminal Salvage Credit}` `app:salvage_credit` with Eqs. (E.1)–(E.2) |
| 2 replace | 169 | 2751 | E.21 group header | `Tables~\ref{tab:res_flex_ladder} and~\ref{…}` → `Fig.~\ref{fig:res_flex_ladder} and Table~\ref{…}` |
| 3 insert | after 214 | 2797–2807 | prose | lattice paragraph and `fig:investment_lattice` moved into E.4 |
| 4 replace | 248 (1 line) | 2841–2848 (8 lines) | prose | E.5: the passive-arrangement curtailment-coefficient sentence; three-phase paragraph rewritten |
| 5 replace | 315 (1 line) | 2915–2934 (20 lines) | prose | E.7: settings paragraph (from §3.6), search-statistics paragraph, closure-slack paragraph, rewritten certification-statistics paragraph ("… without providing a guaranteed bound …") |
| 6 insert | after 321 | 2941 | blank line | |

The entry diff of E.20–E.24 compares 1,637 entries and finds **0 changed**. It also finds 0 row-set changes, 0 caption
changes, 0 note changes and 1 group-header change (hunk 2).

### Negative controls (all 9 detected)

1. A table injected before Table 4 shifts `tab:res_value_ladder` from 4 to 5.
2. A deleted `\label` is absent.
3. A synthetic letter gives the expected verdicts:
   - Table 4 → D.18
   - Section 3.1 MATCH
   - Table 9 removed
   - Figure 2 removed
   - Section 2.2.7 → 2.2.6
   - Tables 7–15 → 4–11, member removed
4. Letter v9 checked against `8f051a2` itself gives 0 MISMATCH and 0 AMBIGUOUS.
5. An injected `\ref{tab:res_flex_ladder}` is reported undefined, and a typed "Table 12" is reported literal.
6. One caption unwrapped in a copy gives exactly its 10 tokens, and a self-compare gives nothing.
7. A paragraph wrapper removed gives 104 tokens.
8. The same paragraph moved without its wrapper is detected as moved.
9. The generated block is located once in `8f051a2`, and the same block with one byte changed is not located.

## Not confirmed

- **Numbering is static, not compiled.** There is no TeX here and no `.aux` or `.bbl` in the clone. The figure-counter
  behaviour in appendices (E.19) and the citation order (elsarticle-num, citation order) are readings of
  `elsarticle.dtx` and the source.
- **The intended target of each letter pointer is assumed** to be the object that carried the printed number at
  `8f051a2`. This rests on the letter being written against round-7 numbering, which equals `8f051a2` for all 132
  labels. A pointer the author meant otherwise would not be detected.
- **Letter v9 was not compared with the expert's own file.** No `response_to_reviewers_v9.tex` or
  `letter_v8_v9.diff` exists in `manuscript_review/`. The review's path
  `/Users/micaelsimoes/PycharmProjects/shared-resources-planning/` does not exist on this machine.
- **The heading successors for 2.2.6, 2.3.1 and 3.3 are not decided.** I recorded only the evidence: the commented
  line, the shingle hints and the contained labels.
- **The A5 classes are a token heuristic.** Text that was rewritten and also moved would appear only through the
  caption-by-label check. 1,083 new non-blue tokens on 213 lines are listed in the JSON (`blue.new_text_not_blue_lines`)
  but not rated, since they are not A5.
- **Not scanned:** `main_latexdiff.tex`, `highlights.tex` and `cover_letter.tex`. They are outside A3–A6; A8 belongs
  to W182d.
