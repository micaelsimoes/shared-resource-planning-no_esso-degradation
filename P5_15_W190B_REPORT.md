# P5.15 W190b: interim audit at Overleaf 5f4d3bd (Addendum 92, W190b (i)–(vii))

**Worker, 2026-11-13.** Static reading, zero solves, zero model builds, zero pickle loads; nothing compiled (no TeX on
this machine). The clone was fetched and fast-forwarded once, before the runs; nothing else in the clone, nothing under
`manuscript_review/`, `data/`, production, harnesses, launchers or hooks was written. `TASKS.md` (modified by the Planner)
was not staged.

**Objective convention.** (i) uses F = I(x) + Q(x), Q = `gross_operational_cost` with the contracted interface settlement
excluded, salvage excluded from F (Addendum 27 item 3); ΔF = F(neighbour) − F(x = 0). Every other rated money token carries
its own convention in `w190b_numbers.json` (gross, settlement excluded, unless stated).

## Blocked on Planner

None. Every predicted count held: W187 numbers, 0 MISMATCH, letter 0 MISMATCH / 8 SUBMITTED, blue on every changed line,
Appendix B bitwise and 40 PASS, 0 CONFIRM / 0 prefixes / 0 hard-coded references. The findings below are wording,
cross-reference and typography items for the author's next Overleaf edit. Nothing is a stop (Addendum 92).

## Changed

- **HEAD audited: `5f4d3bd4c96641c40a42829ae837fd954ec04954`** (committed 2026-11-13 11:04:59 UTC). Fetched and
  fast-forwarded at **11:07:34 UTC** ("Already up to date": the clone was already at `origin/main`). Re-fetched at
  11:24:29 UTC and again before the evidence commit: `origin/main` still `5f4d3bd`. The tracked tree is clean; the only
  untracked entry is `.DS_Store`.
- **`05d86627`**: `p515_s55_w190b_numbers.py` (sha256 `4b157e36…`) and `p515_s55_w190b_structure.py` (`13bf8872…`),
  committed **before** their real runs. Both runs record `committed_clean: true` at repo HEAD `05d86627`.
  - They import the W190a harnesses unedited, `p515_s55_w190a_numbers.py` and `p515_s55_w190a_structure.py`, and through
    them W189 P1 / P2, also unedited. Module constants are re-bound: 3c379fa → 5f4d3bd, and a87e195 for the generator
    integrity check, the pointer chain and the blue rule. The `main()` functions of those harnesses are never called.
  - New code (no earlier harness does this):
    - the W187 ΔF recomputation from the 15 cell `campaign_results.json` files;
    - the formatting scan of the changed lines;
    - the letter declarations re-keyed to their HEAD lines;
    - the counts compared against the committed W190a counts.
- **This commit**: this report, the outputs under `w190b_audit/` (16 files), and the sha256 inventory
  `w190b_audit/W190B_SHA256_INVENTORY.txt`.
- **Commands** (repo root, canonical interpreter, attached, both streams captured):
  ```
  mkdir -p w190b_audit && mkdir w190b_audit/numbers && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_numbers.py > w190b_audit/numbers/launch.log 2>&1      # exit 0
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_numbers.py --post-run                                 # exit 0
  mkdir w190b_audit/structure && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_structure.py > w190b_audit/structure/launch.log 2>&1  # exit 0
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_structure.py --post-run                              # exit 0
  ```
- **Guards.** `SolveProfileGuard(permitted=())` was installed before any project import. It and every guard of both
  imported chains verified at exactly 0. Pickle load / loads / Unpickler were blocked; every counter is 0. Check failures:
  0 in both runs. The one P2 message "no blue changed caption to build the unwrap control" applies to ctx 3c379fa → HEAD,
  where no caption changed. It is declared not applicable, as in W190a, and the same control is built and refused on
  a87e195 → HEAD.
- **Negative controls.** All 23 were refused: 7 in part 1 (including 44 of 44 perturbed tokens) and 16 in part 2.
- **Trial runs**, all into the session scratchpad: two of part 1 and three of part 2. They exposed four faults of mine,
  each fixed before the commit:
  - a too-loose "4.1 states determinacy" test, which matched the table note;
  - the v10 cross-check applied to the two new `Section~4.3` pointers (l. 263–264), which at 3c379fa read `Section~4.7`;
  - tuple JSON keys;
  - a `\b25\b` pattern that matched "25.6 s".

### Diff 3c379fa..5f4d3bd, hunk by hunk (no review copy this round)

There are seven commits: `275398e`, `72de6d7`, `db5e871`, `3da93d9`, `44fe169`, `49f8836` and `5f4d3bd`. They change
`main.tex`, the letter and the cover letter, and no other file.
- `44fe169` placed the W187 paragraph at l. 1562–1565.
- `49f8836` moved it to l. 1508–1511.
- `5f4d3bd` touched letter l. 263 only.

| file | hunk | 3c379fa l. | HEAD l. | change |
|---|---|---|---|---|
| main.tex | 1 | 146 | 146–147 | Abstract split into two lines after "transmission." The text is unchanged (0 changed tokens), with a trailing space on l. 146. |
| main.tex | 2 | 387–390 | 388–391 | The four contributions rewritten (author). Each is inside `\textcolor{blue}`. |
| main.tex | 3 | 393–394 | — | A commented-out paragraph (`%The results show …`) and a blank line removed. Not typeset. |
| main.tex | 4 | 1375–1377 | 1374–1375 | §4.1 paragraph rewritten (author), 3 lines → 2. Still inside the `\textcolor{blue}{` block opened at l. 1373. |
| main.tex | 5 | — | 1508–1511 | New W187 paragraph in §4.3 (`\textcolor{blue}{` … `}` and a blank line). |
| letter | 1 | 58 | 58–60 | "What we did not do" split into three lines. The degradation-neutral sentence is rewritten around the screen ("… which returns the same plan (Section~4.3); a full degradation-neutral search over the lattice was not run …"). The submitted-instance sentence is unchanged. |
| letter | 2 | 261 | 263–265 | R3.6 split into three lines. The selected-plan clause now reads "… and the unit-neighbourhood screen its effect on the selected plan (Section~4.3)" (it was "(Section~4.7)"). Two screen sentences were added; the floor and capacity sentences are unchanged. |
| cover letter | 1 | 21 | 21–22 | The `[If W187 landed: …]` bracket is replaced by "A degradation-neutral evaluation of the unit neighbourhood of the plan without storage is also reported (Section~4.3): without ageing the search would return the same plan." |

The full ± text of every hunk is in `w190b_numbers.md` / `.json` (`state.hunks_3c379fa_head`, `state.per_commit`).

## Found

| | prediction | outcome |
|---|---|---|
| (i) | W187 numbers in §4.3, letter l. 58ff / 261ff and cover letter = the report and `campaign_results.json`: fourteen; 8 / 6 under max{bar sum, σ_Q}; 2.7 / 100.6 k€; "returns the same plan"; salvage sensitivity and the 59.9 k€ bar not printed | **held**: every number and the printed form; 0 not-printed items printed. Finding 1 (the 4.3 cross-reference) |
| (ii) | numbers of the main text 0 MISMATCH; every new token sourced; §4.1 with `\euro{}`, ties, `\ref` | **held, 0 MISMATCH** (58 tokens, 0 UNSOURCED; tables 657 entries 0 MISMATCH). §4.1: 0 literal €, 0 hard-coded references, **3 missing ties** (finding 4); two claim findings (2, 3) |
| (iii) | letter pointers 0 MISMATCH, 8 SUBMITTED | **held**: 182 pointers, 174 MATCH, 8 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS |
| (iv) | blue on every line changed a87e195 → HEAD | **held**: only l. 1687 / 1697 listed (W190a's `\multicolumn` artefact, now +2 lines); 0 unexplained |
| (v) | Appendix B = `gen_appE.py --reduced` bitwise (218 lines, `8e61636b`); `verify_appE.py` 40 PASS | **held**: 218/218, sha256 `8e61636b…`, once at l. 2106–2323; 40 PASS 0 FAIL |
| (vi) | 0 `[CONFIRM`, 0 `(Appendix\|Section)~\ref{app:`, 0 hard-coded refs with a label; 0 undefined, 0 duplicate | **held**: all 0; **102 labels**; numbering identical to 3c379fa |
| (vii) | 17 floats, 41 equations, 6,712 words (W190a) | 17 floats, 41 equations held; **main words 6,768 (+56)**; App. A 2,605, App. B 1,569, abstract 246 unchanged |

### (i) W187: held

The screen comprises the 14 neighbours of x = 0 under no ageing at 1×, plus x = 0 itself (`x0_A_1x`, eval key
`5cfe69a6…`, candidate `8435c718…`). The 15 eval and candidate keys are recorded in
`w190b_numbers.json` → `w187.instance`.

**Recomputation, independent of the W187 JSON's derived fields.**
- Q and bar come from each cell's `campaign_results.json` (`outcome.screen`). All 15 files are committed and clean,
  certified, and bitwise equal to the W187 JSON.
- I(x) is computed as Σ (S · c_P + E · c_E) from the recorded discounted unit costs. It equals the W187 `I_eur` within
  1e-6 € on 15/15 cells.
- ΔF = I(n) + Q(n) − Q(x0). It equals the W187 JSON within 1e-6 € on 14/14.
- T_frozen = max(bar_n + bar_x0, σ_Q = 18,449.66 €). The verdicts equal the W187 JSON on 14/14.

| neighbour (0.25 MVA / 0.5 MWh at) | I (€) | ΔQ (€) | ΔF (€) | T_frozen (€) | ratio | verdict |
|---|---:|---:|---:|---:|---:|---|
| 2025 n9 | 191,018.17 | −179,702.09 | +11,316.08 | 18,449.66 | 0.61 | within |
| 2030 n9 | 141,882.02 | −109,083.95 | +32,798.07 | 18,449.66 | 1.78 | x0 better |
| 2025 n7 | 191,018.17 | −174,715.29 | +16,302.88 | 31,840.29 | 0.51 | within |
| 2030 n7 | 141,882.02 | −100,545.08 | +41,336.94 | 18,449.66 | 2.24 | x0 better |
| 2025 n7+n9 | 382,036.34 | −365,259.33 | +16,777.01 | 25,169.33 | 0.67 | within |
| 2030 n7+n9 | 283,764.04 | −236,142.68 | +47,621.37 | 19,083.78 | 2.50 | x0 better |
| 2025 n5 | 191,018.17 | −185,008.28 | +6,009.89 | 30,661.42 | 0.20 | within |
| 2030 n5 | 141,882.02 | −120,885.50 | +20,996.52 | 18,449.66 | 1.14 | x0 better |
| 2025 n5+n9 | 382,036.34 | −360,972.49 | +21,063.85 | 19,284.67 | 1.09 | x0 better |
| 2030 n5+n9 | 283,764.04 | −186,811.12 | +96,952.92 | 69,555.51 | 1.39 | x0 better |
| 2025 n5+n7 | 382,036.34 | −373,291.34 | +8,745.00 | 21,876.86 | 0.40 | within |
| 2030 n5+n7 | 283,764.04 | −246,747.22 | +37,016.83 | 20,687.08 | 1.79 | x0 better |
| 2025 n5+n7+n9 | 573,054.51 | −570,370.49 | **+2,684.02** | 22,056.04 | 0.12 | within |
| 2030 n5+n7+n9 | 425,646.07 | −325,061.86 | **+100,584.21** | 18,449.66 | 5.45 | x0 better |

- **Counts and range.** ΔF > 0 on 14/14, giving 0 / 8 / 6 (neighbour better / x0 better / within). The range is
  +2,684.02 to +100,584.21 €, which rounds at 1 dp to **2.7 and 100.6 k€**, as printed.
- **Q(x0).** Q(x0) = 653,858,731.5686 € is **bitwise equal to the W86 tail recert**
  (`per_cell.x0.detail.post_certification.certified_cost`).
- **Printed passages.** Every phrase was checked, and all are present:
  - **§4.3 (`main.tex` l. 1509, under `\subsection{Effect of Battery Degradation}`, static §4.3)**:
    - "fourteen admissible unit neighbours";
    - "under the no-ageing calibration with the production stopping rule";
    - "None reduces the total discounted cost";
    - "eight are more expensive beyond the search's acceptance threshold and six lie within it", which is the printed
      form of ruling 2;
    - "cost increases from 2.7 to 100.6~k\euro{}";
    - "the search would also stop at the plan without storage";
    - "cannot discriminate the smallest storage units from no storage";
    - "with ageing, every unit neighbour is determinately worse";
    - the scope sentence "a bounded screen of the unit neighbourhood at the base flexibility price, not a completed
      search".
  - **Letter, "What we did not do" (l. 59)**: "a screen of the fourteen unit neighbours … under the no-ageing
    calibration, which returns the same plan (Section~4.3)" and "a full degradation-neutral search over the lattice was
    not run, and the paper says so (Section~4.7)".
  - **Letter R3.6 (l. 264)**: "the fourteen admissible unit neighbours … as a planning search would evaluate them
    (Section~4.3)", "eight are worse beyond the search's acceptance threshold and six within it", "a degradation-neutral
    search would return the same plan", and "A full degradation-neutral search over the lattice was not run".
  - **Cover letter (l. 22)**: "… is also reported (Section~4.3): without ageing the search would return the same plan".
    The W187 bracket is resolved: 0 brackets are left.
- **Not printed (ruling 2), all held:**
  - "59.9" (and 59,9 / 59{,}9 / 59,925) appears nowhere in main, letter or cover;
  - "salvage" appears on no changed line;
  - the 3b/2τ counts ("five / nine") appear on no changed line.
- **Also not printed:** Q(x0) (no "653.9 / 653,858 / 653{,}858"), W86, and plan §5.3's "not a globally preferred zero
  investment" clause. The last is informational: the printed scope sentence carries "not a completed search".
- **One claim check fails: finding 1.** "with ageing, every unit neighbour is determinately worse" is true per
  Appendix B.3, but its pointer goes to §4.1, which no longer says it.

### (ii) Numbers: 0 MISMATCH

The method is W182b, through W189 P1 `task_numbers`, on every '+' line of `git diff -U0 3c379fa 5f4d3bd`. The lines
unchanged since 3c379fa are the ones W190a / W189 rated, and are byte-identical. The printed tables (E.20 / E.22) were
re-checked against the frozen JSON `590088fe`: **657 entries, 0 MISMATCH**, 0 mapping problems.

| file | '+' lines | tokens | MATCH | DERIVED | MISMATCH | UNSOURCED | NOT RATED |
|---|---|---:|---:|---:|---:|---:|---:|
| main.tex | 146, 147, 388–391, 1374, 1375, 1508–1511 | 26 | 21 | 3 | **0** | **0** | 2 (compound adjectives: "two-stage", "two-node") |
| letter | 58–60, 263–265 | 25 | 13 | 1 | **0** | **0** | 11 (pointer numbers, part 2's) |
| cover letter | 21, 22 | 7 | 4 | 2 | **0** | **0** | 1 (pointer 4.3) |

Negative control: 44/44 perturbed tokens refused.

**Tokens new since 3c379fa** (13, against the '−' lines of their hunk; every other token is carried from the same
hunk's old text):

| place | token | rating | source |
|---|---|---|---|
| main l. 1375 | two ("two-node plans") | NOT RATED | compound adjective; the plans are rows of `tab:res_value_ladder` |
| main l. 1509 | fourteen | MATCH | W187 JSON `lattice.x0_lattice_neighbourhood_size` = 14 = the compared set |
| main l. 1509 | eight | MATCH | ΔF > max(bar_n + bar_x0, σ_Q), recomputed from the 15 `campaign_results.json` |
| main l. 1509 | six | MATCH | the same comparison, within |
| main l. 1509 | 2.7 | MATCH | min ΔF 2,684.02 € (2025 n5+n7+n9) |
| main l. 1509 | 100.6 | MATCH | max ΔF 100,584.21 € (2030 n5+n7+n9) |
| letter l. 59 | fourteen | MATCH | as main l. 1509 |
| letter l. 264 | fourteen, eight, six | MATCH | as main l. 1509 |
| letter l. 59, 263, 264 | 4.3 (×3) | NOT RATED here | pointers, rated MATCH in (iii) |

**Carried tokens re-rated on their new lines:**
- abstract l. 146–147: 108, 15, 12.2, 0.25, 1, 64.4, 1.75, 2, 3, 3;
- contributions l. 391: 108;
- §4.1 l. 1374–1375: fourteen, 0.25, 1, 7, 2025, 253.5, 318.0, 64.4;
- letter l. 58: "Two" (DERIVED: 2 "We did not" sentences in the paragraph);
- letter l. 60: five, twenty-five, 3, 3;
- letter l. 263: 0.70, 0.50, +4.9;
- letter l. 265: 2035, 50;
- cover l. 21: two, 20.5, 0.02, 90.9, 12.2, two.

**§4.1 (l. 1374–1375), formatting:**
- 0 literal € signs;
- 0 hard-coded references (it uses `Table~\ref{tab:res_value_ladder}` and `Subsection~\ref{subsec:res_breakeven}`);
- **3 numbers before `k\euro{}` without a tie** (finding 4).

**§4.1, claims:**
- "saves 253.5 … against … 318.0 …, a shortfall of 64.4" agrees with the records. I − value = 64,417.39 € equals the
  shortfall. The printed 318.0 − 253.5 = 64.5 is a 1-dp rounding effect, and the same three figures stood at 3c379fa.
- "larger units … two-node plans lose more" holds against the table. The 0.5 MVA rows (114.1, 135.5) and the two-node
  rows (102.1, 93.9) are all above 64.4.
- "longer durations … lose more" holds only as 4 h against 2 h at equal converter rating (64.4 > 49.7; 135.5 > 114.1).
  No row is longer than the reference's 4 h.
- Two scope findings: 2 and 3.

### (iii) Letter pointers: held, 0 MISMATCH, 8 SUBMITTED

The W189 P2 chain was used unchanged on a87e195 → 5f4d3bd: letter v10 rated at a87e195, re-rated at HEAD, and paired
through `H.line_map`.

- **Declarations moved.** W189's 16 declared targets and 3 declared SUBMITTED readings were moved to their HEAD lines
  (+2 after l. 58, +4 after l. 261), each through the 3c379fa → HEAD line map. All mapped.
- **Declarations added.** The author's 1 → 3 line splits leave l. 59–60 and the R3.6 block l. 262–266 without a v10
  line, so their 13 pointer keys got **new declared targets** (`NEW_DECLARED` in the script, each with its reason):
  - 4.3 → `subsec:res_degradation`;
  - 4.7 → `subsec:res_certification`;
  - 4.5 → `subsec:res_multiscenario`;
  - Table 6 → `tab:res_ageing`;
  - Figure 5 → `fig:res_ageing`.
- **Cross-checks on the new declarations.**
  - Each content-bearing one has a needle in its target region: "The planning question was also tested directly",
    "without repeating the planning search", "three market scenarios".
  - The others are cross-checked against the v10 counterpart of their 3c379fa line.
  - The two `Section~4.3` pointers of l. 263–264 are **new** pointers: 3c379fa l. 261 printed "(Section~4.7)" there.
    They are checked by needle only.
  - All declaration checks hold, and 0 are stale.
- **Result: 182 pointers, 174 MATCH, 8 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS.**
  - The 8 SUBMITTED are W190a's, now at l. 102, 104, 143, 232 ×2, 286 ×2 and 287.
  - `Fig.~B.9` (l. 289) is MATCH.
- **Controls.** Four controls were all refused: W189's p1–p3, and l. 264 `Section~4.3` → `4.4` with the declaration
  kept, which is rated MISMATCH.

### (iv) Blue: held

- **3c379fa → HEAD.** 0 changed tokens are outside a wrapper on the 12 '+' lines. l. 146–147 (the abstract split) carry
  0 changed tokens. The W189 rule lists 0 lines.
- **a87e195 → HEAD.** The W189 rule lists **l. 1687 and l. 1697** only. These are W190a's l. 1685 / 1695, moved by
  +2 lines.
  - On both, the third `\multicolumn` argument is exactly one `\textcolor{blue}{…}` wrapper, which is W190a's artefact.
  - 0 lines are unexplained, and 0 changed captions lack blue.
  - The two multicolumn controls were refused: l. 1687 unwrapped, and l. 1687 half-wrapped.

### (v) Appendix B: held

- **Reduced output.** `gen_appE.py --reduced` (`441722c9`, unchanged) gives `appE_generated_reduced_w190b.tex`: 218
  lines, sha256 `8e61636b…`.
  - It is **byte-equal to W190a's output**.
  - It is located **once** in HEAD `main.tex`, at l. 2106–2323, which is the whole appendix block. 218/218 lines are
    equal.
  - It is also located once in 3c379fa (l. 2104–2321). The +2 shift comes from the main-text edits.
- **Full mode.** The full output (`66bd3dd0…`) is bitwise equal to `appendixE_evaluation_record.tex`. The verifier's
  own regenerations are equal to it too.
- **Verifier.** `verify_appE.py` gives **40 PASS, 0 FAIL**, with no exception: 33 full-mode checks plus 7 reduced-mode
  checks.

### (vi) Structure: held

- **`[CONFIRM`.** 0 in each of the four files.
- **Prefixes.** `\b(Appendix|Section)~\ref{app:` matches 0 times in `main.tex` (raw and typeset) and 0 in the letter,
  the cover letter and highlights. The general `[A-Za-z]+~\ref{app:` pattern matches 11 times ("in" 9, "of" 2), as at
  W190a.
- **References and labels.**
  - **0 undefined** references (152 `\ref` uses, one more than W190a, namely the new `subsec:res_baseline` in §4.3;
    62 keys).
  - **0 duplicate labels**, and **102 labels**.
  - The label map 3c379fa → HEAD is unchanged on 102 / 102.
  - The static object sequence is equal (126 objects), and every number equals W189's committed numbers. Static §4.3
    is "Effect of Battery Degradation".
- **Hard-coded references in the main text.** **0** typed "Section N.N", "Table N", "Fig. N" or "Eq. (N)": the W182c
  literal-pointer list at HEAD is empty. The complementary lettered scan found 0 typed B.n / A.n pointers.
  - There are 16 typed **supplementary** pointers, S1–S6: 8 in the main text (l. 1249, 1256, 1291, 1292, 1352, 1629,
    1756, 1760) and 8 in Appendix B. No label exists for them in `main.tex` by construction, so they are not findings.

### (vii) Counts

The rule is W189's, and it reproduces the committed W190a counts at 3c379fa exactly.

| region | W190a (3c379fa) | HEAD 5f4d3bd | change |
|---|---:|---:|---|
| main-text floats (tables + figures) | 17 (9 + 8) | 17 (9 + 8) | none; Algorithm 1 apart |
| main-text numbered equations | (1)–(41), 42 rows | (1)–(41), 42 rows | none |
| main-text prose words | 6,712 | **6,768** | **+56** (W187 paragraph; shorter contributions and §4.1) |
| Appendix A prose words | 2,605 | 2,605 | none |
| Appendix B prose words | 1,569 | 1,569 | none |
| abstract prose words | 246 | 246 | none (line split only) |
| caption / heading words | 491 / 95 | 491 / 95 | none |

### Findings for the author's next Overleaf edit

1. **main.tex:1509: a cross-reference that no longer carries its claim.**
   - **Text:** "with ageing, every unit neighbour is determinately worse (Subsection~\ref{subsec:res_baseline})".
   - **Problem:** The claim holds by Appendix B.3 (l. 2263: eight re-evaluated neighbours "determinately more"; the other
     six at "3.47--7.17 times the search resolution"). But §4.1 at HEAD (l. 1374) now says only "none of its fourteen
     admissible unit neighbours reduces the total discounted cost". The only "determinate" in §4.1 is the
     `tab:res_value_ladder` note on its six listed plans.
   - **Fix:** point to `\ref{app:results_neighbours}`, following the paper's "in~\ref{app:…}" convention (never
     "Appendix~\ref{app:", per check (vi)), or restore the determinacy in §4.1 (see 3).
2. **main.tex:1374: the scope of the first sentence.**
   - **Text:** "Under the baseline assumptions, no shared storage investment recovers its cost."
   - **Problem:** This is broader than the abstract (l. 147: "none of the evaluated storage plans recovers its investment
     cost"), §4.7 (l. 1768: "The baseline conclusion applies to the evaluated plans and does not establish global
     optimality") and l. 529 ("it does not establish global optimality").
   - **Fix:** "none of the evaluated storage plans recovers its cost".
3. **main.tex:1374: "beyond the search resolution" and "the result applies to the evaluated neighbourhood" were
   dropped.**
   - **Text:** "none of its fourteen admissible unit neighbours reduces the total discounted cost."
   - **Problem:** Appendix B.3 (l. 2263, generator output, which must not be hand-edited) says the search evaluated the
     fourteen "at its own resolution ($\sigma_Q$, or the sum of the two evaluations' bars where larger;
     Subsection~\ref{subsec:res_baseline})". That points to §4.1, which no longer mentions resolution.
   - **Fix:** "… reduces the total discounted cost beyond the search resolution; each is determinately worse". This also
     settles 1.
4. **main.tex:1375: three missing ties.**
   - **Text:** "saves 253.5 k\euro{} in operation against an investment of 318.0 k\euro{}, a shortfall of 64.4 k\euro{}".
   - **Fix:** `253.5~k\euro{}`, `318.0~k\euro{}`, `64.4~k\euro{}`, as at 3c379fa and everywhere else in the paper.
5. **letter:263: stronger than the main text.**
   - **Text:** "… so the comparison establishes the valuation effect of degradation at a fixed plan, and the
     unit-neighbourhood screen its effect on the selected plan (Section~4.3)."
   - **Problem:** main.tex:1509 calls the screen "a bounded screen of the unit neighbourhood …, not a completed search",
     and letter l. 264 itself says "A full degradation-neutral search over the lattice was not run".
   - **Fix:** "… and the unit-neighbourhood screen bounds its effect on the selected plan within that neighbourhood
     (Section~4.3)". This is the expert's wording call.
6. **letter:264: "what degradation changes at this scale is the margin by which the storage units lose, not the
   selected plan".**
   - **Problem:** Without ageing, 6 of the 14 are within resolution, so their sign is not resolved. main.tex:1509 says
     so ("cannot discriminate the smallest storage units from no storage").
   - **Fix:** "… is whether, and by how much, the storage units lose, not the selected plan". This is the expert's
     wording call.
7. **letter:264: typography.** Two typographic apostrophes (U+2019), in "unit’s" and "search’s". The rest of the
   letter uses `'` (33 occurrences on 21 lines). The file loads `inputenc` utf8 and `T1`, so it compiles.
   - **Fix:** write `'`.
8. **letter:60: pre-existing, unchanged since 3c379fa; the sentence only moved.**
   - **Text:** "The largest instance of the submitted version (five representative years, twenty-five scenarios) could not
     be evaluated … and is stated as future work (Section~4.7)".
   - **Problem:** The pointer number is MATCH. But §4.7 (l. 1723–1776) contains none of "memory", "twenty-five / 25
     scenarios", "five representative years", "future work", "largest" or "submitted", and Conclusions l. 1794 names
     only "larger test systems".
   - **Fix:** add the sentence to §4.7, or reword the letter to "(Section~4.7: limited to the single-scenario and
     3 × 3 configurations)".
9. **cover_letter.tex:26: pre-existing, unchanged since 3c379fa.**
   - **Text:** "Two requests were met only in part … we do not attribute degradation separately … and we report the
     terminal salvage value as a post-evaluation sensitivity …"
   - **Problem:** The letter's "What we did not do" (l. 58–59) names a different second request: re-optimisation under
     every ageing calibration (the degradation-neutral benchmark), now met by the screen at the scale the method
     allows.
   - **Fix:** align the two lists. This is the author's or expert's call.
10. **Harmless, recorded only.** There is trailing whitespace on main.tex:146 and 1374, letter:58 and 263, and
    cover_letter.tex:21. main.tex:393–394 (a commented paragraph) was removed.

## Not confirmed

- **Compiled output.** Nothing was compiled, and no TeX is on PATH. Page count, compiled numbering and the PDF rendering
  of the new paragraph are not checked; the numbering is static (W182c).
- **`PRESENTATION_PASS_PLAN_2026-11-12.md` was not read.** It is not in `manuscript_review/`, and no file of that name
  exists in the repository. I searched with `find . -name "PRESENTATION_PASS*"` outside `.claude/` and `git log --all`,
  and grepped every file outside `.git` / `data` / `.claude`. Only the brief and `ADDENDUM92_2026-11-13.md` name it.
  The audit therefore checks Addendum 92 ruling 4's restriction ("prose only; no number, label, table row, equation or
  Appendix B change") through (ii), (v) and (vi), not against the plan's text.
- **The rewritten contributions' prose claims** are not rated. Examples are "while maintaining institutional
  independence" and "every reported comparison being stated with its numerical resolution". Only their number tokens
  ("two-stage", "108") were checked.
- **Carried registries.** Carried MATCH tokens rest, as in W189 / W190a, on committed records that were not recomputed
  here: the frozen JSON `590088fe`, the W163 registry, the A89 claims JSON and W188.
- **Findings 8–9** come from a pattern scan of §4.7, the Conclusions and the cover letter. "Not stated" is scoped to
  those patterns and those sections.
- **The repository-wide bool-typing test** (`p515_gate_result_bool_typing_test.py`) was not run. W190b's JSON lies
  outside `data/` and was written through `gate_result_io`.
