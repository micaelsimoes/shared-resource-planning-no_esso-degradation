# P5.15 W190a: round-9c audit at Overleaf 3c379fa (Addendum 91 ruling 4, rulings 2 and 5, W190a line)

**Worker, 2026-11-12.** This was a static reading with zero solves, zero model builds and zero pickle loads. Nothing was
compiled, because there is no TeX on this machine. All numbering below is static (the W182c method, as in W189 part 2).
The clone, `manuscript_review/`, production, the earlier harnesses, `TASKS.md`, `REVISION_CONTEXT.md` and the brief were
read and not edited. Nothing was written under `data/`.

**Objective convention.** No objective value is computed here. Each rated money figure carries its own convention in the
JSON. Unless a row says otherwise, that convention is Q = `gross_operational_cost` with the contracted interface
settlement excluded.

## Blocked on Planner

None. No stop condition was met, and every prediction held except the two points below. Neither is a manuscript fault.
They go to the Planner or expert for the record:

1. **The expert's "215/215 lines" was not reproduced as a line count.** The gate itself holds.
   - `gen_appE.py --reduced` prints **218** lines, sha256 `8e61636b…`.
   - That block occurs **exactly once, bitwise**, in the round-9c `main.tex`, at l. 2104–2321. It is the whole appendix
     block from the `% ====` rule to the line before `\section*{Declaration…}`, and every line is equal (218/218).
   - Other ways of counting give: 194 non-blank lines, 183 non-blank non-comment lines, 216 without the two trailing blank
     lines, and 11 comment lines. None of them is 215. I could not find which count the expert used.
2. **The W189 blue rule still lists l. 1685 and l. 1695 against a87e195. This is an artefact of the rule; the wrappers
   are present.**
   - Each line is listed for one token, `\multicolumn{4}{l}{(a)` (resp. `{(b)`). The token rule counts the
     non-typeset `\multicolumn` arguments `4` and `l` as unwrapped content.
   - Parsing the arguments shows that the typeset third argument is exactly one `\textcolor{blue}{…}` wrapper on both
     lines.
   - Two negative controls of that argument check were refused: the 9b heading with no wrapper, and a half-wrapped
     heading.
   - I did not change the W189 rule. It is the reason "all five now blue" rests on the argument check for these two lines.

## Changed

- **`ca71db16`** adds the two harnesses, committed **before** the real runs. Both runs record `committed_clean: true` at
  HEAD `ca71db16`.
  - `p515_s55_w190a_numbers.py` (sha256 `ffe6c0c9…`) imports W189 part 1 `p515_s55_w189_p1_numbers.py` unedited. Its
    `main()` is never called. Its task functions are called with the module's revision constants re-bound to this round.
  - `p515_s55_w190a_structure.py` (sha256 `c6c0d21f…`) imports W189 part 2 `p515_s55_w189_p2_structure.py` unedited, in
    the same way.
  - Through those two harnesses, both import the W183h chain unedited (W182c, W182b, W181a and the rest).
- **This commit** adds `P5_15_W190A_REPORT.md`, the outputs under `w190a_audit/`, and the hash inventory
  `w190a_audit/W190A_SHA256_INVENTORY.txt`.
  - **Output location.** The task forbids writing under `data/` while W187 runs, so the outputs live in a new top-level
    directory, `w190a_audit/`. Both capture checklists refuse an output directory under `data/` or `manuscript_review/`.
  - Each output directory was created fresh with `mkdir`, and every file was opened `'x'`.
  - No W189 output was re-run or written.
- **Commands.** All ran from the repo root with the canonical interpreter, attached, with both streams captured. `-B` was
  used so that no `.pyc` was written.
  ```
  mkdir -p w190a_audit && mkdir w190a_audit/numbers && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_numbers.py > w190a_audit/numbers/launch.log 2>&1   # exit 0
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_numbers.py --post-run                               # exit 0
  mkdir w190a_audit/structure && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_structure.py > w190a_audit/structure/launch.log 2>&1  # exit 0
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_structure.py --post-run                              # exit 0
  ```
- **Results.** 0 check failures in both runs.
  - Part 1: 7 of 7 negative controls refused, plus 51 of 51 perturbed tokens.
  - Part 2: 19 of 19 negative controls refused.
- **Guards.**
  - `SolveProfileGuard(permitted=())` was installed before any project import. It and all 58 distinct guards of each
    imported chain verified at exactly 0.
  - Pickle load / loads / Unpickler were blocked for the whole run; every counter was 0.
  - No production module was imported (part 2 asserts this).
  - JSON was written only through `gate_result_io`.
  - `verify_appE.py` itself launches `gen_appE.py` twice as a subprocess for its reduced-mode checks, outside the guard.
    By `ast`, `gen_appE.py` imports only csv, os, re and sys.
- **Trial runs.** All trials wrote to session-scratchpad directories outside the repo: four of part 1 and two of part 2.
  They exposed three faults of mine, each fixed before the commit:
  - a `KeyError` on the highlights text;
  - a wrong source for the cover letter's "two robustness checks". It counted three `part*` keys of the A89 JSON, one of
    which is `part3_code_heads`, so trial 1 showed 1 MISMATCH. It now counts the two experiment parts, `part1_w184_…` and
    `part2_w185_…`;
  - the `\multicolumn` reading above.
- **Repo state at the runs.** HEAD `ca71db16`. The only modified tracked file was `TASKS.md`, which is the Planner's and
  was not staged.
- **W187.** The campaign was running throughout (launcher PID 31351). No `p515_s4*` file was run. Before each commit I
  checked that `.git/index.lock` was absent.

## Found: each prediction against its outcome

| | prediction | outcome |
|---|---|---|
| (a) | four files match; clone diff = expert diffs hunk for hunk; nothing else changed | **held** |
| (b) | `--reduced` output = pasted Appendix B, bitwise, 215/215; full mode = `appendixE_evaluation_record.tex` | **held bitwise, 218/218** (215 not reproduced; Blocked 1); full mode **held** |
| (c) | `verify_appE.py` 40 PASS (33 + 7) | **held**: 40 PASS, 0 FAIL |
| (d) | numbers 0 MISMATCH; ruling-4 (i)–(iv) and cover letter v4 hold | **held**: 0 MISMATCH, 0 UNSOURCED; every check holds |
| (e) | letter v13 0 MISMATCH; W189's one MISMATCH is l. 285 `Fig.~B.8`, now `Fig.~B.9` | **held**: 172 MATCH, 8 SUBMITTED, 0 MISMATCH; confirmed |
| (f) | 0 `[CONFIRM`, 0 prefixes, 0 undefined refs, 0 duplicates; blue on every changed line; counts against W189; numbering = 9b | **held** (blue: see Blocked 2; words changed +20 / +19) |

### (a) Identity: held

- **Clone.** HEAD is `3c379fa44f3755dbdc6fd540df566b55527d921c`, equal to `origin/main` (read locally; nothing was
  fetched). The tracked tree is clean; the only untracked entry is `.DS_Store`.
- **History.** HEAD's parent is `52a607d`, whose parent is `46c9d77`.
  - **`52a607d` is an empty commit:** its tree `bba62cee…` equals `46c9d77`'s, so `46c9d77..52a607d` changes no file.
  - `46c9d77..3c379fa` changes exactly `main.tex`, `response_to_reviewers_draft.tex` and `cover_letter.tex`.
  - **No other file changed.** `fig_soc_soh_reference_unit.pdf` is unchanged since `46c9d77` and equals the review copy.

| file | md5 | = review copy | 46c9d77 blob = 9b deliverable | hunks clone / expert | ± lines | ranges / context equal | expert diff applied to 46c9d77 = HEAD |
|---|---|---|---|---|---|---|---|
| `main.tex` | `cd1433d7…` | `main_46c9d77_round9c.tex` ✓ | `main_a87e195_round9b.tex` ✓ | 8 / 8 | +9 / −9 | ✓ / ✓ | ✓ |
| letter | `74e88c78…` | `response_to_reviewers_v13.tex` ✓ | v12 ✓ | 2 / 2 | +2 / −2 | ✓ / ✓ | ✓ |
| cover letter | `63b7ddee…` | `cover_letter_v4.tex` ✓ | v3 ✓ | 2 / 2 | +4 / −2 | ✓ / ✓ | ✓ |
| `highlights.tex` | `4cedd45c…` | `highlights_v3.tex` ✓ | v3 ✓ | 0 / — | — | — | unchanged |

- Every hunk's '+' and '−' lines are equal, and so are its ranges and context lines (the W181a `compare_hunks` test). This
  round has no placement difference like W189's hunk 23.
- Negative controls: one '+' line of each expert diff was altered in memory. It was flagged at that hunk, and the patched
  result no longer equals HEAD. All three were refused.

### (b) Generator gate: held bitwise (218 lines, not 215)

- **Generator.** `gen_appE.py` is at `441722c9` (sha256 `8a5f60d7…`). It ran in-process (runpy, as W181a
  `run_generator`) with `--reduced`, and printed `ok 58 61 reduced`.
- **Located once.** The output, `appE_generated_reduced_w190a.tex` (218 lines, sha256 `8e61636b…`), occurs **once** in
  the 9c `main.tex`. It starts at a line start and ends at a line end, at l. 2104–2321.
  - That span is the whole appendix block (W181a `appendix_block`); no block line lies outside the generated text.
  - Lines equal: **218/218**.
  - It occurs 0 times in the 9b `main.tex`, as expected: the 9b B.4 differs.
- **Full mode.** The output `appE_generated_full_w190a.tex` (sha256 `66bd3dd0…`) is **bitwise equal** to
  `manuscript_review/appendixE_evaluation_record.tex`. It occurs 0 times in the 9c `main.tex`, also by construction.
- **The verifier's own regenerations** (its `mkdtemp()` redirected into `w190a_audit/numbers/verify_appE_tmp/`):
  - `appE_check.tex` and `appE_full_check.tex` both equal the full output;
  - `appE_reduced_check.tex` equals the reduced output.
- **Entry check of the printed tables at 9c.** This is the W180a/W181a method via W189 part 1 `task_appendix`, against
  the frozen JSON `590088fe`:
  - 657 entries, **0 MISMATCH**, 0 non-frozen disagreements, 0 mapping problems;
  - E.20 has 58 rows and E.22 has 8, both cell-identical to the generator and to a87e195;
  - the method control on a87e195 also gives 0.
- Negative control: the reduced block with one byte changed is not located. Refused.

### (c) Verifier: held, 40 PASS, 0 FAIL

`verify_appE.py` (`441722c9`, sha256 `27a29d0e…`) gave **40 PASS, 0 FAIL**, with no exception.
- 33 full-mode checks, as at W189.
- 7 reduced-mode checks:
  - E.3/E.5/E.6 absent;
  - B.1–B.4 present;
  - no reference to a moved table;
  - S4–S6 pointers present;
  - kept labels once;
  - 58 comparison rows;
  - every kept full-text line present (0 missing).

The verifier still does not read `main.tex`. It is (b) that links the reduced output to the manuscript.

### (d) Numbers: held, 0 MISMATCH

- **Method.** W182b, through W189 part 1 `task_numbers` (the same tokenizer, spec kinds, evaluator and perturbation). The
  scope is every '+' line of `git diff -U0 46c9d77 3c379fa`.
  - The 9b→9c diff is line-preserving: every hunk replaces k lines by k lines, which was checked. So each carried line
    reuses its **W189p1 declaration at the same line number**, re-evaluated here.
  - New specs were written only for the new tokens.

| file | '+' lines | tokens | MATCH | DERIVED | MISMATCH | UNSOURCED | NOT RATED |
|---|---|---:|---:|---:|---:|---:|---:|
| main.tex | 1250, 1353, 1565, 1627, 1685, 1695, 1708, 1710, 2310 | 50 | 41 | 3 | **0** | **0** | 6 |
| letter | 1 (comment), 285 | 43 | 0 | 0 | **0** | **0** | 43 |
| cover letter | 13, 21, 22, 23 | 12 | 5 | 2 | **0** | **0** | 5 |

- **NOT RATED:**
  - main.tex: 3 identifiers (S1, S5 ×2), 2 table-structure tokens and 1 language word;
  - letter: all on the l. 1 LaTeX comment (version history, not typeset). l. 285 has no number tokens; its pointer is
    part 2's;
  - cover letter: the manuscript-number parts 26 and 06323, W187, the pointer 4.3 and "one".
- **Negative control:** all 51 rated tokens were perturbed, and **51/51 were refused**.
- **Tokens new since 9b** (18; the full table with every source is in `w190a_numbers.md`):

| place | token | rating | source record |
|---|---|---|---|
| main l. 1565 (Fig. 6 prose) | six | MATCH | W188 (d) `soc_block_summary_reference_unit.csv`: 6 of 8 (year, day) blocks with `n_contact_lower_tso` > 0 |
| main l. 1565 | eight | MATCH | W188 (d): 8 blocks (2 years × 4 days) |
| main l. 1565 | 14 | MATCH | W188 (d): 2025 Winter min SoC 0.12219 / E^Av 0.88046 = 13.88 % |
| main l. 1565 | 15 | MATCH | W188 (d): 2035 Winter min SoC 0.10783 / E^Av 0.70000 = 15.40 % |
| cover l. 13 | 26, 06323 | NOT RATED (identifier) | EST-D-26-06323, string-checked below |
| cover l. 21 | two (checks) | MATCH | A89 claims JSON experiment parts `part1_w184_section_4_4`, `part2_w185_spread_table` |
| cover l. 21 | 20.5 | MATCH | A89: Q(C_NRF) − Q(C_OPEN) = 20,463.88 € |
| cover l. 21 | 0.02 | MATCH | A89: share of gap 0.0225 % |
| cover l. 21 | 90.9 | DERIVED | frozen `benchmark.benefit` 90,896,608.40 € |
| cover l. 21 | 12.2 | DERIVED | frozen benefit / price-taker Q (W182b `der_pct_price_taker`) |
| cover l. 21 | two (comparisons) | MATCH | A89 part2: prices {1×, 2×} |
| cover l. 21 | W187, 4.3 | NOT RATED | identifier; pointer (static §4.3 = "Effect of Battery Degradation") |
| cover l. 23 | one | NOT RATED (language) | "a prompt one" |
| letter l. 1 | v13, 2026-11-12, 9c | NOT RATED | LaTeX comment |

- The carried tokens on the changed lines also re-evaluate as MATCH or DERIVED:
  - 2.50 and 2.00 (l. 1250);
  - three (l. 1353);
  - the 14 carried Figure 6 tokens, with 90 and 10 in their new order;
  - the 11 tokens of the NRF sentence (l. 1627);
  - the Table 9 tokens (l. 1685, 1695, 1710);
  - the seven B.4 numbers (l. 2310), including 171 as the row-total bound rounded up.

**Ruling 4 (i)–(iv): all hold**

- **(i) B.4 (l. 2310).**
  - The sentence reads "every penalised slack in $Q$ (daily closure and network feasibility) … and the agent's net-power
    pair, penalised in the agent's own objective, is at its bound as well".
  - The only "in Q (…)" parenthetical is "daily closure and network feasibility"; it does not name the net-power pair.
    Nothing places the pair "in Q".
  - The ruling-3 parts are still present.
  - Negative control: the 9b parenthetical, planted back, is detected.
- **(ii) Figure 6 prose (l. 1565).** The text reads "reaching the 90 % limit of its available energy on every day and the
  10 % limit on six of the eight (the Winter days stop at 14--15 %)". The old "between the 10 % and 90 % limits" is gone.
  Against the W188 (d) CSVs (contact rule: within 1e-4 MWh; TSO copy, as in W189 part 1):
  - upper-limit contacts on **8/8** days;
  - lower-limit contacts on **6/8**, the two without being 2025 Winter and 2035 Winter;
  - Winter minima of 13.9 % and 15.4 % of E^Av, which round to 14 and 15;
  - "once or twice a day across the usable band": charged energy over the band is 1.03–1.95 on every day.
- **(iii) Evaluation 16.** The text reads "a positive slack content of at most 171 €, almost all of it voltage slack".
  - W188 (a) `pb_y2025_n5`: row total **170.69 €** ≤ 171 and ceil = 171, made up of voltage **169.27 €** and closure
    1.42 €. The voltage share is **99.17 %**.
  - The old "voltage-slack content of at most 171" is gone.
- **(iv) "Section S5" → "Table S5".**
  - 9b had "Section~S5" at exactly l. 1353 and l. 1627; both now read "Table~S5".
  - **0 "Section S5" remain** in main.tex, the letter, the cover letter and highlights.
  - "Table~S5" occurs at l. 1353, 1627 and 2120. l. 2120 is the generated Appendix B opening, unchanged.
  - Negative control: a planted "Section~S5" is counted.
- **(v) Blue** is checked in part 2; see (f).

**Cover letter v4: all hold.**
- **The filled paragraph (l. 21)** reads:
  - "the restriction costs 20.5 k€, 0.02 % of the 90.9 M€ saving, so the reported 12.2 % benefit does not depend on the
    interface rule";
  - "halved and doubled: every sign and verdict is unchanged";
  - "about a tenth of their determinacy threshold".
- **Against the A89 claims JSON:**
  - `sign_kept` and `class_kept` are true at both prices;
  - the 8 cells are certified at ρ₀ scale {0.5, 2.0};
  - the baseline spread over threshold is **0.1205 (1×) and 0.1184 (2×)**, which is 0.1 at one significant figure.
  - "a tenth" is **not a token** of the W182b/W174 tokenizer, so it is checked as a statement. The main text and the
    letter print "about 0.12".
- **Against the reports as printed** (each equal to A89 at 2 dp):
  - W184: +20,463.88 €; threshold 12,627.51 €; "determinate, positive, 1.62×"; 90,876,144.52 €.
  - W185: the 4-row pair table (margin, threshold, verdict) and both baselines.
  - From W185's printed table, the spread over the baseline threshold is 1,572.87 / 13,054.22 = 0.120 and
    1,452.62 / 12,272.19 = 0.118. Every verdict is "determinate", negative at 1× and positive at 2×.
- **Against the cell records** (`campaign_results.json` `outcome.cell_report.Q_end`):
  - E3 `e3_cnrf_x0_1x` Q = 653,894,166.0688057, equal bitwise to A89 C_NRF;
  - all 8 E2 cells are equal bitwise to A89 `new_cells`.
- **Exactly one bracket left** (l. 21): "[If W187 landed: A degradation-neutral evaluation of the unit neighbourhood of
  the plan without storage is also reported (Section~4.3).]". This was counted after stripping LaTeX optional arguments.
  A planted second bracket is counted (control).
- **Manuscript number.** EST-D-26-06323 is printed once (l. 13), and `JES-D-[number]` is gone. The same number is in
  Addendum 91 and in the brief.
- **Delay paragraph (l. 23).** It is present once, with **two sentences**: "We apologise for the delay in returning this
  revision." and one giving the reason. Ruling 4 calls it "one sentence on the delay". This is recorded, not rated.

### (e) Letter v13 pointers: held, 0 MISMATCH

- **Method.** The W189 part 2 chain, unchanged, on the static context a87e195 → 3c379fa:
  - letter v10 at a87e195 is re-derived: 167 MATCH, 5 SUBMITTED, equal to the committed W183h JSON on all 172 pointers;
  - it is re-rated at HEAD and paired to the clone's letter, which is v13.
  - W189's 16 declared targets and 3 declared SUBMITTED readings apply unchanged: the v13 line structure is v12's. All
    16 declaration checks hold, and 0 are stale.
- **Result: 180 pointers, 172 MATCH, 8 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS.**
  - The 8 SUBMITTED are W189's: l. 100, 102, 141, 230 ×2, 282 ×2 and 283.
  - "Table~7" (l. 170, R2.5/R2.6) is MATCH against `tab:res_benchmark`.
- **W189's one MISMATCH, named from the committed W189 P2 JSON** (`w189_p2_structure.json`, sha256 `1d007156…`). It is
  **letter v12 l. 285, `Fig.~B.8)`**:
  - kind figure, target `fig:investment_lattice`, expected B.9;
  - v10 counterpart l. 288 "Fig.~E.19".
  - It is the pointer the Planner named, so ruling 5's "if not, report it with its line" does not arise.
- **v13 l. 285** now prints "… a figure of the investment lattice of the case study was added (Fig.~B.9)". Rated
  **MATCH** (expected B.9, target `fig:investment_lattice`). The MISMATCH is closed.
- **Controls, all refused:**
  - W189's three (l. 170 Table 7 → 8; a paired Section 4.7 → 4.8; a declared A.4 → A.5);
  - v13 l. 285 `Fig.~B.9` put back to `Fig.~B.8` in a temp copy is rated MISMATCH.

### (f) Structure: held

- **`[CONFIRM`.** 0 in each of the four files.
- **Tightened prefix `\b(Appendix|Section)~\\ref\{app:`.**
  - **0** in main.tex, both raw and typeset, and 0 in the letter, the cover letter and highlights.
  - The general `[A-Za-z]+~\ref{app:` pattern matches 11 times, all prepositions ("in" 9, "of" 2), as at 9b.
- **References.**
  - **0 undefined** `\ref`/`\eqref` keys (151 uses, 62 keys) and **0 duplicate labels**. The **label count is 102**.
  - 0 uses of the 35 labels removed since a87e195, typeset or in comments.
  - Supplementary pointers lie within S1–S6.
- **Numbering, 9b vs 9c: identical.**
  - The static object sequence is equal (126 objects).
  - All 102 labels are unchanged: 0 added, 0 removed, 0 renumbered.
  - Every label number equals the committed W189 P2 number at 46c9d77.
  - The heading map shows 43 of 43 unchanged.
  - Citation order is identical, unlabelled equations are identical, and the counter adjustments are equal.
  - This matches the expert's compile claim "numbering identical to 9b" statically.
  - Against W189's stated map (a87e195 → 9c), the only disagreement is still `fig:investment_lattice` B.9 against
    STEP6's B.8. That is W189's finding about STEP6's table, not the letter.
- **Blue.**
  - **Every line changed 9b→9c carries a wrapper.** l. 1250, 1353, 1565, 1627, 1685, 1695, 1708, 1710 and 2310 have 0
    changed tokens outside a wrapper. The W189 rule on 9b→9c lists 0 lines.
  - **W189's five lines, at the same line numbers** (each text is unchanged apart from its wrapper):

| line | content blue at 9b | at 9c | W189 rule vs a87e195 at 9c |
|---|---|---|---|
| 1250 (§3.2) | 0 | 0.17 (the changed sentence wrapped; the rest is original, unchanged text) | not listed |
| 1685 (Table 9 (a) heading) | 0 | 0.83; the third `\multicolumn` argument is one blue wrapper | listed (artefact; Blocked 2) |
| 1695 (Table 9 (b) heading) | 0 | 0.83; as above | listed (artefact; Blocked 2) |
| 1709 (note (a)) | 0 | 1.0 (inside the wrapper opened at l. 1708) | not listed |
| 1710 (note (b)) | 0 | 1.0 | not listed |

  - Every caption changed since a87e195 is fully blue.
  - The caption-unwrap control was built on a87e195 → 9c and was refused. On 9b → 9c no caption changed, so P2's control
    there could not be built. That one P2 message is declared not applicable in the JSON, not counted as a failure.
- **Counts** (W189 counting rule; the rule reproduces W189's committed counts on 46c9d77 exactly):

| region | W189 (46c9d77) | 9c (3c379fa) | change |
|---|---:|---:|---|
| main-text floats (tables + figures) | 17 (9 + 8) | 17 (9 + 8) | none; Algorithm 1 apart |
| main-text numbered equations | (1)–(41), 42 rows | (1)–(41), 42 rows | none |
| main-text prose words | 6,692 | **6,712** | **+20** (Figure 6 sentence) |
| Appendix A prose words | 2,605 | 2,605 | none |
| Appendix B prose words | 1,550 | **1,569** | **+19** (B.4 sentence) |
| caption / heading words, source lines | 491 / 95 / 1,446 | 491 / 95 / 1,446 | none |
| appendices | 2 | 2 | none |

## Not confirmed

- **Compilation, pages and printed numbers.** Nothing was compiled: pdflatex, latexmk and tectonic are not on PATH. The
  expert's "0 / 0 / 0, 63 pages" and the compiled figure number B.9 are not reproduced; the static count agrees with B.9.
  - I searched `manuscript/` and `manuscript_review/` (depth 3) for `*.aux`, `*.log`, `*round9c*`, `main*.pdf` and
    `*preview*.pdf`.
  - No round-9c compile output exists there. The previews found are older ones (section 2 and appendix E), and the
    `.log` files found are the supplement generator's.
- **The remote.** `origin/main` = HEAD was read from the clone's local refs. Nothing was fetched.
- **The expert's 215.** See Blocked 1. I tried the line-count variants listed there and nothing else.
- **Carried registries.** As in W189 part 1, some carried MATCH tokens rest on committed records that were not recomputed
  here: the frozen JSON and the W163 registry.
- **The supplement.** Table S5's content (solver starts versus "evaluation procedure and voltage-consistency checks", the
  phrase at l. 1353) was not checked. That is W189a / supplement v3 scope; I did not read the supplement zip.
- **The repository-wide bool-typing test** (`p515_gate_result_bool_typing_test.py`) was not run. It scans `data/`, where
  W187 is writing now, and W190a's JSON is outside `data/`. The JSON went through `gate_result_io`, which refuses string
  flags at write.
- **The W187 bracket's pointer.** "Section~4.3" is static §4.3 "Effect of Battery Degradation". Whether the W187 sentence
  belongs there is round 10's question.
