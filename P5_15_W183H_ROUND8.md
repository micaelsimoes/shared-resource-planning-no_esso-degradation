# P5.15 W183h: round 8 at Overleaf a87e195 (Addendum 84; re-ordered unchanged by Addendum 85 W183g (e))

**Worker, 2026-10-25.** Zero solves, zero model builds, zero pickle loads. The harness read the Overleaf clone,
`manuscript_review/` and committed artifacts only. It ran no fetch, pull or checkout, and it edited nothing in the
clone, `manuscript_review/`, production, earlier harnesses, specs, frozen files, TASKS.md, REVISION_CONTEXT.md or the
brief.

**State.** The Overleaf clone `manuscript/6a67305f25e8348fb71380c3` is at HEAD
`a87e1950fbdbb3667f37e64a43c298ba675e6cff` (2026-10-25 15:48:48 UTC). Its parent is
`edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838`, so the HEAD is one commit after `edd7a63`. The tracked files are clean; the
only untracked entry is `.DS_Store`. `git diff edd7a63 a87e195` touches exactly `main.tex`,
`response_to_reviewers_draft.tex`, `highlights.tex` and `cover_letter.tex`. The repository HEAD at run time was
`3707475a` (the harness commit). The only modified tracked file was `TASKS.md`, which belongs to the Planner and was
not staged.

**No TeX is installed on this machine.** Every section, table, figure, equation and citation number below is a
**static** number (W182c method). Nothing was compiled.

## Blocked on Planner

Three predictions did not hold as stated. None of them changes what runs next (W184 depends on W183g (a)–(d), not on
W183h). All three are manuscript-wording points for the expert or the author.

1. **(d) part 2 — two table occurrences of over-budget plans carry no $^{b}$ mark.** The prediction is that every
   over-budget point is labelled at every table or figure occurrence. It holds at 23 of 25 table/figure occurrences.
   - The two exceptions are both in Table E.24 (`tab:appE_gap`):
     - l. 2904, row of evaluation 39 (`7: 1.25/5`, 2025, $m=2$);
     - l. 2911, row of evaluation 6 (`7: 1.25/5`, 2025, base).
   - E.24's caption has no $^{b}$ legend.
   - The only thing that labels these rows is the §3.5 sentence (l. 1347), which names the five plans. That sentence
     also says the probes are "marked $^{b}$ in the tables", which is not true of E.24.
   - A fix would be a generator edit: mark the two E.24 plan cells and add the legend, or narrow the §3.5 wording.
     That is the expert's call (Addendum 83 ruling 1: Appendix E is edited only through `gen_appE.py`).
2. **(f) — seven label-word prefixes `Appendix~\ref{app:…}` / `Section~\ref{app:…}` are in `main.tex`.**
   The other two parts of (f) hold: there are 0 `[CONFIRM` markers in the four files, and blue is present on all 23 A5
   lines and all 5 A5 captions.
   - The seven prefixes, by line:
     - l. 530: `Section~\ref{app:results_statistics}`
     - l. 1631: `Appendix~\ref{app:results_benchmark}`
     - l. 1639: `Appendix~\ref{app:admm_updated_stopping}`
     - l. 1742: `Appendix~\ref{app:results}`
     - l. 1763: `Appendix~\ref{app:results_statistics}`
     - l. 1780: `Appendix~\ref{app:admm_updated_certification}`
     - l. 2002: `Section~\ref{app:results_statistics}`
   - Count by revision (same static method): 0 at `e5322f8`, `cd57efc` and `8f051a2`; **7 at `edd7a63`**; 7 at
     `a87e195`. The author's `edd7a63` edits introduced them, and round 8 left them in place.
   - One of the seven (l. 1639) sits on a round-8 '+' line (edit M12). Its prefix predates round 8.
   - On the W179b/W180c reading, elsarticle prints `\ref{app:…}` with its own "Appendix" prefix. So
     `Appendix~\ref{app:x}` would print "Appendix Appendix A.4", and `Section~\ref{app:x}` would print
     "Section Appendix E.7". This is not confirmed by a compile.
   - W182c's A4 check did not catch these, because its kind-word table maps "Section"/"Appendix" to "heading", which
     is the kind of an `app:` label.
   - The general pattern `<word>~\ref{app:` also matches 17 prepositions ("in" 14, "of" 2, "and" 1). As in W180c,
     these are not prefixes.
3. **Expert or author decision on items 1 and 2.** Should E.24 and the prefixes be handled in round 9? The Appendix E
   part goes through the generator; the prefixes are main-text edits.

## Changed

- **`3707475a`** adds the harness `p515_s53_w183h_round8.py` (sha256 `baab4416…`). It was committed before the real
  run; the run records that HEAD, the same sha256, and `script_committed_clean: true`.
  - It imports the following without editing them:
    - W182c `p515_s53_w182c_labels_pointers_appE.py`, and through it W179b, W178c, W177b-3′, W177b-3, W167, the S47
      record and W181a;
    - W182b `p515_s53_w182b_number_check.py`, and through it W178b, W177b-1′, W177b-1, W177a, …, W160;
    - W180c, for `GENERAL_PREFIX` and `LABEL_WORDS`.
  - No earlier checker was run onto its own output directory.
  - New code:
    - item (a): the diff comparison per file (W181a's `parse_udiff` / `compare_hunks`);
    - item (b): the mark check, and the removal of the $^{b}$ mark from E.20/E.21 rows before the entry check;
    - item (c): pairing of v10 pointers with v9 pointers, declared targets, and a lettered-number extractor;
    - item (d): the A7 and A8 re-rating;
    - item (e): declarations version W183h;
    - item (f): blue share by line.
- **This commit** adds this report and `data/SRP1/Results/P515S55/w183h_round8/`.
  - The directory was created fresh with `mkdir`. Every output was opened `'x'` and written once.
  - The run was attached, with stdout and stderr in `launch.log`. Exit code 0; 0 check failures.

| file | sha256 |
|---|---|
| `w183h_round8.json` | `64c199ee0765cbfd47831bebe25281d5679994d72508a52aa2e07a9e8e29f133` |
| `w183h_round8.md` | `964f055be740fd8b344a3cd96cca88b866a90eeebf0a4087e5ad6b0230038d77` |
| `appE_generated_w183h.tex` | `66bd3dd0a4d7b623232d95c420d1ba50cdb4bb0c075cfc0093af38e34b3a3d14` |
| `verify_appE_output.txt` | `7a0c47e9eee3b365075f0b46bdc6d20e7ec0ff40eab7f68c8213695861d213a7` |
| `manifest_sha256.json` | `8a3a3e0c4f05ea7256856ec0c9a8717001b7c7b82b49a75fb97f7d22483e260a` |
| `launch.log` | `302e22fc139a8ec6cc5d52818e893311a8d21b7605707b0470e4078372d3aed1` |
| `w183h_bool_typing_test.json` / `.log` | `9494faa7…` / `bc070744…` (VERDICT: PASS) |
| `manifest_post_run_sha256.json` | `6f16e7506100eb2c59a7e93a9f01628869e471efda45712f07acf5f51d629ce2` |

`manifest_sha256.json` hashes:
- every input: the four clone files, the four review copies, the four expert diffs, letter v9, the submitted cover
  letter, `gen_appE.py`, `verify_appE.py`, the committed appendix, all of `export_W160/`, the frozen JSON, the
  W182c/W182d/W182g/W180c JSONs, `STEP6_ROUND8_CORRECTIONS.md`, the brief, every imported script, and every input of
  the W182b context;
- the `git show` blobs of the four files at `edd7a63` and at `a87e195`.

Commands (repo root, canonical interpreter, run attached):

```
mkdir -p data/SRP1/Results/P515S55/w183h_round8 && \
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183h_round8.py \
    > data/SRP1/Results/P515S55/w183h_round8/launch.log 2>&1                                   # exit 0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \
    --out data/SRP1/Results/P515S55/w183h_round8/w183h_bool_typing_test.json \
    > data/SRP1/Results/P515S55/w183h_round8/w183h_bool_typing_test.log 2>&1                   # exit 0, PASS
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183h_round8.py --post-run   # exit 0
```

Guards:
- `SolveProfileGuard(permitted=())` was installed before any project import.
- All 57 distinct guards of the imported chains were verified at exactly 0.
- `pickle.load` / `pickle.loads` were blocked for the whole run; every counter was 0.
- No production module was imported.
- JSON was written only through `gate_result_io`.

Before the commit, three scratch trials ran outside the repository, under the session scratchpad
(`w183h_trial_1..3`). Trial 1 exposed three faults of mine:
- two declarations were missing (l. 145, l. 163);
- one pointer control was pairing to the wrong v9 pointer;
- one control planted the wrong text.

All three were fixed before the commit. Trials 2 and 3 exited 0 with the outcomes below. No `appE.tex` was written
anywhere in the repository.

## Found

Objective convention: no objective value is computed. Numbers are rated against the frozen JSON
`frozen_step6_tables_v1_590088fe.json` and the named committed records. The convention is Q = gross_operational_cost,
**contracted interface settlement excluded** (Addendum 84). Each rated monetary token carries its own convention field
in the JSON.

### (a) Identity — prediction "identical": **HELD**

| file at a87e195 | sha256 | md5 (Addendum 85) | lines | = round-8 deliverable | clone hunks / expert hunks | +/- lines, ranges, context |
|---|---|---|---|---|---|---|
| `main.tex` | `cec2e298…` | `39f6fd86` ✓ | 2,961 | `main_edd7a63_round8.tex` ✓ | 31 / 31 | equal, equal, equal |
| `response_to_reviewers_draft.tex` | `982482a1…` | `2aeb33b2` ✓ | 292 | `response_to_reviewers_v10.tex` ✓ | 12 / 12 | equal, equal, equal |
| `highlights.tex` | `13e37f6d…` | `4cedd45c` ✓ | 17 | `highlights_v3.tex` ✓ | 1 / 1 | equal, equal, equal |
| `cover_letter.tex` | `de7a9174…` | `88eee8ce` ✓ | 35 | `cover_letter_v2.tex` ✓ | 2 / 2 | equal, equal, equal |

- Each working copy equals its `git show HEAD:` blob.
- `cover_letter_v1_v2.diff` is taken against "cover_letter.tex (submitted)". `edd7a63:cover_letter.tex` is
  byte-identical to `manuscript_submitted/cover_letter.tex`, so that diff is the clone diff.
- `edd7a63`'s letter is byte-identical to `manuscript_review/response_to_reviewers_v9.tex`.
- Negative control: one '+' line of each expert diff was altered in memory. Exactly that hunk was flagged, in all
  four files.

### (b) Appendix E — prediction "bitwise; 33 PASS; 0 MISMATCH; 0 entries changed; the marks as listed": **HELD**

- **Bitwise.**
  - The patched `gen_appE.py` ran in-process with the explicit target
    `data/SRP1/Results/P515S55/w183h_round8/appE_generated_w183h.tex`. Its stdout was `ok 58 61`.
  - The output is 391 lines, sha256 `66bd3dd0…`. It occurs exactly **once** in `main.tex` at HEAD, at **lines
    2556–2946** (bytes 164,072–204,526), on line boundaries.
  - That range is the whole Appendix E block as located by content (l. 2556–2946): no line of the block lies outside
    it.
  - The output equals `manuscript_review/appendixE_evaluation_record.tex` and the verifier's own regeneration.
- **`verify_appE.py`: 33 PASS, 0 FAIL**, no exception. The full output is in `verify_appE_output.txt`.
- **Entry check (W180a/W181a `table_check`) at HEAD against the frozen JSON: 1,623 entries, 0 MISMATCH, 0 non-frozen
  disagreements, 0 mapping problems.**

  | table | entries | MATCH (frozen field) | non-frozen, agrees |
  |---|---|---|---|
  | E.20 | 639 | 581 | 58 |
  | E.21 | 854 | 732 | 122 |
  | E.22 | 18 | 18 | 0 |
  | E.23 | 49 | 49 | 0 |
  | E.24 | 63 | 63 | 0 |

  - E.21 has rows 1–61, mapped one-to-one onto the 61 frozen cells, with the same map as at `edd7a63`. E.20 has rows
    1–58. The same check at `edd7a63` gives 0 disagreements.
  - **Adaptation, declared.** The W180a plan decoder cannot read `7: 1/4$^{b}$`. The over-budget mark `$^{b}$` was
    therefore removed in memory from the E.20/E.21 rows only (16 rows, each removal recorded in the JSON) before the
    check.
  - E.23's own `$^{b}$` was left in place. It means "the start reported in Table 8", predates round 8, and is a
    checked field.
- **Entries changed `edd7a63` → HEAD: 0 of 1,637 with the marks removed.**
  - Raw, 16 cells changed. Each is a mark insertion only: new = old with `$^{b}$` inserted.
  - Non-cell changes are the E.20, E.21, E.22 and E.23 captions, all expected round-8 edits:
    - "contracted interface" inserted;
    - the $^{b}$ legend in E.20 and E.21.
  - There are no changes to row sets, notes or group headers.
- **The marks: exactly as stated.**
  - E.21 rows {5, 6, 9, 10, 11, 38, 39}.
  - E.20 rows {4, 5, 8, 9, 10, 26, 27, 30, 31}.
  - Each mark sits on a plan that is over the 1 M€ budget, per W182d's committed I(x) table.
  - Every over-budget evaluation is marked. Row 30 marks only `1/4`; row 31 marks both `1/4` and `1.25/5`.
  - The legends are present in both captions.
- **Block diff `edd7a63` → HEAD: 16 hunks, all expected round-8 edits.**
  - The generator header comment.
  - The intro convention ("contracted interface settlement excluded").
  - E.1 l. 2601 ($y^{\text{End}} = 2040$; terminal SoH = end of the last block, 2035 / 2037).
  - The E.20 and E.21 captions.
  - Marks on four E.20 row groups and three E.21 row groups.
  - Blue on the lattice-figure caption.
  - "contracted interface" in the E.22 and E.23 captions.
  - E.7 "The acceptance tolerance~\eqref{eq:acceptance_tolerance} uses …".

### (c) Letter v10 pointers — prediction "0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED": **HELD**

- **Label numbers.** The static numbering at HEAD equals `edd7a63` for all 136 labels; 0 differ. This statically
  agrees with the expert's `.aux` statement.
- **v9 recomputed at HEAD** with W182c's own functions: 103 MATCH, 65 MISMATCH, 5 SUBMITTED, 0 AMBIGUOUS.
  - This agrees with committed W182c on 172 of 173 pointers.
  - The one difference is l. 97 "Section~3.3". It was AMBIGUOUS at `edd7a63` and is now MATCH through its content
    needle: the restored M18 sentence puts it under §3.3.
- **v10: 172 pointers — 167 MATCH, 5 SUBMITTED, 0 MISMATCH, 0 AMBIGUOUS.**
  - The SUBMITTED five, carried from W180c: l. 100 "Figure 2"; l. 102 "Figure~2 of the submitted version"; l. 141
    "former 2.2.7"; l. 232 "Section~4.4" and "Figure~15".
- **Four v10 pointers needed a declared target**, each taken from `STEP6_ROUND8_CORRECTIONS.md` §C. All four MATCH:

  | line | pointer | why declared | declared target | resolves to |
  |---|---|---|---|---|
  | 58 | `Section~4.3` | new | `subsec:res_degradation` | 4.3 |
  | 136 | `Sections~2.2` | v9's 2.2.6 target removed | `subsec:master` | 2.2 |
  | 229 | `Section~2.2.6` | moved from v9 l. 230 | `subsubsec:certification` | 2.2.6 |
  | 231 | `Appendix~A.5` | new | the heading holding `eq:acceptance_tolerance` | A.5 |

- **Lettered pointers.** The W177b-3′ extractor does not read numbers with a letter, so a lettered-number extractor
  was added. It finds three, all MATCH: l. 220 and l. 222 "Eqs.~(E.1)--(E.2)", and l. 288 "Fig.~E.19". Letter v9 had
  none.
- **v9 pointers dropped in v10:** l. 125, l. 126 and l. 288 "Figure~2" (the removed search figure, as round 8 C says),
  and l. 230 "Section~2.2.7" (moved to l. 229).
- **Ranges.** l. 81 and l. 86 "Tables~4--11" MATCH. The endpoints are correct; the former member `tab:res_flex_ladder`
  was removed since `8f051a2`.
- **Citations [42]/[43]: HELD.**
  - Static citation order at HEAD: `naumann_2018` = [42], `eve_mb31_datasheet` = [43], `boyd_admm_2011` = [44].
  - Letter l. 77 and l. 249 "[43]", and l. 256 "[42]", are MATCH. Both v9 sentences are present with the new numbers.
- Negative controls, all refused:
  - l. 45 "Section~3.5" (MISMATCH);
  - l. 256 "[43]" (MISMATCH);
  - l. 288 "Fig.~E.18" (MISMATCH);
  - l. 231 "Appendix~A.4" (MISMATCH against its declared target).

### (d) A7 — prediction "0 of the 16 remaining": **HELD**

The method, per W182d listed entry:
1. Check that W182d's quote is present at `edd7a63`. This positive control held for 16 of 16.
2. Check that the quote is absent at HEAD.
3. Check that the declared defect phrases are absent from the whole HEAD file.
4. Check that the declared correction phrases (from round 8 B/C/D) are present at the mapped lines ±1.

Result: **16 of 16 RESOLVED.**

| id | HEAD place | correction located |
|---|---|---|
| M-ABS-1 | main.tex:145 | "formulated as a two-stage stochastic program in which one scenario-independent plan …"; "probability-weighted cost trajectories" |
| M-ABS-2 | main.tex:145 | "explores discrete investment plans in a single operating scenario"; "fixed plans are then re-evaluated under … scenarios" |
| M-HL-1 | main.tex:163 | "A two-stage framework plans shared lithium-ion storage at TN--DN interface nodes" |
| M-CON-1 | main.tex:387 | "the planning searches run on a single operating scenario and fixed plans are evaluated under market and operation scenarios" |
| H-3 / H-4 | highlights.tex:12/13 | bullets replaced; no "certif" and no "breaks even" in the file |
| L-01 | letter:42–43 | "during the search under a residual-based stopping rule … under an explicit numerical acceptance rule" |
| L-04 | letter:78 | "by more than the reported numerical resolution"; "stopped by the production exit and are not certified"; "0.933 predicted" gone |
| L-05 | letter:102 | "driving recourse evaluations under a declared acceptance rule" |
| L-07 | letter:137 | "carries a declared acceptance status" |
| L-08 | letter:168–169 | "neither bounds the distance to an optimum nor selects the basin" |
| L-09 | letter:229–230 | "what is assessed is each recourse evaluation" |
| L-10 | letter:230–232 | "a declared tolerance, not a bound on the error of a ratio" |
| L-12 | letter:263 | "within the numerical resolution, so its sign is not resolved" |
| L-13 | letter:264 | "within the numerical resolution of zero, so its sign is not resolved" |
| C-1 | cover_letter (place not mapped; v2 rewrote the paragraph; whole file searched) | "a two-stage stochastic program selecting one scenario-independent investment plan"; "bi-level" and "staged investment decisions" absent |

- Borderlines at HEAD, for information (Addendum 84 ruling 2 says they stand):
  - still present: M-CON-3, M-45-1, M-45-2, M-5-1, L-02, L-06, L-11;
  - no longer present (rewritten): L-00, L-02b, L-03, L-14.
- Negative control: the L-05 quote, planted back into a copy of the letter, is detected as REMAINS.

### (d) A8 — prediction "labelled at every table/figure occurrence": **NOT HELD (23 of 25)**

The 34 places of W182d were mapped `edd7a63` → HEAD by the line diff.

| rating at HEAD | places |
|---|---|
| $^{b}$ at the place | 18: E.20 rows 4, 5, 8, 9, 10, 26, 27, 30, 31; E.21 rows 5, 6, 9, 10, 11, 38, 39; Table 6 rows 4$^{f,b}$, 5$^{u,b}$ |
| every over-budget row under the header marked | 1: the E.21 group-(a) header |
| table marks its rows and carries the legend | 1: the Table 6 caption (legend l. 1502) |
| probe / budget text at the place | 5: §4.2 l. 1427 (P03), Fig. 5 caption l. 1433, §4.2 l. 1469 (P06, P07), E.7 l. 2926 |
| probe text in the same paragraph | 2: l. 1426 (P01, P02; the next line l. 1427 says it) |
| covered by the §3.5 sentence only | 1: the Table 5 caption l. 1439 (the sentence names "the break-even fit of Subsection~\ref{subsec:res_breakeven}") |
| partial, unchanged | 1: the E.4 lattice (P14; the figure legend marks over-budget lattice points) |
| withdrawn | 2: letter l. 58 and l. 264 (the affine-ladder argument is gone) |
| **not marked** | **2: E.24 rows of evaluations 39 and 6 (l. 2904, l. 2911)** — see Blocked 1 |
| not labelled at the place (text, not a table/figure) | 1: §5 l. 1806 (P09) |

The §5 l. 1806 sentence is "The estimated marginal break-even energy costs for 4 h storage at node 7 were below the
baseline…". It is derived from the fit. The prediction concerns table and figure occurrences, so this place is
recorded, not counted.

Outside the 34 places, a token scan of HEAD finds `7: 1/4` on two lines, E.20 row 41 and E.21 row 49. Both are 2030
plans (admissible), not 2025 over-budget occurrences.

### (e) Numbers in the 21 main-text edits — prediction "0 MISMATCH": **HELD**

**Scope.** The '+' lines of `git diff -U0 edd7a63 a87e195 -- main.tex` that lie outside the Appendix E block: 32 lines,
of which 20 carry numeric tokens. That gives 114 tokens:
- **93 MATCH, 7 DERIVED, 0 MISMATCH, 0 UNSOURCED, 14 NOT RATED.** The NOT RATED tokens are 7 compound adjectives
  ("two-stage", "single-node", "single-scenario") and 7 model-structure words (A.1 "Three families …, one copy …").
- Supplement, the E.1 line l. 2601 (generator prose): 4 MATCH and 1 NOT RATED ("1 January").
- Negative control: all 104 rated tokens were perturbed, and 104 of 104 were refused.

Method: W182b's tokenizer, evaluator and spec constructors, with declarations version W183h. Specs of unchanged tokens
were taken from the W182b (`edd7a63`) or W177b declaration of the same sentence and **re-evaluated** here. They were not
carried as ratings.

The named items:

| line | written | rating | source / arithmetic |
|---|---|---|---|
| 1291 | 1.00 % / 2.50 % (TN) | MATCH | `case9_operational_data.xlsx` sheet Growth Factors, Load / Flexible Load × 100 |
| 1291 | 1.25 % / 3.00 % (ADN), 32 | MATCH | the three ADN workbooks; case33 files (32 loads, `fl_reg` 1) |
| 1347 | Five; node 7; 2025; 4 h: 4, 5 MWh; 2 h: 3, 4, 5 MWh | MATCH | W182d A8 over-budget evaluations at the base price, exactly these five |
| 1347 | 1.15–1.91 M€ | MATCH | W182d I(x): min 1,146,109.02 € (7: 1.5/3), max 1,910,181.70 € (7: 2.5/5); expected discounted investment cost |
| 1347, 1433, 1502 | 1 M€ | MATCH | `SRP1_ESS_Params.json` budget 1,000,000 € |
| 1427 | five of the ten; 4/5 MWh at 4 h; 3/4/5 MWh at 2 h | MATCH | over-budget among fit points 2–11 = 5; ten = `break_even.conservative_A64.banded_mid.n` |
| 1469 | 4 and 5 MWh (at $m=2$) | MATCH | W182d over-budget evaluations 38/39 |
| 1576 | 2037 | DERIVED | W182g C2 cell `3x3_n7_4h_e1` (eval `c82522f4…`, cycle 69): the first block with the 0.70 floor active is 2037; SoH before 0.928/0.863/0.804/0.750, at 2037 0.6999999900 |
| 1576 | three(-year blocks) | MATCH | `SRP1__s53_3x3.json` Years, each 3 |
| 1639 | about 3 % | DERIVED | W182g record `3x3_x0` (eval `f6e9cd53…`, cycle 72): (deviation −748,745.77 € + row-18 premium 25,082,401.21 €) / Q 842,832,534.76 € = **2.887 %** (premium alone 2.976 %); 3 at 0 dp. Convention: Q = gross, contracted settlement excluded |
| 145, 387 | "a single operating scenario" | MATCH | SRP1.json (the instance of the searches) has 1 market × 1 operation scenario |
| 2601 | 2040 (both configurations); 2035; 2037 | MATCH | W182g C2: terminal date y0 + ΣY = 2040 for SRP1 and 3 × 3; last blocks 2035 and 2037 |

The other DERIVED tokens:
- 108, from the case-file node counts;
- 24 % and 33 %, from T3;
- "2τ", from T3's interval half-widths;
- "nine" = 3 × 3.

The re-evaluated carried tokens all MATCH. They include 171,310 / 192,560 / 253,880 (l. 1427), Table 6 rows (l. 1489–1490),
318.0, the 3 × 3 236.7 / 318.0 / 81.3 / 18.0 (l. 1639), and the floor sentence 0.70 / 2035 / 50 / 8,000 / 0.50 /
0.18 / 0.20 / 0.27 / 4.9 / 59.5 (l. 1576).

Statement checks, all hold:
- **S1.** The five plans named are exactly W182d's over-budget base set.
- **S2.** "No returned plan … among them". The returned 2× plan (5: 0.25/0.5; 7: 1/3.5 @2030) and every W182g
  prose-anchor plan are admissible. The JSON states the scope: a comparison row that values a probe is the probe, not a
  comparator.
- **S3.** "the contracted settlement is excluded" at 3 × 3. In W182g's three records, gross = including − contracted −
  pin, exactly.

### (f) — prediction "hold": **NOT HELD on one part**

- `[CONFIRM`: 0 in each of the four files (held).
- Label-word prefixes: **7** (see Blocked 2).
- Blue on the 23 A5 lines of W182c, mapped `edd7a63` → HEAD (l. 1317, 1381–1382, 1406–1410, 1474, 1496–1501,
  1503–1505, 1534–1535, 1568–1570): the alphanumeric blue share is 1.0 on all 23 (held).
- The 5 A5 captions (`fig:investment_lattice`, `tab:ageing_calibrations`, `tab:res_value_ladder`,
  `tab:res_energy_ladder_m2`, `tab:res_ageing`) are blue, 1.0 (held).
- The new Table 6 legend line (l. 1502) is inside the blue note.
- Controls, both refused:
  - a planted `[CONFIRM` is counted;
  - the Table 3 caption unwrapped in a copy is detected.

### The letter's prediction-record bullet (quoted only; Addendum 84 ruling 1: the author's call)

Letter v10 l. 286: "`\item The prediction record of the revision (49 predictions, with outcomes) is provided as
supplementary material.`"

## Unexpected findings (not acted on)

1. **$^{b}$ carries two meanings in Appendix E.** In E.20/E.21 it means "investment above the 1 M€ budget" (round 8).
   In E.23 it means "the start reported in Table 8" (its note, pre-existing). Each table defines it locally. Table 6 in
   §4.2 uses `$^{f,b}$` with the over-budget meaning. This is a notation point for the expert.
2. **"Within the numerical resolution of break-even."** Letter v10 now writes "within the numerical resolution of zero,
   so its sign is not resolved". `main.tex` l. 1520 (§4.3) and l. 1806 (§5) keep "within numerical resolution of
   break-even". The new cover letter v2 (l. 19) writes "its margin without ageing is within the numerical resolution of
   break-even, which identifies degradation as the decisive factor in its valuation". This is the wording family of
   H-4 / L-12. These lines were not among the 16 (cover letter v2 is new text; l. 1520 is outside W182d's scope) and
   were not rated.
3. **Cover letter v2 holds a bracketed round-9 placeholder** ("[Paragraph on the additional experiments … to be
   completed when their results are in.]"). It is expected per round 8 D. Recorded so it is not submitted as is.
4. **The W177b-3′ pointer extractor misses lettered numbers** ("Fig.~E.19", "Eqs.~(E.1)--(E.2)"). W183h adds its own
   extractor. No earlier stage was affected: v9 had no such pointers.

## Not confirmed

- **Compilation.** Nothing was compiled; there is no TeX on this machine and no `.aux`/`.bbl` in the clone. Label,
  citation and page numbers are static. The expert's statements ("0 errors, 0 undefined, 76 pages, every label number
  identical to `edd7a63`") were not reproduced; only the static label equality (136/136) was checked.
- **How elsarticle prints `Appendix~\ref{app:…}`.** "Appendix Appendix A.4" is the W179b/W180c reading of
  `elsarticle.dtx`, not a compiled result.
- **The remote.** The clone's HEAD is `a87e195` as present on disk. Nothing was fetched, so whether Overleaf has moved
  since was not checked.
- **Declared pointer targets.** The four declared targets (l. 58, 136, 229, 231) rest on the expert's round-8 note and
  the letter's own sentence, not on an independent derivation.
- **The A7 "RESOLVED" ratings** are phrase-level: the original quote is absent, the defect phrases are absent and the
  correction phrases are present. They are not a fresh reading of each new sentence against all evidence. W182d's
  borderline and "other" entries were not re-rated, except for presence.
- **(e) carried sources.** Some carried tokens rest on committed registries, as in W182b:
  - the W163 check registry (0.18 / 0.20 / 0.27);
  - W182d's I(x) table;
  - W182g's records.

  Those records were not recomputed. The I(x) table was produced through production readers in W182d; this stage did
  not import production.
- **The verifier's temporary file** lives in a system temp directory. It was hashed (equal to the generator output) but
  not preserved.
