# P5.15 W189a-v4: supplement v4 (Amendment to Addendum 91, supplement v4 naming)

**Worker, 2026-11-12.** Zero solves, zero model builds, zero pickle loads. Same procedure as v3
(`P5_15_W188B_SUPPLEMENT_v3.md`): README prose switched to `supplement_README_prose_v3.md`, and the amendment's
name mapping applied to the generated descriptions from version 4 on.

**Submitted file (per the amendment):** `manuscript_review/supplement_v4/SRP1_supplement_v4_d175ea0e.zip`
- sha256 `d175ea0e2c7b14b8f7712a202b8eace79ceeeecf702c93f689f93496d9029bb0`
- 1,326,490 bytes, 20 entries

Left alone:
- `manuscript_review/supplement/` (v1 and the three prose files), `supplement_v2/` and `supplement_v3/`.
  `git diff --quiet HEAD` passes on all three, and `git status --porcelain` is empty on all three.
- Production, harnesses, launchers, hooks and `data/`.
- TASKS.md, REVISION_CONTEXT.md and the brief.

Nothing named `p515_s4*` was run. `.git/index.lock` was checked absent before each commit. Every run was attached,
with both streams captured. W187 (`p515_s55_step7_launcher.py`, pid 31351) was running throughout. Its pinned set
(production modules, run files, imported modules, `data/SRP1` inputs) holds no `manuscript_review/` file. I read this
in `pins_now()` and did not run the launcher.

## Blocked on Planner

Nothing blocks the upload. v4 is generated, verified (0 failures) and committed, and it meets the amendment as
written. Two items are for the expert to take or drop. Neither changes data.

1. **Some old names in the generated half are not covered by the mapping, and I left them as they are** (as ordered).
   They are listed exactly below. None of them matches the task's stale-name patterns, so V10 counts 0.
   - **(a) The generated paragraph under "## Files and columns".** It is not a `COLUMN_SPECS` string. It reads:
     "Empty CSV cell: …, or (Appendix E) printed "---". … Appendix E amounts are in k EUR as printed …; Appendix B-D
     values are at the full precision of the case files."
     - "Appendix E" (2 times) and "Appendix B-D" (1 time) are a87e195 names.
     - The mapping covers "Appendix E.2"–"E.7" and "Appendix B/C/D (…)", but not bare "Appendix E" or "Appendix B-D".
   - **(b) The definitions of the `table` columns.** `E20_E21_groups.csv` has "E.20 or E.21", and
     `E_table_notes.csv` has "E.20, E.22 or E.23".
     - These definitions describe CSV **values**, and the CSVs still hold `E.20` / `E.21` / `E.22` / `E.23`.
     - Renaming the definitions would make them contradict the data, which the ruling keeps unchanged.
     - V10 records these 5 occurrences as residue.
   - **(c) The parenthesised LaTeX labels.** Of the 17 labels, 14 are absent from round-9c `main.tex` (3c379fa).
     - `tab:appE_evaluations`, `tab:appE_benchmark` and `tab:appE_gap` are present in
       `appendixE_evaluation_record.tex`, the full mode.
     - The other 11 are the B–D labels: `fig:cs3_market_prices`, `tab:cs3_tn_res_generators`, `tab:cs1_ieee33_*`,
       `fig:cs3_adn_node_5/7/9_*` and the rest. They are absent from 9c. I checked three of them
       (`fig:cs3_market_prices`, `tab:cs1_ieee33_buses`, `tab:cs3_tn_res_generators`) and found them in a87e195
       `main.tex` and in the earlier review copies; I did not check the other eight one by one.
     - Only `tab:appE_claims`, `tab:appE_neighbours` and `tab:res_capacity_summary` are in 9c.
     - Example: "Section S1 (fig:cs3_market_prices)".
   - **(d) Wording that now reads oddly** (outside my remit):
     - "not printed in Sections S2-S3" on the 2030/2035 capacity columns. Sections S2–S3 point to this archive's own
       files, which do carry those years.
     - "printed mark b of Table S5" and "Table S6 itself prints no mark". Tables S4–S6 are not printed in the
       manuscript; the marks are those of the full-mode tables.

   Options:
   - **(i) Submit v4 as it is.** It is correct to the ruling, and these are wording points.
   - **(ii) A v5 with expert wording** for (a), (c) and (d). It would cost about 1 min of generation plus about 5 s
     of verification, all zero-solve, and needs the expert's strings.

   No recommendation: the wording is the expert's.
2. **Interpretation choices, for acknowledgment.** No decision is needed.
   - **"(supplementary)".** The amendment's "Table S4 (supplementary)" is read as the name "Table S4", with
     "(supplementary)" taken as a note in the table. This is consistent with the prose ("Table S4 — recourse
     evaluations") and with main.tex ("Table~S4").
   - **Plural phrase added.** The source has the phrase "Group titles of Tables E.20 and E.21", so one entry was added
     to the map: 'Tables E.20 and E.21' → 'Tables B.10 and S4'. It is the plural of the rows E.20 → B.10 and E.21 → S4.
   - **"Appendices C-D".** The source uses an ASCII hyphen. The S-section equivalent is written "Sections S2-S3", in
     ASCII to match the generated section.
   - **Unused map entries.** "Appendix E.2"–"E.7" and "Fig. E.19" are in the map but match **0** strings in the
     generator. The generated half never named them, so v4's README contains no "Fig. B.9" and no "Appendix B.4".
     V11 still confirms those numbers in main.tex.
   - **Verifier beyond the literal "add a v4 check".**
     - V10 also rebuilds v4's README from v3's: prose swapped, heading swapped, and the amendment's mapping applied,
       using the verifier's own copy of the mapping. It then requires byte equality.
     - V11 (new) is the round-9c name check, by a static count.
     - Control C22 is new.

   Without V10 and V11, the expected outcomes in task item 4 would rest on ad-hoc scratch commands. In the committed
   verifier, the formula is preserved.

## Changed

| commit | content |
|---|---|
| **`02ebf891`** | `manuscript_review/gen_supplement.py` (sha256 `3445d4324ce86b23…`) and `manuscript_review/verify_supplement.py` (sha256 `cd90e98acdee1b1b…`), +247 / −18, committed **before** any canonical generation |
| **`e4508c54`** | `manuscript_review/supplement_v4/`: the zip, the record, the verifier result, `gen_launch.log`, `verify_launch.log` and `SHA256_INVENTORY.txt`. The directory was created by `mkdir` in the launch command and written once |
| this commit | this report |

**Generator.**
- `readme_prose_rel(version)` returns prose v1 for version ≤ 2, v2 for 3, and v3 (`README_PROSE_V3_REL`, new) for
  version ≥ 4.
- `V4_NAME_MAP` is a new ordered tuple of literal replacements: the amendment's table, plus the plural phrase above.
- `description_names(version, hits)` returns the identity for versions ≤ 3 and applies the map for versions ≥ 4.
- `readme()` applies it to each `COLUMN_SPECS` `table`, `source` and column definition as it writes them.
  - **`COLUMN_SPECS` itself is unchanged.** It still carries the a87e195 names, which is what keeps v2 and v3 bytes.
  - CSV names, CSV builders and the zip writer are untouched.
- The record gains a top-level `description_names` key (map and per-entry hit counts), for v ≥ 4 only. Its `stage`
  string now also names W189a-v4. The record is outside the zip.
- The docstring was updated.

**Verifier.**
- V7 reads the prose through the same version map.
- `V9_DECLARED_BY_VERSION[4]`: changed = [`README.md`, `SHA256SUMS.txt`], added = [], removed = []. RES columns
  added in v4: [].
- **V10** (v ≥ 4), on the README after the prose. CSV file-name identifiers (`*.csv`, `appendices_B_D/`,
  `appendix_E/`) are masked first: 36 masked.
  - It counts `Tables? E\.2\d` (this covers the plural), `Appendix E\.`, `Fig\. E\.19`, `Appendix B \(`,
    `Appendix C`, `Appendix D` and `Appendices C`. It must find 0 of each.
  - It requires that README = v4 prose + (v3's generated half with the heading "version 3" → "version 4" and the
    mapping applied in order).
  - It records, without failing on them, the residue of bare "Appendix E", "Appendix B-D" and `E.2x` values.
- **V11** (v ≥ 4) works on `git -C manuscript/6a67305f25e8348fb71380c3 show 3c379fa:main.tex`, sha256 pinned to
  `b13b390e…`.
  - Comments are stripped, and a walk numbers the sections and subsections and the figure / table counters.
    `\addtocounter` is applied, and the counters are not reset at `\appendix`; this follows elsarticle.dtx l. 1459–1469
    in the clone, which redefines only `\thefigure` / `\thetable`.
  - It requires the following computed numbers:
    - `tab:appE_claims` = Table B.10;
    - `tab:appE_neighbours` = B.11;
    - `fig:investment_lattice` = Fig. B.9;
    - `app:results_claims` / `app:results_neighbours` / `app:results_statistics` = Appendix B.2 / B.3 / B.4.
  - Every `A.n` / `B.n` name in the README must exist in main.tex with its kind.
  - S1–S6 must each be referenced in main.tex, and the README may name no other S-number.
- **C22** is an in-memory control: one "Table B.10" in the generated half is set back to "Table E.20". It must give
  stale count 1, derivation False and V10 False.

**Commands.** All were run from the repo root, with `P=/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`,
attached, with `> log 2>&1`. `<W>` = `<session scratchpad>/w4`, which is outside the repository.
`<W>/drv/emulate_version.py` is a scratch driver and is not committed. It imports the repository's `gen_supplement.py`
as a module, so its guards install at import, pins `resolve_out_dir` to (scratch, N, predecessor = the committed
record of N−1) and calls `main()`.
```
# before the edit, at c86b451d:
$P -u <W>/drv/emulate_version.py <W>/pre_v2 2          # exit 0 -> 864a5779 (cmp = committed v2)
$P -u <W>/drv/emulate_version.py <W>/pre_v3 3          # exit 0 -> e6f5ce6a (cmp = committed v3)
# after the edit, uncommitted:
$P -u <W>/drv/emulate_version.py <W>/post_v2 2         # exit 0 -> 864a5779 (cmp = committed v2)
$P -u <W>/drv/emulate_version.py <W>/post_v3 3         # exit 0 -> e6f5ce6a (cmp = committed v3)
$P -u manuscript_review/gen_supplement.py --force-new-version --out-dir <W>/trial_v4     # exit 0 -> d175ea0e
$P -u manuscript_review/verify_supplement.py --supplement-dir <W>/trial_v4              # PASS, 0 failures
$P -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v3 --out-dir <W>/reverify_v3   # PASS
$P -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v2 --out-dir <W>/reverify_v2   # PASS
# at 02ebf891:
$P -u manuscript_review/gen_supplement.py --force-new-version --out-dir <W>/det_A|det_B|det_C   # exit 0 x 3
mkdir manuscript_review/supplement_v4 && \
  $P -u manuscript_review/gen_supplement.py --force-new-version > manuscript_review/supplement_v4/gen_launch.log 2>&1      # exit 0
$P -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v4 \
  > manuscript_review/supplement_v4/verify_launch.log 2>&1                                                                  # exit 0, VERDICT: PASS
```

| file (`manuscript_review/supplement_v4/`) | bytes | sha256 |
|---|---|---|
| **`SRP1_supplement_v4_d175ea0e.zip`** (20 entries) | 1,326,490 | **`d175ea0e2c7b14b8f7712a202b8eace79ceeeecf702c93f689f93496d9029bb0`** |
| `supplement_v4_record.json` | 16,564 | `e2d169276ad02fe34fd8165fc74334badcc1eeb0d81d5d6c55a4cdcaa6e6f775` |
| `verify_supplement_v4_result.json` (gate_result_io) | 33,225 | `63c8c54f11d889a93a56d570182f106d300135b2c15196e8dfea89349f2e21bd` |
| `gen_launch.log` (exit 0) | 2,026 | `1ac23d8446ebcaba9757d1d459a2c6e24068b26fa27ae12c0843bee7181c42f3` |
| `verify_launch.log` (exit 0) | 7,045 | `83f7645660f431d15483fe99386f2761ac0c12f6162bb72cac625e6b8b3a9885` |
| `SHA256_INVENTORY.txt` | 454 | lists the five above |

**Record provenance.**
- `git_head` is `02ebf891…`, and `generator.committed_clean` is true.
- `readme.prose_prepended` is `manuscript_review/supplement/supplement_README_prose_v3.md`, with sha256
  `3092a177…`. `prose_committed_clean` is true (the prose was committed at `3bd1f018`).
- The predecessor is version 3: `supplement_v3/supplement_v3_record.json`, `SRP1_supplement_v3_e6f5ce6a.zip`, sha256
  `e6f5ce6a5ef9…155d`.
- The guards counted 0 solves, 0 process launches and 0 / 0 pickle calls.
- `generated_equals_committed_appendix` is true.

**Objective convention** (unchanged; the Appendix-E CSVs are v3's byte for byte):
- Q = `gross_operational_cost`, with the contracted interface settlement excluded.
- net = Q − terminal salvage credit.
- value = Q(0) − Q(x).
- F = Q + I.
- Amounts are in k EUR as printed.

## Found

| check | expected | observed |
|---|---|---|
| verifier on v4 | PASS, 0 mismatches | **VERDICT PASS, 0 failures**; 37 PASS lines; **22 of 22** negative controls caught exactly (C1–C21 as v3, plus C22) |
| V1 Appendix E vs frozen tables | 1,623 entries, 0 MISMATCH | **1,623: 1,443 MATCH, 0 MISMATCH**, 180 not-frozen agree, 0 disagree, 0 mapping problems; per table E.20 639 / E.21 854 / E.22 18 / E.23 49 / E.24 63 |
| V5 B–D vs case files | 70,851 cells, 0 | **70,851 cells in 2,700 rows, 0 differences** |
| V8 Table 2 | 48/48 | **48 printed aggregates, 0 differences** (a87e195 and the round-9b copy); multi-scenario sentence 0 violations |
| `D_ieee33_tn_connection.csv` | byte-identical to v3 | **identical**: sha256 `f954ad5057ee2c77…` in both (V9, and independently by unzip + `shasum`). Node 7 is ±100; nodes 5 and 9 are ±250 |
| v3 → v4 entries | README.md + SHA256SUMS.txt only | **exactly these two** (V9: 18 identical, 0 added, 0 removed). Independently, `diff -rq` of the two unzipped trees lists the same two files |
| v3 → v4 README diff | mapping + prose v2→v3 + heading | **33 hunks, 35 changed lines**: 1 prose line (l. 24, the `E20_E21_groups.csv` row = `diff prose_v2 prose_v3`), 1 heading (l. 43, "version 3" → "version 4") and 33 generated lines with one substitution each. V10: README is **byte-equal** to v3's README with the prose, the heading and the map applied |
| map hits (on v3's generated half; the generator records the same) | — | Tables E.20, E.22 and E.23: 1; Tables E.20 and E.21: 1; Table E.20: 3; Table E.22: 1; Table E.21: 8; Table E.23: 2; Table E.24: 2; Appendix B (: 1; Appendix C (: 2; Appendix D (: 8; Appendices C-D: 4. Appendix E.2–E.7 and Fig. E.19: 0 |
| stale-name check | 0 | **0 for each of the 7 patterns** (36 CSV identifiers masked). Residue outside the check: bare "Appendix E" 2, "Appendix B-D" 1, `E.2x` values 5 (Blocked on Planner, item 1) |
| determinism | 3 scratch runs = the real one | **5 byte-identical `d175ea0e…`**: the pre-commit trial, det_A, det_B and det_C at `02ebf891`, and the canonical run (`cmp`). The det_A and canonical records differ only in `out_dir` and `canonical` |
| v2 / v3 reproduce | byte for byte, before and after | **v2 `864a5779…` and v3 `e6f5ce6a…`, before and after the edit** (4 scratch regenerations, each `cmp`-identical to the committed zip) |
| v2 / v3 under the new verifier | (regression) | both **PASS, 0 failures**, 21 controls each (V10 / V11 / C22 do not run for v < 4) |
| names vs round-9c main.tex | B.10, B.11, B.9, B.2–B.4; S1–S6 referenced | V11, static walk of 3c379fa `main.tex` (sha256 `b13b390e…`, 102 labels): `tab:appE_claims` **B.10**, `tab:appE_neighbours` **B.11**, `fig:investment_lattice` **B.9**, `app:results_claims` / `_neighbours` / `_statistics` **B.2 / B.3 / B.4**. Every README name exists with its kind (Appendix A.5, B.1, B.2, B.3; Table(s) B.10, B.11). main.tex references S1 2, S2 2, S3 4, S4 5, S5 3 and S6 3 times (comments stripped). The README names exactly S1–S6 |
| V4 | (inherited) | gen_appE output = committed appendix; verify_appE.py 40 PASS 0 FAIL |
| zero solves | 0 | generator and verifier `SolveProfileGuard(permitted=())`, plus the W180a module's guard: **0 solves, 0 process launches, 0 blocked**; pickle 0 / 0 in each |

## Unexpected findings

1. **I overwrote two files in the shared session scratchpad that the v3 report cites.** The scratchpad was the
   previous Worker's, and both files are outside the repository and uncommitted.
   - My first emulate-driver write replaced `<scratchpad>/drv/emulate_version.py`.
   - My first v2 run, into the existing `<scratchpad>/pre_v2`, refused (the directory was not empty) and overwrote
     `<scratchpad>/pre_v2.log` with that refusal line. The original log content (v3 Worker's pre-edit v2
     regeneration) is lost.
   - The zip in `pre_v2/` was not touched: the refusal happens before any write.
   - The v3 report's result (`864a5779` both times) is independently re-established here, in `<W>/pre_v2` and
     `<W>/post_v2`.
   - After this, all my work went to the fresh `<scratchpad>/w4/`.
2. **The review copy's name does not match its content.** `manuscript_review/main_46c9d77_round9c.tex` has sha256
   `b13b390e…`. That equals `3c379fa:main.tex`, not `46c9d77:main.tex` (`2739a78e…`). V11 pins 3c379fa. This is
   scoped to the two `git show` hashes; I did not trace why.
3. **Carried over from v3.** `verify_appE.py` spawns `gen_appE.py` children that the in-process guard does not see.
   They import only `csv` / `sys` / `os`, so the zero-solve claim rests on that import list for the children. V4 also
   leaves temp directories in the system temp dir.

## Not confirmed

- **Compiled numbering.** No TeX is installed. B.10 / B.11 / B.9 / B.2–B.4 come from a static walk that assumes:
  - `longtable` steps the table counter (hence the source's `\addtocounter{table}{-1}` after the three nomenclature
    longtables);
  - the `algorithm` floats have their own counter.

  The figure count agrees with W190a's l. 285 "B.9" closure, and the label count (102) with W190a's. Confirmation
  needs the Overleaf PDF.
- **Round 9c as a source for V1, V2, V6 and V8.** These checks are still pinned to a87e195, plus round 9b for V8, as in
  v3. Only V11 reads 3c379fa.
- **Cross-machine byte identity.** This was checked on this machine only (Python 3.11; library versions in the
  record).
- **v1 reproducibility.** Not re-examined. As the v3 report says, v1's `a94d8193` bytes are reproducible only with the
  generator at `5d7955da`.
