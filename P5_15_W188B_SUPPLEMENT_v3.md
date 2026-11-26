# P5.15 W189a-v3: supplement v3 (Addendum 91 ruling 3)

**Worker, 2026-11-12.** Zero solves, zero model builds, zero pickle loads. The run is v2's procedure
(`P5_15_W188B_SUPPLEMENT_v2.md`), with the README prose switched to `supplement_README_prose_v2.md`.

Nothing was edited or staged outside the files listed under Changed. In particular, these were left alone:
- `manuscript_review/supplement/` (v1, including both prose files) and `manuscript_review/supplement_v2/`. `git diff
  --quiet HEAD` passes on both, and `git status --porcelain` is empty on both.
- Production, harnesses, launchers, hooks and `data/`.
- TASKS.md (Planner-modified, left unstaged), REVISION_CONTEXT.md and the brief.

Nothing named `p515_s4*` was run.

## Blocked on Planner

v3 is generated, verified (0 failures) and committed. Under ruling 3 its zip is the submitted file. One wording point
remains before upload, and it is outside ruling 3. The prose itself is correct against round 9c (see Found).

1. **The generated half of `README.md` still uses the old a87e195 appendix labels.** This was v2's unexpected finding 3,
   and v3 has not changed it. These labels come from `COLUMN_SPECS` in the generator, not from the prose. Examples:
   - "Table E.20 (tab:appE_claims)" through "Table E.24". There are 16 occurrences of `Table E.2`.
   - "Appendix B (fig:cs3_market_prices)", "Appendix C (tab:cs3_tn_res_generators)" and "Appendix D (...)" for the
     B–D files.
   - "not printed in Appendices C-D" on the 2030 and 2035 capacity columns.

   In round 9c (`main_46c9d77_round9c.tex`), Appendix B is the *Evaluation Record*, and there is no Appendix C or D.
   The prose above the generated section calls the same files "Table B.10", "Table B.11", "Table S4–S6" and
   "Section S1–S3". A reader of the submitted README would therefore find market prices described as "Appendix B",
   and the same table under two names.
   - **(a)** Submit v3 as it is. The prose is correct, but the generated section's labels are stale.
   - **(b), recommended if it fits before the 11-15 freeze:** authorise rewording the `table` strings and the year-role
     text of `COLUMN_SPECS` to round-9c naming. That changes the README bytes only; no CSV changes. Then regenerate
     as v4 with predecessor `e6f5ce6a`, add a V9 entry for v4, and rerun the verifier.
     - Cost: one generator commit, about 1 min of generation and about 5 s of verification (the v3 verifier run took
       3.5 s), all zero-solve.
     - It needs the naming map (E.20 → B.10, E.22 → B.11, E.21/E.23/E.24 → S4/S5/S6, B–D → S1–S3) confirmed by the
       expert, because the wording is the expert's.
2. **A minor wording point, for the expert to take or drop.** The prose describes `E20_E21_groups.csv` as "the group
   headings of the comparison and evaluation tables, as printed". The file's column is `title_latex`. It holds the LaTeX
   source verbatim, with `$…$` and `\ref{tab:res_value_ladder}`-style references, not the rendered text. The E.21
   groups are those of Table S4, which the paper no longer prints. The generated section says "LaTeX source,
   verbatim". If the expert wants the prose to agree, "as printed" → "in the LaTeX source of the tables" would do it.
   This could go into the same v4 as item 1, or be dropped.

No other blocker. The verifier needed version-keyed changes as well as the generator, and that went beyond "the
generator change". They are listed under Changed for the Planner's acknowledgment, and no decision is needed.

## Changed

| commit | content |
|---|---|
| **`5c4074d2`** | `manuscript_review/gen_supplement.py` (sha256 `546f4dc412e4317e…`) and `manuscript_review/verify_supplement.py` (sha256 `e280d9648bef063c…`), committed **before** any canonical generation (+54 / −27) |
| **`13076571`** | `manuscript_review/supplement_v3/`: the zip, the record, the verifier result, both launch logs and `SHA256_INVENTORY.txt`, in a new directory created by `mkdir` in the launch command and written once |
| this commit | this report |

**Generator (`gen_supplement.py`).** It now selects the prose by version:
- `readme_prose_rel(version)` returns `supplement_README_prose_v1.md` for version ≤ 2 and `supplement_README_prose_v2.md`
  for version ≥ 3. `README_PROSE_REL` (v1) is kept unchanged, and `README_PROSE_V2_REL` is added.
- `readme()`, the committed-clean input list and the record's `readme` fields use the selected path.
- The record's `stage` string now also names W189a-v3. The record lies outside the zip.
- The docstring was updated.
- No other change. `COLUMN_SPECS`, the CSV builders and the zip writer are untouched.

**Verifier (`verify_supplement.py`).** Without these changes it would have failed v3 on V7 and V9, and failed V4 for
every version:
- **V7** reads the prose through the same version map. It also requires that the record's `prose_prepended` path
  equals that prose.
- **V9** uses `V9_DECLARED_BY_VERSION`:
  - v2's declared set is unchanged (`SHA256SUMS.txt` and the two RES CSVs changed, `README.md` added,
    `README_PLACEHOLDER.md` removed).
  - **v3: changed = [`README.md`, `SHA256SUMS.txt`], added = [], removed = []**.
  - The RES columns added are `V9_RES_COLUMNS_ADDED_BY_VERSION`: v2 `[capacity_2030_MW, capacity_2035_MW]`, v3 `[]`.
  - Control C20 takes the per-version declared set.
- **V4**: `verify_appE.py` at `441722c9` now has 40 checks (Addendum 91 added 7 reduced-mode checks), and the
  expected count was a literal 33. It is now `VERIFY_APPE_PASS = 40`. Without this, V4 would fail for any version
  verified at HEAD.

**Commands.** All were run from the repo root with the canonical interpreter
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`, attached, with both streams captured. None used nohup,
screen or `&`. `<S>` is the session scratchpad, outside the repository.

```
# before the edit, at 441722c9 (baselines):
python -u manuscript_review/gen_supplement.py --out-dir <S>/pre_v1                       > <S>/pre_v1.log 2>&1   # exit 0
python -u <S>/drv/emulate_version.py <S>/pre_v2 2                                        > <S>/pre_v2.log 2>&1   # exit 0
# after the edit, uncommitted (scratch only):
python -u manuscript_review/gen_supplement.py --out-dir <S>/post_v1                      > <S>/post_v1.log 2>&1  # exit 0
python -u <S>/drv/emulate_version.py <S>/post_v2 2                                       > <S>/post_v2.log 2>&1  # exit 0
python -u manuscript_review/gen_supplement.py --force-new-version --out-dir <S>/trial_v3 > <S>/trial_v3.log 2>&1 # exit 0
python -u manuscript_review/verify_supplement.py --supplement-dir <S>/trial_v3          > <S>/trial_v3_verify.log 2>&1  # PASS
python -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v2 \
       --out-dir <S>/reverify_v2                                                         > <S>/reverify_v2.log 2>&1      # PASS
# at 5c4074d2:
python -u manuscript_review/gen_supplement.py --force-new-version --out-dir <S>/det_A    > <S>/det_A.log 2>&1    # exit 0
python -u manuscript_review/gen_supplement.py --force-new-version --out-dir <S>/det_B    > <S>/det_B.log 2>&1    # exit 0
mkdir manuscript_review/supplement_v3 && \
  python -u manuscript_review/gen_supplement.py --force-new-version > manuscript_review/supplement_v3/gen_launch.log 2>&1   # exit 0
python -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v3 \
  > manuscript_review/supplement_v3/verify_launch.log 2>&1                                                                 # exit 0, VERDICT: PASS
```

`emulate_version.py` is a scratch driver; it is not committed.
- It imports the repository's `gen_supplement.py` as a module, so the module's own `SolveProfileGuard(permitted=())`
  and pickle block are installed at import.
- It pins `resolve_out_dir` to (scratch dir, version 2, predecessor = v1's record) and calls `main()`.
- That is the only way to regenerate v2's bytes into scratch once v2 exists, because `--force-new-version` now targets
  v3. The v2 regeneration went to scratch only, as ordered.

| file (`manuscript_review/supplement_v3/`) | bytes | sha256 |
|---|---|---|
| **`SRP1_supplement_v3_e6f5ce6a.zip`** (20 entries) | 1,326,506 | **`e6f5ce6a5ef9e11c25fb4ef0ee9c273f36fff160bf735e8a56a77c91d903155d`** |
| `supplement_v3_record.json` | 14,520 | `9d8312260c0ad6e6b5b94a6010b315a103c0c5fc635dea0482505f15f16c465e` |
| `verify_supplement_v3_result.json` (gate_result_io) | 29,765 | `3f2235fc8538b55f6e2cf300440e3766d02438b3841db9999e7a65fe0859b4e3` |
| `gen_launch.log` (exit 0) | 2,026 | `cea3cf048f826caa5c9a27c1eb5eba5d2cd4f9b8886b2f3efaa370cd22c6cf7a` |
| `verify_launch.log` (exit 0) | 5,917 | `54e2fa65383b3358bd724f723c66784492f2b66bc73f329deccd133e53be6ee4` |
| `SHA256_INVENTORY.txt` | — | lists the five above |

**Record provenance.**
- `git_head` is `5c4074d2…`, `generator.committed_clean` is true, and `readme.prose_prepended` is
  `manuscript_review/supplement/supplement_README_prose_v2.md`.
- The prose sha256 is `ca88265b4d0a781d…`, and `prose_committed_clean` is true (the prose was committed at `441722c9`).
- The predecessor is version 2: `manuscript_review/supplement_v2/supplement_v2_record.json`,
  `SRP1_supplement_v2_864a5779.zip`, sha256 `864a57797f7b65ce004399f602f166d2d423336acd4592c5e57a15d2f1c0213a`.
- The guards counted 0 solves, 0 process launches and 0 / 0 pickle calls.

**Objective convention** (unchanged from v1 and v2, whose Appendix-E CSVs v3 carries byte for byte):
- Q = `gross_operational_cost`, with the contracted interface settlement excluded.
- net = Q − terminal salvage credit.
- value = Q(0) − Q(x).
- F = Q + I.
- Amounts are in k EUR as printed.

## Found

| check (task item) | expected | observed |
|---|---|---|
| verifier on v3 | PASS, 0 mismatches | **VERDICT PASS, 0 failures**; 21 of 21 negative controls caught exactly |
| V1 Appendix E vs frozen tables | 1,623 entries, 0 MISMATCH | **1,623 entries: 1,443 MATCH, 0 MISMATCH**, 180 not-frozen agree, 0 disagree, 0 mapping problems; per table E.20 639 / E.21 854 / E.22 18 / E.23 49 / E.24 63; E.20 note numbers MATCH, MATCH |
| V5 B–D vs case files | 70,851 cells, 0 differences | **70,851 cells in 2,700 rows, 0 differences** |
| V8 paper Table 2 | 48/48 | **48 printed aggregates, 0 differences** (a87e195 and the round-9b copy); the multi-scenario sentence holds for all generators (0 violations) |
| Table 2 in round 9c | (not in the task; checked) | the `tab:res_capacity_summary` block of `main_46c9d77_round9c.tex` is byte-identical to round 9b's (block sha256 `0f99510c…` for both); the multi-scenario sentence is present once |
| `D_ieee33_tn_connection.csv` | unchanged from v2; node 7 at ±100 | **byte-identical**, sha256 `f954ad5057ee2c77…` in both zips (V9 and an independent unzip + `shasum`); node 7 (`case33_2`) Pmax 100 / Pmin −100 / Qmax 100 / Qmin −100, nodes 5 and 9 at ±250, Vg 1 |
| v2 → v3 entry diff | README.md and SHA256SUMS.txt only | **exactly README.md and SHA256SUMS.txt** (V9: 18 identical, 0 added, 0 removed; independently `diff -rq` of the two unzipped trees lists the same two files) |
| determinism | 3 generations byte-identical, in scratch | **4 byte-identical** `e6f5ce6a…`: the scratch trial (pre-commit), scratch `det_A` and `det_B` at `5c4074d2`, and the canonical run (`cmp` identical to det_A). The det_A and canonical records differ only in `out_dir` and `canonical` |
| v1 / v2 bytes of the full mode unchanged | unchanged | **v2**: regenerated into scratch through the driver, before and after the edit, **`864a5779…` both times** = the committed v2 zip (`cmp`). **Full mode without `--force-new-version`** ("version 1" bytes of the current generator): `e8709fa9…` before and after the edit (`cmp` identical). This was never v1's `a94d8193`; see Not confirmed |
| v2 under the new verifier | (regression check) | `--supplement-dir supplement_v2 --out-dir <scratch>`: **PASS, 0 failures**. V7 reads prose v1; V9's v1 → v2 declared set is unchanged; V4 reports verify_appE 40 PASS |
| README: no "version 1" | none | **0** occurrences of "version 1", "version 2" or "v1" in v3's README.md. Line 3 now says the version number and hash are given in the generated section and in `SHA256SUMS.txt`; the generated heading reads "Generated section (supplement version 3)" |
| README: cell count | 70,851 | line 9: "the 70,851 case-file cells", which equals V5's count. The 1,623 entries also equal V1's count |
| README: `E_table_notes.csv` | described as the file is | the prose says "the numbers printed in the notes of the manuscript's Tables B.10 and B.11 and of Table S5, each with its unit and meaning". The file holds **15 rows**, with columns `table,item,value,unit,description`: 4 from E.20, 2 from E.22, 9 from E.23. **Consistent**, given the 9c mapping E.20 = B.10, E.22 = B.11, E.23 = S5 (see Not confirmed). Two of the E.23 values are block lists ("2035 Summer"; the 8-block list), not numbers. That is a nicety, not an error |
| README: `E20_E21_groups.csv` | described as the file is | "the group headings of the comparison and evaluation tables, as printed". The file holds 10 rows (E.20 a–d, E.21 a–f) in `title_latex`: the LaTeX source, verbatim. Mostly right; "as printed" is loose (Blocked on Planner, item 2) |
| naming "Table S5", not "Section S5" | Table S5 | README row "Table S5 — static arrangements". Round 9c has **0** "Section~S5" (or S4 / S6) and "Table~S5" at l. 1353, 1627 and 2120. Every 9c pointer (Section S1, Sections S2 and S3, Sections S1–S3, Tables S4, S5, S6) has a README row |
| prose prepended byte for byte | yes | V7: `prose_prefix_equal` True; the prose sha256 `ca88265b…` equals HEAD and the record; 18 column tables equal the 18 CSV headers; 0 `PLACEHOLDER` |
| zero solves | 0 | generator and verifier `SolveProfileGuard(permitted=())`, plus the W180a module's guard: **0 solves, 0 process launches, 0 blocked**; pickle load / loads 0 / 0 in all three |

**Prose v1 → v2 as carried in the zip.** The README diff between v2 and v3 is exactly the five prose hunks of
`diff supplement_README_prose_v1.md supplement_README_prose_v2.md`, plus the generated heading "version 2" →
"version 3":
- line 3: "Supplement version 1." → the version-independent sentence;
- line 9: 70,743 → 70,851;
- the S2 row: the seven years named;
- the S4 row: "(+ `E20_E21_groups.csv`)" dropped;
- the `E_table_notes.csv` row rewritten, and a new `E20_E21_groups.csv` row.

## Unexpected findings

1. **The stale appendix labels in the generated README section** (Blocked on Planner, item 1). They were carried over
   from v2 and are not new, but the submission makes them visible.
2. **Since `441722c9`, `verify_appE.py` spawns child processes.** It runs `gen_appE.py` twice through `subprocess.run`,
   in full and `--reduced` mode, into two new `tempfile.mkdtemp()` directories.
   - V4 runs `verify_appE.py`, and the in-process `SolveProfileGuard` does not see into those children.
   - `gen_appE.py` imports only `csv`, `sys` and `os`, so a child cannot reach a solver.
   - The zero-solve claim therefore still holds, but for the children it rests on that import list rather than on an
     armed guard. That was not edited (it is the Planner's file).
   - V4 also leaves three temp directories in the system temp dir per verifier run (one before 441722c9, as v2's
     finding 5 noted).
3. **v1's bytes are not reproducible from the current generator in any mode.** This has been so since `80778c12` and
   is not new. That generator made the seven RES years and README.md unconditional. Without `--force-new-version`, it
   writes "version 1"-labelled bytes `e8709fa9…`, not v1's `a94d8193…`. My change keeps those bytes as they were
   (checked above). v1 itself is reproducible only with the generator at `5d7955da`.

## Not confirmed

- **The compiled table numbers B.10 and B.11.** No TeX is installed on this machine. They were inferred from round 9c's
  source:
  - 9 numbered main-text tables (l. 1223–1679; the three nomenclature tables are offset by `\addtocounter{table}{-1}`);
  - then `tab:appE_claims` (l. 2161) and `tab:appE_neighbours` (l. 2264) in the second appendix, `\section{Evaluation
    Record}`;
  - elsarticle continues the table counter across `\appendix`, which gives B.10 and B.11.

  The subsections B.2 (Comparisons) and B.3 (Neighbours) match the prose. Confirmation needs the Overleaf PDF.
- **Round 9c / Overleaf `46c9d77` as a verifier source.** V2, V4 and V6 stay pinned to a87e195, and V8 to a87e195 plus
  round 9b, as in v2. Only Table 2 and the S-pointers were checked against `main_46c9d77_round9c.tex` (read-only, by
  `awk` / `grep`).
- **Cross-machine byte identity.** Checked on this machine only: Python 3.11, with the pandas / openpyxl versions in the
  record.
- **Whether v3 or a v4 is the uploaded file.** That depends on Blocked on Planner, item 1.
