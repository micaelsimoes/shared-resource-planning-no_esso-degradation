# P5.15 W189a: supplement v2 (Addendum 90 ruling 7)

**Worker, 2026-11-10.** Zero solves, zero model builds, zero pickle loads. This was read and not edited: v1's files (`manuscript_review/supplement/`, the README prose included), `gen_appE.py`,
`verify_appE.py`, `export_W160/`, the frozen JSON, the case files, the Overleaf clone (`git show` only) and production. Nothing
outside the files listed under Changed was edited or staged. That includes TASKS.md (Planner-modified, left unstaged),
REVISION_CONTEXT.md and the brief.

## Blocked on Planner

v2 is generated, verified (0 failures) and committed. Ruling 7 says the zip is the file to submit. **Before it is
submitted, the README prose needs the expert, because it describes v1.** The prose was prepended byte for byte, as
ordered, and none of it was edited.

1. **The prose says "Supplement version 1."** (line 3 of `README.md`). The archive is
   `SRP1_supplement_v2_864a5779.zip`, and the generated section below the prose is headed "supplement version 2".
   The prose also says the verifier found "0 mismatches over … the 70,743 case-file cells". In v2 the verifier compares
   **70,851** case-file cells, with 0 mismatches. The difference, +108, is the two new year columns: 18 generators × 2
   years × 3 cells. The figure of 1,623 entries is still correct.
   - **Option (a):** submit v2 as it is. The README then contradicts itself on the version and on the cell count.
   - **Option (b), recommended:** the expert delivers `supplement_README_prose_v2.md` with "version 3" (or wording that
     does not depend on the version) and the v3 count, 70,851. The run is then `--force-new-version` → v3, with
     predecessor `864a5779`. This costs one zero-solve generation, about 1 min, plus the verifier, about 5 min. The
     generator reads the prose path from `README_PROSE_REL`, so v3 needs a one-line generator change and a commit.
     Option (b) needs that authorisation.
2. **What the prose says `E_table_notes.csv` holds is not what it holds.** The prose's last table row says the file
   holds "the footnote marks used in the evaluation tables and their meaning". The file actually holds the 15 *numbers*
   printed in the notes of E.20, E.22 and E.23: the uncertified-form bars and multiples, the evaluation-16 Δ, the E.23
   spreads, the band and the sweep counts. The marks are the explicit yes/no columns of each CSV. This is an expert
   wording point, and the fix could go into the same prose v2.
3. **Round 9b points to a "Section S5" that the README does not have.** The pointer is at l. 1353, "detailed in
   Section~S5 of the supplementary material", and at l. 1627, "(Section~S5 of the supplementary material)". The README
   maps "Table S5 — static arrangements" to `E23_benchmark.csv`, and l. 2120 of round 9b also says "Table~S5". Every
   other S-pointer has a row in the README: S1–S3, S4 and S6. This is for W189 or the expert. No supplement file is
   missing; only the name differs.

## Changed

- **`80778c12`**: `manuscript_review/gen_supplement.py` (sha256 `e9f352499af8dcee…`) and
  `manuscript_review/verify_supplement.py` (sha256 `6c33492032b42c91…`). Both were committed **before** the real run.
  The v2 record shows `git_head` `80778c12…` and `generator.committed_clean: true`. The prose input shows
  `readme.prose_committed_clean: true`.
- **This commit**: this report and the whole of `manuscript_review/supplement_v2/`, in a new directory created by
  `mkdir` in the launch command and written once. v1's directory is unchanged against HEAD: `git diff --quiet HEAD --
  manuscript_review/supplement` passes and `git status` is clean there.

| file (`manuscript_review/supplement_v2/`) | bytes | sha256 |
|---|---|---|
| **`SRP1_supplement_v2_864a5779.zip`** (20 entries) | 1,326,199 | **`864a57797f7b65ce004399f602f166d2d423336acd4592c5e57a15d2f1c0213a`** |
| `supplement_v2_record.json`: hash inventory of every entry and of the zip, input hashes (the prose included), predecessor, guards | 14,470 | `c25c5ab248859a9f0cc1e5c47aecba4076a50a6e9234187a490e2eabb67b7914` |
| `verify_supplement_v2_result.json`: verifier output, written through gate_result_io | 30,133 | `e2f1317e7e9a6aa7f485f9c5187a59af9fe655938e6f837f727e7fabe5e7eeea` |
| `gen_launch.log`: generator stdout + stderr, exit 0 | 2,018 | `a7c0693780ad540053cd3e65fcb21411790595e513c4eca7ca2af22facc987bf` |
| `verify_launch.log`: verifier stdout + stderr, exit 0 | 6,095 | `294c68c8cfbbd1a1a7f82c3d61114452aaeaab08666a5929bad0e98da33fdd0e` |

**Predecessor recorded** (`supplement_v2_record.json` → `predecessor`): version 1, record
`manuscript_review/supplement/supplement_v1_record.json`, zip `SRP1_supplement_v1_a94d8193.zip`, sha256
`a94d8193047b7e688205549945ba38b5e0372913c2e0b59fc0c0627ba5c90726`. The verifier re-hashed the v1 zip and found it equal
to both the v2 record and v1's own record.

**What the generator change does** (`gen_supplement.py`, +72 / −37 lines):
- **(iii) RES years.** `C_tn_res_generators.csv` and `D_adn_res_generators.csv` now carry
  `RES_YEARS = ALL_CASE_YEARS` = 2025, 2028, **2030**, 2031, 2034, **2035**, 2037, in chronological order. The values are
  read from the same JSON files and the same `Pmax` field as before. Each capacity column's README definition now names
  its role: single-scenario, multi-scenario, or both.
- **(iv) README.** The entry `README.md` replaces `README_PLACEHOLDER.md`.
  - It consists of the bytes of `manuscript_review/supplement/supplement_README_prose_v1.md` (sha256
    `bc6e5a4c702bdba6…`, committed at `e3c81915`), followed by the generated section.
  - The generator refuses prose that does not end in a newline or that contains `PLACEHOLDER`.
  - The prose is listed among the inputs that a canonical run requires to be committed and clean.
  - The generated section is headed "Generated section (supplement version 2)". It holds the acceptance-rule parameters
    read from the code (τ, the frozen `objective_convention`, P_MAX and every numeric `settling_criterion_v6.constants(30)`
    entry, the same content as v1), the "Files and columns" tables, and a closing paragraph on base profiles. That
    paragraph was v1's "[EXPERT PROSE PLACEHOLDER: figures of Appendices B-D]" with the placeholder mark removed and
    the reference to the figures dropped (round 9b no longer has them). Ruling 7 (i) is unchanged.
  - Every `[PLACEHOLDER: …]` heading of v1 is gone. The verifier finds 0 `PLACEHOLDER` strings in `README.md`.
- **Scratch emulation for determinism.** `--out-dir DIR --force-new-version` (previously refused as exclusive) now
  writes the next version's bytes, with the same version number and predecessor, into a scratch DIR outside the
  repository. Without that, a scratch run would produce version-1 README bytes and could not be compared with v2.
- **Record.** New fields: `readme`, with the entry, the prose path, its sha256 and its committed-clean flag, and
  `res_capacity_years`. The `stage` string now names W189a.

**What the verifier change does** (`verify_supplement.py`, +270 / −17 lines):
- **V5** checks every capacity column against the case files. It also requires the capacity columns to be exactly
  the expected years: v1, the 5 printed years; v2 and later, all 7.
- **V7** is versioned. For v2 and later it requires:
  - `README.md` begins with the prose byte for byte;
  - the prose sha256 equals the record's, and the prose equals `HEAD:` that path;
  - there is no `README_PLACEHOLDER.md` entry and no `PLACEHOLDER` string;
  - τ and the objective line are present, and every column table equals its CSV header.
- **New V8 (Table 2).** For every network × technology × year in 2025, 2030 and 2035, the sum of the supplement's
  capacities must equal the printed aggregate of `tab:res_capacity_summary`. The check is run against both main.tex at
  a87e195 and `main_a87e195_round9b.tex`. V8 also checks the sentence beside the table, generator by generator: "the 2025
  capacities also apply in 2028, the 2030 capacities apply in 2031 and 2034, and the 2035 capacities apply in 2037".
- **New V9 (predecessor diff).** V9 compares every entry against the predecessor zip. The differing set must equal
  the declared set. Each RES CSV, restricted to v1's columns and re-serialised, must equal v1's bytes, with exactly
  `capacity_2030_MW` and `capacity_2035_MW` added. `D_ieee33_tn_connection.csv` must be byte-identical.
- **Controls C16–C21** were added; they are listed below.

**Commands.** All were run from the repo root with the canonical interpreter, attached, with both streams captured.
None used nohup, screen or `&`.

```
# determinism, scratch, at 80778c12, before the canonical run:
.../python -u manuscript_review/gen_supplement.py --force-new-version --out-dir <scratch>/w189a_detA  > <scratch>/w189a_detA.log 2>&1  # exit 0
.../python -u manuscript_review/gen_supplement.py --force-new-version --out-dir <scratch>/w189a_detB  > <scratch>/w189a_detB.log 2>&1  # exit 0
# canonical:
mkdir manuscript_review/supplement_v2 && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/gen_supplement.py --force-new-version \
  > manuscript_review/supplement_v2/gen_launch.log 2>&1                                                      # exit 0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/verify_supplement.py \
  --supplement-dir manuscript_review/supplement_v2 > manuscript_review/supplement_v2/verify_launch.log 2>&1   # exit 0, VERDICT: PASS
```

**Objective convention**, as in v1's Appendix E CSVs, which are unchanged: Q = `gross_operational_cost`, with the
contracted interface settlement excluded; net = Q − terminal salvage credit; value = Q(0) − Q(x); F = Q + I; amounts in
k EUR as printed.

## Found

### Verifier: PASS, 0 failures, 21 of 21 controls caught exactly (`verify_supplement_v2_result.json`)

| check | outcome |
|---|---|
| V0 zip / inventories | sha256 = the record's (`864a5779…`); 20 entries, sorted, all dated 1980-01-01, mode 0644, STORED; SHA256SUMS.txt and the record's member list equal the zip |
| **V1 Appendix E vs frozen tables** | **1,623 entries, 0 MISMATCH**: 1,443 MATCH, 180 not-frozen-agree, 0 disagree, 0 mapping problems; per table E.20 639, E.21 854, E.22 18, E.23 49, E.24 63; the E.20 note numbers MATCH. Identical to v1 |
| V2 Appendix E vs main.tex at a87e195 | 1,633 cells, 0 differences, 0 mark differences; declared difference E.24 `over_budget` = yes on [39, 6] (kept, ruling 7 (ii)) |
| V3 `over_budget` vs W182d | E.21 {5, 6, 9, 10, 11, 38, 39}; E.24 {6, 39}; 0 inconsistent E.20 rows |
| V4 generator identity | gen_appE.py output = committed `appendixE_evaluation_record.tex`, once in main.tex a87e195, l. 2556–2946; `verify_appE.py` 33 PASS 0 FAIL |
| **V5 B–D vs case files** | **70,851 cells in 2,700 rows, 0 differences**. v1 had 70,743; the +108 are the new 2030 / 2035 cells (18 generators × 2 years × capacity, node, type) |
| V6 C–D vs printed tables at a87e195 | 905 printed cells, 0 differences. Only the printed years are compared, as in v1. The observation on `tab:cs1_ieee33_generators` (node 7 = ±100) stands |
| **V7 README.md** | 18 column tables = 18 CSV headers; prose sha256 `bc6e5a4c…` = HEAD = record; prepended byte for byte; `README_PLACEHOLDER.md` absent; 0 `PLACEHOLDER` strings |
| **V8 Table 2** | **48 printed aggregates (24 per source × 2 sources), 0 differences**; header years 2025 / 2030 / 2035 and 8 rows in each source; the multi-scenario sentence is printed in both and holds for all 18 generators (0 violations) |
| **V9 vs v1** | v1 zip sha256 `a94d8193…` = record; **16 entries byte-identical**; changed = SHA256SUMS.txt and the two RES CSVs; added = README.md; removed = README_PLACEHOLDER.md, the declared set exactly; both RES CSVs keep v1's columns bytewise (order kept, only `capacity_2030_MW` and `capacity_2035_MW` added); `D_ieee33_tn_connection.csv` identical |
| guards | generator and verifier `SolveProfileGuard(permitted=())`, plus the W180a module's own guard: 0 solves, 0 process launches; pickle load / loads 0 / 0 |

Table 2, as the supplement reproduces it (MW). Each sum was computed by V8 from the CSV columns and equals the print
at a87e195 and at round 9b:

| network | technology | 2025 | 2030 | 2035 | generators |
|---|---|---|---|---|---|
| TN | Wind | 85 | 130 | 150 | 3 |
| TN | Solar PV | 47 | 75 | 95 | 3 |
| ADN at node 5 | Wind | 40 | 50 | 50 | 1 |
| ADN at node 5 | Solar PV | 43 | 60 | 90 | 3 |
| ADN at node 7 | Wind | 80 | 90 | 90 | 2 |
| ADN at node 7 | Solar PV | 20 | 30 | 50 | 2 |
| ADN at node 9 | Wind | 20 | 50 | 50 | 1 |
| ADN at node 9 | Solar PV | 90 | 90 | 120 | 3 |

**Negative controls, all in memory.** C1–C15 are as in v1 and were each caught exactly where altered. The new ones:
- **C16.** TN generator 4, `capacity_2030_MW` 30 → 31. V5 catches it at `case9 gen 4 2030`, and V8 at TN / Wind / 2030 in
  both sources.
- **C17.** ADN node 9, generator 5, `capacity_2035_MW` 40.0 → 40.5. V5 catches it. V8 does not, by construction: Table 2
  prints integers, so 120.5 rounds to 120. This control records V8's resolution, ±0.5 MW per aggregate; V5 is the
  exact check.
- **C18.** The `capacity_2035_MW` column is dropped from `D_adn_res_generators.csv`. V5's capacity-column check fails
  for each of the three ADNs.
- **C19.** One byte of the prose in README.md is changed. V7 reports `prose_prefix_equal` False, and the check fails.
- **C20.** `D_ieee33_tn_connection.csv` node 7 is set to the submitted ±250. V9 lists the file as an undeclared change,
  and `tn_connection_identical` goes False.
- **C21.** One v1 column of `C_tn_res_generators.csv` is changed (generator 5, 2028). V9's projection goes False for
  that file only.

### v1 → v2 entry diff (V9; repeated independently with unzip + cmp + cut in scratch)

| entry | v1 sha256 (16) | v2 sha256 (16) | status |
|---|---|---|---|
| `README_PLACEHOLDER.md` | `26aa291834ccaa4c` | — | removed (ruling 7 (iv)) |
| `README.md` | — | `6edef05bd0050f59` | added: prose (byte for byte) + generated section |
| `SHA256SUMS.txt` | `878fde702edcc9c7` | `22638d37bc6fc180` | changed (it lists the entries above) |
| `appendices_B_D/C_tn_res_generators.csv` | `cecb82e430843e98` | `345ea155c04d078c` | changed: 6 × 8 → 6 × 10; `cut -d, -f1-5,7,8,10` of v2 = v1 byte for byte |
| `appendices_B_D/D_adn_res_generators.csv` | `6af15d85b6c6b57c` | `c16ffa24c122a25e` | changed: 12 × 10 → 12 × 12; `cut -d, -f1-7,9,10,12` of v2 = v1 byte for byte |
| the other 16 entries (all of `appendix_E/`, the other 9 `appendices_B_D/` files) | | | **byte-identical** |

Nothing else differs. The record JSON lies outside the zip; its new fields are listed under Changed.

The generated README section differs from v1's placeholder file only in these places (compared with `diff` against
the bytes after the prose):
- the v1 title and the placeholder paragraph are replaced by the "Generated section (supplement version 2)" heading
  and its one-paragraph description;
- the three `[PLACEHOLDER: …]` headings are merged into "Acceptance-rule parameters (read from the code)", with the
  same bullet lines;
- the two RES files' source line and capacity rows now cover 7 years, each with its role;
- the closing placeholder paragraph is rewritten as described under Changed.

### `D_ieee33_tn_connection.csv`: byte-identical to v1

sha256 `f954ad5057ee2c7775d4dfd288f82c88f5aa7d5d0795abdd9323479df9a585a8` in both zips (`cmp` identical). It has one row
per ADN: node 5 at ±250, **node 7 (`case33_2`) at Pmax 100 / Pmin −100 / Qmax 100 / Qmin −100**, node 9 at ±250,
with Vg 1 in each row. V5 checks it against the case files with 0 differences.

### New capacity columns (MW, from the case files)

| | gen (node) | 2030 | 2035 |
|---|---|---|---|
| TN | 4 (4) Wind, 5 (6) Wind, 6 (8) Wind | 30, 60, 40 | 50, 60, 40 |
| TN | 7 (4) PV, 8 (6) PV, 9 (8) PV | 20, 30, 25 | 20, 50, 25 |
| ADN 5 | 2 (6) Wind; 3 (23), 4 (25), 5 (18) PV | 50; 25, 20, 15 | 50; 25, 30, 35 |
| ADN 7 | 2 (6), 3 (23) Wind; 4 (25), 5 (18) PV | 50, 40; 20, 10 | 50, 40; 20, 30 |
| ADN 9 | 2 (6) Wind; 3 (25), 4 (23), 5 (18) PV | 50; 30, 30, 30 | 50; 30, 50, 40 |

In every generator, 2030 = 2031 = 2034 and 2035 = 2037 (V8). This confirms v1's observation.

### Determinism: four runs byte-identical

| run | zip sha256 |
|---|---|
| scratch `w189a_trial1` (pre-commit, same generation code; differs from the commit only in docstrings and two blank lines) | `864a5779…` |
| scratch `w189a_detA` at `80778c12` | `864a5779…` |
| scratch `w189a_detB` at `80778c12` | `864a5779…` (`cmp` identical to detA) |
| canonical `manuscript_review/supplement_v2/` at `80778c12` | `864a5779…` (`cmp` identical to detA) |

The detA and canonical record JSONs are equal apart from `out_dir` and `canonical`. The verifier also passed (0
failures, 21 controls) on the pre-commit trial, in scratch.

## Unexpected findings

1. **The prose is stale for v2** (Blocked on Planner, items 1–2): it says "version 1", gives 70,743 cells, and
   describes `E_table_notes.csv` as holding marks.
2. **The round-9b "Section S5" pointers** have no README row (Blocked on Planner, item 3).
3. **The generated column section still uses a87e195's appendix labels for the B–D files.** Examples: "Appendix C
   (tab:cs3_tn_res_generators)" and "Appendix B (fig:cs3_market_prices)".
   - Round 9b no longer prints those tables and figures; its pointers are Sections S1–S3, which the prose uses.
   - The labels come from v1's `COLUMN_SPECS`. They were not changed, because that is outside ruling 7.
   - The new year-role text, "not printed in Appendices C-D", follows the same a87e195 reference.
   - If the expert wants S-numbering in the column section, the change touches README bytes only, no CSV, and can go
     with prose v2.
4. **V6 still checks against the a87e195 print of Appendices C–D.** Those tables have left the paper in round 9b. V6
   now checks the supplement against the submitted-era print, which is still a valid check of the values. V8 is the
   check against a table that remains in the paper.
5. As in v1, `verify_appE.py` (run by V4) leaves its own `tempfile.mkdtemp()` file in the system temp directory. That is
   its own code, and it was not edited.

## Not confirmed

- **Cross-machine byte identity.** As for v1, this was checked on this machine only: Python 3.11, pandas 3.0.0,
  openpyxl 3.1.5.
- **The Overleaf clone has a newer commit, `46c9d77`** (the round-9b paste, presumably). Only `tab:res_capacity_summary`
  was compared there (`git show`, read-only), and it is identical to a87e195 and to the round-9b copy. V2, V4 and V6 are
  pinned to a87e195, as in v1. Checking the supplement against `46c9d77` is W189's task and was not done here.
- **The submitted zip.** Whether v2 or a v3 with revised prose is the file submitted depends on Blocked on Planner,
  item 1.
- **No compile.** No TeX is installed on this machine, and none was needed.
