# P5.15 W189a-v5: supplement v5 (Amendment 2 to Addendum 91, ruling 1, with the expert's three re-ruled strings)

**Worker, 2026-11-12.** Zero solves, zero model builds, zero pickle loads. Same procedure as v4
(`P5_15_W188B_SUPPLEMENT_v4.md`). The ruling-1 wording was implemented at `fb886ea4`. This task re-rules three strings
in it and makes the canonical run.

**Submitted file (v5 is the submitted file, per Amendment 2):**
`manuscript_review/supplement_v5/SRP1_supplement_v5_d2fa2172.zip`
- sha256 `d2fa21725399d4f5666f7ba38f171b90f24f1a965a2e679607f92b378578cdf9`
- 1,326,229 bytes, 20 entries; predecessor v4 `d175ea0e`.

## Blocked on Planner

Nothing blocks the upload. v5 is generated, verified (PASS, 0 failures) and committed, and its README carries the
three strings exactly as ruled. One point is for the expert to take or drop (Unexpected findings, item 1); it changes
no data.

## History: the first Worker's hold and the superseded trial

- **The first Worker held before the canonical run.** It implemented ruling 1 at `fb886ea4` (`V5_WORDING`,
  `V5_LABEL_RX`, `V10_EDITS_BY_VERSION[5]` / `V10_V5_WORDING`, V12, controls C23–C25) and produced a scratch trial.
  It stopped because one ruled string was false. Ruling 1 (c) wrote "printed mark b of Table S5" → "the over-budget
  mark b of Table S5". But in Table S5 (old E.23), mark b is "the start reported in Table res_benchmark"
  (`appendixE_evaluation_record.tex` l. 326, confirmed by the Planner), not a budget mark.
- **Trial `SRP1_supplement_v5_1d0c39fe.zip`** (sha256 `1d0c39feca6992a56d7369b5010142f43adca63bc93baa4622dc1735dbdc8b5a`,
  scratch only, verifier PASS, deterministic ×4) is **superseded and was never canonical**. It was never written into
  the repository.
- **Nearest-wording substitutions the first Worker recorded.** These are kept unchanged here, under ruling 1's last
  sentence ("apply the nearest wording and record it"):
  - The generated text quotes `"---"` with **double quotes**, where the ruling writes `'---'`. The (a) entry matches
    the double-quoted form and keeps the double quotes.
  - The generated text writes **"Sections S2-S3" with an ASCII hyphen** (from v4's mapping), where the ruling writes
    "S2–S3". The (c) entry matches the hyphen form. The replacement strings are the ruling's, with en dashes (U+2013).

## The expert's ruling (2026-11-12, relayed by the author to the Planner), verbatim

> 1. `reported_start` row: "mark b of Table S5: the start reported in the main-text benchmark table (not the budget mark)". This replaces the ruled "the over-budget mark b of Table S5", which contradicted the record: in Table S5 mark b is the start reported in Table res_benchmark.
> 2. 2030/2035 capacity columns: "the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years". Drop the leading "single-scenario year;" so the phrase appears once.
> 3. After the semicolon in the preamble: "case-data values (Sections S1–S3)", lower-case.

## Changed

| commit | content |
|---|---|
| **`857da388`** | `manuscript_review/gen_supplement.py` (sha256 `e34ea4ec08f628a6…`) and `manuscript_review/verify_supplement.py` (sha256 `edcb632a1a000532…`), +13 / −8. Committed **before** the canonical run |
| **`909b5add`** | `manuscript_review/supplement_v5/`: the zip, the record, the verifier result, `gen_launch.log`, `verify_launch.log` and `SHA256_INVENTORY.txt`. The directory was created by `mkdir` in the launch command and written once |
| this commit | this report |

**Edit.** The same three entries changed in the generator's `V5_WORDING` and in the verifier's own copy
`V10_V5_WORDING`, with a comment in each. Everything else in ruling 1 is unchanged: the label removal (b), the other
(a) / (c) entries, and (d).

| entry | `fb886ea4` (old → new) | `857da388` (old → new) |
|---|---|---|
| (a) preamble | `Appendix B-D values` → `Case-data values (Sections S1–S3)` | `Appendix B-D values` → `case-data values (Sections S1–S3)` |
| (c) capacity columns | `not printed in Sections S2-S3` → `the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years` | `single-scenario year; not printed in Sections S2-S3` → (same replacement). The match now consumes the leading "single-scenario year; " |
| (c) `reported_start` | `printed mark b of Table S5` → `the over-budget mark b of Table S5` | `printed mark b of Table S5` → `mark b of Table S5` |

The capacity-column entry matches the text **after** the label removal: v4's
"single-scenario year (tab:res_capacity_summary); not printed in Sections S2-S3" becomes
"single-scenario year; not printed in Sections S2-S3" first. The verifier rebuilds v5 from v4 in the same order
(the regex first, then the literals).

**Commands.** All were run from the repo root, attached, with `> log 2>&1` and
`P=/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`. `<W>` = `<session scratchpad>/w5b`, which is
outside the repository. `<scratchpad>/w5/drv/emulate_version.py` is the first Worker's scratch driver, not committed
and not modified. It imports the repository's `gen_supplement.py`, so its guards install at import. It pins
`resolve_out_dir` to (scratch dir, N, predecessor = the committed record of N−1) and calls `main()`.
```
# after the edit, before committing it:
$P -u <scratchpad>/w5/drv/emulate_version.py <W>/repro_v2 2     # exit 0 -> 864a5779 (cmp = committed v2)
$P -u <scratchpad>/w5/drv/emulate_version.py <W>/repro_v3 3     # exit 0 -> e6f5ce6a (cmp = committed v3)
$P -u <scratchpad>/w5/drv/emulate_version.py <W>/repro_v4 4     # exit 0 -> d175ea0e (cmp = committed v4)
$P -u <scratchpad>/w5/drv/emulate_version.py <W>/trial_v5 5     # exit 0 -> d2fa2172
$P -u manuscript_review/verify_supplement.py --supplement-dir <W>/trial_v5 --out-dir <W>/trial_v5_verify   # PASS, 0 failures, 25 controls
# commit 857da388, then (the frozen launch commands, checked against the v4 run's):
mkdir manuscript_review/supplement_v5 && $P -u manuscript_review/gen_supplement.py --force-new-version \
  > manuscript_review/supplement_v5/gen_launch.log 2>&1                                          # exit 0
$P -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v5 \
  > manuscript_review/supplement_v5/verify_launch.log 2>&1                                       # exit 0, VERDICT: PASS
$P -u <scratchpad>/w5/drv/emulate_version.py <W>/det_A|det_B|det_C 5                             # exit 0 x 3
$P -u manuscript_review/verify_supplement.py --supplement-dir manuscript_review/supplement_v{2,3,4} --out-dir <W>/reverify_v{2,3,4}   # PASS x 3
```
After the canonical directory existed, the determinism runs used the driver instead of
`gen_supplement.py --out-dir … --force-new-version`. With `supplement_v5/` in place, "the next free version" would
have been 6.

| file (`manuscript_review/supplement_v5/`) | bytes | sha256 |
|---|---|---|
| **`SRP1_supplement_v5_d2fa2172.zip`** (20 entries) | 1,326,229 | **`d2fa21725399d4f5666f7ba38f171b90f24f1a965a2e679607f92b378578cdf9`** |
| `supplement_v5_record.json` | 18,141 | `002761d001ea8852e126513032327901ae35a3eb689fdc74f59702cf93b559be` |
| `verify_supplement_v5_result.json` (gate_result_io) | 34,837 | `782adfa714b6f36582d81b7d844431e8f3a40b771b0fb47830ad17778ac6fc5d` |
| `gen_launch.log` (exit 0) | 2,026 | `1b0dddd751a6e5aad36c6da4558aad9585177e86a3bdedf2094777f5a6b77711` |
| `verify_launch.log` (exit 0) | 8,367 | `69b03441dd511557b6385b1f0a92cd9be9cad9d93b1301faa5fe15e58494e533` |
| `SHA256_INVENTORY.txt` | 454 | lists the five above (`shasum -c`: 5 OK) |

**Record provenance.**
- `git_head` is `857da388…`, and `generator.committed_clean` is true.
- The predecessor is version 4: `supplement_v4/supplement_v4_record.json`, `SRP1_supplement_v4_d175ea0e.zip`, sha256
  `d175ea0e2c7b…9bb0`.
- `description_wording.hits`:
  - `(Appendix E) printed "---"` 1;
  - `Appendix E amounts` 1;
  - `Appendix B-D values` 1;
  - `single-scenario year; not printed in Sections S2-S3` **4**;
  - `printed mark b of Table S5` 1;
  - `Table S6 itself prints no mark` 1.
- `labels_removed`: 20.
- The guards counted 0 solves, 0 process launches and 0 / 0 pickle calls.

**Git timing (W187 running).** Before each of the three commits:
- `pgrep -f p515_s55_step7_launcher.py` returned pid 48166 (`--phase screen --cell x0_A_1x`);
- `.p515_s44_campaign.lock` existed;
- `.git/index.lock` was absent.

**Left alone.**
- Production, harness, launcher and hooks files.
- `data/` and `.p515_s44_campaign.lock`.
- `supplement/`, `supplement_v2/`, `supplement_v3/` and `supplement_v4/`.
- TASKS.md, REVISION_CONTEXT.md and the brief.

Nothing named `p515_s4*` was run, and neither was the step-7 launcher.

**Objective convention:** unchanged. The Appendix-E CSVs are v4's byte for byte; Q = `gross_operational_cost`, and
amounts are in k EUR as printed.

## Found

| check | expected | observed |
|---|---|---|
| verifier on v5 | PASS, 0 failures, all controls caught | **VERDICT PASS, 0 failures**; **25 of 25** negative controls caught (C1–C22 as v4, plus C23–C25) |
| V1 Appendix E vs frozen tables | 1,623 entries, 0 MISMATCH | **1,623: 1,443 MATCH, 0 MISMATCH**, 180 not-frozen agree, 0 disagree, 0 mapping problems |
| V5 B–D vs case files | 70,851 cells, 0 | **70,851 cells in 2,700 rows, 0 differences** |
| V8 Table 2 | 48/48 | **48 printed aggregates, 0 differences** |
| v4 → v5 entries | README.md + SHA256SUMS.txt only | **exactly these two** (V9: 18 identical, 0 added, 0 removed). Independently, `diff -rq` of the two unzipped trees lists the same two |
| `D_ieee33_tn_connection.csv` | byte-identical | **identical** (V9; and `cmp` after unzip, sha256 `f954ad5057ee2c77…` in both) |
| V10 | README = v4 README + heading + ruling-1 edits | **True**; stale a87e195 names 0 for each of the 7 patterns |
| V12 | all 0 | **all 0**: `(fig:` 0, `(tab:` 0, `(app:` 0, bare label 0, `Appendix E` 0, `Appendix B-D` 0 (v4: 5 / 15 / 0 / 20 / 2 / 1). Punctuation artefacts equal to v4's; **no new artefact** (double space 0) |
| V11 | names vs main.tex 3c379fa | B.10, B.11, B.9, B.2–B.4 and Table 2 = `tab:res_capacity_summary`, all True |
| V4 | (inherited) | verify_appE.py 40 PASS 0 FAIL |
| item 2: the phrase appears once | "single-scenario year" not repeated; no "; ;", double space or leading punctuation | **holds on all 4 lines** (shown below). "single-scenario year" occurs 4 times in the README, once per affected line (the 5th "single-scenario" is "single-scenario recourse evaluation" in the prose table, l. 18) |
| item 3: preceding characters | "; " | **"); case-data values (Sections S1–S3)"**: the character before is "; " |
| v2 / v3 / v4 reproduce | byte for byte | **864a5779, e6f5ce6a, d175ea0e**, each `cmp`-identical to the committed zip with the edited generator |
| v2 / v3 / v4 under the edited verifier | (regression) | **PASS, 0 failures** each (21 / 21 / 22 controls) |
| determinism | 3 scratch runs = canonical | **4 byte-identical `d2fa2172…`**: the pre-commit trial, det_A, det_B and det_C (`cmp` to the canonical zip). The det records differ from the canonical record only in `out_dir` and `canonical` |
| hash vs the superseded trial | differs from `1d0c39fe` | **`d2fa2172`** |
| zero solves | 0 | generator and verifier guards (plus W180a's): 0 solves, 0 process launches, 0 blocked; pickle 0 / 0 |

### The three ruled lines as they appear in v5's README

- l. 77 (preamble): `Empty CSV cell: not applicable, not recorded, or (evaluation-record files, Tables B.10, B.11 and S4–S6) printed "---". Booleans: yes / no. Evaluation-record amounts are in k EUR as printed (rounded per export_W160/README.md); case-data values (Sections S1–S3) are at the full precision of the case files.`
- l. 449: ``| `reported_start` | yes / no | mark b of Table S5: the start reported in the main-text benchmark table (not the budget mark) |``
- The four capacity lines, before (v4) → after (v5):
  - l. 124
    - v4: ``| `capacity_2030_MW` | MW | installed capacity in 2030 (Pmax of case9_2030.json); single-scenario year (tab:res_capacity_summary); not printed in Sections S2-S3 |``
    - v5: ``| `capacity_2030_MW` | MW | installed capacity in 2030 (Pmax of case9_2030.json); the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years |``
  - l. 127
    - v4: ``| `capacity_2035_MW` | MW | installed capacity in 2035 (Pmax of case9_2035.json); single-scenario year (tab:res_capacity_summary); not printed in Sections S2-S3 |``
    - v5: ``| `capacity_2035_MW` | MW | installed capacity in 2035 (Pmax of case9_2035.json); the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years |``
  - l. 250
    - v4: ``| `capacity_2030_MW` | MW | installed capacity in 2030; single-scenario year (tab:res_capacity_summary); not printed in Sections S2-S3 |``
    - v5: ``| `capacity_2030_MW` | MW | installed capacity in 2030; the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years |``
  - l. 253
    - v4: ``| `capacity_2035_MW` | MW | installed capacity in 2035; single-scenario year (tab:res_capacity_summary); not printed in Sections S2-S3 |``
    - v5: ``| `capacity_2035_MW` | MW | installed capacity in 2035; the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years |``

### Full README diff v4 → v5, by edit type

The diff has 24 hunks, one changed line each. The prose half (ll. 1–42) is unchanged.

| edit type | lines | count |
|---|---|---|
| version heading "version 4" → "version 5" | 43 | 1 |
| preamble, three (a) edits on one line: `(Appendix E) printed "---"` → `(evaluation-record files, Tables B.10, B.11 and S4–S6) printed "---"`; `Appendix E amounts` → `Evaluation-record amounts`; `Appendix B-D values` → `case-data values (Sections S1–S3)` | 77 | 1 |
| (b) label removal only (" (label)" removed) | 81 `fig:cs3_market_prices`; 115 `tab:cs3_tn_res_generators`; 132 `fig:cs3_tn_res_scenarios_2025`; 166 `fig:cs3_adn_node_5/7/9_flexibility`; 202 `fig:cs3_adn_node_5/7/9_load_scenarios_2025`; 239 `tab:cs3_adn_node_5/7/9_res_generators`; 258 `fig:cs3_adn_node_5/7/9_res_scenarios_2025`; 294 `tab:cs1_ieee33_branches`; 311 `tab:cs1_ieee33_buses`; 326 `tab:cs1_ieee33_loads`; 342 `tab:cs1_ieee33_generators`; 368 `tab:appE_claims`; 400 `tab:appE_evaluations`; 429 `tab:appE_neighbours`; 443 `tab:appE_benchmark`; 459 `tab:appE_gap` | 16 |
| (b) label removal + (c) capacity wording (`(tab:res_capacity_summary)` removed, then "single-scenario year; not printed in Sections S2-S3" replaced) | 124, 127, 250, 253 | 4 |
| (c) `reported_start`: "printed mark b of Table S5" → "mark b of Table S5" | 449 | 1 |
| (c) S6 `over_budget`: "(Table S6 itself prints no mark)" → "(Table S6 carries the mark in its over_budget column)" | 469 | 1 |

Labels removed: 16 + 4 = 20, which matches the record. `SHA256SUMS.txt` changes in its README line only
(`ef41984b…` → `85afb573…`). README size: 34,874 → 34,613 bytes. The (d) `table` column definitions
("E.20 or E.21", "E.20, E.22 or E.23") are untouched. The full `diff` output is in the session scratchpad
(`w5b/readme_v4_v5.diff`) and can be regenerated from the two committed zips.

## Unexpected findings

1. **The ruled capacity wording leaves the 2025 column ambiguous. This is a wording point, not a data point.** The
   2030 and 2035 rows now say "the single-scenario years of the manuscript's Table 2; the remaining columns are the
   multi-scenario years". The 2025 row, which is unchanged, says "single- and multi-scenario year". Table 2 also sums
   2025 (V8 checks 2025, 2030 and 2035 against its aggregates). So "the remaining columns are the multi-scenario
   years" is literally true of 2025, but it does not name 2025 as a Table 2 single-scenario year. I applied the ruled
   string as written. Whether to change it is for the expert.
2. **The first Worker's `fb886ea4` already yielded the doubled phrase in its trial.** Its trial's capacity lines read
   "…; single-scenario year; the single-scenario years of the manuscript's Table 2; …". Item 2 of the ruling removes
   this, and V12's artefact check would not have caught it, because it is not a punctuation artefact. This is noted so
   that the trial `1d0c39fe` is not mistaken for an acceptable alternative.

## Not confirmed

- **Compiled numbering of Table 2 and B.10 / B.11 / S4–S6.** Same as v4: these come from a static walk of 3c379fa
  `main.tex` (V11). No TeX is installed.
- **The number of the main-text benchmark table.** `appendixE_evaluation_record.tex` l. 326 reads
  "$^{b}$: the start reported in Table~\ref{tab:res_benchmark}", which agrees with the expert's item 1. V11 does not
  check `tab:res_benchmark`'s number, because the README names no number for it.
- **Cross-machine byte identity.** This was checked on this machine only.
- **v1 reproducibility.** Not re-examined (see the v3 report).
