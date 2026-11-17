# P5.15 W188 (b): the machine-readable supplement (Addendum 89 ruling 8)

**Worker, 2026-11-10.** Zero solves, zero model builds and zero pickle loads. The following were only read: `gen_appE.py`, `verify_appE.py`, `export_W160/`,
the frozen JSON, the Overleaf clone (`git show a87e195:main.tex`), the case files under `data/SRP1/`, production and
earlier harnesses. Nothing outside the files listed under Changed was edited or staged. That includes TASKS.md, REVISION_CONTEXT.md,
the brief, the clone and production.

## Blocked on Planner

Nothing blocks the supplement. The verifier passes. Four content questions are for the expert, who writes the README.
None of them changes a number.

1. **What the Appendix B–D figures plot.** The figures show the mean ± std over synthetic scenarios.
   - Production samples those scenarios from the case-file base profiles: `network_data.generate_*_profiles` and
     `shared_resources_planning._generate_market_price_scenarios`, a Gaussian copula with KDE and 100 samples.
   - The supplement carries the base profiles exactly as stored in the case files. It does not carry the plotted
     curves.
   - Exporting the plotted curves would mean running the production sampler. That costs no solves, but it is a
     production code path. It is also not confirmable that today's code and seeds reproduce the submitted figures.
   - The README placeholder says this. **Option (a):** keep the base profiles (recommended; it is what the case files
     hold). **Option (b):** also export the sampled mean/std. That needs a separate order.
2. **E.24 `over_budget` column.** The CSV of Table E.24 has an `over_budget` column. It is taken from the same
   evaluation's E.21 flag: yes for evaluations 39 and 6.
   - The printed E.24 carries no $^{b}$ mark (W183h finding 1).
   - So this is the one place where the supplement says more than the printed table. The verifier lists it as a
     declared difference.
   - Keep it, or drop it to match the print? Expert's call. It sits in the round-9 queue with the E.24 marks.
3. **Years of the RES-capacity tables.** The capacity tables (`tab:cs3_tn_res_generators`,
   `tab:cs3_adn_node_*_res_generators`) are exported as printed: 2025, 2028, 2031, 2034 and 2037, the multi-scenario
   years.
   - The single-scenario SRP1 instance behind Appendix E uses 2025, 2030 and 2035.
   - In the case files, 2030 equals 2031 and 2035 equals 2037. Those columns are not printed, so they are not exported.
   - Add them? Expert's call.
4. **`tab:cs1_ieee33_generators` is not common to all ADNs** (Unexpected findings, item 1). This is a manuscript
   wording or data point for the expert, not a supplement defect.

## Changed

- **`5d7955da`**: `manuscript_review/gen_supplement.py` (sha256 `d61bd0f5a6742453…`) and
  `manuscript_review/verify_supplement.py` (sha256 `03149be4f00029b6…`). Both were committed **before** the real
  run. The generation record shows `git_head` `5d7955da…` and `generator.committed_clean: true`.
- **This commit**: this report and the whole of `manuscript_review/supplement/`. The directory was created by `mkdir`
  in the launch command and written once.

| file | sha256 |
|---|---|
| `SRP1_supplement_v1_a94d8193.zip` (1,320,589 bytes, 20 entries) | `a94d8193047b7e688205549945ba38b5e0372913c2e0b59fc0c0627ba5c90726` |
| `supplement_v1_record.json` (hash inventory of every entry + the zip, input hashes, provenance, guards) | `04ffa1fb151e24b2809a5304e7e591c137aa7ba4b5497f5fa4f96456acacd205` |
| `verify_supplement_v1_result.json` (verifier output, gate_result_io) | `9cc3c787667b57380bac25e5ed98203ae76a222770071acbcce27947550eb1f5` |
| `gen_launch.log` (generator stdout + stderr, exit 0) | `906b8cb934fdf646b1fde3faab3486c9fbd5f0ad553240abc19cd9d8ed4098d4` |
| `verify_launch.log` (verifier stdout + stderr, exit 0) | `2e984ad7556ff55a588ba0a6f16d1dcbf80dfb7ffd0274e7f7317668c4b97db0` |

**Zip entries.** Each entry's sha256 is in `SHA256SUMS.txt` inside the zip and in the record. Rows and columns
exclude the header.

| entry | rows × cols | sha256 (16) |
|---|---|---|
| `appendix_E/E20_comparisons.csv` | 58 × 25 | `9a34706cf93e82ae` |
| `appendix_E/E21_evaluations.csv` | 61 × 22 | `4138d7be72a57bcb` |
| `appendix_E/E22_neighbours.csv` | 8 × 7 | `56952c6ce24defb6` |
| `appendix_E/E23_benchmark.csv` | 7 × 9 | `05134bcf74822350` |
| `appendix_E/E24_gap_refused.csv` | 9 × 13 | `22232e2e3dd9fbe7` |
| `appendix_E/E_table_notes.csv` (numbers of the E.20 / E.22 / E.23 notes) | 15 × 5 | `0e8633dd6b8c85e3` |
| `appendix_E/E20_E21_groups.csv` (group titles, LaTeX verbatim) | 10 × 3 | `a3045c07a31fa4dd` |
| `appendices_B_D/B_market_prices.csv` | 24 × 27 | `dbfd4c477838969a` |
| `appendices_B_D/C_tn_res_generators.csv` | 6 × 8 | `cecb82e430843e98` |
| `appendices_B_D/C_tn_res_profiles.csv` | 24 × 27 | `26a39d06efe5fbc1` |
| `appendices_B_D/D_ieee33_buses.csv` | 33 × 8 | `082b50cff0e772c5` |
| `appendices_B_D/D_ieee33_branches.csv` | 82 × 10 | `7452321e093b15da` |
| `appendices_B_D/D_ieee33_tn_connection.csv` | 3 × 9 | `f954ad5057ee2c77` |
| `appendices_B_D/D_ieee33_loads.csv` | 32 × 9 | `01638227c9967d36` |
| `appendices_B_D/D_adn_res_generators.csv` | 12 × 10 | `6af15d85b6c6b57c` |
| `appendices_B_D/D_adn_res_profiles.csv` | 72 × 29 | `90c581de8f7da7d4` |
| `appendices_B_D/D_adn_load_profiles.csv` | 2,304 × 30 | `2c1a5d1b62958e11` |
| `appendices_B_D/D_adn_flexibility_profiles.csv` | 108 × 29 | `e6adf89ade52723c` |
| `README_PLACEHOLDER.md` | — | `26aa291834ccaa4c` |
| `SHA256SUMS.txt` | — | `878fde702edcc9c7` |

**Commands.** All from the repo root, with the canonical interpreter, attached, both streams captured. No nohup,
screen or `&`.

```
mkdir manuscript_review/supplement && \
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/gen_supplement.py \
  > manuscript_review/supplement/gen_launch.log 2>&1                                   # exit 0
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/verify_supplement.py \
  > manuscript_review/supplement/verify_launch.log 2>&1                                # exit 0, VERDICT: PASS
```

**How the generator works:**
- **Appendix E.** It executes `gen_appE.py` as a module, as `verify_appE.py` does, writing into a temporary file.
  - E.20 and E.21 come from gen_appE's own `e1_row` / `e2_row` over `E1` / `G`. Each row must occur in the generated
    LaTeX.
  - E.22–E.24 are parsed from the generated LaTeX.
  - Each cell is the printed cell with its markup removed:
    - `---` becomes empty and `$-$` becomes `-`;
    - the thousands `\,` is dropped in numeric columns;
    - every mark becomes an explicit yes/no column: $^{b}$ → `over_budget` (+ `over_budget_plans` in E.20),
      $^{f}$ / $^{t}$ / $^{s}$ / $^{u}$ / $^{n}$ / $^{\Delta}$, and E.23's own $^{b}$ → `reported_start`.
  - Identifier columns are added: `cell_id`, `claim_id`, and `eval_key` / `candidate_key` (prefixes from T2.csv).
  - The note numbers come from gen_appE's own variables (`r3`, `r26`, `sup`, `coord`, `bests`). Each is asserted to
    occur in its printed note.
- **Appendices B–D.**
  - The xlsx files are read through production's readers, `shared_resources_planning._read_market_base_profiles`
    and `network_data._read_network_base_profiles`.
  - The network JSON files are read with `json`.
  - Values are written at full precision: repr of the float, integers as integers.
- **Determinism.** Entries are sorted, every entry is dated 1980-01-01 00:00:00 with mode 0644, and storage is
  ZIP_STORED, so no compressor is in the byte path.
- **Write rules.**
  - Output files are opened `'x'`.
  - A non-empty output directory is refused. `--force-new-version` writes `manuscript_review/supplement_v<N>/`,
    which records its predecessor's zip sha256.
  - A canonical run with an uncommitted generator or input is refused.
- **Guards.**
  - `SolveProfileGuard(permitted=())` is installed before any project import.
  - pickle is blocked for the whole run.
  - The verifier also arms the guard of the imported W180a module. Both guards ended at 0 solves and 0 process
    launches, with pickle calls at 0.

**Objective convention**, as in every Appendix E CSV: Q = `gross_operational_cost`, contracted interface settlement
excluded; net = Q − terminal salvage credit; value = Q(0) − Q(x); F = Q + I; amounts in k EUR as printed.

## Found

### Verifier: PASS, 0 failures (`verify_supplement_v1_result.json`)

| check | outcome |
|---|---|
| V0 zip / inventories | sha256 = the record's; 20 entries sorted, all dated 1980-01-01, 0644, STORED; SHA256SUMS.txt and the record's member list equal the zip |
| **V1 Appendix E vs frozen tables** | **1,623 entries, 0 MISMATCH**: 1,443 MATCH, 180 not-frozen-agree, 0 disagree, 0 mapping problems; per table E.20 639, E.21 854, E.22 18, E.23 49, E.24 63 (= W180a / W183h). `cell_id` and `claim_id` columns = the check's own mapping. E.21 `eval_key` / `candidate_key` are prefixes of the frozen keys. The 4 E.20 note numbers (bars 70.7 / 15.9 k€, ×2.94 / 11.12) MATCH the frozen `uncertified_form_beside_A64_ruling4` |
| V2 Appendix E vs main.tex at a87e195 | 1,633 cells, 0 differences, 0 mark differences. This is the 1,637 raw cells less the 4 unrepeated E.23 arrangement labels. The E.20 budget fragments equal the printed ones. One declared difference: E.24 `over_budget` = yes on evaluations [39, 6], which the print does not mark |
| V3 `over_budget` vs W182d | E.21 {5, 6, 9, 10, 11, 38, 39} = W182d's over-budget evaluations; E.24 {6, 39}; 0 inconsistent E.20 rows |
| V4 generator identity | gen_appE.py output sha256 `66bd3dd0…` = committed `appendixE_evaluation_record.tex`, found once in main.tex at a87e195, l. 2556–2946, line-aligned; `verify_appE.py` 33 PASS 0 FAIL |
| **V5 B–D vs case files** | **70,743 cells in 2,700 rows, 0 differences**, read by an independent reader (openpyxl, json). Every source row is present. 3 blank rows in the market Flexibility sheet were skipped by both readers |
| V6 C–D vs printed tables at a87e195 | 905 printed cells at printed rounding, 0 differences. One observation: Unexpected findings, item 1 |
| V7 README placeholder | 18 column tables = 18 CSV headers; all placeholders present; τ = `repr(settling_criterion_v6.TAU)` |
| Negative controls (in memory) | **15 of 15 caught exactly where altered** (list below) |

The negative controls:
- **C1–C7.** One cell each in E.21, E.20, E.23, E.24 and E.22, the E.23 note, and an E.21 status. Each gives exactly
  one MISMATCH, at the edited table, row and key.
- **C8.** An E.20 note number is caught.
- **C9.** An `over_budget` flip is caught by both V2 and V3.
- **C10.** A k0 + 1 is caught against main.tex.
- **C11–C13.** A branch r, a load profile hour (+1e-9) and a market price are each caught by V5.
- **C14.** A bus v_max is caught by both V5 and V6.
- **C15.** A one-byte flip of the zip is caught: sha mismatch, CRC error and inventory mismatch.

### Determinism: three runs byte-identical

| run | zip sha256 |
|---|---|
| scratch `w188b_detA` at `5d7955da` | `a94d8193…` |
| scratch `w188b_detB` at `5d7955da` | `a94d8193…` (`cmp` identical to detA) |
| canonical `manuscript_review/supplement/` | `a94d8193…` (`cmp` identical to detA) |

The pre-commit trial `w188b_trial2` (same generator bytes) also gave `a94d8193…`. The record JSON is outside the zip,
and only its `out_dir` field differs between the runs.

### Source file of every Appendix B–D table

| appendix item (main.tex a87e195) | supplement CSV | case-file source |
|---|---|---|
| B: `fig:cs3_market_prices` (energy, flexibility; 4 seasons) | `B_market_prices.csv` | `data/SRP1/MarketData/SRP1_market_data.xlsx`, sheets Energy, Flexibility (4 seasons × 3 base scenarios × 24 h). The file is named by `MarketData` in `SRP1.json` and `SRP1_5x3y_1x1.json` |
| C: `tab:cs3_tn_res_generators` | `C_tn_res_generators.csv` | `data/SRP1/case9/case9_{2025,2028,2031,2034,2037}.json`, generators of type WIND / PV, `Pmax`, `bus` |
| C: `fig:cs3_tn_res_scenarios_2025` | `C_tn_res_profiles.csv` | `data/SRP1/case9/case9_operational_data.xlsx`, sheet Pg |
| D: `tab:cs1_ieee33_buses` | `D_ieee33_buses.csv` | `data/SRP1/case33_1/case33_1_2025.json` `nodes`. Identical in case33_2/3 and in every case year 2025–2037 |
| D: `tab:cs1_ieee33_branches` | `D_ieee33_branches.csv` | `case33_{1,2,3}_2025.json`, `transformers` (branch 1) + `lines` (2–82). Ratings per ADN; branch 1 is 200 / 100 / 150 MVA; the printed column is node 5 |
| D: `tab:cs1_ieee33_generators` | `D_ieee33_tn_connection.csv` | `case33_{1,2,3}_2025.json`, generator of type REF (one row per ADN) |
| D: `tab:cs1_ieee33_loads` | `D_ieee33_loads.csv` | `case33_1_2025.json` `loads` (bus). `case33_{1,2,3}_operational_data.xlsx`, sheet Characterization: FL Ratio (identical in the three ADNs) and Pc / Qc per ADN (not printed) |
| D: `tab:cs3_adn_node_{5,7,9}_res_generators` | `D_adn_res_generators.csv` | `case33_{1,2,3}_{2025,2028,2031,2034,2037}.json`, generators WIND / PV. The node 5 / 7 / 9 → case33_1 / 2 / 3 mapping is `connection_node_id` in `SRP1.json` (= `SRP1_5x3y_1x1.json`) |
| D: `fig:cs3_adn_node_*_res_scenarios_2025` | `D_adn_res_profiles.csv` | `case33_{1,2,3}_operational_data.xlsx`, sheet Pg |
| D: `fig:cs3_adn_node_*_load_scenarios_2025` | `D_adn_load_profiles.csv` | the same files, sheets Pc and Qc (32 loads × 4 seasons × 3 base scenarios) |
| D: `fig:cs3_adn_node_*_flexibility` | `D_adn_flexibility_profiles.csv` | the same files, sheet Flex. All four seasons are exported; the figure shows Spring |

### README placeholder (filled from code, for the expert)

- **τ.** `settling_criterion_v6.TAU` = 4539.0682750000005 € (DELTA_R × R_REF / 4). The frozen
  `constants.TAU` is the same value.
- **Objective convention.** "Q: gross operational cost, contracted interface settlement excluded", printed beside the
  frozen `objective_convention` string.
- **Certification rule.** Every numeric entry of `settling_criterion_v6.constants(30)` with its formula: DELTA_R,
  R_REF, TAU, EPS0, GAP_BOUND, SWING_FLOOR, W_MIN, W_FACTOR, K_EXCL, P_MAX = 30, L_MONO = 60, CLEAN_FACTOR = 10,
  DRIFT_WINDOW = 25, DETERMINACY_FLOOR {3, 2, 2τ}, TOLERANCES.
  - P_MAX is read from the v6 spec (`96c23404`), where every cell declares 30.
  - The non-numeric rule entries are named, not reproduced.
  - The README also carries the export's two determinacy-rule labels.
- **Column definitions.** Generated from `COLUMN_SPECS` in the generator. Their prose descriptions are the one
  hand-written part. The verifier checks the column names against every CSV header.

## Unexpected findings

1. **`tab:cs1_ieee33_generators` (TN connection data) prints 250 MW / MVAr as common to all ADNs.** The case files
   disagree for the ADN at TN node 7.
   - `case33_2_2025.json` has Pmax 100, Pmin −100, Qmax 100 and Qmin −100.
   - Nodes 5 and 9 match the print, at 250.
   - The supplement carries all three rows. V6 records this as an observation, not a failure: the printed row equals
     node 5's row.
   - This parallels the branch-1 rating, which differs by ADN and which the caption does explain.
2. **The ADN base loads (Characterization Pc / Qc) differ by ADN.**
   - ΣPc is 176.9 / 194.6 / 185.75 MW at nodes 5 / 7 / 9.
   - ΣQc is 92.0 / 26.4 / −8.0 MVAr. In 16 loads at nodes 7 and 9, Qc is negative.
   - These are not printed: the printed load table has only bus and FL ratio, which are common. They are noted
     because the load profiles are multipliers of these values. Both are exported.
3. **`verify_appE.py` leaves its own `tempfile.mkdtemp()` file.** It is in the system temp directory, outside the
   repository. That is its own code, not edited. The new scripts remove their temporaries.

## Not confirmed

- **`--force-new-version` was not exercised end to end.** Writing `supplement_v2/` inside the repository was not
  authorised. Its resolution was probed without writing: a separate process imported the generator and called
  `resolve_out_dir`. The probe returned `manuscript_review/supplement_v2`, version 2, predecessor
  `SRP1_supplement_v1_a94d8193.zip` / `a94d8193…`.
- **These refusals were exercised:**
  - a canonical rerun onto the populated directory: exit 1, nothing written;
  - a non-empty scratch directory: exit 1;
  - an uncommitted canonical run: exit 1;
  - the verifier onto an existing result: exit 1.
- **Cross-machine byte identity.** ZIP_STORED keeps zlib out of the byte path. The CSV bytes still depend on Python's
  float repr and on pandas 3.0.0 / openpyxl 3.1.5 reading the xlsx. Determinism was checked on this machine only.
- **The B–D figures.** They were not compared numerically. Their plotted curves are sampled statistics; see Blocked
  on Planner, item 1.
- **Notes not exported.** The E.21 note's literal "k0 = 123", typed in gen_appE's prose, and the prose of the E.20 /
  E.21 notes are not in the notes CSV. Only the numbers gen_appE takes from the export are in it.
- **Units of the market prices.** They are given as "as stored": the sheet carries no unit, and the figure axis reads
  "€/MW". This is for the expert's README.
- **No compile.** No TeX is installed on this machine, and none was needed.
