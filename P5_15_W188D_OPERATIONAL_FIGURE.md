# P5.15 W188 (d): operational figure data (Addendum 89, Planner order W188 (d))

**Worker, 2026-11-10.** Zero solves, zero model builds, two guarded pickle loads (one per allow-listed file). I edited
no production code, record, `e6_export/`, the manuscript clone, `manuscript_review/`, TASKS.md, REVISION_CONTEXT.md or
the brief. The expert chooses the figure; this task produces data only.

**Objective convention, on every value below.** Q is the certified `gross_operational_cost`: gross operational cost
with the contracted interface settlement excluded. There is one market scenario, so the deviation settlement and the
commitment premium are 0; the terminal salvage is 0 on both E.21 rows. Block values are block-weighted (year count ×
day count × discount annualisation) unless marked unweighted.

**Instances.** E.21 row 1 `ref:7aa017f0`, x = 0, eval `d110bd1a5977df1e_x0`, k* 181, Q 653,873,702.19 €. E.21 row 2
`ref:bd504ecf`, node 7, 0.25 MVA / 1 MWh, 2025 (the reference unit), eval `3f084f2ffaeef2b7_n7_4h_e1`, k* 172,
Q 653,620,162.57 €. The ageing arms are listed with their eval keys in part 3.

---

## Blocked on Planner

None. Two points for the Planner to note; neither needs a decision unless the scope should change:

1. **SoH is one value per (cohort, 5-year block), not per day.** "SoH per (year, block)" is exported per
   representative-year block, 2025 / 2030 / 2035. For each block the export gives the start, end, mid-block (closed
   form) and applied SoH. The model has no per-day SoH.
2. **The run made two pickle loads; trials made two more.**
   - The real run made 2 loads: row 2's file `d0a3d512…`, then row 1's file `99ab1070…`. Each was hash-checked on the
     bytes it unpickled.
   - Before the real run, one trial into scratchpad `w188d_trial1` made the same two loads.
   - One dry run, `w188d_dry1`, made no loads.
   - Addendum 85 ruling 6 covers the trial loads: as many guarded reads as the task needs, each logged with its run.
     The trial's log is in the scratchpad, not committed.

---

## Changed

- **`d5cb6bbb`**: the harness `p515_s55_w188d_operational_figure.py`, committed before the real run. The run records it
  as `committed`, sha256 `3d55fe16f8c639225616369a2c450b47558a56e8bf946e901e9d109b8b8d4c24`.
- **Export commit** (hash in the final message): this report, plus `data/SRP1/Results/P515S55/w188d_operational_figure/`,
  added with `git add -f` by filename. The directory was new, and each file was written once:

  | File | Content | sha256 |
  |---|---|---|
  | `w188d_operational_figure.json` | summary, re-derived E6 sections, SoC summary, x = 0 statement, SoH identity, guards | `44926c47…` |
  | `e6_rederivation_comparison.csv` | 1,183 compared entries: path, re-derived, e6_export, abs/rel diff | `19dbff08…` |
  | `e6_rederived_day_ranking.csv` | 12 blocks | `95e79caa…` |
  | `e6_rederived_panel_b.csv` | 12 blocks × weighted/unweighted | `58d8ce0c…` |
  | `e6_rederived_panel_c.csv` | 2 rows × 3 nodes × 3 years | `625e445d…` |
  | `soc_hourly_reference_unit_2025_2035.csv` | 8 blocks × 24 h, TSO and DSO copies | `459962df…` |
  | `soc_block_summary_reference_unit.csv` | 8 blocks | `5b5b3092…` |
  | `soc_limits_per_year_reference_unit.csv` | 2025 and 2035 | `eb2d05ed…` |
  | `zero_capacity_model_check.csv` | 120 shared-ESS copies (72 of row 1, 48 of row 2's nodes 5/9) | `cef566fa…` |
  | `soh_per_block_calibrations.csv` | 7 calibrations × 3 blocks | `9243aafe…` |
  | `soh_calibration_identity.csv` | 7 calibrations: identity and cross-checks | `62053602…` |
  | `manifest_sha256.json` | 11 outputs, the script, 35 inputs with sha256 and git status | `1719dcd1…` |
  | `launch.log` | stdout + stderr, attached run, exit 0 | `1195a57b…` |
  | `manifest_post_run_sha256.json` | over all of the above (`--post-run`, exit 0) | `19941df2…` |

- **Inputs.** 32 of the 35 inputs are committed. The other three:
  - the two pickles, which are untracked and pinned by hash;
  - the manuscript `main.tex`, which lives in the clone, not in this repository. Its md5 `39f6fd86…` (= a87e195) is
    checked before any reading.

  `e6_export.json` was verified before reading: working-file sha256 = blob at `d794f11b` = blob at HEAD = `48750757…`.
- **Command** (repo root, canonical interpreter; wall time 18 s):

  ```
  mkdir data/SRP1/Results/P515S55/w188d_operational_figure && \
    python -u p515_s55_w188d_operational_figure.py > data/SRP1/Results/P515S55/w188d_operational_figure/launch.log 2>&1
  python -u p515_s55_w188d_operational_figure.py --post-run
  ```

- Repo HEAD at the run: `d5cb6bbb`. There were no tracked changes at the start.

---

## Found

### 1. E6 re-derivation: zero differences beyond floating-point summation order

I re-derived every E6 number with this script's own code. The sources were the committed records of rows 1 and 2:
`component_levels_terminal.json`, `settling_decision.json`, `recourse_blocks_all.jsonl`,
`interface_settlement_detail_s31c.json` and `evaluation_record.json`, plus `SRP1.json`. Only afterwards did I compare
the results with `e6_export.json`, path by path.

| Section | Numeric entries | Bitwise equal | Max abs diff | Exact (string/bool) entries, unequal |
|---|---:|---:|---:|---|
| day ranking (`day_selection`) | 174 | 174 | 0 | 15, 0 |
| panel (b) (`panel_b_per_block_change_vs_x0`) | 825 | 804 | 1.49e-8 € | 0, — |
| panel (c) (`panel_c_e_available`) | 116 | 116 | 0 | 53, 0 |
| **Total** | **1,115** | **1,094** | **1.49e-8 €** | **68, 0** |

- **E6 entries not re-derived:** 0. **Re-derived paths absent from E6:** 0.
- **The 21 non-zero differences are all in panel (b):**
  - 15 are `Q_block` totals (row 1, row 2 or change) of weighted or unweighted blocks;
  - 6 are the `components_sum_to_Q_change_residual` fields.
- **Each is 1–2 ulp of a ~6e7 € block total** (ulp = 7.45e-9 €). The largest relative difference outside the residual
  fields is 7.1e-13. Cause: this script sums the components TSO-then-DSO; the W183e harness summed per network block.
- **The horizon ΔQ differs by 3.7e-9 €:** −253,539.62187865004 here against −253,539.62187865376 in E6.
- **The values that carry claims are bitwise equal:**

  | Quantity | Re-derived = e6_export |
  |---|---|
  | Declared day | **2025 Spring** |
  | c(2025 Spring) = Q(0) − Q(x) | **40,181.22258103639 €** |
  | Share of total | **0.15848103851897147 (15.8 %)** |
  | Σ c over 12 blocks | **253,539.62187865376 €** |
  | Margin rank 1 − rank 2 | 12,170.61018178612 € |
  | Bar rank 1 + rank 2 | 103.3669975027442 € |
  | Margin / bar | **117.74×** |
  | Windows | row 1 [150, 181], row 2 [140, 172] |
  | Panel (b) horizon: TN conventional / DN flexibility | −158,938.59087385796 / −94,581.13244988886 € |
  | Panel (c) E^Av, node 7 (2025 / 2030 / 2035) | 0.8804577542, 0.7798174940, 0.6999999900 MWh |

- The full 12-block ranking, bars and "next rank determinate" flags reproduce the W183e table exactly, including
  rank 3's 8,588.75 € bar.

### 2. SoC export: reference unit, node 7, the four days of 2025 and of 2035

**Source.** Row 2's `certified_models.pkl` (`d0a3d512…`; equal to `post_certification.json`; re-hashed on the
unpickled bytes).
- The TSO copy is shared-ESS index 1, node 7's position in the `SRP1.json` DistributionNetworks order. The run asserts
  that it is the only TSO index with capacity, in every block.
- The DSO-7 copy is index 0.
- Values are × baseMVA 100 (case9 / case33_2).

**Limits.**
- The model's limits are `pe.value` of the bounds of each period's `sess_soc_limit_lower` / `_upper` rows.
- Those bounds equal 10 % / 90 % × `shared_es_e_rated_fixed` to within 1.4e-17 MWh.
- All 24 limit rows of every exported copy are active, and no SoC variable is fixed.

| Year | E^Av record (MWh) | 10 % / 90 % record (MWh) | E^Av in the model copies (MWh) | Model − record |
|---|---:|---:|---:|---:|
| 2025 | 0.8804577542 | 0.0880457754 / 0.7924119788 | 0.8804565377 (TSO = DSO, all 4 days) | −1.22e-6 |
| 2035 | 0.6999999900 | 0.0699999990 / 0.6299999910 | 0.6999999900 (TSO = DSO, all 4 days) | 1.1e-16 |

The 2025 model − record difference is the 1.2e-6 MWh E6 already reported. It is consistent with the copies holding the
E^Av published one cycle earlier; this is a hypothesis, not traced. The start SoC and target end SoC are 0.5 × the
model's E^Av.

**Contacts.** Rule, declared before the run: |SoC − model limit| ≤ 1e-4 MWh, per copy and hour. The classification
does not depend on the tolerance anywhere in [1.0e-6, 5.9e-3] MWh: every contact lies within 1.0e-6 MWh of its limit,
and the closest non-contact hour is 5.9e-3 MWh away.

| Block | SoC min / max, TSO copy (MWh) | Contacts lower / upper (TSO = DSO) | max \|SoC_TSO − SoC_DSO\| (MWh) | Charged / discharged, TSO (MWh) |
|---|---|---|---:|---|
| 2025 Spring | 0.0880447 / 0.7924119 | 7 / 4 | 1.33e-5 | 0.7550 / 0.7031 |
| 2025 Summer | 0.0880447 / 0.7924119 | 11 / 5 | 4.0e-6 | 0.7262 / 0.6762 |
| 2025 Autumn | 0.0880448 / 0.7924119 | 2 / 3 | 8.9e-6 | 1.3394 / 1.2473 |
| 2025 Winter | 0.1221901 / 0.7924119 | 0 / 5 | 1.45e-5 | 1.3747 / 1.2801 |
| 2035 Spring | 0.0699990 / 0.6300010 | 7 / 5 | 1.26e-5 | 0.5773 / 0.5376 |
| 2035 Summer | 0.0699990 / 0.6300010 | 6 / 7 | 9.9e-6 | 0.5773 / 0.5376 |
| 2035 Autumn | 0.0699990 / 0.6300010 | 8 / 5 | 8.7e-6 | 0.5773 / 0.5376 |
| 2035 Winter | 0.1078259 / 0.6300010 | 0 / 4 | 1.21e-5 | 0.8948 / 0.8332 |

- **Contacts:** 41 hours at the lower limit and 38 at the upper, out of 192.
  - The TSO and DSO copies have identical contact flags in all 192 hours (0 mismatches).
  - Every block except the two Winter days touches both limits.
  - Overshoots are at most 1.0e-6 MWh beyond a limit, i.e. at solver tolerance.
- **TSO/DSO copy agreement:**
  - SoC max 1.45e-5 MWh over the 192 hours;
  - pnet max 1.33e-5 MW;
  - day-end SoC − start is −2.6e-11 or smaller (TSO) and −9e-13 or smaller (DSO).
- **The DSO copy's largest simultaneous charge and discharge**, min(pch, pdch), is 1.8e-4 MW (2035 Summer). On the TSO
  copy it is 9.7e-8 MW.
- **Same-pickle consistency:** every exported SoC/pch/pdch/pnet/qnet value of the eight blocks equals E6's
  `supporting_node7_all_blocks_row2` exactly (max diff 0.0).

**x = 0 (row 1): SoC is identically zero, stated from the model; no zeros exported.** Row 1's `certified_models.pkl`
(`99ab1070…`) covers 72 shared-ESS copies: the TSO indices for nodes 5/7/9 and each DSO copy, all 12 blocks. In every
one of them:
- `e_rated_fixed` = `s_rated_fixed` = 0;
- all 1,728 SoC entries are **fixed** at 0 (max |value| 0.0), and so is every pch/pdch/pnet/qnet entry;
- the 5,256 `sess_soc_def / _limit_upper / _limit_lower / _final` rows are **all deactivated** (0 active).

This is what `configure_shared_ess_operational_state` does at zero capacity. Row 2's zero-capacity nodes 5 and 9 (48
copies) show the same state.

### 3. SoH per block under the seven Table 6 calibrations (records only, no pickle)

**How each row was matched.**
- Each Table 6 row was matched to its frozen-table cell: `tables.ageing.rows`, plus `cells_appended_w160.e_soh050` for
  the 0.50 floor.
- Each cell was then matched through its frozen eval key to the committed `evaluation_record.json`: eval key = frozen,
  k* = frozen, status certified, 7/7.

**Calibration identity as read from each record.** "Entered" means `model_variant_applied_in_child`, bus 7.
"Applied" means the `model_variant_readback_pre_run` floor row and SoH point; the terminal readback agrees.

| Table 6 row | Cell (eval key) | Entered (N, δ, R) | k (record) | φ^Cal | SoH point | SoH^Min applied |
|---|---|---|---:|---:|---|---:|
| Baseline (cycling + calendar) | e_c2_calfade (`bbd82994…`) | (10 000, 0.8, 0.8) | 35,851.36 | 0.985 | end | 0.70 |
| Cycling only | e_c2 (`ba969ad2…`) | (10 000, 0.8, 0.8) | 35,851.36 | 1.000 | end | 0.70 |
| Retention 50 % | e_c3_unit (`409740fb…`) | (10 000, 0.8, 0.5) | 11,541.56 | 1.000 | end | 0.70 |
| Retention 50 %, mid-block | e_c3_midblock (`8b1eace9…`) | (10 000, 0.8, 0.5) | 11,541.56 | 1.000 | mid | 0.70 |
| Datasheet 8 000 cycles | e_c4 (`c59d7006…`) | (10 000, 0.8, 0.7) | 22,429.39 | 1.000 | end | 0.70 |
| No ageing | e_no_ageing (`1b10e9e1…`) | ageing disabled (D = 0 row; record carries R 0.5) | — (∞) | 1.000 | end | 0.70 (inactive) |
| Baseline, 0.50 floor | e_soh050 (`3f01b9ea…`) | (10 000, 0.8, 0.8) | 35,851.36 | 0.985 | end | 0.50 |

- **k.** k = N δ / (−ln R) reproduces the record's k exactly (difference 0.0) on all six ageing arms.
- **Printed Table 6.** Its k (rounded), φ, SoH point and SoH^Min match the records on all seven rows.
- **Datasheet row.** Table 6 prints (8 000, 1.00, 0.70). The record enters (10 000, 0.8, 0.7), the same N·δ = 8 000,
  which is what Table 6's footnote states.

**SoH per (block year), node 7, cohort 2025.**
- start = `soh_prev_end`; end = `soh_end`; mid = `soh_mid_closed_form`;
- used = `soh_used_for_available_energy`, which × E_rated (1 MWh) is E^Av;
- floor = the floor row is active at terminal.

| Calibration | Block | SoH start | SoH end | SoH mid | SoH used | EFC/day | Floor |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline (cycling + calendar) | 2025 | 1.000000 | 0.880458 | 0.938327 | 0.880458 | 1.0165 | no |
| | 2030 | 0.880458 | 0.779817 | 0.828611 | 0.779817 | 0.9000 | no |
| | 2035 | 0.779817 | 0.700000 | 0.738832 | 0.700000 | 0.6367 | yes |
| Cycling only | 2025 | 1.000000 | 0.938003 | 0.968506 | 0.938003 | 1.2573 | no |
| | 2030 | 0.938003 | 0.881174 | 0.909145 | 0.881174 | 1.2277 | no |
| | 2035 | 0.881174 | 0.835301 | 0.857931 | 0.835301 | 1.0503 | no |
| Retention 50 % | 2025 | 1.000000 | 0.871156 | 0.933357 | 0.871156 | 0.8723 | no |
| | 2030 | 0.871156 | 0.769752 | 0.818886 | 0.769752 | 0.7826 | no |
| | 2035 | 0.769752 | 0.700000 | 0.734048 | 0.700000 | 0.6007 | yes |
| Retention 50 %, mid-block | 2025 | 1.000000 | 0.871373 | 0.933473 | **0.933473** | 0.8707 | no |
| | 2030 | 0.871373 | 0.770691 | 0.819487 | **0.819487** | 0.7765 | no |
| | 2035 | 0.770691 | 0.700000 | 0.734496 | **0.734496** | 0.6084 | yes |
| Datasheet 8 000 cycles | 2025 | 1.000000 | 0.905177 | 0.951408 | 0.905177 | 1.2244 | no |
| | 2030 | 0.905177 | 0.823381 | 0.863311 | 0.823381 | 1.1640 | no |
| | 2035 | 0.823381 | 0.760154 | 0.791136 | 0.760154 | 0.9820 | no |
| No ageing | 2025–2035 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.3144 / 1.3537 / 1.2013 | no |
| Baseline, 0.50 floor | 2025 | 1.000000 | 0.872631 | 0.934147 | 0.872631 | 1.1919 | no |
| | 2030 | 0.872631 | 0.765059 | 0.817076 | 0.765059 | 1.0999 | no |
| | 2035 | 0.765059 | 0.677290 | 0.719838 | 0.677290 | 0.9092 | no |

- **"0.700000" at an active floor** is 0.69999999 in the record: the floor less 1e-8.
- **Cross-checks against the frozen table and printed Table 8 (all hold):**
  - **AE and EFC.** The frozen values are reproduced exactly by pv(SoH used) and pv(EFC/day), with
    w_y = 1/1.02^(y−2025), on the six frozen rows (difference 0.0). e_soh050 has no frozen AE/EFC entry; it gives
    AE 0.778107 and EFC 1.076295, which round to Table 8's 0.778 and 1.08. All seven rows round to Table 8's printed
    AE and EFC/day.
  - **Binding year.** The record's binding year (the first block with an active floor row) equals the frozen
    `floor_year_070`, and equals Table 8's printed 2035 / never on every row.
- **Baseline arm against the W101 reference unit.** e_c2_calfade's SoH trajectory equals the W101 row 2 record
  (`3f084f2f`) exactly: 7 fields × 3 blocks, max diff 0.0. The certified Q is also identical (653,620,162.5657823 €).
  The SoC export (row 2) and the SoH baseline therefore describe the same operating point.
- **The 0.50-floor EFC increases** over the baseline are +0.175, +0.200 and +0.273 EFC/day in 2025, 2030 and 2035.
  They agree with the manuscript's "0.18, 0.20 and 0.27".

### Guards and load count (`w188d_operational_figure.json` `guards`)

- **Solves:** `SolveProfileGuard(permitted=())` was installed before any project import. `verify(0)` returned no
  failures; counts 0 / 0 / 0 / 0.
- **Builds:** `Model.__init__` was blocked and called 0 times.
- **Pickle loads.**
  - **Exactly 2 delegated loads**, each of an allow-listed file, each hashed on the bytes unpickled:
    1. row 2, `d0a3d512…`, 164,685,741 B, load started 13:59:53 UTC;
    2. row 1, `99ab1070…`, 164,176,491 B, load started 14:00:02 UTC.
  - There were 6 load attempts and 1 `loads` attempt, with 5 refusals, exactly the declared number; there were 0
    `Unpickler` attempts.
- **Negative controls:** all raised, and none delegated.
  - NC1: a wrong sha256 on the allow-list was refused, with a sha mismatch.
  - NC2: `pickle.loads(bytes)` was refused.
  - NC3: `pickle.load(BytesIO)` was refused.
  - NC4: the off-list `e_c2_calfade/certified_models.pkl` was refused on its path, before any byte was read.
  - NC5: a second load of row 2's file was refused.
  - NC6: a second load of row 1's file was refused.
- **Unpickling imported, for name resolution only:** `branch`, `definitions`, `energy_storage`, `generator`,
  `helper_functions`, `hierarchical_coordination`, `load`, `model_construction_helpers`, `network`,
  `network_parameters`, `node`, `shared_energy_storage`, `solver_parameters`.

---

## Not confirmed

- **The cause of the 1.22e-6 MWh gap in 2025** between the model copies' E^Av and the record (the one-cycle-lag
  hypothesis). I did not trace it in the cycle records; it is the same gap E6 reported.
- **The manifest does not list the two `post_certification.json` files as inputs.** Their pins were read and compared
  (`pre_run_checks.pickle_pins` records both, equal to the allow-list), and both files are committed. But the script
  read them without registering them, so the manifest has 35 inputs instead of 37.
- **Table 6 / Table 8 printed values were typed into the script** from `main.tex` at a87e195 (l. 1324–1330 and
  1548–1561). They were not parsed from the file. The file's md5 is checked; the transcription is not.
- **Unpickling and Pyomo `construct`.** I did not check that unpickling never calls a Pyomo `construct`; the build
  guard covers `Model.__init__` only, and it read 0. This is as in E6.
- **Contacts at tolerances other than 1e-4 MWh.** I counted contacts only at the declared tolerance. The gap analysis
  above, with all contacts within 1.0e-6 and the nearest non-contact at 5.9e-3 MWh, shows that any tolerance in that
  range gives the same counts.
- **2030 SoC is not exported.** The order names 2025 and 2035 only.
