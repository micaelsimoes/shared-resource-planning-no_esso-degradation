# P5.15 W170 — how the 0.933 mean-profile prediction for the 3 × 3 ratio R was computed

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 68 decision 2 ("W170 records how the 0.933 prediction was computed, so
that §4.5 can say what it was a prediction of"). ZERO SOLVES, read-only. Nothing existing was edited.

## Blocked on Planner

None. One reading for the Planner to rule on is listed under "Unexpected finding" (the recorded R is a ratio of
weighted **averages**; the value ratio it is compared against is a ratio of weighted **totals**; the two differ by the
ratio of the discounted horizon weights, 0.9806).

## The answer (three lines)

1. **R** = V_3x3 / V_SRP1, with V = Q(x = 0) − Q(unit) under the gross convention (`gross_operational_cost`, settlement excluded) and unit = one storage at node 7, 0.25 MVA / 1.0 MWh (`n7_4h_e1`). Numerator: the 3 × 3 instance (`SRP1__s53_3x3.json`, 2a64e3c5), value at certification (cycle 72 for x0, cycle 69 for the unit), 236,695.25 €. Denominator: the SRP1 instance, V_SRP1 settled 253,539.62 € (T7 / T1; it was 259,375.33 at certification before Addendum 54). T11 reports the ratio as [(V − D)/V_SRP1, V/V_SRP1] = [0.909, 0.934].
2. **0.9331** is W52's `R_r2` for market subset [1, 2, 3] (`p515_s53_3x3_selection.py` @ 7550f441, line 321; record `selection_3x3.json` sha256 291be9c7…). It uses **market prices only**. Numerator: the probability-renormalized (1/3 each) mean of market scenarios 1–3 of the 5 × 5 paper case (bit-identical to the 3 × 3 instance's scenarios 1–3), on the 3 × 3 horizon (2025/2028/2031/2034/2037, three-year blocks, four seasons). Denominator: SRP1's single price scenario on the SRP1 horizon (2025/2030/2035, five-year blocks). Each side is the block-weighted average of the daily 4 h spread (mean of the 4 highest minus mean of the 4 lowest hourly prices), with weight w = num_years × num_days / 1.02^(y − 2025) (production's block weight). That gives 90.5314 / 97.0179 = 0.933142. No load, RES or operation scenario enters.
3. **The inputs are a mix by construction.** The numerator is exactly the 3 × 3 instance as run (its years, block lengths, market scenarios 1–3 at 1/3). The denominator is exactly SRP1 as run (its years, block lengths, one scenario). So the 0.9331 predicts, to first order, the cross-instance ratio of the discount-weighted average daily 4 h market-price spread. That ratio mixes the scenario-set effect (mean-profile flattening) with the horizon effect (different sampled years and block lengths). It does not isolate the scenario effect. At 2025, the only shared year, the same ratio is 0.962. It is also a ratio of weighted averages, not of the weighted totals that V compares; the totals form is 0.9151 (see Unexpected finding).

## Evidence

### Where 0.9331 was computed and recorded (search record)

Searched: `PLANNER_BRIEF_2026-09-13.md` (grep `0.9331|0.933`, lines 1862, 1984, 2016, 2186, 2245, 2274, 2292, 2347, 2390,
2962; Addenda 31, 32, 40, 44, 54, 68 read), root `P5_15_*.md` and `p5*.py` (grep `0.9331`: five reports; scripts W52 via
`R_r2`, W89, W90, W160, W164), stage artefacts `data/SRP1/Results/**/*.json|*.md` (grep `0.9331|0.933141801512`; hits
outside P515S53 are unrelated digit strings in settlement/G-file numbers), the T11 source fields of
`frozen_step6_tables_v1_590088fe.json`, and `git log -S'0.933141801512231'` (earliest commit 7550f441).

Chain of record:

| step | where | what |
|---|---|---|
| origin of the argument | brief Addendum 31 (l. 1340–1374) | "the ratio of the mean-profile spread to SRP1's 98 €/MWh is the first-order estimate of the paper-scale value per MWh relative to SRP1's"; non-anticipative schedule ⇒ expected value = value under the probability-weighted mean price profile |
| first prediction (5 × 5) | Addendum 32 (l. 1413–1416); W22 `p515_s47_market_spreads.py` @ 046d4d00, `market_spreads.json` e0806dc1 | R = 0.937 (mean profile 91.7 vs SRP1 98.0; "SRP1's scenario = paper scenario 1 in 2025") with the caveat that the storage sees the bus-7 marginal cost |
| 3 × 3 rule | Addendum 40 ruling 3 (l. 1705–1710); spec v23 `frozen_s53_spec_v23_39a07fd8.json` `ruling3_3x3` | R for the selected set "computed zero-solve from the selected set's renormalized mean profile, in the same way as the confirmed R = 0.937, and recorded BEFORE the run" |
| **computation** | **W52 `p515_s53_3x3_selection.py` @ 7550f441** (sha256 f129a01e…), record `data/SRP1/Results/P515S53/selection_3x3/selection_3x3.json` (291be9c7…; `.md` fdc8f7d6…; manifest 481d4097…) | ten 3-subsets ranked; subset [1, 2, 3] rank 4: spread_u 91.2994, spread_r2 90.5314, **R_u 0.9317, R_r2 0.933141801512231**, R_gn 0.9111; the selected set was [2, 3, 4] (R_r2 0.9391); §5 found production draws the prefix [1, 2, 3] |
| adoption | `P5_15_ADDENDUM40_42_REPORT.md` l. 165 ("accept the prefix and record R = 0.9331 … for a 0.6 % difference"); brief Addendum 44 l. 1862 | "3 × 3 confirmed with production's prefix draw [1,2,3] × [1,2,3] and R = 0.9331 recorded" |
| carried, not recomputed | `p515_s53_w89_3x3_campaign.py` l. 21, 162 (`R_PREFIX_RECORDED = 0.9331`); `instance_record.json` `prefix_draw` (a47dc358; `R_r2_rounds_to_recorded` true); `campaign_results.json` `value_and_R.R_prefix_recorded` (587d3f1a, commit 47a54c89) | 0.9331 |
| table | `frozen_step6_tables_v1_590088fe.json` `tables.three_by_three.rows[9]` | "R predicted from the mean-profile spread", 0.9331, source = campaign_results.json |

### (i) Definition of R (the measured ratio)

- `frozen_step6_tables_v1_590088fe.json` `three_by_three.formula_R`: "[(V − D) / V_SRP1_settled, V / V_SRP1_settled];
  V = 3 x 3 value at certification, D = the post-hoc settled descent of the 3 x 3 x = 0 cell (W99), V_SRP1_settled =
  253,539.62 EUR". Row 0: V = Q(0) − Q(unit) = 236,695.2493 €. Row 5: D = 6,234.94 €. Row 6: V_SRP1 = 253,539.6219 € (T7
  discount row 2 %, = T1 headline; `tables.ageing.rows[3]` cell `e_c2_calfade`).
- Objective convention (frozen JSON `objective_convention`): Q = `gross_operational_cost`, settlement excluded; value =
  Q(0) − Q(x).
- 3 × 3 cells (`campaign_results.json` `per_cell`): `x0` (nodes 5 / 7 / 9 at (0, 0)) Q = 842,832,534.76, certified at
  cycle 72; `n7_4h_e1` (node 7, 0.25 MVA / 1.0 MWh, 2025) Q = 842,595,839.51, certified at cycle 69.
- Production weighting of Q: `network_data.py:96-103` `get_primal_value` sums block objectives × num_years × num_days
  / (1 + DiscountFactor)^(y − y0). `shared_resources_planning.py:1234-1239` builds the gross cost from these. So V is a
  discounted **total** over the horizon.

### (ii) How 0.9331 was computed

- Per-block spread: `p515_s47_market_spreads.py:152-158` `day_spreads` (`top4_mean_minus_bottom4_mean`, H = 4, the
  unit's E/P). Block weight: `:160-162` `weights`, w_u = num_years × num_days, w_r2 = w_u / 1.02^(y − 2025)
  (`PRODUCTION_RATE` = 0.02; both case files have DiscountFactor 0.02). Horizon average: `:165-167` `horizon`, Σ w s / Σ w.
- Numerator: `p515_s53_3x3_selection.py:278-297` `subset_rows`, pm_S = pm[S] / Σ pm[S] (= 1/3; paper pm = 0.2 × 5), the
  mean profile mean_c = pm_S · c[S] per (year, day), then its spread. `:311-312` sr = horizon(mp, F_TH,
  'weight_model_r2'). Paper horizon (`selection_3x3.json` `instances.paper.years`): {2025: 3, 2028: 3, 2031: 3, 2034: 3,
  2037: 3}. Days: Spring 92, Summer 91, Autumn 91, Winter 91. Prices: read with production's reader from
  `SRP1__paper.json` (d726307c, scenario checksum 1e8bdd3e).
- Denominator: `p515_s47_market_spreads.py:94-95` `SRP1_Z4_SPREAD_R2 = 97.01788674187569`, which is W19 Z4
  (`p515_s46_zero_solve_reports.py:665-731` @ dd3afa6e; `zero_solve_reports.json` 16579f32…). That is SRP1's single
  scenario, years {2025: 5, 2030: 5, 2035: 5}, weight `tn.years[y] * tn.days[d] / (1 + tn.discount_factor)^(y − y0)`
  (l. 714).
- Ratio: `p515_s53_3x3_selection.py:321` `'R_r2': sr / W22.SRP1_Z4_SPREAD_R2`; docstring l. 41-44.
- The record says R is price-only: "The operation-scenario selection does not enter R (R is a price-only measure)"
  (`selection_3x3.md` §3). It also says: "This covers price only; the operation-scenario reduction (and any interaction
  with network constraints) is not captured by R" (§4). Addendum 32's caveat stands unresolved in the formula: the
  storage sees the bus-7 marginal cost (spread 80.6 at SRP1), not the market price.

**Reproduction (zero solves).** `p515_s53_w170_r_prediction_check.py` re-applies the recorded formula to the recorded
per-block inputs: W52's `per_block_spread` for [1, 2, 3] and Z4's `per_representative_day` rows. It recomputes no
price. Output: `data/SRP1/Results/P515S53/w170_r_prediction/w170_r_prediction.json` (sha256 5ca6469c…), `launch.log`
(6258130f…), `manifest_sha256.json`. SolveProfileGuard(permitted=()) verified 0, pickle blocked with 0 calls, exit 0,
31/31 checks True.

| quantity | reproduced | recorded |
|---|---|---|
| numerator spread_r2 (3 × 3 horizon, mean of scen. 1–3) | 90.53144561322347 | 90.53144561322347 |
| denominator spread_r2 (SRP1 horizon, Z4) | 97.01788674187569 | 97.01788674187569 |
| **R_r2** | **0.933141801512231** | 0.933141801512231 → 0.9331 |
| R_u (undiscounted weights) | 0.9317134866483382 | 0.9317134866483382 |

Each reproduction agrees to ≤ 1e-9.

### (iii) Inputs against the instances as run

| element | numerator of 0.9331 | 3 × 3 as run | denominator of 0.9331 | SRP1 as run |
|---|---|---|---|---|
| years (block length) | 2025/28/31/34/37 (3) | same (`SRP1__s53_3x3.json` `Years`, check True) | 2025/30/35 (5) | same (`SRP1.json` `Years`, check True) |
| days | 92/91/91/91 | same | 92/91/91/91 | same |
| discount | 0.02, w / 1.02^(y−2025) | DiscountFactor 0.02 | same | 0.02 |
| market scenarios | paper 1, 2, 3 at 1/3 | realized subset [1, 2, 3], 9,040 arrays bit-identical to the 5 × 5 rows, 0 mismatches, pm 1/3 (`instance_record.json` `prefix_verification`) | single scenario | single scenario |
| operation scenarios / load / RES | not used | 3 operation scenarios | not used | 1 |
| price seen | market price | — | market price | — |

So the numerator is the 3 × 3 instance's horizon and market draw exactly, and the denominator is SRP1's exactly. The
prediction is a prediction of the **cross-instance** value ratio as run. Both differences (scenario set, block length /
sampled years) are inside it. Readings from the committed per-year figures (not predictions, undiscounted):

| year | 3 × 3 mean-profile spread | SRP1 spread |
|---|---|---|
| 2025 | 80.06 | 83.22 |
| 2028 / 2030 | 81.90 (2028) | 98.01 (2030) |
| 2031 | 91.41 | — |
| 2034 / 2035 | 99.37 (2034) | 112.74 (2035) |
| 2037 | 103.75 | — |

At 2025, the one shared year (SRP1 2025 = paper scenario 1, Addendum 32), the ratio is 0.9620. That figure holds the
year fixed and isolates the mean-profile flattening of scenarios 1–3 at first order. The recorded growth-normalized
form R_gn = 0.9111 divides each block by (1.025)^(y − 2025) and is a second view of the same mix. The discounted mean
sampled year is 2030.64 (3 × 3) against 2029.67 (SRP1).

## Unexpected finding (for the Planner; not acted on)

- **Averages against totals.** R_r2 normalizes each instance by its own total block weight:
  [Σ w s / Σ w]_3x3 / [Σ w s / Σ w]_SRP1. V is a weighted **total** (production `get_primal_value`, above).
  - Under the same first-order model, a value ratio corresponds to (Σ w s)_3x3 / (Σ w s)_SRP1 = R_r2 × (Σ w_r2,3x3 /
    Σ w_r2,SRP1).
  - The undiscounted totals are equal: 5,475 = 5,475, so R_u is unaffected (0.9317).
  - The discounted totals differ: 4,878.82 against 4,975.09, a ratio of 0.98065. The totals-form figure is therefore
    **0.9151**, not 0.9331.
  - Both 0.9151 and 0.9331 lie inside the measured [0.909, 0.934]. I did not judge whether §4.5 should cite the
    averages form, the totals form or both; that is a reporting ruling.
  - Computed by the W170 script from recorded weights (`readings.ratio_of_weighted_totals_r2` = 0.9150840754554548);
    zero solves.
- **The selected set was not run.** W52's own recorded PREDICTION line names the *selected* set, market [2, 3, 4] ×
  operation [4, 5, 1], at R_r2 = 0.9391. The 3 × 3 that ran is the production prefix [1, 2, 3] × [1, 2, 3]. 0.9331 is
  that subset's row of W52's ranking table (rank 4 of 10), adopted in Addendum 44. Any manuscript wording should cite
  0.9331 as the row for the realized draw, not as W52's headline prediction.

## Not confirmed

- That V_SRP1 settled (253,539.62) is the same cell pair (SRP1 x0 against node-7 0.25 MVA / 1.0 MWh) as the 3 × 3 pair
  was read only from the frozen table's labels (T7 / ageing row `e_c2_calfade`, T1 "headline"). I did not trace the
  SRP1 cell definitions back to their campaign specs.
- How the first-order spread model maps to V (round-trip efficiency, ageing, the bus-7 marginal cost against the market
  price, network limits) is outside the record. The record states only that R is "first-order" and "price only".
  Nothing here tests that mapping.
- That production's prefix draw holds bit-for-bit was taken from W89's committed `instance_record.json`. It was not
  re-run (that would need a production read).
- The scoped search covered the brief, root reports and scripts, `data/SRP1/Results/**` JSON / MD, and git history for
  the full-precision value. Uncommitted scratch outside `data/SRP1/Results/` and `.claude/worktrees/` was not searched.
