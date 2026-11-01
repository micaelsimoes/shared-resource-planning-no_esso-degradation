# P5.15 Step 6 package: results tables, certification, limitations, reproducibility, scorecard, reviewer map

**Planner handoff, 2026-10-05.** For the Author and the External Expert. This is the package Addendum 64 orders. **The
campaign stops here for review:** the author and the expert review the tables before any manuscript text is written.

- **Authority:** `PLANNER_BRIEF_2026-09-13.md` Addendum 64.
- **Evidence:** every number is read from committed records, with zero solves. Each source is given as a commit and a
  file.
- **Tables:** `data/SRP1/Results/P515S53/w157_step6_tables_a64/w157_step6_tables.{md,json}` (`b913ea94`). Builder
  `p515_s53_w157_step6_tables.py` (`4c89daf8`), every check True. Cited below as **T1–T10**. The version built before
  the Addendum 64 cells ran (`864679b2`) is superseded by it.
- **Draft history:** Worker draft `P5_15_STEP6_PACKAGE_DRAFT.md` (`6880a573`), assembled and completed here by the
  Planner.
- **Objective convention on every value:**
  - Q = `gross_operational_cost`, settlement excluded (primary);
  - net = Q − terminal salvage credit, shown beside;
  - Q_cc = Q + t_sum, report-only;
  - value = Q(0) − Q(x); F = Q + I;
  - τ = 4,539.07 €.
- **Determinacy threshold:**
  - between certified cells: max(3 × larger band, 2τ) (Addendum 61);
  - with an uncertified cell: 3·max(|gap|, |slack|) (Addendum 58);
  - "×" is |margin| / threshold.

## Status after Addendum 65

**Accepted.** Rulings applied in the text below:
1. the net substitute is accepted; the label stays; net is secondary;
2. ten certificates at ≥ 0.95 τ, with the addendum's sentence; `i_5a6a88b4` flagged beside;
3. `d_36686489` attributed to TSO recoveries; the Addendum 62 mechanism is withdrawn for this cell;
4. threading, banners and versions sourced by the closing reads (W159 `62749030`); R3.6 answered by the addendum's
   sentence; the monotone drift at L = 60 stays unsourced.

**One wording point for the expert.** Addendum 65 asked for a sentence that HSL_MA97 gives bit-compatible results
independent of the thread count (Hogg & Scott), confirmed against the author's `coinhsl-2023.11.17` archive.
- **The archive holds no documentation.** It has source code and licences only.
- **The statement exists only in the HSL website specification** (v2.8.1, Section 1). There it carries a caveat
  that it depends on the BLAS library, and it cites Hogg & Scott 2011 (RAL-TR-2011-024), not 2013.
- **It is not needed.** Every evaluation ran with one thread, and the installed HSL library has no OpenMP. The
  reproducibility paragraph therefore states serial execution as recorded, and cites no HSL claim.

## Blocked on the author

Nothing. The review is the next step. The Mac is idle, and no run is queued.

## What changed with the Addendum 64 cells

| cell | result | commit |
|---|---|---|
| `g070_neutrality` (the minimum-SoH path at 0.70) | **G28 bitwise = 3f084f2f through 172**: the router patch and wrappers are neutral | `0e1d7c0e` |
| `h_x0_m175`, `h_unit_m175` | certified, k\* 139 and 121 | `ae2b6574`, `6b5e5e33`, summarize `112bb529` |
| `e_soh050` (minimum SoH 0.50) | certified, k\* 184 (0.916 τ). Its trajectory diverges from 3f084f2f at cycle 1 and by more than τ/10 at cycle 4 (G30); every sidecar line carries soh_min 0.5 | `5fb81cc4`, summarize `52d416cb` |

Two Workers ended on API errors during this stage. No run was lost or repeated: `e_soh050` was launched once and
closed out by a successor.

---

## (i) Results tables — pointer and the figures the manuscript carries

### Where the tables are

The tables are built by `p515_s53_w157_step6_tables.py` (`4c89daf8`) and written to
`data/SRP1/Results/P515S53/w157_step6_tables_a64/` (`b913ea94`, with the Addendum 64 rows filled from
`w155_summary_after_04_e_soh050.json`, its manifest and launch log).

**What the build checked:**
- zero solves, with 29 zero-permit guards verified at 0;
- pickle blocked;
- 14 inputs committed clean, and 12 input manifests matched;
- exit 0.

**The tables:**

| table | content |
|---|---|
| **T1** | 60 claims: gross d, rule, threshold or bar, ×, verdict; net d, net ×, net verdict, net label; Q_cc d and ×, report-only; certification status of both cells; eval and candidate keys; ruling notes |
| **T2** | 49 cells: status, branch, k0, k\* or end, range/τ, ≥ 0.95 τ flag, turning points on non-clean cycles, uncertified cause, gap, slack, band, replay-gate cycle, certifying spec, keys |
| **T3** | break-even fit |
| **T4** | year ladder |
| **T5** | Phase B |
| **T6** | benchmark |
| **T7** | discount |
| **T8** | ageing arms |
| **T9** | dead zone |
| **T10** | the Addendum 64 rows: m = 1.75 pair and the minimum-SoH 0.50 row |

**How the Addendum 64 rulings were applied:**

1. **Slope = b + c/4** (ruling 3).
2. **Break-even, conservative figure** (ruling 4). The interval for `d_4a82a64a` was added by W145's own function
   `banded_breakeven_fit` on W145's committed points. No code change was needed.
   - The committed W145 result was reproduced **bitwise** first.
   - Manuscript figures: break-even ≤ **192,557 €/MWh**; margin to energy cost ≥ **61,320 €/MWh**.
   - The hand arithmetic, labelled as a cross-check, gives the same 61,320.
3. **J 4 MWh at its uncertified bar** (ruling 4): bar 15,879.58, **11.12×**, determinate.
4. **E:C2_calfade vs C3 is within resolution** (ruling 5): +9,329.27 at 0.71×.
5. **Year ladder net −2,286.25** (W154b), within resolution at 0.19×.

**Net label:** every net figure reads "validated by form + salvage identity (W154b)". The 8 G rows are the exception;
they carry recorded net values.

### Results the manuscript carries (from T1–T8; gross primary, net beside where ruled)

| item | result | source |
|---|---|---|
| Uncoordinated benchmark | coordination beats the best static no-reverse-flow arrangement (the price-taker arm) by **+90,896,608.40 € (13.9 % of the coordinated Q181)**, 1,263.75× the larger band 71,926.11, determinate. 4 reverse-flow interface-hours (3,308.30 MWh, block-weighted). Sweep without any interface rule: the TN cannot accept the DN exchange in 1/12 blocks (6 h) passive, 8/12 (25 h) price-taker | T6; `report_v3.json` (`8d42dfb8`, spec v5 `bca69f97`) |
| Benchmark decomposition | TSO +70,166,588 (77 %), DSO +20,730,020 (23 %); 0.81 TWh of TN renewable output left curtailed by the price-taker arm | Advisor review and W124 (`0c3451cf`), as recorded in `TASKS.md` Addendum 57 section; `P5_15_ADDENDUM57_BENCHMARK_AND_RESETTLE_REPORT.md` (`5612b8f1`) |
| x = 0 at SRP1 (headline) | value − I = **−64,417.39**, 4.93×, determinate (V = 253,539.62) | T1 `CHECK:headline_V_minus_I_settled` |
| Break-even fit (node 7) | certified-only e\* 182,701.62; banded e\* range 173,486–190,380 (3 intervals); **conservative 171,309–192,557, margin ≥ 61,320 €/MWh** (4 intervals); energy cost 253,877.68 | T3; W145 `7c1a4ecb` |
| Flexibility ladder, m = 2 | 1 MWh value − I **+46,285.46** (3.77×); second MWh **+46,514.76** (3.64×); 2 MWh value − I +92,800.22 (7.26×) | T1 H, I rows |
| Flexibility ladder, m = 1.5 | −14,450.28, within the uncertified bar 18,751.08 (0.77×) | T1 `H:m1.5` |
| Flexibility ladder, m = 1.75 | value − I **+14,249.16** (Q_cc +12,878.32), **determinate 1.32×** (threshold 10,775.40); both cells certified. **Ladder: pays at m = 1.75 and m = 2; break-even between 1.5 and 1.75** (≈ 1.63 by linear interpolation) | T10; `112bb529` |
| Energy ladder, m = 2 | 2→3 +40,972.63 (1.48×); 3→4 +42,807.46 (1.55×); 4→5 +51,820.48 (1.95×), all determinate. Value − I at 3 / 4 / 5 MWh: +133,772.85 (4.83×), +176,580.30 (manuscript: **11.12× at its uncertified bar**), +228,400.78 (8.60×) | T1 J rows |
| Phase A unit claims | every D-cell F difference determinate, 4.39–45.94× | T1 B rows |
| C cells | +102,096.79 (8.09×); +93,877.52 (7.00×; net +84,022.96) | T1 C rows |
| G cells (year × size) | gross +99,027.04 to +223,582.70 (7.73–17.71×); net +85,088.88 to +113,065.99 | T1 G rows (net recorded) |
| Year ladder 2035 − 2030 | gross **+42,458.65**, determinate 3.46×; net **−2,286.25**, within resolution 0.19× | T4; W154b `cd8fd8a8` |
| Phase B | all determinate under the v6 rule, 3.40–4.15×; pb_y2025_n5 v6 re-run +49,692.73 (3.94×) | T5; T1 `C:y2025__n5_p0.25_e0.5` |
| L (F2 certificate) | 12 neighbour differences all positive; 7 determinate, 5 within the uncertified bar. Net of salvage, one more is determinate: `L:y2030__n5_p0.25_e0.5__n7_p1_e3`, +34,120.26 vs 27,900.27. Separately, the F2 challenger row is −6,107.64, within the uncertified bar (0.22×) | T1 L rows; W153 `48e76c9f` |
| Ageing at soh_min 0.70 | no aged arm pays (−31,607.24 to −73,746.66, all determinate); no ageing −4,140.54 (within resolution) | T8; T1 E rows |
| soh_min 0.50 row | Δvalue = value(0.50) − value(0.70) = **+4,916.67** (Q_cc +4,802.26), **within resolution 0.38×** (threshold 13,054.22). Value − I at 0.50 **−59,500.72, determinate 4.71×**. The floor never binds at 0.50 (2035 SoH_end 0.677). EFC/day 1.19 / 1.10 / 0.91 against 1.02 / 0.90 / 0.64 at 0.70 | T10; `52d416cb`; W153 `48e76c9f` (0.70 EFC) |
| Discount rate | value − I negative and determinate at 0 / 2 / 5 / 8 %: −40,837.32 / −64,417.39 / −92,724.89 / −114,590.38 (2.57× / 4.93× / 7.10× / 8.78×) | T7; W153 `48e76c9f` |
| Salvage convention | 0 sign changes in 60 claims; 1 verdict change (the L row above) | W153 `w153_salvage_addback.json` totals |

### Two sentences for the new rows (accepted, Addendum 65)

1. Flexibility ladder: *the unit pays at a flexibility price 1.75 and 2 times the base price (+14.2 k€ and +46.3 k€,
   determinate) and breaks even between 1.5 and 1.75 (≈ 1.63 by linear interpolation)*.
2. Minimum SoH: *lowering the end-of-life floor from 0.70 to 0.50 releases cycling in every year (EFC/day +0.18,
   +0.20, +0.27) but adds only +4.9 k€, within resolution; the unit still loses 59.5 k€ determinately, so the floor
   is not what makes storage uneconomic at SRP1*.
   Source of the EFC/day differences: T10 (0.50) against W153 `48e76c9f` (0.70 per-year EFC).

### The two sentences Addendum 64 requires verbatim

1. *without ageing the unit is at break-even (−4.1 k€, within resolution); with the baseline calibration and the 0.70
   floor it loses 31.6–73.7 k€ across the aged arms, determinately*
   - Figures: T8; T1 `E:n7_4h_e1_no_ageing:value_minus_I` −4,140.54 (0.32×).
   - Aged arms: C2 −31,607.24 … C3_unit −73,746.66.
2. The coordination interpretation sentence (Addendum 58): *the benefit is the value of dispatching DN flexibility
   against the TN's marginal value λ_t rather than the wholesale price π_t; coordination — or a locational real-time
   signal computed by the TSO, which is coordination by another name — delivers it, a static rule with wholesale
   exposure does not.*

---

## (ii) Certification paragraph — draft text

> **Stopping and certification.** Each recourse evaluation runs consensus ADMM until the primal and dual residuals of
> every consensus channel pass Boyd's absolute-plus-relative test [Boyd et al. 2011, §3.3]. The tolerances are
> ε_abs = 10⁻⁵ and ε_rel = 10⁻⁴, and every local NLP must solve successfully in the same cycle.
>
> Because the objective converges more slowly than the residuals, passing the residual test is not taken as
> settling. After the first residual pass k₀, the certifying regime is held fixed: Anderson acceleration off, the tight
> interior-point tail on (complementarity tolerance 10⁻⁶), penalty frozen. A cell is then certified at the first cycle
> k\* at which all of the following hold:
>
> 1. the objective has shown at least three turning points (extrema) since k₀, so that its period is measured
>    rather than assumed;
> 2. the successive half-swings are not growing. Swings smaller than τ/10 = 453.91 € are treated as noise: they are
>    neither compared nor registered as turning points;
> 3. the range of Q over the last W = max(20, ⌈1.1 P̂⌉) cycles is at most τ = 4,539.07 €;
> 4. the priced interface-consensus gap satisfies |t_sum| ≤ τ/2 = 2,269.53 €;
> 5. every cycle in the window the test reads was solved cleanly. A local solve is clean when it ends `Optimal`, or
>    `Acceptable` on a primary attempt with all four IPOPT error metrics within 10× the tight-tail tolerances.
>
> A monotone branch certifies a cell with no oscillation in 2 P_max = 60 cycles when:
> - its steps are decreasing over the window;
> - the window range is at most τ;
> - |last step| × 60 ≤ τ, a linear bound on the remaining descent with no fit.
>
> τ is set from the value resolution the paper claims: δR = 0.07 of the SRP1 storage value, split over the four cells
> of a ratio.
>
> **What τ bounds.** Ten of the certificates the tables use stop within 5 % of τ; the rule bounds each cell's
> stopping error at about τ, not well inside it. Cells continued past certification moved at most 0.9 τ.
>
> **Replays.** Every gated cell replayed its original run bitwise up to k₀.
>
> **Uncertified cells.** A cell that does not certify by its cap is reported in an uncertified form, and a difference
> involving it is called determinate only if its margin exceeds three times the larger of its consensus gap and its
> settling slack, in both gross and gap-corrected terms.
>
> **Determinacy between certified cells.** A difference between two certified cells is determinate only if it exceeds
> max(3 × the larger band, 2τ).
>
> Across the 42 re-settled SRP1 cells, 32 certified (24 oscillatory, 8 monotone). The 10 uncertified cells divide as
> follows: 5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the growth test. The median
> certification cycle was 174, and certification came a median 65 cycles after the first residual pass (range 21–89).
> The median range/τ at certification was 0.86. The four cells added under Addendum 64 all certified.

### Sources for each element of the paragraph

**Rule components:**

| element | value | source |
|---|---|---|
| Boyd residual test, every channel, AND local solves OK | ε_abs 1e-5, ε_rel 1e-4 | `data/SRP1/SRP1_params.json` `admm.tol.boyd` (line 42); `shared_resources_planning.py:3377–3385`. These are the configured values; the per-cycle `boyd_eps_*` record fields were not read for this draft |
| Holds after the first residual pass | AA off, tail on, ρ frozen | stage spec v6 `96c23404` `holds_after_first_residual_pass` (`689340d2`) |
| Tight tail | compl_inf_tol 1e-6 | Addendum 48; v6 spec `configuration.current_production` |
| Three turning points; window W; range ≤ τ | W = max(20, ⌈1.1·(T[−1].t − T[−3].t)⌉) | Addendum 53 Ruling 1; v6 spec `stop_rule.W`; three-turning-point reading frozen in v39 `8a612429` (`TASKS.md`, Addendum 54 section) |
| τ = δR·V_SRP1/4 | 0.07 × 259,375.33 / 4 = 4,539.07 € | Addendum 53 Ruling 1; `settling_criterion_v6.TAU` |
| Swing floor τ/10 | applied to the growth test (A) and to turning-point registration (B) | Addendum 61 ruling 1; floor replay changed no committed certificate, 14/14 (W142 `689340d2`) |
| Gap clause | \|t_sum\| ≤ τ/2 = 2,269.53 € | Addendum 57 Decision 2 |
| Monotone branch | L = 60 = 2·P_MAX; steps decreasing; \|last step\|·L ≤ τ | Addendum 57 Decision 2; v6 spec `stop_rule.W.monotone`, `p_max` |
| 10× clean-exit rule | clean = `Optimal` on any tier, or `Acceptable` on a primary attempt with all four metrics ≤ 10× tolerance (see below) | Addendum 60; W139 (`2b9e3243`), as recorded in `TASKS.md` line 36 |
| Non-Optimal window rule | the certifying window is the set of cycles the test reads at k\* | Addendum 59 reading (γ) + supplement reading (a); window reads enumerated per certificate (Addendum 61 ruling 3) |

The four metrics and their tolerances under the 10× rule:
- NLP error, scaled, vs tol 1e-5;
- dual infeasibility, unscaled, vs 1;
- constraint violation, unscaled, vs 1e-4;
- complementarity, unscaled, vs 1e-6;
- the ESSO is judged on its own options.

**Measured movement and replays:**

| element | value | source |
|---|---|---|
| Post-certification movement ≤ 0.9 τ | cell 3 (`b_4649234b`) −3,351 (0.738 τ); `d_c52e1670` −3,543 (0.781 τ), or −4,004 (0.882 τ) on the last-pair reading | Addendum 61 item 2 ("0.74–0.88 τ"); W141 `d02efe69` and W142 `689340d2` (`TASKS.md`) |
| Bitwise replays | every gated cell through its k₀ (T2 column "replay bitwise through"); 3 × 3 x = 0 72/72; SRP1 references 3/3 | T2; Addendum 52; Addendum 54 |
| Uncertified form and its bar | bar = 3·max(\|gap\|, \|slack\|), both terms | Addendum 58 Ruling 1; v6 summary `definitions.uncertified_form` (`b95ee706`) |
| Determinacy floor | max(3 × larger band, 2τ) | Addendum 61 ruling 2; `p515_s53_w142_determinacy.py` |

**Certification statistics** (W153 `48e76c9f`, `w153_certification_statistics.json`; 42 cells, excluding the three B
cells):

| statistic | value |
|---|---|
| Uncertified, by cause | gap clause: `h_f9eae48f`, `j_5f3cccb4`, `l_45aa25a6`, `l_7c455554`, `l_b2251bc5`; lapse reset: `d_a12d95a2`, `l_0ee93aca`; growth test: `d_36686489`, `d_f759dd48`, `j_f3aa335e` |
| k\* | min 138, q1 156.75, median 174, q3 198.25, max 400 |
| k\* − N (first residual pass) | min 21, median 65, max 89 (W153 `k_star_minus_N`); measured from k₀ at decision (reset by lapses on `e_c3_midblock`, `e_no_ageing`): median 63.5 |
| range/τ at k\* | min 0.012, median 0.860, max 0.997 |
| Cells at range/τ ≥ 0.95 (W153 set) | 6: `d_3632b0ae`, `d_c7fee8be`, `d_9246ed01`, `c_6597a79d`, `e_c2_calfade`, `d_c52e1670` |
| Terminal step / EPS0 (certified) | median 1.72, max 10.42; 22 of the 32 certified cells above 1 (25 of all 42) |
| Vetoes | 46, in 2 cells (`d_f759dd48` 17, `pb_y2025_n5_v6` 29) |
| Non-clean cycles after N | 21 cycles in 10 cells, **all TSO** |
| Acceptable-clean events after N | 119, **all DSO** |
| Wall time | median 25.6 s per cycle; 220,876 s over 8,570 cycles |

**The manuscript count is ten (Decision 2).** The W153 set's six are listed above. **Outside W153's set: three more
certificates at ≥ 0.95 τ (T2).** Computed here from the summary bands:
- `b_2a0ba8b2` 0.970 (certified under v4);
- `b_4649234b` 0.989 (certified under v5);
- the settled unit reference `bd504ecf` (3f084f2f) 0.959 (W103, `TASKS.md`).

On the cells T5 lists, three W118 Phase B certificates are also above 0.95 τ:
- `pb_y2030_n7` 0.994;
- `pb_y2025_n5` 0.993 (superseded);
- `pb_y2025_n7` 0.974.

The handback's "six certificates stop at ≥ 0.95 τ" is therefore scoped to W153's 42 cells.

**Flagged certificates (Addendum 64 ruling 4).** Two certificates rest on a turning point at a non-clean cycle:

| cell | non-clean turning point | uncertified form beside |
|---|---|---|
| `j_a11d7966` | 138 | bar 15,879.58 |
| `d_4a82a64a` | 108 | bar 70,743.93 |

Source: T2, and W153 `certificates_with_a_turning_point_at_a_non_clean_cycle`.

---

## (iii) Limitations paragraph — draft text

> **Degenerate interface duals.** When storage discharge drives the transmission network's conventional generation to
> its lower bound while distribution flexibility sits at a kink, the interface dual is set-valued (λ ∈ [0, c_flex]).
> ADMM's consensus then converges at the pace of a degenerate dual. This is a known property of the method, not a
> defect.
>
> Certification reports these cases as "objective settled; interface-consensus gap unresolved". They appear with large
> storage and a high flexibility price. They share one fingerprint:
> - a priced gap of −8.8 to −9.4 k€;
> - split about 55 % at node 5, 31 % at node 9 and 14 % at node 7;
> - the power-flow primal residual frozen near 0.69.
>
> At the baseline price, the largest storage cell (5 MWh) fails to certify for a different reason. Repeated
> transmission solver recoveries break the growth test; its consensus gap is small (+1.1 k€) and of the opposite
> sign (Decision 3).
>
> **Solver recoveries.** In a few transmission blocks, some solves reach the iteration limit and recover only to
> IPOPT's acceptable level. These cycles are excluded from certification evidence. The recurring block is 2035 Spring.
>
> **RES bound slack.** RES output is bounded by availability plus a 10⁻⁵ pu numerical slack. The resulting
> over-production totals ≈ 62 MWh-equivalent over the horizon (≈ 10⁻⁵ of renewable energy) and is reported separately.
>
> **Monotone branch.** The monotone branch bounds the remaining descent linearly, without a fit. A cell drifting
> with a half-life longer than its window can certify with drift left beyond τ (≈ 3 τ measured on C\* at the earlier
> window L = 44).
>
> **[Hypothesis, not a result]** The 0.70 SoH floor raises the elasticity of value to available energy (ε_AE 1.04–1.85
> against 0.41–0.62 for the same three resolvable arms at 0.50). This is read as a late-life tail effect: the floor removes the low-SoH years whose throughput
> is least valuable. The decomposition that would show it has not been measured.

### Sources for the limitations paragraph

**Dual dead zone:**
- Mechanism: Addendum 58 Ruling 1; W127 `2f90194b`.
- Fingerprint: T9, from the v6 summary `t_by_node_at_cap` and `pf_primal_ratio_after_k0.last` (`b95ee706`).
- Measured t_sum at the cap:

  | cell | t_sum at cap (€) |
  |---|---|
  | `j_5f3cccb4` | −8,847.90 |
  | `l_45aa25a6` | −9,197.61 |
  | `l_7c455554` | −9,247.76 |
  | `l_b2251bc5` | −9,401.86 |
  | `l_0ee93aca` | −9,300.09 |
  | F2 incumbent | −9,234.42 |
  | F2 challenger | −9,300.13 |

- Node shares (n5 / n7 / n9) are 55 / 14 / 31 % in each of the five m = 2 cells of T9 that carry a per-node record
  (`j_5f3cccb4` and four L cells). The F2 references carry none in the summary.
- pf_primal last 0.6819–0.6999.
- The m = 1.5 cell `h_f9eae48f` has about the same split (55 / 14.5 / 31 %) at −2,481.75. W149 `241e299c` calls it "partially reproduced". The
  TSO-lever column is report-only and post hoc (ruling 7b).
- `l_0ee93aca` is entered by its signature, with the cause stated: two TSO recoveries, lapse resets at 218 and 222
  (ruling 7a).

**Large storage at the baseline price (`d_36686489`):**
- Addendum 62 decision 1; W143 `354b362a`.
- Uncertified by the growth test after five non-clean TSO recoveries.
- T9 records its t_sum at the cap as **+1,082.23** (node split 58/11/30 %). This is the opposite sign to the m = 2
  fingerprint.

**TSO recoveries:**
- W153: 21 non-clean cycles after N in 10 cells, all TSO.
- Blocks, tallied here from W153's per-cell details:

  | block | events |
  |---|---|
  | 2035 Spring | 6 |
  | 2030 Autumn | 4 |
  | 2030 Winter | 4 |
  | 2035 Summer | 3 |
  | 2025 Spring | 2 |
  | 2035 Winter | 2 |

- The 2,100–6,100× complementarity range is the handback's (`0a56ba88`); not re-read here.
- The cause is not investigated; the logs are preserved and the item goes to the post-revision cleanup (Addendum 64).

**RES slack ≈ 62 MWh-equivalent:**
- Addendum 56 gives the sentence.
- The figure is the sum of the negative parts at 1 €/MWh, block-weighted: TSO −19.24 + DSO −42.67 = −61.91 (W109
  `f6e3533f`, `w109_tso_curtailment_look.json` `tso`/`dso.neg_part_eur_at_1_block_weighted`).
- Instance: settled x0 `d110bd1a…`, cycle 181.
- Slack line: `model_construction_helpers.py:188`.

**Monotone-branch caveat:**
- `P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md` (`ce96d492`) line 68: "≈ 3 τ left whenever the decay half-life exceeds
  its 44-cycle window".
- W110 (`TASKS.md`): C\* half-life 102 against L_MONO 44, "≈ 3 τ gross drift left".
- Both figures are for L = 44. The current window is L = 60 (Addendum 57); the remaining drift at L = 60 is not
  measured.
- The 8 monotone certificates of this campaign are listed in T2. Seven are L cells with range/τ 0.012–0.171.
  **`i_5a6a88b4` has 0.939** (flagged in T2 beside the ten, Addendum 65). The remaining drift of the monotone
  certificates at L = 60 is not measured; seven of the eight have range/τ ≤ 0.171 and are not at issue.
- **[SOURCE NOT FOUND: the remaining drift of the 8 monotone certificates at L = 60; searched the W153 certification
  statistics, the v6 summary reports, `TASKS.md`]**

**Late-life-tail hypothesis:**
- Addendum 64 ruling 6.
- ε_AE: extension summary `item_E.statements.S5` (`184229bb`): 1.04 (no ageing), 1.30 (C2), 1.85 (C4); C2_calfade and
  C3_midblock not resolvable.
- The handback's "Not confirmed" section states the decomposition was not measured.

**Coordination does not fail in any block:**
- W130 `2c1b731a` (`TASKS.md` line 14): no block has a negative benefit in either arm.
- The two non-H1 blocks are 2025 Winter +2,306,563 € and 2025 Autumn +2,801 €. Both are beyond the DSO multimodality
  band, which is ≈ 0.
- So Addendum 58's "Q(x = 0) upper bound" limitation is **not triggered** and needs no sentence.

---

## (iv) Reproducibility note — draft text and sources

> **Determinism.** All results were produced on one machine with one solver build, one evaluation at a time, and
> single-threaded: the harness sets OMP, MKL, OpenBLAS, vecLib and NumExpr to one thread in every evaluation's
> environment and records it per evaluation, and the installed HSL library is built without OpenMP, so MA97 ran
> serially. Replaying a run reproduces its trajectory bitwise:
> - the 3 × 3 multi-scenario x = 0 cell replayed its 72 recorded cycles bitwise (72/72);
> - the three SRP1 reference continuations replayed bitwise to their certification cycles (3/3);
> - every gated re-settled cell replayed its original record bitwise up to its first residual pass before continuing.
>
> **Solver exits.** Two distribution blocks (DSO7 2025 Winter and DSO5 2035 Winter) end some primary solves at IPOPT's
> acceptable level, within 10× the tight-tail tolerances; they are counted as clean under the rule stated above.
>
> **Offsets common to every cell.** Two effects cancel in every reported difference:
> - **the tight tail:** tightening the interior-point tail (complementarity 10⁻⁴ → 10⁻⁶) lowered the certified
>   objective of each of the three SRP1 references by ≈ 1.1 × 10⁻⁶ relative, with the same sign;
> - **the RES slack:** the 10⁻⁵ pu RES slack lowers Q by ≈ 7.9 k€ first-order (≈ 1.2 × 10⁻⁵ of Q).

**Bitwise replays:**

| item | value | source |
|---|---|---|
| 3 × 3, 72/72 | "Replay **bitwise 72/72** — the multi-scenario instance's reproducibility statement" | Addendum 52; stage-1 evidence `4689475e`; report `P5_15_ADDENDUM51_CONTINUATION_REPORT.md` (`58ff8d88`) |
| SRP1, 3/3 | "Replays bitwise 3/3": x0 through 132, unit through 112, C\* through 87 | Addendum 54; `P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md` (`ebe34021`); W101–W104 records |
| Every gated cell at SRP1 | per cell in T2 "replay bitwise through" (= k₀ for every gated v6 cell; F2 pair 172 / 152; W118 Phase B in T5) | v6 summary `replay_bitwise_through` / `replay_first_divergence` (`b95ee706`); extension summary (`184229bb`); W118 summary |
| Neutrality gate G28 | `e_c2_calfade` reproduced the settled unit 3f084f2f bitwise through 172 | extension summary `c2_calfade_consistency` (`184229bb`) |
| Common offsets | tail −1.04 to −1.20 × 10⁻⁶ relative on 3/3 references; RES slack ≈ −7.85 k€ first-order (TSO −2,373.97 + DSO −5,474.11 priced negative parts) | Addendum 48; W109 `f6e3533f` (`TASKS.md` line 131) |

**One old-configuration figure remains (Addendum 65 read (a)).** T8's column "ε_AE (0.50, superseded)" comes from the
0.50-era ageing batch (`P515S46/ageing_mechanism.json`, `b5eca2a2`): C3 calibration, soh_min 0.50, E/P up to 10 h for
the C3 point and Q(0), no tight tail. It is shown only as the superseded comparator. Every other figure in T1–T10 is
from a current-configuration certificate or a settled reference (61 evaluations traced, W159).

**The Acceptable blocks:**
- Clean Acceptable events after N in this campaign (W153 `48e76c9f`):
  - DSO5 2035 Winter: 109 events;
  - DSO7 2025 Winter: 10 events.
- They fall in `d_f759dd48` (114) and `c_156ce2d1` (5).
- Earlier, under v5: DSO7 2025 Winter at 2.51–2.69× and DSO5 2030 Winter at 2.51× (W139, `TASKS.md` line 36).
- The "1.2–4.2×" range in the handback (`0a56ba88`, line 184) was not re-read from the per-cycle metrics for this
  draft.

**Machine and solver provenance (from committed records only):**

| item | value | source |
|---|---|---|
| IPOPT | **3.14.18** (aarch64-apple-darwin24.5.0, ASL 20241111), `/usr/local/bin/ipopt`, sha `b316abbe…` = the v6 spec pin; mtime 2025-07-09 | W159 `62749030` (`w159_closing_reads.json`); v6 spec `96c23404` |
| Linear solver | **MA97** (network NLPs), **MA57** (ESSO), from the campaign's own preserved logs: `…/p515s44_s53_w142_resettle_v6_d_4a82a64a_14b00a04ffbcfd33_run/logs/optim_log_case9_2025_Summer.log` line 26 (sha `f451d4af…`) "This is Ipopt version 3.14.18, running with linear solver ma97."; `optim_log_esso_node5_cycle001.txt` line 8 (sha `830ff5a9…`) "… ma57."; e_soh050 likewise. All 495 logs of d_4a82a64a: 6,887 ma97 / 429 ma57 banners. Hashes match the committed launch manifests. P5.12-R banner = corroboration | W159 `62749030` |
| HSL library | `/usr/local/lib/libcoinhsl.2.dylib` (sha `230fffdd…`, v5.5.0) = the copy in the author's `ipopt3.14.18hsl5.5.0arm64` package; no OpenMP link or symbols; MA97 source lines consistent with CoinHSL 2023.11.17 (v2.8.1), not proven | W159 `62749030` |
| Interpreter | `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python` | v6 spec `launch_commands` |
| Python / Pyomo / macOS | 3.11.11 (conda-forge, Clang 18.1.8) / 6.9.5 / macOS 27.0 (26A428), arm64 — **read 2026-10-06; environment unchanged since the campaign** (conda-meta/history: one entry, creation 2026-09-09; 0 environment files modified in the window 2026-09-30T21:25Z – 2026-10-05T13:42Z; last macOS update 2026-09-18, before the window) | W159 `62749030` |
| Machine | Mac Studio (arm64), 32 GiB | v6 spec `memory_preflight` |
| Concurrency | 1 (one evaluation at a time) | v6 spec `configuration.concurrency` |
| Threading | **1** for OMP / MKL / OpenBLAS / vecLib / NumExpr in every evaluation the tables use (harness `THREAD_CAP_ENV`, `p515_s44_campaign_harness.py:320`, enforced at :2233; recorded per cell in `launch.json` and the evaluation record); current shell: unset | W159 `62749030` |

---

## (v) Prediction scorecard (supplementary material)

Scope: every recorded prediction scored at or after Addendum 52, by the addendum or record that scored it. The
outcome is stated as the record states it.

### Scored at Addenda 52–54

| # | prediction (who, where recorded) | outcome | outcome source |
|---|---|---|---|
| 1 | 3 × 3 x = 0 replay bitwise through 72 (expert, A51) | **held** (72/72) | Addendum 52; `4689475e` |
| 2 | after certification: a hump then geometric decay, ratio 0.80–0.95 (expert, A51) | **missed**: a damped oscillation (envelope 0.892/cycle) | Addendum 52; `TASKS.md` (W99) |
| 3 | D_x0 = 20–60 k€ (expert, A51) | **missed** by an order of magnitude (4.5–6.3 k€) | Addendum 52 |
| 4 | stage 2 triggered (expert, A51) | **missed** | Addendum 52 |
| 5 | R_settled ≥ 0.80 (expert, A51) | **held** (trivially) | Addendum 52 |
| 6 | SRP1 settled slack per cell ∈ [5, 25] k€ (expert, A53 / v39) | frozen scorer: x0 **held**, unit **indeterminate**, C\* not scoreable (values x0 +14,971, unit +20,806) | Addendum 54; frozen scorer `62bdeafe` (`TASKS.md`) |
| 7 | \|ΔV\| ≤ 20 k€ (expert, A53) | **held** (ΔV −5,836, resolution 8,561) | Addendum 54; `62bdeafe` |
| 8 | three slacks alike within ≈ 5 k€ (expert, A53) | expert: **missed narrowly** (5.8 k€); frozen scorer: not scoreable (C\* unsettled) | Addendum 54; `TASKS.md` lines 123 and 167 |
| 9 | Advisor k\* projections, SRP1 continuation | **missed** late (period 29–30, not 20) | `62bdeafe` (`TASKS.md`) |
| 10 | C\* H_ess-flat: P_a creep in TSO generation; P_b Σ\|Δp_ess\| not decaying; P_c rising residual is the ESS channel; P_d rate within ×2 of −265 €/cycle (expert, A54) | **MIXED**: P_a held (share 1.416); P_b held (0.853); **P_c failed** (rising primal is pf); P_d held (−153.5 €/cycle). A57: "flat direction real, ESS channel the wrong residual to name" | W110 `c2afe100`; Addendum 57 |

### Scored at Addenda 55–58 (benchmark and Phase B campaign)

| # | prediction (who, where recorded) | outcome | outcome source |
|---|---|---|---|
| 11 | coordination proper resolvable and positive, living where λ_t ≠ π_t (expert, A49) | **held** | Addendum 58 |
| 12 | coordination inside the band (Advisor, A49) | **missed** | Addendum 58 |
| 13 | benchmark size | nobody predicted it | Addendum 58 |
| 14 | sweep: passive [1, 4] blocks, price-taker [2, 8] (Planner, W119) | **held** (1 and 8) | W120 `4858b3a5` |
| 15 | NRF arms feasible at every TSO block; claim positive (Planner, W119) | **held**; **held** | W122 `8d42dfb8` |
| 16 | F2 challenger: C\*-like creep (Advisor); dQ_cc flatter than dQ | **failed** (drift ≈ 380× smaller); **failed** over the last 25 | W123 `aad4197b` |
| 17 | F2 overlaps at N_old (Advisor) | **held** (both) | W123, W125 |
| 18 | F2 incumbent first turning point ≤ k₀ + 15 (Advisor) | **held** (176) | W125 `075823cf` |
| 19 | Phase B k\* ∈ [k₀+50, k₀+70] (Advisor design review Q7) | pb_y2030_n9 **missed** by 2 (k₀+72); the other five **held** | W128 / W129 |
| 20 | Phase B s within 10 k€ of 14,971 | pb_y2030_n9 **missed** (+27,488); the rest **held** | W128 / W129 |
| 21 | year-ladder k\* timing | **held** (k₀+62 / +61) | W129 |

### Scored at Addenda 59–61 (v4–v6 rules)

| # | prediction (who, where recorded) | outcome | outcome source |
|---|---|---|---|
| 22 | cell 1 (`b_2a0ba8b2`) re-run bitwise and certifies at 173 (Planner, A59) | **held** | W138 `903657de`; Addendum 60 |
| 23 | group 1 all certify (Planner/Advisor, W132) | **failed** (cell 3 at v4) | W138 (`TASKS.md` line 34) |
| 24 | group 1 s ∈ [8, 30] k€ | held 2/3 (cell 3 +32.3 k); `pb_y2025_n5_v6` later **held** (+20,913) | W138; W152 |
| 25 | group 1 k\* ∈ k₀ + [50, 75]; gap ≤ τ/2 | **held** 2/2; **held** | W138 |
| 26 | B margins, formula ranges (Planner) | **held 2/3** | W138; Addendum 60 |
| 27 | B margins (Advisor ranges) | **missed 3/3** | W138; Addendum 60 |
| 28 | cell 3 certifies under v5 (expert, A60); bitwise v4 through 148 (Planner, G26) | **held**; **held** | W140 `4b3be392`; Addendum 61 |
| 29 | the nine break-even-fit cells certify (expert, A60) | **failed** (`d_c52e1670` uncertified at cap 198 under v5; recorded failed "for a rule reason") | W140; Addendum 61 |
| 30 | the determinacy floor changes no verdict (expert, A61) | held at W142; **failed on one claim** at W152 (`E:C2_calfade vs C3`, 0.71×) | W142 `689340d2`; W152 `184229bb`; Addendum 64 ruling 5 |
| 31 | the eight D cells certify under v6 (Planner, A61) | **failed** 3/8 (`d_36686489`, `d_a12d95a2`, `d_f759dd48`); a certification-status prediction, recorded, not a stop | W143, W146; Addendum 62 |

### Scored at Addenda 62–64 (break-even, ladders, C/G/L, ageing, Step 5)

| # | prediction (who, where recorded) | outcome | outcome source |
|---|---|---|---|
| 32 | the two break-even fits agree on the slope within 5 % (expert, A62) | **held** at the midpoints: 0.59 % on b, 0.31 % on b + c/4 (the ruled slope). Corners 7.1 % / 3.7 % are interval sensitivity. Conservative 4-interval fit (report-only, W157): 0.47 % at the midpoint | W145 `7c1a4ecb`; Addendum 64 ruling 3; T3 |
| 33 | break-even within ±6 k€/MWh of 182.2 k; margin > 60 k€/MWh (Planner, W132) | **held** (182,702; ≥ 63.5 k; conservative ≥ 61.3 k) | handback; T3 |
| 34 | ≥ 2 of the 4 single-node E ≥ 4 MWh cells uncertified by creep → stop (Planner watch) | not triggered (1 of 4) | `TASKS.md` W143/W146 |
| 35 | H m = 1.5 value − I negative, \|margin\| 15–45 k€ | sign **held**; size **missed low** (−14.45 k); determinacy **missed** | Addendum 63 |
| 36 | H m = 2 +35–60 k€; I second MWh +30–55 k€ | **held** (+46.3 k); **held** (+46.5 k) | W147, W148 |
| 37 | J e2→e3 25–45 k€; e3→e4 35–60 k€; e4→e5 indeterminate | **held**; **held**; **missed** (determinate, 1.95×) | `TASKS.md` line 55 |
| 38 | `j_5f3cccb4` dead zone, gap-refused, t_sum −8 to −10 k | **held** (−8,848) | `TASKS.md` line 55 |
| 39 | C and G certify; G at k₀ + [50, 75] | **held** (k₀ + 62–71) | W150 |
| 40 | C / G magnitudes (Advisor design review) | **missed low on all six rows**; signs and determinacy held | W150 |
| 41 | G net ≥ 90 k€ | **missed** on 2 of 4 (88.3, 85.1 k€) | W150 |
| 42 | L small cells certify (7); s ∈ [−5, +10] k; margins move ≤ 10 k | **held**; **held**; **held** | W151 |
| 43 | L dead-zone cells gap-refused (4) | **3 of 4** (`l_0ee93aca` lapse-reset; entered by its signature, ruling 7a) | W151; Addendum 64 ruling 7 |
| 44 | G28: `e_c2_calfade` reproduces the settled unit bitwise through 172 (Planner) | **held** | W152 |
| 45 | floor-binding arms show a smaller value change per unit of cycle life (expert, A58 supplement) | point estimate held on C2_calfade (ε_k 0.0331 < 0.0394); within resolution (0.20×). Scored "**consistent in direction, not resolved at this precision**" | W152; Addendum 64 ruling 6 |
| 46 | captured spread 55–60 €/MWh (expert, A28) | **held** (59.0) | W153 `48e76c9f` |

### Recorded before W156, scored at W156 / W158b

| # | prediction (who, where recorded) | outcome | outcome source |
|---|---|---|---|
| 47 | **A (m = 1.75):** value − I positive, point +15 k€, range [+8, +22] k€; P(x0 certifies) 0.9, P(unit certifies) 0.55–0.6; threshold in [12.4, 13.2] k€; P(determinate) 0.35–0.45; k₀ ∈ [100, 125]; k\* ∈ [130, 185]; a negative value − I is a STOP (Planner W155, adopting the Advisor's ranges) | **held**: +14,249.16 inside [+8, +22] k€; both cells certified; determinate 1.32×. Threshold 10,775 fell below the predicted 12.4–13.2 k. Unit k\* 121 is below [130, 185]; k₀ 108 / 101 inside. The fallback sentence was not needed | `112bb529`; frozen spec `44a2dce8` (`0ee19c33`) |
| 48 | **B (soh_min 0.50):** Δvalue = value(0.50) − value(0.70) positive, point +12 k€, range [+4, +22] k€; more likely within resolution (threshold 13,054); floor never binds (SoH_end 2035 in [0.64, 0.70], floor duals < 1e-6); EFC/day 2035 0.88 [0.80, 1.00], 2025 [1.10, 1.30], 2030 [1.00, 1.20]; trajectory diverges from 3f084f2f at cycle 1, by more than τ/10 by cycle 5 (Planner W155, Advisor's ranges) | **held**: +4,916.67 inside [+4, +22] k€ (at its lower edge, against a point of +12 k), within resolution (0.38×, threshold 13,054.22 as predicted). Floor never binds (duals ≤ 5.1 × 10⁻¹⁰; 2035 SoH_end 0.677). EFC/day inside every range (1.192 / 1.100 / 0.909). Divergence at cycle 1; > τ/10 at cycle 4 | `52d416cb`, `5fb81cc4` |
| 49 | g070_neutrality reproduces 3f084f2f bitwise through 172 (gate G28 for the A64 cells) | **held** (bitwise 1..172; no field differs) | `0e1d7c0e` |

**Not included:** walls and timing estimates. They are recorded as predictions in several specs (for example
"(b)-on cycle walls … prediction 10.1–12.5") but are not results the manuscript states.

---

## (vi) Response-to-reviewers map

Reviewer items are those of `EXPERT_REVIEW_2_ACTION_PLAN.md` §2. The R1.2 sub-items come from `EXPERT_REVIEW.md` §2.1,
to which §2 points for R1.2(v).

### Formulation and method

| reviewer item | answered by | note |
|---|---|---|
| **R2.1, R2.9** — not a true bilevel; scenario-averaged investments | method text (outside this package); T1/T2 instance keys show one common plan per evaluation (`candidate_canonical`, one investment vector for all scenarios) | Table 6 of the old manuscript must be regenerated from T1/T2, not reused |
| **R2.2** — no complete outer-loop algorithm | outside this package: `STEP4_DFO_METHOD.md` (the planning-method definition) | — |
| **R2.3, R2.4, R2.7, R3.2** — cuts not valid; sensitivities undefined; 0.50 % gap not a certificate | (ii) certification paragraph: each number is a certified recourse evaluation of a named candidate, with no optimality gap claimed. T1: every planning statement is a pairwise comparison with a stated bar and a determinacy verdict. T3: the break-even fit with bands | the "0.50 % gap" sentence is removed; no Benders cut is used |
| **R2.8** — three references | editorial; not in this package | — |

### ADMM convergence

| reviewer item | answered by | note |
|---|---|---|
| **R2.5, R2.6** — ADMM convergence and adaptive penalty unjustified | (ii): Boyd residual test, settling criterion, statistics (42 cells: 32 certified, causes of the 10). T2: k\*, range/τ, flags per cell. (iii): degenerate duals, TSO recoveries. (iv): bitwise replays | the penalty is frozen after the first residual pass (holds); the empirical evidence is the statistics, not a theorem |

### Storage physics

| reviewer item | answered by | note |
|---|---|---|
| **R2.10, R3.4** — apparent-energy throughput | equations (editorial). T8 uses active cell-side energy (AE, EFC columns) | — |
| **R2.11, R2.12, R3.3** — sign/direction, SOC dynamics | equations (editorial); not in this package | — |

### Salvage and ageing

| reviewer item | answered by | note |
|---|---|---|
| **R3.1** — salvage value; SOC limits | T1: net beside gross on every claim; T4: year ladder (gross +42,458.65 determinate; net −2,286.25 within resolution), with the A58 sentence; W153 salvage row: 0 sign changes in 60 claims, 1 verdict change. soh_min: T8 (0.70 baseline) and T10 (0.50: Δvalue +4,916.67 within resolution; value − I −59,500.72 determinate) | salvage is a post-evaluated credit; gross stays primary (Addendum 58 Ruling 3) |
| **R3.5** — calendar ageing missing | T8: C2 (calendar retention 1.0/yr) vs C2_calfade (0.985/yr, the baseline): −31,607.24 vs −64,417.39, both determinate | arm definitions: ext spec v3 `84775dc4` `cells.*.model_variant` |

### R3.6 — degradation-neutral benchmark, row by row

| R3.6 row | arm and cell | value − I | multiple and verdict | floor binds | source |
|---|---|---|---|---|---|
| no degradation | `e_no_ageing` | −4,140.54 | 0.32×, **within resolution** | — | T8 |
| cycling only | C2, `e_c2` (calendar 1.0/yr, EOL retention 0.8) | −31,607.24 | 2.50×, determinate | never | T8 |
| cycling + calendar (the baseline) | C2_calfade, `e_c2_calfade` (calendar 0.985/yr) | −64,417.39 | 4.93×, determinate | 2035 | T8 |
| calibration: harsher | C3_unit (EOL 0.5) | −73,746.66 | 5.74×, determinate | 2035 | T8 |
| calibration: harsher | C4 (EOL 0.7) | −45,196.70 | 3.58×, determinate | never | T8 |
| SoH evaluation point | C3_midblock | −65,891.88 | 5.12×, determinate | 2035 | T8 |
| no degradation vs C3 | — | +69,606.12 | 5.41×, determinate | — | T1 |
| soh_min floor (0.70 → 0.50) | `e_soh050` | −59,500.72 (Δ vs 0.70: +4,916.67) | 4.71×, determinate (Δ: 0.38×, within resolution) | never | T10 |

- The verbatim sentence for these rows is sentence 1 in (i).
- The comparisons are fixed-plan re-evaluations of the unit (node 7, 0.25 MVA / 1 MWh). Re-optimized plans per arm
  are not in the package.
- **Re-optimised plan per arm (Addendum 65 sentence):** under every aged arm the unit — the smallest lattice point —
  loses determinately, and Phase A measured the value as affine in size with slope below cost, so x = 0 is the optimal
  plan under each aged arm provided that affinity holds under the arm (an assumption); under no ageing the unit is at
  break-even within resolution and the plan is indeterminate at this precision.

### Economics and the first reviewer

| reviewer item | answered by | note |
|---|---|---|
| **R3.7, R3.8** — NPV aggregation; "annual" wording | T7, with its formula (W153 `w153_discount_row.json` `formula`: one discount factor per representative year applied to all 5 years of the block; I paid in 2025, not discounted) | wording is editorial |
| **R1.2(v)** — attribute degradation to TSO vs DSO | push back (EXPERT_REVIEW_2 §2): joint schedule, no unique attribution | no table |
| **R1.4(ii)** — discount-rate sensitivity | T7: value − I negative and determinate at 0 / 2 / 5 / 8 % | fixed plan; the production rate stays 2 % (Addendum 28) |
| **R1.5** — Figure 2 | editorial | — |
| **R1.2** — uncertainty, system size, date/basis of benefits (EXPERT_REVIEW §2.1) | T6: benchmark with its definition, instance and bands. The 3 × 3 multi-scenario statements: Addenda 52 and 54 (x = 0 optimal at both instances; R ∈ [0.909, 0.934]) | headline percentages regenerated, see the 18.25 % row below |
| **R1.4** — uncertainty (EXPERT_REVIEW §2.1) | the 3 × 3 instance (Addendum 52); T7 | — |

### The 18.25 % figure, replaced (Addenda 49, 57, 58)

**What it was.** 18.25 % compared the combined coordination + storage case with uncoordinated operation (old Table 8,
per `EXPERT_REVIEW.md` §4). It was measured on the transfer-payment recourse that Addendum 11 retired.

**What happened to it:**
- **withdrawn** in Addendum 49;
- **"replaced, not restored: different recourse, different definition"** (Addendum 58).

**What replaces it:**
- **Definition:** benefit = min(Q_passive-NRF, Q_price-taker-NRF) − Q181. Each arm is the best of three starts.
  - Q is `gross_operational_cost`, settlement excluded, on the same evaluation function.
  - Q181 is the settled coordinated x = 0.
  - The uncoordinated arms carry a **no-reverse-flow** interface rule (Addendum 57 Decision 1(b)).
- **What it measures:** dynamic coordination against a static interface limit.
- **Result:** **+90,896,608.40 € = 13.9 %** of the coordinated Q181, determinate at 1,263.75× the larger band
  (71,926.11). Source: T6; `report_v3.json` `8d42dfb8`; spec v5 `bca69f97`.
- **Caveat stated with it:** 4 reverse-flow interface-hours in the coordinated solution.
- **Mechanism sentence (verbatim):** sentence 2 in (i).

**The incremental storage benefit is not this figure.** At SRP1 the unit does not pay: value − I = −64,417.39. So the
paper reports no positive storage share of the benefit in the baseline.

### Items the record raised that the reviewers did not

| item | answered by | note |
|---|---|---|
| the constants behind the numbers | T8 and T7 instance: baseline C2 + calendar 0.985/yr + soh_min 0.70 (W153 `settled_ageing_baseline`: cycle life 10,000, DoD 0.8, EOL retention 0.8) | the 0.50-era statements are restated at 0.70 (Addendum 58 supplement; Addendum 64 ruling 6) |

---

## Not confirmed

Remaining items, after Addendum 65.

### Still without a source

1. **Remaining drift of the 8 monotone certificates at L = 60.** Searched W153, the v6 summary, `TASKS.md`. It stays
   unsourced by ruling; seven of the eight have range/τ ≤ 0.171.

Threading, the linear-solver banners and the software versions are now sourced (W159 `62749030`; see (iv)). The
R3.6 re-optimised-plan question is answered by the Addendum 65 sentence in (vi).

### Also from the closing reads

- **The F2 pair's run records read `status: certified` at their caps (281 / 261).** That is the run's residual
  certificate. The settling rule (W118 r2) classifies the pair as uncertified, refused by the gap clause, and the
  tables use the settling-rule status. Not reconciled field by field (W159).
- **T6's Markdown lacks** the passive arm's Q by start, the decomposition, the NRF definition, the consistency
  violations, the failing sweep blocks and the curtailment table. These are in `report_v3.json` (`8d42dfb8`); the
  export adds them.

### Rulings that were ambiguous to apply

1. **Ruling 4 beyond J 4 MWh.** The uncertified form is shown beside every claim that names a flagged certificate:
   - `B:n7_4h_e3`: 2.94×, determinate;
   - `J:e3_to_e4`: 1.55×, unchanged (the other cell's bar already binds);
   - `J:e4_to_e5`: 1.95×, unchanged;
   - `J:e4_value_minus_I`: 11.12×.

   Only the break-even and J 4 MWh are named as manuscript figures.
2. **The conservative break-even fit drops `d_4a82a64a` from its certified-only fit** (n = 6; e\* 183,083.50; margin
   70,794.17). Only the banded range (171,309–192,557) changes the manuscript figure. Both are in T3.
3. **Phase B under the v6 rule (T5) was computed here** by `DET.resolve_v6` from the W118 bands; W118 recorded these
   cells under the superseded rule. All verdicts are unchanged. Net is not shown for these cells: they are outside the
   W154b validation set.
4. **The ≥ 0.95 τ flag is applied to every certified cell in T2 and T5**, not only W153's 42. This adds `b_2a0ba8b2`,
   `b_4649234b`, the unit reference and three W118 Phase B cells.
