# Step 6 tables — manuscript-ready export (W160)

Frozen tables: `../frozen_step6_tables_v1_590088fe.json` (sha256 `590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4`). Every file here is generated from that JSON by `p515_s53_w160_step6_freeze_export.py`; the JSON keeps full precision, the CSV and LaTeX files are rounded as below.

**Objective convention (every table).** Q = `gross_operational_cost`, settlement excluded — the primary. Net = Q − terminal salvage credit, shown beside. Q_cc = Q + t_sum, report-only. value = Q(0) − Q(x); F = Q + I; τ = 4,539.07 €. Each LaTeX caption states it.

## Files

- `T1.csv` … `T11.csv`: every column; `T1.tex` … `T11.tex`: booktabs, plain `tabular`, `\caption`, `\label{tab:Tn}`, `\scriptsize`. Requires `\usepackage{booktabs}`; `\texteuro{}` is in the LaTeX kernel (replace by `\euro{}` if `eurosym` is preferred). The LaTeX files were not compiled here (no TeX installation on this machine); they were checked for ASCII-only text, balanced braces and the cell count of every row.
- `paragraphs.md`: the draft paragraphs, the reviewer map, the sentences and the scorecard, verbatim from `P5_15_STEP6_PACKAGE.md` at `e3437284`.
- T11 is not part of T1–T10: it carries the 3 × 3 instance figures with their sources (below).

## Rounding

| quantity | rounding | rule from |
|---|---|---|
| EUR amounts (Q, value, I, differences, bands, thresholds, bars, gaps, slacks, t_sum, salvage) | k EUR, 1 dp | Addendum 65 |
| EUR/MWh (break-even, margin, slope, energy cost) | nearest 10 EUR/MWh | Addendum 65 |
| multiples (x): abs(margin) / threshold or bar | 2 dp | Addendum 65 |
| EUR/MVA (power cost p_cost; fit coefficient c) | nearest 10 EUR/MVA | W160 choice (unit price, as EUR/MWh) |
| percentages (benefit relative, node shares) | 1 dp | W160 choice |
| percentages of slope agreement (T3) | 2 dp | W160 choice (scored against a 5 % criterion; the package quotes 2 dp) |
| EFC/day (incl. the PV-weighted EFC of T8) | 2 dp | W160 choice |
| SoH, available-energy fraction AE | 3 dp | W160 choice |
| range / tau | 2 dp | W160 choice |
| elasticity eps_AE | 2 dp | W160 choice |
| dimensionless ratios (R, pf_primal ratio) | 3 dp (R predicted 0.9331 at 4 dp, as recorded) | W160 choice |
| cycles (k0, k*, end, replay, turning points), counts, hours, blocks | integer | W160 choice |
| energy (reverse-flow MWh) | 1 dp MWh | W160 choice |
| curtailment energy (day-weighted) | GWh, 1 dp | W160 choice |
| p.u. quantities (NRF excess) | 3 significant figures | W160 choice |
| discount rate | %, 0 dp | W160 choice |
| ageing calibration k | integer | W160 choice |

Values are rounded to nearest from the full-precision JSON. A value that rounds to zero is written without a sign. An empty CSV cell (— in LaTeX) means not applicable or not recorded.

## Columns omitted from the LaTeX tables (present in the CSV)

- T1: statement, ref status, other status, net label, notes
- T2: item, candidate key
- T3: none
- T4: net label, eval key
- T5: note
- T6: none
- T7: none
- T8: none
- T9: entry
- T10: statement
- T11: source

## Column dictionary

**T1**

- `claim`: claim id; the prefix before the first colon is the claim family (B, C, CHECK, E, G, H, I, J, L), as in the W157 tables
- `ref cell / other cell`: the two cells of the difference (T2 rows)
- `d gross`: difference on Q = gross_operational_cost, settlement excluded; value form = Q(0) − Q(x) − I where the claim is "value − I"
- `rule`: max(3 × larger bar, 2τ) between certified cells (Addendum 61); uncertified form 3·max(|gap|, |slack|) with an uncertified cell (Addendum 58)
- `threshold / bar`: the determinacy threshold or the uncertified bar of the rule
- `× gross / × net / × Q_cc`: |difference| / threshold-or-bar
- `verdict`: determinate if × > 1 (the uncertified form tests gross and Q_cc), else within resolution
- `d net`: difference on Q_net = Q − terminal salvage credit; net label: "recorded" (G rows) or "validated by form + salvage identity (W154b)"
- `d Q_cc`: difference on Q + t_sum, report-only
- `uncertified form beside`: Addendum 64 ruling 4: the flagged certificates j_a11d7966 / d_4a82a64a treated as uncertified; bar, × and verdict

**T2**

- `table`: T2 = the 49 W157 cells; T4 / T5 / T10 = certificates appended by W160 for the ≥ 0.95 τ scope
- `k0`: first residual pass N of the run (k0_run)
- `k* / end`: certification cycle, or the end / cap
- `range/τ`: range of Q over the certifying window / τ (certified cells)
- `≥ 0.95 τ (range)`: range/τ ≥ 0.95 (W157 flag)
- `at_or_above_0.95_tau_counted`: true on exactly ten certificates (Addendum 65 ruling 2): ≥ 0.95 τ, not superseded, bitwise twins counted once
- `≥ 0.95 τ note`: counted / superseded (excluded) / bitwise twin of the unit / below 0.95 tau / not a certificate
- `flag beside`: i_5a6a88b4 (monotone, 0.939), flagged beside the ten
- `NCTP cycles`: turning points on non-clean cycles (Addendum 64 ruling 4)
- `cause (uncertified)`: the cause the settling rule records; d_36686489 per Addendum 65
- `gap / slack / band`: uncertified view |t_sum| and |s|; certified band width
- `non-clean cycles after N`: count of non-clean cycles after the first residual pass
- `replay bitwise through`: cycle through which the run replayed its original bitwise
- `certifying spec`: stage spec under which the cell certified (v4 / v5 / v6 / ext v3 / W118 r2 / A64 v1)
- `eval key / candidate key`: instance identifiers (prefixes; full keys in the JSON)

**T3**

- `e*`: break-even energy cost = b + c/4 − p_cost/4 (€/MWh)
- `certified-only / banded midpoint`: OLS on certified points only / on all points with uncertified ones at their interval midpoints
- `e* min / max, margin min / max`: range over the box of interval corners; margin = energy cost − e*
- `agreement`: relative difference of the slope between the two fits (Addendum 64 ruling 3)

**T4**

- `M`: I + Q − Q181 (W118 form), gross and net
- `threshold / × / verdict`: v6 rule between the two certified cells

**T5**

- `M gross`: I + Q − Q181 of each Phase B cell
- `v6 threshold / × / verdict`: DET.resolve_v6 against x = 0
- `recorded (W118 rule)`: the verdict under the superseded rule

**T6**

- `Q gross`: gross operational cost of the arrangement, settlement excluded
- `band`: coordinated: reproducibility band 0.011 % of Q181; arm: multimodality band over three starts
- `Q − Q181`: benefit of coordination over the arm; passive arm DERIVED by W160 from recorded values
- `× larger band`: benefit / max(arm band, coordinated band)
- `NRF violations / max excess`: consistency re-evaluation, hard DN no-reverse-flow limit (report_v3 consistency_nrf)
- `consistency pass effect`: Q change of the sequential consistency pass
- `TN / DN curtailment`: RES curtailment, phase A, day-weighted (report_v3 curtailment_table)
- `sweep`: unconstrained arm (no interface rule), cold start: blocks and hours the TN cannot accept the DN exchange, failing blocks

**T7**

- `V`: value Q(0) − Q(unit) at the rate
- `threshold`: Addendum 61 conservative threshold

**T8**

- `value / value − I`: fixed-plan value of the unit under the arm
- `threshold / ×`: from the matching T1 E claim
- `floor binds (0.70)`: first year the 0.70 SoH floor binds, or never
- `AE / EFC/day`: PV-weighted available-energy fraction and EFC/day (ext spec v3 definitions)
- `ε_AE`: elasticity of value to available energy; the 0.50 column is superseded (label in the header)

**T9**

- `t_sum at cap`: priced interface-consensus gap at the cap
- `share`: node split of t_sum
- `pf_primal last`: power-flow primal residual ratio, last cycle
- `lapse resets at`: cycles of Boyd-lapse resets

**T10**

- `ref / other`: cells of the claim, with k0 / k* and range/τ
- `d gross / threshold / × / verdict`: as in T1

**T11**

- `value`: as recorded or derived (label)
- `source`: file and commit, addendum

## Main text and supplementary material — the expert's suggestion (**author decides**)

Addendum 65. This split is the expert's suggestion; the author decides.

| main text | tables |
|---|---|
| headline x = 0 at SRP1 | T1 CHECK:headline_V_minus_I_settled; T7 (2 % row) |
| break-even fit | T3 |
| flexibility ladder 1.5 / 1.75 / 2 | T1 H:m1.5, H:m2; T10 H:m1.75 |
| R3.6 ageing rows | T8; T10 E:soh050 rows |
| year ladder, gross and net | T4 |
| discount | T7 |
| benchmark with both arms (and the mechanism sentence) | T6; paragraphs.md sentence 2 |
| the 3 × 3 instance (x = 0 optimal, R ∈ [0.909, 0.934]) | T11 |

| supplementary | tables |
|---|---|
| Phase B | T5 |
| C / G | T1 C and G rows |
| L (F2) | T1 L rows |
| certification statistics | T2; paragraphs.md (ii) sources |
| dead-zone table | T9 |
| scorecard | paragraphs.md (v) |

Not named in the suggestion: the T1 B rows, the I rows (second MWh at m = 2), the J rows (energy ladder at m = 2) and the E "vs C3" rows — the author places them.

## The 3 × 3 instance (T11)

The figures "x = 0 optimal, R ∈ [0.909, 0.934]" are not in T1–T10. They come from Addendum 52 (brief `4a80c3e2`; report `P5_15_ADDENDUM51_CONTINUATION_REPORT.md` `58ff8d88`) and Addendum 54 (brief `271e9325`; report `P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md` `ebe34021`). T11 gives them from the committed records: `data/SRP1/Results/P515S53/w90_3x3/campaign_s53_w91_3x3_pair/campaign_results.json` (V, I, value − I, resolution) and `data/SRP1/Results/P515S53/w99_stage1_posthoc/w99_stage1_posthoc_analysis.json` (the post-hoc settled descent D). R is derived (labelled) by [(V - D) / V_SRP1_settled, V / V_SRP1_settled]; V = 3 x 3 value at certification, D = the post-hoc settled descent of the 3 x 3 x = 0 cell (W99), V_SRP1_settled = 253,539.62 EUR.

