# P5.15 Section 4 revision — zero-solve extractions for the placeholders and checks (2026-11-16)

**Authority:** the author reopened the numbers freeze (Addendum 91 l. 4977, Addendum 92 l. 5098) for these items only.
The request is the Expert's handback of 2026-11-16 against Overleaf **`e315b07`** (main.tex sha256 `76ff43f5…`).
**No solves, no pickle loads, no model builds:** the script reads committed JSON/CSV only, and its pickle block counted
0/0 at exit. Every new number below is for the **W190** audit list. No manuscript edits were made.

**Objective convention (all tables):** Q = `gross_operational_cost`, contracted interface settlement excluded,
discounted and block-weighted. Terminal salvage is 0 at these cells. Operating value = Q(0) − Q(x).

## Decisions needed (Expert)

1. **a1 / a2 are not resolved at one decimal.**
   - **What the records show.** At the default penalties the split is 62.7 % generation and 37.3 % flexibility. Under
     the W185 runs, with all initial penalties halved and doubled, the total moves by only 1.6 k€, but generation
     shifts against flexibility by up to 10.4 k€. The generation share is then 66.7 % (×0.5) and 65.9 % (×2.0).
   - **Why it matters.** This is the same trade between the two cost terms that set apart the four configurations at
     C\* (Addendum 22 §5). The 77.2 / 22.8 split in Section 4.1 is safe because it sits on 90.9 M€; this one sits on
     253.5 k€.
   - **Options:**
     - (i) print 62.7 / 37.3 alone: not recommended, since the precision is not supported;
     - (ii) print the range, "63–67 % … 33–37 %";
     - (iii) **"about two-thirds … one-third" (recommended)**, with the range in Appendix B.
   - **Cost:** wording only.
2. **Curtailment convention.**
   - **The conflict.** The handback asks for Appendix B.3's convention, which is **net** signed (W184). Addendum 56
     rules that the manuscript reports the **positive part**.
   - **For the no-storage plan both round to the same figures:** 0.7 GWh in the DNs and 0.0 in the TN.
   - **For the reference unit only the net is in the records.** Its positive part would need an hourly extraction from
     the pickle, which the handback puts out of scope.
   - **Recommendation:** print the net figures as requested, and state the convention once in B.3.
   - **Knock-on edits:**
     - since b1 = b2 at one decimal, reword the l. 1423 sentence (wording proposed under Found (b));
     - B.3's "Curtailment was not reported for coordinated operation" is no longer true once b1 enters, so it needs
       rewording.
3. **Replacing "about 10⁻⁴ of Q" at l. 1715.**
   - **The two figures measure different things:**
     - the 10⁻⁴ is the spread across four **penalty policies** at C\*, under the earlier cost data;
     - the W185 figure is **initial-penalty scaling** within the one frozen policy.
   - **Options:**
     - (i) **replace** the sentence with the W185 figure (recommended), worded as a bound inside the certification
       bands, as under Found (c);
     - (ii) keep both, the 10⁻⁴ explicitly as "alternative penalty policies in preliminary evaluations".

## Blocked on the author

- Push from PyCharm: the branch is ahead of origin.

## Changed

- `p515_s55_section4_extractions.py` (sha256 `fc9589cd…`): records only. The pickle guard is armed for the whole
  run, the script imports no production module, and it refuses to overwrite its output.
- `data/SRP1/Results/P515S55/section4_extractions/section4_extractions_v1.json` (sha256 `884e1db7…`):
  - all values below;
  - per-evaluation checks;
  - the sha256 and git-tracked status of each of the 31 inputs, all tracked;
  - `manifest_sha256.json` beside it.
- TASKS.md: one line.

## Found

**Per-evaluation checks (12 evaluations).**
- All 12 are certified.
- The certified cost equals the component file's gross cost bitwise.
- The component file is at the certification cycle (cycles_run = k\*).
- Q = sum of its parts to within 3.3e-07 €; TSO + DSO block sums = Q to within 3.3e-07 €.
- At both prices, the pair's Q equals the frozen tables' Q (`frozen_step6_tables_v1_590088fe.json`).

### (a) Composition of the reference unit's operating value

Weighted (discounted) totals from `component_levels_terminal.json`, Q(0) − Q(x), in k€.

| term | 1× (ref:7aa017f0 / ref:bd504ecf) | 2× (h_50dea31c / h_74eda68d) |
|---|---|---|
| TN generation cost | 158.9 (62.7 %) | 95.7 (26.3 %) |
| DN internal flexibility cost | 94.6 (37.3 %) | 268.5 (73.7 %) |
| shared-ESS day-balance closure slack (detector term in Q) | 0.0 (19.90 €) | 0.0 (19.90 €) |
| load curtailment, RES-curtailment penalty, ESS usage | 0 | 0 |
| voltage and flexibility day-balance slacks | < 1e-6 € | < 1e-6 € |
| **total = reported operating value** | **253.5** (253,539.62 €) | **364.2** (364,242.47 €) |
| residual Q-difference − Σ parts | 4.2e-08 € | 2.6e-08 € |

**Shares.**
- **a1 = 62.7 %, a2 = 37.3 %** at the default penalties.
- The "other" share is 0.008 %, the 19.90 € closure slack. At the unit the slack sits at the solver's bound
  (W185 decision 1).
- **Same convention as the 77.2 / 22.8 split.** That split is by agent: TSO block gross against DSO block gross (W124,
  STEP6_SUPPORT §Fig 5). By agent the shares here are 62.69 / 37.31 %, the same at one decimal.

**Stability across the W185 initial-penalty scalings (×0.5, ×1, ×2):**

| | dQ (k€) | generation (k€) | flexibility (k€) | generation share |
|---|---|---|---|---|
| 1×, ρ₀ × 0.5 | 253.7 | 169.3 | 84.3 | 66.7 % |
| 1×, ρ₀ × 1 | 253.5 | 158.9 | 94.6 | 62.7 % |
| 1×, ρ₀ × 2 | 252.1 | 166.1 | 86.0 | 65.9 % |
| 2×, ρ₀ × 0.5 | 364.9 | 106.2 | 258.7 | 29.1 % |
| 2×, ρ₀ × 1 | 364.2 | 95.7 | 268.5 | 26.3 % |
| 2×, ρ₀ × 2 | 363.5 | 96.0 | 267.4 | 26.4 % |

- **Direction is robust:** generation is the larger part at 1× and flexibility at 2×, under every setting.
- **The one-decimal split is not robust** (decision 1). The terms carry no resolution of their own (W184 Not
  confirmed); this table is the only empirical measure of their stability.

**The 2× pair** is the W142 resettle-v6 cells behind claim `H:m2:value_minus_I` (k\* 146 / 150), the only certified
default-penalty pair at 2×. W185 records that these cells predate production `6f43479e` and carry `gated: true`.

### (b) RES curtailment, coordinated operation (net; Appendix B.3 / W184 convention)

**Convention.**
- **What is summed:** per agent, Σ over blocks of the unweighted `res_curtailment_definitional_at_weight_1` ×
  Years[y] × Days[d] (SRP1.json), / 1000.
- **The quantity:** **active** energy, (p_avail − p) × baseMVA per 1 h period, scenario-probability weighted per
  representative day, net signed.
- **Weighting:** day-weighted and block-weighted (years per block), **undiscounted**.

| GWh | TN | DN | total |
|---|---|---|---|
| **b1** no storage (d110bd1a, k\* 181) | −0.0 (−0.0210) | **0.7** (0.6817) | 0.7 (0.6606) |
| **b2** reference unit (3f084f2f, k\* 172) | −0.0 (−0.0210) | **0.7** (0.6816) | 0.7 (0.6606) |
| difference, b1 − b2 | 4.5e-07 MWh | 0.079 MWh | 0.079 MWh |
| 2× for reference: no storage / reference unit | −0.0210 / −0.0210 | 0.4878 / 0.4872 | |

**Checks.**
- b1 reproduces W184's C_OPEN row (−0.0210 / 0.6817) and W109's day-weighted totals (−21.007 / 681.661 MWh).
- **Positive part (Addendum 56 convention), x = 0 only** (W109): TN 0.000 and DN 0.728 GWh. The TN's net negative is
  entirely the tolerance edge. The reference unit has no record of its positive part.

**Distinguishable from the bound tolerance? No.**
- RES output may exceed availability by 1e-5 pu, i.e. 1e-3 MW per generator-hour (Addendum 56).
- Over the horizon that band reaches −0.61 GWh in the TN and −1.10 GWh in the DNs (x = 0, W109 per-block floors,
  same weighting).
- The 0.079 MWh difference is about 1/14,000 of the DN band.

**Proposed wording for l. 1423 (Expert's call):** "RES curtailment under coordinated operation is 0.7 GWh over the
horizon, almost entirely in the DNs, with or without the reference unit; the difference lies within the solver's bound
tolerance."

### (c) Cost levels under the penalty-sensitivity runs (W185)

| | ρ₀ × 0.5 | ρ₀ × 1 (default) | ρ₀ × 2 | spread |
|---|---|---|---|---|
| Q(0), 1× (k€) | 653,874.5 | 653,873.7 | 653,873.0 | **1.5 k€ = 2.3 × 10⁻⁶ of Q** |
| Q(unit), 1× (k€) | 653,620.8 | 653,620.2 | 653,620.9 | 0.7 k€ = 1.1 × 10⁻⁶ |
| Q(0) − Q(unit), 1× (k€) | 253.7 | 253.5 | 252.1 | 1.6 k€ (= l. 1716 ✓) |
| Q(0), 2× (k€) | 811,016.3 | 811,014.4 | 811,014.4 | 1.9 k€ = 2.4 × 10⁻⁶ |
| Q(unit), 2× (k€) | 810,651.4 | 810,650.2 | 810,650.9 | 1.2 k€ = 1.5 × 10⁻⁶ |
| Q(0) − Q(unit), 2× (k€) | 364.9 | 364.2 | 363.5 | 1.5 k€ (= l. 2292 ✓) |

**Resolution.**
- All the spreads are smaller than the cells' own certification bands: 3.8–4.4 k€ at 1× and 2.2–4.4 k€ at 2×
  (W185 report; frozen default cells 4.2 / 4.4 k€ at 1× and 2.2 / 4.1 k€ at 2×).
- They therefore **bound** the sensitivity; they do not resolve it (bar refinement, CLAUDE.md).

**Suggested sentence:** "Under the adopted configuration, halving or doubling all initial penalty parameters changed Q
by at most 1.9 k€ (2 × 10⁻⁶ of Q), within the evaluations' certification bands."

### (d)–(f) Checks on numbers already in the text

- **(d) Confirmed.**
  - **Unrounded operating values** (frozen ageing rows, Q(0) − Q(x)): no ageing 313,816.47 €; cycling only
    286,349.77 €; baseline 253,539.62 €.
  - **Reductions:** cycling only **8.75 % → "9 %"**; cycling and calendar **19.21 % → "19 %"**.
  - **Reviewer's derivation:** the derived 313.9 and 286.4 are 313.8 and 286.3 unrounded. They come from rounding the
    printed margins; the percentages are unchanged.
- **(e) Confirmed.** The 0.50-floor evaluation (e_soh050, certified, k\* 184) cycles more and fades faster in every
  block:

  | block end | SoH (= available energy, MWh), 0.50 floor | SoH, baseline | EFC/day, 0.50 floor | EFC/day, baseline |
  |---|---|---|---|---|
  | 2029 | 0.873 | 0.880 | 1.19 | 1.02 |
  | 2034 | 0.765 | 0.780 | 1.10 | 0.90 |
  | 2039 | 0.677 | 0.700 | 0.91 | 0.64 |

  - **Present-value-weighted AE** is 0.778 against 0.793 (W155 `3c71d71d`; frozen ageing row), with EFC/day 1.08
    against 0.86.
  - The 4.9 k€ value change is within resolution (4,916.67 € against a 13,054.22 € threshold).
  - Source: W188d `soh_per_block_calibrations.csv` (`9243aafe`).
- **(f) Confirmed.**
  - **Thresholds.** All 13 printed thresholds (6 value-ladder rows, 7 ageing rows) equal the predecessor's separate
    column at Overleaf `3ad0764` and round the frozen claim thresholds. All 13 values round the frozen `d_gross`.
  - **Certification.** Every pair is certified on both sides, so no row needs †.
  - **One presentation note.** The predecessor marked "No ageing" with ^w, "within numerical resolution"; the new table
    drops the mark. The ±12.9 and the text's "unresolved" carry the same information.

## Not confirmed

- **The reference unit's positive-part curtailment.** No record holds it. It needs a guarded hourly read of
  `certified_models.pkl` (`d0a3d512…`), which the handback puts out of scope.
- **Hourly or per-generator attribution of the 62.7 / 37.3 split.** It was not attempted (out of scope).
- **The 2× pair on the current production head.** W185 Not confirmed: rows 34–35 predate `6f43479e` and were not
  re-run.
- **Whether the l. 1715 "about 10⁻⁴" should survive as a statement about penalty policies.** Its origin is
  Addendum 21–23 (C\*, four configurations, max 72,191 € on 651.04 M€). That is a wording decision (decision 3).
