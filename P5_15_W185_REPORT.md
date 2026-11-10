# P5.15 W185 — E2 stage 1: sensitivity of the value of storage to the initial penalty (Addendum 85 W185 line; Addendum 86)

**Authority:** Addendum 85 (W185 line), with Addendum 86 rulings 2 and 3 (i)–(ii). Spec v1
`frozen_step7_campaign_spec_v1_96345241.json`; E2 stage spec `7631a6d7…`.

**Cells:** {x = 0, node-7 reference unit 0.25 MVA / 1 MWh 2025} × {1×, 2×} × {ρ₀ × 0.5, × 2.0}. Each was run as one
certified evaluation (Step-7 resettle protocol, rule v6, cap 300, persist on), serially, attended. Each cell's outputs
were committed before the next launched.

**Code:**
- production `6f43479e`;
- harness `d9a944f2…`;
- hooks `9318ade3…`;
- launcher `3b990d19…`.

No `.py` file changed during W185.

**Objective convention:** Q = `gross_operational_cost`, contracted interface settlement excluded.

**Margin:** Q(0) − Q(unit) − I(unit), with I(unit) = 317,957.01 € from the pinned W2 table.

| part | report | commits |
|---|---|---|
| pair 1 (1×, 0.5) | `P5_15_W185_PAIR1.md` | `7311e0df`, `0134c776`, `c54e49b1` |
| pair 2 (1×, 2.0) | `P5_15_W185_PAIR2.md` | `364ed6f1`, `c5419ccd`, `545c5578` |
| pair 3 (2×, 0.5) | `P5_15_W185_PAIR3.md` | `7a78774f`, `fd5d8279`, `984c79c1` |
| pair 4 (2×, 2.0) | `P5_15_W185_PAIR4.md` | `3955b850`, `f8aed532`, `209a1af7` |
| diagnostics (8 guarded reads) | `P5_15_W185_DIAGNOSTICS.md` | `9587232d`, `333025d7` |

**Stop (as ordered).** W186 (E1) and W187 (the screen) wait for Addendum 87.

## Decisions needed (for Addendum 87; none blocks anything queued)

1. **Feasibility at solver tolerance on the unit cells (expert; wording only).**
   - **SOC.** 222–235 of the 1,152 active SOC-limit rows per cell sit below their limit, none by more than 1e-6 MWh
     (deepest −9.985e-07 MWh). That is IPOPT's relaxed bound, 1e-8 p.u. × 100 MVA × 1 h: the same floor the closure
     slacks sit on (all 192 at [−9.99922e-07, −9.99892e-07] MWh).
   - **Converter circle.** pnet² + qnet² − S² is positive on 40–54 of the 576 active rows per cell, at most 8.37e-09
     p.u.² (+8.37e-05 MVA², `e2_unit_2x_rho2p0`, TSO 2035 Autumn). This is far inside IPOPT's constraint-violation
     tolerance. It reads as 1.34e-03 of S² (apparent power 1.00067 × S^Av) only because a 0.25 MVA unit is small on a
     100 MVA base.
   - **Recommendation.** Wherever the manuscript says the circle or SOC limits hold, add "to the solver's tolerance".
     Not verified: whether IPOPT receives the SOC limits as variable bounds; no `.nl` file or log was read.
2. **The monotone certificate of `e2_unit_2x_rho0p5` (expert; reporting).** This cell certified on the monotone branch
   at k\* 178, after 7 gap refusals at cycles 128–134 and a v6 (B) turning-point floor rejection at cycle 135. It was
   **still descending**: Q(k\*) is the band's lower edge, Q fell 4,022 € over the window, and the last step was
   46.82 €. The rule worked as written, and its band equals that descent. Recommend reporting it with its branch and
   the descent, as the bar refinement in CLAUDE.md requires; no continuation was run.
3. **For the record (no action).**
   - The x = 0 cell at (2×, 0.5) carries the ess label "held after solver failure" on cycles 113–122 with no failure
     (the resettle-hold path; factor 1, read-back unaffected).
   - `e2_x0_1x_rho2p0`'s t_sum sits at 79 % of the gap bound.
   - E^Av in the network models differs from the record's published terminal value by ≤ 1.6e-05 MWh. The guess is the
     previous publication; this was not verified.
   - The launcher's report pools both prices in `margin_range_over_configurations`. The per-price ranges below are
     derived from its `pairs[]` and were re-checked independently.

## Blocked on the author

- Push from PyCharm (the branch is ahead of origin).

## Found — hypotheses against outcomes

| hypothesis (Addendum 86) | outcome |
|---|---|
| H2: paired margins within the baseline pair's threshold of their baseline values at each price; no pair's verdict changes | **held at both prices** — 4/4 pairs (table below). The launcher's `H2_per_pair` is all true, re-checked independently |
| the ρ read-back holds on 8/8 cells (bitwise, replicated mean, Addendum 86 ruling 2) | **held** — 8/8 cells, 4,029 checks (every cycle × 3 channels), 0 mismatches. At cycle 1, v / pf / ess equal the replicated means of S39 × s exactly (e.g. s = 2: 0.023099999999999996 / 0.3960000000000004 / 0.020000000000000007, n = 48/48/51). `rho0_scale_applied_in_child` showed 13/13 entries = S39 × s |

| pair | margin (€) | rule-v6 threshold (€) | ratio | verdict | shift vs baseline (€) | H2 |
|---|---|---|---|---|---|---|
| 1×, ρ₀ × 0.5 | −64,264.41 | 12,978.75 | 4.95 | determinate, negative | +152.98 | held |
| 1×, ρ₀ × 2.0 | −65,837.28 | 12,728.13 | 5.17 | determinate, negative | −1,419.89 | held |
| 2×, ρ₀ × 0.5 | +46,959.42 | 13,206.52 | 3.56 | determinate, positive | +673.96 | held |
| 2×, ρ₀ × 2.0 | +45,506.80 | 11,038.38 | 4.12 | determinate, positive | −778.67 | held |

**Baselines.**
- **1×:** −64,417.39 €, threshold 13,054.22 € (E.21 rows 1–2).
- **2×:** +46,285.46 €, threshold 12,272.19 € (rows 34–35).
- Both are read by eval key from `frozen_step6_tables_v1_590088fe.json`.

**Margin ranges over the ρ₀ configurations:**
- **1×:** [−65,837.28, −64,264.41] €.
- **2×:** [+45,506.80, +46,959.42] €.

Each contains its baseline. Every shift is smaller than the cells' own bands, so these runs cannot attribute any shift to
ρ₀ (stopping slack and path are not excluded).

**Cells.**

| cell | status (branch) | Q (€) | k\* | band (€) | stopping ratio | wall (h) |
|---|---|---|---|---|---|---|
| x0 1× 0.5 | certified (osc.) | 653,874,468.09 | 177 | 4,326.25 | 4.23e-3 | 1.22 |
| unit 1× 0.5 | certified (osc.) | 653,620,775.49 | 177 | 3,929.81 | 3.82e-3 | 1.29 |
| x0 1× 2.0 | certified (osc.) | 653,872,972.52 | 185 | 4,242.71 | 1.30e-3 | 1.23 |
| unit 1× 2.0 | certified (osc.) | 653,620,852.78 | 194 | 3,807.15 | 1.62e-3 | 1.58 |
| x0 2× 0.5 | certified (osc.) | 811,016,337.92 | 162 | 4,402.17 | 2.57e-3 | 1.22 |
| unit 2× 0.5 | certified (**monotone**) | 810,651,421.49 | 178 | 4,022.17 | 5.78e-4 | 1.41 |
| x0 2× 2.0 | certified (osc.) | 811,014,413.27 | 136 | 2,628.38 | 1.06e-3 | 0.89 |
| unit 2× 2.0 | certified (osc.) | 810,650,949.46 | 134 | 3,679.46 | 3.35e-3 | 0.97 |

**Run conditions.**
- 0 vetoes, 0 non-clean cycles and 0 local solve failures in every cell; the only gap refusals are the 7 in pair 3's
  unit cell.
- 1–4 `recovered_tier1` network records in five cells (e.g. DSO `case33_2` cycle 12, maxIterations, recovered).
- Launcher guards were all zero. Peak memory was ≤ 3.68 GB. Total wall time was 9.8 h.

**Diagnostics** (exactly 8 guarded loads of the cells' own `certified_models.pkl`, hash-checked against the committed
manifests; 13/13 negative controls refused; 3 further logged loads in smoke runs; 0 solves, 0 builds):
- At x = 0 every storage variable is fixed at 0 and every storage row inactive.
- For the unit cells see decision 1.
- Interface residual ratios: every channel's test passes in all 8 cells (maxima ≤ 0.61).
- The settlement identity closes exactly in all 8 cells.
- "ESSO σ slack": not applicable (no such name in the code).

## Not confirmed

- **ESSO-side state.** The ESSO-side SOC, slacks and pnet/qnet (`esso_models_s39_D.pkl`) were not read.
- **The mechanism of the tolerance-level exceedances** (variable bounds versus constraint rows) was not traced.
- **Same-head baselines.** The baselines predate production `6f43479e`. Equality rests on the W183c/W183d identity
  gates, not on a re-run on the same head.
- **The TN/DN decomposition** of the E2 margins was not computed (not ordered).
