# P5.15 Section 4 — confirmations and figures (2026-11-16; zero solves)

**Request:** the Expert's handback against Overleaf `0569289` or later. Checked at **`0c4a505`** (17:01 UTC); it
differs from `0569289` by 1 insertion and 4 deletions, none in the checked statements.
**No manuscript edits.** The new figures are in `Claude outputs/figures_s4_v1/` (see its NOTE.md).

## Decisions needed (Expert; wording only)

1. **"0.99 M€ in 2030" is a present value.**
   - **What the record holds.** The recorded I(x) is 992,268.11 €, discounted to 2025 at 2 %/yr. The undiscounted
     outlay in 2030 is 1,095,544.17 €, which is above the 1 M€ budget.
   - **Why the plan is still admissible.** The budget is applied to the present value.
   - **Proposed wording:** "an investment with a present value of 0.99 M€, installed in 2030".
2. **A duplicated paragraph.** "The reference unit provides an operating value of 253.5 k€ … renewable curtailment is
   unchanged." appears twice, at l. 1368–1371 and l. 1423–1426 of `0c4a505`; it was already duplicated at `0569289`.
   One copy should go.

## Blocked on the author

- Push from PyCharm.

## Changed

- `8767a869`: the figure scripts.
  - `p515_s55_fig_flex_ladder.py`;
  - `manuscript_review/fig_storage_profiles.py` with `--v2`;
  - the compare tool's q/Q fix;
  - `manuscript_review/fig_printed_fonts.py`.
- This report, `Claude outputs/figures_s4_v1/` (six PDFs, previews, verification, manifest, NOTE.md) and TASKS.md lines:
  committed together.

## Found

### 1. "Renewable curtailment falls from 808.2 to 0.7 GWh over the horizon": confirmed, same convention

| | TN | DN | total | source |
|---|---|---|---|---|
| price-taker, best start (warm), net | 807.650 | 0.576 | **808.226 → 808.2** | `report_v3.json` (`bf6dbf72`), `curtailment_table.arms_and_variants_phase_A.nrf_arm_price_taker_warm_from_certified_r2` |
| coordinated (x = 0, d110bd1a, k\* 181), net | −0.021 | 0.682 | **0.661 → 0.7** | same file, `curtailment_table.coordinated.settled_cycle_181_recomputed_from_models`; = b1 (`section4_extractions_v1.json`) |

**Convention, identical for both rows:**
- **The quantity:** `mwh_day_weighted` = Σ over blocks of the unweighted `res_curtailment_definitional_at_weight_1` ×
  Years[y] × Days[d], undiscounted.
- **Energy and sign:** active energy, net signed, scenario-probability weighted.
- **Horizon:** the same 3 representative years × 4 days for both, from one file and one function.
- **Best start.** It is the start whose Q is the Table `tab:res_benchmark` price-taker value (`q_best` 744,770,310.59;
  `best_start` = warm_from_certified). The three starts span 807.26–808.23 GWh.
- **The comparison is valid.** The 90.9 M€ saving is taken between the same two cells: Q(price-taker) − Q(coordinated)
  with the coordinated Q = 653,873.7 k€.

**Other conventions give the same figures:**
- Under the positive-part convention (Addendum 56), the price-taker reads 808.291 → 808.3 and the coordinated 0.728 → 0.7.
- With the no-reverse-flow restriction on the coordinated cell (C_NRF, W184) it reads 0.660 → 0.7.

### 2. Recorded I(x) of the returned 2030 plan

**I(x) = 992,268.11 €**, slack 7,731.89 € against the 1 M€ budget, so the plan is at the budget boundary.
- **Plan:** node 5, 0.25 MVA / 0.5 MWh; node 7, 1 MVA / 3.5 MWh; 2030 (cell `ref:5ca4f86c`).
- **Records:**
  - `frozen_step6_tables_v1_590088fe.json`: `I_ref` of every L-family claim;
  - `P5_15_W182G_COST_HORIZON_BUDGET.md` l. 215 and l. 279, where the value equals W2 `I_new_eur` to 1e-6 €.
- **Formula:** I(x) = Σ_n 1.02^−(y−2025) Σ_c ω_c (c^S_{c,y} S_n + c^E_{c,y} E_n), with ω = (0.35, 0.55, 0.10) and the
  2030 unit costs. It is a present value at 2025 (decision 1).
- **Status:** the frozen record lists this evaluation as uncertified, consistent with the † in the text.

### 3. W190 list: the new Section 4 statements, checked against the records

| statement | record value | source | check |
|---|---|---|---|
| "about two-thirds / one-third" (4.1, l. 1370 and l. 1425) | generation 62.7 % (62.7–66.7 % over the W185 scalings), flexibility 37.3 % (33.2–37.3 %) | `section4_extractions_v1.json` (a) | holds |
| "renewable curtailment is unchanged" (same two places) | b1 − b2 = 0.079 MWh, inside the bound-tolerance band | same file (b) | holds |
| "808.2 to 0.7 GWh" (4.1, l. 1363) | 808.226 and 0.661 GWh | item 1 above | holds |
| "about three-quarters" (4.3, l. 1574) | flexibility 73.7 % at 2× (70.9–73.7 %); "about one-third at the baseline price" 37.3 % (33.2–37.3 %) | same file (a), 2× pair `H:m2` | holds |
| "at most 1.9 k€" (4.4, l. 1674), operating costs | largest change of any cell from its default 1,921.25 €; largest spread 1,924.65 € (x = 0, 2×) | same file (c) | holds |
| "at most 1.6 k€" (4.4, l. 1674), the cost difference | spread 1,572.86 € (1×); 1,452.62 € (2×) | same file (c); = Appendix l. 2235 (1.6 and 1.5) | holds |
| "171.3–192.6 k€/MWh" and "253.9 k€/MWh" (4.3, l. 1566) | 171,310–192,560 €/MWh; 253,877.68 €/MWh | frozen T3 `conservative_A64`; W182G | holds; "24–33 %" = 24.2–32.5 % |
| "0.99 M€ in 2030" (4.3, l. 1586) | 992,268.11 € present value | item 2 | holds as a present value (decision 1) |

### 4–6. Figures

`Claude outputs/figures_s4_v1/`. The plotted values are unchanged, checked as stated in its NOTE.md.

| file | sha256 | change |
|---|---|---|
| `fig2_flexibility_ladder.pdf` | `0d47fe22` | "Investment margin (k€)"; two-entry legend; one colour; DejaVu Serif |
| `fig_soh_reference_unit.pdf` | `4c0a163e` | dash-dot line at SoH = 0.70, legend "Minimum SoH, 0.70"; fonts |
| `fig_soc_seasonal_{Spring,Summer,Autumn,Winter}.pdf` | `16792299`, `58e8035f`, `cc69b596`, `ce50d21a` | fonts only |

**Fonts.** All six now print in DejaVu Serif at 8.0 / 7.5 / 7.0 pt (labels / ticks / legend). Before, the flexibility
figure was Arial and the sizes ranged from 6.59 to 8.12 pt.

**The SoH caption needs one sentence;** the proposed text is in NOTE.md.

## Not confirmed

- **Not compiled:** there is no TeX here.
- **The flexibility figure's x label** was also reworded ("Flexibility-price multiplier (× baseline price)"). That was
  not requested; revert on request.
- **Arial elsewhere:** the breakeven split, `fig_q_cycle_v7` and the benchmark figure were outside the request.
