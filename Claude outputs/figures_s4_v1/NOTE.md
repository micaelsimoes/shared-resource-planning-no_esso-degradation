# Section 4 figures: flexibility ladder, SoH minimum line, common fonts (2026-11-16; zero solves)

## Manuscript checked

Overleaf `0c4a505` (2026-11-16 17:01 UTC; the handback named `0569289` or later). The six files there are:
- `fig2_flexibility_ladder.pdf` `7d15f697`, the W165 export;
- the storage files of `storage_profiles_v1` (`44ec0682`, `0251426d`, `deeb1c7b`, `dda1bbec`, `18f137e5`).

Copies are in `verification/overleaf_0c4a505/`.

## Files (same names as in Overleaf)

| file | sha256 (new) | replaces | include at |
|---|---|---|---|
| `fig2_flexibility_ladder.pdf` | `0d47fe225838c4c9` | `7d15f6970e26db9d` | 0.75\textwidth |
| `fig_soh_reference_unit.pdf` | `4c0a163e46b403ae` | `18f137e5d7dbb34b` | 1.00\textwidth |
| `fig_soc_seasonal_Spring.pdf` | `1679229921cc280d` | `44ec068295e4e88c` | 0.48\textwidth |
| `fig_soc_seasonal_Summer.pdf` | `58e8035fbacf78c1` | `0251426dd3fb859b` | 0.48\textwidth |
| `fig_soc_seasonal_Autumn.pdf` | `cc69b5960ff79f38` | `deeb1c7beabc3919` | 0.48\textwidth |
| `fig_soc_seasonal_Winter.pdf` | `ce50d21a07abdb47` | `dda1bbec1fdcbb5d` | 0.48\textwidth |

The full hashes are in `manifest_sha256.json`.

**Sources** (commit `8767a869`; no new numbers):
- `p515_s55_fig_flex_ladder.py` draws the flexibility figure. It uses W165's `data_fig2` unedited, on the frozen tables
  `590088fe`.
- `manuscript_review/fig_storage_profiles.py --v2` draws the storage figures from the committed W188d records.

**Previews** at the inclusion widths, 300 dpi: `preview_fig2_0.75textwidth.png`,
`preview_fig_res_soc_0.48textwidth.png` and `preview_fig_res_soh_1.00textwidth.png`.

## What changed

1. **`fig2_flexibility_ladder.pdf`**
   - **Labels.** The y axis reads "Investment margin (k€)" (was "value − I of the unit (k€)"). The x axis reads
     "Flexibility-price multiplier (× baseline price)" (was "flexibility-price multiplier m (× base price)"). The x
     change was not requested: it capitalises the label like the y label and drops the symbol m, which the text does
     not use. It is easy to revert.
   - **Legend.** Two entries, each sample showing its bar:
     - "evaluation meeting the acceptance criteria" (filled circle);
     - "evaluation not meeting the acceptance criteria" (open square).

     The legend sits in the empty lower-right corner.
   - **One colour.** The old legend also explained a colour code (blue for determinate, vermillion for within
     resolution). With that explanation gone, all points share one colour. Whether a margin is resolved is read from
     whether its bar crosses zero.
   - **Bars.** These are the comparison thresholds, as the current caption says. That caption needs no change.
   - **Size.** The figure is drawn at its inclusion width, 292.5 pt.
2. **`fig_soh_reference_unit.pdf`.** A horizontal line at SoH = 0.70 (grey, dash-dot, so it is not read as a
   trajectory) with the legend entry "Minimum SoH, 0.70". The three calibrations that reach 0.70 in 2040 end on it.
3. **Fonts in all six files.**
   - **Family:** DejaVu Serif throughout, the family the storage figures already used.
   - **Printed sizes at the inclusion width:** axis labels 8.0 pt, tick labels 7.5 pt, legend and value labels 7.0 pt.

   | | family | printed sizes (pt) |
   |---|---|---|
   | before: SoC (4 files) | DejaVu Serif | 6.59 / 7.61 / 8.12 |
   | before: SoH | DejaVu Serif | 7.03 / 7.53 / 8.03 |
   | before: flexibility | **Arial** | 6.88 / 7.45 / 8.03 |
   | now: all six | DejaVu Serif | **7.0 / 7.5 / 8.0** |

   Source: `verification/printed_fonts_new.json` and `printed_fonts_overleaf_0c4a505.json` (`fig_printed_fonts.py`).

   The other figures still use Arial: the breakeven split, `fig_q_cycle_v7` and the benchmark figure. Changing them
   was outside this request.

## Caption text (proposed; only the SoH caption needs a sentence)

```latex
\caption{\textcolor{blue}{State of health of the reference unit under the ageing calibrations of Table~\ref{tab:ageing_calibrations}. Values correspond to installation on 1 January 2025 and the ends of the 2025--2029, 2030--2034 and 2035--2039 representative blocks, shown at 2030, 2035 and 2040, respectively. The dash-dot line marks the minimum state of health of 0.70 assumed by the calibrations with ageing, except Baseline, 0.50 floor, which assumes 0.50.}}
```

## Plotted values unchanged

- **Flexibility figure** (`build_record.json`, 8/8 checks):
  - all W165 figure-2 checks hold;
  - the W165 composite re-rendered from the same data equals `7d15f697`, the Overleaf file, byte for byte;
  - every marker position and error-bar segment equals the composite's in data units;
  - no text is clipped;
  - two renders give the same bytes, with TrueType fonts and no dates;
  - only DejaVu Serif is used;
  - guards are at 0 and pickle loads at 0.

  **Values:** −64.4, −14.5 (open square), +14.2 and +46.3 k€ at 1, 1.5, 1.75 and 2 × the baseline price.
- **Storage figures**, compared with `storage_profiles_v1` (= Overleaf) at 0.01 pt by `fig_storage_profiles_compare.py`
  (`verification/compare_*_v2_vs_v1.json`):
  - **Identical:**
    - the page size and plot-area rectangle;
    - every trajectory, marker, bound line, grid line and tick;
    - the axis limits.
  - **Seasonal files:** only the three legend samples move (≤ 0.9 pt, from the 7 pt legend), together with the text
    sizes.
  - **SoH file:** the legend samples move ≤ 0.3 pt, and the 0.70 line and its legend sample are new. The line sits at
    79.15 pt = axes bottom + (0.70 − 0.56) / 0.46 × axes height.
- **Without `--v2`** the script still reproduces the five v1 files byte for byte.
- **The compare tool** now saves and restores the graphics state at q/Q. Without that, the new line's dash pattern
  leaked into the solid lines drawn after it. The v1 legacy check still gives paths and markers equal.

## Not confirmed

- **Not compiled:** there is no TeX here.
- **The x-label change** on the flexibility figure was not requested.
