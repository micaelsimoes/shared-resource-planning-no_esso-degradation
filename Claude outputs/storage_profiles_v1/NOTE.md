# Storage-profile figures: presentation-only update (2026-11-13; zero solves)

## Manuscript checked

Overleaf `dcd135c48a9f` (2026-11-13 17:41:01 UTC), fetched 17:46 UTC; the work was done against `236fb5ea`
(17:32:18 UTC).
- The five figure files and the two captions are byte-identical at both commits.
- `dcd135c` changes only the break-even and flexibility-price paragraphs. Their new numbers are for W190.

## Deliverables (same filenames as in Overleaf)

| file | sha256 (new) | replaces (Overleaf) | page (pt) |
|---|---|---|---|
| `fig_soc_seasonal_Spring.pdf` | `44ec068295e4e88c` | `7feb5346b71ab023` | 184.536 × 158.4 |
| `fig_soc_seasonal_Summer.pdf` | `0251426dd3fb859b` | `9d3377dc08179c05` | 184.536 × 158.4 |
| `fig_soc_seasonal_Autumn.pdf` | `deeb1c7beabc3919` | `a0b6809947763cf9` | 184.536 × 158.4 |
| `fig_soc_seasonal_Winter.pdf` | `dda1bbec1fdcbb5d` | `6f073902cb820968` | 184.536 × 158.4 |
| `fig_soh_reference_unit.pdf` | `18f137e5d7dbb34b` | `58693e5fe5fd93b7` | 388.512 × 201.6 |

The full hashes are in `manifest_sha256.json`.

- **Source:** `manuscript_review/fig_storage_profiles.py` (commit `52b7b235`), run with the canonical interpreter
  (Matplotlib 3.10.8). It reads only the committed W188d records in
  `data/SRP1/Results/P515S55/w188d_operational_figure/`:
  - `soc_hourly_reference_unit_2025_2035.csv` (`459962df`);
  - `soc_limits_per_year_reference_unit.csv` (`eb2d05ed`);
  - `soh_per_block_calibrations.csv` (`9243aafe`).
- **Previews at the inclusion sizes:**
  - `preview_fig_res_soc_0.48textwidth.png`: the four seasonal files at 0.48 \textwidth each, two per row with
    \hfill;
  - `preview_fig_res_soh_1.00textwidth.png`: the SoH figure at full \textwidth.

  Both are at 300 dpi, with \textwidth = 390 pt (elsarticle `preprint,12pt` → article at 12 pt; no geometry
  package). The grey frame marks the text width. Not typeset by LaTeX.

## Paste-ready captions

```latex
\caption{\textcolor{blue}{Stored energy of the reference unit over the four representative days of 2025 and 2035 under the baseline ageing calibration. Dashed lines mark the operating limits, 10\,\% and 90\,\% of the available energy of each year.}}
\label{fig:res_soc}
```

```latex
\caption{\textcolor{blue}{State of health of the reference unit under the ageing calibrations of Table~\ref{tab:ageing_calibrations}, at installation on 1 January 2025 (2025) and at the ends of the 2025--2029, 2030--2034 and 2035--2039 representative blocks (2030, 2035 and 2040). The ageing calibrations assume a minimum state of health of 0.70, except Baseline, 0.50 floor, which assumes 0.50; the baseline and both Retention 50\,\% calibrations reach 0.70 at the end of the 2035--2039 block.}}
\label{fig:res_soh}
```

**Basis of the SoH caption** (`soh_min_applied`, `floor_row_active_at_terminal_flag` in
`soh_per_block_calibrations.csv`; Table `tab:ageing_calibrations`):
- the minimum is 0.70 for the five ageing calibrations, there is none for No ageing, and it is 0.50 for "Baseline,
  0.50 floor";
- the floor is active at the 2035 block end (the 2040 point) for Baseline, Retention 50 % and Retention 50 %,
  mid-block (0.6999999900).

The figure draws no horizontal floor lines, and the caption no longer implies that it does.

## What changed

**The four seasonal files:**
1. **y-axis label:** "SoC (MWh)" → "Stored energy (MWh)".
2. **Legend entry:** "10 % and 90 % of available capacity" → "10 % and 90 % of available energy", the caption's
   wording.
3. **Dash pattern of the 10 % and 90 % lines:** [1.4 1.4] pt → [3.5 1.75] pt, at the same width (0.7 pt), colours
   and positions.
   - **Why.** At 0.48 \textwidth the old equal on/off pattern reads as dotted, which is presumably why the caption
     said "dotted".
   - **Effect.** The lines and the legend sample now read as dashed, matching the caption. The legacy pattern stays
     available (`--legacy-labels`).

**The SoH file:** no deliberate change. It is redrawn from the same records with the same geometry, axis range
(2024.4–2040.6 × 0.56–1.02), x positions 2025/2030/2035/2040, seven trajectories, markers, colours and legend.

## Plotted values and operating bounds unchanged

Checked with `manuscript_review/fig_storage_profiles_compare.py`, which compares the PDF drawing operations of the
old and new files at 0.01 pt. The JSON reports are in `verification/`.

1. **The source reproduces the Overleaf files.**
   - Run with the old wording and dash (`--legacy-labels`) under Matplotlib 3.11.2, the version that made the
     Overleaf files (a scratch environment, used only for this check), it reproduces all five files exactly:
     - every path and marker;
     - every text string and position;
     - the page size and the plot-area rectangle.
   - The only residue is in the SoH file: the en dash and thin space sit in a different internal font chunk, with
     identical glyphs at identical positions.
   - Renders: `verification/legacy_render_mpl3.11.2/`; reports: `compare_*_legacy_mpl3112_vs_overleaf.json`.
2. **The delivered files against Overleaf** (dash pattern ignored): `compare_*_new_vs_overleaf.json`.
   - **Identical:** the page size, the plot-area rectangle, every data line (both years, all four days), the four
     10 %/90 % bound lines per day, all seven SoH trajectories with their 28 markers, the grid, the ticks and the axis
     limits.
   - **Differs:**
     - the label and legend strings above;
     - legend samples shifted vertically by ≤ 0.6 pt (the 2025 and 2035 samples in each seasonal file; the 7 lines and 7
       markers of the SoH legend);
     - tick and axis-label text offsets of ≤ 1 pt.

     These come from Matplotlib 3.10.8 measuring text height slightly differently from 3.11.2. No data are involved.

**Values drawn** (record, MWh):

| year | available energy | 10 % bound | 90 % bound | 00:00 and 24:00 target (50 %) |
|---|---|---|---|---|
| 2025 | 0.88046 | 0.08805 | 0.79241 | 0.44023 |
| 2035 | 0.70000 | 0.07000 | 0.63000 | 0.35000 |

- **Range:** the usable range of the manuscript text (0.70 MWh in 2025, 0.56 MWh in 2035) equals 90 % − 10 % of
  these values.
- **SoH at 2040:**
  - Baseline 0.70000, Retention 50 % 0.70000, Retention 50 % mid-block 0.70000;
  - Cycling only 0.83530, Datasheet 0.76015, Baseline 0.50 floor 0.67729, No ageing 1.00000.

## Checks

- **Clipping:** every tick label, axis label and legend entry lies inside the page in all five files (a check in the
  source; the run fails otherwise).
- **Axis labels:**
  - seasonal: "Hour" and "Stored energy (MWh)";
  - SoH: "Year" and "SoH (–)", kept.
- **Legends:**
  - seasonal: 2025, 2035, and dashed "10 % and 90 % of available energy";
  - SoH: the seven calibrations, unchanged.
- **Caption consistency:**
  - "dashed" now matches the drawn lines and the legend sample;
  - "available energy" now matches the legend;
  - the SoH caption names the four x positions and the two minimum-SoH assumptions;
  - the main-text sentence at l. 1547 of `dcd135c` agrees with the plotted values.

## For the author (not changed; the instruction was to keep the legend)

- **Legend names that differ from the table rows:**
  - "Baseline, floor 0.70" vs "Baseline (cycling + calendar)";
  - "Baseline, floor 0.50" vs "Baseline, 0.50 floor".
- **"Baseline, 0.50 floor" is a separate evaluation**, not the baseline with only its floor changed. It lies below
  the baseline from the first block onwards (0.8726 vs 0.8805 in 2030), and the caption does not say they differ only
  in the floor.
- **Not compiled:** there is no TeX here, so `\subfloat` under `subcaption` and the typeset sizes are not checked.
