# fig:res_breakeven as three independent PDFs (2026-11-13; zero solves)

## Manuscript checked

Overleaf `df80a5af559d` (2026-11-13 17:56:12 UTC). `fig1_breakeven.pdf` there is sha256 `99ec6a6b…`, byte-identical
to the committed W165 export that the split starts from.

## Files

| file | sha256 | include at | content |
|---|---|---|---|
| `fig1_breakeven_4h.pdf` | `aff2fbf5f9dde514` | 0.48\textwidth | investment margins, 4 h configurations, node 7, 2025; legend: evaluation status and bars |
| `fig1_breakeven_2h.pdf` | `b89f2f5217e353d9` | 0.48\textwidth | investment margins, 2 h configurations, node 7, 2025; legend: the two fits |
| `fig1_breakeven_cost.pdf` | `dc71a0e7240933c9` | 0.70\textwidth | marginal break-even energy-capacity cost of the two fits against the baseline cost |

The full hashes are in `manifest_sha256.json`.

- **Proof:** `proof_fig_res_breakeven.png`, the snippet's arrangement at 300 dpi with \textwidth = 390 pt (elsarticle
  `preprint,12pt`). The grey frame marks the text width. Not typeset by LaTeX.
- **Plotting source:** `p515_s55_fig_breakeven_split.py` (commit `4943e002`). It calls the existing figure script
  `p515_s53_w165_step6_figures.py` unedited: its data function `data_fig1` and its drawing calls.
- **Records read:**
  - the frozen tables `frozen_step6_tables_v1_590088fe.json` (T3 conservative_A64, T2 cells, claims);
  - the W145 committed points `w145_banded_fit.json`.
- **Sizes:** each component is drawn at its inclusion width (187.2 × 175 pt; 273 × 118 pt), so text prints at its
  stated size:
  - tick labels 6.5 pt;
  - axis labels 7 pt;
  - legends and annotations 6.5 pt.

  The composite was 190 mm wide, scaled to 0.72, so its legend printed at about 4.3 pt.

## LaTeX snippet (replaces the current figure block)

```latex
\begin{figure}[htbp!]
    \centering
    \subfloat[4~h configurations.\label{fig:res_breakeven_4h}]{\includegraphics[width=0.48\textwidth]{fig1_breakeven_4h.pdf}}
    \hfill
    \subfloat[2~h configurations.\label{fig:res_breakeven_2h}]{\includegraphics[width=0.48\textwidth]{fig1_breakeven_2h.pdf}}
    \\
    \subfloat[Marginal break-even energy-capacity cost.\label{fig:res_breakeven_cost}]{\includegraphics[width=0.70\textwidth]{fig1_breakeven_cost.pdf}}
    \caption{\textcolor{blue}{Investment margins of the evaluated storage configurations at node~7 installed in 2025, and the marginal break-even energy-capacity cost obtained from them. In (a) and (b), circles are certified evaluations, with bars showing the band width; squares are uncertified evaluations and the diamond is a flagged certificate entered as an interval. For these, thick bars show the comparison bar and thin bars add $2\tau$, defining the fitting intervals. Lines show the certified-only and interval-midpoint fits. In (c), the bar spans the break-even costs over the fitting intervals; it is a range, not a statistical confidence interval. Configurations above the 1~M\euro{} budget (4 and 5~MWh at 4~h; 3--5~MWh at 2~h) are valuation probes outside the admissible set.}}
    \label{fig:res_breakeven}
\end{figure}
```

- **Labels:** `fig:res_breakeven` is kept for the figure. Each subfigure gets its own label: `fig:res_breakeven_4h`,
  `fig:res_breakeven_2h` and `fig:res_breakeven_cost`.
- **Mechanism:** the same `\subfloat` arrangement that `fig:res_soc` uses.
- **References:** the main text's one reference (l. 1417) is to the whole figure.

## What changed (presentation only)

**(a) and (b):**
- **Shared frame:** the same page size, the same axes rectangle, and the composite's shared limits (x 0.8–5.2 MWh,
  y −649.7 to 30.9 k€).
- **Axis labels:** the ordinate reads "Investment margin (k€)" (was "value − I (k€)"), and the abscissa reads
  "Installed energy capacity (MWh)" (was "installed energy E (MWh), node 7, 4 h"). Node, year and duration move to
  the captions.
- **Legends above the axes:** (a) carries the evaluation status and bars, (b) the two fits.
  - The status samples are the plot's own error-bar artists, so each shows the real bars.
  - Wording:
    - "certified; bar: band width";
    - "uncertified; thick bar: ±bar; thin: ±(bar + 2τ)";
    - "flagged certificate, entered as an interval".
  - "banded fit, interval midpoints" is renamed "interval-midpoint fit".

**(c):**
- **Axis label:** "Marginal break-even energy-capacity cost (k€/MWh)" (was "break-even energy cost e* (k€/MWh)").
- **Baseline line:** "baseline cost 253.88" (was "energy cost").
- **Arrow from 192.56 to 253.88:** "≥ 61.32 below baseline" (was "margin ≥ 61.32").
- **Row labels:** "certified-only fit (n = 6)" and "interval-midpoint fit and range over fitting intervals (n = 10)".
  The word "confidence" appears nowhere.

**All three:** no panel letters; LaTeX adds them.

## Plotted values unchanged

- **Same data as the composite.** Before drawing, the script re-rendered the composite from the same data. It matches
  the committed `fig1_breakeven.pdf` (= the Overleaf file) byte for byte, and all W165 figure-1 checks hold. These
  include the W145 fit call reproducing T3 conservative_A64 bitwise and every point's value − I = Q0 − Q − I.
- **Same drawn data in each component.** For each component, every drawn datum was compared exactly, in data units,
  with the corresponding composite panel:
  - fit-line vertices;
  - marker positions;
  - every error-bar segment (band width, ±bar, ±(bar + 2τ));
  - the zero line;
  - the baseline-cost line;
  - the arrow end points.

  All three are equal. The check is in `build_record.json`, `checks.*.plotted_data_equal_composite`.
- **Values** (from the build record):

| | value |
|---|---|
| certified-only fit (n = 6) | break-even 183.08 k€/MWh |
| interval-midpoint fit (n = 10) | break-even 181.93 k€/MWh; range over the fitting intervals 171.31–192.56 k€/MWh |
| baseline energy-capacity cost | 253.88 k€/MWh; differences 61.32–82.57 k€/MWh |
| 2τ | 9,078.14 € |
| 4 h points, E = 1–5 MWh (k€) | −64.42, −135.54, −208.29 (flagged, bar 70.74, interval ±79.82), −273.95, −347.24 (uncertified, bar 23.92, ±33.00) |
| 2 h points, E = 1–5 MWh (k€) | −114.08, −231.71 (uncertified, bar 24.24, ±33.31), −361.68 (uncertified, bar 41.38, ±50.46), −494.42, −616.55 |

The main text at l. 1417 agrees with these values: 171,310–192,560 €/MWh; 253,880 €/MWh; 24–33 % below.

## Checks

- **Clipping:** every drawn label lies inside its page.
- **Determinism and fonts:** each PDF rendered twice gives the same bytes, with TrueType fonts and no dates.
- **(a)/(b) frame:** identical limits and axes rectangle.
- **Zero solves:** the W165 and W157 guards are at 0, and pickle loads at 0.

## Not confirmed

- **Not compiled.** There is no TeX here, so the typeset figure and the `\subfloat` + `subcaption` combination are not
  checked. `fig:res_soc` already uses it.
- **Budget:** the above-budget sets in the caption are as instructed and as in the current caption. They were not
  re-derived from the budget here.
