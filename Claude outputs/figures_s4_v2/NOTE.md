# Section 4 figures: US spelling (2026-11-17; zero solves)

**Upload one file:** `fig_soh_reference_unit.pdf` (`6548969e6c8238b1`). It replaces `4c0a163e46b403ae` of
`figures_s4_v1`.

The other five figures need no change:
- **The four SoC files** here are byte-identical to `figures_s4_v1` (`16792299`, `58e8035f`, `cc69b596`, `ce50d21a`).
  They are kept so that the folder is complete.
- **`fig2_flexibility_ladder.pdf`:** use the `figures_s4_v1` copy (`0d47fe22`).

## What changed

- **Legend entry:** "No ageing" became "No aging" in the SoH figure.
- **How it was built:** `manuscript_review/fig_storage_profiles.py --us`. The new option implies `--v2` and changes
  printed legend text only; the record key stays `No ageing`.
- **Check against v1** (`verification/compare_fig_soh_reference_unit_us_vs_v1.json`): 92 operations against 92. Paths,
  markers and the page are equal. The only operation that differs is that legend string, at the same position and size.
- **Reproducibility:** `--v2` still rebuilds all five `figures_s4_v1` storage files byte for byte, checked before this
  build.
- **Logs:** `build_stdout.log` records the clipping check (no text outside the page). `build_stderr.log` is empty.

## Spelling check of all six figures

Every printed string was decoded from the PDFs (`verification/printed_fonts_v2.json`):

| figure | printed strings | British spelling |
|---|---|---|
| SoC (4 files) | "Hour", "Stored energy (MWh)", "10 % and 90 % of available energy" | none |
| SoH | "Year", "SoH (–)", "No aging", "Cycling only", "Datasheet 8 000 cycles", "Baseline, floor 0.70", "Baseline, floor 0.50", "Retention 50 %", "Retention 50 %, mid-block", "Minimum SoH, 0.70" | none (was "No ageing") |
| flexibility | "Flexibility-price multiplier (× baseline price)", "Investment margin (k€)", "evaluation meeting the acceptance criteria", "evaluation not meeting the acceptance criteria" | none |

**Fonts.** All six print in DejaVu Serif at 8.0 / 7.5 / 7.0 pt, the same as `figures_s4_v1`.

**Number format, not changed.** Three strings use the SI style: "8 000" (thin space) and "10 %" / "50 %" (space
before %). US style would be "8,000" and "10%". This is a spelling-adjacent choice for the Expert: the change is one
string each, if the text uses the US form.
