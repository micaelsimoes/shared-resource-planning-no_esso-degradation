"""Storage-profile figures of the manuscript: fig:res_soc (four seasonal files) and fig:res_soh (one file).

Presentation-only rebuild (2026-11-13). No solves: reads the committed W188d records only.

    fig_soc_seasonal_{Spring,Summer,Autumn,Winter}.pdf  stored energy of the reference unit (node 7) over the
        representative day, 2025 and 2035, baseline calibration (C2_calfade); dashed lines at 10 % and 90 % of the
        available energy of each year
    fig_soh_reference_unit.pdf  state of health at the calendar boundaries 2025 (installation, 1 January 2025), 2030,
        2035 and 2040 (ends of the 2025-2029, 2030-2034 and 2035-2039 representative blocks), seven calibrations

INPUTS (committed, data/SRP1/Results/P515S55/w188d_operational_figure/):
    soc_hourly_reference_unit_2025_2035.csv   soc_tso_copy_mwh per (year, day, hour_ending 1..24)
    soc_limits_per_year_reference_unit.csv    soc_start_and_target_end_50pct_record_mwh (the 00:00 point),
                                              soc_lower_10pct_record_mwh, soc_upper_90pct_record_mwh (the bounds)
    soh_per_block_calibrations.csv            soh_end_of_block per (calibration_table6, block_year)

GEOMETRY. The page and axes rectangles, limits, ticks, fonts, line widths, dash patterns, colours, markers and legend
placement reproduce the files in the Overleaf project as of 2026-11-13 (the separate files uploaded at 14:33-14:43 UTC,
generator not in the repository); they were read from those PDFs' content streams. `--legacy-labels` draws the
previous wording and dash pattern so that the rebuild can be compared with the old files operator by operator.

CHANGES (author/Expert, 2026-11-13): seasonal y label 'SoC (MWh)' -> 'Stored energy (MWh)'; seasonal legend entry
'10 % and 90 % of available capacity' -> '10 % and 90 % of available energy' (the caption's wording); the 10 %/90 %
bound dash pattern [1.4 1.4] pt -> [3.5 1.75] pt, so that the lines read as dashed (same positions, width, colours).
The SoH figure is drawn unchanged.

CHECKS: every text artist (tick labels, axis labels, legend entries) must lie inside the page; the run fails if one is
clipped. --preview-dir writes the two figures at their inclusion sizes in main.tex: each seasonal file at
0.48 \\textwidth (\\subfloat, two per row, \\hfill) and the SoH file at 1.00 \\textwidth, with \\textwidth = 390 pt
(elsarticle 'preprint,12pt' loads article at 12 pt, a4paper, no geometry), at 300 dpi.

Usage: <canonical python> manuscript_review/fig_storage_profiles.py --out-dir DIR [--legacy-labels] [--png-dir DIR]
                                                                        [--preview-dir DIR]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S55', 'w188d_operational_figure')
PT = 1.0 / 72.0

DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')
YEARS = ('2025', '2035')
C_YEAR = {'2025': '#2a78d6', '2035': '#eb6834'}
BOUND_DASH_LEGACY = (0, (2, 2))    # at lw 0.7: [1.4 1.4] pt, the pattern of the 14:33 UTC files (reads as dotted)
BOUND_DASH = (0, (5, 2.5))         # at lw 0.7: [3.5 1.75] pt, reads as dashed at 0.48 textwidth (2026-11-13)
SEASONAL_PAGE_PT = (184.536, 158.4)
SEASONAL_AXES_PT = (35.06184, 30.096, 133.7886, 96.624)
SOH_PAGE_PT = (388.512, 201.6)
SOH_AXES_PT = (34.96608, 28.224, 343.83312, 167.328)

LABELS = {
    'current': {'soc_y': 'Stored energy (MWh)', 'bound': '10 % and 90 % of available energy'},
    'legacy': {'soc_y': 'SoC (MWh)', 'bound': '10 % and 90 % of available capacity'},
}

# (legend label, calibration_table6 in the record, colour, line style, marker)
SOH_SERIES = (
    ('No ageing', 'No ageing', '#777777', '-', 'o'),
    ('Cycling only', 'Cycling only', '#eb6834', '-', 'o'),
    ('Datasheet 8 000 cycles', 'Datasheet 8 000 cycles', '#eda100', '-', 'o'),
    ('Baseline, floor 0.70', 'Baseline (cycling + calendar)', '#2a78d6', '-', 'o'),
    ('Baseline, floor 0.50', 'Baseline, 0.50 floor', '#2a78d6', (0, (4, 2)), 's'),
    ('Retention 50 %', 'Retention 50 %', '#1baf7a', '-', 'o'),
    ('Retention 50 %, mid-block', 'Retention 50 %, mid-block', '#1baf7a', (0, (4, 2)), 's'),
)
SOH_X = (2025, 2030, 2035, 2040)   # installation 1 Jan 2025; ends of the 2025-29, 2030-34, 2035-39 blocks


def style():
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.labelsize': 8, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'grid.linewidth': 0.3, 'grid.color': '#dddddd', 'axes.unicode_minus': True})


def _rows(name):
    with open(os.path.join(DATA, name), encoding='utf-8', newline='') as h:
        return list(csv.DictReader(h))


def load():
    soc = _rows('soc_hourly_reference_unit_2025_2035.csv')
    lim = {r['year']: r for r in _rows('soc_limits_per_year_reference_unit.csv')}
    soh = _rows('soh_per_block_calibrations.csv')
    series = {}
    for d in DAYS:
        for y in YEARS:
            rows = sorted((r for r in soc if r['day'] == d and r['year'] == y), key=lambda r: int(r['hour_ending']))
            assert [int(r['hour_ending']) for r in rows] == list(range(1, 25)), (d, y)
            start = float(lim[y]['soc_start_and_target_end_50pct_record_mwh'])
            series[(d, y)] = ([0] + [int(r['hour_ending']) for r in rows],
                              [start] + [float(r['soc_tso_copy_mwh']) for r in rows])
    bounds = {y: (float(lim[y]['soc_lower_10pct_record_mwh']), float(lim[y]['soc_upper_90pct_record_mwh']))
              for y in YEARS}
    sohs = {}
    for _lab, key, _c, _ls, _m in SOH_SERIES:
        rows = sorted((r for r in soh if r['calibration_table6'] == key), key=lambda r: int(r['block_year']))
        assert [int(r['block_year']) for r in rows] == [2025, 2030, 2035], key
        sohs[key] = [1.0] + [float(r['soh_end_of_block']) for r in rows]
    return series, bounds, sohs


def _axes(page_pt, axes_pt):
    w, h = page_pt
    fig = plt.figure(figsize=(w * PT, h * PT))
    x, y, aw, ah = axes_pt
    return fig, fig.add_axes([x / w, y / h, aw / w, ah / h])


def draw_seasonal(day, series, bounds, labels):
    fig, ax = _axes(SEASONAL_PAGE_PT, SEASONAL_AXES_PT)
    ax.grid(True)
    for y in YEARS:
        ax.hlines(bounds[y], 0, 24, colors=C_YEAR[y], lw=0.7, linestyles=[BOUND_DASH])
    for y in YEARS:
        hx, sv = series[(day, y)]
        ax.plot(hx, sv, color=C_YEAR[y], lw=1.4, solid_capstyle='round', zorder=3)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.85)
    ax.xaxis.set_major_locator(FixedLocator([0, 6, 12, 18, 24]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f'{int(round(v)):02d}:00'))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f'{v:.2f}'))
    ax.set_xlabel('Hour', labelpad=2.0)
    ax.set_ylabel(labels['soc_y'], labelpad=3.0)
    handles = [Line2D([], [], color=C_YEAR['2025'], lw=1.4, label='2025'),
               Line2D([], [], color=C_YEAR['2035'], lw=1.4, label='2035'),
               Line2D([], [], color='#555555', lw=0.7, ls=BOUND_DASH, label=labels['bound'])]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0, fontsize=6.5,
              handlelength=1.8, handletextpad=0.5, labelspacing=0.2)
    return fig


def draw_soh(sohs):
    fig, ax = _axes(SOH_PAGE_PT, SOH_AXES_PT)
    ax.grid(True)
    for lab, key, c, ls, mk in SOH_SERIES:
        ax.plot(SOH_X, sohs[key], color=c, ls=ls, lw=1.4, marker=mk, ms=3.5, mfc='white', mec=c, mew=1.0, label=lab)
    ax.set_xlim(2024.4, 2040.6)
    ax.set_ylim(0.56, 1.02)
    ax.xaxis.set_major_locator(FixedLocator(list(SOH_X)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f'{int(round(v))}'))
    ax.yaxis.set_major_locator(FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f'{v:.2f}'))
    ax.set_xlabel('Year', labelpad=3.0)
    ax.set_ylabel('SoH (–)')
    ax.legend(loc='lower left', ncol=2, fontsize=7, borderaxespad=0.4, handlelength=2.4, labelspacing=0.3,
              columnspacing=1.2)
    return fig


TEXTWIDTH_PT = 390.0
INCLUDE_FRAC = {'seasonal': 0.48, 'soh': 1.00}


def clipped_text(fig):
    """Text artists whose rendered extent leaves the page (tolerance 0.01 pt)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        e = t.get_window_extent(r)
        if e.x0 < fb.x0 - 0.01 or e.y0 < fb.y0 - 0.01 or e.x1 > fb.x1 + 0.01 or e.y1 > fb.y1 + 0.01:
            bad.append(t.get_text())
    return bad


def preview(png_by_name, out_dir):
    """Compose the LaTeX layout at 300 dpi from per-figure PNGs rendered at their inclusion widths."""
    from PIL import Image, ImageDraw
    px = lambda pt: int(round(pt / 72.0 * 300))  # noqa: E731
    tw = px(TEXTWIDTH_PT)
    seas = [Image.open(png_by_name[f'fig_soc_seasonal_{d}']) for d in DAYS]
    w, h = seas[0].size
    gap_v = px(12)
    canvas = Image.new('RGB', (tw + 2 * px(6), 2 * h + gap_v + 2 * px(6)), 'white')
    for i, im in enumerate(seas):
        x = px(6) + (0 if i % 2 == 0 else tw - w)
        y = px(6) + (i // 2) * (h + gap_v)
        canvas.paste(im, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle([px(6) - 1, 0, px(6) + tw, canvas.size[1] - 1], outline=(200, 200, 200))
    canvas.save(os.path.join(out_dir, 'preview_fig_res_soc_0.48textwidth.png'), dpi=(300, 300))
    soh = Image.open(png_by_name['fig_soh_reference_unit'])
    c2 = Image.new('RGB', (tw + 2 * px(6), soh.size[1] + 2 * px(6)), 'white')
    c2.paste(soh, (px(6), px(6)))
    ImageDraw.Draw(c2).rectangle([px(6) - 1, 0, px(6) + tw, c2.size[1] - 1], outline=(200, 200, 200))
    c2.save(os.path.join(out_dir, 'preview_fig_res_soh_1.00textwidth.png'), dpi=(300, 300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--legacy-labels', action='store_true')
    ap.add_argument('--png-dir', default=None)
    ap.add_argument('--preview-dir', default=None)
    args = ap.parse_args()
    labels = LABELS['legacy' if args.legacy_labels else 'current']
    global BOUND_DASH
    if args.legacy_labels:
        BOUND_DASH = BOUND_DASH_LEGACY
    style()
    series, bounds, sohs = load()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {'Creator': 'manuscript_review/fig_storage_profiles.py', 'CreationDate': None, 'ModDate': None}
    figs = [(f'fig_soc_seasonal_{d}', draw_seasonal(d, series, bounds, labels)) for d in DAYS]
    figs.append(('fig_soh_reference_unit', draw_soh(sohs)))
    clip = {name: clipped_text(fig) for name, fig in figs}
    if any(clip.values()):
        raise SystemExit(f'clipped text: {clip}')
    print('clipping check: no text outside the page in', ', '.join(n for n, _f in figs))
    pngs = {}
    for name, fig in figs:
        fig.savefig(os.path.join(args.out_dir, f'{name}.pdf'), metadata=meta)
        if args.png_dir:
            os.makedirs(args.png_dir, exist_ok=True)
            fig.savefig(os.path.join(args.png_dir, f'{name}.png'), dpi=300)
        if args.preview_dir:
            os.makedirs(args.preview_dir, exist_ok=True)
            frac = INCLUDE_FRAC['soh' if name.startswith('fig_soh') else 'seasonal']
            scale = frac * TEXTWIDTH_PT / fig.get_figwidth() / 72.0
            pngs[name] = os.path.join(args.preview_dir, f'_{name}_at_inclusion.png')
            fig.savefig(pngs[name], dpi=300 * scale)
        plt.close(fig)
        print('wrote', os.path.join(args.out_dir, f'{name}.pdf'))
    if args.preview_dir:
        preview(pngs, args.preview_dir)
        for f in pngs.values():
            os.remove(f)
        print('wrote previews in', args.preview_dir)


if __name__ == '__main__':
    main()
