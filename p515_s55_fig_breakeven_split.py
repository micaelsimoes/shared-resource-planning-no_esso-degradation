"""P5.15 S55 (presentation pass): Figure fig:res_breakeven (fig1_breakeven.pdf) split into three independent vector PDFs.

ZERO SOLVES. Imports the W165 figure script (p515_s53_w165_step6_figures.py, unedited): its SolveProfileGuard
(permitted ()) and the W157 guards are armed and pickle is blocked at import; all are verified at 0 before writing.

DATA. W165.data_fig1 is called exactly as W165.main calls it (frozen tables frozen_step6_tables_v1_590088fe.json; the
W145 committed points; the intercept by calling W145.banded_breakeven_fit, which must reproduce T3 conservative_A64
bitwise). Every W165 figure-1 check must hold. Before any component is drawn, the composite is re-rendered from these
data with W165.draw_fig1 and must equal the committed fig1_breakeven.pdf (99ec6a6b..., the Overleaf file) bitwise.

COMPONENTS (each its own figure, drawn at its inclusion width in main.tex, \\textwidth = 390 pt, so text prints at
its stated size):
  fig1_breakeven_4h.pdf   0.48 \\textwidth: investment margin (value - I) of the evaluated 4 h configurations, node 7,
                          2025; the two fits; certified points with band-width bars, uncertified points with the thick
                          comparison bar and the thin bar + 2 tau (the fitting interval), the flagged certificate as a
                          diamond; legend (evaluation status and bars) above the axes
  fig1_breakeven_2h.pdf   0.48 \\textwidth: the same for the 2 h configurations; legend (the two fits) above the axes.
                          (a) and (b) share one figure size, one axes rectangle and identical axis limits (the composite's
                          shared limits)
  fig1_breakeven_cost.pdf 0.70 \\textwidth: marginal break-even energy-capacity cost of the certified-only fit (point)
                          and of the interval fit (midpoint fit, square, with the range over the fitting intervals), the
                          baseline energy-capacity cost (vertical line); the arrow from the upper end of the range to the
                          baseline cost is labelled as a difference from the baseline cost
Every drawing call, value, colour, marker, bar width and interval is the composite's; only labels, legend placement and
wording, figure sizes, and the panel letters (removed; LaTeX labels the subfigures) differ.

CHECK OF THE PLOTTED VALUES. The data of every drawn artist (line vertices, marker positions, error-bar segments, the
reference lines, the annotation arrow end points) are collected in data units from the composite's three axes and from
the three components and compared exactly; the run fails on any difference.

OUTPUTS (OUT_DIR must not exist or be empty): the three PDFs (each rendered twice, sha256 equal; TrueType fonts; no
dates), proof_fig_res_breakeven.png (the LaTeX arrangement at 300 dpi), build_record.json (values, records, checks),
manifest_sha256.json. The captions and the LaTeX snippet are in the NOTE.md written beside them.

Usage: <canonical python> p515_s55_fig_breakeven_split.py [--out-dir DIR] [--trial]
"""
import argparse
import hashlib
import io
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s53_w165_step6_figures as W165  # noqa: E402 -- arms the guards and blocks pickle at import; not edited

plt = W165.plt
np = W165.np
OK = W165.OKABE_ITO
blended_transform_factory = W165.blended_transform_factory
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

SCRIPT_REL = os.path.basename(__file__)
OUT_DIR_DEFAULT = os.path.join('Claude outputs', 'fig_breakeven_v1')
FIG1_COMMITTED = os.path.join(W165.OUT_DIR, 'fig1_breakeven.pdf')
FIG1_SHA = '99ec6a6bf869b1551545758ccbb47dfec76d6225500a83bdd5b73db97aa22808'
CLONE_FIG1 = os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'fig1_breakeven.pdf')
TEXTWIDTH_PT = 390.0           # elsarticle 'preprint,12pt' -> article 12 pt
W_AB = 0.48 * TEXTWIDTH_PT
H_AB = 175.0
W_C = 0.70 * TEXTWIDTH_PT
H_C = 118.0
AXES_AB_PT = (37.0, 27.0, W_AB - 37.0 - 5.0, H_AB - 27.0 - 41.0)   # left, bottom, width, height; legend band above
FONT = {'tick': 6.5, 'label': 7.0, 'legend': 6.5, 'annot': 6.5}
PDF_METADATA = {'Creator': SCRIPT_REL, 'Producer': 'Matplotlib pdf backend', 'CreationDate': None, 'ModDate': None}
FIT_STYLE = {'certified_only': {'color': OK['blue'], 'ls': '-'}, 'banded_mid': {'color': OK['vermillion'], 'ls': '--'}}
FIT_LABEL = {'certified_only': 'certified-only fit', 'banded_mid': 'interval-midpoint fit'}
Y_LABEL_AB = 'Investment margin (k€)'
X_LABEL_AB = 'Installed energy capacity (MWh)'
X_LABEL_C = 'Marginal break-even energy-capacity cost (k€/MWh)'


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def style():
    W165.set_style()
    W165.matplotlib.rcParams.update({'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'],
                                     'axes.labelsize': FONT['label'], 'legend.fontsize': FONT['legend']})


def _fig_fixed(w_pt, h_pt, rect_pt):
    fig = plt.figure(figsize=(w_pt / 72.0, h_pt / 72.0))
    x, y, aw, ah = rect_pt
    return fig, fig.add_axes([x / w_pt, y / h_pt, aw / w_pt, ah / h_pt])


def _draw_margin_panel(ax, D, h):
    """The composite's (a)/(b) drawing calls for duration h (W165.draw_fig1), without panel letter and labels."""
    pts = [p for p in D['points'] if p['h'] == h]
    es = [p['E'] for p in pts]
    ee = np.linspace(min(es), max(es), 2)
    for fk, f in D['fits'].items():
        slope = f['b'] + f['c'] / h - D['p_cost'] / h - D['e_cost']
        ax.plot(ee, W165._k(f['a'] + slope * ee), color=FIT_STYLE[fk]['color'], ls=FIT_STYLE[fk]['ls'], lw=0.9,
                zorder=2)
    ax.axhline(0.0, color=OK['grey'], lw=0.6, zorder=1)
    cont = {}
    for p in pts:
        if p['kind'] == 'certified':
            cont['certified'] = ax.errorbar([p['E']], [W165._k(p['y'])], yerr=[[W165._k(p['band'] / 2)], [W165._k(p['band'] / 2)]],
                        fmt='o', color=OK['black'], ms=3.5, elinewidth=0.8, capsize=0, zorder=4)
        else:
            mk = 'D' if p['status_T2'] == 'certified' else 's'
            thin = ax.errorbar([p['E']], [W165._k(p['y'])], yerr=[[W165._k(p['hw'])], [W165._k(p['hw'])]], fmt='none',
                               ecolor=OK['grey'], elinewidth=0.7, capsize=1.5, zorder=3)
            cont['flagged' if mk == 'D' else 'uncertified'] = (thin, ax.errorbar([p['E']], [W165._k(p['y'])], yerr=[[W165._k(p['bar'])], [W165._k(p['bar'])]], fmt=mk,
                        mfc='white', color=OK['black'], ms=3.2, elinewidth=1.6, capsize=0, zorder=4))
    ax.set_xticks(sorted(set(es)))
    return cont


def draw_margin(D, h, lims, which):
    fig, ax = _fig_fixed(W_AB, H_AB, AXES_AB_PT)
    cont = _draw_margin_panel(ax, D, h)
    ax.set_xlim(*lims['x'])
    ax.set_ylim(*lims['y'])
    ax.set_xlabel(X_LABEL_AB)
    ax.set_ylabel(Y_LABEL_AB)
    W165.kfmt(ax, 'y')
    hmap = {}
    if which == 'status':
        from matplotlib.legend_handler import HandlerTuple
        if set(cont) != {'certified', 'uncertified', 'flagged'}:
            raise RuntimeError(f'status legend needs all three kinds in this panel, found {sorted(cont)}')
        handles = [cont['certified'], cont['uncertified'], cont['flagged']]
        labels = ['certified; bar: band width', 'uncertified; thick bar: ±bar; thin: ±(bar + 2τ)',
                  'flagged certificate, entered as an interval']
        hmap = {tuple: HandlerTuple(ndivide=1)}
    else:
        handles = [Line2D([], [], color=FIT_STYLE[fk]['color'], ls=FIT_STYLE[fk]['ls'], lw=0.9)
                   for fk in D['fits']]
        labels = [f"{FIT_LABEL[fk]} (n = {f['n']})" for fk, f in D['fits'].items()]
    ax.legend(handles=handles, labels=labels, handler_map=hmap, loc='lower left', bbox_to_anchor=(-AXES_AB_PT[0] / AXES_AB_PT[2] + 0.01, 1.02),
              borderaxespad=0.0, handlelength=1.6, handletextpad=0.5, labelspacing=0.3, fontsize=FONT['legend'])
    return fig


def draw_cost(D):
    rect = (118.0, 26.0, W_C - 118.0 - 8.0, H_C - 26.0 - 6.0)
    fig, axc = _fig_fixed(W_C, H_C, rect)
    f_c, f_b = D['fits']['certified_only'], D['fits']['banded_mid']
    yb, yc = 0.0, 1.0
    lo, hi = D['rng']
    k = W165._k
    axc.errorbar([k(f_b['e_star'])], [yb], xerr=[[k(f_b['e_star'] - lo)], [k(hi - f_b['e_star'])]], fmt='s',
                 color=FIT_STYLE['banded_mid']['color'], mfc='white', ms=3.5, elinewidth=1.2, capsize=2.0)
    axc.plot([k(f_c['e_star'])], [yc], 'o', color=FIT_STYLE['certified_only']['color'], ms=3.5)
    axc.axvline(k(D['e_cost']), color=OK['black'], lw=0.9)
    tr = blended_transform_factory(axc.transData, axc.transAxes)
    axc.text(k(D['e_cost']) - 1.5, 0.97, f"baseline cost\n{W165.fmt_kmwh(D['e_cost'])}", transform=tr, ha='right',
             va='top', fontsize=FONT['annot'])
    axc.annotate('', xy=(k(D['e_cost']), -0.55), xytext=(k(hi), -0.55),
                 arrowprops={'arrowstyle': '<->', 'lw': 0.6, 'color': OK['grey'], 'shrinkA': 0, 'shrinkB': 0})
    axc.text(k((hi + D['e_cost']) / 2), -0.47, f"≥ {W165.fmt_kmwh(D['mrg'][0])} below baseline", ha='center',
             va='bottom', fontsize=FONT['annot'])
    axc.text(k(f_b['e_star']), yb + 0.18, f"{W165.fmt_kmwh(lo)}–{W165.fmt_kmwh(hi)}", ha='center', va='bottom',
             fontsize=FONT['annot'])
    axc.text(k(f_c['e_star']), yc + 0.18, f"{W165.fmt_kmwh(f_c['e_star'])}", ha='center', va='bottom',
             fontsize=FONT['annot'])
    axc.set_yticks([yb, yc])
    axc.set_yticklabels([f"interval-midpoint fit and\nrange over fitting intervals\n(n = {f_b['n']})",
                         f"certified-only fit\n(n = {f_c['n']})"])
    axc.set_ylim(-0.9, 1.6)
    axc.set_xlabel(X_LABEL_C)
    axc.set_xlim(min(k(lo), k(f_c['e_star'])) - 15.0, k(D['e_cost']) + 12.0)
    return fig


def render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata=PDF_METADATA)
    return buf.getvalue()


def artist_data(ax):
    """Every drawn datum of an axes in data units: Line2D vertices (with marker and style), LineCollection segments
    (error bars), annotation arrow end points. Texts and legends excluded."""
    out = []
    for ln in ax.get_lines():
        xy = tuple(tuple(round(float(v), 9) for v in r) for r in np.asarray(ln.get_xydata()))
        out.append(('line', str(ln.get_marker()), str(ln.get_linestyle()), round(ln.get_linewidth(), 6),
                    str(ln.get_color()), xy))
    for c in ax.collections:
        if isinstance(c, LineCollection):
            segs = tuple(sorted(tuple(tuple(round(float(v), 9) for v in r) for r in s) for s in c.get_segments()))
            out.append(('segments', tuple(round(float(v), 6) for v in np.atleast_1d(c.get_linewidth())), segs))
    for t in ax.texts:
        if hasattr(t, 'xyann') and t.arrow_patch is not None:
            out.append(('arrow', tuple(round(float(v), 9) for v in t.xy), tuple(round(float(v), 9) for v in t.xyann)))
    return sorted(map(repr, out))


def clipped_text(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    undrawn = set()        # tick labels of ticks outside the view interval are never drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    undrawn.update({id(tk.label1), id(tk.label2)})
    bad = []
    for t in fig.findobj(W165.matplotlib.text.Text):
        if t.get_visible() and t.get_text() and id(t) not in undrawn:
            e = t.get_window_extent(r)
            if e.x0 < fb.x0 - 0.01 or e.y0 < fb.y0 - 0.01 or e.x1 > fb.x1 + 0.01 or e.y1 > fb.y1 + 0.01:
                bad.append(t.get_text())
    return bad


def proof_png(figs, path):
    """The LaTeX arrangement: (a) and (b) side by side (\\hfill), (c) centred below, at 300 dpi; grey frame = textwidth."""
    from PIL import Image, ImageDraw
    px = lambda pt: int(round(pt / 72.0 * 300))  # noqa: E731
    ims = {}
    for k, fig in figs.items():
        b = io.BytesIO()
        fig.savefig(b, format='png', dpi=300)
        ims[k] = Image.open(io.BytesIO(b.getvalue()))
    tw, m, gap = px(TEXTWIDTH_PT), px(6), px(14)
    H = ims['4h'].size[1] + gap + ims['cost'].size[1]
    canvas = Image.new('RGB', (tw + 2 * m, H + 2 * m), 'white')
    canvas.paste(ims['4h'], (m, m))
    canvas.paste(ims['2h'], (m + tw - ims['2h'].size[0], m))
    canvas.paste(ims['cost'], (m + (tw - ims['cost'].size[0]) // 2, m + ims['4h'].size[1] + gap))
    ImageDraw.Draw(canvas).rectangle([m - 1, 0, m + tw, canvas.size[1] - 1], outline=(200, 200, 200))
    canvas.save(path, dpi=(300, 300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    ap.add_argument('--trial', action='store_true')
    args = ap.parse_args()
    os.chdir(REPO)
    od = os.path.join(REPO, args.out_dir)
    pre = []
    if os.path.isdir(od) and [p for p in os.listdir(od) if p != '.DS_Store']:
        pre.append(f'{args.out_dir} already holds files')
    if W165._sha(W165.FZ_REL) != W165.FZ_SHA:
        pre.append('frozen tables sha256 differs from the W165 pin')
    for rel in (() if args.trial else (SCRIPT_REL,)) + (W165.SCRIPT_REL, W165.FZ_REL, W165.W145_JSON, FIG1_COMMITTED):
        if not W165.L132._committed_clean(rel):
            pre.append(f'{rel} not committed clean')
    if W165._sha(FIG1_COMMITTED) != FIG1_SHA:
        pre.append(f'{FIG1_COMMITTED} sha256 != {FIG1_SHA[:8]}')
    if pre:
        W165._log(f'[breakeven split PRECONDITION FAILED] {pre}')
        sys.exit(1)
    clone_sha = W165._sha(CLONE_FIG1) if os.path.exists(os.path.join(REPO, CLONE_FIG1)) else None
    fz = W165._jl(W165.FZ_REL)
    cells, claims = W165.all_cells(fz), W165.claims_by_id(fz)
    fd, D = W165.data_fig1(fz, cells, claims)
    checks = {'w165_fig1_checks_all_hold': (not fd.failed(), {'n': len(fd.d['checks']), 'failed': fd.failed()})}
    comp_pdf, _ = W165.render(W165.draw_fig1, fd, D)
    checks['composite_rerender_equals_committed_fig1'] = (_sha_bytes(comp_pdf) == FIG1_SHA, _sha_bytes(comp_pdf))
    checks['clone_fig1_equals_committed_fig1'] = (clone_sha == FIG1_SHA, clone_sha)
    # the composite's axes: shared limits of (a)/(b), and the reference data of every panel
    W165.set_style()
    comp = W165.draw_fig1(fd, D)
    comp.canvas.draw()
    axa, axb, axc = comp.axes[0], comp.axes[1], comp.axes[2]
    lims = {'x': axa.get_xlim(), 'y': axa.get_ylim()}
    checks['composite_ab_share_limits'] = (axb.get_ylim() == lims['y'] and axb.get_xlim() == lims['x'],
                                           {'a': [axa.get_xlim(), axa.get_ylim()], 'b': [axb.get_xlim(), axb.get_ylim()]})
    ref = {'4h': artist_data(axa), '2h': artist_data(axb), 'cost': artist_data(axc)}
    plt.close(comp)
    bad = [k for k, (ok, _d) in checks.items() if not ok]
    if bad:
        W165._log(f'[breakeven split CHECK FAILED] {bad}: {[checks[k][1] for k in bad]}')
        sys.exit(1)
    h4, h2 = D['durations']
    builders = {'4h': lambda: draw_margin(D, h4, lims, 'status'), '2h': lambda: draw_margin(D, h2, lims, 'fits'),
                'cost': lambda: draw_cost(D)}
    pdfs, figs = {}, {}
    for k, fn in builders.items():
        style()
        fig = fn()
        checks[f'{k}.plotted_data_equal_composite'] = (artist_data(fig.axes[0]) == ref[k],
                                                       {'n_artists': len(ref[k])})
        checks[f'{k}.no_clipped_text'] = (not clipped_text(fig), clipped_text(fig))
        p1 = render(fig)
        style()
        p2 = render(builders[k]())
        fr = W165.pdf_font_report(p1)
        checks[f'{k}.deterministic_truetype_no_dates'] = (
            _sha_bytes(p1) == _sha_bytes(p2) and fr['type3'] == 0 and fr['n_FontFile2'] >= 1
            and not fr['has_creation_date'] and not fr['has_mod_date'], fr)
        pdfs[k], figs[k] = p1, fig
    a_fig, b_fig = figs['4h'], figs['2h']
    checks['ab_identical_limits_and_frame'] = (
        a_fig.axes[0].get_xlim() == b_fig.axes[0].get_xlim() and a_fig.axes[0].get_ylim() == b_fig.axes[0].get_ylim()
        and tuple(a_fig.axes[0].get_position().bounds) == tuple(b_fig.axes[0].get_position().bounds)
        and a_fig.get_size_inches().tolist() == b_fig.get_size_inches().tolist(),
        {'xlim': a_fig.axes[0].get_xlim(), 'ylim': a_fig.axes[0].get_ylim()})
    guards, pk, gok = W165.guards_state()
    checks['guards_0_pickle_0'] = (gok, {'guards': guards, 'pickle': pk})
    bad = [k for k, (ok, _d) in checks.items() if not ok]
    if bad:
        W165._log(f'[breakeven split CHECK FAILED] {bad}: {[str(checks[k][1])[:300] for k in bad]}')
        sys.exit(1)
    os.makedirs(od, exist_ok=True)
    names = {'4h': 'fig1_breakeven_4h.pdf', '2h': 'fig1_breakeven_2h.pdf', 'cost': 'fig1_breakeven_cost.pdf'}
    written = []
    for k, nm in names.items():
        rel = os.path.join(args.out_dir, nm)
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(pdfs[k])
        written.append(rel)
    proof_rel = os.path.join(args.out_dir, 'proof_fig_res_breakeven.png')
    proof_png(figs, os.path.join(REPO, proof_rel))
    written.append(proof_rel)
    for f in figs.values():
        plt.close(f)
    build = {'script': SCRIPT_REL, 'w165_script': W165.SCRIPT_REL,
             'frozen_tables': {'path': W165.FZ_REL, 'sha256': W165.FZ_SHA},
             'composite': {'path': FIG1_COMMITTED, 'sha256': FIG1_SHA, 'clone_copy': CLONE_FIG1},
             'sizes_pt': {'4h': [W_AB, H_AB], '2h': [W_AB, H_AB], 'cost': [W_C, H_C]},
             'inclusion_width': {'4h': '0.48\\textwidth', '2h': '0.48\\textwidth', 'cost': '0.70\\textwidth'},
             'textwidth_pt': TEXTWIDTH_PT, 'fonts_pt': FONT, 'shared_limits_ab': lims,
             'pdf_sha256': {names[k]: _sha_bytes(pdfs[k]) for k in names},
             'values': {'fits': {fk: {kk: f[kk] for kk in ('n', 'a', 'b', 'c', 'e_star')} for fk, f in D['fits'].items()},
                        'breakeven_range': D['rng'], 'difference_to_baseline_range': D['mrg'], 'e_cost': D['e_cost'],
                        'p_cost': D['p_cost'], 'two_tau': D['two_tau'],
                        'points': [{kk: p[kk] for kk in ('label', 'cell', 'E', 'P', 'h', 'y', 'kind', 'bar', 'hw',
                                                          'band', 'status_T2')} for p in D['points']]},
             'checks': {k: {'ok': ok, 'detail': d} for k, (ok, d) in checks.items()}}
    build_rel = os.path.join(args.out_dir, 'build_record.json')
    with open(os.path.join(REPO, build_rel), 'x', encoding='utf-8') as h:
        h.write(W165.GRIO.dumps(W165._norm(build), indent=1, sort_keys=True) + '\n')
    written.append(build_rel)
    man = {rel: W165._sha(rel) for rel in sorted(written)}
    with open(os.path.join(od, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        h.write(W165.GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for rel in sorted(written):
        W165._log(f'[breakeven split] wrote {rel} {man[rel][:8]}')
    W165._log(f'[breakeven split] all {len(checks)} checks held; guards 0; pickle 0')
    sys.exit(0)


if __name__ == '__main__':
    main()
