"""P5.15 S55 (presentation pass): Figure fig:res_flex_ladder (fig2_flexibility_ladder.pdf) redrawn, wording and style only.

ZERO SOLVES. Imports the W165 figure script (p515_s53_w165_step6_figures.py, unedited): its guards are armed and pickle
is blocked at import; all are verified at 0 before writing.

DATA. W165.data_fig2 is called exactly as W165.main calls it (frozen tables frozen_step6_tables_v1_590088fe.json:
claims CHECK:headline_V_minus_I_settled, H:m1.5, H:m2 and A64 row H:m1.75; multipliers from the campaign specs). Every
W165 figure-2 check must hold, and the composite re-rendered from these data must equal the committed
fig2_flexibility_ladder.pdf (7d15f697..., the Overleaf file) bitwise before anything is drawn.

CHANGES (Expert, 2026-11-16):
  - y label 'value − I of the unit (k€)' -> 'Investment margin (k€)';
  - x label 'flexibility-price multiplier m (× base price)' -> 'Flexibility-price multiplier (× baseline price)'
    (capitalised like the y label; the symbol m is not used in the text);
  - legend reduced to two entries, 'evaluation meeting the acceptance criteria' (filled circle) and 'evaluation not
    meeting the acceptance criteria' (open square), each sample showing its bar; the bars are the comparison thresholds
    (the caption says so); the legend sits in the empty lower-right corner. Colour no longer encodes the determinacy verdict, which the legend no longer explains: all
    points are drawn in one colour; the verdict is read from whether the bar crosses zero;
  - font family and printed sizes common to the SoC, SoH and flexibility figures: DejaVu Serif; axis labels 8 pt,
    tick labels 7.5 pt, legend and value labels 7 pt. The figure is drawn at its inclusion width (0.75 \\textwidth =
    292.5 pt, \\textwidth = 390 pt), so these are the printed sizes.

CHECK OF THE PLOTTED VALUES. Marker positions and error-bar segments (data units) of the new figure are compared
exactly with the composite's; colour and marker style are excluded from the comparison (they are the change).

Usage: <canonical python> p515_s55_fig_flex_ladder.py --out-dir DIR [--trial]
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
from matplotlib.collections import LineCollection  # noqa: E402

SCRIPT_REL = os.path.basename(__file__)
FIG2_COMMITTED = os.path.join(W165.OUT_DIR, 'fig2_flexibility_ladder.pdf')
FIG2_SHA = '7d15f6970e26db9d'          # prefix; the full hash is checked against the committed file below
CLONE_FIG2 = os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'fig2_flexibility_ladder.pdf')
TEXTWIDTH_PT = 390.0
W_PT = 0.75 * TEXTWIDTH_PT
H_PT = W_PT * 62.0 / 90.0              # the composite's aspect (90 x 62 mm)
AXES_PT = (44.0, 30.0, W_PT - 44.0 - 10.0, H_PT - 30.0 - 6.0)
FONT_FAMILY = 'DejaVu Serif'           # the family of the SoC and SoH figures
FONT = {'tick': 7.5, 'label': 8.0, 'legend': 7.0, 'annot': 7.0}
PDF_METADATA = {'Creator': SCRIPT_REL, 'Producer': 'Matplotlib pdf backend', 'CreationDate': None, 'ModDate': None}
COLOR = OK['blue']
X_LABEL = 'Flexibility-price multiplier (× baseline price)'
Y_LABEL = 'Investment margin (k€)'
LEG = {False: 'evaluation meeting the acceptance criteria', True: 'evaluation not meeting the acceptance criteria'}


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def style():
    W165.set_style()
    W165.matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': [FONT_FAMILY], 'mathtext.fontset': 'dejavuserif',
        'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'], 'axes.labelsize': FONT['label'],
        'legend.fontsize': FONT['legend']})


def draw(D):
    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    x, y, aw, ah = AXES_PT
    ax = fig.add_axes([x / W_PT, y / H_PT, aw / W_PT, ah / H_PT])
    ax.axhline(0.0, color=OK['grey'], lw=0.6, zorder=1)
    cont = {}
    for r in D['rows']:
        unc = r['uncertified_form']
        c = ax.errorbar([r['m']], [W165._k(r['y'])], yerr=[[W165._k(r['thr'])], [W165._k(r['thr'])]],
                        fmt='s' if unc else 'o', mfc='white' if unc else COLOR, color=COLOR, ms=3.8, elinewidth=1.0,
                        capsize=2.0, zorder=3)
        cont.setdefault(unc, c)
        ax.annotate(f"{W165.fmt_k(r['y'])}", (r['m'], W165._k(r['y'])), xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=FONT['annot'])
    ax.set_xticks([r['m'] for r in D['rows']])
    ax.set_xticklabels([f"{r['m']:g}" for r in D['rows']])
    ax.set_xlim(min(r['m'] for r in D['rows']) - 0.12, max(r['m'] for r in D['rows']) + 0.25)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    W165.kfmt(ax, 'y')
    keys = sorted(cont)                 # False (meeting) first
    ax.legend(handles=[cont[k] for k in keys], labels=[LEG[k] for k in keys], loc='lower right', handlelength=1.4,
              borderaxespad=0.2, fontsize=FONT['legend'])
    return fig


def render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata=PDF_METADATA)
    return buf.getvalue()


def positions(ax):
    """Marker positions and error-bar segments in data units (no colour or style: those are the change)."""
    out = []
    for ln in ax.get_lines():
        out.append(('line', tuple(tuple(round(float(v), 9) for v in r) for r in np.asarray(ln.get_xydata()))))
    for c in ax.collections:
        if isinstance(c, LineCollection):
            out.append(('segments', tuple(sorted(tuple(tuple(round(float(v), 9) for v in r) for r in s)
                                                 for s in c.get_segments()))))
    return sorted(map(repr, out))


def clipped_text(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    undrawn = set()
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


def fonts_used(pdf):
    import re
    return sorted(set(m.decode() for m in re.findall(rb'/BaseFont\s*/([A-Za-z0-9+\-]+)', pdf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--trial', action='store_true')
    args = ap.parse_args()
    os.chdir(REPO)
    od = os.path.join(REPO, args.out_dir)
    pre = []
    if os.path.isdir(od) and [p for p in os.listdir(od) if p != '.DS_Store']:
        pre.append(f'{args.out_dir} already holds files')
    if W165._sha(W165.FZ_REL) != W165.FZ_SHA:
        pre.append('frozen tables sha256 differs from the W165 pin')
    for rel in (() if args.trial else (SCRIPT_REL,)) + (W165.SCRIPT_REL, W165.FZ_REL, FIG2_COMMITTED):
        if not W165.L132._committed_clean(rel):
            pre.append(f'{rel} not committed clean')
    fig2_sha = W165._sha(FIG2_COMMITTED)
    if not fig2_sha.startswith(FIG2_SHA):
        pre.append(f'{FIG2_COMMITTED} sha256 {fig2_sha[:16]} != {FIG2_SHA}')
    if pre:
        W165._log(f'[flex ladder PRECONDITION FAILED] {pre}')
        sys.exit(1)
    clone_sha = W165._sha(CLONE_FIG2) if os.path.exists(os.path.join(REPO, CLONE_FIG2)) else None
    fz = W165._jl(W165.FZ_REL)
    cells, claims = W165.all_cells(fz), W165.claims_by_id(fz)
    fd, D = W165.data_fig2(fz, cells, claims)
    checks = {'w165_fig2_checks_all_hold': (not fd.failed(), {'n': len(fd.d['checks']), 'failed': fd.failed()})}
    comp_pdf, _ = W165.render(W165.draw_fig2, fd, D)
    checks['composite_rerender_equals_committed_fig2'] = (_sha_bytes(comp_pdf) == fig2_sha, _sha_bytes(comp_pdf))
    checks['clone_fig2_equals_committed_fig2'] = (clone_sha == fig2_sha, clone_sha)
    W165.set_style()
    comp = W165.draw_fig2(fd, D)
    comp.canvas.draw()
    ref = positions(comp.axes[0])
    comp_ylim = comp.axes[0].get_ylim()
    plt.close(comp)
    style()
    fig = draw(D)
    checks['plotted_positions_equal_composite'] = (positions(fig.axes[0]) == ref, {'n': len(ref)})
    checks['no_clipped_text'] = (not clipped_text(fig), clipped_text(fig))
    p1 = render(fig)
    style()
    p2 = render(draw(D))
    fr = W165.pdf_font_report(p1)
    fu = fonts_used(p1)
    checks['deterministic_truetype_no_dates'] = (
        _sha_bytes(p1) == _sha_bytes(p2) and fr['type3'] == 0 and fr['n_FontFile2'] >= 1
        and not fr['has_creation_date'] and not fr['has_mod_date'], fr)
    checks['only_dejavu_serif_fonts'] = (bool(fu) and all('DejaVuSerif' in f for f in fu), fu)
    guards, pk, gok = W165.guards_state()
    checks['guards_0_pickle_0'] = (gok, {'guards': guards, 'pickle': pk})
    bad = [k for k, (ok, _d) in checks.items() if not ok]
    if bad:
        W165._log(f'[flex ladder CHECK FAILED] {bad}: {[checks[k][1] for k in bad]}')
        sys.exit(1)
    os.makedirs(od, exist_ok=True)
    rel = os.path.join(args.out_dir, 'fig2_flexibility_ladder.pdf')
    with open(os.path.join(REPO, rel), 'xb') as h:
        h.write(p1)
    fig.savefig(os.path.join(od, 'preview_fig2_0.75textwidth.png'), dpi=300)
    build = {'script': SCRIPT_REL, 'w165_script_sha256': W165._sha(W165.SCRIPT_REL),
             'frozen_tables': {'path': W165.FZ_REL, 'sha256': W165.FZ_SHA},
             'committed_fig2': {'path': FIG2_COMMITTED, 'sha256': fig2_sha}, 'clone_fig2_sha256': clone_sha,
             'page_pt': [W_PT, H_PT], 'axes_pt': list(AXES_PT), 'font_family': FONT_FAMILY, 'font_pt': FONT,
             'composite_ylim': list(comp_ylim), 'new_ylim': list(fig.axes[0].get_ylim()),
             'rows': [{k: r[k] for k in ('claim', 'm', 'y', 'thr', 'verdict', 'uncertified_form')} for r in D['rows']],
             'checks': {k: {'ok': ok, 'detail': d} for k, (ok, d) in checks.items()}}
    brel = os.path.join(args.out_dir, 'build_record.json')
    with open(os.path.join(REPO, brel), 'x', encoding='utf-8') as h:
        h.write(W165.GRIO.dumps(W165._norm(build), indent=1, sort_keys=True) + '\n')
    W165._log(f'[flex ladder] wrote {rel} {_sha_bytes(p1)[:8]}; all {len(checks)} checks held; guards 0; pickle 0')
    sys.exit(0)


if __name__ == '__main__':
    main()
