"""P5.15 S55 (Addendum 92, presentation pass): Figure fig:res_q_cycle re-exported as three separate panel PDFs.

ZERO SOLVES. Imports the W165 figure script (p515_s53_w165_step6_figures.py, unedited), which arms its own
SolveProfileGuard(permitted=()) and the W157 guards and blocks pickle.load/loads at import; every guard is verified at 0
and the pickle counters at 0 before any output is written.

DATA PATH. The panel data come from W165.data_fig4 called exactly as W165.main calls it (frozen tables
frozen_step6_tables_v1_590088fe.json, sha256 pinned in W165; the panel rule; the committed per-cycle records). All 41
W165 figure-4 checks must hold. Before any panel is drawn, W165.draw_fig4 is rendered from these data and its PDF sha256
must equal the committed fig4_q_versus_cycle.pdf (45755c82..., the file in the Overleaf clone): the three panels are
therefore drawn from the data of the current figure, not from a re-derivation.

PANELS. Each panel is drawn by the same calls as in W165.draw_fig4 (same limits, labels, tick format, colours,
markers, turning points, refusal ticks, shading), each in its own figure. The changes, recorded in
the build record:
  all  one size, PANEL_W_MM x PANEL_H_MM (95 x 70 mm; v2 kept the composite sizes 95 x 112 / 56 / 56 mm) and y tick
       labels at two decimals (the composite's formatter prints '5', '2.5'), both at the author's request (2026-11-13);
       no panel letters (the composite prints (a)-(c) top left; the letters come from LaTeX), author, 2026-11-13 (v4);
  (b) shows its own x tick labels and the x label 'ADMM cycle' (in the composite they are hidden: x shared with (c));
      its x limits are set explicitly to (k0 - 1, cap + 1), the limits it inherited from (c);
  (c) v5 (Expert, decision 1 (a), 2026-11-13): no gap-refusal ticks and no legend (v2-v4 drew 59 vermillion ticks with
      the legend 'gap-clause refusals').
  v6 (author, 2026-11-13): (b) and (c) share one axes rectangle (the same figure fractions in two 95 x 70 mm files) and
      the same x ticks, so the x axes coincide when stacked; the x label 'ADMM cycle' on (c) only ((b) keeps its x tick
      labels). The rectangle is the intersection of the two panels' constrained-layout axes; the ticks are (c)'s.
  v5: panel (a) is adopted as is -- the v4 file (sha256 A_ADOPTED_SHA) is copied byte for byte after asserting that a
      fresh render of (a) reproduces it bitwise; (b) and (c) are re-exported.
  v7 (Expert, 2026-11-13): y-axis labels in words, not symbols or record field names -- (a), (b) 'Objective change (k€)'
      (were 'Q − Q(k*) (k€)', 'Q − Q(cap) (k€)'), (c) 'Interface-consensus gap (k€)' (was '$t_{sum}$ (k€)'); the x
      label 'ADMM cycle' already in words. All three panels re-exported (so (a) is no longer the v4 copy). Before
      relabelling, the predecessor labels are rendered and must reproduce v4 (a) and v6 (b), (c) bitwise, so the label
      is the only change.

OUTPUTS (OUT_DIR, which must not exist or be empty): fig_q_cycle_a.pdf, fig_q_cycle_b.pdf, fig_q_cycle_c.pdf (each
rendered twice, sha256 asserted equal; TrueType fonts, no PDF dates), NOTE.md (the record each panel reads),
build_record.json, manifest_sha256.json. 200-dpi preview PNGs go to --preview-dir if given (not deliverables).

Usage: <canonical python> p515_s55_fig_q_cycle_split.py [--out-dir DIR] [--preview-dir DIR] [--trial]
Exit: 0 = written and every check held; 1 = a precondition, check or guard failed (no PDF written).
"""
import argparse
import hashlib
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s53_w165_step6_figures as W165  # noqa: E402 -- arms the guards and blocks pickle at import; not edited

plt = W165.plt
blended_transform_factory = W165.blended_transform_factory
OKABE_ITO = W165.OKABE_ITO
MM = W165.MM

SCRIPT_REL = os.path.basename(__file__)
OUT_DIR_DEFAULT = os.path.join('Claude outputs', 'fig_q_cycle_v7')
YL_PRED = {'a': 'Q − Q(k*) (k€)', 'b': 'Q − Q(cap) (k€)', 'c': '$t_{sum}$ (k€)'}        # v4 (a), v6 (b), (c)
YL_V7 = {'a': 'Objective change (k€)', 'b': 'Objective change (k€)', 'c': 'Interface-consensus gap (k€)'}
YL = dict(YL_PRED)                                                                      # set by main()
PRED_SHA = {'a': ('Claude outputs/fig_q_cycle_v4/fig_q_cycle_a.pdf',
                  '6bfb74a1abff991db4064935fd8b198d88b3ba27c13e9ca67ffe323e760a86a3'),
            'b': ('Claude outputs/fig_q_cycle_v6/fig_q_cycle_b.pdf',
                  '65ec70139332477f4c0469a2e45e2c63e220f63dbc3f99435dd937356e96c549'),
            'c': ('Claude outputs/fig_q_cycle_v6/fig_q_cycle_c.pdf',
                  'cbad023bbf0e001373a9debdbc67d74cdb82757668fde8b2acc91ed38f5cc8ff')}
A_ADOPTED_REL = os.path.join('Claude outputs', 'fig_q_cycle_v4', 'fig_q_cycle_a.pdf')   # Expert: adopted as is (v5)
A_ADOPTED_SHA = '6bfb74a1abff991db4064935fd8b198d88b3ba27c13e9ca67ffe323e760a86a3'
V4_B_SHA = 'f5c8cab9ff86cbb73ae1fa64df27d1160eef3b340980ec6cd17f0a7615edf551'
FIG4_COMMITTED = os.path.join(W165.OUT_DIR, 'fig4_q_versus_cycle.pdf')
FIG4_SHA = '45755c82927e04c75bb4d7dcbbf816347632b83eadaea42de631d20d5a3a13cf'
CLONE_FIG4 = os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'fig4_q_versus_cycle.pdf')
PANEL_W_MM = 95.0
PANEL_H_MM = {'a': 70.0, 'b': 70.0, 'c': 70.0}   # one size for all three (author, 2026-11-13)
PDF_METADATA = {'Creator': SCRIPT_REL, 'Producer': 'Matplotlib pdf backend', 'CreationDate': None, 'ModDate': None}


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def yfmt2(ax):
    """y tick labels at two decimals, thousands separator, typographic minus (author, 2026-11-13)."""
    ax.yaxis.set_major_formatter(W165.matplotlib.ticker.FuncFormatter(lambda v, _p=None: f'{v:,.2f}'.replace('-', '−')))


def _fig(panel):
    return plt.figure(figsize=(PANEL_W_MM * MM, PANEL_H_MM[panel] * MM), layout='constrained')


def _fig_ax(panel, frame):
    """(b), (c): with frame None, constrained layout (the measuring pass); else the shared fixed rectangle (v6)."""
    if frame is None:
        fig = _fig(panel)
        return fig, fig.add_subplot(1, 1, 1)
    fig = plt.figure(figsize=(PANEL_W_MM * MM, PANEL_H_MM[panel] * MM))
    return fig, fig.add_axes(frame['rect'])


def _apply_x(ax, b, frame, label):
    ax.set_xlim(b['k0'] - 1, b['cap'] + 1)
    if frame is not None:
        ax.set_xticks(frame['xticks'])
    if label:
        ax.set_xlabel('ADMM cycle')


def bc_frame(D):
    """The shared axes rectangle of (b) and (c): the intersection of their constrained-layout axes; (c)'s x ticks."""
    W165.set_style()
    pos = {}
    for p, fn in (('b', draw_b), ('c', draw_c)):
        fig = fn(D)
        fig.canvas.draw()
        ax = fig.axes[0]
        pos[p] = ax.get_position().bounds
        if p == 'c':
            lo, hi = ax.get_xlim()
            xt = [float(t) for t in ax.get_xticks() if lo <= t <= hi]
        plt.close(fig)
    left = max(pos['b'][0], pos['c'][0])
    bottom = max(pos['b'][1], pos['c'][1])
    right = min(pos['b'][0] + pos['b'][2], pos['c'][0] + pos['c'][2])
    top = min(pos['b'][1] + pos['b'][3], pos['c'][1] + pos['c'][3])
    return {'rect': [left, bottom, right - left, top - bottom], 'xticks': xt, 'measured': pos}


# ---------------------------------------------------------------------------------------------------------------------
#  the three panels: the calls of W165.draw_fig4, one panel per figure
# ---------------------------------------------------------------------------------------------------------------------
def draw_a(D):
    fig = _fig('a')
    axa = fig.add_subplot(1, 1, 1)
    a = D['a']
    ks = list(range(a['k0'], a['kstar'] + 1))
    ya = [W165._k(a['q'][k] - a['Q_ref']) for k in ks]
    lo, hi = a['window']
    axa.axvspan(lo - 0.5, hi + 0.5, color=OKABE_ITO['sky'], alpha=0.18, lw=0, zorder=0)
    axa.plot(ks, ya, '-', color=OKABE_ITO['black'], lw=0.8, zorder=3)
    axa.plot(ks, ya, '.', color=OKABE_ITO['black'], ms=2.0, zorder=3)
    axa.hlines([W165._k(a['band'][0] - a['Q_ref']), W165._k(a['band'][1] - a['Q_ref'])], lo - 0.5, hi + 0.5,
               colors=OKABE_ITO['blue'], linestyles='--', lw=0.7, zorder=2)
    axa.plot([t for t, _kd, _q in a['T']], [W165._k(qt - a['Q_ref']) for _t, _kd, qt in a['T']], 'o', mfc='white',
             mec=OKABE_ITO['vermillion'], mew=1.0, ms=4.5, zorder=4, label='turning points')
    W165._vline(axa, a['k0'], '$k_0$', OKABE_ITO['green'])
    W165._vline(axa, a['N'], 'N', OKABE_ITO['grey'], ls=':')
    W165._vline(axa, a['kstar'], 'k*', OKABE_ITO['vermillion'], ha='right')
    axa.text((lo + hi) / 2, 0.06, f"window ({hi - lo + 1} cycles)\nrange/τ = {a['range_over_tau']:.2f}",
             transform=blended_transform_factory(axa.transData, axa.transAxes), ha='center', va='bottom',
             fontsize=6.0, color=OKABE_ITO['blue'])
    axa.set_xlim(a['k0'] - 1, a['kstar'] + 1)
    axa.set_xlabel('ADMM cycle')
    axa.set_ylabel(YL['a'])
    yfmt2(axa)
    axa.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), handlelength=1.0, borderaxespad=0.2)
    # no panel letter (author, 2026-11-13): the composite's W165.panel_label(axa, '(a)') is not drawn
    return fig


def draw_b(D, frame=None):
    fig, axb = _fig_ax('b', frame)
    b = D['b']
    kb = list(range(b['k0'], b['cap'] + 1))
    qref = b['q'][b['cap']]
    yb = [W165._k(b['q'][k] - qref) for k in kb]
    blo, bhi = b['band_window']
    axb.axvspan(blo - 0.5, bhi + 0.5, color=OKABE_ITO['grey'], alpha=0.15, lw=0, zorder=0)
    axb.plot(kb, yb, '-', color=OKABE_ITO['black'], lw=0.8, zorder=3)
    axb.plot(kb, yb, '.', color=OKABE_ITO['black'], ms=2.0, zorder=3)
    axb.plot([t for t, _kd, _q in b['T']], [W165._k(qt - qref) for _t, _kd, qt in b['T']], 'o', mfc='white',
             mec=OKABE_ITO['vermillion'], mew=1.0, ms=4.5, zorder=4)
    W165._vline(axb, b['k0'], '$k_0$', OKABE_ITO['green'])
    W165._vline(axb, b['cap'], 'cap', OKABE_ITO['black'], ha='right')
    axb.text((blo + bhi) / 2, 0.80, f'last {bhi - blo + 1} cycles at the cap', ha='center', va='bottom',
             transform=blended_transform_factory(axb.transData, axb.transAxes), fontsize=6.0,
             color=OKABE_ITO['grey'])
    axb.set_ylabel(YL['b'])
    yfmt2(axb)
    # separation: the composite hides these (x shared with (c)); a standalone panel needs its own x axis
    _apply_x(axb, b, frame, label=False)     # v6: x label on (c) only; (b) keeps its x tick labels
    # no panel letter (author, 2026-11-13): the composite's W165.panel_label(axb, '(b)') is not drawn
    return fig


def draw_c(D, frame=None):
    fig, axc = _fig_ax('c', frame)
    b = D['b']
    kb = list(range(b['k0'], b['cap'] + 1))
    tc = [W165._k(b['t'][k]) for k in kb]
    gb = W165._k(b['gap_bound'])
    axc.axhspan(-gb, gb, color=OKABE_ITO['green'], alpha=0.12, lw=0, zorder=0)
    axc.axhline(gb, color=OKABE_ITO['green'], lw=0.6, ls='--')
    axc.axhline(-gb, color=OKABE_ITO['green'], lw=0.6, ls='--')
    axc.axhline(0.0, color=OKABE_ITO['grey'], lw=0.5)
    axc.plot(kb, tc, '-', color=OKABE_ITO['black'], lw=0.8, zorder=3)
    axc.plot(kb, tc, '.', color=OKABE_ITO['black'], ms=2.0, zorder=3)
    # v5 (Expert, 2026-11-13): the gap-refusal ticks and their legend entry are not drawn
    axc.text(b['cap'] - 0.6, gb, '±τ/2', ha='right', va='bottom', fontsize=6.0, color=OKABE_ITO['green'])
    axc.axvline(b['k0'], color=OKABE_ITO['green'], lw=0.7)
    axc.axvline(b['cap'], color=OKABE_ITO['black'], lw=0.7)
    _apply_x(axc, b, frame, label=True)
    axc.set_ylabel(YL['c'])
    yfmt2(axc)
    # no panel letter (author, 2026-11-13): the composite's W165.panel_label(axc, '(c)') is not drawn
    return fig


def render(draw, D):
    W165.set_style()
    fig = draw(D)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata=PDF_METADATA)
    png = io.BytesIO()
    fig.savefig(png, format='png', dpi=W165.PREVIEW_DPI, metadata={'Software': None})
    plt.close(fig)
    return buf.getvalue(), png.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    ap.add_argument('--preview-dir', default=None)
    ap.add_argument('--trial', action='store_true', help='trial run (scratch out-dir): the script need not be committed')
    args = ap.parse_args()
    os.chdir(REPO)
    out_dir = args.out_dir
    od = os.path.join(REPO, out_dir)
    pre = []
    if os.path.isdir(od) and [p for p in os.listdir(od) if p != '.DS_Store']:
        pre.append(f'{out_dir} already holds files {sorted(os.listdir(od))}')
    if W165._sha(W165.FZ_REL) != W165.FZ_SHA:
        pre.append('frozen tables sha256 differs from the W165 pin')
    for rel in ((() if args.trial else (SCRIPT_REL,)) + (W165.SCRIPT_REL, W165.FZ_REL, W165.T2_CSV, FIG4_COMMITTED)):
        if not W165.L132._committed_clean(rel):
            pre.append(f'{rel} not committed clean')
    if W165._sha(FIG4_COMMITTED) != FIG4_SHA:
        pre.append(f'{FIG4_COMMITTED} sha256 != {FIG4_SHA[:8]}')
    for _p, (prel, psha) in PRED_SHA.items():
        if not W165.L132._committed_clean(prel) or W165._sha(prel) != psha:
            pre.append(f'{prel} not committed clean or sha256 != {psha[:8]}')
    clone_sha = W165._sha(CLONE_FIG4) if os.path.exists(os.path.join(REPO, CLONE_FIG4)) else None
    if pre:
        W165._log(f'[split PRECONDITION FAILED] {pre}')
        sys.exit(1)
    fz = W165._jl(W165.FZ_REL)
    cells = W165.all_cells(fz)
    chosen, rule_info = W165.panel_rule(fz)
    fd, D = W165.data_fig4(fz, cells, chosen, rule_info)
    failed = fd.failed()
    checks = {'w165_fig4_checks_all_hold': (not failed and D is not None, {'n': len(fd.d['checks']), 'failed': failed})}
    if D is None or failed:
        W165._log(f'[split CHECK FAILED] W165 fig4 checks {failed}')
        sys.exit(1)
    # records read by the composite: every one committed clean
    rec = {'a': fd.d['records_a'], 'b': fd.d['records_b']}
    rec_paths = [rec['a'][k]['path'] for k in ('per_cycle_record', 'settling_decision', 'w101_summary',
                                               'continuation_record')] + \
                [rec['b'][k]['path'] for k in ('resettle_cycle_record', 'per_cycle_record', 'resettle_decision')]
    unclean = [p for p in rec_paths if not W165.L132._committed_clean(p)]
    checks['records_committed_clean'] = (not unclean, unclean)
    # the data reproduce the current figure bitwise
    comp_pdf, _ = W165.render(W165.draw_fig4, fd, D)
    checks['composite_rerender_equals_committed_fig4'] = (_sha_bytes(comp_pdf) == FIG4_SHA, _sha_bytes(comp_pdf))
    checks['clone_fig4_equals_committed_fig4'] = (clone_sha == FIG4_SHA, clone_sha)
    bad = [k for k, (ok, _d) in checks.items() if not ok]
    if bad:
        W165._log(f'[split CHECK FAILED] {bad}: {[checks[k][1] for k in bad]}')
        sys.exit(1)
    # predecessor labels first: must reproduce v4 (a) and v6 (b), (c) bitwise -- the label is then the only change
    YL.update(YL_PRED)
    fpred = bc_frame(D)
    for p, fn in (('a', draw_a), ('b', lambda d: draw_b(d, fpred)), ('c', lambda d: draw_c(d, fpred))):
        pp, _ = render(fn, D)
        checks[f'{p}.predecessor_labels_reproduce_{os.path.basename(os.path.dirname(PRED_SHA[p][0]))}'] = (
            _sha_bytes(pp) == PRED_SHA[p][1], _sha_bytes(pp))
    YL.update(YL_V7)
    panels = {}
    frame = bc_frame(D)
    checks['bc.frame_positive'] = (frame['rect'][2] > 0 and frame['rect'][3] > 0 and len(frame['xticks']) >= 2, frame)
    for p, fn in (('a', draw_a), ('b', lambda d: draw_b(d, frame)), ('c', lambda d: draw_c(d, frame))):
        pdf1, png1 = render(fn, D)
        pdf2, _ = render(fn, D)
        fr = W165.pdf_font_report(pdf1)
        ok = (_sha_bytes(pdf1) == _sha_bytes(pdf2) and fr['type3'] == 0 and fr['n_FontFile2'] >= 1
              and not fr['has_creation_date'] and not fr['has_mod_date'])
        checks[f'{p}.deterministic_truetype_no_dates'] = (ok, fr)
        panels[p] = {'pdf': pdf1, 'png': png1, 'sha256': _sha_bytes(pdf1), 'fonts': fr}
    guards, pk, gok = W165.guards_state()
    checks['guards_0_pickle_0'] = (gok, {'guards': guards, 'pickle': pk})
    bad = [k for k, (ok, _d) in checks.items() if not ok]
    if bad:
        W165._log(f'[split CHECK FAILED] {bad}')
        sys.exit(1)
    os.makedirs(od, exist_ok=True)
    written = []
    for p in 'abc':
        rel = os.path.join(out_dir, f'fig_q_cycle_{p}.pdf')
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(panels[p]['pdf'])
        written.append(rel)
        if args.preview_dir:
            os.makedirs(args.preview_dir, exist_ok=True)
            with open(os.path.join(args.preview_dir, f'fig_q_cycle_{p}_preview.png'), 'wb') as h:
                h.write(panels[p]['png'])
    a, b = D['a'], D['b']
    build = {
        'script': SCRIPT_REL, 'w165_script': W165.SCRIPT_REL, 'frozen_tables': {'path': W165.FZ_REL,
                                                                             'sha256': W165.FZ_SHA},
        'composite': {'path': FIG4_COMMITTED, 'sha256': FIG4_SHA, 'clone_copy': CLONE_FIG4,
                      'rerender_sha256': _sha_bytes(comp_pdf)},
        'panel_size_mm': {p: [PANEL_W_MM, PANEL_H_MM[p]] for p in 'abc'},
        'changes_all_panels': ['one size for all three panels (author, 2026-11-13)',
                               'y tick labels at two decimals (author, 2026-11-13)',
                               'no panel letters; labelled in LaTeX (author, 2026-11-13)'],
        'v7': {'y_labels': YL_V7, 'replaced': YL_PRED, 'x_label': 'ADMM cycle (unchanged, in words)',
               'predecessors': {p: {'path': v[0], 'sha256': v[1]} for p, v in PRED_SHA.items()},
               'author': 'Expert, 2026-11-13'},
        'v6': {'bc_axes_rect_figure_fraction': frame['rect'], 'bc_xticks': frame['xticks'],
               'measured_constrained_positions': frame['measured'],
               'b': 'x label removed (on (c) only); x tick labels kept', 'author': '2026-11-13'},
        'v5': {'a': f'adopted as is: copy of {A_ADOPTED_REL} (sha256 {A_ADOPTED_SHA}); a fresh render reproduces it',
               'b': 'unchanged in drawing from v4 (no legend, no ticks)',
               'c': 'gap-refusal ticks and their legend entry removed (Expert, 2026-11-13)',
               'legend_text_remaining': {'a': ['turning points'], 'b': [], 'c': []}},
        'separation_changes': {'a': [], 'b': ['x tick labels shown and x label "ADMM cycle" (hidden in the composite, '
                                               'x shared with (c))', 'x limits set to (k0 - 1, cap + 1) explicitly '
                                               '(inherited from (c) in the composite)'], 'c': []},
        'panels': {p: {'pdf': os.path.join(out_dir, f'fig_q_cycle_{p}.pdf'), 'sha256': panels[p]['sha256'],
                       'fonts': panels[p]['fonts']} for p in 'abc'},
        'records': rec, 'panel_rule_chosen': chosen,
        'drawn': {'a': {'cell': a['cell'], 'cycles': [a['k0'], a['kstar']], 'k0': a['k0'], 'N': a['N'],
                        'k_star': a['kstar'], 'window': a['window'], 'turning_points': [t for t, _k, _q in a['T']],
                        'Q_ref_k_star': a['Q_ref']},
                  'b': {'cell': b['cell'], 'cycles': [b['k0'], b['cap']], 'k0': b['k0'], 'cap': b['cap'],
                        'band_window': b['band_window'], 'turning_points': [t for t, _k, _q in b['T']],
                        'Q_ref_cap': b['q'][b['cap']]},
                  'c': {'cell': b['cell'], 'cycles': [b['k0'], b['cap']], 'gap_bound': b['gap_bound'],
                        'n_gap_refusal_ticks': 0}},
        'checks': {k: {'ok': ok, 'detail': d} for k, (ok, d) in checks.items()},
    }
    build_rel = os.path.join(out_dir, 'build_record.json')
    with open(os.path.join(REPO, build_rel), 'x', encoding='utf-8') as h:
        h.write(W165.GRIO.dumps(W165._norm(build), indent=1, sort_keys=True) + '\n')
    written.append(build_rel)

    def r(x):
        return f"`{x['path']}` (sha256 `{x['sha256'][:8]}`)"
    note = f"""# Figure fig:res_q_cycle as three subfigure files, v7 (zero solves)

The manuscript uses these three files as subfigures (Expert, decision 1 option (a), 2026-11-13). v7 supersedes v6:
the y-axis labels are now words, not symbols or record field names (Expert, 2026-11-13), and nothing else changed.

| panel | y label, v7 | y label before |
|---|---|---|
| (a) | {YL_V7['a']} | {YL_PRED['a']} (v4) |
| (b) | {YL_V7['b']} | {YL_PRED['b']} (v6) |
| (c) | {YL_V7['c']} | t_sum (k€) (v6) |

- **x label:** "ADMM cycle", already in words, on (a) and (c) only.
- **Only the labels changed.** Before relabelling, the script rendered the panels with their old labels and
  reproduced the predecessors bitwise:
  - (a): `{PRED_SHA['a'][0]}` (`{PRED_SHA['a'][1][:8]}`);
  - (b): `{PRED_SHA['b'][0]}` (`{PRED_SHA['b'][1][:8]}`);
  - (c): `{PRED_SHA['c'][0]}` (`{PRED_SHA['c'][1][:8]}`).
- **Other in-plot text, kept as before:**
  - markers $k_0$, N and k* in (a), and $k_0$ and "cap" in (b) and (c);
  - the shading notes;
  - ±τ/2;
  - the (a) legend "turning points".

All three are {PANEL_W_MM:g} × {PANEL_H_MM['a']:g} mm with two-decimal y ticks and no panel letters (LaTeX labels
them).

## For W190: the three subfigure files

| file | sha256 | panel |
|---|---|---|
| `fig_q_cycle_a.pdf` | `{panels['a']['sha256']}` | (a) objective of the plan without storage (x = 0), relative to certification |
| `fig_q_cycle_b.pdf` | `{panels['b']['sha256']}` | (b) objective of the reference unit at 1.5×, relative to the cap |
| `fig_q_cycle_c.pdf` | `{panels['c']['sha256']}` | (c) priced interface-consensus gap of the same evaluation, with ±τ/2 |

- **(a)** is v4's panel (a) (the version adopted in v5) with the new y label.
- **(b) and (c)** share one axes rectangle and one set of x ticks, so stacked at the same width their x axes coincide:
  - rectangle (figure fractions): left {frame['rect'][0]:.4f}, bottom {frame['rect'][1]:.4f}, width {frame['rect'][2]:.4f},
    height {frame['rect'][3]:.4f};
  - x range {b['k0'] - 1}–{b['cap'] + 1};
  - x ticks {', '.join(f'{t:g}' for t in frame['xticks'])}.
- **(b)** keeps its x tick labels but drops the x label; "ADMM cycle" is on (c) only.
- **(c)** has no gap-refusal ticks and no legend. v2–v4 drew 59 vermillion ticks labelled "gap-clause refusals".

**Legend text.** The only legend left is (a)'s "turning points", in plain words; (b) and (c) have none.

## The record each panel reads

- **(a)** cell `{a['cell']}` (T2), eval dir `{rec['a']['eval_dir']}`:
  - Q per cycle = `gross_operational_cost` from {r(rec['a']['per_cycle_record'])};
  - k0 = {a['k0']}, N = {a['N']}, k* = {a['kstar']}, the window {a['window'][0]}–{a['window'][1]}, its band and the
    turning points from {r(rec['a']['settling_decision'])};
  - range/τ from the frozen tables (`tables.cells.{a['cell']}.range_over_tau`).

  Cross-checked against {r(rec['a']['w101_summary'])} and {r(rec['a']['continuation_record'])}.
- **(b)** cell `{b['cell']}` (T2, chosen by the W165 panel rule), eval dir `{rec['b']['eval_dir']}`:
  - Q per cycle = `gross` from {r(rec['b']['resettle_cycle_record'])};
  - k0 = {b['k0']}, cap = {b['cap']}, the shaded last {b['band_window'][1] - b['band_window'][0] + 1} cycles
    ({b['band_window'][0]}–{b['band_window'][1]}) and the turning points from {r(rec['b']['resettle_decision'])}.

  Cross-checked against {r(rec['b']['per_cycle_record'])}.
- **(c)** the same evaluation as (b):
  - the gap per cycle (`t_sum`) from {r(rec['b']['resettle_cycle_record'])};
  - the gap bound τ/2 = {b['gap_bound']:,.3f} € from {r(rec['b']['resettle_decision'])}.

**Source of the data.** All panels come from the W165 figure script's data function, `data_fig4`; all
{len(fd.d['checks'])} of its checks held, on the frozen tables `{os.path.basename(W165.FZ_REL)}`. Before any panel was
drawn, the composite re-rendered from the same data was byte-identical to the committed `fig4_q_versus_cycle.pdf`
(sha256 `{FIG4_SHA[:8]}`).

**Objective convention.** Q = `gross_operational_cost`, contracted interface settlement excluded.
- (a) is relative to Q(k*) = {a['Q_ref']:,.4f} €.
- (b) is relative to Q(cap) = {b['q'][b['cap']]:,.4f} €.

## What differs from the composite `fig4_q_versus_cycle.pdf`

- **All three:** one size; y tick labels at two decimals; no panel letters (author, 2026-11-13).
- **(b):** its own x tick labels. In the composite these were hidden because (b) shared its x axis with (c). No x
  label; (c) carries it, as in the composite.
- **(b), (c):** one fixed axes rectangle and the same x ticks (v6), in place of each file's own constrained layout. The
  x limits are ({b['k0']} − 1, {b['cap']} + 1), as before.
- **(c):** no gap-refusal ticks and no legend (Expert, 2026-11-13).
- **Unchanged:**
  - axes and limits (the y labels changed in v7, above);
  - the markers: $k_0$, N and k* in (a); $k_0$ and the cap in (b) and (c);
  - turning points, shading, the band lines and the ±τ/2 band.

**Earlier versions, kept:** `fig_q_cycle_v2/` to `v6/`.

**Zero solves:** the W165 and W157 guards were verified at 0, and the pickle load counters at 0.
"""
    note_rel = os.path.join(out_dir, 'NOTE.md')
    with open(os.path.join(REPO, note_rel), 'x', encoding='utf-8') as h:
        h.write(note)
    written.append(note_rel)
    man = {rel: W165._sha(rel) for rel in sorted(written)}
    man_rel = os.path.join(out_dir, 'manifest_sha256.json')
    with open(os.path.join(REPO, man_rel), 'x', encoding='utf-8') as h:
        h.write(W165.GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for rel in sorted(written):
        W165._log(f'[split] wrote {rel} {man[rel][:8]}')
    W165._log(f'[split] all {len(checks)} checks held; guards 0; pickle 0')
    sys.exit(0)


if __name__ == '__main__':
    main()
