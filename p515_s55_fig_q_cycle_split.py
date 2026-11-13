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
markers, turning points, refusal ticks, shading, panel letters), each in its own figure. The changes, recorded in
the build record:
  all  one size, PANEL_W_MM x PANEL_H_MM (95 x 70 mm; v2 kept the composite sizes 95 x 112 / 56 / 56 mm) and y tick
       labels at two decimals (the composite's formatter prints '5', '2.5'), both at the author's request (2026-11-13);
  (b) shows its own x tick labels and the x label 'ADMM cycle' (in the composite they are hidden: x shared with (c));
      its x limits are set explicitly to (k0 - 1, cap + 1), the limits it inherited from (c);
  (c) none.

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
OUT_DIR_DEFAULT = os.path.join('Claude outputs', 'fig_q_cycle_v3')
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
    axa.set_ylabel('Q − Q(k*) (k€)')
    yfmt2(axa)
    axa.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), handlelength=1.0, borderaxespad=0.2)
    W165.panel_label(axa, '(a)')
    return fig


def draw_b(D):
    fig = _fig('b')
    axb = fig.add_subplot(1, 1, 1)
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
    axb.set_ylabel('Q − Q(cap) (k€)')
    yfmt2(axb)
    # separation: the composite hides these (x shared with (c)); a standalone panel needs its own x axis
    axb.set_xlim(b['k0'] - 1, b['cap'] + 1)
    axb.set_xlabel('ADMM cycle')
    W165.panel_label(axb, '(b)')
    return fig


def draw_c(D):
    fig = _fig('c')
    axc = fig.add_subplot(1, 1, 1)
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
    tr = blended_transform_factory(axc.transData, axc.transAxes)
    axc.plot(b['refusals'], [0.04] * len(b['refusals']), '|', color=OKABE_ITO['vermillion'], ms=4.0, mew=0.7,
             transform=tr, zorder=4, label='gap-clause refusals')
    axc.text(b['cap'] - 0.6, gb, '±τ/2', ha='right', va='bottom', fontsize=6.0, color=OKABE_ITO['green'])
    axc.axvline(b['k0'], color=OKABE_ITO['green'], lw=0.7)
    axc.axvline(b['cap'], color=OKABE_ITO['black'], lw=0.7)
    axc.set_xlim(b['k0'] - 1, b['cap'] + 1)
    axc.set_xlabel('ADMM cycle')
    axc.set_ylabel('$t_{sum}$ (k€)')
    yfmt2(axc)
    axc.legend(loc='lower right', bbox_to_anchor=(1.0, 0.07), handlelength=0.8, borderaxespad=0.2)
    W165.panel_label(axc, '(c)')
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
    panels = {}
    for p, fn in (('a', draw_a), ('b', draw_b), ('c', draw_c)):
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
                               'y tick labels at two decimals (author, 2026-11-13)'],
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
                        'n_gap_refusal_ticks': len(b['refusals']), 'refusal_cycles': b['refusals']}},
        'checks': {k: {'ok': ok, 'detail': d} for k, (ok, d) in checks.items()},
    }
    build_rel = os.path.join(out_dir, 'build_record.json')
    with open(os.path.join(REPO, build_rel), 'x', encoding='utf-8') as h:
        h.write(W165.GRIO.dumps(W165._norm(build), indent=1, sort_keys=True) + '\n')
    written.append(build_rel)

    def r(x):
        return f"`{x['path']}` (sha256 `{x['sha256'][:8]}`)"
    note = f"""# Figure fig:res_q_cycle as three panels (zero solves)

Re-export of `fig4_q_versus_cycle.pdf` (sha256 `{FIG4_SHA[:8]}`, the file in the Overleaf clone) as three PDFs.
Generated by `{SCRIPT_REL}`, which calls the W165 figure script's own data function (`data_fig4`, all
{len(fd.d['checks'])} of its checks held) on the frozen tables `{os.path.basename(W165.FZ_REL)}`. Before drawing the
panels it re-rendered the composite from the same data and matched the committed PDF bitwise. Every plotted value is
therefore the current figure's.

| file | sha256 | size (mm) | panel |
|---|---|---|---|
| `fig_q_cycle_a.pdf` | `{panels['a']['sha256'][:16]}` | {PANEL_W_MM:g} × {PANEL_H_MM['a']:g} | (a) objective of the plan without storage (x = 0), Q − Q(k*) |
| `fig_q_cycle_b.pdf` | `{panels['b']['sha256'][:16]}` | {PANEL_W_MM:g} × {PANEL_H_MM['b']:g} | (b) objective of the reference unit at 1.5×, Q − Q(cap) |
| `fig_q_cycle_c.pdf` | `{panels['c']['sha256'][:16]}` | {PANEL_W_MM:g} × {PANEL_H_MM['c']:g} | (c) priced interface-consensus gap t_sum of the same evaluation, with ±τ/2 |

All three panels have one size (the author's request, 2026-11-13). The predecessor `fig_q_cycle_v2/` kept the
composite sizes (95 × 112, 95 × 56, 95 × 56 mm) and the composite's tick format; it is kept, not overwritten.

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
  - t_sum per cycle from {r(rec['b']['resettle_cycle_record'])};
  - the gap bound τ/2 = {b['gap_bound']:,.3f} € and the {len(b['refusals'])} gap-clause refusal cycles (the ticks) from
    {r(rec['b']['resettle_decision'])}.

Objective convention: Q = `gross_operational_cost`, contracted interface settlement excluded. (a) is relative to
Q(k*) = {a['Q_ref']:,.4f} €; (b) is relative to Q(cap) = {b['q'][b['cap']]:,.4f} €.

## What differs from the composite

- **All three (author, 2026-11-13):** one size, {PANEL_W_MM:g} × {PANEL_H_MM['a']:g} mm; y tick labels at two
  decimals (the composite prints 5, 2.5, …).
- **(a), (c):** nothing else; the drawing calls are the composite's.
- **(b):** shows its own x tick labels and the x label "ADMM cycle". In the composite these are hidden because (b)
  shares its x axis with (c). Its x limits are set to ({b['k0']} − 1, {b['cap']} + 1), the limits it inherited from (c).

As in the current figure:
- **Markers.** (b) and (c) carry k0 and the cap, not N or k*: the evaluation is uncertified, so it has no k*. Its N
  belongs to an earlier run and was never a point of this trajectory, so W165 does not draw it.
- **Ticks.** The refusal ticks are Okabe–Ito vermillion (`#D55E00`), which the caption calls orange. The legend reads
  "gap-clause refusals".
- **Panel letters** (a)–(c) are kept on each file.

Zero solves: the W165 and W157 guards were verified at 0, and the pickle load counters at 0.
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
