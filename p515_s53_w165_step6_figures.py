"""P5.15 Step 6 support, Planner task W165 -- MANUSCRIPT FIGURES 1-5 (STEP6_REVISION_MAP.md section D) FROM THE FROZEN
TABLES AND NAMED COMMITTED RECORDS. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: nothing earlier is edited; the W157 builder
is IMPORTED (it arms its own zero-permit guards and brings W145's fit function, which is CALLED, not copied).

SOURCES
  Frozen tables  data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json (sha256 pinned below);
                 every plotted number is read from it, except where a figure names a committed record:
  Figure 1 (T3)  points value - I = -d_gross of the T1 B rows (F form: d = F(x) - F(0)); E, P from the T2 cell
                 canonical candidate; certified bands from T2 band_width; intervals (bar, bar + 2 tau) and both fits'
                 coefficients b, c, e*, margins and the e* range over the interval box from T3 conservative_A64 (the
                 manuscript figure). The fits' INTERCEPT a is NOT in the frozen JSON: it is computed by CALLING
                 W145.banded_breakeven_fit (the recorded formula, value = a + b E + c P by OLS) on W145's committed
                 points with d_4a82a64a treated as uncertified exactly as W157 built T3, and the call must reproduce
                 every T3 conservative_A64 coefficient and range bitwise (else figure 1 is not drawn).
  Figure 2 (T1 H, T10)  value - I and threshold / bar of CHECK:headline_V_minus_I_settled (m = 1), H:m1.5, H:m2 (T1)
                 and H:m1.75 (T10). The multiplier of each cell is read from its campaign spec
                 (candidates[].flex_price_multiplier); a spec without the key is the base price, read as the harness
                 does (p515_s53_w142_resettle_v6_campaign.py: `... is not None else 1.0`, parsed from that line).
  Figure 3 (T8)  value - I and floor-binding year per arm from T8; threshold from the matching T1 E claim.
  Figure 4 (records) the per-cycle trajectories of (a) the settled x = 0 reference ref:7aa017f0 (W101 continuation
                 d110bd1a5977df1e_x0: per_cycle_record.jsonl, settling_decision.json) and (b, c) the gap-refused cell
                 chosen by the PANEL RULE below (W142 v6 record: resettle_cycle_record.jsonl, resettle_decision.json).
                 Each trajectory must reproduce its T2 row exactly (k0 where T2 records it, k*, end, range / tau,
                 Q at certification or at the cap, gap, slack, turning points); a disagreement stops figure 4.
  Figure 5 (T6)  the three arrangements' Q, the benefit and its relative size from T6; the TSO / DSO decomposition of
                 the price-taker benefit is NOT in T6: it is read from the named record W130
                 (w130_benchmark_addendum.json task1.totals.benefit_price_taker, commit 2c1b731a), cross-checked against
                 T6 (arm Q per start bitwise; benefit total within 1e-6 EUR).

PANEL RULE (figure 4, second panel; fixed BEFORE any trajectory is read): the first row in T2 table order
(export/T2.csv, which must list the frozen JSON's cells then cells_appended_w160 in that order) whose recorded
uncertified cause is a gap refusal: status 'uncertified', cause_uncertified == 'gap clause' and gap_refused is True.
If any uncertified row's cause mentions 'gap' without being exactly 'gap clause', or gap_refused disagrees with the
cause string, the rule is AMBIGUOUS: figure 4 is not drawn and the candidate list is written to the build record.

OBJECTIVE CONVENTION (every figure): Q = gross_operational_cost, settlement excluded; value = Q(0) - Q(x); I = investment
cost; tau = the frozen JSON's constants.TAU (= settling_criterion_v6.TAU, asserted).

OUTPUTS (data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/, created by the launcher; the script refuses to run
if it holds any file other than launch.log; every output opened 'x'):
  fig1_breakeven.pdf ... fig5_benchmark.pdf (vector, pdf.fonttype 42, deterministic: CreationDate / ModDate None,
  Creator / Producer fixed; each figure is generated TWICE in the run and the two byte strings must have equal sha256),
  figN_<name>_data.json (every plotted value with its source path, JSON path and rounding), figN_<name>_preview.png
  (200 dpi PREVIEWS for review, not deliverables), captions.md (draft captions), w165_build_record.json,
  manifest_inputs_sha256.json (written BEFORE any figure), manifest_sha256.json (outputs), and after the typing test
  manifest_post_run_sha256.json (--post-run).

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end together with every guard the W157 import arms; `pickle.load` / `pickle.loads` blocked for the whole run (re-blocked
after the imports) and every counter verified at 0.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w160_step6_frozen/export/figures && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w165_step6_figures.py \\
        > data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/w165_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w160_step6_frozen/export/figures/w165_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w165_step6_figures.py --post-run
  Trial runs: --out-dir <scratch dir> (the script need not be committed; recorded as a trial).
Exit: 0 = every figure written, determinism and every check hold, guards 0; 3 = written, a check failed (the figure it
concerns is NOT written; listed); 1 = precondition or guard fault (nothing written beyond the input manifest).
"""
import argparse
import glob
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W165 Step 6 figures (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W165: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W165: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import settling_criterion as SC1  # noqa: E402 -- version 1 (the reference continuation's rule): window_length
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w157_step6_tables as W157  # noqa: E402 -- arms its guards, blocks pickle (W145 chain); not edited

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
import numpy as np  # noqa: E402

W145 = W157.W145
L132 = W157.L132
GUARDS = W157._dedupe((('w165_step6_figures', GUARD),) + tuple(W157.GUARDS))

SCRIPT_REL = os.path.basename(__file__)
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
FROZEN_DIR = os.path.join(S53, 'w160_step6_frozen')
FZ_REL = os.path.join(FROZEN_DIR, 'frozen_step6_tables_v1_590088fe.json')
FZ_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
FZ_MD = os.path.join(FROZEN_DIR, 'frozen_step6_tables_v1_590088fe.md')
EXPORT = os.path.join(FROZEN_DIR, 'export')
T2_CSV = os.path.join(EXPORT, 'T2.csv')
README = os.path.join(EXPORT, 'README.md')
PARAGRAPHS_V5 = os.path.join(EXPORT, 'paragraphs_v5.md')
MAP_REL = 'STEP6_REVISION_MAP.md'
W145_JSON = os.path.join(S53, 'w145_banded_fit', 'w145_banded_fit.json')
W145_MAN = os.path.join(S53, 'w145_banded_fit', 'manifest_sha256.json')
W130_JSON = os.path.join(S53, 'w130_benchmark_addendum', 'w130_benchmark_addendum.json')
W130_MAN = os.path.join(S53, 'w130_benchmark_addendum', 'manifest_sha256.json')
W130_COMMIT = '2c1b731a'
HARNESS_V6 = 'p515_s53_w142_resettle_v6_campaign.py'
HARNESS_DEFAULT_RE = re.compile(r"'m_flex_price_multiplier': c\['flex_price_multiplier'\] if c\['flex_price_multiplier'\] "
                                r"is not None else ([0-9.]+),")

OUT_DIR = os.path.join(EXPORT, 'figures')
FIGS = (('fig1', 'breakeven'), ('fig2', 'flexibility_ladder'), ('fig3', 'ageing_arms'), ('fig4', 'q_versus_cycle'),
        ('fig5', 'benchmark'))
OUT_CAPTIONS = 'captions.md'
OUT_BUILD = 'w165_build_record.json'
OUT_MAN_IN = 'manifest_inputs_sha256.json'
OUT_MAN = 'manifest_sha256.json'
OUT_POST = 'manifest_post_run_sha256.json'
OUT_LOG = 'launch.log'
OUT_TYPING_JSON = 'w165_bool_typing_test.json'
OUT_TYPING_LOG = 'w165_bool_typing_test.log'

MM = 1.0 / 25.4
WIDTH_MM = {'single': 90.0, 'one_and_half': 140.0, 'double': 190.0}   # Elsevier column widths (layout constants)
PREVIEW_DPI = 200
PDF_METADATA = {'Creator': 'p515_s53_w165_step6_figures.py', 'Producer': 'Matplotlib pdf backend',
                'CreationDate': None, 'ModDate': None}
FONT_FAMILY = 'Arial'
OKABE_ITO = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'yellow': '#F0E442',
             'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999'}
OBJECTIVE_CAPTION = ('Objective convention: Q = gross operational cost (gross_operational_cost), settlement excluded; '
                     'value = Q(0) - Q(x); I = investment cost.')
AGEING_ORDER = ('no_ageing', 'C2', 'C2_calfade', 'C3_unit', 'C3_midblock', 'C4')   # the Planner's order (task W165)
BASELINE_ARM = 'C2_calfade'
FLOOR_KEY = 'floor_year_070'                     # the T8 column (0.70 floor)
BASELINE_PHRASE = 'calendar retention 0.985/yr in the baseline (C2_calfade)'      # STEP6_REVISION_MAP.md section A


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as handle:
        for blk in iter(lambda: handle.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _last_commit(rel):
    try:
        return _git('log', '-1', '--format=%H', '--', rel) or None
    except subprocess.CalledProcessError:
        return None


def _jl(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return json.load(h)


def _rows(rel):
    out = {}
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        for ln in h:
            if ln.strip():
                r = json.loads(ln)
                if r['cycle'] in out:
                    raise RuntimeError(f'{rel}: duplicate cycle {r["cycle"]}')
                out[r['cycle']] = r
    return out


def _norm(x):
    return json.loads(GRIO.dumps(x, sort_keys=True))


def _k(v):
    return v / 1000.0


def fmt_k(v, dp=1):
    """EUR -> 'k EUR, 1 dp' text with thousands separators (README rounding rule for EUR amounts)."""
    s = f'{abs(v) / 1000.0:,.{dp}f}'
    if round(abs(v) / 1000.0, dp) == 0:
        return s
    return ('−' if v < 0 else '') + s


def fmt_kmwh(v):
    """EUR/MWh -> k EUR/MWh text at the README rule (nearest 10 EUR/MWh = 2 dp in k EUR/MWh)."""
    return f'{round(v, -1) / 1000.0:,.2f}'


# ======================================================================================================================
#  sources: every value carries {path, sha256, json_path}
# ======================================================================================================================
class Sources:
    def __init__(self):
        self.sha = {}

    def ref(self, rel, json_path):
        if rel not in self.sha:
            self.sha[rel] = _sha(rel)
        return {'path': rel, 'sha256': self.sha[rel], 'json_path': json_path}


SRC = Sources()


class FigData:
    def __init__(self, fig_id, name, width_key, height_mm):
        self.d = {'figure': f'{fig_id}_{name}', 'width': width_key, 'width_mm': WIDTH_MM[width_key],
                  'height_mm': height_mm, 'objective_convention': OBJECTIVE_CAPTION, 'values': [], 'computed': [],
                  'caption_values': [], 'checks': {}, 'layout_choices': []}

    def val(self, name, value, rel, json_path, rounding='plotted at full precision', unit='EUR', note=None):
        e = {'name': name, 'value': value, 'unit': unit, 'source': SRC.ref(rel, json_path), 'rounding': rounding}
        if note:
            e['note'] = note
        self.d['values'].append(e)
        return value

    def comp(self, name, value, formula, inputs, rounding='plotted at full precision', unit='EUR'):
        self.d['computed'].append({'name': name, 'value': value, 'unit': unit, 'formula': formula, 'inputs': inputs,
                                   'rounding': rounding})
        return value

    def cap(self, name, value, rel, json_path, written, unit='EUR'):
        self.d['caption_values'].append({'name': name, 'value': value, 'unit': unit, 'written': written,
                                         'source': SRC.ref(rel, json_path)})
        return written

    def check(self, name, ok, detail=None):
        self.d['checks'][name] = {'ok': bool(ok), 'detail': detail}
        return bool(ok)

    def layout(self, text):
        self.d['layout_choices'].append(text)

    def failed(self):
        return [k for k, v in self.d['checks'].items() if not v['ok']]


# ======================================================================================================================
#  style
# ======================================================================================================================
def font_record():
    path = font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    return {'family': FONT_FAMILY, 'path': path, 'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest()}


def set_style():
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        'font.family': [FONT_FAMILY, 'DejaVu Sans'], 'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 7.0,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.0, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 0.9, 'lines.markersize': 3.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': FONT_FAMILY, 'mathtext.it': f'{FONT_FAMILY}:italic',
        'mathtext.bf': f'{FONT_FAMILY}:bold', 'axes.unicode_minus': True, 'legend.frameon': False,
        'pdf.compression': 6, 'svg.hashsalt': 'w165', 'figure.dpi': 100, 'savefig.dpi': PREVIEW_DPI,
        'axes.spines.top': False, 'axes.spines.right': False})


def new_fig(fd, **kw):
    return plt.figure(figsize=(fd.d['width_mm'] * MM, fd.d['height_mm'] * MM), layout='constrained', **kw)


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, fontweight='bold')


def _tick(v, _p=None):
    s = f'{v:,.0f}' if abs(v) >= 1000 else f'{v:g}'
    return s.replace('-', '−')


def kfmt(ax, axis='y'):
    f = matplotlib.ticker.FuncFormatter(_tick)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(f)


# ======================================================================================================================
#  helpers on the frozen JSON
# ======================================================================================================================
def claims_by_id(fz):
    out = {}
    for i, c in enumerate(fz['tables']['claims']):
        if c['claim_id'] in out:
            raise RuntimeError(f'duplicate claim id {c["claim_id"]}')
        out[c['claim_id']] = (i, c)
    return out


def all_cells(fz):
    cells = dict(fz['tables']['cells'])
    for k, v in fz['tables']['cells_appended_w160'].items():
        if k in cells:
            raise RuntimeError(f'cell {k} both in cells and cells_appended_w160')
        cells[k] = v
    return cells


def unit_desc(cand):
    """'node 7, 0.25 MVA / 1 MWh, 2025' from a candidate_canonical (every node with a non-zero [P, E])."""
    parts = [f'node {n}, {pe[0]:g} MVA / {pe[1]:g} MWh' for n, pe in sorted(cand['nodes'].items()) if pe != [0.0, 0.0]]
    return '; '.join(parts) + f", {cand['investment_year']}"


def find_eval_dir(eval_key):
    """The unique committed eval dir named <eval_key[:16]>_* under P5.15 S53, whose launch.json carries eval_key."""
    pats = (os.path.join(S53, '*', 'evals', eval_key[:16] + '_*'), os.path.join(S53, '*', '*', 'evals', eval_key[:16] + '_*'))
    hits = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, pat))})
    rels = [os.path.relpath(p, REPO) for p in hits if os.path.isdir(p)]
    good = [r for r in rels if os.path.exists(os.path.join(REPO, r, 'launch.json'))
            and _jl(os.path.join(r, 'launch.json')).get('eval_key') == eval_key]
    return good, rels


def campaign_spec_of(eval_dir):
    root = os.path.dirname(os.path.dirname(eval_dir))
    specs = sorted(glob.glob(os.path.join(REPO, root, 'campaign_spec_*.json')))
    if len(specs) != 1:
        raise RuntimeError(f'{root}: expected one campaign spec, found {specs}')
    return os.path.relpath(specs[0], REPO), root


def harness_default_multiplier():
    with open(os.path.join(REPO, HARNESS_V6), encoding='utf-8') as h:
        txt = h.read()
    ms = HARNESS_DEFAULT_RE.findall(txt)
    lines = [i + 1 for i, ln in enumerate(txt.splitlines()) if HARNESS_DEFAULT_RE.search(ln)]
    if len(set(ms)) != 1:
        raise RuntimeError(f'{HARNESS_V6}: harness default multiplier not found uniquely: {ms}')
    return float(ms[0]), lines


# ======================================================================================================================
#  FIGURE 1 -- break-even fit (T3)
# ======================================================================================================================
def data_fig1(fz, cells, claims):
    fd = FigData('fig1', 'breakeven', 'double', 66.0)
    fd.layout('double column (190 mm): (a) the 4 h points (the map figure), (b) the 2 h points of the same fits, '
              '(c) the break-even energy cost of both fits against the energy cost')
    fd.layout('T3 variant plotted: conservative_A64 (the manuscript figure, T3 row 2); d_4a82a64a (n7_4h_e3) drawn as '
              'an interval as that fit treats it (Addendum 64 ruling 4)')
    be = fz['tables']['break_even']
    A = be['conservative_A64']
    jp = 'tables.break_even'
    p_cost = fd.val('p_cost', be['p_cost_eur_per_MVA'], FZ_REL, f'{jp}.p_cost_eur_per_MVA', unit='EUR/MVA')
    e_cost = fd.val('e_cost', be['e_cost_eur_per_MWh'], FZ_REL, f'{jp}.e_cost_eur_per_MWh', unit='EUR/MWh',
                    rounding='plotted at full precision; annotated at nearest 10 EUR/MWh (k EUR/MWh 2 dp)')
    tau = fz['constants']['TAU']
    two_tau = fz['constants']['TWO_TAU']
    fits = {}
    for fk, lab in (('certified_only', 'certified-only fit'), ('banded_mid', 'banded fit, interval midpoints')):
        f = A[fk]
        fits[fk] = {'label': lab, 'n': fd.val(f'{fk}.n', f['n'], FZ_REL, f'{jp}.conservative_A64.{fk}.n', unit='points'),
                    'b': fd.val(f'{fk}.b', f['b_per_MWh'], FZ_REL, f'{jp}.conservative_A64.{fk}.b_per_MWh',
                                unit='EUR/MWh'),
                    'c': fd.val(f'{fk}.c', f['c_per_MVA'], FZ_REL, f'{jp}.conservative_A64.{fk}.c_per_MVA',
                                unit='EUR/MVA'),
                    'e_star': fd.val(f'{fk}.e_star', f['breakeven_marginal_4h_energy_cost'], FZ_REL,
                                     f'{jp}.conservative_A64.{fk}.breakeven_marginal_4h_energy_cost', unit='EUR/MWh',
                                     rounding='plotted at full precision; annotated nearest 10 EUR/MWh'),
                    'margin': f['margin_to_energy_cost_per_MWh']}
    rng = fd.val('banded.breakeven_range', list(A['breakeven_range']), FZ_REL, f'{jp}.conservative_A64.breakeven_range',
                 unit='EUR/MWh', rounding='plotted at full precision; annotated nearest 10 EUR/MWh')
    mrg = fd.val('banded.margin_to_cost_range', list(A['margin_to_cost_range']), FZ_REL,
                 f'{jp}.conservative_A64.margin_to_cost_range', unit='EUR/MWh',
                 rounding='annotated nearest 10 EUR/MWh (k EUR/MWh 2 dp)')
    fd.check('manuscript_figure_is_conservative_A64',
             be['manuscript_figure']['breakeven_max_eur_per_mwh'] == A['breakeven_range'][1]
             and be['manuscript_figure']['margin_min_eur_per_mwh'] == A['margin_to_cost_range'][0],
             be['manuscript_figure'])
    # ---- the intercept a: W145's function on W145's committed points, d_4a82a64a treated as uncertified (as W157) ----
    w = _jl(W145_JSON)
    q0 = dict(w['q0_reference'])
    pts = []
    for lb in W145.LABELS:
        v = dict(w['points'][lb])
        v['label'] = lb
        pts.append(v)
    d4 = cells['d_4a82a64a']
    cons = [W157._treated_uncertified(p, d4) if p['label'] == 'n7_4h_e3' else p for p in pts]
    pc, ec = w['constants']['p_cost_eur_per_MVA'], w['constants']['e_cost_eur_per_MWh']
    fd.check('w145_costs_equal_T3', (pc, ec) == (p_cost, e_cost), [pc, ec])
    res = W145.banded_breakeven_fit(cons, q0, pc, ec)
    co, mid = res['certified_only_fit'], res['banded_fit']['midpoint_fit']
    rep = {'certified_only.b': co['b_per_MWh'] == fits['certified_only']['b'],
           'certified_only.c': co['c_per_MVA'] == fits['certified_only']['c'],
           'certified_only.e_star': co['breakeven_marginal_4h_energy_cost'] == fits['certified_only']['e_star'],
           'certified_only.margin': co['margin_to_energy_cost_per_MWh'] == fits['certified_only']['margin'],
           'certified_only.n': co['n'] == fits['certified_only']['n'],
           'banded_mid.b': mid['b_per_MWh'] == fits['banded_mid']['b'],
           'banded_mid.c': mid['c_per_MVA'] == fits['banded_mid']['c'],
           'banded_mid.e_star': mid['breakeven_marginal_4h_energy_cost'] == fits['banded_mid']['e_star'],
           'banded_mid.n': mid['n'] == fits['banded_mid']['n'],
           'breakeven_range': res['banded_fit']['breakeven_range'] == rng,
           'margin_to_cost_range': res['banded_fit']['margin_to_cost_range'] == mrg,
           'certified_set': sorted(res['certified']) == sorted(A['certified']),
           'interval_set': sorted(res['uncertified']) == sorted(A['uncertified_as_intervals'])}
    fd.check('w145_call_reproduces_T3_conservative_A64_bitwise', all(rep.values()), rep)
    fd.check('w145_committed_midpoint_a_equal', w['result']['banded_fit']['midpoint_fit']['a'] == mid['a'],
             [w['result']['banded_fit']['midpoint_fit']['a'], mid['a']])
    for fk, src in (('certified_only', co), ('banded_mid', mid)):
        fits[fk]['a'] = fd.comp(f'{fk}.a', src['a'], 'value = a + b E + c P (OLS, np.linalg.lstsq) -- W145.banded_'
                                'breakeven_fit -> L132.d_fit, called on W145 committed points (w145_banded_fit.json '
                                'points, W145.LABELS order) with n7_4h_e3 treated as uncertified by '
                                'W157._treated_uncertified (gap = |t_sum_end|, slack = |s_signed| of T2 d_4a82a64a)',
                                {'w145_json': SRC.ref(W145_JSON, 'points, q0_reference, constants'),
                                 'frozen': SRC.ref(FZ_REL, 'tables.cells.d_4a82a64a.t_sum_end, s_signed')},
                                rounding='full precision; the intercept is NOT in the frozen JSON')
    # ---- the points --------------------------------------------------------------------------------------------
    ref0 = 'ref:7aa017f0'
    Q0 = cells[ref0]['Q']
    prov = w['provenance']
    points = []
    for lb in W145.LABELS:
        cid = f'B:{lb}'
        i, c = claims[cid]
        if c['ref_cell'] != ref0 or c['form'] != 'F':
            raise RuntimeError(f'{cid}: unexpected ref {c["ref_cell"]} / form {c["form"]}')
        cell = c['other_cell']
        cr = cells[cell]
        P, E = cr['candidate_canonical']['nodes']['7']
        others = [v for n, v in cr['candidate_canonical']['nodes'].items() if n != '7']
        e_w, p_w = L132._ep_of(lb)
        fd.check(f'{lb}.canonical_equals_label_E_P', (E, P) == (e_w, p_w) and all(o == [0.0, 0.0] for o in others),
                 {'cell_P_E': [P, E], 'label_E_P': [e_w, p_w]})
        fd.check(f'{lb}.w145_point_Q_equals_T2_Q', w['points'][lb]['Q'] == cr['Q'], [w['points'][lb]['Q'], cr['Q']])
        pek = prov[lb].get('eval_key')
        fd.check(f'{lb}.w145_provenance_eval_key_equals_T2_where_recorded', (not pek) or pek == cr['eval_key'],
                 [pek, cr['eval_key'], prov[lb].get('source')])
        y = fd.val(f'{lb}.value_minus_I', -c['d_gross'], FZ_REL, f'tables.claims[{i}].d_gross (claim {cid}; '
                   'value - I = -d_gross, F form)', rounding='plotted at full precision (k EUR axis)')
        fd.val(f'{lb}.E', E, FZ_REL, f'tables.cells.{cell}.candidate_canonical.nodes.7[1]', unit='MWh')
        fd.val(f'{lb}.P', P, FZ_REL, f'tables.cells.{cell}.candidate_canonical.nodes.7[0]', unit='MVA')
        recomputed = Q0 - cr['Q'] - c['I_other']
        fd.check(f'{lb}.value_minus_I_equals_Q0_minus_Q_minus_I', abs(recomputed - y) <= 1e-6,
                 {'Q0 - Q - I': recomputed, '-d_gross': y})
        if lb in A['uncertified_as_intervals']:
            iv = A['intervals'][lb]
            kind = 'interval'
            bar = fd.val(f'{lb}.bar', iv['bar'], FZ_REL, f'tables.break_even.conservative_A64.intervals.{lb}.bar',
                         rounding='plotted at full precision (thick segment = +/- bar)')
            hw = fd.val(f'{lb}.half_width', iv['half_width'], FZ_REL,
                        f'tables.break_even.conservative_A64.intervals.{lb}.half_width',
                        rounding='plotted at full precision (thin segment = +/- (bar + 2 tau), the fit box)')
            fd.check(f'{lb}.half_width_is_bar_plus_two_tau', abs(iv['half_width'] - (iv['bar'] + two_tau)) <= 1e-6,
                     [iv['half_width'], iv['bar'] + two_tau])
            fd.check(f'{lb}.Q_cap_is_T2_Q', iv['Q_cap'] == cr['Q'], [iv['Q_cap'], cr['Q']])
            band = None
        elif lb in A['certified']:
            kind = 'certified'
            bar = hw = None
            band = fd.val(f'{lb}.band', cr['band_width'], FZ_REL, f'tables.cells.{cell}.band_width',
                          rounding='plotted at full precision (vertical bar of length band width, centred)')
        else:
            raise RuntimeError(f'{lb} neither certified nor an interval in T3 conservative_A64')
        points.append({'label': lb, 'cell': cell, 'E': E, 'P': P, 'h': E / P, 'y': y, 'kind': kind, 'bar': bar,
                       'hw': hw, 'band': band, 'status_T2': cr['status']})
    fd.check('every_T3_point_plotted', sorted(p['label'] for p in points) == sorted(A['certified'] +
                                                                                    A['uncertified_as_intervals']))
    fd.cap('committed_W145_breakeven_range', be['committed_W145']['breakeven_range'], FZ_REL,
           'tables.break_even.committed_W145.breakeven_range',
           f"{fmt_kmwh(be['committed_W145']['breakeven_range'][0])}-{fmt_kmwh(be['committed_W145']['breakeven_range'][1])}",
           unit='EUR/MWh')
    fd.cap('tau', tau, FZ_REL, 'constants.TAU', f'{tau:,.2f}')
    # caption quantities derived from the data (no literal in the caption text)
    ratios = [A['intervals'][lb]['bar'] / max(abs(A['intervals'][lb]['gap']), abs(A['intervals'][lb]['slack']))
              for lb in A['uncertified_as_intervals']]
    unc_factor = round(ratios[0])
    fd.check('uncertified_bar_factor_is_one_integer', all(abs(r - unc_factor) <= 1e-9 for r in ratios), ratios)
    fd.comp('uncertified_bar_factor', unc_factor, 'bar / max(|gap|, |slack|) of every T3 interval (L132.resolve rule)',
            {'frozen': SRC.ref(FZ_REL, 'tables.break_even.conservative_A64.intervals')}, rounding='integer',
            unit='x')
    tau_mult = two_tau / tau
    fd.check('two_tau_over_tau_equals_SC6_multiple', tau_mult == SC6.DETERMINACY_TAU_MULTIPLE, tau_mult)
    nodes = sorted({n for p in points for n, pe in cells[p['cell']]['candidate_canonical']['nodes'].items()
                    if pe != [0.0, 0.0]})
    years = sorted({cells[p['cell']]['candidate_canonical']['investment_year'] for p in points})
    fd.check('one_node_one_year', len(nodes) == 1 and len(years) == 1, [nodes, years])
    return fd, {'points': points, 'fits': fits, 'rng': rng, 'mrg': mrg, 'e_cost': e_cost, 'p_cost': p_cost,
                'two_tau': two_tau, 'unc_factor': unc_factor, 'tau_mult': tau_mult, 'node': nodes[0],
                'year': years[0], 'durations': sorted({p['h'] for p in points}, reverse=True)}


def draw_fig1(fd, D):
    fig = new_fig(fd)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=axa)
    axc = fig.add_subplot(gs[0, 2])
    fit_style = {'certified_only': {'color': OKABE_ITO['blue'], 'ls': '-'},
                 'banded_mid': {'color': OKABE_ITO['vermillion'], 'ls': '--'}}
    if len(D['durations']) != 2:
        raise RuntimeError(f"figure 1 expects two durations, found {D['durations']}")
    for ax, h, lab in ((axa, D['durations'][0], '(a)'), (axb, D['durations'][1], '(b)')):
        pts = [p for p in D['points'] if p['h'] == h]
        es = [p['E'] for p in pts]
        ee = np.linspace(min(es), max(es), 2)
        for fk, f in D['fits'].items():
            slope = f['b'] + f['c'] / h - D['p_cost'] / h - D['e_cost']
            ax.plot(ee, _k(f['a'] + slope * ee), color=fit_style[fk]['color'], ls=fit_style[fk]['ls'], lw=0.9,
                    label=f"{f['label']} (n = {f['n']})", zorder=2)
        ax.axhline(0.0, color=OKABE_ITO['grey'], lw=0.6, zorder=1)
        for p in pts:
            if p['kind'] == 'certified':
                ax.errorbar([p['E']], [_k(p['y'])], yerr=[[_k(p['band'] / 2)], [_k(p['band'] / 2)]], fmt='o',
                            color=OKABE_ITO['black'], ms=3.5, elinewidth=0.8, capsize=0, zorder=4)
            else:
                flagged = p['status_T2'] == 'certified'
                mk = 'D' if flagged else 's'
                ax.errorbar([p['E']], [_k(p['y'])], yerr=[[_k(p['hw'])], [_k(p['hw'])]], fmt='none',
                            ecolor=OKABE_ITO['grey'], elinewidth=0.7, capsize=1.5, zorder=3)
                ax.errorbar([p['E']], [_k(p['y'])], yerr=[[_k(p['bar'])], [_k(p['bar'])]], fmt=mk, mfc='white',
                            color=OKABE_ITO['black'], ms=3.2, elinewidth=1.6, capsize=0, zorder=4)
        ax.set_xlabel(f"installed energy E (MWh), node {D['node']}, {h:g} h")
        ax.set_xticks(sorted(set(es)))
        panel_label(ax, lab)
    axa.set_ylabel('value − I (k€)')
    kfmt(axa, 'y')
    plt.setp(axb.get_yticklabels(), visible=False)
    # legend: lines + marker kinds
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=fit_style[fk]['color'], ls=fit_style[fk]['ls'], lw=0.9,
                      label=f"{f['label']} (n = {f['n']})") for fk, f in D['fits'].items()]
    handles += [Line2D([], [], marker='o', ls='none', color='k', ms=3.5, label='certified (bar: band width)'),
                Line2D([], [], marker='s', ls='none', color='k', mfc='white', ms=3.2,
                       label='uncertified: ±bar (thick), ±(bar + 2τ) (thin)'),
                Line2D([], [], marker='D', ls='none', color='k', mfc='white', ms=3.2,
                       label='certified, entered as an interval')]
    axa.legend(handles=handles, loc='lower left', handlelength=1.8, borderaxespad=0.2)
    # (c) break-even energy cost
    f_c, f_b = D['fits']['certified_only'], D['fits']['banded_mid']
    yb, yc = 0.0, 1.0
    lo, hi = D['rng']
    axc.errorbar([_k(f_b['e_star'])], [yb], xerr=[[_k(f_b['e_star'] - lo)], [_k(hi - f_b['e_star'])]], fmt='s',
                 color=fit_style['banded_mid']['color'], mfc='white', ms=3.5, elinewidth=1.2, capsize=2.0)
    axc.plot([_k(f_c['e_star'])], [yc], 'o', color=fit_style['certified_only']['color'], ms=3.5)
    axc.axvline(_k(D['e_cost']), color=OKABE_ITO['black'], lw=0.9)
    tr = blended_transform_factory(axc.transData, axc.transAxes)
    axc.text(_k(D['e_cost']) - 1.5, 0.97, f"energy cost\n{fmt_kmwh(D['e_cost'])}", transform=tr, ha='right', va='top',
             fontsize=6.0)
    axc.annotate('', xy=(_k(D['e_cost']), -0.55), xytext=(_k(hi), -0.55),
                 arrowprops={'arrowstyle': '<->', 'lw': 0.6, 'color': OKABE_ITO['grey'], 'shrinkA': 0, 'shrinkB': 0})
    axc.text(_k((hi + D['e_cost']) / 2), -0.47, f"margin ≥ {fmt_kmwh(D['mrg'][0])}", ha='center', va='bottom',
             fontsize=6.0)
    axc.text(_k(f_b['e_star']), yb + 0.18, f"{fmt_kmwh(lo)}–{fmt_kmwh(hi)}", ha='center', va='bottom',
             fontsize=6.0)
    axc.text(_k(f_c['e_star']), yc + 0.18, f"{fmt_kmwh(f_c['e_star'])}", ha='center', va='bottom', fontsize=6.0)
    axc.set_yticks([yb, yc])
    axc.set_yticklabels([f"banded fit\n(n = {f_b['n']}; range\nover intervals)", f"certified-only\nfit (n = {f_c['n']})"])
    axc.set_ylim(-0.9, 1.6)
    axc.set_xlabel('break-even energy cost e* (k€/MWh)')
    xlo = min(_k(lo), _k(f_c['e_star'])) - 15.0
    xhi = _k(D['e_cost']) + 12.0
    axc.set_xlim(xlo, xhi)
    panel_label(axc, '(c)')
    fd.layout('panel (c) x-limits: [min(e* range low, certified-only e*) - 15, energy cost + 12] k EUR/MWh (layout)')
    return fig


# ======================================================================================================================
#  FIGURE 2 -- flexibility ladder (T1 H rows, T10)
# ======================================================================================================================
def multiplier_of_cell(cell_row, default_m, fd, tag):
    ek = cell_row['eval_key']
    good, allhits = find_eval_dir(ek)
    if len(good) != 1:
        raise RuntimeError(f'{tag}: eval dir for {ek[:16]} not unique: {good} (hits {allhits})')
    spec_rel, root = campaign_spec_of(good[0])
    spec = _jl(spec_rel)
    cands = [(j, c) for j, c in enumerate(spec['candidates']) if c.get('eval_key') == ek]
    if len(cands) != 1:
        raise RuntimeError(f'{tag}: {spec_rel} has {len(cands)} candidates with eval key {ek[:16]}')
    j, c = cands[0]
    if c.get('flex_price_multiplier') is None:
        m = default_m
        fd.val(f'{tag}.flex_price_multiplier', m, spec_rel, f'candidates[{j}].flex_price_multiplier absent -> the '
               f'harness default ({HARNESS_V6})', unit='x base price')
    else:
        m = fd.val(f'{tag}.flex_price_multiplier', c['flex_price_multiplier'], spec_rel,
                   f'candidates[{j}].flex_price_multiplier', unit='x base price')
    return m, spec_rel


def data_fig2(fz, cells, claims):
    fd = FigData('fig2', 'flexibility_ladder', 'single', 62.0)
    fd.layout('single column (90 mm); x = flexibility-price multiplier, y = value - I of the unit with the claim\'s '
              'threshold (certified pair) or uncertified bar as +/- error bar')
    default_m, lines = harness_default_multiplier()
    fd.val('harness_default_multiplier', default_m, HARNESS_V6, f'line(s) {lines}: m_flex_price_multiplier = '
           'flex_price_multiplier if not None else <default>', unit='x base price')
    rows, units = [], []
    a64 = {r['claim_id']: (i, r) for i, r in enumerate(fz['tables']['a64']['rows'])}
    spec = (('CHECK:headline_V_minus_I_settled', 'T1'), ('H:m1.5:value_minus_I', 'T1'),
            ('H:m1.75:value_minus_I', 'T10'), ('H:m2:value_minus_I', 'T1'))
    for cid, table in spec:
        if table == 'T1':
            i, c = claims[cid]
            jp = f'tables.claims[{i}]'
        else:
            i, c = a64[cid]
            jp = f'tables.a64.rows[{i}]'
        if c['form'] != 'value':
            raise RuntimeError(f'{cid}: form {c["form"]} is not "value"')
        ms, cands = [], {}
        for side in ('ref_cell', 'other_cell'):
            cn = c[side]
            cr = cells.get(cn) or cells.get(f'ref:{cn}')
            if cr is None and table == 'T10':
                cr = c['cells'][cn]
            m, _sp = multiplier_of_cell(cr, default_m, fd, f'{cid}.{side}.{cn}')
            ms.append(m)
            cands[side] = cr['candidate_canonical']
        if len(set(ms)) != 1:
            raise RuntimeError(f'{cid}: the two cells carry different multipliers {ms}')
        fd.check(f'{cid}.ref_is_x0', all(pe == [0.0, 0.0] for pe in cands['ref_cell']['nodes'].values()),
                 cands['ref_cell'])
        units.append(unit_desc(cands['other_cell']))
        y = fd.val(f'{cid}.value_minus_I', c['d_gross'], FZ_REL, f'{jp}.d_gross', rounding='plotted at full '
                   'precision; annotated k EUR 1 dp')
        thr = fd.val(f'{cid}.threshold_or_bar', c['gross_threshold_or_bar'], FZ_REL, f'{jp}.gross_threshold_or_bar',
                     rounding='plotted at full precision (+/- error bar)')
        fd.val(f'{cid}.rule', c['gross_rule'], FZ_REL, f'{jp}.gross_rule', unit='text', rounding='as recorded')
        fd.val(f'{cid}.verdict', c['gross_verdict'], FZ_REL, f'{jp}.gross_verdict', unit='text', rounding='as recorded')
        fd.val(f'{cid}.multiple', c['gross_multiple'], FZ_REL, f'{jp}.gross_multiple', unit='x',
               rounding='caption 2 dp')
        rows.append({'claim': cid, 'table': table, 'm': ms[0], 'y': y, 'thr': thr, 'rule': c['gross_rule'],
                     'verdict': c['gross_verdict'], 'multiple': c['gross_multiple'],
                     'uncertified_form': c['gross_rule'] == 'uncertified form'})
    ms = [r['m'] for r in rows]
    fd.check('multipliers_strictly_increasing', all(a < b for a, b in zip(ms, ms[1:])), ms)
    fd.check('determinate_iff_abs_over_threshold', all((abs(r['y']) > r['thr']) == (r['verdict'] == 'determinate')
                                                        for r in rows), [[r['claim'], r['verdict']] for r in rows])
    fd.check('one_unit_on_every_rung', len(set(units)) == 1, sorted(set(units)))
    return fd, {'rows': rows, 'unit': units[0], 'bar_factor': SC6.DETERMINACY_BAR_FACTOR,
                'tau_mult': SC6.DETERMINACY_TAU_MULTIPLE}


def draw_fig2(fd, D):
    fig = new_fig(fd)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0.0, color=OKABE_ITO['grey'], lw=0.6, zorder=1)
    for r in D['rows']:
        col = OKABE_ITO['blue'] if r['verdict'] == 'determinate' else OKABE_ITO['vermillion']
        ax.errorbar([r['m']], [_k(r['y'])], yerr=[[_k(r['thr'])], [_k(r['thr'])]], fmt='s' if r['uncertified_form']
                    else 'o', mfc='white' if r['uncertified_form'] else col, color=col, ms=3.8, elinewidth=1.0,
                    capsize=2.0, zorder=3)
        ax.annotate(f"{fmt_k(r['y'])}", (r['m'], _k(r['y'])), xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=6.0)
    ax.set_xticks([r['m'] for r in D['rows']])
    ax.set_xticklabels([f"{r['m']:g}" for r in D['rows']])
    ax.set_xlim(min(r['m'] for r in D['rows']) - 0.12, max(r['m'] for r in D['rows']) + 0.25)
    ax.set_xlabel('flexibility-price multiplier m (× base price)')
    ax.set_ylabel('value − I of the unit (k€)')
    kfmt(ax, 'y')
    from matplotlib.lines import Line2D
    kinds = {(r['verdict'] == 'determinate', r['uncertified_form']) for r in D['rows']}
    hs = []
    for det, unc in sorted(kinds, reverse=True):
        col = OKABE_ITO['blue'] if det else OKABE_ITO['vermillion']
        lab = (('determinate' if det else 'within resolution') + ', ' +
               ('uncertified form (± uncertified bar)' if unc else 'certified pair (± threshold)'))
        hs.append(Line2D([], [], marker='s' if unc else 'o', ls='none', color=col, mfc='white' if unc else col,
                         ms=3.8, label=lab))
    ax.legend(handles=hs, loc='upper left', handlelength=1.2, borderaxespad=0.2)
    fd.layout('x-limits [min m - 0.12, max m + 0.25] (room for the value labels; layout)')
    return fig


# ======================================================================================================================
#  FIGURE 3 -- ageing arms (T8)
# ======================================================================================================================
def data_fig3(fz, cells, claims, map_text):
    fd = FigData('fig3', 'ageing_arms', 'single', 64.0)
    fd.layout('single column (90 mm); arms on the vertical axis in the Planner\'s order (task W165), value - I on the '
              'horizontal axis with the T1 E threshold as +/- error bar; the 0.70 floor-binding year written at the '
              'right of each arm')
    fd.check('baseline_phrase_in_map', BASELINE_PHRASE in map_text, BASELINE_PHRASE)
    rows_t8 = fz['tables']['ageing']['rows']
    by_arm = {r['arm']: (i, r) for i, r in enumerate(rows_t8)}
    fd.check('arm_set_equals_T8', set(by_arm) == set(AGEING_ORDER), sorted(by_arm))
    out = []
    for arm in AGEING_ORDER:
        i, r = by_arm[arm]
        jp = f'tables.ageing.rows[{i}]'
        cl = [(cid, ic) for cid, ic in claims.items() if ic[1]['other_cell'] == r['cell']
              and ic[1]['ref_cell'] == 'ref:7aa017f0' and cid.startswith('E:') and cid.endswith('value_minus_I')]
        if len(cl) != 1:
            raise RuntimeError(f'{arm}: matching T1 E value_minus_I claim not unique: {[x[0] for x in cl]}')
        cid, (ci, c) = cl[0]
        y = fd.val(f'{arm}.value_minus_I', r['value_minus_I'], FZ_REL, f'{jp}.value_minus_I',
                   rounding='plotted at full precision; annotated k EUR 1 dp')
        fd.check(f'{arm}.T8_equals_T1_claim', c['d_gross'] == r['value_minus_I'], [c['d_gross'], r['value_minus_I']])
        thr = fd.val(f'{arm}.threshold', c['gross_threshold_or_bar'], FZ_REL,
                     f'tables.claims[{ci}].gross_threshold_or_bar (claim {cid})', rounding='plotted (+/- error bar)')
        fy = fd.val(f'{arm}.{FLOOR_KEY}', r[FLOOR_KEY], FZ_REL, f'{jp}.{FLOOR_KEY}', unit='year',
                    rounding='integer; null written "never"')
        fd.val(f'{arm}.verdict', r['verdict'], FZ_REL, f'{jp}.verdict', unit='text', rounding='as recorded')
        fd.check(f'{arm}.verdict_equals_claim', r['verdict'] == c['gross_verdict'], [r['verdict'], c['gross_verdict']])
        out.append({'arm': arm, 'cell': r['cell'], 'y': y, 'thr': thr, 'floor': fy, 'verdict': r['verdict'],
                    'claim': cid, 'baseline': arm == BASELINE_ARM,
                    'unit': unit_desc(cells[r['cell']]['candidate_canonical'])})
    fd.check('one_unit_in_every_arm', len({o['unit'] for o in out}) == 1, sorted({o['unit'] for o in out}))
    m = re.fullmatch(r'floor_year_0(\d+)', FLOOR_KEY)
    floor_level = float('0.' + m.group(1))
    fd.comp('floor_level', floor_level, f'from the T8 column key {FLOOR_KEY}', {'frozen': SRC.ref(FZ_REL,
            'tables.ageing.rows[].' + FLOOR_KEY)}, rounding='2 dp', unit='SoH')
    return fd, {'rows': out, 'unit': out[0]['unit'], 'floor_level': floor_level,
                'bar_factor': SC6.DETERMINACY_BAR_FACTOR, 'tau_mult': SC6.DETERMINACY_TAU_MULTIPLE}


def draw_fig3(fd, D):
    fig = new_fig(fd)
    ax = fig.add_subplot(1, 1, 1)
    n = len(D['rows'])
    ax.axvline(0.0, color=OKABE_ITO['grey'], lw=0.6, zorder=1)
    for j, r in enumerate(D['rows']):
        yy = n - 1 - j
        col = OKABE_ITO['blue'] if r['verdict'] == 'determinate' else OKABE_ITO['vermillion']
        ax.errorbar([_k(r['y'])], [yy], xerr=[[_k(r['thr'])], [_k(r['thr'])]], fmt='o', color=col, ms=3.8,
                    elinewidth=1.0, capsize=2.0, zorder=3)
        ax.annotate(fmt_k(r['y']), (_k(r['y']), yy), xytext=(0, 4), textcoords='offset points', ha='center',
                    va='bottom', fontsize=6.0)
        ax.text(1.02, yy, 'never' if r['floor'] is None else f"{r['floor']}", transform=blended_transform_factory(
            ax.transAxes, ax.transData), ha='left', va='center', fontsize=6.5)
    ax.text(1.02, n - 0.35, f"{D['floor_level']:.2f} floor\nbinds", transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha='left', va='bottom', fontsize=6.0, style='italic')
    ax.set_yticks([n - 1 - j for j in range(n)])
    ax.set_yticklabels([r['arm'] + (' (baseline)' if r['baseline'] else '') for r in D['rows']])
    ax.set_ylim(-1.4, n - 0.2)
    ax.set_xlabel('value − I of the unit (k€)')
    kfmt(ax, 'x')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='none', color=OKABE_ITO['blue'], ms=3.8,
                              label='determinate (± threshold)'),
                       Line2D([], [], marker='o', ls='none', color=OKABE_ITO['vermillion'], ms=3.8,
                              label='within resolution')], loc='lower left', ncol=2, handlelength=1.0,
              columnspacing=1.0, borderaxespad=0.2)
    fd.layout('y-limits extended below the last arm to hold the legend (layout)')
    return fig


# ======================================================================================================================
#  FIGURE 4 -- Q versus cycle (records)
# ======================================================================================================================
def panel_rule(fz):
    """THE RULE, applied to the T2 table order (export/T2.csv). Returns (cell or None, candidates, ambiguity list)."""
    import csv
    with open(os.path.join(REPO, T2_CSV), encoding='utf-8', newline='') as h:
        t2 = list(csv.DictReader(h))
    order = [r['cell'] for r in t2]
    json_order = list(fz['tables']['cells']) + list(fz['tables']['cells_appended_w160'])
    cells = all_cells(fz)
    amb, cands = [], []
    for nm in order:
        c = cells[nm]
        if c.get('status') != 'uncertified':
            continue
        cause = c.get('cause_uncertified') or ''
        gr = c.get('gap_refused')
        is_gap = cause == 'gap clause'
        if ('gap' in cause.lower() and not is_gap) or (gr is True) != is_gap:
            amb.append({'cell': nm, 'cause_uncertified': cause, 'gap_refused': gr})
        if is_gap and gr is True:
            cands.append(nm)
    chosen = cands[0] if (cands and not amb) else None
    return chosen, {'t2_order_equals_json_order': order == json_order, 'gap_refused_rows_in_order': cands,
                    'ambiguous_rows': amb,
                    'uncertified_rows_in_order': [[nm, cells[nm].get('cause_uncertified'), cells[nm].get('gap_refused')]
                                                  for nm in order if cells[nm].get('status') == 'uncertified']}


def data_fig4(fz, cells, chosen, rule_info):
    fd = FigData('fig4', 'q_versus_cycle', 'double', 112.0)
    fd.layout('double column (190 mm): (a) the settled x = 0 reference, Q - Q(k*) vs cycle; (b) the gap-refused cell, '
              'Q - Q(cap) vs cycle; (c) its priced interface-consensus gap t_sum vs cycle with +/- tau/2')
    fd.layout('plotted cycles: k0 .. end of each run (cycles before the first residual pass are not drawn)')
    fd.d['panel_rule'] = {'rule': 'the first row in T2 table order (export/T2.csv) whose recorded uncertified cause is a '
                                  'gap refusal: status uncertified, cause_uncertified == "gap clause", gap_refused true; '
                                  'ambiguous if a cause mentions gap without being exactly "gap clause" or gap_refused '
                                  'disagrees with the cause',
                          'fixed_before_trajectories_read': True, 'chosen': chosen, **rule_info}
    fd.check('panel_rule_T2_order_equals_frozen_json_order', rule_info['t2_order_equals_json_order'])
    fd.check('panel_rule_unambiguous', chosen is not None and not rule_info['ambiguous_rows'],
             rule_info['ambiguous_rows'])
    tau = fz['constants']['TAU']
    fd.check('tau_equals_SC6', tau == SC6.TAU, [tau, SC6.TAU])
    fd.check('tau_equals_SC1', tau == SC1.TAU, [tau, SC1.TAU])
    fd.val('tau', tau, FZ_REL, 'constants.TAU', rounding='full precision (gap bound tau/2 drawn)')
    out = {'tau': tau}
    # ---------------- (a) the settled x = 0 reference ------------------------------------------------------------
    ref = 'ref:7aa017f0'
    cr = cells[ref]
    good, hits = find_eval_dir(cr['eval_key'])
    fd.check('a.eval_dir_unique', len(good) == 1, {'good': good, 'hits': hits})
    if len(good) != 1:
        return fd, None
    ed = good[0]
    root = os.path.dirname(os.path.dirname(ed))
    rec_rel = os.path.join(ed, 'per_cycle_record.jsonl')
    dec_rel = os.path.join(ed, 'settling_decision.json')
    sum_rel = os.path.join(os.path.dirname(root), 'w101_three_reference_summary.json')
    launch = _jl(os.path.join(ed, 'launch.json'))
    fd.check('a.launch_eval_and_candidate_key', launch['eval_key'] == cr['eval_key'] and launch['key'] ==
             cr['candidate_key'], [launch['eval_key'], launch['key']])
    man = _jl(os.path.join(root, 'campaign_manifest_sha256.json'))
    fd.check('a.records_in_campaign_manifest', man.get(rec_rel) == _sha(rec_rel) and man.get(dec_rel) == _sha(dec_rel),
             {rec_rel: man.get(rec_rel), dec_rel: man.get(dec_rel)})
    rows = _rows(rec_rel)
    dec = _jl(dec_rel)
    summ = _jl(sum_rel)['reports']['x0']
    q = {k: r['gross_operational_cost'] for k, r in rows.items()}
    ks = max(q)
    kstar = dec['k_star']
    k0 = dec['k0']
    fd.check('a.contiguous_cycles', sorted(q) == list(range(1, ks + 1)), [min(q), ks, len(q)])
    fd.check('a.k_star_equals_T2', kstar == cr['k_star'] == summ['k_star'], [kstar, cr['k_star'], summ['k_star']])
    fd.check('a.end_equals_T2', ks == cr['end_cycle'] == kstar, [ks, cr['end_cycle']])
    fd.check('a.Q_k_star_equals_T2_Q', q[kstar] == cr['Q'] == dec['Q_k_star'], [q[kstar], cr['Q'], dec['Q_k_star']])
    lo_w, hi_w = dec['window']
    rngw = max(q[k] for k in range(lo_w, hi_w + 1)) - min(q[k] for k in range(lo_w, hi_w + 1))
    fd.check('a.window_range_over_tau_equals_T2', rngw / tau == cr['range_over_tau'] and rngw == cr['band_width'],
             [rngw / tau, cr['range_over_tau'], rngw, cr['band_width']])
    fd.check('a.band_equals_window_min_max', dec['band'] == [min(q[k] for k in range(lo_w, hi_w + 1)),
                                                             max(q[k] for k in range(lo_w, hi_w + 1))], dec['band'])
    fd.check('a.k0_T2_not_recorded_record_and_summary_agree', cr.get('k0_run') is None and k0 == summ['k0'],
             {'T2_k0_run': cr.get('k0_run'), 'decision_k0': k0, 'w101_summary_k0': summ['k0']})
    boyd = {k: bool(r.get('boyd_all_pass')) and bool(r.get('local_solves_ok')) for k, r in rows.items()}
    fd.check('a.k0_is_first_residual_pass_without_lapse', (not boyd[k0 - 1]) and all(boyd[k] for k in range(k0, ks + 1))
             and dec['lapse_events'] == [], {'boyd_k0_minus_1': boyd[k0 - 1], 'all_pass_from_k0': all(
                 boyd[k] for k in range(k0, ks + 1))})
    T = dec['T']
    fd.check('a.turning_points_Q_equal_record', all(q[t] == qt for t, _kind, qt in T) and T == summ['T'],
             [[t, kind] for t, kind, _ in T])
    p_hat = T[-1][0] - T[-3][0]
    W = SC1.window_length(p_hat)
    fd.check('a.window_is_last_W_cycles', p_hat == dec['P_hat'] and W == dec['W'] and [kstar - W + 1, kstar] ==
             dec['window'], {'P_hat': p_hat, 'W': W, 'window': dec['window']})
    N = dec['N']
    fd.check('a.N_and_slack_equal_summary', N == summ['N'] and q[kstar] - q[N] == summ['s_signed'],
             [N, q[kstar] - q[N], summ['s_signed']])
    regime = {}
    cont_rel = os.path.join(ed, 'settling_continuation_cycle_record.jsonl')
    crow = _rows(cont_rel)
    hold_from = min(k for k, r in crow.items() if r.get('holds') and all(r['holds'].values()))
    fd.check('a.holds_continuous_from_hold_start', all(all(crow[k]['holds'].values()) for k in range(hold_from, ks + 1)),
             hold_from)
    fd.check('a.continuation_gross_equals_per_cycle', all(crow[k]['gross'] == q[k] for k in q), None)
    regime = {'hold_from_cycle': hold_from, 'phase_at_hold_from_minus_1': crow[hold_from - 1].get('phase'),
              'phase_at_hold_from': crow[hold_from].get('phase')}
    for k in range(k0, ks + 1):
        fd.val(f'a.Q[{k}]', q[k], rec_rel, f'cycle {k}.gross_operational_cost',
               rounding='plotted as (Q - Q(k*)) / 1000 at full precision')
    fd.val('a.k0', k0, dec_rel, 'k0', unit='cycle')
    fd.val('a.k_star', kstar, dec_rel, 'k_star', unit='cycle')
    fd.val('a.N', N, dec_rel, 'N', unit='cycle', note='the earlier (residual-rule) certificate; the replay ran 1..N')
    fd.val('a.window', dec['window'], dec_rel, 'window', unit='cycle')
    fd.val('a.band', dec['band'], dec_rel, 'band')
    fd.val('a.T', T, dec_rel, 'T', unit='[cycle, kind, Q]')
    fd.val('a.range_over_tau', cr['range_over_tau'], FZ_REL, f'tables.cells.{ref}.range_over_tau', unit='ratio',
           rounding='annotated 2 dp')
    fd.val('a.hold_from', hold_from, cont_rel, 'first cycle with holds.{aa, tail_apply, tail_next, rho} all true',
           unit='cycle')
    fd.d['records_a'] = {'eval_dir': ed, 'per_cycle_record': SRC.ref(rec_rel, '*'),
                         'settling_decision': SRC.ref(dec_rel, '*'), 'w101_summary': SRC.ref(sum_rel, 'reports.x0'),
                         'continuation_record': SRC.ref(cont_rel, 'holds, gross'),
                         'certifying_rule': 'settling_criterion version 1 (W101; stage spec v39), not version 6',
                         'regime': regime}
    out['a'] = {'q': q, 'k0': k0, 'kstar': kstar, 'N': N, 'window': dec['window'], 'band': dec['band'], 'T': T,
                'range_over_tau': cr['range_over_tau'], 'hold_from': hold_from, 'cell': ref, 'Q_ref': q[kstar]}
    # ---------------- (b, c) the gap-refused cell -------------------------------------------------------------------
    if chosen is None:
        return fd, None
    cu = cells[chosen]
    good, hits = find_eval_dir(cu['eval_key'])
    fd.check('b.eval_dir_unique', len(good) == 1, {'good': good, 'hits': hits})
    if len(good) != 1:
        return fd, None
    ed = good[0]
    root = os.path.dirname(os.path.dirname(ed))
    rc_rel = os.path.join(ed, 'resettle_cycle_record.jsonl')
    pc_rel = os.path.join(ed, 'per_cycle_record.jsonl')
    dd_rel = os.path.join(ed, 'resettle_decision.json')
    launch = _jl(os.path.join(ed, 'launch.json'))
    fd.check('b.launch_eval_and_candidate_key', launch['eval_key'] == cu['eval_key'] and launch['key'] ==
             cu['candidate_key'], [launch['eval_key'], launch['key']])
    man = _jl(os.path.join(root, 'campaign_manifest_sha256.json'))
    fd.check('b.records_in_campaign_manifest', all(man.get(r) == _sha(r) for r in (rc_rel, pc_rel, dd_rel)),
             {r: man.get(r) for r in (rc_rel, pc_rel, dd_rel)})
    rr = _rows(rc_rel)
    pr = _rows(pc_rel)
    dd = _jl(dd_rel)
    qb = {k: r['gross'] for k, r in rr.items()}
    tb = {k: r.get('t_sum') for k, r in rr.items()}
    kc = max(qb)
    fd.check('b.contiguous_cycles', sorted(qb) == list(range(1, kc + 1)) and sorted(pr) == sorted(qb), [kc, len(qb)])
    fd.check('b.resettle_gross_equals_per_cycle_record', all(pr[k]['gross_operational_cost'] == qb[k] for k in qb))
    fd.check('b.status_uncertified_gap_clause', dd['status'] == 'uncertified' and dd['reasons'] == ['gap_clause'] and
             dd['gap_clause_refused_at_cap'] is True and cu['k_star'] is None, [dd['status'], dd['reasons']])
    fd.check('b.end_equals_T2', kc == cu['end_cycle'] == cu['cap'] == dd['k_cap'] == dd['decided_at_cycle'],
             [kc, cu['end_cycle'], cu['cap'], dd['k_cap']])
    fd.check('b.Q_cap_equals_T2_Q', qb[kc] == cu['Q'] == dd['Q_at_cap'], [qb[kc], cu['Q'], dd['Q_at_cap']])
    fd.check('b.t_sum_cap_equals_T2', tb[kc] == cu['t_sum_end'] == dd['t_sum_at_cap'] and abs(tb[kc]) == cu['gap'],
             [tb[kc], cu['t_sum_end'], cu['gap']])
    fd.check('b.Q_cc_cap_equals_T2', qb[kc] + tb[kc] == cu['Q_cc'] or rr[kc]['Q_cc'] == cu['Q_cc'],
             [rr[kc].get('Q_cc'), cu['Q_cc']])
    k0b = dd['k0']
    fd.check('b.k0_equals_T2', k0b == cu['k0_run'] == dd['first_residual_pass_run'] == dd['N'],
             [k0b, cu['k0_run'], dd['first_residual_pass_run']])
    boydb = {k: bool(r.get('boyd_k')) for k, r in rr.items()}
    fd.check('b.k0_is_first_residual_pass', all(not boydb[k] for k in range(1, k0b)) and boydb[k0b], k0b)
    fd.check('b.no_lapse_after_k0', all(boydb[k] for k in range(k0b, kc + 1)) and dd['lapse_events'] == [], None)
    Tb = dd['T']
    fd.check('b.turning_points_equal_T2', [t for t, _k2, _q in Tb] == cu['turning_points'] and
             all(qb[t] == qt for t, _k2, qt in Tb), [[t, kind] for t, kind, _ in Tb])
    s_signed = qb[kc] - dd['Q_N_old_recorded']
    fd.check('b.slack_equals_T2', s_signed == cu['s_signed'] and abs(s_signed) == cu['slack'],
             [s_signed, cu['s_signed'], cu['slack']])
    blo, bhi = dd['band_window']
    brng = max(qb[k] for k in range(blo, bhi + 1)) - min(qb[k] for k in range(blo, bhi + 1))
    fd.check('b.band_width_equals_T2', brng == dd['band_width'] == cu['band_width'],
             [brng, dd['band_width'], cu['band_width']])
    refusals = dd['gap_refusals']
    m = re.search(r'gap clause \((\d+) refusals', ' '.join(cu.get('contributing') or []))
    fd.check('b.refusal_count_equals_T2_contributing', m is not None and int(m.group(1)) == len(refusals),
             [m.group(1) if m else None, len(refusals)])
    fd.check('b.refusals_t_sum_equal_record_and_exceed_bound',
             all(tb[r['cycle']] == r['t_sum'] and abs(r['t_sum']) > dd['gap_bound'] for r in refusals), None)
    fd.check('b.gap_bound_is_tau_over_2', dd['gap_bound'] == SC6.GAP_BOUND and abs(dd['gap_bound'] - tau / 2) <= 1e-9,
             [dd['gap_bound'], tau / 2])
    holdb = min(k for k, r in rr.items() if r.get('holds') and all(r['holds'].values()))
    fd.check('b.holds_continuous_from_hold_start', all(all(rr[k]['holds'].values()) for k in range(holdb, kc + 1)),
             holdb)
    for k in range(k0b, kc + 1):
        fd.val(f'b.Q[{k}]', qb[k], rc_rel, f'cycle {k}.gross', rounding='plotted as (Q - Q(cap)) / 1000')
        fd.val(f'c.t_sum[{k}]', tb[k], rc_rel, f'cycle {k}.t_sum', rounding='plotted as t_sum / 1000')
    fd.val('b.k0', k0b, dd_rel, 'k0', unit='cycle')
    fd.val('b.cap', kc, dd_rel, 'k_cap', unit='cycle')
    fd.val('b.N_old', dd['N_old'], dd_rel, 'N_old', unit='cycle', rounding='caption only; NOT drawn (a cycle of the '
           'earlier run: this run replayed that run bitwise only through k0)',
           note='the earlier run\'s certificate; the slack is Q(cap) - Q_N_old_recorded of THAT run')
    fd.val('b.Q_N_old_recorded', dd['Q_N_old_recorded'], dd_rel, 'Q_N_old_recorded',
           note='recorded from the earlier run; not a point of this trajectory')
    fd.val('b.T', Tb, dd_rel, 'T', unit='[cycle, kind, Q]')
    fd.val('b.band_window', dd['band_window'], dd_rel, 'band_window', unit='cycle')
    fd.val('b.band', dd['band'], dd_rel, 'band')
    fd.val('c.gap_bound', dd['gap_bound'], dd_rel, 'gap_bound', rounding='drawn at +/- gap_bound / 1000')
    fd.val('c.gap_refusal_cycles', [r['cycle'] for r in refusals], dd_rel, 'gap_refusals[].cycle', unit='cycle')
    fd.val('b.gap', cu['gap'], FZ_REL, f'tables.cells.{chosen}.gap', rounding='caption k EUR 1 dp')
    fd.val('b.slack', cu['slack'], FZ_REL, f'tables.cells.{chosen}.slack', rounding='caption k EUR 1 dp')
    fd.val('b.label', cu['label'], FZ_REL, f'tables.cells.{chosen}.label', unit='text', rounding='as recorded')
    fd.d['records_b'] = {'cell': chosen, 'eval_dir': ed, 'resettle_cycle_record': SRC.ref(rc_rel, '*'),
                         'per_cycle_record': SRC.ref(pc_rel, '*'), 'resettle_decision': SRC.ref(dd_rel, '*'),
                         'certifying_rule': f"settling_criterion version {dd['version']} (W142 v6 campaign)",
                         'hold_from_cycle': holdb}
    out['b'] = {'q': qb, 't': tb, 'k0': k0b, 'cap': kc, 'N_old': dd['N_old'], 'T': Tb, 'band_window': dd['band_window'],
                'band': dd['band'], 'gap_bound': dd['gap_bound'], 'refusals': [r['cycle'] for r in refusals],
                'cell': chosen, 'gap': cu['gap'], 'slack': cu['slack'], 'hold_from': holdb,
                'candidate': cu['candidate_canonical'], 'unit': unit_desc(cu['candidate_canonical']),
                'm': multiplier_of_cell(cu, harness_default_multiplier()[0], fd, f'b.{chosen}')[0],
                'gap_div': round(tau / dd['gap_bound'])}
    fd.check('b.gap_bound_divisor_integer', abs(tau / dd['gap_bound'] - out['b']['gap_div']) <= 1e-12,
             tau / dd['gap_bound'])
    return fd, out


def _vline(ax, x, text, color, ls='-', y=0.98, ha='left'):
    ax.axvline(x, color=color, lw=0.7, ls=ls, zorder=1)
    ax.text(x + (0.6 if ha == 'left' else -0.6), y, text, transform=blended_transform_factory(ax.transData,
                                                                                            ax.transAxes),
            ha=ha, va='top', fontsize=6.0, color=color)


def draw_fig4(fd, D):
    fig = new_fig(fd)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1], sharex=axb)
    # (a)
    a = D['a']
    ks = list(range(a['k0'], a['kstar'] + 1))
    ya = [_k(a['q'][k] - a['Q_ref']) for k in ks]
    lo, hi = a['window']
    axa.axvspan(lo - 0.5, hi + 0.5, color=OKABE_ITO['sky'], alpha=0.18, lw=0, zorder=0)
    axa.plot(ks, ya, '-', color=OKABE_ITO['black'], lw=0.8, zorder=3)
    axa.plot(ks, ya, '.', color=OKABE_ITO['black'], ms=2.0, zorder=3)
    axa.hlines([_k(a['band'][0] - a['Q_ref']), _k(a['band'][1] - a['Q_ref'])], lo - 0.5, hi + 0.5,
               colors=OKABE_ITO['blue'], linestyles='--', lw=0.7, zorder=2)
    axa.plot([t for t, _kd, _q in a['T']], [_k(qt - a['Q_ref']) for _t, _kd, qt in a['T']], 'o', mfc='white',
             mec=OKABE_ITO['vermillion'], mew=1.0, ms=4.5, zorder=4, label='turning points')
    _vline(axa, a['k0'], '$k_0$', OKABE_ITO['green'])
    _vline(axa, a['N'], 'N', OKABE_ITO['grey'], ls=':')
    _vline(axa, a['kstar'], 'k*', OKABE_ITO['vermillion'], ha='right')
    axa.text((lo + hi) / 2, 0.06, f"window ({hi - lo + 1} cycles)\nrange/τ = {a['range_over_tau']:.2f}",
             transform=blended_transform_factory(axa.transData, axa.transAxes), ha='center', va='bottom',
             fontsize=6.0, color=OKABE_ITO['blue'])
    axa.set_xlim(a['k0'] - 1, a['kstar'] + 1)
    axa.set_xlabel('ADMM cycle')
    axa.set_ylabel('Q − Q(k*) (k€)')
    kfmt(axa, 'y')
    axa.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), handlelength=1.0, borderaxespad=0.2)
    panel_label(axa, '(a)')
    # (b)
    b = D['b']
    kb = list(range(b['k0'], b['cap'] + 1))
    qref = b['q'][b['cap']]
    yb = [_k(b['q'][k] - qref) for k in kb]
    blo, bhi = b['band_window']
    axb.axvspan(blo - 0.5, bhi + 0.5, color=OKABE_ITO['grey'], alpha=0.15, lw=0, zorder=0)
    axb.plot(kb, yb, '-', color=OKABE_ITO['black'], lw=0.8, zorder=3)
    axb.plot(kb, yb, '.', color=OKABE_ITO['black'], ms=2.0, zorder=3)
    axb.plot([t for t, _kd, _q in b['T']], [_k(qt - qref) for _t, _kd, qt in b['T']], 'o', mfc='white',
             mec=OKABE_ITO['vermillion'], mew=1.0, ms=4.5, zorder=4)
    _vline(axb, b['k0'], '$k_0$', OKABE_ITO['green'])
    _vline(axb, b['cap'], 'cap', OKABE_ITO['black'], ha='right')
    axb.text((blo + bhi) / 2, 0.80, f'last {bhi - blo + 1} cycles at the cap', ha='center', va='bottom',
             transform=blended_transform_factory(axb.transData, axb.transAxes), fontsize=6.0,
             color=OKABE_ITO['grey'])
    axb.set_ylabel('Q − Q(cap) (k€)')
    kfmt(axb, 'y')
    plt.setp(axb.get_xticklabels(), visible=False)
    panel_label(axb, '(b)')
    # (c)
    tc = [_k(b['t'][k]) for k in kb]
    gb = _k(b['gap_bound'])
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
    kfmt(axc, 'y')
    axc.legend(loc='lower right', bbox_to_anchor=(1.0, 0.07), handlelength=0.8, borderaxespad=0.2)
    panel_label(axc, '(c)')
    return fig


# ======================================================================================================================
#  FIGURE 5 -- benchmark (T6 + W130 decomposition)
# ======================================================================================================================
def data_fig5(fz, w130, w130_man):
    fd = FigData('fig5', 'benchmark', 'double', 56.0)
    fd.layout('double column (190 mm): (a) Q of the three arrangements (best start for each NRF arm); '
              '(b) the price-taker benefit split by agent (TSO / DSO at nodes 5, 7, 9) from W130')
    bm = fz['tables']['benchmark']
    jp = 'tables.benchmark'
    qc = fd.val('coordinated_Q', bm['coordinated_Q'], FZ_REL, f'{jp}.coordinated_Q', rounding='bar at full precision; '
                'annotated k EUR 1 dp')
    qp = fd.val('passive_NRF_Q_best', bm['arms']['passive']['Q_best'], FZ_REL, f'{jp}.arms.passive.Q_best',
                rounding='bar at full precision; annotated k EUR 1 dp')
    qt = fd.val('price_taker_NRF_Q_best', bm['arms']['price_taker']['Q_best'], FZ_REL, f'{jp}.arms.price_taker.Q_best',
                rounding='bar at full precision; annotated k EUR 1 dp')
    ben = fd.val('benefit', bm['benefit'], FZ_REL, f'{jp}.benefit', rounding='annotated k EUR 1 dp')
    rel = fd.val('benefit_relative', bm['benefit_relative'], FZ_REL, f'{jp}.benefit_relative', unit='fraction',
                 rounding='annotated % 1 dp')
    dpas = fd.val('passive_minus_coordinated', bm['decomposition']['passive_NRF_minus_coordinated_eur'], FZ_REL,
                  f'{jp}.decomposition.passive_NRF_minus_coordinated_eur', rounding='annotated k EUR 1 dp')
    pas = bm['w160_additions']['passive_arm_vs_coordinated_derived']
    relp = fd.val('passive_benefit_relative_derived_W160', pas['benefit_passive_relative'], FZ_REL,
                  f'{jp}.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative', unit='fraction',
                  rounding='annotated % 1 dp', note='DERIVED by W160 from recorded values (T6 label)')
    fd.check('benefit_is_price_taker_minus_coordinated', bm['decomposition']['price_taker_NRF_minus_coordinated_eur']
             == ben and abs((qt - qc) - ben) <= 1e-6, [qt - qc, ben])
    fd.check('price_taker_is_best_arm', qt < qp, [qt, qp])
    # W130 decomposition (named record; not in T6)
    tot = w130['task1']['totals']['benefit_price_taker']
    comps = {}
    for kk in ('TSO', 'DSO_5', 'DSO_7', 'DSO_9', 'DSO', 'total'):
        comps[kk] = fd.val(f'W130.benefit_price_taker.{kk}', tot[kk], W130_JSON,
                           f'task1.totals.benefit_price_taker.{kk}', rounding='bar at full precision; annotated k EUR '
                           '1 dp')
    fd.check('w130_in_its_manifest', any(k.startswith(W130_JSON) and v == _sha(W130_JSON) for k, v in w130_man.items()),
             [k for k in w130_man if k.startswith(W130_JSON)])
    fd.check('w130_total_equals_T6_benefit', abs(comps['total'] - ben) <= 1e-6, comps['total'] - ben)
    fd.check('w130_parts_sum_to_total', abs(comps['TSO'] + comps['DSO'] - comps['total']) <= 1e-6 and
             abs(comps['DSO_5'] + comps['DSO_7'] + comps['DSO_9'] - comps['DSO']) <= 1e-6, None)
    qps = w130['task1']['arm_Q_per_start']
    t6s = {'passive': bm['w160_additions']['arms_in_full']['passive']['Q_by_start'],
           'price_taker': bm['w160_additions']['arms_in_full']['price_taker']['Q_by_start']}
    fd.check('w130_arm_Q_per_start_equal_T6_bitwise', qps == t6s, None)
    fd.check('w130_coordinated_total_close_to_T6', abs(w130['task1']['totals']['coordinated']['total'] - qc) <= 1e-6,
             w130['task1']['totals']['coordinated']['total'] - qc)
    share_tso = fd.comp('share_TSO', comps['TSO'] / comps['total'], 'TSO / total (W130 totals.benefit_price_taker)',
                        {'W130': SRC.ref(W130_JSON, 'task1.totals.benefit_price_taker')}, rounding='% 1 dp',
                        unit='fraction')
    share_dso = fd.comp('share_DSO', comps['DSO'] / comps['total'], 'DSO / total (W130 totals.benefit_price_taker)',
                        {'W130': SRC.ref(W130_JSON, 'task1.totals.benefit_price_taker')}, rounding='% 1 dp',
                        unit='fraction')
    fd.cap('coordinated_band', bm['coordinated_reproducibility_band'], FZ_REL,
           f'{jp}.coordinated_reproducibility_band', fmt_k(bm['coordinated_reproducibility_band']))
    fd.cap('passive_band', bm['arms']['passive']['multimodality_band'], FZ_REL, f'{jp}.arms.passive.multimodality_band',
           fmt_k(bm['arms']['passive']['multimodality_band']))
    fd.cap('price_taker_band', bm['arms']['price_taker']['multimodality_band'], FZ_REL,
           f'{jp}.arms.price_taker.multimodality_band', fmt_k(bm['arms']['price_taker']['multimodality_band']))
    fd.cap('multiple', bm['multiple'], FZ_REL, f'{jp}.multiple', f"{bm['multiple']:,.0f}", unit='x')
    fd.cap('reverse_flow_interface_hours', bm['reverse_flow_interface_hours'], FZ_REL,
           f'{jp}.reverse_flow_interface_hours', f"{bm['reverse_flow_interface_hours']}", unit='hours')
    sw = bm['sweep']
    fd.cap('sweep_passive_blocks', sw['sweep_passive_cold']['n_blocks'], FZ_REL, f'{jp}.sweep.sweep_passive_cold.n_blocks',
           f"{sw['sweep_passive_cold']['n_blocks']}", unit='blocks of 12')
    fd.cap('sweep_price_taker_blocks', sw['sweep_price_taker_cold']['n_blocks'], FZ_REL,
           f'{jp}.sweep.sweep_price_taker_cold.n_blocks', f"{sw['sweep_price_taker_cold']['n_blocks']}",
           unit='blocks of 12')
    tw = bm['w160_additions']['price_taker_best_start_TN_curtailment_TWh_day_weighted']
    fd.cap('price_taker_TN_curtailment_TWh', tw, FZ_REL,
           f'{jp}.w160_additions.price_taker_best_start_TN_curtailment_TWh_day_weighted', f'{tw:.2f}', unit='TWh')
    af = bm['w160_additions']['arms_in_full']
    n_starts = {k: af[k]['n_starts'] for k in ('passive', 'price_taker')}
    fd.check('both_arms_same_number_of_starts', len(set(n_starts.values())) == 1, n_starts)
    fd.val('n_starts', n_starts['passive'], FZ_REL, f'{jp}.w160_additions.arms_in_full.passive.n_starts',
           unit='starts', rounding='integer (caption)')
    n_tot = {k: int(re.search(r'in \d+ of (\d+) blocks', af[k]['sweep_unconstrained_cold']['statement']).group(1))
             for k in ('passive', 'price_taker')}
    fd.check('sweep_block_total_one_value', len(set(n_tot.values())) == 1, n_tot)
    fd.val('sweep_blocks_total', n_tot['passive'], FZ_REL, f'{jp}.w160_additions.arms_in_full.passive.'
           'sweep_unconstrained_cold.statement (parsed "in n of N blocks")', unit='blocks', rounding='integer (caption)')
    return fd, {'qc': qc, 'qp': qp, 'qt': qt, 'ben': ben, 'rel': rel, 'dpas': dpas, 'relp': relp, 'comps': comps,
                'n_starts': n_starts['passive'], 'n_blocks': n_tot['passive'],
                'share_tso': share_tso, 'share_dso': share_dso,
                'best_start': {'passive': bm['arms']['passive']['best_start'],
                               'price_taker': bm['arms']['price_taker']['best_start']}}


def draw_fig5(fd, D):
    fig = new_fig(fd)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    labels = ['passive NRF', 'price-taker NRF', 'coordinated']
    vals = [D['qp'], D['qt'], D['qc']]
    cols = [OKABE_ITO['grey'], OKABE_ITO['orange'], OKABE_ITO['blue']]
    ys = [2, 1, 0]
    axa.barh(ys, [_k(v) for v in vals], color=cols, height=0.55, zorder=2)
    for y, v in zip(ys, vals):
        axa.text(_k(v) * 0.985, y, fmt_k(v), ha='right', va='center', fontsize=6.0, color='white', zorder=3)
    axa.set_yticks(ys)
    axa.set_yticklabels(labels)
    axa.set_xlabel('Q, gross operational cost (k€)')
    kfmt(axa, 'x')
    xmax = _k(D['qp']) * 1.30
    axa.set_xlim(0, xmax)
    for y_from, v_from, txt in ((1, D['qt'], f"+{fmt_k(D['ben'])} k€ ({100 * D['rel']:.1f} %)"),
                                (2, D['qp'], f"+{fmt_k(D['dpas'])} k€ ({100 * D['relp']:.1f} %)")):
        axa.text(_k(v_from) + xmax * 0.015, y_from, txt + '\nvs coordinated', ha='left', va='center', fontsize=6.0)
    panel_label(axa, '(a)')
    fd.layout('panel (a) x-limit 1.30 x the largest Q (room for the difference labels; layout)')
    # (b)
    c = D['comps']
    dso_keys = sorted(k for k in c if k.startswith('DSO_'))
    parts = [('TSO', 'TSO: TN conventional energy', OKABE_ITO['orange'])] + [
        (k, f"DSO node {k.split('_')[1]}", col) for k, col in zip(dso_keys, (OKABE_ITO['sky'], OKABE_ITO['green'],
                                                                           OKABE_ITO['purple']))]
    if len(dso_keys) != 3:
        raise RuntimeError(f'figure 5 expects three DSO components, found {dso_keys}')
    left = 0.0
    for kk, lab, col in parts:
        w = _k(c[kk])
        axb.barh([0], [w], left=left, color=col, height=0.5, zorder=2, label=f'{lab}: {fmt_k(c[kk])} k€')
        left += w
    axb.text(_k(c['TSO']) / 2, 0, f"{100 * D['share_tso']:.1f} %", ha='center', va='center', fontsize=6.5,
             color='black', zorder=3)
    axb.text(_k(c['TSO']) + _k(c['DSO']) / 2, 0.42, f"DSO (DN flexibility)\n{100 * D['share_dso']:.1f} %",
             ha='center', va='bottom', fontsize=6.0)
    axb.set_xlim(0, _k(c['total']) * 1.02)
    axb.set_ylim(-1.7, 1.0)
    axb.set_yticks([])
    axb.spines['left'].set_visible(False)
    axb.set_xlabel(f"benefit of coordination over price-taker NRF (k€; total {fmt_k(c['total'])})")
    kfmt(axb, 'x')
    axb.legend(loc='lower center', ncol=2, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)
    panel_label(axb, '(b)')
    return fig


# ======================================================================================================================
#  rendering, determinism, font checks
# ======================================================================================================================
def render(draw, fd, D):
    set_style()
    fig = draw(fd, D)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata=PDF_METADATA)
    png = io.BytesIO()
    fig.savefig(png, format='png', dpi=PREVIEW_DPI, metadata={'Software': None})
    plt.close(fig)
    return buf.getvalue(), png.getvalue()


def pdf_font_report(b):
    names = sorted(set(m.decode('latin-1') for m in re.findall(rb'/BaseFont\s*/([A-Za-z0-9+\-_,]+)', b)))
    return {'base_fonts': names, 'n_FontFile2': b.count(b'/FontFile2'), 'n_FontFile3': b.count(b'/FontFile3'),
            'n_FontFile': b.count(b'/FontFile ') + b.count(b'/FontFile\n'), 'type3': b.count(b'/Type3'),
            'has_creation_date': b'/CreationDate' in b, 'has_mod_date': b'/ModDate' in b}


def caption_text(fid, fd, D, extra):
    """Draft caption. Every number is formatted from the figure's data (or the frozen JSON via `extra`)."""
    s = []
    if fid == 'fig1':
        f_c, f_b = D['fits']['certified_only'], D['fits']['banded_mid']
        h1, h2 = D['durations']
        s.append(f"**Figure 1 -- break-even of the energy cost (node {D['node']}).** Value minus investment cost of "
                 f"the storage plans at node {D['node']} (investment year {D['year']}) against installed energy E: "
                 f"(a) {h1:g} h plans (P = E/{h1:g}), (b) {h2:g} h plans (P = E/{h2:g}). Filled circles: certified "
                 f"evaluations (vertical bar: the band width, the range of Q over the certifying window; not visible "
                 f"at this scale). Open squares: evaluations uncertified at their cap, with their uncertified bar "
                 f"{D['unc_factor']} x max(|t_sum|, |slack|) (thick) and the interval the banded fit uses, bar + "
                 f"{D['tau_mult']:g} tau (thin; tau = {extra['tau']} EUR). Open diamond: a certified evaluation "
                 f"entered as an interval in this conservative fit (its certificate is flagged: a turning point on a "
                 f"non-clean cycle). Lines: the affine fit value = a + bE + cP evaluated at the panel's duration, on "
                 f"the certified points only (n = {f_c['n']}) and on all points with the intervals at their midpoints "
                 f"(n = {f_b['n']}). (c) The break-even energy cost e* = b + c/{h1:g} - p/{h1:g} (p: the power cost "
                 f"per MVA) of each fit -- certified-only {fmt_kmwh(f_c['e_star'])} k EUR/MWh; banded "
                 f"{fmt_kmwh(D['rng'][0])}-{fmt_kmwh(D['rng'][1])} k EUR/MWh over every corner of the intervals -- "
                 f"against the energy cost {fmt_kmwh(D['e_cost'])} k EUR/MWh: the margin is at least "
                 f"{fmt_kmwh(D['mrg'][0])} k EUR/MWh. With the flagged certificate kept as certified the banded range "
                 f"is {extra['w145_range']} k EUR/MWh. {OBJECTIVE_CAPTION}")
        s.append(f"*Sources:* frozen tables `frozen_step6_tables_v1_590088fe.json` (sha256 {FZ_SHA[:8]}): T3 "
                 f"(tables.break_even.conservative_A64, the manuscript figure), T1 B rows (value - I = -d_gross), T2 "
                 f"(E, P, band). The fit intercept a is not tabulated; it is computed by the recorded W145 fit "
                 f"function on the committed W145 points (`w145_banded_fit.json`), a call that reproduces every T3 "
                 f"coefficient and range bitwise.")
    elif fid == 'fig2':
        r = {x['claim']: x for x in D['rows']}
        base = r['CHECK:headline_V_minus_I_settled']
        unc = [x for x in D['rows'] if x['uncertified_form']]
        pays = [x for x in D['rows'] if x['verdict'] == 'determinate' and x['y'] > 0]
        s.append(f"**Figure 2 -- flexibility-price ladder.** Value minus investment cost of the smallest unit "
                 f"({D['unit']}) against the multiplier m of the base flexibility price; value = Q(0) - Q(unit), both "
                 f"evaluated at the same m. Error bars: the determinacy threshold of each difference between "
                 f"certified evaluations, max({D['bar_factor']:g} x larger band, {D['tau_mult']:g} tau), or the "
                 f"uncertified bar where one evaluation is uncertified (open square"
                 + (f", m = {unc[0]['m']:g}: the unit's evaluation was refused by the gap clause" if unc else '')
                 + "). A difference is determinate when its bar does not reach zero. The unit pays at m = "
                 + ' and '.join(f"{x['m']:g}" for x in pays) + ' ('
                 + '; '.join(f"+{fmt_k(x['y'])} k EUR, {x['multiple']:.2f} x its threshold" for x in pays)
                 + f") and loses {fmt_k(-base['y'])} k EUR at the base price ({base['multiple']:.2f} x)."
                 f" {OBJECTIVE_CAPTION}")
        s.append(f"*Sources:* T1 rows CHECK:headline_V_minus_I_settled (base price), H:m1.5, H:m2 and T10 row "
                 f"H:m1.75 of `frozen_step6_tables_v1_590088fe.json` (sha256 {FZ_SHA[:8]}); the multiplier of each "
                 f"cell from its campaign spec (`candidates[].flex_price_multiplier`; absent = the base price, the "
                 f"harness default).")
    elif fid == 'fig3':
        na = [x for x in D['rows'] if x['arm'] == 'no_ageing'][0]
        aged = [x for x in D['rows'] if x['arm'] != 'no_ageing']
        lo, hi = min(-x['y'] for x in aged), max(-x['y'] for x in aged)
        s.append(f"**Figure 3 -- ageing arms.** Value minus investment cost of the smallest unit ({D['unit']}) "
                 f"under each ageing calibration, the plan held fixed; error bars: the determinacy threshold "
                 f"max({D['bar_factor']:g} x larger band, {D['tau_mult']:g} tau) of the difference against x = 0. "
                 f"Right: the first year in which the {D['floor_level']:.2f} state-of-health floor binds (never: not "
                 f"within the horizon). Without ageing the unit is at break-even ({fmt_k(na['y'])} k EUR, "
                 f"{na['verdict']}); under every aged arm it loses {fmt_k(lo)}-{fmt_k(hi)} k EUR, "
                 + ('determinately' if all(x['verdict'] == 'determinate' for x in aged) else 'not all determinately')
                 + f". {OBJECTIVE_CAPTION}")
        s.append(f"*Sources:* T8 (tables.ageing.rows) and the matching T1 E rows (threshold) of "
                 f"`frozen_step6_tables_v1_590088fe.json` (sha256 {FZ_SHA[:8]}); the baseline arm as named in "
                 f"STEP6_REVISION_MAP.md section A.")
    elif fid == 'fig4':
        a, b = D['a'], D['b']
        s.append("**Figure 4 -- objective versus ADMM cycle.** (a) A certified evaluation, the settled x = 0 "
                 f"reference: Q - Q(k*) from the first residual pass k0 = {a['k0']} to the certification cycle "
                 f"k* = {a['kstar']}; circles: the turning points; shaded: the certifying window (the last "
                 f"{a['window'][1] - a['window'][0] + 1} cycles), whose range (dashed) is "
                 f"{a['range_over_tau']:.2f} tau. N = {a['N']}: the cycle at which the residual-based rule had "
                 f"stopped this evaluation; Q moved {fmt_k(a['q'][a['kstar']] - a['q'][a['N']])} k EUR after it. This "
                 f"evaluation replayed its recorded run bitwise to N and held the certifying regime from cycle "
                 f"{a['hold_from']}. (b) An evaluation refused by the gap clause (the unit, {b['unit']}, at "
                 f"{b['m']:g} x the base flexibility price): Q - Q(cap) from k0 = {b['k0']} to the cap {b['cap']}; "
                 f"the objective settles, (c) but the priced interface-consensus gap t_sum stays outside "
                 f"+/- tau/{b['gap_div']} (ticks: the {len(b['refusals'])} cycles at which every other clause held and "
                 f"the gap clause refused). It is reported in the uncertified form: gap {fmt_k(b['gap'])} k EUR, "
                 f"slack {fmt_k(b['slack'])} k EUR from the certificate of the earlier run of the same evaluation (its "
                 f"cycle {b['N_old']}). " + OBJECTIVE_CAPTION)
        s.append(f"*Sources:* (a) `{fd.d['records_a']['per_cycle_record']['path']}` (sha256 "
                 f"{fd.d['records_a']['per_cycle_record']['sha256'][:8]}) and `settling_decision.json` (sha256 "
                 f"{fd.d['records_a']['settling_decision']['sha256'][:8]}); certified under settling-rule version 1 "
                 f"(the reference continuation). (b, c) `{fd.d['records_b']['resettle_cycle_record']['path']}` "
                 f"(sha256 {fd.d['records_b']['resettle_cycle_record']['sha256'][:8]}) and `resettle_decision.json` "
                 f"(sha256 {fd.d['records_b']['resettle_decision']['sha256'][:8]}); rule version 6. Both trajectories "
                 f"reproduce their T2 rows of `frozen_step6_tables_v1_590088fe.json` exactly. Panel (b) cell chosen by "
                 f"the rule stated before the trajectories were read: the first T2 row whose recorded uncertified "
                 f"cause is a gap refusal ({b['cell']}).")
    elif fid == 'fig5':
        s.append("**Figure 5 -- value of coordination.** (a) Gross operational cost Q at x = 0 of the two static "
                 f"no-reverse-flow (NRF) arrangements (each the best of {D['n_starts']} starts) and of coordinated "
                 "operation (the settled x = 0 evaluation); labels: the difference to coordinated operation. "
                 f"Coordination beats the best static arrangement (price-taker NRF) by {fmt_k(D['ben'])} k EUR "
                 f"({100 * D['rel']:.1f} %), {extra['multiple']} times the larger band ({extra['cband']} k EUR). "
                 f"(b) Where the {fmt_k(D['ben'])} k EUR comes from, by agent: the transmission system "
                 f"({100 * D['share_tso']:.1f} %: conventional energy, with {extra['tw']} TWh of transmission "
                 f"renewables left curtailed by the price-taker arrangement) and the distribution systems' flexibility "
                 f"({100 * D['share_dso']:.1f} %). Without any interface rule the transmission network cannot accept "
                 f"the distribution exchange in {extra['sp']} (passive) and {extra['st']} (price-taker) of "
                 f"{D['n_blocks']} blocks; coordinated operation uses reverse flow in {extra['rf']} interface-hours. "
                 + OBJECTIVE_CAPTION)
        s.append(f"*Sources:* T6 (tables.benchmark) of `frozen_step6_tables_v1_590088fe.json` (sha256 {FZ_SHA[:8]}); "
                 f"the agent split from `{W130_JSON}` (commit {W130_COMMIT}, task1.totals.benefit_price_taker), "
                 f"cross-checked against T6; the reading of the transmission term as conventional energy and of the "
                 f"distribution term as flexibility is PLANNER_BRIEF_2026-09-13.md Addendum 58's. The passive arm's "
                 f"relative benefit is derived (T6 label).")
    return '\n\n'.join(s)


# ======================================================================================================================
#  main
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W157.pickle_state()
    counts = dict(base['counts'], w165=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, f) for f in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W165 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_POST), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[W165 post-run] wrote {os.path.join(out_dir, OUT_POST)}: ' + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--out-dir', default=None, help='trial runs only: write every output under this directory')
    args = ap.parse_args()
    trial = args.out_dir is not None
    out_dir = args.out_dir if trial else OUT_DIR
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W165'
    # ---- preconditions -----------------------------------------------------------------------------------------
    pre = []
    od = os.path.join(REPO, out_dir)
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (create it before the launch; the launch log goes there)')
    else:
        present = sorted(os.listdir(od))
        if [p for p in present if p != OUT_LOG]:
            pre.append(f'{out_dir} already holds files {present} (only {OUT_LOG} may exist)')
    script_clean = L132._committed_clean(SCRIPT_REL)
    if not trial and not script_clean:
        pre.append(f'{SCRIPT_REL} not committed clean (production run)')
    input_rels = {'FROZEN_JSON': FZ_REL, 'FROZEN_MD': FZ_MD, 'T2_CSV': T2_CSV, 'README': README,
                  'PARAGRAPHS_V5': PARAGRAPHS_V5, 'MAP': MAP_REL, 'W145_JSON': W145_JSON, 'W145_MANIFEST': W145_MAN,
                  'W130_JSON': W130_JSON, 'W130_MANIFEST': W130_MAN, 'HARNESS_V6': HARNESS_V6,
                  'W145_SCRIPT': 'p515_s53_w145_banded_breakeven_fit.py', 'W157_SCRIPT': 'p515_s53_w157_step6_tables.py',
                  'W132_SCRIPT': 'p515_s53_w132_resettle_v3_campaign.py', 'SC1': 'settling_criterion.py',
                  'SC6': 'settling_criterion_v6.py', 'GRIO': 'gate_result_io.py'}
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': L132._committed_clean(rel),
                       'last_commit': _last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    if inputs['FROZEN_JSON']['sha256'] != FZ_SHA:
        pre.append(f"frozen JSON sha {inputs['FROZEN_JSON']['sha256']} != {FZ_SHA}")
    w145man = _jl(W145_MAN)
    if w145man.get(W145_JSON) != inputs['W145_JSON']['sha256']:
        pre.append(f'{W145_JSON} != its W145 manifest entry ({w145man.get(W145_JSON)})')
    if not (inputs['W130_JSON']['last_commit'] or '').startswith(W130_COMMIT):
        pre.append(f"{W130_JSON} last commit {inputs['W130_JSON']['last_commit']} != {W130_COMMIT}")
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    fz = _jl(FZ_REL)
    cells = all_cells(fz)
    claims = claims_by_id(fz)
    with open(os.path.join(REPO, MAP_REL), encoding='utf-8') as h:
        map_text = h.read()
    # ---- the panel rule, fixed before any trajectory is read ----------------------------------------------------
    chosen, rule_info = panel_rule(fz)
    _log(f'[{tag}] panel rule: chosen {chosen}; gap-refused rows in T2 order {rule_info["gap_refused_rows_in_order"]}; '
         f'ambiguous {rule_info["ambiguous_rows"]}; T2 order == JSON order {rule_info["t2_order_equals_json_order"]}')
    # ---- the record inputs (resolved through T2 eval keys) ------------------------------------------------------
    rec_inputs = {}
    for nm in ('ref:7aa017f0',) + ((chosen,) if chosen else ()):
        good, _hits = find_eval_dir(cells[nm]['eval_key'])
        for ed in good:
            root = os.path.dirname(os.path.dirname(ed))
            for f in sorted(os.listdir(os.path.join(REPO, ed))):
                if f in ('per_cycle_record.jsonl', 'settling_decision.json', 'settling_continuation_cycle_record.jsonl',
                         'resettle_cycle_record.jsonl', 'resettle_decision.json', 'launch.json'):
                    rec_inputs[f'{nm}:{f}'] = os.path.join(ed, f)
            rec_inputs[f'{nm}:campaign_manifest'] = os.path.join(root, 'campaign_manifest_sha256.json')
            if nm == 'ref:7aa017f0':
                rec_inputs[f'{nm}:w101_summary'] = os.path.join(os.path.dirname(root), 'w101_three_reference_summary.json')
    # campaign specs read for the multipliers (figure 2)
    for cid in ('CHECK:headline_V_minus_I_settled', 'H:m1.5:value_minus_I', 'H:m2:value_minus_I'):
        c = claims[cid][1]
        for side in ('ref_cell', 'other_cell'):
            good, _h = find_eval_dir(cells[c[side]]['eval_key'])
            for ed in good:
                rec_inputs[f'{cid}:{side}:spec'] = campaign_spec_of(ed)[0]
    a64row = [r for r in fz['tables']['a64']['rows'] if r['claim_id'] == 'H:m1.75:value_minus_I'][0]
    for side in ('ref_cell', 'other_cell'):
        good, _h = find_eval_dir(a64row['cells'][a64row[side]]['eval_key'])
        for ed in good:
            rec_inputs[f'H:m1.75:{side}:spec'] = campaign_spec_of(ed)[0]
    for key, rel in rec_inputs.items():
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': L132._committed_clean(rel),
                       'last_commit': _last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
        if rel.endswith('.pkl'):
            pre.append(f'{rel} is a pickle')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    man_in = {v['path']: v['sha256'] for v in inputs.values()}
    with open(os.path.join(od, OUT_MAN_IN), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man_in, indent=1, sort_keys=True) + '\n')
    _log(f'[{tag}] {len(inputs)} inputs committed clean (script committed clean {script_clean}; trial {trial}); frozen '
         f'JSON {FZ_SHA[:8]} verified; input manifest written ({len(man_in)} files)')

    # ---- data + checks per figure -----------------------------------------------------------------------------
    w130 = _jl(W130_JSON)
    w130man = _jl(W130_MAN)
    built = {}
    errors = {}
    for fid, fn in (('fig1', lambda: data_fig1(fz, cells, claims)), ('fig2', lambda: data_fig2(fz, cells, claims)),
                    ('fig3', lambda: data_fig3(fz, cells, claims, map_text)),
                    ('fig4', lambda: data_fig4(fz, cells, chosen, rule_info)),
                    ('fig5', lambda: data_fig5(fz, w130, w130man))):
        try:
            built[fid] = fn()
        except Exception as exc:  # a data fault stops that figure; recorded
            errors[fid] = f'{type(exc).__name__}: {exc}'
            _log(f'[{tag}] {fid} DATA FAULT: {errors[fid]}')
    be = fz['tables']['break_even']['committed_W145']['breakeven_range']
    bm = fz['tables']['benchmark']
    extra = {'fig1': {'tau': f"{fz['constants']['TAU']:,.2f}", 'w145_range': f'{fmt_kmwh(be[0])}-{fmt_kmwh(be[1])}'},
             'fig5': {'multiple': f"{bm['multiple']:,.0f}", 'cband': fmt_k(bm['coordinated_reproducibility_band']),
                      'tw': f"{bm['w160_additions']['price_taker_best_start_TN_curtailment_TWh_day_weighted']:.2f}",
                      'sp': bm['sweep']['sweep_passive_cold']['n_blocks'],
                      'st': bm['sweep']['sweep_price_taker_cold']['n_blocks'],
                      'rf': bm['reverse_flow_interface_hours']}}
    draws = {'fig1': draw_fig1, 'fig2': draw_fig2, 'fig3': draw_fig3, 'fig4': draw_fig4, 'fig5': draw_fig5}
    font = font_record()
    results, written, captions = {}, [], []
    for fid, name in FIGS:
        r = {'name': name}
        results[fid] = r
        if fid in errors:
            r['status'] = 'NOT DRAWN (data fault)'
            r['error'] = errors[fid]
            continue
        fd, D = built[fid]
        failed = fd.failed()
        r['checks_failed'] = failed
        r['n_checks'] = len(fd.d['checks'])
        if failed or D is None:
            r['status'] = 'NOT DRAWN (check failed)'
            _log(f'[{tag}] {fid} NOT DRAWN: failed checks {failed}')
            data_rel = os.path.join(out_dir, f'{fid}_{name}_data.json')
            with open(os.path.join(REPO, data_rel), 'x', encoding='utf-8') as h:
                h.write(GRIO.dumps(_norm(fd.d), indent=1, sort_keys=True) + '\n')
            written.append(data_rel)
            continue
        pdf1, png1 = render(draws[fid], fd, D)
        pdf2, _png2 = render(draws[fid], fd, D)
        s1, s2 = _sha_bytes(pdf1), _sha_bytes(pdf2)
        fr = pdf_font_report(pdf1)
        r.update({'pdf_sha256_run1': s1, 'pdf_sha256_run2': s2, 'deterministic': s1 == s2, 'pdf_bytes': len(pdf1),
                  'fonts': fr, 'png_sha256': _sha_bytes(png1)})
        font_ok = fr['n_FontFile2'] > 0 and fr['n_FontFile3'] == 0 and fr['type3'] == 0 and fr['n_FontFile'] == 0
        nodate = not fr['has_creation_date'] and not fr['has_mod_date']
        r['fonts_truetype_embedded'] = font_ok
        r['no_dates'] = nodate
        if not (s1 == s2 and font_ok and nodate):
            r['status'] = 'NOT WRITTEN (determinism / font / date check failed)'
            _log(f'[{tag}] {fid} NOT WRITTEN: deterministic {s1 == s2}, fonts {font_ok}, no dates {nodate}')
            continue
        fd.d['pdf'] = {'sha256': s1, 'determinism': 'generated twice in this run; sha256 equal',
                       'metadata': {k: v for k, v in PDF_METADATA.items()}, 'fonts': fr, 'font_file': font}
        fd.d['preview_png'] = {'dpi': PREVIEW_DPI, 'sha256': r['png_sha256'], 'note': 'preview for review, not a deliverable'}
        for suffix, blob in (('.pdf', pdf1), ('_preview.png', png1)):
            rel = os.path.join(out_dir, f'{fid}_{name}{suffix}')
            with open(os.path.join(REPO, rel), 'xb') as h:
                h.write(blob)
            written.append(rel)
        data_rel = os.path.join(out_dir, f'{fid}_{name}_data.json')
        with open(os.path.join(REPO, data_rel), 'x', encoding='utf-8') as h:
            h.write(GRIO.dumps(_norm(fd.d), indent=1, sort_keys=True) + '\n')
        written.append(data_rel)
        r['status'] = 'written'
        captions.append(caption_text(fid, fd, D, extra.get(fid, {})) + f"\n\n*Width:* {fd.d['width']} "
                        f"({fd.d['width_mm']:g} mm x {fd.d['height_mm']:g} mm). *File:* `{fid}_{name}.pdf` (sha256 "
                        f"{s1[:8]}); data `{fid}_{name}_data.json`.")
        _log(f'[{tag}] {fid} written: pdf {s1[:12]} (run 2 {s2[:12]}), {len(fd.d["checks"])} checks pass, fonts '
             f'{fr["base_fonts"]}')
    cap_rel = os.path.join(out_dir, OUT_CAPTIONS)
    with open(os.path.join(REPO, cap_rel), 'x', encoding='utf-8') as h:
        h.write('# Step 6 figures 1-5 -- draft captions (W165)\n\n'
                f'Generated by `{SCRIPT_REL}` from `frozen_step6_tables_v1_590088fe.json` (sha256 {FZ_SHA}) and the '
                'named records. Draft text for the author; numbers rounded per export/README.md (k EUR 1 dp; EUR/MWh '
                'to the nearest 10, written in k EUR/MWh). PNG files are previews for review, not deliverables.\n\n'
                + '\n\n---\n\n'.join(captions) + '\n')
    written.append(cap_rel)
    guards, pk, gok = guards_state()
    all_ok = gok and all(r.get('status') == 'written' for r in results.values())
    build = {'stage': 'P5.15 Step 6 support, W165 -- manuscript figures 1-5 (STEP6_REVISION_MAP.md section D)',
             'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                        'last_commit': _last_commit(SCRIPT_REL)},
             'trial': trial, 'git_head': _git('rev-parse', 'HEAD'), 'utc': _utc(), 'wall_s': time.time() - t0,
             'interpreter': sys.executable, 'versions': {'python': sys.version.split()[0],
                                                         'matplotlib': matplotlib.__version__,
                                                         'numpy': np.__version__},
             'font': font, 'objective_convention': OBJECTIVE_CAPTION, 'frozen_json': {'path': FZ_REL, 'sha256': FZ_SHA},
             'panel_rule': {'chosen': chosen, **rule_info}, 'figures': results, 'data_faults': errors,
             'guards': guards, 'pickle_guard': pk, 'guards_ok': gok, 'solves': 0 if gok else None,
             'inputs': inputs, 'outputs': written, 'all_ok': all_ok}
    build_rel = os.path.join(out_dir, OUT_BUILD)
    with open(os.path.join(REPO, build_rel), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(_norm(build), indent=1, sort_keys=True) + '\n')
    written.append(build_rel)
    man = {rel: _sha(rel) for rel in sorted(written + [os.path.join(out_dir, OUT_MAN_IN)])}
    with open(os.path.join(od, OUT_MAN), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[{tag}] guards ok {gok} (every guard verified at 0 solves; pickle {pk["counts"]}); figures '
         f'{ {k: v["status"] for k, v in results.items()} }; manifest {len(man)} entries; wall {time.time() - t0:.1f} s')
    if not gok:
        sys.exit(1)
    sys.exit(0 if all_ok else 3)


if __name__ == '__main__':
    main()
